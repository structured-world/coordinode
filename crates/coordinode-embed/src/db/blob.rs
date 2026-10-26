//! Direct blobs: objects a client stores, reads and deletes by id, outside
//! the graph.
//!
//! Each upload is its own owner, so two clients storing the same bytes hold
//! two blobs and one deleting its blob leaves the other's. The bytes are
//! shared underneath all the same. An upload that names a client operation
//! takes effect at most once however often it is retried, under an id the
//! operation decides, so a retry whose answer was lost never leaves a second
//! blob nobody knows the id of.

use coordinode_core::graph::blob::{OwnerRecord, SharingScope, direct_owner_key};
use coordinode_core::txn::operation::IntentDigest;
use coordinode_storage::blob::{BlobError, BlobReader, BlobStore, OwnerWrite};
use sha2::{Digest, Sha256};

use super::{Database, DatabaseError};
use crate::operation::OperationDescriptor;

/// The id of a direct blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlobId([u8; 16]);

impl BlobId {
    /// An id from its bytes.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// The id's bytes.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// The id an upload carrying out `descriptor` gets: the same for every
    /// attempt of the operation, distinct across operations. A client whose
    /// retry is refused as settled finds its blob under it.
    pub fn for_operation(descriptor: &OperationDescriptor) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"coordinode.blob.owner.v1");
        hasher.update(descriptor.deployment.as_bytes());
        hasher.update(descriptor.op.issuer.0);
        hasher.update(descriptor.op.incarnation.to_be_bytes());
        hasher.update(descriptor.op.sequence.to_be_bytes());
        let digest = hasher.finalize();
        let mut id = [0u8; 16];
        id.copy_from_slice(&digest[..16]);
        Self(id)
    }

    /// A fresh id: `unique` makes it distinct from every other id this store
    /// issues, the random half keeps it from being guessed.
    fn fresh(unique: u64) -> Self {
        use std::hash::{BuildHasher, Hasher};
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u64(unique);
        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&unique.to_be_bytes());
        id[8..].copy_from_slice(&hasher.finish().to_be_bytes());
        Self(id)
    }

    fn owner_key(&self) -> Vec<u8> {
        direct_owner_key(&self.0)
    }
}

impl std::fmt::Display for BlobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl std::str::FromStr for BlobId {
    type Err = DatabaseError;

    /// Parse the 32 lowercase or uppercase hex digits [`BlobId`] displays as.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid = || DatabaseError::Semantic(format!("{text:?} is not a blob id"));
        if text.len() != 32 {
            return Err(invalid());
        }
        let mut id = [0u8; 16];
        for (byte, pair) in id.iter_mut().zip(text.as_bytes().chunks(2)) {
            let pair = std::str::from_utf8(pair).map_err(|_| invalid())?;
            *byte = u8::from_str_radix(pair, 16).map_err(|_| invalid())?;
        }
        Ok(Self(id))
    }
}

/// A stored upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadedBlob {
    /// The blob's id.
    pub id: BlobId,
    /// What the blob holds.
    pub record: OwnerRecord,
    /// Distinct chunks whose bytes the store already held and shares.
    pub shared: usize,
}

impl Database {
    /// The BlobStore of this database, committing through its write
    /// pipeline like every other write.
    pub fn blob_store(&self) -> BlobStore {
        BlobStore::through_pipeline(
            std::sync::Arc::clone(&self.engine),
            std::sync::Arc::clone(&self.oracle),
            std::sync::Arc::clone(&self.pipeline),
            std::sync::Arc::clone(&self.proposal_id_gen),
        )
    }

    /// Let `members` say how many members the group this database belongs
    /// to holds now. Called by the member that opens it.
    pub fn set_group_members(&mut self, members: std::sync::Arc<dyn Fn() -> usize + Send + Sync>) {
        self.group_members = Some(members);
    }

    /// Store `data` as a new blob. With `operation`, the upload takes effect
    /// at most once however often it is retried, and every attempt names the
    /// same blob.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Unsupported`] while the group holds other members,
    /// which the payload would not reach;
    /// [`DatabaseError::OperationSettled`] for a retry of an operation that
    /// stands already; a storage or commit failure otherwise.
    pub fn upload_blob(
        &self,
        data: &[u8],
        operation: Option<&OperationDescriptor>,
    ) -> Result<UploadedBlob, DatabaseError> {
        self.refuse_blob_writes_in_a_group()?;
        if data.is_empty() {
            return Err(DatabaseError::Semantic(
                "an empty blob has nothing to store".into(),
            ));
        }
        let store = self.blob_store();
        let (id, bound) = match operation {
            Some(descriptor) => {
                let intent = upload_intent(data);
                self.admit_operation(descriptor, &intent)?;
                (
                    BlobId::for_operation(descriptor),
                    Some((descriptor.op, intent)),
                )
            }
            None => (BlobId::fresh(self.oracle.next().as_raw()), None),
        };
        let outcome = store
            .put(
                &id.owner_key(),
                SharingScope::STORE,
                data,
                OwnerWrite::Create,
                bound,
            )
            .map_err(blob_error)?;
        Ok(UploadedBlob {
            id,
            record: outcome.record,
            shared: outcome.shared,
        })
    }

    /// Open the blob `id` names for reading, or `None` when there is none.
    /// Its chunks stay readable until the reader is dropped.
    ///
    /// # Errors
    ///
    /// A storage failure, or a blob whose bytes are missing or damaged.
    pub fn open_blob(&self, id: &BlobId) -> Result<Option<BlobReader>, DatabaseError> {
        self.blob_store().open(&id.owner_key()).map_err(blob_error)
    }

    /// What the blob `id` names holds, or `None` when there is none.
    ///
    /// # Errors
    ///
    /// A storage failure, or a damaged record.
    pub fn blob_record(&self, id: &BlobId) -> Result<Option<OwnerRecord>, DatabaseError> {
        self.blob_store()
            .record(&id.owner_key())
            .map_err(blob_error)
    }

    /// Delete the blob `id` names. Returns what it held, or `None` when
    /// there was none. Bytes another blob shares stay.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::Unsupported`] while the group holds other members; a
    /// storage or commit failure otherwise.
    pub fn delete_blob(&self, id: &BlobId) -> Result<Option<OwnerRecord>, DatabaseError> {
        self.refuse_blob_writes_in_a_group()?;
        self.blob_store()
            .delete(&id.owner_key())
            .map_err(blob_error)
    }

    /// Refuse a blob write while the group holds other members: the payload
    /// is held only on the member that writes it, and a leader change would
    /// lose it.
    fn refuse_blob_writes_in_a_group(&self) -> Result<(), DatabaseError> {
        let members = match &self.group_members {
            Some(members) => members(),
            // A replicated database whose member did not say how many it
            // has: as unknown as a group of many.
            None if self.cluster_mode => usize::MAX,
            None => 1,
        };
        if members <= 1 {
            return Ok(());
        }
        Err(DatabaseError::Unsupported(
            "blob payload is held only on the member that writes it, and this group holds \
             other members"
                .into(),
        ))
    }
}

/// The intent of an upload of `data`: two attempts of one operation upload
/// the same bytes.
fn upload_intent(data: &[u8]) -> IntentDigest {
    let mut hasher = Sha256::new();
    hasher.update(b"coordinode.blob.upload.v1");
    hasher.update((data.len() as u64).to_be_bytes());
    hasher.update(data);
    IntentDigest(hasher.finalize().into())
}

/// The database error a blob operation's failure is, so a caller reading a
/// blob's chunks reports their failures as any other.
impl From<BlobError> for DatabaseError {
    fn from(e: BlobError) -> Self {
        blob_error(e)
    }
}

fn blob_error(e: BlobError) -> DatabaseError {
    match e {
        BlobError::Storage(storage) => storage.into(),
        BlobError::Commit(commit) => {
            coordinode_query::executor::runner::commit_err_to_execution(commit).into()
        }
        BlobError::Fenced | BlobError::Contended { .. } => DatabaseError::TransactionConflict {
            transaction: None,
            message: e.to_string(),
        },
        BlobError::Empty => DatabaseError::Semantic(e.to_string()),
        BlobError::Damaged { .. }
        | BlobError::Unavailable { .. }
        | BlobError::MissingPayload { .. }
        | BlobError::IntegrityMismatch { .. }
        | BlobError::OwnerExists => DatabaseError::Other(e.to_string()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
