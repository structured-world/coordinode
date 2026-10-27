//! Blob store: large property values as owners of shared blob content.
//!
//! A property value of at least [`INLINE_THRESHOLD`] bytes is held as a blob:
//! the (node, property) pair is an owner whose record lists the value's
//! chunks, and the chunks' bytes are shared with every other owner of the
//! same content. The protocol every blob path follows lives in
//! [`coordinode_storage::blob`]; this store adapts it to properties.
//!
//! ## Data plane vs metadata plane
//!
//! Blobs keep bulk **data** out of the consensus log and transactional
//! write buffers, while small **metadata** is transactional:
//!
//! - **Upload (data plane):** [`BlobStore::upload`] admits the value,
//!   pinning its contents, and writes their payload durably, in commits of
//!   its own and before the property write. Payload is large (multi-MB) and
//!   never sits in a transaction's in-memory buffer.
//! - **Publication (metadata plane):** [`BlobStore::put_blob`] stages, in the
//!   active [`Transaction`], the property's owner record together with the
//!   reference counts it moves and the end of the upload, so they commit
//!   atomically with the node write that produced them, or not at all.
//!   [`BlobStore::delete_blob_ref`] releases the property in the same way.
//!   Reads see the property through the transaction's MVCC snapshot, and the
//!   chunks stay readable at that snapshot for as long as the reader lives.
//!
//! An upload whose property write never commits is abandoned
//! ([`BlobStore::abandon`]) or, if its writer is gone, fenced and collected:
//! its pins and payload are never leaked into a reference nobody holds.
//!
//! A release committed through the caller's transaction reaches the orphan
//! collector at its next pass; the committer can wake it at once with
//! [`coordinode_storage::blob::BlobStore::notify_collector`].
//!
//! [`Transaction`]: coordinode_storage::engine::transaction::Transaction

use coordinode_core::graph::blob::{BlobRef, OwnerRecord, SharingScope, encode_blobref_key};
use coordinode_core::graph::node::NodeId;
use coordinode_storage::blob::{BlobReader, OwnerWrite, StagedUpload};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use crate::error::{StoreError, StoreResult};

pub use coordinode_core::graph::blob::INLINE_THRESHOLD;

/// Layer 4 blob store for node properties: uploads ahead of the property
/// write, then publication, release and reads through the active
/// transaction. Hides partition layout and key encoders.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a blob store for node properties",
    label = "`{Self}` does not implement `BlobStore`",
    note = "`LocalBlobStore::new` wraps the store's `coordinode_storage::blob::BlobStore`"
)]
pub trait BlobStore {
    /// Admit `data` and write its payload durably, ahead of the property
    /// write that publishes it with [`Self::put_blob`].
    fn upload(&self, data: &[u8]) -> StoreResult<StagedUpload>;

    /// Stage, in `txn`, `upload` as the value of property `prop_id` of
    /// `node_id`, releasing whatever the property held. Returns the
    /// reference the property now holds.
    fn put_blob(
        &self,
        txn: &mut Transaction,
        node_id: NodeId,
        prop_id: u32,
        upload: &StagedUpload,
    ) -> StoreResult<BlobRef>;

    /// The [`BlobRef`] property `prop_id` of `node_id` holds, through the
    /// transaction's MVCC snapshot.
    fn get_blob_ref(
        &self,
        txn: &Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<Option<BlobRef>>;

    /// Open the blob property `prop_id` of `node_id` holds for reading, at
    /// the transaction's snapshot.
    fn open_blob(
        &self,
        txn: &Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<Option<BlobReader>>;

    /// Stage, in `txn`, the removal of the blob property `prop_id` of
    /// `node_id`, releasing its references. Bytes other owners share stay.
    fn delete_blob_ref(
        &self,
        txn: &mut Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<()>;

    /// Give up `upload` when the property write that would publish it does
    /// not commit.
    fn abandon(&self, upload: &StagedUpload) -> StoreResult<()>;
}

/// CE implementation of [`BlobStore`] over one store's
/// [`coordinode_storage::blob::BlobStore`].
pub struct LocalBlobStore {
    store: coordinode_storage::blob::BlobStore,
}

impl LocalBlobStore {
    /// A property blob store over `store`.
    pub fn new(store: coordinode_storage::blob::BlobStore) -> Self {
        Self { store }
    }
}

impl BlobStore for LocalBlobStore {
    fn upload(&self, data: &[u8]) -> StoreResult<StagedUpload> {
        Ok(self.store.upload(SharingScope::STORE, data)?)
    }

    fn put_blob(
        &self,
        txn: &mut Transaction,
        node_id: NodeId,
        prop_id: u32,
        upload: &StagedUpload,
    ) -> StoreResult<BlobRef> {
        let outcome = self.store.publish_into(
            txn,
            &encode_blobref_key(node_id, prop_id),
            OwnerWrite::Replace,
            upload,
        )?;
        Ok(outcome.record.blob)
    }

    fn get_blob_ref(
        &self,
        txn: &Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<Option<BlobRef>> {
        let Some(bytes) =
            txn.read_untracked(Partition::BlobRef, &encode_blobref_key(node_id, prop_id))?
        else {
            return Ok(None);
        };
        OwnerRecord::from_msgpack(&bytes)
            .map(|record| Some(record.blob))
            .map_err(|e| StoreError::Decode {
                kind: "blob owner record",
                message: format!("decode for node {}: {e}", node_id.as_raw()),
            })
    }

    fn open_blob(
        &self,
        txn: &Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<Option<BlobReader>> {
        let key = encode_blobref_key(node_id, prop_id);
        let reader = match txn.snapshot() {
            Some(snapshot) => self.store.open_at(&key, snapshot)?,
            None => self.store.open(&key)?,
        };
        Ok(reader)
    }

    fn delete_blob_ref(
        &self,
        txn: &mut Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<()> {
        self.store
            .delete_into(txn, &encode_blobref_key(node_id, prop_id))?;
        Ok(())
    }

    fn abandon(&self, upload: &StagedUpload) -> StoreResult<()> {
        self.store.abandon(upload.staging_id())?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
