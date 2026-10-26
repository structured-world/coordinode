//! BlobStore lifetime: how an owner comes to reference content, how it lets
//! go, how a reader is kept safe, and how a representation is retired.
//!
//! Every path that puts, replaces, deletes or reads a blob goes through
//! [`BlobStore`], so one protocol holds for all of them:
//!
//! 1. **Admission.** An upload is staged before any payload is written: a
//!    staging record names every distinct content it holds and the
//!    generation it uses, and each content's upload pin is raised in the
//!    same commit. Sharing an eligible representation is decided under the
//!    catalog entry's version, so a retirement cannot land between the
//!    decision and the pin.
//! 2. **Payload.** New payload is written under a generation no other copy of
//!    the content has used, and made durable before anything names it.
//! 3. **Publication.** One commit writes the owner record, makes new
//!    representations eligible, moves the reference counts by the owner's
//!    old and new contents, records a release candidate for each content the
//!    owner let go, removes the staging record and lowers its pins. It is
//!    conditioned on the staging record and the owner record as they were
//!    read, so a fenced upload cannot publish and two writers of one owner
//!    cannot both count from one state.
//! 4. **Release.** Deleting an owner lowers its counts and records candidates
//!    in the commit that removes it. A candidate is evidence for a collector,
//!    never permission to erase.
//! 5. **Retirement.** A representation is retired by a commit conditioned on
//!    its catalog entry and claiming that no owner references the content
//!    and no upload holds it; the claim is decided against committed state.
//!    New references then need a new representation under a new generation.
//! 6. **Reading.** A reader pins one snapshot and resolves the owner, the
//!    catalog and every chunk at it, so payload it resolved stays readable
//!    until the reader is dropped, however slowly it is consumed.
//!
//! Interim blob payload path: payload is written into the store's own tree
//! and made durable on this member before it is published, a single-member
//! guarantee. The storage substrate's extent/stream write, with payload
//! journaled on the members its write concern names, replaces that step.

use std::sync::Arc;

use coordinode_core::graph::blob::{
    BlobRef, CatalogEntry, Codec, ContentId, OwnerRecord, Representation, ScopedContent,
    SharingScope, StagedContent, StagingRecord, candidate_key, catalog_key, create_blob,
    payload_key, pins_counter_key, refs_counter_key, staging_key,
};
use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope};
use coordinode_core::txn::operation::{IntentDigest, OperationId};
use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::WriteConcern;

use crate::engine::coordinator::SnapshotPin;
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::engine::transaction::{CommitContext, CommitError, Transaction};
use crate::error::{StorageError, printable_key};

/// How long a commit that keeps losing races on its own records is run
/// again from a fresh read before the race is reported. Time rather than a
/// count: a refusal by a reservation comes back at once, while the commit
/// holding it may wait up to [`OVERLAP_WAIT`], so a count would be spent in
/// microseconds.
const RETRY_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

/// The first pause before a lost race is run again, doubled up to
/// [`MAX_BACKOFF`] so retries wait for the winner instead of spinning.
const FIRST_BACKOFF: std::time::Duration = std::time::Duration::from_millis(1);

/// The longest pause between two runs of a commit that lost a race.
const MAX_BACKOFF: std::time::Duration = std::time::Duration::from_millis(64);

/// How long a blob commit waits for the commits already admitted on its
/// records to land before it is refused by them. They are single small
/// commits, so this covers a slow fsync rather than a stuck writer. Waiting
/// instead of retrying at once is what lets uploads and a retirement of the
/// same content take turns rather than refuse each other until one gives up.
const OVERLAP_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

/// Why a blob operation did not complete.
#[derive(Debug, thiserror::Error)]
pub enum BlobError {
    /// Reading or writing the store failed.
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// A commit was refused for a reason no retry here can change.
    #[error(transparent)]
    Commit(#[from] CommitError),
    /// A blob record does not decode. Nothing is concluded from it: a
    /// damaged count or reference is not zero ownership.
    #[error("blob record {key} is damaged: {message}")]
    Damaged {
        /// The record.
        key: String,
        /// What is wrong with it.
        message: String,
    },
    /// The content has no eligible representation to read.
    #[error("content {content} has no eligible representation")]
    Unavailable {
        /// The content.
        content: String,
    },
    /// The catalog names a representation whose payload is not there.
    #[error("payload of {content} generation {generation} is missing")]
    MissingPayload {
        /// The content.
        content: String,
        /// The generation.
        generation: u64,
    },
    /// The payload read is not the content it is stored as.
    #[error("payload of {content} generation {generation} is not that content")]
    IntegrityMismatch {
        /// The content.
        content: String,
        /// The generation.
        generation: u64,
    },
    /// The upload's staging record is gone, so it no longer holds its
    /// contents and cannot publish them.
    #[error("the upload was fenced before it published")]
    Fenced,
    /// An object of no bytes has nothing to store.
    #[error("an empty object has nothing to store")]
    Empty,
    /// The owner a creation names already holds an object.
    #[error("the owner already holds an object")]
    OwnerExists,
    /// Other commits kept winning the race for the same records.
    #[error("blob records kept changing under this operation; nothing was applied: {last}")]
    Contended {
        /// The refusal of the last attempt.
        last: String,
    },
}

/// A result of a blob operation.
pub type BlobResult<T> = Result<T, BlobError>;

/// What a put did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PutOutcome {
    /// The owner's record as published.
    pub record: OwnerRecord,
    /// Distinct contents whose payload was already held and shared.
    pub shared: usize,
}

/// An upload admitted and written, not yet published. It holds its contents
/// until it is published ([`BlobStore::publish_into`]) or abandoned
/// ([`BlobStore::abandon`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedUpload {
    staging_id: u64,
    staged: StagingRecord,
    blob: BlobRef,
}

impl StagedUpload {
    /// The id its staging record is kept under.
    pub fn staging_id(&self) -> u64 {
        self.staging_id
    }

    /// The object it uploads.
    pub fn blob(&self) -> &BlobRef {
        &self.blob
    }

    /// What it holds and how.
    pub fn staged(&self) -> &StagingRecord {
        &self.staged
    }
}

/// What a put may do to the object an owner already holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnerWrite {
    /// The owner must hold nothing: a new owner is created.
    Create,
    /// Whatever the owner holds is released in the same commit.
    Replace,
}

/// What a retirement attempt decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retirement {
    /// The representation is retired: no new reference resolves to it, and
    /// its payload waits for cleanup by exact generation.
    Retired,
    /// An owner still references the content or an upload holds it.
    Held,
    /// The catalog no longer publishes that generation as eligible.
    NotEligible,
}

/// The BlobStore of one store: the engine holding the payload and records,
/// and the path its commits take.
#[derive(Clone)]
pub struct BlobStore {
    engine: Arc<StorageEngine>,
    oracle: Arc<TimestampOracle>,
    pipeline: Option<Arc<dyn ProposalPipeline>>,
    id_gen: Option<Arc<ProposalIdGenerator>>,
    now_micros: fn() -> i64,
}

impl std::fmt::Debug for BlobStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlobStore")
            .field("pipeline", &self.pipeline.is_some())
            .finish_non_exhaustive()
    }
}

impl BlobStore {
    /// A BlobStore committing straight to `engine`, for a store with no log
    /// in front of it.
    pub fn direct(engine: Arc<StorageEngine>, oracle: Arc<TimestampOracle>) -> Self {
        Self {
            engine,
            oracle,
            pipeline: None,
            id_gen: None,
            now_micros: wall_clock_micros,
        }
    }

    /// A BlobStore committing through `pipeline`, as every other write of
    /// the store does.
    pub fn through_pipeline(
        engine: Arc<StorageEngine>,
        oracle: Arc<TimestampOracle>,
        pipeline: Arc<dyn ProposalPipeline>,
        id_gen: Arc<ProposalIdGenerator>,
    ) -> Self {
        Self {
            engine,
            oracle,
            pipeline: Some(pipeline),
            id_gen: Some(id_gen),
            now_micros: wall_clock_micros,
        }
    }

    /// Use `now_micros` as the clock admission and release times are taken
    /// from.
    #[must_use]
    pub fn with_clock(mut self, now_micros: fn() -> i64) -> Self {
        self.now_micros = now_micros;
        self
    }

    /// The engine the store lives in.
    pub fn engine(&self) -> &Arc<StorageEngine> {
        &self.engine
    }

    /// Store `data` as the object `owner_key` holds, in `scope`, as `write`
    /// allows. With `operation`, the publication takes effect at most once
    /// however often it is retried.
    ///
    /// # Errors
    ///
    /// [`BlobError::Empty`] for no bytes, [`BlobError::OwnerExists`] for a
    /// creation at a held owner; a storage, commit or decoding failure
    /// otherwise. A failure leaves the owner as it was.
    pub fn put(
        &self,
        owner_key: &[u8],
        scope: SharingScope,
        data: &[u8],
        write: OwnerWrite,
        operation: Option<(OperationId, IntentDigest)>,
    ) -> BlobResult<PutOutcome> {
        let upload = self.upload(scope, data)?;
        self.publish(owner_key, write, &upload, operation)
            .inspect_err(|_| self.give_up(&upload))
    }

    /// Admit `data` in `scope` and write its payload durably, for a
    /// publication that follows ([`Self::publish_into`]). The upload holds its
    /// contents until it is published or abandoned.
    ///
    /// # Errors
    ///
    /// [`BlobError::Empty`] for no bytes; a storage, commit or decoding
    /// failure otherwise, after which nothing stays held.
    pub fn upload(&self, scope: SharingScope, data: &[u8]) -> BlobResult<StagedUpload> {
        if data.is_empty() {
            return Err(BlobError::Empty);
        }
        let (blob, chunks) = create_blob(scope, data);
        let staging_id = self.oracle.next().as_raw();
        let staged = self.stage(staging_id, &blob)?;
        let upload = StagedUpload {
            staging_id,
            staged,
            blob,
        };
        self.write_payload(&upload.staged, &chunks)
            .inspect_err(|_| self.give_up(&upload))?;
        Ok(upload)
    }

    /// Abandon `upload` after a failure, leaving its record to the collector
    /// when even that fails.
    fn give_up(&self, upload: &StagedUpload) {
        // The upload holds its contents until its record is gone; giving
        // them up now spares the collector.
        if let Err(abandon) = self.abandon(upload.staging_id) {
            tracing::warn!(
                staging_id = upload.staging_id,
                error = %abandon,
                "an upload that did not publish keeps its staging record"
            );
        }
    }

    /// Admit an upload of `blob`: record it as staged under `staging_id` and
    /// pin each of its distinct contents, sharing the eligible
    /// representation of each content that has one.
    fn stage(&self, staging_id: u64, blob: &BlobRef) -> BlobResult<StagingRecord> {
        let distinct = blob.distinct();
        self.retrying(|store| {
            let mut txn = store.begin();
            let mut contents = Vec::with_capacity(distinct.len());
            for content in &distinct {
                let key = catalog_key(content);
                let (entry, version) = store.catalog_entry(&key)?;
                txn.expect_version(Partition::BlobRef, &key, version)?;
                let staged = match entry.and_then(|e| e.eligible) {
                    Some(eligible) => StagedContent {
                        content: *content,
                        generation: eligible.generation,
                        writes: false,
                    },
                    None => StagedContent {
                        content: *content,
                        generation: store.oracle.next().as_raw(),
                        writes: true,
                    },
                };
                txn.push_counter_delta(&pins_counter_key(content), 1);
                contents.push(staged);
            }
            let record = StagingRecord {
                admitted_at_micros: (store.now_micros)(),
                contents,
            };
            txn.expect_version(Partition::BlobRef, &staging_key(staging_id), None)?;
            txn.put(
                Partition::BlobRef,
                &staging_key(staging_id),
                &encode(&record, "staging record")?,
            )?;
            store.commit(txn)?;
            Ok(record)
        })
    }

    /// Write the payload of every content `staged` writes itself, and make
    /// it durable before it can be published.
    fn write_payload(
        &self,
        staged: &StagingRecord,
        chunks: &[(ContentId, &[u8])],
    ) -> BlobResult<()> {
        let by_content: rustc_hash::FxHashMap<ContentId, &[u8]> =
            chunks.iter().map(|(id, data)| (*id, *data)).collect();
        let mut wrote = false;
        for item in staged.contents.iter().filter(|c| c.writes) {
            // Staged from these very chunks, so each content has its bytes.
            let Some(data) = by_content.get(&item.content.content) else {
                return Err(BlobError::Unavailable {
                    content: item.content.content.to_string(),
                });
            };
            self.engine.put(
                Partition::Blob,
                &payload_key(&item.content, item.generation),
                data,
            )?;
            wrote = true;
        }
        if wrote {
            // Interim blob payload path: the substrate's durable sealing
            // replaces this flush.
            self.engine.persist_partition(Partition::Blob)?;
        }
        Ok(())
    }

    /// Publish `upload` as what `owner_key` holds, in a commit of its own.
    fn publish(
        &self,
        owner_key: &[u8],
        write: OwnerWrite,
        upload: &StagedUpload,
        operation: Option<(OperationId, IntentDigest)>,
    ) -> BlobResult<PutOutcome> {
        self.retrying(|store| {
            let mut txn = store.begin();
            if let Some((op, intent)) = operation {
                txn.bind_operation(op, intent);
            }
            let outcome = store.publish_into(&mut txn, owner_key, write, upload)?;
            store.commit(txn)?;
            Ok(outcome)
        })
    }

    /// Stage, in `txn`, the publication of `upload` as what `owner_key`
    /// holds, as `write` allows: the owner record, the catalog entries of its
    /// new representations, the reference counts moved by the owner's old
    /// and new contents, candidates for what it lets go, and the end of the
    /// upload's staging. Everything is conditioned on the records as read
    /// here, so the commit of `txn` applies all of it or none, together with
    /// whatever else `txn` writes, such as the node the property belongs to.
    ///
    /// # Errors
    ///
    /// [`BlobError::Fenced`] when the upload was abandoned,
    /// [`BlobError::OwnerExists`] for a creation at a held owner; a storage or
    /// decoding failure otherwise.
    pub fn publish_into(
        &self,
        txn: &mut Transaction<'_>,
        owner_key: &[u8],
        write: OwnerWrite,
        upload: &StagedUpload,
    ) -> BlobResult<PutOutcome> {
        let record = OwnerRecord {
            incarnation: upload.staging_id,
            blob: upload.blob.clone(),
        };
        let stage_key = staging_key(upload.staging_id);
        let Some(stage_version) = self.engine.record_version(Partition::BlobRef, &stage_key)?
        else {
            return Err(BlobError::Fenced);
        };
        txn.expect_version(Partition::BlobRef, &stage_key, Some(stage_version))?;
        txn.delete(Partition::BlobRef, &stage_key)?;

        let mut shared = 0;
        for item in &upload.staged.contents {
            txn.push_counter_delta(&pins_counter_key(&item.content), -1);
            let key = catalog_key(&item.content);
            let (entry, version) = self.catalog_entry(&key)?;
            txn.expect_version(Partition::BlobRef, &key, version)?;
            let mut entry = entry.unwrap_or_default();
            match entry.eligible {
                Some(eligible) if eligible.generation == item.generation => {
                    shared += 1;
                }
                // Another upload published the content first: readers
                // resolve to its copy, and this one's payload waits for
                // cleanup by its exact generation.
                Some(_) => {
                    if item.writes {
                        entry.retired.push(item.generation);
                        txn.put(Partition::BlobRef, &key, &encode(&entry, "catalog entry")?)?;
                    } else {
                        shared += 1;
                    }
                }
                None if item.writes => {
                    entry.eligible = Some(Representation {
                        generation: item.generation,
                        codec: Codec::Raw,
                        encoded_len: item.content.content.len(),
                    });
                    txn.put(Partition::BlobRef, &key, &encode(&entry, "catalog entry")?)?;
                }
                // A shared representation cannot be retired while this
                // upload pins its content.
                None => {
                    return Err(BlobError::Unavailable {
                        content: item.content.content.to_string(),
                    });
                }
            }
        }

        let (old, owner_version) = self.owner(owner_key)?;
        if write == OwnerWrite::Create && old.is_some() {
            return Err(BlobError::OwnerExists);
        }
        txn.expect_version(Partition::BlobRef, owner_key, owner_version)?;
        self.release_into(txn, owner_key, old.as_ref(), Some(&record.blob))?;
        self.acquire_into(txn, old.as_ref(), &record.blob);
        txn.put(
            Partition::BlobRef,
            owner_key,
            &encode(&record, "owner record")?,
        )?;
        Ok(PutOutcome { record, shared })
    }

    /// Give up the upload staged as `staging_id`: remove its record, lower
    /// its pins, and leave the payload it wrote to cleanup by exact
    /// generation. Doing nothing when the record is already gone makes this
    /// safe to repeat, and it is also how an expired upload is fenced.
    ///
    /// # Errors
    ///
    /// A storage, commit or decoding failure; the record then stays.
    pub fn abandon(&self, staging_id: u64) -> BlobResult<bool> {
        self.retrying(|store| {
            let stage_key = staging_key(staging_id);
            let Some((bytes, version)) =
                store.engine.get_versioned(Partition::BlobRef, &stage_key)?
            else {
                return Ok(false);
            };
            let staged: StagingRecord = decode(&stage_key, &bytes, StagingRecord::from_msgpack)?;
            let mut txn = store.begin();
            txn.expect_version(Partition::BlobRef, &stage_key, Some(version))?;
            txn.delete(Partition::BlobRef, &stage_key)?;
            for item in &staged.contents {
                txn.push_counter_delta(&pins_counter_key(&item.content), -1);
                if !item.writes {
                    continue;
                }
                let key = catalog_key(&item.content);
                let (entry, version) = store.catalog_entry(&key)?;
                let mut entry = entry.unwrap_or_default();
                if entry.eligible.map(|e| e.generation) == Some(item.generation)
                    || entry.retired.contains(&item.generation)
                {
                    continue;
                }
                txn.expect_version(Partition::BlobRef, &key, version)?;
                entry.retired.push(item.generation);
                txn.put(Partition::BlobRef, &key, &encode(&entry, "catalog entry")?)?;
            }
            store.commit(txn)?;
            Ok(true)
        })
    }

    /// Delete what `owner_key` holds. Returns the record it held, or `None`
    /// when it held nothing.
    ///
    /// # Errors
    ///
    /// A storage, commit or decoding failure; the owner then stays.
    pub fn delete(&self, owner_key: &[u8]) -> BlobResult<Option<OwnerRecord>> {
        self.retrying(|store| {
            let mut txn = store.begin();
            let Some(old) = store.delete_into(&mut txn, owner_key)? else {
                return Ok(None);
            };
            store.commit(txn)?;
            Ok(Some(old))
        })
    }

    /// Stage, in `txn`, the deletion of what `owner_key` holds: the owner
    /// record goes, its references are released and a candidate is recorded
    /// for each, conditioned on the record as read here. Returns the record,
    /// or `None` when the owner holds nothing and nothing is staged.
    ///
    /// # Errors
    ///
    /// A storage failure, or a record that does not decode.
    pub fn delete_into(
        &self,
        txn: &mut Transaction<'_>,
        owner_key: &[u8],
    ) -> BlobResult<Option<OwnerRecord>> {
        let (old, version) = self.owner(owner_key)?;
        let Some(old) = old else {
            return Ok(None);
        };
        txn.expect_version(Partition::BlobRef, owner_key, version)?;
        txn.delete(Partition::BlobRef, owner_key)?;
        self.release_into(txn, owner_key, Some(&old), None)?;
        Ok(Some(old))
    }

    /// What `owner_key` holds now, if anything.
    ///
    /// # Errors
    ///
    /// A storage failure, or a record that does not decode.
    pub fn record(&self, owner_key: &[u8]) -> BlobResult<Option<OwnerRecord>> {
        Ok(self.owner(owner_key)?.0)
    }

    /// Open the object `owner_key` holds for reading, or `None` when it holds
    /// nothing. Every chunk resolves at one pinned snapshot, held until the
    /// reader is dropped.
    ///
    /// # Errors
    ///
    /// A storage failure, a damaged record, or a content with no eligible
    /// representation: an object that cannot be read whole is refused
    /// before any of it is read.
    pub fn open(&self, owner_key: &[u8]) -> BlobResult<Option<BlobReader>> {
        let (snapshot, pin) = self.engine.pin_latest_snapshot();
        self.open_pinned(owner_key, snapshot, pin)
    }

    /// [`Self::open`] at `snapshot`, the view a transaction reads at, so the
    /// object is the one the transaction's other reads see.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::open`], and a snapshot no longer retained.
    pub fn open_at(
        &self,
        owner_key: &[u8],
        snapshot: lsm_tree::SeqNo,
    ) -> BlobResult<Option<BlobReader>> {
        let pin = self.engine.pin_snapshot_at(snapshot).ok_or_else(|| {
            StorageError::SnapshotOutsideRetention {
                snapshot,
                watermark: self.engine.oldest_readable_seqno(),
            }
        })?;
        self.open_pinned(owner_key, snapshot, Some(pin))
    }

    fn open_pinned(
        &self,
        owner_key: &[u8],
        snapshot: lsm_tree::SeqNo,
        pin: Option<SnapshotPin>,
    ) -> BlobResult<Option<BlobReader>> {
        let Some(bytes) = self
            .engine
            .snapshot_get(&snapshot, Partition::BlobRef, owner_key)?
        else {
            return Ok(None);
        };
        let record = decode(owner_key, &bytes, OwnerRecord::from_msgpack)?;
        let mut resolved: rustc_hash::FxHashMap<ContentId, u64> = rustc_hash::FxHashMap::default();
        for content in record.blob.distinct() {
            let key = catalog_key(&content);
            let eligible = match self
                .engine
                .snapshot_get(&snapshot, Partition::BlobRef, &key)?
            {
                Some(bytes) => decode(&key, &bytes, CatalogEntry::from_msgpack)?.eligible,
                None => None,
            };
            let Some(eligible) = eligible else {
                return Err(BlobError::Unavailable {
                    content: content.content.to_string(),
                });
            };
            resolved.insert(content.content, eligible.generation);
        }
        let generations = record
            .blob
            .chunks
            .iter()
            .map(|content| {
                resolved
                    .get(content)
                    .copied()
                    .ok_or_else(|| BlobError::Unavailable {
                        content: content.to_string(),
                    })
            })
            .collect::<BlobResult<Vec<u64>>>()?;
        Ok(Some(BlobReader {
            engine: Arc::clone(&self.engine),
            snapshot,
            _pin: pin,
            record,
            generations,
        }))
    }

    /// Retire `content`'s eligible representation `generation`, if no owner
    /// references the content and no upload holds it.
    ///
    /// # Errors
    ///
    /// A storage, commit or decoding failure; nothing is retired then.
    pub fn retire(&self, content: &ScopedContent, generation: u64) -> BlobResult<Retirement> {
        let key = catalog_key(content);
        self.retrying(|store| {
            let (entry, version) = store.catalog_entry(&key)?;
            let Some(mut entry) = entry else {
                return Ok(Retirement::NotEligible);
            };
            if entry.eligible.map(|e| e.generation) != Some(generation) {
                return Ok(Retirement::NotEligible);
            }
            // Every admission of the content is conditioned on the entry this
            // rewrites, so under steady uploads the retirement would lose to
            // whichever came first, every time. Its commit waits for the
            // admissions already on their way, whose references it then
            // counts, and later ones queue behind it.
            let mut txn = store.begin();
            txn.expect_version(Partition::BlobRef, &key, version)?;
            let revision = txn.schema_generation();
            txn.claim(Claim::new(
                ClaimScope::Record(key.clone()),
                ClaimPredicate::BlobRetired { generation },
                revision,
            ));
            entry.eligible = None;
            entry.retired.push(generation);
            txn.put(Partition::BlobRef, &key, &encode(&entry, "catalog entry")?)?;
            match store.commit(txn) {
                Ok(()) => Ok(Retirement::Retired),
                Err(BlobError::Commit(CommitError::InvariantRefused { .. })) => {
                    Ok(Retirement::Held)
                }
                Err(e) => Err(e),
            }
        })
    }

    /// Stage, in `txn`, the release of every content `old` references that
    /// `new` does not: one reference fewer each, and a candidate naming the
    /// release.
    fn release_into(
        &self,
        txn: &mut Transaction<'_>,
        owner_key: &[u8],
        old: Option<&OwnerRecord>,
        new: Option<&BlobRef>,
    ) -> BlobResult<()> {
        let Some(old) = old else {
            return Ok(());
        };
        let kept = new.map(BlobRef::distinct).unwrap_or_default();
        let now = (self.now_micros)();
        for content in old.blob.distinct() {
            if kept.binary_search(&content).is_ok() {
                continue;
            }
            txn.push_counter_delta(&refs_counter_key(&content), -1);
            txn.put(
                Partition::BlobRef,
                &candidate_key(&content, owner_key, old.incarnation),
                &now.to_be_bytes(),
            )?;
        }
        Ok(())
    }

    /// Stage, in `txn`, one reference more for every content `new`
    /// references that `old` did not.
    fn acquire_into(&self, txn: &mut Transaction<'_>, old: Option<&OwnerRecord>, new: &BlobRef) {
        let held = old.map(|o| o.blob.distinct()).unwrap_or_default();
        for content in new.distinct() {
            if held.binary_search(&content).is_err() {
                txn.push_counter_delta(&refs_counter_key(&content), 1);
            }
        }
    }

    /// The owner record at `owner_key` and its version, as committed now.
    fn owner(&self, owner_key: &[u8]) -> BlobResult<(Option<OwnerRecord>, Option<u64>)> {
        match self.engine.get_versioned(Partition::BlobRef, owner_key)? {
            Some((bytes, version)) => Ok((
                Some(decode(owner_key, &bytes, OwnerRecord::from_msgpack)?),
                Some(version),
            )),
            None => Ok((None, None)),
        }
    }

    /// The catalog entry at `key` and its version, as committed now.
    fn catalog_entry(&self, key: &[u8]) -> BlobResult<(Option<CatalogEntry>, Option<u64>)> {
        match self.engine.get_versioned(Partition::BlobRef, key)? {
            Some((bytes, version)) => Ok((
                Some(decode(key, &bytes, CatalogEntry::from_msgpack)?),
                Some(version),
            )),
            None => Ok((None, None)),
        }
    }

    fn begin(&self) -> Transaction<'_> {
        let mut txn = Transaction::begin(&self.engine, Some(&self.oracle), self.oracle.next());
        txn.wait_for_overlapping_commits(OVERLAP_WAIT);
        txn
    }

    /// Commit `txn` durably: a blob operation answered is one a crash keeps.
    fn commit(&self, mut txn: Transaction<'_>) -> BlobResult<()> {
        let concern = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &concern,
            pipeline: self.pipeline.as_deref(),
            id_gen: self.id_gen.as_deref(),
        };
        txn.commit(&ctx)?;
        Ok(())
    }

    /// Run `attempt` until it is not refused by a race on its own records,
    /// reading afresh each time.
    fn retrying<T>(&self, mut attempt: impl FnMut(&Self) -> BlobResult<T>) -> BlobResult<T> {
        let deadline = std::time::Instant::now() + RETRY_BUDGET;
        let mut backoff = FIRST_BACKOFF;
        loop {
            match attempt(self) {
                Err(BlobError::Commit(
                    e @ (CommitError::Conflict(_) | CommitError::RevisionMismatch { .. }),
                )) => {
                    if std::time::Instant::now() + backoff >= deadline {
                        return Err(BlobError::Contended {
                            last: e.to_string(),
                        });
                    }
                    std::thread::sleep(backoff);
                    backoff = (backoff * 2).min(MAX_BACKOFF);
                }
                other => return other,
            }
        }
    }
}

/// An object open for reading at one pinned snapshot.
pub struct BlobReader {
    engine: Arc<StorageEngine>,
    snapshot: lsm_tree::SeqNo,
    _pin: Option<SnapshotPin>,
    record: OwnerRecord,
    /// The generation each chunk resolved to, by chunk index.
    generations: Vec<u64>,
}

impl BlobReader {
    /// What the owner holds.
    pub fn record(&self) -> &OwnerRecord {
        &self.record
    }

    /// The bytes of chunk `index`, verified to be its content.
    ///
    /// # Errors
    ///
    /// The payload is missing or is not the content, or the read failed.
    ///
    /// # Panics
    ///
    /// When `index` is not below the object's chunk count.
    pub fn chunk(&self, index: usize) -> BlobResult<Vec<u8>> {
        let content = self.record.blob.chunks[index];
        let generation = self.generations[index];
        let scoped = ScopedContent::new(self.record.blob.scope, content);
        let data = self
            .engine
            .snapshot_get(
                &self.snapshot,
                Partition::Blob,
                &payload_key(&scoped, generation),
            )?
            .ok_or_else(|| BlobError::MissingPayload {
                content: content.to_string(),
                generation,
            })?;
        if !content.matches(&data) {
            return Err(BlobError::IntegrityMismatch {
                content: content.to_string(),
                generation,
            });
        }
        Ok(data.to_vec())
    }
}

fn encode<T>(value: &T, what: &'static str) -> BlobResult<Vec<u8>>
where
    T: serde::Serialize,
{
    rmp_serde::to_vec(value).map_err(|e| BlobError::Damaged {
        key: what.to_string(),
        message: e.to_string(),
    })
}

fn decode<T, E: std::fmt::Display>(
    key: &[u8],
    bytes: &[u8],
    parse: impl FnOnce(&[u8]) -> Result<T, E>,
) -> BlobResult<T> {
    parse(bytes).map_err(|e| BlobError::Damaged {
        key: printable_key(key),
        message: e.to_string(),
    })
}

/// Microseconds since the epoch by the system clock. A clock set before the
/// epoch reads as the epoch: these times order grace, never identity.
fn wall_clock_micros() -> i64 {
    // no-std: caller-provided clock.
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    // Microseconds since 1970 fit an i64 until the year 294247.
    since.as_micros() as i64
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
