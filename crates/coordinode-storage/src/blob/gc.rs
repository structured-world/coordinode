//! Orphan collection: turning released references into returned space.
//!
//! A delete only releases references and records a candidate. The collector
//! takes it from there, in bounded passes that one store runs one at a time:
//!
//! 1. **Expired uploads.** An upload staged longer than the upload expiry is
//!    fenced: its record goes, its pins drop, and the payload it wrote is
//!    retired, so a delayed uploader can no longer publish it.
//! 2. **Candidates.** Candidates are read oldest first, up to the first one
//!    still in its grace. A content no owner references, no upload holds and
//!    whose candidate's grace has passed has its eligible representation
//!    retired under the claim that decides retirement against committed
//!    counts. A content an owner holds again only drops its candidate; one an
//!    upload holds keeps it for a later pass.
//! 3. **Cleanup.** The payload of every retired representation is deleted by
//!    exact generation, with the count of what was deleted.
//! 4. **Compaction.** Deleting payload returns no space until a compaction
//!    passes the deletion below the GC watermark, which readers, retained
//!    history and durable consumers hold back. Once a value file's worth of
//!    payload is deleted, or the oldest deletion has waited long enough, and
//!    the watermark has passed the deletions, the Blob partition is compacted
//!    under the shared maintenance budget, and the bytes it shrank by are
//!    counted. Batching keeps the compaction's I/O to whole files that died
//!    rather than one rewrite per deletion.
//! 5. **Reconciliation.** Periodically, and before anything is retired or
//!    deleted for the first time, the owners and uploads are read at one
//!    snapshot and checked against the counts, the catalog, the candidates and
//!    the payload: a count that disagrees is corrected, an unreferenced
//!    content without a candidate gets one, and payload nothing names is
//!    retired. A record that does not decode blocks retirement and cleanup
//!    until a later reconciliation finds none: unknown ownership is never
//!    read as none.
//!
//! The grace is the configured one or the MVCC retention window, whichever
//! is longer, and is a minimum, never a promise.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use coordinode_core::graph::blob::{
    CANDIDATE_PREFIX, CATALOG_PREFIX, COLLECTOR_STATE_KEY, COMPACTION_PENDING_KEY, CatalogEntry,
    DIRECT_OWNER_PREFIX, OwnerRecord, PAYLOAD_PREFIX, PINS_COUNTER_PREFIX, PROPERTY_OWNER_PREFIX,
    RECLAIMED_BYTES_KEY, RECLAIMED_CHUNKS_KEY, RECLAIMED_DISK_BYTES_KEY, REFS_COUNTER_PREFIX,
    RETIRED_PREFIX, S3_OWNER_PREFIX, STAGING_PREFIX, ScopedContent, StagingRecord,
    UNCOMPACTED_BYTES_KEY, candidate_key, candidates_released_after, catalog_key,
    decode_candidate_key, decode_counter_key, decode_payload_key, decode_retired_key,
    decode_staging_key, pins_counter_key, refs_counter_key, retired_key,
};
use coordinode_core::txn::wake::Wake;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::{BlobError, BlobResult, BlobStore, CleanedUp, Retirement, decode, encode};
use crate::engine::footprint::{MaintenanceScope, ReclaimScope};
use crate::engine::partition::Partition;
use crate::error::{StorageError, printable_key};

/// Records written in one commit by a cleanup or a reconciliation.
const PER_COMMIT: usize = 64;

/// The owner a reconciliation names in a candidate it records for an
/// unreferenced content that had none.
const RECONCILIATION_OWNER: &[u8] = b"blobgc:reconciliation";

/// Damaged keys a blocker names before it only counts the rest.
const NAMED_DAMAGE: usize = 8;

/// The slices a reconciliation can take: one per first digest byte.
const DIGEST_SLICES: usize = 256;

/// How the collector runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcPolicy {
    /// The least time a released content is kept before it is collected.
    /// The MVCC retention window raises it when longer.
    pub grace: Duration,
    /// How long an upload may stay staged before it is fenced. An upload is
    /// staged between its admission and its publication, which take
    /// milliseconds unless the process failed in between.
    pub upload_expiry: Duration,
    /// The most items one pass takes from each queue: due candidates,
    /// expired uploads and retired representations.
    pub batch: usize,
    /// How often accounting is reconciled against the owners.
    pub reconcile_every: Duration,
    /// About how many distinct contents one reconciliation holds in memory
    /// at a time: a store with more is reconciled in slices, each rereading
    /// the owners.
    pub reconcile_max_contents: usize,
    /// How soon a pass is retried while work is pending but held: by an
    /// upload, by a reader or retained history, or by a failure.
    pub retry_interval: Duration,
    /// Payload bytes deleted since the last compaction at which the next one
    /// runs. A compaction rewrites the partition's index and the value files
    /// it chooses to relocate, so it waits until a value file's worth of
    /// payload is gone: a file whose every value was deleted is then dropped
    /// whole rather than rewritten.
    pub compact_after_bytes: u64,
    /// The longest deleted payload waits for its compaction when fewer
    /// bytes than [`Self::compact_after_bytes`] were deleted.
    pub compact_after: Duration,
}

impl Default for GcPolicy {
    fn default() -> Self {
        Self {
            grace: Duration::from_secs(24 * 60 * 60),
            upload_expiry: Duration::from_secs(60 * 60),
            batch: 1024,
            reconcile_every: Duration::from_secs(24 * 60 * 60),
            reconcile_max_contents: 1_000_000,
            retry_interval: Duration::from_secs(60),
            // The value files' target size.
            compact_after_bytes: 64 << 20,
            compact_after: Duration::from_secs(10 * 60),
        }
    }
}

/// The shortest time between two passes of the background worker: wakeups
/// in between, one per deletion under steady churn, are answered by one
/// pass.
const MIN_PASS_GAP: Duration = Duration::from_secs(1);

/// What one pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GcPass {
    /// When the pass ran, in microseconds since the epoch.
    pub at_micros: i64,
    /// The reconciliation the pass ran, if one was due.
    pub reconciliation: Option<Reconciliation>,
    /// Uploads fenced for staying staged past the upload expiry.
    pub expired_uploads: u64,
    /// Candidates whose grace had passed and that the pass examined.
    pub candidates_examined: u64,
    /// Representations retired.
    pub retired: u64,
    /// Contents kept because an upload holds them.
    pub protected: u64,
    /// What the cleanup deleted.
    pub cleaned: CleanedUpReport,
    /// The compaction the pass ran, if one was due.
    pub compaction: Option<CompactionReport>,
    /// Why some work could not go on.
    pub blockers: Vec<String>,
}

/// [`CleanedUp`], in a pass report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CleanedUpReport {
    /// Retired representations whose cleanup is done.
    pub representations: u64,
    /// Payload chunks deleted.
    pub chunks: u64,
    /// Payload bytes deleted.
    pub bytes: u64,
}

impl CleanedUpReport {
    fn add(&mut self, cleaned: CleanedUp) {
        self.representations += cleaned.representations;
        self.chunks += cleaned.chunks;
        self.bytes += cleaned.bytes;
    }
}

/// A compaction the collector ran to return space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CompactionReport {
    /// The Blob partition's size on disk before.
    pub disk_bytes_before: u64,
    /// Its size after.
    pub disk_bytes_after: u64,
}

/// What a reconciliation found and corrected.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Reconciliation {
    /// Owner records read.
    pub owners: u64,
    /// Uploads staged.
    pub staged_uploads: u64,
    /// The slices the contents were taken in.
    pub slices: u64,
    /// Reference counts that disagreed with the owners and were corrected.
    pub corrected_refs: u64,
    /// Pin counts that disagreed with the staged uploads and were corrected.
    pub corrected_pins: u64,
    /// Unreferenced contents that had no candidate and were given one.
    pub recorded_candidates: u64,
    /// Payload no catalog entry, upload or retirement names, retired.
    pub retired_strays: u64,
    /// Records that do not decode. Any blocks retirement and cleanup, and
    /// nothing is corrected.
    pub damaged: Vec<String>,
}

/// Where the collector stands, read from the store when asked.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GcStatus {
    /// Whether this member may collect: a member of a group that holds
    /// other members does not.
    pub active: bool,
    /// When accounting was last reconciled clean, in microseconds since the
    /// epoch; nothing is retired or deleted before the first time.
    pub reconciled_at_micros: Option<i64>,
    /// Why retirement and cleanup are stopped, if they are.
    pub blocked: Option<String>,
    /// The grace in force, the configured one or the retention window.
    pub grace_secs: u64,
    /// Candidates recorded.
    pub candidates: u64,
    /// Candidates whose grace has passed.
    pub candidates_due: u64,
    /// When the oldest candidate was released.
    pub oldest_candidate_micros: Option<i64>,
    /// When the next candidate in its grace becomes due.
    pub next_due_micros: Option<i64>,
    /// Uploads staged.
    pub staged_uploads: u64,
    /// When the oldest staged upload was admitted.
    pub oldest_upload_micros: Option<i64>,
    /// Retired representations awaiting cleanup.
    pub retired_pending: u64,
    /// Their payload bytes, as written.
    pub retired_pending_bytes: u64,
    /// When the oldest of them was retired.
    pub oldest_retired_micros: Option<i64>,
    /// Whether payload was deleted that a compaction has not yet returned.
    pub compaction_pending: bool,
    /// Payload bytes deleted since the last compaction.
    pub uncompacted_bytes: u64,
    /// When the oldest of those deletions ran.
    pub oldest_uncompacted_micros: Option<i64>,
    /// Payload chunks deleted, ever.
    pub reclaimed_chunks: u64,
    /// Payload bytes deleted, ever.
    pub reclaimed_bytes: u64,
    /// Bytes the Blob partition shrank by in the collector's compactions,
    /// ever. A hard-linked checkpoint may still hold the files.
    pub reclaimed_disk_bytes: u64,
    /// The Blob partition's size on disk.
    pub blob_disk_bytes: u64,
    /// Bytes in its value files no live key points to that a compaction has
    /// yet to return.
    pub stale_blob_bytes: u64,
    /// The last pass.
    pub last_pass: Option<GcPass>,
    /// The last pass's failure, if it failed.
    pub last_error: Option<String>,
}

/// What the collector keeps in the store between passes and restarts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct CollectorRecord {
    reconciled_at_micros: Option<i64>,
    blocked: Option<String>,
}

#[derive(Default)]
struct RunState {
    last_pass: Option<GcPass>,
    last_error: Option<String>,
    /// When the next pass has something to do, as far as the last pass saw.
    next_due_micros: Option<i64>,
}

/// Whether this member may collect now.
pub type Activity = Arc<dyn Fn() -> bool + Send + Sync>;

/// The orphan collector of one store.
pub struct BlobCollector {
    store: BlobStore,
    policy: Mutex<GcPolicy>,
    active: Activity,
    /// Serializes passes: one store runs one at a time.
    run: Mutex<RunState>,
    /// Wakes the background worker when an owner releases content, an upload
    /// is abandoned, or the policy changes.
    wake: Arc<Wake>,
}

impl std::fmt::Debug for BlobCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlobCollector")
            .field("policy", &*self.policy.lock())
            .finish_non_exhaustive()
    }
}

impl BlobCollector {
    /// The collector of `store`'s blobs, running under `policy`, with
    /// `active` deciding whether this member may collect now.
    pub fn new(store: BlobStore, policy: GcPolicy, active: Activity) -> Arc<Self> {
        let wake = Arc::new(Wake::default());
        Arc::new(Self {
            store: store.waking(Arc::clone(&wake)),
            policy: Mutex::new(policy),
            active,
            run: Mutex::new(RunState::default()),
            wake,
        })
    }

    /// The store, with its releases waking this collector: what every write
    /// path of the store commits through.
    pub fn store(&self) -> &BlobStore {
        &self.store
    }

    /// The policy in force.
    pub fn policy(&self) -> GcPolicy {
        self.policy.lock().clone()
    }

    /// Run under `policy` from the next pass on.
    pub fn set_policy(&self, policy: GcPolicy) {
        *self.policy.lock() = policy;
        self.wake.notify();
    }

    /// Run one pass now, on the caller's thread, after any pass already
    /// running. `reconcile` forces a reconciliation whether or not one is
    /// due. A manual pass is the automatic one: it cannot shorten the grace,
    /// fence a live upload early, or pass a reader or retained history.
    ///
    /// # Errors
    ///
    /// A storage, commit or decoding failure; work done before it stands,
    /// and the next pass resumes where this one stopped.
    pub fn run(&self, reconcile: bool) -> BlobResult<GcPass> {
        let mut state = self.run.lock();
        let outcome = self.pass(reconcile);
        match &outcome {
            Ok((pass, next)) => {
                state.last_pass = Some(pass.clone());
                state.last_error = None;
                state.next_due_micros = *next;
            }
            Err(e) => {
                state.last_error = Some(e.to_string());
                state.next_due_micros = None;
            }
        }
        outcome.map(|(pass, _)| pass)
    }

    /// Where the collector stands. Reads every queue it keeps, so its cost
    /// grows with the backlog.
    ///
    /// # Errors
    ///
    /// A storage failure, or a record that does not decode.
    pub fn status(&self) -> BlobResult<GcStatus> {
        use lsm_tree::AbstractTree as _;

        let engine = self.store.engine();
        let now = self.store.now();
        let grace = self.grace_micros();
        let record = self.record()?;
        let mut status = GcStatus {
            active: (self.active)(),
            reconciled_at_micros: record.reconciled_at_micros,
            blocked: record.blocked,
            grace_secs: u64::try_from(grace / 1_000_000).unwrap_or(0),
            ..GcStatus::default()
        };
        let due = now.checked_sub(grace);
        for guard in engine.prefix_scan(Partition::BlobRef, CANDIDATE_PREFIX)? {
            let key = guard.key().map_err(StorageError::from)?;
            let Some(candidate) = decode_candidate_key(&key) else {
                continue;
            };
            status.candidates += 1;
            status
                .oldest_candidate_micros
                .get_or_insert(candidate.released_at_micros);
            if due.is_some_and(|due| candidate.released_at_micros <= due) {
                status.candidates_due += 1;
            } else if status.next_due_micros.is_none() {
                status.next_due_micros = candidate.released_at_micros.checked_add(grace);
            }
        }
        for guard in engine.prefix_scan(Partition::BlobRef, STAGING_PREFIX)? {
            let (key, value) = guard.into_inner().map_err(StorageError::from)?;
            let staged = decode(&key, &value, StagingRecord::from_msgpack)?;
            status.staged_uploads += 1;
            status.oldest_upload_micros = Some(
                status
                    .oldest_upload_micros
                    .map_or(staged.admitted_at_micros, |t| {
                        t.min(staged.admitted_at_micros)
                    }),
            );
        }
        for guard in engine.prefix_scan(Partition::BlobRef, RETIRED_PREFIX)? {
            let (key, value) = guard.into_inner().map_err(StorageError::from)?;
            let Some((content, _)) = decode_retired_key(&key) else {
                continue;
            };
            status.retired_pending += 1;
            status.retired_pending_bytes += u64::from(content.content.len());
            let retired_at = stored_time(&key, &value)?;
            status.oldest_retired_micros = Some(
                status
                    .oldest_retired_micros
                    .map_or(retired_at, |t| t.min(retired_at)),
            );
        }
        if let Some(oldest) = engine.get(Partition::BlobRef, COMPACTION_PENDING_KEY)? {
            status.compaction_pending = true;
            status.oldest_uncompacted_micros = Some(stored_time(COMPACTION_PENDING_KEY, &oldest)?);
        }
        status.uncompacted_bytes = self.measure(UNCOMPACTED_BYTES_KEY)?;
        status.reclaimed_chunks = self.measure(RECLAIMED_CHUNKS_KEY)?;
        status.reclaimed_bytes = self.measure(RECLAIMED_BYTES_KEY)?;
        status.reclaimed_disk_bytes = self.measure(RECLAIMED_DISK_BYTES_KEY)?;
        let tree = engine.tree(Partition::Blob)?;
        status.blob_disk_bytes = tree.disk_space();
        status.stale_blob_bytes = tree.stale_blob_bytes();
        let state = self.run.lock();
        status.last_pass = state.last_pass.clone();
        status.last_error = state.last_error.clone();
        Ok(status)
    }

    /// Start the background worker. It runs a pass when a release or an
    /// abandoned upload wakes it and when the earliest work it knows of
    /// falls due, and sleeps otherwise; dropping the worker stops it.
    pub fn spawn(self: &Arc<Self>) -> CollectorWorker {
        let stop = Arc::new(AtomicBool::new(false));
        let collector = Arc::clone(self);
        let stopping = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("blob-collector".into())
            .spawn(move || collector.work(&stopping))
            .map_err(|e| tracing::error!(error = %e, "could not start the blob collector"))
            .ok();
        CollectorWorker {
            stop,
            wake: Arc::clone(&self.wake),
            handle,
        }
    }

    fn work(&self, stop: &AtomicBool) {
        self.wake.bind();
        while !stop.load(Ordering::Acquire) {
            if (self.active)() {
                if let Err(e) = self.run(false) {
                    tracing::warn!(error = %e, "blob collection pass failed");
                }
            }
            let ran = std::time::Instant::now();
            if stop.load(Ordering::Acquire) {
                break;
            }
            self.wake.wait(Some(self.sleep()));
            // However often it is woken, the next pass waits out the gap:
            // what arrived meanwhile is collected by that one pass.
            loop {
                let since = ran.elapsed();
                if since >= MIN_PASS_GAP || stop.load(Ordering::Acquire) {
                    break;
                }
                self.wake.wait(Some(MIN_PASS_GAP - since));
            }
        }
    }

    /// How long the worker sleeps before its next pass, unless woken.
    fn sleep(&self) -> Duration {
        let policy = self.policy();
        let state = self.run.lock();
        if state.last_error.is_some() {
            return policy.retry_interval;
        }
        // Nothing known to fall due: sleep until the next reconciliation;
        // a release or an abandoned upload wakes the worker sooner.
        let Some(due) = state.next_due_micros else {
            return policy.reconcile_every;
        };
        // A deadline already passed still waits a moment, so a pass that
        // could not make progress does not spin.
        let wait = due
            .checked_sub(self.store.now())
            .and_then(|wait| u64::try_from(wait).ok())
            .unwrap_or(0);
        Duration::from_micros(wait).max(Duration::from_millis(10))
    }

    /// One pass, and when the next one has something to do.
    fn pass(&self, force_reconcile: bool) -> BlobResult<(GcPass, Option<i64>)> {
        let now = self.store.now();
        let mut pass = GcPass {
            at_micros: now,
            ..GcPass::default()
        };
        if !(self.active)() {
            pass.blockers.push(
                "blob payload is held only on the member that writes it, and this group \
                 holds other members"
                    .into(),
            );
            return Ok((pass, None));
        }
        // A store that never held a blob has nothing to collect and no reason
        // to write: a database that stores no blobs commits nothing for them,
        // and the worker sleeps until its first release wakes it.
        if !self.holds_blob_records()? {
            return Ok((pass, None));
        }
        // Collection is how the room a full disk refuses client writes for
        // comes back, so its commits are admitted as the store's own work, and
        // its compaction keeps to the room flushes were promised.
        let _reclaiming = ReclaimScope::enter();
        let _maintenance = MaintenanceScope::enter();
        let policy = self.policy();
        let mut next = Deadline::default();

        let mut record = self.record()?;
        let reconcile_due = record
            .reconciled_at_micros
            .and_then(|at| at.checked_add(duration_micros(policy.reconcile_every)));
        if force_reconcile || reconcile_due.is_none_or(|due| due <= now) {
            pass.reconciliation = Some(self.reconcile(&policy, now)?);
            record = self.record()?;
        }
        if let Some(at) = record.reconciled_at_micros {
            next.at(at.checked_add(duration_micros(policy.reconcile_every)));
        }
        let destructive = record.reconciled_at_micros.is_some() && record.blocked.is_none();
        if let Some(blocked) = &record.blocked {
            pass.blockers.push(blocked.clone());
        } else if record.reconciled_at_micros.is_none() {
            pass.blockers
                .push("accounting has not been reconciled yet".into());
        }

        self.expire_uploads(&policy, now, &mut pass, &mut next)?;
        self.collect_candidates(&policy, now, destructive, &mut pass, &mut next)?;
        if destructive {
            self.clean_up(&policy, &mut pass, &mut next)?;
        }
        self.compact(&policy, &mut pass, &mut next)?;
        Ok((pass, next.0))
    }

    /// Fence uploads staged longer than the upload expiry.
    fn expire_uploads(
        &self,
        policy: &GcPolicy,
        now: i64,
        pass: &mut GcPass,
        next: &mut Deadline,
    ) -> BlobResult<()> {
        let expiry = duration_micros(policy.upload_expiry);
        let mut expired = Vec::new();
        for guard in self
            .store
            .engine()
            .prefix_scan(Partition::BlobRef, STAGING_PREFIX)?
        {
            let (key, value) = guard.into_inner().map_err(StorageError::from)?;
            let Some(staging_id) = decode_staging_key(&key) else {
                continue;
            };
            let staged = decode(&key, &value, StagingRecord::from_msgpack)?;
            match staged.admitted_at_micros.checked_add(expiry) {
                Some(at) if at > now => next.at(Some(at)),
                // An expiry past the representable future never comes.
                None => {}
                Some(_) if expired.len() < policy.batch => expired.push(staging_id),
                // More expired than one pass takes: the next pass is due now.
                Some(_) => next.at(Some(now)),
            }
        }
        for staging_id in expired {
            if self.store.abandon(staging_id)? {
                pass.expired_uploads += 1;
            }
        }
        Ok(())
    }

    /// Examine candidates whose grace has passed, oldest first.
    fn collect_candidates(
        &self,
        policy: &GcPolicy,
        now: i64,
        destructive: bool,
        pass: &mut GcPass,
        next: &mut Deadline,
    ) -> BlobResult<()> {
        let grace = self.grace_micros();
        let retry = later(now, policy.retry_interval);
        let mut due: Vec<(ScopedContent, Vec<u8>)> = Vec::new();
        let bound = now.checked_sub(grace).map(candidates_released_after);
        for guard in self
            .store
            .engine()
            .prefix_scan(Partition::BlobRef, CANDIDATE_PREFIX)?
        {
            let key = guard.key().map_err(StorageError::from)?;
            let Some(candidate) = decode_candidate_key(&key) else {
                continue;
            };
            // Candidates sort by release time: the first one still in its
            // grace ends the due ones.
            if bound
                .as_ref()
                .is_none_or(|bound| key.as_ref() >= bound.as_slice())
            {
                next.at(candidate.released_at_micros.checked_add(grace));
                break;
            }
            if due.len() == policy.batch {
                next.at(Some(now));
                break;
            }
            due.push((candidate.content, key.to_vec()));
        }
        pass.candidates_examined = due.len() as u64;
        // One decision per content, however many owners released it.
        let mut by_content: FxHashMap<ScopedContent, Vec<Vec<u8>>> = FxHashMap::default();
        for (content, key) in due {
            by_content.entry(content).or_default().push(key);
        }
        let mut settled: Vec<Vec<u8>> = Vec::new();
        let mut to_retire: Vec<ToRetire> = Vec::new();
        for (content, keys) in by_content {
            let refs = self.count(&refs_counter_key(&content))?;
            let pins = self.count(&pins_counter_key(&content))?;
            if refs < 0 || pins < 0 {
                pass.blockers.push(format!(
                    "content {} counts {refs} references and {pins} uploads; a negative \
                     count is broken accounting, not an absent owner",
                    content.content
                ));
                continue;
            }
            // An owner holds it again: its later release records a
            // candidate of its own.
            if refs > 0 {
                settled.extend(keys);
                continue;
            }
            let Some(entry) = self.catalog_entry(&content)? else {
                settled.extend(keys);
                continue;
            };
            if pins > 0 {
                pass.protected += 1;
                next.at(Some(retry));
                continue;
            }
            if destructive {
                to_retire.push(((content, entry.eligible.generation), keys));
            }
        }
        // Retired together, a commit per batch rather than per content.
        while !to_retire.is_empty() {
            let rest = to_retire.split_off(to_retire.len().min(PER_COMMIT));
            let batch = std::mem::replace(&mut to_retire, rest);
            let items: Vec<(ScopedContent, u64)> = batch.iter().map(|(item, _)| *item).collect();
            let decided = self.store.retire_many(&items)?;
            for ((_, keys), decision) in batch.into_iter().zip(decided) {
                match decision {
                    Retirement::Retired => {
                        pass.retired += 1;
                        settled.extend(keys);
                    }
                    Retirement::NotEligible => settled.extend(keys),
                    // Admitted again between the read and the retirement.
                    Retirement::Held => {
                        pass.protected += 1;
                        next.at(Some(retry));
                    }
                }
            }
        }
        for keys in settled.chunks(PER_COMMIT) {
            self.store.retrying(|store| {
                let mut txn = store.begin();
                for key in keys {
                    txn.delete(Partition::BlobRef, key)?;
                }
                store.commit(txn)
            })?;
        }
        Ok(())
    }

    /// Delete the payload of retired representations.
    fn clean_up(
        &self,
        policy: &GcPolicy,
        pass: &mut GcPass,
        next: &mut Deadline,
    ) -> BlobResult<()> {
        let mut retired = Vec::new();
        for guard in self
            .store
            .engine()
            .prefix_scan(Partition::BlobRef, RETIRED_PREFIX)?
        {
            if retired.len() == policy.batch {
                next.at(Some(self.store.now()));
                break;
            }
            let key = guard.key().map_err(StorageError::from)?;
            if let Some(item) = decode_retired_key(&key) {
                retired.push(item);
            }
        }
        for batch in retired.chunks(PER_COMMIT) {
            pass.cleaned.add(self.store.clean_up(batch)?);
        }
        Ok(())
    }

    /// Compact the Blob partition once enough payload was deleted, or the
    /// oldest deletion waited long enough, and the deletions are below the
    /// GC watermark, so their space comes back.
    fn compact(&self, policy: &GcPolicy, pass: &mut GcPass, next: &mut Deadline) -> BlobResult<()> {
        use lsm_tree::AbstractTree as _;

        let engine = self.store.engine();
        let Some((oldest, deleted_at)) =
            engine.get_versioned(Partition::BlobRef, COMPACTION_PENDING_KEY)?
        else {
            return Ok(());
        };
        let oldest = stored_time(COMPACTION_PENDING_KEY, &oldest)?;
        let uncompacted = self.measure(UNCOMPACTED_BYTES_KEY)?;
        // A time past the representable future never comes: only the bytes
        // can make the compaction due then.
        let due_at = oldest.checked_add(duration_micros(policy.compact_after));
        if uncompacted < policy.compact_after_bytes
            && due_at.is_none_or(|due| due > self.store.now())
        {
            next.at(due_at);
            return Ok(());
        }
        engine.advance_gc_watermark();
        if engine.gc_watermark() <= deleted_at {
            // A reader, retained history or a durable consumer still sees
            // the payload.
            next.at(Some(later(self.store.now(), policy.retry_interval)));
            return Ok(());
        }
        let tree = engine.tree(Partition::Blob)?;
        let before = tree.disk_space();
        // A value file the compaction empties goes with its own install.
        engine.force_compaction(Partition::Blob)?;
        let after = tree.disk_space();
        pass.compaction = Some(CompactionReport {
            disk_bytes_before: before,
            disk_bytes_after: after,
        });
        // Writes landing meanwhile can grow the partition: only a net shrink
        // is space this compaction returned.
        let shrank = before.saturating_sub(after);
        self.store.retrying(|store| {
            let mut txn = store.begin();
            if shrank > 0 {
                // A partition's size fits an i64 by many orders of magnitude.
                txn.push_counter_delta(RECLAIMED_DISK_BYTES_KEY, shrank as i64);
            }
            // Only this collector cleans up, and its passes run one at a
            // time, so the mark and the count are those read above; checked
            // all the same, and a newer mark keeps its compaction to run.
            if store
                .engine()
                .record_version(Partition::BlobRef, COMPACTION_PENDING_KEY)?
                == Some(deleted_at)
            {
                txn.expect_version(Partition::BlobRef, COMPACTION_PENDING_KEY, Some(deleted_at))?;
                txn.delete(Partition::BlobRef, COMPACTION_PENDING_KEY)?;
                // Bounded by the bytes deleted, which fit an i64.
                txn.push_counter_delta(UNCOMPACTED_BYTES_KEY, -(uncompacted as i64));
            }
            store.commit(txn)
        })?;
        Ok(())
    }

    /// Check accounting against the owners and uploads at one snapshot, and
    /// correct what disagrees unless something does not decode.
    ///
    /// The contents are taken in slices by the first byte of their digest,
    /// enough that each holds about [`GcPolicy::reconcile_max_contents`]:
    /// memory follows the slice, and every slice rereads the owners.
    fn reconcile(&self, policy: &GcPolicy, now: i64) -> BlobResult<Reconciliation> {
        let engine = self.store.engine();
        let (snapshot, _pin) = engine.pin_latest_snapshot();
        let mut contents = 0usize;
        for guard in engine.snapshot_prefix_iter(&snapshot, Partition::BlobRef, CATALOG_PREFIX)? {
            guard.key().map_err(StorageError::from)?;
            contents += 1;
        }
        let slices = contents
            .div_ceil(policy.reconcile_max_contents.max(1))
            .clamp(1, DIGEST_SLICES);
        let mut found = Reconciliation {
            slices: slices as u64,
            ..Reconciliation::default()
        };
        let mut fixes = Fixes::default();
        for slice in 0..slices {
            let range = Slice {
                first: slice * DIGEST_SLICES / slices,
                end: (slice + 1) * DIGEST_SLICES / slices,
                reports_records: slice == 0,
            };
            self.reconcile_slice(&snapshot, &range, &mut found, &mut fixes)?;
        }

        let blocked = (!found.damaged.is_empty()).then(|| {
            let named: Vec<&str> = found
                .damaged
                .iter()
                .take(NAMED_DAMAGE)
                .map(String::as_str)
                .collect();
            let tail = match found.damaged.len().checked_sub(NAMED_DAMAGE) {
                Some(more) if more > 0 => format!(" and {more} more"),
                _ => String::new(),
            };
            format!(
                "blob records do not decode, so ownership is unknown: {}{tail}",
                named.join("; ")
            )
        });
        if blocked.is_none() {
            self.apply(&fixes, now, snapshot)?;
        }
        let last_clean = self.record()?.reconciled_at_micros;
        let record = match &blocked {
            // Nothing is corrected from an incomplete picture, and the last
            // clean reconciliation stands.
            Some(blocked) => CollectorRecord {
                reconciled_at_micros: last_clean,
                blocked: Some(blocked.clone()),
            },
            None => CollectorRecord {
                reconciled_at_micros: Some(now),
                blocked: None,
            },
        };
        let encoded = encode(&record, "collector state")?;
        self.store.retrying(|store| {
            let mut txn = store.begin();
            txn.put(Partition::BlobRef, COLLECTOR_STATE_KEY, &encoded)?;
            store.commit(txn)
        })?;
        if blocked.is_some() {
            tracing::error!(
                damaged = found.damaged.len(),
                "blob accounting cannot be verified; retirement and cleanup are stopped"
            );
        }
        Ok(found)
    }

    /// Reconcile the contents of `slice` at `snapshot`.
    fn reconcile_slice(
        &self,
        snapshot: &lsm_tree::SeqNo,
        slice: &Slice,
        found: &mut Reconciliation,
        fixes: &mut Fixes,
    ) -> BlobResult<()> {
        let engine = self.store.engine();
        let damaged = |found: &mut Reconciliation, key: &[u8], what: &str| {
            if slice.reports_records {
                found
                    .damaged
                    .push(format!("{}: {what}", printable_key(key)));
            }
        };

        // Owners: one reference per owner and distinct content.
        let mut refs: FxHashMap<ScopedContent, i64> = FxHashMap::default();
        for prefix in [DIRECT_OWNER_PREFIX, S3_OWNER_PREFIX, PROPERTY_OWNER_PREFIX] {
            for guard in engine.snapshot_prefix_iter(snapshot, Partition::BlobRef, prefix)? {
                let (key, value) = guard.into_inner().map_err(StorageError::from)?;
                if slice.reports_records {
                    found.owners += 1;
                }
                match OwnerRecord::from_msgpack(&value) {
                    Ok(owner) => {
                        for content in owner.blob.distinct() {
                            if slice.holds(&content) {
                                *refs.entry(content).or_default() += 1;
                            }
                        }
                    }
                    Err(e) => damaged(found, &key, &e.to_string()),
                }
            }
        }

        // Uploads: one pin per upload and distinct content, and the exact
        // generations they hold.
        let mut pins: FxHashMap<ScopedContent, i64> = FxHashMap::default();
        let mut staged: FxHashSet<(ScopedContent, u64)> = FxHashSet::default();
        for guard in engine.snapshot_prefix_iter(snapshot, Partition::BlobRef, STAGING_PREFIX)? {
            let (key, value) = guard.into_inner().map_err(StorageError::from)?;
            if slice.reports_records {
                found.staged_uploads += 1;
            }
            match StagingRecord::from_msgpack(&value) {
                Ok(record) => {
                    for item in record.contents {
                        if slice.holds(&item.content) {
                            *pins.entry(item.content).or_default() += 1;
                            staged.insert((item.content, item.generation));
                        }
                    }
                }
                Err(e) => damaged(found, &key, &e.to_string()),
            }
        }

        // Counts as committed at the same snapshot.
        for (prefix, expected, kind) in [
            (REFS_COUNTER_PREFIX, &refs, Count::Refs),
            (PINS_COUNTER_PREFIX, &pins, Count::Pins),
        ] {
            let mut seen: FxHashSet<ScopedContent> = FxHashSet::default();
            for guard in engine.snapshot_prefix_iter(snapshot, Partition::Counter, prefix)? {
                let (key, value) = guard.into_inner().map_err(StorageError::from)?;
                let (Some(content), Ok(actual)) = (
                    decode_counter_key(prefix, &key),
                    crate::engine::merge::decode_counter(&value),
                ) else {
                    damaged(found, &key, "not a blob count");
                    continue;
                };
                if !slice.holds(&content) {
                    continue;
                }
                seen.insert(content);
                let want = expected.get(&content).copied().unwrap_or(0);
                if want != actual {
                    fixes.count(kind, found, key.to_vec(), want - actual);
                }
            }
            for (content, want) in expected {
                if !seen.contains(content) && *want != 0 {
                    let key = match kind {
                        Count::Refs => refs_counter_key(content),
                        Count::Pins => pins_counter_key(content),
                    };
                    fixes.count(kind, found, key, *want);
                }
            }
        }

        // What the catalog publishes, what is retired, and what has a
        // candidate already.
        let mut eligible: FxHashMap<ScopedContent, u64> = FxHashMap::default();
        for guard in engine.snapshot_prefix_iter(snapshot, Partition::BlobRef, CATALOG_PREFIX)? {
            let (key, value) = guard.into_inner().map_err(StorageError::from)?;
            let content = key
                .strip_prefix(CATALOG_PREFIX)
                .and_then(ScopedContent::decode);
            match (content, CatalogEntry::from_msgpack(&value)) {
                (Some(content), Ok(entry)) => {
                    if slice.holds(&content) {
                        eligible.insert(content, entry.eligible.generation);
                    }
                }
                _ => damaged(found, &key, "not a catalog entry"),
            }
        }
        let mut retired: FxHashSet<(ScopedContent, u64)> = FxHashSet::default();
        for guard in engine.snapshot_prefix_iter(snapshot, Partition::BlobRef, RETIRED_PREFIX)? {
            let key = guard.key().map_err(StorageError::from)?;
            match decode_retired_key(&key) {
                Some(item) if slice.holds(&item.0) => {
                    retired.insert(item);
                }
                Some(_) => {}
                None => damaged(found, &key, "not a retirement"),
            }
        }
        let mut with_candidate: FxHashSet<ScopedContent> = FxHashSet::default();
        for guard in engine.snapshot_prefix_iter(snapshot, Partition::BlobRef, CANDIDATE_PREFIX)? {
            let key = guard.key().map_err(StorageError::from)?;
            match decode_candidate_key(&key) {
                Some(candidate) if slice.holds(&candidate.content) => {
                    with_candidate.insert(candidate.content);
                }
                Some(_) => {}
                None => damaged(found, &key, "not a candidate"),
            }
        }

        // Eligible content nothing references or holds, with no candidate
        // to bring it to a collector.
        for content in eligible.keys() {
            if refs.get(content).copied().unwrap_or(0) == 0
                && pins.get(content).copied().unwrap_or(0) == 0
                && !with_candidate.contains(content)
            {
                found.recorded_candidates += 1;
                fixes.candidates.push(*content);
            }
        }

        // Payload no eligible entry, upload or retirement names. Its
        // generation is unique and every way to name it again needs a record
        // that would exist at this snapshot already, so nothing ever will.
        for guard in engine.snapshot_prefix_iter(snapshot, Partition::Blob, PAYLOAD_PREFIX)? {
            let key = guard.key().map_err(StorageError::from)?;
            let Some((content, generation)) = decode_payload_key(&key) else {
                damaged(found, &key, "not a payload key");
                continue;
            };
            if !slice.holds(&content)
                || eligible.get(&content) == Some(&generation)
                || staged.contains(&(content, generation))
                || retired.contains(&(content, generation))
            {
                continue;
            }
            found.retired_strays += 1;
            fixes.strays.push((content, generation));
        }
        Ok(())
    }

    /// Commit the corrections a clean reconciliation found, in commits of a
    /// bounded size. Each correction is a delta measured at `snapshot` or a
    /// record nothing else writes, so a crash part way leaves nothing the
    /// next reconciliation does not measure again.
    fn apply(&self, fixes: &Fixes, now: i64, snapshot: lsm_tree::SeqNo) -> BlobResult<()> {
        for deltas in fixes.deltas.chunks(PER_COMMIT) {
            self.store.retrying(|store| {
                let mut txn = store.begin();
                for (key, delta) in deltas {
                    txn.push_counter_delta(key, *delta);
                }
                store.commit(txn)
            })?;
        }
        for contents in fixes.candidates.chunks(PER_COMMIT) {
            self.store.retrying(|store| {
                let mut txn = store.begin();
                for content in contents {
                    txn.put(
                        Partition::BlobRef,
                        &candidate_key(now, content, RECONCILIATION_OWNER, snapshot),
                        &[],
                    )?;
                }
                store.commit(txn)
            })?;
        }
        for strays in fixes.strays.chunks(PER_COMMIT) {
            self.store.retrying(|store| {
                let mut txn = store.begin();
                for (content, generation) in strays {
                    txn.put(
                        Partition::BlobRef,
                        &retired_key(content, *generation),
                        &now.to_be_bytes(),
                    )?;
                }
                store.commit(txn)
            })?;
        }
        Ok(())
    }

    /// Whether the store holds any blob record at all, the collector's own
    /// included: one key read per kind.
    fn holds_blob_records(&self) -> BlobResult<bool> {
        let engine = self.store.engine();
        let kinds: [(Partition, &[u8]); 11] = [
            (Partition::BlobRef, COLLECTOR_STATE_KEY),
            (Partition::BlobRef, DIRECT_OWNER_PREFIX),
            (Partition::BlobRef, S3_OWNER_PREFIX),
            (Partition::BlobRef, PROPERTY_OWNER_PREFIX),
            (Partition::BlobRef, STAGING_PREFIX),
            (Partition::BlobRef, CATALOG_PREFIX),
            (Partition::BlobRef, CANDIDATE_PREFIX),
            (Partition::BlobRef, RETIRED_PREFIX),
            (Partition::Blob, PAYLOAD_PREFIX),
            (Partition::Counter, REFS_COUNTER_PREFIX),
            (Partition::Counter, PINS_COUNTER_PREFIX),
        ];
        for (part, prefix) in kinds {
            if engine.prefix_scan(part, prefix)?.next().is_some() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The grace in force, in microseconds.
    fn grace_micros(&self) -> i64 {
        let configured = self.policy.lock().grace;
        duration_micros(configured.max(self.store.engine().retention_window()))
    }

    fn record(&self) -> BlobResult<CollectorRecord> {
        match self
            .store
            .engine()
            .get(Partition::BlobRef, COLLECTOR_STATE_KEY)?
        {
            Some(bytes) => decode(COLLECTOR_STATE_KEY, &bytes, |b| {
                rmp_serde::from_slice::<CollectorRecord>(b)
            }),
            None => Ok(CollectorRecord::default()),
        }
    }

    fn catalog_entry(&self, content: &ScopedContent) -> BlobResult<Option<CatalogEntry>> {
        Ok(self.store.catalog_entry(&catalog_key(content))?.0)
    }

    /// A count as committed now, zero when never counted.
    fn count(&self, key: &[u8]) -> BlobResult<i64> {
        match self.store.engine().get(Partition::Counter, key)? {
            Some(bytes) => decode(key, &bytes, crate::engine::merge::decode_counter),
            None => Ok(0),
        }
    }

    /// A measure the collector only ever adds to.
    fn measure(&self, key: &[u8]) -> BlobResult<u64> {
        let count = self.count(key)?;
        u64::try_from(count).map_err(|_| BlobError::Damaged {
            key: printable_key(key),
            message: format!("a measure that only grows reads {count}"),
        })
    }
}

/// The background worker of a [`BlobCollector`]; dropping it stops it.
pub struct CollectorWorker {
    stop: Arc<AtomicBool>,
    wake: Arc<Wake>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for CollectorWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollectorWorker")
            .field("running", &self.handle.is_some())
            .finish()
    }
}

impl Drop for CollectorWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.wake.interrupt();
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                tracing::error!("the blob collector panicked");
            }
        }
    }
}

/// The earliest of the times the next pass has work.
#[derive(Default)]
struct Deadline(Option<i64>);

impl Deadline {
    fn at(&mut self, at: Option<i64>) {
        if let Some(at) = at {
            self.0 = Some(self.0.map_or(at, |t| t.min(at)));
        }
    }
}

/// A content's eligible copy to retire, and the candidate keys its
/// retirement settles.
type ToRetire = ((ScopedContent, u64), Vec<Vec<u8>>);

/// Which count a correction moves.
#[derive(Clone, Copy)]
enum Count {
    Refs,
    Pins,
}

/// The contents whose digest starts with a byte in `first..end`.
struct Slice {
    first: usize,
    end: usize,
    /// Whether this slice reports the records every slice reads, so each is
    /// counted and named once.
    reports_records: bool,
}

impl Slice {
    fn holds(&self, content: &ScopedContent) -> bool {
        let byte = usize::from(content.content.digest()[0]);
        self.first <= byte && byte < self.end
    }
}

/// The corrections a reconciliation found.
#[derive(Default)]
struct Fixes {
    deltas: Vec<(Vec<u8>, i64)>,
    candidates: Vec<ScopedContent>,
    strays: Vec<(ScopedContent, u64)>,
}

impl Fixes {
    fn count(&mut self, kind: Count, found: &mut Reconciliation, key: Vec<u8>, delta: i64) {
        match kind {
            Count::Refs => found.corrected_refs += 1,
            Count::Pins => found.corrected_pins += 1,
        }
        self.deltas.push((key, delta));
    }
}

/// `duration` in microseconds. A duration past the representable future is
/// the representable future: a grace that long never ends, which is what it
/// asks for.
fn duration_micros(duration: Duration) -> i64 {
    i64::try_from(duration.as_micros()).unwrap_or(i64::MAX)
}

/// `after` past `now`. A time past the representable future is the
/// representable future, which a pass never reaches, as asked.
fn later(now: i64, after: Duration) -> i64 {
    now.saturating_add(duration_micros(after))
}

/// The time the record at `key` holds: when a representation was retired,
/// or when the oldest cleanup still owed a compaction ran.
fn stored_time(key: &[u8], value: &[u8]) -> BlobResult<i64> {
    let bytes: [u8; 8] = value.try_into().map_err(|_| BlobError::Damaged {
        key: printable_key(key),
        message: format!("a stored time holds {} bytes, expected 8", value.len()),
    })?;
    Ok(i64::from_be_bytes(bytes))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
