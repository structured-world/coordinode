//! Embedded oplog journal: the retained write-ahead log for oracle-backed,
//! no-Raft (embedded / single-node) engines.
//!
//! Unlike a truncating standalone WAL, this journal is RETAINED for a
//! configurable window so it can serve WAL-replay-repair — rebuild a corrupt
//! partition from the last checkpoint then replay the oplog forward — in
//! addition to ordinary crash recovery. It reuses the same [`OplogManager`]
//! segment format the cluster Raft log uses.
//!
//! ## What recovery replays
//!
//! An entry's `ts` is its MVCC version, not evidence that it is on disk: a
//! commit that reserved its timestamp early can reach the memtable after a
//! later one was flushed. Each partition tree therefore records, in-band,
//! which journal indices it physically holds (see `engine::coverage`), and
//! recovery replays entry `i` into partition `p` iff `p`'s record lacks `i`.
//!
//! ## Deferred durability
//!
//! An entry appended under a volatile write concern is not synced before the
//! caller is answered: under `memory` it stays in this process's buffer,
//! under `cache` it is handed to the operating system. A flusher thread
//! syncs the tail within the flush interval; a journaled append syncs it at
//! once, earlier entries included, and so does the journal sync every
//! table flush calls first, so that no table holds an effect the journal could
//! still lose. A crash therefore loses the unsynced tail in log order, never a
//! part of an entry and never an entry whose successor survived.
//!
//! A sync that fails leaves the journal refusing every later write: after a
//! failed `fsync` the kernel may have dropped the unwritten pages, so a later
//! sync reporting success would prove nothing (PostgreSQL's `data_sync_retry
//! = off` and MongoDB's journal flusher stop the server for the same reason).

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use coordinode_core::txn::frame::encode_unit;
use coordinode_core::txn::proposal::Mutation;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_core::txn::wake::Wake;
use coordinode_core::txn::write_concern::Journal;
use lsm_tree::{AbstractTree, AnyTree};

use crate::engine::config::{StorageConfig, SyncMethod};
use crate::engine::coverage;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::{OplogEntry, OplogOp, ShardId};
use crate::oplog::manager::OplogManager;
use crate::placement::partition_from_wire_tag;

/// Tuning for the embedded oplog journal.
#[derive(Debug, Clone)]
pub struct OplogJournalConfig {
    /// Retention window in seconds. Entries whose segment `last_ts` is older
    /// than `now - retention_secs` are eligible for purge. Must exceed the
    /// checkpoint interval so a repair base + the oplog since it always
    /// coexist. Default: 7 days.
    pub retention_secs: u64,
    /// Maximum entry bytes per segment before rotation.
    pub max_segment_bytes: u64,
    /// Maximum entries per segment before rotation.
    pub max_segment_entries: u32,
    /// How an append is made durable.
    pub sync_method: SyncMethod,
    /// The interval the flusher syncs entries appended under a volatile
    /// write concern at, in milliseconds. `0` syncs as soon as it runs.
    pub volatile_flush_interval_ms: u64,
    /// Unsynced bytes past which the flusher syncs without waiting out its
    /// interval; past sixteen times this, the appender syncs itself, so the
    /// unsynced tail stays bounded however fast volatile writes arrive.
    pub volatile_flush_bytes: u64,
}

impl Default for OplogJournalConfig {
    fn default() -> Self {
        Self {
            retention_secs: 7 * 24 * 3600,
            max_segment_bytes: 64 * 1024 * 1024,
            max_segment_entries: 50_000,
            sync_method: SyncMethod::default(),
            volatile_flush_interval_ms: 100,
            volatile_flush_bytes: 1024 * 1024,
        }
    }
}

impl From<&StorageConfig> for OplogJournalConfig {
    /// The oplog settings an operator gives the engine, shared by the embedded
    /// journal and the Raft log.
    fn from(config: &StorageConfig) -> Self {
        Self {
            retention_secs: config.oplog_retention_secs,
            max_segment_bytes: config.oplog_segment_max_bytes,
            max_segment_entries: config.oplog_segment_max_entries,
            sync_method: config.oplog_sync_method,
            volatile_flush_interval_ms: config.volatile_flush_interval_ms,
            volatile_flush_bytes: config.volatile_flush_bytes,
        }
    }
}

/// How many times [`OplogJournalConfig::volatile_flush_bytes`] the unsynced
/// tail may reach before the appender syncs it itself instead of waiting for
/// the flusher.
const APPENDER_SYNC_FACTOR: u64 = 16;

/// What the journal shares with its flusher and with readers choosing a view.
#[derive(Debug)]
pub(crate) struct DeferredState {
    /// Notified on every append left unsynced.
    wake: Wake,
    /// Set when the unsynced tail passed the flush size: sync without waiting
    /// out the interval.
    urgent: AtomicBool,
    /// The lowest commit timestamp an appended, not yet durable entry carries;
    /// `u64::MAX` when every entry is durable. A view below it holds only
    /// durable effects. Lowered under the journal lock before the entry is
    /// applied, raised only by a completed sync. The cell belongs to the
    /// engine's pending-commit table, where views and the GC watermark read it.
    durable_ceiling: Arc<AtomicU64>,
}

/// One appended entry not yet durable.
#[derive(Debug, Clone, Copy)]
struct Unsynced {
    index: u64,
    commit_ts: u64,
    bytes: u64,
}

/// Retained, oracle-coupled oplog journal owned by a standalone engine.
pub(crate) struct EmbeddedOplog {
    manager: OplogManager,
    shard: ShardId,
    /// Index to assign to the next appended entry. Recovered from the last
    /// on-disk entry at open so it is monotonic across restarts.
    next_index: u64,
    /// Every entry below this index is durable. What open recovered is:
    /// recovery seals the segments it keeps with a sync.
    durable_below: u64,
    /// Why a sync failed, once one has: the journal then takes no writes.
    failed: Option<String>,
    /// Shared with [`JournalFlusher`] and the engine's view selection.
    deferred: Arc<DeferredState>,
    /// Entries appended and not yet durable, in index order.
    unsynced: std::collections::VecDeque<Unsynced>,
    /// Their encoded bytes.
    unsynced_bytes: u64,
    /// See [`OplogJournalConfig::volatile_flush_bytes`].
    flush_bytes: u64,
}

impl EmbeddedOplog {
    /// Open (or create) the journal directory and recover the next index.
    /// `durable_ceiling` is where it publishes its unsynced tail's lowest
    /// commit timestamp.
    pub(crate) fn open(
        dir: &Path,
        shard: ShardId,
        cfg: &OplogJournalConfig,
        durable_ceiling: Arc<AtomicU64>,
    ) -> StorageResult<Self> {
        let manager = OplogManager::open(
            dir,
            shard,
            cfg.max_segment_bytes,
            cfg.max_segment_entries,
            cfg.retention_secs,
        )?
        .with_sync_method(cfg.sync_method);
        let next_index = manager
            .recover_last_entry()?
            .map(|e| e.index + 1)
            .unwrap_or(0);
        Ok(Self {
            manager,
            shard,
            next_index,
            durable_below: next_index,
            failed: None,
            deferred: Arc::new(DeferredState {
                wake: Wake::default(),
                urgent: AtomicBool::new(false),
                durable_ceiling,
            }),
            unsynced: std::collections::VecDeque::new(),
            unsynced_bytes: 0,
            flush_bytes: cfg.volatile_flush_bytes,
        })
    }

    /// Append one proposal's mutations as a single oplog entry stamped at
    /// `commit_ts`, made as durable as `level` asks before this returns:
    /// synced for [`Journal::Journal`], handed to the operating system for
    /// [`Journal::Cache`], left in this process's buffer for
    /// [`Journal::Memory`]. Returns the assigned entry index.
    ///
    /// The entry holds the unit in its compact frame, the encoding the Raft
    /// log carries a proposal in. Must be called BEFORE the mutations are
    /// applied to the memtable so a record that reached the journal survives
    /// a crash and is replayed.
    pub(crate) fn append(
        &mut self,
        mutations: &[Mutation],
        commit_ts: u64,
        level: Journal,
    ) -> StorageResult<u64> {
        if mutations.iter().any(|m| matches!(m, Mutation::Command(_))) {
            return Err(StorageError::InvalidConfig(
                "a metadata command reached the journal undecided".into(),
            ));
        }
        let frame = encode_unit(mutations, Timestamp::from_raw(commit_ts))?;
        // The frame is the entry's payload; its framing adds a few bytes.
        let bytes = frame.len() as u64;
        self.append_ops(vec![OplogOp::Unit { frame }], commit_ts, level, bytes)
    }

    /// Append a single `STORAGE COLUMNAR` row write as one oplog entry stamped
    /// at `commit_ts`, then fsync. Returns the assigned entry index.
    ///
    /// Columnar tables live outside the `Partition` keyspace, so their writes
    /// carry a [`OplogOp::ColumnarInsert`] tagged with the table id rather than
    /// a partition discriminant. Like [`append`](Self::append), must be called
    /// BEFORE the row is applied to the columnar tree's memtable so a write that
    /// reached the journal survives a crash and is replayed.
    pub(crate) fn append_columnar(
        &mut self,
        table_id: &str,
        key: &[u8],
        value: &[u8],
        commit_ts: u64,
    ) -> StorageResult<u64> {
        let bytes = (table_id.len() + key.len() + value.len()) as u64;
        let ops = vec![OplogOp::ColumnarInsert {
            table_id: table_id.to_owned(),
            key: key.to_vec(),
            value: value.to_vec(),
        }];
        self.append_ops(ops, commit_ts, Journal::Journal, bytes)
    }

    /// Append `ops`, about `bytes` long, as the next entry and take it as far
    /// as `level` asks.
    fn append_ops(
        &mut self,
        ops: Vec<OplogOp>,
        commit_ts: u64,
        level: Journal,
        bytes: u64,
    ) -> StorageResult<u64> {
        if let Some(failed) = &self.failed {
            return Err(StorageError::Io(format!(
                "the journal takes no writes after a failed sync ({failed}); reopen the store"
            )));
        }
        let index = self.next_index;
        self.next_index += 1;
        let entry = OplogEntry {
            ts: commit_ts,
            term: 0,
            index,
            shard: self.shard,
            ops,
            is_migration: false,
            pre_images: None,
        };
        self.manager.append(&entry)?;
        if level != Journal::Journal {
            // Recorded before the entry is applied: a view chosen once its
            // effects are visible already stops below it.
            self.unsynced.push_back(Unsynced {
                index,
                commit_ts,
                bytes,
            });
            self.unsynced_bytes += bytes;
            self.deferred
                .durable_ceiling
                .fetch_min(commit_ts, Ordering::AcqRel);
            metrics::gauge!("coordinode_journal_unsynced_bytes").set(self.unsynced_bytes as f64);
        }
        // A flush size so large its multiple overflows means no appender cap.
        let backlog = self
            .flush_bytes
            .checked_mul(APPENDER_SYNC_FACTOR)
            .is_some_and(|cap| self.unsynced_bytes >= cap);
        match level {
            Journal::Journal => {
                // The sync covers every entry appended before this one.
                let synced = self.manager.flush();
                self.note_sync(self.next_index, synced)?;
            }
            // Bounded outstanding work: past the cap the appender pays the
            // sync itself rather than let the tail grow without limit.
            _ if backlog => {
                metrics::counter!("coordinode_journal_appender_syncs_total").increment(1);
                let synced = self.manager.flush();
                self.note_sync(self.next_index, synced)?;
            }
            Journal::Cache => {
                // A failed hand-over can leave part of the entry written.
                if let Err(e) = self.manager.flush_to_os() {
                    return Err(self.fail(e));
                }
                self.notify_deferred();
            }
            Journal::Memory => self.notify_deferred(),
        }
        Ok(index)
    }

    /// Wake the flusher for a deferred entry, at once when the unsynced tail
    /// passed the flush size.
    fn notify_deferred(&self) {
        if self.unsynced_bytes >= self.flush_bytes {
            self.deferred.urgent.store(true, Ordering::Release);
            self.deferred.wake.interrupt();
        } else {
            self.deferred.wake.notify();
        }
    }

    /// The index the next appended entry receives.
    pub(crate) fn next_index(&self) -> u64 {
        self.next_index
    }

    /// Never hand out an index below `floor`: an index the partition trees
    /// already record as covered must not be reused after the segments that
    /// held it were purged.
    pub(crate) fn advance_next_index(&mut self, floor: u64) {
        // Called at open, before any append: the skipped indices hold no
        // entry, so nothing below the floor waits for a sync.
        self.next_index = self.next_index.max(floor);
        self.durable_below = self.durable_below.max(self.next_index);
    }

    /// What the journal shares with its flusher and with view selection.
    pub(crate) fn deferred(&self) -> Arc<DeferredState> {
        Arc::clone(&self.deferred)
    }

    /// Hand every entry appended so far to the operating system and return
    /// the index they end below with what syncs them, so the sync runs
    /// without this journal locked. `None` when every entry is durable.
    ///
    /// # Errors
    ///
    /// A sync failed earlier, or handing the entries over fails.
    fn begin_sync(&mut self) -> StorageResult<Option<(u64, Option<crate::oplog::SyncHandle>)>> {
        if let Some(failed) = &self.failed {
            return Err(StorageError::Io(format!(
                "the journal is not durable after a failed sync ({failed})"
            )));
        }
        if self.durable_below >= self.next_index {
            return Ok(None);
        }
        // A segment rotated out since the last sync was synced when sealed,
        // so the active segment's handle covers what remains.
        let handle = match self.manager.flush_to_os() {
            Ok(handle) => handle,
            Err(e) => return Err(self.fail(e)),
        };
        Ok(Some((self.next_index, handle)))
    }

    /// Record how a sync of every entry below `through` went.
    pub(crate) fn note_sync(
        &mut self,
        through: u64,
        synced: StorageResult<()>,
    ) -> StorageResult<()> {
        match synced {
            Ok(()) => {
                metrics::counter!("coordinode_journal_syncs_total").increment(1);
                self.durable_below = self.durable_below.max(through);
                while self.unsynced.front().is_some_and(|u| u.index < through) {
                    if let Some(done) = self.unsynced.pop_front() {
                        self.unsynced_bytes -= done.bytes;
                    }
                }
                // Commit timestamps need not follow indices, so the ceiling
                // is the least over what is left, not the next one's.
                let ceiling = self
                    .unsynced
                    .iter()
                    .map(|u| u.commit_ts)
                    .min()
                    .unwrap_or(u64::MAX);
                self.deferred
                    .durable_ceiling
                    .store(ceiling, Ordering::Release);
                metrics::gauge!("coordinode_journal_unsynced_bytes")
                    .set(self.unsynced_bytes as f64);
                Ok(())
            }
            Err(e) => {
                metrics::counter!("coordinode_journal_sync_failures_total").increment(1);
                Err(self.fail(e))
            }
        }
    }

    /// Stop taking writes after `e`, and return it.
    fn fail(&mut self, e: StorageError) -> StorageError {
        let reason = e.to_string();
        tracing::error!(
            error = %reason,
            "oplog journal sync failed; the store takes no more writes until reopened"
        );
        self.failed.get_or_insert(reason);
        e
    }

    /// The active segment's path and the length a power cut leaves it at:
    /// every entry not yet durable is cut off. `None` when nothing would be.
    #[cfg(test)]
    pub(crate) fn power_cut_point(&self) -> Option<(std::path::PathBuf, u64)> {
        self.manager.power_cut_at(self.durable_below)
    }

    /// The active segment's path and the length a killed process leaves it
    /// at; `None` when no segment is active.
    #[cfg(test)]
    pub(crate) fn kill_point(&self) -> Option<(std::path::PathBuf, u64)> {
        self.manager.kill_point()
    }

    /// All retained entries, ascending by index — for crash-recovery replay.
    pub(crate) fn read_all(&mut self) -> StorageResult<Vec<OplogEntry>> {
        self.manager.read_range(0, u64::MAX)
    }

    /// Entries with `index >= from_index` — for WAL-replay-repair, replaying
    /// the journal forward from a checkpoint's cursor.
    pub(crate) fn read_since(&mut self, from_index: u64) -> StorageResult<Vec<OplogEntry>> {
        self.manager.read_range(from_index, u64::MAX)
    }

    /// Delete segments outside the retention window, except those still
    /// needed at or above `keep_from_index` (the latest checkpoint's replay
    /// cursor; `u64::MAX` when none) and those holding any entry
    /// `is_durable` denies — a record whose only copy is this journal.
    pub(crate) fn purge_expired(
        &mut self,
        now_secs: u64,
        keep_from_index: u64,
        is_durable: &dyn Fn(&OplogEntry) -> bool,
    ) -> StorageResult<usize> {
        self.manager
            .purge_with_floor(now_secs, keep_from_index, is_durable)
    }
}

/// Make every entry `oplog` holds durable. The sync itself runs without the
/// journal locked, so appends go on meanwhile; one sync covers every entry
/// appended before it started, whoever asked for it.
///
/// # Errors
///
/// The sync fails, now or earlier; the journal then takes no more writes.
pub(crate) fn sync_journal(oplog: &Mutex<EmbeddedOplog>) -> StorageResult<()> {
    let lock = || {
        oplog
            .lock()
            .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))
    };
    let Some((through, handle)) = lock()?.begin_sync()? else {
        return Ok(());
    };
    let synced = handle.map_or(Ok(()), |handle| handle.sync());
    lock()?.note_sync(through, synced)
}

/// Syncs the journal's deferred tail within the flush interval of its first
/// unsynced entry. Asleep while every entry is durable; a deferred append
/// wakes it. Dropping it syncs what is left.
pub(crate) struct JournalFlusher {
    shared: Arc<FlusherShared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

struct FlusherShared {
    oplog: Arc<Mutex<EmbeddedOplog>>,
    deferred: Arc<DeferredState>,
    interval: Duration,
    stop: AtomicBool,
}

impl JournalFlusher {
    /// Start the flusher of `oplog`, syncing within `interval`.
    ///
    /// # Errors
    ///
    /// The thread cannot be spawned.
    pub(crate) fn start(
        oplog: &Arc<Mutex<EmbeddedOplog>>,
        interval: Duration,
    ) -> StorageResult<Self> {
        let deferred = oplog
            .lock()
            .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?
            .deferred();
        let shared = Arc::new(FlusherShared {
            oplog: Arc::clone(oplog),
            deferred,
            interval,
            stop: AtomicBool::new(false),
        });
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("coord-journal-flush".into())
            .spawn(move || worker.run())
            .map_err(|e| StorageError::InvalidConfig(format!("journal flusher spawn: {e}")))?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }
}

impl Drop for JournalFlusher {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.deferred.wake.interrupt();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::error!("the journal flusher panicked");
            }
        }
    }
}

impl FlusherShared {
    fn run(&self) {
        let wake = &self.deferred.wake;
        wake.bind();
        loop {
            if !self.stop.load(Ordering::Acquire) {
                wake.wait(None);
            }
            // The window opens at the first deferred append; later ones join
            // the same sync. A stop, or a tail past the flush size, cuts it
            // short.
            let deadline = Instant::now() + self.interval;
            while !self.stop.load(Ordering::Acquire)
                && !self.deferred.urgent.swap(false, Ordering::AcqRel)
            {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                wake.wait(Some(deadline - now));
            }
            // A failure leaves the journal refusing writes, so no later round
            // has anything to retry.
            if let Err(e) = sync_journal(&self.oplog) {
                tracing::debug!(error = %e, "journal flusher: the tail stays unsynced");
            }
            if self.stop.load(Ordering::Acquire) {
                return;
            }
        }
    }
}

/// The partition an op targets, or `None` for non-data ops (Raft framing /
/// Noop) and unknown tags (a partition from a future version).
pub(crate) fn op_partition(op: &OplogOp) -> Option<Partition> {
    let tag = match op {
        OplogOp::Insert { partition, .. }
        | OplogOp::Delete { partition, .. }
        | OplogOp::Merge { partition, .. }
        | OplogOp::RemoveRange { partition, .. } => *partition,
        // Its entries are index records. Replay resolves the work before it
        // routes ops; an unresolved one routed here fails where it applies.
        OplogOp::Derive { .. } => return Some(Partition::Idx),
        // A frame spans partitions: replay expands it before it routes ops,
        // so one reaching here is routed where it fails loudly, not skipped.
        OplogOp::Unit { .. } => return Some(Partition::Node),
        // ColumnarInsert is routed to the columnar table registry by table_id,
        // not to a partition tree — see the columnar replay pass in `finish_open`.
        OplogOp::Noop
        | OplogOp::RaftEntry { .. }
        | OplogOp::RaftTruncation { .. }
        | OplogOp::ColumnarInsert { .. } => return None,
    };
    partition_from_wire_tag(tag)
}

/// Replay one entry's data ops into a single partition tree at the entry's
/// commit_ts, together with the entry's coverage marker, in the same order
/// the commit path uses: range tombstones first, then the point ops and the
/// marker as ONE tree batch, so a persisted marker implies every effect of
/// the entry in this tree is persisted. Non-data ops are no-ops.
/// `installations` is the index tree's catalog when `tree` is the index tree.
pub(crate) fn apply_oplog_ops_at(
    tree: &AnyTree,
    installations: Option<&crate::engine::installation::Installations>,
    ops: &[&OplogOp],
    seqno: u64,
    index: u64,
) -> StorageResult<()> {
    let mut addressing = crate::engine::installation::Addressing::new(installations, tree, seqno);
    let mut batch = lsm_tree::WriteBatch::with_capacity(ops.len() + 1);
    for op in ops {
        match op {
            // A replacement being prepared receives the same write in the
            // same batch, as on the live apply path.
            OplogOp::Insert { key, value, .. } => {
                batch.insert(addressing.point(key)?, value.as_slice());
                if let Some(staging) = addressing.staging_point(key) {
                    batch.insert(staging, value.as_slice());
                }
            }
            OplogOp::Delete { key, .. } => {
                batch.remove(addressing.point(key)?);
                if let Some(staging) = addressing.staging_point(key) {
                    batch.remove(staging);
                }
            }
            OplogOp::Merge { key, operand, .. } => {
                batch.merge(addressing.point(key)?, operand.as_slice());
                if let Some(staging) = addressing.staging_point(key) {
                    batch.merge(staging, operand.as_slice());
                }
            }
            OplogOp::RemoveRange { start, end, .. } => {
                let start = coverage::clamp_user_start(start);
                let ranges = addressing.range(start, end);
                let staging = addressing.staging_range(start, end);
                for (start, end) in ranges.into_iter().chain(staging) {
                    if start < end {
                        tree.remove_range(start, end, seqno);
                    }
                }
            }
            // Non-partition ops: Raft framing/heartbeats are not data, and
            // ColumnarInsert targets a columnar table tree (not a partition
            // tree), so it is replayed in a separate pass with the registry.
            OplogOp::Noop
            | OplogOp::RaftEntry { .. }
            | OplogOp::RaftTruncation { .. }
            | OplogOp::ColumnarInsert { .. } => {}
            OplogOp::Derive { .. } | OplogOp::Unit { .. } => {
                return Err(StorageError::InvalidConfig(
                    "a journal entry replayed without its unit resolved".into(),
                ));
            }
        }
    }
    batch.insert(
        coverage::Domain::Journal.marker_key(index, 0).as_slice(),
        &[][..],
    );
    tree.apply_batch(batch, seqno)?;
    Ok(())
}
