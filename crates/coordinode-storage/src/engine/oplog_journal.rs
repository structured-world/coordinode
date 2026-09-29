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

use std::path::Path;

use coordinode_core::txn::frame::encode_unit;
use coordinode_core::txn::proposal::Mutation;
use coordinode_core::txn::timestamp::Timestamp;
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
}

impl Default for OplogJournalConfig {
    fn default() -> Self {
        Self {
            retention_secs: 7 * 24 * 3600,
            max_segment_bytes: 64 * 1024 * 1024,
            max_segment_entries: 50_000,
            sync_method: SyncMethod::default(),
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
        }
    }
}

/// Retained, oracle-coupled oplog journal owned by a standalone engine.
pub(crate) struct EmbeddedOplog {
    manager: OplogManager,
    shard: ShardId,
    /// Index to assign to the next appended entry. Recovered from the last
    /// on-disk entry at open so it is monotonic across restarts.
    next_index: u64,
}

impl EmbeddedOplog {
    /// Open (or create) the journal directory and recover the next index.
    pub(crate) fn open(
        dir: &Path,
        shard: ShardId,
        cfg: &OplogJournalConfig,
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
        })
    }

    /// Append one proposal's mutations as a single oplog entry stamped at
    /// `commit_ts`, then fsync. Returns the assigned entry index.
    ///
    /// The entry holds the unit in its compact frame, the encoding the Raft
    /// log carries a proposal in. Must be called BEFORE the mutations are
    /// applied to the memtable so a record that reached the journal survives
    /// a crash and is replayed.
    pub(crate) fn append(&mut self, mutations: &[Mutation], commit_ts: u64) -> StorageResult<u64> {
        if mutations.iter().any(|m| matches!(m, Mutation::Command(_))) {
            return Err(StorageError::InvalidConfig(
                "a metadata command reached the journal undecided".into(),
            ));
        }
        let frame = encode_unit(mutations, Timestamp::from_raw(commit_ts))?;
        let ops = vec![OplogOp::Unit { frame }];
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
        self.manager.flush()?;
        Ok(index)
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
        let index = self.next_index;
        self.next_index += 1;
        let entry = OplogEntry {
            ts: commit_ts,
            term: 0,
            index,
            shard: self.shard,
            ops: vec![OplogOp::ColumnarInsert {
                table_id: table_id.to_owned(),
                key: key.to_vec(),
                value: value.to_vec(),
            }],
            is_migration: false,
            pre_images: None,
        };
        self.manager.append(&entry)?;
        self.manager.flush()?;
        Ok(index)
    }

    /// The index the next appended entry receives.
    pub(crate) fn next_index(&self) -> u64 {
        self.next_index
    }

    /// Never hand out an index below `floor`: an index the partition trees
    /// already record as covered must not be reused after the segments that
    /// held it were purged.
    pub(crate) fn advance_next_index(&mut self, floor: u64) {
        self.next_index = self.next_index.max(floor);
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
pub(crate) fn apply_oplog_ops_at(
    tree: &AnyTree,
    ops: &[&OplogOp],
    seqno: u64,
    index: u64,
) -> StorageResult<()> {
    let mut batch = lsm_tree::WriteBatch::with_capacity(ops.len() + 1);
    for op in ops {
        match op {
            OplogOp::Insert { key, value, .. } => batch.insert(key.as_slice(), value.as_slice()),
            OplogOp::Delete { key, .. } => batch.remove(key.as_slice()),
            OplogOp::Merge { key, operand, .. } => {
                batch.merge(key.as_slice(), operand.as_slice());
            }
            OplogOp::RemoveRange { start, end, .. } => {
                let start = coverage::clamp_user_start(start);
                if start < end.as_slice() {
                    tree.remove_range(start.to_vec(), end.clone(), seqno);
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
