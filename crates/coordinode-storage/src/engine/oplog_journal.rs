//! Embedded oplog journal: the retained write-ahead log for oracle-backed,
//! no-Raft (embedded / single-node) engines.
//!
//! Unlike a truncating standalone WAL, this journal is RETAINED for a
//! configurable window so it can serve WAL-replay-repair — rebuild a corrupt
//! partition from the last checkpoint then replay the oplog forward — in
//! addition to ordinary crash recovery. It reuses the same [`OplogManager`]
//! segment format the cluster Raft log uses, so the repair routine
//! (`wal_replay_repair`) is shared between cluster and embedded.
//!
//! ## Why recovery needs no separate cursor
//!
//! Embedded engines open with the timestamp oracle, so every mutation's LSM
//! seqno equals its commit_ts equals the [`OplogEntry`] `ts`. An entry is
//! durable in a partition iff that partition's highest-persisted seqno is
//! at least the entry ts. Recovery therefore replays only the entries whose
//! `ts` exceeds a partition's persisted watermark — which is automatically
//! correct even when the background flush worker persists memtables
//! autonomously (a fixed index cursor would lag those flushes and risk
//! double-applying a non-idempotent merge).

use std::path::Path;

use coordinode_core::txn::proposal::Mutation;
use lsm_tree::{AbstractTree, AnyTree};

use crate::engine::partition::Partition;
use crate::error::StorageResult;
use crate::oplog::convert::mutations_to_ops;
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
}

impl Default for OplogJournalConfig {
    fn default() -> Self {
        Self {
            retention_secs: 7 * 24 * 3600,
            max_segment_bytes: 64 * 1024 * 1024,
            max_segment_entries: 50_000,
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
        )?;
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
    /// Must be called BEFORE the mutations are applied to the memtable so a
    /// record that reached the journal survives a crash and is replayed.
    pub(crate) fn append(&mut self, mutations: &[Mutation], commit_ts: u64) -> StorageResult<u64> {
        let index = self.next_index;
        self.next_index += 1;
        let entry = OplogEntry {
            ts: commit_ts,
            term: 0,
            index,
            shard: self.shard,
            ops: mutations_to_ops(mutations),
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

    /// All retained entries, ascending by index — for crash-recovery replay.
    pub(crate) fn read_all(&mut self) -> StorageResult<Vec<OplogEntry>> {
        self.manager.read_range(0, u64::MAX)
    }

    /// Entries with `index >= from_index` — for WAL-replay-repair, replaying
    /// the journal forward from a checkpoint's cursor.
    pub(crate) fn read_since(&mut self, from_index: u64) -> StorageResult<Vec<OplogEntry>> {
        self.manager.read_range(from_index, u64::MAX)
    }

    /// Delete segments fully outside the retention window. Recent un-flushed
    /// entries are always within the window, so this never drops a record
    /// still needed for crash recovery.
    pub(crate) fn purge_expired(&mut self, now_secs: u64) -> StorageResult<usize> {
        self.manager.purge_expired(now_secs)
    }
}

/// Last entry index present in an oplog directory, or `None` if empty — used
/// to derive a checkpoint's replay cursor (the journal copied into a
/// checkpoint). Opened with `u64::MAX` retention so nothing is purged.
pub(crate) fn last_index_in_dir(oplog_dir: &Path) -> StorageResult<Option<u64>> {
    let mgr = OplogManager::open(oplog_dir, 0, 64 * 1024 * 1024, 50_000, u64::MAX)?;
    Ok(mgr.recover_last_entry()?.map(|e| e.index))
}

/// The partition an op targets, or `None` for non-data ops (Raft framing /
/// Noop) and unknown tags (a partition from a future version).
pub(crate) fn op_partition(op: &OplogOp) -> Option<Partition> {
    let tag = match op {
        OplogOp::Insert { partition, .. }
        | OplogOp::Delete { partition, .. }
        | OplogOp::Merge { partition, .. }
        | OplogOp::RemoveRange { partition, .. } => *partition,
        // ColumnarInsert is routed to the columnar table registry by table_id,
        // not to a partition tree — see the columnar replay pass in `finish_open`.
        OplogOp::Noop
        | OplogOp::RaftEntry { .. }
        | OplogOp::RaftTruncation { .. }
        | OplogOp::ColumnarInsert { .. } => return None,
    };
    partition_from_wire_tag(tag)
}

/// Apply one data op to its partition tree at the given LSM seqno (the entry's
/// commit_ts). Non-data ops are no-ops.
pub(crate) fn apply_oplog_op(tree: &AnyTree, op: &OplogOp, seqno: u64) {
    match op {
        OplogOp::Insert { key, value, .. } => {
            tree.insert(key, value, seqno);
        }
        OplogOp::Delete { key, .. } => {
            tree.remove(key, seqno);
        }
        OplogOp::Merge { key, operand, .. } => {
            tree.merge(key, operand, seqno);
        }
        OplogOp::RemoveRange { start, end, .. } => {
            tree.remove_range(start.clone(), end.clone(), seqno);
        }
        // Non-partition ops: Raft framing/heartbeats are not data, and
        // ColumnarInsert targets a columnar table tree (not a partition tree),
        // so it is replayed in a separate pass that has the registry handle.
        OplogOp::Noop
        | OplogOp::RaftEntry { .. }
        | OplogOp::RaftTruncation { .. }
        | OplogOp::ColumnarInsert { .. } => {}
    }
}
