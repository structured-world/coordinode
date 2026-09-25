//! Apply coverage in the Raft log's index space.
//!
//! The Raft state machine applies entries serially, but the partition trees
//! flush independently, so after a crash each tree holds its own prefix of
//! the applied entries, and the state machine's own "last applied" record is
//! just another key in one of them. Each tree therefore records which
//! `(log index, proposal position)` applies it holds, exactly as it does for
//! the embedded journal (see `engine::coverage`), and the state machine
//! resumes from the lowest covered prefix, skipping per tree what a tree
//! already holds.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use coordinode_core::txn::proposal::Mutation;
use lsm_tree::AbstractTree;

use super::{Rows, StorageEngine, claim_checkpoint_dir};
use crate::engine::coverage::{self, Domain, Mark, TreeCoverage};
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};
use crate::placement::partition_wire_tag;

/// One slot per partition wire tag.
const FLOOR_SLOTS: usize = 16;

/// Directory under the data dir where partitions are captured for a copy to
/// another node. Nothing under it outlives the copy; the engine clears what
/// a crash left behind when it opens.
pub(super) const PARTITION_CAPTURE_DIR: &str = "partition-capture";

/// Where this node's Raft applies stand: one past the last applied entry,
/// and, per partition, the entry below which the partition already holds
/// everything because it was installed from a peer further along.
///
/// The Raft state machine owns it and changes it only while applying;
/// anything else reads or changes it through a [`RaftApplyFence`], with the
/// applies paused.
#[derive(Debug, Default)]
pub struct RaftApplyState {
    next: u64,
    floors: [u64; FLOOR_SLOTS],
}

impl RaftApplyState {
    /// Applies standing at `next`, with no partition ahead of them.
    #[must_use]
    pub fn new(next: u64) -> Self {
        Self {
            next,
            floors: [0; FLOOR_SLOTS],
        }
    }

    /// One past the last applied entry.
    #[must_use]
    pub fn next(&self) -> u64 {
        self.next
    }

    /// Record that entry `index` was applied.
    pub fn applied(&mut self, index: u64) {
        self.next = index + 1;
    }

    /// Every partition now holds exactly the entries below `next`: a
    /// snapshot install.
    pub fn reset(&mut self, next: u64) {
        *self = Self::new(next);
    }

    /// Whether `partition` already holds entry `index`, because it was
    /// installed at a later position than the applies had reached.
    #[must_use]
    pub fn skips(&self, partition: Partition, index: u64) -> bool {
        index < self.floor(partition)
    }

    /// The entry below which `partition` holds everything; `0` when it is
    /// not ahead of the applies.
    #[must_use]
    pub fn floor(&self, partition: Partition) -> u64 {
        self.floors[slot(partition)]
    }

    /// `partition` now holds every entry below `next`.
    pub fn raise_floor(&mut self, partition: Partition, next: u64) {
        let floor = &mut self.floors[slot(partition)];
        *floor = (*floor).max(next);
    }
}

fn slot(partition: Partition) -> usize {
    let tag = usize::from(partition_wire_tag(partition));
    debug_assert!(
        tag < FLOOR_SLOTS,
        "partition tag {tag} past the floor slots"
    );
    tag
}

/// Pauses this node's Raft applies so a caller can read or replace a
/// partition at an exact log position.
///
/// A store's data at a moment is the state after some prefix of the log
/// only while nothing applies: entries apply at their own commit timestamp,
/// not in timestamp order, so no MVCC snapshot marks a prefix.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot pause this node's Raft applies",
    label = "not a Raft apply fence",
    note = "the Raft state machine registers its fence with `StorageEngine::register_raft_fence`"
)]
pub trait RaftApplyFence: Send + Sync {
    /// Run `work` while no Raft entry applies on this node. `work` receives
    /// where the applies stand and the last applied entry's log id as a
    /// coverage payload (empty before the first entry).
    ///
    /// Blocks; call it off the async runtime.
    ///
    /// # Errors
    ///
    /// Whatever `work` returns.
    fn with_applies_paused(
        &self,
        work: &mut dyn FnMut(&mut RaftApplyState, &[u8]) -> StorageResult<()>,
    ) -> StorageResult<()>;
}

/// Which Raft entries a partition tree of a checkpoint holds: its base and
/// markers.
#[derive(Debug)]
pub struct RaftHeld(TreeCoverage);

impl RaftHeld {
    /// Whether proposal `sub` of entry `index` is in the tree.
    #[must_use]
    pub fn holds(&self, index: u64, sub: u32) -> bool {
        self.0.contains(index, sub)
    }

    /// Every entry below this is in the tree.
    #[must_use]
    pub fn base_next(&self) -> u64 {
        self.0.base().map_or(0, |(next, _)| next)
    }
}

/// Where a copy of a partition stands in the Raft log: it holds every entry
/// below `next`, and `payload` is the log id of entry `next - 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaftPosition {
    /// One past the last entry the copy holds.
    pub next: u64,
    /// The log id of entry `next - 1`, as a coverage base carries it.
    pub payload: Vec<u8>,
}

/// A partition's replicated rows, and where they stand in the Raft log when
/// the engine they came from runs one.
#[derive(Debug)]
pub struct PartitionCopy {
    /// Every replicated row, in key order.
    pub rows: Rows,
    /// Where the rows stand; `None` from an engine without Raft.
    pub position: Option<RaftPosition>,
}

/// Keys that belong to this node rather than to the replicated data: the
/// Raft partition (vote, log metadata) and, in Schema, the per-node routing
/// (`meta:`) and the state machine's own records (`raft:`). They never
/// travel between nodes and survive a partition's replacement.
#[must_use]
pub fn is_node_local(partition: Partition, key: &[u8]) -> bool {
    match partition {
        Partition::Raft => true,
        Partition::Schema => key.starts_with(b"meta:") || key.starts_with(b"raft:"),
        _ => false,
    }
}

/// What every partition tree records of the Raft entries it holds, read
/// when the state machine opens.
#[derive(Debug)]
pub struct RaftCoverage {
    trees: HashMap<Partition, TreeCoverage>,
}

impl RaftCoverage {
    /// Whether any tree carries a Raft coverage record. A store that applied
    /// Raft entries without one was written by a release that predates apply
    /// coverage.
    #[must_use]
    pub fn has_record(&self) -> bool {
        self.trees.values().any(TreeCoverage::has_record)
    }

    /// Whether proposal `sub` of log entry `index` is physically in the tree
    /// of `partition`.
    #[must_use]
    pub fn holds(&self, partition: Partition, index: u64, sub: u32) -> bool {
        self.trees
            .get(&partition)
            .is_some_and(|c| c.contains(index, sub))
    }

    /// The lowest base across the trees with the payload written beside it:
    /// every entry below it is in every tree, so the state machine resumes
    /// there. `None` when no tree carries a record.
    #[must_use]
    pub fn resume_point(&self) -> Option<(u64, &[u8])> {
        self.trees
            .values()
            .filter_map(TreeCoverage::base)
            .min_by_key(|(next, _)| *next)
    }

    /// One past the highest entry any tree records: from there on, no tree
    /// holds anything the state machine has to skip.
    #[must_use]
    pub fn skip_until(&self) -> u64 {
        self.trees
            .values()
            .map(TreeCoverage::next_uncovered)
            .max()
            .unwrap_or(0)
    }
}

impl StorageEngine {
    /// Read the Raft coverage record of every partition tree.
    ///
    /// # Errors
    ///
    /// A tree read failure, or a malformed record.
    pub fn raft_coverage(&self) -> StorageResult<RaftCoverage> {
        let mut trees = HashMap::with_capacity(Partition::all().len());
        for (&part, tree) in self.coordinator.trees() {
            trees.insert(part, TreeCoverage::read(tree, Domain::Raft)?);
        }
        Ok(RaftCoverage { trees })
    }

    /// Apply proposal `sub` of log entry `index` at `commit_ts`, with its
    /// coverage marker, into every partition the proposal touches except
    /// those `skip` says already hold it. Returns how many mutations landed.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::apply_proposal_at`].
    pub fn apply_raft_proposal(
        &self,
        mutations: &[Mutation],
        commit_ts: u64,
        index: u64,
        sub: u32,
        skip: impl Fn(Partition) -> bool,
    ) -> StorageResult<usize> {
        let mark = Mark {
            domain: Domain::Raft,
            index,
            sub,
        };
        let partition_of = |m: &Mutation| match m {
            Mutation::Put { partition, .. }
            | Mutation::Delete { partition, .. }
            | Mutation::Merge { partition, .. }
            | Mutation::RemoveRange { partition, .. } => Partition::from(*partition),
        };
        if mutations.iter().any(|m| skip(partition_of(m))) {
            let kept: Vec<Mutation> = mutations
                .iter()
                .filter(|m| !skip(partition_of(m)))
                .cloned()
                .collect();
            self.apply_proposal_covered(&kept, commit_ts, Some(mark))?;
            Ok(kept.len())
        } else {
            self.apply_proposal_covered(mutations, commit_ts, Some(mark))?;
            Ok(mutations.len())
        }
    }

    /// Fold every Raft entry below `next` into each tree's base, removing the
    /// markers from `from` on. `payload` describes the last covered entry and
    /// comes back from [`RaftCoverage::resume_point`]. A tree `ahead` says
    /// already holds everything below `next` is left alone: its base is past
    /// `next`, and a fold would lower it.
    pub fn fold_raft_coverage(
        &self,
        from: u64,
        next: u64,
        payload: &[u8],
        ahead: impl Fn(Partition) -> bool,
    ) {
        // Above every marker below `next`: those proposals were applied, and
        // the oracle advanced past their commit_ts, before this call.
        let at = self.next_seqno();
        for (&part, tree) in self.coordinator.trees() {
            if !ahead(part) {
                coverage::write_fold(tree, Domain::Raft, from, next, payload, at);
            }
        }
    }

    /// Make `fence` the one that pauses this node's Raft applies, replacing
    /// any registered before (a state machine reopened over the engine).
    pub fn register_raft_fence(&self, fence: Arc<dyn RaftApplyFence>) {
        *self.raft_fence.write() = Some(fence);
    }

    /// The fence pausing this node's Raft applies, when a Raft state machine
    /// runs over the engine.
    #[must_use]
    pub fn raft_fence(&self) -> Option<Arc<dyn RaftApplyFence>> {
        self.raft_fence.read().clone()
    }

    /// Capture `partition` as it stands into `target` (which must not
    /// exist), for [`Self::open_checkpoint`]. Taken with the applies paused,
    /// the capture is the partition at the applies' position.
    ///
    /// # Errors
    ///
    /// `target` exists or cannot be created, or the tree's checkpoint fails.
    pub fn capture_partition(&self, partition: Partition, target: &Path) -> StorageResult<()> {
        claim_checkpoint_dir(target)?;
        let tree = self.tree(partition)?;
        tree.flush_active_memtable(0)?;
        tree.create_checkpoint(&target.join(partition.name()))
            .map_err(|e| {
                StorageError::Io(format!("capture partition {}: {e}", partition.name()))
            })?;
        Ok(())
    }

    /// The replicated rows of `partition` in this engine: every key but the
    /// node-local ones ([`is_node_local`]) and the coverage records.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn replicated_rows(&self, partition: Partition) -> StorageResult<Rows> {
        let snapshot = self.snapshot();
        Ok(self
            .snapshot_prefix_scan(&snapshot, partition, &[])?
            .into_iter()
            .filter(|(key, _)| !is_node_local(partition, key))
            .map(|(key, value)| (key, value.to_vec()))
            .collect())
    }

    /// The replicated rows of `partition` in the checkpoint at
    /// `checkpoint_dir`, with which Raft entries the checkpoint's tree holds.
    ///
    /// # Errors
    ///
    /// The checkpoint cannot be opened or read.
    pub fn checkpoint_raft_partition(
        checkpoint_dir: &Path,
        partition: Partition,
    ) -> StorageResult<(Rows, RaftHeld)> {
        let ckpt = Self::open_checkpoint(checkpoint_dir)?;
        let rows = ckpt.replicated_rows(partition)?;
        let held = RaftHeld(TreeCoverage::read(ckpt.tree(partition)?, Domain::Raft)?);
        Ok((rows, held))
    }

    /// Replace `partition`'s replicated rows with `rows`, keeping its
    /// node-local ones, under a rebuild intent. On a Raft store the tree
    /// carries no Raft record until [`Self::finish_raft_rebuild`]; entries
    /// applied in between (with [`Self::apply_raft_proposal`]) record their
    /// markers. Call with the applies paused.
    ///
    /// # Errors
    ///
    /// `partition` is the node-local Raft partition, the node-local rows
    /// cannot be read, or a write fails.
    pub fn begin_partition_rebuild(&self, partition: Partition, rows: &Rows) -> StorageResult<()> {
        if partition == Partition::Raft {
            return Err(StorageError::InvalidConfig(
                "the Raft partition holds this node's vote and log metadata; \
                 it is never rebuilt from another copy"
                    .into(),
            ));
        }
        // Read before anything is cleared: a failure leaves the tree as it was.
        let snapshot = self.snapshot();
        let keep: Rows = self
            .snapshot_prefix_scan(&snapshot, partition, &[])?
            .into_iter()
            .filter(|(key, _)| is_node_local(partition, key))
            .map(|(key, value)| (key, value.to_vec()))
            .collect();
        self.begin_rebuild(partition)?;
        self.clear_partition(partition)?;
        for (key, value) in keep.iter().chain(rows) {
            self.put(partition, key, value)?;
        }
        Ok(())
    }

    /// Record that the rebuilt `partition` holds every entry below `next`,
    /// persist it, and drop the rebuild intent. `payload` is the log id of
    /// entry `next - 1`.
    ///
    /// # Errors
    ///
    /// A flush or intent removal failure.
    pub fn finish_raft_rebuild(
        &self,
        partition: Partition,
        next: u64,
        payload: &[u8],
    ) -> StorageResult<()> {
        let tree = self.tree(partition)?;
        // Written after the rebuilt data, so a persisted record implies the
        // data it covers is persisted.
        coverage::write_fold(tree, Domain::Raft, 0, next, payload, self.next_seqno());
        self.finish_rebuild(partition)
    }

    /// Copy `partition` for another node: its replicated rows and, when a
    /// Raft state machine runs over the engine, the log position they stand
    /// at. The position is exact: the partition is captured with the applies
    /// paused, for the time a flush and a hard-link checkpoint take, and
    /// read from the capture after they resume.
    ///
    /// Blocks; call it off the async runtime.
    ///
    /// # Errors
    ///
    /// `partition` is the node-local Raft partition, or the capture or a
    /// read fails.
    pub fn copy_partition(&self, partition: Partition) -> StorageResult<PartitionCopy> {
        if partition == Partition::Raft {
            return Err(StorageError::InvalidConfig(
                "the Raft partition holds this node's vote and log metadata; \
                 it is never copied to another node"
                    .into(),
            ));
        }
        let Some(fence) = self.raft_fence() else {
            return Ok(PartitionCopy {
                rows: self.replicated_rows(partition)?,
                position: None,
            });
        };
        let dir = self.data_dir.join(PARTITION_CAPTURE_DIR).join(format!(
            "{}-{:020}",
            partition.name(),
            self.partition_captures
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        if let Some(parent) = dir.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| StorageError::Io(format!("create {parent:?}: {e}")))?;
        }
        let mut position = None;
        let copied = fence
            .with_applies_paused(&mut |applies, payload| {
                self.capture_partition(partition, &dir)?;
                position = Some(RaftPosition {
                    next: applies.next(),
                    payload: payload.to_vec(),
                });
                Ok(())
            })
            .and_then(|()| Self::open_checkpoint(&dir)?.replicated_rows(partition));
        let removed = std::fs::remove_dir_all(&dir);
        let rows = copied?;
        removed.map_err(|e| StorageError::Io(format!("remove capture {dir:?}: {e}")))?;
        Ok(PartitionCopy { rows, position })
    }

    /// Install `copy`, taken from a peer: the whole of `partition` when
    /// `whole`, else a key range of it. With a Raft state machine over the
    /// engine only a whole partition installs, and the copy must stand at or
    /// past this node's applies; the partition then holds every entry below
    /// the copy's position, and the applies leave it alone until they pass
    /// it. Without Raft a whole copy replaces the partition and a range is
    /// written over it.
    ///
    /// Blocks; call it off the async runtime.
    ///
    /// # Errors
    ///
    /// [`StorageError::PositionBehind`] when the copy stands behind the
    /// applies; a range, or a copy without a position, for a Raft store; the
    /// errors of [`Self::begin_partition_rebuild`] and
    /// [`Self::finish_raft_rebuild`].
    pub fn install_partition(
        &self,
        partition: Partition,
        copy: &PartitionCopy,
        whole: bool,
    ) -> StorageResult<()> {
        let Some(fence) = self.raft_fence() else {
            if whole {
                self.begin_partition_rebuild(partition, &copy.rows)?;
                return self.finish_rebuild(partition);
            }
            for (key, value) in &copy.rows {
                self.put(partition, key, value)?;
            }
            return Ok(());
        };
        if !whole {
            return Err(StorageError::InvalidConfig(format!(
                "a key range of {} cannot replace part of a Raft-replicated partition: \
                 the partition records the log position of its whole content",
                partition.name()
            )));
        }
        let Some(position) = &copy.position else {
            return Err(StorageError::InvalidConfig(format!(
                "the copy of {} names no Raft log position, so nothing tells which \
                 entries it holds",
                partition.name()
            )));
        };
        fence.with_applies_paused(&mut |applies, _| {
            if position.next < applies.next() {
                return Err(StorageError::PositionBehind {
                    partition: partition.name().to_string(),
                    source_next: position.next,
                    local_next: applies.next(),
                });
            }
            self.install_raft_partition(partition, &copy.rows, position.next, &position.payload)?;
            applies.raise_floor(partition, position.next);
            Ok(())
        })
    }

    /// Replace `partition` with `rows`, a peer's copy holding every entry
    /// below `next`. Call with the applies paused and at or behind `next`.
    fn install_raft_partition(
        &self,
        partition: Partition,
        rows: &Rows,
        next: u64,
        payload: &[u8],
    ) -> StorageResult<()> {
        self.begin_partition_rebuild(partition, rows)?;
        self.finish_raft_rebuild(partition, next, payload)
    }

    /// Record, durably, that every tree holds the entries below `next`: what
    /// a freshly created store and a freshly installed snapshot both mean.
    /// The markers below `next` are removed.
    ///
    /// The tombstone stops at `next`. Entries applied after the reset carry
    /// the leader's commit_ts as their seqno, which on a follower whose clock
    /// runs ahead can sit below this reset's seqno; a tombstone reaching past
    /// `next` would suppress their markers (a range tombstone hides every
    /// covered key with a lower seqno) and a crash would re-apply them.
    ///
    /// # Errors
    ///
    /// A flush failure.
    pub fn reset_raft_coverage(&self, next: u64, payload: &[u8]) -> StorageResult<()> {
        let at = self.next_seqno();
        for tree in self.coordinator.trees().values() {
            coverage::write_fold(tree, Domain::Raft, 0, next, payload, at);
        }
        for tree in self.coordinator.trees().values() {
            tree.flush_active_memtable(0)?;
        }
        Ok(())
    }

    /// The Raft entries every tree holds on disk: every index below the
    /// returned one. Reads the lowest base first and flushes after, so the
    /// base read is on disk by the time this returns. `0` when no tree
    /// carries a record.
    ///
    /// # Errors
    ///
    /// A tree read or flush failure.
    pub fn raft_durable_floor(&self) -> StorageResult<u64> {
        let floor = self
            .raft_coverage()?
            .resume_point()
            .map_or(0, |(next, _)| next);
        for tree in self.coordinator.trees().values() {
            tree.flush_active_memtable(0)?;
        }
        // A rebuild from the latest checkpoint replays the log from what its
        // trees lack, so the log keeps that too.
        Ok(floor.min(
            self.raft_log_keep_from
                .load(std::sync::atomic::Ordering::Acquire),
        ))
    }

    /// Keep the Raft log from `index` on for a rebuild from the latest local
    /// checkpoint ([`Self::checkpoint_raft_floor`]); `u64::MAX` when none
    /// needs it.
    pub fn set_raft_log_keep_from(&self, index: u64) {
        self.raft_log_keep_from
            .store(index, std::sync::atomic::Ordering::Release);
    }

    /// The lowest Raft log index some tree of the checkpoint at
    /// `checkpoint_dir` may lack: the lowest of its trees' bases.
    ///
    /// # Errors
    ///
    /// The checkpoint cannot be opened or read.
    pub fn checkpoint_raft_floor(checkpoint_dir: &Path) -> StorageResult<u64> {
        let ckpt = Self::open_checkpoint(checkpoint_dir)?;
        let mut floor = u64::MAX;
        for tree in ckpt.coordinator.trees().values() {
            let held = TreeCoverage::read(tree, Domain::Raft)?;
            floor = floor.min(held.base().map_or(0, |(next, _)| next));
        }
        Ok(if floor == u64::MAX { 0 } else { floor })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
