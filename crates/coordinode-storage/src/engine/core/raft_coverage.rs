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

use super::{Deployment, PartitionRepair, Rows, StorageEngine, claim_checkpoint_dir};
use crate::engine::MAX_DERIVED_EFFECTS;
use crate::engine::coverage::{self, Domain, Mark, TreeCoverage};
use crate::engine::partition::Partition;
use crate::engine::scope::{NamespaceId, TreeId};
use crate::error::{StorageError, StorageResult};

/// The seqno a replaced partition's rows are written at: below every commit
/// timestamp, so each entry applied after the replacement wins over the row
/// it changes. The cleared tree holds no other version for the row to hide.
pub const INSTALLED_ROW_SEQNO: lsm_tree::SeqNo = 0;

/// Directory under the data dir where partitions are captured for a copy to
/// another node. Nothing under it outlives the copy; the engine clears what
/// a crash left behind when it opens.
pub(super) const PARTITION_CAPTURE_DIR: &str = "partition-capture";

/// Where this node's Raft applies stand: one past the last applied entry,
/// and, per tree, the entry below which the tree already holds everything
/// because it was installed from a peer further along.
///
/// The Raft state machine owns it and changes it only while applying;
/// anything else reads or changes it through a [`RaftApplyFence`], with the
/// applies paused.
#[derive(Debug, Default)]
pub struct RaftApplyState {
    next: u64,
    /// The trees ahead of the applies, each with its floor. Empty but for
    /// the time between a tree's install from a peer and the applies passing
    /// it, so the check every applied mutation makes is a scan of nothing.
    floors: Vec<(TreeId, u64)>,
}

impl RaftApplyState {
    /// Applies standing at `next`, with no tree ahead of them.
    #[must_use]
    pub fn new(next: u64) -> Self {
        Self {
            next,
            floors: Vec::new(),
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

    /// Every tree now holds exactly the entries below `next`: a snapshot
    /// install.
    pub fn reset(&mut self, next: u64) {
        *self = Self::new(next);
    }

    /// Whether `tree` already holds entry `index`, because it was installed
    /// at a later position than the applies had reached.
    #[must_use]
    pub fn skips(&self, tree: TreeId, index: u64) -> bool {
        index < self.floor(tree)
    }

    /// The entry below which `tree` holds everything; `0` when it is not
    /// ahead of the applies.
    #[must_use]
    pub fn floor(&self, tree: TreeId) -> u64 {
        self.floors
            .iter()
            .find(|(id, _)| *id == tree)
            .map_or(0, |(_, floor)| *floor)
    }

    /// `tree` now holds every entry below `next`.
    pub fn raise_floor(&mut self, tree: TreeId, next: u64) {
        match self.floors.iter_mut().find(|(id, _)| *id == tree) {
            Some((_, floor)) => *floor = (*floor).max(next),
            None => self.floors.push((tree, next)),
        }
    }
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
/// Raft partition (vote, log metadata), in the deployment's records the
/// per-node routing and records (`meta:`) and the state machine's own
/// (`raft:`), and in a namespace's Schema tree this node's records about the
/// namespace's indexes (`meta:`). They never travel between nodes and
/// survive a partition's replacement.
#[must_use]
pub fn is_node_local(partition: Partition, key: &[u8]) -> bool {
    match partition {
        Partition::Raft => true,
        _ => node_local_prefixes(partition)
            .iter()
            .any(|prefix| key.starts_with(prefix)),
    }
}

/// The key prefixes of `partition`'s node-local rows, outside the Raft
/// partition (which is node-local whole). Records the group's state machine
/// writes are not among them ([`coordinode_core::group::GROUP_RECORD_PREFIX`]).
pub(crate) fn node_local_prefixes(partition: Partition) -> &'static [&'static [u8]] {
    match partition {
        Partition::Deployment => &[b"meta:", b"raft:"],
        Partition::Schema => &[b"meta:"],
        _ => &[],
    }
}

/// The tree mutation `m` writes, under the namespace marker `namespace`:
/// a deployment partition's tree is the system set's, any other the
/// namespace's. `None` for a marker, which writes nothing and is kept
/// whatever is skipped, and for a namespace partition's write no marker
/// names, which no tree skips.
fn tree_of(m: &Mutation, namespace: Option<NamespaceId>) -> Option<TreeId> {
    let part = match m {
        Mutation::Put { partition, .. }
        | Mutation::Delete { partition, .. }
        | Mutation::Merge { partition, .. }
        | Mutation::RemoveRange { partition, .. } => Partition::from(*partition),
        // A command's effects are records of the partition it names.
        Mutation::Command(command) => Partition::from(command.partition()),
        Mutation::Operation(_) => Partition::Deployment,
        // Resolved before the apply; its entries are index records.
        Mutation::Derive(_) => Partition::Idx,
        // Writes nothing; it goes with the node rows it describes.
        Mutation::Effects(_) => Partition::Node,
        // Authorized and left out before the unit applies.
        Mutation::Namespace(_) | Mutation::Actor(_) => return None,
    };
    if part.is_deployment() {
        Some(TreeId::system(part))
    } else {
        namespace.map(|ns| TreeId::namespace(ns, part))
    }
}

/// What every tree records of the Raft entries it holds, read when the
/// state machine opens.
#[derive(Debug)]
pub struct RaftCoverage {
    trees: HashMap<TreeId, TreeCoverage>,
}

impl RaftCoverage {
    /// Whether any tree carries a Raft coverage record. A store that applied
    /// Raft entries without one was written by a release that predates apply
    /// coverage.
    #[must_use]
    pub fn has_record(&self) -> bool {
        self.trees.values().any(TreeCoverage::has_record)
    }

    /// Whether proposal `sub` of log entry `index` is physically in `tree`.
    #[must_use]
    pub fn holds(&self, tree: TreeId, index: u64, sub: u32) -> bool {
        self.trees
            .get(&tree)
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

    /// One past the entries `tree` holds all of, `None` when it records no
    /// base.
    #[must_use]
    pub fn base_of(&self, tree: TreeId) -> Option<u64> {
        self.trees
            .get(&tree)
            .and_then(TreeCoverage::base)
            .map(|(next, _)| next)
    }

    /// Every tree the record was read from.
    pub fn trees(&self) -> impl Iterator<Item = TreeId> + '_ {
        self.trees.keys().copied()
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

impl Deployment {
    /// Read the Raft coverage record of every partition tree.
    ///
    /// # Errors
    ///
    /// A tree read failure, or a malformed record.
    pub fn raft_coverage(&self) -> StorageResult<RaftCoverage> {
        let open = self.coordinator.trees();
        let mut trees = HashMap::with_capacity(open.len());
        for (id, tree) in &open {
            trees.insert(*id, TreeCoverage::read(tree, Domain::Raft)?);
        }
        Ok(RaftCoverage { trees })
    }

    /// Apply proposal `sub` of log entry `index` at `commit_ts`, with its
    /// coverage marker, into every tree the proposal touches except those
    /// `skip` says already hold it. Returns how many mutations landed.
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
        skip: impl Fn(TreeId) -> bool,
    ) -> StorageResult<usize> {
        let mark = Mark {
            domain: Domain::Raft,
            index,
            sub,
        };
        let at = coordinode_core::txn::operation::LogPosition { index, sub };
        // A unit acting for a principal is checked against the grants the
        // log has produced up to it, the check and the apply holding off
        // grant changes together. A refused unit applies only the mark that
        // tells raw readers of the log, and its proposer, why.
        let _access = self.access_guard(mutations);
        let refusal_mark;
        let (actor, mutations) =
            match crate::engine::access::check_unit(self, mutations)?.applying() {
                Ok(applying) => applying,
                Err(refusal) => {
                    refusal_mark = [Mutation::Put {
                        partition: coordinode_core::txn::proposal::PartitionId::Deployment,
                        key: crate::engine::operations::skipped_unit_key(at),
                        value: refusal.encode(),
                    }];
                    (None, &refusal_mark[..])
                }
            };
        // A unit bound to a client operation is decided first, holding the
        // issuer's lock through the apply: the next decision about the
        // operation reads this one's record. A unit its binding refuses
        // applies only the mark that tells raw readers of the log so.
        let mut _decision = None;
        let decided;
        let mutations = match crate::engine::operations::unit_command(mutations)? {
            None => mutations,
            Some((command, rest)) => {
                _decision = Some(self.operation_decisions.lock(&command.issuer()));
                let outcome =
                    crate::engine::operations::decide(self, command, rest, commit_ts, at, actor)?;
                decided = if outcome.skipped {
                    // A refusal for the issuer's ownership tells its proposer
                    // why, as a refusal for a missing right does.
                    vec![Mutation::Put {
                        partition: coordinode_core::txn::proposal::PartitionId::Deployment,
                        key: crate::engine::operations::skipped_unit_key(at),
                        value: outcome
                            .refused
                            .as_ref()
                            .map(coordinode_core::access::AccessRefusal::encode)
                            .unwrap_or_default(),
                    }]
                } else {
                    outcome.effects
                };
                decided.as_slice()
            }
        };
        // Derive before choosing partitions: DERIVED entries are read from
        // the unit's own data effects, which a partition that already holds
        // the entry skips. A data tree ahead of the index tree must not cost
        // the index its entries.
        let resolved =
            coordinode_core::index::derive::resolve_unit(mutations, MAX_DERIVED_EFFECTS)?;
        let mutations = resolved.as_ref();
        // Which mutations a tree that already holds the entry takes none of,
        // each tree named by the namespace marker before the mutation.
        let skipped: Vec<bool> = {
            let mut namespace = None;
            mutations
                .iter()
                .map(|m| {
                    if let Mutation::Namespace(id) = m {
                        namespace = Some(*id);
                    }
                    tree_of(m, namespace).is_some_and(&skip)
                })
                .collect()
        };
        let writes = |ms: &[Mutation]| {
            ms.iter()
                .filter(|m| !matches!(m, Mutation::Namespace(_)))
                .count()
        };
        // A committed entry: a read that sees any of its writes is not ahead
        // of the ceiling, which is raised before the first lands.
        self.note_raft_effects(commit_ts);
        if skipped.iter().any(|&s| s) {
            // A tree that already holds the entry takes none of it.
            let kept: Vec<Mutation> = mutations
                .iter()
                .zip(&skipped)
                .filter(|(_, skipped)| !**skipped)
                .map(|(m, _)| m.clone())
                .collect();
            let landed = writes(&kept);
            let staged = self.applied_feed.stage(index, commit_ts, &kept);
            if landed > 0 {
                self.apply_proposal_covered(&kept, commit_ts, Some(mark), actor)?;
            }
            staged.publish();
            self.note_raft_applied(index, commit_ts);
            Ok(landed)
        } else {
            // Staged before the write lands: a reader that sees it is told
            // the derived indexes may not hold it yet.
            let staged = self.applied_feed.stage(index, commit_ts, mutations);
            self.apply_proposal_covered(mutations, commit_ts, Some(mark), actor)?;
            #[cfg(test)]
            self.applied_feed.pause_after_write();
            staged.publish();
            self.note_raft_applied(index, commit_ts);
            Ok(writes(mutations))
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
        ahead: impl Fn(TreeId) -> bool,
    ) {
        // Above every marker below `next`: those proposals were applied, and
        // the oracle advanced past their commit_ts, before this call.
        let at = self.next_seqno();
        for (id, tree) in &self.coordinator.trees() {
            if !ahead(*id) {
                coverage::write_fold(tree, Domain::Raft, from, next, payload, at);
            }
        }
        // A fold can land in a memtable a rotation just emptied.
        self.coordinator.flush_trigger().wrote_unmeasured();
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
}

impl StorageEngine {
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
        self.flush_memtables(tree)?;
        // The tree lies in its set's directory, made here: the tree's
        // checkpoint makes only its own.
        let tree_dir = target.join(self.tree_id(partition).dir());
        if let Some(set) = tree_dir.parent() {
            std::fs::create_dir_all(set).map_err(|e| {
                StorageError::Io(format!("capture partition {}: {e}", partition.name()))
            })?;
        }
        tree.create_checkpoint(&tree_dir).map_err(|e| {
            StorageError::Io(format!("capture partition {}: {e}", partition.name()))
        })?;
        // The capture is opened as an engine directory, which names its
        // format like any other.
        crate::format::write_marker(
            target,
            coordinode_core::version::engine_format_version(),
            &self.footprints,
        )
    }

    /// The capture at `dir` of this engine's tree of `partition`
    /// ([`Self::capture_partition`]), opened as the tree's set: this
    /// engine's namespace, or the deployment's own.
    ///
    /// # Errors
    ///
    /// The capture cannot be opened.
    pub(crate) fn open_capture(&self, dir: &Path, partition: Partition) -> StorageResult<Self> {
        let capture = Deployment::open_checkpoint(dir)?;
        if partition.is_deployment() {
            Ok(capture.member_engine())
        } else {
            Self::bind(capture, self.namespace())
        }
    }

    /// The replicated rows of `partition` in this engine: every key but the
    /// node-local ones ([`is_node_local`]), and, of the reserved records,
    /// only the deployment tree's closure ledger, read at the same cut: the
    /// position authority a copy of the same history carries. Coverage and
    /// installation records keep their own rules. Each value is as stored,
    /// its revision stamp included, so a copy keeps every record's revision.
    ///
    /// # Errors
    ///
    /// A read failure.
    pub fn replicated_rows(&self, partition: Partition) -> StorageResult<Rows> {
        let snapshot = self.snapshot();
        let mut rows: Rows = self
            .snapshot_prefix_scan_stored(&snapshot, partition, &[])?
            .into_iter()
            .filter(|(key, _)| !is_node_local(partition, key))
            .collect();
        if partition == Partition::Deployment {
            rows.extend(self.closure_rows_at(snapshot)?);
        }
        Ok(rows)
    }

    /// The replicated rows of `partition`, of this engine's namespace or the
    /// deployment's own, in the checkpoint at `checkpoint_dir`, with which
    /// Raft entries the checkpoint's tree holds.
    ///
    /// # Errors
    ///
    /// The checkpoint cannot be opened or read, or holds no such namespace
    /// ([`StorageError::UnknownNamespace`]).
    pub fn checkpoint_raft_partition(
        &self,
        checkpoint_dir: &Path,
        partition: Partition,
    ) -> StorageResult<(Rows, RaftHeld)> {
        let checkpoint = Deployment::open_checkpoint(checkpoint_dir)?;
        let ckpt = if partition.is_deployment() {
            checkpoint.member_engine()
        } else {
            checkpoint.engine(self.namespace())?
        };
        let rows = ckpt.replicated_rows(partition)?;
        let held = RaftHeld(TreeCoverage::read(ckpt.tree(partition)?, Domain::Raft)?);
        Ok((rows, held))
    }

    /// Replace `partition`'s replicated rows with `rows`, keeping its
    /// node-local ones, under a rebuild intent. On a Raft store the tree
    /// carries no Raft record until [`Self::finish_raft_rebuild`]; entries
    /// applied in between (with [`Deployment::apply_raft_proposal`]) record their
    /// markers. Call with the applies paused.
    ///
    /// The rows land at seqno `INSTALLED_ROW_SEQNO` (0), below every commit
    /// timestamp: the entries replayed or applied after the install write at
    /// their own commit timestamp, which can sit below this node's next seqno,
    /// and an installed row above them would hide them.
    ///
    /// # Errors
    ///
    /// `partition` is the node-local Raft partition, the node-local rows
    /// cannot be read, or a write fails.
    pub fn begin_partition_rebuild(
        &self,
        partition: Partition,
        rows: &Rows,
    ) -> StorageResult<PartitionRepair<'_>> {
        if partition == Partition::Raft {
            return Err(StorageError::InvalidConfig(
                "the Raft partition holds this node's vote and log metadata; \
                 it is never rebuilt from another copy"
                    .into(),
            ));
        }
        // The repair is taken before anything changes: a second repair of the
        // partition is refused with the index transfer status and the tree
        // as they were.
        let id = self.tree_id(partition);
        let repair = self.enter_repair(id)?;
        // Read before anything is cleared, under the repair: no other repair
        // moves them meanwhile, and a failure leaves the tree as it was. Only
        // their prefixes are read: a partition repaired because a table of it
        // is corrupt would otherwise fail on that table before the repair
        // could drop it.
        let snapshot = self.snapshot();
        // As stored, revisions included: written back as they were.
        let mut keep: Rows = Vec::new();
        for prefix in node_local_prefixes(partition) {
            keep.extend(self.snapshot_prefix_scan_stored(&snapshot, partition, prefix)?);
        }
        // Durable with the intent, before the clear: a crash after it leaves
        // them in the intent, not only in a memtable.
        self.record_kept(id, &keep)?;
        // The copies the index tree receives are qualified only by the
        // inventory the transfer brings, taken once they are in place.
        if partition == Partition::Idx {
            self.begin_index_transfer()?;
        }
        self.clear_partition_tree(partition, self.tree(partition)?)?;
        for (key, value) in &keep {
            self.put_stored(partition, key, value)?;
        }
        // The rows are addressed as the application addresses them; in the
        // index tree each generation's rows go to this member's installation.
        let tree = self.tree(partition)?;
        let mut addressing = crate::engine::installation::Addressing::new(
            self.installations_of(partition),
            tree,
            self.next_seqno(),
        );
        // A deployment partition's rows land in the system set's tree whatever
        // set is named; a member's engine bound to no namespace names that.
        let set = self.batch_set();
        for (key, value) in rows {
            self.coordinator.put_at(
                &set,
                partition,
                addressing.point(key)?,
                value,
                INSTALLED_ROW_SEQNO,
            )?;
        }
        Ok(repair)
    }

    /// Replace each partition of `partitions` with its rows from a Raft
    /// snapshot standing at `next` (`payload` is the log id of entry
    /// `next - 1`), and bind every tree to that position, or to its own in
    /// `ahead` for a partition the snapshot holds further along. The rows of
    /// one partition at a time are read from `rows_of`, so memory holds one
    /// partition, not the snapshot. Every replaced partition is under a
    /// rebuild intent from before its clear until every tree's record is on
    /// disk, so a crash in between leaves an intent, never a tree that claims
    /// entries it lost. Call with the applies paused.
    ///
    /// # Errors
    ///
    /// The errors of `rows_of`, [`Self::begin_partition_rebuild`] and
    /// [`Deployment::reset_raft_coverage`], a flush failure, or an intent removal
    /// failure.
    pub fn install_raft_image(
        &self,
        partitions: &[Partition],
        mut rows_of: impl FnMut(Partition) -> StorageResult<Rows>,
        next: u64,
        payload: &[u8],
        ahead: &[(TreeId, RaftPosition)],
        index_inventory: Option<&[u8]>,
    ) -> StorageResult<()> {
        // Every partition's repair is held until its tree is installed and
        // durable; a failure drops them, leaving the partitions owed.
        let mut repairs = Vec::with_capacity(partitions.len());
        for &partition in partitions {
            let rows = rows_of(partition)?;
            repairs.push(self.begin_partition_rebuild(partition, &rows)?);
        }
        self.reset_raft_coverage(next, payload)?;
        // Written over the reset's base at a later seqno, and durable before
        // the intents go: the applies leave such a tree alone until they
        // pass its position, which the caller raises as its floor.
        for (id, position) in ahead {
            let replaced = partitions
                .iter()
                .any(|&partition| self.tree_id(partition) == *id);
            if position.next > next && replaced {
                let tree = self.tree(id.part)?;
                coverage::write_fold(
                    tree,
                    Domain::Raft,
                    0,
                    position.next,
                    &position.payload,
                    self.next_seqno(),
                );
                self.flush_memtables(tree)?;
            }
        }
        // The index's inventory is taken, and the Schema tree's ledger
        // installed, while its repair still holds it.
        for repair in repairs {
            let partition = repair.partition();
            self.finish_rebuild_with(repair, || {
                self.establish_replaced(partition, index_inventory)
            })?;
        }
        Ok(())
    }

    /// Record that the partition `repair` rebuilt holds every entry below
    /// `next`, persist it, drop the rebuild intent and end the repair.
    /// `payload` is the log id of entry `next - 1`.
    ///
    /// # Errors
    ///
    /// A flush or intent removal failure.
    pub fn finish_raft_rebuild(
        &self,
        repair: PartitionRepair<'_>,
        next: u64,
        payload: &[u8],
    ) -> StorageResult<()> {
        // Another store's repair is refused before anything changes here.
        self.check_repair(&repair)?;
        let partition = repair.partition();
        let tree = self.tree(partition)?;
        // Written after the rebuilt data, so a persisted record implies the
        // data it covers is persisted.
        coverage::write_fold(tree, Domain::Raft, 0, next, payload, self.next_seqno());
        // A partition copy carries no inventory: every index copy in it is
        // unknown until checked here, under the repair, so no next repair's
        // transfer is taken for this one's. The Schema tree's ledger is
        // installed under the repair too.
        self.finish_rebuild_with(repair, || self.establish_replaced(partition, None))?;
        // The partition was replaced wholesale, from a peer's copy or a
        // checkpoint: followers of the applies read it afresh.
        self.applied_feed.replaced(Some(self.tree_id(partition)));
        Ok(())
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
            .and_then(|()| {
                self.open_capture(&dir, partition)?
                    .replicated_rows(partition)
            });
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
                let repair = self.begin_partition_rebuild(partition, &copy.rows)?;
                return self
                    .finish_rebuild_with(repair, || self.establish_replaced(partition, None));
            }
            // The rows as the peer stores them: each keeps its revision.
            for (key, value) in &copy.rows {
                self.put_stored(partition, key, value)?;
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
            applies.raise_floor(self.tree_id(partition), position.next);
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
        let repair = self.begin_partition_rebuild(partition, rows)?;
        self.finish_raft_rebuild(repair, next, payload)
    }
}

impl Deployment {
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
        for (_, tree) in self.coordinator.trees() {
            coverage::write_fold(&tree, Domain::Raft, 0, next, payload, at);
        }
        self.flush_partition_trees()?;
        // A snapshot now stands in the store in place of whatever the
        // applies had built: followers of the applies read it afresh.
        self.applied_feed.replaced(None);
        self.note_field_dictionary_change();
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
        self.flush_partition_trees()?;
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
        for (_, tree) in ckpt.coordinator.trees() {
            let held = TreeCoverage::read(&tree, Domain::Raft)?;
            floor = floor.min(held.base().map_or(0, |(next, _)| next));
        }
        Ok(if floor == u64::MAX { 0 } else { floor })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
