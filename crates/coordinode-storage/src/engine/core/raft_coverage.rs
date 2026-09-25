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

use coordinode_core::txn::proposal::Mutation;
use lsm_tree::AbstractTree;

use super::StorageEngine;
use crate::engine::coverage::{self, Domain, Mark, TreeCoverage};
use crate::engine::partition::Partition;
use crate::error::StorageResult;

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
    /// comes back from [`RaftCoverage::resume_point`].
    pub fn fold_raft_coverage(&self, from: u64, next: u64, payload: &[u8]) {
        // Above every marker below `next`: those proposals were applied, and
        // the oracle advanced past their commit_ts, before this call.
        let at = self.next_seqno();
        for tree in self.coordinator.trees().values() {
            coverage::write_fold(tree, Domain::Raft, from, next, payload, at);
        }
    }

    /// Record, durably, that every tree holds exactly the entries below
    /// `next` and nothing above: what a freshly created store and a freshly
    /// installed snapshot both mean. Every marker is removed.
    ///
    /// # Errors
    ///
    /// A flush failure.
    pub fn reset_raft_coverage(&self, next: u64, payload: &[u8]) -> StorageResult<()> {
        let at = self.next_seqno();
        let domain = Domain::Raft;
        for tree in self.coordinator.trees().values() {
            tree.insert(domain.base_key(), coverage::encode_base(next, payload), at);
            tree.remove_range(
                domain.marker_key(0, 0).to_vec(),
                domain.marker_end().to_vec(),
                at,
            );
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
        Ok(floor)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
