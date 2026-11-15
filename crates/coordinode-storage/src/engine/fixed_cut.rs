//! Fixed-cut source handoff for one history: a pinned read view below an
//! exclusive bound `B`, the first journal position `L` to replay, and a
//! finite join boundary `H`.
//!
//! Every committed unit before `L` has its commit timestamp below `B`, and
//! every committed unit below `B` is before `H`. The view holds exactly the
//! state below `B`. A consumer copies the view, then reconciles every unit
//! of `[L, H)`: one below `B` is already in the view, one at or above it is
//! applied once. From `H` on it follows the journal as any reader does. A
//! physical journal entry may carry units on both sides of `B`, so the
//! decision is per unit, by the unit's own commit timestamp.

use coordinode_core::txn::operation::LogPosition;
use coordinode_core::txn::proposal::{Mutation, PartitionId};

use crate::engine::coordinator::SnapshotPin;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// The rows a cut hands over: the keys of one partition under a prefix. A
/// cut of rows recovers rows only; it does not stand for the whole store, a
/// columnar table or an index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutScope {
    partition: Partition,
    prefix: Vec<u8>,
}

impl CutScope {
    /// The rows of `partition` whose keys start with `prefix`.
    ///
    /// # Errors
    ///
    /// [`StorageError::CutUnsupported`] for the index and schema partitions,
    /// whose state follows from interpretations a row copy does not carry
    /// (index installations, metadata decisions), and for the Raft
    /// partition, which holds no rows of the history.
    pub fn rows(partition: Partition, prefix: Vec<u8>) -> StorageResult<Self> {
        if matches!(
            partition,
            Partition::Idx | Partition::Schema | Partition::Raft
        ) {
            return Err(StorageError::CutUnsupported {
                reason: format!("the {partition:?} partition is not a row scope"),
            });
        }
        Ok(Self { partition, prefix })
    }

    /// The partition.
    #[must_use]
    pub fn partition(&self) -> Partition {
        self.partition
    }

    /// The key prefix.
    #[must_use]
    pub fn prefix(&self) -> &[u8] {
        &self.prefix
    }

    /// Whether `partition` is this scope's.
    pub(crate) fn holds_partition(&self, partition: PartitionId) -> bool {
        Partition::from(partition) == self.partition
    }

    /// Whether `key` of this scope's partition is in it.
    pub(crate) fn holds_key(&self, key: &[u8]) -> bool {
        key.starts_with(&self.prefix)
    }

    /// Whether some key of `[start, end)` is in it.
    pub(crate) fn meets_range(&self, start: &[u8], end: &[u8]) -> bool {
        self.clip_range(start, end).is_some()
    }

    /// The part of `[start, end)` inside the scope, `None` when nothing of
    /// it is. Keys under the prefix sort from the prefix up to its
    /// successor, or to the end of the key space when it has none.
    pub(crate) fn clip_range(&self, start: &[u8], end: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        let lo = start.max(self.prefix.as_slice());
        let successor = prefix_successor(&self.prefix);
        let hi = match &successor {
            Some(after) => end.min(after.as_slice()),
            None => end,
        };
        (lo < hi).then(|| (lo.to_vec(), hi.to_vec()))
    }
}

/// The least key sorting after every key starting with `prefix`; `None`
/// when no key does (an empty or all-`0xFF` prefix).
pub(crate) fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.pop() {
        if last < 0xFF {
            end.push(last + 1);
            return Some(end);
        }
    }
    None
}

/// A pinned view of one history below `bound`, and where its journal
/// continuation starts and joins. Dropping it releases the view.
pub struct FixedCut {
    pub(crate) first: u64,
    pub(crate) bound: u64,
    pub(crate) join_end: u64,
    pub(crate) history: Option<coordinode_core::group::DeploymentId>,
    pub(crate) scope: CutScope,
    pub(crate) _pin: SnapshotPin,
}

impl std::fmt::Debug for FixedCut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedCut")
            .field("first", &self.first)
            .field("bound", &self.bound)
            .field("join_end", &self.join_end)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl FixedCut {
    /// The rows the cut hands over.
    #[must_use]
    pub fn scope(&self) -> &CutScope {
        &self.scope
    }

    /// `L`: the first journal position the continuation replays.
    #[must_use]
    pub fn first(&self) -> u64 {
        self.first
    }

    /// `B`: the exclusive commit timestamp bound of the view. Reads at this
    /// snapshot see every commit below it and none at or above it.
    #[must_use]
    pub fn bound(&self) -> u64 {
        self.bound
    }

    /// `H`: the end of the join. Units before it are reconciled against the
    /// view; no unit below the bound appears at or after it.
    #[must_use]
    pub fn join_end(&self) -> u64 {
        self.join_end
    }

    /// The history the cut belongs to, the store's deployment: a
    /// continuation read from another history is refused. `None` for a store
    /// that has founded none.
    #[must_use]
    pub fn history(&self) -> Option<coordinode_core::group::DeploymentId> {
        self.history
    }

    /// Whether the unit committed at `commit_ts`, found at `position`, is
    /// already represented by the view.
    #[must_use]
    pub fn in_view(&self, position: LogPosition, commit_ts: u64) -> bool {
        position.index < self.join_end && commit_ts < self.bound
    }
}

/// One committed unit of the journal, as it was recorded: its position, its
/// own commit timestamp and its data changes.
#[derive(Debug, Clone, PartialEq)]
pub struct CutUnit {
    /// Where the unit is: the journal entry and its place in the entry.
    pub position: LogPosition,
    /// The unit's own commit timestamp, not its entry's.
    pub commit_ts: u64,
    /// The unit's data changes; empty when its operation binding refused it.
    pub mutations: Vec<Mutation>,
    /// Whether its operation binding refused it: it took no effect.
    pub refused: bool,
    /// Whether the cut's view already holds it.
    pub in_view: bool,
}

/// A read of a cut's continuation: its units, and the position the next
/// read starts at.
#[derive(Debug, Clone, PartialEq)]
pub struct CutRead {
    /// The units of the entries read, in log order.
    pub units: Vec<CutUnit>,
    /// The position after the last entry read; where it started when the
    /// journal held nothing more yet.
    pub next: u64,
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
