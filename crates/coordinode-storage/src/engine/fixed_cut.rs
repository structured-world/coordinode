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
use coordinode_core::txn::proposal::Mutation;

use crate::engine::coordinator::SnapshotPin;

/// A pinned view of one history below `bound`, and where its journal
/// continuation starts and joins. Dropping it releases the view.
pub struct FixedCut {
    pub(crate) first: u64,
    pub(crate) bound: u64,
    pub(crate) join_end: u64,
    pub(crate) history: u64,
    pub(crate) _pin: SnapshotPin,
}

impl std::fmt::Debug for FixedCut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedCut")
            .field("first", &self.first)
            .field("bound", &self.bound)
            .field("join_end", &self.join_end)
            .finish_non_exhaustive()
    }
}

impl FixedCut {
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

    /// The history the cut belongs to: a continuation read from another
    /// history is refused.
    #[must_use]
    pub fn history(&self) -> u64 {
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

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
