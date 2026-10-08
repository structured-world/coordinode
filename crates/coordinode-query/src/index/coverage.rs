//! Which committed writes an index maintained from the applied commits does
//! not hold yet.
//!
//! The full-text and vector indexes are maintained from the entries applied
//! to the store, each kind by a worker that follows them. A search never
//! waits for the worker: it answers from the index for every node outside
//! the worker's unfolded writes and evaluates the nodes those writes touched
//! exactly, from the store at its own snapshot.

use coordinode_core::graph::node::NodeId;
use coordinode_modality::StoreError;
use coordinode_storage::engine::applied::{AppliedPosition, PendingKeys};
use coordinode_storage::engine::core::StorageEngine;
use rustc_hash::FxHashSet;

/// The writes an index does not hold yet, as one search sees them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexDelta {
    /// The nodes written since the index's position: the index's entries for
    /// them may be stale or missing, and it answers for every other node.
    Nodes(FxHashSet<NodeId>),
    /// Which nodes changed is not known (writes were dropped from the feed or
    /// the store was replaced, until the rebuild covers that): the index
    /// answers for no node.
    Unknown,
}

impl IndexDelta {
    /// No write is outstanding: the index answers alone.
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Nodes(nodes) if nodes.is_empty())
    }

    /// This delta with `nodes` added: a transaction's own writes, which no
    /// index holds until it commits.
    #[must_use]
    pub fn with_nodes(self, nodes: impl IntoIterator<Item = NodeId>) -> Self {
        match self {
            Self::Nodes(mut set) => {
                set.extend(nodes);
                Self::Nodes(set)
            }
            Self::Unknown => Self::Unknown,
        }
    }

    /// Both deltas: unknown when either is.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        match other {
            Self::Nodes(nodes) => self.with_nodes(nodes),
            Self::Unknown => Self::Unknown,
        }
    }

    /// The nodes of `shard_id` written at or after storage seqno `since`: for
    /// a read at timestamp T (`since` = T + 1), the nodes whose state then an
    /// index of the current state does not hold. The store keeps every
    /// version a readable snapshot can see, so the set is complete for any
    /// readable T.
    ///
    /// # Errors
    ///
    /// The store could not list the changed keys.
    pub fn written_since(
        engine: &StorageEngine,
        shard_id: u16,
        since: u64,
    ) -> Result<Self, StoreError> {
        use coordinode_modality::NodeStore as _;
        Ok(Self::Nodes(
            coordinode_modality::LocalNodeStore
                .written_since(engine, shard_id, since)?
                .into_iter()
                .collect(),
        ))
    }

    /// Whether the index's entry for `node` cannot be trusted.
    pub fn contains(&self, node: NodeId) -> bool {
        match self {
            Self::Nodes(nodes) => nodes.contains(&node),
            Self::Unknown => true,
        }
    }
}

/// The writes one kind of index has not folded, read from the feed its
/// worker follows.
#[derive(Debug)]
pub struct IndexCoverage {
    applied: AppliedPosition,
}

impl IndexCoverage {
    /// Coverage of the worker following `applied`, a retained subscription.
    pub fn new(applied: AppliedPosition) -> Self {
        Self { applied }
    }

    /// Record that every applied event up to `seq` is in the indexes.
    pub fn release(&self, seq: u64) {
        self.applied.release(seq);
    }

    /// The nodes of `shard_id` written by entries applied before this call
    /// that the indexes may not hold yet.
    pub fn delta(&self, shard_id: u16) -> IndexDelta {
        match self.applied.pending() {
            PendingKeys::Unknown => IndexDelta::Unknown,
            PendingKeys::Known(events) => IndexDelta::Nodes(
                events
                    .iter()
                    .flat_map(|keys| keys.iter())
                    .filter_map(|key| coordinode_core::graph::node::decode_written_node(key))
                    .filter(|(shard, _)| *shard == shard_id)
                    .map(|(_, id)| id)
                    .collect(),
            ),
        }
    }
}
