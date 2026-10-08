//! The valid-time interval each node's indexed state holds over.
//!
//! A node with a timeline (a temporal node) has a different state at
//! different instants of valid time, and the state at the present changes as
//! time passes, with no write. An index holds one state per node, the one
//! valid when the node was last folded, together with the half-open interval
//! over which the timeline keeps that state. A read at instant V can use the
//! index's document of a node only when V lies in that interval; for any
//! other node the read evaluates the timeline itself.

use std::collections::{BTreeSet, HashMap};
use std::ops::Bound;

/// A half-open interval of valid time, `[from, until)`; an absent bound is
/// unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Validity {
    /// Where it starts; `None` since forever.
    pub from: Option<i64>,
    /// Where it ends, exclusive; `None` for good.
    pub until: Option<i64>,
}

impl Validity {
    /// Every instant: the state of a node without a timeline.
    pub const ALWAYS: Self = Self {
        from: None,
        until: None,
    };

    /// Whether `at` lies inside.
    pub fn contains(&self, at: i64) -> bool {
        self.from.is_none_or(|from| from <= at) && self.until.is_none_or(|until| at < until)
    }
}

/// The bounded intervals of the nodes an index holds, searchable by bound.
/// A node held for every instant has no entry.
#[derive(Debug, Clone, Default)]
pub(crate) struct Validities {
    by_node: HashMap<u64, Validity>,
    starts: BTreeSet<(i64, u64)>,
    ends: BTreeSet<(i64, u64)>,
}

impl Validities {
    /// Record that `node_id`'s indexed state holds over `validity`.
    pub(crate) fn set(&mut self, node_id: u64, validity: Validity) {
        if let Some(old) = self.by_node.remove(&node_id) {
            if let Some(from) = old.from {
                self.starts.remove(&(from, node_id));
            }
            if let Some(until) = old.until {
                self.ends.remove(&(until, node_id));
            }
        }
        if validity == Validity::ALWAYS {
            return;
        }
        if let Some(from) = validity.from {
            self.starts.insert((from, node_id));
        }
        if let Some(until) = validity.until {
            self.ends.insert((until, node_id));
        }
        self.by_node.insert(node_id, validity);
    }

    /// Forget every node.
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    /// The nodes whose indexed state does not hold at `at`, in ascending id
    /// order.
    pub(crate) fn outside(&self, at: i64) -> Vec<u64> {
        let ended = self.ends.range(..=(at, u64::MAX)).map(|(_, node)| *node);
        let not_begun = self
            .starts
            .range((Bound::Excluded((at, u64::MAX)), Bound::Unbounded))
            .map(|(_, node)| *node);
        let mut nodes: Vec<u64> = ended.chain(not_begun).collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    /// The earliest instant at which some node's indexed state stops
    /// holding.
    pub(crate) fn first_end(&self) -> Option<i64> {
        self.ends.first().map(|(until, _)| *until)
    }
}

#[cfg(test)]
mod tests;
