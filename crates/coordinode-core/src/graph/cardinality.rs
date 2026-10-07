//! Exact relationship cardinality: the two measures over one incident scope,
//! and the pair transitions that change them.
//!
//! A scope is the edges of one type and direction incident to one node. Its
//! applicable set E is a set of logical edge identities, each reaching one
//! neighbour: the target for an outgoing scope, the source for an incoming
//! one. `EDGE_INSTANCES` counts the identities, `DISTINCT_NEIGHBORS` the
//! neighbours they reach. Neither is derived from the other, from the number
//! of mutations or from the number of property rows: an ordinary edge with
//! no properties is an instance, and three parallel edges to one neighbour
//! are three instances and one neighbour.
//!
//! Every writer of an edge goes through [`PairInstances`], which keeps the
//! identities of one ordered pair and reports what an effect changed: how
//! many identities joined or left, and whether the pair became adjacent or
//! stopped being. The same report drives the adjacency projection and every
//! maintained count, so they cannot disagree.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::graph::node::NodeId;
pub use crate::txn::invariant::{CardinalityBound, CardinalityMeasure, Direction};

/// The key that tells the instances of one ordered pair apart: the encoded
/// discriminator for a discriminated type, empty for a single-edge type.
pub type InstanceKey = Vec<u8>;

/// One logical edge identity in an incident scope, as the scope's node sees
/// it: the neighbour the edge reaches and the key that identifies it among
/// the edges between the two.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IncidentInstance {
    /// The opposite endpoint.
    pub neighbour: NodeId,
    /// The identity within the pair.
    pub key: InstanceKey,
}

/// Both measures of one incident scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IncidentCount {
    /// Distinct logical edge identities.
    pub instances: u64,
    /// Distinct neighbours those identities reach.
    pub neighbours: u64,
}

impl IncidentCount {
    /// The count `measure` reads.
    pub fn of(self, measure: CardinalityMeasure) -> u64 {
        match measure {
            CardinalityMeasure::EdgeInstances => self.instances,
            CardinalityMeasure::DistinctNeighbours => self.neighbours,
        }
    }

    /// Count the applicable set E. Repeated identities, as a duplicate
    /// binding or a replayed effect produces them, count once.
    pub fn of_set(instances: impl IntoIterator<Item = IncidentInstance>) -> Self {
        let identities: BTreeSet<IncidentInstance> = instances.into_iter().collect();
        let mut neighbours = 0u64;
        let mut last: Option<NodeId> = None;
        // Sorted by neighbour first, so each neighbour's identities are
        // adjacent and a change of neighbour is a new one.
        for instance in &identities {
            if last != Some(instance.neighbour) {
                neighbours += 1;
                last = Some(instance.neighbour);
            }
        }
        Self {
            // A set held in memory cannot hold more members than a u64
            // counts.
            instances: identities.len() as u64,
            neighbours,
        }
    }
}

/// How an effect changed whether an ordered pair is adjacent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Membership {
    /// The pair had no instance and now has one: the neighbour joins both
    /// endpoints' scopes.
    Joined,
    /// The pair lost its last instance: the neighbour leaves both scopes.
    Left,
    /// The pair was adjacent before and is after, or was not and is not.
    Unchanged,
}

/// What one effect changed in an ordered pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairTransition {
    /// Identities that joined.
    pub added: u64,
    /// Identities that left.
    pub removed: u64,
    /// Whether the pair's adjacency changed.
    pub membership: Membership,
}

impl PairTransition {
    const NONE: Self = Self {
        added: 0,
        removed: 0,
        membership: Membership::Unchanged,
    };

    /// The change in `measure` this transition makes to the scope of either
    /// endpoint: a pair is counted once on each side, by the same rule.
    pub fn delta(self, measure: CardinalityMeasure) -> i64 {
        match measure {
            // Bounded by the instances of one pair, which a u64 count in
            // memory holds.
            CardinalityMeasure::EdgeInstances => self.added as i64 - self.removed as i64,
            CardinalityMeasure::DistinctNeighbours => match self.membership {
                Membership::Joined => 1,
                Membership::Left => -1,
                Membership::Unchanged => 0,
            },
        }
    }
}

/// An instance written under a key the destination pair already holds: a
/// redirect may not merge two identities into one.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the destination pair already holds an instance with this identity")]
pub struct IdentityCollision {
    /// The identity both pairs hold.
    pub key: InstanceKey,
}

/// The logical edge identities of one ordered pair, and the transitions
/// every edge effect goes through.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PairInstances {
    keys: BTreeSet<InstanceKey>,
}

impl PairInstances {
    /// A pair holding `keys`.
    pub fn from_keys(keys: impl IntoIterator<Item = InstanceKey>) -> Self {
        Self {
            keys: keys.into_iter().collect(),
        }
    }

    /// The identities held.
    pub fn keys(&self) -> impl Iterator<Item = &InstanceKey> {
        self.keys.iter()
    }

    /// Whether the pair is adjacent: it holds at least one identity.
    pub fn is_adjacent(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Whether the pair holds `key`.
    pub fn contains(&self, key: &[u8]) -> bool {
        self.keys.contains(key)
    }

    /// Write the instance `key`. Writing an identity the pair already holds
    /// is the same instance updated or replayed, never a second one.
    pub fn upsert(&mut self, key: InstanceKey) -> PairTransition {
        let was_adjacent = self.is_adjacent();
        if !self.keys.insert(key) {
            return PairTransition::NONE;
        }
        PairTransition {
            added: 1,
            removed: 0,
            membership: if was_adjacent {
                Membership::Unchanged
            } else {
                Membership::Joined
            },
        }
    }

    /// Remove the instance `key`. Removing one the pair does not hold
    /// changes nothing.
    pub fn remove(&mut self, key: &[u8]) -> PairTransition {
        if !self.keys.remove(key) {
            return PairTransition::NONE;
        }
        PairTransition {
            added: 0,
            removed: 1,
            membership: if self.is_adjacent() {
                Membership::Unchanged
            } else {
                Membership::Left
            },
        }
    }

    /// Remove every instance.
    pub fn clear(&mut self) -> PairTransition {
        let removed = self.keys.len() as u64;
        self.keys.clear();
        PairTransition {
            added: 0,
            removed,
            membership: if removed > 0 {
                Membership::Left
            } else {
                Membership::Unchanged
            },
        }
    }

    /// Move the instance `key` from this pair to `to`, keeping its identity:
    /// a redirect. The two transitions are the source's and the
    /// destination's. Moving an identity the source does not hold changes
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`IdentityCollision`] when `to` already holds `key`; neither pair is
    /// changed.
    pub fn move_to(
        &mut self,
        to: &mut PairInstances,
        key: &[u8],
    ) -> Result<(PairTransition, PairTransition), IdentityCollision> {
        if !self.contains(key) {
            return Ok((PairTransition::NONE, PairTransition::NONE));
        }
        if to.contains(key) {
            return Err(IdentityCollision { key: key.to_vec() });
        }
        let left = self.remove(key);
        let joined = to.upsert(key.to_vec());
        Ok((left, joined))
    }
}

/// A declared relationship cardinality constraint, as every writer and the
/// commit decision read it: what is counted, over which scope and bound,
/// under which schema generation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CardinalityDescriptor {
    /// The edge type counted.
    pub edge_type: String,
    /// Which side of the constrained node is counted.
    pub direction: Direction,
    /// What is counted.
    pub measure: CardinalityMeasure,
    /// The counts admitted.
    pub bound: CardinalityBound,
    /// The schema generation the constraint was declared under. Maintained
    /// state and claims are bound to it; a count kept under another
    /// generation is not evidence for this one.
    pub schema_generation: u64,
}

impl CardinalityDescriptor {
    /// The neighbour an instance of pair `(source, target)` reaches in the
    /// scope of `node`, if the pair is incident to `node` on this
    /// descriptor's side.
    pub fn neighbour_in_scope(
        &self,
        node: NodeId,
        source: NodeId,
        target: NodeId,
    ) -> Option<NodeId> {
        match self.direction {
            Direction::Outgoing if source == node => Some(target),
            Direction::Incoming if target == node => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
