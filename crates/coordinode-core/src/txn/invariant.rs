//! Typed invariant claims: what a mutation needs to stay true, stated so that
//! two attempts can be told apart as compatible or not.
//!
//! Snapshot isolation validates the write set, and that is not enough for a
//! graph. Two attempts can write disjoint keys, pass first-committer-wins and
//! together break a condition each of them checked alone: two edges added
//! under an at-most-one constraint, a node deleted while another attempt
//! attaches to it, the last row of a pair erased while another inserts into
//! it. Atomic publication of the resulting writes proves that they landed
//! together, never that the conditions used to construct them still held.
//!
//! So every mutation states what made its result admissible, as a claim. A
//! claim names the predicate, the logical scope it covers, the schema
//! generation it was evaluated under and the attempt it belongs to. It is not
//! a lock on a physical record: two attempts that reference the same node
//! without destroying it hold compatible claims and neither waits for the
//! other, which is what keeps a high-degree node from serialising every
//! addition to it.
//!
//! This module owns the claims and the rule that decides compatibility. Where
//! that rule is applied, and the atomic transition that checks and reserves
//! before the durable promise, belong to the commit guard.

use alloc::string::String;
use alloc::vec::Vec;

use crate::graph::node::NodeId;

/// Which direction of an edge type a scope covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Edges leaving the node.
    Outgoing,
    /// Edges arriving at the node.
    Incoming,
}

/// What a cardinality bound counts, chosen by the schema and never inferred.
///
/// The two measures answer different questions about the same adjacency, and
/// a claim for one is not evidence for the other: three parallel edges to one
/// neighbour are three instances and one neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CardinalityMeasure {
    /// Every edge instance counts, including parallel edges to one neighbour.
    EdgeInstances,
    /// Each distinct neighbour counts once, however many edges reach it.
    DistinctNeighbours,
}

/// The logical thing a claim covers. Two claims can only conflict if their
/// scopes overlap, so this is what makes independent work independent.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClaimScope {
    /// One node's identity and lifecycle.
    Node(NodeId),
    /// The edges of one type and direction incident to one node. This is the
    /// scope a cardinality bound is declared over.
    Incident {
        /// The node the edges are incident to.
        node: NodeId,
        /// The edge type the bound applies to.
        edge_type: String,
        /// Which side of the node the bound counts.
        direction: Direction,
    },
    /// One ordered pair under one edge type: whether these two are adjacent.
    Pair {
        /// The edge's source.
        source: NodeId,
        /// The edge's target.
        target: NodeId,
        /// The edge type.
        edge_type: String,
    },
    /// One record addressed by its storage key, for a condition that names a
    /// particular row rather than a graph shape.
    Record(Vec<u8>),
    /// One schema element, named by the label or edge type it defines.
    SchemaElement(String),
}

/// What a claim asserts about its scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimPredicate {
    /// The node exists and keeps its identity: the attempt references it and
    /// needs it to still be there. Several attempts may hold this at once.
    EndpointAlive,
    /// The node's identity is being destroyed or reclaimed, which invalidates
    /// every reference right held on it.
    EndpointDestroyed,
    /// The declared bound on the incident scope holds over the whole post
    /// state. Held by any attempt whose mutations change what the bound
    /// counts.
    CardinalityBound {
        /// What the bound counts.
        measure: CardinalityMeasure,
        /// The largest count the bound admits, if it declares one.
        at_most: Option<u32>,
        /// The smallest count the bound admits, if it declares one.
        at_least: Option<u32>,
    },
    /// The pair is adjacent, or is not, and the attempt's result depends on
    /// which. An insertion claims absence, a removal of the last qualifying
    /// row claims presence.
    PairAdjacency {
        /// The state the attempt observed and needs to still hold.
        observed: Adjacency,
    },
    /// The attempt enumerated the complete incident set of its scope and its
    /// result depends on nothing having been added to it since. This is what
    /// protects a scan against a member it never saw.
    IncidentSetComplete,
    /// A conditional cleanup observed this record in a particular version and
    /// its condition was evaluated against that version. A renewal that
    /// changes the record invalidates the condition.
    CleanupCondition {
        /// The version the condition was evaluated against.
        observed_version: u64,
    },
    /// The predicate was evaluated under this schema element as it stood, and
    /// a change of the element changes what is admissible.
    SchemaApplicability,
}

/// Whether a pair was seen as adjacent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adjacency {
    /// The pair has at least one qualifying edge.
    Present,
    /// The pair has none.
    Absent,
}

/// One typed claim: a predicate over a scope, bound to the generation it was
/// evaluated under.
///
/// The generation is part of the claim rather than checked beside it because
/// a predicate evaluated under one schema is not evidence about another: a
/// constraint activated between the evaluation and the commit changes what
/// the same graph shape means.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    /// The logical thing covered.
    pub scope: ClaimScope,
    /// What is asserted about it.
    pub predicate: ClaimPredicate,
    /// The schema revision the predicate was evaluated under.
    pub schema_revision: u64,
}

impl Claim {
    /// A claim over a scope, evaluated under a schema revision.
    pub fn new(scope: ClaimScope, predicate: ClaimPredicate, schema_revision: u64) -> Self {
        Self {
            scope,
            predicate,
            schema_revision,
        }
    }

    /// Whether this claim and `other` can both be admitted.
    ///
    /// Claims over different scopes never conflict, which is what lets
    /// unrelated work proceed and lets many references to one node be held at
    /// once. Within one scope the answer follows the predicate, and the
    /// default for an overlap this function does not recognise is `false`:
    /// an unknown overlap is not permission to bypass protection.
    pub fn compatible_with(&self, other: &Claim) -> bool {
        if !self.scope.overlaps(&other.scope) {
            return true;
        }
        if self.schema_revision != other.schema_revision {
            // One of the two evaluated its predicate under a schema the other
            // did not see, so neither's evidence covers the other's result.
            return false;
        }
        Self::predicates_compatible(&self.predicate, &other.predicate)
            && Self::predicates_compatible(&other.predicate, &self.predicate)
    }

    /// One direction of the compatibility question. Called both ways by
    /// [`Self::compatible_with`], so each arm only has to state the cases it
    /// knows about from its own side.
    fn predicates_compatible(a: &ClaimPredicate, b: &ClaimPredicate) -> bool {
        use ClaimPredicate::*;
        match (a, b) {
            // Independent references to one node coexist. This is the case
            // the whole design exists to keep cheap: without it every edge
            // added to a popular node would queue behind every other.
            (EndpointAlive, EndpointAlive) => true,

            // Destroying the identity invalidates every reference right on
            // it, including one taken a moment ago.
            (EndpointDestroyed, EndpointAlive | EndpointDestroyed) => false,

            // Two attempts that both change what a bound counts decide the
            // same predicate, and only one of them can be right about the
            // post-state it validated against.
            (CardinalityBound { .. }, CardinalityBound { .. }) => false,

            // A bound counts the adjacency a pair claim is about, and an
            // enumeration of the incident set is what a bound is computed
            // from, so neither can be decided without the other.
            (CardinalityBound { .. }, PairAdjacency { .. } | IncidentSetComplete) => false,

            // Both attempts observed the pair and each needs its observation
            // to survive: an insertion that saw absence and a removal that
            // saw presence cannot both be right.
            (PairAdjacency { .. }, PairAdjacency { .. }) => false,

            // A scan that enumerated the set is invalidated by anything that
            // changes membership, including a member it never saw.
            (IncidentSetComplete, IncidentSetComplete | PairAdjacency { .. }) => false,

            // The condition was evaluated against one version of the record;
            // any other claim on that record can have moved it.
            (CleanupCondition { .. }, _) | (_, CleanupCondition { .. }) => false,

            // Changing the element changes what every predicate evaluated
            // under it means.
            (SchemaApplicability, _) | (_, SchemaApplicability) => false,

            // An overlap this function does not recognise is not permission
            // to proceed.
            _ => false,
        }
    }
}

impl ClaimScope {
    /// Whether two scopes cover anything in common.
    ///
    /// A pair overlaps the incident scopes of both its endpoints, because an
    /// edge between them is counted by a bound on either side. Getting that
    /// wrong in the permissive direction is how an insertion escapes the
    /// bound it should have been decided against.
    pub fn overlaps(&self, other: &ClaimScope) -> bool {
        use ClaimScope::*;
        match (self, other) {
            (Node(a), Node(b)) => a == b,
            (
                Incident {
                    node: n1,
                    edge_type: t1,
                    direction: d1,
                },
                Incident {
                    node: n2,
                    edge_type: t2,
                    direction: d2,
                },
            ) => n1 == n2 && t1 == t2 && d1 == d2,
            (
                Pair {
                    source: s1,
                    target: g1,
                    edge_type: t1,
                },
                Pair {
                    source: s2,
                    target: g2,
                    edge_type: t2,
                },
            ) => s1 == s2 && g1 == g2 && t1 == t2,
            (Record(a), Record(b)) => a == b,
            (SchemaElement(a), SchemaElement(b)) => a == b,

            // A pair's edge is incident to both its endpoints, so it is
            // inside the incident scope of either one under the same type
            // and the matching direction.
            (
                Pair {
                    source,
                    target,
                    edge_type: pt,
                },
                Incident {
                    node,
                    edge_type: it,
                    direction,
                },
            )
            | (
                Incident {
                    node,
                    edge_type: it,
                    direction,
                },
                Pair {
                    source,
                    target,
                    edge_type: pt,
                },
            ) => {
                pt == it
                    && match direction {
                        Direction::Outgoing => node == source,
                        Direction::Incoming => node == target,
                    }
            }

            // A node's lifecycle reaches every edge incident to it and every
            // pair it is an endpoint of: destroying it invalidates both.
            (Node(n), Incident { node, .. }) | (Incident { node, .. }, Node(n)) => n == node,
            (Node(n), Pair { source, target, .. }) | (Pair { source, target, .. }, Node(n)) => {
                n == source || n == target
            }

            // A schema element reaches the shapes it defines only through the
            // generation carried on each claim, which is compared separately.
            _ => false,
        }
    }
}

/// The claims of one attempt, composed.
///
/// An operation rarely needs one condition: transferring an edge can need
/// rights on both endpoints, coverage of the source set, the pair projection
/// and the constraint together. They are kept as a set rather than reduced,
/// because reducing them is how a footprint gets lost: folding the mutations
/// that produced them, which the storage layer is free to do, must not fold
/// what they protect.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClaimSet {
    claims: Vec<Claim>,
}

impl ClaimSet {
    /// An empty set.
    pub fn new() -> Self {
        Self { claims: Vec::new() }
    }

    /// Add a claim, keeping duplicates out. The same condition stated twice
    /// by two statements of one attempt is one condition.
    pub fn insert(&mut self, claim: Claim) {
        if !self.claims.contains(&claim) {
            self.claims.push(claim);
        }
    }

    /// The claims held, in the order they were stated.
    pub fn claims(&self) -> &[Claim] {
        &self.claims
    }

    /// Whether the set holds nothing.
    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// How many claims are held. The guard budget is counted in these, so a
    /// caller that has to bound its admission asks here.
    pub fn len(&self) -> usize {
        self.claims.len()
    }

    /// Whether every claim of this attempt can coexist with every claim of
    /// another.
    ///
    /// Compatibility is not transitive and is not summarised: each pair is
    /// asked, because one incompatible pair is enough to exclude the attempt
    /// however many compatible ones surround it.
    pub fn compatible_with(&self, other: &ClaimSet) -> bool {
        self.claims
            .iter()
            .all(|a| other.claims.iter().all(|b| a.compatible_with(b)))
    }

    /// The first pair of claims that cannot coexist, for a caller that has to
    /// say which condition refused the attempt rather than that one did.
    pub fn first_conflict<'a>(&'a self, other: &'a ClaimSet) -> Option<(&'a Claim, &'a Claim)> {
        self.claims.iter().find_map(|a| {
            other
                .claims
                .iter()
                .find(|b| !a.compatible_with(b))
                .map(|b| (a, b))
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
