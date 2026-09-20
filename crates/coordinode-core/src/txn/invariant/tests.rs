//! The schedules the protection exists for.
//!
//! Each test is a pair of attempts that snapshot isolation admits and that
//! together break a condition. They are written as the independent model the
//! acceptance asks for: two claim sets, the question of whether both may be
//! admitted, and the answer the invariant requires. What refuses them at the
//! commit boundary is the guard's job; what makes the refusal correct is
//! here.

use super::*;
use alloc::string::ToString;
use alloc::vec;

const GEN: u64 = 7;

fn node(id: u64) -> NodeId {
    NodeId::from_raw(id)
}

fn incident(id: u64, ty: &str, direction: Direction) -> ClaimScope {
    ClaimScope::Incident {
        node: node(id),
        edge_type: ty.to_string(),
        direction,
    }
}

fn pair(source: u64, target: u64, ty: &str) -> ClaimScope {
    ClaimScope::Pair {
        source: node(source),
        target: node(target),
        edge_type: ty.to_string(),
    }
}

fn set(claims: Vec<Claim>) -> ClaimSet {
    let mut s = ClaimSet::new();
    for c in claims {
        s.insert(c);
    }
    s
}

fn at_most_one() -> ClaimPredicate {
    ClaimPredicate::CardinalityBound {
        measure: CardinalityMeasure::EdgeInstances,
        at_most: Some(1),
        at_least: None,
    }
}

fn at_least_one() -> ClaimPredicate {
    ClaimPredicate::CardinalityBound {
        measure: CardinalityMeasure::EdgeInstances,
        at_most: None,
        at_least: Some(1),
    }
}

/// Two additions under an at-most-one bound. Each sees zero edges and each is
/// admissible alone; together they are two.
#[test]
fn add_add_under_an_upper_bound_cannot_both_be_admitted() {
    let scope = incident(1, "OWNS", Direction::Outgoing);
    let first = set(vec![Claim::new(scope.clone(), at_most_one(), GEN)]);
    let second = set(vec![Claim::new(scope, at_most_one(), GEN)]);

    assert!(
        !first.compatible_with(&second),
        "two additions decide the same bound and only one can be right about the post-state"
    );
    assert!(
        first.first_conflict(&second).is_some(),
        "and the pair is nameable"
    );
}

/// Two removals under an at-least-one bound. Each sees two edges and leaves
/// one; together they leave none.
#[test]
fn remove_remove_under_a_lower_bound_cannot_both_be_admitted() {
    let scope = incident(1, "MEMBER_OF", Direction::Outgoing);
    let first = set(vec![Claim::new(scope.clone(), at_least_one(), GEN)]);
    let second = set(vec![Claim::new(scope, at_least_one(), GEN)]);

    assert!(!first.compatible_with(&second));
}

/// A node deleted while another attempt attaches an edge to it, in both
/// orders. Neither order may admit both.
#[test]
fn an_endpoint_delete_excludes_a_reference_in_either_order() {
    let scope = ClaimScope::Node(node(42));
    let deleter = set(vec![Claim::new(
        scope.clone(),
        ClaimPredicate::EndpointDestroyed,
        GEN,
    )]);
    let referencer = set(vec![Claim::new(scope, ClaimPredicate::EndpointAlive, GEN)]);

    assert!(!deleter.compatible_with(&referencer), "delete then add");
    assert!(!referencer.compatible_with(&deleter), "add then delete");
}

/// The case that must stay cheap: two attempts that both reference a node
/// without destroying it. If these excluded each other, every addition to a
/// popular node would queue behind every other.
#[test]
fn independent_references_to_one_node_are_compatible() {
    let scope = ClaimScope::Node(node(42));
    let first = set(vec![Claim::new(
        scope.clone(),
        ClaimPredicate::EndpointAlive,
        GEN,
    )]);
    let second = set(vec![Claim::new(scope, ClaimPredicate::EndpointAlive, GEN)]);

    assert!(
        first.compatible_with(&second),
        "independent references must not exclusively rewrite one shared record"
    );
}

/// Erasing the last qualifying row of a pair while another attempt inserts
/// into it. The removal's result depends on the pair being adjacent and the
/// insertion's on it not being.
#[test]
fn erasing_the_last_row_cannot_race_an_insertion_into_the_same_pair() {
    let scope = pair(1, 2, "TAGGED");
    let eraser = set(vec![Claim::new(
        scope.clone(),
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Present,
        },
        GEN,
    )]);
    let inserter = set(vec![Claim::new(
        scope,
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Absent,
        },
        GEN,
    )]);

    assert!(!eraser.compatible_with(&inserter));
}

/// Insertions into different pairs are independent of each other, and both
/// still meet a bound declared on the vertex they share.
#[test]
fn insertions_into_different_pairs_are_independent_but_share_the_bound() {
    let first_pair = set(vec![Claim::new(
        pair(1, 2, "TAGGED"),
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Absent,
        },
        GEN,
    )]);
    let second_pair = set(vec![Claim::new(
        pair(1, 3, "TAGGED"),
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Absent,
        },
        GEN,
    )]);
    assert!(
        first_pair.compatible_with(&second_pair),
        "different pairs do not decide each other"
    );

    // The same two insertions against a bound on the vertex they share are
    // not independent: each edge is counted by it.
    let bound = set(vec![Claim::new(
        incident(1, "TAGGED", Direction::Outgoing),
        at_most_one(),
        GEN,
    )]);
    assert!(
        !first_pair.compatible_with(&bound),
        "an insertion is counted by a bound on its source"
    );
}

/// A scan that enumerated the incident set against an insertion into it: the
/// member the scan never saw is exactly what the claim protects.
#[test]
fn a_completed_scan_excludes_an_insertion_it_never_saw() {
    let scanner = set(vec![Claim::new(
        incident(1, "OWNS", Direction::Outgoing),
        ClaimPredicate::IncidentSetComplete,
        GEN,
    )]);
    let inserter = set(vec![Claim::new(
        pair(1, 9, "OWNS"),
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Absent,
        },
        GEN,
    )]);

    assert!(
        !scanner.compatible_with(&inserter),
        "a scan's completeness is invalidated by a member added after it"
    );
}

/// A conditional cleanup against a renewal of the same record. The cleanup
/// evaluated its condition against a version the renewal moves.
#[test]
fn a_cleanup_condition_excludes_a_renewal_of_the_record_it_observed() {
    let key = ClaimScope::Record(b"node:00:0000002a".to_vec());
    let cleanup = set(vec![Claim::new(
        key.clone(),
        ClaimPredicate::CleanupCondition {
            observed_version: 11,
        },
        GEN,
    )]);
    let renewal = set(vec![Claim::new(
        key,
        ClaimPredicate::CleanupCondition {
            observed_version: 12,
        },
        GEN,
    )]);

    assert!(!cleanup.compatible_with(&renewal));
}

/// A predicate evaluated under one schema generation is not evidence about
/// another, so activation of a constraint excludes work that never saw it.
#[test]
fn a_predicate_evaluated_under_another_generation_is_not_evidence() {
    let scope = incident(1, "OWNS", Direction::Outgoing);
    let before = set(vec![Claim::new(scope.clone(), at_most_one(), GEN)]);
    let after = set(vec![Claim::new(scope, at_most_one(), GEN + 1)]);

    assert!(
        !before.compatible_with(&after),
        "a constraint activated in between changes what the same shape means"
    );
}

/// A node and its mandatory edge created in one attempt is a valid final
/// state: the intermediate absence of the edge is not a violation, and the
/// claims of one attempt are never asked about each other.
#[test]
fn a_node_and_its_mandatory_edge_in_one_attempt_are_one_post_state() {
    let attempt = set(vec![
        Claim::new(
            ClaimScope::Node(node(5)),
            ClaimPredicate::EndpointAlive,
            GEN,
        ),
        Claim::new(
            incident(5, "BELONGS_TO", Direction::Outgoing),
            at_least_one(),
            GEN,
        ),
    ]);

    // Composition keeps both, because the footprint of one is not the other.
    assert_eq!(attempt.len(), 2);

    // An unrelated attempt elsewhere is not disturbed by either.
    let elsewhere = set(vec![Claim::new(
        ClaimScope::Node(node(6)),
        ClaimPredicate::EndpointAlive,
        GEN,
    )]);
    assert!(attempt.compatible_with(&elsewhere));
}

/// The two measures answer different questions, so a claim for one is not a
/// claim for the other, and an overlap of scopes still refuses.
#[test]
fn the_two_cardinality_measures_do_not_stand_in_for_each_other() {
    let scope = incident(1, "TAGGED", Direction::Outgoing);
    let instances = set(vec![Claim::new(
        scope.clone(),
        ClaimPredicate::CardinalityBound {
            measure: CardinalityMeasure::EdgeInstances,
            at_most: Some(3),
            at_least: None,
        },
        GEN,
    )]);
    let neighbours = set(vec![Claim::new(
        scope,
        ClaimPredicate::CardinalityBound {
            measure: CardinalityMeasure::DistinctNeighbours,
            at_most: Some(3),
            at_least: None,
        },
        GEN,
    )]);

    assert!(
        !instances.compatible_with(&neighbours),
        "one measure's proof cannot serve the other"
    );
}

/// Unrelated scopes never meet, which is what lets the rest of the graph
/// proceed while one vertex is contended.
#[test]
fn unrelated_scopes_do_not_conflict() {
    let here = set(vec![Claim::new(
        incident(1, "OWNS", Direction::Outgoing),
        at_most_one(),
        GEN,
    )]);
    let there = set(vec![Claim::new(
        incident(2, "OWNS", Direction::Outgoing),
        at_most_one(),
        GEN,
    )]);
    let other_type = set(vec![Claim::new(
        incident(1, "LIKES", Direction::Outgoing),
        at_most_one(),
        GEN,
    )]);
    let other_direction = set(vec![Claim::new(
        incident(1, "OWNS", Direction::Incoming),
        at_most_one(),
        GEN,
    )]);

    assert!(here.compatible_with(&there), "different nodes");
    assert!(here.compatible_with(&other_type), "different edge types");
    assert!(
        here.compatible_with(&other_direction),
        "different directions of one node"
    );
}

/// Destroying a node reaches the edges incident to it and the pairs it is an
/// endpoint of, not only claims that name the node itself.
#[test]
fn destroying_a_node_reaches_its_incident_scopes() {
    let deleter = set(vec![Claim::new(
        ClaimScope::Node(node(1)),
        ClaimPredicate::EndpointDestroyed,
        GEN,
    )]);
    let bound = set(vec![Claim::new(
        incident(1, "OWNS", Direction::Outgoing),
        at_most_one(),
        GEN,
    )]);
    let as_source = set(vec![Claim::new(
        pair(1, 2, "OWNS"),
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Absent,
        },
        GEN,
    )]);
    let as_target = set(vec![Claim::new(
        pair(2, 1, "OWNS"),
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Absent,
        },
        GEN,
    )]);

    assert!(!deleter.compatible_with(&bound));
    assert!(!deleter.compatible_with(&as_source));
    assert!(!deleter.compatible_with(&as_target));
}

/// An empty set conflicts with nothing: a read-only attempt holds no claims
/// and blocks nobody.
#[test]
fn an_attempt_with_no_claims_blocks_nothing() {
    let empty = ClaimSet::new();
    let busy = set(vec![Claim::new(
        incident(1, "OWNS", Direction::Outgoing),
        at_most_one(),
        GEN,
    )]);

    assert!(empty.is_empty());
    assert!(empty.compatible_with(&busy));
    assert!(busy.compatible_with(&empty));
    assert!(empty.first_conflict(&busy).is_none());
}

/// The same condition stated twice by two statements of one attempt is one
/// condition, so a long statement chain does not inflate the guard budget.
#[test]
fn a_repeated_claim_is_stored_once() {
    let claim = Claim::new(incident(1, "OWNS", Direction::Outgoing), at_most_one(), GEN);
    let mut s = ClaimSet::new();
    s.insert(claim.clone());
    s.insert(claim.clone());
    s.insert(claim);
    assert_eq!(s.len(), 1);
}
