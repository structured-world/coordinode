use super::*;
use coordinode_core::graph::node::NodeId;
use coordinode_core::txn::invariant::{CardinalityMeasure, ClaimPredicate, ClaimScope, Direction};

const GEN: u64 = 1;

fn bound_on(node: u64) -> ClaimSet {
    let mut set = ClaimSet::new();
    set.insert(Claim::new(
        ClaimScope::Incident {
            node: NodeId::from_raw(node),
            edge_type: "OWNS".to_string(),
            direction: Direction::Outgoing,
        },
        ClaimPredicate::CardinalityBound {
            measure: CardinalityMeasure::EdgeInstances,
            at_most: Some(1),
            at_least: None,
        },
        GEN,
    ));
    set
}

fn reference_to(node: u64) -> ClaimSet {
    let mut set = ClaimSet::new();
    set.insert(Claim::new(
        ClaimScope::Node(NodeId::from_raw(node)),
        ClaimPredicate::EndpointAlive,
        GEN,
    ));
    set
}

fn destroys(node: u64) -> ClaimSet {
    let mut set = ClaimSet::new();
    set.insert(Claim::new(
        ClaimScope::Node(NodeId::from_raw(node)),
        ClaimPredicate::EndpointDestroyed,
        GEN,
    ));
    set
}

/// The second attempt to decide one bound is refused, and told which claim of
/// which attempt refused it.
#[test]
fn the_second_attempt_on_one_bound_is_refused_by_name() {
    let reg = ClaimRegistry::new(64);
    reg.reserve(1, &bound_on(7))
        .expect("first attempt reserves");

    let refusal = reg
        .reserve(2, &bound_on(7))
        .expect_err("the second cannot decide the same bound");

    match *refusal {
        ClaimRefusal::Incompatible { holder, .. } => {
            assert_eq!(holder, 1, "the refusal names who holds the claim")
        }
        other => panic!("expected an incompatibility, got {other:?}"),
    }
}

/// Independent references to one node are admitted together. Without this the
/// table would serialise every addition to a popular node.
#[test]
fn many_references_to_one_node_are_admitted_together() {
    let reg = ClaimRegistry::new(64);
    for attempt in 1..=16 {
        reg.reserve(attempt, &reference_to(7))
            .expect("independent references coexist");
    }
    assert_eq!(reg.attempts(), 16);
    assert_eq!(reg.reserved_claims(), 16);
}

/// A destructive lifecycle change is refused while references are held, and
/// admitted once they are released.
#[test]
fn a_destroy_waits_for_the_references_to_go() {
    let reg = ClaimRegistry::new(64);
    reg.reserve(1, &reference_to(7)).expect("reference");
    assert!(
        reg.reserve(2, &destroys(7)).is_err(),
        "a reference in flight excludes the destroy"
    );

    reg.release(1);
    reg.reserve(2, &destroys(7))
        .expect("with the reference gone the destroy is admissible");
}

/// Releasing returns the budget, so a long-running leader does not leak the
/// claims of attempts that ended.
#[test]
fn releasing_returns_the_budget() {
    let reg = ClaimRegistry::new(64);
    reg.reserve(1, &bound_on(1)).expect("reserve");
    reg.reserve(2, &bound_on(2)).expect("reserve");
    assert_eq!(reg.reserved_claims(), 2);

    reg.release(1);
    assert_eq!(reg.reserved_claims(), 1);
    reg.release(2);
    assert_eq!(reg.reserved_claims(), 0);
    assert_eq!(reg.attempts(), 0);

    // Releasing an attempt that holds nothing is not an error and does not
    // move the budget.
    reg.release(99);
    assert_eq!(reg.reserved_claims(), 0);
}

/// The table is bounded: at its limit an attempt is refused while it can
/// still be retried, rather than letting guard memory grow with load.
#[test]
fn the_table_refuses_rather_than_growing_without_bound() {
    let reg = ClaimRegistry::new(2);
    reg.reserve(1, &bound_on(1)).expect("one");
    reg.reserve(2, &bound_on(2)).expect("two");

    let refusal = reg.reserve(3, &bound_on(3)).expect_err("at the limit");
    assert!(matches!(*refusal, ClaimRefusal::AtCapacity { limit: 2 }));
    assert_eq!(
        reg.reserved_claims(),
        2,
        "a refused attempt reserved nothing"
    );
}

/// An attempt that states more conditions as it goes replaces its own entry
/// instead of conflicting with itself.
#[test]
fn an_attempt_does_not_conflict_with_itself() {
    let reg = ClaimRegistry::new(64);
    reg.reserve(1, &bound_on(7)).expect("first statement");
    reg.reserve(1, &bound_on(7))
        .expect("the same attempt again");

    let mut grown = bound_on(7);
    for claim in reference_to(9).claims() {
        grown.insert(claim.clone());
    }
    reg.reserve(1, &grown)
        .expect("the set grows with the attempt");

    assert_eq!(reg.attempts(), 1);
    assert_eq!(
        reg.reserved_claims(),
        2,
        "the budget follows the latest set"
    );
}

/// An attempt with no claims reserves nothing and blocks nobody, so a
/// read-only statement never touches the table.
#[test]
fn an_attempt_with_no_claims_is_free() {
    let reg = ClaimRegistry::new(1);
    reg.reserve(1, &ClaimSet::new())
        .expect("nothing to reserve");
    assert_eq!(reg.attempts(), 0);

    // And the one slot is still available to an attempt that needs it.
    reg.reserve(2, &bound_on(1))
        .expect("the budget was untouched");
}

/// Unrelated scopes are admitted concurrently, which is what keeps one
/// contended vertex from stopping the rest of the graph.
#[test]
fn unrelated_scopes_do_not_queue_behind_each_other() {
    let reg = ClaimRegistry::new(64);
    for attempt in 1..=32 {
        reg.reserve(attempt, &bound_on(attempt))
            .expect("different nodes are independent");
    }
    assert_eq!(reg.attempts(), 32);
}
