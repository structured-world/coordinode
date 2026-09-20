use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::merge::{encode_add, encode_remove};
use coordinode_core::txn::invariant::ClaimScope;
use tempfile::TempDir;

const GEN: u64 = 1;

fn engine() -> (StorageEngine, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    (StorageEngine::open(&config).expect("open"), dir)
}

fn node(id: u64) -> NodeId {
    NodeId::from_raw(id)
}

fn bound(node_id: u64, at_most: Option<u32>, at_least: Option<u32>) -> Claim {
    Claim::new(
        ClaimScope::Incident {
            node: node(node_id),
            edge_type: "OWNS".to_string(),
            direction: Direction::Outgoing,
        },
        ClaimPredicate::CardinalityBound {
            measure: CardinalityMeasure::DistinctNeighbours,
            at_most,
            at_least,
        },
        GEN,
    )
}

/// An upper bound is decided against the neighbours actually stored.
#[test]
fn an_upper_bound_is_decided_against_stored_adjacency() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("OWNS", node(1));

    // No edges: at-most-one holds.
    assert_eq!(
        evaluate(&engine, &bound(1, Some(1), None), &[]).expect("evaluate"),
        Verdict::Holds
    );

    engine
        .merge(Partition::Adj, &key, &encode_add(2))
        .expect("merge");
    assert_eq!(
        evaluate(&engine, &bound(1, Some(1), None), &[]).expect("evaluate"),
        Verdict::Holds,
        "one neighbour is within at-most-one"
    );

    engine
        .merge(Partition::Adj, &key, &encode_add(3))
        .expect("merge");
    assert_eq!(
        evaluate(&engine, &bound(1, Some(1), None), &[]).expect("evaluate"),
        Verdict::Broken,
        "two neighbours are not"
    );
}

/// A lower bound refuses the state that would leave the set empty, which is
/// the removal side of the same protection.
#[test]
fn a_lower_bound_refuses_an_empty_result() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("OWNS", node(1));
    engine
        .merge(Partition::Adj, &key, &encode_add(2))
        .expect("merge");

    assert_eq!(
        evaluate(&engine, &bound(1, None, Some(1)), &[]).expect("evaluate"),
        Verdict::Holds
    );

    // The attempt stages the removal of the only neighbour: the post-state it
    // would leave is what the bound is decided against, not the state before.
    let staged = vec![(key.clone(), AdjOp::Remove(2))];
    assert_eq!(
        evaluate(&engine, &bound(1, None, Some(1)), &staged).expect("evaluate"),
        Verdict::Broken,
        "the bound is decided against the post-state the attempt would leave"
    );
}

/// The attempt's own staged writes count, so a node and its mandatory edge
/// created together are a valid post-state even though the edge is not yet in
/// the engine.
#[test]
fn the_attempts_own_writes_are_part_of_the_post_state() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("OWNS", node(5));

    // Nothing stored: a lower bound is broken.
    assert_eq!(
        evaluate(&engine, &bound(5, None, Some(1)), &[]).expect("evaluate"),
        Verdict::Broken
    );

    // The same attempt stages the edge: the post-state satisfies the bound,
    // and the intermediate absence is not a violation.
    let staged = vec![(key, AdjOp::Add(6))];
    assert_eq!(
        evaluate(&engine, &bound(5, None, Some(1)), &staged).expect("evaluate"),
        Verdict::Holds
    );
}

/// Staged writes are applied in order, so an add followed by a remove leaves
/// nothing and a remove followed by an add leaves the member.
#[test]
fn staged_writes_are_applied_in_the_order_they_were_staged() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("OWNS", node(1));

    let add_then_remove = vec![
        (key.clone(), AdjOp::Add(2)),
        (key.clone(), AdjOp::Remove(2)),
    ];
    assert_eq!(
        evaluate(&engine, &bound(1, None, Some(1)), &add_then_remove).expect("evaluate"),
        Verdict::Broken,
        "add then remove leaves the set empty"
    );

    let remove_then_add = vec![(key.clone(), AdjOp::Remove(2)), (key, AdjOp::Add(2))];
    assert_eq!(
        evaluate(&engine, &bound(1, None, Some(1)), &remove_then_add).expect("evaluate"),
        Verdict::Holds,
        "remove then add leaves the member"
    );
}

/// A pair claim is decided against whether the pair is adjacent, in both
/// directions of the observation.
#[test]
fn a_pair_claim_is_decided_against_adjacency() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("TAGGED", node(1));
    let scope = ClaimScope::Pair {
        source: node(1),
        target: node(2),
        edge_type: "TAGGED".to_string(),
    };
    let saw_absent = Claim::new(
        scope.clone(),
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Absent,
        },
        GEN,
    );
    let saw_present = Claim::new(
        scope,
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Present,
        },
        GEN,
    );

    assert_eq!(
        evaluate(&engine, &saw_absent, &[]).expect("evaluate"),
        Verdict::Holds
    );
    assert_eq!(
        evaluate(&engine, &saw_present, &[]).expect("evaluate"),
        Verdict::Broken
    );

    engine
        .merge(Partition::Adj, &key, &encode_add(2))
        .expect("merge");
    assert_eq!(
        evaluate(&engine, &saw_absent, &[]).expect("evaluate"),
        Verdict::Broken,
        "the absence the attempt observed is gone"
    );
    assert_eq!(
        evaluate(&engine, &saw_present, &[]).expect("evaluate"),
        Verdict::Holds
    );
}

/// A removal of an unrelated neighbour does not disturb a pair claim about a
/// different target.
#[test]
fn a_pair_claim_ignores_other_neighbours() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("TAGGED", node(1));
    engine
        .merge(Partition::Adj, &key, &encode_add(9))
        .expect("merge");

    let claim = Claim::new(
        ClaimScope::Pair {
            source: node(1),
            target: node(2),
            edge_type: "TAGGED".to_string(),
        },
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Absent,
        },
        GEN,
    );
    assert_eq!(
        evaluate(&engine, &claim, &[]).expect("evaluate"),
        Verdict::Holds,
        "another neighbour is not this pair"
    );
}

/// An endpoint claim is decided against whether the node record is there.
#[test]
fn an_endpoint_claim_is_decided_against_the_node_record() {
    let (engine, _d) = engine();
    let claim = Claim::new(
        ClaimScope::Node(node(3)),
        ClaimPredicate::EndpointAlive,
        GEN,
    );
    assert_eq!(
        evaluate(&engine, &claim, &[]).expect("evaluate"),
        Verdict::Broken,
        "a node that is not there cannot be referenced"
    );

    let key = coordinode_core::graph::node::encode_node_key(0, node(3));
    engine.put(Partition::Node, &key, b"record").expect("put");
    assert_eq!(
        evaluate(&engine, &claim, &[]).expect("evaluate"),
        Verdict::Holds
    );
}

/// What this evaluator is not given evidence for, it refuses to decide, and
/// the caller must not read that as a pass.
#[test]
fn a_claim_without_evidence_here_is_undecidable_not_satisfied() {
    let (engine, _d) = engine();

    let scan = Claim::new(
        ClaimScope::Incident {
            node: node(1),
            edge_type: "OWNS".to_string(),
            direction: Direction::Outgoing,
        },
        ClaimPredicate::IncidentSetComplete,
        GEN,
    );
    assert_eq!(
        evaluate(&engine, &scan, &[]).expect("evaluate"),
        Verdict::Undecidable,
        "the enumeration the attempt performed is not visible here"
    );

    let cleanup = Claim::new(
        ClaimScope::Record(b"node:00:0000002a".to_vec()),
        ClaimPredicate::CleanupCondition {
            observed_version: 4,
        },
        GEN,
    );
    assert_eq!(
        evaluate(&engine, &cleanup, &[]).expect("evaluate"),
        Verdict::Undecidable
    );

    let schema = Claim::new(
        ClaimScope::SchemaElement("OWNS".to_string()),
        ClaimPredicate::SchemaApplicability,
        GEN,
    );
    assert_eq!(
        evaluate(&engine, &schema, &[]).expect("evaluate"),
        Verdict::Undecidable
    );

    // An incoming instance count is not reachable from the target's side,
    // and saying so is not the same as saying the bound holds.
    let incoming_instances = Claim::new(
        ClaimScope::Incident {
            node: node(1),
            edge_type: "OWNS".to_string(),
            direction: Direction::Incoming,
        },
        ClaimPredicate::CardinalityBound {
            measure: CardinalityMeasure::EdgeInstances,
            at_most: Some(1),
            at_least: None,
        },
        GEN,
    );
    assert_eq!(
        evaluate(&engine, &incoming_instances, &[]).expect("evaluate"),
        Verdict::Undecidable
    );
}

/// The two measures count different things on the same adjacency: two
/// discriminated instances to one neighbour are two identities and one
/// neighbour, and a claim for one must not be answered with the other's
/// number.
#[test]
fn the_two_measures_count_different_things_on_one_adjacency() {
    let (engine, _d) = engine();
    let adj = encode_adj_key_forward("KNOWS", node(1));
    engine
        .merge(Partition::Adj, &adj, &encode_add(2))
        .expect("merge");

    // Two discriminated entries for the one pair: two logical identities.
    for disc in [b"work", b"clge"] {
        let mut key = Vec::new();
        key.extend_from_slice(b"edgeprop:KNOWS:");
        key.extend_from_slice(&node(1).as_raw().to_be_bytes());
        key.push(b':');
        key.extend_from_slice(&node(2).as_raw().to_be_bytes());
        key.push(b':');
        key.extend_from_slice(disc);
        engine
            .put(Partition::EdgeProp, &key, b"props")
            .expect("put");
    }

    let neighbours = Claim::new(
        ClaimScope::Incident {
            node: node(1),
            edge_type: "KNOWS".to_string(),
            direction: Direction::Outgoing,
        },
        ClaimPredicate::CardinalityBound {
            measure: CardinalityMeasure::DistinctNeighbours,
            at_most: Some(1),
            at_least: None,
        },
        GEN,
    );
    let instances = Claim::new(
        ClaimScope::Incident {
            node: node(1),
            edge_type: "KNOWS".to_string(),
            direction: Direction::Outgoing,
        },
        ClaimPredicate::CardinalityBound {
            measure: CardinalityMeasure::EdgeInstances,
            at_most: Some(1),
            at_least: None,
        },
        GEN,
    );

    assert_eq!(
        evaluate(&engine, &neighbours, &[]).expect("evaluate"),
        Verdict::Holds,
        "one neighbour satisfies at-most-one distinct neighbours"
    );
    assert_eq!(
        evaluate(&engine, &instances, &[]).expect("evaluate"),
        Verdict::Broken,
        "the same adjacency holds two identities and violates at-most-one instances"
    );
}

/// A staged removal that does not name this scope's key leaves it alone.
#[test]
fn staged_writes_for_another_scope_are_ignored() {
    let (engine, _d) = engine();
    let mine = encode_adj_key_forward("OWNS", node(1));
    engine
        .merge(Partition::Adj, &mine, &encode_add(2))
        .expect("merge");

    let other = encode_adj_key_forward("OWNS", node(99));
    let staged = vec![(other, AdjOp::Remove(2))];

    assert_eq!(
        evaluate(&engine, &bound(1, None, Some(1)), &staged).expect("evaluate"),
        Verdict::Holds,
        "another scope's staged removal is not this scope's"
    );
    let _ = encode_remove(0);
}
