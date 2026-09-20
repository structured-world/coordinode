use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::merge::{encode_add, encode_remove};
use coordinode_core::txn::invariant::ClaimScope;
use std::collections::HashMap;
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

/// Decide a claim for an attempt that has staged no point writes, which is
/// every case below except the ones about staged writes themselves.
fn decide(engine: &StorageEngine, claim: &Claim, staged: &[(Vec<u8>, AdjOp)]) -> Verdict {
    let no_points = HashMap::new();
    // An attempt whose view is the current state: nothing has been written
    // since it, which is the case every test here but the destruction ones.
    evaluate(engine, claim, staged, &no_points, engine.snapshot()).expect("evaluate")
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
        decide(&engine, &bound(1, Some(1), None), &[]),
        Verdict::Holds
    );

    engine
        .merge(Partition::Adj, &key, &encode_add(2))
        .expect("merge");
    assert_eq!(
        decide(&engine, &bound(1, Some(1), None), &[]),
        Verdict::Holds,
        "one neighbour is within at-most-one"
    );

    engine
        .merge(Partition::Adj, &key, &encode_add(3))
        .expect("merge");
    assert_eq!(
        decide(&engine, &bound(1, Some(1), None), &[]),
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
        decide(&engine, &bound(1, None, Some(1)), &[]),
        Verdict::Holds
    );

    // The attempt stages the removal of the only neighbour: the post-state it
    // would leave is what the bound is decided against, not the state before.
    let staged = vec![(key.clone(), AdjOp::Remove(2))];
    assert_eq!(
        decide(&engine, &bound(1, None, Some(1)), &staged),
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
        decide(&engine, &bound(5, None, Some(1)), &[]),
        Verdict::Broken
    );

    // The same attempt stages the edge: the post-state satisfies the bound,
    // and the intermediate absence is not a violation.
    let staged = vec![(key, AdjOp::Add(6))];
    assert_eq!(
        decide(&engine, &bound(5, None, Some(1)), &staged),
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
        decide(&engine, &bound(1, None, Some(1)), &add_then_remove),
        Verdict::Broken,
        "add then remove leaves the set empty"
    );

    let remove_then_add = vec![(key.clone(), AdjOp::Remove(2)), (key, AdjOp::Add(2))];
    assert_eq!(
        decide(&engine, &bound(1, None, Some(1)), &remove_then_add),
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

    assert_eq!(decide(&engine, &saw_absent, &[]), Verdict::Holds);
    assert_eq!(decide(&engine, &saw_present, &[]), Verdict::Broken);

    engine
        .merge(Partition::Adj, &key, &encode_add(2))
        .expect("merge");
    assert_eq!(
        decide(&engine, &saw_absent, &[]),
        Verdict::Broken,
        "the absence the attempt observed is gone"
    );
    assert_eq!(decide(&engine, &saw_present, &[]), Verdict::Holds);
}

/// The attempt's own staging is not what decides its observation: it is
/// usually the reason the pair looks different, and refusing it for that would
/// refuse every statement that acted on what it saw.
#[test]
fn a_pair_claim_is_not_broken_by_the_attempts_own_edge() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("TAGGED", node(1));
    let saw_absent = Claim::new(
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

    let staged = vec![(key, AdjOp::Add(2))];
    assert_eq!(
        decide(&engine, &saw_absent, &staged),
        Verdict::Holds,
        "creating the edge the absence called for is not a violation of it"
    );
}

/// Somebody else made the pair adjacent after the attempt looked: the
/// observation the attempt built on is gone, and the two write different keys,
/// so nothing else would have caught it.
#[test]
fn a_pair_claim_is_broken_by_a_change_under_the_attempt() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("TAGGED", node(1));
    let view = engine.snapshot();
    engine
        .merge(Partition::Adj, &key, &encode_add(2))
        .expect("merge");

    let saw_absent = Claim::new(
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
    let no_points = HashMap::new();
    assert_eq!(
        evaluate(&engine, &saw_absent, &[], &no_points, view).expect("evaluate"),
        Verdict::Broken
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
        decide(&engine, &claim, &[]),
        Verdict::Holds,
        "another neighbour is not this pair"
    );
}

/// An endpoint claim is about destruction, not about existence: a node row
/// that is simply absent, and that nobody has written since the attempt's
/// view, is not a node that was taken away from it.
#[test]
fn an_endpoint_claim_admits_a_node_that_was_never_written() {
    let (engine, _d) = engine();
    let claim = Claim::new(
        ClaimScope::Node(node(3)),
        ClaimPredicate::EndpointAlive,
        GEN,
    );
    assert_eq!(
        decide(&engine, &claim, &[]),
        Verdict::Holds,
        "refusing here would make every edge writer an enforcer of \
         referential integrity"
    );

    let key = coordinode_core::graph::node::encode_node_key(engine.node_shard(), node(3));
    engine.put(Partition::Node, &key, b"record").expect("put");
    assert_eq!(decide(&engine, &claim, &[]), Verdict::Holds);
}

/// The row was there when the attempt built its result and is gone now: that
/// is the destruction the claim guards against, and the only case in which an
/// absent row refuses.
#[test]
fn an_endpoint_claim_refuses_a_node_deleted_since_the_attempts_view() {
    let (engine, _d) = engine();
    let key = coordinode_core::graph::node::encode_node_key(engine.node_shard(), node(3));
    engine.put(Partition::Node, &key, b"record").expect("put");
    let view = engine.snapshot();
    engine.delete(Partition::Node, &key).expect("delete");

    let claim = Claim::new(
        ClaimScope::Node(node(3)),
        ClaimPredicate::EndpointAlive,
        GEN,
    );
    let no_points = HashMap::new();
    assert_eq!(
        evaluate(&engine, &claim, &[], &no_points, view).expect("evaluate"),
        Verdict::Broken,
        "the identity this attempt references was reclaimed under it"
    );
}

/// The node row is looked up in the shard the engine holds, not in shard zero:
/// a deployment whose statements run against another shard would otherwise
/// read another node's row, or none.
#[test]
fn an_endpoint_claim_reads_the_shard_the_engine_holds() {
    let (engine, _d) = engine();
    engine.set_node_shard(7);

    let held = coordinode_core::graph::node::encode_node_key(7, node(3));
    engine.put(Partition::Node, &held, b"record").expect("put");
    let view = engine.snapshot();
    engine.delete(Partition::Node, &held).expect("delete");

    // The same id in another shard is another node, and its surviving row is
    // no evidence about this one.
    let elsewhere = coordinode_core::graph::node::encode_node_key(0, node(3));
    engine
        .put(Partition::Node, &elsewhere, b"record")
        .expect("put");

    let claim = Claim::new(
        ClaimScope::Node(node(3)),
        ClaimPredicate::EndpointAlive,
        GEN,
    );
    let no_points = HashMap::new();
    assert_eq!(
        evaluate(&engine, &claim, &[], &no_points, view).expect("evaluate"),
        Verdict::Broken,
        "another shard's row is another node"
    );
}

/// A node of a temporal label exists only as versions under the per-version
/// key. Reading the plain row alone would call it dead and refuse every edge
/// attached to it.
#[test]
fn an_endpoint_claim_sees_a_node_stored_as_versions() {
    let (engine, _d) = engine();
    let claim = Claim::new(
        ClaimScope::Node(node(8)),
        ClaimPredicate::EndpointAlive,
        GEN,
    );
    let versioned =
        coordinode_core::graph::node::encode_temporal_node_key(engine.node_shard(), node(8), 1_700);
    engine
        .put(Partition::Node, &versioned, b"record")
        .expect("put");
    assert_eq!(
        decide(&engine, &claim, &[]),
        Verdict::Holds,
        "a version of the node is the node"
    );
}

/// The attempt's own point writes decide the endpoint in both directions: a
/// node it is creating is alive before anything is committed, and one it is
/// deleting is gone before the tombstone lands.
#[test]
fn an_endpoint_claim_reads_the_attempts_own_point_writes() {
    let (engine, _d) = engine();
    let claim = Claim::new(
        ClaimScope::Node(node(4)),
        ClaimPredicate::EndpointAlive,
        GEN,
    );
    let key = coordinode_core::graph::node::encode_node_key(engine.node_shard(), node(4));

    let mut staged_points = HashMap::new();
    staged_points.insert((Partition::Node, key.clone()), Some(b"record".to_vec()));
    assert_eq!(
        evaluate(&engine, &claim, &[], &staged_points, engine.snapshot()).expect("evaluate"),
        Verdict::Holds,
        "a node created by this attempt is alive for this attempt's own edge"
    );

    engine.put(Partition::Node, &key, b"record").expect("put");
    let mut deleted = HashMap::new();
    deleted.insert((Partition::Node, key), None);
    assert_eq!(
        evaluate(&engine, &claim, &[], &deleted, engine.snapshot()).expect("evaluate"),
        Verdict::Broken,
        "a node this attempt deletes is not one it may attach to"
    );
}

/// A destruction is the attempt's own intent, not a condition read from state,
/// so it is admitted here and excluded where it belongs: against the reference
/// rights other attempts hold.
#[test]
fn a_destruction_claim_is_admitted_by_the_evaluator() {
    let (engine, _d) = engine();
    let claim = Claim::new(
        ClaimScope::Node(node(3)),
        ClaimPredicate::EndpointDestroyed,
        GEN,
    );
    assert_eq!(
        decide(&engine, &claim, &[]),
        Verdict::Holds,
        "refusing this would refuse every deletion"
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
        decide(&engine, &scan, &[]),
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
    assert_eq!(decide(&engine, &cleanup, &[]), Verdict::Undecidable);

    let schema = Claim::new(
        ClaimScope::SchemaElement("OWNS".to_string()),
        ClaimPredicate::SchemaApplicability,
        GEN,
    );
    assert_eq!(decide(&engine, &schema, &[]), Verdict::Undecidable);

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
        decide(&engine, &incoming_instances, &[]),
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
        decide(&engine, &neighbours, &[]),
        Verdict::Holds,
        "one neighbour satisfies at-most-one distinct neighbours"
    );
    assert_eq!(
        decide(&engine, &instances, &[]),
        Verdict::Broken,
        "the same adjacency holds two identities and violates at-most-one instances"
    );
}

/// An instance bound is decided against the post-state the attempt would
/// leave, so the entries it stages count and the ones it tombstones do not.
/// Counting only what is committed would admit the instance that breaks the
/// bound and refuse the deletion that repairs it.
#[test]
fn an_instance_bound_counts_the_attempts_own_entries() {
    let (engine, _d) = engine();
    let adj = encode_adj_key_forward("KNOWS", node(1));
    engine
        .merge(Partition::Adj, &adj, &encode_add(2))
        .expect("merge");

    let entry = |disc: &[u8]| {
        let mut key = Vec::new();
        key.extend_from_slice(b"edgeprop:KNOWS:");
        key.extend_from_slice(&node(1).as_raw().to_be_bytes());
        key.push(b':');
        key.extend_from_slice(&node(2).as_raw().to_be_bytes());
        key.push(b':');
        key.extend_from_slice(disc);
        key
    };

    let committed = entry(b"work");
    engine
        .put(Partition::EdgeProp, &committed, b"props")
        .expect("put");

    let at_most_one = Claim::new(
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
    assert_eq!(decide(&engine, &at_most_one, &[]), Verdict::Holds);

    // The attempt stages a second identity: the bound it claims is broken by
    // its own post-state, not by anything committed.
    let mut adding = HashMap::new();
    adding.insert(
        (Partition::EdgeProp, entry(b"college")),
        Some(b"props".to_vec()),
    );
    assert_eq!(
        evaluate(&engine, &at_most_one, &[], &adding, engine.snapshot()).expect("evaluate"),
        Verdict::Broken,
        "the instance this attempt adds is part of what the bound counts"
    );

    // And the reverse: with two committed identities, the attempt that
    // removes one leaves a post-state the bound admits.
    let second = entry(b"college");
    engine
        .put(Partition::EdgeProp, &second, b"props")
        .expect("put");
    assert_eq!(decide(&engine, &at_most_one, &[]), Verdict::Broken);

    let mut removing = HashMap::new();
    removing.insert((Partition::EdgeProp, second), None);
    assert_eq!(
        evaluate(&engine, &at_most_one, &[], &removing, engine.snapshot()).expect("evaluate"),
        Verdict::Holds,
        "the entry this attempt tombstones is not one the bound still counts"
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
        decide(&engine, &bound(1, None, Some(1)), &staged),
        Verdict::Holds,
        "another scope's staged removal is not this scope's"
    );
    let _ = encode_remove(0);
}
