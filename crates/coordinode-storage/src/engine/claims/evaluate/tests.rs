use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::merge::{encode_add, encode_remove};
use crate::engine::transaction::AdjOp;
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

use coordinode_core::txn::invariant::{CardinalityBound, CardinalityMeasure};

fn bound(node_id: u64, bound: CardinalityBound) -> Claim {
    Claim::new(
        ClaimScope::Incident {
            node: node(node_id),
            edge_type: "OWNS".to_string(),
            direction: Direction::Outgoing,
        },
        ClaimPredicate::CardinalityBound {
            measure: CardinalityMeasure::DistinctNeighbours,
            bound,
            trend: coordinode_core::txn::invariant::CountTrend::Changes,
        },
        GEN,
    )
}

/// A bound of `measure` on `node`'s `edge_type` edges in `direction`.
fn counted(
    node_id: u64,
    edge_type: &str,
    direction: Direction,
    measure: CardinalityMeasure,
    bound: CardinalityBound,
) -> Claim {
    Claim::new(
        ClaimScope::Incident {
            node: node(node_id),
            edge_type: edge_type.to_string(),
            direction,
        },
        ClaimPredicate::CardinalityBound {
            measure,
            bound,
            trend: coordinode_core::txn::invariant::CountTrend::Changes,
        },
        GEN,
    )
}

/// Publish the definition of `name`, discriminated by a string `context`
/// when `discriminated`, temporal when `temporal`, at revision 1.
fn define(engine: &StorageEngine, name: &str, discriminated: bool, temporal: bool) {
    use coordinode_core::schema::definition::{
        EdgeTypeSchema, PropertyDef, PropertyType, encode_edge_type_current_revision_key,
        encode_edge_type_schema_key,
    };
    let mut schema = EdgeTypeSchema::new(name);
    schema.set_temporal(temporal);
    schema.add_property(PropertyDef::new("context", PropertyType::String).not_null());
    schema
        .resolve_identity(discriminated.then_some("context"))
        .expect("resolve");
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_schema_key(name, 1),
            &schema.to_msgpack().expect("encode"),
        )
        .expect("definition");
    engine
        .put(
            Partition::Schema,
            &encode_edge_type_current_revision_key(name),
            &1u64.to_be_bytes(),
        )
        .expect("pointer");
}

/// The edge-property key of the instance of `(source, target)` under
/// `edge_type` identified by `disc`.
fn instance_key(edge_type: &str, source: u64, target: u64, disc: &[u8]) -> Vec<u8> {
    let mut key = coordinode_core::graph::edge::temporal_edgeprop_pair_prefix(
        edge_type,
        node(source),
        node(target),
    );
    key.extend_from_slice(disc);
    key
}

/// An adjacency posting that does not decode is an error, never an empty
/// set: read as empty it would confirm an observed absence and an unchanged
/// incident set that the stored data may well contradict.
#[test]
fn a_posting_that_does_not_decode_decides_nothing() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("AT", node(1));
    engine
        .put(Partition::Adj, &key, &[0xFF, 0xFE, 0xFD])
        .expect("store bytes that are not a posting");
    let no_points = HashMap::new();
    let view = engine.snapshot();

    let absent = Claim::new(
        ClaimScope::Pair {
            source: node(1),
            target: node(2),
            edge_type: "AT".to_string(),
        },
        ClaimPredicate::PairAdjacency {
            observed: Adjacency::Absent,
        },
        GEN,
    );
    assert!(
        evaluate(&engine, &absent, &[], &no_points, view).is_err(),
        "an unreadable posting confirmed an absence"
    );

    let scan = Claim::new(
        ClaimScope::Incident {
            node: node(1),
            edge_type: "AT".to_string(),
            direction: Direction::Outgoing,
        },
        ClaimPredicate::IncidentSetComplete,
        GEN,
    );
    assert!(
        evaluate(&engine, &scan, &[], &no_points, view).is_err(),
        "an unreadable posting confirmed an unchanged set"
    );
}

/// An upper bound is decided against the neighbours actually stored.
#[test]
fn an_upper_bound_is_decided_against_stored_adjacency() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("OWNS", node(1));

    // No edges: at-most-one holds.
    assert_eq!(
        decide(&engine, &bound(1, CardinalityBound::AtMostOne), &[]),
        Verdict::Holds
    );

    engine
        .merge(Partition::Adj, &key, &encode_add(2))
        .expect("merge");
    assert_eq!(
        decide(&engine, &bound(1, CardinalityBound::AtMostOne), &[]),
        Verdict::Holds,
        "one neighbour is within at-most-one"
    );

    engine
        .merge(Partition::Adj, &key, &encode_add(3))
        .expect("merge");
    assert_eq!(
        decide(&engine, &bound(1, CardinalityBound::AtMostOne), &[]),
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
        decide(&engine, &bound(1, CardinalityBound::AtLeastOne), &[]),
        Verdict::Holds
    );

    // The attempt stages the removal of the only neighbour: the post-state it
    // would leave is what the bound is decided against, not the state before.
    let staged = vec![(key.clone(), AdjOp::Remove(2))];
    assert_eq!(
        decide(&engine, &bound(1, CardinalityBound::AtLeastOne), &staged),
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
        decide(&engine, &bound(5, CardinalityBound::AtLeastOne), &[]),
        Verdict::Broken
    );

    // The same attempt stages the edge: the post-state satisfies the bound,
    // and the intermediate absence is not a violation.
    let staged = vec![(key, AdjOp::Add(6))];
    assert_eq!(
        decide(&engine, &bound(5, CardinalityBound::AtLeastOne), &staged),
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
        decide(
            &engine,
            &bound(1, CardinalityBound::AtLeastOne),
            &add_then_remove
        ),
        Verdict::Broken,
        "add then remove leaves the set empty"
    );

    let remove_then_add = vec![(key.clone(), AdjOp::Remove(2)), (key, AdjOp::Add(2))];
    assert_eq!(
        decide(
            &engine,
            &bound(1, CardinalityBound::AtLeastOne),
            &remove_then_add
        ),
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

/// An enumeration is invalidated by a member that joined the set after the
/// scan passed: the very member the scan could not have found, and the one
/// first-committer-wins cannot see either, because the two writes name
/// different keys.
#[test]
fn a_completed_scan_is_broken_by_a_member_that_joined_after_it() {
    let (engine, _d) = engine();
    let key = encode_adj_key_forward("OWNS", node(1));
    engine
        .merge(Partition::Adj, &key, &encode_add(2))
        .expect("merge");

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
        Verdict::Holds,
        "nothing joined or left since the scan"
    );

    let view = engine.snapshot();
    engine
        .merge(Partition::Adj, &key, &encode_add(3))
        .expect("merge");
    let no_points = HashMap::new();
    assert_eq!(
        evaluate(&engine, &scan, &[], &no_points, view).expect("evaluate"),
        Verdict::Broken,
        "a member the scan never saw is exactly what completeness excludes"
    );
}

/// A cleanup condition was evaluated against one version of the record, so a
/// renewal that rewrites it invalidates the condition whatever it wrote.
#[test]
fn a_cleanup_condition_is_broken_by_a_renewal_of_the_record() {
    let (engine, _d) = engine();
    let key = coordinode_core::graph::node::encode_node_key(engine.node_shard(), node(42));
    engine.put(Partition::Node, &key, b"v1").expect("put");
    let observed = engine.snapshot();

    let cleanup = Claim::new(
        ClaimScope::Record(key.clone()),
        ClaimPredicate::CleanupCondition {
            observed_version: observed,
        },
        GEN,
    );
    assert_eq!(
        decide(&engine, &cleanup, &[]),
        Verdict::Holds,
        "the record still stands as the condition read it"
    );

    engine.put(Partition::Node, &key, b"v2").expect("touch");
    assert_eq!(
        decide(&engine, &cleanup, &[]),
        Verdict::Broken,
        "an ID-only later deletion would have missed this"
    );
}

/// Store `schema` as `label`'s current schema, the way DDL publishes it.
fn store_label_schema(
    engine: &StorageEngine,
    schema: &coordinode_core::schema::definition::LabelSchema,
) {
    use coordinode_core::schema::definition::{
        encode_label_current_revision_key, encode_label_schema_key,
    };
    engine
        .put(
            Partition::Schema,
            &encode_label_schema_key(&schema.name, schema.schema_revision),
            &schema.to_msgpack().expect("encode schema"),
        )
        .expect("schema body");
    engine
        .put(
            Partition::Schema,
            &encode_label_current_revision_key(&schema.name),
            &schema.schema_revision.to_be_bytes(),
        )
        .expect("schema pointer");
}

fn schema_claim(label: &str, predicate: ClaimPredicate) -> Claim {
    Claim::new(ClaimScope::LabelSchema(label.to_string()), predicate, GEN)
}

/// A write validated under the schema it read stays admissible only while
/// that schema is in force: a new revision breaks it, and so does a property
/// edited in place at the same revision.
#[test]
fn a_schema_read_is_broken_by_any_change_of_the_schema() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let (engine, _d) = engine();
    let mut schema = LabelSchema::new_node_id("Doc");
    store_label_schema(&engine, &schema);
    let read = schema_claim(
        "Doc",
        ClaimPredicate::SchemaRead {
            revision: schema.schema_revision,
        },
    );
    let no_points = HashMap::new();
    let view = engine.snapshot();
    let at_view =
        |engine: &StorageEngine| evaluate(engine, &read, &[], &no_points, view).expect("evaluate");
    assert_eq!(at_view(&engine), Verdict::Holds);

    schema.add_property(PropertyDef::new("title", PropertyType::String).not_null());
    engine
        .put(
            Partition::Schema,
            &coordinode_core::schema::definition::encode_label_schema_key(
                "Doc",
                schema.schema_revision,
            ),
            &schema.to_msgpack().expect("encode"),
        )
        .expect("edit in place");
    assert_eq!(
        at_view(&engine),
        Verdict::Broken,
        "a property edited at the same revision changed what the write was validated against"
    );
}

/// A label that had no schema when the write read it gains one: the write
/// was validated against nothing the new schema promises.
#[test]
fn a_schema_read_of_an_undeclared_label_is_broken_by_its_declaration() {
    use coordinode_core::schema::definition::LabelSchema;

    let (engine, _d) = engine();
    let read = schema_claim("Doc", ClaimPredicate::SchemaRead { revision: 0 });
    let no_points = HashMap::new();
    let view = engine.snapshot();
    assert_eq!(
        evaluate(&engine, &read, &[], &no_points, view).expect("evaluate"),
        Verdict::Holds
    );
    store_label_schema(&engine, &LabelSchema::new_node_id("Doc"));
    assert_eq!(
        evaluate(&engine, &read, &[], &no_points, view).expect("evaluate"),
        Verdict::Broken
    );
}

/// An activation answers for every stored node of its primary label: one that
/// breaks the new schema refuses it, nodes of other labels do not count, and
/// the schema it checks against is the one the attempt stages.
#[test]
fn an_activation_is_decided_by_the_stored_nodes_of_its_label() {
    use coordinode_core::graph::node::{NodeRecord, encode_node_key};
    use coordinode_core::schema::definition::{LabelSchema, SchemaMode, encode_label_schema_key};

    let (engine, _d) = engine();
    let mut staged_schema = LabelSchema::new_node_id("Doc");
    staged_schema.set_mode(SchemaMode::Strict);
    staged_schema.schema_revision = 2;
    let mut points = HashMap::new();
    points.insert(
        (
            Partition::Schema,
            encode_label_schema_key("Doc", staged_schema.schema_revision),
        ),
        Some(staged_schema.to_msgpack().expect("encode")),
    );
    let activation = schema_claim("Doc", ClaimPredicate::SchemaActivated { revision: 2 });
    let decide_with = |engine: &StorageEngine, points: &HashMap<_, _>| {
        evaluate(engine, &activation, &[], points, engine.snapshot()).expect("evaluate")
    };

    // A node with no properties satisfies STRICT; one of another label with
    // an undeclared property is not this label's concern.
    let bare = NodeRecord::with_labels(vec!["Doc".to_string()]);
    engine
        .put(
            Partition::Node,
            &encode_node_key(0, node(1)),
            &bare.to_msgpack().expect("encode"),
        )
        .expect("bare node");
    let mut other = NodeRecord::with_labels(vec!["User".to_string(), "Doc".to_string()]);
    other.set_extra("x", coordinode_core::graph::types::Value::Int(1));
    engine
        .put(
            Partition::Node,
            &encode_node_key(0, node(2)),
            &other.to_msgpack().expect("encode"),
        )
        .expect("node of another primary label");
    assert_eq!(decide_with(&engine, &points), Verdict::Holds);

    // An undeclared property on a node of the label breaks STRICT.
    let mut loose = NodeRecord::with_labels(vec!["Doc".to_string()]);
    loose.set_extra("x", coordinode_core::graph::types::Value::Int(1));
    engine
        .put(
            Partition::Node,
            &encode_node_key(0, node(3)),
            &loose.to_msgpack().expect("encode"),
        )
        .expect("loose node");
    assert_eq!(decide_with(&engine, &points), Verdict::Broken);
}

/// Removing a label's schema admits every node, whatever they hold.
#[test]
fn removing_a_schema_admits_every_node() {
    let (engine, _d) = engine();
    let removal = schema_claim("Doc", ClaimPredicate::SchemaActivated { revision: 0 });
    assert_eq!(decide(&engine, &removal, &[]), Verdict::Holds);
}

/// What this evaluator is not given evidence for, it refuses to decide, and
/// the caller must not read that as a pass.
#[test]
fn a_claim_without_evidence_here_is_undecidable_not_satisfied() {
    let (engine, _d) = engine();

    // A predicate over a scope it was never defined for: an endpoint
    // predicate carries nothing about a storage row, and the pairing has no
    // meaning to decide either way.
    let mismatched = Claim::new(
        ClaimScope::Record(b"node:00:0000002a".to_vec()),
        ClaimPredicate::EndpointAlive,
        GEN,
    );
    assert_eq!(
        decide(&engine, &mismatched, &[]),
        Verdict::Undecidable,
        "an overlap with no evidence behind it is not a pass"
    );

    // A temporal type's bound holds over valid time; a count at one instant
    // decides nothing about it.
    define(&engine, "HELD", false, true);
    engine
        .merge(
            Partition::Adj,
            &encode_adj_key_forward("HELD", node(1)),
            &encode_add(2),
        )
        .expect("merge");
    let temporal = counted(
        1,
        "HELD",
        Direction::Outgoing,
        CardinalityMeasure::EdgeInstances,
        CardinalityBound::AtMostOne,
    );
    assert_eq!(decide(&engine, &temporal, &[]), Verdict::Undecidable);

    // A discriminated pair adjacent with no identity behind it is evidence
    // missing, not a count of zero that would admit anything.
    define(&engine, "KNOWS", true, false);
    engine
        .merge(
            Partition::Adj,
            &encode_adj_key_forward("KNOWS", node(1)),
            &encode_add(2),
        )
        .expect("merge");
    for measure in [
        CardinalityMeasure::EdgeInstances,
        CardinalityMeasure::DistinctNeighbours,
    ] {
        let orphan = counted(
            1,
            "KNOWS",
            Direction::Outgoing,
            measure,
            CardinalityBound::AtLeastOne,
        );
        assert_eq!(decide(&engine, &orphan, &[]), Verdict::Undecidable);
    }
}

/// An ordinary edge with no properties is an instance: a single-edge pair is
/// counted from its adjacency, not from a property row it does not have.
#[test]
fn a_property_less_single_edge_is_an_instance() {
    let (engine, _d) = engine();
    define(&engine, "OWNS", false, false);
    engine
        .merge(
            Partition::Adj,
            &encode_adj_key_forward("OWNS", node(1)),
            &encode_add(2),
        )
        .expect("merge");
    for measure in [
        CardinalityMeasure::EdgeInstances,
        CardinalityMeasure::DistinctNeighbours,
    ] {
        let exactly_one = counted(
            1,
            "OWNS",
            Direction::Outgoing,
            measure,
            CardinalityBound::ExactlyOne,
        );
        assert_eq!(
            decide(&engine, &exactly_one, &[]),
            Verdict::Holds,
            "{measure:?}"
        );
    }
}

/// The incoming side is counted like the outgoing one: from the target's
/// reverse adjacency to each source, and through each source's pair entries
/// for the identities.
#[test]
fn an_incoming_scope_counts_both_measures() {
    let (engine, _d) = engine();
    define(&engine, "KNOWS", true, false);
    let reverse = encode_adj_key_reverse("KNOWS", node(9));
    for source in [1u64, 2] {
        engine
            .merge(Partition::Adj, &reverse, &encode_add(source))
            .expect("merge");
    }
    for (source, disc) in [(1u64, b"work".as_slice()), (1, b"golf"), (2, b"work")] {
        engine
            .put(
                Partition::EdgeProp,
                &instance_key("KNOWS", source, 9, disc),
                b"props",
            )
            .expect("put");
    }
    let incoming = |measure, bound| counted(9, "KNOWS", Direction::Incoming, measure, bound);
    assert_eq!(
        decide(
            &engine,
            &incoming(
                CardinalityMeasure::DistinctNeighbours,
                CardinalityBound::AtMostOne
            ),
            &[]
        ),
        Verdict::Broken,
        "two sources"
    );
    assert_eq!(
        decide(
            &engine,
            &incoming(
                CardinalityMeasure::EdgeInstances,
                CardinalityBound::AtLeastOne
            ),
            &[]
        ),
        Verdict::Holds
    );
    // Three identities: at most one fails on instances as well.
    assert_eq!(
        decide(
            &engine,
            &incoming(
                CardinalityMeasure::EdgeInstances,
                CardinalityBound::AtMostOne
            ),
            &[]
        ),
        Verdict::Broken
    );
}

/// The two measures count different things on the same adjacency: two
/// discriminated instances to one neighbour are two identities and one
/// neighbour, and a claim for one must not be answered with the other's
/// number.
#[test]
fn the_two_measures_count_different_things_on_one_adjacency() {
    let (engine, _d) = engine();
    define(&engine, "KNOWS", true, false);
    let adj = encode_adj_key_forward("KNOWS", node(1));
    engine
        .merge(Partition::Adj, &adj, &encode_add(2))
        .expect("merge");

    // Two discriminated entries for the one pair: two logical identities.
    for disc in [b"work".as_slice(), b"clge"] {
        engine
            .put(
                Partition::EdgeProp,
                &instance_key("KNOWS", 1, 2, disc),
                b"props",
            )
            .expect("put");
    }

    let neighbours = counted(
        1,
        "KNOWS",
        Direction::Outgoing,
        CardinalityMeasure::DistinctNeighbours,
        CardinalityBound::AtMostOne,
    );
    let instances = counted(
        1,
        "KNOWS",
        Direction::Outgoing,
        CardinalityMeasure::EdgeInstances,
        CardinalityBound::AtMostOne,
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
    define(&engine, "KNOWS", true, false);
    let adj = encode_adj_key_forward("KNOWS", node(1));
    engine
        .merge(Partition::Adj, &adj, &encode_add(2))
        .expect("merge");

    let entry = |disc: &[u8]| instance_key("KNOWS", 1, 2, disc);

    let committed = entry(b"work");
    engine
        .put(Partition::EdgeProp, &committed, b"props")
        .expect("put");

    let at_most_one = counted(
        1,
        "KNOWS",
        Direction::Outgoing,
        CardinalityMeasure::EdgeInstances,
        CardinalityBound::AtMostOne,
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
        decide(&engine, &bound(1, CardinalityBound::AtLeastOne), &staged),
        Verdict::Holds,
        "another scope's staged removal is not this scope's"
    );
    let _ = encode_remove(0);
}

mod unique_while_building {
    use super::*;
    use coordinode_core::graph::node::{NodeRecord, encode_node_key};
    use coordinode_core::graph::types::Value;
    use coordinode_core::index::derive::{IndexInterpretation, KEY_CODEC, PropertyRef, tuples};
    use coordinode_core::index::identity::GenerationId;
    use coordinode_core::txn::invariant::UncoveredSource;

    const SHARD: u16 = 1;
    const EMAIL: u32 = 7;

    fn interpretation() -> IndexInterpretation {
        IndexInterpretation {
            codec: KEY_CODEC,
            generation: GenerationId::from_raw(3),
            unique: true,
            sparse: false,
            properties: vec![PropertyRef {
                field: Some(EMAIL),
                name: "email".into(),
            }],
            filter: None,
        }
    }

    fn email(v: &str) -> Value {
        Value::String(v.into())
    }

    fn record(label: &str, value: Value) -> Vec<u8> {
        let mut record = NodeRecord::new(label);
        record.set(EMAIL, value);
        record.to_msgpack().expect("encode")
    }

    fn store(engine: &StorageEngine, id: u64, label: &str, value: Value) {
        engine
            .put(
                Partition::Node,
                &encode_node_key(SHARD, node(id)),
                &record(label, value),
            )
            .expect("put node");
    }

    fn claim(claimant: u64, value: Value, covered: Option<u64>, limit: u64) -> Claim {
        let interpretation = interpretation();
        let tuple = tuples(&[value]).pop().expect("one tuple");
        Claim::new(
            ClaimScope::UniqueValue {
                generation: interpretation.generation,
                tuple,
            },
            ClaimPredicate::UniqueHolder {
                node: node(claimant),
                uncovered: Some(Box::new(UncoveredSource {
                    shard_id: SHARD,
                    label: "User".into(),
                    interpretation,
                    covered_through: covered.map(|id| encode_node_key(SHARD, node(id))),
                    read_limit: limit,
                })),
            },
            GEN,
        )
    }

    fn decide_with(
        engine: &StorageEngine,
        claim: &Claim,
        points: &HashMap<(Partition, Vec<u8>), Option<Vec<u8>>>,
        deltas: &[(Vec<u8>, Vec<u8>)],
    ) -> Verdict {
        Evaluation::new(engine, &[], points, engine.snapshot())
            .with_node_deltas(deltas)
            .decide(claim)
            .expect("decide")
    }

    /// A stored node the build has not reached holds the value: the claim is
    /// refused naming it and the values it holds, although no entry exists.
    #[test]
    fn a_value_an_unreached_node_holds_is_held_by_it() {
        let (engine, _d) = engine();
        store(&engine, 1, "User", email("a@x"));

        assert_eq!(
            decide(&engine, &claim(2, email("a@x"), None, 1_000), &[]),
            Verdict::HeldBy {
                holder: node(1),
                values: vec![email("a@x")],
            }
        );
    }

    /// A free value holds; the claimant's own node, a node of another label
    /// and a node the build already covered do not count.
    #[test]
    fn the_claimant_other_labels_and_covered_nodes_do_not_hold() {
        let (engine, _d) = engine();
        store(&engine, 1, "User", email("a@x"));
        store(&engine, 3, "Admin", email("c@x"));

        for (claimant, value, covered) in [
            (2, "b@x", None),
            (1, "a@x", None),
            (2, "c@x", None),
            // Node 1 holds it, but its entry is the build's to commit, and
            // the attempt meets that entry on the key both write.
            (2, "a@x", Some(1)),
        ] {
            assert_eq!(
                decide(&engine, &claim(claimant, email(value), covered, 1_000), &[]),
                Verdict::Holds,
                "{claimant} taking {value} covered through {covered:?}"
            );
        }
    }

    /// A list is indexed by each element, so a stored list holding the value
    /// holds it.
    #[test]
    fn an_element_of_a_stored_list_is_held() {
        let (engine, _d) = engine();
        store(
            &engine,
            1,
            "User",
            Value::Array(vec![email("x"), email("y")]),
        );

        assert!(matches!(
            decide(&engine, &claim(2, email("y"), None, 1_000), &[]),
            Verdict::HeldBy { holder, .. } if holder == node(1)
        ));
    }

    /// Rows the attempt deletes or rewrites are judged as it leaves them, and
    /// rows it changes through document deltas are left to the entries it
    /// maintained for them, not to their committed bytes.
    #[test]
    fn rows_the_attempt_changes_are_judged_by_its_post_state() {
        let (engine, _d) = engine();
        store(&engine, 1, "User", email("a@x"));
        store(&engine, 3, "User", email("c@x"));
        store(&engine, 5, "User", email("e@x"));
        let mut points = HashMap::new();
        points.insert((Partition::Node, encode_node_key(SHARD, node(1))), None);
        points.insert(
            (Partition::Node, encode_node_key(SHARD, node(3))),
            Some(record("User", email("d@x"))),
        );
        let deltas = vec![(encode_node_key(SHARD, node(5)), Vec::new())];

        for value in ["a@x", "c@x", "e@x"] {
            assert_eq!(
                decide_with(
                    &engine,
                    &claim(9, email(value), None, 1_000),
                    &points,
                    &deltas
                ),
                Verdict::Holds,
                "{value}"
            );
        }
        assert!(matches!(
            decide_with(&engine, &claim(9, email("d@x"), None, 1_000), &points, &deltas),
            Verdict::HeldBy { holder, .. } if holder == node(3)
        ));
    }

    /// A temporal node holds the values of each of its versions: a value
    /// only an older version has is held, by the node.
    #[test]
    fn a_version_of_a_temporal_node_holds_its_value() {
        use coordinode_core::graph::node::encode_temporal_node_key;
        let (engine, _d) = engine();
        for (valid_from, value) in [(100, "old@x"), (200, "new@x")] {
            engine
                .put(
                    Partition::Node,
                    &encode_temporal_node_key(SHARD, node(4), valid_from),
                    &record("User", email(value)),
                )
                .expect("put version");
        }

        for value in ["old@x", "new@x"] {
            assert_eq!(
                decide(&engine, &claim(9, email(value), None, 1_000), &[]),
                Verdict::HeldBy {
                    holder: node(4),
                    values: vec![email(value)],
                },
                "{value}"
            );
        }
        assert_eq!(
            decide(&engine, &claim(4, email("old@x"), None, 1_000), &[]),
            Verdict::Holds,
            "the node's own versions"
        );
    }

    /// A stored row that is not a node record refuses the decision rather
    /// than being passed over: a value it may hold cannot be proved free.
    #[test]
    fn an_undecodable_row_decides_nothing() {
        let (engine, _d) = engine();
        engine
            .put(
                Partition::Node,
                &encode_node_key(SHARD, node(1)),
                &[0xC1, 0xFF],
            )
            .expect("put bytes");

        assert!(
            evaluate(
                &engine,
                &claim(2, email("a@x"), None, 1_000),
                &[],
                &HashMap::new(),
                engine.snapshot()
            )
            .is_err()
        );
    }

    /// A decision that would read more stored rows than the limit is left
    /// unresolved, never called a duplicate or a pass.
    #[test]
    fn a_decision_past_the_limit_is_over_limit() {
        let (engine, _d) = engine();
        for id in 1..=3 {
            store(&engine, id, "User", email(&format!("{id}@x")));
        }

        assert_eq!(
            decide(&engine, &claim(9, email("free@x"), None, 2), &[]),
            Verdict::OverLimit { limit: 2 }
        );
        assert_eq!(
            decide(&engine, &claim(9, email("free@x"), None, 3), &[]),
            Verdict::Holds
        );
        // Rows the build covered are not read, so the same limit suffices.
        assert_eq!(
            decide(&engine, &claim(9, email("free@x"), Some(2), 1), &[]),
            Verdict::Holds
        );
    }
}
