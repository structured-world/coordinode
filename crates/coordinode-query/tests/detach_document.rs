//! Integration tests for `DETACH DOCUMENT` (R167).
//!
//! Verifies the full pipeline (parse → plan → execute → merge flush) against
//! real CoordiNode storage:
//!   1. Basic detach: nested DOCUMENT property becomes a separate node + edge.
//!   2. Nested path: `n.meta.shipping` — multi-segment property path.
//!   3. Edge transfer: `TRANSFER EDGES WHERE type(r) IN [...]` re-points
//!      selected edges from the source node onto the new target.
//!   4. Error cases: missing property, non-document value, unknown node.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse, PostingList};
use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{encode_node_key, NodeId, NodeIdAllocator, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_query::cypher::parse;
use coordinode_query::executor::{execute, AdaptiveConfig, ExecutionContext, Row, WriteStats};
use coordinode_query::planner::build_logical_plan;
use coordinode_storage::engine::config::StorageConfig;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

// ── helpers (copy of query_integration patterns, scoped to this file) ────

fn make_test_ctx<'a>(
    engine: &'a StorageEngine,
    interner: &'a mut FieldInterner,
    allocator: &'a NodeIdAllocator,
) -> ExecutionContext<'a> {
    ExecutionContext {
        engine,
        interner,
        id_allocator: allocator,
        shard_id: 1,
        adaptive: AdaptiveConfig::default(),
        snapshot_ts: None,
        retention_window_us: 7 * 24 * 3600 * 1_000_000,
        warnings: Vec::new(),
        write_stats: WriteStats::default(),
        text_index: None,
        text_index_registry: None,
        vector_index_registry: None,
        btree_index_registry: None,
        vector_loader: None,
        mvcc_oracle: None,
        mvcc_read_ts: coordinode_core::txn::timestamp::Timestamp::ZERO,
        mvcc_write_buffer: HashMap::new(),
        procedure_ctx: None,
        mvcc_read_set: std::collections::HashSet::new(),
        vector_consistency: coordinode_core::graph::types::VectorConsistencyMode::default(),
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: None,
        proposal_id_gen: None,
        read_concern: coordinode_core::txn::read_concern::ReadConcernLevel::Local,
        write_concern: coordinode_core::txn::write_concern::WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        merge_adj_adds: HashMap::new(),
        merge_adj_removes: HashMap::new(),
        mvcc_snapshot: None,
        adj_snapshot: None,
        merge_node_deltas: Vec::new(),
        correlated_row: None,
        feedback_cache: None,
        schema_label_cache: HashMap::new(),
        applied_watermark: None,
        params: HashMap::new(),
    }
}

fn test_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::new(dir);
    StorageEngine::open(&config).expect("open engine")
}

fn run(
    query: &str,
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    allocator: &NodeIdAllocator,
) -> Vec<Row> {
    let ast = parse(query).unwrap_or_else(|e| panic!("parse error: {e}\nquery: {query}"));
    let plan = build_logical_plan(&ast).unwrap_or_else(|e| panic!("plan error: {e}"));
    let mut ctx = make_test_ctx(engine, interner, allocator);
    execute(&plan, &mut ctx).unwrap_or_else(|e| panic!("execute error: {e}\nquery: {query}"))
}

fn run_err(
    query: &str,
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    allocator: &NodeIdAllocator,
) -> coordinode_query::executor::ExecutionError {
    let ast = parse(query).unwrap_or_else(|e| panic!("parse error: {e}"));
    let plan = build_logical_plan(&ast).unwrap_or_else(|e| panic!("plan error: {e}"));
    let mut ctx = make_test_ctx(engine, interner, allocator);
    execute(&plan, &mut ctx).expect_err("expected execute to fail")
}

/// Insert a User node (id=1) with a nested address document and, optionally,
/// other properties. Returns the node's shard_id/node_id.
fn insert_user_with_address(
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    id: u64,
    name: &str,
    address: rmpv::Value,
) {
    let mut record = NodeRecord::new("User");
    let name_fid = interner.intern("name");
    let addr_fid = interner.intern("address");
    record.set(name_fid, Value::String(name.into()));
    record.set(addr_fid, Value::Document(address));
    let key = encode_node_key(1, NodeId::from_raw(id));
    let bytes = record.to_msgpack().expect("serialize");
    engine.put(Partition::Node, &key, &bytes).expect("put node");
}

fn register_schema_edge_type(engine: &StorageEngine, edge_type: &str) {
    const PREFIX: &[u8] = b"schema:edge_type:";
    let mut key = PREFIX.to_vec();
    key.extend_from_slice(edge_type.as_bytes());
    engine
        .put(Partition::Schema, &key, b"")
        .expect("put schema edge");
}

fn insert_edge_direct(engine: &StorageEngine, edge_type: &str, source_id: u64, target_id: u64) {
    let fwd_key = encode_adj_key_forward(edge_type, NodeId::from_raw(source_id));
    let mut fwd_list = match engine.get(Partition::Adj, &fwd_key).expect("get") {
        Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
        None => PostingList::new(),
    };
    fwd_list.insert(target_id);
    engine
        .put(Partition::Adj, &fwd_key, &fwd_list.to_bytes().expect("ser"))
        .expect("put fwd");

    let rev_key = encode_adj_key_reverse(edge_type, NodeId::from_raw(target_id));
    let mut rev_list = match engine.get(Partition::Adj, &rev_key).expect("get") {
        Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
        None => PostingList::new(),
    };
    rev_list.insert(source_id);
    engine
        .put(Partition::Adj, &rev_key, &rev_list.to_bytes().expect("ser"))
        .expect("put rev");

    register_schema_edge_type(engine, edge_type);
}

fn adj_contains(engine: &StorageEngine, edge_type: &str, src: u64, tgt: u64) -> bool {
    let fwd = encode_adj_key_forward(edge_type, NodeId::from_raw(src));
    match engine.get(Partition::Adj, &fwd).expect("get fwd") {
        Some(bytes) => PostingList::from_bytes(&bytes)
            .expect("decode")
            .iter()
            .any(|u| u == tgt),
        None => false,
    }
}

fn read_address_prop(
    engine: &StorageEngine,
    interner: &FieldInterner,
    node_id: u64,
) -> Option<Value> {
    let key = encode_node_key(1, NodeId::from_raw(node_id));
    let bytes = engine.get(Partition::Node, &key).expect("get node")?;
    let record = NodeRecord::from_msgpack(&bytes).expect("decode");
    let fid = interner.lookup("address")?;
    record.props.get(&fid).cloned()
}

// ── tests ────────────────────────────────────────────────────────────────

/// Basic detach: address map becomes an Address node + edge.
#[test]
fn detach_document_basic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let address_doc = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("city".into()),
            rmpv::Value::String("Prague".into()),
        ),
        (
            rmpv::Value::String("zip".into()),
            rmpv::Value::String("11000".into()),
        ),
    ]);
    insert_user_with_address(&engine, &mut interner, 1, "Alice", address_doc.clone());

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let results = run(
        "MATCH (n:User) \
         DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n) \
         RETURN a, a.city AS city, a.zip AS zip",
        &engine,
        &mut interner,
        &allocator,
    );

    // One row with the promoted properties.
    assert_eq!(results.len(), 1, "expected one result row");
    assert_eq!(
        results[0].get("city"),
        Some(&Value::String("Prague".into())),
        "city must come from the promoted document"
    );
    assert_eq!(results[0].get("zip"), Some(&Value::String("11000".into())));

    // The source property must be removed from the User node after flush.
    // NB: lsm-tree merge operators apply deltas on read — `record.props` no
    // longer contains the `address` field.
    let addr_after = read_address_prop(&engine, &interner, 1);
    assert!(
        addr_after.is_none(),
        "address must be removed from User after DETACH, got: {addr_after:?}"
    );

    // Edge (a:Address)-[:HAS_ADDRESS]->(n) must exist.
    // New node id is allocator's next after 1000 = 1001; verify via RETURN.
    let new_id = match results[0].get("a") {
        Some(Value::Int(i)) => *i as u64,
        other => panic!("expected `a` binding, got {other:?}"),
    };
    assert!(
        adj_contains(&engine, "HAS_ADDRESS", new_id, 1),
        "edge HAS_ADDRESS: {new_id} → 1 must exist after DETACH"
    );
}

/// Nested path: detach `n.meta.shipping` — multi-segment property path.
#[test]
fn detach_document_nested_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Build: { meta: { shipping: { city: "Prague" }, billing: { city: "Brno" } } }
    let shipping = rmpv::Value::Map(vec![(
        rmpv::Value::String("city".into()),
        rmpv::Value::String("Prague".into()),
    )]);
    let billing = rmpv::Value::Map(vec![(
        rmpv::Value::String("city".into()),
        rmpv::Value::String("Brno".into()),
    )]);
    let meta = rmpv::Value::Map(vec![
        (rmpv::Value::String("shipping".into()), shipping),
        (rmpv::Value::String("billing".into()), billing.clone()),
    ]);

    let mut record = NodeRecord::new("User");
    let fid = interner.intern("meta");
    record.set(fid, Value::Document(meta.clone()));
    let key = encode_node_key(1, NodeId::from_raw(1));
    engine
        .put(Partition::Node, &key, &record.to_msgpack().unwrap())
        .unwrap();

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let results = run(
        "MATCH (n:User) \
         DETACH DOCUMENT n.meta.shipping AS (s:ShippingAddress)-[:HAS_SHIPPING]->(n) \
         RETURN s.city AS city",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("city"),
        Some(&Value::String("Prague".into()))
    );

    // meta.billing must survive; meta.shipping must be removed.
    let meta_fid = interner.lookup("meta").unwrap();
    let bytes = engine
        .get(Partition::Node, &key)
        .unwrap()
        .expect("node still present");
    let after = NodeRecord::from_msgpack(&bytes).unwrap();
    let Value::Document(meta_after) = after.props.get(&meta_fid).cloned().unwrap() else {
        panic!("meta must still be a Document");
    };
    let rmpv::Value::Map(entries) = meta_after else {
        panic!("meta must still be a map");
    };
    let has_shipping = entries.iter().any(|(k, _)| k.as_str() == Some("shipping"));
    let billing_entry = entries
        .iter()
        .find(|(k, _)| k.as_str() == Some("billing"))
        .map(|(_, v)| v.clone());
    assert!(!has_shipping, "meta.shipping must be deleted");
    assert_eq!(
        billing_entry,
        Some(billing),
        "meta.billing must be preserved"
    );
}

/// TRANSFER EDGES: `type(r) IN [...]` re-points matching edges onto the new node.
#[test]
fn detach_document_transfers_matching_edges() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // User(1) with an address document, plus two pre-existing outgoing edges
    // of different types — only SHIPS_TO should be transferred.
    let address_doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("city".into()),
        rmpv::Value::String("Prague".into()),
    )]);
    insert_user_with_address(&engine, &mut interner, 1, "Alice", address_doc);

    // Peer nodes (ids 100, 200) — content irrelevant.
    insert_user_with_address(
        &engine,
        &mut interner,
        100,
        "PeerShipping",
        rmpv::Value::Map(vec![]),
    );
    insert_user_with_address(
        &engine,
        &mut interner,
        200,
        "PeerKnows",
        rmpv::Value::Map(vec![]),
    );

    // (1)-[:SHIPS_TO]->(100) and (1)-[:KNOWS]->(200)
    insert_edge_direct(&engine, "SHIPS_TO", 1, 100);
    insert_edge_direct(&engine, "KNOWS", 1, 200);

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let results = run(
        "MATCH (n:User {name: 'Alice'}) \
         DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n) \
         TRANSFER EDGES ON n TO a WHERE type(r) IN ['SHIPS_TO'] \
         RETURN a, a.city AS city",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(results.len(), 1);
    let new_id = match results[0].get("a") {
        Some(Value::Int(i)) => *i as u64,
        other => panic!("expected `a` binding, got {other:?}"),
    };

    // SHIPS_TO must have moved 1 → new_id. KNOWS must still be on 1.
    assert!(
        !adj_contains(&engine, "SHIPS_TO", 1, 100),
        "SHIPS_TO(1 → 100) must be removed after transfer"
    );
    assert!(
        adj_contains(&engine, "SHIPS_TO", new_id, 100),
        "SHIPS_TO({new_id} → 100) must exist after transfer"
    );
    assert!(
        adj_contains(&engine, "KNOWS", 1, 200),
        "KNOWS(1 → 200) must remain on source (not transferred)"
    );
    assert!(
        !adj_contains(&engine, "KNOWS", new_id, 200),
        "KNOWS must NOT move to target"
    );

    // HAS_ADDRESS edge (new_id → 1) must exist.
    assert!(
        adj_contains(&engine, "HAS_ADDRESS", new_id, 1),
        "HAS_ADDRESS edge must exist after DETACH"
    );
}

/// Error case: property does not exist.
#[test]
fn detach_document_missing_property_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // User with no `address` field at all.
    let mut record = NodeRecord::new("User");
    let fid = interner.intern("name");
    record.set(fid, Value::String("Alice".into()));
    let key = encode_node_key(1, NodeId::from_raw(1));
    engine
        .put(Partition::Node, &key, &record.to_msgpack().unwrap())
        .unwrap();

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let err = run_err(
        "MATCH (n:User) \
         DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n)",
        &engine,
        &mut interner,
        &allocator,
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("address") && msg.contains("not found"),
        "error should mention missing property: {msg}"
    );
}

/// Error case: property exists but is not a DOCUMENT/MAP.
#[test]
fn detach_document_non_document_value_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // `address` is a plain string — not a document.
    let mut record = NodeRecord::new("User");
    let fid = interner.intern("address");
    record.set(fid, Value::String("just a string".into()));
    let key = encode_node_key(1, NodeId::from_raw(1));
    engine
        .put(Partition::Node, &key, &record.to_msgpack().unwrap())
        .unwrap();

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let err = run_err(
        "MATCH (n:User) \
         DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n)",
        &engine,
        &mut interner,
        &allocator,
    );
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("not a document") || msg.contains("not a map"),
        "error should indicate non-document value: {msg}"
    );
}

/// Default edge type derivation: `HAS_<UPPER_SNAKE(prop)>`.
///
/// No explicit `[:TYPE]` in the AS pattern → executor fills in `HAS_ADDRESS`.
/// Note: grammar currently requires the edge type to be explicit (pest PEG
/// with `rel_type_list`), so this test asserts the derivation path when the
/// grammar omits it. The grammar allows `-[]->` (empty rel_detail) — verify.
#[test]
fn detach_document_default_edge_type_from_camel_case() {
    // Uses the builder-level derivation through a property named `sensorConfig`
    // → expected edge type `HAS_SENSOR_CONFIG`. We plumb this via the parser
    // path that *does* emit an empty edge type (an explicit anonymous rel).
    // Grammar: `-[]->` parses to rel_detail with empty rel_types.
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let cfg = rmpv::Value::Map(vec![(
        rmpv::Value::String("firmware".into()),
        rmpv::Value::String("2.1".into()),
    )]);
    let mut record = NodeRecord::new("Device");
    let fid = interner.intern("sensorConfig");
    record.set(fid, Value::Document(cfg));
    let key = encode_node_key(1, NodeId::from_raw(1));
    engine
        .put(Partition::Node, &key, &record.to_msgpack().unwrap())
        .unwrap();

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let results = run(
        "MATCH (n:Device) \
         DETACH DOCUMENT n.sensorConfig AS (c:SensorConfig)-[]->(n) \
         RETURN c, c.firmware AS fw",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("fw"), Some(&Value::String("2.1".into())));

    let new_id = match results[0].get("c") {
        Some(Value::Int(i)) => *i as u64,
        other => panic!("expected `c` binding, got {other:?}"),
    };
    // Expect HAS_SENSOR_CONFIG derived from camelCase property.
    assert!(
        adj_contains(&engine, "HAS_SENSOR_CONFIG", new_id, 1),
        "edge HAS_SENSOR_CONFIG({new_id} → 1) must exist"
    );
}
