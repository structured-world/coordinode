//! Integration tests for `ATTACH DOCUMENT` (R168).
//!
//! Verifies the full pipeline (parse → plan → execute → merge flush) against
//! real CoordiNode storage:
//!   1. Basic attach: a graph node becomes a nested DOCUMENT property.
//!   2. Edge transfer: `TRANSFER EDGES` re-points selected edges before the
//!      source node is deleted.
//!   3. `ON CONFLICT REPLACE` overwrites an existing target property.
//!   4. `ON REMAINING FAIL` aborts when untransferred edges remain.
//!   5. Error case: target property already exists (default — no REPLACE).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeId, NodeIdAllocator, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_query::cypher::parse;
use coordinode_query::executor::{execute, AdaptiveConfig, ExecutionContext, Row, WriteStats};
use coordinode_query::planner::build_logical_plan;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

// ── helpers ──────────────────────────────────────────────────────────────

fn make_test_ctx<'a>(
    engine: &'a StorageEngine,
    interner: &'a mut FieldInterner,
    allocator: &'a NodeIdAllocator,
) -> ExecutionContext<'a> {
    ExecutionContext {
        engine,
        engine_arc: None,
        interner,
        id_allocator: allocator,
        shard_id: 1,
        adaptive: AdaptiveConfig::default(),
        dedup_varlen_targets: false,
        snapshot_ts: None,
        retention_window_us: 7 * 24 * 3600 * 1_000_000,
        warnings: Vec::new(),
        write_stats: WriteStats::default(),
        text_index: None,
        text_index_registry: None,
        vector_index_registry: None,
        btree_index_registry: None,
        extensions: None,
        vector_loader: None,
        mvcc_oracle: None,
        mvcc_read_ts: coordinode_core::txn::timestamp::Timestamp::ZERO,
        txn: coordinode_storage::engine::transaction::Transaction::new(
            engine,
            None,
            coordinode_core::txn::timestamp::Timestamp::ZERO,
            None,
        ),
        procedure_ctx: None,
        vector_consistency: coordinode_core::graph::types::VectorConsistencyMode::default(),
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: None,
        proposal_id_gen: None,
        read_concern: coordinode_core::txn::read_concern::ReadConcernLevel::Local,
        write_concern: coordinode_core::txn::write_concern::WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        mvcc_snapshot: None,
        cascade_depth: 0,
        cascade_depth_limit: 10,
        cascade_fire_counts: std::collections::HashMap::new(),
        cascade_fanout_limit: 100,
        cascade_chain: Vec::new(),
        correlated_row: None,
        feedback_cache: None,
        schema_label_cache: HashMap::new(),
        applied_watermark: None,
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        read_timeout: std::time::Duration::from_millis(2000),
        params: HashMap::new(),
        pending_vector_writes: Vec::new(),
    }
}

/// Logic-test fixture — uses `engine_for_logic()` which honours
/// `COORDINODE_TEST_BACKEND` (default: memory). ATTACH DOCUMENT
/// tests exercise pure query semantics, not disk persistence, so
/// memory backend is the right default.
fn test_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
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

fn insert_node(
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    id: u64,
    label: &str,
    props: &[(&str, Value)],
) {
    let mut record = NodeRecord::new(label);
    for (name, value) in props {
        let fid = interner.intern(name);
        record.set(fid, value.clone());
    }
    put_node_committed(engine, 1, id, &record);
}

fn register_schema_edge_type(engine: &StorageEngine, edge_type: &str) {
    let key = coordinode_core::schema::definition::encode_edge_type_schema_key(edge_type, 1);
    engine
        .put(Partition::Schema, &key, b"")
        .expect("put schema edge");
}

fn insert_edge(engine: &StorageEngine, edge_type: &str, source_id: u64, target_id: u64) {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    LocalEdgeStore
        .put_edge(
            &mut txn,
            edge_type,
            NodeId::from_raw(source_id),
            NodeId::from_raw(target_id),
            None,
        )
        .expect("put edge");
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).expect("commit edge");
    register_schema_edge_type(engine, edge_type);
}

fn adj_contains(engine: &StorageEngine, edge_type: &str, src: u64, tgt: u64) -> bool {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
    use coordinode_storage::engine::transaction::Transaction;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    LocalEdgeStore
        .scan_neighbors_out(&txn, edge_type, NodeId::from_raw(src))
        .expect("scan out-neighbors")
        .iter()
        .any(|n| n.as_raw() == tgt)
}

/// Seed a node (committed MVCC transaction).
fn put_node_committed(engine: &StorageEngine, shard: u16, id: u64, record: &NodeRecord) {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    LocalNodeStore
        .put(&mut txn, shard, NodeId::from_raw(id), record)
        .expect("put node");
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).expect("commit node");
}

/// Read a node at the latest committed snapshot.
fn read_node_record(engine: &StorageEngine, shard: u16, id: u64) -> Option<NodeRecord> {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::Transaction;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    LocalNodeStore
        .get(&txn, shard, NodeId::from_raw(id))
        .expect("get node")
}

fn node_exists(engine: &StorageEngine, id: u64) -> bool {
    read_node_record(engine, 1, id).is_some()
}

fn read_prop(
    engine: &StorageEngine,
    interner: &FieldInterner,
    node_id: u64,
    prop: &str,
) -> Option<Value> {
    let record = read_node_record(engine, 1, node_id)?;
    let fid = interner.lookup(prop)?;
    record.props.get(&fid).cloned()
}

// ── tests ────────────────────────────────────────────────────────────────

#[test]
fn attach_document_basic() {
    let fx = test_engine();
    let engine = &fx.engine;
    let mut interner = FieldInterner::new();

    // Address node `a` (id=100) with two scalar properties; User node `u`
    // (id=1) with no `address` yet.
    insert_node(
        engine,
        &mut interner,
        100,
        "Address",
        &[
            ("city", Value::String("Prague".into())),
            ("zip", Value::String("11000".into())),
        ],
    );
    insert_node(
        engine,
        &mut interner,
        1,
        "User",
        &[("name", Value::String("Alice".into()))],
    );
    insert_edge(engine, "HAS_ADDRESS", 100, 1);

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let results = run(
        "ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address RETURN u",
        engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(results.len(), 1, "expected one ATTACH result row");

    // Address node must be gone.
    assert!(
        !node_exists(engine, 100),
        "source Address node should be deleted after ATTACH"
    );
    // Edge must be gone.
    assert!(
        !adj_contains(engine, "HAS_ADDRESS", 100, 1),
        "HAS_ADDRESS edge should be removed after ATTACH"
    );

    // `u.address` must now be a Document with city/zip.
    let addr = read_prop(engine, &interner, 1, "address").expect("address present");
    let Value::Document(rmpv::Value::Map(entries)) = addr else {
        panic!("u.address should be a Document map");
    };
    let city = entries
        .iter()
        .find(|(k, _)| k.as_str() == Some("city"))
        .map(|(_, v)| v.clone());
    let zip = entries
        .iter()
        .find(|(k, _)| k.as_str() == Some("zip"))
        .map(|(_, v)| v.clone());
    assert_eq!(city, Some(rmpv::Value::String("Prague".into())));
    assert_eq!(zip, Some(rmpv::Value::String("11000".into())));
}

#[test]
fn attach_document_transfers_matching_edges() {
    let fx = test_engine();
    let engine = &fx.engine;
    let mut interner = FieldInterner::new();

    // Source Address node (100) has a SHIPS_TO edge to peer(200) that should
    // move onto User(1) during ATTACH.
    insert_node(
        engine,
        &mut interner,
        100,
        "Address",
        &[("city", Value::String("Prague".into()))],
    );
    insert_node(
        engine,
        &mut interner,
        1,
        "User",
        &[("name", Value::String("Alice".into()))],
    );
    insert_node(engine, &mut interner, 200, "Peer", &[]);
    insert_edge(engine, "HAS_ADDRESS", 100, 1);
    insert_edge(engine, "SHIPS_TO", 100, 200);

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    run(
        "ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address \
         TRANSFER EDGES ON a TO u WHERE type(r) = 'SHIPS_TO' \
         RETURN u",
        engine,
        &mut interner,
        &allocator,
    );

    assert!(!node_exists(engine, 100), "Address node should be deleted");
    assert!(
        !adj_contains(engine, "SHIPS_TO", 100, 200),
        "SHIPS_TO(100 → 200) must be removed after transfer"
    );
    assert!(
        adj_contains(engine, "SHIPS_TO", 1, 200),
        "SHIPS_TO(1 → 200) must exist after transfer"
    );
}

#[test]
fn attach_document_conflict_without_replace_errors() {
    let fx = test_engine();
    let engine = &fx.engine;
    let mut interner = FieldInterner::new();

    // User `u` already has an `address` → default (no REPLACE) must error.
    insert_node(
        engine,
        &mut interner,
        100,
        "Address",
        &[("city", Value::String("Prague".into()))],
    );
    insert_node(
        engine,
        &mut interner,
        1,
        "User",
        &[(
            "address",
            Value::String("old address, not a document".into()),
        )],
    );
    insert_edge(engine, "HAS_ADDRESS", 100, 1);

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let err = run_err(
        "ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address",
        engine,
        &mut interner,
        &allocator,
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("already exists") || msg.contains("ON CONFLICT"),
        "error should mention existing property: {msg}"
    );

    // Source must NOT be deleted — the whole txn aborts before writes would apply.
    // (In this executor, mvcc_flush is skipped on error path; verify the node
    // still exists from the engine's perspective.)
    assert!(
        node_exists(engine, 100),
        "source node should survive aborted ATTACH"
    );
}

#[test]
fn attach_document_on_conflict_replace_overwrites() {
    let fx = test_engine();
    let engine = &fx.engine;
    let mut interner = FieldInterner::new();

    insert_node(
        engine,
        &mut interner,
        100,
        "Address",
        &[("city", Value::String("Prague".into()))],
    );
    insert_node(
        engine,
        &mut interner,
        1,
        "User",
        &[(
            "address",
            Value::String("old string (to be replaced)".into()),
        )],
    );
    insert_edge(engine, "HAS_ADDRESS", 100, 1);

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    run(
        "ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address \
         ON CONFLICT REPLACE",
        engine,
        &mut interner,
        &allocator,
    );

    let addr = read_prop(engine, &interner, 1, "address").expect("present");
    let Value::Document(rmpv::Value::Map(entries)) = addr else {
        panic!("u.address should be Document after REPLACE");
    };
    assert!(
        entries
            .iter()
            .any(|(k, v)| k.as_str() == Some("city") && v == &rmpv::Value::String("Prague".into())),
        "REPLACE must overwrite with the new Document"
    );
    assert!(!node_exists(engine, 100));
}

#[test]
fn attach_document_on_remaining_fail_errors() {
    let fx = test_engine();
    let engine = &fx.engine;
    let mut interner = FieldInterner::new();

    // Address has an extra edge (SHIPS_TO → peer) that is NOT transferred →
    // ON REMAINING FAIL must abort.
    insert_node(
        engine,
        &mut interner,
        100,
        "Address",
        &[("city", Value::String("Prague".into()))],
    );
    insert_node(
        engine,
        &mut interner,
        1,
        "User",
        &[("name", Value::String("Alice".into()))],
    );
    insert_node(engine, &mut interner, 200, "Peer", &[]);
    insert_edge(engine, "HAS_ADDRESS", 100, 1);
    insert_edge(engine, "SHIPS_TO", 100, 200);

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let err = run_err(
        "ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address \
         ON REMAINING FAIL",
        engine,
        &mut interner,
        &allocator,
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("REMAINING FAIL") || msg.contains("untransferred"),
        "error should mention remaining edges: {msg}"
    );
}

#[test]
fn attach_document_on_remaining_fail_succeeds_when_all_transferred() {
    let fx = test_engine();
    let engine = &fx.engine;
    let mut interner = FieldInterner::new();

    insert_node(
        engine,
        &mut interner,
        100,
        "Address",
        &[("city", Value::String("Prague".into()))],
    );
    insert_node(
        engine,
        &mut interner,
        1,
        "User",
        &[("name", Value::String("Alice".into()))],
    );
    insert_node(engine, &mut interner, 200, "Peer", &[]);
    insert_edge(engine, "HAS_ADDRESS", 100, 1);
    insert_edge(engine, "SHIPS_TO", 100, 200);

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    run(
        "ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address \
         TRANSFER EDGES ON a TO u WHERE type(r) = 'SHIPS_TO' \
         ON REMAINING FAIL",
        engine,
        &mut interner,
        &allocator,
    );

    // Address deleted, edges moved.
    assert!(!node_exists(engine, 100));
    assert!(adj_contains(engine, "SHIPS_TO", 1, 200));
}
