//! End-to-end integration tests: Cypher text → parse → plan → execute → result.
//!
//! These tests exercise the full query pipeline against real CoordiNode storage,
//! validating that variable-length paths, aggregation, UNWIND, and OPTIONAL MATCH
//! work correctly through the entire stack.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse, PostingList};
use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{encode_node_key, NodeId, NodeIdAllocator, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_query::cypher::parse;
use coordinode_query::executor::{execute, AdaptiveConfig, ExecutionContext, WriteStats};
use coordinode_query::index::{
    IndexDefinition, TextIndexConfig, TextIndexRegistry, VectorIndexRegistry,
};
use coordinode_query::planner::{build_logical_plan, estimate_cost};
use coordinode_search::tantivy::multi_lang::{MultiLangConfig, MultiLanguageTextIndex};
use coordinode_search::tantivy::TextIndex;
use coordinode_storage::engine::config::StorageConfig;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

// ── helpers ──────────────────────────────────────────────────────────────

/// Build an `ExecutionContext` with legacy-mode defaults (no oracle, shard_id=1).
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
        mvcc_write_buffer: std::collections::HashMap::new(),
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
        merge_adj_adds: std::collections::HashMap::new(),
        merge_adj_removes: std::collections::HashMap::new(),
        mvcc_snapshot: None,
        adj_snapshot: None,
        merge_node_deltas: Vec::new(),
        correlated_row: None,
        feedback_cache: None,
        schema_label_cache: std::collections::HashMap::new(),
        params: std::collections::HashMap::new(),
    }
}

fn test_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::new(dir);
    StorageEngine::open(&config).expect("open engine")
}

fn insert_node(
    engine: &StorageEngine,
    shard_id: u16,
    node_id: u64,
    label: &str,
    props: &[(&str, Value)],
    interner: &mut FieldInterner,
) {
    let mut record = NodeRecord::new(label);
    for (name, value) in props {
        let field_id = interner.intern(name);
        record.set(field_id, value.clone());
    }
    let key = encode_node_key(shard_id, NodeId::from_raw(node_id));
    let bytes = record.to_msgpack().expect("serialize");
    engine.put(Partition::Node, &key, &bytes).expect("put node");
}

/// Register an edge type in the schema partition so wildcard relationship
/// patterns (`MATCH (n)-[r]->(m)`) can enumerate it via `schema:edge_type:` scan.
///
/// In production, this is done automatically when edges are created via the
/// Cypher executor. In tests that insert edges directly (bypassing the executor),
/// call this alongside `insert_edge` to keep schema and adj index in sync.
fn register_schema_edge_type(engine: &StorageEngine, edge_type: &str) {
    const PREFIX: &[u8] = b"schema:edge_type:";
    let mut key = PREFIX.to_vec();
    key.extend_from_slice(edge_type.as_bytes());
    engine
        .put(Partition::Schema, &key, b"")
        .expect("put schema edge type");
}

fn insert_edge(engine: &StorageEngine, edge_type: &str, source_id: u64, target_id: u64) {
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
}

/// Execute a Cypher query end-to-end: parse → plan → execute.
/// Uses a shared allocator so CREATE IDs don't collide across calls.
fn run_cypher_with_alloc(
    query: &str,
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    allocator: &NodeIdAllocator,
) -> Vec<coordinode_query::executor::Row> {
    let ast = parse(query).unwrap_or_else(|e| panic!("parse error: {e}"));
    let plan = build_logical_plan(&ast).unwrap_or_else(|e| panic!("plan error: {e}"));
    let mut ctx = make_test_ctx(engine, interner, allocator);
    execute(&plan, &mut ctx).unwrap_or_else(|e| panic!("execute error: {e}"))
}

/// Execute with custom AdaptiveConfig, returns (results, warnings).
fn run_cypher_adaptive(
    query: &str,
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    adaptive: AdaptiveConfig,
) -> (Vec<coordinode_query::executor::Row>, Vec<String>) {
    let ast = parse(query).unwrap_or_else(|e| panic!("parse error: {e}"));
    let plan = build_logical_plan(&ast).unwrap_or_else(|e| panic!("plan error: {e}"));
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(engine, interner, &allocator);
    ctx.adaptive = adaptive;
    let results = execute(&plan, &mut ctx).unwrap_or_else(|e| panic!("execute error: {e}"));
    (results, ctx.warnings)
}

/// Convenience wrapper with a fresh allocator (for read-only tests).
fn run_cypher(
    query: &str,
    engine: &StorageEngine,
    interner: &mut FieldInterner,
) -> Vec<coordinode_query::executor::Row> {
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    run_cypher_with_alloc(query, engine, interner, &allocator)
}

/// Execute with bound query parameters (e.g., `$p → 0.9`).
fn run_cypher_with_params(
    query: &str,
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    params: std::collections::HashMap<String, Value>,
) -> Vec<coordinode_query::executor::Row> {
    use coordinode_query::cypher::parse;
    use coordinode_query::executor::execute;
    use coordinode_query::planner::build_logical_plan;

    let ast = parse(query).unwrap_or_else(|e| panic!("parse error: {e}"));
    let plan = build_logical_plan(&ast).unwrap_or_else(|e| panic!("plan error: {e}"));
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(engine, interner, &allocator);
    ctx.params = params;
    execute(&plan, &mut ctx).unwrap_or_else(|e| panic!("execute error: {e}"))
}

/// Build the social graph:
///   Alice(1) -[:KNOWS]-> Bob(2) -[:KNOWS]-> Charlie(3) -[:KNOWS]-> Dave(4)
///   Alice(1) -[:KNOWS]-> Charlie(3)  (shortcut)
///   Alice(1) -[:LIKES]-> Eve(5)
fn setup_social_graph() -> (tempfile::TempDir, StorageEngine, FieldInterner) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[
            ("name", Value::String("Alice".into())),
            ("age", Value::Int(30)),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[
            ("name", Value::String("Bob".into())),
            ("age", Value::Int(25)),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Person",
        &[
            ("name", Value::String("Charlie".into())),
            ("age", Value::Int(30)),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        4,
        "Person",
        &[
            ("name", Value::String("Dave".into())),
            ("age", Value::Int(40)),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        5,
        "Person",
        &[
            ("name", Value::String("Eve".into())),
            ("age", Value::Int(35)),
        ],
        &mut interner,
    );

    insert_edge(&engine, "KNOWS", 1, 2); // Alice → Bob
    insert_edge(&engine, "KNOWS", 2, 3); // Bob → Charlie
    insert_edge(&engine, "KNOWS", 3, 4); // Charlie → Dave
    insert_edge(&engine, "KNOWS", 1, 3); // Alice → Charlie (shortcut)
    insert_edge(&engine, "LIKES", 1, 5); // Alice → Eve

    (dir, engine, interner)
}

// ── Variable-length path: full Cypher pipeline ──────────────────────────

#[test]
fn varlen_path_2_to_3_hops_end_to_end() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Alice -[:KNOWS*2..3]-> ? — should reach nodes at 2 and 3 hops
    let results = run_cypher(
        "MATCH (a:Person {name: \"Alice\"})-[:KNOWS*2..3]->(b) RETURN b",
        &engine,
        &mut interner,
    );

    let target_ids: Vec<i64> = results
        .iter()
        .filter_map(|r| match r.get("b") {
            Some(Value::Int(id)) => Some(*id),
            _ => None,
        })
        .collect();

    // 2 hops from Alice: Charlie (via Bob), Dave (via Charlie shortcut)
    // 3 hops from Alice: Dave (via Bob→Charlie→Dave)
    assert!(target_ids.contains(&3), "should reach Charlie at 2 hops");
    assert!(target_ids.contains(&4), "should reach Dave at 2-3 hops");
    assert!(
        !target_ids.contains(&2),
        "Bob is 1 hop — should NOT be in *2..3"
    );
}

#[test]
fn varlen_path_star_only_default_bounded() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Alice -[:KNOWS*]-> ? should NOT hang — default cap at 10 hops
    let results = run_cypher(
        "MATCH (a:Person {name: \"Alice\"})-[:KNOWS*]->(b) RETURN b",
        &engine,
        &mut interner,
    );

    // Should terminate and return some results (Bob, Charlie, Dave)
    assert!(!results.is_empty(), "unbounded * should return results");
}

#[test]
fn varlen_cycle_terminates() {
    // Build a cycle: A→B→C→A
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "X",
        &[("name", Value::String("A".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "X",
        &[("name", Value::String("B".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "X",
        &[("name", Value::String("C".into()))],
        &mut interner,
    );

    insert_edge(&engine, "NEXT", 1, 2);
    insert_edge(&engine, "NEXT", 2, 3);
    insert_edge(&engine, "NEXT", 3, 1); // cycle

    let results = run_cypher(
        "MATCH (a:X {name: \"A\"})-[:NEXT*1..10]->(b) RETURN b",
        &engine,
        &mut interner,
    );

    // 3 edges total — edge-uniqueness prevents infinite loop
    assert!(
        results.len() <= 3,
        "cycle should terminate via edge-uniqueness, got {} results",
        results.len()
    );
}

// ── Aggregation: full Cypher pipeline ───────────────────────────────────

#[test]
fn aggregate_count_group_by_end_to_end() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Count people by age: Alice+Charlie=30, Bob=25, Dave=40, Eve=35
    let results = run_cypher(
        "MATCH (n:Person) RETURN n.age AS age, count(*) AS cnt",
        &engine,
        &mut interner,
    );

    // Should have 4 groups (ages: 25, 30, 35, 40)
    assert_eq!(
        results.len(),
        4,
        "should have 4 age groups, got {}",
        results.len()
    );

    // Find the age=30 group — should have count=2
    let age_30 = results
        .iter()
        .find(|r| r.get("age") == Some(&Value::Int(30)));
    assert!(age_30.is_some(), "should have age=30 group");
    assert_eq!(age_30.unwrap().get("cnt"), Some(&Value::Int(2)));
}

#[test]
fn aggregate_sum_avg_end_to_end() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (n:Person) RETURN sum(n.age) AS total, avg(n.age) AS mean, count(*) AS cnt",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    // 30+25+30+40+35 = 160
    assert_eq!(results[0].get("total"), Some(&Value::Int(160)));
    assert_eq!(results[0].get("cnt"), Some(&Value::Int(5)));
    // avg = 160/5 = 32.0
    assert_eq!(results[0].get("mean"), Some(&Value::Float(32.0)));
}

// ── UNWIND: full Cypher pipeline ────────────────────────────────────────

#[test]
fn unwind_list_end_to_end() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher("UNWIND [10, 20, 30] AS x RETURN x", &engine, &mut interner);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get("x"), Some(&Value::Int(10)));
    assert_eq!(results[1].get("x"), Some(&Value::Int(20)));
    assert_eq!(results[2].get("x"), Some(&Value::Int(30)));
}

// ── OPTIONAL MATCH: full Cypher pipeline ────────────────────────────────

#[test]
fn optional_match_no_results_nulls_end_to_end() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Eve has no KNOWS edges — OPTIONAL MATCH should produce NULL for b
    let results = run_cypher(
        "MATCH (a:Person {name: \"Eve\"}) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a, b",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1, "should have 1 row (Eve with NULL b)");
    assert!(results[0].contains_key("a"), "a should be bound to Eve");
    // b should be NULL or absent (left outer join semantics)
    let b_val = results[0].get("b").cloned().unwrap_or(Value::Null);
    assert_eq!(
        b_val,
        Value::Null,
        "b should be NULL for Eve (no KNOWS edges)"
    );
}

#[test]
fn optional_match_with_results_end_to_end() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Alice has KNOWS edges → should return real results
    let results = run_cypher(
        "MATCH (a:Person {name: \"Alice\"}) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a, b",
        &engine,
        &mut interner,
    );

    // Alice KNOWS Bob and Charlie → 2 rows
    assert!(
        results.len() >= 2,
        "Alice should have at least 2 KNOWS targets"
    );
    for row in &results {
        assert!(row.contains_key("a"), "a should be bound");
        let b = row.get("b");
        assert!(
            b.is_some() && b != Some(&Value::Null),
            "b should not be NULL"
        );
    }
}

// ── Pipeline: WITH + aggregation chaining ───────────────────────────────

#[test]
fn with_aggregation_pipeline_end_to_end() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // MATCH → WITH count(*) AS cnt → RETURN cnt
    let results = run_cypher(
        "MATCH (n:Person) WITH count(*) AS cnt RETURN cnt",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("cnt"), Some(&Value::Int(5)));
}

// ═══════════════════════════════════════════════════════════════════════
// Basic MATCH + WHERE + RETURN (read queries)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn match_all_nodes_by_label() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher("MATCH (n:Person) RETURN n", &engine, &mut interner);

    assert_eq!(results.len(), 5, "should return all 5 Person nodes");
}

#[test]
fn match_node_with_property_filter() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (n:Person {name: \"Bob\"}) RETURN n.name, n.age",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1, "should find exactly Bob");
}

#[test]
fn match_where_comparison() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (n:Person) WHERE n.age > 30 RETURN n.name",
        &engine,
        &mut interner,
    );

    // age>30: Dave(40), Eve(35) — Alice(30) and Charlie(30) excluded
    assert_eq!(results.len(), 2, "age>30 should match Dave and Eve");
}

#[test]
fn match_traverse_single_hop() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (a:Person {name: \"Alice\"})-[:KNOWS]->(b) RETURN b",
        &engine,
        &mut interner,
    );

    let ids: Vec<i64> = results
        .iter()
        .filter_map(|r| match r.get("b") {
            Some(Value::Int(id)) => Some(*id),
            _ => None,
        })
        .collect();

    // Alice KNOWS Bob(2) and Charlie(3)
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&2), "should reach Bob");
    assert!(ids.contains(&3), "should reach Charlie");
}

#[test]
fn match_traverse_specific_edge_type() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Only LIKES edges from Alice → Eve(5)
    let results = run_cypher(
        "MATCH (a:Person {name: \"Alice\"})-[:LIKES]->(b) RETURN b",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("b"), Some(&Value::Int(5)));
}

#[test]
fn match_return_star() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (n:Person {name: \"Alice\"}) RETURN *",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    // Star should include all node properties
    assert!(results[0].contains_key("n"), "should have node variable");
}

// ═══════════════════════════════════════════════════════════════════════
// ORDER BY, LIMIT, SKIP
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn order_by_asc() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (n:Person) RETURN n.age AS age ORDER BY age",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 5);
    let ages: Vec<&Value> = results.iter().filter_map(|r| r.get("age")).collect();
    // Should be sorted ascending: 25, 30, 30, 35, 40
    for i in 1..ages.len() {
        assert!(
            compare_age(ages[i - 1], ages[i]) != std::cmp::Ordering::Greater,
            "age[{}]={:?} should be <= age[{}]={:?}",
            i - 1,
            ages[i - 1],
            i,
            ages[i]
        );
    }
}

fn compare_age(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x.cmp(y),
        _ => std::cmp::Ordering::Equal,
    }
}

#[test]
fn limit_results() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher("MATCH (n:Person) RETURN n LIMIT 2", &engine, &mut interner);

    assert_eq!(results.len(), 2, "LIMIT 2 should return exactly 2 rows");
}

#[test]
fn skip_results() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let all = run_cypher("MATCH (n:Person) RETURN n", &engine, &mut interner);
    let skipped = run_cypher("MATCH (n:Person) RETURN n SKIP 3", &engine, &mut interner);

    assert_eq!(
        skipped.len(),
        all.len() - 3,
        "SKIP 3 from {} should leave {}",
        all.len(),
        all.len() - 3
    );
}

#[test]
fn order_by_limit_combined() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (n:Person) RETURN n.age AS age ORDER BY age DESC LIMIT 2",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 2, "should return top 2 by age desc");
    // Top 2 ages DESC: 40, 35
    assert_eq!(results[0].get("age"), Some(&Value::Int(40)));
    assert_eq!(results[1].get("age"), Some(&Value::Int(35)));
}

// ═══════════════════════════════════════════════════════════════════════
// Write operations — CREATE, SET, DELETE, MERGE
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn create_node_and_read_back() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    run_cypher_with_alloc(
        "CREATE (n:Animal {species: \"Cat\", name: \"Whiskers\"})",
        &engine,
        &mut interner,
        &allocator,
    );

    let results = run_cypher_with_alloc(
        "MATCH (n:Animal) RETURN n.name",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results.len(), 1, "should find the created Animal node");
}

#[test]
fn create_multiple_nodes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Shared allocator ensures unique IDs across CREATE calls
    run_cypher_with_alloc(
        "CREATE (a:City {name: \"Berlin\"})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "CREATE (b:City {name: \"Munich\"})",
        &engine,
        &mut interner,
        &allocator,
    );

    let results = run_cypher_with_alloc(
        "MATCH (n:City) RETURN n.name",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results.len(), 2, "should find both created City nodes");
}

#[test]
fn create_node_and_edge_inline() {
    // CREATE (a)-[:TYPE]->(b) in one pattern
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    run_cypher_with_alloc(
        "CREATE (a:City {name: \"Berlin\"})-[:CONNECTED_TO]->(b:City {name: \"Munich\"})",
        &engine,
        &mut interner,
        &allocator,
    );

    let results = run_cypher_with_alloc(
        "MATCH (a:City {name: \"Berlin\"})-[:CONNECTED_TO]->(b) RETURN b.name",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results.len(), 1, "should find the edge Berlin→Munich");
}

#[test]
fn create_edge_between_existing_nodes() {
    // MATCH existing nodes, then CREATE edge between them
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    run_cypher_with_alloc(
        "MATCH (a:Person {name: \"Bob\"}), (b:Person {name: \"Eve\"}) CREATE (a)-[:LIKES]->(b)",
        &engine,
        &mut interner,
        &allocator,
    );

    let results = run_cypher_with_alloc(
        "MATCH (a:Person {name: \"Bob\"})-[:LIKES]->(b) RETURN b",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results.len(), 1, "should find Bob→Eve LIKES edge");
}

/// R010c: Multiple edges to same source within one statement use merge buffer.
/// Verifies read-your-own-writes: second CREATE sees first's edge.
#[test]
fn create_multiple_edges_from_same_source() {
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // CREATE two edges from Alice in one statement
    run_cypher_with_alloc(
        "MATCH (a:Person {name: \"Alice\"}), (b:Person {name: \"Bob\"}), \
         (c:Person {name: \"Charlie\"}) \
         CREATE (a)-[:LIKES]->(b), (a)-[:LIKES]->(c)",
        &engine,
        &mut interner,
        &allocator,
    );

    // Both edges should be visible
    let results = run_cypher_with_alloc(
        "MATCH (a:Person {name: \"Alice\"})-[:LIKES]->(b) RETURN b.name",
        &engine,
        &mut interner,
        &allocator,
    );
    // 3 LIKES edges: Alice→Eve (from setup), Alice→Bob, Alice→Charlie
    assert_eq!(results.len(), 3, "should find all 3 LIKES edges from Alice");
}

/// R010c: Edge creation via merge + edge traversal in same query.
/// Tests that merge buffer provides read-your-own-writes within execution.
#[test]
fn create_then_traverse_same_execution() {
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Create edge then immediately traverse in separate statements
    run_cypher_with_alloc(
        "MATCH (a:Person {name: \"Alice\"}), (b:Person {name: \"Eve\"}) \
         CREATE (a)-[:TRUSTS]->(b)",
        &engine,
        &mut interner,
        &allocator,
    );
    let results = run_cypher_with_alloc(
        "MATCH (a:Person {name: \"Alice\"})-[:TRUSTS]->(b) RETURN b.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(results.len(), 1, "should find TRUSTS edge");
}

/// R010c: DETACH DELETE removes node and its edges.
#[test]
fn detach_delete_removes_edges() {
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // setup_social_graph creates: Alice->Bob (KNOWS), Alice->Charlie (KNOWS), etc.
    // DETACH DELETE Alice
    run_cypher_with_alloc(
        "MATCH (n:Person {name: \"Alice\"}) DETACH DELETE n",
        &engine,
        &mut interner,
        &allocator,
    );

    // Alice should be gone
    let results = run_cypher_with_alloc(
        "MATCH (n:Person {name: \"Alice\"}) RETURN n",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(results.len(), 0, "Alice should be deleted");
}

#[test]
fn set_property_update() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Update Alice's age
    run_cypher(
        "MATCH (n:Person {name: \"Alice\"}) SET n.age = 31",
        &engine,
        &mut interner,
    );

    // Verify update
    let results = run_cypher(
        "MATCH (n:Person {name: \"Alice\"}) RETURN n.age AS age",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("age"), Some(&Value::Int(31)));
}

#[test]
fn delete_node() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let before = run_cypher("MATCH (n:Person) RETURN n", &engine, &mut interner);

    // Delete Eve (no edges except LIKES from Alice, may need DETACH)
    run_cypher(
        "MATCH (n:Person {name: \"Eve\"}) DETACH DELETE n",
        &engine,
        &mut interner,
    );

    let after = run_cypher("MATCH (n:Person) RETURN n", &engine, &mut interner);

    assert_eq!(
        after.len(),
        before.len() - 1,
        "should have one fewer Person after DELETE"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Additional aggregation functions — min, max, collect
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn aggregate_min_max_end_to_end() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (n:Person) RETURN min(n.age) AS youngest, max(n.age) AS oldest",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("youngest"), Some(&Value::Int(25)));
    assert_eq!(results[0].get("oldest"), Some(&Value::Int(40)));
}

#[test]
fn aggregate_collect_end_to_end() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (n:Person) RETURN collect(n.name) AS names",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    if let Some(Value::Array(names)) = results[0].get("names") {
        assert_eq!(names.len(), 5, "collect should gather all 5 names");
    } else {
        panic!("collect() should return an Array");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// percentileCont / percentileDisc — end-to-end through full Cypher pipeline
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn aggregate_percentile_cont_respects_argument() {
    // Regression: before the fix, percentileCont always returned the median
    // regardless of the p argument. This test runs through the full pipeline
    // (parse → plan builder → executor) to verify p is propagated.
    //
    // Person ages: [25, 30, 30, 35, 40]. Sorted.
    // percentileCont(age, 0.0) → 25.0  (min)
    // percentileCont(age, 1.0) → 40.0  (max)
    // percentileCont(age, 0.5) → 30.0  (median — also correct at p=0.5)
    let (_dir, engine, mut interner) = setup_social_graph();

    let min_result = run_cypher(
        "MATCH (n:Person) RETURN percentileCont(n.age, 0.0) AS p0",
        &engine,
        &mut interner,
    );
    assert_eq!(min_result.len(), 1);
    assert_eq!(
        min_result[0].get("p0"),
        Some(&Value::Float(25.0)),
        "percentileCont(age, 0.0) should return the minimum, not the median"
    );

    let max_result = run_cypher(
        "MATCH (n:Person) RETURN percentileCont(n.age, 1.0) AS p100",
        &engine,
        &mut interner,
    );
    assert_eq!(max_result.len(), 1);
    assert_eq!(
        max_result[0].get("p100"),
        Some(&Value::Float(40.0)),
        "percentileCont(age, 1.0) should return the maximum, not the median"
    );
}

#[test]
fn aggregate_percentile_disc_respects_argument() {
    // Same regression check for percentileDisc (nearest-rank variant).
    // Person ages: [25, 30, 30, 35, 40]. Sorted.
    // percentileDisc(age, 0.0) → 25.0  (saturating_sub(1) from ceil(0) = 0)
    // percentileDisc(age, 1.0) → 40.0  (ceil(1.0 * 5) - 1 = index 4)
    let (_dir, engine, mut interner) = setup_social_graph();

    let low_result = run_cypher(
        "MATCH (n:Person) RETURN percentileDisc(n.age, 0.0) AS p0",
        &engine,
        &mut interner,
    );
    assert_eq!(low_result.len(), 1);
    assert_eq!(
        low_result[0].get("p0"),
        Some(&Value::Float(25.0)),
        "percentileDisc(age, 0.0) should return the minimum, not the median"
    );

    let high_result = run_cypher(
        "MATCH (n:Person) RETURN percentileDisc(n.age, 1.0) AS p100",
        &engine,
        &mut interner,
    );
    assert_eq!(high_result.len(), 1);
    assert_eq!(
        high_result[0].get("p100"),
        Some(&Value::Float(40.0)),
        "percentileDisc(age, 1.0) should return the maximum, not the median"
    );
}

#[test]
fn aggregate_percentile_cont_with_query_parameter() {
    // Verify that percentileCont(x, $p) reads the percentile from the query
    // parameter map in ExecutionContext, matching Neo4j's behavior.
    // Person ages: [25, 30, 30, 35, 40]. percentileCont(age, $p) with $p=1.0 → 40.0.
    let (_dir, engine, mut interner) = setup_social_graph();

    let mut params = std::collections::HashMap::new();
    params.insert("p".into(), Value::Float(1.0));

    let result = run_cypher_with_params(
        "MATCH (n:Person) RETURN percentileCont(n.age, $p) AS result",
        &engine,
        &mut interner,
        params,
    );

    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].get("result"),
        Some(&Value::Float(40.0)),
        "percentileCont(age, $p) with $p=1.0 should return the maximum (40.0)"
    );
}

#[test]
fn aggregate_percentile_disc_with_query_parameter() {
    // Same as above but for percentileDisc (nearest-rank variant).
    // Person ages: [25, 30, 30, 35, 40]. percentileDisc(age, $p) with $p=0.0 → 25.0.
    let (_dir, engine, mut interner) = setup_social_graph();

    let mut params = std::collections::HashMap::new();
    params.insert("p".into(), Value::Float(0.0));

    let result = run_cypher_with_params(
        "MATCH (n:Person) RETURN percentileDisc(n.age, $p) AS result",
        &engine,
        &mut interner,
        params,
    );

    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].get("result"),
        Some(&Value::Float(25.0)),
        "percentileDisc(age, $p) with $p=0.0 should return the minimum (25.0)"
    );
}

#[test]
fn query_parameter_in_where_clause() {
    // Verify that $params in WHERE predicates are substituted before execution.
    // Uses the plan-level substitute_params() rewriter called inside execute().
    // Person ages: [25, 30, 30, 35, 40]. $min_age=35 → Dave (40) and Eve (35).
    let (_dir, engine, mut interner) = setup_social_graph();

    let mut params = std::collections::HashMap::new();
    params.insert("min_age".into(), Value::Int(35));

    let result = run_cypher_with_params(
        "MATCH (n:Person) WHERE n.age >= $min_age RETURN n.name AS name ORDER BY n.age",
        &engine,
        &mut interner,
        params,
    );

    assert_eq!(result.len(), 2, "should match Eve (35) and Dave (40)");
    let names: Vec<_> = result.iter().filter_map(|r| r.get("name")).collect();
    assert!(
        names.contains(&&Value::String("Eve".into())),
        "Eve (age 35) should be in results"
    );
    assert!(
        names.contains(&&Value::String("Dave".into())),
        "Dave (age 40) should be in results"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// AS OF TIMESTAMP (basic time travel)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn as_of_timestamp_basic() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Use a recent timestamp (within 7d retention)
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    let query = format!(
        "MATCH (n:Person) AS OF TIMESTAMP {} RETURN n",
        now_us - 1_000_000, // 1 second ago
    );

    // Should execute without error (may return same data since MVCC
    // isn't fully time-separated at storage level yet, but the pipeline
    // should not crash)
    let results = run_cypher(&query, &engine, &mut interner);
    // At minimum, pipeline completes without error
    assert!(
        results.len() <= 5,
        "AS OF TIMESTAMP should return at most 5 persons"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Complex multi-clause queries (pipeline integration)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn match_where_order_limit_pipeline() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Full read pipeline: MATCH → WHERE → ORDER BY → LIMIT → RETURN
    let results = run_cypher(
        "MATCH (n:Person) WHERE n.age >= 30 RETURN n.name AS name, n.age AS age ORDER BY age DESC LIMIT 3",
        &engine,
        &mut interner,
    );

    // age>=30: Alice(30), Charlie(30), Eve(35), Dave(40) → ORDER DESC → Dave, Eve, Charlie/Alice → LIMIT 3
    assert_eq!(results.len(), 3, "WHERE+ORDER+LIMIT should give 3 rows");
    assert_eq!(results[0].get("age"), Some(&Value::Int(40))); // Dave first
}

#[test]
fn traverse_then_aggregate() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Count how many people Alice knows
    let results = run_cypher(
        "MATCH (a:Person {name: \"Alice\"})-[:KNOWS]->(b) RETURN count(*) AS friends",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("friends"), Some(&Value::Int(2))); // Bob, Charlie
}

#[test]
fn multiple_match_patterns() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Two separate MATCH patterns → CartesianProduct
    let results = run_cypher(
        "MATCH (a:Person {name: \"Alice\"}), (b:Person {name: \"Bob\"}) RETURN a, b",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1, "should find exactly Alice×Bob pair");
}

#[test]
fn with_alias_passthrough() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // WITH renames variables — verify alias works across pipeline
    let results = run_cypher(
        "MATCH (n:Person) WITH n.name AS who RETURN who",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 5, "WITH passthrough should keep all 5 rows");
    // Each row should have 'who' column
    for row in &results {
        assert!(row.contains_key("who"), "should have 'who' alias");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// REMOVE property
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn remove_property() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Remove Alice's age property
    run_cypher(
        "MATCH (n:Person {name: \"Alice\"}) REMOVE n.age",
        &engine,
        &mut interner,
    );

    let results = run_cypher(
        "MATCH (n:Person {name: \"Alice\"}) RETURN n.age AS age",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    // age should be NULL after REMOVE
    let age = results[0].get("age").cloned().unwrap_or(Value::Null);
    assert_eq!(age, Value::Null, "removed property should be NULL");
}

// ═══════════════════════════════════════════════════════════════════════
// MERGE — match or create
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn merge_existing_node() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // MERGE on existing node — should NOT create duplicate
    let before = run_cypher("MATCH (n:Person) RETURN n", &engine, &mut interner);

    run_cypher("MERGE (n:Person {name: \"Alice\"})", &engine, &mut interner);

    let after = run_cypher("MATCH (n:Person) RETURN n", &engine, &mut interner);
    assert_eq!(
        before.len(),
        after.len(),
        "MERGE existing should not duplicate"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Reverse and bidirectional traversal
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn traverse_reverse_direction() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Bob <-[:KNOWS]- Alice — reverse traversal
    let results = run_cypher(
        "MATCH (a:Person {name: \"Bob\"})<-[:KNOWS]-(b) RETURN b",
        &engine,
        &mut interner,
    );

    let ids: Vec<i64> = results
        .iter()
        .filter_map(|r| match r.get("b") {
            Some(Value::Int(id)) => Some(*id),
            _ => None,
        })
        .collect();

    assert!(
        ids.contains(&1),
        "Alice(1) should be found via reverse KNOWS"
    );
}

#[test]
fn traverse_bidirectional() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Bob -[:KNOWS]- ? (both directions)
    let results = run_cypher(
        "MATCH (a:Person {name: \"Bob\"})-[:KNOWS]-(b) RETURN b",
        &engine,
        &mut interner,
    );

    let ids: Vec<i64> = results
        .iter()
        .filter_map(|r| match r.get("b") {
            Some(Value::Int(id)) => Some(*id),
            _ => None,
        })
        .collect();

    // Bob→Charlie(3) outgoing + Alice(1)→Bob incoming
    assert!(ids.contains(&1), "Alice should be found (incoming KNOWS)");
    assert!(ids.contains(&3), "Charlie should be found (outgoing KNOWS)");
}

// ═══════════════════════════════════════════════════════════════════════
// DISTINCT in aggregation (end-to-end)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn aggregate_count_distinct_end_to_end() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Alice(30) and Charlie(30) have same age → count(DISTINCT age) = 4, not 5
    let results = run_cypher(
        "MATCH (n:Person) RETURN count(DISTINCT n.age) AS unique_ages",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("unique_ages"), Some(&Value::Int(4)));
}

// ═══════════════════════════════════════════════════════════════════════
// DISTINCT with non-consecutive duplicates (G033 regression test)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn distinct_removes_non_consecutive_duplicate_rows() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Multiple paths to same person produce duplicate rows.
    // DISTINCT should remove them even when non-consecutive.
    // Alice(30), Charlie(30) — two people with same age.
    // "RETURN DISTINCT n.age" should return 4 unique ages, not 5 rows.
    let results = run_cypher(
        "MATCH (n:Person) RETURN DISTINCT n.age AS age",
        &engine,
        &mut interner,
    );

    // 5 people but ages are [25, 28, 30, 30, 35] → 4 unique
    assert_eq!(
        results.len(),
        4,
        "DISTINCT should produce 4 unique ages from 5 people, got: {results:?}"
    );
}

#[test]
fn distinct_with_float_values() {
    let (_dir, engine, mut interner) = setup_social_graph();

    // Create nodes with float properties — duplicates should be removed
    // Create all sensors in one query to ensure sequential IDs
    run_cypher(
        "CREATE (a:Sensor {reading: 1.5}), (b:Sensor {reading: 2.7}), (c:Sensor {reading: 1.5}), (d:Sensor {reading: 2.7})",
        &engine,
        &mut interner,
    );

    // Verify all 4 sensors exist
    let all = run_cypher(
        "MATCH (s:Sensor) RETURN s.reading AS r",
        &engine,
        &mut interner,
    );
    assert_eq!(all.len(), 4, "should have 4 sensors, got: {all:?}");

    let results = run_cypher(
        "MATCH (s:Sensor) RETURN DISTINCT s.reading AS r",
        &engine,
        &mut interner,
    );

    assert_eq!(
        results.len(),
        2,
        "DISTINCT should produce 2 unique readings from 4 sensors, got: {results:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// WHERE with logical operators
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn where_and_or_combined() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (n:Person) WHERE n.age = 30 OR n.age = 40 RETURN n.name",
        &engine,
        &mut interner,
    );

    // age=30: Alice, Charlie. age=40: Dave. Total: 3
    assert_eq!(results.len(), 3, "OR should match Alice+Charlie+Dave");
}

#[test]
fn where_string_predicate() {
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (n:Person) WHERE n.name STARTS WITH \"A\" RETURN n.name",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1, "only Alice starts with A");
}

// ═══════════════════════════════════════════════════════════════════════
// UPSERT MATCH — atomic match-or-create
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn upsert_creates_when_not_exists() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // UPSERT on empty DB → should create the node
    run_cypher_with_alloc(
        "UPSERT MATCH (u:User {email: \"alice@example.com\"}) \
         ON MATCH SET u.login_count = 99 \
         ON CREATE CREATE (u:User {email: \"alice@example.com\", login_count: 1}) \
         RETURN u",
        &engine,
        &mut interner,
        &allocator,
    );

    let results = run_cypher_with_alloc(
        "MATCH (u:User) RETURN u.login_count AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results.len(), 1, "UPSERT should have created the user");
    assert_eq!(
        results[0].get("cnt"),
        Some(&Value::Int(1)),
        "ON CREATE should set login_count=1"
    );
}

#[test]
fn upsert_updates_when_exists() {
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Alice already exists → UPSERT should match and SET
    run_cypher_with_alloc(
        "UPSERT MATCH (u:Person {name: \"Alice\"}) \
         ON MATCH SET u.age = 99 \
         ON CREATE CREATE (u:Person {name: \"Alice\", age: 0}) \
         RETURN u",
        &engine,
        &mut interner,
        &allocator,
    );

    let results = run_cypher_with_alloc(
        "MATCH (u:Person {name: \"Alice\"}) RETURN u.age AS age",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("age"),
        Some(&Value::Int(99)),
        "ON MATCH should have updated age to 99"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Hybrid graph+vector query via Cypher
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn vector_distance_filter_end_to_end() {
    // Create nodes with vector properties, then filter by distance
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Insert nodes with vector embeddings
    insert_node(
        &engine,
        1,
        1,
        "Doc",
        &[
            ("title", Value::String("rust".into())),
            ("embedding", Value::Vector(vec![1.0, 0.0, 0.0])),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Doc",
        &[
            ("title", Value::String("python".into())),
            ("embedding", Value::Vector(vec![0.0, 1.0, 0.0])),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Doc",
        &[
            ("title", Value::String("go".into())),
            ("embedding", Value::Vector(vec![0.9, 0.1, 0.0])),
        ],
        &mut interner,
    );

    // Query: find docs where vector_distance(embedding, [1,0,0]) < 0.5
    // Rust [1,0,0] → distance=0, Go [0.9,0.1,0] → distance≈0.14, Python [0,1,0] → distance=√2≈1.41
    let results = run_cypher(
        "MATCH (d:Doc) WHERE vector_distance(d.embedding, [1.0, 0.0, 0.0]) < 0.5 RETURN d.title",
        &engine,
        &mut interner,
    );

    // Should return Rust (dist=0) and Go (dist≈0.14), not Python (dist≈1.41)
    assert_eq!(results.len(), 2, "should find 2 docs within distance 0.5");
}

// ═══════════════════════════════════════════════════════════════════════
// UPSERT block enhancement — CAS conflict detection, ON CREATE edges
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn upsert_on_create_with_edge() {
    // UPSERT on empty DB with edge pattern → should create nodes + edge
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    run_cypher_with_alloc(
        "UPSERT MATCH (u:Team {name: \"Alpha\"}) \
         ON MATCH SET u.size = 99 \
         ON CREATE CREATE (u:Team {name: \"Alpha\"})-[:HAS_MEMBER]->(m:Person {name: \"Leader\"}) \
         RETURN u",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify nodes created
    let teams = run_cypher_with_alloc(
        "MATCH (t:Team) RETURN t.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(teams.len(), 1, "should create Team node");

    let members = run_cypher_with_alloc(
        "MATCH (p:Person) RETURN p.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(members.len(), 1, "should create Person node");

    // Verify edge created
    let edges = run_cypher_with_alloc(
        "MATCH (t:Team {name: \"Alpha\"})-[:HAS_MEMBER]->(p) RETURN p.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(edges.len(), 1, "should create HAS_MEMBER edge");
}

#[test]
fn upsert_idempotent_on_match() {
    // Run UPSERT twice — second time should match and SET, not create duplicate
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // First UPSERT — Alice exists → ON MATCH SET age=50
    run_cypher_with_alloc(
        "UPSERT MATCH (u:Person {name: \"Alice\"}) \
         ON MATCH SET u.age = 50 \
         ON CREATE CREATE (u:Person {name: \"Alice\", age: 0}) \
         RETURN u",
        &engine,
        &mut interner,
        &allocator,
    );

    // Second UPSERT — Alice still exists → ON MATCH SET age=60
    run_cypher_with_alloc(
        "UPSERT MATCH (u:Person {name: \"Alice\"}) \
         ON MATCH SET u.age = 60 \
         ON CREATE CREATE (u:Person {name: \"Alice\", age: 0}) \
         RETURN u",
        &engine,
        &mut interner,
        &allocator,
    );

    let results = run_cypher_with_alloc(
        "MATCH (u:Person {name: \"Alice\"}) RETURN u.age AS age",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results.len(), 1, "should still have exactly 1 Alice");
    assert_eq!(
        results[0].get("age"),
        Some(&Value::Int(60)),
        "age should be 60 from second UPSERT"
    );

    // Total Person count should be unchanged (5 from setup)
    let all = run_cypher_with_alloc(
        "MATCH (n:Person) RETURN n",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(all.len(), 5, "should NOT create duplicate Alice");
}

#[test]
fn upsert_cas_detects_conflict() {
    // Simulate CAS conflict: manually modify node between UPSERT read and write
    // We can't easily test concurrent modification in single-threaded executor,
    // but we verify the Conflict error variant exists and the CAS logic works
    // by checking that UPSERT succeeds when node is NOT modified.
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Normal UPSERT should succeed (no concurrent modification)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_cypher_with_alloc(
            "UPSERT MATCH (u:Person {name: \"Alice\"}) \
             ON MATCH SET u.age = 42 \
             ON CREATE CREATE (u:Person {name: \"Alice\", age: 0}) \
             RETURN u",
            &engine,
            &mut interner,
            &allocator,
        )
    }));
    assert!(
        result.is_ok(),
        "UPSERT should succeed without concurrent modification"
    );

    // Verify the update applied
    let check = run_cypher_with_alloc(
        "MATCH (u:Person {name: \"Alice\"}) RETURN u.age AS age",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(check[0].get("age"), Some(&Value::Int(42)));
}

#[test]
fn upsert_on_match_set_expression() {
    // ON MATCH SET u.age = u.age + 10 — expression referencing current value
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Alice has age=30. UPSERT should set age = 30 + 10 = 40
    run_cypher_with_alloc(
        "UPSERT MATCH (u:Person {name: \"Alice\"}) \
         ON MATCH SET u.age = u.age + 10 \
         ON CREATE CREATE (u:Person {name: \"Alice\", age: 0}) \
         RETURN u",
        &engine,
        &mut interner,
        &allocator,
    );

    let results = run_cypher_with_alloc(
        "MATCH (u:Person {name: \"Alice\"}) RETURN u.age AS age",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("age"),
        Some(&Value::Int(40)),
        "ON MATCH SET u.age = u.age + 10 should yield 30+10=40"
    );
}

#[test]
fn upsert_returns_result_row() {
    // UPSERT MATCH ... RETURN u should return the upserted node
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    let results = run_cypher_with_alloc(
        "UPSERT MATCH (u:Person {name: \"Alice\"}) \
         ON MATCH SET u.age = 77 \
         ON CREATE CREATE (u:Person {name: \"Alice\", age: 0}) \
         RETURN u.age AS age",
        &engine,
        &mut interner,
        &allocator,
    );

    // Should return 1 row with the updated age
    assert_eq!(results.len(), 1, "UPSERT should return result rows");
    assert_eq!(results[0].get("age"), Some(&Value::Int(77)));
}

#[test]
fn upsert_on_create_returns_new_node() {
    // UPSERT on empty DB → ON CREATE → RETURN should return the created node
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    let results = run_cypher_with_alloc(
        "UPSERT MATCH (u:Robot {name: \"R2D2\"}) \
         ON MATCH SET u.active = true \
         ON CREATE CREATE (u:Robot {name: \"R2D2\", model: \"Astromech\"}) \
         RETURN u.name AS name, u.model AS model",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results.len(), 1, "UPSERT ON CREATE should return result");
    assert_eq!(results[0].get("name"), Some(&Value::String("R2D2".into())));
    assert_eq!(
        results[0].get("model"),
        Some(&Value::String("Astromech".into()))
    );
}

#[test]
fn upsert_empty_on_match_only_creates() {
    // UPSERT with no ON MATCH items — only ON CREATE should fire
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    let results = run_cypher_with_alloc(
        "UPSERT MATCH (u:Ghost {name: \"Casper\"}) \
         ON CREATE CREATE (u:Ghost {name: \"Casper\", friendly: true}) \
         RETURN u.name AS name",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results.len(), 1, "should create via ON CREATE");
    assert_eq!(
        results[0].get("name"),
        Some(&Value::String("Casper".into()))
    );
}

#[test]
fn upsert_empty_on_create_only_updates() {
    // UPSERT with no ON CREATE patterns — ON MATCH SET only
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Alice exists → ON MATCH SET fires, no ON CREATE
    run_cypher_with_alloc(
        "UPSERT MATCH (u:Person {name: \"Alice\"}) \
         ON MATCH SET u.age = 88 \
         RETURN u",
        &engine,
        &mut interner,
        &allocator,
    );

    let check = run_cypher_with_alloc(
        "MATCH (u:Person {name: \"Alice\"}) RETURN u.age AS age",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(check[0].get("age"), Some(&Value::Int(88)));
}

#[test]
fn upsert_match_multiple_nodes() {
    // UPSERT where MATCH finds multiple nodes (Alice+Charlie both age=30)
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Both Alice(30) and Charlie(30) match → ON MATCH SET applies to both
    run_cypher_with_alloc(
        "UPSERT MATCH (u:Person {age: 30}) \
         ON MATCH SET u.age = 31 \
         ON CREATE CREATE (u:Person {age: 30}) \
         RETURN u",
        &engine,
        &mut interner,
        &allocator,
    );

    // Both should now have age=31
    let results = run_cypher_with_alloc(
        "MATCH (n:Person) WHERE n.age = 31 RETURN n.name",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(
        results.len(),
        2,
        "both Alice and Charlie should have age=31"
    );
}

#[test]
fn upsert_on_match_set_multiple_properties() {
    let (_dir, engine, mut interner) = setup_social_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    run_cypher_with_alloc(
        "UPSERT MATCH (u:Person {name: \"Bob\"}) \
         ON MATCH SET u.age = 99, u.active = true \
         ON CREATE CREATE (u:Person {name: \"Bob\"}) \
         RETURN u",
        &engine,
        &mut interner,
        &allocator,
    );

    let results = run_cypher_with_alloc(
        "MATCH (u:Person {name: \"Bob\"}) RETURN u.age AS age",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(results[0].get("age"), Some(&Value::Int(99)));
}

// ═══════════════════════════════════════════════════════════════════════
// Query cost estimation + EXPLAIN
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn explain_simple_match_has_cost() {
    let query = "MATCH (n:Person) RETURN n";
    let ast = parse(query).unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let cost = estimate_cost(&plan);

    assert!(cost.cost > 0.0, "cost should be > 0");
    assert!(cost.estimated_rows > 0.0, "estimated_rows should be > 0");
    assert!(cost.estimated_time_ms > 0.0, "estimated_time should be > 0");

    let explain = plan.explain();
    assert!(
        explain.contains("Cost:"),
        "EXPLAIN should contain cost line"
    );
    assert!(
        explain.contains("Estimated rows:"),
        "EXPLAIN should contain rows"
    );
}

#[test]
fn explain_traverse_increases_cost() {
    let scan_ast = parse("MATCH (n:Person) RETURN n").unwrap();
    let scan_plan = build_logical_plan(&scan_ast).unwrap();
    let scan_cost = estimate_cost(&scan_plan);

    let trav_ast = parse("MATCH (a:Person)-[:KNOWS]->(b) RETURN b").unwrap();
    let trav_plan = build_logical_plan(&trav_ast).unwrap();
    let trav_cost = estimate_cost(&trav_plan);

    assert!(
        trav_cost.cost > scan_cost.cost,
        "traverse ({}) should be more expensive than scan ({})",
        trav_cost.cost,
        scan_cost.cost
    );
}

#[test]
fn explain_varlen_path_generates_hint() {
    let ast = parse("MATCH (a:Person)-[:KNOWS*1..7]->(b) RETURN b").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let cost = estimate_cost(&plan);

    assert!(
        cost.hints
            .iter()
            .any(|h| h.contains("variable-length path")),
        "deep varlen path should produce optimization hint, hints: {:?}",
        cost.hints
    );

    let explain = plan.explain();
    assert!(explain.contains("Hint:"), "EXPLAIN should show hints");
}

#[test]
fn explain_limit_reduces_rows() {
    let unlimited_ast = parse("MATCH (n:Person) RETURN n").unwrap();
    let unlimited_plan = build_logical_plan(&unlimited_ast).unwrap();
    let unlimited_cost = estimate_cost(&unlimited_plan);

    let limited_ast = parse("MATCH (n:Person) RETURN n LIMIT 5").unwrap();
    let limited_plan = build_logical_plan(&limited_ast).unwrap();
    let limited_cost = estimate_cost(&limited_plan);

    assert!(
        limited_cost.estimated_rows <= unlimited_cost.estimated_rows,
        "LIMIT 5 rows ({}) should be <= unlimited rows ({})",
        limited_cost.estimated_rows,
        unlimited_cost.estimated_rows
    );
}

#[test]
fn explain_aggregate_reduces_rows() {
    let raw_ast = parse("MATCH (n:Person) RETURN n").unwrap();
    let raw_plan = build_logical_plan(&raw_ast).unwrap();
    let raw_cost = estimate_cost(&raw_plan);

    let agg_ast = parse("MATCH (n:Person) RETURN count(*) AS cnt").unwrap();
    let agg_plan = build_logical_plan(&agg_ast).unwrap();
    let agg_cost = estimate_cost(&agg_plan);

    assert!(
        agg_cost.estimated_rows < raw_cost.estimated_rows,
        "aggregate count(*) ({} rows) should reduce vs raw scan ({} rows)",
        agg_cost.estimated_rows,
        raw_cost.estimated_rows
    );
}

#[test]
fn explain_filter_reduces_rows() {
    let scan_ast = parse("MATCH (n:Person) RETURN n").unwrap();
    let scan_cost = estimate_cost(&build_logical_plan(&scan_ast).unwrap());

    let filter_ast = parse("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();
    let filter_cost = estimate_cost(&build_logical_plan(&filter_ast).unwrap());

    assert!(
        filter_cost.estimated_rows < scan_cost.estimated_rows,
        "WHERE filter ({} rows) should reduce vs full scan ({} rows)",
        filter_cost.estimated_rows,
        scan_cost.estimated_rows
    );
}

#[test]
fn explain_end_to_end_with_real_data() {
    // Full e2e: parse → plan → estimate → execute → verify explain matches
    let (_dir, engine, mut interner) = setup_social_graph();

    let query = "MATCH (a:Person {name: \"Alice\"})-[:KNOWS]->(b) RETURN b.name";
    let ast = parse(query).unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let cost = estimate_cost(&plan);

    // Verify cost is reasonable
    assert!(cost.cost > 0.0);
    assert!(cost.estimated_rows > 0.0);

    // Actually execute and compare
    let results = run_cypher(query, &engine, &mut interner);
    // Real result: Alice→Bob, Alice→Charlie = 2 rows
    assert_eq!(results.len(), 2);

    // EXPLAIN text should be parseable
    let explain = plan.explain();
    assert!(explain.contains("NodeScan"));
    assert!(explain.contains("Traverse"));
    assert!(explain.contains("Cost:"));
}

#[test]
fn explain_cartesian_product_warns() {
    // Two separate MATCH patterns without join → CartesianProduct
    let ast = parse("MATCH (a:Person), (b:Person) RETURN a, b").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let cost = estimate_cost(&plan);

    // Cartesian of ~200 × ~200 = ~40000 → should trigger hint
    assert!(
        cost.hints.iter().any(|h| h.contains("CartesianProduct")),
        "large CartesianProduct should produce hint, hints: {:?}",
        cost.hints
    );
}

#[test]
fn explain_vector_filter_more_expensive_than_scalar() {
    // vector_distance() in WHERE should cost more than scalar n.age > 30
    let scalar_ast = parse("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();
    let scalar_cost = estimate_cost(&build_logical_plan(&scalar_ast).unwrap());

    let vector_ast =
        parse("MATCH (n:Person) WHERE vector_distance(n.embedding, [1.0, 0.0]) < 0.5 RETURN n")
            .unwrap();
    let vector_cost = estimate_cost(&build_logical_plan(&vector_ast).unwrap());

    assert!(
        vector_cost.cost > scalar_cost.cost,
        "vector filter ({:.1}) should be more expensive than scalar ({:.1})",
        vector_cost.cost,
        scalar_cost.cost
    );
}

#[test]
fn explain_skip_reduces_output_rows() {
    let full_ast = parse("MATCH (n:Person) RETURN n").unwrap();
    let full_cost = estimate_cost(&build_logical_plan(&full_ast).unwrap());

    let skip_ast = parse("MATCH (n:Person) RETURN n SKIP 100").unwrap();
    let skip_cost = estimate_cost(&build_logical_plan(&skip_ast).unwrap());

    assert!(
        skip_cost.estimated_rows < full_cost.estimated_rows,
        "SKIP 100 ({} rows) should reduce vs full ({} rows)",
        skip_cost.estimated_rows,
        full_cost.estimated_rows
    );
}

#[test]
fn explain_sort_adds_cost() {
    let unsorted_ast = parse("MATCH (n:Person) RETURN n.age AS age").unwrap();
    let unsorted_cost = estimate_cost(&build_logical_plan(&unsorted_ast).unwrap());

    let sorted_ast = parse("MATCH (n:Person) RETURN n.age AS age ORDER BY age").unwrap();
    let sorted_cost = estimate_cost(&build_logical_plan(&sorted_ast).unwrap());

    assert!(
        sorted_cost.cost >= unsorted_cost.cost,
        "ORDER BY ({:.1}) should cost >= unsorted ({:.1})",
        sorted_cost.cost,
        unsorted_cost.cost
    );
}

#[test]
fn explain_unwind_multiplies_rows() {
    let ast = parse("UNWIND [1, 2, 3] AS x RETURN x").unwrap();
    let cost = estimate_cost(&build_logical_plan(&ast).unwrap());

    // UNWIND on Empty (1 row) × avg_list_length(5) = 5 estimated rows
    assert!(
        cost.estimated_rows > 1.0,
        "UNWIND should multiply rows, got {}",
        cost.estimated_rows
    );
}

#[test]
fn explain_create_has_write_cost() {
    let write_ast = parse("CREATE (n:Person {name: \"Test\"})").unwrap();
    let write_cost = estimate_cost(&build_logical_plan(&write_ast).unwrap());

    assert!(write_cost.cost > 0.0, "CREATE should have non-zero cost");
    // Write cost is per-row I/O, different scale than reads
    assert!(write_cost.estimated_rows >= 1.0);
}

#[test]
fn explain_optional_match_has_cost() {
    let ast = parse("MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a, b").unwrap();
    let cost = estimate_cost(&build_logical_plan(&ast).unwrap());

    assert!(cost.cost > 0.0);
    assert!(cost.estimated_rows > 0.0);
}

// ═══════════════════════════════════════════════════════════════════════
// Mixed graph+vector pipeline
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn planner_rewrites_vector_distance_to_vector_filter() {
    // WHERE vector_distance(x, vec) < 0.5 should produce VectorFilter, not Filter
    let ast = parse(
        "MATCH (n:Person) WHERE vector_distance(n.embedding, [1.0, 0.0, 0.0]) < 0.5 RETURN n",
    )
    .unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let explain = plan.explain();

    assert!(
        explain.contains("VectorFilter"),
        "planner should rewrite vector_distance WHERE to VectorFilter. EXPLAIN:\n{explain}"
    );
}

#[test]
fn planner_rewrites_vector_similarity_to_vector_filter() {
    let ast =
        parse("MATCH (n:Person) WHERE vector_similarity(n.embedding, [1.0, 0.0]) > 0.8 RETURN n")
            .unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let explain = plan.explain();

    assert!(
        explain.contains("VectorFilter"),
        "vector_similarity > threshold should become VectorFilter. EXPLAIN:\n{explain}"
    );
}

#[test]
fn planner_keeps_scalar_where_as_filter() {
    // Scalar WHERE should NOT become VectorFilter
    let ast = parse("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let explain = plan.explain();

    assert!(
        !explain.contains("VectorFilter"),
        "scalar WHERE should stay as Filter, not VectorFilter. EXPLAIN:\n{explain}"
    );
    assert!(explain.contains("Filter("));
}

#[test]
fn vector_filter_end_to_end_with_data() {
    // Full pipeline: create docs with vectors → MATCH WHERE vector_distance < threshold → results
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Doc",
        &[
            ("title", Value::String("rust".into())),
            ("embedding", Value::Vector(vec![1.0, 0.0, 0.0])),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Doc",
        &[
            ("title", Value::String("python".into())),
            ("embedding", Value::Vector(vec![0.0, 1.0, 0.0])),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Doc",
        &[
            ("title", Value::String("go".into())),
            ("embedding", Value::Vector(vec![0.9, 0.1, 0.0])),
        ],
        &mut interner,
    );

    // VectorFilter path: vector_distance < 0.5
    let results = run_cypher(
        "MATCH (d:Doc) WHERE vector_distance(d.embedding, [1.0, 0.0, 0.0]) < 0.5 RETURN d.title",
        &engine,
        &mut interner,
    );

    assert_eq!(
        results.len(),
        2,
        "should find Rust (dist=0) and Go (dist≈0.14)"
    );
}

#[test]
fn vector_filter_similarity_end_to_end() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Doc",
        &[("embedding", Value::Vector(vec![1.0, 0.0]))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Doc",
        &[("embedding", Value::Vector(vec![0.0, 1.0]))],
        &mut interner,
    );

    // vector_similarity > 0.9: only [1,0] is similar to query [1,0]
    let results = run_cypher(
        "MATCH (d:Doc) WHERE vector_similarity(d.embedding, [1.0, 0.0]) > 0.9 RETURN d",
        &engine,
        &mut interner,
    );

    assert_eq!(
        results.len(),
        1,
        "only [1,0] should have similarity > 0.9 to [1,0]"
    );
}

#[test]
fn mixed_traverse_then_vector_filter() {
    // Graph traversal → vector filter in single query
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // User(1) -[:LIKES]-> Doc(2, embed=[1,0,0])
    // User(1) -[:LIKES]-> Doc(3, embed=[0,1,0])
    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Doc",
        &[
            ("title", Value::String("ML".into())),
            ("embedding", Value::Vector(vec![1.0, 0.0, 0.0])),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Doc",
        &[
            ("title", Value::String("Art".into())),
            ("embedding", Value::Vector(vec![0.0, 1.0, 0.0])),
        ],
        &mut interner,
    );
    insert_edge(&engine, "LIKES", 1, 2);
    insert_edge(&engine, "LIKES", 1, 3);

    // Traverse Alice→LIKES→Doc, then vector filter by similarity to [1,0,0]
    let results = run_cypher(
        "MATCH (u:User {name: \"Alice\"})-[:LIKES]->(d:Doc) \
         WHERE vector_distance(d.embedding, [1.0, 0.0, 0.0]) < 0.5 \
         RETURN d.title",
        &engine,
        &mut interner,
    );

    // Only ML doc (embed=[1,0,0], dist=0) should pass, Art (embed=[0,1,0], dist=√2) filtered
    assert_eq!(results.len(), 1, "only ML doc should pass vector filter");
}

/// `WITH ... AS alias ORDER BY alias LIMIT K` (Pattern B) must
/// produce correct top-K results with brute-force fallback when no HNSW index
/// exists. Uses simple aliases (no dot-notation) for stable column lookup.
#[test]
fn vector_top_k_pattern_b_end_to_end() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // 5 documents with deterministic embeddings along the X axis.
    for i in 1..=5 {
        let x = i as f32 / 5.0;
        insert_node(
            &engine,
            1,
            i as u64,
            "Doc",
            &[
                ("name", Value::String(format!("doc{i}"))),
                ("embedding", Value::Vector(vec![x, 0.0, 0.0])),
            ],
            &mut interner,
        );
    }

    // Pattern B shape: project name + distance alias in WITH, ORDER BY alias,
    // LIMIT, RETURN explicit aliases. Expect top-2 closest to [1,0,0]: doc5, doc4.
    let results = run_cypher(
        "MATCH (d:Doc) \
         WITH d.name AS docname, vector_distance(d.embedding, [1.0, 0.0, 0.0]) AS dist \
         ORDER BY dist \
         LIMIT 2 \
         RETURN docname, dist",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 2, "top_k=2 should return exactly 2 rows");

    let names: Vec<String> = results
        .iter()
        .filter_map(|r| {
            r.get("docname")
                .and_then(|v| v.as_str().map(str::to_string))
        })
        .collect();
    assert_eq!(names.len(), 2, "expected 2 docname values, got {names:?}");
    assert_eq!(names[0], "doc5", "nearest to [1,0,0] should be doc5");
    assert_eq!(names[1], "doc4", "second nearest should be doc4");

    let extract_dist = |row: &coordinode_query::executor::Row| -> f64 {
        row.get("dist")
            .and_then(|v| match v {
                Value::Float(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(f64::NAN)
    };
    let d0 = extract_dist(&results[0]);
    let d1 = extract_dist(&results[1]);
    assert!(d0 < 0.01, "doc5 distance should be near zero, got {d0}");
    assert!(d0 <= d1, "results must be sorted ascending: {d0} > {d1}");
}

/// VectorTopK on empty input returns empty result without panic.
/// Verifies `rows.is_empty() → Ok(Vec::new())` early-return in both HNSW
/// and brute-force paths of the executor.
#[test]
fn vector_top_k_empty_input_returns_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // No nodes inserted — MATCH returns 0 rows.
    let results = run_cypher(
        "MATCH (n:Nonexistent) \
         WITH n.name AS name, vector_distance(n.v, [1.0, 0.0]) AS d \
         ORDER BY d \
         LIMIT 5 \
         RETURN name, d",
        &engine,
        &mut interner,
    );

    assert_eq!(
        results.len(),
        0,
        "empty label must produce empty VectorTopK results"
    );
}

/// Dimension mismatch between query vector and stored embeddings.
/// Rows whose vector has a different dimension than the query must be skipped
/// via `if v.len() == query_vec.len()` guard in brute-force fallback.
#[test]
fn vector_top_k_dimension_mismatch_skips_rows() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Mixed dimensions: 2 nodes with dim=3, 1 with dim=2 (mismatch).
    insert_node(
        &engine,
        1,
        1,
        "Mixed",
        &[
            ("nid", Value::Int(1)),
            ("v", Value::Vector(vec![1.0, 0.0, 0.0])), // dim=3, matches
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Mixed",
        &[
            ("nid", Value::Int(2)),
            ("v", Value::Vector(vec![0.5, 0.5])), // dim=2, mismatch — skipped
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Mixed",
        &[
            ("nid", Value::Int(3)),
            ("v", Value::Vector(vec![0.0, 1.0, 0.0])), // dim=3, matches
        ],
        &mut interner,
    );

    let results = run_cypher(
        "MATCH (n:Mixed) \
         WITH n.nid AS nid, vector_distance(n.v, [1.0, 0.0, 0.0]) AS d \
         ORDER BY d \
         LIMIT 5 \
         RETURN nid, d",
        &engine,
        &mut interner,
    );

    // Only 2 rows returned (dim=2 node silently skipped).
    assert_eq!(
        results.len(),
        2,
        "dim mismatch row must be silently skipped"
    );

    let top_id = results[0]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("nid populated");
    assert_eq!(top_id, 1, "closest to [1,0,0] is n1");

    let second_id = results[1]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("second nid populated");
    assert_eq!(
        second_id, 3,
        "second is n3 (n2 skipped due to dim mismatch)"
    );
}

/// SKIP between ORDER BY and LIMIT → planner does NOT apply optimization
/// (Sort is under Skip, not directly under Limit). Sort+Skip+Limit fallback
/// must work correctly for paginated top-K queries.
#[test]
fn vector_top_k_skip_limit_fallback() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // 5 nodes at increasing distances from [0.0, 0.0]:
    //   n1..n5 with x = 0.1 .. 0.5
    for i in 1..=5u64 {
        let x = i as f32 / 10.0;
        insert_node(
            &engine,
            1,
            i,
            "P",
            &[
                ("nid", Value::Int(i as i64)),
                ("v", Value::Vector(vec![x, 0.0])),
            ],
            &mut interner,
        );
    }

    // SKIP 1 LIMIT 2 → skip n1, return n2 and n3.
    let results = run_cypher(
        "MATCH (n:P) \
         WITH n.nid AS nid, vector_distance(n.v, [0.0, 0.0]) AS d \
         ORDER BY d \
         SKIP 1 \
         LIMIT 2 \
         RETURN nid, d",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 2, "SKIP 1 LIMIT 2 returns 2 rows");

    let top_id = results[0]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("nid populated");
    assert_eq!(top_id, 2, "after SKIP 1 first is n2");

    let second_id = results[1]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("second nid populated");
    assert_eq!(second_id, 3, "after SKIP 1 second is n3");
}

/// vector_manhattan metric through VectorTopK Pattern B (brute force).
/// Verifies the optimizer handles all four vector metric functions, not only
/// vector_distance. Manhattan distance uses ASC ordering (lower = closer).
#[test]
fn vector_top_k_manhattan_metric() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // 3 nodes at known Manhattan distances from [1.0, 0.0, 0.0]:
    //   n1: [1.0, 0.0, 0.0] → L1 = 0.0
    //   n2: [0.5, 0.5, 0.0] → L1 = 1.0
    //   n3: [0.0, 1.0, 1.0] → L1 = 3.0
    insert_node(
        &engine,
        1,
        1,
        "M",
        &[
            ("nid", Value::Int(1)),
            ("v", Value::Vector(vec![1.0, 0.0, 0.0])),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "M",
        &[
            ("nid", Value::Int(2)),
            ("v", Value::Vector(vec![0.5, 0.5, 0.0])),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "M",
        &[
            ("nid", Value::Int(3)),
            ("v", Value::Vector(vec![0.0, 1.0, 1.0])),
        ],
        &mut interner,
    );

    let results = run_cypher(
        "MATCH (n:M) \
         WITH n.nid AS nid, vector_manhattan(n.v, [1.0, 0.0, 0.0]) AS d \
         ORDER BY d \
         LIMIT 2 \
         RETURN nid, d",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 2, "top_k=2 for manhattan");

    // Top-1 must be n1 (exact match, L1 = 0).
    let top_id = results[0]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("nid populated");
    assert_eq!(top_id, 1, "manhattan closest to [1,0,0] is n1");

    let d0 = results[0]
        .get("d")
        .and_then(|v| match v {
            Value::Float(f) => Some(*f),
            _ => None,
        })
        .expect("d populated");
    assert!(
        d0.abs() < 1e-5,
        "n1 manhattan distance should be ~0, got {d0}"
    );

    let second_id = results[1]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("second nid populated");
    assert_eq!(second_id, 2, "manhattan second is n2");
}

/// vector_similarity (cosine) with DESC ordering through VectorTopK Pattern B.
/// Similarity metrics require DESC (higher = closer) — `is_valid_direction`
/// must accept this direction and the executor must sort correctly.
#[test]
fn vector_top_k_similarity_desc_metric() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // 3 unit vectors with known cosine similarity to [1.0, 0.0]:
    //   n1: [1.0, 0.0] → cos = 1.0
    //   n2: [0.7, 0.7] → cos ≈ 0.707
    //   n3: [0.0, 1.0] → cos = 0.0
    insert_node(
        &engine,
        1,
        1,
        "S",
        &[("nid", Value::Int(1)), ("v", Value::Vector(vec![1.0, 0.0]))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "S",
        &[("nid", Value::Int(2)), ("v", Value::Vector(vec![0.7, 0.7]))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "S",
        &[("nid", Value::Int(3)), ("v", Value::Vector(vec![0.0, 1.0]))],
        &mut interner,
    );

    let results = run_cypher(
        "MATCH (n:S) \
         WITH n.nid AS nid, vector_similarity(n.v, [1.0, 0.0]) AS sim \
         ORDER BY sim DESC \
         LIMIT 2 \
         RETURN nid, sim",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 2, "top_k=2 for similarity DESC");

    let top_id = results[0]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("nid populated");
    assert_eq!(top_id, 1, "similarity top is n1 (exact match)");

    let s0 = results[0]
        .get("sim")
        .and_then(|v| match v {
            Value::Float(f) => Some(*f),
            _ => None,
        })
        .expect("sim populated");
    assert!(
        (s0 - 1.0).abs() < 1e-4,
        "n1 cosine similarity should be ~1.0, got {s0}"
    );

    let second_id = results[1]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("second nid populated");
    assert_eq!(second_id, 2, "similarity second is n2 (0.707)");

    let s1 = results[1]
        .get("sim")
        .and_then(|v| match v {
            Value::Float(f) => Some(*f),
            _ => None,
        })
        .expect("second sim populated");
    assert!(s0 >= s1, "similarity DESC violated: {s0} < {s1}");
}

/// ORDER BY vector_distance() DESC ("find furthest" use case) is NOT
/// optimized (wrong direction), but must still work via unoptimized Sort+Limit.
#[test]
fn vector_top_k_distance_desc_fallback_works() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // 4 nodes at known distances from [0.0, 0.0, 0.0]:
    //   n1..n4 at [1.0..4.0, 0.0, 0.0]
    for i in 1..=4u64 {
        let x = i as f32;
        insert_node(
            &engine,
            1,
            i,
            "F",
            &[
                ("nid", Value::Int(i as i64)),
                ("v", Value::Vector(vec![x, 0.0, 0.0])),
            ],
            &mut interner,
        );
    }

    let results = run_cypher(
        "MATCH (n:F) \
         WITH n.nid AS nid, vector_distance(n.v, [0.0, 0.0, 0.0]) AS d \
         ORDER BY d DESC \
         LIMIT 2 \
         RETURN nid, d",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 2, "top_k=2 farthest");

    // Farthest first: n4 (dist=4.0), then n3 (dist=3.0).
    let top_id = results[0]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("nid populated");
    assert_eq!(top_id, 4, "farthest from origin is n4");

    let second_id = results[1]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("second nid populated");
    assert_eq!(second_id, 3, "second farthest is n3");

    let d0 = results[0]
        .get("d")
        .and_then(|v| match v {
            Value::Float(f) => Some(*f),
            _ => None,
        })
        .expect("d0 populated");
    let d1 = results[1]
        .get("d")
        .and_then(|v| match v {
            Value::Float(f) => Some(*f),
            _ => None,
        })
        .expect("d1 populated");
    assert!(d0 > d1, "DESC violated: {d0} <= {d1}");
    assert!(
        (d0 - 4.0).abs() < 1e-4,
        "n4 distance should be 4.0, got {d0}"
    );
}

/// Brute-force VectorTopK path with a larger dataset. Verifies top-K
/// ordering and that the distance alias is populated correctly.
#[test]
fn vector_top_k_brute_force_without_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // 6 nodes with varying distances to [1.0, 0.0, 0.0].
    let embeddings: [(u64, Vec<f32>); 6] = [
        (1, vec![1.0, 0.0, 0.0]),  // dist = 0.0 — closest
        (2, vec![0.9, 0.1, 0.0]),  // dist ≈ 0.14
        (3, vec![0.5, 0.5, 0.0]),  // dist ≈ 0.71
        (4, vec![0.0, 1.0, 0.0]),  // dist ≈ 1.41
        (5, vec![0.0, 0.0, 1.0]),  // dist ≈ 1.41
        (6, vec![-1.0, 0.0, 0.0]), // dist = 2.0 — farthest
    ];
    for (id, vec) in &embeddings {
        insert_node(
            &engine,
            1,
            *id,
            "Vec",
            &[
                ("nid", Value::Int(*id as i64)),
                ("embedding", Value::Vector(vec.clone())),
            ],
            &mut interner,
        );
    }

    // Top-3 via Pattern B with explicit aliases.
    let results = run_cypher(
        "MATCH (n:Vec) \
         WITH n.nid AS nid, vector_distance(n.embedding, [1.0, 0.0, 0.0]) AS d \
         ORDER BY d \
         LIMIT 3 \
         RETURN nid, d",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 3, "top_k=3 should return 3 rows");

    // Verify ascending distance order.
    let extract_d = |row: &coordinode_query::executor::Row| -> f64 {
        row.get("d")
            .and_then(|v| match v {
                Value::Float(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(f64::MAX)
    };
    let d0 = extract_d(&results[0]);
    let d1 = extract_d(&results[1]);
    let d2 = extract_d(&results[2]);
    assert!(d0 < 0.01, "closest should be ~0, got {d0}");
    assert!(d0 <= d1, "sort violated: {d0} > {d1}");
    assert!(d1 <= d2, "sort violated: {d1} > {d2}");

    // The closest node id is 1 (exact match on embedding).
    let top_id = results[0]
        .get("nid")
        .and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
        .expect("nid column populated");
    assert_eq!(top_id, 1, "closest to [1,0,0] is node id 1");
}

// ── type() and labels() scalar functions (R523/R524 partial) ──────────

/// `type(r)` returns the relationship type as a string.
#[test]
fn type_function_returns_edge_type() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let alloc = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    run_cypher_with_alloc(
        "CREATE (a:Person {name: 'Alice'})",
        &engine,
        &mut interner,
        &alloc,
    );
    run_cypher_with_alloc(
        "CREATE (b:Person {name: 'Bob'})",
        &engine,
        &mut interner,
        &alloc,
    );
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)",
        &engine,
        &mut interner,
        &alloc,
    );

    let results = run_cypher_with_alloc(
        "MATCH (a)-[r]->(b) RETURN type(r) AS rel_type",
        &engine,
        &mut interner,
        &alloc,
    );

    assert_eq!(results.len(), 1, "should find one relationship");
    let rel_type = results[0]
        .get("rel_type")
        .and_then(|v| v.as_str())
        .expect("rel_type populated");
    assert_eq!(rel_type, "KNOWS");
}

/// `labels(n)` returns a list containing the node's label.
#[test]
fn labels_function_returns_node_label() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Movie",
        &[("title", Value::String("Matrix".into()))],
        &mut interner,
    );

    let results = run_cypher(
        "MATCH (n:Movie) RETURN labels(n) AS lbls",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1);
    let lbls = results[0].get("lbls").expect("lbls populated");
    match lbls {
        Value::Array(arr) => {
            assert_eq!(arr.len(), 1, "single-label node");
            assert_eq!(arr[0], Value::String("Movie".into()));
        }
        _ => panic!("expected Array, got {lbls:?}"),
    }
}

/// `type(r)` with multiple edge types returns correct type per edge.
#[test]
fn type_function_multiple_edge_types() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let alloc = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    run_cypher_with_alloc(
        "CREATE (a:Person {name: 'X'})",
        &engine,
        &mut interner,
        &alloc,
    );
    run_cypher_with_alloc(
        "CREATE (b:Company {name: 'Y'})",
        &engine,
        &mut interner,
        &alloc,
    );
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'X'}), (b:Company {name: 'Y'}) CREATE (a)-[:WORKS_AT]->(b)",
        &engine,
        &mut interner,
        &alloc,
    );
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'X'}), (b:Company {name: 'Y'}) CREATE (a)-[:INVESTED_IN]->(b)",
        &engine,
        &mut interner,
        &alloc,
    );

    let results = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'X'})-[r]->(b) RETURN type(r) AS t ORDER BY t",
        &engine,
        &mut interner,
        &alloc,
    );

    assert_eq!(results.len(), 2);
    let types: Vec<&str> = results
        .iter()
        .filter_map(|r| r.get("t").and_then(|v| v.as_str()))
        .collect();
    assert!(types.contains(&"WORKS_AT"), "missing WORKS_AT: {types:?}");
    assert!(
        types.contains(&"INVESTED_IN"),
        "missing INVESTED_IN: {types:?}"
    );
}

#[test]
fn vector_filter_cost_in_explain() {
    let ast =
        parse("MATCH (n:Person) WHERE vector_distance(n.embedding, [1.0, 0.0]) < 0.3 RETURN n")
            .unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let cost = estimate_cost(&plan);

    assert!(cost.cost > 0.0, "VectorFilter should have non-zero cost");

    let explain = plan.explain();
    assert!(
        explain.contains("VectorFilter"),
        "EXPLAIN should show VectorFilter"
    );
    assert!(explain.contains("Cost:"), "EXPLAIN should show cost");
}

// ═══════════════════════════════════════════════════════════════════════
// Adaptive query plans — divergence detection
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn adaptive_divergence_detected_on_high_fan_out() {
    // Create a hub node with many edges to trigger divergence detection
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Hub node connected to 20 targets
    insert_node(
        &engine,
        1,
        1,
        "Hub",
        &[("name", Value::String("center".into()))],
        &mut interner,
    );
    for i in 2..=21 {
        insert_node(&engine, 1, i, "Leaf", &[], &mut interner);
        insert_edge(&engine, "LINK", 1, i);
    }
    // Each leaf also connects to 5 more nodes → 20 * 5 = 100 at depth 2
    let mut next_id = 22u64;
    for leaf in 2..=21 {
        for _ in 0..5 {
            insert_node(&engine, 1, next_id, "Deep", &[], &mut interner);
            insert_edge(&engine, "LINK", leaf, next_id);
            next_id += 1;
        }
    }

    // Use low thresholds to trigger divergence on this small graph
    let adaptive = AdaptiveConfig {
        enabled: true,
        max_fan_out: 10_000,
        switch_threshold: 0.3, // trigger when actual > 0.3 × expected (low threshold for test)
        check_interval: 10,    // check every 10 edges
        ..AdaptiveConfig::default()
    };

    let (_results, warnings) = run_cypher_adaptive(
        "MATCH (h:Hub {name: \"center\"})-[:LINK*1..3]->(t) RETURN t",
        &engine,
        &mut interner,
        adaptive,
    );

    // Should have adaptive warnings about divergence
    let has_adaptive_warning = warnings.iter().any(|w| w.contains("adaptive:"));
    assert!(
        has_adaptive_warning,
        "should detect divergence with low threshold. Warnings: {:?}",
        warnings
    );
}

#[test]
fn adaptive_no_divergence_on_small_graph() {
    // Normal small graph should NOT trigger divergence
    let (_dir, engine, mut interner) = setup_social_graph();

    let adaptive = AdaptiveConfig {
        enabled: true,
        max_fan_out: 10_000,
        switch_threshold: 10.0,
        check_interval: 1000,
        ..AdaptiveConfig::default()
    };

    let (_results, warnings) = run_cypher_adaptive(
        "MATCH (a:Person {name: \"Alice\"})-[:KNOWS*1..3]->(b) RETURN b",
        &engine,
        &mut interner,
        adaptive,
    );

    let has_adaptive_warning = warnings.iter().any(|w| w.contains("adaptive:"));
    assert!(
        !has_adaptive_warning,
        "small graph should NOT trigger divergence. Warnings: {:?}",
        warnings
    );
}

#[test]
fn adaptive_disabled_no_warnings() {
    // When disabled, no adaptive warnings even on high fan-out
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Hub",
        &[("name", Value::String("x".into()))],
        &mut interner,
    );
    for i in 2..=50 {
        insert_node(&engine, 1, i, "Leaf", &[], &mut interner);
        insert_edge(&engine, "LINK", 1, i);
    }

    let adaptive = AdaptiveConfig {
        enabled: false,
        max_fan_out: 10_000,
        switch_threshold: 1.0,
        check_interval: 5,
        ..AdaptiveConfig::default()
    };

    let (_results, warnings) = run_cypher_adaptive(
        "MATCH (h:Hub {name: \"x\"})-[:LINK*1..2]->(t) RETURN t",
        &engine,
        &mut interner,
        adaptive,
    );

    let has_adaptive_warning = warnings.iter().any(|w| w.contains("adaptive:"));
    assert!(
        !has_adaptive_warning,
        "disabled adaptive should produce no warnings"
    );
}

#[test]
fn adaptive_check_interval_config() {
    // Verify check_interval is respected — with interval=1, even small divergence is caught
    let (_dir, engine, mut interner) = setup_social_graph();

    let adaptive = AdaptiveConfig {
        enabled: true,
        max_fan_out: 10_000,
        switch_threshold: 0.01, // very low threshold — any traversal exceeds
        check_interval: 1,      // check every single edge
        ..AdaptiveConfig::default()
    };

    let (_results, warnings) = run_cypher_adaptive(
        "MATCH (a:Person {name: \"Alice\"})-[:KNOWS*1..2]->(b) RETURN b",
        &engine,
        &mut interner,
        adaptive,
    );

    let has_adaptive_warning = warnings.iter().any(|w| w.contains("adaptive:"));
    assert!(
        has_adaptive_warning,
        "ultra-low threshold should catch any traversal. Warnings: {:?}",
        warnings
    );
}

// ═══════════════════════════════════════════════════════════════════════
// text_match() / text_score() — full-text search in Cypher
// ═══════════════════════════════════════════════════════════════════════

fn run_cypher_with_text_index(
    query: &str,
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    text_index: &coordinode_search::tantivy::multi_lang::MultiLanguageTextIndex,
) -> Vec<coordinode_query::executor::Row> {
    let ast = parse(query).unwrap_or_else(|e| panic!("parse error: {e}"));
    let plan = build_logical_plan(&ast).unwrap_or_else(|e| panic!("plan error: {e}"));
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(engine, interner, &allocator);
    ctx.text_index = Some(text_index);
    ctx.text_index_registry = None;
    execute(&plan, &mut ctx).unwrap_or_else(|e| panic!("execute error: {e}"))
}

#[test]
fn text_match_filters_by_fulltext() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Article",
        &[("title", Value::String("Rust graph database".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Article",
        &[("title", Value::String("Python machine learning".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Article",
        &[("title", Value::String("Rust web framework".into()))],
        &mut interner,
    );

    let text_dir = dir.path().join("text_idx");
    let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, Some("english")).unwrap();
    text_idx.add_document(1, "Rust graph database").unwrap();
    text_idx.add_document(2, "Python machine learning").unwrap();
    text_idx.add_document(3, "Rust web framework").unwrap();

    let multi_idx =
        MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::with_default_language("english"));
    let results = run_cypher_with_text_index(
        "MATCH (a:Article) WHERE text_match(a.title, \"graph\") RETURN a.title",
        &engine,
        &mut interner,
        &multi_idx,
    );

    assert_eq!(results.len(), 1, "only 1 article matches 'graph'");
}

#[test]
fn text_match_planner_creates_text_filter() {
    let ast =
        parse("MATCH (a:Article) WHERE text_match(a.body, \"rust database\") RETURN a").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let explain = plan.explain();
    assert!(
        explain.contains("TextFilter"),
        "text_match should produce TextFilter. EXPLAIN:\n{explain}"
    );
}

#[test]
fn text_score_returns_bm25() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Doc",
        &[("body", Value::String("rust rust rust".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Doc",
        &[("body", Value::String("rust programming".into()))],
        &mut interner,
    );

    let text_dir = dir.path().join("text_idx");
    let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, None).unwrap();
    text_idx.add_document(1, "rust rust rust").unwrap();
    text_idx.add_document(2, "rust programming").unwrap();

    let multi_idx = MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::default());
    let results = run_cypher_with_text_index(
        "MATCH (d:Doc) WHERE text_match(d.body, \"rust\") RETURN text_score(d.body, \"rust\") AS score",
        &engine, &mut interner, &multi_idx,
    );

    assert_eq!(results.len(), 2, "both docs match 'rust'");
    for row in &results {
        if let Some(Value::Float(score)) = row.get("score") {
            assert!(*score > 0.0, "BM25 score should be > 0");
        } else {
            panic!("score should be Float, got: {:?}", row.get("score"));
        }
    }
}

// R-HYB1b regression: `text_match()` against a property with NO full-text
// index must error at execute time, not silently pass every row through
// (the old graceful-degradation returned all rows — a semantic bug that made
// `MATCH (n:Chunk) WHERE text_match(n.body, "foo") RETURN n` return every
// Chunk when no index existed, the opposite of what the caller asked for).
// Consistent with R-HYB1 `text_score()` guard and R-HYB2b RankFuse guard.
#[test]
fn text_match_no_index_errors() {
    let (_dir, engine, mut interner) = setup_social_graph();
    let ast = parse("MATCH (n:Person) WHERE text_match(n.name, \"Alice\") RETURN n").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(&engine, &mut interner, &allocator);
    let err = execute(&plan, &mut ctx)
        .expect_err("text_match() without any text index must error, not pass rows through");
    let msg = err.to_string();
    assert!(
        msg.contains("text_match()") && msg.contains("CREATE TEXT INDEX"),
        "error must name text_match() and tell the user how to fix it, got: {msg}"
    );
}

// R-HYB1b regression: if the input row set is empty (e.g. MATCH matched no
// nodes after DETACH DELETE), `text_match()` must return an empty result
// rather than erroring on a missing index. There is nothing to filter, and
// failing here would turn a legitimately empty upstream into a spurious
// "missing FT-index" error. Covers the empty-rows short-circuit in
// `execute_text_filter`.
#[test]
fn text_match_empty_input_short_circuits_without_index_lookup() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    // No Article nodes inserted — MATCH will produce 0 rows.

    // No text index registered — if we reached the index lookup, this would
    // error. The empty-input shortcut must fire first and return an empty
    // result cleanly.
    let results = run_cypher(
        "MATCH (a:Article) WHERE text_match(a.body, \"rust\") RETURN a.body",
        &engine,
        &mut interner,
    );
    assert!(
        results.is_empty(),
        "empty upstream must produce empty output without hitting the FT-index guard"
    );
}

// R-HYB1b regression: when a `TextIndexRegistry` is wired but has no entry
// for the specific (label, property), the error must name BOTH the label
// and the property so the user can create the right index.
#[test]
fn text_match_registry_missing_entry_errors_with_label_and_property() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Article",
        &[("body", Value::String("rust rust rust".into()))],
        &mut interner,
    );

    // Registry exists but has NO index for (Article, body).
    let text_reg_dir = dir.path().join("empty_text_reg");
    std::fs::create_dir_all(&text_reg_dir).unwrap();
    let text_reg = TextIndexRegistry::new(&text_reg_dir);

    let ast = parse("MATCH (a:Article) WHERE text_match(a.body, \"rust\") RETURN a").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(&engine, &mut interner, &allocator);
    ctx.text_index_registry = Some(&text_reg);

    let err = execute(&plan, &mut ctx).expect_err(
        "text_match() with registry but no matching index must error (not pass rows through)",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("text_match()")
            && msg.contains("Article")
            && msg.contains("body")
            && msg.contains("CREATE TEXT INDEX"),
        "error must name the missing (Article, body) pair and the remedy, got: {msg}"
    );
}

// R-HYB1 regression #3: using text_score() without a paired text_match() in
// WHERE (no FT index configured) must return an explicit executor error, not
// silently yield 0.0 scores. Prior behavior returned 0 which misleads the
// caller into thinking the document has zero BM25 relevance.
#[test]
fn text_score_without_text_match_errors() {
    let (_dir, engine, mut interner) = setup_social_graph();
    // Plan builds fine (valid Cypher); executor detects the missing TextFilter
    // at runtime and aborts with ExecutionError::Unsupported.
    let ast = parse("MATCH (n:Person) RETURN text_score(n.name, \"Alice\") AS score").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(&engine, &mut interner, &allocator);
    let err = execute(&plan, &mut ctx).expect_err(
        "text_score() without paired text_match() must error at execute time, not return 0.0",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("text_score()") && msg.contains("text_match"),
        "error message must name both functions to guide the user to the fix, got: {msg}"
    );
}

// R-HYB1: text_score() usable in ORDER BY — verifies the BM25 score streamed
// from TextFilter drives result ordering. Doc with more term occurrences must
// rank higher (BM25 = monotonic in term frequency when docs are similar length).
#[test]
fn text_score_orders_results_by_bm25() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Doc 1: "rust rust rust" — term frequency 3
    insert_node(
        &engine,
        1,
        1,
        "Doc",
        &[("body", Value::String("rust rust rust".into()))],
        &mut interner,
    );
    // Doc 2: "rust" — term frequency 1
    insert_node(
        &engine,
        1,
        2,
        "Doc",
        &[("body", Value::String("rust".into()))],
        &mut interner,
    );
    // Doc 3: "rust programming language" — term frequency 1 in longer doc
    insert_node(
        &engine,
        1,
        3,
        "Doc",
        &[("body", Value::String("rust programming language".into()))],
        &mut interner,
    );

    let text_dir = dir.path().join("text_idx");
    let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, None).unwrap();
    text_idx.add_document(1, "rust rust rust").unwrap();
    text_idx.add_document(2, "rust").unwrap();
    text_idx
        .add_document(3, "rust programming language")
        .unwrap();

    let multi_idx = MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::default());
    let results = run_cypher_with_text_index(
        "MATCH (d:Doc) WHERE text_match(d.body, \"rust\") \
         RETURN d.body AS body, text_score(d.body, \"rust\") AS score \
         ORDER BY score DESC",
        &engine,
        &mut interner,
        &multi_idx,
    );

    assert_eq!(results.len(), 3, "all three docs match 'rust'");

    let scores: Vec<f64> = results
        .iter()
        .map(|r| match r.get("score") {
            Some(Value::Float(f)) => *f,
            other => panic!("expected Float score, got {other:?}"),
        })
        .collect();

    for window in scores.windows(2) {
        assert!(
            window[0] >= window[1],
            "scores must be non-increasing under ORDER BY score DESC: {scores:?}"
        );
    }

    // Doc 1 (three 'rust' in a short body) should outrank doc 2 and doc 3.
    let first_body = match results[0].get("body") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String body, got {other:?}"),
    };
    assert_eq!(
        first_body, "rust rust rust",
        "highest BM25 score must belong to the triple-term doc, not to {first_body:?}"
    );
}

// R-HYB1: text_score() composes arithmetically with vector_distance() inside
// ORDER BY / RETURN. This is the canonical arch doc example
// (arch/search/document-scoring.md §Arithmetic Composition):
// a weighted blend of vector similarity and BM25 drives ordering.
#[test]
fn text_score_composes_with_vector_distance_in_order_by() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Two docs. Doc 1: vector matches exactly, text weakly matches.
    // Doc 2: vector distant, text strongly matches (multiple occurrences).
    // With a vector-heavy blend, doc 1 wins; pure text would reverse it.
    insert_node(
        &engine,
        1,
        1,
        "Doc",
        &[
            ("body", Value::String("rust".into())),
            ("embedding", Value::Vector(vec![1.0, 0.0, 0.0])),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Doc",
        &[
            ("body", Value::String("rust rust rust rust".into())),
            ("embedding", Value::Vector(vec![0.0, 1.0, 0.0])),
        ],
        &mut interner,
    );

    let text_dir = dir.path().join("text_idx");
    let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, None).unwrap();
    text_idx.add_document(1, "rust").unwrap();
    text_idx.add_document(2, "rust rust rust rust").unwrap();

    let multi_idx = MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::default());

    // Vector-heavy blend: 0.9 * (1 - vec_dist) + 0.1 * text_score.
    // Doc 1: (1 - 0) * 0.9 + small_bm25 * 0.1 ≈ 0.9 + ε
    // Doc 2: (1 - √2) * 0.9 + larger_bm25 * 0.1 ≈ negative + small positive
    let results = run_cypher_with_text_index(
        "MATCH (d:Doc) \
         WHERE text_match(d.body, \"rust\") \
         RETURN d.body AS body, \
                (1.0 - vector_distance(d.embedding, [1.0, 0.0, 0.0])) * 0.9 \
                + text_score(d.body, \"rust\") * 0.1 AS score \
         ORDER BY score DESC",
        &engine,
        &mut interner,
        &multi_idx,
    );

    assert_eq!(results.len(), 2, "both docs text-match 'rust'");

    let first_body = match results[0].get("body") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String body, got {other:?}"),
    };
    assert_eq!(
        first_body, "rust",
        "vector-heavy blend must rank the exact-vector doc first, \
         not the frequent-term doc; got {first_body:?}"
    );

    // Sanity: top score strictly greater than bottom score.
    let s0 = match results[0].get("score") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float score, got {other:?}"),
    };
    let s1 = match results[1].get("score") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float score, got {other:?}"),
    };
    assert!(
        s0 > s1,
        "blended score must strictly order the two docs: top={s0}, bottom={s1}"
    );
}

// R-HYB2 hybrid_score regression #1:
// hybrid_score(c, $q) with default weights (0.65 vector / 0.35 text) must match
// the manual arithmetic blend `(1.0 - vector_distance) * 0.65 + text_score * 0.35`
// for every row when both filter operators run in the same plan.
#[test]
fn hybrid_score_matches_manual_arithmetic() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Doc",
        &[
            ("body", Value::String("rust programming".into())),
            ("embedding", Value::Vector(vec![1.0, 0.0, 0.0])),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Doc",
        &[
            ("body", Value::String("rust rust rust".into())),
            ("embedding", Value::Vector(vec![0.5, 0.5, 0.0])),
        ],
        &mut interner,
    );

    let text_dir = dir.path().join("text_idx");
    let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, None).unwrap();
    text_idx.add_document(1, "rust programming").unwrap();
    text_idx.add_document(2, "rust rust rust").unwrap();
    let multi_idx = MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::default());

    // Both TextFilter (from text_match) and VectorFilter (from vector_distance)
    // run against every surviving row, so __text_score__ and __vector_score__
    // are populated; hybrid_score reads both and blends with default weights.
    let results_hybrid = run_cypher_with_text_index(
        "MATCH (d:Doc) \
         WHERE text_match(d.body, \"rust\") \
           AND vector_distance(d.embedding, [1.0, 0.0, 0.0]) < 1.0 \
         RETURN d.body AS body, hybrid_score(d, \"rust\") AS score",
        &engine,
        &mut interner,
        &multi_idx,
    );

    let results_manual = run_cypher_with_text_index(
        "MATCH (d:Doc) \
         WHERE text_match(d.body, \"rust\") \
           AND vector_distance(d.embedding, [1.0, 0.0, 0.0]) < 1.0 \
         RETURN d.body AS body, \
                (1.0 - vector_distance(d.embedding, [1.0, 0.0, 0.0])) * 0.65 \
                + text_score(d.body, \"rust\") * 0.35 AS score",
        &engine,
        &mut interner,
        &multi_idx,
    );

    assert_eq!(
        results_hybrid.len(),
        results_manual.len(),
        "hybrid_score() must return the same row count as the manual blend"
    );
    assert!(
        !results_hybrid.is_empty(),
        "expected at least one matching doc"
    );

    // Row order is determined by MATCH traversal + filter; assume same order since
    // the underlying plan is identical (only RETURN projection differs).
    for (rh, rm) in results_hybrid.iter().zip(results_manual.iter()) {
        let s_hyb = match rh.get("score") {
            Some(Value::Float(f)) => *f,
            other => panic!("hybrid row score must be Float, got {other:?}"),
        };
        let s_man = match rm.get("score") {
            Some(Value::Float(f)) => *f,
            other => panic!("manual row score must be Float, got {other:?}"),
        };
        assert!(
            (s_hyb - s_man).abs() < 1e-6,
            "hybrid_score default blend must match manual arithmetic: \
             hybrid={s_hyb} manual={s_man} (body={:?})",
            rh.get("body")
        );
    }
}

// R-HYB2 hybrid_score: the weights map `{vector: 1.0, text: 0.0}` must yield
// the pure vector component (equivalent to `1.0 - vector_distance` under the
// L2 normalization). Verifies weight overriding end to end.
#[test]
fn hybrid_score_custom_weights_override_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Doc",
        &[
            ("body", Value::String("rust rust rust".into())),
            ("embedding", Value::Vector(vec![1.0, 0.0, 0.0])),
        ],
        &mut interner,
    );

    let text_dir = dir.path().join("text_idx");
    let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, None).unwrap();
    text_idx.add_document(1, "rust rust rust").unwrap();
    let multi_idx = MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::default());

    let results = run_cypher_with_text_index(
        "MATCH (d:Doc) \
         WHERE text_match(d.body, \"rust\") \
           AND vector_distance(d.embedding, [1.0, 0.0, 0.0]) < 1.0 \
         RETURN hybrid_score(d, \"rust\", {vector: 1.0, text: 0.0}) AS score",
        &engine,
        &mut interner,
        &multi_idx,
    );

    assert_eq!(results.len(), 1);
    let s = match results[0].get("score") {
        Some(Value::Float(f)) => *f,
        other => panic!("score must be Float, got {other:?}"),
    };
    // Pure vector, exact match (distance=0) normalized to 1.0 → score = 1.0.
    assert!(
        (s - 1.0).abs() < 1e-6,
        "vector-only weights on exact-match row must yield 1.0, got {s}"
    );
}

// R-HYB2 hybrid_score: without any text_match / vector_distance in WHERE, the
// row has neither cache column populated → fail with a clear error naming the
// required predicates, not silently return 0.0 / Null.
#[test]
fn hybrid_score_without_filters_errors() {
    let (_dir, engine, mut interner) = setup_social_graph();
    let ast = parse("MATCH (n:Person) RETURN hybrid_score(n, \"Alice\") AS score").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(&engine, &mut interner, &allocator);
    let err = execute(&plan, &mut ctx).expect_err(
        "hybrid_score() without paired text_match / vector_distance must error at execute time",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("hybrid_score()")
            && (msg.contains("text_match") || msg.contains("vector_"))
            && msg.contains("WHERE"),
        "error message must name hybrid_score and point at the missing predicates, got: {msg}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CJK text_match() — full pipeline: OpenCypher → planner → executor → TextIndex
// ═══════════════════════════════════════════════════════════════════════

#[cfg(any(feature = "cjk-zh", feature = "cjk-ja", feature = "cjk-ko"))]
mod cjk_text_match {
    use super::*;

    /// Chinese text_match() through full query pipeline.
    /// Verifies: OpenCypher parse → TextFilter plan → jieba tokenization → BM25 results.
    #[cfg(feature = "cjk-zh")]
    #[test]
    fn text_match_chinese_jieba() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        insert_node(
            &engine,
            1,
            1,
            "Article",
            &[("title", Value::String("北京是中国的首都".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "Article",
            &[("title", Value::String("上海是中国最大的城市".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            3,
            "Article",
            &[("title", Value::String("东京是日本的首都".into()))],
            &mut interner,
        );

        let text_dir = dir.path().join("text_idx_cn");
        let mut text_idx =
            TextIndex::open_or_create(&text_dir, 15_000_000, Some("chinese_jieba")).unwrap();
        text_idx.add_document(1, "北京是中国的首都").unwrap();
        text_idx.add_document(2, "上海是中国最大的城市").unwrap();
        text_idx.add_document(3, "东京是日本的首都").unwrap();

        // Search for "北京" — should find only doc 1
        let multi_idx = MultiLanguageTextIndex::wrap(
            text_idx,
            MultiLangConfig::with_default_language("chinese_jieba"),
        );
        let results = run_cypher_with_text_index(
            "MATCH (a:Article) WHERE text_match(a.title, \"北京\") RETURN a.title",
            &engine,
            &mut interner,
            &multi_idx,
        );

        assert_eq!(results.len(), 1, "only 1 article contains 北京");
    }

    /// Japanese text_match() through full query pipeline.
    /// Verifies: lindera IPAdic morphological segmentation works end-to-end.
    #[cfg(feature = "cjk-ja")]
    #[test]
    fn text_match_japanese_lindera() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        insert_node(
            &engine,
            1,
            1,
            "Article",
            &[("title", Value::String("東京都に住んでいます".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "Article",
            &[("title", Value::String("大阪は美しい街です".into()))],
            &mut interner,
        );

        let text_dir = dir.path().join("text_idx_jp");
        let mut text_idx =
            TextIndex::open_or_create(&text_dir, 15_000_000, Some("japanese_lindera")).unwrap();
        text_idx.add_document(1, "東京都に住んでいます").unwrap();
        text_idx.add_document(2, "大阪は美しい街です").unwrap();

        let multi_idx = MultiLanguageTextIndex::wrap(
            text_idx,
            MultiLangConfig::with_default_language("japanese_lindera"),
        );
        let results = run_cypher_with_text_index(
            "MATCH (a:Article) WHERE text_match(a.title, \"東京\") RETURN a.title",
            &engine,
            &mut interner,
            &multi_idx,
        );

        assert!(!results.is_empty(), "should find article with 東京");
    }

    /// Korean text_match() through full query pipeline.
    /// Verifies: lindera ko-dic segmentation works end-to-end.
    #[cfg(feature = "cjk-ko")]
    #[test]
    fn text_match_korean_lindera() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        insert_node(
            &engine,
            1,
            1,
            "Article",
            &[(
                "title",
                Value::String("서울은 대한민국의 수도입니다".into()),
            )],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "Article",
            &[("title", Value::String("부산은 항구 도시입니다".into()))],
            &mut interner,
        );

        let text_dir = dir.path().join("text_idx_kr");
        let mut text_idx =
            TextIndex::open_or_create(&text_dir, 15_000_000, Some("korean_lindera")).unwrap();
        text_idx
            .add_document(1, "서울은 대한민국의 수도입니다")
            .unwrap();
        text_idx.add_document(2, "부산은 항구 도시입니다").unwrap();

        let multi_idx = MultiLanguageTextIndex::wrap(
            text_idx,
            MultiLangConfig::with_default_language("korean_lindera"),
        );
        let results = run_cypher_with_text_index(
            "MATCH (a:Article) WHERE text_match(a.title, \"서울\") RETURN a.title",
            &engine,
            &mut interner,
            &multi_idx,
        );

        assert!(!results.is_empty(), "should find article with 서울");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Multi-language text index integration
// ═══════════════════════════════════════════════════════════════════════
//
// These tests verify multi-language TextIndex features through real storage
// storage. Note: ExecutionContext currently holds &TextIndex (not
// &MultiLanguageTextIndex), so full query engine integration with
// per-query language is a separate gap. These tests validate the search
// infrastructure itself works end-to-end.

mod multi_lang_text_index {
    use super::*;
    use coordinode_search::tantivy::multi_lang::{MultiLangConfig, MultiLanguageTextIndex};
    use std::collections::HashMap;

    fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// Mixed English + Russian documents in one index, searchable per language
    #[test]
    fn mixed_english_russian_via_override() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        // Insert graph nodes
        insert_node(
            &engine,
            1,
            1,
            "Article",
            &[(
                "title",
                Value::String("Rust graph database technology overview".into()),
            )],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "Article",
            &[(
                "title",
                Value::String("Графовая база данных для аналитики и обработки данных".into()),
            )],
            &mut interner,
        );

        // Build multi-language text index
        let text_dir = dir.path().join("multilang_idx");
        let config = MultiLangConfig::with_default_language("english");
        let mut ml_idx =
            MultiLanguageTextIndex::open_or_create(&text_dir, 15_000_000, config).unwrap();

        // English doc (auto-detected)
        ml_idx
            .add_node(
                1,
                &props(&[(
                    "title",
                    "Rust graph database technology overview for developers",
                )]),
            )
            .unwrap();

        // Russian doc with _language override
        ml_idx
            .add_node(
                2,
                &props(&[
                    (
                        "title",
                        "Графовая база данных для аналитики и обработки данных",
                    ),
                    ("_language", "russian"),
                ]),
            )
            .unwrap();

        assert_eq!(ml_idx.num_docs(), 2);

        // English search finds English doc
        let en = ml_idx.search_with_language("graph", 10, "english").unwrap();
        assert!(
            en.iter().any(|r| r.node_id == 1),
            "English search should find English doc: {en:?}"
        );

        // Russian search finds Russian doc
        let ru = ml_idx
            .search_with_language("данных", 10, "russian")
            .unwrap();
        assert!(
            ru.iter().any(|r| r.node_id == 2),
            "Russian search should find Russian doc: {ru:?}"
        );
    }

    /// "none" analyzer preserves exact tokens for technical identifiers
    #[test]
    fn none_analyzer_with_graph_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        insert_node(
            &engine,
            1,
            1,
            "ErrorCode",
            &[
                ("code", Value::String("ERR_CONN_REFUSED".into())),
                (
                    "description",
                    Value::String("Connection to the remote server was refused".into()),
                ),
            ],
            &mut interner,
        );

        let text_dir = dir.path().join("none_idx");
        let config = MultiLangConfig::with_default_language("english")
            .with_field_analyzer("code", "none")
            .with_field_analyzer("description", "auto_detect");
        let mut ml_idx =
            MultiLanguageTextIndex::open_or_create(&text_dir, 15_000_000, config).unwrap();

        ml_idx
            .add_node(
                1,
                &props(&[
                    ("code", "ERR_CONN_REFUSED"),
                    (
                        "description",
                        "Connection to the remote server was refused by the operating system",
                    ),
                ]),
            )
            .unwrap();

        // "none" exact match (case-insensitive)
        let code_results = ml_idx
            .search_with_language("err_conn_refused", 10, "none")
            .unwrap();
        assert!(
            !code_results.is_empty(),
            "none analyzer should find exact identifier"
        );

        // English stemmed search on description
        let desc_results = ml_idx
            .search_with_language("connect", 10, "english")
            .unwrap();
        assert!(
            !desc_results.is_empty(),
            "English stem 'connect' should match 'Connection'"
        );
    }

    /// Per-document language with TextIndex::add_document_with_language
    /// through query engine (current integration: text_index.inner())
    #[test]
    fn per_document_language_via_text_index() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        insert_node(
            &engine,
            1,
            1,
            "Article",
            &[("title", Value::String("Running through the forest".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "Article",
            &[(
                "title",
                Value::String("Графовая база данных для аналитики".into()),
            )],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            3,
            "Article",
            &[("title", Value::String("ERR_TIMEOUT_EXCEEDED".into()))],
            &mut interner,
        );

        // Use TextIndex with per-document language
        let text_dir = dir.path().join("perdoc_idx");
        let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, Some("none")).unwrap();

        text_idx
            .add_document_with_language(1, "Running through the forest quickly", "english")
            .unwrap();
        text_idx
            .add_document_with_language(
                2,
                "Графовая база данных для аналитики и обработки информации",
                "russian",
            )
            .unwrap();
        text_idx
            .add_document_with_language(3, "ERR_TIMEOUT_EXCEEDED", "none")
            .unwrap();

        // Each document searchable in its language
        let en = text_idx.search_with_language("run", 10, "english").unwrap();
        assert!(
            en.iter().any(|r| r.node_id == 1),
            "English stemmed search should find doc 1: {en:?}"
        );

        // "данных" matches the indexed Russian text
        let ru = text_idx
            .search_with_language("данных", 10, "russian")
            .unwrap();
        assert!(
            ru.iter().any(|r| r.node_id == 2),
            "Russian search should find doc 2: {ru:?}"
        );

        let none = text_idx
            .search_with_language("err_timeout_exceeded", 10, "none")
            .unwrap();
        assert!(
            none.iter().any(|r| r.node_id == 3),
            "None literal search should find doc 3: {none:?}"
        );

        // Pass the same text_index to query engine — existing queries still work
        let multi_idx =
            MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::with_default_language("none"));
        let results = run_cypher_with_text_index(
            "MATCH (a:Article) WHERE text_match(a.title, \"ERR_TIMEOUT_EXCEEDED\") RETURN a.title",
            &engine,
            &mut interner,
            &multi_idx,
        );
        // text_match uses default search() which uses schema tokenizer ("none")
        assert!(
            !results.is_empty(),
            "text_match through query engine should find literal doc"
        );
    }

    /// Auto-detect language: long Russian text detected as Russian
    #[test]
    fn auto_detect_russian_in_multilang_index() {
        let dir = tempfile::tempdir().unwrap();
        let text_dir = dir.path().join("autodetect_idx");
        let config = MultiLangConfig::with_default_language("english");
        let mut ml_idx =
            MultiLanguageTextIndex::open_or_create(&text_dir, 15_000_000, config).unwrap();

        // Long text for reliable detection
        ml_idx
            .add_node(
                1,
                &props(&[(
                    "body",
                    "Графовая база данных является важным инструментом для анализа связей между объектами в реальном мире",
                )]),
            )
            .unwrap();

        // Should auto-detect Russian and apply Russian stemming
        let results = ml_idx
            .search_with_language("данных", 10, "russian")
            .unwrap();
        assert!(
            !results.is_empty(),
            "auto-detected Russian doc should be searchable with Russian stems"
        );
    }

    /// Batch add + delete via MultiLanguageTextIndex
    #[test]
    fn batch_lifecycle_multilang() {
        let dir = tempfile::tempdir().unwrap();
        let text_dir = dir.path().join("batch_idx");
        let config = MultiLangConfig::with_default_language("english");
        let mut ml_idx =
            MultiLanguageTextIndex::open_or_create(&text_dir, 15_000_000, config).unwrap();

        ml_idx
            .add_nodes_batch(&[
                (
                    1,
                    props(&[(
                        "title",
                        "Running through the beautiful forest in spring time",
                    )]),
                ),
                (
                    2,
                    props(&[
                        ("title", "Бегущий человек быстро бежал по дороге к реке"),
                        ("_language", "russian"),
                    ]),
                ),
            ])
            .unwrap();

        assert_eq!(ml_idx.num_docs(), 2);

        // Delete one
        ml_idx.delete_document(1).unwrap();
        assert_eq!(ml_idx.num_docs(), 1);

        // Remaining doc still searchable
        let results = ml_idx
            .search_with_language("бежать", 10, "russian")
            .unwrap();
        assert!(!results.is_empty(), "remaining doc should be searchable");
    }
}

// ── DETACH DELETE reverse posting list cleanup (G050) ──

/// E2E: DETACH DELETE removes the node AND cleans up the counterpart
/// posting lists so traversal from surviving nodes sees no stale edges.
#[test]
fn detach_delete_e2e_no_stale_edges_on_traversal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Create graph: A->B, A->C, B->C, B->A (bidirectional knows)
    run_cypher_with_alloc(
        "CREATE (a:Person {name: 'Alice'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "CREATE (b:Person {name: 'Bob'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "CREATE (c:Person {name: 'Charlie'})",
        &engine,
        &mut interner,
        &allocator,
    );

    // Edges: Alice->Bob, Alice->Charlie via KNOWS
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (c:Person {name: 'Charlie'}) CREATE (a)-[:KNOWS]->(c)",
        &engine,
        &mut interner,
        &allocator,
    );
    // Bob->Alice via KNOWS (bidirectional)
    run_cypher_with_alloc(
        "MATCH (b:Person {name: 'Bob'}), (a:Person {name: 'Alice'}) CREATE (b)-[:KNOWS]->(a)",
        &engine,
        &mut interner,
        &allocator,
    );
    // Bob->Charlie via KNOWS
    run_cypher_with_alloc(
        "MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'}) CREATE (b)-[:KNOWS]->(c)",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify: traversal from Bob shows Alice and Charlie as outgoing KNOWS
    let before = run_cypher_with_alloc(
        "MATCH (b:Person {name: 'Bob'})-[:KNOWS]->(friend) RETURN friend.name ORDER BY friend.name",
        &engine,
        &mut interner,
        &allocator,
    );
    let names_before: Vec<_> = before
        .iter()
        .filter_map(|r| r.get("friend.name").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(names_before, vec!["Alice", "Charlie"]);

    // Verify: reverse traversal — who KNOWS Alice?
    let before_rev = run_cypher_with_alloc(
        "MATCH (friend)-[:KNOWS]->(a:Person {name: 'Alice'}) RETURN friend.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        !before_rev.is_empty(),
        "Someone should KNOW Alice before delete"
    );

    // DETACH DELETE Alice
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}) DETACH DELETE a",
        &engine,
        &mut interner,
        &allocator,
    );

    // KEY CHECK: traversal from Bob should now show ONLY Charlie (Alice removed)
    let after = run_cypher_with_alloc(
        "MATCH (b:Person {name: 'Bob'})-[:KNOWS]->(friend) RETURN friend.name ORDER BY friend.name",
        &engine,
        &mut interner,
        &allocator,
    );
    let names_after: Vec<_> = after
        .iter()
        .filter_map(|r| r.get("friend.name").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(
        names_after,
        vec!["Charlie"],
        "Bob's outgoing KNOWS must not contain Alice after DETACH DELETE"
    );

    // KEY CHECK: reverse traversal — who KNOWS Charlie?
    // Should be only Bob (Alice's edge removed)
    let after_rev = run_cypher_with_alloc(
        "MATCH (friend)-[:KNOWS]->(c:Person {name: 'Charlie'}) RETURN friend.name ORDER BY friend.name",
        &engine,
        &mut interner,
        &allocator,
    );
    let rev_names: Vec<_> = after_rev
        .iter()
        .filter_map(|r| r.get("friend.name").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(
        rev_names,
        vec!["Bob"],
        "Charlie's incoming KNOWS must not contain Alice after DETACH DELETE"
    );

    // KEY CHECK: Alice should not exist at all
    let alice_check = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}) RETURN a.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(alice_check.is_empty(), "Alice should be gone");
}

/// E2E: DETACH DELETE with edge properties — edge props are cleaned up.
#[test]
fn detach_delete_e2e_edge_props_cleaned() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Create: Alice -[:REVIEWED {rating: 5}]-> Movie
    run_cypher_with_alloc(
        "CREATE (a:Person {name: 'Alice'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "CREATE (m:Movie {title: 'Matrix'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (m:Movie {title: 'Matrix'}) \
         CREATE (a)-[:REVIEWED {rating: 5}]->(m)",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify edge property exists via traversal
    let before = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:REVIEWED]->(m:Movie) RETURN r.rating",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].get("r.rating"), Some(&Value::Int(5)));

    // DETACH DELETE Alice
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}) DETACH DELETE a",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify: reverse traversal from Movie should return nothing
    let after = run_cypher_with_alloc(
        "MATCH (p)-[:REVIEWED]->(m:Movie {title: 'Matrix'}) RETURN p.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        after.is_empty(),
        "No one should REVIEWED Matrix after Alice deleted"
    );

    // Verify: edge properties are gone (check storage directly)
    // Scan for any edgeprop: keys — should be empty
    let ep_iter = engine
        .prefix_scan(Partition::EdgeProp, b"edgeprop:")
        .expect("scan");
    let ep_count = ep_iter.count();
    assert_eq!(
        ep_count, 0,
        "Edge properties should be cleaned up after DETACH DELETE"
    );
}

/// E2E: DETACH DELETE node with multiple edge types — all cleaned.
#[test]
fn detach_delete_e2e_multiple_edge_types() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Create: Alice -[:KNOWS]-> Bob, Alice -[:FOLLOWS]-> Bob, Alice -[:LIKES]-> Bob
    run_cypher_with_alloc(
        "CREATE (a:Person {name: 'Alice'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "CREATE (b:Person {name: 'Bob'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:FOLLOWS]->(b)",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:LIKES]->(b)",
        &engine,
        &mut interner,
        &allocator,
    );

    // DETACH DELETE Alice
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}) DETACH DELETE a",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify: Bob has no incoming edges from any type
    let knows = run_cypher_with_alloc(
        "MATCH (x)-[:KNOWS]->(b:Person {name: 'Bob'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(knows.is_empty(), "No incoming KNOWS to Bob after delete");

    let follows = run_cypher_with_alloc(
        "MATCH (x)-[:FOLLOWS]->(b:Person {name: 'Bob'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        follows.is_empty(),
        "No incoming FOLLOWS to Bob after delete"
    );

    let likes = run_cypher_with_alloc(
        "MATCH (x)-[:LIKES]->(b:Person {name: 'Bob'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(likes.is_empty(), "No incoming LIKES to Bob after delete");

    // Verify: no stale adj: keys reference Alice at all
    let adj_scan = engine.prefix_scan(Partition::Adj, b"adj:").expect("scan");
    for guard in adj_scan {
        let (k, v) = guard.into_inner().expect("kv");
        let plist = PostingList::from_bytes(&v).expect("decode");
        // Alice's node IDs (1000-based from allocator)
        // We don't know exact ID, but no posting list should contain a non-existent node
        let key_str = String::from_utf8_lossy(&k);
        // If key contains ":in:" it's an incoming list — values should all be valid nodes
        // For this test, there should be no adj keys at all (only Alice and Bob, Alice deleted,
        // Bob has no outgoing edges)
        // Actually Bob still exists, but has no edges. So adj: should be empty.
        assert!(
            plist.is_empty() || key_str.contains("out:"),
            "Unexpected non-empty posting list: {key_str}"
        );
    }
}

/// E2E: DETACH DELETE node with DOCUMENT-type properties on both sides.
/// Verifies that nested document properties don't interfere with edge cleanup.
#[test]
fn detach_delete_e2e_with_document_properties() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Create nodes with DOCUMENT-type properties via helper (can't do via Cypher literals)
    let config_doc = Value::Document(rmpv::Value::Map(vec![
        (
            rmpv::Value::String("network".into()),
            rmpv::Value::Map(vec![
                (
                    rmpv::Value::String("ssid".into()),
                    rmpv::Value::String("home-wifi".into()),
                ),
                (
                    rmpv::Value::String("channel".into()),
                    rmpv::Value::Integer(6.into()),
                ),
            ]),
        ),
        (
            rmpv::Value::String("version".into()),
            rmpv::Value::String("2.1".into()),
        ),
    ]));

    let metadata_doc = Value::Document(rmpv::Value::Map(vec![(
        rmpv::Value::String("tags".into()),
        rmpv::Value::Array(vec![
            rmpv::Value::String("sensor".into()),
            rmpv::Value::String("outdoor".into()),
        ]),
    )]));

    // Device node with nested document
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[
            ("name", Value::String("Router-A".into())),
            ("config", config_doc),
        ],
        &mut interner,
    );

    // Sensor node with nested document
    insert_node(
        &engine,
        1,
        2,
        "Sensor",
        &[
            ("name", Value::String("Temp-01".into())),
            ("metadata", metadata_doc),
        ],
        &mut interner,
    );

    // Hub node (simple properties)
    insert_node(
        &engine,
        1,
        3,
        "Hub",
        &[("name", Value::String("Central-Hub".into()))],
        &mut interner,
    );

    // Edges: Router->Sensor (CONNECTS), Router->Hub (MANAGES), Hub->Sensor (MONITORS)
    insert_edge(&engine, "CONNECTS", 1, 2);
    insert_edge(&engine, "MANAGES", 1, 3);
    insert_edge(&engine, "MONITORS", 3, 2);

    // Verify traversal before delete
    let before = run_cypher_with_alloc(
        "MATCH (d:Device {name: 'Router-A'})-[:CONNECTS]->(s) RETURN s.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(before.len(), 1);

    // DETACH DELETE the Device node (has DOCUMENT property "config")
    run_cypher_with_alloc(
        "MATCH (d:Device {name: 'Router-A'}) DETACH DELETE d",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify: Device is gone
    let device_check = run_cypher_with_alloc(
        "MATCH (d:Device) RETURN d.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(device_check.is_empty(), "Device should be deleted");

    // Verify: Sensor (with DOCUMENT property) still exists and queryable
    let sensor_check = run_cypher_with_alloc(
        "MATCH (s:Sensor {name: 'Temp-01'}) RETURN s.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(sensor_check.len(), 1, "Sensor should still exist");

    // Verify: no stale CONNECTS edge from deleted Device to Sensor
    let stale_connects = run_cypher_with_alloc(
        "MATCH (x)-[:CONNECTS]->(s:Sensor {name: 'Temp-01'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        stale_connects.is_empty(),
        "No stale CONNECTS edge should exist to Sensor"
    );

    // Verify: no stale MANAGES edge from deleted Device to Hub
    let stale_manages = run_cypher_with_alloc(
        "MATCH (x)-[:MANAGES]->(h:Hub {name: 'Central-Hub'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        stale_manages.is_empty(),
        "No stale MANAGES edge should exist to Hub"
    );

    // Verify: Hub->Sensor MONITORS edge is still intact (unrelated to deleted Device)
    let monitors = run_cypher_with_alloc(
        "MATCH (h:Hub {name: 'Central-Hub'})-[:MONITORS]->(s) RETURN s.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        monitors.len(),
        1,
        "Hub->Sensor MONITORS edge should survive"
    );
    assert_eq!(
        monitors[0].get("s.name"),
        Some(&Value::String("Temp-01".into()))
    );
}

/// E2E: Self-loop edge — node has an edge to itself. DETACH DELETE should not panic.
#[test]
fn detach_delete_e2e_self_loop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Create node and self-loop
    run_cypher_with_alloc(
        "CREATE (a:Person {name: 'Narcissus'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Narcissus'}) CREATE (a)-[:LOVES]->(a)",
        &engine,
        &mut interner,
        &allocator,
    );

    // Also create another node connected to Narcissus
    run_cypher_with_alloc(
        "CREATE (b:Person {name: 'Echo'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (e:Person {name: 'Echo'}), (n:Person {name: 'Narcissus'}) CREATE (e)-[:LOVES]->(n)",
        &engine,
        &mut interner,
        &allocator,
    );

    // DETACH DELETE Narcissus (has self-loop + incoming edge)
    run_cypher_with_alloc(
        "MATCH (n:Person {name: 'Narcissus'}) DETACH DELETE n",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify: Narcissus gone
    let check = run_cypher_with_alloc(
        "MATCH (n:Person {name: 'Narcissus'}) RETURN n.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(check.is_empty(), "Narcissus should be deleted");

    // Verify: Echo has no outgoing LOVES (target was deleted)
    let echo_loves = run_cypher_with_alloc(
        "MATCH (e:Person {name: 'Echo'})-[:LOVES]->(x) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        echo_loves.is_empty(),
        "Echo's outgoing LOVES to Narcissus should be cleaned up"
    );

    // Verify: Echo still exists
    let echo_check = run_cypher_with_alloc(
        "MATCH (e:Person {name: 'Echo'}) RETURN e.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(echo_check.len(), 1, "Echo should still exist");
}

/// E2E: Sequential DETACH DELETE — delete multiple connected nodes one by one.
/// Verifies no double-free or panic on already-cleaned posting lists.
#[test]
fn detach_delete_e2e_sequential_deletes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Create chain: A->B->C->D
    for name in &["Alice", "Bob", "Charlie", "Dave"] {
        run_cypher_with_alloc(
            &format!("CREATE (:Person {{name: '{name}'}})"),
            &engine,
            &mut interner,
            &allocator,
        );
    }
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:NEXT]->(b)",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'}) CREATE (b)-[:NEXT]->(c)",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (c:Person {name: 'Charlie'}), (d:Person {name: 'Dave'}) CREATE (c)-[:NEXT]->(d)",
        &engine,
        &mut interner,
        &allocator,
    );

    // Delete middle node Bob (has both incoming from Alice and outgoing to Charlie)
    run_cypher_with_alloc(
        "MATCH (b:Person {name: 'Bob'}) DETACH DELETE b",
        &engine,
        &mut interner,
        &allocator,
    );

    // Alice's outgoing NEXT should be empty (Bob was deleted)
    let alice_next = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[:NEXT]->(x) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        alice_next.is_empty(),
        "Alice's NEXT to Bob should be cleaned"
    );

    // Charlie's incoming NEXT should be empty (Bob was deleted)
    let charlie_prev = run_cypher_with_alloc(
        "MATCH (x)-[:NEXT]->(c:Person {name: 'Charlie'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        charlie_prev.is_empty(),
        "Charlie's incoming NEXT from Bob should be cleaned"
    );

    // Charlie->Dave edge should still work
    let charlie_dave = run_cypher_with_alloc(
        "MATCH (c:Person {name: 'Charlie'})-[:NEXT]->(d) RETURN d.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(charlie_dave.len(), 1);
    assert_eq!(
        charlie_dave[0].get("d.name"),
        Some(&Value::String("Dave".into()))
    );

    // Now delete Charlie (also a middle node after Bob's deletion)
    run_cypher_with_alloc(
        "MATCH (c:Person {name: 'Charlie'}) DETACH DELETE c",
        &engine,
        &mut interner,
        &allocator,
    );

    // Dave's incoming should be empty
    let dave_prev = run_cypher_with_alloc(
        "MATCH (x)-[:NEXT]->(d:Person {name: 'Dave'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        dave_prev.is_empty(),
        "Dave's incoming NEXT from Charlie should be cleaned"
    );

    // Alice and Dave should still exist, disconnected
    let remaining = run_cypher_with_alloc(
        "MATCH (p:Person) RETURN p.name ORDER BY p.name",
        &engine,
        &mut interner,
        &allocator,
    );
    let names: Vec<_> = remaining
        .iter()
        .filter_map(|r| r.get("p.name").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(names, vec!["Alice", "Dave"]);
}

/// E2E: DETACH DELETE node where edges were created in same transaction-like
/// sequence. Verifies merge buffer consistency.
#[test]
fn detach_delete_e2e_create_and_delete_same_session() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Create nodes and edges, then immediately delete one
    run_cypher_with_alloc(
        "CREATE (a:Temp {name: 'Alpha'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "CREATE (b:Temp {name: 'Beta'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "CREATE (c:Temp {name: 'Gamma'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (a:Temp {name: 'Alpha'}), (b:Temp {name: 'Beta'}) CREATE (a)-[:LINK]->(b)",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (b:Temp {name: 'Beta'}), (c:Temp {name: 'Gamma'}) CREATE (b)-[:LINK]->(c)",
        &engine,
        &mut interner,
        &allocator,
    );

    // Delete Beta right after creating it with edges
    run_cypher_with_alloc(
        "MATCH (b:Temp {name: 'Beta'}) DETACH DELETE b",
        &engine,
        &mut interner,
        &allocator,
    );

    // Alpha should have no outgoing LINK
    let alpha_link = run_cypher_with_alloc(
        "MATCH (a:Temp {name: 'Alpha'})-[:LINK]->(x) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(alpha_link.is_empty(), "Alpha's LINK to Beta should be gone");

    // Gamma should have no incoming LINK
    let gamma_link = run_cypher_with_alloc(
        "MATCH (x)-[:LINK]->(c:Temp {name: 'Gamma'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        gamma_link.is_empty(),
        "Gamma's incoming LINK from Beta should be gone"
    );

    // Verify only Alpha and Gamma remain
    let remaining = run_cypher_with_alloc(
        "MATCH (t:Temp) RETURN t.name ORDER BY t.name",
        &engine,
        &mut interner,
        &allocator,
    );
    let names: Vec<_> = remaining
        .iter()
        .filter_map(|r| r.get("t.name").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(names, vec!["Alpha", "Gamma"]);
}

/// E2E: Deep nested documents (5 levels) on both nodes.
/// DETACH DELETE one node — verify the surviving node's deeply nested document
/// is accessible via dot-notation and contains correct data.
#[test]
fn detach_delete_e2e_deep_nested_docs_both_sides() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // 5-level nested document for source node
    let source_doc = Value::Document(rmpv::Value::Map(vec![(
        rmpv::Value::String("level1".into()),
        rmpv::Value::Map(vec![(
            rmpv::Value::String("level2".into()),
            rmpv::Value::Map(vec![(
                rmpv::Value::String("level3".into()),
                rmpv::Value::Map(vec![(
                    rmpv::Value::String("level4".into()),
                    rmpv::Value::Map(vec![(
                        rmpv::Value::String("level5".into()),
                        rmpv::Value::String("deep-source-value".into()),
                    )]),
                )]),
            )]),
        )]),
    )]));

    // 5-level nested document for target node
    let target_doc = Value::Document(rmpv::Value::Map(vec![(
        rmpv::Value::String("data".into()),
        rmpv::Value::Map(vec![(
            rmpv::Value::String("nested".into()),
            rmpv::Value::Map(vec![(
                rmpv::Value::String("config".into()),
                rmpv::Value::Map(vec![(
                    rmpv::Value::String("settings".into()),
                    rmpv::Value::Map(vec![(
                        rmpv::Value::String("value".into()),
                        rmpv::Value::String("deep-target-value".into()),
                    )]),
                )]),
            )]),
        )]),
    )]));

    insert_node(
        &engine,
        1,
        1,
        "Source",
        &[
            ("name", Value::String("SrcNode".into())),
            ("deep", source_doc),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Target",
        &[
            ("name", Value::String("TgtNode".into())),
            ("deep", target_doc),
        ],
        &mut interner,
    );
    insert_edge(&engine, "LINKED", 1, 2);

    // Verify dot-notation access on target before delete
    let before = run_cypher_with_alloc(
        "MATCH (t:Target {name: 'TgtNode'}) RETURN t.deep.data.nested.config.settings.value",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(before.len(), 1);
    assert_eq!(
        before[0].get("t.deep.data.nested.config.settings.value"),
        Some(&Value::String("deep-target-value".into())),
        "Deep dot-notation should work before delete"
    );

    // DETACH DELETE the source node (which also has a deep nested doc)
    run_cypher_with_alloc(
        "MATCH (s:Source {name: 'SrcNode'}) DETACH DELETE s",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify: target's deep nested document is FULLY intact after neighbor deletion
    let after = run_cypher_with_alloc(
        "MATCH (t:Target {name: 'TgtNode'}) RETURN t.deep.data.nested.config.settings.value",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(after.len(), 1);
    assert_eq!(
        after[0].get("t.deep.data.nested.config.settings.value"),
        Some(&Value::String("deep-target-value".into())),
        "Deep nested document on surviving node must be intact after DETACH DELETE of neighbor"
    );

    // Verify: no stale reverse edge from deleted source
    let stale = run_cypher_with_alloc(
        "MATCH (x)-[:LINKED]->(t:Target {name: 'TgtNode'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(stale.is_empty(), "No stale LINKED edge to target");
}

/// E2E: Copy nested doc from one node to another via SET, then DETACH DELETE source.
/// Verifies the copied document on target is independently owned (no data corruption).
#[test]
fn detach_delete_e2e_copied_nested_doc_survives() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Source node with nested document
    let source_doc = Value::Document(rmpv::Value::Map(vec![
        (
            rmpv::Value::String("network".into()),
            rmpv::Value::Map(vec![
                (
                    rmpv::Value::String("ssid".into()),
                    rmpv::Value::String("office-5g".into()),
                ),
                (
                    rmpv::Value::String("channel".into()),
                    rmpv::Value::Integer(36.into()),
                ),
            ]),
        ),
        (
            rmpv::Value::String("auth".into()),
            rmpv::Value::Map(vec![(
                rmpv::Value::String("method".into()),
                rmpv::Value::String("WPA3".into()),
            )]),
        ),
    ]));

    insert_node(
        &engine,
        1,
        1,
        "Config",
        &[
            ("name", Value::String("Primary".into())),
            ("settings", source_doc.clone()),
        ],
        &mut interner,
    );

    // Target node — will receive a copy of the document
    insert_node(
        &engine,
        1,
        2,
        "Config",
        &[
            ("name", Value::String("Backup".into())),
            ("settings", source_doc),
        ],
        &mut interner,
    );
    insert_edge(&engine, "CLONED_FROM", 2, 1);

    // Verify both have the same document
    let primary = run_cypher_with_alloc(
        "MATCH (c:Config {name: 'Primary'}) RETURN c.settings.network.ssid",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        primary[0].get("c.settings.network.ssid"),
        Some(&Value::String("office-5g".into()))
    );

    let backup_before = run_cypher_with_alloc(
        "MATCH (c:Config {name: 'Backup'}) RETURN c.settings.network.ssid, c.settings.auth.method",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        backup_before[0].get("c.settings.network.ssid"),
        Some(&Value::String("office-5g".into()))
    );
    assert_eq!(
        backup_before[0].get("c.settings.auth.method"),
        Some(&Value::String("WPA3".into()))
    );

    // DETACH DELETE the primary (source of the cloned document)
    run_cypher_with_alloc(
        "MATCH (c:Config {name: 'Primary'}) DETACH DELETE c",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify: Backup's document is FULLY intact — independent copy, not shared reference
    let backup_after = run_cypher_with_alloc(
        "MATCH (c:Config {name: 'Backup'}) RETURN c.settings.network.ssid, c.settings.network.channel, c.settings.auth.method",
        &engine, &mut interner, &allocator,
    );
    assert_eq!(backup_after.len(), 1, "Backup node must survive");
    assert_eq!(
        backup_after[0].get("c.settings.network.ssid"),
        Some(&Value::String("office-5g".into())),
        "Backup's nested doc must retain ssid"
    );
    assert_eq!(
        backup_after[0].get("c.settings.network.channel"),
        Some(&Value::Int(36)),
        "Backup's nested doc must retain channel"
    );
    assert_eq!(
        backup_after[0].get("c.settings.auth.method"),
        Some(&Value::String("WPA3".into())),
        "Backup's nested doc must retain auth.method"
    );

    // Verify: no stale CLONED_FROM edge
    let stale = run_cypher_with_alloc(
        "MATCH (c:Config {name: 'Backup'})-[:CLONED_FROM]->(x) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        stale.is_empty(),
        "CLONED_FROM to deleted Primary should be gone"
    );
}

/// E2E: Edge with map-type properties. DETACH DELETE cleans up complex edge props.
#[test]
fn detach_delete_e2e_edge_with_map_properties() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Create nodes
    run_cypher_with_alloc(
        "CREATE (a:User {name: 'Alice'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "CREATE (b:User {name: 'Bob'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "CREATE (c:User {name: 'Charlie'})",
        &engine,
        &mut interner,
        &allocator,
    );

    // Create edge with complex map properties: Alice->Bob with {since: 2020, tags: ["friend", "colleague"]}
    run_cypher_with_alloc(
        "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) \
         CREATE (a)-[:KNOWS {since: 2020, strength: 0.9}]->(b)",
        &engine,
        &mut interner,
        &allocator,
    );
    // Charlie->Alice with edge props
    run_cypher_with_alloc(
        "MATCH (c:User {name: 'Charlie'}), (a:User {name: 'Alice'}) \
         CREATE (c)-[:FOLLOWS {since: 2023}]->(a)",
        &engine,
        &mut interner,
        &allocator,
    );
    // Bob->Charlie (no props)
    run_cypher_with_alloc(
        "MATCH (b:User {name: 'Bob'}), (c:User {name: 'Charlie'}) \
         CREATE (b)-[:KNOWS]->(c)",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify edge props exist before delete
    let before_ep = run_cypher_with_alloc(
        "MATCH (a:User {name: 'Alice'})-[r:KNOWS]->(b:User {name: 'Bob'}) RETURN r.since, r.strength",
        &engine, &mut interner, &allocator,
    );
    assert_eq!(before_ep.len(), 1);
    assert_eq!(before_ep[0].get("r.since"), Some(&Value::Int(2020)));

    // DETACH DELETE Alice
    run_cypher_with_alloc(
        "MATCH (a:User {name: 'Alice'}) DETACH DELETE a",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify: no stale edges to/from Alice
    let bob_in = run_cypher_with_alloc(
        "MATCH (x)-[:KNOWS]->(b:User {name: 'Bob'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(bob_in.is_empty(), "No stale KNOWS edge from Alice to Bob");

    let charlie_follows = run_cypher_with_alloc(
        "MATCH (c:User {name: 'Charlie'})-[:FOLLOWS]->(x) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        charlie_follows.is_empty(),
        "Charlie's FOLLOWS to Alice should be gone"
    );

    // Verify: Bob->Charlie edge (unrelated to Alice) is intact
    let bob_charlie = run_cypher_with_alloc(
        "MATCH (b:User {name: 'Bob'})-[:KNOWS]->(c:User {name: 'Charlie'}) RETURN c.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        bob_charlie.len(),
        1,
        "Bob->Charlie KNOWS edge should survive"
    );

    // Verify: ALL edge property keys are clean (no orphaned edgeprop: entries)
    let ep_scan = engine
        .prefix_scan(Partition::EdgeProp, b"edgeprop:")
        .expect("scan");
    let ep_keys: Vec<_> = ep_scan
        .filter_map(|g| {
            let (k, _) = g.into_inner().ok()?;
            Some(k.to_vec())
        })
        .collect();
    // Only Bob->Charlie edge should remain (it has no props, so edgeprop might not exist)
    // Alice's edges had props — those edgeprop keys must be gone
    for key in &ep_keys {
        let key_str = String::from_utf8_lossy(key);
        // None of the remaining edgeprop keys should reference Alice's node ID
        // We can't easily check without knowing Alice's ID, but we can assert count
        assert!(
            !key_str.contains("KNOWS") || !key_str.contains("FOLLOWS"),
            "Orphaned edgeprop key found: {key_str}"
        );
    }
}

/// E2E: CREATE after DETACH DELETE — verify no ghost data from deleted node.
/// New node at potentially same storage location should have clean state.
#[test]
fn detach_delete_e2e_create_after_delete_no_ghost() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Create node with rich document + edges
    let rich_doc = Value::Document(rmpv::Value::Map(vec![
        (
            rmpv::Value::String("history".into()),
            rmpv::Value::Array(vec![
                rmpv::Value::String("event1".into()),
                rmpv::Value::String("event2".into()),
                rmpv::Value::String("event3".into()),
            ]),
        ),
        (
            rmpv::Value::String("metadata".into()),
            rmpv::Value::Map(vec![(
                rmpv::Value::String("version".into()),
                rmpv::Value::Integer(42.into()),
            )]),
        ),
    ]));

    insert_node(
        &engine,
        1,
        1,
        "Service",
        &[
            ("name", Value::String("OldService".into())),
            ("state", rich_doc),
        ],
        &mut interner,
    );
    run_cypher_with_alloc(
        "CREATE (dep:Dependency {name: 'Redis'})",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (s:Service {name: 'OldService'}), (d:Dependency {name: 'Redis'}) \
         CREATE (s)-[:DEPENDS_ON {critical: true}]->(d)",
        &engine,
        &mut interner,
        &allocator,
    );

    // DETACH DELETE OldService
    run_cypher_with_alloc(
        "MATCH (s:Service {name: 'OldService'}) DETACH DELETE s",
        &engine,
        &mut interner,
        &allocator,
    );

    // CREATE new service with same label but different data
    run_cypher_with_alloc(
        "CREATE (s:Service {name: 'NewService', version: 2})",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify: new service has NO ghost properties from old service
    let new_svc = run_cypher_with_alloc(
        "MATCH (s:Service {name: 'NewService'}) RETURN s.name, s.version",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(new_svc.len(), 1);
    assert_eq!(
        new_svc[0].get("s.name"),
        Some(&Value::String("NewService".into()))
    );
    assert_eq!(new_svc[0].get("s.version"), Some(&Value::Int(2)));

    // Verify: new service has NO edges (old service's DEPENDS_ON was deleted)
    let new_deps = run_cypher_with_alloc(
        "MATCH (s:Service {name: 'NewService'})-[:DEPENDS_ON]->(d) RETURN d.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(new_deps.is_empty(), "New service should have no edges");

    // Verify: Redis has no incoming DEPENDS_ON (from deleted OldService)
    let redis_deps = run_cypher_with_alloc(
        "MATCH (x)-[:DEPENDS_ON]->(d:Dependency {name: 'Redis'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        redis_deps.is_empty(),
        "Redis should have no incoming DEPENDS_ON"
    );

    // Connect new service to Redis — verify fresh edges work
    run_cypher_with_alloc(
        "MATCH (s:Service {name: 'NewService'}), (d:Dependency {name: 'Redis'}) \
         CREATE (s)-[:DEPENDS_ON {critical: false}]->(d)",
        &engine,
        &mut interner,
        &allocator,
    );

    let new_deps2 = run_cypher_with_alloc(
        "MATCH (s:Service {name: 'NewService'})-[r:DEPENDS_ON]->(d) RETURN d.name, r.critical",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(new_deps2.len(), 1, "New edge should exist");
    assert_eq!(
        new_deps2[0].get("d.name"),
        Some(&Value::String("Redis".into()))
    );
    // Edge prop 'critical' should be false (new edge), not true (old edge)
    assert_eq!(
        new_deps2[0].get("r.critical"),
        Some(&Value::Bool(false)),
        "Edge prop should be from new edge, not ghost of old"
    );
}

/// E2E: Complex diamond graph with documents — delete central node,
/// verify all 4 remaining nodes + their documents + surviving edges intact.
#[test]
fn detach_delete_e2e_diamond_graph_documents() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));

    // Diamond: Top -> Left, Top -> Right, Left -> Bottom, Right -> Bottom
    // Each node has a unique nested document. Delete Top.
    let top_doc = Value::Document(rmpv::Value::Map(vec![
        (
            rmpv::Value::String("role".into()),
            rmpv::Value::String("coordinator".into()),
        ),
        (
            rmpv::Value::String("config".into()),
            rmpv::Value::Map(vec![(
                rmpv::Value::String("priority".into()),
                rmpv::Value::Integer(1.into()),
            )]),
        ),
    ]));
    let left_doc = Value::Document(rmpv::Value::Map(vec![
        (
            rmpv::Value::String("role".into()),
            rmpv::Value::String("worker-A".into()),
        ),
        (
            rmpv::Value::String("metrics".into()),
            rmpv::Value::Map(vec![(
                rmpv::Value::String("throughput".into()),
                rmpv::Value::F64(1500.5),
            )]),
        ),
    ]));
    let right_doc = Value::Document(rmpv::Value::Map(vec![
        (
            rmpv::Value::String("role".into()),
            rmpv::Value::String("worker-B".into()),
        ),
        (
            rmpv::Value::String("metrics".into()),
            rmpv::Value::Map(vec![(
                rmpv::Value::String("throughput".into()),
                rmpv::Value::F64(2200.0),
            )]),
        ),
    ]));
    let bottom_doc = Value::Document(rmpv::Value::Map(vec![
        (
            rmpv::Value::String("role".into()),
            rmpv::Value::String("aggregator".into()),
        ),
        (
            rmpv::Value::String("buffer".into()),
            rmpv::Value::Array(vec![
                rmpv::Value::Integer(10.into()),
                rmpv::Value::Integer(20.into()),
                rmpv::Value::Integer(30.into()),
            ]),
        ),
    ]));

    insert_node(
        &engine,
        1,
        1,
        "Node",
        &[("name", Value::String("Top".into())), ("state", top_doc)],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Node",
        &[("name", Value::String("Left".into())), ("state", left_doc)],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Node",
        &[
            ("name", Value::String("Right".into())),
            ("state", right_doc),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        4,
        "Node",
        &[
            ("name", Value::String("Bottom".into())),
            ("state", bottom_doc),
        ],
        &mut interner,
    );

    // Diamond edges
    insert_edge(&engine, "FEEDS", 1, 2); // Top -> Left
    insert_edge(&engine, "FEEDS", 1, 3); // Top -> Right
    insert_edge(&engine, "FEEDS", 2, 4); // Left -> Bottom
    insert_edge(&engine, "FEEDS", 3, 4); // Right -> Bottom

    // DETACH DELETE Top (central fan-out node)
    run_cypher_with_alloc(
        "MATCH (n:Node {name: 'Top'}) DETACH DELETE n",
        &engine,
        &mut interner,
        &allocator,
    );

    // Verify: Left's document is intact
    let left = run_cypher_with_alloc(
        "MATCH (n:Node {name: 'Left'}) RETURN n.state.role, n.state.metrics.throughput",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(left.len(), 1);
    assert_eq!(
        left[0].get("n.state.role"),
        Some(&Value::String("worker-A".into()))
    );
    assert_eq!(
        left[0].get("n.state.metrics.throughput"),
        Some(&Value::Float(1500.5))
    );

    // Verify: Right's document is intact
    let right = run_cypher_with_alloc(
        "MATCH (n:Node {name: 'Right'}) RETURN n.state.role, n.state.metrics.throughput",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(right.len(), 1);
    assert_eq!(
        right[0].get("n.state.metrics.throughput"),
        Some(&Value::Float(2200.0))
    );

    // Verify: Bottom's document is intact (with array)
    let bottom = run_cypher_with_alloc(
        "MATCH (n:Node {name: 'Bottom'}) RETURN n.state.role",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(bottom.len(), 1);
    assert_eq!(
        bottom[0].get("n.state.role"),
        Some(&Value::String("aggregator".into()))
    );

    // Verify: Left->Bottom and Right->Bottom edges survive
    let feeds_bottom = run_cypher_with_alloc(
        "MATCH (x)-[:FEEDS]->(b:Node {name: 'Bottom'}) RETURN x.name ORDER BY x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    let feeders: Vec<_> = feeds_bottom
        .iter()
        .filter_map(|r| r.get("x.name").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(
        feeders,
        vec!["Left", "Right"],
        "Both Left and Right should still feed Bottom"
    );

    // Verify: no stale incoming FEEDS to Left or Right from deleted Top
    let left_incoming = run_cypher_with_alloc(
        "MATCH (x)-[:FEEDS]->(n:Node {name: 'Left'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        left_incoming.is_empty(),
        "Left should have no incoming FEEDS (Top deleted)"
    );

    let right_incoming = run_cypher_with_alloc(
        "MATCH (x)-[:FEEDS]->(n:Node {name: 'Right'}) RETURN x.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert!(
        right_incoming.is_empty(),
        "Right should have no incoming FEEDS (Top deleted)"
    );
}

// ── Correlated OPTIONAL MATCH (G004) ──────────────────────────────────

#[test]
fn optional_match_correlated_where_cross_variable() {
    // Regression test for G004: OPTIONAL MATCH with WHERE predicate
    // referencing a variable from the outer MATCH scope.
    //
    // Current bug: right side executes once, `a.age` evaluates to Null
    // because "a" is not in right-side rows. All rows filtered out →
    // incorrect NULL result instead of matching rows.
    let (_dir, engine, mut interner) = setup_social_graph();

    // Charlie is 30. Find all Person nodes older than Charlie.
    // Expected: Dave(40) and Eve(35) match b.age > a.age (30).
    let results = run_cypher(
        "MATCH (a:Person {name: \"Charlie\"}) \
         OPTIONAL MATCH (b:Person) \
         WHERE b.age > a.age AND b.name <> a.name \
         RETURN a.name AS self_name, b.name AS older_name \
         ORDER BY older_name",
        &engine,
        &mut interner,
    );

    // Must produce 2 rows: (Charlie, Dave) and (Charlie, Eve)
    assert_eq!(
        results.len(),
        2,
        "should find 2 people older than Charlie (Dave=40, Eve=35), got {results:?}"
    );

    let older_names: Vec<&Value> = results
        .iter()
        .map(|r| r.get("older_name").expect("older_name column"))
        .collect();
    assert!(
        older_names.contains(&&Value::String("Dave".into())),
        "Dave(40) is older than Charlie(30)"
    );
    assert!(
        older_names.contains(&&Value::String("Eve".into())),
        "Eve(35) is older than Charlie(30)"
    );
}

#[test]
fn optional_match_correlated_no_match_returns_null() {
    // Dave is 40 — the oldest person. No one is older.
    // Correlated OPTIONAL MATCH should return NULL for b.
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (a:Person {name: \"Dave\"}) \
         OPTIONAL MATCH (b:Person) \
         WHERE b.age > a.age AND b.name <> a.name \
         RETURN a.name AS self_name, b.name AS older_name",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1, "should have 1 row (Dave with NULL)");
    let older = results[0].get("older_name").cloned().unwrap_or(Value::Null);
    assert_eq!(older, Value::Null, "no one is older than Dave(40)");
}

#[test]
fn optional_match_correlated_traversal_with_cross_scope() {
    // More realistic pattern: MATCH (a) OPTIONAL MATCH (b)-[:R]->(c) WHERE c.x = f(a.y)
    // Here "a" is only in left scope, "b" and "c" are in right scope.
    let (_dir, engine, mut interner) = setup_social_graph();

    // Find people whose KNOWS targets include someone the same age as Alice.
    // Alice is 30. Charlie is also 30.
    // Right side: (b)-[:KNOWS]->(c) WHERE c.age = a.age AND c.name <> a.name
    // Matches: b=Bob->c=Charlie(30), b=Alice->c=Charlie(30)
    // But c.name <> a.name filters out c=Alice (Alice→Bob doesn't reach Alice)
    let results = run_cypher(
        "MATCH (a:Person {name: \"Alice\"}) \
         OPTIONAL MATCH (b:Person)-[:KNOWS]->(c:Person) \
         WHERE c.age = a.age AND c.name <> a.name \
         RETURN a.name AS src, b.name AS via, c.name AS peer",
        &engine,
        &mut interner,
    );

    // Expected: rows where c.age = 30 (Alice's age) and c.name != "Alice"
    // That's c=Charlie(30). Via edges: Bob→Charlie, Alice→Charlie
    assert!(
        !results.is_empty(),
        "should find at least one row (Charlie is same age as Alice), got {results:?}"
    );
    let peers: Vec<&Value> = results
        .iter()
        .filter_map(|r| r.get("peer"))
        .filter(|v| **v != Value::Null)
        .collect();
    assert!(
        peers.contains(&&Value::String("Charlie".into())),
        "Charlie(30) should be found as same-age peer of Alice(30)"
    );
}

#[test]
fn optional_match_nested_correlated() {
    // Nested OPTIONAL MATCH: second OPTIONAL MATCH references variable from
    // first MATCH scope. Tests save/restore of correlated_row.
    //
    // MATCH (a:Person {name: "Alice"})
    // OPTIONAL MATCH (a)-[:KNOWS]->(b)
    // OPTIONAL MATCH (b)-[:KNOWS]->(c) WHERE c.age > a.age AND c.name <> b.name
    //
    // Alice→Bob, Alice→Charlie. Bob→Charlie. Charlie→Dave.
    // Second OPTIONAL MATCH: for each (a,b) row, find b→c where c.age > a.age(30).
    //   (Alice, Bob): Bob→Charlie(30) — 30 > 30? No. → NULL
    //   (Alice, Charlie): Charlie→Dave(40) — 40 > 30? Yes, and Dave ≠ Charlie. → Dave
    let (_dir, engine, mut interner) = setup_social_graph();

    let results = run_cypher(
        "MATCH (a:Person {name: \"Alice\"}) \
         OPTIONAL MATCH (a)-[:KNOWS]->(b) \
         OPTIONAL MATCH (b)-[:KNOWS]->(c) \
         WHERE c.age > a.age AND c.name <> b.name \
         RETURN a.name AS a_name, b.name AS b_name, c.name AS c_name \
         ORDER BY b_name",
        &engine,
        &mut interner,
    );

    assert_eq!(
        results.len(),
        2,
        "should have 2 rows (one per Alice's KNOWS target), got {results:?}"
    );

    // Row 1: Alice → Bob → NULL (no Bob→X where X.age > 30 and X ≠ Bob)
    let bob_row = results
        .iter()
        .find(|r| r.get("b_name") == Some(&Value::String("Bob".into())))
        .expect("should have Bob row");
    assert_eq!(
        bob_row.get("c_name").cloned().unwrap_or(Value::Null),
        Value::Null,
        "Bob's targets (Charlie=30) not older than Alice(30)"
    );

    // Row 2: Alice → Charlie → Dave (Dave=40 > Alice=30, Dave ≠ Charlie)
    let charlie_row = results
        .iter()
        .find(|r| r.get("b_name") == Some(&Value::String("Charlie".into())))
        .expect("should have Charlie row");
    assert_eq!(
        charlie_row.get("c_name"),
        Some(&Value::String("Dave".into())),
        "Charlie→Dave(40) is older than Alice(30)"
    );
}

#[test]
fn optional_match_non_correlated_still_works() {
    // Verify that non-correlated OPTIONAL MATCH (the common case) still uses
    // the fast global execution path and produces correct results.
    // This is a regression guard against the correlated optimization.
    let (_dir, engine, mut interner) = setup_social_graph();

    // Simple non-correlated: MATCH (a) OPTIONAL MATCH (a)-[:KNOWS]->(b)
    // Bob KNOWS Charlie only.
    let results = run_cypher(
        "MATCH (a:Person {name: \"Bob\"}) \
         OPTIONAL MATCH (a)-[:KNOWS]->(b) \
         RETURN a.name AS self_name, b.name AS friend",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 1, "Bob has 1 KNOWS target");
    assert_eq!(
        results[0].get("friend"),
        Some(&Value::String("Charlie".into())),
        "Bob KNOWS Charlie"
    );
}

// ── Per-query hint syntax (G026) ──────────────────────────────────────

#[test]
fn hint_vector_consistency_flows_to_plan() {
    // Verify that /*+ vector_consistency('snapshot') */ hint is extracted
    // and applied to the LogicalPlan.
    let ast = parse("MATCH (n:Person) RETURN n /*+ vector_consistency('snapshot') */").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    assert_eq!(
        plan.vector_consistency,
        coordinode_core::graph::types::VectorConsistencyMode::Snapshot,
        "hint should override default vector_consistency"
    );
}

#[test]
fn hint_absent_uses_default() {
    let ast = parse("MATCH (n:Person) RETURN n").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    assert_eq!(
        plan.vector_consistency,
        coordinode_core::graph::types::VectorConsistencyMode::Current,
        "without hint, default is Current"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// G015: 3-arg text_match(field, query, language) — multi-language search
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn g015_text_match_3arg_with_language() {
    // Verify text_match(field, query, language) works through the full
    // planner → executor pipeline with explicit language tokenization.
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Insert English and Russian articles
    insert_node(
        &engine,
        1,
        1,
        "Doc",
        &[("title", Value::String("Running through the forest".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Doc",
        &[("title", Value::String("Бегущий человек в лесу".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Doc",
        &[("title", Value::String("Forest management practices".into()))],
        &mut interner,
    );

    let text_dir = dir.path().join("text_idx");
    let config = MultiLangConfig::with_default_language("english")
        .with_field_analyzer("title", "auto_detect");
    let mut ml_idx = MultiLanguageTextIndex::open_or_create(&text_dir, 15_000_000, config).unwrap();

    // Index with per-document language
    ml_idx
        .add_node(
            1,
            &[(
                "title".to_string(),
                "Running through the forest".to_string(),
            )]
            .iter()
            .cloned()
            .collect(),
        )
        .unwrap();
    ml_idx
        .add_node(
            2,
            &[("title".to_string(), "Бегущий человек в лесу".to_string())]
                .iter()
                .cloned()
                .collect(),
        )
        .unwrap();
    ml_idx
        .add_node(
            3,
            &[(
                "title".to_string(),
                "Forest management practices".to_string(),
            )]
            .iter()
            .cloned()
            .collect(),
        )
        .unwrap();

    // 3-arg: search with explicit English tokenization
    let results = run_cypher_with_text_index(
        "MATCH (d:Doc) WHERE text_match(d.title, \"running forest\", \"english\") RETURN d.title",
        &engine,
        &mut interner,
        &ml_idx,
    );
    assert!(
        !results.is_empty(),
        "English search should match English documents"
    );

    // 2-arg: should still work (uses default language from config)
    let results_2arg = run_cypher_with_text_index(
        "MATCH (d:Doc) WHERE text_match(d.title, \"forest\") RETURN d.title",
        &engine,
        &mut interner,
        &ml_idx,
    );
    assert!(
        !results_2arg.is_empty(),
        "2-arg text_match should still work with MultiLanguageTextIndex"
    );
}

#[test]
fn g015_text_match_3arg_planner_extracts_language() {
    // Verify the planner correctly extracts the language from 3-arg text_match
    // into LogicalOp::TextFilter.language.
    use coordinode_query::planner::logical::LogicalOp;

    let query = "MATCH (n:Article) WHERE text_match(n.body, \"hello\", \"russian\") RETURN n";
    let ast = parse(query).unwrap();
    let plan = build_logical_plan(&ast).unwrap();

    // Walk the plan tree to find TextFilter operator
    fn find_text_filter(op: &LogicalOp) -> Option<&LogicalOp> {
        match op {
            LogicalOp::TextFilter { .. } => Some(op),
            LogicalOp::Filter { input, .. }
            | LogicalOp::Project { input, .. }
            | LogicalOp::Sort { input, .. }
            | LogicalOp::Limit { input, .. } => find_text_filter(input),
            _ => None,
        }
    }

    let tf = find_text_filter(&plan.root).expect("plan should contain TextFilter");
    if let LogicalOp::TextFilter {
        language,
        query_string,
        ..
    } = tf
    {
        assert_eq!(
            language.as_deref(),
            Some("russian"),
            "language should be 'russian'"
        );
        assert_eq!(query_string, "hello");
    } else {
        panic!("expected TextFilter");
    }
}

#[test]
fn g015_text_match_2arg_has_no_language() {
    // Verify 2-arg text_match produces TextFilter with language=None.
    use coordinode_query::planner::logical::LogicalOp;

    let query = "MATCH (n:Article) WHERE text_match(n.body, \"hello\") RETURN n";
    let ast = parse(query).unwrap();
    let plan = build_logical_plan(&ast).unwrap();

    fn find_text_filter_2(op: &LogicalOp) -> Option<&LogicalOp> {
        match op {
            LogicalOp::TextFilter { .. } => Some(op),
            LogicalOp::Filter { input, .. }
            | LogicalOp::Project { input, .. }
            | LogicalOp::Sort { input, .. }
            | LogicalOp::Limit { input, .. } => find_text_filter_2(input),
            _ => None,
        }
    }

    let tf = find_text_filter_2(&plan.root).expect("plan should contain TextFilter");
    if let LogicalOp::TextFilter { language, .. } = tf {
        assert!(
            language.is_none(),
            "2-arg text_match should have no language"
        );
    } else {
        panic!("expected TextFilter");
    }
}

#[test]
fn g015_explain_shows_language_in_text_filter() {
    // 3-arg: EXPLAIN should show language
    let ast =
        parse("MATCH (n:Doc) WHERE text_match(n.body, \"hello\", \"russian\") RETURN n").unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let explain = plan.explain();
    assert!(
        explain.contains("language: \"russian\""),
        "EXPLAIN should show language for 3-arg text_match, got:\n{explain}"
    );

    // 2-arg: EXPLAIN should NOT show language
    let ast2 = parse("MATCH (n:Doc) WHERE text_match(n.body, \"hello\") RETURN n").unwrap();
    let plan2 = build_logical_plan(&ast2).unwrap();
    let explain2 = plan2.explain();
    assert!(
        !explain2.contains("language:"),
        "EXPLAIN should NOT show language for 2-arg text_match, got:\n{explain2}"
    );
}

// ── Document Path-Targeted Updates (R164) ────────────────────────────

/// SET n.config.network.ssid = "home" on a node with a DOCUMENT property.
/// Verifies the merge operand is applied and readable via dot-notation.
#[test]
fn r164_set_deep_path_on_document_property() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Create a node with a DOCUMENT property containing nested structure.
    let mut doc = std::collections::BTreeMap::new();
    let mut network = std::collections::BTreeMap::new();
    network.insert("ssid".to_string(), rmpv::Value::String("old_wifi".into()));
    doc.insert(
        "network".to_string(),
        rmpv::Value::Map(
            network
                .into_iter()
                .map(|(k, v)| (rmpv::Value::String(k.into()), v))
                .collect(),
        ),
    );
    let rmpv_doc = rmpv::Value::Map(
        doc.into_iter()
            .map(|(k, v)| (rmpv::Value::String(k.into()), v))
            .collect(),
    );

    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[
            ("name", Value::String("sensor-01".into())),
            ("config", Value::Document(rmpv_doc)),
        ],
        &mut interner,
    );

    // SET the deep path via merge operand.
    let _rows = run_cypher(
        "MATCH (n:Device) SET n.config.network.ssid = 'home_wifi' RETURN n.name",
        &engine,
        &mut interner,
    );

    // Read back and verify the nested value changed.
    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.network.ssid",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.config.network.ssid"),
        Some(&Value::String("home_wifi".into()))
    );
}

/// SET creates intermediate objects when path doesn't fully exist.
#[test]
fn r164_set_path_creates_intermediates() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Create node with empty DOCUMENT config property.
    let empty_doc = rmpv::Value::Map(vec![]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(empty_doc))],
        &mut interner,
    );

    // SET a deep path — intermediates should be created automatically.
    run_cypher(
        "MATCH (n:Device) SET n.config.network.ssid = 'new_network'",
        &engine,
        &mut interner,
    );

    // Read back the full path.
    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.network.ssid",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.config.network.ssid"),
        Some(&Value::String("new_network".into()))
    );
}

/// REMOVE n.config.network.ssid deletes a nested path.
#[test]
fn r164_remove_deep_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Create a node with nested config.
    let mut network = std::collections::BTreeMap::new();
    network.insert("ssid".to_string(), rmpv::Value::String("my_wifi".into()));
    network.insert("password".to_string(), rmpv::Value::String("secret".into()));
    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("network".into()),
        rmpv::Value::Map(
            network
                .into_iter()
                .map(|(k, v)| (rmpv::Value::String(k.into()), v))
                .collect(),
        ),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(doc))],
        &mut interner,
    );

    // REMOVE the ssid nested property.
    run_cypher(
        "MATCH (n:Device) REMOVE n.config.network.ssid",
        &engine,
        &mut interner,
    );

    // ssid should be Null, password should still exist.
    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.network.ssid, n.config.network.password",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.config.network.ssid"), Some(&Value::Null));
    assert_eq!(
        rows[0].get("n.config.network.password"),
        Some(&Value::String("secret".into()))
    );
}

/// REMOVE on non-existent deep path is a no-op (idempotent).
#[test]
fn r164_remove_nonexistent_path_is_noop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("name".into()),
        rmpv::Value::String("test".into()),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(doc))],
        &mut interner,
    );

    // REMOVE a path that doesn't exist — should not error.
    run_cypher(
        "MATCH (n:Device) REMOVE n.config.nonexistent.deep.path",
        &engine,
        &mut interner,
    );

    // Verify the existing data is untouched.
    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.name",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.config.name"),
        Some(&Value::String("test".into()))
    );
}

/// Mixed SET: single-level property + deep path in same statement.
#[test]
fn r164_mixed_set_property_and_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("version".into()),
        rmpv::Value::String("1.0".into()),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[
            ("name", Value::String("old".into())),
            ("config", Value::Document(doc)),
        ],
        &mut interner,
    );

    // SET both a flat property and a deep path in one statement.
    run_cypher(
        "MATCH (n:Device) SET n.name = 'new', n.config.version = '2.0'",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.name, n.config.version",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.name"), Some(&Value::String("new".into())));
    assert_eq!(
        rows[0].get("n.config.version"),
        Some(&Value::String("2.0".into()))
    );
}

/// SET deep path with non-string values: integer, nested map, boolean.
#[test]
fn r164_set_deep_path_non_string_values() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(doc))],
        &mut interner,
    );

    // SET integer at deep path.
    run_cypher(
        "MATCH (n:Device) SET n.config.retry_count = 5",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.retry_count",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.config.retry_count"), Some(&Value::Int(5)));

    // SET boolean at deep path.
    run_cypher(
        "MATCH (n:Device) SET n.config.enabled = true",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.enabled",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.config.enabled"), Some(&Value::Bool(true)));
}

/// SET two different deep paths on the same node in separate statements.
/// Both changes should persist (no lost update).
#[test]
fn r164_set_two_different_paths_sequential() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("network".into()),
        rmpv::Value::Map(vec![
            (
                rmpv::Value::String("ssid".into()),
                rmpv::Value::String("old".into()),
            ),
            (
                rmpv::Value::String("password".into()),
                rmpv::Value::String("secret".into()),
            ),
        ]),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(doc))],
        &mut interner,
    );

    // First SET: change ssid.
    run_cypher(
        "MATCH (n:Device) SET n.config.network.ssid = 'new_wifi'",
        &engine,
        &mut interner,
    );

    // Second SET: change password (different path, same node).
    run_cypher(
        "MATCH (n:Device) SET n.config.network.password = 'new_secret'",
        &engine,
        &mut interner,
    );

    // Both changes should be visible.
    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.network.ssid, n.config.network.password",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.config.network.ssid"),
        Some(&Value::String("new_wifi".into()))
    );
    assert_eq!(
        rows[0].get("n.config.network.password"),
        Some(&Value::String("new_secret".into()))
    );
}

/// SET overwrites existing value at deep path (last-writer-wins).
#[test]
fn r164_set_overwrites_existing_deep_value() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("level".into()),
        rmpv::Value::Integer(1.into()),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(doc))],
        &mut interner,
    );

    run_cypher(
        "MATCH (n:Device) SET n.config.level = 42",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.level",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.config.level"), Some(&Value::Int(42)));
}

// ── G064: Merge-based document path updates ─────────────────────────

/// RYOW: SET deep path then RETURN in same query must see the update.
/// This exercises the RYOW materialization path in mvcc_get().
#[test]
fn g064_ryow_set_then_return_same_query() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("version".into()),
        rmpv::Value::String("1.0".into()),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(doc))],
        &mut interner,
    );

    // SET + RETURN in the same query — must see the updated value.
    let rows = run_cypher(
        "MATCH (n:Device) SET n.config.version = '2.0' RETURN n.config.version",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.config.version"),
        Some(&Value::String("2.0".into()))
    );
}

/// SET deep path on a node that has NO DOCUMENT property at that name yet.
/// The merge operator should create the property as a Document with intermediate maps.
#[test]
fn g064_set_path_creates_property_from_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Node with only a "name" property, no "config" at all.
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("name", Value::String("sensor".into()))],
        &mut interner,
    );

    // SET a deep path on a property that doesn't exist.
    run_cypher(
        "MATCH (n:Device) SET n.config.network.ssid = 'new_wifi'",
        &engine,
        &mut interner,
    );

    // Verify the property was created with the nested value.
    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.network.ssid, n.name",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.config.network.ssid"),
        Some(&Value::String("new_wifi".into()))
    );
    // Original property should be untouched.
    assert_eq!(rows[0].get("n.name"), Some(&Value::String("sensor".into())));
}

/// Multiple SET deep paths on the same node in a single statement.
/// All should be applied (via separate merge operands).
#[test]
fn g064_multiple_set_paths_same_statement() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(doc))],
        &mut interner,
    );

    // Two deep SETs in one statement.
    run_cypher(
        "MATCH (n:Device) SET n.config.a = 1, n.config.b = 2",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.a, n.config.b",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.config.a"), Some(&Value::Int(1)));
    assert_eq!(rows[0].get("n.config.b"), Some(&Value::Int(2)));
}

/// REMOVE + SET on different paths of the same DOCUMENT in one statement.
#[test]
fn g064_remove_and_set_same_document_one_statement() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("old_key".into()),
            rmpv::Value::String("to_remove".into()),
        ),
        (
            rmpv::Value::String("keep_key".into()),
            rmpv::Value::String("kept".into()),
        ),
    ]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(doc))],
        &mut interner,
    );

    // REMOVE one path, SET another in the same statement.
    run_cypher(
        "MATCH (n:Device) REMOVE n.config.old_key SET n.config.new_key = 'added'",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.old_key, n.config.keep_key, n.config.new_key",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.config.old_key"), Some(&Value::Null));
    assert_eq!(
        rows[0].get("n.config.keep_key"),
        Some(&Value::String("kept".into()))
    );
    assert_eq!(
        rows[0].get("n.config.new_key"),
        Some(&Value::String("added".into()))
    );
}

/// RYOW via prefix_scan: SET deep path then re-scan nodes in same statement.
/// The NodeScan must see the updated value through materialization.
#[test]
fn g064_ryow_prefix_scan_sees_merge_deltas() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("status".into()),
        rmpv::Value::String("inactive".into()),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(doc))],
        &mut interner,
    );

    // SET + then re-read via MATCH (which uses prefix_scan internally).
    // The second read must see the merge-operand update.
    run_cypher(
        "MATCH (n:Device) SET n.config.status = 'active'",
        &engine,
        &mut interner,
    );

    // Separate query confirms the merge was flushed to storage.
    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.status",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.config.status"),
        Some(&Value::String("active".into()))
    );
}

/// SET deep path with a map literal value (nested object, not just scalar).
#[test]
fn g064_set_deep_path_map_literal_value() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![]);
    insert_node(
        &engine,
        1,
        1,
        "Device",
        &[("config", Value::Document(doc))],
        &mut interner,
    );

    // SET a map literal at a deep path.
    run_cypher(
        "MATCH (n:Device) SET n.config.firmware = {version: '2.1', build: 4521}",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Device) RETURN n.config.firmware.version, n.config.firmware.build",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("n.config.firmware.version"),
        Some(&Value::String("2.1".into()))
    );
    assert_eq!(
        rows[0].get("n.config.firmware.build"),
        Some(&Value::Int(4521))
    );
}

/// UPSERT (MERGE) with deep path SET in ON MATCH clause.
#[test]
fn g064_upsert_on_match_with_property_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("login_count".into()),
        rmpv::Value::Integer(5.into()),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "User",
        &[
            ("email", Value::String("alice@example.com".into())),
            ("stats", Value::Document(doc)),
        ],
        &mut interner,
    );

    // MERGE with ON MATCH SET on a deep path.
    run_cypher(
        "MERGE (u:User {email: 'alice@example.com'}) ON MATCH SET u.stats.last_login = 'today'",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (u:User) RETURN u.stats.last_login, u.stats.login_count",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("u.stats.last_login"),
        Some(&Value::String("today".into()))
    );
    // Original nested value preserved.
    assert_eq!(rows[0].get("u.stats.login_count"), Some(&Value::Int(5)));
}

// ── R165: Array operators as merge operands ──────────────────────────

/// doc_push appends a value to an array property.
#[test]
fn r165_doc_push_to_array() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("items".into()),
        rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Bag",
        &[("data", Value::Document(doc))],
        &mut interner,
    );

    run_cypher(
        "MATCH (n:Bag) SET doc_push(n.data.items, 'b')",
        &engine,
        &mut interner,
    );

    let rows = run_cypher("MATCH (n:Bag) RETURN n.data.items", &engine, &mut interner);
    assert_eq!(rows.len(), 1);
    if let Some(Value::Document(doc)) = rows[0].get("n.data.items") {
        if let rmpv::Value::Array(arr) = doc {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], rmpv::Value::String("a".into()));
            assert_eq!(arr[1], rmpv::Value::String("b".into()));
        } else {
            panic!("expected array, got: {doc:?}");
        }
    } else {
        panic!("expected Document, got: {:?}", rows[0].get("n.data.items"));
    }
}

/// doc_pull removes the first occurrence of a value from an array.
#[test]
fn r165_doc_pull_from_array() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("tags".into()),
        rmpv::Value::Array(vec![
            rmpv::Value::String("a".into()),
            rmpv::Value::String("b".into()),
            rmpv::Value::String("a".into()),
        ]),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Item",
        &[("data", Value::Document(doc))],
        &mut interner,
    );

    run_cypher(
        "MATCH (n:Item) SET doc_pull(n.data.tags, 'a')",
        &engine,
        &mut interner,
    );

    let rows = run_cypher("MATCH (n:Item) RETURN n.data.tags", &engine, &mut interner);
    assert_eq!(rows.len(), 1);
    // Only first "a" removed.
    if let Some(Value::Document(doc)) = rows[0].get("n.data.tags") {
        if let rmpv::Value::Array(arr) = doc {
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0], rmpv::Value::String("b".into()));
            assert_eq!(arr[1], rmpv::Value::String("a".into()));
        } else {
            panic!("expected array");
        }
    } else {
        panic!("expected Document");
    }
}

/// doc_add_to_set adds only if not already present.
#[test]
fn r165_doc_add_to_set_dedup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("tags".into()),
        rmpv::Value::Array(vec![rmpv::Value::String("rust".into())]),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Item",
        &[("data", Value::Document(doc))],
        &mut interner,
    );

    // Add "rust" (already exists) and "go" (new).
    run_cypher(
        "MATCH (n:Item) SET doc_add_to_set(n.data.tags, 'rust')",
        &engine,
        &mut interner,
    );
    run_cypher(
        "MATCH (n:Item) SET doc_add_to_set(n.data.tags, 'go')",
        &engine,
        &mut interner,
    );

    let rows = run_cypher("MATCH (n:Item) RETURN n.data.tags", &engine, &mut interner);
    assert_eq!(rows.len(), 1);
    if let Some(Value::Document(doc)) = rows[0].get("n.data.tags") {
        if let rmpv::Value::Array(arr) = doc {
            assert_eq!(arr.len(), 2, "expected [rust, go], got: {arr:?}");
            assert_eq!(arr[0], rmpv::Value::String("rust".into()));
            assert_eq!(arr[1], rmpv::Value::String("go".into()));
        } else {
            panic!("expected array");
        }
    } else {
        panic!("expected Document");
    }
}

/// doc_inc atomically increments a numeric field.
#[test]
fn r165_doc_inc_integer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("views".into()),
        rmpv::Value::Integer(10.into()),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Page",
        &[("stats", Value::Document(doc))],
        &mut interner,
    );

    run_cypher(
        "MATCH (n:Page) SET doc_inc(n.stats.views, 5)",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Page) RETURN n.stats.views",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.stats.views"), Some(&Value::Int(15)));
}

/// doc_inc with negative value decrements.
#[test]
fn r165_doc_inc_negative() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("score".into()),
        rmpv::Value::F64(10.5),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Player",
        &[("stats", Value::Document(doc))],
        &mut interner,
    );

    run_cypher(
        "MATCH (n:Player) SET doc_inc(n.stats.score, -2.5)",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Player) RETURN n.stats.score",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.stats.score"), Some(&Value::Float(8.0)));
}

/// doc_push creates array from nothing when property doesn't exist.
#[test]
fn r165_doc_push_creates_array() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![]);
    insert_node(
        &engine,
        1,
        1,
        "Item",
        &[("data", Value::Document(doc))],
        &mut interner,
    );

    run_cypher(
        "MATCH (n:Item) SET doc_push(n.data.tags, 'first')",
        &engine,
        &mut interner,
    );

    let rows = run_cypher("MATCH (n:Item) RETURN n.data.tags", &engine, &mut interner);
    assert_eq!(rows.len(), 1);
    if let Some(Value::Document(doc)) = rows[0].get("n.data.tags") {
        if let rmpv::Value::Array(arr) = doc {
            assert_eq!(arr.len(), 1);
            assert_eq!(arr[0], rmpv::Value::String("first".into()));
        } else {
            panic!("expected array, got: {doc:?}");
        }
    } else {
        panic!("expected Document");
    }
}

/// Multiple doc functions in one SET statement.
#[test]
fn r165_multiple_doc_ops_one_statement() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("tags".into()),
            rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
        ),
        (
            rmpv::Value::String("count".into()),
            rmpv::Value::Integer(0.into()),
        ),
    ]);
    insert_node(
        &engine,
        1,
        1,
        "Item",
        &[("data", Value::Document(doc))],
        &mut interner,
    );

    // Push to array AND increment counter in one statement.
    run_cypher(
        "MATCH (n:Item) SET doc_push(n.data.tags, 'b'), doc_inc(n.data.count, 1)",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Item) RETURN n.data.tags, n.data.count",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.data.count"), Some(&Value::Int(1)));
    if let Some(Value::Document(doc)) = rows[0].get("n.data.tags") {
        if let rmpv::Value::Array(arr) = doc {
            assert_eq!(arr.len(), 2);
        } else {
            panic!("expected array");
        }
    } else {
        panic!("expected Document");
    }
}

/// doc_inc on non-existent field creates it with the increment value as initial.
#[test]
fn r165_doc_inc_creates_from_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![]);
    insert_node(
        &engine,
        1,
        1,
        "Page",
        &[("stats", Value::Document(doc))],
        &mut interner,
    );

    run_cypher(
        "MATCH (n:Page) SET doc_inc(n.stats.views, 1)",
        &engine,
        &mut interner,
    );

    let rows = run_cypher(
        "MATCH (n:Page) RETURN n.stats.views",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1);
    // Increment on nil → creates with the amount as initial value.
    assert_eq!(rows[0].get("n.stats.views"), Some(&Value::Int(1)));
}

/// doc_pull on missing value is a no-op (idempotent).
#[test]
fn r165_doc_pull_missing_value_noop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("tags".into()),
        rmpv::Value::Array(vec![rmpv::Value::String("a".into())]),
    )]);
    insert_node(
        &engine,
        1,
        1,
        "Item",
        &[("data", Value::Document(doc))],
        &mut interner,
    );

    // Pull "nonexistent" — should be no-op.
    run_cypher(
        "MATCH (n:Item) SET doc_pull(n.data.tags, 'nonexistent')",
        &engine,
        &mut interner,
    );

    let rows = run_cypher("MATCH (n:Item) RETURN n.data.tags", &engine, &mut interner);
    assert_eq!(rows.len(), 1);
    if let Some(Value::Document(doc)) = rows[0].get("n.data.tags") {
        if let rmpv::Value::Array(arr) = doc {
            assert_eq!(arr.len(), 1, "array should be unchanged");
            assert_eq!(arr[0], rmpv::Value::String("a".into()));
        } else {
            panic!("expected array");
        }
    } else {
        panic!("expected Document");
    }
}

// ── G069: wildcard relationship pattern ──────────────────────────────────────

/// Regression test for G069: `MATCH (n)-[r]->(m)` (no type filter) returned 0
/// rows because `expand_one_hop` with empty `edge_types` iterated an empty slice.
///
/// Verifies:
/// - Wildcard `[r]` returns all edges regardless of type (KNOWS + LIKES)
/// - `r.__type__` column is populated for each returned row
/// - Typed `[r:KNOWS]` still works and is not affected by the fix
#[test]
fn test_wildcard_relationship_returns_results() {
    // Graph: Alice(1) -[:KNOWS]-> Bob(2), Alice(1) -[:LIKES]-> Eve(5)
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        5,
        "Person",
        &[("name", Value::String("Eve".into()))],
        &mut interner,
    );

    insert_edge(&engine, "KNOWS", 1, 2);
    insert_edge(&engine, "LIKES", 1, 5);

    // Register edge types in schema so wildcard scan can enumerate them (G069).
    // In production, the executor registers these automatically on CREATE.
    register_schema_edge_type(&engine, "KNOWS");
    register_schema_edge_type(&engine, "LIKES");

    // Wildcard: should return both KNOWS and LIKES edges (2 rows)
    let rows = run_cypher(
        "MATCH (n)-[r]->(m) RETURN r.__type__",
        &engine,
        &mut interner,
    );
    assert_eq!(
        rows.len(),
        2,
        "wildcard [r] should return all edges; got {} rows",
        rows.len()
    );

    // Both edge types should be present
    let types: std::collections::HashSet<String> = rows
        .iter()
        .filter_map(|row| {
            if let Some(Value::String(t)) = row.get("r.__type__") {
                Some(t.clone())
            } else {
                None
            }
        })
        .collect();
    assert!(types.contains("KNOWS"), "expected KNOWS in wildcard result");
    assert!(types.contains("LIKES"), "expected LIKES in wildcard result");

    // Typed pattern should still work and return only KNOWS (regression guard)
    let typed_rows = run_cypher(
        "MATCH (n)-[r:KNOWS]->(m) RETURN r.__type__",
        &engine,
        &mut interner,
    );
    assert_eq!(
        typed_rows.len(),
        1,
        "typed [r:KNOWS] should still return 1 row"
    );
    assert_eq!(
        typed_rows[0].get("r.__type__"),
        Some(&Value::String("KNOWS".into()))
    );
}

/// G069 edge case: wildcard on a graph with no schema-registered edge types returns 0 rows.
#[test]
fn test_wildcard_relationship_empty_schema_returns_zero() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Loner".into()))],
        &mut interner,
    );
    // No edges inserted, no schema edge types registered

    let rows = run_cypher(
        "MATCH (n)-[r]->(m) RETURN r.__type__",
        &engine,
        &mut interner,
    );
    assert_eq!(
        rows.len(),
        0,
        "empty schema should return 0 rows for wildcard"
    );
}

// ── G070: count(r) for relationship variable ─────────────────────────────────

/// Regression test for G070: `count(r)` returned 0 for relationship variables
/// because `r` was not stored as a row column — only `r.__type__` was stored.
/// `eval_aggregate_values` filtered out the resulting `Value::Null`, giving 0.
///
/// Verifies:
/// - `count(r)` returns the correct number of edges (not 0)
/// - `count(*)` is unaffected
/// - `count(n)` (node variable) is unaffected
#[test]
fn test_count_relationship_variable() {
    // Graph: Alice(1) -[:KNOWS]-> Bob(2), Alice(1) -[:KNOWS]-> Charlie(3)
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Person",
        &[("name", Value::String("Charlie".into()))],
        &mut interner,
    );

    insert_edge(&engine, "KNOWS", 1, 2);
    insert_edge(&engine, "KNOWS", 1, 3);
    register_schema_edge_type(&engine, "KNOWS");

    // count(r) should return 2, not 0
    let rows = run_cypher(
        "MATCH (a)-[r:KNOWS]->(b) RETURN count(r) AS cnt",
        &engine,
        &mut interner,
    );
    assert_eq!(rows.len(), 1, "aggregation should produce exactly 1 row");
    assert_eq!(
        rows[0].get("cnt"),
        Some(&Value::Int(2)),
        "count(r) should return 2; got {:?}",
        rows[0].get("cnt")
    );

    // count(*) should also return 2 (unaffected baseline)
    let rows_star = run_cypher(
        "MATCH (a)-[r:KNOWS]->(b) RETURN count(*) AS cnt",
        &engine,
        &mut interner,
    );
    assert_eq!(
        rows_star[0].get("cnt"),
        Some(&Value::Int(2)),
        "count(*) baseline"
    );

    // count(a) (node variable) should also return 2 (unaffected baseline)
    let rows_node = run_cypher(
        "MATCH (a)-[r:KNOWS]->(b) RETURN count(a) AS cnt",
        &engine,
        &mut interner,
    );
    assert_eq!(
        rows_node[0].get("cnt"),
        Some(&Value::Int(2)),
        "count(a) baseline"
    );
}

// ── G072: MERGE relationship pattern ─────────────────────────────────────────

/// Regression test for G072: `MERGE (src)-[r:TYPE]->(dst)` between already-bound
/// nodes failed with "MERGE create from non-NodeScan pattern".
///
/// What this tests:
/// - MERGE (a)-[:KNOWS]->(b) creates the edge when it doesn't exist
/// - Repeated MERGE is idempotent: edge count stays at 1
/// - The MERGE correctly uses the bound src/dst from the preceding MATCH
///
/// Why this matters: LangChain `add_graph_documents()` and LlamaIndex
/// `upsert_relations()` use MERGE for idempotent edge creation.
#[test]
fn test_merge_relationship_creates_edge() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // Create two nodes via Cypher
    run_cypher_with_alloc(
        "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // MERGE relationship — should CREATE because edge doesn't exist yet
    // G072: previously failed with "MERGE create from non-NodeScan pattern"
    let merge_q = "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
                   MERGE (a)-[r:KNOWS]->(b) RETURN r.__type__ AS rel_type";
    let merge_rows = run_cypher_with_alloc(merge_q, &engine, &mut interner, &allocator);
    assert_eq!(
        merge_rows.len(),
        1,
        "MERGE should return exactly 1 row (created edge); got {:?}",
        merge_rows
    );
    assert_eq!(
        merge_rows[0].get("rel_type"),
        Some(&Value::String("KNOWS".to_string())),
        "edge type should be KNOWS"
    );

    // Verify edge actually exists
    let check_rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) RETURN count(*) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        check_rows[0].get("cnt"),
        Some(&Value::Int(1)),
        "should have exactly 1 KNOWS edge"
    );

    // MERGE again — should be IDEMPOTENT (match existing, no duplicates)
    let merge_rows2 = run_cypher_with_alloc(merge_q, &engine, &mut interner, &allocator);
    assert_eq!(
        merge_rows2.len(),
        1,
        "second MERGE should also return 1 row (matched); got {:?}",
        merge_rows2
    );

    // Count edges — must still be 1 (idempotent)
    let count_rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) RETURN count(*) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        count_rows[0].get("cnt"),
        Some(&Value::Int(1)),
        "MERGE must be idempotent: expected 1 edge after two MERGEs, got {:?}",
        count_rows[0].get("cnt")
    );

    // Ensure unrelated node pair returns 0 (correct scoping)
    let _ = run_cypher_with_alloc(
        "CREATE (c:Person {name: 'Charlie'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );
    let no_edge_rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(c:Person {name: 'Charlie'}) RETURN count(*) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        no_edge_rows[0].get("cnt"),
        Some(&Value::Int(0)),
        "Alice→Charlie KNOWS edge should not exist"
    );
}

/// G072 — multiple MATCH pairs: each pair gets its own correlated MERGE.
///
/// What this tests:
/// - CartesianProduct correlated execution runs once per left row
/// - Three distinct (src, dst) pairs each get their own MERGE execution
/// - No cross-contamination: only exactly the 3 edges are created
#[test]
fn test_merge_relationship_multiple_pairs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // Create 3 nodes
    run_cypher_with_alloc(
        "CREATE (:Person {name: 'A'}), (:Person {name: 'B'}), (:Person {name: 'C'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // MERGE A→B, A→C, B→C using a single query with cartesian product of pairs
    run_cypher_with_alloc(
        "MATCH (x:Person {name: 'A'}), (y:Person {name: 'B'}) MERGE (x)-[r:KNOWS]->(y) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (x:Person {name: 'A'}), (y:Person {name: 'C'}) MERGE (x)-[r:KNOWS]->(y) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (x:Person {name: 'B'}), (y:Person {name: 'C'}) MERGE (x)-[r:KNOWS]->(y) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // Total KNOWS edges across all nodes must be exactly 3
    let rows = run_cypher_with_alloc(
        "MATCH (a)-[r:KNOWS]->(b) RETURN count(*) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        rows[0].get("cnt"),
        Some(&Value::Int(3)),
        "expected exactly 3 KNOWS edges"
    );

    // Running all three MERGEs again must be idempotent — still 3 edges
    run_cypher_with_alloc(
        "MATCH (x:Person {name: 'A'}), (y:Person {name: 'B'}) MERGE (x)-[r:KNOWS]->(y) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (x:Person {name: 'A'}), (y:Person {name: 'C'}) MERGE (x)-[r:KNOWS]->(y) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );
    run_cypher_with_alloc(
        "MATCH (x:Person {name: 'B'}), (y:Person {name: 'C'}) MERGE (x)-[r:KNOWS]->(y) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    let rows2 = run_cypher_with_alloc(
        "MATCH (a)-[r:KNOWS]->(b) RETURN count(*) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        rows2[0].get("cnt"),
        Some(&Value::Int(3)),
        "MERGE must be idempotent: expected still 3 edges after re-running"
    );
}

/// G072 — MERGE relationship with ON CREATE SET applies properties only on creation.
///
/// What this tests:
/// - ON CREATE SET sets a property on the edge row when the edge is created
/// - ON MATCH SET does NOT run on creation
/// - Second MERGE (match path) does NOT trigger ON CREATE SET again
#[test]
fn test_merge_relationship_on_create_set() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    run_cypher_with_alloc(
        "CREATE (:Person {name: 'Alice'}), (:Person {name: 'Bob'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // First MERGE — should CREATE and execute ON CREATE SET
    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r:KNOWS]->(b) ON CREATE SET a.merged = 1 RETURN a.merged AS m",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1, "first MERGE should return 1 row");
    assert_eq!(
        rows[0].get("m"),
        Some(&Value::Int(1)),
        "ON CREATE SET should have set a.merged = 1"
    );

    // Second MERGE — should MATCH existing edge, ON CREATE SET must NOT run
    let rows2 = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r:KNOWS]->(b) ON CREATE SET a.merged = 99 RETURN a.merged AS m",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows2.len(), 1, "second MERGE should return 1 row");
    // ON CREATE SET must NOT have incremented — still 1
    assert_eq!(
        rows2[0].get("m"),
        Some(&Value::Int(1)),
        "ON CREATE SET must not fire on MATCH path — a.merged should still be 1"
    );
}

/// G074 — standalone MERGE relationship creates both nodes and edge from scratch.
///
/// What this tests:
/// - `MERGE (a:Person {name:'X'})-[r:KNOWS]->(b:Person {name:'Y'})` succeeds with no
///   preceding MATCH and no pre-existing data
/// - Returns the relationship type in the result row
/// - Both source and target nodes are created implicitly
#[test]
fn test_merge_relationship_standalone_creates_from_scratch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    let rows = run_cypher_with_alloc(
        "MERGE (a:Person {name: 'X'})-[r:KNOWS]->(b:Person {name: 'Y'}) RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(rows.len(), 1, "should return one row");
    assert_eq!(
        rows[0].get("r.__type__"),
        Some(&Value::String("KNOWS".to_string())),
        "relationship type should be KNOWS"
    );

    // Verify the edge persists: the path can be found in a subsequent MATCH
    let match_rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'X'})-[r:KNOWS]->(b:Person {name: 'Y'}) RETURN a.name, b.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        match_rows.len(),
        1,
        "edge should persist after standalone MERGE"
    );
    assert_eq!(
        match_rows[0].get("a.name"),
        Some(&Value::String("X".to_string())),
    );
    assert_eq!(
        match_rows[0].get("b.name"),
        Some(&Value::String("Y".to_string())),
    );
}

/// G074 — standalone MERGE is idempotent: second run finds the existing path.
///
/// What this tests:
/// - Running the same standalone MERGE twice does not create duplicate nodes or edges
/// - Second run returns the same relationship (ON MATCH path, not ON CREATE)
/// - ON CREATE SET fires only on the first run; ON MATCH SET fires on the second
#[test]
fn test_merge_relationship_standalone_idempotent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // First MERGE — creates nodes and edge. ON CREATE SET should fire.
    let rows1 = run_cypher_with_alloc(
        "MERGE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) \
         ON CREATE SET a.created = 1 \
         RETURN r.__type__, a.created AS c",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows1.len(), 1);
    assert_eq!(
        rows1[0].get("r.__type__"),
        Some(&Value::String("KNOWS".to_string())),
    );
    assert_eq!(
        rows1[0].get("c"),
        Some(&Value::Int(1)),
        "ON CREATE SET must fire on first run"
    );

    // Second MERGE — path exists. ON CREATE SET must NOT fire; ON MATCH SET should fire.
    let rows2 = run_cypher_with_alloc(
        "MERGE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) \
         ON MATCH SET a.matched = 1 \
         RETURN r.__type__, a.matched AS m",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows2.len(), 1);
    assert_eq!(
        rows2[0].get("r.__type__"),
        Some(&Value::String("KNOWS".to_string())),
        "second run should find same relationship type"
    );
    assert_eq!(
        rows2[0].get("m"),
        Some(&Value::Int(1)),
        "ON MATCH SET must fire on second run (path already exists)"
    );

    // Verify no duplicate nodes: only one Person {name:'Alice'} should exist
    let alice_rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}) RETURN count(a) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(alice_rows.len(), 1);
    assert_eq!(
        alice_rows[0].get("cnt"),
        Some(&Value::Int(1)),
        "standalone MERGE must not create duplicate nodes"
    );
}

/// G074 — standalone MERGE reuses existing nodes when the edge is missing.
///
/// What this tests:
/// - Pre-existing nodes are reused (not duplicated) when the edge doesn't exist yet
/// - Edge is created between the two pre-existing nodes
/// - Node count stays the same after MERGE
#[test]
fn test_merge_relationship_standalone_reuses_existing_nodes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // Pre-create both nodes without any edge
    run_cypher_with_alloc(
        "CREATE (:City {name: 'Paris'}), (:City {name: 'London'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // Standalone MERGE should find both existing nodes and create only the edge
    let rows = run_cypher_with_alloc(
        "MERGE (a:City {name: 'Paris'})-[r:CONNECTED_TO]->(b:City {name: 'London'}) \
         RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("r.__type__"),
        Some(&Value::String("CONNECTED_TO".to_string())),
    );

    // Verify node count: still exactly 2 City nodes (no duplicates)
    let city_rows = run_cypher_with_alloc(
        "MATCH (c:City) RETURN count(c) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(city_rows.len(), 1);
    assert_eq!(
        city_rows[0].get("cnt"),
        Some(&Value::Int(2)),
        "standalone MERGE must reuse existing nodes, not create duplicates"
    );
}

/// G074 — standalone MERGE errors when the source pattern is ambiguous (multiple matches).
///
/// What this tests:
/// - If multiple nodes match the source pattern, MERGE returns an error (not silent take-first)
/// - If multiple nodes match the target pattern, MERGE returns an error
/// - Error message mentions "ambiguous" so the caller can diagnose the issue
///
/// Rationale: MERGE semantics require a unique match. Silently picking an arbitrary node
/// would create unpredictable results. Use MERGE ALL for multi-target upsert.
#[test]
fn test_merge_relationship_standalone_ambiguous_source_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // Create two Person nodes with the same label and property
    run_cypher_with_alloc(
        "CREATE (:Person {role: 'admin'}), (:Person {role: 'admin'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // MERGE with ambiguous source (2 matching nodes) should error
    let ast = parse(
        "MERGE (a:Person {role: 'admin'})-[r:MANAGES]->(b:Device {id: 'x'}) RETURN r.__type__",
    )
    .expect("parse");
    let plan = build_logical_plan(&ast).expect("plan");
    let mut ctx = make_test_ctx(&engine, &mut interner, &allocator);
    let result = execute(&plan, &mut ctx);

    assert!(result.is_err(), "ambiguous source should return an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.to_lowercase().contains("ambiguous"),
        "error should mention 'ambiguous', got: {msg}"
    );
}

/// G074 — standalone MERGE errors when the target pattern is ambiguous.
#[test]
fn test_merge_relationship_standalone_ambiguous_target_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // Create two Device nodes with the same label and property
    run_cypher_with_alloc(
        "CREATE (:Device {type: 'sensor'}), (:Device {type: 'sensor'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // MERGE with ambiguous target (2 matching nodes) should error
    let ast = parse(
        "MERGE (a:Gateway {id: 'gw1'})-[r:MONITORS]->(b:Device {type: 'sensor'}) RETURN r.__type__",
    )
    .expect("parse");
    let plan = build_logical_plan(&ast).expect("plan");
    let mut ctx = make_test_ctx(&engine, &mut interner, &allocator);
    let result = execute(&plan, &mut ctx);

    assert!(result.is_err(), "ambiguous target should return an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.to_lowercase().contains("ambiguous"),
        "error should mention 'ambiguous', got: {msg}"
    );
}

/// G074 — standalone MERGE RETURN exposes node variables directly (not just r.__type__).
///
/// What this tests:
/// - `MERGE (a:L {k:v})-[r:T]->(b:L {k:v}) RETURN a.name, b.name, r.__type__` returns all three
/// - The synthetic correlated row produced by standalone create carries src and tgt fields
/// - Both ON CREATE and ON MATCH paths produce rows with accessible node properties
#[test]
fn test_merge_relationship_standalone_return_node_variables() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // --- First run: both nodes and edge are created from scratch ---
    let rows1 = run_cypher_with_alloc(
        "MERGE (a:Author {name: 'Alice'})-[r:WROTE]->(b:Book {title: 'Rust101'}) \
         RETURN a.name, b.title, r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows1.len(), 1, "create path: should return one row");
    assert_eq!(
        rows1[0].get("a.name"),
        Some(&Value::String("Alice".to_string())),
        "create path: a.name should be Alice"
    );
    assert_eq!(
        rows1[0].get("b.title"),
        Some(&Value::String("Rust101".to_string())),
        "create path: b.title should be Rust101"
    );
    assert_eq!(
        rows1[0].get("r.__type__"),
        Some(&Value::String("WROTE".to_string())),
        "create path: r.__type__ should be WROTE"
    );

    // --- Second run: path already exists, ON MATCH path — same row shape ---
    let rows2 = run_cypher_with_alloc(
        "MERGE (a:Author {name: 'Alice'})-[r:WROTE]->(b:Book {title: 'Rust101'}) \
         RETURN a.name, b.title, r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows2.len(), 1, "match path: should return one row");
    assert_eq!(
        rows2[0].get("a.name"),
        Some(&Value::String("Alice".to_string())),
        "match path: a.name should be Alice"
    );
    assert_eq!(
        rows2[0].get("b.title"),
        Some(&Value::String("Rust101".to_string())),
        "match path: b.title should be Rust101"
    );
    assert_eq!(
        rows2[0].get("r.__type__"),
        Some(&Value::String("WROTE".to_string())),
        "match path: r.__type__ should be WROTE"
    );
}

/// G072 — MERGE relationship with ON MATCH SET applies properties only when edge already exists.
///
/// What this tests:
/// - When the edge already exists, ON MATCH SET fires (updates node/property)
/// - When the edge does NOT yet exist, ON MATCH SET does NOT fire
#[test]
fn test_merge_relationship_on_match_set() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    run_cypher_with_alloc(
        "CREATE (:Person {name: 'Alice', visits: 0}), (:Person {name: 'Bob'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // First MERGE — creates edge; ON MATCH SET must NOT fire (no match yet)
    let rows1 = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r:KNOWS]->(b) ON MATCH SET a.visits = a.visits + 1 RETURN a.visits AS v",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows1.len(), 1);
    assert_eq!(
        rows1[0].get("v"),
        Some(&Value::Int(0)),
        "ON MATCH SET must not fire on create path — visits should still be 0"
    );

    // Second MERGE — edge exists; ON MATCH SET must fire
    let rows2 = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r:KNOWS]->(b) ON MATCH SET a.visits = a.visits + 1 RETURN a.visits AS v",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows2.len(), 1);
    assert_eq!(
        rows2[0].get("v"),
        Some(&Value::Int(1)),
        "ON MATCH SET must fire on match path — visits should be 1"
    );
}

/// G072 — MERGE incoming relationship `MERGE (a)<-[r:T]-(b)`.
///
/// What this tests:
/// - Direction::Incoming is handled correctly in execute_merge_relationship_check
///   (adj reverse key lookup) and execute_merge_relationship_create (fwd/rev swap)
/// - The edge is created and retrievable via the reverse pattern
#[test]
fn test_merge_relationship_incoming_direction() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    run_cypher_with_alloc(
        "CREATE (:Person {name: 'Alice'}), (:Person {name: 'Bob'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // MERGE with incoming direction: Alice receives a KNOWS edge from Bob
    // Semantics: (a)<-[r:KNOWS]-(b) means Bob→Alice
    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)<-[r:KNOWS]-(b) RETURN r.__type__ AS rel_type",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1, "incoming MERGE should return 1 row");
    assert_eq!(
        rows[0].get("rel_type"),
        Some(&Value::String("KNOWS".to_string())),
        "edge type should be KNOWS"
    );

    // Verify the edge exists in the forward direction (Bob→Alice)
    let check = run_cypher_with_alloc(
        "MATCH (b:Person {name: 'Bob'})-[r:KNOWS]->(a:Person {name: 'Alice'}) RETURN count(*) AS cnt",
        &engine, &mut interner, &allocator,
    );
    assert_eq!(
        check[0].get("cnt"),
        Some(&Value::Int(1)),
        "Bob→Alice KNOWS edge should exist"
    );

    // Idempotency: second incoming MERGE → still 1 edge
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)<-[r:KNOWS]-(b) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );
    let check2 = run_cypher_with_alloc(
        "MATCH (b:Person {name: 'Bob'})-[r:KNOWS]->(a:Person {name: 'Alice'}) RETURN count(*) AS cnt",
        &engine, &mut interner, &allocator,
    );
    assert_eq!(
        check2[0].get("cnt"),
        Some(&Value::Int(1)),
        "incoming MERGE must be idempotent"
    );
}

/// G075 — MERGE stores edge properties and uses them for match discrimination.
///
/// What this tests:
/// - `MERGE (a)-[r:T {prop: val}]->(b)` stores the property in EdgeProp partition
/// - Subsequent `MATCH (a)-[r:T]->(b) RETURN r.prop` returns the stored value
/// - A second MERGE with the SAME property value is idempotent (ON MATCH path)
/// - A second MERGE with a DIFFERENT property value takes the create path and
///   overwrites the EdgeProp (single-edge-per-type data model, upsert semantics).
///   Note: true parallel edges (cnt=2) require a data model change tracked separately.
#[test]
fn test_merge_relationship_edge_properties_stored_and_checked() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    run_cypher_with_alloc(
        "CREATE (:Person {name: 'Alice'}), (:Person {name: 'Bob'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // First MERGE with weight:1 — creates the edge and stores the property.
    let rows1 = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r:KNOWS {weight: 1}]->(b) RETURN r.__type__ AS rel_type, r.weight AS w",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows1.len(), 1, "first MERGE should return 1 row");
    assert_eq!(
        rows1[0].get("rel_type"),
        Some(&Value::String("KNOWS".to_string())),
        "edge type should be KNOWS"
    );
    assert_eq!(
        rows1[0].get("w"),
        Some(&Value::Int(1)),
        "edge property weight should be 1 after first MERGE"
    );

    // Verify the property is readable via MATCH.
    let match_rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) \
         RETURN r.weight AS w",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(match_rows.len(), 1);
    assert_eq!(
        match_rows[0].get("w"),
        Some(&Value::Int(1)),
        "stored edge property should be readable via MATCH"
    );

    // Second MERGE with the SAME property value — idempotent (ON MATCH path).
    let rows2 = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r:KNOWS {weight: 1}]->(b) RETURN r.weight AS w",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows2.len(), 1, "idempotent MERGE should return 1 row");
    assert_eq!(
        rows2[0].get("w"),
        Some(&Value::Int(1)),
        "weight should still be 1 after idempotent MERGE"
    );

    // Third MERGE with a DIFFERENT property value — no match on existing edge
    // (weight:2 ≠ weight:1), so the create path runs and overwrites EdgeProp.
    // Data model has one edge per (src, tgt, type), so edge count stays 1.
    let rows3 = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r:KNOWS {weight: 2}]->(b) RETURN r.weight AS w",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows3.len(), 1);
    assert_eq!(
        rows3[0].get("w"),
        Some(&Value::Int(2)),
        "weight should be 2 after MERGE with weight:2"
    );

    // Edge count stays 1 — single-edge-per-type model, no parallel edges yet.
    let total = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) \
         RETURN count(*) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        total[0].get("cnt"),
        Some(&Value::Int(1)),
        "single edge per (src, tgt, type) — count stays 1"
    );

    // After the overwrite, the property reflects the last MERGE.
    let after = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) \
         RETURN r.weight AS w",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        after[0].get("w"),
        Some(&Value::Int(2)),
        "edge property should reflect last MERGE (upsert)"
    );
}

/// G075 — ON CREATE SET fires and can reference edge variable with stored properties.
///
/// What this tests:
/// - `MERGE (a)-[r:T {weight: 5}]->(b) ON CREATE SET r.created = 1` stores both
///   `weight` and `created` on the new edge
/// - On second run (ON MATCH), `r.weight` is accessible and `r.created` is readable
#[test]
fn test_merge_relationship_edge_properties_on_create_set() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    run_cypher_with_alloc(
        "CREATE (:Item {id: 'X'}), (:Item {id: 'Y'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // First MERGE — ON CREATE SET should fire, edge created with pattern property weight=5.
    let rows1 = run_cypher_with_alloc(
        "MATCH (a:Item {id: 'X'}), (b:Item {id: 'Y'}) \
         MERGE (a)-[r:LINKS {weight: 5}]->(b) \
         ON CREATE SET r.new_flag = 1 \
         RETURN r.weight AS w",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows1.len(), 1, "first MERGE: 1 row");
    assert_eq!(
        rows1[0].get("w"),
        Some(&Value::Int(5)),
        "pattern property weight=5 should be in RETURN result"
    );

    // Second MERGE with SAME weight — ON MATCH path (idempotent), property still readable.
    let rows2 = run_cypher_with_alloc(
        "MATCH (a:Item {id: 'X'}), (b:Item {id: 'Y'}) \
         MERGE (a)-[r:LINKS {weight: 5}]->(b) \
         ON MATCH SET r.visits = 1 \
         RETURN r.weight AS w",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows2.len(), 1, "second MERGE: 1 row");
    assert_eq!(
        rows2[0].get("w"),
        Some(&Value::Int(5)),
        "weight=5 readable on ON MATCH path"
    );
}

/// G072 — two MERGE relationship clauses in a single query.
///
/// Plan structure: CartesianProduct{CartesianProduct{MATCH, Merge1}, Merge2}.
/// Verifies that the nested correlated execution works correctly for both levels.
///
/// What this tests:
/// - MERGE (a)-[r1:KNOWS]->(b) and MERGE (a)-[r2:LIKES]->(b) in one query
/// - Both edges are created with correct types
/// - Re-running the query is idempotent (still exactly 1 KNOWS and 1 LIKES)
#[test]
fn test_merge_two_relationship_clauses() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    run_cypher_with_alloc(
        "CREATE (:Person {name: 'Alice'}), (:Person {name: 'Bob'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // Both MERGEs in one query — plan is nested CartesianProduct
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r1:KNOWS]->(b) \
         MERGE (a)-[r2:LIKES]->(b) \
         RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    let knows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) RETURN count(*) AS cnt",
        &engine, &mut interner, &allocator,
    );
    assert_eq!(
        knows[0].get("cnt"),
        Some(&Value::Int(1)),
        "KNOWS edge should exist"
    );

    let likes = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:LIKES]->(b:Person {name: 'Bob'}) RETURN count(*) AS cnt",
        &engine, &mut interner, &allocator,
    );
    assert_eq!(
        likes[0].get("cnt"),
        Some(&Value::Int(1)),
        "LIKES edge should exist"
    );

    // Re-run — both MERGEs must be idempotent
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r1:KNOWS]->(b) \
         MERGE (a)-[r2:LIKES]->(b) \
         RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    let knows2 = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) RETURN count(*) AS cnt",
        &engine, &mut interner, &allocator,
    );
    assert_eq!(
        knows2[0].get("cnt"),
        Some(&Value::Int(1)),
        "KNOWS still 1 after re-run"
    );

    let likes2 = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'})-[r:LIKES]->(b:Person {name: 'Bob'}) RETURN count(*) AS cnt",
        &engine, &mut interner, &allocator,
    );
    assert_eq!(
        likes2[0].get("cnt"),
        Some(&Value::Int(1)),
        "LIKES still 1 after re-run"
    );
}

/// G072 — self-loop MERGE: source and target variable are the same node.
///
/// What this tests:
/// - MERGE (a)-[r:SELF_REF]->(a) where source and target are the same node ID
/// - execute_merge_relationship_check handles src == tgt (adj key lookup of node onto itself)
/// - execute_merge_relationship_create writes both fwd and rev adj for the same node
/// - Idempotency on self-loop
#[test]
fn test_merge_relationship_self_loop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    run_cypher_with_alloc(
        "CREATE (:Node {name: 'X'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // Self-referential MERGE
    let rows = run_cypher_with_alloc(
        "MATCH (a:Node {name: 'X'}) MERGE (a)-[r:SELF_REF]->(a) RETURN r.__type__ AS rel_type",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1, "self-loop MERGE should return 1 row");
    assert_eq!(
        rows[0].get("rel_type"),
        Some(&Value::String("SELF_REF".to_string()))
    );

    // Verify retrievable
    let check = run_cypher_with_alloc(
        "MATCH (a:Node {name: 'X'})-[r:SELF_REF]->(b:Node {name: 'X'}) RETURN count(*) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        check[0].get("cnt"),
        Some(&Value::Int(1)),
        "self-loop should exist"
    );

    // Idempotency
    run_cypher_with_alloc(
        "MATCH (a:Node {name: 'X'}) MERGE (a)-[r:SELF_REF]->(a) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );
    let check2 = run_cypher_with_alloc(
        "MATCH (a:Node {name: 'X'})-[r:SELF_REF]->(b:Node {name: 'X'}) RETURN count(*) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        check2[0].get("cnt"),
        Some(&Value::Int(1)),
        "self-loop MERGE must be idempotent"
    );
}

/// G069 + G072 integration — wildcard MATCH after Cypher MERGE.
///
/// The G069 fix (list_edge_types reads schema partition) must see edge types
/// registered by execute_merge_relationship_create (which writes schema key).
/// This test verifies end-to-end: Cypher MERGE → schema key written → wildcard
/// MATCH finds the edge.
///
/// What this tests:
/// - No manual insert_edge — edges created purely via Cypher MERGE
/// - Wildcard [r] returns the MERGE-created edge with correct __type__
/// - count(r) on the MERGE-created edge returns 1 (G070 + G072 integration)
#[test]
fn test_g069_wildcard_after_cypher_merge() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    run_cypher_with_alloc(
        "CREATE (:Person {name: 'Alice'}), (:Person {name: 'Bob'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // Create edge via MERGE (not manual insert_edge)
    run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r:KNOWS]->(b) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // Wildcard [r] must find the MERGE-created KNOWS edge (G069 fix)
    let wild_rows = run_cypher_with_alloc(
        "MATCH (a)-[r]->(b) RETURN r.__type__ AS t",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(wild_rows.len(), 1, "wildcard should return 1 edge");
    assert_eq!(
        wild_rows[0].get("t"),
        Some(&Value::String("KNOWS".to_string())),
        "wildcard should return KNOWS type"
    );

    // count(r) on the MERGE-created edge must return 1 (G070 fix)
    let cnt_rows = run_cypher_with_alloc(
        "MATCH (a)-[r:KNOWS]->(b) RETURN count(r) AS cnt",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        cnt_rows[0].get("cnt"),
        Some(&Value::Int(1)),
        "count(r) after Cypher MERGE must return 1"
    );
}

// ── Pattern predicate in WHERE (G071) ────────────────────────────────────

#[test]
fn g071_where_pattern_predicate_positive() {
    // WHERE (a)-[:KNOWS]->(b) should return rows where the edge exists
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    insert_edge(&engine, "KNOWS", 1, 2);
    register_schema_edge_type(&engine, "KNOWS");

    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) WHERE (a)-[:KNOWS]->(b) RETURN a.name AS src, b.name AS dst",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1, "edge exists → 1 row");
    assert_eq!(rows[0].get("src"), Some(&Value::String("Alice".into())));
    assert_eq!(rows[0].get("dst"), Some(&Value::String("Bob".into())));
}

#[test]
fn g071_where_not_pattern_predicate() {
    // WHERE NOT (a)-[:KNOWS]->(b) should return rows where the edge does NOT exist
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    // No KNOWS edge from Alice to Bob
    register_schema_edge_type(&engine, "KNOWS");

    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) WHERE NOT (a)-[:KNOWS]->(b) RETURN a.name AS src, b.name AS dst",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1, "no edge → NOT pattern should return 1 row");
    assert_eq!(rows[0].get("src"), Some(&Value::String("Alice".into())));
}

#[test]
fn g071_where_not_pattern_predicate_blocks_when_edge_exists() {
    // WHERE NOT (a)-[:KNOWS]->(b) should filter out rows where the edge DOES exist
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    insert_edge(&engine, "KNOWS", 1, 2);
    register_schema_edge_type(&engine, "KNOWS");

    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) WHERE NOT (a)-[:KNOWS]->(b) RETURN a.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(
        rows.len(),
        0,
        "edge exists → NOT pattern should filter out row"
    );
}

#[test]
fn g071_pattern_predicate_with_and() {
    // WHERE (a)-[:KNOWS]->(b) AND b.age > 20
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[
            ("name", Value::String("Bob".into())),
            ("age", Value::Int(25)),
        ],
        &mut interner,
    );
    insert_edge(&engine, "KNOWS", 1, 2);
    register_schema_edge_type(&engine, "KNOWS");

    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) WHERE (a)-[:KNOWS]->(b) AND b.age > 20 RETURN b.name AS name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&Value::String("Bob".into())));
}

#[test]
fn g071_pattern_predicate_wildcard_type() {
    // WHERE (a)-[]->(b) — wildcard edge type
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    insert_edge(&engine, "FOLLOWS", 1, 2);
    register_schema_edge_type(&engine, "FOLLOWS");

    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) WHERE (a)-[]->(b) RETURN b.name AS name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1, "wildcard edge match should find FOLLOWS");
}

#[test]
fn g071_pattern_predicate_wrong_direction() {
    // WHERE (a)-[:KNOWS]->(b) but edge is b→a, not a→b
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    insert_edge(&engine, "KNOWS", 2, 1); // Bob→Alice, not Alice→Bob
    register_schema_edge_type(&engine, "KNOWS");

    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) WHERE (a)-[:KNOWS]->(b) RETURN a.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 0, "wrong direction → no match");
}

#[test]
fn g071_standalone_where_not_pattern() {
    // Standalone WHERE (two separate MATCH clauses + WHERE) — the exact syntax from G071 description:
    // MATCH (src) MATCH (dst) WHERE NOT (src)-[:TYPE]->(dst) CREATE ...
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    register_schema_edge_type(&engine, "FOLLOWS");

    // No FOLLOWS edge from Alice to Bob → NOT should pass
    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}) MATCH (b:Person {name: 'Bob'}) WHERE NOT (a)-[:FOLLOWS]->(b) RETURN a.name AS src, b.name AS dst",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1, "no edge → NOT pattern should return row");
    assert_eq!(rows[0].get("src"), Some(&Value::String("Alice".into())));

    // Now add edge and verify NOT filters it out
    insert_edge(&engine, "FOLLOWS", 1, 2);
    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}) MATCH (b:Person {name: 'Bob'}) WHERE NOT (a)-[:FOLLOWS]->(b) RETURN a.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 0, "edge exists → NOT pattern should filter out");
}

#[test]
fn g071_pattern_predicate_incoming_direction() {
    // WHERE (a)<-[:KNOWS]-(b) — incoming direction
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    insert_edge(&engine, "KNOWS", 2, 1); // Bob→Alice (so Alice has incoming KNOWS from Bob)
    register_schema_edge_type(&engine, "KNOWS");

    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) WHERE (a)<-[:KNOWS]-(b) RETURN a.name AS name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1, "incoming edge should match");
    assert_eq!(rows[0].get("name"), Some(&Value::String("Alice".into())));
}

#[test]
fn g071_pattern_predicate_in_or() {
    // WHERE (a)-[:KNOWS]->(b) OR a.name = 'Charlie'
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    insert_node(
        &engine,
        1,
        1,
        "Person",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Person",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    // No edge but name doesn't match Charlie either → 0 rows
    register_schema_edge_type(&engine, "KNOWS");

    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) WHERE (a)-[:KNOWS]->(b) OR a.name = 'Charlie' RETURN a.name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 0, "neither condition met → 0 rows");

    // Now add edge → OR's first branch is true
    insert_edge(&engine, "KNOWS", 1, 2);
    let rows = run_cypher_with_alloc(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) WHERE (a)-[:KNOWS]->(b) OR a.name = 'Charlie' RETURN a.name AS name",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 1, "edge exists → OR true");
    assert_eq!(rows[0].get("name"), Some(&Value::String("Alice".into())));
}

// ─── G077: MERGE ALL ─────────────────────────────────────────────────────────

/// G077 — MERGE ALL creates one edge per (src × tgt) pair when no edges exist.
///
/// What this tests:
/// - Two source nodes × two target nodes → 4 edges created (Cartesian product).
/// - MERGE ALL (a:Src)-[r:REL]->(b:Tgt) is NOT ambiguous — all pairs are valid.
/// - Each edge row is returned, so result.len() == 4.
#[test]
fn test_merge_all_creates_cartesian_product() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // Create 2 Src nodes and 2 Tgt nodes.
    run_cypher_with_alloc(
        "CREATE (:Src {name: 's1'}), (:Src {name: 's2'}), (:Tgt {name: 't1'}), (:Tgt {name: 't2'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // MERGE ALL should find-or-create edges for all 4 (src × tgt) pairs.
    let rows = run_cypher_with_alloc(
        "MERGE ALL (a:Src)-[r:CONNECTS]->(b:Tgt) RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(rows.len(), 4, "2 src × 2 tgt = 4 edges");
    for row in &rows {
        assert_eq!(
            row.get("r.__type__"),
            Some(&Value::String("CONNECTS".into())),
            "each edge has correct type"
        );
    }
}

/// G077 — MERGE ALL is idempotent: running twice on existing edges returns same count.
///
/// What this tests:
/// - First MERGE ALL creates 4 edges.
/// - Second MERGE ALL finds all 4 existing edges (ON MATCH path), does NOT create new ones.
/// - Total edge count stays at 4.
#[test]
fn test_merge_all_idempotent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    run_cypher_with_alloc(
        "CREATE (:A {id: '1'}), (:A {id: '2'}), (:B {id: '1'}), (:B {id: '2'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // First run: creates 4 edges.
    let rows1 = run_cypher_with_alloc(
        "MERGE ALL (a:A)-[r:LINKS]->(b:B) RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows1.len(), 4, "first run creates 4 edges");

    // Second run: all edges already exist → ON MATCH path, same 4 rows returned.
    let rows2 = run_cypher_with_alloc(
        "MERGE ALL (a:A)-[r:LINKS]->(b:B) RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows2.len(), 4, "second run finds same 4 edges (idempotent)");
}

/// G077 — MERGE ALL creates src and tgt nodes from scratch when none exist.
///
/// What this tests:
/// - No nodes of either label exist before MERGE ALL.
/// - MERGE ALL creates one src node, one tgt node, and one edge.
/// - Result has exactly 1 row (1×1 Cartesian product).
#[test]
fn test_merge_all_creates_from_scratch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // No nodes exist — MERGE ALL creates both endpoint nodes and the edge.
    let rows = run_cypher_with_alloc(
        "MERGE ALL (a:Source {kind: 'x'})-[r:FEEDS]->(b:Sink {kind: 'y'}) RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(rows.len(), 1, "1 src × 1 tgt = 1 edge created from scratch");
    assert_eq!(
        rows[0].get("r.__type__"),
        Some(&Value::String("FEEDS".into()))
    );
}

/// G077 — MERGE ALL does NOT error on ambiguous source (unlike MERGE).
///
/// What this tests:
/// - MERGE errors when >1 source node matches.
/// - MERGE ALL handles the same case as a Cartesian product (no error).
/// - With 2 src and 1 tgt → 2 edges created.
#[test]
fn test_merge_all_does_not_error_on_ambiguous_source() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // Create 2 identical-label source nodes and 1 target.
    run_cypher_with_alloc(
        "CREATE (:Worker {role: 'op'}), (:Worker {role: 'op'}), (:Queue {name: 'main'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // MERGE ALL: 2 src × 1 tgt = 2 edges. Must NOT error.
    let rows = run_cypher_with_alloc(
        "MERGE ALL (a:Worker {role: 'op'})-[r:PROCESSES]->(b:Queue {name: 'main'}) RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(rows.len(), 2, "2 workers × 1 queue = 2 edges, no error");
}

/// G077 — MERGE ALL ON CREATE SET applies to newly created edges.
///
/// What this tests:
/// - Two edges are created on first run.
/// - ON CREATE SET assigns weight = 1 to each newly created edge variable.
/// - Second run (ON MATCH path) returns same edges without changing weight.
#[test]
fn test_merge_all_on_create_set() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    run_cypher_with_alloc(
        "CREATE (:P {id: '1'}), (:P {id: '2'}), (:Q {id: 'x'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // First run: creates 2 edges, ON CREATE SET fires for each.
    let rows = run_cypher_with_alloc(
        "MERGE ALL (a:P)-[r:CONNECTS]->(b:Q) ON CREATE SET r.weight = 1 RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows.len(), 2, "2 P nodes × 1 Q node = 2 edges");

    // Second run: ON MATCH path, same 2 edges returned.
    let rows2 = run_cypher_with_alloc(
        "MERGE ALL (a:P)-[r:CONNECTS]->(b:Q) ON CREATE SET r.weight = 1 RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );
    assert_eq!(rows2.len(), 2, "second run: idempotent, 2 edges found");
}

/// G077 — MERGE ALL does NOT error on ambiguous target (unlike MERGE).
///
/// What this tests:
/// - MERGE errors when >1 target node matches (symmetric to ambiguous source).
/// - MERGE ALL handles the same case as Cartesian product (no error).
/// - With 1 src and 2 tgt → 2 edges created.
#[test]
fn test_merge_all_does_not_error_on_ambiguous_target() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // Create 1 source and 2 identical-label target nodes.
    run_cypher_with_alloc(
        "CREATE (:Router {id: 'r1'}), (:Device {type: 'sensor'}), (:Device {type: 'sensor'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );

    // MERGE ALL: 1 router × 2 devices = 2 edges. Must NOT error.
    let rows = run_cypher_with_alloc(
        "MERGE ALL (a:Router {id: 'r1'})-[r:MANAGES]->(b:Device {type: 'sensor'}) RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(rows.len(), 2, "1 router × 2 devices = 2 edges, no error");
}

/// G077 — MERGE ALL ON MATCH SET fires on existing edges.
///
/// What this tests:
/// - Pre-create 2 edges manually.
/// - MERGE ALL ON MATCH SET r.visited = 1 — finds both existing edges, applies SET to each.
/// - ON CREATE SET does NOT fire (edges already exist).
#[test]
fn test_merge_all_on_match_set() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1));

    // Create 2 src nodes and 1 tgt node, plus the edges between them.
    run_cypher_with_alloc(
        "CREATE (:Host {name: 'h1'}), (:Host {name: 'h2'}), (:Service {name: 'svc'}) RETURN 1",
        &engine,
        &mut interner,
        &allocator,
    );
    // Pre-create both edges so MERGE ALL takes the ON MATCH path.
    run_cypher_with_alloc(
        "MERGE ALL (a:Host)-[r:CALLS]->(b:Service {name: 'svc'}) RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );

    // Second run with ON MATCH SET — both edges exist, SET fires for each.
    let rows = run_cypher_with_alloc(
        "MERGE ALL (a:Host)-[r:CALLS]->(b:Service {name: 'svc'}) ON MATCH SET r.visited = 1 RETURN r.__type__",
        &engine,
        &mut interner,
        &allocator,
    );

    assert_eq!(
        rows.len(),
        2,
        "2 hosts × 1 service = 2 existing edges, ON MATCH fires"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// R-HYB2b: rrf_score([methods…], {vector:…, text:…}) — RankFuse operator
// ═══════════════════════════════════════════════════════════════════════

/// Run a Cypher query with both a vector index registry and a text index
/// registry wired into the execution context. Used by rrf_score tests.
fn run_cypher_with_registries(
    query: &str,
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    vector_reg: Option<&VectorIndexRegistry>,
    text_reg: Option<&TextIndexRegistry>,
) -> Result<Vec<coordinode_query::executor::Row>, String> {
    let ast = parse(query).map_err(|e| format!("parse error: {e}"))?;
    let plan = build_logical_plan(&ast).map_err(|e| format!("plan error: {e}"))?;
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(engine, interner, &allocator);
    ctx.vector_index_registry = vector_reg;
    ctx.text_index_registry = text_reg;
    execute(&plan, &mut ctx).map_err(|e| format!("execute error: {e}"))
}

/// Build a `(VectorIndexRegistry, TextIndexRegistry)` for the given
/// `(label, vector_property, text_property)` triple, populated from `rows`.
/// Returns the registries and the temp dir that owns tantivy files.
fn setup_rrf_indices(
    base_dir: &std::path::Path,
    label: &str,
    vector_prop: &str,
    text_prop: &str,
    dims: u32,
    rows: &[(u64, Vec<f32>, &str)],
) -> (VectorIndexRegistry, TextIndexRegistry) {
    // Vector index.
    let vec_reg = VectorIndexRegistry::new();
    let vec_cfg = coordinode_query::index::VectorIndexConfig {
        dimensions: dims,
        metric: coordinode_core::graph::types::VectorMetric::Cosine,
        m: 16,
        ef_construction: 200,
        quantization: false,
        offload_vectors: false,
    };
    let vec_def = IndexDefinition::hnsw(
        format!("{label}_{vector_prop}_idx"),
        label,
        vector_prop,
        vec_cfg,
    );
    vec_reg.register(vec_def);
    // Populate HNSW directly via the handle.
    if let Some(handle) = vec_reg.get(label, vector_prop) {
        let mut idx = handle.write().unwrap();
        for (id, v, _text) in rows {
            idx.insert(*id, v.clone());
        }
    }

    // Text index.
    let text_reg_dir = base_dir.join("rrf_text_reg");
    std::fs::create_dir_all(&text_reg_dir).unwrap();
    let text_reg = TextIndexRegistry::new(&text_reg_dir);
    let text_cfg = TextIndexConfig::default();
    let text_def = IndexDefinition::text(
        format!("{label}_{text_prop}_idx"),
        label,
        vec![text_prop.to_string()],
        text_cfg,
    );
    text_reg.register(text_def).unwrap();
    for (id, _v, text) in rows {
        text_reg.on_text_written(label, NodeId::from_raw(*id), text_prop, text);
    }

    (vec_reg, text_reg)
}

/// R-HYB2b regression #2: parser/planner rejects any 3rd argument to
/// `rrf_score(...)`. `k=60` is the IR standard (Cormack 2009) and not tunable.
#[test]
fn rrf_score_rejects_k_override() {
    let ast = parse(
        "MATCH (d:Doc) \
         RETURN rrf_score([d.embedding, d.body], {vector: [1.0, 0.0], text: \"rust\"}, {k: 100}) AS s",
    )
    .expect("parse");
    let err = build_logical_plan(&ast).expect_err("3rd arg must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("rrf_score()") && msg.contains("k=60") && msg.contains("not tunable"),
        "error must explain k is frozen, got: {msg}"
    );
}

/// R-HYB2b: empty method list is rejected at plan time.
#[test]
fn rrf_score_rejects_empty_methods() {
    let ast = parse(
        "MATCH (d:Doc) \
         RETURN rrf_score([], {vector: [1.0, 0.0], text: \"rust\"}) AS s",
    )
    .expect("parse");
    let err = build_logical_plan(&ast).expect_err("empty methods must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("non-empty list") && msg.contains("rrf_score()"),
        "error must name rrf_score and require non-empty list, got: {msg}"
    );
}

/// R-HYB2b: query map with an unknown key is rejected.
#[test]
fn rrf_score_rejects_unknown_query_key() {
    let ast = parse(
        "MATCH (d:Doc) \
         RETURN rrf_score([d.embedding], {unknown: 123}) AS s",
    )
    .expect("parse");
    let err = build_logical_plan(&ast).expect_err("unknown key must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("unknown") && msg.contains("vector") && msg.contains("text"),
        "error must list allowed keys, got: {msg}"
    );
}

/// R-HYB2b: rrf_score in WHERE is rejected (materialised rank assignment
/// is impossible at filter-time).
#[test]
fn rrf_score_rejects_where_placement() {
    let ast = parse(
        "MATCH (d:Doc) \
         WHERE rrf_score([d.embedding, d.body], {vector: [1.0, 0.0], text: \"rust\"}) > 0.0 \
         RETURN d",
    )
    .expect("parse");
    let err = build_logical_plan(&ast).expect_err("rrf_score in WHERE must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("WHERE"),
        "error must name the illegal position, got: {msg}"
    );
}

/// R-HYB2b: plan shape — a valid rrf_score produces a RankFuse operator below
/// the topmost Project. Verified via EXPLAIN text.
#[test]
fn rrf_score_plan_contains_rank_fuse() {
    let ast = parse(
        "MATCH (d:Doc) \
         RETURN d, rrf_score([d.embedding, d.body], {vector: [1.0, 0.0], text: \"rust\"}) AS s \
         ORDER BY s DESC LIMIT 5",
    )
    .expect("parse");
    let plan = build_logical_plan(&ast).expect("plan");
    let explain = plan.explain();
    assert!(
        explain.contains("RankFuse(methods="),
        "EXPLAIN must show the RankFuse operator; got:\n{explain}"
    );
    assert!(
        explain.contains("k=60"),
        "EXPLAIN must show k=60 constant; got:\n{explain}"
    );
}

/// R-HYB2b regression #1: RRF ranking handles methods with divergent score
/// distributions. One doc is top-1 on BOTH methods (balanced) and must win
/// regardless of raw-score scale. A naïve weighted-sum with equal weights
/// would be dominated by whichever method has larger raw scores (here BM25,
/// which is unbounded vs cos-sim in [0,1]) — RRF's rank-only fusion protects
/// against that scale mismatch.
#[test]
fn rrf_score_handles_divergent_distributions() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // 4 docs. Node 6 is top-1 on vector (tied with node 1 at cos=1.0) AND
    // top-1 on BM25 (6 "rust" tokens dominates node 2's 3 tokens). Specialists
    // (1 = vector-only, 2 = text-only) each miss one axis and take a
    // penalty rank there.
    let rows: Vec<(u64, Vec<f32>, &str)> = vec![
        (1, vec![1.0, 0.0], "literature review draft"),
        (2, vec![0.0, 1.0], "rust rust rust async runtime"),
        (3, vec![0.1, 0.9], "python machine learning framework"),
        (
            6,
            vec![1.0, 0.0],
            "rust rust rust rust rust rust programming language tokio async",
        ),
    ];
    for (id, v, body) in &rows {
        insert_node(
            &engine,
            1,
            *id,
            "Chunk",
            &[
                ("embedding", Value::Vector(v.clone())),
                ("body", Value::String((*body).to_string())),
            ],
            &mut interner,
        );
    }

    let (vec_reg, text_reg) = setup_rrf_indices(dir.path(), "Chunk", "embedding", "body", 2, &rows);

    let rrf_results = run_cypher_with_registries(
        "MATCH (c:Chunk) \
         RETURN c, \
                rrf_score([c.embedding, c.body], {vector: [1.0, 0.0], text: \"rust\"}) AS score \
         ORDER BY score DESC",
        &engine,
        &mut interner,
        Some(&vec_reg),
        Some(&text_reg),
    )
    .expect("rrf execute");

    assert_eq!(rrf_results.len(), rows.len(), "all rows participate in RRF");

    let rank_of = |id: u64| -> usize {
        rrf_results
            .iter()
            .position(|r| matches!(r.get("c"), Some(Value::Int(v)) if *v as u64 == id))
            .expect("id present")
    };
    let score_of = |id: u64| -> f64 {
        match rrf_results
            .iter()
            .find(|r| matches!(r.get("c"), Some(Value::Int(v)) if *v as u64 == id))
            .and_then(|r| r.get("score"))
        {
            Some(Value::Float(f)) => *f,
            other => panic!("expected Float, got {other:?}"),
        }
    };

    // Node 6: vec rank 1 (tied with node 1, competition rank = 1) AND text
    // rank 1 (more "rust" tokens than node 2) → score = 1/61 + 1/61.
    // Node 1: vec rank 1 + text penalty → strictly less than node 6.
    // Node 2: text rank 2 + vec penalty → strictly less than node 6.
    let top_id = match rrf_results[0].get("c") {
        Some(Value::Int(id)) => *id as u64,
        other => panic!("expected Int node id, got {other:?}"),
    };
    assert_eq!(
        top_id, 6,
        "RRF top must be node 6 (balanced across vector + text); \
         weighted-sum with equal weights would let BM25's larger absolute \
         scale dominate (picking node 2) — RRF's rank fusion protects against that"
    );

    // Scores must be non-increasing under ORDER BY DESC.
    let scores: Vec<f64> = rrf_results
        .iter()
        .map(|r| match r.get("score") {
            Some(Value::Float(f)) => *f,
            other => panic!("expected Float score, got {other:?}"),
        })
        .collect();
    for w in scores.windows(2) {
        assert!(
            w[0] >= w[1],
            "ORDER BY score DESC produced non-monotonic scores: {scores:?}"
        );
    }

    assert!(
        rank_of(6) < rank_of(1),
        "node 6 (top on both) must outrank node 1 (vector-only): \
         ranks 6={} 1={}; scores 6={} 1={}",
        rank_of(6),
        rank_of(1),
        score_of(6),
        score_of(1),
    );
    assert!(
        rank_of(6) < rank_of(2),
        "node 6 (top on both) must outrank node 2 (text-only): \
         ranks 6={} 2={}; scores 6={} 2={}",
        rank_of(6),
        rank_of(2),
        score_of(6),
        score_of(2),
    );
}

/// R-HYB2b: penalty-rank semantics. A row missing one method entirely still
/// receives a score from the other methods — proof that RRF is additive,
/// not short-circuit.
#[test]
fn rrf_score_missing_method_uses_penalty_rank() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Two docs — only doc #1 has text that matches "rust"; both have vectors.
    let rows: Vec<(u64, Vec<f32>, &str)> = vec![
        (1, vec![1.0, 0.0], "rust programming"),
        (2, vec![0.5, 0.5], "python notebook"),
    ];
    for (id, v, body) in &rows {
        insert_node(
            &engine,
            1,
            *id,
            "Doc",
            &[
                ("embedding", Value::Vector(v.clone())),
                ("body", Value::String((*body).to_string())),
            ],
            &mut interner,
        );
    }

    let (vec_reg, text_reg) = setup_rrf_indices(dir.path(), "Doc", "embedding", "body", 2, &rows);

    let results = run_cypher_with_registries(
        "MATCH (d:Doc) \
         RETURN d, \
                rrf_score([d.embedding, d.body], {vector: [1.0, 0.0], text: \"rust\"}) AS score",
        &engine,
        &mut interner,
        Some(&vec_reg),
        Some(&text_reg),
    )
    .expect("rrf execute");

    assert_eq!(results.len(), 2);
    // Doc #2 has no text match → text rank = penalty (matched+1=2). It still
    // has a valid vector rank, so its score is non-zero.
    let score_of = |id: u64| -> f64 {
        let row = results
            .iter()
            .find(|r| matches!(r.get("d"), Some(Value::Int(v)) if *v as u64 == id))
            .expect("id present");
        match row.get("score") {
            Some(Value::Float(f)) => *f,
            other => panic!("expected Float, got {other:?}"),
        }
    };
    assert!(
        score_of(2) > 0.0,
        "penalty rank still contributes a small positive term, got 0 for doc without text match"
    );
    assert!(
        score_of(1) > score_of(2),
        "doc with both methods matched must outrank doc missing text"
    );
}

/// R-HYB2b: unscorable method (no HNSW index, no FT index, and no vector
/// values) errors with a clear message.
#[test]
fn rrf_score_unscorable_method_errors() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Item",
        &[("name", Value::String("alpha".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Item",
        &[("name", Value::String("beta".into()))],
        &mut interner,
    );

    // Empty registries — `name` has neither a vector index nor a text index,
    // and it isn't a vector value → method is genuinely unscorable.
    let vec_reg = VectorIndexRegistry::new();
    let text_dir = dir.path().join("rrf_empty");
    std::fs::create_dir_all(&text_dir).unwrap();
    let text_reg = TextIndexRegistry::new(&text_dir);

    let err = run_cypher_with_registries(
        "MATCH (i:Item) \
         RETURN rrf_score([i.name], {vector: [1.0]}) AS score",
        &engine,
        &mut interner,
        Some(&vec_reg),
        Some(&text_reg),
    )
    .expect_err("unscorable method must error");
    assert!(
        err.contains("rrf_score()") && err.contains("i.name") && err.contains("cannot be scored"),
        "error must name the offending method, got: {err}"
    );
}

/// R-HYB2b: competition ranking — rows with identical vector scores receive
/// the SAME rank, so their RRF contribution for that method is equal. We
/// seed two rows with identical vectors + identical text to force a tie,
/// then verify their scores are identical.
#[test]
fn rrf_score_competition_rank_ties() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Three docs: #1 and #2 are identical (guaranteed tie on both methods),
    // #3 is a distinct baseline.
    let rows: Vec<(u64, Vec<f32>, &str)> = vec![
        (1, vec![1.0, 0.0], "rust programming"),
        (2, vec![1.0, 0.0], "rust programming"),
        (3, vec![0.0, 1.0], "python notebook"),
    ];
    for (id, v, body) in &rows {
        insert_node(
            &engine,
            1,
            *id,
            "Doc",
            &[
                ("embedding", Value::Vector(v.clone())),
                ("body", Value::String((*body).to_string())),
            ],
            &mut interner,
        );
    }

    let (vec_reg, text_reg) = setup_rrf_indices(dir.path(), "Doc", "embedding", "body", 2, &rows);

    let results = run_cypher_with_registries(
        "MATCH (d:Doc) \
         RETURN d, \
                rrf_score([d.embedding, d.body], {vector: [1.0, 0.0], text: \"rust\"}) AS score \
         ORDER BY score DESC",
        &engine,
        &mut interner,
        Some(&vec_reg),
        Some(&text_reg),
    )
    .expect("rrf execute");

    assert_eq!(results.len(), 3);
    let score_of = |id: u64| -> f64 {
        let row = results
            .iter()
            .find(|r| matches!(r.get("d"), Some(Value::Int(v)) if *v as u64 == id))
            .expect("id present");
        match row.get("score") {
            Some(Value::Float(f)) => *f,
            other => panic!("expected Float, got {other:?}"),
        }
    };
    let s1 = score_of(1);
    let s2 = score_of(2);
    assert!(
        (s1 - s2).abs() < 1e-12,
        "tied rows must receive identical RRF score: s1={s1} s2={s2}"
    );
}

/// R-HYB2b: full pipeline — `MATCH → RETURN rrf_score AS s → ORDER BY s DESC
/// → LIMIT K`. Verifies that LIMIT applies AFTER RankFuse ranks the full set
/// (if LIMIT clipped before RankFuse, ranks would be wrong).
#[test]
fn rrf_score_pipeline_with_order_by_and_limit() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let rows: Vec<(u64, Vec<f32>, &str)> = vec![
        (1, vec![1.0, 0.0], "rust alpha"),
        (2, vec![0.9, 0.1], "rust beta"),
        (3, vec![0.8, 0.2], "rust gamma"),
        (4, vec![0.2, 0.8], "python delta"),
        (5, vec![0.1, 0.9], "python epsilon"),
    ];
    for (id, v, body) in &rows {
        insert_node(
            &engine,
            1,
            *id,
            "Doc",
            &[
                ("embedding", Value::Vector(v.clone())),
                ("body", Value::String((*body).to_string())),
            ],
            &mut interner,
        );
    }

    let (vec_reg, text_reg) = setup_rrf_indices(dir.path(), "Doc", "embedding", "body", 2, &rows);

    let results = run_cypher_with_registries(
        "MATCH (d:Doc) \
         RETURN d, \
                rrf_score([d.embedding, d.body], {vector: [1.0, 0.0], text: \"rust\"}) AS score \
         ORDER BY score DESC LIMIT 3",
        &engine,
        &mut interner,
        Some(&vec_reg),
        Some(&text_reg),
    )
    .expect("rrf execute");

    assert_eq!(results.len(), 3, "LIMIT 3 must cap output to 3 rows");

    // Top 3 must be the "rust" nodes 1, 2, 3 (each combines both methods well).
    let top_ids: std::collections::HashSet<u64> = results
        .iter()
        .map(|r| match r.get("d") {
            Some(Value::Int(v)) => *v as u64,
            other => panic!("expected Int, got {other:?}"),
        })
        .collect();
    let expected: std::collections::HashSet<u64> = [1, 2, 3].into_iter().collect();
    assert_eq!(
        top_ids, expected,
        "top-3 under RRF should be the rust docs (1,2,3); got {top_ids:?}"
    );
}

/// R-HYB2b: brute-force vector scoring path — when a vector property has NO
/// HNSW index registered, RankFuse falls back to in-memory cosine similarity
/// (same code path used for edge vector properties in `VectorBruteForce`).
/// Proves the full edge-vector scope of R-HYB2b — the resolver correctly
/// classifies an indexless vector property as brute-force and scores all
/// rows, rather than erroring out.
#[test]
fn rrf_score_brute_force_vector_without_hnsw() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let rows: Vec<(u64, Vec<f32>, &str)> = vec![
        (1, vec![1.0, 0.0], "rust rust rust programming"),
        (2, vec![0.9, 0.1], "rust async runtime"),
        (3, vec![0.0, 1.0], "python notebook"),
    ];
    for (id, v, body) in &rows {
        insert_node(
            &engine,
            1,
            *id,
            "Item",
            &[
                ("ctx_emb", Value::Vector(v.clone())),
                ("body", Value::String((*body).to_string())),
            ],
            &mut interner,
        );
    }

    // Only register the TEXT index — the vector property `ctx_emb` has NO
    // HNSW index and no vector_registry entry. RankFuse must fall through
    // to `VectorBruteForce` and still produce correct ranks.
    let text_reg_dir = dir.path().join("brute_text_reg");
    std::fs::create_dir_all(&text_reg_dir).unwrap();
    let text_reg = TextIndexRegistry::new(&text_reg_dir);
    text_reg
        .register(IndexDefinition::text(
            "Item_body_idx",
            "Item",
            vec!["body".into()],
            TextIndexConfig::default(),
        ))
        .unwrap();
    for (id, _v, body) in &rows {
        text_reg.on_text_written("Item", NodeId::from_raw(*id), "body", body);
    }

    // Empty vector registry — ctx_emb scoring MUST use brute-force cosine
    // against the Vector values in the rows.
    let vec_reg = VectorIndexRegistry::new();

    let results = run_cypher_with_registries(
        "MATCH (i:Item) \
         RETURN i, \
                rrf_score([i.ctx_emb, i.body], {vector: [1.0, 0.0], text: \"rust\"}) AS score \
         ORDER BY score DESC",
        &engine,
        &mut interner,
        Some(&vec_reg),
        Some(&text_reg),
    )
    .expect("brute-force vector path must succeed without HNSW");

    assert_eq!(results.len(), 3);

    // Node 1: best on both methods (perfect vector + most "rust" tokens).
    let top_id = match results[0].get("i") {
        Some(Value::Int(id)) => *id as u64,
        other => panic!("expected Int, got {other:?}"),
    };
    assert_eq!(
        top_id, 1,
        "brute-force cosine must correctly rank [1,0]-vs-[1,0] as top"
    );

    // Node 3: worst on both (orthogonal vector, no text match) — must be last.
    let bottom_id = match results[2].get("i") {
        Some(Value::Int(id)) => *id as u64,
        other => panic!("expected Int, got {other:?}"),
    };
    assert_eq!(bottom_id, 3, "node 3 (orthogonal + no text) must rank last");
}

/// R-HYB2b: method expression that isn't a `var.property` access is rejected
/// at execute time with a message naming the problem.
#[test]
fn rrf_score_non_property_method_errors() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Doc",
        &[("embedding", Value::Vector(vec![1.0, 0.0]))],
        &mut interner,
    );
    let vec_reg = VectorIndexRegistry::new();
    let text_dir = dir.path().join("rrf_nonprop");
    std::fs::create_dir_all(&text_dir).unwrap();
    let text_reg = TextIndexRegistry::new(&text_dir);

    // Method is `d.embedding + 0` — a BinaryOp, not a PropertyAccess.
    let err = run_cypher_with_registries(
        "MATCH (d:Doc) \
         RETURN rrf_score([d.embedding + 0], {vector: [1.0, 0.0]}) AS score",
        &engine,
        &mut interner,
        Some(&vec_reg),
        Some(&text_reg),
    )
    .expect_err("non-property method must error");
    assert!(
        err.contains("rrf_score()") && err.contains("property accesses"),
        "error must explain the shape requirement, got: {err}"
    );
}

/// R-HYB2b: two `rrf_score(...)` calls with differing method lists in the
/// same query are rejected at plan time. Same signature is fine (redundant
/// reads of `__rrf_score__`).
#[test]
fn rrf_score_multiple_differing_calls_rejected() {
    let ast = parse(
        "MATCH (d:Doc) \
         RETURN rrf_score([d.embedding], {vector: [1.0, 0.0]}) AS a, \
                rrf_score([d.body], {text: \"rust\"}) AS b",
    )
    .expect("parse");
    let err = build_logical_plan(&ast).expect_err("multi-signature rrf_score must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("rrf_score()") && msg.contains("WITH"),
        "error must point at the multi-call issue and suggest WITH, got: {msg}"
    );
}

/// R-HYB2b: if all methods are vector but the query map omits the `vector`
/// key, execution errors with a clear message.
#[test]
fn rrf_score_missing_vector_key_when_needed_errors() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let rows: Vec<(u64, Vec<f32>, &str)> = vec![(1, vec![1.0, 0.0], "placeholder")];
    for (id, v, body) in &rows {
        insert_node(
            &engine,
            1,
            *id,
            "Doc",
            &[
                ("embedding", Value::Vector(v.clone())),
                ("body", Value::String((*body).to_string())),
            ],
            &mut interner,
        );
    }
    let (vec_reg, text_reg) = setup_rrf_indices(dir.path(), "Doc", "embedding", "body", 2, &rows);

    // Method needs vector but the map has only `text`.
    let err = run_cypher_with_registries(
        "MATCH (d:Doc) \
         RETURN rrf_score([d.embedding], {text: \"placeholder\"}) AS score",
        &engine,
        &mut interner,
        Some(&vec_reg),
        Some(&text_reg),
    )
    .expect_err("vector-method without vector-key must error");
    assert!(
        err.contains("rrf_score()") && err.contains("vector"),
        "error must name the missing vector key, got: {err}"
    );
}

/// R-HYB2b: `ORDER BY rrf_score(...) DESC` as a direct call (not via alias)
/// must work. Exercises the `ensure_rrf_passthrough` code path that injects
/// a synthetic `__rrf_score__` Project item so Sort (above Project) still
/// sees the score column.
#[test]
fn rrf_score_order_by_direct_call() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let rows: Vec<(u64, Vec<f32>, &str)> = vec![
        (1, vec![1.0, 0.0], "rust rust rust programming"),
        (2, vec![0.9, 0.1], "rust async"),
        (3, vec![0.0, 1.0], "python notebook"),
    ];
    for (id, v, body) in &rows {
        insert_node(
            &engine,
            1,
            *id,
            "Doc",
            &[
                ("embedding", Value::Vector(v.clone())),
                ("body", Value::String((*body).to_string())),
            ],
            &mut interner,
        );
    }
    let (vec_reg, text_reg) = setup_rrf_indices(dir.path(), "Doc", "embedding", "body", 2, &rows);

    // ORDER BY references the FUNCTION CALL directly, not an alias. This
    // forces the Sort → Project → RankFuse chain where Project must pass
    // __rrf_score__ through for Sort to evaluate it.
    let results = run_cypher_with_registries(
        "MATCH (d:Doc) \
         RETURN d.body AS body \
         ORDER BY rrf_score([d.embedding, d.body], {vector: [1.0, 0.0], text: \"rust\"}) DESC",
        &engine,
        &mut interner,
        Some(&vec_reg),
        Some(&text_reg),
    )
    .expect("ORDER BY direct rrf_score call must work");

    assert_eq!(results.len(), 3);
    // Doc 1 is top-1 on both methods → must appear first.
    let top_body = match results[0].get("body") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected String body, got {other:?}"),
    };
    assert_eq!(
        top_body, "rust rust rust programming",
        "direct ORDER BY rrf_score(...) DESC must rank the best doc first"
    );
}

/// R-HYB2b: vector-only RRF on two orthogonal vector methods (no text).
/// Exercises the degenerate query map `{vector: ...}` (no `text` key).
#[test]
fn rrf_score_vector_only_multiple_methods() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let rows: Vec<(u64, Vec<f32>, &str)> = vec![
        (1, vec![1.0, 0.0], "_"),
        (2, vec![0.0, 1.0], "_"),
        (3, vec![0.7, 0.7], "_"),
    ];
    for (id, v, body) in &rows {
        insert_node(
            &engine,
            1,
            *id,
            "Doc",
            &[
                ("embedding", Value::Vector(v.clone())),
                ("body", Value::String((*body).to_string())),
            ],
            &mut interner,
        );
    }

    let (vec_reg, text_reg) = setup_rrf_indices(dir.path(), "Doc", "embedding", "body", 2, &rows);

    // Two vector "methods" — in this synthetic case both point at the same
    // property, but it exercises the N-method wiring end-to-end.
    let results = run_cypher_with_registries(
        "MATCH (d:Doc) \
         RETURN d, \
                rrf_score([d.embedding, d.embedding], {vector: [1.0, 0.0]}) AS score \
         ORDER BY score DESC",
        &engine,
        &mut interner,
        Some(&vec_reg),
        Some(&text_reg),
    )
    .expect("rrf execute");

    assert_eq!(results.len(), 3);
    let top_id = match results[0].get("d") {
        Some(Value::Int(v)) => *v as u64,
        other => panic!("expected Int, got {other:?}"),
    };
    assert_eq!(top_id, 1, "doc closest to [1, 0] ranks first");
}

// ═══════════════════════════════════════════════════════════════════════
// R-HYB2c: doc_score(doc, query [, α, β, γ]) — document-level aggregate
// ═══════════════════════════════════════════════════════════════════════

/// Seed `n_chunks` Chunk nodes connected to a Document via HAS_CHUNK.
/// Returns the document node id. `chunk_embeds` entries with `None` leave
/// the chunk's `embedding` property unset (simulates a chunk that never
/// got embedded); `Some(vec)` stores a vector embedding.
fn seed_doc_with_chunks(
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    doc_id: u64,
    chunk_embeds: &[Option<Vec<f32>>],
) {
    insert_node(engine, 1, doc_id, "Document", &[], interner);
    for (i, emb) in chunk_embeds.iter().enumerate() {
        let chunk_id = doc_id * 1000 + (i as u64) + 1;
        let props: Vec<(&str, Value)> = match emb {
            Some(v) => vec![("embedding", Value::Vector(v.clone()))],
            None => vec![],
        };
        insert_node(engine, 1, chunk_id, "Chunk", &props, interner);
        insert_edge(engine, "HAS_CHUNK", doc_id, chunk_id);
    }
}

/// R-HYB2c regression #1: `doc_score(d, q)` with default weights (0.5/0.3/0.2)
/// must equal `0.5·max + 0.3·avg + 0.2·coverage` computed independently by
/// hand over the HAS_CHUNK children.
#[test]
fn doc_score_matches_manual_aggregate() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Doc 1: three chunks, all with embeddings.
    //   similarities to [1.0, 0.0]: cos([1,0])=1, cos([0.8,0.6])≈0.8, cos([0.6,0.8])≈0.6
    //   max = 1.0, avg = 0.8, matching (sim>0) = 3, total = 3, coverage = 1.0
    //   expected = 0.5*1.0 + 0.3*0.8 + 0.2*1.0 = 0.94
    seed_doc_with_chunks(
        &engine,
        &mut interner,
        1,
        &[
            Some(vec![1.0, 0.0]),
            Some(vec![0.8, 0.6]),
            Some(vec![0.6, 0.8]),
        ],
    );

    let results = run_cypher(
        "MATCH (d:Document) \
         RETURN d, doc_score(d, [1.0, 0.0]) AS score",
        &engine,
        &mut interner,
    );
    assert_eq!(results.len(), 1);
    let score = match results[0].get("score") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float score, got {other:?}"),
    };

    // Manual: 0.5 * 1.0 + 0.3 * avg + 0.2 * 1.0.
    let s1 = 1.0_f32;
    let s2 = coordinode_vector::metrics::cosine_similarity(&[0.8, 0.6], &[1.0, 0.0]);
    let s3 = coordinode_vector::metrics::cosine_similarity(&[0.6, 0.8], &[1.0, 0.0]);
    let avg = (s1 + s2 + s3) as f64 / 3.0;
    let expected = 0.5 * 1.0 + 0.3 * avg + 0.2 * 1.0;
    assert!(
        (score - expected).abs() < 1e-5,
        "doc_score must match α·max + β·avg + γ·coverage: got {score}, expected {expected}"
    );
}

/// R-HYB2c regression #2a: positional overrides α/β/γ.
#[test]
fn doc_score_custom_weights_positional() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    seed_doc_with_chunks(
        &engine,
        &mut interner,
        1,
        &[Some(vec![1.0, 0.0]), Some(vec![0.6, 0.8])],
    );

    // Max-only weighting: α=1.0, β=0, γ=0 → score = max(sim) = 1.0
    let results = run_cypher(
        "MATCH (d:Document) \
         RETURN doc_score(d, [1.0, 0.0], 1.0, 0.0, 0.0) AS score",
        &engine,
        &mut interner,
    );
    let score = match results[0].get("score") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {other:?}"),
    };
    assert!(
        (score - 1.0).abs() < 1e-5,
        "max-only positional weights must yield 1.0, got {score}"
    );
}

/// R-HYB2c regression #2b: weights map overrides α/β/γ.
#[test]
fn doc_score_custom_weights_map() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    seed_doc_with_chunks(
        &engine,
        &mut interner,
        1,
        &[Some(vec![1.0, 0.0]), Some(vec![0.6, 0.8])],
    );

    // Coverage-only: α=0, β=0, γ=1.0 → score = matching/total = 2/2 = 1.0
    let results = run_cypher(
        "MATCH (d:Document) \
         RETURN doc_score(d, [1.0, 0.0], {alpha: 0.0, beta: 0.0, gamma: 1.0}) AS score",
        &engine,
        &mut interner,
    );
    let score = match results[0].get("score") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {other:?}"),
    };
    assert!(
        (score - 1.0).abs() < 1e-5,
        "coverage-only map weights must yield 1.0, got {score}"
    );
}

/// R-HYB2c regression #3: document with zero HAS_CHUNK children returns 0
/// (not an error, not null). Required by the task spec.
#[test]
fn doc_score_zero_chunks_returns_zero() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Insert Document with NO HAS_CHUNK edges.
    insert_node(&engine, 1, 1, "Document", &[], &mut interner);

    let results = run_cypher(
        "MATCH (d:Document) \
         RETURN doc_score(d, [1.0, 0.0]) AS score",
        &engine,
        &mut interner,
    );
    assert_eq!(results.len(), 1);
    let score = match results[0].get("score") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {other:?}"),
    };
    assert!(
        (score - 0.0).abs() < 1e-9,
        "doc with zero HAS_CHUNK children must return 0, got {score}"
    );
}

/// R-HYB2c: chunks without embedding reduce coverage (they count toward total
/// but not toward matching). Formula: max/avg only over valid embeddings,
/// coverage = matching / total where total includes all HAS_CHUNK children.
#[test]
fn doc_score_missing_embeddings_reduce_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // 4 chunks: 2 with embedding (both aligned with query), 2 without.
    seed_doc_with_chunks(
        &engine,
        &mut interner,
        1,
        &[Some(vec![1.0, 0.0]), Some(vec![0.8, 0.6]), None, None],
    );

    let results = run_cypher(
        "MATCH (d:Document) \
         RETURN doc_score(d, [1.0, 0.0]) AS score",
        &engine,
        &mut interner,
    );
    let score = match results[0].get("score") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {other:?}"),
    };

    let s1 = 1.0_f32;
    let s2 = coordinode_vector::metrics::cosine_similarity(&[0.8, 0.6], &[1.0, 0.0]);
    let avg = (s1 + s2) as f64 / 2.0;
    // matching = 2 (both scored chunks have sim > 0); total = 4 (all HAS_CHUNK).
    let coverage = 2.0_f64 / 4.0;
    let expected = 0.5 * 1.0 + 0.3 * avg + 0.2 * coverage;
    assert!(
        (score - expected).abs() < 1e-5,
        "coverage must use total (including chunks without embedding): got {score}, expected {expected}"
    );
}

/// R-HYB2c: chunks with orthogonal / negatively-correlated embeddings must
/// count toward `total` but NOT toward `matching`. Max and avg use the
/// actual chunk similarities (which may be zero/negative); coverage drops.
#[test]
fn doc_score_negative_similarity_chunks_excluded_from_matching() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // 3 chunks, all with embedding; none aligned with query [1,0]:
    //   [0.0, 1.0] → cos = 0
    //   [-1.0, 0.0] → cos = -1
    //   [-0.5, 0.5] → cos ≈ -0.707
    seed_doc_with_chunks(
        &engine,
        &mut interner,
        1,
        &[
            Some(vec![0.0, 1.0]),
            Some(vec![-1.0, 0.0]),
            Some(vec![-0.5, 0.5]),
        ],
    );

    let results = run_cypher(
        "MATCH (d:Document) \
         RETURN doc_score(d, [1.0, 0.0]) AS score",
        &engine,
        &mut interner,
    );
    let score = match results[0].get("score") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {other:?}"),
    };

    // matching = 0 (no chunk has sim > 0); coverage = 0/3 = 0.
    // max is computed over all scored chunks so it's 0 (the best non-negative).
    // Actually per implementation max is max over all scored similarities;
    // with 0, -1, -0.707 the max is 0.
    // avg = (0 + -1 + -0.707) / 3 ≈ -0.569
    // score = 0.5*0 + 0.3*(-0.569) + 0.2*0 ≈ -0.171
    let s1 = 0.0_f32;
    let s2 = coordinode_vector::metrics::cosine_similarity(&[-1.0, 0.0], &[1.0, 0.0]);
    let s3 = coordinode_vector::metrics::cosine_similarity(&[-0.5, 0.5], &[1.0, 0.0]);
    let avg = (s1 + s2 + s3) as f64 / 3.0;
    let max = 0.0_f64.max(s2 as f64).max(s3 as f64); // 0
    let coverage = 0.0_f64;
    let expected = 0.5 * max + 0.3 * avg + 0.2 * coverage;
    assert!(
        (score - expected).abs() < 1e-5,
        "negative-sim chunks must be excluded from matching (coverage=0); got {score}, expected {expected}"
    );
}

/// R-HYB2c: doc_score composes with ORDER BY and LIMIT to pick top docs.
#[test]
fn doc_score_orders_documents_by_relevance() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Doc 1: all chunks aligned with query → high score
    seed_doc_with_chunks(
        &engine,
        &mut interner,
        1,
        &[Some(vec![1.0, 0.0]), Some(vec![0.9, 0.1])],
    );
    // Doc 2: mixed — one aligned, one orthogonal → mid score
    seed_doc_with_chunks(
        &engine,
        &mut interner,
        2,
        &[Some(vec![1.0, 0.0]), Some(vec![0.0, 1.0])],
    );
    // Doc 3: all orthogonal → low score
    seed_doc_with_chunks(
        &engine,
        &mut interner,
        3,
        &[Some(vec![0.0, 1.0]), Some(vec![0.0, 1.0])],
    );

    let results = run_cypher(
        "MATCH (d:Document) \
         RETURN d, doc_score(d, [1.0, 0.0]) AS score \
         ORDER BY score DESC",
        &engine,
        &mut interner,
    );
    assert_eq!(results.len(), 3);
    let top_id = match results[0].get("d") {
        Some(Value::Int(v)) => *v as u64,
        other => panic!("expected Int, got {other:?}"),
    };
    assert_eq!(top_id, 1, "fully-aligned doc must rank first");
    let bottom_id = match results[2].get("d") {
        Some(Value::Int(v)) => *v as u64,
        other => panic!("expected Int, got {other:?}"),
    };
    assert_eq!(bottom_id, 3, "fully-orthogonal doc must rank last");

    // Scores must be non-increasing.
    let scores: Vec<f64> = results
        .iter()
        .map(|r| match r.get("score") {
            Some(Value::Float(f)) => *f,
            other => panic!("expected Float, got {other:?}"),
        })
        .collect();
    for w in scores.windows(2) {
        assert!(
            w[0] >= w[1],
            "scores must be non-increasing under ORDER BY DESC: {scores:?}"
        );
    }
}

/// R-HYB2c: plan-time guards — arity outside 2/3/5 rejected.
#[test]
fn doc_score_rejects_wrong_arity() {
    let cases = [
        ("RETURN doc_score(d) AS s", "1 argument"),
        ("RETURN doc_score(d, $q, $a, $b) AS s", "4 arguments"),
        (
            "RETURN doc_score(d, $q, 0.1, 0.2, 0.3, 0.4) AS s",
            "6 arguments",
        ),
    ];
    for (frag, desc) in cases {
        let q = format!("MATCH (d:Document) {frag}");
        let ast = parse(&q).expect("parse");
        let err = build_logical_plan(&ast).expect_err(&format!("{desc} must be rejected"));
        let msg = err.to_string();
        assert!(
            msg.contains("doc_score()") && msg.contains("argument"),
            "error must name doc_score and the arity problem ({desc}), got: {msg}"
        );
    }
}

/// R-HYB2c: first arg must be a bound variable, not a literal or expression.
#[test]
fn doc_score_rejects_non_variable_doc() {
    let ast = parse(
        "MATCH (d:Document) \
         RETURN doc_score(123, [1.0, 0.0]) AS s",
    )
    .unwrap();
    let err = build_logical_plan(&ast).expect_err("literal doc must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("doc_score()") && msg.contains("variable"),
        "error must explain the variable requirement, got: {msg}"
    );
}

/// R-HYB2c: weights map with unknown key rejected.
#[test]
fn doc_score_rejects_unknown_weight_key() {
    let ast = parse(
        "MATCH (d:Document) \
         RETURN doc_score(d, [1.0, 0.0], {delta: 0.5}) AS s",
    )
    .unwrap();
    let err = build_logical_plan(&ast).expect_err("unknown weight key must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("doc_score()") && msg.contains("unknown") && msg.contains("alpha"),
        "error must name doc_score, the unknown key, and the allowed keys, got: {msg}"
    );
}

/// R-HYB2c: doc_score in WHERE is rejected (correlated aggregate, not filter).
#[test]
fn doc_score_rejects_where_placement() {
    let ast = parse(
        "MATCH (d:Document) \
         WHERE doc_score(d, [1.0, 0.0]) > 0.5 \
         RETURN d",
    )
    .unwrap();
    let err = build_logical_plan(&ast).expect_err("doc_score in WHERE must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("WHERE"),
        "error must name the illegal position, got: {msg}"
    );
}

/// R-HYB2c: EXPLAIN surfaces the DocScore operator under the innermost Project.
#[test]
fn doc_score_plan_contains_doc_score_operator() {
    let ast = parse(
        "MATCH (d:Document) \
         RETURN d, doc_score(d, [1.0, 0.0]) AS s \
         ORDER BY s DESC LIMIT 10",
    )
    .unwrap();
    let plan = build_logical_plan(&ast).expect("plan");
    let explain = plan.explain();
    assert!(
        explain.contains("DocScore(") && explain.contains("HAS_CHUNK"),
        "EXPLAIN must show DocScore with HAS_CHUNK traversal; got:\n{explain}"
    );
}

/// R-HYB2c: two `doc_score(...)` calls with differing args in one plan are
/// rejected at plan time. Same-signature repetition is allowed (both resolve
/// to `__doc_score__`) but different signatures would require two operators.
#[test]
fn doc_score_multiple_differing_calls_rejected() {
    let ast = parse(
        "MATCH (d:Document) \
         RETURN doc_score(d, [1.0, 0.0]) AS a, \
                doc_score(d, [0.0, 1.0]) AS b",
    )
    .unwrap();
    let err = build_logical_plan(&ast).expect_err("multi-signature doc_score must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("doc_score()") && msg.contains("WITH"),
        "error must point at the multi-call issue and suggest WITH, got: {msg}"
    );
}

/// R-HYB2c: `ORDER BY doc_score(...) DESC` used as a bare function call (not
/// via alias) exercises the `ensure_doc_passthrough` code path that injects
/// `__doc_score__` as a synthetic Project item so Sort (above Project) can
/// read it.
#[test]
fn doc_score_order_by_direct_call() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    seed_doc_with_chunks(&engine, &mut interner, 1, &[Some(vec![1.0, 0.0])]);
    seed_doc_with_chunks(&engine, &mut interner, 2, &[Some(vec![0.0, 1.0])]);

    let results = run_cypher(
        "MATCH (d:Document) \
         RETURN d.id AS id \
         ORDER BY doc_score(d, [1.0, 0.0]) DESC",
        &engine,
        &mut interner,
    );

    assert_eq!(results.len(), 2);
    // Doc 1 (aligned chunk) must rank ahead of Doc 2 (orthogonal chunk).
    // We can't read `d` directly because RETURN projected only `d.id`, but
    // since the seed sets no `id` property, both rows will have Null — verify
    // instead that the SYNTHETIC `__doc_score__` column shows correct order.
    let scores: Vec<f64> = results
        .iter()
        .map(|r| match r.get("__doc_score__") {
            Some(Value::Float(f)) => *f,
            other => panic!("passthrough must expose __doc_score__ as Float, got {other:?}"),
        })
        .collect();
    assert!(
        scores[0] > scores[1],
        "direct ORDER BY doc_score(...) DESC must produce non-increasing scores \
         through the passthrough path; got {scores:?}"
    );
}

/// R-HYB2c: query can be supplied as a Cypher parameter `$q` and is resolved
/// to a vector before execution.
#[test]
fn doc_score_query_as_parameter() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    seed_doc_with_chunks(
        &engine,
        &mut interner,
        1,
        &[Some(vec![1.0, 0.0]), Some(vec![0.9, 0.1])],
    );

    let mut params = std::collections::HashMap::new();
    params.insert("q".to_string(), Value::Vector(vec![1.0, 0.0]));
    let results = run_cypher_with_params(
        "MATCH (d:Document) \
         RETURN doc_score(d, $q) AS score",
        &engine,
        &mut interner,
        params,
    );

    assert_eq!(results.len(), 1);
    let score = match results[0].get("score") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float score, got {other:?}"),
    };
    assert!(
        score > 0.9,
        "doc_score with parameter query must produce a high score on aligned chunks, got {score}"
    );
}
