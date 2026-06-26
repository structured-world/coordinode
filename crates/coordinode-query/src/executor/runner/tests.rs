use super::*;
use crate::cypher::ast::BinaryOperator;
// Adjacency key encoders are no longer used by production code in this
// module (the typed EdgeStore wrappers own them) — only test fixtures
// that plant/inspect raw adj posting lists need them.
use coordinode_core::graph::edge::{
    encode_adj_key_forward, encode_adj_key_reverse, encode_edgeprop_key,
};
use coordinode_core::graph::node::NodeRecord;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

/// Create a test engine in a temp directory.
fn test_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open engine")
}

/// Insert a test node into storage via the typed Layer-4
/// [`coordinode_modality::LocalNodeStore`]. The helper used to
/// hand-build the node key via `encode_node_key`; routing
/// through `NodeStore::put` keeps the fixture aligned with the
/// engine's idiomatic write path (R165 / R166 encoder lockdown).
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
    seed_node_record(engine, shard_id, NodeId::from_raw(node_id), &record);
}

/// Commit a built node record in its own MVCC transaction.
fn seed_node_record(engine: &StorageEngine, shard_id: u16, node_id: NodeId, record: &NodeRecord) {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    LocalNodeStore
        .put(&mut txn, shard_id, node_id, record)
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

/// Commit a temporal node version in its own MVCC transaction.
fn seed_node_temporal(
    engine: &StorageEngine,
    shard_id: u16,
    node_id: NodeId,
    valid_from_ms: i64,
    record: &NodeRecord,
) {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    LocalNodeStore
        .put_temporal(&mut txn, shard_id, node_id, valid_from_ms, record)
        .expect("put temporal node");
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

/// Read a node at the latest committed snapshot via an MVCC transaction.
fn read_node(engine: &StorageEngine, shard_id: u16, node_id: NodeId) -> Option<NodeRecord> {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::Transaction;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    LocalNodeStore
        .get(&txn, shard_id, node_id)
        .expect("get node")
}

/// Insert an edge via merge operator (both forward and reverse posting list).
/// Also registers the edge type in Schema so DETACH DELETE can use targeted lookup.
fn insert_edge(engine: &StorageEngine, edge_type: &str, source_id: u64, target_id: u64) {
    use coordinode_storage::engine::merge::encode_add;

    // Register edge type in schema (required for targeted DETACH DELETE lookup).
    let et_key = edge_type_schema_key(edge_type);
    engine
        .put(Partition::Schema, &et_key, b"")
        .expect("register edge type in schema");

    // Forward posting list: merge add (no read needed)
    let fwd_key = encode_adj_key_forward(edge_type, NodeId::from_raw(source_id));
    engine
        .merge(Partition::Adj, &fwd_key, &encode_add(target_id))
        .expect("merge fwd");

    // Reverse posting list: merge add (no read needed)
    let rev_key = encode_adj_key_reverse(edge_type, NodeId::from_raw(target_id));
    engine
        .merge(Partition::Adj, &rev_key, &encode_add(source_id))
        .expect("merge rev");
}

fn setup_test_graph() -> (tempfile::TempDir, StorageEngine, FieldInterner) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Create test nodes
    insert_node(
        &engine,
        1,
        1,
        "User",
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
        "User",
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
        "User",
        &[
            ("name", Value::String("Charlie".into())),
            ("age", Value::Int(35)),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        4,
        "Movie",
        &[("title", Value::String("Matrix".into()))],
        &mut interner,
    );

    // Create edges: Alice->Bob, Alice->Charlie, Bob->Charlie
    insert_edge(&engine, "KNOWS", 1, 2);
    insert_edge(&engine, "KNOWS", 1, 3);
    insert_edge(&engine, "KNOWS", 2, 3);

    // Alice likes Matrix
    insert_edge(&engine, "LIKES", 1, 4);

    (dir, engine, interner)
}

fn make_ctx<'a>(
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
        retention_window_us: 7 * 24 * 3600 * 1_000_000, // 7 days in micros
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
        procedure_ctx: None,
        txn: Transaction::new(
            engine,
            None,
            coordinode_core::txn::timestamp::Timestamp::ZERO,
            None,
        ),
        vector_consistency: VectorConsistencyMode::default(),
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: None,
        proposal_id_gen: None,
        read_concern: coordinode_core::txn::read_concern::ReadConcernLevel::Local,
        write_concern: coordinode_core::txn::write_concern::WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        mvcc_snapshot: None,
        // L1/L2 cascade tracking (the trigger architecture) — counters start at zero per
        // originating user mutation; defaults match cluster setting
        // defaults documented in the trigger architecture.
        cascade_depth: 0,
        cascade_depth_limit: 10,
        cascade_fire_counts: std::collections::HashMap::new(),
        cascade_fanout_limit: 100,
        cascade_chain: Vec::new(),
        after_commit_generation: 0,
        correlated_row: None,
        foreach_scope: None,
        feedback_cache: None,
        schema_label_cache: std::collections::HashMap::new(),
        applied_watermark: None,
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        read_timeout: std::time::Duration::from_millis(2000),
        params: std::collections::HashMap::new(),
        pending_vector_writes: Vec::new(),
    }
}

#[test]
fn extension_op_dispatches_to_registered_handler() {
    use std::sync::atomic::{AtomicBool, Ordering};

    struct RecordingHandler {
        called: Arc<AtomicBool>,
    }
    impl ExtensionHandler for RecordingHandler {
        fn execute(
            &self,
            _ctx: &mut ExecutionContext<'_>,
            payload: &[u8],
        ) -> Result<Vec<Row>, ExecutionError> {
            self.called.store(true, Ordering::SeqCst);
            // One row per payload byte: proves the opaque payload reaches
            // the handler intact through the dispatch arm.
            Ok((0..payload.len()).map(|_| Row::new()).collect())
        }
    }

    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    let called = Arc::new(AtomicBool::new(false));
    let mut registry = ExtensionRegistry::new();
    registry.register(
        "ee.create_index.sharded",
        Arc::new(RecordingHandler {
            called: Arc::clone(&called),
        }),
    );

    let mut ctx = ExecutionContext {
        extensions: Some(&registry),
        ..make_ctx(&engine, &mut interner, &allocator)
    };

    // A registered handler runs and receives the opaque payload.
    let op = LogicalOp::Extension {
        name: "ee.create_index.sharded".to_string(),
        payload: vec![7, 8, 9],
    };
    let rows = execute_op(&op, &mut ctx).expect("registered extension handler runs");
    assert!(called.load(Ordering::SeqCst), "handler was invoked");
    assert_eq!(
        rows.len(),
        3,
        "payload (3 bytes) reached the handler intact"
    );

    // An unknown op errors clearly — the CE default registers no handlers,
    // so this is the state of a pure-CE engine seeing an extension op.
    let unknown = LogicalOp::Extension {
        name: "ee.nonexistent".to_string(),
        payload: vec![],
    };
    assert!(matches!(
        execute_op(&unknown, &mut ctx).unwrap_err(),
        ExecutionError::Unsupported(_)
    ));
}

// -- NodeScan --

#[test]
fn node_scan_all() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec![],
                property_filters: vec![],
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::Star,
                alias: None,
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 4); // Alice, Bob, Charlie, Matrix
}

#[test]
fn node_scan_by_label() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::Star,
                alias: None,
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 3); // Alice, Bob, Charlie
}

#[test]
fn node_scan_with_property_filter() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::Star,
                alias: None,
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].get("n.name"),
        Some(&Value::String("Alice".into()))
    );
}

// -- Traverse --

#[test]
fn traverse_outgoing() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // MATCH (a:User {name: 'Alice'})-[:KNOWS]->(b) RETURN b.name
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["KNOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: None,
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
                temporal_filter: None,
                path_variable: None,
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("b".into())),
                    property: "name".into(),
                },
                alias: Some("name".into()),
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 2); // Alice knows Bob and Charlie

    let names: Vec<&Value> = result.iter().filter_map(|r| r.get("name")).collect();
    assert!(names.contains(&&Value::String("Bob".into())));
    assert!(names.contains(&&Value::String("Charlie".into())));
}

// -- Filter --

#[test]
fn filter_by_age() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // MATCH (n:User) WHERE n.age > 28 RETURN n.name
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::Filter {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                predicate: Expr::BinaryOp {
                    left: Box::new(Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    }),
                    op: BinaryOperator::Gt,
                    right: Box::new(Expr::Literal(Value::Int(28))),
                },
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "name".into(),
                },
                alias: Some("name".into()),
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 2); // Alice (30) and Charlie (35)
}

// -- Aggregate --

#[test]
fn aggregate_count() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![AggregateItem {
                    function: "count".into(),
                    arg: Expr::Star,
                    distinct: false,
                    alias: Some("cnt".into()),
                    percentile_expr: None,
                }],
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::Variable("cnt".into()),
                alias: None,
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get("cnt"), Some(&Value::Int(3)));
}

#[test]
fn aggregate_sum_avg() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // sum(n.age) and avg(n.age) for User nodes (ages: 30, 25, 35)
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![
                    AggregateItem {
                        function: "sum".into(),
                        arg: Expr::PropertyAccess {
                            expr: Box::new(Expr::Variable("n".into())),
                            property: "age".into(),
                        },
                        distinct: false,
                        alias: Some("total".into()),
                        percentile_expr: None,
                    },
                    AggregateItem {
                        function: "avg".into(),
                        arg: Expr::PropertyAccess {
                            expr: Box::new(Expr::Variable("n".into())),
                            property: "age".into(),
                        },
                        distinct: false,
                        alias: Some("average".into()),
                        percentile_expr: None,
                    },
                ],
            }),
            items: vec![
                crate::planner::logical::ProjectItem {
                    expr: Expr::Variable("total".into()),
                    alias: None,
                },
                crate::planner::logical::ProjectItem {
                    expr: Expr::Variable("average".into()),
                    alias: None,
                },
            ],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get("total"), Some(&Value::Int(90))); // 30+25+35 (all Int)
    assert_eq!(result[0].get("average"), Some(&Value::Float(30.0))); // 90/3
}

#[test]
fn aggregate_min_max() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![
                AggregateItem {
                    function: "min".into(),
                    arg: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    },
                    distinct: false,
                    alias: Some("youngest".into()),
                    percentile_expr: None,
                },
                AggregateItem {
                    function: "max".into(),
                    arg: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    },
                    distinct: false,
                    alias: Some("oldest".into()),
                    percentile_expr: None,
                },
            ],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get("youngest"), Some(&Value::Int(25)));
    assert_eq!(result[0].get("oldest"), Some(&Value::Int(35)));
}

#[test]
fn aggregate_collect() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![AggregateItem {
                function: "collect".into(),
                arg: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "name".into(),
                },
                distinct: false,
                alias: Some("names".into()),
                percentile_expr: None,
            }],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    if let Some(Value::Array(names)) = result[0].get("names") {
        assert_eq!(names.len(), 3);
        // Names should include Alice, Bob, Charlie (order may vary)
        let name_strings: Vec<&str> = names.iter().filter_map(|v| v.as_str()).collect();
        assert!(name_strings.contains(&"Alice"));
        assert!(name_strings.contains(&"Bob"));
        assert!(name_strings.contains(&"Charlie"));
    } else {
        panic!("expected Array for collect()");
    }
}

#[test]
fn aggregate_percentile_cont() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // percentileCont(n.age, 0.5) — median of [25, 30, 35] = 30.0
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![AggregateItem {
                function: "percentileCont".into(),
                arg: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "age".into(),
                },
                distinct: false,
                alias: Some("median".into()),
                percentile_expr: None,
            }],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    // Median of [25, 30, 35] = 30.0
    assert_eq!(result[0].get("median"), Some(&Value::Float(30.0)));
}

#[test]
fn aggregate_percentile_cont_non_median() {
    // Regression test: verifies that the percentile argument is actually used,
    // not silently replaced with 0.5 (median). Ages: [25, 30, 35].
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // percentileCont(n.age, 1.0) — 100th percentile of [25, 30, 35] = 35.0
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![AggregateItem {
                function: "percentileCont".into(),
                arg: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "age".into(),
                },
                distinct: false,
                alias: Some("p100".into()),
                percentile_expr: Some(Expr::Literal(Value::Float(1.0))),
            }],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    // 100th percentile of [25, 30, 35] = 35.0 (not 30.0 = median)
    assert_eq!(result[0].get("p100"), Some(&Value::Float(35.0)));

    // Also verify percentileDisc(n.age, 0.0) = 25.0
    let plan2 = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![AggregateItem {
                function: "percentileDisc".into(),
                arg: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "age".into(),
                },
                distinct: false,
                alias: Some("p0".into()),
                percentile_expr: Some(Expr::Literal(Value::Float(0.0))),
            }],
        },
    };

    let result2 = execute(&plan2, &mut ctx).expect("execute");
    assert_eq!(result2.len(), 1);
    // 0th percentile of [25, 30, 35] = 25.0 (not 30.0 = median)
    assert_eq!(result2[0].get("p0"), Some(&Value::Float(25.0)));
}

#[test]
fn aggregate_stdev() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![AggregateItem {
                function: "stDev".into(),
                arg: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "age".into(),
                },
                distinct: false,
                alias: Some("sd".into()),
                percentile_expr: None,
            }],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    // stDev of [25, 30, 35] = 5.0 (sample)
    if let Some(Value::Float(sd)) = result[0].get("sd") {
        assert!((sd - 5.0).abs() < 0.01, "expected ~5.0, got {sd}");
    } else {
        panic!("expected Float for stDev");
    }
}

#[test]
fn aggregate_empty_returns_null() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // sum/avg on empty set
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["NonExistent".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![
                AggregateItem {
                    function: "sum".into(),
                    arg: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    },
                    distinct: false,
                    alias: Some("total".into()),
                    percentile_expr: None,
                },
                AggregateItem {
                    function: "count".into(),
                    arg: Expr::Star,
                    distinct: false,
                    alias: Some("cnt".into()),
                    percentile_expr: None,
                },
            ],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    // count(*) on empty set = 0
    assert_eq!(result[0].get("cnt"), Some(&Value::Int(0)));
    // sum on empty set = null
    assert_eq!(result[0].get("total"), Some(&Value::Null));
}

// -- Sort + Limit --

#[test]
fn sort_and_limit() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Limit {
            input: Box::new(LogicalOp::Sort {
                input: Box::new(LogicalOp::Project {
                    input: Box::new(LogicalOp::NodeScan {
                        variable: "n".into(),
                        labels: vec!["User".into()],
                        property_filters: vec![],
                    }),
                    items: vec![
                        crate::planner::logical::ProjectItem {
                            expr: Expr::PropertyAccess {
                                expr: Box::new(Expr::Variable("n".into())),
                                property: "name".into(),
                            },
                            alias: Some("name".into()),
                        },
                        crate::planner::logical::ProjectItem {
                            expr: Expr::PropertyAccess {
                                expr: Box::new(Expr::Variable("n".into())),
                                property: "age".into(),
                            },
                            alias: Some("age".into()),
                        },
                    ],
                    distinct: false,
                }),
                items: vec![crate::cypher::ast::SortItem {
                    expr: Expr::Variable("age".into()),
                    ascending: false,
                }],
            }),
            count: Expr::Literal(Value::Int(2)),
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 2);
    // Sorted by age DESC: Charlie (35), Alice (30)
    assert_eq!(
        result[0].get("name"),
        Some(&Value::String("Charlie".into()))
    );
    assert_eq!(result[1].get("name"), Some(&Value::String("Alice".into())));
}

// -- Empty result --

#[test]
fn empty_scan() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["NonExistent".into()],
                property_filters: vec![],
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::Star,
                alias: None,
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert!(result.is_empty());
}

// ====== Write operations ======

#[test]
fn create_node_writes_to_storage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::CreateNode {
            input: None,
            variable: Some("n".into()),
            labels: vec!["User".into()],
            properties: vec![
                ("name".into(), Expr::Literal(Value::String("Alice".into()))),
                ("age".into(), Expr::Literal(Value::Int(30))),
            ],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    assert!(result[0].contains_key("n")); // Should have node ID

    // Verify node was written to storage
    let node_id = result[0]
        .get("n")
        .and_then(|v| v.as_int())
        .expect("node id");
    let record = read_node(&engine, 1, NodeId::from_raw(node_id as u64)).expect("node exists");
    assert_eq!(record.primary_label(), "User");
}

#[test]
fn create_and_return() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::CreateNode {
                input: None,
                variable: Some("n".into()),
                labels: vec!["User".into()],
                properties: vec![("name".into(), Expr::Literal(Value::String("Bob".into())))],
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "name".into(),
                },
                alias: Some("name".into()),
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get("name"), Some(&Value::String("Bob".into())));
}

#[test]
fn set_property_updates_storage() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // MATCH (n:User {name: 'Alice'}) SET n.name = 'Alicia'
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Update {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            items: vec![crate::cypher::ast::SetItem::Property {
                variable: "n".into(),
                property: "name".into(),
                expr: Expr::Literal(Value::String("Alicia".into())),
            }],
            violation_mode: crate::cypher::ast::ViolationMode::Fail,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);

    // Verify the update persisted in storage
    let node_id = result[0].get("n").and_then(|v| v.as_int()).expect("id");
    let record = read_node(&engine, 1, NodeId::from_raw(node_id as u64)).expect("exists");
    let name_id = interner.lookup("name").expect("field id");
    assert_eq!(record.get(name_id), Some(&Value::String("Alicia".into())));
}

#[test]
fn delete_removes_from_storage() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // First, verify Alice exists
    assert!(read_node(&engine, 1, NodeId::from_raw(1)).is_some());

    // DETACH DELETE node 1 (Alice) — she has edges, so DETACH is required.
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Delete {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            variables: vec!["n".into()],
            detach: true,
        },
    };

    execute(&plan, &mut ctx).expect("execute");

    // Verify Alice was deleted
    assert!(read_node(&engine, 1, NodeId::from_raw(1)).is_none());
}

#[test]
fn remove_property_from_node() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // MATCH (n:User {name: 'Alice'}) REMOVE n.age
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::RemoveOp {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            items: vec![crate::cypher::ast::RemoveItem::Property {
                variable: "n".into(),
                property: "age".into(),
            }],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);

    // Verify age was removed
    let node_id = result[0].get("n").and_then(|v| v.as_int()).expect("id");
    let record = read_node(&engine, 1, NodeId::from_raw(node_id as u64)).expect("exists");
    let age_id = interner.lookup("age").expect("field id");
    assert!(record.get(age_id).is_none());
}

#[test]
fn create_multiple_nodes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // Create first node
    let plan1 = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::CreateNode {
            input: None,
            variable: Some("a".into()),
            labels: vec!["User".into()],
            properties: vec![("name".into(), Expr::Literal(Value::String("X".into())))],
        },
    };
    execute(&plan1, &mut ctx).expect("create a");

    // Create second node
    let plan2 = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::CreateNode {
            input: None,
            variable: Some("b".into()),
            labels: vec!["User".into()],
            properties: vec![("name".into(), Expr::Literal(Value::String("Y".into())))],
        },
    };
    execute(&plan2, &mut ctx).expect("create b");

    // Verify both exist with unique IDs
    assert_eq!(allocator.current().as_raw(), 2);
}

// ====== MERGE / UPSERT ======

#[test]
fn merge_creates_when_not_found() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // MERGE (n:User {email: 'alice@test.com'}) ON CREATE SET n.name = 'Alice'
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Merge {
            pattern: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "email".into(),
                    Expr::Literal(Value::String("alice@test.com".into())),
                )],
            }),
            on_match: vec![],
            on_create: vec![crate::cypher::ast::SetItem::Property {
                variable: "n".into(),
                property: "name".into(),
                expr: Expr::Literal(Value::String("Alice".into())),
            }],
            multi: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    // Node should have been created
    assert!(result[0].contains_key("n"));
}

#[test]
fn merge_updates_when_found() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // MERGE (n:User {name: 'Alice'}) ON MATCH SET n.age = 31
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Merge {
            pattern: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            on_match: vec![crate::cypher::ast::SetItem::Property {
                variable: "n".into(),
                property: "age".into(),
                expr: Expr::Literal(Value::Int(31)),
            }],
            on_create: vec![],
            multi: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);

    // Verify age was updated
    let node_id = result[0].get("n").and_then(|v| v.as_int()).expect("id");
    let record = read_node(&engine, 1, NodeId::from_raw(node_id as u64)).expect("exists");
    let age_id = interner.lookup("age").expect("field id");
    assert_eq!(record.get(age_id), Some(&Value::Int(31)));
}

#[test]
fn merge_no_duplicate_on_existing() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // MERGE on existing node should NOT create a new node
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Merge {
            pattern: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            on_match: vec![],
            on_create: vec![],
            multi: false,
        },
    };

    execute(&plan, &mut ctx).expect("execute");

    // Allocator should NOT have advanced (no new node created)
    assert_eq!(allocator.current().as_raw(), 100);
}

#[test]
fn upsert_creates_when_not_found() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // UPSERT MATCH (u:User {email: 'bob@test.com'})
    // ON CREATE CREATE (u:User {email: 'bob@test.com', name: 'Bob'})
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Upsert {
            pattern: Box::new(LogicalOp::NodeScan {
                variable: "u".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "email".into(),
                    Expr::Literal(Value::String("bob@test.com".into())),
                )],
            }),
            on_match: vec![],
            on_create_patterns: vec![Pattern {
                elements: vec![PatternElement::Node(crate::cypher::ast::NodePattern {
                    variable: Some("u".into()),
                    labels: vec!["User".into()],
                    properties: vec![
                        (
                            "email".into(),
                            Expr::Literal(Value::String("bob@test.com".into())),
                        ),
                        ("name".into(), Expr::Literal(Value::String("Bob".into()))),
                    ],
                })],
                path_variable: None,
                shortest_path: false,
            }],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get("u.name"), Some(&Value::String("Bob".into())));
}

#[test]
fn upsert_updates_when_found() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // UPSERT MATCH (n:User {name: 'Alice'})
    // ON MATCH SET n.age = 31
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Upsert {
            pattern: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            on_match: vec![crate::cypher::ast::SetItem::Property {
                variable: "n".into(),
                property: "age".into(),
                expr: Expr::Literal(Value::Int(31)),
            }],
            on_create_patterns: vec![],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);

    // Verify no new node was created
    assert_eq!(allocator.current().as_raw(), 100);

    // Verify age was updated
    let node_id = result[0].get("n").and_then(|v| v.as_int()).expect("id");
    let record = read_node(&engine, 1, NodeId::from_raw(node_id as u64)).expect("exists");
    let age_id = interner.lookup("age").expect("field id");
    assert_eq!(record.get(age_id), Some(&Value::Int(31)));
}

// ====== UPSERT CAS conflict ======

#[test]
fn upsert_cas_conflict_on_external_modification() {
    // Test CAS conflict detection: modify node externally between
    // two sequential UPSERTs that target the same node.
    // The first UPSERT changes the node, the second should still succeed
    // because CAS reads fresh bytes before comparing.
    //
    // To test actual conflict detection, we'd need to inject a
    // modification between the MATCH and SET phases inside execute_upsert.
    // Since execute_upsert is private, we verify the mechanism indirectly:
    // two sequential UPSERTs both succeed (no false positives).
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

    // First UPSERT: set age=50
    {
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(
            ),
            root: LogicalOp::Upsert {
                pattern: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                on_match: vec![crate::cypher::ast::SetItem::Property {
                    variable: "n".into(),
                    property: "age".into(),
                    expr: Expr::Literal(Value::Int(50)),
                }],
                on_create_patterns: vec![],
            },
        };
        execute(&plan, &mut ctx).expect("first upsert");
    }

    // Externally modify Alice's age directly via storage (simulates
    // concurrent modification from another connection)
    let alice_id = NodeId::from_raw(1);
    {
        let mut record = read_node(&engine, 1, alice_id).unwrap();
        let age_id = interner.lookup("age").unwrap();
        record.set(age_id, Value::Int(999)); // external modification
        seed_node_record(&engine, 1, alice_id, &record);
    }

    // Second UPSERT: this reads fresh bytes in CAS snapshot,
    // then CAS re-reads and compares — they match because CAS
    // snapshot was taken after the external modification.
    // So this should succeed (CAS reads at the same moment).
    {
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(
            ),
            root: LogicalOp::Upsert {
                pattern: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                on_match: vec![crate::cypher::ast::SetItem::Property {
                    variable: "n".into(),
                    property: "age".into(),
                    expr: Expr::Literal(Value::Int(60)),
                }],
                on_create_patterns: vec![],
            },
        };
        let result = execute(&plan, &mut ctx).expect("second upsert should succeed");
        assert_eq!(result.len(), 1);
    }

    // Verify final value is 60 (second UPSERT applied)
    let record = read_node(&engine, 1, alice_id).unwrap();
    let age_id = interner.lookup("age").unwrap();
    assert_eq!(record.get(age_id), Some(&Value::Int(60)));
}

#[test]
fn upsert_errconflict_variant_exists() {
    // Verify ExecutionError::Conflict variant exists and formats correctly
    let err = ExecutionError::Conflict("test conflict".into());
    let msg = format!("{err}");
    assert!(msg.contains("write conflict"));
    assert!(msg.contains("test conflict"));
}

// ====== Adaptive query plans ======

#[test]
fn adaptive_parallel_on_super_node() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);

    // Create source node
    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("name", Value::String("Hub".into()))],
        &mut interner,
    );

    // Create 20 target nodes and edges (simulate high fan-out)
    for i in 2..=21u64 {
        insert_node(
            &engine,
            1,
            i,
            "User",
            &[("name", Value::String(format!("T{i}")))],
            &mut interner,
        );
        insert_edge(&engine, "FOLLOWS", 1, i);
    }

    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    // Set low threshold to trigger parallel processing on 20 edges
    ctx.adaptive.parallel_threshold = 5;

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Hub".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["FOLLOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: None,
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
                temporal_filter: None,
                path_variable: None,
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::Star,
                alias: None,
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    // ALL 20 results returned (no truncation — parallel processes all edges)
    assert_eq!(result.len(), 20);
    // Should have a warning about parallel activation
    assert!(
        ctx.warnings.iter().any(|w| w.contains("parallel")),
        "expected parallel activation warning, got: {:?}",
        ctx.warnings,
    );
}

#[test]
fn adaptive_disabled_no_cap() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);

    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("name", Value::String("Hub".into()))],
        &mut interner,
    );
    for i in 2..=11u64 {
        insert_node(
            &engine,
            1,
            i,
            "User",
            &[("name", Value::String(format!("T{i}")))],
            &mut interner,
        );
        insert_edge(&engine, "FOLLOWS", 1, i);
    }

    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.adaptive.enabled = false; // Disable adaptive

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![("name".into(), Expr::Literal(Value::String("Hub".into())))],
            }),
            source: "a".into(),
            edge_types: vec!["FOLLOWS".into()],
            direction: Direction::Outgoing,
            target_variable: "b".into(),
            target_labels: vec![],
            length: None,
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    // All 10 results should be returned (no cap)
    assert_eq!(result.len(), 10);
    assert!(ctx.warnings.is_empty());
}

#[test]
fn adaptive_normal_fan_out_no_warning() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    // Default max_fan_out is 10_000, our test graph has 2-3 edges per node

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            source: "a".into(),
            edge_types: vec!["KNOWS".into()],
            direction: Direction::Outgoing,
            target_variable: "b".into(),
            target_labels: vec![],
            length: None,
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 2); // Alice knows Bob and Charlie
    assert!(ctx.warnings.is_empty()); // No super-node warning
}

#[test]
fn feedback_cache_records_super_node() {
    let cache = FeedbackCache::new(100);
    assert!(cache.lookup(42).is_none());

    cache.record(42, 50_000);
    assert_eq!(cache.lookup(42), Some(50_000));

    // Overwrite with new degree
    cache.record(42, 75_000);
    assert_eq!(cache.lookup(42), Some(75_000));
}

#[test]
fn feedback_cache_evicts_when_full() {
    let cache = FeedbackCache::new(10);
    for i in 0..10 {
        cache.record(i, 1000 + i as usize);
    }
    assert_eq!(cache.lookup(0), Some(1000));
    assert_eq!(cache.lookup(9), Some(1009));

    // Adding one more should trigger eviction of half (5 entries)
    cache.record(100, 9999);
    // After eviction, some early entries should be gone
    let remaining: usize = (0..10).filter(|i| cache.lookup(*i).is_some()).count();
    assert!(
        remaining < 10,
        "expected eviction, but {remaining}/10 entries remain"
    );
    // New entry should be present
    assert_eq!(cache.lookup(100), Some(9999));
}

#[test]
fn adaptive_parallel_correctness_matches_sequential() {
    // Verify parallel path produces the same results as sequential
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);

    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("name", Value::String("Hub".into()))],
        &mut interner,
    );
    for i in 2..=11u64 {
        insert_node(
            &engine,
            1,
            i,
            "User",
            &[("name", Value::String(format!("T{i}")))],
            &mut interner,
        );
        insert_edge(&engine, "FOLLOWS", 1, i);
    }

    // Run with parallel (threshold = 5, so 10 edges triggers parallel)
    let mut ctx_par = make_ctx(&engine, &mut interner, &allocator);
    ctx_par.adaptive.parallel_threshold = 5;

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![("name".into(), Expr::Literal(Value::String("Hub".into())))],
            }),
            source: "a".into(),
            edge_types: vec!["FOLLOWS".into()],
            direction: Direction::Outgoing,
            target_variable: "b".into(),
            target_labels: vec![],
            length: None,
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        },
    };

    let result_par = execute(&plan, &mut ctx_par).expect("parallel execute");

    // Run without parallel (disabled)
    let mut ctx_seq = make_ctx(&engine, &mut interner, &allocator);
    ctx_seq.adaptive.enabled = false;

    let result_seq = execute(&plan, &mut ctx_seq).expect("sequential execute");

    // Same number of results
    assert_eq!(result_par.len(), result_seq.len());
    assert_eq!(result_par.len(), 10);

    // Same node IDs (order may differ due to parallel execution)
    let mut par_ids: Vec<i64> = result_par
        .iter()
        .filter_map(|r| {
            r.get("b").and_then(|v| {
                if let Value::Int(i) = v {
                    Some(*i)
                } else {
                    None
                }
            })
        })
        .collect();
    let mut seq_ids: Vec<i64> = result_seq
        .iter()
        .filter_map(|r| {
            r.get("b").and_then(|v| {
                if let Value::Int(i) = v {
                    Some(*i)
                } else {
                    None
                }
            })
        })
        .collect();
    par_ids.sort();
    seq_ids.sort();
    assert_eq!(par_ids, seq_ids);
}

// ====== AS OF TIMESTAMP ======

#[test]
fn as_of_timestamp_sets_snapshot() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // Use a recent timestamp (within retention window)
    let recent_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
        - 3600 * 1_000_000; // 1 hour ago

    let plan = LogicalPlan {
        snapshot_ts: Some(Expr::Literal(Value::Int(recent_ts))),
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::Star,
                alias: None,
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert!(!result.is_empty());
    // Snapshot timestamp should be set
    assert_eq!(ctx.snapshot_ts, Some(recent_ts));
}

#[test]
fn as_of_timestamp_rejects_expired() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // Use a very old timestamp (30 days ago — outside 7-day retention)
    let old_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
        - 30 * 24 * 3600 * 1_000_000; // 30 days ago

    let plan = LogicalPlan {
        snapshot_ts: Some(Expr::Literal(Value::Timestamp(old_ts))),
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::Star,
                alias: None,
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("retention window"));
}

#[test]
fn as_of_timestamp_string_warning() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: Some(Expr::Literal(Value::String("2025-06-15T10:00:00Z".into()))),
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::Star,
                alias: None,
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert!(!result.is_empty());
    // Should have a warning about string parsing
    assert!(ctx.warnings.iter().any(|w| w.contains("AS OF TIMESTAMP")));
}

#[test]
fn no_timestamp_leaves_snapshot_none() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            items: vec![crate::planner::logical::ProjectItem {
                expr: Expr::Star,
                alias: None,
            }],
            distinct: false,
        },
    };

    execute(&plan, &mut ctx).expect("execute");
    assert!(ctx.snapshot_ts.is_none());
}

// -- Variable-length path traversal --

/// Build a longer test graph for multi-hop tests:
/// A(1)→B(2)→C(3)→D(5)→E(6), plus A→C, B→C (already in setup)
fn setup_varlen_graph() -> (tempfile::TempDir, StorageEngine, FieldInterner) {
    let (dir, engine, mut interner) = setup_test_graph();

    // Add nodes D and E
    insert_node(
        &engine,
        1,
        5,
        "User",
        &[("name", Value::String("Dave".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        6,
        "User",
        &[("name", Value::String("Eve".into()))],
        &mut interner,
    );

    // Extend chain: Charlie(3)→Dave(5), Dave(5)→Eve(6)
    insert_edge(&engine, "KNOWS", 3, 5);
    insert_edge(&engine, "KNOWS", 5, 6);

    (dir, engine, interner)
}

#[test]
fn varlen_traverse_exact_2_hops() {
    // Alice -[:KNOWS*2..2]-> ? should find Charlie (via Alice→Bob→Charlie)
    // and Dave (via Alice→Charlie→Dave)
    let (_dir, engine, mut interner) = setup_varlen_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            source: "a".into(),
            edge_types: vec!["KNOWS".into()],
            direction: Direction::Outgoing,
            target_variable: "b".into(),
            target_labels: vec![],
            length: Some(LengthBound {
                min: Some(2),
                max: Some(2),
            }),
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        },
    };

    let results = execute(&plan, &mut ctx).expect("execute");
    let target_ids: Vec<i64> = results
        .iter()
        .filter_map(|r| match r.get("b") {
            Some(Value::Int(id)) => Some(*id),
            _ => None,
        })
        .collect();
    // At 2 hops from Alice: Charlie (via Bob) and Dave (via Charlie)
    assert!(target_ids.contains(&3), "should reach Charlie at 2 hops");
    assert!(target_ids.contains(&5), "should reach Dave at 2 hops");
}

/// `count(DISTINCT b)` over a var-length traverse whose targets are reachable
/// via multiple paths must equal the number of DISTINCT reached nodes. This
/// is the shape the planner enables target dedup for, so the test guards that
/// the per-node emission optimisation does not drop or double-count nodes.
#[test]
fn varlen_count_distinct_dedups_multipath_targets() {
    // A(1)→B(2)→C(3)→D(5)→E(6), plus A→C, B→C. From Alice within 1..3 the
    // distinct reachable set is {B, C, D, E} = 4, but C and D are each
    // reachable by two paths, so a per-edge emission would over-count.
    let (_dir, engine, mut interner) = setup_varlen_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let traverse = LogicalOp::Traverse {
        input: Box::new(LogicalOp::NodeScan {
            variable: "a".into(),
            labels: vec!["User".into()],
            property_filters: vec![("name".into(), Expr::Literal(Value::String("Alice".into())))],
        }),
        source: "a".into(),
        edge_types: vec!["KNOWS".into()],
        direction: Direction::Outgoing,
        target_variable: "b".into(),
        target_labels: vec![],
        length: Some(LengthBound {
            min: Some(1),
            max: Some(3),
        }),
        edge_variable: None,
        target_filters: vec![],
        edge_filters: vec![],
        temporal_filter: None,
        path_variable: None,
    };

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(traverse),
            group_by: vec![],
            aggregates: vec![crate::planner::logical::AggregateItem {
                function: "count".into(),
                arg: Expr::Variable("b".into()),
                distinct: true,
                alias: Some("cnt".into()),
                percentile_expr: None,
            }],
        },
    };

    // The planner detector must enable dedup for exactly this shape.
    assert!(
        plan_allows_varlen_target_dedup(&plan.root),
        "count(DISTINCT b) over a lone var-length traverse must allow dedup"
    );

    let results = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(results.len(), 1, "count aggregate yields one row");
    let cnt = match results[0].get("cnt") {
        Some(Value::Int(n)) => *n,
        other => panic!("expected Int count, got {other:?}"),
    };
    assert_eq!(
        cnt, 4,
        "distinct reachable nodes within 1..3 are B, C, D, E"
    );
}

/// A plain `RETURN b` (no DISTINCT, no aggregate) must NOT trigger target
/// dedup: per-path multiplicity is observable, so a multi-path node appears
/// once per reaching path. Guards the detector against a false positive.
#[test]
fn varlen_plain_return_keeps_path_multiplicity() {
    let (_dir, engine, mut interner) = setup_varlen_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            source: "a".into(),
            edge_types: vec!["KNOWS".into()],
            direction: Direction::Outgoing,
            target_variable: "b".into(),
            target_labels: vec![],
            length: Some(LengthBound {
                min: Some(1),
                max: Some(3),
            }),
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        },
    };

    // A bare traverse (no count(DISTINCT) parent) must not allow dedup.
    assert!(
        !plan_allows_varlen_target_dedup(&plan.root),
        "a bare traverse must not enable target dedup"
    );

    let results = execute(&plan, &mut ctx).expect("execute");
    let charlie_rows = results
        .iter()
        .filter(|r| matches!(r.get("b"), Some(Value::Int(3))))
        .count();
    // Charlie (3) is reachable as A→C (1 hop) and A→B→C (2 hops): two paths,
    // so without dedup it must appear at least twice.
    assert!(
        charlie_rows >= 2,
        "multi-path node must keep per-path multiplicity without DISTINCT, got {charlie_rows}"
    );
}

#[test]
fn varlen_traverse_range_1_to_3() {
    // Alice -[:KNOWS*1..3]-> ? should find Bob(1hop), Charlie(1hop+2hop), Dave(2hop+3hop), Eve(3hop)
    let (_dir, engine, mut interner) = setup_varlen_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            source: "a".into(),
            edge_types: vec!["KNOWS".into()],
            direction: Direction::Outgoing,
            target_variable: "b".into(),
            target_labels: vec![],
            length: Some(LengthBound {
                min: Some(1),
                max: Some(3),
            }),
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        },
    };

    let results = execute(&plan, &mut ctx).expect("execute");
    let target_ids: Vec<i64> = results
        .iter()
        .filter_map(|r| match r.get("b") {
            Some(Value::Int(id)) => Some(*id),
            _ => None,
        })
        .collect();
    // Should reach: Bob(1), Charlie(1+2), Dave(2+3), Eve(3)
    assert!(target_ids.contains(&2), "should reach Bob");
    assert!(target_ids.contains(&3), "should reach Charlie");
    assert!(target_ids.contains(&5), "should reach Dave");
    assert!(target_ids.contains(&6), "should reach Eve");
}

#[test]
fn varlen_traverse_cycle_detection() {
    // Create a cycle: A→B→C→A
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("name", Value::String("A".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "User",
        &[("name", Value::String("B".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "User",
        &[("name", Value::String("C".into()))],
        &mut interner,
    );

    insert_edge(&engine, "KNOWS", 1, 2); // A→B
    insert_edge(&engine, "KNOWS", 2, 3); // B→C
    insert_edge(&engine, "KNOWS", 3, 1); // C→A (cycle!)

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // Traverse *1..10 — should NOT loop forever thanks to edge-uniqueness
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![("name".into(), Expr::Literal(Value::String("A".into())))],
            }),
            source: "a".into(),
            edge_types: vec!["KNOWS".into()],
            direction: Direction::Outgoing,
            target_variable: "b".into(),
            target_labels: vec![],
            length: Some(LengthBound {
                min: Some(1),
                max: Some(10),
            }),
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        },
    };

    let results = execute(&plan, &mut ctx).expect("execute should not hang");
    // With edge-uniqueness: 3 edges total (A→B, B→C, C→A), so max 3 results
    assert!(
        results.len() <= 3,
        "cycle detection should cap at 3 edges, got {}",
        results.len()
    );
}

#[test]
fn varlen_traverse_min_greater_than_max_yields_empty() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            source: "a".into(),
            edge_types: vec!["KNOWS".into()],
            direction: Direction::Outgoing,
            target_variable: "b".into(),
            target_labels: vec![],
            length: Some(LengthBound {
                min: Some(5),
                max: Some(2),
            }),
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        },
    };

    let results = execute(&plan, &mut ctx).expect("execute");
    assert!(results.is_empty(), "min > max should yield no results");
}

// -- UNWIND --

#[test]
fn unwind_list_expansion() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // UNWIND [1, 2, 3] AS x → 3 rows
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Unwind {
            input: Box::new(LogicalOp::Empty),
            expr: Expr::List(vec![
                Expr::Literal(Value::Int(1)),
                Expr::Literal(Value::Int(2)),
                Expr::Literal(Value::Int(3)),
            ]),
            variable: "x".into(),
        },
    };

    let results = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(results.len(), 3, "UNWIND [1,2,3] should produce 3 rows");
    assert_eq!(results[0].get("x"), Some(&Value::Int(1)));
    assert_eq!(results[1].get("x"), Some(&Value::Int(2)));
    assert_eq!(results[2].get("x"), Some(&Value::Int(3)));
}

#[test]
fn unwind_null_produces_empty() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Unwind {
            input: Box::new(LogicalOp::Empty),
            expr: Expr::Literal(Value::Null),
            variable: "x".into(),
        },
    };

    let results = execute(&plan, &mut ctx).expect("execute");
    assert!(results.is_empty(), "UNWIND NULL should produce zero rows");
}

#[test]
fn unwind_scalar_single_row() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Unwind {
            input: Box::new(LogicalOp::Empty),
            expr: Expr::Literal(Value::Int(42)),
            variable: "x".into(),
        },
    };

    let results = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(results.len(), 1, "UNWIND scalar should produce 1 row");
    assert_eq!(results[0].get("x"), Some(&Value::Int(42)));
}

// -- OPTIONAL MATCH (LeftOuterJoin) --

#[test]
fn optional_match_with_results() {
    // Alice has KNOWS edges → should return real results
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::LeftOuterJoin {
            left: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            right: Box::new(LogicalOp::NodeScan {
                variable: "m".into(),
                labels: vec!["Movie".into()],
                property_filters: vec![],
            }),
        },
    };

    let results = execute(&plan, &mut ctx).expect("execute");
    // Alice × Movie(Matrix) → at least 1 row
    assert!(!results.is_empty(), "should have results from Movie scan");
    assert!(results[0].contains_key("a"), "left variable should be set");
    assert!(results[0].contains_key("m"), "right variable should be set");
}

#[test]
fn optional_match_no_results_nulls() {
    // Scan for non-existent label → right side empty → NULLs
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::LeftOuterJoin {
            left: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            right: Box::new(LogicalOp::NodeScan {
                variable: "x".into(),
                labels: vec!["NonExistent".into()],
                property_filters: vec![],
            }),
        },
    };

    let results = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(results.len(), 1, "should have 1 row (left row with NULLs)");
    assert!(results[0].contains_key("a"), "left variable should be set");
    assert_eq!(
        results[0].get("x"),
        Some(&Value::Null),
        "right var should be NULL"
    );
}

// -- Shortest Path --

#[test]
fn shortest_path_direct() {
    // Alice→Bob is 1 hop
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let mut input_row = Row::new();
    input_row.insert("a".into(), Value::Int(1)); // Alice
    input_row.insert("b".into(), Value::Int(2)); // Bob

    let sp = ShortestPathParams {
        source: "a",
        target: "b",
        edge_types: &["KNOWS".into()],
        direction: Direction::Outgoing,
        max_depth: 10,
        path_variable: "p",
    };

    let results = execute_shortest_path(&[input_row], &sp, &mut ctx).expect("sp");
    assert_eq!(results.len(), 1);
    // p is now a Path: Alice -[:KNOWS]-> Bob (length 1, nodes [1, 2]).
    assert_eq!(
        results[0].get("p"),
        Some(&Value::Path(coordinode_core::graph::types::PathValue {
            nodes: vec![1, 2],
            rels: vec![coordinode_core::graph::types::PathRel {
                edge_type: "KNOWS".into(),
                source: 1,
                target: 2,
            }],
        }))
    );
}

#[test]
fn shortest_path_two_hops() {
    // Alice→Bob→Charlie is 2 hops; Alice→Charlie is 1 hop (direct)
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let mut input_row = Row::new();
    input_row.insert("a".into(), Value::Int(1)); // Alice
    input_row.insert("c".into(), Value::Int(3)); // Charlie

    let sp = ShortestPathParams {
        source: "a",
        target: "c",
        edge_types: &["KNOWS".into()],
        direction: Direction::Outgoing,
        max_depth: 10,
        path_variable: "p",
    };

    let results = execute_shortest_path(&[input_row], &sp, &mut ctx).expect("sp");
    assert_eq!(results.len(), 1);
    // Alice→Charlie is direct (1 hop): path nodes [1, 3], one KNOWS rel.
    assert_eq!(
        results[0].get("p"),
        Some(&Value::Path(coordinode_core::graph::types::PathValue {
            nodes: vec![1, 3],
            rels: vec![coordinode_core::graph::types::PathRel {
                edge_type: "KNOWS".into(),
                source: 1,
                target: 3,
            }],
        }))
    );
}

#[test]
fn shortest_path_unreachable() {
    // Matrix(4) has no KNOWS edges to anyone
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let mut input_row = Row::new();
    input_row.insert("a".into(), Value::Int(4)); // Matrix
    input_row.insert("b".into(), Value::Int(1)); // Alice

    let sp = ShortestPathParams {
        source: "a",
        target: "b",
        edge_types: &["KNOWS".into()],
        direction: Direction::Outgoing,
        max_depth: 10,
        path_variable: "p",
    };

    let results = execute_shortest_path(&[input_row], &sp, &mut ctx).expect("sp");
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("p"),
        Some(&Value::Null),
        "unreachable → NULL"
    );
}

#[test]
fn shortest_path_same_node() {
    // Alice→Alice is 0 hops
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let mut input_row = Row::new();
    input_row.insert("a".into(), Value::Int(1));
    input_row.insert("b".into(), Value::Int(1));

    let sp = ShortestPathParams {
        source: "a",
        target: "b",
        edge_types: &["KNOWS".into()],
        direction: Direction::Outgoing,
        max_depth: 10,
        path_variable: "p",
    };

    let results = execute_shortest_path(&[input_row], &sp, &mut ctx).expect("sp");
    // Alice→Alice is a zero-length path: a single node, no relationships.
    assert_eq!(
        results[0].get("p"),
        Some(&Value::Path(coordinode_core::graph::types::PathValue {
            nodes: vec![1],
            rels: vec![],
        }))
    );
}

// -- Aggregation: DISTINCT --

#[test]
fn aggregate_count_distinct() {
    // Create rows with duplicate age values
    let (_dir, engine, mut interner) = setup_test_graph();

    // Add another User with age=30 (same as Alice) — before ctx borrow
    insert_node(
        &engine,
        1,
        10,
        "User",
        &[
            ("name", Value::String("Frank".into())),
            ("age", Value::Int(30)),
        ],
        &mut interner,
    );

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // count(DISTINCT n.age) on all Users should be 3 (30, 25, 35) not 4
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![
                AggregateItem {
                    function: "count".into(),
                    arg: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    },
                    distinct: true,
                    alias: Some("unique_ages".into()),
                    percentile_expr: None,
                },
                AggregateItem {
                    function: "count".into(),
                    arg: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    },
                    distinct: false,
                    alias: Some("total_ages".into()),
                    percentile_expr: None,
                },
            ],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get("unique_ages"), Some(&Value::Int(3))); // 30, 25, 35
    assert_eq!(result[0].get("total_ages"), Some(&Value::Int(4))); // 30, 25, 35, 30
}

#[test]
fn aggregate_collect_distinct() {
    let (_dir, engine, mut interner) = setup_test_graph();

    // Add user with duplicate age — before ctx borrow
    insert_node(
        &engine,
        1,
        10,
        "User",
        &[
            ("name", Value::String("Frank".into())),
            ("age", Value::Int(30)),
        ],
        &mut interner,
    );

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![AggregateItem {
                function: "collect".into(),
                arg: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "age".into(),
                },
                distinct: true,
                alias: Some("ages".into()),
                percentile_expr: None,
            }],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    if let Some(Value::Array(ages)) = result[0].get("ages") {
        assert_eq!(ages.len(), 3, "collect(DISTINCT) should have 3 unique ages");
    } else {
        panic!("expected Array for collect(DISTINCT)");
    }
}

#[test]
fn aggregate_sum_int_preserves_type() {
    // sum() on all-Int values should return Int, not Float
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![AggregateItem {
                function: "sum".into(),
                arg: Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "age".into(),
                },
                distinct: false,
                alias: Some("total".into()),
                percentile_expr: None,
            }],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    // All ages are Int, so sum should be Int
    assert!(matches!(result[0].get("total"), Some(Value::Int(_))));
}

#[test]
fn aggregate_group_by_with_null() {
    // GROUP BY should create separate group for NULL values
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "Item",
        &[("category", Value::String("A".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "Item",
        &[("category", Value::String("A".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Item",
        &[("category", Value::String("B".into()))],
        &mut interner,
    );
    insert_node(&engine, 1, 4, "Item", &[], &mut interner); // No category → NULL group

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["Item".into()],
                property_filters: vec![],
            }),
            group_by: vec![Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".into())),
                property: "category".into(),
            }],
            aggregates: vec![AggregateItem {
                function: "count".into(),
                arg: Expr::Star,
                distinct: false,
                alias: Some("cnt".into()),
                percentile_expr: None,
            }],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    // Should have 3 groups: A (2 items), B (1 item), NULL (1 item)
    assert_eq!(result.len(), 3, "should have 3 groups (A, B, NULL)");
}

#[test]
fn aggregate_empty_input() {
    // Aggregation on empty input: count=0, sum/avg=NULL
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Aggregate {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["NonExistent".into()],
                property_filters: vec![],
            }),
            group_by: vec![],
            aggregates: vec![
                AggregateItem {
                    function: "count".into(),
                    arg: Expr::Star,
                    distinct: false,
                    alias: Some("cnt".into()),
                    percentile_expr: None,
                },
                AggregateItem {
                    function: "sum".into(),
                    arg: Expr::Literal(Value::Int(1)),
                    distinct: false,
                    alias: Some("total".into()),
                    percentile_expr: None,
                },
                AggregateItem {
                    function: "avg".into(),
                    arg: Expr::Literal(Value::Int(1)),
                    distinct: false,
                    alias: Some("mean".into()),
                    percentile_expr: None,
                },
            ],
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1, "aggregate on empty → 1 row with defaults");
    assert_eq!(result[0].get("cnt"), Some(&Value::Int(0)));
    assert_eq!(result[0].get("total"), Some(&Value::Null));
    assert_eq!(result[0].get("mean"), Some(&Value::Null));
}

#[test]
fn aggregate_pipeline_with_then_group_by() {
    // Test multi-stage: scan → aggregate → project (WITH) → uses alias
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // Simulate: WITH count(*) AS cnt RETURN cnt
    // This is Aggregate → Project → Project
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Project {
            input: Box::new(LogicalOp::Project {
                input: Box::new(LogicalOp::Aggregate {
                    input: Box::new(LogicalOp::NodeScan {
                        variable: "n".into(),
                        labels: vec!["User".into()],
                        property_filters: vec![],
                    }),
                    group_by: vec![],
                    aggregates: vec![AggregateItem {
                        function: "count".into(),
                        arg: Expr::Star,
                        distinct: false,
                        alias: Some("cnt".into()),
                        percentile_expr: None,
                    }],
                }),
                items: vec![ProjectItem {
                    expr: Expr::Variable("cnt".into()),
                    alias: Some("cnt".into()),
                }],
                distinct: false,
            }),
            items: vec![ProjectItem {
                expr: Expr::Variable("cnt".into()),
                alias: Some("total".into()),
            }],
            distinct: false,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get("total"), Some(&Value::Int(3))); // 3 User nodes
}

// -- DETACH DELETE reverse posting list cleanup (G050) --

#[test]
fn detach_delete_cleans_reverse_posting_lists() {
    // Setup: Alice(1)->Bob(2) via KNOWS, Alice(1)->Charlie(3) via KNOWS,
    //        Bob(2)->Charlie(3) via KNOWS, Alice(1)->Matrix(4) via LIKES.
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // Verify Bob's incoming KNOWS posting list contains Alice(1) before delete
    let bob_in_key = encode_adj_key_reverse("KNOWS", NodeId::from_raw(2));
    let plist = ctx.adj_get(&bob_in_key).expect("read").expect("non-empty");
    assert!(
        plist.contains(1),
        "Bob's incoming KNOWS should contain Alice(1)"
    );

    // Verify Charlie's incoming KNOWS posting list contains Alice(1) and Bob(2)
    let charlie_in_key = encode_adj_key_reverse("KNOWS", NodeId::from_raw(3));
    let plist = ctx
        .adj_get(&charlie_in_key)
        .expect("read")
        .expect("non-empty");
    assert!(
        plist.contains(1),
        "Charlie's incoming KNOWS should contain Alice(1)"
    );
    assert!(
        plist.contains(2),
        "Charlie's incoming KNOWS should contain Bob(2)"
    );

    // Verify Matrix's incoming LIKES posting list contains Alice(1)
    let matrix_in_key = encode_adj_key_reverse("LIKES", NodeId::from_raw(4));
    let plist = ctx
        .adj_get(&matrix_in_key)
        .expect("read")
        .expect("non-empty");
    assert!(
        plist.contains(1),
        "Matrix's incoming LIKES should contain Alice(1)"
    );

    // DETACH DELETE Alice (node 1)
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Delete {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            variables: vec!["n".into()],
            detach: true,
        },
    };

    execute(&plan, &mut ctx).expect("execute DETACH DELETE");

    // Flush pending merge removes to storage
    // merge removes already flushed by execute() → mvcc_flush()

    // Verify Alice's node is gone
    assert!(read_node(&engine, 1, NodeId::from_raw(1)).is_none());

    // Verify Alice's own adj: keys are gone
    let alice_out_knows = encode_adj_key_forward("KNOWS", NodeId::from_raw(1));
    assert!(engine
        .get(Partition::Adj, &alice_out_knows)
        .expect("get")
        .is_none());

    // KEY CHECK: Bob's incoming KNOWS should NO LONGER contain Alice(1)
    let bob_in_plist = match engine.get(Partition::Adj, &bob_in_key).expect("get") {
        Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
        None => PostingList::new(),
    };
    assert!(
        !bob_in_plist.contains(1),
        "Bob's incoming KNOWS must not contain Alice(1) after DETACH DELETE"
    );

    // KEY CHECK: Charlie's incoming KNOWS should NO LONGER contain Alice(1)
    // but should STILL contain Bob(2)
    let charlie_in_plist = match engine.get(Partition::Adj, &charlie_in_key).expect("get") {
        Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
        None => PostingList::new(),
    };
    assert!(
        !charlie_in_plist.contains(1),
        "Charlie's incoming KNOWS must not contain Alice(1) after DETACH DELETE"
    );
    assert!(
        charlie_in_plist.contains(2),
        "Charlie's incoming KNOWS must still contain Bob(2)"
    );

    // KEY CHECK: Matrix's incoming LIKES should NO LONGER contain Alice(1)
    let matrix_in_plist = match engine.get(Partition::Adj, &matrix_in_key).expect("get") {
        Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
        None => PostingList::new(),
    };
    assert!(
        !matrix_in_plist.contains(1),
        "Matrix's incoming LIKES must not contain Alice(1) after DETACH DELETE"
    );
}

#[test]
fn detach_delete_cleans_incoming_edge_counterparts() {
    // Test: node with incoming edges only.
    // Bob(2)->Alice(1), Charlie(3)->Alice(1) via FOLLOWS.
    // DETACH DELETE Alice should remove Alice from Bob's and Charlie's outgoing FOLLOWS.
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Create nodes
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
        "User",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "User",
        &[("name", Value::String("Charlie".into()))],
        &mut interner,
    );

    // Bob->Alice and Charlie->Alice via FOLLOWS
    insert_edge(&engine, "FOLLOWS", 2, 1);
    insert_edge(&engine, "FOLLOWS", 3, 1);

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // DETACH DELETE Alice
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Delete {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            variables: vec!["n".into()],
            detach: true,
        },
    };

    execute(&plan, &mut ctx).expect("execute DETACH DELETE");
    // merge removes already flushed by execute() → mvcc_flush()

    // Bob's outgoing FOLLOWS should not contain Alice(1)
    let bob_out_key = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(2));
    let bob_plist = match engine.get(Partition::Adj, &bob_out_key).expect("get") {
        Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
        None => PostingList::new(),
    };
    assert!(
        !bob_plist.contains(1),
        "Bob's outgoing FOLLOWS must not contain Alice(1)"
    );

    // Charlie's outgoing FOLLOWS should not contain Alice(1)
    let charlie_out_key = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(3));
    let charlie_plist = match engine.get(Partition::Adj, &charlie_out_key).expect("get") {
        Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
        None => PostingList::new(),
    };
    assert!(
        !charlie_plist.contains(1),
        "Charlie's outgoing FOLLOWS must not contain Alice(1)"
    );
}

// ── list_edge_types unit tests ──────────────────────────────────────────

/// Empty schema → empty edge type list.
#[test]
fn list_edge_types_empty_schema() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let types = ctx.list_edge_types().expect("list_edge_types");
    assert!(types.is_empty(), "no edge types in empty schema");
}

/// Registering edge types via insert_edge makes list_edge_types return them.
#[test]
fn list_edge_types_returns_registered() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    insert_edge(&engine, "FOLLOWS", 1, 2);
    insert_edge(&engine, "LIKES", 1, 3);
    // Duplicate registration for FOLLOWS is idempotent.
    insert_edge(&engine, "FOLLOWS", 2, 3);

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let mut types = ctx.list_edge_types().expect("list_edge_types");
    types.sort();
    assert_eq!(types, vec!["FOLLOWS", "LIKES"]);
}

/// Edge type registered in mvcc_write_buffer (same-tx CREATE edge) is visible.
#[test]
fn list_edge_types_includes_write_buffer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // Write edge type directly into write buffer (simulates execute_create_edge).
    let et_key = edge_type_schema_key("WORKS_AT");
    ctx.txn
        .write_buffer_mut()
        .insert((Partition::Schema, et_key), Some(b"".to_vec()));

    let types = ctx.list_edge_types().expect("list_edge_types");
    assert!(
        types.contains(&"WORKS_AT".to_string()),
        "write-buffer edge type must be visible"
    );
}

/// DETACH DELETE with two edge types uses targeted lookup: cleans up both.
#[test]
fn detach_delete_multi_edge_type_targeted_lookup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Node 1 (Alice): has FOLLOWS and LIKES edges
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
        "User",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "User",
        &[("name", Value::String("Carol".into()))],
        &mut interner,
    );

    // Alice -[FOLLOWS]-> Bob
    insert_edge(&engine, "FOLLOWS", 1, 2);
    // Alice -[LIKES]-> Carol
    insert_edge(&engine, "LIKES", 1, 3);
    // Bob -[FOLLOWS]-> Carol  (should not be affected)
    insert_edge(&engine, "FOLLOWS", 2, 3);

    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    // DETACH DELETE Alice (node_id = 1)
    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Delete {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["User".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            variables: vec!["n".into()],
            detach: true,
        },
    };

    execute(&plan, &mut ctx).expect("DETACH DELETE Alice");

    // Bob's FOLLOWS posting list must not contain Alice(1).
    let bob_follows_out = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(2));
    let plist = engine.get(Partition::Adj, &bob_follows_out).expect("get");
    if let Some(bytes) = plist {
        let pl = PostingList::from_bytes(&bytes).expect("decode");
        assert!(
            !pl.contains(1),
            "Bob's FOLLOWS must not contain Alice after delete"
        );
    }

    // Carol's LIKES incoming posting list must not contain Alice(1).
    let carol_likes_in = encode_adj_key_reverse("LIKES", NodeId::from_raw(3));
    let plist = engine.get(Partition::Adj, &carol_likes_in).expect("get");
    if let Some(bytes) = plist {
        let pl = PostingList::from_bytes(&bytes).expect("decode");
        assert!(
            !pl.contains(1),
            "Carol's LIKES-in must not contain Alice after delete"
        );
    }

    // Bob-[FOLLOWS]->Carol edge must be untouched.
    let bob_carol = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(2));
    let plist = engine.get(Partition::Adj, &bob_carol).expect("get");
    if let Some(bytes) = plist {
        let pl = PostingList::from_bytes(&bytes).expect("decode");
        assert!(pl.contains(3), "Bob's FOLLOWS must still contain Carol(3)");
    }
}

// ── Correlated OPTIONAL MATCH detection (G004) ─────────────────────

#[test]
fn needs_correlated_non_correlated() {
    // OPTIONAL MATCH (a)-[:KNOWS]->(b) — right side introduces a, b.
    // Filter predicate only references b.age (right-side variable).
    // → NOT correlated.
    let right = LogicalOp::Filter {
        input: Box::new(LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["Person".into()],
                property_filters: vec![],
            }),
            source: "a".into(),
            edge_types: vec!["KNOWS".into()],
            target_variable: "b".into(),
            target_labels: vec![],
            edge_variable: None,
            direction: Direction::Outgoing,
            length: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        }),
        predicate: Expr::BinaryOp {
            left: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("b".into())),
                property: "age".into(),
            }),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Literal(Value::Int(30))),
        },
    };
    assert!(
        !super::needs_correlated_execution(&right),
        "b.age > 30 references only right-side variable b"
    );
}

#[test]
fn needs_correlated_yes_cross_scope() {
    // OPTIONAL MATCH (b:Person) WHERE b.age > a.age
    // Right side introduces "b". Predicate references "a" (not introduced).
    // → IS correlated.
    let right = LogicalOp::Filter {
        input: Box::new(LogicalOp::NodeScan {
            variable: "b".into(),
            labels: vec!["Person".into()],
            property_filters: vec![],
        }),
        predicate: Expr::BinaryOp {
            left: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("b".into())),
                property: "age".into(),
            }),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("a".into())),
                property: "age".into(),
            }),
        },
    };
    assert!(
        super::needs_correlated_execution(&right),
        "a.age references left-scope variable a"
    );
}

#[test]
fn needs_correlated_no_filter() {
    // OPTIONAL MATCH (a)-[:KNOWS]->(b) — no filter at all.
    // → NOT correlated.
    let right = LogicalOp::Traverse {
        input: Box::new(LogicalOp::NodeScan {
            variable: "a".into(),
            labels: vec!["Person".into()],
            property_filters: vec![],
        }),
        source: "a".into(),
        edge_types: vec!["KNOWS".into()],
        target_variable: "b".into(),
        target_labels: vec![],
        edge_variable: None,
        direction: Direction::Outgoing,
        length: None,
        target_filters: vec![],
        edge_filters: vec![],
        temporal_filter: None,
        path_variable: None,
    };
    assert!(
        !super::needs_correlated_execution(&right),
        "no filter = no correlation"
    );
}

#[test]
fn collect_expr_vars_covers_in_and_is_null() {
    // Verify collect_expr_vars extracts variables from In and IsNull.
    let expr = Expr::In {
        expr: Box::new(Expr::Variable("x".into())),
        list: Box::new(Expr::List(vec![Expr::Variable("y".into())])),
    };
    let mut vars = Vec::new();
    super::collect_expr_vars(&expr, &mut vars);
    assert!(vars.contains(&"x".to_string()));
    assert!(vars.contains(&"y".to_string()));

    let expr2 = Expr::IsNull {
        expr: Box::new(Expr::Variable("z".into())),
        negated: false,
    };
    let mut vars2 = Vec::new();
    super::collect_expr_vars(&expr2, &mut vars2);
    assert!(vars2.contains(&"z".to_string()));
}

#[test]
fn collect_expr_vars_covers_string_match() {
    let expr = Expr::StringMatch {
        expr: Box::new(Expr::Variable("a".into())),
        op: crate::cypher::ast::StringOp::StartsWith,
        pattern: Box::new(Expr::Variable("b".into())),
    };
    let mut vars = Vec::new();
    super::collect_expr_vars(&expr, &mut vars);
    assert!(vars.contains(&"a".to_string()));
    assert!(vars.contains(&"b".to_string()));
}

// -- G067: Parallel path OCC read-set tracking --

#[test]
fn g067_parallel_traversal_populates_occ_read_set() {
    // Verify that parallel traversal collects read keys into
    // the Layer-3 OccScope so OCC conflict detection works for
    // write transactions on super-nodes.
    // so OCC conflict detection works for write transactions on super-nodes.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open with oracle");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);

    // Hub node with 10 targets (threshold=5 triggers parallel)
    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("name", Value::String("Hub".into()))],
        &mut interner,
    );
    for i in 2..=11u64 {
        insert_node(
            &engine,
            1,
            i,
            "User",
            &[("name", Value::String(format!("T{i}")))],
            &mut interner,
        );
        insert_edge(&engine, "FOLLOWS", 1, i);
    }

    // Allocate a read timestamp and take a snapshot
    let read_ts = oracle.next();
    let snap = engine.snapshot_at(read_ts.as_raw());

    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = read_ts;
    ctx.mvcc_snapshot = snap;
    ctx.adaptive.parallel_threshold = 5; // trigger parallel on 10 edges

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![("name".into(), Expr::Literal(Value::String("Hub".into())))],
            }),
            source: "a".into(),
            edge_types: vec!["FOLLOWS".into()],
            direction: Direction::Outgoing,
            target_variable: "b".into(),
            target_labels: vec![],
            length: None,
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 10, "should return all 10 targets");

    // Verify parallel was triggered
    assert!(
        ctx.warnings.iter().any(|w| w.contains("parallel")),
        "expected parallel activation, got: {:?}",
        ctx.warnings,
    );

    // Verify OCC read-set contains Node keys for target nodes.
    // The source node (Hub, id=1) is read via sequential mvcc_get in NodeScan,
    // and 10 target nodes are read via parallel path — all should be tracked.
    let scope = ctx
        .txn
        .occ_scope()
        .expect("MVCC mode must have an OCC scope");

    // Verify specific target keys are tracked via the typed OCC probe
    // (builds the key internally — no raw encoder). Done before draining.
    for target_id in 2..=11u64 {
        assert!(
            scope.contains_node(1, NodeId::from_raw(target_id)),
            "target node {target_id} should be in OCC read-set",
        );
    }

    // Drain to count Node-partition entries.
    let tracked: Vec<_> = scope.drain();
    let node_read_keys: Vec<_> = tracked
        .iter()
        .filter(|(part, _)| *part == Partition::Node)
        .collect();
    // At least 10 target Node keys must be in read-set (from parallel path)
    // plus 1 for the Hub node (from sequential NodeScan).
    assert!(
        node_read_keys.len() >= 10,
        "expected ≥10 Node keys in OCC read-set (parallel targets), got {}",
        node_read_keys.len(),
    );
}

#[test]
fn g104_ensure_occ_scope_idempotent_in_mvcc_mode() {
    // ensure_occ_scope must create scope exactly once per
    // transaction and return the same handle on subsequent
    // calls — otherwise tracked keys collected before the
    // second call would be lost.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open with oracle");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();

    // First call materialises the scope and tracks one key.
    {
        let scope = ctx.ensure_occ_scope().expect("MVCC mode → Some");
        scope.track(Partition::Node, b"k1");
    }
    // Second call returns the SAME scope — k1 is still tracked.
    let scope = ctx.ensure_occ_scope().expect("still Some");
    assert!(scope.contains(Partition::Node, b"k1"));
    scope.track(Partition::Node, b"k2");
    assert_eq!(scope.tracked_count(), 2);
}

#[test]
fn mvcc_get_node_temporal_tracks_temporal_key_in_occ_scope() {
    // Critical correctness: the temporal helper must enter the
    // 25-byte temporal key (NOT the 16-byte non-temporal key) into
    // the OCC scope. A bug here would silently miss conflicts on
    // bitemporal reads.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let id = NodeId::from_raw(77);
    // Seed a temporal version so the read returns Some, via the
    // typed Layer-4 `put_temporal` (raw-encoder-free fixture).
    let rec = NodeRecord::new("E");
    seed_node_temporal(&engine, 0, id, 1234567890, &rec);

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();

    let _ = ctx
        .mvcc_get_node_temporal(0, id, 1234567890)
        .expect("get")
        .expect("Some");

    let scope = ctx.txn.occ_scope().expect("scope");
    assert!(
        scope.contains_node_temporal(0, id, 1234567890),
        "OCC scope must contain the 25-byte temporal key, not the non-temporal one",
    );
    // Cross-check: non-temporal 16-byte key for the same id must NOT
    // be tracked — temporal reads are version-specific.
    assert!(
        !scope.contains_node(0, id),
        "temporal read must NOT track the non-temporal 16-byte key",
    );
}

#[test]
fn mvcc_put_node_temporal_does_not_track_in_occ_scope() {
    // Symmetric to mvcc_put_node_does_not_track: pure temporal write
    // must NOT enter the OCC scope (RYOW for own writes — never
    // self-conflict).
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();

    let id = NodeId::from_raw(88);
    ctx.mvcc_put_node_temporal(0, id, 1000, &NodeRecord::new("E"))
        .expect("put");
    match ctx.txn.occ_scope() {
        None => { /* fine — no scope materialised on pure write */ }
        Some(scope) => assert!(
            !scope.contains_node_temporal(0, id, 1000),
            "pure temporal write must NOT enter OCC scope",
        ),
    }
}

#[test]
fn mvcc_get_node_temporal_decode_error_surfaces() {
    // Corrupt bytes at a temporal key → ExecutionError::Serialization
    // with valid_from_ms in the diagnostic, not panic.
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
    )
    .expect("open");
    let id = NodeId::from_raw(66);
    let key = coordinode_core::graph::node::encode_temporal_node_key(0, id, 4242);
    engine
        .put(Partition::Node, &key, b"definitely-not-msgpack")
        .expect("seed");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);

    let err = ctx
        .mvcc_get_node_temporal(0, id, 4242)
        .expect_err("must surface as error");
    match err {
        ExecutionError::Serialization(msg) => {
            assert!(msg.contains("66"), "diag has node id: {msg}");
            assert!(msg.contains("4242"), "diag has valid_from: {msg}");
        }
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn mvcc_temporal_handles_negative_valid_from_ms() {
    // Pre-epoch timestamps (negative valid_from) MUST round-trip
    // through the helper — encode_valid_from_sortable uses XOR-flip
    // for sortable signed encoding. Guards against a regression that
    // would silently break pre-1970 historical data.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    let id = NodeId::from_raw(99);
    let pre_epoch = -1_577_836_800_000_i64; // ~1920
    let rec = NodeRecord::new("Hist");
    ctx.mvcc_put_node_temporal(0, id, pre_epoch, &rec)
        .expect("put pre-epoch");
    let back = ctx
        .mvcc_get_node_temporal(0, id, pre_epoch)
        .expect("get")
        .expect("Some");
    assert_eq!(back.primary_label(), "Hist");
}

#[test]
fn mvcc_get_edge_props_tracks_key_in_occ_scope() {
    // Critical correctness — typed edge-prop read must enter the
    // Layer-3 OCC scope under the encoded EdgeProp key, otherwise
    // OCC misses concurrent writers on edges that a transaction
    // reads but does not write.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let src = NodeId::from_raw(1);
    let tgt = NodeId::from_raw(2);
    // Seed via the Layer-4 store. Post-ADR-040 `EdgeProperties` and the
    // executor's `Vec<(field_id, Value)>` shape serialise to identical
    // bytes through the one canonical codec, so the executor reads this
    // fixture back verbatim.
    {
        use coordinode_core::graph::edge::EdgeProperties;
        use coordinode_core::txn::write_concern::WriteConcern;
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        use coordinode_storage::engine::transaction::{CommitContext, Transaction};
        let mut props = EdgeProperties::new();
        props.set(0, Value::Int(7));
        let read_ts = oracle.next();
        let mut txn = Transaction::new(&engine, Some(&*oracle), read_ts, Some(engine.snapshot()));
        LocalEdgeStore
            .put_edge(&mut txn, "REL", src, tgt, Some(&props))
            .expect("seed");
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        txn.commit(&ctx).expect("commit seed");
    }

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();

    let _ = ctx
        .mvcc_get_edge_props("REL", src, tgt)
        .expect("get")
        .expect("Some");

    // Typed OCC-scope assertion — `contains_edge_props` builds
    // the key internally so the assertion is raw-encoder-free
    // even though the fixture seeding above is not.
    let scope = ctx.txn.occ_scope().expect("scope");
    assert!(
        scope.contains_edge_props("REL", src, tgt),
        "OCC scope must contain the encoded EdgeProp key after a typed read",
    );
}

#[test]
fn mvcc_get_edge_props_temporal_tracks_25byte_key_not_short() {
    // Temporal read must populate OCC with the per-version key,
    // NOT the non-temporal `(src, tgt)` key — otherwise concurrent
    // writers on a different version would falsely conflict (and
    // vice versa).
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let src = NodeId::from_raw(11);
    let tgt = NodeId::from_raw(22);
    // Seed a temporal edge version through the Layer-4 store (canonical
    // edgeprop codec — wire-identical to the executor, ADR-040).
    {
        use coordinode_core::graph::edge::EdgeProperties;
        use coordinode_core::txn::write_concern::WriteConcern;
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        use coordinode_storage::engine::transaction::{CommitContext, Transaction};
        let mut props = EdgeProperties::new();
        props.set(1, Value::String("v".into()));
        let read_ts = oracle.next();
        let mut txn = Transaction::new(&engine, Some(&*oracle), read_ts, Some(engine.snapshot()));
        LocalEdgeStore
            .put_edge_temporal(&mut txn, "REL", src, tgt, 5000, &props)
            .expect("seed");
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        txn.commit(&ctx).expect("commit seed");
    }

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();

    let _ = ctx
        .mvcc_get_edge_props_temporal("REL", src, tgt, 5000)
        .expect("get")
        .expect("Some");

    // Typed OCC-scope assertions — verify the temporal key is
    // tracked but the non-temporal (short) one for the same
    // pair is NOT.
    let scope = ctx.txn.occ_scope().expect("scope");
    assert!(
        scope.contains_edge_props_temporal("REL", src, tgt, 5000),
        "OCC scope must record the temporal (per-version) key",
    );
    assert!(
        !scope.contains_edge_props("REL", src, tgt),
        "OCC scope must NOT record the short non-temporal key on a temporal read",
    );
}

#[test]
fn mvcc_put_edge_props_does_not_track_in_occ_scope() {
    // Symmetric invariant to the node-side test: a pure write
    // must NOT enter the OCC scope, otherwise the put + subsequent
    // same-txn read would self-conflict at commit.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();

    let src = NodeId::from_raw(33);
    let tgt = NodeId::from_raw(44);
    let payload: Vec<(u32, Value)> = vec![(2, Value::Bool(true))];
    ctx.mvcc_put_edge_props("REL", src, tgt, &payload)
        .expect("put");

    match ctx.txn.occ_scope() {
        None => { /* no scope materialised on pure write — fine */ }
        Some(scope) => assert!(
            !scope.contains_edge_props("REL", src, tgt),
            "pure edge-prop write must NOT enter OCC scope",
        ),
    }
}

#[test]
fn mvcc_get_edge_props_decode_error_surfaces() {
    // Corrupt bytes at the EdgeProp key surface as
    // ExecutionError::Serialization with the (edge_type, src, tgt)
    // diagnostic, not panic.
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
    )
    .expect("open");
    let src = NodeId::from_raw(55);
    let tgt = NodeId::from_raw(66);
    let ep_key = encode_edgeprop_key("REL", src, tgt);
    engine
        .put(Partition::EdgeProp, &ep_key, b"this-is-not-msgpack")
        .expect("seed garbage");

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    let err = ctx
        .mvcc_get_edge_props("REL", src, tgt)
        .expect_err("garbage must surface as error");
    match err {
        ExecutionError::Serialization(msg) => {
            assert!(msg.contains("REL"), "diag has edge_type: {msg}");
            assert!(msg.contains("55"), "diag has src id: {msg}");
            assert!(msg.contains("66"), "diag has tgt id: {msg}");
        }
        other => panic!("wrong error variant: {other:?}"),
    }
}

#[test]
fn mvcc_delete_edge_props_either_dispatches_on_temporal_flag() {
    // delete_edge_props_either must tombstone exactly the keyed
    // version. The non-temporal-key write at (src, tgt) survives
    // when we delete temporally at (src, tgt, vf), and vice versa.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let src = NodeId::from_raw(1);
    let tgt = NodeId::from_raw(2);
    let payload: Vec<(u32, Value)> = vec![(0, Value::Int(42))];

    // Seed BOTH the non-temporal AND a temporal version at vf=5000.
    {
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        ctx.mvcc_oracle = Some(&oracle);
        ctx.mvcc_read_ts = oracle.next();
        ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();
        ctx.mvcc_put_edge_props_either("REL", src, tgt, None, &payload)
            .expect("put non-temporal");
        ctx.mvcc_put_edge_props_either("REL", src, tgt, Some(5000), &payload)
            .expect("put temporal");
        ctx.mvcc_flush().expect("flush seed");
    }
    // Delete only the temporal version.
    {
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        ctx.mvcc_oracle = Some(&oracle);
        ctx.mvcc_read_ts = oracle.next();
        ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();
        ctx.mvcc_delete_edge_props_either("REL", src, tgt, Some(5000))
            .expect("delete temporal");
        ctx.mvcc_flush().expect("flush delete");
    }
    // Verify: temporal v gone; non-temporal still readable.
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    assert!(
        ctx.mvcc_get_edge_props_either("REL", src, tgt, Some(5000))
            .expect("read temporal")
            .is_none(),
        "temporal version must be tombstoned",
    );
    assert!(
        ctx.mvcc_get_edge_props_either("REL", src, tgt, None)
            .expect("read non-temporal")
            .is_some(),
        "non-temporal key must remain — delete_either dispatched on Some(vf)",
    );
}

#[test]
fn mvcc_get_edge_props_round_trip_through_put() {
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    let src = NodeId::from_raw(10);
    let tgt = NodeId::from_raw(20);
    let fid_weight = ctx.interner.intern("weight");
    let fid_label = ctx.interner.intern("label");
    let payload: Vec<(u32, Value)> = vec![
        (fid_weight, Value::Float(0.85)),
        (fid_label, Value::String("close-friend".into())),
    ];

    ctx.mvcc_put_edge_props("KNOWS", src, tgt, &payload)
        .expect("put");
    // RYOW read.
    let back = ctx
        .mvcc_get_edge_props("KNOWS", src, tgt)
        .expect("get")
        .expect("Some");
    assert_eq!(back.len(), 2);
    assert!(back
        .iter()
        .any(|(fid, v)| *fid == fid_weight
            && matches!(v, Value::Float(f) if (*f - 0.85).abs() < 1e-9)));
    assert!(back
        .iter()
        .any(|(fid, v)| *fid == fid_label && matches!(v, Value::String(s) if s == "close-friend")));

    // Reverse direction must NOT see the entry — key includes (src, tgt) order.
    let reverse = ctx.mvcc_get_edge_props("KNOWS", tgt, src).expect("rev");
    assert!(
        reverse.is_none(),
        "edge props keyed by (src, tgt) — reverse order is a distinct key",
    );
}

#[test]
fn mvcc_get_edge_props_missing_returns_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    let res = ctx
        .mvcc_get_edge_props("NONE", NodeId::from_raw(1), NodeId::from_raw(2))
        .expect("get");
    assert!(res.is_none());
}

#[test]
fn upsert_on_match_concurrent_write_is_caught_by_layer3_occ() {
    // R165 S6 removed the manual byte-CAS pre-flight from
    // execute_merge. Layer-3 OCC must now catch the same
    // "concurrent writer modified a matched node between MATCH
    // and SET" scenario at commit time via has_write_after.
    //
    // Scenario: txn reads node `k`, then a sibling txn writes to
    // `k`, then the original txn writes (independent key) and
    // flushes — the OCC scope tracked `k` during the read, so
    // validate_occ at flush must surface ExecutionError::Conflict.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    // Seed a node so the simulated MATCH read returns Some.
    let id = NodeId::from_raw(700);
    let seed = NodeRecord::new("U");
    seed_node_record(&engine, 0, id, &seed);

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    // MATCH-phase read: populates the OCC scope with `k`.
    let _ = ctx.mvcc_get_node(0, id).expect("match read").expect("Some");

    // Concurrent writer modifies the same key out-of-band.
    // Stamps a fresh seqno that is necessarily > mvcc_read_ts.
    let mut altered = NodeRecord::new("U");
    altered.set(ctx.interner.intern("name"), Value::String("Bob".into()));
    seed_node_record(&engine, 0, id, &altered);

    // ON MATCH SET: buffer a write on an UNRELATED key so the txn
    // is not read-only and flush actually runs OCC validation.
    let other = NodeId::from_raw(701);
    ctx.mvcc_put_node(0, other, &NodeRecord::new("Other"))
        .expect("unrelated put");

    let err = ctx
        .mvcc_flush()
        .expect_err("flush must surface the OCC conflict on the MATCH-read key");
    match err {
        ExecutionError::Conflict(msg) => {
            assert!(
                msg.contains("OCC") || msg.contains("conflict"),
                "conflict message expected: {msg}",
            );
        }
        other => panic!("expected ExecutionError::Conflict, got {other:?}"),
    }
}

#[test]
fn mvcc_get_node_either_dispatches_on_temporal_flag() {
    // The runtime-branching helper must dispatch correctly:
    //   None         → 16-byte non-temporal key
    //   Some(vf)     → 25-byte temporal key
    // Cross-contamination (e.g. non-temporal write read as temporal)
    // would silently return wrong data — pin both paths.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    let id = NodeId::from_raw(150);
    let nt = NodeRecord::new("NT");
    let tmp = NodeRecord::new("T");
    // Write non-temporal at id; write temporal at same id, vf=5000.
    ctx.mvcc_put_node_either(0, id, None, &nt).expect("put nt");
    ctx.mvcc_put_node_either(0, id, Some(5000), &tmp)
        .expect("put t");

    // Non-temporal read must surface NT, not T.
    let read_nt = ctx
        .mvcc_get_node_either(0, id, None)
        .expect("read nt")
        .expect("Some");
    assert_eq!(read_nt.primary_label(), "NT");
    // Temporal read at vf=5000 surfaces T.
    let read_t = ctx
        .mvcc_get_node_either(0, id, Some(5000))
        .expect("read t")
        .expect("Some");
    assert_eq!(read_t.primary_label(), "T");
    // Temporal read at a vf we did not write to is None.
    let read_miss = ctx
        .mvcc_get_node_either(0, id, Some(9999))
        .expect("read miss");
    assert!(read_miss.is_none());
}

#[test]
fn mvcc_delete_node_temporal_ryow_within_txn() {
    // Same-transaction delete of a temporal version must be visible
    // to subsequent same-txn reads (RYOW for temporal tombstones).
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    // Seed v@1000 outside of the transaction-under-test.
    let id = NodeId::from_raw(44);
    let rec = NodeRecord::new("Tx");
    seed_node_temporal(&engine, 0, id, 1000, &rec);

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    assert!(ctx.mvcc_get_node_temporal(0, id, 1000).unwrap().is_some());
    ctx.mvcc_delete_node_temporal(0, id, 1000)
        .expect("delete in-txn");
    assert!(
        ctx.mvcc_get_node_temporal(0, id, 1000).unwrap().is_none(),
        "RYOW: in-txn temporal tombstone visible to subsequent read",
    );
}

#[test]
fn mvcc_temporal_handles_i64_extreme_valid_from() {
    // Sortable encoding (encode_valid_from_sortable XOR-flip) must
    // handle i64::MIN and i64::MAX without corruption. Guards
    // against regressions in the boundary handling.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    let id = NodeId::from_raw(45);
    let min_rec = NodeRecord::new("MinEdge");
    let max_rec = NodeRecord::new("MaxEdge");

    ctx.mvcc_put_node_temporal(0, id, i64::MIN, &min_rec)
        .expect("put MIN");
    ctx.mvcc_put_node_temporal(0, id, i64::MAX, &max_rec)
        .expect("put MAX");

    let back_min = ctx
        .mvcc_get_node_temporal(0, id, i64::MIN)
        .expect("get MIN")
        .expect("Some");
    let back_max = ctx
        .mvcc_get_node_temporal(0, id, i64::MAX)
        .expect("get MAX")
        .expect("Some");
    assert_eq!(back_min.primary_label(), "MinEdge");
    assert_eq!(back_max.primary_label(), "MaxEdge");
}

#[test]
fn mvcc_temporal_keys_isolated_per_node_id_at_same_valid_from() {
    // Two different node_ids at identical valid_from must NOT
    // collide. The temporal key includes node_id, so this is the
    // expected behaviour — but a regression in key layout (e.g.
    // accidentally dropping the id byte block) would silently
    // collapse them. Pin it.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    let id_a = NodeId::from_raw(100);
    let id_b = NodeId::from_raw(200);
    let vf = 9999;
    ctx.mvcc_put_node_temporal(0, id_a, vf, &NodeRecord::new("A"))
        .expect("put A");
    ctx.mvcc_put_node_temporal(0, id_b, vf, &NodeRecord::new("B"))
        .expect("put B");

    let back_a = ctx
        .mvcc_get_node_temporal(0, id_a, vf)
        .expect("get A")
        .expect("Some");
    let back_b = ctx
        .mvcc_get_node_temporal(0, id_b, vf)
        .expect("get B")
        .expect("Some");
    assert_eq!(back_a.primary_label(), "A");
    assert_eq!(back_b.primary_label(), "B");
}

#[test]
fn mvcc_get_node_temporal_round_trip() {
    // Write a versioned record at valid_from=1000, read back via
    // typed helper, verify per-version key isolation (write at 1000
    // does NOT show up when reading at 2000).
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    let id = NodeId::from_raw(20);
    let mut rec_v1 = NodeRecord::new("Event");
    let name_fid = ctx.interner.intern("name");
    rec_v1.set(name_fid, Value::String("v1".into()));

    ctx.mvcc_put_node_temporal(0, id, 1000, &rec_v1)
        .expect("put v1");
    // RYOW: read back same version sees v1.
    let read_v1 = ctx
        .mvcc_get_node_temporal(0, id, 1000)
        .expect("get v1")
        .expect("Some");
    assert_eq!(read_v1.get(name_fid), Some(&Value::String("v1".into())));
    // Read at a DIFFERENT valid_from returns None — per-version key
    // isolation (writes do not bleed across versions).
    let read_v2 = ctx.mvcc_get_node_temporal(0, id, 2000).expect("get v2");
    assert!(
        read_v2.is_none(),
        "per-version key isolation: write at 1000 must not show at 2000",
    );
}

#[test]
fn mvcc_temporal_close_then_open_two_versions() {
    // The bitemporal close-current + open-new pair: write v1 at
    // valid_from=1000, then close it (set valid_to=2000 at SAME
    // key) + open v2 at valid_from=2000. Both versions remain
    // queryable at their respective per-version keys.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    let id = NodeId::from_raw(21);
    let vf_fid = ctx.interner.intern("valid_from");
    let vt_fid = ctx.interner.intern("valid_to");

    // v1 open at 1000.
    let mut rec_v1 = NodeRecord::new("Event");
    rec_v1.set(vf_fid, Value::Int(1000));
    ctx.mvcc_put_node_temporal(0, id, 1000, &rec_v1)
        .expect("put v1");

    // Close v1: set valid_to=2000 at same per-version key.
    let mut closed_v1 = rec_v1.clone();
    closed_v1.set(vt_fid, Value::Int(2000));
    ctx.mvcc_put_node_temporal(0, id, 1000, &closed_v1)
        .expect("close v1");

    // Open v2 at fresh per-version key valid_from=2000.
    let mut rec_v2 = NodeRecord::new("Event");
    rec_v2.set(vf_fid, Value::Int(2000));
    ctx.mvcc_put_node_temporal(0, id, 2000, &rec_v2)
        .expect("put v2");

    // Read v1 — sees the closed version (valid_to=2000 present).
    let read_v1 = ctx
        .mvcc_get_node_temporal(0, id, 1000)
        .expect("get v1")
        .expect("Some");
    assert_eq!(
        read_v1.get(vt_fid),
        Some(&Value::Int(2000)),
        "v1 must carry the close (valid_to=2000)",
    );
    // Read v2 — sees the open version (no valid_to).
    let read_v2 = ctx
        .mvcc_get_node_temporal(0, id, 2000)
        .expect("get v2")
        .expect("Some");
    assert!(
        !read_v2.props.contains_key(&vt_fid),
        "v2 is open — must not carry valid_to",
    );
}

#[test]
fn mvcc_delete_node_temporal_tombstones_specific_version() {
    // Tombstone version at valid_from=1000 must hide that version
    // only — other versions remain readable.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let id = NodeId::from_raw(22);
    let rec = NodeRecord::new("Event");

    // Seed two versions in a flush'd txn.
    {
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        ctx.mvcc_oracle = Some(&oracle);
        ctx.mvcc_read_ts = oracle.next();
        ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();
        ctx.mvcc_put_node_temporal(0, id, 1000, &rec).expect("v1");
        ctx.mvcc_put_node_temporal(0, id, 2000, &rec).expect("v2");
        ctx.mvcc_flush().expect("flush seed");
    }
    // Delete v1 in a fresh txn.
    {
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        ctx.mvcc_oracle = Some(&oracle);
        ctx.mvcc_read_ts = oracle.next();
        ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();
        ctx.mvcc_delete_node_temporal(0, id, 1000)
            .expect("delete v1");
        ctx.mvcc_flush().expect("flush delete");
    }
    // Verify: v1 gone, v2 intact.
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    assert!(
        ctx.mvcc_get_node_temporal(0, id, 1000)
            .expect("get v1")
            .is_none(),
        "v1 tombstoned",
    );
    assert!(
        ctx.mvcc_get_node_temporal(0, id, 2000)
            .expect("get v2")
            .is_some(),
        "v2 untouched",
    );
}

#[test]
fn mvcc_put_node_does_not_track_in_occ_scope() {
    // OCC tracks READS, not writes. A pure mvcc_put_node call
    // must NOT enter the OCC scope (writes are flushed via
    // mvcc_write_buffer, not validated against read-set). If
    // they DID enter the scope, every put would self-conflict
    // when paired with a later read of the same key in the same
    // transaction.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();

    let id = NodeId::from_raw(33);
    let rec = NodeRecord::new("Probe");
    ctx.mvcc_put_node(0, id, &rec).expect("put");

    match ctx.txn.occ_scope() {
        None => { /* fine — pure write created no scope */ }
        Some(scope) => assert!(
            !scope.contains_node(0, id),
            "pure write must NOT enter OCC scope — would self-conflict on later read",
        ),
    }
}

#[test]
fn mvcc_get_node_tracks_in_occ_scope() {
    // Critical correctness: typed read must enter the Layer-3
    // OCC scope (otherwise OCC conflict detection misses node
    // dependencies and writes appear to commit cleanly even
    // when another transaction modified the read node).
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open with oracle");
    // Seed a node via the typed Layer-4 store.
    let id = NodeId::from_raw(99);
    let seed = NodeRecord::new("Probe");
    seed_node_record(&engine, 0, id, &seed);

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();

    let _ = ctx.mvcc_get_node(0, id).expect("get").expect("Some");
    // OCC scope must contain the encoded node key.
    let scope = ctx.txn.occ_scope().expect("MVCC mode → scope present");
    assert!(
        scope.contains_node(0, id),
        "typed read must populate OCC scope under Node partition",
    );
}

#[test]
fn mvcc_get_node_decode_error_surfaces_as_serialization_error() {
    // Corrupt bytes in the partition must not panic — they must
    // propagate as ExecutionError::Serialization through the
    // typed helper's decode boundary.
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
    )
    .expect("open");
    let id = NodeId::from_raw(55);
    let key = coordinode_core::graph::node::encode_node_key(0, id);
    // Plant garbage — not a valid MessagePack NodeRecord.
    engine
        .put(Partition::Node, &key, b"this-is-not-msgpack")
        .expect("seed garbage");

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    let err = ctx
        .mvcc_get_node(0, id)
        .expect_err("garbage must surface as error");
    match err {
        ExecutionError::Serialization(msg) => {
            assert!(
                msg.contains("55"),
                "error message includes the node id for diagnostics: {msg}",
            );
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn mvcc_delete_node_then_get_returns_none_within_txn() {
    // Same-transaction delete must be visible to subsequent
    // typed reads (RYOW for tombstones).
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open");
    // Seed.
    let id = NodeId::from_raw(11);
    let rec = NodeRecord::new("Item");
    seed_node_record(&engine, 0, id, &rec);

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    // Initial read sees the seeded record.
    assert!(ctx.mvcc_get_node(0, id).expect("get").is_some());
    // Delete in-txn.
    ctx.mvcc_delete_node(0, id).expect("delete");
    // RYOW: subsequent same-txn read sees the tombstone.
    assert!(
        ctx.mvcc_get_node(0, id).expect("get").is_none(),
        "RYOW must surface the in-txn tombstone",
    );
}

#[test]
fn mvcc_get_node_round_trip_through_put() {
    // Write a NodeRecord via the typed helper, read it back via
    // the typed helper — values match end-to-end. Confirms encode +
    // serialize + mvcc_put / mvcc_get + decode wire up correctly.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open with oracle");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    let id = NodeId::from_raw(7);
    let mut rec = NodeRecord::new("User");
    let fid = ctx.interner.intern("name");
    rec.set(fid, Value::String("Alice".into()));

    ctx.mvcc_put_node(1, id, &rec).expect("put");
    // RYOW: same-txn read sees the buffered write.
    let fetched = ctx.mvcc_get_node(1, id).expect("get").expect("Some — RYOW");
    assert_eq!(fetched.primary_label(), "User");
    assert_eq!(
        fetched.get(fid),
        Some(&Value::String("Alice".into())),
        "value field round-trips",
    );

    // Flush so the engine actually holds the record.
    ctx.mvcc_flush().expect("flush");
    // Re-open a fresh ctx — confirm post-flush durability through
    // the typed reader (legacy mode, no oracle).
    let mut ctx2 = make_ctx(&engine, &mut interner, &allocator);
    let post_flush = ctx2.mvcc_get_node(1, id).expect("get").expect("Some");
    assert_eq!(post_flush.primary_label(), "User");
}

#[test]
fn mvcc_get_node_missing_returns_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    // Legacy mode, no oracle — typed read goes through engine.get.
    let res = ctx.mvcc_get_node(0, NodeId::from_raw(99)).expect("get ok");
    assert!(res.is_none());
}

#[test]
fn mvcc_delete_node_tombstones_node_key() {
    // After mvcc_delete_node + flush, subsequent reads return None.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open with oracle");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    let id = NodeId::from_raw(42);
    let rec = NodeRecord::new("Tag");
    ctx.mvcc_put_node(0, id, &rec).expect("put");
    ctx.mvcc_flush().expect("flush 1");

    // Fresh ctx for the delete — would conflict otherwise.
    let mut ctx2 = make_ctx(&engine, &mut interner, &allocator);
    ctx2.mvcc_oracle = Some(&oracle);
    ctx2.mvcc_read_ts = oracle.next();
    ctx2.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();
    ctx2.mvcc_delete_node(0, id).expect("delete");
    ctx2.mvcc_flush().expect("flush 2");

    let mut ctx3 = make_ctx(&engine, &mut interner, &allocator);
    let after_delete = ctx3.mvcc_get_node(0, id).expect("get ok");
    assert!(after_delete.is_none(), "tombstone hides the row");
}

#[test]
fn mvcc_flush_idempotent_under_second_call() {
    // Calling mvcc_flush twice on the same context must not
    // re-apply writes (otherwise w:0 fan-out triggers spurious
    // duplicate puts on retry / control-flow seams).
    // Expected semantics:
    //   1st call: drains write_buffer + merge buffers → commit_ts.
    //   2nd call: buffers empty → read-only path → returns
    //   Ok(Some(read_ts)), no further engine writes.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open with oracle");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();
    ctx.write_concern = coordinode_core::txn::write_concern::WriteConcern::w0();

    ctx.mvcc_put(Partition::Node, b"flush_key", b"v1")
        .expect("put");
    // First flush: drains buffer, returns commit_ts (Some).
    let first = ctx.mvcc_flush().expect("first flush ok");
    assert!(first.is_some(), "first flush must return commit_ts");
    // Engine sees the write after the first flush.
    assert_eq!(
        engine
            .get(Partition::Node, b"flush_key")
            .expect("get")
            .as_deref(),
        Some(b"v1".as_slice()),
    );
    // Second flush: buffer drained → read-only path → no engine
    // mutation. Capture state before and after.
    let second = ctx.mvcc_flush().expect("second flush ok");
    assert!(
        second.is_some(),
        "second flush returns read_ts (read-only path)"
    );
    // Engine state unchanged — second flush must be a no-op write.
    assert_eq!(
        engine
            .get(Partition::Node, b"flush_key")
            .expect("get")
            .as_deref(),
        Some(b"v1".as_slice()),
    );
}

#[test]
fn mvcc_flush_read_only_returns_read_ts_without_commit() {
    // Read-only transaction → flush takes the early-return path
    // (no oracle.next() allocation, no engine writes). Returns
    // the read_ts itself.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open with oracle");
    engine.put(Partition::Node, b"k", b"v").expect("seed");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    let read_ts = oracle.next();
    ctx.mvcc_read_ts = read_ts;
    // Read-only — only mvcc_get, no mvcc_put.
    let _ = ctx.mvcc_get(Partition::Node, b"k").expect("get");
    let result = ctx.mvcc_flush().expect("flush ok");
    assert_eq!(
        result,
        Some(read_ts),
        "read-only flush returns the original read_ts unchanged",
    );
}

#[test]
fn ryow_read_does_not_track_in_occ_scope() {
    // Reading-your-own-write (write_buffer hit) must NOT enter
    // the OCC scope — otherwise your own writes trigger
    // self-conflict at commit (a transaction that puts a key
    // and then reads it would always abort).
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open with oracle");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = oracle.next();

    // Write into the buffer (this is our own transaction's write).
    ctx.mvcc_put(Partition::Node, b"own_key", b"own_value")
        .expect("put");
    // Read it back — RYOW hit, returns from write_buffer.
    let v = ctx.mvcc_get(Partition::Node, b"own_key").expect("get");
    assert_eq!(v.as_deref(), Some(b"own_value".as_slice()));
    // The Layer-3 scope must NOT contain this key.
    // Two valid post-states:
    //   - no scope materialised at all (write-only path didn't
    //     trip ensure_occ_scope from the read side, since the
    //     read returned before the track call), OR
    //   - scope exists but does not contain own_key.
    match ctx.txn.occ_scope() {
        None => { /* fine — no scope means no tracking happened */ }
        Some(scope) => assert!(
            !scope.contains(Partition::Node, b"own_key"),
            "RYOW key must not be tracked in OCC scope",
        ),
    }
}

#[test]
fn legacy_mode_prefix_scan_does_not_materialise_scope() {
    // Without an MVCC oracle, mvcc_prefix_scan must not create
    // an OCC scope — there is no conflict detection in legacy
    // mode and any scope would be dead weight.
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
    )
    .expect("open");
    engine.put(Partition::Node, b"k1", b"v1").expect("seed");
    engine.put(Partition::Node, b"k2", b"v2").expect("seed");

    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    // ctx.mvcc_oracle remains None — legacy mode.

    let results = ctx.mvcc_prefix_scan(Partition::Node, b"k").expect("scan");
    assert_eq!(results.len(), 2);
    assert!(
        ctx.txn.occ_scope().is_none(),
        "legacy mode must NOT materialise an OCC scope",
    );
}

#[test]
fn g104_ensure_occ_scope_returns_none_in_legacy_mode() {
    // Legacy mode (no MVCC oracle) → no OCC scope, no conflict
    // detection. Calling ensure_occ_scope must be safe and
    // return None.
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
    )
    .expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    // ctx.mvcc_oracle stays None.
    assert!(ctx.ensure_occ_scope().is_none());
    assert!(ctx.txn.occ_scope().is_none(), "no scope materialised");
}

#[test]
fn g067_parallel_occ_detects_conflict_on_target_node() {
    // End-to-end: parallel traversal reads target nodes, concurrent write
    // modifies one target, OCC conflict detection catches it.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open with oracle");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);

    // Hub + 10 targets
    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("name", Value::String("Hub".into()))],
        &mut interner,
    );
    for i in 2..=11u64 {
        insert_node(
            &engine,
            1,
            i,
            "User",
            &[("name", Value::String(format!("T{i}")))],
            &mut interner,
        );
        insert_edge(&engine, "FOLLOWS", 1, i);
    }

    // T1: Start read transaction
    let read_ts = oracle.next();
    let snap = engine.snapshot_at(read_ts.as_raw());

    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = read_ts;
    ctx.mvcc_snapshot = snap;
    ctx.adaptive.parallel_threshold = 5;

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["User".into()],
                property_filters: vec![("name".into(), Expr::Literal(Value::String("Hub".into())))],
            }),
            source: "a".into(),
            edge_types: vec!["FOLLOWS".into()],
            direction: Direction::Outgoing,
            target_variable: "b".into(),
            target_labels: vec![],
            length: None,
            edge_variable: None,
            target_filters: vec![],
            edge_filters: vec![],
            temporal_filter: None,
            path_variable: None,
        },
    };

    let result = execute(&plan, &mut ctx).expect("execute");
    assert_eq!(result.len(), 10);

    // T2: Concurrent transaction modifies target node 5 (after T1's read_ts)
    let _write_ts = oracle.next();
    let mut modified_record = NodeRecord::new("User");
    // Use field_id 0 directly — "name" was interned first by insert_node,
    // so it has id 0. Avoids borrowing interner while ctx is alive.
    modified_record.set(0, Value::String("T5-modified".into()));
    seed_node_record(&engine, 1, NodeId::from_raw(5), &modified_record);

    // T1: Add a dummy write so mvcc_flush doesn't skip conflict check
    // (read-only transactions return early without OCC check)
    let dummy_key = coordinode_core::graph::node::encode_node_key(1, NodeId::from_raw(999));
    ctx.txn
        .write_buffer_mut()
        .insert((Partition::Node, dummy_key), Some(b"dummy".to_vec()));

    // T1: OCC conflict check via mvcc_flush should detect the write to target 5
    let conflict = ctx.mvcc_flush();
    assert!(
        conflict.is_err(),
        "OCC should detect conflict on target node 5 modified after read_ts",
    );
    let err_msg = format!("{}", conflict.unwrap_err());
    assert!(
        err_msg.contains("OCC conflict"),
        "expected OCC conflict error, got: {err_msg}",
    );
}

/// Regression: DETACH DELETE must purge the deleted node's own adjacency
/// posting lists through the MVCC write buffer (so they roll back with the
/// transaction), NOT via an immediate `engine.delete`. Before the fix the
/// DETACH path deleted the posting list directly on disk, so an OCC
/// conflict (or any error) raised at `mvcc_flush` rolled back the buffered
/// node-delete while the adjacency list stayed gone — orphaning the
/// surviving peer's edges.
///
/// The discriminating observation: between `execute` and `mvcc_flush` the
/// on-disk posting list is still present (buffered tombstone), while a
/// read-your-own-writes probe inside the transaction sees it as gone. The
/// immediate-delete bug fails the "still on disk before flush" assertion.
#[test]
fn detach_delete_adj_purge_is_buffered_not_immediate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(
        &coordinode_storage::engine::config::StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
        ]),
        oracle.clone(),
    )
    .expect("open with oracle");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::new(0);

    // Alice (1) --KNOWS--> Bob (2)
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

    let alice_fwd = encode_adj_key_forward("KNOWS", NodeId::from_raw(1));
    assert!(
        engine
            .get(Partition::Adj, &alice_fwd)
            .expect("get")
            .is_some(),
        "precondition: Alice's forward posting list exists on disk",
    );

    // Transaction: DETACH DELETE Alice. Do NOT flush — observe buffer state.
    let read_ts = oracle.next();
    let snap = engine.snapshot_at(read_ts.as_raw());
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.mvcc_oracle = Some(&oracle);
    ctx.mvcc_read_ts = read_ts;
    ctx.mvcc_snapshot = snap;

    let plan = LogicalPlan {
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
        root: LogicalOp::Delete {
            input: Box::new(LogicalOp::NodeScan {
                variable: "n".into(),
                labels: vec!["Person".into()],
                property_filters: vec![(
                    "name".into(),
                    Expr::Literal(Value::String("Alice".into())),
                )],
            }),
            variables: vec!["n".into()],
            detach: true,
        },
    };
    // Use `execute_op` (not `execute`, which auto-commits) so the
    // transaction's buffered writes are observable before flush.
    let deleted = execute_op(&plan.root, &mut ctx).expect("detach delete execute");
    assert_eq!(deleted.len(), 1, "DETACH DELETE matched Alice");

    // The regression assertion: before flush the on-disk posting list is
    // STILL present because the purge is MVCC-buffered. The immediate-delete
    // bug (`engine.delete`) removes it here, so an OCC conflict or any error
    // at flush would roll back the buffered node-delete while the adjacency
    // stayed gone — orphaning Bob.
    assert!(
        engine
            .get(Partition::Adj, &alice_fwd)
            .expect("get")
            .is_some(),
        "DETACH purge must be MVCC-buffered, not an immediate engine.delete \
             — otherwise an aborted transaction orphans the surviving peer",
    );
    assert!(
        ctx.txn.buffered(Partition::Adj, &alice_fwd).is_some(),
        "DETACH purge must buffer an Adj tombstone in the MVCC write buffer",
    );

    // After commit the deletion lands on disk, proving the purge was real
    // (buffered, not skipped).
    ctx.mvcc_flush().expect("flush");
    assert!(
        engine
            .get(Partition::Adj, &alice_fwd)
            .expect("get")
            .is_none(),
        "post-commit: Alice's forward posting list is removed",
    );
    assert!(
        read_node(&engine, 1, NodeId::from_raw(1)).is_none(),
        "post-commit: Alice's node record is removed",
    );
}

// --- CREATE INDEX / DROP INDEX DDL integration tests (R-API2) ---

/// Helper: build an ExecutionContext with the btree_index_registry wired in.
fn make_ctx_with_btree<'a>(
    engine: &'a StorageEngine,
    interner: &'a mut FieldInterner,
    allocator: &'a NodeIdAllocator,
    registry: &'a crate::index::IndexRegistry,
) -> ExecutionContext<'a> {
    ExecutionContext {
        btree_index_registry: Some(registry),
        ..make_ctx(engine, interner, allocator)
    }
}

#[test]
fn create_index_registers_and_backfills() {
    // Verify that CREATE INDEX registers the index in the registry and backfills
    // existing nodes.
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let registry = crate::index::IndexRegistry::new();

    // Pre-condition: no index for User.name yet.
    assert!(registry.get("user_name_idx").is_none());

    let mut ctx = make_ctx_with_btree(&engine, &mut interner, &allocator, &registry);

    let result = execute_op(
        &LogicalOp::CreateIndex {
            name: "user_name_idx".to_string(),
            label: "User".to_string(),
            property: "name".to_string(),
            unique: false,
            sparse: false,
            filter: None,
        },
        &mut ctx,
    )
    .expect("CREATE INDEX failed");

    // Should return one row with index metadata.
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].get("index"),
        Some(&Value::String("user_name_idx".to_string()))
    );

    // Registry should now contain the new index.
    assert!(
        registry.get("user_name_idx").is_some(),
        "index should be registered after CREATE INDEX"
    );

    // Backfill count: setup_test_graph inserts 3 User nodes and 1 Post.
    // All 3 Users have a 'name' property so nodes_indexed should be >= 1.
    let nodes_indexed = match result[0].get("nodes_indexed") {
        Some(Value::Int(n)) => *n,
        other => panic!("expected nodes_indexed as Int, got {other:?}"),
    };
    assert!(
        nodes_indexed >= 1,
        "expected at least 1 node backfilled, got {nodes_indexed}"
    );
}

#[test]
fn create_unique_index_enforces_constraint_on_insert() {
    // After CREATE UNIQUE INDEX, inserting a node with a duplicate property value
    // must return UniqueViolation.
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let registry = crate::index::IndexRegistry::new();

    // Register a unique index on User.name (skip backfill — insert two fresh nodes).
    let unique_def = crate::index::IndexDefinition::btree("u_name", "User", "name").unique();
    registry
        .register(&engine, unique_def)
        .expect("register unique index");

    let mut ctx = make_ctx_with_btree(&engine, &mut interner, &allocator, &registry);

    // First insert: should succeed.
    let r1 = execute_op(
        &LogicalOp::CreateNode {
            input: None,
            variable: None,
            labels: vec!["User".to_string()],
            properties: vec![(
                "name".to_string(),
                crate::cypher::ast::Expr::Literal(Value::String("UniqueUser".into())),
            )],
        },
        &mut ctx,
    );
    assert!(r1.is_ok(), "first insert should succeed");

    // Second insert with same 'name' value: must fail with unique violation.
    let r2 = execute_op(
        &LogicalOp::CreateNode {
            input: None,
            variable: None,
            labels: vec!["User".to_string()],
            properties: vec![(
                "name".to_string(),
                crate::cypher::ast::Expr::Literal(Value::String("UniqueUser".into())),
            )],
        },
        &mut ctx,
    );
    assert!(
        r2.is_err(),
        "second insert with duplicate name should fail with unique constraint violation"
    );
    let err_msg = format!("{}", r2.unwrap_err());
    assert!(
        err_msg.to_lowercase().contains("unique") || err_msg.to_lowercase().contains("constraint"),
        "error should mention unique constraint, got: {err_msg}"
    );
}

#[test]
fn drop_index_removes_from_registry() {
    // CREATE then DROP should leave the registry empty for that index name.
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let registry = crate::index::IndexRegistry::new();

    // Pre-register an index.
    let def = crate::index::IndexDefinition::btree("to_drop", "User", "age");
    registry.register(&engine, def).expect("register");
    assert!(registry.get("to_drop").is_some());

    let mut ctx = make_ctx_with_btree(&engine, &mut interner, &allocator, &registry);

    let result = execute_op(
        &LogicalOp::DropIndex {
            name: "to_drop".to_string(),
        },
        &mut ctx,
    )
    .expect("DROP INDEX failed");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].get("dropped"), Some(&Value::Bool(true)));

    // Index should be gone from registry.
    assert!(
        registry.get("to_drop").is_none(),
        "index should be absent after DROP INDEX"
    );
}

#[test]
fn drop_index_not_found_returns_error() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let registry = crate::index::IndexRegistry::new();
    let mut ctx = make_ctx_with_btree(&engine, &mut interner, &allocator, &registry);

    let result = execute_op(
        &LogicalOp::DropIndex {
            name: "nonexistent".to_string(),
        },
        &mut ctx,
    );
    assert!(
        result.is_err(),
        "DROP INDEX on nonexistent index should fail"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("nonexistent") || err_msg.contains("not found"),
        "error should mention missing index, got: {err_msg}"
    );
}

#[test]
fn create_index_duplicate_name_returns_error() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let registry = crate::index::IndexRegistry::new();

    // Register once.
    let def = crate::index::IndexDefinition::btree("dup_idx", "User", "age");
    registry.register(&engine, def).expect("register");

    let mut ctx = make_ctx_with_btree(&engine, &mut interner, &allocator, &registry);

    // Attempt to CREATE INDEX with the same name again.
    let result = execute_op(
        &LogicalOp::CreateIndex {
            name: "dup_idx".to_string(),
            label: "User".to_string(),
            property: "age".to_string(),
            unique: false,
            sparse: false,
            filter: None,
        },
        &mut ctx,
    );
    assert!(result.is_err(), "duplicate CREATE INDEX should fail");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("dup_idx") || err_msg.contains("already exists"),
        "error should mention duplicate index, got: {err_msg}"
    );
}

/// Regression test (R-API2): after CREATE INDEX, EXPLAIN must show IndexScan
/// instead of NodeScan for a matching WHERE clause.
///
/// Without `optimize_index_selection`, MATCH (n:User) WHERE n.name = "Alice"
/// produces `Filter(NodeScan)`. After registering the index and running the
/// optimizer, the plan must be rewritten to `IndexScan`.
#[test]
fn explain_shows_index_scan_after_create_index() {
    use crate::planner::optimize_index_selection;
    use coordinode_core::graph::types::VectorConsistencyMode;

    let registry = crate::index::IndexRegistry::new();

    // Register a B-tree index on User.name (no storage needed for planner test).
    // We skip storage-backed register and use register_in_memory directly.
    let def = crate::index::IndexDefinition::btree("user_name_idx", "User", "name");
    registry.register_in_memory(def);

    // Build Filter(NodeScan) — what the planner emits BEFORE optimization.
    let node_scan = LogicalOp::NodeScan {
        variable: "n".to_string(),
        labels: vec!["User".to_string()],
        property_filters: vec![],
    };
    let filter_plan = LogicalOp::Filter {
        input: Box::new(node_scan),
        predicate: Expr::BinaryOp {
            left: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".to_string())),
                property: "name".to_string(),
            }),
            op: BinaryOperator::Eq,
            right: Box::new(Expr::Literal(coordinode_core::graph::types::Value::String(
                "Alice".into(),
            ))),
        },
    };

    // Verify baseline EXPLAIN contains NodeScan (optimizer not applied yet).
    let baseline = crate::planner::logical::LogicalPlan {
        root: filter_plan.clone(),
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
    };
    let baseline_explain = baseline.explain();
    assert!(
        baseline_explain.contains("NodeScan"),
        "baseline plan should contain NodeScan, got:\n{baseline_explain}"
    );

    // Apply index selection optimizer — this is the post-build pass.
    let optimized_root = optimize_index_selection(filter_plan, &registry);
    let optimized_plan = crate::planner::logical::LogicalPlan {
        root: optimized_root,
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
    };

    let explain = optimized_plan.explain();

    // Primary assertion: IndexScan must appear, NodeScan must NOT.
    assert!(
        explain.contains("IndexScan"),
        "optimized plan EXPLAIN must contain 'IndexScan', got:\n{explain}"
    );
    assert!(
        !explain.contains("NodeScan"),
        "optimized plan EXPLAIN must NOT contain 'NodeScan' (should be rewritten), got:\n{explain}"
    );

    // Secondary: the EXPLAIN line should name the index and property.
    assert!(
        explain.contains("user_name_idx") && explain.contains("name"),
        "EXPLAIN must reference index name and property, got:\n{explain}"
    );
}

/// A correlated equality lifted above a join (a later MATCH building on a
/// prior binding, `... MATCH (b:Person) WHERE b.pid = e.d`) must become a
/// correlated IndexScan on the join's right side, not a full label scan.
/// Without this the endpoint lookup is O(label) per outer row, which is why
/// bulk edge loading was slow.
#[test]
fn lifted_correlated_equality_uses_index_scan() {
    use crate::planner::optimize_index_selection;

    let registry = crate::index::IndexRegistry::new();
    registry.register_in_memory(crate::index::IndexDefinition::btree(
        "person_pid",
        "Person",
        "pid",
    ));

    // Filter(CartesianProduct(NodeScan(e), NodeScan(b:Person)), b.pid = e.d):
    // the shape the planner emits for the lifted correlated endpoint.
    let left = LogicalOp::NodeScan {
        variable: "e".to_string(),
        labels: vec![],
        property_filters: vec![],
    };
    let right = LogicalOp::NodeScan {
        variable: "b".to_string(),
        labels: vec!["Person".to_string()],
        property_filters: vec![],
    };
    let predicate = Expr::BinaryOp {
        left: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("b".to_string())),
            property: "pid".to_string(),
        }),
        op: BinaryOperator::Eq,
        right: Box::new(Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("e".to_string())),
            property: "d".to_string(),
        }),
    };
    let plan = LogicalOp::Filter {
        input: Box::new(LogicalOp::CartesianProduct {
            left: Box::new(left),
            right: Box::new(right),
        }),
        predicate,
    };

    let optimized = optimize_index_selection(plan, &registry);
    // The right side is now an IndexScan; the Person NodeScan is gone.
    assert!(
        matches!(
            &optimized,
            LogicalOp::CartesianProduct { right, .. }
                if matches!(right.as_ref(), LogicalOp::IndexScan { .. })
        ),
        "expected CartesianProduct(.., IndexScan), got: {optimized:?}"
    );
}

/// Integration test: IndexScan execution returns nodes matching the index lookup.
///
/// Creates an index, inserts nodes, then queries via `LogicalOp::IndexScan`
/// directly and verifies the correct node is returned.
#[test]
fn index_scan_returns_correct_node() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let registry = crate::index::IndexRegistry::new();

    // CREATE INDEX on User.name — will backfill Alice, Bob, Charlie.
    let mut ctx = make_ctx_with_btree(&engine, &mut interner, &allocator, &registry);
    execute_op(
        &LogicalOp::CreateIndex {
            name: "user_name_idx".to_string(),
            label: "User".to_string(),
            property: "name".to_string(),
            unique: false,
            sparse: false,
            filter: None,
        },
        &mut ctx,
    )
    .expect("CREATE INDEX failed");

    // Execute IndexScan for name = "Bob".
    let rows = execute_op(
        &LogicalOp::IndexScan {
            variable: "n".to_string(),
            label: "User".to_string(),
            index_name: "user_name_idx".to_string(),
            property: "name".to_string(),
            value_expr: Expr::Literal(coordinode_core::graph::types::Value::String("Bob".into())),
        },
        &mut ctx,
    )
    .expect("IndexScan failed");

    // Should return exactly one row for Bob.
    assert_eq!(
        rows.len(),
        1,
        "IndexScan for 'Bob' should return exactly one row, got {}",
        rows.len()
    );

    // The row should bind variable 'n' with name = "Bob".
    let name_val = rows[0].get("n.name");
    assert_eq!(
        name_val,
        Some(&coordinode_core::graph::types::Value::String("Bob".into())),
        "row should have n.name = Bob, got {name_val:?}"
    );
}

/// Planner: a correlated equality (`a.pid = e.s` lowered to a property
/// filter) on an indexed property is rewritten to IndexScan, even when it
/// sits on the right of a CartesianProduct (optimizer must recurse there).
#[test]
fn correlated_property_filter_rewrites_to_index_scan() {
    let registry = crate::index::IndexRegistry::new();
    registry.register_in_memory(crate::index::IndexDefinition::btree(
        "person_pid",
        "Person",
        "pid",
    ));

    // right = NodeScan(a:Person {pid: e.s}) — correlated key e.s.
    let right = LogicalOp::NodeScan {
        variable: "a".to_string(),
        labels: vec!["Person".to_string()],
        property_filters: vec![(
            "pid".to_string(),
            Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("e".to_string())),
                property: "s".to_string(),
            },
        )],
    };
    let left = LogicalOp::NodeScan {
        variable: "e".to_string(),
        labels: vec!["Edge".to_string()],
        property_filters: vec![],
    };
    let plan = LogicalOp::CartesianProduct {
        left: Box::new(left),
        right: Box::new(right),
    };

    let optimized = crate::planner::optimize_index_selection(plan, &registry);
    let explain = crate::planner::logical::LogicalPlan {
        root: optimized,
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
    }
    .explain();

    assert!(
        explain.contains("IndexScan(a:Person ON person_pid(pid))"),
        "correlated filter must become IndexScan, got:\n{explain}"
    );
}

/// Planner: a self-referential equality (`a.pid = a.other`) must NOT be
/// rewritten to IndexScan — the key depends on the scanned row.
#[test]
fn self_referential_filter_stays_node_scan() {
    let registry = crate::index::IndexRegistry::new();
    registry.register_in_memory(crate::index::IndexDefinition::btree(
        "person_pid",
        "Person",
        "pid",
    ));

    let plan = LogicalOp::NodeScan {
        variable: "a".to_string(),
        labels: vec!["Person".to_string()],
        property_filters: vec![(
            "pid".to_string(),
            Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("a".to_string())),
                property: "other".to_string(),
            },
        )],
    };

    let optimized = crate::planner::optimize_index_selection(plan, &registry);
    let explain = crate::planner::logical::LogicalPlan {
        root: optimized,
        snapshot_ts: None,
        vector_consistency: VectorConsistencyMode::default(),
        read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(),
    }
    .explain();

    assert!(
        !explain.contains("IndexScan"),
        "self-referential key must not use an index point lookup, got:\n{explain}"
    );
    assert!(
        explain.contains("NodeScan"),
        "expected NodeScan, got:\n{explain}"
    );
}

/// Executor: a correlated IndexScan resolves `value_expr` against
/// `correlated_row`, so a per-outer-row key (`e.s`) reaches the index.
#[test]
fn index_scan_resolves_correlated_key() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let registry = crate::index::IndexRegistry::new();

    let mut ctx = make_ctx_with_btree(&engine, &mut interner, &allocator, &registry);
    execute_op(
        &LogicalOp::CreateIndex {
            name: "user_name_idx".to_string(),
            label: "User".to_string(),
            property: "name".to_string(),
            unique: false,
            sparse: false,
            filter: None,
        },
        &mut ctx,
    )
    .expect("CREATE INDEX failed");

    // Outer row binds e.s = "Bob"; the index key is the correlated e.s.
    let mut corr = Row::new();
    corr.insert(
        "e.s".to_string(),
        coordinode_core::graph::types::Value::String("Bob".into()),
    );
    ctx.correlated_row = Some(corr);

    let rows = execute_op(
        &LogicalOp::IndexScan {
            variable: "n".to_string(),
            label: "User".to_string(),
            index_name: "user_name_idx".to_string(),
            property: "name".to_string(),
            value_expr: Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("e".to_string())),
                property: "s".to_string(),
            },
        },
        &mut ctx,
    )
    .expect("correlated IndexScan failed");

    assert_eq!(
        rows.len(),
        1,
        "correlated IndexScan for e.s='Bob' should return one row, got {}",
        rows.len()
    );
    assert_eq!(
        rows[0].get("n.name"),
        Some(&coordinode_core::graph::types::Value::String("Bob".into())),
        "correlated IndexScan should resolve to Bob"
    );
}

// -- R171: edgeprop_write_key routing --

#[test]
fn edgeprop_write_key_non_temporal_uses_legacy_shape() {
    let key = edgeprop_write_key("KNOWS", NodeId::from_raw(1), NodeId::from_raw(2), None);
    // Legacy shape: `edgeprop:KNOWS:<src 8B>:<tgt 8B>` — 9 + 5 + 1 + 8 + 1 + 8 = 32 bytes.
    assert_eq!(key.len(), 9 + "KNOWS".len() + 1 + 8 + 1 + 8);
    assert!(key.starts_with(b"edgeprop:KNOWS:"));
}

#[test]
fn edgeprop_write_key_temporal_appends_valid_from() {
    let key = edgeprop_write_key(
        "WORKS_AT",
        NodeId::from_raw(1),
        NodeId::from_raw(2),
        Some(1_700_000_000_000),
    );
    // Temporal shape adds `:<valid_from 8B>` (9 more bytes).
    let legacy_len = 9 + "WORKS_AT".len() + 1 + 8 + 1 + 8;
    assert_eq!(key.len(), legacy_len + 1 + 8);
}

#[test]
fn edgeprop_write_key_temporal_keys_sort_by_valid_from() {
    let early = edgeprop_write_key(
        "WORKS_AT",
        NodeId::from_raw(1),
        NodeId::from_raw(2),
        Some(1_000),
    );
    let late = edgeprop_write_key(
        "WORKS_AT",
        NodeId::from_raw(1),
        NodeId::from_raw(2),
        Some(2_000),
    );
    assert!(early < late, "earlier valid_from must sort first");
}

// ── the trigger architecture: L1+L2 cascade tracking ────────────────────────────

/// L1 trip: nesting deeper than the depth limit returns `CascadeOverflow`
/// with the full chain attached for diagnostics.
#[test]
fn cascade_l1_depth_trips_with_chain() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.cascade_depth_limit = 2;
    ctx.cascade_fanout_limit = 100;

    ctx.cascade_enter("audit_a", None, None).expect("depth 1");
    ctx.cascade_enter("audit_b", None, None).expect("depth 2");
    let err = ctx.cascade_enter("audit_c", None, None).unwrap_err();
    match err {
        ExecutionError::CascadeOverflow {
            current,
            limit,
            chain,
        } => {
            assert_eq!(current, 3);
            assert_eq!(limit, 2);
            assert_eq!(chain, vec!["audit_a", "audit_b", "audit_c"]);
        }
        other => panic!("expected CascadeOverflow, got {other:?}"),
    }
    ctx.cascade_exit();
    ctx.cascade_exit();
}

/// L2 trip: a single trigger firing more than `cascade_fanout_limit`
/// times within one cascade root returns `CascadeFanoutOverflow`.
#[test]
fn cascade_l2_fanout_trips_for_repeated_trigger() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.cascade_depth_limit = 100;
    ctx.cascade_fanout_limit = 3;

    for _ in 0..3 {
        ctx.cascade_enter("counter", None, None)
            .expect("fanout within limit");
        ctx.cascade_exit();
    }
    let err = ctx.cascade_enter("counter", None, None).unwrap_err();
    match err {
        ExecutionError::CascadeFanoutOverflow {
            trigger,
            count,
            limit,
        } => {
            assert_eq!(trigger, "counter");
            assert_eq!(count, 4);
            assert_eq!(limit, 3);
        }
        other => panic!("expected CascadeFanoutOverflow, got {other:?}"),
    }
}

/// Per-trigger `CASCADE_LIMIT n` tightens the cluster default; effective limit = min.
#[test]
fn cascade_per_trigger_override_takes_min() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.cascade_depth_limit = 5;

    ctx.cascade_enter("t", Some(2), None)
        .expect("depth 1 within tight override");
    ctx.cascade_enter("t", Some(2), None)
        .expect("depth 2 within tight override");
    let err = ctx.cascade_enter("t", Some(2), None).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::CascadeOverflow { limit: 2, .. }
    ));
    ctx.cascade_exit();
    ctx.cascade_exit();
}

/// Per-trigger override cannot raise the limit above the cluster cap.
#[test]
fn cascade_per_trigger_override_cannot_exceed_cluster() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.cascade_depth_limit = 3;

    ctx.cascade_enter("t", Some(100), None).expect("depth 1");
    ctx.cascade_enter("t", Some(100), None).expect("depth 2");
    ctx.cascade_enter("t", Some(100), None).expect("depth 3");
    let err = ctx.cascade_enter("t", Some(100), None).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::CascadeOverflow { limit: 3, .. }
    ));
    ctx.cascade_exit();
    ctx.cascade_exit();
    ctx.cascade_exit();
}

/// `cascade_exit` decrements depth so sibling cascades start fresh.
#[test]
fn cascade_exit_decrements_depth_for_siblings() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.cascade_depth_limit = 2;

    ctx.cascade_enter("a", None, None).unwrap();
    ctx.cascade_enter("b", None, None).unwrap();
    ctx.cascade_exit();
    ctx.cascade_exit();
    assert_eq!(ctx.cascade_depth, 0);

    ctx.cascade_enter("c", None, None).expect("fresh cascade");
    ctx.cascade_enter("d", None, None).expect("nested");
    ctx.cascade_exit();
    ctx.cascade_exit();
}

/// `cascade_reset()` wipes per-trigger fanout counts.
#[test]
fn cascade_reset_clears_fanout_counts() {
    let (_dir, engine, mut interner) = setup_test_graph();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
    let mut ctx = make_ctx(&engine, &mut interner, &allocator);
    ctx.cascade_fanout_limit = 2;
    ctx.cascade_depth_limit = 100;

    ctx.cascade_enter("t", None, None).unwrap();
    ctx.cascade_exit();
    ctx.cascade_enter("t", None, None).unwrap();
    ctx.cascade_exit();

    ctx.cascade_reset();
    ctx.cascade_enter("t", None, None)
        .expect("fanout counter cleared");
    ctx.cascade_exit();
}
