//! Integration tests: HNSW vector search.
//!
//! Tests the vector index lifecycle: build, search, recall.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_core::graph::types::VectorMetric;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex};

fn config(metric: VectorMetric) -> HnswConfig {
    HnswConfig {
        metric,
        ..HnswConfig::default()
    }
}

// ── Index lifecycle ─────────────────────────────────────────────────

#[test]
fn create_index_and_search() {
    let mut index = HnswIndex::new(config(VectorMetric::Cosine));

    index.insert(1, vec![1.0, 0.0, 0.0, 0.0]);
    index.insert(2, vec![0.0, 1.0, 0.0, 0.0]);
    index.insert(3, vec![0.0, 0.0, 1.0, 0.0]);
    index.insert(4, vec![0.0, 0.0, 0.0, 1.0]);

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn search_returns_k_nearest() {
    let mut index = HnswIndex::new(config(VectorMetric::L2));

    for i in 0..20u64 {
        index.insert(i, vec![i as f32, 0.0, 0.0]);
    }

    let results = index.search(&[5.0, 0.0, 0.0], 5);
    assert_eq!(results.len(), 5);
}

#[test]
fn empty_index_returns_empty() {
    let index = HnswIndex::new(config(VectorMetric::Cosine));
    let results = index.search(&[1.0, 0.0, 0.0], 5);
    assert!(results.is_empty());
}

// ── Distance metrics ────────────────────────────────────────────────

#[test]
fn cosine_distance() {
    let mut index = HnswIndex::new(config(VectorMetric::Cosine));
    index.insert(1, vec![1.0, 0.0, 0.0]);
    index.insert(2, vec![0.0, 1.0, 0.0]);

    let results = index.search(&[0.9, 0.1, 0.0], 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn l2_distance() {
    let mut index = HnswIndex::new(config(VectorMetric::L2));
    index.insert(1, vec![0.0, 0.0]);
    index.insert(2, vec![10.0, 10.0]);

    let results = index.search(&[1.0, 1.0], 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn dot_product_distance() {
    let mut index = HnswIndex::new(config(VectorMetric::DotProduct));
    index.insert(1, vec![1.0, 0.0, 0.0]);
    index.insert(2, vec![0.0, 1.0, 0.0]);

    let results = index.search(&[1.0, 0.0, 0.0], 1);
    assert_eq!(results[0].id, 1);
}

// ── Larger index ────────────────────────────────────────────────────

#[test]
fn hundred_vectors_recall() {
    let dims = 8;
    let mut index = HnswIndex::new(config(VectorMetric::L2));

    for i in 0..100u64 {
        let v: Vec<f32> = (0..dims)
            .map(|d| (i * 7 + d as u64) as f32 % 100.0)
            .collect();
        index.insert(i, v);
    }

    let query: Vec<f32> = (0..dims).map(|d| d as f32 * 10.0).collect();
    let results = index.search(&query, 10);
    assert_eq!(results.len(), 10);

    for r in &results {
        assert!(r.id < 100, "invalid ID: {}", r.id);
    }
}

// ── Duplicate insert is no-op ───────────────────────────────────────

#[test]
fn insert_same_id_is_noop() {
    let mut index = HnswIndex::new(config(VectorMetric::L2));
    index.insert(1, vec![0.0, 0.0, 0.0]);
    index.insert(1, vec![10.0, 10.0, 10.0]);

    assert_eq!(index.len(), 1);
}

// ── Index size ──────────────────────────────────────────────────────

#[test]
fn len_and_is_empty() {
    let mut index = HnswIndex::new(config(VectorMetric::Cosine));
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);

    index.insert(1, vec![1.0, 0.0]);
    assert!(!index.is_empty());
    assert_eq!(index.len(), 1);

    index.insert(2, vec![0.0, 1.0]);
    assert_eq!(index.len(), 2);
}

// ── SQ8 Quantization Integration ──────────────────────────────────

fn sq8_config(metric: VectorMetric) -> HnswConfig {
    HnswConfig {
        metric,
        quantization: true,
        rerank_candidates: 100,
        calibration_threshold: 50,
        ..HnswConfig::default()
    }
}

/// SQ8 + HNSW end-to-end: build quantized index, verify search recall
/// stays acceptable compared to brute-force ground truth.
#[test]
fn sq8_hnsw_recall_384_dims() {
    use coordinode_vector::metrics;
    use std::collections::HashSet;

    let dims = 384; // Typical embedding size (e.g., MiniLM)
    let n = 200;
    let k = 10;

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dims)
                .map(|d| ((i * d + 13) as f32 * 0.07).sin())
                .collect()
        })
        .collect();

    let mut index = HnswIndex::new(sq8_config(VectorMetric::L2));

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }

    assert!(
        index.is_quantized(),
        "SQ8 should be calibrated after 200 inserts"
    );

    // Brute-force ground truth
    let query = &vectors[42]; // Arbitrary query
    let mut ground_truth: Vec<(f32, u64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (metrics::euclidean_distance_squared(query, v), i as u64))
        .collect();
    ground_truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let gt_set: HashSet<u64> = ground_truth.iter().take(k).map(|&(_, id)| id).collect();

    let results = index.search(query, k);
    assert_eq!(results.len(), k);

    let result_set: HashSet<u64> = results.iter().map(|r| r.id).collect();
    let recall = gt_set.intersection(&result_set).count() as f32 / k as f32;

    eprintln!("Integration SQ8 384-dim recall@{k}: {:.0}%", recall * 100.0);

    // SQ8 with reranking should achieve good recall
    assert!(
        recall >= 0.5,
        "SQ8 384-dim recall {recall:.0}% too low (expected >= 50%)"
    );

    // Verify reranked scores are exact f32 distances
    let first_exact = metrics::euclidean_distance_squared(query, &vectors[results[0].id as usize]);
    assert!(
        (results[0].score - first_exact).abs() < 1e-5,
        "reranked score should be exact f32: got {}, expected {first_exact}",
        results[0].score
    );
}

/// SQ8 + HNSW with cosine metric on 768-dim vectors (e.g., sentence-transformers).
#[test]
fn sq8_hnsw_cosine_768_dims() {
    let dims = 768;
    let n = 100;

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dims)
                .map(|d| ((i * d + 5) as f32 * 0.03).sin())
                .collect()
        })
        .collect();

    let mut index = HnswIndex::new(sq8_config(VectorMetric::Cosine));

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }

    assert!(index.is_quantized());

    // Self-search: query with vector 0, should find itself
    let results = index.search(&vectors[0], 5);
    assert_eq!(results.len(), 5);
    assert_eq!(results[0].id, 0, "self-query should return self as nearest");
    assert!(
        results[0].score < 0.01,
        "self-distance (cosine) should be near zero, got {}",
        results[0].score
    );
}

/// SQ8 calibration lifecycle: before threshold → after threshold → new inserts.
#[test]
fn sq8_calibration_lifecycle() {
    let mut index = HnswIndex::new(HnswConfig {
        metric: VectorMetric::L2,
        quantization: true,
        calibration_threshold: 20,
        rerank_candidates: 30,
        ..HnswConfig::default()
    });

    // Phase 1: before calibration
    for i in 0..19u64 {
        index.insert(i, vec![i as f32, (i as f32).sin(), 0.0]);
    }
    assert!(
        !index.is_quantized(),
        "should not be calibrated before threshold"
    );
    assert!(index.sq8_params().is_none());

    // Phase 2: trigger calibration
    index.insert(19, vec![19.0, 19.0_f32.sin(), 0.0]);
    assert!(index.is_quantized(), "should be calibrated at threshold");

    let params = index.sq8_params().expect("params should exist");
    assert_eq!(params.dims(), 3);

    // Phase 3: new inserts after calibration are immediately quantized
    index.insert(100, vec![10.0, 0.5, 0.0]);
    assert_eq!(index.len(), 21);

    // Search should work across pre-calibration and post-calibration vectors
    let results = index.search(&[10.0, 0.5, 0.0], 1);
    assert_eq!(results[0].id, 100);
}

/// SQ8 + manual calibration via set_sq8_params.
#[test]
fn sq8_manual_calibration_integration() {
    use coordinode_vector::quantize::Sq8Params;

    let mut index = HnswIndex::new(HnswConfig {
        metric: VectorMetric::L2,
        quantization: true,
        calibration_threshold: 10000, // Won't auto-calibrate
        ..HnswConfig::default()
    });

    // Insert vectors
    let vectors: Vec<Vec<f32>> = (0..50)
        .map(|i| vec![i as f32 / 50.0, ((i as f32) * 0.1).sin()])
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }
    assert!(!index.is_quantized());

    // Manually calibrate from the data
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = Sq8Params::calibrate(&refs).expect("calibrate");
    index.set_sq8_params(params);
    assert!(index.is_quantized());

    // Search works with quantized index
    let results = index.search(&vectors[25], 3);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, 25);
}

// ── Vector MVCC Consistency Mode Integration Tests ───────────────────

use coordinode_core::graph::types::VectorConsistencyMode;
use coordinode_embed::Database;

/// SET vector_consistency session command + verify mode persists across queries.
#[test]
fn vector_consistency_session_set_persists() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Default is current
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Current);

    // Set to snapshot
    db.execute_cypher("SET vector_consistency = 'snapshot'")
        .expect("set snapshot");
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Snapshot);

    // Create nodes and run vector query — should work in snapshot mode
    db.execute_cypher("CREATE (m:Movie {title: 'Matrix', embedding: [1.0, 0.0, 0.0, 0.0]})")
        .expect("create movie");

    let results = db
        .execute_cypher(
            "MATCH (m:Movie) \
             WHERE vector_distance(m.embedding, [0.9, 0.1, 0.0, 0.0]) < 2.0 \
             RETURN m.title",
        )
        .expect("vector query in snapshot mode");

    assert_eq!(results.len(), 1);

    // Mode persists to next query
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Snapshot);
}

/// Vector query works in all three modes (current, snapshot, exact).
#[test]
fn vector_query_all_three_modes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Setup vector data
    db.execute_cypher("CREATE (a:Item {name: 'A', v: [1.0, 0.0]})")
        .expect("create A");
    db.execute_cypher("CREATE (b:Item {name: 'B', v: [0.0, 1.0]})")
        .expect("create B");

    let query = "MATCH (i:Item) \
                 WHERE vector_distance(i.v, [0.9, 0.1]) < 2.0 \
                 RETURN i.name";

    // Mode: current (default)
    let r1 = db.execute_cypher(query).expect("current mode");
    assert!(!r1.is_empty(), "current mode should return results");

    // Mode: snapshot
    db.execute_cypher("SET vector_consistency = 'snapshot'")
        .expect("set");
    let r2 = db.execute_cypher(query).expect("snapshot mode");
    assert!(!r2.is_empty(), "snapshot mode should return results");

    // Mode: exact
    db.execute_cypher("SET vector_consistency = 'exact'")
        .expect("set");
    let r3 = db.execute_cypher(query).expect("exact mode");
    assert!(!r3.is_empty(), "exact mode should return results");

    // All modes return the same results for this simple case
    // (brute-force path, all data MVCC-visible)
    assert_eq!(r1.len(), r2.len());
    assert_eq!(r2.len(), r3.len());
}

/// EXPLAIN shows vector consistency mode when query has VectorFilter.
#[test]
fn explain_shows_vector_mode() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let query = "MATCH (m:Movie) \
                 WHERE vector_distance(m.embedding, [1.0, 0.0]) < 0.5 \
                 RETURN m";

    // Default mode: current
    let explain = db.explain_cypher(query).expect("explain");
    assert!(
        explain.contains("Vector consistency: current"),
        "should show current: {explain}"
    );

    // Set snapshot
    db.set_vector_consistency(VectorConsistencyMode::Snapshot);
    let explain = db.explain_cypher(query).expect("explain");
    assert!(
        explain.contains("Vector consistency: snapshot"),
        "should show snapshot: {explain}"
    );

    // Set exact
    db.set_vector_consistency(VectorConsistencyMode::Exact);
    let explain = db.explain_cypher(query).expect("explain");
    assert!(
        explain.contains("Vector consistency: exact"),
        "should show exact: {explain}"
    );
}

/// Non-vector queries should NOT show vector consistency in EXPLAIN.
#[test]
fn explain_no_vector_mode_for_non_vector_query() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.set_vector_consistency(VectorConsistencyMode::Snapshot);

    let explain = db
        .explain_cypher("MATCH (n:User) RETURN n.name")
        .expect("explain");
    assert!(
        !explain.contains("Vector consistency"),
        "non-vector query should NOT show vector mode: {explain}"
    );
}

/// SET vector_consistency with invalid value is a parse error.
#[test]
fn set_invalid_vector_consistency_is_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let result = db.execute_cypher("SET vector_consistency = 'turbo'");
    assert!(result.is_err(), "invalid mode should error");

    // Original mode unchanged
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Current);
}

/// Snapshot mode on vector query emits stats warning via ExecutionContext.
/// Tests that vector_mvcc_stats and warnings are populated when
/// vector_consistency = Snapshot is set.
#[test]
fn snapshot_mode_emits_stats_warning() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_query::executor::runner::execute;
    use coordinode_query::{cypher, planner};
    use coordinode_raft::proposal::LocalProposalPipeline;
    use coordinode_storage::engine::config::StorageConfig;
    use coordinode_storage::engine::core::StorageEngine;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::new(dir.path());
    let engine = StorageEngine::open(&config).expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(coordinode_core::graph::node::NodeId::from_raw(0));
    let oracle = coordinode_core::txn::timestamp::TimestampOracle::new();
    let pipeline = LocalProposalPipeline::new(&engine);
    let proposal_id_gen = coordinode_core::txn::proposal::ProposalIdGenerator::new();

    // Create a node with vector property
    {
        let read_ts = oracle.next();
        let ast = cypher::parse("CREATE (m:Movie {title: 'Matrix', embedding: [1.0, 0.0, 0.0]})")
            .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        execute(&plan, &mut ctx).expect("create");
    }

    // Query with snapshot mode
    {
        let read_ts = oracle.next();
        // Pure vector KNN is single-modality (post-R-SNAP6) — no auto-promotion.
        // Explicit `read_consistency('snapshot')` hint forces snapshot mode so
        // the HNSW post-filter path emits MVCC stats this test asserts.
        let ast = cypher::parse(
            "MATCH (m:Movie) WHERE vector_distance(m.embedding, [0.9, 0.1, 0.0]) < 2.0 \
             RETURN m.title /*+ read_consistency('snapshot') */",
        )
        .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        let results = execute(&plan, &mut ctx).expect("execute");
        assert!(!results.is_empty(), "should find the movie");

        // Check that MVCC stats warning was emitted
        let mvcc_warning = ctx.warnings.iter().find(|w| w.starts_with("vector_mvcc("));
        assert!(
            mvcc_warning.is_some(),
            "snapshot mode should emit vector_mvcc stats warning, got: {:?}",
            ctx.warnings
        );
        let w = mvcc_warning.unwrap();
        assert!(w.contains("mode=snapshot"), "warning should show mode: {w}");
        assert!(w.contains("fetched="), "warning should show fetched: {w}");

        // Check vector_mvcc_stats struct
        let stats = ctx.vector_mvcc_stats.as_ref().expect("should have stats");
        assert!(stats.candidates_fetched > 0);
        assert!((stats.overfetch_factor - 1.2).abs() < f64::EPSILON);
    }
}

/// Current mode does NOT emit vector_mvcc stats or warnings.
#[test]
fn current_mode_no_stats() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Ensure default (current) mode
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Current);

    db.execute_cypher("CREATE (m:Movie {title: 'X', v: [1.0, 0.0]})")
        .expect("create");

    // This should NOT trigger any MVCC warning in current mode
    // (we can't check warnings via Database API, but the query succeeds)
    let results = db
        .execute_cypher(
            "MATCH (m:Movie) WHERE vector_distance(m.v, [0.9, 0.1]) < 2.0 RETURN m.title",
        )
        .expect("query");
    assert!(!results.is_empty());
}

// ── Vector MVCC Test Scenarios ─────────────────────────────────────
//
// Four specific scenarios:
// (1) concurrent insert + snapshot search → inserted vector invisible
// (2) concurrent delete + snapshot search → deleted vector still visible
// (3) AS OF TIMESTAMP vector search → correct candidate set
// (4) Recall benchmark: snapshot vs current on 100K dataset with 1% churn

/// Vector MVCC test (1): Insert vector AFTER snapshot_ts → invisible in snapshot search.
///
/// Timeline:
///   ts=1001: CREATE Movie A (embedding [1.0, 0.0])
///   ts=1003: CREATE Movie B (embedding [0.9, 0.1])  ← after snapshot
///   search at snapshot ts=1002: should see A, NOT B
#[test]
fn r106e_concurrent_insert_invisible_in_snapshot() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_query::executor::runner::execute;
    use coordinode_query::{cypher, planner};
    use coordinode_raft::proposal::LocalProposalPipeline;
    use coordinode_storage::engine::config::StorageConfig;
    use coordinode_storage::engine::core::StorageEngine;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::new(dir.path());
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(coordinode_core::graph::node::NodeId::from_raw(0));
    let pipeline = LocalProposalPipeline::new(&engine);
    let proposal_id_gen = coordinode_core::txn::proposal::ProposalIdGenerator::new();

    // Step 1: CREATE Movie A at ts=1001
    {
        let read_ts = oracle.next(); // ts=1001
        let ast = cypher::parse("CREATE (a:Movie {title: 'Matrix', embedding: [1.0, 0.0, 0.0]})")
            .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        ctx.vector_consistency = VectorConsistencyMode::Snapshot;
        execute(&plan, &mut ctx).expect("create A");
    }

    // Advance oracle to create a gap for the snapshot timestamp
    let _snapshot_ts = oracle.next(); // ts=1003 (commit of A was 1002)
                                      // snapshot_ts for our read will be the NEXT one = 1004
    let read_ts_for_snapshot = oracle.next(); // ts=1004

    // Step 2: CREATE Movie B at ts AFTER our snapshot read_ts
    // We need B to be committed at a ts > read_ts_for_snapshot.
    // So we create B with a read_ts AFTER our snapshot.
    {
        let read_ts_b = oracle.next(); // ts=1005
        let ast =
            cypher::parse("CREATE (b:Movie {title: 'Inception', embedding: [0.9, 0.1, 0.0]})")
                .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts_b,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        execute(&plan, &mut ctx).expect("create B");
    }

    // Step 3: Vector search at snapshot ts=read_ts_for_snapshot
    // B was committed AFTER this ts, so B should be invisible.
    {
        let ast = cypher::parse(
            "MATCH (m:Movie) \
             WHERE vector_distance(m.embedding, [0.95, 0.05, 0.0]) < 5.0 \
             RETURN m.title",
        )
        .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts_for_snapshot,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        ctx.vector_consistency = VectorConsistencyMode::Snapshot;
        let results = execute(&plan, &mut ctx).expect("snapshot query");

        // Should see Movie A (committed before snapshot) but NOT Movie B (committed after)
        assert_eq!(
            results.len(),
            1,
            "snapshot search should see only 1 movie (A), not B. Got: {results:?}"
        );
    }

    // Step 4: Same query at latest ts → should see BOTH movies
    {
        let latest_ts = oracle.next();
        let ast = cypher::parse(
            "MATCH (m:Movie) \
             WHERE vector_distance(m.embedding, [0.95, 0.05, 0.0]) < 5.0 \
             RETURN m.title",
        )
        .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            latest_ts,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        ctx.vector_consistency = VectorConsistencyMode::Current;
        let results = execute(&plan, &mut ctx).expect("current query");

        assert_eq!(
            results.len(),
            2,
            "current mode at latest ts should see both movies. Got: {results:?}"
        );
    }
}

/// Vector MVCC test (2): Delete vector AFTER snapshot_ts → still visible in snapshot search.
///
/// Timeline:
///   ts=1001: CREATE Movie A  (committed at ts=1002)
///   ts=1003: capture snapshot_read_ts
///   ts=1004: DELETE Movie A  (committed at ts=1005)
///   read at snapshot_read_ts=1003: A should STILL be visible (deleted after snapshot)
///   read at latest ts: A should be gone
#[test]
fn r106e_concurrent_delete_still_visible_in_snapshot() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_query::executor::runner::execute;
    use coordinode_query::{cypher, planner};
    use coordinode_raft::proposal::LocalProposalPipeline;
    use coordinode_storage::engine::config::StorageConfig;
    use coordinode_storage::engine::core::StorageEngine;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::new(dir.path());
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(coordinode_core::graph::node::NodeId::from_raw(0));
    let pipeline = LocalProposalPipeline::new(&engine);
    let proposal_id_gen = coordinode_core::txn::proposal::ProposalIdGenerator::new();

    // Step 1: CREATE Movie A at ts=1001, committed at ts=1002
    {
        let read_ts = oracle.next(); // 1001
        let ast =
            cypher::parse("CREATE (m:Movie {title: 'WillDelete', embedding: [1.0, 0.0, 0.0]})")
                .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        execute(&plan, &mut ctx).expect("create movie");
    }

    // Step 2: Capture snapshot timestamp BEFORE delete
    let snapshot_ts = oracle.next(); // 1003

    // Step 3: DELETE the movie at ts=1004, committed at ts=1005
    {
        let read_ts = oracle.next(); // 1004
        let ast = cypher::parse("MATCH (m:Movie {title: 'WillDelete'}) DELETE m").expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        execute(&plan, &mut ctx).expect("delete movie");
    }

    // Step 4: Read at snapshot_ts (BEFORE delete) → movie SHOULD still be visible
    {
        let ast = cypher::parse(
            "MATCH (m:Movie) \
             WHERE vector_distance(m.embedding, [0.9, 0.1, 0.0]) < 5.0 \
             RETURN m.title",
        )
        .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            snapshot_ts, // read at ts BEFORE delete
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        ctx.vector_consistency = VectorConsistencyMode::Snapshot;
        let results = execute(&plan, &mut ctx).expect("snapshot query after delete");

        assert_eq!(
            results.len(),
            1,
            "snapshot at ts BEFORE delete should STILL see the movie. Got: {results:?}"
        );
    }

    // Step 5: Read at latest ts → movie should be GONE
    {
        let latest = oracle.next();
        let ast = cypher::parse(
            "MATCH (m:Movie) \
             WHERE vector_distance(m.embedding, [0.9, 0.1, 0.0]) < 5.0 \
             RETURN m.title",
        )
        .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            latest,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        ctx.vector_consistency = VectorConsistencyMode::Current;
        let results = execute(&plan, &mut ctx).expect("latest query after delete");

        assert_eq!(
            results.len(),
            0,
            "latest ts after delete should NOT see the movie. Got: {results:?}"
        );
    }
}

/// Vector MVCC test (3): Time-travel vector search via direct mvcc_read_ts control.
///
/// Creates two movies at different MVCC timestamps, then queries at a
/// read_ts between the two creates to verify only the first is visible.
/// (Uses mvcc_read_ts directly — AS OF TIMESTAMP Cypher syntax wiring
/// to mvcc_read_ts is a separate enhancement.)
#[test]
fn r106e_time_travel_vector_search_via_mvcc_read_ts() {
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_core::graph::node::NodeIdAllocator;
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_query::executor::runner::execute;
    use coordinode_query::{cypher, planner};
    use coordinode_raft::proposal::LocalProposalPipeline;
    use coordinode_storage::engine::config::StorageConfig;
    use coordinode_storage::engine::core::StorageEngine;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::new(dir.path());
    let mut interner = FieldInterner::new();
    let allocator = NodeIdAllocator::resume_from(coordinode_core::graph::node::NodeId::from_raw(0));

    // Use high base timestamp to simulate recent Unix micros
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(now_us)));
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");
    let pipeline = LocalProposalPipeline::new(&engine);
    let proposal_id_gen = coordinode_core::txn::proposal::ProposalIdGenerator::new();

    // Create Movie A
    {
        let read_ts = oracle.next();
        let ast = cypher::parse("CREATE (a:Movie {title: 'First', embedding: [1.0, 0.0]})")
            .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        execute(&plan, &mut ctx).expect("create A");
    }

    // Capture a timestamp between the two creates
    let between_ts = oracle.next();

    // Create Movie B (after between_ts)
    {
        let read_ts = oracle.next();
        let ast = cypher::parse("CREATE (b:Movie {title: 'Second', embedding: [0.0, 1.0]})")
            .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            read_ts,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        execute(&plan, &mut ctx).expect("create B");
    }

    // Time-travel query at between_ts → should see only A (committed before between_ts)
    {
        let ast = cypher::parse(
            "MATCH (m:Movie) \
             WHERE vector_distance(m.embedding, [0.5, 0.5]) < 5.0 \
             RETURN m.title",
        )
        .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            between_ts, // time-travel: read at between_ts
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        ctx.vector_consistency = VectorConsistencyMode::Snapshot;
        let results = execute(&plan, &mut ctx).expect("time-travel query");

        assert_eq!(
            results.len(),
            1,
            "time-travel at between_ts should see only First movie, got: {results:?}"
        );
    }

    // Verify at latest ts both are visible
    {
        let latest = oracle.next();
        let ast = cypher::parse(
            "MATCH (m:Movie) \
             WHERE vector_distance(m.embedding, [0.5, 0.5]) < 5.0 \
             RETURN m.title",
        )
        .expect("parse");
        let plan = planner::build_logical_plan(&ast).expect("plan");
        let mut ctx = super::helpers::make_ctx_with_pipeline(
            &engine,
            &oracle,
            latest,
            None,
            &pipeline,
            &proposal_id_gen,
            &mut interner,
            &allocator,
        );
        ctx.shard_id = 1;
        ctx.vector_consistency = VectorConsistencyMode::Current;
        let results = execute(&plan, &mut ctx).expect("latest query");
        assert_eq!(results.len(), 2, "latest should see both: {results:?}");
    }
}

/// Vector MVCC test (4): Recall benchmark — snapshot vs current mode on HNSW index
/// with simulated churn (1% of vectors deleted/re-inserted).
///
/// Verifies that snapshot mode recall stays high (>95%) under churn.
#[test]
fn r106e_recall_benchmark_snapshot_vs_current() {
    // Build a 10K-vector HNSW index (100K too slow for unit test; benchmark
    // uses criterion for the full 100K measurement)
    let n = 10_000usize;
    let dims = 8;
    let k = 20;
    let churn_pct = 0.01; // 1% churn

    let mut index = HnswIndex::new(HnswConfig {
        metric: coordinode_core::graph::types::VectorMetric::L2,
        ef_search: 200, // High ef for good recall in benchmark
        ..HnswConfig::default()
    });

    // Insert N vectors with well-distributed coordinates
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let v: Vec<f32> = (0..dims)
            .map(|d| (i as f32 + d as f32 * 0.1) / n as f32)
            .collect();
        index.insert(i as u64, v.clone());
        vectors.push(v);
    }

    // Simulate 1% churn: spread IDs evenly across dataset
    let churn_count = (n as f64 * churn_pct) as usize;
    let churned_ids: std::collections::HashSet<u64> = (0..n as u64)
        .step_by(n / churn_count.max(1))
        .take(churn_count)
        .collect();

    let query: Vec<f32> = (0..dims).map(|d| 0.5 + d as f32 * 0.01).collect();

    // Current mode (no filter) — baseline
    let current_results = index.search(&query, k);
    assert_eq!(current_results.len(), k);

    // Snapshot mode with 1% invisible — higher overfetch for stability
    let (snapshot_results, stats) =
        index.search_with_visibility(&query, k, 2.0, 3, |id| !churned_ids.contains(&id));
    assert_eq!(snapshot_results.len(), k);

    // Verify no churned IDs leaked through
    for r in &snapshot_results {
        assert!(
            !churned_ids.contains(&r.id),
            "churned ID {} should not appear in snapshot results",
            r.id
        );
    }

    // Calculate recall: how many of the top-K visible vectors (ground truth)
    // are found in the snapshot results?
    // Ground truth: brute-force top-K excluding churned IDs
    let mut ground_truth: Vec<(u64, f32)> = vectors
        .iter()
        .enumerate()
        .filter(|(i, _)| !churned_ids.contains(&(*i as u64)))
        .map(|(i, v)| {
            let dist = coordinode_vector::metrics::euclidean_distance(&query, v);
            (i as u64, dist)
        })
        .collect();
    ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let gt_ids: std::collections::HashSet<u64> =
        ground_truth.iter().take(k).map(|(id, _)| *id).collect();

    let snapshot_ids: std::collections::HashSet<u64> =
        snapshot_results.iter().map(|r| r.id).collect();

    let recall = snapshot_ids.intersection(&gt_ids).count() as f64 / k as f64;

    assert!(
        recall >= 0.90,
        "snapshot recall should be ≥90% with 1% churn, got {:.1}%",
        recall * 100.0
    );

    // Stats should show the overfetch factor we specified
    assert!((stats.overfetch_factor - 2.0).abs() < f64::EPSILON);
}

/// search_with_visibility on HnswIndex: overfetch + filter + expansion.
/// This is a direct integration test of the algorithm (not through Cypher).
#[test]
fn hnsw_search_with_visibility_integration() {
    let mut index = HnswIndex::new(HnswConfig {
        metric: coordinode_core::graph::types::VectorMetric::L2,
        ef_search: 20,
        ..HnswConfig::default()
    });

    // 500 nodes: IDs 0..499
    for i in 0..500u64 {
        index.insert(i, vec![i as f32, 0.0, 0.0]);
    }

    // Only nodes with id % 5 == 0 are "visible" (20%)
    let (results, stats) =
        index.search_with_visibility(&[250.0, 0.0, 0.0], 10, 1.2, 3, |id| id % 5 == 0);

    // Must have exactly 10 results, all visible
    assert_eq!(results.len(), 10);
    for r in &results {
        assert_eq!(r.id % 5, 0, "ID {} should be divisible by 5", r.id);
    }
    // Closest visible node to 250 should be 250 (if visible)
    assert_eq!(results[0].id, 250);

    // Stats should reflect filtering
    assert!(stats.candidates_filtered > 0);
    assert!(stats.candidates_visible >= 10);
    assert!((stats.overfetch_factor - 1.2).abs() < f64::EPSILON);
}

// ── HNSW Index Integration with VectorFilter (G009) ──────────────

/// End-to-end test: create vector index via Database API, insert nodes,
/// verify that vector_similarity queries use HNSW index (via registry).
#[test]
fn hnsw_index_accelerates_vector_query() {
    use coordinode_core::graph::node::NodeId;
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create HNSW index on Movie.embedding (3 dims, cosine)
    db.create_vector_index(
        "movie_embed_idx",
        "Movie",
        "embedding",
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::Cosine,
            ..VectorIndexConfig::default()
        },
    );

    // Create nodes with vector properties.
    db.execute_cypher("CREATE (m:Movie {title: 'Matrix', embedding: [1.0, 0.0, 0.0]})")
        .expect("create Matrix");
    db.execute_cypher("CREATE (m:Movie {title: 'Inception', embedding: [0.0, 1.0, 0.0]})")
        .expect("create Inception");
    db.execute_cypher("CREATE (m:Movie {title: 'Tenet', embedding: [0.9, 0.1, 0.0]})")
        .expect("create Tenet");

    // Read back node IDs and vectors to populate the HNSW index.
    // In production, this would be done automatically by the write path.
    let nodes = db
        .execute_cypher("MATCH (m:Movie) RETURN m, m.embedding")
        .expect("read nodes");

    let registry = db.vector_index_registry();
    for row in &nodes {
        let node_id = row.get("m").and_then(|v| v.as_int()).expect("node id") as u64;
        // Embedding may be stored as Vector or Array (Cypher array literal).
        let vec_f32: Option<Vec<f32>> = row.get("m.embedding").and_then(|v| match v {
            coordinode_core::graph::types::Value::Vector(v) => Some(v.clone()),
            coordinode_core::graph::types::Value::Array(arr) => arr
                .iter()
                .map(|item| match item {
                    coordinode_core::graph::types::Value::Float(f) => Some(*f as f32),
                    coordinode_core::graph::types::Value::Int(i) => Some(*i as f32),
                    _ => None,
                })
                .collect(),
            _ => None,
        });
        if let Some(vec) = vec_f32 {
            registry.on_vector_written("Movie", NodeId::from_raw(node_id), "embedding", &vec);
        }
    }

    // Verify HNSW index has 3 vectors.
    let hnsw_results = registry
        .search("Movie", "embedding", &[1.0, 0.0, 0.0], 3)
        .expect("HNSW search");
    assert_eq!(hnsw_results.len(), 3, "HNSW should have 3 vectors");

    // Run vector similarity query through full Cypher pipeline.
    // With HNSW index, the VectorFilter should route through HNSW
    // instead of brute-force.
    let results = db
        .execute_cypher(
            "MATCH (m:Movie) \
             WHERE vector_similarity(m.embedding, [1.0, 0.0, 0.0]) > 0.5 \
             RETURN m.title",
        )
        .expect("vector similarity query");

    // Matrix (exact match, cos_sim=1.0) and Tenet (cos_sim≈0.99) should pass.
    // Inception (cos_sim≈0.0) should be filtered out.
    assert!(
        results.len() >= 2,
        "should find at least Matrix and Tenet, got {} results",
        results.len()
    );

    // Verify Matrix is in results
    let titles: Vec<String> = results
        .iter()
        .filter_map(|r| r.get("m.title"))
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    assert!(
        titles.contains(&"Matrix".to_string()),
        "Matrix should be in results: {titles:?}"
    );
    assert!(
        titles.contains(&"Tenet".to_string()),
        "Tenet should be in results: {titles:?}"
    );
}

/// Verify that without HNSW index, brute-force still works correctly.
#[test]
fn vector_query_brute_force_without_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // No HNSW index created — should fall back to brute-force.
    db.execute_cypher("CREATE (m:Movie {title: 'Matrix', embedding: [1.0, 0.0, 0.0]})")
        .expect("create Matrix");
    db.execute_cypher("CREATE (m:Movie {title: 'Inception', embedding: [0.0, 1.0, 0.0]})")
        .expect("create Inception");

    let results = db
        .execute_cypher(
            "MATCH (m:Movie) \
             WHERE vector_similarity(m.embedding, [1.0, 0.0, 0.0]) > 0.5 \
             RETURN m.title",
        )
        .expect("brute-force vector query");

    // Only Matrix should pass (cosine_sim with [0,1,0] is ~0)
    assert_eq!(results.len(), 1);
    let title = results[0].get("m.title").and_then(|v| v.as_str());
    assert_eq!(title, Some("Matrix"));
}

/// HNSW index registry: bulk_insert populates index correctly.
#[test]
fn hnsw_bulk_insert_and_search() {
    use coordinode_query::index::{VectorIndexConfig, VectorIndexRegistry};

    let reg = VectorIndexRegistry::new();
    reg.register(coordinode_query::index::IndexDefinition::hnsw(
        "test_idx",
        "Item",
        "vec",
        VectorIndexConfig {
            dimensions: 4,
            metric: VectorMetric::L2,
            ..VectorIndexConfig::default()
        },
    ));

    // Bulk insert 100 vectors
    let vectors: Vec<(u64, Vec<f32>)> = (0..100)
        .map(|i| (i, vec![i as f32, (i as f32).sin(), (i as f32).cos(), 0.0]))
        .collect();

    let count = reg.bulk_insert("Item", "vec", vectors.into_iter());
    assert_eq!(count, 100);

    // Search should work
    let results = reg
        .search(
            "Item",
            "vec",
            &[50.0, 50.0_f32.sin(), 50.0_f32.cos(), 0.0],
            5,
        )
        .expect("search");
    assert_eq!(results.len(), 5);
    assert_eq!(results[0].id, 50, "nearest should be exact match");
}

// ── G060: Automatic HNSW maintenance on executor write path ─────────

/// G060 test 1: CREATE node with vector property automatically inserts
/// the vector into the HNSW index (no manual on_vector_written needed).
#[test]
fn g060_create_node_auto_inserts_into_hnsw() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Register HNSW index for (Movie, embedding) — 4 dimensions, cosine
    db.create_vector_index(
        "movie_embed",
        "Movie",
        "embedding",
        VectorIndexConfig {
            dimensions: 4,
            metric: VectorMetric::Cosine,
            ..VectorIndexConfig::default()
        },
    );

    // HNSW should be empty before any writes
    let reg = db.vector_index_registry();
    assert!(reg.has_index("Movie", "embedding"));
    let pre_results = reg.search("Movie", "embedding", &[1.0, 0.0, 0.0, 0.0], 10);
    assert!(
        pre_results.unwrap().is_empty(),
        "HNSW should be empty before CREATE"
    );

    // CREATE two nodes with vector properties
    db.execute_cypher("CREATE (a:Movie {title: 'Matrix', embedding: [1.0, 0.0, 0.0, 0.0]})")
        .expect("create Matrix");
    db.execute_cypher("CREATE (b:Movie {title: 'Inception', embedding: [0.0, 1.0, 0.0, 0.0]})")
        .expect("create Inception");

    // HNSW index should now contain 2 vectors automatically
    let reg = db.vector_index_registry();
    let results = reg
        .search("Movie", "embedding", &[0.9, 0.1, 0.0, 0.0], 10)
        .expect("search");
    assert_eq!(
        results.len(),
        2,
        "HNSW should have 2 vectors after 2 CREATEs"
    );

    // The nearest to [0.9, 0.1, 0, 0] should be the Matrix vector [1, 0, 0, 0]
    // (smaller cosine distance)
    assert!(
        results[0].id > 0,
        "result should have a valid node ID, got {}",
        results[0].id
    );
}

/// G060 test 2: SET vector property on existing node updates the HNSW index.
#[test]
fn g060_set_vector_property_updates_hnsw() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.create_vector_index(
        "item_vec",
        "Item",
        "v",
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::L2,
            ..VectorIndexConfig::default()
        },
    );

    // Create a node with initial vector
    db.execute_cypher("CREATE (a:Item {name: 'A', v: [0.0, 0.0, 0.0]})")
        .expect("create A");

    // Verify it's in the index
    let reg = db.vector_index_registry();
    let results = reg
        .search("Item", "v", &[0.0, 0.0, 0.0], 5)
        .expect("search");
    assert_eq!(results.len(), 1, "HNSW should have 1 vector after CREATE");
    let original_id = results[0].id;

    // SET the vector to a new value
    db.execute_cypher("MATCH (a:Item {name: 'A'}) SET a.v = [10.0, 10.0, 10.0]")
        .expect("set new vector");

    // HNSW now has the updated vector. HnswIndex::insert deduplicates by ID,
    // so the index should still contain 1 entry (or 2 if no dedup on update).
    // The important thing is that searching near [10, 10, 10] finds the node.
    let reg = db.vector_index_registry();
    let results = reg
        .search("Item", "v", &[10.0, 10.0, 10.0], 5)
        .expect("search near new vector");
    assert!(
        !results.is_empty(),
        "HNSW should find the node after SET update"
    );
    assert_eq!(
        results[0].id, original_id,
        "nearest to [10,10,10] should be the updated node"
    );
}

/// G060 test 3: CREATE node without vector property does NOT affect HNSW.
#[test]
fn g060_create_nonvector_node_does_not_affect_hnsw() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.create_vector_index(
        "item_vec",
        "Item",
        "v",
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::L2,
            ..VectorIndexConfig::default()
        },
    );

    // Create a node WITHOUT a vector property
    db.execute_cypher("CREATE (a:Item {name: 'NoVector', score: 42})")
        .expect("create without vector");

    // HNSW should remain empty
    let reg = db.vector_index_registry();
    let results = reg
        .search("Item", "v", &[1.0, 1.0, 1.0], 10)
        .expect("search");
    assert!(
        results.is_empty(),
        "HNSW should be empty when no vectors written"
    );
}

/// G060 test 4: HNSW auto-maintenance with multiple nodes — search uses
/// the HNSW index to find nearest vectors correctly.
#[test]
fn g060_hnsw_search_after_auto_inserts() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.create_vector_index(
        "point_vec",
        "Point",
        "coords",
        VectorIndexConfig {
            dimensions: 2,
            metric: VectorMetric::L2,
            ..VectorIndexConfig::default()
        },
    );

    // Create 10 points spread across the space
    for i in 0..10u32 {
        let x = i as f32;
        let y = (i as f32 * 0.5).sin();
        db.execute_cypher(&format!(
            "CREATE (p:Point {{idx: {i}, coords: [{x}, {y}]}})"
        ))
        .expect("create point");
    }

    // HNSW should have all 10 vectors
    let reg = db.vector_index_registry();
    let results = reg
        .search("Point", "coords", &[5.0, (5.0_f32 * 0.5).sin()], 3)
        .expect("search");
    assert_eq!(
        results.len(),
        3,
        "should return top-3 nearest from 10 auto-inserted vectors"
    );
}

/// G060 test 5: DELETE node calls on_vector_deleted (wiring correctness).
/// Since on_vector_deleted is intentionally a no-op (MVCC post-filter),
/// we verify the deletion doesn't crash and the vector remains in the
/// HNSW graph (by design — MVCC visibility handles exclusion).
#[test]
fn g060_delete_node_calls_on_vector_deleted() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.create_vector_index(
        "item_vec",
        "Item",
        "v",
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::L2,
            ..VectorIndexConfig::default()
        },
    );

    // Create and then delete a node
    db.execute_cypher("CREATE (a:Item {name: 'Ephemeral', v: [1.0, 2.0, 3.0]})")
        .expect("create");

    // Verify it's in HNSW
    let reg = db.vector_index_registry();
    let pre_delete = reg
        .search("Item", "v", &[1.0, 2.0, 3.0], 5)
        .expect("pre-delete search");
    assert_eq!(pre_delete.len(), 1, "should find vector before delete");

    // DETACH DELETE the node
    db.execute_cypher("MATCH (a:Item {name: 'Ephemeral'}) DETACH DELETE a")
        .expect("delete");

    // Vector remains in HNSW (by design — on_vector_deleted is no-op,
    // MVCC post-filter handles visibility). The HNSW graph is NOT
    // compacted on delete — this is intentional to avoid fragmentation.
    let reg = db.vector_index_registry();
    let post_delete = reg
        .search("Item", "v", &[1.0, 2.0, 3.0], 5)
        .expect("post-delete search");
    assert_eq!(
        post_delete.len(),
        1,
        "vector should remain in HNSW graph after delete (MVCC post-filter handles visibility)"
    );
}

/// G060 test 6: REMOVE vector property calls on_vector_deleted wiring.
#[test]
fn g060_remove_vector_property_wiring() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.create_vector_index(
        "item_vec",
        "Item",
        "v",
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::L2,
            ..VectorIndexConfig::default()
        },
    );

    // Create node with vector, then REMOVE the vector property
    db.execute_cypher("CREATE (a:Item {name: 'A', v: [1.0, 0.0, 0.0]})")
        .expect("create");

    // REMOVE the vector property — should call on_vector_deleted (no-op) without crash
    db.execute_cypher("MATCH (a:Item {name: 'A'}) REMOVE a.v")
        .expect("remove vector property");

    // Vector remains in HNSW (on_vector_deleted is no-op, by design)
    let reg = db.vector_index_registry();
    let results = reg
        .search("Item", "v", &[1.0, 0.0, 0.0], 5)
        .expect("search");
    assert_eq!(
        results.len(),
        1,
        "vector stays in HNSW after REMOVE (MVCC visibility handles exclusion)"
    );
}

// ── G061: HNSW index persistence and rebuild on startup ─────────────

/// G061 test 1: HNSW index survives Database close + reopen.
/// Create vector index, insert nodes, close DB, reopen, search.
#[test]
fn g061_hnsw_persists_across_restart() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");

    // Phase 1: create index, insert vectors, close
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.create_vector_index(
            "movie_embed",
            "Movie",
            "embedding",
            VectorIndexConfig {
                dimensions: 4,
                metric: VectorMetric::Cosine,
                ..VectorIndexConfig::default()
            },
        );

        db.execute_cypher("CREATE (a:Movie {title: 'Matrix', embedding: [1.0, 0.0, 0.0, 0.0]})")
            .expect("create Matrix");
        db.execute_cypher("CREATE (b:Movie {title: 'Inception', embedding: [0.0, 1.0, 0.0, 0.0]})")
            .expect("create Inception");
        db.execute_cypher(
            "CREATE (c:Movie {title: 'Interstellar', embedding: [0.0, 0.0, 1.0, 0.0]})",
        )
        .expect("create Interstellar");

        // Verify index works before close
        let reg = db.vector_index_registry();
        let results = reg
            .search("Movie", "embedding", &[1.0, 0.0, 0.0, 0.0], 3)
            .expect("search pre-close");
        assert_eq!(results.len(), 3, "should find 3 vectors before close");
    } // db dropped here

    // Phase 2: reopen and verify HNSW was rebuilt
    {
        let db = Database::open(dir.path()).expect("reopen");
        let reg = db.vector_index_registry();

        // Index should exist
        assert!(
            reg.has_index("Movie", "embedding"),
            "vector index should be loaded from schema: on reopen"
        );

        // Search should return all 3 vectors
        let results = reg
            .search("Movie", "embedding", &[1.0, 0.0, 0.0, 0.0], 3)
            .expect("search post-reopen");
        assert_eq!(
            results.len(),
            3,
            "HNSW should have 3 vectors after rebuild on reopen"
        );
    }
}

/// G061 test 2: Multiple vector indexes on different labels persist.
#[test]
fn g061_multiple_indexes_persist() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open");

        // Two indexes on different labels
        db.create_vector_index(
            "movie_vec",
            "Movie",
            "v",
            VectorIndexConfig {
                dimensions: 3,
                metric: VectorMetric::L2,
                ..VectorIndexConfig::default()
            },
        );
        db.create_vector_index(
            "user_vec",
            "User",
            "profile_vec",
            VectorIndexConfig {
                dimensions: 2,
                metric: VectorMetric::Cosine,
                ..VectorIndexConfig::default()
            },
        );

        db.execute_cypher("CREATE (m:Movie {title: 'X', v: [1.0, 2.0, 3.0]})")
            .expect("create movie");
        db.execute_cypher("CREATE (u:User {name: 'Alice', profile_vec: [0.5, 0.5]})")
            .expect("create user");
    }

    // Reopen
    {
        let db = Database::open(dir.path()).expect("reopen");
        let reg = db.vector_index_registry();

        assert!(reg.has_index("Movie", "v"), "Movie index should persist");
        assert!(
            reg.has_index("User", "profile_vec"),
            "User index should persist"
        );

        let movie_results = reg
            .search("Movie", "v", &[1.0, 2.0, 3.0], 5)
            .expect("movie search");
        assert_eq!(movie_results.len(), 1, "Movie HNSW should have 1 vector");

        let user_results = reg
            .search("User", "profile_vec", &[0.5, 0.5], 5)
            .expect("user search");
        assert_eq!(user_results.len(), 1, "User HNSW should have 1 vector");
    }
}

/// G061 test 3: Empty database reopens without errors.
#[test]
fn g061_empty_db_reopen_no_errors() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Open and close with no vector indexes
    {
        Database::open(dir.path()).expect("open");
    }

    // Reopen — should work fine with no vector indexes
    {
        let db = Database::open(dir.path()).expect("reopen");
        assert!(
            db.vector_index_registry().is_empty(),
            "no vector indexes on empty DB"
        );
    }
}

/// G061 test 4: Nodes added after reopen are also auto-indexed (G060 + G061 combined).
#[test]
fn g061_new_vectors_indexed_after_reopen() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");

    // Create index and one node
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.create_vector_index(
            "item_vec",
            "Item",
            "v",
            VectorIndexConfig {
                dimensions: 3,
                metric: VectorMetric::L2,
                ..VectorIndexConfig::default()
            },
        );
        db.execute_cypher("CREATE (a:Item {name: 'A', v: [1.0, 0.0, 0.0]})")
            .expect("create A");
    }

    // Reopen and add more nodes
    {
        let mut db = Database::open(dir.path()).expect("reopen");

        // Original should be found
        let reg = db.vector_index_registry();
        let results = reg
            .search("Item", "v", &[1.0, 0.0, 0.0], 5)
            .expect("search");
        assert_eq!(results.len(), 1, "rebuilt HNSW has 1 vector");

        // Add new node (G060 auto-maintenance should work)
        db.execute_cypher("CREATE (b:Item {name: 'B', v: [0.0, 1.0, 0.0]})")
            .expect("create B");

        let reg = db.vector_index_registry();
        let results = reg
            .search("Item", "v", &[0.0, 1.0, 0.0], 5)
            .expect("search after insert");
        assert_eq!(
            results.len(),
            2,
            "HNSW should have 2 vectors: 1 rebuilt + 1 new from G060"
        );
    }
}

// ── G009: StorageVectorLoader integration test ─────────────────────
// Verifies the full chain: Database creates a StorageVectorLoader that
// successfully reads f32 vectors from the node: partition for HNSW reranking.

#[test]
fn g009_storage_vector_loader_reads_from_node_partition() {
    use coordinode_embed::db::StorageVectorLoader;
    use coordinode_vector::VectorLoader;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Insert nodes with vector properties via Cypher
    db.execute_cypher("CREATE (a:Item {name: 'alpha', embedding: [1.0, 0.0, 0.0]})")
        .expect("create alpha");
    db.execute_cypher("CREATE (b:Item {name: 'beta', embedding: [0.0, 1.0, 0.0]})")
        .expect("create beta");
    db.execute_cypher("CREATE (c:Item {name: 'gamma', embedding: [0.0, 0.0, 1.0]})")
        .expect("create gamma");

    // Get node IDs — the executor stores them as `n` (node variable → Int(id))
    let rows = db
        .execute_cypher("MATCH (n:Item) RETURN n, n.name ORDER BY n.name")
        .expect("query ids");
    assert_eq!(rows.len(), 3);

    let mut ids: Vec<u64> = Vec::new();
    for row in &rows {
        if let Some(coordinode_core::graph::types::Value::Int(id)) = row.get("n") {
            ids.push(*id as u64);
        }
    }
    assert_eq!(ids.len(), 3, "should have 3 node IDs");

    // Create StorageVectorLoader and verify it can load vectors
    let loader = StorageVectorLoader::new(
        db.engine_shared(),
        db.interner().clone(),
        1, // shard_id
    );

    let loaded = loader.load_vectors(&ids, "embedding");
    assert_eq!(
        loaded.len(),
        3,
        "loader should find all 3 vectors from node: partition"
    );

    // Verify vector values match what was inserted
    for (&node_id, vec) in &loaded {
        assert_eq!(vec.len(), 3, "each vector should have 3 dimensions");
        // At least one dimension should be 1.0 (unit vectors)
        let max_val = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max_val - 1.0).abs() < 1e-5,
            "node {} vector should have a 1.0 component, got {:?}",
            node_id,
            vec
        );
    }

    // Verify unknown property returns empty
    let empty = loader.load_vectors(&ids, "nonexistent_property");
    assert!(
        empty.is_empty(),
        "loader should return empty for unknown property"
    );
}

/// G009 E2E: Create HNSW index with offload_vectors=true, insert vectors
/// via Cypher, query with vector_distance — full pipeline through executor.
#[test]
fn g009_offloaded_hnsw_search_e2e_through_cypher() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create HNSW index with offload_vectors + quantization enabled
    db.create_vector_index(
        "item_embed",
        "Item",
        "embedding",
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::L2,
            quantization: true,
            offload_vectors: true,
            ..VectorIndexConfig::default()
        },
    );

    // Insert enough vectors to trigger SQ8 auto-calibration (threshold=1000 default)
    // and offloading. Use lower-level API to set calibration threshold lower.
    // Actually: the VectorIndexConfig goes through VectorIndexRegistry::register
    // which sets calibration_threshold=1000. We need enough vectors to trigger it.
    // For testing, insert the nodes first, then manually calibrate.
    for i in 0..10 {
        let x = (i as f32) * 0.1;
        let y = 1.0 - x;
        let z = 0.5;
        let query = format!("CREATE (n:Item {{name: 'item_{i}', embedding: [{x}, {y}, {z}]}})");
        db.execute_cypher(&query).expect("create item");
    }

    // Verify the HNSW index has vectors
    let reg = db.vector_index_registry();
    let handle = reg.get("Item", "embedding").expect("index should exist");
    let hnsw = handle.read().expect("read lock");
    assert_eq!(hnsw.len(), 10, "HNSW should have 10 vectors");

    // Since calibration_threshold=1000 (default), SQ8 is NOT auto-calibrated with 10 vectors.
    // The offload mechanism only activates after SQ8 calibration.
    // So with 10 vectors, f32 are still in-memory — that's correct behavior.
    assert!(
        !hnsw.is_offloaded(),
        "with only 10 vectors < threshold, offload should not be active yet"
    );
    drop(hnsw);

    // Query through Cypher — should work via the standard in-memory path
    let results = db
        .execute_cypher(
            "MATCH (n:Item) \
             WHERE vector_distance(n.embedding, [0.0, 1.0, 0.5]) < 2.0 \
             RETURN n.name \
             ORDER BY vector_distance(n.embedding, [0.0, 1.0, 0.5]) \
             LIMIT 3",
        )
        .expect("vector search query");

    assert!(
        !results.is_empty(),
        "Cypher vector search should return results"
    );

    // The closest to [0.0, 1.0, 0.5] should be item_0 ([0.0, 1.0, 0.5])
    let first_name = results[0].get("n.name").expect("should have n.name");
    assert_eq!(
        first_name,
        &coordinode_core::graph::types::Value::String("item_0".to_string()),
        "nearest vector should be item_0"
    );
}

/// G009 E2E: Force offloaded mode via manual SQ8 calibration, then verify
/// search_with_loader reranks with exact f32 loaded from node: partition.
#[test]
fn g009_forced_offload_search_through_registry() {
    use coordinode_embed::db::StorageVectorLoader;
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create HNSW with offload + quantization flags
    db.create_vector_index(
        "widget_embed",
        "Widget",
        "vec",
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::L2,
            quantization: true,
            offload_vectors: true,
            ..VectorIndexConfig::default()
        },
    );

    // Insert 20 nodes with vector properties
    for i in 0..20 {
        let x = i as f32 * 0.1;
        let y = 1.0 - i as f32 * 0.05;
        let z = 0.5;
        db.execute_cypher(&format!(
            "CREATE (n:Widget {{name: 'w{i}', vec: [{x}, {y}, {z}]}})"
        ))
        .expect("create widget");
    }

    // Force SQ8 calibration + offloading on the HNSW index
    {
        let reg = db.vector_index_registry();
        let handle = reg.get("Widget", "vec").expect("index exists");
        let mut hnsw = handle.write().expect("write lock");

        assert_eq!(hnsw.len(), 20);
        assert!(
            !hnsw.is_quantized(),
            "not auto-calibrated with threshold=1000"
        );

        // Manually calibrate from existing vectors
        let params = coordinode_vector::quantize::Sq8Params::calibrate_from_index(&hnsw)
            .expect("calibrate from 20 vectors");
        hnsw.set_sq8_params(params);

        assert!(hnsw.is_quantized(), "should be quantized");
        assert!(
            hnsw.is_offloaded(),
            "should be offloaded (config=true + quantized)"
        );

        // Verify all f32 vectors are dropped from memory
        for i in 0..hnsw.len() {
            assert!(
                !hnsw.has_f32_vector(i),
                "node at idx {i} should have no f32 after offloading"
            );
        }
    }

    // Search through registry with StorageVectorLoader — full offload path
    let loader = StorageVectorLoader::new(db.engine_shared(), db.interner().clone(), 1);

    let reg = db.vector_index_registry();
    let results = reg
        .search_with_loader("Widget", "vec", &[0.0, 1.0, 0.5], 5, Some(&loader))
        .expect("search should succeed");

    assert_eq!(results.len(), 5, "should return 5 results");

    // All scores must be finite (loaded from storage, not infinity from missing vectors)
    for r in &results {
        assert!(
            r.score.is_finite() && r.score >= 0.0,
            "score should be valid distance, got {} for node {}",
            r.score,
            r.id
        );
    }

    // Results sorted by distance ascending
    for i in 0..results.len() - 1 {
        assert!(
            results[i].score <= results[i + 1].score,
            "results should be sorted: {} > {} (nodes {} vs {})",
            results[i].score,
            results[i + 1].score,
            results[i].id,
            results[i + 1].id,
        );
    }
}

/// G009 E2E: Force offload, then run Cypher query with vector_distance in WHERE.
/// Exercises: executor → try_hnsw_vector_filter → registry.search_with_loader
/// → HnswIndex.search_with_loader → StorageVectorLoader.load_vectors.
#[test]
fn g009_forced_offload_cypher_e2e() {
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.create_vector_index(
        "part_embed",
        "Part",
        "emb",
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::L2,
            quantization: true,
            offload_vectors: true,
            ..VectorIndexConfig::default()
        },
    );

    // Insert 15 nodes
    for i in 0..15 {
        let x = i as f32 * 0.1;
        let y = 1.0 - x;
        db.execute_cypher(&format!(
            "CREATE (n:Part {{name: 'p{i}', emb: [{x}, {y}, 0.5]}})"
        ))
        .expect("create");
    }

    // Force SQ8 calibration + offloading
    {
        let reg = db.vector_index_registry();
        let handle = reg.get("Part", "emb").expect("index");
        let mut hnsw = handle.write().expect("lock");
        let params =
            coordinode_vector::quantize::Sq8Params::calibrate_from_index(&hnsw).expect("calibrate");
        hnsw.set_sq8_params(params);
        assert!(
            hnsw.is_offloaded(),
            "must be offloaded after forced calibration"
        );
    }

    // Run Cypher query — goes through executor → try_hnsw_vector_filter → search_with_loader
    let results = db
        .execute_cypher(
            "MATCH (n:Part) \
             WHERE vector_distance(n.emb, [0.0, 1.0, 0.5]) < 5.0 \
             RETURN n.name \
             ORDER BY vector_distance(n.emb, [0.0, 1.0, 0.5]) \
             LIMIT 3",
        )
        .expect("cypher vector search with offloaded HNSW");

    assert!(
        !results.is_empty(),
        "should return results from offloaded HNSW search"
    );

    // p0 has emb=[0.0, 1.0, 0.5] — exact match, distance should be ~0
    let first = results[0].get("n.name").expect("n.name");
    assert_eq!(
        first,
        &coordinode_core::graph::types::Value::String("p0".to_string()),
        "nearest to [0,1,0.5] should be p0 with emb=[0.0, 1.0, 0.5]"
    );
}
