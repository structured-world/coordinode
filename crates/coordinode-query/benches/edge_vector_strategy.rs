//! Criterion benchmarks for edge vector plan selection.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//!
//! Measures:
//! 1. Strategy selection throughput (pure decision logic)
//! 2. Edge vector query EXPLAIN generation
//! 3. End-to-end edge vector query with real storage

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use coordinode_query::planner::logical::select_edge_vector_strategy;

/// Benchmark the strategy selection function itself.
/// This is a pure function — measures decision logic throughput.
fn bench_strategy_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector/edge_strategy_selection");

    let scenarios: Vec<(&str, f64, f64)> = vec![
        ("low_fanout_50", 50.0, 0.5),
        ("low_fanout_199", 199.0, 0.01),
        ("mid_fanout_graph_first", 500.0, 0.05),
        ("mid_fanout_vector_first", 500.0, 0.005),
        ("high_fanout_10k", 10_001.0, 0.5),
        ("high_fanout_100k", 100_000.0, 0.01),
    ];

    for (name, fan_out, selectivity) in &scenarios {
        group.bench_with_input(BenchmarkId::new("decide", name), name, |b, _| {
            b.iter(|| {
                let strategy = select_edge_vector_strategy(*fan_out, *selectivity);
                std::hint::black_box(strategy);
            });
        });
    }

    group.finish();
}

/// Benchmark EXPLAIN plan generation for edge vector queries.
/// Measures the full parse → plan → optimize → explain pipeline.
fn bench_edge_vector_explain(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector/edge_vector_explain");

    let queries = vec![
        (
            "simple_edge_vector",
            "MATCH (u:User)-[r:KNOWS]->(f) \
             WHERE vector_distance(r.embedding, [1.0, 0.0, 0.0]) < 0.3 \
             RETURN f",
        ),
        (
            "node_vector_baseline",
            "MATCH (n:Product) \
             WHERE vector_distance(n.embedding, [1.0, 0.0]) < 0.5 \
             RETURN n",
        ),
    ];

    for (name, query) in &queries {
        group.bench_with_input(BenchmarkId::new("explain", name), query, |b, q| {
            b.iter(|| {
                let ast = coordinode_query::cypher::parse(q).expect("parse");
                let plan = coordinode_query::planner::build_logical_plan(&ast).expect("plan");
                let explain = plan.explain();
                std::hint::black_box(explain);
            });
        });
    }

    group.finish();
}

/// Benchmark end-to-end edge vector query execution with real CoordiNode storage.
/// Creates a small graph with edge vectors, then queries it.
fn bench_edge_vector_execution(c: &mut Criterion) {
    use coordinode_embed::Database;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Setup: create nodes and edges with vectors
    db.execute_cypher("CREATE (a:User {name: 'Alice'})")
        .expect("create");
    db.execute_cypher("CREATE (b:User {name: 'Bob'})")
        .expect("create");

    // Create 20 edges with vector properties to simulate moderate fan-out
    for i in 0..20 {
        let v0 = (i as f64 * 0.05).sin() as f32;
        let v1 = (i as f64 * 0.05).cos() as f32;
        db.execute_cypher(&format!(
            "MATCH (a:User {{name: 'Alice'}}), (b:User {{name: 'Bob'}}) \
             CREATE (a)-[:KNOWS {{embedding: [{v0}, {v1}, 0.0]}}]->(b)"
        ))
        .expect("create edge");
    }

    let mut group = c.benchmark_group("vector/edge_vector_execution");

    group.bench_function("edge_vector_query_20_edges", |b| {
        b.iter(|| {
            let results = db
                .execute_cypher(
                    "MATCH (a:User {name: 'Alice'})-[r:KNOWS]->(f) \
                     WHERE vector_distance(r.embedding, [1.0, 0.0, 0.0]) < 0.5 \
                     RETURN f.name",
                )
                .expect("query");
            std::hint::black_box(results);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_strategy_selection,
    bench_edge_vector_explain,
    bench_edge_vector_execution
);
criterion_main!(benches);
