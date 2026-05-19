//! Modality store throughput benchmarks.
//!
//! Tight micro-benchmarks for the operations on the query hot path.
//! Not exhaustive — focused on the surfaces R165 will dispatch to from
//! the query layer (knn, bbox scan, posting-list put/scan). Per
//! CLAUDE.md engineering principles these measure throughput AND
//! tail-friendliness (criterion histograms).

#![allow(clippy::unwrap_used)]

use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_modality::{
    Bbox, Crs, EdgeStore, LocalEdgeStore, LocalNodeStore, LocalSpatialStore, LocalVectorStore,
    NodeStore, Point, SpatialStore, VectorStore,
};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_vector::hnsw::HnswConfig;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::TempDir;

fn mk_engine() -> (TempDir, StorageEngine) {
    let dir = TempDir::new().unwrap();
    let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&cfg).unwrap();
    (dir, engine)
}

/// Node put throughput at varying record sizes — small overhead dominated
/// by MessagePack encode + LSM put.
fn bench_node_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_put");
    group.sample_size(20);
    let (_dir, engine) = mk_engine();
    let store = LocalNodeStore::new(&engine);
    let record = NodeRecord::new("User");

    group.throughput(Throughput::Elements(1));
    group.bench_function("single_put", |b| {
        let mut id = 0u64;
        b.iter(|| {
            store.put(0, NodeId::from_raw(id), &record).unwrap();
            id = id.wrapping_add(1);
        });
    });
    group.finish();
}

/// Edge super-node: scaling neighbour-list scan with posting list size.
fn bench_edge_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_scan_neighbours");
    group.sample_size(10);
    let src = NodeId::from_raw(1);

    for &n in &[100usize, 1000, 10_000] {
        // Build a fresh engine per size so prior runs don't pollute.
        let (_dir, engine) = mk_engine();
        let store = LocalEdgeStore::new(&engine);
        for i in 0..n as u64 {
            store
                .put_edge("F", src, NodeId::from_raw(i + 1000), None)
                .unwrap();
        }
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| store.scan_neighbors_out("F", src).unwrap())
        });
    }
    group.finish();
}

/// Vector KNN at varying index sizes — exercises the HNSW search path
/// the query planner calls from runner.rs.
fn bench_vector_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_knn");
    group.sample_size(20);
    let cfg = HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ef_search: 32,
        max_dimensions: 16,
        ..HnswConfig::default()
    };

    for &n in &[100usize, 1000, 10_000] {
        let store = LocalVectorStore::new(cfg.clone());
        for i in 0..n as u64 {
            // Spread points around the unit sphere deterministically.
            let theta = (i as f32) * 0.123;
            let phi = (i as f32) * 0.0789;
            store
                .insert(i, vec![theta.cos(), theta.sin(), phi.cos()])
                .unwrap();
        }
        let query = vec![0.5f32, 0.5, 0.5];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| store.knn_search(&query, 10).unwrap())
        });
    }
    group.finish();
}

/// Spatial bbox scan at varying point counts. Exercises the curve
/// windowing + post-filter pipeline.
fn bench_spatial_bbox(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_scan_within_bbox");
    group.sample_size(10);

    for &n in &[100usize, 1000, 10_000] {
        let (_dir, engine) = mk_engine();
        let store = LocalSpatialStore::new(&engine);
        // Spread points across a 1000x1000 square.
        for i in 0..n as u64 {
            let x = ((i * 37) % 1000) as f64;
            let y = ((i * 91) % 1000) as f64;
            store
                .insert(
                    1,
                    NodeId::from_raw(i + 1),
                    &Point::new_2d(Crs::Cartesian2d, x, y),
                )
                .unwrap();
        }
        let bbox = Bbox {
            // ~1% of the population area => early-break should kick in.
            lower: Point::new_2d(Crs::Cartesian2d, 0.0, 0.0),
            upper: Point::new_2d(Crs::Cartesian2d, 100.0, 100.0),
        };
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| store.scan_within_bbox(1, Crs::Cartesian2d, &bbox).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_node_put,
    bench_edge_scan,
    bench_vector_knn,
    bench_spatial_bbox
);
criterion_main!(benches);
