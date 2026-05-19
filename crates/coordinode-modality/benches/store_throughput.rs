//! Modality store throughput benchmarks.
//!
//! Tight micro-benchmarks for the operations on the query hot path.
//! Not exhaustive — focused on the surfaces R165 will dispatch to from
//! the query layer (knn, bbox scan, posting-list put/scan). Per
//! CLAUDE.md engineering principles these measure throughput AND
//! tail-friendliness (criterion histograms).

#![allow(clippy::unwrap_used)]

use coordinode_core::graph::blob::ChunkId;
use coordinode_core::graph::doc_delta::PathTarget;
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_modality::{
    Bbox, BlobStore, Bucket, Crs, DocumentStore, EdgeStore, IndexStore, LocalBlobStore,
    LocalDocumentStore, LocalEdgeStore, LocalIndexStore, LocalNodeStore, LocalSpatialStore,
    LocalTimeSeriesStore, LocalVectorStore, Measurement, NodeStore, Point, SpatialStore,
    TimeSeriesStore, VectorStore,
};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_vector::hnsw::HnswConfig;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::BTreeMap;
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

/// Index throughput: single-column put + exact lookup.
fn bench_index_put_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_put_then_scan_exact");
    group.sample_size(10);
    for &n in &[100usize, 1000, 10_000] {
        let (_dir, engine) = mk_engine();
        let store = LocalIndexStore::new(&engine);
        for i in 0..n as u64 {
            store
                .put_entry("by_id", &[Value::Int(i as i64)], NodeId::from_raw(i))
                .unwrap();
        }
        let probe = (n / 2) as i64;
        group.bench_with_input(BenchmarkId::from_parameter(n), &probe, |b, p| {
            b.iter(|| store.scan_exact("by_id", &[Value::Int(*p)]).unwrap())
        });
    }
    group.finish();
}

/// Document merge throughput: set_path delta on cold node.
fn bench_document_set_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_set_path");
    group.sample_size(20);
    let (_dir, engine) = mk_engine();
    let docs = LocalDocumentStore::new(&engine);
    let nodes = LocalNodeStore::new(&engine);
    nodes
        .put(0, NodeId::from_raw(1), &NodeRecord::new("L"))
        .unwrap();
    group.bench_function("single_set_path", |b| {
        let mut i = 0u64;
        b.iter(|| {
            docs.set_path(
                0,
                NodeId::from_raw(1),
                PathTarget::Extra,
                vec![format!("k{i}")],
                rmpv::Value::Integer((i as i64).into()),
            )
            .unwrap();
            i = i.wrapping_add(1);
        });
    });
    group.finish();
}

/// TimeSeries bucket put + get round-trip.
fn bench_timeseries_bucket(c: &mut Criterion) {
    let mut group = c.benchmark_group("timeseries_bucket_round_trip");
    group.sample_size(10);
    for &n in &[10usize, 100, 1000] {
        let (_dir, engine) = mk_engine();
        let store = LocalTimeSeriesStore::new(&engine);
        let mut measurements = Vec::with_capacity(n);
        for i in 0..n {
            let mut fields = BTreeMap::new();
            fields.insert("v".into(), i as f64);
            measurements.push(Measurement {
                timestamp_us: i as i64 * 1000,
                fields,
            });
        }
        let bucket = Bucket::from_measurements(rmpv::Value::Nil, measurements);
        store.put_bucket(0, NodeId::from_raw(1), &bucket).unwrap();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| store.get_bucket(0, NodeId::from_raw(1)).unwrap())
        });
    }
    group.finish();
}

/// Blob chunk round-trip at varying payload sizes.
fn bench_blob_chunk(c: &mut Criterion) {
    let mut group = c.benchmark_group("blob_chunk_round_trip");
    group.sample_size(20);
    for &n in &[1024usize, 8 * 1024, 64 * 1024] {
        let (_dir, engine) = mk_engine();
        let store = LocalBlobStore::new(&engine);
        let payload = vec![0xab_u8; n];
        let id = ChunkId::from_data(&payload);
        store.put_chunk(&id, &payload).unwrap();
        group.throughput(Throughput::Bytes(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| store.get_chunk(&id).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_node_put,
    bench_edge_scan,
    bench_vector_knn,
    bench_spatial_bbox,
    bench_index_put_scan,
    bench_document_set_path,
    bench_timeseries_bucket,
    bench_blob_chunk
);
criterion_main!(benches);
