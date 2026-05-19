//! Coordinator dispatch overhead — measures the cost of the
//! partition-handle lookup and the seqno-stamp on the hot read/write
//! paths. R164 introduced one extra struct boundary on every access;
//! this bench pins the additional cost (should be in the
//! 10-100 ns/op range — single HashMap lookup + atomic load).

#![allow(clippy::unwrap_used)]

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use criterion::{criterion_group, criterion_main, Criterion};
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

fn bench_tree_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinator_tree_lookup");
    group.sample_size(50);
    let (_dir, engine) = mk_engine();
    group.bench_function("partition_node", |b| {
        b.iter(|| engine.coordinator().tree(Partition::Node).unwrap())
    });
    group.finish();
}

fn bench_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinator_snapshot");
    group.sample_size(50);
    let (_dir, engine) = mk_engine();
    group.bench_function("snapshot", |b| b.iter(|| engine.coordinator().snapshot()));
    group.bench_function("current_seqno", |b| {
        b.iter(|| engine.coordinator().current_seqno())
    });
    group.finish();
}

fn bench_has_write_after(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinator_has_write_after");
    group.sample_size(50);
    let (_dir, engine) = mk_engine();
    engine.put(Partition::Node, b"k", b"v").unwrap();
    let snap = engine.snapshot();
    group.bench_function("present_key_no_writes_after", |b| {
        b.iter(|| engine.has_write_after(Partition::Node, b"k", snap).unwrap())
    });
    group.bench_function("absent_key", |b| {
        b.iter(|| {
            engine
                .has_write_after(Partition::Node, b"absent", snap)
                .unwrap()
        })
    });
    group.finish();
}

fn bench_get_through_coordinator(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinator_get");
    group.sample_size(50);
    let (_dir, engine) = mk_engine();
    engine.put(Partition::Node, b"hot", b"v").unwrap();
    group.bench_function("hit_through_engine", |b| {
        b.iter(|| engine.get(Partition::Node, b"hot").unwrap())
    });
    group.bench_function("hit_through_coordinator", |b| {
        b.iter(|| engine.coordinator().get(Partition::Node, b"hot").unwrap())
    });
    group.bench_function("miss", |b| {
        b.iter(|| engine.coordinator().get(Partition::Node, b"miss").unwrap())
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_tree_lookup,
    bench_snapshot,
    bench_has_write_after,
    bench_get_through_coordinator
);
criterion_main!(benches);
