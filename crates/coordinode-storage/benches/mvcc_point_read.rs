//! Benchmark: MVCC point read latency (R070).
//!
//! Measures native seqno MVCC read performance on hot keys with many versions.
//! Target (ADR-016, V2): 1-3µs point read on hot keys with 100+ versions.
//!
//! Scenarios:
//!   1. point_read/latest — read current value of key with N versions
//!   2. point_read/historical — snapshot_at(old_ts) + get on key with N versions
//!   3. write_throughput — sequential puts to measure write path overhead

#![allow(clippy::expect_used)]

use std::sync::Arc;

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

/// Write N versions of a single key, persist, return (engine, seqnos).
fn setup_versioned_key(num_versions: usize) -> (StorageEngine, tempfile::TempDir, Vec<u64>) {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle).expect("open");

    let mut seqnos = Vec::with_capacity(num_versions);
    let value_buf = [0xABu8; 256]; // 256-byte value (typical node record)

    for i in 0..num_versions {
        // Vary the value slightly so LSM doesn't deduplicate
        let mut val = value_buf;
        val[0] = (i & 0xFF) as u8;
        val[1] = ((i >> 8) & 0xFF) as u8;
        engine
            .put(Partition::Node, b"node:0:hotkey", &val)
            .expect("put");
        seqnos.push(engine.snapshot());
    }

    // Persist to SST so reads go through the full LSM path
    engine.persist().expect("persist");

    (engine, dir, seqnos)
}

/// Benchmark: read latest value of a key with N historical versions.
///
/// This is the primary metric from ADR-016: "point read on hot key".
/// With native seqno MVCC, this should be O(1) regardless of version count.
fn bench_point_read_latest(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/mvcc_point_read/latest");
    group.sample_size(100);

    for &num_versions in &[1, 10, 100, 500] {
        let (engine, _dir, _seqnos) = setup_versioned_key(num_versions);

        group.bench_with_input(
            BenchmarkId::new("versions", num_versions),
            &num_versions,
            |b, _| {
                b.iter(|| {
                    let val = engine.get(Partition::Node, b"node:0:hotkey").expect("get");
                    assert!(val.is_some());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: snapshot_at historical read on key with N versions.
///
/// Read from the midpoint of the version history. Measures the cost
/// of point-in-time reads which should also be O(1) with native seqno.
fn bench_point_read_historical(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/mvcc_point_read/historical");
    group.sample_size(100);

    for &num_versions in &[10, 100, 500] {
        let (engine, _dir, seqnos) = setup_versioned_key(num_versions);
        // Read at the midpoint of version history
        let mid_seqno = seqnos[num_versions / 2];

        group.bench_with_input(
            BenchmarkId::new("versions", num_versions),
            &num_versions,
            |b, _| {
                b.iter(|| {
                    let snap = engine.snapshot_at(mid_seqno).expect("snapshot_at");
                    let val = engine
                        .snapshot_get(&snap, Partition::Node, b"node:0:hotkey")
                        .expect("snapshot_get");
                    assert!(val.is_some());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: sequential write throughput (puts per second).
///
/// Measures the oracle-driven write path cost. Target (V2): 50-80K/s.
fn bench_write_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/mvcc_write");
    group.sample_size(20);

    for &batch_size in &[100, 1000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("sequential_puts", batch_size),
            &batch_size,
            |b, &n| {
                b.iter(|| {
                    let dir = tempfile::TempDir::new().expect("tempdir");
                    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
                    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
                        "default",
                        dir.path(),
                        Media::Hdd,
                        Durability::Durable,
                        Tier::Warm,
                    )]);
                    let engine = StorageEngine::open_with_oracle(&config, oracle).expect("open");

                    let value = [0xCDu8; 256];
                    for i in 0..n {
                        let key = format!("node:0:{i}");
                        engine
                            .put(Partition::Node, key.as_bytes(), &value)
                            .expect("put");
                    }
                    engine.persist().expect("persist");
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_point_read_latest,
    bench_point_read_historical,
    bench_write_throughput
);
criterion_main!(benches);
