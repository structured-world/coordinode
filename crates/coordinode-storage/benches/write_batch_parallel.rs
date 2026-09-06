//! Benchmark: parallel memtable writes within a write batch (R091).
//!
//! Measures the throughput of `WriteBatch::commit()` on multi-partition
//! workloads, comparing the serial path (< PARALLEL_THRESHOLD mutations or
//! single partition) against the parallel rayon path.
//!
//! Scenarios:
//!   1. write_batch/serial_single_partition  — N puts, one partition
//!   2. write_batch/serial_small_batch       — 8 puts across 4 partitions
//!   3. write_batch/parallel_multi_partition — N puts across 4+ partitions
//!      (triggers the rayon path when N >= PARALLEL_THRESHOLD = 1024)
//!   4. write_batch/threshold_sweep          — the same 4-partition workload
//!      on both sides of the threshold, per-mutation cost comparable

#![allow(clippy::expect_used)]

use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::config::{
    Durability, EndpointConfig, FlushPolicy, Media, StorageConfig, Tier,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// Partitions used for multi-partition benchmarks.
const MULTI_PARTITIONS: &[Partition] = &[
    Partition::Node,
    Partition::Adj,
    Partition::Schema,
    Partition::Idx,
];

/// Open a storage engine in a temporary directory with Manual flush policy.
fn open_engine() -> (StorageEngine, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.flush_policy = FlushPolicy::Manual;
    let engine = StorageEngine::open(&config).expect("open engine");
    (engine, dir)
}

/// Benchmark: all N puts go to a single partition (serial path always).
fn bench_serial_single_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_batch");

    for n in [16_usize, 64, 256] {
        group.bench_with_input(
            BenchmarkId::new("serial_single_partition", n),
            &n,
            |b, &n| {
                let (engine, _dir) = open_engine();
                let value = vec![0xABu8; 64];
                b.iter(|| {
                    let mut batch = WriteBatch::new(&engine);
                    for i in 0..n {
                        batch.put(
                            Partition::Node,
                            format!("node:0:{i}").into_bytes(),
                            value.clone(),
                        );
                    }
                    batch.commit().expect("commit");
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: 8 puts across 4 partitions — below PARALLEL_THRESHOLD (serial path).
fn bench_serial_small_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_batch");

    group.bench_function("serial_small_batch", |b| {
        let (engine, _dir) = open_engine();
        let value = vec![0xABu8; 64];
        b.iter(|| {
            let mut batch = WriteBatch::new(&engine);
            // 2 mutations per partition × 4 partitions = 8 mutations, below the threshold
            for (idx, &part) in MULTI_PARTITIONS.iter().enumerate() {
                batch.put(part, format!("{idx}:0:key").into_bytes(), value.clone());
                batch.put(part, format!("{idx}:1:key").into_bytes(), value.clone());
            }
            batch.commit().expect("commit");
        });
    });

    group.finish();
}

/// Benchmark: N puts distributed across 4 partitions — engages the parallel
/// rayon path when N >= PARALLEL_THRESHOLD (1024).
fn bench_parallel_multi_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_batch");

    for n in [1024_usize, 2048, 4096] {
        group.bench_with_input(
            BenchmarkId::new("parallel_multi_partition", n),
            &n,
            |b, &n| {
                let (engine, _dir) = open_engine();
                let value = vec![0xABu8; 64];
                b.iter(|| {
                    let mut batch = WriteBatch::new(&engine);
                    for i in 0..n {
                        let part = MULTI_PARTITIONS[i % MULTI_PARTITIONS.len()];
                        batch.put(
                            part,
                            format!("{}:0:{i}", part.name()).into_bytes(),
                            value.clone(),
                        );
                    }
                    batch.commit().expect("commit");
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: the SAME 4-partition workload on both sides of
/// `PARALLEL_THRESHOLD`, so the per-mutation cost of the serial path
/// (n < 1024) and the rayon path (n >= 1024) is directly comparable. The
/// threshold is right where the per-mutation cost of the parallel path drops
/// below the serial one.
fn bench_threshold_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_batch/threshold_sweep");

    for n in [
        8_usize, 12, 15, 16, 20, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096,
    ] {
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let (engine, _dir) = open_engine();
            let value = vec![0xABu8; 64];
            b.iter(|| {
                let mut batch = WriteBatch::new(&engine);
                for i in 0..n {
                    let part = MULTI_PARTITIONS[i % MULTI_PARTITIONS.len()];
                    batch.put(
                        part,
                        format!("{}:0:{i}", part.name()).into_bytes(),
                        value.clone(),
                    );
                }
                batch.commit().expect("commit");
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_serial_single_partition,
    bench_serial_small_batch,
    bench_parallel_multi_partition,
    bench_threshold_sweep,
);
criterion_main!(benches);
