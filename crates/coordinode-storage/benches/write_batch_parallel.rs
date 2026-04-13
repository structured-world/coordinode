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
//!      (triggers the rayon path when N >= PARALLEL_THRESHOLD = 16)

#![allow(clippy::expect_used)]

use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::config::{FlushPolicy, StorageConfig};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

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
    let mut config = StorageConfig::new(dir.path());
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
            // 2 mutations per partition × 4 partitions = 8 mutations < 16 threshold
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
/// rayon path when N >= PARALLEL_THRESHOLD (16).
fn bench_parallel_multi_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_batch");

    for n in [16_usize, 64, 256] {
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

criterion_group!(
    benches,
    bench_serial_single_partition,
    bench_serial_small_batch,
    bench_parallel_multi_partition,
);
criterion_main!(benches);
