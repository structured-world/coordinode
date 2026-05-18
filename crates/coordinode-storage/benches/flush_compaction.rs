//! Benchmark: flush and compaction throughput (R072, R073).
//!
//! Measures:
//!   1. flush/write_then_wait — time to write N keys and wait for FlushManager
//!      to drain all sealed memtables to SST (end-to-end flush latency)
//!   2. flush/put_throughput — raw put throughput with background flush enabled
//!      (background workers must not block the write path)
//!   3. compaction/l0_convergence — time for CompactionScheduler to compact
//!      5 L0 SST files down below l0_urgent_threshold after manual flush batches

#![allow(clippy::expect_used, clippy::panic)]

use std::time::Duration;

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lsm_tree::AbstractTree;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Config with aggressive flush so FlushManager triggers immediately.
fn flush_config(dir: &std::path::Path) -> StorageConfig {
    let mut cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    cfg.max_write_buffer_bytes = 1;
    cfg.max_sealed_memtables = 0;
    cfg.flush_poll_interval_ms = 5;
    cfg.compaction_poll_interval_ms = 10;
    cfg.compaction_l0_urgent_threshold = 2;
    cfg
}

/// Poll condition `f` every 5ms for up to `timeout`. Panics on timeout.
fn wait_for(timeout: Duration, label: &str, f: impl Fn() -> bool) {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if f() {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("wait_for timeout: {label}");
}

// ── Flush benchmarks (R072) ───────────────────────────────────────────────────

/// Benchmark: write N keys → wait for FlushManager to drain all sealed memtables.
///
/// Measures end-to-end flush latency: how quickly the background FlushManager
/// converts in-memory writes to on-disk SST files.
/// Target: sealed_count == 0 within 500ms for up to 1000 keys.
fn bench_flush_write_then_wait(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/flush/write_then_wait");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    for &num_keys in &[50_usize, 200, 1000] {
        group.bench_with_input(BenchmarkId::new("keys", num_keys), &num_keys, |b, &n| {
            b.iter(|| {
                let dir = tempfile::TempDir::new().expect("tempdir");
                let config = flush_config(dir.path());
                let engine = StorageEngine::open(&config).expect("open");

                let value = [0xABu8; 128];
                for i in 0..n {
                    let key = format!("bench_flush_{i:06}");
                    engine
                        .put(Partition::Node, key.as_bytes(), &value)
                        .expect("put");
                }

                let tree = engine.tree(Partition::Node).expect("tree");
                wait_for(Duration::from_millis(500), "flush drain", || {
                    tree.sealed_memtable_count() == 0 && tree.active_memtable().size() == 0
                });
            });
        });
    }

    group.finish();
}

/// Benchmark: put throughput with background FlushManager running.
///
/// Measures raw write path throughput while the background flush worker
/// is active. FlushManager must not stall the write path.
/// Baseline: should match or exceed sequential write without background workers.
fn bench_flush_put_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/flush/put_throughput");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    for &batch_size in &[1000_usize, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("keys", batch_size),
            &batch_size,
            |b, &n| {
                b.iter(|| {
                    let dir = tempfile::TempDir::new().expect("tempdir");
                    let config = flush_config(dir.path());
                    let engine = StorageEngine::open(&config).expect("open");

                    let value = [0xCDu8; 128];
                    for i in 0..n {
                        let key = format!("bench_put_{i:08}");
                        engine
                            .put(Partition::Node, key.as_bytes(), &value)
                            .expect("put");
                    }
                });
            },
        );
    }

    group.finish();
}

// ── Compaction benchmarks (R073) ──────────────────────────────────────────────

/// Benchmark: CompactionScheduler L0 convergence time.
///
/// Creates N L0 SST files via manual flush cycles, then measures how quickly
/// the background CompactionScheduler reduces L0 run count below the urgent
/// threshold. This measures the compaction scheduling latency end-to-end.
fn bench_compaction_l0_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/compaction/l0_convergence");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &l0_files in &[5_usize, 10] {
        group.bench_with_input(
            BenchmarkId::new("l0_files", l0_files),
            &l0_files,
            |b, &n| {
                b.iter(|| {
                    let dir = tempfile::TempDir::new().expect("tempdir");
                    let config = flush_config(dir.path());
                    let engine = StorageEngine::open(&config).expect("open");
                    let tree = engine.tree(Partition::Node).expect("tree");

                    // Create N distinct L0 SST files via rotate + flush.
                    let value = [0xEFu8; 64];
                    for batch in 0..n {
                        for i in 0..10_usize {
                            let key = format!("compact_b{batch}_k{i:04}");
                            engine
                                .put(Partition::Node, key.as_bytes(), &value)
                                .expect("put");
                        }
                        tree.rotate_memtable();
                        let lock = tree.get_flush_lock();
                        let _ = tree.flush(&lock, 0);
                    }

                    // Wait for CompactionScheduler to compact below urgent threshold.
                    wait_for(Duration::from_millis(2000), "l0 convergence", || {
                        tree.l0_run_count() < 4
                    });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_flush_write_then_wait,
    bench_flush_put_throughput,
    bench_compaction_l0_convergence,
);
criterion_main!(benches);
