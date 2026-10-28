//! Benchmark: flush and compaction latency of a `STORAGE COLUMNAR` table.
//!
//! Measures:
//!   1. table/statement_flush: a statement writes rows into a table and ends
//!      with the table's flush (an engine without a journal flushes at the
//!      statement's end); each sample is a fixed series of statements on a
//!      fresh store, so the series claims table files from the folder's pool
//!      and their refills fall inside the measured flushes
//!   2. table/l0_convergence: time for the compaction scheduler to bring a
//!      table's L0 down to the urgent threshold after N statement flushes

#![allow(clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lsm_tree::AbstractTree;

fn config(dir: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn open(dir: &std::path::Path) -> (StorageEngine, Arc<TimestampOracle>) {
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config(dir), Arc::clone(&oracle)).expect("open");
    (engine, oracle)
}

/// Write `rows` rows into `table` and end the statement.
fn statement(engine: &StorageEngine, oracle: &TimestampOracle, table: &str, round: u64, rows: u32) {
    let value = [0xABu8; 128];
    for i in 0..rows {
        let key = format!("r{round:08}:{i:04}");
        engine
            .columnar_insert(
                table,
                key.into_bytes(),
                value.to_vec(),
                oracle.next().as_raw(),
            )
            .expect("insert");
    }
    engine
        .finish_columnar_statement(table)
        .expect("statement flush");
}

/// Statements in one measured series. Long enough that the folder's pool runs
/// through its refills inside the series, so the series is not one cold flush.
const SERIES: u64 = 64;

fn bench_statement_flush(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/table/statement_flush");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));
    for &rows in &[16_u32, 256] {
        group.bench_with_input(BenchmarkId::new("rows", rows), &rows, |b, &rows| {
            // Every series starts from a fresh store and writes the same keys,
            // so the state each flush sees does not depend on how many
            // iterations criterion chose for this or an earlier group. Opening
            // and dropping the store stay outside the timed span; the reported
            // time is the mean statement of the series.
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let dir = tempfile::TempDir::new().expect("tempdir");
                    let (engine, oracle) = open(dir.path());
                    let started = Instant::now();
                    for round in 0..SERIES {
                        statement(&engine, &oracle, "t", round, rows);
                    }
                    total += started.elapsed() / SERIES as u32;
                    drop(engine);
                    drop(dir);
                }
                total
            });
        });
    }
    group.finish();
}

fn bench_l0_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/table/l0_convergence");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    for &runs in &[8_u64, 16] {
        group.bench_with_input(BenchmarkId::new("runs", runs), &runs, |b, &runs| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let dir = tempfile::TempDir::new().expect("tempdir");
                    let (engine, oracle) = open(dir.path());
                    let urgent = config(dir.path()).compaction_l0_urgent_threshold;
                    let started = Instant::now();
                    for round in 0..runs {
                        statement(&engine, &oracle, "t", round, 16);
                    }
                    let tree = engine.columnar_table_tree("t").expect("table");
                    let deadline = Instant::now() + Duration::from_secs(10);
                    while tree.l0_run_count() > urgent {
                        assert!(Instant::now() < deadline, "the table's L0 never converged");
                        std::thread::sleep(Duration::from_micros(200));
                    }
                    total += started.elapsed();
                }
                total
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_statement_flush, bench_l0_convergence);
criterion_main!(benches);
