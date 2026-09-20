//! Benchmark: what an unfolded operand chain costs the reader of a counter.
//!
//! The planner's node-count statistics are a single key written only by
//! merge: one delta per node insert or delete. The engine folds a chain of
//! operands only where it can prove the base, which above the last level it
//! cannot, so between compactions that reach the last level the chain on that
//! key grows with the write volume and every read resolves all of it.
//!
//! Each case writes `n` deltas to the one key and times the point read, once
//! with the chain as the writes left it and once after a compaction has
//! materialised it. The ratio between the two is the cost of the chain, and
//! its growth with `n` is what says whether folding the chain earlier would
//! pay for itself.

#![allow(clippy::expect_used)]

use coordinode_core::graph::stats::NODES_TOTAL_KEY;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::merge::encode_counter_delta;
use coordinode_storage::engine::partition::Partition;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// Deltas written between flushes, so the chain is spread over many tables
/// rather than sitting in one memtable.
const DELTAS_PER_FLUSH: usize = 256;

/// An engine holding `n` counter deltas on the statistics key, flushed every
/// `DELTAS_PER_FLUSH` writes. The directory is returned with it: dropping it
/// removes the data, so it has to outlive the engine.
fn engine_with_chain(n: usize) -> (tempfile::TempDir, StorageEngine) {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    let operand = encode_counter_delta(1);
    for i in 1..=n {
        engine
            .merge(Partition::Counter, NODES_TOTAL_KEY, &operand)
            .expect("merge");
        if i % DELTAS_PER_FLUSH == 0 {
            engine.persist().expect("persist");
        }
    }
    engine.persist().expect("persist");

    (dir, engine)
}

/// The read must answer with the whole chain applied, whatever the chain's
/// physical shape, so a case that answers differently before and after the
/// compaction is measuring something other than the fold.
fn read_total(engine: &StorageEngine) -> i64 {
    let value = engine
        .get(Partition::Counter, NODES_TOTAL_KEY)
        .expect("get")
        .expect("the key was written");
    i64::from_le_bytes(value.as_ref().try_into().expect("eight bytes"))
}

fn bench_counter_chain_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/counter_chain_read");
    group.sample_size(20);

    for &n in &[1_000usize, 10_000, 100_000] {
        let (_dir, engine) = engine_with_chain(n);
        assert_eq!(read_total(&engine), n as i64, "chain reads as its sum");

        group.bench_with_input(BenchmarkId::new("as_written", n), &n, |b, _| {
            b.iter(|| read_total(&engine));
        });

        engine.major_compact(Partition::Counter).expect("compact");
        assert_eq!(read_total(&engine), n as i64, "compaction keeps the sum");

        group.bench_with_input(BenchmarkId::new("after_compaction", n), &n, |b, _| {
            b.iter(|| read_total(&engine));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_counter_chain_read);
criterion_main!(benches);
