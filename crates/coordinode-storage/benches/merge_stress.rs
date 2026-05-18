//! Benchmark: merge operator throughput under concurrent writes.
//!
//! R010d: measures merge-based posting list writes at various scales
//! (10K, 100K, 1M UIDs) and compares batch vs single-operand encoding.

#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::thread;

use coordinode_core::graph::edge::PostingList;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::merge::{encode_add, encode_add_batch};
use coordinode_storage::engine::partition::Partition;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_merge_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/merge_stress");
    group.sample_size(10); // Large write benchmarks need fewer samples.

    for &(writers, edges_per) in &[(10, 1_000), (50, 2_000), (100, 10_000)] {
        let total = writers * edges_per;
        let label = format!("{writers}w×{edges_per}e={total}");

        group.bench_with_input(
            BenchmarkId::new("batch_merge", &label),
            &(writers, edges_per),
            |b, &(num_writers, edges_per_writer)| {
                b.iter(|| {
                    let dir = tempfile::TempDir::new().expect("tempdir");
                    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
                        "default",
                        dir.path(),
                        Media::Hdd,
                        Durability::Durable,
                        Tier::Warm,
                    )]);
                    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
                    let key = b"adj:FOLLOWS:out:bench";

                    let handles: Vec<_> = (0..num_writers)
                        .map(|wid| {
                            let engine = Arc::clone(&engine);
                            thread::spawn(move || {
                                let base = wid * edges_per_writer;
                                let uids: Vec<u64> =
                                    (base..base + edges_per_writer).map(|x| x as u64).collect();
                                engine
                                    .merge(Partition::Adj, key, &encode_add_batch(&uids))
                                    .expect("merge");
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().expect("join");
                    }

                    engine.persist().expect("persist");

                    // Verify correctness.
                    let data = engine
                        .get(Partition::Adj, key)
                        .expect("get")
                        .expect("exists");
                    let plist = PostingList::from_bytes(&data).expect("decode");
                    assert_eq!(plist.len(), num_writers * edges_per_writer);
                });
            },
        );
    }

    // Baseline comparison: single-operand merge (no batching).
    group.bench_function("single_merge_10w×1000e", |b| {
        b.iter(|| {
            let dir = tempfile::TempDir::new().expect("tempdir");
            let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]);
            let engine = Arc::new(StorageEngine::open(&config).expect("open"));
            let key = b"adj:FOLLOWS:out:bench_single";

            let handles: Vec<_> = (0..10)
                .map(|wid| {
                    let engine = Arc::clone(&engine);
                    thread::spawn(move || {
                        let base = wid * 1000;
                        for i in 0..1000 {
                            engine
                                .merge(Partition::Adj, key, &encode_add((base + i) as u64))
                                .expect("merge");
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().expect("join");
            }

            engine.persist().expect("persist");

            let data = engine
                .get(Partition::Adj, key)
                .expect("get")
                .expect("exists");
            let plist = PostingList::from_bytes(&data).expect("decode");
            assert_eq!(plist.len(), 10_000);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_merge_throughput);
criterion_main!(benches);
