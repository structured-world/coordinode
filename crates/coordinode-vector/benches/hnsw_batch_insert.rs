//! Criterion benchmarks comparing `HnswIndex::insert_batch` against the
//! sequential `insert` loop.
//!
//! Reference numbers on M-series macOS, 64-dim vectors, M=16, ef=200:
//!
//!   | Workload         | Serial    | Batch     | Speedup |
//!   |------------------|-----------|-----------|---------|
//!   | 512  vectors     |  64.4 ms  |  6.76 ms  |  9.5×   |
//!   | 2048 vectors     | 286.6 ms  |  19.6 ms  |  14.6×  |
//!
//! Progression across the phases:
//!   * **C2 day 3** (parallel planning + serial apply): ~3× on 2K.
//!   * **C3 day 4** (+ parallel apply with lossy back-edges + serial
//!     prune-pass that restores recall): ~6-8× expected.
//!   * **C3 day 5b** (+ dedupe + parallel prune-pass): **14.6× on 2K**.
//!
//! The 14.6× number lands in the C3 arch-doc target range (14-18×).
//! Each iteration starts from a fresh index so the bench measures
//! end-to-end build time, not steady-state insert throughput.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use coordinode_core::graph::types::VectorMetric;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex};

fn make_vectors(n: usize, dim: usize) -> Vec<(u64, Vec<f32>)> {
    (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..dim)
                .map(|d| {
                    let seed = (i as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add((d as u64).wrapping_mul(1442695040888963407));
                    (seed >> 33) as f32 / (1u64 << 31) as f32
                })
                .collect();
            (i as u64, v)
        })
        .collect()
}

fn make_config(max_elements: u32) -> HnswConfig {
    HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 200,
        ef_search: 50,
        metric: VectorMetric::L2,
        max_elements,
        ..Default::default()
    }
}

fn bench_serial_vs_batch(c: &mut Criterion) {
    let dim = 64;
    let mut group = c.benchmark_group("hnsw/insert_throughput");
    // Build benches are expensive — keep sample count modest.
    group.sample_size(10);

    for n in [512usize, 2_048] {
        let items = make_vectors(n, dim);

        group.bench_with_input(BenchmarkId::new("serial", n), &n, |b, _| {
            b.iter_batched(
                || (items.clone(), HnswIndex::new(make_config(n as u32))),
                |(items, mut index)| {
                    for (id, v) in items {
                        index.insert(id, v);
                    }
                    std::hint::black_box(index);
                },
                criterion::BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("batch", n), &n, |b, _| {
            b.iter_batched(
                || (items.clone(), HnswIndex::new(make_config(n as u32))),
                |(items, mut index)| {
                    index.insert_batch(items);
                    std::hint::black_box(index);
                },
                criterion::BatchSize::LargeInput,
            );
        });

        // bulk_build: leader-seed + cluster-ordered insert. Below the
        // internal threshold it routes through insert_batch verbatim;
        // above it the leader-seed pre-build of the upper graph plus
        // the cluster-grouped follower order engage.
        group.bench_with_input(BenchmarkId::new("bulk_build", n), &n, |b, _| {
            b.iter_batched(
                || (items.clone(), HnswIndex::new(make_config(n as u32))),
                |(items, mut index)| {
                    index.bulk_build(items);
                    std::hint::black_box(index);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_serial_vs_batch);
criterion_main!(benches);
