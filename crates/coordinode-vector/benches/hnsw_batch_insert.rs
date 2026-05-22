//! Criterion benchmarks comparing `HnswIndex::insert_batch` against the
//! sequential `insert` loop.
//!
//! Headline number for the C2 phase (R858b): batch ingestion runs the
//! plan phase in parallel across rayon workers while applying mutations
//! serially. Expected throughput uplift is **5-8×** on multi-core hosts
//! since planning dominates ~80% of insert cost.
//!
//! Workload defaults: 1 000 vectors × 64 dims, M=16, ef_construction=200.
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
    }

    group.finish();
}

criterion_group!(benches, bench_serial_vs_batch);
criterion_main!(benches);
