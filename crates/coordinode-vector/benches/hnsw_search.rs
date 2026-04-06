//! Criterion benchmarks for HNSW search performance.
//!
//! Measures search QPS at different index sizes.
//! Baseline for R850 (visited pool) + R851 (prefetch) optimization tracking.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use coordinode_core::graph::types::VectorMetric;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex};

fn make_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| {
                    let seed = (i as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add((d as u64).wrapping_mul(1442695040888963407));
                    (seed >> 33) as f32 / (1u64 << 31) as f32
                })
                .collect()
        })
        .collect()
}

fn build_index(vectors: &[Vec<f32>]) -> HnswIndex {
    let mut index = HnswIndex::new(HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 200,
        ef_search: 50,
        metric: VectorMetric::L2,
        ..Default::default()
    });
    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }
    index
}

fn bench_search(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("hnsw/search");

    for n in [1_000, 10_000] {
        let vectors = make_vectors(n, dim);
        let index = build_index(&vectors);
        let query = &vectors[0];

        group.bench_with_input(BenchmarkId::new("top10", n), &n, |b, _| {
            b.iter(|| {
                let results = index.search(query, 10);
                std::hint::black_box(results);
            });
        });
    }

    group.finish();
}

fn bench_sequential_searches(c: &mut Criterion) {
    let dim = 128;
    let n = 5_000;
    let vectors = make_vectors(n, dim);
    let index = build_index(&vectors);

    c.bench_function("hnsw/sequential_100_searches", |b| {
        b.iter(|| {
            for i in 0..100 {
                let query = &vectors[i % n];
                let results = index.search(query, 10);
                std::hint::black_box(results);
            }
        });
    });
}

criterion_group!(benches, bench_search, bench_sequential_searches);
criterion_main!(benches);
