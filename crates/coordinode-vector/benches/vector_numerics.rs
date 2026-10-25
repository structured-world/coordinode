//! Exact vector evaluation qualification: the same workloads measured before
//! and after a change to the metric arithmetic, on pinned hardware.
//!
//! - `kernel/<metric>/d<dim>`: one pair, every metric, dimensions 128 to 65536.
//! - `exact_warm|exact_cold/<metric>/d<dim>`: an exhaustive top-10 scan over a
//!   population of about 16 MiB of retained values; the cold variant evicts the
//!   caches before each scan.
//! - `hnsw/<metric>/d<dim>/ef<ef>`: top-10 search with a small and a large
//!   candidate pool.
//!
//! The scores each workload produces are folded into a checksum printed once,
//! so an equal-cost wrong result shows up as a changed checksum.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::Duration;

use coordinode_core::graph::types::VectorMetric;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex, SearchMode};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

/// Retained values of a vector population, in `[-1, 1)`, deterministic per
/// seed.
fn synth(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u32 << 24) as f32;
            2.0 * u - 1.0
        })
        .collect()
}

const METRICS: [(VectorMetric, &str); 4] = [
    (VectorMetric::Cosine, "cosine"),
    (VectorMetric::L2, "l2"),
    (VectorMetric::DotProduct, "dot"),
    (VectorMetric::L1, "l1"),
];

const DIMS: [usize; 6] = [128, 768, 1024, 1536, 3072, 65_536];

/// Retained floats per exhaustive-scan population: 16 MiB.
const SCAN_FLOATS: usize = 4 << 20;

/// The measured API. Kept in one place so the same workloads run against the
/// arithmetic before and after a change.
mod api {
    use super::*;

    pub fn score(a: &[f32], b: &[f32], metric: VectorMetric) -> f64 {
        f64::from(coordinode_vector::metrics::distance(a, b, metric))
    }

    pub fn fill(index: &mut HnswIndex, items: Vec<(u64, Vec<f32>)>) {
        index.insert_batch(items);
    }

    pub fn exact_top10(index: &HnswIndex, query: &[f32]) -> f64 {
        index
            .search_with_mode(query, 10, SearchMode::Exact)
            .iter()
            .map(|r| f64::from(r.score))
            .sum()
    }

    pub fn hnsw_top10(index: &HnswIndex, query: &[f32]) -> f64 {
        index
            .search(query, 10)
            .iter()
            .map(|r| f64::from(r.score))
            .sum()
    }
}

fn configure() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1))
}

fn bench_kernels(c: &mut Criterion) {
    let mut checksum = 0.0f64;
    for (metric, name) in METRICS {
        for dim in DIMS {
            let a = synth(1, dim);
            let b = synth(2, dim);
            checksum += api::score(&a, &b, metric);
            c.bench_function(&format!("kernel/{name}/d{dim}"), |bch| {
                bch.iter(|| black_box(api::score(black_box(&a), black_box(&b), metric)))
            });
        }
    }
    println!("kernel checksum {checksum:.17e}");
}

/// A buffer larger than the last-level cache, written before a cold scan.
fn evict(buffer: &mut [u64]) {
    for (i, word) in buffer.iter_mut().enumerate() {
        *word = word.wrapping_add(i as u64);
    }
    black_box(buffer);
}

fn bench_exhaustive(c: &mut Criterion) {
    let mut checksum = 0.0f64;
    let mut eviction = vec![0u64; 8 << 20];
    for (metric, name) in [METRICS[0], METRICS[1]] {
        for dim in DIMS {
            let n = (SCAN_FLOATS / dim).max(16) as u64;
            let mut index = HnswIndex::new(HnswConfig {
                metric,
                max_dimensions: 65_536,
                max_elements: n as u32,
                ..Default::default()
            });
            api::fill(
                &mut index,
                (0..n).map(|i| (i, synth(i + 10, dim))).collect(),
            );
            let query = synth(3, dim);
            checksum += api::exact_top10(&index, &query);
            c.bench_function(&format!("exact_warm/{name}/d{dim}"), |bch| {
                bch.iter(|| black_box(api::exact_top10(&index, black_box(&query))))
            });
            c.bench_function(&format!("exact_cold/{name}/d{dim}"), |bch| {
                bch.iter_batched(
                    || evict(&mut eviction),
                    |()| black_box(api::exact_top10(&index, black_box(&query))),
                    BatchSize::PerIteration,
                )
            });
        }
    }
    println!("exact checksum {checksum:.17e}");
}

fn bench_hnsw(c: &mut Criterion) {
    let mut checksum = 0.0f64;
    for (metric, name) in [METRICS[0], METRICS[1]] {
        for (dim, n) in [(128usize, 20_000u64), (768, 5_000)] {
            for ef in [64usize, 512] {
                let mut index = HnswIndex::new(HnswConfig {
                    metric,
                    ef_search: ef,
                    max_dimensions: 65_536,
                    max_elements: n as u32,
                    ..Default::default()
                });
                api::fill(
                    &mut index,
                    (0..n).map(|i| (i, synth(i + 10, dim))).collect(),
                );
                let queries: Vec<Vec<f32>> = (0..64).map(|q| synth(q + 1_000_000, dim)).collect();
                checksum += queries
                    .iter()
                    .map(|q| api::hnsw_top10(&index, q))
                    .sum::<f64>();
                let mut next = 0usize;
                c.bench_function(&format!("hnsw/{name}/d{dim}/ef{ef}"), |bch| {
                    bch.iter(|| {
                        next = (next + 1) % queries.len();
                        black_box(api::hnsw_top10(&index, black_box(&queries[next])))
                    })
                });
            }
        }
    }
    println!("hnsw checksum {checksum:.17e}");
}

criterion_group! {
    name = benches;
    config = configure();
    targets = bench_kernels, bench_exhaustive, bench_hnsw
}
criterion_main!(benches);
