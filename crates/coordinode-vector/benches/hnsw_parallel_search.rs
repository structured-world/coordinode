//! Criterion + custom-driver benchmarks for **parallel** HNSW search QPS.
//!
//! Headline metric for the C1 lock-free read path (R858a):
//! `QPS(N threads) / QPS(1 thread)` should approach `N` on the SIFT1M-class
//! workload as long as no shared-state mutex stands in the way. The search
//! hot path itself is wait-free since C1 day 3b; the remaining
//! single-shared-state surface is [`hnsw::visited::VisitedPool`], which
//! takes a `Mutex<Vec<VisitedList>>` lock per search to recycle scratch
//! buffers. This bench measures both: pure search QPS (criterion) AND a
//! per-thread-count scaling sweep (custom driver) so any regression in
//! scalability surfaces in CI.
//!
//! Workload defaults: 10 000 vectors × 128 dims, M=16, ef=50, top-10. The
//! workload size is intentionally small enough to run on CI hardware in
//! seconds while still saturating memory bandwidth across cores.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use coordinode_core::graph::types::VectorMetric;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex};

/// Deterministic-seeded f32 vectors so the same workload reproduces across
/// CI runs and developer laptops.
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
        max_elements: vectors.len() as u32,
        ..Default::default()
    });
    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }
    index
}

/// Criterion bench: per-thread-count throughput. Reports wall-clock time
/// for N parallel searches per iteration, so QPS(N) = N / time(N).
fn bench_parallel_search(c: &mut Criterion) {
    let n_vectors = 10_000;
    let dim = 128;
    let queries = 64; // power-of-two — divides cleanly across thread counts.

    let vectors = make_vectors(n_vectors, dim);
    let index = Arc::new(build_index(&vectors));
    let query_set: Arc<Vec<Vec<f32>>> = Arc::new(vectors.into_iter().take(queries).collect());

    let mut group = c.benchmark_group("hnsw/parallel_search");
    // Each iteration runs `queries` searches across `threads` workers, so
    // throughput = queries / wall_time. Higher thread counts must keep
    // per-search latency from collapsing on the shared visited pool lock.
    for threads in [1usize, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("queries_per_iter_64_top10", threads),
            &threads,
            |b, &threads| {
                b.iter(|| {
                    let next_query = Arc::new(AtomicUsize::new(0));
                    thread::scope(|scope| {
                        for _ in 0..threads {
                            let index = index.clone();
                            let query_set = query_set.clone();
                            let next_query = next_query.clone();
                            scope.spawn(move || loop {
                                let idx = next_query.fetch_add(1, Ordering::Relaxed);
                                if idx >= queries {
                                    break;
                                }
                                let results = index.search(&query_set[idx], 10);
                                std::hint::black_box(results);
                            });
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

/// `cargo bench -- scaling_report --quick-mode` style driver: when invoked
/// via `cargo bench --bench hnsw_parallel_search -- --nocapture
/// scaling_report`, this prints a human-readable scaling table to stdout
/// suitable for pasting into release notes / docs.
///
/// Criterion's machine-readable format is great for regression detection
/// but lousy for narrative; this complementary output answers the question
/// "is C1 actually scaling?" without parsing JSON.
fn bench_scaling_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw/scaling_report");
    group.sample_size(10); // Single-shot summary; not for regression detection.

    let n_vectors = 10_000;
    let dim = 128;
    let queries_per_thread = 1_000;

    let vectors = make_vectors(n_vectors, dim);
    let index = Arc::new(build_index(&vectors));
    let query_set: Arc<Vec<Vec<f32>>> = Arc::new(vectors.into_iter().take(64).collect());

    let mut baseline_qps: Option<f64> = None;

    for threads in [1usize, 2, 4, 8] {
        // Time a fixed total work budget on this thread count.
        let total_queries = queries_per_thread * threads;
        group.bench_function(BenchmarkId::new("threads", threads), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let next_query = Arc::new(AtomicUsize::new(0));
                    thread::scope(|scope| {
                        for _ in 0..threads {
                            let index = index.clone();
                            let query_set = query_set.clone();
                            let next_query = next_query.clone();
                            scope.spawn(move || loop {
                                let idx = next_query.fetch_add(1, Ordering::Relaxed);
                                if idx >= total_queries {
                                    break;
                                }
                                let q = &query_set[idx % query_set.len()];
                                let results = index.search(q, 10);
                                std::hint::black_box(results);
                            });
                        }
                    });
                }
                let elapsed = start.elapsed();
                let qps = (iters as f64 * total_queries as f64) / elapsed.as_secs_f64();
                if threads == 1 {
                    baseline_qps = Some(qps);
                }
                if let Some(baseline) = baseline_qps {
                    let scale = qps / baseline;
                    let ideal = threads as f64;
                    let efficiency = scale / ideal * 100.0;
                    eprintln!(
                        "  threads={threads:>2}  qps={qps:>10.0}  scale={scale:>5.2}x  \
                         (ideal {ideal:>4.1}x, efficiency {efficiency:>5.1}%)",
                    );
                }
                elapsed
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_parallel_search, bench_scaling_report);
criterion_main!(benches);
