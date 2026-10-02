//! Bounded before/after measurement for neighbour-list publication.
//!
//! A task-gating run, sized to finish in well under two minutes on the bench
//! host: one fixed 20k x 128 index built from a deterministic generator, 2000
//! queries timed one by one (QPS, P50, P99) in three repetitions, and recall@10
//! against exact search, so a change to how lists are published is compared at
//! equal recall. A last phase searches while four threads insert another 10k
//! vectors through the shared borrow, and reports the search latency under
//! that load, the insert rate and the recall once it is done. Run with
//! `cargo bench -p coordinode-vector --bench hnsw_publication`.

use std::time::{Duration, Instant};

use coordinode_core::graph::types::VectorMetric;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex, SearchMode};

const N: usize = 20_000;
const DIM: usize = 128;
const QUERIES: usize = 2_000;
const K: usize = 10;
const REPEATS: usize = 3;
/// Vectors inserted during the live phase, and the threads inserting them.
const LIVE_N: usize = 10_000;
const INSERTERS: usize = 4;

/// splitmix64: a full-avalanche mix, so components are independent. A linear
/// generator makes the vectors one lattice with tied distances, and recall
/// measured by id then says nothing about the graph.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic pseudo-random vectors in [0, 1); `salt` separates data from
/// queries.
fn vectors(n: usize, salt: u64) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..DIM)
                .map(|d| {
                    let bits = mix(salt ^ ((i as u64) << 16) ^ d as u64);
                    (bits >> 40) as f32 / (1u64 << 24) as f32
                })
                .collect()
        })
        .collect()
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[rank]
}

fn main() {
    let data = vectors(N, 0);
    let queries = vectors(QUERIES, 0x5EED_0000_0000_0000);

    let mut index = HnswIndex::new(HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 200,
        ef_search: 64,
        metric: VectorMetric::L2,
        ..Default::default()
    });
    // `HNSW_BENCH_BUILD=sequential` builds through one insert per vector
    // instead of the batch path, for comparing the two graphs.
    let sequential = std::env::var("HNSW_BENCH_BUILD").is_ok_and(|v| v == "sequential");
    let build = Instant::now();
    if sequential {
        for (i, v) in data.into_iter().enumerate() {
            index.insert(i as u64, v);
        }
    } else {
        index.insert_batch(
            data.into_iter()
                .enumerate()
                .map(|(i, v)| (i as u64, v))
                .collect(),
        );
    }
    println!(
        "build ({}): {N} x {DIM} in {:.2?}",
        if sequential { "sequential" } else { "batch" },
        build.elapsed()
    );

    println!("recall@{K}: {:.4}", recall(&index, &queries));

    for run in 1..=REPEATS {
        let mut times = Vec::with_capacity(QUERIES);
        let start = Instant::now();
        for q in &queries {
            let t = Instant::now();
            std::hint::black_box(index.search(q, K));
            times.push(t.elapsed());
        }
        let total = start.elapsed();
        times.sort_unstable();
        println!(
            "run {run}: {:.0} qps, p50 {:.1?}, p99 {:.1?}",
            QUERIES as f64 / total.as_secs_f64(),
            percentile(&times, 0.50),
            percentile(&times, 0.99),
        );
    }

    live_phase(&index, &queries);
    control_recall(&queries);
}

/// Recall of a batch-built index over the same 30k vectors the live phase
/// ends with: the reference that separates a recall drop caused by
/// concurrent insertion from the one a larger graph has at the same ef.
fn control_recall(queries: &[Vec<f32>]) {
    let mut index = HnswIndex::new(HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 200,
        ef_search: 64,
        metric: VectorMetric::L2,
        ..Default::default()
    });
    let all = vectors(N, 0)
        .into_iter()
        .chain(vectors(LIVE_N, 0x11FE_0000_0000_0000))
        .enumerate()
        .map(|(i, v)| (i as u64, v))
        .collect();
    index.insert_batch(all);
    println!(
        "recall@{K} of a batch-built control ({} vectors): {:.4}",
        index.len(),
        recall(&index, queries)
    );
}

fn recall(index: &HnswIndex, queries: &[Vec<f32>]) -> f64 {
    let mut hits = 0usize;
    for q in queries {
        let exact = index.search_with_mode(q, K, SearchMode::Exact);
        let approx = index.search(q, K);
        hits += approx
            .iter()
            .filter(|a| exact.iter().any(|e| e.id == a.id))
            .count();
    }
    hits as f64 / (queries.len() * K) as f64
}

/// Search from one thread while `INSERTERS` threads insert `LIVE_N` more
/// vectors into the same index, then measure recall over the grown index.
fn live_phase(index: &HnswIndex, queries: &[Vec<f32>]) {
    let fresh = vectors(LIVE_N, 0x11FE_0000_0000_0000);
    let inserting = std::sync::atomic::AtomicUsize::new(INSERTERS);
    let mut times = Vec::new();
    let mut insert_time = Duration::ZERO;
    std::thread::scope(|s| {
        let per = LIVE_N / INSERTERS;
        let started = Instant::now();
        let inserters: Vec<_> = (0..INSERTERS)
            .map(|t| {
                let (fresh, inserting) = (&fresh, &inserting);
                s.spawn(move || {
                    for (i, v) in fresh.iter().enumerate().skip(t * per).take(per) {
                        index.insert_shared((N + i) as u64, v);
                    }
                    inserting.fetch_sub(1, std::sync::atomic::Ordering::Release);
                })
            })
            .collect();
        let mut q = 0usize;
        while inserting.load(std::sync::atomic::Ordering::Acquire) > 0 {
            let t = Instant::now();
            std::hint::black_box(index.search(&queries[q % queries.len()], K));
            times.push(t.elapsed());
            q += 1;
        }
        for handle in inserters {
            let _ = handle.join();
        }
        insert_time = started.elapsed();
    });
    let searched: Duration = times.iter().sum();
    times.sort_unstable();
    println!(
        "live: {} searches during {LIVE_N} inserts on {INSERTERS} threads: {:.0} qps, p50 {:.1?}, p99 {:.1?}; {:.0} inserts/s",
        times.len(),
        times.len() as f64 / searched.as_secs_f64(),
        percentile(&times, 0.50),
        percentile(&times, 0.99),
        LIVE_N as f64 / insert_time.as_secs_f64(),
    );
    println!(
        "recall@{K} after the live phase ({} vectors): {:.4}",
        index.len(),
        recall(index, queries)
    );
}
