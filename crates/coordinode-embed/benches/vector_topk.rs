//! The index top-10 with its distance returned, over 20,000 vectors of
//! dimension 128, measured the same way before and after a change.
//!
//! Outside the timing it prints, for each of the 64 query vectors, the ids
//! and distances the index answers and the exact answer of the same
//! vectors without an index, and the recall against it: two revisions
//! compare by those lines, not by row counts. Then it times one pass over
//! the 64 queries on a store that has not run them (cold), runs the whole
//! sequence once more untimed, and times `VECTOR_TOPK_SAMPLES` statements
//! (3000 by default) cycling through it (steady), printing every latency in
//! order.
//!
//! Under the system allocator by default. Built with the `bench-alloc`
//! feature it counts instead: per statement, the most bytes the allocations
//! born on its thread held at once, the allocations made by other threads
//! meanwhile, and the allocations per statement; its latencies then carry
//! the counting's cost and are not printed.

#![allow(clippy::expect_used, clippy::panic, clippy::print_stdout)]

#[cfg(feature = "bench-alloc")]
#[path = "support/alloc_peak.rs"]
mod alloc_peak;
#[path = "support/vectors.rs"]
mod vectors;

use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

#[cfg(feature = "bench-alloc")]
#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

const DIM: usize = 128;
const N: usize = 20_000;
const QUERIES: u64 = 64;

const TOP_K: &str =
    "MATCH (n:V) RETURN n.i AS i, vector_distance(n.e, $q) AS d ORDER BY d LIMIT 10";
const EXACT: &str =
    "MATCH (n:W) RETURN n.i AS i, vector_distance(n.e, $q) AS d ORDER BY d LIMIT 10";

/// The `(i, d)` rows of `query` for query vector `q`, in order.
fn answer(db: &Database, query: &str, q: &[f32]) -> Vec<(i64, f64)> {
    db.execute_cypher_shared(
        query,
        Some(vectors::params(vec![("q", Value::Vector(q.to_vec()))])),
        None,
        None,
        None,
    )
    .expect("statement")
    .rows
    .iter()
    .map(|row| match (row.get("i"), row.get("d")) {
        (Some(Value::Int(i)), Some(Value::Float(d))) => (*i, *d),
        other => panic!("an id and a distance: {other:?}"),
    })
    .collect()
}

/// How long `query` took for `q`; the answer is dropped inside the timing.
fn timed(db: &Database, q: &[f32]) -> Duration {
    let params = vectors::params(vec![("q", Value::Vector(q.to_vec()))]);
    let start = Instant::now();
    let rows = db
        .execute_cypher_shared(TOP_K, Some(params), None, None, None)
        .expect("statement")
        .rows;
    drop(rows);
    start.elapsed()
}

#[cfg(not(feature = "bench-alloc"))]
fn report(label: &str, samples: &[Duration]) {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let at = |q: usize| sorted[(sorted.len() - 1) * q / 1000];
    println!(
        "{label}: n={} p50={:?} p99={:?} p999={:?} max={:?}",
        sorted.len(),
        at(500),
        at(990),
        at(999),
        sorted[sorted.len() - 1],
    );
    let raw: Vec<String> = samples.iter().map(|d| d.as_nanos().to_string()).collect();
    println!("{label} raw ns in order: {}", raw.join(","));
}

/// The answers of the index and the exact query for every query vector,
/// and the recall of the index against the exact ids.
fn verify(db: &Database, queries: &[Vec<f32>]) {
    let mut found = 0;
    for (n, q) in queries.iter().enumerate() {
        let index = answer(db, TOP_K, q);
        let exact = answer(db, EXACT, q);
        let hits = index
            .iter()
            .filter(|(i, _)| exact.iter().any(|(e, _)| e == i))
            .count();
        found += hits;
        println!("q{n} index={index:?} exact={exact:?} recall={hits}/10");
    }
    // 640 answers at most: far inside f64's exact range.
    println!(
        "recall over {} queries: {found}/{} = {:.4}",
        queries.len(),
        queries.len() * 10,
        found as f64 / (queries.len() * 10) as f64
    );
}

fn main() {
    #[cfg(feature = "bench-alloc")]
    alloc_peak::install();
    let samples: usize = std::env::var("VECTOR_TOPK_SAMPLES")
        .map_or(3000, |s| s.parse().expect("VECTOR_TOPK_SAMPLES is a count"));
    // Beside the build, not in the system temporary directory: hosts keep a
    // small root disk.
    let dir = tempfile::tempdir_in(".").expect("tempdir");
    let db = vectors::open_loaded(dir.path(), DIM, N);
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| vectors::synth(1_000_000 + q, DIM))
        .collect();

    #[cfg(not(feature = "bench-alloc"))]
    {
        let cold: Vec<Duration> = queries.iter().map(|q| timed(&db, q)).collect();
        verify(&db, &queries);
        for q in &queries {
            timed(&db, q);
        }
        let steady: Vec<Duration> = (0..samples)
            .map(|s| timed(&db, &queries[s % queries.len()]))
            .collect();
        report("d128 n20000 index top-10 cold", &cold);
        report("d128 n20000 index top-10 steady", &steady);
    }

    #[cfg(feature = "bench-alloc")]
    {
        use std::sync::atomic::Ordering;
        verify(&db, &queries);
        for q in &queries {
            timed(&db, q);
        }
        let allocations = alloc_peak::ALLOCATIONS.load(Ordering::Relaxed);
        let allocated = alloc_peak::ALLOCATED.load(Ordering::Relaxed);
        let elsewhere = alloc_peak::ELSEWHERE.load(Ordering::Relaxed);
        let mut peaks: Vec<u64> = (0..samples)
            .map(|s| alloc_peak::peak_of(|| timed(&db, &queries[s % queries.len()])).1)
            .collect();
        let allocations = alloc_peak::ALLOCATIONS.load(Ordering::Relaxed) - allocations;
        let allocated = alloc_peak::ALLOCATED.load(Ordering::Relaxed) - allocated;
        let elsewhere = alloc_peak::ELSEWHERE.load(Ordering::Relaxed) - elsewhere;
        peaks.sort_unstable();
        let at = |q: usize| peaks[(peaks.len() - 1) * q / 1000];
        println!(
            "d128 n20000 index top-10: allocations/statement={} bytes allocated/statement={} \
             peak held by its thread's allocations p50={} p99={} max={} bytes allocated by \
             other threads meanwhile={elsewhere}",
            allocations / samples,
            allocated / samples,
            at(500),
            at(990),
            peaks[peaks.len() - 1],
        );
    }
}
