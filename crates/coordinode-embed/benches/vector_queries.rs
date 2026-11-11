//! Vector statements end to end, measured the same way before and after a
//! change to the vector arithmetic: the latency distribution (p50, p99,
//! p999) of an index top-k with its exact rerank (with the distance returned
//! and with only the ordering reading it), an exhaustive top-k, a
//! distance threshold, a similarity projection over every row, and the index
//! top-k while another thread writes vectors.
//!
//! The statements use only the public Cypher surface, so the same file runs
//! against the code before the change. Each runs the whole sequence of query
//! vectors once untimed first and prints the rows of the first query (a
//! changed count shows a wrong answer, an equal one proves nothing; the
//! vector_topk bench compares the answers themselves), the heap allocations
//! and bytes allocated per statement, the parameter map it is handed
//! included, and the most bytes the allocations born on the statement's
//! thread held at once (p50, p99 and max over the runs, whichever thread
//! frees them). Its latencies carry the counting allocator's cost: compare
//! them only with this bench's own. Under concurrent writes the allocation
//! counts include the writer's; its allocations are not in the peak.

#![allow(clippy::expect_used, clippy::print_stdout)]

#[path = "support/alloc_peak.rs"]
mod alloc_peak;
#[path = "support/vectors.rs"]
mod vectors;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use vectors::{params, synth};

#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

/// The populations measured: dimension and vectors per label.
const SHAPES: [(usize, usize); 2] = [(128, 20_000), (768, 4_000)];
/// Timed runs per statement: enough for a p999.
const HOT: usize = 1_000;
/// Timed runs of the statements that read every row.
const SCANS: usize = 100;

/// Top-10 of the indexed label.
const TOP_K_V: &str =
    "MATCH (n:V) RETURN n.i AS i, vector_distance(n.e, $q) AS d ORDER BY d LIMIT 10";

/// Rows of `query`, and how long it took.
fn run(db: &Database, query: &str, p: &HashMap<String, Value>) -> (usize, Duration) {
    let start = Instant::now();
    let rows = db
        .execute_cypher_shared(query, Some(p.clone()), None, None, None)
        .expect("statement")
        .rows
        .len();
    (rows, start.elapsed())
}

fn report(label: &str, rows: usize, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let at = |q: usize| samples[(samples.len() - 1) * q / 1000];
    println!(
        "{label}: rows of the first query={rows} n={} p50={:?} p99={:?} p999={:?} max={:?}",
        samples.len(),
        at(500),
        at(990),
        at(999),
        samples[samples.len() - 1],
    );
}

/// Time `runs` executions of `query`, each with the next query vector, after
/// one untimed pass over all of them.
fn measure(db: &Database, label: &str, query: &str, queries: &[Vec<f32>], runs: usize) {
    let p = |i: usize| {
        params(vec![(
            "q",
            Value::Vector(queries[i % queries.len()].clone()),
        )])
    };
    let (rows, _) = run(db, query, &p(0));
    for i in 1..queries.len() {
        run(db, query, &p(i));
    }
    let allocations = alloc_peak::ALLOCATIONS.load(Ordering::Relaxed);
    let allocated = alloc_peak::ALLOCATED.load(Ordering::Relaxed);
    let mut peaks = Vec::with_capacity(runs);
    let samples = (0..runs)
        .map(|i| {
            let params = p(i);
            let (elapsed, peak) = alloc_peak::peak_of(|| run(db, query, &params).1);
            peaks.push(peak);
            elapsed
        })
        .collect();
    let allocations = alloc_peak::ALLOCATIONS.load(Ordering::Relaxed) - allocations;
    let allocated = alloc_peak::ALLOCATED.load(Ordering::Relaxed) - allocated;
    report(label, rows, samples);
    peaks.sort_unstable();
    let at = |q: usize| peaks[(peaks.len() - 1) * q / 1000];
    println!(
        "{label}: allocations/statement={} bytes allocated/statement={} peak held p50={} p99={} max={}",
        allocations / runs,
        allocated / runs,
        at(500),
        at(990),
        peaks[peaks.len() - 1],
    );
}

fn main() {
    alloc_peak::install();
    for (dim, n) in SHAPES {
        // Beside the build, not in the system temporary directory: hosts keep
        // a small root disk and the store grows under the concurrent writes.
        let dir = tempfile::tempdir_in(".").expect("tempdir");
        // The index exists before the vectors arrive: they enter it on write,
        // with no background build to wait for.
        let db = vectors::open_loaded(dir.path(), dim, n);
        let queries: Vec<Vec<f32>> = (0..64).map(|q| synth(1_000_000 + q, dim)).collect();
        let shape = format!("d{dim} n{n}");

        // The distance projected under an alias: the shape the planner serves
        // from the index.
        measure(
            &db,
            &format!("{shape} index top-10"),
            TOP_K_V,
            &queries,
            HOT,
        );
        // The ordering read before a projection that drops the vector: the
        // same top-10, served from the index too.
        measure(
            &db,
            &format!("{shape} index top-10, vector not returned"),
            "MATCH (n:V) RETURN n.i AS i ORDER BY vector_distance(n.e, $q) LIMIT 10",
            &queries,
            HOT,
        );
        measure(
            &db,
            &format!("{shape} exhaustive top-10"),
            "MATCH (n:W) RETURN n.i AS i, vector_distance(n.e, $q) AS d ORDER BY d LIMIT 10",
            &queries,
            SCANS,
        );
        // About the closest tenth of a uniform population in this cube.
        let threshold = (dim as f64 * 2.0 / 3.0).sqrt() * 0.85;
        measure(
            &db,
            &format!("{shape} threshold"),
            &format!("MATCH (n:W) WHERE vector_distance(n.e, $q) < {threshold} RETURN n.i AS i"),
            &queries,
            SCANS,
        );
        measure(
            &db,
            &format!("{shape} similarity of every row"),
            "MATCH (n:W) RETURN sum(vector_similarity(n.e, $q)) AS s",
            &queries,
            SCANS,
        );

        // The index top-k while another thread keeps writing vectors to it.
        let stop = AtomicBool::new(false);
        let written = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut i = n as u64;
                while !stop.load(Ordering::Relaxed) {
                    db.execute_cypher_shared(
                        "CREATE (:V {i: $i, e: $e})",
                        Some(params(vec![
                            ("i", Value::Int(i as i64)),
                            ("e", Value::Vector(synth(10 + i, dim))),
                        ])),
                        None,
                        None,
                        None,
                    )
                    .expect("concurrent write");
                    i += 1;
                    written.fetch_add(1, Ordering::Relaxed);
                }
            });
            let start = Instant::now();
            measure(
                &db,
                &format!("{shape} index top-10 under writes"),
                TOP_K_V,
                &queries,
                HOT,
            );
            stop.store(true, Ordering::Relaxed);
            let secs = start.elapsed().as_secs_f64();
            // A count of writes in seconds, far inside f64's exact range.
            println!(
                "{shape} concurrent writes: {} in {secs:.2}s ({:.0}/s)",
                written.load(Ordering::Relaxed),
                written.load(Ordering::Relaxed) as f64 / secs
            );
        });
    }
}
