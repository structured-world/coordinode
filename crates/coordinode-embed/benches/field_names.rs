//! What property names cost a write statement.
//!
//! A statement whose property names are all bound pays only for looking them
//! up; one that introduces a name also pays for binding it before its data
//! can be encoded. The two are reported apart, each as a distribution, and
//! each on one thread and on eight: a dictionary guarded by one lock for the
//! whole statement serializes writers even when every name is known, which a
//! single-threaded run never shows.
//!
//! Run: cargo bench -p coordinode-embed --bench field_names

#![allow(clippy::expect_used, clippy::print_stdout, clippy::panic)]

use std::time::{Duration, Instant};

use coordinode_embed::Database;

/// Statements per case.
const OPS: usize = 4_000;

/// Writers in the concurrent cases.
const WRITERS: usize = 8;

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn line(label: &str, wall: Duration, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    println!(
        "{label:<30} n={:>6}  {:>8.0} stmt/s  mean={:>9.3?}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}",
        samples.len(),
        samples.len() as f64 / wall.as_secs_f64(),
        total / samples.len() as u32,
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
    );
}

fn exec(db: &Database, query: &str) -> Duration {
    let t = Instant::now();
    db.execute_cypher_shared(query, None, None, None, None)
        .expect("statement");
    t.elapsed()
}

/// Run `query(writer, i)` `OPS` times, split across `writers` threads.
fn run(label: &str, writers: usize, query: impl Fn(usize, usize) -> String + Sync) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    // Bind the names the known-name cases use, so they measure lookups only.
    exec(
        &db,
        "CREATE (:Warm {alpha: 0, beta: 0, gamma: 0, delta: 0})",
    );
    let per_writer = OPS / writers;
    let started = Instant::now();
    let samples: Vec<Duration> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..writers)
            .map(|w| {
                let (db, query) = (&db, &query);
                s.spawn(move || {
                    (0..per_writer)
                        .map(|i| exec(db, &query(w, i)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("writer"))
            .collect()
    });
    line(label, started.elapsed(), samples);
}

fn known(w: usize, i: usize) -> String {
    format!("CREATE (:Hot {{alpha: {w}, beta: {i}, gamma: 1, delta: 2}})")
}

fn new_name(w: usize, i: usize) -> String {
    format!("CREATE (:Cold {{alpha: 1, n_{w}_{i}: 1}})")
}

fn main() {
    println!("== property names on the write path ==");
    run("known names, 1 writer", 1, known);
    run("known names, 8 writers", WRITERS, known);
    run("one new name, 1 writer", 1, new_name);
    run("one new name, 8 writers", WRITERS, new_name);
}
