//! Vector statements end to end, measured the same way before and after a
//! change to the vector arithmetic: the latency distribution (p50, p99,
//! p999) of an index top-k with its exact rerank, an exhaustive top-k, a
//! distance threshold, a similarity projection over every row, and the index
//! top-k while another thread writes vectors.
//!
//! The statements use only the public Cypher surface, so the same file runs
//! against the code before the change. Each prints its row count beside its
//! latencies: an equal-cost wrong answer shows up as a changed count.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// The populations measured: dimension and vectors per label.
const SHAPES: [(usize, usize); 2] = [(128, 20_000), (768, 4_000)];
/// Timed runs per statement: enough for a p999.
const HOT: usize = 1_000;
/// Timed runs of the statements that read every row.
const SCANS: usize = 100;
/// Vectors created per statement while loading.
const BATCH: usize = 500;

/// Top-10 of the indexed label.
const TOP_K_V: &str =
    "MATCH (n:V) RETURN n.i AS i, vector_distance(n.e, $q) AS d ORDER BY d LIMIT 10";

/// Components in `[-1, 1)`, deterministic per seed.
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

fn params(entries: Vec<(&str, Value)>) -> HashMap<String, Value> {
    entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

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
        "{label}: rows={rows} n={} p50={:?} p99={:?} p999={:?} max={:?}",
        samples.len(),
        at(500),
        at(990),
        at(999),
        samples[samples.len() - 1],
    );
}

/// Time `runs` executions of `query`, each with the next query vector.
fn measure(db: &Database, label: &str, query: &str, queries: &[Vec<f32>], runs: usize) {
    let p = |i: usize| {
        params(vec![(
            "q",
            Value::Vector(queries[i % queries.len()].clone()),
        )])
    };
    let (rows, _) = run(db, query, &p(0)); // Bind the plan before measurement.
    let samples = (0..runs).map(|i| run(db, query, &p(i)).1).collect();
    report(label, rows, samples);
}

fn load(db: &Database, label: &str, dim: usize, n: usize, seed: u64) {
    for start in (0..n).step_by(BATCH) {
        let rows: Vec<Value> = (start..(start + BATCH).min(n))
            .map(|i| {
                Value::Map(
                    [
                        ("i".to_string(), Value::Int(i as i64)),
                        ("e".to_string(), Value::Vector(synth(seed + i as u64, dim))),
                    ]
                    .into(),
                )
            })
            .collect();
        db.execute_cypher_shared(
            &format!("UNWIND $rows AS r CREATE (:{label} {{i: r.i, e: r.e}})"),
            Some(params(vec![("rows", Value::Array(rows))])),
            None,
            None,
            None,
        )
        .expect("load");
    }
}

fn main() {
    for (dim, n) in SHAPES {
        // Beside the build, not in the system temporary directory: hosts keep
        // a small root disk and the store grows under the concurrent writes.
        let dir = tempfile::tempdir_in(".").expect("tempdir");
        let db = Database::open(dir.path()).expect("open");
        // The index exists before the vectors arrive: they enter it on write,
        // with no background build to wait for.
        db.execute_cypher_shared(
            &format!(
                "CREATE VECTOR INDEX v_e ON :V(e) OPTIONS {{metric: \"euclidean\", dimensions: {dim}}}"
            ),
            None,
            None,
            None,
            None,
        )
        .expect("index");
        load(&db, "V", dim, n, 10);
        load(&db, "W", dim, n, 10);
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
