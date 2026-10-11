//! Latency of a label scan at a fixed label population while the rest of
//! the database grows. A scan that reads the label's members stays flat; the
//! same nodes found by a scan of every node (a label test on `labels(n)`) grow with
//! the unrelated data, and are shown beside it as the cost avoided.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_embed::Database;

/// Nodes of the scanned label.
const MEMBERS: usize = 200;
/// Unrelated nodes added before each measurement, cumulative.
const STEPS: [usize; 4] = [0, 10_000, 40_000, 100_000];
/// Bytes of each unrelated node's payload.
const PAYLOAD: usize = 256;

const BY_LABEL: &str = "MATCH (n:Probe) RETURN n.i AS i";
const BY_FILTER: &str = "MATCH (n) WHERE 'Probe' IN labels(n) RETURN n.i AS i";

fn run(db: &Database, query: &str) -> Duration {
    let start = Instant::now();
    let rows = db
        .execute_cypher_shared(query, None, None, None, None)
        .expect("scan")
        .rows;
    assert_eq!(rows.len(), MEMBERS);
    start.elapsed()
}

fn report(label: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    println!(
        "{label}: n={} p50={:?} p99={:?} max={:?}",
        samples.len(),
        samples[samples.len() / 2],
        samples[(samples.len() - 1) * 99 / 100],
        samples[samples.len() - 1],
    );
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher(&format!(
        "UNWIND range(0, {}) AS i CREATE (:Probe {{i: i}})",
        MEMBERS - 1
    ))
    .expect("members");
    let payload = "x".repeat(PAYLOAD);
    let mut unrelated = 0usize;
    for step in STEPS {
        while unrelated < step {
            let batch = (step - unrelated).min(5_000);
            db.execute_cypher(&format!(
                "UNWIND range(1, {batch}) AS i CREATE (:Other {{i: i, payload: '{payload}'}})"
            ))
            .expect("unrelated");
            unrelated += batch;
        }
        run(&db, BY_LABEL); // Bind the plan before measurement.
        report(
            &format!("label scan, {unrelated} unrelated"),
            (0..200).map(|_| run(&db, BY_LABEL)).collect(),
        );
        report(
            &format!("every node, {unrelated} unrelated"),
            (0..20).map(|_| run(&db, BY_FILTER)).collect(),
        );
    }
}
