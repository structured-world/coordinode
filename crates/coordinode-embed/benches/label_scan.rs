//! Latency of a label scan at a fixed label population while the rest of
//! the database grows. A scan that reads the label's members stays flat; the
//! same nodes found by a scan of every node (a label test on `labels(n)`)
//! grow with the unrelated data, and are shown beside it as the cost
//! avoided. Then, at the largest size: the cold scan (the first after the
//! database is opened again, with nothing cached), and what keeping the
//! label lookup costs a write, in latency and in index entries per write.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_embed::Database;
use coordinode_storage::engine::partition::Partition;

/// Nodes of the scanned label.
const MEMBERS: usize = 200;
/// Unrelated nodes added before each measurement, cumulative.
const STEPS: [usize; 4] = [0, 10_000, 40_000, 100_000];
/// Bytes of each unrelated node's payload.
const PAYLOAD: usize = 256;
/// Hot scans timed per step: enough for a P999.
const HOT: usize = 1_000;
/// Times the database is opened again for a cold scan.
const COLD: usize = 20;
/// Writes timed per kind.
const WRITES: usize = 1_000;

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
    let at = |q: usize| samples[(samples.len() - 1) * q / 1000];
    println!(
        "{label}: n={} p50={:?} p99={:?} p999={:?} max={:?}",
        samples.len(),
        at(500),
        at(990),
        at(999),
        samples[samples.len() - 1],
    );
}

/// Time `count` statements built by `statement`, and the index entries they
/// added per statement.
fn writes(db: &mut Database, label: &str, count: usize, statement: impl Fn(usize) -> String) {
    let before = db.engine().approximate_len(Partition::Idx).expect("idx");
    let samples = (0..count)
        .map(|i| {
            let start = Instant::now();
            db.execute_cypher(&statement(i)).expect("write");
            start.elapsed()
        })
        .collect();
    let after = db.engine().approximate_len(Partition::Idx).expect("idx");
    report(label, samples);
    // Entry counts of a store this size, far below f64's exact range.
    println!(
        "{label}: index entries per write ~{:.2}",
        (after as f64 - before as f64) / count as f64
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
            (0..HOT).map(|_| run(&db, BY_LABEL)).collect(),
        );
        report(
            &format!("every node, {unrelated} unrelated"),
            (0..20).map(|_| run(&db, BY_FILTER)).collect(),
        );
    }

    // Cold: the first scan after the database is opened again.
    db.engine().persist().expect("persist");
    drop(db);
    let cold = (0..COLD)
        .map(|_| {
            let db = Database::open(dir.path()).expect("reopen");
            let took = run(&db, BY_LABEL);
            drop(db);
            took
        })
        .collect();
    report(&format!("cold label scan, {unrelated} unrelated"), cold);

    // What the lookup costs a write: a create stages one entry per label, a
    // label change one, a property edit none.
    let mut db = Database::open(dir.path()).expect("reopen");
    writes(&mut db, "create with one label", WRITES, |i| {
        format!("CREATE (:Fresh {{i: {i}}})")
    });
    writes(&mut db, "property edit, labels kept", WRITES, |i| {
        format!("MATCH (n:Fresh {{i: {i}}}) SET n.v = {i}")
    });
    writes(&mut db, "label added", WRITES, |i| {
        format!("MATCH (n:Fresh {{i: {i}}}) SET n:Tagged")
    });
}
