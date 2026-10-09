//! Separate scan CPU from repeated snapshot waits under an unrelated commit.
//! Run with SNAPSHOT_SCAN_PROFILE=1 to repeat the idle scan for CPU sampling.
//! Every query checks its complete result; the held commit publishes no data.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_embed::Database;
use coordinode_storage::engine::partition::Partition;

const ROWS: usize = 128;
const QUERY: &str = "MATCH (n:Probe) RETURN n.i AS i";

fn scan(db: &Database) -> Duration {
    let start = Instant::now();
    let rows = db
        .execute_cypher_shared(QUERY, None, None, None, None)
        .expect("scan")
        .rows;
    assert_eq!(rows.len(), ROWS);
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
        ROWS - 1
    ))
    .expect("seed");
    scan(&db); // Bind the plan before measurement.
    report("idle scan", (0..100).map(|_| scan(&db)).collect());
    let engine = db.engine_shared();
    let oracle = engine.oracle().expect("oracle");
    let (_, held) = engine
        .pending_commits()
        .admit_allocated(
            || oracle.next().as_raw(),
            vec![(Partition::Node, b"unrelated-pending-key".to_vec())],
            Vec::new(),
        )
        .expect("hold unrelated commit");
    report(
        "held unrelated commit",
        (0..20).map(|_| scan(&db)).collect(),
    );
    // The floor still excludes the held commit. Disabling the freshness wait
    // isolates waiting from the remaining local lock, decoding and row work;
    // this is a measurement case, not a proposed production default.
    let wait_ms = engine.snapshot_wait_ms();
    engine.set_snapshot_wait_ms(0);
    report(
        "held commit, no freshness wait",
        (0..100).map(|_| scan(&db)).collect(),
    );
    engine.set_snapshot_wait_ms(wait_ms);
    drop(held);
    if std::env::var_os("SNAPSHOT_SCAN_PROFILE").is_some() {
        println!("PROFILE idle scan");
        let until = Instant::now() + Duration::from_secs(60);
        while Instant::now() < until {
            std::hint::black_box(scan(&db));
        }
    }
}
