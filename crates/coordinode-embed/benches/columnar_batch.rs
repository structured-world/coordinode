//! Latency of a statement writing a batch of `STORAGE COLUMNAR` rows, and of
//! a read of the table: what an operation on a table's trees costs a
//! statement, measured the same way on any revision.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_embed::Database;

/// Rows each write statement creates.
const BATCH: usize = 100;
/// Write statements timed.
const WRITES: usize = 400;
/// Reads timed.
const READS: usize = 400;

fn report(label: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let at = |q: usize| samples[(samples.len() - 1) * q / 1000];
    println!(
        "{label}: n={} p50={:?} p99={:?} max={:?}",
        samples.len(),
        at(500),
        at(990),
        samples[samples.len() - 1],
    );
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE TABLE Trade (id BIGINT PRIMARY KEY, qty INT) STORAGE COLUMNAR")
        .expect("create");
    let writes = (0..WRITES)
        .map(|w| {
            let from = w * BATCH;
            let statement = format!(
                "UNWIND range({from}, {}) AS i CREATE (:Trade {{id: i, qty: i}})",
                from + BATCH - 1
            );
            let start = Instant::now();
            db.execute_cypher(&statement).expect("write");
            start.elapsed()
        })
        .collect();
    report(&format!("columnar write of {BATCH} rows"), writes);
    let reads = (0..READS)
        .map(|r| {
            let start = Instant::now();
            let rows = db
                .execute_cypher(&format!(
                    "MATCH (t:Trade {{id: {}}}) RETURN t.qty AS qty",
                    r * BATCH
                ))
                .expect("read");
            assert_eq!(rows.len(), 1);
            start.elapsed()
        })
        .collect();
    report("columnar read by key", reads);
}
