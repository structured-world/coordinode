//! What a point read through a unique index costs as the indexed node is
//! rewritten: the node's indexed property never changes, another one does,
//! the way a heartbeat updates a record found by its key.
//!
//! Run: cargo bench -p coordinode-embed --bench index_rewrites

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_embed::Database;

/// Reads per measured step.
const READS: usize = 2_000;

/// Rewrites of the node, cumulative, one line per step.
const REWRITES: [usize; 4] = [0, 1_000, 10_000, 50_000];

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE CONSTRAINT agent_sid FOR (a:Agent) REQUIRE a.session_id IS UNIQUE")
        .expect("constraint");
    for i in 0..200 {
        db.execute_cypher(&format!(
            "CREATE (:Agent {{session_id: 'sid-{i}', seen: 0}})"
        ))
        .expect("agent");
    }
    let read = "MATCH (a:Agent {session_id: 'sid-7'}) RETURN a.seen AS seen";
    let plan = db.explain_cypher(read).expect("explain");
    assert!(plan.contains("IndexScan"), "{plan}");

    let mut done = 0;
    for target in REWRITES {
        while done < target {
            db.execute_cypher(&format!(
                "MATCH (a:Agent {{session_id: 'sid-7'}}) SET a.seen = {done}"
            ))
            .expect("rewrite");
            done += 1;
        }
        let mut samples: Vec<Duration> = (0..READS)
            .map(|_| {
                let started = Instant::now();
                let rows = db.execute_cypher(read).expect("read");
                assert_eq!(rows.len(), 1);
                started.elapsed()
            })
            .collect();
        samples.sort_unstable();
        println!(
            "after {target:>6} rewrites: p50={:>9.3?} p99={:>9.3?}",
            percentile(&samples, 0.5),
            percentile(&samples, 0.99)
        );
    }
}
