//! A correlated equality join through a unique index: for every row of one
//! label, the row of another whose indexed key the first row names, three
//! left rows per key (the fields of one record pointing at it). The latency
//! distribution (p50, p99, p999) of the whole statement, and the heap
//! allocations and bytes per statement, for a growing number of probes.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::alloc::System;
use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Probes per statement measured.
const SIZES: [usize; 3] = [1_000, 10_000, 40_000];
/// Timed statements per size.
const RUNS: usize = 30;

const JOIN: &str = "MATCH (s:S) MATCH (a:A) WHERE a.key = s.parent RETURN count(*) AS n";

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

fn main() {
    for size in SIZES {
        // Beside the build: hosts keep a small root disk.
        let dir = tempfile::tempdir_in(".").expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE UNIQUE INDEX a_key ON :A(key)")
            .expect("index");
        // `size` probes: a third as many records, three fields each.
        let records = size / 3;
        for start in (0..records).step_by(5_000) {
            let end = (start + 5_000).min(records) - 1;
            db.execute_cypher(&format!(
                "UNWIND range({start}, {end}) AS i \
                 CREATE (:A {{key: 'a/' + toString(i), name: 'n' + toString(i)}}), \
                        (:S {{parent: 'a/' + toString(i), field: 'name'}}), \
                        (:S {{parent: 'a/' + toString(i), field: 'description'}}), \
                        (:S {{parent: 'a/' + toString(i), field: 'serial'}})"
            ))
            .expect("load");
        }
        let size = records * 3;
        let run = |db: &Database| {
            let start = Instant::now();
            let rows = db
                .execute_cypher_shared(JOIN, None, None, None, None)
                .expect("join")
                .rows;
            assert_eq!(rows[0].get("n"), Some(&Value::Int(size as i64)));
            start.elapsed()
        };
        run(&db); // Bind the plan before measurement.
        let region = Region::new(GLOBAL);
        let samples = (0..RUNS).map(|_| run(&db)).collect();
        let change = region.change();
        report(&format!("{size} probes"), samples);
        println!(
            "{size} probes: allocations/statement={} bytes allocated/statement={}",
            change.allocations / RUNS,
            change.bytes_allocated / RUNS,
        );
    }
}
