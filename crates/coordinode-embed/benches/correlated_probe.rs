//! A correlated equality join through a unique index: for every row of one
//! label, the row of another whose indexed key the first row names. Three
//! shapes: three left rows per key (the fields of one record pointing at
//! it), one left row per key, and a search that filters the left rows,
//! joins their records and counts and pages the distinct records. The
//! latency distribution (p50, p99, p999) of the whole statement, and the
//! heap allocations and bytes per statement, for a growing number of probes.

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

/// The search: left rows by an indexed layer and a substring, their records
/// by key, record and field conditions after the join, in the form an
/// application's query builder gives it (pass-through projections, guards
/// that resolved to `true`).
const SEARCH: &str = "MATCH (s:R {layer: 'intent'}) WHERE s.vtype = 1 \
     AND toLower(s.vtext) CONTAINS 'bench' \
     MATCH (a:R) WHERE a.key = s.parent \
     WITH a, s WITH a, s WHERE a.parent = 'assets' AND true AND a.adoption = 1 \
     AND ((s.field = 'name' AND true) OR (s.field = 'description' AND true) \
          OR (s.field = 'serial' AND true))";

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

/// Time `RUNS` runs of `query` after one unmeasured run, checking each
/// answer with `check`.
fn measure(
    db: &Database,
    label: &str,
    query: &str,
    check: &dyn Fn(&[coordinode_query::executor::Row]),
) {
    let run = |db: &Database| {
        let start = Instant::now();
        let rows = db
            .execute_cypher_shared(query, None, None, None, None)
            .expect("statement")
            .rows;
        let took = start.elapsed();
        check(&rows);
        took
    };
    run(db); // Bind the plan before measurement.
    let region = Region::new(GLOBAL);
    let samples = (0..RUNS).map(|_| run(db)).collect();
    let change = region.change();
    report(label, samples);
    println!(
        "{label}: allocations/statement={} bytes allocated/statement={}",
        change.allocations / RUNS,
        change.bytes_allocated / RUNS,
    );
}

/// The plain join with `fields` left rows per key, `size` probes in all.
fn join(size: usize, fields: &[&str]) {
    // Beside the build: hosts keep a small root disk.
    let dir = tempfile::tempdir_in(".").expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE UNIQUE INDEX a_key ON :A(key)")
        .expect("index");
    let records = size / fields.len();
    let left: Vec<String> = fields
        .iter()
        .map(|field| format!("(:S {{parent: 'a/' + toString(i), field: '{field}'}})"))
        .collect();
    for start in (0..records).step_by(5_000) {
        let end = (start + 5_000).min(records) - 1;
        db.execute_cypher(&format!(
            "UNWIND range({start}, {end}) AS i \
             CREATE (:A {{key: 'a/' + toString(i), name: 'n' + toString(i)}}), {}",
            left.join(", ")
        ))
        .expect("load");
    }
    let size = records * fields.len();
    measure(
        &db,
        &format!("{size} probes, {} per key", fields.len()),
        JOIN,
        &|rows| assert_eq!(rows[0].get("n"), Some(&Value::Int(size as i64))),
    );
}

/// The search over `size` records, each with one matching name field and
/// two other fields.
fn search(size: usize) {
    let dir = tempfile::tempdir_in(".").expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for ddl in [
        "CREATE UNIQUE INDEX r_key ON :R(key)",
        "CREATE INDEX r_parent ON :R(parent)",
        "CREATE INDEX r_layer ON :R(layer)",
    ] {
        db.execute_cypher(ddl).expect("index");
    }
    for start in (0..size).step_by(5_000) {
        let end = (start + 5_000).min(size) - 1;
        db.execute_cypher(&format!(
            "UNWIND range({start}, {end}) AS i \
             CREATE (:R {{key: 'asset/' + toString(i), parent: 'assets', adoption: 1, \
                          name_sort: 'n' + toString(100000 - i)}}), \
                    (:R {{key: 'asset/' + toString(i) + '/name', parent: 'asset/' + toString(i), \
                          layer: 'intent', field: 'name', vtype: 1, \
                          vtext: 'Bench-Edge-' + toString(i)}}), \
                    (:R {{key: 'asset/' + toString(i) + '/serial', parent: 'asset/' + toString(i), \
                          layer: 'intent', field: 'serial', vtype: 1, vtext: 'S' + toString(i)}}), \
                    (:R {{key: 'asset/' + toString(i) + '/rack', parent: 'asset/' + toString(i), \
                          layer: 'intent', field: 'rack', vtype: 2, vint: i}})"
        ))
        .expect("load");
    }
    measure(
        &db,
        &format!("search count over {size} records"),
        &format!("{SEARCH} RETURN count(DISTINCT a.key) AS total"),
        &|rows| assert_eq!(rows[0].get("total"), Some(&Value::Int(size as i64))),
    );
    measure(
        &db,
        &format!("search page over {size} records"),
        &format!(
            "{SEARCH} RETURN DISTINCT a.key AS key, \
             CASE WHEN true THEN a.name_sort ELSE a.asset END AS ordering \
             ORDER BY ordering, key SKIP 0 LIMIT 25"
        ),
        &|rows| assert_eq!(rows.len(), 25.min(size)),
    );
}

fn main() {
    // `search` alone: `CORRELATED_PROBE=search`.
    if std::env::var("CORRELATED_PROBE").as_deref() != Ok("search") {
        for size in SIZES {
            join(size, &["name", "description", "serial"]);
            join(size, &["name"]);
        }
    }
    search(10_000);
}
