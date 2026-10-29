//! What label kinds cost a write statement.
//!
//! The same insert runs with one undeclared label (the baseline), with a
//! declared type, with a type implying labels, and with auxiliary labels
//! beside it, each without indexes and with a B-tree index on every label it
//! carries: an index on a label covers every node carrying it, so a node
//! carrying more indexed labels writes more entries. Label edits (an
//! auxiliary label added and removed) run with and without an index on it.
//! Each case is the latency distribution of one writer and the heap
//! allocations and bytes per statement.
//!
//! Run: cargo bench -p coordinode-embed --bench label_kinds

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::alloc::System;
use std::time::{Duration, Instant};

use coordinode_embed::Database;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Statements per case.
const OPS: usize = 3_000;

/// Declarations shared by every case: `Goal` implies `Thing`, which implies
/// `Item`, and three auxiliary labels.
const KINDS: &[&str] = &[
    "CREATE NODE TYPE Plain WITH (title: STRING)",
    "CREATE NODE TYPE Item WITH (title: STRING)",
    "CREATE NODE TYPE Thing IMPLIES (Item) WITH (title: STRING)",
    "CREATE NODE TYPE Goal IMPLIES (Thing) WITH (title: STRING)",
    "CREATE LABEL A1 AUXILIARY",
    "CREATE LABEL A2 AUXILIARY",
    "CREATE LABEL A3 AUXILIARY",
];

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn line(label: &str, mut samples: Vec<Duration>, allocations: usize, bytes: usize) {
    samples.sort_unstable();
    let n = samples.len().max(1);
    println!(
        "{label:<34} n={:>5}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}  allocs/stmt={:>6}  bytes/stmt={:>7}",
        samples.len(),
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
        allocations / n,
        bytes / n,
    );
}

fn exec(db: &Database, query: &str) -> Duration {
    let t = Instant::now();
    db.execute_cypher_shared(query, None, None, None, None)
        .expect("statement");
    t.elapsed()
}

fn open(ddl: &[&str]) -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for statement in KINDS.iter().chain(ddl) {
        db.execute_cypher(statement).expect(statement);
    }
    (db, dir)
}

/// Insert `OPS` nodes written with `labels` (as in `:A:B`), after `ddl`.
fn insert(label: &str, labels: &str, ddl: &[&str]) {
    let (db, _dir) = open(ddl);
    // Bind the names and the plan shape first, so every case measures the
    // same steady-state write.
    exec(&db, &format!("CREATE ({labels} {{title: 'warm'}})"));
    let region = Region::new(GLOBAL);
    let samples: Vec<Duration> = (0..OPS)
        .map(|i| exec(&db, &format!("CREATE ({labels} {{title: 't-{i}'}})")))
        .collect();
    let change = region.change();
    line(label, samples, change.allocations, change.bytes_allocated);
}

/// Add and remove the auxiliary label `A1` on `OPS` stored `Goal` nodes,
/// after `ddl`.
fn relabel(label: &str, ddl: &[&str]) {
    let (db, _dir) = open(ddl);
    for i in 0..OPS {
        exec(&db, &format!("CREATE (:Goal {{title: 't-{i}'}})"));
    }
    exec(&db, "MATCH (n:Goal {title: 't-0'}) SET n:A1");
    exec(&db, "MATCH (n:Goal {title: 't-0'}) REMOVE n:A1");
    let region = Region::new(GLOBAL);
    let samples: Vec<Duration> = (0..OPS)
        .flat_map(|i| {
            [
                exec(&db, &format!("MATCH (n:Goal {{title: 't-{i}'}}) SET n:A1")),
                exec(
                    &db,
                    &format!("MATCH (n:Goal {{title: 't-{i}'}}) REMOVE n:A1"),
                ),
            ]
        })
        .collect();
    let change = region.change();
    line(label, samples, change.allocations, change.bytes_allocated);
}

fn main() {
    let index = |label: &str| format!("CREATE INDEX {label}_title ON :{label}(title)");
    let goal_title = index("Goal");

    println!("== Labels on the write path, one writer ==");
    insert("1 undeclared label", ":User", &[]);
    insert("1 declared type", ":Plain", &[]);
    insert("type + 2 implied", ":Goal", &[]);
    insert("type + 2 implied + 1 aux", ":Goal:A1", &[]);
    insert("type + 2 implied + 3 aux", ":Goal:A1:A2:A3", &[]);

    println!("== The same, an index on every label carried ==");
    let user = index("User");
    insert("1 undeclared label, 1 index", ":User", &[user.as_str()]);
    let plain = index("Plain");
    insert("1 declared type, 1 index", ":Plain", &[plain.as_str()]);
    let implied = [goal_title.clone(), index("Thing"), index("Item")];
    let implied: Vec<&str> = implied.iter().map(String::as_str).collect();
    insert("type + 2 implied, 3 indexes", ":Goal", &implied);
    let all = [
        goal_title.clone(),
        index("Thing"),
        index("Item"),
        index("A1"),
        index("A2"),
        index("A3"),
    ];
    let all: Vec<&str> = all.iter().map(String::as_str).collect();
    insert("type + 2 implied + 3 aux, 6 idx", ":Goal:A1:A2:A3", &all);

    println!("== Label edits: SET and REMOVE of an auxiliary label ==");
    relabel("aux label, lookup index only", &[goal_title.as_str()]);
    let a1 = index("A1");
    relabel("aux label, indexed", &[goal_title.as_str(), a1.as_str()]);
}
