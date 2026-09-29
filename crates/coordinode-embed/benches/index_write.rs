//! What a B-tree index costs a write statement.
//!
//! The same insert runs against a label with no index, with a plain index on
//! the written property, and with a unique one, each maintained RESOLVED and
//! DERIVED. A RESOLVED index adds its entry to the statement's log entry; a
//! DERIVED one adds the work the members derive it from instead. The unique
//! index also reads the value's entry first, to refuse a second holder. Each
//! case is a distribution, on
//! one thread and on eight, followed by the size of the log entries the
//! inserts wrote: what a replicated write carries to every member.
//!
//! Run: cargo bench -p coordinode-embed --bench index_write

#![allow(clippy::expect_used, clippy::print_stdout)]

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

/// Insert `OPS` users with distinct emails over `writers` threads, after
/// running `ddl` (if any).
fn run(label: &str, writers: usize, ddl: Option<&str>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    // Bind the names first, so every case measures the same write.
    db.execute_cypher("CREATE (:Warm {email: 'warm'})")
        .expect("warm");
    if let Some(ddl) = ddl {
        db.execute_cypher(ddl).expect("ddl");
    }
    let before = last_journal_index(&db);
    let db = &db;
    let per_writer = OPS / writers;
    let started = Instant::now();
    let samples: Vec<Duration> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..writers)
            .map(|w| {
                s.spawn(move || {
                    (0..per_writer)
                        .map(|i| {
                            exec(
                                db,
                                &format!("CREATE (:User {{email: 'user-{w}-{i}@example.com'}})"),
                            )
                        })
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
    entry_sizes(db, before);
}

/// The index of the last journal entry, or 0 when there is none.
fn last_journal_index(db: &Database) -> u64 {
    db.engine()
        .oplog_read_since(0)
        .expect("read journal")
        .expect("the database journals its writes")
        .last()
        .map_or(0, |e| e.index)
}

/// Size of the journal entries after `after`, one per insert statement.
fn entry_sizes(db: &Database, after: u64) {
    let entries = db
        .engine()
        .oplog_read_since(after + 1)
        .expect("read journal")
        .expect("the database journals its writes");
    let mut sizes: Vec<usize> = entries
        .iter()
        .map(|e| e.encode().expect("encode entry").len())
        .collect();
    // The operations an entry encodes, a unit frame counted as what it holds.
    let ops: usize = entries
        .iter()
        .map(|e| {
            coordinode_storage::oplog::convert::expand_units(&e.ops)
                .expect("expand entry")
                .len()
        })
        .sum();
    sizes.sort_unstable();
    let total: usize = sizes.iter().sum();
    let n = sizes.len().max(1);
    println!(
        "{:<30} entries={:>6}  ops/entry={:>5.2}  bytes/entry mean={:>6.1}  p50={:>5}  max={:>5}",
        "  log entries",
        sizes.len(),
        ops as f64 / n as f64,
        total as f64 / n as f64,
        sizes.get(sizes.len() / 2).copied().unwrap_or(0),
        sizes.last().copied().unwrap_or(0),
    );
}

fn main() {
    println!("== B-tree index on the write path ==");
    for (writers, suffix) in [(1, "1 writer"), (WRITERS, "8 writers")] {
        run(&format!("no index, {suffix}"), writers, None);
        run(
            &format!("plain index, {suffix}"),
            writers,
            Some("CREATE INDEX user_email ON :User(email)"),
        );
        run(
            &format!("unique index, {suffix}"),
            writers,
            Some("CREATE UNIQUE INDEX user_email ON :User(email)"),
        );
        run(
            &format!("plain derived, {suffix}"),
            writers,
            Some("CREATE INDEX user_email ON :User(email) OPTIONS {maintenance: 'derived'}"),
        );
        run(
            &format!("unique derived, {suffix}"),
            writers,
            Some(
                "CREATE UNIQUE INDEX user_email ON :User(email) \
                 OPTIONS {maintenance: 'derived'}",
            ),
        );
    }
}
