//! What an index check costs while lookups go on: their p50 and p99 before
//! and during the check, the check's time, the entries and records it read,
//! what it repaired, the bytes it wrote, and this member's recovery-state
//! flushes. The index holds damage the check repairs entry by entry.
//!
//! Run: cargo bench -p coordinode-embed --bench index_check

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::unwrap_used
)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{encode_tuple, encode_unique_entry_key};
use coordinode_embed::Database;
use coordinode_modality::{IndexStore as _, LocalIndexStore};
use coordinode_query::index::{CheckOutcome, IndexSelector};
use coordinode_storage::engine::partition::Partition;

/// Nodes of the indexed label.
const USERS: usize = 20_000;

/// Entries lost before the check, every `USERS / LOST`-th one.
const LOST: usize = 200;

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

/// Lookups of users by email, round-robin, until `stop` is set or `max`
/// are done; their latencies.
fn lookups(db: &Database, stop: &AtomicBool, max: usize) -> Vec<Duration> {
    let mut samples = Vec::new();
    let mut i = 0usize;
    while samples.len() < max && !stop.load(Ordering::Relaxed) {
        // Users the check leaves alone: a damaged one answers from the
        // records, which is the fallback's cost, not the check's.
        i = i % USERS + 1;
        if i.is_multiple_of(USERS / LOST) {
            continue;
        }
        let query = format!("MATCH (u:U {{email: 'u{i}@x'}}) RETURN u.n AS n");
        let started = Instant::now();
        let rows = db
            .execute_cypher_shared(&query, None, None, None, None)
            .expect("read")
            .rows;
        samples.push(started.elapsed());
        assert_eq!(rows.len(), 1, "{query}");
    }
    samples.sort_unstable();
    samples
}

fn report(name: &str, samples: &[Duration]) {
    println!(
        "{name:<26} lookups={:>6} p50={:>10.3?} p99={:>10.3?}",
        samples.len(),
        percentile(samples, 0.5),
        percentile(samples, 0.99)
    );
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher(&format!(
        "UNWIND range(1, {USERS}) AS i CREATE (:U {{email: 'u' + toString(i) + '@x', n: i}})"
    ))
    .expect("users");

    let quiet = lookups(&db, &AtomicBool::new(false), 5_000);
    report("lookups, no check", &quiet);

    let store = LocalIndexStore::new(db.engine());
    let id = store.resolve_name("u_email").unwrap().unwrap();
    let index = store.load_definition(id).unwrap().unwrap();
    for i in (USERS / LOST..=USERS).step_by(USERS / LOST) {
        let tuple = encode_tuple(&[Value::String(format!("u{i}@x"))]).expect("tuple");
        db.engine()
            .delete(
                Partition::Idx,
                &encode_unique_entry_key(index.generation, &tuple),
            )
            .expect("lose an entry");
    }
    db.engine().persist().expect("persist");

    let disk = |db: &Database| fs_extra_size(db.engine().data_dir());
    let bytes_before = disk(&db);
    let flushes_before = db.index_recovery_flushes();
    let stop = AtomicBool::new(false);
    let (during, check, took) = std::thread::scope(|scope| {
        let reader = scope.spawn(|| lookups(&db, &stop, usize::MAX));
        let started = Instant::now();
        let operation = db
            .check_index(&IndexSelector::Name("u_email".into()))
            .expect("start the check");
        let (status, outcome) = db
            .index_check(operation, Duration::from_secs(600))
            .expect("read the check")
            .expect("the check exists");
        let took = started.elapsed();
        stop.store(true, Ordering::Relaxed);
        (
            reader.join().expect("reader"),
            (status, outcome.expect("the check ended")),
            took,
        )
    });
    report("lookups, during the check", &during);
    let (status, outcome) = check;
    assert!(
        matches!(outcome, CheckOutcome::Verified { .. }),
        "{outcome:?}"
    );
    let record = status.record.check.expect("check record");
    db.engine().persist().expect("persist");
    println!(
        "check                      took={took:>10.3?} read={} found={} repaired={} passes={}",
        record.checked, record.mismatches, record.repaired, record.passes
    );
    // Signed: compaction during the check may leave the disk smaller.
    let grew = i128::from(disk(&db)) - i128::from(bytes_before);
    println!(
        "writes                     disk changed {} KiB, recovery flushes {}",
        grew / 1024,
        db.index_recovery_flushes() - flushes_before
    );
}

/// Bytes under `dir`, every file counted.
fn fs_extra_size(dir: &std::path::Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.metadata() {
            Ok(meta) if meta.is_dir() => total += fs_extra_size(&path),
            Ok(meta) => total += meta.len(),
            Err(_) => {}
        }
    }
    total
}
