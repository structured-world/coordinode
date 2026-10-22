//! What an index check costs while lookups go on: their p50, p99 and p999
//! before and during the check, the check's time, the entries and records it
//! read, what it repaired, the bytes it wrote, the journal syncs it cost and
//! this member's recovery-state flushes. The index holds damage the check
//! repairs entry by entry.
//!
//! A second case checks three indexes of one label at once, over a store
//! where the label's records carry a large unrelated property and another
//! label's records outweigh them: one shared pass reads the store once for
//! all three, where three checks one after another read it three times.
//! Each pass logs its rows read and decoded and its scan, sort, compare and
//! record times.
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
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

/// Nodes of the indexed label.
const USERS: usize = 20_000;

/// Entries lost before the check, every `USERS / LOST`-th one.
const LOST: usize = 200;

/// Nodes of a label no checked index covers, each with a large payload.
const DOCS: usize = 5_000;

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
        "{name:<26} lookups={:>6} p50={:>10.3?} p99={:>10.3?} p999={:>10.3?}",
        samples.len(),
        percentile(samples, 0.5),
        percentile(samples, 0.99),
        percentile(samples, 0.999)
    );
}

/// Run the checks of `names` together, with lookups going on; their
/// outcomes' records, the time until the last ended, and the lookups.
fn checks(
    db: &Database,
    names: &[&str],
) -> (
    Vec<coordinode_modality::index_def::IndexCheck>,
    Duration,
    Vec<Duration>,
) {
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let reader = scope.spawn(|| lookups(db, &stop, usize::MAX));
        let started = Instant::now();
        // Requested before any is awaited: checks waiting together are read
        // by one pass.
        let operations: Vec<_> = names
            .iter()
            .map(|name| {
                db.check_index(&IndexSelector::Name((*name).into()))
                    .expect("start the check")
            })
            .collect();
        let records = operations
            .into_iter()
            .map(|operation| {
                let (status, outcome) = db
                    .index_check(operation, Duration::from_secs(600))
                    .expect("read the check")
                    .expect("the check exists");
                let outcome = outcome.expect("the check ended");
                assert!(
                    matches!(outcome, CheckOutcome::Verified { .. }),
                    "{outcome:?}"
                );
                status.record.check.expect("check record")
            })
            .collect();
        let took = started.elapsed();
        stop.store(true, Ordering::Relaxed);
        (records, took, reader.join().expect("reader"))
    })
}

/// The check passes' cost since the last call, from the metrics they
/// record: rows read and decoded, entries, spills and seconds per phase.
fn pass_cost(snapshotter: &Snapshotter) -> String {
    let mut parts = Vec::new();
    for (key, _, _, value) in snapshotter.snapshot().into_vec() {
        let name = key.key().name();
        let Some(short) = name.strip_prefix("coordinode_index_check_") else {
            continue;
        };
        match value {
            // A snapshot drains what it reads: the value is this run's.
            DebugValue::Counter(n) => parts.push(format!("{short}={n}")),
            DebugValue::Histogram(samples) => {
                let phase = key
                    .key()
                    .labels()
                    .find(|l| l.key() == "phase")
                    .map_or("?", |l| l.value());
                let total: f64 = samples.iter().map(|s| s.into_inner()).sum();
                parts.push(format!("{phase}={:.3}s", total));
            }
            DebugValue::Gauge(_) => {}
        }
    }
    parts.sort();
    parts.join(" ")
}

fn main() {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    recorder.install().expect("install the metrics recorder");
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE INDEX u_city ON :U(city)")
        .expect("index");
    db.execute_cypher("CREATE INDEX u_n ON :U(n)")
        .expect("index");
    let bio = "b".repeat(512);
    db.execute_cypher(&format!(
        "UNWIND range(1, {USERS}) AS i CREATE (:U {{email: 'u' + toString(i) + '@x', n: i, \
         city: 'c' + toString(i % 100), bio: '{bio}'}})"
    ))
    .expect("users");
    let body = "d".repeat(4096);
    db.execute_cypher(&format!(
        "UNWIND range(1, {DOCS}) AS i CREATE (:Doc {{n: i, body: '{body}'}})"
    ))
    .expect("docs");

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
    let syncs_before = db.engine().journal_syncs().unwrap_or(0);
    let (records, took, during) = checks(&db, &["u_email"]);
    report("lookups, during the check", &during);
    let record = &records[0];
    db.engine().persist().expect("persist");
    println!(
        "check                      took={took:>10.3?} read={} found={} repaired={} passes={}",
        record.checked, record.mismatches, record.repaired, record.passes
    );
    // Signed: compaction during the check may leave the disk smaller.
    let grew = i128::from(disk(&db)) - i128::from(bytes_before);
    println!(
        "writes                     disk changed {} KiB, journal syncs {}, recovery flushes {}",
        grew / 1024,
        db.engine().journal_syncs().unwrap_or(0) - syncs_before,
        db.index_recovery_flushes() - flushes_before
    );
    println!("  passes: {}", pass_cost(&snapshotter));

    // The same three indexes checked one after another, then together.
    let mut apart = Duration::ZERO;
    let mut apart_passes = Vec::new();
    let syncs_before = db.engine().journal_syncs().unwrap_or(0);
    for name in ["u_email", "u_city", "u_n"] {
        let (records, took, _) = checks(&db, &[name]);
        apart += took;
        apart_passes.extend(records.iter().map(|r| r.passes));
    }
    let apart_syncs = db.engine().journal_syncs().unwrap_or(0) - syncs_before;
    println!(
        "three checks apart         took={apart:>10.3?} journal syncs {apart_syncs} \
         passes={apart_passes:?}\n  passes: {}",
        pass_cost(&snapshotter)
    );
    let syncs_before = db.engine().journal_syncs().unwrap_or(0);
    let (records, together, during) = checks(&db, &["u_email", "u_city", "u_n"]);
    let together_syncs = db.engine().journal_syncs().unwrap_or(0) - syncs_before;
    report("lookups, three checks", &during);
    println!(
        "three checks together      took={together:>10.3?} journal syncs {together_syncs} \
         passes={:?}\n  passes: {}",
        records.iter().map(|r| r.passes).collect::<Vec<_>>(),
        pass_cost(&snapshotter)
    );

    // A window a sampling profiler can attach to: the three checks again
    // and again for INDEX_CHECK_PROFILE_SECONDS.
    let window = std::env::var("INDEX_CHECK_PROFILE_SECONDS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs);
    if let Some(window) = window {
        println!("profile window");
        let started = Instant::now();
        let mut rounds = 0u64;
        while started.elapsed() < window {
            checks(&db, &["u_email", "u_city", "u_n"]);
            rounds += 1;
        }
        println!(
            "profile window             rounds={rounds} per round={:>10.3?}\n  passes: {}",
            started.elapsed() / u32::try_from(rounds).expect("rounds"),
            pass_cost(&snapshotter)
        );
    }
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
