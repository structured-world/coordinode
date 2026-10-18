//! The volatile write path end to end: how long a write under the memory
//! and cache concerns waits for its acknowledgement against a journaled one,
//! how long one write waits until a committed read sees it, and how long a
//! burst waits until every write is acknowledged and committed, with its
//! throughput. Each concern runs on its own store; the arms of the
//! acknowledgement comparison alternate so neither is always measured after
//! the other.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_core::txn::read_concern::{ReadConcern, ReadConcernLevel};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_embed::Database;
use coordinode_embed::db::StatementOptions;
use coordinode_embed::operation::IssuerId;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

/// Writes per acknowledgement arm.
const ROUNDS: usize = 2000;
/// Writes per commit-latency arm.
const COMMITS: usize = 200;
/// Writes per burst, unless `VOLATILE_BURST` names another count: a longer
/// burst gives a profiler time to sample it.
fn burst_size() -> usize {
    std::env::var("VOLATILE_BURST")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(5000)
}

fn report(label: &str, mut times: Vec<Duration>) {
    times.sort_unstable();
    let at = |q: usize| times[(times.len() - 1) * q / 1000];
    println!(
        "{label}: n={} p50={:?} p99={:?} p999={:?}",
        times.len(),
        at(500),
        at(990),
        at(999)
    );
}

fn open(dir: &std::path::Path) -> Database {
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.join("data"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    // `VOLATILE_FLUSH_MS` moves the journal's flush interval: past the end
    // of a burst, it shows what the burst costs with no sync beside it.
    if let Some(ms) = std::env::var("VOLATILE_FLUSH_MS")
        .ok()
        .and_then(|ms| ms.parse().ok())
    {
        config.volatile_flush_interval_ms = ms;
    }
    Database::open_with_config(config).expect("open")
}

fn write(db: &Database, concern: WriteConcern, label: &str, k: usize) -> Duration {
    let options = StatementOptions {
        write_concern: Some(concern),
        ..StatementOptions::default()
    };
    let start = Instant::now();
    db.execute_cypher_shared_with(
        &format!("CREATE (:{label} {{k: {k}}})"),
        None,
        None,
        &options,
    )
    .expect("write");
    start.elapsed()
}

fn landed(db: &Database, label: &str) -> u64 {
    db.label_counts()
        .expect("counts")
        .get(label)
        .copied()
        .unwrap_or(0)
}

/// Acknowledgement latency of each concern, arms alternating.
fn acknowledgements() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open(dir.path());
    let issuer = db.enroll_issuer(IssuerId([0x7a; 16])).expect("enroll");
    let mut journal = Vec::with_capacity(ROUNDS);
    let mut memory = Vec::with_capacity(ROUNDS);
    let mut cache = Vec::with_capacity(ROUNDS);
    let mut bound = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        let mut arms = [0u8, 1, 2, 3];
        arms.rotate_left(round % 4);
        for arm in arms {
            match arm {
                0 => journal.push(write(&db, WriteConcern::majority(), "J", round)),
                1 => memory.push(write(&db, WriteConcern::memory(), "M", round)),
                2 => cache.push(write(&db, WriteConcern::cache(), "C", round)),
                _ => {
                    let options = StatementOptions {
                        write_concern: Some(WriteConcern::memory()),
                        operation: Some(issuer.next().expect("number")),
                        ..StatementOptions::default()
                    };
                    let start = Instant::now();
                    db.execute_cypher_shared_with("CREATE (:B {k: 1})", None, None, &options)
                        .expect("bound write");
                    bound.push(start.elapsed());
                }
            }
        }
    }
    report("ack journal", journal);
    report("ack memory", memory);
    report("ack cache", cache);
    report("ack memory with an operation", bound);
}

/// How long one write under `concern`, sent alone, takes until a committed
/// read sees it. Measured from the send, write after write.
fn commit_latency(concern: WriteConcern, name: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open(dir.path());
    let mut times = Vec::with_capacity(COMMITS);
    for k in 0..COMMITS {
        let start = Instant::now();
        write(&db, concern, "Lone", k);
        while landed(&db, "Lone") < k as u64 + 1 {
            std::thread::yield_now();
        }
        times.push(start.elapsed());
    }
    report(&format!("commit {name}"), times);
}

fn read_concern(level: ReadConcernLevel) -> ReadConcern {
    ReadConcern {
        level,
        ..ReadConcern::default()
    }
}

fn count_at(db: &mut Database, level: ReadConcernLevel, label: &str) -> usize {
    db.execute_cypher_with_read_concern(
        &format!("MATCH (n:{label}) RETURN n.k"),
        read_concern(level),
    )
    .expect("read")
    .len()
}

/// How long one write under `concern`, sent alone, takes until a majority
/// read sees it: for a volatile concern, until the journal has synced it.
fn durable_latency(concern: WriteConcern, name: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    let mut times = Vec::with_capacity(COMMITS);
    for k in 0..COMMITS {
        let start = Instant::now();
        write(&db, concern, "Lone", k);
        while count_at(&mut db, ReadConcernLevel::Majority, "Lone") < k + 1 {
            std::thread::yield_now();
        }
        times.push(start.elapsed());
    }
    report(&format!("durable {name}"), times);
}

/// Ordinary read latency at each read concern while a volatile write sits
/// unsynced (run with a long `VOLATILE_FLUSH_MS` to hold the flusher off):
/// choosing the durable view costs no sync and no wait.
fn reads_over_an_unsynced_tail() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = open(dir.path());
    for k in 0..100 {
        write(&db, WriteConcern::majority(), "Seen", k);
    }
    write(&db, WriteConcern::memory(), "Seen", 100);
    for (level, name) in [
        (ReadConcernLevel::Local, "local"),
        (ReadConcernLevel::Majority, "majority"),
    ] {
        let mut times = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let start = Instant::now();
            count_at(&mut db, level, "Seen");
            times.push(start.elapsed());
        }
        report(&format!("read {name} over an unsynced write"), times);
    }
}

/// A burst under `concern`: the time until every write is acknowledged and
/// the time until every write is committed.
fn burst(concern: WriteConcern, name: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open(dir.path());
    let n = burst_size();
    let start = Instant::now();
    for k in 0..n {
        write(&db, concern, "Burst", k);
    }
    let acked = start.elapsed();
    while landed(&db, "Burst") < n as u64 {
        std::thread::sleep(Duration::from_millis(1));
    }
    let committed = start.elapsed();
    println!(
        "burst {name}: n={n} acked={acked:?} committed={committed:?} lag={:?} \
         throughput={:.0}/s",
        committed - acked,
        n as f64 / committed.as_secs_f64(),
    );
}

/// Journaled writes the killed writer leaves for the next open to replay.
const KILLED_WRITES: usize = 2000;
/// Set to a directory, the bench runs as the killed writer there.
const KILLED_WRITER: &str = "VOLATILE_KILLED_WRITER";

/// The killed writer: journaled writes, then the process dies without
/// closing the store, as a crash leaves it.
fn killed_writer(dir: &std::path::Path) -> ! {
    let db = open(dir);
    for k in 0..KILLED_WRITES {
        write(&db, WriteConcern::majority(), "Killed", k);
    }
    std::process::abort()
}

/// How long an open takes: of a new store, and of one a killed process left
/// with its journal to replay.
fn opens() {
    let mut fresh = Vec::with_capacity(20);
    for _ in 0..20 {
        let dir = tempfile::tempdir().expect("tempdir");
        let start = Instant::now();
        let db = open(dir.path());
        fresh.push(start.elapsed());
        drop(db);
    }
    report("open new", fresh);
    let mut recovered = Vec::with_capacity(5);
    for _ in 0..5 {
        let dir = tempfile::tempdir().expect("tempdir");
        let status = std::process::Command::new(std::env::current_exe().expect("own path"))
            .env(KILLED_WRITER, dir.path())
            .status()
            .expect("run the writer");
        assert!(!status.success(), "the writer was not killed");
        let start = Instant::now();
        let db = open(dir.path());
        recovered.push(start.elapsed());
        assert_eq!(
            landed(&db, "Killed"),
            KILLED_WRITES as u64,
            "an acknowledged write was lost"
        );
    }
    report(
        &format!("open after a kill with {KILLED_WRITES} writes"),
        recovered,
    );
}

fn main() {
    if let Some(dir) = std::env::var_os(KILLED_WRITER) {
        killed_writer(std::path::Path::new(&dir));
    }
    // `VOLATILE_PARTS` picks parts by name (`ack`, `commit`, `durable`,
    // `read`, `burst`, `open`); all run when it is unset.
    let parts = std::env::var("VOLATILE_PARTS").unwrap_or_default();
    let runs = |part: &str| parts.is_empty() || parts.split(',').any(|p| p == part);
    if runs("ack") {
        acknowledgements();
    }
    if runs("commit") {
        commit_latency(WriteConcern::majority(), "journal");
        commit_latency(WriteConcern::memory(), "memory");
        commit_latency(WriteConcern::cache(), "cache");
    }
    if runs("durable") {
        durable_latency(WriteConcern::majority(), "journal");
        durable_latency(WriteConcern::memory(), "memory");
        durable_latency(WriteConcern::cache(), "cache");
    }
    if runs("read") {
        reads_over_an_unsynced_tail();
    }
    if runs("open") {
        opens();
    }
    if !runs("burst") {
        return;
    }
    burst(WriteConcern::majority(), "journal");
    burst(WriteConcern::memory(), "memory");
    burst(WriteConcern::cache(), "cache");
}
