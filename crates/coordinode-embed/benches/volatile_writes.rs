//! The volatile write path end to end: how long a write under the memory
//! and cache concerns waits for its acknowledgement against a journaled one,
//! how long a burst waits until it is committed (the drain lag), the burst's
//! throughput into the log, and the bytes the buffer retains meanwhile.
//! Each concern runs on its own store; the arms of the acknowledgement
//! comparison alternate so neither is always measured after the other.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_embed::Database;
use coordinode_embed::db::StatementOptions;
use coordinode_embed::operation::IssuerId;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

/// Writes per acknowledgement arm.
const ROUNDS: usize = 2000;
/// Writes per burst.
const BURST: usize = 5000;

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
    config.nvme_write_buffer_path = Some(dir.join("nvme"));
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

/// A burst under `concern`: the time until every write is acknowledged, the
/// time until every write is committed, and the most bytes buffered between.
fn burst(concern: WriteConcern, name: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open(dir.path());
    let start = Instant::now();
    for k in 0..BURST {
        write(&db, concern, "Burst", k);
    }
    let acked = start.elapsed();
    let mut retained = db.volatile_buffered_bytes();
    while landed(&db, "Burst") < BURST as u64 {
        retained = retained.max(db.volatile_buffered_bytes());
        std::thread::sleep(Duration::from_millis(1));
    }
    let committed = start.elapsed();
    println!(
        "burst {name}: n={BURST} acked={acked:?} committed={committed:?} drain_lag={:?} \
         throughput={:.0}/s retained_peak_bytes={retained}",
        committed - acked,
        BURST as f64 / committed.as_secs_f64(),
    );
}

fn main() {
    acknowledgements();
    burst(WriteConcern::majority(), "journal");
    burst(WriteConcern::memory(), "memory");
    burst(WriteConcern::cache(), "cache");
}
