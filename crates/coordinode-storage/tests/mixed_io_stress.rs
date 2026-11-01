//! Writers, readers, flushes and compactions on one engine at once, then a
//! reopen: every read and maintenance call succeeds while the files under
//! them are rewritten, and the store reopens holding each key's last write.
//! It runs on the engine's default file system (io_uring on Linux), where a
//! read once found a table file shorter than its footer. Its length is
//! `COORDINODE_STRESS_SECS` (default 3).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

const WRITERS: u64 = 4;
const READERS: usize = 4;
const KEYS_PER_WRITER: u64 = 2_000;

fn config(dir: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir,
        Media::Ssd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn key(writer: u64, n: u64) -> Vec<u8> {
    format!("node:{writer}:{n:08}").into_bytes()
}

/// A value of a few KiB, distinct per write.
fn value(writer: u64, n: u64, round: u64) -> Vec<u8> {
    let len = 512 + ((writer * 7919 + n * 104_729 + round) % 4096) as usize;
    let mut out = format!("{writer}:{n}:{round}:").into_bytes();
    out.resize(len, b'a' + (round % 26) as u8);
    out
}

#[test]
fn reads_and_maintenance_survive_files_rewritten_under_them() {
    let secs = std::env::var("COORDINODE_STRESS_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3u64);
    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(&config(dir.path())).unwrap();
    let stop = AtomicBool::new(false);
    let deadline = Instant::now() + Duration::from_secs(secs);
    let failures = parking_lot::Mutex::new(Vec::<String>::new());

    let last: Vec<HashMap<u64, Option<Vec<u8>>>> = std::thread::scope(|scope| {
        let writers: Vec<_> = (0..WRITERS)
            .map(|writer| {
                let (engine, stop, failures) = (&engine, &stop, &failures);
                scope.spawn(move || {
                    let mut last = HashMap::new();
                    let mut round = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        round += 1;
                        for n in (round % 7..KEYS_PER_WRITER).step_by(7) {
                            let k = key(writer, n);
                            let outcome = if (n + round).is_multiple_of(11) {
                                engine.delete(Partition::Node, &k).map(|()| None)
                            } else {
                                let v = value(writer, n, round);
                                engine.put(Partition::Node, &k, &v).map(|()| Some(v))
                            };
                            match outcome {
                                Ok(written) => {
                                    last.insert(n, written);
                                }
                                Err(e) => failures.lock().push(format!("write: {e}")),
                            }
                        }
                    }
                    last
                })
            })
            .collect();
        for reader in 0..READERS {
            let (engine, stop, failures) = (&engine, &stop, &failures);
            scope.spawn(move || {
                let mut n = reader as u64;
                while !stop.load(Ordering::Relaxed) {
                    n = n.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1)
                        % (WRITERS * KEYS_PER_WRITER);
                    if let Err(e) = engine.get(Partition::Node, &key(n % WRITERS, n / WRITERS)) {
                        failures.lock().push(format!("get: {e}"));
                    }
                    if n.is_multiple_of(64) {
                        let prefix = format!("node:{}:", n % WRITERS);
                        match engine.prefix_scan(Partition::Node, prefix.as_bytes()) {
                            Ok(rows) => {
                                for row in rows {
                                    if let Err(e) = row.into_inner() {
                                        failures.lock().push(format!("scan row: {e}"));
                                        break;
                                    }
                                }
                            }
                            Err(e) => failures.lock().push(format!("scan: {e}")),
                        }
                    }
                }
            });
        }
        {
            let (engine, stop, failures) = (&engine, &stop, &failures);
            scope.spawn(move || {
                let mut pass = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    pass += 1;
                    let done = if pass.is_multiple_of(3) {
                        engine.major_compact(Partition::Node)
                    } else {
                        engine
                            .persist_partition(Partition::Node)
                            .and_then(|()| engine.force_compaction(Partition::Node))
                    };
                    if let Err(e) = done {
                        failures.lock().push(format!("maintenance: {e}"));
                    }
                }
            });
        }
        while Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        stop.store(true, Ordering::Relaxed);
        writers.into_iter().map(|w| w.join().unwrap()).collect()
    });

    let failures = failures.into_inner();
    assert!(
        failures.is_empty(),
        "{} failures, first: {:?}",
        failures.len(),
        &failures[..failures.len().min(5)]
    );

    engine.persist().unwrap();
    drop(engine);
    let engine = StorageEngine::open(&config(dir.path())).unwrap();
    for (writer, written) in last.iter().enumerate() {
        for (n, want) in written {
            let got = engine
                .get(Partition::Node, &key(writer as u64, *n))
                .unwrap();
            assert_eq!(
                got.as_deref(),
                want.as_deref(),
                "key {writer}:{n} after the reopen"
            );
        }
    }
}
