//! Recovery of `w:cache` writes a process accepted and did not drain before
//! it stopped: each is applied exactly once, whether the crash came before
//! its drain or after the drain applied it and before the buffer file was
//! cleaned up.
//!
//! A crash is reproduced on disk rather than with a sleep: the buffer files
//! are written through the buffer's own API, and "applied before the crash"
//! is the entry's unit applied to the store, its mark included, with the
//! files left behind.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::txn::drain::DrainEntry;
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_embed::Database;
use coordinode_storage::cache::write_buffer::NvmeWriteBuffer;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::merge::{decode_counter, encode_counter_delta};
use coordinode_storage::engine::partition::Partition;

const COUNTER: &[u8] = b"counter:recovered";

fn config(dir: &std::path::Path) -> StorageConfig {
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.join("data"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.nvme_write_buffer_path = Some(dir.join("nvme"));
    config
}

/// One buffered entry: adds `delta` to the counter, an additive operand that
/// shows a second application.
fn increment(delta: i64) -> DrainEntry {
    DrainEntry::new(
        vec![Mutation::Merge {
            partition: PartitionId::Counter,
            key: COUNTER.to_vec(),
            operand: encode_counter_delta(delta),
        }],
        Timestamp::from_raw(1),
        Timestamp::from_raw(1),
    )
}

fn counter(db: &Database) -> i64 {
    db.engine()
        .get(Partition::Counter, COUNTER)
        .unwrap()
        .map_or(0, |bytes| decode_counter(&bytes).unwrap())
}

/// Leave behind the buffer files of a crashed process holding `entries`,
/// each numbered and marked as the write path does. Returns them as the
/// log would have received them.
fn crashed_buffer(dir: &std::path::Path, entries: Vec<DrainEntry>) -> Vec<DrainEntry> {
    let buffer = NvmeWriteBuffer::open(&dir.join("nvme"), 42).unwrap();
    // The drain never runs: the process is gone before it would.
    let drain = coordinode_core::txn::drain::DrainBuffer::new(1 << 20);
    for entry in entries {
        buffer.append(entry, &drain).unwrap();
    }
    drain.take_all()
}

/// Apply `entry` to the store as the drain would have before the crash: its
/// whole unit, mark included.
fn applied_before_the_crash(dir: &std::path::Path, entry: &DrainEntry) {
    let mut config = config(dir);
    config.nvme_write_buffer_path = None;
    let db = Database::open_with_config(config).unwrap();
    db.engine()
        .apply_proposal_at(
            &entry.mutations,
            db.engine().oracle().unwrap().next().as_raw(),
        )
        .unwrap();
}

/// Accepted, then the process stopped before the drain: reopening applies
/// the entry, once, and a second reopen finds nothing left to apply.
#[test]
fn a_cache_write_lost_before_its_drain_lands_once_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    crashed_buffer(dir.path(), vec![increment(5)]);

    let db = Database::open_with_config(config(dir.path())).unwrap();
    assert_eq!(counter(&db), 5);
    drop(db);
    let db = Database::open_with_config(config(dir.path())).unwrap();
    assert_eq!(counter(&db), 5, "replayed again on the next open");
}

/// The drain applied the entry, then the process stopped before the buffer
/// file was cleaned up: the mark the unit carried says so, and reopening
/// applies nothing again.
#[test]
fn a_cache_write_applied_before_the_crash_is_not_applied_again() {
    let dir = tempfile::tempdir().unwrap();
    let entries = crashed_buffer(dir.path(), vec![increment(5)]);
    applied_before_the_crash(dir.path(), &entries[0]);

    let db = Database::open_with_config(config(dir.path())).unwrap();
    assert_eq!(counter(&db), 5);
}

/// A drain pass that applied the first entries and not the rest: each lands
/// once, the applied ones skipped, the others applied in order.
#[test]
fn a_partly_drained_buffer_lands_each_entry_once() {
    let dir = tempfile::tempdir().unwrap();
    let entries = crashed_buffer(
        dir.path(),
        vec![increment(1), increment(10), increment(100)],
    );
    applied_before_the_crash(dir.path(), &entries[0]);

    let db = Database::open_with_config(config(dir.path())).unwrap();
    assert_eq!(counter(&db), 111);
}

/// A buffer file written before entries carried numbers cannot say which
/// of them were applied: the store does not open, and the file stays where
/// it is, neither replayed nor deleted.
#[test]
fn an_unnumbered_buffer_file_refuses_the_open_and_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    let nvme = dir.path().join("nvme");
    std::fs::create_dir_all(&nvme).unwrap();
    let file = nvme.join("write_buffer_current.bin");
    std::fs::write(&file, [20u8, 0, 0, 0, 1, 0, 0, 0, 9, 9, 9, 9]).unwrap();

    let refused = Database::open_with_config(config(dir.path()));
    let message = match refused {
        Err(e) => e.to_string(),
        Ok(_) => panic!("an unnumbered buffer was accepted"),
    };
    assert!(message.contains("predates numbered entries"), "{message}");
    assert!(file.exists());
}

/// A cache write through the database is numbered and drained like the ones
/// above: closed and reopened, it is in the store once.
#[test]
fn a_cache_write_through_the_database_lands_once() {
    use coordinode_core::txn::write_concern::WriteConcern;
    let dir = tempfile::tempdir().unwrap();
    {
        let mut db = Database::open_with_config(config(dir.path())).unwrap();
        db.set_write_concern(WriteConcern::cache());
        db.execute_cypher("CREATE (n:Cached {k: 1})").unwrap();
    }
    let db = Database::open_with_config(config(dir.path())).unwrap();
    let counts = db.label_counts().unwrap();
    assert_eq!(counts.get("Cached").copied(), Some(1), "{counts:?}");
}
