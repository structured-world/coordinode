use std::sync::Arc;

use coordinode_core::txn::proposal::{Mutation, PartitionId, UnitEffects};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::manager::OplogManager;

use super::{CdcHub, HubRead};

/// The entry at `index` carrying one unit frame.
fn entry(index: u64) -> OplogEntry {
    let frame = coordinode_core::txn::frame::encode_unit(
        &coordinode_storage::engine::scope::store_unit([
            Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:k{index}").into_bytes(),
                value: b"v".to_vec(),
            },
            Mutation::Effects(UnitEffects::new()),
        ]),
        Timestamp::from_raw(1000 + index),
    )
    .expect("frame");
    OplogEntry {
        ts: 1000 + index,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Unit { frame }],
        is_migration: false,
        pre_images: None,
    }
}

/// A log of `n` entries, indexes 0..n, left unsealed like a live one.
fn log(n: u64) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut manager =
        OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open");
    for i in 0..n {
        manager.append(&entry(i)).expect("append");
    }
    manager.flush().expect("flush");
    dir
}

fn indexes(read: HubRead) -> (Vec<u64>, u64) {
    match read {
        HubRead::Entries { entries, next } => (entries.iter().map(|e| e.index).collect(), next),
        HubRead::Behind(base) => panic!("behind the hub at {base}"),
    }
}

fn held(hub: &CdcHub) -> usize {
    hub.inner.lock().ring.len()
}

/// Two streams get the same entries from one read of the log, and an entry
/// stays in memory until both have passed it: a stream handed entries still
/// needs them until it says how far it got, since a read stopped inside an
/// entry reads it again.
#[test]
fn streams_share_entries_until_every_one_has_passed_them() {
    let dir = log(5);
    let hub = Arc::new(CdcHub::new(&[dir.path().to_path_buf()], 0, 0, 1 << 20).expect("hub"));
    let (a, b) = (hub.reader(0), hub.reader(0));

    let (got, next) = indexes(hub.read(&a, 0, 100, 5).expect("read a"));
    assert_eq!((got, next), (vec![0, 1, 2, 3, 4], 5));
    hub.passed(&a, next);
    assert_eq!(held(&hub), 5, "the other stream has not read them");

    let (got, _) = indexes(hub.read(&b, 0, 2, 5).expect("read b"));
    assert_eq!(got, [0, 1]);
    assert_eq!(held(&hub), 5, "the stream may still read entry 1 again");
    hub.passed(&b, 1);
    assert_eq!(held(&hub), 4, "it stopped inside entry 1, which it keeps");
    hub.passed(&b, 2);
    assert_eq!(held(&hub), 3, "entries both streams passed are released");

    drop(b);
    assert_eq!(held(&hub), 0, "a stream that leaves holds nothing back");
}

/// A stream positioned before the oldest entry the hub holds reads the gap
/// from the log itself.
#[test]
fn a_stream_behind_the_hub_is_told_where_it_starts() {
    let dir = log(6);
    let hub = Arc::new(CdcHub::new(&[dir.path().to_path_buf()], 0, 4, 1 << 20).expect("hub"));
    let early = hub.reader(1);
    let late = hub.reader(4);
    let (got, _) = indexes(hub.read(&late, 4, 100, 6).expect("read late"));
    assert_eq!(got, [4, 5]);
    assert!(matches!(
        hub.read(&early, 1, 100, 6).expect("read early"),
        HubRead::Behind(4)
    ));
}

/// Past the byte bound the oldest entries are released even if a slow
/// stream has not read them; that stream then reads from the log. A unit's
/// frame counts toward the bound.
#[test]
fn the_byte_bound_releases_entries_a_slow_stream_still_needs() {
    let dir = log(10);
    // Room for about three entries.
    let hub = Arc::new(CdcHub::new(&[dir.path().to_path_buf()], 0, 0, 900).expect("hub"));
    let (fast, slow) = (hub.reader(0), hub.reader(0));
    let (got, next) = indexes(hub.read(&fast, 0, 100, 10).expect("read fast"));
    assert_eq!(got.len(), 10);
    hub.passed(&fast, next);
    assert!(held(&hub) < 10, "the bound holds");
    assert!(matches!(
        hub.read(&slow, 0, 100, 10).expect("read slow"),
        HubRead::Behind(_)
    ));
}

/// The hub hands entries over as stored: a unit's frame whole, with the
/// effect context every stream projects it by, never expanded into the row
/// operations that lose it.
#[test]
fn entries_are_shared_with_their_unit_frames_whole() {
    let dir = log(2);
    let hub = Arc::new(CdcHub::new(&[dir.path().to_path_buf()], 0, 0, 1 << 20).expect("hub"));
    let reader = hub.reader(0);
    let HubRead::Entries { entries, .. } = hub.read(&reader, 0, 100, 2).expect("read") else {
        panic!("behind the hub");
    };
    assert_eq!(entries.len(), 2);
    for (shared, index) in entries.iter().zip(0..) {
        assert_eq!(**shared, entry(index));
    }
}
