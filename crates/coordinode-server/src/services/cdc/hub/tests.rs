use std::sync::Arc;

use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::manager::OplogManager;
use coordinode_storage::oplog::tailer::CdcFilters;

use super::{CdcHub, HubRead};

fn entry(index: u64, key: &str) -> OplogEntry {
    OplogEntry {
        ts: 1000 + index,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 1,
            key: key.as_bytes().to_vec(),
            value: b"v".to_vec(),
        }],
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
        manager
            .append(&entry(i, &format!("node:k{i}")))
            .expect("append");
    }
    manager.flush().expect("flush");
    dir
}

fn indexes(read: HubRead) -> (Vec<u64>, u64) {
    match read {
        HubRead::Entries { entries, next } => (entries.iter().map(|e| e.0.index).collect(), next),
        HubRead::Behind(base) => panic!("behind the hub at {base}"),
    }
}

fn held(hub: &CdcHub) -> usize {
    hub.inner.lock().ring.len()
}

/// Two streams get the same entries from one read of the log, and an entry
/// stays in memory until both have passed it.
#[test]
fn streams_share_entries_until_every_one_has_passed_them() {
    let dir = log(5);
    let hub = Arc::new(CdcHub::new(&[dir.path().to_path_buf()], 0, 0, 1 << 20).expect("hub"));
    let (a, b) = (hub.reader(0), hub.reader(0));
    let all = CdcFilters::default();

    let (got, next) = indexes(hub.read(&a, 0, 100, 5, &all).expect("read a"));
    assert_eq!((got, next), (vec![0, 1, 2, 3, 4], 5));
    assert_eq!(held(&hub), 5, "the other stream has not read them");

    let (got, _) = indexes(hub.read(&b, 0, 2, 5, &all).expect("read b"));
    assert_eq!(got, [0, 1]);
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
    let all = CdcFilters::default();
    let (got, _) = indexes(hub.read(&late, 4, 100, 6, &all).expect("read late"));
    assert_eq!(got, [4, 5]);
    assert!(matches!(
        hub.read(&early, 1, 100, 6, &all).expect("read early"),
        HubRead::Behind(4)
    ));
}

/// Past the byte bound the oldest entries are released even if a slow
/// stream has not read them; that stream then reads from the log.
#[test]
fn the_byte_bound_releases_entries_a_slow_stream_still_needs() {
    let dir = log(10);
    // Room for about three entries.
    let hub = Arc::new(CdcHub::new(&[dir.path().to_path_buf()], 0, 0, 700).expect("hub"));
    let (fast, slow) = (hub.reader(0), hub.reader(0));
    let all = CdcFilters::default();
    let (got, _) = indexes(hub.read(&fast, 0, 100, 10, &all).expect("read fast"));
    assert_eq!(got.len(), 10);
    assert!(held(&hub) < 10, "the bound holds");
    assert!(matches!(
        hub.read(&slow, 0, 100, 10, &all).expect("read slow"),
        HubRead::Behind(_)
    ));
}

/// Each stream applies its own filters to the shared entries, and still
/// moves past the ones they drop.
#[test]
fn each_stream_filters_the_shared_entries_its_own_way() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut manager =
        OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open");
    manager.append(&entry(0, "node:a")).expect("append");
    manager
        .append(&entry(1, "adj:KNOWS:out:00000001"))
        .expect("append");
    manager.append(&entry(2, "node:b")).expect("append");
    manager.flush().expect("flush");
    let hub = Arc::new(CdcHub::new(&[dir.path().to_path_buf()], 0, 0, 1 << 20).expect("hub"));
    let (edges, everything) = (hub.reader(0), hub.reader(0));
    let knows = CdcFilters {
        edge_types: vec!["KNOWS".into()],
        is_migration: None,
    };
    let (got, next) = indexes(hub.read(&edges, 0, 100, 3, &knows).expect("read edges"));
    assert_eq!((got, next), (vec![1], 3));
    let (got, _) = indexes(
        hub.read(&everything, 0, 100, 3, &CdcFilters::default())
            .expect("read all"),
    );
    assert_eq!(got, [0, 1, 2]);
}
