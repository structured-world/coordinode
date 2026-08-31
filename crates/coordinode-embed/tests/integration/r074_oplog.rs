//! Integration tests: Oplog segment format (R074).
//!
//! Tests verify that:
//!   - OplogManager can append entries and read them back from sealed segments
//!   - Rotation, multi-segment reads, and purge work end-to-end
//!   - Checksum corruption is detected on open
//!   - Config fields from StorageConfig are respected

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_storage::oplog::{OplogEntry, OplogManager, OplogOp};

fn make_entry(index: u64, partition: u8, key: &[u8], value: &[u8]) -> OplogEntry {
    OplogEntry {
        ts: (index + 1) << 18,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition,
            key: key.to_vec(),
            value: value.to_vec(),
        }],
        is_migration: false,
        pre_images: None,
    }
}

fn open_manager(dir: &std::path::Path) -> OplogManager {
    OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open manager")
}

/// Entries appended to one segment survive a close + reopen cycle.
#[test]
fn oplog_entries_survive_close_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut mgr = open_manager(dir.path());
        for i in 0..20u64 {
            let key = format!("key-{i:04}");
            let val = format!("val-{i:04}");
            mgr.append(&make_entry(i, 1, key.as_bytes(), val.as_bytes()))
                .expect("append");
        }
        mgr.close().expect("close");
    }

    {
        let mut mgr = open_manager(dir.path());
        let entries = mgr.read_range(0, 20).expect("read_range");
        assert_eq!(entries.len(), 20);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.index, i as u64);
            let expected_key = format!("key-{i:04}").into_bytes();
            if let OplogOp::Insert { ref key, .. } = e.ops[0] {
                assert_eq!(key, &expected_key, "key mismatch at index {i}");
            }
        }
    }
}

/// Entries spanning multiple rotated segments are all readable.
#[test]
fn oplog_multi_segment_read_range() {
    let dir = tempfile::tempdir().expect("tempdir");

    let mut mgr =
        OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 10, 7 * 24 * 3600).expect("open");
    for i in 0..30u64 {
        mgr.append(&make_entry(i, 1, b"k", b"v")).expect("append");
    }
    mgr.rotate().expect("rotate");

    assert_eq!(
        mgr.sealed_segment_count(),
        3,
        "should have 3 segments after 30 entries / limit 10"
    );

    let entries = mgr.read_range(0, 30).expect("read_range");
    assert_eq!(entries.len(), 30);

    let partial = mgr.read_range(8, 18).expect("partial range");
    assert_eq!(partial.len(), 10);
    assert_eq!(partial[0].index, 8);
    assert_eq!(partial[9].index, 17);
}

/// Checksum corruption in an entry is detected when the segment is opened.
#[test]
fn oplog_corruption_detected_on_read() {
    let dir = tempfile::tempdir().expect("tempdir");

    let mut mgr = open_manager(dir.path());
    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1, b"key", b"val"))
            .expect("append");
    }
    mgr.rotate().expect("rotate");

    let seg_path = mgr.sealed_segment_paths()[0].to_path_buf();
    let mut data = std::fs::read(&seg_path).expect("read segment");
    let mid = data.len() / 2;
    data[mid] ^= 0xFF;
    std::fs::write(&seg_path, &data).expect("write corrupt");

    let result = mgr.read_range(0, 5);
    assert!(result.is_err(), "should fail on corrupt segment");
}

/// Auto-rotation respects max_entries from config.
#[test]
fn oplog_auto_rotation_max_entries() {
    let dir = tempfile::tempdir().expect("tempdir");

    let mut mgr =
        OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 5, 7 * 24 * 3600).expect("open");

    for i in 0..12u64 {
        mgr.append(&make_entry(i, 1, b"k", b"v")).expect("append");
    }
    // entries 0-4 → segment 1, 5-9 → segment 2, 10-11 → current
    assert_eq!(mgr.sealed_segment_count(), 2);

    let entries = mgr.read_range(0, 12).expect("read_range");
    assert_eq!(entries.len(), 12);
}

/// `verify_all` detects corruption in sealed segments.
#[test]
fn oplog_verify_all_detects_corruption() {
    let dir = tempfile::tempdir().expect("tempdir");

    let mut mgr = open_manager(dir.path());
    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1, b"k", b"v")).expect("append");
    }
    mgr.rotate().expect("rotate");

    let seg_path = mgr.sealed_segment_paths()[0].to_path_buf();
    let mut data = std::fs::read(&seg_path).expect("read");
    let mid = data.len() / 2;
    data[mid] ^= 0xFF;
    std::fs::write(&seg_path, &data).expect("write");

    let result = mgr.verify_all();
    assert!(result.is_err(), "verify_all must detect corruption");
}

/// `purge_expired` removes old segments and leaves recent ones intact.
#[test]
fn oplog_purge_expired_selective() {
    let dir = tempfile::tempdir().expect("tempdir");

    let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 3600).expect("open");

    let old_ts = 1u64 << 18;
    for i in 0..3u64 {
        let mut e = make_entry(i, 1, b"k", b"v");
        e.ts = old_ts;
        mgr.append(&e).expect("append old");
    }
    mgr.rotate().expect("rotate old");

    let recent_ts = 10_000_000u64 << 18;
    for i in 3..6u64 {
        let mut e = make_entry(i, 1, b"k", b"v");
        e.ts = recent_ts;
        mgr.append(&e).expect("append recent");
    }
    mgr.rotate().expect("rotate recent");

    assert_eq!(mgr.sealed_segment_count(), 2);

    // Time-only policy: no consumer floor, every entry durable.
    let purged = mgr
        .purge_with_floor(10_000, u64::MAX, &|_| true)
        .expect("purge");
    assert_eq!(purged, 1, "only the old segment should be purged");
    assert_eq!(mgr.sealed_segment_count(), 1, "recent segment must remain");

    let entries = mgr.read_range(3, 6).expect("read after purge");
    assert_eq!(entries.len(), 3);
}
