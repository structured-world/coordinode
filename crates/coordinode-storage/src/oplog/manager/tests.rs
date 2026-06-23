use super::*;
use crate::oplog::entry::OplogOp;

fn make_entry(index: u64, ts: u64) -> OplogEntry {
    OplogEntry {
        ts,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 1,
            key: format!("k{index}").into_bytes(),
            value: b"v".to_vec(),
        }],
        is_migration: false,
        pre_images: None,
    }
}

/// The on-disk segment filename is a backup / recovery contract:
/// once shipped, external tooling depends on it. Pin the exact
/// shape so an accidental change to `segment_path` /
/// `parse_first_index` fails loudly here.
#[test]
fn segment_filename_is_canonical_contract() {
    let dir = Path::new("/oplog");
    let name = |idx| {
        segment_path(dir, idx)
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_owned)
            .expect("utf8 filename")
    };
    // `oplog-<first_index:020>.bin` — 20-digit zero-padded first_index.
    assert_eq!(name(1), "oplog-00000000000000000001.bin");
    assert_eq!(name(u64::MAX), "oplog-18446744073709551615.bin");

    // first_index round-trips out of the name.
    for idx in [0u64, 1, 42, 1_000_000, u64::MAX] {
        assert_eq!(
            parse_first_index(&segment_path(dir, idx)),
            Some(idx),
            "round-trip failed for {idx}",
        );
    }

    // 20-digit zero-pad makes lexicographic filename order == index
    // order (what recovery relies on to locate a segment by index).
    assert!(
        name(2) > name(1) && name(1_000_000) > name(999_999),
        "lexicographic filename order must match first_index order",
    );
}

fn test_manager(dir: &Path) -> OplogManager {
    OplogManager::open(
        dir,
        0,                // shard_id
        64 * 1024 * 1024, // max_bytes
        50_000,           // max_entries
        7 * 24 * 3600,    // retention_secs
    )
    .expect("open manager")
}

#[test]
fn append_and_read_range() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = test_manager(dir.path());

    for i in 0..10u64 {
        mgr.append(&make_entry(i, 1000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate");

    let entries = mgr.read_range(0, 10).expect("read_range");
    assert_eq!(entries.len(), 10);
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(e.index, i as u64, "index mismatch at position {i}");
    }
}

#[test]
fn read_range_partial_window() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = test_manager(dir.path());

    for i in 0..20u64 {
        mgr.append(&make_entry(i, 2000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate");

    let entries = mgr.read_range(5, 15).expect("read_range");
    assert_eq!(entries.len(), 10);
    assert_eq!(entries[0].index, 5);
    assert_eq!(entries[9].index, 14);
}

#[test]
fn read_range_forces_rotation_of_active_writer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = test_manager(dir.path());

    for i in 0..5u64 {
        mgr.append(&make_entry(i, 3000 + i)).expect("append");
    }
    // Current writer is NOT explicitly rotated — read_range must seal it.
    let entries = mgr.read_range(0, 5).expect("read_range");
    assert_eq!(entries.len(), 5);
}

#[test]
fn rotation_creates_new_segment_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = test_manager(dir.path());

    for i in 0..5u64 {
        mgr.append(&make_entry(i, 4000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate first");
    assert_eq!(mgr.sealed.len(), 1);

    for i in 5..10u64 {
        mgr.append(&make_entry(i, 5000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate second");
    assert_eq!(mgr.sealed.len(), 2);
}

#[test]
fn auto_rotation_on_entry_limit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = OplogManager::open(
        dir.path(),
        0,
        64 * 1024 * 1024,
        3, // max_entries = 3
        7 * 24 * 3600,
    )
    .expect("open");

    // entries 0-2 → segment 1 (auto-rotate when writing 3)
    // entries 3-5 → segment 2 (auto-rotate when writing 6)
    // entry   6   → current writer
    for i in 0..7u64 {
        mgr.append(&make_entry(i, 5000 + i)).expect("append");
    }
    assert_eq!(mgr.sealed.len(), 2, "should have 2 sealed segments");

    // read_range seals entry 6 → 3 sealed total
    let entries = mgr.read_range(0, 7).expect("read_range");
    assert_eq!(entries.len(), 7);
    assert_eq!(entries[6].index, 6);
}

#[test]
fn auto_rotation_on_byte_limit() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Very small byte limit: 1 byte forces rotation on every append after first
    let mut mgr = OplogManager::open(dir.path(), 0, 1, 50_000, 7 * 24 * 3600).expect("open");

    for i in 0..4u64 {
        mgr.append(&make_entry(i, 6000 + i)).expect("append");
    }
    // Each entry > 1 byte, so after writing entry 0 (total_bytes > 1),
    // entries 1, 2, 3 each trigger a rotation before being written.
    // Sealed: segments for entries 0, 1, 2 = 3 sealed, current has entry 3.
    assert_eq!(mgr.sealed.len(), 3);

    let entries = mgr.read_range(0, 4).expect("read_range");
    assert_eq!(entries.len(), 4);
}

#[test]
fn purge_expired_removes_old_segments() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 3600).expect("open");

    // Entries with ts = 100ms in HLC (100 << 18)
    let old_ts = 100u64 << 18;
    for i in 0..3u64 {
        let mut e = make_entry(i, old_ts);
        e.ts = old_ts;
        mgr.append(&e).expect("append old");
    }
    mgr.rotate().expect("rotate");
    assert_eq!(mgr.sealed.len(), 1);

    // now_secs=10000; cutoff = 10000-3600 = 6400s = 6_400_000ms
    // old segment last_ts ≈ 100ms << 18 ≪ 6_400_000ms << 18 → purged
    let purged = mgr.purge_expired(10_000).expect("purge");
    assert_eq!(purged, 1, "one expired segment should be removed");
    assert_eq!(mgr.sealed.len(), 0);
}

#[test]
fn purge_keeps_recent_segments() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 3600).expect("open");

    // Entries with ts at "now" = 10000s = 10_000_000ms
    let recent_ts = 10_000_000u64 << 18;
    for i in 0..3u64 {
        let mut e = make_entry(i, recent_ts);
        e.ts = recent_ts;
        mgr.append(&e).expect("append recent");
    }
    mgr.rotate().expect("rotate");

    let purged = mgr.purge_expired(10_000).expect("purge");
    assert_eq!(purged, 0, "recent segment must not be purged");
    assert_eq!(mgr.sealed.len(), 1);
}

/// Feed (b): a time-expired segment is KEPT when its last index is at or
/// above the consumer oplog floor, and PURGED once below it — the logical
/// OR of time-window and CDC-consumer need.
#[test]
fn purge_with_floor_keeps_consumer_needed_segments() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 3600).expect("open");

    // Two time-expired segments: seg0 = indices [0,2], seg1 = [3,5].
    let old_ts = 100u64 << 18;
    for i in 0..3u64 {
        let mut e = make_entry(i, old_ts);
        e.ts = old_ts;
        mgr.append(&e).expect("append seg0");
    }
    mgr.rotate().expect("rotate seg0");
    for i in 3..6u64 {
        let mut e = make_entry(i, old_ts);
        e.ts = old_ts;
        mgr.append(&e).expect("append seg1");
    }
    mgr.rotate().expect("rotate seg1");
    assert_eq!(mgr.sealed.len(), 2);

    // Consumer floor = 4: seg0 (last_index 2 < 4) is expired AND below the
    // floor → purged; seg1 (last_index 5 >= 4) is needed → kept despite
    // being time-expired.
    let purged = mgr.purge_with_floor(10_000, 4).expect("purge with floor");
    assert_eq!(
        purged, 1,
        "only the segment fully below the floor is purged"
    );
    assert_eq!(mgr.sealed.len(), 1, "consumer-needed segment retained");

    // Once the consumer advances past it (or none registered → u64::MAX),
    // the time-expired segment is collected.
    let purged = mgr
        .purge_with_floor(10_000, u64::MAX)
        .expect("purge time-only");
    assert_eq!(purged, 1, "no consumer need → pure time retention");
    assert_eq!(mgr.sealed.len(), 0);
}

#[test]
fn verify_all_passes_on_valid_data() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = test_manager(dir.path());

    for i in 0..5u64 {
        mgr.append(&make_entry(i, 9000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate");

    let count = mgr.verify_all().expect("verify_all");
    assert_eq!(count, 1);
}

#[test]
fn empty_manager_reads_empty_range() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = test_manager(dir.path());

    let entries = mgr.read_range(0, 100).expect("read_range on empty");
    assert!(entries.is_empty());
}

#[test]
fn purge_before_removes_fully_covered_segments() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 5, 86400).expect("open");

    // Segment 1: entries 0-4 (first_idx=0, entry_count=5, next_index=5)
    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate");

    // Segment 2: entries 5-9 (first_idx=5, entry_count=5, next_index=10)
    for i in 5..10u64 {
        mgr.append(&make_entry(i, 2000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate");

    assert_eq!(mgr.sealed.len(), 2);

    // purge_before(5, u64::MAX): index gate eligible, SST gate satisfied
    // by sentinel safe_ts — seg1 next_index=5 <= 5 and last_ts=1004 <= MAX
    // → purged; seg2 next_index=10 > 5 → kept by index gate.
    let purged = mgr.purge_before(5, u64::MAX).expect("purge_before");
    assert_eq!(purged, 1, "only the first segment should be purged");
    assert_eq!(mgr.sealed.len(), 1, "second segment must remain");

    // Entries 5-9 must still be readable.
    let entries = mgr.read_range(5, 10).expect("read after purge");
    assert_eq!(entries.len(), 5);
    assert_eq!(entries[0].index, 5);
    assert_eq!(entries[4].index, 9);
}

#[test]
fn purge_before_defers_when_partition_flush_lags() {
    // Regression for the cross-partition crash-safety hole: even though
    // openraft has applied the entries and called purge, the SST flush
    // watermark trails the segment's last_ts, so the segment must stay.
    // Replay through openraft is the only way to reconstruct mutations
    // still sitting in a partition memtable.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 5, 86400).expect("open");

    // Segment with entries ts=1000..1005.
    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate");
    assert_eq!(mgr.sealed.len(), 1);

    // applied_index is past the segment, but the SST watermark is below
    // the segment's last_ts (1004) — purge must skip.
    let purged = mgr
        .purge_before(/*applied_index*/ u64::MAX, /*safe_ts*/ 500)
        .expect("purge_before");
    assert_eq!(
        purged, 0,
        "segment must be retained when min_partition_flushed_seqno < last_ts"
    );
    assert_eq!(mgr.sealed.len(), 1, "segment still on disk");

    // Once flush catches up, the same call succeeds — the segment is now
    // safe to drop because every partition has the mutations in SST form.
    let purged = mgr
        .purge_before(u64::MAX, /*safe_ts*/ 1004)
        .expect("purge_before");
    assert_eq!(purged, 1, "segment purged after flush watermark advanced");
    assert!(mgr.sealed.is_empty());
}

#[test]
fn purge_before_with_zero_safe_ts_is_noop() {
    // Cold-start invariant: no SST has been written yet, so safe_ts == 0
    // and every non-empty segment has last_ts >= 1. Nothing must be purged
    // during startup recovery, even if applied_index is advanced.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 5, 86400).expect("open");

    for i in 0..3u64 {
        mgr.append(&make_entry(i, 1000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate");

    let purged = mgr
        .purge_before(/*applied_index*/ u64::MAX, /*safe_ts*/ 0)
        .expect("purge_before");
    assert_eq!(purged, 0, "fresh engine must never purge during startup");
    assert_eq!(mgr.sealed.len(), 1);
}

#[test]
fn truncate_all_clears_all_segments_including_active() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = test_manager(dir.path());

    // Two sealed segments + an active (unsealed) writer.
    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate first");
    for i in 5..10u64 {
        mgr.append(&make_entry(i, 2000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate second");
    for i in 10..15u64 {
        mgr.append(&make_entry(i, 3000 + i)).expect("append");
    }
    assert_eq!(mgr.sealed.len(), 2);
    assert!(mgr.current.is_some());

    mgr.truncate_all().expect("truncate_all");

    assert_eq!(mgr.sealed.len(), 0, "all sealed segments must be removed");
    assert!(
        mgr.current.is_none(),
        "active writer must be gone after seal+delete"
    );

    // No .bin files should remain on disk.
    let bin_files: Vec<_> = std::fs::read_dir(dir.path())
        .expect("readdir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
        .collect();
    assert!(
        bin_files.is_empty(),
        "no segment files should remain on disk"
    );
}

#[test]
fn read_range_across_multiple_segments() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 5, 86400).expect("open");

    // Writes 15 entries, rotating every 5
    for i in 0..15u64 {
        mgr.append(&make_entry(i, 7000 + i)).expect("append");
    }
    mgr.rotate().expect("rotate");
    assert_eq!(mgr.sealed.len(), 3);

    // Read a window that spans two segments
    let entries = mgr.read_range(3, 12).expect("read_range");
    assert_eq!(entries.len(), 9);
    assert_eq!(entries[0].index, 3);
    assert_eq!(entries[8].index, 11);
}

// ── Multi-endpoint open ─────────────────────────────────────────

/// `open_multi` discovers sealed segments living in a different
/// directory than the active one and merges them into the sealed
/// list in chronological order — proves cross-endpoint oplog
/// recovery actually scans `recovery_dirs`, not just the active
/// directory.
#[test]
fn open_multi_recovers_segments_from_recovery_dirs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let active = tmp.path().join("active");
    let recovery = tmp.path().join("recovery");

    // Pre-populate recovery dir with two sealed segments (manually
    // create empty files matching the segment_path naming convention
    // — open_multi only enumerates filenames, doesn't validate
    // contents for the path-merge logic).
    std::fs::create_dir_all(&recovery).expect("create recovery");
    let seg_a = segment_path(&recovery, 0);
    let seg_b = segment_path(&recovery, 100);
    std::fs::write(&seg_a, b"").expect("write seg a");
    std::fs::write(&seg_b, b"").expect("write seg b");

    // Write one segment in active dir at first_index = 200.
    std::fs::create_dir_all(&active).expect("create active");
    let seg_c = segment_path(&active, 200);
    std::fs::write(&seg_c, b"").expect("write seg c");

    let mgr = OplogManager::open_multi(
        &active,
        &[recovery.clone()],
        0,
        64 * 1024 * 1024,
        50_000,
        7 * 24 * 3600,
    )
    .expect("open_multi");

    // All three segments discovered, sorted by first_index.
    assert_eq!(mgr.sealed.len(), 3);
    assert_eq!(mgr.sealed[0].0, 0);
    assert_eq!(mgr.sealed[1].0, 100);
    assert_eq!(mgr.sealed[2].0, 200);
    // Active dir wins for new writes — active path preserved.
    assert_eq!(mgr.dir, active);
}

/// `open_multi` rejects ambiguous fork: two segments with the same
/// `first_index` across different endpoints means the operator
/// must reconcile before the engine boots.
#[test]
fn open_multi_rejects_duplicate_first_index_across_endpoints() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let active = tmp.path().join("active");
    let recovery = tmp.path().join("recovery");
    std::fs::create_dir_all(&active).expect("create active");
    std::fs::create_dir_all(&recovery).expect("create recovery");

    // Same first_index = 42 in both directories.
    std::fs::write(segment_path(&active, 42), b"").expect("write a");
    std::fs::write(segment_path(&recovery, 42), b"").expect("write b");

    let result = OplogManager::open_multi(
        &active,
        &[recovery],
        0,
        64 * 1024 * 1024,
        50_000,
        7 * 24 * 3600,
    );
    let err = match result {
        Ok(_) => panic!("duplicate first_index must fail"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("duplicate") && msg.contains("first_index"),
        "error should mention duplicate first_index, got: {msg}"
    );
}

/// `open_multi` tolerates missing recovery directories silently —
/// a previously-routed endpoint may have been removed from the
/// config and its directory pruned by the operator; this MUST NOT
/// fail the open.
#[test]
fn open_multi_skips_missing_recovery_dirs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let active = tmp.path().join("active");
    let missing = tmp.path().join("does_not_exist");

    let mgr = OplogManager::open_multi(
        &active,
        &[missing],
        0,
        64 * 1024 * 1024,
        50_000,
        7 * 24 * 3600,
    )
    .expect("missing recovery dir must not error");
    assert_eq!(mgr.sealed.len(), 0);
    assert!(active.exists(), "active dir must be created");
}
