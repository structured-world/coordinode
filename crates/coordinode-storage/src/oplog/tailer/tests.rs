use super::*;
use crate::oplog::entry::{OplogEntry, OplogOp};
use crate::oplog::manager::OplogManager;

fn make_entry(index: u64, ts: u64, is_migration: bool) -> OplogEntry {
    OplogEntry {
        ts,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 1,
            key: format!("node:k{index}").into_bytes(),
            value: b"v".to_vec(),
        }],
        is_migration,
        pre_images: None,
    }
}

fn make_adj_entry(index: u64, edge_type: &str) -> OplogEntry {
    OplogEntry {
        ts: 1000 + index,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 2,
            key: format!("adj:{edge_type}:out:00000001").into_bytes(),
            value: b"postinglist".to_vec(),
        }],
        is_migration: false,
        pre_images: None,
    }
}

fn open_manager(dir: &std::path::Path) -> OplogManager {
    OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open manager")
}

/// Seal the manager so all entries are in sealed segments readable by the tailer.
fn seal_manager(mgr: &mut OplogManager) {
    mgr.rotate().expect("seal");
}

#[test]
fn tailer_empty_dir_returns_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let token = ResumeToken::from_start(0);
    let mut tailer = OplogTailer::new(dir.path(), token);
    let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");
    assert!(batch.is_empty(), "empty dir = empty batch");
}

/// Live consumers must see entries in the ACTIVE (unsealed) segment;
/// waiting for the seal means up to a full rotation of lag.
#[test]
fn tailer_reads_active_segment_incrementally() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    for i in 0..3u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    mgr.flush().expect("flush");
    // NO seal: the segment is still active.

    let token = ResumeToken::from_start(0);
    let mut tailer = OplogTailer::new(dir.path(), token);
    let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");
    assert_eq!(batch.len(), 3, "active segment entries must be visible");
    assert_eq!(batch[2].0.index, 2);

    // Entries appended AFTER the first read are picked up on the next
    // read from the same tailer position.
    for i in 3..5u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    mgr.flush().expect("flush");
    let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");
    assert_eq!(batch.len(), 2, "newly appended entries must be visible");
    assert_eq!(batch[0].0.index, 3);
    assert_eq!(batch[1].0.index, 4);
}

/// A torn write at the tail of the active segment must not break the
/// prefix read: complete entries before it are returned.
#[test]
fn tailer_active_segment_ignores_torn_tail() {
    use std::io::Write as _;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());
    for i in 0..3u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    mgr.flush().expect("flush");
    drop(mgr); // release the file handle; segment stays unsealed

    // Simulate a torn write: garbage bytes at the end of the file.
    let seg_path = std::fs::read_dir(dir.path())
        .expect("dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "oplog"))
        .or_else(|| {
            std::fs::read_dir(dir.path())
                .ok()?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.is_file())
        })
        .expect("segment file");
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(&seg_path)
        .expect("open for append");
    f.write_all(&[0x07, 0xde, 0xad, 0xbe]).expect("torn bytes");
    drop(f);

    let token = ResumeToken::from_start(0);
    let mut tailer = OplogTailer::new(dir.path(), token);
    let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");
    assert_eq!(batch.len(), 3, "complete prefix must survive a torn tail");
}

#[test]
fn tailer_reads_sealed_segment() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);

    let token = ResumeToken::from_start(0);
    let mut tailer = OplogTailer::new(dir.path(), token);
    let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");

    assert_eq!(batch.len(), 5, "all 5 entries must be returned");
    assert_eq!(batch[0].0.index, 0);
    assert_eq!(batch[4].0.index, 4);
    // Token after last entry should point to entry_offset=5
    assert_eq!(batch[4].1.entry_offset, 5);
}

#[test]
fn tailer_resumes_from_token() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    for i in 0..6u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);

    // First batch: consume entries 0-2
    let token = ResumeToken::from_start(0);
    let mut tailer = OplogTailer::new(dir.path(), token);
    let batch1 = tailer.read_next(3, &CdcFilters::default()).expect("read");
    assert_eq!(batch1.len(), 3);
    let resume = batch1[2].1.clone();

    // Second batch from resume token: entries 3-5
    let mut tailer2 = OplogTailer::new(dir.path(), resume);
    let batch2 = tailer2
        .read_next(100, &CdcFilters::default())
        .expect("read");
    assert_eq!(batch2.len(), 3, "must get 3 remaining entries");
    assert_eq!(batch2[0].0.index, 3);
    assert_eq!(batch2[2].0.index, 5);
}

#[test]
fn tailer_filter_is_migration() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    mgr.append(&make_entry(0, 1000, false)).expect("append");
    mgr.append(&make_entry(1, 1001, true)).expect("append");
    mgr.append(&make_entry(2, 1002, false)).expect("append");
    mgr.append(&make_entry(3, 1003, true)).expect("append");
    seal_manager(&mut mgr);

    let token = ResumeToken::from_start(0);
    let filters = CdcFilters {
        is_migration: Some(false),
        ..Default::default()
    };
    let mut tailer = OplogTailer::new(dir.path(), token);
    let batch = tailer.read_next(100, &filters).expect("read");

    assert_eq!(batch.len(), 2, "only non-migration entries");
    assert!(batch.iter().all(|(e, _)| !e.is_migration));
}

#[test]
fn tailer_filter_edge_type() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    mgr.append(&make_adj_entry(0, "FOLLOWS")).expect("append");
    mgr.append(&make_adj_entry(1, "LIKES")).expect("append");
    mgr.append(&make_adj_entry(2, "FOLLOWS")).expect("append");
    mgr.append(&make_entry(3, 1003, false)).expect("append"); // non-adj
    seal_manager(&mut mgr);

    let token = ResumeToken::from_start(0);
    let filters = CdcFilters {
        edge_types: vec!["FOLLOWS".to_string()],
        ..Default::default()
    };
    let mut tailer = OplogTailer::new(dir.path(), token);
    let batch = tailer.read_next(100, &filters).expect("read");

    assert_eq!(batch.len(), 2, "only FOLLOWS entries");
    assert_eq!(batch[0].0.index, 0);
    assert_eq!(batch[1].0.index, 2);
}

#[test]
fn tailer_reads_across_multiple_segments() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    // Segment 1: entries 0-4
    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);

    // Segment 2: entries 5-9
    for i in 5..10u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);

    let token = ResumeToken::from_start(0);
    let mut tailer = OplogTailer::new(dir.path(), token);
    let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");

    assert_eq!(batch.len(), 10, "all 10 entries across 2 segments");
    assert_eq!(batch[0].0.index, 0);
    assert_eq!(batch[9].0.index, 9);
}

#[test]
fn passes_filter_migration() {
    let no_filter = CdcFilters::default();
    let only_normal = CdcFilters {
        is_migration: Some(false),
        ..Default::default()
    };
    let only_migration = CdcFilters {
        is_migration: Some(true),
        ..Default::default()
    };

    let normal = make_entry(0, 1000, false);
    let migration = make_entry(1, 1001, true);

    assert!(passes_filter(&normal, &no_filter));
    assert!(passes_filter(&migration, &no_filter));
    assert!(passes_filter(&normal, &only_normal));
    assert!(!passes_filter(&migration, &only_normal));
    assert!(!passes_filter(&normal, &only_migration));
    assert!(passes_filter(&migration, &only_migration));
}

#[test]
fn passes_filter_edge_type() {
    let follows = make_adj_entry(0, "FOLLOWS");
    let likes = make_adj_entry(1, "LIKES");
    let node = make_entry(2, 1002, false);

    let filter_follows = CdcFilters {
        edge_types: vec!["FOLLOWS".to_string()],
        ..Default::default()
    };
    let filter_both = CdcFilters {
        edge_types: vec!["FOLLOWS".to_string(), "LIKES".to_string()],
        ..Default::default()
    };

    assert!(passes_filter(&follows, &filter_follows));
    assert!(!passes_filter(&likes, &filter_follows));
    assert!(!passes_filter(&node, &filter_follows));

    assert!(passes_filter(&follows, &filter_both));
    assert!(passes_filter(&likes, &filter_both));
    assert!(!passes_filter(&node, &filter_both));
}
