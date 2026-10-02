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

fn tailer(dir: &std::path::Path, token: ResumeToken) -> OplogTailer {
    OplogTailer::new(&[dir.to_path_buf()], token).expect("tailer")
}

fn indexes(batch: &[(OplogEntry, ResumeToken)]) -> Vec<u64> {
    batch.iter().map(|(e, _)| e.index).collect()
}

#[test]
fn tailer_empty_dir_returns_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut tailer = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = tailer
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
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

    let mut tailer = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = tailer
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(batch.len(), 3, "active segment entries must be visible");
    assert_eq!(batch[2].0.index, 2);

    // Entries appended AFTER the first read are picked up on the next
    // read from the same tailer position.
    for i in 3..5u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    mgr.flush().expect("flush");
    let batch = tailer
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(indexes(&batch), vec![3, 4], "newly appended entries");
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

    let mut tailer = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = tailer
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
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

    let mut tailer = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = tailer
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");

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
    let mut tailer1 = tailer(dir.path(), ResumeToken::from_start(0));
    let batch1 = tailer1
        .read_next(3, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(batch1.len(), 3);
    let resume = batch1[2].1.clone();

    // Second batch from resume token: entries 3-5
    let mut tailer2 = tailer(dir.path(), resume);
    let batch2 = tailer2
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(indexes(&batch2), vec![3, 4, 5], "the 3 remaining entries");
}

/// A Raft log truncation deletes every segment and re-appends the kept
/// prefix under new file boundaries. A token issued before it still names
/// the same log index, and the stream resumes there: nothing below it is
/// sent again and nothing above it is skipped.
#[test]
fn tailer_resumes_by_index_across_a_log_rewrite() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());
    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);
    for i in 5..10u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);

    let mut first = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = first
        .read_next(8, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(indexes(&batch), (0..8).collect::<Vec<_>>());
    let token = batch[7].1.clone();
    assert_eq!((token.segment_id, token.entry_offset), (5, 3));

    // Truncate after index 8 the way the Raft log store does: wipe every
    // segment, re-append the kept prefix, then new entries follow.
    mgr.truncate_all().expect("truncate");
    for i in 0..=8u64 {
        mgr.append(&make_entry(i, 1000 + i, false))
            .expect("re-append");
    }
    for i in 9..12u64 {
        mgr.append(&make_entry(i, 2000 + i, false)).expect("append");
    }
    mgr.flush().expect("flush");

    let mut resumed = tailer(dir.path(), token);
    let batch = resumed
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(indexes(&batch), vec![8, 9, 10, 11], "resumes at index 8");
    assert_eq!(batch[1].0.ts, 2009, "the entry written after the rewrite");

    // The live cursor carries on the same way.
    let batch = first
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(indexes(&batch), vec![8, 9, 10, 11]);
}

/// Entries at or above the bound are not read: a Raft log holds entries
/// that are not committed yet, and a consumer must not see them.
#[test]
fn tailer_stops_below_the_bound() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());
    for i in 0..6u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    mgr.flush().expect("flush");

    let mut tailer = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = tailer
        .read_next(100, &CdcFilters::default(), 4)
        .expect("read");
    assert_eq!(indexes(&batch), vec![0, 1, 2, 3]);
    assert_eq!(tailer.next_index(), 4, "the cursor stays at the bound");
    assert!(
        tailer
            .read_next(100, &CdcFilters::default(), 4)
            .expect("read")
            .is_empty(),
        "nothing more until the bound moves"
    );

    let batch = tailer
        .read_next(100, &CdcFilters::default(), 6)
        .expect("read");
    assert_eq!(indexes(&batch), vec![4, 5], "the bound moved");
}

/// A sealed segment that cannot be read ends the read before it: the
/// entries it holds are not skipped for the ones in the next segment.
#[test]
fn tailer_does_not_skip_an_unreadable_segment() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());
    for i in 0..3u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);
    for i in 3..6u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);
    for i in 6..9u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);

    // Corrupt the middle segment, the one starting at index 3.
    let middle = std::fs::read_dir(dir.path())
        .expect("dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_prefix("oplog-"))
                .and_then(|s| s.parse::<u64>().ok())
                == Some(3)
        })
        .expect("middle segment");
    let good = std::fs::read(&middle).expect("read segment");
    std::fs::write(&middle, b"not a segment").expect("corrupt");

    let mut tailer = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = tailer
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(
        indexes(&batch),
        vec![0, 1, 2],
        "stops before the bad segment"
    );
    assert!(
        tailer
            .read_next(100, &CdcFilters::default(), u64::MAX)
            .expect("read")
            .is_empty(),
        "still waits there"
    );

    // Once readable again, the stream carries on in order.
    std::fs::write(&middle, good).expect("restore");
    let batch = tailer
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(indexes(&batch), (3..9).collect::<Vec<_>>());
}

#[test]
fn resume_token_names_a_log_index() {
    let token = ResumeToken {
        shard_id: 0,
        segment_id: 5,
        entry_offset: 3,
    };
    assert_eq!(token.next_index().expect("index"), 8);
    let overflow = ResumeToken {
        shard_id: 0,
        segment_id: u64::MAX,
        entry_offset: 1,
    };
    assert!(overflow.next_index().is_err());
    assert!(OplogTailer::new(&[], overflow).is_err());
}

/// A change of endpoint routing leaves the older segments of a log in the
/// directory they were written to; the tailer reads the log across every
/// directory in index order, and a directory that does not exist is skipped.
#[test]
fn tailer_reads_a_log_spread_over_directories() {
    let old = tempfile::tempdir().expect("tempdir");
    let new = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(old.path());
    for i in 0..3u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);
    drop(mgr);
    let mut mgr = OplogManager::open_multi(
        new.path(),
        &[old.path().to_path_buf(), new.path().to_path_buf()],
        0,
        64 * 1024 * 1024,
        50_000,
        7 * 24 * 3600,
    )
    .expect("open over both");
    for i in 3..6u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    mgr.flush().expect("flush");

    let missing = old.path().join("never-created");
    let mut tailer = OplogTailer::new(
        &[new.path().to_path_buf(), missing, old.path().to_path_buf()],
        ResumeToken::from_start(0),
    )
    .expect("tailer");
    let batch = tailer
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(indexes(&batch), (0..6).collect::<Vec<_>>());
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

    let filters = CdcFilters {
        is_migration: Some(false),
        ..Default::default()
    };
    let mut tailer = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = tailer.read_next(100, &filters, u64::MAX).expect("read");

    assert_eq!(batch.len(), 2, "only non-migration entries");
    assert!(batch.iter().all(|(e, _)| !e.is_migration));
    assert_eq!(tailer.next_index(), 4, "filtered entries are passed too");
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

    let filters = CdcFilters {
        edge_types: vec!["FOLLOWS".to_string()],
        ..Default::default()
    };
    let mut tailer = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = tailer.read_next(100, &filters, u64::MAX).expect("read");

    assert_eq!(indexes(&batch), vec![0, 2], "only FOLLOWS entries");
}

/// Two sealed segments, entries 0-4 and 5-9, with the first one purged.
fn purged_log(dir: &std::path::Path) -> OplogManager {
    let mut mgr = open_manager(dir);
    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);
    for i in 5..10u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);
    assert_eq!(
        mgr.purge_before(5).expect("purge"),
        1,
        "the first segment goes"
    );
    mgr
}

/// A token naming an index the log no longer holds is refused. Reading on
/// from the first retained segment would hand the reader a stream that
/// silently lacks every entry in between.
#[test]
fn a_token_below_the_retained_log_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let _mgr = purged_log(dir.path());

    let mut resumed = tailer(
        dir.path(),
        ResumeToken {
            shard_id: 0,
            segment_id: 0,
            entry_offset: 2,
        },
    );
    let err = resumed
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect_err("index 2 is purged");
    assert!(
        matches!(
            err,
            StorageError::RetentionLost {
                requested: 2,
                first_retained: 5
            }
        ),
        "expected retention lost at 2 with the log starting at 5, got {err:?}"
    );
}

/// A stream started without a token begins at the oldest entry the log
/// holds, whatever was purged before it.
#[test]
fn a_stream_from_the_start_begins_at_the_retained_log() {
    let dir = tempfile::tempdir().expect("tempdir");
    let _mgr = purged_log(dir.path());

    let mut fresh = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = fresh
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(indexes(&batch), (5..10).collect::<Vec<_>>());
}

/// A cursor that has read part of the log is refused once the entries after
/// its position are purged, instead of jumping over them.
#[test]
fn a_live_cursor_overtaken_by_a_purge_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());
    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);
    for i in 5..10u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    seal_manager(&mut mgr);

    let mut live = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = live
        .read_next(2, &CdcFilters::default(), u64::MAX)
        .expect("read");
    assert_eq!(indexes(&batch), vec![0, 1]);

    mgr.purge_before(5).expect("purge");
    let err = live
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect_err("indexes 2-4 are purged under the cursor");
    assert!(
        matches!(
            err,
            StorageError::RetentionLost {
                requested: 2,
                first_retained: 5
            }
        ),
        "expected retention lost at 2 with the log starting at 5, got {err:?}"
    );
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

    let mut tailer = tailer(dir.path(), ResumeToken::from_start(0));
    let batch = tailer
        .read_next(100, &CdcFilters::default(), u64::MAX)
        .expect("read");

    assert_eq!(indexes(&batch), (0..10).collect::<Vec<_>>());
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

    let passes = |entry: &OplogEntry, filters| passes_filter(entry, &entry.ops, filters);
    assert!(passes(&normal, &no_filter));
    assert!(passes(&migration, &no_filter));
    assert!(passes(&normal, &only_normal));
    assert!(!passes(&migration, &only_normal));
    assert!(!passes(&normal, &only_migration));
    assert!(passes(&migration, &only_migration));
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

    let passes = |entry: &OplogEntry, filters| passes_filter(entry, &entry.ops, filters);
    assert!(passes(&follows, &filter_follows));
    assert!(!passes(&likes, &filter_follows));
    assert!(!passes(&node, &filter_follows));

    assert!(passes(&follows, &filter_both));
    assert!(passes(&likes, &filter_both));
    assert!(!passes(&node, &filter_both));
}

/// A reader sees the operations a unit frame encodes, and filters by them:
/// an entry recorded as one frame streams and filters like one recorded op
/// by op.
#[test]
fn a_unit_frame_streams_as_its_operations() {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());
    let mutations = vec![
        Mutation::Put {
            partition: PartitionId::Adj,
            key: b"adj:FOLLOWS:out:1".to_vec(),
            value: b"v".to_vec(),
        },
        Mutation::Delete {
            partition: PartitionId::Counter,
            key: b"c".to_vec(),
        },
    ];
    let frame = coordinode_core::txn::frame::encode_unit(
        &mutations,
        coordinode_core::txn::timestamp::Timestamp::from_raw(1000),
    )
    .expect("frame");
    mgr.append(&OplogEntry {
        ts: 1000,
        term: 0,
        index: 0,
        shard: 0,
        ops: vec![OplogOp::Unit { frame }],
        is_migration: false,
        pre_images: None,
    })
    .expect("append");
    seal_manager(&mut mgr);

    let follows = CdcFilters {
        edge_types: vec!["FOLLOWS".to_string()],
        ..Default::default()
    };
    let batch = tailer(dir.path(), ResumeToken::from_start(0))
        .read_next(100, &follows, u64::MAX)
        .expect("read");
    assert_eq!(batch.len(), 1, "the frame's adjacency op passes the filter");
    assert_eq!(
        batch[0].0.ops,
        crate::oplog::convert::mutations_to_ops(&mutations).expect("ops")
    );

    let likes = CdcFilters {
        edge_types: vec!["LIKES".to_string()],
        ..Default::default()
    };
    let batch = tailer(dir.path(), ResumeToken::from_start(0))
        .read_next(100, &likes, u64::MAX)
        .expect("read");
    assert!(batch.is_empty());
}
