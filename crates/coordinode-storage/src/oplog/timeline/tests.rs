use super::*;
use crate::oplog::entry::{OplogEntry, OplogOp};
use crate::oplog::manager::OplogManager;

/// Bytes of value each entry carries: about a kilobyte per frame.
const VALUE_BYTES: usize = 1000;

fn entry(index: u64) -> OplogEntry {
    OplogEntry {
        ts: 1_000_000 * (index + 1),
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 1,
            key: format!("node:k{index}").into_bytes(),
            value: vec![7u8; VALUE_BYTES],
        }],
        is_migration: false,
        pre_images: None,
    }
}

fn open_manager(dir: &Path) -> OplogManager {
    OplogManager::open(dir, 0, 64 * 1024 * 1024, 1_000_000, 7 * 24 * 3600).expect("open manager")
}

/// Append entries `from..to` and make them readable.
fn append(mgr: &mut OplogManager, from: u64, to: u64) {
    for index in from..to {
        mgr.append(&entry(index)).expect("append");
    }
    mgr.flush().expect("flush");
}

fn timeline(dir: &Path) -> OplogTimeline {
    OplogTimeline::new(vec![dir.to_path_buf()])
}

/// Every position answers with the timestamp its entry was written with,
/// in sealed segments and in the open one, and a position not written yet
/// has no answer.
#[test]
fn answers_the_timestamp_written_at_each_position() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());
    append(&mut mgr, 0, 150);
    mgr.rotate().expect("seal");
    append(&mut mgr, 150, 400);

    let timeline = timeline(dir.path());
    for index in [0, 1, 63, 64, 149, 150, 151, 260, 399] {
        assert_eq!(
            timeline.written_at(index).expect("read"),
            Some(entry(index).ts),
            "position {index}"
        );
    }
    assert_eq!(timeline.written_at(400).expect("read"), None);
}

/// The bound check asks the age of the same acknowledged positions every
/// sweep. Reading the segment holding a position to answer made every ask
/// cost that segment's size, about 100 MB/s with two consumers on a busy
/// log. Asked again, a position costs nothing; a new position in an already
/// scanned segment costs about one stride.
#[test]
fn a_position_asked_again_reads_nothing_and_a_new_one_about_a_stride() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());
    append(&mut mgr, 0, 4_000);
    let size = std::fs::metadata(
        list_segments(&[dir.path().to_path_buf()]).expect("list")[0]
            .1
            .clone(),
    )
    .expect("size")
    .len();

    // The first ask of the newest position scans the segment once.
    let timeline = timeline(dir.path());
    assert_eq!(
        timeline.written_at(3_999).expect("read"),
        Some(entry(3_999).ts)
    );
    let after_scan = timeline.bytes_read();
    assert!(
        after_scan < size + 4 * MARK_STRIDE,
        "{after_scan} of {size}"
    );

    assert_eq!(timeline.written_at(10).expect("read"), Some(entry(10).ts));
    let asked = timeline.bytes_read();
    for _ in 0..100 {
        assert_eq!(timeline.written_at(10).expect("read"), Some(entry(10).ts));
    }
    assert_eq!(timeline.bytes_read(), asked, "asked again: no read");

    let mut before = timeline.bytes_read();
    for index in (20..4_000).step_by(97) {
        assert_eq!(
            timeline.written_at(index).expect("read"),
            Some(entry(index).ts)
        );
        let cost = timeline.bytes_read() - before;
        assert!(cost <= 3 * MARK_STRIDE, "position {index} read {cost}");
        before = timeline.bytes_read();
    }
}

/// The open segment grows: a scan reads what was appended since the last
/// one, not the segment from its start.
#[test]
fn a_growing_segment_is_scanned_only_where_it_grew() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());
    append(&mut mgr, 0, 2_000);

    let timeline = timeline(dir.path());
    assert_eq!(
        timeline.written_at(1_999).expect("read"),
        Some(entry(1_999).ts)
    );
    let before = timeline.bytes_read();

    append(&mut mgr, 2_000, 2_100);
    assert_eq!(
        timeline.written_at(2_050).expect("read"),
        Some(entry(2_050).ts)
    );
    let cost = timeline.bytes_read() - before;
    // A hundred kilobyte frames, plus the read from the mark before 2050.
    assert!(
        cost <= 100 * (VALUE_BYTES as u64 + 64) + 3 * MARK_STRIDE,
        "read {cost}"
    );
}

/// Released positions free the segments wholly below them, and a purged
/// segment no longer answers.
#[test]
fn released_and_purged_segments_are_forgotten() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());
    append(&mut mgr, 0, 100);
    mgr.rotate().expect("seal");
    append(&mut mgr, 100, 200);
    mgr.rotate().expect("seal");
    append(&mut mgr, 200, 300);

    let timeline = timeline(dir.path());
    for index in [50, 150, 250] {
        timeline.written_at(index).expect("read");
    }
    assert_eq!(timeline.inner.lock().segments.len(), 3);

    timeline.release_below(150);
    let kept: Vec<u64> = timeline.inner.lock().segments.keys().copied().collect();
    assert_eq!(kept, vec![100, 200], "the segment holding 150 stays");
    assert!(!timeline.inner.lock().answers.contains_key(&50));

    assert_eq!(mgr.purge_before(100).expect("purge"), 1);
    assert_eq!(timeline.written_at(50).expect("read"), None, "purged");
    assert_eq!(timeline.written_at(150).expect("read"), Some(entry(150).ts));

    timeline.release_below(u64::MAX);
    assert!(timeline.inner.lock().answers.is_empty());
}
