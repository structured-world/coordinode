use coordinode_raft::storage::{Entry, Request};
use coordinode_storage::oplog::{OplogOp, SegmentReader};
use openraft::entry::EntryPayload;

use crate::test_support::{current_entry, old_bytes, proposal, record, write_segment};
use crate::{BACKUP_DIR, apply, survey};

fn envelope(entry: &coordinode_storage::oplog::OplogEntry) -> Entry {
    let Some(OplogOp::RaftEntry { data }) = entry.ops.first() else {
        unreachable!("every record here is a Raft entry")
    };
    rmp_serde::from_slice(data).expect("reads as a current entry")
}

/// A data directory as a crash left it under a build before the closed
/// bound: an old sealed segment, an old segment the crash left open, the
/// same old journal copied into a checkpoint, and a segment already
/// current. A survey changes nothing; applying rewrites exactly the old
/// segments, keeps every original in the backup, leaves the current one
/// byte for byte, and a second run finds nothing.
#[test]
fn a_crashed_directory_of_the_old_shape_is_brought_current() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path();
    let old = |i: u64| {
        record(
            i,
            old_bytes(&current_entry(i, Request::single(proposal(i, 100 + i)))),
        )
    };
    let sealed = data.join("oplog/0/oplog-00000000000000000001.bin");
    let crashed = data.join("oplog/0/oplog-00000000000000000003.bin");
    let copied = data.join("checkpoints/ckpt-1/oplog/0/oplog-00000000000000000001.bin");
    let current = data.join("oplog/1/oplog-00000000000000000001.bin");
    write_segment(&sealed, &[old(1), old(2)], true);
    write_segment(&crashed, &[old(3)], false);
    write_segment(&copied, &[old(1), old(2)], true);
    let current_bytes = rmp_serde::to_vec(&current_entry(1, Request::closing(7))).expect("encode");
    write_segment(&current, &[record(1, current_bytes)], true);
    let before = |p: &std::path::Path| std::fs::read(p).expect("read");
    let (sealed_before, crashed_before, current_before) =
        (before(&sealed), before(&crashed), before(&current));

    let surveyed = survey(data).expect("survey");
    let mut paths: Vec<_> = surveyed
        .iter()
        .flat_map(|(_, f)| f.iter().map(|f| f.path.clone()))
        .collect();
    paths.sort();
    let mut expected = vec![sealed.clone(), crashed.clone(), copied.clone()];
    expected.sort();
    assert_eq!(paths, expected);
    assert_eq!(before(&sealed), sealed_before, "a survey changes nothing");
    assert!(!data.join(BACKUP_DIR).exists());

    apply(data, "run1").expect("apply");

    for (path, indexes) in [
        (&sealed, vec![1, 2]),
        (&crashed, vec![3]),
        (&copied, vec![1, 2]),
    ] {
        let reader = SegmentReader::open(path).expect("rewritten segments are sealed");
        let got: Vec<u64> = reader.entries().iter().map(|e| e.index).collect();
        assert_eq!(got, indexes, "{}", path.display());
        for e in reader.entries() {
            let EntryPayload::Normal(request) = envelope(e).payload else {
                unreachable!("Normal entries")
            };
            assert_eq!(request.closed_below, 0);
            assert_eq!(request.proposals, vec![proposal(e.index, 100 + e.index)]);
        }
    }
    assert_eq!(
        before(&current),
        current_before,
        "a current segment is untouched"
    );
    let kept = |p: &std::path::Path| {
        before(
            &data
                .join(BACKUP_DIR)
                .join("run1")
                .join(p.strip_prefix(data).expect("inside")),
        )
    };
    assert_eq!(kept(&sealed), sealed_before);
    assert_eq!(kept(&crashed), crashed_before);

    let again = survey(data).expect("survey");
    assert!(
        again.iter().all(|(_, f)| f.is_empty()),
        "a second run finds nothing: {again:?}",
        again = again.iter().map(|(_, f)| f.clone()).collect::<Vec<_>>()
    );
}

/// Every directory with an engine format marker is a store to migrate,
/// wherever the server keeps it: the data directory, its checkpoints, and
/// the capture a Raft snapshot is served from. A directory without the
/// marker and the run's own backup are not stores.
#[test]
fn every_marked_store_under_the_data_directory_is_found() {
    use coordinode_storage::format::MARKER_FILE;

    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path();
    let mark = |p: &std::path::Path| {
        std::fs::create_dir_all(p).expect("dir");
        std::fs::write(p.join(MARKER_FILE), b"x").expect("marker");
    };
    mark(data);
    mark(&data.join("checkpoints/ckpt-1"));
    mark(&data.join("raft-snapshot/0001-0010.capt"));
    mark(&data.join(BACKUP_DIR).join("run1"));
    std::fs::create_dir_all(data.join("snapshot-capture")).expect("dir");
    std::fs::create_dir_all(data.join("node/tables")).expect("dir");

    let mut found = crate::stores(data).expect("stores");
    found.sort();
    let mut expected = vec![
        data.to_path_buf(),
        data.join("checkpoints/ckpt-1"),
        data.join("raft-snapshot/0001-0010.capt"),
    ];
    expected.sort();
    assert_eq!(found, expected);
}

/// A segment holding an entry of an unknown shape stops the run before
/// anything is written, naming the segment and the entry.
#[test]
fn an_unknown_entry_stops_the_run_before_any_write() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path();
    let path = data.join("oplog/0/oplog-00000000000000000001.bin");
    write_segment(&path, &[record(1, vec![0xc1])], true);
    let before = std::fs::read(&path).expect("read");

    let err = apply(data, "run1").err().expect("refused");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("log entry 1") && msg.contains("oplog-00000000000000000001.bin"),
        "{msg}"
    );
    assert_eq!(std::fs::read(&path).expect("read"), before);
    assert!(!data.join(BACKUP_DIR).exists());
}
