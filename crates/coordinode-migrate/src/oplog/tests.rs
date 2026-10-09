use super::{read, segment_files};
use crate::test_support::{current_entry, record, write_segment};
use coordinode_raft::storage::Request;

fn current_record(index: u64) -> coordinode_storage::oplog::OplogEntry {
    record(
        index,
        rmp_serde::to_vec(&current_entry(index, Request::closing(0))).expect("encode"),
    )
}

/// The store's journal on every endpoint and the copies inside checkpoints
/// are found; earlier backups and files of other names are not.
#[test]
fn segments_are_found_everywhere_but_the_backup() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path();
    let wanted = [
        data.join("oplog/0/oplog-00000000000000000001.bin"),
        data.join("hot/oplog/0/oplog-00000000000000000005.bin"),
        data.join("checkpoints/ckpt-1/oplog/0/oplog-00000000000000000001.bin"),
    ];
    for path in &wanted {
        write_segment(path, &[current_record(1)], true);
    }
    write_segment(
        &data.join("migrate-backup/1/oplog/0/oplog-00000000000000000001.bin"),
        &[current_record(1)],
        true,
    );
    for other in [
        "oplog/0/oplog-.bin",
        "oplog/0/oplog-12.bin.migrating",
        "oplog/0/LOCK",
    ] {
        std::fs::write(data.join(other), b"x").expect("write");
    }

    let mut expected = wanted.to_vec();
    expected.sort();
    assert_eq!(segment_files(data).expect("list"), expected);
}

/// A sealed segment and one a crash left open both read, the second with
/// the complete entries it holds and marked unsealed.
#[test]
fn sealed_and_crash_left_segments_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    let sealed = dir.path().join("oplog-00000000000000000001.bin");
    let open = dir.path().join("oplog-00000000000000000003.bin");
    write_segment(&sealed, &[current_record(1), current_record(2)], true);
    write_segment(&open, &[current_record(3)], false);

    let s = read(&sealed).expect("sealed");
    assert!(s.sealed);
    assert_eq!((s.first_index, s.entries.len()), (1, 2));
    let o = read(&open).expect("open");
    assert!(!o.sealed);
    assert_eq!((o.first_index, o.entries.len()), (3, 1));
}

/// A file that is not a segment is an error naming it.
#[test]
fn a_file_that_is_not_a_segment_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("oplog-00000000000000000001.bin");
    std::fs::write(&path, b"not a segment at all").expect("write");
    let err = read(&path).err().expect("refused");
    assert!(
        format!("{err:#}").contains("oplog-00000000000000000001.bin"),
        "{err:#}"
    );
}
