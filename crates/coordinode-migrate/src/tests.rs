use std::path::Path;

use coordinode_core::txn::frame::decode_unit;
use coordinode_core::txn::proposal::Mutation;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::format::MARKER_FILE;
use coordinode_storage::oplog::{OplogOp, SegmentReader};

use crate::test_support::{
    NAMESPACE, journal_record, named_frame, raft_record, unnamed, write_segment,
};
use crate::{BACKUP_DIR, apply, survey};

/// A store as 0.10.1 laid its trees out under `dir`: its marker and a tree
/// of every partition, and the tables, at the top, each holding one file.
/// It has a Deployment tree already, so the step that moves the
/// deployment's records into one, which opens the store, passes it over:
/// these trees are not ones an engine opens. That step runs on a store
/// 0.10.1 wrote, in `tests/from_0_10_1.rs`.
fn flat_store(dir: &Path) {
    std::fs::create_dir_all(dir.join("system/deployment")).expect("dir");
    std::fs::write(dir.join(MARKER_FILE), b"5").expect("marker");
    for name in Partition::all()
        .iter()
        .filter(|part| part.name() != "deployment")
        .map(|part| part.name())
        .chain(["tables"])
    {
        std::fs::create_dir_all(dir.join(name)).expect("tree");
        std::fs::write(dir.join(name).join("MANIFEST"), name).expect("tree file");
    }
}

/// The operations of every unit of the segment at `path`.
fn units(path: &Path) -> Vec<Vec<Mutation>> {
    SegmentReader::open(path)
        .expect("rewritten segments are sealed")
        .entries()
        .iter()
        .flat_map(|e| e.ops.iter())
        .filter_map(|op| match op {
            OplogOp::Unit { frame } => Some(decode_unit(frame).expect("the unit reads")),
            _ => None,
        })
        .collect()
}

/// A data directory 0.10.1 left, a crash among it: a store and a
/// checkpoint of it with their trees at the top, the store's journal with a
/// sealed segment and one a crash left open, a group's Raft log, and a
/// segment this build wrote already. A survey changes nothing; applying
/// moves every tree into its set, gives every unit the namespace marker,
/// keeps every replaced segment in the backup and leaves the current one
/// byte for byte; a second run finds nothing.
#[test]
fn a_directory_0_10_1_left_is_brought_current() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path();
    let checkpoint = data.join("checkpoints/ckpt-1");
    flat_store(data);
    flat_store(&checkpoint);
    let old = |i: u64| unnamed(&named_frame(i, 100 + i));
    let sealed = data.join("oplog/0/oplog-00000000000000000001.bin");
    let crashed = data.join("oplog/0/oplog-00000000000000000003.bin");
    let raft = data.join("oplog/1/oplog-00000000000000000001.bin");
    let current = data.join("oplog/2/oplog-00000000000000000001.bin");
    write_segment(
        &sealed,
        &[journal_record(1, old(1)), journal_record(2, old(2))],
        true,
    );
    write_segment(&crashed, &[journal_record(3, old(3))], false);
    write_segment(&raft, &[raft_record(1, old(1))], true);
    write_segment(&current, &[journal_record(1, named_frame(1, 101))], true);
    let read = |p: &Path| std::fs::read(p).expect("read");
    let (sealed_before, current_before) = (read(&sealed), read(&current));

    let surveyed = survey(data).expect("survey");
    let found: Vec<(&str, std::path::PathBuf)> = surveyed
        .iter()
        .flat_map(|(m, f)| f.iter().map(|f| (m.name, f.path.clone())))
        .collect();
    let mut stores: Vec<_> = found
        .iter()
        .filter(|(m, _)| *m == "tree-sets")
        .map(|(_, p)| p.clone())
        .collect();
    stores.sort();
    let mut expected = vec![data.to_path_buf(), checkpoint.clone()];
    expected.sort();
    assert_eq!(stores, expected);
    let mut segments: Vec<_> = found
        .iter()
        .filter(|(m, _)| *m == "unit-namespace")
        .map(|(_, p)| p.clone())
        .collect();
    segments.sort();
    let mut expected = vec![sealed.clone(), crashed.clone(), raft.clone()];
    expected.sort();
    assert_eq!(segments, expected);
    assert_eq!(read(&sealed), sealed_before, "a survey changes nothing");
    assert!(data.join("node").is_dir(), "a survey moves nothing");

    apply(data, "run1").expect("apply");

    for store in [data, checkpoint.as_path()] {
        for part in Partition::all() {
            let top = store.join(part.name());
            assert!(!top.exists(), "{} left at the top", top.display());
        }
        for (tree, name) in [
            ("system/raft", "raft"),
            ("system/registry", "registry"),
            ("ns/0000000000000001/node", "node"),
            ("ns/0000000000000001/schema", "schema"),
            ("ns/0000000000000001/tables", "tables"),
        ] {
            assert_eq!(
                read(&store.join(tree).join("MANIFEST")),
                name.as_bytes(),
                "{tree} moved whole in {}",
                store.display()
            );
        }
    }
    for path in [&sealed, &crashed, &raft] {
        for unit in units(path) {
            assert_eq!(
                unit.first(),
                Some(&Mutation::Namespace(NAMESPACE)),
                "{}",
                path.display()
            );
        }
    }
    assert_eq!(units(&sealed).len(), 2);
    assert_eq!(
        read(&current),
        current_before,
        "a current segment is untouched"
    );
    let kept = data
        .join(BACKUP_DIR)
        .join("run1")
        .join(sealed.strip_prefix(data).expect("inside"));
    assert_eq!(read(&kept), sealed_before);

    let again = survey(data).expect("survey");
    assert!(
        again.iter().all(|(_, f)| f.is_empty()),
        "a second run finds nothing: {:?}",
        again.iter().map(|(_, f)| f.clone()).collect::<Vec<_>>()
    );
}

/// Every directory with an engine format marker is a store to migrate,
/// wherever the server keeps it: the data directory, its checkpoints, and
/// the capture a Raft snapshot is served from. A directory without the
/// marker and the run's own backup are not stores.
#[test]
fn every_marked_store_under_the_data_directory_is_found() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path();
    let mark = |p: &Path| {
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

/// A unit that reads neither now nor once named stops the run before
/// anything is written or moved, naming the segment and the entry.
#[test]
fn an_unknown_unit_stops_the_run_before_any_write() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path();
    flat_store(data);
    let path = data.join("oplog/0/oplog-00000000000000000001.bin");
    write_segment(&path, &[journal_record(1, vec![0xc1, 0, 0, 0, 0])], true);
    let before = std::fs::read(&path).expect("read");

    let err = apply(data, "run1").err().expect("refused");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("log entry 1") && msg.contains("oplog-00000000000000000001.bin"),
        "{msg}"
    );
    assert_eq!(std::fs::read(&path).expect("read"), before);
    assert!(data.join("node").is_dir(), "no tree moved");
    assert!(!data.join(BACKUP_DIR).exists());
}

/// A file kept again under the same run (a run redone after it stopped
/// between keeping a file and replacing it) leaves both the file and its
/// backup as they were: the backup already names the file's own bytes, and
/// a copy onto that name would truncate the file through the shared link.
#[test]
fn keeping_a_file_twice_leaves_the_file_and_its_backup_intact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path();
    let path = data.join("oplog/0/oplog-00000000000000000001.bin");
    write_segment(&path, &[journal_record(1, named_frame(1, 101))], true);
    let before = std::fs::read(&path).expect("read");
    let backup = crate::Backup::new(data, "run1");
    let kept = backup.keep(&path).expect("keep");
    backup.keep(&path).expect("keep again");
    assert_eq!(
        std::fs::read(&path).expect("read"),
        before,
        "the file changed"
    );
    assert_eq!(
        std::fs::read(&kept).expect("read"),
        before,
        "the backup changed"
    );
}
