use super::*;

fn version<'a>(key: &'a [u8], value: &'a [u8], seqno: u64) -> StoredVersion<'a> {
    StoredVersion {
        key,
        value,
        seqno,
        operand: false,
    }
}

fn split(stored: &[u8]) -> (&[u8], u64) {
    let (payload, stamp) = stored.split_at(stored.len() - 8);
    (
        payload,
        u64::from_le_bytes(stamp.try_into().expect("eight bytes")),
    )
}

/// A version the journal holds the write of, at its timestamp with its key
/// and bytes, keeps that commit; any other version, whatever its sequence
/// number or the bytes it ends in, gets the unknown revision, its payload
/// byte for byte.
#[test]
fn a_version_keeps_only_the_commit_the_journal_proves() {
    let mut known = Writes::new();
    known.insert((Partition::Node, b"node:1".to_vec(), 70), b"v".to_vec());

    let proved = stamped(Partition::Node, version(b"node:1", b"v", 70), &known);
    assert_eq!(split(&proved), (&b"v"[..], 70));

    // Another value at that timestamp, the same value at another one.
    for (value, seqno) in [(&b"w"[..], 70), (&b"v"[..], 71)] {
        let stored = stamped(Partition::Node, version(b"node:1", value, seqno), &known);
        assert_eq!(split(&stored), (value, UNKNOWN));
    }

    // An installed row at a sequence number of its own, its last eight
    // bytes anything: its payload kept whole, its revision unknown.
    let long = b"a long payload ending in eight bytes 12345678";
    let stored = stamped(Partition::Schema, version(b"schema:x", long, 9_000), &known);
    assert_eq!(split(&stored), (&long[..], UNKNOWN));

    // A settled row: no commit is ever at zero.
    let mut settled = Writes::new();
    settled.insert((Partition::Node, b"node:2".to_vec(), 0), b"v".to_vec());
    let stored = stamped(Partition::Node, version(b"node:2", b"v", 0), &settled);
    assert_eq!(split(&stored), (&b"v"[..], UNKNOWN));
}

fn config(store: &Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        store,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn open_tree(store: &Path) -> lsm_tree::AnyTree {
    config(store)
        .tree_config_at(
            Partition::Node,
            Path::new("node"),
            lsm_tree::SharedSequenceNumberGenerator::from(lsm_tree::SequenceNumberCounter::new(
                100,
            )),
            None,
        )
        .open()
        .expect("open")
}

/// A store with a 0.10.1 node tree at its top: one value, unstamped.
fn store_with_a_tree() -> tempfile::TempDir {
    use lsm_tree::AbstractTree as _;
    let dir = tempfile::tempdir().expect("tempdir");
    let tree = open_tree(dir.path());
    tree.insert("node:1", "value", 7);
    let lock = tree.get_flush_lock();
    tree.rotate_memtable();
    tree.flush(&lock, 0).expect("flush");
    dir
}

/// The value of `node:1` the store's node tree holds, as stored.
fn stored_value(store: &Path) -> Vec<u8> {
    use lsm_tree::AbstractTree as _;
    open_tree(store)
        .get("node:1", u64::MAX)
        .expect("read")
        .expect("there")
        .to_vec()
}

/// A run cut short after the written tree is in place, before or after it
/// is recorded, ends on the next run with the tree stamped once, byte for
/// byte what an uncut run leaves, and nothing aside.
#[test]
fn a_run_cut_short_anywhere_stamps_each_value_once() {
    let clean = store_with_a_tree();
    let backup = Backup::new(clean.path(), "run1");
    stamp_tree(
        clean.path(),
        &config(clean.path()),
        Partition::Node,
        &Writes::new(),
        &backup,
    )
    .expect("an uncut run");
    let expected = stored_value(clean.path());
    assert_eq!(split(&expected), (&b"value"[..], UNKNOWN));

    for at in [Cut::AfterRenames, Cut::AfterRecord] {
        let store = store_with_a_tree();
        let backup = Backup::new(store.path(), "run1");
        CUT.with(|cut| cut.set(Some(at)));
        stamp_tree(
            store.path(),
            &config(store.path()),
            Partition::Node,
            &Writes::new(),
            &backup,
        )
        .expect_err("cut short");
        // The next run sees what a restart there sees.
        for part in pending(store.path()).expect("pending") {
            stamp_tree(
                store.path(),
                &config(store.path()),
                part,
                &Writes::new(),
                &backup,
            )
            .expect("the run goes on");
        }
        assert_eq!(stored_value(store.path()), expected, "{at:?}");
        assert!(pending(store.path()).expect("pending").is_empty(), "{at:?}");
        assert!(
            !store.path().join("node.replaced").exists(),
            "{at:?}: the old tree is gone"
        );
    }
}

/// A unit frame putting `value` at `key` of the node partition at `commit`.
fn unit(key: &[u8], value: &[u8], commit: u64) -> Vec<u8> {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    coordinode_core::txn::frame::encode_unit(
        &coordinode_core::txn::proposal::in_namespace(
            crate::test_support::NAMESPACE,
            [Mutation::Put {
                partition: PartitionId::Node,
                key: key.to_vec(),
                value: value.to_vec(),
            }],
        ),
        coordinode_core::txn::timestamp::Timestamp::from_raw(commit),
    )
    .expect("encode")
}

/// Each write is known at the commit its own unit names, not at the
/// timestamp of the log entry carrying several units; an operation that is
/// not a put or a merge proves nothing; and the journal of a checkpoint
/// inside the store is not this store's history.
#[test]
fn each_write_is_known_at_its_own_units_commit() {
    use coordinode_storage::oplog::{OplogEntry, OplogOp};
    let dir = tempfile::tempdir().expect("tempdir");
    let store = dir.path();
    let deleted = coordinode_core::txn::frame::encode_unit(
        &coordinode_core::txn::proposal::in_namespace(
            crate::test_support::NAMESPACE,
            [coordinode_core::txn::proposal::Mutation::Delete {
                partition: coordinode_core::txn::proposal::PartitionId::Node,
                key: b"node:d".to_vec(),
            }],
        ),
        coordinode_core::txn::timestamp::Timestamp::from_raw(650),
    )
    .expect("encode");
    // Two units in one entry, the entry stamped with the later commit.
    let entry = OplogEntry {
        ts: 600,
        term: 1,
        index: 1,
        shard: 0,
        ops: vec![
            OplogOp::Unit {
                frame: unit(b"node:a", b"x", 500),
            },
            OplogOp::Unit {
                frame: unit(b"node:b", b"y", 600),
            },
            OplogOp::Unit { frame: deleted },
        ],
        is_migration: false,
        pre_images: None,
    };
    crate::test_support::write_segment(
        &store.join("oplog/0/oplog-00000000000000000001.bin"),
        &[entry],
        true,
    );
    let checkpoint = store.join("checkpoints/ckpt-1");
    std::fs::create_dir_all(&checkpoint).expect("dir");
    std::fs::write(
        checkpoint.join(coordinode_storage::format::MARKER_FILE),
        b"5",
    )
    .expect("marker");
    crate::test_support::write_segment(
        &checkpoint.join("oplog/0/oplog-00000000000000000001.bin"),
        &[crate::test_support::journal_record(
            1,
            unit(b"node:c", b"z", 700),
        )],
        true,
    );

    let writes = journal_writes(store).expect("read the journal");
    let at = |key: &[u8], commit: u64| writes.get(&(Partition::Node, key.to_vec(), commit));
    assert_eq!(at(b"node:a", 500), Some(&b"x".to_vec()));
    assert_eq!(at(b"node:a", 600), None, "not at the entry's timestamp");
    assert_eq!(at(b"node:b", 600), Some(&b"y".to_vec()));
    assert_eq!(
        writes.len(),
        2,
        "the delete and the checkpoint's journal prove nothing"
    );
}

/// A key of the engine's reserved namespace carries no revision: it is
/// written as it was.
#[test]
fn a_reserved_key_is_written_as_it_was() {
    let stored = stamped(
        Partition::Node,
        version(&[0x00, b'c'], b"coverage", 5),
        &Writes::new(),
    );
    assert_eq!(stored, b"coverage");
}
