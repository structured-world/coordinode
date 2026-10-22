use std::io::Read;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

fn open(dir: &Path) -> StorageEngine {
    StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open")
}

/// One partition's rows, key and value.
type Rows = Vec<(Vec<u8>, Vec<u8>)>;

/// Every replicated partition's rows, as the receiver's install reads them.
fn rows(engine: &StorageEngine) -> Vec<(Partition, Rows)> {
    Partition::all()
        .iter()
        .copied()
        .filter(|p| *p != Partition::Raft)
        .map(|p| (p, engine.replicated_rows(p).expect("rows")))
        .collect()
}

fn stream(transfer: &mut StoreTransfer) -> Vec<u8> {
    let mut bytes = Vec::new();
    transfer.read_to_end(&mut bytes).expect("read the transfer");
    bytes
}

/// A store with flushed rows, rows still in the memtable, merge operands, a
/// range delete and a value large enough to live in a blob file.
fn populated(engine: &StorageEngine) {
    for i in 0..200u32 {
        engine
            .put(Partition::Node, format!("node:1:{i:05}").as_bytes(), b"v")
            .expect("put");
    }
    engine
        .merge(
            Partition::Counter,
            b"counter:degree:1",
            &crate::engine::merge::encode_counter_delta(3),
        )
        .expect("merge");
    engine
        .put(Partition::Blob, b"blob:1", &vec![7u8; 64 * 1024])
        .expect("blob");
    engine.persist().expect("persist");
    for i in 200..260u32 {
        engine
            .put(Partition::Node, format!("node:1:{i:05}").as_bytes(), b"w")
            .expect("put");
    }
    engine
        .remove_range(Partition::Node, b"node:1:00010", b"node:1:00020")
        .expect("remove range");
    engine
        .merge(
            Partition::Counter,
            b"counter:degree:1",
            &crate::engine::merge::encode_counter_delta(4),
        )
        .expect("merge");
}

/// The received store reads as the source did at the pin, partition by
/// partition, whether a row came from a pinned file or a pinned memtable, and
/// the stream is exactly as long as it said.
#[test]
fn a_received_store_reads_as_the_source_at_the_pin() {
    let source_dir = tempfile::tempdir().expect("tempdir");
    let source = open(source_dir.path());
    populated(&source);
    let expected = rows(&source);

    let mut transfer = source.pin_store().expect("pin");
    let bytes = stream(&mut transfer);
    assert_eq!(bytes.len() as u64, transfer.len());

    let target = tempfile::tempdir().expect("tempdir");
    let dir = target.path().join("received");
    let received = receive_store(&mut bytes.as_slice(), &dir).expect("receive");
    assert!(received.positions().is_empty());
    let copy = StorageEngine::open_checkpoint(received.dir()).expect("open the received store");
    assert_eq!(rows(&copy), expected);
    drop(copy);
    drop(received);
    assert!(!dir.exists(), "the received store goes when dropped");
}

/// A pin keeps what it pinned: writes and compactions after it change the
/// source, never the stream.
#[test]
fn a_transfer_holds_the_store_as_it_was_pinned() {
    let source_dir = tempfile::tempdir().expect("tempdir");
    let source = open(source_dir.path());
    populated(&source);
    let expected = rows(&source);
    let mut transfer = source.pin_store().expect("pin");

    for i in 0..100u32 {
        source
            .put(
                Partition::Node,
                format!("node:1:{i:05}").as_bytes(),
                b"later",
            )
            .expect("put");
    }
    source.persist().expect("persist");
    source.major_compact(Partition::Node).expect("compact");
    source.major_compact(Partition::Counter).expect("compact");

    let bytes = stream(&mut transfer);
    let target = tempfile::tempdir().expect("tempdir");
    let received = receive_store(&mut bytes.as_slice(), &target.path().join("r")).expect("receive");
    let copy = StorageEngine::open_checkpoint(received.dir()).expect("open");
    assert_eq!(rows(&copy), expected);
}

/// A stream damaged in transit installs nothing: a flipped byte fails the
/// checksum, a cut one ends early, and trailing bytes are refused. Each
/// leaves no directory behind.
#[test]
fn a_damaged_stream_is_refused_and_leaves_nothing() {
    let source_dir = tempfile::tempdir().expect("tempdir");
    let source = open(source_dir.path());
    populated(&source);
    let bytes = stream(&mut source.pin_store().expect("pin"));
    let target = tempfile::tempdir().expect("tempdir");

    let mut flipped = bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0x40;
    let mut cut = bytes.clone();
    cut.truncate(bytes.len() - 3);
    let mut trailing = bytes.clone();
    trailing.push(0);

    for (name, damaged) in [("flipped", flipped), ("cut", cut), ("trailing", trailing)] {
        let dir = target.path().join(name);
        let err = receive_store(&mut damaged.as_slice(), &dir).expect_err(name);
        assert!(
            matches!(err, StorageError::Serialization(_)),
            "{name}: {err}"
        );
        assert!(!dir.exists(), "{name}: the directory is removed");
    }
}

/// A table name in the stream places files under the received directory
/// only: one that names another place is refused.
#[test]
fn a_table_name_cannot_leave_the_received_directory() {
    for name in [
        "tables/",
        "tables/..",
        "tables/.",
        "tables/a/b",
        "tables/a\\b",
    ] {
        assert!(TreeName::decode(name).is_err(), "{name}");
    }
    assert!(
        TreeName::decode("raft").is_err(),
        "the Raft partition never travels"
    );
    assert_eq!(
        TreeName::decode("tables/t1").expect("table"),
        TreeName::Table("t1".into())
    );
    assert_eq!(
        TreeName::decode("node").expect("partition"),
        TreeName::Partition(Partition::Node)
    );
}
