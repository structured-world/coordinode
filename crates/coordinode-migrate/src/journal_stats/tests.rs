use coordinode_core::graph::node::NodeRecord;
use coordinode_core::graph::types::Value;
use coordinode_core::txn::frame::encode_proposal;
use coordinode_core::txn::proposal::{Mutation, PartitionId, ProposalId, RaftProposal};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::oplog::{OplogEntry, OplogOp};

use super::journal_stats;
use crate::test_support::write_segment;

fn node(status: i64, blob: &[u8]) -> Vec<u8> {
    let mut record = NodeRecord::new("State");
    record.set(1, Value::Int(status));
    record.set(2, Value::Blob(blob.to_vec()));
    record.to_msgpack().expect("encode")
}

fn record(index: u64, key: &[u8], value: Vec<u8>) -> OplogEntry {
    let proposal = RaftProposal {
        id: ProposalId::from_raw(index),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.to_vec(),
            value,
        }],
        commit_ts: Timestamp::from_raw(1000 + index),
        start_ts: Timestamp::from_raw(999 + index),
        bypass_rate_limiter: false,
    };
    OplogEntry {
        ts: 1000 + index,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Unit {
            frame: encode_proposal(&proposal).expect("frame"),
        }],
        is_migration: false,
        pre_images: None,
    }
}

/// Three writes of one node where only a small field changes: the class
/// counts three writes of one key, the rewrites are two, and what changed
/// is the small field alone, not the large one rewritten with it. Copies of
/// the journal inside checkpoints are not counted.
#[test]
fn rewrites_are_measured_against_what_changed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = dir.path();
    let blob = vec![7u8; 4096];
    let key = b"node:\x00\x01:\x00\x00\x00\x00\x00\x00\x00\x01";
    let records = [
        record(1, key, node(1, &blob)),
        record(2, key, node(2, &blob)),
        record(3, key, node(3, &blob)),
    ];
    write_segment(
        &data.join("oplog/0/oplog-00000000000000000001.bin"),
        &records,
        true,
    );
    write_segment(
        &data.join("checkpoints/ckpt-1/oplog/0/oplog-00000000000000000001.bin"),
        &records,
        true,
    );

    let stats = journal_stats(data, 5).expect("stats");
    assert_eq!((stats.records, stats.proposals), (3, 3));
    let (class, c) = &stats.classes[0];
    assert!(class.starts_with("put Node node:"), "{class}");
    assert_eq!((c.writes, c.distinct_keys, c.max_per_key), (3, 1, 3));

    let (label, r) = &stats.node_rewrites[0];
    assert_eq!(label, "State");
    assert_eq!(r.rewrites, 2);
    let field = |name: &str| {
        r.fields
            .iter()
            .find(|(f, _, _)| f == name)
            .expect("field")
            .clone()
    };
    let (_, blob_changes, blob_size) = field("#2");
    let (_, status_changes, status_size) = field("#1");
    assert_eq!(blob_changes, 0, "the blob never changed");
    assert!(blob_size > 4096);
    assert_eq!(status_changes, 2);
    assert_eq!(r.changed_bytes, 2 * status_size);
    assert!(r.bytes > 2 * 4096, "the whole record was written each time");
}
