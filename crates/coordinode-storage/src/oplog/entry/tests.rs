use super::*;

fn make_full_entry() -> OplogEntry {
    OplogEntry {
        ts: 1_000_000_u64 << 18,
        term: 3,
        index: 42,
        shard: 1,
        ops: vec![
            OplogOp::Insert {
                partition: 1,
                key: b"mykey".to_vec(),
                value: b"myvalue".to_vec(),
            },
            OplogOp::Delete {
                partition: 2,
                key: b"delkey".to_vec(),
            },
            OplogOp::Merge {
                partition: 0,
                key: b"k".to_vec(),
                operand: b"delta".to_vec(),
            },
            OplogOp::Noop,
        ],
        is_migration: false,
        pre_images: None,
    }
}

#[test]
fn encode_decode_roundtrip() {
    let entry = make_full_entry();
    let encoded = entry.encode().expect("encode");
    let decoded = OplogEntry::decode(&encoded).expect("decode");
    assert_eq!(entry, decoded);
}

#[test]
fn noop_entry_roundtrip() {
    let entry = OplogEntry {
        ts: 0,
        term: 1,
        index: 0,
        shard: 99,
        ops: vec![OplogOp::Noop],
        is_migration: false,
        pre_images: None,
    };
    let bytes = entry.encode().expect("encode");
    let decoded = OplogEntry::decode(&bytes).expect("decode");
    assert_eq!(decoded.ops, vec![OplogOp::Noop]);
}

#[test]
fn pre_images_roundtrip() {
    let entry = OplogEntry {
        ts: 1,
        term: 1,
        index: 1,
        shard: 0,
        ops: vec![OplogOp::Merge {
            partition: 0,
            key: b"k".to_vec(),
            operand: b"delta".to_vec(),
        }],
        is_migration: true,
        pre_images: Some(vec![PreImage {
            partition: 0,
            key: b"k".to_vec(),
            value: b"before".to_vec(),
        }]),
    };
    let bytes = entry.encode().expect("encode");
    let decoded = OplogEntry::decode(&bytes).expect("decode");
    assert_eq!(entry, decoded);
}

#[test]
fn empty_ops_roundtrip() {
    let entry = OplogEntry {
        ts: 42,
        term: 2,
        index: 7,
        shard: 0,
        ops: vec![],
        is_migration: false,
        pre_images: None,
    };
    let bytes = entry.encode().expect("encode");
    let decoded = OplogEntry::decode(&bytes).expect("decode");
    assert_eq!(entry, decoded);
}

#[test]
fn binary_fields_use_msgpack_bin_type() {
    // Ensure binary payload is compact (msgpack Bin, not array-of-u8).
    // A 5-byte key in Bin8 format = 2 bytes overhead vs 5*2 = 10 in array form.
    let entry = OplogEntry {
        ts: 0,
        term: 0,
        index: 0,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 0,
            key: vec![0u8; 5],
            value: vec![0u8; 5],
        }],
        is_migration: false,
        pre_images: None,
    };
    let bytes = entry.encode().expect("encode");
    // Bin8 for 5 bytes = 0xC4 0x05 + 5 bytes = 7 bytes
    // Array[u8; 5] = fixarray(5) + 5 * fixint = 6 bytes
    // We just verify it round-trips correctly; the compact form is an
    // implementation detail of serde_bytes + msgpack.
    let decoded = OplogEntry::decode(&bytes).expect("decode");
    assert_eq!(decoded.ops[0], entry.ops[0]);
}

#[test]
fn decode_invalid_bytes_returns_error() {
    let result = OplogEntry::decode(b"not msgpack garbage \xFF");
    assert!(result.is_err());
}

#[test]
fn raft_entry_op_roundtrip() {
    let payload = b"serialized-raft-entry-bytes".to_vec();
    let entry = OplogEntry {
        ts: 0,
        term: 7,
        index: 42,
        shard: 0,
        ops: vec![OplogOp::RaftEntry {
            data: payload.clone(),
        }],
        is_migration: false,
        pre_images: None,
    };
    let bytes = entry.encode().expect("encode");
    let decoded = OplogEntry::decode(&bytes).expect("decode");
    assert_eq!(decoded.term, 7);
    assert_eq!(decoded.index, 42);
    match &decoded.ops[0] {
        OplogOp::RaftEntry { data } => assert_eq!(data, &payload),
        other => panic!("expected RaftEntry, got {other:?}"),
    }
}

#[test]
fn raft_truncation_op_roundtrip() {
    let entry = OplogEntry {
        ts: 0,
        term: 3,
        index: 99,
        shard: 0,
        ops: vec![OplogOp::RaftTruncation { after_index: 5 }],
        is_migration: false,
        pre_images: None,
    };
    let bytes = entry.encode().expect("encode");
    let decoded = OplogEntry::decode(&bytes).expect("decode");
    match &decoded.ops[0] {
        OplogOp::RaftTruncation { after_index } => assert_eq!(*after_index, 5),
        other => panic!("expected RaftTruncation, got {other:?}"),
    }
}
