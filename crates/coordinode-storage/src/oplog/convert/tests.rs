use coordinode_core::txn::proposal::{Mutation, PartitionId};

use super::{mutation_to_op, mutations_to_ops};
use crate::engine::partition::Partition;
use crate::oplog::entry::OplogOp;
use crate::placement::partition_wire_tag;

#[test]
fn put_maps_to_insert_with_partition_tag() {
    let m = Mutation::Put {
        partition: PartitionId::Node,
        key: b"k".to_vec(),
        value: b"v".to_vec(),
    };
    match mutation_to_op(&m) {
        OplogOp::Insert {
            partition,
            key,
            value,
        } => {
            assert_eq!(partition, partition_wire_tag(Partition::Node));
            assert_eq!(key, b"k");
            assert_eq!(value, b"v");
        }
        other => panic!("expected Insert, got {other:?}"),
    }
}

#[test]
fn delete_maps_to_delete() {
    let m = Mutation::Delete {
        partition: PartitionId::Adj,
        key: b"e".to_vec(),
    };
    match mutation_to_op(&m) {
        OplogOp::Delete { partition, key } => {
            assert_eq!(partition, partition_wire_tag(Partition::Adj));
            assert_eq!(key, b"e");
        }
        other => panic!("expected Delete, got {other:?}"),
    }
}

#[test]
fn merge_maps_to_merge() {
    let m = Mutation::Merge {
        partition: PartitionId::Adj,
        key: b"src".to_vec(),
        operand: b"op".to_vec(),
    };
    match mutation_to_op(&m) {
        OplogOp::Merge {
            partition,
            key,
            operand,
        } => {
            assert_eq!(partition, partition_wire_tag(Partition::Adj));
            assert_eq!(key, b"src");
            assert_eq!(operand, b"op");
        }
        other => panic!("expected Merge, got {other:?}"),
    }
}

#[test]
fn remove_range_maps_to_remove_range() {
    let m = Mutation::RemoveRange {
        partition: PartitionId::Idx,
        start: b"a".to_vec(),
        end: b"z".to_vec(),
    };
    match mutation_to_op(&m) {
        OplogOp::RemoveRange {
            partition,
            start,
            end,
        } => {
            assert_eq!(partition, partition_wire_tag(Partition::Idx));
            assert_eq!(start, b"a");
            assert_eq!(end, b"z");
        }
        other => panic!("expected RemoveRange, got {other:?}"),
    }
}

#[test]
fn batch_preserves_order_and_count() {
    let muts = vec![
        Mutation::Put {
            partition: PartitionId::Node,
            key: b"1".to_vec(),
            value: b"a".to_vec(),
        },
        Mutation::Delete {
            partition: PartitionId::Node,
            key: b"2".to_vec(),
        },
    ];
    let ops = mutations_to_ops(&muts);
    assert_eq!(ops.len(), 2);
    assert!(matches!(ops[0], OplogOp::Insert { .. }));
    assert!(matches!(ops[1], OplogOp::Delete { .. }));
}

#[test]
fn tag_round_trips_through_inverse() {
    // Every partition a mutation can target must round-trip tag→partition so the
    // replay/repair side routes the op back to the correct tree.
    for pid in [
        PartitionId::Node,
        PartitionId::Adj,
        PartitionId::EdgeProp,
        PartitionId::Blob,
        PartitionId::BlobRef,
        PartitionId::Schema,
        PartitionId::Idx,
        PartitionId::Counter,
        PartitionId::VectorF32,
        PartitionId::Registry,
    ] {
        let part = Partition::from(pid);
        let tag = partition_wire_tag(part);
        assert_eq!(
            crate::placement::partition_from_wire_tag(tag),
            Some(part),
            "tag {tag} did not round-trip for {part:?}"
        );
    }
}
