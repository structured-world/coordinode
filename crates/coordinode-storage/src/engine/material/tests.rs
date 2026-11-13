use super::*;

/// A record's key sorts by seqno inside its source's scope, apart from the
/// coverage records (`0x00`) and the tree's own records (`0xFE`, `0xFF`), and
/// scopes of different sources or incarnations never interleave.
#[test]
fn records_sort_by_seqno_inside_their_scope() {
    let heads = Heads::default();
    let node = heads.head(Partition::Node);
    assert!(key(node, 1) < key(node, 2));
    assert!(key(node, 255) < key(node, 256));
    assert!(key(node, u64::MAX).as_slice() < node.key(&END).as_slice());
    // Neither a coverage record nor one of the tree's own.
    for &part in Partition::all() {
        let first = heads.head(part).bytes()[0];
        assert!(first != 0x00 && first < 0xFE, "{part:?} tags {first:#x}");
    }
    // The next scope starts past every record of this one.
    let schema = heads.head(Partition::Schema);
    assert!(key(node, u64::MAX).as_slice() < schema.bytes().as_slice());
    // A renewed scope sorts past every record of the one before it.
    let mut renewed = [0u8; HEAD_LEN];
    renewed[0] = node.bytes()[0];
    renewed[1..].copy_from_slice(&1u32.to_be_bytes());
    assert!(key(node, u64::MAX).as_slice() < renewed.as_slice());
    assert_eq!(node.end(), renewed.to_vec());
}

/// A unit's logical size is its key and what it writes; engine bookkeeping
/// counts nothing.
#[test]
fn a_unit_is_its_key_and_what_it_writes() {
    assert_eq!(unit_bytes(b"node:1", b"value"), 11);
    assert_eq!(unit_bytes(b"node:1", &[]), 6);
    assert_eq!(unit_bytes(&[0x00, b'c', b'j', b'b'], b"12345678"), 0);
}

/// The consensus log, the consumer registry and the accounting itself are
/// not source material.
#[test]
fn the_log_the_registry_and_the_accounting_are_not_accounted() {
    assert!(!accounted(Partition::Raft));
    assert!(!accounted(Partition::Registry));
    assert!(!accounted(Partition::Accounting));
    assert!(accounted(Partition::Node));
    assert!(accounted(Partition::Schema));
}

/// The accounting of a unit's mutations follows each source's own order,
/// whatever order the sources come in: the ordinals the live apply and a
/// replay give match.
#[test]
fn tallies_keep_each_sources_order() {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    let unit = vec![
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: b"adj:a".to_vec(),
            operand: b"1".to_vec(),
        },
        Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1".to_vec(),
            value: b"v".to_vec(),
        },
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: b"adj:a".to_vec(),
            operand: b"1".to_vec(),
        },
        Mutation::Put {
            partition: PartitionId::Registry,
            key: b"registry:c".to_vec(),
            value: b"x".to_vec(),
        },
    ];
    let tallies = tally_mutations(&unit);
    let parts: Vec<Partition> = tallies.iter().map(|(p, _)| *p).collect();
    assert_eq!(parts, vec![Partition::Adj, Partition::Node]);
    // Two equal operands on one key stay two units.
    assert_eq!(tallies[0].1.units.len(), 2);
    assert_eq!(tallies[0].1.bytes, 12);
}
