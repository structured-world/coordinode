use super::*;

/// A record's key sorts by seqno and stays inside the engine-reserved
/// namespace, below every user key and apart from the coverage records.
#[test]
fn records_sort_by_seqno_in_the_reserved_namespace() {
    assert!(key(1) < key(2));
    assert!(key(255) < key(256));
    assert!(coverage::is_reserved(&key(u64::MAX)));
    assert!(key(u64::MAX).as_slice() < END.as_slice());
    assert!(END.as_slice() < coverage::USER_KEYSPACE_START);
}

/// A unit's logical size is its key and what it writes; engine bookkeeping
/// counts nothing.
#[test]
fn a_unit_is_its_key_and_what_it_writes() {
    assert_eq!(unit_bytes(b"node:1", b"value"), 11);
    assert_eq!(unit_bytes(b"node:1", &[]), 6);
    assert_eq!(unit_bytes(&key(7), b"12345678"), 0);
}

/// The consensus log and the consumer registry are not source material.
#[test]
fn the_log_and_the_registry_are_not_accounted() {
    assert!(!accounted(Partition::Raft));
    assert!(!accounted(Partition::Registry));
    assert!(accounted(Partition::Node));
    assert!(accounted(Partition::Schema));
}
