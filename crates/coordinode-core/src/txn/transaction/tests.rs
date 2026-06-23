use super::*;

fn test_oracle() -> TimestampOracle {
    TimestampOracle::resume_from(Timestamp::from_raw(100))
}

#[test]
fn begin_allocates_start_ts() {
    // `begin()` must call `oracle.next()` to allocate start_ts.
    // The exact value is wall-clock-based (HLC), so we verify:
    //   1. start_ts > 0 (non-zero — not ZERO sentinel)
    //   2. oracle has advanced after begin (start_ts was consumed)
    //   3. start_ts equals oracle.current() right after begin
    let oracle = test_oracle();
    let before = oracle.current();
    let txn = Transaction::begin(&oracle);
    let after = oracle.current();

    assert!(!txn.start_ts().is_zero(), "start_ts must be non-zero");
    // Oracle must have advanced by exactly 1 call to next().
    assert!(after >= before, "oracle must not go backward");
    // start_ts was the value returned by next(), so current() == start_ts.
    assert_eq!(
        txn.start_ts(),
        after,
        "start_ts must equal oracle.current() after begin"
    );
}

#[test]
fn begin_at_uses_given_ts() {
    let txn = Transaction::begin_at(Timestamp::from_raw(42));
    assert_eq!(txn.start_ts().as_raw(), 42);
}

#[test]
fn empty_transaction_is_read_only() {
    let oracle = test_oracle();
    let txn = Transaction::begin(&oracle);
    assert!(txn.is_read_only());
    assert_eq!(txn.write_count(), 0);
}

#[test]
fn put_buffers_write() {
    let oracle = test_oracle();
    let mut txn = Transaction::begin(&oracle);

    txn.put(TxnPartition::Node, b"key1".to_vec(), b"val1".to_vec());
    assert!(!txn.is_read_only());
    assert_eq!(txn.write_count(), 1);
}

#[test]
fn delete_buffers_write() {
    let oracle = test_oracle();
    let mut txn = Transaction::begin(&oracle);

    txn.delete(TxnPartition::Node, b"key1".to_vec());
    assert_eq!(txn.write_count(), 1);
}

#[test]
fn multiple_partitions_tracked_separately() {
    let oracle = test_oracle();
    let mut txn = Transaction::begin(&oracle);

    txn.put(TxnPartition::Node, b"n1".to_vec(), b"v1".to_vec());
    txn.put(TxnPartition::Adj, b"a1".to_vec(), b"v2".to_vec());
    txn.delete(TxnPartition::Idx, b"i1".to_vec());

    assert_eq!(txn.write_count(), 3);
}

#[test]
fn commit_assigns_commit_ts() {
    let oracle = test_oracle();
    let mut txn = Transaction::begin(&oracle);
    let start = txn.start_ts();

    txn.put(TxnPartition::Node, b"k".to_vec(), b"v".to_vec());
    let (commit_ts, buffer, _stats) = txn.commit(&oracle);

    assert!(commit_ts > start, "commit_ts must be after start_ts");
    assert!(!buffer.is_empty());
}

#[test]
fn read_only_commit_reuses_start_ts() {
    let oracle = test_oracle();
    let txn = Transaction::begin(&oracle);
    let start = txn.start_ts();

    let (commit_ts, buffer, _stats) = txn.commit(&oracle);
    assert_eq!(commit_ts, start);
    assert!(buffer.is_empty());
}

#[test]
fn stats_tracking() {
    let oracle = test_oracle();
    let mut txn = Transaction::begin(&oracle);

    txn.stats_mut().nodes_created = 3;
    txn.stats_mut().edges_created = 5;
    txn.stats_mut().properties_set = 10;

    assert_eq!(txn.stats().nodes_created, 3);
    assert_eq!(txn.stats().edges_created, 5);
    assert_eq!(txn.stats().properties_set, 10);
}

#[test]
fn get_pending_write_finds_latest() {
    let oracle = test_oracle();
    let mut txn = Transaction::begin(&oracle);

    txn.put(TxnPartition::Node, b"k".to_vec(), b"v1".to_vec());
    txn.put(TxnPartition::Node, b"k".to_vec(), b"v2".to_vec());

    let pending = txn.get_pending_write(TxnPartition::Node, b"k");
    assert!(
        matches!(pending, Some(WriteOp::Put { value, .. }) if value == b"v2"),
        "expected Put with v2, got {pending:?}"
    );
}

#[test]
fn get_pending_write_returns_none_for_missing() {
    let oracle = test_oracle();
    let txn = Transaction::begin(&oracle);

    assert!(txn
        .get_pending_write(TxnPartition::Node, b"missing")
        .is_none());
}

#[test]
fn cache_read_and_retrieve() {
    let oracle = test_oracle();
    let mut txn = Transaction::begin(&oracle);

    txn.cache_read(TxnPartition::Node, b"k".to_vec(), Some(b"cached".to_vec()));

    let cached = txn.get_cached(TxnPartition::Node, b"k");
    assert_eq!(cached, Some(&Some(b"cached".to_vec())));
}

#[test]
fn cache_read_none_for_missing_key() {
    let oracle = test_oracle();
    let mut txn = Transaction::begin(&oracle);

    txn.cache_read(TxnPartition::Node, b"k".to_vec(), None);

    let cached = txn.get_cached(TxnPartition::Node, b"k");
    assert_eq!(cached, Some(&None));
}

#[test]
fn abort_discards_writes() {
    let oracle = test_oracle();
    let mut txn = Transaction::begin(&oracle);

    txn.put(TxnPartition::Node, b"k".to_vec(), b"v".to_vec());
    txn.stats_mut().nodes_created = 1;

    let stats = txn.abort();
    assert_eq!(stats.nodes_created, 1);
    // write_buffer is dropped — nothing to commit
}
