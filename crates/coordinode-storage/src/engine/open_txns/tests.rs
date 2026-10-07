use std::sync::Arc;
use std::time::Duration;

use tempfile::TempDir;

use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::engine::transaction::Transaction;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};

fn engine(dir: &TempDir, oracle: &Arc<TimestampOracle>) -> StorageEngine {
    StorageEngine::open_with_oracle(
        &StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]),
        Arc::clone(oracle),
    )
    .expect("open")
}

const POLL: Duration = Duration::from_millis(1);
const SHORT: Duration = Duration::from_millis(20);

/// A transaction open before the boundary is waited for, one opened after it
/// is not, and the wait ends once the older one is gone.
#[test]
fn the_wait_covers_the_transactions_opened_before_the_boundary() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = engine(&dir, &oracle);

    let older = Transaction::begin(&engine, Some(&oracle), oracle.next());
    let boundary = engine.snapshot_boundary();
    let newer = Transaction::begin(&engine, Some(&oracle), oracle.next());

    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Err(1),
        "the transaction opened before the boundary is still open"
    );
    drop(older);
    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Ok(()),
        "only the newer transaction is open, and it is not waited for"
    );
    drop(newer);
}

/// A caller that waits from inside a transaction of its own, one that staged
/// nothing, leaves the wait without hiding any other transaction: once it
/// has left, the wait still covers the older one, and the caller's
/// transaction ending later changes nothing.
#[test]
fn a_released_transaction_leaves_the_wait_and_hides_no_other() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = engine(&dir, &oracle);

    let older = Transaction::begin(&engine, Some(&oracle), oracle.next());
    let mut own = Transaction::begin(&engine, Some(&oracle), oracle.next());
    let boundary = engine.snapshot_boundary();

    assert!(own.release_from_schema_waits());
    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Err(1),
        "the older transaction is still waited for"
    );
    drop(own);
    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Err(1),
        "the caller's transaction ending does not count for the older one"
    );
    drop(older);
    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Ok(())
    );
}

/// A transaction holding staged writes stays among the waited ones: an index
/// built past it would miss them.
#[test]
fn a_transaction_with_staged_writes_is_not_released() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = engine(&dir, &oracle);

    let mut own = Transaction::begin(&engine, Some(&oracle), oracle.next());
    own.put(crate::engine::partition::Partition::Node, b"k", b"v")
        .expect("stage");
    let boundary = engine.snapshot_boundary();

    assert!(!own.release_from_schema_waits());
    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Err(1)
    );
    drop(own);
}

/// A transaction with no snapshot yet, the auto-commit shape before the
/// executor pins one, is open all the same.
#[test]
fn a_transaction_without_a_snapshot_is_waited_for() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = engine(&dir, &oracle);

    let txn = Transaction::new(&engine, Some(&oracle), Timestamp::ZERO, None);
    let boundary = engine.snapshot_boundary();
    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Err(1)
    );
    drop(txn);
    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Ok(())
    );
}

/// An interactive transaction parks its state between statements; it stays
/// open while parked and after it is resumed, until the last state drops.
#[test]
fn a_parked_transaction_stays_open_until_its_state_drops() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = engine(&dir, &oracle);

    let mut txn = Transaction::begin(&engine, Some(&oracle), oracle.next());
    let parked = txn.take_state();
    drop(txn);
    let boundary = engine.snapshot_boundary();
    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Err(1),
        "parked"
    );

    let resumed = Transaction::resume(&engine, Some(&oracle), parked);
    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Err(1),
        "resumed"
    );
    let parked = resumed.into_state();
    drop(parked);
    assert_eq!(
        engine.await_transactions_through(boundary, POLL, SHORT),
        Ok(())
    );
}
