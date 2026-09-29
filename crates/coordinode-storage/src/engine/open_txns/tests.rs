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
        engine.await_transactions_through(boundary, 0, POLL, SHORT),
        Err(1),
        "the transaction opened before the boundary is still open"
    );
    assert_eq!(
        engine.await_transactions_through(boundary, 1, POLL, SHORT),
        Ok(()),
        "a caller waiting from inside the older transaction does not wait for itself"
    );
    drop(older);
    assert_eq!(
        engine.await_transactions_through(boundary, 0, POLL, SHORT),
        Ok(()),
        "only the newer transaction is open, and it is not waited for"
    );
    drop(newer);
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
        engine.await_transactions_through(boundary, 0, POLL, SHORT),
        Err(1)
    );
    drop(txn);
    assert_eq!(
        engine.await_transactions_through(boundary, 0, POLL, SHORT),
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
        engine.await_transactions_through(boundary, 0, POLL, SHORT),
        Err(1),
        "parked"
    );

    let resumed = Transaction::resume(&engine, Some(&oracle), parked);
    assert_eq!(
        engine.await_transactions_through(boundary, 0, POLL, SHORT),
        Err(1),
        "resumed"
    );
    let parked = resumed.into_state();
    drop(parked);
    assert_eq!(
        engine.await_transactions_through(boundary, 0, POLL, SHORT),
        Ok(())
    );
}
