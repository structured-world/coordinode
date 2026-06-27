use std::time::Duration;

use super::*;
use coordinode_core::operations::{OperationsView, TxnOrdering};

#[test]
fn registers_sessions_with_distinct_ids_and_snapshots_them() {
    let reg = SessionRegistry::new(Duration::from_secs(30));
    let a = reg.register_session("10.0.0.1:1".into());
    let b = reg.register_session("10.0.0.2:2".into());
    assert_ne!(a, b, "session ids are distinct");

    let snap = reg.sessions();
    assert_eq!(snap.len(), 2);
    assert!(snap
        .iter()
        .all(|s| s.in_flight == 0 && s.transactions.is_empty()));
    assert!(snap.iter().any(|s| s.peer == "10.0.0.1:1"));
}

#[test]
fn close_session_removes_it() {
    let reg = SessionRegistry::new(Duration::from_secs(30));
    let id = reg.register_session("p".into());
    reg.close_session(id);
    assert!(reg.sessions().is_empty());
}

#[test]
fn in_flight_counts_up_and_down_without_underflow() {
    let reg = SessionRegistry::new(Duration::from_secs(30));
    let id = reg.register_session("p".into());
    reg.request_started(id);
    reg.request_started(id);
    assert_eq!(reg.sessions()[0].in_flight, 2);
    reg.request_finished(id);
    assert_eq!(reg.sessions()[0].in_flight, 1);
    // Extra finishes must not underflow past zero.
    reg.request_finished(id);
    reg.request_finished(id);
    assert_eq!(reg.sessions()[0].in_flight, 0);
}

#[test]
fn transactions_appear_with_ordering_and_a_live_countdown() {
    let reg = SessionRegistry::new(Duration::from_secs(30));
    let id = reg.register_session("p".into());
    reg.begin_txn(id, 7, Ordering::Ordered);

    let snap = reg.sessions();
    let txns = &snap[0].transactions;
    assert_eq!(txns.len(), 1);
    assert_eq!(txns[0].txid, 7);
    assert_eq!(txns[0].ordering, TxnOrdering::Ordered);
    // Fresh transaction: countdown is close to the full timeout, never above it.
    assert!(txns[0].auto_abort_in_ms > 0);
    assert!(txns[0].auto_abort_in_ms <= 30_000);
}

#[test]
fn end_txn_removes_only_that_transaction() {
    let reg = SessionRegistry::new(Duration::from_secs(30));
    let id = reg.register_session("p".into());
    reg.begin_txn(id, 1, Ordering::Ordered);
    reg.begin_txn(id, 2, Ordering::Unordered);
    reg.end_txn(id, 1);

    let txns = reg.sessions()[0].transactions.clone();
    assert_eq!(txns.len(), 1);
    assert_eq!(txns[0].txid, 2);
    assert_eq!(txns[0].ordering, TxnOrdering::Unordered);
}

#[test]
fn reap_idle_drops_expired_transactions_and_reports_them() {
    // Zero timeout: any begun transaction is immediately reapable.
    let reg = SessionRegistry::new(Duration::from_millis(0));
    let id = reg.register_session("p".into());
    reg.begin_txn(id, 42, Ordering::Ordered);

    let reaped = reg.reap_idle();
    assert_eq!(reaped, vec![(id, 42)]);
    assert!(reg.sessions()[0].transactions.is_empty());
    // A second pass finds nothing left to reap.
    assert!(reg.reap_idle().is_empty());
}

#[test]
fn reap_idle_keeps_fresh_transactions() {
    let reg = SessionRegistry::new(Duration::from_secs(3600));
    let id = reg.register_session("p".into());
    reg.begin_txn(id, 1, Ordering::Ordered);
    assert!(reg.reap_idle().is_empty(), "fresh txn must survive reaping");
    assert_eq!(reg.sessions()[0].transactions.len(), 1);
}
