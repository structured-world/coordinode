use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use std::sync::Arc;
use tempfile::TempDir;

fn test_engine() -> (Arc<StorageEngine>, Arc<TimestampOracle>, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().to_string_lossy().as_ref(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
    (engine, oracle, dir)
}

fn mvcc_txn<'a>(engine: &'a StorageEngine, oracle: &'a TimestampOracle) -> Transaction<'a> {
    let snap = engine.snapshot();
    Transaction::new(engine, Some(oracle), Timestamp::from_raw(snap), Some(snap))
}

#[test]
fn put_then_get_reads_own_write() {
    let (engine, oracle, _d) = test_engine();
    let mut txn = mvcc_txn(&engine, &oracle);
    txn.put(Partition::Node, b"k1", b"v1").unwrap();
    // Read-your-own-writes from the buffer, before any flush.
    assert_eq!(
        txn.get(Partition::Node, b"k1").unwrap().as_deref(),
        Some(&b"v1"[..])
    );
}

#[test]
fn delete_buffers_tombstone_visible_to_own_read() {
    let (engine, oracle, _d) = test_engine();
    let mut txn = mvcc_txn(&engine, &oracle);
    txn.put(Partition::Node, b"k", b"v").unwrap();
    txn.delete(Partition::Node, b"k").unwrap();
    assert_eq!(txn.get(Partition::Node, b"k").unwrap(), None);
}

#[test]
fn get_absent_key_returns_none() {
    let (engine, oracle, _d) = test_engine();
    let mut txn = mvcc_txn(&engine, &oracle);
    assert_eq!(txn.get(Partition::Node, b"missing").unwrap(), None);
}

#[test]
fn reads_track_occ_scope_own_writes_do_not() {
    let (engine, oracle, _d) = test_engine();
    let mut txn = mvcc_txn(&engine, &oracle);
    // A storage read tracks the key in the OCC scope.
    txn.get(Partition::Node, b"r1").unwrap();
    txn.get(Partition::EdgeProp, b"r2").unwrap();
    assert_eq!(txn.occ_scope.as_ref().map(|s| s.tracked_count()), Some(2));
    // Reading own buffered write does NOT add to the read-set.
    txn.put(Partition::Node, b"w1", b"v").unwrap();
    txn.get(Partition::Node, b"w1").unwrap();
    assert_eq!(txn.occ_scope.as_ref().map(|s| s.tracked_count()), Some(2));
}

#[test]
fn into_state_resume_preserves_buffer_occ_and_read_ts() {
    // The interactive-transaction park/resume cycle (ADR-042): a
    // transaction's progress survives being parked as TransactionState
    // and rebuilt with fresh engine/oracle borrows.
    let (engine, oracle, _d) = test_engine();
    let mut txn = mvcc_txn(&engine, &oracle);
    let read_ts = txn.read_ts();
    txn.put(Partition::Node, b"k1", b"v1").unwrap();
    txn.delete(Partition::Node, b"k2").unwrap();
    txn.get(Partition::Node, b"r1").unwrap(); // tracks OCC
    let occ_before = txn.occ_scope.as_ref().map(|s| s.tracked_count());

    // Park and rebuild.
    let state = txn.into_state();
    let mut resumed = Transaction::resume(&engine, Some(&oracle), state);

    // read_ts pinned (repeatable read across statements).
    assert_eq!(resumed.read_ts(), read_ts);
    // Buffered write + tombstone survive (read-your-own-writes still works).
    assert_eq!(
        resumed.get(Partition::Node, b"k1").unwrap().as_deref(),
        Some(&b"v1"[..])
    );
    assert_eq!(resumed.get(Partition::Node, b"k2").unwrap(), None);
    // OCC read-set survives; the rebuilt `r1` read does not double-count.
    resumed.get(Partition::Node, b"r1").unwrap();
    assert_eq!(
        resumed.occ_scope.as_ref().map(|s| s.tracked_count()),
        occ_before,
    );
}

#[test]
fn prefix_scan_overlays_buffer_over_snapshot() {
    let (engine, oracle, _d) = test_engine();
    // Seed a committed row directly.
    engine.put(Partition::Node, b"p:a", b"old").unwrap();
    let mut txn = Transaction::new(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
        Some(engine.snapshot()),
    );
    // Buffer overrides the committed value and adds a new key.
    txn.put(Partition::Node, b"p:a", b"new").unwrap();
    txn.put(Partition::Node, b"p:b", b"b").unwrap();
    let mut got = txn.prefix_scan(Partition::Node, b"p:").unwrap();
    got.sort();
    assert_eq!(
        got,
        vec![
            (b"p:a".to_vec(), b"new".to_vec()),
            (b"p:b".to_vec(), b"b".to_vec()),
        ]
    );
}

#[test]
fn prefix_scan_buffered_tombstone_does_not_hide_storage_row() {
    // Behavioural parity with the executor: a buffered in-transaction
    // delete does NOT remove a storage row from a prefix scan (only a
    // buffered *value* overlays). Point reads still see the tombstone via
    // `get`; scans surface the snapshot row.
    let (engine, oracle, _d) = test_engine();
    engine.put(Partition::Node, b"p:x", b"v").unwrap();
    let mut txn = Transaction::new(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
        Some(engine.snapshot()),
    );
    txn.delete(Partition::Node, b"p:x").unwrap();
    // Point read sees the tombstone (RYOW).
    assert_eq!(txn.get(Partition::Node, b"p:x").unwrap(), None);
    // Scan still surfaces the storage row (documented parity behaviour).
    assert_eq!(
        txn.prefix_scan(Partition::Node, b"p:").unwrap(),
        vec![(b"p:x".to_vec(), b"v".to_vec())]
    );
}

#[test]
fn legacy_mode_writes_directly_no_buffer() {
    let (engine, _oracle, _d) = test_engine();
    // No oracle → legacy: put hits the engine immediately.
    let mut txn = Transaction::new(&engine, None, Timestamp::from_raw(0), None);
    txn.put(Partition::Node, b"lk", b"lv").unwrap();
    assert!(txn.is_mvcc().eq(&false));
    // Visible through a fresh engine read (not via buffer).
    assert_eq!(
        engine.get(Partition::Node, b"lk").unwrap().as_deref(),
        Some(&b"lv"[..])
    );
}

#[test]
fn prefix_scan_paged_walks_the_prefix_in_keyset_pages() {
    let (engine, oracle, _d) = test_engine();
    // Five committed rows under "p:", plus one outside it.
    for i in 0..5u8 {
        engine.put(Partition::Node, &[b'p', b':', i], &[i]).unwrap();
    }
    engine.put(Partition::Node, b"q:z", b"x").unwrap();
    let mut txn = Transaction::new(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
        Some(engine.snapshot()),
    );

    // Page in batches of two, resuming by keyset off the last key.
    let mut all = Vec::new();
    let mut resume: Option<Vec<u8>> = None;
    loop {
        let page = txn
            .prefix_scan_paged(Partition::Node, b"p:", resume.as_deref(), 2)
            .unwrap();
        all.extend(page.rows.clone());
        if page.exhausted {
            assert!(page.rows.len() <= 2);
            break;
        }
        assert_eq!(page.rows.len(), 2, "a non-final page is full");
        resume = page.last_key;
        assert!(resume.is_some());
    }

    // All five prefix rows, in key order, none from outside the prefix.
    assert_eq!(all.len(), 5);
    for (idx, (key, _)) in all.iter().enumerate() {
        assert_eq!(key, &vec![b'p', b':', idx as u8]);
    }
}

#[test]
fn prefix_scan_paged_empty_prefix_is_exhausted_with_no_last_key() {
    let (engine, oracle, _d) = test_engine();
    engine.put(Partition::Node, b"q:a", b"x").unwrap();
    let mut txn = Transaction::new(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
        Some(engine.snapshot()),
    );
    let page = txn
        .prefix_scan_paged(Partition::Node, b"p:", None, 10)
        .unwrap();
    assert!(page.rows.is_empty());
    assert!(page.exhausted);
    assert!(page.last_key.is_none());
}

#[test]
fn prefix_scan_paged_exact_limit_reports_exhausted() {
    let (engine, oracle, _d) = test_engine();
    engine.put(Partition::Node, b"p:a", b"1").unwrap();
    engine.put(Partition::Node, b"p:b", b"2").unwrap();
    let mut txn = Transaction::new(
        &engine,
        Some(&oracle),
        Timestamp::from_raw(engine.snapshot()),
        Some(engine.snapshot()),
    );
    // Exactly `limit` matching rows: exhausted, no phantom extra page.
    let page = txn
        .prefix_scan_paged(Partition::Node, b"p:", None, 2)
        .unwrap();
    assert_eq!(page.rows.len(), 2);
    assert!(page.exhausted);
    assert_eq!(page.last_key, Some(b"p:b".to_vec()));
}
