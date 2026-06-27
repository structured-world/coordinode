use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use coordinode_core::graph::types::Value;
use tokio::sync::mpsc;

use super::*;
use crate::engine::{CursorEngine, EngineError, QueryCursor};
use crate::types::{Ordering, SessionEvent, SessionOp, SessionStats};

/// A cursor over a fixed in-memory result, paged by `next_batch`.
struct MockCursor {
    columns: Vec<String>,
    rows: Vec<Vec<Value>>,
    pos: usize,
}

impl QueryCursor for MockCursor {
    fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }
    fn next_batch(&mut self, max: usize) -> Result<Vec<Vec<Value>>, EngineError> {
        let end = (self.pos + max).min(self.rows.len());
        let batch = self.rows[self.pos..end].to_vec();
        self.pos = end;
        Ok(batch)
    }
    fn stats(&self) -> SessionStats {
        SessionStats::default()
    }
}

/// An engine returning a fixed result, or a fixed error.
struct MockEngine {
    columns: Vec<String>,
    rows: Vec<Vec<Value>>,
    /// When set, every `open_cursor` fails with this message.
    fail: Option<String>,
    /// When set, only a statement whose query equals this string fails (the rest
    /// succeed): used to abort a transaction mid-way.
    fail_query: Option<String>,
    next_txn: AtomicU64,
}

impl CursorEngine for MockEngine {
    fn open_cursor(
        &self,
        query: &str,
        _params: HashMap<String, Value>,
        _txid: u64,
    ) -> Result<Box<dyn QueryCursor>, EngineError> {
        if let Some(message) = &self.fail {
            return Err(EngineError(message.clone()));
        }
        if self.fail_query.as_deref() == Some(query) {
            return Err(EngineError("statement failed".to_string()));
        }
        Ok(Box::new(MockCursor {
            columns: self.columns.clone(),
            rows: self.rows.clone(),
            pos: 0,
        }))
    }

    fn begin_transaction(&self) -> Result<u64, EngineError> {
        Ok(self.next_txn.fetch_add(1, AtomicOrdering::Relaxed) + 1)
    }

    fn commit_transaction(&self, _txid: u64) -> Result<u64, EngineError> {
        Ok(0)
    }

    fn rollback_transaction(&self, _txid: u64) -> Result<(), EngineError> {
        Ok(())
    }
}

fn engine(columns: &[&str], rows: Vec<Vec<Value>>) -> Arc<dyn CursorEngine> {
    Arc::new(MockEngine {
        columns: columns.iter().map(|s| s.to_string()).collect(),
        rows,
        fail: None,
        fail_query: None,
        next_txn: AtomicU64::new(0),
    })
}

fn exec() -> SessionOp {
    SessionOp::Execute {
        query: "RETURN 1".to_string(),
        params: HashMap::new(),
        txid: 0,
        nonce: 0,
    }
}

/// Drive a fresh session with `ops` (request_id = index + 1) and collect every
/// event grouped by request_id. Exercises the real `Session::run` path.
async fn run_session(
    engine: Arc<dyn CursorEngine>,
    ops: Vec<SessionOp>,
) -> HashMap<u64, Vec<SessionEvent>> {
    let registry = Arc::new(SessionRegistry::new(std::time::Duration::from_secs(30)));
    let manager = SessionManager::new(engine, registry);
    let session = manager.open("test".into());
    let (op_tx, op_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);
    let handle = tokio::spawn(session.run(op_rx, ev_tx));
    for (i, op) in ops.into_iter().enumerate() {
        op_tx.send((i as u64 + 1, op)).await.unwrap();
    }
    drop(op_tx);
    let mut by_id: HashMap<u64, Vec<SessionEvent>> = HashMap::new();
    while let Some((request_id, event)) = ev_rx.recv().await {
        by_id.entry(request_id).or_default().push(event);
    }
    handle.await.unwrap();
    by_id
}

/// Drive a single op and return its events.
async fn run_one(engine: Arc<dyn CursorEngine>, op: SessionOp) -> Vec<SessionEvent> {
    run_session(engine, vec![op])
        .await
        .remove(&1)
        .unwrap_or_default()
}

#[tokio::test]
async fn execute_pages_cursor_open_then_rows_then_end() {
    let engine = engine(&["n"], vec![vec![Value::Int(1)], vec![Value::Int(2)]]);
    let events = run_one(engine, exec()).await;
    match events.as_slice() {
        [SessionEvent::CursorOpen { columns }, SessionEvent::Rows { rows }, SessionEvent::CursorEnd { .. }] =>
        {
            assert_eq!(columns, &vec!["n".to_string()]);
            assert_eq!(rows, &vec![vec![Value::Int(1)], vec![Value::Int(2)]]);
        }
        other => panic!("expected open/rows/end, got {other:?}"),
    }
}

#[tokio::test]
async fn execute_with_an_empty_result_opens_then_ends_with_no_rows() {
    let engine = engine(&["n"], vec![]);
    let events = run_one(engine, exec()).await;
    assert!(matches!(
        events.as_slice(),
        [
            SessionEvent::CursorOpen { .. },
            SessionEvent::CursorEnd { .. }
        ]
    ));
}

#[tokio::test]
async fn execute_surfaces_an_engine_error() {
    let engine: Arc<dyn CursorEngine> = Arc::new(MockEngine {
        columns: vec![],
        rows: vec![],
        fail: Some("boom".to_string()),
        fail_query: None,
        next_txn: AtomicU64::new(0),
    });
    let events = run_one(engine, exec()).await;
    match events.as_slice() {
        [SessionEvent::Error { code, message }] => {
            assert_eq!(*code, ErrorCode::Internal);
            assert_eq!(message, "boom");
        }
        other => panic!("expected a single Error, got {other:?}"),
    }
}

#[tokio::test]
async fn begin_allocates_nonzero_handles_from_the_engine() {
    let engine = engine(&[], vec![]);
    let begin = || SessionOp::Begin {
        ordering: Ordering::Ordered,
        drain_timeout_ms: 0,
    };
    let by_id = run_session(engine, vec![begin(), begin()]).await;
    let t1 = match by_id[&1].as_slice() {
        [SessionEvent::Begun { txid }] => *txid,
        other => panic!("expected Begun, got {other:?}"),
    };
    let t2 = match by_id[&2].as_slice() {
        [SessionEvent::Begun { txid }] => *txid,
        other => panic!("expected Begun, got {other:?}"),
    };
    assert_ne!(t1, 0, "transaction handles are non-zero by contract");
    assert_ne!(t1, t2, "each begin gets a distinct handle");
}

#[tokio::test]
async fn begin_then_commit_acknowledges_and_rollback_cancel_are_silent() {
    let engine = engine(&[], vec![]);
    // Begin (req 1) → Begun{txid=1 from the mock}; Commit that txn (req 2) →
    // Committed; Rollback an open txn (req 3, begun at req... here unknown) and
    // Cancel (req 4) emit nothing. Begin is handled inline before the next op,
    // so Commit{txid:1} finds the open transaction.
    let by_id = run_session(
        engine,
        vec![
            SessionOp::Begin {
                ordering: Ordering::Ordered,
                drain_timeout_ms: 0,
            },
            SessionOp::Commit {
                txid: 1,
                last_nonce: 0,
            },
            SessionOp::Rollback { txid: 1 },
            SessionOp::Cancel {
                target_request_id: 1,
            },
        ],
    )
    .await;
    assert!(matches!(
        by_id[&1].as_slice(),
        [SessionEvent::Begun { txid: 1 }]
    ));
    assert!(matches!(
        by_id[&2].as_slice(),
        [SessionEvent::Committed { .. }]
    ));
    // Rollback of the now-resolved txn and Cancel produce no events.
    assert!(!by_id.contains_key(&3));
    assert!(!by_id.contains_key(&4));
}

#[tokio::test]
async fn a_failed_statement_aborts_the_transaction_and_rejects_the_rest() {
    // UNORDERED first-failure semantics: once a statement fails, the transaction
    // is dead, so later statements and the commit are rejected with the
    // aborted-transaction error, not run.
    let engine: Arc<dyn CursorEngine> = Arc::new(MockEngine {
        columns: vec!["n".to_string()],
        rows: vec![vec![Value::Int(1)]],
        fail: None,
        fail_query: Some("FAIL".to_string()),
        next_txn: AtomicU64::new(0),
    });
    // Begin (req1) → txid 1; ok statement (req2); failing statement (req3);
    // a later statement (req4); commit (req5).
    let by_id = run_session(
        engine,
        vec![
            SessionOp::Begin {
                ordering: Ordering::Unordered,
                drain_timeout_ms: 0,
            },
            SessionOp::Execute {
                query: "ok".to_string(),
                params: HashMap::new(),
                txid: 1,
                nonce: 0,
            },
            SessionOp::Execute {
                query: "FAIL".to_string(),
                params: HashMap::new(),
                txid: 1,
                nonce: 0,
            },
            SessionOp::Execute {
                query: "after".to_string(),
                params: HashMap::new(),
                txid: 1,
                nonce: 0,
            },
            SessionOp::Commit {
                txid: 1,
                last_nonce: 0,
            },
        ],
    )
    .await;

    // The first statement succeeded.
    assert!(matches!(
        by_id[&2].last(),
        Some(SessionEvent::CursorEnd { .. })
    ));
    // The failing statement surfaced its error.
    assert!(matches!(by_id[&3].as_slice(), [SessionEvent::Error { .. }]));
    // The later statement and the commit are both rejected as aborted.
    for rid in [4u64, 5] {
        match by_id[&rid].as_slice() {
            [SessionEvent::Error { message, .. }] => {
                assert!(
                    message.contains("aborted"),
                    "req {rid} expected aborted error, got {message:?}"
                );
            }
            other => panic!("req {rid} expected a single aborted Error, got {other:?}"),
        }
    }
}

/// Drive a session, collecting events in arrival order (cross-request order
/// preserved, unlike `run_session`).
async fn run_session_flat(
    engine: Arc<dyn CursorEngine>,
    ops: Vec<SessionOp>,
) -> Vec<(u64, SessionEvent)> {
    let registry = Arc::new(SessionRegistry::new(std::time::Duration::from_secs(30)));
    let manager = SessionManager::new(engine, registry);
    let session = manager.open("test".into());
    let (op_tx, op_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);
    let handle = tokio::spawn(session.run(op_rx, ev_tx));
    for (i, op) in ops.into_iter().enumerate() {
        op_tx.send((i as u64 + 1, op)).await.unwrap();
    }
    drop(op_tx);
    let mut events = Vec::new();
    while let Some(ev) = ev_rx.recv().await {
        events.push(ev);
    }
    handle.await.unwrap();
    events
}

#[tokio::test]
async fn ordered_applies_statements_in_nonce_order_not_arrival_order() {
    // ORDERED reorders by nonce: req2 carries nonce 2 and req3 carries nonce 1,
    // sent in that (reversed) order. They must be applied nonce-1-then-nonce-2,
    // so req3's cursor opens before req2's in the event stream.
    let engine = engine(&["n"], vec![vec![Value::Int(1)]]);
    let stmt = |nonce| SessionOp::Execute {
        query: "x".to_string(),
        params: HashMap::new(),
        txid: 1,
        nonce,
    };
    let events = run_session_flat(
        engine,
        vec![
            SessionOp::Begin {
                ordering: Ordering::Ordered,
                drain_timeout_ms: 0,
            },
            stmt(2), // request_id 2, nonce 2 (arrives first)
            stmt(1), // request_id 3, nonce 1 (arrives second, applies first)
            SessionOp::Commit {
                txid: 1,
                last_nonce: 2,
            },
        ],
    )
    .await;
    // Index of the first CursorOpen for each request id.
    let first_open = |rid: u64| {
        events
            .iter()
            .position(|(id, e)| *id == rid && matches!(e, SessionEvent::CursorOpen { .. }))
    };
    let open2 = first_open(2).expect("req2 opened");
    let open3 = first_open(3).expect("req3 opened");
    assert!(
        open3 < open2,
        "nonce 1 (req3) must be applied before nonce 2 (req2): {events:?}"
    );
}

#[tokio::test]
async fn ordered_commit_drain_times_out_on_a_missing_nonce() {
    // A gap that never fills: nonce 2 arrives, nonce 1 does not, and the commit
    // expects last_nonce 2. The commit must error once the (short) drain timeout
    // elapses rather than hang.
    let engine = engine(&["n"], vec![vec![Value::Int(1)]]);
    let registry = Arc::new(SessionRegistry::new(std::time::Duration::from_secs(30)));
    let manager = SessionManager::new(engine, registry);
    let session = manager.open("test".into());
    let (op_tx, op_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);
    let handle = tokio::spawn(session.run(op_rx, ev_tx));

    op_tx
        .send((
            1,
            SessionOp::Begin {
                ordering: Ordering::Ordered,
                drain_timeout_ms: 50,
            },
        ))
        .await
        .unwrap();
    let txid = loop {
        if let (1, SessionEvent::Begun { txid }) = ev_rx.recv().await.unwrap() {
            break txid;
        }
    };
    // Send only nonce 2 (nonce 1 is the missing gap), then commit expecting 2.
    op_tx
        .send((
            2,
            SessionOp::Execute {
                query: "x".to_string(),
                params: HashMap::new(),
                txid,
                nonce: 2,
            },
        ))
        .await
        .unwrap();
    op_tx
        .send((
            3,
            SessionOp::Commit {
                txid,
                last_nonce: 2,
            },
        ))
        .await
        .unwrap();
    // Keep op_tx alive so the transaction stays routable while it drains; the
    // commit errors after the drain timeout fires on the unfilled gap.
    let ev = loop {
        let (rid, e) = ev_rx.recv().await.unwrap();
        if rid == 3 {
            break e;
        }
    };
    assert!(
        matches!(ev, SessionEvent::Error { .. }),
        "commit must error on drain timeout, got {ev:?}"
    );
    drop(op_tx);
    handle.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_dispatches_a_burst_concurrently_with_correlated_cursors() {
    let registry = Arc::new(SessionRegistry::new(std::time::Duration::from_secs(30)));
    let manager = SessionManager::new(engine(&["n"], vec![vec![Value::Int(7)]]), registry);
    let session = manager.open("test".into());
    let (op_tx, op_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);
    let handle = tokio::spawn(session.run(op_rx, ev_tx));

    for i in 1..=10u64 {
        op_tx.send((i, exec())).await.unwrap();
    }
    drop(op_tx);

    let mut by_id: HashMap<u64, Vec<SessionEvent>> = HashMap::new();
    while let Some((request_id, event)) = ev_rx.recv().await {
        by_id.entry(request_id).or_default().push(event);
    }
    handle.await.unwrap();

    assert_eq!(by_id.len(), 10, "every request id is answered exactly once");
    for i in 1..=10u64 {
        assert!(
            matches!(
                by_id[&i].as_slice(),
                [
                    SessionEvent::CursorOpen { .. },
                    SessionEvent::Rows { .. },
                    SessionEvent::CursorEnd { .. }
                ]
            ),
            "request {i} got {:?}",
            by_id[&i]
        );
    }
}
