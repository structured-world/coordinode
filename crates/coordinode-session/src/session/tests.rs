use std::collections::HashMap;
use std::sync::atomic::AtomicU64;

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
    fail: Option<String>,
}

impl CursorEngine for MockEngine {
    fn open_cursor(
        &self,
        _query: &str,
        _params: HashMap<String, Value>,
        _txid: u64,
    ) -> Result<Box<dyn QueryCursor>, EngineError> {
        match &self.fail {
            Some(message) => Err(EngineError(message.clone())),
            None => Ok(Box::new(MockCursor {
                columns: self.columns.clone(),
                rows: self.rows.clone(),
                pos: 0,
            })),
        }
    }
}

fn engine(columns: &[&str], rows: Vec<Vec<Value>>) -> Arc<dyn CursorEngine> {
    Arc::new(MockEngine {
        columns: columns.iter().map(|s| s.to_string()).collect(),
        rows,
        fail: None,
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

/// Dispatch one op against `engine` and collect the events it emits.
async fn run_op(
    engine: &Arc<dyn CursorEngine>,
    counter: &AtomicU64,
    op: SessionOp,
) -> Vec<SessionEvent> {
    let (tx, mut rx) = mpsc::channel(64);
    let registry = SessionRegistry::new(std::time::Duration::from_secs(30));
    let session_id = registry.register_session("test".into());
    dispatch(engine, &registry, session_id, counter, 1, op, &tx).await;
    drop(tx);
    let mut out = Vec::new();
    while let Some((_request_id, event)) = rx.recv().await {
        out.push(event);
    }
    out
}

#[tokio::test]
async fn execute_pages_cursor_open_then_rows_then_end() {
    let engine = engine(&["n"], vec![vec![Value::Int(1)], vec![Value::Int(2)]]);
    let counter = AtomicU64::new(0);
    let events = run_op(&engine, &counter, exec()).await;
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
    let counter = AtomicU64::new(0);
    let events = run_op(&engine, &counter, exec()).await;
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
    });
    let counter = AtomicU64::new(0);
    let events = run_op(&engine, &counter, exec()).await;
    match events.as_slice() {
        [SessionEvent::Error { code, message }] => {
            assert_eq!(*code, ErrorCode::Internal);
            assert_eq!(message, "boom");
        }
        other => panic!("expected a single Error, got {other:?}"),
    }
}

#[tokio::test]
async fn begin_allocates_monotonic_nonzero_txids() {
    let engine = engine(&[], vec![]);
    let counter = AtomicU64::new(0);
    let begin = || SessionOp::Begin {
        ordering: Ordering::Ordered,
        drain_timeout_ms: 0,
    };
    let t1 = match run_op(&engine, &counter, begin()).await.as_slice() {
        [SessionEvent::Begun { txid }] => *txid,
        other => panic!("expected Begun, got {other:?}"),
    };
    let t2 = match run_op(&engine, &counter, begin()).await.as_slice() {
        [SessionEvent::Begun { txid }] => *txid,
        other => panic!("expected Begun, got {other:?}"),
    };
    assert_ne!(t1, 0, "transaction handles are non-zero by contract");
    assert_eq!(t2, t1 + 1, "handles are monotonic");
}

#[tokio::test]
async fn commit_acknowledges_while_rollback_and_cancel_are_silent() {
    let engine = engine(&[], vec![]);
    let counter = AtomicU64::new(0);
    assert!(matches!(
        run_op(
            &engine,
            &counter,
            SessionOp::Commit {
                txid: 1,
                last_nonce: 0
            }
        )
        .await
        .as_slice(),
        [SessionEvent::Committed { .. }]
    ));
    assert!(run_op(&engine, &counter, SessionOp::Rollback { txid: 1 })
        .await
        .is_empty());
    assert!(run_op(
        &engine,
        &counter,
        SessionOp::Cancel {
            target_request_id: 1
        }
    )
    .await
    .is_empty());
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
