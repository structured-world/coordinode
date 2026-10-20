use std::collections::HashMap;

use coordinode_core::graph::types::Value;
use coordinode_core::operations::{OperationsView, TxnOrdering};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_core::txn::transaction::CommitReceipt;
use coordinode_core::txn::write_concern::WriteConcern;
use tokio::sync::mpsc;

use super::*;
use crate::engine::{CursorEngine, EngineError, QueryCursor};
use crate::types::{ConnectionSettings, SessionEvent, SessionOp, SessionStats};

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

fn empty_cursor() -> Box<dyn QueryCursor> {
    Box::new(MockCursor {
        columns: Vec::new(),
        rows: Vec::new(),
        pos: 0,
    })
}

/// The receipt the mock engine hands back on commit. Distinctive values so a
/// test can tell a pass-through from a default.
const MOCK_RECEIPT: CommitReceipt = CommitReceipt {
    commit_ts: Timestamp::from_raw(4242),
    applied_index: Some(7),
};

/// Transaction ID number `n` of a test.
fn txid(n: u64) -> TransactionId {
    let mut bytes = [0u8; 16];
    bytes[8..].copy_from_slice(&n.to_be_bytes());
    TransactionId::from_raw(bytes)
}

/// An engine returning a fixed result, or a fixed error.
struct MockEngine {
    columns: Vec<String>,
    rows: Vec<Vec<Value>>,
    /// When set, every `open_cursor` fails with this message.
    fail: Option<String>,
}

impl CursorEngine for MockEngine {
    fn open_cursor(
        &self,
        _query: &str,
        _params: HashMap<String, Value>,
        transaction: Option<&InTransaction>,
        _settings: &ConnectionSettings,
        _source: Option<&StatementSource>,
        _operation: Option<&OperationDescriptor>,
        _cancel: &coordinode_core::budget::CancelFlag,
    ) -> Result<Box<dyn QueryCursor>, EngineError> {
        if let Some(message) = &self.fail {
            return Err(EngineError(Failure {
                code: ErrorCode::Unavailable,
                message: message.clone(),
                details: b"opaque".to_vec(),
            }));
        }
        if let Some(transaction) = transaction {
            transaction.admission.admitted();
        }
        Ok(Box::new(MockCursor {
            columns: self.columns.clone(),
            rows: self.rows.clone(),
            pos: 0,
        }))
    }

    fn begin_transaction(
        &self,
        _transaction: TransactionId,
        _ordering: TxnOrdering,
        _drain: Option<Duration>,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    fn commit_transaction(
        &self,
        _transaction: TransactionId,
        _last_nonce: Option<u64>,
        _operation: Option<&OperationDescriptor>,
        admission: &AdmissionSignal,
    ) -> Result<CommitReceipt, EngineError> {
        admission.admitted();
        Ok(MOCK_RECEIPT)
    }

    fn rollback_transaction(
        &self,
        _transaction: TransactionId,
        admission: &AdmissionSignal,
    ) -> Result<(), EngineError> {
        admission.admitted();
        Ok(())
    }
}

fn engine(columns: &[&str], rows: Vec<Vec<Value>>) -> Arc<dyn CursorEngine> {
    Arc::new(MockEngine {
        columns: columns.iter().map(|s| s.to_string()).collect(),
        rows,
        fail: None,
    })
}

fn statement(query: &str, transaction: Option<TransactionId>) -> SessionOp {
    SessionOp::Execute {
        query: query.to_string(),
        params: HashMap::new(),
        transaction,
        nonce: 0,
        settings: ConnectionSettings::default(),
        source: None,
        operation: None,
    }
}

fn exec() -> SessionOp {
    statement("RETURN 1", None)
}

/// What reached a recording engine, in the order it arrived.
#[derive(Debug, Clone, PartialEq)]
enum Call {
    // Boxed fields: a test record, kept small beside the other calls.
    Statement {
        query: String,
        transaction: Option<(TransactionId, u64)>,
        settings: Box<ConnectionSettings>,
        source: Option<StatementSource>,
        operation: Option<OperationDescriptor>,
    },
    Begin(TransactionId, TxnOrdering, Option<Duration>),
    Commit(TransactionId, Option<u64>, Option<OperationDescriptor>),
    Rollback(TransactionId),
}

/// An engine that answers every statement with an empty result and records
/// every call it receives.
#[derive(Default)]
struct RecordingEngine {
    calls: parking_lot::Mutex<Vec<Call>>,
}

impl RecordingEngine {
    fn calls(&self) -> Vec<Call> {
        self.calls.lock().clone()
    }

    fn statement(&self, query: &str) -> Call {
        self.calls()
            .into_iter()
            .find(|call| matches!(call, Call::Statement { query: q, .. } if q == query))
            .unwrap_or_else(|| panic!("{query} never reached the engine"))
    }
}

impl CursorEngine for RecordingEngine {
    fn session_setting(&self, query: &str) -> Option<Result<ConnectionSettings, EngineError>> {
        match query {
            "SET exact" => Some(Ok(ConnectionSettings {
                vector_consistency: Some(
                    coordinode_core::graph::types::VectorConsistencyMode::Exact,
                ),
                ..ConnectionSettings::default()
            })),
            "SET refused" => Some(Err(EngineError::internal("no such value"))),
            _ => None,
        }
    }

    fn open_cursor(
        &self,
        query: &str,
        _params: HashMap<String, Value>,
        transaction: Option<&InTransaction>,
        settings: &ConnectionSettings,
        source: Option<&StatementSource>,
        operation: Option<&OperationDescriptor>,
        _cancel: &coordinode_core::budget::CancelFlag,
    ) -> Result<Box<dyn QueryCursor>, EngineError> {
        self.calls.lock().push(Call::Statement {
            query: query.to_string(),
            transaction: transaction.map(|t| (t.id, t.nonce)),
            settings: Box::new(settings.clone()),
            source: source.cloned(),
            operation: operation.copied(),
        });
        if let Some(transaction) = transaction {
            transaction.admission.admitted();
        }
        Ok(empty_cursor())
    }

    fn begin_transaction(
        &self,
        transaction: TransactionId,
        ordering: TxnOrdering,
        drain: Option<Duration>,
    ) -> Result<(), EngineError> {
        self.calls
            .lock()
            .push(Call::Begin(transaction, ordering, drain));
        Ok(())
    }

    fn commit_transaction(
        &self,
        transaction: TransactionId,
        last_nonce: Option<u64>,
        operation: Option<&OperationDescriptor>,
        admission: &AdmissionSignal,
    ) -> Result<CommitReceipt, EngineError> {
        admission.admitted();
        self.calls
            .lock()
            .push(Call::Commit(transaction, last_nonce, operation.copied()));
        Ok(MOCK_RECEIPT)
    }

    fn rollback_transaction(
        &self,
        transaction: TransactionId,
        admission: &AdmissionSignal,
    ) -> Result<(), EngineError> {
        admission.admitted();
        self.calls.lock().push(Call::Rollback(transaction));
        Ok(())
    }
}

/// Drive a fresh session with `ops` (request_id = index + 1) and collect every
/// event grouped by request_id. Exercises the real `Session::run` path.
async fn run_session(
    engine: Arc<dyn CursorEngine>,
    ops: Vec<SessionOp>,
) -> HashMap<u64, Vec<SessionEvent>> {
    run_session_on(engine, Arc::new(SessionRegistry::new()), ops).await
}

async fn run_session_on(
    engine: Arc<dyn CursorEngine>,
    registry: Arc<SessionRegistry>,
    ops: Vec<SessionOp>,
) -> HashMap<u64, Vec<SessionEvent>> {
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

/// A statement that fails after some of its rows were sent ends with the
/// error, never with `CursorEnd`: the rows already sent are not a complete
/// answer, and a client reading the stream can tell.
#[tokio::test]
async fn a_statement_failing_after_its_first_rows_ends_with_the_error() {
    /// Gives one row, then fails as a statement past its limit does.
    struct FailsOnSecondPage {
        pages: u8,
    }
    impl QueryCursor for FailsOnSecondPage {
        fn columns(&self) -> Vec<String> {
            vec!["n".to_string()]
        }
        fn next_batch(&mut self, _max: usize) -> Result<Vec<Vec<Value>>, EngineError> {
            self.pages += 1;
            if self.pages == 1 {
                return Ok(vec![vec![Value::Int(1)]]);
            }
            Err(EngineError(Failure::new(
                ErrorCode::DeadlineExceeded,
                "the statement passed its deadline",
            )))
        }
        fn stats(&self) -> SessionStats {
            SessionStats::default()
        }
    }
    struct Engine;
    impl CursorEngine for Engine {
        fn open_cursor(
            &self,
            _query: &str,
            _params: HashMap<String, Value>,
            _transaction: Option<&InTransaction>,
            _settings: &ConnectionSettings,
            _source: Option<&StatementSource>,
            _operation: Option<&OperationDescriptor>,
            _cancel: &coordinode_core::budget::CancelFlag,
        ) -> Result<Box<dyn QueryCursor>, EngineError> {
            Ok(Box::new(FailsOnSecondPage { pages: 0 }))
        }
        fn begin_transaction(
            &self,
            _transaction: TransactionId,
            _ordering: TxnOrdering,
            _drain: Option<Duration>,
        ) -> Result<(), EngineError> {
            Ok(())
        }
        fn commit_transaction(
            &self,
            _transaction: TransactionId,
            _last_nonce: Option<u64>,
            _operation: Option<&OperationDescriptor>,
            _admission: &AdmissionSignal,
        ) -> Result<CommitReceipt, EngineError> {
            Ok(MOCK_RECEIPT)
        }
        fn rollback_transaction(
            &self,
            _transaction: TransactionId,
            _admission: &AdmissionSignal,
        ) -> Result<(), EngineError> {
            Ok(())
        }
    }

    let by_id = run_session(Arc::new(Engine), vec![exec()]).await;
    match by_id[&1].as_slice() {
        [
            SessionEvent::CursorOpen { .. },
            SessionEvent::Rows { rows },
            SessionEvent::Error(failure),
        ] => {
            assert_eq!(rows.len(), 1);
            assert_eq!(failure.code, ErrorCode::DeadlineExceeded);
        }
        other => panic!("expected rows then the error, got {other:?}"),
    }
}

/// A session SET changes that session's settings: it answers with an empty
/// result without reaching the engine, and the statements after it run under
/// the new setting. A second session is untouched, which is the point: before,
/// a SET sent to a server changed the database every client reads.
#[tokio::test]
async fn a_session_set_changes_only_its_session() {
    use coordinode_core::graph::types::VectorConsistencyMode;

    let recording = Arc::new(RecordingEngine::default());
    let engine = || Arc::clone(&recording) as Arc<dyn CursorEngine>;

    let by_id = run_session(
        engine(),
        vec![statement("SET exact", None), statement("after", None)],
    )
    .await;
    assert!(
        matches!(
            by_id[&1].as_slice(),
            [
                SessionEvent::CursorOpen { .. },
                SessionEvent::CursorEnd { .. }
            ]
        ),
        "{:?}",
        by_id[&1]
    );
    run_session(engine(), vec![statement("other session", None)]).await;

    assert!(
        recording
            .calls()
            .iter()
            .all(|call| !matches!(call, Call::Statement { query, .. } if query == "SET exact")),
        "the SET is the session's"
    );
    let Call::Statement { settings, .. } = recording.statement("after") else {
        unreachable!()
    };
    assert_eq!(
        settings.vector_consistency,
        Some(VectorConsistencyMode::Exact)
    );
    let Call::Statement { settings, .. } = recording.statement("other session") else {
        unreachable!()
    };
    assert_eq!(settings.vector_consistency, None);
}

/// A SET the engine refuses answers with that error, never with an empty
/// result, and leaves the session's settings as they were: the statement
/// after it runs under the earlier SET.
#[tokio::test]
async fn a_refused_set_is_an_error_and_changes_nothing() {
    let recording = Arc::new(RecordingEngine::default());
    let by_id = run_session(
        Arc::clone(&recording) as Arc<dyn CursorEngine>,
        vec![
            statement("SET exact", None),
            statement("SET refused", None),
            statement("after", None),
        ],
    )
    .await;
    assert!(
        matches!(by_id[&2].as_slice(), [SessionEvent::Error(failure)] if failure.message.contains("no such value")),
        "{:?}",
        by_id[&2]
    );
    let Call::Statement { settings, .. } = recording.statement("after") else {
        unreachable!()
    };
    assert_eq!(
        settings.vector_consistency,
        Some(coordinode_core::graph::types::VectorConsistencyMode::Exact)
    );
}

/// A statement reaches the engine with its session's settings, its own in
/// place of any it names, and the session's as they stood when it arrived; a
/// transaction's statement also carries its handle and nonce.
#[tokio::test]
async fn a_statement_runs_under_its_session_settings_and_its_own() {
    let recording = Arc::new(RecordingEngine::default());
    let here = StatementSource {
        file: "app.rs".to_string(),
        line: 7,
        function: "pay".to_string(),
        ..StatementSource::default()
    };
    let session = ConnectionSettings {
        read_concern: Some(2),
        read_preference: Some(4),
        write_concern: Some(WriteConcern::w1()),
        ..ConnectionSettings::default()
    };
    run_session(
        Arc::clone(&recording) as Arc<dyn CursorEngine>,
        vec![
            SessionOp::Configure(session.clone()),
            statement("plain", None),
            SessionOp::Execute {
                query: "own".to_string(),
                params: HashMap::new(),
                transaction: None,
                nonce: 0,
                settings: ConnectionSettings {
                    read_concern: Some(3),
                    ..ConnectionSettings::default()
                },
                source: None,
                operation: None,
            },
            SessionOp::Execute {
                query: "in-txn".to_string(),
                params: HashMap::new(),
                transaction: Some(txid(9)),
                nonce: 4,
                settings: ConnectionSettings {
                    write_concern: Some(WriteConcern::majority()),
                    ..ConnectionSettings::default()
                },
                source: Some(Box::new(here.clone())),
                operation: None,
            },
        ],
    )
    .await;

    let Call::Statement {
        settings, source, ..
    } = recording.statement("plain")
    else {
        unreachable!()
    };
    assert_eq!(*settings, session);
    assert_eq!(source, None);
    let Call::Statement { settings, .. } = recording.statement("own") else {
        unreachable!()
    };
    assert_eq!(
        *settings,
        ConnectionSettings {
            read_concern: Some(3),
            ..session.clone()
        }
    );
    let Call::Statement {
        settings,
        source,
        transaction,
        ..
    } = recording.statement("in-txn")
    else {
        unreachable!()
    };
    assert_eq!(transaction, Some((txid(9), 4)));
    assert_eq!(source, Some(here), "the source rides along");
    assert_eq!(
        *settings,
        ConnectionSettings {
            write_concern: Some(WriteConcern::majority()),
            ..session
        }
    );
}

/// The client operation of an autonomous statement and of a commit reaches
/// the engine as sent; one named on a statement inside a transaction is
/// refused before the engine sees the statement, since the transaction's
/// operation is named on its commit.
#[tokio::test]
async fn operations_reach_the_engine_and_a_transaction_statement_names_none() {
    use coordinode_core::group::DeploymentId;
    use coordinode_core::txn::operation::{IssuerId, OperationId};

    let op = |sequence| OperationDescriptor {
        op: OperationId {
            issuer: IssuerId([4; 16]),
            incarnation: 1,
            sequence,
        },
        deployment: DeploymentId([5; 16]),
    };
    let with_op = |query: &str, transaction, operation| SessionOp::Execute {
        query: query.to_string(),
        params: HashMap::new(),
        transaction,
        nonce: 0,
        settings: ConnectionSettings::default(),
        source: None,
        operation,
    };
    let recording = Arc::new(RecordingEngine::default());
    let by_id = run_session(
        Arc::clone(&recording) as Arc<dyn CursorEngine>,
        vec![
            with_op("bound", None, Some(op(1))),
            with_op("unbound", None, None),
            with_op("inside", Some(txid(1)), Some(op(2))),
            SessionOp::Commit {
                transaction: txid(1),
                last_nonce: 0,
                operation: Some(op(3)),
            },
        ],
    )
    .await;

    match by_id[&3].as_slice() {
        [SessionEvent::Error(failure)] => {
            assert_eq!(failure.code, ErrorCode::InvalidArgument);
        }
        other => panic!("a transaction statement naming an operation is refused, got {other:?}"),
    }
    let Call::Statement { operation, .. } = recording.statement("bound") else {
        unreachable!()
    };
    assert_eq!(operation, Some(op(1)));
    let Call::Statement { operation, .. } = recording.statement("unbound") else {
        unreachable!()
    };
    assert_eq!(operation, None);
    let calls = recording.calls();
    assert!(
        !calls
            .iter()
            .any(|call| matches!(call, Call::Statement { query, .. } if query == "inside")),
        "the engine never saw it"
    );
    assert!(calls.contains(&Call::Commit(txid(1), None, Some(op(3)))));
}

/// An engine whose statement works until it is cancelled, checking the
/// switch as a statement's budget does, then answers with a cancellation.
struct UntilCancelled;

impl CursorEngine for UntilCancelled {
    fn open_cursor(
        &self,
        _query: &str,
        _params: HashMap<String, Value>,
        _transaction: Option<&InTransaction>,
        _settings: &ConnectionSettings,
        _source: Option<&StatementSource>,
        _operation: Option<&OperationDescriptor>,
        cancel: &coordinode_core::budget::CancelFlag,
    ) -> Result<Box<dyn QueryCursor>, EngineError> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !cancel.is_cancelled() {
            assert!(
                std::time::Instant::now() < deadline,
                "the Cancel never reached the statement"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        Err(EngineError(Failure::new(
            ErrorCode::Cancelled,
            "the query was cancelled",
        )))
    }

    fn begin_transaction(
        &self,
        _transaction: TransactionId,
        _ordering: TxnOrdering,
        _drain: Option<Duration>,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    fn commit_transaction(
        &self,
        _transaction: TransactionId,
        _last_nonce: Option<u64>,
        _operation: Option<&OperationDescriptor>,
        _admission: &AdmissionSignal,
    ) -> Result<CommitReceipt, EngineError> {
        Ok(MOCK_RECEIPT)
    }

    fn rollback_transaction(
        &self,
        _transaction: TransactionId,
        _admission: &AdmissionSignal,
    ) -> Result<(), EngineError> {
        Ok(())
    }
}

/// A Cancel naming a running statement reaches the engine, which stops the
/// statement; the client gets the cancellation, not a result. A Cancel
/// naming no statement in flight does nothing.
#[tokio::test]
async fn a_cancel_stops_the_statement_it_names() {
    let by_id = run_session(
        Arc::new(UntilCancelled),
        vec![
            exec(),
            SessionOp::Cancel {
                target_request_id: 99,
            },
            SessionOp::Cancel {
                target_request_id: 1,
            },
        ],
    )
    .await;
    assert!(
        matches!(
            by_id[&1].as_slice(),
            [SessionEvent::Error(Failure {
                code: ErrorCode::Cancelled,
                ..
            })]
        ),
        "{:?}",
        by_id[&1]
    );
    assert!(!by_id.contains_key(&2) && !by_id.contains_key(&3));
}

#[tokio::test]
async fn execute_pages_cursor_open_then_rows_then_end() {
    let engine = engine(&["n"], vec![vec![Value::Int(1)], vec![Value::Int(2)]]);
    let events = run_one(engine, exec()).await;
    match events.as_slice() {
        [
            SessionEvent::CursorOpen { columns },
            SessionEvent::Rows { rows },
            SessionEvent::CursorEnd { .. },
        ] => {
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

/// An engine failure reaches the client as the engine reported it: class,
/// message and the binding's opaque details, so a retryable failure is not
/// flattened into an internal one.
#[tokio::test]
async fn execute_surfaces_an_engine_error() {
    let engine: Arc<dyn CursorEngine> = Arc::new(MockEngine {
        columns: vec![],
        rows: vec![],
        fail: Some("boom".to_string()),
    });
    let events = run_one(engine, exec()).await;
    match events.as_slice() {
        [SessionEvent::Error(failure)] => {
            assert_eq!(failure.code, ErrorCode::Unavailable);
            assert_eq!(failure.message, "boom");
            assert_eq!(failure.details, b"opaque");
        }
        other => panic!("expected a single Error, got {other:?}"),
    }
}

/// Begin passes on the client's transaction ID, the ordering and the drain
/// bound (the frame's own, else the session's setting, else none), and a
/// Begin that opens is not answered: the client sends the transaction's
/// statements right after it.
#[tokio::test]
async fn begin_passes_its_id_ordering_and_drain_bound_to_the_engine() {
    let recording = Arc::new(RecordingEngine::default());
    let by_id = run_session(
        Arc::clone(&recording) as Arc<dyn CursorEngine>,
        vec![
            SessionOp::Begin {
                transaction: txid(1),
                ordering: TxnOrdering::Ordered,
                drain_timeout_ms: 0,
            },
            SessionOp::Configure(ConnectionSettings {
                drain_timeout_ms: Some(250),
                ..ConnectionSettings::default()
            }),
            SessionOp::Begin {
                transaction: txid(3),
                ordering: TxnOrdering::Unordered,
                drain_timeout_ms: 0,
            },
        ],
    )
    .await;
    assert!(
        !by_id.contains_key(&1) && !by_id.contains_key(&3),
        "an opened transaction is not answered: {by_id:?}"
    );
    let begins: Vec<Call> = recording
        .calls()
        .into_iter()
        .filter(|call| matches!(call, Call::Begin(..)))
        .collect();
    assert!(begins.contains(&Call::Begin(txid(1), TxnOrdering::Ordered, None)));
    assert!(begins.contains(&Call::Begin(
        txid(3),
        TxnOrdering::Unordered,
        Some(Duration::from_millis(250))
    )));
}

/// A statement sent right after its Begin runs once the transaction is
/// open: the Begin holds the transaction's lane until the engine has opened
/// it, so the statement cannot reach the engine first.
#[tokio::test]
async fn a_statement_sent_right_after_its_begin_runs_after_it() {
    let recording = Arc::new(RecordingEngine::default());
    run_session(
        Arc::clone(&recording) as Arc<dyn CursorEngine>,
        vec![
            SessionOp::Begin {
                transaction: txid(7),
                ordering: TxnOrdering::Unordered,
                drain_timeout_ms: 0,
            },
            statement("first", Some(txid(7))),
        ],
    )
    .await;
    let calls = recording.calls();
    let begin = calls
        .iter()
        .position(|call| matches!(call, Call::Begin(id, ..) if *id == txid(7)))
        .expect("the begin reached the engine");
    let first = calls
        .iter()
        .position(|call| matches!(call, Call::Statement { query, .. } if query == "first"))
        .expect("the statement reached the engine");
    assert!(
        begin < first,
        "the statement ran before its begin: {calls:?}"
    );
}

/// A Begin the engine refuses is answered with the refusal.
#[tokio::test]
async fn a_refused_begin_is_answered() {
    struct Refuses;
    impl CursorEngine for Refuses {
        fn open_cursor(
            &self,
            _query: &str,
            _params: HashMap<String, Value>,
            _transaction: Option<&InTransaction>,
            _settings: &ConnectionSettings,
            _source: Option<&StatementSource>,
            _operation: Option<&OperationDescriptor>,
            _cancel: &coordinode_core::budget::CancelFlag,
        ) -> Result<Box<dyn QueryCursor>, EngineError> {
            Ok(empty_cursor())
        }
        fn begin_transaction(
            &self,
            _transaction: TransactionId,
            _ordering: TxnOrdering,
            _drain: Option<Duration>,
        ) -> Result<(), EngineError> {
            Err(EngineError(Failure::new(
                ErrorCode::AlreadyExists,
                "transaction ID in use",
            )))
        }
        fn commit_transaction(
            &self,
            _transaction: TransactionId,
            _last_nonce: Option<u64>,
            _operation: Option<&OperationDescriptor>,
            _admission: &AdmissionSignal,
        ) -> Result<CommitReceipt, EngineError> {
            Ok(MOCK_RECEIPT)
        }
        fn rollback_transaction(
            &self,
            _transaction: TransactionId,
            _admission: &AdmissionSignal,
        ) -> Result<(), EngineError> {
            Ok(())
        }
    }

    let events = run_one(
        Arc::new(Refuses),
        SessionOp::Begin {
            transaction: txid(1),
            ordering: TxnOrdering::Unordered,
            drain_timeout_ms: 0,
        },
    )
    .await;
    assert!(
        matches!(
            events.as_slice(),
            [SessionEvent::Error(Failure {
                code: ErrorCode::AlreadyExists,
                ..
            })]
        ),
        "{events:?}"
    );
}

#[tokio::test]
async fn commit_acknowledges_and_rollback_cancel_are_silent() {
    let by_id = run_session(
        engine(&[], vec![]),
        vec![
            SessionOp::Commit {
                transaction: txid(1),
                last_nonce: 0,
                operation: None,
            },
            SessionOp::Rollback {
                transaction: txid(2),
            },
            SessionOp::Cancel {
                target_request_id: 1,
            },
        ],
    )
    .await;
    // The engine's receipt reaches the binding untouched: commit timestamp and
    // causal token are the engine's, not something the core re-derives.
    assert!(matches!(
        by_id[&1].as_slice(),
        [SessionEvent::Committed { receipt }] if *receipt == MOCK_RECEIPT
    ));
    assert!(!by_id.contains_key(&2));
    assert!(!by_id.contains_key(&3));
}

/// Every commit is answered, a repeat sent before the first one's answer
/// included: a client retrying a commit whose reply it lost must not wait
/// forever.
#[tokio::test]
async fn a_repeated_commit_is_answered() {
    let commit = SessionOp::Commit {
        transaction: txid(1),
        last_nonce: 0,
        operation: None,
    };
    let by_id = run_session(
        engine(&[], vec![]),
        vec![commit.clone(), commit.clone(), commit],
    )
    .await;
    for rid in [1u64, 2, 3] {
        assert!(
            matches!(
                by_id.get(&rid).map(Vec::as_slice),
                Some([SessionEvent::Committed { .. }])
            ),
            "req {rid}: {:?}",
            by_id.get(&rid)
        );
    }
}

/// An engine that holds a statement before it admits it, until the test
/// lets it go, and records the order statements and commits got their
/// place in.
struct Gated {
    gate: Arc<tokio::sync::Semaphore>,
    order: parking_lot::Mutex<Vec<String>>,
}

impl CursorEngine for Gated {
    fn open_cursor(
        &self,
        query: &str,
        _params: HashMap<String, Value>,
        transaction: Option<&InTransaction>,
        _settings: &ConnectionSettings,
        _source: Option<&StatementSource>,
        _operation: Option<&OperationDescriptor>,
        _cancel: &coordinode_core::budget::CancelFlag,
    ) -> Result<Box<dyn QueryCursor>, EngineError> {
        if query == "held" {
            // Blocking wait on the blocking pool, as the owner's queue does.
            let permit = tokio::runtime::Handle::current()
                .block_on(Arc::clone(&self.gate).acquire_owned())
                .expect("gate");
            drop(permit);
        }
        self.order.lock().push(query.to_string());
        if let Some(transaction) = transaction {
            transaction.admission.admitted();
        }
        Ok(empty_cursor())
    }

    fn begin_transaction(
        &self,
        _transaction: TransactionId,
        _ordering: TxnOrdering,
        _drain: Option<Duration>,
    ) -> Result<(), EngineError> {
        Ok(())
    }

    fn commit_transaction(
        &self,
        _transaction: TransactionId,
        _last_nonce: Option<u64>,
        _operation: Option<&OperationDescriptor>,
        admission: &AdmissionSignal,
    ) -> Result<CommitReceipt, EngineError> {
        self.order.lock().push("commit".to_string());
        admission.admitted();
        Ok(MOCK_RECEIPT)
    }

    fn rollback_transaction(
        &self,
        _transaction: TransactionId,
        _admission: &AdmissionSignal,
    ) -> Result<(), EngineError> {
        Ok(())
    }
}

/// The commands one stream sends for one transaction reach its owner in the
/// order sent: one held before it got its place holds back the next, even
/// when they run on separate tasks; a statement of another transaction, or
/// none, is not held back by it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stream_keeps_its_order_for_each_transaction() {
    let gated = Arc::new(Gated {
        gate: Arc::new(tokio::sync::Semaphore::new(0)),
        order: parking_lot::Mutex::new(Vec::new()),
    });
    let manager = SessionManager::new(
        Arc::clone(&gated) as Arc<dyn CursorEngine>,
        Arc::new(SessionRegistry::new()),
    );
    let session = manager.open("test".into());
    let (op_tx, op_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);
    let run = tokio::spawn(session.run(op_rx, ev_tx));

    let ops = [
        statement("held", Some(txid(1))),
        statement("next", Some(txid(1))),
        SessionOp::Commit {
            transaction: txid(1),
            last_nonce: 0,
            operation: None,
        },
        statement("other", Some(txid(2))),
        statement("free", None),
    ];
    for (i, op) in ops.into_iter().enumerate() {
        op_tx.send((i as u64 + 1, op)).await.unwrap();
    }
    // The unrelated statements finish while the first one is still held.
    let mut done = std::collections::HashSet::new();
    while !(done.contains(&4) && done.contains(&5)) {
        let (rid, event) = ev_rx.recv().await.expect("event");
        if matches!(event, SessionEvent::CursorEnd { .. }) {
            done.insert(rid);
        }
    }
    // Taken before the gate opens, checked after the run ends: a failing
    // check must not leave the held statement blocked forever.
    let mut so_far = gated.order.lock().clone();
    so_far.sort();
    gated.gate.add_permits(1);
    drop(op_tx);
    while ev_rx.recv().await.is_some() {}
    run.await.unwrap();

    assert_eq!(
        so_far,
        ["free", "other"],
        "nothing of the held transaction got ahead of its first statement"
    );
    let order = gated.order.lock().clone();
    let at = |q: &str| order.iter().position(|o| o == q).expect(q);
    assert!(
        at("held") < at("next") && at("next") < at("commit"),
        "{order:?}"
    );
    assert!(
        at("other") < at("held") && at("free") < at("held"),
        "{order:?}"
    );
}

/// Closing the stream rolls nothing back: the transactions it used stay
/// open on their owner for another stream to continue, and the session's
/// attachments go with it.
#[tokio::test]
async fn closing_the_stream_detaches_its_transactions() {
    let recording = Arc::new(RecordingEngine::default());
    let registry = Arc::new(SessionRegistry::new());
    run_session_on(
        Arc::clone(&recording) as Arc<dyn CursorEngine>,
        Arc::clone(&registry),
        vec![
            SessionOp::Begin {
                transaction: txid(5),
                ordering: TxnOrdering::Unordered,
                drain_timeout_ms: 0,
            },
            statement("open", Some(txid(5))),
        ],
    )
    .await;
    assert!(
        !recording
            .calls()
            .iter()
            .any(|call| matches!(call, Call::Rollback(_) | Call::Commit(..))),
        "{:?}",
        recording.calls()
    );
    assert!(registry.sessions().is_empty(), "the session is gone");
}

/// A session's attachments follow its commands: begun and used transactions
/// are attached, and a commit or rollback through it detaches them.
#[tokio::test]
async fn attachments_follow_the_commands_a_session_sends() {
    let registry = Arc::new(SessionRegistry::new());
    let manager = SessionManager::new(engine(&[], vec![]), Arc::clone(&registry));
    let session = manager.open("test".into());
    let (op_tx, op_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);
    let run = tokio::spawn(session.run(op_rx, ev_tx));
    let mut answer = async |rid: u64, op: SessionOp| {
        op_tx.send((rid, op)).await.unwrap();
        loop {
            let (got, event) = ev_rx.recv().await.expect("event");
            if got == rid
                && matches!(
                    event,
                    SessionEvent::CursorEnd { .. } | SessionEvent::Committed { .. }
                )
            {
                return event;
            }
        }
    };

    // A Begin is not answered; the statement after it in the same lane is,
    // once the transaction is open.
    let begun = txid(1);
    op_tx
        .send((
            1,
            SessionOp::Begin {
                transaction: begun,
                ordering: TxnOrdering::Unordered,
                drain_timeout_ms: 0,
            },
        ))
        .await
        .unwrap();
    answer(2, statement("in-begun", Some(begun))).await;
    answer(3, statement("used", Some(txid(42)))).await;
    let attached = |registry: &SessionRegistry| {
        let mut attached = registry.sessions()[0].attached.clone();
        attached.sort();
        attached
    };
    let mut expected = vec![begun.to_string(), txid(42).to_string()];
    expected.sort();
    assert_eq!(attached(&registry), expected);

    answer(
        4,
        SessionOp::Commit {
            transaction: begun,
            last_nonce: 0,
            operation: None,
        },
    )
    .await;
    assert_eq!(attached(&registry), vec![txid(42).to_string()]);

    drop(op_tx);
    while ev_rx.recv().await.is_some() {}
    run.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_dispatches_a_burst_concurrently_with_correlated_cursors() {
    let manager = SessionManager::new(
        engine(&["n"], vec![vec![Value::Int(7)]]),
        Arc::new(SessionRegistry::new()),
    );
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

/// A Configure that sets one setting must leave the others alone, and the
/// answer must report what is in effect rather than what was asked for.
///
/// Restating every setting on each change is what a client does when it cannot
/// trust a partial update, and confirming from its own request is what makes a
/// client believe a change that never took.
#[tokio::test]
async fn configure_changes_one_setting_and_reports_all_of_them() {
    let engine = engine(&["n"], vec![vec![Value::Int(1)]]);
    let events = run_session(
        engine,
        vec![
            SessionOp::Configure(ConnectionSettings {
                read_concern: Some(2),
                write_concern: Some(WriteConcern::w1()),
                ..Default::default()
            }),
            // Changes only the preference; the concerns above must survive.
            SessionOp::Configure(ConnectionSettings {
                read_preference: Some(3),
                ..Default::default()
            }),
            // Asks for nothing: a pure status request.
            SessionOp::Configure(ConnectionSettings::default()),
        ],
    )
    .await;

    let settings_of = |id: u64| match events.get(&id).and_then(|e| e.first()) {
        Some(SessionEvent::ConnectionStatus { settings, .. }) => settings.clone(),
        other => panic!("request {id} answered with {other:?}, expected a ConnectionStatus"),
    };

    let first = settings_of(1);
    assert_eq!(first.read_concern, Some(2));
    assert_eq!(first.write_concern, Some(WriteConcern::w1()));

    let second = settings_of(2);
    assert_eq!(second.read_preference, Some(3));
    assert_eq!(
        (second.read_concern, second.write_concern),
        (Some(2), Some(WriteConcern::w1())),
        "a partial Configure must not reset the settings it did not mention"
    );

    assert_eq!(
        settings_of(3),
        second,
        "an empty Configure asks for the status and changes nothing"
    );
}

/// The status a Configure returns describes the connection, so it carries what
/// the node can do for this client, not just the settings.
#[tokio::test]
async fn configure_answers_with_the_connection_state() {
    let engine = engine(&["n"], vec![vec![Value::Int(1)]]);
    let events = run_one(engine, SessionOp::Configure(ConnectionSettings::default())).await;

    match events.first() {
        Some(SessionEvent::ConnectionStatus { state, .. }) => {
            // A session with no cluster behind it is writable by definition:
            // there is no leadership to lose.
            assert!(state.writable);
            assert!(state.connected);
            assert!(state.served_by_leader);
        }
        other => panic!("expected a ConnectionStatus, got {other:?}"),
    }
}

/// A change in what the connection can do is pushed to the client without
/// being asked for.
///
/// This is the point of the whole frame: a client on a node that lost touch
/// with the cluster holds its session and is told the moment writes can go
/// through again, instead of retrying on a timer whose interval is a guess
/// about how long an election takes. The event carries request id zero
/// because no request produced it.
#[tokio::test]
async fn a_connection_that_regains_a_leader_tells_the_client() {
    let cut_off = ConnectionState {
        writable: false,
        connected: false,
        leader_id: None,
        served_by_leader: false,
        raft_term: 7,
        voters: 3,
        voters_reachable: 1,
    };
    let (state_tx, state_rx) = tokio::sync::watch::channel(cut_off);
    let manager = SessionManager::new(
        engine(&["n"], vec![vec![Value::Int(1)]]),
        Arc::new(SessionRegistry::new()),
    )
    .with_connection(state_rx);
    let session = manager.open("test".into());
    let (op_tx, op_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);
    let handle = tokio::spawn(session.run(op_rx, ev_tx));

    // The cluster reaches a leader again. Nobody asked about it.
    state_tx
        .send(ConnectionState {
            writable: true,
            connected: true,
            leader_id: Some(2),
            served_by_leader: false,
            raft_term: 8,
            voters: 3,
            voters_reachable: 3,
        })
        .unwrap();

    let (request_id, event) = ev_rx.recv().await.expect("the client must be told");
    assert_eq!(request_id, 0, "an unsolicited event answers no request");
    match event {
        SessionEvent::ConnectionStatus { state, .. } => {
            assert!(
                state.writable,
                "the point of the event is that writes work again"
            );
            assert_eq!(state.leader_id, Some(2));
            assert_eq!(state.voters_reachable, 3);
        }
        other => panic!("expected a ConnectionStatus, got {other:?}"),
    }

    drop(op_tx);
    handle.await.unwrap();
}
