use super::*;
use coordinode_core::budget::CancelFlag;
use coordinode_session::{SessionEvent, SessionManager, SessionOp};

/// Build a tempdir-backed database seeded with `n` `:Page {k}` nodes, returned
/// behind the shared lock the cursor engine expects.
fn seeded_db(n: i64) -> Arc<RwLock<Database>> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for k in 0..n {
        db.execute_cypher(&format!("CREATE (n:Page {{k: {k}}})"))
            .expect("seed");
    }
    // Leak the tempdir so the path stays alive for the test's duration; the OS
    // reclaims it on process exit (test-only).
    std::mem::forget(dir);
    Arc::new(RwLock::new(db))
}

impl DatabaseCursorEngine {
    /// An engine over `database`, standalone and with the built-in defaults.
    fn new(database: Arc<RwLock<Database>>) -> Self {
        Self::from_executor(StatementExecutor::new(database))
    }
}

/// The settings of a statement that names none, in a session that set none.
fn unset() -> ConnectionSettings {
    ConnectionSettings::default()
}

/// Drain a cursor in `batch`-sized steps until it returns an empty batch,
/// returning every row in order.
fn drain(cursor: &mut dyn QueryCursor, batch: usize) -> Vec<Vec<Value>> {
    let mut all = Vec::new();
    loop {
        let rows = cursor.next_batch(batch).expect("next_batch");
        if rows.is_empty() {
            break;
        }
        all.extend(rows);
    }
    all
}

#[test]
fn keyset_cursor_drains_every_row_across_small_batches() {
    // A plain label scan routes to the keyset cursor; draining it in batches
    // smaller than the result must still visit every node exactly once, with
    // the column header established before the first batch.
    let engine = DatabaseCursorEngine::new(seeded_db(25));
    let mut cursor = engine
        .open_cursor(
            "MATCH (n:Page) RETURN n.k",
            HashMap::new(),
            None,
            &unset(),
            None,
            None,
            &CancelFlag::new(),
        )
        .expect("open");

    assert_eq!(cursor.columns(), vec!["n.k".to_string()]);

    let rows = drain(cursor.as_mut(), 4);
    let mut seen: Vec<i64> = rows
        .iter()
        .map(|r| match r.first() {
            Some(Value::Int(v)) => *v,
            other => panic!("unexpected cell: {other:?}"),
        })
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..25).collect::<Vec<_>>(), "every node drained once");
}

#[test]
fn keyset_cursor_with_filter_still_drains_all_matches() {
    // A Filter above the scan keeps the plan keyset-eligible; the cursor must
    // refill across pages (filtered pages yield fewer rows) and return exactly
    // the matching nodes.
    let engine = DatabaseCursorEngine::new(seeded_db(40));
    let mut cursor = engine
        .open_cursor(
            "MATCH (n:Page) WHERE n.k >= 30 RETURN n.k",
            HashMap::new(),
            None,
            &unset(),
            None,
            None,
            &CancelFlag::new(),
        )
        .expect("open");

    let rows = drain(cursor.as_mut(), 3);
    let mut seen: Vec<i64> = rows
        .iter()
        .map(|r| match r.first() {
            Some(Value::Int(v)) => *v,
            other => panic!("unexpected cell: {other:?}"),
        })
        .collect();
    seen.sort_unstable();
    assert_eq!(
        seen,
        (30..40).collect::<Vec<_>>(),
        "only matches, all of them"
    );
}

#[test]
fn blocking_plan_routes_to_materialize_and_still_returns_all_rows() {
    // ORDER BY blocks keyset paging, so the engine materializes. The cursor
    // protocol is identical from the caller's side: drain still yields every
    // row, here in sorted order.
    let engine = DatabaseCursorEngine::new(seeded_db(10));
    let mut cursor = engine
        .open_cursor(
            "MATCH (n:Page) RETURN n.k ORDER BY n.k",
            HashMap::new(),
            None,
            &unset(),
            None,
            None,
            &CancelFlag::new(),
        )
        .expect("open");

    let rows = drain(cursor.as_mut(), 4);
    let ordered: Vec<i64> = rows
        .iter()
        .map(|r| match r.first() {
            Some(Value::Int(v)) => *v,
            other => panic!("unexpected cell: {other:?}"),
        })
        .collect();
    assert_eq!(ordered, (0..10).collect::<Vec<_>>(), "sorted, complete");
}

/// A statement whose cancel switch is thrown does not run: the session's
/// Cancel answers it with CANCELLED, in an interactive transaction too.
#[test]
fn a_cancelled_statement_answers_with_a_cancellation() {
    let engine = DatabaseCursorEngine::new(seeded_db(3));
    let cancelled = CancelFlag::new();
    cancelled.cancel();
    let refused = engine
        .open_cursor(
            "MATCH (n:Page) RETURN n.k ORDER BY n.k",
            HashMap::new(),
            None,
            &unset(),
            None,
            None,
            &cancelled,
        )
        .err()
        .expect("cancelled");
    assert_eq!(refused.0.code, coordinode_session::ErrorCode::Cancelled);

    let id = coordinode_embed::db::draw_transaction_id();
    engine.begin_transaction(id, &unset(), None).expect("begin");
    let transaction = coordinode_session::InTransaction {
        id,
        admission: AdmissionSignal::new(|| {}),
    };
    let refused = engine
        .open_cursor(
            "CREATE (:Page {k: 99})",
            HashMap::new(),
            Some(&transaction),
            &unset(),
            None,
            None,
            &cancelled,
        )
        .err()
        .expect("cancelled in a transaction");
    assert_eq!(refused.0.code, coordinode_session::ErrorCode::Cancelled);
}

#[test]
fn keyset_cursor_empty_label_yields_no_rows() {
    let engine = DatabaseCursorEngine::new(seeded_db(0));
    let mut cursor = engine
        .open_cursor(
            "MATCH (n:Page) RETURN n.k",
            HashMap::new(),
            None,
            &unset(),
            None,
            None,
            &CancelFlag::new(),
        )
        .expect("open");
    assert!(cursor.next_batch(8).expect("batch").is_empty());
}

// ----- integration: real engine driven through the session core -----

/// Run one `Execute` op through the full session core (`SessionManager::run` →
/// `dispatch` → real `DatabaseCursorEngine` → keyset cursor paging) and collect
/// the events it streams back. This exercises the production composition the
/// gRPC binding wires up, minus the transport.
async fn run_execute(engine: DatabaseCursorEngine, query: &str) -> Vec<SessionEvent> {
    run_execute_with(engine, query, unset()).await
}

/// [`run_execute`] for a statement under `settings`.
async fn run_execute_with(
    engine: DatabaseCursorEngine,
    query: &str,
    settings: ConnectionSettings,
) -> Vec<SessionEvent> {
    use tokio::sync::mpsc;

    let registry = Arc::new(coordinode_session::SessionRegistry::new());
    let manager = SessionManager::new(Arc::new(engine), registry);
    let (op_tx, op_rx) = mpsc::channel(64);
    let (ev_tx, mut ev_rx) = mpsc::channel(64);
    let handle = tokio::spawn(manager.open("test".into()).run(op_rx, ev_tx));

    op_tx
        .send((
            1,
            SessionOp::Execute {
                query: query.to_string(),
                params: HashMap::new(),
                transaction: None,
                settings,
                source: None,
                operation: None,
            },
        ))
        .await
        .expect("send");
    drop(op_tx);

    let mut events = Vec::new();
    while let Some((_request_id, event)) = ev_rx.recv().await {
        events.push(event);
    }
    handle.await.expect("session task");
    events
}

#[tokio::test]
async fn session_core_streams_keyset_cursor_open_rows_end() {
    // The real keyset cursor, driven through the session core, must stream a
    // CursorOpen (with the column header), one or more Rows batches covering
    // every node, then CursorEnd. The result spans many internal pages
    // (KEYSET_PAGE) but the protocol is identical to a small one.
    let events = run_execute(
        DatabaseCursorEngine::new(seeded_db(50)),
        "MATCH (n:Page) RETURN n.k",
    )
    .await;

    let mut iter = events.iter();
    match iter.next() {
        Some(SessionEvent::CursorOpen { columns }) => {
            assert_eq!(columns.as_slice(), ["n.k".to_string()]);
        }
        other => panic!("expected CursorOpen first, got {other:?}"),
    }

    let mut seen: Vec<i64> = Vec::new();
    let mut ended = false;
    for event in iter {
        match event {
            SessionEvent::Rows { rows } => {
                for row in rows {
                    match row.first() {
                        Some(Value::Int(v)) => seen.push(*v),
                        other => panic!("unexpected cell: {other:?}"),
                    }
                }
            }
            SessionEvent::CursorEnd { .. } => {
                ended = true;
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert!(ended, "stream must terminate with CursorEnd");
    seen.sort_unstable();
    assert_eq!(
        seen,
        (0..50).collect::<Vec<_>>(),
        "every node streamed once"
    );
}

#[tokio::test]
async fn session_core_streams_empty_keyset_result_without_rows() {
    let events = run_execute(
        DatabaseCursorEngine::new(seeded_db(0)),
        "MATCH (n:Page) RETURN n.k",
    )
    .await;
    assert!(
        matches!(
            events.as_slice(),
            [
                SessionEvent::CursorOpen { .. },
                SessionEvent::CursorEnd { .. }
            ]
        ),
        "empty result is open-then-end, no Rows: {events:?}"
    );
}

#[tokio::test]
async fn session_core_surfaces_a_query_error() {
    // A semantically invalid query fails at open_cursor; the session core must
    // surface a single Error event rather than a partial cursor stream.

    let events = run_execute(
        DatabaseCursorEngine::new(seeded_db(1)),
        "MATCH (n) RETURN undefined_var",
    )
    .await;
    // The class is the one the unary path answers with for the same query: a
    // fault in the request, not in the server.
    let expected = crate::services::cypher::db_error_to_status(
        seeded_db(1)
            .read()
            .execute_cypher_shared("MATCH (n) RETURN undefined_var", None, None, None, None)
            .expect_err("the query is invalid"),
    );
    match events.as_slice() {
        [SessionEvent::Error(failure)] => {
            assert_ne!(failure.code, coordinode_session::ErrorCode::Internal);
            assert_eq!(failure.code, super::super::failure(&expected).code);
            assert_eq!(failure.details, expected.details());
        }
        other => panic!("expected a single Error, got {other:?}"),
    }
}

/// A session statement's source reaches the advisor as a unary statement's
/// metadata does: counted at its file, line and function, with the
/// application its session named, and a Windows path read as the same file
/// written with `/`. Before, the session recorded nothing for the advisor.
#[test]
fn a_session_statement_is_counted_where_it_was_issued() {
    use coordinode_query::advisor::QueryRegistry;
    use coordinode_query::advisor::nplus1::NPlus1Detector;

    let registry = Arc::new(QueryRegistry::new());
    let engine = DatabaseCursorEngine::from_executor(
        StatementExecutor::new(seeded_db(1))
            .with_advisor(Arc::clone(&registry), Arc::new(NPlus1Detector::new())),
    );
    let source = StatementSource {
        file: "src\\pay.rs".to_string(),
        line: 12,
        function: "charge".to_string(),
        app: "billing".to_string(),
        version: "1.2".to_string(),
    };
    for query in ["MATCH (n:Page) RETURN n.k", "RETURN 1 AS one"] {
        let mut cursor = engine
            .open_cursor(
                query,
                HashMap::new(),
                None,
                &unset(),
                Some(&source),
                None,
                &CancelFlag::new(),
            )
            .expect("open");
        drain(cursor.as_mut(), 8);
    }

    let top = registry.top_by_count(10);
    assert_eq!(
        top.len(),
        2,
        "the keyset and the materialized path both count"
    );
    for entry in &top {
        let src = &entry.sources[0];
        assert_eq!(src.file, "src/pay.rs");
        assert_eq!(src.line, 12);
        assert_eq!(src.function, "charge");
        assert_eq!(src.app, "billing");
    }
}

/// A Cypher SET on a session changes that session's settings: the session
/// reports the new setting, and the database's own, which every other client
/// reads, stays as it was. Before, it changed the database for all clients.
#[tokio::test]
async fn a_cypher_set_changes_its_session_not_the_database() {
    use coordinode_core::graph::types::VectorConsistencyMode;
    use tokio::sync::mpsc;

    let database = seeded_db(0);
    let registry = Arc::new(coordinode_session::SessionRegistry::new());
    let manager = SessionManager::new(
        Arc::new(DatabaseCursorEngine::new(Arc::clone(&database))),
        registry,
    );
    let (op_tx, op_rx) = mpsc::channel(8);
    let (ev_tx, mut ev_rx) = mpsc::channel(8);
    let handle = tokio::spawn(manager.open("test".into()).run(op_rx, ev_tx));
    op_tx
        .send((
            1,
            SessionOp::Execute {
                query: "SET vector_consistency = 'exact'".to_string(),
                params: HashMap::new(),
                transaction: None,
                settings: unset(),
                source: None,
                operation: None,
            },
        ))
        .await
        .expect("send");
    op_tx
        .send((2, SessionOp::Configure(unset())))
        .await
        .expect("send");
    drop(op_tx);

    let mut status = None;
    while let Some((request_id, event)) = ev_rx.recv().await {
        if let (2, SessionEvent::ConnectionStatus { settings, .. }) = (request_id, event) {
            status = Some(settings);
        }
    }
    handle.await.expect("session task");
    assert_eq!(
        status.expect("a status").vector_consistency,
        Some(VectorConsistencyMode::Exact),
        "the session holds the setting"
    );
    assert_eq!(
        database.read().vector_consistency(),
        VectorConsistencyMode::default(),
        "the database's own setting is untouched"
    );
}

/// A session statement is checked with the settings it runs under, as a unary
/// one is: a causal write under a concern that can be lost is refused before
/// anything is written, and the same write under the majority default runs.
/// Before, the session ran every statement on the defaults and never looked.
#[tokio::test]
async fn a_session_write_is_checked_with_its_settings() {
    use coordinode_core::txn::write_concern::WriteConcern;

    let database = seeded_db(0);
    let causal = |write_concern| ConnectionSettings {
        read_concern: Some(2), // MAJORITY
        after_index: Some(1),
        write_concern,
        ..unset()
    };

    let events = run_execute_with(
        DatabaseCursorEngine::new(Arc::clone(&database)),
        "CREATE (:Causal)",
        causal(Some(WriteConcern::w1())),
    )
    .await;
    match events.as_slice() {
        [SessionEvent::Error(failure)] => {
            assert_eq!(
                failure.code,
                coordinode_session::ErrorCode::FailedPrecondition
            );
            assert!(failure.message.contains("writeConcern"), "{failure:?}");
        }
        other => panic!("expected the write refused, got {other:?}"),
    }
    let count = |db: &Arc<RwLock<Database>>| {
        db.write()
            .execute_cypher("MATCH (n:Causal) RETURN count(n) AS c")
            .expect("count")[0]
            .get("c")
            .cloned()
    };
    assert_eq!(count(&database), Some(Value::Int(0)), "nothing was written");

    let events = run_execute_with(
        DatabaseCursorEngine::new(Arc::clone(&database)),
        "CREATE (:Causal)",
        causal(None),
    )
    .await;
    assert!(
        matches!(events.last(), Some(SessionEvent::CursorEnd { .. })),
        "{events:?}"
    );
    assert_eq!(count(&database), Some(Value::Int(1)));
}

/// A session statement passed to the leader stops waiting when its Cancel is
/// thrown: the call to the leader is dropped, which cancels the leader's
/// statement, and the session answers CANCELLED. Before, the session waited
/// for the leader's whole answer whatever the client asked. The leader here
/// accepts the connection and never answers, so only the Cancel ends it.
#[tokio::test(flavor = "multi_thread")]
async fn a_cancel_stops_a_statement_waiting_on_the_leader() {
    use coordinode_raft::cluster::RaftNode;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };

    let silent = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let silent_addr = silent.local_addr().expect("addr");
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((stream, _)) = silent.accept().await {
            held.push(stream);
        }
    });

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let storage = Arc::new(
        coordinode_storage::engine::core::StorageEngine::open_with_oracle(
            &config,
            Arc::clone(&oracle),
        )
        .expect("engine"),
    );
    let (node, _handler) = RaftNode::open_cluster_embedded(
        1,
        Arc::clone(&storage),
        silent_addr.to_string(),
        coordinode_test_fixtures::TestPki::new().connections(1),
    )
    .await
    .expect("raft");
    let node = Arc::new(node);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> = Arc::new(
        coordinode_raft::proposal::RaftProposalPipeline::new(Arc::clone(node.raft())),
    );
    let database = Database::from_engine(dir.path(), storage, oracle, pipeline).expect("db");
    let engine = Arc::new(DatabaseCursorEngine::from_executor(
        StatementExecutor::new(Arc::new(RwLock::new(database))).with_raft_node(Arc::clone(&node)),
    ));

    let cancel = CancelFlag::new();
    let waiting = {
        let engine = Arc::clone(&engine);
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || {
            let statement = DatabaseCursorEngine::autonomous(
                "RETURN 1 AS one",
                None,
                &crate::services::statement::Requested::default(),
                None,
            );
            engine
                .routed(
                    1,
                    statement,
                    &coordinode_core::txn::command::AdmissionSignal::new(|| {}),
                    &cancel,
                )
                .err()
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(!waiting.is_finished(), "the leader has not answered");
    cancel.cancel();
    let refused = tokio::time::timeout(std::time::Duration::from_secs(10), waiting)
        .await
        .expect("answered once cancelled")
        .expect("task")
        .expect("refused");
    assert_eq!(refused.0.code, coordinode_session::ErrorCode::Cancelled);

    node.shutdown().await.expect("shutdown");
}

/// In a cluster a session statement goes through the read fence with the
/// preference it runs under, and its statistics say where it was served: a
/// SECONDARY read is refused on the leader, a default one is served there,
/// with the applied index a causal client continues from.
#[tokio::test(flavor = "multi_thread")]
async fn a_session_read_is_fenced_and_says_where_it_was_served() {
    let (executor, node, _dir) = crate::services::statement::tests::raft().await;

    let events = run_execute_with(
        DatabaseCursorEngine::from_executor(executor.clone()),
        "RETURN 1 AS one",
        ConnectionSettings {
            read_preference: Some(3), // SECONDARY
            ..unset()
        },
    )
    .await;
    match events.as_slice() {
        [SessionEvent::Error(failure)] => assert_eq!(
            failure.code,
            coordinode_session::ErrorCode::FailedPrecondition
        ),
        other => panic!("expected the secondary read refused, got {other:?}"),
    }

    let events = run_execute_with(
        DatabaseCursorEngine::from_executor(executor),
        "RETURN 1 AS one",
        unset(),
    )
    .await;
    match events.last() {
        Some(SessionEvent::CursorEnd { stats }) => {
            assert!(stats.served_by_leader, "{stats:?}");
            assert!(stats.applied_index > 0, "{stats:?}");
        }
        other => panic!("expected the read served, got {other:?}"),
    }

    node.shutdown().await.expect("shutdown");
}

/// A session is driven one request at a time: each op is sent and every event
/// up to the one that ends it is collected before the next.
struct Driver {
    ops: tokio::sync::mpsc::Sender<(u64, SessionOp)>,
    events: tokio::sync::mpsc::Receiver<(u64, SessionEvent)>,
    next_id: u64,
}

impl Driver {
    fn open(engine: DatabaseCursorEngine) -> Self {
        Self::open_shared(Arc::new(engine))
    }

    /// A session over `engine`, which other sessions may share: the streams
    /// of one client reaching the same server.
    fn open_shared(engine: Arc<DatabaseCursorEngine>) -> Self {
        let registry = Arc::new(coordinode_session::SessionRegistry::new());
        let manager = SessionManager::new(engine, registry);
        let (ops, op_rx) = tokio::sync::mpsc::channel(8);
        let (ev_tx, events) = tokio::sync::mpsc::channel(64);
        tokio::spawn(manager.open("test".into()).run(op_rx, ev_tx));
        Self {
            ops,
            events,
            next_id: 1,
        }
    }

    /// Begin a transaction under a newly drawn ID, as a client does: the
    /// Begin is not answered when it opens, and the transaction's next
    /// command waits for it on the session.
    async fn begin(&mut self) -> TransactionId {
        let transaction = coordinode_embed::db::draw_transaction_id();
        self.begin_with(transaction, None).await;
        transaction
    }

    /// Begin `transaction` carrying out `operation`; nothing answers it.
    async fn begin_with(
        &mut self,
        transaction: TransactionId,
        operation: Option<coordinode_embed::operation::OperationDescriptor>,
    ) {
        let id = self.next_id;
        self.next_id += 1;
        self.ops
            .send((
                id,
                SessionOp::Begin {
                    transaction,
                    operation,
                },
            ))
            .await
            .expect("send");
    }

    /// Change the session's settings and wait for the status confirming it.
    async fn send_configure(&mut self, change: ConnectionSettings) {
        let id = self.next_id;
        self.next_id += 1;
        self.ops
            .send((id, SessionOp::Configure(change)))
            .await
            .expect("send");
        loop {
            let received =
                tokio::time::timeout(std::time::Duration::from_secs(30), self.events.recv()).await;
            match received {
                Ok(Some((got, SessionEvent::ConnectionStatus { .. }))) if got == id => return,
                Ok(Some(_)) => {}
                other => panic!("the Configure was not confirmed: {other:?}"),
            }
        }
    }

    /// Send `op` and collect its answer; a request left unanswered fails the
    /// test naming it rather than hanging the run.
    async fn send(&mut self, op: SessionOp) -> Vec<SessionEvent> {
        let id = self.next_id;
        self.next_id += 1;
        let asked = format!("{op:?}");
        self.ops.send((id, op)).await.expect("send");
        let mut out = Vec::new();
        loop {
            let received =
                tokio::time::timeout(std::time::Duration::from_secs(30), self.events.recv()).await;
            let Ok(next) = received else {
                panic!("request {id} unanswered after 30 s: {asked}; got so far {out:?}");
            };
            let Some((request_id, event)) = next else {
                break;
            };
            if request_id != id {
                continue;
            }
            let last = matches!(
                event,
                SessionEvent::CursorEnd { .. }
                    | SessionEvent::Error(_)
                    | SessionEvent::Committed { .. }
            );
            out.push(event);
            if last {
                return out;
            }
        }
        panic!("the session ended before request {id} was answered: {out:?}");
    }

    /// Send every op of `ops` without waiting between them, as a pipelining
    /// client does, and collect each one's answer.
    async fn send_all(&mut self, ops: Vec<SessionOp>) -> Vec<Vec<SessionEvent>> {
        let first = self.next_id;
        for op in ops {
            let id = self.next_id;
            self.next_id += 1;
            self.ops.send((id, op)).await.expect("send");
        }
        let count = (self.next_id - first) as usize;
        let mut answers: Vec<Vec<SessionEvent>> = vec![Vec::new(); count];
        let mut open = count;
        while open > 0 {
            let received =
                tokio::time::timeout(std::time::Duration::from_secs(30), self.events.recv()).await;
            let Ok(Some((request_id, event))) = received else {
                panic!("requests unanswered after 30 s: {answers:?}");
            };
            let Some(slot) = request_id
                .checked_sub(first)
                .and_then(|i| answers.get_mut(i as usize))
            else {
                continue;
            };
            let last = matches!(
                event,
                SessionEvent::CursorEnd { .. }
                    | SessionEvent::Error(_)
                    | SessionEvent::Committed { .. }
            );
            slot.push(event);
            if last {
                open -= 1;
            }
        }
        answers
    }

    async fn execute(
        &mut self,
        query: &str,
        transaction: Option<TransactionId>,
        operation: Option<coordinode_embed::operation::OperationDescriptor>,
    ) -> Vec<SessionEvent> {
        self.send(statement(query, transaction, operation)).await
    }
}

/// A statement as a session sends it, naming no settings of its own.
fn statement(
    query: &str,
    transaction: Option<TransactionId>,
    operation: Option<coordinode_embed::operation::OperationDescriptor>,
) -> SessionOp {
    SessionOp::Execute {
        query: query.to_string(),
        params: HashMap::new(),
        transaction,
        settings: unset(),
        source: None,
        operation,
    }
}

/// A commit as a session sends it.
fn commit(transaction: TransactionId) -> SessionOp {
    SessionOp::Commit {
        transaction,
        operation: None,
        expect: Vec::new(),
    }
}

/// The single row's single value a statement answered with.
fn only_value(events: &[SessionEvent]) -> Value {
    let rows: Vec<&Vec<Value>> = events
        .iter()
        .filter_map(|event| match event {
            SessionEvent::Rows { rows } => Some(rows),
            _ => None,
        })
        .flatten()
        .collect();
    match rows.as_slice() {
        [row] => row[0].clone(),
        other => panic!("expected one row, got {other:?} from {events:?}"),
    }
}

/// A transaction outlives the stream that began it: closing that stream
/// between commands rolls nothing back, and another stream continues it by
/// its ID, with the same pinned view and its own writes, then commits it.
#[tokio::test(flavor = "multi_thread")]
async fn a_transaction_continues_on_another_stream_after_the_first_closes() {
    let database = seeded_db(0);
    let engine = Arc::new(DatabaseCursorEngine::new(Arc::clone(&database)));

    let mut first = Driver::open_shared(Arc::clone(&engine));
    let transaction = first.begin().await;
    let wrote = first
        .execute("CREATE (:Moved {k: 1})", Some(transaction), None)
        .await;
    assert!(
        matches!(wrote.last(), Some(SessionEvent::CursorEnd { .. })),
        "{wrote:?}"
    );
    // A write committed elsewhere after BEGIN is outside the pinned view.
    database
        .write()
        .execute_cypher("CREATE (:Moved {k: 2})")
        .expect("a later write");
    drop(first);

    let mut second = Driver::open_shared(engine);
    let seen = second
        .execute(
            "MATCH (n:Moved) RETURN count(n) AS c",
            Some(transaction),
            None,
        )
        .await;
    assert_eq!(
        only_value(&seen),
        Value::Int(1),
        "its own write, and not the one committed after it began"
    );
    let committed = second.send(commit(transaction)).await;
    assert!(
        matches!(committed.as_slice(), [SessionEvent::Committed { .. }]),
        "{committed:?}"
    );
    drop(second);

    let rows = database
        .write()
        .execute_cypher("MATCH (n:Moved) RETURN n.k AS k")
        .expect("read");
    assert_eq!(rows.len(), 2, "the transaction's write and the later one");
}

/// A transaction's statements pipelined on one stream, without waiting for
/// answers, run in the order they were sent, and a commit sent with them
/// commits after all of them.
#[tokio::test(flavor = "multi_thread")]
async fn a_pipelined_chain_runs_in_the_order_it_was_sent() {
    let database = seeded_db(0);
    let mut session = Driver::open(DatabaseCursorEngine::new(Arc::clone(&database)));
    let transaction = session.begin().await;
    let answers = session
        .send_all(vec![
            statement("CREATE (:Chain {step: 1})", Some(transaction), None),
            statement(
                "MATCH (n:Chain) SET n.step = n.step * 10",
                Some(transaction),
                None,
            ),
            commit(transaction),
        ])
        .await;
    for (i, answer) in answers.iter().enumerate().take(2) {
        assert!(
            matches!(answer.last(), Some(SessionEvent::CursorEnd { .. })),
            "statement {i}: {answer:?}"
        );
    }
    assert!(
        matches!(answers[2].as_slice(), [SessionEvent::Committed { .. }]),
        "{:?}",
        answers[2]
    );
    drop(session);

    let rows = database
        .write()
        .execute_cypher("MATCH (n:Chain) RETURN n.step AS step")
        .expect("read");
    assert_eq!(
        rows[0].get("step"),
        Some(&Value::Int(10)),
        "the first statement ran before the second"
    );
}

/// A transaction runs under the session's settings in effect at its Begin:
/// one that commits at idle expiry, begun on a session configured so, is
/// committed by the idle sweep under the operation its Begin named.
#[tokio::test(flavor = "multi_thread")]
async fn a_begin_takes_its_timeout_outcome_from_the_session() {
    use coordinode_core::txn::command::TimeoutOutcome;
    use coordinode_embed::operation::IssuerId;

    let database = seeded_db(0);
    let operation = database
        .read()
        .enroll_issuer(IssuerId([0x18; 16]))
        .expect("enroll")
        .next()
        .expect("number");
    let mut session = Driver::open(DatabaseCursorEngine::new(Arc::clone(&database)));
    session
        .send_configure(ConnectionSettings {
            timeout_outcome: Some(TimeoutOutcome::Commit),
            ..unset()
        })
        .await;
    let transaction = coordinode_embed::db::draw_transaction_id();
    session.begin_with(transaction, Some(operation)).await;
    let wrote = session
        .execute("CREATE (:Expired {k: 1})", Some(transaction), None)
        .await;
    assert!(
        matches!(wrote.last(), Some(SessionEvent::CursorEnd { .. })),
        "{wrote:?}"
    );
    drop(session);

    database
        .read()
        .reap_idle_transactions(std::time::Duration::ZERO);
    let rows = database
        .write()
        .execute_cypher("MATCH (n:Expired) RETURN n")
        .expect("read");
    assert_eq!(rows.len(), 1, "committed at expiry, as the session said");
}

/// The reason and metadata a session failure carries in its details.
fn settled_reason(events: &[SessionEvent]) -> (String, std::collections::HashMap<String, String>) {
    use tonic_types::StatusExt;
    let [SessionEvent::Error(failure)] = events else {
        panic!("expected one refusal, got {events:?}");
    };
    assert_eq!(
        failure.code,
        coordinode_session::ErrorCode::FailedPrecondition
    );
    let status = tonic::Status::with_details(
        tonic::Code::FailedPrecondition,
        failure.message.clone(),
        failure.details.clone().into(),
    );
    let info = status
        .get_details_error_info()
        .expect("a refusal carries ErrorInfo");
    (info.reason, info.metadata)
}

/// A session write carrying its operation runs once: a statement sent again
/// and a commit repeated after its transaction is gone are both answered
/// with the outcome of the first attempt, and nothing is written twice. This
/// is the path a client on a session takes after a lost reply.
#[tokio::test(flavor = "multi_thread")]
async fn a_session_write_retried_with_its_operation_runs_once() {
    use coordinode_embed::operation::IssuerId;

    let database = seeded_db(0);
    let issuer = database
        .read()
        .enroll_issuer(IssuerId([0x17; 16]))
        .expect("enroll");
    let statement_op = issuer.next().expect("number");
    let commit_op = issuer.next().expect("number");
    let mut session = Driver::open(DatabaseCursorEngine::new(Arc::clone(&database)));

    let first = session
        .execute("CREATE (:Paid {k: 1})", None, Some(statement_op))
        .await;
    let Some(SessionEvent::CursorEnd { stats }) = first.last() else {
        panic!("expected the write to run, got {first:?}");
    };
    let statement_ts = stats.commit_ts;
    let again = session
        .execute("CREATE (:Paid {k: 1})", None, Some(statement_op))
        .await;
    let (why, metadata) = settled_reason(&again);
    assert_eq!(why, "OPERATION_SETTLED");
    assert_eq!(metadata["outcome"], "COMMITTED");
    assert_eq!(metadata["commit_ts"], statement_ts.to_string());

    let transaction = session.begin().await;
    let inside = session
        .execute("CREATE (:Basket {k: 1})", Some(transaction), None)
        .await;
    assert!(
        matches!(inside.last(), Some(SessionEvent::CursorEnd { .. })),
        "{inside:?}"
    );
    let commit = SessionOp::Commit {
        transaction,
        operation: Some(commit_op),
        expect: Vec::new(),
    };
    let committed = session.send(commit.clone()).await;
    let [SessionEvent::Committed { receipt }] = committed.as_slice() else {
        panic!("expected the commit, got {committed:?}");
    };
    let commit_ts = receipt.commit_ts.as_raw();
    let repeated = session.send(commit).await;
    let (why, metadata) = settled_reason(&repeated);
    assert_eq!(why, "OPERATION_SETTLED");
    assert_eq!(metadata["outcome"], "COMMITTED");
    assert_eq!(metadata["commit_ts"], commit_ts.to_string());

    drop(session);
    for (label, expected) in [("Paid", 1), ("Basket", 1)] {
        let rows = database
            .try_write_for(std::time::Duration::from_secs(30))
            .expect("nothing holds the database once the session is idle")
            .execute_cypher(&format!("MATCH (n:{label}) RETURN n"))
            .expect("count");
        assert_eq!(rows.len(), expected, "{label}");
    }
}
