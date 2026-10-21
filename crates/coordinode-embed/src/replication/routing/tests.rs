use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;

use super::*;

/// The deadline the tests' commits are sent with.
const COMMIT_DEADLINE: Duration = Duration::from_secs(10);

/// What every statement the scripted executor runs answers.
#[derive(Clone)]
enum Script {
    /// These rows.
    Rows(RoutedAnswer),
    /// The error this builds.
    Refuse(fn() -> DatabaseError),
    /// These rows, then the error this builds.
    RowsThenRefuse(RoutedAnswer, fn() -> DatabaseError),
    /// `rows` rows of one value, [`noise`] `width` bytes long, made as they
    /// are written.
    Many { rows: usize, width: usize },
}

/// `width` bytes that do not compress, the same for the same `seed`: an
/// answer whose size on the wire is its size here.
fn noise(seed: u64, width: usize) -> Value {
    // xorshift64*, never zero for a seed that is not.
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let bytes = (0..width)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
        })
        .collect();
    Value::Blob(bytes)
}

/// An executor that answers from what the test gave it and records what
/// arrived.
#[derive(Default)]
struct Scripted {
    /// The answer every statement gets; none runs it until cancelled.
    answer: Mutex<Option<Script>>,
    /// Each statement as it arrived.
    statements: Mutex<Vec<RoutedStatement>>,
    /// Held by a transaction's statement until the test lets it take its
    /// place: (taken, released).
    gate: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    /// Set when a statement ran to its end with its cancel switch thrown.
    cancelled: AtomicBool,
    /// How long a commit takes before it commits.
    commit_delay: Mutex<Duration>,
    /// Commits that took effect.
    commits: AtomicUsize,
    /// Rollbacks that arrived, in order.
    rollbacks: Mutex<Vec<RoutedRollback>>,
    /// Rows written to answers so far.
    written: AtomicUsize,
}

impl Scripted {
    /// Write `rows` under `columns` to `answer`, counting each.
    fn write(&self, columns: &[String], rows: &[Vec<Value>], answer: &mut AnswerWriter) {
        if !answer.columns(columns) {
            return;
        }
        for row in rows {
            self.written.fetch_add(1, Ordering::AcqRel);
            if !answer.row(row) {
                return;
            }
        }
    }
}

impl CommandExecutor for Scripted {
    fn execute(
        &self,
        statement: RoutedStatement,
        cancel: &CancelFlag,
        admitted: &AdmissionSignal,
        answer: &mut AnswerWriter,
    ) -> Result<RoutedStats, DatabaseError> {
        let in_transaction = statement.transaction.is_some();
        self.statements.lock().unwrap().push(statement);
        if in_transaction {
            if let Some(gate) = self.gate.lock().unwrap().take() {
                gate.recv().expect("the test lets the statement on");
            }
            admitted.admitted();
        }
        // A statement told to stop by its ingress leaving runs until the
        // switch is thrown.
        if self.answer.lock().unwrap().is_none() {
            while !cancel.is_cancelled() {
                std::thread::sleep(Duration::from_millis(5));
            }
            self.cancelled.store(true, Ordering::Release);
            return Err(DatabaseError::Budget(
                coordinode_core::budget::BudgetStop::Cancelled,
            ));
        }
        let script = self.answer.lock().unwrap().clone().expect("an answer");
        match script {
            Script::Rows(rows) => {
                self.write(&rows.columns, &rows.rows, answer);
                Ok(rows.stats)
            }
            Script::Refuse(refusal) => Err(refusal()),
            Script::RowsThenRefuse(rows, refusal) => {
                self.write(&rows.columns, &rows.rows, answer);
                Err(refusal())
            }
            Script::Many { rows, width } => {
                if answer.columns(&["b".to_owned()]) {
                    for k in 0..rows {
                        self.written.fetch_add(1, Ordering::AcqRel);
                        if !answer.row(&[noise(k as u64 + 1, width)]) {
                            break;
                        }
                    }
                }
                Ok(RoutedStats::default())
            }
        }
    }

    fn begin(&self, begin: RoutedBegin) -> Result<(), DatabaseError> {
        assert_eq!(begin.transaction, txid());
        assert_eq!(begin.actor, principal());
        // The concerns the ingress chose cross as chosen; one left unset is
        // not filled in on the way.
        assert_eq!(begin.read_concern, Some(ReadConcernLevel::Snapshot));
        assert_eq!(begin.write_concern, None);
        assert_eq!(begin.timeout_outcome, Some(TimeoutOutcome::Commit));
        assert_eq!(begin.operation, Some(operation(1)));
        Ok(())
    }

    fn commit(&self, commit: RoutedCommit) -> Result<CommitReceipt, DatabaseError> {
        assert_eq!(commit.transaction, txid());
        assert_eq!(commit.expect, expectations());
        // The time left is never more than the ingress gave the commit.
        assert!(
            commit
                .deadline
                .is_some_and(|left| left <= COMMIT_DEADLINE && !left.is_zero()),
            "{:?}",
            commit.deadline
        );
        std::thread::sleep(*self.commit_delay.lock().unwrap());
        self.commits.fetch_add(1, Ordering::AcqRel);
        Ok(CommitReceipt {
            commit_ts: Timestamp::from_raw(77),
            applied_index: Some(12),
        })
    }

    fn rollback(&self, rollback: RoutedRollback) -> Result<(), DatabaseError> {
        assert_eq!(rollback.transaction, txid());
        let id = rollback.transaction;
        self.rollbacks.lock().unwrap().push(rollback);
        Err(DatabaseError::UnknownTransaction(id))
    }

    /// Answers each kind of call with something that names what arrived, so
    /// a test sees the call crossed whole.
    fn operation(
        &self,
        actor: ActingContext,
        call: RoutedOperationCall,
    ) -> Result<RoutedOperationAnswer, DatabaseError> {
        assert_eq!(actor, principal());
        Ok(match call {
            RoutedOperationCall::Enroll(issuer) => RoutedOperationAnswer::Enrolled {
                incarnation: u64::from(issuer.0[0]),
                next_sequence: 1,
                deployment: deployment(),
            },
            RoutedOperationCall::Resolve(descriptor, Resolve::Exclude) => {
                RoutedOperationAnswer::Outcome(OperationOutcome::Committed {
                    attempt: descriptor.op.sequence,
                    commit_ts: 99,
                })
            }
            RoutedOperationCall::Resolve(_, Resolve::Lookup) => {
                RoutedOperationAnswer::Outcome(OperationOutcome::Unknown)
            }
            RoutedOperationCall::Acknowledge(descriptors) => {
                assert_eq!(descriptors, vec![operation(3), operation(4)]);
                RoutedOperationAnswer::Done
            }
            RoutedOperationCall::Retire(below) => {
                assert_eq!(below, operation(10));
                RoutedOperationAnswer::Outcome(OperationOutcome::Retired)
            }
        })
    }
}

fn deployment() -> DeploymentId {
    DeploymentId([9; 16])
}

fn operation(sequence: u64) -> OperationDescriptor {
    OperationDescriptor {
        op: coordinode_core::txn::operation::OperationId {
            issuer: IssuerId([5; 16]),
            incarnation: 2,
            sequence,
        },
        deployment: deployment(),
    }
}

/// Every kind of call on operation records reaches the executing member as
/// sent, acting for the caller, and every kind of answer comes back.
#[tokio::test]
async fn operation_calls_cross_both_ways() {
    let router = serve(Arc::new(Scripted::default())).await;
    let actor = principal();

    assert_eq!(
        router
            .operation(&actor, RoutedOperationCall::Enroll(IssuerId([7; 16])))
            .await
            .expect("enrolled"),
        RoutedOperationAnswer::Enrolled {
            incarnation: 7,
            next_sequence: 1,
            deployment: deployment(),
        }
    );
    assert_eq!(
        router
            .operation(
                &actor,
                RoutedOperationCall::Resolve(operation(6), Resolve::Exclude)
            )
            .await
            .expect("resolved"),
        RoutedOperationAnswer::Outcome(OperationOutcome::Committed {
            attempt: 6,
            commit_ts: 99,
        })
    );
    assert_eq!(
        router
            .operation(
                &actor,
                RoutedOperationCall::Resolve(operation(6), Resolve::Lookup)
            )
            .await
            .expect("looked up"),
        RoutedOperationAnswer::Outcome(OperationOutcome::Unknown)
    );
    assert_eq!(
        router
            .operation(
                &actor,
                RoutedOperationCall::Acknowledge(vec![operation(3), operation(4)])
            )
            .await
            .expect("released"),
        RoutedOperationAnswer::Done
    );
    assert_eq!(
        router
            .operation(&actor, RoutedOperationCall::Retire(operation(10)))
            .await
            .expect("retired"),
        RoutedOperationAnswer::Outcome(OperationOutcome::Retired)
    );
}

fn txid() -> TransactionId {
    TransactionId::from_raw([0x2a; 16])
}

/// A condition on a version and one requiring a node to be absent.
fn expectations() -> Vec<ExpectedVersion> {
    vec![
        ExpectedVersion {
            node: NodeId::from_raw(4),
            version: Some(19),
        },
        ExpectedVersion {
            node: NodeId::from_raw(5),
            version: None,
        },
    ]
}

fn principal() -> ActingContext {
    ActingContext::Principal(Principal {
        subject: "alice".into(),
        trust_scope: "idp".into(),
        namespace: "ns".into(),
    })
}

/// Serve `executor` on a local port and return a router to it.
async fn serve(executor: Arc<Scripted>) -> CommandRouter {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("address");
    let incoming = tonic::transport::server::TcpIncoming::from(listener);
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(CommandRoutingHandler::new(executor).into_service(MAX_ROUTED_MESSAGE))
            .serve_with_incoming(incoming)
            .await
            .expect("serve");
    });
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
        .expect("endpoint")
        .connect()
        .await
        .expect("connect");
    CommandRouter::new(MEMBER, channel)
}

/// The member the routers in these tests reach.
const MEMBER: u64 = 3;

fn statement(transaction: Option<TransactionId>) -> RoutedStatement {
    RoutedStatement {
        actor: principal(),
        query: "MATCH (n) RETURN n.k AS k".into(),
        params: HashMap::from([("limit".to_string(), Value::Int(3))]),
        settings: RoutedSettings {
            read_concern: Some(ReadConcernLevel::Majority),
            after_index: Some(9),
            at_timestamp: None,
            write_concern: Some(WriteConcern {
                w: WriteAck::Acks(2),
                journal: Journal::Cache,
                timeout_ms: 250,
            }),
            vector_consistency: Some(VectorConsistencyMode::Exact),
            vector_build_wait: Some(Duration::from_millis(40)),
            query_memory_limit: Some(1 << 20),
        },
        transaction,
        operation: None,
        deadline: Some(Duration::from_secs(30)),
        source: Some(SourceContext {
            file: "app.rs".into(),
            line: 12,
            function: "main".into(),
            app: "svc".into(),
            version: "1".into(),
        }),
        start: None,
    }
}

/// The start a transaction's first statement carries crosses the wire as
/// the ingress set it, every field, an unset one staying unset.
#[test]
fn a_start_crosses_unchanged() {
    for start in [
        TransactionStart::default(),
        TransactionStart {
            read_concern: Some(ReadConcernLevel::Snapshot),
            write_concern: Some(WriteConcern {
                w: WriteAck::Acks(2),
                journal: Journal::Cache,
                timeout_ms: 250,
            }),
            timeout_outcome: Some(TimeoutOutcome::Commit),
            operation: Some(operation(7)),
        },
    ] {
        let sent = RoutedStatement {
            start: Some(start),
            ..statement(Some(txid()))
        };
        let arrived =
            statement_from_wire(statement_to_wire(sent).expect("encoded")).expect("decoded");
        assert_eq!(arrived.start, Some(start));
    }
    let none = statement_from_wire(statement_to_wire(statement(Some(txid()))).expect("encoded"))
        .expect("decoded");
    assert_eq!(none.start, None);
}

/// An answer that fits one batch travels as one message: its columns, rows
/// and end together, with no admission event for a statement that commits
/// on its own. Every message is a frame and a socket write on both members.
#[tokio::test]
async fn a_small_answer_is_one_message() {
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Script::Rows(RoutedAnswer {
        columns: vec!["k".into()],
        rows: (0..5).map(|k| vec![Value::Int(k)]).collect(),
        stats: RoutedStats {
            nodes_created: 2,
            ..RoutedStats::default()
        },
    }));
    let router = serve(executor).await;
    let mut frames = router
        .client
        .clone()
        .execute(CommandRouter::request(
            statement_to_wire(statement(None)).expect("encoded"),
            None,
        ))
        .await
        .expect("answering")
        .into_inner();
    let mut messages = Vec::new();
    while let Some(frame) = frames.message().await.expect("frame") {
        messages.push(frame);
    }
    assert_eq!(messages.len(), 1, "{messages:?}");
    let kinds: Vec<&str> = messages[0]
        .events
        .iter()
        .map(|event| match event.event {
            Some(Event::Admitted(_)) => "admitted",
            Some(Event::Columns(_)) => "columns",
            Some(Event::Rows(_)) => "rows",
            Some(Event::End(_)) => "end",
            Some(Event::Error(_)) => "error",
            None => "none",
        })
        .collect();
    assert_eq!(kinds, ["columns", "rows", "end"]);
}

/// A statement reaches the executing member as the ingress sent it: who it
/// acts for, its parameters, every setting, its transaction and source; and
/// its answer comes back whole, rows in order.
#[tokio::test]
async fn a_statement_and_its_answer_cross_unchanged() {
    let executor = Arc::new(Scripted::default());
    let rows: Vec<Vec<Value>> = (0..5).map(|k| vec![Value::Int(k)]).collect();
    *executor.answer.lock().unwrap() = Some(Script::Rows(RoutedAnswer {
        columns: vec!["k".into()],
        rows: rows.clone(),
        stats: RoutedStats {
            nodes_created: 1,
            applied_index: 41,
            commit_ts: 9,
            read_index: 40,
            ..RoutedStats::default()
        },
    }));
    let router = serve(Arc::clone(&executor)).await;

    let sent = statement(None);
    let answer = router
        .execute(sent.clone(), &AdmissionSignal::new(|| {}))
        .await
        .expect("answered");
    assert_eq!(answer.columns, vec!["k".to_string()]);
    assert_eq!(answer.rows, rows);
    assert_eq!(answer.stats.applied_index, 41);
    assert_eq!(answer.stats.read_index, 40);

    let arrived = executor.statements.lock().unwrap().pop().expect("arrived");
    assert_eq!(arrived.actor, sent.actor);
    assert_eq!(arrived.query, sent.query);
    assert_eq!(arrived.params, sent.params);
    assert_eq!(arrived.settings, sent.settings);
    // The time left, never more than the ingress had.
    assert!(
        arrived
            .deadline
            .is_some_and(|left| sent.deadline.is_some_and(|sent| left <= sent) && !left.is_zero()),
        "{:?}",
        arrived.deadline
    );
    assert_eq!(arrived.transaction, None);
    assert_eq!(arrived.source.map(|s| s.line), Some(12));
}

/// An answer larger than one batch arrives in several and is put back
/// together in order.
#[tokio::test]
async fn a_large_answer_is_split_and_reassembled_in_order() {
    let executor = Arc::new(Scripted::default());
    // Each row about 64 KiB: well past one batch in all.
    let rows: Vec<Vec<Value>> = (0..48)
        .map(|k| vec![Value::Int(k), Value::String("x".repeat(64 << 10))])
        .collect();
    *executor.answer.lock().unwrap() = Some(Script::Rows(RoutedAnswer {
        columns: vec!["k".into(), "pad".into()],
        rows: rows.clone(),
        stats: RoutedStats::default(),
    }));
    let router = serve(executor).await;
    let answer = router
        .execute(statement(None), &AdmissionSignal::new(|| {}))
        .await
        .expect("answered");
    assert_eq!(answer.rows, rows);
}

/// The ingress reads a routed answer as its caller asks for rows, not whole
/// before handing any out: the columns come first, each read returns at most
/// what was asked for, and the statistics, sent last, are unknown until the
/// rows run out. The rows still arrive whole and in order.
#[tokio::test]
async fn an_answer_is_read_in_the_pieces_asked_for() {
    let executor = Arc::new(Scripted::default());
    // About 3 MiB in all: several batches on the wire.
    let rows: Vec<Vec<Value>> = (0..48)
        .map(|k| vec![Value::Int(k), Value::String("x".repeat(64 << 10))])
        .collect();
    *executor.answer.lock().unwrap() = Some(Script::Rows(RoutedAnswer {
        columns: vec!["k".into(), "pad".into()],
        rows: rows.clone(),
        stats: RoutedStats {
            nodes_created: 7,
            ..RoutedStats::default()
        },
    }));
    let router = serve(executor).await;
    let mut answer = router
        .execute_rows(statement(None), &AdmissionSignal::new(|| {}))
        .await
        .expect("answering");
    assert_eq!(answer.columns(), ["k".to_string(), "pad".to_string()]);

    let first = answer.next_rows(1).await.expect("first row");
    assert_eq!(first, rows[..1]);
    assert_eq!(
        answer.stats(),
        None,
        "the rest of the answer is not read yet"
    );

    let mut read = first;
    loop {
        let piece = answer.next_rows(5).await.expect("rows");
        assert!(piece.len() <= 5, "a read returns at most what was asked");
        if piece.is_empty() {
            break;
        }
        read.extend(piece);
    }
    assert_eq!(read, rows);
    assert_eq!(answer.stats().map(|s| s.nodes_created), Some(7));
}

/// An ingress that does not read holds the executing member back: the
/// member writes a few batches ahead and then waits, rather than producing
/// the whole answer into memory, and goes on as the ingress reads. Every row
/// still arrives, in order.
#[tokio::test(flavor = "multi_thread")]
async fn an_ingress_that_does_not_read_holds_the_member_back() {
    const ROWS: usize = 800;
    const WIDTH: usize = 32 << 10;
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Script::Many {
        rows: ROWS,
        width: WIDTH,
    });
    let router = serve(Arc::clone(&executor)).await;
    let mut answer = router
        .execute_rows(statement(None), &AdmissionSignal::new(|| {}))
        .await
        .expect("answering");

    // Until the member stops writing: it is waiting for the ingress.
    let mut seen = executor.written.load(Ordering::Acquire);
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let now = executor.written.load(Ordering::Acquire);
        if now == seen {
            break;
        }
        seen = now;
    }
    let per_batch = ROW_BATCH_BYTES / WIDTH;
    // The queue, the batch being sent, and what HTTP/2 lets through before
    // the ingress's window is spent.
    let ahead = (FRAMES_IN_FLIGHT + 6) * per_batch;
    assert!(
        seen < ROWS && seen <= ahead,
        "{seen} of {ROWS} rows written with none read, at most {ahead} expected"
    );

    let mut read = 0;
    loop {
        let piece = answer.next_rows(64).await.expect("rows");
        if piece.is_empty() {
            break;
        }
        for row in &piece {
            read += 1;
            assert_eq!(row, &vec![noise(read as u64, WIDTH)], "row {read}");
        }
    }
    assert_eq!(read, ROWS);
    assert!(answer.stats().is_some());
}

/// A statement that fails after part of its answer went out delivers those
/// rows and then its failure: the ingress reads the rows, then the error the
/// member raised, never the end of a complete answer.
#[tokio::test]
async fn a_failure_after_some_rows_follows_them() {
    let executor = Arc::new(Scripted::default());
    let rows: Vec<Vec<Value>> = (0..3).map(|k| vec![Value::Int(k)]).collect();
    *executor.answer.lock().unwrap() = Some(Script::RowsThenRefuse(
        RoutedAnswer {
            columns: vec!["k".into()],
            rows: rows.clone(),
            stats: RoutedStats::default(),
        },
        || DatabaseError::Budget(coordinode_core::budget::BudgetStop::Deadline),
    ));
    let router = serve(executor).await;
    let mut answer = router
        .execute_rows(statement(None), &AdmissionSignal::new(|| {}))
        .await
        .expect("answering");
    assert_eq!(answer.columns(), ["k".to_string()]);
    assert_eq!(answer.next_rows(10).await.expect("the rows sent"), rows);
    let failed = answer.next_rows(10).await;
    assert!(
        matches!(
            failed,
            Err(DatabaseError::Budget(
                coordinode_core::budget::BudgetStop::Deadline
            ))
        ),
        "{failed:?}"
    );
    assert_eq!(answer.stats(), None, "the answer never completed");

    // Read whole, the failure is the answer.
    let whole = router
        .execute(statement(None), &AdmissionSignal::new(|| {}))
        .await;
    assert!(matches!(whole, Err(DatabaseError::Budget(_))), "{whole:?}");
}

/// A transaction's statement tells the ingress it has its place before its
/// answer is ready: the ingress passes the next command of the transaction
/// on at that point, not after the whole statement ran.
#[tokio::test]
async fn the_ingress_learns_of_the_admission_before_the_answer() {
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Script::Rows(RoutedAnswer::default()));
    let (release, gate) = std::sync::mpsc::channel();
    *executor.gate.lock().unwrap() = Some(gate);
    let router = serve(Arc::clone(&executor)).await;

    let admitted = Arc::new(tokio::sync::Notify::new());
    let signal = {
        let admitted = Arc::clone(&admitted);
        AdmissionSignal::new(move || admitted.notify_one())
    };
    let call = tokio::spawn(async move { router.execute(statement(Some(txid())), &signal).await });
    // Not admitted while the executing member holds the statement.
    assert!(
        tokio::time::timeout(Duration::from_millis(100), admitted.notified())
            .await
            .is_err()
    );
    release.send(()).expect("release");
    tokio::time::timeout(Duration::from_secs(10), admitted.notified())
        .await
        .expect("admitted");
    call.await.expect("joined").expect("answered");
    let arrived = executor.statements.lock().unwrap().pop().expect("arrived");
    assert_eq!(arrived.transaction, Some(txid()));
}

/// An ingress that leaves before a transaction's statement answered aborts
/// the transaction at the executing member: the client cannot know what the
/// statement did, and it never runs twice. An autonomous statement in the
/// same situation leaves nothing to abort.
#[tokio::test]
async fn an_ingress_that_leaves_mid_statement_abandons_its_transaction() {
    for transaction in [Some(txid()), None] {
        let executor = Arc::new(Scripted::default());
        let router = serve(Arc::clone(&executor)).await;
        let call = tokio::spawn(async move {
            router
                .execute(statement(transaction), &AdmissionSignal::new(|| {}))
                .await
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while executor.statements.lock().unwrap().is_empty() {
            assert!(std::time::Instant::now() < deadline, "never arrived");
            tokio::task::yield_now().await;
        }
        call.abort();
        while !executor.cancelled.load(Ordering::Acquire) {
            assert!(std::time::Instant::now() < deadline, "never cancelled");
            tokio::task::yield_now().await;
        }
        // The abandon, if any, follows the cancelled statement on its thread.
        let abandoned = loop {
            let rollbacks = executor.rollbacks.lock().unwrap().clone();
            if transaction.is_none() || !rollbacks.is_empty() {
                break rollbacks;
            }
            assert!(std::time::Instant::now() < deadline, "never abandoned");
            tokio::task::yield_now().await;
        };
        match transaction {
            Some(id) => {
                assert_eq!(abandoned.len(), 1, "{abandoned:?}");
                assert_eq!(abandoned[0].transaction, id);
                assert!(abandoned[0].abandoned, "not one the client asked for");
                assert_eq!(abandoned[0].end(), TransactionEnd::StreamLost);
            }
            None => assert!(abandoned.is_empty(), "{abandoned:?}"),
        }
    }
}

/// A refusal reaches the ingress as the error the executing member raised,
/// with every field: the client of the ingress is told what it would have
/// been told at the executing member.
#[tokio::test]
async fn a_refusal_crosses_as_the_same_error() {
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Script::Refuse(|| DatabaseError::RevisionMismatch {
        transaction: Some(txid()),
        expected: Some(4),
        current: None,
    }));
    let router = serve(executor).await;
    let err = router
        .execute(statement(Some(txid())), &AdmissionSignal::new(|| {}))
        .await
        .expect_err("refused");
    assert!(
        matches!(
            err,
            DatabaseError::RevisionMismatch {
                transaction: Some(id),
                expected: Some(4),
                current: None,
            } if id == txid()
        ),
        "{err:?}"
    );
}

/// A commit whose answer does not arrive before the deadline is an unknown
/// outcome, never a refusal or an abort: here the member did commit it.
#[tokio::test]
async fn a_commit_whose_answer_is_lost_is_not_an_abort() {
    let executor = Arc::new(Scripted::default());
    *executor.commit_delay.lock().unwrap() = Duration::from_millis(300);
    let router = serve(Arc::clone(&executor)).await;
    let err = router
        .commit(RoutedCommit {
            actor: principal(),
            transaction: txid(),
            operation: None,
            deadline: Some(Duration::from_millis(50)),
            expect: expectations(),
        })
        .await
        .expect_err("the answer came too late");
    assert!(
        matches!(
            err,
            DatabaseError::PeerOutcomeUnknown {
                member: MEMBER,
                failure: PeerFailure::Deadline,
                ..
            }
        ),
        "{err:?}"
    );
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while executor.commits.load(Ordering::Acquire) == 0 {
        assert!(std::time::Instant::now() < deadline, "never committed");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A call from an ingress that reads another error encoding is refused before
/// anything runs, and the ingress learns that nothing ran.
#[tokio::test]
async fn a_call_under_another_codec_is_refused_unread() {
    let executor = Arc::new(Scripted::default());
    let router = serve(Arc::clone(&executor)).await;
    let mut request = Request::new(statement_to_wire(statement(None)).expect("encoded"));
    request
        .metadata_mut()
        .insert(CODEC_HEADER, MetadataValue::from(ERROR_CODEC_VERSION + 1));
    let status = router
        .client
        .clone()
        .execute(request)
        .await
        .expect_err("refused");
    assert!(status.metadata().contains_key(REFUSED_HEADER), "{status:?}");
    assert!(
        matches!(
            router.lost(&status),
            DatabaseError::PeerUnreachable { member: MEMBER, .. }
        ),
        "the command did not run"
    );
    assert!(executor.statements.lock().unwrap().is_empty());
}

/// An error body the ingress cannot read is a lost answer, not a refusal: the
/// command may have run.
#[tokio::test]
async fn an_unreadable_error_is_an_unknown_outcome() {
    let channel = tonic::transport::Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
    let router = CommandRouter::new(MEMBER, channel);
    let undeliverable = errors::encode(DatabaseError::DuplicateKey {
        table: "Account".into(),
        key: "k".repeat(MAX_ROUTED_ERROR),
        element_id: "0000000000007".into(),
    });
    for bytes in [
        vec![0xff, 0xff, 0xff],
        vec![0; MAX_ROUTED_ERROR + 1],
        undeliverable,
    ] {
        let err = router.failure(&bytes).expect_err("a failure");
        assert!(
            matches!(
                err,
                DatabaseError::PeerOutcomeUnknown {
                    member: MEMBER,
                    failure: PeerFailure::Protocol,
                    ..
                }
            ),
            "{err:?}"
        );
    }
    assert!(router.failure(&[]).is_ok(), "no bytes, no failure");
}

/// An ingress that stops waiting (its client went away) stops the statement
/// at the executing member.
#[tokio::test]
async fn an_ingress_that_leaves_cancels_the_statement() {
    let executor = Arc::new(Scripted::default());
    let router = serve(Arc::clone(&executor)).await;
    let call = tokio::spawn(async move {
        router
            .execute(statement(None), &AdmissionSignal::new(|| {}))
            .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    call.abort();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !executor.cancelled.load(Ordering::Acquire) {
        assert!(std::time::Instant::now() < deadline, "never cancelled");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Begin, commit and rollback carry their transaction and settings, and the
/// receipt and refusals come back.
#[tokio::test]
async fn transaction_commands_cross_both_ways() {
    let executor = Arc::new(Scripted::default());
    let router = serve(Arc::clone(&executor)).await;
    router
        .begin(RoutedBegin {
            transaction: txid(),
            actor: principal(),
            read_concern: Some(ReadConcernLevel::Snapshot),
            write_concern: None,
            timeout_outcome: Some(TimeoutOutcome::Commit),
            operation: Some(operation(1)),
        })
        .await
        .expect("begun");
    let receipt = router
        .commit(RoutedCommit {
            actor: principal(),
            transaction: txid(),
            operation: None,
            deadline: Some(COMMIT_DEADLINE),
            expect: expectations(),
        })
        .await
        .expect("committed");
    assert_eq!(receipt.commit_ts, Timestamp::from_raw(77));
    assert_eq!(receipt.applied_index, Some(12));
    assert_eq!(executor.commits.load(Ordering::Acquire), 1);
    for abandoned in [false, true] {
        let err = router
            .rollback(RoutedRollback {
                actor: principal(),
                transaction: txid(),
                abandoned,
            })
            .await
            .expect_err("refused");
        assert!(
            matches!(err, DatabaseError::UnknownTransaction(id) if id == txid()),
            "{err:?}"
        );
    }
    let arrived: Vec<bool> = executor
        .rollbacks
        .lock()
        .unwrap()
        .iter()
        .map(|rollback| rollback.abandoned)
        .collect();
    assert_eq!(
        arrived,
        [false, true],
        "the cause crosses with the rollback"
    );
}

/// A command that names nobody is refused rather than run as anyone.
#[test]
fn a_command_without_an_actor_is_refused() {
    let mut wire = statement_to_wire(statement(None)).expect("encoded");
    wire.actor = None;
    let err = statement_from_wire(wire).expect_err("refused");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

/// Settings left unset stay unset across the wire, so the executing member
/// applies its own rather than a default the ingress never chose.
#[test]
fn unset_settings_stay_unset() {
    let settings = RoutedSettings::default();
    assert_eq!(
        settings_from_wire(Some(settings_to_wire(settings.clone()))).expect("decoded"),
        settings
    );
}

/// A deadline under a millisecond still stops the call: it is not sent as
/// "none".
#[test]
fn a_tiny_deadline_is_kept() {
    assert_eq!(deadline_to_wire(Some(Duration::from_micros(10))), 1);
    assert_eq!(deadline_to_wire(None), 0);
}
