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
    // xorshift64*, never zero for a seed that is not; eight bytes a step, so
    // the megabytes these answers carry are cheap to make in a debug build.
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut bytes = Vec::with_capacity(width.next_multiple_of(8));
    while bytes.len() < width {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        bytes.extend_from_slice(&state.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
    }
    bytes.truncate(width);
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
    serve_handler(CommandRoutingHandler::new(executor)).await
}

/// Serve `handler` on a local port and return a router to it.
async fn serve_handler(handler: CommandRoutingHandler<Scripted>) -> CommandRouter {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("address");
    let incoming = tonic::transport::server::TcpIncoming::from(listener);
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(handler.into_service(MAX_ROUTED_MESSAGE))
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

/// Serve `executor` within `limits` on a local port and return a router to
/// it.
async fn serve_limited(executor: Arc<Scripted>, limits: RoutingLimits) -> CommandRouter {
    serve_handler(CommandRoutingHandler::limited(
        executor,
        limits.check().expect("valid limits"),
    ))
    .await
}

fn routed_reply(call: u64) -> Result<wire::RoutedReply, Status> {
    reply(
        call,
        wire::routed_reply::Reply::Refused(wire::RoutedRefusal::default()),
    )
}

/// The reply queue never holds more than its bound, read or not: the reply
/// past it is not queued, every later one is refused, and the session's
/// reader learns the session is over without the stream being polled.
#[tokio::test]
async fn the_reply_queue_holds_no_more_than_its_bound_unread() {
    let (replies, mut stream) = Replies::new(4);
    for call in 1..=4 {
        assert!(replies.send(routed_reply(call)).is_ok(), "{call} fits");
    }
    for call in 5..=1000 {
        assert!(replies.send(routed_reply(call)).is_err(), "{call} does not");
    }
    assert_eq!(
        stream.receiver.len(),
        4,
        "nothing past the bound was queued"
    );
    assert_eq!(replies.queue.queued.load(Ordering::Acquire), 4);
    tokio::time::timeout(Duration::from_secs(10), replies.closed())
        .await
        .expect("the reader learns the session is over");

    use tokio_stream::StreamExt;
    let ended = stream.next().await.expect("an item");
    assert_eq!(
        ended.expect_err("the stream ends with an error").code(),
        Code::ResourceExhausted
    );
    assert!(stream.next().await.is_none(), "and then ends");
}

/// Replies the stream takes give their places back.
#[tokio::test]
async fn a_read_reply_frees_its_place() {
    use tokio_stream::StreamExt;
    let (replies, mut stream) = Replies::new(2);
    for round in 0..100 {
        assert!(replies.send(routed_reply(round)).is_ok());
        assert!(replies.send(routed_reply(round)).is_ok());
        stream.next().await.expect("one").expect("a reply");
        stream.next().await.expect("two").expect("a reply");
    }
    assert_eq!(replies.queue.queued.load(Ordering::Acquire), 0);
}

/// Bounds the member cannot keep are refused, not quietly lifted.
#[test]
fn routing_limits_the_member_cannot_keep_are_refused() {
    assert!(RoutingLimits::default().check().is_ok());
    for bad in [
        RoutingLimits {
            commands: 0,
            ..RoutingLimits::default()
        },
        RoutingLimits {
            session_commands: tokio::sync::Semaphore::MAX_PERMITS + 1,
            ..RoutingLimits::default()
        },
        RoutingLimits {
            frames_ahead: 0,
            ..RoutingLimits::default()
        },
        RoutingLimits {
            row_batch_bytes: 0,
            ..RoutingLimits::default()
        },
        RoutingLimits {
            session_commands: tokio::sync::Semaphore::MAX_PERMITS,
            frames_ahead: u32::MAX,
            ..RoutingLimits::default()
        },
    ] {
        assert!(bad.check().is_err(), "{bad:?}");
    }
    // The checked defaults are the defaults, with the bound they give.
    assert_eq!(
        RoutingLimits::default().check().expect("valid"),
        CheckedRoutingLimits::default()
    );
}

/// Open a session to the member `router` reaches by hand, naming `codec` as
/// the error encoding read: the requests to send on it, and its replies.
async fn open_session(
    router: &CommandRouter,
    codec: u32,
) -> Result<
    (
        tokio::sync::mpsc::UnboundedSender<wire::RoutedRequest>,
        tonic::Streaming<wire::RoutedReply>,
    ),
    Status,
> {
    let (requests, outbound) = tokio::sync::mpsc::unbounded_channel();
    let mut request = Request::new(tokio_stream::wrappers::UnboundedReceiverStream::new(
        outbound,
    ));
    request
        .metadata_mut()
        .insert(CODEC_HEADER, MetadataValue::from(codec));
    let replies = router
        .link
        .client
        .clone()
        .session(request)
        .await?
        .into_inner();
    Ok((requests, replies))
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
    let (requests, mut replies) = open_session(&router, ERROR_CODEC_VERSION)
        .await
        .expect("opened");
    requests
        .send(wire::RoutedRequest {
            call: 1,
            request: Some(wire::routed_request::Request::Execute(Box::new(
                statement_to_wire(statement(None)).expect("encoded"),
            ))),
        })
        .expect("sent");
    let reply = replies.message().await.expect("reply").expect("a reply");
    assert_eq!(reply.call, 1);
    let Some(wire::routed_reply::Reply::Frame(frame)) = reply.reply else {
        panic!("{reply:?}");
    };
    let kinds: Vec<&str> = frame
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
    // Small batches and little credit, so the answer is many times what the
    // credit lets ahead while staying a megabyte on the wire.
    const FRAMES: u32 = 2;
    const BATCH_BYTES: usize = 16 << 10;
    const WIDTH: usize = 4 << 10;
    const ROWS: usize = 256;
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Script::Many {
        rows: ROWS,
        width: WIDTH,
    });
    let router = serve_limited(
        Arc::clone(&executor),
        RoutingLimits {
            frames_ahead: FRAMES,
            row_batch_bytes: BATCH_BYTES,
            ..RoutingLimits::default()
        },
    )
    .await;
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
    let per_batch = BATCH_BYTES / WIDTH;
    // The frames the credit lets ahead, and the batch filling while the
    // member waits for more.
    let ahead = (FRAMES as usize + 1) * per_batch + 1;
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

/// A session dropped before it could finish, as its runtime shutting down
/// drops it, still stops the statements it runs: one waiting for credit that
/// will never come returns, so the shutdown, which waits for the blocking
/// threads, ends.
#[test]
fn a_session_dropped_unfinished_stops_its_statements() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Script::Many {
        rows: 64,
        width: 4 << 10,
    });
    // The ingress keeps its answer open and unread, so the session never
    // ends on its own: only the shutdown drops it.
    let (router, answer) = runtime.block_on(async {
        let router = serve_limited(
            Arc::clone(&executor),
            RoutingLimits {
                frames_ahead: 1,
                row_batch_bytes: 16 << 10,
                ..RoutingLimits::default()
            },
        )
        .await;
        let answer = router
            .execute_rows(statement(None), &AdmissionSignal::new(|| {}))
            .await
            .expect("answering");
        let mut seen = executor.written.load(Ordering::Acquire);
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let now = executor.written.load(Ordering::Acquire);
            if now == seen {
                break;
            }
            seen = now;
        }
        (router, answer)
    });
    assert!(
        executor.written.load(Ordering::Acquire) < 64,
        "the statement waits for credit"
    );

    let (done, shut) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        drop(runtime);
        let _ = done.send(());
    });
    assert!(
        shut.recv_timeout(Duration::from_secs(10)).is_ok(),
        "the shutdown waits on a statement nobody stopped"
    );
    drop((answer, router));
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

/// A session from an ingress that reads another error encoding is refused
/// before any command of it is read, and the ingress learns that nothing ran.
#[tokio::test]
async fn a_session_under_another_codec_is_refused_unread() {
    let executor = Arc::new(Scripted::default());
    let router = serve(Arc::clone(&executor)).await;
    let status = open_session(&router, ERROR_CODEC_VERSION + 1)
        .await
        .expect_err("refused");
    assert!(status.metadata().contains_key(REFUSED_HEADER), "{status:?}");
    assert!(executor.statements.lock().unwrap().is_empty());
}

/// A session that does not open reached no member: every command sent on it
/// is told it did not run, never that its outcome is unknown, so the ingress
/// may send it again.
#[tokio::test]
async fn a_session_that_never_opened_ran_nothing() {
    let channel = tonic::transport::Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
    let router = CommandRouter::new(MEMBER, channel);
    let err = router
        .execute(statement(None), &AdmissionSignal::new(|| {}))
        .await
        .expect_err("nobody listens");
    assert!(
        matches!(err, DatabaseError::PeerUnreachable { member: MEMBER, .. }),
        "{err:?}"
    );
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

/// The session a router's commands travel on, while one is open.
fn current_session(router: &CommandRouter) -> Option<Arc<Session>> {
    router.link.session.lock().clone()
}

/// Commands sent at once through one router share one session and each gets
/// its own answer, none another's.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_commands_share_one_session() {
    let executor = Arc::new(Scripted::default());
    let rows: Vec<Vec<Value>> = (0..3).map(|k| vec![Value::Int(k)]).collect();
    *executor.answer.lock().unwrap() = Some(Script::Rows(RoutedAnswer {
        columns: vec!["k".into()],
        rows: rows.clone(),
        stats: RoutedStats {
            nodes_created: 1,
            ..RoutedStats::default()
        },
    }));
    let router = serve(Arc::clone(&executor)).await;
    router
        .execute(statement(None), &AdmissionSignal::new(|| {}))
        .await
        .expect("first");
    let session = current_session(&router).expect("a session");

    let calls: Vec<_> = (0..64)
        .map(|_| {
            let router = router.clone();
            tokio::spawn(async move {
                router
                    .execute(statement(None), &AdmissionSignal::new(|| {}))
                    .await
            })
        })
        .collect();
    for call in calls {
        let answer = call.await.expect("joined").expect("answered");
        assert_eq!(answer.rows, rows);
        assert_eq!(answer.stats.nodes_created, 1);
    }
    assert!(
        Arc::ptr_eq(&session, &current_session(&router).expect("a session")),
        "every command went on the first session"
    );
    assert_eq!(executor.statements.lock().unwrap().len(), 65);
}

/// An answer nobody reads holds back its own statement only: another
/// statement on the same session streams its whole answer meanwhile, and the
/// unread one still arrives whole once read.
#[tokio::test(flavor = "multi_thread")]
async fn an_unread_answer_holds_back_no_other_command() {
    const ROWS: usize = 200;
    const WIDTH: usize = 32 << 10;
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Script::Many {
        rows: ROWS,
        width: WIDTH,
    });
    let router = serve(Arc::clone(&executor)).await;
    let mut unread = router
        .execute_rows(statement(None), &AdmissionSignal::new(|| {}))
        .await
        .expect("answering");
    let other = tokio::time::timeout(
        Duration::from_secs(30),
        router.execute(statement(None), &AdmissionSignal::new(|| {})),
    )
    .await
    .expect("not held back by the unread answer")
    .expect("answered");
    assert_eq!(other.rows.len(), ROWS);

    let mut read = 0;
    loop {
        let piece = unread.next_rows(64).await.expect("rows");
        if piece.is_empty() {
            break;
        }
        read += piece.len();
    }
    assert_eq!(read, ROWS);
}

/// Whether the test's admission lets the connection route commands.
static ADMITTED: AtomicBool = AtomicBool::new(true);

fn scripted_admission(_: Option<&PeerConnectInfo>) -> Result<(), Status> {
    if ADMITTED.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err(Status::unauthenticated("withdrawn"))
    }
}

/// Admission is checked before every command of an open session, not only
/// when it opened: once the connection is no longer admitted, the next
/// command is refused and does not run, though the session stays open.
#[tokio::test]
async fn every_command_of_an_open_session_is_admitted_again() {
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Script::Rows(RoutedAnswer::default()));
    let router = serve_handler(
        CommandRoutingHandler::new(Arc::clone(&executor)).admitting(scripted_admission),
    )
    .await;
    router
        .execute(statement(None), &AdmissionSignal::new(|| {}))
        .await
        .expect("admitted");
    let session = current_session(&router).expect("a session");

    ADMITTED.store(false, Ordering::Release);
    let err = router
        .execute(statement(None), &AdmissionSignal::new(|| {}))
        .await
        .expect_err("no longer admitted");
    assert!(
        matches!(&err, DatabaseError::PeerUnreachable { member: MEMBER, message } if message.contains("admission")),
        "{err:?}"
    );
    assert_eq!(
        executor.statements.lock().unwrap().len(),
        1,
        "the refused command did not run"
    );
    assert!(Arc::ptr_eq(
        &session,
        &current_session(&router).expect("a session")
    ));
}

/// A second command under a number already running on the session is
/// refused without running, and the first one runs on; a cancel then stops
/// it.
#[tokio::test]
async fn a_number_in_use_is_refused_and_a_cancel_stops_its_command() {
    let executor = Arc::new(Scripted::default());
    let router = serve(Arc::clone(&executor)).await;
    let (requests, mut replies) = open_session(&router, ERROR_CODEC_VERSION)
        .await
        .expect("opened");
    let execute = || wire::RoutedRequest {
        call: 7,
        request: Some(wire::routed_request::Request::Execute(Box::new(
            statement_to_wire(statement(None)).expect("encoded"),
        ))),
    };
    requests.send(execute()).expect("sent");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while executor.statements.lock().unwrap().is_empty() {
        assert!(std::time::Instant::now() < deadline, "never arrived");
        tokio::task::yield_now().await;
    }
    requests.send(execute()).expect("sent again");
    let reply = replies.message().await.expect("reply").expect("a reply");
    assert_eq!(reply.call, 7);
    assert!(
        matches!(&reply.reply, Some(wire::routed_reply::Reply::Refused(r)) if r.reason == "duplicate"),
        "{reply:?}"
    );
    assert_eq!(executor.statements.lock().unwrap().len(), 1);

    requests
        .send(wire::RoutedRequest {
            call: 7,
            request: Some(wire::routed_request::Request::Cancel(wire::RoutedCancel {})),
        })
        .expect("cancel sent");
    while !executor.cancelled.load(Ordering::Acquire) {
        assert!(std::time::Instant::now() < deadline, "never cancelled");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// A session whose ingress goes away stops what still runs for it and
/// aborts the transactions those statements ran in.
#[tokio::test]
async fn a_lost_session_stops_its_statements() {
    let executor = Arc::new(Scripted::default());
    let router = serve(Arc::clone(&executor)).await;
    let (requests, replies) = open_session(&router, ERROR_CODEC_VERSION)
        .await
        .expect("opened");
    requests
        .send(wire::RoutedRequest {
            call: 1,
            request: Some(wire::routed_request::Request::Execute(Box::new(
                statement_to_wire(statement(Some(txid()))).expect("encoded"),
            ))),
        })
        .expect("sent");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while executor.statements.lock().unwrap().is_empty() {
        assert!(std::time::Instant::now() < deadline, "never arrived");
        tokio::task::yield_now().await;
    }
    drop(requests);
    drop(replies);
    loop {
        let rollbacks = executor.rollbacks.lock().unwrap().clone();
        if let Some(rollback) = rollbacks.first() {
            assert!(rollback.abandoned);
            assert_eq!(rollback.transaction, txid());
            break;
        }
        assert!(std::time::Instant::now() < deadline, "never abandoned");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(executor.cancelled.load(Ordering::Acquire));
}

/// An autonomous statement sent by hand under `call`.
fn execute_request(call: u64) -> wire::RoutedRequest {
    in_transaction(call, None)
}

/// A statement of `transaction` sent by hand under `call`.
fn in_transaction(call: u64, transaction: Option<TransactionId>) -> wire::RoutedRequest {
    wire::RoutedRequest {
        call,
        request: Some(wire::routed_request::Request::Execute(Box::new(
            statement_to_wire(statement(transaction)).expect("encoded"),
        ))),
    }
}

/// Whether `reply` is a frame that ends its answer with its statistics.
fn ends(reply: &wire::routed_reply::Reply) -> bool {
    matches!(reply, wire::routed_reply::Reply::Frame(f)
        if f.events.iter().any(|e| matches!(e.event, Some(Event::End(_)))))
}

/// Read `call`'s replies up to the one that ends its answer.
async fn end_of(
    replies: &mut tonic::Streaming<wire::RoutedReply>,
    call: u64,
) -> wire::routed_reply::Reply {
    loop {
        let reply = reply_to(replies, call).await;
        if !matches!(reply, wire::routed_reply::Reply::Frame(ref f)
            if f.events.iter().all(|e| matches!(e.event, Some(Event::Admitted(_)))))
        {
            return reply;
        }
    }
}

/// Wait until `executor` has taken `count` statements.
async fn taken(executor: &Scripted, count: usize) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while executor.statements.lock().unwrap().len() < count {
        assert!(std::time::Instant::now() < deadline, "never arrived");
        tokio::task::yield_now().await;
    }
}

/// The next reply on `replies` to `call`, skipping the others.
async fn reply_to(
    replies: &mut tonic::Streaming<wire::RoutedReply>,
    call: u64,
) -> wire::routed_reply::Reply {
    loop {
        let reply = replies.message().await.expect("reply").expect("a reply");
        if reply.call == call {
            return reply.reply.expect("a reply body");
        }
    }
}

fn is_busy(reply: &wire::routed_reply::Reply) -> bool {
    matches!(reply, wire::routed_reply::Reply::Refused(r) if r.reason == "busy")
}

/// A command past the bound on one session's running commands is refused
/// as busy and does not run. Once the running one has answered, its slot is
/// free: the next command sent on that answer runs, with no busy in between.
#[tokio::test]
async fn a_command_past_the_session_bound_is_refused_until_a_slot_frees() {
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Script::Rows(RoutedAnswer::default()));
    // The transaction's statement waits here until the test lets it on.
    let (release, gate) = std::sync::mpsc::channel();
    *executor.gate.lock().unwrap() = Some(gate);
    let router = serve_limited(
        Arc::clone(&executor),
        RoutingLimits {
            session_commands: 1,
            ..RoutingLimits::default()
        },
    )
    .await;
    let (requests, mut replies) = open_session(&router, ERROR_CODEC_VERSION)
        .await
        .expect("opened");
    requests
        .send(in_transaction(1, Some(txid())))
        .expect("sent");
    taken(&executor, 1).await;

    requests.send(execute_request(2)).expect("sent");
    let refused = reply_to(&mut replies, 2).await;
    assert!(is_busy(&refused), "{refused:?}");
    assert_eq!(
        executor.statements.lock().unwrap().len(),
        1,
        "the refused command did not run"
    );

    release.send(()).expect("released");
    let ended = end_of(&mut replies, 1).await;
    assert!(ends(&ended), "{ended:?}");

    requests.send(execute_request(3)).expect("sent");
    let ran = end_of(&mut replies, 3).await;
    assert!(ends(&ran), "{ran:?}");
}

/// An ingress that keeps sending commands and stops reading cannot grow the
/// member's queue of replies without bound, refusals included: once the
/// stream's window is full and the queue past its bound, the session ends
/// with a resource error, and it ends what still runs.
#[tokio::test]
async fn an_ingress_that_sends_and_never_reads_has_its_session_ended() {
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Script::Rows(RoutedAnswer::default()));
    let limits = RoutingLimits {
        session_commands: 1,
        frames_ahead: 1,
        ..RoutingLimits::default()
    };
    let router = serve_limited(Arc::clone(&executor), limits).await;
    let (requests, mut replies) = open_session(&router, ERROR_CODEC_VERSION)
        .await
        .expect("opened");
    // Far more answers than the stream's window and the queue hold
    // together; none is read while they are sent.
    for call in 1..=20_000 {
        if requests.send(execute_request(call)).is_err() {
            break;
        }
    }

    let ended = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match replies.message().await {
                Ok(Some(_)) => {}
                Ok(None) => return None,
                Err(status) => return Some(status),
            }
        }
    })
    .await
    .expect("the session ended");
    let status = ended.expect("ended with an error, not cleanly");
    assert_eq!(status.code(), Code::ResourceExhausted, "{status:?}");
    assert_eq!(limits.session_replies(), Some(4));
}

/// The bound on the member's running commands spans its sessions: with one
/// slot held by one peer's statement, another peer's command is refused as
/// busy without running.
#[tokio::test]
async fn the_member_bound_spans_sessions() {
    let executor = Arc::new(Scripted::default());
    let router = serve_limited(
        Arc::clone(&executor),
        RoutingLimits {
            commands: 1,
            ..RoutingLimits::default()
        },
    )
    .await;
    let (first, _first_replies) = open_session(&router, ERROR_CODEC_VERSION)
        .await
        .expect("opened");
    let (second, mut second_replies) = open_session(&router, ERROR_CODEC_VERSION)
        .await
        .expect("opened");
    first.send(execute_request(1)).expect("sent");
    taken(&executor, 1).await;

    second.send(execute_request(1)).expect("sent");
    let refused = reply_to(&mut second_replies, 1).await;
    assert!(is_busy(&refused), "{refused:?}");
    assert_eq!(executor.statements.lock().unwrap().len(), 1);

    // The held statement ends with its session.
    drop(first);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !executor.cancelled.load(Ordering::Acquire) {
        assert!(std::time::Instant::now() < deadline, "never stopped");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// The frame that ends an answer goes out without credit, so a statement
/// whose ingress has taken nothing yet still ends and gives its slot back;
/// the batch size and credit are the ones the limits set.
#[tokio::test]
async fn the_last_frame_of_an_answer_waits_for_no_credit() {
    let executor = Arc::new(Scripted::default());
    // Two rows fill a batch; the third goes with the end.
    *executor.answer.lock().unwrap() = Some(Script::Many {
        rows: 3,
        width: 600,
    });
    let router = serve_limited(
        Arc::clone(&executor),
        RoutingLimits {
            frames_ahead: 1,
            row_batch_bytes: 1024,
            ..RoutingLimits::default()
        },
    )
    .await;
    let (requests, mut replies) = open_session(&router, ERROR_CODEC_VERSION)
        .await
        .expect("opened");
    requests.send(execute_request(1)).expect("sent");

    // No credit is ever granted: the one frame ahead, then the last.
    let first = reply_to(&mut replies, 1).await;
    let wire::routed_reply::Reply::Frame(first) = first else {
        panic!("{first:?}");
    };
    assert!(
        first
            .events
            .iter()
            .any(|e| matches!(e.event, Some(Event::Rows(_))))
    );
    assert!(
        !first
            .events
            .iter()
            .any(|e| matches!(e.event, Some(Event::End(_))))
    );
    let last = tokio::time::timeout(Duration::from_secs(10), reply_to(&mut replies, 1))
        .await
        .expect("the answer ended without credit");
    assert!(
        matches!(&last, wire::routed_reply::Reply::Frame(f) if f.events.iter().any(|e| matches!(e.event, Some(Event::End(_))))),
        "{last:?}"
    );
}
