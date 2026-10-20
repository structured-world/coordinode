use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;

use super::*;

/// An executor that answers from what the test gave it and records what
/// arrived.
#[derive(Default)]
struct Scripted {
    /// The answer every statement gets.
    answer: Mutex<Option<Result<RoutedAnswer, Status>>>,
    /// Each statement as it arrived.
    statements: Mutex<Vec<RoutedStatement>>,
    /// Held by a transaction's statement until the test lets it take its
    /// place: (taken, released).
    gate: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    /// Set when a statement ran to its end with its cancel switch thrown.
    cancelled: AtomicBool,
    /// Commits that arrived.
    commits: AtomicUsize,
    /// Rollbacks that arrived, in order.
    rollbacks: Mutex<Vec<RoutedRollback>>,
}

impl CommandExecutor for Scripted {
    fn execute(
        &self,
        statement: RoutedStatement,
        cancel: &CancelFlag,
        admitted: &AdmissionSignal,
    ) -> Result<RoutedAnswer, Status> {
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
            return Err(Status::cancelled("cancelled"));
        }
        self.answer.lock().unwrap().clone().expect("an answer")
    }

    fn begin(&self, begin: RoutedBegin) -> Result<(), Status> {
        assert_eq!(begin.transaction, txid());
        assert_eq!(begin.timeout_outcome, Some(TimeoutOutcome::Commit));
        assert_eq!(begin.operation, Some(operation(1)));
        Ok(())
    }

    fn commit(&self, commit: RoutedCommit) -> Result<CommitReceipt, Status> {
        assert_eq!(commit.transaction, txid());
        assert_eq!(commit.expect, expectations());
        self.commits.fetch_add(1, Ordering::AcqRel);
        Ok(CommitReceipt {
            commit_ts: Timestamp::from_raw(77),
            applied_index: Some(12),
        })
    }

    fn rollback(&self, rollback: RoutedRollback) -> Result<(), Status> {
        assert_eq!(rollback.transaction, txid());
        self.rollbacks.lock().unwrap().push(rollback);
        Err(Status::aborted("already ended"))
    }

    /// Answers each kind of call with something that names what arrived, so
    /// a test sees the call crossed whole.
    fn operation(
        &self,
        actor: ActingContext,
        call: RoutedOperationCall,
    ) -> Result<RoutedOperationAnswer, Status> {
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
    CommandRouter::new(channel)
}

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
    }
}

/// A statement reaches the executing member as the ingress sent it: who it
/// acts for, its parameters, every setting, its transaction and source; and
/// its answer comes back whole, rows in order.
#[tokio::test]
async fn a_statement_and_its_answer_cross_unchanged() {
    let executor = Arc::new(Scripted::default());
    let rows: Vec<Vec<Value>> = (0..5).map(|k| vec![Value::Int(k)]).collect();
    *executor.answer.lock().unwrap() = Some(Ok(RoutedAnswer {
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
    *executor.answer.lock().unwrap() = Some(Ok(RoutedAnswer {
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

/// A transaction's statement tells the ingress it has its place before its
/// answer is ready: the ingress passes the next command of the transaction
/// on at that point, not after the whole statement ran.
#[tokio::test]
async fn the_ingress_learns_of_the_admission_before_the_answer() {
    let executor = Arc::new(Scripted::default());
    *executor.answer.lock().unwrap() = Some(Ok(RoutedAnswer::default()));
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

/// A refusal reaches the ingress with its code, message and details, so the
/// client is told what the executing member said.
#[tokio::test]
async fn a_refusal_crosses_with_its_details() {
    let executor = Arc::new(Scripted::default());
    let refused = Status::with_details(
        tonic::Code::Aborted,
        "the transaction ended",
        tonic::codegen::Bytes::from_static(b"details"),
    );
    *executor.answer.lock().unwrap() = Some(Err(refused));
    let router = serve(executor).await;
    let err = router
        .execute(statement(None), &AdmissionSignal::new(|| {}))
        .await
        .expect_err("refused");
    assert_eq!(err.code(), tonic::Code::Aborted);
    assert_eq!(err.message(), "the transaction ended");
    assert_eq!(err.details(), b"details");
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
            deadline: None,
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
        assert_eq!(err.code(), tonic::Code::Aborted);
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
