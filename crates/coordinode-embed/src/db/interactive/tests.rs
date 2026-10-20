use std::sync::Arc;

use coordinode_core::txn::command::Principal;
use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_core::txn::write_concern::WriteConcern;

use super::*;
use crate::Database;
use crate::operation::{IssuerId, OperationOutcome, Resolve};

fn open() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    (db, dir)
}

fn principal(subject: &str) -> ActingContext {
    ActingContext::Principal(Principal {
        subject: Arc::from(subject),
        trust_scope: Arc::from("host"),
        namespace: Arc::from("default"),
    })
}

fn count(db: &Database, label: &str) -> usize {
    db.execute_cypher_shared(
        &format!("MATCH (n:{label}) RETURN n"),
        None,
        None,
        None,
        None,
    )
    .expect("count")
    .rows
    .len()
}

/// A client operation of `db`'s, enrolled for the test.
fn operation(db: &Database) -> OperationDescriptor {
    db.enroll_issuer(IssuerId([0x42; 16]))
        .expect("enroll")
        .next()
        .expect("an operation")
}

/// Take `id`'s context as a command would, to hold it busy.
fn hold(db: &Database, id: &TransactionId) -> CheckedOut {
    db.interactive
        .check_out(id, &ActingContext::Trusted, None)
        .expect("hold the context")
}

/// Commands waiting for their turn on `id`, the running one not counted.
fn waiting(db: &Database, id: &TransactionId) -> u64 {
    db.interactive.contexts.lock().get(id).map_or(0, |context| {
        context.pending() - u64::from(context.state.is_none())
    })
}

/// Until `n` commands wait on `id`: the barrier a test crosses before it
/// releases them, with no sleep standing in for it.
fn until_waiting(db: &Database, id: &TransactionId, n: u64) {
    while waiting(db, id) < n {
        std::thread::yield_now();
    }
}

/// A statement reaching a transaction while another command runs on it waits
/// for its turn and then runs; it is neither refused nor told the
/// transaction is unknown.
#[test]
fn a_statement_waits_for_the_command_running_on_its_transaction() {
    let (db, _dir) = open();
    let id = db.begin_transaction().expect("begin");
    let running = hold(&db, &id);
    std::thread::scope(|s| {
        let next = s.spawn(|| db.execute_in_transaction(id, "CREATE (:Queued {k: 1})", None));
        until_waiting(&db, &id, 1);
        db.interactive
            .check_in(&id, running.state, running.chain)
            .expect("first command done");
        next.join()
            .expect("join")
            .expect("the waiting statement runs");
    });
    db.commit_transaction(id).expect("commit");
    assert_eq!(count(&db, "Queued"), 1);
}

/// Commands run in the order they reach the transaction, not in the order
/// a wakeup happens to favour: a statement that arrived second, and reads
/// what the first one writes, sees that write.
#[test]
fn commands_run_in_the_order_they_arrive() {
    let (db, _dir) = open();
    let id = db.begin_transaction().expect("begin");
    let running = hold(&db, &id);
    std::thread::scope(|s| {
        let first = s.spawn(|| db.execute_in_transaction(id, "CREATE (:Step {first: 1})", None));
        until_waiting(&db, &id, 1);
        let second = s.spawn(|| {
            db.execute_in_transaction(id, "MATCH (n:Step) SET n.second = n.first + 1", None)
        });
        until_waiting(&db, &id, 2);
        db.interactive
            .check_in(&id, running.state, running.chain)
            .expect("release");
        first.join().expect("join").expect("first");
        second.join().expect("join").expect("second");
    });
    db.commit_transaction(id).expect("commit");
    let rows = db
        .execute_cypher_shared(
            "MATCH (n:Step) RETURN n.second AS s",
            None,
            None,
            None,
            None,
        )
        .expect("read")
        .rows;
    assert_eq!(
        rows[0].get("s"),
        Some(&coordinode_core::graph::types::Value::Int(2)),
        "the second statement ran after the first one and saw its write"
    );
}

/// A commit sent while statements wait their turn commits after them: every
/// command that reached the transaction before it is in it.
#[test]
fn a_commit_runs_after_the_commands_that_arrived_before_it() {
    let (db, _dir) = open();
    let id = db.begin_transaction().expect("begin");
    let running = hold(&db, &id);
    std::thread::scope(|s| {
        let statement =
            s.spawn(|| db.execute_in_transaction(id, "CREATE (:BeforeCommit {k: 1})", None));
        until_waiting(&db, &id, 1);
        let commit = s.spawn(|| db.commit_transaction(id));
        until_waiting(&db, &id, 2);
        db.interactive
            .check_in(&id, running.state, running.chain)
            .expect("release");
        statement.join().expect("join").expect("the statement");
        commit.join().expect("join").expect("the commit");
    });
    assert_eq!(count(&db, "BeforeCommit"), 1);
}

/// A statement queued behind a running command is told it has its place
/// before it runs, so whoever sent it can pass the next command along
/// without waiting for its answer.
#[test]
fn a_queued_statement_is_told_it_has_its_place() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let (db, _dir) = open();
    let id = db.begin_transaction().expect("begin");
    let running = hold(&db, &id);
    let told = Arc::new(AtomicBool::new(false));
    let options = StatementOptions {
        admission: Some(AdmissionSignal::new({
            let told = Arc::clone(&told);
            move || told.store(true, Ordering::SeqCst)
        })),
        ..StatementOptions::default()
    };
    std::thread::scope(|s| {
        let queued =
            s.spawn(|| db.execute_in_transaction_with(id, "RETURN 1 AS one", None, &options));
        until_waiting(&db, &id, 1);
        assert!(
            told.load(Ordering::SeqCst),
            "told while still queued, not after it ran"
        );
        db.interactive
            .check_in(&id, running.state, running.chain)
            .expect("release");
        queued.join().expect("join").expect("then it runs");
    });
}

/// The owner holds a bounded queue: past it a command is refused as busy,
/// and the transaction stays usable.
#[test]
fn a_full_queue_refuses_the_next_command_as_busy() {
    let (db, _dir) = open();
    let id = db.begin_transaction().expect("begin");
    let running = hold(&db, &id);
    let set_next_turn = |turn: u64| {
        db.interactive
            .contexts
            .lock()
            .get_mut(&id)
            .expect("context")
            .next_turn = turn;
    };
    // The running command holds turn 0; a full queue holds the next MAILBOX.
    set_next_turn(MAILBOX + 1);
    assert!(matches!(
        db.execute_in_transaction(id, "RETURN 1 AS one", None),
        Err(DatabaseError::TransactionBusy(h)) if h == id
    ));
    set_next_turn(1);
    db.interactive
        .check_in(&id, running.state, running.chain)
        .expect("release");
    db.execute_in_transaction(id, "RETURN 1 AS one", None)
        .expect("usable again");
}

/// A failed statement ends the transaction: later commands for it are told
/// it was aborted, and nothing of it is applied.
#[test]
fn a_failed_statement_aborts_the_transaction() {
    let (db, _dir) = open();
    let id = db.begin_transaction().expect("begin");
    db.execute_in_transaction(id, "CREATE (:Failed {k: 1})", None)
        .expect("first");
    assert!(
        db.execute_in_transaction(id, "RETURN 1 / 0 AS boom", None)
            .is_err()
    );
    for refused in [
        db.execute_in_transaction(id, "RETURN 1 AS one", None)
            .map(drop),
        db.commit_transaction(id).map(drop),
    ] {
        assert!(
            matches!(
                refused,
                Err(DatabaseError::TransactionAborted {
                    end: TransactionEnd::StatementFailed,
                    ..
                })
            ),
            "{refused:?}"
        );
    }
    assert_eq!(count(&db, "Failed"), 0);
}

/// After a rollback, a rollback for a lost stream, or the idle sweep, a
/// command for the transaction says why it ended.
#[test]
fn an_ended_transaction_says_why() {
    let (db, _dir) = open();
    let rolled = db.begin_transaction().expect("begin");
    db.rollback_transaction(rolled).expect("rollback");
    assert!(matches!(
        db.execute_in_transaction(rolled, "RETURN 1 AS one", None),
        Err(DatabaseError::TransactionAborted {
            end: TransactionEnd::RolledBack,
            ..
        })
    ));

    let lost = db.begin_transaction().expect("begin");
    db.rollback_transaction_with(
        lost,
        &RollbackOptions {
            end: TransactionEnd::StreamLost,
            ..RollbackOptions::default()
        },
    )
    .expect("abandon");
    assert!(matches!(
        db.execute_in_transaction(lost, "RETURN 1 AS one", None),
        Err(DatabaseError::TransactionAborted {
            end: TransactionEnd::StreamLost,
            ..
        })
    ));

    let idle = db.begin_transaction().expect("begin");
    let busy = db.begin_transaction().expect("begin");
    let running = hold(&db, &busy);
    db.reap_idle_transactions(Duration::ZERO);
    assert!(matches!(
        db.commit_transaction(idle),
        Err(DatabaseError::TransactionAborted {
            end: TransactionEnd::IdleExpired,
            ..
        })
    ));
    db.interactive
        .check_in(&busy, running.state, running.chain)
        .expect("a context in use is never idle");
    db.commit_transaction(busy).expect("commit");
}

/// A transaction that commits at idle expiry is committed by the idle sweep,
/// under the operation its start named: its writes land, the operation
/// resolves as committed, and a later command is told it was committed
/// rather than that it was aborted.
#[test]
fn a_transaction_that_commits_on_timeout_is_committed_by_the_sweep() {
    let (db, _dir) = open();
    let op = operation(&db);
    let id = db
        .begin_transaction_with(&TransactionOptions {
            timeout_outcome: Some(TimeoutOutcome::Commit),
            operation: Some(op),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    db.execute_in_transaction(id, "CREATE (:Kept {k: 1})", None)
        .expect("statement");
    db.reap_idle_transactions(Duration::ZERO);
    assert_eq!(count(&db, "Kept"), 1, "the sweep committed it");
    assert!(matches!(
        db.resolve_operation(&op, Resolve::Lookup).expect("resolve"),
        OperationOutcome::Committed { .. }
    ));
    assert!(matches!(
        db.execute_in_transaction(id, "RETURN 1 AS one", None),
        Err(DatabaseError::TransactionCommittedOnTimeout(h)) if h == id
    ));
}

/// A commit at idle expiry that is refused applied nothing, and the
/// transaction is reported as ended by that refusal, never as committed:
/// expiry chose to commit, only the commit's result says whether it did.
#[test]
fn a_refused_commit_on_timeout_is_not_reported_as_committed() {
    let (db, _dir) = open();
    db.execute_cypher_shared("CREATE (:Account {k: 1, v: 0})", None, None, None, None)
        .expect("seed");
    let op = operation(&db);
    let id = db
        .begin_transaction_with(&TransactionOptions {
            timeout_outcome: Some(TimeoutOutcome::Commit),
            operation: Some(op),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    db.execute_in_transaction(id, "MATCH (n:Account {k: 1}) SET n.v = 1", None)
        .expect("statement");
    // A writer that commits first: the idle commit loses first-committer-wins.
    db.execute_cypher_shared(
        "MATCH (n:Account {k: 1}) SET n.v = 2",
        None,
        None,
        None,
        None,
    )
    .expect("concurrent write");
    db.reap_idle_transactions(Duration::ZERO);
    for refused in [
        db.execute_in_transaction(id, "RETURN 1 AS one", None)
            .map(drop),
        db.commit_transaction(id).map(drop),
    ] {
        assert!(
            matches!(
                refused,
                Err(DatabaseError::TransactionAborted {
                    end: TransactionEnd::IdleCommitRefused,
                    ..
                })
            ),
            "{refused:?}"
        );
    }
    let rows = db
        .execute_cypher_shared(
            "MATCH (n:Account {k: 1}) RETURN n.v AS v",
            None,
            None,
            None,
            None,
        )
        .expect("read")
        .rows;
    assert_eq!(
        rows[0].get("v"),
        Some(&coordinode_core::graph::types::Value::Int(2)),
        "the refused commit applied nothing"
    );
    assert!(!matches!(
        db.resolve_operation(&op, Resolve::Lookup).expect("resolve"),
        OperationOutcome::Committed { .. }
    ));
}

/// What the next proposal through a [`Gate`] meets.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Armed {
    Open,
    /// Held until [`Gate::release`], then applied.
    Hold,
    /// Applied, then reported failed the way a leader that lost its
    /// authority after handing the entry to its log reports it.
    LoseReply,
}

type Proposed = Result<
    coordinode_core::txn::proposal::ProposalOutcome,
    coordinode_core::txn::proposal::ProposalError,
>;

/// The database's write pipeline with a controllable next proposal.
struct Gate {
    inner: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline>,
    armed: Mutex<Armed>,
    /// `(entered, released)` of a held proposal.
    held: Mutex<(bool, bool)>,
    changed: Condvar,
}

impl Gate {
    /// Put a gate in front of `db`'s pipeline.
    fn install(db: &mut Database) -> Arc<Self> {
        let gate = Arc::new(Self {
            inner: Arc::clone(&db.pipeline),
            armed: Mutex::new(Armed::Open),
            held: Mutex::new((false, false)),
            changed: Condvar::new(),
        });
        db.pipeline =
            Arc::clone(&gate) as Arc<dyn coordinode_core::txn::proposal::ProposalPipeline>;
        gate
    }

    fn arm(&self, armed: Armed) {
        *self.armed.lock() = armed;
    }

    /// Until a held proposal has entered the pipeline.
    fn until_entered(&self) {
        let mut held = self.held.lock();
        while !held.0 {
            self.changed.wait(&mut held);
        }
    }

    fn release(&self) {
        self.held.lock().1 = true;
        self.changed.notify_all();
    }

    fn pass(&self, apply: impl FnOnce() -> Proposed) -> Proposed {
        let armed = core::mem::replace(&mut *self.armed.lock(), Armed::Open);
        match armed {
            Armed::Open => apply(),
            Armed::Hold => {
                let mut held = self.held.lock();
                held.0 = true;
                self.changed.notify_all();
                while !held.1 {
                    self.changed.wait(&mut held);
                }
                drop(held);
                apply()
            }
            Armed::LoseReply => {
                apply()?;
                Err(coordinode_core::txn::proposal::ProposalError::LeadershipLost { term: 1 })
            }
        }
    }
}

impl coordinode_core::txn::proposal::ProposalPipeline for Gate {
    fn propose_and_wait(
        &self,
        proposal: &coordinode_core::txn::proposal::RaftProposal,
    ) -> Proposed {
        self.pass(|| self.inner.propose_and_wait(proposal))
    }

    fn propose_with_timeout(
        &self,
        proposal: &coordinode_core::txn::proposal::RaftProposal,
        timeout: Duration,
    ) -> Proposed {
        self.pass(|| self.inner.propose_with_timeout(proposal, timeout))
    }

    fn propose_with_concern(
        &self,
        proposal: &coordinode_core::txn::proposal::RaftProposal,
        concern: &WriteConcern,
    ) -> Proposed {
        self.pass(|| self.inner.propose_with_concern(proposal, concern))
    }
}

/// A transaction that commits on timeout, under an operation, with one write.
fn committing_on_timeout(db: &Database, label: &str) -> (TransactionId, OperationDescriptor) {
    let op = operation(db);
    let id = db
        .begin_transaction_with(&TransactionOptions {
            timeout_outcome: Some(TimeoutOutcome::Commit),
            operation: Some(op),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    db.execute_in_transaction(id, &format!("CREATE (:{label} {{k: 1}})"), None)
        .expect("statement");
    (id, op)
}

/// While the commit made at idle expiry is in flight the transaction is
/// neither open nor ended: a command arriving then waits for the commit and
/// is told its result, and nothing reports the transaction committed before
/// the commit has.
#[test]
fn a_command_during_a_commit_on_timeout_waits_for_its_result() {
    let (mut db, _dir) = open();
    let gate = Gate::install(&mut db);
    let (id, _op) = committing_on_timeout(&db, "Held");
    gate.arm(Armed::Hold);
    std::thread::scope(|s| {
        let sweep = s.spawn(|| db.reap_idle_transactions(Duration::ZERO));
        gate.until_entered();
        assert_eq!(
            db.interactive.ended_with(&id),
            None,
            "not reported ended before the commit's result"
        );
        let late = s.spawn(|| db.execute_in_transaction(id, "RETURN 1 AS one", None));
        until_waiting(&db, &id, 1);
        gate.release();
        sweep.join().expect("join");
        assert!(matches!(
            late.join().expect("join"),
            Err(DatabaseError::TransactionCommittedOnTimeout(h)) if h == id
        ));
    });
    assert_eq!(count(&db, "Held"), 1);
}

/// A commit made at idle expiry that failed after its entry may have been
/// applied is reported with an unknown outcome, never as aborted, and its
/// operation tells how it went: here the write had landed.
#[test]
fn a_commit_on_timeout_whose_reply_is_lost_has_an_unknown_outcome() {
    let (mut db, _dir) = open();
    let gate = Gate::install(&mut db);
    let (id, op) = committing_on_timeout(&db, "Landed");
    gate.arm(Armed::LoseReply);
    db.reap_idle_transactions(Duration::ZERO);
    for refused in [
        db.execute_in_transaction(id, "RETURN 1 AS one", None)
            .map(drop),
        db.rollback_transaction(id),
    ] {
        assert!(
            matches!(refused, Err(DatabaseError::TransactionOutcomeUnknown(h)) if h == id),
            "{refused:?}"
        );
    }
    assert!(matches!(
        db.resolve_operation(&op, Resolve::Lookup).expect("resolve"),
        OperationOutcome::Committed { .. }
    ));
    assert_eq!(count(&db, "Landed"), 1);
}

/// An early successful w:0 acknowledgement is not proof of COMMIT. Model
/// the interval after acceptance but before the background apply: the
/// operation remains unknown and the timeout must not report committed.
#[test]
fn an_early_ack_on_timeout_is_not_reported_as_committed() {
    use coordinode_core::txn::proposal::{ProposalOutcome, ProposalPipeline, RaftProposal};

    struct EarlyAck(Arc<dyn ProposalPipeline>);
    impl ProposalPipeline for EarlyAck {
        fn propose_and_wait(&self, proposal: &RaftProposal) -> Proposed {
            self.0.propose_and_wait(proposal)
        }

        fn propose_with_concern(
            &self,
            proposal: &RaftProposal,
            concern: &WriteConcern,
        ) -> Proposed {
            if concern.w.is_fire_and_forget() {
                return Ok(ProposalOutcome::local());
            }
            self.0.propose_with_concern(proposal, concern)
        }
    }

    let (mut db, _dir) = open();
    let op = operation(&db);
    let id = db
        .begin_transaction_with(&TransactionOptions {
            timeout_outcome: Some(TimeoutOutcome::Commit),
            operation: Some(op),
            write_concern: Some(WriteConcern::w0()),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    db.execute_in_transaction(id, "CREATE (:EarlyAck {k: 1})", None)
        .expect("statement");
    db.pipeline = Arc::new(EarlyAck(Arc::clone(&db.pipeline)));
    db.reap_idle_transactions(Duration::ZERO);
    assert_eq!(
        db.resolve_operation(&op, Resolve::Lookup).expect("resolve"),
        OperationOutcome::Unknown
    );
    assert!(matches!(
        db.execute_in_transaction(id, "RETURN 1 AS one", None),
        Err(DatabaseError::TransactionOutcomeUnknown(h)) if h == id
    ));
}

/// How a commit at idle expiry ended is read from its result: success is a
/// candidate for committed, which the sweep confirms against the operation's
/// record, a refusal that proves nothing was applied is refused, and any
/// failure that proves nothing is unknown, for the client to resolve by its
/// operation. A settled operation is unknown too: its record, not this
/// attempt, says whether the operation took effect.
#[test]
fn the_end_of_a_commit_on_timeout_follows_its_result() {
    let receipt = coordinode_core::txn::transaction::CommitReceipt {
        commit_ts: coordinode_core::txn::timestamp::Timestamp::from_raw(7),
        applied_index: None,
    };
    let id = TransactionId::from_raw([0x11; 16]);
    assert_eq!(idle_commit_end(&Ok(receipt)), TransactionEnd::IdleCommitted);
    assert_eq!(
        idle_commit_end(&Err(DatabaseError::TransactionConflict {
            id,
            source_message: "lost".into(),
        })),
        TransactionEnd::IdleCommitRefused
    );
    assert_eq!(
        idle_commit_end(&Err(DatabaseError::NotLeader { leader_id: None })),
        TransactionEnd::IdleCommitRefused
    );
    assert_eq!(
        idle_commit_end(&Err(DatabaseError::Other("pipeline failed".into()))),
        TransactionEnd::IdleCommitUnknown
    );
    let op = coordinode_core::txn::operation::OperationId {
        issuer: IssuerId([0x12; 16]),
        incarnation: 1,
        sequence: 1,
    };
    assert_eq!(
        idle_commit_end(&Err(DatabaseError::OperationSettled {
            op,
            outcome: OperationOutcome::Unknown,
            intent_differs: false,
        })),
        TransactionEnd::IdleCommitUnknown
    );
}

/// The database's timeout outcome applies to a transaction whose start
/// names none, and a start that would commit at expiry without an operation
/// to resolve it by is refused before anything opens.
#[test]
fn a_commit_on_timeout_needs_an_operation() {
    let (db, _dir) = open();
    db.set_interactive_timeout_outcome(TimeoutOutcome::Commit);
    let id = TransactionId::from_raw([0x7c; 16]);
    assert!(matches!(
        db.begin_transaction_as(id, &TransactionOptions::trusted()),
        Err(DatabaseError::OperationRequired(h)) if h == id
    ));
    assert!(matches!(
        db.execute_in_transaction(id, "RETURN 1 AS one", None),
        Err(DatabaseError::UnknownTransaction(_))
    ));
    db.begin_transaction_as(
        id,
        &TransactionOptions {
            timeout_outcome: Some(TimeoutOutcome::Rollback),
            ..TransactionOptions::trusted()
        },
    )
    .expect("a rollback outcome needs none");
}

/// A transaction one of whose statements failed is rolled back at expiry
/// even when it commits on timeout: nothing it ran is published.
#[test]
fn a_failed_transaction_is_never_committed_on_timeout() {
    let (db, _dir) = open();
    let op = operation(&db);
    let id = db
        .begin_transaction_with(&TransactionOptions {
            timeout_outcome: Some(TimeoutOutcome::Commit),
            operation: Some(op),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    db.execute_in_transaction(id, "CREATE (:Doomed {k: 1})", None)
        .expect("statement");
    assert!(
        db.execute_in_transaction(id, "RETURN 1 / 0 AS boom", None)
            .is_err()
    );
    db.reap_idle_transactions(Duration::ZERO);
    assert_eq!(count(&db, "Doomed"), 0);
    assert!(!matches!(
        db.resolve_operation(&op, Resolve::Lookup).expect("resolve"),
        OperationOutcome::Committed { .. }
    ));
}

/// The commit of a transaction started under an operation binds it without
/// naming it again, and one naming another operation is refused and leaves
/// the transaction open.
#[test]
fn a_commit_binds_the_operation_its_start_named() {
    let (db, _dir) = open();
    let issuer = db.enroll_issuer(IssuerId([0x43; 16])).expect("enroll");
    let started = issuer.next().expect("operation");
    let other = issuer.next().expect("operation");
    let id = db
        .begin_transaction_with(&TransactionOptions {
            operation: Some(started),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    db.execute_in_transaction(id, "CREATE (:Bound {k: 1})", None)
        .expect("statement");
    assert!(matches!(
        db.commit_transaction_as(id, &other),
        Err(DatabaseError::OperationDiffers(h)) if h == id
    ));
    db.commit_transaction(id).expect("commit");
    assert!(matches!(
        db.resolve_operation(&started, Resolve::Lookup)
            .expect("resolve"),
        OperationOutcome::Committed { .. }
    ));
    assert_eq!(count(&db, "Bound"), 1);
}

/// The end of the tenure that opened a transaction drops it: a command for
/// it is told no such transaction is known, as a new leader that never held
/// it tells it, nothing of it is applied, and the same member leading again
/// does not bring it back.
#[test]
fn a_tenure_change_drops_open_transactions_for_good() {
    let (db, _dir) = open();
    let id = db.begin_transaction().expect("begin");
    db.execute_in_transaction(id, "CREATE (:Lost {k: 1})", None)
        .expect("statement");

    db.set_execution_tenure(7, Some(7), 41);
    for refused in [
        db.execute_in_transaction(id, "CREATE (:Lost {k: 2})", None)
            .map(drop),
        db.commit_transaction(id).map(drop),
        db.rollback_transaction(id),
    ] {
        assert!(
            matches!(refused, Err(DatabaseError::UnknownTransaction(h)) if h == id),
            "{refused:?}"
        );
    }

    db.set_execution_tenure(7, Some(7), 42);
    assert!(matches!(
        db.commit_transaction(id),
        Err(DatabaseError::UnknownTransaction(_))
    ));
    assert_eq!(count(&db, "Lost"), 0, "nothing of it was applied");

    let fresh = db.begin_transaction().expect("a new transaction");
    assert_ne!(fresh, id);
    db.execute_in_transaction(fresh, "CREATE (:Lost {k: 3})", None)
        .expect("statement");
    db.commit_transaction(fresh).expect("commit");
    assert_eq!(count(&db, "Lost"), 1);
}

/// A command still running when its tenure ends cannot put the transaction
/// back, and commands waiting behind it are told no such transaction is
/// known.
#[test]
fn a_running_command_cannot_revive_a_dropped_transaction() {
    let (db, _dir) = open();
    let id = db.begin_transaction().expect("begin");
    let running = hold(&db, &id);
    std::thread::scope(|s| {
        let queued = s.spawn(|| db.execute_in_transaction(id, "RETURN 1 AS one", None));
        until_waiting(&db, &id, 1);
        db.set_execution_tenure(3, Some(3), 9);
        assert!(matches!(
            queued.join().expect("join"),
            Err(DatabaseError::UnknownTransaction(_))
        ));
    });
    assert!(matches!(
        db.interactive.check_in(&id, running.state, running.chain),
        Err(DatabaseError::UnknownTransaction(_))
    ));
    assert!(matches!(
        db.commit_transaction(id),
        Err(DatabaseError::UnknownTransaction(_))
    ));
}

/// A member that does not lead holds no transactions: BEGIN is refused
/// naming the leader it knows of, or none during an election, and nothing is
/// opened.
#[test]
fn a_member_without_a_tenure_refuses_begin() {
    let (db, _dir) = open();
    db.set_execution_tenure(2, Some(5), 7);
    assert!(matches!(
        db.begin_transaction(),
        Err(DatabaseError::NotLeader { leader_id: Some(5) })
    ));
    db.set_execution_tenure(2, None, 8);
    assert!(matches!(
        db.begin_transaction(),
        Err(DatabaseError::NotLeader { leader_id: None })
    ));
    db.set_execution_tenure(2, Some(2), 9);
    let id = db.begin_transaction().expect("leads now");
    db.execute_in_transaction(id, "RETURN 1 AS one", None)
        .expect("the transaction is open on the leader");
}

/// A transaction is used only by the caller that began it: another caller,
/// or the host's trusted access, is told nothing is known of it, which does
/// not reveal that it exists.
#[test]
fn another_caller_cannot_use_a_transaction() {
    let (db, _dir) = open();
    let id = db
        .begin_transaction_with(&TransactionOptions {
            actor: Some(principal("alice")),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    let as_bob = StatementOptions {
        actor: Some(principal("bob")),
        ..StatementOptions::default()
    };
    let as_alice = StatementOptions {
        actor: Some(principal("alice")),
        ..StatementOptions::default()
    };
    let bob = principal("bob");
    let alice = principal("alice");

    for refused in [
        db.execute_in_transaction_with(id, "CREATE (:Owned {k: 1})", None, &as_bob)
            .map(drop),
        db.execute_in_transaction(id, "CREATE (:Owned {k: 1})", None)
            .map(drop),
        db.commit_transaction_with(
            id,
            &CommitOptions {
                actor: Some(&bob),
                ..CommitOptions::default()
            },
        )
        .map(drop),
        db.rollback_transaction_with(
            id,
            &RollbackOptions {
                actor: Some(&bob),
                ..RollbackOptions::default()
            },
        ),
    ] {
        assert!(
            matches!(refused, Err(DatabaseError::UnknownTransaction(h)) if h == id),
            "{refused:?}"
        );
    }

    db.execute_in_transaction_with(id, "CREATE (:Owned {k: 1})", None, &as_alice)
        .expect("the owner's statement");
    db.commit_transaction_with(
        id,
        &CommitOptions {
            actor: Some(&alice),
            ..CommitOptions::default()
        },
    )
    .expect("the owner's commit");
    assert_eq!(count(&db, "Owned"), 1);
}

/// A transaction of an earlier open of the same store names nothing in this
/// one.
#[test]
fn a_transaction_of_an_earlier_open_names_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let earlier = {
        let db = Database::open(dir.path()).expect("open");
        db.begin_transaction().expect("begin")
    };
    let db = Database::open(dir.path()).expect("reopen");
    let fresh = db.begin_transaction().expect("begin");
    assert_ne!(fresh, earlier, "every transaction draws a new ID");
    assert!(matches!(
        db.execute_in_transaction(earlier, "CREATE (:Stale {k: 1})", None),
        Err(DatabaseError::UnknownTransaction(h)) if h == earlier
    ));
}

/// A transaction opens under the ID its client drew, and runs under it.
#[test]
fn a_transaction_opens_under_the_id_its_client_drew() {
    let (db, _dir) = open();
    let id = TransactionId::from_raw([0x5a; 16]);
    db.begin_transaction_as(id, &TransactionOptions::trusted())
        .expect("begin");
    db.execute_in_transaction(id, "CREATE (:Drawn {k: 1})", None)
        .expect("statement");
    db.commit_transaction(id).expect("commit");
    assert_eq!(count(&db, "Drawn"), 1);
}

/// An ID names one transaction: a start under an ID that is open, or that
/// ended recently, is refused and leaves that transaction as it was; it is
/// never joined to it.
#[test]
fn a_start_under_an_id_in_use_is_refused() {
    let (db, _dir) = open();
    let id = TransactionId::from_raw([0x11; 16]);
    db.begin_transaction_as(id, &TransactionOptions::trusted())
        .expect("begin");
    db.execute_in_transaction(id, "CREATE (:Taken {k: 1})", None)
        .expect("statement");
    assert!(matches!(
        db.begin_transaction_as(id, &TransactionOptions::trusted()),
        Err(DatabaseError::TransactionIdInUse(taken)) if taken == id
    ));
    db.execute_in_transaction(id, "MATCH (n:Taken) RETURN n", None)
        .expect("the open transaction is untouched");
    db.rollback_transaction(id).expect("rollback");
    assert!(matches!(
        db.begin_transaction_as(id, &TransactionOptions::trusted()),
        Err(DatabaseError::TransactionIdInUse(_))
    ));
    assert_eq!(count(&db, "Taken"), 0);
}

/// The concerns a transaction began with are its own until it ends.
#[test]
fn a_transaction_keeps_the_concerns_it_began_with() {
    let (mut db, _dir) = open();
    let id = db
        .begin_transaction_with(&TransactionOptions {
            read_concern: Some(ReadConcernLevel::Majority),
            write_concern: Some(WriteConcern::memory()),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    db.set_write_concern(WriteConcern::majority());
    let checked = hold(&db, &id);
    assert_eq!(checked.read_concern, ReadConcernLevel::Majority);
    assert_eq!(checked.write_concern, WriteConcern::memory());
    db.interactive
        .check_in(&id, checked.state, checked.chain)
        .expect("check in");
}
