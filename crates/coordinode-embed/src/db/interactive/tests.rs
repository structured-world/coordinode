use std::sync::Arc;

use coordinode_core::txn::command::Principal;
use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_core::txn::write_concern::WriteConcern;

use super::*;
use crate::Database;

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

fn ordered(drain: Duration) -> TransactionOptions {
    TransactionOptions {
        ordering: TxnOrdering::Ordered,
        drain_timeout: Some(drain),
        ..TransactionOptions::trusted()
    }
}

fn numbered(nonce: u64) -> StatementOptions {
    StatementOptions {
        nonce: Some(nonce),
        ..StatementOptions::default()
    }
}

/// Take `handle`'s context as a command would, to hold it busy.
fn hold(db: &Database, handle: &TransactionId) -> CheckedOut {
    db.interactive
        .check_out(
            handle,
            &ActingContext::Trusted,
            None,
            &crate::operation::EMPTY_CHAIN,
            None,
        )
        .expect("hold the context")
}

/// Commands waiting for their turn on `handle`.
fn waiting(db: &Database, handle: &TransactionId) -> usize {
    db.interactive
        .contexts
        .lock()
        .get(handle)
        .map_or(0, |context| context.waiting)
}

/// Until `n` commands wait on `handle`: the barrier a test crosses before it
/// releases them, with no sleep standing in for it.
fn until_waiting(db: &Database, handle: &TransactionId, n: usize) {
    while waiting(db, handle) < n {
        std::thread::yield_now();
    }
}

/// A statement reaching a transaction while another command runs on it waits
/// for its turn and then runs; it is neither refused nor told the
/// transaction is unknown.
#[test]
fn a_statement_waits_for_the_command_running_on_its_transaction() {
    let (db, _dir) = open();
    let handle = db.begin_transaction().expect("begin");
    let running = hold(&db, &handle);
    std::thread::scope(|s| {
        let next = s.spawn(|| db.execute_in_transaction(handle, "CREATE (:Queued {k: 1})", None));
        until_waiting(&db, &handle, 1);
        db.interactive
            .check_in(&handle, running.state, running.chain)
            .expect("first command done");
        next.join()
            .expect("join")
            .expect("the waiting statement runs");
    });
    db.commit_transaction(handle).expect("commit");
    assert_eq!(count(&db, "Queued"), 1);
}

/// A statement queued behind a running command is told it has its place
/// before it runs, so whoever sent it can pass the next command along
/// without waiting for its answer.
#[test]
fn a_queued_statement_is_told_it_has_its_place() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let (db, _dir) = open();
    let handle = db.begin_transaction().expect("begin");
    let running = hold(&db, &handle);
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
            s.spawn(|| db.execute_in_transaction_with(handle, "RETURN 1 AS one", None, &options));
        until_waiting(&db, &handle, 1);
        assert!(
            told.load(Ordering::SeqCst),
            "told while still queued, not after it ran"
        );
        db.interactive
            .check_in(&handle, running.state, running.chain)
            .expect("release");
        queued.join().expect("join").expect("then it runs");
    });
}

/// The owner holds a bounded queue: past it a command is refused as busy,
/// and the transaction stays usable.
#[test]
fn a_full_queue_refuses_the_next_command_as_busy() {
    let (db, _dir) = open();
    let handle = db.begin_transaction().expect("begin");
    let running = hold(&db, &handle);
    db.interactive
        .contexts
        .lock()
        .get_mut(&handle)
        .expect("context")
        .waiting = MAILBOX;
    assert!(matches!(
        db.execute_in_transaction(handle, "RETURN 1 AS one", None),
        Err(DatabaseError::TransactionBusy(h)) if h == handle
    ));
    db.interactive
        .contexts
        .lock()
        .get_mut(&handle)
        .expect("context")
        .waiting = 0;
    db.interactive
        .check_in(&handle, running.state, running.chain)
        .expect("release");
    db.execute_in_transaction(handle, "RETURN 1 AS one", None)
        .expect("usable again");
}

/// An ordered transaction applies its statements by nonce whatever order
/// they arrive in: a later nonce waits for the earlier one.
#[test]
fn an_ordered_transaction_applies_statements_by_nonce() {
    let (db, _dir) = open();
    let handle = db
        .begin_transaction_with(&ordered(Duration::from_secs(30)))
        .expect("begin");
    std::thread::scope(|s| {
        let second = s.spawn(|| {
            db.execute_in_transaction_with(
                handle,
                "MATCH (n:Step) SET n.second = n.first + 1",
                None,
                &numbered(2),
            )
        });
        until_waiting(&db, &handle, 1);
        db.execute_in_transaction_with(handle, "CREATE (:Step {first: 1})", None, &numbered(1))
            .expect("first");
        second
            .join()
            .expect("join")
            .expect("second, after the first");
    });
    db.commit_transaction_with(
        handle,
        &CommitOptions {
            last_nonce: Some(2),
            ..CommitOptions::default()
        },
    )
    .expect("commit");
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
        "the second statement saw the first one's write"
    );
}

/// A retransmitted statement is applied once: the repeat is refused, and a
/// nonce reused for a different statement is refused too.
#[test]
fn a_repeated_nonce_is_never_applied_twice() {
    for options in [
        TransactionOptions::trusted(),
        ordered(Duration::from_secs(30)),
    ] {
        let (db, _dir) = open();
        let handle = db.begin_transaction_with(&options).expect("begin");
        db.execute_in_transaction_with(handle, "CREATE (:Once {k: 1})", None, &numbered(1))
            .expect("first send");
        assert!(matches!(
            db.execute_in_transaction_with(handle, "CREATE (:Once {k: 1})", None, &numbered(1)),
            Err(DatabaseError::StatementAlreadyAdmitted { nonce: 1, .. })
        ));
        assert!(matches!(
            db.execute_in_transaction_with(handle, "CREATE (:Once {k: 2})", None, &numbered(1)),
            Err(DatabaseError::StatementNonceReused { nonce: 1, .. })
        ));
        db.commit_transaction(handle).expect("commit");
        assert_eq!(count(&db, "Once"), 1, "{:?}", options.ordering);
    }
}

/// An ordered transaction needs every statement numbered.
#[test]
fn an_ordered_statement_without_a_nonce_is_refused() {
    let (db, _dir) = open();
    let handle = db
        .begin_transaction_with(&ordered(Duration::from_secs(30)))
        .expect("begin");
    for nonce in [None, Some(0)] {
        let options = StatementOptions {
            nonce,
            ..StatementOptions::default()
        };
        assert!(matches!(
            db.execute_in_transaction_with(handle, "RETURN 1 AS one", None, &options),
            Err(DatabaseError::StatementNonceRequired(h)) if h == handle
        ));
    }
    db.execute_in_transaction_with(handle, "RETURN 1 AS one", None, &numbered(1))
        .expect("numbered");
}

/// A missing statement that never comes ends an ordered transaction once its
/// drain timeout passes, whether a later statement or the commit waits for
/// it; nothing of it is applied.
#[test]
fn a_gap_past_the_drain_timeout_aborts_the_transaction() {
    let (db, _dir) = open();
    let handle = db
        .begin_transaction_with(&ordered(Duration::from_millis(50)))
        .expect("begin");
    db.execute_in_transaction_with(handle, "CREATE (:Gap {k: 1})", None, &numbered(1))
        .expect("first");
    assert!(matches!(
        db.execute_in_transaction_with(handle, "CREATE (:Gap {k: 3})", None, &numbered(3)),
        Err(DatabaseError::TransactionAborted {
            end: TransactionEnd::DrainTimedOut,
            ..
        })
    ));
    assert!(matches!(
        db.commit_transaction(handle),
        Err(DatabaseError::TransactionAborted {
            end: TransactionEnd::DrainTimedOut,
            ..
        })
    ));

    let handle = db
        .begin_transaction_with(&ordered(Duration::from_millis(50)))
        .expect("begin");
    db.execute_in_transaction_with(handle, "CREATE (:Gap {k: 1})", None, &numbered(1))
        .expect("first");
    assert!(matches!(
        db.commit_transaction_with(
            handle,
            &CommitOptions {
                last_nonce: Some(2),
                ..CommitOptions::default()
            }
        ),
        Err(DatabaseError::TransactionAborted {
            end: TransactionEnd::DrainTimedOut,
            ..
        })
    ));
    assert_eq!(count(&db, "Gap"), 0);
}

/// A transaction whose BEGIN names no drain bound takes the database's,
/// as the server's configuration sets it.
#[test]
fn the_database_drain_bound_applies_when_begin_names_none() {
    let (db, _dir) = open();
    db.set_interactive_drain_timeout(Duration::from_millis(30));
    let handle = db
        .begin_transaction_with(&TransactionOptions {
            ordering: TxnOrdering::Ordered,
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    let started = Instant::now();
    assert!(matches!(
        db.execute_in_transaction_with(handle, "RETURN 1 AS one", None, &numbered(2)),
        Err(DatabaseError::TransactionAborted {
            end: TransactionEnd::DrainTimedOut,
            ..
        })
    ));
    assert!(
        started.elapsed() < DEFAULT_DRAIN_TIMEOUT,
        "the database's bound, not the built-in one"
    );
}

/// A commit sent before the chain's last statement waits for it, then
/// commits all of it.
#[test]
fn a_commit_drains_the_chain_up_to_its_last_nonce() {
    let (db, _dir) = open();
    let handle = db
        .begin_transaction_with(&ordered(Duration::from_secs(30)))
        .expect("begin");
    std::thread::scope(|s| {
        let commit = s.spawn(|| {
            db.commit_transaction_with(
                handle,
                &CommitOptions {
                    last_nonce: Some(1),
                    ..CommitOptions::default()
                },
            )
        });
        while !db
            .interactive
            .contexts
            .lock()
            .get(&handle)
            .is_some_and(|context| context.draining)
        {
            std::thread::yield_now();
        }
        db.execute_in_transaction_with(handle, "CREATE (:Drained {k: 1})", None, &numbered(1))
            .expect("the last statement");
        commit
            .join()
            .expect("join")
            .expect("commit after the chain");
    });
    assert_eq!(count(&db, "Drained"), 1);
}

/// A failed statement ends the transaction: later commands for it are told
/// it was aborted, and nothing of it is applied.
#[test]
fn a_failed_statement_aborts_the_transaction() {
    let (db, _dir) = open();
    let handle = db.begin_transaction().expect("begin");
    db.execute_in_transaction(handle, "CREATE (:Failed {k: 1})", None)
        .expect("first");
    assert!(
        db.execute_in_transaction(handle, "RETURN 1 / 0 AS boom", None)
            .is_err()
    );
    for refused in [
        db.execute_in_transaction(handle, "RETURN 1 AS one", None)
            .map(drop),
        db.commit_transaction(handle).map(drop),
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

/// After a rollback or the idle sweep, a command for the transaction says
/// why it ended.
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

/// The end of the tenure that opened a transaction drops it: a command for
/// it is told no such transaction is known, as a new leader that never held
/// it tells it, nothing of it is applied, and the same member leading again
/// does not bring it back.
#[test]
fn a_tenure_change_drops_open_transactions_for_good() {
    let (db, _dir) = open();
    let handle = db.begin_transaction().expect("begin");
    db.execute_in_transaction(handle, "CREATE (:Lost {k: 1})", None)
        .expect("statement");

    db.set_execution_tenure(7, Some(7), 41);
    for refused in [
        db.execute_in_transaction(handle, "CREATE (:Lost {k: 2})", None)
            .map(drop),
        db.commit_transaction(handle).map(drop),
        db.rollback_transaction(handle),
    ] {
        assert!(
            matches!(refused, Err(DatabaseError::UnknownTransaction(id)) if id == handle),
            "{refused:?}"
        );
    }

    db.set_execution_tenure(7, Some(7), 42);
    assert!(matches!(
        db.commit_transaction(handle),
        Err(DatabaseError::UnknownTransaction(_))
    ));
    assert_eq!(count(&db, "Lost"), 0, "nothing of it was applied");

    let fresh = db.begin_transaction().expect("a new transaction");
    assert_ne!(fresh, handle);
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
    let handle = db.begin_transaction().expect("begin");
    let running = hold(&db, &handle);
    std::thread::scope(|s| {
        let queued = s.spawn(|| db.execute_in_transaction(handle, "RETURN 1 AS one", None));
        until_waiting(&db, &handle, 1);
        db.set_execution_tenure(3, Some(3), 9);
        assert!(matches!(
            queued.join().expect("join"),
            Err(DatabaseError::UnknownTransaction(_))
        ));
    });
    assert!(matches!(
        db.interactive
            .check_in(&handle, running.state, running.chain),
        Err(DatabaseError::UnknownTransaction(_))
    ));
    assert!(matches!(
        db.commit_transaction(handle),
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
    let handle = db.begin_transaction().expect("leads now");
    db.execute_in_transaction(handle, "RETURN 1 AS one", None)
        .expect("the transaction is open on the leader");
}

/// A transaction is used only by the caller that began it: another caller,
/// or the host's trusted access, is told nothing is known of it, which does
/// not reveal that it exists.
#[test]
fn another_caller_cannot_use_a_transaction() {
    let (db, _dir) = open();
    let handle = db
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
        db.execute_in_transaction_with(handle, "CREATE (:Owned {k: 1})", None, &as_bob)
            .map(drop),
        db.execute_in_transaction(handle, "CREATE (:Owned {k: 1})", None)
            .map(drop),
        db.commit_transaction_with(
            handle,
            &CommitOptions {
                actor: Some(&bob),
                ..CommitOptions::default()
            },
        )
        .map(drop),
        db.rollback_transaction_with(
            handle,
            &RollbackOptions {
                actor: Some(&bob),
                ..RollbackOptions::default()
            },
        ),
    ] {
        assert!(
            matches!(refused, Err(DatabaseError::UnknownTransaction(h)) if h == handle),
            "{refused:?}"
        );
    }

    db.execute_in_transaction_with(handle, "CREATE (:Owned {k: 1})", None, &as_alice)
        .expect("the owner's statement");
    db.commit_transaction_with(
        handle,
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
    let handle = {
        let db = Database::open(dir.path()).expect("open");
        db.begin_transaction().expect("begin")
    };
    let db = Database::open(dir.path()).expect("reopen");
    let fresh = db.begin_transaction().expect("begin");
    assert_ne!(fresh, handle, "every transaction draws a new ID");
    assert!(matches!(
        db.execute_in_transaction(handle, "CREATE (:Stale {k: 1})", None),
        Err(DatabaseError::UnknownTransaction(h)) if h == handle
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
    let handle = db
        .begin_transaction_with(&TransactionOptions {
            read_concern: Some(ReadConcernLevel::Majority),
            write_concern: Some(WriteConcern::memory()),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    db.set_write_concern(WriteConcern::majority());
    let checked = hold(&db, &handle);
    assert_eq!(checked.read_concern, ReadConcernLevel::Majority);
    assert_eq!(checked.write_concern, WriteConcern::memory());
    db.interactive
        .check_in(&handle, checked.state, checked.chain)
        .expect("check in");
}
