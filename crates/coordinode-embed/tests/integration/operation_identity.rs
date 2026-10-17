//! Client operation identity through the embedded API: a write carrying a
//! descriptor takes effect at most once, however it is retried, and the
//! host can find out how it stands after a lost reply or a restart.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;

use coordinode_core::graph::types::Value;
use coordinode_embed::db::StatementOptions;
use coordinode_embed::operation::{
    IssuerId, OperationDescriptor, OperationIssuer, OperationOutcome, Resolve,
};
use coordinode_embed::{Database, DatabaseError};

const ISSUER: IssuerId = IssuerId([0x42; 16]);

fn run(db: &Database, op: &OperationDescriptor, query: &str) -> Result<(), DatabaseError> {
    db.execute_cypher_shared_with(
        query,
        None,
        None,
        &StatementOptions {
            operation: Some(*op),
            ..StatementOptions::default()
        },
    )
    .map(drop)
}

fn count(db: &mut Database, label: &str) -> usize {
    db.execute_cypher(&format!("MATCH (n:{label}) RETURN n"))
        .unwrap()
        .len()
}

fn settled(result: Result<(), DatabaseError>) -> (OperationOutcome, bool) {
    match result {
        Err(DatabaseError::OperationSettled {
            outcome,
            intent_differs,
            ..
        }) => (outcome, intent_differs),
        other => panic!("expected the operation to be settled, got {other:?}"),
    }
}

fn enrolled(db: &Database) -> OperationIssuer {
    db.enroll_issuer(ISSUER).unwrap()
}

/// A write whose reply was lost is sent again with its descriptor: the
/// retry runs nothing and reports the commit of the first attempt.
#[test]
fn a_retry_after_a_lost_reply_applies_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let issuer = enrolled(&db);
    let op = issuer.next().unwrap();
    run(&db, &op, "CREATE (:Purchase {k: 1})").unwrap();
    let (outcome, differs) = settled(run(&db, &op, "CREATE (:Purchase {k: 1})"));
    assert!(matches!(outcome, OperationOutcome::Committed { .. }));
    assert!(!differs);
    assert_eq!(count(&mut db, "Purchase"), 1);
    assert!(matches!(
        db.resolve_operation(&op, Resolve::Lookup).unwrap(),
        OperationOutcome::Committed { .. }
    ));
}

/// A host acknowledges a completed write while an earlier number is still
/// unresolved: that outcome is released at once, the write stays settled,
/// and the earlier number is untouched and still admissible.
#[test]
fn an_acknowledged_result_is_released_above_an_unresolved_number() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let issuer = enrolled(&db);
    let pending = issuer.next().unwrap();
    let done = issuer.next().unwrap();
    run(&db, &done, "CREATE (:Done)").unwrap();

    db.acknowledge_operation(&done).unwrap();
    assert_eq!(
        db.resolve_operation(&done, Resolve::Lookup).unwrap(),
        OperationOutcome::Retired
    );
    let (outcome, differs) = settled(run(&db, &done, "CREATE (:Done)"));
    assert_eq!(outcome, OperationOutcome::Retired);
    assert!(!differs);
    assert_eq!(count(&mut db, "Done"), 1);

    // The acknowledgement of a number nothing is recorded of declares
    // nothing: the earlier write still runs, once.
    db.acknowledge_operation(&pending).unwrap();
    assert_eq!(
        db.resolve_operation(&pending, Resolve::Lookup).unwrap(),
        OperationOutcome::Unknown
    );
    run(&db, &pending, "CREATE (:Pending)").unwrap();
    assert_eq!(count(&mut db, "Pending"), 1);
}

/// Each database is its own domain once it issues descriptors: a
/// descriptor issued by one is refused by another that enrolled the same
/// issuer, rather than admitted there as a new operation.
#[test]
fn a_descriptor_from_another_database_is_refused() {
    let (dir_a, dir_b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let a = Database::open(dir_a.path()).unwrap();
    let mut b = Database::open(dir_b.path()).unwrap();
    let theirs = enrolled(&a).next().unwrap();
    let ours = enrolled(&b).next().unwrap();
    assert_eq!(theirs.op, ours.op, "the same issuer numbers alike in both");
    assert_ne!(theirs.deployment, ours.deployment);

    assert!(matches!(
        run(&b, &theirs, "CREATE (:Foreign)"),
        Err(DatabaseError::ForeignOperation { .. })
    ));
    assert!(matches!(
        b.resolve_operation(&theirs, Resolve::Lookup),
        Err(DatabaseError::ForeignOperation { .. })
    ));
    assert_eq!(count(&mut b, "Foreign"), 0);
    run(&b, &ours, "CREATE (:Own)").unwrap();

    // Its bytes with the domain stripped are no descriptor at all.
    let bytes = theirs.to_bytes();
    assert_eq!(
        OperationDescriptor::from_bytes(&bytes[..bytes.len() - 16]),
        None
    );
    // A database that issued no descriptor refuses every one.
    let fresh_dir = tempfile::tempdir().unwrap();
    let fresh = Database::open(fresh_dir.path()).unwrap();
    assert!(matches!(
        fresh.resolve_operation(&theirs, Resolve::Lookup),
        Err(DatabaseError::ForeignOperation { .. })
    ));
    assert!(fresh.resume_issuer(ISSUER, 1, 1).is_err());
}

/// A logical backup carries the data a write left, never its operation
/// record: a store restored from one is a new domain, so a descriptor
/// issued before the backup is refused there rather than run again or
/// reported with an outcome the restored store never recorded.
#[test]
fn a_logical_restore_refuses_descriptors_issued_before_it() {
    use coordinode_embed::backup::restore::RestoreOptions;
    use coordinode_embed::backup::{BackupFormat, export};

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let issuer = enrolled(&db);
    let op = issuer.next().unwrap();
    run(&db, &op, "CREATE (:Purchase {k: 1})").unwrap();
    let interner = db.interner().unwrap();

    for format in [
        BackupFormat::Binary,
        BackupFormat::Json,
        BackupFormat::Cypher,
    ] {
        let mut dump = Vec::new();
        let snapshot = db.engine().snapshot();
        match format {
            BackupFormat::Binary => {
                export::export_binary(db.engine(), &interner, 1, &snapshot, &mut dump)
            }
            BackupFormat::Json => {
                export::export_json(db.engine(), &interner, 1, &snapshot, &mut dump)
            }
            _ => export::export_cypher(db.engine(), &interner, 1, &snapshot, &mut dump),
        }
        .unwrap();
        let restored_dir = tempfile::tempdir().unwrap();
        let mut restored = Database::open(restored_dir.path()).unwrap();
        restored
            .restore(format, &dump, &RestoreOptions::default())
            .unwrap();

        assert_eq!(count(&mut restored, "Purchase"), 1, "{format:?}");
        assert!(
            matches!(
                restored.resolve_operation(&op, Resolve::Lookup),
                Err(DatabaseError::ForeignOperation { .. })
            ),
            "{format:?}"
        );
        assert!(
            matches!(
                run(&restored, &op, "CREATE (:Purchase {k: 1})"),
                Err(DatabaseError::ForeignOperation { .. })
            ),
            "{format:?}"
        );
        assert_eq!(count(&mut restored, "Purchase"), 1, "{format:?}");
        // The restored store issues descriptors of its own.
        let fresh = enrolled(&restored).next().unwrap();
        assert_ne!(fresh.deployment, op.deployment, "{format:?}");
        run(&restored, &fresh, "CREATE (:Purchase {k: 2})").unwrap();
    }
}

/// One number used for two different writes: the second is refused and
/// told the first meant something else; nothing of it applies.
#[test]
fn the_same_number_with_another_intent_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let op = enrolled(&db).next().unwrap();
    run(&db, &op, "CREATE (:A)").unwrap();
    let (outcome, differs) = settled(run(&db, &op, "CREATE (:B)"));
    assert!(matches!(outcome, OperationOutcome::Committed { .. }));
    assert!(differs);
    assert_eq!(count(&mut db, "B"), 0);
}

/// A retry of a write whose statement is not deterministic does not run it
/// again: the value the first run chose stays the only one.
#[test]
fn a_nondeterministic_statement_is_not_run_again() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let op = enrolled(&db).next().unwrap();
    let query = "CREATE (:Tick {id: randomUUID(), r: rand()})";
    run(&db, &op, query).unwrap();
    let first = db.execute_cypher("MATCH (t:Tick) RETURN t.r AS r").unwrap();
    settled(run(&db, &op, query));
    let after = db.execute_cypher("MATCH (t:Tick) RETURN t.r AS r").unwrap();
    assert_eq!(first, after);
}

/// An increment sent many times counts once.
#[test]
fn a_retried_increment_counts_once() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.execute_cypher("CREATE (:Counter {k: 1, n: 0})").unwrap();
    let op = enrolled(&db).next().unwrap();
    let query = "MATCH (c:Counter {k: 1}) SET c.n = c.n + 1";
    run(&db, &op, query).unwrap();
    for _ in 0..3 {
        settled(run(&db, &op, query));
    }
    let rows = db
        .execute_cypher("MATCH (c:Counter {k: 1}) RETURN c.n AS n")
        .unwrap();
    assert_eq!(rows[0].get("n"), Some(&Value::Int(1)));
}

/// Concurrent attempts of one operation take effect once between them.
#[test]
fn concurrent_attempts_take_effect_once() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Database::open(dir.path()).unwrap());
    let op = enrolled(&db).next().unwrap();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || run(&db, &op, "CREATE (:Once)"))
        })
        .collect();
    let mut applied = 0;
    for handle in handles {
        match handle.join().unwrap() {
            Ok(()) => applied += 1,
            Err(DatabaseError::OperationSettled {
                outcome: OperationOutcome::Committed { .. },
                intent_differs: false,
                ..
            }) => {}
            // A concurrent attempt can also lose first-committer-wins on the
            // label's counter; nothing of it applied either.
            Err(DatabaseError::Execution(_)) => {}
            Err(other) => panic!("unexpected: {other:?}"),
        }
    }
    let mut db = Arc::try_unwrap(db).ok().unwrap();
    assert_eq!(count(&mut db, "Once"), 1);
    assert!(applied <= 1);
}

/// An interactive transaction committed under an operation takes effect
/// once: a second transaction with the same statements under the same
/// descriptor applies nothing, and one with other statements is told it
/// means something else.
#[test]
fn a_transaction_commit_carries_its_operation() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let op = enrolled(&db).next().unwrap();
    let transaction = |statements: &[&str]| {
        let txn = db.begin_transaction();
        for statement in statements {
            db.execute_in_transaction(txn, statement, None).unwrap();
        }
        db.commit_transaction_as(txn, &op).map(drop)
    };
    transaction(&["CREATE (:Item {k: 1})", "CREATE (:Item {k: 2})"]).unwrap();
    let (_, differs) = settled(transaction(&[
        "CREATE (:Item {k: 1})",
        "CREATE (:Item {k: 2})",
    ]));
    assert!(!differs);
    let (_, differs) = settled(transaction(&["CREATE (:Item {k: 3})"]));
    assert!(differs);
    assert_eq!(count(&mut db, "Item"), 2);
}

/// A COMMIT repeated after its reply was lost finds the transaction already
/// gone; it answers with the operation's outcome, not with an unknown
/// transaction, and needs neither the old handle's buffer nor a replay of
/// its statements.
#[test]
fn a_repeated_commit_resolves_its_operation() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let op = enrolled(&db).next().unwrap();
    let txn = db.begin_transaction();
    db.execute_in_transaction(txn, "CREATE (:Once {k: 1})", None)
        .unwrap();
    db.commit_transaction_as(txn, &op).unwrap();

    let (outcome, differs) = settled(db.commit_transaction_as(txn, &op).map(drop));
    assert!(matches!(outcome, OperationOutcome::Committed { .. }));
    assert!(!differs);
    assert_eq!(count(&mut db, "Once"), 1);

    // An unbound commit of a gone handle still says the handle is unknown.
    assert!(matches!(
        db.commit_transaction(txn),
        Err(DatabaseError::UnknownTransaction(_))
    ));
}

/// Excluding an operation nothing is recorded of proves it never takes
/// effect: a late attempt of it applies nothing.
#[test]
fn an_excluded_operation_never_takes_effect() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let op = enrolled(&db).next().unwrap();
    assert_eq!(
        db.resolve_operation(&op, Resolve::Lookup).unwrap(),
        OperationOutcome::Unknown
    );
    assert_eq!(
        db.resolve_operation(&op, Resolve::Exclude).unwrap(),
        OperationOutcome::Aborted
    );
    let (outcome, _) = settled(run(&db, &op, "CREATE (:Late)"));
    assert_eq!(outcome, OperationOutcome::Aborted);
    assert_eq!(count(&mut db, "Late"), 0);
}

/// The admission window bounds what one incarnation can make the database
/// keep: a write numbered past it is refused and runs nothing; releasing
/// single outcomes does not move it, retiring the prefix below does; and an
/// incarnation keeps the window it was enrolled with when the setting
/// changes.
#[test]
fn a_full_admission_window_refuses_until_the_prefix_is_retired() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.set_operation_admission_window(core::num::NonZeroU64::new(4).unwrap());
    let issuer = enrolled(&db);
    let ops: Vec<_> = (0..6).map(|_| issuer.next().unwrap()).collect();
    for (i, op) in ops.iter().take(4).enumerate() {
        run(&db, op, &format!("CREATE (:W {{k: {i}}})")).unwrap();
    }
    let (outcome, differs) = settled(run(&db, &ops[4], "CREATE (:W {k: 4})"));
    assert_eq!(outcome, OperationOutcome::Unknown, "past the window");
    assert!(!differs);
    assert_eq!(count(&mut db, "W"), 4, "the refused write ran nothing");

    // Releasing outcomes one by one keeps the prefix where it is.
    db.acknowledge_operation(&ops[3]).unwrap();
    settled(run(&db, &ops[4], "CREATE (:W {k: 4})"));

    // A window enrolled under a larger setting is not this incarnation's.
    db.set_operation_admission_window(core::num::NonZeroU64::new(1000).unwrap());
    settled(run(&db, &ops[4], "CREATE (:W {k: 4})"));

    // Retiring below the third number moves the window over the fifth and
    // the sixth.
    db.retire_operations(&ops[2]).unwrap();
    run(&db, &ops[4], "CREATE (:W {k: 4})").unwrap();
    run(&db, &ops[5], "CREATE (:W {k: 5})").unwrap();
    assert_eq!(count(&mut db, "W"), 6);
}

/// Retired numbers report retired and admit nothing, numbers never used
/// included.
#[test]
fn retired_operations_admit_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let issuer = enrolled(&db);
    let used = issuer.next().unwrap();
    let unused = issuer.next().unwrap();
    run(&db, &used, "CREATE (:R)").unwrap();
    // The bound is the next number: everything below it is retired.
    let bound = issuer.next().unwrap();
    assert_eq!(bound.op.sequence, 3);
    db.retire_operations(&bound).unwrap();
    for op in [&used, &unused] {
        assert_eq!(
            db.resolve_operation(op, Resolve::Lookup).unwrap(),
            OperationOutcome::Retired
        );
    }
    let (outcome, _) = settled(run(&db, &unused, "CREATE (:R)"));
    assert_eq!(outcome, OperationOutcome::Retired);
    assert_eq!(count(&mut db, "R"), 1);
}

/// After a restart the outcome is still there, the issuer resumes its
/// numbering, and a new enrollment closes the old incarnation: an attempt of
/// the old one that never took effect is aborted for good.
#[test]
fn outcomes_and_fences_survive_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let (done, pending, next) = {
        let db = Database::open(dir.path()).unwrap();
        let issuer = enrolled(&db);
        let done = issuer.next().unwrap();
        run(&db, &done, "CREATE (:Kept)").unwrap();
        let pending = issuer.next().unwrap();
        (done, pending, issuer.next_sequence())
    };
    let mut db = Database::open(dir.path()).unwrap();
    assert!(matches!(
        db.resolve_operation(&done, Resolve::Lookup).unwrap(),
        OperationOutcome::Committed { .. }
    ));
    let resumed = db.resume_issuer(ISSUER, done.op.incarnation, next).unwrap();
    assert_eq!(resumed.next().unwrap().op.sequence, next);

    let fresh = enrolled(&db);
    assert!(fresh.incarnation() > done.op.incarnation);
    let (outcome, _) = settled(run(&db, &pending, "CREATE (:Kept)"));
    assert_eq!(outcome, OperationOutcome::Aborted);
    assert_eq!(count(&mut db, "Kept"), 1);
    assert!(db.resume_issuer(ISSUER, done.op.incarnation, next).is_err());
}

/// Two copies of one allocator hand out the same number: the first write
/// of it takes effect, and the other copy's different write is refused.
#[test]
fn a_cloned_allocator_cannot_apply_a_number_twice() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let issuer = enrolled(&db);
    let copy = db
        .resume_issuer(ISSUER, issuer.incarnation(), issuer.next_sequence())
        .unwrap();
    let mine = issuer.next().unwrap();
    let theirs = copy.next().unwrap();
    assert_eq!(mine, theirs);
    run(&db, &mine, "CREATE (:Mine)").unwrap();
    let (_, differs) = settled(run(&db, &theirs, "CREATE (:Theirs)"));
    assert!(differs);
    assert_eq!(count(&mut db, "Theirs"), 0);
}

/// A provisional write (memory concern) carrying a descriptor is admitted
/// durably before it is acknowledged, and its drain completes the same
/// operation: a retry runs nothing, before or after the drain, and after a
/// restart the write stands committed, once.
#[test]
fn a_provisional_write_is_admitted_then_completed_by_its_drain() {
    use coordinode_core::txn::write_concern::WriteConcern;

    let dir = tempfile::tempdir().unwrap();
    let op = {
        let db = Database::open(dir.path()).unwrap();
        let op = enrolled(&db).next().unwrap();
        let provisional = |query: &str| {
            db.execute_cypher_shared_with(
                query,
                None,
                None,
                &StatementOptions {
                    operation: Some(op),
                    write_concern: Some(WriteConcern::memory()),
                    ..StatementOptions::default()
                },
            )
            .map(drop)
        };
        provisional("CREATE (:Early {k: 1})").unwrap();
        // Admitted at once; committed once the drain has run.
        assert!(matches!(
            db.resolve_operation(&op, Resolve::Lookup).unwrap(),
            OperationOutcome::Admitted { .. } | OperationOutcome::Committed { .. }
        ));
        let (outcome, differs) = settled(provisional("CREATE (:Early {k: 1})"));
        assert!(matches!(
            outcome,
            OperationOutcome::Admitted { .. } | OperationOutcome::Committed { .. }
        ));
        assert!(!differs);
        // Closing the database drains what is buffered.
        op
    };
    let mut db = Database::open(dir.path()).unwrap();
    let outcome = db.resolve_operation(&op, Resolve::Lookup).unwrap();
    assert!(
        matches!(outcome, OperationOutcome::Committed { .. }),
        "{outcome:?}"
    );
    assert_eq!(count(&mut db, "Early"), 1);
}

/// A bound write with parameters: the same values under the same
/// descriptor are one operation.
#[test]
fn parameters_belong_to_the_intent() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let op = enrolled(&db).next().unwrap();
    let with = |k: i64| {
        db.execute_cypher_shared_with(
            "CREATE (:P {k: $k})",
            Some(HashMap::from([("k".to_owned(), Value::Int(k))])),
            None,
            &StatementOptions {
                operation: Some(op),
                ..StatementOptions::default()
            },
        )
        .map(drop)
    };
    with(1).unwrap();
    assert!(!settled(with(1)).1);
    assert!(settled(with(2)).1);
    assert_eq!(count(&mut db, "P"), 1);
}
