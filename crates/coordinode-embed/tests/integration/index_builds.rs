//! Index builds through the database: a build runs on the engine whatever
//! its statement does, inspection shows where it stands, and cancellation,
//! a drop, a timeout of the wait for older transactions and a restart each
//! leave the catalog in the state the build's outcome names.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use coordinode_modality::IndexState;
use coordinode_query::index::{BuildPhase, BuildState, BuildStatus, IndexBuildConfig};

use super::helpers::{admit_index, index_named};

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open db");
    db.execute_cypher("CREATE (:User {email: 'a@x'}), (:User {email: 'b@x'})")
        .expect("seed users");
    (db, dir)
}

/// The build of the index named `name`, as inspection lists it.
fn status_of(db: &Database, name: &str) -> Option<BuildStatus> {
    let def = index_named(db.engine(), name)?;
    db.index_build_status()
        .expect("inspect")
        .into_iter()
        .find(|s| s.generation == def.generation)
}

/// Wait until the build of `name` waits for older transactions, which the
/// caller holds open.
fn await_held(db: &Database, name: &str) -> BuildStatus {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(status) = status_of(db, name) {
            if status.phase == Some(BuildPhase::AwaitingOlderTransactions) {
                return status;
            }
        }
        assert!(Instant::now() < deadline, "the build never waited");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn create_index(db: &Database, statement: &str) -> Result<(), String> {
    db.execute_cypher_shared(statement, None, None, None, None)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn holders(db: &mut Database, email: &str) -> usize {
    db.execute_cypher(&format!(
        "MATCH (u:User) WHERE u.email = '{email}' RETURN u"
    ))
    .expect("match")
    .len()
}

/// A build held behind an older transaction is listed as running and
/// waiting; once the transaction ends the statement returns, the index is
/// ready and serves lookups, and inspection keeps the published record with
/// no phase.
#[test]
fn a_running_build_is_inspectable_and_publishes_the_index() {
    let (mut db, _dir) = open_db();
    let older = db.begin_transaction();

    std::thread::scope(|s| {
        let created = s.spawn(|| create_index(&db, "CREATE INDEX user_email ON :User(email)"));
        let held = await_held(&db, "user_email");
        assert!(
            matches!(
                held.record.as_ref().map(|r| &r.state),
                Some(BuildState::Running { .. })
            ),
            "{held:?}"
        );
        assert!(!created.is_finished(), "the statement waits for its build");
        db.rollback_transaction(older)
            .expect("end the older transaction");
        created.join().expect("join").expect("create index");
    });

    let status = status_of(&db, "user_email").expect("listed");
    assert_eq!(status.record.map(|r| r.state), Some(BuildState::Published));
    assert_eq!(status.phase, None);
    assert_eq!(
        index_named(db.engine(), "user_email")
            .expect("defined")
            .state,
        IndexState::Ready
    );
    assert_eq!(holders(&mut db, "a@x"), 1);
}

/// Cancelling a running build of a new index fails its statement, withdraws
/// the index, and leaves the name free for a new one.
#[test]
fn a_cancelled_build_withdraws_its_new_index() {
    let (mut db, _dir) = open_db();
    let older = db.begin_transaction();

    std::thread::scope(|s| {
        let created = s.spawn(|| create_index(&db, "CREATE INDEX user_email ON :User(email)"));
        await_held(&db, "user_email");
        assert!(
            db.cancel_index_build("user_email"),
            "a running build cancels"
        );
        db.rollback_transaction(older)
            .expect("end the older transaction");
        let error = created.join().expect("join").expect_err("cancelled");
        assert!(error.contains("cancelled"), "{error}");
    });

    assert!(index_named(db.engine(), "user_email").is_none());
    assert!(
        !db.cancel_index_build("user_email"),
        "nothing is left to cancel"
    );
    db.execute_cypher("CREATE INDEX user_email ON :User(email)")
        .expect("the name is free");
    assert_eq!(holders(&mut db, "b@x"), 1);
}

/// DROP INDEX while the index's build runs ends the build: its statement
/// fails cancelled, no record or definition remains, and no entry lands.
#[test]
fn dropping_an_index_during_its_build_ends_the_build() {
    let (mut db, _dir) = open_db();
    let older = db.begin_transaction();
    let mut generation = None;

    std::thread::scope(|s| {
        let created = s.spawn(|| create_index(&db, "CREATE INDEX user_email ON :User(email)"));
        generation = Some(await_held(&db, "user_email").generation);
        db.execute_cypher_shared("DROP INDEX user_email", None, None, None, None)
            .expect("drop during the build");
        db.rollback_transaction(older)
            .expect("end the older transaction");
        let error = created.join().expect("join").expect_err("ended");
        assert!(error.contains("cancelled"), "{error}");
    });

    assert!(index_named(db.engine(), "user_email").is_none());
    let generation = generation.expect("seen");
    assert!(
        db.index_build_status()
            .expect("inspect")
            .iter()
            .all(|s| s.generation != generation),
        "the build's record went with its index"
    );
    assert_eq!(holders(&mut db, "a@x"), 1, "the data is untouched");
}

/// A build that outwaits the configured wait for older transactions fails
/// its statement, names the open transactions, records the failure, and
/// withdraws the new index.
#[test]
fn a_build_outwaited_by_an_older_transaction_fails() {
    let (db, _dir) = open_db();
    db.set_index_build_config(IndexBuildConfig {
        older_transactions_wait: Duration::from_millis(100),
        ..db.index_build_config()
    });
    let older = db.begin_transaction();

    let error =
        create_index(&db, "CREATE INDEX user_email ON :User(email)").expect_err("outwaited");
    db.rollback_transaction(older)
        .expect("end the older transaction");

    assert!(error.contains("still open"), "{error}");
    assert!(index_named(db.engine(), "user_email").is_none());
    assert!(
        db.index_build_status()
            .expect("inspect")
            .iter()
            .any(|s| matches!(
                s.record.as_ref().map(|r| &r.state),
                Some(BuildState::Failed { .. })
            )),
        "the failure is recorded"
    );
}

/// Two statements creating an index under one name at the same time, on
/// different properties: exactly one wins, the loser is refused without an
/// index of its own, and the winner's index alone is defined and built.
/// Repeated so that the two publications actually overlap.
#[test]
fn concurrent_creates_of_one_index_name_have_one_winner() {
    let (db, _dir) = open_db();
    for round in 0..16 {
        let name = format!("idx_{round}");
        let statements = [
            format!("CREATE INDEX {name} ON :User(email)"),
            format!("CREATE INDEX {name} ON :User(name)"),
        ];
        let start = std::sync::Barrier::new(2);
        let outcomes: Vec<Result<(), String>> = std::thread::scope(|s| {
            let handles: Vec<_> = statements
                .iter()
                .map(|statement| {
                    let (db, start) = (&db, &start);
                    s.spawn(move || {
                        start.wait();
                        create_index(db, statement)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("the statement thread"))
                .collect()
        });

        let winners: Vec<&str> = outcomes
            .iter()
            .zip(["email", "name"])
            .filter(|(outcome, _)| outcome.is_ok())
            .map(|(_, property)| property)
            .collect();
        assert_eq!(winners.len(), 1, "round {round}: {outcomes:?}");
        let def = index_named(db.engine(), &name).expect("the winner's index");
        assert_eq!(def.properties, [winners[0]], "round {round}");
        assert_eq!(def.state, IndexState::Ready, "round {round}");
        assert!(
            status_of(&db, &name)
                .and_then(|s| s.record)
                .is_some_and(|r| r.state == BuildState::Published),
            "round {round}"
        );
    }
}

/// Run `body` while the build of the unique index on `:User(email)` waits
/// for an older transaction: the index is registered with the writers, and
/// none of the stored users has an entry yet. The build then finishes.
fn while_unique_build_waits(db: &Database, body: impl FnOnce()) -> Result<(), String> {
    let older = db.begin_transaction();
    std::thread::scope(|s| {
        let created =
            s.spawn(|| create_index(db, "CREATE UNIQUE INDEX user_email ON :User(email)"));
        await_held(db, "user_email");
        body();
        db.rollback_transaction(older)
            .expect("end the older transaction");
        created.join().expect("join")
    })
}

fn write(db: &Database, statement: &str) -> Result<(), String> {
    db.execute_cypher_shared(statement, None, None, None, None)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// While a unique index is being built, a value a stored node holds is
/// refused as a duplicate naming its holder, though the build has not given
/// that node an entry, whether the write creates a node or moves one onto
/// the value; a free value is taken, and the index the build publishes holds
/// each value once.
#[test]
fn a_unique_value_a_stored_node_holds_is_refused_during_the_build() {
    let (mut db, _dir) = open_db();

    while_unique_build_waits(&db, || {
        let created = write(&db, "CREATE (:User {email: 'a@x'})").expect_err("held");
        assert!(created.contains("unique constraint violated"), "{created}");
        assert!(created.contains("a@x"), "{created}");
        let moved =
            write(&db, "MATCH (u:User {email: 'b@x'}) SET u.email = 'a@x'").expect_err("held");
        assert!(moved.contains("unique constraint violated"), "{moved}");
        write(&db, "CREATE (:User {email: 'c@x'})").expect("a free value");
    })
    .expect("the build publishes");

    assert_eq!(holders(&mut db, "a@x"), 1);
    assert_eq!(holders(&mut db, "b@x"), 1);
    assert_eq!(holders(&mut db, "c@x"), 1);
    assert_eq!(
        index_named(db.engine(), "user_email")
            .expect("defined")
            .state,
        IndexState::Ready
    );
}

/// One statement taking many values while the index is being built is
/// decided as a whole: a single held value among them refuses it and
/// nothing of it is written; the free values alone are written.
#[test]
fn a_statement_taking_many_values_during_the_build_is_decided_whole() {
    let (mut db, _dir) = open_db();
    let many: Vec<String> = (0..200).map(|i| format!("'n{i}@x'")).collect();

    while_unique_build_waits(&db, || {
        let refused = write(
            &db,
            &format!(
                "UNWIND [{}, 'b@x'] AS e CREATE (:User {{email: e}})",
                many.join(", ")
            ),
        )
        .expect_err("b@x is held");
        assert!(refused.contains("unique constraint violated"), "{refused}");
        write(
            &db,
            &format!(
                "UNWIND [{}] AS e CREATE (:User {{email: e}})",
                many.join(", ")
            ),
        )
        .expect("free values");
    })
    .expect("the build publishes");

    assert_eq!(holders(&mut db, "b@x"), 1);
    assert_eq!(holders(&mut db, "n0@x"), 1);
    assert_eq!(holders(&mut db, "n199@x"), 1);
}

/// The same refusals reach an interactive transaction at its commit: a
/// value a stored node holds is a duplicate naming the index and the value,
/// and past the read limit the commit is unresolved, not a duplicate.
#[test]
fn an_interactive_commit_is_refused_like_a_statement_during_the_build() {
    let (db, _dir) = open_db();

    while_unique_build_waits(&db, || {
        let txn = db.begin_transaction();
        db.execute_in_transaction(txn, "CREATE (:User {email: 'a@x'})", None)
            .expect("staged");
        let held = db.commit_transaction(txn).expect_err("held").to_string();
        assert!(held.contains("unique constraint violated"), "{held}");
        assert!(held.contains("user_email"), "{held}");
        assert!(held.contains("a@x"), "{held}");

        db.set_index_build_config(IndexBuildConfig {
            unique_admission_read_limit: 1,
            ..db.index_build_config()
        });
        let txn = db.begin_transaction();
        db.execute_in_transaction(txn, "CREATE (:User {email: 'z@x'})", None)
            .expect("staged");
        let unresolved = db
            .commit_transaction(txn)
            .expect_err("unresolved")
            .to_string();
        assert!(unresolved.contains("still being built"), "{unresolved}");
        assert!(
            !unresolved.contains("unique constraint violated"),
            "{unresolved}"
        );
    })
    .expect("the build publishes");
}

/// While a partial unique index is being built, only stored nodes its
/// filter admits hold its values: a value an excluded node has is free, one
/// an admitted node has is refused.
#[test]
fn a_partial_unique_index_holds_only_admitted_nodes_during_the_build() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE (:User {email: 'on@x', active: true}), (:User {email: 'off@x', active: false})",
    )
    .expect("seed");
    let older = db.begin_transaction();

    std::thread::scope(|s| {
        let created = s.spawn(|| {
            create_index(
                &db,
                "CREATE UNIQUE INDEX active_email ON :User(email) WHERE n.active = true",
            )
        });
        await_held(&db, "active_email");
        let held = write(&db, "CREATE (:User {email: 'on@x', active: true})").expect_err("held");
        assert!(held.contains("unique constraint violated"), "{held}");
        write(&db, "CREATE (:User {email: 'off@x', active: true})")
            .expect("the stored holder is outside the filter");
        write(&db, "CREATE (:User {email: 'on@x', active: false})")
            .expect("the new node is outside the filter");
        db.rollback_transaction(older)
            .expect("end the older transaction");
        created.join().expect("join").expect("the build publishes");
    });

    assert_eq!(holders(&mut db, "on@x"), 2);
    assert_eq!(holders(&mut db, "off@x"), 2);
}

/// A write whose proof would read more stored nodes than the configured
/// limit is refused as unresolved, not as a duplicate, and the same write
/// succeeds once the build is done.
#[test]
fn a_unique_value_past_the_read_limit_is_unresolved_until_the_build_is_done() {
    let (mut db, _dir) = open_db();
    db.set_index_build_config(IndexBuildConfig {
        unique_admission_read_limit: 1,
        ..db.index_build_config()
    });

    while_unique_build_waits(&db, || {
        let refused = write(&db, "CREATE (:User {email: 'z@x'})").expect_err("unresolved");
        assert!(refused.contains("still being built"), "{refused}");
        assert!(!refused.contains("unique constraint violated"), "{refused}");
    })
    .expect("the build publishes");

    write(&db, "CREATE (:User {email: 'z@x'})").expect("the build is done");
    assert_eq!(holders(&mut db, "z@x"), 1);
}

/// A transaction that wrote before a partial unique index existed keeps no
/// entry of it and states no claim on its values, so its commit is refused
/// once the index is published, rather than landing a second holder of a
/// value under the build.
#[test]
fn a_writer_older_than_a_partial_unique_index_is_refused() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:User {email: 'p@x', active: true})")
        .expect("the holder");
    let older = db.begin_transaction();
    db.execute_in_transaction(older, "CREATE (:User {email: 'p@x', active: true})", None)
        .expect("written before the index");

    std::thread::scope(|s| {
        let created = s.spawn(|| {
            create_index(
                &db,
                "CREATE UNIQUE INDEX active_email ON :User(email) WHERE n.active = true",
            )
        });
        await_held(&db, "active_email");
        assert!(
            db.commit_transaction(older).is_err(),
            "a writer older than the index committed"
        );
        created.join().expect("join").expect("the build publishes");
    });

    assert_eq!(holders(&mut db, "p@x"), 1);
}

/// A build its statement never saw finish, as after a crash, is taken up
/// when the database opens again: the index ends ready and serves lookups.
#[test]
fn an_interrupted_build_finishes_after_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (:User {email: 'a@x'})")
            .expect("seed");
        admit_index(
            db.engine(),
            coordinode_modality::IndexDescriptor::btree("user_email", "User", "email"),
        );
    }

    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(
        index_named(db.engine(), "user_email")
            .expect("defined")
            .state,
        IndexState::Ready
    );
    assert_eq!(
        status_of(&db, "user_email")
            .and_then(|s| s.record)
            .map(|r| r.state),
        Some(BuildState::Published)
    );
    assert_eq!(holders(&mut db, "a@x"), 1);
    let rows = db
        .execute_cypher("MATCH (u:User) WHERE u.email = 'a@x' RETURN u.email AS e")
        .expect("match");
    assert_eq!(rows[0].get("e"), Some(&Value::String("a@x".into())));
}
