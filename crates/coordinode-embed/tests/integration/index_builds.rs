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
