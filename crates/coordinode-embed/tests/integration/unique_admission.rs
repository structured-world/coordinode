//! Unique values under concurrency, in both effect profiles: writers racing
//! for one value, a holder replaced while another writer takes its value,
//! compound and list values, and writers racing while the index is still
//! being built. In every schedule the stored data ends with one holder per
//! value, and a refused writer is told the value is held.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use coordinode_embed::Database;
use coordinode_query::index::BuildPhase;

use super::helpers::index_named;

/// The two ways a unique index's entries reach the log.
const PROFILES: [&str; 2] = ["resolved", "derived"];

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

fn run(db: &Database, statement: &str) -> Result<(), String> {
    db.execute_cypher_shared(statement, None, None, None, None)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn holders(db: &mut Database, label: &str, property: &str, value: &str) -> usize {
    db.execute_cypher(&format!(
        "MATCH (n:{label}) WHERE n.{property} = '{value}' RETURN n"
    ))
    .expect("match")
    .len()
}

/// Run every statement on a thread of its own, released together.
fn race(db: &Database, statements: &[String]) -> Vec<Result<(), String>> {
    let start = std::sync::Barrier::new(statements.len());
    std::thread::scope(|s| {
        let handles: Vec<_> = statements
            .iter()
            .map(|statement| {
                let start = &start;
                s.spawn(move || {
                    start.wait();
                    run(db, statement)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("the writer thread"))
            .collect()
    })
}

/// Each refusal says the value is held, or asks for a retry of a race it
/// lost; none is any other failure.
fn assert_refusals_explained(outcomes: &[Result<(), String>]) {
    for err in outcomes.iter().filter_map(|o| o.as_ref().err()) {
        assert!(
            err.contains("unique constraint violated") || err.contains("conflict"),
            "{err}"
        );
    }
}

/// Writers inserting one value at once, in each profile: exactly one
/// succeeds, and every other is told the value exists.
#[test]
fn concurrent_inserts_of_one_value_leave_one_holder_in_each_profile() {
    for profile in PROFILES {
        let (mut db, _dir) = open_db();
        db.execute_cypher(&format!(
            "CREATE UNIQUE INDEX u_email ON :U(email) OPTIONS {{maintenance: '{profile}'}}"
        ))
        .expect("index");
        let writers: Vec<String> = (0..8)
            .map(|w| format!("CREATE (:U {{email: 'same', w: {w}}})"))
            .collect();

        let outcomes = race(&db, &writers);

        assert_eq!(
            outcomes.iter().filter(|o| o.is_ok()).count(),
            1,
            "{profile}: {outcomes:?}"
        );
        for err in outcomes.iter().filter_map(|o| o.as_ref().err()) {
            assert!(
                err.contains("unique constraint violated"),
                "{profile}: {err}"
            );
        }
        assert_eq!(holders(&mut db, "U", "email", "same"), 1, "{profile}");
    }
}

/// Writers racing for one value while the index moves between profiles:
/// the handover keeps one protection, so one holder remains.
#[test]
fn concurrent_inserts_across_a_profile_handover_leave_one_holder() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    let mut statements: Vec<String> = (0..6)
        .map(|w| format!("CREATE (:U {{email: 'same', w: {w}}})"))
        .collect();
    statements.push("ALTER INDEX u_email SET MAINTENANCE DERIVED".to_string());

    let outcomes = race(&db, &statements);

    let (writes, handover) = outcomes.split_at(6);
    handover[0].as_ref().expect("the handover");
    assert!(
        writes.iter().filter(|o| o.is_ok()).count() <= 1,
        "{outcomes:?}"
    );
    for err in writes.iter().filter_map(|o| o.as_ref().err()) {
        // A write staged under the epoch the handover replaced is refused
        // with the binding it staged, to be retried under the new one.
        assert!(
            err.contains("unique constraint violated")
                || err.contains("conflict")
                || err.contains("version mismatch"),
            "{err}"
        );
    }
    // After the handover the value is held exactly when a racer took it.
    let won = writes.iter().any(Result::is_ok);
    let retried = run(&db, "CREATE (:U {email: 'same'})");
    assert_eq!(retried.is_ok(), !won, "{retried:?}");
    assert_eq!(holders(&mut db, "U", "email", "same"), 1);
}

/// A handover against a writer whose commit is already admitted under the
/// old epoch: the handover waits for that commit to land and then takes
/// effect, instead of being refused because the writer came first.
#[test]
fn a_handover_waits_for_a_writer_admitted_under_the_old_epoch() {
    use coordinode_storage::engine::partition::Partition;

    let (db, _dir) = open_db();
    run(&db, "CREATE UNIQUE INDEX u_email ON :U(email)").expect("index");
    let before = index_named(db.engine(), "u_email").expect("index");
    let key = coordinode_modality::IndexDefinition::schema_key_of(before.id);
    // A writer in flight: admitted, conditioned on the definition as every
    // index writer's commit is, not yet applied. Its timestamp sits past the
    // clock, so no read waits for it.
    let (_, writer) = db
        .engine()
        .pending_commits()
        .admit_allocated(|| u64::MAX, Vec::new(), vec![(Partition::Schema, key)])
        .expect("the writer is admitted");

    std::thread::scope(|s| {
        let handover = s.spawn(|| run(&db, "ALTER INDEX u_email SET MAINTENANCE DERIVED"));
        std::thread::sleep(Duration::from_millis(300));
        assert!(!handover.is_finished(), "the handover waits for the writer");
        drop(writer);
        handover
            .join()
            .expect("the handover thread")
            .expect("the handover");
    });

    let after = index_named(db.engine(), "u_email").expect("index");
    assert_eq!(after.maintenance.epoch, before.maintenance.epoch + 1);
    assert_eq!(
        after.maintenance.profile,
        coordinode_modality::IndexProfile::Derived
    );
}

/// A writer replacing the holder of a value (delete, then reinsert under a
/// new node) races a writer inserting the value: whichever lands, one node
/// holds it.
#[test]
fn an_insert_racing_a_delete_and_reinsert_leaves_one_holder() {
    for profile in PROFILES {
        for round in 0..4 {
            let (mut db, _dir) = open_db();
            db.execute_cypher(&format!(
                "CREATE UNIQUE INDEX u_email ON :U(email) OPTIONS {{maintenance: '{profile}'}}"
            ))
            .expect("index");
            db.execute_cypher("CREATE (:U {email: 'same', gen: 0})")
                .expect("the holder");

            let outcomes = race(
                &db,
                &[
                    "MATCH (u:U {email: 'same'}) DELETE u CREATE (:U {email: 'same', gen: 1})"
                        .to_string(),
                    "CREATE (:U {email: 'same', gen: 2})".to_string(),
                ],
            );

            assert_refusals_explained(&outcomes);
            assert_eq!(
                holders(&mut db, "U", "email", "same"),
                1,
                "{profile} round {round}: {outcomes:?}"
            );
        }
    }
}

/// Two nodes moving onto one free value at once: one moves, the other is
/// refused, in each profile.
#[test]
fn concurrent_updates_into_one_value_leave_one_holder() {
    for profile in PROFILES {
        let (mut db, _dir) = open_db();
        db.execute_cypher(&format!(
            "CREATE UNIQUE INDEX u_email ON :U(email) OPTIONS {{maintenance: '{profile}'}}"
        ))
        .expect("index");
        db.execute_cypher("CREATE (:U {email: 'a'}), (:U {email: 'b'}), (:U {email: 'c'})")
            .expect("seed");

        let outcomes = race(
            &db,
            &[
                "MATCH (u:U {email: 'a'}) SET u.email = 'target'".to_string(),
                "MATCH (u:U {email: 'b'}) SET u.email = 'target'".to_string(),
                "MATCH (u:U {email: 'c'}) SET u.email = 'target'".to_string(),
            ],
        );

        assert_refusals_explained(&outcomes);
        assert_eq!(
            holders(&mut db, "U", "email", "target"),
            1,
            "{profile}: {outcomes:?}"
        );
        // A refused writer kept its own value.
        let kept = ["a", "b", "c"]
            .iter()
            .map(|v| holders(&mut db, "U", "email", v))
            .sum::<usize>();
        assert_eq!(kept, 2, "{profile}");
    }
}

/// A compound unique key is held by the combination, not by either value:
/// writers racing for one combination leave one holder, and a combination
/// differing in one value is free.
#[test]
fn a_compound_key_is_held_by_its_combination() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT u_key FOR (n:U) REQUIRE (n.a, n.b) IS UNIQUE")
        .expect("compound constraint");
    db.execute_cypher("CREATE (:U {a: 'x', b: 'y'})")
        .expect("the holder");
    db.execute_cypher("CREATE (:U {a: 'x', b: 'z'})")
        .expect("another combination");

    let outcomes = race(
        &db,
        &(0..4)
            .map(|w| format!("CREATE (:U {{a: 'x', b: 'w', n: {w}}})"))
            .collect::<Vec<_>>(),
    );

    assert_eq!(
        outcomes.iter().filter(|o| o.is_ok()).count(),
        1,
        "{outcomes:?}"
    );
    let err = run(&db, "CREATE (:U {a: 'x', b: 'y'})").expect_err("held");
    assert!(err.contains("unique constraint violated"), "{err}");
    assert_eq!(
        db.execute_cypher("MATCH (n:U) WHERE n.a = 'x' AND n.b = 'w' RETURN n")
            .expect("match")
            .len(),
        1
    );
}

/// A list is held element by element: a list sharing one element with a
/// stored list is refused, in each profile.
#[test]
fn a_list_element_is_held_by_the_list_holding_it() {
    for profile in PROFILES {
        let (mut db, _dir) = open_db();
        db.execute_cypher(&format!(
            "CREATE UNIQUE INDEX u_tags ON :U(tags) OPTIONS {{maintenance: '{profile}'}}"
        ))
        .expect("index");
        db.execute_cypher("CREATE (:U {tags: ['x', 'y']})")
            .expect("the holder");

        let err = run(&db, "CREATE (:U {tags: ['y', 'z']})").expect_err("y is held");
        assert!(
            err.contains("unique constraint violated"),
            "{profile}: {err}"
        );
        run(&db, "CREATE (:U {tags: ['z']})").expect("z is free");
        assert_eq!(
            db.execute_cypher("MATCH (u:U) RETURN u")
                .expect("match")
                .len(),
            2,
            "{profile}"
        );
    }
}

/// Wait until the build of `name` waits for older transactions.
fn await_held(db: &Database, name: &str) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let waiting = index_named(db.engine(), name).is_some_and(|def| {
            db.index_build_status().expect("inspect").iter().any(|s| {
                s.generation == def.generation
                    && s.phase == Some(BuildPhase::AwaitingOlderTransactions)
            })
        });
        if waiting {
            return;
        }
        assert!(Instant::now() < deadline, "the build never waited");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Run `body` while the build of a unique index of `profile` on `:U(email)`
/// waits for an older transaction, so none of the stored nodes has an entry
/// yet; then let the build finish and return its outcome.
fn during_build(db: &Database, profile: &str, body: impl FnOnce()) -> Result<(), String> {
    let older = db.begin_transaction();
    let statement =
        format!("CREATE UNIQUE INDEX u_email ON :U(email) OPTIONS {{maintenance: '{profile}'}}");
    std::thread::scope(|s| {
        let created = s.spawn(|| run(db, &statement));
        await_held(db, "u_email");
        body();
        db.rollback_transaction(older)
            .expect("end the older transaction");
        created.join().expect("join")
    })
}

/// Writers racing for one value while the index is being built: one takes
/// it, the build publishes, and the published index holds the value once.
#[test]
fn concurrent_inserts_during_a_build_leave_one_holder() {
    for profile in PROFILES {
        let (mut db, _dir) = open_db();
        db.execute_cypher("CREATE (:U {email: 'stored'})")
            .expect("a stored node");

        let mut outcomes = Vec::new();
        during_build(&db, profile, || {
            outcomes = race(
                &db,
                &(0..6)
                    .map(|w| format!("CREATE (:U {{email: 'fresh', w: {w}}})"))
                    .collect::<Vec<_>>(),
            );
        })
        .expect("the build publishes");

        assert_eq!(
            outcomes.iter().filter(|o| o.is_ok()).count(),
            1,
            "{profile}: {outcomes:?}"
        );
        assert_refusals_explained(&outcomes);
        assert_eq!(holders(&mut db, "U", "email", "fresh"), 1, "{profile}");
        let err = run(&db, "CREATE (:U {email: 'fresh'})").expect_err("published");
        assert!(
            err.contains("unique constraint violated"),
            "{profile}: {err}"
        );
    }
}

/// While the index is being built, a statement that deletes the stored
/// holder of a value may give the value to a new node, and one that moves
/// the holder off the value frees it: the stored node is judged as the
/// statement leaves it.
#[test]
fn a_holder_removed_during_a_build_frees_its_value() {
    for profile in PROFILES {
        let (mut db, _dir) = open_db();
        db.execute_cypher("CREATE (:U {email: 'a', gen: 0}), (:U {email: 'b'})")
            .expect("stored holders");

        during_build(&db, profile, || {
            run(
                &db,
                "MATCH (u:U {email: 'a'}) DELETE u CREATE (:U {email: 'a', gen: 1})",
            )
            .expect("the deleted holder's value is free");
            run(&db, "MATCH (u:U {email: 'b'}) SET u.email = 'c'").expect("move off");
            run(&db, "CREATE (:U {email: 'b'})").expect("the moved value is free");
        })
        .expect("the build publishes");

        for value in ["a", "b", "c"] {
            assert_eq!(
                holders(&mut db, "U", "email", value),
                1,
                "{profile} {value}"
            );
        }
    }
}
