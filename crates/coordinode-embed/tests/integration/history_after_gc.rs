//! `AS OF TIMESTAMP` on temporal types older than the operational retention:
//! answered from the retained history of the nodes and of the catalog that
//! interprets them, after the row versions of that time are collected and
//! the database is reopened, never from what holds now.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use coordinode_core::graph::types::Value;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_embed::{Database, DatabaseError};
use coordinode_storage::engine::partition::Partition;

/// `Thing`, `X` and `Y`, temporal types with a name; `X` and `Y` hold one
/// node each, valid since the epoch.
fn types(db: &mut Database) {
    for statement in [
        "CREATE NODE TYPE Thing TEMPORAL WITH (name: STRING, valid_from: INT)",
        "CREATE NODE TYPE X TEMPORAL WITH (name: STRING, valid_from: INT)",
        "CREATE NODE TYPE Y TEMPORAL WITH (name: STRING, valid_from: INT)",
        "CREATE (:X {name: 'x', valid_from: 0}), (:Y {name: 'y', valid_from: 0})",
    ] {
        db.execute_cypher(statement)
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
    }
}

/// Whether no index keeps a change's parts apart any more.
fn folded(db: &Database) -> bool {
    use coordinode_modality::IndexStore as _;
    coordinode_modality::LocalIndexStore::new(db.engine())
        .list_definitions()
        .expect("definitions")
        .iter()
        .all(|def| def.selection.is_empty())
}

/// Run `alter`, wait for its parts to be folded, and return a system time
/// at which it is in force: the commit of a write after it.
fn settled(db: &mut Database, alter: &str) -> u64 {
    db.execute_cypher(alter)
        .unwrap_or_else(|e| panic!("{alter}: {e}"));
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !folded(db) {
        assert!(std::time::Instant::now() < deadline, "{alter}: unfolded");
        std::thread::sleep(Duration::from_millis(10));
    }
    mark(db)
}

/// The commit timestamp of a write of an unrelated node.
fn mark(db: &mut Database) -> u64 {
    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, "CREATE (:Mark)", None)
        .expect("mark");
    db.commit_transaction(tx)
        .expect("commit")
        .commit_ts
        .as_raw()
}

/// The names of the things at `as_of`, in order.
fn things(db: &mut Database, as_of: u64) -> Result<Vec<String>, DatabaseError> {
    let rows = db.execute_cypher(&format!(
        "MATCH (n:Thing) RETURN n.name AS name ORDER BY name AS OF TIMESTAMP {as_of}"
    ))?;
    Ok(rows
        .iter()
        .map(|row| match row.get("name") {
            Some(Value::String(name)) => name.clone(),
            other => panic!("name {other:?}"),
        })
        .collect())
}

/// Move the clock past the retention window and collect every row version
/// older than it, in every partition the reads touch.
fn collect_history(db: &mut Database) {
    let oracle = db.engine().oracle().expect("embedded engine has an oracle");
    oracle.advance_to(Timestamp::from_raw(oracle.current().as_raw() + 10_000_000));
    mark(db);
    for part in [
        Partition::Node,
        Partition::Schema,
        Partition::Idx,
        Partition::NodeHistory,
    ] {
        db.engine().force_compaction(part).expect("compact");
    }
}

/// Three interpretations of `Thing` over unchanged nodes: `X` implies it,
/// then `Y` too, then only `Y`. After the row versions of all three times
/// are collected, and again after a reopen, a read at each answers exactly
/// the things of its interpretation: the catalog and the nodes of that time
/// are read from their retained history, not from the current state.
#[test]
fn each_interpretation_answers_after_collection_and_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (before, x_only, both, y_only) = {
        let mut db = Database::open(dir.path()).expect("open");
        db.set_retention_window(Duration::from_secs(2));
        types(&mut db);
        let before = mark(&mut db);
        let x_only = settled(&mut db, "ALTER NODE TYPE X SET IMPLIES (Thing)");
        let both = settled(&mut db, "ALTER NODE TYPE Y SET IMPLIES (Thing)");
        let y_only = settled(&mut db, "ALTER NODE TYPE X SET IMPLIES ()");
        collect_history(&mut db);
        assert!(
            db.oldest_readable_timestamp().as_raw() > y_only,
            "every read below is past the operational retention"
        );
        (before, x_only, both, y_only)
    };
    let expected: [(u64, &[&str]); 4] = [
        (before, &[]),
        (x_only, &["x"]),
        (both, &["x", "y"]),
        (y_only, &["y"]),
    ];
    let mut db = Database::open(dir.path()).expect("reopen");
    db.set_retention_window(Duration::from_secs(2));
    for (as_of, names) in expected {
        assert_eq!(
            things(&mut db, as_of).unwrap_or_else(|e| panic!("as of {as_of}: {e}")),
            names,
            "as of {as_of}"
        );
    }
    assert_eq!(
        db.execute_cypher("MATCH (n:Thing) RETURN n")
            .expect("current")
            .len(),
        1,
        "the current interpretation is the last one"
    );
}

/// A thing as a read knows it: its name and `valid_to`.
type Known = (String, Option<i64>);

/// The things at `as_of`, valid now, in name order.
fn things_with_end(db: &mut Database, as_of: u64) -> Vec<Known> {
    let rows = db
        .execute_cypher(&format!(
            "MATCH (n:Thing) RETURN n.name AS name, n.valid_to AS valid_to ORDER BY name \
             AS OF TIMESTAMP {as_of}"
        ))
        .unwrap_or_else(|e| panic!("as of {as_of}: {e}"));
    rows.iter()
        .map(|row| {
            let name = match row.get("name") {
                Some(Value::String(name)) => name.clone(),
                other => panic!("name {other:?}"),
            };
            let valid_to = match row.get("valid_to") {
                Some(Value::Int(v)) => Some(*v),
                Some(Value::Null) | None => None,
                other => panic!("valid_to {other:?}"),
            };
            (name, valid_to)
        })
        .collect()
}

/// Later facts about a thing leave what was known before as it was, past
/// the operational retention and after a reopen. `Y` comes to imply
/// `Thing`; then its node is renamed (a new version), its current version's
/// end is corrected twice at the same `valid_from` (set, then cleared), a
/// thing valid only from a future time is written, and the node is deleted.
/// A read at each system time answers what was known then, at the current
/// valid time: the corrections each at its own time, the future thing at
/// none, the deleted node up to its deletion.
#[test]
fn later_facts_leave_what_was_known_before() {
    const FUTURE: i64 = 4_000_000_000_000_000;
    let dir = tempfile::tempdir().expect("tempdir");
    let expected = {
        let mut db = Database::open(dir.path()).expect("open");
        db.set_retention_window(Duration::from_secs(2));
        types(&mut db);
        let implied = settled(&mut db, "ALTER NODE TYPE Y SET IMPLIES (Thing)");
        let at = |db: &mut Database, statement: &str| {
            db.execute_cypher(statement)
                .unwrap_or_else(|e| panic!("{statement}: {e}"));
            mark(db)
        };
        let renamed = at(&mut db, "MATCH (n:Y) SET n.name = 'y2'");
        let ended = at(&mut db, &format!("MATCH (n:Y) SET n.valid_to = {FUTURE}"));
        let reopened = at(&mut db, "MATCH (n:Y) SET n.valid_to = null");
        let future = at(
            &mut db,
            &format!("CREATE (:Y {{name: 'later', valid_from: {FUTURE}}})"),
        );
        let deleted = at(&mut db, "MATCH (n:Y) WHERE n.name = 'y2' DELETE n");
        let expected: Vec<(u64, Vec<Known>)> = vec![
            (implied, vec![("y".into(), None)]),
            (renamed, vec![("y2".into(), None)]),
            (ended, vec![("y2".into(), Some(FUTURE))]),
            (reopened, vec![("y2".into(), None)]),
            (future, vec![("y2".into(), None)]),
            (deleted, vec![]),
        ];
        for (as_of, things) in &expected {
            assert_eq!(
                &things_with_end(&mut db, *as_of),
                things,
                "as of {as_of}, within the retention"
            );
        }
        collect_history(&mut db);
        assert!(
            db.oldest_readable_timestamp().as_raw() > deleted,
            "every read below is past the operational retention"
        );
        expected
    };
    let mut db = Database::open(dir.path()).expect("reopen");
    db.set_retention_window(Duration::from_secs(2));
    for (as_of, things) in &expected {
        assert_eq!(
            &things_with_end(&mut db, *as_of),
            things,
            "as of {as_of}, past the retention after a reopen"
        );
    }
}

/// A read past the operational retention of a label whose carriers are not
/// all temporal is refused as past the retention: their past was in the
/// collected row versions only, and part of the answer is no answer. So is
/// a label not bound then, whose absence nothing proves.
#[test]
fn a_label_with_carriers_without_history_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.set_retention_window(Duration::from_secs(2));
    db.execute_cypher("CREATE NODE TYPE Plain WITH (name: STRING)")
        .expect("type");
    db.execute_cypher("CREATE (:Plain {name: 'p'})")
        .expect("node");
    let then = mark(&mut db);
    collect_history(&mut db);
    let read = db.execute_cypher(&format!(
        "MATCH (n:Plain) RETURN n.name AS name AS OF TIMESTAMP {then}"
    ));
    assert!(
        matches!(&read, Err(DatabaseError::OutsideRetention { .. })),
        "{read:?}"
    );
    let read = db.execute_cypher(&format!("MATCH (n:Unseen) RETURN n AS OF TIMESTAMP {then}"));
    assert!(
        matches!(&read, Err(DatabaseError::OutsideRetention { .. })),
        "{read:?}"
    );
}

/// The store's history and its operation descriptors share one domain,
/// whichever founds it: a first temporal write founds the deployment a later
/// descriptor names, and a descriptor issued first founds the one later
/// history binds to, which the write does not replace.
#[test]
fn history_and_descriptors_share_one_domain() {
    use coordinode_core::txn::operation::IssuerId;
    let bound = |db: &Database| {
        coordinode_storage::engine::metadata::bound_deployment(db.engine())
            .expect("binding")
            .expect("founded")
    };
    // History first.
    let mut db = Database::open_in_memory().expect("open");
    types(&mut db);
    let founded = bound(&db);
    let issuer = db.enroll_issuer(IssuerId([0x31; 16])).expect("enroll");
    assert_eq!(issuer.deployment(), founded);
    // A descriptor first.
    let mut db = Database::open_in_memory().expect("open");
    let issuer = db.enroll_issuer(IssuerId([0x32; 16])).expect("enroll");
    types(&mut db);
    assert_eq!(bound(&db), issuer.deployment());
}

/// A read that also reads what only the operational snapshot holds (here
/// an edge) is refused as before, not answered from the history in part.
#[test]
fn a_read_beyond_the_history_keeps_the_retention_refusal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.set_retention_window(Duration::from_secs(2));
    types(&mut db);
    let then = mark(&mut db);
    collect_history(&mut db);
    let read = db.execute_cypher(&format!(
        "MATCH (a:X)-[:R]->(b) RETURN b AS OF TIMESTAMP {then}"
    ));
    assert!(
        matches!(read, Err(DatabaseError::OutsideRetention { .. })),
        "{read:?}"
    );
}
