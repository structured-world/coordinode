//! B-tree indexes whose entries disagree with the records they index.
//!
//! An entry is derived from a record; when the two disagree (an entry naming
//! the wrong node, an entry with no record behind it, a record with no
//! entry), the record is the truth. Reads answer from the records rather
//! than return what the entry claims, a value is never reported held by a
//! node that does not hold it, and never accepted as free on the word of an
//! entry that is known wrong.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::node::{NodeId, NodeRecord, encode_node_key};
use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{encode_entry_key, encode_tuple, encode_unique_entry_key};
use coordinode_embed::Database;
use coordinode_query::index::{CheckOutcome, CheckStatus, IndexSelector, Integrity};
use coordinode_storage::engine::partition::Partition;

use super::helpers::index_named;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// The id of the one `:U` node whose email is `email`, read without any
/// index: a scan of the label.
fn id_of(db: &mut Database, email: &str) -> i64 {
    let rows = db
        .execute_cypher(&format!(
            "MATCH (u:U) WHERE u.email + '' = '{email}' RETURN id(u) AS id"
        ))
        .expect("scan");
    assert_eq!(rows.len(), 1, "one node holds {email}");
    match rows[0].get("id") {
        Some(Value::Int(id)) => *id,
        other => panic!("expected an id, got {other:?}"),
    }
}

/// The ids of the `:U` nodes the indexed equality on `email` finds.
fn found(db: &mut Database, email: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .execute_cypher(&format!(
            "MATCH (u:U {{email: '{email}'}}) RETURN id(u) AS id"
        ))
        .expect("lookup")
        .iter()
        .map(|row| match row.get("id") {
            Some(Value::Int(id)) => *id,
            other => panic!("expected an id, got {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids
}

/// Overwrite the unique entry of `email` in the index `name` so it names
/// `holder`, as a defect that wrote the wrong node would have left it.
fn misattribute(db: &Database, name: &str, email: &str, holder: i64) {
    let index = index_named(db.engine(), name).expect("index");
    let tuple = encode_tuple(&[Value::String(email.into())]).expect("tuple");
    let key = encode_unique_entry_key(index.generation, &tuple);
    let holder = u64::try_from(holder).expect("id").to_be_bytes();
    db.engine()
        .put(Partition::Idx, &key, &holder)
        .expect("overwrite the entry");
}

/// A unique entry names node B while node A is the one holding the value: a
/// lookup by the value finds A, the node that holds it, and not nothing.
#[test]
fn a_lookup_through_a_misattributed_unique_entry_finds_the_real_holder() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'a@x'})").expect("a");
    db.execute_cypher("CREATE (:U {email: 'z@x'})").expect("z");
    let owner = id_of(&mut db, "a@x");
    let wrong = id_of(&mut db, "z@x");
    misattribute(&db, "u_email", "a@x", wrong);

    assert_eq!(found(&mut db, "a@x"), vec![owner]);
}

/// A unique value held by A, whose entry names B: inserting it again is a
/// duplicate of A, the real holder, not of B.
#[test]
fn a_duplicate_of_a_misattributed_value_names_the_real_holder() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'a@x'})").expect("a");
    db.execute_cypher("CREATE (:U {email: 'z@x'})").expect("z");
    let owner = id_of(&mut db, "a@x");
    let wrong = id_of(&mut db, "z@x");
    misattribute(&db, "u_email", "a@x", wrong);

    let err = db
        .execute_cypher("CREATE (:U {email: 'a@x'})")
        .expect_err("a@x is held");
    let message = err.to_string();
    let owner_element =
        coordinode_core::graph::node::NodeId::from_raw(owner as u64).to_element_id();
    assert!(
        message.contains(&owner_element),
        "the duplicate names the holder {owner_element}: {message}"
    );
    assert_eq!(found(&mut db, "a@x"), vec![owner], "nothing was inserted");
}

/// A unique entry left naming B after the value's real holder moved to
/// another value: nobody holds the value, so it can be taken, and the new
/// holder is what a lookup finds.
#[test]
fn a_value_whose_entry_names_a_node_that_never_held_it_can_be_taken() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'a@x'})").expect("a");
    db.execute_cypher("CREATE (:U {email: 'z@x'})").expect("z");
    let wrong = id_of(&mut db, "z@x");
    misattribute(&db, "u_email", "a@x", wrong);
    db.execute_cypher("MATCH (u:U) WHERE u.email + '' = 'a@x' SET u.email = 'moved@x'")
        .expect("move a@x away");

    db.execute_cypher("CREATE (:U {email: 'a@x'})")
        .expect("a@x is free");
    let taker = id_of(&mut db, "a@x");
    assert_eq!(found(&mut db, "a@x"), vec![taker]);
}

/// A MERGE on a misattributed value matches the node that holds it, as the
/// stuck enrollment needed: no false miss, no false duplicate.
#[test]
fn a_merge_on_a_misattributed_value_matches_the_real_holder() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'a@x', n: 1})")
        .expect("a");
    db.execute_cypher("CREATE (:U {email: 'z@x'})").expect("z");
    let owner = id_of(&mut db, "a@x");
    let wrong = id_of(&mut db, "z@x");
    misattribute(&db, "u_email", "a@x", wrong);

    db.execute_cypher("MERGE (u:U {email: 'a@x'}) SET u.n = 2")
        .expect("merge");
    let rows = db
        .execute_cypher("MATCH (u:U) WHERE u.email + '' = 'a@x' RETURN id(u) AS id, u.n AS n")
        .expect("scan");
    assert_eq!(rows.len(), 1, "the merge created no second holder");
    assert_eq!(rows[0].get("id"), Some(&Value::Int(owner)));
    assert_eq!(rows[0].get("n"), Some(&Value::Int(2)));
}

// ── Checks ──────────────────────────────────────────────────────────────

/// How long a test waits for a check or a rebuild.
const WAIT: std::time::Duration = std::time::Duration::from_secs(60);

/// Check the index `name` and wait for the outcome.
fn check(db: &Database, name: &str) -> (CheckStatus, CheckOutcome) {
    let operation = db
        .check_index(&IndexSelector::Name(name.into()))
        .expect("start the check");
    let (status, outcome) = db
        .index_check(operation, WAIT)
        .expect("read the check")
        .expect("the check exists");
    (status, outcome.expect("the check ended"))
}

/// The ids of the `:U` nodes the indexed equality `property = value` finds.
fn found_by(db: &mut Database, property: &str, value: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .execute_cypher(&format!(
            "MATCH (u:U {{{property}: '{value}'}}) RETURN id(u) AS id"
        ))
        .expect("lookup")
        .iter()
        .map(|row| match row.get("id") {
            Some(Value::Int(id)) => *id,
            other => panic!("expected an id, got {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids
}

/// The id of the one `:U` node whose `property` is `value`, by a scan.
fn scan_id(db: &mut Database, property: &str, value: &str) -> i64 {
    let rows = db
        .execute_cypher(&format!(
            "MATCH (u:U) WHERE u.{property} + '' = '{value}' RETURN id(u) AS id"
        ))
        .expect("scan");
    assert_eq!(rows.len(), 1, "one node holds {value}");
    match rows[0].get("id") {
        Some(Value::Int(id)) => *id,
        other => panic!("expected an id, got {other:?}"),
    }
}

/// The key of the entry of node `node` under `value` in the non-unique
/// index `name`.
fn entry_key(db: &Database, name: &str, value: &str, node: i64) -> Vec<u8> {
    let index = index_named(db.engine(), name).expect("index");
    let tuple = encode_tuple(&[Value::String(value.into())]).expect("tuple");
    encode_entry_key(index.generation, &tuple, u64::try_from(node).expect("id"))
}

/// The node the unique entry of `value` in the index `name` names.
fn unique_holder(db: &Database, name: &str, value: &str) -> Option<u64> {
    let index = index_named(db.engine(), name).expect("index");
    let tuple = encode_tuple(&[Value::String(value.into())]).expect("tuple");
    db.engine()
        .get(
            Partition::Idx,
            &encode_unique_entry_key(index.generation, &tuple),
        )
        .expect("read the entry")
        .map(|bytes| u64::from_be_bytes(bytes[..].try_into().expect("8 bytes")))
}

/// A record whose entry is missing gives no sign on any read: the lookup
/// just does not find it. A check finds it from the record side, restores
/// the entry, and verifies the index.
#[test]
fn a_check_restores_an_entry_whose_absence_no_read_shows() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX u_city ON :U(city)")
        .expect("index");
    db.execute_cypher("CREATE (:U {name: 'a', city: 'oslo'})")
        .expect("a");
    db.execute_cypher("CREATE (:U {name: 'b', city: 'oslo'})")
        .expect("b");
    let a = scan_id(&mut db, "name", "a");
    let b = scan_id(&mut db, "name", "b");
    db.engine()
        .delete(Partition::Idx, &entry_key(&db, "u_city", "oslo", b))
        .expect("lose b's entry");

    let (status, outcome) = check(&db, "u_city");
    assert_eq!(
        outcome,
        CheckOutcome::Verified {
            checked: status.record.check.as_ref().expect("check").checked,
            repaired: 1,
        }
    );
    assert_eq!(status.record.integrity, Integrity::Verified);
    assert_eq!(found_by(&mut db, "city", "oslo"), vec![a, b]);
}

/// An entry no record holds is found from the entry side and removed, and
/// a lookup that met it meanwhile answered from the records.
#[test]
fn a_check_removes_an_entry_that_no_record_holds() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX u_city ON :U(city)")
        .expect("index");
    db.execute_cypher("CREATE (:U {name: 'a', city: 'oslo'})")
        .expect("a");
    db.execute_cypher("CREATE (:U {name: 'c', city: 'rome'})")
        .expect("c");
    let a = scan_id(&mut db, "name", "a");
    let c = scan_id(&mut db, "name", "c");
    let stray = entry_key(&db, "u_city", "oslo", c);
    db.engine()
        .put(Partition::Idx, &stray, &[])
        .expect("an entry c does not hold");

    assert_eq!(found_by(&mut db, "city", "oslo"), vec![a]);
    let (status, outcome) = check(&db, "u_city");
    assert!(
        matches!(outcome, CheckOutcome::Verified { .. }),
        "verified: {outcome:?}"
    );
    assert_eq!(status.record.integrity, Integrity::Verified);
    assert!(
        db.engine()
            .get(Partition::Idx, &stray)
            .expect("read")
            .is_none(),
        "the stray entry is gone"
    );
}

/// A unique entry naming the wrong node is rewritten to name the real
/// holder, and the generation is verified again.
#[test]
fn a_check_rewrites_a_misattributed_unique_entry() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'a@x'})").expect("a");
    db.execute_cypher("CREATE (:U {email: 'z@x'})").expect("z");
    let owner = id_of(&mut db, "a@x");
    let wrong = id_of(&mut db, "z@x");
    misattribute(&db, "u_email", "a@x", wrong);

    let (status, outcome) = check(&db, "u_email");
    assert!(
        matches!(outcome, CheckOutcome::Verified { .. }),
        "verified: {outcome:?}"
    );
    assert_eq!(status.record.integrity, Integrity::Verified);
    assert_eq!(
        unique_holder(&db, "u_email", "a@x"),
        Some(u64::try_from(owner).expect("id"))
    );
    assert_eq!(found(&mut db, "a@x"), vec![owner]);
}

/// Two stored records holding one unique value cannot be represented by
/// any entry: the check reports them, changes neither record, and the
/// generation stays suspect, across a reopen too, so every lookup keeps
/// answering from the records.
#[test]
fn records_breaking_a_unique_index_are_reported_and_left_alone() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (a, b, c);
    {
        let mut db = Database::open(dir.path()).expect("open db");
        db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
            .expect("index");
        db.execute_cypher("CREATE (:U {email: 'a@x'})").expect("a");
        db.execute_cypher("CREATE (:U {email: 'b@x'})").expect("b");
        db.execute_cypher("CREATE (:U {email: 'c@x'})").expect("c");
        a = id_of(&mut db, "a@x");
        b = id_of(&mut db, "b@x");
        c = id_of(&mut db, "c@x");
        // b's record changed to a@x behind the index's back.
        let key = encode_node_key(1, NodeId::from_raw(u64::try_from(b).expect("id")));
        let bytes = db
            .engine()
            .get(Partition::Node, &key)
            .expect("read b")
            .expect("b exists");
        let mut record = NodeRecord::from_msgpack(&bytes).expect("decode b");
        let email = db
            .interner()
            .expect("field dictionary")
            .lookup("email")
            .expect("email field");
        record.set(email, Value::String("a@x".into()));
        db.engine()
            .put(
                Partition::Node,
                &key,
                &record.to_msgpack().expect("encode b"),
            )
            .expect("write b");

        let (status, outcome) = check(&db, "u_email");
        assert!(
            matches!(&outcome, CheckOutcome::SourceConflicts { conflicts } if !conflicts.is_empty()),
            "a source conflict: {outcome:?}"
        );
        assert_eq!(status.record.integrity, Integrity::Suspect);
        assert_eq!(found(&mut db, "a@x"), vec![a, b], "both records answer");
    }
    let mut db = Database::open(dir.path()).expect("reopen db");
    let checks = db.index_checks().expect("checks");
    assert!(
        checks
            .iter()
            .any(|s| s.record.integrity == Integrity::Suspect),
        "the suspicion survives the reopen"
    );
    // c's entry lost after the reopen: no read could tell, yet the suspect
    // generation answers c from the records.
    let index = index_named(db.engine(), "u_email").expect("index");
    let tuple = encode_tuple(&[Value::String("c@x".into())]).expect("tuple");
    db.engine()
        .delete(
            Partition::Idx,
            &encode_unique_entry_key(index.generation, &tuple),
        )
        .expect("lose c's entry");
    assert_eq!(found(&mut db, "c@x"), vec![c]);
}

/// Damage wider than a check may repair entry by entry rebuilds the index
/// into a fresh generation, which answers every lookup.
#[test]
fn wide_damage_rebuilds_the_index() {
    let (mut db, _dir) = open_db();
    db.set_index_build_config(coordinode_query::index::IndexBuildConfig {
        check_max_repairs: 1,
        ..db.index_build_config()
    });
    db.execute_cypher("CREATE INDEX u_city ON :U(city)")
        .expect("index");
    let mut ids = Vec::new();
    for name in ["a", "b", "c", "d"] {
        db.execute_cypher(&format!("CREATE (:U {{name: '{name}', city: 'oslo'}})"))
            .expect("node");
        ids.push(scan_id(&mut db, "name", name));
    }
    let before = index_named(db.engine(), "u_city")
        .expect("index")
        .generation;
    for id in &ids[1..] {
        db.engine()
            .delete(Partition::Idx, &entry_key(&db, "u_city", "oslo", *id))
            .expect("lose an entry");
    }

    let (_, outcome) = check(&db, "u_city");
    let CheckOutcome::Rebuilding { generation } = outcome else {
        panic!("a rebuild: {outcome:?}");
    };
    assert_ne!(generation, before);
    db.index_build(generation, WAIT)
        .expect("wait for the build");
    ids.sort_unstable();
    assert_eq!(found_by(&mut db, "city", "oslo"), ids);
}

/// REINDEX by name and by identity rebuilds into a fresh generation each
/// time, and lookups answer throughout.
#[test]
fn reindex_by_name_and_by_identity() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX u_city ON :U(city)")
        .expect("index");
    db.execute_cypher("CREATE (:U {name: 'a', city: 'oslo'})")
        .expect("a");
    let a = scan_id(&mut db, "name", "a");
    let first = index_named(db.engine(), "u_city").expect("index");

    let second = db
        .reindex(&IndexSelector::Name("u_city".into()), WAIT)
        .expect("reindex by name");
    assert_ne!(second, first.generation);
    let third = db
        .reindex(&IndexSelector::Id(first.id), WAIT)
        .expect("reindex by identity");
    assert_ne!(third, second);
    assert_eq!(
        index_named(db.engine(), "u_city")
            .expect("index")
            .generation,
        third
    );
    assert_eq!(found_by(&mut db, "city", "oslo"), vec![a]);

    let err = db
        .reindex(&IndexSelector::Name("no_such".into()), WAIT)
        .expect_err("no such index");
    assert!(err.to_string().contains("no_such"), "{err}");
}

/// `REINDEX` rebuilds into a fresh generation, reports the build like
/// `CREATE INDEX`, refuses a label the index is not on and an index that
/// does not exist.
#[test]
fn reindex_statement() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX u_city ON :U(city)")
        .expect("index");
    db.execute_cypher("CREATE (:U {name: 'a', city: 'oslo'})")
        .expect("a");
    let a = scan_id(&mut db, "name", "a");
    let before = index_named(db.engine(), "u_city")
        .expect("index")
        .generation;

    let rows = db.execute_cypher("REINDEX u_city ON :U").expect("reindex");
    assert_eq!(rows[0].get("state"), Some(&Value::String("READY".into())));
    let after = index_named(db.engine(), "u_city")
        .expect("index")
        .generation;
    assert_ne!(after, before);
    assert_eq!(
        rows[0].get("operation"),
        Some(&Value::Int(
            i64::try_from(after.as_raw()).expect("operation")
        ))
    );
    assert_eq!(found_by(&mut db, "city", "oslo"), vec![a]);

    let wrong_label = db
        .execute_cypher("REINDEX u_city ON :V")
        .expect_err("not on :V");
    assert!(wrong_label.to_string().contains(":U"), "{wrong_label}");
    assert!(db.execute_cypher("REINDEX no_such").is_err());
}

/// The maintenance procedures: a check started by name reaches its
/// outcome, inspection shows it verified, and REINDEX by identity reports
/// its build.
#[test]
fn maintenance_procedures() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX u_city ON :U(city)")
        .expect("index");
    db.execute_cypher("CREATE (:U {name: 'a', city: 'oslo'})")
        .expect("a");
    let a = scan_id(&mut db, "name", "a");
    db.engine()
        .delete(Partition::Idx, &entry_key(&db, "u_city", "oslo", a))
        .expect("lose a's entry");

    let started = db
        .execute_cypher("CALL db.checkIndex('u_city')")
        .expect("start a check");
    let Some(Value::Int(operation)) = started[0].get("operation").cloned() else {
        panic!("an operation: {started:?}");
    };
    let inspected = db
        .execute_cypher(&format!(
            "CALL db.indexCheck({operation}, 60000) YIELD integrity, repaired \
             RETURN integrity, repaired"
        ))
        .expect("inspect the check");
    assert_eq!(
        inspected[0].get("integrity"),
        Some(&Value::String("VERIFIED".into()))
    );
    assert_eq!(inspected[0].get("repaired"), Some(&Value::Int(1)));
    let listed = db
        .execute_cypher("CALL db.indexChecks()")
        .expect("list checks");
    assert_eq!(listed.len(), 1);
    assert_eq!(found_by(&mut db, "city", "oslo"), vec![a]);

    let id = index_named(db.engine(), "u_city").expect("index").id;
    let rebuilt = db
        .execute_cypher(&format!("CALL db.reindex({}, 60000)", id.as_raw()))
        .expect("reindex by identity");
    assert_eq!(
        rebuilt[0].get("state"),
        Some(&Value::String("PUBLISHED".into()))
    );
    let err = db
        .execute_cypher("CALL db.checkIndex('no_such')")
        .expect_err("no such index");
    assert!(err.to_string().contains("no_such"), "{err}");
}

/// Dropping an index removes its integrity records with it.
#[test]
fn dropping_an_index_removes_its_integrity_records() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX u_city ON :U(city)")
        .expect("index");
    db.execute_cypher("CREATE (:U {name: 'a', city: 'oslo'})")
        .expect("a");
    let (_, outcome) = check(&db, "u_city");
    assert!(
        matches!(outcome, CheckOutcome::Verified { .. }),
        "{outcome:?}"
    );
    assert_eq!(db.index_checks().expect("checks").len(), 1);
    db.execute_cypher("DROP INDEX u_city").expect("drop");
    assert!(db.index_checks().expect("checks").is_empty());
}

/// A check of an index that is not a B-tree index, or of no index, is
/// refused rather than reported verified.
#[test]
fn a_check_of_a_missing_index_is_refused() {
    let (db, _dir) = open_db();
    let err = db
        .check_index(&IndexSelector::Name("no_such".into()))
        .expect_err("no such index");
    assert!(err.to_string().contains("no_such"), "{err}");
}

/// A list holding the looked-up value is a legitimate candidate the query's
/// own equality rejects, not a damaged entry: nothing is reported.
#[test]
fn a_multikey_candidate_is_not_reported() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX u_tags ON :U(tags)")
        .expect("index");
    db.execute_cypher("CREATE (:U {name: 'a', tags: ['red', 'blue']})")
        .expect("a");
    assert!(found_by(&mut db, "tags", "red").is_empty());
    std::thread::sleep(std::time::Duration::from_millis(1500));
    assert!(
        db.index_checks().expect("checks").is_empty(),
        "no disagreement was reported"
    );
}
