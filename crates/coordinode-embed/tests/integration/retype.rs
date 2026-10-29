//! RETYPE NODE end to end: a node changes its type label by an explicit
//! statement, keeping its id, its edges and its auxiliary labels, and the
//! complete result is admitted under the new type.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{entries_prefix, unique_entries_prefix};
use coordinode_embed::Database;
use coordinode_storage::engine::partition::Partition;

use super::helpers::index_named;

/// `Thing`, `Goal` implying it, both STRICT; `Person`, STRICT and unrelated;
/// `Activity` auxiliary.
fn kinds(db: &mut Database) {
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING NOT NULL)",
        "CREATE NODE TYPE Goal IMPLIES (Thing) WITH (title: STRING NOT NULL, due: INT)",
        "CREATE NODE TYPE Person WITH (name: STRING NOT NULL)",
        "CREATE LABEL Activity AUXILIARY",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
}

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open db");
    kinds(&mut db);
    (db, dir)
}

fn rows(db: &mut Database, query: &str) -> Vec<std::collections::HashMap<String, Value>> {
    db.execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect()
}

fn count(db: &mut Database, query: &str) -> usize {
    rows(db, query).len()
}

fn labels_of(db: &mut Database, id: i64) -> Vec<String> {
    let found = rows(
        db,
        &format!("MATCH (n) WHERE id(n) = {id} RETURN labels(n) AS l"),
    );
    assert_eq!(found.len(), 1, "node {id}: {found:?}");
    match found[0].get("l") {
        Some(Value::Array(labels)) => labels
            .iter()
            .map(|l| match l {
                Value::String(s) => s.clone(),
                other => panic!("label {other:?}"),
            })
            .collect(),
        other => panic!("labels {other:?}"),
    }
}

fn id_of(db: &mut Database, query: &str) -> i64 {
    let found = rows(db, query);
    assert_eq!(found.len(), 1, "{query}: {found:?}");
    match found[0].get("id") {
        Some(Value::Int(id)) => *id,
        other => panic!("id {other:?}"),
    }
}

fn refused(db: &mut Database, query: &str) -> String {
    match db.execute_cypher(query) {
        Ok(rows) => panic!("{query} was accepted: {rows:?}"),
        Err(e) => e.to_string(),
    }
}

/// How many entries the B-tree index `name` holds.
fn entry_count(db: &Database, name: &str) -> usize {
    let generation = index_named(db.engine(), name)
        .unwrap_or_else(|| panic!("index {name}"))
        .generation;
    [
        entries_prefix(generation),
        unique_entries_prefix(generation),
    ]
    .iter()
    .map(|prefix| {
        db.engine()
            .prefix_scan(Partition::Idx, prefix)
            .expect("scan")
            .count()
    })
    .sum()
}

/// A Thing becomes a Goal: same id, same edge, its auxiliary label kept,
/// and the label Goal implies still carried; a property Goal declares is
/// supplied in the same statement.
#[test]
fn a_thing_becomes_a_goal_keeping_its_identity() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Thing:Activity {title: 't'})-[:REL]->(:Other {k: 1})")
        .expect("nodes");
    let id = id_of(&mut db, "MATCH (n:Thing) RETURN id(n) AS id");

    db.execute_cypher("MATCH (n:Thing {title: 't'}) RETYPE NODE n TO Goal SET n.due = 5")
        .expect("retype");

    assert_eq!(labels_of(&mut db, id), ["Goal", "Activity", "Thing"]);
    assert_eq!(
        id_of(&mut db, "MATCH (n:Goal {due: 5}) RETURN id(n) AS id"),
        id
    );
    assert_eq!(
        count(&mut db, "MATCH (:Goal)-[:REL]->(:Other) RETURN 1 AS one"),
        1,
        "the edge stays"
    );
}

/// A Goal becomes an unrelated Person: the labels Goal implied leave with
/// it, the auxiliary label stays, and the properties Person refuses or
/// requires are changed in the same statement or the change is refused.
#[test]
fn a_goal_becomes_an_unrelated_type_only_with_its_properties_changed() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Goal:Activity {title: 'g'})")
        .expect("goal");
    let id = id_of(&mut db, "MATCH (n:Goal) RETURN id(n) AS id");

    let err = refused(&mut db, "MATCH (n:Goal) RETYPE NODE n TO Person");
    assert!(err.contains("name") || err.contains("title"), "{err}");
    let err = refused(
        &mut db,
        "MATCH (n:Goal) RETYPE NODE n TO Person SET n.name = 'p'",
    );
    assert!(err.contains("title"), "{err}");
    assert_eq!(
        labels_of(&mut db, id),
        ["Goal", "Activity", "Thing"],
        "a refused change leaves the node as it was"
    );

    db.execute_cypher("MATCH (n:Goal) RETYPE NODE n TO Person SET n.name = 'p' REMOVE n.title")
        .expect("retype with its properties");

    assert_eq!(labels_of(&mut db, id), ["Person", "Activity"]);
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 0);
    assert_eq!(count(&mut db, "MATCH (n:Person {name: 'p'}) RETURN n"), 1);
}

/// What cannot be a node's type is refused: an auxiliary label, a temporal
/// type, and a temporal node's move out of its history; ordinary SET of a
/// type label still never changes the type.
#[test]
fn a_retype_outside_its_contract_is_refused() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT)")
        .expect("temporal type");
    db.execute_cypher("CREATE (:Thing {title: 't'})")
        .expect("thing");

    let err = refused(&mut db, "MATCH (n:Thing) RETYPE NODE n TO Activity");
    assert!(err.contains("auxiliary"), "{err}");
    let err = refused(
        &mut db,
        "MATCH (n:Thing) RETYPE NODE n TO Emp SET n.name = 'e' REMOVE n.title",
    );
    assert!(err.contains("temporal"), "{err}");
    let err = refused(&mut db, "MATCH (n:Thing) SET n:Goal");
    assert!(err.contains("cannot change"), "{err}");

    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros() as i64
        - 1_000_000;
    db.execute_cypher(&format!("CREATE (:Emp {{name: 'e', valid_from: {since}}})"))
        .expect("temporal node");
    let err = refused(
        &mut db,
        "MATCH (n:Emp) RETYPE NODE n TO Person SET n.name = 'p'",
    );
    assert!(err.contains("temporal"), "{err}");
    assert_eq!(count(&mut db, "MATCH (n:Goal) RETURN n"), 0);
}

/// The indexes and constraints of the labels a node gains hold it at the
/// change, and those of the labels it loses let it go.
#[test]
fn a_retype_moves_the_node_between_label_indexes_and_constraints() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE")
        .expect("unique on Thing");
    db.execute_cypher("CREATE (:Thing {title: 'x'}), (:Note {title: 'x'}), (:Note {title: 'y'})")
        .expect("nodes");
    let entries = entry_count(&db, "thing_title");
    assert_eq!(entries, 1);

    let err = refused(&mut db, "MATCH (n:Note {title: 'x'}) RETYPE NODE n TO Goal");
    assert!(
        err.contains("unique"),
        "the gained label's constraint: {err}"
    );
    db.execute_cypher("MATCH (n:Note {title: 'y'}) RETYPE NODE n TO Goal")
        .expect("a free value");
    assert_eq!(entry_count(&db, "thing_title"), 2, "joined Thing's index");
    assert_eq!(count(&mut db, "MATCH (n:Thing {title: 'y'}) RETURN n"), 1);

    db.execute_cypher(
        "MATCH (n:Goal {title: 'y'}) RETYPE NODE n TO Person SET n.name = 'y' REMOVE n.title",
    )
    .expect("to an unrelated type");
    assert_eq!(entry_count(&db, "thing_title"), 1, "left Thing's index");
    db.execute_cypher("CREATE (:Thing {title: 'y'})")
        .expect("the value is free again");
}

/// A read at a timestamp before the change sees the node under its old
/// type; a read after it, under the new one. A rolled-back change leaves
/// nothing behind.
#[test]
fn readers_see_the_old_or_the_new_type_never_a_mix() {
    let (mut db, _dir) = open_db();
    let commit = |db: &mut Database, statement: &str| {
        let tx = db.begin_transaction().expect("begin");
        db.execute_in_transaction(tx, statement, None)
            .expect(statement);
        db.commit_transaction(tx)
            .expect("commit")
            .commit_ts
            .as_raw()
    };
    let before = commit(&mut db, "CREATE (:Thing:Activity {title: 't'})");
    commit(&mut db, "MATCH (n:Thing) RETYPE NODE n TO Goal");

    let then = rows(
        &mut db,
        &format!("MATCH (n:Thing) RETURN labels(n) AS l AS OF TIMESTAMP {before}"),
    );
    assert_eq!(
        then[0].get("l"),
        Some(&Value::Array(vec![
            Value::String("Thing".into()),
            Value::String("Activity".into()),
        ]))
    );
    assert_eq!(count(&mut db, "MATCH (n:Goal) RETURN n"), 1);

    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(
        tx,
        "MATCH (n:Goal) RETYPE NODE n TO Person SET n.name = 'p' REMOVE n.title",
        None,
    )
    .expect("retype in a transaction");
    let own = db
        .execute_in_transaction(tx, "MATCH (n:Person) RETURN n", None)
        .expect("own read");
    assert_eq!(own.len(), 1, "the transaction sees its own change");
    assert_eq!(
        count(&mut db, "MATCH (n:Person) RETURN n"),
        0,
        "others do not"
    );
    db.rollback_transaction(tx).expect("rollback");
    assert_eq!(count(&mut db, "MATCH (n:Goal) RETURN n"), 1);
    assert_eq!(count(&mut db, "MATCH (n:Person) RETURN n"), 0);
}

/// Two statements retyping one node at once: one commits, the other is
/// refused, and the node carries exactly one type.
#[test]
fn concurrent_retypes_of_one_node_leave_one_type() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Thing {title: 't'})")
        .expect("thing");
    let statements = [
        "MATCH (n:Thing) RETYPE NODE n TO Goal".to_string(),
        "MATCH (n:Thing) RETYPE NODE n TO Person SET n.name = 'p' REMOVE n.title".to_string(),
    ];
    let start = std::sync::Barrier::new(statements.len());
    let outcomes: Vec<Result<(), String>> = std::thread::scope(|s| {
        let handles: Vec<_> = statements
            .iter()
            .map(|statement| {
                let (db, start) = (&db, &start);
                s.spawn(move || {
                    start.wait();
                    db.execute_cypher_shared(statement, None, None, None, None)
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("writer"))
            .collect()
    });
    let committed = outcomes.iter().filter(|o| o.is_ok()).count();
    assert!(committed >= 1, "{outcomes:?}");
    let goals = count(&mut db, "MATCH (n:Goal) RETURN n");
    let persons = count(&mut db, "MATCH (n:Person) RETURN n");
    assert_eq!(goals + persons, 1, "{outcomes:?}");
    if committed == 2 {
        // Serialized, not concurrent: the second saw the first's result.
        assert_eq!(persons, 1, "{outcomes:?}");
    }
}

/// A changed type survives a restart: the reopened database holds the node
/// under its new type, with its index entries.
#[test]
fn a_retype_survives_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let id = {
        let mut db = Database::open(dir.path()).expect("open");
        kinds(&mut db);
        db.execute_cypher("CREATE INDEX thing_title ON :Thing(title)")
            .expect("index");
        db.execute_cypher("CREATE (:Thing {title: 'a'})")
            .expect("thing");
        db.execute_cypher("MATCH (n:Thing) RETYPE NODE n TO Goal")
            .expect("retype");
        id_of(&mut db, "MATCH (n:Goal) RETURN id(n) AS id")
    };
    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(labels_of(&mut db, id), ["Goal", "Thing"]);
    assert_eq!(count(&mut db, "MATCH (n:Thing {title: 'a'}) RETURN n"), 1);
    assert_eq!(entry_count(&db, "thing_title"), 1);
}

/// A check of the B-tree index `name` against its records finds every entry
/// the records call for and nothing else: it repairs nothing.
fn assert_index_agrees(db: &Database, name: &str) {
    use coordinode_query::index::{CheckOutcome, IndexSelector};
    let operation = db
        .check_index(&IndexSelector::Name(name.into()))
        .expect("start the check");
    let (_, outcome) = db
        .index_check(operation, std::time::Duration::from_secs(30))
        .expect("read the check")
        .expect("the check exists");
    assert!(
        matches!(outcome, Some(CheckOutcome::Verified { repaired: 0, .. })),
        "{name}: {outcome:?}"
    );
}

/// The type, labels, properties, edges and index entries of node `id`,
/// read back to compare one state with another.
fn state_of(db: &mut Database, id: i64) -> Vec<std::collections::HashMap<String, Value>> {
    rows(
        db,
        &format!(
            "MATCH (n) WHERE id(n) = {id} OPTIONAL MATCH (n)-[r]->(m) \
             RETURN labels(n) AS l, n.title AS title, n.due AS due, n.name AS name, \
             type(r) AS rel, id(m) AS other"
        ),
    )
}

/// The value a node holds when it is retyped may clash with a unique index
/// of its new type as long as the same statement changes it: only the
/// complete result answers to the index. The holder of the clashing value
/// keeps it; the retyped node holds its final value under its own id, with
/// its auxiliary label and its edge.
#[test]
fn a_retype_is_judged_by_its_final_values_not_by_its_steps() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE")
        .expect("unique on Thing");
    db.execute_cypher("CREATE (:Thing {title: 'x'})")
        .expect("holder");
    db.execute_cypher("CREATE (:Note:Activity {title: 'x'})-[:REL]->(:Other {k: 1})")
        .expect("note");
    let holder = id_of(&mut db, "MATCH (n:Thing {title: 'x'}) RETURN id(n) AS id");
    let note = id_of(&mut db, "MATCH (n:Note) RETURN id(n) AS id");

    db.execute_cypher("MATCH (n:Note) RETYPE NODE n TO Goal SET n.title = 'y'")
        .expect("the final value is free");

    assert_eq!(
        id_of(&mut db, "MATCH (n:Thing {title: 'x'}) RETURN id(n) AS id"),
        holder,
        "the holder keeps its value"
    );
    assert_eq!(
        id_of(&mut db, "MATCH (n:Thing {title: 'y'}) RETURN id(n) AS id"),
        note,
        "the final value belongs to the retyped node"
    );
    assert_eq!(labels_of(&mut db, note), ["Goal", "Activity", "Thing"]);
    assert_eq!(count(&mut db, "MATCH (n:Note) RETURN n"), 0);
    assert_eq!(
        count(&mut db, "MATCH (:Goal)-[:REL]->(:Other) RETURN 1 AS one"),
        1
    );
    assert_eq!(entry_count(&db, "thing_title"), 2);
    assert_index_agrees(&db, "thing_title");
}

/// A retype whose final value breaks a unique index of its new type is
/// refused, and its transaction with it: the write made earlier in the same
/// transaction is gone, and the node keeps its type, labels, properties,
/// edge and index entries, before and after a reopen.
#[test]
fn a_retype_whose_result_breaks_a_unique_index_leaves_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let note;
    let before;
    {
        let mut db = Database::open(dir.path()).expect("open");
        kinds(&mut db);
        db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE")
            .expect("unique on Thing");
        db.execute_cypher("CREATE (:Thing {title: 'x'})")
            .expect("holder");
        db.execute_cypher("CREATE (:Note:Activity {title: 'z'})-[:REL]->(:Other {k: 1})")
            .expect("note");
        note = id_of(&mut db, "MATCH (n:Note) RETURN id(n) AS id");
        before = state_of(&mut db, note);

        let tx = db.begin_transaction().expect("begin");
        db.execute_in_transaction(tx, "CREATE (:Witness {k: 1})", None)
            .expect("the earlier write");
        let refused = db.execute_in_transaction(
            tx,
            "MATCH (n:Note) RETYPE NODE n TO Goal SET n.title = 'x'",
            None,
        );
        let err = refused.expect_err("the final value is held").to_string();
        assert!(err.contains("unique"), "{err}");
        assert!(
            db.commit_transaction(tx).is_err(),
            "the refused statement ends its transaction"
        );

        assert_eq!(count(&mut db, "MATCH (n:Witness) RETURN n"), 0);
        assert_eq!(state_of(&mut db, note), before);
        assert_eq!(count(&mut db, "MATCH (n:Goal) RETURN n"), 0);
        assert_eq!(entry_count(&db, "thing_title"), 1);
        assert_index_agrees(&db, "thing_title");
    }
    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(count(&mut db, "MATCH (n:Witness) RETURN n"), 0);
    assert_eq!(state_of(&mut db, note), before);
    assert_eq!(entry_count(&db, "thing_title"), 1);
    assert_index_agrees(&db, "thing_title");
}

/// A node of an undeclared type becomes another undeclared type with every
/// property it had.
#[test]
fn an_undeclared_type_becomes_another_with_its_properties() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Draft {a: 1, b: 'x'})")
        .expect("draft");
    db.execute_cypher("MATCH (n:Draft) RETYPE NODE n TO Note")
        .expect("retype");
    let found = rows(&mut db, "MATCH (n:Note) RETURN n.a AS a, n.b AS b");
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].get("a"), Some(&Value::Int(1)));
    assert_eq!(found[0].get("b"), Some(&Value::String("x".into())));
    assert_eq!(count(&mut db, "MATCH (n:Draft) RETURN n"), 0);
}
