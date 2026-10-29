//! Label kinds end to end: a node carries exactly one type label, found by
//! kind whatever order its labels are written in, the labels that type
//! implies, and auxiliary labels. The type's schema governs the whole node,
//! and every write that sets labels keeps that rule.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{entries_prefix, unique_entries_prefix};
use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};
use coordinode_embed::Database;
use coordinode_query::index::{BuildPhase, CheckOutcome, IndexSelector};
use coordinode_storage::engine::partition::Partition;

use super::helpers::index_named;

/// The documented example: `Goal` implies `Thing`, both STRICT, and
/// `Activity` and `Policy` auxiliary.
fn goals() -> Database {
    let mut db = Database::open_in_memory().expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING NOT NULL)",
        "CREATE NODE TYPE Goal IMPLIES (Thing) WITH (title: STRING NOT NULL, due: TIMESTAMP)",
        "CREATE LABEL Activity AUXILIARY",
        "CREATE LABEL Policy AUXILIARY",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    db
}

fn labels(db: &mut Database, query: &str) -> Vec<String> {
    let rows = db
        .execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"));
    assert_eq!(rows.len(), 1, "{query}: {rows:?}");
    match rows[0].get("l") {
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

fn count(db: &mut Database, query: &str) -> usize {
    db.execute_cypher(query)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .len()
}

fn refused(db: &mut Database, query: &str) -> String {
    match db.execute_cypher(query) {
        Ok(rows) => panic!("{query} was accepted: {rows:?}"),
        Err(e) => e.to_string(),
    }
}

/// The first escape: an undeclared label beside a STRICT type is a second
/// type label, so it cannot carry an undeclared property past the type.
#[test]
fn an_undeclared_label_cannot_switch_the_schema_off() {
    let mut db = goals();
    let err = refused(&mut db, "CREATE (:Extra:Goal {title: 'x', unheard_of: 1})");
    assert!(err.contains("two type labels"), "{err}");
    assert!(err.contains("Extra") && err.contains("Goal"), "{err}");
    assert_eq!(count(&mut db, "MATCH (n:Goal) RETURN n"), 0);
}

/// The second escape: an auxiliary label alone, or two of them, type
/// nothing.
#[test]
fn an_auxiliary_label_never_stands_alone() {
    let mut db = goals();
    let err = refused(&mut db, "CREATE (:Activity)");
    assert!(err.contains("no type label"), "{err}");
    let err = refused(&mut db, "CREATE (:Activity:Policy)");
    assert!(err.contains("no type label"), "{err}");
    let err = refused(&mut db, "MERGE (:Activity {k: 1})");
    assert!(err.contains("no type label"), "{err}");
    assert_eq!(count(&mut db, "MATCH (n:Activity) RETURN n"), 0);
}

/// The type is found by kind: the order the labels are written in changes
/// neither the schema checked nor the labels stored, and the labels the
/// type implies are written with the node and found by MATCH.
#[test]
fn the_type_is_found_by_kind_and_implied_labels_are_written() {
    let mut db = goals();
    db.execute_cypher("CREATE (:Activity:Goal {title: 'a'})")
        .expect("auxiliary first");
    db.execute_cypher("CREATE (:Goal:Activity {title: 'b'})")
        .expect("type first");
    for title in ["a", "b"] {
        assert_eq!(
            labels(
                &mut db,
                &format!("MATCH (n {{title: '{title}'}}) RETURN labels(n) AS l")
            ),
            ["Goal", "Activity", "Thing"],
            "{title}"
        );
    }
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 2);
    // The type's schema governs whichever position it was written in.
    for query in [
        "CREATE (:Activity:Goal {title: 'c', note: 1})",
        "CREATE (:Goal:Activity {title: 'c', note: 1})",
    ] {
        let err = refused(&mut db, query);
        assert!(err.contains("note"), "{query}: {err}");
    }
    let err = refused(&mut db, "CREATE (:Activity:Goal)");
    assert!(err.contains("title"), "{err}");
}

/// An implied label written beside its type is not a second type.
#[test]
fn a_written_implied_label_is_not_a_second_type() {
    let mut db = goals();
    db.execute_cypher("CREATE (:Thing:Goal {title: 'g'})")
        .expect("implied label written");
    assert_eq!(
        labels(&mut db, "MATCH (n:Goal) RETURN labels(n) AS l"),
        ["Goal", "Thing"]
    );
}

/// SET of a label keeps one type: an auxiliary label is added, a second
/// type is refused, and so is a type that would replace the node's own.
#[test]
fn set_of_a_label_keeps_one_type() {
    let mut db = goals();
    db.execute_cypher("CREATE NODE TYPE Person").expect("type");
    db.execute_cypher("CREATE (:Goal {title: 'g'})")
        .expect("goal");
    db.execute_cypher("MATCH (n:Goal) SET n:Policy")
        .expect("auxiliary added");
    assert_eq!(
        labels(&mut db, "MATCH (n:Goal) RETURN labels(n) AS l"),
        ["Goal", "Policy", "Thing"]
    );
    let err = refused(&mut db, "MATCH (n:Goal) SET n:Person");
    assert!(err.contains("two type labels"), "{err}");

    db.execute_cypher("CREATE (:Thing {title: 't'})")
        .expect("thing");
    let err = refused(&mut db, "MATCH (n:Thing {title: 't'}) SET n:Goal");
    assert!(err.contains("cannot change"), "{err}");
    assert_eq!(count(&mut db, "MATCH (n:Goal) RETURN n"), 1);
}

/// REMOVE of a label keeps one type: the type cannot be taken away while
/// auxiliary labels remain, nor a label the type implies; an auxiliary one
/// can.
#[test]
fn remove_of_a_label_keeps_one_type() {
    let mut db = goals();
    db.execute_cypher("CREATE (:Goal:Activity {title: 'g'})")
        .expect("goal");
    let err = refused(&mut db, "MATCH (n:Goal) REMOVE n:Goal");
    assert!(err.contains("type"), "{err}");
    let err = refused(&mut db, "MATCH (n:Goal) REMOVE n:Thing");
    assert!(err.contains("implied"), "{err}");
    db.execute_cypher("MATCH (n:Goal) REMOVE n:Activity")
        .expect("auxiliary removed");
    assert_eq!(
        labels(&mut db, "MATCH (n:Goal) RETURN labels(n) AS l"),
        ["Goal", "Thing"]
    );
    db.execute_cypher("CREATE (:Lone {k: 1})").expect("lone");
    let err = refused(&mut db, "MATCH (n:Lone) REMOVE n:Lone");
    assert!(err.contains("no type label"), "{err}");
}

/// A write opening a new version of a temporal node keeps the rule: an
/// auxiliary label joins the new version, a second type does not.
#[test]
fn a_new_temporal_version_keeps_one_type() {
    let mut db = goals();
    db.execute_cypher("CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT)")
        .expect("temporal type");
    db.execute_cypher("CREATE NODE TYPE Person").expect("type");
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros() as i64
        - 1_000_000;
    db.execute_cypher(&format!("CREATE (:Emp {{name: 'e', valid_from: {since}}})"))
        .expect("temporal node");
    db.execute_cypher("MATCH (n:Emp) SET n:Policy, n.name = 'f'")
        .expect("auxiliary label in a new version");
    assert_eq!(
        labels(&mut db, "MATCH (n:Emp) RETURN labels(n) AS l"),
        ["Emp", "Policy"]
    );
    let err = refused(&mut db, "MATCH (n:Emp) SET n:Person, n.name = 'g'");
    assert!(err.contains("two type labels"), "{err}");
}

/// What a declaration may say is checked when it is made.
#[test]
fn declarations_are_checked() {
    let mut db = goals();
    // An implied label's property must be in the implying schema, with its
    // type.
    let err = refused(
        &mut db,
        "CREATE NODE TYPE Task IMPLIES (Thing) WITH (due: TIMESTAMP)",
    );
    assert!(err.contains("title"), "{err}");
    let err = refused(
        &mut db,
        "CREATE NODE TYPE Task IMPLIES (Thing) WITH (title: INT)",
    );
    assert!(err.contains("title"), "{err}");
    // Nothing implies an auxiliary label, nor itself.
    let err = refused(&mut db, "CREATE NODE TYPE Task IMPLIES (Activity)");
    assert!(err.contains("auxiliary"), "{err}");
    let err = refused(&mut db, "CREATE NODE TYPE Task IMPLIES (Task)");
    assert!(err.contains("itself"), "{err}");
    // A label already declared is not declared again.
    refused(&mut db, "CREATE LABEL Goal AUXILIARY");
    assert_eq!(count(&mut db, "MATCH (n:Task) RETURN n"), 0);
}

/// A declaration cannot leave stored nodes invalid: a label nodes already
/// carry as their type is not declared auxiliary, and a type whose stored
/// nodes lack a label does not come to imply it.
#[test]
fn a_declaration_keeps_stored_nodes_valid() {
    let mut db = Database::open_in_memory().expect("open");
    db.execute_cypher("CREATE (:Tag {k: 1}), (:Item {k: 2})")
        .expect("nodes");
    let err = refused(&mut db, "CREATE LABEL Tag AUXILIARY");
    assert!(err.contains("Tag"), "{err}");
    let err = refused(&mut db, "CREATE NODE TYPE Item IMPLIES (Thing)");
    assert!(err.contains("Thing"), "{err}");
    // A label no node carries is declared at once.
    db.execute_cypher("CREATE LABEL Fresh AUXILIARY")
        .expect("unused label");
}

/// The kind and the implied labels are fixed when a label is created: the
/// typed catalog calls refuse a definition that changes either, and refuse
/// an auxiliary label with properties.
#[test]
fn the_typed_catalog_keeps_kinds_fixed() {
    let db = Database::open_in_memory().expect("open");
    let mut goal = LabelSchema::new_node_id("Goal");
    goal.add_property(PropertyDef::new("title", PropertyType::String));
    db.define_label(goal.clone()).expect("define");

    let mut implying = goal.clone();
    implying.set_implies(vec!["Thing".to_string()]);
    let err = db
        .create_label_schema(implying)
        .expect_err("implied labels changed");
    assert!(err.to_string().contains("implies"), "{err}");

    let mut auxiliary = LabelSchema::new_node_id("Goal");
    auxiliary.set_auxiliary();
    let err = db.create_label_schema(auxiliary).expect_err("kind changed");
    assert!(err.to_string().contains("auxiliary"), "{err}");

    let mut with_props = LabelSchema::new_node_id("Scope");
    with_props.set_auxiliary();
    with_props.add_property(PropertyDef::new("x", PropertyType::Int));
    let err = db
        .define_label(with_props)
        .expect_err("auxiliary with properties");
    assert!(err.to_string().contains("properties"), "{err}");
}

/// A node is found under a label its type implies by every plan: a scan of
/// the label and a lookup through an index on it return the same nodes.
#[test]
fn an_implied_label_is_found_by_its_index() {
    let mut db = goals();
    db.execute_cypher("CREATE INDEX thing_title ON :Thing(title)")
        .expect("index");
    db.execute_cypher("CREATE (:Goal {title: 'x'}), (:Thing {title: 'x'})")
        .expect("nodes");
    assert_eq!(
        count(&mut db, "MATCH (n:Thing) WHERE n.title + '' = 'x' RETURN n"),
        2,
        "scanned"
    );
    let lookup = "MATCH (n:Thing {title: 'x'}) RETURN n";
    let plan = db.explain_cypher(lookup).expect("explain");
    assert!(plan.contains("IndexScan"), "the index path:\n{plan}");
    assert_eq!(count(&mut db, lookup), 2, "through the index");
}

/// An index built over nodes already stored covers the ones carrying its
/// label implied, as one maintained from the first write does.
#[test]
fn an_index_built_later_covers_implied_carriers() {
    let mut db = goals();
    db.execute_cypher("CREATE (:Goal {title: 'x'}), (:Thing {title: 'x'})")
        .expect("nodes");
    db.execute_cypher("CREATE INDEX thing_title ON :Thing(title)")
        .expect("index over stored nodes");
    assert_eq!(
        count(&mut db, "MATCH (n:Thing {title: 'x'}) RETURN n"),
        2,
        "through the index"
    );
}

/// An index on an auxiliary label covers the nodes carrying it: a node
/// joins it when the label is added and leaves it when the label is
/// removed, whatever its type.
#[test]
fn an_auxiliary_label_index_follows_label_changes() {
    let mut db = goals();
    db.execute_cypher("CREATE INDEX activity_title ON :Activity(title)")
        .expect("index");
    db.execute_cypher("CREATE (:Goal:Activity {title: 'a'}), (:Thing {title: 'a'})")
        .expect("nodes");
    let through_index = |db: &mut Database| count(db, "MATCH (n:Activity {title: 'a'}) RETURN n");
    let plan = db
        .explain_cypher("MATCH (n:Activity {title: 'a'}) RETURN n")
        .expect("explain");
    assert!(plan.contains("IndexScan"), "the index path:\n{plan}");
    assert_eq!(through_index(&mut db), 1);
    db.execute_cypher("MATCH (n:Thing {title: 'a'}) WHERE NOT 'Goal' IN labels(n) SET n:Activity")
        .expect("label added");
    assert_eq!(through_index(&mut db), 2, "joined at SET");
    db.execute_cypher("MATCH (n:Goal) REMOVE n:Activity")
        .expect("label removed");
    assert_eq!(through_index(&mut db), 1, "left at REMOVE");
    assert_eq!(
        count(
            &mut db,
            "MATCH (n:Activity) WHERE n.title + '' = 'a' RETURN n"
        ),
        1,
        "the scan agrees"
    );
}

/// A uniqueness declared on an implied label holds across every type
/// implying it: a Goal and a Thing cannot share the value.
#[test]
fn a_uniqueness_on_an_implied_label_spans_its_types() {
    let mut db = goals();
    db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE")
        .expect("constraint");
    db.execute_cypher("CREATE (:Thing {title: 'one'})")
        .expect("first");
    let err = refused(&mut db, "CREATE (:Goal {title: 'one'})");
    assert!(err.contains("one") || err.contains("unique"), "{err}");
    db.execute_cypher("CREATE (:Goal {title: 'two'})")
        .expect("another value");
}

/// A presence constraint declared on an auxiliary label binds every node
/// carrying it, whatever its type, and is refused over a stored carrier
/// that breaks it.
#[test]
fn a_constraint_on_an_auxiliary_label_binds_its_carriers() {
    let mut db = goals();
    db.execute_cypher("CREATE (:Goal:Policy {title: 'p'})")
        .expect("carrier without owner");
    let err = refused(
        &mut db,
        "CREATE CONSTRAINT policy_owner FOR (n:Policy) REQUIRE n.owner IS NOT NULL",
    );
    assert!(err.contains("owner"), "{err}");
    db.execute_cypher("MATCH (n:Policy) REMOVE n:Policy")
        .expect("no carrier left");
    db.execute_cypher("CREATE CONSTRAINT policy_owner FOR (n:Policy) REQUIRE n.owner IS NOT NULL")
        .expect("constraint");
    db.execute_cypher("CREATE NODE TYPE Rule WITH (owner: STRING)")
        .expect("type");
    let err = refused(&mut db, "CREATE (:Rule:Policy)");
    assert!(err.contains("owner"), "{err}");
    db.execute_cypher("CREATE (:Rule:Policy {owner: 'ops'})")
        .expect("carrier with owner");
    db.execute_cypher("CREATE (:Rule)").expect("no carrier");
}

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

/// How many entries the B-tree index `name` holds, read from storage
/// without the planner.
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

/// A check of the index `name` against its records finds every entry the
/// records call for and nothing else: it repairs nothing.
fn assert_index_agrees(db: &Database, name: &str) {
    let operation = db
        .check_index(&IndexSelector::Name(name.into()))
        .expect("start the check");
    let (_, outcome) = db
        .index_check(operation, Duration::from_secs(30))
        .expect("read the check")
        .expect("the check exists");
    assert!(
        matches!(outcome, Some(CheckOutcome::Verified { repaired: 0, .. })),
        "{name}: {outcome:?}"
    );
}

/// Diamond and transitive membership: `Top` implies `Left` and `Right`,
/// both imply `Base`. A `Top` node carries `Base` once, holds one entry in
/// an index on `Base`, and that entry follows its value and its deletion.
#[test]
fn a_label_reached_twice_is_carried_and_indexed_once() {
    let (mut db, _dir) = open_db();
    for ddl in [
        "CREATE NODE TYPE Base WITH (k: STRING)",
        "CREATE NODE TYPE Left IMPLIES (Base) WITH (k: STRING)",
        "CREATE NODE TYPE Right IMPLIES (Base) WITH (k: STRING)",
        "CREATE NODE TYPE Top IMPLIES (Left, Right) WITH (k: STRING)",
        "CREATE INDEX base_k ON :Base(k)",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    db.execute_cypher("CREATE (:Top {k: 'x'})").expect("top");
    assert_eq!(
        labels(&mut db, "MATCH (n:Top) RETURN labels(n) AS l"),
        ["Top", "Base", "Left", "Right"]
    );
    assert_eq!(entry_count(&db, "base_k"), 1, "one entry for one node");
    for label in ["Base", "Left", "Right", "Top"] {
        assert_eq!(
            count(&mut db, &format!("MATCH (n:{label} {{k: 'x'}}) RETURN n")),
            1,
            "{label}"
        );
    }
    db.execute_cypher("MATCH (n:Top) SET n.k = 'y'")
        .expect("value moved");
    assert_eq!(count(&mut db, "MATCH (n:Base {k: 'x'}) RETURN n"), 0);
    assert_eq!(count(&mut db, "MATCH (n:Base {k: 'y'}) RETURN n"), 1);
    assert_index_agrees(&db, "base_k");
    db.execute_cypher("MATCH (n:Top) DETACH DELETE n")
        .expect("deleted");
    assert_eq!(
        entry_count(&db, "base_k"),
        0,
        "the entry left with its node"
    );
}

/// A uniqueness on a label reached through a diamond holds across every
/// type reaching it.
#[test]
fn a_uniqueness_reached_through_a_diamond_spans_its_types() {
    let mut db = Database::open_in_memory().expect("open");
    for ddl in [
        "CREATE NODE TYPE Base WITH (k: STRING)",
        "CREATE NODE TYPE Left IMPLIES (Base) WITH (k: STRING)",
        "CREATE NODE TYPE Right IMPLIES (Base) WITH (k: STRING)",
        "CREATE NODE TYPE Top IMPLIES (Left, Right) WITH (k: STRING)",
        "CREATE CONSTRAINT base_k FOR (n:Base) REQUIRE n.k IS UNIQUE",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    db.execute_cypher("CREATE (:Right {k: 'z'})")
        .expect("holder");
    for query in ["CREATE (:Top {k: 'z'})", "CREATE (:Left {k: 'z'})"] {
        let err = refused(&mut db, query);
        assert!(err.contains("unique"), "{query}: {err}");
    }
    db.execute_cypher("CREATE (:Top {k: 'w'})")
        .expect("another value");
}

/// A compound uniqueness on an auxiliary label binds its carriers by the
/// combination, whatever their types; a node without the label is free,
/// and adding the label to it is refused while the combination is held.
#[test]
fn a_compound_uniqueness_on_an_auxiliary_label_binds_its_carriers() {
    let mut db = Database::open_in_memory().expect("open");
    db.execute_cypher("CREATE LABEL Policy AUXILIARY")
        .expect("auxiliary");
    db.execute_cypher("CREATE CONSTRAINT p_key FOR (n:Policy) REQUIRE (n.a, n.b) IS UNIQUE")
        .expect("compound constraint");
    db.execute_cypher("CREATE (:Rec:Policy {a: 'x', b: 'y'})")
        .expect("holder");
    let err = refused(&mut db, "CREATE (:Doc:Policy {a: 'x', b: 'y'})");
    assert!(err.contains("unique"), "{err}");
    db.execute_cypher("CREATE (:Doc:Policy {a: 'x', b: 'z'})")
        .expect("another combination");
    db.execute_cypher("CREATE (:Doc {a: 'x', b: 'y'})")
        .expect("no carrier");
    let err = refused(&mut db, "MATCH (n:Doc {b: 'y'}) SET n:Policy");
    assert!(err.contains("unique"), "{err}");
    assert_eq!(count(&mut db, "MATCH (n:Policy) RETURN n"), 2);
}

/// Multikey, sparse and partial indexes on an auxiliary label hold exactly
/// the entries their carriers call for, through label adds and removals,
/// and a check of each against the records repairs nothing.
#[test]
fn multikey_sparse_and_partial_indexes_follow_their_carriers() {
    let (mut db, _dir) = open_db();
    for ddl in [
        "CREATE LABEL Policy AUXILIARY",
        "CREATE INDEX p_tags ON :Policy(tags)",
        "CREATE SPARSE INDEX p_code ON :Policy(code)",
        "CREATE INDEX p_active ON :Policy(name) WHERE n.active = true",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    db.execute_cypher(
        "CREATE (:Rec:Policy {name: 'a', tags: ['r', 'b'], code: 'c1', active: true}), \
         (:Doc:Policy {name: 'b', tags: ['r'], active: false}), \
         (:Doc {name: 'c', tags: ['g', 'b'], code: 'c3', active: true})",
    )
    .expect("nodes");
    let all = |db: &Database| {
        [
            entry_count(db, "p_tags"),
            entry_count(db, "p_code"),
            entry_count(db, "p_active"),
        ]
    };
    // Tags: two of `a`, one of `b`; codes: `a` only (`b` has none); active:
    // `a` only.
    assert_eq!(all(&db), [3, 1, 1]);
    db.execute_cypher("MATCH (n:Doc {name: 'c'}) SET n:Policy")
        .expect("label added");
    assert_eq!(all(&db), [5, 2, 2], "c joined each index");
    db.execute_cypher("MATCH (n:Rec) REMOVE n:Policy")
        .expect("label removed");
    assert_eq!(all(&db), [3, 1, 1], "a left each index");
    for name in ["p_tags", "p_code", "p_active"] {
        assert_index_agrees(&db, name);
    }
    assert_eq!(
        count(
            &mut db,
            "MATCH (n:Policy) WHERE n.name = 'c' AND n.active = true RETURN n"
        ),
        1
    );
}

/// A full-text index on a label a type implies finds every carrier, and
/// one on an auxiliary label follows the label's adds and removals.
#[test]
fn a_text_index_covers_implied_and_auxiliary_carriers() {
    let (mut db, _dir) = open_db();
    for ddl in [
        "CREATE NODE TYPE Thing WITH (body: STRING)",
        "CREATE NODE TYPE Goal IMPLIES (Thing) WITH (body: STRING)",
        "CREATE LABEL Pinned AUXILIARY",
        "CREATE TEXT INDEX thing_body ON :Thing(body)",
        "CREATE TEXT INDEX pinned_body ON :Pinned(body)",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    db.execute_cypher(
        "CREATE (:Goal {body: 'wandering words'}), (:Thing {body: 'wandering stars'})",
    )
    .expect("nodes");
    let found = |db: &mut Database, label: &str| {
        count(
            db,
            &format!("MATCH (n:{label}) WHERE text_match(n.body, 'wandering') RETURN n"),
        )
    };
    assert_eq!(found(&mut db, "Thing"), 2, "implied carriers");
    assert_eq!(found(&mut db, "Pinned"), 0);
    db.execute_cypher("MATCH (n:Goal) SET n:Pinned")
        .expect("label added");
    assert_eq!(found(&mut db, "Pinned"), 1, "joined at SET");
    db.execute_cypher("MATCH (n:Goal) REMOVE n:Pinned")
        .expect("label removed");
    assert_eq!(found(&mut db, "Pinned"), 0, "left at REMOVE");
}

/// A vector index on an auxiliary label ranks only its carriers; a read at
/// a timestamp when a node still carried the label finds it there.
#[test]
fn a_vector_index_covers_auxiliary_carriers_now_and_then() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE LABEL Pinned AUXILIARY")
        .expect("auxiliary");
    db.execute_cypher("CREATE VECTOR INDEX pinned_emb ON :Pinned(emb) OPTIONS {metric: \"l2\"}")
        .expect("vector index");
    let commit = |db: &mut Database, statement: &str| {
        let tx = db.begin_transaction().expect("begin");
        db.execute_in_transaction(tx, statement, None)
            .expect(statement);
        db.commit_transaction(tx)
            .expect("commit")
            .commit_ts
            .as_raw()
    };
    commit(
        &mut db,
        "CREATE (:Item:Pinned {name: 'a', emb: [1.0, 0.0]}), \
         (:Item {name: 'near', emb: [1.0, 0.0]}), \
         (:Item:Pinned {name: 'far', emb: [-5.0, 0.0]})",
    );
    let top = |at: Option<u64>| {
        let at = at.map_or(String::new(), |ts| format!(" AS OF TIMESTAMP {ts}"));
        format!(
            "MATCH (n:Pinned) WITH n, vector_distance(n.emb, [1.0, 0.0]) AS d \
             ORDER BY d LIMIT 2 RETURN n.name AS name{at}"
        )
    };
    let names = |db: &mut Database, query: &str| -> Vec<String> {
        db.execute_cypher(query)
            .unwrap_or_else(|e| panic!("{query}: {e}"))
            .iter()
            .map(|row| match row.get("name") {
                Some(Value::String(s)) => s.clone(),
                other => panic!("name {other:?}"),
            })
            .collect()
    };
    let plan = db.explain_cypher(&top(None)).expect("explain");
    assert!(plan.contains("HnswScan"), "the index path:\n{plan}");
    assert_eq!(names(&mut db, &top(None)), ["a", "far"]);
    let before = commit(&mut db, "MATCH (n:Item {name: 'near'}) SET n:Pinned");
    assert_eq!(names(&mut db, &top(None)), ["a", "near"], "joined at SET");
    commit(&mut db, "MATCH (n:Item {name: 'a'}) REMOVE n:Pinned");
    assert_eq!(
        names(&mut db, &top(None)),
        ["near", "far"],
        "left at REMOVE"
    );
    let mut then = names(&mut db, &top(Some(before)));
    then.sort();
    assert_eq!(then, ["a", "near"], "the carriers at the earlier timestamp");
}

/// A label added in a transaction is visible to that transaction's own
/// lookups through the index and to nobody else, and a rollback leaves no
/// entry behind.
#[test]
fn a_label_change_is_private_until_commit_and_gone_on_rollback() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE LABEL Policy AUXILIARY")
        .expect("auxiliary");
    db.execute_cypher("CREATE INDEX p_name ON :Policy(name)")
        .expect("index");
    db.execute_cypher("CREATE (:Rec {name: 'a'})")
        .expect("node");
    let lookup = "MATCH (n:Policy {name: 'a'}) RETURN n";

    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(tx, "MATCH (n:Rec) SET n:Policy", None)
        .expect("label added");
    let own = db
        .execute_in_transaction(tx, lookup, None)
        .expect("own lookup");
    assert_eq!(own.len(), 1, "the transaction sees its own label");
    assert_eq!(count(&mut db, lookup), 0, "others do not");
    db.rollback_transaction(tx).expect("rollback");

    assert_eq!(count(&mut db, lookup), 0);
    assert_eq!(entry_count(&db, "p_name"), 0, "no entry survives");
    assert_index_agrees(&db, "p_name");
}

/// Two writers adding a uniquely indexed auxiliary label at once to nodes
/// holding one value: exactly one commits, and one carrier remains.
#[test]
fn concurrent_label_adds_leave_one_unique_holder() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE LABEL Policy AUXILIARY")
        .expect("auxiliary");
    db.execute_cypher("CREATE UNIQUE INDEX p_code ON :Policy(code)")
        .expect("unique index");
    let writers: Vec<String> = (0..6)
        .map(|w| {
            db.execute_cypher(&format!("CREATE (:Rec {{code: 'same', w: {w}}})"))
                .expect("node");
            format!("MATCH (n:Rec {{w: {w}}}) SET n:Policy")
        })
        .collect();

    let outcomes = race(&db, &writers);

    assert_eq!(
        outcomes.iter().filter(|o| o.is_ok()).count(),
        1,
        "{outcomes:?}"
    );
    for err in outcomes.iter().filter_map(|o| o.as_ref().err()) {
        assert!(
            err.contains("unique constraint violated") || err.contains("conflict"),
            "{err}"
        );
    }
    assert_eq!(count(&mut db, "MATCH (n:Policy) RETURN n"), 1);
    assert_eq!(entry_count(&db, "p_code"), 1);
}

/// A declaration of `IMPLIES` racing writers of its type: whatever order
/// they commit in, a declaration that succeeds leaves no node of the type
/// without the implied label.
#[test]
fn an_implies_declaration_racing_writers_leaves_no_node_without_it() {
    for round in 0..8 {
        let (mut db, _dir) = open_db();
        db.execute_cypher("CREATE NODE TYPE Base").expect("base");
        let mut statements: Vec<String> = (0..4)
            .map(|w| format!("CREATE (:Late {{w: {w}}})"))
            .collect();
        statements.insert(
            round % 5,
            "CREATE NODE TYPE Late IMPLIES (Base)".to_string(),
        );

        let outcomes = race(&db, &statements);

        let declared = outcomes[round % 5].is_ok();
        let without = count(
            &mut db,
            "MATCH (n:Late) WHERE NOT 'Base' IN labels(n) RETURN n",
        );
        if declared {
            assert_eq!(without, 0, "round {round}: {outcomes:?}");
        }
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

/// Nodes written while an index on an implied label is being built carry
/// into it like the stored ones: the published index holds both.
#[test]
fn a_build_on_an_implied_label_takes_writers_during_it() {
    let (mut db, _dir) = open_db();
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Goal IMPLIES (Thing) WITH (title: STRING)",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    db.execute_cypher("CREATE (:Goal {title: 'stored'})")
        .expect("stored");
    let older = db.begin_transaction().expect("older");
    std::thread::scope(|s| {
        let created = s.spawn(|| run(&db, "CREATE INDEX thing_title ON :Thing(title)"));
        await_held(&db, "thing_title");
        run(&db, "CREATE (:Goal {title: 'during'})").expect("writer during the build");
        db.rollback_transaction(older).expect("end the older one");
        created.join().expect("join").expect("index built");
    });
    assert_eq!(entry_count(&db, "thing_title"), 2);
    for title in ["stored", "during"] {
        assert_eq!(
            count(
                &mut db,
                &format!("MATCH (n:Thing {{title: '{title}'}}) RETURN n")
            ),
            1,
            "{title}"
        );
    }
    assert_index_agrees(&db, "thing_title");
}

/// While a unique index on an implied label is being built, a stored node
/// carrying the label implied still holds its value: a writer giving the
/// value to another carrier is refused before the build reaches the holder.
#[test]
fn a_unique_build_on_an_implied_label_sees_uncovered_holders() {
    let (mut db, _dir) = open_db();
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Goal IMPLIES (Thing) WITH (title: STRING)",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    db.execute_cypher("CREATE (:Goal {title: 'stored'})")
        .expect("stored holder");
    let older = db.begin_transaction().expect("older");
    let refused_during = std::thread::scope(|s| {
        let created = s.spawn(|| run(&db, "CREATE UNIQUE INDEX thing_title ON :Thing(title)"));
        await_held(&db, "thing_title");
        let outcome = run(&db, "CREATE (:Thing {title: 'stored'})");
        db.rollback_transaction(older).expect("end the older one");
        created.join().expect("join").expect("index built");
        outcome
    });
    let err = refused_during.expect_err("the stored Goal holds the value");
    assert!(err.contains("unique constraint violated"), "{err}");
    assert_eq!(
        count(&mut db, "MATCH (n:Thing {title: 'stored'}) RETURN n"),
        1
    );
}

/// A presence constraint on an auxiliary label still being proved binds the
/// carriers written meanwhile: one without the property is refused, one with
/// it is admitted, and the proof over the stored carriers activates it.
#[test]
fn a_constraint_being_proved_on_an_auxiliary_label_binds_new_carriers() {
    use coordinode_core::schema::definition::ConstraintState;
    use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
    use coordinode_query::index::GenerationId;

    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE LABEL Policy AUXILIARY")
        .expect("auxiliary");
    db.execute_cypher("CREATE (:Rec:Policy {owner: 'a'})")
        .expect("stored carrier");
    // One seat, taken by a build waiting for a transaction opened before
    // it: the proof queues behind it.
    db.set_index_build_config(coordinode_query::index::IndexBuildConfig {
        max_running: 1,
        statement_wait: Duration::from_millis(200),
        ..db.index_build_config()
    });
    let older = db.begin_transaction().expect("older");
    db.execute_cypher("CREATE CONSTRAINT org_name FOR (o:Org) REQUIRE o.name IS UNIQUE")
        .expect("the blocking build");
    let rows = db
        .execute_cypher("CREATE CONSTRAINT policy_owner FOR (n:Policy) REQUIRE n.owner IS NOT NULL")
        .expect("published");
    let Some(Value::Int(operation)) = rows[0].get("operation").cloned() else {
        panic!("a validating constraint names its operation: {rows:?}");
    };
    let state = |db: &Database| {
        LocalSchemaStore::new(db.engine())
            .load_label("Policy")
            .expect("load schema")
            .and_then(|s| s.constraint("policy_owner").map(|c| c.state))
    };
    assert_eq!(state(&db), Some(ConstraintState::Validating));

    let err = refused(&mut db, "CREATE (:Doc:Policy)");
    assert!(err.contains("owner"), "{err}");
    db.execute_cypher("CREATE (:Doc:Policy {owner: 'b'})")
        .expect("a carrier with the property");

    db.rollback_transaction(older).expect("end the older one");
    let operation = GenerationId::from_raw(u64::try_from(operation).expect("operation"));
    db.index_build(operation, Duration::from_secs(30))
        .expect("wait");
    assert_eq!(state(&db), Some(ConstraintState::Active));
    assert_eq!(count(&mut db, "MATCH (n:Policy) RETURN n"), 2);
}

/// Kinds, implied labels and the entries of their indexes survive a
/// restart: the reopened database refuses what it refused before and finds
/// what it found.
#[test]
fn label_kinds_and_their_coverage_survive_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open");
        for ddl in [
            "CREATE NODE TYPE Thing WITH (title: STRING)",
            "CREATE NODE TYPE Goal IMPLIES (Thing) WITH (title: STRING)",
            "CREATE LABEL Policy AUXILIARY",
            "CREATE INDEX thing_title ON :Thing(title)",
            "CREATE INDEX policy_title ON :Policy(title)",
        ] {
            db.execute_cypher(ddl)
                .unwrap_or_else(|e| panic!("{ddl}: {e}"));
        }
        db.execute_cypher("CREATE (:Goal:Policy {title: 'g'}), (:Thing {title: 'g'})")
            .expect("nodes");
    }
    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(count(&mut db, "MATCH (n:Thing {title: 'g'}) RETURN n"), 2);
    assert_eq!(count(&mut db, "MATCH (n:Policy {title: 'g'}) RETURN n"), 1);
    let err = refused(&mut db, "CREATE (:Policy {title: 'p'})");
    assert!(err.contains("no type label"), "{err}");
    db.execute_cypher("CREATE (:Goal {title: 'h'})")
        .expect("goal after reopen");
    assert_eq!(
        labels(&mut db, "MATCH (n:Goal {title: 'h'}) RETURN labels(n) AS l"),
        ["Goal", "Thing"]
    );
    for name in ["thing_title", "policy_title"] {
        assert_index_agrees(&db, name);
    }
}

/// MERGE creates a node of a type with the labels the type implies, and a
/// MERGE on an implied label finds that node through its index instead of
/// creating another.
#[test]
fn merge_creates_and_finds_a_node_by_its_implied_label() {
    let mut db = goals();
    db.execute_cypher("CREATE INDEX thing_title ON :Thing(title)")
        .expect("index");
    db.execute_cypher("MERGE (:Goal {title: 'm'})")
        .expect("merge creates");
    assert_eq!(
        labels(&mut db, "MATCH (n:Goal {title: 'm'}) RETURN labels(n) AS l"),
        ["Goal", "Thing"]
    );
    db.execute_cypher("MERGE (:Thing {title: 'm'})")
        .expect("merge matches the goal");
    assert_eq!(count(&mut db, "MATCH (n:Thing {title: 'm'}) RETURN n"), 1);
    db.execute_cypher("MERGE (:Activity:Goal {title: 'n'})")
        .expect("merge with an auxiliary label");
    assert_eq!(
        labels(&mut db, "MATCH (n:Goal {title: 'n'}) RETURN labels(n) AS l"),
        ["Goal", "Activity", "Thing"]
    );
    let err = refused(&mut db, "MERGE (:Extra:Goal {title: 'o'})");
    assert!(err.contains("two type labels"), "{err}");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 2);
}

/// Thing, Goal implying it, Policy auxiliary, and an index on each of the
/// two labels a Goal can carry besides its own, all FLEXIBLE so a type can
/// add an expiry; `goal` is the Goal schema to declare.
fn expiring_goals(db: &mut Database, goal: LabelSchema) {
    use coordinode_core::schema::definition::SchemaMode;
    let mut thing = LabelSchema::new_node_id("Thing");
    thing.set_mode(SchemaMode::Flexible);
    thing.add_property(PropertyDef::new("title", PropertyType::String));
    db.create_label_schema(thing).expect("thing");
    db.create_label_schema(goal).expect("goal with an expiry");
    for ddl in [
        "CREATE LABEL Policy AUXILIARY",
        "CREATE INDEX thing_title ON :Thing(title)",
        "CREATE INDEX policy_created ON :Policy(created_at)",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
}

fn goal_expiring(scope: coordinode_core::schema::computed::TtlScope) -> LabelSchema {
    use coordinode_core::schema::computed::ComputedSpec;
    use coordinode_core::schema::definition::SchemaMode;
    let mut goal = LabelSchema::new_node_id("Goal");
    goal.set_mode(SchemaMode::Flexible);
    goal.set_implies(vec!["Thing".to_string()]);
    goal.add_property(PropertyDef::new("title", PropertyType::String));
    goal.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    goal.add_property(PropertyDef::computed(
        "_ttl",
        ComputedSpec::Ttl {
            duration_secs: 3600,
            anchor_field: "created_at".into(),
            scope,
            target_field: None,
        },
    ));
    goal
}

/// Now, and two hours ago, in epoch microseconds.
fn now_and_expired() -> (i64, i64) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros() as i64;
    (now, now - 2 * 3600 * 1_000_000)
}

/// A node its type expires leaves the index of every label it carries,
/// implied and auxiliary, while an unexpired carrier stays in both.
#[test]
fn an_expired_node_leaves_the_index_of_every_label_it_carries() {
    use coordinode_core::schema::computed::TtlScope;
    let (mut db, _dir) = open_db();
    expiring_goals(&mut db, goal_expiring(TtlScope::Node));
    let (now, expired) = now_and_expired();
    db.execute_cypher(&format!(
        "CREATE (:Goal:Policy {{title: 'old', created_at: {expired}}}), \
         (:Goal:Policy {{title: 'new', created_at: {now}}})"
    ))
    .expect("nodes");
    assert_eq!(entry_count(&db, "thing_title"), 2);
    assert_eq!(entry_count(&db, "policy_created"), 2);

    let reaped = db.reap_expired().expect("reap");

    assert_eq!(reaped.nodes_deleted, 1, "{reaped:?}");
    assert_eq!(entry_count(&db, "thing_title"), 1, "implied label");
    assert_eq!(entry_count(&db, "policy_created"), 1, "auxiliary label");
    assert_eq!(count(&mut db, "MATCH (n:Thing {title: 'old'}) RETURN n"), 0);
    assert_eq!(count(&mut db, "MATCH (n:Thing {title: 'new'}) RETURN n"), 1);
    for name in ["thing_title", "policy_created"] {
        assert_index_agrees(&db, name);
    }
}

/// An expiry that takes a field away keeps it on a node where an index on
/// one of the node's other labels holds it, so no entry is left pointing at
/// a value the node no longer has; elsewhere it takes the field.
#[test]
fn an_expiry_keeps_a_field_an_auxiliary_label_indexes() {
    use coordinode_core::schema::computed::TtlScope;
    let (mut db, _dir) = open_db();
    expiring_goals(&mut db, goal_expiring(TtlScope::Field));
    let (_, expired) = now_and_expired();
    db.execute_cypher(&format!(
        "CREATE (:Goal:Policy {{title: 'kept', created_at: {expired}}}), \
         (:Goal {{title: 'cut', created_at: {expired}}})"
    ))
    .expect("nodes");

    let reaped = db.reap_expired().expect("reap");

    assert_eq!(reaped.fields_removed, 1, "{reaped:?}");
    assert!(
        reaped
            .errors
            .iter()
            .any(|e| e.contains("indexed property 'created_at'")),
        "{reaped:?}"
    );
    assert_eq!(
        count(
            &mut db,
            "MATCH (n:Goal) WHERE n.created_at IS NULL RETURN n.title"
        ),
        1,
        "the field left the unindexed node only"
    );
    assert_eq!(entry_count(&db, "policy_created"), 1);
    assert_index_agrees(&db, "policy_created");
}

/// A declaration that fails leaves the guarantees in force as they were: a
/// constraint refused over a stored carrier binds nobody, the constraint
/// already active still holds, and a refused IMPLIES adds no label.
#[test]
fn a_failed_declaration_keeps_the_old_guarantees() {
    let mut db = goals();
    db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE")
        .expect("active constraint");
    db.execute_cypher("CREATE (:Goal:Policy {title: 'p'}), (:Item {k: 1})")
        .expect("nodes");

    refused(
        &mut db,
        "CREATE CONSTRAINT policy_owner FOR (n:Policy) REQUIRE n.owner IS NOT NULL",
    );
    refused(&mut db, "CREATE NODE TYPE Item IMPLIES (Thing)");

    db.execute_cypher("CREATE (:Goal:Policy {title: 'q'})")
        .expect("the refused constraint binds nobody");
    let err = refused(&mut db, "CREATE (:Thing {title: 'p'})");
    assert!(err.contains("unique"), "the active one still holds: {err}");
    db.execute_cypher("CREATE (:Item {k: 2})")
        .expect("the item type is unchanged");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 2);
    assert_eq!(
        count(
            &mut db,
            "MATCH (n:Item) WHERE 'Thing' IN labels(n) RETURN n"
        ),
        0
    );
}

/// Kinds, implied labels and their indexes survive a restore in every
/// format: the restored database finds each carrier through its indexes and
/// refuses what the source refused. A cypher dump carries data only, so its
/// target declares the labels and indexes first.
#[test]
fn label_kinds_and_their_coverage_survive_a_restore_in_every_format() {
    use coordinode_embed::backup::restore::RestoreOptions;
    use coordinode_embed::backup::{BackupFormat, export};
    const DDL: [&str; 5] = [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Goal IMPLIES (Thing) WITH (title: STRING)",
        "CREATE LABEL Policy AUXILIARY",
        "CREATE INDEX thing_title ON :Thing(title)",
        "CREATE INDEX policy_title ON :Policy(title)",
    ];
    for format in [
        BackupFormat::Json,
        BackupFormat::Cypher,
        BackupFormat::Binary,
    ] {
        let (mut db1, _dir1) = open_db();
        for ddl in DDL {
            db1.execute_cypher(ddl)
                .unwrap_or_else(|e| panic!("{ddl}: {e}"));
        }
        db1.execute_cypher("CREATE (:Goal:Policy {title: 'g'}), (:Thing {title: 'g'})")
            .expect("nodes");
        let interner = db1.interner().expect("dictionary");
        let snapshot = db1.engine().snapshot();
        let mut dump = Vec::new();
        match format {
            BackupFormat::Json => {
                export::export_json(db1.engine(), &interner, 1, &snapshot, &mut dump)
            }
            BackupFormat::Cypher => {
                export::export_cypher(db1.engine(), &interner, 1, &snapshot, &mut dump)
            }
            _ => export::export_binary(db1.engine(), &interner, 1, &snapshot, &mut dump),
        }
        .expect("export");

        let (mut db2, _dir2) = open_db();
        if format == BackupFormat::Cypher {
            for ddl in DDL {
                db2.execute_cypher(ddl)
                    .unwrap_or_else(|e| panic!("{ddl}: {e}"));
            }
        }
        db2.restore(format, &dump, &RestoreOptions::default())
            .unwrap_or_else(|e| panic!("{format:?}: restore: {e}"));

        assert_eq!(
            count(&mut db2, "MATCH (n:Thing {title: 'g'}) RETURN n"),
            2,
            "{format:?}"
        );
        assert_eq!(
            count(&mut db2, "MATCH (n:Policy {title: 'g'}) RETURN n"),
            1,
            "{format:?}"
        );
        let err = refused(&mut db2, "CREATE (:Policy {title: 'p'})");
        assert!(err.contains("no type label"), "{format:?}: {err}");
        db2.execute_cypher("CREATE (:Goal {title: 'h'})")
            .unwrap_or_else(|e| panic!("{format:?}: {e}"));
        assert_eq!(
            labels(
                &mut db2,
                "MATCH (n:Goal {title: 'h'}) RETURN labels(n) AS l"
            ),
            ["Goal", "Thing"],
            "{format:?}"
        );
        for name in ["thing_title", "policy_title"] {
            assert_index_agrees(&db2, name);
        }
    }
}

/// A database whose nodes carry one label each behaves as before: nothing
/// is declared, nothing is refused, and the stored labels are unchanged.
#[test]
fn a_single_label_database_is_unchanged() {
    let mut db = Database::open_in_memory().expect("open");
    db.execute_cypher("CREATE (:User {name: 'a'}), (:Post {t: 1})")
        .expect("create");
    db.execute_cypher("MATCH (n:User) SET n.name = 'b'")
        .expect("set");
    assert_eq!(
        labels(&mut db, "MATCH (n:User) RETURN labels(n) AS l"),
        ["User"]
    );
    assert_eq!(count(&mut db, "MATCH (n:Post) RETURN n"), 1);
}
