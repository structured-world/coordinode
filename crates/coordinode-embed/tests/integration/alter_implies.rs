//! `ALTER NODE TYPE ... SET IMPLIES (...)` end to end: the new implication is
//! proved over the stored carriers of the type before it is put in force,
//! refused with the node that breaks it, and checked as a declaration is.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// `Thing` and `Task`, both STRICT with a title; `Activity` auxiliary.
fn things() -> Database {
    let mut db = Database::open_in_memory().expect("open");
    for ddl in [
        "CREATE NODE TYPE Thing WITH (title: STRING)",
        "CREATE NODE TYPE Task WITH (title: STRING, due: TIMESTAMP)",
        "CREATE LABEL Activity AUXILIARY",
    ] {
        db.execute_cypher(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e}"));
    }
    db
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

/// The implication as the statement's `implies` column reports it.
fn implies_of(column: Option<&Value>) -> Vec<String> {
    match column {
        Some(Value::Array(labels)) => labels
            .iter()
            .map(|l| match l {
                Value::String(s) => s.clone(),
                other => panic!("label {other:?}"),
            })
            .collect(),
        other => panic!("implies {other:?}"),
    }
}

/// A type without stored nodes comes to imply a label at once, and its nodes
/// written afterwards are found under it.
#[test]
fn a_type_without_nodes_comes_to_imply_a_label() {
    let mut db = things();
    let rows = db
        .execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("alter");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(implies_of(rows[0].get("implies")), ["Thing"]);
    assert_eq!(rows[0].get("changed"), Some(&Value::Bool(true)));
    db.execute_cypher("CREATE (:Task {title: 't'})")
        .expect("node");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 1);
}

/// A scan of a label a type comes to imply finds the type's stored nodes,
/// beside the label's own and those carrying it explicitly, each once; and
/// stops finding them when the implication is dropped, while the others
/// stay. Counted and paged the same way.
#[test]
fn a_label_scan_follows_the_implication_both_ways() {
    let mut db = things();
    db.execute_cypher(
        "CREATE (:Task {title: 'a'}), (:Task:Activity {title: 'b'}), (:Thing {title: 'c'})",
    )
    .expect("nodes");
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("gain");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 3);
    assert_eq!(
        count(
            &mut db,
            "MATCH (n:Thing) RETURN n.title AS t ORDER BY t LIMIT 2"
        ),
        2
    );
    let rows = db
        .execute_cypher("MATCH (n:Thing) RETURN count(n) AS c")
        .expect("count");
    assert_eq!(rows[0].get("c"), Some(&Value::Int(3)));
    db.execute_cypher("CREATE (:Task {title: 'd'})")
        .expect("a task written after the change");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 4);

    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES ()")
        .expect("loss");
    assert_eq!(
        count(&mut db, "MATCH (n:Thing) RETURN n"),
        1,
        "the thing alone"
    );
    assert_eq!(
        count(&mut db, "MATCH (n:Task) RETURN n"),
        3,
        "tasks stay tasks"
    );
    assert_eq!(
        count(&mut db, "MATCH (n:Activity) RETURN n"),
        1,
        "an explicit label stays"
    );
    let rows = db
        .execute_cypher("MATCH (n:Thing) RETURN count(n) AS c")
        .expect("count");
    assert_eq!(rows[0].get("c"), Some(&Value::Int(1)));
}

/// The stored nodes of a type are found under a label it comes to imply,
/// by a scan of the label and through an index on it, once the change
/// answers; and leave it when the implication is dropped.
#[test]
fn stored_carriers_follow_the_implication() {
    let mut db = things();
    db.execute_cypher("CREATE INDEX thing_title ON :Thing(title)")
        .expect("index");
    db.execute_cypher("CREATE (:Task {title: 'x'}), (:Task {title: 'y'}), (:Thing {title: 'x'})")
        .expect("nodes");
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("alter over stored tasks");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 3, "scanned");
    assert_eq!(
        count(&mut db, "MATCH (n:Thing {title: 'x'}) RETURN n"),
        2,
        "through the index"
    );
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES ()")
        .expect("implication dropped");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 1, "scanned");
    assert_eq!(
        count(&mut db, "MATCH (n:Thing {title: 'x'}) RETURN n"),
        1,
        "through the index"
    );
}

/// Setting the implication a type already has changes nothing.
#[test]
fn the_implication_in_force_is_unchanged() {
    let mut db = things();
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("alter");
    let rows = db
        .execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing, Thing)")
        .expect("again");
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(implies_of(rows[0].get("implies")), ["Thing"]);
    assert_eq!(rows[0].get("changed"), Some(&Value::Bool(false)));
}

/// What the new implication may say is checked as a declaration is: the
/// implied label's properties must be declared by the type with their
/// types, nothing implies an auxiliary label or itself, the implication
/// forms no cycle, and only a declared type label is altered.
#[test]
fn the_new_implication_is_checked_as_a_declaration() {
    let mut db = things();
    db.execute_cypher("CREATE NODE TYPE Note WITH (body: STRING)")
        .expect("type without title");
    let err = refused(&mut db, "ALTER NODE TYPE Note SET IMPLIES (Thing)");
    assert!(err.contains("title"), "{err}");
    let err = refused(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Activity)");
    assert!(err.contains("auxiliary"), "{err}");
    let err = refused(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Task)");
    assert!(err.contains("itself"), "{err}");
    db.execute_cypher("CREATE NODE TYPE Twin WITH (title: STRING)")
        .expect("a type declaring what Thing does");
    db.execute_cypher("ALTER NODE TYPE Twin SET IMPLIES (Thing)")
        .expect("Twin implies Thing");
    let err = refused(&mut db, "ALTER NODE TYPE Thing SET IMPLIES (Twin)");
    assert!(err.contains("cycle") || err.contains("itself"), "{err}");
    let err = refused(&mut db, "ALTER NODE TYPE Activity SET IMPLIES (Thing)");
    assert!(err.contains("auxiliary"), "{err}");
    let err = refused(&mut db, "ALTER NODE TYPE Nowhere SET IMPLIES (Thing)");
    assert!(err.contains("not declared"), "{err}");
}

/// A stored carrier of the type that breaks an explicit constraint of a
/// label it would come to carry refuses the change: the node is named, the
/// implication in force stays, and the constraint keeps holding.
#[test]
fn a_carrier_breaking_a_gained_constraint_refuses_the_change() {
    let mut db = things();
    db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS NOT NULL")
        .expect("constraint");
    db.execute_cypher("CREATE (:Task)")
        .expect("a task without a title");
    let err = refused(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Thing)");
    assert!(err.contains("Thing") && err.contains("title"), "{err}");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 0);
    // The implication in force is still none, and the label stays free to
    // change once the carrier is fixed.
    db.execute_cypher("CREATE (:Task)")
        .expect("a task without a title is still a task");
    let rows = db
        .execute_cypher("ALTER NODE TYPE Task SET IMPLIES ()")
        .expect("nothing implied");
    assert_eq!(rows[0].get("changed"), Some(&Value::Bool(false)));
}
