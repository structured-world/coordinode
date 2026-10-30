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

/// A uniqueness on a label a type comes to imply spans the type's stored
/// nodes: one holding a value a carrier of the label holds refuses the
/// change, which leaves the implication, the label's members and the
/// uniqueness as they were; once the value is free the change goes through
/// and the uniqueness binds the type's nodes.
#[test]
fn a_gained_uniqueness_over_a_held_value_refuses_the_change() {
    let mut db = things();
    db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE")
        .expect("constraint");
    db.execute_cypher("CREATE (:Thing {title: 'x'}), (:Task {title: 'x'}), (:Task {title: 'y'})")
        .expect("nodes");
    let err = refused(&mut db, "ALTER NODE TYPE Task SET IMPLIES (Thing)");
    assert!(err.contains('x') || err.contains("unique"), "{err}");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 1, "unchanged");
    db.execute_cypher("CREATE (:Thing {title: 'y'})")
        .expect("the old guarantee covers things alone");
    db.execute_cypher("MATCH (n:Task {title: 'x'}) SET n.title = 'z'")
        .expect("free x");
    db.execute_cypher("MATCH (n:Thing {title: 'y'}) DELETE n")
        .expect("free y");
    db.execute_cypher("ALTER NODE TYPE Task SET IMPLIES (Thing)")
        .expect("no value held twice now");
    assert_eq!(count(&mut db, "MATCH (n:Thing) RETURN n"), 3);
    let err = refused(&mut db, "CREATE (:Task {title: 'x'})");
    assert!(err.contains('x') || err.contains("unique"), "{err}");
    let err = refused(&mut db, "CREATE (:Thing {title: 'z'})");
    assert!(err.contains('z') || err.contains("unique"), "{err}");
}

/// Whether no index keeps parts apart any more: every concluded change is
/// folded into the indexes' own entries.
fn folded(db: &Database) -> bool {
    use coordinode_modality::IndexStore as _;
    coordinode_modality::LocalIndexStore::new(db.engine())
        .list_definitions()
        .expect("definitions")
        .iter()
        .all(|def| def.selection.is_empty())
}

/// Wait up to five seconds for the parts of the changes made to be folded.
fn await_folded(db: &Database) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !folded(db) {
        assert!(std::time::Instant::now() < deadline, "parts left unfolded");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Once a change is put in force, what it kept apart is folded back into
/// the lookup's and the index's own entries: each scope is one again, the
/// label's counter answers the count and agrees with a scan, and an index
/// lookup finds the same nodes; both ways.
#[test]
fn a_concluded_change_is_folded_into_one_scope() {
    let mut db = things();
    db.execute_cypher("CREATE INDEX thing_title ON :Thing(title)")
        .expect("index");
    db.execute_cypher("CREATE (:Task {title: 'x'}), (:Task {title: 'y'}), (:Thing {title: 'x'})")
        .expect("nodes");
    let counted = |db: &mut Database| {
        let rows = db
            .execute_cypher("MATCH (n:Thing) RETURN count(n) AS c")
            .expect("count");
        rows[0].get("c").cloned()
    };
    for (alter, members, through_index) in [
        ("ALTER NODE TYPE Task SET IMPLIES (Thing)", 3, 2),
        ("ALTER NODE TYPE Task SET IMPLIES ()", 1, 1),
    ] {
        db.execute_cypher(alter).expect(alter);
        await_folded(&db);
        assert_eq!(
            counted(&mut db),
            Some(Value::Int(members)),
            "{alter}: counter"
        );
        assert_eq!(
            count(&mut db, "MATCH (n:Thing) WHERE n.title + '' <> '' RETURN n"),
            members as usize,
            "{alter}: scan"
        );
        assert_eq!(
            count(&mut db, "MATCH (n:Thing {title: 'x'}) RETURN n"),
            through_index,
            "{alter}: index"
        );
    }
}

/// One schema package moving a value's holder out of a label and another
/// into it: Chore stops implying Thing while Task starts, a Chore and a Task
/// holding the same title under a uniqueness on Thing. Each interpretation
/// holds the title once, so the change goes through although both holders
/// stand in the index while it is certified; then the uniqueness binds the
/// Task and no longer the Chore.
#[test]
fn a_coupled_change_moves_a_unique_value_between_holders() {
    use coordinode_embed::{Intent, LabelDeclaration, SchemaPackage};
    let mut db = things();
    db.execute_cypher("CREATE NODE TYPE Chore IMPLIES (Thing) WITH (title: STRING)")
        .expect("chore");
    db.execute_cypher("CREATE CONSTRAINT thing_title FOR (n:Thing) REQUIRE n.title IS UNIQUE")
        .expect("constraint");
    db.execute_cypher("CREATE (:Chore {title: 'v'}), (:Task {title: 'v'})")
        .expect("nodes");
    let declared = |name: &str, implies: &[&str]| {
        let current = db
            .label_schemas()
            .expect("schemas")
            .into_iter()
            .find(|s| s.name == name)
            .expect("declared");
        let revision = current.schema_revision;
        let mut schema = current;
        schema.set_implies(implies.iter().map(|l| (*l).to_string()).collect());
        LabelDeclaration {
            schema,
            intent: Intent::Change {
                expected_revision: revision,
            },
        }
    };
    let package = SchemaPackage {
        labels: vec![declared("Chore", &[]), declared("Task", &["Thing"])],
        constraints: Vec::new(),
        edge_types: Vec::new(),
    };
    let published = db.publish_schema(package).expect("publish");
    db.await_schema(&published, std::time::Duration::from_secs(30))
        .expect("await");
    let titles = |db: &mut Database, query: &str| {
        db.execute_cypher(query)
            .expect(query)
            .iter()
            .filter_map(|row| match row.get("l") {
                Some(Value::Array(labels)) => labels.first().cloned(),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        titles(
            &mut db,
            "MATCH (n:Thing {title: 'v'}) RETURN labels(n) AS l"
        ),
        [Value::String("Task".into())],
        "the Task holds the title under Thing now"
    );
    db.execute_cypher("CREATE (:Chore {title: 'v'})")
        .expect("a Chore is no Thing any more");
    let err = refused(&mut db, "CREATE (:Thing {title: 'v'})");
    assert!(err.contains('v') || err.contains("unique"), "{err}");
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
