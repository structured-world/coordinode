//! A node written with several labels carries all of them: MERGE creates it
//! with every label of its pattern and matches on all of them, and
//! `labels()` returns them all.

#![allow(clippy::expect_used)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

fn count(db: &mut Database, query: &str) -> usize {
    db.execute_cypher(query).expect(query).len()
}

fn labels_of(db: &mut Database, query: &str) -> Value {
    let rows = db.execute_cypher(query).expect(query);
    assert_eq!(rows.len(), 1, "{query}");
    rows[0].get("l").cloned().expect("labels column")
}

fn strings(names: &[&str]) -> Value {
    Value::Array(names.iter().map(|n| Value::String((*n).into())).collect())
}

/// MERGE creates the node with every label of its pattern, as CREATE does.
#[test]
fn merge_creates_every_label_of_its_pattern() {
    let mut db = Database::open_in_memory().expect("open");
    db.execute_cypher("MERGE (c:A:B {k: 1}) ON CREATE SET c.v = 1")
        .expect("merge A:B");
    db.execute_cypher("MERGE (c:C:D {k: 1})")
        .expect("merge C:D");
    db.execute_cypher("CREATE (c:E:F {k: 1})")
        .expect("create E:F");

    for (label, expected) in [("A", 1), ("B", 1), ("C", 1), ("D", 1), ("E", 1), ("F", 1)] {
        assert_eq!(
            count(&mut db, &format!("MATCH (n:{label}) RETURN n")),
            expected,
            "nodes labelled {label}"
        );
    }
    assert_eq!(
        labels_of(&mut db, "MATCH (n:B) RETURN labels(n) AS l"),
        strings(&["A", "B"])
    );
    assert_eq!(
        labels_of(&mut db, "MATCH (n:F) RETURN labels(n) AS l"),
        strings(&["E", "F"])
    );
    assert_eq!(
        count(&mut db, "MATCH (n:B {k: 1}) WHERE n.v = 1 RETURN n"),
        1,
        "ON CREATE SET applied to the node MERGE created"
    );
}

/// MERGE matches on every label of its pattern: a second MERGE with the
/// same labels finds the node, one with a label the node lacks creates a
/// new one.
#[test]
fn merge_matches_on_every_label() {
    let mut db = Database::open_in_memory().expect("open");
    db.execute_cypher("MERGE (c:A:B {k: 1})").expect("first");
    db.execute_cypher("MERGE (c:A:B {k: 1})").expect("again");
    assert_eq!(
        count(&mut db, "MATCH (n:A) RETURN n"),
        1,
        "matched, not duplicated"
    );
    db.execute_cypher("MERGE (c:A:Z {k: 1})")
        .expect("other label");
    assert_eq!(
        count(&mut db, "MATCH (n:A) RETURN n"),
        2,
        "A:Z is a different node"
    );
    assert_eq!(count(&mut db, "MATCH (n:Z) RETURN n"), 1);
}
