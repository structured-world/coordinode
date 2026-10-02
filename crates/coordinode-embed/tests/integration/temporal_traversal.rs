//! Traversals into a temporal node find it whether or not the pattern names
//! its label: a temporal node is stored under per-version keys, so a target
//! the pattern leaves unlabelled has to be resolved the same way a labelled
//! one is, never read as absent.

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// A store with one plain source, one temporal target, and one edge between
/// them; `temporal_edge` makes the edge type temporal as well.
fn seeded(temporal_edge: bool) -> Database {
    let mut db = Database::open_in_memory().expect("open");
    for statement in [
        "CREATE NODE TYPE T TEMPORAL",
        "ALTER LABEL T SET SCHEMA FLEXIBLE",
        "CREATE (:Src {k: 1})",
        "CREATE (:T {name: 'one', valid_from: 1})",
    ] {
        db.execute_cypher(statement).expect(statement);
    }
    if temporal_edge {
        db.execute_cypher("CREATE EDGE TYPE E TEMPORAL WITH (valid_from: INT, valid_to: INT)")
            .expect("temporal edge type");
        db.execute_cypher("MATCH (s:Src), (t:T) CREATE (s)-[:E {valid_from: 5}]->(t)")
            .expect("edge");
    } else {
        db.execute_cypher("MATCH (s:Src), (t:T) CREATE (s)-[:E]->(t)")
            .expect("edge");
    }
    db
}

/// How many rows `query` returns.
fn rows(db: &mut Database, query: &str) -> usize {
    db.execute_cypher(query).expect(query).len()
}

fn assert_every_form_finds_the_edge(temporal_edge: bool) {
    let mut db = seeded(temporal_edge);
    for query in [
        "MATCH (s:Src)-[r:E]->(t:T) RETURN t.name",
        "MATCH (s:Src)-[r:E]->(t) RETURN t.name",
        "MATCH ()-[r:E]->() RETURN r",
        "MATCH (s)-[r]->(t) RETURN type(r)",
        "MATCH (s:Src)-[:E*1..2]->(t) RETURN t.name",
        "MATCH (t)<-[:E]-(s:Src) RETURN t.name",
        "MATCH (t {name: 'one'})<-[:E]-(s) RETURN s.k",
    ] {
        assert_eq!(
            rows(&mut db, query),
            1,
            "{query} (temporal edge: {temporal_edge})"
        );
    }
    let named = db
        .execute_cypher("MATCH (s:Src)-[r:E]->(t) RETURN t.name AS name")
        .expect("unlabelled target");
    assert_eq!(
        named[0].get("name"),
        Some(&Value::String("one".into())),
        "the unlabelled target reads the temporal node's current version"
    );
}

/// A plain edge into a temporal node is found by every pattern form.
#[test]
fn an_unlabelled_temporal_target_is_found() {
    assert_every_form_finds_the_edge(false);
}

/// The same with a temporal edge type.
#[test]
fn an_unlabelled_temporal_target_is_found_over_a_temporal_edge() {
    assert_every_form_finds_the_edge(true);
}
