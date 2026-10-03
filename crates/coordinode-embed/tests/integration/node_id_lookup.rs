//! Integration tests: a MATCH pinned to one node id (`n = $id`,
//! `id(n) = $id`) reads that node alone instead of scanning every node.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// The shard an in-memory database keeps its nodes in.
const SHARD: u16 = 1;

fn id_of(db: &mut Database, query: &str) -> i64 {
    let rows = db.execute_cypher(query).expect("create");
    match rows[0].get("n") {
        Some(Value::Int(id)) => *id,
        other => panic!("node id: {other:?}"),
    }
}

fn names(db: &mut Database, query: &str, id: i64) -> Vec<String> {
    let params: HashMap<String, Value> = [("id".to_string(), Value::Int(id))].into();
    db.execute_cypher_with_params(query, params)
        .expect("query")
        .iter()
        .map(|r| match r.get("name") {
            Some(Value::String(s)) => s.clone(),
            other => panic!("name: {other:?}"),
        })
        .collect()
}

/// Put a version record that does not decode under node `raw_id`: any
/// statement that reads it fails, so one that succeeds did not read it.
fn plant_undecodable_node(db: &Database, raw_id: u64) {
    use coordinode_core::graph::node::NodeId;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::partition::Partition;
    let mut key = LocalNodeStore.version_prefix(SHARD, NodeId::from_raw(raw_id));
    key.extend_from_slice(&0u64.to_be_bytes());
    db.engine()
        .put(Partition::Node, &key, &[0xC1])
        .expect("plant");
}

/// Both spellings find the node, honour its label and inline filters, and a
/// missing, negative or other-label id finds nothing.
#[test]
fn a_node_id_pin_finds_exactly_that_node() {
    let mut db = Database::open_in_memory().expect("open");
    let ada = id_of(&mut db, "CREATE (n:User {name: 'ada'}) RETURN n");
    id_of(&mut db, "CREATE (n:User {name: 'bob'}) RETURN n");
    let acme = id_of(&mut db, "CREATE (n:Org {name: 'acme'}) RETURN n");

    for query in [
        "MATCH (n:User) WHERE id(n) = $id RETURN n.name AS name",
        "MATCH (n:User) WHERE n = $id RETURN n.name AS name",
        "MATCH (n) WHERE $id = id(n) RETURN n.name AS name",
        "MATCH (n:User {name: 'ada'}) WHERE id(n) = $id RETURN n.name AS name",
    ] {
        assert_eq!(names(&mut db, query, ada), ["ada"], "{query}");
    }
    let by_id = "MATCH (n:User) WHERE id(n) = $id RETURN n.name AS name";
    assert!(names(&mut db, by_id, acme).is_empty(), "other label");
    assert!(names(&mut db, by_id, 987_654).is_empty(), "missing id");
    assert!(names(&mut db, by_id, -1).is_empty(), "negative id");
    assert!(
        names(
            &mut db,
            "MATCH (n:User {name: 'bob'}) WHERE id(n) = $id RETURN n.name AS name",
            ada
        )
        .is_empty(),
        "inline filter still applies"
    );
    // A disjunction does not pin the scan, and still answers correctly.
    let mut either = names(
        &mut db,
        "MATCH (n:User) WHERE id(n) = $id OR n.name = 'bob' RETURN n.name AS name",
        ada,
    );
    either.sort();
    assert_eq!(either, ["ada", "bob"]);
}

/// The pinned read touches no other node: an undecodable record of another
/// node fails a scan, and leaves the pinned read unaffected.
#[test]
fn a_node_id_pin_reads_no_other_node() {
    let mut db = Database::open_in_memory().expect("open");
    let ada = id_of(&mut db, "CREATE (n:User {name: 'ada'}) RETURN n");
    plant_undecodable_node(&db, 999_999);

    assert!(
        db.execute_cypher("MATCH (n:User) RETURN n.name AS name")
            .is_err(),
        "a scan reads the planted record"
    );
    assert_eq!(
        names(
            &mut db,
            "MATCH (n:User) WHERE id(n) = $id RETURN n.name AS name",
            ada
        ),
        ["ada"]
    );
}

/// On a temporal label the pinned node resolves to its state valid now, or
/// at the instant `temporal_active_at` names, and a deleted node to nothing.
#[test]
fn a_node_id_pin_resolves_a_temporal_timeline() {
    let mut db = Database::open_in_memory().expect("open");
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    let emp = id_of(
        &mut db,
        "CREATE (n:Emp {name: 'old', valid_from: 100}) RETURN n",
    );
    db.execute_cypher("MATCH (n:Emp) SET n.name = 'new'")
        .expect("new version");
    let gone = id_of(
        &mut db,
        "CREATE (n:Emp {name: 'gone', valid_from: 100}) RETURN n",
    );
    db.execute_cypher("MATCH (n:Emp {name: 'gone'}) DELETE n")
        .expect("delete");

    let now = "MATCH (n:Emp) WHERE id(n) = $id RETURN n.name AS name";
    assert_eq!(names(&mut db, now, emp), ["new"]);
    assert_eq!(
        names(
            &mut db,
            "MATCH (n:Emp) WHERE id(n) = $id AND temporal_active_at(n, 150) \
             RETURN n.name AS name",
            emp
        ),
        ["old"]
    );
    assert!(names(&mut db, now, gone).is_empty(), "deleted");
}
