//! `MATCH (n:L) RETURN count(n)` answered from the per-label node counter
//! the write path keeps on the same transaction as the nodes: the plan names
//! the counter, and the answer equals counting the nodes in every case the
//! counter has to follow, including the cases where it must not be used.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// The single integer a counting query returns.
fn count(db: &mut Database, query: &str) -> i64 {
    let rows = db.execute_cypher(query).expect("count");
    assert_eq!(rows.len(), 1, "{query}: {rows:?}");
    let value = rows[0].values().next().cloned();
    match value {
        Some(Value::Int(n)) => n,
        other => panic!("{query}: not a count: {other:?}"),
    }
}

/// What a scan of the nodes says, by a query the counter cannot answer.
fn scanned(db: &mut Database, label: &str) -> i64 {
    count(
        db,
        &format!("MATCH (n:{label}) WHERE n IS NOT NULL RETURN count(n) AS n"),
    )
}

/// The plan answers from the counter, not by scanning the label.
#[test]
fn the_plan_reads_the_counter() {
    let (db, _dir) = open_db();
    for query in [
        "MATCH (n:Project) RETURN count(n) AS n",
        "MATCH (n:Project) RETURN count(*) AS n",
    ] {
        let plan = db.explain_cypher(query).expect("explain");
        assert!(plan.contains("NodeCountFromCounter"), "{query}: {plan}");
        assert!(!plan.contains("NodeScan"), "{query}: {plan}");
    }
    // Anything the counter does not count keeps the scan.
    for query in [
        "MATCH (n:Project) WHERE n.x = 1 RETURN count(n) AS n",
        "MATCH (n:Project {x: 1}) RETURN count(n) AS n",
        "MATCH (n:Project:Active) RETURN count(n) AS n",
        "MATCH (n) RETURN count(n) AS n",
        "MATCH (n:Project) RETURN n.x, count(n) AS n",
        "MATCH (n:Project) RETURN count(n.x) AS n",
    ] {
        let plan = db.explain_cypher(query).expect("explain");
        assert!(!plan.contains("NodeCountFromCounter"), "{query}: {plan}");
    }
}

/// The counted answer follows creates, deletes, label changes and nodes
/// with several labels, and always equals counting the nodes.
#[test]
fn the_count_follows_every_change() {
    let (mut db, _dir) = open_db();
    assert_eq!(count(&mut db, "MATCH (n:Project) RETURN count(n) AS n"), 0);

    db.execute_cypher("UNWIND range(1, 7) AS i CREATE (:Project {i: i})")
        .expect("projects");
    db.execute_cypher("CREATE (:Project:Archived {i: 100}), (:Other)")
        .expect("others");
    db.execute_cypher("MATCH (n:Project) WHERE n.i <= 2 DELETE n")
        .expect("delete two");
    db.execute_cypher("MATCH (n:Other) SET n:Project")
        .expect("label added");
    db.execute_cypher("MATCH (n:Project) WHERE n.i = 3 REMOVE n:Project")
        .expect("label removed");

    for label in ["Project", "Archived", "Other"] {
        let counted = count(&mut db, &format!("MATCH (n:{label}) RETURN count(n) AS n"));
        assert_eq!(counted, scanned(&mut db, label), "{label}");
    }
    assert_eq!(count(&mut db, "MATCH (n:Project) RETURN count(*) AS c"), 6);
}

/// Inside a transaction the count includes the transaction's own writes not
/// yet committed, as the scan does, and leaves nothing behind on rollback.
#[test]
fn the_count_sees_its_own_transaction() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Project), (:Project)")
        .expect("committed");
    let txn = db.begin_transaction();
    db.execute_in_transaction(txn, "CREATE (:Project), (:Project), (:Project)", None)
        .expect("own writes");
    db.execute_in_transaction(txn, "MATCH (n:Project) WITH n LIMIT 1 DELETE n", None)
        .expect("own delete");
    let rows = db
        .execute_in_transaction(txn, "MATCH (n:Project) RETURN count(n) AS n", None)
        .expect("count in the transaction");
    assert_eq!(rows[0].get("n"), Some(&Value::Int(4)));
    db.rollback_transaction(txn).expect("rollback");
    assert_eq!(count(&mut db, "MATCH (n:Project) RETURN count(n) AS n"), 2);
}

/// A count at a past timestamp is the number of nodes then, by either path:
/// the counter is versioned with the nodes, and a read naming its timestamp
/// in the query counts by scan.
#[test]
fn a_past_count_is_the_count_of_then() {
    use coordinode_core::txn::read_concern::{ReadConcern, ReadConcernLevel};

    let (mut db, _dir) = open_db();
    db.execute_cypher("UNWIND range(1, 3) AS i CREATE (:Project {i: i})")
        .expect("three");
    let then = db.engine().snapshot();
    db.execute_cypher("UNWIND range(1, 2) AS i CREATE (:Project {i: i})")
        .expect("two more");

    let at_then = db
        .execute_cypher_with_read_concern(
            "MATCH (n:Project) RETURN count(n) AS n",
            ReadConcern {
                level: ReadConcernLevel::Snapshot,
                after_index: None,
                at_timestamp: Some(then),
            },
        )
        .expect("count then");
    assert_eq!(at_then[0].get("n"), Some(&Value::Int(3)));
    let named = count(
        &mut db,
        &format!("MATCH (n:Project) RETURN count(n) AS n AS OF TIMESTAMP {then}"),
    );
    assert_eq!(named, 3);
    assert_eq!(count(&mut db, "MATCH (n:Project) RETURN count(n) AS n"), 5);
}

/// A plan cached while no temporal label existed still counts by scan once
/// one exists: the executor asks the catalog again.
#[test]
fn a_cached_plan_rechecks_the_catalog() {
    let (mut db, _dir) = open_db();
    let query = "MATCH (n:Project) RETURN count(n) AS n";
    db.execute_cypher("CREATE (:Project)").expect("plain node");
    assert_eq!(count(&mut db, query), 1);

    db.execute_cypher("CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT)")
        .expect("temporal label");
    db.execute_cypher("CREATE (:Emp:Project {name: 'a', valid_from: 0})")
        .expect("temporal node also labelled Project");
    db.execute_cypher("MATCH (n:Emp) SET n.name = 'b'")
        .expect("a second version");
    assert_eq!(count(&mut db, query), 2);
}

/// A node one statement of a transaction deleted is gone for the later
/// statements of that transaction, whether they scan the label or look the
/// node up by a property, exactly as it is gone once the transaction commits.
#[test]
fn a_node_deleted_earlier_in_the_transaction_is_gone_for_it() {
    let (db, _dir) = open_db();
    db.execute_cypher_shared(
        "CREATE (:Project {i: 1}), (:Project {i: 2})",
        None,
        None,
        None,
        None,
    )
    .expect("committed");
    let txn = db.begin_transaction();
    db.execute_in_transaction(txn, "MATCH (n:Project {i: 1}) DELETE n", None)
        .expect("own delete");
    let ids = |rows: Vec<coordinode_query::executor::Row>| -> Vec<Value> {
        rows.into_iter()
            .filter_map(|row| row.get("i").cloned())
            .collect()
    };
    let scanned = db
        .execute_in_transaction(txn, "MATCH (n:Project) RETURN n.i AS i", None)
        .expect("scan");
    assert_eq!(ids(scanned), [Value::Int(2)]);
    let looked_up = db
        .execute_in_transaction(txn, "MATCH (n:Project {i: 1}) RETURN n.i AS i", None)
        .expect("lookup");
    assert_eq!(ids(looked_up), Vec::<Value>::new());
    db.commit_transaction(txn).expect("commit");
    let after = db
        .execute_cypher_shared("MATCH (n:Project) RETURN n.i AS i", None, None, None, None)
        .expect("after commit");
    assert_eq!(ids(after.rows), [Value::Int(2)]);
}

/// A counter row counts stored node rows, and a temporal node keeps one row
/// per version, under every label it carries. While a temporal label exists
/// the count is the scan's, so versions are never counted as nodes.
#[test]
fn a_temporal_label_keeps_the_scan_answer() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE NODE TYPE Emp TEMPORAL WITH (name: STRING, valid_from: INT)")
        .expect("temporal label");
    db.execute_cypher("CREATE (:Emp:Project {name: 'a', valid_from: 0})")
        .expect("temporal node also labelled Project");
    db.execute_cypher("MATCH (n:Emp) SET n.name = 'b'")
        .expect("a second version");
    db.execute_cypher("CREATE (:Project)").expect("plain node");

    assert_eq!(count(&mut db, "MATCH (n:Project) RETURN count(n) AS n"), 2);
    assert_eq!(count(&mut db, "MATCH (n:Emp) RETURN count(n) AS n"), 1);
    assert_eq!(
        count(&mut db, "MATCH (n:Project) RETURN count(n) AS n"),
        scanned(&mut db, "Project")
    );
}
