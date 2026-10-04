//! End-to-end integration tests for B-tree index DDL via Cypher.
//!
//! Tests the full pipeline: Cypher string → parser → planner → executor →
//! IndexRegistry. Covers CREATE INDEX, DROP INDEX, EXPLAIN output, and
//! index-backed MATCH/WHERE queries.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

// ── CREATE INDEX ──────────────────────────────────────────────────────

#[test]
fn create_index_via_cypher_succeeds() {
    // CREATE INDEX must succeed and return a result row with index metadata.
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher("CREATE INDEX user_name_idx ON :User(name)")
        .expect("CREATE INDEX should succeed");

    assert_eq!(rows.len(), 1, "CREATE INDEX should return one row");
    let row = &rows[0];

    // Response must identify the index name.
    let index_val = row.get("index");
    assert!(
        index_val.is_some(),
        "response row must contain 'index' field, got keys: {:?}",
        row.keys().collect::<Vec<_>>()
    );
    assert_eq!(
        index_val,
        Some(&coordinode_core::graph::types::Value::String(
            "user_name_idx".into()
        ))
    );
}

#[test]
fn create_unique_index_via_cypher() {
    // CREATE UNIQUE INDEX must parse and register successfully.
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher("CREATE UNIQUE INDEX u_email_idx ON :User(email)")
        .expect("CREATE UNIQUE INDEX should succeed");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("index"),
        Some(&coordinode_core::graph::types::Value::String(
            "u_email_idx".into()
        ))
    );
}

#[test]
fn create_sparse_index_via_cypher() {
    // CREATE SPARSE INDEX must parse and register successfully.
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher("CREATE SPARSE INDEX s_age_idx ON :User(age)")
        .expect("CREATE SPARSE INDEX should succeed");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("index"),
        Some(&coordinode_core::graph::types::Value::String(
            "s_age_idx".into()
        ))
    );
}

// ── DROP INDEX ────────────────────────────────────────────────────────

#[test]
fn drop_index_via_cypher_succeeds() {
    // CREATE then DROP: the registry must be empty for that name after DROP.
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE INDEX to_drop_idx ON :User(name)")
        .expect("CREATE INDEX");

    let rows = db
        .execute_cypher("DROP INDEX to_drop_idx")
        .expect("DROP INDEX should succeed");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("dropped"),
        Some(&coordinode_core::graph::types::Value::Bool(true))
    );
}

#[test]
fn drop_nonexistent_index_returns_error() {
    let (mut db, _dir) = open_db();

    let result = db.execute_cypher("DROP INDEX does_not_exist");
    assert!(result.is_err(), "DROP INDEX on nonexistent index must fail");

    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("does_not_exist") || msg.contains("not found"),
        "error must mention missing index, got: {msg}"
    );
}

// ── EXPLAIN regression: IndexScan after CREATE INDEX ─────────────────

/// Writes that find their node through the index change exactly that node:
/// a guarded heartbeat SET, a REMOVE and a DETACH DELETE, each leaving the
/// other nodes of the label as they were.
#[test]
fn writes_through_the_index_change_only_their_node() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX agent_session ON :Agent(session_id)")
        .expect("CREATE INDEX");
    for (sid, seen) in [("s1", 10), ("s2", 10), ("s3", 10)] {
        db.execute_cypher(&format!(
            "CREATE (:Agent {{session_id: '{sid}', fingerprint: 'f', last_seen_at_ms: {seen}}})"
        ))
        .expect("create");
    }
    let mut params = std::collections::HashMap::new();
    params.insert("sid".to_string(), Value::String("s2".into()));
    params.insert("fp".to_string(), Value::String("f".into()));
    params.insert("now".to_string(), Value::Int(20));
    let rows = db
        .execute_cypher_with_params(
            "MATCH (a:Agent {session_id: $sid}) WHERE a.fingerprint = $fp AND a.last_seen_at_ms <= $now \
             SET a.last_seen_at_ms = $now RETURN a.session_id AS sid",
            params.clone(),
        )
        .expect("heartbeat");
    assert_eq!(rows.len(), 1);
    // The guard holds back a stale heartbeat.
    params.insert("now".to_string(), Value::Int(15));
    let stale = db
        .execute_cypher_with_params(
            "MATCH (a:Agent {session_id: $sid}) WHERE a.fingerprint = $fp AND a.last_seen_at_ms <= $now \
             SET a.last_seen_at_ms = $now RETURN a.session_id AS sid",
            params,
        )
        .expect("stale heartbeat");
    assert!(stale.is_empty(), "the guard refused it: {stale:?}");
    db.execute_cypher("MATCH (a:Agent {session_id: 's1'}) REMOVE a.fingerprint")
        .expect("remove");
    db.execute_cypher("MATCH (a:Agent {session_id: 's3'}) DETACH DELETE a")
        .expect("delete");

    let rows = db
        .execute_cypher(
            "MATCH (a:Agent) RETURN a.session_id AS sid, a.last_seen_at_ms AS seen, \
             a.fingerprint AS fp ORDER BY sid",
        )
        .expect("read back");
    let got: Vec<(Value, Value, Value)> = rows
        .iter()
        .map(|r| (r["sid"].clone(), r["seen"].clone(), r["fp"].clone()))
        .collect();
    assert_eq!(
        got,
        vec![
            (Value::String("s1".into()), Value::Int(10), Value::Null),
            (
                Value::String("s2".into()),
                Value::Int(20),
                Value::String("f".into())
            ),
        ]
    );
}

/// A SET on a temporal label found through the index opens the next version
/// of exactly that node.
#[test]
fn a_temporal_set_through_the_index_opens_the_next_version() {
    use coordinode_core::graph::types::Value;
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE NODE TYPE Emp TEMPORAL WITH (code: STRING, name: STRING, valid_from: INT, valid_to: INT)",
    )
    .expect("temporal label");
    db.execute_cypher("CREATE INDEX emp_code ON :Emp(code)")
        .expect("CREATE INDEX");
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros() as i64
        - 1_000_000;
    for code in ["a", "b"] {
        db.execute_cypher(&format!(
            "CREATE (:Emp {{code: '{code}', name: 'old', valid_from: {t}}})"
        ))
        .expect("create");
    }
    let explain = db
        .explain_cypher("MATCH (n:Emp {code: 'a'}) SET n.name = 'new'")
        .expect("EXPLAIN");
    assert!(explain.contains("IndexScan"), "{explain}");
    db.execute_cypher("MATCH (n:Emp {code: 'a'}) SET n.name = 'new'")
        .expect("set");
    let rows = db
        .execute_cypher("MATCH (n:Emp) RETURN n.code AS code, n.name AS name ORDER BY code")
        .expect("read back");
    let got: Vec<(Value, Value)> = rows
        .iter()
        .map(|r| (r["code"].clone(), r["name"].clone()))
        .collect();
    assert_eq!(
        got,
        vec![
            (Value::String("a".into()), Value::String("new".into())),
            (Value::String("b".into()), Value::String("old".into())),
        ]
    );
}

/// An equality on one property is looked up only in an index holding every
/// node by exactly that property: a compound index keyed by more columns,
/// or a partial one holding only the nodes its filter admits, would miss
/// nodes. Reads and writes over such properties find every node.
#[test]
fn a_compound_or_partial_index_never_answers_a_one_property_lookup() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE (p.first, p.last) IS NODE KEY",
    )
    .expect("compound key");
    db.execute_cypher("CREATE INDEX u_active ON :U(email) WHERE n.status = 'active'")
        .expect("partial index");
    db.execute_cypher("CREATE (:Person {first: 'Ada', last: 'Byron'})")
        .expect("person");
    db.execute_cypher("CREATE (:U {email: 'c@x', status: 'gone'})")
        .expect("inactive user");

    for (query, rows) in [
        ("MATCH (p:Person {last: 'Byron'}) RETURN p.first AS v", 1),
        (
            "MATCH (p:Person) WHERE p.last = 'Byron' RETURN p.first AS v",
            1,
        ),
        ("MATCH (u:U {email: 'c@x'}) RETURN u.status AS v", 1),
        (
            "MATCH (u:U {email: 'c@x'}) SET u.seen = 1 RETURN u.status AS v",
            1,
        ),
    ] {
        let explain = db.explain_cypher(query).expect("EXPLAIN");
        assert!(
            !explain.contains("IndexScan"),
            "{query}\nno index holds every node by this property alone:\n{explain}"
        );
        let got = db.execute_cypher(query).expect("query").len();
        assert_eq!(got, rows, "{query}");
    }
    db.execute_cypher("MATCH (u:U {email: 'c@x'}) DETACH DELETE u")
        .expect("delete");
    assert!(
        db.execute_cypher("MATCH (u:U) RETURN u")
            .expect("scan")
            .is_empty(),
        "the delete found its node"
    );
}

/// A write whose MATCH finds its node by an indexed property looks it up in
/// the index, as the same MATCH does for a read: a SET, REMOVE or DELETE
/// over a label scan costs every node of the label on every statement. The
/// first shape is a heartbeat: an inline key plus residual conditions.
#[test]
fn a_write_finds_its_node_through_the_index() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX agent_session ON :Agent(session_id)")
        .expect("CREATE INDEX");
    for query in [
        "MATCH (a:Agent {session_id: $sid}) WHERE a.fingerprint = $fp AND a.last_seen_at_ms <= $now \
         SET a.last_seen_at_ms = $now RETURN a.session_id",
        "MATCH (a:Agent) WHERE a.session_id = 's1' SET a.seen = 1",
        "MATCH (a:Agent {session_id: 's1'}) REMOVE a.seen",
        "MATCH (a:Agent {session_id: 's1'}) DELETE a",
        "MATCH (a:Agent {session_id: 's1'}) DETACH DELETE a",
        "MATCH (a:Agent {session_id: 's1'}) SET a += {seen: 1}",
        "MATCH (a:Agent {session_id: 's1'}) FOREACH (x IN [1] | SET a.seen = x)",
    ] {
        let explain = db.explain_cypher(query).expect("EXPLAIN");
        assert!(
            explain.contains("IndexScan(a:Agent ON agent_session(session_id))")
                && !explain.contains("NodeScan"),
            "{query}\nmust find its node through the index, got:\n{explain}"
        );
    }
}

#[test]
fn explain_shows_index_scan_after_create_index_via_cypher() {
    // Regression test: after CREATE INDEX, EXPLAIN for a matching
    // WHERE clause must show IndexScan, not NodeScan.
    let (mut db, _dir) = open_db();

    // Create the index.
    db.execute_cypher("CREATE INDEX user_name_idx ON :User(name)")
        .expect("CREATE INDEX");

    // EXPLAIN the query that should use the index.
    let explain = db
        .explain_cypher("MATCH (n:User) WHERE n.name = 'Alice' RETURN n")
        .expect("EXPLAIN should succeed");

    assert!(
        explain.contains("IndexScan"),
        "EXPLAIN must contain 'IndexScan' after CREATE INDEX, got:\n{explain}"
    );
    assert!(
        !explain.contains("NodeScan"),
        "EXPLAIN must NOT contain 'NodeScan' (should be rewritten to IndexScan), got:\n{explain}"
    );
    assert!(
        explain.contains("user_name_idx"),
        "EXPLAIN must reference the index name, got:\n{explain}"
    );
}

#[test]
fn explain_shows_node_scan_without_index() {
    // Without CREATE INDEX, EXPLAIN must show NodeScan (no index rewrite).
    let (db, _dir) = open_db();

    let explain = db
        .explain_cypher("MATCH (n:User) WHERE n.name = 'Alice' RETURN n")
        .expect("EXPLAIN should succeed");

    assert!(
        explain.contains("NodeScan"),
        "EXPLAIN without index must contain 'NodeScan', got:\n{explain}"
    );
    assert!(
        !explain.contains("IndexScan"),
        "EXPLAIN without index must NOT contain 'IndexScan', got:\n{explain}"
    );
}

// ── Index-backed MATCH/WHERE queries ─────────────────────────────────

#[test]
fn match_where_uses_index_and_returns_correct_node() {
    // After CREATE INDEX, MATCH (n:User) WHERE n.name = 'Bob' must return Bob.
    // This test verifies execution correctness, not just plan shape.
    let (mut db, _dir) = open_db();

    // Insert test data.
    db.execute_cypher("CREATE (:User {name: 'Alice', age: 30})")
        .expect("insert Alice");
    db.execute_cypher("CREATE (:User {name: 'Bob', age: 25})")
        .expect("insert Bob");
    db.execute_cypher("CREATE (:User {name: 'Charlie', age: 35})")
        .expect("insert Charlie");

    // Create index AFTER inserting data — backfill must pick up all three nodes.
    db.execute_cypher("CREATE INDEX user_name_idx ON :User(name)")
        .expect("CREATE INDEX");

    // Query using the index.
    let rows = db
        .execute_cypher("MATCH (n:User) WHERE n.name = 'Bob' RETURN n.name, n.age")
        .expect("MATCH WHERE should succeed");

    assert_eq!(rows.len(), 1, "should return exactly one row for Bob");
    assert_eq!(
        rows[0].get("n.name"),
        Some(&coordinode_core::graph::types::Value::String("Bob".into())),
        "returned node should be Bob"
    );
    assert_eq!(
        rows[0].get("n.age"),
        Some(&coordinode_core::graph::types::Value::Int(25)),
        "Bob's age should be 25"
    );
}

#[test]
fn index_backfills_nodes_created_before_create_index() {
    // Nodes inserted before CREATE INDEX must be backfilled and findable via index.
    let (mut db, _dir) = open_db();

    // Insert BEFORE creating index.
    db.execute_cypher("CREATE (:Product {sku: 'P001', price: 9.99})")
        .expect("insert P001");
    db.execute_cypher("CREATE (:Product {sku: 'P002', price: 19.99})")
        .expect("insert P002");

    // Create index — should backfill P001 and P002.
    let result = db
        .execute_cypher("CREATE INDEX product_sku_idx ON :Product(sku)")
        .expect("CREATE INDEX");

    let nodes_indexed = match result[0].get("nodes_indexed") {
        Some(coordinode_core::graph::types::Value::Int(n)) => *n,
        other => panic!("expected nodes_indexed as Int, got {other:?}"),
    };
    assert!(
        nodes_indexed >= 2,
        "backfill should index at least 2 Product nodes, got {nodes_indexed}"
    );

    // Both nodes should be findable.
    let rows_p001 = db
        .execute_cypher("MATCH (p:Product) WHERE p.sku = 'P001' RETURN p.price")
        .expect("MATCH P001");
    assert_eq!(rows_p001.len(), 1, "P001 should be findable via index");

    let rows_p002 = db
        .execute_cypher("MATCH (p:Product) WHERE p.sku = 'P002' RETURN p.price")
        .expect("MATCH P002");
    assert_eq!(rows_p002.len(), 1, "P002 should be findable via index");
}

#[test]
fn unique_index_rejects_duplicate_insert() {
    // CREATE UNIQUE INDEX must reject duplicate property values on subsequent inserts.
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :User(email)")
        .expect("CREATE UNIQUE INDEX");

    db.execute_cypher("CREATE (:User {email: 'alice@example.com'})")
        .expect("first insert should succeed");

    let result = db.execute_cypher("CREATE (:User {email: 'alice@example.com'})");
    assert!(
        result.is_err(),
        "inserting duplicate email with UNIQUE INDEX must fail"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.to_lowercase().contains("unique") || msg.to_lowercase().contains("constraint"),
        "error must mention unique constraint violation, got: {msg}"
    );
}

/// A unique index is the constraint of the same name, owning it: DROP INDEX
/// refuses it and leaves uniqueness enforced, DROP CONSTRAINT lifts it and
/// takes the index with it.
#[test]
fn a_unique_index_is_dropped_through_its_constraint() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE UNIQUE INDEX u_code ON :Item(code)")
        .expect("CREATE UNIQUE INDEX");
    db.execute_cypher("CREATE (:Item {code: 'X1'})")
        .expect("first insert");

    let err = db
        .execute_cypher("DROP INDEX u_code")
        .expect_err("the index belongs to its constraint");
    assert!(
        err.to_string()
            .contains("index 'u_code' belongs to constraint 'u_code'"),
        "{err}"
    );
    assert!(
        db.execute_cypher("CREATE (:Item {code: 'X1'})").is_err(),
        "a refused DROP INDEX leaves uniqueness enforced"
    );

    db.execute_cypher("DROP CONSTRAINT u_code")
        .expect("DROP CONSTRAINT");
    db.execute_cypher("CREATE (:Item {code: 'X1'})")
        .expect("uniqueness lifted with the constraint");
    db.execute_cypher("CREATE INDEX u_code ON :Item(code)")
        .expect("the index went with the constraint, its name is free");
}

// ── DETACH DELETE index cleanup regression ────────────────────────────

#[test]
fn detach_delete_cleans_unique_btree_index() {
    // Regression: DETACH DELETE must remove B-tree index entries for the
    // deleted node. Without this cleanup, re-creating a node with the same
    // unique property value fails with "unique constraint violated" even
    // though the original node no longer exists.
    //
    // Root cause: execute_delete() in runner.rs notified vector/text index
    // registries but never called btree_index_registry.on_node_deleted().
    let (mut db, _dir) = open_db();

    // Create a unique index.
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :User(email)")
        .expect("CREATE UNIQUE INDEX");

    // Insert a node that is covered by the unique index.
    db.execute_cypher("CREATE (:User {email: 'alice@example.com', name: 'Alice'})")
        .expect("initial CREATE");

    // Delete the node with DETACH DELETE.
    db.execute_cypher("MATCH (n:User {email: 'alice@example.com'}) DETACH DELETE n")
        .expect("DETACH DELETE");

    // The node must be gone from MATCH results.
    let after_delete = db
        .execute_cypher("MATCH (n:User {email: 'alice@example.com'}) RETURN n.email")
        .expect("MATCH after delete");
    assert_eq!(
        after_delete.len(),
        0,
        "node must not exist after DETACH DELETE"
    );

    // Re-create a node with the SAME unique property value.
    // Without proper index cleanup this fails with "unique constraint violated".
    let result = db.execute_cypher("CREATE (:User {email: 'alice@example.com', name: 'Alice2'})");
    assert!(
        result.is_ok(),
        "re-CREATE with same unique value after DETACH DELETE must succeed; \
         unique index entry must have been cleaned up on delete. \
         Got error: {:?}",
        result.err()
    );
}

#[test]
fn plain_delete_cleans_unique_btree_index() {
    // Same regression as detach_delete_cleans_unique_btree_index but for
    // plain DELETE on a disconnected node (no edges to DETACH).
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE UNIQUE INDEX u_sku ON :Product(sku)")
        .expect("CREATE UNIQUE INDEX");

    db.execute_cypher("CREATE (:Product {sku: 'P-999'})")
        .expect("initial CREATE");

    // Plain DELETE (node has no edges, so this is valid without DETACH).
    db.execute_cypher("MATCH (n:Product {sku: 'P-999'}) DELETE n")
        .expect("DELETE");

    let result = db.execute_cypher("CREATE (:Product {sku: 'P-999'})");
    assert!(
        result.is_ok(),
        "re-CREATE with same unique value after DELETE must succeed; \
         got error: {:?}",
        result.err()
    );
}

// ── REMOVE property B-tree index cleanup regression ───────────────────

#[test]
fn remove_property_cleans_unique_btree_index() {
    // Regression: REMOVE n.prop must remove the B-tree index entry for that
    // property. Without cleanup, another node cannot be created with the
    // same value because the stale unique entry still exists.
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :User(email)")
        .expect("CREATE UNIQUE INDEX");

    db.execute_cypher("CREATE (:User {email: 'alice@example.com', name: 'Alice'})")
        .expect("initial CREATE");

    // Remove the indexed property from the node.
    db.execute_cypher("MATCH (n:User {email: 'alice@example.com'}) REMOVE n.email")
        .expect("REMOVE email");

    // Now another node should be creatable with the same email value.
    let result = db.execute_cypher("CREATE (:User {email: 'alice@example.com', name: 'Alice2'})");
    assert!(
        result.is_ok(),
        "CREATE with previously-removed unique value must succeed; \
         stale index entry must be removed on REMOVE. Got error: {:?}",
        result.err()
    );
}

// ── SET property B-tree index update regression ───────────────────────

#[test]
fn set_property_updates_unique_btree_index() {
    // Regression: SET must update the B-tree index entries for a node.
    // Without this update:
    //   1. The old value stays in the index → another node cannot be
    //      created with the old value (stale unique constraint).
    //   2. The new value is never added to the index → IndexScan won't
    //      find the node by its new value.
    //
    // Root cause: execute_update() in runner.rs notified vector/text
    // registries but never touched btree_index_registry.
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :User(email)")
        .expect("CREATE UNIQUE INDEX");

    db.execute_cypher("CREATE (:User {email: 'alice@example.com'})")
        .expect("initial CREATE");

    // Update the indexed property to a new value.
    db.execute_cypher(
        "MATCH (n:User {email: 'alice@example.com'}) SET n.email = 'alice2@example.com'",
    )
    .expect("SET email");

    // 1. The new value must be findable via index.
    let rows = db
        .execute_cypher("MATCH (n:User) WHERE n.email = 'alice2@example.com' RETURN n.email")
        .expect("MATCH by new value");
    assert_eq!(
        rows.len(),
        1,
        "node must be findable by new indexed value; B-tree index must be updated on SET"
    );

    // 2. The old value must NOT block a new CREATE (stale unique entry must be gone).
    let result = db.execute_cypher("CREATE (:User {email: 'alice@example.com'})");
    assert!(
        result.is_ok(),
        "CREATE with old value after SET must succeed; \
         stale index entry must be removed on SET. Got error: {:?}",
        result.err()
    );
}

#[test]
fn set_property_unique_conflict_still_enforced() {
    // After SET, the unique constraint on the NEW value must still be enforced.
    // Setting n.email to a value already taken by another node must fail.
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :User(email)")
        .expect("CREATE UNIQUE INDEX");

    db.execute_cypher("CREATE (:User {email: 'alice@example.com'})")
        .expect("create alice");
    db.execute_cypher("CREATE (:User {email: 'bob@example.com'})")
        .expect("create bob");

    // Try to SET bob's email to alice's email — must fail (unique violation).
    let result = db.execute_cypher(
        "MATCH (n:User {email: 'bob@example.com'}) SET n.email = 'alice@example.com'",
    );
    assert!(
        result.is_err(),
        "SET to an already-taken unique value must fail"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.to_lowercase().contains("unique") || msg.to_lowercase().contains("constraint"),
        "error must mention unique constraint, got: {msg}"
    );
}

// ── Entries in the statement transaction ─────────────────────────────

use coordinode_core::graph::types::Value;

fn count(db: &mut Database, query: &str, params: &[(&str, Value)]) -> i64 {
    let params = params
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect();
    let rows = db
        .execute_cypher_with_params(query, params)
        .expect("count query");
    match rows[0].values().next() {
        Some(Value::Int(n)) => *n,
        other => panic!("expected a count, got {other:?}"),
    }
}

/// An equality answered by the index finds exactly the nodes holding the
/// value. The old encoding wrote a string's NUL unescaped, so the entries of
/// "a" were a prefix of those of "a\0:x" and a lookup of one found both.
#[test]
fn an_index_lookup_finds_only_the_value_asked_for() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX u_name ON :U(name)")
        .expect("index");
    for name in ["a", "a\0:x"] {
        db.execute_cypher_with_params(
            "CREATE (:U {name: $n})",
            [("n".to_string(), Value::String(name.into()))].into(),
        )
        .expect("create");
    }
    let query = "MATCH (u:U) WHERE u.name = $n RETURN count(u)";
    assert_eq!(
        count(&mut db, query, &[("n", Value::String("a".into()))]),
        1
    );
    assert_eq!(
        count(&mut db, query, &[("n", Value::String("a\0:x".into()))]),
        1
    );
}

/// A value with no key is not indexed, so it neither collides with another
/// under a unique index nor hides from an equality.
#[test]
fn values_without_a_key_do_not_collide_under_a_unique_index() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_meta ON :U(meta)")
        .expect("index");
    db.execute_cypher("CREATE (:U {meta: {a: 1}})")
        .expect("first map");
    db.execute_cypher("CREATE (:U {meta: {b: 2}})")
        .expect("a second, different map is no duplicate");
}

/// Nothing a failed statement staged survives it: the value its first
/// write claimed is free once the statement fails.
#[test]
fn a_failed_statement_leaves_no_index_entry() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    db.execute_cypher("CREATE (:U {email: 'taken'})")
        .expect("seed");
    db.execute_cypher("CREATE (:U {email: 'fresh'}), (:U {email: 'taken'})")
        .expect_err("the second write breaks the index");
    db.execute_cypher("CREATE (:U {email: 'fresh'})")
        .expect("the failed statement's value was never claimed");
}

/// Two writers inserting one unique value at once: exactly one succeeds,
/// and the other is told the value exists, not that it conflicted.
#[test]
fn concurrent_inserts_of_one_unique_value_leave_one_node() {
    const WRITERS: usize = 8;
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect("index");
    let db = &db;
    let outcomes: Vec<Result<(), String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                s.spawn(move || {
                    db.execute_cypher_shared(
                        &format!("CREATE (:U {{email: 'same', w: {w}}})"),
                        None,
                        None,
                        None,
                        None,
                    )
                    .map(|_| ())
                    .map_err(|e| e.to_string())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(
        outcomes.iter().filter(|o| o.is_ok()).count(),
        1,
        "exactly one writer inserts: {outcomes:?}"
    );
    for err in outcomes.iter().filter_map(|o| o.as_ref().err()) {
        assert!(
            err.contains("unique constraint violated"),
            "a loser is told why: {err}"
        );
    }
}

/// A unique index over data that already breaks it is not created: the
/// statement fails naming a holder, and nothing of the index stays.
#[test]
fn a_unique_index_over_duplicates_is_not_created() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:U {email: 'same'})")
        .expect("first");
    db.execute_cypher("CREATE (:U {email: 'same'})")
        .expect("second");
    let err = db
        .execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .expect_err("the data breaks the index");
    assert!(
        err.to_string().contains("unique constraint violated"),
        "{err}"
    );
    db.execute_cypher("CREATE (:U {email: 'same'})")
        .expect("no index was left to refuse a third");
    db.execute_cypher("CREATE INDEX u_email ON :U(email)")
        .expect("the name is free again");
}

/// An index finds a list by each of its elements, but an equality on the
/// property holds only for the list itself.
#[test]
fn an_equality_does_not_match_a_list_holding_the_value() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE INDEX u_tags ON :U(tags)")
        .expect("index");
    db.execute_cypher("CREATE (:U {tags: ['x', 'y']})")
        .expect("list");
    db.execute_cypher("CREATE (:U {tags: 'x'})")
        .expect("scalar");
    assert_eq!(
        count(
            &mut db,
            "MATCH (u:U) WHERE u.tags = $v RETURN count(u)",
            &[("v", Value::String("x".into()))]
        ),
        1
    );
}

/// An index stored in the layout that preceded transactional entries is
/// rebuilt when the store opens: its own entries are cleared, the stored
/// nodes indexed again, and the index answers lookups.
#[test]
fn a_legacy_index_is_rebuilt_on_open() {
    use coordinode_query::index::IndexDefinition;
    use coordinode_query::index::ops::{load_index_definition, save_index_definition};
    use coordinode_storage::engine::partition::Partition;

    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (:U {email: 'a@x'})").expect("a");
        db.execute_cypher("CREATE (:U {email: 'b@x'})").expect("b");
        let mut legacy = IndexDefinition::btree("u_email", "U", "email").unique();
        legacy.layout = 0;
        save_index_definition(db.engine(), &legacy).expect("plant the legacy definition");
        db.engine()
            .put(Partition::Idx, b"idx:u_email:stale", b"")
            .expect("plant a legacy entry");
    }

    let mut db = Database::open(dir.path()).expect("reopen");
    let def = load_index_definition(db.engine(), "u_email")
        .expect("load")
        .expect("the index is still defined");
    assert_eq!(def.layout, coordinode_modality::ENTRY_LAYOUT);
    assert_eq!(def.state, coordinode_query::index::IndexState::Ready);
    assert!(
        db.engine()
            .get(Partition::Idx, b"idx:u_email:stale")
            .expect("get")
            .is_none(),
        "the legacy entries are cleared"
    );
    assert_eq!(
        count(
            &mut db,
            "MATCH (u:U) WHERE u.email = $e RETURN count(u)",
            &[("e", Value::String("a@x".into()))]
        ),
        1
    );
    db.execute_cypher("CREATE (:U {email: 'a@x'})")
        .expect_err("the rebuilt index enforces the constraint");
}
