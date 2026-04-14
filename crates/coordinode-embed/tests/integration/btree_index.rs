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

#[test]
fn explain_shows_index_scan_after_create_index_via_cypher() {
    // Regression test (R-API2): after CREATE INDEX, EXPLAIN for a matching
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

#[test]
fn drop_index_then_duplicate_insert_succeeds() {
    // After DROP INDEX, the unique constraint is lifted and duplicates are allowed.
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE UNIQUE INDEX u_code ON :Item(code)")
        .expect("CREATE UNIQUE INDEX");
    db.execute_cypher("CREATE (:Item {code: 'X1'})")
        .expect("first insert");

    // Drop the unique index.
    db.execute_cypher("DROP INDEX u_code").expect("DROP INDEX");

    // Now duplicate should be allowed.
    let result = db.execute_cypher("CREATE (:Item {code: 'X1'})");
    assert!(
        result.is_ok(),
        "after DROP INDEX, duplicate insert must succeed, got: {:?}",
        result.err()
    );
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
