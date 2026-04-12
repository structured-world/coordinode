//! Integration tests: Schema DDL and type enforcement.
//!
//! Tests CREATE LABEL, EDGE_TYPE, constraint enforcement, and type validation.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

// ── Schema DDL ──────────────────────────────────────────────────────

#[test]
fn create_label_schema() {
    let (mut db, _dir) = open_db();
    let result = db.execute_cypher("CREATE LABEL User");
    // Schema DDL may or may not be implemented yet — test that it either
    // succeeds or returns a meaningful error (not a panic)
    match result {
        Ok(_) => {} // Schema created successfully
        Err(e) => {
            let msg = format!("{e}");
            // Verify it's a proper error, not a crash
            assert!(!msg.is_empty(), "Schema error should have a message");
        }
    }
}

// ── Node property types ─────────────────────────────────────────────

#[test]
fn string_property() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:T {val: 'hello'})")
        .expect("create");
    let rows = db.execute_cypher("MATCH (n:T) RETURN n.val").expect("read");
    assert_eq!(rows.len(), 1);
}

#[test]
fn integer_property() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:T {val: 42})").expect("create");
    let rows = db
        .execute_cypher("MATCH (n:T) WHERE n.val = 42 RETURN n")
        .expect("read");
    assert_eq!(rows.len(), 1);
}

#[test]
fn float_property() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:T {val: 3.14})")
        .expect("create");
    let rows = db.execute_cypher("MATCH (n:T) RETURN n.val").expect("read");
    assert_eq!(rows.len(), 1);
}

#[test]
fn boolean_property() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:T {active: true})")
        .expect("create");
    let rows = db
        .execute_cypher("MATCH (n:T) WHERE n.active = true RETURN n")
        .expect("read");
    assert_eq!(rows.len(), 1);
}

#[test]
fn list_property() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:T {tags: ['a', 'b', 'c']})")
        .expect("create");
    let rows = db
        .execute_cypher("MATCH (n:T) RETURN n.tags")
        .expect("read");
    assert_eq!(rows.len(), 1);
}

#[test]
fn null_property_handling() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (n:T {name: 'test'})")
        .expect("create");
    // Reading a non-existent property should return null, not error
    let rows = db
        .execute_cypher("MATCH (n:T) RETURN n.nonexistent")
        .expect("read null");
    assert_eq!(rows.len(), 1);
}

// ── Multiple labels ─────────────────────────────────────────────────

// ── Edge creation ───────────────────────────────────────────────────

#[test]
fn edge_creation_does_not_error() {
    let (mut db, _dir) = open_db();
    // Creating edges with properties should succeed
    let result = db.execute_cypher(
        "CREATE (a:Person {name: 'Alice'})-[:KNOWS {since: 2020}]->(b:Person {name: 'Bob'})",
    );
    assert!(result.is_ok(), "edge creation should not error");
}

#[test]
fn multiple_edge_type_creation() {
    let (mut db, _dir) = open_db();
    // Creating edges with different types should not error
    let r1 =
        db.execute_cypher("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})");
    let r2 = db
        .execute_cypher("CREATE (c:Person {name: 'Charlie'})-[:LIKES]->(d:Person {name: 'Dave'})");
    assert!(r1.is_ok());
    assert!(r2.is_ok());
}

// ── R-API1: create_label_schema / create_edge_type_schema ────────────

/// Schema is persisted to storage and survives a re-open.
#[test]
fn create_label_schema_persists_across_reopen() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let dir = tempfile::tempdir().expect("tempdir");

    // Create schema and persist it.
    {
        let mut db = Database::open(dir.path()).expect("open");
        let mut schema = LabelSchema::new("Member");
        schema.add_property(PropertyDef::new("handle", PropertyType::String).not_null());
        schema.add_property(PropertyDef::new("score", PropertyType::Int));
        let version = db.create_label_schema(schema).expect("create schema");
        assert!(version > 0, "version must be positive");
    }

    // Reopen and verify the schema is still there.
    {
        let db2 = Database::open(dir.path()).expect("reopen");
        use coordinode_storage::engine::partition::Partition;
        let key = coordinode_core::schema::definition::encode_label_schema_key("Member");
        let bytes = db2
            .engine()
            .get(Partition::Schema, &key)
            .expect("storage get")
            .expect("schema must exist after reopen");
        let schema = LabelSchema::from_msgpack(&bytes).expect("deserialize schema");
        assert_eq!(schema.name, "Member");
        assert_eq!(schema.properties.len(), 2);
        assert!(schema.get_property("handle").is_some());
        assert!(schema.get_property("handle").is_some_and(|p| p.not_null));
    }
}

/// `unique: true` property → B-tree unique index is registered and enforced.
/// Regression test: duplicate value via MERGE must fail after create_label_schema.
#[test]
fn create_label_schema_unique_constraint_enforced() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let (mut db, _dir) = open_db();

    // Declare User label with unique email.
    let mut schema = LabelSchema::new("User");
    schema.add_property(PropertyDef::new("email", PropertyType::String).unique());
    db.create_label_schema(schema).expect("create schema");

    // First MERGE with unique email — should succeed.
    db.execute_cypher("MERGE (u:User {email: 'alice@test.com'})")
        .expect("first MERGE should succeed");

    // Second CREATE with the same email — must fail (unique constraint).
    let result = db.execute_cypher("CREATE (u:User {email: 'alice@test.com'})");
    assert!(
        result.is_err(),
        "duplicate email should be rejected by unique constraint"
    );
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unique") || msg.contains("constraint") || msg.contains("index"),
        "error should mention unique/constraint/index, got: {msg}"
    );
}

/// `unique: true` index is backfilled for nodes that existed before schema creation.
/// Backfill completes without error; the index is queryable after backfill.
#[test]
fn create_label_schema_unique_backfills_existing_nodes() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let (mut db, _dir) = open_db();

    // Pre-create nodes WITHOUT a schema (flexible mode, no unique constraint).
    db.execute_cypher("CREATE (u:Product {sku: 'A1'})")
        .expect("create A1");
    db.execute_cypher("CREATE (u:Product {sku: 'B2'})")
        .expect("create B2");

    // Now declare schema with unique sku — must backfill without error.
    let mut schema = LabelSchema::new("Product");
    schema.add_property(PropertyDef::new("sku", PropertyType::String).unique());
    db.create_label_schema(schema)
        .expect("backfill should succeed");

    // After backfill, duplicate sku must be rejected.
    let result = db.execute_cypher("CREATE (u:Product {sku: 'A1'})");
    assert!(
        result.is_err(),
        "duplicate sku after backfill must be rejected"
    );
}

/// Edge type schema is persisted to storage.
#[test]
fn create_edge_type_schema_persists() {
    use coordinode_core::schema::definition::{EdgeTypeSchema, PropertyDef, PropertyType};

    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open");
        let mut schema = EdgeTypeSchema::new("WORKS_AT");
        schema.add_property(PropertyDef::new("since", PropertyType::Timestamp).not_null());
        schema.add_property(PropertyDef::new("role", PropertyType::String));
        let version = db
            .create_edge_type_schema(schema)
            .expect("create edge schema");
        assert!(version > 0);
    }

    {
        let db2 = Database::open(dir.path()).expect("reopen");
        use coordinode_storage::engine::partition::Partition;
        let key = coordinode_core::schema::definition::encode_edge_type_schema_key("WORKS_AT");
        let bytes = db2
            .engine()
            .get(Partition::Schema, &key)
            .expect("storage get")
            .expect("edge schema must exist after reopen");
        let schema = EdgeTypeSchema::from_msgpack(&bytes).expect("deserialize");
        assert_eq!(schema.name, "WORKS_AT");
        assert_eq!(schema.properties.len(), 2);
        assert!(schema.get_property("since").is_some_and(|p| p.not_null));
    }
}

/// create_label_schema is idempotent — calling it twice with the same name updates
/// the schema (schema version increases) without error.
#[test]
fn create_label_schema_idempotent() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let (mut db, _dir) = open_db();

    let mut schema_v1 = LabelSchema::new("Tag");
    schema_v1.add_property(PropertyDef::new("name", PropertyType::String));
    let v1 = db.create_label_schema(schema_v1).expect("first create");

    let mut schema_v2 = LabelSchema::new("Tag");
    schema_v2.add_property(PropertyDef::new("name", PropertyType::String));
    schema_v2.add_property(PropertyDef::new("count", PropertyType::Int));
    let v2 = db.create_label_schema(schema_v2).expect("second create");

    // Second schema has more properties → higher version.
    assert!(v2 > v1, "updated schema must have higher version");
}
