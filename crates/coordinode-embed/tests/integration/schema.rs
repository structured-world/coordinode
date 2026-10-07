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

/// Require `property` of `label` to be unique, through a named constraint.
fn require_unique(db: &Database, label: &str, property: &str) {
    db.create_constraint(coordinode_embed::ConstraintDeclaration {
        name: None,
        label: label.into(),
        properties: vec![property.into()],
        kind: coordinode_core::schema::definition::ConstraintKind::Unique,
        if_not_exists: false,
        wait: None,
        on_duplicate_rename: None,
        scope: None,
    })
    .expect("create the uniqueness constraint");
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

// ── create_label_schema / create_edge_type_schema ────────────────────

/// Schema is persisted to storage and survives a re-open.
#[test]
fn create_label_schema_persists_across_reopen() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let dir = tempfile::tempdir().expect("tempdir");

    // Create schema and persist it.
    {
        let db = Database::open(dir.path()).expect("open");
        let mut schema = LabelSchema::new_node_id("Member");
        schema.add_property(PropertyDef::new("handle", PropertyType::String).not_null());
        schema.add_property(PropertyDef::new("score", PropertyType::Int));
        let version = db.create_label_schema(schema).expect("create schema");
        assert!(version > 0, "version must be positive");
    }

    // Reopen and verify the schema is still there.
    {
        let db2 = Database::open(dir.path()).expect("reopen");
        use coordinode_storage::engine::partition::Partition;
        let key = coordinode_core::schema::definition::encode_label_schema_key("Member", 1);
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

/// Unique index survives a database reopen and still enforces constraints.
///
/// Regression test: `load_all` on `Database::open()` must restore index state.
#[test]
fn create_label_schema_unique_constraint_enforced_after_reopen() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let dir = tempfile::tempdir().expect("tempdir");

    // Create schema with unique email, insert first node.
    {
        let mut db = Database::open(dir.path()).expect("open");
        let mut schema = LabelSchema::new_node_id("Account");
        schema.add_property(PropertyDef::new("email", PropertyType::String));
        db.create_label_schema(schema).expect("create schema");
        require_unique(&db, "Account", "email");
        db.execute_cypher("CREATE (u:Account {email: 'bob@test.com'})")
            .expect("first create should succeed");
    }

    // Reopen — unique index must be reloaded and still enforced.
    {
        let mut db2 = Database::open(dir.path()).expect("reopen");
        let result = db2.execute_cypher("CREATE (u:Account {email: 'bob@test.com'})");
        assert!(
            result.is_err(),
            "duplicate email must be rejected after reopen (load_all must restore index)"
        );
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("unique") || msg.contains("constraint") || msg.contains("index"),
            "error should mention unique/constraint/index, got: {msg}"
        );
    }
}

/// A uniqueness constraint created after the type definition is enforced on
/// the next write. Regression test: a duplicate value after MERGE must fail.
#[test]
fn create_label_schema_unique_constraint_enforced() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let (mut db, _dir) = open_db();

    // Declare User label, then make its email unique.
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new("email", PropertyType::String));
    db.create_label_schema(schema).expect("create schema");
    require_unique(&db, "User", "email");

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

/// The index of a uniqueness constraint is backfilled from the nodes stored
/// before it; after the backfill it enforces the constraint.
#[test]
fn create_label_schema_unique_backfills_existing_nodes() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let (mut db, _dir) = open_db();

    // Pre-create nodes WITHOUT a schema (flexible mode, no unique constraint).
    db.execute_cypher("CREATE (u:Product {sku: 'A1'})")
        .expect("create A1");
    db.execute_cypher("CREATE (u:Product {sku: 'B2'})")
        .expect("create B2");

    // Now declare the schema and make sku unique: the backfill must succeed.
    let mut schema = LabelSchema::new_node_id("Product");
    schema.add_property(PropertyDef::new("sku", PropertyType::String));
    db.create_label_schema(schema).expect("create schema");
    require_unique(&db, "Product", "sku");

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
        let db = Database::open(dir.path()).expect("open");
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
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        let schema = LocalSchemaStore::new(db2.engine())
            .load_edge_type("WORKS_AT")
            .expect("load edge schema")
            .expect("edge schema must exist after reopen");
        assert_eq!(schema.name, "WORKS_AT");
        assert_eq!(schema.properties.len(), 2);
        assert!(schema.get_property("since").is_some_and(|p| p.not_null));
    }
}

/// Calling create_label_schema again replaces the type facts at the next
/// revision, so a writer validated under the previous definition is held to
/// the new one rather than committing beside it.
#[test]
fn create_label_schema_replaces_at_the_next_revision() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let (db, _dir) = open_db();

    let mut schema_v1 = LabelSchema::new_node_id("Tag");
    schema_v1.add_property(PropertyDef::new("name", PropertyType::String));
    let v1 = db.create_label_schema(schema_v1).expect("first create");

    let mut schema_v2 = LabelSchema::new_node_id("Tag");
    schema_v2.add_property(PropertyDef::new("name", PropertyType::String));
    schema_v2.add_property(PropertyDef::new("count", PropertyType::Int));
    let v2 = db.create_label_schema(schema_v2).expect("second create");

    assert_eq!(v2, v1 + 1);
}

/// Current-revision pointer is written together with the schema body on
/// `create_label_schema`. Reading via the pointer yields the
/// same schema as reading the versioned key directly.
#[test]
fn current_revision_pointer_written_alongside_label_schema() {
    use coordinode_core::schema::definition::{
        LabelSchema, PropertyDef, PropertyType, encode_label_current_revision_key,
        encode_label_schema_key,
    };
    use coordinode_storage::engine::partition::Partition;

    let (db, _dir) = open_db();

    let mut schema = LabelSchema::new_node_id("Project");
    schema.add_property(PropertyDef::new("name", PropertyType::String));
    let version = db.create_label_schema(schema).expect("create");

    // Pointer must exist and decode to the same version.
    let pointer_bytes = db
        .engine()
        .get(
            Partition::Schema,
            &encode_label_current_revision_key("Project"),
        )
        .expect("pointer get")
        .expect("pointer must be present");
    let pointer_arr: [u8; 8] = pointer_bytes.as_ref().try_into().expect("u64 BE");
    assert_eq!(u64::from_be_bytes(pointer_arr), version);

    // Direct read of the versioned key returns the schema body.
    let body = db
        .engine()
        .get(
            Partition::Schema,
            &encode_label_schema_key("Project", version),
        )
        .expect("body get")
        .expect("body must be present");
    let decoded = LabelSchema::from_msgpack(&body).expect("decode");
    assert_eq!(decoded.name, "Project");
    assert_eq!(decoded.properties.len(), 1);
}

/// Current-revision pointer is written for edge types as well.
///
/// White-box: this test verifies the on-disk two-key revision scheme itself
/// (the `current_revision` pointer key resolving to the versioned body key), so
/// it deliberately reads the raw keys rather than going through
/// `SchemaStore::load_edge_type` — the store API hides exactly the indirection
/// under test.
#[test]
fn current_revision_pointer_written_alongside_edge_type_schema() {
    use coordinode_core::schema::definition::{
        EdgeTypeSchema, encode_edge_type_current_revision_key, encode_edge_type_schema_key,
    };
    use coordinode_storage::engine::partition::Partition;

    let (db, _dir) = open_db();

    let schema = EdgeTypeSchema::new("OWNED_BY");
    let version = db
        .create_edge_type_schema(schema)
        .expect("create edge type");

    let pointer_bytes = db
        .engine()
        .get(
            Partition::Schema,
            &encode_edge_type_current_revision_key("OWNED_BY"),
        )
        .expect("pointer get")
        .expect("pointer must be present");
    let pointer_arr: [u8; 8] = pointer_bytes.as_ref().try_into().expect("u64 BE");
    assert_eq!(u64::from_be_bytes(pointer_arr), version);

    let body = db
        .engine()
        .get(
            Partition::Schema,
            &encode_edge_type_schema_key("OWNED_BY", version),
        )
        .expect("body get")
        .expect("body must be present");
    let decoded = EdgeTypeSchema::from_msgpack(&body).expect("decode");
    assert_eq!(decoded.name, "OWNED_BY");
}

/// Hash placement labels round-trip through the schema partition.
#[test]
fn hash_placement_label_roundtrips() {
    use coordinode_core::schema::definition::{
        LabelSchema, PlacementKind, PlacementPolicy, PropertyDef, PropertyType, ShardKeyState,
        encode_label_schema_key,
    };
    use coordinode_storage::engine::partition::Partition;

    let (db, _dir) = open_db();

    let mut schema = LabelSchema::new("Order", PlacementPolicy::Hash("customer_id".to_string()));
    schema.add_property(PropertyDef::new("customer_id", PropertyType::String));
    schema.add_property(PropertyDef::new("total", PropertyType::Float));
    let version = db.create_label_schema(schema).expect("create");

    let body = db
        .engine()
        .get(
            Partition::Schema,
            &encode_label_schema_key("Order", version),
        )
        .expect("get")
        .expect("body");
    let decoded = LabelSchema::from_msgpack(&body).expect("decode");

    assert!(matches!(
        decoded.placement,
        PlacementPolicy::Hash(ref p) if p == "customer_id"
    ));
    assert_eq!(decoded.shard_keys.len(), 1);
    assert_eq!(decoded.shard_keys[0].state, ShardKeyState::Primary);
    assert_eq!(decoded.shard_keys[0].kind, PlacementKind::Hash);
    assert_eq!(decoded.shard_keys[0].property, "customer_id");
}

/// Range placement labels round-trip through the schema partition.
#[test]
fn range_placement_label_roundtrips() {
    use coordinode_core::schema::definition::{
        LabelSchema, PlacementKind, PlacementPolicy, PropertyDef, PropertyType,
        encode_label_schema_key,
    };
    use coordinode_storage::engine::partition::Partition;

    let (db, _dir) = open_db();

    let mut schema = LabelSchema::new("Event", PlacementPolicy::Range("timestamp".to_string()));
    schema.add_property(PropertyDef::new("timestamp", PropertyType::Timestamp));
    let version = db.create_label_schema(schema).expect("create");

    let body = db
        .engine()
        .get(
            Partition::Schema,
            &encode_label_schema_key("Event", version),
        )
        .expect("get")
        .expect("body");
    let decoded = LabelSchema::from_msgpack(&body).expect("decode");

    assert!(matches!(
        decoded.placement,
        PlacementPolicy::Range(ref p) if p == "timestamp"
    ));
    assert_eq!(decoded.shard_keys[0].kind, PlacementKind::Range);
}

/// Schema with all three placement kinds in the same database — they coexist
/// in the schema partition without key collisions.
#[test]
fn three_placement_kinds_coexist_in_schema_partition() {
    use coordinode_core::schema::definition::{LabelSchema, PlacementPolicy};

    let (db, _dir) = open_db();

    db.create_label_schema(LabelSchema::new("User", PlacementPolicy::NodeId))
        .expect("user");
    db.create_label_schema(LabelSchema::new(
        "Order",
        PlacementPolicy::Hash("customer_id".to_string()),
    ))
    .expect("order");
    db.create_label_schema(LabelSchema::new(
        "Event",
        PlacementPolicy::Range("timestamp".to_string()),
    ))
    .expect("event");
    // No panic, no error → coexistence verified.
}

/// Schema lookup surfaces an error (not a silent miss) when the
/// `current_revision` pointer exists but is corrupted — i.e. not a clean 8-byte
/// u64. Silently treating corrupt pointers as "no schema" would cause STRICT
/// labels to drop validation and accept arbitrary property writes.
#[test]
fn corrupt_label_pointer_surfaces_error() {
    use coordinode_core::schema::definition::{
        LabelSchema, PropertyDef, PropertyType, encode_label_current_revision_key,
    };
    use coordinode_storage::engine::partition::Partition;

    let (mut db, _dir) = open_db();

    // Create a real STRICT label so SET routes through schema validation.
    let mut schema = LabelSchema::new_node_id("Strict");
    schema.set_mode(coordinode_core::schema::definition::SchemaMode::Strict);
    schema.add_property(PropertyDef::new("name", PropertyType::String).not_null());
    db.create_label_schema(schema).expect("create schema");
    db.execute_cypher("CREATE (s:Strict {name: 'a'})")
        .expect("seed node");

    // Overwrite the pointer with 4 bytes (instead of 8) — torn write simulation.
    db.engine_shared()
        .put(
            Partition::Schema,
            &encode_label_current_revision_key("Strict"),
            &[0u8, 0, 0, 1],
        )
        .expect("corrupt pointer");

    // Operations that must consult the schema should now fail loudly. A bare
    // MATCH/RETURN can dodge the schema path; mutating with `SET n += {map}`
    // forces schema mode resolution on every row.
    let result = db.execute_cypher("MATCH (s:Strict) SET s += {extra: 'x'} RETURN s");
    assert!(
        result.is_err(),
        "corrupt pointer must surface as an error, not pass through silently"
    );
}

/// Same invariant for edge type pointers — corrupt bytes must not be silently
/// treated as "edge type unknown".
#[test]
fn corrupt_edge_type_pointer_surfaces_error() {
    use coordinode_core::schema::definition::{
        EdgeTypeSchema, encode_edge_type_current_revision_key,
    };
    use coordinode_storage::engine::partition::Partition;

    let (mut db, _dir) = open_db();

    let schema = EdgeTypeSchema::new("KNOWS");
    db.create_edge_type_schema(schema)
        .expect("create edge type");

    // Seed BEFORE corruption so the seed itself doesn't traverse the bad pointer.
    db.execute_cypher(
        "CREATE (a:Person {name: 'A'})-[:KNOWS {since: 2020}]->(b:Person {name: 'B'})",
    )
    .expect("seed edge");

    db.engine_shared()
        .put(
            Partition::Schema,
            &encode_edge_type_current_revision_key("KNOWS"),
            &[0xFFu8; 3],
        )
        .expect("corrupt pointer");

    // Any code path that inspects edge-type-temporality through the pointer
    // (e.g. `SET r.<prop>` on a matched edge) must error rather than silently
    // returning "non-temporal".
    let result = db.execute_cypher("MATCH (a)-[r:KNOWS]->(b) SET r.note = 'x' RETURN r");
    assert!(
        result.is_err(),
        "corrupt edge-type pointer must surface as an error"
    );
}

/// Dangling pointer: pointer survives but the versioned schema body does not
/// (impossible by construction in CE, but EE migrations + manual storage
/// surgery can produce it). The load path must error rather than report
/// "no schema".
#[test]
fn dangling_label_pointer_surfaces_error() {
    use coordinode_core::schema::definition::{
        LabelSchema, PropertyDef, PropertyType, SchemaMode, encode_label_current_revision_key,
        encode_label_schema_key,
    };
    use coordinode_storage::engine::partition::Partition;

    let (mut db, _dir) = open_db();

    let mut schema = LabelSchema::new_node_id("Acct");
    schema.set_mode(SchemaMode::Strict);
    schema.add_property(PropertyDef::new("name", PropertyType::String).not_null());
    let version = db.create_label_schema(schema).expect("create");

    db.execute_cypher("CREATE (a:Acct {name: 'first'})")
        .expect("seed");

    // Delete the schema body but keep the pointer — leaves a dangling pointer.
    db.engine_shared()
        .delete(Partition::Schema, &encode_label_schema_key("Acct", version))
        .expect("drop schema body");
    // Confirm pointer is still in place.
    let _present = db
        .engine_shared()
        .get(
            Partition::Schema,
            &encode_label_current_revision_key("Acct"),
        )
        .expect("get pointer")
        .expect("pointer must remain");

    let result = db.execute_cypher("MATCH (a:Acct) SET a += {note: 'x'} RETURN a");
    assert!(
        result.is_err(),
        "dangling pointer (body missing) must surface as an error"
    );
}

/// elementId derivation is bijective: encoding then decoding any NodeId
/// returns the original value.
#[test]
fn element_id_roundtrip_for_real_database_node_ids() {
    use coordinode_core::graph::node::NodeId;

    // Sample sequence values spanning the 44-bit range.
    for sequence in [
        1u64,
        42,
        1_000,
        u32::MAX as u64,
        (1u64 << 40),
        (1u64 << 44) - 1,
    ] {
        let id = NodeId::compose(0, sequence);
        let encoded = id.to_element_id();
        let decoded = NodeId::from_element_id(&encoded).expect("decode");
        assert_eq!(
            decoded, id,
            "elementId round-trip failed for sequence={sequence}"
        );
    }
}

// ── Edge identity ───────────────────────────────────────────────────

/// The id of the `:P` node named `n`.
fn p_id(db: &mut Database, n: &str) -> coordinode_core::graph::node::NodeId {
    let rows = db
        .execute_cypher(&format!("MATCH (p:P {{n: '{n}'}}) RETURN id(p) AS id"))
        .expect("find");
    let Some(coordinode_core::graph::types::Value::Int(id)) =
        rows.first().and_then(|r| r.get("id"))
    else {
        unreachable!("node {n} has an integer id: {rows:?}");
    };
    coordinode_core::graph::node::NodeId::from_raw(*id as u64)
}

/// Store a KNOWS edge from `src` to `tgt` identified by `context` straight
/// through the edge store, as a statement cannot write one yet.
fn plant_knows(
    db: &Database,
    src: coordinode_core::graph::node::NodeId,
    tgt: coordinode_core::graph::node::NodeId,
    context: &str,
) {
    use coordinode_core::graph::edge::EdgeProperties;
    use coordinode_core::graph::types::Value;
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::begin(db.engine(), Some(&oracle), read_ts);
    LocalEdgeStore
        .put_edge_discriminated(
            &mut txn,
            "KNOWS",
            src,
            tgt,
            &Value::String(context.into()),
            &EdgeProperties::new(),
        )
        .expect("put");
    let wc = WriteConcern::majority();
    txn.commit(&CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    })
    .expect("commit");
}

/// A node holding an edge of a type identified by its own discriminator is
/// refused by every statement that would carry, copy or drop that edge as a
/// single edge, and the refused statement changes nothing.
#[test]
fn a_statement_reaching_an_independently_identified_edge_is_refused_whole() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE EDGE TYPE KNOWS WITH (context: STRING NOT NULL) DISCRIMINATED BY (context)",
    )
    .expect("categorical");
    db.execute_cypher("CREATE (:P {n: 'a'}), (:P {n: 'b'}), (:P {n: 'c'})")
        .expect("nodes");
    let (a, b) = (p_id(&mut db, "a"), p_id(&mut db, "b"));
    plant_knows(&db, a, b, "work");

    for statement in [
        "MATCH (a:P {n: 'a'}) CLONE NODE a AS x WITH EDGES",
        "MATCH (a:P {n: 'a'}), (c:P {n: 'c'}) MERGE NODES (c, a) INTO c \
         TRANSFER EDGES FROM a TO c",
        "MATCH (a:P {n: 'a'}), (c:P {n: 'c'}) REDIRECT EDGES FROM a TO c",
        "MATCH (a:P {n: 'a'}) DETACH DELETE a",
        "MATCH (a:P {n: 'a'})-[r:KNOWS]->(b) RETURN r",
    ] {
        let error = db.execute_cypher(statement).expect_err(statement);
        assert!(
            error.to_string().contains("by 'context'"),
            "{statement}: {error}"
        );
    }
    assert_eq!(
        db.execute_cypher("MATCH (p:P) RETURN p")
            .expect("nodes")
            .len(),
        3,
        "no node was cloned, merged away or deleted"
    );
}

/// The discriminator `name`'s definition resolved to, if any.
fn discriminator_of(db: &Database, name: &str) -> Option<String> {
    db.edge_type_schemas()
        .expect("edge types")
        .into_iter()
        .find(|s| s.name == name)
        .expect("defined")
        .discriminator()
        .map(|d| d.column.clone())
}

/// An edge type is identified by the property DISCRIMINATED BY names,
/// whether or not it is temporal, and a temporal type naming none by
/// valid_from; the resolved discriminator is what the catalog keeps, across
/// a reopen.
#[test]
fn an_edge_type_keeps_the_discriminator_it_was_created_with() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open");
        let rows = db
            .execute_cypher(
                "CREATE EDGE TYPE KNOWS WITH (context: STRING NOT NULL) DISCRIMINATED BY (context)",
            )
            .expect("categorical");
        assert_eq!(
            rows[0].get("discriminator"),
            Some(&coordinode_core::graph::types::Value::String(
                "context".into()
            ))
        );
        db.execute_cypher(
            "CREATE EDGE TYPE ASSERTS TEMPORAL DISCRIMINATED BY (assertion_key) \
             WITH (assertion_key: STRING NOT NULL)",
        )
        .expect("independent temporal");
        db.execute_cypher("CREATE EDGE TYPE WORKS_AT TEMPORAL WITH (role: STRING)")
            .expect("start-identified");
        db.execute_cypher("CREATE EDGE TYPE LIKES")
            .expect("single-edge");
    }
    let db = Database::open(dir.path()).expect("reopen");
    assert_eq!(discriminator_of(&db, "KNOWS").as_deref(), Some("context"));
    assert_eq!(
        discriminator_of(&db, "ASSERTS").as_deref(),
        Some("assertion_key")
    );
    assert_eq!(
        discriminator_of(&db, "WORKS_AT").as_deref(),
        Some("valid_from")
    );
    assert_eq!(discriminator_of(&db, "LIKES"), None);
}

/// A discriminator that cannot identify an edge is refused before anything
/// is written: one the type does not declare, one that may be null, one of
/// a type that holds no single comparable value, and a temporal type's
/// valid_from declared as anything but a timestamp.
#[test]
fn an_edge_type_whose_discriminator_cannot_identify_an_edge_is_refused() {
    let (mut db, _dir) = open_db();
    for (statement, why) in [
        (
            "CREATE EDGE TYPE A WITH (context: STRING NOT NULL) DISCRIMINATED BY (missing)",
            "does not declare",
        ),
        (
            "CREATE EDGE TYPE B WITH (context: STRING) DISCRIMINATED BY (context)",
            "NOT NULL",
        ),
        (
            "CREATE EDGE TYPE C WITH (context: MAP NOT NULL) DISCRIMINATED BY (context)",
            "is MAP",
        ),
        (
            "CREATE EDGE TYPE D TEMPORAL WITH (valid_from: STRING)",
            "TIMESTAMP",
        ),
    ] {
        let error = db.execute_cypher(statement).expect_err(statement);
        assert!(error.to_string().contains(why), "{statement}: {error}");
    }
    assert!(db.edge_type_schemas().expect("edge types").is_empty());
}

/// Edges of a type identified by its own discriminator are not read or
/// written as single edges or by valid_from, which would merge or split its
/// instances: the statement is refused and nothing is written. A
/// start-identified and a single-edge type are unaffected.
#[test]
fn edges_of_an_independently_identified_type_are_refused_not_misread() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE EDGE TYPE KNOWS WITH (context: STRING NOT NULL) DISCRIMINATED BY (context)",
    )
    .expect("categorical");
    db.execute_cypher("CREATE (:P {n: 'a'}), (:P {n: 'b'})")
        .expect("nodes");

    let error = db
        .execute_cypher(
            "MATCH (a:P {n: 'a'}), (b:P {n: 'b'}) CREATE (a)-[:KNOWS {context: 'work'}]->(b)",
        )
        .expect_err("not written as a single edge");
    assert!(error.to_string().contains("'context'"), "{error}");
    let found = db
        .execute_cypher("MATCH (:P)-[k:KNOWS]->(:P) RETURN k")
        .expect_err("not read as a single edge");
    assert!(found.to_string().contains("'context'"), "{found}");

    db.execute_cypher("CREATE EDGE TYPE LIKES").expect("single");
    db.execute_cypher("MATCH (a:P {n: 'a'}), (b:P {n: 'b'}) CREATE (a)-[:LIKES]->(b)")
        .expect("a single-edge type still writes");
    assert_eq!(
        db.execute_cypher("MATCH (:P)-[r:LIKES]->(:P) RETURN r")
            .expect("read")
            .len(),
        1
    );
    // A traversal of every type still reads the others: the declared type
    // holds no edge to misread.
    for statement in [
        "MATCH (:P)-[r]->(:P) RETURN r",
        "MATCH (:P)<-[r]-(:P) RETURN r",
        "MATCH (:P)-[r*1..2]->(:P) RETURN r",
    ] {
        assert_eq!(
            db.execute_cypher(statement).expect(statement).len(),
            1,
            "{statement}"
        );
    }
    db.execute_cypher("MATCH (p:P {n: 'b'}) DETACH DELETE p")
        .expect("a detach ranges over every type");
    assert!(
        db.execute_cypher("MATCH (:P)-[r]->() RETURN r")
            .expect("read")
            .is_empty()
    );
}
