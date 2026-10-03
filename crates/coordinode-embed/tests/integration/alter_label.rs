//! Integration tests: ALTER LABEL SET SCHEMA DDL.
//!
//! Verifies that schema mode can be changed via Cypher DDL
//! and that write-time validation respects the new mode.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

// ── Basic ALTER LABEL ───────────────────────────────────────────────

/// ALTER LABEL SET SCHEMA VALIDATED returns result with label, mode, version.
#[test]
fn alter_label_returns_result() {
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher("ALTER LABEL User SET SCHEMA VALIDATED")
        .expect("alter label");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("label"), Some(&Value::String("User".into())));
    assert_eq!(
        rows[0].get("mode"),
        Some(&Value::String("VALIDATED".into()))
    );
}

/// ALTER LABEL SET SCHEMA FLEXIBLE works.
#[test]
fn alter_label_flexible() {
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher("ALTER LABEL Config SET SCHEMA FLEXIBLE")
        .expect("alter");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("mode"), Some(&Value::String("FLEXIBLE".into())));
}

/// ALTER LABEL SET SCHEMA STRICT works.
#[test]
fn alter_label_strict() {
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher("ALTER LABEL Product SET SCHEMA STRICT")
        .expect("alter");
    assert_eq!(rows[0].get("mode"), Some(&Value::String("STRICT".into())));
}

// ── Case insensitive ────────────────────────────────────────────────

#[test]
fn alter_label_case_insensitive() {
    let (mut db, _dir) = open_db();

    let rows = db
        .execute_cypher("alter label Test set schema flexible")
        .expect("alter");
    assert_eq!(rows[0].get("mode"), Some(&Value::String("FLEXIBLE".into())));
}

// ── Schema mode persists across reopen ──────────────────────────────

/// Schema mode change persists after Database close + reopen.
#[test]
fn alter_label_persists_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Session 1: set schema mode to FLEXIBLE
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("ALTER LABEL Device SET SCHEMA FLEXIBLE")
            .expect("alter");
    }

    // Session 2: verify the mode persisted by altering again
    // (if the first alter didn't persist, this creates a new STRICT schema)
    {
        let mut db = Database::open(dir.path()).expect("reopen");
        // ALTER again to VALIDATED — if previous was persisted, version > 1
        let rows = db
            .execute_cypher("ALTER LABEL Device SET SCHEMA VALIDATED")
            .expect("alter again");
        let version = rows[0].get("version");
        // Version should be > 1 if the first ALTER persisted
        assert!(
            matches!(version, Some(Value::Int(v)) if *v >= 2),
            "version should be >= 2 after two ALTERs, got: {version:?}"
        );
    }
}

// ── Invalid mode ────────────────────────────────────────────────────

/// Invalid schema mode should fail at parse level (PEG rejects unknown modes).
#[test]
fn alter_label_invalid_mode_parse_error() {
    let (mut db, _dir) = open_db();

    let result = db.execute_cypher("ALTER LABEL User SET SCHEMA UNKNOWN");
    assert!(result.is_err(), "invalid mode should fail");
}

// ── Activation validates the data it governs ────────────────────────

/// A mode that a stored node already breaks is refused, and the label keeps
/// the mode it had: activating it would declare a rule the data does not
/// follow.
#[test]
fn alter_label_refuses_a_mode_existing_nodes_break() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Doc {x: 1})").expect("create");

    let refused = db.execute_cypher("ALTER LABEL Doc SET SCHEMA STRICT");
    let message = match refused {
        Ok(rows) => panic!("STRICT with an undeclared stored property was accepted: {rows:?}"),
        Err(e) => e.to_string(),
    };
    assert!(
        message.contains("breaks it") && message.contains("'x'"),
        "the refusal names the node and the property it breaks: {message}"
    );

    // Still accepts what the old mode accepts.
    db.execute_cypher("CREATE (:Doc {y: 2})")
        .expect("the label kept its previous mode");
}

/// A mode the stored nodes satisfy is accepted.
#[test]
fn alter_label_accepts_a_mode_existing_nodes_satisfy() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Doc)").expect("create");
    db.execute_cypher("ALTER LABEL Doc SET SCHEMA STRICT")
        .expect("no stored node breaks STRICT");
    assert!(
        db.execute_cypher("CREATE (:Doc {x: 1})").is_err(),
        "STRICT is in force for new writes"
    );
}

/// A writer that validated under the old mode and commits after the new one
/// landed is refused: its node was admitted by a rule no longer in force,
/// and the activation never saw it.
#[test]
fn a_write_validated_under_a_replaced_mode_is_refused() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Doc)").expect("label exists");

    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "CREATE (:Doc {x: 1})", None)
        .expect("FLEXIBLE accepts the property");

    db.execute_cypher("ALTER LABEL Doc SET SCHEMA STRICT")
        .expect("no committed node breaks STRICT");

    assert!(
        db.commit_transaction(tx).is_err(),
        "a node validated under FLEXIBLE committed under STRICT"
    );
    let rows = db
        .execute_cypher("MATCH (n:Doc) WHERE n.x IS NOT NULL RETURN count(n) AS n")
        .expect("count");
    assert_eq!(rows[0].get("n"), Some(&Value::Int(0)));
}

/// A transaction that only read under the old mode commits: reading is not a
/// write the new rule could be broken by.
#[test]
fn a_read_under_a_replaced_mode_still_commits() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Doc)").expect("label exists");

    let tx = db.begin_transaction();
    db.execute_in_transaction(tx, "MATCH (n:Doc) RETURN n", None)
        .expect("read");
    db.execute_cypher("ALTER LABEL Doc SET SCHEMA STRICT")
        .expect("alter");
    db.commit_transaction(tx)
        .expect("a read-only transaction is not refused by a schema change");
}

/// A declared property made NOT NULL through the typed API is refused while
/// a stored node lacks it, the same as through DDL.
#[test]
fn a_typed_schema_change_is_validated_against_stored_nodes() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:Doc {x: 1})").expect("create");

    let mut schema = LabelSchema::new_node_id("Doc");
    schema.add_property(PropertyDef::new("title", PropertyType::String).not_null());
    assert!(
        db.create_label_schema(schema).is_err(),
        "a NOT NULL property a stored node lacks was declared"
    );
}

// ── Regression: revision-bump semantics ─────────────────────────────

/// Every change of a label's definition publishes the next `schema_revision`:
/// a replacement through `Database::create_label_schema` and each ALTER LABEL
/// ... SET SCHEMA <mode>. A published revision is never rewritten, so a
/// writer validated under one is refused once another is in force.
#[test]
fn every_definition_change_publishes_the_next_revision() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let (mut db, _dir) = open_db();

    // Create label via the typed API — revision should be 1 with one property.
    let mut schema = LabelSchema::new_node_id("Doc");
    schema.add_property(PropertyDef::new("title", PropertyType::String));
    let rev_after_create = db.create_label_schema(schema).expect("create");
    assert_eq!(
        rev_after_create, 1,
        "fresh label must start at schema_revision=1"
    );

    // Replacing the definition is a new revision: a writer validated under
    // the previous one is held to the new definition, never committed beside
    // it under the same revision.
    let mut schema_v2 = LabelSchema::new_node_id("Doc");
    schema_v2.add_property(PropertyDef::new("title", PropertyType::String));
    schema_v2.add_property(PropertyDef::new("body", PropertyType::String));
    let rev_after_property_add = db.create_label_schema(schema_v2).expect("re-create");
    assert_eq!(
        rev_after_property_add, 2,
        "replacing the definition publishes the next schema_revision"
    );

    // ALTER LABEL SET SCHEMA <mode> mutates write-path semantics → MUST bump.
    let rows = db
        .execute_cypher("ALTER LABEL Doc SET SCHEMA FLEXIBLE")
        .expect("alter mode");
    let version_after_mode = rows[0].get("version");
    assert!(
        matches!(version_after_mode, Some(Value::Int(3))),
        "ALTER LABEL SET SCHEMA must bump schema_revision to 3, got: {version_after_mode:?}"
    );

    // A subsequent mode change bumps again.
    let rows2 = db
        .execute_cypher("ALTER LABEL Doc SET SCHEMA VALIDATED")
        .expect("alter mode again");
    let version_after_second = rows2[0].get("version");
    assert!(
        matches!(version_after_second, Some(Value::Int(4))),
        "second ALTER LABEL SET SCHEMA must bump again to 4, got: {version_after_second:?}"
    );
}
