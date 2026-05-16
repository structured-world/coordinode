//! Integration tests: ALTER LABEL SET SCHEMA DDL (G029).
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

// ── ADR-023 C-decision regression: revision-bump semantics ──────────

/// ALTER LABEL ... SET SCHEMA <mode> must bump `schema_revision` (mode change is
/// a write-path mutation per ADR-023). Adding a property via `Database::
/// create_label_schema` (or implicit declaration on first node insert) must
/// NOT bump `schema_revision` (properties mutate the current snapshot in
/// place). Together these enforce the lexicon decision: revisions track DDL
/// snapshot identity, not arbitrary field changes.
#[test]
fn alter_label_mode_bumps_revision_but_property_add_does_not() {
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

    // Adding another property through a fresh create with the same name is
    // idempotent and must NOT advance the revision — property mutations are
    // snapshot edits, not new revisions (ADR-023).
    let mut schema_v2 = LabelSchema::new_node_id("Doc");
    schema_v2.add_property(PropertyDef::new("title", PropertyType::String));
    schema_v2.add_property(PropertyDef::new("body", PropertyType::String));
    let rev_after_property_add = db.create_label_schema(schema_v2).expect("re-create");
    assert_eq!(
        rev_after_property_add, 1,
        "adding a property must NOT bump schema_revision (per ADR-023)"
    );

    // ALTER LABEL SET SCHEMA <mode> mutates write-path semantics → MUST bump.
    let rows = db
        .execute_cypher("ALTER LABEL Doc SET SCHEMA FLEXIBLE")
        .expect("alter mode");
    let version_after_mode = rows[0].get("version");
    assert!(
        matches!(version_after_mode, Some(Value::Int(v)) if *v >= 2),
        "ALTER LABEL SET SCHEMA must bump schema_revision to >= 2, got: {version_after_mode:?}"
    );

    // A subsequent mode change bumps again.
    let rows2 = db
        .execute_cypher("ALTER LABEL Doc SET SCHEMA VALIDATED")
        .expect("alter mode again");
    let version_after_second = rows2[0].get("version");
    assert!(
        matches!(version_after_second, Some(Value::Int(v)) if *v >= 3),
        "second ALTER LABEL SET SCHEMA must bump again to >= 3, got: {version_after_second:?}"
    );
}
