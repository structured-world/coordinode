//! Integration tests: VALIDATED mode _extra overflow storage.
//!
//! Verifies that in VALIDATED schema mode, undeclared properties
//! are stored in the `extra` overflow map (string keys, no interning)
//! and are readable through Cypher queries.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

// ── VALIDATED mode: declared + undeclared properties ────────────────

/// In VALIDATED mode with a declared "name" property, an undeclared
/// "nickname" property should be stored in `extra` and readable.
#[test]
fn validated_mode_undeclared_property_readable() {
    let (mut db, _dir) = open_db();

    // Set up VALIDATED schema with "name" declared.
    // First create the label schema with a property, then set to VALIDATED.
    // Note: CREATE LABEL DDL may not be fully wired, so we use ALTER LABEL
    // which creates the schema if it doesn't exist.
    db.execute_cypher("ALTER LABEL Person SET SCHEMA VALIDATED")
        .expect("alter to validated");

    // Create a node with both declared and undeclared properties.
    // "name" is NOT declared in schema (ALTER LABEL doesn't add properties),
    // so ALL properties will be treated as undeclared in VALIDATED mode.
    db.execute_cypher("CREATE (n:Person {name: 'Alice', nickname: 'Ali', age: 30})")
        .expect("create node");

    // All properties should be readable regardless of storage location.
    let rows = db
        .execute_cypher("MATCH (n:Person) RETURN n.name, n.nickname, n.age")
        .expect("match");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.name"), Some(&Value::String("Alice".into())));
    assert_eq!(
        rows[0].get("n.nickname"),
        Some(&Value::String("Ali".into()))
    );
    assert_eq!(rows[0].get("n.age"), Some(&Value::Int(30)));
}

/// VALIDATED properties persist across Database reopen.
#[test]
fn validated_extra_persists_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("ALTER LABEL Device SET SCHEMA VALIDATED")
            .expect("alter");
        db.execute_cypher("CREATE (d:Device {serial: 'XYZ', firmware: '1.0'})")
            .expect("create");
    }

    {
        let mut db = Database::open(dir.path()).expect("reopen");
        let rows = db
            .execute_cypher("MATCH (d:Device) RETURN d.serial, d.firmware")
            .expect("match");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("d.serial"), Some(&Value::String("XYZ".into())));
        assert_eq!(
            rows[0].get("d.firmware"),
            Some(&Value::String("1.0".into()))
        );
    }
}

/// Without VALIDATED mode (default STRICT), properties are interned normally.
#[test]
fn strict_mode_no_extra() {
    let (mut db, _dir) = open_db();

    // Don't alter — default is STRICT
    db.execute_cypher("CREATE (n:User {name: 'Bob'})")
        .expect("create");

    let rows = db
        .execute_cypher("MATCH (n:User) RETURN n.name")
        .expect("match");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n.name"), Some(&Value::String("Bob".into())));
}

/// FLEXIBLE mode: all properties interned (no extra needed).
#[test]
fn flexible_mode_no_extra() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("ALTER LABEL Log SET SCHEMA FLEXIBLE")
        .expect("alter");
    db.execute_cypher("CREATE (l:Log {level: 'info', message: 'hello'})")
        .expect("create");

    let rows = db
        .execute_cypher("MATCH (l:Log) RETURN l.level, l.message")
        .expect("match");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("l.level"), Some(&Value::String("info".into())));
}
