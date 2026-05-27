//! Backup / restore integrity tests.
//!
//! The existing `binary_roundtrip` / `json_roundtrip` tests in
//! `crates/coordinode-embed/src/backup/mod.rs` only assert restored
//! *counts* — they don't catch property drops, edge corruption, or
//! label loss. This module adds end-to-end equality tests: build a
//! rich graph in `db1`, export, restore into a fresh `db2`, then
//! query `db2` with Cypher and compare each field against the
//! original.
//!
//! All tests share the same shape:
//! 1. Open `db1` in a tempdir, seed via Cypher.
//! 2. `export_binary` (or `_json`) into a `Vec<u8>` against a
//!    consistent snapshot.
//! 3. Open a fresh `db2` in a separate tempdir.
//! 4. `restore_binary` into `db2.engine()`, then install the
//!    returned `FieldInterner` into `db2` via `interner_arc()` —
//!    without this swap, the restored payload's interned property /
//!    label ids reference strings the fresh interner has never seen,
//!    so any Cypher query against `db2` would read garbage.
//! 5. Run MATCH queries on `db2`, assert each property / label /
//!    edge endpoint matches what `db1` had.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::backup::{export, restore};
use coordinode_embed::Database;

/// Build `(db2, _tempdir_keepalive)` from `db1`'s binary dump with the
/// restored interner properly installed. The tempdir handle must stay
/// alive (held by the caller) for the duration of the test — dropping
/// it removes the on-disk state of `db2`.
fn dump_restore_binary(db1: &Database) -> (Database, tempfile::TempDir) {
    let mut buf = Vec::new();
    let snapshot = db1.engine().snapshot();
    export::export_binary(db1.engine(), &db1.interner(), 1, &snapshot, &mut buf)
        .expect("export_binary");

    let dir2 = tempfile::tempdir().expect("tempdir for db2");
    let db2 = Database::open(dir2.path()).expect("open db2");
    let mut cursor = std::io::Cursor::new(&buf);
    let (_stats, restored_interner) =
        restore::restore_binary(db2.engine(), &mut cursor).expect("restore_binary");

    // Install the restored interner so Cypher queries against db2 can
    // resolve the property / label ids in the restored payload.
    if let Some(interner) = restored_interner {
        let arc = db2.interner_arc();
        *arc.write() = interner;
    }

    (db2, dir2)
}

#[test]
fn node_properties_survive_binary_roundtrip() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();
    db1.execute_cypher("CREATE (n:User {name: 'Alice', age: 30, height: 1.65, vip: true})")
        .unwrap();

    let (mut db2, _keep_dir2) = dump_restore_binary(&db1);

    let rows = db2
        .execute_cypher(
            "MATCH (n:User {name: 'Alice'}) RETURN n.name AS name, n.age AS age, \
             n.height AS height, n.vip AS vip",
        )
        .expect("MATCH on restored db");
    assert_eq!(rows.len(), 1, "exactly one Alice node should restore");
    let r = &rows[0];
    assert_eq!(r.get("name"), Some(&Value::String("Alice".into())));
    assert_eq!(r.get("age"), Some(&Value::Int(30)));
    assert_eq!(r.get("height"), Some(&Value::Float(1.65)));
    assert_eq!(r.get("vip"), Some(&Value::Bool(true)));
}

#[test]
fn edge_endpoints_and_props_survive_binary_roundtrip() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();
    db1.execute_cypher("CREATE (a:User {name: 'Alice'})")
        .unwrap();
    db1.execute_cypher("CREATE (b:User {name: 'Bob'})").unwrap();
    db1.execute_cypher(
        "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) \
         CREATE (a)-[:FOLLOWS {since: 2020, weight: 0.5}]->(b)",
    )
    .unwrap();

    let (mut db2, _keep_dir2) = dump_restore_binary(&db1);

    let rows = db2
        .execute_cypher(
            "MATCH (a:User)-[r:FOLLOWS]->(b:User) \
             RETURN a.name AS src, b.name AS dst, r.since AS since, r.weight AS weight",
        )
        .expect("MATCH edge on restored db");
    assert_eq!(rows.len(), 1, "exactly one FOLLOWS edge should restore");
    let r = &rows[0];
    assert_eq!(r.get("src"), Some(&Value::String("Alice".into())));
    assert_eq!(r.get("dst"), Some(&Value::String("Bob".into())));
    assert_eq!(r.get("since"), Some(&Value::Int(2020)));
    assert_eq!(r.get("weight"), Some(&Value::Float(0.5)));
}

#[test]
fn multi_label_node_survives_binary_roundtrip() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();
    // Two labels on one node — User AND Admin.
    db1.execute_cypher("CREATE (n:User:Admin {name: 'root'})")
        .unwrap();

    let (mut db2, _keep_dir2) = dump_restore_binary(&db1);

    // Restored node must match both labels independently.
    let by_user = db2
        .execute_cypher("MATCH (n:User {name: 'root'}) RETURN n.name AS name")
        .expect("MATCH :User");
    assert_eq!(by_user.len(), 1, "User label must survive");
    assert_eq!(by_user[0].get("name"), Some(&Value::String("root".into())));

    let by_admin = db2
        .execute_cypher("MATCH (n:Admin {name: 'root'}) RETURN n.name AS name")
        .expect("MATCH :Admin");
    assert_eq!(by_admin.len(), 1, "Admin label must survive");
    assert_eq!(by_admin[0].get("name"), Some(&Value::String("root".into())));
}

#[test]
fn vector_property_survives_binary_roundtrip() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();
    // 4-dim vector literal — the property comes back as `Value::Array(Float)`
    // because the label has no schema declaring this column as VECTOR.
    // A schema-typed VECTOR column would round-trip through `Value::Vector`;
    // both paths are interchangeable from a data-preservation standpoint.
    db1.execute_cypher("CREATE (n:Doc {title: 'paper', embedding: [0.1, 0.2, 0.3, 0.4]})")
        .unwrap();

    let (mut db2, _keep_dir2) = dump_restore_binary(&db1);

    let rows = db2
        .execute_cypher("MATCH (n:Doc {title: 'paper'}) RETURN n.embedding AS v")
        .expect("MATCH vector on restored db");
    assert_eq!(rows.len(), 1, "vector node must restore");
    match rows[0].get("v") {
        Some(Value::Array(elems)) => {
            assert_eq!(elems.len(), 4, "dim preserved");
            let floats: Vec<f64> = elems
                .iter()
                .map(|v| match v {
                    Value::Float(f) => *f,
                    other => panic!("expected Float element, got {other:?}"),
                })
                .collect();
            assert!((floats[0] - 0.1).abs() < 1e-6);
            assert!((floats[1] - 0.2).abs() < 1e-6);
            assert!((floats[2] - 0.3).abs() < 1e-6);
            assert!((floats[3] - 0.4).abs() < 1e-6);
        }
        Some(Value::Vector(v)) => {
            // Alternative path if a future schema enforcement classifies
            // the column as VECTOR — also acceptable, equivalent content.
            assert_eq!(v.len(), 4);
            assert!((v[0] - 0.1).abs() < 1e-6);
            assert!((v[1] - 0.2).abs() < 1e-6);
            assert!((v[2] - 0.3).abs() < 1e-6);
            assert!((v[3] - 0.4).abs() < 1e-6);
        }
        other => panic!("expected Array or Vector, got {other:?}"),
    }
}

#[test]
fn nested_document_survives_binary_roundtrip() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();
    // Map-typed property: a nested document subtree on the node.
    db1.execute_cypher(
        "CREATE (n:Profile {handle: 'alice', \
         meta: {tier: 'gold', joined: 2020, prefs: {theme: 'dark', notify: true}}})",
    )
    .unwrap();

    let (mut db2, _keep_dir2) = dump_restore_binary(&db1);

    let rows = db2
        .execute_cypher("MATCH (n:Profile {handle: 'alice'}) RETURN n.meta AS m")
        .expect("MATCH nested document on restored db");
    assert_eq!(rows.len(), 1, "profile must restore");
    // Nested map property comes back as `Value::Document(rmpv::Value)`
    // because anything beyond a flat scalar property gets the document
    // path. rmpv::Value serialises cleanly to JSON, which we compare
    // structurally against the expected shape.
    let doc = match rows[0].get("m") {
        Some(Value::Document(d)) => d,
        other => panic!("expected Document for n.meta, got {other:?}"),
    };
    let actual_json: serde_json::Value =
        serde_json::to_value(doc).expect("rmpv::Value → serde_json::Value");
    let expected_json = serde_json::json!({
        "tier": "gold",
        "joined": 2020,
        "prefs": { "theme": "dark", "notify": true },
    });
    assert_eq!(
        actual_json, expected_json,
        "nested document subtree must round-trip byte-exact"
    );
}

#[test]
fn temporal_node_survives_binary_roundtrip() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();

    // Declare a temporal node type (R172a contract — valid_from must
    // be present on CREATE; valid_to optional). `valid_from`/`valid_to`
    // are typed INT here, matching the working pattern in crud.rs:4687.
    // Writing TIMESTAMP literals from raw Cypher needs a `datetime(...)`
    // wrapper that's orthogonal to what we're testing (data integrity
    // of the bitemporal storage path).
    db1.execute_cypher(
        "CREATE NODE TYPE Person TEMPORAL WITH \
         (name: STRING NOT NULL, valid_from: INT NOT NULL, valid_to: INT)",
    )
    .unwrap();

    // Two temporal versions of the same logical entity — distinct
    // valid_from values so per-version storage (R172b) keeps them
    // separately.
    db1.execute_cypher(
        "CREATE (:Person {name: 'Alice', valid_from: 1577836800000, valid_to: 1640995200000})",
    )
    .unwrap();
    db1.execute_cypher("CREATE (:Person {name: 'Alice', valid_from: 1640995200000})")
        .unwrap();

    let (mut db2, _keep_dir2) = dump_restore_binary(&db1);

    let rows = db2
        .execute_cypher(
            "MATCH (n:Person {name: 'Alice'}) \
             RETURN n.valid_from AS vf, n.valid_to AS vt",
        )
        .expect("MATCH temporal on restored db");
    assert_eq!(rows.len(), 2, "both temporal versions must restore");

    // Both versions present, distinct by valid_from. Sort by vf for
    // deterministic assertions.
    let mut versions: Vec<_> = rows
        .into_iter()
        .map(|r| (r.get("vf").cloned(), r.get("vt").cloned()))
        .collect();
    versions.sort_by_key(|(vf, _)| match vf {
        Some(Value::Int(t)) => *t,
        _ => i64::MAX,
    });

    // Earlier version: closed interval.
    assert_eq!(versions[0].0, Some(Value::Int(1577836800000)));
    assert_eq!(versions[0].1, Some(Value::Int(1640995200000)));
    // Later version: open interval (valid_to is null / absent).
    assert_eq!(versions[1].0, Some(Value::Int(1640995200000)));
    assert!(
        matches!(versions[1].1, Some(Value::Null) | None),
        "later valid_to must be null / absent, got {:?}",
        versions[1].1,
    );
}
