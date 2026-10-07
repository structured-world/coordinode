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
//! 4. `Database::restore` into `db2`, which publishes the dump's bindings
//!    before the records encoded with them are written.
//! 5. Run MATCH queries on `db2`, assert each property / label /
//!    edge endpoint matches what `db1` had.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use coordinode_embed::backup::restore::RestoreOptions;
use coordinode_embed::backup::{BackupFormat, export};

/// Build `(db2, _tempdir_keepalive)` from `db1`'s binary dump. The tempdir
/// handle must stay alive (held by the caller) for the duration of the
/// test: dropping it removes the on-disk state of `db2`.
fn dump_restore_binary(db1: &Database) -> (Database, tempfile::TempDir) {
    let mut buf = Vec::new();
    let snapshot = db1.engine().snapshot();
    export::export_binary(
        db1.engine(),
        &db1.interner().expect("dictionary"),
        1,
        &snapshot,
        &mut buf,
    )
    .expect("export_binary");

    let dir2 = tempfile::tempdir().expect("tempdir for db2");
    let db2 = Database::open(dir2.path()).expect("open db2");
    db2.restore(BackupFormat::Binary, &buf, &RestoreOptions::default())
        .expect("restore binary");

    (db2, dir2)
}

/// Round-trip through the cypher dump format: export to a cypher string,
/// restore it into a fresh db. Mirrors the binary path
/// but exercises the text format's parser.
fn dump_restore_cypher(db1: &Database) -> (Database, tempfile::TempDir) {
    let mut buf = Vec::new();
    let snapshot = db1.engine().snapshot();
    export::export_cypher(
        db1.engine(),
        &db1.interner().expect("dictionary"),
        1,
        &snapshot,
        &mut buf,
    )
    .expect("export_cypher");

    let dir2 = tempfile::tempdir().expect("tempdir for db2");
    let db2 = Database::open(dir2.path()).expect("open db2");
    db2.restore(BackupFormat::Cypher, &buf, &RestoreOptions::default())
        .expect("restore cypher");
    (db2, dir2)
}

#[test]
fn node_and_edge_props_survive_cypher_roundtrip() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();
    db1.execute_cypher("CREATE (a:User:Admin {name: 'Alice', age: 30, vip: true, score: 1.5})")
        .unwrap();
    db1.execute_cypher("CREATE (b:User {name: 'Bob'})").unwrap();
    db1.execute_cypher(
        "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) \
         CREATE (a)-[:FOLLOWS {since: 2020, weight: 0.5}]->(b)",
    )
    .unwrap();

    let (mut db2, _keep) = dump_restore_cypher(&db1);

    // Multi-label node + scalar props.
    let node = db2
        .execute_cypher(
            "MATCH (n:Admin {name: 'Alice'}) \
             RETURN n.age AS age, n.vip AS vip, n.score AS score",
        )
        .expect("MATCH restored node");
    assert_eq!(node.len(), 1, "Alice (multi-label) must restore");
    assert_eq!(node[0].get("age"), Some(&Value::Int(30)));
    assert_eq!(node[0].get("vip"), Some(&Value::Bool(true)));
    assert_eq!(node[0].get("score"), Some(&Value::Float(1.5)));

    // Edge endpoints + edge props.
    let edge = db2
        .execute_cypher(
            "MATCH (a:User)-[r:FOLLOWS]->(b:User) \
             RETURN a.name AS src, b.name AS dst, r.since AS since, r.weight AS weight",
        )
        .expect("MATCH restored edge");
    assert_eq!(edge.len(), 1, "FOLLOWS edge must restore");
    assert_eq!(edge[0].get("src"), Some(&Value::String("Alice".into())));
    assert_eq!(edge[0].get("dst"), Some(&Value::String("Bob".into())));
    assert_eq!(edge[0].get("since"), Some(&Value::Int(2020)));
    assert_eq!(edge[0].get("weight"), Some(&Value::Float(0.5)));
}

#[test]
fn string_with_special_chars_survives_cypher_roundtrip() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();
    // Commas, colons, braces and quotes inside a string value must not
    // confuse the property parser's top-level splitter.
    db1.execute_cypher("CREATE (n:Note {body: 'a, b: c {x} \"q\"', tag: 'plain'})")
        .unwrap();

    let (mut db2, _keep) = dump_restore_cypher(&db1);

    let rows = db2
        .execute_cypher("MATCH (n:Note {tag: 'plain'}) RETURN n.body AS body")
        .expect("MATCH note");
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("body"),
        Some(&Value::String("a, b: c {x} \"q\"".into())),
        "string with delimiters must survive the property parser"
    );
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

    // Declare a temporal node type (valid_from must be present on
    // CREATE; valid_to optional). `valid_from`/`valid_to` are typed INT
    // here, matching the temporal tests in crud.rs.
    // Writing TIMESTAMP literals from raw Cypher needs a `datetime(...)`
    // wrapper that's orthogonal to what we're testing (data integrity
    // of the bitemporal storage path).
    db1.execute_cypher(
        "CREATE NODE TYPE Person TEMPORAL WITH \
         (name: STRING NOT NULL, valid_from: INT NOT NULL, valid_to: INT)",
    )
    .unwrap();

    // Two temporal versions of the same logical entity — distinct
    // valid_from values so per-version storage keeps them
    // separately.
    db1.execute_cypher(
        "CREATE (:Person {name: 'Alice', valid_from: 1577836800000, valid_to: 1640995200000})",
    )
    .unwrap();
    db1.execute_cypher("CREATE (:Person {name: 'Alice', valid_from: 1640995200000})")
        .unwrap();

    let (db2, _keep_dir2) = dump_restore_binary(&db1);

    // Every stored version, not only the state valid now.
    let rows = super::helpers::temporal_versions(&db2, "Person");
    assert_eq!(rows.len(), 2, "both temporal versions must restore");

    // Both versions present, distinct by valid_from. Sort by vf for
    // deterministic assertions.
    let mut versions: Vec<_> = rows
        .into_iter()
        .map(|r| (r.get("valid_from").cloned(), r.get("valid_to").cloned()))
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

/// Round-trip through the json dump format.
fn dump_restore_json(db1: &Database) -> (Database, tempfile::TempDir) {
    let mut buf = Vec::new();
    let snapshot = db1.engine().snapshot();
    export::export_json(
        db1.engine(),
        &db1.interner().expect("dictionary"),
        1,
        &snapshot,
        &mut buf,
    )
    .expect("export_json");

    let dir2 = tempfile::tempdir().expect("tempdir for db2");
    let db2 = Database::open(dir2.path()).expect("open db2");
    db2.restore(BackupFormat::Json, &buf, &RestoreOptions::default())
        .expect("restore json");
    (db2, dir2)
}

/// The ids of every `Person`, by name.
fn person_ids(db: &mut Database) -> Vec<(String, i64)> {
    let mut rows: Vec<(String, i64)> = db
        .execute_cypher("MATCH (n:Person) RETURN n.name AS name, id(n) AS id")
        .expect("MATCH people")
        .into_iter()
        .map(|r| match (r.get("name"), r.get("id")) {
            (Some(Value::String(name)), Some(Value::Int(id))) => (name.clone(), *id),
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    rows.sort();
    rows
}

/// A node created after a restore gets an identifier no restored node has,
/// and no restored node is overwritten. A restore writes each node under its
/// original identifier; a target that still allocates from the start of the
/// sequence space would hand the first restored identifier out again, and the
/// new node, a put by key, would replace the restored one.
#[test]
fn a_node_created_after_a_restore_takes_a_fresh_identifier() {
    for (format, dump_restore) in [
        ("json", dump_restore_json as fn(&Database) -> _),
        ("binary", dump_restore_binary),
        ("cypher", dump_restore_cypher),
    ] {
        let dir1 = tempfile::tempdir().unwrap();
        let mut db1 = Database::open(dir1.path()).unwrap();
        for name in ["Ada", "Bea", "Cal"] {
            db1.execute_cypher(&format!("CREATE (:Person {{name: '{name}'}})"))
                .unwrap();
        }
        let restored = person_ids(&mut db1);

        let (mut db2, _keep) = dump_restore(&db1);
        assert_eq!(person_ids(&mut db2), restored, "{format}: ids kept");

        db2.execute_cypher("CREATE (:Person {name: 'Dan'})")
            .unwrap();
        let after = person_ids(&mut db2);
        assert_eq!(after.len(), 4, "{format}: no restored node was replaced");
        for kept in &restored {
            assert!(after.contains(kept), "{format}: {kept:?} survived");
        }
        let dan = after.iter().find(|(n, _)| n == "Dan").expect("Dan").1;
        assert!(
            restored.iter().all(|(_, id)| *id != dan),
            "{format}: the new node took a restored identifier {dan}"
        );
    }
}

/// Every format CoordiNode writes and reads back.
const OWN_FORMATS: [BackupFormat; 3] = [
    BackupFormat::Json,
    BackupFormat::Cypher,
    BackupFormat::Binary,
];

/// A dump of `db` in one of the formats CoordiNode writes.
fn dump_of(db: &Database, format: BackupFormat) -> Vec<u8> {
    let interner = db.interner().expect("dictionary");
    let snapshot = db.engine().snapshot();
    let mut buf = Vec::new();
    match format {
        BackupFormat::Json => export::export_json(db.engine(), &interner, 1, &snapshot, &mut buf),
        BackupFormat::Cypher => {
            export::export_cypher(db.engine(), &interner, 1, &snapshot, &mut buf)
        }
        _ => export::export_binary(db.engine(), &interner, 1, &snapshot, &mut buf),
    }
    .expect("export");
    buf
}

/// Every instance of a temporal edge between one pair comes back with its own
/// discriminator: the text formats used to key all instances of a pair alike,
/// so the restore kept one and dropped the others.
#[test]
fn every_temporal_edge_instance_survives_every_format() {
    const EDGE_TYPE: &str = "CREATE EDGE TYPE WORKS_AT TEMPORAL \
                             WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP, role: STRING)";
    let instances = |db: &mut Database| {
        let mut rows: Vec<(Option<Value>, Option<Value>)> = db
            .execute_cypher(
                "MATCH (:Person {name: 'B'})-[r:WORKS_AT]->(:Co {name: 'Acme'}) \
                 RETURN r.role AS role, r.valid_from AS vf",
            )
            .unwrap()
            .into_iter()
            .map(|r| (r.get("role").cloned(), r.get("vf").cloned()))
            .collect();
        rows.sort_by_key(|(role, _)| format!("{role:?}"));
        rows
    };

    for format in OWN_FORMATS {
        let dir1 = tempfile::tempdir().unwrap();
        let mut db1 = Database::open(dir1.path()).unwrap();
        db1.execute_cypher(EDGE_TYPE).unwrap();
        db1.execute_cypher("CREATE (:Person {name: 'B'}), (:Co {name: 'Acme'})")
            .unwrap();
        for (from, to, role) in [(1000, 2000, "SWE"), (2000, 3000, "Staff")] {
            db1.execute_cypher(&format!(
                "MATCH (b:Person {{name: 'B'}}), (c:Co {{name: 'Acme'}}) \
                 CREATE (b)-[:WORKS_AT {{valid_from: {from}, valid_to: {to}, role: '{role}'}}]->(c)"
            ))
            .unwrap();
        }
        let expected = instances(&mut db1);
        assert_eq!(expected.len(), 2, "the source holds both instances");

        let dir2 = tempfile::tempdir().unwrap();
        let mut db2 = Database::open(dir2.path()).unwrap();
        // A cypher dump carries data only; the others bring their schema.
        if format == BackupFormat::Cypher {
            db2.execute_cypher(EDGE_TYPE).unwrap();
        }
        db2.restore(format, &dump_of(&db1, format), &RestoreOptions::default())
            .unwrap();
        assert_eq!(instances(&mut db2), expected, "{format:?}");
    }
}

/// What identifies an edge travels with the schema a dump brings: a
/// categorical discriminator, an independent temporal one and the
/// start-identified shorthand come back as they were resolved.
#[test]
fn edge_discriminators_survive_every_format_with_a_schema() {
    for format in [BackupFormat::Json, BackupFormat::Binary] {
        let dir1 = tempfile::tempdir().unwrap();
        let mut db1 = Database::open(dir1.path()).unwrap();
        for statement in [
            "CREATE EDGE TYPE KNOWS WITH (context: STRING NOT NULL) DISCRIMINATED BY (context)",
            "CREATE EDGE TYPE ASSERTS TEMPORAL DISCRIMINATED BY (key) WITH (key: BLOB NOT NULL)",
            "CREATE EDGE TYPE WORKS_AT TEMPORAL",
        ] {
            db1.execute_cypher(statement).unwrap();
        }
        let dir2 = tempfile::tempdir().unwrap();
        let db2 = Database::open(dir2.path()).unwrap();
        db2.restore(format, &dump_of(&db1, format), &RestoreOptions::default())
            .unwrap();

        assert_eq!(
            db2.edge_type_schemas().unwrap(),
            db1.edge_type_schemas().unwrap(),
            "{format:?}"
        );
        let resolved: Vec<_> = db2
            .edge_type_schemas()
            .unwrap()
            .into_iter()
            .map(|s| (s.name.clone(), s.discriminator().map(|d| d.column.clone())))
            .collect();
        assert_eq!(
            resolved,
            [
                ("ASSERTS".to_string(), Some("key".to_string())),
                ("KNOWS".to_string(), Some("context".to_string())),
                ("WORKS_AT".to_string(), Some("valid_from".to_string())),
            ],
            "{format:?}"
        );
    }
}

/// Every version of a temporal node comes back in every format, under its
/// own identifier and valid_from: the text formats used to skip versions.
#[test]
fn every_temporal_node_version_survives_every_format() {
    for format in OWN_FORMATS {
        let dir1 = tempfile::tempdir().unwrap();
        let mut db1 = Database::open(dir1.path()).unwrap();
        db1.execute_cypher(
            "CREATE NODE TYPE Person TEMPORAL WITH \
             (name: STRING NOT NULL, valid_from: INT NOT NULL, valid_to: INT)",
        )
        .unwrap();
        db1.execute_cypher("CREATE (:Person {name: 'Alice', valid_from: 1000, valid_to: 2000})")
            .unwrap();
        db1.execute_cypher("CREATE (:Person {name: 'Alice', valid_from: 2000})")
            .unwrap();
        // Every stored version, not only the state valid now.
        let versions = |db: &Database| {
            super::helpers::temporal_versions(db, "Person")
                .into_iter()
                .map(|r| {
                    (
                        r.get("id").cloned(),
                        r.get("valid_from").cloned(),
                        r.get("valid_to").cloned(),
                        r.get("name").cloned(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let expected = versions(&db1);
        assert_eq!(expected.len(), 2);

        let dir2 = tempfile::tempdir().unwrap();
        let db2 = Database::open(dir2.path()).unwrap();
        db2.restore(format, &dump_of(&db1, format), &RestoreOptions::default())
            .unwrap();
        assert_eq!(versions(&db2), expected, "{format:?}");
    }
}

/// A restored database answers through its B-tree indexes and enforces their
/// constraints for the restored nodes: the load writes node records directly,
/// so the indexes are built from them before the restore returns, whether the
/// dump brought the definition or the target declared it.
#[test]
fn indexes_cover_the_restored_nodes_in_every_format() {
    const INDEX: &str = "CREATE UNIQUE INDEX u_email ON :U(email)";
    for format in OWN_FORMATS {
        let dir1 = tempfile::tempdir().unwrap();
        let mut db1 = Database::open(dir1.path()).unwrap();
        db1.execute_cypher(INDEX).unwrap();
        db1.execute_cypher("CREATE (:U {email: 'a@x'}), (:U {email: 'b@x'})")
            .unwrap();

        let dir2 = tempfile::tempdir().unwrap();
        let mut db2 = Database::open(dir2.path()).unwrap();
        // A cypher dump carries data only; the others bring the index.
        if format == BackupFormat::Cypher {
            db2.execute_cypher(INDEX).unwrap();
        }
        db2.restore(format, &dump_of(&db1, format), &RestoreOptions::default())
            .unwrap();

        let found = db2
            .execute_cypher("MATCH (u:U {email: 'a@x'}) RETURN u.email AS email")
            .unwrap();
        assert_eq!(
            found.len(),
            1,
            "{format:?}: the index finds a restored node"
        );
        let refused = db2.execute_cypher("CREATE (:U {email: 'b@x'})");
        assert!(
            refused.is_err(),
            "{format:?}: a restored value is held by the unique index"
        );
        db2.execute_cypher("CREATE (:U {email: 'c@x'})")
            .unwrap_or_else(|e| panic!("{format:?}: a new value is free: {e}"));
    }
}

/// A `json` dump brings its schema, and a target that declares one of its
/// types differently is refused before anything is written: reading the dump
/// under the target's declaration would give its records another meaning.
#[test]
fn a_json_restore_refuses_a_target_that_declares_its_types_otherwise() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();
    db1.execute_cypher(
        "CREATE EDGE TYPE WORKS_AT TEMPORAL \
         WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP, role: STRING)",
    )
    .unwrap();
    db1.execute_cypher("CREATE (:Person {name: 'B'}), (:Co {name: 'Acme'})")
        .unwrap();
    let dump = dump_of(&db1, BackupFormat::Json);

    let dir2 = tempfile::tempdir().unwrap();
    let mut db2 = Database::open(dir2.path()).unwrap();
    db2.execute_cypher("CREATE EDGE TYPE WORKS_AT WITH (role: STRING)")
        .unwrap();
    let refused = db2
        .restore(BackupFormat::Json, &dump, &RestoreOptions::default())
        .unwrap_err();
    assert!(
        matches!(
            refused,
            coordinode_embed::backup::restore::RestoreError::SchemaMismatch(_)
        ),
        "got {refused:?}"
    );
    let rows = db2.execute_cypher("MATCH (n) RETURN n").unwrap();
    assert!(rows.is_empty(), "nothing written: {rows:?}");
}

/// A vector index holds every restored vector before any reopen, in every
/// format: the restore builds it from the stored nodes, as an open does.
#[test]
fn a_vector_index_holds_the_restored_vectors_in_every_format() {
    const INDEX: &str = "CREATE VECTOR INDEX item_emb ON :Item(emb) OPTIONS {metric: \"l2\"}";
    for format in OWN_FORMATS {
        let dir1 = tempfile::tempdir().unwrap();
        let mut db1 = Database::open(dir1.path()).unwrap();
        db1.execute_cypher(INDEX).unwrap();
        db1.execute_cypher(
            "CREATE (:Item {emb: [1.0, 0.0]}), (:Item {emb: [0.0, 1.0]}), (:Item {emb: [0.5, 0.5]})",
        )
        .unwrap();

        let dir2 = tempfile::tempdir().unwrap();
        let mut db2 = Database::open(dir2.path()).unwrap();
        if format == BackupFormat::Cypher {
            db2.execute_cypher(INDEX).unwrap();
        }
        db2.restore(format, &dump_of(&db1, format), &RestoreOptions::default())
            .unwrap();
        let held = db2
            .vector_index_registry()
            .get("Item", "emb")
            .map(|hnsw| hnsw.read().expect("hnsw lock").len());
        assert_eq!(held, Some(3), "{format:?}");
    }
}

/// A text index answers for the restored nodes before any reopen, in every
/// format: its documents come from the stored nodes, as on open.
#[test]
fn a_text_index_covers_the_restored_nodes_in_every_format() {
    const INDEX: &str = "CREATE TEXT INDEX article_body ON :Article(body)";
    for format in OWN_FORMATS {
        let dir1 = tempfile::tempdir().unwrap();
        let mut db1 = Database::open(dir1.path()).unwrap();
        db1.execute_cypher(INDEX).unwrap();
        db1.execute_cypher(
            "CREATE (:Article {body: 'rust storage engines'}), (:Article {body: 'gardening'})",
        )
        .unwrap();

        let dir2 = tempfile::tempdir().unwrap();
        let mut db2 = Database::open(dir2.path()).unwrap();
        if format == BackupFormat::Cypher {
            db2.execute_cypher(INDEX).unwrap();
        }
        db2.restore(format, &dump_of(&db1, format), &RestoreOptions::default())
            .unwrap();
        let rows = db2
            .execute_cypher(
                "MATCH (a:Article) WHERE text_match(a.body, 'rust') RETURN a.body AS body",
            )
            .unwrap();
        assert_eq!(rows.len(), 1, "{format:?}: {rows:?}");
    }
}

/// Restored data that breaks a unique index the target declared is reported
/// by the restore, and the index is set aside as failed rather than left to
/// answer lookups it holds no entries for: a read still finds every
/// restored node.
#[test]
fn restored_data_breaking_a_unique_index_is_reported() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();
    db1.execute_cypher("CREATE (:U {email: 'a@x'}), (:U {email: 'a@x'})")
        .unwrap();

    let dir2 = tempfile::tempdir().unwrap();
    let mut db2 = Database::open(dir2.path()).unwrap();
    db2.execute_cypher("CREATE UNIQUE INDEX u_email ON :U(email)")
        .unwrap();
    let refused = db2
        .restore(
            BackupFormat::Json,
            &dump_of(&db1, BackupFormat::Json),
            &RestoreOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(
            refused,
            coordinode_embed::backup::restore::RestoreError::Indexes(_)
        ),
        "got {refused:?}"
    );
    let found = db2
        .execute_cypher("MATCH (u:U {email: 'a@x'}) RETURN u")
        .unwrap();
    assert_eq!(found.len(), 2, "a read finds both restored nodes");
}

/// An export reads adjacency, edge bodies and nodes at one snapshot: an edge
/// written after the snapshot is in none of them, so no dump holds half of it.
#[test]
fn an_edge_written_after_the_export_snapshot_is_not_exported() {
    for format in OWN_FORMATS {
        let dir1 = tempfile::tempdir().unwrap();
        let mut db1 = Database::open(dir1.path()).unwrap();
        db1.execute_cypher("CREATE (:Person {name: 'A'}), (:Person {name: 'B'})")
            .unwrap();
        let snapshot = db1.engine().snapshot();
        db1.execute_cypher(
            "MATCH (a:Person {name: 'A'}), (b:Person {name: 'B'}) \
             CREATE (a)-[:KNOWS {since: 2020}]->(b)",
        )
        .unwrap();

        let interner = db1.interner().unwrap();
        let mut buf = Vec::new();
        let stats = match format {
            BackupFormat::Json => {
                export::export_json(db1.engine(), &interner, 1, &snapshot, &mut buf)
            }
            BackupFormat::Cypher => {
                export::export_cypher(db1.engine(), &interner, 1, &snapshot, &mut buf)
            }
            _ => export::export_binary(db1.engine(), &interner, 1, &snapshot, &mut buf),
        }
        .unwrap();
        assert_eq!(stats.nodes, 2, "{format:?}");
        assert_eq!(stats.edges, 0, "{format:?}: the later edge exported");

        let dir2 = tempfile::tempdir().unwrap();
        let mut db2 = Database::open(dir2.path()).unwrap();
        db2.restore(format, &buf, &RestoreOptions::default())
            .unwrap();
        let rows = db2.execute_cypher("MATCH ()-[r]->() RETURN r").unwrap();
        assert!(rows.is_empty(), "{format:?}: restored {rows:?}");
    }
}

/// Restored data that breaks a constraint the target declared is reported by
/// the restore, naming the node, instead of landing past the check every
/// other write meets.
#[test]
fn restored_data_breaking_a_constraint_is_reported() {
    let dir1 = tempfile::tempdir().unwrap();
    let mut db1 = Database::open(dir1.path()).unwrap();
    db1.execute_cypher("CREATE (:User {email: 'a@x'})").unwrap();

    let dir2 = tempfile::tempdir().unwrap();
    let mut db2 = Database::open(dir2.path()).unwrap();
    db2.execute_cypher("CREATE CONSTRAINT FOR (u:User) REQUIRE u.name IS NOT NULL")
        .unwrap();
    let refused = db2
        .restore(
            BackupFormat::Cypher,
            &dump_of(&db1, BackupFormat::Cypher),
            &RestoreOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(
            refused,
            coordinode_embed::backup::restore::RestoreError::Constraints(_)
        ),
        "got {refused:?}"
    );
}

/// Constraints travel with a dump that brings its schema: the restored
/// database holds each one under its name, in the state it had, with the
/// unique index it owns, and enforces them. A cypher dump carries data only,
/// so its target declares them and the restored data is held to them.
#[test]
fn constraints_survive_a_restore_in_every_format() {
    use coordinode_core::schema::definition::ConstraintState;
    use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
    use coordinode_query::index::definition::PartialFilter;
    const CONSTRAINTS: [&str; 4] = [
        "CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE",
        "CREATE CONSTRAINT user_name FOR (u:User) REQUIRE u.name IS NOT NULL",
        "CREATE CONSTRAINT user_age FOR (u:User) REQUIRE u.age IS :: INTEGER",
        "CREATE UNIQUE INDEX active_nick ON :User(nick) WHERE n.active = true",
    ];
    for format in OWN_FORMATS {
        let dir1 = tempfile::tempdir().unwrap();
        let mut db1 = Database::open(dir1.path()).unwrap();
        for statement in CONSTRAINTS {
            db1.execute_cypher(statement).unwrap();
        }
        db1.execute_cypher(
            "CREATE (:User {email: 'a@x', name: 'a', age: 1, nick: 'n', active: true})",
        )
        .unwrap();

        let dir2 = tempfile::tempdir().unwrap();
        let mut db2 = Database::open(dir2.path()).unwrap();
        if format == BackupFormat::Cypher {
            for statement in CONSTRAINTS {
                db2.execute_cypher(statement).unwrap();
            }
        }
        db2.restore(format, &dump_of(&db1, format), &RestoreOptions::default())
            .unwrap_or_else(|e| panic!("{format:?}: restore: {e}"));

        let schema = LocalSchemaStore::new(db2.engine())
            .load_label("User")
            .unwrap()
            .unwrap_or_else(|| panic!("{format:?}: the schema is restored"));
        for name in ["user_email", "user_name", "user_age", "active_nick"] {
            assert_eq!(
                schema.constraint(name).map(|c| c.state),
                Some(ConstraintState::Active),
                "{format:?}: {name}"
            );
        }
        assert_eq!(
            schema
                .constraint("active_nick")
                .and_then(|c| c.scope.clone()),
            Some(PartialFilter::PropertyEqualsBool {
                property: "active".into(),
                value: true,
            }),
            "{format:?}: the scope is restored"
        );
        db2.execute_cypher("CREATE (:User {email: 'n1@x', name: 'n1', nick: 'n', active: true})")
            .expect_err("the restored value is held in scope");
        db2.execute_cypher("CREATE (:User {email: 'n2@x', name: 'n2', nick: 'n', active: false})")
            .unwrap_or_else(|e| panic!("{format:?}: out of scope: {e}"));
        db2.execute_cypher("CREATE (:User {email: 'a@x', name: 'b'})")
            .expect_err("the restored value is held by the unique constraint");
        db2.execute_cypher("CREATE (:User {email: 'b@x'})")
            .expect_err("the name is required");
        db2.execute_cypher("CREATE (:User {email: 'c@x', name: 'c', age: 'old'})")
            .expect_err("the age is an integer");
        db2.execute_cypher("DROP CONSTRAINT user_email")
            .unwrap_or_else(|e| panic!("{format:?}: the name is restored: {e}"));
        db2.execute_cypher("CREATE (:User {email: 'a@x', name: 'b'})")
            .unwrap_or_else(|e| panic!("{format:?}: its index went with it: {e}"));
    }
}
