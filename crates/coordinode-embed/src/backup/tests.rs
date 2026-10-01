use super::BackupFormat;
use super::export;
use super::restore::{self, RestoreOptions};
use crate::Database;

/// Options for a binary restore that may override its compatibility gates.
fn forced() -> RestoreOptions<'static> {
    RestoreOptions {
        force: true,
        ..Default::default()
    }
}

/// The encoded form of a dictionary with no bindings.
fn empty_dictionary() -> Vec<u8> {
    coordinode_core::graph::intern::FieldInterner::new()
        .to_bytes()
        .unwrap()
}

#[test]
fn json_export_nodes_and_edges() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    db.execute_cypher("CREATE (a:User {name: 'Alice', age: 30})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {name: 'Bob', age: 25})")
        .unwrap();
    db.execute_cypher(
        "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) CREATE (a)-[:FOLLOWS]->(b)",
    )
    .unwrap();

    let mut buf = Vec::new();
    let snapshot = db.engine().snapshot();
    let stats =
        export::export_json(db.engine(), &db.interner().unwrap(), 1, &snapshot, &mut buf).unwrap();

    assert_eq!(stats.nodes, 2, "should export 2 nodes");
    assert_eq!(stats.edges, 1, "should export 1 edge");

    let output = String::from_utf8(buf).unwrap();
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 3, "2 nodes + 1 edge = 3 lines");

    // Parse first node line
    let node_json: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(node_json["type"], "node");
    assert!(node_json["id"].is_number());
    assert!(node_json["labels"].is_array());
}

#[test]
fn cypher_export_format() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    db.execute_cypher("CREATE (a:User {name: 'Alice'})")
        .unwrap();

    let mut buf = Vec::new();
    let snapshot = db.engine().snapshot();
    let stats = export::export_cypher(db.engine(), &db.interner().unwrap(), 1, &snapshot, &mut buf)
        .unwrap();

    assert_eq!(stats.nodes, 1);
    let output = String::from_utf8(buf).unwrap();
    assert!(
        output.contains("CREATE (n"),
        "should contain CREATE statement"
    );
    assert!(output.contains(":User"), "should contain label");
}

#[test]
fn binary_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    db.execute_cypher("CREATE (a:User {name: 'Alice', age: 30})")
        .unwrap();
    db.execute_cypher("CREATE (b:User {name: 'Bob'})").unwrap();
    db.execute_cypher(
        "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) CREATE (a)-[:FOLLOWS]->(b)",
    )
    .unwrap();

    // Export
    let mut buf = Vec::new();
    let snapshot = db.engine().snapshot();
    let export_stats =
        export::export_binary(db.engine(), &db.interner().unwrap(), 1, &snapshot, &mut buf)
            .unwrap();
    assert!(export_stats.nodes >= 2);

    // Restore to new database
    let dir2 = tempfile::tempdir().unwrap();
    let mut db2 = Database::open(dir2.path()).unwrap();

    let restore_stats = db2
        .restore(BackupFormat::Binary, &buf, &RestoreOptions::default())
        .unwrap();

    assert_eq!(restore_stats.nodes, export_stats.nodes);
    // The records are read through the bindings the dump carried.
    let rows = db2
        .execute_cypher("MATCH (n:User {name: 'Alice'}) RETURN n.age AS age")
        .unwrap();
    assert_eq!(
        rows[0].get("age"),
        Some(&coordinode_core::graph::types::Value::Int(30))
    );
}

/// A binary dump's records carry the source's ids, so the restore publishes
/// exactly those bindings, and refuses a target whose own bindings give the
/// same ids other meanings instead of reading the dump under the wrong names.
#[test]
fn binary_restore_keeps_the_dump_ids_and_refuses_contradicting_targets() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.execute_cypher("CREATE (:User {name: 'Alice', age: 30})")
        .unwrap();
    let mut buf = Vec::new();
    let snapshot = db.engine().snapshot();
    let source = db.interner().unwrap();
    export::export_binary(db.engine(), &source, 1, &snapshot, &mut buf).unwrap();

    let dir2 = tempfile::tempdir().unwrap();
    let db2 = Database::open(dir2.path()).unwrap();
    db2.restore(BackupFormat::Binary, &buf, &RestoreOptions::default())
        .unwrap();
    let restored = db2.interner().unwrap();
    for (name, id) in source.iter() {
        assert_eq!(restored.lookup(name), Some(id), "binding of {name}");
    }

    // A target that bound these names differently is refused.
    let dir3 = tempfile::tempdir().unwrap();
    let mut db3 = Database::open(dir3.path()).unwrap();
    db3.execute_cypher("CREATE (:Other {zzz: 1, age: 2, name: 'x'})")
        .unwrap();
    let refused = db3.restore(BackupFormat::Binary, &buf, &forced());
    assert!(
        refused.is_err(),
        "a contradicting target restored: {refused:?}"
    );
}

/// A restored database tells the planner how much it holds.
///
/// The node and label counts the cost estimator reads are counter rows
/// maintained by the write path, and a binary restore writes the rows
/// themselves rather than replaying the writes. Without a rebuild the
/// restored database reports an empty graph while holding a full one, and
/// every plan chosen on it is costed against zero.
#[test]
fn a_restored_database_reports_what_it_holds() {
    use coordinode_core::graph::stats::StorageStats;
    use coordinode_storage::engine::stats::StorageStatsComputer;

    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    db.execute_cypher("CREATE (:User {name: 'Alice'})").unwrap();
    db.execute_cypher("CREATE (:User {name: 'Bob'})").unwrap();
    db.execute_cypher("CREATE (:Post {title: 'Hello'})")
        .unwrap();

    let source = StorageStatsComputer::compute(db.engine()).unwrap();
    assert_eq!(source.total_node_count(), 3, "the source counts its nodes");
    assert_eq!(source.node_count_for_label("User"), Some(2));
    assert_eq!(source.node_count_for_label("Post"), Some(1));

    let mut buf = Vec::new();
    let snapshot = db.engine().snapshot();
    export::export_binary(db.engine(), &db.interner().unwrap(), 1, &snapshot, &mut buf).unwrap();

    let dir2 = tempfile::tempdir().unwrap();
    let db2 = Database::open(dir2.path()).unwrap();
    db2.restore(BackupFormat::Binary, &buf, &RestoreOptions::default())
        .unwrap();

    let restored = StorageStatsComputer::compute(db2.engine()).unwrap();
    assert_eq!(
        restored.total_node_count(),
        source.total_node_count(),
        "a restored database holds as many nodes as the one it came from"
    );
    assert_eq!(
        restored.node_count_for_label("User"),
        Some(2),
        "and as many of each label"
    );
    assert_eq!(restored.node_count_for_label("Post"), Some(1));
}

/// The same obligation on the JSON path, which writes its rows through the
/// typed node store rather than through the executor that stages the
/// counters, and so leaves them behind exactly as the binary path does.
#[test]
fn a_json_restore_also_reports_what_it_holds() {
    use coordinode_core::graph::stats::StorageStats;
    use coordinode_storage::engine::stats::StorageStatsComputer;

    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.execute_cypher("CREATE (:User {name: 'Alice'})").unwrap();
    db.execute_cypher("CREATE (:User {name: 'Bob'})").unwrap();

    let mut buf = Vec::new();
    let snapshot = db.engine().snapshot();
    export::export_json(db.engine(), &db.interner().unwrap(), 1, &snapshot, &mut buf).unwrap();

    let dir2 = tempfile::tempdir().unwrap();
    let db2 = Database::open(dir2.path()).unwrap();
    db2.restore(BackupFormat::Json, &buf, &RestoreOptions::default())
        .unwrap();

    let restored = StorageStatsComputer::compute(db2.engine()).unwrap();
    assert_eq!(restored.total_node_count(), 2);
    assert_eq!(restored.node_count_for_label("User"), Some(2));
}

/// A JSON restore registers the names it writes, so the restored properties
/// read back, and still do after a reopen.
#[test]
fn json_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    db.execute_cypher("CREATE (a:User {name: 'Alice', age: 30})")
        .unwrap();

    // Export JSON
    let mut buf = Vec::new();
    let snapshot = db.engine().snapshot();
    export::export_json(db.engine(), &db.interner().unwrap(), 1, &snapshot, &mut buf).unwrap();

    // Restore to new database
    let dir2 = tempfile::tempdir().unwrap();
    {
        let db2 = Database::open(dir2.path()).unwrap();
        let stats = db2
            .restore(BackupFormat::Json, &buf, &RestoreOptions::default())
            .unwrap();
        assert_eq!(stats.nodes, 1, "should restore 1 node");
        db2.persist().unwrap();
    }

    let mut db2 = Database::open(dir2.path()).unwrap();
    let rows = db2
        .execute_cypher("MATCH (n:User) RETURN n.name AS name, n.age AS age")
        .unwrap();
    assert_eq!(
        rows[0].get("name"),
        Some(&coordinode_core::graph::types::Value::String(
            "Alice".into()
        ))
    );
    assert_eq!(
        rows[0].get("age"),
        Some(&coordinode_core::graph::types::Value::Int(30))
    );
}

#[test]
fn apoc_json_restore_loads_nodes_edges_and_is_queryable() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    // apoc.export.json.all shape: string ids, `relationship` records.
    let dump = concat!(
        r#"{"type":"node","id":"0","labels":["User"],"properties":{"name":"Alice"}}"#,
        "\n",
        r#"{"type":"node","id":"1","labels":["User"],"properties":{"name":"Bob"}}"#,
        "\n",
        r#"{"type":"relationship","id":"0","label":"KNOWS","start":{"id":"0"},"end":{"id":"1"},"properties":{"since":2020}}"#,
    );

    let stats = db
        .restore(
            BackupFormat::ApocJson,
            &dump.as_bytes(),
            &RestoreOptions::default(),
        )
        .unwrap();

    assert_eq!(stats.nodes, 2, "two nodes");
    assert_eq!(stats.edges, 1, "one relationship");

    // The restored edge is traversable end to end.
    let rows = db
        .execute_cypher("MATCH (a)-[:KNOWS]->(b) RETURN b.name")
        .unwrap();
    assert_eq!(rows.len(), 1, "KNOWS edge must traverse");
}

#[test]
fn apoc_cypher_plain_restore_loads_nodes_and_edges() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    // Non-optimized apoc.export.cypher.all output.
    let dump = concat!(
        "BEGIN\n",
        "CREATE (:`User`:`UNIQUE IMPORT LABEL` {`name`:\"Alice\", `UNIQUE IMPORT ID`:0});\n",
        "CREATE (:`User`:`UNIQUE IMPORT LABEL` {`name`:\"Bob\", `UNIQUE IMPORT ID`:1});\n",
        "COMMIT\n",
        "BEGIN\n",
        "MATCH (n1:`UNIQUE IMPORT LABEL`{`UNIQUE IMPORT ID`:0}), ",
        "(n2:`UNIQUE IMPORT LABEL`{`UNIQUE IMPORT ID`:1}) ",
        "CREATE (n1)-[r:`KNOWS` {`since`:2020}]->(n2);\n",
        "COMMIT\n",
    );

    let stats = db
        .restore(
            BackupFormat::ApocCypher,
            &dump.as_bytes(),
            &RestoreOptions::default(),
        )
        .unwrap();

    assert_eq!(stats.nodes, 2, "two nodes (constraint statements skipped)");
    assert_eq!(stats.edges, 1, "one relationship");

    let rows = db
        .execute_cypher("MATCH (a)-[:KNOWS]->(b) RETURN b.name")
        .unwrap();
    assert_eq!(rows.len(), 1);
}

#[test]
fn apoc_cypher_unwind_batch_restore_loads_nodes_and_edges() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    // APOC's default optimized (UNWIND-batch) output, multi-line.
    let dump = concat!(
        "UNWIND [{_id:0, properties:{`name`:\"Alice\"}}, ",
        "{_id:1, properties:{`name`:\"Bob\"}}] AS row\n",
        "CREATE (n:`User`{`UNIQUE IMPORT ID`: row._id}) SET n += row.properties;\n",
        "UNWIND [{start: {_id:0}, end: {_id:1}, properties:{`since`:2020}}] AS row\n",
        "MATCH (start:`UNIQUE IMPORT LABEL`{`UNIQUE IMPORT ID`: row.start._id})\n",
        "MATCH (end:`UNIQUE IMPORT LABEL`{`UNIQUE IMPORT ID`: row.end._id})\n",
        "CREATE (start)-[r:`KNOWS`]->(end) SET r += row.properties;\n",
    );

    let stats = db
        .restore(
            BackupFormat::ApocCypher,
            &dump.as_bytes(),
            &RestoreOptions::default(),
        )
        .unwrap();

    assert_eq!(stats.nodes, 2, "two nodes from the UNWIND batch");
    assert_eq!(stats.edges, 1, "one relationship from the UNWIND batch");

    let rows = db
        .execute_cypher("MATCH (a)-[:KNOWS]->(b) RETURN b.name")
        .unwrap();
    assert_eq!(rows.len(), 1);
}

#[test]
fn hetio_json_restore_maps_kinds_and_resolves_edges() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    // Hetnet shape: nodes keyed by (kind, identifier) where identifier is a
    // string OR an integer; edges reference endpoints by [kind, identifier].
    let doc = concat!(
        r#"{"nodes":["#,
        r#"{"kind":"Gene","identifier":9489,"name":"GeneA","data":{"chromosome":"1"}},"#,
        r#"{"kind":"Disease","identifier":"DOID:1","name":"DiseaseB","data":{}}"#,
        r#"],"edges":["#,
        r#"{"source_id":["Gene",9489],"target_id":["Disease","DOID:1"],"kind":"associates","direction":"both","data":{"score":0.9}}"#,
        r#"]}"#,
    );

    let stats = db
        .restore(
            BackupFormat::HetioJson,
            &doc.as_bytes(),
            &RestoreOptions::default(),
        )
        .unwrap();

    assert_eq!(stats.nodes, 2, "two hetnet nodes");
    assert_eq!(stats.edges, 1, "one hetnet edge");

    // kind became the label; the integer-id Gene resolves the edge to the
    // string-id Disease (mixed identifier types map consistently).
    let rows = db
        .execute_cypher("MATCH (g:Gene)-[:associates]->(d:Disease) RETURN d.name")
        .unwrap();
    assert_eq!(rows.len(), 1, "edge resolves across mixed-type identifiers");
    // identifier is preserved as a property.
    let g = db
        .execute_cypher("MATCH (g:Gene {identifier: 9489}) RETURN g.name")
        .unwrap();
    assert_eq!(g.len(), 1, "node found by its original hetnet identifier");
}

#[test]
fn json_restore_only_labels_filters_nodes_and_edges() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    // Two User nodes + one Post node; a User->User FOLLOWS edge (kept) and a
    // User->Post WROTE edge (must be dropped because Post is filtered out).
    let dump = concat!(
        r#"{"type":"node","id":1,"labels":["User"],"properties":{"name":"Alice"}}"#,
        "\n",
        r#"{"type":"node","id":2,"labels":["User"],"properties":{"name":"Bob"}}"#,
        "\n",
        r#"{"type":"node","id":3,"labels":["Post"],"properties":{"title":"Hi"}}"#,
        "\n",
        r#"{"type":"edge","source":1,"target":2,"edge_type":"FOLLOWS","properties":{}}"#,
        "\n",
        r#"{"type":"edge","source":1,"target":3,"edge_type":"WROTE","properties":{}}"#,
    );
    let only: std::collections::HashSet<String> = ["User".to_string()].into_iter().collect();
    let stats = db
        .restore(
            BackupFormat::Json,
            &dump.as_bytes(),
            &RestoreOptions {
                only_labels: Some(&only),
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(stats.nodes, 2, "only the two User nodes are kept");
    assert_eq!(
        stats.edges, 1,
        "only the User->User edge kept; User->Post dropped"
    );
    let f = db
        .execute_cypher("MATCH (a:User)-[:FOLLOWS]->(b:User) RETURN b.name")
        .unwrap();
    assert_eq!(f.len(), 1, "kept FOLLOWS edge traverses");
    // The filtered-out WROTE edge to the dropped Post must be gone.
    let w = db
        .execute_cypher("MATCH (a)-[:WROTE]->(b) RETURN b")
        .unwrap();
    assert_eq!(w.len(), 0, "edge to filtered node dropped");
}

#[test]
fn empty_database_export() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();

    let mut buf = Vec::new();
    let snapshot = db.engine().snapshot();
    let stats =
        export::export_json(db.engine(), &db.interner().unwrap(), 1, &snapshot, &mut buf).unwrap();

    assert_eq!(stats.nodes, 0);
    assert_eq!(stats.edges, 0);
    assert!(buf.is_empty(), "empty DB should produce no output");
}

/// Encode a list of backup entries into the length-prefixed binary
/// stream that `restore_binary` consumes.
fn encode_dump(entries: &[export::BackupEntry]) -> Vec<u8> {
    let mut buf = Vec::new();
    for e in entries {
        let encoded = rmp_serde::to_vec(e).unwrap();
        buf.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        buf.extend_from_slice(&encoded);
    }
    buf
}

#[test]
fn binary_restore_accepts_manifest_at_current_version() {
    // A normally-produced dump carries a manifest at the current format
    // version and restores into a fresh database without forcing.
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.execute_cypher("CREATE (a:User {name: 'Alice'})")
        .unwrap();

    let mut buf = Vec::new();
    let snapshot = db.engine().snapshot();
    export::export_binary(db.engine(), &db.interner().unwrap(), 1, &snapshot, &mut buf).unwrap();

    let dir2 = tempfile::tempdir().unwrap();
    let db2 = Database::open(dir2.path()).unwrap();
    let stats = db2
        .restore(BackupFormat::Binary, &buf, &RestoreOptions::default())
        .unwrap();
    assert_eq!(stats.nodes, 1, "restore should accept current-version dump");
}

#[test]
fn binary_restore_rejects_newer_format_version() {
    // A dump whose format version is newer than this build understands
    // must be refused (the encodings inside may be undecodable here).
    let newer = export::BINARY_FORMAT_VERSION + 1;
    let dump = encode_dump(&[
        export::BackupEntry::Manifest {
            format_version: newer,
            producer: "coordinode-embed/99.0.0".to_string(),
            schema_fingerprint: 0,
        },
        export::BackupEntry::Interner(empty_dictionary()),
    ]);

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();

    let err = db
        .restore(BackupFormat::Binary, &dump, &RestoreOptions::default())
        .unwrap_err();
    assert!(
        matches!(err, restore::RestoreError::IncompatibleVersion(_)),
        "newer format version must be rejected, got {err:?}"
    );

    // Force overrides the version gate for a best-effort restore.
    db.restore(BackupFormat::Binary, &dump, &forced())
        .expect("force should bypass the version gate");
}

#[test]
fn binary_restore_rejects_missing_manifest() {
    // A pre-versioned or truncated dump that does not lead with a
    // manifest is refused unless forced.
    let dump = encode_dump(&[export::BackupEntry::Interner(empty_dictionary())]);

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();

    let err = db
        .restore(BackupFormat::Binary, &dump, &RestoreOptions::default())
        .unwrap_err();
    assert!(
        matches!(err, restore::RestoreError::IncompatibleVersion(_)),
        "missing manifest must be rejected, got {err:?}"
    );

    db.restore(BackupFormat::Binary, &dump, &forced())
        .expect("force should bypass the manifest requirement");
}

/// A dump whose records come before the dictionary that encodes them cannot
/// be restored safely: the records would land with no meaning.
#[test]
fn binary_restore_refuses_records_ahead_of_their_dictionary() {
    let dump = encode_dump(&[
        export::BackupEntry::Manifest {
            format_version: export::BINARY_FORMAT_VERSION,
            producer: export::producer_tag(),
            schema_fingerprint: 0xcbf2_9ce4_8422_2325,
        },
        export::BackupEntry::Node {
            key: b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01".to_vec(),
            value: Vec::new(),
        },
    ]);
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let err = db
        .restore(BackupFormat::Binary, &dump, &RestoreOptions::default())
        .unwrap_err();
    assert!(
        matches!(err, restore::RestoreError::InvalidFormat(_)),
        "records ahead of the dictionary must be refused, got {err:?}"
    );
}

#[test]
fn binary_restore_rejects_schema_fingerprint_mismatch() {
    use coordinode_storage::engine::partition::Partition;

    // Target database already holds schema whose fingerprint differs
    // from the dump's: merging would risk conflicting type definitions.
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    db.engine()
        .put(Partition::Schema, b"schema:label:Widget", b"v1")
        .unwrap();

    let dump = encode_dump(&[
        export::BackupEntry::Manifest {
            format_version: export::BINARY_FORMAT_VERSION,
            producer: export::producer_tag(),
            schema_fingerprint: 0xdead_beef,
        },
        export::BackupEntry::Interner(empty_dictionary()),
    ]);

    let err = db
        .restore(BackupFormat::Binary, &dump, &RestoreOptions::default())
        .unwrap_err();
    assert!(
        matches!(err, restore::RestoreError::SchemaMismatch(_)),
        "differing schema fingerprint must be rejected, got {err:?}"
    );

    // Force overrides the schema guard.
    db.restore(BackupFormat::Binary, &dump, &forced())
        .expect("force should bypass the schema fingerprint guard");
}

/// The formats CoordiNode writes and reads back.
const OWN_FORMATS: [BackupFormat; 3] = [
    BackupFormat::Json,
    BackupFormat::Cypher,
    BackupFormat::Binary,
];

/// A dump of `db` in one of its own formats.
fn dump(db: &Database, format: BackupFormat) -> Vec<u8> {
    let mut buf = Vec::new();
    let snapshot = db.engine().snapshot();
    let interner = db.interner().unwrap();
    match format {
        BackupFormat::Json => export::export_json(db.engine(), &interner, 1, &snapshot, &mut buf),
        BackupFormat::Cypher => {
            export::export_cypher(db.engine(), &interner, 1, &snapshot, &mut buf)
        }
        BackupFormat::Binary => {
            export::export_binary(db.engine(), &interner, 1, &snapshot, &mut buf)
        }
        other => panic!("not an export format: {other:?}"),
    }
    .unwrap();
    buf
}

/// A database holding the named people, and their ids in name order.
fn people(names: &[&str]) -> (Database, tempfile::TempDir, Vec<u64>) {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let mut ids = Vec::new();
    for name in names {
        let rows = db
            .execute_cypher(&format!(
                "CREATE (n:Person {{name: '{name}'}}) RETURN id(n) AS id"
            ))
            .unwrap();
        match rows[0].get("id") {
            Some(coordinode_core::graph::types::Value::Int(id)) => ids.push(*id as u64),
            other => panic!("no id: {other:?}"),
        }
    }
    (db, dir, ids)
}

/// How many nodes `db` holds.
fn node_count(db: &mut Database) -> i64 {
    let rows = db.execute_cypher("MATCH (n) RETURN count(n) AS c").unwrap();
    match rows[0].get("c") {
        Some(coordinode_core::graph::types::Value::Int(c)) => *c,
        other => panic!("no count: {other:?}"),
    }
}

/// Whether a load is recorded as unfinished on `db`.
fn load_recorded(db: &Database) -> bool {
    db.engine()
        .get(
            coordinode_storage::engine::partition::Partition::Schema,
            restore::LOAD_KEY,
        )
        .unwrap()
        .is_some()
}

/// A restore onto a target that already issued the dump's identifiers is
/// refused whole, names them, and writes nothing: a written node would replace
/// the target's own under the same key.
#[test]
fn a_restore_onto_issued_identifiers_is_refused_and_writes_nothing() {
    for format in OWN_FORMATS {
        let (source, _keep, ids) = people(&["Ada", "Bea", "Cal"]);
        let buf = dump(&source, format);

        let (mut target, _keep_target, _) = people(&["Zed"]);
        let refused = target.restore(format, &buf, &forced()).unwrap_err();
        match refused {
            restore::RestoreError::IdentifiersIssued { count, first } => {
                assert_eq!(count, 3, "{format:?}: every issued id counted");
                assert_eq!(first, ids, "{format:?}: the issued ids named");
            }
            other => panic!("{format:?}: expected IdentifiersIssued, got {other:?}"),
        }
        assert_eq!(node_count(&mut target), 1, "{format:?}: nothing written");
        assert!(!load_recorded(&target), "{format:?}: no load recorded");
        let rows = target
            .execute_cypher("MATCH (n:Person) RETURN n.name AS name")
            .unwrap();
        assert_eq!(
            rows[0].get("name"),
            Some(&coordinode_core::graph::types::Value::String("Zed".into())),
            "{format:?}: the target's own node untouched"
        );
    }
}

/// An identifier the target handed out stays issued after its node is
/// deleted: loading a node under it would reinstate a deleted identity.
#[test]
fn a_deleted_identifier_is_still_issued() {
    let (mut target, _keep, ids) = people(&["Gone"]);
    target
        .execute_cypher("MATCH (n:Person) DETACH DELETE n")
        .unwrap();
    assert_eq!(node_count(&mut target), 0);

    let dump = format!(
        r#"{{"type":"node","id":{},"labels":["Person"],"properties":{{"name":"Back"}}}}"#,
        ids[0]
    );
    let refused = target
        .restore(
            BackupFormat::Json,
            &dump.as_bytes(),
            &RestoreOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(
            refused,
            restore::RestoreError::IdentifiersIssued { count: 1, .. }
        ),
        "a deleted id must stay issued, got {refused:?}"
    );
    assert_eq!(node_count(&mut target), 0, "nothing reinstated");
}

/// An identifier of another origin hint is never allocated by this target, so
/// only a node already stored under it makes it issued.
#[test]
fn a_foreign_hint_identifier_is_issued_only_by_a_record() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    // Hint 3, sequence 5.
    let foreign = (3u64 << 44) | 5;
    let dump = format!(
        r#"{{"type":"node","id":{foreign},"labels":["Person"],"properties":{{"name":"Far"}}}}"#
    );

    db.restore(
        BackupFormat::Json,
        &dump.as_bytes(),
        &RestoreOptions::default(),
    )
    .expect("a foreign id no record holds loads");
    let again = db
        .restore(
            BackupFormat::Json,
            &dump.as_bytes(),
            &RestoreOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(&again, restore::RestoreError::IdentifiersIssued { count: 1, first } if first == &[foreign]),
        "a stored foreign id is issued, got {again:?}"
    );

    let rows = db
        .execute_cypher("CREATE (n:Person {name: 'Near'}) RETURN id(n) AS id")
        .unwrap();
    assert_ne!(
        rows[0].get("id"),
        Some(&coordinode_core::graph::types::Value::Int(foreign as i64)),
        "the local allocator never hands out a foreign id"
    );
    assert_eq!(node_count(&mut db), 2);
}

/// `force` relaxes the compatibility gates of a binary dump, never the
/// identifier check: overwriting nodes is not a compatibility question.
#[test]
fn force_does_not_bypass_the_identifier_check() {
    let (source, _keep, _) = people(&["Ada"]);
    let buf = dump(&source, BackupFormat::Binary);
    let (mut target, _keep_target, _) = people(&["Zed"]);

    let refused = target
        .restore(BackupFormat::Binary, &buf, &forced())
        .unwrap_err();
    assert!(
        matches!(refused, restore::RestoreError::IdentifiersIssued { .. }),
        "force must not bypass the identifier check, got {refused:?}"
    );
    assert_eq!(node_count(&mut target), 1);
}

/// Record a load of `input` as unfinished on `db`.
fn record_load(db: &Database, input: &[u8], max_sequence: u64) {
    use sha2::Digest as _;

    let digest: [u8; 32] = sha2::Sha256::digest(input).into();
    db.engine()
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            restore::LOAD_KEY,
            &restore::load_record(&digest, max_sequence),
        )
        .unwrap();
}

/// A load interrupted after its record was written completes when the same
/// input is loaded again, whether none or all of its nodes were already in:
/// its own nodes are not taken for collisions, and a node created afterwards
/// still gets a fresh id.
#[test]
fn an_interrupted_load_completes_when_rerun_with_the_same_input() {
    for format in OWN_FORMATS {
        let (source, _keep, ids) = people(&["Ada", "Bea"]);
        let buf = dump(&source, format);
        let max = ids.iter().copied().max().unwrap();
        let record_load = |db: &Database| record_load(db, &buf, max);

        // Interrupted before it took its sequences, so before any record.
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path()).unwrap();
        record_load(&db);
        db.restore(format, &buf, &RestoreOptions::default())
            .unwrap_or_else(|e| panic!("{format:?}: resume from nothing: {e:?}"));
        assert_eq!(node_count(&mut db), 2, "{format:?}");
        assert!(!load_recorded(&db), "{format:?}: load finished");

        // Interrupted after the last record, before the load was closed, and
        // the process gone: its sequences are taken under its own token, so
        // the rerun after a reopen knows the nodes it finds are its own.
        record_load(&db);
        db.persist().unwrap();
        drop(db);
        let mut db = Database::open(dir.path()).unwrap();
        assert!(
            load_recorded(&db),
            "{format:?}: the record survives a reopen"
        );
        db.restore(format, &buf, &RestoreOptions::default())
            .unwrap_or_else(|e| panic!("{format:?}: resume from everything: {e:?}"));
        assert_eq!(node_count(&mut db), 2, "{format:?}: no node doubled");
        assert!(!load_recorded(&db), "{format:?}: load finished");

        let rows = db
            .execute_cypher("CREATE (n:Person {name: 'Cal'}) RETURN id(n) AS id")
            .unwrap();
        match rows[0].get("id") {
            Some(coordinode_core::graph::types::Value::Int(id)) => {
                assert!(!ids.contains(&(*id as u64)), "{format:?}: fresh id {id}")
            }
            other => panic!("no id: {other:?}"),
        }
        assert_eq!(node_count(&mut db), 3, "{format:?}");
    }
}

/// A load that may have written records holds the target until the same
/// input completes it: a different input is refused before it writes
/// anything. A load that stopped before taking its sequences wrote nothing,
/// so it holds nothing and a different input proceeds.
#[test]
fn an_unfinished_load_refuses_a_different_input() {
    let (source, _keep, _) = people(&["Ada"]);
    let buf = dump(&source, BackupFormat::Json);
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();

    // A load of identifiers of other hints only writes from its start.
    record_load(&db, b"another input", 0);
    let refused = db
        .restore(BackupFormat::Json, &buf, &RestoreOptions::default())
        .unwrap_err();
    assert!(
        matches!(refused, restore::RestoreError::UnfinishedLoad),
        "got {refused:?}"
    );
    assert_eq!(node_count(&mut db), 0, "nothing written");
    assert!(load_recorded(&db), "the unfinished load stays recorded");

    // A load whose sequences no grant took never reached its records.
    record_load(&db, b"another input", 7);
    db.restore(BackupFormat::Json, &buf, &RestoreOptions::default())
        .expect("a load that wrote nothing holds nothing");
    assert_eq!(node_count(&mut db), 1);
    assert!(!load_recorded(&db), "the load finished");
}

/// Imports that bring their own numbering (Hetionet from 0, APOC with the
/// source's ids) keep it, raise the lease past it, and refuse a second load
/// onto the nodes the first one wrote.
#[test]
fn imported_identifiers_are_never_issued_again() {
    let hetio = concat!(
        r#"{"nodes":["#,
        r#"{"kind":"Gene","identifier":1,"name":"A","data":{}},"#,
        r#"{"kind":"Gene","identifier":2,"name":"B","data":{}}"#,
        r#"],"edges":[]}"#,
    );
    let apoc = concat!(
        r#"{"type":"node","id":"7","labels":["Gene"],"properties":{"name":"A"}}"#,
        "\n",
        r#"{"type":"node","id":"9","labels":["Gene"],"properties":{"name":"B"}}"#,
    );
    for (format, input) in [
        (BackupFormat::HetioJson, hetio),
        (BackupFormat::ApocJson, apoc),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path()).unwrap();
        db.restore(format, &input.as_bytes(), &RestoreOptions::default())
            .unwrap_or_else(|e| panic!("{format:?}: {e:?}"));
        let loaded: Vec<i64> = db
            .execute_cypher("MATCH (n:Gene) RETURN id(n) AS id")
            .unwrap()
            .iter()
            .filter_map(|r| match r.get("id") {
                Some(coordinode_core::graph::types::Value::Int(id)) => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(loaded.len(), 2, "{format:?}");

        let rows = db
            .execute_cypher("CREATE (n:Gene {name: 'C'}) RETURN id(n) AS id")
            .unwrap();
        match rows[0].get("id") {
            Some(coordinode_core::graph::types::Value::Int(id)) => {
                assert!(!loaded.contains(id), "{format:?}: fresh id {id}")
            }
            other => panic!("no id: {other:?}"),
        }
        assert_eq!(node_count(&mut db), 3, "{format:?}: nothing replaced");

        let again = db
            .restore(format, &input.as_bytes(), &RestoreOptions::default())
            .unwrap_err();
        assert!(
            matches!(
                again,
                restore::RestoreError::IdentifiersIssued { count: 2, .. }
            ),
            "{format:?}: got {again:?}"
        );
    }
}

/// The text formats' hex reads back what it wrote, in either case, and
/// refuses an odd length or a non-hex character instead of guessing.
#[test]
fn hex_round_trips_and_refuses_malformed_text() {
    use super::export::hex;

    let bytes: Vec<u8> = (0..=255).collect();
    let text = hex::encode(&bytes);
    assert_eq!(text.len(), 512);
    assert_eq!(&text[..8], "00010203");
    assert_eq!(hex::decode(&text), Some(bytes.clone()));
    assert_eq!(hex::decode(&text.to_uppercase()), Some(bytes));
    assert_eq!(hex::decode(""), Some(Vec::new()));
    assert_eq!(hex::decode("abc"), None, "odd length");
    assert_eq!(hex::decode("0g"), None, "not a hex digit");
}

/// A string value that holds the text of a cypher comment marker stays a
/// value: only a comment that ends the line counts, so the node is not taken
/// for a temporal version and the edge not for a discriminated instance.
#[test]
fn a_string_holding_a_comment_marker_is_not_taken_for_one() {
    use coordinode_core::graph::types::Value;

    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.execute_cypher("CREATE (:Note {body: 'x; // valid_from 5'}), (:Note {body: 'y'})")
        .unwrap();
    db.execute_cypher(
        "MATCH (a:Note {body: 'y'}), (b:Note {body: 'x; // valid_from 5'}) \
         CREATE (a)-[:REFS {why: 'z; // discriminator ab'}]->(b)",
    )
    .unwrap();
    let buf = dump(&db, BackupFormat::Cypher);

    let dir2 = tempfile::tempdir().unwrap();
    let mut db2 = Database::open(dir2.path()).unwrap();
    db2.restore(BackupFormat::Cypher, &buf, &RestoreOptions::default())
        .unwrap();
    let rows = db2
        .execute_cypher("MATCH (:Note)-[r:REFS]->(b:Note) RETURN b.body AS body, r.why AS why")
        .unwrap();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(
        rows[0].get("body"),
        Some(&Value::String("x; // valid_from 5".into()))
    );
    assert_eq!(
        rows[0].get("why"),
        Some(&Value::String("z; // discriminator ab".into()))
    );
}
