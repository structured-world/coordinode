use super::export;
use super::restore;
use crate::Database;

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

    let mut cursor = std::io::Cursor::new(&buf);
    let restore_stats = restore::restore_binary(
        db2.engine(),
        db2.field_registrar().as_ref(),
        &mut cursor,
        false,
    )
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
    let mut cursor = std::io::Cursor::new(&buf);
    restore::restore_binary(
        db2.engine(),
        db2.field_registrar().as_ref(),
        &mut cursor,
        false,
    )
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
    let mut cursor = std::io::Cursor::new(&buf);
    let refused = restore::restore_binary(
        db3.engine(),
        db3.field_registrar().as_ref(),
        &mut cursor,
        true,
    );
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
    let mut cursor = std::io::Cursor::new(&buf);
    restore::restore_binary(
        db2.engine(),
        db2.field_registrar().as_ref(),
        &mut cursor,
        false,
    )
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
    let mut cursor = std::io::BufReader::new(std::io::Cursor::new(&buf));
    restore::restore_json(
        db2.engine(),
        db2.field_registrar().as_ref(),
        1,
        &mut cursor,
        None,
    )
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
        let mut cursor = std::io::BufReader::new(std::io::Cursor::new(&buf));
        let stats = restore::restore_json(
            db2.engine(),
            db2.field_registrar().as_ref(),
            1,
            &mut cursor,
            None,
        )
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

    let mut cursor = std::io::BufReader::new(std::io::Cursor::new(dump.as_bytes()));
    let stats = restore::restore_apoc_json(
        db.engine(),
        db.field_registrar().as_ref(),
        1,
        &mut cursor,
        None,
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

    let mut cursor = std::io::BufReader::new(std::io::Cursor::new(dump.as_bytes()));
    let stats =
        restore::restore_apoc_cypher(db.engine(), db.field_registrar().as_ref(), 1, &mut cursor)
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

    let mut cursor = std::io::BufReader::new(std::io::Cursor::new(dump.as_bytes()));
    let stats =
        restore::restore_apoc_cypher(db.engine(), db.field_registrar().as_ref(), 1, &mut cursor)
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

    let mut cursor = std::io::BufReader::new(std::io::Cursor::new(doc.as_bytes()));
    let stats = restore::restore_hetio_json(
        db.engine(),
        db.field_registrar().as_ref(),
        1,
        &mut cursor,
        None,
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
    let mut cursor = std::io::BufReader::new(std::io::Cursor::new(dump.as_bytes()));
    let stats = restore::restore_json(
        db.engine(),
        db.field_registrar().as_ref(),
        1,
        &mut cursor,
        Some(&only),
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
    let mut cursor = std::io::Cursor::new(&buf);
    let stats = restore::restore_binary(
        db2.engine(),
        db2.field_registrar().as_ref(),
        &mut cursor,
        false,
    )
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
    let fields = db.field_registrar();

    let mut cursor = std::io::Cursor::new(&dump);
    let err =
        restore::restore_binary(db.engine(), fields.as_ref(), &mut cursor, false).unwrap_err();
    assert!(
        matches!(err, restore::RestoreError::IncompatibleVersion(_)),
        "newer format version must be rejected, got {err:?}"
    );

    // Force overrides the version gate for a best-effort restore.
    let mut cursor = std::io::Cursor::new(&dump);
    restore::restore_binary(db.engine(), fields.as_ref(), &mut cursor, true)
        .expect("force should bypass the version gate");
}

#[test]
fn binary_restore_rejects_missing_manifest() {
    // A pre-versioned or truncated dump that does not lead with a
    // manifest is refused unless forced.
    let dump = encode_dump(&[export::BackupEntry::Interner(empty_dictionary())]);

    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let fields = db.field_registrar();

    let mut cursor = std::io::Cursor::new(&dump);
    let err =
        restore::restore_binary(db.engine(), fields.as_ref(), &mut cursor, false).unwrap_err();
    assert!(
        matches!(err, restore::RestoreError::IncompatibleVersion(_)),
        "missing manifest must be rejected, got {err:?}"
    );

    let mut cursor = std::io::Cursor::new(&dump);
    restore::restore_binary(db.engine(), fields.as_ref(), &mut cursor, true)
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
    let mut cursor = std::io::Cursor::new(&dump);
    let err = restore::restore_binary(
        db.engine(),
        db.field_registrar().as_ref(),
        &mut cursor,
        false,
    )
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
    let fields = db.field_registrar();

    let dump = encode_dump(&[
        export::BackupEntry::Manifest {
            format_version: export::BINARY_FORMAT_VERSION,
            producer: export::producer_tag(),
            schema_fingerprint: 0xdead_beef,
        },
        export::BackupEntry::Interner(empty_dictionary()),
    ]);

    let mut cursor = std::io::Cursor::new(&dump);
    let err =
        restore::restore_binary(db.engine(), fields.as_ref(), &mut cursor, false).unwrap_err();
    assert!(
        matches!(err, restore::RestoreError::SchemaMismatch(_)),
        "differing schema fingerprint must be rejected, got {err:?}"
    );

    // Force overrides the schema guard.
    let mut cursor = std::io::Cursor::new(&dump);
    restore::restore_binary(db.engine(), fields.as_ref(), &mut cursor, true)
        .expect("force should bypass the schema fingerprint guard");
}
