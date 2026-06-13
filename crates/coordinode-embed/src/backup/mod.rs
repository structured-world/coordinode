//! Logical backup and restore for CoordiNode.
//!
//! Exports graph data (nodes, edges, schema) as portable formats:
//! - **JSON Lines**: one entity per line, schema + data (application integration)
//! - **OpenCypher**: CREATE statements, cross-database portable
//! - **Binary**: MessagePack dump, fastest, coordinode-native
//!
//! Backup takes a consistent snapshot: all exported data reflects the
//! state at the moment backup started. Ongoing writes are not blocked.
//!
//! Restore additionally understands two **import-only** Neo4j formats so a
//! database can be migrated in without standing up Neo4j here: APOC
//! json-export (`apoc.export.json.all`) and APOC cypher-export
//! (`apoc.export.cypher.all`). Both are read by structural parsers that
//! write straight to storage; we never execute APOC procedures (a Neo4j
//! plugin that does not run correctly on a sharded cluster).

pub mod export;
pub mod restore;

/// Backup output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackupFormat {
    /// JSON Lines — one JSON object per line. Human-readable, ETL-friendly.
    Json,
    /// OpenCypher CREATE statements. Cross-database portable.
    Cypher,
    /// MessagePack binary dump. Fastest backup/restore, not portable.
    Binary,
    /// Neo4j APOC json-export (`apoc.export.json.all`). Import-only:
    /// our restore ingests APOC's portable output without running APOC.
    ApocJson,
    /// Neo4j APOC cypher-export (`apoc.export.cypher.all`). Import-only:
    /// a structural parser reads the foreign CREATE statements directly.
    ApocCypher,
    /// Full Raft data snapshot: a single self-contained binary blob of every
    /// user-data partition, the same artifact the Raft layer ships between
    /// nodes. Fast whole-database backup/restore; not human-readable.
    RaftSnapshot,
    /// Hetionet "hetnet" JSON (dhimmel/hetio source format). Import-only:
    /// `{nodes, edges}` with `(kind, identifier)` node keys, mapped to node
    /// labels / relationship types the way hetnetpy loads it into Neo4j.
    HetioJson,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::export;
    use super::restore;
    use crate::Database;

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
            export::export_json(db.engine(), &db.interner(), 1, &snapshot, &mut buf).unwrap();

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
        let stats =
            export::export_cypher(db.engine(), &db.interner(), 1, &snapshot, &mut buf).unwrap();

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
            export::export_binary(db.engine(), &db.interner(), 1, &snapshot, &mut buf).unwrap();
        assert!(export_stats.nodes >= 2);

        // Restore to new database
        let dir2 = tempfile::tempdir().unwrap();
        let db2 = Database::open(dir2.path()).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let (restore_stats, _interner) =
            restore::restore_binary(db2.engine(), &mut cursor, false).unwrap();

        assert_eq!(restore_stats.nodes, export_stats.nodes);
    }

    #[test]
    fn json_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path()).unwrap();

        db.execute_cypher("CREATE (a:User {name: 'Alice', age: 30})")
            .unwrap();

        // Export JSON
        let mut buf = Vec::new();
        let snapshot = db.engine().snapshot();
        export::export_json(db.engine(), &db.interner(), 1, &snapshot, &mut buf).unwrap();

        // Restore to new database
        let dir2 = tempfile::tempdir().unwrap();
        let db2 = Database::open(dir2.path()).unwrap();

        let mut interner2 = coordinode_core::graph::intern::FieldInterner::new();
        let mut cursor = std::io::BufReader::new(std::io::Cursor::new(&buf));
        let stats =
            restore::restore_json(db2.engine(), &mut interner2, 1, &mut cursor, None).unwrap();

        assert_eq!(stats.nodes, 1, "should restore 1 node");
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

        let mut interner = db.interner().clone();
        let mut cursor = std::io::BufReader::new(std::io::Cursor::new(dump.as_bytes()));
        let stats =
            restore::restore_apoc_json(db.engine(), &mut interner, 1, &mut cursor, None).unwrap();
        *db.interner_arc().write() = interner;

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

        let mut interner = db.interner().clone();
        let mut cursor = std::io::BufReader::new(std::io::Cursor::new(dump.as_bytes()));
        let stats =
            restore::restore_apoc_cypher(db.engine(), &mut interner, 1, &mut cursor).unwrap();
        *db.interner_arc().write() = interner;

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

        let mut interner = db.interner().clone();
        let mut cursor = std::io::BufReader::new(std::io::Cursor::new(dump.as_bytes()));
        let stats =
            restore::restore_apoc_cypher(db.engine(), &mut interner, 1, &mut cursor).unwrap();
        *db.interner_arc().write() = interner;

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

        let mut interner = db.interner().clone();
        let mut cursor = std::io::BufReader::new(std::io::Cursor::new(doc.as_bytes()));
        let stats =
            restore::restore_hetio_json(db.engine(), &mut interner, 1, &mut cursor, None).unwrap();
        *db.interner_arc().write() = interner;

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
        let mut interner = db.interner().clone();
        let mut cursor = std::io::BufReader::new(std::io::Cursor::new(dump.as_bytes()));
        let stats =
            restore::restore_json(db.engine(), &mut interner, 1, &mut cursor, Some(&only)).unwrap();
        *db.interner_arc().write() = interner;

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
            export::export_json(db.engine(), &db.interner(), 1, &snapshot, &mut buf).unwrap();

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
        export::export_binary(db.engine(), &db.interner(), 1, &snapshot, &mut buf).unwrap();

        let dir2 = tempfile::tempdir().unwrap();
        let db2 = Database::open(dir2.path()).unwrap();
        let mut cursor = std::io::Cursor::new(&buf);
        let (stats, _interner) = restore::restore_binary(db2.engine(), &mut cursor, false).unwrap();
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
            export::BackupEntry::Interner(Vec::new()),
        ]);

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        let mut cursor = std::io::Cursor::new(&dump);
        let err = restore::restore_binary(db.engine(), &mut cursor, false).unwrap_err();
        assert!(
            matches!(err, restore::RestoreError::IncompatibleVersion(_)),
            "newer format version must be rejected, got {err:?}"
        );

        // Force overrides the version gate for a best-effort restore.
        let mut cursor = std::io::Cursor::new(&dump);
        restore::restore_binary(db.engine(), &mut cursor, true)
            .expect("force should bypass the version gate");
    }

    #[test]
    fn binary_restore_rejects_missing_manifest() {
        // A pre-versioned or truncated dump that does not lead with a
        // manifest is refused unless forced.
        let dump = encode_dump(&[export::BackupEntry::Interner(Vec::new())]);

        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        let mut cursor = std::io::Cursor::new(&dump);
        let err = restore::restore_binary(db.engine(), &mut cursor, false).unwrap_err();
        assert!(
            matches!(err, restore::RestoreError::IncompatibleVersion(_)),
            "missing manifest must be rejected, got {err:?}"
        );

        let mut cursor = std::io::Cursor::new(&dump);
        restore::restore_binary(db.engine(), &mut cursor, true)
            .expect("force should bypass the manifest requirement");
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
            export::BackupEntry::Interner(Vec::new()),
        ]);

        let mut cursor = std::io::Cursor::new(&dump);
        let err = restore::restore_binary(db.engine(), &mut cursor, false).unwrap_err();
        assert!(
            matches!(err, restore::RestoreError::SchemaMismatch(_)),
            "differing schema fingerprint must be rejected, got {err:?}"
        );

        // Force overrides the schema guard.
        let mut cursor = std::io::Cursor::new(&dump);
        restore::restore_binary(db.engine(), &mut cursor, true)
            .expect("force should bypass the schema fingerprint guard");
    }
}
