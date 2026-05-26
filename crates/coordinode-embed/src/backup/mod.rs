//! Logical backup and restore for CoordiNode.
//!
//! Exports graph data (nodes, edges, schema) as portable formats:
//! - **JSON Lines**: one entity per line, schema + data (application integration)
//! - **OpenCypher**: CREATE statements, cross-database portable
//! - **Binary**: MessagePack dump, fastest, coordinode-native
//!
//! Backup takes a consistent snapshot: all exported data reflects the
//! state at the moment backup started. Ongoing writes are not blocked.

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
            restore::restore_binary(db2.engine(), &mut cursor).unwrap();

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
        let stats = restore::restore_json(db2.engine(), &mut interner2, 1, &mut cursor).unwrap();

        assert_eq!(stats.nodes, 1, "should restore 1 node");
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
}
