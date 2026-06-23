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
mod tests;
