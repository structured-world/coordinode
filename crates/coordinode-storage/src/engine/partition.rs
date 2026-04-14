//! Storage partitions (column families) for different data categories.

/// The logical partitions used by the storage engine.
///
/// Each partition maps to a storage partition (similar to a column family in
/// RocksDB). Data is separated by category for independent compaction,
/// compression tuning, and scan isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Partition {
    /// `node:` — Node properties (MessagePack-encoded).
    /// Key format: `node:<shard_id>:<node_id>`
    Node,

    /// `adj:` — Adjacency posting lists (forward + reverse edges).
    /// Key format: `adj:<TYPE>:{out|in}:<src_or_tgt>`
    Adj,

    /// `edgeprop:` — Edge properties (facets).
    /// Key format: `edgeprop:<TYPE>:<src>:<tgt>`
    EdgeProp,

    /// `blob:` — BlobStore content-addressed chunks (256KB, SHA-256 key).
    /// Key format: `blob:<sha256_hex>`
    Blob,

    /// `blobref:` — Blob references from nodes to blob chunks.
    /// Key format: `blobref:<node_id>:<property_name>`
    BlobRef,

    /// `schema:` — Graph schema definitions (labels, edge types, constraints).
    /// Key format: `schema:<kind>:<name>`
    Schema,

    /// `idx:` — Secondary index entries (B-tree, vector, full-text).
    /// Key format: `idx:<index_name>:<encoded_value>:<node_id>`
    Idx,

    /// `raft:` — Raft log entries and consensus metadata.
    /// Key format: `raft:<kind>:<index>`
    Raft,

    /// `counter:` — Atomic i64 counters (node degree cache, analytics).
    /// Key format: `counter:<scope>:<id>` (e.g., `counter:degree:42`)
    /// Uses `CounterMerge` operator: base i64 + sum of delta operands.
    Counter,
}

impl From<coordinode_core::txn::proposal::PartitionId> for Partition {
    fn from(id: coordinode_core::txn::proposal::PartitionId) -> Self {
        use coordinode_core::txn::proposal::PartitionId;
        match id {
            PartitionId::Node => Self::Node,
            PartitionId::Adj => Self::Adj,
            PartitionId::EdgeProp => Self::EdgeProp,
            PartitionId::Blob => Self::Blob,
            PartitionId::BlobRef => Self::BlobRef,
            PartitionId::Schema => Self::Schema,
            PartitionId::Idx => Self::Idx,
            PartitionId::Counter => Self::Counter,
        }
    }
}

impl Partition {
    /// The storage partition name for this logical partition.
    pub fn name(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Adj => "adj",
            Self::EdgeProp => "edgeprop",
            Self::Blob => "blob",
            Self::BlobRef => "blobref",
            Self::Schema => "schema",
            Self::Idx => "idx",
            Self::Raft => "raft",
            Self::Counter => "counter",
        }
    }

    /// All partitions in creation order.
    pub fn all() -> &'static [Partition] {
        &[
            Self::Node,
            Self::Adj,
            Self::EdgeProp,
            Self::Blob,
            Self::BlobRef,
            Self::Schema,
            Self::Idx,
            Self::Raft,
            Self::Counter,
        ]
    }
}
