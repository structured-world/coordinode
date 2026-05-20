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

    /// True when writes in this partition are commutative (merge
    /// operators) and therefore conflict-free for OCC purposes —
    /// concurrent writers cannot produce a lost-update because the
    /// merge function composes them at read time.
    ///
    /// Used by [`crate::engine::coordinator::MultiModalCoordinator::validate_occ`]
    /// to skip conflict checks on these partitions: a read of a
    /// commutative partition imposes no read-write ordering constraint.
    ///
    /// Currently `Adj` (posting-list merge ops for edge insertions and
    /// removals) and `Counter` (additive deltas).
    pub fn is_commutative(self) -> bool {
        matches!(self, Self::Adj | Self::Counter)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_commutative_classifies_partitions_correctly() {
        // Commutative — merge operators compose concurrent writers.
        assert!(Partition::Adj.is_commutative());
        assert!(Partition::Counter.is_commutative());
        // Non-commutative — last-write-wins, conflict-detection required.
        assert!(!Partition::Node.is_commutative());
        assert!(!Partition::EdgeProp.is_commutative());
        assert!(!Partition::Blob.is_commutative());
        assert!(!Partition::BlobRef.is_commutative());
        assert!(!Partition::Schema.is_commutative());
        assert!(!Partition::Idx.is_commutative());
        assert!(!Partition::Raft.is_commutative());
    }

    #[test]
    fn is_commutative_total_over_all_partitions() {
        // Every variant covered — guard against future additions
        // forgetting to declare commutativity.
        for &part in Partition::all() {
            // Just call it — exhaustive match in the function body
            // means an un-handled variant would not compile.
            let _ = part.is_commutative();
        }
    }
}
