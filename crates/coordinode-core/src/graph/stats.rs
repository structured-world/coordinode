//! Storage statistics for query cost estimation.
//!
//! The `StorageStats` trait provides an interface for the query planner
//! to access real storage statistics instead of hardcoded defaults.
//! Implementations should cache results and refresh periodically,
//! as computing exact statistics requires storage scans.

/// Storage statistics interface for query cost estimation.
///
/// Enables the query planner to use real data distribution information
/// (node counts, fan-out) instead of hardcoded defaults. Implementations
/// should be lightweight to call (cached internally).
pub trait StorageStats {
    /// Total number of nodes in storage.
    fn total_node_count(&self) -> u64;

    /// Number of nodes with a specific label.
    /// Returns None if per-label statistics are not available.
    fn node_count_for_label(&self, label: &str) -> Option<u64>;

    /// Average outgoing fan-out for a specific edge type.
    /// Returns None if per-edge-type statistics are not available.
    fn avg_fan_out_for_type(&self, edge_type: &str) -> Option<f64>;

    /// Overall average fan-out across all edge types.
    fn avg_fan_out(&self) -> f64;

    /// Number of distinct labels in the database.
    fn label_count(&self) -> u64;

    // ── Vector index statistics (R-PUSH1) ──────────────────────────────
    //
    // The graph predicate push-down rule (`arch/core/query-engine.md`
    // § Graph Predicate Push-Down) compares candidate-set size `|C|` against
    // the vector index size `|V|` to pick a strategy. These methods expose
    // the index parameters that the rule depends on. All return `None` when
    // the index is not registered or statistics are unavailable; callers
    // must handle absence gracefully (fall back to safe defaults).

    /// Number of vectors in the HNSW index for `(label, property)`.
    fn vector_index_size(&self, _label: &str, _property: &str) -> Option<u64> {
        None
    }

    /// Vector dimensionality of the HNSW index for `(label, property)`.
    fn vector_index_dim(&self, _label: &str, _property: &str) -> Option<u32> {
        None
    }

    /// Per-index crossover threshold: `|C|` below this triggers graph-first.
    /// Returns `None` when no index is registered. Per `arch`, default is
    /// 500 for node-typed indexes, 200 for edge-typed indexes; the planner
    /// applies that default when this method returns `None`.
    fn vector_index_crossover(&self, _label: &str, _property: &str) -> Option<usize> {
        None
    }
}
