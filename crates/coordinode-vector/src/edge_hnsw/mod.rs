//! Edge HNSW index: vector similarity search on edge properties.
//!
//! Unique to CoordiNode — no other graph database supports vector indexes on edges.
//! Each edge type can have its own HNSW index on a vector property.
//!
//! Use case: semantic search over relationship types, fraud detection by
//! transaction similarity, recommendations by relationship patterns.
//!
//! # Cluster-ready notes
//! Same as node HNSW: each CE replica builds its own in-memory index
//! from replicated edge property data in CoordiNode storage.

use crate::hnsw::{HnswConfig, HnswIndex};

/// An edge identified by (source_id, target_id).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId {
    pub source: u64,
    pub target: u64,
}

impl EdgeId {
    /// Pack source and target into a single u64 for HNSW node ID.
    /// Uses Cantor pairing function for unique mapping.
    fn to_hnsw_id(self) -> u64 {
        let s = self.source;
        let t = self.target;
        // Cantor pairing: (s + t) * (s + t + 1) / 2 + t
        // For large IDs, use a simpler hash-like encoding
        s.wrapping_mul(2_147_483_647).wrapping_add(t)
    }
}

/// Edge HNSW index for a specific edge type and vector property.
pub struct EdgeHnswIndex {
    /// The edge type this index covers (e.g., "KNOWS").
    pub edge_type: String,
    /// The vector property name on the edge (e.g., "relationship_embedding").
    pub property: String,
    /// Underlying HNSW index.
    hnsw: HnswIndex,
    /// Reverse mapping: HNSW node ID → EdgeId.
    id_map: std::collections::HashMap<u64, EdgeId>,
}

/// Search result for edge vector search.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSearchResult {
    /// Source node ID of the edge.
    pub source: u64,
    /// Target node ID of the edge.
    pub target: u64,
    /// Similarity/distance score.
    pub score: f32,
}

impl EdgeHnswIndex {
    /// Create a new edge HNSW index.
    pub fn new(
        edge_type: impl Into<String>,
        property: impl Into<String>,
        config: HnswConfig,
    ) -> Self {
        Self {
            edge_type: edge_type.into(),
            property: property.into(),
            hnsw: HnswIndex::new(config),
            id_map: std::collections::HashMap::new(),
        }
    }

    /// Number of indexed edge vectors.
    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.hnsw.is_empty()
    }

    /// Insert an edge vector into the index.
    pub fn insert(&mut self, source: u64, target: u64, vector: Vec<f32>) {
        let edge_id = EdgeId { source, target };
        let hnsw_id = edge_id.to_hnsw_id();

        self.id_map.insert(hnsw_id, edge_id);
        self.hnsw.insert(hnsw_id, vector);
    }

    /// Search for K nearest edge vectors to the query.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<EdgeSearchResult> {
        self.hnsw
            .search(query, k)
            .into_iter()
            .filter_map(|r| {
                self.id_map.get(&r.id).map(|edge_id| EdgeSearchResult {
                    source: edge_id.source,
                    target: edge_id.target,
                    score: r.score,
                })
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
