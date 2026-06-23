//! Flat (brute-force) vector index: exact nearest neighbor search.
//!
//! For small datasets (<100K vectors). Computes distance to every vector
//! and returns exact top-K results. No approximation, 100% recall.
//!
//! Automatically selected by the query planner when the dataset is small
//! enough that brute-force is faster than HNSW index overhead.

use coordinode_core::graph::types::VectorMetric;

use crate::metrics;

/// Flat vector index: stores all vectors and computes exact distances.
pub struct FlatIndex {
    /// Distance metric.
    metric: VectorMetric,
    /// Stored vectors: (id, vector).
    vectors: Vec<(u64, Vec<f32>)>,
}

/// Search result with node ID and distance/similarity score.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatSearchResult {
    pub id: u64,
    pub score: f32,
}

impl FlatIndex {
    /// Create a new empty flat index.
    pub fn new(metric: VectorMetric) -> Self {
        Self {
            metric,
            vectors: Vec::new(),
        }
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Insert a vector. Duplicates (same ID) are ignored.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        if self
            .vectors
            .iter()
            .any(|(existing_id, _)| *existing_id == id)
        {
            return;
        }
        self.vectors.push((id, vector));
    }

    /// Remove a vector by ID.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.vectors.len();
        self.vectors.retain(|(vid, _)| *vid != id);
        self.vectors.len() < before
    }

    /// Search for K exact nearest neighbors.
    ///
    /// Returns results sorted by distance (nearest first for L2/L1,
    /// highest similarity first for Cosine/DotProduct).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<FlatSearchResult> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut scored: Vec<FlatSearchResult> = self
            .vectors
            .iter()
            .map(|(id, vec)| {
                let score = self.compute_distance(query, vec);
                FlatSearchResult { id: *id, score }
            })
            .collect();

        // Sort by distance (ascending for L2/L1, "ascending" for transformed cosine/dot)
        scored.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.truncate(k);
        scored
    }

    /// Compute distance using the configured metric.
    /// Returns a value where lower = more similar (consistent with HNSW).
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            VectorMetric::Cosine => 1.0 - metrics::cosine_similarity(a, b),
            VectorMetric::L2 => metrics::euclidean_distance_squared(a, b),
            VectorMetric::DotProduct => -metrics::dot_product(a, b),
            VectorMetric::L1 => metrics::manhattan_distance(a, b),
        }
    }

    /// Check if brute-force is recommended for this dataset size.
    /// Returns true for <100K vectors (flat is faster than HNSW overhead).
    pub fn is_recommended_size(&self) -> bool {
        self.vectors.len() < 100_000
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
