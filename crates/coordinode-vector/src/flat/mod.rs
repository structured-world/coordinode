//! Flat (brute-force) vector index: exact nearest neighbor search.
//!
//! For small datasets (<100K vectors). Scores every vector exactly (binary64,
//! canonical reduction; see [`crate::metrics::exact`]) and returns the top-K.
//! No approximation, 100% recall.

use coordinode_core::graph::types::VectorMetric;

use crate::metrics::exact::{self, ExactQuery, VectorDomainError};

/// Flat vector index: stores all vectors and scores each query exactly.
pub struct FlatIndex {
    /// Distance metric.
    metric: VectorMetric,
    /// Stored vectors: (id, vector), all of one dimension.
    vectors: Vec<(u64, Vec<f32>)>,
}

/// Search result: the node and its exact ranking distance under the index's
/// metric (see [`ExactQuery`]): lower ranks first, ties by id.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatSearchResult {
    pub id: u64,
    pub score: f64,
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

    /// The dimension every stored vector has, once one is stored.
    fn dimension(&self) -> Option<usize> {
        self.vectors.first().map(|(_, v)| v.len())
    }

    /// Insert a vector. Duplicates (same ID) are ignored.
    ///
    /// # Errors
    ///
    /// The vector is empty, has a non-finite component, has another
    /// dimension than the stored ones, or is zero under cosine; nothing is
    /// stored.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) -> Result<(), VectorDomainError> {
        exact::admit(&vector, self.metric)?;
        if let Some(dim) = self.dimension()
            && dim != vector.len()
        {
            return Err(VectorDomainError::DimensionMismatch {
                left: vector.len(),
                right: dim,
            });
        }
        if !self.vectors.iter().any(|(existing, _)| *existing == id) {
            self.vectors.push((id, vector));
        }
        Ok(())
    }

    /// Remove a vector by ID.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.vectors.len();
        self.vectors.retain(|(vid, _)| *vid != id);
        self.vectors.len() < before
    }

    /// Search for K exact nearest neighbors, nearest first, equal distances by
    /// id.
    ///
    /// # Errors
    ///
    /// The query is empty, non-finite, of another dimension than the stored
    /// vectors, or zero under cosine.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<FlatSearchResult>, VectorDomainError> {
        let exact = ExactQuery::new(query, self.metric)?;
        if self.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let mut scored = self
            .vectors
            .iter()
            .map(|(id, vec)| {
                Ok(FlatSearchResult {
                    id: *id,
                    score: exact.distance(vec)?,
                })
            })
            .collect::<Result<Vec<_>, VectorDomainError>>()?;
        scored.sort_by(|a, b| exact::ranked_order(a.score, a.id, b.score, b.id));
        scored.truncate(k);
        Ok(scored)
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
