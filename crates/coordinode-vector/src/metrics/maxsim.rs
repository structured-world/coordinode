//! MaxSim scoring kernel for late-interaction retrieval (ColBERT-style).
//!
//! Given a document represented by `D` token-level f32 vectors and a query
//! represented by `Q` token-level f32 vectors of identical dimensionality,
//! MaxSim is defined as
//!
//! ```text
//! score(doc, query) = Sum_{q in query} max_{d in doc} similarity(q, d)
//! ```
//!
//! Where `similarity` is the dot product. For cosine semantics the caller
//! must L2-normalise rows before invocation; the kernel itself is metric-
//! agnostic and just computes the dot product per pair.
//!
//! The inner pairwise kernel routes through [`super::dot_product`] which
//! already carries AVX512 / AVX2+FMA / NEON dispatch with scalar fallback.
//! No additional SIMD wiring is needed here; the wins from those kernels
//! propagate transparently.
//!
//! Complexity: `O(Q * D * dim)` per scoring call. For typical ColBERTv2
//! workloads (`Q ~ 32-64`, `D ~ 100-220`, `dim = 128`) a single call is
//! a few million multiply-adds. The forthcoming PLAID-style centroid
//! index will prune the `D` factor for large corpora; this brute-force
//! kernel is the per-pair primitive that pruning ultimately reduces to.

use super::dot_product;

/// Compute the MaxSim score between a document matrix and a query matrix.
///
/// Returns `0.0` for any of these inputs (none are errors; the score is
/// merely meaningless in those cases):
/// - `doc` is empty
/// - `query` is empty
/// - any document row's dimensionality differs from the first query row's
///
/// Pre-normalise both inputs to unit L2 norm if you want a cosine-based
/// MaxSim; otherwise the result is plain dot-product MaxSim.
#[inline]
pub fn maxsim(doc: &[Vec<f32>], query: &[Vec<f32>]) -> f32 {
    let Some(dim) = query.first().map(Vec::len) else {
        return 0.0;
    };
    if dim == 0 || doc.is_empty() {
        return 0.0;
    }
    // Reject mismatched dim across either matrix. Caller-side construction
    // through `Value::try_multi_vector` already enforces this for stored
    // multi-vectors, but the kernel must not trust that invariant for
    // ad-hoc query matrices arriving over the wire.
    if query.iter().any(|row| row.len() != dim) {
        return 0.0;
    }
    if doc.iter().any(|row| row.len() != dim) {
        return 0.0;
    }

    let mut total = 0.0f32;
    for q in query {
        let mut best = f32::NEG_INFINITY;
        for d in doc {
            let s = dot_product(q, d);
            if s > best {
                best = s;
            }
        }
        // best is guaranteed finite because doc is non-empty and rejected
        // earlier if any row had a different length; dot_product on equal-
        // length non-empty slices returns a finite f32.
        total += best;
    }
    total
}

/// Per-query-token best-match similarities, in the same order as `query`.
///
/// Returns an empty vector for the same shape-mismatch cases as
/// [`maxsim`]. Useful for diagnostics, score decomposition, and tests
/// that want to assert on the intermediate per-token maxima.
#[inline]
pub fn maxsim_per_query_token(doc: &[Vec<f32>], query: &[Vec<f32>]) -> Vec<f32> {
    let Some(dim) = query.first().map(Vec::len) else {
        return Vec::new();
    };
    if dim == 0 || doc.is_empty() {
        return Vec::new();
    }
    if query.iter().any(|row| row.len() != dim) {
        return Vec::new();
    }
    if doc.iter().any(|row| row.len() != dim) {
        return Vec::new();
    }
    query
        .iter()
        .map(|q| {
            let mut best = f32::NEG_INFINITY;
            for d in doc {
                let s = dot_product(q, d);
                if s > best {
                    best = s;
                }
            }
            best
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
