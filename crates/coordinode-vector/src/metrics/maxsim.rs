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
mod tests {
    use super::*;

    fn vec_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn empty_inputs_score_zero() {
        let q = vec![vec![1.0, 0.0]];
        let d = vec![vec![1.0, 0.0]];
        assert_eq!(maxsim(&[], &q), 0.0);
        assert_eq!(maxsim(&d, &[]), 0.0);
        assert_eq!(maxsim(&[], &[]), 0.0);
    }

    #[test]
    fn mismatched_dim_scores_zero() {
        let q = vec![vec![1.0, 0.0]];
        let d_bad = vec![vec![1.0, 0.0, 0.0]];
        assert_eq!(maxsim(&d_bad, &q), 0.0);
        let q_mixed = vec![vec![1.0, 0.0], vec![1.0]];
        let d = vec![vec![1.0, 0.0]];
        assert_eq!(maxsim(&d, &q_mixed), 0.0);
    }

    #[test]
    fn zero_dim_rows_score_zero() {
        let q: Vec<Vec<f32>> = vec![vec![]];
        let d: Vec<Vec<f32>> = vec![vec![]];
        assert_eq!(maxsim(&d, &q), 0.0);
    }

    #[test]
    fn single_pair_equals_dot_product() {
        let q = vec![vec![1.0, 2.0, 3.0]];
        let d = vec![vec![4.0, 5.0, 6.0]];
        // Single q, single d: max = dot. Sum across 1 q-token = dot.
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
        assert!(vec_eq(maxsim(&d, &q), expected, 1e-5));
    }

    #[test]
    fn max_picks_best_doc_token_per_query() {
        // Hand-computed example.
        // q0 = [1, 0], q1 = [0, 1]
        // d0 = [1, 0], d1 = [0, 1], d2 = [0.5, 0.5]
        // q0 . d0 = 1, q0 . d1 = 0, q0 . d2 = 0.5 -> max = 1
        // q1 . d0 = 0, q1 . d1 = 1, q1 . d2 = 0.5 -> max = 1
        // total = 2
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let d = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        assert!(vec_eq(maxsim(&d, &q), 2.0, 1e-5));
    }

    #[test]
    fn per_query_token_decomposition_matches_sum() {
        let q = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let d = vec![
            vec![0.9, 0.1, 0.0],
            vec![0.1, 0.9, 0.0],
            vec![0.0, 0.1, 0.9],
        ];
        let per_q = maxsim_per_query_token(&d, &q);
        assert_eq!(per_q.len(), 3);
        let total = maxsim(&d, &q);
        let sum: f32 = per_q.iter().sum();
        assert!(vec_eq(total, sum, 1e-5));
    }

    #[test]
    fn cosine_semantics_on_normalised_rows() {
        // Unit-norm rows; dot product equals cosine similarity.
        let q = vec![vec![1.0, 0.0]];
        let d = vec![vec![0.6, 0.8]]; // 0.6^2 + 0.8^2 = 1.0, unit norm
                                      // cos angle between q and d = 0.6
        assert!(vec_eq(maxsim(&d, &q), 0.6, 1e-5));
    }

    #[test]
    fn larger_random_doc_score_finite_and_correct() {
        // Synthetic but deterministic input; verify the kernel's sum
        // matches a naive triple-loop reference implementation.
        let dim = 8;
        let q: Vec<Vec<f32>> = (0..4)
            .map(|i| (0..dim).map(|j| ((i + j) as f32) * 0.1).collect())
            .collect();
        let d: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..dim).map(|j| ((i * 2 + j) as f32) * 0.05).collect())
            .collect();

        // Reference (naive triple loop) without going through dot_product.
        let mut reference = 0.0f32;
        for q_row in &q {
            let mut best = f32::NEG_INFINITY;
            for d_row in &d {
                let s: f32 = q_row.iter().zip(d_row.iter()).map(|(x, y)| x * y).sum();
                if s > best {
                    best = s;
                }
            }
            reference += best;
        }

        let got = maxsim(&d, &q);
        assert!(
            vec_eq(got, reference, 1e-4),
            "maxsim={got} reference={reference}"
        );
        assert!(got.is_finite());
    }
}
