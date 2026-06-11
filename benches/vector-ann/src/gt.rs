//! Brute-force ground-truth computation shared by the bench bins.
//!
//! The in-process bench recomputes groundtruth when `--subset-size`
//! truncates the train set (the published full-dataset ivecs reference
//! neighbours outside the subset). The server-mode loader reuses the
//! same kernel to WRITE a subset groundtruth file so `bench-vector-grpc`
//! can measure recall against the exact data that was loaded.

use coordinode_core::graph::types::VectorMetric;

/// Exact k-NN over the full train set for every query, parallel over
/// queries. Returns a row-major `n_query x k` matrix of train row ids.
pub fn brute_force_gt(
    train: &[f32],
    queries: &[f32],
    d: usize,
    k: usize,
    metric: VectorMetric,
) -> Vec<i32> {
    use rayon::prelude::*;

    let n_train = train.len() / d;
    let n_query = queries.len() / d;
    let mut result = vec![0i32; n_query * k];

    // The recompute is NOT part of any measured phase, so it must not
    // inherit the bench's `--threads` cap: a dedicated full-width pool
    // keeps a T1 sweep from paying tens of seconds of single-core
    // brute force per cell (48s measured on glove-100-50k before).
    // Falls back to the ambient pool if the builder fails.
    let pool = rayon::ThreadPoolBuilder::new().build().ok();
    let mut compute = || {
        result
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(q_idx, out)| {
                use coordinode_vector::metrics;
                let q = &queries[q_idx * d..(q_idx + 1) * d];
                // SIMD-dispatched kernels from the engine: the scalar
                // helpers below are kept only for the tiny recall checks.
                let q_norm = metrics::norm_l2(q);
                let mut scored: Vec<(f32, i32)> = (0..n_train)
                    .map(|i| {
                        let v = &train[i * d..(i + 1) * d];
                        let score = match metric {
                            VectorMetric::L2 => metrics::euclidean_distance_squared(q, v),
                            VectorMetric::L1 => metrics::manhattan_distance(q, v),
                            // Cosine similarity / dot product both ordered
                            // descending; negate so smaller-is-closer
                            // matches the L2 / L1 path.
                            VectorMetric::Cosine => {
                                -metrics::cosine_similarity_with_query_norm(q, v, q_norm)
                            }
                            VectorMetric::DotProduct => -metrics::dot_product(q, v),
                        };
                        (score, i as i32)
                    })
                    .collect();
                let take = k.min(scored.len());
                if take < scored.len() {
                    scored.select_nth_unstable_by(take, |a, b| a.0.total_cmp(&b.0));
                }
                scored[..take].sort_by(|a, b| a.0.total_cmp(&b.0));
                for (i, (_, idx)) in scored[..take].iter().enumerate() {
                    out[i] = *idx;
                }
            });
    };
    match pool {
        Some(p) => p.install(&mut compute),
        None => compute(),
    }

    result
}

pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x - *y;
            d * d
        })
        .sum()
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x * *y;
        na += *x * *x;
        nb += *y * *y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}
