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

    result
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(q_idx, out)| {
            let q = &queries[q_idx * d..(q_idx + 1) * d];
            let mut scored: Vec<(f32, i32)> = (0..n_train)
                .map(|i| {
                    let v = &train[i * d..(i + 1) * d];
                    let score = match metric {
                        VectorMetric::L2 => l2_sq(q, v),
                        VectorMetric::L1 => l1(q, v),
                        // Cosine similarity / dot product both ordered
                        // descending; negate so smaller-is-closer
                        // matches the L2 / L1 path.
                        VectorMetric::Cosine => -cosine_sim(q, v),
                        VectorMetric::DotProduct => -dot(q, v),
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

pub fn l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (*x - *y).abs()).sum()
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
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
