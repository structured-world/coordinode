//! Brute-force ground-truth computation shared by the bench bins.
//!
//! The in-process bench recomputes groundtruth when `--subset-size`
//! truncates the train set (the published full-dataset ivecs reference
//! neighbours outside the subset). The server-mode loader reuses the
//! same kernel to WRITE a subset groundtruth file so `bench-vector-grpc`
//! can measure recall against the exact data that was loaded.

use coordinode_core::graph::types::VectorMetric;
use coordinode_vector::hnsw::VectorDomainError;

/// Exact k-NN over the full train set for every query, parallel over
/// queries: the engine's exact evaluation and order (ranking distance, equal
/// distances by row id), the reference an exact search answers. Returns a
/// row-major `n_query x k` matrix of train row ids.
///
/// # Errors
///
/// A query or train row the metric cannot evaluate.
pub fn brute_force_gt(
    train: &[f32],
    queries: &[f32],
    d: usize,
    k: usize,
    metric: VectorMetric,
) -> Result<Vec<i32>, VectorDomainError> {
    use coordinode_vector::metrics::exact::{ExactQuery, ranked_order};
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
            .try_for_each(|(q_idx, out)| {
                let q = ExactQuery::new(&queries[q_idx * d..(q_idx + 1) * d], metric)?;
                let mut scored: Vec<(f64, i32)> = (0..n_train)
                    .map(|i| Ok((q.distance(&train[i * d..(i + 1) * d])?, i as i32)))
                    .collect::<Result<_, VectorDomainError>>()?;
                let order =
                    |a: &(f64, i32), b: &(f64, i32)| ranked_order(a.0, a.1 as u64, b.0, b.1 as u64);
                let take = k.min(scored.len());
                if take < scored.len() {
                    scored.select_nth_unstable_by(take, order);
                }
                scored[..take].sort_by(order);
                for (i, (_, idx)) in scored[..take].iter().enumerate() {
                    out[i] = *idx;
                }
                Ok(())
            })
    };
    match pool {
        Some(p) => p.install(&mut compute),
        None => compute(),
    }?;

    Ok(result)
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
