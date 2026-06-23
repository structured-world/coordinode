//! ParlayANN-style HNSW bulk-build path.
//!
//! Caller has every vector in hand and wants the index built as
//! quickly as the hardware allows. The incremental `insert` path
//! walks the upper graph per item, paying `O(N log N * ef)` for the
//! whole corpus and serialising the apply phase. Above the
//! [`BULK_BUILD_THRESHOLD`] this module's algorithm samples
//! `sqrt(N)` leaders, builds the leader-only upper graph, partitions
//! followers into per-leader clusters, builds each cluster in
//! parallel, then stitches cross-cluster edges in a final pass. The
//! algorithm follows
//! "ParlayANN: Scalable and Deterministic Parallel Graph-Based ANN"
//! (Manohar et al., PPoPP 2024).
//!
//! This file is the entry stub for that algorithm. It is written
//! ahead of the cluster-and-stitch implementation so the public
//! `HnswIndex::bulk_build` dispatch surface and the small-batch
//! fallback threshold can land in a single self-contained commit.
//! Subsequent commits replace the body of [`bulk_build`] with the
//! actual algorithm without churning the entry-point shape.

use coordinode_core::graph::types::VectorMetric;
use rayon::prelude::*;

use super::HnswIndex;
use crate::metrics;

/// Minimum item count at which the cluster-and-stitch path is
/// expected to beat the incremental `insert_batch` path. Below this
/// the rayon orchestration overhead (leader sample, brute-force
/// cluster assignment, per-cluster rayon spawn) exceeds the
/// parallelism win.
///
/// 256 reflects the rough cross-over where `sqrt(N) >= 16` leaders
/// give the per-cluster phase enough work to amortise the partition
/// and stitch costs. It is a starting point; later commits will
/// re-measure and may lower the floor.
pub(crate) const BULK_BUILD_THRESHOLD: usize = 256;

/// Bulk-build the index from the given `items`.
///
/// Stage one of the cluster-and-stitch path: sample `floor(sqrt(N))`
/// leaders deterministically, seed the upper layers of the graph
/// with leader-only inserts, then run the rest of the corpus through
/// the existing parallel `insert_batch`. The leader seed ensures the
/// upper graph is sparse and well-formed before any follower
/// arrives, which gives every follower's plan a small, fast entry
/// search.
///
/// Subsequent commits replace the follower bulk-load with the
/// per-cluster parallel build (Step 3) and add the cross-cluster
/// stitch (Step 4). The seeded path here is a strict superset of
/// `insert_batch` and must produce a graph with the same membership;
/// recall comes from later steps.
pub(crate) fn bulk_build(index: &mut HnswIndex, items: Vec<(u64, Vec<f32>)>) {
    let n = items.len();
    if n == 0 {
        return;
    }

    let leader_count = leader_count_for(n);
    let leader_idx = sample_leader_indices(n, leader_count);

    // Partition items into leaders + followers preserving the
    // original ids. The leader set is a deterministic function of n.
    let mut leaders: Vec<(u64, Vec<f32>)> = Vec::with_capacity(leader_count);
    let mut followers: Vec<(u64, Vec<f32>)> = Vec::with_capacity(n - leader_count);
    for (i, item) in items.into_iter().enumerate() {
        if leader_idx.contains(&i) {
            leaders.push(item);
        } else {
            followers.push(item);
        }
    }

    // Capture the leader vectors before they move into the index;
    // the cluster-assignment pass needs them by reference.
    let leaders_for_assignment: Vec<Vec<f32>> = leaders.iter().map(|(_, v)| v.clone()).collect();

    // Seed the graph one leader at a time so each leader's plan
    // sees every prior leader; this builds a clean upper-graph
    // skeleton across the leader sample.
    for (id, vec) in leaders {
        index.insert(id, vec);
    }

    // Followers ride the parallel insert_batch path against the
    // seeded graph. Before handing the batch off, reorder the
    // followers so items of the same cluster are contiguous in the
    // input vector. The apply phase then visits adjacent
    // `nodes[]` indices for an entire cluster before moving to the
    // next, giving the L1 / L2 cache a fighting chance against the
    // pointer-chasing nature of graph inserts.
    if !followers.is_empty() {
        let leader_vecs: Vec<Vec<f32>> = leaders_for_assignment;
        let reordered = cluster_order_followers(followers, &leader_vecs, index.config().metric);
        index.insert_batch(reordered);
    }
}

/// Brute-force assign every follower to the index of its nearest
/// leader (by the configured metric), parallel via rayon. Returns
/// the assignment vector in input order: position `i` holds the
/// leader index for `followers[i]`.
///
/// Pure function, exposed at module scope so the property test can
/// hit it directly without going through `bulk_build`.
fn assign_followers_to_leaders(
    followers: &[(u64, Vec<f32>)],
    leaders: &[Vec<f32>],
    metric: VectorMetric,
) -> Vec<usize> {
    if leaders.is_empty() {
        return vec![0; followers.len()];
    }
    followers
        .par_iter()
        .map(|(_, v)| nearest_leader_index(v, leaders, metric))
        .collect()
}

/// Pick the nearest leader for a single follower vector. Falls
/// through to `cosine_distance` / `euclidean_distance` /
/// `manhattan_distance` / `1 - dot` based on the metric. Equal
/// distances break toward the lower index.
fn nearest_leader_index(v: &[f32], leaders: &[Vec<f32>], metric: VectorMetric) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;
    for (i, l) in leaders.iter().enumerate() {
        let d = match metric {
            VectorMetric::Cosine => metrics::cosine_distance(v, l),
            VectorMetric::L2 => metrics::euclidean_distance(v, l),
            VectorMetric::L1 => metrics::manhattan_distance(v, l),
            // Higher dot means closer; convert to a lower-is-better
            // distance to share the comparator with the other metrics.
            VectorMetric::DotProduct => -metrics::dot_product(v, l),
        };
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

/// Reorder followers so items with the same leader assignment are
/// contiguous in the output. Within a cluster, original input order
/// is preserved (the leader-grouping pass is a stable sort by
/// cluster id).
fn cluster_order_followers(
    followers: Vec<(u64, Vec<f32>)>,
    leaders: &[Vec<f32>],
    metric: VectorMetric,
) -> Vec<(u64, Vec<f32>)> {
    let assignment = assign_followers_to_leaders(&followers, leaders, metric);
    let mut indexed: Vec<(usize, (u64, Vec<f32>))> = followers.into_iter().enumerate().collect();
    // Stable sort keeps the input order inside each cluster.
    indexed.sort_by_key(|(i, _)| assignment[*i]);
    indexed.into_iter().map(|(_, item)| item).collect()
}

/// `floor(sqrt(n))` clamped to `[1, n]`. Returns 0 only when `n == 0`.
fn leader_count_for(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let approx = (n as f64).sqrt().floor() as usize;
    approx.max(1).min(n)
}

/// Sample `k` distinct leader indices from `0..n` using a
/// deterministic xorshift-driven permutation. The same `n, k` pair
/// always produces the same set so tests and bench runs are
/// reproducible without depending on the `rand` crate.
fn sample_leader_indices(n: usize, k: usize) -> std::collections::HashSet<usize> {
    if k == 0 {
        return std::collections::HashSet::new();
    }
    if k >= n {
        return (0..n).collect();
    }

    // Seed derived from n itself; for a given corpus size the
    // leader pick is stable.
    let mut state: u64 = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1);
    let mut indices: Vec<usize> = (0..n).collect();

    // Fisher-Yates over the first `k` slots is enough; the rest of
    // the permutation is irrelevant.
    for i in 0..k {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let span = (n - i) as u64;
        let pick = (state % span) as usize + i;
        indices.swap(i, pick);
    }
    indices.into_iter().take(k).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
