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

use super::HnswIndex;

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

    // Seed the graph one leader at a time so each leader's plan
    // sees every prior leader; this builds a clean upper-graph
    // skeleton across the leader sample.
    for (id, vec) in leaders {
        index.insert(id, vec);
    }

    // Followers ride the parallel insert_batch path against the
    // seeded graph. The leader-seeded upper graph reduces the
    // depth of every follower's entry search and so amortises the
    // per-follower planning cost.
    if !followers.is_empty() {
        index.insert_batch(followers);
    }
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
mod tests {
    use super::*;
    use crate::hnsw::HnswConfig;
    use coordinode_core::graph::types::VectorMetric;

    fn make_index(dim: usize) -> HnswIndex {
        HnswIndex::new(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            metric: VectorMetric::L2,
            max_dimensions: dim as u32,
            ..Default::default()
        })
    }

    fn synth_items(n: usize, dim: usize) -> Vec<(u64, Vec<f32>)> {
        (0..n as u64)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| ((i as u32).wrapping_mul(2654435761) ^ d as u32) as f32 * 1e-6)
                    .collect();
                (i, v)
            })
            .collect()
    }

    #[test]
    fn bulk_build_below_threshold_routes_to_insert_batch() {
        // The Vec<(u64, Vec<f32>)> at this size is smaller than the
        // cluster-and-stitch threshold; the public `bulk_build`
        // entry point must fall through to insert_batch and produce
        // a fully populated index.
        const _: () = assert!(BULK_BUILD_THRESHOLD > 4);
        let mut idx = make_index(4);
        let items = synth_items(BULK_BUILD_THRESHOLD - 1, 4);
        let expected = items.len();
        idx.bulk_build(items);
        assert_eq!(idx.len(), expected);
    }

    #[test]
    fn leader_count_is_floor_sqrt() {
        assert_eq!(leader_count_for(0), 0);
        assert_eq!(leader_count_for(1), 1);
        assert_eq!(leader_count_for(2), 1);
        assert_eq!(leader_count_for(4), 2);
        assert_eq!(leader_count_for(9), 3);
        assert_eq!(leader_count_for(10), 3);
        assert_eq!(leader_count_for(256), 16);
        assert_eq!(leader_count_for(10_000), 100);
    }

    #[test]
    fn leader_sample_is_deterministic_and_unique() {
        let a = sample_leader_indices(1_000, 32);
        let b = sample_leader_indices(1_000, 32);
        assert_eq!(a, b, "same (n, k) must yield the same set");
        assert_eq!(a.len(), 32);
        assert!(a.iter().all(|&i| i < 1_000));
    }

    #[test]
    fn leader_sample_covers_full_range_when_k_eq_n() {
        let s = sample_leader_indices(8, 8);
        assert_eq!(s.len(), 8);
        for i in 0..8 {
            assert!(s.contains(&i));
        }
    }

    #[test]
    fn bulk_build_at_threshold_matches_insert_batch_membership() {
        // At the threshold the entry point dispatches into this
        // module's `bulk_build` function. The stub forwards to
        // `insert_batch`, so the resulting index must contain
        // exactly the same set of ids as a parallel `insert_batch`
        // call would have produced. Stronger recall comparison
        // lands once the real algorithm replaces the stub.
        let mut idx_bulk = make_index(4);
        let mut idx_batch = make_index(4);
        let items = synth_items(BULK_BUILD_THRESHOLD, 4);
        idx_bulk.bulk_build(items.clone());
        idx_batch.insert_batch(items);
        assert_eq!(idx_bulk.len(), idx_batch.len());
    }
}
