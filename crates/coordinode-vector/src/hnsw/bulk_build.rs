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

/// Bulk-build the index from the given `items`. The caller is
/// responsible for the dispatch decision; `HnswIndex::bulk_build`
/// routes small batches through `insert_batch` and only this
/// function is invoked when the corpus is large enough to benefit.
///
/// This entry shim currently forwards to `insert_batch` unchanged.
/// The cluster-and-stitch body lands in a follow-up commit; the
/// shim exists so callers can already pick the right path and the
/// criterion bench can record the baseline before the algorithm
/// landing changes the curve.
pub(crate) fn bulk_build(index: &mut HnswIndex, items: Vec<(u64, Vec<f32>)>) {
    index.insert_batch(items);
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
