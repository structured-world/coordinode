//! Reinserting a node that is already in the graph must leave it reachable.
//!
//! A node's vector reaches the index more than once when a write is
//! maintained by two paths (the statement that wrote it and the replicated
//! log both feed the index), and a `SET` of a vector property moves an
//! existing node. In both cases the node has to stay findable by its own
//! vector, the way a node inserted once is.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_core::graph::types::VectorMetric;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex, QuantizationCodec, RerankMode};

fn config() -> HnswConfig {
    HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ef_search: 200,
        metric: VectorMetric::L2,
        max_dimensions: 8,
        quantization: QuantizationCodec::None,
        rerank_candidates: 100,
        calibration_threshold: 100_000,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: 20_000,
        retired_bytes_budget: coordinode_vector::hnsw::DEFAULT_RETIRED_BYTES_BUDGET,
    }
}

/// Deterministic 8-dimensional vector, coordinates independent uniform draws
/// (splitmix64), so every miss is the graph's fault and not the data's.
fn v(seed: u64) -> Vec<f32> {
    (0..8u64)
        .map(|d| {
            let mut z = (seed * 8 + d).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f32 / (1u64 << 53) as f32
        })
        .collect()
}

/// Ids of `ids` whose own vector does not bring them into the top 10.
fn unreachable(
    idx: &HnswIndex,
    ids: std::ops::Range<u64>,
    vector: impl Fn(u64) -> Vec<f32>,
) -> Vec<u64> {
    ids.filter(|&id| !idx.search(&vector(id), 10).iter().any(|r| r.id == id))
        .collect()
}

/// Build 6000 nodes, then insert 2000 more while each one is inserted again
/// 256 inserts later with `second(id)`, as a lagging second maintenance path
/// or a later `SET` would.
fn build_with_late_reinserts(second: impl Fn(u64) -> Vec<f32>) -> HnswIndex {
    let mut idx = HnswIndex::new(config());
    for id in 0..6000 {
        idx.insert(id, v(id));
    }
    for id in 6000..8000 {
        idx.insert(id, v(id));
        if id >= 6256 {
            idx.insert(id - 256, second(id - 256));
        }
    }
    for id in 7744..8000 {
        idx.insert(id, second(id));
    }
    idx
}

/// The same vector inserted again leaves the node where it is. Rewiring it
/// used to cut it off the graph for about one node in sixty.
#[test]
fn inserting_the_same_vector_again_keeps_the_node_reachable() {
    let idx = build_with_late_reinserts(v);
    assert_eq!(idx.len(), 8000);
    let lost = unreachable(&idx, 6000..8000, v);
    assert!(lost.is_empty(), "{} unreachable: {lost:?}", lost.len());
}

/// A node moved to a new vector is found there as reliably as a node
/// inserted at that vector. The rewiring used to take the node itself as its
/// nearest neighbour, which left one node in twenty with only a self-loop.
#[test]
fn a_node_moved_to_a_new_vector_is_reachable_there() {
    let moved = |id: u64| v(id + 1_000_000);
    let idx = build_with_late_reinserts(moved);
    assert_eq!(idx.len(), 8000);
    let lost = unreachable(&idx, 6000..8000, moved);
    assert!(lost.is_empty(), "{} unreachable: {lost:?}", lost.len());
}
