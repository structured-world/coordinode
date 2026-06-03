//! End-to-end RaBitQ + LSM truth tier wiring.
//!
//! Closes the R861 coverage gap: HNSW unit tests exercised RaBitQ
//! without a tier (in-RAM only); LSM tier tests exercised put/get
//! without HNSW. Neither proved the two work together — that the
//! same HnswIndex configured with `QuantizationCodec::RaBitQ` and a
//! `VectorTierHandle` both calibrates RaBitQ and persists every
//! original f32 vector through the tier per ADR-033.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use coordinode_core::graph::types::VectorMetric;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex, QuantizationCodec};
use coordinode_vector::storage::lsm_backed::LsmVectorTier;
use coordinode_vector::storage::{VectorTierHandle, VectorTierStorage};

fn open_engine() -> (Arc<StorageEngine>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    (Arc::new(engine), dir)
}

/// `dim` must be a multiple of 64 — RaBitQ rejects calibration
/// otherwise. 64 is the smallest valid dim and keeps the test
/// fast.
const DIM: usize = 64;

/// `n` strictly above `calibration_threshold` so RaBitQ has
/// definitely fired by the assertion phase.
const N: u64 = 1500;

/// Deterministic per-node f32 vector: each component encodes
/// `(node_id, dim_idx)` so we can verify exactly the bytes the
/// caller inserted came back through the tier.
fn synth_vec(node_id: u64) -> Vec<f32> {
    (0..DIM)
        .map(|i| (node_id as f32) * 0.01 + (i as f32) * 0.001)
        .collect()
}

#[test]
fn rabitq_inserts_persist_originals_through_tier() {
    let (engine, _dir) = open_engine();

    let tier_backend: Arc<dyn VectorTierStorage> = Arc::new(LsmVectorTier::new(engine));
    // Arbitrary (label_id, property_id) — the tier handle just
    // namespaces the vec: keyspace, the actual values don't have
    // to match anything else in this test.
    let tier_handle = VectorTierHandle::new(Arc::clone(&tier_backend), 42, 7);

    let mut index = HnswIndex::new(HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 64,
        ef_search: 32,
        metric: VectorMetric::L2,
        max_dimensions: DIM as u32,
        quantization: QuantizationCodec::RaBitQ { bits: 1 },
        rerank_candidates: 32,
        // Low threshold so calibration fires partway through and
        // we exercise BOTH pre-calibration and post-calibration
        // insert paths against the tier.
        calibration_threshold: 500,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: coordinode_vector::hnsw::RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: N as u32,
    });
    index.set_vector_tier(Some(tier_handle));

    for id in 0..N {
        index.insert(id, synth_vec(id));
    }

    // RaBitQ side: codec must be active and calibrated.
    assert!(
        index.is_rabitq_active(),
        "RaBitQ must be active after {N} inserts (threshold=500)"
    );
    assert!(
        index.rabitq_params().is_some(),
        "calibration params must be populated"
    );

    // Tier side: every inserted f32 vector must round-trip
    // through the LSM exactly — bit-for-bit, not just close.
    // This is the load-bearing claim of ADR-033: the tier is
    // the truth, not the quantized in-RAM codec.
    let ids: Vec<u64> = (0..N).collect();
    let got = tier_backend
        .multi_get_f32(42, 7, &ids)
        .expect("tier multi_get_f32");

    assert_eq!(got.len(), N as usize);
    for (id, slot) in got.iter().enumerate() {
        let expected = synth_vec(id as u64);
        let actual = slot
            .as_ref()
            .unwrap_or_else(|| panic!("tier missing f32 for node {id}"));
        assert_eq!(actual, &expected, "tier returned wrong bytes for node {id}");
    }
}
