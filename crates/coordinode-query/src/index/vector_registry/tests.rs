use super::*;
use crate::index::VectorIndexConfig;
use coordinode_cluster::{PartitionSet, VectorShardRouter};
use coordinode_core::graph::types::VectorMetric;
use std::sync::Arc;

/// Test-double router: two partitions split on the sign of the first
/// coordinate; a near-zero first coordinate (`|x| < 0.5`) is a boundary
/// point replicated to BOTH partitions (closure replication) and a
/// boundary query fans out to both (adaptive fan-out). Deterministic, no
/// EE dependency.
struct TwoPartitionRouter;

impl VectorShardRouter for TwoPartitionRouter {
    fn assign(&self, v: &[f32]) -> PartitionSet {
        let x = v.first().copied().unwrap_or(0.0);
        let mut s = PartitionSet::new();
        if x.abs() < 0.5 {
            s.push(0);
            s.push(1);
        } else if x < 0.0 {
            s.push(0);
        } else {
            s.push(1);
        }
        s
    }

    fn route(&self, q: &[f32], _top_m: usize) -> PartitionSet {
        self.assign(q)
    }

    fn n_partitions(&self) -> usize {
        2
    }
}

fn test_config() -> VectorIndexConfig {
    VectorIndexConfig {
        dimensions: 3,
        metric: VectorMetric::Cosine,
        m: 16,
        ef_construction: 200,
        quantization: coordinode_vector::hnsw::QuantizationCodec::None,
        offload_vectors: false,
        ef_search: None,
        rerank_candidates: None,
    }
}

#[test]
fn register_and_lookup() {
    let reg = VectorIndexRegistry::new();
    let def = IndexDefinition::hnsw("movie_embedding", "Movie", "embedding", test_config());

    reg.register(def);

    assert!(reg.has_index("Movie", "embedding"));
    assert!(!reg.has_index("Movie", "title"));
    assert!(!reg.has_index("User", "embedding"));
    assert_eq!(reg.len(), 1);
}

#[test]
fn health_tracks_state_and_freshness_watermark() {
    let reg = VectorIndexRegistry::new();
    reg.register(IndexDefinition::hnsw(
        "movie_embedding",
        "Movie",
        "embedding",
        test_config(),
    ));

    // Freshly registered → Ready with a zero watermark.
    assert_eq!(
        reg.health_snapshot("Movie", "embedding"),
        Some(IndexHealthState::Ready { indexed_hlc: 0 })
    );
    // Unknown index → no health.
    assert!(reg.health_snapshot("Movie", "missing").is_none());

    // Worker cursor advance lifts the watermark for every maintained index.
    reg.advance_indexed_hlc_all(4_200);
    assert_eq!(
        reg.health_snapshot("Movie", "embedding"),
        Some(IndexHealthState::Ready { indexed_hlc: 4_200 })
    );

    // Rebuild keeps the already-folded watermark.
    reg.report_health_rebuild("Movie", "embedding", 0.42, 1_000);
    let snap = reg.health_snapshot("Movie", "embedding");
    assert!(
        matches!(snap, Some(IndexHealthState::Rebuilding { .. })),
        "expected Rebuilding, got {snap:?}"
    );
    if let Some(IndexHealthState::Rebuilding {
        progress,
        indexed_hlc,
        ..
    }) = snap
    {
        assert!((progress - 0.42).abs() < 1e-3);
        assert_eq!(indexed_hlc, 4_200);
    }

    // Completion returns to Ready, watermark intact.
    reg.mark_health_ready("Movie", "embedding");
    assert_eq!(
        reg.health_snapshot("Movie", "embedding"),
        Some(IndexHealthState::Ready { indexed_hlc: 4_200 })
    );

    // Unregister drops the health entry.
    reg.unregister("Movie", "embedding");
    assert!(reg.health_snapshot("Movie", "embedding").is_none());
}

#[test]
fn search_empty_index_returns_empty() {
    let reg = VectorIndexRegistry::new();
    let def = IndexDefinition::hnsw("movie_embedding", "Movie", "embedding", test_config());
    reg.register(def);

    let results = reg.search("Movie", "embedding", &[1.0, 0.0, 0.0], 10);
    assert!(results.is_some());
    assert!(results.unwrap().is_empty());
}

#[test]
fn insert_and_search() {
    let reg = VectorIndexRegistry::new();
    let def = IndexDefinition::hnsw("movie_embedding", "Movie", "embedding", test_config());
    reg.register(def);

    // Insert vectors
    reg.on_vector_written("Movie", NodeId::from_raw(1), "embedding", &[1.0, 0.0, 0.0]);
    reg.on_vector_written("Movie", NodeId::from_raw(2), "embedding", &[0.0, 1.0, 0.0]);
    reg.on_vector_written("Movie", NodeId::from_raw(3), "embedding", &[0.9, 0.1, 0.0]);

    // Search for vector closest to [1.0, 0.0, 0.0]
    let results = reg
        .search("Movie", "embedding", &[1.0, 0.0, 0.0], 2)
        .expect("search should return results");

    assert_eq!(results.len(), 2);
    // Node 1 should be closest (exact match), then node 3
    assert_eq!(results[0].id, 1);
    assert_eq!(results[1].id, 3);
}

#[test]
fn indexes_for_label() {
    let reg = VectorIndexRegistry::new();
    reg.register(IndexDefinition::hnsw(
        "movie_embed",
        "Movie",
        "embedding",
        test_config(),
    ));
    reg.register(IndexDefinition::hnsw(
        "movie_thumb",
        "Movie",
        "thumbnail_vec",
        test_config(),
    ));
    reg.register(IndexDefinition::hnsw(
        "user_embed",
        "User",
        "embedding",
        test_config(),
    ));

    let movie_idxs = reg.indexes_for_label("Movie");
    assert_eq!(movie_idxs.len(), 2);

    let user_idxs = reg.indexes_for_label("User");
    assert_eq!(user_idxs.len(), 1);
}

#[test]
fn unregister() {
    let reg = VectorIndexRegistry::new();
    reg.register(IndexDefinition::hnsw(
        "movie_embed",
        "Movie",
        "embedding",
        test_config(),
    ));
    assert!(reg.has_index("Movie", "embedding"));

    reg.unregister("Movie", "embedding");
    assert!(!reg.has_index("Movie", "embedding"));
    assert_eq!(reg.len(), 0);
}

#[test]
fn no_index_search_returns_none() {
    let reg = VectorIndexRegistry::new();
    let results = reg.search("Movie", "embedding", &[1.0, 0.0, 0.0], 10);
    assert!(results.is_none());
}

#[test]
fn register_sharded_marks_label_sharded() {
    let reg = VectorIndexRegistry::new();
    reg.register_sharded(
        IndexDefinition::hnsw("emb", "Doc", "embedding", test_config()),
        Arc::new(TwoPartitionRouter),
    );
    assert!(reg.is_sharded("Doc", "embedding"));
    assert!(reg.has_index("Doc", "embedding"));
    // A sharded key lives in the sharded map, not the single-index map.
    assert!(reg.get("Doc", "embedding").is_none());
    assert_eq!(reg.len(), 1);

    // Unregister clears the sharded entry too.
    reg.unregister("Doc", "embedding");
    assert!(!reg.is_sharded("Doc", "embedding"));
    assert!(!reg.has_index("Doc", "embedding"));
}

#[test]
fn unsharded_register_is_not_sharded() {
    let reg = VectorIndexRegistry::new();
    reg.register(IndexDefinition::hnsw(
        "emb",
        "Doc",
        "embedding",
        test_config(),
    ));
    assert!(!reg.is_sharded("Doc", "embedding"));
    assert!(reg.get("Doc", "embedding").is_some());
}

#[test]
fn sharded_search_matches_single_index_top1() {
    let vectors = [
        (1u64, vec![-1.0, 0.0, 0.0]),
        (2, vec![-0.9, 0.1, 0.0]),
        (3, vec![-0.8, 0.0, 0.1]),
        (4, vec![1.0, 0.0, 0.0]),
        (5, vec![0.9, 0.1, 0.0]),
        (6, vec![0.8, 0.0, 0.1]),
    ];

    // Single-index baseline.
    let single = VectorIndexRegistry::new();
    single.register(IndexDefinition::hnsw(
        "emb",
        "Doc",
        "embedding",
        test_config(),
    ));
    single.bulk_insert("Doc", "embedding", vectors.iter().cloned());

    // Sharded index, 2 partitions split on the first-coord sign.
    let sharded = VectorIndexRegistry::new();
    sharded.register_sharded(
        IndexDefinition::hnsw("emb", "Doc", "embedding", test_config()),
        Arc::new(TwoPartitionRouter),
    );
    sharded.bulk_insert("Doc", "embedding", vectors.iter().cloned());

    for q in [vec![-1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]] {
        let base = single.search("Doc", "embedding", &q, 1).unwrap();
        let sh = sharded.search("Doc", "embedding", &q, 1).unwrap();
        assert_eq!(sh.len(), 1);
        assert_eq!(
            sh[0].id, base[0].id,
            "sharded top-1 must match the single-index baseline"
        );
    }
}

#[test]
fn sharded_dedup_replicated_boundary_id() {
    let reg = VectorIndexRegistry::new();
    reg.register_sharded(
        IndexDefinition::hnsw("emb", "Doc", "embedding", test_config()),
        Arc::new(TwoPartitionRouter),
    );
    // id 7 sits on the boundary (|x| < 0.5) -> replicated into BOTH
    // partitions; id 4 lands only in the positive partition.
    reg.bulk_insert(
        "Doc",
        "embedding",
        [(4u64, vec![1.0, 0.0, 0.0]), (7, vec![0.0, 1.0, 0.0])].into_iter(),
    );
    // The boundary query fans out to both partitions, so id 7 is found in
    // each; the merge must dedup it to a single result.
    let results = reg
        .search("Doc", "embedding", &[0.0, 1.0, 0.0], 10)
        .unwrap();
    let sevens = results.iter().filter(|r| r.id == 7).count();
    assert_eq!(sevens, 1, "a replicated id must appear once after merge");
}

#[test]
fn sharded_search_with_visibility_filters_hidden_ids() {
    let reg = VectorIndexRegistry::new();
    reg.register_sharded(
        IndexDefinition::hnsw("emb", "Doc", "embedding", test_config()),
        Arc::new(TwoPartitionRouter),
    );
    reg.bulk_insert(
        "Doc",
        "embedding",
        [
            (1u64, vec![-1.0, 0.0, 0.0]),
            (2, vec![-0.9, 0.1, 0.0]),
            (4, vec![1.0, 0.0, 0.0]),
        ]
        .into_iter(),
    );

    // The query routes to the negative partition {1, 2}. Hiding id 1 must
    // prune it from the scatter-gathered, merged result.
    let filtered = reg
        .search_with_visibility("Doc", "embedding", &[-1.0, 0.0, 0.0], 5, 2.0, 4, |id| {
            id != 1
        })
        .unwrap();
    assert!(
        filtered.iter().all(|r| r.id != 1),
        "hidden id must not appear in sharded filtered search"
    );
    assert!(
        filtered.iter().any(|r| r.id == 2),
        "a visible neighbour is still returned"
    );

    // Unfiltered, the exact match (id 1) is the top hit through the
    // sharded visibility path.
    let unfiltered = reg
        .search_with_visibility("Doc", "embedding", &[-1.0, 0.0, 0.0], 5, 2.0, 4, |_| true)
        .unwrap();
    assert_eq!(
        unfiltered[0].id, 1,
        "unfiltered sharded top-1 is the exact match"
    );
}
