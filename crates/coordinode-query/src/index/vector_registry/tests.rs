use super::*;
use crate::index::VectorIndexConfig;
use coordinode_cluster::{PartitionSet, VectorShardRouter};
use coordinode_core::graph::types::VectorMetric;

/// A merge across partitions ranks equal scores by id: the merged map's
/// iteration order is arbitrary, and the result must not follow it. The
/// nearer occurrence of an id replicated to two partitions wins, and a later
/// farther occurrence does not replace it.
#[test]
fn a_merge_ranks_equal_scores_by_id_and_keeps_the_nearer_occurrence() {
    let mut best = HashMap::new();
    let tied: Vec<SearchResult> = (0..50u64)
        .rev()
        .map(|id| SearchResult { id, score: 1.0 })
        .collect();
    VectorIndexRegistry::accumulate_best(&mut best, tied);
    VectorIndexRegistry::accumulate_best(
        &mut best,
        vec![
            SearchResult { id: 7, score: 0.5 },
            SearchResult { id: 3, score: 2.0 },
            SearchResult {
                id: 100,
                score: 3.0,
            },
        ],
    );

    let merged = VectorIndexRegistry::finalize_merge(best, 51);
    let ids: Vec<u64> = merged.iter().map(|r| r.id).collect();
    let mut expected = vec![7];
    expected.extend((0..50u64).filter(|id| *id != 7));
    expected.push(100);
    assert_eq!(ids, expected);
    assert_eq!(merged[0].score, 0.5, "the nearer occurrence of id 7 wins");
    assert_eq!(merged[3].score, 1.0, "the farther occurrence of id 3 loses");
}
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

/// `reg`'s search under no query budget.
fn search(
    reg: &VectorIndexRegistry,
    label: &str,
    property: &str,
    query: &[f32],
    k: usize,
) -> Option<Vec<SearchResult>> {
    reg.search(
        label,
        property,
        query,
        k,
        &mut coordinode_core::budget::Unmetered,
    )
    .expect("an evaluable query")
}

/// `reg`'s filtered search with overfetch 2.0 and 4 rounds, under no query
/// budget.
fn search_visible(
    reg: &VectorIndexRegistry,
    query: &[f32],
    k: usize,
    is_visible: impl Fn(u64) -> bool,
) -> Option<Vec<SearchResult>> {
    reg.search_with_visibility(
        "Doc",
        "embedding",
        query,
        k,
        2.0,
        4,
        is_visible,
        &mut coordinode_core::budget::Unmetered,
    )
    .expect("an evaluable query")
}

/// The vector index `name` as the index numbered `raw`, serving from
/// generation `raw`: the registry is driven without a catalog here.
fn hnsw(raw: u64, name: &str, label: &str, property: &str) -> IndexDefinition {
    crate::index::IndexDescriptor::hnsw(name, label, property, test_config()).bind(
        crate::index::IndexId::from_raw(raw),
        GenerationId::from_raw(raw),
    )
}

#[test]
fn register_and_lookup() {
    let reg = VectorIndexRegistry::new();
    let def = hnsw(1, "movie_embedding", "Movie", "embedding");

    reg.register(def);

    assert!(reg.has_index("Movie", "embedding"));
    assert!(!reg.has_index("Movie", "title"));
    assert!(!reg.has_index("User", "embedding"));
    assert_eq!(reg.len(), 1);
}

/// A generation is registered for its build once. `CREATE VECTOR INDEX` and
/// the member's own catalog follower both register the definition it
/// published; a second registration that replaced the graph and its health
/// signal left a search reading a graph no build was filling, while the
/// running build filled the one no longer served.
#[test]
fn registering_a_generation_for_its_build_twice_keeps_the_first() {
    let reg = VectorIndexRegistry::new();
    let def = hnsw(7, "doc_embedding", "Doc", "embedding");

    assert!(reg.register_for_build(def.clone(), None));
    let graph = reg.get("Doc", "embedding").expect("graph");
    let health = reg.health_handle("Doc", "embedding").expect("health");

    assert!(!reg.register_for_build(def, None));
    assert!(Arc::ptr_eq(
        &graph,
        &reg.get("Doc", "embedding").expect("graph")
    ));
    assert!(Arc::ptr_eq(
        &health,
        &reg.health_handle("Doc", "embedding").expect("health")
    ));

    // A new generation of the index is a new graph.
    assert!(reg.register_for_build(hnsw(8, "doc_embedding", "Doc", "embedding"), None));
    assert!(!Arc::ptr_eq(
        &graph,
        &reg.get("Doc", "embedding").expect("graph")
    ));
}

#[test]
fn health_tracks_state_and_freshness_watermark() {
    let reg = VectorIndexRegistry::new();
    reg.register(hnsw(1, "movie_embedding", "Movie", "embedding"));

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
    let def = hnsw(1, "movie_embedding", "Movie", "embedding");
    reg.register(def);

    let results = search(&reg, "Movie", "embedding", &[1.0, 0.0, 0.0], 10);
    assert!(results.is_some());
    assert!(results.unwrap().is_empty());
}

#[test]
fn insert_and_search() {
    let reg = VectorIndexRegistry::new();
    let def = hnsw(1, "movie_embedding", "Movie", "embedding");
    reg.register(def);

    // Insert vectors
    for (id, vector) in [
        (1, [1.0, 0.0, 0.0]),
        (2, [0.0, 1.0, 0.0]),
        (3, [0.9, 0.1, 0.0]),
    ] {
        reg.on_vector_written("Movie", NodeId::from_raw(id), "embedding", &vector)
            .unwrap();
    }

    // Search for vector closest to [1.0, 0.0, 0.0]
    let results = search(&reg, "Movie", "embedding", &[1.0, 0.0, 0.0], 2)
        .expect("search should return results");

    assert_eq!(results.len(), 2);
    // Node 1 should be closest (exact match), then node 3
    assert_eq!(results[0].id, 1);
    assert_eq!(results[1].id, 3);
}

#[test]
fn indexes_for_label() {
    let reg = VectorIndexRegistry::new();
    reg.register(hnsw(1, "movie_embed", "Movie", "embedding"));
    reg.register(hnsw(2, "movie_thumb", "Movie", "thumbnail_vec"));
    reg.register(hnsw(3, "user_embed", "User", "embedding"));

    let movie_idxs = reg.indexes_for_label("Movie");
    assert_eq!(movie_idxs.len(), 2);

    let user_idxs = reg.indexes_for_label("User");
    assert_eq!(user_idxs.len(), 1);
}

#[test]
fn unregister() {
    let reg = VectorIndexRegistry::new();
    reg.register(hnsw(1, "movie_embed", "Movie", "embedding"));
    assert!(reg.has_index("Movie", "embedding"));

    reg.unregister("Movie", "embedding");
    assert!(!reg.has_index("Movie", "embedding"));
    assert_eq!(reg.len(), 0);
}

#[test]
fn no_index_search_returns_none() {
    let reg = VectorIndexRegistry::new();
    let results = search(&reg, "Movie", "embedding", &[1.0, 0.0, 0.0], 10);
    assert!(results.is_none());
}

#[test]
fn register_sharded_marks_label_sharded() {
    let reg = VectorIndexRegistry::new();
    reg.register_sharded(
        hnsw(1, "emb", "Doc", "embedding"),
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
    reg.register(hnsw(1, "emb", "Doc", "embedding"));
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
    single.register(hnsw(1, "emb", "Doc", "embedding"));
    single
        .bulk_insert("Doc", "embedding", vectors.iter().cloned())
        .unwrap();

    // Sharded index, 2 partitions split on the first-coord sign.
    let sharded = VectorIndexRegistry::new();
    sharded.register_sharded(
        hnsw(1, "emb", "Doc", "embedding"),
        Arc::new(TwoPartitionRouter),
    );
    sharded
        .bulk_insert("Doc", "embedding", vectors.iter().cloned())
        .unwrap();

    for q in [vec![-1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]] {
        let base = search(&single, "Doc", "embedding", &q, 1).unwrap();
        let sh = search(&sharded, "Doc", "embedding", &q, 1).unwrap();
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
        hnsw(1, "emb", "Doc", "embedding"),
        Arc::new(TwoPartitionRouter),
    );
    // id 7 sits on the boundary (|x| < 0.5) -> replicated into BOTH
    // partitions; id 4 lands only in the positive partition.
    reg.bulk_insert(
        "Doc",
        "embedding",
        [(4u64, vec![1.0, 0.0, 0.0]), (7, vec![0.0, 1.0, 0.0])].into_iter(),
    )
    .unwrap();
    // The boundary query fans out to both partitions, so id 7 is found in
    // each; the merge must dedup it to a single result.
    let results = search(&reg, "Doc", "embedding", &[0.0, 1.0, 0.0], 10).unwrap();
    let sevens = results.iter().filter(|r| r.id == 7).count();
    assert_eq!(sevens, 1, "a replicated id must appear once after merge");
}

/// A scattered search holds the merge of its partitions' answers within the
/// statement's budget: after each partition answers, its results are charged
/// as merge scratch (an id and a score each) before they join the merge.
#[test]
fn a_scattered_search_charges_the_merge_of_its_partitions() {
    /// Records the scratch each step asks for.
    #[derive(Default)]
    struct Recorded(Vec<u64>);
    impl coordinode_core::budget::Meter for Recorded {
        type Stop = core::convert::Infallible;
        fn work(&mut self, _units: u64) -> Result<(), Self::Stop> {
            Ok(())
        }
        fn scratch(&mut self, bytes: u64) -> Result<(), Self::Stop> {
            self.0.push(bytes);
            Ok(())
        }
    }

    let reg = VectorIndexRegistry::new();
    reg.register_sharded(
        hnsw(1, "emb", "Doc", "embedding"),
        Arc::new(TwoPartitionRouter),
    );
    // id 7 is on the boundary, in both partitions; id 4 only in the second.
    reg.bulk_insert(
        "Doc",
        "embedding",
        [(4u64, vec![1.0, 0.0, 0.0]), (7, vec![0.0, 1.0, 0.0])].into_iter(),
    )
    .unwrap();
    let mut recorded = Recorded::default();
    let found = reg
        .search("Doc", "embedding", &[0.0, 1.0, 0.0], 10, &mut recorded)
        .unwrap();
    assert_eq!(found.map(|r| r.len()), Some(2));
    // The boundary query reaches the first partition (one answer) and then
    // the second (two answers); each is charged before it is merged.
    let entry = core::mem::size_of::<(u64, f64)>() as u64;
    assert!(recorded.0.contains(&entry), "{:?}", recorded.0);
    assert_eq!(recorded.0.last(), Some(&(2 * entry)), "{:?}", recorded.0);
}

#[test]
fn sharded_search_with_visibility_filters_hidden_ids() {
    let reg = VectorIndexRegistry::new();
    reg.register_sharded(
        hnsw(1, "emb", "Doc", "embedding"),
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
    )
    .unwrap();

    // The query routes to the negative partition {1, 2}. Hiding id 1 must
    // prune it from the scatter-gathered, merged result.
    let filtered = search_visible(&reg, &[-1.0, 0.0, 0.0], 5, |id| id != 1).unwrap();
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
    let unfiltered = search_visible(&reg, &[-1.0, 0.0, 0.0], 5, |_| true).unwrap();
    assert_eq!(
        unfiltered[0].id, 1,
        "unfiltered sharded top-1 is the exact match"
    );
}

/// The check a write is refused with before it commits: the declared
/// dimension holds before the index has any vector, a zero vector is refused
/// under cosine, and the check changes nothing.
#[test]
fn admission_checks_the_declared_dimension_and_the_metric() {
    use coordinode_vector::hnsw::VectorDomainError;
    let reg = VectorIndexRegistry::new();
    reg.register(hnsw(1, "emb", "Doc", "embedding"));
    let node = NodeId::from_raw(9);
    let check = |v: &[f32]| {
        reg.check_admission("Doc", "embedding", node, v)
            .map_err(|e| e.error)
    };

    assert_eq!(
        check(&[1.0, 0.0]),
        Err(VectorDomainError::DimensionMismatch { left: 2, right: 3 }),
        "the declared width binds before any vector is indexed"
    );
    assert_eq!(
        check(&[0.0, 0.0, 0.0]),
        Err(VectorDomainError::ZeroVectorCosine)
    );
    assert_eq!(
        check(&[0.0, f32::NAN, 1.0]),
        Err(VectorDomainError::NonFinite { index: 1 })
    );
    assert_eq!(
        check(&[0.0, 1e-30, 0.0]),
        Ok(()),
        "a tiny nonzero vector has a direction"
    );
    assert_eq!(
        reg.get("Doc", "embedding").unwrap().read().unwrap().len(),
        0
    );
    // No index, nothing to check.
    assert!(reg.check_admission("Doc", "other", node, &[0.0]).is_ok());
}

/// A write of an unevaluable vector inserts nothing, through the single and
/// the partitioned index alike: a partitioned batch is checked whole before
/// any partition is written.
#[test]
fn a_refused_batch_changes_no_partition() {
    let single = VectorIndexRegistry::new();
    single.register(hnsw(1, "emb", "Doc", "embedding"));
    let sharded = VectorIndexRegistry::new();
    sharded.register_sharded(
        hnsw(1, "emb", "Doc", "embedding"),
        Arc::new(TwoPartitionRouter),
    );
    let batch = || {
        vec![
            (NodeId::from_raw(1), vec![-1.0, 0.0, 0.0]),
            (NodeId::from_raw(2), vec![1.0, 0.0, 0.0]),
            (NodeId::from_raw(3), vec![0.0, 0.0, 0.0]),
        ]
    };
    for reg in [&single, &sharded] {
        let refused = reg
            .on_vectors_written("Doc", "embedding", batch())
            .unwrap_err();
        assert_eq!(refused.id, 3);
        for q in [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]] {
            assert!(search(reg, "Doc", "embedding", &q, 5).unwrap().is_empty());
        }
    }
}

/// A search scores its hits exactly: the score of an exact match is exactly
/// zero cosine distance, and a vector past the f32 range of its dot product
/// still gets a finite score.
#[test]
fn search_scores_are_exact() {
    let reg = VectorIndexRegistry::new();
    reg.register(hnsw(1, "emb", "Doc", "embedding"));
    reg.bulk_insert(
        "Doc",
        "embedding",
        [(1u64, vec![1e20, 1e20, 0.0]), (2, vec![0.0, 1.0, 0.0])].into_iter(),
    )
    .unwrap();
    let hits = search(&reg, "Doc", "embedding", &[1e20, 1e20, 0.0], 2).unwrap();
    assert_eq!(hits[0].id, 1);
    assert_eq!(hits[0].score, 0.0);
    assert!(hits[1].score.is_finite());
}
