use std::collections::HashSet;

use super::*;

/// The index under test, its admission and search errors turned into test
/// failures: these tests build valid indexes and exercise graph behaviour,
/// so any refusal is a defect. Named like the index so it shadows the glob
/// import; everything else reaches the index through `Deref`. Tests of the
/// refusals themselves call [`super::HnswIndex`] directly.
struct HnswIndex(super::HnswIndex);

impl core::ops::Deref for HnswIndex {
    type Target = super::HnswIndex;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for HnswIndex {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[allow(clippy::type_complexity)]
impl HnswIndex {
    fn new(config: HnswConfig) -> Self {
        Self(super::HnswIndex::new(config))
    }

    fn insert(&mut self, id: u64, vector: Vec<f32>) {
        self.0.insert(id, vector).expect("admitted");
    }

    fn insert_shared(&self, id: u64, vector: &[f32]) -> bool {
        self.0.insert_shared(id, vector).expect("admitted")
    }

    fn insert_batch(&mut self, items: Vec<(u64, Vec<f32>)>) {
        self.0.insert_batch(items).expect("admitted");
    }

    fn bulk_build(&mut self, items: Vec<(u64, Vec<f32>)>) {
        self.0.bulk_build(items).expect("admitted");
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.0.search(query, k).expect("searchable")
    }

    fn search_with_mode(&self, query: &[f32], k: usize, mode: SearchMode) -> Vec<SearchResult> {
        self.0.search_with_mode(query, k, mode).expect("searchable")
    }

    fn search_with_visibility<F: Fn(u64) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        overfetch_factor: f64,
        max_expansion_rounds: usize,
        is_visible: F,
    ) -> (
        Vec<SearchResult>,
        coordinode_core::graph::types::VectorMvccStats,
    ) {
        self.0
            .search_with_visibility(query, k, overfetch_factor, max_expansion_rounds, is_visible)
            .expect("searchable")
    }

    fn search_with_loader(
        &self,
        query: &[f32],
        k: usize,
        loader: &dyn VectorLoader,
    ) -> Vec<SearchResult> {
        self.0
            .search_with_loader(query, k, loader)
            .expect("searchable")
    }

    /// The meter's stop, or the results; a search failure is a defect.
    #[allow(clippy::panic, reason = "a refused search fails the test")]
    fn search_metered<M: Meter>(
        &self,
        query: &[f32],
        k: usize,
        meter: &mut M,
    ) -> Result<Vec<SearchResult>, M::Stop>
    where
        M::Stop: core::fmt::Debug,
    {
        match self.0.search_metered(query, k, meter) {
            Ok(found) => Ok(found),
            Err(SearchError::Stopped(stop)) => Err(stop),
            Err(SearchError::Failed(error)) => panic!("searchable: {error}"),
        }
    }
}

fn make_config(metric: VectorMetric) -> HnswConfig {
    HnswConfig {
        m: 4,
        m_max0: 8,
        ef_construction: 16,
        ef_search: 10,
        metric,
        max_dimensions: 65_536,
        ..Default::default()
    }
}

/// A search under a query budget returns what the unmetered search returns,
/// counts the vectors it compared and holds its scratch only while it runs;
/// a cancelled budget stops it with that stop.
#[test]
fn a_metered_search_answers_alike_and_stops_with_its_budget() {
    use coordinode_core::budget::{BatchedWork, BudgetStop, QueryBudget};
    let mut index = HnswIndex::new(make_config(VectorMetric::L2));
    for i in 0..300u64 {
        index.insert(i, vec![i as f32, (i % 17) as f32, (i % 5) as f32]);
    }
    let query = [150.0, 3.0, 1.0];

    let budget = QueryBudget::new(1 << 20);
    let mut batch = BatchedWork::new(&budget);
    let metered = index
        .search_metered(&query, 5, &mut batch)
        .expect("metered");
    assert!(
        budget.memory_used() > 0,
        "the scratch is held while it runs"
    );
    batch.finish().expect("finish");
    assert_eq!(budget.memory_used(), 0);
    assert!(budget.work_done() > 5, "compared {}", budget.work_done());
    let plain = index.search(&query, 5);
    assert_eq!(
        metered.iter().map(|r| r.id).collect::<Vec<_>>(),
        plain.iter().map(|r| r.id).collect::<Vec<_>>()
    );

    let cancelled = QueryBudget::new(1 << 20);
    cancelled.cancel();
    let mut batch = BatchedWork::new(&cancelled);
    let mut stopped = Ok(Vec::new());
    for _ in 0..coordinode_core::budget::CHECK_EVERY {
        stopped = index.search_metered(&query, 5, &mut batch);
        if stopped.is_err() {
            break;
        }
    }
    assert_eq!(stopped.map(|_| ()), Err(BudgetStop::Cancelled));

    let tight = QueryBudget::new(16);
    let mut batch = BatchedWork::new(&tight);
    assert!(matches!(
        index.search_metered(&query, 5, &mut batch),
        Err(BudgetStop::Memory { .. })
    ));
}

#[test]
fn effective_alpha_resolves_auto_and_explicit() {
    let mut cfg = make_config(VectorMetric::Cosine);
    // Default (0.0) = auto -> cosine-tuned 1.15.
    assert_eq!(cfg.effective_alpha(), 1.15);
    // Explicit 1.0 forces RobustPrune off even for cosine.
    cfg.alpha_pruning = 1.0;
    assert_eq!(cfg.effective_alpha(), 1.0);
    // Explicit values win as-is.
    cfg.alpha_pruning = 1.3;
    assert_eq!(cfg.effective_alpha(), 1.3);
    // Auto on a non-cosine metric stays off (unmeasured).
    let l2 = make_config(VectorMetric::L2);
    assert_eq!(l2.effective_alpha(), 1.0);
}

fn rabitq_cosine_config() -> HnswConfig {
    let mut cfg = make_config(VectorMetric::Cosine);
    cfg.quantization = QuantizationCodec::RaBitQ { bits: 1 };
    cfg.calibration_threshold = 16;
    cfg.max_elements = 64;
    cfg
}

/// dim 64 makes the 1-bit code exactly one u64 word, the shape the search
/// fast path reads from the code block.
fn rabitq_vector(i: u64, phase: f32) -> Vec<f32> {
    (0..64)
        .map(|d| ((i as f32 * 0.37) + d as f32 * 0.11 + phase).sin())
        .collect()
}

/// Every node's slot in the code block holds exactly its current SoA code.
fn assert_rabitq_block_matches_codes(index: &HnswIndex) {
    let block = index
        .rabitq_block()
        .expect("a RaBitQ index has a code block after inserts");
    for idx in 0..index.node_len() {
        let soa = index.node_rabitq(idx);
        assert!(
            matches!(soa, Some(RabitqEncoded::OneBit(_))),
            "node {idx} has no 1-bit code after calibration"
        );
        let Some(RabitqEncoded::OneBit(code)) = soa else {
            continue;
        };
        let words: &[u64] = code.code.as_slice();
        // SAFETY: idx < nodes.len() <= max_elements == block capacity.
        let (bytes, scalars) = unsafe { (block.rabitq(idx).to_vec(), block.rabitq_scalars(idx)) };
        let expected: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(bytes, expected[..bytes.len()], "code bytes of node {idx}");
        assert_eq!(scalars.norm, code.norm, "norm of node {idx}");
        assert_eq!(
            scalars.signed_sum, code.signed_sum,
            "signed_sum of node {idx}"
        );
    }
}

#[test]
fn rabitq_block_is_lazy_and_only_for_rabitq() {
    let index = HnswIndex::new(rabitq_cosine_config());
    assert!(
        index.rabitq_block().is_none(),
        "the code block must not allocate before the first insert observes a dim"
    );
    // Other codecs never produce RaBitQ codes, so they get no block.
    let mut plain = HnswIndex::new(make_config(VectorMetric::Cosine));
    plain.insert(1, rabitq_vector(1, 0.0));
    assert!(plain.rabitq_block().is_none());
}

/// The block is filled at calibration and on every later insert, and an
/// update replaces a node's slot with its new code.
#[test]
fn rabitq_block_tracks_codes_through_calibration_insert_and_update() {
    let mut index = HnswIndex::new(rabitq_cosine_config());
    for i in 0..32u64 {
        index.insert(i, rabitq_vector(i, 0.0));
    }
    assert_rabitq_block_matches_codes(&index);
    // Re-inserting an existing id moves the node to its new vector.
    index.insert(3, rabitq_vector(3, 1.7));
    assert_rabitq_block_matches_codes(&index);
}

/// Reordering renumbers the nodes; the block is refilled in the new order.
#[test]
fn rabitq_block_follows_reorder() {
    let mut index = HnswIndex::new(rabitq_cosine_config());
    for i in 0..40u64 {
        index.insert(i, rabitq_vector(i, 0.0));
    }
    index.reorder_for_cache_locality();
    assert_rabitq_block_matches_codes(&index);
}

#[test]
fn data_level0_neighbours_form_valid_layer0_graph() {
    // The contiguous data_level0 block is the sole layer-0 neighbour store
    // (SoA neighbours_l0 removed). After a build, every node's neighbour
    // ids must be valid indices (no EMPTY sentinel leak, no torn u32, no
    // self-loops) and the graph must connect essentially every node.
    let mut cfg = make_config(VectorMetric::L2);
    cfg.max_elements = 128;
    let mut index = HnswIndex::new(cfg);
    let dim = 8;
    for i in 0..96u64 {
        let v: Vec<f32> = (0..dim)
            .map(|d| ((i * 13 + d as u64 * 7) as f32).sin())
            .collect();
        index.insert(i, v);
    }
    let block = index
        .data_level0
        .get()
        .expect("data_level0 present after inserts");
    let n = index.node_len();
    let mut with_neighbours = 0usize;
    for idx in 0..n {
        let mut blk = Vec::new();
        // SAFETY: idx < nodes.len() <= block.capacity().
        unsafe {
            block.read_neighbours_into(idx, &mut blk);
        }
        // Every stored id is a valid node index (no EMPTY sentinel leak,
        // no torn u32) and no node links to itself.
        for &nid in &blk {
            assert!(
                (nid as usize) < n,
                "node {idx}: neighbour {nid} out of range"
            );
            assert_ne!(nid as usize, idx, "node {idx} links to itself");
        }
        if !blk.is_empty() {
            with_neighbours += 1;
        }
    }
    // A built graph connects essentially every node at layer 0.
    assert!(
        with_neighbours >= n - 1,
        "only {with_neighbours}/{n} nodes have layer-0 neighbours"
    );
}

#[test]
fn rabitq_block_tracks_codes_on_batch_insert() {
    let mut index = HnswIndex::new(rabitq_cosine_config());
    // Above BATCH_PARALLEL_THRESHOLD so the rayon-planned path fires.
    let items: Vec<(u64, Vec<f32>)> = (0..48u64).map(|i| (i, rabitq_vector(i, 0.0))).collect();
    index.insert_batch(items);
    assert_rabitq_block_matches_codes(&index);
}

#[test]
fn rerank_mode_end_of_search_returns_results_on_cosine_rabitq() {
    // Smoke test that EndOfSearch dispatch produces the right shape
    // — populated result set, top hit on a known-good query is
    // itself. Recall correctness is exercised more thoroughly in
    // the heavy rabitq_recall_* tests; this one just guards against
    // the new dispatch arm returning empty / panicking.
    let mut cfg = make_config(VectorMetric::Cosine);
    cfg.quantization = QuantizationCodec::RaBitQ { bits: 1 };
    cfg.rerank_mode = RerankMode::EndOfSearch;
    cfg.calibration_threshold = 16;
    let mut index = HnswIndex::new(cfg);
    for i in 0..32u64 {
        let v: Vec<f32> = (0..16)
            .map(|d| ((i as f32 * 0.3) + d as f32 * 0.1).sin())
            .collect();
        index.insert(i, v);
    }
    for i in 0..32u64 {
        let v: Vec<f32> = (0..16)
            .map(|d| ((i as f32 * 0.3) + d as f32 * 0.1).sin())
            .collect();
        let results = index.search(&v, 1);
        assert_eq!(
            results.first().map(|r| r.id),
            Some(i),
            "EndOfSearch rerank must still find a node's own self at rank 1 (i={i})",
        );
    }
}

#[test]
fn rerank_mode_end_of_search_oversample_returns_results() {
    // Smoke test: oversample factor > 1.0 inflates the frontier but
    // the final result set still respects the user's k bound and
    // top hit on a known-good query is itself.
    let mut cfg = make_config(VectorMetric::Cosine);
    cfg.quantization = QuantizationCodec::RaBitQ { bits: 1 };
    cfg.rerank_mode = RerankMode::EndOfSearch;
    cfg.rerank_oversample_factor = 2.5;
    cfg.calibration_threshold = 16;
    let mut index = HnswIndex::new(cfg);
    for i in 0..32u64 {
        let v: Vec<f32> = (0..16)
            .map(|d| ((i as f32 * 0.3) + d as f32 * 0.1).sin())
            .collect();
        index.insert(i, v);
    }
    for i in 0..32u64 {
        let v: Vec<f32> = (0..16)
            .map(|d| ((i as f32 * 0.3) + d as f32 * 0.1).sin())
            .collect();
        let results = index.search(&v, 1);
        assert_eq!(
            results.first().map(|r| r.id),
            Some(i),
            "EndOfSearch + oversample must still find a node's own self at rank 1 (i={i})",
        );
    }
    // k bound preserved.
    let q: Vec<f32> = (0..16).map(|d| (d as f32 * 0.1).sin()).collect();
    let results = index.search(&q, 4);
    assert!(
        results.len() <= 4,
        "search must respect the k bound even with oversample",
    );
}

#[test]
fn rerank_mode_none_returns_results_on_cosine_rabitq() {
    // Smoke test that the None (no rerank) path produces a populated
    // result set on a tiny RaBitQ index. Recall will be lower than
    // Inline / EndOfSearch but the search must not panic or return
    // empty.
    let mut cfg = make_config(VectorMetric::Cosine);
    cfg.quantization = QuantizationCodec::RaBitQ { bits: 1 };
    cfg.rerank_mode = RerankMode::None;
    cfg.calibration_threshold = 16;
    let mut index = HnswIndex::new(cfg);
    for i in 0..32u64 {
        let v: Vec<f32> = (0..16)
            .map(|d| ((i as f32 * 0.3) + d as f32 * 0.1).sin())
            .collect();
        index.insert(i, v);
    }
    let q: Vec<f32> = (0..16).map(|d| (d as f32 * 0.1).sin()).collect();
    let results = index.search(&q, 3);
    assert!(
        !results.is_empty(),
        "None rerank mode must still return search results",
    );
    assert!(results.len() <= 3, "search must respect the k bound");
}

#[test]
fn robust_prune_default_alpha_one_is_noop() {
    // α=1.0 must degenerate to legacy "take M closest" — neighbour
    // selection should be identical to the simple strategy. We
    // smoke-test by running an insert + immediate search and
    // verifying recall on a small known-good dataset.
    let mut cfg = make_config(VectorMetric::L2);
    cfg.alpha_pruning = 1.0;
    let mut index = HnswIndex::new(cfg);
    for i in 0..50u64 {
        let v = vec![i as f32, (i * 2) as f32];
        index.insert(i, v);
    }
    // Nearest neighbour of (0, 0) in this set is id=0 at distance 0.
    let results = index.search(&[0.0, 0.0], 5);
    assert!(!results.is_empty(), "α=1.0 must return non-empty result");
    assert_eq!(
        results[0].id, 0,
        "α=1.0 default must find the exact nearest neighbour",
    );
}

#[test]
fn robust_prune_alpha_greater_one_still_recalls_query() {
    // α > 1.0 produces a sparser graph; check that the recall
    // contract still holds on a small dataset where every query is
    // its own nearest neighbour. This is the smoke test that the
    // pruning loop doesn't accidentally strand the inserted node
    // (e.g. by selecting zero neighbours).
    let mut cfg = make_config(VectorMetric::L2);
    cfg.alpha_pruning = 1.2;
    let mut index = HnswIndex::new(cfg);
    let n = 50u64;
    for i in 0..n {
        let v = vec![i as f32, (i * 2) as f32];
        index.insert(i, v);
    }
    // Each inserted point queried against itself must rank itself
    // #1 — this fails if RobustPrune over-prunes and disconnects
    // the inserted node from the graph.
    for i in 0..n {
        let v = vec![i as f32, (i * 2) as f32];
        let results = index.search(&v, 1);
        assert_eq!(
            results.first().map(|r| r.id),
            Some(i),
            "α=1.2 must still find a node's own self at rank 1 (i={i})",
        );
    }
}

#[test]
fn empty_index() {
    let index = HnswIndex::new(HnswConfig::default());
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert!(index.search(&[1.0, 0.0, 0.0], 5).is_empty());
}

#[test]
fn insert_single() {
    let mut index = HnswIndex::new(make_config(VectorMetric::L2));
    index.insert(1, vec![1.0, 0.0, 0.0]);
    assert_eq!(index.len(), 1);

    let results = index.search(&[1.0, 0.0, 0.0], 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn insert_duplicate_ignored() {
    let mut index = HnswIndex::new(make_config(VectorMetric::L2));
    index.insert(1, vec![1.0, 0.0]);
    index.insert(1, vec![2.0, 0.0]); // Same ID — ignored
    assert_eq!(index.len(), 1);
}

#[test]
fn search_nearest_l2() {
    let mut index = HnswIndex::new(make_config(VectorMetric::L2));

    // Insert points at known positions
    index.insert(1, vec![0.0, 0.0]);
    index.insert(2, vec![1.0, 0.0]);
    index.insert(3, vec![0.0, 1.0]);
    index.insert(4, vec![10.0, 10.0]);

    // Query near origin — should find IDs 1, 2, 3 before 4
    let results = index.search(&[0.1, 0.1], 3);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, 1); // Nearest to (0.1, 0.1) is (0, 0)
    // IDs 2 and 3 should be next (equidistant from query)
    assert!(results.iter().any(|r| r.id == 2));
    assert!(results.iter().any(|r| r.id == 3));
}

#[test]
fn search_nearest_cosine() {
    let mut index = HnswIndex::new(make_config(VectorMetric::Cosine));

    // Normalized-ish vectors in different directions
    index.insert(1, vec![1.0, 0.0]);
    index.insert(2, vec![0.0, 1.0]);
    index.insert(3, vec![0.707, 0.707]);
    index.insert(4, vec![-1.0, 0.0]);

    // Query in direction (1, 0) — should find ID 1 first
    let results = index.search(&[1.0, 0.0], 2);
    assert_eq!(results[0].id, 1);
}

#[test]
fn search_k_greater_than_size() {
    let mut index = HnswIndex::new(make_config(VectorMetric::L2));
    index.insert(1, vec![0.0, 0.0]);
    index.insert(2, vec![1.0, 1.0]);

    let results = index.search(&[0.0, 0.0], 10);
    assert_eq!(results.len(), 2); // Only 2 elements exist
}

#[test]
fn larger_dataset() {
    let mut index = HnswIndex::new(make_config(VectorMetric::L2));

    // Insert 100 points in a grid
    for i in 0..10 {
        for j in 0..10 {
            let id = (i * 10 + j) as u64;
            index.insert(id, vec![i as f32, j as f32]);
        }
    }

    assert_eq!(index.len(), 100);

    // Query at (0.5, 0.5) — nearest should be (0,0), (1,0), (0,1), (1,1)
    let results = index.search(&[0.5, 0.5], 4);
    assert_eq!(results.len(), 4);

    let ids: HashSet<u64> = results.iter().map(|r| r.id).collect();
    assert!(ids.contains(&0)); // (0,0)
    assert!(ids.contains(&1)); // (0,1)
    assert!(ids.contains(&10)); // (1,0)
    assert!(ids.contains(&11)); // (1,1)
}

#[test]
fn high_dimensional() {
    let mut index = HnswIndex::new(make_config(VectorMetric::Cosine));

    // 128-dimensional vectors, none all zeros (cosine has no direction for
    // one).
    for i in 0..20u64 {
        let vec: Vec<f32> = (0..128).map(|d| ((i * d + 1) as f32).sin()).collect();
        index.insert(i, vec);
    }

    assert_eq!(index.len(), 20);

    let query: Vec<f32> = (0..128).map(|d| (d as f32 * 0.1).sin()).collect();
    let results = index.search(&query, 5);
    assert_eq!(results.len(), 5);
    // All results should have valid IDs
    for r in &results {
        assert!(r.id < 20);
    }
}

/// Mid-scale recall regression test.
///
/// The 200-vector unit test is too small to expose connectivity bugs
/// in the lock-free insert path. At ~10K vectors the graph topology
/// becomes large enough that lost back-edges become visible as recall
/// drift. Override scale with `RECALL_N=<n>`.
///
/// `#[ignore]` so default `cargo nextest run` stays fast; invoke via
/// `cargo nextest run hnsw::tests::recall_mid_scale_l2_10k --run-ignored only`
/// before any PR that touches HNSW insert or search.
#[test]
#[ignore = "mid-scale recall regression — run manually before HNSW PRs"]
fn recall_mid_scale_l2_10k() {
    // Use the same M parameters as the SIFT1M bench so we exercise the
    // `max_conn == M_MAX0` saturation path on layer 0. With smaller `m`
    // the saturation branch is unreachable and the bug doesn't surface.
    let mut index = HnswIndex::new(HnswConfig {
        m: 32,
        m_max0: 64,
        ef_construction: 200,
        ef_search: 64,
        metric: VectorMetric::L2,
        max_dimensions: 65_536,
        ..Default::default()
    });

    let n = std::env::var("RECALL_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10_000);
    let dim = 64usize;
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| {
                    let seed = (i.wrapping_mul(2_654_435_761).wrapping_add(d * 6_700_417)) as u32;
                    let bits = (seed ^ (seed >> 13)) & 0x00FF_FFFF;
                    (bits as f32 / 16_777_216.0) - 0.5
                })
                .collect()
        })
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }

    let k = 10usize;
    let queries: Vec<&[f32]> = (0..50).map(|i| vectors[i * 199 % n].as_slice()).collect();

    let mut total_recall = 0.0f32;
    for &q in &queries {
        let mut gt: Vec<(f32, u64)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (metrics::euclidean_distance_squared(q, v), i as u64))
            .collect();
        gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let gt_set: HashSet<u64> = gt.iter().take(k).map(|&(_, id)| id).collect();

        let results = index.search(q, k);
        let res_set: HashSet<u64> = results.iter().map(|r| r.id).collect();
        total_recall += gt_set.intersection(&res_set).count() as f32 / k as f32;
    }
    let avg_recall = total_recall / queries.len() as f32;
    eprintln!("mid-scale recall@{k} on n={n} dim={dim}: {:.3}", avg_recall);
    assert!(
        avg_recall >= 0.90,
        "mid-scale recall {avg_recall:.3} below 0.90 — graph connectivity regression at scale"
    );
}

#[test]
fn recall_test_l2() {
    // Build an index and verify approximate recall
    let mut index = HnswIndex::new(HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 50,
        ef_search: 20,
        metric: VectorMetric::L2,
        max_dimensions: 65_536,
        ..Default::default()
    });

    let n = 200;
    let dim = 16;
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| ((i * d + 7) as f32 * 0.13).sin())
                .collect()
        })
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }

    let query = &vectors[0]; // Use first vector as query
    let k = 10;
    let results = index.search(query, k);
    assert_eq!(results.len(), k);

    // Brute-force ground truth
    let mut ground_truth: Vec<(f32, u64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist = metrics::euclidean_distance_squared(query, v);
            (dist, i as u64)
        })
        .collect();
    ground_truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let gt_set: HashSet<u64> = ground_truth.iter().take(k).map(|&(_, id)| id).collect();
    let result_set: HashSet<u64> = results.iter().map(|r| r.id).collect();

    let recall = gt_set.intersection(&result_set).count() as f32 / k as f32;
    let correct = gt_set.intersection(&result_set).count();
    eprintln!(
        "HNSW recall@{k}: {:.0}% ({correct} of {k} correct)",
        recall * 100.0
    );
    assert!(
        recall >= 0.5,
        "recall {recall} too low (expected >= 50% with small ef)"
    );
}

/// Verify that random_level produces proper exponential distribution.
/// With M=16, level_mult = 1/ln(16) ≈ 0.36:
///   - ~64% of nodes at level 0 only
///   - ~23% at level 1
///   - ~8% at level 2
///   - <5% at level 3+
///
/// Old deterministic hash gave identical levels for nodes inserted at same index count.
#[test]
fn random_level_distribution() {
    let mut index = HnswIndex::new(HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 50,
        ef_search: 10,
        metric: VectorMetric::L2,
        ..Default::default()
    });

    let n = 1000;
    let dim = 4;
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|d| (i * 10 + d) as f32).collect();
        index.insert(i as u64, v);
    }

    // Count nodes per max layer
    let mut layer_counts = [0usize; 10];
    for idx in 0..index.node_len() {
        let max_layer = index.node_levels(idx).saturating_sub(1);
        if max_layer < layer_counts.len() {
            layer_counts[max_layer] += 1;
        }
    }

    // Layer 0 should have majority (>40% for M=16)
    let layer0_frac = layer_counts[0] as f64 / n as f64;
    assert!(
        layer0_frac > 0.4,
        "layer 0 fraction {layer0_frac:.2} too low (expected >40%)"
    );

    // Should have SOME nodes at layer 1+ (proper RNG, not all same level)
    let higher_layers: usize = layer_counts[1..].iter().sum();
    assert!(
        higher_layers > 50,
        "only {higher_layers} nodes at layer 1+ out of {n} — RNG may be broken"
    );

    // Should NOT have all nodes at same layer (old deterministic bug)
    let unique_layers = layer_counts.iter().filter(|&&c| c > 0).count();
    assert!(
        unique_layers >= 3,
        "only {unique_layers} distinct layers used — expected ≥3 for proper exponential distribution"
    );
}

/// Integration test: multiple sequential searches correctly reuse visited pool.
/// Verifies that epoch-based reset doesn't leak state between searches
/// (regression test for the HashSet→epoch visited-list refactor).
#[test]
fn sequential_searches_reuse_visited_pool() {
    let mut index = HnswIndex::new(HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 100,
        ef_search: 50,
        metric: VectorMetric::L2,
        ..Default::default()
    });

    let dim = 64;
    let n = 500;
    // Generate linearly independent vectors using hash-like spreading.
    // Simple LCG per (i, d) avoids periodic cos() collisions.
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| {
                    let seed = (i as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add((d as u64).wrapping_mul(1442695040888963407));
                    (seed >> 33) as f32 / (1u64 << 31) as f32
                })
                .collect()
        })
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }

    // Run 300 sequential searches — forces visited pool reuse + epoch advancement.
    // With u8 epoch, epoch wraps at 255 → fill(0) happens once in 300 searches.
    let k = 5;
    for query_idx in 0..300 {
        let query = &vectors[query_idx % n];
        let results = index.search(query, k);
        assert_eq!(
            results.len(),
            k,
            "search #{query_idx} returned {} results, expected {k}",
            results.len()
        );

        // Query vector itself must be in the top-k results (distance = 0).
        // We check contains (not first position) because HNSW is approximate
        // and with small M=8 some vectors may have higher-scoring near-duplicates.
        let result_ids: HashSet<u64> = results.iter().map(|r| r.id).collect();
        assert!(
            result_ids.contains(&((query_idx % n) as u64)),
            "search #{query_idx}: query id={} not found in top-{k} results: {:?}",
            query_idx % n,
            result_ids
        );
    }
}

#[test]
fn dot_product_metric() {
    let mut index = HnswIndex::new(make_config(VectorMetric::DotProduct));

    index.insert(1, vec![1.0, 0.0]);
    index.insert(2, vec![0.0, 1.0]);
    index.insert(3, vec![0.5, 0.5]);

    // Dot product: higher = more similar → search returns highest dot product
    let results = index.search(&[1.0, 0.0], 1);
    assert_eq!(results[0].id, 1); // Highest dot with (1,0) is (1,0)
}

#[test]
fn manhattan_metric() {
    let mut index = HnswIndex::new(make_config(VectorMetric::L1));

    index.insert(1, vec![0.0, 0.0]);
    index.insert(2, vec![1.0, 1.0]);
    index.insert(3, vec![5.0, 5.0]);

    let results = index.search(&[0.1, 0.1], 1);
    assert_eq!(results[0].id, 1); // Nearest in L1
}

// --- SQ8 Quantization Tests ---

fn make_sq8_config(metric: VectorMetric) -> HnswConfig {
    HnswConfig {
        m: 4,
        m_max0: 8,
        ef_construction: 16,
        ef_search: 10,
        metric,
        max_dimensions: 65_536,
        quantization: QuantizationCodec::Sq8,
        rerank_candidates: 20,
        calibration_threshold: 5, // Low threshold for testing
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: 1_000_000,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    }
}

#[test]
fn sq8_not_calibrated_before_threshold() {
    let mut index = HnswIndex::new(make_sq8_config(VectorMetric::L2));

    // Insert fewer vectors than calibration_threshold
    for i in 0..4u64 {
        index.insert(i, vec![i as f32, 0.0, 0.0]);
    }

    assert!(!index.is_quantized());
    assert!(index.sq8_params().is_none());

    // Search still works (uses f32)
    let results = index.search(&[0.0, 0.0, 0.0], 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
}

#[test]
fn sq8_auto_calibrates_at_threshold() {
    let mut index = HnswIndex::new(make_sq8_config(VectorMetric::L2));

    // Insert exactly calibration_threshold vectors
    for i in 0..5u64 {
        index.insert(i, vec![i as f32, (i as f32).sin(), 0.0]);
    }

    assert!(index.is_quantized());
    assert!(index.sq8_params().is_some());

    // All nodes should now have quantized vectors
    for idx in 0..index.node_len() {
        assert!(index.node_sq8(idx).is_some());
    }
}

#[test]
fn sq8_new_inserts_after_calibration_are_quantized() {
    let mut index = HnswIndex::new(make_sq8_config(VectorMetric::L2));

    // Trigger calibration
    for i in 0..5u64 {
        index.insert(i, vec![i as f32, 0.0]);
    }
    assert!(index.is_quantized());

    // Insert after calibration
    index.insert(100, vec![2.5, 0.5]);
    let idx = index.id_to_idx.get(100).expect("inserted");
    assert!(index.node_sq8(idx).is_some());
}

#[test]
fn sq8_search_returns_correct_nearest() {
    let mut index = HnswIndex::new(make_sq8_config(VectorMetric::L2));

    // Insert enough vectors to trigger calibration
    index.insert(1, vec![0.0, 0.0]);
    index.insert(2, vec![1.0, 0.0]);
    index.insert(3, vec![0.0, 1.0]);
    index.insert(4, vec![10.0, 10.0]);
    index.insert(5, vec![5.0, 5.0]); // Triggers calibration

    assert!(index.is_quantized());

    // Query near origin
    let results = index.search(&[0.1, 0.1], 3);
    assert_eq!(results.len(), 3);
    // Nearest should be (0,0), and reranking with f32 should give exact order
    assert_eq!(results[0].id, 1);
}

#[test]
fn sq8_reranking_improves_accuracy() {
    // With SQ8, the reranking step should produce scores computed
    // from exact f32 vectors, not approximate dequantized ones.
    let mut index = HnswIndex::new(HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 50,
        ef_search: 20,
        metric: VectorMetric::L2,
        max_dimensions: 65_536,
        quantization: QuantizationCodec::Sq8,
        rerank_candidates: 50,
        calibration_threshold: 10,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: 1_000_000,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    });

    let dim = 16;
    let n = 50;
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| ((i * d + 7) as f32 * 0.13).sin())
                .collect()
        })
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }

    assert!(index.is_quantized());

    let query = &vectors[0];
    let results = index.search(query, 5);
    assert_eq!(results.len(), 5);

    // First result should be the query vector itself, at distance 0.
    assert_eq!(results[0].id, 0);
    assert_eq!(results[0].score, 0.0);

    // Scores are the exact evaluation of the retained f32 vectors, not of
    // their SQ8 codes: bit for bit the canonical L2.
    let expected = metrics::exact::l2(query, &vectors[results[1].id as usize]).unwrap();
    assert_eq!(results[1].score, expected);
}

#[test]
fn sq8_recall_vs_non_quantized() {
    // Compare recall between quantized and non-quantized search
    let dim = 16;
    let n = 200;
    let k = 10;

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| ((i * d + 7) as f32 * 0.13).sin())
                .collect()
        })
        .collect();

    // Non-quantized index
    let mut plain_index = HnswIndex::new(HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 50,
        ef_search: 30,
        metric: VectorMetric::L2,
        max_dimensions: 65_536,
        quantization: QuantizationCodec::None,
        ..Default::default()
    });

    // Quantized index
    let mut sq8_index = HnswIndex::new(HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 50,
        ef_search: 30,
        metric: VectorMetric::L2,
        max_dimensions: 65_536,
        quantization: QuantizationCodec::Sq8,
        rerank_candidates: 50,
        calibration_threshold: 50,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: 1_000_000,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    });

    for (i, v) in vectors.iter().enumerate() {
        plain_index.insert(i as u64, v.clone());
        sq8_index.insert(i as u64, v.clone());
    }

    assert!(sq8_index.is_quantized());
    assert!(!plain_index.is_quantized());

    // Brute-force ground truth
    let query = &vectors[0];
    let mut ground_truth: Vec<(f32, u64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist = metrics::euclidean_distance_squared(query, v);
            (dist, i as u64)
        })
        .collect();
    ground_truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let gt_set: HashSet<u64> = ground_truth.iter().take(k).map(|&(_, id)| id).collect();

    let plain_results = plain_index.search(query, k);
    let plain_set: HashSet<u64> = plain_results.iter().map(|r| r.id).collect();
    let plain_recall = gt_set.intersection(&plain_set).count() as f32 / k as f32;

    let sq8_results = sq8_index.search(query, k);
    let sq8_set: HashSet<u64> = sq8_results.iter().map(|r| r.id).collect();
    let sq8_recall = gt_set.intersection(&sq8_set).count() as f32 / k as f32;

    eprintln!(
        "SQ8 recall@{k}: plain={:.0}%, sq8={:.0}%",
        plain_recall * 100.0,
        sq8_recall * 100.0,
    );

    // SQ8 recall should be close to plain recall (within 20%)
    // Architecture spec says <2% recall loss for SQ8
    assert!(
        sq8_recall >= 0.4,
        "SQ8 recall {sq8_recall} too low (expected >= 40%)"
    );
}

#[test]
fn rabitq_recall_sanity_cosine() {
    // Sanity check: a RaBitQ-quantized HNSW index returns SOME of the
    // ground-truth top-K under cosine metric. This is not a tuned-recall
    // test (that lives in ann-benchmarks); it proves the popcount kernel
    // is wired into the hot path correctly. The bar is intentionally
    // low because at d=128 and 200 vectors RaBitQ's quantization error
    // dominates — at production scale (d≥768, N≥100k) recall climbs into
    // the 0.85-0.95 range per the SIGMOD 2024 paper.
    let dim = 64usize;
    let n = 80usize;
    let k = 10usize;
    let threshold = 30usize;

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let raw: Vec<f32> = (0..dim)
                .map(|d| ((i * d + 11) as f32 * 0.07).cos())
                .collect();
            let norm = metrics::norm_l2(&raw).max(f32::EPSILON);
            raw.into_iter().map(|x| x / norm).collect()
        })
        .collect();

    let mut index = HnswIndex::new(HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ef_search: 64,
        metric: VectorMetric::Cosine,
        max_dimensions: dim as u32,
        quantization: QuantizationCodec::RaBitQ { bits: 1 },
        rerank_candidates: 64,
        calibration_threshold: threshold,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: n as u32,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    });

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }

    assert!(
        index.is_rabitq_active(),
        "RaBitQ should be active after threshold reached"
    );
    assert!(index.rabitq_params().is_some());
    // Every post-calibration node carries an encoded code.
    let coded = (0..index.node_len())
        .filter(|&i| index.node_rabitq(i).is_some())
        .count();
    assert_eq!(coded, n, "all {n} nodes should have RaBitQ codes");

    // Brute-force ground truth by cosine distance.
    let query = &vectors[0];
    let mut ground_truth: Vec<(f32, u64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let cos = metrics::cosine_similarity_with_query_norm(query, v, metrics::norm_l2(query));
            (1.0 - cos, i as u64)
        })
        .collect();
    ground_truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let gt_set: HashSet<u64> = ground_truth.iter().take(k).map(|&(_, id)| id).collect();

    let results = index.search(query, k);
    let result_set: HashSet<u64> = results.iter().map(|r| r.id).collect();
    let recall = gt_set.intersection(&result_set).count() as f32 / k as f32;
    eprintln!(
        "RaBitQ recall@{k} at d={dim}, n={n}: {:.0}%",
        recall * 100.0
    );

    // At tiny scale RaBitQ noise can erase several true neighbours; require
    // only that the index returns something correct (>= 1/k hits) so the
    // hot path is exercised. Production-grade recall claims live in the
    // ann-benchmarks dimension-ladder runs, not in this unit test.
    assert!(
        recall >= 0.1,
        "RaBitQ recall {recall} below the wired-up sanity floor (0.1)",
    );
}

#[test]
fn rabitq_recall_cosine_dim_100_with_padding() {
    // Reproducer for the glove-100-angular bench (8fa0f2f, 3381b6d):
    // recall plateau at 0.17 across the full ef sweep with dim=100,
    // Cosine metric, RaBitQ. Same test as `rabitq_recall_sanity_cosine`
    // but with dim that ISN'T a multiple of 64 so the codec padding
    // path (effective_dims = 128) is exercised end-to-end through
    // HNSW build + search.
    //
    // If recall here drops to the ~0.1-0.2 floor, the bug reproduces
    // in a 2000-vector / dim=100 test we can debug locally without
    // waiting on the 4-minute glove bench. If recall is ≥ 0.4, the
    // bug is scale-dependent (only triggers at 1M+ vectors) and the
    // hunt moves to bench-vector-ann or HnswIndex calibration timing.
    let dim = 100usize;
    let n = 2000usize;
    let k = 10usize;
    let threshold = 100usize;

    // Synthetic unit-norm Gaussian-ish vectors via deterministic
    // sinusoid mix. Two close pairs are planted so brute-force has
    // a well-defined top-K (otherwise small-N cosine collapses to
    // near-ties and recall becomes meaningless).
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let raw: Vec<f32> = (0..dim)
                .map(|d| {
                    let phase = (i as f32) * 0.013 + (d as f32) * 0.07;
                    phase.sin() + 0.3 * (phase * 2.5).cos()
                })
                .collect();
            let norm = metrics::norm_l2(&raw).max(f32::EPSILON);
            raw.into_iter().map(|x| x / norm).collect()
        })
        .collect();

    let mut index = HnswIndex::new(HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ef_search: 200,
        metric: VectorMetric::Cosine,
        max_dimensions: dim as u32,
        quantization: QuantizationCodec::RaBitQ { bits: 1 },
        rerank_candidates: 64,
        calibration_threshold: threshold,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: n as u32,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    });

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }

    assert!(
        index.is_rabitq_active(),
        "RaBitQ should be active after threshold reached"
    );
    // Every post-calibration node must carry a code. If this fails,
    // calibration is the bug — codes are missing for some nodes and
    // search falls back to f32 for them, mixing two distance scales.
    let coded = (0..index.node_len())
        .filter(|&i| index.node_rabitq(i).is_some())
        .count();
    assert_eq!(
        coded, n,
        "all {n} nodes should have RaBitQ codes after calibration"
    );
    let params = index.rabitq_params().expect("rabitq calibrated");
    assert_eq!(params.dims(), dim as u32);
    assert_eq!(params.effective_dims(), 128, "padding path active");

    // Brute-force top-K by exact cosine. Average recall over 20
    // queries — single-query recall noise on a 2k corpus is high.
    let n_queries = 20usize;
    let mut total_hits = 0usize;
    for qi in 0..n_queries {
        let query = &vectors[qi * (n / n_queries)];
        let mut gt: Vec<(f32, u64)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let cos =
                    metrics::cosine_similarity_with_query_norm(query, v, metrics::norm_l2(query));
                (1.0 - cos, i as u64)
            })
            .collect();
        gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let gt_set: HashSet<u64> = gt.iter().take(k).map(|&(_, id)| id).collect();

        let results = index.search(query, k);
        let result_set: HashSet<u64> = results.iter().map(|r| r.id).collect();
        total_hits += gt_set.intersection(&result_set).count();
    }
    let recall = total_hits as f32 / (n_queries * k) as f32;
    eprintln!(
        "RaBitQ cosine recall@{k} at d={dim} (padded→128), n={n}: {:.0}%",
        recall * 100.0
    );

    // Bar: 0.3. With n=2000 / 20 queries / k=10 = 200 trials, binomial
    // std is ~3.4%, so anything in the 0.3-0.5 range overlaps within
    // 2σ. The bar's only job is "well above the 0.17 plateau, well
    // below f32-exact". Real production-scale recall expectations
    // live in the glove-100-angular bench (N=1.18M), not here.
    assert!(
        recall >= 0.3,
        "RaBitQ cosine+padding recall {recall:.3} below 0.3 — \
             below the wired-up floor, bug reproduces at small scale",
    );
}

#[test]
fn extended_rabitq_recall_sanity_cosine_2bit() {
    // Wires-up test for 2-bit Extended-RaBitQ: same shape as the
    // 1-bit recall sanity check, but `quantization = RaBitQ { bits: 2 }`.
    // Exercises the RabitqEncoded::Multi search path end-to-end:
    // calibration → encode_ext → estimate_cosine_distance_ext → top-K.
    let dim = 64usize;
    let n = 80usize;
    let k = 10usize;
    let threshold = 30usize;

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let raw: Vec<f32> = (0..dim)
                .map(|d| ((i * d + 11) as f32 * 0.07).cos())
                .collect();
            let norm = metrics::norm_l2(&raw).max(f32::EPSILON);
            raw.into_iter().map(|x| x / norm).collect()
        })
        .collect();

    let mut index = HnswIndex::new(HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ef_search: 64,
        metric: VectorMetric::Cosine,
        max_dimensions: dim as u32,
        quantization: QuantizationCodec::RaBitQ { bits: 2 },
        rerank_candidates: 64,
        calibration_threshold: threshold,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: n as u32,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    });

    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }

    assert!(
        index.is_rabitq_active(),
        "Extended-RaBitQ should be active after threshold reached"
    );
    // Every post-calibration node must carry a Multi(_) code with bits=2.
    // Encode "variant + bits" as a single u8 (0 = OneBit, 2/3/4 = Multi)
    // so the assertion lives in `assert_eq!` and avoids the `panic!` lint.
    for i in 0..index.node_len() {
        let code = index.node_rabitq(i).expect("rabitq code populated");
        let bits = match code {
            RabitqEncoded::Multi(c) => c.bits,
            RabitqEncoded::OneBit(_) => 0,
        };
        assert_eq!(
            bits, 2,
            "node {i}: expected Multi(bits=2), got {bits} (0=OneBit, n=Multi)"
        );
    }

    let query = &vectors[0];
    let mut ground_truth: Vec<(f32, u64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let cos = metrics::cosine_similarity_with_query_norm(query, v, metrics::norm_l2(query));
            (1.0 - cos, i as u64)
        })
        .collect();
    ground_truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let gt_set: HashSet<u64> = ground_truth.iter().take(k).map(|&(_, id)| id).collect();

    let results = index.search(query, k);
    let result_set: HashSet<u64> = results.iter().map(|r| r.id).collect();
    let recall = gt_set.intersection(&result_set).count() as f32 / k as f32;
    eprintln!(
        "Extended-RaBitQ 2-bit recall@{k} at d={dim}, n={n}: {:.0}%",
        recall * 100.0
    );

    // At d=64 / n=80 the synthetic noise floor dominates regardless of
    // bit width — this test proves wiring, not recall. The paper's
    // 0.95-0.97 claim for 2-bit is at SIFT1M scale (d≥128, n≥10⁶).
    // Same floor as the 1-bit companion test so a kernel regression
    // in either path trips at least one assertion.
    assert!(
        recall >= 0.1,
        "Extended-RaBitQ 2-bit recall {recall} below wired-up sanity floor (0.1)",
    );
}

#[test]
fn vector_tier_persists_f32_on_insert() {
    use crate::storage::{VectorTierHandle, VectorTierStorage};
    use std::collections::HashMap;
    use std::sync::Mutex;

    // Inline mock tier (same surface as the storage-crate impl).
    struct Mock {
        f32_store: Mutex<HashMap<(u32, u32, u64), Vec<f32>>>,
    }
    impl VectorTierStorage for Mock {
        fn put_f32(
            &self,
            l: u32,
            p: u32,
            n: u64,
            v: &[f32],
        ) -> Result<(), crate::storage::VectorTierError> {
            self.f32_store.lock().unwrap().insert((l, p, n), v.to_vec());
            Ok(())
        }
        fn multi_get_f32(
            &self,
            l: u32,
            p: u32,
            ids: &[u64],
        ) -> Result<Vec<Option<Vec<f32>>>, crate::storage::VectorTierError> {
            let g = self.f32_store.lock().unwrap();
            Ok(ids.iter().map(|&n| g.get(&(l, p, n)).cloned()).collect())
        }
    }

    let mock = std::sync::Arc::new(Mock {
        f32_store: Mutex::new(HashMap::new()),
    });

    let mut index = HnswIndex::new(HnswConfig {
        m: 4,
        m_max0: 8,
        ef_construction: 16,
        ef_search: 16,
        metric: VectorMetric::L2,
        max_dimensions: 8,
        quantization: QuantizationCodec::Sq8,
        calibration_threshold: 4,
        offload_vectors: false,
        ..Default::default()
    });
    index.set_vector_tier(Some(VectorTierHandle::new(mock.clone(), 7, 13)));

    let vectors: Vec<Vec<f32>> = (0..4)
        .map(|i| (0..8).map(|d| ((i * 7 + d) as f32 * 0.1).sin()).collect())
        .collect();
    for (id, v) in vectors.iter().enumerate() {
        index.insert(id as u64, v.clone());
    }

    // Truth tier: every insert persisted byte-exact (f32 only;
    // SQ8 / RaBitQ codes stay in RAM).
    let got_f32 = mock.multi_get_f32(7, 13, &[0, 1, 2, 3]).unwrap();
    for (i, slot) in got_f32.iter().enumerate() {
        assert_eq!(
            slot.as_deref(),
            Some(vectors[i].as_slice()),
            "f32 mismatch at {i}"
        );
    }

    // Distinct (label, property) handles don't alias — a different
    // handle into the same backend sees nothing.
    let empty = mock.multi_get_f32(99, 99, &[0]).unwrap();
    assert!(empty[0].is_none());
}

#[test]
fn set_rabitq_params_re_encodes_existing_nodes() {
    // Simulates the segment-reload path: an index opens with no RaBitQ
    // params (no calibration has run yet because the caller will inject
    // the persisted params explicitly), then `set_rabitq_params` is
    // called with the durable rotation matrix. Every existing node must
    // come out with an encoded code matching what the saved rotation
    // would have produced.
    let dim = 64usize;
    let n = 12usize;

    let mut index = HnswIndex::new(HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 32,
        ef_search: 16,
        metric: VectorMetric::Cosine,
        max_dimensions: dim as u32,
        quantization: QuantizationCodec::RaBitQ { bits: 1 },
        // Threshold above n so auto-calibration does NOT fire during inserts.
        calibration_threshold: 10_000,
        offload_vectors: false,
        ..Default::default()
    });

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let raw: Vec<f32> = (0..dim)
                .map(|d| ((i * d + 3) as f32 * 0.07).sin())
                .collect();
            let norm = metrics::norm_l2(&raw).max(f32::EPSILON);
            raw.into_iter().map(|x| x / norm).collect()
        })
        .collect();
    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }
    assert!(
        index.rabitq_params().is_none(),
        "no auto-calibration expected"
    );

    // Inject persisted params (here freshly built; in production these
    // would be loaded from disk alongside the segment).
    let persisted = RaBitQParams::calibrate(dim as u32, 0xDEAD_BEEF);
    index.set_rabitq_params(persisted.clone());

    assert!(index.is_rabitq_active());
    // Every node must now carry an encoded code, and that code must
    // match the persisted rotation's encoding of its f32 vector.
    for i in 0..index.node_len() {
        let code = index
            .node_rabitq(i)
            .expect("node missing rabitq code after set_rabitq_params");
        let expected =
            RabitqEncoded::OneBit(persisted.encode(index.read_node_f32(i).expect("f32 retained")));
        assert_eq!(*code, expected, "node {i} code mismatch after reload");
    }
}

#[test]
fn sq8_memory_savings() {
    // Verify quantized vectors use 4x less memory than f32
    let mut index = HnswIndex::new(make_sq8_config(VectorMetric::L2));

    let dims = 384;
    for i in 0..10u64 {
        let v: Vec<f32> = (0..dims).map(|d| ((i * d) as f32).sin()).collect();
        index.insert(i, v);
    }

    assert!(index.is_quantized());

    for i in 0..index.node_len() {
        let v = index
            .read_node_f32(i)
            .expect("f32 should be retained (offload_vectors=false)");
        assert_eq!(v.len(), dims as usize);
        let q = index.node_sq8(i).expect("should be quantized");
        assert_eq!(q.len(), dims as usize);
        assert_eq!(std::mem::size_of_val(v), q.len() * 4);
    }
}

#[test]
fn sq8_manual_calibration() {
    use crate::quantize::Sq8Params;

    let mut index = HnswIndex::new(HnswConfig {
        quantization: QuantizationCodec::Sq8,
        calibration_threshold: 1000, // High threshold — won't auto-calibrate
        ..make_config(VectorMetric::L2)
    });

    index.insert(1, vec![0.0, 0.0]);
    index.insert(2, vec![1.0, 1.0]);
    index.insert(3, vec![0.5, 0.5]);

    assert!(!index.is_quantized());

    // Manually provide calibration
    let params = Sq8Params {
        mins: vec![-1.0, -1.0],
        maxs: vec![2.0, 2.0],
    };
    index.set_sq8_params(params);

    assert!(index.is_quantized());
    // All existing nodes should now be quantized
    for idx in 0..index.node_len() {
        assert!(index.node_sq8(idx).is_some());
    }

    // Search should work with quantization
    let results = index.search(&[0.0, 0.0], 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn sq8_cosine_metric() {
    let mut index = HnswIndex::new(make_sq8_config(VectorMetric::Cosine));

    // Insert enough to trigger calibration
    index.insert(1, vec![1.0, 0.0]);
    index.insert(2, vec![0.0, 1.0]);
    index.insert(3, vec![0.707, 0.707]);
    index.insert(4, vec![-1.0, 0.0]);
    index.insert(5, vec![0.5, 0.5]);

    assert!(index.is_quantized());

    // Query in direction (1, 0) — should find ID 1 first after reranking
    let results = index.search(&[1.0, 0.0], 2);
    assert_eq!(results[0].id, 1);
}

#[test]
fn sq8_disabled_by_default() {
    let config = HnswConfig::default();
    assert!(!config.quantization.is_active());

    let mut index = HnswIndex::new(config);
    // The default metric is cosine: no vector is all zeros.
    for i in 0..200u64 {
        index.insert(i, vec![i as f32, 1.0]);
    }
    assert!(!index.is_quantized());
}

#[test]
fn search_with_visibility_all_visible() {
    // All nodes visible: should return same results as regular search.
    let mut index = HnswIndex::new(HnswConfig {
        metric: coordinode_core::graph::types::VectorMetric::L2,
        ..HnswConfig::default()
    });
    for i in 0..50u64 {
        index.insert(i, vec![i as f32, 0.0, 0.0]);
    }

    let query = vec![25.0, 0.0, 0.0];
    let regular = index.search(&query, 5);
    let (visible, stats) = index.search_with_visibility(&query, 5, 1.2, 3, |_| true);

    assert_eq!(visible.len(), 5);
    assert_eq!(stats.candidates_filtered, 0);
    assert_eq!(stats.expansion_rounds, 0);
    // Same top result
    assert_eq!(regular[0].id, visible[0].id);
}

#[test]
fn search_with_visibility_filters_invisible() {
    // Even-numbered nodes are invisible. Search should return only odd IDs.
    let mut index = HnswIndex::new(HnswConfig {
        metric: coordinode_core::graph::types::VectorMetric::L2,
        ..HnswConfig::default()
    });
    for i in 0..100u64 {
        index.insert(i, vec![i as f32, 0.0]);
    }

    let query = vec![50.0, 0.0];
    let (results, stats) = index.search_with_visibility(
        &query,
        5,
        1.2,
        3,
        |id| id % 2 != 0, // only odd IDs visible
    );

    assert_eq!(results.len(), 5);
    for r in &results {
        assert!(
            r.id % 2 != 0,
            "invisible even ID {} should be filtered",
            r.id
        );
    }
    assert!(stats.candidates_filtered > 0);
}

#[test]
fn search_with_visibility_all_invisible() {
    // No nodes visible: returns empty.
    let mut index = HnswIndex::new(HnswConfig {
        metric: coordinode_core::graph::types::VectorMetric::L2,
        ..HnswConfig::default()
    });
    for i in 0..20u64 {
        index.insert(i, vec![i as f32]);
    }

    let (results, stats) = index.search_with_visibility(
        &[10.0],
        5,
        1.2,
        3,
        |_| false, // nothing visible
    );

    assert!(results.is_empty());
    assert_eq!(stats.candidates_visible, 0);
    // Should have tried expansion rounds
    assert!(stats.expansion_rounds > 0);
}

#[test]
fn search_with_visibility_expansion_rounds() {
    // Many nodes invisible → expansion rounds needed to fill K results.
    let mut index = HnswIndex::new(HnswConfig {
        metric: coordinode_core::graph::types::VectorMetric::L2,
        ef_search: 10,
        ..HnswConfig::default()
    });
    // Insert 200 nodes, only every 10th is visible
    for i in 0..200u64 {
        index.insert(i, vec![i as f32, 0.0]);
    }

    let (results, stats) = index.search_with_visibility(
        &[100.0, 0.0],
        5,
        1.2,
        3,
        |id| id % 10 == 0, // only 10% visible
    );

    assert_eq!(results.len(), 5);
    for r in &results {
        assert_eq!(r.id % 10, 0, "non-visible ID {} leaked through", r.id);
    }
    // Likely needed at least 1 expansion round due to low visibility
    assert!(stats.candidates_fetched > 5);
}

#[test]
fn search_with_visibility_empty_index() {
    let index = HnswIndex::new(HnswConfig::default());
    let (results, stats) = index.search_with_visibility(&[1.0, 2.0], 5, 1.2, 3, |_| true);
    assert!(results.is_empty());
    assert_eq!(stats.candidates_fetched, 0);
}

#[test]
fn sq8_small_index_still_calibrates() {
    // SQ8 on small index (<1000 vectors) should still work
    // (warning is emitted but calibration proceeds).
    let mut index = HnswIndex::new(make_sq8_config(VectorMetric::L2));

    // Insert 5 vectors (well below SQ8_MIN_VECTORS=1000)
    for i in 0..5u64 {
        index.insert(i, vec![i as f32, 0.0, 0.0]);
    }

    // Should be quantized despite being small (soft warning, not hard block)
    assert!(
        index.is_quantized(),
        "SQ8 should still calibrate on small index"
    );
    assert!(index.sq8_params().is_some());

    // Search should still work correctly
    let results = index.search(&[0.0, 0.0, 0.0], 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
}

#[test]
fn sq8_large_index_no_warning_threshold() {
    // Below 1000 vectors SQ8 calibration overhead is not worth it.
    assert_eq!(
        super::SQ8_MIN_VECTORS,
        1000,
        "SQ8_MIN_VECTORS should be 1000"
    );
}

// ── Offloaded f32 vectors to disk ──────────────────────────────────

fn make_offload_config(metric: VectorMetric) -> HnswConfig {
    HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 50,
        ef_search: 50,
        metric,
        max_dimensions: 65_536,
        quantization: QuantizationCodec::Sq8,
        rerank_candidates: 20,
        calibration_threshold: 5, // Low threshold for testing
        offload_vectors: true,
        property_name: "embedding".to_string(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: 1_000,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    }
}

/// In-memory VectorLoader for testing — stores vectors in a HashMap.
struct TestVectorLoader {
    vectors: HashMap<u64, Vec<f32>>,
}

impl TestVectorLoader {
    fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }

    fn add(&mut self, id: u64, vector: Vec<f32>) {
        self.vectors.insert(id, vector);
    }
}

impl super::VectorLoader for TestVectorLoader {
    fn load_vectors(&self, ids: &[u64], _property: &str) -> HashMap<u64, Vec<f32>> {
        ids.iter()
            .filter_map(|&id| self.vectors.get(&id).map(|v| (id, v.clone())))
            .collect()
    }
}

#[test]
fn offload_drops_f32_after_calibration() {
    // After SQ8 calibration with offload_vectors=true,
    // nodes should have quantized vectors but no f32.
    let mut index = HnswIndex::new(make_offload_config(VectorMetric::L2));
    let mut loader = TestVectorLoader::new();

    for i in 0..10u64 {
        let v: Vec<f32> = vec![i as f32 * 0.1, (i as f32 * 0.2).sin()];
        loader.add(i, v.clone());
        index.insert(i, v);
    }

    assert!(index.is_quantized(), "should be calibrated after threshold");
    assert!(index.is_offloaded(), "should be in offload mode");

    // Verify f32 vectors are dropped from in-memory nodes
    for i in 0..index.node_len() {
        assert!(
            index.read_node_f32(i).is_none(),
            "f32 should be None when offloaded (node {})",
            index.node_id(i)
        );
        assert!(
            index.node_sq8(i).is_some(),
            "quantized should be present (node {})",
            index.node_id(i)
        );
    }
}

#[test]
fn offload_search_with_loader_returns_correct_results() {
    // search_with_loader should produce same ordering as non-offloaded search.
    let mut index = HnswIndex::new(make_offload_config(VectorMetric::L2));
    let mut loader = TestVectorLoader::new();

    let vectors: Vec<Vec<f32>> = (0..20u64)
        .map(|i| vec![i as f32 * 0.1, (i as f32 * 0.3).cos()])
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        loader.add(i as u64, v.clone());
        index.insert(i as u64, v.clone());
    }

    let query = vec![0.5, 0.5];
    let results = index.search_with_loader(&query, 5, &loader);

    assert_eq!(results.len(), 5, "should return 5 results");

    // Verify ordering: each result should be <= the next (distance ascending)
    for i in 0..results.len() - 1 {
        assert!(
            results[i].score <= results[i + 1].score,
            "results should be sorted by distance: {} > {}",
            results[i].score,
            results[i + 1].score
        );
    }

    // Scores are the exact evaluation of the loaded f32 vectors (not of the
    // in-memory SQ8 codes): bit for bit the canonical L2.
    for result in &results {
        let v = loader.vectors.get(&result.id).unwrap();
        assert_eq!(
            result.score,
            metrics::exact::l2(&query, v).unwrap(),
            "node {}",
            result.id
        );
    }
}

#[test]
fn offload_search_fallback_when_not_offloaded() {
    // When offload_vectors=false, search_with_loader should behave
    // identically to regular search().
    let mut index = HnswIndex::new(HnswConfig {
        offload_vectors: false,
        ..make_offload_config(VectorMetric::Cosine)
    });
    let loader = TestVectorLoader::new(); // empty, shouldn't be called

    for i in 0..10u64 {
        let v: Vec<f32> = vec![(i as f32).cos(), (i as f32).sin()];
        index.insert(i, v);
    }

    let query = vec![1.0, 0.0];
    let regular = index.search(&query, 3);
    let with_loader = index.search_with_loader(&query, 3, &loader);

    assert_eq!(regular.len(), with_loader.len());
    for (r, l) in regular.iter().zip(with_loader.iter()) {
        assert_eq!(r.id, l.id, "same result ordering expected");
    }
}

#[test]
fn offload_memory_savings() {
    // Verify offloaded index uses less memory than non-offloaded.
    let dims = 384usize;
    let n = 10u64;

    // Build offloaded index
    let mut offloaded = HnswIndex::new(make_offload_config(VectorMetric::L2));
    for i in 0..n {
        let v: Vec<f32> = (0..dims).map(|d| ((i * d as u64) as f32).sin()).collect();
        offloaded.insert(i, v);
    }

    // Build non-offloaded index
    let mut retained = HnswIndex::new(HnswConfig {
        offload_vectors: false,
        ..make_offload_config(VectorMetric::L2)
    });
    for i in 0..n {
        let v: Vec<f32> = (0..dims).map(|d| ((i * d as u64) as f32).sin()).collect();
        retained.insert(i, v);
    }

    // f32 now lives in the contiguous data_level0 block; offload's
    // drop_f32 re-lays it out without the f32 slot, actually freeing the
    // bytes. The offloaded index must hold no in-RAM f32 so rerank loads
    // from disk; the retained index keeps it.
    let offloaded_has_f32 = offloaded.data_level0.get().is_some_and(|b| b.has_f32());
    let retained_has_f32 = retained.data_level0.get().is_some_and(|b| b.has_f32());
    assert!(
        !offloaded_has_f32,
        "offloaded index must free contiguous-block f32"
    );
    assert!(
        retained_has_f32,
        "retained index must keep contiguous-block f32"
    );
    // read_node_f32 reflects it: None when offloaded, Some otherwise.
    assert!(
        offloaded.read_node_f32(0).is_none(),
        "offloaded f32 read is None"
    );
    assert!(
        retained.read_node_f32(0).is_some(),
        "retained f32 read is Some"
    );
    let _ = (n, dims);
}

/// Regression test: HNSW must update the graph when a vector
/// property is overwritten via SET (e.g. `MATCH (n) SET n.emb = $new_vec`).
///
/// Previously `insert()` returned early ("Already indexed") so the node
/// kept its old position in the graph. Vector similarity searches then
/// returned garbage results because the search used the stale HNSW graph.
#[test]
fn insert_updates_existing_node_vector_and_graph_position() {
    // Build an index with a clear directional layout:
    //   node 1: "up"    [0.0, 1.0]
    //   node 2: "right" [1.0, 0.0]  (decoy)
    //   node 3: "down"  [0.0, -1.0]  (decoy)
    let mut index = HnswIndex::new(make_config(VectorMetric::Cosine));
    index.insert(2, vec![1.0, 0.0]);
    index.insert(3, vec![0.0, -1.0]);
    index.insert(1, vec![0.0, 1.0]);

    // Verify initial state: query "up" → node 1 wins.
    let before = index.search(&[0.0, 1.0], 1);
    assert_eq!(
        before.first().map(|r| r.id),
        Some(1),
        "before update: 'up' query should find node 1"
    );

    // Update node 1 from "up" [0.0, 1.0] to "right" [1.0, 0.0].
    // This simulates: MATCH (n) SET n.emb = [1.0, 0.0]
    index.insert(1, vec![1.0, 0.0]);

    // After update, query "right" → node 1 must now be among top-2 results
    // (node 2 also points "right", so both should score high).
    let after_right = index.search(&[1.0, 0.0], 2);
    let ids_right: Vec<u64> = after_right.iter().map(|r| r.id).collect();
    assert!(
        ids_right.contains(&1),
        "after update: 'right' query must include node 1 (updated vector). Got: {:?}",
        ids_right
    );

    // Query "up" → nothing points up any more: node 1 answers with its new
    // vector's distance, never the 0.0 of its old one. Nodes 1 and 2 now both
    // point right and tie at cosine distance 1.0, so either may come first.
    let after_up = index.search(&[0.0, 1.0], 3);
    let node1 = after_up
        .iter()
        .find(|r| r.id == 1)
        .expect("node 1 is still in the index");
    assert!(
        (node1.score - 1.0).abs() < 1e-6,
        "after update: node 1 must score its new 'right' vector against 'up'. Got: {after_up:?}"
    );
    assert!(
        after_up.iter().all(|r| r.score > 0.5),
        "after update: no node points up any more. Got: {after_up:?}"
    );
}

#[test]
fn atomic_neighbours_track_inserts_and_updates() {
    // Atomic neighbour storage is the sole source of truth.
    // Insert a fan of vectors, then mutate one via update_existing_node;
    // verify every neighbour list (a) tracks the layer count of the
    // node, (b) is bounded by m_max0, (c) only references node IDs
    // present in the index.
    let mut idx = HnswIndex::new(HnswConfig {
        m: 4,
        m_max0: 8,
        ef_construction: 20,
        ef_search: 10,
        metric: VectorMetric::L2,
        max_dimensions: 4,
        quantization: QuantizationCodec::None,
        rerank_candidates: 10,
        calibration_threshold: 1_000,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: 1_000_000,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    });

    for i in 0..30u64 {
        let v: Vec<f32> = (0..4).map(|d| ((i * 13 + d) as f32).sin()).collect();
        idx.insert(i, v);
    }
    // Re-insert one node to exercise update_existing_node.
    idx.insert(7, vec![0.5, -0.5, 0.5, -0.5]);

    // Layer-0 neighbours live in the contiguous block now; it must cover
    // every node.
    assert!(
        idx.data_level0
            .get()
            .is_some_and(|b| b.capacity() >= idx.node_len())
    );
    assert_eq!(idx.len(), 30, "the re-insert keeps 30 ids");
    assert_eq!(
        idx.node_len(),
        31,
        "the re-insert built node 7 in a new slot"
    );

    let mut scratch = Vec::with_capacity(M_MAX0);
    for node_idx in 0..idx.node_len() {
        for level in 0..idx.node_levels(node_idx) {
            idx.layer_snapshot_into(node_idx, level, &mut scratch);
            assert!(
                scratch.len() <= M_MAX0,
                "node {node_idx} level {level} exceeds m_max0 cap",
            );
            for nid in &scratch {
                assert!(
                    (*nid as usize) < idx.node_len(),
                    "dangling neighbour {nid} at node {node_idx} level {level}",
                );
            }
        }
    }
}

#[test]
fn search_exact_returns_recall_1_top_k() {
    // Build a small L2 index with 200 random-ish vectors, then ask
    // search_with_mode(Exact) for the top-10 of a chosen query and
    // confirm the result exactly matches a brute-force reference.
    // The HNSW path can drop true neighbours under low ef; the exact
    // path must not.
    let cfg = HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 40,
        ef_search: 20,
        max_elements: 200,
        metric: VectorMetric::L2,
        ..HnswConfig::default()
    };
    let mut idx = HnswIndex::new(cfg);
    let mut all: Vec<(u64, Vec<f32>)> = Vec::with_capacity(200);
    for i in 0..200u64 {
        // Deterministic spread; uses the id as a coordinate seed so
        // distances are well-separated and the top-k is unambiguous.
        let v: Vec<f32> = (0..8)
            .map(|d| ((i * 13 + d as u64) % 97) as f32 / 17.0)
            .collect();
        idx.insert(i, v.clone());
        all.push((i, v));
    }
    // Query is offset from every inserted vector so distances are
    // unique and the top-k ordering is unambiguous (no tie-break
    // ambiguity to assert against).
    let query: Vec<f32> = (0..8).map(|d| 0.31 + d as f32 * 0.07).collect();

    // The brute-force reference under the same evaluation: canonical L2,
    // ties by id.
    let mut reference: Vec<(u64, f64)> = all
        .iter()
        .map(|(id, v)| (*id, metrics::exact::l2(&query, v).unwrap()))
        .collect();
    reference.sort_by(|a, b| metrics::exact::ranked_order(a.1, a.0, b.1, b.0));
    reference.truncate(10);

    let got = idx.search_with_mode(&query, 10, SearchMode::Exact);
    // Exhaustive search is the reference itself: same ids, same order, same
    // score bits, ties included.
    let got: Vec<(u64, f64)> = got.iter().map(|r| (r.id, r.score)).collect();
    assert_eq!(got, reference);

    // Hnsw mode keeps working untouched.
    let hnsw_got = idx.search_with_mode(&query, 5, SearchMode::Hnsw);
    assert_eq!(hnsw_got.len(), 5);
}

#[test]
fn search_exact_handles_empty_index_and_zero_k() {
    let cfg = HnswConfig {
        metric: VectorMetric::L2,
        ..HnswConfig::default()
    };
    let idx = HnswIndex::new(cfg);
    let q = vec![0.0; 4];
    assert!(idx.search_with_mode(&q, 10, SearchMode::Exact).is_empty());

    let cfg2 = HnswConfig {
        metric: VectorMetric::L2,
        max_elements: 4,
        ..HnswConfig::default()
    };
    let mut idx2 = HnswIndex::new(cfg2);
    idx2.insert(0, vec![1.0, 0.0, 0.0, 0.0]);
    assert!(idx2.search_with_mode(&q, 0, SearchMode::Exact).is_empty());
}

/// Exact search selects and orders its top-k by distance, equal distances
/// by ascending id: of four identical vectors inserted 9, 3, 5, 7 the two
/// smallest ids come back, in id order. Before, the selection kept the first
/// inserted and the order followed insertion.
#[test]
fn exact_search_breaks_ties_by_id() {
    let cfg = HnswConfig {
        metric: VectorMetric::L2,
        max_elements: 8,
        ..HnswConfig::default()
    };
    let mut idx = HnswIndex::new(cfg);
    for id in [9, 3, 5, 7] {
        idx.insert(id, vec![1.0, 0.0, 0.0, 0.0]);
    }
    let ids: Vec<u64> = idx
        .search_with_mode(&[1.0, 0.0, 0.0, 0.0], 2, SearchMode::Exact)
        .iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(ids, [3, 5]);
}

/// A Dot distance whose terms overflow with opposite signs is NaN: it never
/// takes a top-k place from a number, wherever it was inserted. Before, a NaN
/// at the top of the selection heap could never be replaced.
#[test]
fn exact_search_ranks_a_nan_distance_last() {
    let cfg = HnswConfig {
        metric: VectorMetric::DotProduct,
        max_elements: 8,
        ..HnswConfig::default()
    };
    let mut idx = HnswIndex::new(cfg);
    idx.insert(1, vec![1e20, -1e20, 0.0, 0.0]);
    idx.insert(2, vec![1.0, 0.0, 0.0, 0.0]);
    idx.insert(3, vec![0.0, 1.0, 0.0, 0.0]);
    let results = idx.search_with_mode(&[1e20, 1e20, 0.0, 0.0], 2, SearchMode::Exact);
    let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
    assert_eq!(ids, [2, 3], "{results:?}");
}

#[test]
fn max_elements_preallocates_node_storage() {
    // HnswConfig::max_elements sizes the node store's first segment, so
    // steady-state inserts don't pay growth on the hot path.
    let cfg = HnswConfig {
        max_elements: 50_000,
        ..HnswConfig::default()
    };
    let mut idx = HnswIndex::new(cfg);
    // The store allocates lazily on the first insert, sized to max_elements.
    idx.insert(0, vec![0.1; 16]);

    assert!(
        idx.data_level0
            .get()
            .is_some_and(|b| b.capacity() >= 50_000),
        "data_level0 capacity < max_elements 50_000"
    );
    // HashMap::with_capacity may round up; just assert it's non-zero.
    assert!(idx.id_to_idx.capacity() >= 50_000);
}

#[test]
fn inserts_past_max_elements_keep_node_addresses() {
    // A node's vector keeps its address while the index grows, inside the
    // max_elements estimate and past it: readers hold such pointers across
    // concurrent inserts, so growth must never move a node.
    let cfg = HnswConfig {
        m: 4,
        m_max0: 8,
        max_elements: 200,
        max_dimensions: 4,
        ..HnswConfig::default()
    };
    let mut idx = HnswIndex::new(cfg);
    let vector = |i: u64| -> Vec<f32> { (0..4).map(|d| ((i * 7 + d) as f32).sin()).collect() };
    idx.insert(0, vector(0));
    let first = idx.get_vector(0).map(<[f32]>::as_ptr);
    for i in 1..700u64 {
        idx.insert(i, vector(i));
    }
    assert_eq!(idx.len(), 700);
    assert_eq!(
        idx.get_vector(0).map(<[f32]>::as_ptr),
        first,
        "node 0 moved"
    );
    for i in [0u64, 199, 200, 399, 400, 699] {
        let at = idx.idx_for_id_for_test(i).expect("inserted");
        assert_eq!(idx.get_vector(at), Some(vector(i).as_slice()), "node {i}");
    }
}

#[test]
fn insert_batch_matches_serial_insert_topology() {
    // insert_batch must produce a functional index for query-correctness
    // purposes, even though its parallel build is nondeterministic and its
    // topology never equals the sequential graph's. The test builds the
    // same deterministic workload two ways (per-item insert and one
    // insert_batch call), then measures search recall@10 against exact
    // brute-force ground truth. Serial recall is the deterministic sanity
    // anchor; batched recall must clear an absolute functional floor.
    // Comparing batched topology to the (also approximate, deterministic)
    // serial graph was the previous bar and was inherently flaky.
    fn make_cfg() -> HnswConfig {
        HnswConfig {
            m: 8,
            m_max0: 16,
            ef_construction: 50,
            ef_search: 50,
            metric: VectorMetric::L2,
            max_dimensions: 8,
            quantization: QuantizationCodec::None,
            rerank_candidates: 50,
            calibration_threshold: 10_000,
            offload_vectors: false,
            property_name: String::new(),
            rerank_mode: RerankMode::Inline,
            rerank_oversample_factor: 1.0,
            alpha_pruning: 1.0,
            max_elements: 1_000,
            retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
        }
    }
    fn make_vec(i: u64) -> Vec<f32> {
        (0..8).map(|d| ((i * 31 + d) as f32 * 0.1).sin()).collect()
    }

    let n = 500u64;
    let mut serial = HnswIndex::new(make_cfg());
    for i in 0..n {
        serial.insert(i, make_vec(i));
    }

    let mut batched = HnswIndex::new(make_cfg());
    batched.insert_batch((0..n).map(|i| (i, make_vec(i))).collect());

    // Sanity: both indexes ingest every item.
    assert_eq!(serial.len(), n as usize);
    assert_eq!(batched.len(), n as usize);

    // Query-correctness invariant. The batched graph is built by a
    // nondeterministic parallel plan (within-batch plan staleness, the
    // shared level RNG, rayon apply order), so its topology never equals
    // the serial graph's run to run. Asserting topology overlap with the
    // (also approximate) serial index is therefore inherently flaky.
    // Instead measure what actually matters: recall@10 against exact
    // brute-force ground truth. The bar is that batched insert retrieves
    // about as well as serial insert (within a small margin) and clears
    // an absolute floor. Both quantities are stable because absolute
    // recall does not depend on the exact graph structure.
    fn brute_force_top10(corpus: &[(u64, Vec<f32>)], q: &[f32]) -> std::collections::HashSet<u64> {
        let mut scored: Vec<(u64, f32)> = corpus
            .iter()
            .map(|(id, v)| {
                let d: f32 = v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
                (*id, d)
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.into_iter().take(10).map(|(id, _)| id).collect()
    }

    let corpus: Vec<(u64, Vec<f32>)> = (0..n).map(|i| (i, make_vec(i))).collect();
    let q_ids = [3u64, 17, 31, 64, 128, 257, 384, 499];
    let mut serial_hits = 0usize;
    let mut batched_hits = 0usize;
    let mut total = 0usize;
    for &qid in &q_ids {
        let q = make_vec(qid);
        let truth = brute_force_top10(&corpus, &q);
        let serial_top: std::collections::HashSet<u64> =
            serial.search(&q, 10).into_iter().map(|r| r.id).collect();
        let batched_top: std::collections::HashSet<u64> =
            batched.search(&q, 10).into_iter().map(|r| r.id).collect();
        serial_hits += serial_top.intersection(&truth).count();
        batched_hits += batched_top.intersection(&truth).count();
        total += truth.len();
    }
    let serial_recall = serial_hits as f64 / total as f64;
    let batched_recall = batched_hits as f64 / total as f64;

    // Serial insert is deterministic and near-perfect on this easy
    // 8-dim / 500-point workload: a sanity check that the harness and the
    // ground truth line up.
    assert!(
        serial_recall >= 0.9,
        "serial recall@10 {serial_recall:.2} unexpectedly low (harness issue?)",
    );
    // Batched insert is built by a nondeterministic parallel plan, so its
    // recall legitimately varies run to run (measured ~0.70 to ~0.99 here).
    // This guards against a catastrophically broken batch build (the
    // apply-phase backfill bug once collapsed recall to ~0.02), not fine
    // recall regressions, which the dedicated recall-snapshot tests track.
    // The floor sits below the observed spread so it never flakes while
    // still catching a builder that loses the majority of true neighbours.
    assert!(
        batched_recall >= 0.55,
        "batched recall@10 {batched_recall:.2} below the functional floor (build broken?)",
    );
}

#[test]
fn insert_batch_chunked_preserves_recall_vs_brute_force() {
    // Regression test for the apply-phase backfill bug: previously the
    // prune-then-cas_append sequence dropped every backfilled
    // candidate, leaving new nodes with valid outgoing edges but no
    // incoming back-edges from existing hubs. At chunked-batch scale
    // (SIFT1M: 1k items × 1000 calls) this collapsed search recall to
    // ~0.02 across the entire ef_search sweep. Smaller in-tree test:
    // 5k vectors, 1k chunks, recall@10 must clear a meaningful floor.
    let dim = 16usize;
    let n_train = 5_000usize;
    let n_query = 50usize;
    let k = 10usize;
    let chunk = 1_000usize;

    let cfg = HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 100,
        ef_search: 64,
        metric: VectorMetric::L2,
        max_dimensions: dim as u32,
        quantization: QuantizationCodec::None,
        rerank_candidates: 64,
        calibration_threshold: 100_000,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: n_train as u32,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    };
    fn make_vec(i: u64, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|d| ((i * 31 + d as u64) as f32 * 0.13).sin())
            .collect()
    }

    let mut idx = HnswIndex::new(cfg);
    let mut inserted = 0usize;
    while inserted < n_train {
        let end = (inserted + chunk).min(n_train);
        let batch: Vec<(u64, Vec<f32>)> = (inserted..end)
            .map(|i| (i as u64, make_vec(i as u64, dim)))
            .collect();
        idx.insert_batch(batch);
        inserted = end;
    }
    assert_eq!(idx.len(), n_train);

    // Recall@k against brute-force ground truth for held-out queries.
    // Queries derived from corpus IDs near the middle of the batch
    // boundary so they exercise nodes from multiple chunks.
    let mut hits = 0u64;
    let mut total = 0u64;
    for q in (0..n_query).map(|i| (i * 73) as u64) {
        let query = make_vec(q, dim);
        let mut bf: Vec<(f32, u64)> = (0..n_train as u64)
            .map(|i| {
                let v = make_vec(i, dim);
                let d = metrics::euclidean_distance_squared(&query, &v);
                (d, i)
            })
            .collect();
        bf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let gt: HashSet<u64> = bf.iter().take(k).map(|&(_, id)| id).collect();
        let got: HashSet<u64> = idx.search(&query, k).into_iter().map(|r| r.id).collect();
        hits += gt.intersection(&got).count() as u64;
        total += k as u64;
    }
    let recall = hits as f64 / total as f64;
    eprintln!(
        "insert_batch chunked recall@{k} on {n_train} vectors / {n_query} queries: {recall:.3}"
    );
    // Floor of 0.5 catches the catastrophic-collapse regression
    // (pre-fix observed ~0.02) without making the test fragile to
    // small graph-quality variations across rng seeds.
    assert!(
        recall >= 0.5,
        "chunked insert_batch recall {recall:.3} below floor 0.5 — \
             apply-phase backfill regression?"
    );
}

/// One large `insert_batch` must build a usable graph, not only chunked ones.
/// Every plan of a batch is computed against the graph as it stood before
/// the batch; when the batch dwarfs that graph (5k items over a 64-node
/// seed), each new node links only to the seed and the graph degenerates
/// into a star with near-zero recall. The batch must reach the same floor
/// as the chunked build of the test above, on the same data.
#[test]
fn a_single_large_insert_batch_keeps_recall() {
    let dim = 16usize;
    let n_train = 5_000usize;
    let n_query = 50usize;
    let k = 10usize;
    let cfg = HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 100,
        ef_search: 64,
        metric: VectorMetric::L2,
        max_dimensions: dim as u32,
        quantization: QuantizationCodec::None,
        rerank_candidates: 64,
        calibration_threshold: 100_000,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: n_train as u32,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    };
    fn make_vec(i: u64, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|d| ((i * 31 + d as u64) as f32 * 0.13).sin())
            .collect()
    }

    let mut idx = HnswIndex::new(cfg);
    idx.insert_batch((0..n_train as u64).map(|i| (i, make_vec(i, dim))).collect());
    assert_eq!(idx.len(), n_train);

    let mut hits = 0u64;
    let mut total = 0u64;
    for q in (0..n_query).map(|i| (i * 73) as u64) {
        let query = make_vec(q, dim);
        let mut bf: Vec<(f32, u64)> = (0..n_train as u64)
            .map(|i| {
                (
                    metrics::euclidean_distance_squared(&query, &make_vec(i, dim)),
                    i,
                )
            })
            .collect();
        bf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let gt: HashSet<u64> = bf.iter().take(k).map(|&(_, id)| id).collect();
        let got: HashSet<u64> = idx.search(&query, k).into_iter().map(|r| r.id).collect();
        hits += gt.intersection(&got).count() as u64;
        total += k as u64;
    }
    let recall = hits as f64 / total as f64;
    assert!(
        recall >= 0.5,
        "single-batch insert_batch recall {recall:.3} below floor 0.5"
    );
}

/// Live nodes no path from the entry point reaches, over the edges of every
/// layer: what a search can never return.
fn unreachable_live_nodes(index: &HnswIndex) -> usize {
    let Some((entry, _)) = index.entry_point.for_search() else {
        return 0;
    };
    let n = index.node_len();
    let mut reached = vec![false; n];
    let mut stack = vec![entry];
    reached[entry] = true;
    while let Some(i) = stack.pop() {
        for level in 0..index.node_levels(i) {
            for nb in index.layer_snapshot(i, level) {
                let nb = nb as usize;
                if nb < n && !reached[nb] {
                    reached[nb] = true;
                    stack.push(nb);
                }
            }
        }
    }
    (0..n)
        .filter(|&i| !reached[i] && index.nodes().state(i) == data_level0::NodeState::Live)
        .count()
}

/// `n` points of `dim` components spread uniformly over `[-0.5, 0.5)`,
/// deterministic per index.
fn uniform_points(n: usize, dim: usize) -> Vec<(u64, Vec<f32>)> {
    (0..n)
        .map(|i| {
            let v = (0..dim)
                .map(|d| {
                    let seed = (i.wrapping_mul(2_654_435_761).wrapping_add(d * 6_700_417)) as u32;
                    let bits = (seed ^ (seed >> 13)) & 0x00FF_FFFF;
                    (bits as f32 / 16_777_216.0) - 0.5
                })
                .collect();
            (i as u64, v)
        })
        .collect()
}

/// How many of `items` a search for their own vector returns first.
fn self_recall(index: &HnswIndex, items: &[(u64, Vec<f32>)]) -> usize {
    items
        .iter()
        .filter(|(id, v)| index.search(v, 1).first().is_some_and(|r| r.id == *id))
        .count()
}

/// An L2 index over the build tests' points, its level draws seeded by
/// `seed`.
fn seeded_l2_index(seed: u64) -> HnswIndex {
    let index = HnswIndex::new(HnswConfig {
        ef_search: 64,
        metric: VectorMetric::L2,
        max_dimensions: 65_536,
        ..Default::default()
    });
    index.rng_state.store(
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        std::sync::atomic::Ordering::Relaxed,
    );
    index
}

/// Inserting one by one leaves every node findable, whatever levels the
/// draws give. A full layer-0 list used to keep its closest `m_max0` by
/// plain truncation, so nodes between dense regions lost their last
/// incoming edge: with these seeds 116, 131 and 123 of the 600 points were
/// unreachable and no search returned them.
#[test]
fn serial_inserts_leave_every_node_findable() {
    let items = uniform_points(600, 20);
    for seed in [7u64, 35, 38] {
        let mut index = seeded_l2_index(seed);
        for (id, v) in items.clone() {
            index.insert(id, v);
        }
        assert_eq!(unreachable_live_nodes(&index), 0, "seed {seed}");
        assert_eq!(self_recall(&index, &items), items.len(), "seed {seed}");
    }
}

/// A batch links as well as inserting one by one even at the worst a wide
/// pool can interleave: every insert of each wave planned before any of
/// them links, so none sees another. The bound a concurrent build of the
/// same points cannot fall below.
#[test]
fn a_blind_wave_build_leaves_every_node_findable() {
    let items = uniform_points(600, 20);
    for seed in [7u64, 35, 38] {
        for bulk in [false, true] {
            let mut index = seeded_l2_index(seed);
            index
                .blind_waves
                .store(true, std::sync::atomic::Ordering::Relaxed);
            if bulk {
                index.bulk_build(items.clone());
            } else {
                index.insert_batch(items.clone());
            }
            assert_eq!(
                unreachable_live_nodes(&index),
                0,
                "seed {seed} (bulk: {bulk})"
            );
            assert_eq!(
                self_recall(&index, &items),
                items.len(),
                "seed {seed} (bulk: {bulk})"
            );
        }
    }
}

/// A draw whose uniform is zero gives a finite level: an infinite one cast
/// to `usize` asked for `usize::MAX` layers and aborted the insert. The
/// generator state 1 reaches that draw at once.
#[test]
fn a_level_draw_is_always_finite() {
    let index = seeded_l2_index(0);
    assert_eq!(
        index.rng_state.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "the state whose first draw has a zero uniform"
    );
    let level = index.random_level();
    assert!(level < 64, "a level of {level}");
}

/// Concurrent inserts each take a draw of their own: the generator advances
/// once per draw however many threads draw at once. A read and a separate
/// store let threads take the same state, so they drew the same level and
/// the build's level spread followed how the threads ran.
#[test]
fn concurrent_level_draws_never_repeat_a_state() {
    let index = seeded_l2_index(7);
    let start = index.rng_state.load(std::sync::atomic::Ordering::Relaxed);
    let threads = 8u64;
    let draws = 20_000u64;
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..draws {
                    index.random_level();
                }
            });
        }
    });
    let mut expected = start;
    for _ in 0..threads * draws {
        expected ^= expected << 13;
        expected ^= expected >> 7;
        expected ^= expected << 17;
    }
    assert_eq!(
        index.rng_state.load(std::sync::atomic::Ordering::Relaxed),
        expected,
        "every draw advanced the generator once"
    );
}

/// A build batch on a wide thread pool leaves no node unreachable, and its
/// self-recall stays near that of inserting one by one. Pruning a full
/// list can drop a node's last incoming edge, and inserts in flight together
/// never see each other, so before the reachability repair a 128-thread
/// build of 600 points left nodes no search could return. Many builds,
/// since the graph depends on how the inserts interleave.
#[test]
fn a_build_batch_on_a_wide_pool_leaves_no_node_unreachable() {
    let n = 600usize;
    let items = uniform_points(n, 20);
    let cfg = || HnswConfig {
        ef_search: 64,
        metric: VectorMetric::L2,
        max_dimensions: 65_536,
        ..Default::default()
    };
    let self_recall = |index: &HnswIndex| self_recall(index, &items);
    let mut serial = HnswIndex::new(cfg());
    for (id, v) in items.clone() {
        serial.insert(id, v);
    }
    // Reachability is exact; recall is approximate, and a concurrent build
    // is a different graph from the serial one. A graph that lost
    // reachability fell far below this floor.
    let floor = self_recall(&serial) * 9 / 10;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(128)
        .build()
        .unwrap();
    for build in 0..12 {
        for bulk in [false, true] {
            let mut batched = HnswIndex::new(cfg());
            pool.install(|| {
                if bulk {
                    batched.bulk_build(items.clone());
                } else {
                    batched.insert_batch(items.clone());
                }
            });
            assert_eq!(
                unreachable_live_nodes(&batched),
                0,
                "build {build} (bulk: {bulk}): a node no search can reach"
            );
            let got = self_recall(&batched);
            assert!(
                got >= floor,
                "build {build} (bulk: {bulk}): self-recall {got}/{n} below {floor}"
            );
        }
    }
}

#[test]
fn insert_batch_below_threshold_runs_sequentially() {
    // Small batches (< 16 items) bypass rayon; the result must still
    // include every item with correct neighbour ids.
    let cfg = HnswConfig {
        m: 4,
        m_max0: 8,
        ef_construction: 20,
        ef_search: 10,
        metric: VectorMetric::L2,
        max_dimensions: 4,
        quantization: QuantizationCodec::None,
        rerank_candidates: 10,
        calibration_threshold: 1_000,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: 100,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    };
    let mut idx = HnswIndex::new(cfg);
    let items: Vec<(u64, Vec<f32>)> = (0..8u64)
        .map(|i| (i, (0..4).map(|d| ((i * 5 + d) as f32).sin()).collect()))
        .collect();
    idx.insert_batch(items);
    assert_eq!(idx.len(), 8);
    // Each known id must round-trip through search.
    for i in 0..8u64 {
        let q: Vec<f32> = (0..4).map(|d| ((i * 5 + d) as f32).sin()).collect();
        let top = idx.search(&q, 1);
        assert_eq!(top[0].id, i, "search miss on its own vector for id {i}");
    }
}

#[test]
fn insert_batch_handles_mixed_new_and_existing_ids() {
    // Half the batch is updates of existing ids — those must route
    // through update_existing_node, not the plan/apply path. After
    // the batch the index size matches the unique-id count.
    let cfg = HnswConfig {
        m: 4,
        m_max0: 8,
        ef_construction: 20,
        ef_search: 10,
        metric: VectorMetric::L2,
        max_dimensions: 4,
        quantization: QuantizationCodec::None,
        rerank_candidates: 10,
        calibration_threshold: 1_000,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: 100,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    };
    let mut idx = HnswIndex::new(cfg);
    for i in 0..10u64 {
        let v: Vec<f32> = (0..4).map(|d| ((i + d as u64) as f32).sin()).collect();
        idx.insert(i, v);
    }
    // Batch: update first 5 + insert 10 new.
    let mut items = Vec::new();
    for i in 0..5u64 {
        let v: Vec<f32> = (0..4)
            .map(|d| ((100 + i + d as u64) as f32).cos())
            .collect();
        items.push((i, v));
    }
    for i in 10..20u64 {
        let v: Vec<f32> = (0..4).map(|d| ((i + d as u64) as f32).sin()).collect();
        items.push((i, v));
    }
    idx.insert_batch(items);
    assert_eq!(idx.len(), 20, "expected 20 unique ids after batch");
}

/// Inserts through a shared borrow from many threads, while other threads
/// search the same index, land every node: each one is found again by its
/// own vector, and no search observes a result outside the inserted ids or a
/// panic from a half-built node.
#[test]
fn concurrent_inserts_under_live_search_land_every_node() {
    let cfg = HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 50,
        ef_search: 50,
        metric: VectorMetric::L2,
        max_dimensions: 8,
        quantization: QuantizationCodec::None,
        rerank_candidates: 50,
        calibration_threshold: 10_000,
        offload_vectors: false,
        property_name: String::new(),
        rerank_mode: RerankMode::Inline,
        rerank_oversample_factor: 1.0,
        alpha_pruning: 1.0,
        max_elements: 1_000,
        retired_bytes_budget: super::DEFAULT_RETIRED_BYTES_BUDGET,
    };
    let idx = HnswIndex::new(cfg);
    // Vectors with no locality along the id, so the comparison with a
    // sequential build below measures concurrency, not insert order (vectors
    // on a smooth curve favour inserting along it).
    let vector = |i: u64| -> Vec<f32> { scattered_vector(i, 8) };
    let inserting = std::sync::atomic::AtomicUsize::new(4);
    std::thread::scope(|s| {
        for t in 0..4u64 {
            let (idx, inserting) = (&idx, &inserting);
            s.spawn(move || {
                for i in (t * 200)..(t * 200 + 200) {
                    idx.insert_shared(i, &vector(i));
                }
                inserting.fetch_sub(1, std::sync::atomic::Ordering::Release);
            });
        }
        for _ in 0..2 {
            let (idx, inserting) = (&idx, &inserting);
            s.spawn(move || {
                let mut q = 0u64;
                while inserting.load(std::sync::atomic::Ordering::Acquire) > 0 {
                    for hit in idx.search(&vector(q % 800), 5) {
                        assert!(
                            hit.id < 800,
                            "search returned an id never inserted: {}",
                            hit.id
                        );
                    }
                    q += 7;
                }
            });
        }
    });

    assert_eq!(idx.len(), 800);
    let recovered = |index: &HnswIndex| {
        (0..800u64)
            .filter(|&i| index.search(&vector(i), 1).first().map(|r| r.id) == Some(i))
            .count()
    };
    // The same items inserted one by one into a twin index set the bar:
    // concurrency may cost a little, never a collapse.
    let mut serial = HnswIndex::new(idx.config().clone());
    for i in 0..800u64 {
        serial.insert(i, vector(i));
    }
    let (concurrent, sequential) = (recovered(&idx), recovered(&serial));
    assert!(
        concurrent + 800 / 200 >= sequential,
        "{concurrent}/800 nodes found by their own vector after concurrent inserts, \
         {sequential}/800 after sequential ones",
    );
}

/// A vector of `dim` components in [-1, 1) with no order along `i`: a
/// splitmix64 stream seeded by `i`. Sequential ids then carry no geometric
/// locality, so the insert order does not favour one build over another.
fn scattered_vector(i: u64, dim: usize) -> Vec<f32> {
    let mut state = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    (0..dim)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Updating the entry point keeps the index reachable: with a single node,
/// the update's only link candidate is the node it replaces, so the
/// replacement must take over the entry point or nothing leads to it.
#[test]
fn updating_the_entry_point_keeps_the_node_reachable() {
    let mut index = HnswIndex::new(make_config(VectorMetric::L2));
    index.insert(1, vec![0.0, 0.0]);
    index.insert(1, vec![5.0, 5.0]);
    let hits = index.search(&[5.0, 5.0], 1);
    assert_eq!(hits.len(), 1, "the updated node is unreachable");
    assert_eq!(hits[0].id, 1);
    assert!(hits[0].score.abs() < 1e-6);
    // With more nodes, updating whichever node is the entry point hands the
    // entry point to the replacement, and every id stays findable.
    for i in 2..30u64 {
        index.insert(i, vec![i as f32, 1.0]);
    }
    let ep = index.entry_point_idx_for_test().expect("non-empty");
    let ep_id = index.node_id(ep);
    let moved = [ep_id as f32 + 0.25, 1.0];
    index.insert(ep_id, moved.to_vec());
    let replacement = index.idx_for_id_for_test(ep_id).expect("present");
    assert_eq!(index.entry_point_idx_for_test(), Some(replacement));
    assert_eq!(index.search(&moved, 1)[0].id, ep_id);
    for i in 1..30u64 {
        let at = if i == ep_id {
            moved.to_vec()
        } else if i == 1 {
            vec![5.0, 5.0]
        } else {
            vec![i as f32, 1.0]
        };
        assert_eq!(index.search(&at, 1)[0].id, i, "id {i} lost");
    }
}

/// An update through a shared borrow moves the id to its new vector: the
/// old node is never a result again, and the id is counted once.
#[test]
fn update_moves_the_id_and_retires_the_old_node() {
    let mut index = HnswIndex::new(make_config(VectorMetric::L2));
    for i in 0..50u64 {
        index.insert(i, vec![i as f32, 0.0, 0.0]);
    }
    let before = index.idx_for_id_for_test(7).expect("inserted");
    index.insert(7, vec![1000.0, 0.0, 0.0]);
    let after = index.idx_for_id_for_test(7).expect("still present");
    assert_ne!(before, after, "an update builds the node in a new slot");
    assert_eq!(index.nodes().state(before), data_level0::NodeState::Retired);
    assert_eq!(index.len(), 50);
    // The old position no longer answers for id 7; the new one does.
    let near_old = index.search(&[7.0, 0.0, 0.0], 3);
    assert!(
        near_old.iter().all(|r| r.id != 7),
        "retired node returned: {near_old:?}"
    );
    assert_eq!(index.search(&[1000.0, 0.0, 0.0], 1)[0].id, 7);
    let exact = index.search_with_mode(&[7.0, 0.0, 0.0], 50, SearchMode::Exact);
    assert_eq!(
        exact.iter().filter(|r| r.id == 7).count(),
        1,
        "id counted once"
    );
}

/// Let the epoch run what it deferred. Each test runs in a process of its
/// own, so once this thread stops pinning, flushes move the epoch past every
/// pin the test took; each flush runs only a few queued batches, so it takes
/// many to run them all.
fn drain_epoch() {
    for _ in 0..4_096 {
        crossbeam_epoch::pin().flush();
    }
}

/// Take every removed slot of `index` to free now, whatever the sweep
/// threshold: the epoch, the sweep, the epoch again.
fn reclaim_now(index: &HnswIndex) {
    drain_epoch();
    let guard = crossbeam_epoch::pin();
    index.sweep_unlinkable(&guard);
    drop(guard);
    drain_epoch();
}

/// No list of a node that can be reached names a free slot: a free slot is
/// about to hold another node, and an edge to it would lead there.
fn assert_no_edge_to_a_free_slot(index: &HnswIndex) {
    let store = index.nodes();
    for idx in 0..index.node_len() {
        if !store.state(idx).is_linkable() {
            continue;
        }
        for level in 0..index.node_levels(idx) {
            for nb in index.layer_snapshot(idx, level) {
                assert_ne!(
                    store.state(nb as usize),
                    data_level0::NodeState::Free,
                    "node {idx} layer {level} names free slot {nb}"
                );
            }
        }
    }
}

/// A graph dense enough that every node is found by its own vector before
/// any removal, so a miss after one is the removal's doing.
fn removal_config() -> HnswConfig {
    HnswConfig {
        m: 16,
        m_max0: 32,
        ef_construction: 100,
        ef_search: 64,
        metric: VectorMetric::L2,
        ..Default::default()
    }
}

/// A removed id is never a result again, by approximate or exact search, and
/// is counted out; every other id stays findable, through the repair of the
/// removed nodes' neighbours. Removing it twice, or removing an id never
/// inserted, is a no-op that says so.
#[test]
fn removed_ids_are_never_results() {
    let index = HnswIndex::new(removal_config());
    for i in 0..200u64 {
        index.insert_shared(i, &scattered_vector(i, 8));
    }
    for i in 0..200u64 {
        assert_eq!(
            index.search(&scattered_vector(i, 8), 1)[0].id,
            i,
            "baseline {i}"
        );
    }
    for i in (0..200u64).step_by(3) {
        assert!(index.remove(i), "id {i} was present");
    }
    assert!(!index.remove(0), "a second removal finds nothing");
    assert!(!index.remove(9_999), "an id never inserted is not removed");
    assert_eq!(index.len(), 200 - 67);
    for i in 0..200u64 {
        let removed = i % 3 == 0;
        assert_eq!(index.contains(i), !removed, "id {i}");
        let hits = index.search(&scattered_vector(i, 8), 5);
        if removed {
            assert!(hits.iter().all(|h| h.id != i), "removed id {i} returned");
        } else {
            assert_eq!(hits[0].id, i, "live id {i} lost");
        }
        let exact = index.search_with_mode(&scattered_vector(i, 8), 3, SearchMode::Exact);
        assert_eq!(exact[0].id == i, !removed, "exact search on id {i}");
    }
}

/// Removing the entry point hands it to a node that serves, so every other
/// id stays reachable; removing every node empties the index, which then
/// takes inserts again.
#[test]
fn removing_the_entry_point_hands_it_over() {
    let index = HnswIndex::new(removal_config());
    for i in 0..50u64 {
        index.insert_shared(i, &scattered_vector(i, 6));
    }
    let ep = index.entry_point_idx_for_test().expect("non-empty");
    let ep_id = index.node_id(ep);
    assert!(index.remove(ep_id));
    let new_ep = index.entry_point_idx_for_test().expect("still non-empty");
    assert_ne!(new_ep, ep);
    assert_eq!(index.nodes().state(new_ep), data_level0::NodeState::Live);
    for i in (0..50u64).filter(|&i| i != ep_id) {
        assert_eq!(
            index.search(&scattered_vector(i, 6), 1)[0].id,
            i,
            "id {i} lost"
        );
    }
    for i in 0..50u64 {
        index.remove(i);
    }
    assert_eq!(index.len(), 0);
    assert!(
        index.entry_point_idx_for_test().is_none(),
        "an empty index has no entry"
    );
    assert!(index.search(&scattered_vector(1, 6), 3).is_empty());
    index.insert_shared(77, &scattered_vector(77, 6));
    assert_eq!(index.search(&scattered_vector(77, 6), 1)[0].id, 77);
}

/// The slots of removed nodes come back: once the epoch has passed the
/// removal and the sweep, no list names them and new inserts take them
/// instead of growing the store.
#[test]
fn removed_slots_are_reused_after_reclamation() {
    let index = HnswIndex::new(removal_config());
    for i in 0..300u64 {
        index.insert_shared(i, &scattered_vector(i, 8));
    }
    for i in 0..100u64 {
        index.remove(i * 3);
    }
    reclaim_now(&index);
    let stats = index.publication_stats();
    assert_eq!(stats.free_slots, 100, "{stats:?}");
    assert_eq!(stats.retired_nodes, 0, "{stats:?}");
    assert_no_edge_to_a_free_slot(&index);
    for i in 1_000..1_100u64 {
        index.insert_shared(i, &scattered_vector(i, 8));
    }
    assert_eq!(index.node_len(), 300, "the inserts took the free slots");
    assert_eq!(index.publication_stats().free_slots, 0);
    assert_eq!(index.len(), 300);
    for i in (0..300u64).chain(1_000..1_100) {
        let removed = i < 300 && i % 3 == 0;
        let exact = index.search_with_mode(&scattered_vector(i, 8), 1, SearchMode::Exact);
        assert_eq!(exact[0].id == i, !removed, "id {i}");
        if !removed {
            assert_eq!(exact[0].score, 0.0, "id {i} reads its own vector");
        }
    }
}

/// A reorder compacts the slots of removed and replaced nodes away: the
/// store holds the live nodes only, each id maps to its own node, and the
/// queued slots are forgotten rather than reused under the new numbering.
#[test]
fn reorder_compacts_removed_and_replaced_slots() {
    let mut index = HnswIndex::new(removal_config());
    for i in 0..200u64 {
        index.insert(i, scattered_vector(i, 8));
    }
    for i in 0..50u64 {
        index.insert(i, scattered_vector(i + 10_000, 8));
    }
    for i in 150..180u64 {
        assert!(index.remove(i));
    }
    index.reorder_for_cache_locality();
    assert_eq!(index.node_len(), 170);
    assert_eq!(index.len(), 170);
    assert_eq!(index.publication_stats().retired_nodes, 0);
    assert_eq!(index.publication_stats().free_slots, 0);
    for idx in 0..170 {
        assert_eq!(index.nodes().state(idx), data_level0::NodeState::Live);
        let id = index.node_id(idx);
        assert_eq!(index.idx_for_id_for_test(id), Some(idx), "id {id}");
    }
    for i in (0..150u64).chain(180..200) {
        let at = if i < 50 {
            scattered_vector(i + 10_000, 8)
        } else {
            scattered_vector(i, 8)
        };
        assert_eq!(index.search(&at, 1)[0].id, i, "id {i}");
    }
    drain_epoch();
    index.insert_shared(5_000, &scattered_vector(5_000, 8));
    assert_eq!(index.node_len(), 171, "no stale free slot was reused");
}

/// Removals, re-inserts and searches run together on one index: searches
/// return only ids that exist, no slot is reused while it is reachable (a
/// reused slot would answer for the wrong id, which the final exact pass
/// catches), and the slots of removed nodes are reused.
#[test]
fn concurrent_removals_reinserts_and_searches_stay_consistent() {
    let index = HnswIndex::new(removal_config());
    const IDS: u64 = 400;
    const ROUNDS: u64 = 6;
    for i in 0..IDS {
        index.insert_shared(i, &scattered_vector(i, 8));
    }
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        let writers: Vec<_> = (0..2u64)
            .map(|w| {
                let index = &index;
                s.spawn(move || {
                    for round in 1..=ROUNDS {
                        for i in (w..IDS).step_by(2) {
                            assert!(index.remove(i), "id {i} round {round}");
                            index.insert_shared(i, &scattered_vector(i + round * IDS, 8));
                        }
                    }
                })
            })
            .collect();
        for _ in 0..2 {
            let (index, stop) = (&index, &stop);
            s.spawn(move || {
                let mut q = 0u64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let hits = index.search(&scattered_vector(q % (IDS * ROUNDS), 8), 10);
                    let mut seen = HashSet::new();
                    for h in &hits {
                        assert!(h.id < IDS, "a search returned unknown id {}", h.id);
                        assert!(seen.insert(h.id), "id {} returned twice", h.id);
                    }
                    q += 1;
                }
            });
        }
        // The searchers run until the writers are done.
        for writer in writers {
            writer.join().expect("writer panicked");
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    assert_eq!(index.len(), IDS as usize);
    for i in 0..IDS {
        let at = scattered_vector(i + ROUNDS * IDS, 8);
        let exact = index.search_with_mode(&at, 1, SearchMode::Exact);
        assert_eq!(exact[0].id, i, "id {i} answers for its final vector");
        assert_eq!(exact[0].score, 0.0, "id {i} reads its own final vector");
    }
    let total_inserts = (IDS * (ROUNDS + 1)) as usize;
    assert!(
        index.node_len() < total_inserts,
        "no slot was reused: {} slots for {total_inserts} inserts",
        index.node_len()
    );
    assert_no_edge_to_a_free_slot(&index);
}

/// A removed node stops being a result before the count drops: a reader that
/// sees the count fall finds no more results than it says. Two nodes; in
/// each round one is removed on another thread while a reader waits for the
/// count to read one and searches at once, then it is inserted again. Many
/// rounds, so the reader lands inside a removal.
#[test]
fn a_removed_node_is_no_result_once_the_count_says_it_is_gone() {
    let index = HnswIndex::new(removal_config());
    index.insert_shared(1, &scattered_vector(1, 8));
    index.insert_shared(2, &scattered_vector(2, 8));
    let query = scattered_vector(1, 8);
    for round in 0..10_000u64 {
        let id = 1 + round % 2;
        std::thread::scope(|s| {
            let remover = s.spawn(|| assert!(index.remove(id)));
            while index.len() != 1 {
                std::hint::spin_loop();
            }
            let hits = index.search(&query, 10);
            assert!(
                hits.len() <= 1,
                "round {round}: the count reads one, the search found {hits:?}"
            );
            remover.join().expect("remover panicked");
        });
        index.insert_shared(id, &scattered_vector(id, 8));
    }
}

/// While an operation that protects memory runs, a writer whose index holds
/// more replaced lists than the budget waits for it to end instead of
/// growing the retired memory further, and goes ahead once it has.
#[test]
fn a_writer_over_the_retired_budget_waits_for_running_operations() {
    let index = HnswIndex::new(make_config(VectorMetric::L2));
    for i in 0..50u64 {
        index.insert_shared(i, &scattered_vector(i, 4));
    }
    drain_epoch();
    index.set_retired_bytes_budget(1);
    let started = std::sync::Barrier::new(2);
    let hold = std::time::Duration::from_millis(300);
    std::thread::scope(|s| {
        let (index, started) = (&index, &started);
        s.spawn(move || {
            let _op = index.begin_operation();
            started.wait();
            std::thread::sleep(hold);
        });
        started.wait();
        let began = std::time::Instant::now();
        // The first update retires lists the running operation may read; the
        // next ones find the budget exceeded and wait.
        for round in 0..3u64 {
            index.insert_shared(7, &scattered_vector(100 + round, 4));
        }
        assert!(
            began.elapsed() >= hold / 2,
            "the writer did not wait: {:?}",
            began.elapsed()
        );
    });
    let stats = index.publication_stats();
    assert!(stats.admission_waits >= 1, "{stats:?}");
    assert!(stats.admission_wait >= hold / 2, "{stats:?}");
    assert_eq!(index.search(&scattered_vector(102, 4), 1)[0].id, 7);
}

/// Replaced lists are counted while a pin can still read them and freed
/// once none can; a running operation shows as the oldest protection.
#[test]
fn publication_stats_follow_retired_lists_and_running_operations() {
    let index = HnswIndex::new(make_config(VectorMetric::L2));
    for i in 0..50u64 {
        index.insert_shared(i, &scattered_vector(i, 4));
    }
    drain_epoch();
    assert_eq!(index.publication_stats().retired_bytes, 0);
    let op = index.begin_operation();
    for i in 0..10u64 {
        index.insert_shared(i, &scattered_vector(i + 500, 4));
    }
    std::thread::sleep(std::time::Duration::from_millis(20));
    let held = index.publication_stats();
    assert!(held.retired_bytes > 0, "{held:?}");
    assert!(held.retired_lists > 0, "{held:?}");
    assert!(
        held.oldest_operation >= std::time::Duration::from_millis(20),
        "{held:?}"
    );
    drop(op);
    drain_epoch();
    let freed = index.publication_stats();
    assert_eq!(freed.retired_bytes, 0, "{freed:?}");
    assert_eq!(freed.retired_lists, 0, "{freed:?}");
    assert_eq!(
        freed.oldest_operation,
        std::time::Duration::ZERO,
        "{freed:?}"
    );
}

/// Every inserted node ends live, through the single, batched and parallel
/// paths, and a reorder keeps that state with the node.
#[test]
fn inserted_nodes_are_live_and_reorder_keeps_them_live() {
    let mut index = HnswIndex::new(make_config(VectorMetric::L2));
    for i in 0..10u64 {
        index.insert(i, vec![i as f32, 1.0, 0.5]);
    }
    let batch: Vec<(u64, Vec<f32>)> = (10..200u64)
        .map(|i| (i, vec![(i as f32).sin(), (i as f32).cos(), 0.5]))
        .collect();
    index.insert_batch(batch);
    let all_live = |index: &HnswIndex| {
        (0..index.node_len()).all(|i| index.nodes().state(i) == data_level0::NodeState::Live)
    };
    assert!(all_live(&index));
    index.reorder_for_cache_locality();
    assert!(all_live(&index));
    assert_eq!(
        index
            .search_with_mode(&[0.0, 1.0, 0.5], 5, SearchMode::Exact)
            .len(),
        5
    );
}

/// A vector whose dimension is zero or differs from the index's is refused,
/// with the node and the reason, on every insert path, and leaves the index
/// as it was: a batch holding one is refused whole, before any item is
/// applied. The distance kernels assume equal lengths, so it must never reach
/// planning or the store.
#[test]
fn inserts_with_the_wrong_dimension_are_rejected() {
    let mut index = super::HnswIndex::new(make_config(VectorMetric::L2));
    assert_eq!(
        index.insert(1, vec![]),
        Err(AdmissionError {
            id: 1,
            error: VectorDomainError::Empty
        })
    );
    assert!(index.is_empty(), "a zero-length vector sized the index");
    for i in 0..10u64 {
        index.insert(i, vec![i as f32, 1.0, 2.0]).unwrap();
    }
    let mismatch = |id, left| AdmissionError {
        id,
        error: VectorDomainError::DimensionMismatch { left, right: 3 },
    };
    assert_eq!(index.insert(100, vec![1.0, 2.0]), Err(mismatch(100, 2)));
    assert_eq!(
        index.insert(3, vec![1.0, 2.0, 3.0, 4.0]),
        Err(mismatch(3, 4))
    );
    let batch: Vec<(u64, Vec<f32>)> = (200..300u64)
        .map(|i| {
            let dim = if i % 3 == 0 { 5 } else { 3 };
            (i, vec![i as f32; dim])
        })
        .collect();
    // 201 is the first multiple of three in the batch: its first wrong item.
    assert_eq!(index.insert_batch(batch.clone()), Err(mismatch(201, 5)));
    assert_eq!(index.bulk_build(batch), Err(mismatch(201, 5)));
    assert!(!index.contains(100));
    assert_eq!(
        index.get_vector(index.idx_for_id_for_test(3).expect("kept")),
        Some([3.0, 1.0, 2.0].as_slice()),
        "a wrong-dimension update replaced node 3"
    );
    assert!((200..300u64).all(|i| !index.contains(i)));
    assert_eq!(index.len(), 10);
}

/// A vector the metric cannot evaluate is refused on every insert path, and a
/// query it cannot evaluate is refused on every search path, with the reason:
/// non-finite components for every metric, a zero vector under cosine. A zero
/// vector is an ordinary vector for the other metrics.
#[test]
fn vectors_the_metric_cannot_evaluate_are_refused() {
    let mut cosine = super::HnswIndex::new(make_config(VectorMetric::Cosine));
    let zero = AdmissionError {
        id: 7,
        error: VectorDomainError::ZeroVectorCosine,
    };
    assert_eq!(cosine.insert(7, vec![0.0, -0.0]), Err(zero));
    assert_eq!(cosine.insert_shared(7, &[0.0, 0.0]), Err(zero));
    assert_eq!(
        cosine.insert_batch(vec![(1, vec![1.0, 0.0]), (7, vec![0.0, 0.0])]),
        Err(zero)
    );
    assert!(cosine.is_empty(), "a refused batch applied an item");
    cosine.insert(1, vec![1.0, 0.0]).unwrap();
    // A tiny nonzero vector has a direction.
    cosine.insert(2, vec![1e-40, 0.0]).unwrap();
    let domain = |e| VectorSearchError::Domain(e);
    assert_eq!(
        cosine.search(&[0.0, 0.0], 1),
        Err(domain(VectorDomainError::ZeroVectorCosine))
    );
    assert_eq!(
        cosine.search_with_mode(&[0.0, 0.0], 1, SearchMode::Exact),
        Err(domain(VectorDomainError::ZeroVectorCosine))
    );
    assert_eq!(
        cosine
            .search_with_visibility(&[0.0, 0.0], 1, 1.2, 3, |_| true)
            .map(|(found, _)| found),
        Err(domain(VectorDomainError::ZeroVectorCosine))
    );
    assert_eq!(
        cosine.search(&[1.0, f32::NAN], 1),
        Err(domain(VectorDomainError::NonFinite { index: 1 }))
    );
    assert_eq!(
        cosine.search(&[1.0, 0.0, 0.0], 1),
        Err(domain(VectorDomainError::DimensionMismatch {
            left: 3,
            right: 2
        }))
    );
    let ids: Vec<u64> = cosine
        .search(&[1.0, 0.0], 2)
        .unwrap()
        .iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(ids, [1, 2]);

    for metric in [VectorMetric::L2, VectorMetric::DotProduct, VectorMetric::L1] {
        let mut index = super::HnswIndex::new(make_config(metric));
        assert_eq!(
            index.insert(1, vec![f32::INFINITY, 0.0]),
            Err(AdmissionError {
                id: 1,
                error: VectorDomainError::NonFinite { index: 0 }
            })
        );
        index.insert(1, vec![0.0, 0.0]).unwrap();
        assert_eq!(index.search(&[0.0, 0.0], 1).unwrap()[0].id, 1);
    }
}

/// Updating a node's vector refreshes the norms the cosine kernels read: a
/// query along the new vector scores it at distance 0, not at a value
/// computed from the old vector's norm.
#[test]
fn update_refreshes_the_cosine_norms() {
    let mut index = HnswIndex::new(make_config(VectorMetric::Cosine));
    for i in 0..20u64 {
        let v: Vec<f32> = (0..4).map(|d| ((i * 5 + d) as f32 * 0.3).sin()).collect();
        index.insert(i, v);
    }
    index.insert(3, vec![1.0, 0.0, 0.0, 0.0]);
    // Same id, a longer vector in a new direction.
    index.insert(3, vec![5.0, 5.0, 0.0, 0.0]);
    let hit = index
        .search(&[1.0, 1.0, 0.0, 0.0], 1)
        .into_iter()
        .next()
        .expect("a result");
    assert_eq!(hit.id, 3);
    assert!(
        hit.score.abs() < 1e-5,
        "cosine distance to an identical direction is {}, expected 0",
        hit.score
    );
}

/// Cosine search over large finite vectors ranks by direction through the
/// cached inverse norms and the build's both-norms path: the vector along
/// the query is found at distance 0, and no distance comes back NaN or
/// below 0 (`1 - inf * 0` would rank a far vector first).
#[test]
fn cosine_search_over_large_finite_vectors_ranks_by_direction() {
    let mut index = HnswIndex::new(make_config(VectorMetric::Cosine));
    for i in 0..20u64 {
        let v: Vec<f32> = (0..4)
            .map(|d| ((i * 5 + d) as f32 * 0.3).sin() * 1e20)
            .collect();
        index.insert(i, v);
    }
    index.insert(100, vec![1e20, 1e20, 0.0, 0.0]);
    let results = index.search(&[3e20, 3e20, 0.0, 0.0], 5);
    assert_eq!(results[0].id, 100);
    assert!(results[0].score.abs() < 1e-5, "{}", results[0].score);
    for r in &results {
        assert!(
            (0.0..=2.0 + 1e-5).contains(&r.score),
            "distance {} of {} outside [0, 2]",
            r.score,
            r.id
        );
    }
}

/// A prune that races a back-edge append must keep the append: the prune
/// recomputes against the list that won instead of overwriting it with a
/// result computed from an older snapshot. `max_conn` here exceeds every
/// list the test builds, so a prune never legitimately drops an id and any
/// missing accepted append is a lost update.
#[test]
fn prune_racing_an_append_keeps_the_append() {
    let mut cfg = make_config(VectorMetric::L2);
    cfg.m = 32;
    cfg.m_max0 = 64;
    cfg.max_elements = 128;
    let mut index = HnswIndex::new(cfg);
    for i in 0..100u64 {
        let v: Vec<f32> = (0..8).map(|d| ((i * 13 + d) as f32 * 0.07).sin()).collect();
        index.insert(i, v);
    }
    for round in 0..200 {
        index.layer_update(0, 0, |_| Some(Vec::new()));
        let appending = std::sync::atomic::AtomicBool::new(true);
        let accepted = std::thread::scope(|s| {
            s.spawn(|| {
                while appending.load(std::sync::atomic::Ordering::Acquire) {
                    index.prune_connections(0, 0, 64);
                }
            });
            let mut accepted = Vec::new();
            for id in 1..=60u64 {
                if index.layer_cas_append(0, 0, id) {
                    accepted.push(id);
                }
                std::thread::yield_now();
            }
            appending.store(false, std::sync::atomic::Ordering::Release);
            accepted
        });
        let list = index.layer_snapshot(0, 0);
        for id in accepted {
            assert!(
                list.contains(&id),
                "round {round}: accepted append {id} lost by a racing prune; list {list:?}"
            );
        }
    }
}

/// Removing edges while another writer appends keeps every accepted append
/// and drops exactly the removed ids. The removal is the repair a retired
/// node's neighbours run, with nothing to refill from.
#[test]
fn removal_racing_an_append_keeps_the_append() {
    let mut cfg = make_config(VectorMetric::L2);
    cfg.m = 32;
    cfg.m_max0 = 64;
    cfg.max_elements = 128;
    let mut index = HnswIndex::new(cfg);
    for i in 0..100u64 {
        let v: Vec<f32> = (0..8).map(|d| ((i * 13 + d) as f32 * 0.07).sin()).collect();
        index.insert(i, v);
    }
    let doomed: Vec<u64> = (61..=90).collect();
    for round in 0..200 {
        index.layer_update(0, 0, |_| Some(doomed.clone()));
        let accepted = std::thread::scope(|s| {
            s.spawn(|| {
                for &id in &doomed {
                    index.layer_update(0, 0, |current| {
                        index.repaired_list(0, 0, current, &|nb| nb == id, &[])
                    });
                    std::thread::yield_now();
                }
            });
            let mut accepted = Vec::new();
            for id in 1..=34u64 {
                if index.layer_cas_append(0, 0, id) {
                    accepted.push(id);
                }
                std::thread::yield_now();
            }
            accepted
        });
        let list = index.layer_snapshot(0, 0);
        for id in &accepted {
            assert!(
                list.contains(id),
                "round {round}: accepted append {id} lost by a racing removal; list {list:?}"
            );
        }
        for id in &doomed {
            assert!(
                !list.contains(id),
                "round {round}: removed id {id} is back; list {list:?}"
            );
        }
        assert_eq!(list.len(), accepted.len(), "round {round}: list {list:?}");
    }
}

// Regression: HNSW search must return min(k, n) results regardless of
// ef_search. Standard HNSW invariant — the layer-0 beam must be at
// least `k` wide, otherwise low-ef configurations both truncate the
// result set AND cripple recall (visited pool too small to reach the
// true k-NN). Reproduces the recall gap vs Qdrant at ef_search ≪ k
// on the sift-128 ann-benchmarks run.
#[test]
fn search_returns_k_results_when_ef_below_k() {
    let mut cfg = make_config(VectorMetric::L2);
    cfg.ef_search = 4; // deliberately well below k
    let mut index = HnswIndex::new(cfg);
    for i in 0..50u64 {
        let v = vec![(i as f32) * 0.1, ((i % 7) as f32) * 0.2, (i as f32).sin()];
        index.insert(i, v);
    }

    // k=10, ef_search=4 — must still return 10 results.
    let results = index.search(&[0.0, 0.0, 0.0], 10);
    assert_eq!(results.len(), 10, "ef_search<k must not truncate top-k");
}

// Regression: recall@k must stay above a sane floor even when caller
// passes a small ef_search. Standard HNSW guarantees this by enforcing
// an effective beam of max(ef_search, k); without that floor recall
// collapses (observed on sift-128: 0.86 vs Qdrant 0.95 at ef=16,k=10).
#[test]
fn search_low_ef_recall_floor() {
    let mut cfg = make_config(VectorMetric::L2);
    cfg.m = 8;
    cfg.m_max0 = 16;
    cfg.ef_construction = 64;
    cfg.ef_search = 4; // tiny — exercise the floor
    let mut index = HnswIndex::new(cfg);

    let n = 300usize;
    let dim = 8;
    let points: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| (((i * 31 + d) as f32) * 0.137).sin())
                .collect()
        })
        .collect();
    for (i, p) in points.iter().enumerate() {
        index.insert(i as u64, p.clone());
    }

    let k = 10;
    let mut hits = 0usize;
    let mut total = 0usize;
    for qi in (0..n).step_by(7) {
        let q = &points[qi];
        // Exact top-k by brute force.
        let mut exact: Vec<(usize, f32)> = points
            .iter()
            .enumerate()
            .map(|(j, v)| {
                let d = v
                    .iter()
                    .zip(q.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>();
                (j, d)
            })
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let truth: HashSet<u64> = exact.iter().take(k).map(|(j, _)| *j as u64).collect();

        let got = index.search(q, k);
        for r in &got {
            if truth.contains(&r.id) {
                hits += 1;
            }
        }
        total += k;
    }
    let recall = hits as f64 / total as f64;
    assert!(
        recall >= 0.80,
        "recall@k={k} at ef_search<k was {recall:.3}, expected ≥ 0.80",
    );
}

/// Measurement probe, not a check: self-recall of 40 concurrent builds of
/// each kind on a wide pool, as min / mean / max.
#[test]
#[ignore = "measurement probe"]
fn probe_wide_pool_recall_distribution() {
    let n = 600usize;
    let dim = 20usize;
    let items: Vec<(u64, Vec<f32>)> = (0..n)
        .map(|i| {
            let v = (0..dim)
                .map(|d| {
                    let seed = (i.wrapping_mul(2_654_435_761).wrapping_add(d * 6_700_417)) as u32;
                    let bits = (seed ^ (seed >> 13)) & 0x00FF_FFFF;
                    (bits as f32 / 16_777_216.0) - 0.5
                })
                .collect();
            (i as u64, v)
        })
        .collect();
    let cfg = || HnswConfig {
        ef_search: 64,
        metric: VectorMetric::L2,
        max_dimensions: 65_536,
        ..Default::default()
    };
    let self_recall = |index: &HnswIndex| {
        items
            .iter()
            .filter(|(id, v)| index.search(v, 1).first().is_some_and(|r| r.id == *id))
            .count()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(128)
        .build()
        .unwrap();
    for bulk in [false, true] {
        let mut got = Vec::new();
        for _ in 0..40 {
            let mut index = HnswIndex::new(cfg());
            pool.install(|| {
                if bulk {
                    index.bulk_build(items.clone());
                } else {
                    index.insert_batch(items.clone());
                }
            });
            got.push(self_recall(&index));
        }
        let min = got.iter().min().unwrap();
        let max = got.iter().max().unwrap();
        let mean = got.iter().sum::<usize>() as f64 / got.len() as f64;
        eprintln!("PROBE bulk={bulk} min={min} mean={mean:.1} max={max}");
    }
}

/// recall@10 of `index` over `queries`, against its exact top-10.
fn recall_at_10(index: &HnswIndex, queries: &[Vec<f32>]) -> f64 {
    let mut hits = 0usize;
    for q in queries {
        let exact: HashSet<u64> = index
            .search_with_mode(q, 10, SearchMode::Exact)
            .iter()
            .map(|r| r.id)
            .collect();
        hits += index
            .search(q, 10)
            .iter()
            .filter(|r| exact.contains(&r.id))
            .count();
    }
    hits as f64 / (queries.len() * 10) as f64
}

/// The traversal estimator keeps its recall over the whole range of finite
/// f32 inputs. The same points scaled by an exact power of two have the same
/// exact neighbours under every metric; an estimator that overflows (huge
/// components) or underflows (tiny ones) to one value for every candidate
/// walks the graph blind and loses them, which no exact rerank of the
/// candidates it kept can recover.
#[test]
fn traversal_recall_holds_across_the_f32_range() {
    let base = uniform_points(2000, 16);
    let queries: Vec<Vec<f32>> = uniform_points(2050, 16)[2000..]
        .iter()
        .map(|(_, v)| v.clone())
        .collect();
    let scaled = |v: &[f32], s: f32| v.iter().map(|x| x * s).collect::<Vec<f32>>();
    let mut lost = Vec::new();
    for metric in [
        VectorMetric::L2,
        VectorMetric::DotProduct,
        VectorMetric::L1,
        VectorMetric::Cosine,
    ] {
        let recall_at = |s: f32| {
            let mut index = HnswIndex::new(HnswConfig {
                ef_construction: 100,
                ef_search: 64,
                metric,
                max_dimensions: 65_536,
                ..Default::default()
            });
            for (id, v) in &base {
                index.insert(*id, scaled(v, s));
            }
            let q: Vec<Vec<f32>> = queries.iter().map(|v| scaled(v, s)).collect();
            recall_at_10(&index, &q)
        };
        let reference = recall_at(1.0);
        for s in [2f32.powi(66), 2f32.powi(-84)] {
            let r = recall_at(s);
            eprintln!("{metric:?} scale {s:e}: recall@10 {r:.3} (unscaled {reference:.3})");
            if r < reference - 0.02 {
                lost.push(format!(
                    "{metric:?} at scale {s:e}: {r:.3} vs {reference:.3}"
                ));
            }
        }
    }
    assert!(lost.is_empty(), "recall lost: {lost:#?}");
}

/// The quantized estimators keep their recall over the f32 range too: SQ8
/// and RaBitQ codes calibrated on the same points scaled by an exact power of
/// two serve the same neighbours.
#[test]
fn quantized_traversal_recall_holds_across_the_f32_range() {
    let base = uniform_points(1200, 16);
    let queries: Vec<Vec<f32>> = uniform_points(1240, 16)[1200..]
        .iter()
        .map(|(_, v)| v.clone())
        .collect();
    let scaled = |v: &[f32], s: f32| v.iter().map(|x| x * s).collect::<Vec<f32>>();
    let mut lost = Vec::new();
    for (metric, quantization) in [
        (VectorMetric::L2, QuantizationCodec::Sq8),
        (VectorMetric::Cosine, QuantizationCodec::Sq8),
        (VectorMetric::Cosine, QuantizationCodec::RaBitQ { bits: 1 }),
        (VectorMetric::Cosine, QuantizationCodec::RaBitQ { bits: 4 }),
    ] {
        let recall_at = |s: f32| {
            let mut index = HnswIndex::new(HnswConfig {
                ef_construction: 100,
                ef_search: 128,
                metric,
                quantization,
                calibration_threshold: 400,
                rerank_candidates: 100,
                max_dimensions: 65_536,
                ..Default::default()
            });
            for (id, v) in &base {
                index.insert(*id, scaled(v, s));
            }
            let q: Vec<Vec<f32>> = queries.iter().map(|v| scaled(v, s)).collect();
            recall_at_10(&index, &q)
        };
        let reference = recall_at(1.0);
        for s in [2f32.powi(66), 2f32.powi(-84)] {
            let r = recall_at(s);
            eprintln!(
                "{metric:?} {quantization:?} scale {s:e}: recall@10 {r:.3} (unscaled {reference:.3})"
            );
            if r < reference - 0.02 {
                lost.push(format!(
                    "{metric:?} {quantization:?} at {s:e}: {r:.3} vs {reference:.3}"
                ));
            }
        }
    }
    assert!(lost.is_empty(), "recall lost: {lost:#?}");
}

/// The range-safe key orders as the binary64 distance does, over the whole
/// range of signs and magnitudes, and stands for it to 20 bits.
#[test]
fn the_range_safe_key_orders_as_the_distance() {
    // Down to 2^-298, the smallest nonzero square of an f32 difference; the
    // key's truncation merges only binary64 subnormals far below it.
    let values = [
        -1e300, -3.5e40, -1.0, -1e-30, -1e-89, 0.0, 1e-89, 1e-45, 0.5, 1.0, 2.0, 3.4e38, 1e60,
        1e300,
    ];
    for pair in values.windows(2) {
        assert!(
            ordered_key(pair[0]) < ordered_key(pair[1]),
            "{} vs {}",
            pair[0],
            pair[1]
        );
    }
    assert_eq!(ordered_key(-0.0).to_bits(), 0.0f32.to_bits());
    for &d in &values {
        assert!(ordered_key(d).is_finite(), "{d}");
        let back = key_distance(ordered_key(d));
        assert!(
            (back - d).abs() <= d.abs() * 2f64.powi(-20),
            "{d} -> {back}"
        );
    }
}

/// The fast estimators' domain: zero and in-range vectors are in it,
/// vectors whose largest component is too large or too small are not, the
/// f32 cosine and SQ8 codes take any finite vector, RaBitQ codes do not.
#[test]
fn the_fast_estimator_domain_is_bounded_by_the_largest_component() {
    let none = QuantizationCodec::None;
    let fast = |v: &[f32], metric| in_fast_domain(v, metric, &none);
    assert!(fast(&[0.0, 0.0], VectorMetric::L2));
    assert!(
        fast(&[1e-10, 3.0], VectorMetric::L2),
        "a tiny component beside a large one"
    );
    assert!(!fast(&[1e20, 0.0], VectorMetric::L2));
    assert!(!fast(&[1e-20, 0.0], VectorMetric::DotProduct));
    assert!(fast(&[1e20, 0.0], VectorMetric::L1));
    assert!(!fast(&[1e31, 0.0], VectorMetric::L1));
    assert!(fast(&[1e38, 1e-38], VectorMetric::Cosine));
    assert!(!fast(&vec![1.0; MAX_FAST_DIMS + 1], VectorMetric::L2));
    let sq8 = QuantizationCodec::Sq8;
    assert!(in_fast_domain(&[1e38, 0.0], VectorMetric::Cosine, &sq8));
    let rabitq = QuantizationCodec::RaBitQ { bits: 1 };
    assert!(in_fast_domain(&[1.0, 0.0], VectorMetric::Cosine, &rabitq));
    assert!(!in_fast_domain(&[1e20, 0.0], VectorMetric::Cosine, &rabitq));
}

/// The estimator's error band drops only candidates that cannot be among the
/// k best by exact score: over the whole population as candidates, ranking
/// what the band keeps returns the same results, scores and order as ranking
/// every candidate. The data pushes the bound: near-duplicate vectors one
/// ulp apart, exact duplicates (ties broken by id), several magnitudes
/// inside the fast domain and dimensions from 3 to 515.
#[test]
fn the_error_band_keeps_every_possible_exact_result() {
    let mut pruned_somewhere = false;
    for metric in [
        VectorMetric::L2,
        VectorMetric::L1,
        VectorMetric::DotProduct,
        VectorMetric::Cosine,
    ] {
        for (dim, scale) in [
            (3usize, 1.0f32),
            (17, 2f32.powi(-30)),
            (128, 1.0),
            (515, 2f32.powi(30)),
        ] {
            let base = uniform_points(60, dim);
            let mut items: Vec<(u64, Vec<f32>)> = Vec::new();
            for (id, v) in &base {
                let v: Vec<f32> = v.iter().map(|x| (x + 0.5) * scale).collect();
                // One ulp away in a single component, and an exact duplicate.
                let mut near = v.clone();
                near[0] = f32::from_bits(near[0].to_bits() + 1);
                items.push((id * 3, v.clone()));
                items.push((id * 3 + 1, near));
                items.push((id * 3 + 2, v));
            }
            let mut index = HnswIndex::new(HnswConfig {
                metric,
                max_dimensions: 65_536,
                max_elements: items.len() as u32,
                ..Default::default()
            });
            index.insert_batch(items);
            for (q, query) in uniform_points(8, dim).into_iter().enumerate() {
                let query: Vec<f32> = query.1.iter().map(|x| (x + 0.25) * scale).collect();
                let exact = index.exact_query(&query).unwrap();
                let qctx = QueryCtx::new(
                    &query,
                    metric,
                    index.rabitq_params.as_ref(),
                    &index.config.quantization,
                    index.wide_for(&query),
                );
                let all: Vec<Candidate> = (0..index.node_len())
                    .map(|idx| Candidate {
                        distance: index.compute_distance(&qctx, idx),
                        idx: idx as u32,
                    })
                    .collect();
                for k in [1usize, 10, 50] {
                    let mut full = index.rank_exactly(&exact, all.clone()).unwrap();
                    full.truncate(k);
                    let mut kept = all.clone();
                    index.exact_band(&qctx, &mut kept, k);
                    pruned_somewhere |= kept.len() < all.len();
                    let mut banded = index.rank_exactly(&exact, kept).unwrap();
                    banded.truncate(k);
                    let bits = |r: &[SearchResult]| -> Vec<(u64, u64)> {
                        r.iter().map(|s| (s.id, s.score.to_bits())).collect()
                    };
                    assert_eq!(
                        bits(&banded),
                        bits(&full),
                        "{metric:?} d{dim} scale {scale} query {q} k {k}"
                    );
                }
            }
        }
    }
    assert!(pruned_somewhere, "the band never dropped a candidate");
}

/// The first vector outside the fast domain switches the index to the
/// range-safe estimator for good, and the index then serves both scales; a
/// query outside the domain is evaluated range-safely on an index that is
/// not.
#[test]
fn an_out_of_domain_vector_switches_the_index_to_the_range_safe_estimator() {
    let mut index = HnswIndex::new(HnswConfig {
        metric: VectorMetric::L2,
        ef_search: 64,
        max_dimensions: 65_536,
        ..Default::default()
    });
    let wide = |index: &HnswIndex| {
        index
            .wide_estimator
            .load(core::sync::atomic::Ordering::Acquire)
    };
    for (id, v) in uniform_points(200, 8) {
        index.insert(id, v);
    }
    assert!(!wide(&index));
    // The out-of-domain query alone does not switch the index.
    let huge: Vec<f32> = uniform_points(201, 8)[200]
        .1
        .iter()
        .map(|x| x * 2f32.powi(66))
        .collect();
    assert!(index.wide_for(&huge));
    assert!(!wide(&index));
    index.insert(500, huge.clone());
    assert!(wide(&index));
    assert_eq!(index.search(&huge, 1)[0].id, 500);
    let near = uniform_points(1, 8)[0].1.clone();
    assert_eq!(index.search(&near, 1)[0].id, 0);
}
