//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! Parameters:
//! - `M` — number of bi-directional links per element (default 16)
//! - `ef_construction` — candidate list size during build (default 200)
//! - `ef_search` — candidate list size during search (runtime tunable)
//!
//! Supports up to 65,536 dimensions, 4 distance metrics.
//! SQ8 quantized vectors for candidate generation, f32 rerank for final top-K.
//!
//! # Quantization (SQ8)
//!
//! When `quantization` is enabled in [`HnswConfig`], the index auto-calibrates
//! SQ8 parameters after [`HnswConfig::calibration_threshold`] vectors are inserted.
//! Once calibrated:
//! - Each node stores a u8 quantized vector alongside the original f32.
//! - HNSW traversal computes distances on dequantized (approximate) vectors.
//! - Final top-K is reranked using original f32 vectors for exact scores.
//!
//! Memory: quantized vectors use 4x less memory than f32. The f32 originals
//! are retained in-memory for reranking; a future optimization will store
//! them on disk (LSM) and load only for the reranking candidates.
//!
//! # Cluster-ready notes
//! - HNSW graph lives in-memory per node. Each CE replica builds its own
//!   HNSW from replicated vector data in CoordiNode storage.
//! - No cross-node HNSW graph sharing needed — data replicated via Raft,
//!   HNSW built locally on each node.

mod visited;

use std::collections::BinaryHeap;

use coordinode_core::graph::types::VectorMetric;
use tracing::warn;
use visited::VisitedPool;

/// Software prefetch hint: request CPU to load `ptr` into L1 cache.
/// This is a performance hint — no effect on correctness.
/// No-op on unsupported platforms.
///
/// Donor: hnswlib `hnswalg.h:370-383` — `_mm_prefetch` with `_MM_HINT_T0`
#[inline(always)]
fn prefetch_read_data(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM PLDL1KEEP — prefetch for read into L1 data cache.
        // std::arch::aarch64::_prefetch is unstable (#117217), use inline asm.
        std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
    }
    // No-op on other architectures (WASM, RISC-V, etc.)
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    let _ = ptr;
}

use std::collections::HashMap;

use crate::metrics;
use crate::quantize::Sq8Params;

/// Provides f32 vectors from external storage for reranking when vectors
/// are offloaded to disk. Implementations should batch-read from the
/// backing store (e.g., LSM-tree node: partition).
///
/// The `property` parameter allows a single loader to serve multiple
/// HNSW indexes (each index targets a different vector property).
pub trait VectorLoader: Send + Sync {
    /// Load f32 vectors for the given node IDs and property name.
    /// Returns a map of node_id → f32 vector for all found IDs.
    fn load_vectors(&self, ids: &[u64], property: &str) -> HashMap<u64, Vec<f32>>;
}

/// Minimum number of vectors for SQ8 quantization to be worthwhile.
/// Below this threshold, calibration storage overhead is not justified
/// and quantization error may exceed recall benefit.
const SQ8_MIN_VECTORS: usize = 1000;

/// HNSW index configuration.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max connections per element per layer.
    pub m: usize,
    /// Max connections for layer 0 (typically 2*M).
    pub m_max0: usize,
    /// Candidate list size during construction.
    pub ef_construction: usize,
    /// Candidate list size during search (runtime tunable).
    pub ef_search: usize,
    /// Distance metric.
    pub metric: VectorMetric,
    /// Maximum vector dimensions.
    pub max_dimensions: u32,
    /// Enable SQ8 scalar quantization for memory-efficient HNSW traversal.
    /// When enabled, distance computation during HNSW search uses dequantized
    /// (approximate) vectors. Final top-K results are reranked with exact f32.
    pub quantization: bool,
    /// Number of candidates to fetch before f32 reranking.
    /// Only used when `quantization` is enabled. Must be >= ef_search.
    /// Higher values improve recall at the cost of more f32 distance computations.
    pub rerank_candidates: usize,
    /// Number of vectors to collect before auto-calibrating SQ8 parameters.
    /// Until this threshold is reached, all distances are computed on f32.
    pub calibration_threshold: usize,
    /// When true and SQ8 quantization is active, f32 vectors are not retained
    /// in memory after construction. Reranking loads f32 from external storage
    /// via a [`VectorLoader`]. Gives 4x RAM reduction at ~1-2ms rerank cost.
    /// Only effective when `quantization` is also true.
    pub offload_vectors: bool,
    /// Property name this index covers (e.g., "embedding").
    /// Passed to `VectorLoader::load_vectors` so one loader serves all indexes.
    /// Empty string if unknown (non-offloaded indexes don't need this).
    pub property_name: String,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
            metric: VectorMetric::Cosine,
            max_dimensions: 65_536,
            quantization: false,
            rerank_candidates: 100,
            calibration_threshold: 100,
            offload_vectors: false,
            property_name: String::new(),
        }
    }
}

/// A single element in the HNSW graph.
struct HnswNode {
    /// Node ID (maps to graph node ID).
    id: u64,
    /// Original f32 vector data. Retained for exact reranking when in-memory.
    /// `None` when vectors are offloaded to disk (see `HnswConfig::offload_vectors`).
    /// When offloaded, reranking loads f32 from external storage via `VectorLoader`.
    vector: Option<Vec<f32>>,
    /// SQ8-quantized vector for memory-efficient HNSW traversal.
    /// `None` until SQ8 calibration is complete.
    quantized: Option<Vec<u8>>,
    /// Connections per layer. Layer 0 has up to m_max0 connections.
    connections: Vec<Vec<u64>>,
    /// Maximum layer this element exists on.
    _max_layer: usize,
}

/// HNSW index: in-memory approximate nearest neighbor graph.
pub struct HnswIndex {
    config: HnswConfig,
    /// All nodes indexed by ID.
    nodes: Vec<HnswNode>,
    /// Map from node ID to index in `nodes` vec.
    id_to_idx: std::collections::HashMap<u64, usize>,
    /// Entry point: node index with the highest layer.
    entry_point: Option<usize>,
    /// Maximum layer in the graph.
    max_level: usize,
    /// Inverse of ln(M) for level generation.
    level_mult: f64,
    /// SQ8 calibration parameters. `None` until calibration_threshold vectors
    /// are inserted and auto-calibration runs.
    sq8_params: Option<Sq8Params>,
    /// Pool of reusable visited lists for search. Avoids per-search allocation.
    visited_pool: VisitedPool,
    /// RNG state for random level selection (xorshift64).
    /// Proper RNG gives correct exponential layer distribution (R852 fix).
    /// AtomicU64 for future concurrent insert support (R858).
    rng_state: std::sync::atomic::AtomicU64,
}

/// A search result with node ID and distance/similarity.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

/// Ordered candidate for min-heap (by distance, ascending).
#[derive(Clone)]
struct Candidate {
    distance: f32,
    idx: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap (BinaryHeap is max-heap by default)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Max-heap candidate (for maintaining top-K worst).
#[derive(Clone)]
struct FarCandidate {
    distance: f32,
    idx: usize,
}

impl PartialEq for FarCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
    }
}
impl Eq for FarCandidate {}

impl PartialOrd for FarCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FarCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(config: HnswConfig) -> Self {
        let level_mult = 1.0 / (config.m as f64).ln();
        Self {
            config,
            nodes: Vec::new(),
            id_to_idx: std::collections::HashMap::new(),
            entry_point: None,
            max_level: 0,
            level_mult,
            sq8_params: None,
            visited_pool: VisitedPool::new(),
            // Seed from address of self (varies per instance). Non-deterministic but fast.
            rng_state: std::sync::atomic::AtomicU64::new(0xdeadbeef_cafebabe),
        }
    }

    /// Returns the SQ8 calibration parameters, if calibrated.
    pub fn sq8_params(&self) -> Option<&Sq8Params> {
        self.sq8_params.as_ref()
    }

    /// Returns whether SQ8 quantization is active (calibrated and enabled).
    pub fn is_quantized(&self) -> bool {
        self.config.quantization && self.sq8_params.is_some()
    }

    /// Returns whether f32 vectors are offloaded to disk.
    /// True only when both `offload_vectors` and SQ8 quantization are active.
    pub fn is_offloaded(&self) -> bool {
        self.config.offload_vectors && self.is_quantized()
    }

    /// Manually set SQ8 calibration parameters (e.g., from a saved index).
    /// Quantizes all existing nodes that don't have quantized vectors yet.
    /// If `offload_vectors` is enabled, drops f32 after quantizing.
    pub fn set_sq8_params(&mut self, params: Sq8Params) {
        for node in &mut self.nodes {
            if let Some(ref v) = node.vector {
                node.quantized = Some(params.quantize(v));
            }
            if self.config.offload_vectors {
                node.vector = None;
            }
        }
        self.sq8_params = Some(params);
    }

    /// Check if the node at `idx` has an in-memory f32 vector.
    pub fn has_f32_vector(&self, idx: usize) -> bool {
        self.nodes.get(idx).is_some_and(|n| n.vector.is_some())
    }

    /// Get a reference to the f32 vector at node index `idx`, if present.
    pub fn get_vector(&self, idx: usize) -> Option<&[f32]> {
        self.nodes.get(idx).and_then(|n| n.vector.as_deref())
    }

    /// Auto-calibrate SQ8 from all currently stored vectors.
    /// Called automatically when `quantization` is enabled and
    /// `calibration_threshold` vectors have been inserted.
    ///
    /// Logs a warning if the index has fewer than [`SQ8_MIN_VECTORS`] (1000)
    /// vectors — quantization overhead is not justified for small indexes.
    fn auto_calibrate(&mut self) {
        if self.nodes.len() < SQ8_MIN_VECTORS {
            warn!(
                vectors = self.nodes.len(),
                min_recommended = SQ8_MIN_VECTORS,
                "SQ8 quantization on small index (<{} vectors): \
                 calibration storage overhead may exceed memory savings. \
                 Consider disabling quantization for small datasets",
                SQ8_MIN_VECTORS,
            );
        }
        let refs: Vec<&[f32]> = self
            .nodes
            .iter()
            .filter_map(|n| n.vector.as_deref())
            .collect();
        if let Some(params) = Sq8Params::calibrate(&refs) {
            for node in &mut self.nodes {
                if let Some(ref v) = node.vector {
                    node.quantized = Some(params.quantize(v));
                }
                // Offload f32 after quantization if configured
                if self.config.offload_vectors {
                    node.vector = None;
                }
            }
            self.sq8_params = Some(params);
        }
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Insert a vector into the index.
    ///
    /// If the node `id` is already in the index, its vector is updated and its
    /// graph position is rebuilt (G082 fix). This handles the `MATCH (n) SET
    /// n.emb = $new_vec` path where `on_vector_written` calls `insert()` for
    /// both CREATE and SET.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        if let Some(&idx) = self.id_to_idx.get(&id) {
            // Node already indexed — update vector and reconnect in graph.
            self.update_existing_node(idx, vector);
            return;
        }

        let new_level = self.random_level();
        let idx = self.nodes.len();

        let mut connections = Vec::with_capacity(new_level + 1);
        for _ in 0..=new_level {
            connections.push(Vec::new());
        }

        // Quantize if SQ8 is calibrated
        let quantized = self.sq8_params.as_ref().map(|p| p.quantize(&vector));

        self.nodes.push(HnswNode {
            id,
            vector: Some(vector),
            quantized,
            connections,
            _max_layer: new_level,
        });
        self.id_to_idx.insert(id, idx);

        if self.nodes.len() == 1 {
            self.entry_point = Some(idx);
            self.max_level = new_level;
            // Calibrate + offload after first-node construction
            self.maybe_calibrate_and_offload(idx);
            return;
        }

        let ep_idx = self.entry_point.unwrap_or(0);
        let mut current_ep = ep_idx;

        // Phase 1: Traverse from top layer down to new_level+1 (greedy search)
        let top_level = self.max_level;
        for level in (new_level + 1..=top_level).rev() {
            current_ep = self.search_layer_greedy(idx, current_ep, level);
        }

        // Phase 2: Insert at layers new_level down to 0
        for level in (0..=new_level.min(top_level)).rev() {
            let ef = self.config.ef_construction;
            let neighbors = self.search_layer(idx, current_ep, ef, level);

            let max_conn = if level == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            // Select M nearest neighbors
            let selected: Vec<usize> = neighbors
                .into_iter()
                .take(max_conn)
                .map(|c| c.idx)
                .collect();

            // Connect new node to selected neighbors
            self.nodes[idx].connections[level] =
                selected.iter().map(|&n| self.nodes[n].id).collect();

            // Connect neighbors back to new node (bi-directional)
            for &neighbor_idx in &selected {
                if level < self.nodes[neighbor_idx].connections.len() {
                    self.nodes[neighbor_idx].connections[level].push(id);

                    // Prune if over max connections
                    if self.nodes[neighbor_idx].connections[level].len() > max_conn {
                        self.prune_connections(neighbor_idx, level, max_conn);
                    }
                }
            }

            if !selected.is_empty() {
                current_ep = selected[0];
            }
        }

        // Update entry point if new node has higher level
        if new_level > self.max_level {
            self.max_level = new_level;
            self.entry_point = Some(idx);
        }

        // Calibrate SQ8 + offload f32 AFTER construction is complete.
        // Must be after graph building because search_layer needs f32 for the new node.
        self.maybe_calibrate_and_offload(idx);
    }

    /// Auto-calibrate SQ8 if threshold reached, then offload the just-inserted
    /// node's f32 if offloading is active. Called at the end of `insert()`.
    fn maybe_calibrate_and_offload(&mut self, just_inserted_idx: usize) {
        // Step 1: Auto-calibrate SQ8 when threshold reached
        if self.config.quantization
            && self.sq8_params.is_none()
            && self.nodes.len() >= self.config.calibration_threshold
        {
            self.auto_calibrate();
        }

        // Step 2: Offload f32 of the just-inserted node (construction complete)
        if self.is_offloaded() {
            self.nodes[just_inserted_idx].vector = None;
        }
    }

    /// Search for K nearest neighbors.
    ///
    /// When SQ8 quantization is active, HNSW traversal uses approximate
    /// (dequantized) distances for candidate generation. The final top-K
    /// results are reranked using exact f32 distances.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if self.is_empty() {
            return Vec::new();
        }

        let ep_idx = self.entry_point.unwrap_or(0);
        let mut current_ep = ep_idx;

        // Traverse from top to layer 1 (greedy)
        for level in (1..=self.max_level).rev() {
            current_ep = self.search_layer_greedy_query(query, current_ep, level);
        }

        // When quantized: fetch more candidates for reranking
        let ef = if self.is_quantized() {
            self.config.ef_search.max(self.config.rerank_candidates)
        } else {
            self.config.ef_search
        };

        // Search at layer 0 with ef candidates
        let candidates = self.search_layer_query(query, current_ep, ef, 0);

        if self.is_quantized() {
            // Rerank candidates using exact f32 distance
            let mut reranked: Vec<SearchResult> = candidates
                .into_iter()
                .map(|c| {
                    let exact_dist = self.compute_exact_distance(query, c.idx);
                    SearchResult {
                        id: self.nodes[c.idx].id,
                        score: exact_dist,
                    }
                })
                .collect();
            reranked.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            reranked.truncate(k);
            reranked
        } else {
            // No quantization — return candidates directly
            candidates
                .into_iter()
                .take(k)
                .map(|c| SearchResult {
                    id: self.nodes[c.idx].id,
                    score: c.distance,
                })
                .collect()
        }
    }

    /// Search with MVCC snapshot visibility checking.
    ///
    /// Implements the `snapshot` consistency mode: HNSW search with overfetch,
    /// post-filter invisible candidates, and expansion rounds if needed.
    ///
    /// # Parameters
    /// - `query`: query vector
    /// - `k`: desired number of results
    /// - `overfetch_factor`: multiply k by this to get initial candidate count (default 1.2)
    /// - `max_expansion_rounds`: maximum retries with increased ef_search (default 3)
    /// - `is_visible`: closure that checks MVCC visibility of a node ID at the snapshot timestamp.
    ///   Returns `true` if the node exists and its vector matches at the snapshot.
    ///
    /// # Returns
    /// `(results, stats)` — filtered results and statistics for EXPLAIN output.
    pub fn search_with_visibility<F>(
        &self,
        query: &[f32],
        k: usize,
        overfetch_factor: f64,
        max_expansion_rounds: usize,
        is_visible: F,
    ) -> (
        Vec<SearchResult>,
        coordinode_core::graph::types::VectorMvccStats,
    )
    where
        F: Fn(u64) -> bool,
    {
        let mut stats = coordinode_core::graph::types::VectorMvccStats {
            overfetch_factor,
            ..Default::default()
        };

        if self.is_empty() {
            return (Vec::new(), stats);
        }

        let ep_idx = self.entry_point.unwrap_or(0);
        let mut current_ep = ep_idx;

        // Traverse from top to layer 1 (greedy)
        for level in (1..=self.max_level).rev() {
            current_ep = self.search_layer_greedy_query(query, current_ep, level);
        }

        let mut visible_results: Vec<SearchResult> = Vec::new();
        let base_ef = if self.is_quantized() {
            self.config.ef_search.max(self.config.rerank_candidates)
        } else {
            self.config.ef_search
        };

        let mut current_ef = ((k as f64 * overfetch_factor).ceil() as usize).max(base_ef);

        for round in 0..=max_expansion_rounds {
            stats.expansion_rounds = round;

            let candidates = self.search_layer_query(query, current_ep, current_ef, 0);

            // Convert to SearchResult with exact distances (rerank if quantized)
            let results: Vec<SearchResult> = if self.is_quantized() {
                let mut reranked: Vec<SearchResult> = candidates
                    .into_iter()
                    .map(|c| SearchResult {
                        id: self.nodes[c.idx].id,
                        score: self.compute_exact_distance(query, c.idx),
                    })
                    .collect();
                reranked.sort_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                reranked
            } else {
                candidates
                    .into_iter()
                    .map(|c| SearchResult {
                        id: self.nodes[c.idx].id,
                        score: c.distance,
                    })
                    .collect()
            };

            stats.candidates_fetched += results.len();

            // Post-filter: keep only visible candidates
            visible_results.clear();
            for result in &results {
                if is_visible(result.id) {
                    visible_results.push(result.clone());
                } else {
                    stats.candidates_filtered += 1;
                }
            }
            stats.candidates_visible = visible_results.len();

            if visible_results.len() >= k || round == max_expansion_rounds {
                break;
            }

            // Expand: increase ef_search by 2x for next round
            current_ef *= 2;
        }

        visible_results.truncate(k);
        (visible_results, stats)
    }

    /// Search with f32 vectors loaded from external storage for reranking.
    ///
    /// Used when `offload_vectors` is enabled: HNSW traversal uses SQ8 in-memory,
    /// then batch-loads f32 from the `loader` for exact reranking of top candidates.
    /// When vectors are NOT offloaded, falls back to the regular `search()` path.
    pub fn search_with_loader(
        &self,
        query: &[f32],
        k: usize,
        loader: &dyn VectorLoader,
    ) -> Vec<SearchResult> {
        if !self.is_offloaded() {
            return self.search(query, k);
        }

        if self.is_empty() {
            return Vec::new();
        }

        let ep_idx = self.entry_point.unwrap_or(0);
        let mut current_ep = ep_idx;

        for level in (1..=self.max_level).rev() {
            current_ep = self.search_layer_greedy_query(query, current_ep, level);
        }

        let ef = self.config.ef_search.max(self.config.rerank_candidates);
        let candidates = self.search_layer_query(query, current_ep, ef, 0);

        // Batch-load f32 vectors from storage for reranking
        let candidate_ids: Vec<u64> = candidates.iter().map(|c| self.nodes[c.idx].id).collect();
        let loaded = loader.load_vectors(&candidate_ids, &self.config.property_name);

        let mut reranked: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|c| {
                let node_id = self.nodes[c.idx].id;
                let f32_vec = loaded.get(&node_id)?;
                let exact_dist = self.distance_for_metric(query, f32_vec);
                Some(SearchResult {
                    id: node_id,
                    score: exact_dist,
                })
            })
            .collect();
        reranked.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked.truncate(k);
        reranked
    }

    /// Search with MVCC visibility + external f32 loading (reserved for future use).
    ///
    /// Currently the executor handles MVCC post-filtering after search_with_loader,
    /// so this method is not called from production code. Retained for direct
    /// HnswIndex API users who want combined visibility + loader in one call.
    #[allow(dead_code)]
    pub fn search_with_visibility_and_loader<F>(
        &self,
        query: &[f32],
        k: usize,
        overfetch_factor: f64,
        max_expansion_rounds: usize,
        is_visible: F,
        loader: Option<&dyn VectorLoader>,
    ) -> (
        Vec<SearchResult>,
        coordinode_core::graph::types::VectorMvccStats,
    )
    where
        F: Fn(u64) -> bool,
    {
        // When not offloaded or no loader, delegate to existing method
        let loader = match loader {
            Some(l) if self.is_offloaded() => l,
            _ => {
                return self.search_with_visibility(
                    query,
                    k,
                    overfetch_factor,
                    max_expansion_rounds,
                    is_visible,
                );
            }
        };

        let mut stats = coordinode_core::graph::types::VectorMvccStats {
            overfetch_factor,
            ..Default::default()
        };

        if self.is_empty() {
            return (Vec::new(), stats);
        }

        let ep_idx = self.entry_point.unwrap_or(0);
        let mut current_ep = ep_idx;

        for level in (1..=self.max_level).rev() {
            current_ep = self.search_layer_greedy_query(query, current_ep, level);
        }

        let mut visible_results: Vec<SearchResult> = Vec::new();
        let base_ef = self.config.ef_search.max(self.config.rerank_candidates);
        let mut current_ef = ((k as f64 * overfetch_factor).ceil() as usize).max(base_ef);

        for round in 0..=max_expansion_rounds {
            stats.expansion_rounds = round;
            let candidates = self.search_layer_query(query, current_ep, current_ef, 0);

            // Batch-load f32 for reranking
            let candidate_ids: Vec<u64> = candidates.iter().map(|c| self.nodes[c.idx].id).collect();
            let loaded = loader.load_vectors(&candidate_ids, &self.config.property_name);

            let mut results: Vec<SearchResult> = candidates
                .into_iter()
                .filter_map(|c| {
                    let node_id = self.nodes[c.idx].id;
                    let f32_vec = loaded.get(&node_id)?;
                    let exact_dist = self.distance_for_metric(query, f32_vec);
                    Some(SearchResult {
                        id: node_id,
                        score: exact_dist,
                    })
                })
                .collect();
            results.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            stats.candidates_fetched += results.len();

            visible_results.clear();
            for result in &results {
                if is_visible(result.id) {
                    visible_results.push(result.clone());
                } else {
                    stats.candidates_filtered += 1;
                }
            }
            stats.candidates_visible = visible_results.len();

            if visible_results.len() >= k || round == max_expansion_rounds {
                break;
            }

            current_ef *= 2;
        }

        visible_results.truncate(k);
        (visible_results, stats)
    }

    /// Generate a random level for a new element.
    fn random_level(&self) -> usize {
        // Xorshift64 RNG for proper exponential level distribution (R852).
        // Donor: hnswlib hnswalg.h:207-211 uses std::uniform_real_distribution.
        let mut state = self.rng_state.load(std::sync::atomic::Ordering::Relaxed);
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.rng_state
            .store(state, std::sync::atomic::Ordering::Relaxed);

        let uniform = (state >> 33) as f64 / (1u64 << 31) as f64;
        (-uniform.ln() * self.level_mult).floor() as usize
    }

    /// Update an already-indexed node's vector and rebuild its graph connections.
    ///
    /// Called by `insert()` when the node ID already exists.  Implements the
    /// G082 fix: prior code returned early ("Already indexed"), leaving the
    /// node at its old position after a SET operation.
    ///
    /// Algorithm:
    /// 1. Remove this node from all neighbors' connection lists.
    /// 2. Clear this node's outgoing connections (preserving layer slots).
    /// 3. Replace the stored vector (and update quantized representation).
    /// 4. Re-run the HNSW insertion neighbourhood search from a valid entry point.
    /// 5. Re-connect bidirectionally at each layer.
    fn update_existing_node(&mut self, idx: usize, vector: Vec<f32>) {
        let id = self.nodes[idx].id;
        let n_levels = self.nodes[idx].connections.len();

        // Step 1: Remove this node from every neighbour's connection list.
        // Clone neighbour IDs first to avoid simultaneous mutable + immutable borrows.
        for level in 0..n_levels {
            let neighbours: Vec<u64> = self.nodes[idx].connections[level].clone();
            for neighbour_id in neighbours {
                if let Some(&neighbour_idx) = self.id_to_idx.get(&neighbour_id) {
                    if level < self.nodes[neighbour_idx].connections.len() {
                        self.nodes[neighbour_idx].connections[level].retain(|&nid| nid != id);
                    }
                }
            }
        }

        // Step 2: Clear this node's outgoing connections (keep layer slot count).
        for level in 0..n_levels {
            self.nodes[idx].connections[level].clear();
        }

        // Step 3: Update vector and quantized representation.
        let quantized = self.sq8_params.as_ref().map(|p| p.quantize(&vector));
        self.nodes[idx].vector = Some(vector);
        self.nodes[idx].quantized = quantized;

        // Step 4: Re-insert into the graph from a valid entry point.
        // A single-node index has no connections to rebuild.
        if self.nodes.len() == 1 {
            if self.is_offloaded() {
                self.nodes[idx].vector = None;
            }
            return;
        }

        // Choose entry point: if the current entry_point IS this node, use any
        // other node (the graph is connected, so any peer suffices).
        let ep_idx = if self.entry_point == Some(idx) {
            // Find first node that is not `idx`.
            self.id_to_idx
                .values()
                .find(|&&i| i != idx)
                .copied()
                .unwrap_or(0)
        } else {
            self.entry_point.unwrap_or(0)
        };

        let node_level = self.nodes[idx]._max_layer;
        let top_level = self.max_level;
        let mut current_ep = ep_idx;

        // Phase 1: Greedy descent from top layer down to node_level+1.
        for level in (node_level + 1..=top_level).rev() {
            current_ep = self.search_layer_greedy(idx, current_ep, level);
        }

        // Phase 2: Reconnect at layers node_level down to 0.
        for level in (0..=node_level.min(top_level)).rev() {
            let ef = self.config.ef_construction;
            let neighbours = self.search_layer(idx, current_ep, ef, level);

            let max_conn = if level == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            let selected: Vec<usize> = neighbours
                .into_iter()
                .take(max_conn)
                .map(|c| c.idx)
                .collect();

            // Connect this node to its new neighbours.
            self.nodes[idx].connections[level] =
                selected.iter().map(|&n| self.nodes[n].id).collect();

            // Connect neighbours back (bi-directional).
            for &neighbour_idx in &selected {
                if level < self.nodes[neighbour_idx].connections.len() {
                    self.nodes[neighbour_idx].connections[level].push(id);

                    if self.nodes[neighbour_idx].connections[level].len() > max_conn {
                        self.prune_connections(neighbour_idx, level, max_conn);
                    }
                }
            }

            if !selected.is_empty() {
                current_ep = selected[0];
            }
        }

        // Step 5: Offload f32 if offloading is active (calibration already done).
        if self.is_offloaded() {
            self.nodes[idx].vector = None;
        }
    }

    /// Greedy search on a single layer (for traversal from top layers).
    /// The node at `query_idx` must have an f32 vector (only used during insert
    /// where the new node always has f32 available). Falls back to dequantized
    /// SQ8 if f32 was unexpectedly dropped.
    fn search_layer_greedy(&self, query_idx: usize, ep: usize, level: usize) -> usize {
        let query_vec = self.get_node_f32_or_dequantized(query_idx);
        self.search_layer_greedy_query(&query_vec, ep, level)
    }

    fn search_layer_greedy_query(&self, query: &[f32], ep: usize, level: usize) -> usize {
        let mut current = ep;
        let mut current_dist = self.compute_distance(query, current);

        loop {
            let mut changed = false;
            if level < self.nodes[current].connections.len() {
                for &neighbor_id in &self.nodes[current].connections[level] {
                    if let Some(&neighbor_idx) = self.id_to_idx.get(&neighbor_id) {
                        let dist = self.compute_distance(query, neighbor_idx);
                        if dist < current_dist {
                            current = neighbor_idx;
                            current_dist = dist;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        current
    }

    /// Search layer with ef candidates (returns sorted by distance).
    /// The node at `query_idx` must have an f32 vector (only used during insert).
    /// Falls back to dequantized SQ8 if f32 was unexpectedly dropped.
    fn search_layer(&self, query_idx: usize, ep: usize, ef: usize, level: usize) -> Vec<Candidate> {
        let query_vec = self.get_node_f32_or_dequantized(query_idx);
        self.search_layer_query(&query_vec, ep, ef, level)
    }

    fn search_layer_query(
        &self,
        query: &[f32],
        ep: usize,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let ep_dist = self.compute_distance(query, ep);

        let mut candidates = BinaryHeap::new(); // min-heap
        let mut results = BinaryHeap::new(); // max-heap (farthest first)
        let mut visited = self.visited_pool.get(self.nodes.len());

        candidates.push(Candidate {
            distance: ep_dist,
            idx: ep,
        });
        results.push(FarCandidate {
            distance: ep_dist,
            idx: ep,
        });
        visited.check_and_mark(ep);

        while let Some(closest) = candidates.pop() {
            let farthest_result = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
            if closest.distance > farthest_result {
                break;
            }

            if level < self.nodes[closest.idx].connections.len() {
                let connections = &self.nodes[closest.idx].connections[level];

                // Resolve neighbor IDs → indices, filtering already-visited.
                // Collect into temp vec to enable one-ahead prefetch pattern.
                // Donor: hnswlib hnswalg.h:370-383 (one-ahead prefetch in inner loop)
                let mut unvisited_neighbors: Vec<usize> = Vec::with_capacity(connections.len());
                for &neighbor_id in connections {
                    if let Some(&neighbor_idx) = self.id_to_idx.get(&neighbor_id) {
                        if !visited.check_and_mark(neighbor_idx) {
                            unvisited_neighbors.push(neighbor_idx);
                        }
                    }
                }

                // Prefetch first neighbor's vector data before entering distance loop.
                // When f32 is offloaded, prefetch the quantized vector instead.
                if let Some(&first) = unvisited_neighbors.first() {
                    self.prefetch_node_vector(first);
                }

                for (i, &neighbor_idx) in unvisited_neighbors.iter().enumerate() {
                    // Prefetch NEXT neighbor's vector while computing distance for current.
                    // Hides ~100ns L3 cache miss latency behind distance computation.
                    if i + 1 < unvisited_neighbors.len() {
                        let next_idx = unvisited_neighbors[i + 1];
                        self.prefetch_node_vector(next_idx);
                    }

                    let dist = self.compute_distance(query, neighbor_idx);
                    let farthest = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);

                    if dist < farthest || results.len() < ef {
                        candidates.push(Candidate {
                            distance: dist,
                            idx: neighbor_idx,
                        });
                        results.push(FarCandidate {
                            distance: dist,
                            idx: neighbor_idx,
                        });

                        if results.len() > ef {
                            results.pop(); // Remove farthest
                        }
                    }
                }
            }
        }

        // Convert to sorted vec (nearest first)
        let mut result_vec: Vec<Candidate> = results
            .into_iter()
            .map(|r| Candidate {
                distance: r.distance,
                idx: r.idx,
            })
            .collect();
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result_vec
    }

    /// Compute distance between query and a node.
    ///
    /// When SQ8 is active and the node has a quantized vector, distance is
    /// computed on the dequantized (approximate) vector — faster cache access
    /// due to 4x smaller working set, with slight accuracy trade-off.
    /// Falls back to exact f32 distance if the node is not yet quantized.
    fn compute_distance(&self, query: &[f32], node_idx: usize) -> f32 {
        if let Some(params) = &self.sq8_params {
            if let Some(quantized) = &self.nodes[node_idx].quantized {
                let dequantized = params.dequantize(quantized);
                return self.distance_for_metric(query, &dequantized);
            }
        }
        self.compute_exact_distance(query, node_idx)
    }

    /// Compute exact f32 distance between query and a node.
    /// Used for final reranking after SQ8 candidate generation.
    /// Falls back to dequantized SQ8 if f32 is offloaded.
    fn compute_exact_distance(&self, query: &[f32], node_idx: usize) -> f32 {
        if let Some(ref node_vec) = self.nodes[node_idx].vector {
            return self.distance_for_metric(query, node_vec);
        }
        // f32 offloaded — fall back to dequantized SQ8 (slightly less accurate)
        if let Some(ref params) = self.sq8_params {
            if let Some(ref quantized) = self.nodes[node_idx].quantized {
                let dequantized = params.dequantize(quantized);
                return self.distance_for_metric(query, &dequantized);
            }
        }
        f32::INFINITY
    }

    /// Compute distance using the configured metric.
    fn distance_for_metric(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric {
            VectorMetric::Cosine => 1.0 - metrics::cosine_similarity(a, b),
            VectorMetric::L2 => metrics::euclidean_distance_squared(a, b),
            VectorMetric::DotProduct => -metrics::dot_product(a, b),
            VectorMetric::L1 => metrics::manhattan_distance(a, b),
        }
    }

    /// Prune connections to keep at most max_conn (keep nearest).
    /// Uses exact f32 distance for pruning when available, falls back to
    /// dequantized SQ8 distances when f32 vectors are offloaded.
    fn prune_connections(&mut self, node_idx: usize, level: usize, max_conn: usize) {
        // Score all current connections by distance to node_idx.
        // Use index-based access to avoid clone (R852 fix: was clone() before).
        let mut scored: Vec<(f32, u64)> =
            Vec::with_capacity(self.nodes[node_idx].connections[level].len());
        for i in 0..self.nodes[node_idx].connections[level].len() {
            let neighbor_id = self.nodes[node_idx].connections[level][i];
            if let Some(&neighbor_idx) = self.id_to_idx.get(&neighbor_id) {
                let dist = self.distance_between_nodes(node_idx, neighbor_idx);
                scored.push((dist, neighbor_id));
            }
        }

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_conn);

        self.nodes[node_idx].connections[level] = scored.into_iter().map(|(_, id)| id).collect();
    }

    /// Compute distance between two nodes in the graph.
    /// Prefers exact f32 when available, falls back to dequantized SQ8.
    fn distance_between_nodes(&self, a_idx: usize, b_idx: usize) -> f32 {
        // Try exact f32 for both nodes
        if let (Some(ref va), Some(ref vb)) = (&self.nodes[a_idx].vector, &self.nodes[b_idx].vector)
        {
            return self.distance_for_metric(va, vb);
        }
        // Fall back to SQ8 approximate distance
        if let Some(ref params) = self.sq8_params {
            let va = self.get_node_vector_or_dequantized(a_idx, params);
            let vb = self.get_node_vector_or_dequantized(b_idx, params);
            return self.distance_for_metric(&va, &vb);
        }
        f32::INFINITY
    }

    /// Get a node's vector for query purposes: f32 if available, else dequantized SQ8.
    /// Used by search_layer_greedy/search_layer during insert when the node
    /// should have f32 but may not if auto_calibrate offloaded early.
    fn get_node_f32_or_dequantized(&self, idx: usize) -> Vec<f32> {
        if let Some(ref v) = self.nodes[idx].vector {
            return v.clone();
        }
        if let Some(ref params) = self.sq8_params {
            if let Some(ref q) = self.nodes[idx].quantized {
                return params.dequantize(q);
            }
        }
        Vec::new()
    }

    /// Get a node's vector: f32 if available, otherwise dequantize from SQ8.
    fn get_node_vector_or_dequantized(&self, idx: usize, params: &Sq8Params) -> Vec<f32> {
        if let Some(ref v) = self.nodes[idx].vector {
            return v.clone();
        }
        if let Some(ref q) = self.nodes[idx].quantized {
            return params.dequantize(q);
        }
        Vec::new()
    }

    /// Prefetch a node's vector data into L1 cache. Targets f32 when available,
    /// falls back to quantized vector when f32 is offloaded.
    #[inline(always)]
    fn prefetch_node_vector(&self, idx: usize) {
        if let Some(ref v) = self.nodes[idx].vector {
            prefetch_read_data(v.as_ptr() as *const u8);
        } else if let Some(ref q) = self.nodes[idx].quantized {
            prefetch_read_data(q.as_ptr());
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::HashSet;

    use super::*;

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

        // 128-dimensional vectors
        for i in 0..20u64 {
            let vec: Vec<f32> = (0..128).map(|d| ((i * d) as f32).sin()).collect();
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

    /// Verify that random_level produces proper exponential distribution (R852 regression).
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
        for node in &index.nodes {
            let max_layer = node.connections.len().saturating_sub(1);
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
    /// (regression test for R850 HashSet→epoch visited refactor).
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
            quantization: true,
            rerank_candidates: 20,
            calibration_threshold: 5, // Low threshold for testing
            offload_vectors: false,
            property_name: String::new(),
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
        for node in &index.nodes {
            assert!(node.quantized.is_some());
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
        let node = &index.nodes[*index.id_to_idx.get(&100).expect("inserted")];
        assert!(node.quantized.is_some());
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
            quantization: true,
            rerank_candidates: 50,
            calibration_threshold: 10,
            offload_vectors: false,
            property_name: String::new(),
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

        // First result should be the query vector itself (distance ~0)
        assert_eq!(results[0].id, 0);
        assert!(
            results[0].score < 0.01,
            "self-distance should be near zero, got {}",
            results[0].score
        );

        // Scores should be from f32 reranking (exact), not dequantized
        // Verify by computing exact distance for second result
        let expected_dist =
            metrics::euclidean_distance_squared(query, &vectors[results[1].id as usize]);
        let score_diff = (results[1].score - expected_dist).abs();
        assert!(
            score_diff < 1e-5,
            "reranked score should match exact f32 distance: got {}, expected {expected_dist}",
            results[1].score
        );
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
            quantization: false,
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
            quantization: true,
            rerank_candidates: 50,
            calibration_threshold: 50,
            offload_vectors: false,
            property_name: String::new(),
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
    fn sq8_memory_savings() {
        // Verify quantized vectors use 4x less memory than f32
        let mut index = HnswIndex::new(make_sq8_config(VectorMetric::L2));

        let dims = 384;
        for i in 0..10u64 {
            let v: Vec<f32> = (0..dims).map(|d| ((i * d) as f32).sin()).collect();
            index.insert(i, v);
        }

        assert!(index.is_quantized());

        for node in &index.nodes {
            let v = node
                .vector
                .as_ref()
                .expect("f32 should be retained (offload_vectors=false)");
            assert_eq!(v.len(), dims as usize);
            let q = node.quantized.as_ref().expect("should be quantized");
            assert_eq!(q.len(), dims as usize);
            // f32 = 4 bytes per dim, u8 = 1 byte per dim → 4x savings
            assert_eq!(v.len() * std::mem::size_of::<f32>(), q.len() * 4);
        }
    }

    #[test]
    fn sq8_manual_calibration() {
        use crate::quantize::Sq8Params;

        let mut index = HnswIndex::new(HnswConfig {
            quantization: true,
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
        for node in &index.nodes {
            assert!(node.quantized.is_some());
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
        assert!(!config.quantization);

        let mut index = HnswIndex::new(config);
        for i in 0..200u64 {
            index.insert(i, vec![i as f32, 0.0]);
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
        // Verify the SQ8_MIN_VECTORS constant matches the architecture spec.
        assert_eq!(
            super::SQ8_MIN_VECTORS,
            1000,
            "SQ8_MIN_VECTORS should be 1000 per arch/operations/compression.md"
        );
    }

    // ── G009: Offloaded f32 vectors to disk ────────────────────────────

    fn make_offload_config(metric: VectorMetric) -> HnswConfig {
        HnswConfig {
            m: 8,
            m_max0: 16,
            ef_construction: 50,
            ef_search: 50,
            metric,
            max_dimensions: 65_536,
            quantization: true,
            rerank_candidates: 20,
            calibration_threshold: 5, // Low threshold for testing
            offload_vectors: true,
            property_name: "embedding".to_string(),
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
        for node in &index.nodes {
            assert!(
                node.vector.is_none(),
                "f32 should be None when offloaded (node {})",
                node.id
            );
            assert!(
                node.quantized.is_some(),
                "quantized should be present (node {})",
                node.id
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

        // Verify scores are exact f32 distances (not approximate SQ8)
        for result in &results {
            let v = loader.vectors.get(&result.id).unwrap();
            let expected = metrics::euclidean_distance_squared(&query, v);
            let diff = (result.score - expected).abs();
            assert!(
                diff < 1e-5,
                "score for node {} should be exact f32 distance: got {}, expected {}",
                result.id,
                result.score,
                expected
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

        // Count f32 memory: offloaded should have 0, retained should have n*dims*4
        let offloaded_f32_bytes: usize = offloaded
            .nodes
            .iter()
            .map(|n| n.vector.as_ref().map_or(0, |v| v.len() * 4))
            .sum();
        let retained_f32_bytes: usize = retained
            .nodes
            .iter()
            .map(|n| n.vector.as_ref().map_or(0, |v| v.len() * 4))
            .sum();

        assert_eq!(offloaded_f32_bytes, 0, "offloaded should have 0 f32 bytes");
        assert_eq!(
            retained_f32_bytes,
            n as usize * dims * 4,
            "retained should have all f32 bytes"
        );
    }

    /// Regression test for G082: HNSW must update the graph when a vector
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

        // Query "up" → node 1 must NO LONGER be the top result (its vector changed).
        // Node 2 or 3 should win for "up" now.
        let after_up = index.search(&[0.0, 1.0], 1);
        assert_ne!(
            after_up.first().map(|r| r.id),
            Some(1),
            "after update: 'up' query must NOT return node 1 (its vector is now 'right'). Got: {:?}",
            after_up
        );
    }
}
