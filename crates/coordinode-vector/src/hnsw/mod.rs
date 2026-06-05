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

mod entry_point;
mod inline_layer0;
mod neighbours;
mod visited;

pub use neighbours::AtomicNeighbourList;

// Loom integration tests live in `tests/loom_neighbours.rs` and
// `tests/loom_entry_point.rs`; they need the lock-free primitives in
// scope. We expose the types publicly only under the model-checker
// build flag — regular builds keep them crate-private.
#[cfg(loom)]
pub use entry_point::EntryPoint as LoomEntryPoint;
#[cfg(loom)]
pub use entry_point::PromoteOutcome as LoomPromoteOutcome;
#[cfg(loom)]
pub use neighbours::AtomicNeighbourList as LoomAtomicNeighbourList;

use entry_point::EntryPoint;

/// Compile-time cap on the inline neighbour slots per node per layer
/// (`m_max0` in HNSW literature). The atomic neighbour list stores its
/// slots inline as `[AtomicU64; M_MAX0]`, so this is a compile-time
/// constant rather than a runtime field. Default value covers
/// `m_max0 = 2 × M` for `M ≤ 32`, the common range for production
/// configurations.
///
/// Runtime configurations with `config.m_max0 > M_MAX0` are rejected at
/// [`HnswIndex::new`]. Cluster-wide homogeneity (every replica compiled
/// with the same `M_MAX0`) is also a precondition for the ship-graph-bytes
/// transfer mode (ADR-030).
pub const M_MAX0: usize = 64;

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
use crate::quantize::rabitq::{RaBitQCode, RaBitQExtCode, RaBitQParams, RaBitQQuery};
use crate::quantize::Sq8Params;

/// Per-vector RaBitQ encoding. The variant is fixed at index calibration
/// time from [`HnswConfig::quantization`] and must match across all nodes
/// in the same index — mixing 1-bit and multi-bit codes in a single index
/// is an invariant violation (different distance kernels, different code
/// shapes, no shared comparison semantics).
#[derive(Debug, Clone, PartialEq)]
pub enum RabitqEncoded {
    /// 1-bit sign-bit code, popcount distance kernel (R860). Default codec.
    OneBit(RaBitQCode),
    /// 2/3/4-bit Extended-RaBitQ code, centroid-LUT distance kernel (R862).
    /// `bits` is carried inside the [`RaBitQExtCode`].
    Multi(RaBitQExtCode),
}

/// Pull the precomputed `‖x‖` out of whichever encoded variant a node
/// carries. The cosine-rerank fast path in `compute_exact_distance` uses
/// this to skip the per-neighbour `norm_l2(b)` pass — the f32 dot
/// product already lives in the hot path; computing `b`'s norm again
/// per call was pure waste once we started doing dual-distance (RaBitQ
/// frontier + f32 results-heap).
#[inline]
fn rabitq_code_norm(enc: &RabitqEncoded, params: &RaBitQParams) -> f32 {
    match enc {
        RabitqEncoded::OneBit(c) => {
            // K=1 IVF: `c.norm = ‖r‖`, NOT `‖x‖`. Reconstruct the true
            // data-vector norm via the chroma identity
            // `‖x‖² = ‖centroid‖² + 2·radial + ‖r‖²` so the f32 cosine
            // rerank denominator (‖q‖·‖x‖) stays exact. Un-centered
            // codecs (empty centroid) have c_norm = radial = 0 and the
            // identity collapses to `‖x‖ = ‖r‖ = c.norm`, preserving the
            // 2114679 cached-norm fast path bit-for-bit.
            let c_norm = params.c_norm(c.cluster_id);
            let d_norm_sq = c_norm * c_norm + 2.0 * c.radial + c.norm * c.norm;
            d_norm_sq.sqrt()
        }
        RabitqEncoded::Multi(c) => c.norm,
    }
}

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

/// In-RAM quantization codec selector.
///
/// Per ADR-032, RaBitQ supersedes SQ8 as the primary in-RAM codec; SQ8 is
/// retained for the Phase 1.5 cross-shard disk rerank pool. `None` means
/// search runs entirely on f32 originals — appropriate for small indexes
/// where quantization overhead exceeds savings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QuantizationCodec {
    /// f32 originals only; no quantized representation.
    None,
    /// SQ8 scalar quantization (1 byte / dim, ~4× compression).
    /// Used by HNSW traversal when active; final top-K is reranked on f32.
    Sq8,
    /// RaBitQ 1-bit-per-dim with popcount distance kernel.
    /// `bits=1` is the default (~30× compression, mandatory rerank);
    /// `bits ∈ {2,3,4}` selects Extended-RaBitQ variants (separate task).
    RaBitQ { bits: u8 },
}

impl QuantizationCodec {
    /// True when this codec stores any quantized representation alongside
    /// (or in place of) the f32 original.
    pub fn is_active(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Reranking strategy on the RaBitQ search path (`quantization = RaBitQ`).
///
/// CoordiNode's traditional behaviour was [`RerankMode::Inline`]: every
/// neighbour visit computes the cheap RaBitQ-popcount distance AND an
/// exact f32 cosine in the same pass, so the `farthest_dist` termination
/// gate uses the exact metric. That trades ~2× per-visit work for the
/// best recall — the cron's recall-0.95 target on glove-100-angular only
/// ever hit it under inline rerank.
///
/// [`RerankMode::EndOfSearch`] follows the qdrant Binary Quantization
/// (`rescore: true`) / DiskANN / RaBitQ SIGMOD 2024 reference pattern:
/// run the whole HNSW traversal on cheap distances alone, then rerank
/// the final ef-sized result heap once at the end. Per-visit cost drops
/// to popcount + one Vec lookup (vs popcount + dot + two Vec lookups
/// under Inline), at the cost of using a noisy threshold during the
/// traversal — recall depends on how representative the cheap distance
/// ranking is.
///
/// [`RerankMode::None`] skips rerank entirely — fastest, recall ceiling
/// is whatever the RaBitQ popcount estimator delivers on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RerankMode {
    /// Exact f32 rerank on every neighbour visit. Preserves the highest
    /// recall but pays the structural ~2× per-visit cost.
    #[default]
    Inline,
    /// Single-pass HNSW search on cheap distances, then exact rerank of
    /// the final ef-sized result heap. Industry-standard pattern (qdrant,
    /// chroma, DiskANN). Per-visit cost is ~1 distance call instead of 2.
    EndOfSearch,
    /// No rerank — return the cheap-distance top-ef ranking as-is.
    /// Lowest cost, lowest recall ceiling. Intended for QPS-critical
    /// workloads where the popcount estimator is already accurate enough.
    None,
}

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
    /// In-RAM quantization codec for HNSW traversal. See [`QuantizationCodec`].
    pub quantization: QuantizationCodec,
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
    /// Rerank strategy on the RaBitQ search path. See [`RerankMode`].
    /// Default [`RerankMode::Inline`] preserves the historical recall
    /// trade. Bench / production callers wanting the qdrant-style
    /// end-of-search rerank pattern opt into [`RerankMode::EndOfSearch`];
    /// QPS-critical low-recall workloads can opt into [`RerankMode::None`].
    pub rerank_mode: RerankMode,
    /// Oversampling factor for [`RerankMode::EndOfSearch`]. The search
    /// traverses the graph with `frontier_ef = ceil(ef * factor)` so
    /// the cheap-distance heap accumulates a larger candidate pool,
    /// then the exact f32 rerank picks the best `ef` of those. Direct
    /// equivalent of qdrant's `oversampling` parameter; chroma and
    /// DiskANN expose the same knob under different names.
    ///
    /// `factor = 1.0` (default) means "no oversampling" — frontier and
    /// rerank pool are both `ef`. `factor = 2.0` doubles the frontier;
    /// recall climbs back toward inline-rerank parity at a modest QPS
    /// cost (the extra cheap-distance work scales linearly, the rerank
    /// pass scales linearly with the factor, but the f32 dot per
    /// candidate is amortised over the bigger window).
    ///
    /// Ignored when `rerank_mode != RerankMode::EndOfSearch`.
    pub rerank_oversample_factor: f32,
    /// α parameter for the RobustPrune neighbour-selection heuristic
    /// (Vamana paper / DiskANN, Algorithm 3). When `alpha_pruning > 1.0`,
    /// the construction phase replaces "take M closest" with α-pruning:
    /// for each kept neighbour `p*`, drop any candidate `p'` where
    /// `α · d(p*, p') ≤ d(p, p')` — i.e. p' is closer to p* than (1/α)×
    /// to the inserted node. This produces a sparser, more *diverse*
    /// graph: fewer redundant edges → lower fanout per query → fewer
    /// hops to reach the same recall → higher QPS at fixed recall.
    ///
    /// Vamana paper §3 recommends `α = 1.2` as the production default.
    /// `α = 1.0` is a no-op (degenerates to "take M closest", the
    /// original HNSW behaviour). The cost is O(R · |V|) extra
    /// `distance_between_nodes` calls per insert (R = max_conn,
    /// |V| = ef_construction) — a build-time tax for a search-time
    /// win.
    ///
    /// Default `1.0` (off) so existing recall thresholds in tests don't
    /// shift unexpectedly; production / bench callers opt in.
    pub alpha_pruning: f32,
    /// Expected upper bound on the number of vectors this index will hold.
    /// Used at construction to pre-allocate the `nodes` and
    /// `neighbours_l0` / `neighbours_upper` vectors so the insert hot path never pays
    /// `Vec` reallocation cost. Insert beyond `max_elements` is supported
    /// (the vectors grow normally) but the first overflow re-allocation
    /// pauses inserts briefly; size this generously when ingestion volume
    /// is known. Default: 1_000_000.
    pub max_elements: u32,
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
            quantization: QuantizationCodec::None,
            rerank_candidates: 100,
            calibration_threshold: 100,
            offload_vectors: false,
            property_name: String::new(),
            rerank_mode: RerankMode::Inline,
            rerank_oversample_factor: 1.0,
            alpha_pruning: 1.0,
            max_elements: 1_000_000,
        }
    }
}

/// A single element in the HNSW graph.
///
/// Per-layer neighbour lists are stored separately on
/// [`HnswIndex::neighbours_l0`] (hot path, layer 0) and
/// [`HnswIndex::neighbours_upper`] (cold path, layers ≥1) using lock-free
/// [`AtomicNeighbourList`]s — never inside this struct. The node's layer
/// count equals `HnswIndex::node_levels(node_idx)`; the node's max layer
/// is also captured in [`HnswNode::max_layer`] for cheap access from the
/// update / rebuild paths.
struct HnswNode {
    /// Node ID (maps to graph node ID).
    id: u64,
    /// Highest layer this element exists on. Same value as
    /// `node_levels(node_idx) - 1`; cached here so the rebuild path
    /// doesn't have to indirect through the mirror on every read.
    max_layer: usize,
}

/// HNSW index: in-memory approximate nearest neighbor graph.
pub struct HnswIndex {
    config: HnswConfig,
    /// All nodes' light metadata (id + max_layer). The hot per-node payload
    /// (f32 vector, SQ8 quantized, RaBitQ code) lives in the parallel
    /// `node_vectors` / `node_quantized` / `node_rabitq_codes` arrays so a
    /// search visit reads only the payload it actually needs. Before this
    /// SoA split the whole `HnswNode` struct (id + 3× Option<Vec> + usize
    /// = ~80 B) was loaded per visit; for a 1.18 M-node glove index that
    /// pulled ~90 MiB of mostly-unused metadata into L1/L2 during search.
    /// hnswlib's contiguous `data_level0_memory_` chose the opposite layout
    /// for the same reason — one allocation, one prefetch covers everything
    /// the inner loop needs. SoA is the Rust-friendly version of that
    /// invariant: parallel arrays sized to `nodes.len()`, lock-step on
    /// every push.
    nodes: Vec<HnswNode>,
    /// Original f32 vector data, parallel to `nodes`. `None` when offloaded
    /// to disk (see `HnswConfig::offload_vectors`). Read by the rerank path
    /// (`compute_exact_distance`) and the build path (`distance_between_
    /// nodes`).
    node_vectors: Vec<Option<Vec<f32>>>,
    /// SQ8-quantized vector, parallel to `nodes`. `None` until SQ8
    /// calibration completes (or always None when SQ8 is disabled).
    node_quantized: Vec<Option<Vec<u8>>>,
    /// RaBitQ code (1-bit popcount kernel or 2/3/4-bit Extended-RaBitQ),
    /// parallel to `nodes`. `None` until calibration completes; the
    /// variant is fixed at calibration time and never mixed within one
    /// index. This is the array the cosine-RaBitQ hot path hits on every
    /// neighbour visit.
    node_rabitq_codes: Vec<Option<RabitqEncoded>>,
    /// Lock-free layer-0 neighbour lists, one entry per node. Flat `Vec`
    /// indexed by node index — a search visit reads `neighbours_l0[idx]`
    /// with ONE pointer-stride into a contiguous heap allocation, no inner
    /// `Vec` heap chase. Mirrors the d365611 SmallVec inline pattern for
    /// RaBitQ codes: the same `Vec<Vec<…>>` double-indirection that hurt
    /// code reads also showed up here on the dominant visit-time lookup
    /// (every neighbour walk touches layer 0; upper layers only fire on
    /// the top-down descent in `search_layer`).
    neighbours_l0: Vec<AtomicNeighbourList<M_MAX0>>,
    /// Lock-free upper-layer neighbour lists. `neighbours_upper[idx]`
    /// holds layers 1..=top_level for node `idx` (length =
    /// `nodes[idx].max_layer`). Cold path — only walked during the
    /// top-down greedy descent in `search_layer_greedy`, never on the
    /// per-visit `search_layer` candidate expansion that dominates QPS.
    /// Keeping upper layers in `Vec<Vec<…>>` avoids paying the
    /// O(total_layers) hot-path penalty across nodes whose `max_layer ==
    /// 0` (the overwhelming majority on default `level_mult = 1/ln(M)`).
    neighbours_upper: Vec<Vec<AtomicNeighbourList<M_MAX0>>>,
    /// Map from node ID to index in `nodes` vec.
    id_to_idx: std::collections::HashMap<u64, usize>,
    /// Lock-free entry point: packed `(level, idx)` in a single
    /// `AtomicU64` with `u64::MAX` as the "empty index" sentinel.
    /// Multiple inserts that land on novel max-layers race through
    /// [`EntryPoint::try_promote`] (CAS-loop on a single atomic, max
    /// two iterations under realistic contention — see
    /// `arch/search/vector-parallel-insert.md` §"Layer-promotion
    /// race"). Replaces the previous `(Option<usize>, usize)` pair
    /// that was mutated under `&mut self` in the batch allocation
    /// phase, the last serialisation point on the lock-free insert
    /// path before this commit.
    ///
    /// Read at every search start (top→bottom layer iteration). Write
    /// on the first insert and on every novel-max-layer promotion;
    /// no-op CAS otherwise.
    entry_point: EntryPoint,
    /// Inverse of ln(M) for level generation.
    level_mult: f64,
    /// SQ8 calibration parameters. `None` until calibration_threshold vectors
    /// are inserted and auto-calibration runs.
    sq8_params: Option<Sq8Params>,
    /// RaBitQ rotation matrix + scalars. Constructed at calibration time
    /// (deterministic from a seed derived from `max_dimensions`) and stable
    /// for the lifetime of the index. `None` until calibration runs.
    rabitq_params: Option<RaBitQParams>,
    /// Pool of reusable visited lists for search. Avoids per-search allocation.
    visited_pool: VisitedPool,
    /// RNG state for random level selection (xorshift64).
    /// Proper RNG gives correct exponential layer distribution (R852 fix).
    /// AtomicU64 for future concurrent insert support (R858).
    rng_state: std::sync::atomic::AtomicU64,
    /// Optional persistent vector tier backing per ADR-033 (truth tier
    /// f32 + quantized rerank tier). `None` for in-memory-only indexes
    /// (tests, ad-hoc analytics); `Some` when the caller has wired up
    /// LSM-backed storage. Writes through this handle log f32 on insert
    /// and quantized bytes on (re)calibration; reads through it power
    /// Phase 1.5 cross-shard rerank and application-side custom rerank.
    vector_tier: Option<crate::storage::VectorTierHandle>,
    /// Contiguous per-node store covering layer-0 neighbours, f32 vector,
    /// RaBitQ code and external label in a single stride-addressable
    /// allocation. Mirrors the hnswlib `data_level0_memory_` layout that
    /// shows super-linear MT4 scaling on sift-128 f32 search (worker
    /// threads share L3 fills on common neighbour visits, whereas the
    /// SoA fields above keep cache lines uncorrelated). Allocated on
    /// the first insert once the vector dim is known. Populated as
    /// write-through alongside the SoA fields; the search side reads
    /// remain on SoA today, the follow-up commit on the same plan
    /// switches the layer-0 neighbour read to this store.
    inline_layer0: Option<inline_layer0::InlineLayer0>,
}

/// Read-only result of the planning phase of an insert (C2, R858b).
///
/// Produced by [`HnswIndex::compute_insert_plan`] (takes `&self`) and
/// consumed by [`HnswIndex::apply_insert_plan`] (takes `&mut self`). The
/// type carries no references into the index, so it can outlive the
/// borrow used to compute it — required for batch ingestion where N plans
/// are computed in parallel under `&self` and then applied one-by-one
/// under `&mut self`.
#[derive(Debug, Clone)]
pub(crate) struct InsertPlan {
    /// Node ID being inserted (the user-facing identifier, not the index).
    pub id: u64,
    /// Layer the new node is being inserted at (top of its layer stack).
    pub new_level: usize,
    /// One [`LayerPlan`] per layer from `new_level` down to `0`.
    /// Empty when [`InsertPlan::is_first_node`] is true.
    pub per_layer: Vec<LayerPlan>,
    /// `true` if this plan was computed against an empty index. The apply
    /// path uses this to short-circuit the bidirectional connection loop.
    pub is_first_node: bool,
}

/// Per-layer outcome of the planning phase: which existing nodes (by
/// index into [`HnswIndex::nodes`]) the new node should connect to, plus
/// the layer-specific max-fanout used for the bidirectional prune.
#[derive(Debug, Clone)]
pub(crate) struct LayerPlan {
    pub level: usize,
    /// Indices into [`HnswIndex::nodes`] of the chosen neighbours, ordered
    /// nearest-first (which is also the source-of-truth for entry-point
    /// hand-off between layers during apply).
    pub selected_idxs: Vec<usize>,
    /// Max neighbours per side at this layer — `m_max0` at layer 0,
    /// `m` everywhere else.
    pub max_conn: usize,
}

/// A search result with node ID and distance/similarity.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

/// Ordered candidate for min-heap (by distance, ascending).
///
/// Packed at 8 bytes total (f32 + u32) — halves the heap memory
/// footprint vs a `usize`-idx layout (16 bytes after padding) so the
/// BinaryHeap sift-up/down passes touch half as many cache lines.
/// `max_elements: u32` in HnswConfig already bounds the per-shard
/// node count below `u32::MAX`, so u32 idx is correctness-safe.
#[derive(Clone, Copy)]
struct Candidate {
    distance: f32,
    idx: u32,
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

/// Per-search cached state.
///
/// HNSW search calls the metric distance function thousands of times per query,
/// always with the same query vector. Anything that depends only on the query
/// (norms, projection sums, lookup tables) is computed once and stored here.
struct QueryCtx<'a> {
    /// The query vector.
    vec: &'a [f32],
    /// ‖query‖₂ — pre-computed once per search. Used by Cosine metric only;
    /// other metrics ignore this field.
    norm_l2: f32,
    /// Pre-encoded RaBitQ representation for the active codec, populated
    /// once per search and reused across thousands of `compute_distance`
    /// calls. Encoding is `O(D²)` (matrix-vector); paying it per call
    /// would defeat the codec kernel.
    ///
    /// For 1-bit RaBitQ this carries [`RaBitQQuery`] — the 4-bit-plane
    /// quantization of the rotated query used by the asymmetric paper-
    /// Equation-20 kernel. For Extended (2/3/4-bit) it carries the symmetric
    /// LUT-friendly code via [`RabitqEncoded::Multi`].
    rabitq_query: Option<RabitqQuery>,
}

/// Codec-typed query encoding cached in [`QueryCtx`]. Mirrors
/// [`RabitqEncoded`] on the storage side: each variant pairs with the
/// matching stored code shape and the kernel that consumes it.
#[derive(Debug, Clone)]
enum RabitqQuery {
    /// 1-bit data × 4-bit-plane query (asymmetric, paper §3.3.2).
    OneBit(RaBitQQuery),
    /// 2/3/4-bit Extended-RaBitQ — query has the same packed shape as
    /// stored codes (symmetric LUT kernel from R862).
    Multi(RaBitQExtCode),
}

impl<'a> QueryCtx<'a> {
    /// Build a context for SEARCH-time queries: encodes the query against
    /// the active RaBitQ rotation so the popcount kernel can score each
    /// neighbour without per-call encoding cost.
    fn new(
        vec: &'a [f32],
        metric: VectorMetric,
        rabitq: Option<&RaBitQParams>,
        codec: &QuantizationCodec,
    ) -> Self {
        let norm_l2 = if matches!(metric, VectorMetric::Cosine) {
            metrics::norm_l2(vec)
        } else {
            0.0
        };
        // Encode against the active rotation matrix iff RaBitQ is calibrated
        // AND the query's dimensionality matches. Mismatched dims fall back
        // to the f32 distance path with no encode cost. 1-bit uses the
        // asymmetric paper kernel (4-bit-plane query against 1-bit data);
        // 2/3/4-bit Extended uses the symmetric LUT kernel.
        let rabitq_query = rabitq.and_then(|p| {
            if vec.len() != p.dims() as usize {
                return None;
            }
            match codec {
                QuantizationCodec::RaBitQ { bits: 1 } => {
                    Some(RabitqQuery::OneBit(p.encode_query(vec)))
                }
                QuantizationCodec::RaBitQ { bits } if (2..=4).contains(bits) => {
                    Some(RabitqQuery::Multi(p.encode_ext(vec, *bits)))
                }
                _ => None,
            }
        });
        Self {
            vec,
            norm_l2,
            rabitq_query,
        }
    }

    /// Build a context for BUILD-time graph traversal. Skips RaBitQ
    /// encoding so `compute_distance` falls through to the exact f32
    /// metric for every candidate. This is what the RaBitQ SIGMOD 2024
    /// paper and the Milvus / IVF-RaBitQ implementations do: construct
    /// the HNSW graph on f32 truth, only compress at search time.
    ///
    /// Reason: noisy RaBitQ distance estimates corrupt neighbour
    /// selection during the ~N×log(N) `search_layer_query` calls that
    /// drive `compute_insert_plan`. At small N the noise is tolerable,
    /// but at N≥10⁵ the cumulative selection error degrades graph
    /// connectivity to the point where no search-time ef budget can
    /// recover the true top-K (recall plateau independent of ef — what
    /// the glove-100-angular bench showed at recall=0.17).
    fn new_for_build(vec: &'a [f32], metric: VectorMetric) -> Self {
        let norm_l2 = if matches!(metric, VectorMetric::Cosine) {
            metrics::norm_l2(vec)
        } else {
            0.0
        };
        Self {
            vec,
            norm_l2,
            rabitq_query: None,
        }
    }
}

/// Max-heap candidate (for maintaining top-K worst). 8-byte packed —
/// same layout rationale as [`Candidate`].
#[derive(Clone, Copy)]
struct FarCandidate {
    distance: f32,
    idx: u32,
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
    ///
    /// Configurations with `config.m_max0 > M_MAX0` are clamped down with a
    /// `warn!` log — the compile-time const cap is a hard upper bound for
    /// the inline atomic neighbour storage.
    pub fn new(config: HnswConfig) -> Self {
        if config.m_max0 > M_MAX0 {
            warn!(
                requested = config.m_max0,
                cap = M_MAX0,
                "HnswConfig::m_max0 exceeds compile-time M_MAX0 cap; \
                 inline atomic neighbour storage limits effective fan-out \
                 to {M_MAX0}. Increase M_MAX0 and rebuild for higher caps.",
            );
        }
        let level_mult = 1.0 / (config.m as f64).ln();
        // Pre-size storage so steady-state inserts never pay reallocation
        // cost on the hot path. `max_elements` is advisory — exceeding it
        // is supported, the first overflow simply triggers Vec growth.
        let capacity = config.max_elements as usize;
        let id_to_idx = std::collections::HashMap::with_capacity(capacity);
        Self {
            config,
            nodes: Vec::with_capacity(capacity),
            node_vectors: Vec::with_capacity(capacity),
            node_quantized: Vec::with_capacity(capacity),
            node_rabitq_codes: Vec::with_capacity(capacity),
            neighbours_l0: Vec::with_capacity(capacity),
            neighbours_upper: Vec::with_capacity(capacity),
            id_to_idx,
            entry_point: EntryPoint::new(),
            // max_level is derived from entry_point.load() at every
            // search start (top→bottom layer iteration). No separate
            // field — the packed AtomicU64 carries it.
            level_mult,
            sq8_params: None,
            rabitq_params: None,
            visited_pool: VisitedPool::new(),
            // Seed from address of self (varies per instance). Non-deterministic but fast.
            rng_state: std::sync::atomic::AtomicU64::new(0xdeadbeef_cafebabe),
            vector_tier: None,
            inline_layer0: None,
        }
    }

    /// Wire a persistent vector tier backend (truth f32 + quantized rerank
    /// per ADR-033). After this call every successful insert writes the
    /// f32 bytes to the truth tier, and every (re)calibration writes
    /// quantized codes to the rerank tier. Failures during tier writes
    /// are logged through `tracing::warn` but do NOT roll back the
    /// in-RAM insert — the in-RAM graph is authoritative; the tier is
    /// rebuilt from data on recovery (per ADR-018 + replication.md).
    /// Pass `None` to disable (default state for in-memory tests).
    pub fn set_vector_tier(&mut self, tier: Option<crate::storage::VectorTierHandle>) {
        self.vector_tier = tier;
    }

    /// Whether persistent vector tier is wired. Used by tests + by the
    /// upper layers to decide whether to skip the legacy Node-partition
    /// vector-property write.
    pub fn has_vector_tier(&self) -> bool {
        self.vector_tier.is_some()
    }

    /// Encode a vector to the active RaBitQ variant per
    /// [`HnswConfig::quantization`]. Returns `None` if RaBitQ is not
    /// the configured codec, dims mismatch, or `bits` is unsupported.
    /// Single source of truth for "which variant lives in this index"
    /// — used by every insert / calibration / search-side encode path.
    fn encode_rabitq(&self, params: &RaBitQParams, vector: &[f32]) -> Option<RabitqEncoded> {
        if vector.len() != params.dims() as usize {
            return None;
        }
        match self.config.quantization {
            QuantizationCodec::RaBitQ { bits: 1 } => {
                Some(RabitqEncoded::OneBit(params.encode(vector)))
            }
            QuantizationCodec::RaBitQ { bits } if (2..=4).contains(&bits) => {
                Some(RabitqEncoded::Multi(params.encode_ext(vector, bits)))
            }
            _ => None,
        }
    }

    /// Returns the SQ8 calibration parameters, if calibrated.
    pub fn sq8_params(&self) -> Option<&Sq8Params> {
        self.sq8_params.as_ref()
    }

    /// Returns whether SQ8 quantization is active (calibrated and enabled).
    pub fn is_quantized(&self) -> bool {
        matches!(self.config.quantization, QuantizationCodec::Sq8) && self.sq8_params.is_some()
    }

    /// Returns the RaBitQ rotation parameters, if calibrated.
    pub fn rabitq_params(&self) -> Option<&RaBitQParams> {
        self.rabitq_params.as_ref()
    }

    /// Returns whether RaBitQ quantization is active (calibrated and enabled).
    pub fn is_rabitq_active(&self) -> bool {
        matches!(self.config.quantization, QuantizationCodec::RaBitQ { .. })
            && self.rabitq_params.is_some()
    }

    /// Returns whether f32 vectors are offloaded to disk.
    /// True only when both `offload_vectors` and SQ8 quantization are active.
    pub fn is_offloaded(&self) -> bool {
        self.config.offload_vectors && self.is_quantized()
    }

    /// Manually set RaBitQ calibration parameters (e.g., from a saved
    /// index). Encodes every existing node against the provided rotation,
    /// overwriting any prior code. Required on segment reload: the rotation
    /// matrix is part of the durable index — auto-calibrating on reload
    /// would pick a different `R` and produce codes incomparable with the
    /// ones already on disk.
    pub fn set_rabitq_params(&mut self, params: RaBitQParams) {
        // Two-pass to satisfy the borrow checker: collect (idx, encoded)
        // first using `&self` (encode_rabitq needs `&self.config`), then
        // assign back via `&mut self`. The encoded vector is `Option<_>` —
        // mismatched dims / disabled codec yield None and the slot stays
        // empty (consistent with prior behaviour).
        let encoded: Vec<(usize, Option<RabitqEncoded>)> = self
            .node_vectors
            .iter()
            .enumerate()
            .map(|(i, v_opt)| {
                let enc = v_opt
                    .as_deref()
                    .and_then(|v| self.encode_rabitq(&params, v));
                (i, enc)
            })
            .collect();
        for (i, enc) in encoded {
            self.mirror_rabitq_to_inline(i, enc.as_ref());
            self.node_rabitq_codes[i] = enc;
        }
        self.rabitq_params = Some(params);
    }

    /// Manually set SQ8 calibration parameters (e.g., from a saved index).
    /// Quantizes all existing nodes that don't have quantized vectors yet.
    /// If `offload_vectors` is enabled, drops f32 after quantizing.
    pub fn set_sq8_params(&mut self, params: Sq8Params) {
        for idx in 0..self.nodes.len() {
            if let Some(v) = self.node_vectors[idx].as_ref() {
                self.node_quantized[idx] = Some(params.quantize(v));
            }
            if self.config.offload_vectors {
                self.node_vectors[idx] = None;
            }
        }
        self.sq8_params = Some(params);
    }

    /// Check if the node at `idx` has an in-memory f32 vector.
    pub fn has_f32_vector(&self, idx: usize) -> bool {
        self.node_vectors.get(idx).is_some_and(|v| v.is_some())
    }

    /// Get a reference to the f32 vector at node index `idx`, if present.
    pub fn get_vector(&self, idx: usize) -> Option<&[f32]> {
        self.node_vectors.get(idx).and_then(|v| v.as_deref())
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
            .node_vectors
            .iter()
            .filter_map(|v| v.as_deref())
            .collect();
        if let Some(params) = Sq8Params::calibrate(&refs) {
            for idx in 0..self.nodes.len() {
                if let Some(v) = self.node_vectors[idx].as_ref() {
                    let code = params.quantize(v);
                    self.node_quantized[idx] = Some(code);
                }
                if self.config.offload_vectors {
                    self.node_vectors[idx] = None;
                }
            }
            self.sq8_params = Some(params);
        }
    }

    /// Auto-calibrate RaBitQ from the inferred dimensionality of stored
    /// vectors. Called once when `calibration_threshold` vectors are present.
    ///
    /// The rotation matrix is deterministic in `(dims, seed)` where `seed`
    /// is derived from the index's configured `max_dimensions` to give a
    /// stable identity across restarts without requiring callers to provide
    /// one (R860 starting point; a per-shard seed comes with R-PUSH chains).
    fn auto_calibrate_rabitq(&mut self) {
        // Need at least one vector to infer D.
        let dims = match self.node_vectors.iter().find_map(|v| v.as_ref()) {
            Some(v) => v.len(),
            None => return,
        };
        // RaBitQ now rounds dim up to the next multiple of 64 internally
        // (encoder pads input vectors with zeros for the padded slots);
        // any dims > 0 are accepted. The padded slots add 0 to popcount
        // and 0 to ‖x‖, so codes stay comparable inside one index.

        // Seed derived from the configured dimensionality so two indexes with
        // the same shape get the same rotation across process restarts; this
        // keeps RaBitQ codes stable when the segment is reopened. Different
        // shards will switch to per-shard seeds in a follow-up.
        let seed = 0x9E37_79B9_7F4A_7C15u64 ^ self.config.max_dimensions as u64;

        // K-cluster IVF: run K-means Lloyd on the vectors present at
        // calibration time. For cosine workloads with cluster structure
        // (glove, sentence-transformers, OpenAI embeddings) residuals
        // against a per-cluster centroid are markedly tighter than
        // against a single global mean, so the sign-bit code captures
        // sharper direction information. K=16 matches the SIGMOD 2024
        // RaBitQ-Library reference. Calibration vectors are collected
        // by cloning the node vectors present at threshold; the K-means
        // upper bound of 12 iterations caps calibration latency well
        // under one second even at calibration_threshold = 100k.
        const N_CLUSTERS: u32 = 16;
        let training: Vec<Vec<f32>> = self.node_vectors.iter().filter_map(|v| v.clone()).collect();
        let params = if training.is_empty() {
            RaBitQParams::calibrate(dims as u32, seed)
        } else {
            RaBitQParams::calibrate_with_kmeans(dims as u32, seed, &training, N_CLUSTERS)
        };

        // Two-pass borrow split — encode_rabitq needs `&self.config`
        // while encoding, then we mutate node.rabitq_code separately.
        let encoded: Vec<(usize, Option<RabitqEncoded>)> = self
            .node_vectors
            .iter()
            .enumerate()
            .map(|(i, v_opt)| {
                let enc = v_opt
                    .as_deref()
                    .and_then(|v| self.encode_rabitq(&params, v));
                (i, enc)
            })
            .collect();
        for (i, enc) in encoded {
            self.mirror_rabitq_to_inline(i, enc.as_ref());
            self.node_rabitq_codes[i] = enc;
        }
        self.rabitq_params = Some(params);
    }

    /// Read-only access to the active configuration. Useful for
    /// bench harnesses that need to inspect M / ef_construction
    /// without holding a separate copy.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Override `ef_search` at runtime. The HNSW graph topology
    /// (M, ef_construction, layer assignments) does not depend on
    /// `ef_search` — it's a pure runtime knob that trades recall
    /// for latency. Mutating it between queries is safe.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.config.ef_search = ef;
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
    ///
    /// Internally a two-phase operation since C2 (R858b):
    /// 1. [`compute_insert_plan`] — read-only graph traversal that picks
    ///    neighbours for each layer (no mutation, takes `&self`).
    /// 2. [`apply_insert_plan`] — single-threaded mutation phase that
    ///    publishes the new node + neighbour edges.
    ///
    /// Batch ingestion (next C2 day) parallelises step 1 across rayon
    /// workers and then applies step 2 serially.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        if let Some(&idx) = self.id_to_idx.get(&id) {
            // Node already indexed — update vector and reconnect in graph.
            self.update_existing_node(idx, vector);
            return;
        }

        let plan = self.compute_insert_plan(id, &vector);
        self.apply_insert_plan(plan, vector);
    }

    /// Batched insert (C2, R858b). For `items.len() ≥ BATCH_PARALLEL_THRESHOLD`,
    /// planning runs across the rayon thread pool while apply remains
    /// single-threaded — composes with the wait-free C1 search hot path that
    /// the planning phase relies on.
    ///
    /// IDs already present in the index are routed through the sequential
    /// `update_existing_node` path after the parallel batch is applied;
    /// updates are rare in typical batch ingestion.
    ///
    /// Plans for the batch are computed against the pre-batch graph state.
    /// Inside a batch, a later insert's plan does not see earlier inserts
    /// from the same batch — acceptable for an approximate algorithm and
    /// the standard trade-off for batched HNSW construction (see hnswlib's
    /// `addPointsThreadPool`). Recall convergence is unaffected at typical
    /// batch sizes (≤ 1k); for larger batches, callers may chunk.
    ///
    /// Expected throughput: 5-8× over per-item `insert` on multi-core
    /// hardware (planning dominates ~80% of insert cost).
    pub fn insert_batch(&mut self, items: Vec<(u64, Vec<f32>)>) {
        // Threshold below which rayon overhead exceeds the parallelism win.
        // Tuned empirically; values from 4-32 perform equivalently on the
        // current bench host. 16 keeps small admin-style batches sequential.
        const BATCH_PARALLEL_THRESHOLD: usize = 16;

        // Seed density: until the graph holds this many nodes, batched
        // planning is unsafe because plans see a sparse / empty graph and
        // produce under-connected (in the limit: disconnected) nodes.
        // The seed phase inserts items one-by-one so each plan sees every
        // prior insert. Beyond this point a few-stale plans are acceptable
        // per the standard HNSW batch-construction trade-off.
        const SEED_DENSITY: usize = 64;

        // Dedupe within the batch — last-write-wins for repeated ids.
        // Without this, two `(5, vec_a)` and `(5, vec_b)` entries both
        // pass the `!contains_key(&id)` check (the index has no id=5
        // yet), both land in `inserts`, and the parallel apply creates
        // two `nodes[*].id == 5` entries while `id_to_idx[5]` records
        // only the second — diverging `nodes.len()` from logical
        // membership. Found by proptest (concurrent_proptest test
        // suite, 2026-05-22).
        let mut deduped: std::collections::HashMap<u64, Vec<f32>> =
            std::collections::HashMap::with_capacity(items.len());
        for (id, vec) in items {
            deduped.insert(id, vec);
        }

        // Partition into fresh inserts and updates of existing IDs.
        // Updates can't go through the plan/apply split because
        // `update_existing_node` rebuilds the node's edges in place.
        let mut updates = Vec::new();
        let mut inserts = Vec::new();
        for (id, vec) in deduped {
            if self.id_to_idx.contains_key(&id) {
                updates.push((id, vec));
            } else {
                inserts.push((id, vec));
            }
        }

        // Seed phase: bring graph up to SEED_DENSITY before batching.
        let mut iter = inserts.into_iter();
        while self.nodes.len() < SEED_DENSITY {
            match iter.next() {
                Some((id, vec)) => self.insert(id, vec),
                None => break,
            }
        }
        let remaining: Vec<(u64, Vec<f32>)> = iter.collect();

        // Parallel-plan, serial-apply for the remainder.
        let plans: Vec<(InsertPlan, Vec<f32>)> = if remaining.len() >= BATCH_PARALLEL_THRESHOLD {
            use rayon::prelude::*;
            remaining
                .into_par_iter()
                .map(|(id, vec)| {
                    let plan = self.compute_insert_plan(id, &vec);
                    (plan, vec)
                })
                .collect()
        } else {
            remaining
                .into_iter()
                .map(|(id, vec)| {
                    let plan = self.compute_insert_plan(id, &vec);
                    (plan, vec)
                })
                .collect()
        };

        // C3 day 4: the parallel apply path now runs a post-batch prune-
        // pass that backfills any back-edges dropped on capacity, so its
        // resulting graph holds the C2 recall contract (≥ 0.7 vs serial).
        // Dispatch to it for large batches; sequential apply for small.
        if plans.len() >= BATCH_PARALLEL_THRESHOLD {
            self.apply_insert_plans_parallel(plans);
        } else {
            for (plan, vec) in plans {
                self.apply_insert_plan(plan, vec);
            }
        }

        for (id, vec) in updates {
            // Routes to update_existing_node via the insert() shim.
            self.insert(id, vec);
        }
    }

    /// Read-only planning phase of an insert. Picks the new node's layer,
    /// runs the greedy descent + ef-search from the current entry point,
    /// and records the chosen neighbour set per layer.
    ///
    /// Takes `&self` — multiple concurrent callers can plan in parallel
    /// against the same (immutable) snapshot of the graph. The result is a
    /// pure-data [`InsertPlan`] that [`apply_insert_plan`] consumes from a
    /// single writer thread.
    pub(crate) fn compute_insert_plan(&self, id: u64, vector: &[f32]) -> InsertPlan {
        let new_level = self.random_level();

        // EntryPoint stays empty until the first insert lands —
        // `for_search()` returning None is the first-node fast path:
        // no graph traversal possible, plan records empty neighbour
        // sets so apply just pushes the seed node. Otherwise the
        // single-load snapshot gives `(start_idx, top_level)` from
        // ONE atomic read.
        let Some((start_idx, top_level)) = self.entry_point.for_search() else {
            return InsertPlan {
                id,
                new_level,
                per_layer: Vec::new(),
                is_first_node: true,
            };
        };
        let mut current_ep = start_idx;

        // Phase 1: greedy descent down to new_level + 1.
        //
        // Use the BUILD variant so neighbour selection runs on exact
        // f32 distance, not the RaBitQ popcount estimate. With RaBitQ
        // active in compute_distance, every `search_layer_*` call in
        // build path would score candidates by `2·pop/D` — a noisy
        // monotonic-but-not-equal function of the true cosine — and
        // pick neighbours whose ranking is corrupted by that noise.
        // At N≥10⁵ the cumulative error makes the graph unrecoverable
        // (no ef_search can find the true top-K). See `QueryCtx::new_
        // for_build` doc comment for the full reasoning.
        for level in (new_level + 1..=top_level).rev() {
            current_ep = self.search_layer_greedy_query_for_build(vector, current_ep, level);
        }

        // Phase 2: select neighbours at every layer from new_level down to 0.
        let lowest_planning_layer = new_level.min(top_level);
        let mut per_layer = Vec::with_capacity(lowest_planning_layer + 1);
        for level in (0..=lowest_planning_layer).rev() {
            let ef = self.config.ef_construction;
            // Same f32-build rationale as Phase 1 above.
            let candidates = self.search_layer_query_for_build(vector, current_ep, ef, level);

            let max_conn = if level == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };
            // RobustPrune when alpha > 1.0; legacy "take M closest"
            // otherwise. Both feed the same downstream insert plan path.
            let selected: Vec<usize> = if self.config.alpha_pruning > 1.0 {
                self.select_neighbours_robust_prune(&candidates, max_conn)
            } else {
                candidates
                    .into_iter()
                    .take(max_conn)
                    .map(|c| c.idx as usize)
                    .collect()
            };

            if !selected.is_empty() {
                current_ep = selected[0];
            }

            per_layer.push(LayerPlan {
                level,
                selected_idxs: selected,
                max_conn,
            });
        }

        InsertPlan {
            id,
            new_level,
            per_layer,
            is_first_node: false,
        }
    }

    /// Mutation phase of an insert. Pushes the new node, allocates atomic
    /// neighbour storage, then publishes outgoing + bidirectional edges
    /// from the plan via the write helpers (`set_outgoing` /
    /// `add_neighbour_to`). Single-writer — caller holds `&mut self`.
    /// Mirror `(id, vector)` for node `idx` into the contiguous layer-0
    /// store. Lazy-allocates the store on the first call once `dim` is
    /// known; subsequent calls reuse it. No-op when the store is already
    /// allocated for a different dim (programmer error elsewhere) or when
    /// `idx` overruns the pre-allocated capacity; this keeps the SoA path
    /// authoritative until the search-side switch lands.
    ///
    /// Neighbour ids and RaBitQ codes are NOT written here; they hook in
    /// from the dedicated CAS neighbour-write path and from
    /// `auto_calibrate_rabitq` respectively, in follow-up commits on the
    /// same plan. Writing them now without the matching reads would just
    /// double the build-time cost.
    fn mirror_inline_layer0(&mut self, idx: usize, id: u64, vector: &[f32]) {
        let dim = vector.len();
        if dim == 0 {
            return;
        }
        if self.inline_layer0.is_none() {
            let capacity = (self.config.max_elements as usize).max(idx + 1);
            // Pick the RaBitQ packed-code width from the configured codec
            // so a later calibration can mirror into the contiguous store
            // without a layout mismatch. Non-RaBitQ configs use 1 bit (the
            // minimum legal value) since the code slot stays unused.
            let rabitq_bits = match self.config.quantization {
                QuantizationCodec::RaBitQ { bits } => bits,
                _ => 1,
            };
            self.inline_layer0 = Some(inline_layer0::InlineLayer0::new_with_rabitq_bits(
                capacity,
                M_MAX0,
                dim,
                rabitq_bits,
            ));
        }
        let Some(inline) = self.inline_layer0.as_mut() else {
            return;
        };
        if inline.dim() != dim || idx >= inline.capacity() {
            return;
        }
        // SAFETY: idx < capacity (checked above), vector.len() == dim
        // (checked above), and we hold &mut self so no other reader
        // observes the partial write.
        unsafe {
            inline.set_label(idx, id);
            inline.set_vector_f32(idx, vector);
        }
    }

    /// Borrow the per-node f32 vector for `idx`. Prefers the SoA
    /// `node_vectors` slot because measurements on real cosine and
    /// euclidean workloads (sift-128, glove-100, M=16) showed the
    /// contiguous per-node block layout regresses f32 search QPS
    /// 13-16% under both ST and MT4 versus reading from SoA. The
    /// contiguous store still owns the f32 mirror for the RaBitQ
    /// search path (which benefits from co-locating code + scalars)
    /// and as a fallback for nodes whose SoA slot is unavailable.
    /// Returns `None` when neither path holds an f32 payload
    /// (offloaded SQ8 or pre-insert states); callers that need a
    /// non-`None` value already handle the `None` arm.
    #[inline]
    fn read_node_f32(&self, idx: usize) -> Option<&[f32]> {
        if let Some(vec) = self.node_vectors.get(idx).and_then(|v| v.as_deref()) {
            return Some(vec);
        }
        if let Some(inline) = self.inline_layer0.as_ref() {
            if idx < inline.capacity() {
                // SAFETY: idx < capacity per the gate above; payload
                // bytes were installed under &mut self and we only ever
                // take &self after that.
                unsafe {
                    return Some(inline.vector_f32(idx));
                }
            }
        }
        None
    }

    /// Read the layer-0 neighbour id snapshot into `out` from the
    /// contiguous store when present, falling back to the SoA
    /// `AtomicNeighbourList` otherwise. Keeps the search hot path on a
    /// single allocation per node so concurrent workers visiting the same
    /// graph share L3 fills, the cache-locality property that drives
    /// hnswlib's super-linear MT4 scaling on sift-128 f32.
    #[inline]
    fn read_layer0_neighbours_into(&self, idx: usize, out: &mut Vec<u64>) {
        out.clear();
        if let Some(inline) = self.inline_layer0.as_ref() {
            if idx < inline.capacity() {
                // SAFETY: idx < inline.capacity() by the gate above; slots
                // bounded by inline.m_max0() which the constructor enforces
                // is the in-bounds slot count.
                unsafe {
                    let raw_len = inline
                        .neighbour_len(idx)
                        .load(core::sync::atomic::Ordering::Relaxed)
                        as usize;
                    let len = raw_len.min(inline.m_max0());
                    out.reserve(len);
                    for slot in 0..len {
                        let id = inline
                            .neighbour(idx, slot)
                            .load(core::sync::atomic::Ordering::Relaxed);
                        out.push(id);
                    }
                }
                return;
            }
        }
        self.neighbours_l0[idx].snapshot_into(out);
    }

    /// Mirror the RaBitQ code (packed bytes) and scalar header for node
    /// `idx` into the contiguous store. Skips silently when:
    /// - the contiguous store is not allocated yet,
    /// - the store's `rabitq_bits` does not match the active codec
    ///   (e.g. lazy alloc happened before calibration set bits),
    /// - `idx` exceeds the pre-allocated capacity, or
    /// - the encoded variant is `None`.
    ///
    /// SoA remains authoritative for search reads, so a missed mirror is
    /// a perf miss, not a correctness bug.
    fn mirror_rabitq_to_inline(&mut self, idx: usize, enc: Option<&RabitqEncoded>) {
        let Some(enc) = enc else { return };
        let Some(inline) = self.inline_layer0.as_mut() else {
            return;
        };
        if idx >= inline.capacity() {
            return;
        }
        match enc {
            RabitqEncoded::OneBit(code) => {
                if inline.rabitq_bits() != 1 {
                    return;
                }
                // `CodeWords` derefs to `&[u64]`; reinterpret as bytes for
                // the packed-code slot.
                let words: &[u64] = code.code.as_slice();
                let byte_len = core::mem::size_of_val(words);
                // SAFETY: `words` outlives the slice borrow; reinterpreting
                // an aligned `&[u64]` as `&[u8]` is sound.
                let byte_slice =
                    unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, byte_len) };
                let scalars = inline_layer0::RaBitQScalars {
                    norm: code.norm,
                    cross_term: code.cross_term,
                    signed_sum: code.signed_sum,
                    correction: code.correction,
                    radial: code.radial,
                    cluster_id: code.cluster_id,
                    _pad: 0,
                };
                // SAFETY: idx < capacity per the gate above; the byte
                // count is capped at the inline rabitq slot length.
                unsafe {
                    let dst_len = inline.rabitq(idx).len();
                    let take = byte_slice.len().min(dst_len);
                    if take > 0 {
                        let mut tmp = vec![0u8; dst_len];
                        tmp[..take].copy_from_slice(&byte_slice[..take]);
                        inline.set_rabitq(idx, &tmp);
                    }
                    inline.set_rabitq_scalars(idx, scalars);
                }
            }
            RabitqEncoded::Multi(code) => {
                if inline.rabitq_bits() != code.bits {
                    return;
                }
                let scalars = inline_layer0::RaBitQScalars {
                    norm: code.norm,
                    cross_term: code.cross_term,
                    signed_sum: 0,
                    correction: 0.0,
                    radial: 0.0,
                    cluster_id: 0,
                    _pad: 0,
                };
                // SAFETY: idx < capacity; `code.packed` is the canonical
                // byte layout for the configured (dim, bits) pair.
                unsafe {
                    let dst_len = inline.rabitq(idx).len();
                    if code.packed.len() == dst_len {
                        inline.set_rabitq(idx, &code.packed);
                    }
                    inline.set_rabitq_scalars(idx, scalars);
                }
            }
        }
    }

    /// Read-only view of the contiguous layer-0 store, for parity tests
    /// and the follow-up search-side switch.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "consumer of this accessor lands with the search-side switch on the same plan"
        )
    )]
    pub(crate) fn inline_layer0(&self) -> Option<&inline_layer0::InlineLayer0> {
        self.inline_layer0.as_ref()
    }

    /// Refresh the layer-0 neighbour ids and `neighbour_len` for node
    /// `idx` in the contiguous store from the authoritative SoA
    /// `AtomicNeighbourList`. Called from every layer-0 mutator
    /// (`set_outgoing`, `cas_add_neighbour_to`, `add_neighbour_to`) so the
    /// inline store stays in sync with SoA. Search still reads SoA, so
    /// brief transient divergence under concurrent back-edge writers
    /// resolves itself on the next mirror call from any writer; the
    /// search-side switch on the same plan adds the synchronisation that
    /// removes the race entirely.
    fn mirror_layer0_neighbours_to_inline(&self, idx: usize) {
        let Some(inline) = self.inline_layer0.as_ref() else {
            return;
        };
        if idx >= inline.capacity() {
            return;
        }
        let list = &self.neighbours_l0[idx];
        let mut snap: Vec<u64> = Vec::with_capacity(M_MAX0);
        list.snapshot_into(&mut snap);
        let take = snap.len().min(inline.m_max0());
        let len_byte = u8::try_from(take).unwrap_or(u8::MAX);
        // SAFETY: idx < inline.capacity() per the gate above. Slot writes
        // are bounded by inline.m_max0() which the constructor guarantees
        // is the in-bounds slot count.
        unsafe {
            for (slot, &id) in snap.iter().enumerate().take(take) {
                inline.set_neighbour(idx, slot, id);
            }
            for slot in take..inline.m_max0() {
                inline.set_neighbour(idx, slot, 0);
            }
            inline.set_neighbour_len(idx, len_byte);
        }
    }

    pub(crate) fn apply_insert_plan(&mut self, plan: InsertPlan, vector: Vec<f32>) {
        let InsertPlan {
            id,
            new_level,
            per_layer,
            is_first_node,
        } = plan;
        let idx = self.nodes.len();

        // Quantize if SQ8 is calibrated.
        let quantized = self.sq8_params.as_ref().map(|p| p.quantize(&vector));
        // Encode if RaBitQ is calibrated. Encoded against the rotation matrix
        // already chosen at calibration time — codes from before vs after
        // calibration are not interchangeable, so this branch only fires
        // post-calibration (pre-calibration nodes get a code on first
        // calibration via `auto_calibrate_rabitq`). Variant is picked
        // from config so 1-bit vs 2/3/4-bit indexes stay homogeneous.
        let rabitq_code = self
            .rabitq_params
            .as_ref()
            .and_then(|p| self.encode_rabitq(p, &vector));

        // Persist f32 truth tier per ADR-033. Quantized codes (SQ8 /
        // RaBitQ / PolarQuant / PQ) stay in RAM only — Phase 1.5
        // cross-shard rerank reads f32 directly from the truth tier.
        // Tier writes never roll back the in-RAM insert — the in-RAM
        // graph is authoritative; the truth tier regenerates from
        // data on recovery (replication.md HNSW rebuild path).
        if let Some(tier) = self.vector_tier.as_ref() {
            if let Err(e) = tier.put_f32(id, &vector) {
                warn!(node_id = id, error = %e, "vector_tier put_f32 failed");
            }
        }

        self.nodes.push(HnswNode {
            id,
            max_layer: new_level,
        });
        // Mirror into the contiguous layer-0 store before moving `vector`.
        // No-op until the field has been allocated AND we are inside
        // capacity; the SoA path below remains authoritative either way.
        self.mirror_inline_layer0(idx, id, &vector);
        // SoA payload pushes in lockstep — same idx, no extra clone.
        self.node_vectors.push(Some(vector));
        self.node_quantized.push(quantized);
        let rabitq_for_mirror = rabitq_code.clone();
        self.node_rabitq_codes.push(rabitq_code);
        let mirror_idx = self.node_rabitq_codes.len() - 1;
        self.mirror_rabitq_to_inline(mirror_idx, rabitq_for_mirror.as_ref());
        self.id_to_idx.insert(id, idx);

        // Allocate atomic neighbour storage in lockstep — write helpers
        // index by (node, layer) and would panic on a missing entry.
        // Layer 0 goes in the flat hot vec; upper layers (if any) in the
        // cold per-node Vec.
        self.neighbours_l0.push(AtomicNeighbourList::new());
        let mut upper = Vec::with_capacity(new_level);
        for _ in 0..new_level {
            upper.push(AtomicNeighbourList::new());
        }
        self.neighbours_upper.push(upper);

        if is_first_node {
            // First insert seeds the entry-point. try_promote on a
            // fresh EntryPoint always succeeds — no other writer
            // has touched it yet, and we're holding &mut self.
            let _ = self.entry_point.try_promote(new_level as u8, idx as u64);
            self.maybe_calibrate_and_offload(idx);
            return;
        }

        for layer in per_layer {
            let LayerPlan {
                level,
                selected_idxs,
                max_conn,
            } = layer;

            // Outgoing: new_node → selected. We store internal indices
            // (`idx`), not external `NodeId`s, so the search hot path can
            // skip an `id_to_idx` HashMap lookup per neighbour visit (the
            // dominant cost per profiler — see commit message).
            let outgoing: Vec<u64> = selected_idxs.iter().map(|&n| n as u64).collect();
            self.set_outgoing(idx, level, &outgoing);

            // Bidirectional: selected → new_node (with optional prune).
            for &neighbor_idx in &selected_idxs {
                if level < self.node_levels(neighbor_idx) {
                    self.add_neighbour_to(neighbor_idx, level, idx as u64, max_conn);
                }
            }
        }

        // Promote the new node to entry-point if it pierced a new top
        // layer. CAS-loop returns `NotNeeded` when another insert has
        // already promoted past us (possible from a concurrent batch
        // arriving at try_promote first); in that case we leave the
        // existing higher-layer entry-point alone.
        let _ = self.entry_point.try_promote(new_level as u8, idx as u64);

        // SQ8 calibrate + offload happens after the topology is in place
        // so search_layer can use f32 for the just-inserted node.
        self.maybe_calibrate_and_offload(idx);
    }

    /// Parallel apply phase for C3 — applies many plans through a
    /// (serial allocation, parallel edge-write) two-step.
    ///
    /// Step 1 (serial, `&mut self`):
    ///   * push each new node into `nodes` + allocate matching atomic
    ///     neighbour layers in `neighbours_l0` / `neighbours_upper`;
    ///   * register the id → idx mapping;
    ///   * promote `entry_point` / `max_level` if a plan's `new_level`
    ///     pierces a new top.
    ///
    /// Step 2 (parallel, `&self` via `rayon::par_iter`):
    ///   * each thread takes one allocated `(plan, idx)` pair and writes
    ///     `set_outgoing` for the new node (conflict-free across distinct
    ///     `idx`) plus `cas_add_neighbour_to` for each chosen back-edge
    ///     (multi-writer-safe through `AtomicNeighbourList::cas_append`).
    ///   * if a back-edge target is at capacity, this commit drops the
    ///     edge silently — the prune-pass that fills these in is C3
    ///     day 4. Recall hit is the standard hnswlib batch trade-off.
    ///
    /// Step 3 (serial): call `maybe_calibrate_and_offload` once for the
    /// last-allocated node; SQ8 calibration sees the post-batch state.
    pub(crate) fn apply_insert_plans_parallel(&mut self, plans: Vec<(InsertPlan, Vec<f32>)>) {
        if plans.is_empty() {
            return;
        }

        // Step 1 — serial allocation phase.
        let mut allocated: Vec<(InsertPlan, usize)> = Vec::with_capacity(plans.len());
        for (plan, vec) in plans {
            let idx = self.nodes.len();
            let quantized = self.sq8_params.as_ref().map(|p| p.quantize(&vec));
            let rabitq_code = self
                .rabitq_params
                .as_ref()
                .and_then(|p| self.encode_rabitq(p, &vec));
            let new_level = plan.new_level;

            // Persist f32 truth tier per ADR-033 (mirrors apply_insert_plan).
            if let Some(tier) = self.vector_tier.as_ref() {
                if let Err(e) = tier.put_f32(plan.id, &vec) {
                    warn!(node_id = plan.id, error = %e, "vector_tier put_f32 failed");
                }
            }

            self.nodes.push(HnswNode {
                id: plan.id,
                max_layer: new_level,
            });
            // Mirror into the contiguous layer-0 store before moving `vec`.
            self.mirror_inline_layer0(idx, plan.id, &vec);
            self.node_vectors.push(Some(vec));
            self.node_quantized.push(quantized);
            let rabitq_for_mirror = rabitq_code.clone();
            self.node_rabitq_codes.push(rabitq_code);
            let mirror_idx = self.node_rabitq_codes.len() - 1;
            self.mirror_rabitq_to_inline(mirror_idx, rabitq_for_mirror.as_ref());
            self.id_to_idx.insert(plan.id, idx);

            self.neighbours_l0.push(AtomicNeighbourList::new());
            let mut upper = Vec::with_capacity(new_level);
            for _ in 0..new_level {
                upper.push(AtomicNeighbourList::new());
            }
            self.neighbours_upper.push(upper);

            // Entry-point promotion through the lock-free CAS-loop.
            // The first insert (`nodes.len() == 1`) hits an empty
            // EntryPoint and unconditionally installs; every later
            // plan only wins when its `new_level` strictly exceeds the
            // current top. Either way the post-condition matches the
            // arch doc's layer-promotion linearisability invariant:
            // entry-point sits at the global max layer after the call
            // returns. Still runs serially within this batch's
            // allocation phase so the parallel writers below observe
            // a consistent entry-point, but the primitive is now
            // safe under cross-batch / cross-thread races too.
            let _ = self.entry_point.try_promote(new_level as u8, idx as u64);

            allocated.push((plan, idx));
        }

        // Step 2 — parallel edge writes. Failed back-edge appends are
        // collected into a Mutex<Vec> so the day-4 prune-pass can backfill
        // them serially under &mut self. The mutex is only contended on
        // overflow (rare under typical workloads), so the parallel write
        // path is still effectively wait-free in the hot case.
        use rayon::prelude::*;
        let backfill: std::sync::Mutex<Vec<(usize, usize, u64, usize)>> =
            std::sync::Mutex::new(Vec::new());
        allocated.par_iter().for_each(|(plan, idx)| {
            if plan.is_first_node {
                return;
            }
            for layer in &plan.per_layer {
                // Store internal indices in neighbour lists — search hot
                // path reads them directly without an id→idx HashMap hop.
                let outgoing: Vec<u64> = layer.selected_idxs.iter().map(|&n| n as u64).collect();
                self.set_outgoing(*idx, layer.level, &outgoing);

                for &neighbour_idx in &layer.selected_idxs {
                    if layer.level < self.node_levels(neighbour_idx)
                        && !self.cas_add_neighbour_to(neighbour_idx, layer.level, *idx as u64)
                    {
                        // cas_append returned false (list at capacity).
                        // Record (neighbour_idx, level, idx_to_add, max_conn)
                        // for the serial prune-pass below.
                        backfill.lock().unwrap_or_else(|e| e.into_inner()).push((
                            neighbour_idx,
                            layer.level,
                            *idx as u64,
                            layer.max_conn,
                        ));
                    }
                }
            }
        });

        // Step 3 — parallel prune-pass with dedupe.
        //
        // Hub-vertex amplification: when many new nodes pick the same
        // hot neighbour, the lossy parallel phase fills the
        // per-neighbour backfill bucket K times. We group by
        // (neighbour_idx, level) so each unique list pays prune cost
        // once and then cas_appends all K queued ids in one batch:
        //
        //   O(K × prune) → O(prune + K) per hot list.
        //
        // Then the groups themselves are disjoint per neighbour_idx, so
        // we run prune-pass via rayon par_iter — each thread touches a
        // distinct neighbour list at (X, Y), no contention. This lifts
        // the serial floor that would otherwise cap parallel speedup
        // under Amdahl's law when hot vertices saturate.
        let mut backfill = backfill.into_inner().unwrap_or_else(|e| e.into_inner());
        backfill.sort_unstable_by_key(|&(nb, lvl, _, _)| (nb, lvl));

        // Materialise the groups as `(neighbour_idx, level, max_conn,
        // start..end)` so par_iter can dispatch each independently. We
        // slice into the shared `backfill` vec below — no per-group
        // allocation of ids.
        let mut groups: Vec<(usize, usize, usize, std::ops::Range<usize>)> = Vec::new();
        let mut i = 0;
        while i < backfill.len() {
            let (neighbour_idx, level, _, max_conn) = backfill[i];
            let mut j = i + 1;
            while j < backfill.len() && backfill[j].0 == neighbour_idx && backfill[j].1 == level {
                j += 1;
            }
            groups.push((neighbour_idx, level, max_conn, i..j));
            i = j;
        }

        let backfill_ref = &backfill;
        groups
            .par_iter()
            .for_each(|(neighbour_idx, level, max_conn, range)| {
                // Collect the backfilled candidate ids for this
                // (neighbour_idx, level) group and let the prune pass
                // choose top-max_conn across the union of current
                // neighbours and these queued candidates. The earlier
                // "prune then cas_append" sequence dropped every backfill
                // candidate because prune truncated *at* max_conn, leaving
                // no room for the subsequent appends — the new closer
                // candidates were silently discarded even when they
                // should have replaced farther incumbents.
                let extras: Vec<u64> = backfill_ref[range.clone()].iter().map(|e| e.2).collect();
                self.prune_connections_with_extras(*neighbour_idx, *level, *max_conn, &extras);
            });

        // Step 4 — serial post-phase. SQ8 calibration sees the final
        // population. We call once with the last allocated idx; the
        // calibration path itself looks at `self.nodes` as a whole.
        if let Some((_, last_idx)) = allocated.last() {
            self.maybe_calibrate_and_offload(*last_idx);
        }
    }

    /// Auto-calibrate SQ8 if threshold reached, then offload the just-inserted
    /// node's f32 if offloading is active. Called at the end of `insert()`.
    fn maybe_calibrate_and_offload(&mut self, just_inserted_idx: usize) {
        // Step 1: Auto-calibrate the configured codec when threshold reached.
        let threshold_reached = self.nodes.len() >= self.config.calibration_threshold;
        match self.config.quantization {
            QuantizationCodec::Sq8 if self.sq8_params.is_none() && threshold_reached => {
                self.auto_calibrate();
            }
            QuantizationCodec::RaBitQ { .. }
                if self.rabitq_params.is_none() && threshold_reached =>
            {
                self.auto_calibrate_rabitq();
            }
            _ => {}
        }

        // Step 2: Offload f32 of the just-inserted node (construction complete).
        // Offload is only valid for SQ8 today; RaBitQ disk-rerank with SQ8 on
        // disk is the R861 follow-up.
        if self.is_offloaded() {
            self.node_vectors[just_inserted_idx] = None;
        }
    }

    /// Search for K nearest neighbors.
    ///
    /// When SQ8 quantization is active, HNSW traversal uses approximate
    /// (dequantized) distances for candidate generation. The final top-K
    /// results are reranked using exact f32 distances.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        // Cache query-side state once per search — for Cosine the query norm
        // would otherwise be recomputed on every distance call (hundreds of
        // times per level). Other metrics ignore the cached norm.
        let qctx = QueryCtx::new(
            query,
            self.config.metric,
            self.rabitq_params.as_ref(),
            &self.config.quantization,
        );

        // Single-load entry-point snapshot — `for_search()` returns
        // `(start_idx, top_level)` from ONE atomic read, so a
        // concurrent `try_promote` cannot split idx and level into
        // separate snapshots mid-search. The None branch IS the
        // empty-index early-return (EntryPoint stays empty until
        // the first insert lands).
        let Some((start_idx, top_level)) = self.entry_point.for_search() else {
            return Vec::new();
        };
        let mut current_ep = start_idx;

        // Traverse from top to layer 1 (greedy)
        for level in (1..=top_level).rev() {
            current_ep = self.search_layer_greedy_query(query, current_ep, level);
        }

        // Effective layer-0 beam must be at least `k`. Without this floor
        // a caller passing `ef_search < k` gets both a truncated top-k AND
        // catastrophic recall — the visited pool is too small to reach the
        // true k-NN. Standard HNSW (Malkov 2018, hnswlib, Qdrant, FAISS)
        // all enforce this.
        let mut ef = self.config.ef_search.max(k);
        if self.is_quantized() {
            ef = ef.max(self.config.rerank_candidates);
        }

        // Search at layer 0 with ef candidates
        let candidates = self.search_layer_query(query, current_ep, ef, 0);

        if self.is_quantized() {
            // Rerank candidates using exact f32 distance
            let mut reranked: Vec<SearchResult> = candidates
                .into_iter()
                .map(|c| {
                    let exact_dist = self.compute_exact_distance(&qctx, c.idx as usize);
                    SearchResult {
                        id: self.nodes[c.idx as usize].id,
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
                    id: self.nodes[c.idx as usize].id,
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

        // Cache query-side state once per search — same rationale as `search`.
        let qctx = QueryCtx::new(
            query,
            self.config.metric,
            self.rabitq_params.as_ref(),
            &self.config.quantization,
        );

        // Single-load entry-point snapshot doubles as the empty-index
        // guard — see `search` above for the consistency rationale.
        let Some((start_idx, top_level)) = self.entry_point.for_search() else {
            return (Vec::new(), stats);
        };
        let mut current_ep = start_idx;

        // Traverse from top to layer 1 (greedy)
        for level in (1..=top_level).rev() {
            current_ep = self.search_layer_greedy_ctx(&qctx, current_ep, level);
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

            let candidates = self.search_layer_ctx(&qctx, current_ep, current_ef, 0);

            // Convert to SearchResult with exact distances (rerank if quantized)
            let results: Vec<SearchResult> = if self.is_quantized() {
                let mut reranked: Vec<SearchResult> = candidates
                    .into_iter()
                    .map(|c| SearchResult {
                        id: self.nodes[c.idx as usize].id,
                        score: self.compute_exact_distance(&qctx, c.idx as usize),
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
                        id: self.nodes[c.idx as usize].id,
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

        let qctx = QueryCtx::new(
            query,
            self.config.metric,
            self.rabitq_params.as_ref(),
            &self.config.quantization,
        );

        // Single-load entry-point snapshot doubles as the empty-index
        // guard — see `search` above for the consistency rationale.
        let Some((start_idx, top_level)) = self.entry_point.for_search() else {
            return Vec::new();
        };
        let mut current_ep = start_idx;

        for level in (1..=top_level).rev() {
            current_ep = self.search_layer_greedy_ctx(&qctx, current_ep, level);
        }

        // See `search()` for why the beam floor at `k` is mandatory.
        let ef = self
            .config
            .ef_search
            .max(self.config.rerank_candidates)
            .max(k);
        let candidates = self.search_layer_ctx(&qctx, current_ep, ef, 0);

        // Batch-load f32 vectors from storage for reranking
        let candidate_ids: Vec<u64> = candidates
            .iter()
            .map(|c| self.nodes[c.idx as usize].id)
            .collect();
        let loaded = loader.load_vectors(&candidate_ids, &self.config.property_name);

        let mut reranked: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|c| {
                let node_id = self.nodes[c.idx as usize].id;
                let f32_vec = loaded.get(&node_id)?;
                let exact_dist = self.distance_for_metric(&qctx, f32_vec);
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

        let qctx = QueryCtx::new(
            query,
            self.config.metric,
            self.rabitq_params.as_ref(),
            &self.config.quantization,
        );

        // Single-load entry-point snapshot doubles as the empty-index
        // guard — see `search` above for the consistency rationale.
        let Some((start_idx, top_level)) = self.entry_point.for_search() else {
            return (Vec::new(), stats);
        };
        let mut current_ep = start_idx;

        for level in (1..=top_level).rev() {
            current_ep = self.search_layer_greedy_ctx(&qctx, current_ep, level);
        }

        let mut visible_results: Vec<SearchResult> = Vec::new();
        let base_ef = self.config.ef_search.max(self.config.rerank_candidates);
        let mut current_ef = ((k as f64 * overfetch_factor).ceil() as usize).max(base_ef);

        for round in 0..=max_expansion_rounds {
            stats.expansion_rounds = round;
            let candidates = self.search_layer_ctx(&qctx, current_ep, current_ef, 0);

            // Batch-load f32 for reranking
            let candidate_ids: Vec<u64> = candidates
                .iter()
                .map(|c| self.nodes[c.idx as usize].id)
                .collect();
            let loaded = loader.load_vectors(&candidate_ids, &self.config.property_name);

            let mut results: Vec<SearchResult> = candidates
                .into_iter()
                .filter_map(|c| {
                    let node_id = self.nodes[c.idx as usize].id;
                    let f32_vec = loaded.get(&node_id)?;
                    let exact_dist = self.distance_for_metric(&qctx, f32_vec);
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
        let n_levels = self.node_levels(idx);

        // Step 1: Remove this node from every neighbour's connection list.
        // Snapshot first to avoid simultaneous mutable + immutable borrows.
        for level in 0..n_levels {
            let neighbours = self.neighbours_at(idx, level).snapshot();
            for neighbour_idx_u64 in neighbours {
                let neighbour_idx = neighbour_idx_u64 as usize;
                if neighbour_idx < self.nodes.len() && level < self.node_levels(neighbour_idx) {
                    self.remove_neighbour_from(neighbour_idx, level, idx as u64);
                }
            }
        }

        // Step 2: Clear this node's outgoing connections (keep layer slot count).
        for level in 0..n_levels {
            self.clear_outgoing(idx, level);
        }

        // Step 3: Update vector and quantized representation.
        let quantized = self.sq8_params.as_ref().map(|p| p.quantize(&vector));
        let rabitq_code = self
            .rabitq_params
            .as_ref()
            .and_then(|p| self.encode_rabitq(p, &vector));

        // Overwrite f32 truth tier on existing-node update so the
        // tier reflects the latest write (ADR-033).
        if let Some(tier) = self.vector_tier.as_ref() {
            if let Err(e) = tier.put_f32(id, &vector) {
                warn!(node_id = id, error = %e, "vector_tier put_f32 on update failed");
            }
        }

        // Refresh the contiguous-store payload alongside the SoA write so
        // a subsequent search reads the updated vector and not a stale
        // mirror left over from the original insert.
        self.mirror_inline_layer0(idx, id, &vector);
        self.node_vectors[idx] = Some(vector);
        self.node_quantized[idx] = quantized;
        let rabitq_for_mirror = rabitq_code.clone();
        self.node_rabitq_codes[idx] = rabitq_code;
        self.mirror_rabitq_to_inline(idx, rabitq_for_mirror.as_ref());

        // Step 4: Re-insert into the graph from a valid entry point.
        // A single-node index has no connections to rebuild.
        if self.nodes.len() == 1 {
            if self.is_offloaded() {
                self.node_vectors[idx] = None;
            }
            return;
        }

        // Choose entry point: if the current entry_point IS this node, use any
        // other node (the graph is connected, so any peer suffices).
        // Single load: top_level and ep idx come from one snapshot.
        let (ep_idx, top_level) = match self.entry_point.for_search() {
            Some((ep, lvl)) if ep == idx => {
                // Self is the entry-point — pick any peer.
                let peer = self
                    .id_to_idx
                    .values()
                    .find(|&&i| i != idx)
                    .copied()
                    .unwrap_or(0);
                (peer, lvl)
            }
            Some((ep, lvl)) => (ep, lvl),
            // Empty index — preserved by the prior `nodes[idx]` access
            // which would have panicked already if the graph was empty.
            // Defensive fallback so the function is total.
            None => (0, 0),
        };

        let node_level = self.nodes[idx].max_layer;
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
                .map(|c| c.idx as usize)
                .collect();

            // Connect this node to its new neighbours. Store internal
            // indices (idx, not NodeId) so the search hot path skips the
            // id_to_idx HashMap hop.
            let outgoing: Vec<u64> = selected.iter().map(|&n| n as u64).collect();
            self.set_outgoing(idx, level, &outgoing);

            // Connect neighbours back (bi-directional).
            for &neighbour_idx in &selected {
                if level < self.node_levels(neighbour_idx) {
                    self.add_neighbour_to(neighbour_idx, level, idx as u64, max_conn);
                }
            }

            if !selected.is_empty() {
                current_ep = selected[0];
            }
        }

        // Step 5: Offload f32 if offloading is active (calibration already done).
        if self.is_offloaded() {
            self.node_vectors[idx] = None;
        }
    }

    /// Greedy search on a single layer (for traversal from top layers).
    /// The node at `query_idx` must have an f32 vector (only used during insert
    /// where the new node always has f32 available). Falls back to dequantized
    /// SQ8 if f32 was unexpectedly dropped.
    fn search_layer_greedy(&self, query_idx: usize, ep: usize, level: usize) -> usize {
        let query_vec = self.get_node_f32_or_dequantized(query_idx);
        // Build-path wrapper: callers are existing-node insert/reconnect
        // loops, never search. Force f32 to keep graph quality consistent
        // with the rest of construction (see `QueryCtx::new_for_build`).
        self.search_layer_greedy_query_for_build(&query_vec, ep, level)
    }

    fn search_layer_greedy_query(&self, query: &[f32], ep: usize, level: usize) -> usize {
        let ctx = QueryCtx::new(
            query,
            self.config.metric,
            self.rabitq_params.as_ref(),
            &self.config.quantization,
        );
        self.search_layer_greedy_ctx(&ctx, ep, level)
    }

    /// Build-path variant of [`Self::search_layer_greedy_query`] that
    /// forces exact f32 distance for every comparison. See
    /// [`QueryCtx::new_for_build`] for why build must not use RaBitQ.
    fn search_layer_greedy_query_for_build(&self, query: &[f32], ep: usize, level: usize) -> usize {
        let ctx = QueryCtx::new_for_build(query, self.config.metric);
        self.search_layer_greedy_ctx(&ctx, ep, level)
    }

    fn search_layer_greedy_ctx(&self, ctx: &QueryCtx<'_>, ep: usize, level: usize) -> usize {
        let mut current = ep;
        let mut current_dist = self.compute_distance(ctx, current);

        // Per-call scratch buffer for the atomic neighbour snapshot. The
        // BFS reads at most one neighbour list per loop iteration, so we
        // recycle this single allocation across iterations rather than
        // pay per-level allocation cost.
        let mut neighbours_scratch: Vec<u64> = Vec::with_capacity(M_MAX0);

        loop {
            let mut changed = false;
            // Lock-free read from the atomic mirror. snapshot_into is
            // wait-free: one Acquire load on `len` + that many Relaxed
            // loads on the slot array. No mutex, no Arc clone, no lock
            // contention with concurrent readers.
            if level < self.node_levels(current) {
                self.neighbours_at(current, level)
                    .snapshot_into(&mut neighbours_scratch);
                for &neighbor_idx_u64 in &neighbours_scratch {
                    // Stored u64s ARE internal indices now (HNSW search hot
                    // path: no id_to_idx HashMap hop per neighbour).
                    let neighbor_idx = neighbor_idx_u64 as usize;
                    if neighbor_idx < self.nodes.len() {
                        let dist = self.compute_distance(ctx, neighbor_idx);
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
        // Build-path wrapper — see `search_layer_greedy` for rationale.
        self.search_layer_query_for_build(&query_vec, ep, ef, level)
    }

    fn search_layer_query(
        &self,
        query: &[f32],
        ep: usize,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let ctx = QueryCtx::new(
            query,
            self.config.metric,
            self.rabitq_params.as_ref(),
            &self.config.quantization,
        );
        self.search_layer_ctx(&ctx, ep, ef, level)
    }

    /// Build-path variant of [`Self::search_layer_query`] that forces
    /// exact f32 distance. See [`QueryCtx::new_for_build`] for why
    /// neighbour selection during construction must not use RaBitQ.
    fn search_layer_query_for_build(
        &self,
        query: &[f32],
        ep: usize,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let ctx = QueryCtx::new_for_build(query, self.config.metric);
        self.search_layer_ctx(&ctx, ep, ef, level)
    }

    fn search_layer_ctx(
        &self,
        ctx: &QueryCtx<'_>,
        ep: usize,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        // Dispatch on the configured rerank policy. Inline (the legacy
        // default) keeps the two-heap two-distance design that preserves
        // glove-class recall. EndOfSearch / None drop the per-visit f32
        // call and reach much higher QPS at the cost of using a noisy
        // cheap threshold during traversal; see [`RerankMode`].
        match self.config.rerank_mode {
            RerankMode::Inline => self.search_layer_ctx_inline_rerank(ctx, ep, ef, level),
            RerankMode::EndOfSearch => {
                self.search_layer_ctx_end_of_search_rerank(ctx, ep, ef, level)
            }
            RerankMode::None => self.search_layer_ctx_no_rerank(ctx, ep, ef, level),
        }
    }

    /// Cheap-distance HNSW traversal with NO rerank. Returns whatever the
    /// configured `compute_distance` (RaBitQ popcount when active) ranks
    /// as top-ef. Lowest cost per visit (one distance call, one Vec
    /// lookup); recall is bounded by the cheap estimator's accuracy.
    fn search_layer_ctx_no_rerank(
        &self,
        ctx: &QueryCtx<'_>,
        ep: usize,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let ep_dist = self.compute_distance(ctx, ep);

        let heap_cap = ef + 16;
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(heap_cap);
        let mut results: BinaryHeap<FarCandidate> = BinaryHeap::with_capacity(heap_cap);
        // Lifted out of the `while let Some(...)` loop body — `nodes.len()`
        // is invariant for the duration of a search call (insert is
        // exclusive on `&mut self`, search holds `&self`). Saves one
        // field load per inner iteration.
        let n_nodes = self.nodes.len();
        let mut visited = self.visited_pool.get(n_nodes);

        let mut connections: Vec<u64> = Vec::with_capacity(M_MAX0);
        let mut unvisited_neighbors: Vec<usize> = Vec::with_capacity(M_MAX0);

        candidates.push(Candidate {
            distance: ep_dist,
            idx: ep as u32,
        });
        results.push(FarCandidate {
            distance: ep_dist,
            idx: ep as u32,
        });
        visited.check_and_mark(ep);

        let mut farthest_dist = ep_dist;

        while let Some(closest) = candidates.pop() {
            // Standard HNSW termination (Malkov 2018, hnswlib `hnswalg.h`):
            // only stop when the cheap-frontier minimum is worse than the
            // top-ef worst AND we already have `ef` results. Without the
            // size gate, a tiny index (results.len() < ef forever) or an
            // entry-point whose cheap distance overestimates exact would
            // break out on iteration 1 and return just the seeded EP. That
            // was the SQ8 manual-calibration test regression.
            if closest.distance > farthest_dist && results.len() >= ef {
                break;
            }
            if level >= self.node_levels(closest.idx as usize) {
                continue;
            }
            if level == 0 {
                self.read_layer0_neighbours_into(closest.idx as usize, &mut connections);
            } else {
                connections.clear();
                self.neighbours_at(closest.idx as usize, level)
                    .snapshot_into(&mut connections);
            }

            unvisited_neighbors.clear();
            for (i, &neighbor_idx_u64) in connections.iter().enumerate() {
                if i + 1 < connections.len() {
                    let next_idx = connections[i + 1] as usize;
                    if let Some(p) = visited.counter_ptr(next_idx) {
                        prefetch_read_data(p);
                    }
                }
                let neighbor_idx = neighbor_idx_u64 as usize;
                if neighbor_idx < n_nodes {
                    // SAFETY: visited_pool.get(self.nodes.len()) has
                    // already resized counters >= n_nodes; the neighbor
                    // gate above just bounded neighbor_idx < n_nodes.
                    if !unsafe { visited.check_and_mark_unchecked(neighbor_idx) } {
                        unvisited_neighbors.push(neighbor_idx);
                    }
                }
            }

            if let Some(&first) = unvisited_neighbors.first() {
                self.prefetch_node_vector(first);
            }

            for (i, &neighbor_idx) in unvisited_neighbors.iter().enumerate() {
                if i + 1 < unvisited_neighbors.len() {
                    self.prefetch_node_vector(unvisited_neighbors[i + 1]);
                }

                let dist = self.compute_distance(ctx, neighbor_idx);

                if dist < farthest_dist || results.len() < ef {
                    candidates.push(Candidate {
                        distance: dist,
                        idx: neighbor_idx as u32,
                    });
                    if results.len() < ef {
                        results.push(FarCandidate {
                            distance: dist,
                            idx: neighbor_idx as u32,
                        });
                    } else if let Some(mut top) = results.peek_mut() {
                        if dist < top.distance {
                            *top = FarCandidate {
                                distance: dist,
                                idx: neighbor_idx as u32,
                            };
                        }
                    }
                    if let Some(top) = results.peek() {
                        farthest_dist = top.distance;
                    }
                }
            }
        }

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

    /// Cheap-distance HNSW traversal followed by ONE exact f32 rerank
    /// pass on the final ef-sized result heap. Industry-standard pattern
    /// (qdrant `rescore: true`, chroma single_bit, DiskANN, RaBitQ paper
    /// §4). Per-visit cost during traversal drops to popcount + one Vec
    /// lookup; the exact-distance work is amortised to ef calls at the
    /// very end, instead of being paid per visit.
    fn search_layer_ctx_end_of_search_rerank(
        &self,
        ctx: &QueryCtx<'_>,
        ep: usize,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        // Oversample: traverse the graph with a larger cheap-distance
        // frontier so the rerank pool sees more candidates. qdrant's
        // `oversampling` parameter equivalent — `factor = 1.0` (default)
        // collapses to the original "ef in, ef out" behaviour.
        let factor = self.config.rerank_oversample_factor.max(1.0);
        let frontier_ef = ((ef as f32) * factor).ceil() as usize;

        let mut result_vec = self.search_layer_ctx_no_rerank(ctx, ep, frontier_ef, level);

        // End-of-search rerank: replace every candidate's distance with
        // the exact f32 value, then sort by it. This is the only place
        // the f32 vector is read on the RaBitQ + EndOfSearch path, so
        // its memory-bandwidth bill is paid once per query per ef slot,
        // not once per neighbour visit.
        for c in result_vec.iter_mut() {
            c.distance = self.compute_exact_distance(ctx, c.idx as usize);
        }
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Truncate to user-requested ef — the oversample factor inflated
        // the frontier only, the caller still wants `ef` results.
        result_vec.truncate(ef);
        result_vec
    }

    fn search_layer_ctx_inline_rerank(
        &self,
        ctx: &QueryCtx<'_>,
        ep: usize,
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        // Two-heap pattern: cheap frontier vs accurate threshold.
        //
        // The `candidates` (frontier) heap is keyed on the cheap distance
        // — RaBitQ popcount when active, plain f32 otherwise. It drives
        // which neighbour to expand next. Noise in this score only delays
        // exploration; it cannot drop a true neighbour permanently from
        // the result set.
        //
        // The `results` heap is keyed on the EXACT distance metric.
        // `farthest_dist` (the worst score in the kept top-ef) gates the
        // prune condition `dist < farthest_dist`, so it MUST be accurate
        // — otherwise a noisy RaBitQ "worst" admits trash candidates and
        // evicts true top-K, which is exactly the recall=0.30 plateau the
        // glove bench produced on bd207af / 7b6fbe3 (single-precision
        // navigation everywhere).
        //
        // Cost: one extra f32 compute per visited neighbour (~2x distance
        // work). On glove M=16 ef=200 this drops QPS measurably but is
        // the only way to recover the recall the SIGMOD 2024 reference
        // implementation gets (0.85+) on the same data — its
        // `searchBaseLayerST_AdaptiveRerankOpt` does the same f32-on-
        // results trick under another name.
        let cheap_ep = self.compute_distance(ctx, ep);
        let exact_ep = self.compute_exact_distance(ctx, ep);

        let heap_cap = ef + 16;
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(heap_cap);
        let mut results: BinaryHeap<FarCandidate> = BinaryHeap::with_capacity(heap_cap);
        // Lifted out of the `while let Some(...)` loop body — `nodes.len()`
        // is invariant for the duration of a search call (insert is
        // exclusive on `&mut self`, search holds `&self`). Saves one
        // field load per inner iteration.
        let n_nodes = self.nodes.len();
        let mut visited = self.visited_pool.get(n_nodes);

        let mut connections: Vec<u64> = Vec::with_capacity(M_MAX0);
        let mut unvisited_neighbors: Vec<usize> = Vec::with_capacity(M_MAX0);

        candidates.push(Candidate {
            distance: cheap_ep,
            idx: ep as u32,
        });
        results.push(FarCandidate {
            distance: exact_ep,
            idx: ep as u32,
        });
        visited.check_and_mark(ep);

        let mut farthest_dist = exact_ep;

        while let Some(closest) = candidates.pop() {
            // Standard HNSW termination (Malkov 2018, hnswlib `hnswalg.h`):
            // only stop when the cheap-frontier minimum is worse than the
            // top-ef worst AND we already have `ef` results. Without the
            // size gate, a tiny index or an entry-point whose cheap
            // distance overestimates exact would break out on iteration 1
            // and return just the seeded EP.
            if closest.distance > farthest_dist && results.len() >= ef {
                break;
            }

            if level < self.node_levels(closest.idx as usize) {
                if level == 0 {
                    self.read_layer0_neighbours_into(closest.idx as usize, &mut connections);
                } else {
                    connections.clear();
                    self.neighbours_at(closest.idx as usize, level)
                        .snapshot_into(&mut connections);
                }

                unvisited_neighbors.clear();
                // hnswlib trick: prefetch the NEXT neighbour's visited
                // counter byte one iteration ahead of the read. The
                // counters array is `Vec<u8>` sized to N (1.18M bytes
                // on glove) — random-access reads on a 1MB array spill
                // out of L1 every iteration. Prefetching one ahead lets
                // the cache line arrive in time for the
                // `check_and_mark` random read on the next iteration.
                //
                // Donor: hnswlib `hnswalg.h:371-374`
                //   _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                //   _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                for (i, &neighbor_idx_u64) in connections.iter().enumerate() {
                    if i + 1 < connections.len() {
                        let next_idx = connections[i + 1] as usize;
                        if let Some(p) = visited.counter_ptr(next_idx) {
                            prefetch_read_data(p);
                        }
                    }
                    let neighbor_idx = neighbor_idx_u64 as usize;
                    if neighbor_idx < n_nodes {
                        // SAFETY: visited_pool.get(self.nodes.len())
                        // resized counters >= n_nodes; neighbor_idx <
                        // n_nodes per the gate above.
                        if !unsafe { visited.check_and_mark_unchecked(neighbor_idx) } {
                            unvisited_neighbors.push(neighbor_idx);
                        }
                    }
                }

                if let Some(&first) = unvisited_neighbors.first() {
                    self.prefetch_node_vector(first);
                }

                for (i, &neighbor_idx) in unvisited_neighbors.iter().enumerate() {
                    if i + 1 < unvisited_neighbors.len() {
                        let next_idx = unvisited_neighbors[i + 1];
                        self.prefetch_node_vector(next_idx);
                    }

                    let cheap_dist = self.compute_distance(ctx, neighbor_idx);

                    if cheap_dist < farthest_dist || results.len() < ef {
                        // Promising under the cheap metric — verify with
                        // exact f32 before letting it into the kept top-ef.
                        let exact_dist = self.compute_exact_distance(ctx, neighbor_idx);

                        candidates.push(Candidate {
                            distance: cheap_dist,
                            idx: neighbor_idx as u32,
                        });

                        // Fixed-length results: peek_mut + in-place swap when
                        // full, single push during fill. The push+pop pattern
                        // we used before paid two `O(log ef)` percolations per
                        // overflow update (one to insert, one to evict); the
                        // peek_mut+swap pattern (Qdrant's
                        // `FixedLengthPriorityQueue::push`) does one
                        // re-sift-on-drop and the work is amortised to a single
                        // percolation. At ef=800 each overflow now costs ~10
                        // swaps instead of ~20.
                        if results.len() < ef {
                            results.push(FarCandidate {
                                distance: exact_dist,
                                idx: neighbor_idx as u32,
                            });
                            if let Some(top) = results.peek() {
                                farthest_dist = top.distance;
                            }
                        } else {
                            if let Some(mut top) = results.peek_mut() {
                                if exact_dist < top.distance {
                                    *top = FarCandidate {
                                        distance: exact_dist,
                                        idx: neighbor_idx as u32,
                                    };
                                }
                                // peek_mut sifts on drop here.
                            }
                            if let Some(top) = results.peek() {
                                farthest_dist = top.distance;
                            }
                        }
                    }
                }
            }
        }

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

    /// Compute distance between the search query and a node.
    ///
    /// Codec priority on the hot path:
    /// 1. **RaBitQ** (cosine only): XOR + popcount via the shared kernel,
    ///    ~10× faster than SQ8 dequant+dot at d=1024. Skipped for non-cosine
    ///    metrics — the polarisation identity for L2 needs more than a
    ///    popcount and is deferred to a follow-up.
    /// 2. **SQ8**: dequantize the stored u8 vector then run the metric on
    ///    f32. Faster than full f32 due to 4× smaller working set.
    /// 3. **Exact f32** fallback when neither codec has an encoded
    ///    representation of this node (pre-calibration).
    #[inline(always)]
    fn compute_distance(&self, ctx: &QueryCtx<'_>, node_idx: usize) -> f32 {
        // RaBitQ fast path — cosine metric only in this slice.
        // Variants must match: a query encoded as Multi must hit a node
        // also encoded as Multi (calibration sets all nodes uniformly).
        // Mismatched variants fall through to f32 — keeps a partially
        // recalibrated index queryable rather than panicking.
        if matches!(self.config.metric, VectorMetric::Cosine) {
            // Contiguous-store fast path: read packed code + scalars from
            // the per-node block instead of dereferencing the SoA
            // `node_rabitq_codes[idx]`. Gated on byte-length match between
            // the inline slot and the query bit-planes so dims whose
            // effective code width differs from `dim/8` (e.g. dim=100
            // padded to 128) cleanly fall through to the SoA path.
            if let (Some(params), Some(RabitqQuery::OneBit(q)), Some(inline)) = (
                self.rabitq_params.as_ref(),
                ctx.rabitq_query.as_ref(),
                self.inline_layer0.as_ref(),
            ) {
                let slot_words = inline.rabitq_byte_len() / 8;
                if inline.rabitq_bits() == 1
                    && inline.rabitq_byte_len().is_multiple_of(8)
                    && slot_words == q.planes[0].len()
                    && node_idx < inline.capacity()
                {
                    // SAFETY: node_idx < capacity; the rabitq slot is
                    // 8-byte aligned by construction (rabitq_offset
                    // aligned in `new_with_rabitq_bits`) and we verified
                    // the byte length is a multiple of 8 above. The
                    // slice borrow lives for the duration of the call.
                    let scalars = unsafe { inline.rabitq_scalars(node_idx) };
                    if scalars.norm > 0.0 {
                        let bytes = unsafe { inline.rabitq(node_idx) };
                        let code_words: &[u64] = unsafe {
                            core::slice::from_raw_parts(bytes.as_ptr() as *const u64, slot_words)
                        };
                        return params.estimate_cosine_distance_q_from_slice(
                            code_words,
                            scalars.norm,
                            scalars.signed_sum,
                            scalars.correction,
                            scalars.radial,
                            scalars.cluster_id,
                            q,
                        );
                    }
                }
            }
            if let (Some(params), Some(qenc), Some(xcode)) = (
                self.rabitq_params.as_ref(),
                ctx.rabitq_query.as_ref(),
                self.node_rabitq_codes[node_idx].as_ref(),
            ) {
                match (qenc, xcode) {
                    // 1-bit data × 4-bit-plane query — paper §3.3.2 kernel.
                    // Lifts the cosine estimator from `O(1/√(D/4))` (legacy
                    // XOR popcount) to `O(1/√D)`, which is what closes the
                    // glove-100 recall=0.17 plateau (chroma single_bit.rs
                    // does the same).
                    (RabitqQuery::OneBit(q), RabitqEncoded::OneBit(x)) => {
                        return params.estimate_cosine_distance_q(x, q);
                    }
                    (RabitqQuery::Multi(q), RabitqEncoded::Multi(x)) if q.bits == x.bits => {
                        return params.estimate_cosine_distance_ext(q, x);
                    }
                    _ => {}
                }
            }
        }
        if let Some(params) = &self.sq8_params {
            if let Some(quantized) = &self.node_quantized[node_idx] {
                let dequantized = params.dequantize(quantized);
                return self.distance_for_metric(ctx, &dequantized);
            }
        }
        self.compute_exact_distance(ctx, node_idx)
    }

    /// Compute exact f32 distance between the search query and a node.
    /// Used for final reranking after SQ8 candidate generation.
    /// Falls back to dequantized SQ8 if f32 is offloaded.
    #[inline(always)]
    fn compute_exact_distance(&self, ctx: &QueryCtx<'_>, node_idx: usize) -> f32 {
        if let Some(node_vec) = self.read_node_f32(node_idx) {
            // Cosine fast path: rebuild `‖x‖` from per-code scalars when a
            // RaBitQ code is available, then feed the both-norms helper to
            // skip the `norm_l2(b)` pass per neighbour visit.
            if matches!(self.config.metric, VectorMetric::Cosine) {
                if let (Some(enc), Some(params)) = (
                    self.node_rabitq_codes[node_idx].as_ref(),
                    self.rabitq_params.as_ref(),
                ) {
                    let b_norm = rabitq_code_norm(enc, params);
                    return 1.0
                        - metrics::cosine_similarity_with_both_norms(
                            ctx.vec,
                            node_vec,
                            ctx.norm_l2,
                            b_norm,
                        );
                }
            }
            return self.distance_for_metric(ctx, node_vec);
        }
        // f32 offloaded — fall back to dequantized SQ8 (slightly less accurate)
        if let Some(ref params) = self.sq8_params {
            if let Some(ref quantized) = self.node_quantized[node_idx] {
                let dequantized = params.dequantize(quantized);
                return self.distance_for_metric(ctx, &dequantized);
            }
        }
        f32::INFINITY
    }

    /// Compute distance using the configured metric, with query-side state
    /// pre-computed in `ctx`. Cosine uses the cached ‖query‖₂; other metrics
    /// ignore it and the field is set to a sentinel by `QueryCtx::new`.
    #[inline(always)]
    fn distance_for_metric(&self, ctx: &QueryCtx<'_>, b: &[f32]) -> f32 {
        match self.config.metric {
            VectorMetric::Cosine => {
                1.0 - metrics::cosine_similarity_with_query_norm(ctx.vec, b, ctx.norm_l2)
            }
            VectorMetric::L2 => metrics::euclidean_distance_squared(ctx.vec, b),
            VectorMetric::DotProduct => -metrics::dot_product(ctx.vec, b),
            VectorMetric::L1 => metrics::manhattan_distance(ctx.vec, b),
        }
    }

    /// Prune connections to keep at most max_conn (keep nearest).
    /// Uses exact f32 distance for pruning when available, falls back to
    /// dequantized SQ8 distances when f32 vectors are offloaded.
    fn prune_connections(&self, node_idx: usize, level: usize, max_conn: usize) {
        self.prune_connections_with_extras(node_idx, level, max_conn, &[]);
    }

    /// Prune `node_idx`'s neighbour list at `level` down to `max_conn`,
    /// considering both the currently-attached neighbours AND a set of
    /// `extras` queued candidates. Each entry is scored by distance to
    /// `node_idx`; the nearest `max_conn` are kept.
    ///
    /// Folding the extras into the same selection pass is what fixes the
    /// batched-apply recall collapse: the previous code pruned-then-
    /// `cas_append`ed, but `prune` truncates *at* `max_conn` (not below),
    /// so the subsequent append loop always saw a full list and dropped
    /// every backfilled candidate. New nodes ended up with valid outgoing
    /// edges but zero incoming back-edges from existing hubs, making the
    /// graph unreachable from the entry point during search.
    fn prune_connections_with_extras(
        &self,
        node_idx: usize,
        level: usize,
        max_conn: usize,
        extras: &[u64],
    ) {
        // Stored u64s ARE neighbour indices now (not NodeIds) — no
        // id_to_idx hop. Dedupe via HashSet on those indices.
        let neighbours = self.neighbours_at(node_idx, level).snapshot();
        let mut seen: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(neighbours.len() + extras.len());
        let mut scored: Vec<(f32, u64)> = Vec::with_capacity(neighbours.len() + extras.len());
        for &neighbor_idx_u64 in neighbours.iter().chain(extras.iter()) {
            if !seen.insert(neighbor_idx_u64) {
                continue;
            }
            let neighbor_idx = neighbor_idx_u64 as usize;
            if neighbor_idx < self.nodes.len() {
                let dist = self.distance_between_nodes(node_idx, neighbor_idx);
                scored.push((dist, neighbor_idx_u64));
            }
        }

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_conn);

        let kept: Vec<u64> = scored.into_iter().map(|(_, idx)| idx).collect();
        self.set_outgoing(node_idx, level, &kept);
    }

    // ── Atomic neighbour write helpers (C1 day 5, refined C3 day 2) ────────
    //
    // Single source of truth: `neighbours_l0` + `neighbours_upper`. The
    // legacy `node.connections` is gone.
    //
    // **C3 day 2:** helpers that touch only the atomic neighbour storage now take
    // `&self` instead of `&mut self`. The new node's atomic layer Vec is
    // append-only at slot-creation time (`apply_insert_plan` pushes once),
    // and once the layers exist their internal state mutates through atomic
    // APIs that need only `&self`. This unlocks parallel apply for distinct
    // node indices in the C3 concurrent insert path:
    //
    // * `set_outgoing(&self, idx, …)` is conflict-free across distinct
    //   `idx` because the new-node's atomic list is freshly created by
    //   the (single-writer) node-allocation phase.
    // * `cas_add_neighbour_to(&self, neighbour_idx, …)` uses
    //   `AtomicNeighbourList::cas_append` so multiple threads inserting
    //   incoming edges into the same existing neighbour list never lose
    //   each other's updates.
    //
    // The legacy `&mut self` `add_neighbour_to` / `remove_neighbour_from`
    // / `clear_outgoing` paths are kept for the sequential C1/C2 callers
    // (update_existing_node's rebuild, prune fallback).

    /// Resolve `(node, layer)` to the underlying
    /// [`AtomicNeighbourList`]. Layer 0 lives in the flat hot-path
    /// `neighbours_l0` vec; layers ≥1 live in `neighbours_upper[idx]`
    /// (cold path). One source of truth so call-sites read the same
    /// addressing rule whether they're hot (per-visit) or cold
    /// (top-down descent / rebuild).
    #[inline]
    fn neighbours_at(&self, idx: usize, level: usize) -> &AtomicNeighbourList<M_MAX0> {
        if level == 0 {
            &self.neighbours_l0[idx]
        } else {
            &self.neighbours_upper[idx][level - 1]
        }
    }

    /// Number of layers the node participates in (`top_level + 1`).
    /// Replaces the legacy `neighbours_atomic[idx].len()` idiom from
    /// before the layer-0 / upper-layer split.
    #[inline]
    fn node_levels(&self, idx: usize) -> usize {
        1 + self.neighbours_upper[idx].len()
    }

    /// Replace the entire neighbour set at `(idx, level)` with `ids`.
    /// Truncates to `M_MAX0` on overflow (logged at construction time).
    fn set_outgoing(&self, idx: usize, level: usize, ids: &[u64]) {
        let n = ids.len().min(M_MAX0);
        self.neighbours_at(idx, level).set(&ids[..n]);
        if level == 0 {
            self.mirror_layer0_neighbours_to_inline(idx);
        }
    }

    /// Multi-writer append-edge primitive (C3). Tries to append `id` to
    /// `(neighbour_idx, level)` via [`AtomicNeighbourList::cas_append`].
    /// Returns `true` on success, `false` if the neighbour list is at
    /// capacity (`m_max0`/`m`) and the caller must fall back to a single-
    /// writer prune protocol.
    ///
    /// Today the only caller is C3's parallel apply path (next day's
    /// commit). C2's serial apply continues to use [`add_neighbour_to`].
    #[allow(dead_code)] // Wired into parallel apply in C3 day 3.
    fn cas_add_neighbour_to(&self, neighbour_idx: usize, level: usize, id: u64) -> bool {
        let ok = self.neighbours_at(neighbour_idx, level).cas_append(id);
        if ok && level == 0 {
            self.mirror_layer0_neighbours_to_inline(neighbour_idx);
        }
        ok
    }

    /// Append `id` to `(neighbour_idx, level)`. If the resulting list
    /// exceeds `max_conn`, run `prune_connections` to shrink back to the
    /// nearest `max_conn` neighbours.
    ///
    /// Single-writer (C1/C2 path). The new node case uses `cas_append`
    /// internally so cas-based callers can race with this safely on the
    /// `len` counter, but the prune branch needs `&mut self` because
    /// `prune_connections` reads vectors + reorders the list.
    fn add_neighbour_to(&mut self, neighbour_idx: usize, level: usize, id: u64, max_conn: usize) {
        if self.neighbours_at(neighbour_idx, level).cas_append(id) {
            let len_now = self.neighbours_at(neighbour_idx, level).len();
            if len_now > max_conn {
                self.prune_connections(neighbour_idx, level, max_conn);
                // prune_connections funnels through set_outgoing, which
                // mirrors. Done.
            } else if level == 0 {
                self.mirror_layer0_neighbours_to_inline(neighbour_idx);
            }
        } else {
            // List at the inline cap (`M_MAX0`). When `max_conn == M_MAX0`
            // (default config: `m_max0 = 2 * m = 64` matches the compile-
            // time cap on layer 0) the previous "force-prune + retry" path
            // was a no-op — prune of N elements down to `max_conn == N`
            // keeps everything, then `cas_append` still finds no slot and
            // the new edge is silently dropped. That's the connectivity-
            // collapse bug that drove SIFT1M recall to ~1.5%.
            //
            // Fix: do the prune in-memory with the new id included, so it
            // competes fairly against existing neighbours. The kept set
            // has ≤ `max_conn` elements (`≤ M_MAX0`) so `set_outgoing` is
            // guaranteed to fit without truncation.
            let mut snap = self.neighbours_at(neighbour_idx, level).snapshot();
            snap.push(id);
            // Stored u64s ARE neighbour indices — direct cast, no map hop.
            let mut scored: Vec<(f32, u64)> = Vec::with_capacity(snap.len());
            for &nidx_u64 in &snap {
                let nidx = nidx_u64 as usize;
                if nidx < self.nodes.len() {
                    let dist = self.distance_between_nodes(neighbour_idx, nidx);
                    scored.push((dist, nidx_u64));
                }
            }
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(max_conn);
            let kept: Vec<u64> = scored.into_iter().map(|(_, nid)| nid).collect();
            self.set_outgoing(neighbour_idx, level, &kept);
        }
    }

    /// Remove every occurrence of `id` from `(idx, level)`. No-op if `id`
    /// is absent.
    fn remove_neighbour_from(&mut self, idx: usize, level: usize, id: u64) {
        let mut snap = self.neighbours_at(idx, level).snapshot();
        snap.retain(|&nid| nid != id);
        self.set_outgoing(idx, level, &snap);
    }

    /// Clear the entire neighbour set at `(idx, level)`.
    fn clear_outgoing(&mut self, idx: usize, level: usize) {
        self.neighbours_at(idx, level).set(&[]);
    }

    /// Compute distance between two nodes in the graph.
    /// Prefers exact f32 when available, falls back to dequantized SQ8.
    /// RobustPrune neighbour selection (Vamana paper §3, Algorithm 3).
    ///
    /// `candidates` come from `search_layer_query_for_build` ordered by
    /// distance to the inserted node `p` (lower = closer). Iterate in
    /// closest-first order: each pick `p*` joins the kept set, then we
    /// drop every later candidate `p'` for which `α · d(p*, p') ≤ d(p, p')`
    /// — these are the redundant edges (p' is closer to p* than 1/α the
    /// way back to p, so the edge `p → p'` would just retrace
    /// `p → p* → p'`). Stops when `kept.len() == max_conn`.
    ///
    /// Cost is `O(max_conn · candidates.len())` `distance_between_nodes`
    /// calls — for default ef_construction=200, max_conn=32 that's ~6.4k
    /// extra distance evaluations per insert, paid once at build time
    /// for a graph-quality win every search reads.
    fn select_neighbours_robust_prune(
        &self,
        candidates: &[Candidate],
        max_conn: usize,
    ) -> Vec<usize> {
        if max_conn == 0 || candidates.is_empty() {
            return Vec::new();
        }
        let alpha = self.config.alpha_pruning;

        // Sort by ascending distance to inserted node (lower = closer).
        // `Candidate.distance` is min-heap-ordered in the source but we
        // copy out so we can mark pruned entries and pop in closest-first
        // order without owning a heap.
        let mut pool: Vec<Candidate> = candidates.to_vec();
        pool.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept: Vec<usize> = Vec::with_capacity(max_conn);
        let mut pruned = vec![false; pool.len()];

        for i in 0..pool.len() {
            if pruned[i] {
                continue;
            }
            let p_star = pool[i].idx as usize;
            kept.push(p_star);
            if kept.len() == max_conn {
                break;
            }
            // Prune later candidates that fail the α-test against p_star.
            for j in (i + 1)..pool.len() {
                if pruned[j] {
                    continue;
                }
                let p_prime = pool[j].idx as usize;
                // d(p, p') — distance from inserted node to p'.
                let d_p_pprime = pool[j].distance;
                // d(p*, p') — pairwise distance between two candidates.
                let d_pstar_pprime = self.distance_between_nodes(p_star, p_prime);
                if alpha * d_pstar_pprime <= d_p_pprime {
                    pruned[j] = true;
                }
            }
        }

        kept
    }

    fn distance_between_nodes(&self, a_idx: usize, b_idx: usize) -> f32 {
        // Try exact f32 for both nodes
        if let (Some(ref va), Some(ref vb)) = (&self.node_vectors[a_idx], &self.node_vectors[b_idx])
        {
            let ctx = QueryCtx::new(
                va,
                self.config.metric,
                self.rabitq_params.as_ref(),
                &self.config.quantization,
            );
            return self.distance_for_metric(&ctx, vb);
        }
        // Fall back to SQ8 approximate distance
        if let Some(ref params) = self.sq8_params {
            let va = self.get_node_vector_or_dequantized(a_idx, params);
            let vb = self.get_node_vector_or_dequantized(b_idx, params);
            let ctx = QueryCtx::new(
                &va,
                self.config.metric,
                self.rabitq_params.as_ref(),
                &self.config.quantization,
            );
            return self.distance_for_metric(&ctx, &vb);
        }
        f32::INFINITY
    }

    /// Get a node's vector for query purposes: f32 if available, else dequantized SQ8.
    /// Used by search_layer_greedy/search_layer during insert when the node
    /// should have f32 but may not if auto_calibrate offloaded early.
    fn get_node_f32_or_dequantized(&self, idx: usize) -> Vec<f32> {
        if let Some(ref v) = self.node_vectors[idx] {
            return v.clone();
        }
        if let Some(ref params) = self.sq8_params {
            if let Some(ref q) = self.node_quantized[idx] {
                return params.dequantize(q);
            }
        }
        Vec::new()
    }

    /// Get a node's vector: f32 if available, otherwise dequantize from SQ8.
    fn get_node_vector_or_dequantized(&self, idx: usize, params: &Sq8Params) -> Vec<f32> {
        if let Some(ref v) = self.node_vectors[idx] {
            return v.clone();
        }
        if let Some(ref q) = self.node_quantized[idx] {
            return params.dequantize(q);
        }
        Vec::new()
    }

    /// Prefetch a node's vector data into L1 cache. Targets f32 when available,
    /// falls back to quantized vector when f32 is offloaded.
    #[inline(always)]
    fn prefetch_node_vector(&self, idx: usize) {
        // Tier the prefetch to whichever representation the hot loop is
        // going to read first. When RaBitQ is active the distance kernel
        // touches `code.code` (the u64 sign-bit array) and `code.norm /
        // correction / radial` scalars first — long before the f32
        // rerank's `node.vector` load. Prefetching the wrong allocation
        // costs one cache miss per neighbour visit, which on glove M=16
        // ef=200 the e554e72 profile measured as ~7% of search cycles
        // hidden inside `search_layer_ctx`'s self-time.
        //
        // Cosine + RaBitQ is the only path where this distinction
        // matters in practice; other codecs fall through to the legacy
        // vector / quantized prefetch.
        // Profile of d365611 (perf record glove M=16 cosine RaBitQ
        // ef={200,800}) showed this helper at 33.7% of search_layer_ctx
        // cycles. The prefetch INSTRUCTIONS are near-free; the cost is
        // the SoA cache misses the helper performs to DECIDE what to
        // prefetch — `node_rabitq_codes[idx]` (Option<RabitqEncoded>),
        // `node_vectors[idx]` (Option<Vec<f32>>), `node_quantized[idx]`
        // (Option<Vec<u8>>) — three independent allocations sized at
        // ~1.18M slots each on glove. Reading three of them per
        // neighbour visit pulls three cache lines just to issue one
        // prefetch hint, which is the opposite of the intended win.
        //
        // Fix: on the cosine + RaBitQ hot path the distance kernel
        // reads ONLY `code` (the popcount frontier). The original f32
        // vector is consumed by the results-heap exact rerank — which
        // happens after `search_layer_ctx` returns, only on the
        // top-ef candidates, not on every visit. So skip the f32 /
        // quantized lookup entirely when we know the kernel is going
        // to do RaBitQ. Cuts the helper's SoA-read cost in half
        // (one cache line + one Option discriminant + one enum match
        // instead of two cache lines + two Option discriminants).
        if matches!(self.config.metric, VectorMetric::Cosine) {
            if let Some(ref enc) = self.node_rabitq_codes[idx] {
                let code_words = match enc {
                    RabitqEncoded::OneBit(c) => c.code.as_ptr() as *const u8,
                    RabitqEncoded::Multi(c) => c.packed.as_ptr(),
                };
                prefetch_read_data(code_words);
                return;
            }
        }
        if let Some(ref v) = self.node_vectors[idx] {
            prefetch_read_data(v.as_ptr() as *const u8);
        } else if let Some(ref q) = self.node_quantized[idx] {
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
    fn inline_layer0_is_lazy_until_first_insert() {
        let index = HnswIndex::new(make_config(VectorMetric::L2));
        assert!(
            index.inline_layer0().is_none(),
            "inline_layer0 must not allocate until the first insert observes a vector dim"
        );
    }

    #[test]
    fn inline_layer0_mirrors_soa_on_per_item_insert() {
        let mut cfg = make_config(VectorMetric::L2);
        // Cap small so the test does not allocate 100s of MB for the
        // contiguous store; idx values stay below this.
        cfg.max_elements = 32;
        let mut index = HnswIndex::new(cfg);
        let payload = |i: u64| -> Vec<f32> {
            (0..8)
                .map(|d| (i as f32) * 0.5 + (d as f32) * 0.01)
                .collect()
        };
        for i in 0..8u64 {
            index.insert(i, payload(i));
        }
        let inline = index
            .inline_layer0()
            .expect("inline store must be populated after the first insert");
        for idx in 0..8 {
            // SAFETY: idx < 8 < inline.capacity() (32).
            unsafe {
                let label = inline.label(idx).load(std::sync::atomic::Ordering::Relaxed);
                let expected_id = index.nodes[idx].id;
                assert_eq!(label, expected_id, "label mismatch at idx={idx}");
                let inline_vec: Vec<f32> = inline.vector_f32(idx).to_vec();
                let soa_vec = index.node_vectors[idx]
                    .as_ref()
                    .expect("SoA vector present");
                assert_eq!(inline_vec, *soa_vec, "vector mismatch at idx={idx}");
            }
        }
    }

    #[test]
    fn inline_layer0_mirrors_soa_layer0_neighbours() {
        // Insert enough nodes for the build to populate non-trivial layer-0
        // neighbour lists. After the build, every node's layer-0 neighbour
        // ids and length must match between SoA and the contiguous store.
        let mut cfg = make_config(VectorMetric::L2);
        cfg.max_elements = 64;
        let mut index = HnswIndex::new(cfg);
        for i in 0..32u64 {
            let v: Vec<f32> = (0..8)
                .map(|d| ((i as f32) * 0.7 + (d as f32) * 0.13).sin())
                .collect();
            index.insert(i, v);
        }
        let inline = index
            .inline_layer0()
            .expect("inline store must be populated");
        for idx in 0..index.nodes.len() {
            let soa = &index.neighbours_l0[idx];
            let mut soa_snap: Vec<u64> = Vec::with_capacity(M_MAX0);
            soa.snapshot_into(&mut soa_snap);
            let soa_len = soa_snap.len();
            // SAFETY: idx < inline.capacity() and slots < m_max0.
            unsafe {
                let inline_len = inline
                    .neighbour_len(idx)
                    .load(std::sync::atomic::Ordering::Relaxed)
                    as usize;
                assert_eq!(inline_len, soa_len, "neighbour_len mismatch at idx={idx}");
                for (slot, &expected) in soa_snap.iter().enumerate().take(soa_len) {
                    let inline_id = inline
                        .neighbour(idx, slot)
                        .load(std::sync::atomic::Ordering::Relaxed);
                    assert_eq!(
                        inline_id, expected,
                        "neighbour id mismatch at idx={idx} slot={slot}"
                    );
                }
            }
        }
    }

    #[test]
    fn inline_layer0_mirrors_soa_on_batch_insert() {
        let mut cfg = make_config(VectorMetric::L2);
        cfg.max_elements = 64;
        let mut index = HnswIndex::new(cfg);
        // Above BATCH_PARALLEL_THRESHOLD so the rayon-planned path fires.
        let items: Vec<(u64, Vec<f32>)> = (0..32u64)
            .map(|i| {
                let v: Vec<f32> = (0..8).map(|d| -(i as f32) + (d as f32) * 0.1).collect();
                (i, v)
            })
            .collect();
        index.insert_batch(items);
        let inline = index
            .inline_layer0()
            .expect("inline store must be populated after a batch insert");
        for idx in 0..index.nodes.len() {
            // SAFETY: idx < nodes.len() < inline.capacity() (64).
            unsafe {
                let label = inline.label(idx).load(std::sync::atomic::Ordering::Relaxed);
                let expected_id = index.nodes[idx].id;
                assert_eq!(label, expected_id, "label mismatch at idx={idx}");
                let inline_vec: Vec<f32> = inline.vector_f32(idx).to_vec();
                let soa_vec = index.node_vectors[idx]
                    .as_ref()
                    .expect("SoA vector present");
                assert_eq!(inline_vec, *soa_vec, "vector mismatch at idx={idx}");
            }
        }
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
                        let seed =
                            (i.wrapping_mul(2_654_435_761).wrapping_add(d * 6_700_417)) as u32;
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
        for idx in 0..index.nodes.len() {
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
            quantization: QuantizationCodec::Sq8,
            rerank_candidates: 20,
            calibration_threshold: 5, // Low threshold for testing
            offload_vectors: false,
            property_name: String::new(),
            rerank_mode: RerankMode::Inline,
            rerank_oversample_factor: 1.0,
            alpha_pruning: 1.0,
            max_elements: 1_000_000,
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
        for q in &index.node_quantized {
            assert!(q.is_some());
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
        let idx = *index.id_to_idx.get(&100).expect("inserted");
        assert!(index.node_quantized[idx].is_some());
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
        let coded = index
            .node_rabitq_codes
            .iter()
            .filter(|c| c.is_some())
            .count();
        assert_eq!(coded, n, "all {n} nodes should have RaBitQ codes");

        // Brute-force ground truth by cosine distance.
        let query = &vectors[0];
        let mut ground_truth: Vec<(f32, u64)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let cos =
                    metrics::cosine_similarity_with_query_norm(query, v, metrics::norm_l2(query));
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
        let coded = index
            .node_rabitq_codes
            .iter()
            .filter(|c| c.is_some())
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
                    let cos = metrics::cosine_similarity_with_query_norm(
                        query,
                        v,
                        metrics::norm_l2(query),
                    );
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
        // Wires-up test for 2-bit Extended-RaBitQ (R862): same shape as the
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
        for (i, code_opt) in index.node_rabitq_codes.iter().enumerate() {
            let code = code_opt.as_ref().expect("rabitq code populated");
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
                let cos =
                    metrics::cosine_similarity_with_query_norm(query, v, metrics::norm_l2(query));
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
        // SQ8 / RaBitQ codes stay in RAM per ADR-033 revised).
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
        for i in 0..index.nodes.len() {
            let code_opt = &index.node_rabitq_codes[i];
            assert!(
                code_opt.is_some(),
                "node {i} missing rabitq code after set_rabitq_params"
            );
            let code = code_opt.as_ref().expect("checked is_some above");
            let expected = RabitqEncoded::OneBit(
                persisted.encode(index.node_vectors[i].as_ref().expect("f32 retained")),
            );
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

        for i in 0..index.nodes.len() {
            let v = index.node_vectors[i]
                .as_ref()
                .expect("f32 should be retained (offload_vectors=false)");
            assert_eq!(v.len(), dims as usize);
            let q = index.node_quantized[i]
                .as_ref()
                .expect("should be quantized");
            assert_eq!(q.len(), dims as usize);
            assert_eq!(v.len() * std::mem::size_of::<f32>(), q.len() * 4);
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
        for q in &index.node_quantized {
            assert!(q.is_some());
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
            quantization: QuantizationCodec::Sq8,
            rerank_candidates: 20,
            calibration_threshold: 5, // Low threshold for testing
            offload_vectors: true,
            property_name: "embedding".to_string(),
            rerank_mode: RerankMode::Inline,
            rerank_oversample_factor: 1.0,
            alpha_pruning: 1.0,
            max_elements: 1_000,
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
        for (i, node) in index.nodes.iter().enumerate() {
            assert!(
                index.node_vectors[i].is_none(),
                "f32 should be None when offloaded (node {})",
                node.id
            );
            assert!(
                index.node_quantized[i].is_some(),
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
            .node_vectors
            .iter()
            .map(|v| v.as_ref().map_or(0, |x| x.len() * 4))
            .sum();
        let retained_f32_bytes: usize = retained
            .node_vectors
            .iter()
            .map(|v| v.as_ref().map_or(0, |x| x.len() * 4))
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

    #[test]
    fn atomic_neighbours_track_inserts_and_updates() {
        // C1 day 5: atomic neighbour storage is the sole source of truth.
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
        });

        for i in 0..30u64 {
            let v: Vec<f32> = (0..4).map(|d| ((i * 13 + d) as f32).sin()).collect();
            idx.insert(i, v);
        }
        // Re-insert one node to exercise update_existing_node.
        idx.insert(7, vec![0.5, -0.5, 0.5, -0.5]);

        assert_eq!(idx.neighbours_l0.len(), idx.nodes.len());
        assert_eq!(idx.neighbours_upper.len(), idx.nodes.len());

        let mut scratch = Vec::with_capacity(M_MAX0);
        for node_idx in 0..idx.nodes.len() {
            // Node layer count tracks the node's max_layer + 1.
            assert_eq!(
                idx.node_levels(node_idx),
                idx.nodes[node_idx].max_layer + 1,
                "node {node_idx} layer count diverged from max_layer + 1",
            );
            for level in 0..idx.node_levels(node_idx) {
                idx.neighbours_at(node_idx, level)
                    .snapshot_into(&mut scratch);
                assert!(
                    scratch.len() <= M_MAX0,
                    "node {node_idx} level {level} exceeds m_max0 cap",
                );
                for nid in &scratch {
                    assert!(
                        idx.id_to_idx.contains_key(nid),
                        "dangling neighbour {nid} at node {node_idx} level {level}",
                    );
                }
            }
        }
    }

    #[test]
    fn max_elements_preallocates_node_storage() {
        // C1 day 6: HnswConfig::max_elements drives Vec::with_capacity for
        // nodes + neighbours_l0 / neighbours_upper so steady-state inserts
        // don't pay reallocation cost on the hot path.
        let cfg = HnswConfig {
            max_elements: 50_000,
            ..HnswConfig::default()
        };
        let idx = HnswIndex::new(cfg);

        assert!(
            idx.nodes.capacity() >= 50_000,
            "nodes Vec capacity {} < max_elements 50_000",
            idx.nodes.capacity()
        );
        assert!(
            idx.neighbours_l0.capacity() >= 50_000,
            "neighbours_l0 Vec capacity {} < max_elements 50_000",
            idx.neighbours_l0.capacity()
        );
        // HashMap::with_capacity may round up; just assert it's non-zero.
        assert!(idx.id_to_idx.capacity() >= 50_000);
    }

    #[test]
    fn insert_within_max_elements_does_not_reallocate_node_vec() {
        // The whole point of pre-allocation: stable Vec capacity through
        // the full max_elements range of inserts.
        let cfg = HnswConfig {
            m: 4,
            m_max0: 8,
            max_elements: 200,
            max_dimensions: 4,
            ..HnswConfig::default()
        };
        let mut idx = HnswIndex::new(cfg);
        let cap_before = idx.nodes.capacity();
        for i in 0..200u64 {
            let v: Vec<f32> = (0..4).map(|d| ((i * 7 + d) as f32).sin()).collect();
            idx.insert(i, v);
        }
        assert_eq!(
            idx.nodes.capacity(),
            cap_before,
            "nodes Vec reallocated within max_elements window",
        );
    }

    #[test]
    fn insert_batch_matches_serial_insert_topology() {
        // C2 day 2: insert_batch must produce a graph indistinguishable
        // (modulo level-assignment RNG and within-batch plan staleness)
        // from sequential inserts for query-correctness purposes.
        //
        // The test pins both the RNG seed and uses a deterministic
        // workload, then compares search recall@10 on a held-out query
        // set between (a) per-item insert and (b) insert_batch of the
        // same items. Recall agreement ≥ 0.7 is the bar — exact topology
        // equality is NOT achievable because within-batch plans see the
        // pre-batch graph state.
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

        // Recall agreement on a held-out query set: queries close to
        // random points in the corpus. For each query, retrieve top-10
        // from both indexes; count how many of the serial top-10 are
        // also in batched top-10.
        let q_ids = [3u64, 17, 31, 64, 128, 257, 384, 499];
        let mut hits = 0;
        let mut total = 0;
        for &qid in &q_ids {
            let q = make_vec(qid);
            let serial_top: std::collections::HashSet<u64> =
                serial.search(&q, 10).into_iter().map(|r| r.id).collect();
            let batched_top: std::collections::HashSet<u64> =
                batched.search(&q, 10).into_iter().map(|r| r.id).collect();
            hits += serial_top.intersection(&batched_top).count();
            total += serial_top.len();
        }
        let agreement = hits as f64 / total as f64;
        assert!(
            agreement >= 0.7,
            "serial/batched top-10 agreement = {agreement:.2} (expected ≥ 0.7)",
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

    #[test]
    fn apply_insert_plans_parallel_ingests_every_item() {
        // C3 day 3: explicit parallel-apply variant. Until the prune-pass
        // (day 4) backfills dropped back-edges, recall agreement vs serial
        // hovers in the 0.5-0.6 range; the property we assert here is
        // weaker: every plan must result in a present, self-recoverable
        // node (search for own vector returns it as top-1).
        let cfg = HnswConfig {
            m: 8,
            m_max0: 16,
            ef_construction: 50,
            ef_search: 50,
            metric: VectorMetric::L2,
            max_dimensions: 4,
            quantization: QuantizationCodec::None,
            rerank_candidates: 50,
            calibration_threshold: 10_000,
            offload_vectors: false,
            property_name: String::new(),
            rerank_mode: RerankMode::Inline,
            rerank_oversample_factor: 1.0,
            alpha_pruning: 1.0,
            max_elements: 1_000,
        };
        let mut idx = HnswIndex::new(cfg);

        // Seed 64 nodes so the parallel apply's plans have a real graph.
        for i in 0..64u64 {
            let v: Vec<f32> = (0..4).map(|d| ((i * 31 + d) as f32 * 0.1).sin()).collect();
            idx.insert(i, v);
        }
        // Pre-compute plans against the seeded graph, then apply in
        // parallel via the C3 day 3 entry point.
        let plans: Vec<(InsertPlan, Vec<f32>)> = (64..200u64)
            .map(|i| {
                let v: Vec<f32> = (0..4).map(|d| ((i * 31 + d) as f32 * 0.1).sin()).collect();
                let plan = idx.compute_insert_plan(i, &v);
                (plan, v)
            })
            .collect();
        idx.apply_insert_plans_parallel(plans);

        assert_eq!(idx.len(), 200);
        // Every node must search-recover to itself as the closest result.
        // This is a much looser invariant than recall agreement: it says
        // "the node landed in the index and its outgoing edges are
        // sufficient to reach itself from any nearby entry".
        let mut self_recovered = 0;
        for i in 64..200u64 {
            let q: Vec<f32> = (0..4).map(|d| ((i * 31 + d) as f32 * 0.1).sin()).collect();
            if idx.search(&q, 1)[0].id == i {
                self_recovered += 1;
            }
        }
        let ratio = self_recovered as f64 / 136.0;
        assert!(
            ratio >= 0.85,
            "self-recover ratio {ratio:.2} after parallel apply (expected ≥ 0.85)",
        );
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
}
