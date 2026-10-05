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

mod bulk_build;
mod data_level0;
mod entry_point;
mod id_map;
mod neighbours;
mod rabitq_block;
mod reclaim;
mod reorder;
mod search_scratch;
mod stats;
mod visited;

pub use stats::PublicationSnapshot;

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
/// transfer mode.
pub const M_MAX0: usize = 64;

use std::collections::BinaryHeap;

use coordinode_core::graph::types::VectorMetric;
use search_scratch::SearchScratchPool;
use tracing::warn;
use visited::VisitedPool;

/// Software prefetch hint: request CPU to load `ptr` into L1 cache.
/// This is a performance hint — no effect on correctness.
/// No-op on unsupported platforms. `_MM_HINT_T0` targets every cache
/// level, the hint the next neighbour's vector needs before it is scored.
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
use crate::quantize::rabitq::{RaBitQCode, RaBitQExtCode, RaBitQParams, RaBitQQuery};

/// Per-vector RaBitQ encoding. The variant is fixed at index calibration
/// time from [`HnswConfig::quantization`] and must match across all nodes
/// in the same index — mixing 1-bit and multi-bit codes in a single index
/// is an invariant violation (different distance kernels, different code
/// shapes, no shared comparison semantics).
#[derive(Debug, Clone, PartialEq)]
pub enum RabitqEncoded {
    /// 1-bit sign-bit code, popcount distance kernel. Default codec.
    OneBit(RaBitQCode),
    /// 2/3/4-bit Extended-RaBitQ code, centroid-LUT distance kernel.
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
/// RaBitQ supersedes SQ8 as the primary in-RAM codec; SQ8 is retained for
/// the cross-shard disk rerank pool. `None` means
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
    /// Values below `1.0` (including the default `0.0`) mean AUTO:
    /// the metric-tuned default from [`HnswConfig::effective_alpha`]
    /// (Cosine = 1.15, measured +16% QPS at recall 0.95 on
    /// glove-100; other metrics = 1.0 until measured). Set `1.0`
    /// explicitly to force α-pruning off.
    pub alpha_pruning: f32,
    /// Expected upper bound on the number of vectors this index will hold.
    /// Used at construction to pre-allocate the `nodes` and
    /// `neighbours_l0` / `neighbours_upper` vectors so the insert hot path never pays
    /// `Vec` reallocation cost. Insert beyond `max_elements` is supported
    /// (the vectors grow normally) but the first overflow re-allocation
    /// pauses inserts briefly; size this generously when ingestion volume
    /// is known. Default: 1_000_000.
    pub max_elements: u32,
    /// Bytes of replaced neighbour lists the index lets wait for reclamation
    /// before an insert or removal holds off until running searches release
    /// them. A search that stalls keeps every list replaced after it began;
    /// the budget bounds that memory by slowing writers, never by dropping
    /// a write. `usize::MAX` disables the bound. Live-settable with
    /// [`HnswIndex::set_retired_bytes_budget`]. Default:
    /// [`DEFAULT_RETIRED_BYTES_BUDGET`].
    pub retired_bytes_budget: usize,
}

/// Fewest retired slots a sweep waits for (see `retire_node`).
const SWEEP_MIN_SLOTS: usize = 64;

/// Default [`HnswConfig::retired_bytes_budget`]: 256 MiB.
pub const DEFAULT_RETIRED_BYTES_BUDGET: usize = 256 << 20;

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
            alpha_pruning: 0.0,
            max_elements: 1_000_000,
            retired_bytes_budget: DEFAULT_RETIRED_BYTES_BUDGET,
        }
    }
}

impl HnswConfig {
    /// Resolve the effective RobustPrune α. Explicit values (`>= 1.0`)
    /// win; anything below 1.0 (α < 1 is meaningless for the pruning
    /// rule) selects the metric-tuned default: 1.15 for Cosine
    /// (measured on glove-100: same recall at ~2/3 the ef budget),
    /// 1.0 (off) for metrics where the gain is not yet measured.
    #[inline]
    pub fn effective_alpha(&self) -> f32 {
        if self.alpha_pruning >= 1.0 {
            return self.alpha_pruning;
        }
        match self.metric {
            VectorMetric::Cosine => 1.15,
            _ => 1.0,
        }
    }
}

/// HNSW index: in-memory approximate nearest neighbor graph.
pub struct HnswIndex {
    config: HnswConfig,
    /// Number of nodes. Every node's id, norms, f32 vector, codes and
    /// neighbour lists live in `data_level0` at stable addresses, indexed
    /// `0..len`. A node's SQ8 code and RaBitQ code (1-bit or 2/3/4-bit
    /// Extended-RaBitQ) are `None` until calibration; the RaBitQ variant is
    /// fixed at calibration time and never mixed within one index.
    node_count: core::sync::atomic::AtomicUsize,
    /// Number of ids in the index. Below `node_count` once updates retire
    /// slots: an update moves its id to a new slot.
    live_count: core::sync::atomic::AtomicUsize,
    /// Map from node ID to its index in the node store.
    id_to_idx: id_map::IdMap,
    /// Lock-free entry point: packed `(level, idx)` in a single
    /// `AtomicU64` with `u64::MAX` as the "empty index" sentinel.
    /// Multiple inserts that land on novel max-layers race through
    /// [`EntryPoint::try_promote`] (CAS-loop on a single atomic, max
    /// two iterations under realistic contention). Replaces the previous `(Option<usize>, usize)` pair
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
    /// Pool of reusable search scratch buffers (candidate / result
    /// storage + connection / unvisited indices). Avoids the per-query
    /// allocator round-trip that becomes a choke point under MT4.
    #[allow(
        dead_code,
        reason = "consumer lands in the search-side wiring chunk on the same plan"
    )]
    search_scratch_pool: SearchScratchPool,
    /// RNG state for random level selection (xorshift64).
    /// Proper RNG gives correct exponential layer distribution.
    /// AtomicU64 so concurrent inserts can draw levels.
    rng_state: std::sync::atomic::AtomicU64,
    /// Optional persistent vector tier backing (truth tier
    /// f32 + quantized rerank tier). `None` for in-memory-only indexes
    /// (tests, ad-hoc analytics); `Some` when the caller has wired up
    /// LSM-backed storage. Writes through this handle log f32 on insert
    /// and quantized bytes on (re)calibration; reads through it power
    /// cross-shard rerank and application-side custom rerank.
    vector_tier: Option<crate::storage::VectorTierHandle>,
    /// Contiguous per-node RaBitQ code + scalar header, read by the cosine
    /// search fast path so a neighbour visit touches one stride-addressed
    /// block instead of the per-node `Vec` behind the node's code in the
    /// store. Allocated on the first insert of a RaBitQ-configured index;
    /// empty for other codecs. A `OnceLock` so the first of concurrent
    /// inserts creates it.
    // no-std: once_cell::race::OnceBox
    rabitq_block: std::sync::OnceLock<rabitq_block::RabitqBlock>,
    /// The node store: every node's f32 vector in one stride-addressed block
    /// per segment (hnswlib `data_level0_memory_`), its scalars, codes and
    /// neighbour lists, published whole. Created by the first insert, which
    /// fixes the dimension; a `OnceLock` so concurrent inserts create it once.
    // no-std: once_cell::race::OnceBox
    data_level0: std::sync::OnceLock<data_level0::DataLevel0Block>,
    /// Retired lists, lost CAS, admission waits and running operations.
    stats: std::sync::Arc<stats::PublicationStats>,
    /// The slots of removed and replaced nodes on their way to reuse.
    reclaim: std::sync::Arc<reclaim::SlotReclaim>,
    /// [`HnswConfig::retired_bytes_budget`], settable while the index serves.
    retired_bytes_budget: core::sync::atomic::AtomicUsize,
}

/// One search, insert or removal: the epoch pin that keeps every list and
/// slot it reads in place until it ends, and its entry in the running
/// operations.
struct Operation<'a> {
    guard: crossbeam_epoch::Guard,
    _tracked: stats::OperationGuard<'a>,
}

/// Per-layer outcome of planning an insert: which existing nodes the new node
/// connects to, plus the layer-specific max fan-out used for the
/// bidirectional prune.
#[derive(Debug, Clone)]
pub(crate) struct LayerPlan {
    pub level: usize,
    /// Indices of the chosen neighbours, ordered nearest-first.
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

/// Search strategy. Callers can opt into recall=1.0 brute-force kNN
/// instead of the HNSW approximate path, matching qdrant's
/// `SearchParams(exact=True)` semantics. The HNSW index continues to
/// own the stored vectors either way, so exact search reads the same
/// data without an auxiliary index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Graph traversal using the index's configured `ef_search`. Fast,
    /// approximate, typical recall 0.95-0.99 depending on `M` and `ef`.
    Hnsw,
    /// Linear scan over every indexed vector. Slow, exact, recall=1.0.
    /// Use when correctness matters more than throughput, or when the
    /// per-label index is small enough that brute force wins outright.
    Exact,
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
    /// 1 / ‖query‖₂ (0.0 for zero vectors) — lets the per-visit cosine
    /// turn its division into a multiply when the node's inverse norm
    /// is also cached. Cosine only; other metrics ignore it.
    inv_norm_l2: f32,
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
    /// stored codes (symmetric LUT kernel).
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
            inv_norm_l2: inv_or_zero(norm_l2),
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
            inv_norm_l2: inv_or_zero(norm_l2),
            rabitq_query: None,
        }
    }
}

/// `1 / n`, or 0.0 when `n` is too small to invert safely. The zero
/// sentinel makes `dot * inv_a * inv_b` evaluate to 0.0 for zero
/// vectors — the same "no direction" answer the division-based cosine
/// helpers return through their epsilon guard.
#[inline]
fn inv_or_zero(n: f32) -> f32 {
    if n < f32::EPSILON { 0.0 } else { 1.0 / n }
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
        let id_to_idx = id_map::IdMap::with_capacity(capacity);
        let retired_bytes_budget =
            core::sync::atomic::AtomicUsize::new(config.retired_bytes_budget);
        Self {
            stats: std::sync::Arc::new(stats::PublicationStats::new()),
            reclaim: std::sync::Arc::new(reclaim::SlotReclaim::default()),
            retired_bytes_budget,
            config,
            node_count: core::sync::atomic::AtomicUsize::new(0),
            live_count: core::sync::atomic::AtomicUsize::new(0),
            id_to_idx,
            entry_point: EntryPoint::new(),
            // max_level is derived from entry_point.load() at every
            // search start (top→bottom layer iteration). No separate
            // field — the packed AtomicU64 carries it.
            level_mult,
            sq8_params: None,
            rabitq_params: None,
            visited_pool: VisitedPool::new(),
            search_scratch_pool: SearchScratchPool::new(),
            // Seed from address of self (varies per instance). Non-deterministic but fast.
            rng_state: std::sync::atomic::AtomicU64::new(0xdeadbeef_cafebabe),
            vector_tier: None,
            rabitq_block: std::sync::OnceLock::new(),
            data_level0: std::sync::OnceLock::new(),
        }
    }

    /// Wire a persistent vector tier backend (truth f32 + quantized
    /// rerank). After this call every successful insert writes the
    /// f32 bytes to the truth tier, and every (re)calibration writes
    /// quantized codes to the rerank tier. Failures during tier writes
    /// are logged through `tracing::warn` but do NOT roll back the
    /// in-RAM insert — the in-RAM graph is authoritative; the tier is
    /// rebuilt from data on recovery.
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
        let encoded: Vec<(usize, Option<RabitqEncoded>)> = (0..self.node_len())
            .map(|i| {
                let enc = self
                    .read_node_f32(i)
                    .and_then(|v| self.encode_rabitq(&params, v));
                (i, enc)
            })
            .collect();
        for (i, enc) in encoded {
            // SAFETY: `&mut self` excludes every other reader and writer.
            unsafe { self.mirror_rabitq_to_block(i, enc.as_ref()) };
            if let Some(store) = self.nodes_mut() {
                store.set_rabitq(i, enc);
            }
        }
        self.rabitq_params = Some(params);
    }

    /// Manually set SQ8 calibration parameters (e.g., from a saved index).
    /// Quantizes all existing nodes that don't have quantized vectors yet.
    /// If `offload_vectors` is enabled, drops f32 after quantizing.
    pub fn set_sq8_params(&mut self, params: Sq8Params) {
        self.quantize_all(&params);
        // Offload: free the f32 from the layer-0 store so the RAM is actually
        // returned and `read_node_f32` reports None (rerank then loads f32
        // from disk).
        if self.config.offload_vectors {
            if let Some(b) = self.data_level0.get_mut() {
                b.drop_f32();
            }
        }
        self.sq8_params = Some(params);
    }

    /// Give every node with an in-memory f32 vector its SQ8 code under
    /// `params`.
    fn quantize_all(&mut self, params: &Sq8Params) {
        for idx in 0..self.node_len() {
            if let Some(code) = self.read_node_f32(idx).map(|v| params.quantize(v)) {
                if let Some(store) = self.nodes_mut() {
                    store.set_sq8(idx, Some(code));
                }
            }
        }
    }

    /// Check if the node at `idx` has an in-memory f32 vector. Offloaded
    /// nodes (f32 on disk) report `false`.
    pub fn has_f32_vector(&self, idx: usize) -> bool {
        self.read_node_f32(idx).is_some()
    }

    /// Get a reference to the f32 vector at node index `idx`, if present.
    ///
    /// Takes `&mut self`: through a shared borrow an insert could reuse the
    /// slot of a removed node and rewrite the vector under the reference.
    pub fn get_vector(&mut self, idx: usize) -> Option<&[f32]> {
        self.read_node_f32(idx)
    }

    /// The in-memory f32 vectors of the nodes that are results (removed and
    /// replaced nodes left out), for a calibration that holds the index
    /// exclusively.
    pub fn calibration_vectors(&mut self) -> Vec<&[f32]> {
        let Some(store) = self.data_level0.get() else {
            return Vec::new();
        };
        (0..self.node_len())
            .filter(|&idx| store.state(idx) == data_level0::NodeState::Live)
            .filter_map(|idx| store.vector(idx))
            .collect()
    }

    /// Auto-calibrate SQ8 from all currently stored vectors.
    /// Called automatically when `quantization` is enabled and
    /// `calibration_threshold` vectors have been inserted.
    ///
    /// Logs a warning if the index has fewer than [`SQ8_MIN_VECTORS`] (1000)
    /// vectors — quantization overhead is not justified for small indexes.
    fn auto_calibrate(&mut self) {
        if self.node_len() < SQ8_MIN_VECTORS {
            warn!(
                vectors = self.node_len(),
                min_recommended = SQ8_MIN_VECTORS,
                "SQ8 quantization on small index (<{} vectors): \
                 calibration storage overhead may exceed memory savings. \
                 Consider disabling quantization for small datasets",
                SQ8_MIN_VECTORS,
            );
        }
        let refs = self.calibration_vectors();
        if let Some(params) = Sq8Params::calibrate(&refs) {
            self.quantize_all(&params);
            // Offload: free the f32 from the layer-0 store so the RAM is
            // actually returned and `read_node_f32` reports None for these
            // nodes (rerank then loads f32 from disk via VectorLoader).
            if self.config.offload_vectors {
                if let Some(b) = self.data_level0.get_mut() {
                    b.drop_f32();
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
    /// one.
    fn auto_calibrate_rabitq(&mut self) {
        // Need at least one vector to infer D.
        let dims = match (0..self.node_len()).find_map(|idx| self.read_node_f32(idx)) {
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
        let training: Vec<Vec<f32>> = self
            .calibration_vectors()
            .into_iter()
            .map(<[f32]>::to_vec)
            .collect();
        let params = if training.is_empty() {
            RaBitQParams::calibrate(dims as u32, seed)
        } else {
            RaBitQParams::calibrate_with_kmeans(dims as u32, seed, &training, N_CLUSTERS)
        };

        // Two-pass borrow split — encode_rabitq needs `&self.config`
        // while encoding, then we mutate node.rabitq_code separately.
        let encoded: Vec<(usize, Option<RabitqEncoded>)> = (0..self.node_len())
            .map(|i| {
                let enc = self
                    .read_node_f32(i)
                    .and_then(|v| self.encode_rabitq(&params, v));
                (i, enc)
            })
            .collect();
        for (i, enc) in encoded {
            // SAFETY: `&mut self` excludes every other reader and writer.
            unsafe { self.mirror_rabitq_to_block(i, enc.as_ref()) };
            if let Some(store) = self.nodes_mut() {
                store.set_rabitq(i, enc);
            }
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
        self.live_count.load(core::sync::atomic::Ordering::Acquire)
    }

    /// Whether a vector for `id` is in the graph.
    pub fn contains(&self, id: u64) -> bool {
        self.id_to_idx.contains(id)
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a vector into the index.
    ///
    /// If the node `id` is already in the index, its vector is updated and its
    /// graph position is rebuilt. This handles the `MATCH (n) SET
    /// n.emb = $new_vec` path where `on_vector_written` calls `insert()` for
    /// both CREATE and SET.
    ///
    /// Through [`Self::insert_shared`], then the calibration it reports due.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        if self.insert_shared(id, &vector) {
            self.calibrate_if_due();
        }
    }

    /// Batched insert: the items go in concurrently across the rayon pool,
    /// each through [`Self::insert_shared`] against the live graph, so every
    /// insert links to what the earlier ones already published.
    ///
    /// Until the graph holds `SEED_DENSITY` nodes the items go in one by one:
    /// concurrent inserts into a near-empty graph would mostly see each other
    /// still unlinked and connect poorly. Repeated ids keep their last
    /// vector, as if inserted in order.
    ///
    /// Through [`Self::insert_batch_shared`], then the calibration it reports
    /// due.
    pub fn insert_batch(&mut self, items: Vec<(u64, Vec<f32>)>) {
        if self.insert_batch_shared(items) {
            self.calibrate_if_due();
        }
    }

    /// [`Self::insert_batch`] through a shared borrow, concurrently with
    /// searches and other inserts. Returns whether a calibration is now due;
    /// the caller runs [`Self::calibrate_if_due`] once it holds the index
    /// exclusively.
    pub fn insert_batch_shared(&self, mut items: Vec<(u64, Vec<f32>)>) -> bool {
        use rayon::prelude::*;
        const SEED_DENSITY: usize = 64;
        // A batch of at least a quarter of the graph runs the
        // reachability repair after it.
        const CONNECT_SHARE: usize = 4;
        // Bound on repair passes; each finds fewer unreached nodes.
        const CONNECT_PASSES: usize = 8;

        self.screen_dims(&mut items);
        // Concurrent inserts of one id finish in any order; keeping only the
        // last occurrence keeps the batch's own order meaningful.
        let mut last: rustc_hash::FxHashMap<u64, usize> = rustc_hash::FxHashMap::default();
        for (pos, (id, _)) in items.iter().enumerate() {
            last.insert(*id, pos);
        }
        let mut items: Vec<(u64, Vec<f32>)> = items
            .into_iter()
            .enumerate()
            .filter(|(pos, (id, _))| last.get(id) == Some(pos))
            .map(|(_, item)| item)
            .collect();

        // As many one-by-one inserts as the graph lacks to reach the seed
        // density; none once it is that dense (hence the clamp at zero).
        let seed = items.len().min(SEED_DENSITY.saturating_sub(self.len()));
        let rest = items.split_off(seed);
        for (id, vector) in items {
            self.insert_shared(id, &vector);
        }
        rest.par_iter().for_each(|(id, vector)| {
            self.insert_shared(*id, vector);
        });
        // A batch this large is a build: one pass over the graph per node it
        // adds four times over is cheap next to linking them.
        if !rest.is_empty() && rest.len() * CONNECT_SHARE >= self.len() {
            // A pass that found nothing unreached proves the graph whole; a
            // later pass sees what an earlier one's edges reached.
            for _ in 0..CONNECT_PASSES {
                if self.connect_unreachable() == 0 {
                    break;
                }
            }
        }
        self.calibration_due()
    }

    /// Bulk-build path for static corpora where every vector is known
    /// at call time. Prefer this over a manual `insert_batch` loop
    /// when the caller has the full dataset in hand.
    ///
    /// Below the module's threshold the call falls through to
    /// [`Self::insert_batch`] verbatim because the leader-seed and
    /// cluster-ordering overhead does not pay back at small N. Above
    /// the threshold the call routes through three stages:
    ///
    /// 1. Sample `floor(sqrt(N))` leaders deterministically (xorshift
    ///    Fisher-Yates over the input index set; reproducible for a
    ///    given N).
    /// 2. Seed the upper graph with sequential `insert` calls over
    ///    just the leaders; each leader's plan sees every prior
    ///    leader, producing a sparse, well-formed entry topology
    ///    before any follower lands.
    /// 3. Brute-force-assign followers to their nearest leader via
    ///    rayon, stable-sort followers by cluster id so a cluster's
    ///    items are contiguous, and hand the reordered batch to
    ///    [`Self::insert_batch`]. The apply phase then visits adjacent
    ///    `nodes[]` indices for a whole cluster before moving on,
    ///    giving the cache a working set that fits between graph
    ///    pointer chases.
    ///
    /// Each stage above leaves the graph in the same observable state
    /// a plain `insert_batch` would; the change is order and locality,
    /// not topology. A future task may extend this with cluster-
    /// restricted plans + parallel per-cluster builds (the full
    /// ParlayANN topology); that work requires threading an
    /// allowed-node bitmap through the search-internals and is
    /// deliberately not part of this entry point.
    pub fn bulk_build(&mut self, mut items: Vec<(u64, Vec<f32>)>) {
        self.screen_dims(&mut items);
        if items.len() < bulk_build::BULK_BUILD_THRESHOLD {
            self.insert_batch(items);
            return;
        }
        bulk_build::bulk_build(self, items, false);
    }

    /// Bulk-build then run a BFS cache-locality reorder.
    /// Renumbers nodes so graph-adjacent nodes are memory-adjacent, trading a
    /// one-off post-build pass for better search cache locality. Prefer this for
    /// read-heavy indexes built once and queried many times; use
    /// [`bulk_build`](Self::bulk_build) when the index keeps mutating.
    pub fn bulk_build_cache_optimized(&mut self, mut items: Vec<(u64, Vec<f32>)>) {
        self.screen_dims(&mut items);
        if items.len() < bulk_build::BULK_BUILD_THRESHOLD {
            self.insert_batch(items);
            self.reorder_for_cache_locality();
            return;
        }
        bulk_build::bulk_build(self, items, true);
    }

    /// Choose the neighbours of a node at `vector` on every layer from
    /// `new_level` down: the greedy descent and ef-search from the current
    /// entry point, against the live graph. `None` while the index has no
    /// entry point, i.e. no node to link to.
    fn plan_links(&self, vector: &[f32], new_level: usize) -> Option<Vec<LayerPlan>> {
        // The single-load snapshot gives `(start_idx, top_level)` from ONE
        // atomic read.
        let (start_idx, top_level) = self.entry_point.for_search()?;
        let mut current_ep = start_idx;

        // Step 1: greedy descent down to new_level + 1.
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

        // Step 2: select neighbours at every layer from new_level down to 0.
        let lowest_planning_layer = new_level.min(top_level);
        let mut per_layer = Vec::with_capacity(lowest_planning_layer + 1);
        for level in (0..=lowest_planning_layer).rev() {
            let ef = self.config.ef_construction;
            // Same f32-build rationale as step 1 above.
            let candidates = self.search_layer_query_for_build(vector, current_ep, ef, level);

            let max_conn = if level == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };
            // RobustPrune when alpha > 1.0; legacy "take M closest"
            // otherwise. Both feed the same downstream insert plan path.
            let selected: Vec<usize> = if self.config.effective_alpha() > 1.0 {
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

        Some(per_layer)
    }

    /// The node store for vectors of `dim` components, created on the first
    /// call. `None` when `dim` is zero or differs from the store's:
    /// concurrent first inserts of different dimensions race to create it,
    /// and only the winner's fits.
    fn store_for(&self, dim: usize) -> Option<&data_level0::DataLevel0Block> {
        if dim == 0 {
            return None;
        }
        let block = self.data_level0.get_or_init(|| {
            // Size the lists to the effective per-node layer-0 degree
            // (`config.m_max0`, already capped to `M_MAX0`), not the
            // compile-time `M_MAX0` cap.
            data_level0::DataLevel0Block::new(
                (self.config.max_elements as usize).max(1),
                self.config.m_max0,
                dim,
            )
        });
        (block.dim() == dim).then_some(block)
    }

    /// Store the f32 vector of node `idx` in `block`, growing it to fit
    /// `idx`: the store holds every node's lists and scalars, including those
    /// inserted beyond the initial `max_elements` estimate, whether or not the
    /// vectors were offloaded.
    ///
    /// # Safety
    ///
    /// `vector.len() == block.dim()`, and the caller is the only writer of
    /// node `idx` and writes before the node is reachable, or holds the index
    /// exclusively.
    unsafe fn place_vector(block: &data_level0::DataLevel0Block, idx: usize, vector: &[f32]) {
        block.ensure_capacity(idx + 1);
        if block.has_f32() {
            // SAFETY: idx < capacity after `ensure_capacity`; the length and
            // exclusivity are the caller's.
            unsafe { block.set_vector(idx, vector) };
        }
    }

    /// Drop, with a warning, the batch items whose dimension is zero or
    /// differs from the index's (or, before the first insert, from the first
    /// non-empty item's). Batch planning and cluster assignment compare items
    /// with each other, and the distance kernels assume equal lengths.
    fn screen_dims(&self, items: &mut Vec<(u64, Vec<f32>)>) {
        let reference = self
            .data_level0
            .get()
            .map(data_level0::DataLevel0Block::dim)
            .or_else(|| items.iter().map(|(_, v)| v.len()).find(|&d| d > 0));
        items.retain(|(id, vec)| {
            let accepted = !vec.is_empty() && Some(vec.len()) == reference;
            if !accepted {
                warn!(
                    node_id = *id,
                    dim = vec.len(),
                    index_dim = reference,
                    "HNSW insert rejected: vector dimension is zero or differs from the index"
                );
            }
            accepted
        });
    }

    /// Allocate the RaBitQ code block on the first insert of a RaBitQ index,
    /// once `dim` is known, with the code width of the configured codec so
    /// calibration fills it without a layout mismatch. Other codecs never
    /// produce RaBitQ codes and get no block.
    fn ensure_rabitq_block(&self, idx: usize, dim: usize) {
        if dim == 0 || self.rabitq_block.get().is_some() {
            return;
        }
        let QuantizationCodec::RaBitQ { bits } = self.config.quantization else {
            return;
        };
        if !(1..=4).contains(&bits) {
            return;
        }
        let capacity = (self.config.max_elements as usize).max(idx + 1);
        self.rabitq_block
            .get_or_init(|| rabitq_block::RabitqBlock::new_with_rabitq_bits(capacity, dim, bits));
    }

    /// Borrow the f32 vector of node `idx` from the layer-0 store. `None`
    /// before the first insert and once the vectors were offloaded to disk.
    #[inline]
    fn read_node_f32(&self, idx: usize) -> Option<&[f32]> {
        self.data_level0.get()?.vector(idx)
    }

    /// Read the layer-0 neighbour id snapshot into `out` from the
    /// contiguous store when present, falling back to the SoA
    /// `AtomicNeighbourList` otherwise. Keeps the search hot path on a
    /// single allocation per node so concurrent workers visiting the same
    /// graph share L3 fills, the cache-locality property that drives
    /// hnswlib's super-linear MT4 scaling on sift-128 f32.
    #[inline]
    fn read_layer0_neighbours_into(&self, idx: usize, out: &mut Vec<u64>) {
        let guard = crossbeam_epoch::pin();
        self.read_layer0_neighbours_with(idx, out, &guard);
    }

    /// [`Self::read_layer0_neighbours_into`] under the caller's epoch pin:
    /// the search hot path pins once per layer pass and reads every visited
    /// list under that one guard.
    #[inline]
    fn read_layer0_neighbours_with(
        &self,
        idx: usize,
        out: &mut Vec<u64>,
        guard: &crossbeam_epoch::Guard,
    ) {
        // `data_level0` holds every node's published layer-0 list (u32 ids);
        // read straight into `out`, widening u32 -> u64.
        if let Some(block) = self.data_level0.get() {
            if block.read_list_u64(idx, out, guard) {
                return;
            }
        }
        // data_level0 is the sole layer-0 neighbour store and is grown to
        // cover every node during allocation, so this fallback only fires for
        // an idx with no block yet (no nodes inserted) — an empty list.
        out.clear();
    }

    /// Test accessor: the entry point's current node idx, if any. Reads private
    /// lock-free state the reorder tests need to assert BFS rooting.
    #[cfg(test)]
    fn entry_point_idx_for_test(&self) -> Option<usize> {
        // `for_search` yields `(idx, top_level)`.
        self.entry_point.for_search().map(|(idx, _)| idx)
    }

    /// Test accessor: snapshot of node `idx`'s layer-0 neighbour idxs.
    #[cfg(test)]
    fn layer0_neighbours_for_test(&self, idx: usize) -> Vec<u64> {
        let mut out = Vec::new();
        self.read_layer0_neighbours_into(idx, &mut out);
        out
    }

    /// Test accessor: resolve a node id to its current idx via the id→idx map.
    #[cfg(test)]
    fn idx_for_id_for_test(&self, id: u64) -> Option<usize> {
        self.id_to_idx.get(id)
    }

    /// Copy the RaBitQ code (packed bytes) and scalar header of node `idx`
    /// into the code block, so the slot holds the node's current code or
    /// nothing. When the code cannot go in (no code, a width or length that
    /// does not match the slot) the slot is cleared: a zero `norm` sends search
    /// to the node's code in the store, which costs speed, never a result.
    ///
    /// # Safety
    ///
    /// The caller is the only writer of node `idx` and writes before the node
    /// is reachable, or holds the index exclusively.
    unsafe fn mirror_rabitq_to_block(&self, idx: usize, enc: Option<&RabitqEncoded>) {
        let Some(block) = self.rabitq_block.get() else {
            return;
        };
        if idx >= block.capacity() {
            return;
        }
        // SAFETY: forwarded caller exclusivity over node idx.
        let installed = enc.is_some_and(|enc| unsafe { Self::install_rabitq(block, idx, enc) });
        if !installed {
            // SAFETY: idx < capacity per the gate above; caller exclusivity.
            unsafe { block.set_rabitq_scalars(idx, rabitq_block::RaBitQScalars::default()) };
        }
    }

    /// Write `enc` into slot `idx` of `inline`; `false` when its width or
    /// length does not match the slot.
    ///
    /// # Safety
    ///
    /// `idx < inline.capacity()`, and the caller is the only writer of node
    /// `idx` and writes before the node is reachable, or holds the block
    /// exclusively.
    unsafe fn install_rabitq(
        inline: &rabitq_block::RabitqBlock,
        idx: usize,
        enc: &RabitqEncoded,
    ) -> bool {
        debug_assert!(idx < inline.capacity(), "idx out of the code block");
        match enc {
            RabitqEncoded::OneBit(code) => {
                if inline.rabitq_bits() != 1 {
                    return false;
                }
                // `CodeWords` derefs to `&[u64]`; reinterpret as bytes for
                // the packed-code slot.
                let words: &[u64] = code.code.as_slice();
                let byte_len = core::mem::size_of_val(words);
                // SAFETY: `words` outlives the slice borrow; reinterpreting
                // an aligned `&[u64]` as `&[u8]` is sound.
                let byte_slice =
                    unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, byte_len) };
                // The words round the code up to whole u64s; the slot holds
                // the exact byte length. A code shorter than the slot was
                // encoded for another dimension: skip it rather than install
                // a partial code under valid scalars.
                let dst_len = inline.rabitq_byte_len();
                if byte_slice.len() < dst_len {
                    return false;
                }
                let scalars = rabitq_block::RaBitQScalars {
                    norm: code.norm,
                    cross_term: code.cross_term,
                    signed_sum: code.signed_sum,
                    correction: code.correction,
                    radial: code.radial,
                    cluster_id: code.cluster_id,
                    _pad: 0,
                };
                // SAFETY: idx < capacity per the caller; the slice is exactly
                // the slot length.
                unsafe {
                    inline.set_rabitq(idx, &byte_slice[..dst_len]);
                    inline.set_rabitq_scalars(idx, scalars);
                }
                true
            }
            RabitqEncoded::Multi(code) => {
                if inline.rabitq_bits() != code.bits
                    || code.packed.len() != inline.rabitq_byte_len()
                {
                    return false;
                }
                let scalars = rabitq_block::RaBitQScalars {
                    norm: code.norm,
                    cross_term: code.cross_term,
                    signed_sum: 0,
                    correction: 0.0,
                    radial: 0.0,
                    cluster_id: 0,
                    _pad: 0,
                };
                // SAFETY: idx < capacity per the caller; the packed code is
                // exactly the slot length per the gate above.
                unsafe {
                    inline.set_rabitq(idx, &code.packed);
                    inline.set_rabitq_scalars(idx, scalars);
                }
                true
            }
        }
    }

    /// The RaBitQ code block, for tests.
    #[cfg(test)]
    pub(crate) fn rabitq_block(&self) -> Option<&rabitq_block::RabitqBlock> {
        self.rabitq_block.get()
    }

    /// Add node `id` with no links yet, in a slot of its own: its vector,
    /// norms, codes and `new_level` empty lists above layer 0. The node stays
    /// [`Reserved`](data_level0::NodeState::Reserved) and unreachable until
    /// the caller links it. `None` when the vector's dimension is zero or
    /// differs from the index's; then no slot is taken.
    fn allocate_node(&self, id: u64, new_level: usize, vector: &[f32]) -> Option<usize> {
        let Some(block) = self.store_for(vector.len()) else {
            warn!(
                node_id = id,
                dim = vector.len(),
                index_dim = self
                    .data_level0
                    .get()
                    .map(data_level0::DataLevel0Block::dim),
                "HNSW insert rejected: vector dimension is zero or differs from the index"
            );
            return None;
        };
        // The slot is this insert's alone: no link or entry point names it
        // until the caller links the node. A reused slot is reachable from
        // nothing either: it became free only after every operation that
        // could still hold it had ended.
        let idx = match self.reclaim.take_free() {
            Some(idx) => {
                block.set_state(idx, data_level0::NodeState::Reserved);
                // SAFETY: idx < capacity (the slot was in use); its old
                // layer-0 list names nodes the new one has no edge to.
                unsafe {
                    block.update_neighbours(
                        idx,
                        |ids| (!ids.is_empty()).then(Box::default),
                        &self.stats,
                    )
                };
                idx
            }
            None => self
                .node_count
                .fetch_add(1, core::sync::atomic::Ordering::AcqRel),
        };
        // SAFETY: the dimension matches the store per `store_for`, and the
        // slot is unreachable and written by this insert only.
        unsafe { Self::place_vector(block, idx, vector) };

        // Quantize if SQ8 is calibrated.
        let quantized = self.sq8_params.as_ref().map(|p| p.quantize(vector));
        // Encode if RaBitQ is calibrated. Encoded against the rotation matrix
        // already chosen at calibration time — codes from before vs after
        // calibration are not interchangeable, so this branch only fires
        // post-calibration (pre-calibration nodes get a code on first
        // calibration via `auto_calibrate_rabitq`). Variant is picked
        // from config so 1-bit vs 2/3/4-bit indexes stay homogeneous.
        let rabitq_code = self
            .rabitq_params
            .as_ref()
            .and_then(|p| self.encode_rabitq(p, vector));

        // Persist the f32 truth tier. Quantized codes (SQ8 / RaBitQ /
        // PolarQuant / PQ) stay in RAM only; cross-shard rerank reads
        // f32 directly from the truth tier. Tier writes never roll back
        // the in-RAM insert: the in-RAM graph is authoritative, and the
        // truth tier regenerates from data on recovery.
        if let Some(tier) = self.vector_tier.as_ref() {
            if let Err(e) = tier.put_f32(id, vector) {
                warn!(node_id = id, error = %e, "vector_tier put_f32 failed");
            }
        }

        self.ensure_rabitq_block(idx, vector.len());
        // SAFETY: `place_vector` grew the store to cover idx; the slot is
        // unreachable and written by this insert only.
        unsafe {
            self.mirror_rabitq_to_block(idx, rabitq_code.as_ref());
            block.init_node(idx, id, metrics::norm_l2(vector), new_level);
            block.set_codes(idx, quantized, rabitq_code);
        }
        Some(idx)
    }

    /// Insert `id` with `vector`, or move an existing `id` to `vector`,
    /// through a shared borrow: inserts and searches run concurrently on the
    /// same index, with no lock over the graph.
    ///
    /// The node is built in a slot of its own and becomes reachable only once
    /// its payload and lists are in place: through its links, then the entry
    /// point. An update builds the new node the same way and then retires the
    /// old one, which stays navigable for readers that still reach it but is
    /// never a result again; a node's vector never changes in place under a
    /// reader. Inserting the same vector again is a no-op.
    ///
    /// Returns whether the codec's calibration threshold is reached with no
    /// calibration run yet. Calibration rewrites every node's code and takes
    /// the index exclusively, so the caller runs it with
    /// [`Self::calibrate_if_due`].
    pub fn insert_shared(&self, id: u64, vector: &[f32]) -> bool {
        self.admit();
        let op = self.begin_operation();
        if let Some(current) = self.id_to_idx.get(id) {
            if self.read_node_f32(current) == Some(vector) {
                return false;
            }
        }
        let new_level = self.random_level();
        let Some(idx) = self.allocate_node(id, new_level, vector) else {
            return false;
        };
        self.link_node(idx, id, new_level, vector);
        self.nodes().set_state(idx, data_level0::NodeState::Live);
        match self.id_to_idx.insert(id, idx) {
            Some(old) => {
                // The descent must not start from a node that is never a
                // result: with a single node it is the only way in.
                self.entry_point
                    .try_replace(old as u64, new_level as u8, idx as u64);
                self.retire_node(old, &op.guard);
            }
            None => {
                self.live_count
                    .fetch_add(1, core::sync::atomic::Ordering::AcqRel);
            }
        }
        // Linked, so it may now head the descent if it reached a new top
        // layer.
        let _ = self.entry_point.try_promote(new_level as u8, idx as u64);
        self.calibration_due()
    }

    /// Remove `id` through a shared borrow, concurrently with searches and
    /// inserts. Its node stops being a result at once; its slot is reused by
    /// a later insert once no list names it and no operation that could
    /// reach it is still running. Returns whether `id` was in the index.
    pub fn remove(&self, id: u64) -> bool {
        self.admit();
        let op = self.begin_operation();
        let Some(idx) = self.id_to_idx.remove(id) else {
            return false;
        };
        // No longer a result before the count says it is gone: a reader
        // that sees the count drop finds no more results than it says.
        self.nodes().set_state(idx, data_level0::NodeState::Retired);
        self.live_count
            .fetch_sub(1, core::sync::atomic::Ordering::AcqRel);
        self.hand_over_entry_point(idx);
        self.retire_node(idx, &op.guard);
        true
    }

    /// Move the entry point off node `old`, which is being removed: to its
    /// nearest linkable neighbour on the highest layer that has one, else to
    /// any node that can be a result, else nowhere (the index is empty).
    fn hand_over_entry_point(&self, old: usize) {
        if self.entry_point.for_search().map(|(idx, _)| idx) != Some(old) {
            return;
        }
        let store = self.nodes();
        let n = self.node_len();
        for level in (0..self.node_levels(old)).rev() {
            for nb in self.layer_snapshot(old, level) {
                let nb = nb as usize;
                if nb < n && nb != old && store.state(nb) == data_level0::NodeState::Live {
                    let top = self.node_levels(nb) - 1;
                    if self
                        .entry_point
                        .try_replace(old as u64, top as u8, nb as u64)
                    {
                        return;
                    }
                    // Someone else moved it already.
                    return;
                }
            }
        }
        // No neighbour left: the rare isolated node. A linear pass finds any
        // node still serving; the entry point must not stay on a node whose
        // slot is about to be reused.
        if let Some(nb) =
            (0..n).find(|&i| i != old && store.state(i) == data_level0::NodeState::Live)
        {
            let top = self.node_levels(nb) - 1;
            self.entry_point
                .try_replace(old as u64, top as u8, nb as u64);
        } else {
            self.entry_point.try_clear(old as u64);
        }
    }

    /// Pin the epoch and track the operation: every list and slot read until
    /// the returned value drops stays in place.
    #[inline]
    fn begin_operation(&self) -> Operation<'_> {
        Operation {
            _tracked: self.stats.begin(),
            guard: crossbeam_epoch::pin(),
        }
    }

    /// Hold an insert or removal off while the replaced lists awaiting
    /// reclamation exceed the budget. Never called under a pin: a writer
    /// waiting here must not itself hold back the reclamation it waits for.
    #[inline]
    fn admit(&self) {
        let budget = self
            .retired_bytes_budget
            .load(core::sync::atomic::Ordering::Relaxed);
        if self.stats.retired_bytes() <= budget {
            return;
        }
        self.wait_for_reclamation(budget);
    }

    #[cold]
    #[inline(never)]
    fn wait_for_reclamation(&self, budget: usize) {
        let started = std::time::Instant::now();
        let mut last = usize::MAX;
        loop {
            // Running the deferred frees of this thread and advancing the
            // epoch is the part of reclamation a writer can do itself; the
            // rest waits for the operations still pinned to finish.
            crossbeam_epoch::pin().flush();
            let now = self.stats.retired_bytes();
            if now <= budget {
                break;
            }
            // With no operation running nothing protects what is left: it
            // sits in the bounded per-thread queues of threads that have not
            // pinned since, and waiting would not free it.
            if !self.stats.any_running() && now >= last {
                break;
            }
            last = now;
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        self.stats
            .admission_waited(started.elapsed().as_nanos() as u64);
    }

    /// Set [`HnswConfig::retired_bytes_budget`] while the index serves; the
    /// next insert or removal sees it.
    pub fn set_retired_bytes_budget(&self, bytes: usize) {
        self.retired_bytes_budget
            .store(bytes, core::sync::atomic::Ordering::Relaxed);
    }

    /// Publication and reclamation figures: replaced lists not yet freed,
    /// lost compare-and-swaps, admission waits, the oldest running operation
    /// and the slots on their way to reuse.
    pub fn publication_stats(&self) -> PublicationSnapshot {
        let (retired_nodes, free_slots) = self.reclaim.counts();
        PublicationSnapshot {
            retired_nodes,
            free_slots,
            ..self.stats.snapshot()
        }
    }

    /// Link the freshly allocated node `idx` into the graph: plan against
    /// the live graph, publish its own lists, then the back-edges that make it
    /// reachable. The first node has no one to link to and seeds the entry
    /// point instead; losing that race to another first node means linking to
    /// it.
    fn link_node(&self, idx: usize, id: u64, new_level: usize, vector: &[f32]) {
        let per_layer = loop {
            if let Some(plan) = self.plan_links(vector, new_level) {
                break plan;
            }
            if self.entry_point.try_seed(new_level as u8, idx as u64) {
                return;
            }
        };
        let store = self.nodes();
        let per_layer: Vec<(usize, Vec<u64>, usize)> = per_layer
            .into_iter()
            .map(
                |LayerPlan {
                     level,
                     selected_idxs,
                     max_conn,
                 }| {
                    // An older node of the same id is about to be retired, and a
                    // retired node is never a result: neither is worth a slot.
                    // Never linking a retired node is also what lets its slot
                    // be reused once the writers that saw it live are done.
                    let selected = selected_idxs
                        .into_iter()
                        .filter(|&n| {
                            n != idx && store.state(n).is_linkable() && self.node_id(n) != id
                        })
                        .map(|n| n as u64)
                        .collect();
                    (level, selected, max_conn)
                },
            )
            .collect();
        // Every own list goes in before the first back-edge makes the node
        // reachable: a search that reaches it on one layer descends through
        // it, and an empty list below would leave that search nowhere to go.
        for (level, selected, _) in &per_layer {
            self.layer_update(idx, *level, |_| Some(selected.clone()));
        }
        for (level, selected, max_conn) in &per_layer {
            for &n in selected {
                let n = n as usize;
                if *level < self.node_levels(n) {
                    self.add_neighbour_to(n, *level, idx as u64, *max_conn);
                }
            }
        }
    }

    /// Make every live node reachable on layer 0 from the entry point again,
    /// as Lucene's builder connects the components of a finished graph.
    ///
    /// Pruning a full list may drop a node's last incoming edge, and inserts
    /// in flight together never see each other, so close nodes inserted at
    /// once can end up linked only among themselves. A search cannot reach
    /// such a node, whatever its ef. Each unreached node gets an edge from
    /// the nearest reached node a search finds, preferring one with room; a
    /// full list gives up its farthest neighbour that another reached node
    /// also links. Everything the linked node reaches is then reached too, so
    /// one edge connects a whole component.
    ///
    /// One pass over every list, so it runs after a batch that is a large
    /// share of the graph (a build), not after each insert. Nodes whose f32
    /// vector was offloaded cannot be searched for and are left as they are.
    /// Returns how many live nodes were unreached when the pass began; a pass
    /// that found none left nothing to repair.
    fn connect_unreachable(&self) -> usize {
        let Some((entry, _)) = self.entry_point.for_search() else {
            return 0;
        };
        let store = self.nodes();
        let n = self.node_len();
        let live = |i: usize| store.state(i) == data_level0::NodeState::Live;
        let mut list = Vec::new();
        let mut reached = vec![false; n];
        let mut frontier = Vec::new();
        let mut reach_from = |start: usize, reached: &mut [bool], list: &mut Vec<u64>| {
            if reached[start] {
                return;
            }
            reached[start] = true;
            frontier.push(start);
            while let Some(i) = frontier.pop() {
                // A search walks the upper layers before layer 0, so an
                // edge on any layer of a reached node leads somewhere it
                // can get to.
                for level in 0..self.node_levels(i) {
                    self.layer_snapshot_into(i, level, list);
                    for &nb in list.iter() {
                        let nb = nb as usize;
                        if nb < n && !reached[nb] {
                            reached[nb] = true;
                            frontier.push(nb);
                        }
                    }
                }
            }
        };
        reach_from(entry, &mut reached, &mut list);
        // Layer-0 edges from reached nodes only: an edge from a node no
        // search gets to keeps nothing reachable.
        let mut in_degree = vec![0u32; n];
        for i in (0..n).filter(|&i| reached[i] && live(i)) {
            self.layer_snapshot_into(i, 0, &mut list);
            for &nb in &list {
                if let Some(d) = in_degree.get_mut(nb as usize) {
                    *d += 1;
                }
            }
        }

        let max_conn = self.config.m_max0;
        let mut unreached = 0;
        for u in 0..n {
            if reached[u] || !live(u) {
                continue;
            }
            unreached += 1;
            let Some(vector) = self.read_node_f32(u) else {
                continue;
            };
            let Some(plan) = self.plan_links(vector, 0) else {
                return unreached;
            };
            let candidates: Vec<usize> = plan
                .into_iter()
                .find(|p| p.level == 0)
                .map(|p| p.selected_idxs)
                .unwrap_or_default()
                .into_iter()
                .filter(|&p| p != u && p < n && reached[p] && live(p))
                .collect();
            let mut linked = false;
            for p in candidates {
                let mut evicted = None;
                let published = self.layer_update(p, 0, |current| {
                    evicted = None;
                    if current.contains(&(u as u64)) {
                        return None;
                    }
                    let mut next = current.to_vec();
                    if next.len() >= max_conn {
                        // Only a neighbour something else also reaches may
                        // go, or the repair would orphan it.
                        let w = next
                            .iter()
                            .copied()
                            .filter(|&w| in_degree.get(w as usize).is_some_and(|&d| d > 1))
                            .max_by(|&a, &b| {
                                self.distance_between_nodes(p, a as usize)
                                    .total_cmp(&self.distance_between_nodes(p, b as usize))
                            })?;
                        evicted = Some(w);
                        next.retain(|&x| x != w);
                    }
                    next.push(u as u64);
                    Some(next)
                });
                if published {
                    if let Some(w) = evicted {
                        in_degree[w as usize] -= 1;
                    }
                    linked = true;
                    break;
                }
            }
            if linked {
                in_degree[u] += 1;
                reach_from(u, &mut reached, &mut list);
            }
        }
        unreached
    }

    /// Retire node `old`, removed or replaced by a newer node of the same id:
    /// it stops being a result and a link target, and its neighbours drop
    /// their edges to it. Edges that other nodes keep to it still lead
    /// somewhere valid: the slot and its lists stay in place for readers that
    /// reach it until the sweep and the epoch have made it free.
    fn retire_node(&self, old: usize, guard: &crossbeam_epoch::Guard) {
        let store = self.nodes();
        store.set_state(old, data_level0::NodeState::Retired);
        let n = self.node_len();
        let dead = |id: u64| id == old as u64;
        for level in 0..self.node_levels(old) {
            let bridge = self.layer_snapshot(old, level);
            for &nb in &bridge {
                let nb = nb as usize;
                if nb < n && store.state(nb).is_linkable() && level < self.node_levels(nb) {
                    self.layer_update(nb, level, |current| {
                        self.repaired_list(nb, level, current, &dead, &bridge)
                    });
                }
            }
        }
        self.reclaim.retired(old, guard);
        // A sweep reads every list once, so it waits for enough slots to be
        // worth it: a sixteenth of the index, and never fewer than
        // `SWEEP_MIN_SLOTS`.
        if self.reclaim.unlinkable_len() >= (n / 16).max(SWEEP_MIN_SLOTS) {
            self.sweep_unlinkable(guard);
        }
    }

    /// `current`, the list of `idx` at `level`, without the slots `dead`
    /// accepts, refilled from `bridge` (the dead slots' own neighbours at that
    /// level) by the prune rule, so the gap they leave is bridged rather than
    /// cut (FreshDiskANN's delete consolidation). `None` when `current` names
    /// no dead slot.
    fn repaired_list(
        &self,
        idx: usize,
        level: usize,
        current: &[u64],
        dead: &impl Fn(u64) -> bool,
        bridge: &[u64],
    ) -> Option<Vec<u64>> {
        if !current.iter().any(|&nb| dead(nb)) {
            return None;
        }
        let store = self.nodes();
        let n = self.node_len();
        let kept: Vec<u64> = current.iter().copied().filter(|&nb| !dead(nb)).collect();
        let extras: Vec<u64> = bridge
            .iter()
            .copied()
            .filter(|&e| {
                let e_idx = e as usize;
                e_idx != idx
                    && !dead(e)
                    && e_idx < n
                    && store.state(e_idx).is_linkable()
                    && level < self.node_levels(e_idx)
                    && !kept.contains(&e)
            })
            .collect();
        let max_conn = if level == 0 {
            self.config.m_max0
        } else {
            self.config.m
        };
        Some(self.prune_selection(idx, &kept, max_conn, &extras))
    }

    /// Republish every list that names a slot no writer can link any more
    /// without it, then queue those slots for reuse. The entry point's slot
    /// stays queued until the entry point has moved off it.
    fn sweep_unlinkable(&self, guard: &crossbeam_epoch::Guard) {
        let store = self.nodes();
        self.reclaim.sweep_with(guard, |queued| {
            let entry = self.entry_point.for_search().map(|(idx, _)| idx);
            let (swept, kept): (Vec<usize>, Vec<usize>) =
                queued.into_iter().partition(|&idx| Some(idx) != entry);
            if swept.is_empty() {
                return (swept, kept);
            }
            let dead: rustc_hash::FxHashSet<u64> = swept.iter().map(|&idx| idx as u64).collect();
            let is_dead = |id: u64| dead.contains(&id);
            // The swept slots' own lists, per layer: what a list that loses
            // one of them is refilled from. Still readable, since nothing
            // reuses a slot before this sweep's pin has ended.
            let max_level = swept
                .iter()
                .map(|&d| self.node_levels(d))
                .max()
                .unwrap_or(0);
            let bridges: Vec<Vec<u64>> = (0..max_level)
                .map(|level| {
                    let mut bridge: Vec<u64> = swept
                        .iter()
                        .filter(|&&d| level < self.node_levels(d))
                        .flat_map(|&d| self.layer_snapshot(d, level))
                        .collect();
                    bridge.sort_unstable();
                    bridge.dedup();
                    bridge
                })
                .collect();
            let n = self.node_len();
            for idx in 0..n {
                // A reserved slot is still being written by its insert, which
                // started after these slots were retired and so never linked
                // them; a free one is unreachable.
                if !matches!(
                    store.state(idx),
                    data_level0::NodeState::Live | data_level0::NodeState::Retired
                ) || dead.contains(&(idx as u64))
                {
                    continue;
                }
                for level in 0..self.node_levels(idx) {
                    let bridge = bridges.get(level).map_or(&[][..], Vec::as_slice);
                    self.layer_update(idx, level, |current| {
                        self.repaired_list(idx, level, current, &is_dead, bridge)
                    });
                }
            }
            for &idx in &swept {
                store.set_state(idx, data_level0::NodeState::Free);
            }
            (swept, kept)
        });
    }

    /// Whether the configured codec has reached its calibration threshold
    /// and has not been calibrated.
    fn calibration_due(&self) -> bool {
        let reached = self.len() >= self.config.calibration_threshold;
        reached
            && match self.config.quantization {
                QuantizationCodec::Sq8 => self.sq8_params.is_none(),
                QuantizationCodec::RaBitQ { .. } => self.rabitq_params.is_none(),
                _ => false,
            }
    }

    /// Run the calibration [`Self::insert_shared`] reported as due, and the
    /// offload that follows it. A no-op when none is due.
    pub fn calibrate_if_due(&mut self) {
        if self.calibration_due() {
            self.maybe_calibrate_and_offload(0);
        }
    }

    /// Auto-calibrate SQ8 if threshold reached, then offload the just-inserted
    /// node's f32 if offloading is active. Called at the end of `insert()`.
    fn maybe_calibrate_and_offload(&mut self, _just_inserted_idx: usize) {
        // Step 1: Auto-calibrate the configured codec when threshold reached.
        let threshold_reached = self.len() >= self.config.calibration_threshold;
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

        // Offload is handled at calibration: `auto_calibrate` / `set_sq8_params`
        // free the f32 from the contiguous blocks (drop_f32), and the mirror is
        // gated so post-calibration inserts never write f32 for an offloaded
        // index. SQ8-only today; RaBitQ disk-rerank with SQ8 on disk is a
        // follow-up.
    }

    /// Search for K nearest neighbors.
    ///
    /// When SQ8 quantization is active, HNSW traversal uses approximate
    /// (dequantized) distances for candidate generation. The final top-K
    /// results are reranked using exact f32 distances.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let _op = self.begin_operation();
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
        let candidates = self.results_only(self.search_layer_query(query, current_ep, ef, 0));

        if self.is_quantized() {
            // Rerank candidates using exact f32 distance
            let mut reranked: Vec<SearchResult> = candidates
                .into_iter()
                .map(|c| {
                    let exact_dist = self.compute_exact_distance(&qctx, c.idx as usize);
                    SearchResult {
                        id: self.node_id(c.idx as usize),
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
                    id: self.node_id(c.idx as usize),
                    score: c.distance,
                })
                .collect()
        }
    }

    /// Search with explicit mode selection. `SearchMode::Hnsw` matches
    /// `search()`; `SearchMode::Exact` performs a brute-force linear scan
    /// over every indexed vector and returns recall=1.0 top-k. Both modes
    /// share the same stored vectors (no auxiliary index) and the same
    /// distance kernels, so callers can mix exact and approximate queries
    /// against one collection without reloading the data.
    pub fn search_with_mode(&self, query: &[f32], k: usize, mode: SearchMode) -> Vec<SearchResult> {
        match mode {
            SearchMode::Hnsw => self.search(query, k),
            SearchMode::Exact => self.search_exact(query, k),
        }
    }

    /// Brute-force exact k-NN over every indexed vector.
    ///
    /// Iterates `0..nodes.len()`, computes the exact f32 distance for each,
    /// and heap-selects the top-k. No graph traversal, no `ef_search`. The
    /// cost is linear in the index size and independent of `M`, so it is
    /// the right choice when the per-label collection is small enough that
    /// HNSW overhead exceeds a straight scan, or when recall=1.0 is a hard
    /// requirement (regulatory queries, ground-truth validation).
    fn search_exact(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let _op = self.begin_operation();
        let n = self.node_len();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let qctx = QueryCtx::new(
            query,
            self.config.metric,
            self.rabitq_params.as_ref(),
            &self.config.quantization,
        );
        // Max-heap keyed by distance; pop the current worst when the heap
        // grows past k. `FarCandidate` already implements `Ord` with
        // ascending distance, which makes `BinaryHeap` behave as a
        // max-heap over distance — exactly the shape we want for a
        // k-smallest selector.
        let mut heap: BinaryHeap<FarCandidate> = BinaryHeap::with_capacity(k + 1);
        let store = self.nodes();
        for idx in 0..n {
            // A scan reaches every slot, not only linked nodes: one still
            // being initialized or already replaced is not a result.
            if store.state(idx) != data_level0::NodeState::Live {
                continue;
            }
            let distance = self.compute_exact_distance(&qctx, idx);
            let cand = FarCandidate {
                distance,
                idx: idx as u32,
            };
            if heap.len() < k {
                heap.push(cand);
            } else if let Some(top) = heap.peek() {
                if cand.distance < top.distance {
                    heap.pop();
                    heap.push(cand);
                }
            }
        }
        let mut out: Vec<SearchResult> = heap
            .into_iter()
            .map(|c| SearchResult {
                id: self.node_id(c.idx as usize),
                score: c.distance,
            })
            .collect();
        out.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
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
        let _op = self.begin_operation();
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

            let candidates =
                self.results_only(self.search_layer_ctx(&qctx, current_ep, current_ef, 0));

            // Convert to SearchResult with exact distances (rerank if quantized)
            let results: Vec<SearchResult> = if self.is_quantized() {
                let mut reranked: Vec<SearchResult> = candidates
                    .into_iter()
                    .map(|c| SearchResult {
                        id: self.node_id(c.idx as usize),
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
                        id: self.node_id(c.idx as usize),
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
        let _op = self.begin_operation();

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
        let candidates = self.results_only(self.search_layer_ctx(&qctx, current_ep, ef, 0));

        // Batch-load f32 vectors from storage for reranking
        let candidate_ids: Vec<u64> = candidates
            .iter()
            .map(|c| self.node_id(c.idx as usize))
            .collect();
        let loaded = loader.load_vectors(&candidate_ids, &self.config.property_name);

        let mut reranked: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|c| {
                let node_id = self.node_id(c.idx as usize);
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
        let _op = self.begin_operation();

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
            let candidates =
                self.results_only(self.search_layer_ctx(&qctx, current_ep, current_ef, 0));

            // Batch-load f32 for reranking
            let candidate_ids: Vec<u64> = candidates
                .iter()
                .map(|c| self.node_id(c.idx as usize))
                .collect();
            let loaded = loader.load_vectors(&candidate_ids, &self.config.property_name);

            let mut results: Vec<SearchResult> = candidates
                .into_iter()
                .filter_map(|c| {
                    let node_id = self.node_id(c.idx as usize);
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
        // Xorshift64 RNG feeding the exponential level distribution.
        let mut state = self.rng_state.load(std::sync::atomic::Ordering::Relaxed);
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.rng_state
            .store(state, std::sync::atomic::Ordering::Relaxed);

        let uniform = (state >> 33) as f64 / (1u64 << 31) as f64;
        (-uniform.ln() * self.level_mult).floor() as usize
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
        // One epoch pin for the descent; the list reads below nest in it.
        let _epoch = crossbeam_epoch::pin();
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
                    if neighbor_idx < self.node_len() {
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
        // One epoch pin for the whole layer pass, handed down to every
        // layer-0 list read and prefetch, so a visit pays no pin of its own.
        let guard = crossbeam_epoch::pin();
        // Dispatch on the configured rerank policy. Inline (the legacy
        // default) keeps the two-heap two-distance design that preserves
        // glove-class recall. EndOfSearch / None drop the per-visit f32
        // call and reach much higher QPS at the cost of using a noisy
        // cheap threshold during traversal; see [`RerankMode`].
        match self.config.rerank_mode {
            RerankMode::Inline => self.search_layer_ctx_inline_rerank(ctx, ep, ef, level, &guard),
            RerankMode::EndOfSearch => {
                self.search_layer_ctx_end_of_search_rerank(ctx, ep, ef, level, &guard)
            }
            RerankMode::None => self.search_layer_ctx_no_rerank(ctx, ep, ef, level, &guard),
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
        guard: &crossbeam_epoch::Guard,
    ) -> Vec<Candidate> {
        let ep_dist = self.compute_distance(ctx, ep);

        let heap_cap = ef + 16;
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(heap_cap);
        let mut results: BinaryHeap<FarCandidate> = BinaryHeap::with_capacity(heap_cap);
        // One snapshot of the node count per search: nodes inserted after it
        // are skipped by this search, and the visited list covers every index
        // below it. Saves one atomic load per inner iteration.
        let n_nodes = self.node_len();
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
            // Every node takes part in layer 0, so only upper layers ask.
            if level > 0 && level >= self.node_levels(closest.idx as usize) {
                continue;
            }
            if level == 0 {
                self.read_layer0_neighbours_with(closest.idx as usize, &mut connections, guard);
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
        guard: &crossbeam_epoch::Guard,
    ) -> Vec<Candidate> {
        // Oversample: traverse the graph with a larger cheap-distance
        // frontier so the rerank pool sees more candidates. qdrant's
        // `oversampling` parameter equivalent — `factor = 1.0` (default)
        // collapses to the original "ef in, ef out" behaviour.
        let factor = self.config.rerank_oversample_factor.max(1.0);
        let frontier_ef = ((ef as f32) * factor).ceil() as usize;

        let mut result_vec = self.search_layer_ctx_no_rerank(ctx, ep, frontier_ef, level, guard);

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
        guard: &crossbeam_epoch::Guard,
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
        // With no quantizer active there IS no cheap metric: the
        // frontier score already falls through to the exact f32 path,
        // and the verify pass below would recompute the identical
        // value, doubling the distance work per visit. Measured on
        // glove-100-50k ST at ef=200: 4929 distance calls per query
        // (~2x hnswlib's volume), accounting for the whole 1.9x QPS
        // gap on unquantized cells.
        let cheap_is_exact = self.rabitq_params.is_none() && self.sq8_params.is_none();

        let cheap_ep = self.compute_distance(ctx, ep);
        let exact_ep = if cheap_is_exact {
            cheap_ep
        } else {
            self.compute_exact_distance(ctx, ep)
        };

        let heap_cap = ef + 16;
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(heap_cap);
        let mut results: BinaryHeap<FarCandidate> = BinaryHeap::with_capacity(heap_cap);
        // One snapshot of the node count per search: nodes inserted after it
        // are skipped by this search, and the visited list covers every index
        // below it. Saves one atomic load per inner iteration.
        let n_nodes = self.node_len();
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

            // Hint the NEXT frontier candidate's neighbour-id region
            // while the current one is being expanded: the pop on the
            // following iteration reads that region immediately, and
            // it is a random access the hardware prefetcher cannot
            // predict (hnswlib issues the same hint on its candidate
            // top inside `searchBaseLayerST`).
            if level == 0 {
                if let (Some(next), Some(block)) = (candidates.peek(), self.data_level0.get()) {
                    block.prefetch_neighbours(next.idx as usize, guard);
                }
            }

            // Every node takes part in layer 0, so only upper layers ask.
            if level == 0 || level < self.node_levels(closest.idx as usize) {
                unvisited_neighbors.clear();
                // Read this node's neighbour row: layer 0 from the contiguous
                // data_level0 block (co-located with the f32 vector, ids u32),
                // layers >= 1 from the SoA upper store. The hnswlib one-ahead
                // prefetch trick (`hnswalg.h:371-374`) is preserved by peeking
                // the next id in `connections`.
                {
                    if level == 0 {
                        self.read_layer0_neighbours_with(
                            closest.idx as usize,
                            &mut connections,
                            guard,
                        );
                    } else {
                        connections.clear();
                        self.neighbours_at(closest.idx as usize, level)
                            .snapshot_into(&mut connections);
                    }
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
                        // Unless no quantizer is active: then the frontier
                        // score IS the exact f32 distance already.
                        let exact_dist = if cheap_is_exact {
                            cheap_dist
                        } else {
                            self.compute_exact_distance(ctx, neighbor_idx)
                        };

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
            // Code-block fast path: read packed code + scalars from the
            // per-node block instead of dereferencing the SoA
            // per-node code in the store. Gated on byte-length match between
            // the slot and the query bit-planes so dims whose effective code
            // width differs from `dim/8` (e.g. dim=100 padded to 128) cleanly
            // fall through to the SoA path.
            if let (Some(params), Some(RabitqQuery::OneBit(q)), Some(inline)) = (
                self.rabitq_params.as_ref(),
                ctx.rabitq_query.as_ref(),
                self.rabitq_block.get(),
            ) {
                let slot_words = inline.rabitq_byte_len() / 8;
                if inline.rabitq_bits() == 1
                    && inline.rabitq_byte_len().is_multiple_of(8)
                    && slot_words == q.planes[0].len()
                    && node_idx < inline.capacity()
                {
                    // SAFETY: node_idx < capacity; the code starts each
                    // per-node block, whose stride is a multiple of 8 over a
                    // u64 backing, so it is 8-aligned, and we verified the
                    // byte length is a multiple of 8 above. The slice borrow
                    // lives for the duration of the call.
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
            // Nested, not a tuple pattern: a tuple would read the node's
            // RaBitQ code (random 72B-stride read, a guaranteed cache miss
            // per visit) even when the params or query are None, i.e. on
            // every unquantized search.
            if let (Some(params), Some(qenc)) =
                (self.rabitq_params.as_ref(), ctx.rabitq_query.as_ref())
            {
                if let Some(xcode) = self.node_rabitq(node_idx) {
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
        }
        if let Some(params) = &self.sq8_params {
            if let Some(quantized) = self.node_sq8(node_idx) {
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
                // Nested, not a tuple pattern: a tuple would read the node's
                // RaBitQ code (random 72B-stride read, a cache miss per
                // visit) even on unquantized indexes where the params are
                // None.
                if let Some(params) = self.rabitq_params.as_ref() {
                    if let Some(enc) = self.node_rabitq(node_idx) {
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
                // Cached inverse norm available even without RaBitQ — skip
                // the per-visit `norm_l2(b)` pass AND the divide: the score
                // is `1 - dot * inv_a * inv_b`. Zero-vector inverses are
                // stored as 0.0, which collapses the product to 0.0 — the
                // same answer the division helpers' epsilon guard gives.
                if let Some(b_inv) = self
                    .data_level0
                    .get()
                    .and_then(|block| block.inv_norm(node_idx))
                {
                    if b_inv.is_finite() {
                        let dot = metrics::dot_product(ctx.vec, node_vec);
                        return 1.0 - dot * ctx.inv_norm_l2 * b_inv;
                    }
                }
            }
            return self.distance_for_metric(ctx, node_vec);
        }
        // f32 offloaded — fall back to dequantized SQ8 (slightly less accurate)
        if let Some(ref params) = self.sq8_params {
            if let Some(quantized) = self.node_sq8(node_idx) {
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
        // Recomputed against the list that won when a concurrent writer
        // replaced it between the read and the publish.
        self.layer_update(node_idx, level, |neighbours| {
            let kept = self.prune_selection(node_idx, neighbours, max_conn, extras);
            (kept != neighbours).then_some(kept)
        });
    }

    /// The pruned list for `node_idx` from its current `neighbours` and the
    /// queued `extras`: the RobustPrune selection, backfilled by distance to
    /// `max_conn`.
    fn prune_selection(
        &self,
        node_idx: usize,
        neighbours: &[u64],
        max_conn: usize,
        extras: &[u64],
    ) -> Vec<u64> {
        // Stored u64s ARE neighbour indices now (not NodeIds) — no
        // id_to_idx hop. Dedupe via HashSet on those indices.
        let mut seen: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(neighbours.len() + extras.len());
        let mut candidates: Vec<Candidate> = Vec::with_capacity(neighbours.len() + extras.len());
        for &neighbor_idx_u64 in neighbours.iter().chain(extras.iter()) {
            if !seen.insert(neighbor_idx_u64) {
                continue;
            }
            let neighbor_idx = neighbor_idx_u64 as usize;
            if neighbor_idx < self.node_len() {
                let dist = self.distance_between_nodes(node_idx, neighbor_idx);
                candidates.push(Candidate {
                    distance: dist,
                    idx: neighbor_idx_u64 as u32,
                });
            }
        }

        // Prune back-edges with the SAME diversity heuristic used for new-node
        // selection (RobustPrune / hnswlib `getNeighborsByHeuristic2`), not a
        // naive closest-`max_conn` truncation. Truncation drops the long-range /
        // diverse edges that keep the graph navigable, which measurably lowers
        // recall-per-ef; the heuristic keeps them. Construction-time only, so the
        // extra pairwise-distance cost is off the search hot path.
        let mut kept: Vec<u64> = self
            .select_neighbours_robust_prune(&candidates, max_conn)
            .into_iter()
            .map(|idx| idx as u64)
            .collect();
        // hnswlib `keepPrunedConnections`: if the diversity heuristic kept fewer
        // than `max_conn`, backfill with the closest not-yet-kept candidates so
        // the back-edge list stays dense. A sparse list (heuristic alone) loses
        // recall, especially on low-dim / small graphs; this keeps the diverse
        // long-range edges first, then fills the remaining slots by distance.
        if kept.len() < max_conn {
            let kept_set: std::collections::HashSet<u64> = kept.iter().copied().collect();
            let mut rest: Vec<&Candidate> = candidates
                .iter()
                .filter(|c| !kept_set.contains(&u64::from(c.idx)))
                .collect();
            rest.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for c in rest {
                if kept.len() >= max_conn {
                    break;
                }
                kept.push(u64::from(c.idx));
            }
        }
        kept
    }

    // ── Atomic neighbour write helpers ─────────────────────────────────────
    //
    // Single source of truth: `data_level0` for layer 0, `neighbours_upper`
    // for the layers above. Every list is published whole, and every write
    // to a list another writer can reach goes through compare-and-swap:
    //
    // * `cas_add_neighbour_to` appends, retrying against the list that won.
    // * `layer_update` (prune, full-list insert, removal) recomputes its
    //   edit against the list that won, so it never overwrites an edge
    //   another writer added after it read the list.
    // * `set_outgoing` replaces a list blindly; it is for a node nobody else
    //   can reach yet (the new node's own outgoing edges) or a caller that
    //   holds `&mut self`.

    /// Number of nodes in the index.
    #[inline(always)]
    fn node_len(&self) -> usize {
        self.node_count.load(core::sync::atomic::Ordering::Acquire)
    }

    /// The node store. Present from the first insert on, so every `idx <
    /// node_len()` finds it.
    #[inline(always)]
    #[allow(
        clippy::expect_used,
        reason = "only called for an existing node; a node exists only in the store"
    )]
    fn nodes(&self) -> &data_level0::DataLevel0Block {
        self.data_level0
            .get()
            .expect("a node exists only after the store was created")
    }

    /// Node `idx`'s SQ8 code, `None` before calibration.
    #[inline(always)]
    fn node_sq8(&self, idx: usize) -> Option<&[u8]> {
        self.data_level0.get()?.sq8(idx)
    }

    /// Node `idx`'s RaBitQ code, `None` before calibration.
    #[inline(always)]
    fn node_rabitq(&self, idx: usize) -> Option<&RabitqEncoded> {
        self.data_level0.get()?.rabitq(idx)
    }

    /// The node store for a calibration that rewrites every node's codes.
    fn nodes_mut(&mut self) -> Option<&mut data_level0::DataLevel0Block> {
        self.data_level0.get_mut()
    }

    /// `candidates` without the retired and free nodes: a node removed or
    /// replaced by a newer one of the same id still guides the descent but is
    /// never a result.
    fn results_only(&self, mut candidates: Vec<Candidate>) -> Vec<Candidate> {
        if let Some(store) = self.data_level0.get() {
            candidates.retain(|c| store.state(c.idx as usize).is_linkable());
        }
        candidates
    }

    /// External id of node `idx`.
    #[inline]
    fn node_id(&self, idx: usize) -> u64 {
        debug_assert!(idx < self.node_len(), "node {idx} does not exist");
        // SAFETY: idx < node_len, and every counted node was initialized.
        unsafe { self.nodes().id(idx) }
    }

    /// Resolve `(node, layer >= 1)` to its [`AtomicNeighbourList`]. Layer 0
    /// lists are compact and go through the `layer_*` helpers below; calling
    /// this with `level == 0` is a bug.
    #[inline]
    fn neighbours_at(&self, idx: usize, level: usize) -> &AtomicNeighbourList<M_MAX0> {
        debug_assert!(
            level >= 1 && level < self.node_levels(idx),
            "node {idx} has no list at layer {level}"
        );
        // SAFETY: idx is an existing node and 1 <= level < its layer count.
        unsafe { self.nodes().upper(idx, level) }
    }

    /// Snapshot node `idx`'s neighbours at `level` into `out` (cleared first).
    /// Layer 0 lists widen their u32 ids to u64.
    fn layer_snapshot_into(&self, idx: usize, level: usize, out: &mut Vec<u64>) {
        if level == 0 {
            self.read_layer0_neighbours_into(idx, out);
        } else {
            out.clear();
            self.neighbours_at(idx, level).snapshot_into(out);
        }
    }

    /// Allocating variant of [`Self::layer_snapshot_into`].
    fn layer_snapshot(&self, idx: usize, level: usize) -> Vec<u64> {
        let mut out = Vec::new();
        self.layer_snapshot_into(idx, level, &mut out);
        out
    }

    /// Number of neighbours published at `(idx, level)`. Layer 0 from
    /// `data_level0`'s atomic count.
    fn layer_len(&self, idx: usize, level: usize) -> usize {
        if level == 0 {
            self.data_level0.get().map_or(0, |block| {
                if block.contains(idx) {
                    // SAFETY: idx < capacity per the gate.
                    unsafe { block.neighbour_count(idx) as usize }
                } else {
                    0
                }
            })
        } else {
            self.neighbours_at(idx, level).len()
        }
    }

    /// Concurrent append of `id` at `(idx, level)`; `false` if at capacity.
    /// Layer 0 appends to `data_level0` (atomic CAS, multi-writer-safe).
    fn layer_cas_append(&self, idx: usize, level: usize, id: u64) -> bool {
        if level == 0 {
            self.data_level0.get().is_some_and(|block| {
                if block.contains(idx) {
                    // SAFETY: idx < capacity per the gate.
                    unsafe { block.cas_append_neighbour(idx, id as u32, &self.stats) }
                } else {
                    false
                }
            })
        } else {
            self.neighbours_at(idx, level)
                .cas_append_up_to_with(id, M_MAX0, Some(&self.stats))
        }
    }

    /// Publish `edit` of the neighbour list at `(idx, level)` under
    /// concurrent writers. `edit` receives the current ids and returns the
    /// complete new list, or `None` to keep it; when another writer replaced
    /// the list in between, `edit` runs again against the list that won, so a
    /// concurrently accepted edge is never overwritten by a result computed
    /// from an older list. The result is truncated to the layer's capacity.
    /// Returns whether a new list was published.
    fn layer_update(
        &self,
        idx: usize,
        level: usize,
        mut edit: impl FnMut(&[u64]) -> Option<Vec<u64>>,
    ) -> bool {
        if level > 0 {
            return self.neighbours_at(idx, level).update_with(
                |current| {
                    edit(current).map(|mut next| {
                        next.truncate(M_MAX0);
                        next.into_boxed_slice()
                    })
                },
                Some(&self.stats),
            );
        }
        let Some(block) = self.data_level0.get() else {
            return false;
        };
        if !block.contains(idx) {
            return false;
        }
        let cap = block.m_max0();
        let mut wide: Vec<u64> = Vec::with_capacity(cap + 1);
        // SAFETY: idx < capacity per the gate above.
        unsafe {
            block.update_neighbours(
                idx,
                |current| {
                    wide.clear();
                    wide.extend(current.iter().map(|&id| u64::from(id)));
                    // Graph indices fit u32: per-shard node count is well
                    // below `u32::MAX`, the same narrowing every layer-0
                    // write makes.
                    edit(&wide).map(|next| next.iter().take(cap).map(|&id| id as u32).collect())
                },
                &self.stats,
            )
        }
    }

    /// Number of layers the node participates in (`top_level + 1`).
    #[inline]
    fn node_levels(&self, idx: usize) -> usize {
        debug_assert!(idx < self.node_len(), "node {idx} does not exist");
        // SAFETY: idx < node_len, and every counted node was initialized.
        unsafe { self.nodes().levels(idx) }
    }

    /// Append `id` to `(neighbour_idx, level)`. If the resulting list
    /// exceeds `max_conn`, run `prune_connections` to shrink back to the
    /// nearest `max_conn` neighbours. Every step publishes by CAS, so it is
    /// safe under concurrent writers to the same list.
    fn add_neighbour_to(&self, neighbour_idx: usize, level: usize, id: u64, max_conn: usize) {
        if self.layer_cas_append(neighbour_idx, level, id) {
            let len_now = self.layer_len(neighbour_idx, level);
            if len_now > max_conn {
                self.prune_connections(neighbour_idx, level, max_conn);
            }
            // Otherwise the append already landed in the sole neighbour store
            // for this layer; there is nothing else to mirror.
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
            // has ≤ `max_conn` elements (`≤ M_MAX0`) so the published list
            // fits without truncation. Recomputed against the winning list
            // if another writer replaced it meanwhile.
            self.layer_update(neighbour_idx, level, |current| {
                if current.contains(&id) {
                    return None;
                }
                // Stored u64s ARE neighbour indices: direct cast, no map hop.
                let mut scored: Vec<(f32, u64)> = Vec::with_capacity(current.len() + 1);
                for &nidx_u64 in current.iter().chain(core::iter::once(&id)) {
                    let nidx = nidx_u64 as usize;
                    if nidx < self.node_len() {
                        let dist = self.distance_between_nodes(neighbour_idx, nidx);
                        scored.push((dist, nidx_u64));
                    }
                }
                scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(max_conn);
                Some(scored.into_iter().map(|(_, nid)| nid).collect())
            });
        }
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
        let alpha = self.config.effective_alpha();

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
        if let (Some(va), Some(vb)) = (self.read_node_f32(a_idx), self.read_node_f32(b_idx)) {
            // Cosine fast path: both norms are precomputed per node, so
            // skip QueryCtx entirely. The generic path recomputes ||a||
            // on every call (QueryCtx::new), and this function runs
            // O(max_conn x candidates) times per insert; on glove-100
            // angular the redundant norm passes dominated build cycles.
            if matches!(self.config.metric, VectorMetric::Cosine) {
                let norms = self
                    .data_level0
                    .get()
                    .map(|block| (block.norm(a_idx), block.norm(b_idx)));
                if let Some((Some(na), Some(nb))) = norms {
                    if na.is_finite() && na > 0.0 && nb.is_finite() && nb > 0.0 {
                        return 1.0 - metrics::cosine_similarity_with_both_norms(va, vb, na, nb);
                    }
                }
            }
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

    /// Get a node's vector: f32 if available, otherwise dequantize from SQ8.
    fn get_node_vector_or_dequantized(&self, idx: usize, params: &Sq8Params) -> Vec<f32> {
        if let Some(v) = self.read_node_f32(idx) {
            return v.to_vec();
        }
        if let Some(q) = self.node_sq8(idx) {
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
        // prefetch — the node's RaBitQ code (Option<RabitqEncoded>), its
        // f32 vector (Option<Vec<f32>>) and its SQ8 code
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
        // Gate on the INDEX-level codec state before touching the
        // per-node code: it is a random read with a 72B stride, and on
        // unquantized indexes (codes all None) it was a guaranteed cache
        // miss per neighbour visit spent only to DECIDE what to prefetch —
        // perf measured it at ~32% of search_layer self-time on glove-50k
        // ST. RaBitQ codes exist only after calibration, which requires
        // `rabitq_params`.
        if self.rabitq_params.is_some() && matches!(self.config.metric, VectorMetric::Cosine) {
            if let Some(enc) = self.node_rabitq(idx) {
                let code_words = match enc {
                    RabitqEncoded::OneBit(c) => c.code.as_ptr() as *const u8,
                    RabitqEncoded::Multi(c) => c.packed.as_ptr(),
                };
                prefetch_read_data(code_words);
                return;
            }
        }
        // Prefer the contiguous f32-only block when present: prefetch
        // target address is `base + idx * stride` — one ALU op, zero
        // SoA cache misses. Matches hnswlib's `_mm_prefetch(data_level0_memory_
        // + idx * size_data_per_element_)` shape.
        if let Some(block) = self.data_level0.get() {
            if block.has_f32() {
                block.prefetch(idx);
                return;
            }
        }
        if let Some(q) = self.node_sq8(idx) {
            prefetch_read_data(q.as_ptr());
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
