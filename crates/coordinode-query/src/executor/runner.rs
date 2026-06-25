//! Physical query executor: runs logical plan operators against storage.
//!
//! Each operator produces a `Vec<Row>` from its input.
//! Future optimization: streaming iterator model.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use coordinode_core::graph::edge::{decode_edge_props, AdjDirection, AdjKeyParts, PostingList};
use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::NodeIdAllocator;
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::{Value, VectorConsistencyMode, VectorMvccStats};
use coordinode_core::schema::definition::{
    EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType, SchemaMode,
};
use coordinode_core::schema::validation::validate_one;
use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::core::StorageEngine;
// ADR-041 storage-partition guard: the only legitimate Partition users in this
// crate are the partition-parameterised transaction primitives below (which
// take it by argument) and test fixtures. Production execution goes through the
// typed Layer-4 stores. See the crate-level `#![deny(clippy::disallowed_types)]`.
#[allow(clippy::disallowed_types)]
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_storage::engine::StorageSnapshot;

use super::eval::{eval_binary_op, eval_expr, eval_unary_op, is_truthy};
use super::row::Row;
use crate::cypher::ast::{
    BinaryOperator, Direction, Expr, LengthBound, NodePattern, Pattern, PatternElement,
    RelationshipPattern, ViolationMode,
};
use crate::index::{IndexState, OnlineDuringBuild};
use crate::planner::logical::*;

/// Default maximum hops for unbounded variable-length paths.
/// Prevents exponential fan-out on `*` or `*1..` patterns.
const DEFAULT_MAX_HOPS: u64 = 10;

/// Key-value pair returned by MVCC prefix scan: (user_key, value).
type KvPair = (Vec<u8>, Vec<u8>);

/// Execution error.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("storage error: {0}")]
    Storage(#[from] coordinode_storage::error::StorageError),

    /// Modality-store error from `coordinode-modality`. Wraps the typed
    /// store error, which itself preserves the underlying `StorageError`
    /// chain — capacity-exhausted, checksum-mismatch, and other engine
    /// errors propagate end-to-end.
    #[error("modality store error: {0}")]
    Modality(#[from] coordinode_modality::StoreError),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("write conflict: {0}")]
    Conflict(String),

    /// Schema mode violation: STRICT label rejected an undeclared property, or
    /// a write attempted to SET a COMPUTED (read-only) property.
    #[error("schema violation: {0}")]
    SchemaViolation(String),

    /// L1 cycle protection trip (the trigger architecture): cumulative trigger cascade depth
    /// for the current originating mutation exceeded its limit. `chain` lists
    /// the trigger names that fired, in firing order, to help diagnose the
    /// runaway cascade.
    #[error("trigger cascade depth exceeded: current={current}, limit={limit}, chain={chain:?}")]
    CascadeOverflow {
        current: u32,
        limit: u32,
        chain: Vec<String>,
    },

    /// L2 cycle protection trip (the trigger architecture): a single trigger fired more times
    /// than its `CASCADE_FANOUT` allows within one cascade root. Wide-but-
    /// shallow runaways (one trigger re-firing per row of a batch) trip this
    /// well before L1.
    #[error("trigger cascade fanout exceeded for `{trigger}`: count={count}, limit={limit}")]
    CascadeFanoutOverflow {
        trigger: String,
        count: u32,
        limit: u32,
    },
}

/// Configuration for adaptive query plan behavior.
///
/// When traversal fan-out exceeds expectations, the executor switches from
/// sequential to parallel processing (rayon work-stealing). At runtime, the
/// executor checks divergence every `check_interval` edges processed — if
/// `actual_fan_out > estimated × switch_threshold`, it switches strategy.
///
/// **Parallel mode:** When a posting list exceeds `parallel_threshold` edges,
/// target node processing is parallelized via rayon `par_chunks`. This
/// processes ALL edges without truncation, unlike the previous cap-only
/// approach.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Enable adaptive fan-out detection and parallel switching.
    pub enabled: bool,
    /// Maximum edges to process per source node in sequential mode.
    /// When exceeded AND parallel is enabled, switches to parallel processing.
    /// When parallel is disabled, this acts as a hard cap (truncation).
    /// Default: 10_000.
    pub max_fan_out: usize,
    /// Factor: if actual_fan_out > estimated × threshold → switch strategy.
    /// Default: 10.0.
    pub switch_threshold: f64,
    /// Check divergence every N edges processed during variable-length traversal.
    /// Default: 1000.
    pub check_interval: usize,
    /// Minimum edges to trigger parallel processing via rayon.
    /// Below this threshold, sequential processing is faster (avoids rayon overhead).
    /// Default: 1000.
    pub parallel_threshold: usize,
    /// Chunk size for rayon parallel iteration.
    /// Each chunk processes this many target nodes before synchronizing.
    /// Default: 256.
    pub parallel_chunk_size: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_fan_out: 10_000,
            switch_threshold: 10.0,
            check_interval: 1000,
            parallel_threshold: 1000,
            parallel_chunk_size: 256,
        }
    }
}

/// Cache of known super-node fan-out degrees.
///
/// When a node's fan-out exceeds `parallel_threshold`, its degree is stored here.
/// On subsequent queries, the executor checks this cache first — if the node
/// is known to be a super-node, it skips the sequential attempt and goes
/// directly to parallel processing.
///
/// Thread-safe via `Mutex` for shared access across queries.
/// Bounded to `max_entries` to prevent unbounded growth.
#[derive(Debug, Clone)]
pub struct FeedbackCache {
    /// Map of node_id → known fan-out degree.
    inner: std::sync::Arc<Mutex<HashMap<u64, usize>>>,
    /// Maximum entries before eviction (FIFO via insert order — approximate).
    max_entries: usize,
}

impl FeedbackCache {
    /// Create a new feedback cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            inner: std::sync::Arc::new(Mutex::new(HashMap::with_capacity(max_entries.min(1024)))),
            max_entries,
        }
    }

    /// Record a super-node's fan-out degree.
    pub fn record(&self, node_id: u64, fan_out: usize) {
        if let Ok(mut cache) = self.inner.lock() {
            if cache.len() >= self.max_entries {
                // Approximate FIFO: clear half the cache when full
                let to_remove: Vec<u64> =
                    cache.keys().take(self.max_entries / 2).copied().collect();
                for key in to_remove {
                    cache.remove(&key);
                }
            }
            cache.insert(node_id, fan_out);
        }
    }

    /// Check if a node is a known super-node. Returns its fan-out if known.
    pub fn lookup(&self, node_id: u64) -> Option<usize> {
        self.inner.lock().ok()?.get(&node_id).copied()
    }
}

impl Default for FeedbackCache {
    fn default() -> Self {
        Self::new(100_000) // matches arch doc: feedback_cache_size: 100000
    }
}

/// Write statistics tracked during statement execution.
#[derive(Debug, Clone, Default)]
pub struct WriteStats {
    pub nodes_created: u64,
    pub nodes_deleted: u64,
    pub edges_created: u64,
    pub edges_deleted: u64,
    pub properties_set: u64,
    pub properties_removed: u64,
    pub labels_added: u64,
    pub labels_removed: u64,
    /// Raft committed log index of this statement's write, when it was
    /// replicated through a `RaftProposalPipeline`. `None` for read-only
    /// statements and for non-replicated (local / embedded) writes. The
    /// gRPC layer surfaces it as the causal `operationTime` token so a
    /// client's follow-up causal read fences on *its own* write index
    /// rather than the node's current applied index.
    pub applied_index: Option<u64>,
}

impl WriteStats {
    /// Returns `true` if any mutation was recorded during execution.
    pub fn has_mutations(&self) -> bool {
        self.nodes_created > 0
            || self.nodes_deleted > 0
            || self.edges_created > 0
            || self.edges_deleted > 0
            || self.properties_set > 0
            || self.properties_removed > 0
            || self.labels_added > 0
            || self.labels_removed > 0
    }
}

/// Context for query execution: provides access to storage and metadata.
///
/// MVCC-aware: reads use snapshot isolation at `mvcc_read_ts`, writes are
/// buffered in `mvcc_write_buffer` and flushed atomically at commit with
/// a `commit_ts` assigned from the oracle.
///
/// When `mvcc_oracle` is `None`, the executor operates in legacy mode
/// (direct engine reads/writes without MVCC versioning) for backward
/// compatibility with existing tests.
/// A registered handler for an extension operator ([`LogicalOp::Extension`]).
/// The CE engine carries an empty [`ExtensionRegistry`] by default; the EE
/// server registers handlers at startup so the executor can dispatch extension
/// ops without any CE crate depending on EE. The handler decodes the opaque
/// `payload` the parser-extension produced for its named op.
pub trait ExtensionHandler: Send + Sync {
    /// Execute the named extension op against the opaque `payload`.
    fn execute(
        &self,
        ctx: &mut ExecutionContext<'_>,
        payload: &[u8],
    ) -> Result<Vec<Row>, ExecutionError>;
}

/// Registry of [`ExtensionHandler`]s keyed by op name. CE default is empty (no
/// extensions registered, so [`LogicalOp::Extension`] never appears in a CE
/// plan); the EE server populates it once at startup and shares it by
/// reference into every [`ExecutionContext`]. Lookup is one hash probe on the
/// cold extension path; the query hot path never touches it.
#[derive(Default)]
pub struct ExtensionRegistry {
    handlers: std::collections::HashMap<String, std::sync::Arc<dyn ExtensionHandler>>,
}

impl ExtensionRegistry {
    /// Create an empty registry (the CE default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `handler` under `name`; a later registration with the same
    /// name replaces the earlier one.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        handler: std::sync::Arc<dyn ExtensionHandler>,
    ) {
        self.handlers.insert(name.into(), handler);
    }

    /// Look up (and clone the `Arc` of) the handler registered under `name`.
    pub fn get(&self, name: &str) -> Option<std::sync::Arc<dyn ExtensionHandler>> {
        self.handlers.get(name).cloned()
    }

    /// Whether no handlers are registered (the CE default state).
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }
}

pub struct ExecutionContext<'a> {
    pub engine: &'a StorageEngine,
    /// Optional `Arc` handle to the same engine the borrow above points at.
    ///
    /// Set by callers that already own an `Arc<StorageEngine>` (the embed /
    /// server stack) and want to enable execution paths that need owned
    /// engine handles for background work — first consumer is the HNSW
    /// backfill task spawned by `CREATE VECTOR INDEX`. Construction sites
    /// that build an ExecutionContext from a borrowed-only test engine
    /// leave this `None`, which forces the legacy synchronous backfill.
    pub engine_arc: Option<Arc<StorageEngine>>,
    pub interner: &'a mut FieldInterner,
    /// Node ID allocator for CREATE operations.
    pub id_allocator: &'a NodeIdAllocator,
    /// Default shard ID for single-node deployment.
    pub shard_id: u16,
    /// Adaptive query plan configuration.
    pub adaptive: AdaptiveConfig,
    /// When true, variable-length traversal emits each reached target node at
    /// most once instead of one row per reaching edge. Set by the planner only
    /// when the whole query provably cannot observe target multiplicity (a lone
    /// var-length traverse feeding `count(DISTINCT target)` with no path or edge
    /// variable), so it never changes a result. Collapses the emitted row count
    /// from O(edges) to O(reached nodes) on dense / near-global traversals.
    pub dedup_varlen_targets: bool,
    /// Feedback cache for known super-node fan-out degrees.
    /// Shared across queries within the same session/connection.
    pub feedback_cache: Option<FeedbackCache>,
    /// Snapshot timestamp for AS OF TIMESTAMP queries (microseconds since epoch).
    /// When set, reads return data as of this point in time.
    pub snapshot_ts: Option<i64>,
    /// MVCC retention window in microseconds (default: 7 days).
    pub retention_window_us: i64,
    /// Warnings collected during execution (e.g., fan-out capping).
    pub warnings: Vec<String>,
    /// Write statistics accumulated during this statement.
    pub write_stats: WriteStats,
    /// Optional full-text search index for text_match()/text_score() queries.
    /// Uses MultiLanguageTextIndex which wraps TextIndex with per-language support.
    /// 2-arg text_match(field, query) uses default language; 3-arg adds explicit language.
    /// When `text_index_registry` is set, this field is ignored in favor of
    /// registry-based lookup by (label, property).
    pub text_index: Option<&'a coordinode_search::tantivy::multi_lang::MultiLanguageTextIndex>,
    /// Text index registry for automatic full-text index management.
    /// When set, `execute_text_filter` resolves the text index by (label, property).
    /// Write operations auto-maintain text indexes via `on_text_written`/`on_text_deleted`.
    pub text_index_registry: Option<&'a crate::index::TextIndexRegistry>,
    /// Vector index registry for HNSW-accelerated vector search.
    /// When set, VectorFilter checks for applicable HNSW indexes
    /// before falling back to brute-force distance computation.
    pub vector_index_registry: Option<&'a crate::index::VectorIndexRegistry>,
    /// B-tree index registry for unique constraint enforcement.
    /// When set, `execute_create_node` calls `on_node_created` to check
    /// unique constraints and maintain B-tree index entries.
    pub btree_index_registry: Option<&'a crate::index::IndexRegistry>,
    /// Registry of extension-op handlers (EE-populated, CE default empty).
    /// Reached only by the [`LogicalOp::Extension`] dispatch arm; `None` /
    /// empty in pure-CE contexts means no extension ops are dispatchable.
    pub extensions: Option<&'a ExtensionRegistry>,
    /// Optional VectorLoader for disk-backed f32 reranking (G009).
    /// When HNSW indexes have `offload_vectors` enabled, this loader provides
    /// f32 vectors from storage for exact reranking of SQ8 candidates.
    pub vector_loader: Option<&'a dyn coordinode_vector::VectorLoader>,
    /// MVCC timestamp oracle. When set, enables MVCC-versioned reads/writes.
    pub mvcc_oracle: Option<&'a TimestampOracle>,
    /// R-SNAP2: per-shard `maxAssigned` watermark handle.
    ///
    /// Readers under `read_consistency = 'snapshot'` call
    /// `applied_watermark.wait_for(snapshot_ts, read_timeout)` before
    /// dispatching the read, so every modality on this shard observes the
    /// fully-applied state at `snapshot_ts`. `None` in legacy /
    /// single-writer test contexts — the executor then skips the wait and
    /// reads "current" state. Wired in by `R-SNAP1` at the planner
    /// auto-promotion site.
    pub applied_watermark:
        Option<std::sync::Arc<coordinode_core::txn::watermark::MaxAssignedWatermark>>,
    /// R-SNAP1: cross-modality read consistency mode for this statement.
    /// Set by the planner from `LogicalPlan::read_consistency` (hint or
    /// auto-promotion). Default `Current` preserves the single-modality
    /// fast path.
    pub read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode,
    /// R-SNAP1: timeout for `applied_watermark.wait_for(snapshot_ts, …)`
    /// under `Snapshot` / `Exact` consistency. Default 2s matches
    /// `arch/core/transactions.md § Cross-Modality Snapshot Protocol`.
    pub read_timeout: std::time::Duration,
    /// MVCC read timestamp (start_ts). Allocated from oracle at statement start.
    /// All reads see a consistent snapshot at this timestamp.
    pub mvcc_read_ts: Timestamp,
    /// Layer-3 transaction context (ADR-041): owns the MVCC read snapshot,
    /// the read-your-own-writes write buffer, and the OCC read-set. The
    /// executor routes primitive `(partition, key)` reads/writes through it
    /// (`mvcc_get` / `mvcc_put` / `mvcc_delete` / `mvcc_prefix_scan` delegate
    /// here); modality-specific RYOW (node merge deltas) and the commit path
    /// still live on the executor and drive the transaction via its
    /// `pub(crate)` surface until they migrate too.
    pub txn: Transaction<'a>,
    /// Procedure context for CALL statements. When set, enables
    /// `db.advisor.*` procedures in the executor.
    pub procedure_ctx: Option<crate::advisor::procedures::ProcedureContext>,

    /// Vector MVCC consistency mode. Controls how vector search interacts
    /// with snapshot isolation. Default: `Current` (no visibility filter).
    /// Set via `SET vector_consistency = 'snapshot'` or per-query hint.
    pub vector_consistency: VectorConsistencyMode,
    /// Overfetch factor for snapshot mode vector search (default 1.2).
    /// Higher values improve recall at the cost of more MVCC checks.
    pub vector_overfetch_factor: f64,
    /// Statistics from the last vector MVCC operation (for EXPLAIN output).
    pub vector_mvcc_stats: Option<VectorMvccStats>,
    /// Raft proposal pipeline for durable mutation application.
    ///
    /// When set, `mvcc_flush()` sends validated mutations through the
    /// pipeline instead of writing directly to MvccEngine. In single-node
    /// mode, the pipeline applies directly to CoordiNode storage. In cluster mode
    /// (distributed mode), it replicates via Raft before applying.
    ///
    /// When `None`, legacy direct-write behavior is used (backward
    /// compatibility with tests that don't set up a pipeline).
    pub proposal_pipeline: Option<&'a dyn ProposalPipeline>,
    /// Proposal ID generator. Shared across all transactions on this node.
    pub proposal_id_gen: Option<&'a ProposalIdGenerator>,
    /// Read concern level for this query. Controls snapshot selection:
    /// - Local: read from applied_index (current default behavior)
    /// - Majority: read from commit_index (durable, no rollback)
    /// - Linearizable: verify leadership + read (strongest)
    /// - Snapshot: pinned to explicit timestamp
    pub read_concern: coordinode_core::txn::read_concern::ReadConcernLevel,
    /// Write concern for mutations. Controls durability:
    /// - W0: fire-and-forget (direct local write, no pipeline)
    /// - Memory: RAM only (~1µs), drain to Raft in background
    /// - Cache: RAM + NVMe (~100µs), drain to Raft in background
    /// - W1: leader WAL fsync
    /// - Majority: Raft quorum acknowledgement (production default)
    /// - journal: force WAL fsync after commit
    /// - timeout_ms: proposal timeout (0 = no timeout)
    pub write_concern: coordinode_core::txn::write_concern::WriteConcern,
    /// Volatile write drain buffer for w:memory and w:cache writes.
    /// When set, volatile writes are buffered here for background
    /// Raft replication instead of going through the synchronous
    /// proposal pipeline.
    pub drain_buffer: Option<&'a coordinode_core::txn::drain::DrainBuffer>,
    /// NVMe-backed write buffer for `w:cache` crash recovery.
    ///
    /// When set and write concern is `Cache`, mutations are persisted to this
    /// NVMe file before ACK so they survive process crashes. The drain thread's
    /// checkpoint protocol (`begin_drain` / `complete_drain`) ensures entries
    /// are cleaned up after successful Raft commit.
    pub nvme_write_buffer: Option<&'a coordinode_storage::cache::write_buffer::NvmeWriteBuffer>,
    /// Pending merge-add UIDs per adj key (raw key, no MVCC timestamp).
    ///
    /// MVCC snapshot for point-in-time reads (ADR-016: native seqno MVCC).
    ///
    /// When MVCC is enabled, this snapshot is created at `mvcc_read_ts` via
    /// `engine.snapshot_at(seqno)`. All reads (mvcc_get, mvcc_prefix_scan)
    /// go through this snapshot for O(1) lookups instead of key-suffix prefix
    /// scanning. Also used for adj: partition reads (replaces adj_snapshot).
    ///
    /// When None (legacy mode), reads go directly through engine.get().
    pub mvcc_snapshot: Option<StorageSnapshot>,
    /// L1 cascade depth counter (the trigger architecture). Shared across all triggers in
    /// one originating user mutation. Incremented before each trigger body
    /// is executed, decremented (RAII-style) when the body returns. When
    /// `cascade_depth > cascade_depth_limit` the firing is rejected with
    /// `ExecutionError::CascadeOverflow`.
    ///
    /// Wired by the trigger architecture probe; enforced by future trigger executors executors.
    pub cascade_depth: u32,
    /// Cluster-default cap on `cascade_depth`. Per-trigger `CASCADE_LIMIT n`
    /// in the trigger definition tightens this further when present.
    /// Source: cluster setting `triggers.max_cascade_depth` (default 10).
    pub cascade_depth_limit: u32,
    /// L2 unique-trigger fanout map (the trigger architecture). Keyed by trigger name; value
    /// is the number of times that trigger has fired within the current
    /// cascade root. When `cascade_fire_counts[name] > cascade_fanout_limit`
    /// the firing is rejected with `ExecutionError::CascadeFanoutOverflow`.
    ///
    /// Wired by the trigger architecture probe; enforced by future trigger executors executors.
    pub cascade_fire_counts: HashMap<String, u32>,
    /// Cluster-default cap on per-trigger fanout. Per-trigger `CASCADE_FANOUT n`
    /// tightens this further when present. Source: cluster setting
    /// `triggers.max_cascade_fanout` (default 100).
    pub cascade_fanout_limit: u32,
    /// Ordered chain of trigger names that have fired within the current
    /// cascade — used to populate the `cascade_chain` diagnostic field in
    /// dead-letter records when L1/L2 trip.
    pub cascade_chain: Vec<String>,
    /// Outer-scope row for correlated OPTIONAL MATCH execution.
    ///
    /// When set, the Filter operator merges these variables into each row
    /// before predicate evaluation. This allows correlated patterns like
    /// `OPTIONAL MATCH (b)-[:R]->(c) WHERE c.x = a.y` where `a` comes
    /// from the outer MATCH scope.
    pub correlated_row: Option<Row>,
    /// Scope injected at the `Empty` leaf by `FOREACH` and `CALL { subquery }`.
    /// When set, the `Empty` leaf yields this row (instead of a bare empty row)
    /// so a FOREACH body sees the loop variable, and a correlated CALL body's
    /// leading `WITH` can project imported outer variables. `None` everywhere
    /// else, so ordinary plans are unaffected.
    pub foreach_scope: Option<Row>,
    /// Per-statement cache: NodeId → primary label string.
    ///
    /// Schema checks in PropertyPath and DocFunction SET items read the node's
    /// primary label via `schema_peek_node`. For a statement like
    /// `SET n.a.x=1, n.a.y=2, n.a.z=3` targeting 100 nodes, each SET-item
    /// calls `schema_peek_node` per node → N×M engine reads without this cache.
    ///
    /// With the cache: first access reads + deserializes NodeRecord once per
    /// node per statement; subsequent SET-items on the same node hit the cache
    /// (O(1) HashMap lookup, no engine read, no deserialization).
    ///
    /// Primary labels are immutable within a transaction (no Cypher clause
    /// changes the primary label after node creation), so the cache is always
    /// valid for the lifetime of the statement. No invalidation required.
    pub schema_label_cache: HashMap<NodeId, String>,
    /// Named query parameters bound to this statement.
    ///
    /// Populated before `execute()` is called. Accessible inside aggregate
    /// functions such as `percentileCont(x, $p)` where `$p` is a bound parameter.
    /// Keys omit the `$` prefix (e.g., `"p"` for `$p`).
    pub params: HashMap<String, coordinode_core::graph::types::Value>,
    /// Buffered HNSW inserts produced by CREATE operators in this
    /// statement. Each entry is `(label, property, NodeId, vector)`.
    /// Drained at the end of [`execute()`] via
    /// [`Self::flush_pending_vector_writes`] which groups by
    /// `(label, property)` and calls
    /// `VectorIndexRegistry::on_vectors_written` once per group — so
    /// a bulk INSERT (UNWIND … CREATE …) pays one HNSW write-lock
    /// acquisition per index per batch instead of one per row.
    ///
    /// The non-batched single-row CREATE / SET / MERGE paths still
    /// dispatch directly to `on_vector_written` for backward
    /// compatibility; this buffer is only used by the CREATE-from-
    /// CreateNode operator hot path (line ≈ 7290) which dominates
    /// bulk loads.
    pub pending_vector_writes: Vec<(String, String, NodeId, Vec<f32>)>,
}

impl<'a> ExecutionContext<'a> {
    /// Drain buffered HNSW inserts and apply them in one batched
    /// write per (label, property) index. The CREATE-row hot path
    /// inside [`execute_create_node`] appends to
    /// `pending_vector_writes` instead of taking the HNSW write-lock
    /// per insert; the caller (`execute_cypher_impl`) calls this
    /// once after [`execute()`] returns to flush the batch.
    ///
    /// Safe to call repeatedly: a second call sees an empty buffer
    /// and is a no-op.
    pub fn flush_pending_vector_writes(&mut self) {
        if self.pending_vector_writes.is_empty() {
            return;
        }
        let Some(registry) = self.vector_index_registry else {
            self.pending_vector_writes.clear();
            return;
        };
        let drained = std::mem::take(&mut self.pending_vector_writes);
        // Group by (label, property) so each HNSW index gets one
        // write-lock + one insert_batch call.
        type VectorBatchKey = (String, String);
        type VectorBatchItems = Vec<(NodeId, Vec<f32>)>;
        let mut grouped: HashMap<VectorBatchKey, VectorBatchItems> = HashMap::new();
        for (label, property, node_id, vector) in drained {
            grouped
                .entry((label, property))
                .or_default()
                .push((node_id, vector));
        }
        for ((label, property), items) in grouped {
            registry.on_vectors_written(&label, &property, items);
        }
    }

    /// L1+L2 cascade entry (the trigger architecture). Call before executing a trigger body.
    /// Increments depth + per-trigger fire count, appends to chain, and trips
    /// `CascadeOverflow` / `CascadeFanoutOverflow` when limits are exceeded.
    ///
    /// `per_trigger_depth_limit` / `per_trigger_fanout_limit` are the per-trigger
    /// `CASCADE_LIMIT` / `CASCADE_FANOUT` overrides parsed from DDL; `None` means
    /// "use the context-wide cluster default". The effective limit is the MIN of
    /// the cluster default and the per-trigger override (tighter wins).
    ///
    /// **Caller must pair every successful `cascade_enter` with one
    /// `cascade_exit`, in LIFO order matching firings, on both success and
    /// error paths of the trigger body.** RAII via a guard returning `&'g mut
    /// Self` would lock the context for the entire body execution, but the
    /// body itself needs `&mut ctx` for nested mutations — so the explicit
    /// enter/exit pairing keeps the borrow window tight.
    ///
    /// The L2 fanout counter is intentionally NOT decremented by
    /// `cascade_exit` — it tracks total firings of a given trigger across the
    /// entire cascade root, which is what L2 wants to bound.
    pub fn cascade_enter(
        &mut self,
        trigger_name: &str,
        per_trigger_depth_limit: Option<u32>,
        per_trigger_fanout_limit: Option<u32>,
    ) -> Result<(), ExecutionError> {
        let depth_limit = match per_trigger_depth_limit {
            Some(v) => v.min(self.cascade_depth_limit),
            None => self.cascade_depth_limit,
        };
        let fanout_limit = match per_trigger_fanout_limit {
            Some(v) => v.min(self.cascade_fanout_limit),
            None => self.cascade_fanout_limit,
        };

        // L1: cumulative depth across all triggers.
        let next_depth = self.cascade_depth.saturating_add(1);
        if next_depth > depth_limit {
            let mut chain = self.cascade_chain.clone();
            chain.push(trigger_name.to_string());
            return Err(ExecutionError::CascadeOverflow {
                current: next_depth,
                limit: depth_limit,
                chain,
            });
        }

        // L2: per-trigger fanout in this cascade root.
        let entry = self
            .cascade_fire_counts
            .entry(trigger_name.to_string())
            .or_insert(0);
        let next_count = (*entry).saturating_add(1);
        if next_count > fanout_limit {
            return Err(ExecutionError::CascadeFanoutOverflow {
                trigger: trigger_name.to_string(),
                count: next_count,
                limit: fanout_limit,
            });
        }
        *entry = next_count;

        self.cascade_depth = next_depth;
        self.cascade_chain.push(trigger_name.to_string());
        Ok(())
    }

    /// Paired with a successful `cascade_enter`. Decrements `cascade_depth`
    /// and pops the trailing chain entry. The L2 fanout counter is preserved
    /// — see `cascade_enter` doc.
    pub fn cascade_exit(&mut self) {
        self.cascade_depth = self.cascade_depth.saturating_sub(1);
        self.cascade_chain.pop();
    }

    /// Reset all L1/L2 cascade tracking. Called at the start of each user
    /// mutation root so trigger chains from a prior statement do not leak
    /// into the next one.
    pub fn cascade_reset(&mut self) {
        self.cascade_depth = 0;
        self.cascade_fire_counts.clear();
        self.cascade_chain.clear();
    }

    /// load all enabled triggers that match a single
    /// `(target_segment, event)` mutation. The lookup is O(matching_triggers)
    /// — it reads exactly one index key + N definition keys, never scans the
    /// trigger table. future trigger executors call this at trigger firing time; the trigger architecture
    /// itself only ships the helper + tests.
    ///
    /// `target_segment` must come from
    /// `TriggerTargetSchema::index_key_segment` (`n:Label` or `e:EdgeType`).
    /// `event_segment` is `"c"` / `"u"` / `"d"` (CREATE / UPDATE / DELETE).
    ///
    /// Disabled triggers (via `ALTER TRIGGER … DISABLE`) are filtered out
    /// before returning — the index entry persists across enable/disable so
    /// re-enabling does not have to re-index, but firing must skip disabled.
    pub fn lookup_matching_triggers(
        &mut self,
        target_segment: &str,
        event_segment: &str,
    ) -> Result<Vec<coordinode_core::schema::triggers::TriggerSchema>, ExecutionError> {
        use coordinode_core::schema::triggers::TriggerSchema;
        use coordinode_modality::{LocalTriggerStore, TriggerStore as _};
        self.sync_txn_state();
        let names: Vec<String> =
            match LocalTriggerStore.get_index(&mut self.txn, target_segment, event_segment)? {
                Some(bytes) => rmp_serde::from_slice(&bytes).map_err(|e| {
                    ExecutionError::Serialization(format!(
                        "trigger_index decode for {target_segment}/{event_segment}: {e}"
                    ))
                })?,
                None => return Ok(Vec::new()),
            };
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let Some(bytes) = LocalTriggerStore.get_definition(&mut self.txn, &name)? else {
                // Inconsistent state — index references a missing definition.
                // Skip rather than fail the mutation; the next DDL will
                // rebuild the index.
                continue;
            };
            let schema: TriggerSchema = rmp_serde::from_slice(&bytes).map_err(|e| {
                ExecutionError::Serialization(format!("trigger `{name}` decode: {e}"))
            })?;
            if schema.enabled {
                out.push(schema);
            }
        }
        Ok(out)
    }

    /// Read a Node key for schema checks without triggering RYOW merge-delta materialization.
    ///
    /// Schema checks in PropertyPath and DocFunction SET items need the node's primary
    /// label to look up the label schema, but must not consume pending merge deltas.
    /// Calling `mvcc_get` would trigger `materialize_node_deltas` for any key with
    /// in-flight deltas, breaking multi-item SET statements like
    /// `SET n.doc.a = 1, n.doc.b = 2` (the second item would not see delta from first).
    ///
    /// Read order: write buffer first, then committed engine state.
    /// Does not consult `merge_node_deltas`. Non-materialising read with
    /// RYOW from the write buffer + snapshot fallback, no OCC scope tracking
    /// (a schema-introspection read, not a transactional read). Delegates to
    /// [`coordinode_modality::NodeStore::peek_raw`].
    pub fn schema_peek_node_typed(
        &self,
        shard_id: u16,
        node_id: NodeId,
    ) -> Result<Option<NodeRecord>, ExecutionError> {
        // Layer-4 LocalNodeStore owns key encoding; this is an untracked peek
        // (no OCC, no delta materialisation). The transaction is pinned to the
        // statement snapshot in the prelude, so this `&self` read sees the same
        // point-in-time as `mvcc_snapshot`.
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        let Some(bytes) = LocalNodeStore.peek_raw(&self.txn, shard_id, node_id)? else {
            return Ok(None);
        };
        NodeRecord::from_msgpack(&bytes).map(Some).map_err(|e| {
            ExecutionError::Serialization(format!(
                "node {} schema-peek deserialization: {e}",
                node_id.as_raw(),
            ))
        })
    }

    /// Return the primary label for a node, using the per-statement cache.
    ///
    /// On first access for a given `node_id`, reads the node bytes via
    /// `schema_peek_node` and deserializes the `NodeRecord` to extract the
    /// primary label, then stores it in `schema_label_cache`.
    ///
    /// Subsequent calls for the same `node_id` within the same statement return
    /// the cached label without any engine I/O or deserialization.
    ///
    /// Returns `None` if the node does not exist (deleted or never created).
    pub fn schema_label_for_node(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
    ) -> Result<Option<String>, ExecutionError> {
        if let Some(label) = self.schema_label_cache.get(&node_id) {
            return Ok(Some(label.clone()));
        }
        let Some(record) = self.schema_peek_node_typed(shard_id, node_id)? else {
            return Ok(None);
        };
        let label = record.primary_label().to_string();
        self.schema_label_cache.insert(node_id, label.clone());
        Ok(Some(label))
    }

    /// Lazily materialise the Layer-3 OCC scope for this transaction.
    /// Returns `None` when MVCC is inactive (legacy mode has no
    /// conflict detection).
    ///
    /// The scope is pinned at `mvcc_read_ts.as_raw()` — every read
    /// recorded via [`OccScope::track`] becomes part of the read-set
    /// validated at commit time.
    fn ensure_occ_scope(&mut self) -> Option<&coordinode_storage::engine::coordinator::OccScope> {
        self.sync_txn_state();
        self.txn.ensure_occ_scope()
    }

    /// Sync the Layer-3 transaction's read snapshot + read timestamp from the
    /// executor's current values. Transitional (ADR-041): the executor still
    /// owns `mvcc_snapshot` / `mvcc_read_ts` (assigned post-construction and
    /// read pervasively), so we refresh the transaction's copies at the point
    /// of each read. Called at the entry of every read primitive; the
    /// transaction's OCC scope is created lazily at the first tracked read,
    /// pinned at the then-current `read_ts` (matching prior behaviour).
    fn sync_txn_state(&mut self) {
        self.txn.set_oracle(self.mvcc_oracle);
        self.txn.set_snapshot(self.mvcc_snapshot);
        self.txn.set_read_ts(self.mvcc_read_ts);
    }

    /// MVCC-aware read: write buffer → snapshot O(1) → legacy fallback.
    ///
    /// 1. Check write buffer (read-your-own-writes within this statement)
    /// 2. If MVCC snapshot set: snapshot.get() — O(1) native seqno MVCC (ADR-016)
    /// 3. If no snapshot: direct engine.get() — legacy mode
    ///
    /// Generic test-access primitive used by this crate's and downstream
    /// crates' tests. Production execution reads through the typed Layer-4
    /// stores (no `Partition` in production query paths, ADR-041).
    #[allow(clippy::disallowed_types)] // partition-parameterised primitive
    pub fn mvcc_get(
        &mut self,
        part: Partition,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, ExecutionError> {
        self.sync_txn_state();
        // RYOW for pending node merge deltas: materialize into the write
        // buffer so the transaction read below sees the correct state. This
        // node-modality concern stays above the modality-agnostic Layer-3
        // transaction (ADR-041).
        if part == Partition::Node && self.txn.node_deltas().iter().any(|(k, _)| k == key) {
            self.materialize_node_deltas(key)?;
        }
        // Buffer (RYOW) → snapshot read + OCC tracking, all owned by the
        // Layer-3 transaction.
        Ok(self.txn.get(part, key)?)
    }

    /// MVCC-aware write: buffers in write_buffer for atomic flush at commit.
    ///
    /// When MVCC is disabled (legacy mode), writes directly to engine.
    ///
    /// Generic test-access primitive (see [`Self::mvcc_get`]). Production DDL
    /// and data writes go through the typed Layer-4 stores.
    #[allow(clippy::disallowed_types)] // partition-parameterised primitive
    pub fn mvcc_put(
        &mut self,
        part: Partition,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), ExecutionError> {
        self.sync_txn_state();
        Ok(self.txn.put(part, key, value)?)
    }

    /// Read the current label schema by name. Returns `None` if the label
    /// has no schema declared.
    ///
    /// Resolves the indirection through `schema:current_revision:label:<name>`
    /// to find the active schema revision, then loads
    /// `schema:label:<name>:<revision>`. Per ADR-023 the schema partition is
    /// revision-prefixed from day one; CE deployments only ever see revision 1
    /// (no `ALTER LABEL SHARD BY` in CE), but the read path traverses the
    /// pointer regardless so the same code handles future multi-revision EE.
    pub fn load_current_label_schema(
        &mut self,
        name: &str,
    ) -> Result<Option<LabelSchema>, ExecutionError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        Ok(LocalSchemaStore::new(self.engine).load_label_txn(&mut self.txn, name)?)
    }

    /// Read the current edge type schema by name. Returns `None` if no schema
    /// is declared for this edge type.
    ///
    /// Symmetric to [`Self::load_current_label_schema`]: resolves the indirection
    /// through `schema:current_revision:edge_type:<name>` to find the active
    /// revision, then loads `schema:edge_type:<name>:<revision>`. Handles legacy
    /// zero-length idempotent existence markers (predates DDL) by returning
    /// `None` rather than erroring.
    pub fn load_current_edge_type_schema(
        &mut self,
        name: &str,
    ) -> Result<Option<EdgeTypeSchema>, ExecutionError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        Ok(LocalSchemaStore::new(self.engine).load_edge_type_txn(&mut self.txn, name)?)
    }

    /// Write an edge type schema as the current revision: writes the schema body
    /// at `schema:edge_type:<name>:<schema.schema_revision>` and updates the pointer
    /// `schema:current_revision:edge_type:<name>` to that revision.
    pub fn save_current_edge_type_schema(
        &mut self,
        schema: &EdgeTypeSchema,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        Ok(LocalSchemaStore::new(self.engine).save_edge_type_txn(&mut self.txn, schema)?)
    }

    /// Write a label schema as the current revision: writes the schema body
    /// at `schema:label:<name>:<schema.schema_revision>` and updates the pointer
    /// `schema:current_revision:label:<name>` to that revision. Use this for
    /// the canonical "create or update current schema" operation.
    pub fn save_current_label_schema(
        &mut self,
        schema: &LabelSchema,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        Ok(LocalSchemaStore::new(self.engine).save_label_txn(&mut self.txn, schema)?)
    }

    /// MVCC-aware delete: buffers tombstone in write_buffer for atomic flush.
    ///
    /// When MVCC is disabled (legacy mode), deletes directly from engine.
    ///
    /// Generic test-access primitive (see [`Self::mvcc_get`]). Production DDL
    /// and data writes go through the typed Layer-4 stores.
    #[allow(clippy::disallowed_types)] // partition-parameterised primitive
    pub fn mvcc_delete(&mut self, part: Partition, key: &[u8]) -> Result<(), ExecutionError> {
        self.sync_txn_state();
        Ok(self.txn.delete(part, key)?)
    }

    /// Persist an index definition transactionally (CREATE INDEX DDL). Buffers
    /// on the statement transaction through the Layer-4 index store, which
    /// owns the `schema:idx:` keyspace and encoding — Layer 5 names neither.
    pub fn mvcc_put_index_def(
        &mut self,
        def: &crate::index::IndexDefinition,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{IndexStore as _, LocalIndexStore};
        self.sync_txn_state();
        Ok(LocalIndexStore::new(self.engine).put_definition_txn(&mut self.txn, def)?)
    }

    /// Delete an index definition by name transactionally (DROP INDEX DDL).
    pub fn mvcc_delete_index_def(&mut self, name: &str) -> Result<(), ExecutionError> {
        use coordinode_modality::{IndexStore as _, LocalIndexStore};
        self.sync_txn_state();
        Ok(LocalIndexStore::new(self.engine).delete_definition_txn(&mut self.txn, name)?)
    }

    /// Persist an encrypted-index definition transactionally (CREATE ENCRYPTED
    /// INDEX DDL) through the Layer-4 store, which owns the `encrypted_index:`
    /// keyspace and encoding.
    pub fn mvcc_put_encrypted_index_def(
        &mut self,
        def: &coordinode_modality::EncryptedIndexDefinition,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EncryptedIndexStore as _, LocalEncryptedIndexStore};
        self.sync_txn_state();
        Ok(LocalEncryptedIndexStore::new(self.engine).put_definition_txn(&mut self.txn, def)?)
    }

    /// Delete an encrypted-index definition by name transactionally (DROP
    /// ENCRYPTED INDEX DDL).
    pub fn mvcc_delete_encrypted_index_def(&mut self, name: &str) -> Result<(), ExecutionError> {
        use coordinode_modality::{EncryptedIndexStore as _, LocalEncryptedIndexStore};
        self.sync_txn_state();
        Ok(
            LocalEncryptedIndexStore::new(self.engine)
                .delete_definition_txn(&mut self.txn, name)?,
        )
    }

    /// MVCC-aware typed node read.
    ///
    /// Combines key encoding, raw read through [`Self::mvcc_get`]
    /// (which handles snapshot pin, RYOW, and OCC tracking), and
    /// MessagePack decode in one call. Replaces the manual
    /// `encode_node_key` + `mvcc_get` + `from_msgpack` boilerplate at
    /// runner-level Node read sites — Layer-5 wrapper over Layer-4
    /// `LocalNodeStore` with MVCC orchestration preserved.
    pub fn mvcc_get_node(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
    ) -> Result<Option<NodeRecord>, ExecutionError> {
        // Layer-4 LocalNodeStore owns key encoding + node-delta RYOW and does
        // the OCC-tracked read (ADR-041); the query layer keeps the decode +
        // its diagnostic error contract (node id in the message).
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        self.sync_txn_state();
        let Some(bytes) = LocalNodeStore.get_raw_tracked(&mut self.txn, shard_id, node_id)? else {
            return Ok(None);
        };
        let record = NodeRecord::from_msgpack(&bytes).map_err(|e| {
            ExecutionError::Serialization(format!("node {} deserialization: {e}", node_id.as_raw()))
        })?;
        Ok(Some(record))
    }

    /// Read a temporal node's valid-version active at `at_ms` — the version
    /// whose `valid_from <= at_ms` is largest. Used to read the source body for
    /// CLONE NODE on a temporal label (current version when `at_ms = NOW`, a
    /// historical version with `AS OF <ts>`). `None` if no version is at-or-
    /// before that instant.
    pub fn mvcc_get_node_at(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
        at_ms: i64,
    ) -> Result<Option<NodeRecord>, ExecutionError> {
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        self.sync_txn_state();
        Ok(LocalNodeStore.get_at(&self.txn, shard_id, node_id, at_ms)?)
    }

    /// MVCC-aware typed node write. Buffers the put through
    /// [`Self::mvcc_put`] (for atomic flush + RYOW visibility). Replaces
    /// `encode_node_key + record.to_msgpack + mvcc_put` triples scattered
    /// across CREATE / SET / UPDATE executors.
    pub fn mvcc_put_node(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
        record: &NodeRecord,
    ) -> Result<(), ExecutionError> {
        // Delegate to Layer-4 LocalNodeStore (owns key encoding + msgpack);
        // the put buffers on the transaction for atomic flush (ADR-041).
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        self.sync_txn_state();
        Ok(LocalNodeStore.put(&mut self.txn, shard_id, node_id, record)?)
    }

    /// MVCC-aware typed read of a temporal node version.
    ///
    /// Reads the row at the 25-byte temporal key (shard, id,
    /// `valid_from_ms`). Preserves snapshot pin, RYOW, and OCC
    /// tracking — same machinery as [`Self::mvcc_get_node`], just
    /// scoped to a specific per-version key. Returns `None` when no
    /// row exists at that exact `valid_from`.
    pub fn mvcc_get_node_temporal(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
    ) -> Result<Option<NodeRecord>, ExecutionError> {
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        self.sync_txn_state();
        let Some(bytes) = LocalNodeStore.get_temporal_raw_tracked(
            &mut self.txn,
            shard_id,
            node_id,
            valid_from_ms,
        )?
        else {
            return Ok(None);
        };
        let record = NodeRecord::from_msgpack(&bytes).map_err(|e| {
            ExecutionError::Serialization(format!(
                "node {} temporal@{valid_from_ms} deserialization: {e}",
                node_id.as_raw(),
            ))
        })?;
        Ok(Some(record))
    }

    /// MVCC-aware typed write of a temporal node version.
    ///
    /// Buffers the put at the 25-byte temporal key (shard, id,
    /// `valid_from_ms`) through `mvcc_put` (atomic flush + RYOW).
    /// Used by the bitemporal close-current / open-new write pair
    /// in the temporal executor.
    pub fn mvcc_put_node_temporal(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
        record: &NodeRecord,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        self.sync_txn_state();
        Ok(LocalNodeStore.put_temporal(&mut self.txn, shard_id, node_id, valid_from_ms, record)?)
    }

    /// MVCC-aware typed delete of a temporal node version (tombstone
    /// at the specific 25-byte temporal key). Reserved for a future
    /// GDPR erasure path that hard-deletes individual versions; the
    /// standard bitemporal delete path uses close-current +
    /// tombstone-version inserts via [`Self::mvcc_put_node_temporal`]
    /// instead. No production caller in coordinode-query yet — added
    /// alongside the read/write pair so the typed surface stays
    /// symmetric.
    pub fn mvcc_delete_node_temporal(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: i64,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        self.sync_txn_state();
        Ok(LocalNodeStore.delete_temporal(&mut self.txn, shard_id, node_id, valid_from_ms)?)
    }

    /// MVCC-aware typed edge-property read. Hides the
    /// `encode_edgeprop_key + mvcc_get + rmp_serde decode` triple
    /// used by traversal and merge paths. Returns `None` when the
    /// edge has no property body (the common case for property-less
    /// edges). The decoded shape matches the on-disk layout — a
    /// `Vec<(interned_field_id, Value)>` — so callers reuse it
    /// without conversion to/from `EdgeProperties::HashMap`.
    pub fn mvcc_get_edge_props(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> Result<Option<Vec<(u32, Value)>>, ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        let Some(bytes) =
            LocalEdgeStore.get_props_raw_tracked(&mut self.txn, edge_type, src, tgt, None)?
        else {
            return Ok(None);
        };
        let decoded = decode_edge_props(&bytes).map_err(|e| {
            ExecutionError::Serialization(format!(
                "edge prop {edge_type}/{}/{} decode: {e}",
                src.as_raw(),
                tgt.as_raw(),
            ))
        })?;
        Ok(Some(decoded))
    }

    /// MVCC-aware typed edge-property write. Counterpart to
    /// [`Self::mvcc_get_edge_props`] — encodes the
    /// `Vec<(field_id, Value)>` property list with rmp_serde and
    /// buffers a write at the `(edge_type, src, tgt)` EdgeProp key.
    pub fn mvcc_put_edge_props(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        prop_map: &[(u32, Value)],
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.put_props(&mut self.txn, edge_type, src, tgt, None, prop_map)?)
    }

    /// MVCC-aware typed edge-property read at a specific temporal
    /// version. Reads the row at the temporal EdgeProp key
    /// `(edge_type, src, tgt, valid_from_ms)`. Same MVCC semantics
    /// as [`Self::mvcc_get_edge_props`].
    pub fn mvcc_get_edge_props_temporal(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
    ) -> Result<Option<Vec<(u32, Value)>>, ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        let Some(bytes) = LocalEdgeStore.get_props_raw_tracked(
            &mut self.txn,
            edge_type,
            src,
            tgt,
            Some(valid_from_ms),
        )?
        else {
            return Ok(None);
        };
        let decoded = decode_edge_props(&bytes).map_err(|e| {
            ExecutionError::Serialization(format!(
                "edge prop {edge_type}/{}/{} temporal@{valid_from_ms} decode: {e}",
                src.as_raw(),
                tgt.as_raw(),
            ))
        })?;
        Ok(Some(decoded))
    }

    /// MVCC-aware typed edge-property read that branches on a
    /// runtime temporal flag. `Some(vf)` reads the per-version key,
    /// `None` reads the non-temporal key. Counterpart to
    /// [`Self::mvcc_get_node_either`] for edges.
    pub fn mvcc_get_edge_props_either(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
    ) -> Result<Option<Vec<(u32, Value)>>, ExecutionError> {
        match valid_from_ms {
            Some(vf) => self.mvcc_get_edge_props_temporal(edge_type, src, tgt, vf),
            None => self.mvcc_get_edge_props(edge_type, src, tgt),
        }
    }

    /// MVCC-aware typed edge-property write at a specific temporal
    /// version. Counterpart to [`Self::mvcc_get_edge_props_temporal`].
    pub fn mvcc_put_edge_props_temporal(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
        prop_map: &[(u32, Value)],
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.put_props(
            &mut self.txn,
            edge_type,
            src,
            tgt,
            Some(valid_from_ms),
            prop_map,
        )?)
    }

    /// MVCC-aware typed edge-property write that branches on a
    /// runtime temporal flag. `Some(vf)` writes at the per-version
    /// key, `None` writes at the non-temporal key.
    pub fn mvcc_put_edge_props_either(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
        prop_map: &[(u32, Value)],
    ) -> Result<(), ExecutionError> {
        match valid_from_ms {
            Some(vf) => self.mvcc_put_edge_props_temporal(edge_type, src, tgt, vf, prop_map),
            None => self.mvcc_put_edge_props(edge_type, src, tgt, prop_map),
        }
    }

    /// MVCC-aware typed edge-property delete. Tombstones the
    /// non-temporal EdgeProp key. Used by edge DELETE paths and
    /// the transfer-edges remap (old key drops after the new key
    /// has been written).
    pub fn mvcc_delete_edge_props(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.delete_props(&mut self.txn, edge_type, src, tgt, None)?)
    }

    /// Raw bytes of every temporal edge-property version for
    /// `(edge_type, src, tgt)`, optionally bounded to `valid_from <= upper_ms`.
    /// Delegates the key shape + scan to the Layer-4 store; the caller decodes.
    pub fn mvcc_scan_edge_prop_version_bytes(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        upper_ms: Option<i64>,
    ) -> Result<Vec<(i64, Vec<u8>)>, ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.scan_versions_raw_tracked(
            &mut self.txn,
            edge_type,
            src,
            tgt,
            upper_ms,
        )?)
    }

    /// Raw bytes of a single edge-property body (non-temporal or per-version).
    /// Delegates the key shape to the Layer-4 store; the caller decodes.
    pub fn mvcc_get_edge_prop_bytes(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
    ) -> Result<Option<Vec<u8>>, ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.get_props_raw_tracked(
            &mut self.txn,
            edge_type,
            src,
            tgt,
            valid_from_ms,
        )?)
    }

    /// Move a non-temporal edge-property body from `(old_src, old_tgt)` to
    /// `(new_src, new_tgt)` (edge-rewiring). Delegates to the Layer-4 store.
    pub fn mvcc_move_edge_props(
        &mut self,
        edge_type: &str,
        old_src: NodeId,
        old_tgt: NodeId,
        new_src: NodeId,
        new_tgt: NodeId,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.move_props(
            &mut self.txn,
            edge_type,
            old_src,
            old_tgt,
            new_src,
            new_tgt,
        )?)
    }

    /// Tombstone every temporal edge-property version of `(edge_type, src, tgt)`.
    pub fn mvcc_delete_all_edge_prop_versions(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.delete_all_versions(&mut self.txn, edge_type, src, tgt)?)
    }

    /// Move every temporal edge-property version preserving each `valid_from`.
    pub fn mvcc_transfer_all_edge_prop_versions(
        &mut self,
        edge_type: &str,
        old_src: NodeId,
        old_tgt: NodeId,
        new_src: NodeId,
        new_tgt: NodeId,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.transfer_all_versions(
            &mut self.txn,
            edge_type,
            old_src,
            old_tgt,
            new_src,
            new_tgt,
        )?)
    }

    /// MVCC-aware typed edge-property delete at a specific temporal
    /// version. Tombstones the 25-byte per-version EdgeProp key.
    pub fn mvcc_delete_edge_props_temporal(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: i64,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.delete_props(&mut self.txn, edge_type, src, tgt, Some(valid_from_ms))?)
    }

    /// MVCC-aware typed edge-property delete that branches on a
    /// runtime temporal flag. `Some(vf)` tombstones the per-version
    /// key, `None` tombstones the non-temporal key.
    pub fn mvcc_delete_edge_props_either(
        &mut self,
        edge_type: &str,
        src: NodeId,
        tgt: NodeId,
        valid_from_ms: Option<i64>,
    ) -> Result<(), ExecutionError> {
        match valid_from_ms {
            Some(vf) => self.mvcc_delete_edge_props_temporal(edge_type, src, tgt, vf),
            None => self.mvcc_delete_edge_props(edge_type, src, tgt),
        }
    }

    /// MVCC-aware typed node read that branches on a runtime
    /// temporal flag. When `valid_from_ms` is `Some(vf)`, reads the
    /// per-version row at the 25-byte temporal key; when `None`,
    /// reads the 16-byte non-temporal key. Used by DETACH / ATTACH
    /// executor paths where the source label's temporal flag is only
    /// known at runtime from the bound row.
    pub fn mvcc_get_node_either(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: Option<i64>,
    ) -> Result<Option<NodeRecord>, ExecutionError> {
        match valid_from_ms {
            Some(vf) => self.mvcc_get_node_temporal(shard_id, node_id, vf),
            None => self.mvcc_get_node(shard_id, node_id),
        }
    }

    /// MVCC-aware typed node write that branches on a runtime
    /// temporal flag. Symmetric counterpart to
    /// [`Self::mvcc_get_node_either`] — writes at the temporal key
    /// when `valid_from_ms = Some(vf)`, otherwise at the non-temporal
    /// 16-byte key.
    pub fn mvcc_put_node_either(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
        valid_from_ms: Option<i64>,
        record: &NodeRecord,
    ) -> Result<(), ExecutionError> {
        match valid_from_ms {
            Some(vf) => self.mvcc_put_node_temporal(shard_id, node_id, vf, record),
            None => self.mvcc_put_node(shard_id, node_id, record),
        }
    }

    /// Buffer a node document-delta operand for atomic flush. Hides
    /// the `encode_node_key` + `merge_node_deltas.push` pattern.
    /// The operand is a pre-encoded `DocDelta` operand bytes blob —
    /// callers build it via `DocDelta::encode()`. Used by SET / REMOVE
    /// nested-path executors (e.g. `SET n.config.host = "x"`,
    /// `REMOVE n.tags[0]`).
    pub fn mvcc_merge_node_delta(&mut self, shard_id: u16, node_id: NodeId, operand: Vec<u8>) {
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        LocalNodeStore.buffer_node_delta(&mut self.txn, shard_id, node_id, operand);
    }

    /// MVCC-aware typed node delete. Buffers a tombstone for the
    /// non-temporal node key (16-byte `encode_node_key` form). Does
    /// NOT iterate temporal version rows (25-byte `temporal_node_key`
    /// form) — temporal cleanup is handled by the per-version delete
    /// paths (close-current + tombstone) that the temporal executor
    /// invokes directly.
    pub fn mvcc_delete_node(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        self.sync_txn_state();
        Ok(LocalNodeStore.delete(&mut self.txn, shard_id, node_id)?)
    }

    /// Flush MVCC write buffer to storage with a commit timestamp.
    ///
    /// Called at the end of statement execution. Assigns commit_ts from
    /// the oracle, performs OCC conflict detection against the read-set,
    /// and writes all buffered mutations with versioned keys.
    ///
    /// ## OCC Conflict Detection
    ///
    /// Before flushing writes, delegates to Layer-3
    /// `MultiModalCoordinator::validate_occ` which walks `occ_scope`'s
    /// tracked keys and checks each for a version with `seqno >
    /// read_ts`. If any such version exists, another transaction
    /// modified a key we read → read-write conflict → `ErrConflict`.
    ///
    /// `adj:` partition keys are excluded from conflict checking because
    /// posting list operations are commutative (use merge operators).
    ///
    /// Returns the commit_ts used, or `ErrConflict` if a conflict is detected.
    pub fn mvcc_flush(&mut self) -> Result<Option<Timestamp>, ExecutionError> {
        // Push the executor's per-statement context (oracle, snapshot, read_ts)
        // into the Layer-3 transaction, then delegate to the single commit
        // locus (ADR-041): OCC validation, commit_ts assignment, write-concern
        // fan-out, and the Raft proposal pipeline all live in `Transaction`.
        self.sync_txn_state();
        let write_concern = self.write_concern.clone();
        let ctx = coordinode_storage::engine::transaction::CommitContext {
            write_concern: &write_concern,
            pipeline: self.proposal_pipeline,
            id_gen: self.proposal_id_gen,
            drain_buffer: self.drain_buffer,
            nvme_write_buffer: self.nvme_write_buffer,
        };
        let outcome = self.txn.commit(&ctx).map_err(commit_err_to_execution)?;
        // operationTime spans every Raft entry the statement produced — `max`,
        // not assign: a statement may have issued an earlier in-execute
        // proposal (e.g. CREATE VECTOR INDEX schema persist) whose index must
        // remain covered. (`Option` orders `None < Some`.)
        self.write_stats.applied_index = self.write_stats.applied_index.max(outcome.applied_index);
        Ok(outcome.commit_ts)
    }

    /// Materialize pending node merge deltas for a key into the write buffer.
    ///
    /// Called lazily from the generic `mvcc_get()` primitive. Production node
    /// reads materialise deltas inside `LocalNodeStore` (ADR-041).
    fn materialize_node_deltas(&mut self, node_key: &[u8]) -> Result<(), ExecutionError> {
        // Node-delta read-your-own-writes lives in Layer-4 LocalNodeStore
        // (node-modality concern); the modality-agnostic transaction does not
        // own it (ADR-041).
        Ok(
            coordinode_modality::LocalNodeStore::materialize_pending_deltas(
                &mut self.txn,
                node_key,
            )?,
        )
    }

    /// MVCC-aware prefix scan: returns deduplicated (user_key, value) pairs
    /// visible at the current snapshot timestamp.
    ///
    /// In legacy mode, returns raw prefix scan results as (key, value) pairs.
    ///
    /// Generic test-access primitive (see [`Self::mvcc_get`]).
    #[allow(clippy::disallowed_types)] // partition-parameterised primitive
    pub fn mvcc_prefix_scan(
        &mut self,
        part: Partition,
        prefix: &[u8],
    ) -> Result<Vec<KvPair>, ExecutionError> {
        self.sync_txn_state();
        // RYOW for pending node merge deltas: materialize all matching keys
        // into write buffer so the scan sees up-to-date values.
        if part == Partition::Node && !self.txn.node_deltas().is_empty() {
            let matching_keys: Vec<Vec<u8>> = self
                .txn
                .node_deltas()
                .iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .map(|(k, _)| k.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            for key in matching_keys {
                self.materialize_node_deltas(&key)?;
            }
        }
        // Buffer overlay + snapshot scan + OCC tracking, owned by the Layer-3
        // transaction.
        Ok(self.txn.prefix_scan(part, prefix)?)
    }

    // ── Adj partition: raw posting read (test-only) ──
    // Production adjacency access goes through the typed EdgeStore methods
    // (posting_fwd/rev, posting_for_key, merge_add_fwd/rev, …) — no raw adj
    // key or `Partition::Adj` in the query layer (ADR-041).

    /// Read an adjacency posting list (raw key, no MVCC timestamp).
    ///
    /// Reads the latest merged value from StorageEngine, then applies
    /// pending adds/removes from this transaction's merge buffer
    /// (read-your-own-writes). Generic test-access primitive.
    pub fn adj_get(&self, adj_key: &[u8]) -> Result<Option<PostingList>, ExecutionError> {
        // RYOW for buffered tombstones / puts on the adj key. When a prior
        // operation in this transaction wrote (`mvcc_put`) or deleted
        // (`mvcc_delete`) the same adj key, the buffered effect must win over
        // the on-disk base — otherwise reads return stale data and a
        // transaction abort would leave partial mutations applied.
        //
        // Read-only traversals carry an empty write buffer, so probe it only
        // when it holds something: that skips a per-read key `Vec` allocation
        // on the adjacency hot path (the lookup key owns its bytes).
        let buffered = if self.txn.write_buffer_is_empty() {
            None
        } else {
            self.txn.buffered(Partition::Adj, adj_key)
        };

        // Parse the base posting list directly from the borrowed bytes in every
        // branch: no intermediate `Vec` copy. The storage reads hand back a
        // refcounted `Bytes` and the buffered overlay holds a `Vec`; both deref
        // to `&[u8]`, which is all `from_bytes` needs.
        let mut plist = match buffered {
            // Buffered tombstone wins over on-disk state. Merge operands
            // accumulated AFTER the tombstone still apply (start from empty).
            Some(None) => PostingList::new(),
            Some(Some(bytes)) => PostingList::from_bytes(bytes)
                .map_err(|e| ExecutionError::Serialization(format!("posting list: {e}")))?,
            None => {
                // No buffered overlay — read base posting list from storage
                // through the Layer-3 adjacency base read (snapshot-aware when
                // adj_snapshot is set, e.g. AS OF TIMESTAMP).
                match self.txn.adj_base_get(adj_key)? {
                    Some(b) => PostingList::from_bytes(&b)
                        .map_err(|e| ExecutionError::Serialization(format!("posting list: {e}")))?,
                    None => PostingList::new(),
                }
            }
        };

        // Apply pending adds from this transaction (read-your-own-writes).
        if let Some(adds) = self.txn.merge_adj_adds().get(adj_key) {
            for &uid in adds {
                plist.insert(uid);
            }
        }
        // Apply pending removes from this transaction.
        if let Some(removes) = self.txn.merge_adj_removes().get(adj_key) {
            for &uid in removes {
                plist.remove(uid);
            }
        }

        if plist.is_empty() {
            Ok(None)
        } else {
            Ok(Some(plist))
        }
    }

    // ── Typed adjacency wrappers ──────────────────────────────────
    // Hide adj-key encoding from Layer-5 call sites: callers pass
    // `(edge_type, node)` instead of a pre-encoded key. All delegate to
    // the raw merge-path methods above, so the commutative merge +
    // read-your-own-writes + AS-OF snapshot semantics are unchanged.
    // Adjacency stays off the OCC path by construction (edge add/remove
    // is commutative — see storage-engine.md merge-operator contract).

    /// Forward-adjacency read (`src`'s out-neighbours for `edge_type`).
    pub fn adj_get_fwd(
        &self,
        edge_type: &str,
        src: NodeId,
    ) -> Result<Option<PostingList>, ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        Ok(LocalEdgeStore.posting_fwd(&self.txn, edge_type, src)?)
    }

    /// Reverse-adjacency read (`tgt`'s in-neighbours for `edge_type`).
    pub fn adj_get_rev(
        &self,
        edge_type: &str,
        tgt: NodeId,
    ) -> Result<Option<PostingList>, ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        Ok(LocalEdgeStore.posting_rev(&self.txn, edge_type, tgt)?)
    }

    /// Buffer a forward-adjacency add (`src` gains out-neighbour `uid`).
    pub fn adj_merge_add_fwd(&mut self, edge_type: &str, src: NodeId, uid: u64) {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        LocalEdgeStore.merge_add_fwd(&mut self.txn, edge_type, src, uid);
    }

    /// Buffer a reverse-adjacency add (`tgt` gains in-neighbour `uid`).
    pub fn adj_merge_add_rev(&mut self, edge_type: &str, tgt: NodeId, uid: u64) {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        LocalEdgeStore.merge_add_rev(&mut self.txn, edge_type, tgt, uid);
    }

    /// Buffer a forward-adjacency remove (`src` loses out-neighbour `uid`).
    pub fn adj_merge_remove_fwd(&mut self, edge_type: &str, src: NodeId, uid: u64) {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        LocalEdgeStore.merge_remove_fwd(&mut self.txn, edge_type, src, uid);
    }

    /// Buffer a reverse-adjacency remove (`tgt` loses in-neighbour `uid`).
    pub fn adj_merge_remove_rev(&mut self, edge_type: &str, tgt: NodeId, uid: u64) {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        LocalEdgeStore.merge_remove_rev(&mut self.txn, edge_type, tgt, uid);
    }

    /// Buffered wholesale delete of a node's forward posting list. Goes
    /// through the MVCC write buffer so it rolls back with the
    /// transaction (used by MERGE NODES source-side teardown).
    pub fn mvcc_delete_adj_fwd(
        &mut self,
        edge_type: &str,
        src: NodeId,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.delete_adj_fwd(&mut self.txn, edge_type, src)?)
    }

    /// Buffered wholesale delete of a node's reverse posting list.
    pub fn mvcc_delete_adj_rev(
        &mut self,
        edge_type: &str,
        tgt: NodeId,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.delete_adj_rev(&mut self.txn, edge_type, tgt)?)
    }

    /// Buffered wholesale purge of a node's posting list on `dir`, plus
    /// drop of any pending merge operands still buffered for that key.
    /// The MVCC tombstone rolls back with the transaction; clearing the
    /// pending adds/removes stops a buffered edge from resurrecting the
    /// posting list the node-delete cascade just tombstoned.
    pub fn mvcc_purge_adj(
        &mut self,
        edge_type: &str,
        node: NodeId,
        dir: AdjDirection,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
        self.sync_txn_state();
        Ok(LocalEdgeStore.purge_adj(&mut self.txn, edge_type, node, dir == AdjDirection::Out)?)
    }

    /// MVCC-aware idempotent edge-type registration. Writes the empty
    /// existence marker at `schema:edge_type:<name>:1` only if it is not
    /// already present — checking the transaction's write buffer (RYOW)
    /// before the OCC-tracked snapshot read so a concurrent
    /// `CREATE EDGE TYPE` body is never clobbered. Hides the
    /// `Partition::Schema` key from Layer-5 call sites while keeping the
    /// read on the Layer-3 OCC path (schema reads are not commutative, so
    /// they must participate in conflict detection).
    pub fn mvcc_register_edge_type(&mut self, edge_type: &str) -> Result<(), ExecutionError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        Ok(LocalSchemaStore::new(self.engine)
            .register_edge_type_marker(&mut self.txn, edge_type)?)
    }

    /// MVCC-aware existence probe for an edge type. `true` if either the
    /// explicit `CREATE EDGE TYPE` revision pointer or the implicit
    /// edge-create existence marker (revision 1) is present. Both reads go
    /// through the OCC-tracked snapshot path. Hides `Partition::Schema`
    /// from DDL call sites.
    pub fn mvcc_edge_type_exists(&mut self, name: &str) -> Result<bool, ExecutionError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        Ok(LocalSchemaStore::new(self.engine).edge_type_exists(&mut self.txn, name)?)
    }

    /// Raw prefix scan on adj: partition (no MVCC timestamp filtering).
    ///
    /// Returns (raw_key, raw_value) pairs.
    pub fn adj_prefix_scan(&self, prefix: &[u8]) -> Result<Vec<KvPair>, ExecutionError> {
        // Snapshot-aware base adjacency scan lives at Layer 3 (reads through the
        // adjacency snapshot when set, so post-snapshot merge operands stay
        // invisible).
        Ok(self.txn.adj_base_prefix_scan(prefix)?)
    }

    /// Return all edge type names registered in the Schema partition.
    ///
    /// Used by DETACH DELETE to build targeted adj key lookups instead of
    /// scanning every adj: key in the database.
    pub(crate) fn list_edge_types(&mut self) -> Result<Vec<String>, ExecutionError> {
        // Layer-4 SchemaStore owns the edge-type key shape; the tracked scan
        // overlays this transaction's write buffer, so same-tx
        // CREATE + DETACH DELETE sees freshly-registered types.
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        Ok(LocalSchemaStore::new(self.engine).list_edge_type_names(&mut self.txn)?)
    }
}

/// Decide whether variable-length traversal may dedup target-node emission for
/// this plan without changing any result (sets `dedup_varlen_targets`).
///
/// Safe ONLY when the whole plan is a read-only linear spine of the exact shape
/// `Aggregate[count(DISTINCT v), no GROUP BY] -> (Filter|Project|Sort|Limit|Skip
/// |single-hop Traverse)* -> Traverse(var-length, target = v, no path/edge var)
/// -> (Filter|Project|...)* -> Scan`. The aggregate collapses multiplicity, so
/// emitting each reached `v` once instead of once-per-edge yields the identical
/// `count(DISTINCT v)`. Any branch (CartesianProduct), mutation, second
/// aggregate, second var-length traverse, or unrecognised op makes the function
/// return `false` (the conservative default leaves emission unchanged), so it
/// can never enable dedup for a plan whose multiplicity is observable.
pub(crate) fn plan_allows_varlen_target_dedup(root: &LogicalOp) -> bool {
    // Skip read-only passthrough wrappers above the aggregate: a `RETURN ... AS
    // alias` becomes a Project, and ORDER BY / LIMIT / SKIP act on the single
    // already-aggregated row. None can re-expose the traverse's per-edge
    // multiplicity, so the count(DISTINCT v) below is the only consumer.
    let mut top = root;
    let aggregate = loop {
        match top {
            LogicalOp::Project { input, .. }
            | LogicalOp::Sort { input, .. }
            | LogicalOp::Limit { input, .. }
            | LogicalOp::Skip { input, .. } => top = input.as_ref(),
            LogicalOp::Aggregate { .. } => break top,
            _ => return false,
        }
    };
    let LogicalOp::Aggregate {
        input,
        group_by,
        aggregates,
    } = aggregate
    else {
        return false;
    };
    if !group_by.is_empty() || aggregates.len() != 1 {
        return false;
    }
    let item = &aggregates[0];
    if !item.distinct || !item.function.eq_ignore_ascii_case("count") {
        return false;
    }
    let Expr::Variable(target_var) = &item.arg else {
        return false;
    };

    // Walk the single-input spine below the aggregate. Exactly one var-length
    // traverse must appear, binding `target_var`, with no path or edge variable.
    let mut cur = input.as_ref();
    let mut found_qualifying = false;
    loop {
        match cur {
            LogicalOp::Traverse {
                input,
                target_variable,
                length,
                path_variable,
                edge_variable,
                ..
            } => {
                if length.is_some() {
                    // A second var-length traverse would also be deduped by the
                    // global flag — bail rather than reason about it.
                    if found_qualifying {
                        return false;
                    }
                    if path_variable.is_some()
                        || edge_variable.is_some()
                        || target_variable != target_var
                    {
                        return false;
                    }
                    found_qualifying = true;
                }
                cur = input.as_ref();
            }
            LogicalOp::Filter { input, .. }
            | LogicalOp::Project { input, .. }
            | LogicalOp::Sort { input, .. }
            | LogicalOp::Limit { input, .. }
            | LogicalOp::Skip { input, .. } => {
                cur = input.as_ref();
            }
            LogicalOp::NodeScan { .. }
            | LogicalOp::IndexScan { .. }
            | LogicalOp::HnswScan { .. }
            | LogicalOp::Empty => return found_qualifying,
            // Any branch / mutation / unrecognised op: stay safe, do not dedup.
            _ => return false,
        }
    }
}

/// Execute a logical plan against storage, returning result rows, and
/// auto-commit the statement's writes.
///
/// This is the single-statement (auto-commit) entry point: it runs the plan
/// via [`execute_no_commit`] and then flushes the transaction
/// ([`ExecutionContext::mvcc_flush`] — assign `commit_ts`, OCC validate,
/// persist via the Raft proposal pipeline). Interactive multi-statement
/// transactions (ADR-042) call [`execute_no_commit`] directly and commit the
/// shared transaction once, after the final statement.
pub fn execute(
    plan: &LogicalPlan,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let result = execute_no_commit(plan, ctx)?;
    // Flush MVCC write buffer: assign commit_ts and persist all buffered
    // writes. In legacy mode (mvcc_oracle: None) this is a no-op.
    ctx.mvcc_flush()?;
    Ok(result)
}

/// Execute a logical plan against storage WITHOUT committing.
///
/// Runs the full plan (snapshot setup, watermark wait, operator tree) and
/// leaves every mutation buffered on `ctx.txn` — the caller is responsible
/// for committing (or rolling back). [`execute`] wraps this with an
/// auto-commit; interactive transactions (ADR-042) run N statements through
/// this entry against one shared `ctx.txn` and commit once at the end.
pub fn execute_no_commit(
    plan: &LogicalPlan,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // ADR-016: Take MVCC snapshot at mvcc_read_ts for native seqno reads.
    // All reads (node, schema, edgeprop) go through this snapshot for O(1) lookups.
    // When oracle is set, snapshot_at(seqno) pins the LSM tree at read_ts.
    // For legacy mode (no oracle), mvcc_snapshot stays None → direct engine reads.
    if ctx.mvcc_snapshot.is_none() && ctx.mvcc_oracle.is_some() {
        ctx.mvcc_snapshot = ctx
            .engine
            .snapshot_at(ctx.mvcc_read_ts.as_raw())
            .or_else(|| Some(ctx.engine.snapshot()));
    }

    // Take a storage snapshot for statement-level adj: partition consistency.
    // When mvcc_snapshot is set, reuse it for adj reads too (same point-in-time).
    if ctx.txn.adj_snapshot().is_none() {
        if let Some(snap) = ctx.mvcc_snapshot {
            ctx.txn.set_adj_snapshot(Some(snap));
        } else {
            let snap = ctx.engine.snapshot();
            ctx.txn.set_adj_snapshot(Some(snap));
        }
    }

    // Pin the Layer-3 transaction to the statement snapshot/oracle/read_ts now,
    // so even `&self` reads (e.g. schema peeks) that go through the store see
    // the same point-in-time as the executor's `mvcc_snapshot` field.
    ctx.sync_txn_state();

    // Handle AS OF TIMESTAMP: evaluate the timestamp expression, override snapshots.
    // Since commit_ts = seqno (ADR-016, OracleSeqnoGenerator), the timestamp value
    // is directly usable as a snapshot seqno for both node and adj partitions.
    if let Some(ref ts_expr) = plan.snapshot_ts {
        let ts_val = eval_expr(ts_expr, &Row::new());
        let resolved_ts: Option<i64> = match ts_val {
            Value::Timestamp(ts) => {
                // Validate within retention window
                let now_us = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_micros() as i64)
                    .unwrap_or(0);
                let cutoff = now_us - ctx.retention_window_us;
                if ts < cutoff {
                    return Err(ExecutionError::Unsupported(format!(
                        "AS OF TIMESTAMP {} is outside retention window \
                         (oldest allowed: {})",
                        ts, cutoff
                    )));
                }
                Some(ts)
            }
            Value::String(ref s) => {
                ctx.warnings.push(format!(
                    "AS OF TIMESTAMP '{s}': string timestamps parsed as current \
                     (full datetime parsing pending)"
                ));
                None
            }
            Value::Int(ts) => Some(ts),
            _ => {
                return Err(ExecutionError::Unsupported(
                    "AS OF TIMESTAMP requires a timestamp or integer value".into(),
                ));
            }
        };

        if let Some(ts) = resolved_ts {
            ctx.snapshot_ts = Some(ts);

            // Override MVCC and adj snapshots to the requested timestamp.
            // This enables time-travel for BOTH node reads and edge traversal.
            #[allow(clippy::cast_sign_loss)]
            let seqno = ts as u64;
            if let Some(snap) = ctx.engine.snapshot_at(seqno) {
                ctx.mvcc_snapshot = Some(snap);
                ctx.txn.set_adj_snapshot(Some(snap));
                ctx.txn.set_snapshot(Some(snap));
            }
        }
    }

    // Substitute query parameters ($name → literal value) before execution.
    // Cloned only when params are present to avoid heap allocation in the common case.
    let plan_owned;
    let plan = if ctx.params.is_empty() {
        plan
    } else {
        plan_owned = {
            let mut p = plan.clone();
            p.substitute_params(&ctx.params);
            // Re-run the temporal-filter lift pass: it ran once at plan build
            // time, but parameter expressions weren't literals yet. Now that
            // substitution turned them into Literal values, push-down is
            // possible. Idempotent on plans where the lift already happened
            // (the matched arm requires `temporal_filter.is_none()`).
            p.root = crate::planner::builder::lift_temporal_filter(p.root);
            p
        };
        &plan_owned
    };

    // R-SNAP1: propagate the plan's cross-modality consistency decision into
    // the execution context so downstream operators (VectorFilter, etc.)
    // observe it without a separate parameter. The planner has already
    // applied auto-promotion and the narrower `vector_consistency` override.
    ctx.read_consistency = plan.read_consistency;
    ctx.vector_consistency = plan.vector_consistency;

    // R-SNAP1: cross-modality snapshot wait. When `read_consistency` is
    // `Snapshot` or `Exact`, every modality on this shard must observe the
    // fully-applied state at a single HLC timestamp T. Block on the
    // `MaxAssignedWatermark` until the applier has persisted every write
    // with `commit_ts ≤ T`; timeout returns `ErrReadTimeout` so the client
    // can retry or fall back to `Current`.
    //
    // Target ts: prefer the explicit `AS OF TIMESTAMP` value when present,
    // otherwise `mvcc_read_ts` (statement start_ts). If the watermark is not
    // wired into this context (legacy / single-writer test), skip the wait —
    // the test is responsible for sequencing reads after writes itself.
    if plan.read_consistency.requires_snapshot_wait() {
        if let Some(ref wm) = ctx.applied_watermark.clone() {
            let target_raw = ctx
                .snapshot_ts
                .map(|t| t.max(0) as u64)
                .unwrap_or_else(|| ctx.mvcc_read_ts.as_raw());
            if target_raw > 0 {
                let target = coordinode_core::txn::timestamp::Timestamp::from_raw(target_raw);
                let timeout = ctx.read_timeout;
                // The executor is sync; use the blocking helper that builds
                // a private current-thread runtime per call. `wait_for` is
                // typically sub-millisecond, so this is cheap.
                if let Err(err) = wm.wait_for_blocking(target, timeout) {
                    return Err(ExecutionError::Unsupported(format!(
                        "read_consistency='{}' timed out waiting for applier \
                         to reach commit_ts={target_raw}: {err}. Retry or fall \
                         back to read_consistency='current'.",
                        plan.read_consistency
                    )));
                }
            }
        }
    }

    // Enable per-node dedup of variable-length traversal emission when the plan
    // provably cannot observe target multiplicity (count(DISTINCT v) over a lone
    // var-length traverse). Collapses O(edges) emitted rows to O(reached nodes).
    ctx.dedup_varlen_targets = plan_allows_varlen_target_dedup(&plan.root);

    let result = execute_op(&plan.root, ctx)?;
    Ok(result)
}

/// Execute a single logical operator recursively.
fn execute_op(op: &LogicalOp, ctx: &mut ExecutionContext<'_>) -> Result<Vec<Row>, ExecutionError> {
    match op {
        // Extension ops are dispatched to the registered handler by name. The
        // Arc is cloned out first so the registry borrow ends before the
        // handler takes `&mut ctx`. CE registers no handlers, so an Extension
        // op without a registered handler is a clear error rather than silent.
        LogicalOp::Extension { name, payload } => {
            let handler = ctx.extensions.and_then(|r| r.get(name)).ok_or_else(|| {
                ExecutionError::Unsupported(format!("no handler for extension op: {name}"))
            })?;
            handler.execute(ctx, payload)
        }

        LogicalOp::NodeScan {
            variable,
            labels,
            property_filters,
        } => execute_node_scan(variable, labels, property_filters, ctx),

        LogicalOp::IndexScan {
            variable,
            label,
            index_name,
            property,
            value_expr,
        } => execute_btree_index_scan(variable, label, index_name, property, value_expr, ctx),

        // Index access path for pure vector top-K: the index IS the row
        // source; only the k result nodes are fetched from storage.
        LogicalOp::HnswScan {
            label,
            property,
            binding,
            query_vector,
            k,
            function,
            distance_alias,
            index_name,
        } => execute_hnsw_scan(
            label,
            property,
            binding,
            query_vector,
            *k,
            function,
            distance_alias.as_deref(),
            index_name,
            ctx,
        ),

        LogicalOp::Traverse {
            input,
            source,
            edge_types,
            direction,
            target_variable,
            target_labels,
            length,
            edge_variable,
            target_filters,
            edge_filters,
            temporal_filter,
            path_variable,
        } => {
            let input_rows = execute_op(input, ctx)?;

            // G069: wildcard relationship pattern `MATCH (n)-[r]->(m)` — no type filter.
            // When edge_types is empty, expand over all schema-registered edge types.
            // This scans `schema:edge_type:<name>` keys (written on every CREATE edge)
            // plus any uncommitted registrations in the current transaction's write buffer.
            let resolved_types: Vec<String>;
            let effective_types: &[String] = if edge_types.is_empty() {
                resolved_types = ctx.list_edge_types()?;
                &resolved_types
            } else {
                edge_types
            };

            // Hoist temporal flag per edge type once per traversal.
            let mut edge_temporal: Vec<bool> = Vec::with_capacity(effective_types.len());
            for et in effective_types {
                edge_temporal.push(lookup_edge_type_temporal(et, ctx)?);
            }

            let params = TraverseParams {
                source,
                edge_types: effective_types,
                direction: *direction,
                target_variable,
                target_labels,
                length: *length,
                edge_variable: edge_variable.as_deref(),
                target_filters,
                edge_filters,
                edge_temporal: &edge_temporal,
                temporal_filter: temporal_filter.as_ref(),
                path_variable: path_variable.as_deref(),
            };
            execute_traverse(&input_rows, &params, ctx)
        }

        LogicalOp::Filter { input, predicate } => {
            let rows = execute_op(input, ctx)?;
            let corr = ctx.correlated_row.clone();
            if expr_contains_pattern_predicate(predicate) {
                // Storage-aware path: pattern predicates need edge lookups.
                let mut result = Vec::new();
                for row in rows {
                    let effective_row = if let Some(ref outer) = corr {
                        let mut merged = outer.clone();
                        merged.extend(row.iter().map(|(k, v)| (k.clone(), v.clone())));
                        merged
                    } else {
                        row.clone()
                    };
                    let val = eval_predicate_with_storage(predicate, &effective_row, ctx)?;
                    if is_truthy(&val) {
                        result.push(row);
                    }
                }
                Ok(result)
            } else {
                Ok(rows
                    .into_iter()
                    .filter(|row| {
                        if let Some(ref outer) = corr {
                            // Correlated OPTIONAL MATCH: merge outer-scope variables
                            // so predicates like `c.age > a.age` can resolve `a`.
                            // Current row takes precedence over outer scope.
                            let mut merged = outer.clone();
                            merged.extend(row.iter().map(|(k, v)| (k.clone(), v.clone())));
                            is_truthy(&eval_expr(predicate, &merged))
                        } else {
                            is_truthy(&eval_expr(predicate, row))
                        }
                    })
                    .collect())
            }
        }

        LogicalOp::Project {
            input,
            items,
            distinct,
        } => {
            let rows = execute_op(input, ctx)?;

            // R-HYB1 guard: `text_score()` relies on `__text_score__` populated
            // by an upstream `TextFilter`. If a projection references `text_score`
            // but TextFilter never ran (missing FT index, or no paired
            // `text_match(...)` in WHERE), we must fail with a clear error rather
            // than silently returning 0.0. See ADR-020 and regression test
            // `text_score_without_text_match_errors`.
            let score_reqs: crate::executor::eval::ScoreRequirements = items
                .iter()
                .map(|it| crate::executor::eval::expr_score_requirements(&it.expr))
                .fold(Default::default(), |mut acc, r| {
                    acc.needs_text_score |= r.needs_text_score;
                    acc.needs_hybrid_score |= r.needs_hybrid_score;
                    acc.needs_rrf_score |= r.needs_rrf_score;
                    acc.needs_doc_score |= r.needs_doc_score;
                    acc
                });
            if let Some(first) = rows.first() {
                let has_text = first.contains_key("__text_score__");
                let has_vec = first.contains_key("__vector_score__");
                let has_rrf = first.contains_key("__rrf_score__");
                if score_reqs.needs_text_score && !has_text {
                    return Err(ExecutionError::Unsupported(
                        "text_score() requires a paired text_match(...) predicate in WHERE \
                         against a full-text-indexed field; none found in the plan"
                            .to_string(),
                    ));
                }
                if score_reqs.needs_hybrid_score && !has_text && !has_vec {
                    return Err(ExecutionError::Unsupported(
                        "hybrid_score() requires at least one of text_match(...) or \
                         vector_distance(...)/vector_similarity(...) in WHERE against \
                         the same node; none found in the plan"
                            .to_string(),
                    ));
                }
                if score_reqs.needs_rrf_score && !has_rrf {
                    return Err(ExecutionError::Unsupported(
                        "rrf_score() requires a RankFuse upstream operator to populate \
                         ranks — this typically means the planner failed to detect the \
                         rrf_score call-site; please file a bug with the query"
                            .to_string(),
                    ));
                }
                let has_doc = first.contains_key("__doc_score__");
                if score_reqs.needs_doc_score && !has_doc {
                    return Err(ExecutionError::Unsupported(
                        "doc_score() requires a DocScore upstream operator — this typically \
                         means the planner failed to detect the doc_score call-site; please \
                         file a bug with the query"
                            .to_string(),
                    ));
                }
            }

            // For-loop (not `.map`) so projection items containing pattern
            // comprehensions / EXISTS can evaluate through the storage-aware,
            // `&mut ctx`-borrowing path and propagate errors with `?`.
            let mut result: Vec<Row> = Vec::with_capacity(rows.len());
            for row in rows {
                let mut out = Row::new();
                for item in items {
                    if item.expr == Expr::Star {
                        // Star: copy all columns
                        out.extend(row.clone());
                    } else {
                        let val = if expr_contains_pattern_predicate(&item.expr) {
                            eval_predicate_with_storage(&item.expr, &row, ctx)?
                        } else {
                            eval_expr(&item.expr, &row)
                        };
                        let key = item
                            .alias
                            .clone()
                            .unwrap_or_else(|| expr_display_name(&item.expr));
                        out.insert(key.clone(), val);

                        // Variable passthrough: when a projection item is
                        // bare `Variable(x)` (and is left unrenamed, OR is
                        // aliased — in which case we re-bind the alias's
                        // property columns), also propagate every `x.prop`
                        // and `x.__*__` auxiliary column from the input
                        // row. Without this, `MATCH (a) WITH a RETURN
                        // a.prop` would lose all property bindings at the
                        // WITH barrier and `a.prop` would resolve to NULL.
                        if let Expr::Variable(var_name) = &item.expr {
                            let prefix = format!("{var_name}.");
                            for (col, value) in &row {
                                if let Some(suffix) = col.strip_prefix(&prefix) {
                                    out.insert(format!("{key}.{suffix}"), value.clone());
                                }
                            }
                        }
                    }
                }
                result.push(out);
            }

            if *distinct {
                // Full dedup — not just consecutive. O(n²) but correct for
                // all Value types including Float (which lacks Hash).
                let mut seen: Vec<Row> = Vec::new();
                result.retain(|row| {
                    if seen.iter().any(|s| s == row) {
                        false
                    } else {
                        seen.push(row.clone());
                        true
                    }
                });
            }

            Ok(result)
        }

        LogicalOp::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            let rows = execute_op(input, ctx)?;
            execute_aggregate(&rows, group_by, aggregates, &ctx.params)
        }

        LogicalOp::Sort { input, items } => {
            let mut rows = execute_op(input, ctx)?;

            // Same R-HYB1 guard as Project: an `ORDER BY text_score(...)` (bare
            // or inside arithmetic) without an upstream TextFilter would sort
            // by silent zeros. Error instead.
            let score_reqs: crate::executor::eval::ScoreRequirements = items
                .iter()
                .map(|it| crate::executor::eval::expr_score_requirements(&it.expr))
                .fold(Default::default(), |mut acc, r| {
                    acc.needs_text_score |= r.needs_text_score;
                    acc.needs_hybrid_score |= r.needs_hybrid_score;
                    acc.needs_rrf_score |= r.needs_rrf_score;
                    acc.needs_doc_score |= r.needs_doc_score;
                    acc
                });
            if let Some(first) = rows.first() {
                let has_text = first.contains_key("__text_score__");
                let has_vec = first.contains_key("__vector_score__");
                let has_rrf = first.contains_key("__rrf_score__");
                if score_reqs.needs_text_score && !has_text {
                    return Err(ExecutionError::Unsupported(
                        "text_score() requires a paired text_match(...) predicate in WHERE \
                         against a full-text-indexed field; none found in the plan"
                            .to_string(),
                    ));
                }
                if score_reqs.needs_hybrid_score && !has_text && !has_vec {
                    return Err(ExecutionError::Unsupported(
                        "hybrid_score() requires at least one of text_match(...) or \
                         vector_distance(...)/vector_similarity(...) in WHERE against \
                         the same node; none found in the plan"
                            .to_string(),
                    ));
                }
                if score_reqs.needs_rrf_score && !has_rrf {
                    return Err(ExecutionError::Unsupported(
                        "rrf_score() requires a RankFuse upstream operator to populate \
                         ranks — this typically means the planner failed to detect the \
                         rrf_score call-site; please file a bug with the query"
                            .to_string(),
                    ));
                }
                let has_doc = first.contains_key("__doc_score__");
                if score_reqs.needs_doc_score && !has_doc {
                    return Err(ExecutionError::Unsupported(
                        "doc_score() requires a DocScore upstream operator — this typically \
                         means the planner failed to detect the doc_score call-site; please \
                         file a bug with the query"
                            .to_string(),
                    ));
                }
            }

            rows.sort_by(|a, b| {
                for item in items {
                    let va = eval_expr(&item.expr, a);
                    let vb = eval_expr(&item.expr, b);
                    let cmp = compare_values(&va, &vb);
                    let cmp = if item.ascending { cmp } else { cmp.reverse() };
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                std::cmp::Ordering::Equal
            });
            Ok(rows)
        }

        LogicalOp::Limit { input, count } => {
            let rows = execute_op(input, ctx)?;
            let n = eval_expr(count, &Row::new());
            if let Value::Int(limit) = n {
                Ok(rows.into_iter().take(limit.max(0) as usize).collect())
            } else {
                Ok(rows)
            }
        }

        LogicalOp::Skip { input, count } => {
            let rows = execute_op(input, ctx)?;
            let n = eval_expr(count, &Row::new());
            if let Value::Int(skip) = n {
                Ok(rows.into_iter().skip(skip.max(0) as usize).collect())
            } else {
                Ok(rows)
            }
        }

        LogicalOp::CartesianProduct { left, right } => {
            let left_rows = execute_op(left, ctx)?;

            // G072: MERGE (src)-[r:TYPE]->(tgt) in a CartesianProduct context requires
            // correlated execution so the Merge can access the bound src/tgt variables.
            // Without this, execute_merge gets no source/target IDs and fails when it
            // tries to create the edge from a non-NodeScan pattern.
            //
            // The same correlated path is needed when the right side is a
            // NodeScan whose inline property filter references a variable
            // bound by the LEFT (e.g. `UNWIND ... AS e MATCH (a {p: e.x})`).
            // Evaluating that scan once globally cannot resolve `e.x`; it
            // must run per-left-row with `e` in scope. The detector keys on
            // a filter variable not bound within the right subtree, so a
            // genuinely uncorrelated cross product keeps the fast global path.
            if is_relationship_merge(right) || right_has_correlated_filter(right) {
                let prev_corr = ctx.correlated_row.take();
                let mut result = Vec::new();
                for lr in &left_rows {
                    ctx.correlated_row = Some(lr.clone());
                    let rr = execute_op(right, ctx)?;
                    for r in rr {
                        let mut merged = lr.clone();
                        merged.extend(r);
                        result.push(merged);
                    }
                }
                ctx.correlated_row = prev_corr;
                return Ok(result);
            }

            let right_rows = execute_op(right, ctx)?;
            let mut result = Vec::with_capacity(left_rows.len() * right_rows.len());
            for lr in &left_rows {
                for rr in &right_rows {
                    let mut merged = lr.clone();
                    merged.extend(rr.clone());
                    result.push(merged);
                }
            }
            Ok(result)
        }

        // UNION / UNION ALL: run each branch, concatenate the rows in branch
        // order. Plain UNION (`all == false`) de-duplicates the combined set,
        // preserving first-seen order; UNION ALL keeps every row.
        LogicalOp::Union { inputs, all } => {
            let mut result: Vec<Row> = Vec::new();
            for branch in inputs {
                let rows = execute_op(branch, ctx)?;
                if *all {
                    result.extend(rows);
                } else {
                    for row in rows {
                        if !result.contains(&row) {
                            result.push(row);
                        }
                    }
                }
            }
            Ok(result)
        }

        LogicalOp::VectorFilter {
            input,
            vector_expr,
            query_vector,
            function,
            less_than,
            threshold,
            decay_field,
            push_down: _,
        } => {
            let rows = execute_op(input, ctx)?;
            let mode = ctx.vector_consistency;

            let score_params = VectorScoreParams {
                function,
                less_than: *less_than,
                threshold: *threshold,
                decay_field: decay_field.as_ref(),
            };

            // Try HNSW-accelerated path when vector index registry is available.
            let result = if let Some(hnsw_result) =
                try_hnsw_vector_filter(&rows, vector_expr, query_vector, &score_params, ctx)?
            {
                hnsw_result
            } else {
                // Fallback to brute-force distance computation per row.
                execute_vector_filter(&rows, vector_expr, query_vector, &score_params)?
            };
            // In snapshot/exact mode, apply MVCC visibility post-filter.
            // For brute-force path, rows are already MVCC-consistent from
            // upstream operators (NodeScan reads via mvcc_get). This check
            // is a safety net and prepares for HNSW index integration where
            // candidates may not be MVCC-filtered.
            let needs_mvcc_filter =
                mode != VectorConsistencyMode::Current && ctx.mvcc_snapshot.is_some();
            if needs_mvcc_filter {
                // SAFETY: checked is_some() in condition above
                #[allow(clippy::expect_used)]
                let snap = ctx.mvcc_snapshot.as_ref().expect("mvcc_snapshot is_some");
                let mut stats = VectorMvccStats {
                    candidates_fetched: result.len(),
                    overfetch_factor: ctx.vector_overfetch_factor,
                    ..Default::default()
                };
                let mut visible = Vec::with_capacity(result.len());
                for row in &result {
                    // Extract node ID from the row to verify MVCC visibility.
                    // Check if the node key is visible at the snapshot timestamp.
                    if let Some(id_val) = row.get("_node_id") {
                        if let Some(id) = id_val.as_int() {
                            use coordinode_modality::NodeStore as _;
                            let nodes = coordinode_modality::LocalNodeStore;
                            match nodes.get_at_seqno(
                                &ctx.txn,
                                ctx.shard_id,
                                coordinode_core::graph::node::NodeId::from_raw(id as u64),
                                *snap,
                            ) {
                                Ok(Some(_)) => {
                                    visible.push(row.clone());
                                    stats.candidates_visible += 1;
                                }
                                _ => {
                                    stats.candidates_filtered += 1;
                                }
                            }
                        } else {
                            // Non-integer ID — keep the row (edge case)
                            visible.push(row.clone());
                            stats.candidates_visible += 1;
                        }
                    } else {
                        // No _node_id in row — keep as-is (brute-force path
                        // already MVCC-filtered by upstream operators)
                        visible.push(row.clone());
                        stats.candidates_visible += 1;
                    }
                }
                // Emit stats as structured warning for client visibility.
                // Full PROFILE output will include these in plan tree.
                ctx.warnings.push(format!(
                    "vector_mvcc(mode={}, fetched={}, visible={}, filtered={}, \
                     expansion_rounds={}, overfetch={:.1})",
                    mode.as_str(),
                    stats.candidates_fetched,
                    stats.candidates_visible,
                    stats.candidates_filtered,
                    stats.expansion_rounds,
                    stats.overfetch_factor,
                ));
                ctx.vector_mvcc_stats = Some(stats);
                Ok(visible)
            } else {
                Ok(result)
            }
        }

        LogicalOp::VectorTopK {
            input,
            vector_expr,
            query_vector,
            function,
            k,
            distance_alias,
            hnsw_index,
            predicate,
        } => {
            let rows = execute_op(input, ctx)?;

            // Extract the index name from the planner annotation ("name, metric" → "name").
            // The annotation is set by `annotate_vector_top_k` during planning when an HNSW
            // index exists for (label, property). Using it allows the executor to skip
            // row-based label detection and resolve the index directly by name.
            let hnsw_index_name: Option<&str> = hnsw_index.as_deref().map(|s| {
                // Format is "name, metric" (e.g. "item_emb, cosine") — take name part.
                s.split(", ").next().unwrap_or(s)
            });

            // Try HNSW-accelerated top-K path.
            let result = if let Some(hnsw_result) = try_hnsw_vector_top_k(
                &rows,
                vector_expr,
                query_vector,
                function,
                *k,
                distance_alias.as_deref(),
                hnsw_index_name,
                predicate.as_ref(),
                ctx,
            )? {
                hnsw_result
            } else {
                // Fallback: brute-force distance computation per row, sort, take top-K.
                execute_vector_top_k_brute_force(
                    rows,
                    vector_expr,
                    query_vector,
                    function,
                    *k,
                    distance_alias.as_deref(),
                )?
            };
            Ok(result)
        }

        LogicalOp::TextFilter {
            input,
            text_expr,
            query_string,
            language,
        } => {
            let rows = execute_op(input, ctx)?;
            execute_text_filter(&rows, text_expr, query_string, language.as_deref(), ctx)
        }

        LogicalOp::EncryptedFilter {
            input,
            field_expr,
            token_expr,
        } => {
            let rows = execute_op(input, ctx)?;
            execute_encrypted_filter(&rows, field_expr, token_expr, ctx)
        }

        LogicalOp::Unwind {
            input,
            expr,
            variable,
        } => {
            let rows = execute_op(input, ctx)?;
            execute_unwind(&rows, expr, variable)
        }

        LogicalOp::LeftOuterJoin { left, right } => {
            let left_rows = execute_op(left, ctx)?;
            execute_left_outer_join(&left_rows, right, ctx)
        }

        LogicalOp::ShortestPath {
            input,
            source,
            target,
            edge_types,
            direction,
            max_depth,
            path_variable,
        } => {
            let rows = execute_op(input, ctx)?;
            let sp = ShortestPathParams {
                source,
                target,
                edge_types,
                direction: *direction,
                max_depth: *max_depth,
                path_variable,
            };
            execute_shortest_path(&rows, &sp, ctx)
        }

        LogicalOp::EdgeVectorSearch {
            input,
            vector_expr,
            query_vector,
            function,
            less_than,
            threshold,
            ..
        } => {
            let rows = execute_op(input, ctx)?;
            let edge_params = VectorScoreParams {
                function,
                less_than: *less_than,
                threshold: *threshold,
                decay_field: None,
            };
            execute_vector_filter(&rows, vector_expr, query_vector, &edge_params)
        }

        LogicalOp::ProcedureCall {
            procedure,
            args,
            yield_items,
        } => execute_procedure_call(procedure, args, yield_items, ctx),

        LogicalOp::AlterLabel { label, mode } => execute_alter_label(label, mode, ctx),

        LogicalOp::CreateTextIndex {
            name,
            label,
            fields,
            default_language,
            language_override,
        } => execute_create_text_index(
            name,
            label,
            fields,
            default_language.as_deref(),
            language_override.as_deref(),
            ctx,
        ),

        LogicalOp::DropTextIndex { name } => execute_drop_text_index(name, ctx),

        LogicalOp::CreateEncryptedIndex {
            name,
            label,
            property,
        } => execute_create_encrypted_index(name, label, property, ctx),

        LogicalOp::DropEncryptedIndex { name } => execute_drop_encrypted_index(name, ctx),

        LogicalOp::CreateIndex {
            name,
            label,
            property,
            unique,
            sparse,
            filter,
        } => execute_create_btree_index(
            name,
            label,
            property,
            *unique,
            *sparse,
            filter.as_ref(),
            ctx,
        ),

        LogicalOp::DropIndex { name } => execute_drop_btree_index(name, ctx),

        LogicalOp::CreateVectorIndex {
            name,
            label,
            property,
            m,
            ef_construction,
            metric,
            dimensions,
            quantization,
            online_during_build,
            ef_search,
            rerank_candidates,
        } => execute_create_vector_index(
            name,
            label,
            property,
            *m,
            *ef_construction,
            *metric,
            *dimensions,
            *quantization,
            *online_during_build,
            *ef_search,
            *rerank_candidates,
            ctx,
        ),

        LogicalOp::DropVectorIndex { name } => execute_drop_vector_index(name, ctx),

        LogicalOp::CreateEdgeType {
            name,
            temporal,
            properties,
        } => execute_create_edge_type(name, *temporal, properties, ctx),

        LogicalOp::CreateNodeType {
            name,
            temporal,
            properties,
        } => execute_create_node_type(name, *temporal, properties, ctx),

        LogicalOp::CreateTrigger { clause } => execute_create_trigger(clause, ctx),
        LogicalOp::DropTrigger { name } => execute_drop_trigger(name, ctx),
        LogicalOp::ShowTriggers => execute_show_triggers(ctx),
        LogicalOp::AlterTrigger { clause } => execute_alter_trigger(clause, ctx),

        LogicalOp::MaxSimTopK {
            input,
            doc_expr,
            query_expr,
            k,
            score_alias,
        } => execute_maxsim_top_k(input, doc_expr, query_expr, *k, score_alias.as_deref(), ctx),

        // Empty normally yields a single empty row (standalone CREATE source).
        // Inside a FOREACH body, the loop injects the per-iteration scope here
        // so the body's update operators see the outer bindings + loop variable.
        LogicalOp::Empty => Ok(vec![ctx.foreach_scope.clone().unwrap_or_default()]),

        LogicalOp::Foreach {
            input,
            variable,
            list,
            body,
        } => {
            let input_rows = execute_op(input, ctx)?;
            // Save any enclosing FOREACH scope; the input rows already carry it
            // (the body's Empty leaf consumed it above), so each iteration only
            // adds the loop variable on top of the current row.
            let prev_scope = ctx.foreach_scope.take();
            for row in &input_rows {
                let elems = match eval_expr(list, row) {
                    Value::Array(items) => items,
                    // FOREACH over NULL is a no-op (matches Cypher).
                    Value::Null => continue,
                    other => {
                        ctx.foreach_scope = prev_scope;
                        return Err(ExecutionError::Unsupported(format!(
                            "FOREACH expects a list, got {other:?}"
                        )));
                    }
                };
                for elem in elems {
                    let mut scoped = row.clone();
                    scoped.insert(variable.clone(), elem);
                    ctx.foreach_scope = Some(scoped);
                    // Body is run for its side effects; FOREACH is pass-through.
                    execute_op(body, ctx)?;
                }
            }
            ctx.foreach_scope = prev_scope;
            Ok(input_rows)
        }

        LogicalOp::CallSubquery {
            input,
            body,
            optional,
        } => {
            let input_rows = execute_op(input, ctx)?;
            let prev_scope = ctx.foreach_scope.take();
            let mut result = Vec::new();
            for row in &input_rows {
                // Inject the outer row at the body's Empty leaf so a leading
                // importing WITH can project the correlated variables. An
                // uncorrelated body (its own scan leaf) ignores this.
                ctx.foreach_scope = Some(row.clone());
                let sub_rows = execute_op(body, ctx)?;
                if sub_rows.is_empty() && *optional {
                    // OPTIONAL CALL: keep the outer row; subquery columns read
                    // as NULL since they are absent.
                    result.push(row.clone());
                } else {
                    for sr in sub_rows {
                        let mut merged = row.clone();
                        merged.extend(sr);
                        result.push(merged);
                    }
                }
            }
            ctx.foreach_scope = prev_scope;
            Ok(result)
        }

        LogicalOp::CreateNode {
            input,
            variable,
            labels,
            properties,
        } => {
            let input_rows = match input {
                Some(inp) => execute_op(inp, ctx)?,
                None => vec![Row::new()],
            };
            execute_create_node(&input_rows, variable.as_deref(), labels, properties, ctx)
        }

        LogicalOp::CreateEdge {
            input,
            source,
            target,
            edge_type,
            direction: _,
            variable,
            properties,
        } => {
            let input_rows = execute_op(input, ctx)?;
            execute_create_edge(
                &input_rows,
                source,
                target,
                edge_type,
                variable.as_deref(),
                properties,
                ctx,
            )
        }

        LogicalOp::Update {
            input,
            items,
            violation_mode,
        } => {
            let input_rows = execute_op(input, ctx)?;
            execute_update(&input_rows, items, violation_mode, ctx)
        }

        LogicalOp::RemoveOp { input, items } => {
            let input_rows = execute_op(input, ctx)?;
            execute_remove(&input_rows, items, ctx)
        }

        LogicalOp::Delete {
            input,
            variables,
            detach,
        } => {
            let input_rows = execute_op(input, ctx)?;
            execute_delete(&input_rows, variables, *detach, ctx)
        }

        LogicalOp::Merge {
            pattern,
            on_match,
            on_create,
            multi,
        } => execute_merge(pattern, on_match, on_create, *multi, ctx),

        LogicalOp::Upsert {
            pattern,
            on_match,
            on_create_patterns,
        } => execute_upsert(pattern, on_match, on_create_patterns, ctx),

        LogicalOp::DetachDocument {
            input,
            source_variable,
            property_path,
            target_variable,
            target_labels,
            edge_type,
            edge_direction,
            edge_variable: _,
            transfer,
        } => {
            let input_rows = execute_op(input, ctx)?;
            execute_detach_document(
                &input_rows,
                source_variable,
                property_path,
                target_variable,
                target_labels,
                edge_type,
                *edge_direction,
                transfer.as_ref(),
                ctx,
            )
        }

        LogicalOp::AttachDocument {
            input,
            source_variable,
            target_variable,
            edge_type,
            edge_direction,
            target_property_path,
            transfer,
            on_conflict_replace,
            on_remaining_fail,
        } => {
            let input_rows = execute_op(input, ctx)?;
            execute_attach_document(
                &input_rows,
                source_variable,
                target_variable,
                edge_type,
                *edge_direction,
                target_property_path,
                transfer.as_ref(),
                *on_conflict_replace,
                *on_remaining_fail,
                ctx,
            )
        }

        LogicalOp::MergeNodes {
            input,
            source_a,
            source_b,
            target,
            conflict,
            transfer_edges,
            duplicate,
            transfer_edge_properties,
        } => {
            let input_rows = execute_op(input, ctx)?;
            execute_merge_nodes(
                &input_rows,
                source_a,
                source_b,
                target,
                conflict,
                transfer_edges.as_ref(),
                duplicate,
                *transfer_edge_properties,
                ctx,
            )
        }

        LogicalOp::CloneNode {
            input,
            source,
            target,
            with_edges,
            with_properties,
            set_items,
            as_of,
        } => {
            let input_rows = execute_op(input, ctx)?;
            execute_clone_node(
                &input_rows,
                source,
                target,
                *with_edges,
                *with_properties,
                set_items,
                as_of.as_ref(),
                ctx,
            )
        }

        LogicalOp::RedirectEdges {
            input,
            source,
            target,
            edge_types,
            direction,
        } => {
            let input_rows = execute_op(input, ctx)?;
            execute_redirect_edges(
                &input_rows,
                source,
                target,
                edge_types.as_deref(),
                *direction,
                ctx,
            )
        }

        LogicalOp::RankFuse {
            input,
            methods,
            query_vector,
            query_text,
            shard_overfetch_cap,
            fusion,
        } => {
            let rows = execute_op(input, ctx)?;
            execute_rank_fuse(
                rows,
                methods,
                query_vector.as_ref(),
                query_text.as_ref(),
                *shard_overfetch_cap,
                fusion,
                ctx,
            )
        }

        LogicalOp::DocScore {
            input,
            doc_variable,
            query_vector,
            alpha,
            beta,
            gamma,
        } => {
            let rows = execute_op(input, ctx)?;
            execute_doc_score(rows, doc_variable, query_vector, alpha, beta, gamma, ctx)
        }
    }
}

/// Scan nodes from storage, optionally filtering by label.
fn execute_node_scan(
    variable: &str,
    labels: &[String],
    property_filters: &[(String, Expr)],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let mut results = Vec::new();

    // Scan all nodes in the shard using prefix scan. The Layer-4 store owns
    // the node-key shape; the query layer just runs the tracked prefix scan.
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let prefix_bytes = LocalNodeStore.shard_scan_prefix(ctx.shard_id);
    ctx.sync_txn_state();
    let scan_results = LocalNodeStore.prefix_scan_tracked(&mut ctx.txn, &prefix_bytes)?;

    for (key_bytes, value_bytes) in &scan_results {
        let record = NodeRecord::from_msgpack(value_bytes).map_err(|e| {
            ExecutionError::Serialization(format!("node deserialization error: {e}"))
        })?;

        // Label filter: if labels specified, node must match one of them
        if !labels.is_empty() && !labels.iter().any(|l| record.has_label(l)) {
            continue;
        }

        // Extract node ID from key
        let node_id = decode_node_id_from_key(key_bytes);

        // Build row with node variable and its properties
        let mut row = Row::new();
        row.insert(variable.to_string(), Value::Int(node_id as i64));

        // Add interned properties as variable.property columns
        for (field_id, value) in &record.props {
            if let Some(field_name) = ctx.interner.resolve(*field_id) {
                let col_name = format!("{variable}.{field_name}");
                row.insert(col_name, value.clone());
            }
        }
        // Add extra overflow properties (VALIDATED mode undeclared props)
        if let Some(extra) = &record.extra {
            for (name, value) in extra {
                let col_name = format!("{variable}.{name}");
                row.insert(col_name, value.clone());
            }
        }

        // Add label info
        let primary_label = record.primary_label().to_string();
        row.insert(
            format!("{variable}.__label__"),
            Value::String(primary_label.clone()),
        );

        // Inject COMPUTED property values from schema (R082).
        inject_computed_properties(&mut row, variable, &primary_label, ctx);

        // Apply inline property filters from pattern. When this scan runs
        // inside a correlated join (e.g. `UNWIND ... AS e MATCH (a {p: e.x})`)
        // the filter value can reference outer-bound variables; evaluate it
        // against the correlated row extended with this node's bindings so
        // `e.x` resolves. An inline `{p: e.x}` is semantically identical to
        // `WHERE a.p = e.x`, which already works through the post-join Filter
        // path — this closes the asymmetry.
        let mut matches = true;
        for (prop_name, filter_expr) in property_filters {
            let actual = row
                .get(&format!("{variable}.{prop_name}"))
                .cloned()
                .unwrap_or(Value::Null);
            let expected = match &ctx.correlated_row {
                Some(corr) => {
                    let mut eval_row = corr.clone();
                    eval_row.extend(row.clone());
                    eval_expr(filter_expr, &eval_row)
                }
                None => eval_expr(filter_expr, &row),
            };
            if actual != expected {
                matches = false;
                break;
            }
        }

        if matches {
            results.push(row);
        }
    }

    Ok(results)
}

/// Execute a B-tree index point-lookup (`IndexScan` logical operator).
///
/// Evaluates `value_expr` against an empty row to obtain the lookup value,
/// calls `index_scan_exact` to retrieve matching node IDs from the index,
/// then fetches each node record and builds result rows in the same format
/// as `execute_node_scan` (variable → node_id, variable.prop → value).
fn execute_btree_index_scan(
    variable: &str,
    label: &str,
    index_name: &str,
    _property: &str,
    value_expr: &Expr,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // R172b safe-reject: B-tree index lookup returns node ids, then reads
    // node records via 16-byte `encode_node_key` to project. Temporal
    // records live at the 25-byte per-version key — would silently return
    // None and the row is dropped. Version-aware index entries
    // (`(node_id, valid_from)` point identity) land in R172d.
    if let Ok(Some(s)) = ctx.load_current_label_schema(label) {
        if s.temporal {
            return Err(ExecutionError::Unsupported(format!(
                "B-tree index scan on temporal label '{label}' is not yet \
                 supported (lands in R172d — version-aware index entries)."
            )));
        }
    }

    // Evaluate the lookup value. A correlated key (e.g. `WHERE a.pid = e.s`
    // driven per outer row) resolves against `correlated_row`; a literal /
    // parameter key ignores the row, so the empty-row fallback is equivalent.
    let lookup_val = match &ctx.correlated_row {
        Some(corr) => eval_expr(value_expr, corr),
        None => eval_expr(value_expr, &Row::new()),
    };

    // Use index_scan_exact to find node IDs matching the lookup value.
    let node_ids = crate::index::ops::index_scan_exact(ctx.engine, index_name, &lookup_val)
        .map_err(ExecutionError::Storage)?;

    let mut results = Vec::with_capacity(node_ids.len());

    use coordinode_modality::NodeStore as _;
    let nodes = coordinode_modality::LocalNodeStore;

    // Materialize every matching node in one batched multi_get (single version
    // snapshot + batched bloom/SST traversal) rather than a per-id lookup loop.
    let ids: Vec<NodeId> = node_ids.iter().map(|&raw| NodeId::from_raw(raw)).collect();
    let records = nodes.get_many(&ctx.txn, ctx.shard_id, &ids)?;

    for (raw_id, record_opt) in node_ids.into_iter().zip(records) {
        let Some(record) = record_opt else {
            // Node was deleted since the index entry was created — skip stale entry.
            continue;
        };

        // Verify label matches (guards against stale index entries for deleted/relabeled nodes).
        if !record.has_label(label) {
            continue;
        }

        let mut row = Row::new();
        row.insert(variable.to_string(), Value::Int(raw_id as i64));

        // Add properties under `variable.prop` columns.
        for (field_id, value) in &record.props {
            if let Some(field_name) = ctx.interner.resolve(*field_id) {
                row.insert(format!("{variable}.{field_name}"), value.clone());
            }
        }
        if let Some(extra) = &record.extra {
            for (name, value) in extra {
                row.insert(format!("{variable}.{name}"), value.clone());
            }
        }

        let primary_label = record.primary_label().to_string();
        row.insert(
            format!("{variable}.__label__"),
            Value::String(primary_label.clone()),
        );

        inject_computed_properties(&mut row, variable, &primary_label, ctx);

        results.push(row);
    }

    Ok(results)
}

/// Execute the `HnswScan` index access path: ask the HNSW index for the
/// top-k candidates, then point-fetch ONLY those k node records into
/// rows. O(k) storage reads — the whole point of the access path versus
/// the scan-then-rank pipeline that materialises every node of the
/// label before ranking.
///
/// Row shape matches `execute_node_scan` / `execute_btree_index_scan`
/// (binding -> node_id, binding.prop -> value, binding.__label__,
/// computed properties injected) so every downstream operator works
/// unchanged. When `distance_alias` is set, the column carries the
/// SAME scalar the brute-force path would have computed for the
/// original ORDER BY function — recomputed exactly per candidate (k is
/// tiny) instead of trusting the index's internal score, whose units
/// differ per metric (e.g. squared L2 inside the engine vs sqrt L2
/// from the Cypher scalar).
#[allow(clippy::too_many_arguments)]
fn execute_hnsw_scan(
    label: &str,
    property: &str,
    binding: &str,
    query_vector: &Expr,
    k: usize,
    function: &str,
    distance_alias: Option<&str>,
    index_name: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let Some(registry) = ctx.vector_index_registry else {
        return Err(ExecutionError::Unsupported(format!(
            "HnswScan({index_name}) requires vector_index_registry in ExecutionContext"
        )));
    };
    // Honour the online-during-build policy exactly like the
    // scan-then-rank path does.
    gate_vector_index_read(ctx.engine, registry, label, property)?;

    let qv_val = eval_expr(query_vector, &Row::new());
    let Some(qv) = coerce_value_to_vec(&qv_val) else {
        return Err(ExecutionError::Unsupported(format!(
            "HnswScan({index_name}): query vector did not evaluate to a vector"
        )));
    };

    let Some(hits) = registry.search(label, property, &qv, k) else {
        // Index disappeared between planning and execution (concurrent
        // DROP). Empty result keeps the read path total; the planner
        // will not pick HnswScan on the next statement.
        return Ok(Vec::new());
    };

    use coordinode_modality::NodeStore as _;
    let nodes = coordinode_modality::LocalNodeStore;
    let mut results = Vec::with_capacity(hits.len());

    // Hydrate the k-NN hit set in one batched multi_get; the input order is
    // preserved, so the rows stay in similarity order.
    let ids: Vec<NodeId> = hits.iter().map(|h| NodeId::from_raw(h.id)).collect();
    let records = nodes.get_many(&ctx.txn, ctx.shard_id, &ids)?;

    for (hit, record_opt) in hits.into_iter().zip(records) {
        let Some(record) = record_opt else {
            // Node deleted after the index entry was written — skip.
            continue;
        };
        if !record.has_label(label) {
            continue;
        }

        let mut row = Row::new();
        row.insert(binding.to_string(), Value::Int(hit.id as i64));
        for (field_id, value) in &record.props {
            if let Some(field_name) = ctx.interner.resolve(*field_id) {
                row.insert(format!("{binding}.{field_name}"), value.clone());
            }
        }
        if let Some(extra) = &record.extra {
            for (name, value) in extra {
                row.insert(format!("{binding}.{name}"), value.clone());
            }
        }
        let primary_label = record.primary_label().to_string();
        row.insert(
            format!("{binding}.__label__"),
            Value::String(primary_label.clone()),
        );
        inject_computed_properties(&mut row, binding, &primary_label, ctx);

        if let Some(alias) = distance_alias {
            let node_vec = row
                .get(&format!("{binding}.{property}"))
                .and_then(coerce_value_to_vec);
            let score = match node_vec {
                Some(nv) => match function {
                    "vector_distance" => {
                        coordinode_vector::metrics::euclidean_distance(&qv, &nv) as f64
                    }
                    "vector_similarity" => {
                        coordinode_vector::metrics::cosine_similarity(&qv, &nv) as f64
                    }
                    "vector_dot" => coordinode_vector::metrics::dot_product(&qv, &nv) as f64,
                    "vector_manhattan" => {
                        coordinode_vector::metrics::manhattan_distance(&qv, &nv) as f64
                    }
                    _ => hit.score as f64,
                },
                None => hit.score as f64,
            };
            row.insert(alias.to_string(), Value::Float(score));
        }
        results.push(row);
    }
    Ok(results)
}

/// Parameters for edge traversal.
struct TraverseParams<'a> {
    source: &'a str,
    edge_types: &'a [String],
    direction: Direction,
    target_variable: &'a str,
    target_labels: &'a [String],
    length: Option<LengthBound>,
    edge_variable: Option<&'a str>,
    target_filters: &'a [(String, Expr)],
    /// Inline edge property filters from pattern (e.g., `[r:TYPE {prop: val}]`).
    edge_filters: &'a [(String, Expr)],
    /// Parallel to `edge_types`: `true` if the type was declared TEMPORAL.
    /// Hoisted once per traversal so the inner loop never re-queries the
    /// schema partition.
    edge_temporal: &'a [bool],
    /// Optional pushed-down time-slice filter for temporal edges.
    temporal_filter: Option<&'a crate::planner::logical::TemporalFilter>,
    /// Named-path variable to bind the source-to-target route. When set, the
    /// traversal runs sequentially and records the predecessor chain.
    path_variable: Option<&'a str>,
}

/// Traverse edges from source nodes to target nodes.
///
/// Dispatches to single-hop or variable-length BFS based on `params.length`.
fn execute_traverse(
    input_rows: &[Row],
    params: &TraverseParams<'_>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // R172d (initial slice): traversal into a temporal target label
    // materialises EVERY version of the target node (prefix-scan over
    // `node:<shard>:<target_uid>:*`). The version's `valid_from` is
    // surfaced as `<target>.valid_from`, and each version emits its own
    // row so downstream RETURN / WHERE can filter by interval. Without
    // an `AS OF VALID_TIME` clause (G096) the read is "all versions" —
    // the same default as label-scoped MATCH on temporal labels. Once
    // R172d gains the AS OF clause + planner push-down, this scan will
    // be narrowed to the requested time slice.
    //
    // Behaviour is detected per target label at row-build time; the
    // dispatch happens in `build_target_rows`.

    if let Some(lb) = params.length {
        execute_varlen_traverse(input_rows, params, lb, ctx)
    } else {
        execute_single_hop_traverse(input_rows, params, ctx)
    }
}

/// Expand one hop from `src_id` in the given direction and edge types.
///
/// Returns ALL `(target_uid, edge_type_index)` pairs — no truncation.
/// The caller decides whether to process them sequentially or in parallel
/// based on `AdaptiveConfig::parallel_threshold`.
fn expand_one_hop(
    src_id: NodeId,
    edge_types: &[String],
    direction: Direction,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<(u64, usize)>, ExecutionError> {
    let mut neighbors = Vec::new();
    // Single buffer reused for all key constructions in this call.
    // Avoids N_edge_types × N_directions heap allocations per traversal step.
    let mut adj_key = Vec::with_capacity(64);

    use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
    for (et_idx, edge_type) in edge_types.iter().enumerate() {
        match direction {
            Direction::Outgoing | Direction::Both => {
                LocalEdgeStore.write_fwd_key(&mut adj_key, edge_type, src_id);
            }
            Direction::Incoming => {
                LocalEdgeStore.write_rev_key(&mut adj_key, edge_type, src_id);
            }
        }

        if let Some(posting_list) = LocalEdgeStore.posting_for_key(&ctx.txn, &adj_key)? {
            let fan_out = posting_list.len();
            // Reserve the whole fan-out up front so a high-degree node does not
            // repeatedly reallocate `neighbors` mid-expansion (super-node path).
            neighbors.reserve(fan_out);
            if ctx.adaptive.enabled && fan_out > ctx.adaptive.parallel_threshold {
                // Diagnostic only: a per-super-node log on the traversal hot path
                // must stay at debug so a near-global query does not pay
                // formatting + writer-lock cost (and flood the log) per hub.
                tracing::debug!(
                    node_id = src_id.as_raw(),
                    fan_out,
                    edge_type = edge_type.as_str(),
                    "super-node detected, parallel processing will be used"
                );
                // Record in feedback cache for future queries
                if let Some(ref cache) = ctx.feedback_cache {
                    cache.record(src_id.as_raw(), fan_out);
                }
            }

            for tgt_uid in posting_list.iter() {
                neighbors.push((tgt_uid, et_idx));
            }
        }

        if direction == Direction::Both {
            LocalEdgeStore.write_rev_key(&mut adj_key, edge_type, src_id);
            if let Some(posting_list) = LocalEdgeStore.posting_for_key(&ctx.txn, &adj_key)? {
                neighbors.reserve(posting_list.len());
                for tgt_uid in posting_list.iter() {
                    neighbors.push((tgt_uid, et_idx));
                }
            }
        }
    }

    Ok(neighbors)
}

/// Parameters for building a target node row.
struct TargetRowParams<'a> {
    input_row: &'a Row,
    target_uid: u64,
    edge_type: Option<&'a str>,
    /// Whether the edge type is declared TEMPORAL. When true, the row builder
    /// emits one row per `(src, tgt)` edgeprop version; when false it emits at
    /// most one row using the legacy single edgeprop entry.
    edge_is_temporal: bool,
}

/// Fetch a target node and build output rows. Returns `Vec` because temporal
/// edges fan out across versions: a single neighbor pair contributes one row
/// per stored `valid_from`. Non-temporal edges still emit 0 or 1 rows.
fn build_target_rows(
    trp: &TargetRowParams<'_>,
    params: &TraverseParams<'_>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let target_uid = trp.target_uid;
    let target_variable = params.target_variable;
    let target_labels = params.target_labels;
    let target_filters = params.target_filters;
    let edge_variable = params.edge_variable;
    let edge_type = trp.edge_type;

    let target_id = NodeId::from_raw(target_uid);

    // R172d: detect whether ANY of the target labels is temporal. If
    // so, every version of the target is materialised (prefix scan);
    // otherwise the legacy 16-byte point read is used. Detection runs
    // at row-build time — label schemas don't change within a single
    // query, so the lookup is cheap enough not to cache further.
    let target_is_temporal = target_labels.iter().any(|lbl| {
        ctx.load_current_label_schema(lbl)
            .ok()
            .flatten()
            .is_some_and(|s| s.temporal)
    });

    // Collect (record, optional valid_from) pairs to emit. Non-temporal
    // path produces one pair; temporal path produces one per version.
    let target_records: Vec<(NodeRecord, Option<i64>)> = if target_is_temporal {
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        let prefix = LocalNodeStore.version_prefix(ctx.shard_id, target_id);
        ctx.sync_txn_state();
        let scanned = LocalNodeStore.prefix_scan_tracked(&mut ctx.txn, &prefix)?;
        if scanned.is_empty() {
            return Ok(Vec::new());
        }
        let mut out: Vec<(NodeRecord, Option<i64>)> = Vec::with_capacity(scanned.len());
        for (key, bytes) in scanned {
            let Some((_, _, vf)) = coordinode_core::graph::node::decode_temporal_node_key(&key)
            else {
                continue;
            };
            let rec = NodeRecord::from_msgpack(&bytes).map_err(|e| {
                ExecutionError::Serialization(format!(
                    "target temporal node deserialization error: {e}"
                ))
            })?;
            out.push((rec, Some(vf)));
        }
        if out.is_empty() {
            return Ok(Vec::new());
        }
        out
    } else {
        let target_record = match ctx.mvcc_get_node(ctx.shard_id, target_id)? {
            Some(rec) => rec,
            None => return Ok(Vec::new()),
        };
        vec![(target_record, None)]
    };

    // For each version (or the single non-temporal record), build a
    // base out_row carrying the target's properties + version axes,
    // then apply the edge-property fan-out + filter pipeline below.
    let mut materialised_rows: Vec<Row> = Vec::with_capacity(target_records.len());
    for (target_record, valid_from_opt) in target_records {
        // Label filter
        if !target_labels.is_empty() && !target_labels.iter().any(|l| target_record.has_label(l)) {
            continue;
        }

        let mut out_row = trp.input_row.clone();
        out_row.insert(target_variable.to_string(), Value::Int(target_uid as i64));

        for (field_id, value) in &target_record.props {
            if let Some(field_name) = ctx.interner.resolve(*field_id) {
                let col_name = format!("{target_variable}.{field_name}");
                out_row.insert(col_name, value.clone());
            }
        }
        if let Some(extra) = &target_record.extra {
            for (name, value) in extra {
                let col_name = format!("{target_variable}.{name}");
                out_row.insert(col_name, value.clone());
            }
        }

        let target_label = target_record.primary_label().to_string();
        out_row.insert(
            format!("{target_variable}.__label__"),
            Value::String(target_label.clone()),
        );

        // R172d: re-surface valid_from from the key suffix so callers
        // always see a non-null binding even if the stored property map
        // happens to omit it (defensive — write path requires it).
        if let Some(vf) = valid_from_opt {
            out_row.insert(format!("{target_variable}.valid_from"), Value::Int(vf));
        }

        // Inject COMPUTED property values from schema (R082).
        inject_computed_properties(&mut out_row, target_variable, &target_label, ctx);

        materialised_rows.push(out_row);
    }

    if materialised_rows.is_empty() {
        return Ok(Vec::new());
    }

    // Edge variable bindings: type marker + per-version property fan-out
    // for temporal edges. For non-temporal types this resolves to a single
    // optional edgeprop blob. Cartesian-product across `materialised_rows`
    // so each target version pairs with each edge version.
    let edge_property_rows: Vec<Row> = if let (Some(ev), Some(et)) = (edge_variable, edge_type) {
        let source_id_raw = trp.input_row.get(params.source).and_then(|v| {
            if let Value::Int(id) = v {
                Some(*id as u64)
            } else {
                None
            }
        });
        if let Some(src_raw) = source_id_raw {
            let (ep_src, ep_tgt) = match params.direction {
                Direction::Outgoing | Direction::Both => (NodeId::from_raw(src_raw), target_id),
                Direction::Incoming => (target_id, NodeId::from_raw(src_raw)),
            };

            if trp.edge_is_temporal {
                // Layer-4 store owns the temporal edgeprop key shape + scan;
                // returns (valid_from, raw props bytes) so the query layer
                // projects fields (and keeps the props-decode contract).
                use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
                ctx.sync_txn_state();
                let upper_ms = params.temporal_filter.and_then(|tf| tf.upper_ms);
                let versions = LocalEdgeStore.scan_versions_raw_tracked(
                    &mut ctx.txn,
                    et,
                    ep_src,
                    ep_tgt,
                    upper_ms,
                )?;
                let mut acc: Vec<Row> =
                    Vec::with_capacity(materialised_rows.len() * versions.len());
                if !versions.is_empty() {
                    for base in &materialised_rows {
                        for (vf, ep_bytes) in &versions {
                            let mut version_row = base.clone();
                            version_row
                                .insert(format!("{ev}.__type__"), Value::String(et.to_string()));
                            version_row.insert(ev.to_string(), Value::String(et.to_string()));
                            version_row.insert(
                                format!("{ev}.__src__"),
                                Value::Int(ep_src.as_raw() as i64),
                            );
                            version_row.insert(
                                format!("{ev}.__tgt__"),
                                Value::Int(ep_tgt.as_raw() as i64),
                            );
                            version_row.insert(format!("{ev}.valid_from"), Value::Int(*vf));
                            if let Ok(prop_map) = decode_edge_props(ep_bytes) {
                                for (field_id, value) in prop_map {
                                    if let Some(field_name) = ctx.interner.resolve(field_id) {
                                        version_row.insert(format!("{ev}.{field_name}"), value);
                                    }
                                }
                            }
                            acc.push(version_row);
                        }
                    }
                }
                acc
            } else {
                let ep_props_opt = ctx.mvcc_get_edge_props(et, ep_src, ep_tgt)?;
                let mut acc: Vec<Row> = Vec::with_capacity(materialised_rows.len());
                for base in materialised_rows {
                    let mut row = base;
                    row.insert(format!("{ev}.__type__"), Value::String(et.to_string()));
                    row.insert(ev.to_string(), Value::String(et.to_string()));
                    row.insert(format!("{ev}.__src__"), Value::Int(ep_src.as_raw() as i64));
                    row.insert(format!("{ev}.__tgt__"), Value::Int(ep_tgt.as_raw() as i64));
                    if let Some(ref prop_map) = ep_props_opt {
                        for (field_id, value) in prop_map {
                            if let Some(field_name) = ctx.interner.resolve(*field_id) {
                                row.insert(format!("{ev}.{field_name}"), value.clone());
                            }
                        }
                    }
                    acc.push(row);
                }
                acc
            }
        } else {
            // No source id binding — emit each materialised row carrying
            // only the bare edge-type marker.
            materialised_rows
                .into_iter()
                .map(|mut row| {
                    row.insert(format!("{ev}.__type__"), Value::String(et.to_string()));
                    row.insert(ev.to_string(), Value::String(et.to_string()));
                    row
                })
                .collect()
        }
    } else {
        materialised_rows
    };

    // Apply target / edge inline filters to each candidate row. Filters drop
    // rows individually, so a temporal pair may keep some versions and shed
    // others.
    let mut kept: Vec<Row> = Vec::with_capacity(edge_property_rows.len());
    'each_row: for row in edge_property_rows {
        for (prop_name, filter_expr) in target_filters {
            let actual = row
                .get(&format!("{target_variable}.{prop_name}"))
                .cloned()
                .unwrap_or(Value::Null);
            let expected = eval_expr(filter_expr, &row);
            if actual != expected {
                continue 'each_row;
            }
        }
        if let Some(ev) = edge_variable {
            for (prop_name, filter_expr) in params.edge_filters {
                let actual = row
                    .get(&format!("{ev}.{prop_name}"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let expected = eval_expr(filter_expr, &row);
                if actual != expected {
                    continue 'each_row;
                }
            }
        }
        kept.push(row);
    }
    Ok(kept)
}

/// Collected OCC read-set keys from parallel processing, kept per partition
/// so the query layer never names a `Partition` variant — node and edge-prop
/// keys go in separate buffers and merge through the typed
/// `OccScope::extend_node_keys` / `extend_edge_prop_keys`.
#[derive(Default)]
struct OccReadKeys {
    node: Mutex<Vec<Vec<u8>>>,
    edge_prop: Mutex<Vec<Vec<u8>>>,
}

/// Read-only context extracted from `ExecutionContext` for parallel processing.
/// All fields are `Send + Sync`, enabling safe rayon parallelism.
struct ParallelCtx<'a> {
    engine: &'a StorageEngine,
    interner: &'a FieldInterner,
    shard_id: u16,
    mvcc_snapshot: Option<StorageSnapshot>,
    chunk_size: usize,
    /// OCC read-set keys collected during parallel processing. When `Some`,
    /// parallel workers push each read's raw key into the matching per-partition
    /// buffer; the caller merges them into `ExecutionContext::occ_scope` via the
    /// typed `OccScope::extend_*_keys` after the parallel block. `None` when the
    /// MVCC oracle is inactive (legacy mode).
    occ_read_keys: Option<OccReadKeys>,
}

/// Process target nodes in parallel using rayon when fan-out exceeds threshold.
///
/// Bypasses `ExecutionContext::mvcc_get` (which requires `&mut self`) by reading
/// directly from `StorageEngine` which is `Send + Sync`. Safe for read-only
/// traversal because:
/// - Target nodes are not being modified in the current transaction
/// - RYOW (read-your-own-writes) is irrelevant for reading OTHER nodes
///
/// OCC read-set tracking: when `pctx.occ_read_keys` is `Some`, all read
/// keys are collected into the `Mutex<Vec>` for the caller to merge into
/// `ExecutionContext::occ_scope` via `OccScope::extend` after the
/// parallel block completes.
fn process_targets_parallel(
    neighbors: &[(u64, u64, usize)],
    input_row: &Row,
    params: &TraverseParams<'_>,
    pctx: &ParallelCtx<'_>,
) -> Vec<Row> {
    let target_variable = params.target_variable;
    let target_labels = params.target_labels;
    let target_filters = params.target_filters;
    let edge_variable = params.edge_variable;
    let direction = params.direction;
    let source = params.source;

    neighbors
        .par_chunks(pctx.chunk_size.max(1))
        .flat_map_iter(|chunk| {
            chunk.iter().filter_map(|(src_uid, tgt_uid, et_idx)| {
                let target_id = NodeId::from_raw(*tgt_uid);

                // Stateless snapshot read through the Layer-4 store: the
                // snapshot seqno (a Copy value) is the shared unit across
                // rayon workers, so no &mut Transaction crosses threads. The
                // store owns key encoding and hands the key back for OCC.
                use coordinode_modality::{LocalNodeStore, NodeStore as _};
                let (target_key, bytes) = LocalNodeStore
                    .read_at_snapshot(pctx.engine, pctx.mvcc_snapshot, pctx.shard_id, target_id)
                    .ok()?;
                let bytes = bytes?;

                // Track the node key in the OCC read-set: each worker records
                // into the shared accumulator; merged into the statement's
                // read-set after the parallel section.
                if let Some(ref keys) = pctx.occ_read_keys {
                    if let Ok(mut guard) = keys.node.lock() {
                        guard.push(target_key);
                    }
                }

                let target_record = NodeRecord::from_msgpack(&bytes).ok()?;

                // Label filter
                if !target_labels.is_empty()
                    && !target_labels.iter().any(|l| target_record.has_label(l))
                {
                    return None;
                }

                let mut out_row = input_row.clone();
                // Update source in row for correct edge property lookup (G066 fix)
                out_row.insert(source.to_string(), Value::Int(*src_uid as i64));
                out_row.insert(target_variable.to_string(), Value::Int(*tgt_uid as i64));

                // Resolve property names from interner (read-only, thread-safe)
                for (field_id, value) in &target_record.props {
                    if let Some(field_name) = pctx.interner.resolve(*field_id) {
                        let col_name = format!("{target_variable}.{field_name}");
                        out_row.insert(col_name, value.clone());
                    }
                }
                if let Some(extra) = &target_record.extra {
                    for (name, value) in extra {
                        let col_name = format!("{target_variable}.{name}");
                        out_row.insert(col_name, value.clone());
                    }
                }

                let target_label = target_record.primary_label().to_string();
                out_row.insert(
                    format!("{target_variable}.__label__"),
                    Value::String(target_label.clone()),
                );

                // Inject COMPUTED property values (R082) in parallel path
                inject_computed_from_engine(
                    &mut out_row,
                    target_variable,
                    &target_label,
                    pctx.engine,
                );

                // Edge variable: type + edge properties
                let edge_type = params.edge_types.get(*et_idx).map(|s| s.as_str());
                if let Some(ev) = edge_variable {
                    if let Some(et) = edge_type {
                        out_row.insert(format!("{ev}.__type__"), Value::String(et.to_string()));
                        // G070: store relationship variable itself for count(r) support
                        out_row.insert(ev.to_string(), Value::String(et.to_string()));

                        {
                            let (ep_src, ep_tgt) = match direction {
                                Direction::Outgoing | Direction::Both => {
                                    (NodeId::from_raw(*src_uid), target_id)
                                }
                                Direction::Incoming => (target_id, NodeId::from_raw(*src_uid)),
                            };
                            // Hidden metadata for downstream SET / DELETE on r.
                            out_row.insert(
                                format!("{ev}.__src__"),
                                Value::Int(ep_src.as_raw() as i64),
                            );
                            out_row.insert(
                                format!("{ev}.__tgt__"),
                                Value::Int(ep_tgt.as_raw() as i64),
                            );
                            // Stateless snapshot read via the Layer-4 store
                            // (Variant A): shared snapshot seqno, no &mut txn
                            // across rayon threads; the store returns the key
                            // for the worker's OCC accumulator.
                            use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
                            let (ep_key, ep_bytes) = LocalEdgeStore
                                .edgeprop_at_snapshot(
                                    pctx.engine,
                                    pctx.mvcc_snapshot,
                                    et,
                                    ep_src,
                                    ep_tgt,
                                )
                                .unwrap_or((Vec::new(), None));
                            // Track the EdgeProp key in the OCC read-set.
                            if let Some(ref keys) = pctx.occ_read_keys {
                                if let Ok(mut guard) = keys.edge_prop.lock() {
                                    guard.push(ep_key);
                                }
                            }
                            if let Some(ep_bytes) = ep_bytes {
                                if let Ok(prop_map) = decode_edge_props(&ep_bytes) {
                                    for (field_id, value) in prop_map {
                                        if let Some(field_name) = pctx.interner.resolve(field_id) {
                                            out_row.insert(format!("{ev}.{field_name}"), value);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Apply target inline property filters
                for (prop_name, filter_expr) in target_filters {
                    let actual = out_row
                        .get(&format!("{target_variable}.{prop_name}"))
                        .cloned()
                        .unwrap_or(Value::Null);
                    let expected = eval_expr(filter_expr, &out_row);
                    if actual != expected {
                        return None;
                    }
                }

                // Apply inline edge property filters
                if let Some(ev) = edge_variable {
                    for (prop_name, filter_expr) in params.edge_filters {
                        let actual = out_row
                            .get(&format!("{ev}.{prop_name}"))
                            .cloned()
                            .unwrap_or(Value::Null);
                        let expected = eval_expr(filter_expr, &out_row);
                        if actual != expected {
                            return None;
                        }
                    }
                }

                Some(out_row)
            })
        })
        .collect()
}

/// Single-hop traversal: the original behavior for patterns without `*`.
fn execute_single_hop_traverse(
    input_rows: &[Row],
    params: &TraverseParams<'_>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let mut results = Vec::new();
    let use_parallel = ctx.adaptive.enabled && ctx.adaptive.parallel_threshold > 0;

    for row in input_rows {
        let source_id = match row.get(params.source) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => continue,
        };

        let neighbors = expand_one_hop(source_id, params.edge_types, params.direction, ctx)?;

        // Parallel path doesn't yet support temporal version fan-out (it bypasses
        // ExecutionContext and prefix scans), so any temporal edge type in the
        // traversal forces the sequential path. Non-temporal queries keep the
        // parallel optimization.
        let has_temporal = params.edge_temporal.iter().any(|t| *t);
        // R172d: temporal target labels need per-version prefix scans
        // which the parallel target-materialisation path doesn't speak
        // yet. Force sequential when any target label is temporal —
        // mirrors the same gate as temporal edges above.
        let target_has_temporal = params.target_labels.iter().any(|lbl| {
            ctx.load_current_label_schema(lbl)
                .ok()
                .flatten()
                .is_some_and(|s| s.temporal)
        });

        // Switch to parallel when fan-out exceeds threshold. A requested path
        // projection forces the sequential path so the route is bound exactly.
        if use_parallel
            && !has_temporal
            && !target_has_temporal
            && params.path_variable.is_none()
            && neighbors.len() >= ctx.adaptive.parallel_threshold
        {
            ctx.warnings.push(format!(
                "adaptive: parallel processing activated for node {} ({} edges, threshold {})",
                source_id.as_raw(),
                neighbors.len(),
                ctx.adaptive.parallel_threshold,
            ));
            let pctx = ParallelCtx {
                engine: ctx.engine,
                interner: ctx.interner,
                shard_id: ctx.shard_id,
                mvcc_snapshot: ctx.mvcc_snapshot,
                chunk_size: ctx.adaptive.parallel_chunk_size,
                occ_read_keys: if ctx.mvcc_oracle.is_some() {
                    Some(OccReadKeys::default())
                } else {
                    None
                },
            };
            let src_raw = source_id.as_raw();
            let with_src: Vec<(u64, u64, usize)> = neighbors
                .iter()
                .map(|&(tgt, et_idx)| (src_raw, tgt, et_idx))
                .collect();
            let parallel_rows = process_targets_parallel(&with_src, row, params, &pctx);
            // Merge OCC read keys from parallel workers into the Layer-3 scope
            // via the typed per-partition extends.
            if let Some(ref keys) = pctx.occ_read_keys {
                if let Some(scope) = ctx.ensure_occ_scope() {
                    if let Ok(mut node) = keys.node.lock() {
                        scope.extend_node_keys(node.drain(..));
                    }
                    if let Ok(mut ep) = keys.edge_prop.lock() {
                        scope.extend_edge_prop_keys(ep.drain(..));
                    }
                }
            }
            results.extend(parallel_rows);
        } else {
            // Sequential path for normal fan-out
            for (target_uid, et_idx) in neighbors {
                let edge_type = params.edge_types.get(et_idx).map(|s| s.as_str());
                let edge_is_temporal = params.edge_temporal.get(et_idx).copied().unwrap_or(false);
                let trp = TargetRowParams {
                    input_row: row,
                    target_uid,
                    edge_type,
                    edge_is_temporal,
                };
                let mut target_rows = build_target_rows(&trp, params, ctx)?;
                if let Some(pv) = params.path_variable {
                    // One-hop named path: source -> target via this edge.
                    let path = Value::Path(coordinode_core::graph::types::PathValue {
                        nodes: vec![source_id.as_raw(), target_uid],
                        rels: vec![coordinode_core::graph::types::PathRel {
                            edge_type: edge_type.unwrap_or_default().to_string(),
                            source: source_id.as_raw(),
                            target: target_uid,
                        }],
                    });
                    for r in &mut target_rows {
                        r.insert(pv.to_string(), path.clone());
                    }
                }
                results.extend(target_rows);
            }
        }
    }

    Ok(results)
}

/// Reconstruct the route from `source` to `target` as a path value, using the
/// BFS predecessor map (`pred[n] = Some((prev, edge_type_idx))`; the source's
/// entry is `None`). The route is the shortest one the level-synchronous BFS
/// discovered to `target`. Returns a zero-length path when `source == target`.
fn reconstruct_bfs_path(
    source: u64,
    target: u64,
    pred: &rustc_hash::FxHashMap<u64, Option<(u64, usize)>>,
    edge_types: &[String],
) -> Value {
    let mut back: Vec<(u64, usize)> = Vec::new();
    let mut cur = target;
    while cur != source {
        match pred.get(&cur).copied().flatten() {
            Some((prev, et_idx)) => {
                back.push((cur, et_idx));
                cur = prev;
            }
            // Target unreachable from source in `pred` (should not happen for a
            // node the BFS emitted); fall back to a lone-node path.
            None => break,
        }
    }
    back.reverse();

    let mut nodes = Vec::with_capacity(back.len() + 1);
    let mut rels = Vec::with_capacity(back.len());
    nodes.push(source);
    let mut prev = source;
    for (node, et_idx) in back {
        let edge_type = edge_types.get(et_idx).cloned().unwrap_or_default();
        rels.push(coordinode_core::graph::types::PathRel {
            edge_type,
            source: prev,
            target: node,
        });
        nodes.push(node);
        prev = node;
    }
    Value::Path(coordinode_core::graph::types::PathValue { nodes, rels })
}

/// Expand a whole BFS frontier by one hop, returning the `(src, tgt,
/// edge_type_idx)` triples in frontier-then-adjacency order.
///
/// This is the level-synchronous scatter-gather seam. The single-shard engine
/// expands every source locally here. A distributed engine keeps the BFS
/// coordinator identical and replaces this one call with a per-shard scatter:
/// route each source to the shard owning its out-edges, run this same local
/// expansion there, and return the slice. Keeping all expansion behind a
/// single frontier-level step is what makes that swap localised.
fn expand_frontier(
    to_expand: &[u64],
    params: &TraverseParams<'_>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<(u64, u64, usize)>, ExecutionError> {
    let mut out = Vec::new();
    for &src_uid in to_expand {
        let src_nid = NodeId::from_raw(src_uid);
        let neighbors = expand_one_hop(src_nid, params.edge_types, params.direction, ctx)?;
        out.reserve(neighbors.len());
        for (tgt_uid, et_idx) in neighbors {
            out.push((src_uid, tgt_uid, et_idx));
        }
    }
    Ok(out)
}

/// Variable-length path traversal: level-synchronous BFS with edge-based
/// cycle detection.
///
/// Semantics: finds all nodes reachable from source within [min_hops..max_hops]
/// via the specified edge types. Uses relationship-uniqueness (same edge cannot
/// be traversed twice in a single BFS from one source).
///
/// Inspired by Dgraph's `recurse.go` level-synchronous BFS with `reachMap`.
fn execute_varlen_traverse(
    input_rows: &[Row],
    params: &TraverseParams<'_>,
    lb: LengthBound,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let min_hops = lb.min.unwrap_or(1) as usize;
    let max_hops = lb
        .max
        .map(|m| m.min(DEFAULT_MAX_HOPS) as usize)
        .unwrap_or(DEFAULT_MAX_HOPS as usize);

    if min_hops > max_hops {
        return Ok(Vec::new());
    }

    // Per-node emission: when the planner proved the result cannot observe target
    // multiplicity (count(DISTINCT target), no path/edge variable), emit each
    // reached target once instead of once per reaching edge. Traversal itself is
    // unchanged; only row emission is deduped. Cuts O(edges) rows to O(nodes).
    let dedup_targets = ctx.dedup_varlen_targets
        && params.path_variable.is_none()
        && params.edge_variable.is_none();

    // Minimal emission: when dedup is active AND the target carries no label or
    // inline property filter to check, the sole consumer (count(DISTINCT target))
    // reads only the target id. Emit a bare target binding and skip the whole
    // per-row materialisation (target node read + decode + property formatting +
    // computed-property injection) that dominates a dense traversal. When labels
    // or filters are present we keep deduping but still materialise so the filter
    // can be applied.
    let minimal_emit =
        dedup_targets && params.target_labels.is_empty() && params.target_filters.is_empty();

    let mut results = Vec::new();

    for row in input_rows {
        let source_id = match row.get(params.source) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => continue,
        };

        // Edge-level cycle detection: (source_uid, target_uid, edge_type_idx).
        // Prevents traversing the same relationship twice per BFS invocation.
        // Keys are internal ids (not attacker-controlled), so a fast non-DoS
        // hasher is the right choice on this hot loop.
        let mut visited_edges: rustc_hash::FxHashSet<(u64, u64, usize)> =
            rustc_hash::FxHashSet::default();

        // Nodes already expanded in this traversal. A node is expanded at most
        // once: re-expanding it (reached again via a different edge at the same
        // or a later depth) would only re-encounter its already-visited
        // out-edges, so the emitted rows are identical and the adjacency read
        // is pure waste. Guarding here removes that waste across every depth.
        let mut expanded: rustc_hash::FxHashSet<u64> = rustc_hash::FxHashSet::default();

        // Target nodes already emitted as a result row for this source. Only
        // consulted when `dedup_targets` is set; keeps each reached node to one
        // emitted row within the [min..max] window.
        let mut emitted_targets: rustc_hash::FxHashSet<u64> = rustc_hash::FxHashSet::default();

        // Predecessor map for named-path reconstruction: only built when a path
        // variable is requested. `pred[n] = Some((prev, edge_type_idx))`, the
        // source's entry is `None`. Records the first (shortest) route to each
        // reached node.
        let mut pred: rustc_hash::FxHashMap<u64, Option<(u64, usize)>> =
            rustc_hash::FxHashMap::default();
        if params.path_variable.is_some() {
            pred.insert(source_id.as_raw(), None);
        }

        // Adaptive: track total edges processed for divergence detection.
        let mut edges_processed: usize = 0;
        let mut divergence_detected = false;

        // Expected fan-out per depth (from cost estimation defaults).
        let expected_per_depth = 50.0_f64; // matches CostDefaults::avg_fan_out

        // Frontier: UIDs at the current BFS depth.
        let mut frontier: Vec<u64> = vec![source_id.as_raw()];

        for depth in 1..=max_hops {
            if frontier.is_empty() {
                break;
            }

            let mut next_frontier: Vec<u64> = Vec::new();
            let depth_start_edges = edges_processed;

            // Collect all unique neighbors across the frontier for this depth
            let mut depth_neighbors: Vec<(u64, u64, usize)> = Vec::new(); // (src, tgt, et_idx)

            // Expand the whole depth frontier in one step. Each node expands at
            // most once across the traversal, so filter to the not-yet-expanded
            // members (in frontier order) and hand them to the frontier-level
            // expansion seam. The single-shard engine expands locally; the
            // distributed engine routes each source to its owning shard, which
            // runs this same expansion and returns its slice (scatter-gather).
            let to_expand: Vec<u64> = frontier
                .iter()
                .copied()
                .filter(|&src_uid| expanded.insert(src_uid))
                .collect();
            let expansions = expand_frontier(&to_expand, params, ctx)?;

            for (src_uid, tgt_uid, et_idx) in expansions {
                if !visited_edges.insert((src_uid, tgt_uid, et_idx)) {
                    continue;
                }
                edges_processed += 1;
                next_frontier.push(tgt_uid);
                if params.path_variable.is_some() {
                    pred.entry(tgt_uid).or_insert(Some((src_uid, et_idx)));
                }
                if depth >= min_hops && (!dedup_targets || emitted_targets.insert(tgt_uid)) {
                    depth_neighbors.push((src_uid, tgt_uid, et_idx));
                }
            }

            // Adaptive check: detect divergence at this depth
            if ctx.adaptive.enabled && ctx.adaptive.check_interval > 0 && !divergence_detected {
                let expected = expected_per_depth * depth as f64;
                let actual = edges_processed as f64;
                if expected > 0.0 && actual / expected > ctx.adaptive.switch_threshold {
                    divergence_detected = true;
                    ctx.warnings.push(format!(
                        "adaptive: variable-length traversal divergence detected \
                         at depth {depth}: {edges_processed} edges processed \
                         (expected ~{:.0}, threshold {:.0}x). \
                         Switching to parallel processing.",
                        expected, ctx.adaptive.switch_threshold,
                    ));
                }
            }

            // Per-depth fan-out check
            let depth_edges = edges_processed - depth_start_edges;
            let frontier_size = frontier.len();
            if ctx.adaptive.enabled && frontier_size > 0 && !divergence_detected {
                let actual_fan_out = depth_edges as f64 / frontier_size as f64;
                if actual_fan_out > expected_per_depth * ctx.adaptive.switch_threshold {
                    divergence_detected = true;
                    ctx.warnings.push(format!(
                        "adaptive: super-node fan-out at depth {depth}: \
                         avg {actual_fan_out:.0} edges/node \
                         (expected ~{expected_per_depth:.0}, threshold {:.0}x). \
                         Switching to parallel processing.",
                        ctx.adaptive.switch_threshold,
                    ));
                }
            }

            // Build result rows: parallel if enough neighbors, sequential otherwise.
            // Temporal edges and temporal target labels force sequential —
            // the parallel path doesn't yet know how to fan out across
            // stored versions on either axis.
            let has_temporal = params.edge_temporal.iter().any(|t| *t);
            let target_has_temporal = params.target_labels.iter().any(|lbl| {
                ctx.load_current_label_schema(lbl)
                    .ok()
                    .flatten()
                    .is_some_and(|s| s.temporal)
            });
            let use_parallel = ctx.adaptive.enabled
                && !has_temporal
                && !target_has_temporal
                && params.path_variable.is_none()
                && depth_neighbors.len() >= ctx.adaptive.parallel_threshold;

            if minimal_emit {
                // Bare target binding only: count(DISTINCT target) reads nothing
                // else. depth_neighbors is already deduped to distinct targets.
                for &(_src_uid, tgt_uid, _et_idx) in &depth_neighbors {
                    let mut out = row.clone();
                    out.insert(
                        params.target_variable.to_string(),
                        Value::Int(tgt_uid as i64),
                    );
                    results.push(out);
                }
            } else if use_parallel {
                // depth_neighbors already has (src, tgt, et_idx) — pass directly
                let pctx = ParallelCtx {
                    engine: ctx.engine,
                    interner: ctx.interner,
                    shard_id: ctx.shard_id,
                    mvcc_snapshot: ctx.mvcc_snapshot,
                    chunk_size: ctx.adaptive.parallel_chunk_size,
                    occ_read_keys: if ctx.mvcc_oracle.is_some() {
                        Some(OccReadKeys::default())
                    } else {
                        None
                    },
                };
                let parallel_rows = process_targets_parallel(&depth_neighbors, row, params, &pctx);
                // Merge OCC read keys from parallel workers into the Layer-3
                // scope via the typed per-partition extends.
                if let Some(ref keys) = pctx.occ_read_keys {
                    if let Some(scope) = ctx.ensure_occ_scope() {
                        if let Ok(mut node) = keys.node.lock() {
                            scope.extend_node_keys(node.drain(..));
                        }
                        if let Ok(mut ep) = keys.edge_prop.lock() {
                            scope.extend_edge_prop_keys(ep.drain(..));
                        }
                    }
                }
                results.extend(parallel_rows);
            } else {
                for &(src_uid, tgt_uid, et_idx) in &depth_neighbors {
                    // For depth > 1 the source becomes an intermediate node, so
                    // edge property lookups must rebind it to that node. At depth
                    // 1 (and any hop whose source is still the original) the row
                    // already binds `source_id`, so skip the per-edge row clone
                    // and the source-key allocation entirely on that hot path.
                    let hop_row;
                    let input_row: &Row = if src_uid == source_id.as_raw() {
                        row
                    } else {
                        let mut r = row.clone();
                        r.insert(params.source.to_string(), Value::Int(src_uid as i64));
                        hop_row = r;
                        &hop_row
                    };

                    let edge_type = params.edge_types.get(et_idx).map(|s| s.as_str());
                    let edge_is_temporal =
                        params.edge_temporal.get(et_idx).copied().unwrap_or(false);
                    let trp = TargetRowParams {
                        input_row,
                        target_uid: tgt_uid,
                        edge_type,
                        edge_is_temporal,
                    };
                    let mut target_rows = build_target_rows(&trp, params, ctx)?;
                    if let Some(pv) = params.path_variable {
                        let path = reconstruct_bfs_path(
                            source_id.as_raw(),
                            tgt_uid,
                            &pred,
                            params.edge_types,
                        );
                        for r in &mut target_rows {
                            r.insert(pv.to_string(), path.clone());
                        }
                    }
                    results.extend(target_rows);
                }
            }

            frontier = next_frontier;
        }
    }

    Ok(results)
}

/// Try to use HNSW index for vector filtering instead of brute-force.
///
/// Returns `Ok(Some(rows))` if an HNSW index was used successfully,
/// `Ok(None)` if no applicable index exists (caller should fall back to brute-force),
/// or `Err` on execution failure.
///
/// Strategy: extract (label, property) from the vector expression, look up
/// the VectorIndexRegistry, use HNSW search to get candidate node IDs,
/// then intersect with input rows and apply threshold filter on HNSW scores.
/// Parameters for vector score filtering (threshold comparison + optional decay).
struct VectorScoreParams<'a> {
    function: &'a str,
    less_than: bool,
    threshold: f64,
    decay_field: Option<&'a Expr>,
}

fn try_hnsw_vector_filter(
    rows: &[Row],
    vector_expr: &Expr,
    query_vector_expr: &Expr,
    params: &VectorScoreParams<'_>,
    ctx: &ExecutionContext<'_>,
) -> Result<Option<Vec<Row>>, ExecutionError> {
    // Need a vector index registry to attempt HNSW path.
    let registry = match ctx.vector_index_registry {
        Some(r) => r,
        None => return Ok(None),
    };

    // Extract variable name and property from vector_expr (e.g. n.embedding).
    let (variable, property) = match vector_expr {
        Expr::PropertyAccess { expr, property } => match expr.as_ref() {
            Expr::Variable(var) => (var.as_str(), property.as_str()),
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };

    if rows.is_empty() {
        return Ok(Some(Vec::new()));
    }

    // Determine the label from the first row's __label__ field.
    let label_key = format!("{variable}.__label__");
    let label = match rows[0].get(&label_key) {
        Some(Value::String(l)) => l.as_str(),
        _ => return Ok(None), // no label info → can't look up index
    };

    // Check if an HNSW index exists for this (label, property).
    if !registry.has_index(label, property) {
        return Ok(None);
    }

    // Evaluate the query vector (constant across all rows).
    let query_val = eval_expr(query_vector_expr, &rows[0]);
    let query_vec = match coerce_value_to_vec(&query_val) {
        Some(v) => v,
        None => return Ok(None),
    };

    // Determine K for HNSW search: fetch enough candidates to cover threshold.
    // When decay is present, overfetch by 2x since decay reduces scores,
    // requiring more candidates to find enough passing the combined threshold.
    let base_k = rows.len().clamp(100, 10_000);
    let k = if params.decay_field.is_some() {
        (base_k * 2).min(10_000)
    } else {
        base_k
    };

    gate_vector_index_read(ctx.engine, registry, label, property)?;
    let results =
        match registry.search_with_loader(label, property, &query_vec, k, ctx.vector_loader) {
            Some(r) => r,
            None => return Ok(None),
        };

    // Build a set of candidate node IDs from HNSW results for fast membership check.
    let candidate_set: std::collections::HashSet<u64> = results.iter().map(|r| r.id).collect();

    let mut filtered = Vec::new();
    for row in rows {
        // Extract node ID from the row. NodeScan stores it as row[variable] = Int(id).
        let node_id = match row.get(variable) {
            Some(Value::Int(id)) => *id as u64,
            _ => continue,
        };

        // Only consider rows whose node_id is in the HNSW candidate set.
        if !candidate_set.contains(&node_id) {
            continue;
        }

        // Re-compute exact score for threshold comparison.
        let vec_val = eval_expr(vector_expr, row);
        let a = match coerce_value_to_vec(&vec_val) {
            Some(v) if v.len() == query_vec.len() => v,
            _ => continue,
        };

        let raw_score = match params.function {
            "vector_distance" => {
                coordinode_vector::metrics::euclidean_distance(&a, &query_vec) as f64
            }
            "vector_similarity" => {
                coordinode_vector::metrics::cosine_similarity(&a, &query_vec) as f64
            }
            "vector_dot" => coordinode_vector::metrics::dot_product(&a, &query_vec) as f64,
            "vector_manhattan" => {
                coordinode_vector::metrics::manhattan_distance(&a, &query_vec) as f64
            }
            _ => continue,
        };

        let score = apply_decay_multiplier(raw_score, params.decay_field, row);

        let passes = if params.less_than {
            score < params.threshold
        } else {
            score > params.threshold
        };

        if passes {
            filtered.push(row.clone());
        }
    }

    Ok(Some(filtered))
}

/// Threshold below which brute-force top-K is used regardless of HNSW availability.
///
/// For small row sets (e.g. after a selective filter or traversal), computing
/// distance per row is cheaper than an HNSW search + row-set intersection.
/// The HNSW search itself has O(ef_search * log N) overhead plus memory allocation
/// for the candidate list; brute force on <1000 rows is typically faster and
/// gives exact results without any intersection bookkeeping.
const VECTOR_TOP_K_BRUTE_FORCE_THRESHOLD: usize = 1000;

/// Try HNSW-accelerated top-K search for `LogicalOp::VectorTopK`.
///
/// Returns `Some(Vec<Row>)` when the HNSW path is applicable (index exists,
/// input rows map to a single label, and the row set is large enough to benefit
/// from HNSW acceleration). Returns `None` to signal fallback to brute force.
///
/// Algorithm:
/// 1. Extract `(variable, property)` from `vector_expr` (must be `n.prop` form)
/// 2. Resolve `(label, property)` via planner annotation (preferred) or `rows[0]["variable.__label__"]`
/// 3. Evaluate the query vector
/// 5. Call `registry.search(label, property, query_vec, overfetch)` — may return
///    candidates not present in input rows (after Filter/Traverse)
/// 6. Intersect HNSW results with input row set (lookup by node_id)
/// 7. Return top-K rows in HNSW order, augmented with `distance_alias` column
///
/// **Small row sets**: when `rows.len() < VECTOR_TOP_K_BRUTE_FORCE_THRESHOLD`,
/// brute force is faster — HNSW overhead outweighs its benefit. Returns `None`
/// immediately to force the fallback path.
///
/// **Overfetch strategy**: request `max(k * 4, rows.len() * 2, 100)` candidates
/// from HNSW to tolerate row set reduction from upstream filters. The
/// `rows.len() * 2` term ensures that filter/traverse subsets which are large
/// but still narrower than the global index get enough HNSW candidates to
/// produce a meaningful intersection. If the intersection is still < k, fall
/// back to brute force on the full row set.
/// `hnsw_index_name`: optional index name from the planner annotation (set by
/// `annotate_vector_top_k`). When provided, (label, property) are resolved via
/// `registry.get_definition_by_name` — skipping the `__label__` row heuristic.
/// When None, falls back to detecting label from `rows[0].__label__`.
#[allow(clippy::too_many_arguments)]
/// Walk a [`VectorPredicate`] tree and collect every property name into the
/// supplied map, with its resolved field id (or skipped when the interner
/// doesn't know the name — predicate evaluation will then reject the leaf).
///
/// Called once per query, outside the HNSW hot loop, so the closure built
/// from the resulting map can answer field lookups without ever touching
/// the shared interner lock again.
fn collect_predicate_property_ids(
    predicate: &crate::planner::logical::VectorPredicate,
    interner: &FieldInterner,
    out: &mut std::collections::HashMap<String, u32>,
) {
    use crate::planner::logical::VectorPredicate as VP;
    match predicate {
        VP::LabelEq(_) => {}
        VP::PropertyEq { property, .. } | VP::PropertyCmp { property, .. } => {
            if let Some(fid) = interner.lookup(property) {
                out.insert(property.clone(), fid);
            }
        }
        VP::And(left, right) => {
            collect_predicate_property_ids(left, interner, out);
            collect_predicate_property_ids(right, interner, out);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn try_hnsw_vector_top_k(
    rows: &[Row],
    vector_expr: &Expr,
    query_vector_expr: &Expr,
    function: &str,
    k: usize,
    distance_alias: Option<&str>,
    hnsw_index_name: Option<&str>,
    predicate: Option<&crate::planner::logical::VectorPredicate>,
    ctx: &ExecutionContext<'_>,
) -> Result<Option<Vec<Row>>, ExecutionError> {
    if rows.is_empty() || k == 0 {
        return Ok(Some(Vec::new()));
    }

    // Small input: brute force is cheaper than HNSW + intersection overhead.
    // This covers the typical hybrid_search case where traversal narrows the
    // candidate set to a handful of nodes per query.
    if rows.len() < VECTOR_TOP_K_BRUTE_FORCE_THRESHOLD {
        return Ok(None);
    }

    let registry = match ctx.vector_index_registry {
        Some(r) => r,
        None => return Ok(None),
    };

    // Extract variable name and property from vector_expr (e.g. n.embedding).
    let (variable, property) = match vector_expr {
        Expr::PropertyAccess { expr, property } => match expr.as_ref() {
            Expr::Variable(var) => (var.as_str(), property.as_str()),
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };

    // Resolve (label, property) — prefer the planner annotation over row heuristic.
    //
    // Using the planner annotation avoids a runtime string lookup in `rows[0]` and
    // handles cases where `__label__` may not be projected into the row set.
    // Falls back to `__label__` detection when the annotation is absent or the
    // named index was dropped between plan and execution.
    let (label_str, property_str): (String, String) = if let Some(name) = hnsw_index_name {
        if let Some(def) = registry.get_definition_by_name(name) {
            // Planner annotation resolves index directly by name — no row scan needed.
            (def.label.clone(), def.property().to_string())
        } else {
            // Index was dropped after planning — fall back to row-based detection.
            let label_key = format!("{variable}.__label__");
            let l = match rows[0].get(&label_key) {
                Some(Value::String(l)) => l.clone(),
                _ => return Ok(None),
            };
            if !registry.has_index(&l, property) {
                return Ok(None);
            }
            (l, property.to_string())
        }
    } else {
        // No planner annotation — detect label from the first row's __label__ field.
        let label_key = format!("{variable}.__label__");
        let l = match rows[0].get(&label_key) {
            Some(Value::String(l)) => l.clone(),
            _ => return Ok(None),
        };
        if !registry.has_index(&l, property) {
            return Ok(None);
        }
        (l, property.to_string())
    };

    // Evaluate the query vector (constant across all rows).
    let query_val = eval_expr(query_vector_expr, &rows[0]);
    let query_vec = match coerce_value_to_vec(&query_val) {
        Some(v) => v,
        None => return Ok(None),
    };

    // Overfetch strategy — request enough HNSW candidates to cover:
    // - k * 4: baseline margin for ordering stability
    // - rows.len() * 2: double the filter-reduced subset size, so intersection
    //   likely contains at least k actual rows even when HNSW's globally-nearest
    //   are not in our filtered subset
    // - 100: floor for very small k
    // Capped at 10_000 to prevent excessive memory for massive result sets.
    let overfetch = (k * 4).max(rows.len() * 2).clamp(100, 10_000);

    gate_vector_index_read(ctx.engine, registry, &label_str, &property_str)?;

    // ACORN-style filtered search: when the planner pushed a predicate down,
    // pass it as a visibility closure so the HNSW traversal prunes branches
    // that can't pass the filter. Otherwise fall back to the unfiltered path.
    let search_results = if let Some(pred) = predicate {
        // Resolve property names referenced by the predicate once, outside
        // the closure, so the search hot path never re-enters the interner
        // lock. We snapshot every PropertyEq leaf into a stable map.
        let mut field_ids: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        collect_predicate_property_ids(pred, ctx.interner, &mut field_ids);

        let engine = ctx.engine;
        let shard_id = ctx.shard_id;
        let pred_clone = pred.clone();
        let lookup = move |name: &str| field_ids.get(name).copied();
        let is_visible = move |node_id: u64| {
            crate::executor::vector_predicate::evaluate_predicate(
                engine,
                shard_id,
                coordinode_core::graph::node::NodeId::from_raw(node_id),
                &pred_clone,
                &lookup,
            )
        };
        // Overfetch factor 2.0 + 3 expansion rounds matches the existing
        // visibility-aware MVCC search defaults; ACORN literature suggests
        // higher overfetch for very selective filters but those tunables
        // remain a follow-up.
        match registry.search_with_visibility(
            &label_str,
            &property_str,
            &query_vec,
            overfetch,
            2.0,
            3,
            is_visible,
        ) {
            Some(r) => r,
            None => return Ok(None),
        }
    } else {
        match registry.search_with_loader(
            &label_str,
            &property_str,
            &query_vec,
            overfetch,
            ctx.vector_loader,
        ) {
            Some(r) => r,
            None => return Ok(None),
        }
    };

    // Build node_id → row map from input rows for intersection.
    let mut row_by_id: std::collections::HashMap<u64, &Row> = std::collections::HashMap::new();
    for row in rows {
        if let Some(Value::Int(id)) = row.get(variable) {
            row_by_id.insert(*id as u64, row);
        }
    }

    // Intersect HNSW candidates with input rows, preserving HNSW order.
    // `SearchResult::score` is raw HNSW distance (L2 or configured metric).
    // For `vector_distance` (lower is better), HNSW already returns in that
    // order. For `vector_similarity`/`vector_dot` (higher is better), we recompute
    // the score for the requested function when writing distance_alias.
    let mut intersected: Vec<(f32, &Row)> = Vec::new();
    for result in &search_results {
        if let Some(row) = row_by_id.get(&result.id) {
            intersected.push((result.score, row));
        }
    }

    // If intersection is smaller than k, fall back to brute force. The HNSW
    // top-`overfetch` missed too many candidates that are in `rows` — meaning
    // the upstream filter was very restrictive and our overfetch was too small.
    if intersected.len() < k.min(rows.len()) {
        return Ok(None);
    }

    // For distance functions, HNSW order is already correct (ascending).
    // For similarity/dot_product, we need to re-score with the actual function
    // because HNSW returns L2 distances and the user asked for a different metric.
    // Simplification: HNSW index is built for a specific metric; we trust it.
    let mut result_rows: Vec<Row> = Vec::with_capacity(k);
    for (dist, row) in intersected.into_iter().take(k) {
        let mut cloned = row.clone();
        if let Some(alias) = distance_alias {
            // Compute the exact score using the requested function — HNSW may
            // have returned raw L2 distance even when the user asked for
            // `vector_similarity`. Re-evaluate to get the correct value.
            let recomputed = recompute_score_for_row(vector_expr, &query_vec, function, &cloned);
            cloned.insert(
                alias.to_string(),
                Value::Float(recomputed.unwrap_or(dist as f64)),
            );
        }
        result_rows.push(cloned);
    }

    Ok(Some(result_rows))
}

/// Recompute an exact vector score for a single row, using the requested function.
///
/// HNSW indexes return raw L2 distances; if the user asked for `vector_similarity`
/// (cosine) or `vector_dot`, we must recompute the score from the row's vector.
fn recompute_score_for_row(
    vector_expr: &Expr,
    query_vec: &[f32],
    function: &str,
    row: &Row,
) -> Option<f64> {
    let vec_val = eval_expr(vector_expr, row);
    let a = coerce_value_to_vec(&vec_val)?;
    if a.len() != query_vec.len() {
        return None;
    }
    let score = match function {
        "vector_distance" => coordinode_vector::metrics::euclidean_distance(&a, query_vec) as f64,
        "vector_similarity" => coordinode_vector::metrics::cosine_similarity(&a, query_vec) as f64,
        "vector_dot" => coordinode_vector::metrics::dot_product(&a, query_vec) as f64,
        "vector_manhattan" => coordinode_vector::metrics::manhattan_distance(&a, query_vec) as f64,
        _ => return None,
    };
    Some(score)
}

/// Brute-force top-K: compute distance per row, sort, take first K.
///
/// Used as fallback when `try_hnsw_vector_top_k` returns `None` (no index,
/// non-NodeScan input rows, or insufficient HNSW intersection).
fn execute_vector_top_k_brute_force(
    rows: Vec<Row>,
    vector_expr: &Expr,
    query_vector_expr: &Expr,
    function: &str,
    k: usize,
    distance_alias: Option<&str>,
) -> Result<Vec<Row>, ExecutionError> {
    if rows.is_empty() || k == 0 {
        return Ok(Vec::new());
    }

    // Evaluate query vector once (constant across rows).
    let query_val = eval_expr(query_vector_expr, &rows[0]);
    let query_vec = match coerce_value_to_vec(&query_val) {
        Some(v) => v,
        None => return Ok(Vec::new()),
    };

    // Compute (score, row) pairs, skipping rows with missing/mismatched vectors.
    let mut scored: Vec<(f64, Row)> = Vec::with_capacity(rows.len());
    for row in rows {
        let vec_val = eval_expr(vector_expr, &row);
        let a = match coerce_value_to_vec(&vec_val) {
            Some(v) if v.len() == query_vec.len() => v,
            _ => continue,
        };
        let score = match function {
            "vector_distance" => {
                coordinode_vector::metrics::euclidean_distance(&a, &query_vec) as f64
            }
            "vector_similarity" => {
                coordinode_vector::metrics::cosine_similarity(&a, &query_vec) as f64
            }
            "vector_dot" => coordinode_vector::metrics::dot_product(&a, &query_vec) as f64,
            "vector_manhattan" => {
                coordinode_vector::metrics::manhattan_distance(&a, &query_vec) as f64
            }
            _ => continue,
        };
        scored.push((score, row));
    }

    // Sort: distance/manhattan ASC (lower is better), similarity/dot DESC (higher is better).
    let ascending = matches!(function, "vector_distance" | "vector_manhattan");
    scored.sort_by(|(a, _), (b, _)| {
        if ascending {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    // Take top-K, optionally augment with distance alias.
    let mut result = Vec::with_capacity(k.min(scored.len()));
    for (score, mut row) in scored.into_iter().take(k) {
        if let Some(alias) = distance_alias {
            row.insert(alias.to_string(), Value::Float(score));
        }
        result.push(row);
    }
    Ok(result)
}

/// VectorFilter: evaluate vector distance/similarity per row and filter.
///
/// For each row, compute vector function(vector_expr, query_vector),
/// compare against threshold. Keep rows that pass the comparison.
/// Apply decay multiplier to a raw vector score when a decay field is present.
fn apply_decay_multiplier(raw_score: f64, decay_field: Option<&Expr>, row: &Row) -> f64 {
    if let Some(decay_expr) = decay_field {
        let decay_val = eval_expr(decay_expr, row);
        let decay_factor = match decay_val {
            Value::Float(f) => f,
            Value::Int(i) => i as f64,
            _ => 1.0, // missing decay → no attenuation
        };
        raw_score * decay_factor
    } else {
        raw_score
    }
}

fn execute_vector_filter(
    rows: &[Row],
    vector_expr: &Expr,
    query_vector: &Expr,
    params: &VectorScoreParams<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let mut results = Vec::new();

    for row in rows {
        let vec_val = eval_expr(vector_expr, row);
        let query_val = eval_expr(query_vector, row);

        // Coerce values to f32 vectors
        let vec_a = coerce_value_to_vec(&vec_val);
        let vec_b = coerce_value_to_vec(&query_val);

        let (Some(a), Some(b)) = (vec_a, vec_b) else {
            continue; // skip rows with non-vector values
        };

        if a.len() != b.len() {
            continue; // dimension mismatch
        }

        let raw_score = match params.function {
            "vector_distance" => coordinode_vector::metrics::euclidean_distance(&a, &b) as f64,
            "vector_similarity" => coordinode_vector::metrics::cosine_similarity(&a, &b) as f64,
            "vector_dot" => coordinode_vector::metrics::dot_product(&a, &b) as f64,
            "vector_manhattan" => coordinode_vector::metrics::manhattan_distance(&a, &b) as f64,
            _ => continue,
        };

        let score = apply_decay_multiplier(raw_score, params.decay_field, row);

        let passes = if params.less_than {
            score < params.threshold
        } else {
            score > params.threshold
        };

        if passes {
            // Cache the raw (pre-decay) vector score + function name so downstream
            // scalars like `hybrid_score(node, query)` can normalize and blend it
            // with `__text_score__`. Raw (not decay-adjusted) because hybrid_score
            // is an orthogonal scoring surface and must not double-apply decay.
            let mut out_row = row.clone();
            out_row.insert("__vector_score__".to_string(), Value::Float(raw_score));
            out_row.insert(
                "__vector_function__".to_string(),
                Value::String(params.function.to_string()),
            );
            results.push(out_row);
        }
    }

    Ok(results)
}

/// RRF constant `k`: the industry standard from Cormack et al. 2009.
/// Default for `FusionStrategy::Rrf` — callers that pass a different `k`
/// via the strategy use that value; this constant stays as the
/// documented baseline.
#[allow(dead_code)]
const RRF_K: f64 = 60.0;

/// Resolved scoring mode for a single `rrf_score` method expression.
///
/// Classified on-demand during `execute_rank_fuse` from the first row that
/// carries a label for the referenced variable and the available registries.
#[derive(Debug, Clone)]
enum RankFuseMethodKind {
    /// Node vector property backed by HNSW index (metric from index config).
    /// Direction: `desc = true` for similarity/dot, `false` for distance metrics.
    VectorHnsw {
        // Kept for future EXPLAIN annotations / HNSW-accelerated scoring.
        #[allow(dead_code)]
        label: String,
        #[allow(dead_code)]
        property: String,
        metric: coordinode_core::graph::types::VectorMetric,
    },
    /// Vector property without an HNSW index (e.g. edge vector property).
    /// Always scored via cosine similarity (DESC direction).
    VectorBruteForce,
    /// Text property backed by `TextIndexRegistry` — BM25 scoring, DESC direction.
    TextBm25 { label: String, property: String },
}

impl RankFuseMethodKind {
    /// Higher score = better (for rank assignment). `true` → sort DESC, rank 1
    /// to the largest score. `false` → sort ASC, rank 1 to the smallest score.
    fn desc(&self) -> bool {
        match self {
            // Similarity: larger is better. Dot product: larger is better.
            // Cosine distance / L2 / L1: smaller is better.
            Self::VectorHnsw { metric, .. } => matches!(
                metric,
                coordinode_core::graph::types::VectorMetric::Cosine
                    | coordinode_core::graph::types::VectorMetric::DotProduct
            ),
            // Cosine similarity brute-force: larger is better.
            Self::VectorBruteForce => true,
            // BM25: larger is better.
            Self::TextBm25 { .. } => true,
        }
    }
}

/// Extract `(variable, property)` from a method expression.
///
/// RRF methods are always property accesses on a variable —
/// `n.embedding`, `r.context_emb`, `c.body`, etc. Any other shape is rejected.
fn extract_method_ident(expr: &Expr) -> Option<(String, String)> {
    match expr {
        Expr::PropertyAccess { expr, property } => {
            if let Expr::Variable(var) = expr.as_ref() {
                return Some((var.clone(), property.clone()));
            }
            None
        }
        _ => None,
    }
}

/// Resolve `(label, property)` for a variable by consulting `row[{var}.__label__]`.
/// Returns the first non-empty label found across the given rows.
fn resolve_label_for_var(rows: &[Row], variable: &str) -> Option<String> {
    let key = format!("{variable}.__label__");
    for row in rows {
        if let Some(Value::String(s)) = row.get(&key) {
            if !s.is_empty() {
                return Some(s.clone());
            }
        }
    }
    None
}

/// Extract the `NodeId` bound to a variable from a row. Rows produced by
/// `execute_node_scan` bind the raw id under `row[variable]` as `Value::Int`.
fn row_node_id(row: &Row, variable: &str) -> Option<u64> {
    match row.get(variable)? {
        Value::Int(id) => Some(*id as u64),
        _ => None,
    }
}

/// Classify a single RRF method. Queries both registries; if neither matches
/// and at least one row has `Value::Vector` for the method expression, treat
/// it as brute-force vector (edge vector property or schemaless vector).
/// Returns `Err` with a user-facing message when the method cannot be scored.
fn resolve_rank_fuse_method(
    method_expr: &Expr,
    rows: &[Row],
    ctx: &ExecutionContext<'_>,
) -> Result<RankFuseMethodKind, ExecutionError> {
    let (variable, property) = extract_method_ident(method_expr).ok_or_else(|| {
        ExecutionError::Unsupported(
            "rrf_score(): method expressions must be property accesses \
                 (e.g. n.embedding, r.context_emb, c.body); \
                 complex expressions are not scorable"
                .to_string(),
        )
    })?;

    let label_opt = resolve_label_for_var(rows, &variable);

    // Prefer typed registry hits keyed by the variable's label.
    if let Some(label) = label_opt.as_deref() {
        if let Some(reg) = ctx.vector_index_registry {
            if let Some(def) = reg.get_definition(label, &property) {
                if let Some(cfg) = def.vector_config.as_ref() {
                    return Ok(RankFuseMethodKind::VectorHnsw {
                        label: label.to_string(),
                        property,
                        metric: cfg.metric,
                    });
                }
            }
        }
        if let Some(reg) = ctx.text_index_registry {
            if reg.get(label, &property).is_some() {
                return Ok(RankFuseMethodKind::TextBm25 {
                    label: label.to_string(),
                    property,
                });
            }
        }
    }

    // Fallback: brute-force vector if any row evaluates the method to a Vector.
    // Covers edge vector properties (no query-time edge HNSW registry yet) and
    // schemaless vectors. Text fields MUST have a full-text index — no fallback.
    let any_vector = rows
        .iter()
        .any(|row| matches!(eval_expr(method_expr, row), Value::Vector(_)));
    if any_vector {
        return Ok(RankFuseMethodKind::VectorBruteForce);
    }

    // Unscorable.
    let label_part = label_opt.map(|l| format!(":{l}")).unwrap_or_default();
    Err(ExecutionError::Unsupported(format!(
        "rrf_score(): method {variable}.{property} on ({variable}{label_part}) \
         cannot be scored — no HNSW vector index, no full-text index, and no \
         vector values observed in input. Create a CREATE VECTOR INDEX or \
         CREATE TEXT INDEX on the property, or remove this method from the list."
    )))
}

/// Execute `LogicalOp::RankFuse`: materialize input, score each method,
/// assign competition ranks (1-based, ties broken by node_id ASC), compute
/// `Σ 1/(60 + rank_i)` and write it to `__rrf_score__` on every output row.
#[allow(clippy::too_many_arguments)]
fn execute_rank_fuse(
    rows: Vec<Row>,
    methods: &[Expr],
    query_vector: Option<&Expr>,
    query_text: Option<&Expr>,
    shard_overfetch_cap: Option<usize>,
    fusion: &crate::planner::logical::FusionStrategy,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    if methods.is_empty() {
        return Err(ExecutionError::Unsupported(
            "rrf_score(): method list is empty; \
             provide at least one vector or text property expression"
                .to_string(),
        ));
    }
    if rows.is_empty() {
        return Ok(rows);
    }

    // Resolve all methods up-front so we fail fast on a single bad method.
    let mut kinds = Vec::with_capacity(methods.len());
    let mut needs_vec = false;
    let mut needs_text = false;
    for m in methods {
        let kind = resolve_rank_fuse_method(m, &rows, ctx)?;
        match &kind {
            RankFuseMethodKind::VectorHnsw { .. } | RankFuseMethodKind::VectorBruteForce => {
                needs_vec = true;
            }
            RankFuseMethodKind::TextBm25 { .. } => {
                needs_text = true;
            }
        }
        kinds.push(kind);
    }

    // Evaluate query once — it may be a Parameter resolved elsewhere, so a
    // zero-row eval against an empty Row is sufficient for literal / parameter
    // shapes. Params have already been substituted by `substitute_params`.
    let qv_value = query_vector
        .map(|e| eval_expr(e, &Row::new()))
        .unwrap_or(Value::Null);
    let qt_value = query_text
        .map(|e| eval_expr(e, &Row::new()))
        .unwrap_or(Value::Null);

    let query_vec: Option<Vec<f32>> = coerce_value_to_vec(&qv_value);
    let query_text_str: Option<String> = match &qt_value {
        Value::String(s) => Some(s.clone()),
        _ => None,
    };

    if needs_vec && query_vec.is_none() {
        return Err(ExecutionError::Unsupported(
            "rrf_score(): at least one method is a vector property but the \
             query map has no `vector` key (or it is not a vector value)"
                .to_string(),
        ));
    }
    if needs_text && query_text_str.is_none() {
        return Err(ExecutionError::Unsupported(
            "rrf_score(): at least one method is a text property but the \
             query map has no `text` key (or it is not a string value)"
                .to_string(),
        ));
    }

    let n = rows.len();
    // ranks[method_i][row_idx] = Some(rank) if matched, None → penalty = matched+1.
    let mut ranks: Vec<Vec<Option<usize>>> = Vec::with_capacity(methods.len());

    // Defensive: the needs_vec / needs_text guards above already rejected
    // missing-query cases with user-facing errors. These `ok_or_else` checks
    // convert any drift in the guard logic into a regular ExecutionError
    // rather than an internal panic.
    let qv_slice: &[f32] = match query_vec.as_deref() {
        Some(v) => v,
        None if needs_vec => {
            return Err(ExecutionError::Unsupported(
                "rrf_score(): internal invariant violated — vector query missing after guard"
                    .into(),
            ));
        }
        None => &[],
    };
    let qt_slice: &str = match query_text_str.as_deref() {
        Some(s) => s,
        None if needs_text => {
            return Err(ExecutionError::Unsupported(
                "rrf_score(): internal invariant violated — text query missing after guard".into(),
            ));
        }
        None => "",
    };

    for (method_expr, kind) in methods.iter().zip(kinds.iter()) {
        let row_ranks = match kind {
            RankFuseMethodKind::VectorHnsw { metric, .. } => score_vector_method(
                &rows,
                method_expr,
                qv_slice,
                Some(*metric),
                kind.desc(),
                &variable_for_method(method_expr),
            ),
            RankFuseMethodKind::VectorBruteForce => score_vector_method(
                &rows,
                method_expr,
                qv_slice,
                None, // default: cosine similarity
                kind.desc(),
                &variable_for_method(method_expr),
            ),
            RankFuseMethodKind::TextBm25 { label, property } => score_text_method(
                &rows,
                method_expr,
                qt_slice,
                label,
                property,
                &variable_for_method(method_expr),
                ctx,
            )?,
        };
        ranks.push(row_ranks);
    }

    // Per-method matched counts → penalty rank = matched + 1.
    let penalties: Vec<usize> = ranks
        .iter()
        .map(|mr| mr.iter().filter(|r| r.is_some()).count() + 1)
        .collect();

    // Compute fused score per row. RRF uses ranks; CC and DBSF need raw
    // scores per method, so the score-based branches recompute method
    // scores instead of falling back to the rank vector.
    let mut out_rows: Vec<Row> = Vec::with_capacity(n);
    use crate::planner::logical::FusionStrategy;
    match fusion {
        FusionStrategy::Rrf { k } => {
            let rrf_k = *k as f64;
            for (i, row) in rows.into_iter().enumerate() {
                let mut score = 0.0_f64;
                for (m, method_ranks) in ranks.iter().enumerate() {
                    let rank = method_ranks[i].unwrap_or(penalties[m]);
                    score += 1.0 / (rrf_k + rank as f64);
                }
                let mut out = row;
                // Keep historical `__rrf_score__` column for RRF callers; also
                // emit `__hybrid_score__` so the universal sort column works
                // regardless of fusion strategy.
                out.insert("__rrf_score__".to_string(), Value::Float(score));
                out.insert("__hybrid_score__".to_string(), Value::Float(score));
                out_rows.push(out);
            }
        }
        FusionStrategy::ConvexCombination { weights } | FusionStrategy::Dbsf { weights } => {
            let use_zscore = matches!(fusion, FusionStrategy::Dbsf { .. });
            let raw = compute_raw_method_scores(&rows, methods, &kinds, qv_slice, qt_slice, ctx)?;
            let fused = fuse_raw_scores(&raw, &kinds, weights, use_zscore);
            for (i, row) in rows.into_iter().enumerate() {
                let mut out = row;
                out.insert("__hybrid_score__".to_string(), Value::Float(fused[i]));
                out_rows.push(out);
            }
        }
    }

    // Apply shard overfetch cap (R-HYB5 future path; None in CE).
    if let Some(cap) = shard_overfetch_cap {
        if out_rows.len() > cap {
            // Sort by __rrf_score__ DESC and truncate to keep best candidates.
            out_rows.sort_by(|a, b| {
                let sa = a.get("__rrf_score__").and_then(|v| match v {
                    Value::Float(f) => Some(*f),
                    _ => None,
                });
                let sb = b.get("__rrf_score__").and_then(|v| match v {
                    Value::Float(f) => Some(*f),
                    _ => None,
                });
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            out_rows.truncate(cap);
        }
    }

    Ok(out_rows)
}

fn variable_for_method(expr: &Expr) -> String {
    match expr {
        Expr::PropertyAccess { expr: inner, .. } => {
            if let Expr::Variable(v) = inner.as_ref() {
                return v.clone();
            }
            String::new()
        }
        _ => String::new(),
    }
}

/// Score rows by a vector method, returning `row_ranks[i]` = competition rank
/// (1-based) when the row has a usable vector for this method, `None` when
/// the value was missing/wrong-type (→ penalty rank applied later).
///
/// When `metric` is `None`, uses cosine similarity (brute-force default).
fn score_vector_method(
    rows: &[Row],
    method_expr: &Expr,
    query_vec: &[f32],
    metric: Option<coordinode_core::graph::types::VectorMetric>,
    desc: bool,
    variable: &str,
) -> Vec<Option<usize>> {
    // (row_idx, score, node_id) for matched rows.
    let mut scored: Vec<(usize, f64, u64)> = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let val = eval_expr(method_expr, row);
        let Some(v) = coerce_value_to_vec(&val) else {
            continue;
        };
        if v.len() != query_vec.len() {
            continue;
        }
        let s = match metric {
            Some(coordinode_core::graph::types::VectorMetric::Cosine) | None => {
                coordinode_vector::metrics::cosine_similarity(&v, query_vec) as f64
            }
            Some(coordinode_core::graph::types::VectorMetric::L2) => {
                coordinode_vector::metrics::euclidean_distance(&v, query_vec) as f64
            }
            Some(coordinode_core::graph::types::VectorMetric::DotProduct) => {
                coordinode_vector::metrics::dot_product(&v, query_vec) as f64
            }
            Some(coordinode_core::graph::types::VectorMetric::L1) => {
                coordinode_vector::metrics::manhattan_distance(&v, query_vec) as f64
            }
        };
        let nid = row_node_id(row, variable).unwrap_or(u64::MAX);
        scored.push((i, s, nid));
    }

    assign_competition_ranks(rows.len(), &mut scored, desc)
}

/// Score rows by a text (BM25) method via `TextIndexRegistry`. Missing FT-index
/// for the (label, property) is a hard error (matches R-HYB1 guard spirit).
fn score_text_method(
    rows: &[Row],
    method_expr: &Expr,
    query_text: &str,
    label: &str,
    property: &str,
    variable: &str,
    ctx: &ExecutionContext<'_>,
) -> Result<Vec<Option<usize>>, ExecutionError> {
    let _ = method_expr; // kept for symmetry + future column inspection
    let registry = ctx.text_index_registry.ok_or_else(|| {
        ExecutionError::Unsupported(
            "rrf_score(): text method requires a TextIndexRegistry; \
             none is wired into the execution context"
                .to_string(),
        )
    })?;
    let handle = registry.get(label, property).ok_or_else(|| {
        ExecutionError::Unsupported(format!(
            "rrf_score(): text method {variable}.{property} on :{label} requires a \
             full-text index; create one with `CREATE TEXT INDEX … ON :{label}({property})`"
        ))
    })?;
    let idx = handle
        .read()
        .map_err(|_| ExecutionError::Unsupported("rrf_score(): text index lock poisoned".into()))?;
    // Overfetch 3× to catch boundary matches; at least 1000.
    let limit = (rows.len() * 3).max(1000);
    let results = idx
        .search(query_text, limit)
        .map_err(|e| ExecutionError::Unsupported(format!("rrf_score(): text search error: {e}")))?;
    drop(idx);

    let scores: std::collections::HashMap<u64, f32> =
        results.into_iter().map(|r| (r.node_id, r.score)).collect();

    let mut scored: Vec<(usize, f64, u64)> = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let Some(nid) = row_node_id(row, variable) else {
            continue;
        };
        let Some(&s) = scores.get(&nid) else {
            continue;
        };
        scored.push((i, s as f64, nid));
    }

    Ok(assign_competition_ranks(rows.len(), &mut scored, true))
}

/// Competition ranking (1,2,2,4). Sort `scored` by score in the requested
/// direction, ties broken deterministically by ascending `node_id`. Assign
/// rank 1 to the best row; tied rows receive the same rank; the next
/// distinct score receives `previous_rank + group_size`.
///
/// Returns `row_ranks[i]` for `i in 0..n_rows`: `Some(rank)` when row i was
/// in `scored`, `None` otherwise (penalty applied by the caller).
/// Build the per-(method, row) matrix of raw scores for the CC / DBSF
/// fusion kernels. Each cell is `Some(score)` when the row has a usable
/// value for that method, `None` otherwise. The score sign convention
/// follows the rank assignment: higher = better. Distance metrics are
/// negated so a smaller raw distance lands as a larger score.
fn compute_raw_method_scores(
    rows: &[Row],
    methods: &[Expr],
    kinds: &[RankFuseMethodKind],
    qv_slice: &[f32],
    qt_slice: &str,
    ctx: &ExecutionContext<'_>,
) -> Result<Vec<Vec<Option<f64>>>, ExecutionError> {
    let mut out: Vec<Vec<Option<f64>>> = Vec::with_capacity(methods.len());
    for (method_expr, kind) in methods.iter().zip(kinds.iter()) {
        let variable = variable_for_method(method_expr);
        let scores = match kind {
            RankFuseMethodKind::VectorHnsw { metric, .. } => {
                raw_scores_vector_method(rows, method_expr, qv_slice, Some(*metric), kind.desc())
            }
            RankFuseMethodKind::VectorBruteForce => {
                raw_scores_vector_method(rows, method_expr, qv_slice, None, kind.desc())
            }
            RankFuseMethodKind::TextBm25 { label, property } => raw_scores_text_method(
                rows,
                method_expr,
                qt_slice,
                label,
                property,
                &variable,
                ctx,
            )?,
        };
        out.push(scores);
    }
    Ok(out)
}

/// Sibling of `score_vector_method` returning raw normalised-direction
/// scores instead of competition ranks. "Normalised direction" means the
/// sign convention is `higher = better` regardless of the metric: cosine
/// / dot stay as-is, L1 / L2 are negated. Callers that need ascending
/// distance can re-negate.
fn raw_scores_vector_method(
    rows: &[Row],
    method_expr: &Expr,
    query_vec: &[f32],
    metric: Option<coordinode_core::graph::types::VectorMetric>,
    desc: bool,
) -> Vec<Option<f64>> {
    let mut out: Vec<Option<f64>> = vec![None; rows.len()];
    for (i, row) in rows.iter().enumerate() {
        let val = eval_expr(method_expr, row);
        let Some(v) = coerce_value_to_vec(&val) else {
            continue;
        };
        if v.len() != query_vec.len() {
            continue;
        }
        let raw = match metric {
            Some(coordinode_core::graph::types::VectorMetric::Cosine) | None => {
                coordinode_vector::metrics::cosine_similarity(&v, query_vec) as f64
            }
            Some(coordinode_core::graph::types::VectorMetric::L2) => {
                coordinode_vector::metrics::euclidean_distance(&v, query_vec) as f64
            }
            Some(coordinode_core::graph::types::VectorMetric::DotProduct) => {
                coordinode_vector::metrics::dot_product(&v, query_vec) as f64
            }
            Some(coordinode_core::graph::types::VectorMetric::L1) => {
                coordinode_vector::metrics::manhattan_distance(&v, query_vec) as f64
            }
        };
        // Normalise so higher = better. When the metric is ascending-by-
        // default (`desc == false`, distance metrics), flip the sign.
        let normalised = if desc { raw } else { -raw };
        out[i] = Some(normalised);
    }
    out
}

/// Sibling of `score_text_method` returning raw BM25 scores per row.
#[allow(clippy::too_many_arguments)]
fn raw_scores_text_method(
    rows: &[Row],
    method_expr: &Expr,
    query_text: &str,
    label: &str,
    property: &str,
    variable: &str,
    ctx: &ExecutionContext<'_>,
) -> Result<Vec<Option<f64>>, ExecutionError> {
    let _ = method_expr;
    let registry = ctx.text_index_registry.ok_or_else(|| {
        ExecutionError::Unsupported(
            "hybrid fusion: text method requires a TextIndexRegistry".to_string(),
        )
    })?;
    let handle = registry.get(label, property).ok_or_else(|| {
        ExecutionError::Unsupported(format!(
            "hybrid fusion: no text index on :{label}({property})"
        ))
    })?;
    let idx = handle.read().map_err(|_| {
        ExecutionError::Unsupported("hybrid fusion: text index lock poisoned".into())
    })?;
    let limit = (rows.len() * 3).max(1000);
    let results = idx
        .search(query_text, limit)
        .map_err(|e| ExecutionError::Unsupported(format!("hybrid fusion: text search: {e}")))?;
    drop(idx);

    let scores: std::collections::HashMap<u64, f32> =
        results.into_iter().map(|r| (r.node_id, r.score)).collect();

    let mut out: Vec<Option<f64>> = vec![None; rows.len()];
    for (i, row) in rows.iter().enumerate() {
        let Some(nid) = row_node_id(row, variable) else {
            continue;
        };
        if let Some(&s) = scores.get(&nid) {
            out[i] = Some(s as f64);
        }
    }
    Ok(out)
}

/// Combine per-(method, row) raw scores into a single fused score per row
/// using either min-max normalisation (Convex Combination) or z-score
/// normalisation (DBSF). `weights` maps the method's category ("vector"
/// or "text") to its blending coefficient. Missing weights default to 0
/// (method contributes nothing). Methods whose range / stddev is
/// degenerate (min == max or σ == 0) contribute zero to every row, since
/// the normalisation is undefined.
fn fuse_raw_scores(
    raw: &[Vec<Option<f64>>],
    kinds: &[RankFuseMethodKind],
    weights: &std::collections::BTreeMap<String, f64>,
    use_zscore: bool,
) -> Vec<f64> {
    let n_rows = raw.first().map(|c| c.len()).unwrap_or(0);
    let mut fused = vec![0.0_f64; n_rows];
    for (method_idx, column) in raw.iter().enumerate() {
        let category = match &kinds[method_idx] {
            RankFuseMethodKind::VectorHnsw { .. } | RankFuseMethodKind::VectorBruteForce => {
                "vector"
            }
            RankFuseMethodKind::TextBm25 { .. } => "text",
        };
        let weight = weights.get(category).copied().unwrap_or(0.0);
        if weight == 0.0 {
            continue;
        }
        let normalised = if use_zscore {
            zscore_normalise(column)
        } else {
            min_max_normalise(column)
        };
        for (i, n) in normalised.iter().enumerate() {
            if let Some(v) = n {
                fused[i] += weight * v;
            }
        }
    }
    fused
}

/// Min-max normalise: `(x - min) / (max - min)`. Returns None for any row
/// when the column's min and max coincide (degenerate range — the
/// normalised value is undefined; treat as "no contribution").
fn min_max_normalise(column: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in column.iter().flatten() {
        if *v < min {
            min = *v;
        }
        if *v > max {
            max = *v;
        }
    }
    if !min.is_finite() || !max.is_finite() || (max - min).abs() < f64::EPSILON {
        return vec![None; column.len()];
    }
    let range = max - min;
    column
        .iter()
        .map(|cell| cell.map(|v| (v - min) / range))
        .collect()
}

/// Z-score normalise: `(x - μ) / σ`. Returns None per row when σ == 0 or
/// the column has fewer than 2 matched samples.
fn zscore_normalise(column: &[Option<f64>]) -> Vec<Option<f64>> {
    let values: Vec<f64> = column.iter().filter_map(|c| *c).collect();
    if values.len() < 2 {
        return vec![None; column.len()];
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let sigma = var.sqrt();
    if sigma < f64::EPSILON {
        return vec![None; column.len()];
    }
    column
        .iter()
        .map(|cell| cell.map(|v| (v - mean) / sigma))
        .collect()
}

fn assign_competition_ranks(
    n_rows: usize,
    scored: &mut [(usize, f64, u64)],
    desc: bool,
) -> Vec<Option<usize>> {
    // Sort by primary score (direction-aware), tiebreak by node_id ASC for
    // determinism. Tied rows get the SAME rank — node_id is only used to
    // stabilise iteration order, not to break the competition tie.
    scored.sort_by(|a, b| {
        let ord = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
        let ord = if desc { ord.reverse() } else { ord };
        if ord != std::cmp::Ordering::Equal {
            ord
        } else {
            a.2.cmp(&b.2)
        }
    });

    let mut ranks = vec![None; n_rows];
    let mut i = 0;
    while i < scored.len() {
        let group_start_rank = i + 1;
        let mut j = i;
        while j < scored.len() && float_bits_eq(scored[j].1, scored[i].1) {
            ranks[scored[j].0] = Some(group_start_rank);
            j += 1;
        }
        i = j;
    }
    ranks
}

/// Equality check that treats NaN as not equal to anything (standard IEEE
/// semantics); required because `f64` does not implement `Eq`.
fn float_bits_eq(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    a == b
}

/// Eval an α/β/γ weight expression to f64 with a fallback default.
///
/// `doc_score` weights are literal Floats / Ints in the AST after builder
/// normalisation; Parameters are substituted before execute time. Anything
/// that does not resolve to a finite number falls back to the default.
fn eval_weight(expr: &Expr, default: f64) -> f64 {
    let v = eval_expr(expr, &Row::new());
    match v {
        Value::Float(f) if f.is_finite() => f,
        Value::Int(i) => i as f64,
        _ => default,
    }
}

/// Execute `LogicalOp::DocScore`: per doc row, traverse outward HAS_CHUNK,
/// read chunk nodes, score each chunk against the query vector via cosine
/// similarity, compute `α·max + β·avg + γ·coverage`, write it to
/// `__doc_score__` on the output row.
#[allow(clippy::too_many_arguments)]
fn execute_doc_score(
    rows: Vec<Row>,
    doc_variable: &str,
    query_vector: &Expr,
    alpha: &Expr,
    beta: &Expr,
    gamma: &Expr,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let alpha_f = eval_weight(alpha, 0.5);
    let beta_f = eval_weight(beta, 0.3);
    let gamma_f = eval_weight(gamma, 0.2);
    let query_val = eval_expr(query_vector, &Row::new());
    let query_vec = coerce_value_to_vec(&query_val).ok_or_else(|| {
        ExecutionError::Unsupported(
            "doc_score(): query argument must be a vector (Vec<f32>) — resolve the parameter \
             to a vector literal at call time"
                .to_string(),
        )
    })?;
    if query_vec.is_empty() {
        return Err(ExecutionError::Unsupported(
            "doc_score(): query vector is empty".to_string(),
        ));
    }

    let has_chunk = "HAS_CHUNK".to_string();
    let embedding_fid = ctx.interner.lookup("embedding");

    let mut out_rows: Vec<Row> = Vec::with_capacity(rows.len());
    for row in rows {
        let doc_id = match row.get(doc_variable) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => {
                // Not a bound node — emit 0 and move on; matches "zero
                // matching chunks returns 0" spirit rather than erroring.
                let mut out = row;
                out.insert("__doc_score__".to_string(), Value::Float(0.0));
                out_rows.push(out);
                continue;
            }
        };

        // Traverse outward HAS_CHUNK edges; each neighbour is a chunk node id.
        let neighbours = expand_one_hop(
            doc_id,
            std::slice::from_ref(&has_chunk),
            Direction::Outgoing,
            ctx,
        )?;

        if neighbours.is_empty() {
            let mut out = row;
            out.insert("__doc_score__".to_string(), Value::Float(0.0));
            out_rows.push(out);
            continue;
        }

        let total = neighbours.len() as f64;
        let mut max_score: f64 = 0.0;
        let mut sum_score: f64 = 0.0;
        let mut scored_count: usize = 0;
        let mut matching: usize = 0;

        for (chunk_uid, _et_idx) in &neighbours {
            let chunk = match ctx.mvcc_get_node(ctx.shard_id, NodeId::from_raw(*chunk_uid))? {
                Some(rec) => rec,
                None => continue,
            };
            let emb = embedding_fid.and_then(|fid| chunk.get(fid)).or_else(|| {
                // Fallback: linear search by interned name (handles cases where
                // the writer interned "embedding" under a different id than
                // the reader has seen yet).
                chunk.props.iter().find_map(|(fid, v)| {
                    if ctx.interner.resolve(*fid) == Some("embedding") {
                        Some(v)
                    } else {
                        None
                    }
                })
            });
            let vec_val = match emb {
                Some(v) => v.clone(),
                None => continue,
            };
            let Some(chunk_vec) = coerce_value_to_vec(&vec_val) else {
                continue;
            };
            if chunk_vec.len() != query_vec.len() {
                continue;
            }
            let sim = coordinode_vector::metrics::cosine_similarity(&chunk_vec, &query_vec) as f64;
            if scored_count == 0 || sim > max_score {
                max_score = sim;
            }
            sum_score += sim;
            scored_count += 1;
            if sim > 0.0 {
                matching += 1;
            }
        }

        let avg_score = if scored_count > 0 {
            sum_score / scored_count as f64
        } else {
            0.0
        };
        let coverage = matching as f64 / total;
        let effective_max = if scored_count > 0 { max_score } else { 0.0 };
        let score = alpha_f * effective_max + beta_f * avg_score + gamma_f * coverage;

        let mut out = row;
        out.insert("__doc_score__".to_string(), Value::Float(score));
        out_rows.push(out);
    }

    Ok(out_rows)
}

/// Execute MaxSimTopK: late-interaction (ColBERT-style) top-K via the
/// MaxSim scalar with a bounded min-heap. Brute-force over the input
/// rows in v1; replaces the generic Sort + Limit pipeline with an
/// O(N log K) pass that never materialises the full sort buffer.
fn execute_maxsim_top_k(
    input: &LogicalOp,
    doc_expr: &Expr,
    query_expr: &Expr,
    k: usize,
    score_alias: Option<&str>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let rows = execute_op(input, ctx)?;
    if k == 0 || rows.is_empty() {
        return Ok(Vec::new());
    }

    let query_val = eval_expr(query_expr, &Row::new());
    let query = coerce_value_to_multi_vector(&query_val);
    let Some(query) = query else {
        // Mirror the scalar's degenerate behaviour: missing / malformed
        // query yields no rows rather than a hard error so a plan that
        // pre-binds a stale parameter still completes.
        return Ok(Vec::new());
    };

    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    // Wrap (score, row_index) so we can use the std BinaryHeap as a
    // bounded min-heap on score. The row index is the tie-breaker and
    // keeps the heap order deterministic when scores collide.
    #[derive(Debug)]
    struct Scored {
        score: f32,
        idx: usize,
    }
    impl Eq for Scored {}
    impl PartialEq for Scored {
        fn eq(&self, other: &Self) -> bool {
            self.score == other.score && self.idx == other.idx
        }
    }
    impl Ord for Scored {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse score for min-at-top, then natural idx order.
            other
                .score
                .partial_cmp(&self.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| self.idx.cmp(&other.idx))
        }
    }
    impl PartialOrd for Scored {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(k.saturating_add(1));
    let mut scores: Vec<f32> = Vec::with_capacity(rows.len());

    for (idx, row) in rows.iter().enumerate() {
        let doc_val = eval_expr(doc_expr, row);
        let Some(doc) = coerce_value_to_multi_vector(&doc_val) else {
            scores.push(f32::NEG_INFINITY);
            continue;
        };
        let score = coordinode_vector::metrics::maxsim(&doc, &query);
        scores.push(score);
        heap.push(Scored { score, idx });
        if heap.len() > k {
            heap.pop();
        }
    }

    // Drain into descending-score order. `into_sorted_vec` sorts
    // ascending by our reversed Ord (min-score at the top of the
    // heap), which means the natural produced order is already
    // descending by actual score.
    let picks: Vec<Scored> = heap.into_sorted_vec();

    let mut out = Vec::with_capacity(picks.len());
    for pick in picks {
        let mut row = rows[pick.idx].clone();
        if let Some(alias) = score_alias {
            row.insert(alias.to_string(), Value::Float(pick.score as f64));
        }
        out.push(row);
    }
    Ok(out)
}

/// Coerce a Value to a multi-vector matrix for the MaxSim executor.
/// Mirrors the eval-layer helper but lives next to its single caller
/// so the executor doesn't have to depend on the private eval helper.
fn coerce_value_to_multi_vector(val: &Value) -> Option<Vec<Vec<f32>>> {
    match val {
        Value::MultiVector(rows) => Some(rows.clone()),
        Value::Array(arr) => {
            let mut rows: Vec<Vec<f32>> = Vec::with_capacity(arr.len());
            for item in arr {
                let row = coerce_value_to_vec(item)?;
                rows.push(row);
            }
            let width = rows.first().map(Vec::len)?;
            if width == 0 || rows.iter().any(|r| r.len() != width) {
                return None;
            }
            Some(rows)
        }
        _ => None,
    }
}

/// Coerce a Value to Vec<f32> for vector operations in VectorFilter.
fn coerce_value_to_vec(val: &Value) -> Option<Vec<f32>> {
    match val {
        Value::Vector(v) => Some(v.clone()),
        Value::Array(arr) => {
            let mut vec = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    Value::Float(f) => vec.push(*f as f32),
                    Value::Int(i) => vec.push(*i as f32),
                    _ => return None,
                }
            }
            Some(vec)
        }
        _ => None,
    }
}

/// TextFilter: search TextIndex for matching documents, filter rows.
///
/// For each row, evaluates `text_expr` to get the text content of a node field,
/// then searches the TextIndex for matches. Keeps rows whose node_id appears
/// in the search results.
/// Build the R-HYB1b hard-fail error message for a missing full-text index.
///
/// Consistent with R-HYB1 `text_score()` guard and R-HYB2b RankFuse text-method
/// guard — `text_match()` no longer silently passes every row when the index
/// is missing (that was the old graceful-degradation bug that turned
/// `WHERE text_match(...)` into a no-op filter). Tells the user how to fix it.
fn text_match_missing_index_error(label: Option<&str>, property: Option<&str>) -> ExecutionError {
    let msg = match (label, property) {
        (Some(l), Some(p)) => format!(
            "text_match() requires a full-text index on (:{l}, {p}); \
             create one with CREATE TEXT INDEX idx_name ON :{l}({p})"
        ),
        (None, Some(p)) => format!(
            "text_match() requires a full-text index on the property `{p}`; \
             create one with CREATE TEXT INDEX idx_name ON :Label({p})"
        ),
        _ => "text_match() requires a full-text index on the text field; \
              create one with CREATE TEXT INDEX idx_name ON :Label(property)"
            .to_string(),
    };
    ExecutionError::Unsupported(msg)
}

fn execute_text_filter(
    rows: &[Row],
    text_expr: &Expr,
    query_string: &str,
    language: Option<&str>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Empty input — nothing to filter; skip the index lookup entirely so an
    // earlier operator that produced zero rows (e.g. DETACH DELETE'd the
    // whole label) does not turn into a spurious "missing FT-index" error.
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    // Try text_index_registry first (automatic mode), fallback to legacy text_index.
    // The registry is keyed by (label, property) — extract property from text_expr.
    let limit = rows.len().max(1000);
    // Extract property name from text_expr (PropertyAccess { var, property }).
    let property = match text_expr {
        Expr::PropertyAccess { property, .. } => Some(property.as_str()),
        _ => None,
    };
    // Extract label from the first row's __label__ field, if the variable is
    // bound and carries its label.
    let label_owned: Option<String> = match text_expr {
        Expr::PropertyAccess { expr, .. } => match expr.as_ref() {
            Expr::Variable(var) => rows.first().and_then(|r| {
                r.get(&format!("{var}.__label__"))
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
            }),
            _ => None,
        },
        _ => None,
    };
    let label = label_owned.as_deref();

    let search_results = if let Some(registry) = ctx.text_index_registry {
        if let (Some(l), Some(p)) = (label, property) {
            if let Some(handle) = registry.get(l, p) {
                let idx = handle
                    .read()
                    .map_err(|_| ExecutionError::Unsupported("text index lock poisoned".into()))?;
                if let Some(lang) = language {
                    idx.search_with_language(query_string, limit, lang)
                        .map_err(|e| {
                            ExecutionError::Unsupported(format!("text search error: {e}"))
                        })?
                } else {
                    idx.search(query_string, limit).map_err(|e| {
                        ExecutionError::Unsupported(format!("text search error: {e}"))
                    })?
                }
            } else {
                // R-HYB1b: registry is wired but has no index for (label, property) —
                // hard-fail, don't silently pass every row through.
                return Err(text_match_missing_index_error(Some(l), Some(p)));
            }
        } else if let Some(text_index) = ctx.text_index {
            // Registry present but we couldn't determine (label, property) from
            // the text_expr shape — fall back to the legacy single text_index
            // for back-compat with tests that wire one directly.
            if let Some(lang) = language {
                text_index
                    .search_with_language(query_string, limit, lang)
                    .map_err(|e| ExecutionError::Unsupported(format!("text search error: {e}")))?
            } else {
                text_index
                    .search(query_string, limit)
                    .map_err(|e| ExecutionError::Unsupported(format!("text search error: {e}")))?
            }
        } else {
            // R-HYB1b: neither registry lookup worked nor legacy index present.
            return Err(text_match_missing_index_error(label, property));
        }
    } else if let Some(text_index) = ctx.text_index {
        // Legacy path: single text_index passed directly.
        if let Some(lang) = language {
            text_index
                .search_with_language(query_string, limit, lang)
                .map_err(|e| ExecutionError::Unsupported(format!("text search error: {e}")))?
        } else {
            text_index
                .search(query_string, limit)
                .map_err(|e| ExecutionError::Unsupported(format!("text search error: {e}")))?
        }
    } else {
        // R-HYB1b: no registry and no legacy index — hard-fail.
        return Err(text_match_missing_index_error(label, property));
    };

    // Build a map of matching node IDs → BM25 scores
    let matching_scores: std::collections::HashMap<u64, f32> = search_results
        .iter()
        .map(|r| (r.node_id, r.score))
        .collect();

    // Filter rows: keep those whose node ID is in the match set.
    // The node ID is extracted from the text_expr's variable.
    let mut results = Vec::new();
    for row in rows {
        // Extract node ID from the row based on text_expr.
        // text_expr is typically PropertyAccess { expr: Variable("a"), property: "body" }.
        // We need the variable's ID, not the property value.
        let node_id = match text_expr {
            Expr::PropertyAccess { expr, .. } => {
                if let Expr::Variable(var) = expr.as_ref() {
                    row.get(var).and_then(|v| {
                        if let Value::Int(id) = v {
                            Some(*id as u64)
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(nid) = node_id {
            if let Some(&score) = matching_scores.get(&nid) {
                let mut out_row = row.clone();
                // Store BM25 score for text_score() access in RETURN clause
                out_row.insert("__text_score__".to_string(), Value::Float(score as f64));
                results.push(out_row);
            }
        }
    }

    Ok(results)
}

/// EncryptedFilter: SSE equality search via storage-backed token index.
///
/// For each row, extracts the variable's label and the property name from `field_expr`,
/// creates an `EncryptedIndex` on the fly from `ctx.engine`, evaluates `token_expr`
/// to get the search token bytes, and filters rows by matching node IDs.
/// Decode a hex string to bytes. Returns `None` if the string is not valid hex.
fn decode_hex_string(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut bytes = Vec::with_capacity(s.len() / 2);
    for chunk in s.as_bytes().chunks(2) {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        bytes.push((hi << 4) | lo);
    }
    Some(bytes)
}

/// Convert an ASCII hex character to its nibble value (0-15).
fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn execute_encrypted_filter(
    rows: &[Row],
    field_expr: &Expr,
    token_expr: &Expr,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_search::encrypted::{EncryptedIndex, SearchToken};

    if rows.is_empty() {
        return Ok(Vec::new());
    }

    // Evaluate token from expression (parameter or literal).
    // Use the first row for context (token expression is typically a parameter,
    // independent of row data).
    let token_val = eval_expr(token_expr, &rows[0]);
    let token_bytes = match &token_val {
        Value::Binary(b) => b.clone(),
        Value::String(s) => {
            // Accept hex-encoded token strings: decode hex → raw bytes.
            // If not valid hex (odd length or invalid chars), fall back to raw UTF-8 bytes.
            decode_hex_string(s).unwrap_or_else(|| s.as_bytes().to_vec())
        }
        _ => {
            ctx.warnings.push(
                "encrypted_match() token expression did not evaluate to Binary or String."
                    .to_string(),
            );
            return Ok(Vec::new());
        }
    };

    let search_token = match SearchToken::from_bytes(&token_bytes) {
        Some(t) => t,
        None => {
            ctx.warnings.push(format!(
                "encrypted_match() token is {} bytes, expected 32.",
                token_bytes.len()
            ));
            return Ok(Vec::new());
        }
    };

    // Extract variable name and property from field_expr (e.g., "u" and "email" from u.email).
    let (variable, property) = match field_expr {
        Expr::PropertyAccess { expr, property } => {
            if let Expr::Variable(var) = expr.as_ref() {
                (var.clone(), property.clone())
            } else {
                return Ok(rows.to_vec()); // can't determine variable
            }
        }
        _ => return Ok(rows.to_vec()),
    };

    // Extract label from the first row's __label__ field.
    let label = rows
        .first()
        .and_then(|r| {
            r.get(&format!("{variable}.__label__"))
                .and_then(|v| v.as_str().map(|s| s.to_string()))
        })
        .unwrap_or_default();

    if label.is_empty() {
        ctx.warnings.push(format!(
            "encrypted_match() could not determine label for variable '{variable}'."
        ));
        return Ok(rows.to_vec());
    }

    // Create storage-backed SSE index handle on the fly (cheap — just
    // holds the (label, field) scoping strings). The search reads
    // through the active transaction's committed snapshot.
    let index = EncryptedIndex::new(&label, &property);
    let matching_ids = index
        .search(&ctx.txn, &search_token)
        .map_err(|e| ExecutionError::Unsupported(format!("encrypted search error: {e}")))?;

    // Build a set for O(1) lookup.
    let matching_set: std::collections::HashSet<u64> = matching_ids.into_iter().collect();

    // Filter rows: keep those whose node ID is in the match set.
    let mut results = Vec::new();
    for row in rows {
        let node_id = row.get(&variable).and_then(|v| {
            if let Value::Int(id) = v {
                Some(*id as u64)
            } else {
                None
            }
        });

        if let Some(nid) = node_id {
            if matching_set.contains(&nid) {
                results.push(row.clone());
            }
        }
    }

    Ok(results)
}

/// UNWIND: expand a list expression into individual rows.
///
/// For each input row, evaluate `expr`. If it yields a list, create one output
/// row per element with the element bound to `variable`. Non-list values are
/// treated as single-element lists. NULL expands to zero rows.
fn execute_unwind(rows: &[Row], expr: &Expr, variable: &str) -> Result<Vec<Row>, ExecutionError> {
    let mut results = Vec::new();

    for row in rows {
        let val = eval_expr(expr, row);
        match val {
            Value::Array(items) => {
                for item in items {
                    let mut out = row.clone();
                    out.insert(variable.to_string(), item);
                    results.push(out);
                }
            }
            Value::Null => {
                // UNWIND on NULL produces zero rows (standard OpenCypher)
            }
            other => {
                // Non-list scalar: treat as single-element list
                let mut out = row.clone();
                out.insert(variable.to_string(), other);
                results.push(out);
            }
        }
    }

    Ok(results)
}

/// Left outer join for OPTIONAL MATCH.
///
/// Two execution modes:
/// - **Non-correlated** (common): right side executes once, join by shared
///   variable matching. O(left + right + join).
/// - **Correlated** (detected automatically): right side executes per left
///   row with `ctx.correlated_row` set so Filter can resolve outer-scope
///   variables. Required for `WHERE c.age > a.age` where `a` is from the
///   outer MATCH scope. O(left × right).
fn execute_left_outer_join(
    left_rows: &[Row],
    right_op: &LogicalOp,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let right_vars = collect_introduced_variables(right_op);
    let correlated = needs_correlated_execution(right_op);

    if correlated {
        execute_left_outer_join_correlated(left_rows, right_op, &right_vars, ctx)
    } else {
        execute_left_outer_join_global(left_rows, right_op, &right_vars, ctx)
    }
}

/// Non-correlated path: execute right side once, join by shared variables.
fn execute_left_outer_join_global(
    left_rows: &[Row],
    right_op: &LogicalOp,
    right_vars: &[String],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let right_rows = execute_op(right_op, ctx)?;
    let mut results = Vec::new();

    for left_row in left_rows {
        let mut matched = false;
        for rr in &right_rows {
            let shared_match = rr.iter().all(|(key, rval)| match left_row.get(key) {
                Some(lval) => lval == rval,
                None => true,
            });

            if shared_match {
                let mut merged = left_row.clone();
                merged.extend(rr.clone());
                results.push(merged);
                matched = true;
            }
        }

        if !matched {
            let mut out = left_row.clone();
            for var in right_vars {
                out.entry(var.clone()).or_insert(Value::Null);
            }
            results.push(out);
        }
    }

    Ok(results)
}

/// Correlated path: execute right side per left row with outer-scope variables.
fn execute_left_outer_join_correlated(
    left_rows: &[Row],
    right_op: &LogicalOp,
    right_vars: &[String],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let prev_correlated = ctx.correlated_row.take();
    let mut results = Vec::new();

    for left_row in left_rows {
        ctx.correlated_row = Some(left_row.clone());

        let right_rows = execute_op(right_op, ctx)?;

        let mut matched = false;
        for rr in &right_rows {
            let shared_match = rr.iter().all(|(key, rval)| match left_row.get(key) {
                Some(lval) => lval == rval,
                None => true,
            });

            if shared_match {
                let mut merged = left_row.clone();
                merged.extend(rr.clone());
                results.push(merged);
                matched = true;
            }
        }

        if !matched {
            let mut out = left_row.clone();
            for var in right_vars {
                out.entry(var.clone()).or_insert(Value::Null);
            }
            results.push(out);
        }
    }

    ctx.correlated_row = prev_correlated;
    Ok(results)
}

/// Check if the right side of a LeftOuterJoin needs correlated (per-row)
/// execution. Returns true when filter predicates reference variables
/// not introduced by the right side itself.
fn needs_correlated_execution(right_op: &LogicalOp) -> bool {
    let introduced: Vec<String> = collect_introduced_variables(right_op);
    let mut predicate_vars = Vec::new();
    collect_filter_variables(right_op, &mut predicate_vars);
    predicate_vars.iter().any(|v| !introduced.contains(v))
}

/// Collect variable names referenced in Filter predicates within an operator tree.
fn collect_filter_variables(op: &LogicalOp, vars: &mut Vec<String>) {
    match op {
        LogicalOp::Filter { input, predicate } => {
            collect_expr_vars(predicate, vars);
            collect_filter_variables(input, vars);
        }
        LogicalOp::Traverse {
            input,
            target_filters,
            edge_filters,
            ..
        } => {
            for (_, expr) in target_filters {
                collect_expr_vars(expr, vars);
            }
            for (_, expr) in edge_filters {
                collect_expr_vars(expr, vars);
            }
            collect_filter_variables(input, vars);
        }
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            collect_filter_variables(left, vars);
            collect_filter_variables(right, vars);
        }
        _ => {}
    }
}

/// Extract variable names from an expression tree.
/// Covers all `Expr` variants that can contain nested variables.
fn collect_expr_vars(expr: &Expr, vars: &mut Vec<String>) {
    match expr {
        Expr::Variable(name) => vars.push(name.clone()),
        Expr::PropertyAccess { expr, .. } => collect_expr_vars(expr, vars),
        Expr::BinaryOp { left, right, .. } => {
            collect_expr_vars(left, vars);
            collect_expr_vars(right, vars);
        }
        Expr::UnaryOp { expr, .. } => collect_expr_vars(expr, vars),
        Expr::FunctionCall { args, .. } => {
            for arg in args {
                collect_expr_vars(arg, vars);
            }
        }
        Expr::List(items) => {
            for item in items {
                collect_expr_vars(item, vars);
            }
        }
        Expr::MapLiteral(entries) => {
            for (_, val) in entries {
                collect_expr_vars(val, vars);
            }
        }
        Expr::MapProjection { expr, items } => {
            collect_expr_vars(expr, vars);
            for item in items {
                if let crate::cypher::ast::MapProjectionItem::Computed(_, e) = item {
                    collect_expr_vars(e, vars);
                }
            }
        }
        Expr::In { expr, list } => {
            collect_expr_vars(expr, vars);
            collect_expr_vars(list, vars);
        }
        Expr::IsNull { expr, .. } => collect_expr_vars(expr, vars),
        Expr::IsTyped { expr, .. } => collect_expr_vars(expr, vars),
        Expr::StringMatch { expr, pattern, .. } => {
            collect_expr_vars(expr, vars);
            collect_expr_vars(pattern, vars);
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                collect_expr_vars(op, vars);
            }
            for (cond, then) in when_clauses {
                collect_expr_vars(cond, vars);
                collect_expr_vars(then, vars);
            }
            if let Some(el) = else_clause {
                collect_expr_vars(el, vars);
            }
        }
        Expr::PatternPredicate(pattern) => {
            for elem in &pattern.elements {
                match elem {
                    crate::cypher::ast::PatternElement::Node(node) => {
                        if let Some(ref name) = node.variable {
                            vars.push(name.clone());
                        }
                        for (_, v) in &node.properties {
                            collect_expr_vars(v, vars);
                        }
                    }
                    crate::cypher::ast::PatternElement::Relationship(rel) => {
                        if let Some(ref name) = rel.variable {
                            vars.push(name.clone());
                        }
                        for (_, v) in &rel.properties {
                            collect_expr_vars(v, vars);
                        }
                    }
                }
            }
        }
        Expr::Subscript { expr, index } => {
            collect_expr_vars(expr, vars);
            collect_expr_vars(index, vars);
        }
        Expr::Slice { expr, start, end } => {
            collect_expr_vars(expr, vars);
            if let Some(s) = start {
                collect_expr_vars(s, vars);
            }
            if let Some(e) = end {
                collect_expr_vars(e, vars);
            }
        }
        Expr::Reduce {
            acc,
            init,
            var,
            list,
            expr,
        } => {
            collect_expr_vars(init, vars);
            collect_expr_vars(list, vars);
            // acc and var are bound locally inside the fold — exclude them from
            // the outer variables the step expression depends on.
            let mut inner = Vec::new();
            collect_expr_vars(expr, &mut inner);
            vars.extend(inner.into_iter().filter(|v| v != acc && v != var));
        }
        Expr::ListPredicate {
            var, list, pred, ..
        } => {
            collect_expr_vars(list, vars);
            // var is bound locally — exclude it from the predicate's outer deps.
            let mut inner = Vec::new();
            collect_expr_vars(pred, &mut inner);
            vars.extend(inner.into_iter().filter(|v| v != var));
        }
        Expr::ListComprehension {
            var,
            list,
            pred,
            map,
        } => {
            collect_expr_vars(list, vars);
            let mut inner = Vec::new();
            if let Some(p) = pred {
                collect_expr_vars(p, &mut inner);
            }
            if let Some(m) = map {
                collect_expr_vars(m, &mut inner);
            }
            vars.extend(inner.into_iter().filter(|v| v != var));
        }
        // The inner MATCH binds its own variables; any outer-correlation
        // variables it references are already provisioned by the outer clauses,
        // so it contributes no extra outer dependencies here.
        Expr::ExistsSubquery(_)
        | Expr::CountSubquery(_)
        | Expr::CollectSubquery { .. }
        | Expr::PatternComprehension { .. } => {}
        // Literal, Parameter, Star — no variable references.
        Expr::Literal(_) | Expr::Parameter(_) | Expr::Star => {}
    }
}

/// Collect variable names introduced by a logical operator.
/// Used by OPTIONAL MATCH to know which variables to set to NULL.
fn collect_introduced_variables(op: &LogicalOp) -> Vec<String> {
    match op {
        LogicalOp::NodeScan { variable, .. } => vec![variable.clone()],
        LogicalOp::Traverse {
            input,
            target_variable,
            edge_variable,
            ..
        } => {
            let mut vars = collect_introduced_variables(input);
            vars.push(target_variable.clone());
            if let Some(ev) = edge_variable {
                vars.push(ev.clone());
            }
            vars
        }
        LogicalOp::Filter { input, .. } => collect_introduced_variables(input),
        LogicalOp::CartesianProduct { left, right } => {
            let mut vars = collect_introduced_variables(left);
            vars.extend(collect_introduced_variables(right));
            vars
        }
        _ => Vec::new(),
    }
}

/// BFS-based shortest path between two bound nodes.
///
/// Finds the shortest (unweighted) path from source to target, traversing
/// the specified edge types. Returns path length as an integer.
///
/// Parameters for shortest path computation.
struct ShortestPathParams<'a> {
    source: &'a str,
    target: &'a str,
    edge_types: &'a [String],
    direction: Direction,
    max_depth: u64,
    path_variable: &'a str,
}

/// Inspired by Dgraph's `shortest.go` Dijkstra/BFS hybrid, simplified
/// for unweighted single-pair shortest path.
fn execute_shortest_path(
    rows: &[Row],
    sp: &ShortestPathParams<'_>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use std::collections::VecDeque;

    let max_d = sp.max_depth.min(DEFAULT_MAX_HOPS) as usize;
    let mut results = Vec::new();

    for row in rows {
        let src_uid = match row.get(sp.source) {
            Some(Value::Int(id)) => *id as u64,
            _ => continue,
        };
        let tgt_uid = match row.get(sp.target) {
            Some(Value::Int(id)) => *id as u64,
            _ => continue,
        };

        // BFS from src_uid to tgt_uid, recording each node's predecessor so the
        // actual path can be reconstructed. Keys are internal ids, so a fast
        // non-DoS hasher fits this hot loop. `pred[n]` is None for the source
        // and Some((predecessor, edge_type_idx)) otherwise.
        let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
        let mut pred: rustc_hash::FxHashMap<u64, Option<(u64, usize)>> =
            rustc_hash::FxHashMap::default();

        queue.push_back((src_uid, 0));
        pred.insert(src_uid, None);

        let mut found = false;
        while let Some((uid, depth)) = queue.pop_front() {
            if uid == tgt_uid {
                found = true;
                break;
            }
            if depth >= max_d {
                continue;
            }

            let nid = NodeId::from_raw(uid);
            let neighbors = expand_one_hop(nid, sp.edge_types, sp.direction, ctx)?;

            for (neighbor_uid, et_idx) in neighbors {
                if let std::collections::hash_map::Entry::Vacant(e) = pred.entry(neighbor_uid) {
                    e.insert(Some((uid, et_idx)));
                    queue.push_back((neighbor_uid, depth + 1));
                }
            }
        }

        let mut out = row.clone();
        if found {
            // Walk predecessors back from tgt to src, then reverse into a
            // forward node/relationship sequence. A zero-length path (src ==
            // tgt) yields a single node and no relationships.
            let mut back: Vec<(u64, usize)> = Vec::new();
            let mut cur = tgt_uid;
            while let Some(Some((p, et))) = pred.get(&cur).copied() {
                back.push((cur, et));
                cur = p;
            }
            back.reverse();

            let mut nodes = Vec::with_capacity(back.len() + 1);
            let mut rels = Vec::with_capacity(back.len());
            nodes.push(src_uid);
            let mut prev = src_uid;
            for (node, et_idx) in back {
                let edge_type = sp.edge_types.get(et_idx).cloned().unwrap_or_default();
                rels.push(coordinode_core::graph::types::PathRel {
                    edge_type,
                    source: prev,
                    target: node,
                });
                nodes.push(node);
                prev = node;
            }
            out.insert(
                sp.path_variable.to_string(),
                Value::Path(coordinode_core::graph::types::PathValue { nodes, rels }),
            );
        } else {
            out.insert(sp.path_variable.to_string(), Value::Null);
        }
        results.push(out);
    }

    Ok(results)
}

/// Evaluate a scalar expression that may be a literal or a query parameter.
///
/// Used for aggregate function arguments (e.g., the `p` in `percentileCont(x, p)`)
/// where the value is expected to be a numeric constant or a bound parameter (`$p`).
/// Returns `None` for complex expressions or unresolvable/non-numeric values.
fn eval_scalar_expr(
    expr: &Expr,
    params: &HashMap<String, coordinode_core::graph::types::Value>,
) -> Option<f64> {
    match expr {
        Expr::Literal(Value::Float(f)) => Some(*f),
        Expr::Literal(Value::Int(i)) => Some(*i as f64),
        Expr::Parameter(name) => params.get(name).and_then(|v| match v {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }),
        _ => None,
    }
}

/// Execute aggregation: group rows and compute aggregate functions.
fn execute_aggregate(
    rows: &[Row],
    group_by: &[Expr],
    aggregates: &[AggregateItem],
    params: &HashMap<String, coordinode_core::graph::types::Value>,
) -> Result<Vec<Row>, ExecutionError> {
    // Group rows by group-by key
    // Group rows by group-by key.
    // Value doesn't implement Ord/Hash, so we use linear search for grouping.
    let mut groups: Vec<(Vec<Value>, Vec<&Row>)> = Vec::new();

    if group_by.is_empty() {
        // No group-by: all rows form a single group
        let all: Vec<&Row> = rows.iter().collect();
        groups.push((Vec::new(), all));
    } else {
        for row in rows {
            let key: Vec<Value> = group_by.iter().map(|e| eval_expr(e, row)).collect();
            let found = groups.iter_mut().find(|(k, _)| k == &key);
            if let Some((_, group_rows)) = found {
                group_rows.push(row);
            } else {
                groups.push((key, vec![row]));
            }
        }
    }

    let mut results = Vec::new();

    for (key, group_rows) in &groups {
        let mut out = Row::new();

        // Add group-by values
        for (i, expr) in group_by.iter().enumerate() {
            let val = key.get(i).cloned().unwrap_or(Value::Null);
            let col = expr_display_name(expr);
            out.insert(col, val);
        }

        // Compute aggregates
        for agg in aggregates {
            let val = compute_aggregate(agg, group_rows, params);
            let col = agg.alias.clone().unwrap_or_else(|| agg.function.clone());
            out.insert(col, val);
        }

        results.push(out);
    }

    Ok(results)
}

/// Evaluate aggregate argument for all rows, applying DISTINCT dedup if needed.
/// Returns non-null values only.
fn eval_aggregate_values(agg: &AggregateItem, rows: &[&Row]) -> Vec<Value> {
    let mut values: Vec<Value> = rows
        .iter()
        .map(|r| eval_expr(&agg.arg, r))
        .filter(|v| !v.is_null())
        .collect();

    if agg.distinct {
        // Deduplicate using linear scan (Value doesn't impl Hash).
        let mut unique = Vec::with_capacity(values.len());
        for v in values {
            if !unique.contains(&v) {
                unique.push(v);
            }
        }
        values = unique;
    }

    values
}

/// Compute a single aggregate function over a group of rows.
fn compute_aggregate(
    agg: &AggregateItem,
    rows: &[&Row],
    params: &HashMap<String, coordinode_core::graph::types::Value>,
) -> Value {
    match agg.function.as_str() {
        "count" => {
            if agg.arg == Expr::Star {
                // count(*) ignores DISTINCT — counts all rows
                Value::Int(rows.len() as i64)
            } else {
                let values = eval_aggregate_values(agg, rows);
                Value::Int(values.len() as i64)
            }
        }
        "sum" => {
            let values = eval_aggregate_values(agg, rows);
            let mut int_sum: i64 = 0;
            let mut float_sum: f64 = 0.0;
            let mut has_float = false;
            let mut has_value = false;
            for v in &values {
                match v {
                    Value::Int(n) => {
                        int_sum = int_sum.wrapping_add(*n);
                        float_sum += *n as f64;
                        has_value = true;
                    }
                    Value::Float(f) => {
                        float_sum += f;
                        has_float = true;
                        has_value = true;
                    }
                    _ => {}
                }
            }
            if !has_value {
                Value::Null
            } else if has_float {
                Value::Float(float_sum)
            } else {
                Value::Int(int_sum)
            }
        }
        "avg" => {
            let values = eval_aggregate_values(agg, rows);
            let mut sum = 0.0f64;
            let mut count = 0u64;
            for v in &values {
                match v {
                    Value::Int(n) => {
                        sum += *n as f64;
                        count += 1;
                    }
                    Value::Float(f) => {
                        sum += f;
                        count += 1;
                    }
                    _ => {}
                }
            }
            if count > 0 {
                Value::Float(sum / count as f64)
            } else {
                Value::Null
            }
        }
        "min" => {
            let values = eval_aggregate_values(agg, rows);
            values
                .into_iter()
                .reduce(|a, b| {
                    if compare_values(&b, &a) == std::cmp::Ordering::Less {
                        b
                    } else {
                        a
                    }
                })
                .unwrap_or(Value::Null)
        }
        "max" => {
            let values = eval_aggregate_values(agg, rows);
            values
                .into_iter()
                .reduce(|a, b| {
                    if compare_values(&b, &a) == std::cmp::Ordering::Greater {
                        b
                    } else {
                        a
                    }
                })
                .unwrap_or(Value::Null)
        }
        "collect" => {
            let values = eval_aggregate_values(agg, rows);
            Value::Array(values)
        }
        "percentileCont" | "percentileDisc" => {
            let agg_values = eval_aggregate_values(agg, rows);
            let mut values: Vec<f64> = agg_values
                .iter()
                .filter_map(|v| match v {
                    Value::Int(n) => Some(*n as f64),
                    Value::Float(f) => Some(*f),
                    _ => None,
                })
                .collect();

            if values.is_empty() {
                return Value::Null;
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Percentile from the second argument expression; supports literals and $params.
            // Falls back to 0.5 (median) when the argument is absent or not a numeric scalar.
            let percentile = agg
                .percentile_expr
                .as_ref()
                .and_then(|e| eval_scalar_expr(e, params))
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);

            if agg.function == "percentileDisc" {
                // Nearest rank method: ceil(p * n) gives 1-based index; clamp to [0, n-1].
                let idx = ((percentile * values.len() as f64).ceil() as usize)
                    .saturating_sub(1)
                    .min(values.len() - 1);
                Value::Float(values[idx])
            } else {
                // Linear interpolation (percentileCont)
                let rank = percentile * (values.len() - 1) as f64;
                let lower = rank.floor() as usize;
                let upper = rank.ceil() as usize;
                let frac = rank - lower as f64;

                if lower == upper || upper >= values.len() {
                    Value::Float(values[lower])
                } else {
                    Value::Float(values[lower] * (1.0 - frac) + values[upper] * frac)
                }
            }
        }
        "stDev" | "stDevP" => {
            let agg_values = eval_aggregate_values(agg, rows);
            let values: Vec<f64> = agg_values
                .iter()
                .filter_map(|v| match v {
                    Value::Int(n) => Some(*n as f64),
                    Value::Float(f) => Some(*f),
                    _ => None,
                })
                .collect();

            if values.is_empty() {
                return Value::Null;
            }

            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>();

            if agg.function == "stDevP" {
                // Population standard deviation
                Value::Float((variance / n).sqrt())
            } else {
                // Sample standard deviation
                if values.len() < 2 {
                    Value::Float(0.0)
                } else {
                    Value::Float((variance / (n - 1.0)).sqrt())
                }
            }
        }
        _ => Value::Null,
    }
}

/// Compare two values for sorting.
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (Value::Int(a), Value::Float(b)) => (*a as f64)
            .partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal),
        (Value::Float(a), Value::Int(b)) => a
            .partial_cmp(&(*b as f64))
            .unwrap_or(std::cmp::Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Timestamp(a), Value::Timestamp(b)) => a.cmp(b),
        (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
        (Value::Null, _) => std::cmp::Ordering::Greater, // NULLs sort last
        (_, Value::Null) => std::cmp::Ordering::Less,
        _ => std::cmp::Ordering::Equal,
    }
}

// --- Write operations ---

/// MERGE: match pattern → if found apply ON MATCH SET, if not found create + ON CREATE SET.
fn execute_merge(
    pattern: &LogicalOp,
    on_match: &[crate::cypher::ast::SetItem],
    on_create: &[crate::cypher::ast::SetItem],
    multi: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // G072: MERGE (src)-[r:TYPE]->(tgt) — relationship pattern with correlated bindings.
    // When the pattern is a Traverse and ctx.correlated_row has src/tgt bound, use the
    // targeted match+create path instead of the generic execute_op + execute_create_from_pattern.
    // The generic path scans all nodes and fails to create edges from Traverse patterns.
    // MERGE ALL with correlated row: same per-pair logic (the CartesianProduct executor
    // feeds each (src, tgt) pair as a correlated row, so multi=true behaves identically here).
    if let (Some(traverse), Some(correlated)) =
        (as_traverse_op(pattern), ctx.correlated_row.clone())
    {
        let matches = execute_merge_relationship_check(traverse, &correlated, ctx)?;
        if !matches.is_empty() {
            if on_match.is_empty() {
                return Ok(matches);
            }
            return execute_update(&matches, on_match, &ViolationMode::Fail, ctx);
        }
        let created = execute_merge_relationship_create(traverse, &correlated, ctx)?;
        if on_create.is_empty() {
            return Ok(created);
        }
        return execute_update(&created, on_create, &ViolationMode::Fail, ctx);
    }

    // G077: Standalone MERGE ALL — Cartesian product across all matching src × tgt nodes.
    // Pattern: MERGE ALL (a:L {k:v})-[r:T]->(b:L {k:v})
    // Algorithm:
    //   1. Find or create source nodes (all of them).
    //   2. Find or create target nodes (all of them).
    //   3. For each (src, tgt) pair: find-or-create the relationship individually.
    if multi {
        if let Some(traverse) = as_traverse_op(pattern) {
            return execute_mergemany_standalone(traverse, on_match, on_create, ctx);
        }
    }

    // G074: Standalone relationship MERGE — no correlated_row (no preceding MATCH).
    // Pattern: MERGE (a:L {k:v})-[r:T]->(b:L {k:v})
    // Algorithm:
    //   1. Try to find the complete existing path via execute_op (full graph scan).
    //   2. If found → ON MATCH path (reuse existing nodes and edge).
    //   3. If not found → find-or-create src node, find-or-create tgt node,
    //      then create the edge between them → ON CREATE path.
    if let Some(traverse) = as_traverse_op(pattern) {
        let matches = execute_op(pattern, ctx)?;
        if !matches.is_empty() {
            if on_match.is_empty() {
                return Ok(matches);
            }
            return execute_update(&matches, on_match, &ViolationMode::Fail, ctx);
        }
        let created = execute_merge_relationship_standalone_create(traverse, ctx)?;
        if on_create.is_empty() {
            return Ok(created);
        }
        return execute_update(&created, on_create, &ViolationMode::Fail, ctx);
    }

    // Generic path: NodeScan (node-only MERGE) — find or create a single node.
    let matches = execute_op(pattern, ctx)?;

    if !matches.is_empty() {
        // Pattern found — apply ON MATCH SET items
        if on_match.is_empty() {
            return Ok(matches);
        }
        execute_update(&matches, on_match, &ViolationMode::Fail, ctx)
    } else {
        // Pattern not found — create new node from pattern, then apply ON CREATE SET
        let created = execute_create_from_pattern(pattern, ctx)?;
        if on_create.is_empty() {
            return Ok(created);
        }
        execute_update(&created, on_create, &ViolationMode::Fail, ctx)
    }
}

/// MERGE ALL standalone: Cartesian product of all matching src × tgt nodes.
///
/// For each (src, tgt) pair from all matching nodes, find-or-create the relationship.
/// If no src nodes exist → create one; if no tgt nodes exist → create one.
/// Unlike MERGE, multiple matching nodes are NOT an error — this is the intended semantics.
fn execute_mergemany_standalone(
    traverse: &LogicalOp,
    on_match: &[crate::cypher::ast::SetItem],
    on_create: &[crate::cypher::ast::SetItem],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let (input, target_variable, target_labels, target_filters) = match traverse {
        LogicalOp::Traverse {
            input,
            target_variable,
            target_labels,
            target_filters,
            ..
        } => (input, target_variable, target_labels, target_filters),
        _ => unreachable!("execute_mergemany_standalone: not a Traverse"),
    };

    // Step 1: Collect all matching source nodes (or create one if none exist).
    let src_rows = execute_op(input, ctx)?;
    let src_rows = if src_rows.is_empty() {
        execute_create_from_pattern(input, ctx)?
    } else {
        src_rows
    };

    // Step 2: Collect all matching target nodes (or create one if none exist).
    let target_scan = LogicalOp::NodeScan {
        variable: target_variable.clone(),
        labels: target_labels.clone(),
        property_filters: target_filters.clone(),
    };
    let tgt_rows = execute_op(&target_scan, ctx)?;
    let tgt_rows = if tgt_rows.is_empty() {
        execute_create_from_pattern(&target_scan, ctx)?
    } else {
        tgt_rows
    };

    // Step 3: For each (src, tgt) pair, find-or-create the relationship.
    let mut all_rows: Vec<Row> = Vec::new();
    for src_row in &src_rows {
        for tgt_row in &tgt_rows {
            let mut correlated = src_row.clone();
            correlated.extend(tgt_row.clone());

            let matches = execute_merge_relationship_check(traverse, &correlated, ctx)?;
            let pair_rows = if !matches.is_empty() {
                // Relationship already exists — ON MATCH path.
                if on_match.is_empty() {
                    matches
                } else {
                    execute_update(&matches, on_match, &ViolationMode::Fail, ctx)?
                }
            } else {
                // Relationship absent — create it → ON CREATE path.
                let created = execute_merge_relationship_create(traverse, &correlated, ctx)?;
                if on_create.is_empty() {
                    created
                } else {
                    execute_update(&created, on_create, &ViolationMode::Fail, ctx)?
                }
            };
            all_rows.extend(pair_rows);
        }
    }

    Ok(all_rows)
}

/// Returns true if op is a `Merge` whose inner pattern is (or wraps) a `Traverse`.
///
/// Used by `CartesianProduct` to detect `MATCH (a), (b) MERGE (a)-[r:T]->(b)` patterns
/// that need correlated per-left-row execution so the Merge can access bound variables.
fn is_relationship_merge(op: &LogicalOp) -> bool {
    match op {
        LogicalOp::Merge { pattern, .. } => as_traverse_op(pattern).is_some(),
        _ => false,
    }
}

/// True if `op`'s subtree contains a `NodeScan` whose inline property
/// filter references a variable NOT bound within `op` itself — i.e. the
/// filter is correlated with an outer (left) input and the scan must run
/// per-left-row to resolve it. A genuinely self-contained pattern returns
/// false and keeps the fast global cross-product path.
fn right_has_correlated_filter(op: &LogicalOp) -> bool {
    let mut bound = std::collections::HashSet::new();
    collect_bound_vars(op, &mut bound);
    scan_filter_references_outside(op, &bound)
}

fn collect_bound_vars(op: &LogicalOp, out: &mut std::collections::HashSet<String>) {
    match op {
        LogicalOp::NodeScan { variable, .. } | LogicalOp::IndexScan { variable, .. } => {
            out.insert(variable.clone());
        }
        LogicalOp::Traverse {
            input,
            target_variable,
            edge_variable,
            ..
        } => {
            collect_bound_vars(input, out);
            out.insert(target_variable.clone());
            if let Some(ev) = edge_variable {
                out.insert(ev.clone());
            }
        }
        LogicalOp::Unwind {
            input, variable, ..
        } => {
            collect_bound_vars(input, out);
            out.insert(variable.clone());
        }
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            collect_bound_vars(left, out);
            collect_bound_vars(right, out);
        }
        LogicalOp::Filter { input, .. }
        | LogicalOp::VectorFilter { input, .. }
        | LogicalOp::TextFilter { input, .. }
        | LogicalOp::Aggregate { input, .. }
        | LogicalOp::Project { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. } => collect_bound_vars(input, out),
        _ => {}
    }
}

fn scan_filter_references_outside(
    op: &LogicalOp,
    bound: &std::collections::HashSet<String>,
) -> bool {
    match op {
        LogicalOp::NodeScan {
            property_filters, ..
        } => property_filters
            .iter()
            .any(|(_, expr)| expr_references_outside(expr, bound)),
        // A correlated index point-lookup carries its key in `value_expr`;
        // it must drive per-outer-row execution just like a correlated scan.
        LogicalOp::IndexScan { value_expr, .. } => expr_references_outside(value_expr, bound),
        LogicalOp::Traverse { input, .. }
        | LogicalOp::Filter { input, .. }
        | LogicalOp::VectorFilter { input, .. }
        | LogicalOp::TextFilter { input, .. }
        | LogicalOp::Aggregate { input, .. }
        | LogicalOp::Project { input, .. }
        | LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. }
        | LogicalOp::Unwind { input, .. } => scan_filter_references_outside(input, bound),
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            scan_filter_references_outside(left, bound)
                || scan_filter_references_outside(right, bound)
        }
        _ => false,
    }
}

/// True if `expr` references any `Variable` not present in `bound`.
fn expr_references_outside(expr: &Expr, bound: &std::collections::HashSet<String>) -> bool {
    match expr {
        Expr::Variable(name) => !bound.contains(name),
        Expr::PropertyAccess { expr, .. } | Expr::UnaryOp { expr, .. } => {
            expr_references_outside(expr, bound)
        }
        Expr::BinaryOp { left, right, .. } => {
            expr_references_outside(left, bound) || expr_references_outside(right, bound)
        }
        Expr::FunctionCall { args, .. } | Expr::List(args) => {
            args.iter().any(|e| expr_references_outside(e, bound))
        }
        _ => false,
    }
}

/// Returns a reference to the innermost `Traverse` op if `op` is Traverse or a chain of
/// Filter wrappers over a Traverse (e.g., Filter { input: Traverse { .. } }).
fn as_traverse_op(op: &LogicalOp) -> Option<&LogicalOp> {
    match op {
        LogicalOp::Traverse { .. } => Some(op),
        LogicalOp::Filter { input, .. } => as_traverse_op(input),
        _ => None,
    }
}

/// Check whether the relationship described by `traverse` already exists between the
/// source and target nodes bound in `correlated`.
///
/// Returns a non-empty Vec (one row with the edge variable set) if the edge exists,
/// or an empty Vec if it does not. Called from `execute_merge` for the correlated path.
fn execute_merge_relationship_check(
    traverse: &LogicalOp,
    correlated: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let (source, edge_types, direction, target_variable, edge_variable, edge_filters) =
        match traverse {
            LogicalOp::Traverse {
                source,
                edge_types,
                direction,
                target_variable,
                edge_variable,
                edge_filters,
                ..
            } => (
                source,
                edge_types,
                direction,
                target_variable,
                edge_variable,
                edge_filters,
            ),
            _ => unreachable!("execute_merge_relationship_check: not a Traverse"),
        };

    let source_id = match correlated.get(source.as_str()) {
        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
        _ => {
            return Err(ExecutionError::Unsupported(format!(
                "MERGE relationship: source variable '{source}' not bound in scope"
            )))
        }
    };
    let target_id = match correlated.get(target_variable.as_str()) {
        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
        _ => {
            return Err(ExecutionError::Unsupported(format!(
                "MERGE relationship: target variable '{target_variable}' not bound in scope"
            )))
        }
    };

    // Resolve wildcard edge types against schema.
    let resolved_types: Vec<String>;
    let effective_types: &[String] = if edge_types.is_empty() {
        resolved_types = ctx.list_edge_types()?;
        &resolved_types
    } else {
        edge_types
    };

    let target_raw = target_id.as_raw();
    // Reject MERGE on temporal edge types: the (src, tgt) pair can carry many
    // versions, and MERGE's "match by adj-posting existence" semantics would
    // either silently no-op when versions exist (wrong for new-version intent)
    // or duplicate-create (wrong for idempotent intent). Until per-version
    // MERGE semantics ship, force users to pick a concrete operation.
    for et in effective_types {
        if lookup_edge_type_temporal(et, ctx)? {
            return Err(ExecutionError::Unsupported(format!(
                "MERGE on temporal edge type '{et}' is not supported: temporal \
                 edges have multiple versions per (src, tgt) pair, so MERGE's \
                 single-existence semantics don't apply. Use CREATE to add a \
                 new version, or MATCH + SET / DELETE to update / remove an \
                 existing one."
            )));
        }
    }
    for et in effective_types {
        let neighbors = expand_one_hop(source_id, std::slice::from_ref(et), *direction, ctx)?;
        if !neighbors.iter().any(|(tgt, _)| *tgt == target_raw) {
            continue;
        }

        // Edge (src → tgt) exists in adjacency list.
        // G075: if edge_filters are specified, also verify that the stored edge properties
        // match. Two MERGEs with different property values for the same (src, tgt, type) are
        // treated as distinct — no match if properties differ (since the data model stores
        // one EdgeProp record per (type, src, tgt), a mismatch means the edge's current
        // properties don't satisfy this MERGE pattern).
        if !edge_filters.is_empty() {
            let (ep_src, ep_tgt) = match direction {
                Direction::Outgoing | Direction::Both => (source_id, target_id),
                Direction::Incoming => (target_id, source_id),
            };
            // Load stored edge properties into a flat name→value map.
            let mut stored: std::collections::HashMap<String, Value> =
                std::collections::HashMap::new();
            if let Some(prop_map) = ctx.mvcc_get_edge_props(et, ep_src, ep_tgt)? {
                for (field_id, value) in prop_map {
                    if let Some(field_name) = ctx.interner.resolve(field_id) {
                        stored.insert(field_name.to_string(), value);
                    }
                }
            }
            // All filter expressions must match stored values.
            let filters_match = edge_filters.iter().all(|(prop_name, filter_expr)| {
                let actual = stored.get(prop_name).cloned().unwrap_or(Value::Null);
                let expected = eval_expr(filter_expr, correlated);
                actual == expected
            });
            if !filters_match {
                // This edge exists but has different property values — treat as no match.
                continue;
            }
        }

        // Edge found and all property filters match.
        let mut row = correlated.clone();
        if let Some(ev) = edge_variable {
            row.insert(format!("{ev}.__type__"), Value::String(et.clone()));
            row.insert(ev.clone(), Value::String(et.clone()));
            // Populate stored edge properties into the result row so ON MATCH SET can
            // reference them via `r.prop_name`.
            if !edge_filters.is_empty() {
                let (ep_src, ep_tgt) = match direction {
                    Direction::Outgoing | Direction::Both => (source_id, target_id),
                    Direction::Incoming => (target_id, source_id),
                };
                if let Some(prop_map) = ctx.mvcc_get_edge_props(et, ep_src, ep_tgt)? {
                    for (field_id, value) in prop_map {
                        if let Some(field_name) = ctx.interner.resolve(field_id) {
                            row.insert(format!("{ev}.{field_name}"), value);
                        }
                    }
                }
            }
        }
        return Ok(vec![row]);
    }

    Ok(vec![])
}

/// Check whether an expression tree contains any `PatternPredicate` nodes.
fn expr_contains_pattern_predicate(expr: &Expr) -> bool {
    match expr {
        Expr::PatternPredicate(_)
        | Expr::ExistsSubquery(_)
        | Expr::CountSubquery(_)
        | Expr::CollectSubquery { .. }
        | Expr::PatternComprehension { .. } => true,
        Expr::UnaryOp { expr, .. } => expr_contains_pattern_predicate(expr),
        Expr::BinaryOp { left, right, .. } => {
            expr_contains_pattern_predicate(left) || expr_contains_pattern_predicate(right)
        }
        _ => false,
    }
}

/// Evaluate a predicate expression that may contain pattern predicates.
/// Unlike `eval_expr`, this has access to the storage engine for edge lookups.
fn eval_predicate_with_storage(
    expr: &Expr,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Value, ExecutionError> {
    match expr {
        Expr::PatternPredicate(pattern) => {
            // Trivial single-hop patterns use the fast-path existence walker.
            // Multi-hop chains (which the walker mishandles when an intermediate
            // node is unbound) and any pattern carrying inline edge-property
            // filters (which the walker does not evaluate) desugar into an
            // EXISTS subquery routed through the full MATCH planner, matching
            // `MATCH` semantics exactly.
            let has_edge_filter = pattern
                .elements
                .iter()
                .any(|e| matches!(e, PatternElement::Relationship(r) if !r.properties.is_empty()));
            if pattern.elements.len() > 3 || has_edge_filter {
                let mc = crate::cypher::ast::MatchClause {
                    patterns: vec![pattern.clone()],
                    where_clause: None,
                };
                exists_subquery_matches(&mc, row, ctx)
            } else {
                check_pattern_exists(pattern, row, ctx)
            }
        }
        Expr::ExistsSubquery(match_clause) => exists_subquery_matches(match_clause, row, ctx),
        Expr::CountSubquery(match_clause) => count_subquery_eval(match_clause, row, ctx),
        Expr::CollectSubquery { match_clause, expr } => {
            collect_subquery_eval(match_clause, expr, row, ctx)
        }
        Expr::PatternComprehension {
            pattern,
            where_clause,
            map,
        } => pattern_comprehension_eval(pattern, where_clause.as_deref(), map, row, ctx),
        Expr::UnaryOp { op, expr } => {
            let v = eval_predicate_with_storage(expr, row, ctx)?;
            Ok(eval_unary_op(*op, &v))
        }
        Expr::BinaryOp { left, op, right } => {
            let lv = eval_predicate_with_storage(left, row, ctx)?;
            // Short-circuit for AND/OR
            match op {
                BinaryOperator::And if !is_truthy(&lv) => return Ok(Value::Bool(false)),
                BinaryOperator::Or if is_truthy(&lv) => return Ok(Value::Bool(true)),
                _ => {}
            }
            let rv = eval_predicate_with_storage(right, row, ctx)?;
            Ok(eval_binary_op(&lv, *op, &rv))
        }
        other => Ok(eval_expr(other, row)),
    }
}

/// Evaluate an `EXISTS { MATCH … }` subquery: true when the inner MATCH yields
/// at least one row consistent with the outer-scope bindings.
///
/// The inner MATCH (plus its optional WHERE) is wrapped in a synthetic
/// `… RETURN 1` query and planned through the same logical planner that serves
/// top-level queries — so pattern semantics in `EXISTS { (a)-[:R]->(b) }` match
/// those of a top-level `MATCH (a)-[:R]->(b)` exactly. It executes correlated:
/// the outer row is installed as `correlated_row` so inner references to
/// outer-bound variables resolve, and only inner rows agreeing with the outer
/// bindings on shared variables count.
fn exists_subquery_matches(
    mc: &crate::cypher::ast::MatchClause,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Value, ExecutionError> {
    use crate::cypher::ast::{Clause, Query, ReturnClause, ReturnItem};
    let query = Query {
        clauses: vec![
            Clause::Match(mc.clone()),
            // RETURN * so the inner pattern variables survive projection — the
            // post-filter below matches them against the outer bindings.
            Clause::Return(ReturnClause {
                distinct: false,
                items: vec![ReturnItem {
                    expr: Expr::Star,
                    alias: None,
                }],
            }),
        ],
        hints: Vec::new(),
        unions: Vec::new(),
    };
    let plan = crate::planner::builder::build_logical_plan(&query).map_err(|e| {
        ExecutionError::Unsupported(format!("EXISTS subquery planning failed: {e}"))
    })?;

    // Execute the inner plan uncorrelated (a fresh scan), then keep only rows
    // that agree with the outer bindings on shared variables — that post-filter
    // IS the correlation. (Installing `correlated_row` instead would merge the
    // outer values into every inner row, defeating the shared-variable check.)
    // The inner plan still evaluates its own WHERE on inner-bound variables.
    let rows = execute_op(&plan.root, ctx)?;
    let any = rows
        .iter()
        .any(|rr| rr.iter().all(|(k, v)| row.get(k).is_none_or(|ov| ov == v)));
    Ok(Value::Bool(any))
}

/// Execute a correlated subquery's inner MATCH (planned with `RETURN *` so the
/// inner variables survive), keeping only rows that agree with the outer `row`
/// on shared variables, each merged with the outer bindings. Shared by
/// `COUNT { … }` and `COLLECT { … }` (mirrors the EXISTS correlation path).
fn correlated_subquery_rows(
    mc: &crate::cypher::ast::MatchClause,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use crate::cypher::ast::{Clause, Query, ReturnClause, ReturnItem};
    let query = Query {
        clauses: vec![
            Clause::Match(mc.clone()),
            Clause::Return(ReturnClause {
                distinct: false,
                items: vec![ReturnItem {
                    expr: Expr::Star,
                    alias: None,
                }],
            }),
        ],
        hints: Vec::new(),
        unions: Vec::new(),
    };
    let plan = crate::planner::builder::build_logical_plan(&query)
        .map_err(|e| ExecutionError::Unsupported(format!("subquery planning failed: {e}")))?;
    let rows = execute_op(&plan.root, ctx)?;
    Ok(rows
        .into_iter()
        .filter(|rr| rr.iter().all(|(k, v)| row.get(k).is_none_or(|ov| ov == v)))
        .map(|rr| {
            let mut merged = row.clone();
            merged.extend(rr);
            merged
        })
        .collect())
}

/// `COUNT { MATCH … [WHERE …] }` → the number of correlated matching rows.
fn count_subquery_eval(
    mc: &crate::cypher::ast::MatchClause,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Value, ExecutionError> {
    let rows = correlated_subquery_rows(mc, row, ctx)?;
    Ok(Value::Int(i64::try_from(rows.len()).unwrap_or(i64::MAX)))
}

/// `COLLECT { MATCH … [WHERE …] RETURN expr }` → a list of `expr` evaluated over
/// each correlated matching row.
fn collect_subquery_eval(
    mc: &crate::cypher::ast::MatchClause,
    expr: &Expr,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Value, ExecutionError> {
    let rows = correlated_subquery_rows(mc, row, ctx)?;
    Ok(Value::Array(
        rows.iter().map(|er| eval_expr(expr, er)).collect(),
    ))
}

/// Evaluate a `[(a)-[:R]->(b) WHERE pred | map]` pattern comprehension: match
/// the inner pattern (correlated with `row`), then collect `map` evaluated over
/// each matching, outer-consistent row into a list.
///
/// Shares the EXISTS machinery: the pattern is planned through the MATCH planner
/// with a `RETURN *` so the inner variables survive, executed uncorrelated, then
/// post-filtered against the outer bindings on shared variables. For each kept
/// row, `map` is evaluated against the outer row merged with the inner bindings.
fn pattern_comprehension_eval(
    pattern: &Pattern,
    where_clause: Option<&Expr>,
    map: &Expr,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Value, ExecutionError> {
    use crate::cypher::ast::{Clause, MatchClause, Query, ReturnClause, ReturnItem};
    let mc = MatchClause {
        patterns: vec![pattern.clone()],
        where_clause: where_clause.cloned(),
    };
    let query = Query {
        clauses: vec![
            Clause::Match(mc),
            Clause::Return(ReturnClause {
                distinct: false,
                items: vec![ReturnItem {
                    expr: Expr::Star,
                    alias: None,
                }],
            }),
        ],
        hints: Vec::new(),
        unions: Vec::new(),
    };
    let plan = crate::planner::builder::build_logical_plan(&query).map_err(|e| {
        ExecutionError::Unsupported(format!("pattern comprehension planning failed: {e}"))
    })?;

    let rows = execute_op(&plan.root, ctx)?;
    let mut out = Vec::new();
    for rr in rows {
        if rr.iter().all(|(k, v)| row.get(k).is_none_or(|ov| ov == v)) {
            let mut eval_row = row.clone();
            eval_row.extend(rr);
            out.push(eval_expr(map, &eval_row));
        }
    }
    Ok(Value::Array(out))
}

/// Check if a pattern predicate matches: does the described path exist in the graph?
///
/// Handles bound endpoints (variables already in the row) by checking specific
/// edge existence via `expand_one_hop`, and unbound endpoints by checking if
/// any matching neighbor exists.
fn check_pattern_exists(
    pattern: &Pattern,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Value, ExecutionError> {
    // Pattern must be node-rel-node (possibly chained).
    // Extract triples: (src_node, relationship, dst_node)
    let elements = &pattern.elements;
    if elements.len() < 3 {
        return Ok(Value::Bool(false));
    }

    // Walk the pattern in (node, rel, node) triples
    let mut i = 0;
    while i + 2 < elements.len() {
        let src_node = match &elements[i] {
            PatternElement::Node(n) => n,
            _ => return Ok(Value::Bool(false)),
        };
        let rel = match &elements[i + 1] {
            PatternElement::Relationship(r) => r,
            _ => return Ok(Value::Bool(false)),
        };
        let dst_node = match &elements[i + 2] {
            PatternElement::Node(n) => n,
            _ => return Ok(Value::Bool(false)),
        };

        let exists = check_single_hop_exists(src_node, rel, dst_node, row, ctx)?;
        if !exists {
            return Ok(Value::Bool(false));
        }

        i += 2; // advance to next triple (overlapping nodes)
    }

    Ok(Value::Bool(true))
}

/// Check if a single hop (src)-[rel]->(dst) exists.
fn check_single_hop_exists(
    src_node: &NodePattern,
    rel: &RelationshipPattern,
    dst_node: &NodePattern,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<bool, ExecutionError> {
    // R172d: pattern predicate destination label-filter on a temporal
    // target prefix-scans every version of the candidate node and
    // returns true if ANY version carries all the requested labels.
    // This matches the "every version is a fact" semantics of the
    // current label-scoped MATCH on temporal labels — without AS OF
    // (G096), an existence check spans the full version history.
    let dst_is_temporal = dst_node.labels.iter().any(|lbl| {
        ctx.load_current_label_schema(lbl)
            .ok()
            .flatten()
            .is_some_and(|s| s.temporal)
    });

    // Resolve source node ID from bound variable
    let src_id = match &src_node.variable {
        Some(name) => match row.get(name.as_str()) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => return Ok(false), // unbound or not a node → no match
        },
        None => return Ok(false), // anonymous source not supported in predicate context
    };

    // Resolve edge types (empty = wildcard)
    let resolved_types: Vec<String>;
    let effective_types: &[String] = if rel.rel_types.is_empty() {
        resolved_types = ctx.list_edge_types()?;
        &resolved_types
    } else {
        &rel.rel_types
    };

    // Check if destination is bound
    let dst_bound = dst_node.variable.as_ref().and_then(|name| {
        row.get(name.as_str()).and_then(|v| {
            if let Value::Int(id) = v {
                Some(*id as u64)
            } else {
                None
            }
        })
    });

    for et in effective_types {
        let neighbors = expand_one_hop(src_id, std::slice::from_ref(et), rel.direction, ctx)?;
        if let Some(target_raw) = dst_bound {
            // Both endpoints bound: check specific edge
            if neighbors.iter().any(|(tgt, _)| *tgt == target_raw) {
                return Ok(true);
            }
        } else {
            // Destination unbound: check if any neighbor exists
            // Optionally filter by label
            if dst_node.labels.is_empty() {
                if !neighbors.is_empty() {
                    return Ok(true);
                }
            } else {
                // Check labels on neighbors. Temporal targets: prefix-scan
                // every version and accept the match if ANY version
                // carries all the requested labels.
                for (tgt_uid, _) in &neighbors {
                    let tgt_id = NodeId::from_raw(*tgt_uid);
                    if dst_is_temporal {
                        use coordinode_modality::{LocalNodeStore, NodeStore as _};
                        let prefix = LocalNodeStore.version_prefix(ctx.shard_id, tgt_id);
                        ctx.sync_txn_state();
                        let versions = LocalNodeStore.prefix_scan_tracked(&mut ctx.txn, &prefix)?;
                        for (_k, data) in versions {
                            if let Ok(record) = NodeRecord::from_msgpack(&data) {
                                if dst_node.labels.iter().all(|l| record.labels.contains(l)) {
                                    return Ok(true);
                                }
                            }
                        }
                    } else if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, tgt_id)? {
                        if dst_node.labels.iter().all(|l| record.labels.contains(l)) {
                            return Ok(true);
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}

/// Create a relationship edge described by `traverse` between the source and target
/// nodes bound in `correlated`. Called from `execute_merge` when no existing edge
/// was found by `execute_merge_relationship_check`.
///
/// G075: edge properties from `edge_filters` are now stored in the EdgeProp partition
/// so that subsequent MATCH or MERGE can retrieve and compare them.
fn execute_merge_relationship_create(
    traverse: &LogicalOp,
    correlated: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let (source, edge_types, direction, target_variable, edge_variable, edge_filters) =
        match traverse {
            LogicalOp::Traverse {
                source,
                edge_types,
                direction,
                target_variable,
                edge_variable,
                edge_filters,
                ..
            } => (
                source,
                edge_types,
                direction,
                target_variable,
                edge_variable,
                edge_filters,
            ),
            _ => unreachable!("execute_merge_relationship_create: not a Traverse"),
        };

    if edge_types.is_empty() {
        return Err(ExecutionError::Unsupported(
            "MERGE relationship: cannot create edge with wildcard type — specify a relationship type".into(),
        ));
    }

    let source_id = match correlated.get(source.as_str()) {
        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
        _ => {
            return Err(ExecutionError::Unsupported(format!(
                "MERGE relationship: source variable '{source}' not bound in scope"
            )))
        }
    };
    let target_id = match correlated.get(target_variable.as_str()) {
        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
        _ => {
            return Err(ExecutionError::Unsupported(format!(
                "MERGE relationship: target variable '{target_variable}' not bound in scope"
            )))
        }
    };

    // Use the first (and typically only) edge type for creation.
    let et = &edge_types[0];

    // Direction determines which node is "from" vs "to" in the adjacency lists.
    let (from_id, to_id) = match direction {
        Direction::Outgoing | Direction::Both => (source_id, target_id),
        Direction::Incoming => (target_id, source_id),
    };
    ctx.adj_merge_add_fwd(et, from_id, to_id.as_raw());
    ctx.adj_merge_add_rev(et, to_id, from_id.as_raw());
    ctx.write_stats.edges_created += 1;

    // Register edge type in schema (idempotent — never clobber an existing
    // EdgeTypeSchema written by `CREATE EDGE TYPE`).
    ctx.mvcc_register_edge_type(et)?;

    let mut row = correlated.clone();
    if let Some(ev) = edge_variable {
        row.insert(format!("{ev}.__type__"), Value::String(et.clone()));
        row.insert(ev.clone(), Value::String(et.clone()));
    }

    // G075: store edge properties (from pattern `[r:TYPE {prop: val}]`) in EdgeProp partition.
    // Key: edgeprop:<TYPE>:<from_id BE>:<to_id BE>  (same format as CREATE clause).
    // Value: MessagePack Vec<(field_id, Value)>.
    // Note: if this edge already existed with different properties, the new values
    // overwrite the old (upsert semantics within a single-edge-per-type data model).
    let mut resolved_props: Vec<(String, Value)> = Vec::with_capacity(edge_filters.len());
    if !edge_filters.is_empty() {
        let mut prop_map: Vec<(u32, Value)> = Vec::with_capacity(edge_filters.len());
        for (prop_name, expr) in edge_filters {
            let field_id = ctx.interner.intern(prop_name);
            let value = eval_expr(expr, correlated);
            prop_map.push((field_id, value.clone()));
            resolved_props.push((prop_name.clone(), value.clone()));
            if let Some(ev) = edge_variable {
                row.insert(format!("{ev}.{prop_name}"), value);
            }
        }
        ctx.mvcc_put_edge_props(et, from_id, to_id, &prop_map)?;
        ctx.write_stats.properties_set += edge_filters.len() as u64;
    }

    // Fire BEFORE COMMIT CREATE triggers registered on this edge type. Same
    // semantics as `execute_create_edge`: `$src` / `$tgt` / `$edge_type` /
    // `$after` describe the freshly-created edge. Runs after the edgeprop
    // write so the trigger body sees the edge via RYOW.
    let trigger_params = trigger_params_for_edge_create(et, from_id, to_id, &resolved_props);
    let target_segment =
        coordinode_core::schema::triggers::TriggerTargetSchema::edge_type(et).index_key_segment();
    let matched = ctx.lookup_matching_triggers(&target_segment, "c")?;
    if !matched.is_empty() {
        fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
    }

    Ok(vec![row])
}

/// G074: Create a complete relationship pattern for standalone MERGE (no preceding MATCH).
///
/// Called from `execute_merge` when the pattern is a Traverse but `correlated_row` is `None`
/// and no existing complete path was found by `execute_op`.
///
/// Algorithm:
///   1. Find or create the source node using the Traverse's `input` (NodeScan).
///   2. Build a target NodeScan from `target_variable`, `target_labels`, `target_filters`.
///   3. Find or create the target node.
///   4. Merge the two node rows into a synthetic correlated row.
///   5. Delegate edge creation to `execute_merge_relationship_create`.
///
/// "Find or create" semantics per GAPS.md G074:
///   - If a node matching the label+property pattern already exists, reuse the first match.
///   - If no matching node exists, create a new one with the given labels and properties.
fn execute_merge_relationship_standalone_create(
    traverse: &LogicalOp,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let (input, target_variable, target_labels, target_filters) = match traverse {
        LogicalOp::Traverse {
            input,
            target_variable,
            target_labels,
            target_filters,
            ..
        } => (input, target_variable, target_labels, target_filters),
        _ => unreachable!("execute_merge_relationship_standalone_create: not a Traverse"),
    };

    // Step 1: Find or create source node from the Traverse's input (NodeScan).
    // Ambiguous pattern (multiple matching nodes) is an error: MERGE requires a unique match.
    // Use MERGE ALL if multi-target upsert is needed.
    let src_rows = execute_op(input, ctx)?;
    let src_row = match src_rows.len() {
        0 => {
            let created = execute_create_from_pattern(input, ctx)?;
            created.into_iter().next().ok_or_else(|| {
                ExecutionError::Unsupported("MERGE: failed to find or create source node".into())
            })?
        }
        1 => src_rows
            .into_iter()
            .next()
            .ok_or_else(|| ExecutionError::Unsupported("MERGE: source row missing".into()))?,
        n => {
            return Err(ExecutionError::Unsupported(format!(
                "MERGE relationship: ambiguous source pattern — {n} nodes match. \
                 Use a more specific property filter or MERGE ALL for multi-target upsert."
            )))
        }
    };

    // Step 2: Find or create target node synthesized from Traverse target fields.
    // Same ambiguity check as Step 1.
    let target_scan = LogicalOp::NodeScan {
        variable: target_variable.clone(),
        labels: target_labels.clone(),
        property_filters: target_filters.clone(),
    };
    let tgt_rows = execute_op(&target_scan, ctx)?;
    let tgt_row = match tgt_rows.len() {
        0 => {
            let created = execute_create_from_pattern(&target_scan, ctx)?;
            created.into_iter().next().ok_or_else(|| {
                ExecutionError::Unsupported("MERGE: failed to find or create target node".into())
            })?
        }
        1 => tgt_rows
            .into_iter()
            .next()
            .ok_or_else(|| ExecutionError::Unsupported("MERGE: target row missing".into()))?,
        n => {
            return Err(ExecutionError::Unsupported(format!(
                "MERGE relationship: ambiguous target pattern — {n} nodes match. \
                 Use a more specific property filter or MERGE ALL for multi-target upsert."
            )))
        }
    };

    // Step 3: Build synthetic correlated row with both node IDs in scope.
    // Target row entries overwrite source row entries on key collision, but
    // source and target variables are always distinct in valid Cypher patterns.
    let mut correlated = src_row;
    correlated.extend(tgt_row);

    // Step 4: Create the edge. The complete path was verified absent by the caller.
    execute_merge_relationship_create(traverse, &correlated, ctx)
}

/// UPSERT MATCH: atomic match-or-create.
/// ON MATCH → SET items on existing match.
/// ON CREATE → CREATE new patterns.
/// Atomic UPSERT: query-then-mutate with key-based CAS conflict detection.
///
/// 1. MATCH phase: execute pattern, capture raw node bytes per matched variable
/// 2. If match found → ON MATCH: re-read nodes, compare bytes (CAS). If changed → ErrConflict.
///    Apply SET items.
/// 3. If no match → ON CREATE: create nodes and edges from patterns (two-pass).
///
/// Inspired by Dgraph's atomic query+mutate (edgraph/server.go do→processQuery→doMutate).
fn execute_upsert(
    pattern: &LogicalOp,
    on_match: &[crate::cypher::ast::SetItem],
    on_create_patterns: &[Pattern],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Phase 1: MATCH — find existing nodes
    let matches = execute_op(pattern, ctx)?;

    if !matches.is_empty() {
        // Phase 2: ON MATCH — CAS check + apply SET items
        if on_match.is_empty() {
            return Ok(matches);
        }

        // Concurrent-modification detection is owned by Layer-3 OCC:
        // every `mvcc_get` issued during MATCH (above) tracked the
        // node key in the per-transaction `occ_scope`; at commit time
        // `mvcc_flush` calls `coordinator.validate_occ` which probes
        // `has_write_after` on each tracked key and surfaces
        // `ExecutionError::Conflict` if any concurrent writer landed
        // since `mvcc_read_ts`. The byte-level CAS that used to live
        // here was pre-G104 manual machinery and is now redundant —
        // also strictly less safe (it tolerated ABA writes, OCC does
        // not).
        execute_update(&matches, on_match, &ViolationMode::Fail, ctx)
    } else {
        // Phase 3: ON CREATE — create nodes and edges from patterns (two-pass)
        // R172b safe-reject for temporal labels: UPSERT ON CREATE uses
        // `encode_node_key` (16-byte form) directly and would silently
        // bypass the per-version 25-byte key on a temporal label, also
        // skipping `__ingestion_ts__` auto-population and the
        // `valid_from`-required guard. Per-version semantics for UPSERT
        // ON CREATE land in R172c alongside MERGE.
        for create_pattern in on_create_patterns {
            for element in &create_pattern.elements {
                if let PatternElement::Node(np) = element {
                    for lbl in &np.labels {
                        if let Ok(Some(s)) = ctx.load_current_label_schema(lbl) {
                            if s.temporal {
                                return Err(ExecutionError::Unsupported(format!(
                                    "UPSERT ON CREATE into temporal label '{lbl}' is not \
                                     yet supported (lands in R172c — per-version write \
                                     executor). Use an explicit CREATE clause for \
                                     temporal labels in the R172b scope."
                                )));
                            }
                        }
                    }
                }
            }
        }

        let mut results = vec![Row::new()];

        for create_pattern in on_create_patterns {
            let elements = &create_pattern.elements;

            // Pass 1: create nodes
            let mut new_results = Vec::new();
            for row in &results {
                let mut current_row = row.clone();
                for element in elements {
                    if let PatternElement::Node(np) = element {
                        let node_id = ctx.id_allocator.next();
                        let label = np.labels.first().cloned().unwrap_or_default();

                        let mut record = NodeRecord::new(&label);
                        for (prop_name, expr) in &np.properties {
                            let val = eval_expr(expr, &current_row);
                            let field_id = ctx.interner.intern(prop_name);
                            record.set(field_id, val);
                        }

                        ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;
                        ctx.write_stats.nodes_created += 1;

                        // Fire BEFORE COMMIT CREATE triggers on the new
                        // node's label. UPSERT's ON CREATE branch is
                        // logically `CREATE (n:Label {…})`; users expect
                        // the same trigger semantics as a hand-written
                        // CREATE.
                        if !label.is_empty() {
                            let props_map: std::collections::HashMap<String, Value> = np
                                .properties
                                .iter()
                                .map(|(name, expr)| {
                                    let val = eval_expr(expr, row).map_to_document();
                                    (name.clone(), val)
                                })
                                .collect();
                            let trigger_params =
                                trigger_params_for_node_create(node_id, &props_map);
                            let target_segment =
                                coordinode_core::schema::triggers::TriggerTargetSchema::label(
                                    label.clone(),
                                )
                                .index_key_segment();
                            let matched = ctx.lookup_matching_triggers(&target_segment, "c")?;
                            if !matched.is_empty() {
                                fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
                            }
                        }

                        let var_name = np.variable.as_deref().unwrap_or("_");
                        current_row
                            .insert(var_name.to_string(), Value::Int(node_id.as_raw() as i64));
                        current_row.insert(format!("{var_name}.__label__"), Value::String(label));
                        for (prop_name, expr) in &np.properties {
                            let val = eval_expr(expr, row);
                            current_row.insert(format!("{var_name}.{prop_name}"), val);
                        }
                    }
                }
                new_results.push(current_row);
            }

            // Pass 2: create edges (all node IDs now in row)
            let mut edge_results = Vec::new();
            for row in &new_results {
                let mut current_row = row.clone();
                for (i, element) in elements.iter().enumerate() {
                    if let PatternElement::Relationship(rp) = element {
                        let source_var = if i > 0 {
                            if let PatternElement::Node(np) = &elements[i - 1] {
                                np.variable.as_deref().unwrap_or("")
                            } else {
                                ""
                            }
                        } else {
                            ""
                        };
                        let target_var = if i + 1 < elements.len() {
                            if let PatternElement::Node(np) = &elements[i + 1] {
                                np.variable.as_deref().unwrap_or("")
                            } else {
                                ""
                            }
                        } else {
                            ""
                        };

                        let (src, tgt) = match rp.direction {
                            Direction::Incoming => (target_var, source_var),
                            _ => (source_var, target_var),
                        };

                        let source_id = match current_row.get(src) {
                            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                            _ => continue,
                        };
                        let target_id = match current_row.get(tgt) {
                            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                            _ => continue,
                        };

                        let edge_type = rp.rel_types.first().cloned().unwrap_or_default();

                        // Forward + reverse posting lists (commutative merge,
                        // no read needed) via the typed Layer-4 store.
                        ctx.adj_merge_add_fwd(&edge_type, source_id, target_id.as_raw());
                        ctx.adj_merge_add_rev(&edge_type, target_id, source_id.as_raw());
                        ctx.write_stats.edges_created += 1;

                        // Fire BEFORE COMMIT CREATE triggers on the new
                        // edge type. UPSERT's ON CREATE branch on a
                        // relationship pattern is logically the same as
                        // `CREATE (a)-[:TYPE]->(b)` and must fire the same
                        // trigger. `$after` is empty (no inline properties
                        // captured at the executor level for this path —
                        // UPSERT ON CREATE patterns currently store edge
                        // properties via subsequent SET items, not inline).
                        if !edge_type.is_empty() {
                            let resolved_props: Vec<(String, Value)> = Vec::new();
                            let trigger_params = trigger_params_for_edge_create(
                                &edge_type,
                                source_id,
                                target_id,
                                &resolved_props,
                            );
                            let target_segment =
                                coordinode_core::schema::triggers::TriggerTargetSchema::edge_type(
                                    &edge_type,
                                )
                                .index_key_segment();
                            let matched = ctx.lookup_matching_triggers(&target_segment, "c")?;
                            if !matched.is_empty() {
                                fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
                            }
                        }

                        if let Some(ev) = &rp.variable {
                            current_row
                                .insert(format!("{ev}.__type__"), Value::String(edge_type.clone()));
                        }
                    }
                }
                edge_results.push(current_row);
            }

            results = if edge_results.is_empty() {
                new_results
            } else {
                edge_results
            };
        }

        Ok(results)
    }
}

/// Create a node from a pattern scan operator (extracts label + properties from NodeScan).
fn execute_create_from_pattern(
    pattern: &LogicalOp,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    match pattern {
        LogicalOp::NodeScan {
            variable,
            labels,
            property_filters,
        } => {
            // R172b safe-reject for temporal labels: this code path (used by
            // MERGE's create branch and by `MERGE (a)-[:E]->(b)` endpoint
            // synthesis) writes via the 16-byte non-temporal key
            // unconditionally. A temporal target would silently land in
            // non-temporal storage. Per-version semantics for MERGE/UPSERT
            // on temporal labels land in R172c.
            for lbl in labels {
                if let Ok(Some(s)) = ctx.load_current_label_schema(lbl) {
                    if s.temporal {
                        return Err(ExecutionError::Unsupported(format!(
                            "MERGE / UPSERT into temporal label '{lbl}' is not yet \
                             supported (lands in R172c — per-version write executor). \
                             Use explicit CREATE for temporal labels in the R172b scope."
                        )));
                    }
                }
            }

            let node_id = ctx.id_allocator.next();
            let label = labels.first().cloned().unwrap_or_default();

            let mut record = NodeRecord::new(&label);
            let empty_row = Row::new();
            for (prop_name, expr) in property_filters {
                let val = eval_expr(expr, &empty_row);
                let field_id = ctx.interner.intern(prop_name);
                record.set(field_id, val);
            }

            // Enforce unique constraints via B-tree index registry.
            if let Some(btree_reg) = ctx.btree_index_registry {
                if !label.is_empty() {
                    let props_for_index: Vec<(String, Value)> = property_filters
                        .iter()
                        .map(|(name, expr)| (name.clone(), eval_expr(expr, &empty_row)))
                        .collect();
                    btree_reg
                        .on_node_created(ctx.engine, node_id, &label, &props_for_index)
                        .map_err(|v| {
                            ExecutionError::Conflict(format!(
                                "unique constraint violated on index `{}`: \
                                 property `{}` already has value {:?}",
                                v.index_name, v.property, v.value
                            ))
                        })?;
                }
            }

            ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;

            // Fire BEFORE COMMIT CREATE triggers registered on the new
            // node's label. This path is reached from MERGE (create branch)
            // and from standalone MERGE relationship create when the
            // endpoint node has to be invented; both must fire the same
            // CREATE trigger as `execute_create_node`.
            if !label.is_empty() {
                let props_map: std::collections::HashMap<String, Value> = property_filters
                    .iter()
                    .map(|(name, expr)| {
                        let val = eval_expr(expr, &empty_row).map_to_document();
                        (name.clone(), val)
                    })
                    .collect();
                let trigger_params = trigger_params_for_node_create(node_id, &props_map);
                let target_segment =
                    coordinode_core::schema::triggers::TriggerTargetSchema::label(label.clone())
                        .index_key_segment();
                let matched = ctx.lookup_matching_triggers(&target_segment, "c")?;
                if !matched.is_empty() {
                    fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
                }
            }

            let mut row = Row::new();
            row.insert(variable.to_string(), Value::Int(node_id.as_raw() as i64));
            row.insert(format!("{variable}.__label__"), Value::String(label));
            for (prop_name, expr) in property_filters {
                let val = eval_expr(expr, &Row::new());
                row.insert(format!("{variable}.{prop_name}"), val);
            }

            Ok(vec![row])
        }
        LogicalOp::Filter { input, .. } => {
            // If there's a filter wrapping a scan, use the inner scan for creation
            execute_create_from_pattern(input, ctx)
        }
        _ => Err(ExecutionError::Unsupported(
            "MERGE create from non-NodeScan pattern".into(),
        )),
    }
}

/// Try to extract f32 vector data from a Value.
///
/// Handles both `Value::Vector` (native) and `Value::Array` containing
/// only Float/Int elements (Cypher array literals like `[1.0, 0.0]`).
/// Gate a vector search against the index's persisted build state and
/// online-during-build policy. Returns `Ok(())` when the caller can use
/// the in-memory HNSW handle, `Err(...)` when the caller must abort.
///
/// Cost: when the in-memory registry policy is `PartialRecall` this is a
/// single map lookup with no schema read — the dominant common case.
/// `Block` and `Offline` policies plus the rarely-hit `Failed` recovery
/// path consult the persisted schema for a fresh state.
fn gate_vector_index_read(
    engine: &StorageEngine,
    registry: &crate::index::VectorIndexRegistry,
    label: &str,
    property: &str,
) -> Result<(), ExecutionError> {
    let Some(def) = registry.get_definition(label, property) else {
        // No registered def — caller will fall back to brute-force or
        // return an empty result. Not our gate to enforce.
        return Ok(());
    };

    // PartialRecall: short-circuit before any schema read. Matches the
    // pre-policy behaviour (search whatever's in the graph right now).
    if def.online_during_build == OnlineDuringBuild::PartialRecall {
        return Ok(());
    }

    // For Block / Offline, fetch the live state from schema so we react
    // to backfill completion that happened after registry registration.
    // Block polls until the backfill completes (matching the legacy
    // synchronous-build semantic), Offline returns an error immediately.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let poll_step = std::time::Duration::from_millis(25);
    loop {
        let live_state = crate::index::ops::load_index_definition(engine, &def.name)
            .ok()
            .flatten()
            .map(|d| d.state)
            .unwrap_or(IndexState::Ready);

        match live_state {
            IndexState::Ready => return Ok(()),
            IndexState::Failed { reason } => {
                return Err(ExecutionError::Unsupported(format!(
                    "vector index '{}' failed to build: {reason}",
                    def.name
                )));
            }
            IndexState::Building { .. } => match def.online_during_build {
                OnlineDuringBuild::Offline => {
                    return Err(ExecutionError::Unsupported(format!(
                        "vector index '{}' is offline during build",
                        def.name
                    )));
                }
                OnlineDuringBuild::Block => {
                    if std::time::Instant::now() >= deadline {
                        return Err(ExecutionError::Unsupported(format!(
                            "vector index '{}' still building after 30s",
                            def.name
                        )));
                    }
                    std::thread::sleep(poll_step);
                    continue;
                }
                OnlineDuringBuild::PartialRecall => {
                    // Early-return above handles this; unreachable here.
                    return Ok(());
                }
            },
        }
    }
}

use coordinode_core::graph::types::try_extract_vector;

/// CREATE node: allocate ID, build record, write to storage.
fn execute_create_node(
    input_rows: &[Row],
    variable: Option<&str>,
    labels: &[String],
    properties: &[(String, Expr)],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let mut results = Vec::new();

    // Load schema for the primary label (if any) to determine property routing.
    // In VALIDATED mode, undeclared properties go to `extra` (string keys).
    let schema = if let Some(primary) = labels.first() {
        ctx.load_current_label_schema(primary).ok().flatten()
    } else {
        None
    };

    // Effective mode: use schema mode when a schema is declared, otherwise FLEXIBLE
    // (no schema = no enforcement, preserving backward-compatible behaviour for
    // labels that were never declared via CreateLabel).
    let mode = schema
        .as_ref()
        .map(|s| s.mode)
        .unwrap_or(SchemaMode::Flexible);

    // R172a reserved-name guard at CREATE time: `__ingestion_ts__` is
    // engine-owned on temporal labels — populated automatically with the HLC
    // commit timestamp, user-immutable. Rejecting user-supplied values up
    // front prevents accidental shadowing and matches the symmetric DDL-time
    // reserved-name diagnostic in `execute_create_node_type`.
    for (prop_name, _) in properties {
        if prop_name == "__ingestion_ts__" {
            return Err(ExecutionError::Unsupported(
                "property name '__ingestion_ts__' is reserved for engine-internal use \
                 (auto-populated on temporal labels with the HLC commit timestamp); \
                 it cannot be assigned in CREATE"
                    .into(),
            ));
        }
    }

    // R172a write-time guard: temporal node types require `valid_from` on every
    // CREATE (mirror of the edge-side enforcement at `execute_create_edge`).
    // Mechanical bitemporal storage (per-version node key with the i64 BE
    // valid_from suffix) lands in R172b; this guard is here so the contract is
    // honoured from R172a onward — a CREATE on a TEMPORAL label without
    // `valid_from` is rejected at write time rather than silently writing a
    // record that the future per-version key encoder cannot place.
    //
    // Multi-label case: scan EVERY label, not just `labels.first()`. A node
    // declared `CREATE (n:Foo:Bar)` where any of Foo / Bar carries the
    // TEMPORAL flag must satisfy the bitemporal contract. This is conservative
    // — if a temporal mix is invalid (R172b will need to decide which key
    // layout wins on conflict), the write must still reject without
    // `valid_from`. Reports the first temporal label name to the user.
    let mut temporal_label: Option<String> = None;
    for lbl in labels {
        if let Ok(Some(s)) = ctx.load_current_label_schema(lbl) {
            if s.temporal {
                temporal_label = Some(lbl.clone());
                break;
            }
        }
    }
    if let Some(ref tlabel) = temporal_label {
        let has_valid_from = properties.iter().any(|(name, _)| name == "valid_from");
        if !has_valid_from {
            return Err(ExecutionError::Unsupported(format!(
                "label '{tlabel}' is TEMPORAL: CREATE requires a 'valid_from' \
                 timestamp property on every node"
            )));
        }
    }

    for input_row in input_rows {
        let node_id = ctx.id_allocator.next();

        let mut record = NodeRecord::with_labels(labels.to_vec());
        for (prop_name, expr) in properties {
            // Map literals → Document for full dot-notation support in storage.
            let val = eval_expr(expr, input_row).map_to_document();

            match mode {
                SchemaMode::Validated => {
                    let Some(schema_ref) = schema.as_ref() else {
                        unreachable!()
                    };
                    match schema_ref.get_property(prop_name) {
                        Some(def) if def.is_computed() => {
                            return Err(ExecutionError::SchemaViolation(format!(
                                "cannot SET computed property '{prop_name}'"
                            )));
                        }
                        Some(def) => {
                            // Declared non-computed property → validate type, then intern and set.
                            validate_one(prop_name, &val, def)
                                .map_err(|e| ExecutionError::SchemaViolation(e.to_string()))?;
                            let field_id = ctx.interner.intern(prop_name);
                            record.set(field_id, val.clone());
                        }
                        None => {
                            // Undeclared in VALIDATED mode → extra overflow map.
                            record.set_extra(prop_name, val.clone());
                        }
                    }
                }
                SchemaMode::Strict => {
                    let label_name = labels.first().map_or("?", String::as_str);
                    match schema.as_ref().and_then(|s| s.get_property(prop_name)) {
                        None => {
                            return Err(ExecutionError::SchemaViolation(format!(
                                "unknown property '{prop_name}' for strict label '{label_name}'"
                            )));
                        }
                        Some(def) if def.is_computed() => {
                            return Err(ExecutionError::SchemaViolation(format!(
                                "cannot SET computed property '{prop_name}'"
                            )));
                        }
                        Some(def) => {
                            // Declared non-computed property → validate type, then intern and set.
                            validate_one(prop_name, &val, def)
                                .map_err(|e| ExecutionError::SchemaViolation(e.to_string()))?;
                            let field_id = ctx.interner.intern(prop_name);
                            record.set(field_id, val.clone());
                        }
                    }
                }
                SchemaMode::Flexible => {
                    // No schema enforcement — intern and set unconditionally.
                    let field_id = ctx.interner.intern(prop_name);
                    record.set(field_id, val.clone());
                }
            }
        }

        // For STRICT and VALIDATED: verify all required (NOT NULL) properties
        // are present in the CREATE clause. Per proto PropertyDefinition.required:
        // "Writes missing this property are rejected in STRICT and VALIDATED modes."
        if matches!(mode, SchemaMode::Strict | SchemaMode::Validated) {
            if let Some(schema_ref) = schema.as_ref() {
                let provided: std::collections::HashSet<&str> =
                    properties.iter().map(|(n, _)| n.as_str()).collect();
                for (prop_name, def) in &schema_ref.properties {
                    if def.not_null
                        && def.default.is_none()
                        && !provided.contains(prop_name.as_str())
                    {
                        return Err(ExecutionError::SchemaViolation(format!(
                            "required property '{prop_name}' is missing in CREATE"
                        )));
                    }
                }
            }
        }

        // R172b temporal storage path: when the (primary or any) label is
        // TEMPORAL, extract `valid_from` from the supplied properties and
        // emit the per-version key. Auto-populate `__ingestion_ts__` from
        // the current HLC commit timestamp so the bitemporal system-axis is
        // queryable without an additional lookup. Mirror of the temporal-
        // edge write path in `execute_create_edge`.
        let valid_from_for_key: Option<i64> = if let Some(ref tlabel) = temporal_label {
            let mut vf: Option<i64> = None;
            for (prop_name, expr) in properties {
                if prop_name == "valid_from" {
                    let val = eval_expr(expr, input_row);
                    vf = match &val {
                        Value::Int(ms) => Some(*ms),
                        Value::Timestamp(ms) => Some(*ms),
                        Value::Null => {
                            return Err(ExecutionError::Unsupported(format!(
                                "label '{tlabel}' is TEMPORAL: valid_from must not be NULL"
                            )));
                        }
                        other => {
                            return Err(ExecutionError::Unsupported(format!(
                                "label '{tlabel}' is TEMPORAL: valid_from must be INT or \
                                 TIMESTAMP (epoch milliseconds), got {other:?}"
                            )));
                        }
                    };
                    break;
                }
            }
            // The earlier guard ensured a valid_from property was supplied;
            // an absent value here would be a programmer error.
            vf
        } else {
            None
        };

        // Optional `valid_to` interval invariant: when both ends present,
        // `valid_to` must be strictly greater than `valid_from`. Same rule
        // as temporal edges — a zero-duration version is never useful and
        // almost certainly a user mistake.
        if let (Some(ref tlabel), Some(vf)) = (&temporal_label, valid_from_for_key) {
            for (prop_name, expr) in properties {
                if prop_name == "valid_to" {
                    let val = eval_expr(expr, input_row);
                    let vt_opt: Option<i64> = match &val {
                        Value::Int(ms) => Some(*ms),
                        Value::Timestamp(ms) => Some(*ms),
                        Value::Null => None,
                        other => {
                            return Err(ExecutionError::Unsupported(format!(
                                "label '{tlabel}' is TEMPORAL: valid_to must be INT, TIMESTAMP, \
                                 or NULL, got {other:?}"
                            )));
                        }
                    };
                    if let Some(vt) = vt_opt {
                        if vt <= vf {
                            return Err(ExecutionError::Unsupported(format!(
                                "label '{tlabel}' is TEMPORAL: valid_to ({vt}) must be strictly \
                                 greater than valid_from ({vf})"
                            )));
                        }
                    }
                    break;
                }
            }

            // Auto-populate `__ingestion_ts__` from current HLC commit time.
            // The field is engine-owned, user-immutable, and lets bitemporal
            // AS-OF queries resolve the system-time axis without an extra
            // lookup. Microseconds (HLC native unit) — same precision as
            // `current_hlc_us` used by the trigger machinery.
            let ingestion_us = current_hlc_us() as i64;
            let field_id = ctx.interner.intern("__ingestion_ts__");
            record.set(field_id, Value::Int(ingestion_us));
        }

        // Enforce unique constraints via B-tree index registry BEFORE writing
        // the node to storage. If a constraint would be violated, fail early.
        if let Some(btree_reg) = ctx.btree_index_registry {
            if let Some(primary_label) = labels.first() {
                let props_for_index: Vec<(String, Value)> = properties
                    .iter()
                    .map(|(name, expr)| (name.clone(), eval_expr(expr, input_row)))
                    .collect();
                btree_reg
                    .on_node_created(ctx.engine, node_id, primary_label, &props_for_index)
                    .map_err(|v| {
                        ExecutionError::Conflict(format!(
                            "unique constraint violated on index `{}`: \
                             property `{}` already has value {:?}",
                            v.index_name, v.property, v.value
                        ))
                    })?;
            }
        }

        // `valid_from_for_key` is `Some(_)` for temporal labels and
        // `None` otherwise — the typed dispatch helper picks the
        // 25-byte temporal key or the 16-byte non-temporal key.
        ctx.mvcc_put_node_either(ctx.shard_id, node_id, valid_from_for_key, &record)?;
        ctx.write_stats.nodes_created += 1;
        ctx.write_stats.properties_set += properties.len() as u64;

        // Buffer HNSW writes so they hit the index once per batch
        // instead of once per row. The buffer is drained at the end
        // of `execute()` by `ExecutionContext::flush_pending_vector_writes`
        // which groups by (label, property) and calls
        // `VectorIndexRegistry::on_vectors_written`. Existence of an
        // index is still resolved here so we only buffer writes that
        // would actually land somewhere — the registry lookup itself
        // is cheap (RwLock::read on a HashMap).
        if let Some(registry) = ctx.vector_index_registry {
            if let Some(primary_label) = labels.first() {
                for (prop_name, expr) in properties {
                    if !registry.has_index(primary_label, prop_name) {
                        continue;
                    }
                    let val = eval_expr(expr, input_row);
                    if let Some(vec_data) = try_extract_vector(&val) {
                        ctx.pending_vector_writes.push((
                            primary_label.clone(),
                            prop_name.clone(),
                            node_id,
                            vec_data,
                        ));
                    }
                }
            }
        }

        // Notify text index registry of any text properties.
        if let Some(registry) = ctx.text_index_registry {
            if let Some(primary_label) = labels.first() {
                for (prop_name, expr) in properties {
                    let val = eval_expr(expr, input_row);
                    if let Some(text) = val.as_str() {
                        registry.on_text_written(primary_label, node_id, prop_name, text);
                    }
                }
            }
        }

        // Fire BEFORE COMMIT triggers registered on any of the new node's
        // labels for the CREATE event. Triggers fire after the node write
        // is staged in the MVCC buffer so a trigger body can MATCH the new
        // node via read-your-own-writes. On a trigger error (ON ERROR
        // PROPAGATE — the BEFORE-COMMIT default) the surrounding error
        // path drops the buffer and the node is never persisted. AFTER
        // COMMIT triggers are skipped here; they fire through the async
        // oplog-consumer path once that lands.
        if !labels.is_empty() {
            let props_map: std::collections::HashMap<String, Value> = properties
                .iter()
                .map(|(name, expr)| {
                    let val = eval_expr(expr, input_row).map_to_document();
                    (name.clone(), val)
                })
                .collect();
            let trigger_params = trigger_params_for_node_create(node_id, &props_map);

            for label in labels {
                let target_segment =
                    coordinode_core::schema::triggers::TriggerTargetSchema::label(label.clone())
                        .index_key_segment();
                let matched = ctx.lookup_matching_triggers(&target_segment, "c")?;
                if !matched.is_empty() {
                    fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
                }
            }
        }

        // Build output row
        let mut row = input_row.clone();
        let var_name = variable.unwrap_or("_");
        row.insert(var_name.to_string(), Value::Int(node_id.as_raw() as i64));
        let primary_label = record.primary_label().to_string();
        row.insert(
            format!("{var_name}.__label__"),
            Value::String(primary_label),
        );
        for (prop_name, expr) in properties {
            let val = eval_expr(expr, input_row);
            row.insert(format!("{var_name}.{prop_name}"), val);
        }

        results.push(row);
    }

    Ok(results)
}

/// CREATE edge: add to both forward and reverse posting lists.
fn execute_create_edge(
    input_rows: &[Row],
    source: &str,
    target: &str,
    edge_type: &str,
    edge_variable: Option<&str>,
    properties: &[(String, Expr)],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let mut results = Vec::new();

    // Check if the edge type is registered as temporal. Temporal edges require
    // a `valid_from` property at write time so every version can be keyed by
    // its validity start. Per-version storage layout lands in a follow-up step;
    // here we only enforce the API contract.
    let is_temporal = lookup_edge_type_temporal(edge_type, ctx)?;
    if is_temporal {
        let has_valid_from = properties.iter().any(|(name, _)| name == "valid_from");
        if !has_valid_from {
            return Err(ExecutionError::Unsupported(format!(
                "edge type '{edge_type}' is TEMPORAL: CREATE requires a 'valid_from' \
                 timestamp property on every instance"
            )));
        }
    }

    for row in input_rows {
        // Get source and target node IDs from the row
        let source_id = match row.get(source) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => continue,
        };
        let target_id = match row.get(target) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => continue,
        };

        // Forward + reverse posting lists (commutative merge) via the store.
        ctx.adj_merge_add_fwd(edge_type, source_id, target_id.as_raw());
        ctx.adj_merge_add_rev(edge_type, target_id, source_id.as_raw());
        ctx.write_stats.edges_created += 1;

        // Register edge type in schema (idempotent marker) ONLY if no entry
        // already exists. Overwriting would clobber an `EdgeTypeSchema` written
        // by `CREATE EDGE TYPE` (msgpack body carrying `temporal: true`) with
        // an empty marker — breaking subsequent temporal lookups for this type.
        // This enables O(edge_types) targeted lookup in DETACH DELETE instead
        // of O(all_edges) full scan.
        ctx.mvcc_register_edge_type(edge_type)?;

        // Store edge properties (facets) in EdgeProp partition.
        // Non-temporal: key = edgeprop:<TYPE>:<src BE>:<tgt BE>
        // Temporal:     key = edgeprop:<TYPE>:<src BE>:<tgt BE>:<valid_from BE>
        // Value: MessagePack map of field_id → Value (same as node properties)
        if !properties.is_empty() {
            let mut prop_map: Vec<(u32, Value)> = Vec::with_capacity(properties.len());
            let mut valid_from_value: Option<i64> = None;
            let mut valid_to_value: Option<i64> = None;
            for (prop_name, expr) in properties {
                // Reject writes to reserved metadata field names that would
                // otherwise collide with engine-internal row columns.
                if matches!(prop_name.as_str(), "__src__" | "__tgt__" | "__type__") {
                    return Err(ExecutionError::Unsupported(format!(
                        "edge property name '{prop_name}' is reserved for engine-internal \
                         metadata; choose a different name"
                    )));
                }
                let field_id = ctx.interner.intern(prop_name);
                let value = eval_expr(expr, row).map_to_document();
                if is_temporal && prop_name == "valid_from" {
                    // Accept both Int (epoch ms) and Timestamp (engine native).
                    // Reject Null (explicit null violates the temporal contract)
                    // and any other type.
                    valid_from_value = match &value {
                        Value::Int(ms) => Some(*ms),
                        Value::Timestamp(ms) => Some(*ms),
                        Value::Null => {
                            return Err(ExecutionError::Unsupported(format!(
                                "temporal edge '{edge_type}': valid_from must not be NULL"
                            )));
                        }
                        other => {
                            return Err(ExecutionError::Unsupported(format!(
                                "temporal edge '{edge_type}': valid_from must be INT or \
                                 TIMESTAMP (epoch milliseconds), got {other:?}"
                            )));
                        }
                    };
                }
                if is_temporal && prop_name == "valid_to" {
                    valid_to_value = match &value {
                        Value::Int(ms) => Some(*ms),
                        Value::Timestamp(ms) => Some(*ms),
                        Value::Null => None,
                        other => {
                            return Err(ExecutionError::Unsupported(format!(
                                "temporal edge '{edge_type}': valid_to must be INT, \
                                 TIMESTAMP, or NULL, got {other:?}"
                            )));
                        }
                    };
                }
                prop_map.push((field_id, value));
            }
            // Interval sanity check: valid_to (if set) must be > valid_from.
            // A zero-duration version (valid_to == valid_from) is rejected too
            // — temporal_active_at would never return true for it, so it is
            // never a useful piece of data and almost certainly a bug.
            if let (Some(vf), Some(vt)) = (valid_from_value, valid_to_value) {
                if vt <= vf {
                    return Err(ExecutionError::Unsupported(format!(
                        "temporal edge '{edge_type}': valid_to ({vt}) must be strictly \
                         greater than valid_from ({vf})"
                    )));
                }
            }
            let valid_from_for_key = if is_temporal { valid_from_value } else { None };
            ctx.mvcc_put_edge_props_either(
                edge_type,
                source_id,
                target_id,
                valid_from_for_key,
                &prop_map,
            )?;
            ctx.write_stats.properties_set += properties.len() as u64;
        }

        // Fire BEFORE COMMIT CREATE triggers registered on this edge type.
        // The trigger sees `$event = "CREATE"`, `$src` / `$tgt` as endpoint
        // NodeIds, `$edge_type` as the type name, and `$after` as the edge
        // property map.
        let resolved_props: Vec<(String, Value)> = properties
            .iter()
            .map(|(name, expr)| (name.clone(), eval_expr(expr, row).map_to_document()))
            .collect();
        let trigger_params =
            trigger_params_for_edge_create(edge_type, source_id, target_id, &resolved_props);
        let target_segment =
            coordinode_core::schema::triggers::TriggerTargetSchema::edge_type(edge_type)
                .index_key_segment();
        let matched = ctx.lookup_matching_triggers(&target_segment, "c")?;
        if !matched.is_empty() {
            fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
        }

        let mut out_row = row.clone();
        if let Some(ev) = edge_variable {
            out_row.insert(
                format!("{ev}.__type__"),
                Value::String(edge_type.to_string()),
            );
            // Also add edge properties to the output row
            for (prop_name, expr) in properties {
                let value = eval_expr(expr, row);
                out_row.insert(format!("{ev}.{prop_name}"), value);
            }
        }
        results.push(out_row);
    }

    Ok(results)
}

/// SET: update properties/labels on existing nodes.
///
/// When `violation_mode` is `ViolationMode::Skip`, nodes that would violate
/// schema constraints are silently skipped. The output contains only rows
/// for nodes that were successfully updated (or had no schema to check).
/// When `violation_mode` is `ViolationMode::Fail` (default), any schema
/// violation immediately aborts the entire SET with an error.
fn execute_update(
    input_rows: &[Row],
    items: &[crate::cypher::ast::SetItem],
    violation_mode: &ViolationMode,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let skip_on_violation = matches!(violation_mode, ViolationMode::Skip);

    let mut results = Vec::new();

    // R172c temporal-node SET routing. For each SET item targeting a node
    // on a temporal label, classify by property:
    //
    //   * `valid_from` → reject (immutable storage-key suffix).
    //   * `valid_to`   → mutate in place at the matched per-version key
    //     (close-version path, Phase 1).
    //   * Other property / label mutations → write a NEW version row at
    //     `valid_from = NOW` (close current + open new — Phase 2).
    //
    // Mixed `valid_to` + other items on the SAME variable in the SAME
    // clause is rejected as ambiguous: should `valid_to` close the
    // current version, or set the new version's valid_to? User must
    // split into separate statements.
    //
    // Edge SET items (Value::String binding) skip this whole block —
    // they have their own temporal handling in `update_edge_property`.
    let mut temporal_new_version_vars: std::collections::HashSet<(usize, String)> =
        std::collections::HashSet::new();
    let mut temporal_in_place_vars: std::collections::HashSet<(usize, String)> =
        std::collections::HashSet::new();
    for (row_idx, row) in input_rows.iter().enumerate() {
        for item in items {
            let var = match item {
                crate::cypher::ast::SetItem::Property { variable, .. }
                | crate::cypher::ast::SetItem::PropertyPath { variable, .. }
                | crate::cypher::ast::SetItem::DocFunction { variable, .. }
                | crate::cypher::ast::SetItem::ReplaceProperties { variable, .. }
                | crate::cypher::ast::SetItem::MergeProperties { variable, .. }
                | crate::cypher::ast::SetItem::AddLabel { variable, .. } => variable,
            };
            if !matches!(row.get(var), Some(Value::Int(_))) {
                continue;
            }
            let Some(Value::String(primary)) = row.get(&format!("{var}.__label__")) else {
                continue;
            };
            let is_temporal = ctx
                .load_current_label_schema(primary)
                .ok()
                .flatten()
                .is_some_and(|s| s.temporal);
            if !is_temporal {
                continue;
            }
            match item {
                crate::cypher::ast::SetItem::Property { property, .. }
                    if property == "valid_from" =>
                {
                    return Err(ExecutionError::Unsupported(format!(
                        "SET {var}.valid_from is rejected on temporal label '{primary}': \
                         valid_from is the version-key suffix and is immutable. To \
                         re-key a version, DELETE the row and CREATE a new one with \
                         the desired valid_from."
                    )));
                }
                crate::cypher::ast::SetItem::Property { property, .. }
                    if property == "valid_to" =>
                {
                    temporal_in_place_vars.insert((row_idx, var.clone()));
                }
                _ => {
                    temporal_new_version_vars.insert((row_idx, var.clone()));
                }
            }
        }
    }
    // Mixed in-place + new-version on the same (row, var) is ambiguous.
    for entry in &temporal_in_place_vars {
        if temporal_new_version_vars.contains(entry) {
            let (_idx, var) = entry;
            return Err(ExecutionError::Unsupported(format!(
                "SET clause mixes `{var}.valid_to = …` (close-version) with other \
                 property mutations on the same temporal node in the same clause. \
                 This is ambiguous: should valid_to close the current version or \
                 set the new version's interval? Split into separate statements: \
                 first close the current version with `SET {var}.valid_to = …`, \
                 then CREATE a new version (or vice versa with the open+set form)."
            )));
        }
    }

    // 'row_loop label: when violation_mode == Skip, `continue 'row_loop` silently
    // drops the row instead of propagating the schema error.
    'row_loop: for (row_idx, row) in input_rows.iter().enumerate() {
        let mut out_row = row.clone();

        // R172c Phase 2: close-current + open-new processing for temporal
        // nodes mutated by non-`valid_to` SET items. This block fires
        // BEFORE the normal SET loop. For each (var) on this row that the
        // pre-scan classified as "needs new version", we:
        //   1. Read the current matched version's record (via the bound
        //      `n.valid_from`-suffixed key).
        //   2. Clone it; apply all relevant SET items to the clone.
        //   3. Close current: rewrite the matched record with
        //      `valid_to = NOW`.
        //   4. Open new: write the clone at a fresh per-version key
        //      `node:<shard>:<node_id>:<NOW>` with `valid_to = NULL` and
        //      a fresh `__ingestion_ts__`. Same `node_id` — the new row
        //      is a new VERSION of the same logical node.
        //
        // All non-`valid_to` SET items targeting a temporal node are then
        // marked processed and skipped in the main SET loop below.
        let mut processed_temporal_items: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let temporal_new_vars_for_row: Vec<String> = temporal_new_version_vars
            .iter()
            .filter(|(idx, _)| *idx == row_idx)
            .map(|(_, v)| v.clone())
            .collect();
        if !temporal_new_vars_for_row.is_empty() {
            let now_us = current_hlc_us() as i64;
            for var in &temporal_new_vars_for_row {
                let node_id = match out_row.get(var) {
                    Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                    _ => continue,
                };
                let current_valid_from = match out_row.get(&format!("{var}.valid_from")) {
                    Some(Value::Int(ms)) => *ms,
                    Some(Value::Timestamp(ms)) => *ms,
                    _ => {
                        return Err(ExecutionError::Unsupported(format!(
                            "temporal SET on `{var}`: matched row is missing \
                             `{var}.valid_from` (the planner must surface valid_from \
                             on every temporal node materialised for mutation)"
                        )));
                    }
                };
                // NOW must be strictly greater than the matched version's
                // valid_from so the new version's interval is valid. If the
                // wall clock is at or before valid_from (legitimate during
                // backfill / replay scenarios) we bump to valid_from + 1 µs.
                let new_valid_from = if now_us > current_valid_from {
                    now_us
                } else {
                    current_valid_from + 1
                };

                // Step 1: read current matched version record.
                let mut closing_record = ctx
                    .mvcc_get_node_temporal(ctx.shard_id, node_id, current_valid_from)?
                    .ok_or_else(|| {
                        ExecutionError::Unsupported(format!(
                            "temporal SET on `{var}`: matched version record not \
                         found at (node_id={node_id}, valid_from={current_valid_from})"
                        ))
                    })?;
                let mut new_record = closing_record.clone();

                // Step 2: apply each relevant SET item to the new record.
                // Mark each item's index in `processed_temporal_items` so
                // the main SET loop skips it for this row.
                for (item_idx, item) in items.iter().enumerate() {
                    let item_var = match item {
                        crate::cypher::ast::SetItem::Property { variable, .. }
                        | crate::cypher::ast::SetItem::PropertyPath { variable, .. }
                        | crate::cypher::ast::SetItem::DocFunction { variable, .. }
                        | crate::cypher::ast::SetItem::ReplaceProperties { variable, .. }
                        | crate::cypher::ast::SetItem::MergeProperties { variable, .. }
                        | crate::cypher::ast::SetItem::AddLabel { variable, .. } => {
                            variable.as_str()
                        }
                    };
                    if item_var != var.as_str() {
                        continue;
                    }
                    // Skip valid_to (in-place) and valid_from (rejected at pre-scan).
                    if let crate::cypher::ast::SetItem::Property { property, .. } = item {
                        if property == "valid_to" || property == "valid_from" {
                            continue;
                        }
                    }
                    match item {
                        crate::cypher::ast::SetItem::Property { property, expr, .. } => {
                            let val = eval_expr(expr, &out_row).map_to_document();
                            let field_id = ctx.interner.intern(property);
                            new_record.set(field_id, val);
                        }
                        crate::cypher::ast::SetItem::AddLabel { label, .. } => {
                            new_record.add_label(label.clone());
                        }
                        crate::cypher::ast::SetItem::ReplaceProperties { expr, .. } => {
                            let val = eval_expr(expr, &out_row);
                            if let Value::Map(map) = val {
                                // Clear existing user props, keep engine-managed
                                // fields (__ingestion_ts__, valid_from, valid_to
                                // are reapplied below).
                                new_record.props.clear();
                                new_record.extra = None;
                                for (k, v) in map {
                                    let fid = ctx.interner.intern(&k);
                                    new_record.set(fid, v.map_to_document());
                                }
                            }
                        }
                        crate::cypher::ast::SetItem::MergeProperties { expr, .. } => {
                            let val = eval_expr(expr, &out_row);
                            if let Value::Map(map) = val {
                                for (k, v) in map {
                                    let fid = ctx.interner.intern(&k);
                                    new_record.set(fid, v.map_to_document());
                                }
                            }
                        }
                        crate::cypher::ast::SetItem::PropertyPath { path, expr, .. } => {
                            // R172c Phase 3b: nested PropertyPath SET on
                            // temporal. Build the same DocDelta the
                            // non-temporal path queues as a merge operand,
                            // but apply it in-memory to `new_record` so the
                            // close+open writes carry the post-delta state.
                            let val = eval_expr(expr, &out_row).map_to_document();
                            if path.is_empty() {
                                return Err(ExecutionError::Unsupported(format!(
                                    "SET on temporal node `{var}`: empty property path"
                                )));
                            }
                            let field_id = ctx.interner.intern(&path[0]);
                            let sub_path = path[1..].to_vec();
                            let delta = coordinode_core::graph::doc_delta::DocDelta::SetPath {
                                target: coordinode_core::graph::doc_delta::PathTarget::PropField(
                                    field_id,
                                ),
                                path: sub_path,
                                value: val.to_rmpv(),
                            };
                            coordinode_storage::engine::merge::apply_doc_deltas_to_record(
                                &mut new_record,
                                &[delta],
                            );
                            ctx.write_stats.properties_set += 1;
                            // Surface the leaf path in out_row for RETURN.
                            let path_str = path.join(".");
                            out_row.insert(format!("{var}.{path_str}"), val);
                        }
                        crate::cypher::ast::SetItem::DocFunction {
                            function,
                            path,
                            value_expr,
                            ..
                        } => {
                            // R172c Phase 3b: doc_push / doc_pull /
                            // doc_add_to_set / doc_inc on temporal nodes.
                            // Same construction as the non-temporal path,
                            // but applied in-memory to `new_record`.
                            let val = eval_expr(value_expr, &out_row);
                            let (field_id, sub_path) = if path.is_empty() {
                                (ctx.interner.intern(var), vec![])
                            } else {
                                (ctx.interner.intern(&path[0]), path[1..].to_vec())
                            };
                            let target =
                                coordinode_core::graph::doc_delta::PathTarget::PropField(field_id);
                            let delta = match function.as_str() {
                                "doc_push" => {
                                    coordinode_core::graph::doc_delta::DocDelta::ArrayPush {
                                        target,
                                        path: sub_path,
                                        value: val.to_rmpv(),
                                    }
                                }
                                "doc_pull" => {
                                    coordinode_core::graph::doc_delta::DocDelta::ArrayPull {
                                        target,
                                        path: sub_path,
                                        value: val.to_rmpv(),
                                    }
                                }
                                "doc_add_to_set" => {
                                    coordinode_core::graph::doc_delta::DocDelta::ArrayAddToSet {
                                        target,
                                        path: sub_path,
                                        value: val.to_rmpv(),
                                    }
                                }
                                "doc_inc" => {
                                    let amount = match &val {
                                        Value::Int(i) => *i as f64,
                                        Value::Float(f) => *f,
                                        _ => 0.0,
                                    };
                                    coordinode_core::graph::doc_delta::DocDelta::Increment {
                                        target,
                                        path: sub_path,
                                        amount,
                                    }
                                }
                                other => {
                                    return Err(ExecutionError::Unsupported(format!(
                                        "unknown doc function on temporal node `{var}`: {other}"
                                    )));
                                }
                            };
                            coordinode_storage::engine::merge::apply_doc_deltas_to_record(
                                &mut new_record,
                                &[delta],
                            );
                            ctx.write_stats.properties_set += 1;
                        }
                    }
                    processed_temporal_items.insert(item_idx);
                }

                // Apply the new version's bitemporal axes:
                // - new record: valid_from = NOW, valid_to = NULL (open),
                //   refreshed __ingestion_ts__.
                // - closing record: valid_to = NOW.
                let vf_fid = ctx.interner.intern("valid_from");
                let vt_fid = ctx.interner.intern("valid_to");
                let its_fid = ctx.interner.intern("__ingestion_ts__");
                new_record.set(vf_fid, Value::Int(new_valid_from));
                new_record.props.remove(&vt_fid);
                new_record.set(its_fid, Value::Int(now_us));
                closing_record.set(vt_fid, Value::Int(new_valid_from));

                // Step 3: write close-current (mutate at same per-version key).
                ctx.mvcc_put_node_temporal(
                    ctx.shard_id,
                    node_id,
                    current_valid_from,
                    &closing_record,
                )?;

                // Step 4: write open-new (at fresh per-version key for NOW).
                ctx.mvcc_put_node_temporal(ctx.shard_id, node_id, new_valid_from, &new_record)?;
                ctx.write_stats.nodes_created += 1;

                // Reflect the new version's prop columns in the output row.
                out_row.insert(format!("{var}.valid_from"), Value::Int(new_valid_from));
                out_row.insert(format!("{var}.valid_to"), Value::Null);
                out_row.insert(format!("{var}.__ingestion_ts__"), Value::Int(now_us));
                for (&field_id, value) in &new_record.props {
                    if let Some(name) = ctx.interner.resolve(field_id) {
                        if name == "valid_from" || name == "valid_to" || name == "__ingestion_ts__"
                        {
                            continue;
                        }
                        out_row.insert(format!("{var}.{name}"), value.clone());
                    }
                }
            }
        }

        // Snapshot the pre-mutation state of each node variable referenced
        // by the SET items in this row, so we can fire UPDATE triggers
        // with `$before` after all items apply.
        let mut update_snapshots: std::collections::HashMap<
            NodeId,
            (Vec<String>, std::collections::BTreeMap<String, Value>),
        > = std::collections::HashMap::new();
        // Edge UPDATE snapshots: pre-mutation property map keyed by the edge
        // variable name (one logical edge per variable per row). Carries the
        // resolved key so we can read the post-state through the same key
        // after the SET items apply. Temporal edges resolve to a per-version
        // key (keyed on `valid_from`); non-temporal resolve to a single key.
        struct EdgeUpdateSnapshot {
            edge_type: String,
            src: NodeId,
            tgt: NodeId,
            /// `Some(vf)` for temporal edges, `None` for non-temporal.
            /// Lets the post-SET re-read pick the same per-version /
            /// non-temporal EdgeProp key via `mvcc_get_edge_props_either`.
            valid_from_ms: Option<i64>,
            before: std::collections::BTreeMap<String, Value>,
        }
        let mut edge_update_snapshots: std::collections::HashMap<String, EdgeUpdateSnapshot> =
            std::collections::HashMap::new();
        for (snap_item_idx, item) in items.iter().enumerate() {
            // R172c Phase 2: skip items already processed in the close+open
            // pass above. Their snapshot is implicitly the pre-mutation
            // record (which still exists at the matched per-version key
            // until we rewrote its valid_to). For UPDATE-trigger purposes
            // the closing snapshot is captured by the close+open block.
            if processed_temporal_items.contains(&snap_item_idx) {
                continue;
            }
            let variable = match item {
                crate::cypher::ast::SetItem::Property { variable, .. }
                | crate::cypher::ast::SetItem::PropertyPath { variable, .. }
                | crate::cypher::ast::SetItem::DocFunction { variable, .. }
                | crate::cypher::ast::SetItem::ReplaceProperties { variable, .. }
                | crate::cypher::ast::SetItem::MergeProperties { variable, .. }
                | crate::cypher::ast::SetItem::AddLabel { variable, .. } => variable,
            };
            // Edge mutation only flows through `SetItem::Property` —
            // `update_edge_property` is the single edgeprop-writing path.
            // The other variants (`PropertyPath`, `DocFunction`,
            // `MergeProperties`, `ReplaceProperties`, `AddLabel`) require
            // `Value::Int` and silently no-op on edge variables. Snapshotting
            // for those would fire an UPDATE trigger with `$before == $after`
            // — semantically wrong (no mutation happened) and a needless
            // round trip through the trigger body.
            let is_property_mutation = matches!(item, crate::cypher::ast::SetItem::Property { .. });
            // Edge variable: bound to Value::String(edge_type). Snapshot the
            // current edge property map so we can fire UPDATE triggers with
            // `$before` / `$after` after the mutation lands. We probe the
            // trigger index lazily after applying SET — but the snapshot
            // must be taken BEFORE, since `update_edge_property` overwrites
            // the edgeprop value in the same key.
            if let Some(Value::String(edge_type)) = out_row.get(variable).cloned() {
                if !is_property_mutation {
                    // No mutation path exists for this variant on an edge.
                    continue;
                }
                if edge_update_snapshots.contains_key(variable) {
                    continue;
                }
                let src_raw = match out_row.get(&format!("{variable}.__src__")) {
                    Some(Value::Int(n)) => *n as u64,
                    _ => continue,
                };
                let tgt_raw = match out_row.get(&format!("{variable}.__tgt__")) {
                    Some(Value::Int(n)) => *n as u64,
                    _ => continue,
                };
                let src = NodeId::from_raw(src_raw);
                let tgt = NodeId::from_raw(tgt_raw);
                let is_temporal = lookup_edge_type_temporal(&edge_type, ctx)?;
                let valid_from_ms: Option<i64> = if is_temporal {
                    match out_row.get(&format!("{variable}.valid_from")) {
                        Some(Value::Int(ms)) => Some(*ms),
                        _ => continue,
                    }
                } else {
                    None
                };
                let before =
                    match ctx.mvcc_get_edge_props_either(&edge_type, src, tgt, valid_from_ms)? {
                        Some(prop_map) => decode_edgeprop_map_into_named(&prop_map, ctx),
                        None => std::collections::BTreeMap::new(),
                    };
                edge_update_snapshots.insert(
                    variable.clone(),
                    EdgeUpdateSnapshot {
                        edge_type,
                        src,
                        tgt,
                        valid_from_ms,
                        before,
                    },
                );
                continue;
            }
            let node_id = match out_row.get(variable) {
                Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                _ => continue,
            };
            if update_snapshots.contains_key(&node_id) {
                continue;
            }
            // Use the delta-non-materialising peek (same read used by
            // SET's own schema checks) to avoid consuming any pending
            // merge_node_deltas from a preceding REMOVE clause in the
            // same query. mvcc_get would materialise those deltas into
            // the write buffer, and in legacy (no-oracle) mode
            // mvcc_flush ignores the buffer — the deltas would be lost.
            if let Some(record) = ctx.schema_peek_node_typed(ctx.shard_id, node_id)? {
                update_snapshots.insert(node_id, snapshot_node_record(&record, ctx));
            }
        }

        for (item_idx, item) in items.iter().enumerate() {
            // R172c Phase 2: skip items already handled by the close+open
            // path at the top of this row's processing. Their property
            // values are already in the new version's record and the
            // output row was updated to reflect that state.
            if processed_temporal_items.contains(&item_idx) {
                continue;
            }
            match item {
                crate::cypher::ast::SetItem::Property {
                    variable,
                    property,
                    expr,
                } => {
                    // R172a reserved-name guard at SET time: `__ingestion_ts__`
                    // is engine-owned on temporal labels (auto-populated with
                    // HLC commit-ts, user-immutable). Reject before any storage
                    // mutation so the bitemporal contract cannot be subverted
                    // via `SET n.__ingestion_ts__ = ...`. Edge metadata names
                    // (`__src__`/`__tgt__`/`__type__`) are rejected separately
                    // in `update_edge_property` for the edge SET path.
                    if property == "__ingestion_ts__" {
                        return Err(ExecutionError::Unsupported(
                            "SET on '__ingestion_ts__' is reserved: this field is \
                             engine-managed on temporal labels and cannot be \
                             assigned by SET"
                                .into(),
                        ));
                    }

                    // Map literals → Document for nested property storage.
                    let val = eval_expr(expr, &out_row).map_to_document();

                    // Edge variable bindings carry Value::String(edge_type),
                    // not Value::Int. Route to the edge-prop update path,
                    // which preserves the existing edgeprop key (non-temporal
                    // single key OR temporal per-version key keyed on the
                    // matched valid_from).
                    if let Some(Value::String(edge_type)) = out_row.get(variable).cloned() {
                        update_edge_property(
                            variable,
                            &edge_type,
                            property,
                            val,
                            &mut out_row,
                            ctx,
                        )?;
                        out_row.insert(format!("{variable}.{property}"), eval_expr(expr, row));
                        continue;
                    }

                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    // R172c temporal node close-version path: when the label
                    // is TEMPORAL and the property being set is `valid_to`,
                    // mutate the record at the per-version (25-byte) key
                    // bound by the row's `valid_from`. The valid_from key
                    // suffix does NOT change — only the in-value `valid_to`
                    // field flips from NULL (open) to a concrete end-of-
                    // validity timestamp (closed). Pre-check temporal label
                    // routing already rejected `valid_from` SETs and any
                    // non-`valid_to` Property/PropertyPath/etc. on temporal
                    // labels, so reaching this point implies a legitimate
                    // close-version write.
                    let is_temporal_label = out_row
                        .get(&format!("{variable}.__label__"))
                        .and_then(|v| match v {
                            Value::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .and_then(|lbl| ctx.load_current_label_schema(&lbl).ok().flatten())
                        .is_some_and(|s| s.temporal);
                    if is_temporal_label && property == "valid_to" {
                        // The bound row carries `n.valid_from` from the
                        // earlier NodeScan; without it we cannot locate
                        // the per-version key for this match. A missing
                        // value here means the planner produced a row
                        // referencing a temporal node without surfacing
                        // its `valid_from` — that's a planner bug, not a
                        // user-recoverable state, so fail loudly.
                        let valid_from = match out_row.get(&format!("{variable}.valid_from")) {
                            Some(Value::Int(ms)) => *ms,
                            Some(Value::Timestamp(ms)) => *ms,
                            _ => {
                                return Err(ExecutionError::Unsupported(format!(
                                    "SET {variable}.valid_to on temporal node: matched \
                                     row is missing `{variable}.valid_from` (planner \
                                     must surface valid_from for temporal node mutations)"
                                )));
                            }
                        };
                        // Validate new valid_to per the temporal interval
                        // contract (must be > valid_from, or NULL to reopen).
                        let new_valid_to: Option<i64> = match &val {
                            Value::Int(ms) => Some(*ms),
                            Value::Timestamp(ms) => Some(*ms),
                            Value::Null => None,
                            other => {
                                return Err(ExecutionError::Unsupported(format!(
                                    "SET {variable}.valid_to on temporal node: value \
                                     must be INT, TIMESTAMP, or NULL, got {other:?}"
                                )));
                            }
                        };
                        if let Some(vt) = new_valid_to {
                            if vt <= valid_from {
                                return Err(ExecutionError::Unsupported(format!(
                                    "SET {variable}.valid_to ({vt}) must be strictly \
                                     greater than valid_from ({valid_from})"
                                )));
                            }
                        }
                        let mut record = ctx
                            .mvcc_get_node_temporal(ctx.shard_id, node_id, valid_from)?
                            .ok_or_else(|| {
                                ExecutionError::Unsupported(format!(
                                    "SET {variable}.valid_to: temporal node record \
                                     not found at version (node_id={node_id}, \
                                     valid_from={valid_from})"
                                ))
                            })?;
                        let field_id = ctx.interner.intern("valid_to");
                        match new_valid_to {
                            Some(vt) => record.set(field_id, Value::Int(vt)),
                            None => {
                                // Re-open a closed version: drop the
                                // `valid_to` field entirely so the version
                                // becomes open again.
                                record.props.remove(&field_id);
                            }
                        }
                        ctx.mvcc_put_node_temporal(ctx.shard_id, node_id, valid_from, &record)?;
                        ctx.write_stats.properties_set += 1;
                        out_row.insert(
                            format!("{variable}.valid_to"),
                            match new_valid_to {
                                Some(vt) => Value::Int(vt),
                                None => Value::Null,
                            },
                        );
                        continue;
                    }

                    // Read current node record
                    if let Some(mut record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                        // Enforce schema mode before writing the property.
                        // Load the label schema for the node's primary label and
                        // check STRICT/VALIDATED constraints on the property name.
                        // Returns None if no schema exists (schemaless node → always allowed).
                        let label = record.primary_label().to_string();
                        let label_schema = ctx.load_current_label_schema(&label)?;

                        // Collect potential schema violation into Option<ExecutionError> so we can
                        // choose between skip (ON VIOLATION SKIP) and fail (default) after the check.
                        let schema_err: Option<ExecutionError> = if let Some(ref ls) = label_schema
                        {
                            match ls.mode {
                                SchemaMode::Strict => match ls.get_property(property) {
                                    None => Some(ExecutionError::SchemaViolation(format!(
                                        "unknown property '{property}' for strict label '{label}'"
                                    ))),
                                    Some(def) if def.is_computed() => {
                                        Some(ExecutionError::SchemaViolation(format!(
                                            "cannot SET computed property '{property}'"
                                        )))
                                    }
                                    Some(def) => validate_one(property, &val, def)
                                        .map_err(|e| ExecutionError::SchemaViolation(e.to_string()))
                                        .err(),
                                },
                                SchemaMode::Validated => {
                                    if let Some(def) = ls.get_property(property) {
                                        if def.is_computed() {
                                            Some(ExecutionError::SchemaViolation(format!(
                                                "cannot SET computed property '{property}'"
                                            )))
                                        } else {
                                            validate_one(property, &val, def)
                                                .map_err(|e| {
                                                    ExecutionError::SchemaViolation(e.to_string())
                                                })
                                                .err()
                                        }
                                    } else {
                                        None
                                    }
                                }
                                SchemaMode::Flexible => None,
                            }
                        } else {
                            None
                        };

                        // ON VIOLATION SKIP: silently drop this row and move to next.
                        // Default (Fail): propagate the error immediately.
                        if let Some(err) = schema_err {
                            if skip_on_violation {
                                continue 'row_loop;
                            }
                            return Err(err);
                        }

                        let field_id = ctx.interner.intern(property);

                        // Capture old value BEFORE updating the record, so
                        // the B-tree index can remove the stale entry.
                        let old_value: Option<Value> = record.get(field_id).cloned();

                        // Update B-tree index: remove old entry, add new entry
                        // (which also enforces uniqueness on the new value).
                        // Must happen BEFORE writing the record to storage so
                        // that a unique violation rolls back cleanly.
                        if let Some(btree_reg) = ctx.btree_index_registry {
                            let label = record.primary_label().to_string();
                            btree_reg
                                .on_property_changed(
                                    ctx.engine,
                                    node_id,
                                    &label,
                                    property,
                                    old_value.as_ref(),
                                    &val,
                                )
                                .map_err(|v| {
                                    ExecutionError::Conflict(format!(
                                        "unique constraint violated on index `{}`: \
                                         property `{}` already has value {:?}",
                                        v.index_name, v.property, v.value
                                    ))
                                })?;
                        }

                        record.set(field_id, val.clone());

                        ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;
                        ctx.write_stats.properties_set += 1;

                        // Notify vector index registry if setting a vector property.
                        if let Some(registry) = ctx.vector_index_registry {
                            if let Some(vec_data) = try_extract_vector(&val) {
                                let label = record.primary_label().to_string();
                                registry.on_vector_written(&label, node_id, property, &vec_data);
                            }
                        }

                        // Notify text index registry if setting a text property.
                        if let Some(registry) = ctx.text_index_registry {
                            if let Some(text) = val.as_str() {
                                let label = record.primary_label().to_string();
                                registry.on_text_written(&label, node_id, property, text);
                            }
                        }

                        // Reflect the new value in the output row only when the
                        // write was actually applied. If mvcc_get returned None
                        // (e.g. node was DELETEd earlier in the same query), the
                        // `if let Some` block above is skipped and we must not
                        // expose the unapplied value through RETURN.
                        out_row.insert(format!("{variable}.{property}"), val);
                    }
                }
                crate::cypher::ast::SetItem::PropertyPath {
                    variable,
                    path,
                    expr,
                } => {
                    let val = eval_expr(expr, &out_row);
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    // Schema check for the root property (path[0]).
                    // PropertyPath writes nested doc fields (e.g. SET n.config.host = "x").
                    // In STRICT mode the root property must be declared in the schema.
                    // In VALIDATED mode unknown root props are accepted (extra fields allowed).
                    // The merge operand write is O(1); this read is only for schema validation
                    // and is skipped when no schema exists (schemaless node) or mode = FLEXIBLE.
                    // schema_label_for_node caches the primary label per node per statement:
                    // SET n.a.x=1, n.a.y=2, n.a.z=3 on 100 nodes = 100 reads (not 300).
                    let schema_err: Option<ExecutionError> = {
                        if let Some(label) = ctx.schema_label_for_node(ctx.shard_id, node_id)? {
                            match ctx.load_current_label_schema(&label)? {
                                Some(ls) => {
                                    let root = &path[0];
                                    match ls.mode {
                                        SchemaMode::Strict => match ls.get_property(root) {
                                            None => Some(ExecutionError::SchemaViolation(format!(
                                                "unknown property '{root}' for strict label '{label}'"
                                            ))),
                                            Some(def) if def.is_computed() => {
                                                Some(ExecutionError::SchemaViolation(format!(
                                                    "cannot SET computed property '{root}'"
                                                )))
                                            }
                                            Some(_) => None,
                                        },
                                        SchemaMode::Validated => {
                                            if let Some(def) = ls.get_property(root) {
                                                if def.is_computed() {
                                                    Some(ExecutionError::SchemaViolation(format!(
                                                        "cannot SET computed property '{root}'"
                                                    )))
                                                } else {
                                                    None
                                                }
                                            } else {
                                                None
                                            }
                                        }
                                        SchemaMode::Flexible => None,
                                    }
                                }
                                None => None,
                            }
                        } else {
                            None
                        }
                    };
                    if let Some(err) = schema_err {
                        if skip_on_violation {
                            continue 'row_loop;
                        }
                        return Err(err);
                    }

                    // O(1) write via merge operand.
                    // path[0] = property name → resolved to field_id via interner
                    // path[1..] = nested path within the DOCUMENT value
                    let field_id = ctx.interner.intern(&path[0]);
                    let sub_path = &path[1..];

                    let delta = coordinode_core::graph::doc_delta::DocDelta::SetPath {
                        target: coordinode_core::graph::doc_delta::PathTarget::PropField(field_id),
                        path: sub_path.to_vec(),
                        value: val.to_rmpv(),
                    };
                    let operand = delta.encode().map_err(|e| {
                        ExecutionError::Serialization(format!("DocDelta encode: {e}"))
                    })?;
                    ctx.mvcc_merge_node_delta(ctx.shard_id, node_id, operand);
                    ctx.write_stats.properties_set += 1;

                    let path_str = path.join(".");
                    out_row.insert(format!("{variable}.{path_str}"), val);
                }
                crate::cypher::ast::SetItem::DocFunction {
                    function,
                    variable,
                    path,
                    value_expr,
                } => {
                    let val = eval_expr(value_expr, &out_row);
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    // Schema validation: root property must be declared in STRICT mode.
                    // DocFunction path[0] is the root property (e.g. `doc` in `doc_push(n.doc, v)`).
                    // If path is empty the function targets the node itself (edge case), use variable.
                    let root_prop = if path.is_empty() {
                        variable.as_str()
                    } else {
                        path[0].as_str()
                    };
                    // schema_label_for_node caches the primary label per node per statement —
                    // same invariant as PropertyPath: must not trigger RYOW materialization.
                    let schema_err: Option<ExecutionError> = {
                        if let Some(label) = ctx.schema_label_for_node(ctx.shard_id, node_id)? {
                            match ctx.load_current_label_schema(&label)? {
                                Some(ls) => match ls.mode {
                                    SchemaMode::Strict => match ls.get_property(root_prop) {
                                        None => Some(ExecutionError::SchemaViolation(format!(
                                            "unknown property '{root_prop}' for strict label '{label}'"
                                        ))),
                                        Some(def) if def.is_computed() => {
                                            Some(ExecutionError::SchemaViolation(format!(
                                                "cannot SET computed property '{root_prop}'"
                                            )))
                                        }
                                        Some(_) => None,
                                    },
                                    SchemaMode::Validated => {
                                        if let Some(def) = ls.get_property(root_prop) {
                                            if def.is_computed() {
                                                Some(ExecutionError::SchemaViolation(format!(
                                                    "cannot SET computed property '{root_prop}'"
                                                )))
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    }
                                    SchemaMode::Flexible => None,
                                },
                                None => None,
                            }
                        } else {
                            None
                        }
                    };
                    if let Some(err) = schema_err {
                        if skip_on_violation {
                            continue 'row_loop;
                        }
                        return Err(err);
                    }

                    // Resolve property name → field_id. For doc_* functions,
                    // path[0] is the root property, path[1..] is the nested path.
                    // If path is empty (bare variable, e.g. doc_push(n, "x")),
                    // use the variable name as the property — unlikely but handled.
                    let (field_id, sub_path) = if path.is_empty() {
                        // Bare variable — treat variable as property name on itself.
                        // This is an edge case; normally path has at least one element.
                        (ctx.interner.intern(variable), vec![])
                    } else {
                        (ctx.interner.intern(&path[0]), path[1..].to_vec())
                    };

                    let target = coordinode_core::graph::doc_delta::PathTarget::PropField(field_id);

                    let delta = match function.as_str() {
                        "doc_push" => coordinode_core::graph::doc_delta::DocDelta::ArrayPush {
                            target,
                            path: sub_path,
                            value: val.to_rmpv(),
                        },
                        "doc_pull" => coordinode_core::graph::doc_delta::DocDelta::ArrayPull {
                            target,
                            path: sub_path,
                            value: val.to_rmpv(),
                        },
                        "doc_add_to_set" => {
                            coordinode_core::graph::doc_delta::DocDelta::ArrayAddToSet {
                                target,
                                path: sub_path,
                                value: val.to_rmpv(),
                            }
                        }
                        "doc_inc" => {
                            let amount = match &val {
                                Value::Int(i) => *i as f64,
                                Value::Float(f) => *f,
                                _ => 0.0,
                            };
                            coordinode_core::graph::doc_delta::DocDelta::Increment {
                                target,
                                path: sub_path,
                                amount,
                            }
                        }
                        other => {
                            return Err(ExecutionError::Unsupported(format!(
                                "unknown doc function: {other}"
                            )));
                        }
                    };

                    let operand = delta.encode().map_err(|e| {
                        ExecutionError::Serialization(format!("DocDelta encode: {e}"))
                    })?;
                    ctx.mvcc_merge_node_delta(ctx.shard_id, node_id, operand);
                    ctx.write_stats.properties_set += 1;
                }
                crate::cypher::ast::SetItem::ReplaceProperties { variable, expr } => {
                    let map_val = eval_expr(expr, &out_row);
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    if let Some(mut record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                        // Schema validation: SET n = {map} replaces ALL properties.
                        // In STRICT mode every key must be declared; VALIDATED checks declared keys only.
                        let label = record.primary_label().to_string();
                        let label_schema = ctx.load_current_label_schema(&label)?;
                        if let (Some(ref ls), Value::Map(ref map)) = (&label_schema, &map_val) {
                            let schema_err: Option<ExecutionError> = 'schema: {
                                for (k, v) in map {
                                    match ls.mode {
                                        SchemaMode::Strict => match ls.get_property(k) {
                                            None => break 'schema Some(
                                                ExecutionError::SchemaViolation(format!(
                                                    "unknown property '{k}' for strict label '{label}'"
                                                )),
                                            ),
                                            Some(def) if def.is_computed() => break 'schema Some(
                                                ExecutionError::SchemaViolation(format!(
                                                    "cannot SET computed property '{k}'"
                                                )),
                                            ),
                                            Some(def) => {
                                                if let Err(e) = validate_one(k, v, def) {
                                                    break 'schema Some(
                                                        ExecutionError::SchemaViolation(
                                                            e.to_string(),
                                                        ),
                                                    );
                                                }
                                            }
                                        },
                                        SchemaMode::Validated => {
                                            if let Some(def) = ls.get_property(k) {
                                                if def.is_computed() {
                                                    break 'schema Some(
                                                        ExecutionError::SchemaViolation(format!(
                                                            "cannot SET computed property '{k}'"
                                                        )),
                                                    );
                                                }
                                                if let Err(e) = validate_one(k, v, def) {
                                                    break 'schema Some(
                                                        ExecutionError::SchemaViolation(
                                                            e.to_string(),
                                                        ),
                                                    );
                                                }
                                            }
                                        }
                                        SchemaMode::Flexible => {}
                                    }
                                }
                                None
                            };
                            if let Some(err) = schema_err {
                                if skip_on_violation {
                                    continue 'row_loop;
                                }
                                return Err(err);
                            }
                        }

                        // Clear existing props and set new ones from map
                        record.props.clear();
                        if let Value::Map(ref map) = map_val {
                            for (k, v) in map {
                                let field_id = ctx.interner.intern(k);
                                record.set(field_id, v.clone());
                            }
                        }

                        ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;

                        // Update out_row so that RETURN clauses in the same statement
                        // see the post-SET values. Remove all old variable.* entries
                        // (replaced), then insert the new map contents.
                        let prefix = format!("{variable}.");
                        out_row.retain(|k, _| !k.starts_with(&prefix));
                        if let Value::Map(ref map) = map_val {
                            for (k, v) in map {
                                out_row.insert(format!("{variable}.{k}"), v.clone());
                            }
                        }

                        // Notify vector index registry for any vector properties in the replacement map.
                        if let Some(registry) = ctx.vector_index_registry {
                            let label = record.primary_label().to_string();
                            if let Value::Map(ref map) = map_val {
                                for (k, v) in map {
                                    if let Some(vec_data) = try_extract_vector(v) {
                                        registry.on_vector_written(&label, node_id, k, &vec_data);
                                    }
                                }
                            }
                        }

                        // Notify text index registry for any text properties in the replacement map.
                        if let Some(registry) = ctx.text_index_registry {
                            let label = record.primary_label().to_string();
                            if let Value::Map(ref map) = map_val {
                                for (k, v) in map {
                                    if let Some(text) = v.as_str() {
                                        registry.on_text_written(&label, node_id, k, text);
                                    }
                                }
                            }
                        }
                    }
                }
                crate::cypher::ast::SetItem::MergeProperties { variable, expr } => {
                    let map_val = eval_expr(expr, &out_row);
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    if let Some(mut record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                        // Schema validation: SET n += {map} merges new properties.
                        // STRICT: every key in map must be declared; VALIDATED: checks declared keys.
                        let label = record.primary_label().to_string();
                        let label_schema = ctx.load_current_label_schema(&label)?;
                        if let (Some(ref ls), Value::Map(ref map)) = (&label_schema, &map_val) {
                            let schema_err: Option<ExecutionError> = 'schema: {
                                for (k, v) in map {
                                    match ls.mode {
                                        SchemaMode::Strict => match ls.get_property(k) {
                                            None => break 'schema Some(
                                                ExecutionError::SchemaViolation(format!(
                                                    "unknown property '{k}' for strict label '{label}'"
                                                )),
                                            ),
                                            Some(def) if def.is_computed() => break 'schema Some(
                                                ExecutionError::SchemaViolation(format!(
                                                    "cannot SET computed property '{k}'"
                                                )),
                                            ),
                                            Some(def) => {
                                                if let Err(e) = validate_one(k, v, def) {
                                                    break 'schema Some(
                                                        ExecutionError::SchemaViolation(
                                                            e.to_string(),
                                                        ),
                                                    );
                                                }
                                            }
                                        },
                                        SchemaMode::Validated => {
                                            if let Some(def) = ls.get_property(k) {
                                                if def.is_computed() {
                                                    break 'schema Some(
                                                        ExecutionError::SchemaViolation(format!(
                                                            "cannot SET computed property '{k}'"
                                                        )),
                                                    );
                                                }
                                                if let Err(e) = validate_one(k, v, def) {
                                                    break 'schema Some(
                                                        ExecutionError::SchemaViolation(
                                                            e.to_string(),
                                                        ),
                                                    );
                                                }
                                            }
                                        }
                                        SchemaMode::Flexible => {}
                                    }
                                }
                                None
                            };
                            if let Some(err) = schema_err {
                                if skip_on_violation {
                                    continue 'row_loop;
                                }
                                return Err(err);
                            }
                        }

                        if let Value::Map(ref map) = map_val {
                            for (k, v) in map {
                                let field_id = ctx.interner.intern(k);
                                record.set(field_id, v.clone());
                            }
                        }

                        ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;

                        // Update out_row so RETURN clauses see the merged values.
                        // MergeProperties adds/overwrites; existing untouched props stay.
                        if let Value::Map(ref map) = map_val {
                            for (k, v) in map {
                                out_row.insert(format!("{variable}.{k}"), v.clone());
                            }
                        }

                        // Notify vector index registry for any vector properties in the merged map.
                        if let Some(registry) = ctx.vector_index_registry {
                            let label = record.primary_label().to_string();
                            if let Value::Map(ref map) = map_val {
                                for (k, v) in map {
                                    if let Some(vec_data) = try_extract_vector(v) {
                                        registry.on_vector_written(&label, node_id, k, &vec_data);
                                    }
                                }
                            }
                        }

                        // Notify text index registry for any text properties in the merged map.
                        if let Some(registry) = ctx.text_index_registry {
                            let label = record.primary_label().to_string();
                            if let Value::Map(ref map) = map_val {
                                for (k, v) in map {
                                    if let Some(text) = v.as_str() {
                                        registry.on_text_written(&label, node_id, k, text);
                                    }
                                }
                            }
                        }
                    }
                }
                crate::cypher::ast::SetItem::AddLabel { variable, label } => {
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    if let Some(mut record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                        record.add_label(label.clone());
                        ctx.write_stats.labels_added += 1;

                        ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;

                        out_row.insert(
                            format!("{variable}.__label__"),
                            Value::String(record.primary_label().to_string()),
                        );
                    }
                }
            }
        }

        // After all SET items applied for this row, fire BEFORE COMMIT
        // UPDATE triggers per touched node — once per node, not once per
        // item.
        //
        // IMPORTANT: probe the trigger index BEFORE materialising the
        // post-state via `mvcc_get`. The Node-partition mvcc_get has a
        // side-effect (it materialises pending `merge_node_deltas` into
        // the write buffer) which is harmless in MVCC mode but in legacy
        // (no-oracle) mode loses the deltas — they would otherwise be
        // re-applied by `mvcc_flush` as MERGE operands. Reading post-state
        // only when at least one trigger matches keeps the happy path
        // (no triggers registered) free of that side-effect.
        for (node_id, (pre_labels, before_props)) in &update_snapshots {
            let mut all_matched: Vec<coordinode_core::schema::triggers::TriggerSchema> = Vec::new();
            for label in pre_labels {
                let target_segment =
                    coordinode_core::schema::triggers::TriggerTargetSchema::label(label.clone())
                        .index_key_segment();
                let matched = ctx.lookup_matching_triggers(&target_segment, "u")?;
                all_matched.extend(matched);
            }
            if all_matched.is_empty() {
                continue;
            }

            let Some(post_record) = ctx.mvcc_get_node(ctx.shard_id, *node_id)? else {
                continue;
            };
            let (_post_labels, after_props) = snapshot_node_record(&post_record, ctx);
            let trigger_params =
                trigger_params_for_node_update(*node_id, before_props, &after_props);
            fire_before_commit_triggers(&all_matched, &trigger_params, ctx)?;
        }

        // Edge UPDATE triggers: same "probe index first, materialize after"
        // pattern. The probe-before-read invariant doesn't apply to the
        // EdgeProp partition (no merge_node_deltas equivalent), but we
        // keep the index lookup first to avoid the materialise cost on
        // the happy path (no triggers).
        // Collect snapshot views first to avoid borrowing ctx mutably
        // while iterating an immutable borrow of edge_update_snapshots.
        let snapshots_to_probe: Vec<_> = edge_update_snapshots
            .values()
            .map(|s| {
                (
                    s.edge_type.clone(),
                    s.src,
                    s.tgt,
                    s.valid_from_ms,
                    s.before.clone(),
                )
            })
            .collect();
        for (edge_type, src, tgt, valid_from_ms, before) in snapshots_to_probe {
            let target_segment =
                coordinode_core::schema::triggers::TriggerTargetSchema::edge_type(&edge_type)
                    .index_key_segment();
            let matched = ctx.lookup_matching_triggers(&target_segment, "u")?;
            if matched.is_empty() {
                continue;
            }
            let after = match ctx.mvcc_get_edge_props_either(&edge_type, src, tgt, valid_from_ms)? {
                Some(prop_map) => decode_edgeprop_map_into_named(&prop_map, ctx),
                None => std::collections::BTreeMap::new(),
            };
            let trigger_params =
                trigger_params_for_edge_update(&edge_type, src, tgt, &before, &after);
            fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
        }

        results.push(out_row);
    }

    Ok(results)
}

/// REMOVE: remove properties/labels from existing nodes.
fn execute_remove(
    input_rows: &[Row],
    items: &[crate::cypher::ast::RemoveItem],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // R172c Phase 3: REMOVE on a temporal node is a *new version* — same
    // close-current + open-new dance as Phase 2 SET. Removing a property
    // / label is a state change that must be visible as a new version in
    // the bitemporal record, not a silent in-place mutation of the
    // matched historical row.
    //
    // Rules:
    //   * REMOVE `n.valid_from` / `n.valid_to` → REJECT. These are the
    //     bitemporal axis fields; close-current is done via
    //     `SET n.valid_to = …`, not REMOVE.
    //   * REMOVE `n.__ingestion_ts__` / `n.__deleted__` → REJECT.
    //     Engine-managed system fields.
    //   * REMOVE `n.<other_prop>` → close current + open new with the
    //     property absent in the new record.
    //   * REMOVE `n:Label` → close current + open new with the label
    //     dropped.
    //   * REMOVE `n.<path.to.nested>` (PropertyPath) → REJECT (Phase 3b,
    //     same reason as SET nested path — merge_node_deltas is keyed on
    //     the non-temporal key and needs a temporal-aware rewrite).
    let mut temporal_remove_vars: std::collections::HashSet<(usize, String)> =
        std::collections::HashSet::new();
    for (row_idx, row) in input_rows.iter().enumerate() {
        for item in items {
            let var = match item {
                crate::cypher::ast::RemoveItem::Property { variable, .. }
                | crate::cypher::ast::RemoveItem::PropertyPath { variable, .. }
                | crate::cypher::ast::RemoveItem::Label { variable, .. } => variable,
            };
            if !matches!(row.get(var), Some(Value::Int(_))) {
                continue;
            }
            let Some(Value::String(primary)) = row.get(&format!("{var}.__label__")) else {
                continue;
            };
            let is_temporal = ctx
                .load_current_label_schema(primary)
                .ok()
                .flatten()
                .is_some_and(|s| s.temporal);
            if !is_temporal {
                continue;
            }
            match item {
                crate::cypher::ast::RemoveItem::Property { property, .. } => {
                    if property == "valid_from" || property == "valid_to" {
                        return Err(ExecutionError::Unsupported(format!(
                            "REMOVE {var}.{property} is rejected on temporal label \
                             '{primary}': bitemporal axis fields are engine-managed. \
                             Use `SET {var}.valid_to = …` to close the current version."
                        )));
                    }
                    if property == "__ingestion_ts__" || property == "__deleted__" {
                        return Err(ExecutionError::Unsupported(format!(
                            "REMOVE {var}.{property} is rejected on temporal label \
                             '{primary}': '{property}' is an engine-managed system field."
                        )));
                    }
                    temporal_remove_vars.insert((row_idx, var.clone()));
                }
                crate::cypher::ast::RemoveItem::PropertyPath { .. } => {
                    // R172c Phase 3b: classify; the delta is built and
                    // applied to `new_record` in the close+open block.
                    temporal_remove_vars.insert((row_idx, var.clone()));
                }
                crate::cypher::ast::RemoveItem::Label { .. } => {
                    temporal_remove_vars.insert((row_idx, var.clone()));
                }
            }
        }
    }

    let mut results = Vec::new();

    for (row_idx, row) in input_rows.iter().enumerate() {
        let mut out_row = row.clone();

        // R172c Phase 3: close-current + open-new processing for temporal
        // nodes whose REMOVE pre-scan classified as new-version. Mirrors
        // the SET Phase 2 block — same shape, same invariants. Items
        // applied here are recorded in `processed_temporal_items` so the
        // standard REMOVE loop below skips them on this row.
        let mut processed_temporal_items: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let temporal_remove_vars_for_row: Vec<String> = temporal_remove_vars
            .iter()
            .filter(|(idx, _)| *idx == row_idx)
            .map(|(_, v)| v.clone())
            .collect();
        if !temporal_remove_vars_for_row.is_empty() {
            let now_us = current_hlc_us() as i64;
            for var in &temporal_remove_vars_for_row {
                let node_id = match out_row.get(var) {
                    Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                    _ => continue,
                };
                let current_valid_from = match out_row.get(&format!("{var}.valid_from")) {
                    Some(Value::Int(ms)) => *ms,
                    Some(Value::Timestamp(ms)) => *ms,
                    _ => {
                        return Err(ExecutionError::Unsupported(format!(
                            "temporal REMOVE on `{var}`: matched row is missing \
                             `{var}.valid_from` (planner must surface valid_from on \
                             every temporal node materialised for mutation)"
                        )));
                    }
                };
                let new_valid_from = if now_us > current_valid_from {
                    now_us
                } else {
                    current_valid_from + 1
                };
                let mut closing_record = ctx
                    .mvcc_get_node_temporal(ctx.shard_id, node_id, current_valid_from)?
                    .ok_or_else(|| {
                        ExecutionError::Unsupported(format!(
                            "temporal REMOVE on `{var}`: matched version record not \
                             found at (node_id={node_id}, valid_from={current_valid_from})"
                        ))
                    })?;
                let mut new_record = closing_record.clone();

                for (item_idx, item) in items.iter().enumerate() {
                    let item_var = match item {
                        crate::cypher::ast::RemoveItem::Property { variable, .. }
                        | crate::cypher::ast::RemoveItem::PropertyPath { variable, .. }
                        | crate::cypher::ast::RemoveItem::Label { variable, .. } => {
                            variable.as_str()
                        }
                    };
                    if item_var != var.as_str() {
                        continue;
                    }
                    match item {
                        crate::cypher::ast::RemoveItem::Property { property, .. } => {
                            if let Some(field_id) = ctx.interner.lookup(property) {
                                new_record.remove(field_id);
                            }
                            ctx.write_stats.properties_removed += 1;
                        }
                        crate::cypher::ast::RemoveItem::Label { label, .. } => {
                            new_record.remove_label(label);
                            ctx.write_stats.labels_removed += 1;
                        }
                        crate::cypher::ast::RemoveItem::PropertyPath { path, .. } => {
                            // R172c Phase 3b: nested REMOVE on temporal —
                            // build DeletePath DocDelta, apply in-memory.
                            if path.is_empty() {
                                return Err(ExecutionError::Unsupported(format!(
                                    "REMOVE on temporal node `{var}`: empty property path"
                                )));
                            }
                            let field_id = ctx.interner.intern(&path[0]);
                            let sub_path = path[1..].to_vec();
                            let delta = if sub_path.is_empty() {
                                // Top-level prop removal — RemoveProperty
                                // (no sub-path traversal).
                                coordinode_core::graph::doc_delta::DocDelta::RemoveProperty {
                                    target:
                                        coordinode_core::graph::doc_delta::PathTarget::PropField(
                                            field_id,
                                        ),
                                    key: None,
                                }
                            } else {
                                coordinode_core::graph::doc_delta::DocDelta::DeletePath {
                                    target:
                                        coordinode_core::graph::doc_delta::PathTarget::PropField(
                                            field_id,
                                        ),
                                    path: sub_path,
                                }
                            };
                            coordinode_storage::engine::merge::apply_doc_deltas_to_record(
                                &mut new_record,
                                &[delta],
                            );
                            ctx.write_stats.properties_removed += 1;
                        }
                    }
                    processed_temporal_items.insert(item_idx);
                }

                let vf_fid = ctx.interner.intern("valid_from");
                let vt_fid = ctx.interner.intern("valid_to");
                let its_fid = ctx.interner.intern("__ingestion_ts__");
                new_record.set(vf_fid, Value::Int(new_valid_from));
                new_record.props.remove(&vt_fid);
                new_record.set(its_fid, Value::Int(now_us));
                closing_record.set(vt_fid, Value::Int(new_valid_from));

                ctx.mvcc_put_node_temporal(
                    ctx.shard_id,
                    node_id,
                    current_valid_from,
                    &closing_record,
                )?;

                ctx.mvcc_put_node_temporal(ctx.shard_id, node_id, new_valid_from, &new_record)?;

                // Surface the new version's valid_from on out_row so any
                // downstream RETURN sees the latest version, symmetric with
                // SET Phase 2 behaviour.
                out_row.insert(format!("{var}.valid_from"), Value::Int(new_valid_from));
                if new_record.primary_label() != closing_record.primary_label() {
                    out_row.insert(
                        format!("{var}.__label__"),
                        Value::String(new_record.primary_label().to_string()),
                    );
                }
            }
        }

        // Snapshot pre-mutation node state for UPDATE trigger firing —
        // REMOVE is symmetric with SET: it mutates the node, so a registered
        // UPDATE trigger must observe the change. Use schema_peek_node to
        // avoid consuming pending merge_node_deltas (same rule as SET).
        let mut update_snapshots: std::collections::HashMap<
            NodeId,
            (Vec<String>, std::collections::BTreeMap<String, Value>),
        > = std::collections::HashMap::new();
        for item in items {
            let variable = match item {
                crate::cypher::ast::RemoveItem::Property { variable, .. }
                | crate::cypher::ast::RemoveItem::PropertyPath { variable, .. }
                | crate::cypher::ast::RemoveItem::Label { variable, .. } => variable,
            };
            let node_id = match out_row.get(variable) {
                Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                _ => continue,
            };
            if update_snapshots.contains_key(&node_id) {
                continue;
            }
            if let Some(record) = ctx.schema_peek_node_typed(ctx.shard_id, node_id)? {
                update_snapshots.insert(node_id, snapshot_node_record(&record, ctx));
            }
        }

        for (item_idx, item) in items.iter().enumerate() {
            if processed_temporal_items.contains(&item_idx) {
                continue;
            }
            match item {
                crate::cypher::ast::RemoveItem::Property { variable, property } => {
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    if let Some(mut record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                        if let Some(field_id) = ctx.interner.lookup(property) {
                            let old_value: Option<Value> = record.props.get(&field_id).cloned();

                            // Remove B-tree index entry for the old value so
                            // unique constraints don't block future nodes with
                            // the same property value.
                            if let Some(btree_reg) = ctx.btree_index_registry {
                                if let Some(ref old_val) = old_value {
                                    let label = record.primary_label().to_string();
                                    btree_reg
                                        .on_node_deleted(
                                            ctx.engine,
                                            node_id,
                                            &label,
                                            &[(property.to_string(), old_val.clone())],
                                        )
                                        .map_err(ExecutionError::Storage)?;
                                }
                            }

                            // Notify vector index if removing a vector property.
                            if let Some(registry) = ctx.vector_index_registry {
                                if old_value
                                    .as_ref()
                                    .is_some_and(|v| try_extract_vector(v).is_some())
                                {
                                    let label = record.primary_label().to_string();
                                    registry.on_vector_deleted(&label, node_id, property);
                                }
                            }
                            // Notify text index if removing a text property.
                            if let Some(registry) = ctx.text_index_registry {
                                let label = record.primary_label().to_string();
                                registry.on_text_deleted(&label, node_id, property);
                            }
                            record.remove(field_id);
                            ctx.write_stats.properties_removed += 1;
                        }

                        ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;
                    }

                    out_row.remove(&format!("{variable}.{property}"));
                }
                crate::cypher::ast::RemoveItem::PropertyPath { variable, path } => {
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    // O(1) delete via merge operand — no read required.
                    let field_id = ctx.interner.intern(&path[0]);
                    let sub_path = &path[1..];

                    let delta = coordinode_core::graph::doc_delta::DocDelta::DeletePath {
                        target: coordinode_core::graph::doc_delta::PathTarget::PropField(field_id),
                        path: sub_path.to_vec(),
                    };
                    let operand = delta.encode().map_err(|e| {
                        ExecutionError::Serialization(format!("DocDelta encode: {e}"))
                    })?;
                    ctx.mvcc_merge_node_delta(ctx.shard_id, node_id, operand);
                    ctx.write_stats.properties_removed += 1;

                    let path_str = path.join(".");
                    out_row.remove(&format!("{variable}.{path_str}"));
                }
                crate::cypher::ast::RemoveItem::Label { variable, label } => {
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    if let Some(mut record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                        record.remove_label(label);
                        ctx.write_stats.labels_removed += 1;

                        ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;

                        out_row.insert(
                            format!("{variable}.__label__"),
                            Value::String(record.primary_label().to_string()),
                        );
                    }
                }
            }
        }

        // After all REMOVE items applied, fire UPDATE triggers once per
        // touched node. The trigger registration uses the PRE-mutation
        // labels: REMOVE n:Label can shrink the label set, but a trigger
        // registered on the removed label should still observe the change
        // (it's the last firing window before the label is gone).
        // Same probe-before-materialise rule as `execute_update`.
        for (node_id, (pre_labels, before_props)) in &update_snapshots {
            let mut all_matched: Vec<coordinode_core::schema::triggers::TriggerSchema> = Vec::new();
            for label in pre_labels {
                let target_segment =
                    coordinode_core::schema::triggers::TriggerTargetSchema::label(label.clone())
                        .index_key_segment();
                let matched = ctx.lookup_matching_triggers(&target_segment, "u")?;
                all_matched.extend(matched);
            }
            if all_matched.is_empty() {
                continue;
            }
            let Some(post_record) = ctx.mvcc_get_node(ctx.shard_id, *node_id)? else {
                continue;
            };
            let (_post_labels, after_props) = snapshot_node_record(&post_record, ctx);
            let trigger_params =
                trigger_params_for_node_update(*node_id, before_props, &after_props);
            fire_before_commit_triggers(&all_matched, &trigger_params, ctx)?;
        }

        results.push(out_row);
    }

    Ok(results)
}

/// DELETE: remove nodes (and optionally connected edges with DETACH).
///
/// Without DETACH, fails with an error if the node has connected edges.
/// Per OpenCypher spec: `DELETE n` requires n to be disconnected.
fn execute_delete(
    input_rows: &[Row],
    variables: &[String],
    detach: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Temporal edge matching produces one row per stored version of the
    // same `(src, tgt, edge_type)` pair. `DELETE r` removes every version
    // of the pair in one call, so the second pass would be a no-op on
    // storage — but would still re-fire BEFORE COMMIT DELETE triggers,
    // because the pre-snapshot in `delete_single_edge` reads from the
    // engine snapshot (tombstones in the write buffer don't suppress the
    // snapshot rows). Dedupe at this layer so each logical edge is
    // processed exactly once per DELETE statement.
    let mut deleted_edges: std::collections::HashSet<(String, u64, u64)> =
        std::collections::HashSet::new();

    // R172c Phase 3: DELETE on temporal nodes is a *positive bitemporal
    // fact* (XTDB-donor pattern, ADR-027). Instead of hard-deleting the
    // stored per-version records, we append a tombstone row at
    // `valid_from = NOW, valid_to = NULL` carrying `__deleted__: true`
    // — an explicit assertion that the node ceased to exist at NOW.
    // History before NOW is preserved and remains queryable. The
    // current open version (if any — matched row with valid_to IS NULL)
    // also has its valid_to closed at NOW so a bitemporal scan sees:
    //
    //     ─── valid_from = 100, valid_to = NOW ──── (was alive)
    //     ─── valid_from = NOW, __deleted__ = true ── (tombstone)
    //
    // Hard erasure across ALL versions (GDPR right-to-erase) requires
    // an explicit `WITH GDPR_ERASURE` modifier on DELETE — parser
    // support lands in R172c Phase 3b. Until then a tombstone is the
    // only DELETE mode on temporal labels.
    //
    // Pre-scan: collect temporal-label node deletions for this DELETE
    // clause; edge variables (Value::String) take the normal path.
    let mut temporal_node_deletes: Vec<(usize, String, NodeId, i64, String)> = Vec::new();
    for (row_idx, row) in input_rows.iter().enumerate() {
        for var in variables {
            if !matches!(row.get(var), Some(Value::Int(_))) {
                continue;
            }
            let Some(Value::String(primary)) = row.get(&format!("{var}.__label__")) else {
                continue;
            };
            let is_temporal = ctx
                .load_current_label_schema(primary)
                .ok()
                .flatten()
                .is_some_and(|s| s.temporal);
            if !is_temporal {
                continue;
            }
            let node_id = match row.get(var) {
                Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                _ => continue,
            };
            let current_valid_from = match row.get(&format!("{var}.valid_from")) {
                Some(Value::Int(ms)) => *ms,
                Some(Value::Timestamp(ms)) => *ms,
                _ => {
                    return Err(ExecutionError::Unsupported(format!(
                        "DELETE {var} on temporal label '{primary}': matched row is \
                         missing `{var}.valid_from` (planner must surface valid_from)"
                    )));
                }
            };
            temporal_node_deletes.push((
                row_idx,
                var.clone(),
                node_id,
                current_valid_from,
                primary.clone(),
            ));
        }
    }

    let mut tombstoned_temporal_rows: std::collections::HashSet<(usize, String)> =
        std::collections::HashSet::new();
    if !temporal_node_deletes.is_empty() {
        let now_us = current_hlc_us() as i64;
        for (row_idx, var, node_id, current_valid_from, _primary) in &temporal_node_deletes {
            let new_valid_from = if now_us > *current_valid_from {
                now_us
            } else {
                current_valid_from + 1
            };

            // Step 1: close the matched open version (if it WAS open —
            // valid_to absent or NULL). If the matched version was
            // already closed (a historical version), skip closing — the
            // tombstone alone marks the deletion event.
            let Some(mut closing_record) =
                ctx.mvcc_get_node_temporal(ctx.shard_id, *node_id, *current_valid_from)?
            else {
                // Already gone — skip (idempotent).
                continue;
            };
            let vt_fid = ctx.interner.intern("valid_to");
            let was_open = !closing_record.props.contains_key(&vt_fid);
            if was_open {
                closing_record.set(vt_fid, Value::Int(new_valid_from));
                ctx.mvcc_put_node_temporal(
                    ctx.shard_id,
                    *node_id,
                    *current_valid_from,
                    &closing_record,
                )?;
            }

            // Step 2: write the tombstone row at NOW. Carries the same
            // labels as the closing record (so downstream MATCH still
            // sees the row under the right label), no user properties,
            // `__deleted__: true`, refreshed `__ingestion_ts__`. The
            // tombstone has `valid_from = NOW`, `valid_to = NULL` —
            // the deletion is "current" from NOW onward.
            let mut tombstone = NodeRecord::with_labels(closing_record.labels.clone());
            let deleted_fid = ctx.interner.intern("__deleted__");
            tombstone.set(deleted_fid, Value::Bool(true));
            let vf_fid = ctx.interner.intern("valid_from");
            tombstone.set(vf_fid, Value::Int(new_valid_from));
            let its_fid = ctx.interner.intern("__ingestion_ts__");
            tombstone.set(its_fid, Value::Int(now_us));
            ctx.mvcc_put_node_temporal(ctx.shard_id, *node_id, new_valid_from, &tombstone)?;
            ctx.write_stats.nodes_deleted += 1;

            tombstoned_temporal_rows.insert((*row_idx, var.clone()));
        }
    }

    for (row_idx, row) in input_rows.iter().enumerate() {
        for var in variables {
            // R172c Phase 3: if this (row, var) was tombstoned above as a
            // temporal-node positive bitemporal fact, skip the legacy
            // hard-delete path entirely — the tombstone IS the delete.
            // Edges connected to a temporal node are intentionally left
            // intact: at past valid times the node still existed, so its
            // edges are still part of history. Edge cascade for temporal
            // nodes is a separate concern tracked under R172c Phase 4.
            if tombstoned_temporal_rows.contains(&(row_idx, var.clone())) {
                continue;
            }
            // Edge variable bindings carry the edge type as a String value,
            // not a node id. DELETE on an edge variable hard-deletes that
            // logical edge: non-temporal types remove the one edgeprop entry
            // and clear adj-posting; temporal types remove every version
            // under the (src, tgt) prefix and clear adj-posting once the
            // version count drops to zero.
            if let Some(Value::String(edge_type)) = row.get(var).cloned() {
                let src_raw = match row.get(&format!("{var}.__src__")) {
                    Some(Value::Int(n)) => *n as u64,
                    _ => continue,
                };
                let tgt_raw = match row.get(&format!("{var}.__tgt__")) {
                    Some(Value::Int(n)) => *n as u64,
                    _ => continue,
                };
                if !deleted_edges.insert((edge_type.clone(), src_raw, tgt_raw)) {
                    continue;
                }
                delete_single_edge(
                    NodeId::from_raw(src_raw),
                    NodeId::from_raw(tgt_raw),
                    &edge_type,
                    ctx,
                )?;
                continue;
            }

            let node_id = match row.get(var) {
                Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                _ => continue,
            };

            // Check if node has edges (for both DELETE and DETACH DELETE).
            //
            // Targeted lookup: for each registered edge type, check the two
            // canonical adj keys for this node:
            //   adj:<TYPE>:out:<node_id BE>  — edges where node is the source
            //   adj:<TYPE>:in:<node_id BE>   — edges where node is the target
            //
            // This is O(registered_edge_types × 2) instead of O(all_edges_in_db).
            // adj_get applies pending merge_adj_adds/removes (RYOW) automatically.
            let edge_types = ctx.list_edge_types()?;
            let mut edge_count: u64 = 0;
            let mut edges_to_delete: Vec<AdjKeyParts> = Vec::new();

            for edge_type in &edge_types {
                for direction in [AdjDirection::Out, AdjDirection::In] {
                    let present = match direction {
                        AdjDirection::Out => ctx.adj_get_fwd(edge_type, node_id)?,
                        AdjDirection::In => ctx.adj_get_rev(edge_type, node_id)?,
                    }
                    .is_some();
                    if present {
                        edge_count += 1;
                        if detach {
                            edges_to_delete.push(AdjKeyParts {
                                edge_type: edge_type.clone(),
                                direction,
                                node_id,
                            });
                        }
                    }
                }
            }

            if !detach && edge_count > 0 {
                return Err(ExecutionError::Unsupported(format!(
                    "cannot delete node {} because it still has {edge_count} \
                     connected edge(s). Use DETACH DELETE to remove edges first",
                    node_id
                )));
            }

            // Delete edges if DETACH:
            // 1. For each adj: key, read the posting list and issue merge_remove
            //    on the counterpart key (forward→reverse, reverse→forward)
            //    so the deleted node's UID is removed from OTHER nodes' posting lists.
            // 2. Clean up edge properties (edgeprop:) for each edge.
            // 3. Delete the adj: key itself.
            //
            // Cascaded edges fire BEFORE COMMIT DELETE triggers registered on
            // their edge types — `$before` carries the deleted edge's props
            // (one firing per temporal version). The trigger probe runs once
            // per `(edge_type)` and is reused across every (src, tgt) pair on
            // that key; we snapshot per-pair *before* deletion so the body
            // reading the same key returns empty (RYOW).
            for parts in &edges_to_delete {
                {
                    // Look up BEFORE COMMIT DELETE triggers for this edge type
                    // once per adj key. Re-used across every (node, peer) pair
                    // walked from the posting list.
                    let edge_target_segment =
                        coordinode_core::schema::triggers::TriggerTargetSchema::edge_type(
                            &parts.edge_type,
                        )
                        .index_key_segment();
                    let matched_edge_delete =
                        ctx.lookup_matching_triggers(&edge_target_segment, "d")?;
                    let is_temporal = lookup_edge_type_temporal(&parts.edge_type, ctx)?;

                    // Read posting list to find connected nodes
                    let plist = match parts.direction {
                        AdjDirection::Out => ctx.adj_get_fwd(&parts.edge_type, parts.node_id)?,
                        AdjDirection::In => ctx.adj_get_rev(&parts.edge_type, parts.node_id)?,
                    };
                    if let Some(plist) = plist {
                        for peer_uid in plist.iter() {
                            let peer_id = NodeId::from_raw(peer_uid);
                            // adj:TYPE:out:NODE → counterpart adj:TYPE:in:PEER,
                            // and adj:TYPE:in:NODE → counterpart adj:TYPE:out:PEER.
                            match parts.direction {
                                AdjDirection::Out => ctx.adj_merge_remove_rev(
                                    &parts.edge_type,
                                    peer_id,
                                    node_id.as_raw(),
                                ),
                                AdjDirection::In => ctx.adj_merge_remove_fwd(
                                    &parts.edge_type,
                                    peer_id,
                                    node_id.as_raw(),
                                ),
                            };

                            // Clean up edge properties for this edge. Temporal
                            // edge types keep one edgeprop entry per version
                            // (keyed on valid_from), so a single-key delete
                            // would leak N-1 orphan entries. Prefix-scan the
                            // pair and tombstone every version.
                            let (ep_src, ep_tgt) = match parts.direction {
                                AdjDirection::Out => (node_id, peer_id),
                                AdjDirection::In => (peer_id, node_id),
                            };

                            // Pre-snapshot edge props for trigger firing (if
                            // any trigger registered). Captured *before*
                            // mvcc_delete so the body's RYOW reads see the
                            // edge as deleted.
                            let mut edge_delete_snapshots: Vec<
                                std::collections::BTreeMap<String, Value>,
                            > = Vec::new();
                            if !matched_edge_delete.is_empty() {
                                if is_temporal {
                                    for (_vf, bytes) in ctx.mvcc_scan_edge_prop_version_bytes(
                                        &parts.edge_type,
                                        ep_src,
                                        ep_tgt,
                                        None,
                                    )? {
                                        edge_delete_snapshots
                                            .push(decode_edgeprop_into_map(&bytes, ctx));
                                    }
                                } else {
                                    if let Some(bytes) = ctx.mvcc_get_edge_prop_bytes(
                                        &parts.edge_type,
                                        ep_src,
                                        ep_tgt,
                                        None,
                                    )? {
                                        edge_delete_snapshots
                                            .push(decode_edgeprop_into_map(&bytes, ctx));
                                    } else {
                                        edge_delete_snapshots
                                            .push(std::collections::BTreeMap::new());
                                    }
                                }
                            }

                            if is_temporal {
                                ctx.mvcc_delete_all_edge_prop_versions(
                                    &parts.edge_type,
                                    ep_src,
                                    ep_tgt,
                                )?;
                            } else {
                                ctx.mvcc_delete_edge_props(&parts.edge_type, ep_src, ep_tgt)?;
                            }

                            // Fire edge-DELETE triggers per snapshotted version.
                            if !matched_edge_delete.is_empty() {
                                for before in &edge_delete_snapshots {
                                    let trigger_params = trigger_params_for_edge_delete(
                                        &parts.edge_type,
                                        ep_src,
                                        ep_tgt,
                                        before,
                                    );
                                    fire_before_commit_triggers(
                                        &matched_edge_delete,
                                        &trigger_params,
                                        ctx,
                                    )?;
                                }
                            }
                        }
                    }
                }

                ctx.mvcc_purge_adj(&parts.edge_type, parts.node_id, parts.direction)?;
                ctx.write_stats.edges_deleted += 1;
            }

            // Delete the node record.
            // First snapshot the pre-mutation state so we can both clean up
            // index entries AND build the `$before` map for any BEFORE
            // COMMIT DELETE trigger that fires on the node's labels.
            let pre_snapshot: Option<(Vec<String>, std::collections::BTreeMap<String, Value>)> =
                ctx.mvcc_get_node(ctx.shard_id, node_id)?
                    .map(|rec| snapshot_node_record(&rec, ctx));

            let needs_index_cleanup = ctx.btree_index_registry.is_some()
                || ctx.vector_index_registry.is_some()
                || ctx.text_index_registry.is_some();
            if needs_index_cleanup {
                if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                    {
                        let label = record.primary_label().to_string();

                        // B-tree index cleanup: remove all indexed property
                        // entries so unique constraints don't block re-creation
                        // of a node with the same property values.
                        if let Some(btree_reg) = ctx.btree_index_registry {
                            let props: Vec<(String, Value)> = record
                                .props
                                .iter()
                                .filter_map(|(&field_id, value)| {
                                    ctx.interner
                                        .resolve(field_id)
                                        .map(|name| (name.to_string(), value.clone()))
                                })
                                .collect();
                            btree_reg
                                .on_node_deleted(ctx.engine, node_id, &label, &props)
                                .map_err(ExecutionError::Storage)?;
                        }

                        for (&field_id, value) in &record.props {
                            if let Some(prop_name) = ctx.interner.resolve(field_id) {
                                if let Some(registry) = ctx.vector_index_registry {
                                    if try_extract_vector(value).is_some() {
                                        registry.on_vector_deleted(&label, node_id, prop_name);
                                    }
                                }
                                if let Some(registry) = ctx.text_index_registry {
                                    if value.as_str().is_some() {
                                        registry.on_text_deleted(&label, node_id, prop_name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            ctx.mvcc_delete_node(ctx.shard_id, node_id)?;
            ctx.write_stats.nodes_deleted += 1;

            // Fire BEFORE COMMIT DELETE triggers on each of the deleted
            // node's labels. The probe runs AFTER mvcc_delete so the
            // trigger body's MATCH against the deleted node returns
            // empty via RYOW — correct semantics: the node is gone for
            // any read inside the trigger. `$before` carries the
            // snapshot taken above; `$after` is NULL.
            if let Some((labels, before_props)) = pre_snapshot {
                let trigger_params = trigger_params_for_node_delete(node_id, &before_props);
                for label in &labels {
                    let target_segment =
                        coordinode_core::schema::triggers::TriggerTargetSchema::label(
                            label.clone(),
                        )
                        .index_key_segment();
                    let matched = ctx.lookup_matching_triggers(&target_segment, "d")?;
                    if !matched.is_empty() {
                        fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
                    }
                }
            }
        }
    }

    // DELETE returns the input rows (so downstream RETURN can reference them)
    Ok(input_rows.to_vec())
}

// =====================================================================
// MERGE NODES (R180)
// =====================================================================

/// Execute a `MERGE NODES (a, b) INTO target` clause for each input row.
///
/// Collapses the non-surviving source into the surviving target within a
/// single MVCC transaction. See arch/compatibility/native-procedures.md for
/// the full semantic contract.
#[allow(clippy::too_many_arguments)]
fn execute_merge_nodes(
    input_rows: &[Row],
    source_a: &str,
    source_b: &str,
    target: &str,
    conflict: &crate::cypher::ast::MergeNodesConflictStrategy,
    transfer_edges: Option<&crate::cypher::ast::TransferEdgesEndpoints>,
    duplicate: &crate::cypher::ast::MergeNodesDuplicateStrategy,
    transfer_edge_properties: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Semantic validation deferred to here so we have a concrete error path
    // when the planner skipped it (e.g. direct LogicalOp construction in tests).
    if target != source_a && target != source_b {
        return Err(ExecutionError::Unsupported(format!(
            "MERGE NODES target `{target}` must be one of `{source_a}`, `{source_b}`"
        )));
    }
    if let Some(t) = transfer_edges {
        if t.dst != target {
            return Err(ExecutionError::Unsupported(format!(
                "TRANSFER EDGES TO `{}` must match INTO target `{target}`",
                t.dst
            )));
        }
        let expected_src = if target == source_a {
            source_b
        } else {
            source_a
        };
        if t.src != expected_src {
            return Err(ExecutionError::Unsupported(format!(
                "TRANSFER EDGES FROM `{}` must be the non-surviving source `{expected_src}`",
                t.src
            )));
        }
    }

    // R172b safe-reject for temporal labels: MERGE NODES reads source
    // nodes via 16-byte `encode_node_key`, mutates the target record, and
    // deletes the non-survivor via `detach_delete_node` — none of these
    // paths are temporal-aware. Merging two temporal nodes (whether the
    // intent is "fold version histories" or "treat all versions as one
    // logical node") is genuinely a per-version write-executor concern
    // and lands in R172c.
    for row in input_rows {
        for var in [source_a, source_b] {
            if !matches!(row.get(var), Some(Value::Int(_))) {
                continue;
            }
            let Some(Value::String(primary)) = row.get(&format!("{var}.__label__")) else {
                continue;
            };
            if let Ok(Some(s)) = ctx.load_current_label_schema(primary) {
                if s.temporal {
                    return Err(ExecutionError::Unsupported(format!(
                        "MERGE NODES on temporal label '{primary}' is not yet supported \
                         (lands in R172c — per-version write executor)."
                    )));
                }
            }
        }
    }

    let mut out = Vec::with_capacity(input_rows.len());

    for row in input_rows {
        let a_id = match row.get(source_a) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => continue,
        };
        let b_id = match row.get(source_b) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => continue,
        };

        let (target_id, source_id) = if target == source_a {
            (a_id, b_id)
        } else {
            (b_id, a_id)
        };

        // Idempotent no-op: same node on both sides (already merged previously).
        if target_id == source_id {
            out.push(row.clone());
            continue;
        }

        // Source missing → no-op (idempotent: already merged in a prior attempt).
        let Some(source_rec) = ctx.mvcc_get_node(ctx.shard_id, source_id)? else {
            out.push(row.clone());
            continue;
        };
        // Target missing → hard error (the surviving node must exist).
        let mut target_rec = ctx.mvcc_get_node(ctx.shard_id, target_id)?.ok_or_else(|| {
            ExecutionError::Unsupported(format!("MERGE NODES target node {target_id} not found"))
        })?;

        // Capture target's properties BEFORE merge so we can issue per-property
        // index notifications afterwards (delete-old / insert-new).
        let target_props_before = target_rec.props.clone();
        let target_extra_before: HashMap<String, Value> =
            target_rec.extra.clone().unwrap_or_default();
        let target_label = target_rec.primary_label().to_string();

        merge_node_properties(&mut target_rec, &source_rec, conflict, row, ctx)?;

        // STRICT / VALIDATED schema enforcement: a merged record may carry
        // properties that the target's label doesn't accept (source-only
        // declared props, type mismatches). Reject before mutating storage —
        // otherwise the merge would commit invalid records that violate the
        // schema contract.
        if let Some(schema) = ctx.load_current_label_schema(&target_label)? {
            for (&field_id, val) in &target_rec.props {
                let Some(prop_name) = ctx.interner.resolve(field_id) else {
                    continue;
                };
                let prop_name = prop_name.to_string();
                // Skip properties that already existed on target before the
                // merge — they were valid then, still valid now.
                if target_props_before.get(&field_id) == Some(val) {
                    continue;
                }
                match schema.mode {
                    SchemaMode::Strict => match schema.get_property(&prop_name) {
                        None => {
                            return Err(ExecutionError::SchemaViolation(format!(
                                "MERGE NODES would set unknown property '{prop_name}' \
                                 on strict label '{target_label}'"
                            )));
                        }
                        Some(def) if def.is_computed() => {
                            return Err(ExecutionError::SchemaViolation(format!(
                                "MERGE NODES cannot SET computed property '{prop_name}' \
                                 on label '{target_label}'"
                            )));
                        }
                        Some(def) => {
                            validate_one(&prop_name, val, def)
                                .map_err(|e| ExecutionError::SchemaViolation(e.to_string()))?;
                        }
                    },
                    SchemaMode::Validated => {
                        if let Some(def) = schema.get_property(&prop_name) {
                            if def.is_computed() {
                                return Err(ExecutionError::SchemaViolation(format!(
                                    "MERGE NODES cannot SET computed property '{prop_name}' \
                                     on label '{target_label}'"
                                )));
                            }
                            validate_one(&prop_name, val, def)
                                .map_err(|e| ExecutionError::SchemaViolation(e.to_string()))?;
                        }
                    }
                    SchemaMode::Flexible => {}
                }
            }
            // The overflow `extra` map carries undeclared properties; under
            // STRICT this slot must be empty. Catch source-only fields that
            // were merged into the target's extra and bail rather than
            // committing an out-of-schema record.
            if matches!(schema.mode, SchemaMode::Strict) {
                if let Some(extra) = &target_rec.extra {
                    for (name, val) in extra {
                        if target_extra_before.get(name) == Some(val) {
                            continue;
                        }
                        return Err(ExecutionError::SchemaViolation(format!(
                            "MERGE NODES would set unknown property '{name}' \
                             on strict label '{target_label}'"
                        )));
                    }
                }
            }
        }

        // Edge transfer first — needs source's adj entries intact.
        if transfer_edges.is_some() {
            transfer_node_edges(
                source_id,
                target_id,
                duplicate,
                transfer_edge_properties,
                ctx,
            )?;
        }

        // Delete source SECOND, before issuing target's index updates. Unique
        // B-tree indexes on shared properties would otherwise reject the
        // target's new value because the source still holds the old key.
        detach_delete_node(source_id, ctx)?;

        // Now safe to register target's merged property changes with the
        // index registries — any colliding source entries are gone.
        notify_indexes_for_target_change(
            target_id,
            &target_label,
            &target_props_before,
            &target_rec.props,
            ctx,
        )?;

        // Persist merged target record.
        ctx.mvcc_put_node(ctx.shard_id, target_id, &target_rec)?;
        ctx.write_stats.properties_set += 1;

        // Fire BEFORE COMMIT UPDATE triggers on the merged target node's
        // labels. MERGE NODES logically rewrites the target's prop set —
        // `$before` is the target's pre-merge property map (resolved via the
        // interner, plus pre-merge `extra`); `$after` is the post-merge
        // prop map. Symmetric with SET-driven UPDATE firing.
        let mut before_props: std::collections::BTreeMap<String, Value> =
            std::collections::BTreeMap::new();
        for (&fid, v) in &target_props_before {
            if let Some(name) = ctx.interner.resolve(fid) {
                before_props.insert(name.to_string(), v.clone());
            }
        }
        for (name, v) in &target_extra_before {
            before_props.insert(name.clone(), v.clone());
        }
        let (target_labels_now, after_props) = snapshot_node_record(&target_rec, ctx);
        let trigger_params = trigger_params_for_node_update(target_id, &before_props, &after_props);
        for label in &target_labels_now {
            let target_segment =
                coordinode_core::schema::triggers::TriggerTargetSchema::label(label.clone())
                    .index_key_segment();
            let matched = ctx.lookup_matching_triggers(&target_segment, "u")?;
            if !matched.is_empty() {
                fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
            }
        }

        // Refresh the row's pre-bound property columns for the target variable
        // so downstream RETURN / WITH / WHERE expressions see merged values
        // without having to re-read storage. Property access in the executor
        // is row-column-first.
        let target_var = target;
        let mut row_with_refresh = row.clone();
        // Remove all stale columns for the target var first (handles drops via
        // ON CONFLICT SET — currently impossible, but defensive).
        let stale_keys: Vec<String> = row_with_refresh
            .iter()
            .filter(|(k, _)| k.starts_with(&format!("{target_var}.")))
            .map(|(k, _)| k.clone())
            .collect();
        for k in stale_keys {
            row_with_refresh.remove(&k);
        }
        for (&field_id, value) in &target_rec.props {
            if let Some(field_name) = ctx.interner.resolve(field_id) {
                row_with_refresh.insert(format!("{target_var}.{field_name}"), value.clone());
            }
        }
        if let Some(extra) = &target_rec.extra {
            for (name, value) in extra {
                row_with_refresh.insert(format!("{target_var}.{name}"), value.clone());
            }
        }
        row_with_refresh.insert(
            format!("{target_var}.__label__"),
            Value::String(target_label.clone()),
        );

        // The non-surviving variable's columns refer to a deleted node — drop
        // them so RETURN/WITH/WHERE never resolve them to stale values.
        let source_var = if target == source_a {
            source_b
        } else {
            source_a
        };
        let drop_keys: Vec<String> = row_with_refresh
            .iter()
            .filter(|(k, _)| k == &source_var || k.starts_with(&format!("{source_var}.")))
            .map(|(k, _)| k.clone())
            .collect();
        for k in drop_keys {
            row_with_refresh.remove(&k);
        }

        out.push(row_with_refresh);
    }

    Ok(out)
}

/// CLONE NODE executor: deep-copy a bound node into a fresh node.
///
/// The source's current stored record is read via `mvcc_get_node`. COMPUTED
/// properties are never part of the stored body (they are injected on read), so
/// they are naturally excluded from the copy. The clone is created through the
/// same path as `CREATE` ([`execute_create_node`]) so it inherits cluster-safe
/// id allocation, b-tree / vector / text / spatial index registration,
/// BlobStore dedup, schema enforcement, and CREATE triggers. `SET` overrides
/// run post-create through the standard update path so every `SetItem` form
/// (assignment, nested path, doc-function, replace) and its index notifications
/// are handled identically to a `SET` clause.
///
/// Temporal-labelled sources are safe-rejected (mirror of MERGE NODES): cloning
/// the current version into a fresh node needs the per-version write path, and
/// version history is never copied (it would forge the system-time axis). Edge
/// cloning (`WITH EDGES`) is a tracked follow-up.
#[allow(clippy::too_many_arguments)]
fn execute_clone_node(
    input_rows: &[Row],
    source: &str,
    target: &str,
    with_edges: bool,
    with_properties: bool,
    set_items: &[crate::cypher::ast::SetItem],
    as_of: Option<&Expr>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    if source == target {
        return Err(ExecutionError::Unsupported(
            "CLONE NODE source and clone variables must differ".into(),
        ));
    }

    let mut out = Vec::with_capacity(input_rows.len());
    for row in input_rows {
        let a_id = match row.get(source) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            // Source not bound on this row — nothing to clone for it.
            _ => continue,
        };

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        // `AS OF <ts>` selects the source's valid-version active at that
        // valid-time instant; default is the current version (active now).
        let as_of_ms: Option<i64> = match as_of {
            Some(expr) => match eval_expr(expr, row) {
                Value::Int(ms) => Some(ms),
                Value::Timestamp(ms) => Some(ms),
                other => {
                    return Err(ExecutionError::Unsupported(format!(
                        "CLONE NODE AS OF requires an INT or TIMESTAMP (epoch ms), got {other:?}"
                    )));
                }
            },
            None => None,
        };
        // Non-temporal nodes read through the point key; temporal nodes have no
        // point key (per-version layout), so read the valid-version active at the
        // requested instant (the current version when there is no AS OF).
        let source_rec = match ctx.mvcc_get_node(ctx.shard_id, a_id)? {
            Some(r) if as_of_ms.is_none() => r,
            _ => ctx
                .mvcc_get_node_at(ctx.shard_id, a_id, as_of_ms.unwrap_or(now_ms))?
                .ok_or_else(|| {
                    let suffix = as_of_ms
                        .map(|t| format!(" as of valid-time {t}"))
                        .unwrap_or_default();
                    ExecutionError::Unsupported(format!(
                        "CLONE NODE source node {a_id} not found{suffix}"
                    ))
                })?,
        };

        let labels = source_rec.labels.clone();

        // Temporal labels are cloned into a fresh node whose FIRST version is
        // valid from NOW (the clone's valid-time history begins at clone time;
        // the source's transaction-time/version history is never copied). The
        // create path requires `valid_from` on temporal labels — injected below.
        let mut is_temporal = false;
        for lbl in &labels {
            if let Ok(Some(s)) = ctx.load_current_label_schema(lbl) {
                if s.temporal {
                    is_temporal = true;
                    break;
                }
            }
        }

        // Reconstruct create-time properties from the source's stored body.
        // Version- and engine-axis fields are never forwarded: `__ingestion_ts__`
        // is the engine-assigned system-time, and `valid_from` / `valid_to` are
        // the source's valid-time interval — the clone gets its own `valid_from`
        // (NOW) below, so inheriting the source's would back-date the clone.
        let mut properties: Vec<(String, Expr)> = if with_properties {
            let mut props = Vec::new();
            for (field_id, val) in &source_rec.props {
                let Some(name) = ctx.interner.resolve(*field_id) else {
                    continue;
                };
                if matches!(name, "__ingestion_ts__" | "valid_from" | "valid_to") {
                    continue;
                }
                props.push((name.to_string(), Expr::Literal(val.clone())));
            }
            if let Some(extra) = &source_rec.extra {
                for (k, v) in extra {
                    if matches!(k.as_str(), "__ingestion_ts__" | "valid_from" | "valid_to") {
                        continue;
                    }
                    props.push((k.clone(), Expr::Literal(v.clone())));
                }
            }
            props
        } else {
            Vec::new()
        };

        // valid_from = NOW for the clone's first version. An explicit
        // `SET b.valid_from = ...` (applied after create) back-dates it.
        if is_temporal {
            properties.push(("valid_from".to_string(), Expr::Literal(Value::Int(now_ms))));
        }

        let created = execute_create_node(
            std::slice::from_ref(row),
            Some(target),
            &labels,
            &properties,
            ctx,
        )?;

        let mut rows_after = if set_items.is_empty() {
            created
        } else {
            execute_update(
                &created,
                set_items,
                &crate::cypher::ast::ViolationMode::Fail,
                ctx,
            )?
        };

        if with_edges {
            // Clone every incident edge onto the new node. Extract the clone's
            // id from the created row first so the immutable row borrow ends
            // before the mutating edge writes.
            let clone_ids: Vec<NodeId> = rows_after
                .iter()
                .filter_map(|r| match r.get(target) {
                    Some(Value::Int(id)) => Some(NodeId::from_raw(*id as u64)),
                    _ => None,
                })
                .collect();
            for b_id in clone_ids {
                clone_incident_edges(a_id, b_id, ctx)?;
            }
        }

        out.append(&mut rows_after);
    }
    Ok(out)
}

/// Clone every incident edge of `a` onto `b`, preserving edge type, direction,
/// and edge properties. Outgoing `a→x` becomes `b→x`, incoming `x→a` becomes
/// `x→b`, and the self-loop `a→a` becomes `b→b` (handled once via the forward
/// scan, skipped in the reverse scan to avoid a duplicate). Adjacency updates
/// use posting-list merge operators (conflict-free with concurrent edge writes
/// on unrelated nodes). Edge properties are copied verbatim through the
/// canonical edge-property codec.
///
/// Edge-vector index registration for cloned edges carrying a vector property
/// is not performed here (rare; a follow-up) — the property bytes are still
/// copied, so the data is intact.
fn clone_incident_edges(
    a_id: NodeId,
    b_id: NodeId,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    let edge_types = ctx.list_edge_types()?;
    for et in &edge_types {
        // Temporal edge types keep one edgeprop entry per `valid_from`; cloning
        // them needs the per-version adjacency-move path (shared with REDIRECT
        // EDGES on temporal types). Until that lands, reject WITH EDGES when the
        // source actually has incident edges of a temporal type rather than
        // silently writing an adjacency with no per-version edgeprop.
        if lookup_edge_type_temporal(et, ctx)? {
            let has_incident =
                ctx.adj_get_fwd(et, a_id)?.is_some() || ctx.adj_get_rev(et, a_id)?.is_some();
            if has_incident {
                return Err(ExecutionError::Unsupported(format!(
                    "CLONE NODE WITH EDGES across temporal edge type '{et}' is not yet \
                     supported (per-version edge cloning is a follow-up); the node and its \
                     non-temporal edges are cloned, temporal edges are not."
                )));
            }
            continue;
        }
        // Outgoing edges a→x  ⇒  b→x  (self-loop a→a ⇒ b→b).
        if let Some(fwd) = ctx.adj_get_fwd(et, a_id)? {
            let targets: Vec<u64> = fwd.iter().collect();
            for x_uid in targets {
                let x = NodeId::from_raw(x_uid);
                let tgt = if x == a_id { b_id } else { x };
                let props = ctx.mvcc_get_edge_props(et, a_id, x)?;
                ctx.adj_merge_add_fwd(et, b_id, tgt.as_raw());
                ctx.adj_merge_add_rev(et, tgt, b_id.as_raw());
                if let Some(p) = props {
                    ctx.mvcc_put_edge_props(et, b_id, tgt, &p)?;
                }
            }
        }
        // Incoming edges x→a  ⇒  x→b. The self-loop was already cloned by the
        // forward scan, so skip x == a here.
        if let Some(rev) = ctx.adj_get_rev(et, a_id)? {
            let sources: Vec<u64> = rev.iter().collect();
            for x_uid in sources {
                let x = NodeId::from_raw(x_uid);
                if x == a_id {
                    continue;
                }
                let props = ctx.mvcc_get_edge_props(et, x, a_id)?;
                ctx.adj_merge_add_fwd(et, x, b_id.as_raw());
                ctx.adj_merge_add_rev(et, b_id, x.as_raw());
                if let Some(p) = props {
                    ctx.mvcc_put_edge_props(et, x, b_id, &p)?;
                }
            }
        }
    }
    Ok(())
}

/// A snapshot of one edge type's neighbours of the source node, captured before
/// any mutation so re-pointing never reads a partially-rewritten posting list.
struct RedirectSnap {
    edge_type: String,
    out_neighbours: Vec<u64>,
    in_neighbours: Vec<u64>,
}

/// REDIRECT EDGES executor: move a bound node's edges onto another bound node.
///
/// Outgoing `a→x` becomes `b→x`, incoming `x→a` becomes `x→b`, via posting-list
/// merge operators (no read-modify-write); edge properties move through the
/// canonical edge-rewiring helper. The self-loop `a→a` re-points by direction:
/// `BOTH` ⇒ `b→b` (both endpoints move, processed once), `OUTGOING` ⇒ `b→a`,
/// `INCOMING` ⇒ `a→b`. Adjacency is a set, so re-pointing onto an edge the
/// destination already has is naturally idempotent. Temporal edge types are
/// safe-rejected (per-version re-pointing is a follow-up).
fn execute_redirect_edges(
    input_rows: &[Row],
    source: &str,
    target: &str,
    edge_types: Option<&[String]>,
    direction: crate::cypher::ast::RedirectDirection,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use crate::cypher::ast::RedirectDirection;

    if source == target {
        // Redirecting a node's edges onto itself changes nothing.
        return Ok(input_rows.to_vec());
    }
    let do_out = matches!(
        direction,
        RedirectDirection::Both | RedirectDirection::Outgoing
    );
    let do_in = matches!(
        direction,
        RedirectDirection::Both | RedirectDirection::Incoming
    );
    let both = matches!(direction, RedirectDirection::Both);

    let types: Vec<String> = match edge_types {
        Some(ts) => ts.to_vec(),
        None => ctx.list_edge_types()?,
    };

    // Temporal safe-reject (mirror of CLONE / MERGE NODES).
    for et in &types {
        if lookup_edge_type_temporal(et, ctx)? {
            return Err(ExecutionError::Unsupported(format!(
                "REDIRECT EDGES on temporal edge type '{et}' is not yet supported \
                 (per-version edge re-pointing is a follow-up)."
            )));
        }
    }

    let mut out = Vec::with_capacity(input_rows.len());
    for row in input_rows {
        let a_id = match row.get(source) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => {
                out.push(row.clone());
                continue;
            }
        };
        let b_id = match row.get(target) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => {
                out.push(row.clone());
                continue;
            }
        };
        if a_id == b_id {
            out.push(row.clone());
            continue;
        }

        // Snapshot every selected type's neighbours BEFORE mutating, so a later
        // remove never perturbs a list still being read.
        let mut snaps: Vec<RedirectSnap> = Vec::with_capacity(types.len());
        for et in &types {
            let out_neighbours = if do_out {
                ctx.adj_get_fwd(et, a_id)?
                    .map(|p| p.iter().collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let in_neighbours = if do_in {
                ctx.adj_get_rev(et, a_id)?
                    .map(|p| p.iter().collect())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            if !out_neighbours.is_empty() || !in_neighbours.is_empty() {
                snaps.push(RedirectSnap {
                    edge_type: et.clone(),
                    out_neighbours,
                    in_neighbours,
                });
            }
        }

        for snap in &snaps {
            let et = &snap.edge_type;
            // Outgoing a→x ⇒ b→x (self-loop ⇒ b→b in BOTH, else b→a).
            for &x_uid in &snap.out_neighbours {
                let x = NodeId::from_raw(x_uid);
                if x == a_id {
                    let new_tgt = if both { b_id } else { a_id };
                    ctx.adj_merge_remove_fwd(et, a_id, a_id.as_raw());
                    ctx.adj_merge_remove_rev(et, a_id, a_id.as_raw());
                    ctx.adj_merge_add_fwd(et, b_id, new_tgt.as_raw());
                    ctx.adj_merge_add_rev(et, new_tgt, b_id.as_raw());
                    ctx.mvcc_move_edge_props(et, a_id, a_id, b_id, new_tgt)?;
                } else {
                    ctx.adj_merge_remove_fwd(et, a_id, x.as_raw());
                    ctx.adj_merge_remove_rev(et, x, a_id.as_raw());
                    ctx.adj_merge_add_fwd(et, b_id, x.as_raw());
                    ctx.adj_merge_add_rev(et, x, b_id.as_raw());
                    ctx.mvcc_move_edge_props(et, a_id, x, b_id, x)?;
                }
            }
            // Incoming x→a ⇒ x→b.
            for &x_uid in &snap.in_neighbours {
                let x = NodeId::from_raw(x_uid);
                if x == a_id {
                    // Self-loop already handled by the outgoing pass in BOTH.
                    if both {
                        continue;
                    }
                    // INCOMING-only: a→a ⇒ a→b (target moves, source stays).
                    ctx.adj_merge_remove_fwd(et, a_id, a_id.as_raw());
                    ctx.adj_merge_remove_rev(et, a_id, a_id.as_raw());
                    ctx.adj_merge_add_fwd(et, a_id, b_id.as_raw());
                    ctx.adj_merge_add_rev(et, b_id, a_id.as_raw());
                    ctx.mvcc_move_edge_props(et, a_id, a_id, a_id, b_id)?;
                } else {
                    ctx.adj_merge_remove_fwd(et, x, a_id.as_raw());
                    ctx.adj_merge_remove_rev(et, a_id, x.as_raw());
                    ctx.adj_merge_add_fwd(et, x, b_id.as_raw());
                    ctx.adj_merge_add_rev(et, b_id, x.as_raw());
                    ctx.mvcc_move_edge_props(et, x, a_id, x, b_id)?;
                }
            }
        }

        out.push(row.clone());
    }
    Ok(out)
}

/// Merge `source.props` into `target.props` per the chosen conflict strategy.
/// `extra` overflow maps are merged with the same policy.
fn merge_node_properties(
    target: &mut NodeRecord,
    source: &NodeRecord,
    conflict: &crate::cypher::ast::MergeNodesConflictStrategy,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use crate::cypher::ast::MergeNodesConflictStrategy as S;
    match conflict {
        S::KeepFirst => {
            // Target wins on collision; source fills only missing keys.
            for (k, v) in &source.props {
                target.props.entry(*k).or_insert_with(|| v.clone());
            }
            if let Some(src_extra) = &source.extra {
                let tgt_extra = target.extra.get_or_insert_with(HashMap::new);
                for (k, v) in src_extra {
                    tgt_extra.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        }
        S::KeepLast => {
            for (k, v) in &source.props {
                target.props.insert(*k, v.clone());
            }
            if let Some(src_extra) = &source.extra {
                let tgt_extra = target.extra.get_or_insert_with(HashMap::new);
                for (k, v) in src_extra {
                    tgt_extra.insert(k.clone(), v.clone());
                }
            }
        }
        S::Coalesce => {
            // Non-null source fills nulls on target (or missing keys).
            for (k, v) in &source.props {
                let existing = target.props.get(k);
                let target_is_null_or_missing = matches!(existing, None | Some(Value::Null));
                if target_is_null_or_missing && !matches!(v, Value::Null) {
                    target.props.insert(*k, v.clone());
                }
            }
            if let Some(src_extra) = &source.extra {
                let tgt_extra = target.extra.get_or_insert_with(HashMap::new);
                for (k, v) in src_extra {
                    let existing = tgt_extra.get(k);
                    let null_or_missing = matches!(existing, None | Some(Value::Null));
                    if null_or_missing && !matches!(v, Value::Null) {
                        tgt_extra.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        S::SetExpressions(items) => {
            // SET expressions are evaluated against the input row, which already
            // binds both source variable names → node ids. The SET items target
            // the surviving node's variable; we apply each one to `target` here.
            for item in items {
                apply_merge_nodes_set_item(target, item, row, ctx)?;
            }
        }
    }
    Ok(())
}

/// Apply a single `SET` item against an in-memory `NodeRecord` during MERGE NODES.
///
/// This is a thin wrapper that evaluates the expression value against the input
/// row and writes it into `target.props` (resolving the property name via the
/// field interner). Unlike `execute_update`, no schema enforcement runs here:
/// MERGE NODES consolidates already-present data, so we trust the user-supplied
/// strategy — explicit downstream SET clauses still go through `execute_update`.
fn apply_merge_nodes_set_item(
    target: &mut NodeRecord,
    item: &crate::cypher::ast::SetItem,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use crate::cypher::ast::SetItem;
    match item {
        SetItem::Property { property, expr, .. } => {
            let val = eval_expr(expr, row).map_to_document();
            let field_id = ctx.interner.intern(property);
            target.set(field_id, val);
        }
        SetItem::PropertyPath { path, expr, .. } => {
            // Nested path SET: walk into `extra` (top-level path segment as
            // string key), creating a document along the way. For first
            // implementation, only support single-segment paths via the
            // declared props path. Multi-segment paths fall through to extra
            // with a dotted string key — matches existing executor behavior.
            let val = eval_expr(expr, row).map_to_document();
            if path.len() == 1 {
                let field_id = ctx.interner.intern(&path[0]);
                target.set(field_id, val);
            } else {
                target.set_extra(path.join("."), val);
            }
        }
        SetItem::AddLabel { label, .. } => {
            target.add_label(label.clone());
        }
        SetItem::MergeProperties { .. }
        | SetItem::ReplaceProperties { .. }
        | SetItem::DocFunction { .. } => {
            return Err(ExecutionError::Unsupported(
                "bulk map assignment (n = {..} / n += {..}) and document mutation \
                 functions (doc_push/doc_pull/...) are not supported inside \
                 MERGE NODES ON CONFLICT SET — express per-property: a.prop = expr"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

/// Propagate property-level changes on the surviving target node to every
/// index registry (B-tree / vector / text). Treats added/replaced properties
/// as `delete(old) + insert(new)` so unique constraints, vector neighbour
/// lists, and inverted indexes stay in sync.
///
/// `old` is the property map snapshotted BEFORE the merge; `new` is the same
/// map after `merge_node_properties` ran. Properties that were dropped
/// entirely (which the current merge strategies don't produce, but we handle
/// for completeness) emit a delete with no follow-up insert.
fn notify_indexes_for_target_change(
    target_id: NodeId,
    label: &str,
    old: &HashMap<u32, Value>,
    new: &HashMap<u32, Value>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    // Collect old entries that changed/disappeared, and the full new set
    // (insert is idempotent on the registries — re-inserting an unchanged
    // value is a no-op cost-wise but no-harm semantically).
    for (field_id, old_val) in old {
        let unchanged = new.get(field_id) == Some(old_val);
        if unchanged {
            continue;
        }
        let Some(name) = ctx.interner.resolve(*field_id) else {
            continue;
        };
        let name = name.to_string();
        if let Some(btree_reg) = ctx.btree_index_registry {
            btree_reg
                .on_node_deleted(
                    ctx.engine,
                    target_id,
                    label,
                    &[(name.clone(), old_val.clone())],
                )
                .map_err(ExecutionError::Storage)?;
        }
        if let Some(registry) = ctx.vector_index_registry {
            if try_extract_vector(old_val).is_some() {
                registry.on_vector_deleted(label, target_id, &name);
            }
        }
        if let Some(registry) = ctx.text_index_registry {
            if old_val.as_str().is_some() {
                registry.on_text_deleted(label, target_id, &name);
            }
        }
    }
    for (field_id, new_val) in new {
        let unchanged = old.get(field_id) == Some(new_val);
        if unchanged {
            continue;
        }
        let Some(name) = ctx.interner.resolve(*field_id) else {
            continue;
        };
        let name = name.to_string();
        if let Some(btree_reg) = ctx.btree_index_registry {
            btree_reg
                .on_node_created(
                    ctx.engine,
                    target_id,
                    label,
                    &[(name.clone(), new_val.clone())],
                )
                .map_err(|v| {
                    ExecutionError::Conflict(format!(
                        "MERGE NODES would violate unique constraint on `{}`: \
                         property `{}` already has value {:?}",
                        v.index_name, v.property, v.value
                    ))
                })?;
        }
        if let Some(registry) = ctx.vector_index_registry {
            if let Some(vec_data) = try_extract_vector(new_val) {
                registry.on_vector_written(label, target_id, &name, &vec_data);
            }
        }
        if let Some(registry) = ctx.text_index_registry {
            if let Some(text) = new_val.as_str() {
                registry.on_text_written(label, target_id, &name, text);
            }
        }
    }
    Ok(())
}

/// Transfer every edge of `source_id` onto `target_id`.
///
/// For each registered edge type, re-points both outgoing and incoming posting
/// lists via merge operators (conflict-free with concurrent writes). When the
/// edge has properties, the edgeprop record is renamed by writing the new key
/// and deleting the old one. Duplicate edges (target↔peer already present) are
/// handled per `duplicate`.
fn transfer_node_edges(
    source_id: NodeId,
    target_id: NodeId,
    duplicate: &crate::cypher::ast::MergeNodesDuplicateStrategy,
    transfer_edge_properties: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use crate::cypher::ast::MergeNodesDuplicateStrategy as D;

    let edge_types = ctx.list_edge_types()?;
    for edge_type in &edge_types {
        let temporal = lookup_edge_type_temporal(edge_type, ctx)?;
        for direction in [AdjDirection::Out, AdjDirection::In] {
            let Some(plist) = (match direction {
                AdjDirection::Out => ctx.adj_get_fwd(edge_type, source_id),
                AdjDirection::In => ctx.adj_get_rev(edge_type, source_id),
            })?
            else {
                continue;
            };

            // Snapshot peers — iterating the live posting list while issuing
            // merge_remove on it during the loop would be safe (deferred) but
            // collecting up-front keeps the loop body simpler.
            let peers: Vec<u64> = plist.iter().collect();
            let target_plist = match direction {
                AdjDirection::Out => ctx.adj_get_fwd(edge_type, target_id),
                AdjDirection::In => ctx.adj_get_rev(edge_type, target_id),
            }?;

            for peer_uid in peers {
                let mut peer_id = NodeId::from_raw(peer_uid);
                // Self-loop b→b becomes target→target.
                if peer_id == source_id {
                    peer_id = target_id;
                }

                let already_on_target = target_plist
                    .as_ref()
                    .is_some_and(|p| p.iter().any(|u| u == peer_id.as_raw()));

                // Edge endpoints in the canonical (src, tgt) order used by edgeprop.
                let (old_src, old_tgt) = match direction {
                    AdjDirection::Out => (source_id, NodeId::from_raw(peer_uid)),
                    AdjDirection::In => (NodeId::from_raw(peer_uid), source_id),
                };
                let (new_src, new_tgt) = match direction {
                    AdjDirection::Out => (target_id, peer_id),
                    AdjDirection::In => (peer_id, target_id),
                };

                let is_duplicate = already_on_target;
                let keep_old_edge = matches!(duplicate, D::KeepTarget) && is_duplicate;

                if keep_old_edge {
                    // Drop the source-side edge entirely (no transfer).
                    // Remove from source's posting list + counterpart, and
                    // delete edgeprop for the old endpoints.
                    match direction {
                        AdjDirection::Out => ctx.adj_merge_remove_rev(
                            edge_type,
                            NodeId::from_raw(peer_uid),
                            source_id.as_raw(),
                        ),
                        AdjDirection::In => ctx.adj_merge_remove_fwd(
                            edge_type,
                            NodeId::from_raw(peer_uid),
                            source_id.as_raw(),
                        ),
                    };
                    delete_edgeprop_for_pair(edge_type, old_src, old_tgt, temporal, ctx)?;
                    ctx.write_stats.edges_deleted += 1;
                    continue;
                }

                // Re-point this edge.
                // 1. Adjacency rewrite: remove source from posting lists; add target.
                match direction {
                    AdjDirection::Out => ctx.adj_merge_remove_rev(
                        edge_type,
                        NodeId::from_raw(peer_uid),
                        source_id.as_raw(),
                    ),
                    AdjDirection::In => ctx.adj_merge_remove_fwd(
                        edge_type,
                        NodeId::from_raw(peer_uid),
                        source_id.as_raw(),
                    ),
                };

                // For KeepBoth duplicate strategy: add even if duplicate (parallel edge).
                // For MergeProperties/KeepTarget where edge persists, also add — uniqueness
                // is enforced by posting list de-dup on the key value.
                match direction {
                    AdjDirection::Out => {
                        ctx.adj_merge_add_rev(edge_type, peer_id, target_id.as_raw())
                    }
                    AdjDirection::In => {
                        ctx.adj_merge_add_fwd(edge_type, peer_id, target_id.as_raw())
                    }
                };

                // Target's own posting list also has to learn about the new
                // peer (otherwise traversals starting from target won't see it
                // — only inverse traversals would). The source's own posting
                // list is deleted wholesale after the peer loop completes.
                match direction {
                    AdjDirection::Out => {
                        ctx.adj_merge_add_fwd(edge_type, target_id, peer_id.as_raw())
                    }
                    AdjDirection::In => {
                        ctx.adj_merge_add_rev(edge_type, target_id, peer_id.as_raw())
                    }
                };

                // 2. Edge-property transfer. Per spec, edge properties always
                // move with the edge; `transfer_edge_properties` is a redundant
                // syntactic ack. The flag is consulted only so future opt-out
                // policies can be encoded without re-shaping the call site.
                let _ = transfer_edge_properties;
                transfer_edgeprop_record(
                    edge_type,
                    old_src,
                    old_tgt,
                    new_src,
                    new_tgt,
                    temporal,
                    is_duplicate && matches!(duplicate, D::MergeProperties),
                    ctx,
                )?;
                // For non-duplicate transfers, edge count is preserved
                // (one source-side edge becomes one target-side edge).
                // For duplicate with MergeProperties, source-side edge is
                // collapsed onto target → count drops by one.
                if is_duplicate && matches!(duplicate, D::MergeProperties) {
                    ctx.write_stats.edges_deleted += 1;
                }
            }

            // Drop the source-side posting list — every entry was either
            // re-pointed onto target or dropped per KEEP_TARGET. Goes through
            // the MVCC buffer so that an error later in the merge (STRICT
            // schema violation on the next input row, OCC conflict on flush,
            // etc.) rolls back this drop together with all other writes.
            match direction {
                AdjDirection::Out => ctx.mvcc_delete_adj_fwd(edge_type, source_id)?,
                AdjDirection::In => ctx.mvcc_delete_adj_rev(edge_type, source_id)?,
            };
        }
    }
    Ok(())
}

/// Move the edge-property record for an edge from `(old_src, old_tgt)` to `(new_src, new_tgt)`.
///
/// When `merge_with_existing` is true, the source-side edgeprop is merged into
/// any existing target-side record using COALESCE semantics (non-null source
/// fills null target), then deleted.
#[allow(clippy::too_many_arguments)]
fn transfer_edgeprop_record(
    edge_type: &str,
    old_src: NodeId,
    old_tgt: NodeId,
    new_src: NodeId,
    new_tgt: NodeId,
    temporal: bool,
    merge_with_existing: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    if temporal {
        // Temporal edge type: move every version (preserving valid_from) via
        // the Layer-4 store, then return.
        ctx.mvcc_transfer_all_edge_prop_versions(edge_type, old_src, old_tgt, new_src, new_tgt)?;
        return Ok(());
    }

    let Some(old_props) = ctx.mvcc_get_edge_props(edge_type, old_src, old_tgt)? else {
        return Ok(());
    };

    if merge_with_existing {
        if let Some(mut tgt) = ctx.mvcc_get_edge_props(edge_type, new_src, new_tgt)? {
            // Edgeprop records are `Vec<(field_id, Value)>` of interned
            // (field_id, value) pairs. COALESCE: non-null source fills
            // missing/null target keys.
            let mut tgt_index: HashMap<u32, usize> =
                tgt.iter().enumerate().map(|(i, (k, _))| (*k, i)).collect();
            for (k, v) in old_props {
                if matches!(v, Value::Null) {
                    continue;
                }
                match tgt_index.get(&k) {
                    Some(&idx) => {
                        if matches!(tgt[idx].1, Value::Null) {
                            tgt[idx].1 = v;
                        }
                    }
                    None => {
                        tgt_index.insert(k, tgt.len());
                        tgt.push((k, v));
                    }
                }
            }
            ctx.mvcc_put_edge_props(edge_type, new_src, new_tgt, &tgt)?;
            ctx.mvcc_delete_edge_props(edge_type, old_src, old_tgt)?;
            return Ok(());
        }
    }

    // No collision (or merge not requested): simple rename.
    ctx.mvcc_put_edge_props(edge_type, new_src, new_tgt, &old_props)?;
    ctx.mvcc_delete_edge_props(edge_type, old_src, old_tgt)?;
    Ok(())
}

/// Delete every edgeprop entry for an edge — single key (non-temporal) or
/// prefix-scan (temporal).
fn delete_edgeprop_for_pair(
    edge_type: &str,
    src: NodeId,
    tgt: NodeId,
    temporal: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    if temporal {
        ctx.mvcc_delete_all_edge_prop_versions(edge_type, src, tgt)?;
    } else {
        ctx.mvcc_delete_edge_props(edge_type, src, tgt)?;
    }
    Ok(())
}

/// Hard-delete a node and all of its remaining adjacency entries.
///
/// Called from `execute_merge_nodes` after edge transfer (or directly when
/// `TRANSFER EDGES` was omitted, in which case all of the source's edges are
/// dropped). Mirrors the DETACH branch of `execute_delete`.
fn detach_delete_node(
    node_id: NodeId,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    // Notify all index registries (B-tree, vector, text) of the soon-to-be-deleted
    // properties so their entries are cleaned up before primary storage drops
    // the node. Without this, indexes leak: unique constraints would still
    // reject re-creation with the same value, vector/text searches would
    // return stale UIDs.
    // Snapshot pre-mutation node state once for trigger firing — re-used after
    // the node is deleted to populate `$before` for the BEFORE COMMIT DELETE
    // trigger. Mirrors the `execute_delete` / `cascade_delete_source_node`
    // wiring so that MERGE NODES' source-cleanup fires the same triggers as
    // DETACH DELETE on the same node.
    let node_pre_snapshot: Option<(Vec<String>, std::collections::BTreeMap<String, Value>)> = ctx
        .mvcc_get_node(ctx.shard_id, node_id)?
        .map(|rec| snapshot_node_record(&rec, ctx));
    if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
        {
            let label = record.primary_label().to_string();
            if let Some(btree_reg) = ctx.btree_index_registry {
                let props: Vec<(String, Value)> = record
                    .props
                    .iter()
                    .filter_map(|(&field_id, value)| {
                        ctx.interner
                            .resolve(field_id)
                            .map(|name| (name.to_string(), value.clone()))
                    })
                    .collect();
                btree_reg
                    .on_node_deleted(ctx.engine, node_id, &label, &props)
                    .map_err(ExecutionError::Storage)?;
            }
            for (&field_id, value) in &record.props {
                if let Some(prop_name) = ctx.interner.resolve(field_id) {
                    if let Some(registry) = ctx.vector_index_registry {
                        if try_extract_vector(value).is_some() {
                            registry.on_vector_deleted(&label, node_id, prop_name);
                        }
                    }
                    if let Some(registry) = ctx.text_index_registry {
                        if value.as_str().is_some() {
                            registry.on_text_deleted(&label, node_id, prop_name);
                        }
                    }
                }
            }
        }
    }

    let edge_types = ctx.list_edge_types()?;
    let mut adj_targets: Vec<(String, AdjDirection)> = Vec::new();
    for edge_type in &edge_types {
        if ctx.adj_get_fwd(edge_type, node_id)?.is_some() {
            adj_targets.push((edge_type.clone(), AdjDirection::Out));
        }
        if ctx.adj_get_rev(edge_type, node_id)?.is_some() {
            adj_targets.push((edge_type.clone(), AdjDirection::In));
        }
    }
    for (edge_type, direction) in &adj_targets {
        let edge_type = edge_type.as_str();
        let direction = *direction;
        // Probe edge DELETE triggers once per adj key. Re-used across all
        // peer pairs walked from the posting list. Mirrors the wiring in
        // `execute_delete` DETACH branch and `cascade_delete_source_node`.
        let edge_target_segment =
            coordinode_core::schema::triggers::TriggerTargetSchema::edge_type(edge_type)
                .index_key_segment();
        let matched_edge_delete = ctx.lookup_matching_triggers(&edge_target_segment, "d")?;
        let plist = match direction {
            AdjDirection::Out => ctx.adj_get_fwd(edge_type, node_id)?,
            AdjDirection::In => ctx.adj_get_rev(edge_type, node_id)?,
        };
        if let Some(plist) = plist {
            let temporal = lookup_edge_type_temporal(edge_type, ctx)?;
            for peer_uid in plist.iter() {
                let peer_id = NodeId::from_raw(peer_uid);
                match direction {
                    AdjDirection::Out => {
                        ctx.adj_merge_remove_rev(edge_type, peer_id, node_id.as_raw())
                    }
                    AdjDirection::In => {
                        ctx.adj_merge_remove_fwd(edge_type, peer_id, node_id.as_raw())
                    }
                };

                let (ep_src, ep_tgt) = match direction {
                    AdjDirection::Out => (node_id, peer_id),
                    AdjDirection::In => (peer_id, node_id),
                };

                // Pre-snapshot edge prop maps for trigger firing — one per
                // version for temporal edges, single entry for non-temporal.
                // Captured before edgeprop deletion so the body's RYOW
                // reads see the edge as gone.
                let mut edge_delete_snapshots: Vec<std::collections::BTreeMap<String, Value>> =
                    Vec::new();
                if !matched_edge_delete.is_empty() {
                    if temporal {
                        for (_vf, bytes) in
                            ctx.mvcc_scan_edge_prop_version_bytes(edge_type, ep_src, ep_tgt, None)?
                        {
                            edge_delete_snapshots.push(decode_edgeprop_into_map(&bytes, ctx));
                        }
                    } else if let Some(prop_map) =
                        ctx.mvcc_get_edge_props(edge_type, ep_src, ep_tgt)?
                    {
                        edge_delete_snapshots.push(decode_edgeprop_map_into_named(&prop_map, ctx));
                    } else {
                        edge_delete_snapshots.push(std::collections::BTreeMap::new());
                    }
                }

                delete_edgeprop_for_pair(edge_type, ep_src, ep_tgt, temporal, ctx)?;

                if !matched_edge_delete.is_empty() {
                    for before in &edge_delete_snapshots {
                        let trigger_params =
                            trigger_params_for_edge_delete(edge_type, ep_src, ep_tgt, before);
                        fire_before_commit_triggers(&matched_edge_delete, &trigger_params, ctx)?;
                    }
                }
            }
        }
        // MVCC-buffered purge (tombstone + pending-merge clear): rolled
        // back atomically with the surrounding transaction on any later error.
        ctx.mvcc_purge_adj(edge_type, node_id, direction)?;
        ctx.write_stats.edges_deleted += 1;
    }
    // Drop the primary node record.
    ctx.mvcc_delete_node(ctx.shard_id, node_id)?;
    ctx.write_stats.nodes_deleted += 1;

    // Fire BEFORE COMMIT DELETE triggers on the deleted node's labels —
    // mirrors `execute_delete` / `cascade_delete_source_node`. Probe runs
    // AFTER mvcc_delete so the trigger body's MATCH returns empty via RYOW.
    if let Some((labels, before_props)) = node_pre_snapshot {
        let trigger_params = trigger_params_for_node_delete(node_id, &before_props);
        for label in &labels {
            let target_segment =
                coordinode_core::schema::triggers::TriggerTargetSchema::label(label.clone())
                    .index_key_segment();
            let matched = ctx.lookup_matching_triggers(&target_segment, "d")?;
            if !matched.is_empty() {
                fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
            }
        }
    }
    Ok(())
}

// =====================================================================
// DETACH DOCUMENT (R167)
// =====================================================================

/// Execute a DETACH DOCUMENT clause for each input row.
///
/// For each bound source node:
///  1. Read the node record.
///  2. Extract the DOCUMENT value at `property_path` (shallow).
///  3. CREATE a new node with `target_labels`, populated from the document's
///     top-level keys.
///  4. CREATE an edge of `edge_type` between source and target. The canonical
///     form is `(a:Label)-[:TYPE]->(n)` i.e. target → source, so when
///     `edge_direction == Incoming` (from the source's view) we use
///     (target → source); otherwise (source → target).
///  5. Remove `property_path` from the source node via a `DocDelta` merge
///     operand (O(1), no read).
///  6. If `TRANSFER EDGES ON source TO target WHERE type(r) IN [...]` was
///     given, re-point each matching edge on the source to the new target
///     via merge operators on adjacency posting lists.
///
/// All writes happen within the current MVCC transaction — the executor
/// batches them and commits atomically at the end of the query.
#[allow(clippy::too_many_arguments)]
fn execute_detach_document(
    input_rows: &[Row],
    source_variable: &str,
    property_path: &[String],
    target_variable: &str,
    target_labels: &[String],
    edge_type: &str,
    edge_direction: crate::cypher::ast::EdgeFromSource,
    transfer: Option<&crate::cypher::ast::TransferEdgesSpec>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    if property_path.is_empty() {
        return Err(ExecutionError::Unsupported(
            "DETACH DOCUMENT requires a non-empty property path".to_string(),
        ));
    }

    let transfer_types: Option<Vec<String>> = match transfer {
        Some(t) => Some(extract_transfer_edge_types(&t.predicate)?),
        None => None,
    };

    // R172c Phase 3c: DETACH DOCUMENT on a temporal source no longer
    // safe-rejects. The source-mutation side ("remove property from
    // source") routes through the close+open dance — read source's
    // current per-version record, build a new version with the property
    // removed via DocDelta (applied in-memory by
    // `apply_doc_deltas_to_record`), close current at valid_to = NOW.
    //
    // TRANSFER EDGES on a temporal source remains rejected: edge
    // transfer semantics on bitemporal nodes are Phase 4 territory
    // (the new version of the source has a different temporal identity
    // than the closed version — which version owns the transferred
    // edges?). Without that decision we'd silently re-point edges to
    // an ambiguous (node_id, valid_from) pair.

    let mut results: Vec<Row> = Vec::new();

    for input_row in input_rows {
        let source_id = match input_row.get(source_variable) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => {
                return Err(ExecutionError::Unsupported(format!(
                    "DETACH DOCUMENT: variable `{source_variable}` is not a bound node"
                )));
            }
        };

        // Detect whether the source label is temporal — affects how we
        // read the record and how we apply the property removal.
        let source_is_temporal = match input_row.get(&format!("{source_variable}.__label__")) {
            Some(Value::String(primary)) => ctx
                .load_current_label_schema(primary)
                .ok()
                .flatten()
                .is_some_and(|s| s.temporal),
            _ => false,
        };

        if source_is_temporal && transfer_types.is_some() {
            return Err(ExecutionError::Unsupported(format!(
                "DETACH DOCUMENT with TRANSFER EDGES on a temporal source \
                 (var `{source_variable}`) is not yet supported: edge \
                 ownership across version boundaries requires the Phase 4 \
                 (node_id, valid_from) routing. Use plain DETACH DOCUMENT \
                 (no TRANSFER EDGES) and add the desired edges to the new \
                 target explicitly."
            )));
        }

        // 1. Read the source node.
        //
        // Non-temporal: 16-byte node key — same as before.
        // Temporal: 25-byte per-version key derived from the row's
        // `<source>.valid_from` binding. The matched version is the one
        // we'll close; the property is removed on the new version.
        let source_valid_from: Option<i64> = if source_is_temporal {
            match input_row.get(&format!("{source_variable}.valid_from")) {
                Some(Value::Int(ms)) => Some(*ms),
                Some(Value::Timestamp(ms)) => Some(*ms),
                _ => {
                    return Err(ExecutionError::Unsupported(format!(
                        "DETACH DOCUMENT on temporal source `{source_variable}`: \
                         matched row is missing `{source_variable}.valid_from` \
                         (planner must surface valid_from on every temporal node \
                         materialised for mutation)"
                    )));
                }
            }
        } else {
            None
        };
        let Some(record) = ctx.mvcc_get_node_either(ctx.shard_id, source_id, source_valid_from)?
        else {
            return Err(ExecutionError::Unsupported(format!(
                "DETACH DOCUMENT: node {source_id} not found"
            )));
        };

        // 2. Resolve the property value at `property_path`.
        let (field_id_opt, doc_value) =
            resolve_document_property(&record, property_path, ctx.interner)?;

        // 3. Extract the document's top-level (key, Value) pairs for the new node.
        let props = document_top_level_to_props(&doc_value)?;

        // 4. Allocate the new node ID and build its record. We go through
        //    `execute_create_node` via literal-expression properties so that
        //    schema validation / index registries fire identically to a
        //    hand-written CREATE. Build a minimal single-row input so the
        //    executor allocates exactly one new node per detach.
        let literal_props: Vec<(String, Expr)> = props
            .iter()
            .map(|(k, v)| (k.clone(), Expr::Literal(v.clone())))
            .collect();
        let create_rows = execute_create_node(
            std::slice::from_ref(input_row),
            Some(target_variable),
            target_labels,
            &literal_props,
            ctx,
        )?;
        let Some(created) = create_rows.into_iter().next() else {
            return Err(ExecutionError::Unsupported(
                "DETACH DOCUMENT: failed to create target node".to_string(),
            ));
        };
        let target_id = match created.get(target_variable) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => {
                return Err(ExecutionError::Unsupported(
                    "DETACH DOCUMENT: created node missing id".to_string(),
                ));
            }
        };

        // 5. Create the connecting edge.
        //
        //   canonical: `(a:Address)-[:HAS_ADDRESS]->(n)` — a is target, n is source,
        //   edge flows target → source.
        let (edge_src, edge_tgt) = match edge_direction {
            crate::cypher::ast::EdgeFromSource::Incoming => (target_id, source_id),
            crate::cypher::ast::EdgeFromSource::Outgoing => (source_id, target_id),
        };
        create_single_edge(edge_src, edge_tgt, edge_type, ctx)?;

        // 6. Remove the property from the source node.
        //
        // Non-temporal: queue a DocDelta merge operand against the
        // 16-byte node key (O(1), no read).
        // Temporal: read the matched per-version record, apply the
        // DocDelta to the clone in-memory via
        // `apply_doc_deltas_to_record`, write closing record at the
        // current temporal key (valid_to = NOW) and the new version at
        // a fresh temporal key. Preserves history.
        if source_is_temporal {
            let Some(current_valid_from) = source_valid_from else {
                // source_is_temporal implies valid_from was captured
                // above; unreachable in practice.
                return Err(ExecutionError::Unsupported(format!(
                    "DETACH DOCUMENT on temporal source `{source_variable}`: \
                     internal state missing valid_from"
                )));
            };
            let now_us = current_hlc_us() as i64;
            let new_valid_from = if now_us > current_valid_from {
                now_us
            } else {
                current_valid_from + 1
            };
            let mut closing_record = record.clone();
            let mut new_record = record.clone();

            // Build and apply the DocDelta for the property removal,
            // mirroring `emit_property_removal` but applied in-memory.
            use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};
            let delta = if property_path.len() == 1 {
                match field_id_opt {
                    Some(fid) => DocDelta::RemoveProperty {
                        target: PathTarget::PropField(fid),
                        key: None,
                    },
                    None => DocDelta::RemoveProperty {
                        target: PathTarget::Extra,
                        key: Some(property_path[0].clone()),
                    },
                }
            } else {
                let target = match field_id_opt {
                    Some(fid) => PathTarget::PropField(fid),
                    None => PathTarget::Extra,
                };
                DocDelta::DeletePath {
                    target,
                    path: property_path[1..].to_vec(),
                }
            };
            coordinode_storage::engine::merge::apply_doc_deltas_to_record(
                &mut new_record,
                &[delta],
            );

            // Refresh bitemporal axis fields on both records.
            let vf_fid = ctx.interner.intern("valid_from");
            let vt_fid = ctx.interner.intern("valid_to");
            let its_fid = ctx.interner.intern("__ingestion_ts__");
            closing_record.set(vt_fid, Value::Int(new_valid_from));
            new_record.set(vf_fid, Value::Int(new_valid_from));
            new_record.props.remove(&vt_fid);
            new_record.set(its_fid, Value::Int(now_us));

            // Write close-current at the matched key; open-new at a
            // fresh per-version key. `source_valid_from` is `Some(_)` on
            // this temporal branch by construction; surface the bug as
            // an error rather than an internal panic if that invariant
            // were ever violated.
            let current_vf = source_valid_from.ok_or_else(|| {
                ExecutionError::Unsupported(
                    "DETACH temporal branch reached with valid_from=None — internal invariant violation"
                        .into(),
                )
            })?;
            ctx.mvcc_put_node_temporal(ctx.shard_id, source_id, current_vf, &closing_record)?;
            ctx.mvcc_put_node_temporal(ctx.shard_id, source_id, new_valid_from, &new_record)?;
            ctx.write_stats.properties_removed += 1;
        } else {
            emit_property_removal(source_id, property_path, field_id_opt, ctx)?;
        }

        // 7. Optional TRANSFER EDGES.
        if let Some(ref types) = transfer_types {
            transfer_edges_on_node(source_id, target_id, types, ctx)?;
        }

        // 8. Produce output row: preserve source row bindings, add the new
        //    target variable (mirroring `execute_create_node`).
        let mut out = created.clone();
        // The source variable should still reference the (still-existing) source node.
        out.insert(
            source_variable.to_string(),
            Value::Int(source_id.as_raw() as i64),
        );
        // Invalidate the removed property in the row cache.
        let path_str = property_path.join(".");
        out.remove(&format!("{source_variable}.{path_str}"));
        results.push(out);
    }

    Ok(results)
}

/// Resolve a property value on a node record by path.
///
/// Returns `(field_id, value)` where `field_id` is `Some(fid)` if the first
/// path segment corresponds to an interned property (i.e. `PathTarget::PropField`
/// removal is possible) or `None` if the value was found in the `extra`
/// overflow map.
fn resolve_document_property(
    record: &NodeRecord,
    path: &[String],
    interner: &FieldInterner,
) -> Result<(Option<u32>, rmpv::Value), ExecutionError> {
    let first = &path[0];
    let (field_id, root): (Option<u32>, Value) = match interner.lookup(first) {
        Some(fid) => match record.props.get(&fid) {
            Some(v) => (Some(fid), v.clone()),
            None => {
                return Err(ExecutionError::Unsupported(format!(
                    "DETACH DOCUMENT: property `{first}` not found on node"
                )));
            }
        },
        None => {
            // Try the overflow map.
            let extra = record.extra.as_ref().and_then(|m| m.get(first));
            match extra {
                Some(v) => (None, v.clone()),
                None => {
                    return Err(ExecutionError::Unsupported(format!(
                        "DETACH DOCUMENT: property `{first}` not found on node"
                    )));
                }
            }
        }
    };

    // Descend remaining path segments (rmpv navigation).
    let mut current = value_to_rmpv(&root);
    for seg in &path[1..] {
        current = match current {
            rmpv::Value::Map(entries) => {
                let hit = entries
                    .into_iter()
                    .find(|(k, _)| k.as_str() == Some(seg.as_str()))
                    .map(|(_, v)| v);
                match hit {
                    Some(v) => v,
                    None => {
                        return Err(ExecutionError::Unsupported(format!(
                            "DETACH DOCUMENT: path segment `{seg}` not found"
                        )));
                    }
                }
            }
            _ => {
                return Err(ExecutionError::Unsupported(format!(
                    "DETACH DOCUMENT: path segment `{seg}` traverses a non-map value"
                )));
            }
        };
    }

    // The resolved value must be a document/map (Nil is treated as absent).
    match &current {
        rmpv::Value::Nil => Err(ExecutionError::Unsupported(
            "DETACH DOCUMENT: property value is NULL".to_string(),
        )),
        rmpv::Value::Map(_) => Ok((field_id, current)),
        _ => Err(ExecutionError::Unsupported(format!(
            "DETACH DOCUMENT: property at `{}` is not a DOCUMENT/MAP",
            path.join(".")
        ))),
    }
}

/// Lift a `Value` to an `rmpv::Value` for document path navigation.
fn value_to_rmpv(v: &Value) -> rmpv::Value {
    match v {
        Value::Document(doc) => doc.clone(),
        other => other.to_rmpv(),
    }
}

/// Decompose a document (top level must be a map) into (String, Value) pairs
/// suitable for property assignment on the new target node. Nested maps become
/// `Value::Document`, preserving the arch doc's shallow-promotion semantics.
fn document_top_level_to_props(doc: &rmpv::Value) -> Result<Vec<(String, Value)>, ExecutionError> {
    let rmpv::Value::Map(entries) = doc else {
        return Err(ExecutionError::Unsupported(
            "DETACH DOCUMENT: document value is not a map".to_string(),
        ));
    };

    let mut out = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        let Some(key) = k.as_str() else {
            return Err(ExecutionError::Unsupported(
                "DETACH DOCUMENT: non-string key in document".to_string(),
            ));
        };
        // Nested maps/arrays stay as documents (shallow promotion per arch
        // doc); scalars become the matching typed Value.
        out.push((key.to_string(), rmpv_scalar_to_value(v)));
    }
    Ok(out)
}

/// Convert an `rmpv::Value` into the corresponding `Value` variant.
///
/// Scalars map to their typed equivalents; maps and arrays are preserved
/// as `Value::Document(...)` so that nested document structure survives
/// a DETACH DOCUMENT promotion (arch: shallow — nested documents become
/// DOCUMENT properties on the new node).
fn rmpv_scalar_to_value(v: &rmpv::Value) -> Value {
    match v {
        rmpv::Value::Nil => Value::Null,
        rmpv::Value::Boolean(b) => Value::Bool(*b),
        rmpv::Value::Integer(i) => {
            if let Some(n) = i.as_i64() {
                Value::Int(n)
            } else if let Some(n) = i.as_u64() {
                // Saturate unsigned values larger than i64::MAX — keep the bits,
                // accept wraparound for the rare out-of-range case.
                Value::Int(n as i64)
            } else {
                Value::Null
            }
        }
        rmpv::Value::F32(f) => Value::Float(*f as f64),
        rmpv::Value::F64(f) => Value::Float(*f),
        rmpv::Value::String(s) => s
            .as_str()
            .map(|s| Value::String(s.to_string()))
            .unwrap_or(Value::Null),
        rmpv::Value::Binary(b) => Value::Binary(b.clone()),
        rmpv::Value::Array(_) | rmpv::Value::Map(_) | rmpv::Value::Ext(_, _) => {
            Value::Document(v.clone())
        }
    }
}

/// Create a single edge (`src → tgt`) of `edge_type`, mirroring the logic
/// in `execute_create_edge` but without requiring row bindings.
fn create_single_edge(
    src: NodeId,
    tgt: NodeId,
    edge_type: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    ctx.adj_merge_add_fwd(edge_type, src, tgt.as_raw());
    ctx.adj_merge_add_rev(edge_type, tgt, src.as_raw());

    ctx.write_stats.edges_created += 1;

    // Register edge type in schema (idempotent — never clobber existing schema).
    ctx.mvcc_register_edge_type(edge_type)?;

    // Fire BEFORE COMMIT CREATE triggers registered on this edge type.
    // `create_single_edge` is the bare-bones edge insertion path used by
    // DETACH DOCUMENT (for the new connecting edge); semantically a
    // `CREATE (a)-[:TYPE]->(b)` so the same trigger must fire. No inline
    // properties are written here, so `$after` is empty.
    let resolved_props: Vec<(String, Value)> = Vec::new();
    let trigger_params = trigger_params_for_edge_create(edge_type, src, tgt, &resolved_props);
    let target_segment =
        coordinode_core::schema::triggers::TriggerTargetSchema::edge_type(edge_type)
            .index_key_segment();
    let matched = ctx.lookup_matching_triggers(&target_segment, "c")?;
    if !matched.is_empty() {
        fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
    }
    Ok(())
}

/// Emit a `DocDelta::RemoveProperty` (for single-segment paths on an interned
/// field) or `DocDelta::DeletePath` (for nested paths / extra properties).
fn emit_property_removal(
    node_id: NodeId,
    path: &[String],
    field_id_opt: Option<u32>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};

    let delta = if path.len() == 1 {
        match field_id_opt {
            Some(fid) => DocDelta::RemoveProperty {
                target: PathTarget::PropField(fid),
                key: None,
            },
            None => DocDelta::RemoveProperty {
                target: PathTarget::Extra,
                key: Some(path[0].clone()),
            },
        }
    } else {
        // Nested path: `DeletePath` on either an interned prop or extra key.
        let target = match field_id_opt {
            Some(fid) => PathTarget::PropField(fid),
            None => PathTarget::Extra,
        };
        let sub = if field_id_opt.is_some() {
            path[1..].to_vec()
        } else {
            // For extra: the first segment is the extra-map key; remaining are sub-path.
            path[1..].to_vec()
        };
        DocDelta::DeletePath { target, path: sub }
    };

    let operand = delta
        .encode()
        .map_err(|e| ExecutionError::Serialization(format!("DocDelta encode: {e}")))?;
    ctx.mvcc_merge_node_delta(ctx.shard_id, node_id, operand);
    ctx.write_stats.properties_removed += 1;
    Ok(())
}

/// Attempt to extract the edge-type list from a `TRANSFER EDGES WHERE` predicate.
///
/// Supported shapes (enough for the spec in arch/core/document-operations.md):
///   - `type(r) IN ['T1', 'T2', ...]`
///   - `type(r) = 'T1'`
///
/// Anything more complex returns an error; we prefer explicit scope to a
/// misinterpreted transfer.
fn extract_transfer_edge_types(predicate: &Expr) -> Result<Vec<String>, ExecutionError> {
    fn is_type_of_r(expr: &Expr) -> bool {
        matches!(
            expr,
            Expr::FunctionCall { name, args, .. }
                if name.eq_ignore_ascii_case("type")
                    && args.len() == 1
                    && matches!(&args[0], Expr::Variable(v) if v == "r")
        )
    }

    fn lit_string(e: &Expr) -> Option<String> {
        if let Expr::Literal(Value::String(s)) = e {
            Some(s.clone())
        } else {
            None
        }
    }

    match predicate {
        Expr::In { expr, list } if is_type_of_r(expr) => {
            let Expr::List(items) = list.as_ref() else {
                return Err(ExecutionError::Unsupported(
                    "TRANSFER EDGES WHERE: `IN` requires a list literal".to_string(),
                ));
            };
            let mut types = Vec::with_capacity(items.len());
            for it in items {
                let Some(s) = lit_string(it) else {
                    return Err(ExecutionError::Unsupported(
                        "TRANSFER EDGES WHERE: list must contain string literals".to_string(),
                    ));
                };
                types.push(s);
            }
            Ok(types)
        }
        Expr::BinaryOp {
            left,
            op: crate::cypher::ast::BinaryOperator::Eq,
            right,
        } if is_type_of_r(left) => {
            let Some(s) = lit_string(right) else {
                return Err(ExecutionError::Unsupported(
                    "TRANSFER EDGES WHERE: `=` requires a string literal".to_string(),
                ));
            };
            Ok(vec![s])
        }
        _ => Err(ExecutionError::Unsupported(
            "TRANSFER EDGES WHERE supports only `type(r) IN [...]` or `type(r) = '...'`"
                .to_string(),
        )),
    }
}

/// Re-point all edges of the listed types on `source_id` to `target_id`.
///
/// For each edge type T, both directions are scanned:
///  - `adj:T:out:source` — edges where source is the edge's source
///    → counterpart key `adj:T:in:peer` is already `source`; rewrite to `target`
///  - `adj:T:in:source` — edges where source is the edge's target
///    → symmetric
///
/// Implementation uses posting-list merge operators (`adj_merge_add`/`remove`)
/// so there are no OCC conflicts even on high-degree vertices. Edge properties
/// (edgeprop: partition) are physically rewritten via delete+put.
fn transfer_edges_on_node(
    source_id: NodeId,
    target_id: NodeId,
    edge_types: &[String],
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    for edge_type in edge_types {
        // Forward: source is the edge source.
        if let Some(plist) = ctx.adj_get_fwd(edge_type, source_id)? {
            let peers: Vec<u64> = plist.iter().collect();
            for peer_uid in peers {
                let peer_id = NodeId::from_raw(peer_uid);
                // Remove source → peer
                ctx.adj_merge_remove_fwd(edge_type, source_id, peer_uid);
                ctx.adj_merge_remove_rev(edge_type, peer_id, source_id.as_raw());
                // Add target → peer
                ctx.adj_merge_add_fwd(edge_type, target_id, peer_uid);
                ctx.adj_merge_add_rev(edge_type, peer_id, target_id.as_raw());

                // Move edge properties (Layer-4 store owns key shape + body move).
                ctx.mvcc_move_edge_props(edge_type, source_id, peer_id, target_id, peer_id)?;
            }
        }

        // Reverse: source is the edge target.
        if let Some(plist) = ctx.adj_get_rev(edge_type, source_id)? {
            let peers: Vec<u64> = plist.iter().collect();
            for peer_uid in peers {
                let peer_id = NodeId::from_raw(peer_uid);
                ctx.adj_merge_remove_rev(edge_type, source_id, peer_uid);
                ctx.adj_merge_remove_fwd(edge_type, peer_id, source_id.as_raw());
                ctx.adj_merge_add_rev(edge_type, target_id, peer_uid);
                ctx.adj_merge_add_fwd(edge_type, peer_id, target_id.as_raw());

                ctx.mvcc_move_edge_props(edge_type, peer_id, source_id, peer_id, target_id)?;
            }
        }
    }
    Ok(())
}

// =====================================================================
// ATTACH DOCUMENT (R168)
// =====================================================================

/// Execute an ATTACH DOCUMENT clause for each input row.
///
/// For each (source, target) row produced by the ATTACH pattern:
///  1. Verify the target property is absent (unless `on_conflict_replace`).
///  2. Read all properties from the source node.
///  3. Write them as a DOCUMENT into `target_property_path` on the target
///     via a `DocDelta::SetPath` merge operand (O(1) write, no read).
///  4. Delete the connecting edge (a → u): adj forward + reverse + edgeprop.
///  5. Optional `TRANSFER EDGES ON source TO target WHERE ...` — re-points
///     matching edges via posting-list merges (reuses R167 helper).
///  6. Cascade-delete remaining edges on the source node, unless
///     `on_remaining_fail` is true and any untransferred edges remain — in
///     which case abort with an error. Delete the source node record.
///
/// All writes land in the current MVCC transaction's buffer.
#[allow(clippy::too_many_arguments)]
fn execute_attach_document(
    input_rows: &[Row],
    source_variable: &str,
    target_variable: &str,
    edge_type: &str,
    edge_direction: crate::cypher::ast::EdgeFromSource,
    target_property_path: &[String],
    transfer: Option<&crate::cypher::ast::TransferEdgesSpec>,
    on_conflict_replace: bool,
    on_remaining_fail: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    if target_property_path.is_empty() {
        return Err(ExecutionError::Unsupported(
            "ATTACH DOCUMENT requires a non-empty target property path".to_string(),
        ));
    }

    let transfer_types: Option<Vec<String>> = match transfer {
        Some(t) => Some(extract_transfer_edge_types(&t.predicate)?),
        None => None,
    };

    // R172c Phase 3c: ATTACH DOCUMENT supports a temporal *target* —
    // the target's matched per-version record is read via its
    // `<target>.valid_from` binding and the property added via the
    // close+open dance (DocDelta::SetPath applied in-memory to the
    // new-version clone). Temporal *source* remains rejected: ATTACH
    // cascade-deletes the source, which on a temporal label needs the
    // positive-bitemporal-fact tombstone composition + cross-version
    // edge cleanup (Phase 4).
    for row in input_rows {
        if let Some(Value::String(primary)) = row.get(&format!("{source_variable}.__label__")) {
            if let Ok(Some(s)) = ctx.load_current_label_schema(primary) {
                if s.temporal {
                    return Err(ExecutionError::Unsupported(format!(
                        "ATTACH DOCUMENT on a temporal *source* (var \
                         `{source_variable}`, label '{primary}') is not yet \
                         supported: ATTACH cascade-deletes the source, which on \
                         a temporal label requires the positive-bitemporal-fact \
                         tombstone composition + cross-version edge cleanup \
                         (Phase 4). ATTACH onto a temporal *target* is supported."
                    )));
                }
            }
        }
    }

    let mut results: Vec<Row> = Vec::new();

    for input_row in input_rows {
        let source_id = match input_row.get(source_variable) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => {
                return Err(ExecutionError::Unsupported(format!(
                    "ATTACH DOCUMENT: variable `{source_variable}` is not a bound node"
                )));
            }
        };
        let target_id = match input_row.get(target_variable) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => {
                return Err(ExecutionError::Unsupported(format!(
                    "ATTACH DOCUMENT: variable `{target_variable}` is not a bound node"
                )));
            }
        };

        // Detect temporal target — affects how we read/write target's
        // record. Source is always non-temporal at this point (temporal
        // source was rejected at the pre-scan above).
        let target_is_temporal = match input_row.get(&format!("{target_variable}.__label__")) {
            Some(Value::String(primary)) => ctx
                .load_current_label_schema(primary)
                .ok()
                .flatten()
                .is_some_and(|s| s.temporal),
            _ => false,
        };

        // --- 1. Conflict check on the target property ---
        //
        // Non-temporal target: read the 16-byte node-key record.
        // Temporal target: read the matched per-version record at the
        // 25-byte key derived from `<target>.valid_from`.
        let target_valid_from: Option<i64> = if target_is_temporal {
            match input_row.get(&format!("{target_variable}.valid_from")) {
                Some(Value::Int(ms)) => Some(*ms),
                Some(Value::Timestamp(ms)) => Some(*ms),
                _ => {
                    return Err(ExecutionError::Unsupported(format!(
                        "ATTACH DOCUMENT on temporal target `{target_variable}`: \
                         matched row is missing `{target_variable}.valid_from`"
                    )));
                }
            }
        } else {
            None
        };
        let target_record = ctx
            .mvcc_get_node_either(ctx.shard_id, target_id, target_valid_from)?
            .ok_or_else(|| {
                ExecutionError::Unsupported(format!(
                    "ATTACH DOCUMENT: target node {target_id} not found"
                ))
            })?;
        if !on_conflict_replace
            && target_property_exists(&target_record, target_property_path, ctx.interner)
        {
            return Err(ExecutionError::Unsupported(format!(
                "ATTACH DOCUMENT: property `{}.{}` already exists (use ON CONFLICT REPLACE to overwrite)",
                target_variable,
                target_property_path.join(".")
            )));
        }

        // --- 2. Read source node (always non-temporal here) ---
        let source_record = ctx.mvcc_get_node(ctx.shard_id, source_id)?.ok_or_else(|| {
            ExecutionError::Unsupported(format!(
                "ATTACH DOCUMENT: source node {source_id} not found"
            ))
        })?;

        // --- 3. Package source properties as a DOCUMENT ---
        let doc = source_record_to_document(&source_record, ctx.interner);

        // Write the document at `target_property_path` on the target.
        //
        // Non-temporal target: queue a `DocDelta::SetPath` merge operand
        // (O(1) write, no read).
        // Temporal target: build the same SetPath delta, apply it
        // in-memory to a clone of the target record, then write
        // close-current at the matched key (valid_to = NOW) and
        // open-new at a fresh per-version key. Preserves history.
        if target_is_temporal {
            let Some(current_valid_from) = target_valid_from else {
                return Err(ExecutionError::Unsupported(format!(
                    "ATTACH DOCUMENT on temporal target `{target_variable}`: \
                     internal state missing valid_from"
                )));
            };
            let now_us = current_hlc_us() as i64;
            let new_valid_from = if now_us > current_valid_from {
                now_us
            } else {
                current_valid_from + 1
            };
            let mut closing_record = target_record.clone();
            let mut new_record = target_record.clone();

            // Build SetPath delta — same shape `emit_attach_set_path`
            // produces, but applied in-memory.
            use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};
            let first = &target_property_path[0];
            let field_id = ctx.interner.intern(first);
            let sub_path = target_property_path[1..].to_vec();
            let delta = DocDelta::SetPath {
                target: PathTarget::PropField(field_id),
                path: sub_path,
                value: doc,
            };
            coordinode_storage::engine::merge::apply_doc_deltas_to_record(
                &mut new_record,
                &[delta],
            );

            let vf_fid = ctx.interner.intern("valid_from");
            let vt_fid = ctx.interner.intern("valid_to");
            let its_fid = ctx.interner.intern("__ingestion_ts__");
            closing_record.set(vt_fid, Value::Int(new_valid_from));
            new_record.set(vf_fid, Value::Int(new_valid_from));
            new_record.props.remove(&vt_fid);
            new_record.set(its_fid, Value::Int(now_us));

            // Close-current at matched key + open-new at fresh per-
            // version key. `target_valid_from` is `Some(_)` on this
            // temporal branch by construction; surface a bug as an
            // error rather than panic.
            let current_vf = target_valid_from.ok_or_else(|| {
                ExecutionError::Unsupported(
                    "ATTACH temporal branch reached with valid_from=None — internal invariant violation"
                        .into(),
                )
            })?;
            ctx.mvcc_put_node_temporal(ctx.shard_id, target_id, current_vf, &closing_record)?;
            ctx.mvcc_put_node_temporal(ctx.shard_id, target_id, new_valid_from, &new_record)?;
            ctx.write_stats.properties_set += 1;
        } else {
            emit_attach_set_path(target_id, target_property_path, doc, ctx)?;
        }

        // --- 4. Delete the connecting edge (both directions) ---
        let (edge_src, edge_tgt) = match edge_direction {
            crate::cypher::ast::EdgeFromSource::Outgoing => (source_id, target_id),
            crate::cypher::ast::EdgeFromSource::Incoming => (target_id, source_id),
        };
        delete_single_edge(edge_src, edge_tgt, edge_type, ctx)?;

        // --- 5. Optional TRANSFER EDGES (before we delete source) ---
        if let Some(ref types) = transfer_types {
            transfer_edges_on_node(source_id, target_id, types, ctx)?;
        }

        // --- 6. Delete the source node (cascade or fail) ---
        cascade_delete_source_node(source_id, on_remaining_fail, ctx)?;

        // --- 7. Emit output row: preserve bindings, drop stale source row cols ---
        let mut out = input_row.clone();
        // Source is gone — remove its binding so downstream clauses cannot
        // reference deleted data.
        out.remove(source_variable);
        // Target is still live; its property bindings may be stale.
        out.remove(&format!(
            "{target_variable}.{}",
            target_property_path.join(".")
        ));
        results.push(out);
    }

    Ok(results)
}

/// Check whether a property path is already present on a node record.
///
/// For single-segment paths: looks in `props` (interned) or `extra` map.
/// For multi-segment paths: navigates into the nested Document/Map.
fn target_property_exists(record: &NodeRecord, path: &[String], interner: &FieldInterner) -> bool {
    let first = &path[0];
    let root: Option<rmpv::Value> = match interner.lookup(first) {
        Some(fid) => record.props.get(&fid).map(value_to_rmpv),
        None => record
            .extra
            .as_ref()
            .and_then(|m| m.get(first))
            .map(value_to_rmpv),
    };
    let Some(mut current) = root else {
        return false;
    };
    for seg in &path[1..] {
        current = match current {
            rmpv::Value::Map(entries) => {
                let hit = entries
                    .into_iter()
                    .find(|(k, _)| k.as_str() == Some(seg.as_str()))
                    .map(|(_, v)| v);
                match hit {
                    Some(v) => v,
                    None => return false,
                }
            }
            _ => return false,
        };
    }
    // Final value must be non-nil to count as "exists".
    !matches!(current, rmpv::Value::Nil)
}

/// Package a source node's properties into a single `rmpv::Value::Map` for
/// nesting into a target node. Interned props contribute their resolved
/// string names; `extra` entries contribute their string keys verbatim.
fn source_record_to_document(record: &NodeRecord, interner: &FieldInterner) -> rmpv::Value {
    let mut entries: Vec<(rmpv::Value, rmpv::Value)> = Vec::new();
    for (&fid, value) in &record.props {
        if let Some(name) = interner.resolve(fid) {
            entries.push((rmpv::Value::String(name.into()), value_to_rmpv(value)));
        }
    }
    if let Some(extra) = record.extra.as_ref() {
        for (name, value) in extra {
            entries.push((
                rmpv::Value::String(name.clone().into()),
                value_to_rmpv(value),
            ));
        }
    }
    rmpv::Value::Map(entries)
}

/// Emit a `DocDelta::SetPath` merge operand placing `doc` at
/// `target_property_path` on `target_id`.
fn emit_attach_set_path(
    target_id: NodeId,
    path: &[String],
    doc: rmpv::Value,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};

    // First segment → interned field id (matches the write-path used by
    // `SET n.address.x = y`).
    let first = &path[0];
    let field_id = ctx.interner.intern(first);
    let sub: Vec<String> = path[1..].to_vec();

    let delta = DocDelta::SetPath {
        target: PathTarget::PropField(field_id),
        path: sub,
        value: doc,
    };
    let operand = delta
        .encode()
        .map_err(|e| ExecutionError::Serialization(format!("DocDelta encode: {e}")))?;
    ctx.mvcc_merge_node_delta(ctx.shard_id, target_id, operand);
    ctx.write_stats.properties_set += 1;
    Ok(())
}

/// Delete one edge `(src → tgt)` of `edge_type`.
///
/// Non-temporal: issue merge_remove on both adjacency halves and delete the
/// single edgeprop entry.
///
/// Temporal: `DELETE r` is a hard delete of the logical edge — every version
/// of the pair is removed, then adj-posting is cleared. The "soft-close one
/// version" workflow is `SET r.valid_to = <now>`, not DELETE. Hard deletes are
/// rare (GDPR erase, mistaken insert) but must be supported.
fn delete_single_edge(
    src: NodeId,
    tgt: NodeId,
    edge_type: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    let is_temporal = lookup_edge_type_temporal(edge_type, ctx)?;

    // Probe BEFORE COMMIT DELETE triggers for this edge type up front. If at
    // least one trigger is registered, snapshot every version's property map
    // before deletion so the trigger body can read `$before`. For temporal
    // edges this is one snapshot per version (same number of trigger firings);
    // for non-temporal it is the single edgeprop entry.
    let target_segment =
        coordinode_core::schema::triggers::TriggerTargetSchema::edge_type(edge_type)
            .index_key_segment();
    let matched_delete = ctx.lookup_matching_triggers(&target_segment, "d")?;
    let mut delete_snapshots: Vec<std::collections::BTreeMap<String, Value>> = Vec::new();
    if !matched_delete.is_empty() {
        if is_temporal {
            for (_vf, bytes) in ctx.mvcc_scan_edge_prop_version_bytes(edge_type, src, tgt, None)? {
                delete_snapshots.push(decode_edgeprop_into_map(&bytes, ctx));
            }
        } else if let Some(prop_map) = ctx.mvcc_get_edge_props(edge_type, src, tgt)? {
            delete_snapshots.push(decode_edgeprop_map_into_named(&prop_map, ctx));
        } else {
            // The pair has no edgeprop entry (e.g. propertyless edge);
            // still fire once with an empty `$before` map so the trigger
            // observes the deletion event.
            delete_snapshots.push(std::collections::BTreeMap::new());
        }
    }

    if is_temporal {
        ctx.mvcc_delete_all_edge_prop_versions(edge_type, src, tgt)?;
    } else {
        ctx.mvcc_delete_edge_props(edge_type, src, tgt)?;
    }

    // Adj-posting tracks pair existence, not version count, so once all
    // versions are gone we clear it. For non-temporal this is the only
    // version, so the post-delete count is trivially zero.
    let remaining = if is_temporal {
        temporal_pair_remaining_versions(edge_type, src, tgt, ctx)?
    } else {
        0
    };
    if remaining == 0 {
        ctx.adj_merge_remove_fwd(edge_type, src, tgt.as_raw());
        ctx.adj_merge_remove_rev(edge_type, tgt, src.as_raw());
    }

    ctx.write_stats.edges_deleted += 1;

    // Fire BEFORE COMMIT DELETE triggers AFTER the deletion so the trigger
    // body's MATCH against the edge returns empty via RYOW (correct semantics:
    // the edge is gone for any read inside the trigger). One firing per
    // snapshotted version, with `$before` carrying that version's properties.
    if !matched_delete.is_empty() {
        for before in &delete_snapshots {
            let trigger_params = trigger_params_for_edge_delete(edge_type, src, tgt, before);
            fire_before_commit_triggers(&matched_delete, &trigger_params, ctx)?;
        }
    }

    Ok(())
}

/// Delete the source node and its remaining edges. If `on_remaining_fail` is
/// true and the node still has any edges (after TRANSFER), error out.
fn cascade_delete_source_node(
    source_id: NodeId,
    on_remaining_fail: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    let edge_types = ctx.list_edge_types()?;
    let mut remaining_edges: Vec<AdjKeyParts> = Vec::new();
    let mut remaining_count: u64 = 0;

    for et in &edge_types {
        for direction in [AdjDirection::Out, AdjDirection::In] {
            let present = match direction {
                AdjDirection::Out => ctx.adj_get_fwd(et, source_id)?,
                AdjDirection::In => ctx.adj_get_rev(et, source_id)?,
            }
            .is_some();
            if present {
                remaining_count += 1;
                remaining_edges.push(AdjKeyParts {
                    edge_type: et.clone(),
                    direction,
                    node_id: source_id,
                });
            }
        }
    }

    if on_remaining_fail && remaining_count > 0 {
        return Err(ExecutionError::Unsupported(format!(
            "ATTACH DOCUMENT with ON REMAINING FAIL: source node {source_id} \
             still has {remaining_count} untransferred edge(s)"
        )));
    }

    // Cascade-delete: mirror `execute_delete` DETACH path logic, including
    // BEFORE COMMIT DELETE trigger firings for each cascaded edge and for
    // the source node itself. Without this, ATTACH DOCUMENT silently bypasses
    // any DELETE trigger registered on the source's label or on its connected
    // edges' types — asymmetric with DETACH DELETE and a real correctness
    // gap for audit/compliance workloads.
    for parts in &remaining_edges {
        {
            let edge_target_segment =
                coordinode_core::schema::triggers::TriggerTargetSchema::edge_type(&parts.edge_type)
                    .index_key_segment();
            let matched_edge_delete = ctx.lookup_matching_triggers(&edge_target_segment, "d")?;
            let is_temporal = lookup_edge_type_temporal(&parts.edge_type, ctx)?;

            let plist = match parts.direction {
                AdjDirection::Out => ctx.adj_get_fwd(&parts.edge_type, parts.node_id)?,
                AdjDirection::In => ctx.adj_get_rev(&parts.edge_type, parts.node_id)?,
            };
            if let Some(plist) = plist {
                for peer_uid in plist.iter() {
                    let peer_id = NodeId::from_raw(peer_uid);
                    match parts.direction {
                        AdjDirection::Out => {
                            ctx.adj_merge_remove_rev(&parts.edge_type, peer_id, source_id.as_raw())
                        }
                        AdjDirection::In => {
                            ctx.adj_merge_remove_fwd(&parts.edge_type, peer_id, source_id.as_raw())
                        }
                    };

                    let (ep_src, ep_tgt) = match parts.direction {
                        AdjDirection::Out => (source_id, peer_id),
                        AdjDirection::In => (peer_id, source_id),
                    };

                    // Pre-snapshot edge props for trigger firing — same as
                    // DETACH DELETE: captured before mvcc_delete so the body's
                    // RYOW reads see the edge as deleted.
                    let mut edge_delete_snapshots: Vec<std::collections::BTreeMap<String, Value>> =
                        Vec::new();
                    if !matched_edge_delete.is_empty() {
                        if is_temporal {
                            for (_vf, bytes) in ctx.mvcc_scan_edge_prop_version_bytes(
                                &parts.edge_type,
                                ep_src,
                                ep_tgt,
                                None,
                            )? {
                                edge_delete_snapshots.push(decode_edgeprop_into_map(&bytes, ctx));
                            }
                        } else if let Some(prop_map) =
                            ctx.mvcc_get_edge_props(&parts.edge_type, ep_src, ep_tgt)?
                        {
                            edge_delete_snapshots
                                .push(decode_edgeprop_map_into_named(&prop_map, ctx));
                        } else {
                            edge_delete_snapshots.push(std::collections::BTreeMap::new());
                        }
                    }

                    if is_temporal {
                        ctx.mvcc_delete_all_edge_prop_versions(&parts.edge_type, ep_src, ep_tgt)?;
                    } else {
                        ctx.mvcc_delete_edge_props(&parts.edge_type, ep_src, ep_tgt)?;
                    }

                    if !matched_edge_delete.is_empty() {
                        for before in &edge_delete_snapshots {
                            let trigger_params = trigger_params_for_edge_delete(
                                &parts.edge_type,
                                ep_src,
                                ep_tgt,
                                before,
                            );
                            fire_before_commit_triggers(
                                &matched_edge_delete,
                                &trigger_params,
                                ctx,
                            )?;
                        }
                    }
                }
            }
        }
        ctx.mvcc_purge_adj(&parts.edge_type, parts.node_id, parts.direction)?;
        ctx.write_stats.edges_deleted += 1;
    }

    // Delete the node record itself (B-tree / vector / text indexes left to
    // the standard delete path; ATTACH DOCUMENT's source is typically a
    // short-lived node so we keep this tight — cleanup mirrors DETACH DELETE
    // behaviour in `execute_delete`).
    // Snapshot the pre-mutation node record once: re-used for both the
    // BEFORE COMMIT DELETE trigger firing and (where present) index cleanup.
    let pre_snapshot: Option<(Vec<String>, std::collections::BTreeMap<String, Value>)> = ctx
        .mvcc_get_node(ctx.shard_id, source_id)?
        .map(|rec| snapshot_node_record(&rec, ctx));
    let needs_index_cleanup = ctx.btree_index_registry.is_some()
        || ctx.vector_index_registry.is_some()
        || ctx.text_index_registry.is_some();
    if needs_index_cleanup {
        if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, source_id)? {
            {
                let label = record.primary_label().to_string();
                if let Some(btree_reg) = ctx.btree_index_registry {
                    let props: Vec<(String, Value)> = record
                        .props
                        .iter()
                        .filter_map(|(&fid, v)| {
                            ctx.interner
                                .resolve(fid)
                                .map(|name| (name.to_string(), v.clone()))
                        })
                        .collect();
                    btree_reg
                        .on_node_deleted(ctx.engine, source_id, &label, &props)
                        .map_err(ExecutionError::Storage)?;
                }
                for (&fid, value) in &record.props {
                    if let Some(prop_name) = ctx.interner.resolve(fid) {
                        if let Some(registry) = ctx.vector_index_registry {
                            if try_extract_vector(value).is_some() {
                                registry.on_vector_deleted(&label, source_id, prop_name);
                            }
                        }
                        if let Some(registry) = ctx.text_index_registry {
                            if value.as_str().is_some() {
                                registry.on_text_deleted(&label, source_id, prop_name);
                            }
                        }
                    }
                }
            }
        }
    }
    ctx.mvcc_delete_node(ctx.shard_id, source_id)?;
    ctx.write_stats.nodes_deleted += 1;

    // Fire BEFORE COMMIT DELETE triggers on the source node's labels —
    // mirrors the DETACH DELETE behaviour in `execute_delete`. Probe runs
    // AFTER mvcc_delete so the trigger body's MATCH against the deleted
    // node returns empty via RYOW.
    if let Some((labels, before_props)) = pre_snapshot {
        let trigger_params = trigger_params_for_node_delete(source_id, &before_props);
        for label in &labels {
            let target_segment =
                coordinode_core::schema::triggers::TriggerTargetSchema::label(label.clone())
                    .index_key_segment();
            let matched = ctx.lookup_matching_triggers(&target_segment, "d")?;
            if !matched.is_empty() {
                fire_before_commit_triggers(&matched, &trigger_params, ctx)?;
            }
        }
    }
    Ok(())
}

/// Load the current `LabelSchema` for `name` via the schema partition pointer
/// (`schema:current_revision:label:<name>` → `schema:label:<name>:<revision>`),
/// directly through a `StorageEngine` handle without MVCC visibility.
///
/// Used by code paths that only have engine access (parallel scans, background
/// helpers) and don't need RYOW semantics. MVCC-aware callers should use
/// [`ExecutionContext::load_current_label_schema`] instead.
///
/// Returns `None` for missing pointer, missing schema body, corrupt pointer,
/// or msgpack decode error — never errors. This is a best-effort read used
/// for non-blocking schema enrichment (computed properties, vector index
/// dispatch). Hard failures show up at the next MVCC read.
fn load_current_label_schema_from_engine(
    engine: &StorageEngine,
    name: &str,
) -> Option<LabelSchema> {
    use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
    // Best-effort engine-direct (untracked) read: any error → None.
    LocalSchemaStore::new(engine)
        .load_label(name)
        .ok()
        .flatten()
}

/// Build the Schema-partition key for a given edge type name (test fixtures
/// that plant/inspect raw edge-type markers). Production code goes through
/// [`coordinode_modality::SchemaStore`].
///
/// Format: `schema:edge_type:<name>:1`
#[cfg(test)]
fn edge_type_schema_key(edge_type: &str) -> Vec<u8> {
    coordinode_core::schema::definition::encode_edge_type_schema_key(edge_type, 1)
}

/// Look up whether an edge type was declared with the TEMPORAL modifier.
///
/// Reads `schema:edge_type:<name>` and attempts to decode the body as an
/// `EdgeTypeSchema`. A zero-length body is a legacy idempotent marker written
/// on first CREATE for the type (predates DDL) — treated as non-temporal.
/// Missing key also returns `false`: an edge type used without a prior
/// `CREATE EDGE TYPE` is implicitly non-temporal.
fn lookup_edge_type_temporal(
    edge_type: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<bool, ExecutionError> {
    Ok(ctx
        .load_current_edge_type_schema(edge_type)?
        .map(|s| s.temporal)
        .unwrap_or(false))
}

/// Apply `SET r.<property> = <value>` to the matched edge.
///
/// Locates the edgeprop entry via the hidden `__src__` / `__tgt__` row columns
/// written by `build_target_rows`. For temporal edges, also reads
/// `<ev>.valid_from` from the row so the per-version key resolves to the
/// SAME row that was matched (no new version is created). Reads the current
/// MessagePack-encoded property map, replaces or inserts the named field,
/// writes it back.
fn update_edge_property(
    edge_variable: &str,
    edge_type: &str,
    property: &str,
    value: Value,
    row: &mut Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    let src_raw = match row.get(&format!("{edge_variable}.__src__")) {
        Some(Value::Int(n)) => *n as u64,
        _ => return Ok(()),
    };
    let tgt_raw = match row.get(&format!("{edge_variable}.__tgt__")) {
        Some(Value::Int(n)) => *n as u64,
        _ => return Ok(()),
    };
    let src = NodeId::from_raw(src_raw);
    let tgt = NodeId::from_raw(tgt_raw);

    let is_temporal = lookup_edge_type_temporal(edge_type, ctx)?;

    // Reject mutation of key-immutable fields and reserved metadata columns.
    // valid_from is part of the storage key — rewriting it would either
    // create a phantom version (insert at new key without removing the old)
    // or corrupt the value/key invariant. Workflow: DELETE + CREATE.
    if is_temporal && property == "valid_from" {
        return Err(ExecutionError::Unsupported(format!(
            "SET {edge_variable}.valid_from is not allowed on temporal edges: \
             valid_from is part of the storage key. DELETE the version and \
             CREATE a new one with the updated timestamp."
        )));
    }
    if matches!(property, "__src__" | "__tgt__" | "__type__") {
        return Err(ExecutionError::Unsupported(format!(
            "SET {edge_variable}.{property} is reserved: '{property}' is \
             engine-internal row metadata and cannot be assigned"
        )));
    }

    let valid_from_for_key: Option<i64> = if is_temporal {
        match row.get(&format!("{edge_variable}.valid_from")) {
            Some(Value::Int(ms)) => Some(*ms),
            _ => {
                return Err(ExecutionError::Unsupported(format!(
                    "SET on temporal edge '{edge_variable}': matched row is missing valid_from"
                )));
            }
        }
    } else {
        None
    };

    let mut prop_map = ctx
        .mvcc_get_edge_props_either(edge_type, src, tgt, valid_from_for_key)?
        .unwrap_or_default();
    let field_id = ctx.interner.intern(property);
    let mut replaced = false;
    for entry in &mut prop_map {
        if entry.0 == field_id {
            entry.1 = value.clone();
            replaced = true;
            break;
        }
    }
    if !replaced {
        prop_map.push((field_id, value));
    }
    ctx.mvcc_put_edge_props_either(edge_type, src, tgt, valid_from_for_key, &prop_map)?;
    ctx.write_stats.properties_set += 1;
    Ok(())
}

/// Count remaining edgeprop versions for a temporal `(type, src, tgt)` pair.
///
/// Used after a temporal DELETE to decide whether the adj-posting forward and
/// reverse entries for the pair must also be removed. While at least one
/// version of the edge exists between `src` and `tgt`, the adj-posting entry
/// stays. When the count drops to zero the caller fires `adj_merge_remove`
/// on both directions; otherwise it leaves adj-posting alone.
///
/// `mvcc_prefix_scan` returns snapshot entries even when a tombstone exists
/// in the write buffer for the same key (the buffer filter only adds `Some(_)`
/// writes). To get an accurate post-delete count within a single transaction
/// we walk the scan result and subtract any key that has a `None` tombstone
/// in the write buffer.
pub(crate) fn temporal_pair_remaining_versions(
    edge_type: &str,
    source_id: NodeId,
    target_id: NodeId,
    ctx: &mut ExecutionContext<'_>,
) -> Result<usize, ExecutionError> {
    use coordinode_modality::{EdgeStore as _, LocalEdgeStore};
    ctx.sync_txn_state();
    Ok(LocalEdgeStore.count_live_versions(&mut ctx.txn, edge_type, source_id, target_id)?)
}

/// Test-only helper preserved for legacy tests that build edgeprop
/// keys directly. Production code uses
/// [`ExecutionContext::mvcc_put_edge_props_either`] /
/// [`ExecutionContext::mvcc_get_edge_props_either`] instead.
#[cfg(test)]
pub(crate) fn edgeprop_write_key(
    edge_type: &str,
    source_id: NodeId,
    target_id: NodeId,
    valid_from_ms: Option<i64>,
) -> Vec<u8> {
    match valid_from_ms {
        Some(vf) => coordinode_core::graph::edge::encode_temporal_edgeprop_key(
            edge_type, source_id, target_id, vf,
        ),
        None => coordinode_core::graph::edge::encode_edgeprop_key(edge_type, source_id, target_id),
    }
}

/// Extract node ID from a node key (after the "node:XXXX:" prefix).
/// Inject COMPUTED property values into a row for a given node.
///
/// Loads the label schema, finds COMPUTED properties, evaluates each using
/// the node's anchor field value and current time. Injected values appear
/// in the row as regular `{variable}.{property}` entries.
///
/// ~10ns per computed field (formula evaluation only; schema lookup amortized).
fn inject_computed_properties(
    row: &mut Row,
    variable: &str,
    label: &str,
    ctx: &ExecutionContext<'_>,
) {
    inject_computed_from_engine(row, variable, label, ctx.engine);
}

/// Inject COMPUTED properties using direct engine access (no ExecutionContext needed).
/// Used by both the sequential path (via inject_computed_properties) and the
/// parallel path (process_targets_parallel).
fn inject_computed_from_engine(row: &mut Row, variable: &str, label: &str, engine: &StorageEngine) {
    let schema = match load_current_label_schema_from_engine(engine, label) {
        Some(s) => s,
        None => return,
    };

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);

    for (prop_name, prop_def) in &schema.properties {
        let spec = match &prop_def.property_type {
            coordinode_core::schema::definition::PropertyType::Computed(s) => s,
            _ => continue,
        };
        let val = evaluate_computed_spec(spec, variable, row, now_us);
        let col_name = format!("{variable}.{prop_name}");
        row.insert(col_name, val);
    }
}

/// Evaluate a single ComputedSpec to produce a Value.
fn evaluate_computed_spec(
    spec: &coordinode_core::schema::computed::ComputedSpec,
    variable: &str,
    row: &Row,
    now_us: i64,
) -> Value {
    use coordinode_core::schema::computed::ComputedSpec;

    let anchor_field = spec.anchor_field();
    let anchor_key = format!("{variable}.{anchor_field}");
    // Accept both Timestamp and Int as anchor values (Cypher literals are Int).
    let anchor_us = match row.get(&anchor_key) {
        Some(Value::Timestamp(ts)) => *ts,
        Some(Value::Int(ts)) => *ts,
        _ => return Value::Null, // anchor field missing → cannot compute
    };

    let elapsed_secs = ((now_us - anchor_us).max(0) as f64) / 1_000_000.0;

    match spec {
        ComputedSpec::Decay {
            formula,
            initial,
            target,
            duration_secs,
            ..
        } => {
            if *duration_secs == 0 {
                return Value::Float(*target);
            }
            let t = (elapsed_secs / *duration_secs as f64).min(1.0);
            let weight = formula.evaluate(t);
            // weight = 1.0 at t=0 (fresh) → value = initial
            // weight = 0.0 at t=1 (decayed) → value = target
            Value::Float(*initial * weight + *target * (1.0 - weight))
        }
        ComputedSpec::Ttl { duration_secs, .. } => {
            let remaining = *duration_secs as f64 - elapsed_secs;
            if remaining <= 0.0 {
                Value::Null // expired → triggers background cleanup
            } else {
                Value::Int(remaining as i64)
            }
        }
        ComputedSpec::VectorDecay {
            formula,
            duration_secs,
            ..
        } => {
            if *duration_secs == 0 {
                return Value::Float(0.0);
            }
            let t = (elapsed_secs / *duration_secs as f64).min(1.0);
            Value::Float(formula.evaluate(t))
        }
    }
}

fn decode_node_id_from_key(key: &[u8]) -> u64 {
    // Key format: "node:" (5) + shard_id BE (2) + ":" (1) + node_id BE (8)
    if key.len() >= 16 {
        let id_bytes = &key[8..16];
        u64::from_be_bytes([
            id_bytes[0],
            id_bytes[1],
            id_bytes[2],
            id_bytes[3],
            id_bytes[4],
            id_bytes[5],
            id_bytes[6],
            id_bytes[7],
        ])
    } else {
        0
    }
}

/// Generate a display name for an expression (used as column name when no alias).
fn expr_display_name(expr: &Expr) -> String {
    match expr {
        Expr::Variable(name) => name.clone(),
        Expr::PropertyAccess { expr, property } => {
            // Recursively build dotted name for multi-level access:
            // n.config.network.ssid → "n.config.network.ssid"
            let parent = expr_display_name(expr);
            format!("{parent}.{property}")
        }
        Expr::FunctionCall { name, .. } => name.clone(),
        Expr::Star => "*".to_string(),
        _ => format!("{expr:?}"),
    }
}

/// Execute ALTER LABEL: load schema, change mode, persist.
///
/// Returns a single row with the label name, new mode, and version.
fn execute_alter_label(
    label: &str,
    mode_str: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let mode = match mode_str {
        "strict" => SchemaMode::Strict,
        "validated" => SchemaMode::Validated,
        "flexible" => SchemaMode::Flexible,
        other => {
            return Err(ExecutionError::Unsupported(format!(
                "unknown schema mode '{other}'. Valid: STRICT, VALIDATED, FLEXIBLE"
            )));
        }
    };

    // Load existing schema via pointer or create a new one.
    let mut schema = ctx
        .load_current_label_schema(label)?
        .unwrap_or_else(|| LabelSchema::new_node_id(label));

    schema.set_mode(mode);
    schema.schema_revision += 1;

    // Persist new version + pointer atomically via save helper.
    ctx.save_current_label_schema(&schema)?;

    // Return result row with label info.
    let mut row = Row::new();
    row.insert("label".to_string(), Value::String(label.to_string()));
    row.insert("mode".to_string(), Value::String(format!("{mode}")));
    row.insert(
        "version".to_string(),
        Value::Int(schema.schema_revision as i64),
    );
    Ok(vec![row])
}

/// Execute CREATE EDGE TYPE: register an edge-type schema in the Schema partition.
///
/// Persists an `EdgeTypeSchema` keyed by `schema:edge_type:<name>` via the MVCC
/// write buffer. Subsequent edge writes that name this type can be validated
/// against the declared properties; if `temporal == true`, the write path
/// requires `valid_from` and stores per-version edgeprop entries.
///
/// Returns one row: `{ name, temporal, version, properties }`.
/// Execute `CREATE NODE TYPE <name> [TEMPORAL] [WITH (...)]` (R172a per
/// ADR-027). Mirror of `execute_create_edge_type` for node labels.
///
/// Persists a new `LabelSchema` with the bitemporal flag set as declared and
/// the user-supplied property declarations. Rejects:
///   - Duplicate label (label already has a current-revision pointer)
///   - Reserved engine-internal property names (`__ingestion_ts__`,
///     `valid_from`, `valid_to`, `__src__`, `__tgt__`, `__type__`) — these
///     are populated/owned by the engine; user declarations would shadow them
///   - Unsupported property type spellings
///
/// The TEMPORAL flag is immutable from this point forward: changing it
/// requires creating a new label type and copying data (per ADR-027).
fn execute_create_node_type(
    name: &str,
    temporal: bool,
    properties: &[crate::cypher::ast::EdgePropertyDecl],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_core::schema::definition::{
        LabelSchema, PlacementPolicy, PropertyDef, PropertyType,
    };

    // Reject if the label was registered before (either via explicit DDL or
    // via implicit label creation on first node write — both write a current-
    // revision pointer at schema:current_revision:label:<name>).
    if ctx.load_current_label_schema(name)?.is_some() {
        return Err(ExecutionError::Unsupported(format!(
            "label '{name}' already exists"
        )));
    }

    // Reject reserved engine-internal property names. `__ingestion_ts__`
    // (HLC commit-ts on temporal labels) is fully engine-owned — declaring
    // it would shadow the canonical engine value. `__src__` / `__tgt__` /
    // `__type__` are edge-row metadata and have no meaning on a node label;
    // rejecting them keeps the reserved-name surface symmetric and prevents
    // accidental confusion. `valid_from` / `valid_to` are NOT in this list
    // by design: they are user-supplied bitemporal interval fields on
    // temporal labels (see arch/core/temporal-edges.md) — engine validates
    // their type and invariants at write time but does not own the values.
    for decl in properties {
        if matches!(
            decl.name.as_str(),
            "__ingestion_ts__" | "__src__" | "__tgt__" | "__type__"
        ) {
            return Err(ExecutionError::Unsupported(format!(
                "property name '{}' is reserved for engine-internal use and \
                 cannot be declared in CREATE NODE TYPE",
                decl.name
            )));
        }
    }

    // CE default placement is `NodeId`. EE DDL surface for non-NodeId
    // placement is `SHARD BY HASH(...)` / `SHARD BY RANGE(...)` on a
    // separate clause; CREATE NODE TYPE does not currently accept inline
    // placement specs.
    let mut schema = LabelSchema::new(name, PlacementPolicy::NodeId);
    schema.set_temporal(temporal);

    for decl in properties {
        let ptype = match decl.type_name.as_str() {
            "STRING" => PropertyType::String,
            "INT" => PropertyType::Int,
            "FLOAT" => PropertyType::Float,
            "BOOL" => PropertyType::Bool,
            "TIMESTAMP" => PropertyType::Timestamp,
            "BLOB" => PropertyType::Blob,
            "MAP" => PropertyType::Map,
            "GEO" => PropertyType::Geo,
            "BINARY" => PropertyType::Binary,
            "DOCUMENT" => PropertyType::Document,
            other => {
                return Err(ExecutionError::Unsupported(format!(
                    "unsupported property type '{other}' for label '{name}'"
                )));
            }
        };
        let mut prop = PropertyDef::new(&decl.name, ptype);
        if decl.not_null {
            prop = prop.not_null();
        }
        schema.add_property(prop);
    }

    // Persist new schema + current-revision pointer atomically.
    ctx.save_current_label_schema(&schema)?;

    let mut row = Row::new();
    row.insert("name".to_string(), Value::String(name.to_string()));
    row.insert("temporal".to_string(), Value::Bool(temporal));
    row.insert(
        "revision".to_string(),
        Value::Int(schema.schema_revision as i64),
    );
    row.insert(
        "properties".to_string(),
        Value::Int(properties.len() as i64),
    );
    Ok(vec![row])
}

fn execute_create_edge_type(
    name: &str,
    temporal: bool,
    properties: &[crate::cypher::ast::EdgePropertyDecl],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Reject if the edge type was registered before (either via explicit DDL,
    // which writes a pointer, or implicitly by a prior edge create, which
    // writes a revisioned existence marker). Probe both:
    //  1. current_revision pointer → set by explicit CREATE EDGE TYPE
    //  2. legacy unparametrised existence marker at revision 1 → set by
    //     implicit edge registration in `create_edges_for_correlated_row`.
    if ctx.mvcc_edge_type_exists(name)? {
        return Err(ExecutionError::Unsupported(format!(
            "edge type '{name}' already exists"
        )));
    }

    let mut schema = EdgeTypeSchema::new(name);
    schema.set_temporal(temporal);

    for decl in properties {
        let ptype = match decl.type_name.as_str() {
            "STRING" => PropertyType::String,
            "INT" => PropertyType::Int,
            "FLOAT" => PropertyType::Float,
            "BOOL" => PropertyType::Bool,
            "TIMESTAMP" => PropertyType::Timestamp,
            "BLOB" => PropertyType::Blob,
            "MAP" => PropertyType::Map,
            "GEO" => PropertyType::Geo,
            "BINARY" => PropertyType::Binary,
            "DOCUMENT" => PropertyType::Document,
            other => {
                return Err(ExecutionError::Unsupported(format!(
                    "unsupported property type '{other}' for edge type '{name}'"
                )));
            }
        };
        let mut prop = PropertyDef::new(&decl.name, ptype);
        if decl.not_null {
            prop = prop.not_null();
        }
        schema.add_property(prop);
    }

    // Persist new version + pointer atomically.
    ctx.save_current_edge_type_schema(&schema)?;

    let mut row = Row::new();
    row.insert("name".to_string(), Value::String(name.to_string()));
    row.insert("temporal".to_string(), Value::Bool(temporal));
    row.insert(
        "version".to_string(),
        Value::Int(schema.schema_revision as i64),
    );
    row.insert(
        "properties".to_string(),
        Value::Int(properties.len() as i64),
    );
    Ok(vec![row])
}

// ======================================================================
// the trigger architecture: Trigger DDL executors
// ======================================================================

fn current_hlc_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Translate parser AST → persisted schema form (no logic difference, just
/// a layering boundary — the parser's `cypher::ast` types live in
/// coordinode-query; storage uses coordinode-core types).
fn trigger_target_to_schema(
    t: &crate::cypher::ast::TriggerTarget,
) -> coordinode_core::schema::triggers::TriggerTargetSchema {
    use crate::cypher::ast::TriggerTarget as A;
    use coordinode_core::schema::triggers::TriggerTargetSchema as S;
    match t {
        A::Label(name) => S::label(name),
        A::EdgeType(name) => S::edge_type(name),
    }
}

fn trigger_events_to_schema(
    e: crate::cypher::ast::TriggerEvents,
) -> coordinode_core::schema::triggers::TriggerEventsSchema {
    coordinode_core::schema::triggers::TriggerEventsSchema {
        on_create: e.on_create,
        on_update: e.on_update,
        on_delete: e.on_delete,
    }
}

fn trigger_timing_to_schema(
    t: crate::cypher::ast::TriggerTiming,
) -> coordinode_core::schema::triggers::TriggerTimingSchema {
    use crate::cypher::ast::TriggerTiming as A;
    use coordinode_core::schema::triggers::TriggerTimingSchema as S;
    match t {
        A::BeforeCommit => S::BeforeCommit,
        A::AfterCommit => S::AfterCommit,
    }
}

fn on_error_to_schema(
    p: &crate::cypher::ast::OnErrorPolicy,
) -> coordinode_core::schema::triggers::OnErrorPolicySchema {
    use crate::cypher::ast::OnErrorPolicy as A;
    use coordinode_core::schema::triggers::OnErrorPolicySchema as S;
    match p {
        A::Propagate => S::Propagate,
        A::Retry { n, backoff_ms } => S::Retry {
            n: *n,
            backoff_ms: *backoff_ms,
        },
        A::DeadLetter => S::DeadLetter,
    }
}

/// Load the existing trigger-name list for an index key, append `name`, and
/// persist it back. Idempotent: re-adding an already-present name is a no-op.
fn append_to_trigger_index(
    ctx: &mut ExecutionContext<'_>,
    target_segment: &str,
    event_segment: &str,
    name: &str,
) -> Result<(), ExecutionError> {
    use coordinode_modality::{LocalTriggerStore, TriggerStore as _};
    ctx.sync_txn_state();
    let mut names: Vec<String> =
        match LocalTriggerStore.get_index(&mut ctx.txn, target_segment, event_segment)? {
            Some(bytes) => rmp_serde::from_slice(&bytes).map_err(|e| {
                ExecutionError::Serialization(format!(
                    "trigger_index decode for {target_segment}/{event_segment}: {e}"
                ))
            })?,
            None => Vec::new(),
        };
    if !names.iter().any(|n| n == name) {
        names.push(name.to_string());
        let bytes = rmp_serde::to_vec(&names)
            .map_err(|e| ExecutionError::Serialization(format!("trigger_index encode: {e}")))?;
        LocalTriggerStore.put_index(&mut ctx.txn, target_segment, event_segment, &bytes)?;
    }
    Ok(())
}

/// Remove `name` from the index entry; if the list becomes empty, delete the key.
fn remove_from_trigger_index(
    ctx: &mut ExecutionContext<'_>,
    target_segment: &str,
    event_segment: &str,
    name: &str,
) -> Result<(), ExecutionError> {
    use coordinode_modality::{LocalTriggerStore, TriggerStore as _};
    ctx.sync_txn_state();
    let mut names: Vec<String> =
        match LocalTriggerStore.get_index(&mut ctx.txn, target_segment, event_segment)? {
            Some(bytes) => rmp_serde::from_slice(&bytes)
                .map_err(|e| ExecutionError::Serialization(format!("trigger_index decode: {e}")))?,
            None => return Ok(()),
        };
    let before = names.len();
    names.retain(|n| n != name);
    if names.len() == before {
        return Ok(());
    }
    if names.is_empty() {
        LocalTriggerStore.delete_index(&mut ctx.txn, target_segment, event_segment)?;
    } else {
        let bytes = rmp_serde::to_vec(&names)
            .map_err(|e| ExecutionError::Serialization(format!("trigger_index encode: {e}")))?;
        LocalTriggerStore.put_index(&mut ctx.txn, target_segment, event_segment, &bytes)?;
    }
    Ok(())
}

/// Reject a trigger body whose Cypher source does not parse. Without this
/// gate, an invalid body would install cleanly and only error at firing
/// time — by which point dropping the bad trigger requires manual
/// intervention. The parsed AST is discarded; the executor re-parses on
/// firing so the AST shape can evolve independently of stored definitions.
fn validate_trigger_body_source(name: &str, source: &str) -> Result<(), ExecutionError> {
    crate::cypher::parser::parse(source).map_err(|e| {
        ExecutionError::Unsupported(format!(
            "trigger `{name}` body fails to parse — refusing to install. \
             Cypher parser error: {e}"
        ))
    })?;
    Ok(())
}

fn execute_create_trigger(
    c: &crate::cypher::ast::CreateTriggerClause,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_core::schema::triggers::TriggerSchema;
    use coordinode_modality::{LocalTriggerStore, TriggerStore as _};

    ctx.sync_txn_state();
    if LocalTriggerStore
        .get_definition(&mut ctx.txn, &c.name)?
        .is_some()
    {
        return Err(ExecutionError::Conflict(format!(
            "trigger `{}` already exists; use DROP TRIGGER first or ALTER it",
            c.name
        )));
    }

    validate_trigger_body_source(&c.name, &c.body_source)?;

    let target_schema = trigger_target_to_schema(&c.target);
    let target_segment = target_schema.index_key_segment();
    let events_schema = trigger_events_to_schema(c.events);

    let schema = TriggerSchema {
        name: c.name.clone(),
        target: target_schema,
        events: events_schema,
        timing: trigger_timing_to_schema(c.timing),
        body_source: c.body_source.clone(),
        cascade_limit: c.cascade_limit,
        cascade_fanout: c.cascade_fanout,
        on_error: c.on_error.as_ref().map(on_error_to_schema),
        enabled: true,
        created_at_hlc_us: current_hlc_us(),
    };
    let bytes = rmp_serde::to_vec(&schema)
        .map_err(|e| ExecutionError::Serialization(format!("trigger `{}` encode: {e}", c.name)))?;
    LocalTriggerStore.put_definition(&mut ctx.txn, &c.name, &bytes)?;

    for event_seg in events_schema.enabled_segments() {
        append_to_trigger_index(ctx, &target_segment, event_seg, &c.name)?;
    }

    let mut row = Row::new();
    row.insert("name".into(), Value::String(c.name.clone()));
    row.insert("status".into(), Value::String("created".into()));
    Ok(vec![row])
}

fn execute_drop_trigger(
    name: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_core::schema::triggers::TriggerSchema;
    use coordinode_modality::{LocalTriggerStore, TriggerStore as _};

    ctx.sync_txn_state();
    let bytes = match LocalTriggerStore.get_definition(&mut ctx.txn, name)? {
        Some(b) => b,
        None => {
            return Err(ExecutionError::Unsupported(format!(
                "no such trigger `{name}`"
            )));
        }
    };
    let schema: TriggerSchema = rmp_serde::from_slice(&bytes)
        .map_err(|e| ExecutionError::Serialization(format!("trigger `{name}` decode: {e}")))?;
    let target_segment = schema.target.index_key_segment();
    for event_seg in schema.events.enabled_segments() {
        remove_from_trigger_index(ctx, &target_segment, event_seg, name)?;
    }
    LocalTriggerStore.delete_definition(&mut ctx.txn, name)?;

    let mut row = Row::new();
    row.insert("name".into(), Value::String(name.into()));
    row.insert("status".into(), Value::String("dropped".into()));
    Ok(vec![row])
}

fn execute_show_triggers(ctx: &mut ExecutionContext<'_>) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_core::schema::triggers::TriggerSchema;
    use coordinode_modality::{LocalTriggerStore, TriggerStore as _};

    ctx.sync_txn_state();
    let scanned = LocalTriggerStore.scan_definitions(&mut ctx.txn)?;

    let mut rows: Vec<Row> = Vec::new();
    for (_k, v) in scanned {
        let schema: TriggerSchema = match rmp_serde::from_slice(&v) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut row = Row::new();
        row.insert("name".into(), Value::String(schema.name.clone()));
        let target_kind = match &schema.target {
            coordinode_core::schema::triggers::TriggerTargetSchema::Label { .. } => "label",
            coordinode_core::schema::triggers::TriggerTargetSchema::EdgeType { .. } => "edge_type",
        };
        row.insert("target_kind".into(), Value::String(target_kind.into()));
        row.insert(
            "target_name".into(),
            Value::String(schema.target.name().to_string()),
        );
        row.insert(
            "events".into(),
            Value::String(
                schema
                    .events
                    .enabled_segments()
                    .join(",")
                    .to_uppercase()
                    .replace('C', "CREATE")
                    .replace('U', "UPDATE")
                    .replace('D', "DELETE"),
            ),
        );
        row.insert(
            "timing".into(),
            Value::String(match schema.timing {
                coordinode_core::schema::triggers::TriggerTimingSchema::BeforeCommit => {
                    "BEFORE_COMMIT".into()
                }
                coordinode_core::schema::triggers::TriggerTimingSchema::AfterCommit => {
                    "AFTER_COMMIT".into()
                }
            }),
        );
        row.insert("enabled".into(), Value::Bool(schema.enabled));
        row.insert(
            "body_source".into(),
            Value::String(schema.body_source.clone()),
        );
        rows.push(row);
    }
    Ok(rows)
}

fn execute_alter_trigger(
    c: &crate::cypher::ast::AlterTriggerClause,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use crate::cypher::ast::AlterTriggerAction;
    use coordinode_core::schema::triggers::TriggerSchema;
    use coordinode_modality::{LocalTriggerStore, TriggerStore as _};

    ctx.sync_txn_state();
    let bytes = match LocalTriggerStore.get_definition(&mut ctx.txn, &c.name)? {
        Some(b) => b,
        None => {
            return Err(ExecutionError::Unsupported(format!(
                "no such trigger `{}`",
                c.name
            )));
        }
    };
    let mut schema: TriggerSchema = rmp_serde::from_slice(&bytes)
        .map_err(|e| ExecutionError::Serialization(format!("trigger `{}` decode: {e}", c.name)))?;

    let status: &'static str = match &c.action {
        AlterTriggerAction::Disable => {
            schema.enabled = false;
            "disabled"
        }
        AlterTriggerAction::Enable => {
            schema.enabled = true;
            "enabled"
        }
        AlterTriggerAction::SetBody(src) => {
            // Parse the replacement body up-front: a syntactically broken
            // body must not silently overwrite a working one, leaving the
            // trigger fire-broken until the next ALTER.
            validate_trigger_body_source(&c.name, src)?;
            schema.body_source = src.clone();
            "body_replaced"
        }
        AlterTriggerAction::SetOnError(pol) => {
            schema.on_error = Some(on_error_to_schema(pol));
            "on_error_replaced"
        }
    };

    let bytes = rmp_serde::to_vec(&schema)
        .map_err(|e| ExecutionError::Serialization(format!("trigger `{}` encode: {e}", c.name)))?;
    LocalTriggerStore.put_definition(&mut ctx.txn, &c.name, &bytes)?;

    let mut row = Row::new();
    row.insert("name".into(), Value::String(c.name.clone()));
    row.insert("status".into(), Value::String(status.into()));
    Ok(vec![row])
}

// ======================================================================
// Trigger firing engine (BEFORE COMMIT, synchronous, leader-only)
// ======================================================================

/// Fire all matching BEFORE COMMIT triggers for a mutation. Caller probes
/// `ctx.lookup_matching_triggers(target_segment, event)` to get the
/// candidate list, then passes the list here along with the event params.
///
/// Each trigger body is parsed, parameterised with `params` (which must
/// contain `$event`, `$node` or `$edge`, and `$before` / `$after` per the
/// trigger contract), planned, and executed in the same MVCC transaction
/// as the originating mutation. L1/L2 cascade counters are enforced.
///
/// `ON ERROR PROPAGATE` (the BEFORE COMMIT default) bubbles errors up to
/// abort the originating transaction. `RETRY` and `DEAD_LETTER` on
/// BEFORE COMMIT are simplified to PROPAGATE for now — synchronous retry
/// inside the same transaction would deadlock against write locks, and
/// dead-lettering inside an aborting transaction is paradoxical (the
/// dead-letter write would itself be rolled back). The doc comment in
/// the trigger architecture document spells out this constraint.
///
/// AFTER COMMIT triggers in the matched list are silently skipped — the
/// async-trigger executor consumes them via the oplog instead.
pub(crate) fn fire_before_commit_triggers(
    matched: &[coordinode_core::schema::triggers::TriggerSchema],
    params: &std::collections::HashMap<String, Value>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use coordinode_core::schema::triggers::TriggerTimingSchema;
    for trigger in matched {
        if !matches!(trigger.timing, TriggerTimingSchema::BeforeCommit) {
            continue;
        }
        ctx.cascade_enter(&trigger.name, trigger.cascade_limit, trigger.cascade_fanout)?;
        let result = execute_trigger_body_inline(trigger, params, ctx);
        ctx.cascade_exit();
        result?;
    }
    Ok(())
}

/// Parse the trigger body source, substitute params, plan, and execute
/// against `ctx`. Body writes land in the same `mvcc_write_buffer` as
/// the originating mutation — succeeding triggers commit together with
/// the user's transaction; a failing trigger aborts the whole batch via
/// the standard error-propagation path.
fn execute_trigger_body_inline(
    trigger: &coordinode_core::schema::triggers::TriggerSchema,
    params: &std::collections::HashMap<String, Value>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    let query = crate::cypher::parser::parse(&trigger.body_source).map_err(|e| {
        ExecutionError::Unsupported(format!(
            "trigger `{}` body re-parse failed at fire time: {e}",
            trigger.name
        ))
    })?;
    let mut plan = crate::planner::builder::build_logical_plan(&query).map_err(|e| {
        ExecutionError::Unsupported(format!("trigger `{}` body plan failed: {e}", trigger.name))
    })?;
    plan.root.substitute_params(params);
    let _ = execute_op(&plan.root, ctx)?;
    Ok(())
}

/// Build the parameter map for a node-CREATE trigger firing.
/// `$event = "CREATE"`, `$before = NULL`, `$after = props as Map`,
/// `$node = NodeId`.
fn trigger_params_for_node_create(
    node_id: NodeId,
    props: &std::collections::HashMap<String, Value>,
) -> std::collections::HashMap<String, Value> {
    let mut after: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    for (k, v) in props {
        after.insert(k.clone(), v.clone());
    }
    let mut params = std::collections::HashMap::with_capacity(4);
    params.insert("event".into(), Value::String("CREATE".into()));
    params.insert("before".into(), Value::Null);
    params.insert("after".into(), Value::Map(after));
    params.insert("node".into(), Value::Int(node_id.as_raw() as i64));
    params
}

/// Build the parameter map for a node-DELETE trigger firing.
/// `$event = "DELETE"`, `$before = props as Map`, `$after = NULL`,
/// `$node = NodeId`.
fn trigger_params_for_node_delete(
    node_id: NodeId,
    pre_props: &std::collections::BTreeMap<String, Value>,
) -> std::collections::HashMap<String, Value> {
    let mut params = std::collections::HashMap::with_capacity(4);
    params.insert("event".into(), Value::String("DELETE".into()));
    params.insert("before".into(), Value::Map(pre_props.clone()));
    params.insert("after".into(), Value::Null);
    params.insert("node".into(), Value::Int(node_id.as_raw() as i64));
    params
}

/// Build the parameter map for an edge-CREATE trigger firing.
/// `$event = "CREATE"`, `$before = NULL`, `$after = props as Map`,
/// `$src` / `$tgt` = endpoint NodeIds, `$edge_type` = edge type name.
fn trigger_params_for_edge_create(
    edge_type: &str,
    source_id: NodeId,
    target_id: NodeId,
    props: &[(String, Value)],
) -> std::collections::HashMap<String, Value> {
    let mut after: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    for (k, v) in props {
        after.insert(k.clone(), v.clone());
    }
    let mut params = std::collections::HashMap::with_capacity(6);
    params.insert("event".into(), Value::String("CREATE".into()));
    params.insert("before".into(), Value::Null);
    params.insert("after".into(), Value::Map(after));
    params.insert("src".into(), Value::Int(source_id.as_raw() as i64));
    params.insert("tgt".into(), Value::Int(target_id.as_raw() as i64));
    params.insert("edge_type".into(), Value::String(edge_type.to_string()));
    params
}

/// Build the parameter map for a node-UPDATE trigger firing.
/// `$event = "UPDATE"`, `$before = pre-mutation props`,
/// `$after = post-mutation props`, `$node = NodeId`.
fn trigger_params_for_node_update(
    node_id: NodeId,
    before_props: &std::collections::BTreeMap<String, Value>,
    after_props: &std::collections::BTreeMap<String, Value>,
) -> std::collections::HashMap<String, Value> {
    let mut params = std::collections::HashMap::with_capacity(4);
    params.insert("event".into(), Value::String("UPDATE".into()));
    params.insert("before".into(), Value::Map(before_props.clone()));
    params.insert("after".into(), Value::Map(after_props.clone()));
    params.insert("node".into(), Value::Int(node_id.as_raw() as i64));
    params
}

/// Build the parameter map for an edge-UPDATE trigger firing.
/// `$event = "UPDATE"`, `$before` / `$after` carry the edge's property maps
/// before and after the mutation, `$src` / `$tgt` are the endpoint NodeIds,
/// and `$edge_type` is the type name.
fn trigger_params_for_edge_update(
    edge_type: &str,
    src: NodeId,
    tgt: NodeId,
    before: &std::collections::BTreeMap<String, Value>,
    after: &std::collections::BTreeMap<String, Value>,
) -> std::collections::HashMap<String, Value> {
    let mut params = std::collections::HashMap::with_capacity(6);
    params.insert("event".into(), Value::String("UPDATE".into()));
    params.insert("before".into(), Value::Map(before.clone()));
    params.insert("after".into(), Value::Map(after.clone()));
    params.insert("src".into(), Value::Int(src.as_raw() as i64));
    params.insert("tgt".into(), Value::Int(tgt.as_raw() as i64));
    params.insert("edge_type".into(), Value::String(edge_type.to_string()));
    params
}

/// Build the parameter map for an edge-DELETE trigger firing.
/// `$event = "DELETE"`, `$before` carries the deleted edge's property map,
/// `$after = NULL`. `$src` / `$tgt` are the endpoint NodeIds and
/// `$edge_type` is the type name.
fn trigger_params_for_edge_delete(
    edge_type: &str,
    src: NodeId,
    tgt: NodeId,
    before: &std::collections::BTreeMap<String, Value>,
) -> std::collections::HashMap<String, Value> {
    let mut params = std::collections::HashMap::with_capacity(6);
    params.insert("event".into(), Value::String("DELETE".into()));
    params.insert("before".into(), Value::Map(before.clone()));
    params.insert("after".into(), Value::Null);
    params.insert("src".into(), Value::Int(src.as_raw() as i64));
    params.insert("tgt".into(), Value::Int(tgt.as_raw() as i64));
    params.insert("edge_type".into(), Value::String(edge_type.to_string()));
    params
}

/// Decode an edgeprop value (msgpack `Vec<(field_id, Value)>`) into a name→
/// value BTreeMap by resolving each interned field id back to its string name.
/// Field ids not in the interner are silently skipped — they cannot have been
/// produced by this engine's writes, so their presence indicates a corrupt
/// blob and we prefer a partial-but-consistent snapshot to an error.
fn decode_edgeprop_into_map(
    bytes: &[u8],
    ctx: &ExecutionContext<'_>,
) -> std::collections::BTreeMap<String, Value> {
    let mut out: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    if let Ok(prop_map) = decode_edge_props(bytes) {
        for (field_id, value) in prop_map {
            if let Some(name) = ctx.interner.resolve(field_id) {
                out.insert(name.to_string(), value);
            }
        }
    }
    out
}

/// Same projection as [`decode_edgeprop_into_map`] but takes the
/// already-decoded `Vec<(field_id, Value)>` returned by the typed
/// edge-prop helpers — avoids the redundant decode of bytes.
fn decode_edgeprop_map_into_named(
    prop_map: &[(u32, Value)],
    ctx: &ExecutionContext<'_>,
) -> std::collections::BTreeMap<String, Value> {
    let mut out: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    for (field_id, value) in prop_map {
        if let Some(name) = ctx.interner.resolve(*field_id) {
            out.insert(name.to_string(), value.clone());
        }
    }
    out
}

/// Extract `(labels, props_map)` from a NodeRecord using the interner to
/// resolve field IDs to property names. Used when building trigger
/// parameter snapshots for UPDATE / DELETE events.
fn snapshot_node_record(
    record: &NodeRecord,
    ctx: &ExecutionContext<'_>,
) -> (Vec<String>, std::collections::BTreeMap<String, Value>) {
    let labels = record.labels.clone();
    let mut props: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
    for (&field_id, value) in &record.props {
        if let Some(name) = ctx.interner.resolve(field_id) {
            props.insert(name.to_string(), value.clone());
        }
    }
    if let Some(extra) = &record.extra {
        for (name, value) in extra {
            props.insert(name.clone(), value.clone());
        }
    }
    (labels, props)
}

/// Execute CREATE TEXT INDEX: creates a text index definition and registers it.
///
/// The actual index creation and backfill is delegated to the text_index_registry.
/// The index definition is persisted to the schema: partition via MVCC write buffer.
fn execute_create_text_index(
    name: &str,
    label: &str,
    fields: &[crate::cypher::ast::TextIndexFieldSpec],
    default_language: Option<&str>,
    language_override: Option<&str>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let Some(registry) = ctx.text_index_registry else {
        return Err(ExecutionError::Unsupported(
            "CREATE TEXT INDEX requires text_index_registry in ExecutionContext".into(),
        ));
    };

    // Check if index already exists for any of the fields.
    for field in fields {
        if registry.has_index(label, &field.property) {
            return Err(ExecutionError::Unsupported(format!(
                "text index already exists for ({label}, {})",
                field.property
            )));
        }
    }

    let lang = default_language.unwrap_or("english").to_string();
    let lang_override = language_override.unwrap_or("_language").to_string();

    // Build per-field analyzer config from DDL field specs.
    let mut field_configs = std::collections::HashMap::new();
    let properties: Vec<String> = fields.iter().map(|f| f.property.clone()).collect();
    for field in fields {
        if let Some(ref analyzer) = field.analyzer {
            field_configs.insert(
                field.property.clone(),
                crate::index::TextFieldConfig {
                    analyzer: analyzer.clone(),
                },
            );
        }
    }

    let config = crate::index::TextIndexConfig {
        fields: field_configs,
        default_language: lang.clone(),
        language_override_property: lang_override,
    };
    let def = crate::index::IndexDefinition::text(name, label, properties.clone(), config);

    // Persist index definition transactionally through the index store.
    ctx.mvcc_put_index_def(&def)?;

    // Register in text index registry (creates tantivy directory + empty index).
    registry
        .register(def)
        .map_err(|e| ExecutionError::Unsupported(format!("register text index: {e}")))?;

    // Backfill: scan existing nodes with this label and any indexed property
    // through the node store (it owns the partition + key encoding).
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let shard_id = ctx.shard_id;
    let interner = &*ctx.interner;
    let mut count = 0u64;
    let _ = LocalNodeStore.for_each_in_shard_at_snapshot(
        ctx.engine,
        None,
        shard_id,
        &mut |node_id, _key, record| {
            if record.primary_label() == label {
                // Index all matching properties for this node.
                let mut indexed = false;
                for prop in &properties {
                    if let Some(field_id) = interner.lookup(prop) {
                        if let Some(val) = record.props.get(&field_id) {
                            if let Some(text) = val.as_str() {
                                registry.on_text_written(label, node_id, prop, text);
                                indexed = true;
                            }
                        }
                    }
                }
                if indexed {
                    count += 1;
                }
            }
            Ok(std::ops::ControlFlow::Continue(()))
        },
    );

    let props_str = properties.join(", ");
    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("label".to_string(), Value::String(label.to_string()));
    row.insert("properties".to_string(), Value::String(props_str));
    row.insert("default_language".to_string(), Value::String(lang));
    row.insert("documents_indexed".to_string(), Value::Int(count as i64));
    Ok(vec![row])
}

/// Execute DROP TEXT INDEX: removes a text index.
fn execute_drop_text_index(
    name: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let Some(registry) = ctx.text_index_registry else {
        return Err(ExecutionError::Unsupported(
            "DROP TEXT INDEX requires text_index_registry in ExecutionContext".into(),
        ));
    };

    // Find the definition by name to get (label, property).
    let def = registry.definitions().into_iter().find(|d| d.name == name);

    let Some(def) = def else {
        return Err(ExecutionError::Unsupported(format!(
            "text index '{name}' not found"
        )));
    };

    // Remove from registry.
    registry.unregister(&def.label, def.property());

    // Remove the definition transactionally through the index store.
    ctx.mvcc_delete_index_def(&def.name)?;

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("dropped".to_string(), Value::Bool(true));
    Ok(vec![row])
}

/// Execute CREATE ENCRYPTED INDEX — stores blind-index metadata in the schema partition.
///
/// The encrypted index enables token-based equality search on encrypted properties
/// without exposing plaintext to the server.
fn execute_create_encrypted_index(
    name: &str,
    label: &str,
    property: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Persist the definition transactionally through the encrypted-index store.
    let def = coordinode_modality::EncryptedIndexDefinition::new(name, label, property);
    ctx.mvcc_put_encrypted_index_def(&def)?;

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("label".to_string(), Value::String(label.to_string()));
    row.insert("property".to_string(), Value::String(property.to_string()));
    row.insert("created".to_string(), Value::Bool(true));
    Ok(vec![row])
}

/// Execute DROP ENCRYPTED INDEX — removes blind-index metadata from the schema partition.
fn execute_drop_encrypted_index(
    name: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Delete the definition transactionally through the encrypted-index store.
    // A future registry would enumerate definitions; today the DDL drops the
    // single name-keyed record.
    ctx.mvcc_delete_encrypted_index_def(name)?;

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("dropped".to_string(), Value::Bool(true));
    Ok(vec![row])
}

/// Execute `CREATE VECTOR INDEX idx ON :Label(property) OPTIONS {m, ef_construction, metric, dimensions}`.
///
/// 1. Validates vector_index_registry is available.
/// 2. Builds a `VectorIndexConfig` from the OPTIONS.
/// 3. Persists the `IndexDefinition` to the `Schema` partition.
/// 4. Registers the empty HNSW graph in the registry.
/// 5. Backfills existing nodes that have the indexed vector property.
#[allow(clippy::too_many_arguments)]
fn execute_create_vector_index(
    name: &str,
    label: &str,
    property: &str,
    m: usize,
    ef_construction: usize,
    metric: coordinode_core::graph::types::VectorMetric,
    dimensions: u32,
    quantization: coordinode_vector::hnsw::QuantizationCodec,
    online_during_build: crate::index::OnlineDuringBuild,
    ef_search: Option<usize>,
    rerank_candidates: Option<usize>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let Some(registry) = ctx.vector_index_registry else {
        return Err(ExecutionError::Unsupported(
            "CREATE VECTOR INDEX requires vector_index_registry in ExecutionContext".into(),
        ));
    };

    // Reject duplicate index names.
    if registry.has_index(label, property) {
        return Err(ExecutionError::Unsupported(format!(
            "vector index on :{label}({property}) already exists"
        )));
    }

    // Build the index definition. `quantization` is resolved earlier
    // in the planner from the Cypher OPTIONS string; the executor
    // just plugs it through.
    let config = crate::index::VectorIndexConfig {
        dimensions,
        metric,
        m,
        ef_construction,
        quantization,
        offload_vectors: false,
        ef_search,
        rerank_candidates,
    };
    let mut def = crate::index::IndexDefinition::hnsw(name, label, property, config);
    def.online_during_build = online_during_build;

    // Persist the definition to the schema partition THROUGH the
    // proposal pipeline: replicas discover the index by observing this
    // key in their applied stream and run their own local backfill
    // (the HNSW graph itself is never replicated, only the data is).
    // Falls back to a direct engine write in legacy/test contexts that
    // carry no pipeline.
    if let (Some(pipeline), Some(id_gen)) = (ctx.proposal_pipeline, ctx.proposal_id_gen) {
        let value = rmp_serde::to_vec(&def).map_err(|e| {
            ExecutionError::Unsupported(format!("serialize vector index '{name}': {e}"))
        })?;
        let proposal = coordinode_core::txn::proposal::RaftProposal {
            id: id_gen.next(),
            mutations: vec![coordinode_core::txn::proposal::Mutation::Put {
                partition: coordinode_core::txn::proposal::PartitionId::Schema,
                key: def.schema_key(),
                value,
            }],
            commit_ts: ctx
                .mvcc_oracle
                .map(|o| o.next())
                .unwrap_or(ctx.mvcc_read_ts),
            start_ts: ctx.mvcc_read_ts,
            bypass_rate_limiter: false,
        };
        let outcome = pipeline.propose_and_wait(&proposal).map_err(|e| {
            ExecutionError::Unsupported(format!("persist vector index '{name}': {e}"))
        })?;
        // DDL persists the index definition as its own Raft entry — fold its
        // committed index into operationTime so a causal read after a
        // CREATE VECTOR INDEX fences past the definition's replication, not
        // just the (often empty) data-write flush. `max` keeps any larger
        // index a later data flush in the same statement might mint.
        ctx.write_stats.applied_index = ctx.write_stats.applied_index.max(outcome.applied_index);
    } else {
        crate::index::ops::save_index_definition(ctx.engine, &def).map_err(|e| {
            ExecutionError::Unsupported(format!("persist vector index '{name}': {e}"))
        })?;
    }

    // Register the empty HNSW graph in memory with its tier handle
    // resolved from the executor's interner. Building the tier here
    // (rather than inside `register`) keeps the registry free of any
    // shared interner reference — register would otherwise need to
    // re-enter the same parking_lot RwLock the executor already
    // holds write-locked, which deadlocks on parking_lot.
    let label_id = ctx.interner.intern(label);
    let property_id = ctx.interner.intern(property);
    let tier = registry.tier_handle(label_id, property_id);
    registry.register_with_tier(def.clone(), tier);

    let field_id = ctx.interner.lookup(property);
    let shard_id = ctx.shard_id;

    // Snapshot the live HNSW handle BEFORE deciding sync vs background.
    // The handle is `Arc<RwLock<HnswIndex>>`, cheap to clone, and stays
    // valid for the lifetime of the registry entry.
    let hnsw_handle = registry.get(label, property);

    // Pick the execution mode based on whether the caller plumbed an
    // owned engine handle. With `engine_arc = Some(...)` we move the
    // backfill into a background thread and return immediately; with
    // None (test paths that build ExecutionContext with only a borrowed
    // engine) we keep the legacy synchronous loop.
    let (final_state, nodes_indexed): (IndexState, i64) =
        match (&ctx.engine_arc, hnsw_handle, field_id) {
            (Some(engine_arc), Some(hnsw), Some(fid)) => {
                // Publish the Building state so concurrent readers and the
                // crash-recovery path see "backfill in progress" before the
                // thread starts touching SSTs.
                let initial_state = IndexState::Building {
                    written: 0,
                    estimated_total: 0,
                };
                let _ =
                    crate::index::ops::save_index_state(ctx.engine, name, initial_state.clone());

                let engine = Arc::clone(engine_arc);
                let label_owned = label.to_string();
                let property_owned = property.to_string();
                let name_owned = name.to_string();

                std::thread::Builder::new()
                    .name(format!("vec-backfill-{name}"))
                    .spawn(move || {
                        let outcome =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                backfill_vector_index(
                                    engine.as_ref(),
                                    hnsw.as_ref(),
                                    &label_owned,
                                    &property_owned,
                                    fid,
                                    shard_id,
                                    &name_owned,
                                )
                            }));
                        let terminal = match outcome {
                            Ok(Ok(_written)) => IndexState::Ready,
                            Ok(Err(e)) => IndexState::Failed { reason: e },
                            Err(panic) => {
                                let reason = panic
                                    .downcast_ref::<&'static str>()
                                    .map(|s| (*s).to_string())
                                    .or_else(|| panic.downcast_ref::<String>().cloned())
                                    .unwrap_or_else(|| "panic in backfill thread".to_string());
                                IndexState::Failed { reason }
                            }
                        };
                        let _ = crate::index::ops::save_index_state(
                            engine.as_ref(),
                            &name_owned,
                            terminal,
                        );
                    })
                    .map_err(|e| {
                        ExecutionError::Unsupported(format!("spawn backfill thread: {e}"))
                    })?;

                (initial_state, 0)
            }
            _ => {
                // Synchronous fallback (legacy path / tests).
                let written = match field_id {
                    Some(fid) => {
                        run_backfill_sync(ctx.engine, registry, label, property, fid, shard_id)
                    }
                    None => 0,
                };
                (IndexState::Ready, written as i64)
            }
        };

    let state_label = match &final_state {
        IndexState::Building { .. } => "building",
        IndexState::Ready => "ready",
        IndexState::Failed { .. } => "failed",
    };

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("label".to_string(), Value::String(label.to_string()));
    row.insert("property".to_string(), Value::String(property.to_string()));
    row.insert("nodes_indexed".to_string(), Value::Int(nodes_indexed));
    row.insert("state".to_string(), Value::String(state_label.to_string()));
    Ok(vec![row])
}

/// Synchronous backfill: scans every node in shard, extracts the indexed
/// vector property, inserts into the registry's HNSW handle. Returns the
/// number of nodes inserted. Used by the legacy non-Arc test path.
fn run_backfill_sync(
    engine: &StorageEngine,
    registry: &crate::index::VectorIndexRegistry,
    label: &str,
    property: &str,
    field_id: u32,
    shard_id: u16,
) -> u64 {
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let mut backfilled = 0u64;
    let _ = LocalNodeStore.for_each_in_shard_at_snapshot(
        engine,
        None,
        shard_id,
        &mut |node_id, _key, record| {
            if record.primary_label() == label {
                if let Some(val) = record.props.get(&field_id) {
                    if let Some(vec_data) = try_extract_vector(val) {
                        registry.on_vector_written(label, node_id, property, &vec_data);
                        backfilled += 1;
                    }
                }
            }
            Ok(std::ops::ControlFlow::Continue(()))
        },
    );
    backfilled
}

/// Background backfill: same scan + extract + insert as the sync path,
/// but writes directly into the cloned HNSW handle without going through
/// the registry (registry isn't Arc-shared, but the handle is). Persists
/// progress every PROGRESS_INTERVAL nodes so a crash mid-build resumes
/// from the last checkpoint instead of starting over.
fn backfill_vector_index(
    engine: &StorageEngine,
    hnsw: &std::sync::RwLock<coordinode_vector::hnsw::HnswIndex>,
    label: &str,
    property: &str,
    field_id: u32,
    shard_id: u16,
    index_name: &str,
) -> std::result::Result<u64, String> {
    const PROGRESS_INTERVAL: u64 = 1000;
    let _ = property;

    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let mut written = 0u64;
    let mut since_checkpoint = 0u64;
    LocalNodeStore
        .for_each_in_shard_at_snapshot(engine, None, shard_id, &mut |node_id, _key, record| {
            if record.primary_label() == label {
                if let Some(val) = record.props.get(&field_id) {
                    if let Some(vec_data) = try_extract_vector(val) {
                        if let Ok(mut graph) = hnsw.write() {
                            graph.insert(node_id.as_raw(), vec_data);
                        }
                        written += 1;
                        since_checkpoint += 1;
                        if since_checkpoint >= PROGRESS_INTERVAL {
                            since_checkpoint = 0;
                            let _ = crate::index::ops::save_index_state(
                                engine,
                                index_name,
                                IndexState::Building {
                                    written,
                                    estimated_total: 0,
                                },
                            );
                        }
                    }
                }
            }
            Ok(std::ops::ControlFlow::Continue(()))
        })
        .map_err(|e| format!("shard scan: {e}"))?;
    Ok(written)
}

/// Execute `DROP VECTOR INDEX idx`: removes an HNSW vector index by name.
///
/// 1. Looks up the definition in the vector registry by label+property matching index name.
/// 2. Removes definition from schema partition.
/// 3. Unregisters from the in-memory vector registry.
fn execute_drop_vector_index(
    name: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let Some(registry) = ctx.vector_index_registry else {
        return Err(ExecutionError::Unsupported(
            "DROP VECTOR INDEX requires vector_index_registry in ExecutionContext".into(),
        ));
    };

    // Find the definition by name to get (label, property).
    let def = registry
        .all_definitions()
        .into_iter()
        .find(|d| d.name == name);

    let Some(def) = def else {
        return Err(ExecutionError::Unsupported(format!(
            "vector index '{name}' not found"
        )));
    };

    let label = def.label.clone();
    let property = def.property().to_string();

    // Tombstone the definition transactionally through the index store.
    ctx.mvcc_delete_index_def(&def.name)?;

    // Remove from in-memory registry.
    registry.unregister(&label, &property);

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("label".to_string(), Value::String(label));
    row.insert("property".to_string(), Value::String(property));
    row.insert("dropped".to_string(), Value::Bool(true));
    Ok(vec![row])
}

/// Execute `CREATE [UNIQUE] [SPARSE] INDEX idx ON :Label(prop) [WHERE pred]`.
///
/// 1. Validates the registry is available.
/// 2. Checks for duplicate index name.
/// 3. Builds an `IndexDefinition` with optional `.unique()` / `.sparse()` / `.with_filter()`.
/// 4. Persists the definition to the `Schema` partition via the registry.
/// 5. Backfills existing nodes of the target label into the new index.
fn execute_create_btree_index(
    name: &str,
    label: &str,
    property: &str,
    unique: bool,
    sparse: bool,
    filter: Option<&crate::index::definition::PartialFilter>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let Some(registry) = ctx.btree_index_registry else {
        return Err(ExecutionError::Unsupported(
            "CREATE INDEX requires btree_index_registry in ExecutionContext".into(),
        ));
    };

    // Reject duplicate index names.
    if registry.get(name).is_some() {
        return Err(ExecutionError::Unsupported(format!(
            "index '{name}' already exists"
        )));
    }

    // Build the index definition.
    let mut def = crate::index::IndexDefinition::btree(name, label, property);
    if unique {
        def = def.unique();
    }
    if sparse {
        def = def.sparse();
    }
    if let Some(f) = filter {
        def = def.with_filter(f.clone());
    }

    // Persist to storage and update in-memory registry.
    registry
        .register(ctx.engine, def)
        .map_err(|e| ExecutionError::Unsupported(format!("register index '{name}': {e}")))?;

    // Backfill existing nodes that match the label, scanned through the
    // node store (it owns the partition + key encoding).
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let field_id = ctx.interner.lookup(property);
    let engine = ctx.engine;
    let mut backfilled = 0u64;
    let _ = LocalNodeStore.for_each_in_shard_at_snapshot(
        engine,
        None,
        ctx.shard_id,
        &mut |node_id, _key, record| {
            if record.primary_label() != label {
                return Ok(std::ops::ControlFlow::Continue(()));
            }
            let prop_val = if let Some(fid) = field_id {
                record.props.get(&fid).cloned().unwrap_or(Value::Null)
            } else {
                Value::Null
            };
            // Sparse indexes skip null values.
            if sparse && prop_val.is_null() {
                return Ok(std::ops::ControlFlow::Continue(()));
            }
            // Apply partial filter if any.
            if let Some(f) = filter {
                let props = [(property.to_string(), prop_val.clone())];
                if !f.matches(&props) {
                    return Ok(std::ops::ControlFlow::Continue(()));
                }
            }
            let props = [(property.to_string(), prop_val)];
            if let Err(e) = registry.on_node_created(engine, node_id, label, &props) {
                tracing::warn!(
                    "CREATE INDEX backfill: unique violation on node {node_id:?} ({label}.{property}): {e}"
                );
            } else {
                backfilled += 1;
            }
            Ok(std::ops::ControlFlow::Continue(()))
        },
    );

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("label".to_string(), Value::String(label.to_string()));
    row.insert("property".to_string(), Value::String(property.to_string()));
    row.insert("unique".to_string(), Value::Bool(unique));
    row.insert("sparse".to_string(), Value::Bool(sparse));
    row.insert("nodes_indexed".to_string(), Value::Int(backfilled as i64));
    Ok(vec![row])
}

/// Execute `DROP INDEX idx`: removes a B-tree index by name.
///
/// 1. Looks up the definition in the registry.
/// 2. Drops all index entries from storage (`ops::drop_index`).
/// 3. Unregisters from the in-memory registry.
fn execute_drop_btree_index(
    name: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let Some(registry) = ctx.btree_index_registry else {
        return Err(ExecutionError::Unsupported(
            "DROP INDEX requires btree_index_registry in ExecutionContext".into(),
        ));
    };

    // Verify the index exists before attempting to drop it.
    let def = registry
        .get(name)
        .ok_or_else(|| ExecutionError::Unsupported(format!("index '{name}' not found")))?;

    let label = def.label.clone();
    let property = def.property().to_string();

    // Drop definition + all index entries from storage.
    crate::index::ops::drop_index(ctx.engine, &def)
        .map_err(|e| ExecutionError::Unsupported(format!("drop index '{name}': {e}")))?;

    // Remove from in-memory registry.
    registry.unregister(name);

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("label".to_string(), Value::String(label));
    row.insert("property".to_string(), Value::String(property));
    row.insert("dropped".to_string(), Value::Bool(true));
    Ok(vec![row])
}

/// Execute a procedure call: evaluate args, dispatch to the procedure registry,
/// convert output rows to Row format.
fn execute_procedure_call(
    procedure: &str,
    args: &[Expr],
    yield_items: &[String],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let proc_ctx = ctx
        .procedure_ctx
        .as_ref()
        .ok_or_else(|| ExecutionError::Unsupported("no procedure context available".into()))?;

    // Evaluate arguments to values
    let empty_row = Row::new();
    let arg_values: Vec<Value> = args.iter().map(|a| eval_expr(a, &empty_row)).collect();

    // Dispatch to the procedure
    let proc_rows = crate::advisor::procedures::execute_procedure(procedure, &arg_values, proc_ctx)
        .map_err(ExecutionError::Unsupported)?;

    // Convert procedure rows to Row format
    let mut rows = Vec::with_capacity(proc_rows.len());
    for proc_row in proc_rows {
        let mut row = Row::new();
        for (col_name, value) in proc_row {
            // If yield_items is not empty, only include requested columns
            if yield_items.is_empty() || yield_items.contains(&col_name) {
                row.insert(col_name, value);
            }
        }
        rows.push(row);
    }

    Ok(rows)
}

/// Map a Layer-3 [`CommitError`](coordinode_storage::engine::transaction::CommitError)
/// from [`Transaction::commit`](coordinode_storage::engine::transaction::Transaction::commit)
/// into an [`ExecutionError`], preserving the typed `CapacityExhausted`
/// variant. Without this, the gRPC handler would see a generic `Internal`
/// Status for a capacity-exhausted write rather than the gRPC-canonical
/// `RESOURCE_EXHAUSTED` (and the operator-actionable `endpoint-id` /
/// `used-bytes` / `hard-limit-bytes` metadata headers would never be set).
/// The structured `CapacityExhausted` survives because `Transaction::commit`
/// already folds the pipeline's `ProposalError::CapacityExhausted` into a
/// `CommitError::Storage(StorageError::CapacityExhausted { .. })`, which this
/// mapping forwards verbatim; the server's `db_err_to_status` drills into both
/// `Storage` and `Execution` variants when resolving capacity.
fn commit_err_to_execution(
    err: coordinode_storage::engine::transaction::CommitError,
) -> ExecutionError {
    use coordinode_storage::engine::transaction::CommitError;
    match err {
        CommitError::Conflict(msg) => ExecutionError::Conflict(msg),
        CommitError::Storage(e) => ExecutionError::Storage(e),
        CommitError::Serialization(msg) => ExecutionError::Serialization(msg),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod fusion_kernel_tests;
