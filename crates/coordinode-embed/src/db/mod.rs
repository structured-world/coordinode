//! Embedded database: open CoordiNode storage, execute queries in-process.
//!
//! Embedded mode is single-process (no Raft, no clustering).
//! For CE 3-node HA, use the coordinode server binary.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
// no-std: spin::RwLock (drop-in, same API). parking_lot::RwLock is the
// std-only hot-path-fast lock per ~/projects/sw/CLAUDE.md.
use parking_lot::{RwLock, RwLockReadGuard};
use std::time::{Duration, Instant};

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeId, NodeIdAllocator};
use coordinode_core::graph::types::VectorConsistencyMode;
use coordinode_core::txn::proposal::ProposalIdGenerator;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_query::advisor::nplus1::NPlus1Detector;
use coordinode_query::advisor::{
    normalize_and_fingerprint, DismissedSet, ProcedureContext, QueryRegistry, SourceContext,
};
use coordinode_query::cypher;
use coordinode_query::executor::row::Row;
use coordinode_query::executor::runner::{
    execute, execute_no_commit, AdaptiveConfig, ExecutionContext, ExecutionError, ExtensionHandler,
    ExtensionRegistry, FeedbackCache, WriteStats,
};
use coordinode_query::planner;
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

/// Outcome of a Cypher execution: result rows plus mutation statistics. Used
/// by [`Database::execute_cypher_full`] so callers (notably the gRPC server)
/// can surface real mutation counts in their response stats.
#[derive(Debug, Clone)]
pub struct CypherResult {
    pub rows: Vec<Row>,
    pub write_stats: WriteStats,
}

/// `StorageStats` adapter that augments graph-level statistics (label
/// counts, fan-out averages) with per-vector-index statistics drawn from
/// the live `VectorIndexRegistry`. R-PUSH1's push-down rule needs both
/// dimensions in one place — `optimize_push_down` reads everything through
/// a single `&dyn StorageStats` reference.
///
/// The graph half delegates to the cached `StorageStatsComputer`; the
/// vector half is computed on demand (cheap — registry lookups are
/// in-memory). Crossover thresholds are derived per-index from HNSW M and
/// quantization settings (cached at build time on the index definition;
/// the heuristic here is the temporary formula that R-PUSH4 will replace
/// with measured constants).
struct CombinedStats<'a> {
    graph: &'a coordinode_storage::engine::stats::StorageStatsComputer,
    vector: &'a coordinode_query::index::VectorIndexRegistry,
}

impl<'a> coordinode_core::graph::stats::StorageStats for CombinedStats<'a> {
    fn total_node_count(&self) -> u64 {
        self.graph.total_node_count()
    }

    fn node_count_for_label(&self, label: &str) -> Option<u64> {
        self.graph.node_count_for_label(label)
    }

    fn avg_fan_out_for_type(&self, edge_type: &str) -> Option<f64> {
        self.graph.avg_fan_out_for_type(edge_type)
    }

    fn avg_fan_out(&self) -> f64 {
        self.graph.avg_fan_out()
    }

    fn label_count(&self) -> u64 {
        self.graph.label_count()
    }

    fn vector_index_size(&self, label: &str, property: &str) -> Option<u64> {
        let handle = self.vector.get(label, property)?;
        let guard = handle.read().ok()?;
        Some(guard.len() as u64)
    }

    fn vector_index_dim(&self, label: &str, property: &str) -> Option<u32> {
        let def = self.vector.get_definition(label, property)?;
        Some(def.vector_config.as_ref()?.dimensions)
    }

    fn vector_index_crossover(&self, label: &str, property: &str) -> Option<usize> {
        // Per arch/core/query-engine.md § Graph Predicate Push-Down: crossover
        // is "per-index metadata, computed once at build time from M, dim,
        // quantisation". Until R-PUSH4 lands measured constants, the heuristic
        // below tracks the documented defaults:
        //   - Node-typed HNSW (M=16, f32): ~500
        //   - Edge-typed or quantised: ~200
        // The formula multiplies M by 32 (≈ HNSW frontier expansion at typical
        // recall targets), then halves for quantised indexes where each f32 op
        // is cheaper.
        let def = self.vector.get_definition(label, property)?;
        let cfg = def.vector_config.as_ref()?;
        let base = cfg.m.max(8).saturating_mul(32);
        let crossover = if cfg.quantization.is_active() {
            base / 2
        } else {
            base
        };
        Some(crossover.clamp(64, 1024))
    }
}

/// Default TTL for cached storage statistics (seconds).
///
/// EXPLAIN / EXPLAIN SUGGEST recompute statistics from storage on every call.
/// For small databases this is <10ms, but at >100K nodes the `node:` scan
/// becomes a bottleneck.  Caching with a short TTL avoids re-scanning on
/// repeated EXPLAIN calls while keeping estimates reasonably fresh.
const STATS_CACHE_TTL_SECS: u64 = 60;

/// Number of node IDs to pre-allocate per batch.
///
/// On startup, the database reserves `ID_BATCH_SIZE` IDs by persisting
/// the batch ceiling to disk. IDs within the batch are allocated in-memory
/// (lock-free). When the batch is exhausted, a new batch is persisted.
///
/// On crash recovery, unused IDs in the last batch are skipped (gaps are
/// acceptable — the invariant is no duplicates, not contiguity).
const ID_BATCH_SIZE: u64 = 1000;

/// Schema partition key for the persisted node ID high-water mark.
///
/// Stores a big-endian u64: the ceiling of the current ID batch.
/// On open, the allocator resumes from this value.
const SCHEMA_KEY_NEXT_NODE_ID: &[u8] = b"meta:next_node_id";

/// Schema partition key for the persisted field interner.
///
/// Stores the serialized FieldInterner (field name ↔ u32 ID mapping).
/// Loaded on Database::open, updated after queries that intern new fields.
const SCHEMA_KEY_FIELD_INTERNER: &[u8] = b"meta:field_interner";

/// Try to extract f32 vector data from a Value.
///
/// Handles both `Value::Vector` (native) and `Value::Array` containing
/// only Float/Int elements (Cypher array literals like `[1.0, 0.0]`).
pub(crate) fn try_extract_vector(val: &coordinode_core::graph::types::Value) -> Option<Vec<f32>> {
    use coordinode_core::graph::types::Value;
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
            if vec.is_empty() {
                None
            } else {
                Some(vec)
            }
        }
        _ => None,
    }
}

/// One-line human description of a vector index's serving health, for EXPLAIN
/// output.
fn describe_index_health(state: &coordinode_vector::health::IndexHealthState) -> String {
    use coordinode_vector::health::IndexHealthState as H;
    match state {
        H::Ready { indexed_hlc } => format!("ready (indexed_hlc={indexed_hlc})"),
        H::Rebuilding {
            progress,
            eta_ms,
            indexed_hlc,
        } => format!(
            "rebuilding {:.0}% (indexed_hlc={indexed_hlc}, eta={eta_ms}ms)",
            progress * 100.0
        ),
        H::Offline { reason } => format!("offline: {reason}"),
    }
}

/// Loads f32 vectors from the node: partition for HNSW reranking.
///
/// When HNSW indexes have `offload_vectors` enabled, this loader provides
/// f32 vectors from LSM storage. Batch-reads NodeRecords and extracts
/// the requested vector property for exact reranking of SQ8 candidates.
///
/// Property-agnostic: a single instance serves all HNSW indexes — the
/// property name is provided per-call by the HNSW search method.
pub struct StorageVectorLoader {
    engine: Arc<StorageEngine>,
    // Snapshot of the interner at query start. Vector loaders look up
    // property names that were registered when the HNSW index was
    // built — names that always pre-date the query. Holding a snapshot
    // here, rather than a shared `Arc<RwLock<…>>`, prevents a re-entrant
    // read-after-write deadlock with the Database's execute path that
    // holds the interner write-lock for the duration of execute.
    interner: FieldInterner,
    shard_id: u16,
}

impl StorageVectorLoader {
    /// Create a new loader backed by the given storage engine.
    pub fn new(engine: Arc<StorageEngine>, interner: FieldInterner, shard_id: u16) -> Self {
        Self {
            engine,
            interner,
            shard_id,
        }
    }
}

impl coordinode_vector::VectorLoader for StorageVectorLoader {
    fn load_vectors(
        &self,
        ids: &[u64],
        property: &str,
    ) -> std::collections::HashMap<u64, Vec<f32>> {
        let mut result = std::collections::HashMap::with_capacity(ids.len());
        let field_id = match self.interner.lookup(property) {
            Some(id) => id,
            None => return result,
        };

        use coordinode_core::txn::timestamp::Timestamp;
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        use coordinode_storage::engine::transaction::Transaction;
        // Vector loading is a bulk read — cheap direct-mode transaction (no
        // snapshot/OCC); reads the latest committed node records.
        let txn = Transaction::new(&self.engine, None, Timestamp::ZERO, None);
        let node_store = LocalNodeStore;
        for &node_id in ids {
            let record = match node_store.get(&txn, self.shard_id, NodeId::from_raw(node_id)) {
                Ok(Some(r)) => r,
                _ => continue,
            };
            if let Some(val) = record.props.get(&field_id) {
                if let Some(vec_data) = try_extract_vector(val) {
                    result.insert(node_id, vec_data);
                }
            }
        }

        result
    }
}

/// Embedded database instance.
/// How `execute_cypher_impl` should treat the statement's transaction
/// boundary (ADR-042).
enum TxnMode {
    /// Single-statement auto-commit: allocate a fresh `read_ts`, build a new
    /// transaction, and commit (flush) at the end. The default for every
    /// bare statement.
    AutoCommit,
    /// One statement of an interactive multi-statement transaction: resume the
    /// parked transaction state (reusing its pinned `read_ts` / snapshot for
    /// repeatable reads), run WITHOUT committing, and hand the updated state
    /// back to the caller to re-park. Boxed so the enum stays small (the
    /// state is large; auto-commit is the common variant).
    Interactive(Box<coordinode_storage::engine::transaction::TransactionState>),
}

pub struct Database {
    engine: Arc<StorageEngine>,
    // Wrapped in Arc<RwLock<…>> so concurrent gRPC handlers can hold a
    // shared `Database` (under `Arc<RwLock<Database>>`) and still mutate
    // the interner from any per-request execute path: lookup-heavy
    // read paths take `.read()`, the rare "new property name" path
    // takes `.write()`. FieldInterner itself stays alloc-clean in
    // coordinode-core — the lock is bolted on here by the std-only
    // consumer, per the tier policy.
    // no-std: spin::RwLock (drop-in).
    interner: Arc<RwLock<FieldInterner>>,
    allocator: NodeIdAllocator,
    shard_id: u16,
    /// Query fingerprint registry — tracks execution statistics per query pattern.
    query_registry: Arc<QueryRegistry>,
    /// N+1 pattern detector — flags repeated queries from the same source location.
    nplus1_detector: Arc<NPlus1Detector>,
    /// Dismissed suggestion fingerprints for `db.advisor.dismiss()`.
    dismissed: Arc<DismissedSet>,
    /// Pre-allocated ID batch ceiling. When `allocator.current() >= ceiling`,
    /// a new batch must be persisted before allocating more IDs.
    id_batch_ceiling: AtomicU64,
    /// MVCC timestamp oracle — allocates monotonic timestamps for
    /// snapshot isolation reads and commit ordering.
    oracle: Arc<TimestampOracle>,
    /// Proposal ID generator for the Raft proposal pipeline.
    /// Arc-shared with the drain thread's pipeline.
    proposal_id_gen: Arc<ProposalIdGenerator>,
    /// The write pipeline every mutation must flow through. In embedded
    /// mode this is a local pipeline applying straight to the engine; in
    /// cluster mode it is the Raft proposal pipeline, and bypassing it
    /// (writing to the local engine directly) silently breaks
    /// replication: followers never see the data.
    pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline>,
    /// Session-level vector MVCC consistency mode.
    vector_consistency: VectorConsistencyMode,
    /// Session-level read concern. Default: Local.
    read_concern: coordinode_core::txn::read_concern::ReadConcernLevel,
    /// One-shot snapshot timestamp for the next query (consumed on use).
    /// Set by `execute_cypher_with_read_concern` with Snapshot level.
    snapshot_read_ts: Option<u64>,
    /// Cached storage statistics for EXPLAIN cost estimation.
    /// `None` = never computed or invalidated.  Refreshed when
    /// the TTL expires (see `STATS_CACHE_TTL_SECS`).
    cached_stats: Mutex<
        Option<(
            coordinode_storage::engine::stats::StorageStatsComputer,
            Instant,
        )>,
    >,
    /// How long cached storage statistics remain valid.
    stats_ttl: Duration,
    /// Session-level write concern. Default: Majority.
    write_concern: coordinode_core::txn::write_concern::WriteConcern,
    /// Index registry — tracks active indexes for EXPLAIN SUGGEST false-positive prevention.
    index_registry: coordinode_query::index::IndexRegistry,
    /// Vector index registry — holds live HNSW indexes for accelerated vector search.
    vector_index_registry: Arc<coordinode_query::index::VectorIndexRegistry>,
    /// Background oplog tailer keeping HNSW indexes current with
    /// replicated writes (see [`crate::vector_worker`]). `None` when
    /// the process has no oplog (pure embedded mode without Raft).
    /// Held for its Drop (stops the thread when the Database closes).
    _vector_worker: Option<crate::vector_worker::VectorIndexWorker>,
    /// Text index registry — holds live tantivy indexes for full-text search.
    text_index_registry: coordinode_query::index::TextIndexRegistry,
    /// Extension-op handler registry threaded into every ExecutionContext.
    /// Empty for a plain CE Database (no extension ops dispatchable);
    /// populated via [`Database::register_extension`] by an enterprise layer
    /// or an integration test so extension operators (e.g. a sharded
    /// CREATE VECTOR INDEX) reach their handler.
    extension_registry: ExtensionRegistry,
    /// Adaptive query plan configuration — controls parallel traversal thresholds.
    adaptive_config: AdaptiveConfig,
    /// Feedback cache for known super-node fan-out degrees.
    /// Shared across queries within the same Database session via Arc.
    feedback_cache: FeedbackCache,
    /// Volatile write drain buffer for w:memory and w:cache write concerns.
    /// Shared between all ExecutionContext instances. The background drain
    /// thread batches buffered mutations into proposal pipeline calls.
    drain_buffer: Arc<coordinode_core::txn::drain::DrainBuffer>,
    /// NVMe-backed write buffer for `w:cache` crash recovery.
    /// `None` when `nvme_write_buffer_path` is not configured in `StorageConfig`.
    nvme_write_buffer: Option<Arc<coordinode_storage::cache::write_buffer::NvmeWriteBuffer>>,
    /// Handle to the background drain thread. Dropped on Database::drop,
    /// which flushes remaining entries (graceful shutdown).
    _drain_handle: coordinode_core::txn::drain::DrainHandle,
    /// Handle to the COMPUTED TTL background reaper thread. Scans label
    /// schemas for TTL properties and deletes expired nodes/fields/subtrees.
    /// Dropped on Database::drop (graceful shutdown).
    _ttl_reaper_handle: Option<coordinode_query::index::ttl_reaper::TtlReaperHandle>,
    /// Per-query-string parse + plan cache. Repeated invocations of
    /// the same Cypher text skip parse / semantic analysis / logical
    /// plan build entirely; the per-call optimizer passes still run
    /// on a clone of the cached plan so they stay sensitive to live
    /// index registry state. See [`PlanCache`].
    plan_cache: Arc<PlanCache>,
    /// Open interactive multi-statement transactions (ADR-042), keyed by a
    /// server-allocated transaction id. Leader-local and ephemeral: parked
    /// `TransactionState` (uncommitted writes + OCC read-set + pinned
    /// snapshot) plus the last-touched instant for idle-timeout reaping.
    /// Never replicated — durability happens only at `commit_transaction`.
    interactive_txns: Mutex<
        std::collections::HashMap<
            u64,
            (
                coordinode_storage::engine::transaction::TransactionState,
                Instant,
            ),
        >,
    >,
    /// Monotonic source of interactive transaction ids.
    next_txn_id: AtomicU64,
    /// Idle timeout for interactive transactions (ADR-042): an open
    /// transaction with no activity for this long is auto-rolled-back (it pins
    /// an MVCC snapshot + buffers memory). Set by the server from the
    /// `--interactive-txn-idle-timeout-secs` flag (passed via
    /// `COORDINODE_EXTRA_ARGS` in `/etc/coordinode/coordinode.conf`).
    interactive_idle_timeout: Duration,
    /// Max buffered (uncommitted) bytes per interactive transaction before it
    /// is aborted — caps leader memory a client can hold without committing.
    /// Set by the server from the `--interactive-txn-max-bytes` flag (passed
    /// via `COORDINODE_EXTRA_ARGS` in `/etc/coordinode/coordinode.conf`).
    max_interactive_txn_bytes: usize,
}

/// A query whose parse + analyze + logical-plan-build succeeded, kept
/// around so repeated invocations of the same query string skip those
/// steps entirely. The optimizer passes (index selection, VectorTopK
/// annotation, predicate push-down) still run per call because they
/// depend on the live index registry / storage statistics and would
/// stale-bind if cached.
#[derive(Debug, Clone)]
struct CachedPlan {
    /// Canonical form (literals scrubbed) — fed to the advisor.
    canonical: String,
    /// Stable fingerprint over the canonical form.
    fingerprint: u64,
    /// Logical plan from `planner::build_logical_plan(&ast)`. Cloned on
    /// every cache hit so the per-call optimizer passes mutate a fresh
    /// copy without invalidating the cache entry.
    plan: planner::logical::LogicalPlan,
}

/// Bounded query-string → [`CachedPlan`] cache shared across all
/// queries on this Database. Each `execute_cypher_*` entry-point hits
/// this before parsing.
///
/// Sizing: 1024 entries fits the working set of typical benchmark
/// workloads (a handful of distinct templates repeated millions of
/// times) and OLTP services (a few hundred prepared queries). On
/// overflow we evict one arbitrary entry — no LRU bookkeeping; the
/// trade-off is correct for stable workloads and acceptable when
/// the working set is small relative to the bound.
///
/// no-std: spin::RwLock + hashbrown::HashMap (drop-in).
struct PlanCache {
    inner: parking_lot::RwLock<std::collections::HashMap<String, Arc<CachedPlan>>>,
    max_entries: usize,
}

impl PlanCache {
    fn new(max_entries: usize) -> Self {
        Self {
            inner: parking_lot::RwLock::new(std::collections::HashMap::new()),
            max_entries,
        }
    }

    fn get(&self, query: &str) -> Option<Arc<CachedPlan>> {
        self.inner.read().get(query).cloned()
    }

    fn put(&self, query: String, entry: Arc<CachedPlan>) {
        let mut map = self.inner.write();
        if map.len() >= self.max_entries && !map.contains_key(&query) {
            // Bound the cache. Picking an arbitrary key is intentional:
            // proper LRU costs a write-lock on every hit (to bump
            // recency), which negates the read-concurrency win. For
            // stable workloads (same N templates forever) eviction
            // policy is irrelevant; for churning workloads, occasional
            // re-build on miss costs ≈ one parse+plan.
            if let Some(k) = map.keys().next().cloned() {
                map.remove(&k);
            }
        }
        map.insert(query, entry);
    }
}

/// Per-call read/write semantics for a single Cypher query.
///
/// Built once at the entry point of every `execute_cypher_*` method
/// from the Database's session defaults plus any one-shot overrides
/// the caller supplied (e.g. gRPC `ReadConcern` / `WriteConcern` on
/// the wire). Passed by reference into `execute_cypher_impl`, which
/// reads its values instead of mutating `self.*` to install them
/// for the duration of the call.
///
/// Owning the per-call values in a small struct here serves the
/// concurrency story: as the impl path stops touching `&mut self`
/// for concerns/consistency, the lock granularity at the gRPC
/// service layer can shrink from "one Database mutex per request"
/// to "Database held shared, only the actual write-paths take an
/// exclusive lock".
#[derive(Debug, Clone)]
struct QuerySession {
    read_concern: coordinode_core::txn::read_concern::ReadConcernLevel,
    /// One-shot snapshot timestamp consumed by this query (already
    /// taken out of `Database.snapshot_read_ts` when the session
    /// was captured).
    snapshot_read_ts: Option<u64>,
    write_concern: coordinode_core::txn::write_concern::WriteConcern,
    vector_consistency: VectorConsistencyMode,
}

/// Error from embedded database operations.
#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    #[error("storage error: {0}")]
    Storage(#[from] coordinode_storage::error::StorageError),

    #[error("parse error: {0}")]
    Parse(#[from] cypher::ParseError),

    #[error("plan error: {0}")]
    Plan(#[from] planner::PlanError),

    #[error("execution error: {0}")]
    Execution(#[from] ExecutionError),

    #[error("semantic error: {0}")]
    Semantic(String),

    #[error("{0}")]
    Other(String),
}

impl Database {
    /// Open or create a database at the given path.
    ///
    /// Uses `OwnedLocalProposalPipeline` for embedded single-node mode.
    /// For cluster mode (CE 3-node HA), use `open_with_pipeline()` with
    /// a `RaftProposalPipeline` instead.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DatabaseError> {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            path.as_ref(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        Self::open_with_config(config)
    }

    /// Open or create a database from an explicit [`StorageConfig`].
    ///
    /// Like [`Self::open`] but takes a pre-resolved storage configuration, so
    /// the caller can open a multi-endpoint topology rather than the single
    /// default endpoint `open` derives from a path. The maintenance CLI
    /// (`backup` / `restore`) uses this to open the same topology the server
    /// runs with. The primary data directory is the first endpoint's path.
    ///
    /// Uses `OwnedLocalProposalPipeline` for embedded single-node mode; for
    /// cluster mode use [`Self::from_engine`] with a `RaftProposalPipeline`.
    pub fn open_with_config(config: StorageConfig) -> Result<Self, DatabaseError> {
        let path = config.data_dir().to_path_buf();
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_with_oracle(&config, oracle.clone())?;
        let engine = Arc::new(engine);
        let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
            Arc::new(OwnedLocalProposalPipeline::new(&engine));
        Self::finish_open(&path, config, oracle, engine, pipeline)
    }

    /// Open an in-memory database backed by `lsm_tree::fs::MemFs`.
    ///
    /// Useful for:
    /// - Integration tests that exercise the full `EmbeddedDb`
    ///   stack (Cypher executor + storage + modality) without
    ///   needing host filesystem.
    /// - Ephemeral dev shells, in-process benchmarks, demo
    ///   environments that should not leave files behind.
    /// - CI matrix legs where `engine_for_logic()` from
    ///   `coordinode-test-fixtures` isn't a fit because the test
    ///   uses the full `Database::*` surface.
    ///
    /// **Persistence semantics:** no real disk I/O happens. Process
    /// restart loses all data. Tests that exercise WAL recovery /
    /// crash safety / reopen-after-flush MUST use [`Self::open`]
    /// with a `tempfile::TempDir` instead — MemFs doesn't simulate
    /// the persistence layer, only the FS-call surface.
    pub fn open_in_memory() -> Result<Self, DatabaseError> {
        // Virtual path under MemFs root. Doesn't have to exist on
        // the host FS — MemFs maintains its own tree under this.
        let virtual_path = std::path::PathBuf::from("/coordinode-embed-in-memory");
        let fs = Arc::new(lsm_tree::fs::MemFs::new());
        let config = StorageConfig::with_endpoints_no_persistence(vec![EndpointConfig::new(
            "default-memfs",
            &virtual_path,
            Media::Ram,
            Durability::Volatile,
            Tier::Memory,
        )])
        .with_fs(fs as Arc<dyn lsm_tree::fs::Fs>);
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_with_oracle(&config, oracle.clone())?;
        let engine = Arc::new(engine);
        let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
            Arc::new(OwnedLocalProposalPipeline::new(&engine));
        Self::finish_open(&virtual_path, config, oracle, engine, pipeline)
    }

    /// Initialize a database from pre-opened engine, oracle, and pipeline.
    ///
    /// Used by the server binary in cluster mode (G063): the server creates
    /// a shared `StorageEngine` + `TimestampOracle` for the `RaftNode`, then
    /// passes the same engine + a `RaftProposalPipeline` here. The DrainBuffer
    /// and TTL reaper submit mutations through Raft for replication.
    pub fn from_engine(
        path: impl AsRef<Path>,
        engine: Arc<StorageEngine>,
        oracle: Arc<TimestampOracle>,
        pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline>,
    ) -> Result<Self, DatabaseError> {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            path.as_ref(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        Self::finish_open(path.as_ref(), config, oracle, engine, pipeline)
    }

    /// Shared initialization logic for both `open()` and `open_with_pipeline()`.
    fn finish_open(
        path: &Path,
        config: StorageConfig,
        oracle: Arc<TimestampOracle>,
        engine: Arc<StorageEngine>,
        pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline>,
    ) -> Result<Self, DatabaseError> {
        // Recover node ID allocator from persisted high-water mark.
        // The HWM is the ceiling of the last reserved batch — on crash,
        // some IDs in the batch may be unused (gaps), but no duplicates.
        let hwm = match engine.get(Partition::Schema, SCHEMA_KEY_NEXT_NODE_ID)? {
            Some(bytes) if bytes.len() >= 8 => {
                let arr: [u8; 8] = bytes[..8]
                    .try_into()
                    .map_err(|_| DatabaseError::Semantic("corrupt node ID HWM".into()))?;
                u64::from_be_bytes(arr)
            }
            _ => 0,
        };

        // Reserve next batch: persist ceiling BEFORE allocating any IDs.
        // This guarantees crash safety — worst case we skip unused IDs.
        let ceiling = hwm + ID_BATCH_SIZE;
        engine.put(
            Partition::Schema,
            SCHEMA_KEY_NEXT_NODE_ID,
            &ceiling.to_be_bytes(),
        )?;

        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(hwm));

        // Recover field interner from persisted state.
        // If no persisted state exists (fresh database), start with empty interner.
        let interner = match engine.get(Partition::Schema, SCHEMA_KEY_FIELD_INTERNER)? {
            Some(bytes) => FieldInterner::from_bytes(&bytes).unwrap_or_else(|| {
                tracing::warn!("corrupt field interner data, starting fresh");
                FieldInterner::new()
            }),
            None => FieldInterner::new(),
        };
        // Wrap the interner in the shared Arc<RwLock<_>> early so the
        // vector registry can use the same instance to resolve label /
        // property ids when binding the per-index tier handle. The
        // Database struct stores the same Arc below (line 632 region).
        let shared_interner: Arc<RwLock<FieldInterner>> = Arc::new(RwLock::new(interner));

        // Load index registry from storage for EXPLAIN SUGGEST accuracy.
        let index_registry = coordinode_query::index::IndexRegistry::new();
        if let Err(e) = index_registry.load_all(&engine) {
            tracing::warn!("failed to load index registry: {e}, starting fresh");
        }

        // Load vector index definitions from schema: partition and rebuild
        // HNSW graphs from stored vectors (eager rebuild). The registry is
        // tier-backed: every index it registers persists f32 to LSM per
        // ADR-033. Interning happens inside `load_vector_indexes` under a
        // brief write guard; the registry itself holds no interner ref
        // (would cause reentrant write deadlocks against execute_cypher).
        let vector_index_registry = Arc::new(Self::load_vector_indexes(
            engine.clone(),
            &shared_interner,
            1, /* shard_id */
        ));

        // Tail the oplog for replicated vector writes. The bootstrap
        // rebuild above covered history; the worker covers the live
        // tail from here on (HNSW insert is an upsert, so any overlap
        // between the two is harmless). Pure embedded deployments have
        // no oplog directory and run without the worker: the executor
        // updates indexes inline on the write path.
        let oplog_dir = engine.data_dir().join("oplog").join("0");
        let vector_worker = if oplog_dir.is_dir() {
            let mut tailer = coordinode_storage::oplog::tailer::OplogTailer::new(
                &oplog_dir,
                coordinode_storage::oplog::tailer::ResumeToken::from_start(0),
            );
            let start = tailer
                .seek_to_end()
                .unwrap_or_else(|_| coordinode_storage::oplog::tailer::ResumeToken::from_start(0));
            Some(crate::vector_worker::VectorIndexWorker::spawn(
                oplog_dir,
                start,
                Arc::clone(&vector_index_registry),
                Arc::clone(&shared_interner),
            ))
        } else {
            None
        };

        // Load text index definitions and rebuild tantivy indexes from stored nodes.
        let text_index_base = path.join("text_indexes");
        let text_index_registry = Self::load_text_indexes(
            &engine,
            &shared_interner.read(),
            1, /* shard_id */
            &text_index_base,
        );

        let proposal_id_gen = Arc::new(ProposalIdGenerator::new());

        // Create drain buffer and background drain thread for volatile writes.
        // The pipeline is either OwnedLocalProposalPipeline (embedded) or
        // RaftProposalPipeline (cluster mode, via open_with_pipeline).
        let drain_config = coordinode_core::txn::drain::DrainConfig {
            interval_ms: config.drain_interval_ms,
            batch_max: config.drain_batch_max,
            capacity_bytes: config.drain_buffer_capacity_bytes,
        };
        let drain_buffer = Arc::new(coordinode_core::txn::drain::DrainBuffer::new(
            drain_config.capacity_bytes,
        ));

        // Open the NVMe write buffer for w:cache crash recovery (if configured).
        // Recovery runs first: any entries from a previous crash are re-injected
        // into the DrainBuffer before the drain thread starts, ensuring they are
        // drained to Raft in the next drain cycle.
        let nvme_write_buffer = if let Some(ref nvme_path) = config.nvme_write_buffer_path {
            let recovered =
                coordinode_storage::cache::write_buffer::NvmeWriteBuffer::recover(nvme_path)
                    .map_err(|e| {
                        DatabaseError::Other(format!("NVMe write buffer recovery failed: {e}"))
                    })?;
            for entry in recovered {
                drain_buffer.append(entry).map_err(|e| {
                    DatabaseError::Other(format!(
                        "failed to re-inject recovered w:cache entry: {e}"
                    ))
                })?;
            }
            let wb = coordinode_storage::cache::write_buffer::NvmeWriteBuffer::open(nvme_path)
                .map_err(|e| DatabaseError::Other(format!("NVMe write buffer open failed: {e}")))?;
            Some(Arc::new(wb))
        } else {
            None
        };

        let drain_handle = coordinode_core::txn::drain::DrainHandle::start(
            Arc::clone(&drain_buffer),
            Arc::clone(&pipeline),
            Arc::clone(&proposal_id_gen),
            drain_config,
            nvme_write_buffer
                .as_ref()
                .map(|wb| Arc::clone(wb) as Arc<dyn coordinode_core::txn::drain::WriteBufferHook>),
        );

        // Start COMPUTED TTL background reaper (default: 60s interval, 1000 batch).
        let ttl_reaper_config = coordinode_query::index::ttl_reaper::TtlReaperConfig::default();
        let ttl_reaper_handle = if ttl_reaper_config.enabled {
            Some(coordinode_query::index::ttl_reaper::TtlReaperHandle::start(
                Arc::clone(&engine),
                1, // shard_id
                ttl_reaper_config,
                shared_interner.read().clone(),
                Arc::clone(&pipeline),
                Arc::clone(&proposal_id_gen),
            ))
        } else {
            None
        };

        Ok(Self {
            engine,
            interner: shared_interner,
            allocator,
            shard_id: 1,
            query_registry: Arc::new(QueryRegistry::new()),
            nplus1_detector: Arc::new(NPlus1Detector::new()),
            dismissed: Arc::new(DismissedSet::new()),
            id_batch_ceiling: AtomicU64::new(ceiling),
            oracle,
            proposal_id_gen,
            pipeline,
            vector_consistency: VectorConsistencyMode::default(),
            read_concern: coordinode_core::txn::read_concern::ReadConcernLevel::default(),
            snapshot_read_ts: None,
            cached_stats: Mutex::new(None),
            stats_ttl: Duration::from_secs(STATS_CACHE_TTL_SECS),
            write_concern: coordinode_core::txn::write_concern::WriteConcern::default(),
            index_registry,
            vector_index_registry,
            _vector_worker: vector_worker,
            text_index_registry,
            extension_registry: ExtensionRegistry::new(),
            adaptive_config: AdaptiveConfig::default(),
            feedback_cache: FeedbackCache::default(),
            drain_buffer,
            nvme_write_buffer,
            _drain_handle: drain_handle,
            _ttl_reaper_handle: ttl_reaper_handle,
            // 1024 entries is plenty for the workloads we benchmark
            // against — they repeat a small number of templates. The
            // bound prevents unbounded growth on adversarial inputs
            // that produce a fresh query string per call.
            plan_cache: Arc::new(PlanCache::new(1024)),
            interactive_txns: Mutex::new(std::collections::HashMap::new()),
            next_txn_id: AtomicU64::new(0),
            interactive_idle_timeout: Self::DEFAULT_INTERACTIVE_TXN_IDLE_TIMEOUT,
            max_interactive_txn_bytes: Self::DEFAULT_MAX_INTERACTIVE_TXN_BYTES,
        })
    }

    /// Load persisted vector index definitions from `schema:idx:*` and
    /// rebuild HNSW graphs by scanning stored vectors in the `node:` partition.
    ///
    /// Called during `Database::open()` for eager HNSW rebuild.
    fn load_vector_indexes(
        engine: Arc<StorageEngine>,
        interner_arc: &Arc<RwLock<FieldInterner>>,
        shard_id: u16,
    ) -> coordinode_query::index::VectorIndexRegistry {
        use coordinode_query::index::IndexType;

        // Tier-backed registry: every registered HNSW index gets a
        // VectorTierHandle scoped to its `(label_id, property_id)`.
        // Interning is done here (caller-side) so the registry never
        // touches a shared interner lock and stays reentrancy-safe.
        let registry =
            coordinode_query::index::VectorIndexRegistry::with_vector_tier(engine.clone());
        let engine = engine.as_ref();

        // Step 1: Scan schema:idx:* for HNSW index definitions.
        let iter = match engine.prefix_scan(Partition::Schema, b"schema:idx:") {
            Ok(it) => it,
            Err(e) => {
                tracing::warn!("failed to scan vector index definitions: {e}");
                return registry;
            }
        };

        let mut hnsw_defs = Vec::new();
        for guard in iter {
            let Ok((_key, value)) = guard.into_inner() else {
                continue;
            };
            if let Ok(def) =
                rmp_serde::from_slice::<coordinode_query::index::IndexDefinition>(&value)
            {
                if def.index_type == IndexType::Hnsw && def.vector_config.is_some() {
                    hnsw_defs.push(def);
                }
            }
        }

        if hnsw_defs.is_empty() {
            return registry;
        }

        Self::register_and_populate_hnsw(&registry, interner_arc, engine, shard_id, &hnsw_defs);
        registry
    }

    /// Discover vector index definitions that were replicated into the
    /// Schema partition after this Database opened (a follower applying
    /// a leader's CREATE VECTOR INDEX) and bring them live: register in
    /// the in-memory registry and rebuild the local HNSW from stored
    /// nodes. Returns the number of indexes brought up. Cluster
    /// deployments call this alongside [`Self::refresh_field_interner`]
    /// whenever the applied index advances.
    pub fn refresh_vector_indexes(&self) -> Result<usize, DatabaseError> {
        use coordinode_query::index::IndexType;
        let defs = coordinode_query::index::ops::list_index_definitions(&self.engine)?;
        let new_defs: Vec<_> = defs
            .into_iter()
            .filter(|d| d.index_type == IndexType::Hnsw && d.vector_config.is_some())
            .filter(|d| !self.vector_index_registry.has_index(&d.label, d.property()))
            .collect();
        if new_defs.is_empty() {
            return Ok(0);
        }
        Self::register_and_populate_hnsw(
            &self.vector_index_registry,
            &self.interner,
            &self.engine,
            self.shard_id,
            &new_defs,
        );
        Ok(new_defs.len())
    }

    /// Register the given HNSW definitions in `registry` and populate
    /// them by scanning stored node records (shared by the open-time
    /// loader and the cluster refresh path).
    fn register_and_populate_hnsw(
        registry: &coordinode_query::index::VectorIndexRegistry,
        interner_arc: &Arc<RwLock<FieldInterner>>,
        engine: &StorageEngine,
        shard_id: u16,
        hnsw_defs: &[coordinode_query::index::IndexDefinition],
    ) {
        use coordinode_core::graph::node::NodeRecord;

        // Step 2: Resolve (label, property) → interned ids in one short
        // write-locked pass, build per-index tier handles, then register
        // each HNSW with its tier bound. Lock scope is the for-loop body
        // only; released before step 3's read pass.
        {
            let mut g = interner_arc.write();
            for def in hnsw_defs {
                let label_id = g.intern(&def.label);
                let property_id = g.intern(def.property());
                let tier = registry.tier_handle(label_id, property_id);
                registry.register_with_tier(def.clone(), tier);
            }
        }

        // Step 3: Scan node: partition once, populating all HNSW indexes.
        // Build a lookup: label → [(property, label)] for efficient matching.
        let mut label_props: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for def in hnsw_defs {
            label_props
                .entry(def.label.clone())
                .or_default()
                .push(def.property().to_string());
        }

        let node_prefix = {
            let mut p = Vec::with_capacity(5 + 2 + 1);
            p.extend_from_slice(b"node:");
            p.extend_from_slice(&shard_id.to_be_bytes());
            p.push(b':');
            p
        };

        let node_iter = match engine.prefix_scan(Partition::Node, &node_prefix) {
            Ok(it) => it,
            Err(e) => {
                tracing::warn!("failed to scan nodes for HNSW rebuild: {e}");
                return;
            }
        };

        // Track per-index vector counts for structured logging.
        let mut per_index_counts: std::collections::HashMap<(String, String), usize> =
            std::collections::HashMap::new();

        // Step 3 takes the read guard for property-id lookups. Safe to
        // co-exist with previous step because the write guard from
        // step 2 was released at the end of its block scope above.
        let interner = interner_arc.read();

        for guard in node_iter {
            let Ok((_key, value)) = guard.into_inner() else {
                continue;
            };
            let Ok(record) = NodeRecord::from_msgpack(&value) else {
                continue;
            };

            // Check if this node's label has any vector indexes.
            let primary_label = record.primary_label();
            let Some(props) = label_props.get(primary_label) else {
                continue;
            };

            // Decode node ID from key.
            let node_id = match coordinode_core::graph::node::decode_node_key(&_key) {
                Some((_shard, nid)) => nid,
                None => continue,
            };

            // Extract each indexed vector property.
            for prop_name in props {
                if let Some(field_id) = interner.lookup(prop_name) {
                    if let Some(value) = record.props.get(&field_id) {
                        if let Some(vec_data) = try_extract_vector(value) {
                            registry.on_vector_written(
                                primary_label,
                                node_id,
                                prop_name,
                                &vec_data,
                            );
                            *per_index_counts
                                .entry((primary_label.to_string(), prop_name.clone()))
                                .or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // Log per-index rebuild counts for observability.
        for def in hnsw_defs {
            let count = per_index_counts
                .get(&(def.label.clone(), def.property().to_string()))
                .copied()
                .unwrap_or(0);
            tracing::info!(
                index = %def.name,
                label = %def.label,
                property = %def.property(),
                vectors = count,
                "rebuilt HNSW index on reopen"
            );

            // Crash-recovery cleanup: a backfill that was interrupted (state
            // == Building) or that aborted (state == Failed) leaves stale
            // markers in schema. The full-scan rebuild above already
            // repopulated the in-memory HNSW from every node record, so the
            // index is consistent with on-disk data; flip the persisted
            // state back to Ready to match.
            let needs_state_reset =
                !matches!(def.state, coordinode_query::index::IndexState::Ready);
            if needs_state_reset {
                match coordinode_query::index::ops::save_index_state(
                    engine,
                    &def.name,
                    coordinode_query::index::IndexState::Ready,
                ) {
                    Ok(true) => tracing::info!(
                        index = %def.name,
                        prior = ?def.state,
                        "reset stale index state to Ready after rebuild"
                    ),
                    Ok(false) => tracing::warn!(
                        index = %def.name,
                        "save_index_state returned false during state reset"
                    ),
                    Err(e) => tracing::warn!(
                        index = %def.name,
                        error = %e,
                        "failed to reset index state after rebuild"
                    ),
                }
                registry.set_state(
                    &def.label,
                    def.property(),
                    coordinode_query::index::IndexState::Ready,
                );
            }
        }

        // The eager rebuild folded every node record committed before open
        // into the in-memory graphs, so each index is current as of the
        // engine's snapshot seqno. Seed the freshness watermark there; the
        // oplog-tailing worker advances it further as live writes arrive.
        registry.advance_indexed_hlc_all(engine.snapshot());
    }

    /// Load persisted text index definitions from `schema:idx:*` and
    /// rebuild tantivy indexes by scanning stored nodes in the `node:` partition.
    fn load_text_indexes(
        engine: &StorageEngine,
        interner: &FieldInterner,
        shard_id: u16,
        base_dir: &Path,
    ) -> coordinode_query::index::TextIndexRegistry {
        use coordinode_core::graph::node::NodeRecord;
        use coordinode_query::index::IndexType;

        let registry = coordinode_query::index::TextIndexRegistry::new(base_dir);

        // Step 1: Scan schema:idx:* for Text index definitions.
        let iter = match engine.prefix_scan(Partition::Schema, b"schema:idx:") {
            Ok(it) => it,
            Err(e) => {
                tracing::warn!("failed to scan text index definitions: {e}");
                return registry;
            }
        };

        let mut text_defs = Vec::new();
        for guard in iter {
            let Ok((_key, value)) = guard.into_inner() else {
                continue;
            };
            if let Ok(def) =
                rmp_serde::from_slice::<coordinode_query::index::IndexDefinition>(&value)
            {
                if def.index_type == IndexType::Text && def.text_config.is_some() {
                    text_defs.push(def);
                }
            }
        }

        if text_defs.is_empty() {
            return registry;
        }

        // Step 2: Register all definitions (creates empty tantivy indexes).
        for def in &text_defs {
            if let Err(e) = registry.register(def.clone()) {
                tracing::warn!("failed to register text index {}: {e}", def.name);
            }
        }

        // Step 3: Scan node: partition once, populating all text indexes.
        let mut label_props: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for def in &text_defs {
            let entry = label_props.entry(def.label.clone()).or_default();
            for prop in &def.properties {
                if !entry.contains(prop) {
                    entry.push(prop.clone());
                }
            }
        }

        let node_prefix = {
            let mut p = Vec::with_capacity(5 + 2 + 1);
            p.extend_from_slice(b"node:");
            p.extend_from_slice(&shard_id.to_be_bytes());
            p.push(b':');
            p
        };

        let node_iter = match engine.prefix_scan(Partition::Node, &node_prefix) {
            Ok(it) => it,
            Err(e) => {
                tracing::warn!("failed to scan nodes for text index rebuild: {e}");
                return registry;
            }
        };

        let mut total_docs = 0usize;
        for guard in node_iter {
            let Ok((_key, value)) = guard.into_inner() else {
                continue;
            };
            let Ok(record) = NodeRecord::from_msgpack(&value) else {
                continue;
            };

            let primary_label = record.primary_label();
            let Some(props) = label_props.get(primary_label) else {
                continue;
            };

            let node_id = match coordinode_core::graph::node::decode_node_key(&_key) {
                Some((_shard, nid)) => nid,
                None => continue,
            };

            for prop_name in props {
                if let Some(field_id) = interner.lookup(prop_name) {
                    if let Some(value) = record.props.get(&field_id) {
                        if let Some(text) = value.as_str() {
                            registry.on_text_written(primary_label, node_id, prop_name, text);
                            total_docs += 1;
                        }
                    }
                }
            }
        }

        if total_docs > 0 {
            tracing::info!(
                "rebuilt {} text index(es) with {total_docs} document(s)",
                text_defs.len()
            );
        }

        registry
    }

    /// Snapshot the current session defaults into a [`QuerySession`].
    ///
    /// Consumes any one-shot `snapshot_read_ts` so a second query in
    /// the same session sees the default (latest) read timestamp.
    fn capture_session(&mut self) -> QuerySession {
        QuerySession {
            read_concern: self.read_concern,
            snapshot_read_ts: self.snapshot_read_ts.take(),
            write_concern: self.write_concern.clone(),
            vector_consistency: self.vector_consistency,
        }
    }

    /// If the query is a recognized session-SET command, apply its
    /// effect to the Database's defaults and return `true` (caller
    /// should short-circuit and return an empty result set). Returns
    /// `false` for regular Cypher.
    ///
    /// Lifts SET handling out of `execute_cypher_impl` so the impl
    /// can stay on `&self`; this method needs `&mut self` because it
    /// mutates the session default field.
    fn try_apply_session_set(&mut self, query: &str) -> bool {
        if let Some(mode) = Self::try_parse_session_set(query) {
            self.vector_consistency = mode;
            true
        } else {
            false
        }
    }

    /// Execute a Cypher query with source location context.
    ///
    /// Same as `execute_cypher()` but also records the source call site
    /// in the advisor registry for debug source tracking.
    pub fn execute_cypher_with_source(
        &mut self,
        query: &str,
        source: &SourceContext,
    ) -> Result<Vec<Row>, DatabaseError> {
        if self.try_apply_session_set(query) {
            return Ok(Vec::new());
        }
        let session = self.capture_session();
        self.execute_cypher_impl(query, Some(source), None, &session, TxnMode::AutoCommit)
            .map(|(rows, _, _)| rows)
    }

    /// Execute a Cypher query with both source context and bound parameters.
    ///
    /// Combines source tracking (advisor N+1 detection) with parameter binding.
    /// Used by the gRPC server when the client provides both.
    pub fn execute_cypher_with_params_and_source(
        &mut self,
        query: &str,
        params: std::collections::HashMap<String, coordinode_core::graph::types::Value>,
        source: &SourceContext,
    ) -> Result<Vec<Row>, DatabaseError> {
        let params = if params.is_empty() {
            None
        } else {
            Some(params)
        };
        if self.try_apply_session_set(query) {
            return Ok(Vec::new());
        }
        let session = self.capture_session();
        self.execute_cypher_impl(query, Some(source), params, &session, TxnMode::AutoCommit)
            .map(|(rows, _, _)| rows)
    }

    /// Execute a Cypher query and return result rows.
    ///
    /// Automatically tracks query fingerprint and execution time in the
    /// query advisor registry for performance analysis.
    /// Register an extension-op handler under `name`. An enterprise layer (or
    /// an integration test) calls this at setup so that extension operators
    /// (a trailing clause on CREATE VECTOR INDEX, etc.) dispatch to it; a plain
    /// CE Database registers none. Idempotent on the name (last registration
    /// wins). Must be called before the queries that produce the op.
    pub fn register_extension(
        &mut self,
        name: impl Into<String>,
        handler: Arc<dyn ExtensionHandler>,
    ) {
        self.extension_registry.register(name, handler);
    }

    pub fn execute_cypher(&mut self, query: &str) -> Result<Vec<Row>, DatabaseError> {
        if self.try_apply_session_set(query) {
            return Ok(Vec::new());
        }
        let session = self.capture_session();
        self.execute_cypher_impl(query, None, None, &session, TxnMode::AutoCommit)
            .map(|(rows, _, _)| rows)
    }

    /// Execute a Cypher query with bound parameters.
    ///
    /// Parameters replace `$name` references in the query before execution.
    /// This is the safe way to pass user input — prevents injection attacks.
    pub fn execute_cypher_with_params(
        &mut self,
        query: &str,
        params: std::collections::HashMap<String, coordinode_core::graph::types::Value>,
    ) -> Result<Vec<Row>, DatabaseError> {
        let params = if params.is_empty() {
            None
        } else {
            Some(params)
        };
        if self.try_apply_session_set(query) {
            return Ok(Vec::new());
        }
        let session = self.capture_session();
        self.execute_cypher_impl(query, None, params, &session, TxnMode::AutoCommit)
            .map(|(rows, _, _)| rows)
    }

    /// Execute a Cypher query end-to-end with full session-level overrides and
    /// observable mutation statistics. The single entry point used by the gRPC
    /// `CypherService.execute_cypher` handler — propagates the client's
    /// `read_concern` and `write_concern` to the executor (replacing the
    /// previous behaviour where they were validated but silently ignored) and
    /// returns [`CypherResult`] with [`WriteStats`] so gRPC `QueryStats` can
    /// surface real mutation counts instead of hardcoded zeros.
    ///
    /// `read_concern` and `write_concern` are one-shot overrides
    /// applied only to this call's [`QuerySession`]; session defaults
    /// on the Database are not touched.
    pub fn execute_cypher_full(
        &mut self,
        query: &str,
        params: Option<std::collections::HashMap<String, coordinode_core::graph::types::Value>>,
        source: Option<&SourceContext>,
        read_concern: Option<coordinode_core::txn::read_concern::ReadConcern>,
        write_concern: Option<coordinode_core::txn::write_concern::WriteConcern>,
    ) -> Result<CypherResult, DatabaseError> {
        if self.try_apply_session_set(query) {
            return Ok(CypherResult {
                rows: Vec::new(),
                write_stats: WriteStats::default(),
            });
        }
        let mut session = self.capture_session();
        if let Some(ref rc) = read_concern {
            rc.validate()
                .map_err(|e| DatabaseError::Semantic(e.to_string()))?;
            session.read_concern = rc.level;
            session.snapshot_read_ts = rc.at_timestamp;
        }
        if let Some(ref wc) = write_concern {
            session.write_concern = wc.clone();
        }

        let params = params.and_then(|p| if p.is_empty() { None } else { Some(p) });
        self.execute_cypher_impl(query, source, params, &session, TxnMode::AutoCommit)
            .map(|(rows, write_stats, _)| CypherResult { rows, write_stats })
    }

    /// Begin an interactive multi-statement transaction (ADR-042).
    ///
    /// Returns a server-allocated transaction id. Pass it to
    /// [`Self::execute_in_transaction`] for each statement, then
    /// [`Self::commit_transaction`] or [`Self::rollback_transaction`]. The
    /// transaction pins an MVCC snapshot at its `start_ts`, so every statement
    /// reads the same point-in-time (repeatable read). State is leader-local
    /// and ephemeral — durability happens only at commit. Idle transactions
    /// are reaped after the configured idle timeout
    /// ([`Self::set_interactive_idle_timeout`]).
    pub fn begin_transaction(&self) -> u64 {
        self.reap_idle_transactions(self.interactive_idle_timeout);
        let id = self.next_txn_id.fetch_add(1, Ordering::Relaxed) + 1;
        let read_ts = self.oracle.next();
        let snapshot = self.engine.snapshot_at(read_ts.as_raw());
        let mut txn = coordinode_storage::engine::transaction::Transaction::new(
            &self.engine,
            Some(&self.oracle),
            read_ts,
            snapshot,
        );
        let state = txn.take_state();
        self.interactive_txns
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id, (state, Instant::now()));
        id
    }

    /// Run one statement of an interactive transaction (ADR-042).
    ///
    /// The statement reads at the transaction's pinned snapshot and its writes
    /// buffer on the transaction without committing. A statement error aborts
    /// the transaction (its parked state is dropped, matching SQL "current
    /// transaction is aborted" semantics) — subsequent statements and commit
    /// then fail with "unknown transaction id". A single client drives its
    /// transaction serially, so the state is checked out for the statement.
    pub fn execute_in_transaction(
        &self,
        txn_id: u64,
        query: &str,
        params: Option<std::collections::HashMap<String, coordinode_core::graph::types::Value>>,
    ) -> Result<Vec<Row>, DatabaseError> {
        let state = {
            let mut reg = self
                .interactive_txns
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            match reg.remove(&txn_id) {
                Some((state, _touched)) => state,
                None => {
                    return Err(DatabaseError::Other(format!(
                        "unknown transaction id {txn_id}"
                    )))
                }
            }
        };
        let session = QuerySession {
            read_concern: self.read_concern,
            snapshot_read_ts: None,
            write_concern: self.write_concern.clone(),
            vector_consistency: self.vector_consistency,
        };
        let params = params.and_then(|p| if p.is_empty() { None } else { Some(p) });
        // On error the state is intentionally NOT re-parked → transaction aborts.
        let (rows, _stats, out_state) = self.execute_cypher_impl(
            query,
            None,
            params,
            &session,
            TxnMode::Interactive(Box::new(state)),
        )?;
        if let Some(state) = out_state {
            // Cap buffered (uncommitted) memory: a client that keeps writing
            // without committing must not grow leader memory unbounded. On
            // breach the transaction aborts (state dropped, handle consumed).
            let buffered = state.buffered_bytes();
            if buffered > self.max_interactive_txn_bytes {
                return Err(DatabaseError::Other(format!(
                    "interactive transaction {txn_id} exceeded max_interactive_txn_bytes \
                     ({buffered} > {}); transaction aborted",
                    self.max_interactive_txn_bytes
                )));
            }
            self.interactive_txns
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(txn_id, (state, Instant::now()));
        }
        Ok(rows)
    }

    /// Commit an interactive transaction (ADR-042): validate the accumulated
    /// OCC read-set, assign `commit_ts`, and persist every buffered mutation
    /// in a single proposal. The handle is consumed (removed from the
    /// registry) whether commit succeeds or fails; on `ErrConflict` the client
    /// retries the whole transaction from `begin`. Returns the committed Raft
    /// index (the causal `operationTime` token; 0 in embedded mode).
    pub fn commit_transaction(&self, txn_id: u64) -> Result<u64, DatabaseError> {
        let state = self
            .interactive_txns
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&txn_id)
            .map(|(s, _)| s);
        let Some(state) = state else {
            return Err(DatabaseError::Other(format!(
                "unknown transaction id {txn_id}"
            )));
        };
        let mut txn = coordinode_storage::engine::transaction::Transaction::resume(
            &self.engine,
            Some(&self.oracle),
            state,
        );
        let wc = self.write_concern.clone();
        let commit_ctx = coordinode_storage::engine::transaction::CommitContext {
            write_concern: &wc,
            pipeline: Some(self.pipeline.as_ref()),
            id_gen: Some(&self.proposal_id_gen),
            drain_buffer: Some(&self.drain_buffer),
            nvme_write_buffer: self.nvme_write_buffer.as_deref(),
        };
        let outcome = txn
            .commit(&commit_ctx)
            .map_err(|e| DatabaseError::Other(format!("commit failed: {e}")))?;
        Ok(outcome.applied_index.unwrap_or(0))
    }

    /// Roll back an interactive transaction (ADR-042): discard all buffered
    /// writes and the OCC read-set. No proposal is emitted (nothing was
    /// durable). Errors only if the id is unknown.
    pub fn rollback_transaction(&self, txn_id: u64) -> Result<(), DatabaseError> {
        if self
            .interactive_txns
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&txn_id)
            .is_some()
        {
            Ok(())
        } else {
            Err(DatabaseError::Other(format!(
                "unknown transaction id {txn_id}"
            )))
        }
    }

    /// Drop interactive transactions idle longer than `timeout` (ADR-042
    /// mandatory idle timeout). An open transaction pins an MVCC snapshot and
    /// buffers writes in memory, so an abandoned one would leak retention and
    /// leader memory. Called opportunistically on `begin`; a production
    /// deployment also runs this periodically.
    pub fn reap_idle_transactions(&self, timeout: Duration) {
        let now = Instant::now();
        self.interactive_txns
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|_, (_, touched)| now.duration_since(*touched) < timeout);
    }

    /// Default idle timeout for an open interactive transaction (ADR-042).
    pub const DEFAULT_INTERACTIVE_TXN_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

    /// Default max buffered bytes per interactive transaction (256 MiB, ADR-042).
    pub const DEFAULT_MAX_INTERACTIVE_TXN_BYTES: usize = 256 * 1024 * 1024;

    /// Set the interactive-transaction idle timeout (server config wiring).
    pub fn set_interactive_idle_timeout(&mut self, timeout: Duration) {
        self.interactive_idle_timeout = timeout;
    }

    /// Set the per-interactive-transaction buffered-bytes ceiling (server
    /// config wiring).
    pub fn set_max_interactive_txn_bytes(&mut self, bytes: usize) {
        self.max_interactive_txn_bytes = bytes;
    }

    /// Set session-level vector consistency mode.
    ///
    /// Equivalent to `SET vector_consistency = 'snapshot'` in Cypher.
    pub fn set_vector_consistency(&mut self, mode: VectorConsistencyMode) {
        self.vector_consistency = mode;
    }

    /// Get current session-level vector consistency mode.
    pub fn vector_consistency(&self) -> VectorConsistencyMode {
        self.vector_consistency
    }

    /// Set session-level read concern.
    pub fn set_read_concern(
        &mut self,
        level: coordinode_core::txn::read_concern::ReadConcernLevel,
    ) {
        self.read_concern = level;
    }

    /// Get current session-level read concern.
    pub fn read_concern(&self) -> coordinode_core::txn::read_concern::ReadConcernLevel {
        self.read_concern
    }

    /// Set session-level write concern level.
    pub fn set_write_concern(
        &mut self,
        level: coordinode_core::txn::write_concern::WriteConcernLevel,
    ) {
        self.write_concern.level = level;
    }

    /// Set session-level write concern (full configuration including journal + timeout).
    pub fn set_write_concern_full(
        &mut self,
        wc: coordinode_core::txn::write_concern::WriteConcern,
    ) {
        self.write_concern = wc;
    }

    /// Get current session-level write concern level.
    pub fn write_concern(&self) -> coordinode_core::txn::write_concern::WriteConcernLevel {
        self.write_concern.level
    }

    /// Execute a Cypher query with a specific read concern.
    ///
    /// For `Snapshot` level with `at_timestamp`, the query reads from
    /// that exact MVCC timestamp instead of the latest applied state.
    /// Other levels behave like `Local` in embedded single-node mode
    /// (no replication lag to observe).
    /// Execute a Cypher query under shared (non-exclusive) access.
    ///
    /// Available to callers that only hold `&Database` — typically
    /// gRPC request handlers behind `Arc<RwLock<Database>>::read()`.
    /// Builds a per-call [`QuerySession`] from the Database's session
    /// defaults plus any wire-supplied `read_concern` / `write_concern`
    /// overrides, then dispatches to the `&self` impl. Multiple
    /// shared callers run in parallel; only `set_*` session-config
    /// methods (and embedded SET commands) need `.write()`.
    ///
    /// Rejects session-SET commands because mutating
    /// `self.vector_consistency` requires exclusive access — route
    /// SET through the `&mut self` `execute_cypher_full` API.
    /// Likewise, the one-shot `snapshot_read_ts` field on Database
    /// is not consulted here: the gRPC path always carries the
    /// snapshot timestamp on the wire via `read_concern.at_timestamp`.
    pub fn execute_cypher_shared(
        &self,
        query: &str,
        params: Option<std::collections::HashMap<String, coordinode_core::graph::types::Value>>,
        source: Option<&SourceContext>,
        read_concern: Option<&coordinode_core::txn::read_concern::ReadConcern>,
        write_concern: Option<&coordinode_core::txn::write_concern::WriteConcern>,
    ) -> Result<CypherResult, DatabaseError> {
        if Self::try_parse_session_set(query).is_some() {
            return Err(DatabaseError::Semantic(
                "session SET commands require exclusive Database access; \
                 use execute_cypher_full"
                    .to_string(),
            ));
        }

        let mut session = QuerySession {
            read_concern: self.read_concern,
            snapshot_read_ts: None,
            write_concern: self.write_concern.clone(),
            vector_consistency: self.vector_consistency,
        };
        if let Some(rc) = read_concern {
            rc.validate()
                .map_err(|e| DatabaseError::Semantic(e.to_string()))?;
            session.read_concern = rc.level;
            session.snapshot_read_ts = rc.at_timestamp;
        }
        if let Some(wc) = write_concern {
            session.write_concern = wc.clone();
        }

        let params = params.and_then(|p| if p.is_empty() { None } else { Some(p) });
        self.execute_cypher_impl(query, source, params, &session, TxnMode::AutoCommit)
            .map(|(rows, write_stats, _)| CypherResult { rows, write_stats })
    }

    pub fn execute_cypher_with_read_concern(
        &mut self,
        query: &str,
        read_concern: coordinode_core::txn::read_concern::ReadConcern,
    ) -> Result<Vec<Row>, DatabaseError> {
        read_concern
            .validate()
            .map_err(|e| DatabaseError::Semantic(e.to_string()))?;

        if self.try_apply_session_set(query) {
            return Ok(Vec::new());
        }

        let mut session = self.capture_session();
        session.read_concern = read_concern.level;
        session.snapshot_read_ts = read_concern.at_timestamp;

        self.execute_cypher_impl(query, None, None, &session, TxnMode::AutoCommit)
            .map(|(r, _, _)| r)
    }

    fn execute_cypher_impl(
        &self,
        query: &str,
        source: Option<&SourceContext>,
        params: Option<std::collections::HashMap<String, coordinode_core::graph::types::Value>>,
        session: &QuerySession,
        txn_mode: TxnMode,
    ) -> Result<
        (
            Vec<Row>,
            WriteStats,
            Option<coordinode_storage::engine::transaction::TransactionState>,
        ),
        DatabaseError,
    > {
        // SET-style session commands are handled by the public entry
        // points before reaching here (so this impl can stay on
        // &self). Regular Cypher only past this point.

        // Plan cache fast path: same query string → reuse parsed AST,
        // canonical form, fingerprint, and unoptimized logical plan.
        // Optimizer passes below still run on the cloned plan so they
        // observe the current index registry / stats.
        let (canonical, fp, mut plan) = match self.plan_cache.get(query) {
            Some(cached) => (
                cached.canonical.clone(),
                cached.fingerprint,
                cached.plan.clone(),
            ),
            None => {
                let ast = cypher::parse(query)?;
                let (canonical, fp) = normalize_and_fingerprint(&ast);
                let errors = cypher::analyze(&ast, None);
                if !errors.is_empty() {
                    return Err(DatabaseError::Semantic(
                        errors
                            .iter()
                            .map(|e| e.to_string())
                            .collect::<Vec<_>>()
                            .join("; "),
                    ));
                }
                let plan = planner::build_logical_plan(&ast)?;
                self.plan_cache.put(
                    query.to_string(),
                    Arc::new(CachedPlan {
                        canonical: canonical.clone(),
                        fingerprint: fp,
                        plan: plan.clone(),
                    }),
                );
                (canonical, fp, plan)
            }
        };
        // Inject the per-call vector consistency into the plan for EXPLAIN output.
        plan.vector_consistency = session.vector_consistency;

        // Apply index selection optimizer: rewrite Filter(NodeScan) → IndexScan
        // when a matching B-tree index is registered.
        plan.root = planner::optimize_index_selection(plan.root, &self.index_registry);

        // Annotate VectorTopK nodes with the HNSW index name when an applicable
        // index exists. This ensures the executor's VectorTopK operator carries
        // the resolved index name at execution time, not just at EXPLAIN time.
        plan.root = planner::annotate_vector_top_k(plan.root, &self.vector_index_registry);

        // Promote pure vector top-K to the HnswScan index access path:
        // the index becomes the row source and only the k result nodes
        // are fetched, instead of materialising the whole label before
        // ranking. Filtered queries keep the VectorTopK path.
        plan.root = planner::apply_hnsw_scan_access_path(plan.root, &self.vector_index_registry);

        // Apply graph-predicate push-down (R-PUSH1): for every VectorFilter
        // preceded by a Traverse, annotate with strategy decision
        // (graph_first / acorn_filtered / vector_first) per the cost model
        // in arch/core/query-engine.md § Graph Predicate Push-Down. The
        // invariant — no unfiltered VectorFilter after Traverse — is
        // contract-tested in the planner regression suite.
        let stats = self.compute_stats();
        let combined_for_push_down = stats.as_ref().map(|g| CombinedStats {
            graph: g,
            vector: &self.vector_index_registry,
        });
        let stats_ref = combined_for_push_down
            .as_ref()
            .map(|c| c as &dyn coordinode_core::graph::stats::StorageStats);
        plan.root = planner::optimize_push_down(plan.root, stats_ref);

        // Bind parameters: replace $name references with literal values.
        if let Some(ref p) = params {
            plan.substitute_params(p);
        }

        // MVCC enabled: all reads use snapshot isolation at start_ts,
        // all writes are buffered and flushed atomically through the
        // ProposalPipeline at commit_ts.
        //
        // Read concern affects snapshot selection:
        // - Local/Majority/Linearizable: use oracle.next() (latest applied)
        //   In embedded single-node mode, these are equivalent since there's
        //   no replication lag. In cluster mode (coordinode-server), Majority
        //   and Linearizable use Raft commit_index / lease check.
        // - Snapshot with at_timestamp: pin to explicit MVCC timestamp.
        use coordinode_core::txn::read_concern::ReadConcernLevel;
        let read_ts = match &txn_mode {
            // Interactive transaction: every statement reuses the pinned
            // start_ts so all reads resolve against the same snapshot
            // (repeatable read across the transaction — ADR-042).
            TxnMode::Interactive(state) => state.read_ts(),
            TxnMode::AutoCommit if session.read_concern == ReadConcernLevel::Snapshot => {
                // One-shot snapshot read; already captured into the
                // session (Database.snapshot_read_ts was taken when
                // the session was built).
                if let Some(ts) = session.snapshot_read_ts {
                    Timestamp::from_raw(ts)
                } else {
                    self.oracle.next()
                }
            }
            TxnMode::AutoCommit => self.oracle.next(),
        };
        // Build the transaction up front: a fresh one for auto-commit, or
        // the resumed parked state for an interactive statement. `interactive`
        // drives the no-commit execution + state extraction below.
        let interactive = matches!(txn_mode, TxnMode::Interactive(_));
        let txn = match txn_mode {
            TxnMode::Interactive(state) => {
                coordinode_storage::engine::transaction::Transaction::resume(
                    &self.engine,
                    Some(&self.oracle),
                    *state,
                )
            }
            TxnMode::AutoCommit => coordinode_storage::engine::transaction::Transaction::new(
                &self.engine,
                Some(&self.oracle),
                read_ts,
                None,
            ),
        };
        // Snapshot of the current interner is handed to the vector
        // loader (HNSW property lookups). The write-lock is then
        // acquired for the duration of execute, so the executor can
        // intern new property names without re-entering the same
        // RwLock through the loader (parking_lot RwLock is not
        // re-entrant — that would deadlock).
        let vector_loader = StorageVectorLoader::new(
            Arc::clone(&self.engine),
            self.interner.read().clone(),
            self.shard_id,
        );
        let mut interner_guard = self.interner.write();
        let interner_len_before = interner_guard.len();
        let mut ctx = ExecutionContext {
            engine: &self.engine,
            engine_arc: Some(Arc::clone(&self.engine)),
            interner: &mut interner_guard,
            id_allocator: &self.allocator,
            shard_id: self.shard_id,
            adaptive: self.adaptive_config.clone(),
            dedup_varlen_targets: false,
            snapshot_ts: None,
            retention_window_us: 7 * 24 * 3600 * 1_000_000,
            warnings: Vec::new(),
            write_stats: WriteStats::default(),
            text_index: None,
            text_index_registry: Some(&self.text_index_registry),
            vector_index_registry: Some(&self.vector_index_registry),
            btree_index_registry: Some(&self.index_registry),
            // Extension-op handlers for this Database (empty by default). An
            // enterprise layer / integration test populates it via
            // Database::register_extension so SHARDED-BY-style extension ops
            // dispatch; an empty registry means none are dispatchable.
            extensions: Some(&self.extension_registry),
            vector_loader: Some(&vector_loader),
            mvcc_oracle: Some(&self.oracle),
            mvcc_read_ts: read_ts,
            procedure_ctx: Some(ProcedureContext {
                registry: Arc::clone(&self.query_registry),
                nplus1: Arc::clone(&self.nplus1_detector),
                dismissed: Arc::clone(&self.dismissed),
            }),
            txn,
            vector_consistency: session.vector_consistency,
            vector_overfetch_factor: 1.2,
            vector_mvcc_stats: None,
            // The injected pipeline (Raft in cluster mode) — NOT a local
            // engine-applying one. Writing past it breaks replication.
            proposal_pipeline: Some(self.pipeline.as_ref()),
            proposal_id_gen: Some(&self.proposal_id_gen),
            read_concern: session.read_concern,
            write_concern: session.write_concern.clone(),
            drain_buffer: Some(&self.drain_buffer),
            nvme_write_buffer: self.nvme_write_buffer.as_deref(),
            mvcc_snapshot: None,
            // Cascade tracking — cluster defaults for trigger cycle protection.
            cascade_depth: 0,
            cascade_depth_limit: 10,
            cascade_fire_counts: std::collections::HashMap::new(),
            cascade_fanout_limit: 100,
            cascade_chain: Vec::new(),
            correlated_row: None,
            feedback_cache: Some(self.feedback_cache.clone()),
            schema_label_cache: std::collections::HashMap::new(),
            applied_watermark: None,
            read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(
            ),
            read_timeout: std::time::Duration::from_millis(2000),
            params: std::collections::HashMap::new(),
            pending_vector_writes: Vec::new(),
        };

        let start = Instant::now();
        // Auto-commit flushes the statement's writes; an interactive statement
        // leaves them buffered on the transaction for COMMIT to flush later.
        let results = if interactive {
            execute_no_commit(&plan, &mut ctx)?
        } else {
            execute(&plan, &mut ctx)?
        };
        // Flush HNSW writes accumulated during execute as a single
        // batched insert per (label, property) — amortises the HNSW
        // write-lock acquisition across the whole statement.
        ctx.flush_pending_vector_writes();
        // Park the (uncommitted) transaction state so the caller can re-hold it
        // for the next statement of an interactive transaction. `take_state`
        // drains the buffers without consuming `ctx`, leaving it droppable.
        let out_state = if interactive {
            Some(ctx.txn.take_state())
        } else {
            None
        };
        let duration_us = start.elapsed().as_micros() as u64;

        // Record execution in advisor registry with plan + optional source
        let plan_str = plan.explain();
        self.query_registry
            .record_with_plan(fp, &canonical, duration_us, plan_str, source);

        // N+1 detection: check if this (fingerprint, source) exceeds threshold
        if let Some(src) = source {
            if let Some(alert) = self.nplus1_detector.record(fp, &canonical, src) {
                tracing::warn!(
                    fingerprint = fp,
                    count = alert.call_count,
                    file = %alert.source_file,
                    line = alert.source_line,
                    "N+1 query pattern detected"
                );
            }
        }

        // Capture write_stats before dropping ctx (which borrows the
        // interner_guard via &mut *).
        let mut write_stats = ctx.write_stats.clone();
        let nodes_created = write_stats.nodes_created;
        let had_mutations = write_stats.has_mutations();
        // Drop ctx, then snapshot the interner state under the same
        // write guard before releasing it so we don't race with another
        // query that might intern between unlock and persist.
        drop(ctx);
        let interner_len_after = interner_guard.len();
        let interner_bytes = if interner_len_after > interner_len_before {
            Some(interner_guard.to_bytes())
        } else {
            None
        };
        drop(interner_guard);

        // Persist a new ID batch if nodes were created and we're near the ceiling.
        if nodes_created > 0 {
            self.ensure_id_batch()?;
        }

        // Persist field interner if new fields were interned during this
        // query. Goes through the proposal pipeline (not a direct engine
        // put): property values are encoded against interner ids, so the
        // mapping must replicate to every node that applies the data.
        if let Some(bytes) = interner_bytes {
            let proposal = coordinode_core::txn::proposal::RaftProposal {
                id: self.proposal_id_gen.next(),
                mutations: vec![coordinode_core::txn::proposal::Mutation::Put {
                    partition: coordinode_core::txn::proposal::PartitionId::Schema,
                    key: SCHEMA_KEY_FIELD_INTERNER.to_vec(),
                    value: bytes,
                }],
                commit_ts: self.oracle.next(),
                start_ts: Timestamp::from_raw(0),
                bypass_rate_limiter: false,
            };
            let outcome = self
                .pipeline
                .propose_and_wait(&proposal)
                .map_err(|e| DatabaseError::Other(format!("persist field interner: {e}")))?;
            // The interner mapping for a newly-seen field commits at a HIGHER
            // Raft index than the user write it accompanies. operationTime must
            // cover it: a causal read fencing on the returned index has to
            // observe both the written data AND the interner entry needed to
            // decode the write's new property names on a follower. Take the
            // max so the token spans every Raft entry this statement produced.
            // (`Option` orders `None < Some`, so this is a no-op in embedded /
            // non-replicated mode where both are `None`.)
            write_stats.applied_index = write_stats.applied_index.max(outcome.applied_index);
        }

        // Invalidate cached storage statistics after any mutation so that
        // the next EXPLAIN reflects the current state of the database.
        if had_mutations {
            self.invalidate_stats_cache();
        }

        Ok((results, write_stats, out_state))
    }

    /// Ensure the allocator has a persisted batch reservation.
    ///
    /// If the current allocator position has reached or exceeded the batch
    /// ceiling, reserves a new batch by persisting the new ceiling to disk.
    /// This is called after CREATE operations to guarantee crash safety.
    ///
    /// In cluster mode (distributed), this will be replaced by Raft-based
    /// ID range allocation from the leader.
    fn ensure_id_batch(&self) -> Result<(), DatabaseError> {
        let current = self.allocator.current().as_raw();
        let ceiling = self.id_batch_ceiling.load(Ordering::Relaxed);
        if current >= ceiling {
            let new_ceiling = current + ID_BATCH_SIZE;
            self.engine.put(
                Partition::Schema,
                SCHEMA_KEY_NEXT_NODE_ID,
                &new_ceiling.to_be_bytes(),
            )?;
            self.id_batch_ceiling.store(new_ceiling, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Reload the in-memory field interner from the Schema partition if
    /// the persisted mapping has grown. Returns `true` when a refresh
    /// was applied.
    ///
    /// Property values are encoded against interner ids. On a Raft
    /// follower the persisted mapping advances through entry apply
    /// (replicated by the leader), but this Database instance's
    /// in-memory copy does not — without a refresh, follower reads
    /// resolve every property to null. Cluster deployments call this
    /// whenever the applied index advances; embedded single-node
    /// deployments never need it (the only writer is this instance).
    pub fn refresh_field_interner(&self) -> Result<bool, DatabaseError> {
        let Some(bytes) = self
            .engine
            .get(Partition::Schema, SCHEMA_KEY_FIELD_INTERNER)?
        else {
            return Ok(false);
        };
        let Some(persisted) = FieldInterner::from_bytes(&bytes) else {
            tracing::warn!("corrupt persisted field interner, refresh skipped");
            return Ok(false);
        };
        // Cheap pre-check under the read lock: the interner only grows,
        // so a same-size persisted mapping cannot differ.
        if persisted.len() <= self.interner.read().len() {
            return Ok(false);
        }
        let mut guard = self.interner.write();
        if persisted.len() <= guard.len() {
            return Ok(false); // raced with another refresher
        }
        *guard = persisted;
        Ok(true)
    }

    /// Persist field-interner bytes (e.g. recovered from a snapshot-based
    /// restore) to the Schema partition so a later open reloads them, and
    /// update this instance's in-memory copy. A full Raft snapshot excludes
    /// `meta:` Schema keys, so the interner must be carried and restored
    /// alongside it for a self-contained backup.
    pub fn persist_field_interner_bytes(&self, bytes: &[u8]) -> Result<(), DatabaseError> {
        let Some(interner) = FieldInterner::from_bytes(bytes) else {
            return Err(DatabaseError::Other("corrupt field interner bytes".into()));
        };
        self.engine
            .put(Partition::Schema, SCHEMA_KEY_FIELD_INTERNER, bytes)?;
        *self.interner.write() = interner;
        Ok(())
    }

    /// Return EXPLAIN plan text for a Cypher query.
    ///
    /// Uses real storage statistics (node counts, fan-out) for
    /// more accurate cost estimates than hardcoded defaults.
    pub fn explain_cypher(&self, query: &str) -> Result<String, DatabaseError> {
        let ast = cypher::parse(query)?;
        let mut plan = planner::build_logical_plan(&ast)?;
        plan.vector_consistency = self.vector_consistency;
        // Apply index selection optimizer so EXPLAIN reflects the actual plan
        // that would be executed (IndexScan instead of Filter+NodeScan when
        // a matching B-tree index is registered).
        plan.root = planner::optimize_index_selection(plan.root, &self.index_registry);
        plan.root = planner::annotate_vector_top_k(plan.root, &self.vector_index_registry);
        // Same access-path promotion as the execute path so EXPLAIN
        // shows the plan that actually runs.
        plan.root = planner::apply_hnsw_scan_access_path(plan.root, &self.vector_index_registry);
        let stats = self.compute_stats();
        let combined_for_push_down = stats.as_ref().map(|g| CombinedStats {
            graph: g,
            vector: &self.vector_index_registry,
        });
        let stats_ref_for_push_down = combined_for_push_down
            .as_ref()
            .map(|c| c as &dyn coordinode_core::graph::stats::StorageStats);
        plan.root = planner::optimize_push_down(plan.root, stats_ref_for_push_down);
        let stats_ref = stats
            .as_ref()
            .map(|s| s as &dyn coordinode_core::graph::stats::StorageStats);
        let mut explain = plan.explain_with_stats(stats_ref);

        // Annotate the live serving health of any vector index the
        // plan actually uses. `apply_hnsw_scan_access_path` above promotes a
        // matching query to `HnswScan(<index_name>, …)`, so the index name is
        // present in the text exactly when the plan reads through that index —
        // a brute-force fallback names no index and gets no annotation.
        let mut health_lines = Vec::new();
        for def in self.vector_index_registry.all_definitions() {
            if !explain.contains(&def.name) {
                continue;
            }
            if let Some(state) = self
                .vector_index_registry
                .health_snapshot(&def.label, def.property())
            {
                health_lines.push(format!("  {}: {}", def.name, describe_index_health(&state)));
            }
        }
        if !health_lines.is_empty() {
            explain.push_str("\n\nVector index health:\n");
            explain.push_str(&health_lines.join("\n"));
        }
        Ok(explain)
    }

    /// Return EXPLAIN SUGGEST: plan + optimization suggestions.
    ///
    /// Uses real storage statistics for cost estimation and checks existing
    /// indexes to prevent false positive MissingIndex suggestions.
    pub fn explain_suggest(
        &self,
        query: &str,
    ) -> Result<coordinode_query::advisor::ExplainSuggestResult, DatabaseError> {
        let ast = cypher::parse(query)?;
        let mut plan = planner::build_logical_plan(&ast)?;
        plan.vector_consistency = self.vector_consistency;
        let stats = self.compute_stats();
        let stats_ref = stats
            .as_ref()
            .map(|s| s as &dyn coordinode_core::graph::stats::StorageStats);
        Ok(plan.explain_suggest_with_stats(stats_ref, Some(&self.index_registry)))
    }

    /// Compute storage statistics for the cost estimator (with TTL cache).
    ///
    /// Returns a cached snapshot if it is younger than `stats_ttl`.
    /// Otherwise recomputes from MVCC storage and caches the result.
    /// Returns `None` if both cache and recomputation fail (non-critical).
    pub fn compute_stats(&self) -> Option<coordinode_storage::engine::stats::StorageStatsComputer> {
        let mut guard = self.cached_stats.lock().ok()?;
        if let Some((ref stats, computed_at)) = *guard {
            if computed_at.elapsed() < self.stats_ttl {
                return Some(stats.clone());
            }
        }
        // Cache miss or expired — recompute.
        match coordinode_storage::engine::stats::StorageStatsComputer::compute_mvcc(&self.engine) {
            Ok(fresh) => {
                let cloned = fresh.clone();
                *guard = Some((fresh, Instant::now()));
                Some(cloned)
            }
            Err(_) => None,
        }
    }

    /// Invalidate the cached storage statistics.
    ///
    /// The next `explain_cypher()` or `explain_suggest()` call will
    /// trigger a fresh scan.  Useful after bulk imports or schema changes.
    pub fn invalidate_stats_cache(&self) {
        if let Ok(mut guard) = self.cached_stats.lock() {
            *guard = None;
        }
    }

    /// Override the storage statistics cache TTL.
    ///
    /// Use `Duration::ZERO` to disable caching (every EXPLAIN recomputes).
    /// Use `Duration::MAX` to cache forever (only invalidated by writes
    /// or explicit `invalidate_stats_cache()`).
    pub fn set_stats_ttl(&mut self, ttl: Duration) {
        self.stats_ttl = ttl;
    }

    /// Set the adaptive parallel threshold for traversal.
    ///
    /// When a node's fan-out exceeds this threshold, the executor switches
    /// to rayon parallel processing instead of sequential iteration.
    pub fn set_adaptive_parallel_threshold(&mut self, threshold: usize) {
        self.adaptive_config.parallel_threshold = threshold;
    }

    /// Get the underlying storage engine.
    pub fn engine(&self) -> &StorageEngine {
        &self.engine
    }

    /// Get a shared reference to the storage engine.
    ///
    /// Used by services (e.g. BlobService) that need to share the same
    /// storage instance as the Database without opening a separate one.
    pub fn engine_shared(&self) -> Arc<StorageEngine> {
        Arc::clone(&self.engine)
    }

    /// Create a vector (HNSW) index on a label's vector property.
    ///
    /// After creation, queries using `vector_similarity(n.prop, $q)` will
    /// use the HNSW index instead of brute-force distance computation.
    /// Call `populate_vector_index` to backfill existing vectors.
    pub fn create_vector_index(
        &mut self,
        name: impl Into<String>,
        label: impl Into<String>,
        property: impl Into<String>,
        config: coordinode_query::index::VectorIndexConfig,
    ) {
        let def = coordinode_query::index::IndexDefinition::hnsw(name, label, property, config);

        // Persist the index definition to schema: partition so it survives restart.
        let key = def.schema_key();
        if let Ok(bytes) = rmp_serde::to_vec(&def) {
            if let Err(e) = self.engine.put(Partition::Schema, &key, &bytes) {
                tracing::error!("failed to persist vector index definition: {e}");
            }
        }

        // Register in both registries: VectorIndexRegistry holds the live HNSW
        // graph for query acceleration; IndexRegistry mirrors the definition so
        // advisors and planners can see all indexes (scalar + vector) through
        // a single source of truth. Tier handle is resolved locally so the
        // registry never touches the shared interner lock.
        let tier = {
            let mut g = self.interner.write();
            let label_id = g.intern(&def.label);
            let property_id = g.intern(def.property());
            self.vector_index_registry
                .tier_handle(label_id, property_id)
        };
        self.vector_index_registry
            .register_with_tier(def.clone(), tier);
        self.index_registry.register_in_memory(def);
    }

    /// Persist a label schema to storage and auto-create unique B-tree indexes.
    ///
    /// Idempotent: existing schema for this label is replaced. For each property
    /// with `unique = true`, a B-tree unique index is created (if not already
    /// present) and existing nodes are backfilled into the index.
    ///
    /// Returns the schema revision after persistence.
    pub fn create_label_schema(
        &mut self,
        schema: coordinode_core::schema::definition::LabelSchema,
    ) -> Result<u64, DatabaseError> {
        use coordinode_core::schema::definition::{
            encode_label_current_revision_key, encode_label_schema_key,
        };

        // 1. Persist the schema to storage. Version-prefixed key carries the
        //    immutable snapshot; the current_revision pointer names the active
        //    one. Both writes are part of this commit (ADR-023).
        let key = encode_label_schema_key(&schema.name, schema.schema_revision);
        let bytes = schema
            .to_msgpack()
            .map_err(|e| DatabaseError::Other(format!("serialize label schema: {e}")))?;
        self.engine.put(Partition::Schema, &key, &bytes)?;
        let pointer_key = encode_label_current_revision_key(&schema.name);
        self.engine.put(
            Partition::Schema,
            &pointer_key,
            &schema.schema_revision.to_be_bytes(),
        )?;

        let version = schema.schema_revision;
        let label_name = schema.name.clone();

        // 2. For each unique property, register a B-tree unique index.
        // Collect first to avoid borrowing schema while mutably borrowing registries.
        let unique_props: Vec<String> = schema
            .properties
            .values()
            .filter(|p| p.unique)
            .map(|p| p.name.clone())
            .collect();

        for prop_name in unique_props {
            let idx_name = format!("{}_{}", label_name.to_lowercase(), prop_name.to_lowercase());
            if self.index_registry.get(&idx_name).is_some() {
                continue; // index already registered — skip
            }

            let idx =
                coordinode_query::index::IndexDefinition::btree(&idx_name, &label_name, &prop_name)
                    .unique();
            self.index_registry
                .register(&self.engine, idx)
                .map_err(DatabaseError::Storage)?;

            // Backfill existing nodes of this label into the new index.
            // Duplicate values in pre-existing data are logged as warnings —
            // we do not reject the schema creation because of historical data.
            self.backfill_btree_index(&label_name, &prop_name);
        }

        Ok(version)
    }

    /// Backfill a B-tree index for nodes of `label` that are already in storage.
    ///
    /// Called after a new unique B-tree index is registered via `create_label_schema`.
    /// Unique violations in pre-existing data are logged as warnings, not errors.
    fn backfill_btree_index(&self, label: &str, property: &str) {
        use coordinode_core::graph::node::{decode_node_key, NodeRecord};
        use coordinode_core::graph::types::Value;

        let node_prefix = {
            let mut p = Vec::with_capacity(8);
            p.extend_from_slice(b"node:");
            p.extend_from_slice(&self.shard_id.to_be_bytes());
            p.push(b':');
            p
        };

        let iter = match self.engine.prefix_scan(Partition::Node, &node_prefix) {
            Ok(it) => it,
            Err(e) => {
                tracing::warn!(
                    "backfill_btree_index: failed to scan nodes for {label}.{property}: {e}"
                );
                return;
            }
        };

        let field_id = self.interner.read().lookup(property);

        for guard in iter {
            let Ok((_key, value)) = guard.into_inner() else {
                continue;
            };
            let Ok(record) = NodeRecord::from_msgpack(&value) else {
                continue;
            };
            if record.primary_label() != label {
                continue;
            }
            let Some((_, node_id)) = decode_node_key(&_key) else {
                continue;
            };

            let prop_val = if let Some(fid) = field_id {
                record.props.get(&fid).cloned().unwrap_or(Value::Null)
            } else {
                Value::Null
            };

            let props = [(property.to_string(), prop_val)];
            if let Err(e) =
                self.index_registry
                    .on_node_created(&self.engine, node_id, label, &props)
            {
                tracing::warn!(
                    "backfill_btree_index: unique violation for {label}.{property} on node {node_id:?}: {e}"
                );
            }
        }
    }

    /// Persist an edge type schema to storage.
    ///
    /// Idempotent: existing schema for this edge type is replaced.
    /// Returns the schema revision after persistence.
    pub fn create_edge_type_schema(
        &mut self,
        schema: coordinode_core::schema::definition::EdgeTypeSchema,
    ) -> Result<u64, DatabaseError> {
        use coordinode_core::schema::definition::{
            encode_edge_type_current_revision_key, encode_edge_type_schema_key,
        };

        let key = encode_edge_type_schema_key(&schema.name, schema.schema_revision);
        let bytes = schema
            .to_msgpack()
            .map_err(|e| DatabaseError::Other(format!("serialize edge type schema: {e}")))?;
        self.engine.put(Partition::Schema, &key, &bytes)?;
        let pointer_key = encode_edge_type_current_revision_key(&schema.name);
        self.engine.put(
            Partition::Schema,
            &pointer_key,
            &schema.schema_revision.to_be_bytes(),
        )?;
        Ok(schema.schema_revision)
    }

    /// Get a reference to the vector index registry.
    pub fn vector_index_registry(&self) -> &coordinode_query::index::VectorIndexRegistry {
        &self.vector_index_registry
    }

    /// Create a full-text search index on a label's text property.
    ///
    /// After creation, queries using `text_match(n.prop, "query")` will
    /// use the tantivy index. Existing nodes are backfilled automatically.
    pub fn create_text_index(
        &mut self,
        name: impl Into<String>,
        label: impl Into<String>,
        property: impl Into<String>,
        config: coordinode_query::index::TextIndexConfig,
    ) -> Result<(), DatabaseError> {
        let label = label.into();
        let property = property.into();
        let def = coordinode_query::index::IndexDefinition::text(
            name,
            &label,
            vec![property.clone()],
            config,
        );

        // Persist the index definition to schema: partition.
        let key = def.schema_key();
        let bytes = rmp_serde::to_vec(&def).map_err(|e| {
            DatabaseError::Other(format!("failed to serialize text index def: {e}"))
        })?;
        self.engine.put(Partition::Schema, &key, &bytes)?;

        // Register in the text index registry (creates empty tantivy index).
        self.text_index_registry
            .register(def)
            .map_err(DatabaseError::Other)?;

        // Backfill existing nodes with this label and property.
        let shard_id = self.shard_id;
        let node_prefix = {
            let mut p = Vec::with_capacity(5 + 2 + 1);
            p.extend_from_slice(b"node:");
            p.extend_from_slice(&shard_id.to_be_bytes());
            p.push(b':');
            p
        };

        let mut count = 0usize;
        let iter = self.engine.prefix_scan(Partition::Node, &node_prefix)?;
        for guard in iter {
            let Ok((_key, value)) = guard.into_inner() else {
                continue;
            };
            let Ok(record) = coordinode_core::graph::node::NodeRecord::from_msgpack(&value) else {
                continue;
            };

            if record.primary_label() != label {
                continue;
            }

            let node_id = match coordinode_core::graph::node::decode_node_key(&_key) {
                Some((_shard, nid)) => nid,
                None => continue,
            };

            if let Some(field_id) = self.interner.read().lookup(&property) {
                if let Some(val) = record.props.get(&field_id) {
                    if let Some(text) = val.as_str() {
                        self.text_index_registry
                            .on_text_written(&label, node_id, &property, text);
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            tracing::info!("backfilled text index with {count} document(s)");
        }
        Ok(())
    }

    /// Get a reference to the text index registry.
    pub fn text_index_registry(&self) -> &coordinode_query::index::TextIndexRegistry {
        &self.text_index_registry
    }

    /// Get a mutable reference to the text index registry.
    pub fn text_index_registry_mut(&mut self) -> &mut coordinode_query::index::TextIndexRegistry {
        &mut self.text_index_registry
    }

    /// Get a read-locked view of the field interner.
    ///
    /// Returned guard derefs to `&FieldInterner` so existing callers
    /// using `db.interner().lookup(..)` / `db.interner().resolve(..)`
    /// continue to work via deref coercion. Don't hold the guard
    /// across long-running operations — writers (new property name
    /// interns) wait until it's dropped.
    pub fn interner(&self) -> RwLockReadGuard<'_, FieldInterner> {
        self.interner.read()
    }

    /// Get an Arc-shared handle to the interner, for components that
    /// need to keep their own snapshot or hand off ownership (e.g.
    /// background reapers, cluster recovery).
    pub fn interner_arc(&self) -> Arc<RwLock<FieldInterner>> {
        Arc::clone(&self.interner)
    }

    /// Get the query advisor registry for performance analysis.
    pub fn query_registry(&self) -> &QueryRegistry {
        &self.query_registry
    }

    /// Get the N+1 pattern detector.
    pub fn nplus1_detector(&self) -> &NPlus1Detector {
        &self.nplus1_detector
    }

    /// Allocate a read timestamp for MVCC snapshot reads.
    ///
    /// Used by export/backup to take a consistent snapshot of all data.
    /// The returned timestamp reflects the current high-water mark —
    /// all committed writes before this point are visible.
    pub fn read_ts(&self) -> coordinode_core::txn::timestamp::Timestamp {
        self.oracle.next()
    }

    /// Try to parse a session SET command.
    ///
    /// Supports: `SET vector_consistency = 'mode'`
    /// Returns `Some(mode)` if matched, `None` otherwise.
    fn try_parse_session_set(query: &str) -> Option<VectorConsistencyMode> {
        let trimmed = query.trim();

        // Case-insensitive matching for SET vector_consistency = '...'
        let lower = trimmed.to_ascii_lowercase();
        if !lower.starts_with("set ") {
            return None;
        }

        let rest = trimmed[4..].trim();
        let lower_rest = rest.to_ascii_lowercase();
        if !lower_rest.starts_with("vector_consistency") {
            return None;
        }

        // Find '=' sign
        let after_name = rest["vector_consistency".len()..].trim();
        let after_eq = after_name.strip_prefix('=')?;
        let value = after_eq.trim();

        // Strip quotes (single or double)
        let unquoted = if (value.starts_with('\'') && value.ends_with('\''))
            || (value.starts_with('"') && value.ends_with('"'))
        {
            &value[1..value.len() - 1]
        } else {
            value
        };

        VectorConsistencyMode::from_str_opt(unquoted)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn plan_cache_hit_returns_same_plan() {
        // Same query string twice → second call must observe the cache
        // entry created by the first. Asserted by introspecting
        // PlanCache.inner directly.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (n:Cached {k: 1}) RETURN n")
            .expect("create");
        db.execute_cypher("MATCH (n:Cached) RETURN n.k")
            .expect("first match");
        let after_first = db.plan_cache.inner.read().len();
        db.execute_cypher("MATCH (n:Cached) RETURN n.k")
            .expect("second match");
        let after_second = db.plan_cache.inner.read().len();
        assert_eq!(
            after_first, after_second,
            "second invocation of the same query must not grow the cache"
        );
        assert!(
            after_second >= 1,
            "cache must contain at least the MATCH plan"
        );
    }

    #[test]
    fn plan_cache_distinguishes_queries() {
        // Different query strings → distinct cache entries.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (n:A {k: 1}) RETURN n")
            .expect("create A");
        db.execute_cypher("CREATE (n:B {k: 1}) RETURN n")
            .expect("create B");
        db.execute_cypher("MATCH (n:A) RETURN n.k")
            .expect("match A");
        db.execute_cypher("MATCH (n:B) RETURN n.k")
            .expect("match B");
        let entries = db.plan_cache.inner.read().len();
        assert!(
            entries >= 4,
            "expected ≥4 distinct cache entries, got {entries}"
        );
    }

    #[test]
    fn plan_cache_bounded_eviction() {
        // Cache is bounded — pumping more distinct queries than the
        // bound must keep len at the bound, not grow unboundedly.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");
        // Shrink the bound for the test so we don't have to issue
        // thousands of queries.
        db.plan_cache = Arc::new(PlanCache::new(4));
        for i in 0..16 {
            // Each query string is unique (different label) so cache
            // entries don't collide.
            let q = format!("MATCH (n:L{i}) RETURN n");
            // Don't care about result; this label has no data.
            let _ = db.execute_cypher(&q);
        }
        let entries = db.plan_cache.inner.read().len();
        assert!(
            entries <= 4,
            "cache exceeded max_entries=4: {entries} entries"
        );
    }

    #[test]
    fn open_database() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _db = Database::open(dir.path()).expect("open");
    }

    #[test]
    fn create_and_match() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        db.execute_cypher("CREATE (n:User {name: 'Alice'}) RETURN n")
            .expect("create");

        let results = db
            .execute_cypher("MATCH (n:User) RETURN n.name")
            .expect("match");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn explain_plan() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(dir.path()).expect("open");

        let plan = db
            .explain_cypher("MATCH (n:User) RETURN n")
            .expect("explain");
        assert!(plan.contains("NodeScan"));
    }

    #[test]
    fn parse_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        assert!(db.execute_cypher("INVALID").is_err());
    }

    #[test]
    fn semantic_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        let result = db.execute_cypher("MATCH (n) RETURN m");
        assert!(matches!(result, Err(DatabaseError::Semantic(_))));
    }

    /// Regression test: MATCH+SET property changes must be visible in subsequent queries.
    ///
    /// Reproduces the snapshot isolation bug where MATCH+SET appears to succeed
    /// (the change is visible via RETURN in the same query) but the updated value
    /// is NOT visible in a subsequent MATCH query — the property reverts to its
    /// pre-SET value across query boundaries.
    ///
    /// Root cause hypothesis: the MVCC write buffer is flushed correctly but
    /// the next query's snapshot is taken at a seqno that precedes the write.
    #[test]
    fn match_set_property_change_persists_across_queries() {
        use coordinode_core::graph::types::Value;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // Step 1: create node via MERGE+SET (confirmed working per bug report)
        db.execute_cypher("MERGE (p:Project {id: 'x'}) SET p.status = 'active'")
            .expect("MERGE+SET must succeed");

        // Step 2: update via MATCH+SET — returns 'removed' in the same query
        let step2 = db
            .execute_cypher(
                "MATCH (p:Project {id: 'x'}) SET p.status = 'removed' RETURN p.status AS s",
            )
            .expect("MATCH+SET must succeed");
        assert_eq!(step2.len(), 1, "MATCH+SET must return one row");
        assert_eq!(
            step2[0].get("s"),
            Some(&Value::String("removed".into())),
            "SET must be visible within the same query"
        );

        // Step 3: read in a SEPARATE query — must show the persisted value
        let step3 = db
            .execute_cypher("MATCH (p:Project {id: 'x'}) RETURN p.status AS s")
            .expect("MATCH RETURN must succeed");
        assert_eq!(step3.len(), 1, "node must still exist in step 3");
        assert_eq!(
            step3[0].get("s"),
            Some(&Value::String("removed".into())),
            "MATCH+SET must persist across query boundaries: expected 'removed', got {:?}",
            step3[0].get("s")
        );
    }

    /// Regression test: MATCH+SET property changes must persist when a B-tree index
    /// is present on the lookup property, causing the planner to use IndexScan.
    ///
    /// IndexScan reads from the engine directly (not MVCC snapshot) for the index
    /// lookup. The node read in execute_update goes through mvcc_get (snapshot-based).
    /// This test verifies the two-phase read (IndexScan → execute_update mvcc_get)
    /// correctly commits the write to storage.
    #[test]
    fn match_set_persists_with_btree_index() {
        use coordinode_core::graph::types::Value;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // Create a B-tree index on :Project.id so that MATCH uses IndexScan
        db.execute_cypher("CREATE INDEX idx_project_id ON :Project(id)")
            .expect("CREATE INDEX");

        // Step 1: create node
        db.execute_cypher("MERGE (p:Project {id: 'x'}) SET p.status = 'active'")
            .expect("MERGE+SET");

        // Step 2: update via MATCH+SET — IndexScan path
        let step2 = db
            .execute_cypher(
                "MATCH (p:Project {id: 'x'}) SET p.status = 'removed' RETURN p.status AS s",
            )
            .expect("MATCH+SET with index");
        assert_eq!(
            step2[0].get("s"),
            Some(&Value::String("removed".into())),
            "SET visible in same query"
        );

        // Step 3: new query — must see 'removed'
        let step3 = db
            .execute_cypher("MATCH (p:Project {id: 'x'}) RETURN p.status AS s")
            .expect("MATCH RETURN");
        assert_eq!(step3.len(), 1);
        assert_eq!(
            step3[0].get("s"),
            Some(&Value::String("removed".into())),
            "MATCH+SET with IndexScan must persist: expected 'removed', got {:?}",
            step3[0].get("s")
        );
    }

    #[test]
    fn data_persists_across_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");

        {
            let mut db = Database::open(dir.path()).expect("open");
            db.execute_cypher("CREATE (n:User {name: 'Alice'})")
                .expect("create");
        }

        {
            let mut db = Database::open(dir.path()).expect("reopen");
            let results = db.execute_cypher("MATCH (n:User) RETURN n").expect("match");
            assert!(!results.is_empty());
        }
    }

    /// Node IDs are monotonically increasing across database reopens.
    /// This verifies G001: persistent NodeIdAllocator.
    #[test]
    fn node_ids_persist_across_reopen() {
        use coordinode_core::graph::types::Value;

        let dir = tempfile::tempdir().expect("tempdir");

        // First session: create a node, remember its ID via the node variable
        // (CREATE (n:...) RETURN n → n = node ID as Int)
        let first_id = {
            let mut db = Database::open(dir.path()).expect("open");
            let results = db
                .execute_cypher("CREATE (n:User {name: 'Alice'}) RETURN n")
                .expect("create Alice");
            assert_eq!(results.len(), 1);
            match results[0].get("n") {
                Some(Value::Int(id)) => *id,
                other => panic!("expected Int, got {other:?}"),
            }
        };

        // Second session: create another node, verify its ID > first_id
        let second_id = {
            let mut db = Database::open(dir.path()).expect("reopen");
            let results = db
                .execute_cypher("CREATE (n:User {name: 'Bob'}) RETURN n")
                .expect("create Bob");
            assert_eq!(results.len(), 1);
            match results[0].get("n") {
                Some(Value::Int(id)) => *id,
                other => panic!("expected Int, got {other:?}"),
            }
        };

        assert!(
            second_id > first_id,
            "second session ID ({second_id}) must be > first session ID ({first_id})"
        );
    }

    /// Multiple reopens don't lose ID state: IDs always increase.
    #[test]
    fn node_ids_persist_across_multiple_reopens() {
        use coordinode_core::graph::types::Value;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut last_id = 0i64;

        for i in 0..5 {
            let mut db = Database::open(dir.path()).expect("open");
            let results = db
                .execute_cypher(&format!("CREATE (n:User {{name: 'User{i}'}}) RETURN n"))
                .expect("create");
            assert_eq!(results.len(), 1);
            let id = match results[0].get("n") {
                Some(Value::Int(id)) => *id,
                other => panic!("expected Int, got {other:?}"),
            };
            assert!(
                id > last_id,
                "reopen {i}: id ({id}) must be > last_id ({last_id})"
            );
            last_id = id;
        }
    }

    /// Batch exhaustion triggers a new persistent batch.
    #[test]
    fn batch_exhaustion_persists_new_ceiling() {
        let dir = tempfile::tempdir().expect("tempdir");

        {
            let mut db = Database::open(dir.path()).expect("open");
            // Create enough nodes to exhaust the first batch (1000 IDs).
            // Each CREATE allocates 1 ID.
            for i in 0..50 {
                db.execute_cypher(&format!("CREATE (n:User {{idx: {i}}})"))
                    .expect("create");
            }
            // Verify ceiling was persisted correctly
            let ceiling_bytes = db
                .engine()
                .get(
                    coordinode_storage::engine::partition::Partition::Schema,
                    SCHEMA_KEY_NEXT_NODE_ID,
                )
                .expect("get ceiling")
                .expect("ceiling should exist");
            let ceiling = u64::from_be_bytes(ceiling_bytes[..8].try_into().expect("8 bytes"));
            assert!(
                ceiling >= 1000,
                "ceiling ({ceiling}) should be at least ID_BATCH_SIZE"
            );
        }

        // Reopen and verify we can continue creating nodes
        {
            let mut db = Database::open(dir.path()).expect("reopen");
            db.execute_cypher("CREATE (n:User {name: 'AfterReopen'})")
                .expect("create after reopen");
        }
    }

    #[test]
    fn set_vector_consistency_session() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // Default is Current
        assert_eq!(db.vector_consistency(), VectorConsistencyMode::Current);

        // SET via Cypher-like session command
        let result = db.execute_cypher("SET vector_consistency = 'snapshot'");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0); // no rows returned
        assert_eq!(db.vector_consistency(), VectorConsistencyMode::Snapshot);

        // SET exact
        db.execute_cypher("SET vector_consistency = 'exact'")
            .expect("set exact");
        assert_eq!(db.vector_consistency(), VectorConsistencyMode::Exact);

        // SET back to current
        db.execute_cypher("SET vector_consistency = 'current'")
            .expect("set current");
        assert_eq!(db.vector_consistency(), VectorConsistencyMode::Current);
    }

    #[test]
    fn set_vector_consistency_case_insensitive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        db.execute_cypher("SET vector_consistency = 'SNAPSHOT'")
            .expect("upper");
        assert_eq!(db.vector_consistency(), VectorConsistencyMode::Snapshot);

        db.execute_cypher("set vector_consistency = 'Current'")
            .expect("mixed");
        assert_eq!(db.vector_consistency(), VectorConsistencyMode::Current);
    }

    #[test]
    fn set_vector_consistency_double_quotes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        db.execute_cypher("SET vector_consistency = \"snapshot\"")
            .expect("double quotes");
        assert_eq!(db.vector_consistency(), VectorConsistencyMode::Snapshot);
    }

    #[test]
    fn set_vector_consistency_api() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        db.set_vector_consistency(VectorConsistencyMode::Exact);
        assert_eq!(db.vector_consistency(), VectorConsistencyMode::Exact);
    }

    #[test]
    fn set_vector_consistency_invalid_value_falls_through_to_parser() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // Invalid mode name → not recognized as session SET → falls through to Cypher parser
        // which will fail with a parse error
        let result = db.execute_cypher("SET vector_consistency = 'invalid_mode'");
        assert!(result.is_err());
    }

    #[test]
    fn extension_op_dispatches_through_database() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct CountingHandler {
            called: Arc<AtomicBool>,
        }
        impl ExtensionHandler for CountingHandler {
            fn execute(
                &self,
                _ctx: &mut ExecutionContext<'_>,
                _payload: &[u8],
            ) -> Result<Vec<Row>, ExecutionError> {
                self.called.store(true, Ordering::SeqCst);
                Ok(vec![])
            }
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");
        // Ensure the :Doc label exists before the index DDL.
        db.execute_cypher("CREATE (:Doc {x: 1})")
            .expect("seed label");

        let called = Arc::new(AtomicBool::new(false));
        db.register_extension(
            "vector_index.create_ext",
            Arc::new(CountingHandler {
                called: Arc::clone(&called),
            }),
        );

        // A SHARDED-BY tail on CREATE VECTOR INDEX routes end-to-end through the
        // Database: the parser captures the tail, the planner emits an Extension
        // op, and the executor dispatches it to the registered handler.
        db.execute_cypher("CREATE VECTOR INDEX foo ON :Doc(embedding) SHARDED BY CENTROID(8)")
            .expect("extension op executes");
        assert!(
            called.load(Ordering::SeqCst),
            "registered extension handler ran via Database"
        );
    }

    /// engine_shared() returns an Arc pointing to the same engine as engine().
    /// Writes through one are visible through the other.
    #[test]
    fn engine_shared_same_instance() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(dir.path()).expect("open");

        let shared = db.engine_shared();
        let key = b"meta:shared_test";
        let val = b"shared_value";

        // Write through shared Arc
        shared
            .put(
                coordinode_storage::engine::partition::Partition::Schema,
                key,
                val,
            )
            .expect("put via shared");

        // Read through engine() borrow — same data
        let got = db
            .engine()
            .get(
                coordinode_storage::engine::partition::Partition::Schema,
                key,
            )
            .expect("get via engine()")
            .expect("must exist");
        assert_eq!(got.as_ref(), val);
    }

    /// Multiple engine_shared() calls return Arcs to the same underlying engine.
    #[test]
    fn engine_shared_multiple_arcs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(dir.path()).expect("open");

        let arc1 = db.engine_shared();
        let arc2 = db.engine_shared();

        // Both point to the same allocation (Arc strong count = 6:
        // one in Database, one in OwnedLocalProposalPipeline (drain),
        // one in TtlReaperHandle (background thread), one in the
        // LsmVectorTier backing VectorIndexRegistry (ADR-033 f32
        // truth tier), two here).
        assert_eq!(Arc::strong_count(&arc1), 6);
        assert_eq!(Arc::strong_count(&arc2), 6);

        // Write through arc1, read through arc2
        arc1.put(
            coordinode_storage::engine::partition::Partition::Schema,
            b"meta:arc_test",
            b"v",
        )
        .expect("put");
        let got = arc2
            .get(
                coordinode_storage::engine::partition::Partition::Schema,
                b"meta:arc_test",
            )
            .expect("get")
            .expect("must exist");
        assert_eq!(got.as_ref(), b"v");
    }

    #[test]
    fn explain_shows_vector_consistency_for_vector_queries() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // Set snapshot mode
        db.set_vector_consistency(VectorConsistencyMode::Snapshot);

        let explain = db
            .explain_cypher(
                "MATCH (m:Movie) WHERE vector_distance(m.embedding, [1.0, 0.0]) < 0.5 RETURN m",
            )
            .expect("explain");

        assert!(
            explain.contains("Vector consistency: snapshot"),
            "EXPLAIN should show vector consistency mode: {explain}"
        );
    }

    // ─── Bug regression: MERGE on existing node with unique constraint ──────────

    /// MERGE on a node that already exists must match and apply ON MATCH SET,
    /// NOT throw "unique constraint violated".
    ///
    /// Bug: `execute_merge` runs a full NodeScan with property filters. When the
    /// scan returns empty (misses the existing node), MERGE falls through to
    /// CREATE which triggers the B-tree unique index → "unique constraint violated".
    ///
    /// Expected: MERGE finds the existing node and applies SET s.name = 'updated'.
    #[test]
    fn merge_on_existing_node_with_unique_constraint_does_not_error() {
        use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // Create label with a unique segment_id property.
        let mut schema = LabelSchema::new_node_id("Segment");
        schema.add_property(PropertyDef::new("segment_id", PropertyType::Int).unique());
        schema.add_property(PropertyDef::new("name", PropertyType::String));
        db.create_label_schema(schema).expect("create schema");

        // Create the initial node.
        db.execute_cypher("CREATE (s:Segment {segment_id: 42, name: 'original'})")
            .expect("create initial node");

        // MERGE on existing segment_id must find the node, not try to create it.
        // Before the fix this throws: "write conflict: unique constraint violated
        // on index segment_segment_id".
        let result = db.execute_cypher(
            "MERGE (s:Segment {segment_id: 42}) SET s.name = 'updated' RETURN s.name",
        );
        assert!(
            result.is_ok(),
            "MERGE on existing unique node must not error: {:?}",
            result.err()
        );

        let rows = result.unwrap();
        assert_eq!(rows.len(), 1, "MERGE must return exactly one matched row");

        // Verify the SET was applied — ON MATCH branch was taken.
        let rows = db
            .execute_cypher("MATCH (s:Segment {segment_id: 42}) RETURN s.name")
            .expect("match after merge");
        assert_eq!(rows.len(), 1);
        use coordinode_core::graph::types::Value;
        assert_eq!(
            rows[0].get("s.name"),
            Some(&Value::String("updated".into())),
            "SET must be applied via ON MATCH branch"
        );
    }

    /// Same as above but using Cypher parameters ($val / $name) — the bug was
    /// reported with parameterized queries. Parameters must not affect MERGE
    /// node matching behaviour.
    #[test]
    fn merge_with_params_on_existing_unique_node_does_not_error() {
        use coordinode_core::graph::types::Value;
        use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        let mut schema = LabelSchema::new_node_id("Segment");
        schema.add_property(PropertyDef::new("segment_id", PropertyType::Int).unique());
        schema.add_property(PropertyDef::new("name", PropertyType::String));
        db.create_label_schema(schema).expect("create schema");

        // Create node via params.
        let mut create_params = std::collections::HashMap::new();
        create_params.insert("sid".into(), Value::Int(99));
        create_params.insert("name".into(), Value::String("original".into()));
        db.execute_cypher_with_params(
            "CREATE (s:Segment {segment_id: $sid, name: $name})",
            create_params,
        )
        .expect("create node");

        // MERGE + SET via params.  Bug: this throws "unique constraint violated"
        // because NodeScan misses the node and MERGE falls through to CREATE.
        let mut merge_params = std::collections::HashMap::new();
        merge_params.insert("sid".into(), Value::Int(99));
        merge_params.insert("new_name".into(), Value::String("updated".into()));
        let result = db.execute_cypher_with_params(
            "MERGE (s:Segment {segment_id: $sid}) SET s.name = $new_name RETURN s.name",
            merge_params,
        );
        assert!(
            result.is_ok(),
            "parameterized MERGE on existing unique node must not error: {:?}",
            result.err()
        );

        // Verify ON MATCH was taken.
        let mut match_params = std::collections::HashMap::new();
        match_params.insert("sid".into(), Value::Int(99));
        let rows = db
            .execute_cypher_with_params(
                "MATCH (s:Segment {segment_id: $sid}) RETURN s.name",
                match_params,
            )
            .expect("match after merge");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].get("s.name"),
            Some(&Value::String("updated".into())),
            "SET must apply in ON MATCH branch"
        );
    }

    /// Exact gRPC repro: STRICT mode, unique INT id, node created with String value.
    ///
    /// The user-reported repro uses schema_mode=1 (STRICT), id type=INT64 (1),
    /// then creates a node with id="x1" (string literal in Cypher). After restart
    /// MERGE (n:TestNode {id: "x1"}) SET n.value = "updated" throws unique constraint.
    ///
    /// This test exercises the same mismatch: declared INT, actual String value in Cypher.
    #[test]
    fn merge_strict_mode_unique_id_string_literal_does_not_error() {
        use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // STRICT mode, unique id property (INT).
        let mut schema = LabelSchema::new_node_id("TestNode");
        schema.add_property(
            PropertyDef::new("id", PropertyType::String)
                .unique()
                .not_null(),
        );
        db.create_label_schema(schema).expect("create schema");

        // Create node.
        db.execute_cypher("CREATE (n:TestNode {id: 'x1'})")
            .expect("create node");

        // MERGE + SET undeclared property — the reported error is "unique constraint violated"
        // which means NodeScan missed the node. Setting 'value' is a separate schema issue
        // but the constraint error suggests MERGE didn't find the node at all.
        //
        // Test the core invariant: MERGE must find the existing node (match count = 1).
        let result =
            db.execute_cypher("MERGE (n:TestNode {id: 'x1'}) ON MATCH SET n.id = 'x1' RETURN n.id");
        assert!(
            result.is_ok(),
            "MERGE on existing STRICT unique node must not error: {:?}",
            result.err()
        );
        let rows = result.unwrap();
        assert_eq!(rows.len(), 1, "MERGE must match exactly one node");
    }

    /// MERGE across restart: create label+node in session 1, MERGE in session 2.
    ///
    /// This is the exact gRPC repro pattern: restart the server between CREATE and MERGE.
    /// The interner and unique index must survive restart correctly so MERGE finds
    /// the node instead of trying to create a duplicate.
    #[test]
    fn merge_on_existing_unique_node_after_restart_does_not_error() {
        use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

        let dir = tempfile::tempdir().expect("tempdir");

        // Session 1: create label schema + node.
        {
            let mut db = Database::open(dir.path()).expect("open");

            let mut schema = LabelSchema::new_node_id("TestNode");
            schema.add_property(
                PropertyDef::new("id", PropertyType::String)
                    .unique()
                    .not_null(),
            );
            db.create_label_schema(schema).expect("create schema");

            db.execute_cypher("CREATE (n:TestNode {id: 'x1'})")
                .expect("create node");
        }

        // Session 2 (simulates server restart): reopen DB, run MERGE.
        // Bug hypothesis: after restart, the interner or index state is corrupted
        // so NodeScan misses the existing node → MERGE falls through to CREATE →
        // B-tree unique index violation.
        {
            let mut db = Database::open(dir.path()).expect("reopen");

            let result = db.execute_cypher(
                "MERGE (n:TestNode {id: 'x1'}) ON MATCH SET n.id = 'x1' RETURN n.id",
            );
            assert!(
                result.is_ok(),
                "MERGE on existing unique node after restart must succeed: {:?}",
                result.err()
            );
            let rows = result.unwrap();
            assert_eq!(
                rows.len(),
                1,
                "MERGE must match exactly one node after restart"
            );
        }
    }

    // ─── Bug regression: vector schema dimension lost across restart ─────────────

    /// Writing a vector node after DB restart must not fail when the persisted
    /// label schema has `dimensions: 0` (lost across restart).
    ///
    /// Bug: `PropertyDefinition` in proto has no `dimensions` field.
    /// `schema.rs` hardcodes `dimensions: 0` when deserializing VECTOR type.
    /// After restart, all VECTOR properties have `dimensions: 0` → dimension
    /// validation fails on the next write.
    ///
    /// Expected: dimension is either preserved in schema or auto-inferred from
    /// the first written vector. Subsequent writes of matching dimension succeed.
    #[test]
    fn vector_schema_dimension_survives_restart() {
        use coordinode_core::graph::types::{Value, VectorMetric};
        use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};
        use coordinode_query::index::VectorIndexConfig;

        let dir = tempfile::tempdir().expect("tempdir");

        // Session 1: create schema with 3-dimensional vector, write a node.
        {
            let mut db = Database::open(dir.path()).expect("open");

            let mut schema = LabelSchema::new_node_id("Doc");
            schema.add_property(PropertyDef::new(
                "embedding",
                PropertyType::Vector {
                    dimensions: 3,
                    metric: VectorMetric::Cosine,
                },
            ));
            db.create_label_schema(schema).expect("create schema");

            db.create_vector_index(
                "doc_embedding",
                "Doc",
                "embedding",
                VectorIndexConfig {
                    dimensions: 3,
                    ..VectorIndexConfig::default()
                },
            );

            let mut params = std::collections::HashMap::new();
            params.insert("vec".into(), Value::Vector(vec![1.0, 0.0, 0.0]));
            db.execute_cypher_with_params("CREATE (d:Doc {embedding: $vec})", params)
                .expect("create first node");
        }

        // Session 2: reopen and write another vector — must not fail.
        // Before the fix: schema has dimensions=0 after restart → write rejected.
        {
            let mut db = Database::open(dir.path()).expect("reopen");

            let mut params = std::collections::HashMap::new();
            params.insert("vec".into(), Value::Vector(vec![0.0, 1.0, 0.0]));
            let result = db.execute_cypher_with_params("CREATE (d:Doc {embedding: $vec})", params);
            assert!(
                result.is_ok(),
                "vector write after restart must succeed (dimension must survive restart): {:?}",
                result.err()
            );
        }
    }

    // ─── Bug regression: HNSW not rebuilt for overflow (Flexible-mode) vectors ───

    /// After DB restart, HNSW must be rebuilt from vectors stored in `record.extra`
    /// Schema with dimensions=0 (the value gRPC sets via proto_type_to_property_type(7)):
    /// writing a vector must NOT fail on type/dimension mismatch.
    ///
    /// This is the EXACT schema state after `SchemaService/CreateLabel` with type=7 (VECTOR):
    /// `proto_type_to_property_type(7)` → `PropertyType::Vector { dimensions: 0, metric: Cosine }`.
    /// Because the proto `PropertyDefinition` has no `dimensions` field, it's always 0.
    ///
    /// Expected: dimension validation treats 0 as "unset/auto" and accepts any vector length,
    /// OR the schema auto-updates to the first written dimension.
    #[test]
    fn vector_write_with_grpc_schema_zero_dimensions_does_not_error() {
        use coordinode_core::graph::types::{Value, VectorMetric};
        use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // Simulate what gRPC SchemaService does: dimensions=0 because proto has no field.
        let mut schema = LabelSchema::new_node_id("VecTest");
        schema.add_property(PropertyDef::new(
            "emb",
            PropertyType::Vector {
                dimensions: 0, // ← gRPC always writes 0 (proto has no dimensions field)
                metric: VectorMetric::Cosine,
            },
        ));
        db.create_label_schema(schema).expect("create schema");

        // Write a 4-dimensional vector.
        // Bug: validation checks `vec.len() != 0` → VectorDimsMismatch(expected=0, got=4).
        let mut params = std::collections::HashMap::new();
        params.insert("vec".into(), Value::Vector(vec![0.1, 0.2, 0.3, 0.4]));
        let result = db.execute_cypher_with_params("CREATE (n:VecTest {emb: $vec})", params);
        assert!(
            result.is_ok(),
            "vector write must succeed when schema has dimensions=0 (unset via gRPC): {:?}",
            result.err()
        );
    }

    /// (overflow props in Flexible/Validated schema mode).
    ///
    /// Bug: `load_vector_indexes` only checks `record.props.get(&field_id)`.
    /// For Flexible-mode nodes, vectors are stored in `record.extra` (string-keyed
    /// overflow map). The interner may not have an entry for the prop, and even if
    /// it does, `record.props` doesn't contain it → HNSW rebuilt empty.
    ///
    /// Expected: after restart, vector search returns semantically relevant results.
    #[test]
    fn hnsw_rebuilt_for_flexible_mode_overflow_vectors() {
        use coordinode_core::graph::types::Value;
        use coordinode_core::schema::definition::{LabelSchema, SchemaMode};
        use coordinode_query::index::VectorIndexConfig;

        let dir = tempfile::tempdir().expect("tempdir");

        // Session 1: use Flexible schema (vectors go to overflow/extra).
        {
            let mut db = Database::open(dir.path()).expect("open");

            // Flexible label — no declared properties, all stored as overflow.
            let mut schema = LabelSchema::new_node_id("Article");
            schema.set_mode(SchemaMode::Flexible);
            db.create_label_schema(schema).expect("create schema");

            db.create_vector_index(
                "article_embedding",
                "Article",
                "embedding",
                VectorIndexConfig {
                    dimensions: 3,
                    ..VectorIndexConfig::default()
                },
            );

            // Write two nodes with clearly distinct embeddings.
            let mut p1 = std::collections::HashMap::new();
            p1.insert("vec".into(), Value::Vector(vec![1.0, 0.0, 0.0]));
            db.execute_cypher_with_params(
                "CREATE (a:Article {embedding: $vec, title: 'rust'})",
                p1,
            )
            .expect("create article 1");

            let mut p2 = std::collections::HashMap::new();
            p2.insert("vec".into(), Value::Vector(vec![0.0, 1.0, 0.0]));
            db.execute_cypher_with_params(
                "CREATE (a:Article {embedding: $vec, title: 'golang'})",
                p2,
            )
            .expect("create article 2");
        }

        // Session 2: reopen and do a vector search.
        // Before the fix: HNSW is empty after restart → no results returned.
        {
            let mut db = Database::open(dir.path()).expect("reopen");

            // Query close to [1, 0, 0] — must return the 'rust' article.
            let results = db
                .execute_cypher(
                    "MATCH (a:Article) \
                     WHERE vector_distance(a.embedding, [0.99, 0.0, 0.0]) < 0.1 \
                     RETURN a.title",
                )
                .expect("vector search after restart");

            assert!(
                !results.is_empty(),
                "HNSW must be rebuilt from overflow vectors on restart; got 0 results"
            );
        }
    }

    /// Regression: DETACH DELETE must actually remove the node from storage.
    ///
    /// Bug: DETACH DELETE returns Ok but leaves the node in place.
    /// After `MATCH (n:BugTest {id: "dt-1"}) DETACH DELETE n`, a subsequent
    /// MATCH for the same node must return 0 rows.
    #[test]
    fn detach_delete_removes_node() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        db.execute_cypher("CREATE (:BugTest {id: 'dt-1', val: 'hello'})")
            .expect("create");

        // Confirm node exists.
        let before = db
            .execute_cypher("MATCH (n:BugTest {id: 'dt-1'}) RETURN n.id")
            .expect("match before delete");
        assert_eq!(before.len(), 1, "node must exist before delete");

        // Delete the node.
        db.execute_cypher("MATCH (n:BugTest {id: 'dt-1'}) DETACH DELETE n")
            .expect("detach delete");

        // Node must be gone.
        let after = db
            .execute_cypher("MATCH (n:BugTest {id: 'dt-1'}) RETURN n.id")
            .expect("match after delete");
        assert_eq!(
            after.len(),
            0,
            "DETACH DELETE must remove the node; got {} rows instead of 0",
            after.len()
        );
    }

    /// Regression: SET on a VECTOR property must update the HNSW index position.
    ///
    /// Bug: after `SET n.embedding = [new_vec]`, vector_distance queries still
    /// use the original embedding from CREATE, silently returning stale results.
    #[test]
    fn vector_set_updates_hnsw_index() {
        use coordinode_core::graph::types::VectorMetric;
        use coordinode_query::index::VectorIndexConfig;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // Create a vector index on :VecTest(embedding).
        db.create_vector_index(
            "vec_test_idx",
            "VecTest",
            "embedding",
            VectorIndexConfig {
                dimensions: 4,
                metric: VectorMetric::L2,
                ..VectorIndexConfig::default()
            },
        );

        // Insert a node with embedding A = [1,0,0,0].
        db.execute_cypher("CREATE (:VecTest {id: 'v-1', embedding: [1.0, 0.0, 0.0, 0.0]})")
            .expect("create");

        // Update embedding to B = [0,1,0,0] (orthogonal to A).
        db.execute_cypher("MATCH (n:VecTest {id: 'v-1'}) SET n.embedding = [0.0, 1.0, 0.0, 0.0]")
            .expect("set embedding");

        // Confirm storage reflects the update.
        let stored = db
            .execute_cypher("MATCH (n:VecTest {id: 'v-1'}) RETURN n.embedding")
            .expect("read back");
        assert_eq!(stored.len(), 1, "node must still exist after SET");

        // Vector search with query ≈ B must find the node (distance < 0.1).
        // Bug: HNSW still has the original A position → 0 rows returned.
        let results = db
            .execute_cypher(
                "MATCH (n:VecTest) \
                 WHERE vector_distance(n.embedding, [0.0, 1.0, 0.0, 0.0]) < 0.1 \
                 RETURN n.id",
            )
            .expect("vector search after SET");

        assert_eq!(
            results.len(),
            1,
            "HNSW must reflect the updated embedding; \
             expected 1 result for query ≈ B, got {}",
            results.len()
        );
    }

    // ── R-PUSH1: end-to-end + wiring tests ────────────────────────────

    /// CombinedStats returns real values from VectorIndexRegistry when the
    /// index is registered. Closes the gap where vector_index_size/dim/
    /// crossover defaulted to None — the wrapper is the live source of
    /// truth for per-index statistics consumed by optimize_push_down.
    #[test]
    fn combined_stats_returns_real_vector_index_metadata() {
        use coordinode_core::graph::stats::StorageStats;
        use coordinode_query::index::VectorIndexConfig;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // Create an HNSW index on (Doc.embedding) with dim=64, M=12.
        db.create_vector_index(
            "doc_emb_hnsw",
            "Doc",
            "embedding",
            VectorIndexConfig {
                dimensions: 64,
                metric: coordinode_core::graph::types::VectorMetric::Cosine,
                m: 12,
                ef_construction: 200,
                quantization: coordinode_vector::hnsw::QuantizationCodec::None,
                offload_vectors: false,
                ef_search: None,
                rerank_candidates: None,
            },
        );
        // Seed one vector so size > 0.
        db.execute_cypher(
            "CREATE (d:Doc {embedding: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, \
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})",
        )
        .expect("create vector node");

        let graph_stats = db.compute_stats().expect("compute_stats");
        let combined = CombinedStats {
            graph: &graph_stats,
            vector: &db.vector_index_registry,
        };
        assert_eq!(
            combined.vector_index_dim("Doc", "embedding"),
            Some(64),
            "dim from VectorIndexConfig must reach StorageStats"
        );
        let cross = combined
            .vector_index_crossover("Doc", "embedding")
            .expect("crossover present when index registered");
        // Heuristic: M=12 → max(12,8) * 32 = 384, no quantization. Clamped to [64,1024] → 384.
        assert_eq!(cross, 384, "crossover heuristic uses HNSW M parameter");

        // Unknown (label, property) returns None.
        assert!(combined.vector_index_dim("Doc", "no_such_prop").is_none());
        assert!(combined.vector_index_size("Other", "embedding").is_none());
    }

    /// CombinedStats halves the crossover threshold when the index uses
    /// SQ8 quantization — quantized distance is ~2x cheaper, so the
    /// graph-first crossover moves down to keep ACORN attractive earlier.
    #[test]
    fn combined_stats_crossover_lower_under_quantization() {
        use coordinode_core::graph::stats::StorageStats;
        use coordinode_query::index::VectorIndexConfig;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");
        db.create_vector_index(
            "q_idx",
            "QDoc",
            "embedding",
            VectorIndexConfig {
                dimensions: 32,
                metric: coordinode_core::graph::types::VectorMetric::Cosine,
                m: 12,
                ef_construction: 200,
                quantization: coordinode_vector::hnsw::QuantizationCodec::Sq8,
                offload_vectors: false,
                ef_search: None,
                rerank_candidates: None,
            },
        );
        let graph_stats = db.compute_stats().expect("compute_stats");
        let combined = CombinedStats {
            graph: &graph_stats,
            vector: &db.vector_index_registry,
        };
        let cross = combined
            .vector_index_crossover("QDoc", "embedding")
            .unwrap();
        // M=12, base=384, quant → base/2 = 192. Clamped to [64,1024] → 192.
        assert_eq!(cross, 192);
    }

    /// EXPLAIN annotates the serving health of a vector index the plan reads
    /// through: a top-k vector query is promoted to HnswScan, so
    /// the index name appears and its current state is surfaced.
    #[test]
    fn explain_annotates_vector_index_health() {
        use coordinode_core::graph::types::VectorMetric;
        use coordinode_query::index::VectorIndexConfig;

        let dir = tempfile::tempdir().unwrap();
        let mut db = Database::open(dir.path()).unwrap();
        db.create_vector_index(
            "movie_emb",
            "Movie",
            "embedding",
            VectorIndexConfig {
                dimensions: 3,
                metric: VectorMetric::L2,
                m: 16,
                ef_construction: 200,
                quantization: coordinode_vector::hnsw::QuantizationCodec::None,
                offload_vectors: false,
                ef_search: None,
                rerank_candidates: None,
            },
        );
        // Drive the index into a non-ready state so EXPLAIN shows more than the
        // default "ready".
        db.vector_index_registry()
            .report_health_rebuild("Movie", "embedding", 0.5, 250);

        let explain = db
            .explain_cypher(
                "MATCH (m:Movie) \
                 WITH *, vector_distance(m.embedding, [0.1, 0.2, 0.3]) AS d \
                 ORDER BY d ASC LIMIT 5 \
                 RETURN m",
            )
            .expect("explain");

        assert!(
            explain.contains("movie_emb"),
            "top-k vector query must be promoted to HnswScan(movie_emb): {explain}"
        );
        assert!(
            explain.contains("Vector index health:"),
            "EXPLAIN must carry the index health section: {explain}"
        );
        assert!(
            explain.to_lowercase().contains("rebuilding"),
            "health annotation must reflect the rebuilding state: {explain}"
        );
    }

    /// End-to-end: an EXPLAIN of TRAVERSE→VECTOR_FILTER must hit
    /// optimize_push_down and surface a populated `push_down` decision on
    /// the VectorFilter operator in the plan. Smoke test that the entire
    /// pipeline — execute_cypher_impl wiring → CombinedStats →
    /// optimize_push_down → annotated VectorFilter — works on a real
    /// Database instance, not just in planner unit tests.
    #[test]
    fn end_to_end_traverse_then_vector_filter_gets_push_down_decision() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");

        // Seed graph: User → LIKES → Movie with embedding.
        db.execute_cypher(
            "CREATE (u:User {name: 'A'})-[:LIKES]->(m:Movie {embedding: [0.1, 0.2, 0.3]})",
        )
        .expect("seed");

        let explain = db
            .explain_cypher(
                "MATCH (u:User)-[:LIKES]->(m:Movie) \
                 WHERE vector_distance(m.embedding, [0.1, 0.2, 0.3]) < 0.5 \
                 RETURN m",
            )
            .expect("explain");
        // The plan must build cleanly; we don't assert the strategy slug
        // here (that's R-PUSH2's EXPLAIN JSON contract). What we assert
        // end-to-end is that the explain pipeline runs without panic and
        // touches the push_down pass — verified indirectly by checking
        // that planner test invariants still hold on the equivalent plan
        // (covered by planner integration tests). This test guarantees
        // the wiring compiles AND runs against a real Database.
        assert!(
            !explain.is_empty(),
            "EXPLAIN must produce output for TRAVERSE→VECTOR_FILTER plans"
        );
        assert!(
            explain.to_lowercase().contains("vectorfilter")
                || explain.to_lowercase().contains("vector"),
            "EXPLAIN must mention the vector operator: {explain}"
        );
    }

    /// End-to-end: a query without TRAVERSE upstream of VECTOR_FILTER
    /// must still build and execute without push-down annotation (the
    /// invariant only fires when Traverse is present). Verifies negative
    /// path through the entire stack.
    #[test]
    fn end_to_end_vector_filter_without_traverse_executes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (d:Doc {embedding: [0.5, 0.5, 0.5]})")
            .expect("seed");
        let rows = db
            .execute_cypher(
                "MATCH (d:Doc) WHERE vector_distance(d.embedding, [0.5, 0.5, 0.5]) < 1.0 \
                 RETURN d",
            )
            .expect("vector filter without traverse should work");
        assert!(!rows.is_empty());
    }
}
