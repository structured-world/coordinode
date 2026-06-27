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
    ExtensionRegistry, FeedbackCache, ScanPaging, WriteStats,
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

/// One page of a keyset-resumable server-side cursor.
///
/// A cursor pins `read_ts` once and feeds it back into every subsequent
/// [`Database::execute_cypher_paged`] call, so all pages observe the same
/// MVCC snapshot even under concurrent writes. `last_key` is the opaque
/// resume token for the next page (the last storage key the scan emitted);
/// `exhausted` is `true` once the underlying scan has no more rows.
#[derive(Debug, Clone)]
pub struct PagedCypherResult {
    pub rows: Vec<Row>,
    /// Resume token for the next page: pass back as `resume`. `None` when the
    /// page produced no scan key (empty result or a non-keyset plan).
    pub last_key: Option<Vec<u8>>,
    /// `true` once the scan is fully drained: no further pages remain.
    pub exhausted: bool,
    /// The pinned MVCC snapshot timestamp this cursor reads against. Echo it
    /// into the next page's `read_ts` to keep the snapshot stable.
    pub read_ts: u64,
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

/// Canonical f32-vector coercion (handles `Value::Vector` and numeric
/// `Value::Array`). Re-exported so `crate::db::try_extract_vector` callers
/// keep resolving; the single definition lives in `coordinode-core`.
pub(crate) use coordinode_core::graph::types::try_extract_vector;

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
    /// `true` in cluster mode (writes applied by the Raft state machine), where
    /// the AFTER COMMIT trigger queue is drained by a leader-gated background
    /// worker. `false` in embedded single-node mode, where the queue is drained
    /// inline at the end of each committed write (deterministic, no extra
    /// thread). Mirrors the `spawn_oplog_worker` discriminator.
    cluster_mode: bool,
    /// Operator-tunable knobs for the AFTER COMMIT trigger dispatcher (R192):
    /// cascade-depth cap + default retry policy. Defaults match ADR-026; the
    /// server overrides them from `coordinode.conf` via
    /// [`Database::set_trigger_dispatch_config`].
    trigger_dispatch_config: after_commit::TriggerDispatchConfig,
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
    /// Async AFTER COMMIT cascade generation for this statement. `0` for user
    /// statements; set to the queued event's generation when the dispatcher
    /// runs a trigger body so enqueued child events are stamped `generation + 1`.
    after_commit_generation: u32,
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
        let engine = StorageEngine::open_embedded(&config, oracle.clone())?;
        let engine = Arc::new(engine);
        let pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline> =
            Arc::new(OwnedLocalProposalPipeline::new(&engine));
        // Embedded: HNSW updated inline, no oplog-tailing worker.
        Self::finish_open(&path, config, oracle, engine, pipeline, false)
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
        // Embedded in-memory: HNSW updated inline, no oplog-tailing worker.
        Self::finish_open(&virtual_path, config, oracle, engine, pipeline, false)
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
        // Cluster mode (RaftProposalPipeline): the state machine applies writes,
        // so the oplog-tailing worker maintains HNSW (no inline index path).
        Self::finish_open(path.as_ref(), config, oracle, engine, pipeline, true)
    }

    /// Create a checkpoint (repair base + oplog copy) under
    /// `<data_dir>/checkpoints`, retaining the newest `keep`. Returns the new
    /// checkpoint path.
    ///
    /// A checkpoint is the base [`verify_and_repair`](Self::verify_and_repair)
    /// rebuilds a corrupt partition from. Take them periodically (see
    /// [`crate::repair::CheckpointScheduler`]) or before risky operations; the
    /// retained oplog journal rolls the base forward to the moment of repair.
    pub fn checkpoint(&self, keep: usize) -> Result<std::path::PathBuf, DatabaseError> {
        let root = crate::repair::checkpoint_root(self.engine.data_dir());
        let path = crate::repair::create_checkpoint(&self.engine, &root)?;
        crate::repair::prune_checkpoints(&root, keep)?;
        Ok(path)
    }

    /// Flush all in-memory writes to durable on-disk SSTs.
    ///
    /// Writes are already durable in the oplog journal once they return; this
    /// forces the active memtables out to SST segments so the on-disk image
    /// reflects the current state without waiting for a background flush. Useful
    /// before snapshotting the data directory or asserting on-disk layout.
    pub fn persist(&self) -> Result<(), DatabaseError> {
        Ok(self.engine.persist()?)
    }

    /// Scrub every partition and rebuild any corrupt one from the latest
    /// checkpoint plus oplog replay (single-node WAL-replay-repair, repair
    /// path 2).
    ///
    /// Returns a [`RepairReport`]. `report.is_clean()` is `false` only if
    /// corruption remained — e.g. no checkpoint existed to rebuild from, in
    /// which case the operator must restore from an off-device backup. Called
    /// automatically on open when a checkpoint is present.
    pub fn verify_and_repair(&self) -> Result<crate::repair::RepairReport, DatabaseError> {
        let root = crate::repair::checkpoint_root(self.engine.data_dir());
        Ok(crate::repair::verify_and_repair(&self.engine, &root)?)
    }

    /// Shared initialization logic for both `open()` and `open_with_pipeline()`.
    fn finish_open(
        path: &Path,
        config: StorageConfig,
        oracle: Arc<TimestampOracle>,
        engine: Arc<StorageEngine>,
        pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline>,
        // Whether to run the oplog-tailing vector index worker. Only cluster
        // mode (writes applied by the Raft state machine) needs it; embedded
        // mode updates HNSW inline on the write path, so spawning the worker
        // there would double-insert and race the inline path — degrading
        // recall. (Embedded gained a retained oplog dir with G111, so the mere
        // presence of `oplog/` is no longer the cluster discriminator.)
        spawn_oplog_worker: bool,
    ) -> Result<Self, DatabaseError> {
        // Auto-repair on open (G111). For an embedded engine with a retained
        // oplog journal, if a checkpoint exists, scrub and rebuild any corrupt
        // partition from that checkpoint + oplog replay BEFORE any state is read
        // below. Cluster engines (no journal) and journal-less in-memory engines
        // skip this; deployments that never checkpoint pay no scrub cost. A
        // repair failure is logged, not fatal — the caller can re-run
        // `verify_and_repair()` to inspect.
        if engine.has_journal() {
            let root = crate::repair::checkpoint_root(engine.data_dir());
            if crate::repair::latest_checkpoint(&root).is_some() {
                match crate::repair::verify_and_repair(&engine, &root) {
                    Ok(report) if !report.repaired.is_empty() => {
                        tracing::warn!(
                            repaired = ?report.repaired,
                            clean = report.clean_after,
                            "auto-repaired corrupt partitions on open"
                        );
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::error!(error = %e, "auto-repair on open failed; continuing")
                    }
                }
            }
        }

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
        // between the two is harmless). Embedded deployments update indexes
        // inline on the write path and run without the worker — `spawn_oplog_worker`
        // is the cluster discriminator (embedded now also has an `oplog/` dir).
        let oplog_dir = engine.data_dir().join("oplog").join("0");
        let vector_worker = if spawn_oplog_worker && oplog_dir.is_dir() {
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
            cluster_mode: spawn_oplog_worker,
            trigger_dispatch_config: after_commit::TriggerDispatchConfig::default(),
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
            after_commit_generation: 0,
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
        self.execute_cypher_impl(
            query,
            Some(source),
            None,
            &session,
            TxnMode::AutoCommit,
            &mut None,
        )
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
        self.execute_cypher_impl(
            query,
            Some(source),
            params,
            &session,
            TxnMode::AutoCommit,
            &mut None,
        )
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
        self.execute_cypher_impl(query, None, None, &session, TxnMode::AutoCommit, &mut None)
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
        self.execute_cypher_impl(
            query,
            None,
            params,
            &session,
            TxnMode::AutoCommit,
            &mut None,
        )
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
        self.execute_cypher_impl(
            query,
            source,
            params,
            &session,
            TxnMode::AutoCommit,
            &mut None,
        )
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
            after_commit_generation: 0,
        };
        let params = params.and_then(|p| if p.is_empty() { None } else { Some(p) });
        // On error the state is intentionally NOT re-parked → transaction aborts.
        let (rows, _stats, out_state) = self.execute_cypher_impl(
            query,
            None,
            params,
            &session,
            TxnMode::Interactive(Box::new(state)),
            &mut None,
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

    /// Override the AFTER COMMIT trigger dispatcher knobs (cascade-depth cap +
    /// default retry policy). The server calls this once at startup from
    /// `coordinode.conf` / CLI flags; embedded callers may tune it directly.
    pub fn set_trigger_dispatch_config(&mut self, cfg: TriggerDispatchConfig) {
        self.trigger_dispatch_config = cfg;
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
            after_commit_generation: 0,
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
        self.execute_cypher_impl(
            query,
            source,
            params,
            &session,
            TxnMode::AutoCommit,
            &mut None,
        )
        .map(|(rows, write_stats, _)| CypherResult { rows, write_stats })
    }

    /// Execute one page of a keyset-resumable server-side cursor.
    ///
    /// The cursor reads against a pinned MVCC snapshot: pass `read_ts = None`
    /// for the first page (a fresh snapshot is taken and returned in
    /// [`PagedCypherResult::read_ts`]); echo that timestamp back as
    /// `read_ts = Some(t)` for every following page so the whole result set
    /// stays stable under concurrent writes. `resume` is the previous page's
    /// [`PagedCypherResult::last_key`] (`None` for the first page); `limit`
    /// bounds the rows scanned per page, keeping cursor memory `O(limit)`
    /// rather than `O(result)`.
    ///
    /// This is the keyset primitive: it assumes a non-blocking single-scan
    /// plan whose scan honours `resume` + `limit` and reports `last_key` +
    /// `exhausted`. The caller (the server cursor layer) MUST classify the
    /// plan first and route blocking plans (sort / aggregate / distinct) to
    /// the materialise-once cursor instead: under a blocking operator the
    /// scan would emit only one page's worth of rows and the operator above
    /// would compute over a partial set. A plan with no scan at all (e.g. a
    /// bare `RETURN`) reports a single exhausted page.
    pub fn execute_cypher_paged(
        &self,
        query: &str,
        params: Option<std::collections::HashMap<String, coordinode_core::graph::types::Value>>,
        read_ts: Option<u64>,
        resume: Option<Vec<u8>>,
        limit: usize,
    ) -> Result<PagedCypherResult, DatabaseError> {
        // Pin the snapshot once; later pages re-pin to the same timestamp.
        let pinned = read_ts.unwrap_or_else(|| self.oracle.next().as_raw());
        let session = QuerySession {
            read_concern: coordinode_core::txn::read_concern::ReadConcernLevel::Snapshot,
            snapshot_read_ts: Some(pinned),
            write_concern: self.write_concern.clone(),
            vector_consistency: self.vector_consistency,
            after_commit_generation: 0,
        };
        let params = params.and_then(|p| if p.is_empty() { None } else { Some(p) });
        let mut paging = Some(ScanPaging {
            resume,
            limit,
            last_key: None,
            exhausted: false,
        });
        let (rows, write_stats, _) = self.execute_cypher_impl(
            query,
            None,
            params,
            &session,
            TxnMode::AutoCommit,
            &mut paging,
        )?;
        // The executor leaves `paging` populated for a keyset plan; a blocking
        // plan ignores it (the scan path never ran), so treat an untouched
        // page as a single exhausted page over the whole materialised result.
        let paging = paging.unwrap_or(ScanPaging {
            resume: None,
            limit,
            last_key: None,
            exhausted: true,
        });
        // A page is exhausted when the scan reported end, or when there is no
        // resume token at all (a no-scan plan, or a page that emitted no key):
        // without a `last_key` the next call has nothing to resume from.
        let exhausted = paging.exhausted || paging.last_key.is_none();
        Ok(PagedCypherResult {
            rows,
            last_key: paging.last_key,
            exhausted,
            read_ts: pinned,
            write_stats,
        })
    }

    /// Classify whether `query` can be served by a keyset-resumable cursor via
    /// [`Database::execute_cypher_paged`].
    ///
    /// Eligible plans stream from a single `NodeScan` through only
    /// row-preserving, non-collapsing operators (`Filter`, non-`DISTINCT`
    /// `Project`), so the executor can page by storage key and the cursor stays
    /// `O(batch)`. Any blocking operator (sort, aggregate, `DISTINCT`),
    /// row-multiplying source (traverse, cartesian product, union), bounded
    /// operator (`LIMIT` / `SKIP`), index point-lookup, or write makes the plan
    /// ineligible: the cursor layer runs those materialise-once instead. A
    /// query that fails to parse or plan is reported ineligible (the cursor
    /// surfaces the real error on execution).
    pub fn keyset_pageable(&self, query: &str) -> bool {
        let Ok(ast) = cypher::parse(query) else {
            return false;
        };
        let Ok(plan) = planner::build_logical_plan(&ast) else {
            return false;
        };
        Self::spine_is_keyset_pageable(&plan.root)
    }

    /// Walk a plan spine: keyset-pageable iff it is a single `NodeScan` under a
    /// chain of only `Filter` and order-preserving (`distinct == false`)
    /// `Project` nodes.
    fn spine_is_keyset_pageable(op: &planner::LogicalOp) -> bool {
        match op {
            planner::LogicalOp::NodeScan { .. } => true,
            planner::LogicalOp::Filter { input, .. } => Self::spine_is_keyset_pageable(input),
            planner::LogicalOp::Project {
                input,
                distinct: false,
                ..
            } => Self::spine_is_keyset_pageable(input),
            _ => false,
        }
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

        self.execute_cypher_impl(query, None, None, &session, TxnMode::AutoCommit, &mut None)
            .map(|(r, _, _)| r)
    }

    fn execute_cypher_impl(
        &self,
        query: &str,
        source: Option<&SourceContext>,
        params: Option<std::collections::HashMap<String, coordinode_core::graph::types::Value>>,
        session: &QuerySession,
        txn_mode: TxnMode,
        // Keyset cursor in/out channel: on entry carries `resume` + `limit` for
        // a server-side cursor page; on return the executor has overwritten
        // `last_key` + `exhausted`. `&mut None` for a non-paged (whole-result)
        // execution: the executor then materialises the full result set.
        scan_paging: &mut Option<ScanPaging>,
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
            scan_paging: scan_paging.clone(),
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
            after_commit_generation: session.after_commit_generation,
            correlated_row: None,
            foreach_scope: None,
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
        // Hand the executor-updated keyset state (last_key + exhausted) back to
        // the caller through the in/out channel. `None` stays `None` for a
        // non-paged execution; the cursor path reads this to build the next
        // page's resume token.
        *scan_paging = ctx.scan_paging.clone();
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

        // Drain AFTER COMMIT triggers enqueued by this committed write. Embedded
        // only (cluster drains via the leader-gated worker); a no-op when no
        // events were enqueued and when already inside a trigger body. Skipped
        // for interactive statements — their writes are not yet committed.
        if had_mutations && !interactive {
            self.drive_after_commit_inline();
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

mod after_commit;
pub use after_commit::{AfterCommitDispatchReport, TriggerDispatchConfig};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
