//! Embedded database: open CoordiNode storage, execute queries in-process.
//!
//! Embedded mode is single-process (no Raft, no clustering).
//! For CE 3-node HA, use the coordinode server binary.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use coordinode_core::graph::intern::{FieldInterner, FieldRegistrar};
use coordinode_core::graph::node::{NodeId, NodeIdAllocator};
use coordinode_core::graph::types::VectorConsistencyMode;
use coordinode_core::txn::proposal::{ProposalIdGenerator, fresh_proposal_id_base};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_query::advisor::nplus1::NPlus1Detector;
use coordinode_query::advisor::{AdvisorContext, DismissedSet, QueryRegistry, SourceContext};
use coordinode_query::cypher;
use coordinode_query::executor::row::Row;
use coordinode_query::executor::runner::{
    AdaptiveConfig, ExecutionContext, ExecutionError, ExtensionHandler, ExtensionRegistry,
    FeedbackCache, ScanPaging, WriteStats, execute, execute_no_commit,
};
use coordinode_query::frontend::{CypherFrontend, QueryFrontend};
use coordinode_query::planner;
use coordinode_query::procedure::{Procedure, ProcedureError, ProcedureRegistry};
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

/// Outcome of a Cypher execution: result rows plus mutation statistics. Used
/// by [`Database::execute_cypher_full`] so callers (notably the gRPC server)
/// can surface real mutation counts in their response stats.
#[derive(Debug, Clone)]
pub struct CypherResult {
    pub rows: Vec<Row>,
    pub write_stats: WriteStats,
}

impl CypherResult {
    /// The timestamp this statement's writes landed at, when it committed on
    /// its own. `None` for a read and for a statement inside an interactive
    /// transaction, whose timestamp belongs to the commit that ends it.
    ///
    /// This is the record's new version. A caller that writes and then means
    /// to write again conditionally already has what to state, without
    /// reading the record back and racing in between.
    pub fn commit_ts(&self) -> Option<u64> {
        self.write_stats.commit_ts
    }
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
    /// Rows produced by this page (already filtered/projected by the plan).
    pub rows: Vec<Row>,
    /// Resume token for the next page: pass back as `resume`. `None` when the
    /// page produced no scan key (empty result or a non-keyset plan).
    pub last_key: Option<Vec<u8>>,
    /// `true` once the scan is fully drained: no further pages remain.
    pub exhausted: bool,
    /// The pinned MVCC snapshot timestamp this cursor reads against. Echo it
    /// into the next page's `read_ts` to keep the snapshot stable.
    pub read_ts: u64,
    /// Mutation counters for this page. Zero for a read-only cursor; populated
    /// only if the paged statement also wrote (uncommon for a cursor).
    pub write_stats: WriteStats,
}

/// `StorageStats` adapter that augments graph-level statistics (label
/// counts, fan-out averages) with per-vector-index statistics drawn from
/// the live `VectorIndexRegistry`. The graph-predicate push-down rule needs both
/// dimensions in one place — `optimize_push_down` reads everything through
/// a single `&dyn StorageStats` reference.
///
/// The graph half delegates to the cached `StorageStatsComputer`; the
/// vector half is computed on demand (cheap — registry lookups are
/// in-memory). Crossover thresholds are derived per-index from HNSW M and
/// quantization settings (cached at build time on the index definition;
/// the heuristic here stands until measured constants replace it).
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
        // The crossover is per-index metadata derived from M and quantisation.
        // Until measured constants replace it, the heuristic below tracks the
        // expected defaults:
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

/// How long a read waits for the state its timestamp names to be complete.
///
/// It bounds the wait, never the obligation: when it passes, the read is
/// refused with what it was waiting for rather than answered from a state
/// missing a write its own timestamp covers.
const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(2000);

/// How long a query waits for a vector index still being built, under the
/// `block` policy, when neither the query nor the session names a bound.
pub const DEFAULT_VECTOR_BUILD_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// Applied Raft entries queued for the vector index worker before it is
/// behind: past this the applies do not wait, the queue drops, and the
/// worker rebuilds its indexes from the store.
const APPLIED_QUEUE_CAPACITY: usize = 16_384;

/// Canonical f32-vector coercion (handles `Value::Vector` and numeric
/// `Value::Array`). Re-exported so `crate::db::try_extract_vector` callers
/// keep resolving; the single definition lives in `coordinode-core`.
pub(crate) use coordinode_core::graph::types::try_extract_vector;

/// How a set of vector indexes is populated once registered.
#[derive(Debug, Clone, Copy)]
enum PopulateMode {
    /// Until whole, before the database serves anything: the open path.
    Blocking,
    /// One background build per index, owned by the registry: a replica
    /// bringing up an index another member defined, from the async runtime.
    Background,
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

/// How `execute_cypher_impl` should treat the statement's transaction
/// boundary.
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

/// Embedded database instance.
pub struct Database {
    engine: Arc<StorageEngine>,
    /// The field dictionary: each statement takes its verified view without
    /// a lock held for the statement, and new names are registered through
    /// the write pipeline before any data uses them.
    fields: Arc<fields::FieldDictionary>,
    /// Hands out NodeIds from leases the proposal pipeline grants.
    allocator: NodeIdAllocator,
    shard_id: u16,
    /// Live session registry for operational introspection, injected by the
    /// server (`SHOW SESSIONS` / `SHOW TRANSACTIONS`). `None` in embedded use,
    /// where there is no session layer; those statements then report nothing.
    operations: Option<Arc<dyn coordinode_core::operations::OperationsView>>,
    /// Query fingerprint registry — tracks execution statistics per query pattern.
    query_registry: Arc<QueryRegistry>,
    /// N+1 pattern detector — flags repeated queries from the same source location.
    nplus1_detector: Arc<NPlus1Detector>,
    /// Dismissed suggestion fingerprints for `db.advisor.dismiss()`.
    dismissed: Arc<DismissedSet>,
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
    /// Session-level vector MVCC consistency mode, set by
    /// `SET vector_consistency` or [`Database::set_vector_consistency`]. It
    /// replaces what a query's `read_consistency` implies, but not a mode the
    /// query names in a hint. `None`: each query decides.
    vector_consistency: Option<VectorConsistencyMode>,
    /// How long a query waits for a vector index still being built, under
    /// the `block` policy, unless it names its own bound in a hint. Set by
    /// `SET vector_build_wait` or [`Database::set_vector_build_wait`]; the
    /// server sets it from its configuration.
    vector_build_wait: Duration,
    /// Session-level read concern. Default: Local.
    read_concern: coordinode_core::txn::read_concern::ReadConcernLevel,
    /// One-shot snapshot timestamp for the next query (consumed on use).
    /// Set by `execute_cypher_with_read_concern` with Snapshot level.
    snapshot_read_ts: Option<u64>,
    /// Cached storage statistics for EXPLAIN cost estimation, with the time
    /// and the [`Self::stats_generation`] they were computed at. `None` =
    /// never computed. Refreshed when the TTL expires (see
    /// `STATS_CACHE_TTL_SECS`) or the generation moved. A failed computation
    /// is cached as `(None, ..)` for the same TTL, so a damaged counter is
    /// reported once per window rather than recomputed on every query.
    cached_stats: Mutex<
        Option<(
            Option<coordinode_storage::engine::stats::StorageStatsComputer>,
            Instant,
            u64,
        )>,
    >,
    /// Bumped by every invalidation. A write invalidates with one atomic add
    /// instead of taking the cache lock, which every concurrent writer would
    /// otherwise queue on.
    stats_generation: AtomicU64,
    /// How long cached storage statistics remain valid.
    stats_ttl: Duration,
    /// Times the statistics were recomputed from storage.
    #[cfg(test)]
    stats_computations: AtomicU64,
    /// Session-level write concern. Default: Majority.
    write_concern: coordinode_core::txn::write_concern::WriteConcern,
    /// Index registry — tracks active indexes for EXPLAIN SUGGEST false-positive prevention.
    index_registry: Arc<coordinode_query::index::IndexRegistry>,
    /// The executor of index builds: CREATE INDEX, constraints and rebuilds
    /// admit builds the engine runs, independent of the statement that asked.
    index_builds: coordinode_query::index::IndexBuildService,
    /// Vector index registry — holds live HNSW indexes for accelerated vector search.
    vector_index_registry: Arc<coordinode_query::index::VectorIndexRegistry>,
    /// Background follower of the applied commits keeping HNSW indexes
    /// current with them, removals included (see [`crate::vector_worker`]).
    /// Held for its Drop (stops the thread when the Database closes).
    _vector_worker: crate::vector_worker::VectorIndexWorker,
    /// Text index registry — holds live tantivy indexes for full-text search.
    text_index_registry: Arc<coordinode_query::index::TextIndexRegistry>,
    /// Background follower of the applied commits keeping the text indexes
    /// current with them (see [`crate::text_worker`]). Held for its Drop.
    _text_worker: crate::text_worker::TextIndexWorker,
    /// Extension-op handler registry threaded into every ExecutionContext.
    /// Empty for a plain CE Database (no extension ops dispatchable);
    /// populated via [`Database::register_extension`] by an enterprise layer
    /// or an integration test so extension operators (e.g. a sharded
    /// CREATE VECTOR INDEX) reach their handler.
    extension_registry: ExtensionRegistry,
    /// The procedures `CALL` dispatches to: the CE built-ins plus whatever an
    /// enterprise layer or embedder adds with [`Database::register_procedure`].
    procedure_registry: ProcedureRegistry,
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
    /// Operator-tunable knobs for the AFTER COMMIT trigger dispatcher:
    /// cascade-depth cap + default retry policy. The server overrides them from `coordinode.conf` via
    /// [`Database::set_trigger_dispatch_config`].
    trigger_dispatch_config: after_commit::TriggerDispatchConfig,
    /// Per-query-string parse + plan cache. Repeated invocations of
    /// the same Cypher text skip parse / semantic analysis / logical
    /// plan build entirely; the per-call optimizer passes still run
    /// on a clone of the cached plan so they stay sensitive to live
    /// index registry state. See [`PlanCache`].
    plan_cache: Arc<PlanCache>,
    /// Open interactive multi-statement transactions, keyed by a
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
    /// Idle timeout for interactive transactions: an open
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
    /// Called after an interactive transaction opens, so an idle reaper with
    /// nothing to reap can sleep until there is something.
    interactive_begun: Option<Arc<dyn Fn() + Send + Sync>>,
}

/// A query whose parse + analyze + logical-plan-build succeeded, kept
/// around so repeated invocations of the same query string skip those
/// steps entirely. The optimizer passes (index selection, VectorTopK
/// annotation, predicate push-down) still run per call because they
/// depend on the live index registry / storage statistics and would
/// stale-bind if cached.
/// Settle a plan's vector consistency: a mode the query named in a hint wins,
/// then the session's `SET vector_consistency`, then what the query's
/// `read_consistency` implies (already in the plan).
fn apply_session_vector_consistency(
    plan: &mut planner::logical::LogicalPlan,
    hinted: bool,
    session: Option<VectorConsistencyMode>,
) {
    if let (false, Some(mode)) = (hinted, session) {
        plan.vector_consistency = mode;
    }
}

/// A lowered plan ready to run, with the identity the advisor records it by.
struct Statement {
    plan: planner::logical::LogicalPlan,
    /// Canonical form of the statement, literals scrubbed.
    canonical: String,
    /// Fingerprint over the canonical form.
    fingerprint: u64,
    /// The bound a `vector_build_wait` hint named, if any.
    build_wait: Option<Duration>,
}

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
    /// The query named its own vector consistency in a hint.
    vector_consistency_hinted: bool,
    /// The bound the query named in a `vector_build_wait` hint, if any.
    vector_build_wait: Option<Duration>,
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
/// A session setting a `SET` command changes. An embedded [`Database`] is one
/// session and applies it to itself; a server holds it per client session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionSetting {
    /// `SET vector_consistency = 'mode'`.
    VectorConsistency(VectorConsistencyMode),
    /// `SET vector_build_wait = '5s'`.
    VectorBuildWait(Duration),
}

/// What one statement runs under where its caller decides it, in place of
/// the database's own settings: a server passes its client session's here,
/// so one client's settings never reach another's statements. `None` keeps
/// the database's setting.
#[derive(Debug, Clone, Default)]
pub struct StatementOptions {
    /// Read concern of this statement.
    pub read_concern: Option<coordinode_core::txn::read_concern::ReadConcern>,
    /// Write concern of this statement.
    pub write_concern: Option<coordinode_core::txn::write_concern::WriteConcern>,
    /// Vector consistency, below a hint the query names.
    pub vector_consistency: Option<VectorConsistencyMode>,
    /// Bound on waiting for a vector index still being built, below a hint
    /// the query names.
    pub vector_build_wait: Option<Duration>,
}

#[derive(Debug, Clone)]
struct QuerySession {
    read_concern: coordinode_core::txn::read_concern::ReadConcernLevel,
    /// One-shot snapshot timestamp consumed by this query (already
    /// taken out of `Database.snapshot_read_ts` when the session
    /// was captured).
    snapshot_read_ts: Option<u64>,
    write_concern: coordinode_core::txn::write_concern::WriteConcern,
    /// The session's vector consistency, when it set one.
    vector_consistency: Option<VectorConsistencyMode>,
    /// How long the statement waits for a vector index still being built,
    /// when the query names no bound of its own.
    vector_build_wait: Duration,
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

    /// No interactive transaction is held under this id.
    ///
    /// Either it was never opened, or it is already finished: committed,
    /// rolled back, aborted by a failed statement, or collected by the idle
    /// sweep. A caller cannot tell those apart from the id alone, and does not
    /// need to: in every case the server holds nothing for it.
    #[error("unknown transaction id {0}")]
    UnknownTransaction(u64),

    /// A commit refused by OCC validation: a key this transaction READ was
    /// modified by a concurrent writer after the snapshot was pinned at begin,
    /// so the transaction's view is no longer the state it would commit into.
    /// Nothing was applied; re-running the whole transaction from begin is the
    /// intended response, and the only failure of the commit for which that
    /// helps.
    #[error("transaction {id} conflicts with a concurrent write: {source_message}")]
    TransactionConflict { id: u64, source_message: String },

    /// A commit refused by a declared condition: the state the transaction's
    /// result depends on no longer holds, or a concurrent transaction holds an
    /// incompatible claim on it. Nothing was applied; re-running the whole
    /// transaction from begin re-reads that state.
    ///
    /// Separate from `TransactionConflict` because the two transactions need
    /// not touch a common key: disjoint writes can break one graph condition
    /// together, and reporting that as a write conflict names the wrong cause.
    /// A write conditioned on a record's version found a different one.
    /// Nothing was applied.
    ///
    /// The version that is there now is part of the error: a caller that has
    /// to decide whether to retry, merge or give up needs it, and making it
    /// read the record again would hand it a second race instead of an
    /// answer.
    #[error(
        "transaction {id}: record version mismatch, expected {expected:?}, \
         found {current:?}; nothing was applied"
    )]
    RevisionMismatch {
        /// The transaction the refusal belongs to.
        id: u64,
        /// The version the write was conditioned on; `None` required absence.
        expected: Option<u64>,
        /// The version the record has now; `None` means it is absent.
        current: Option<u64>,
    },

    #[error("transaction {id} was refused by an invariant: {reason}")]
    InvariantRefused {
        /// The transaction the refusal belongs to.
        id: u64,
        /// Which condition refused it, in the words of the condition.
        reason: String,
    },

    /// A transaction buffered more uncommitted data than it is allowed to, and
    /// was discarded. Splitting the work into smaller transactions is the fix.
    #[error("transaction {id} exceeded its buffer limit ({buffered} > {limit} bytes)")]
    TransactionTooLarge {
        id: u64,
        buffered: usize,
        limit: usize,
    },

    /// The storage engine is over its compaction-debt stop threshold and
    /// rejected the commit's writes. Nothing was applied; retrying after a
    /// short delay succeeds once compaction catches up.
    #[error(
        "write rejected: storage is over its compaction-debt stop threshold; \
         retry after compaction catches up"
    )]
    WriteBackpressure,

    /// The write reached a node that is not the leader, so it could not be
    /// replicated. Nothing was applied; the same write succeeds at the leader,
    /// whose id is carried here when the cluster has named one.
    #[error("not the leader{}", match leader_id {
        Some(id) => format!("; leader is node {id}"),
        None => String::from("; no leader known yet"),
    })]
    NotLeader {
        /// The node the cluster last named leader, if any. `None` while an
        /// election is in flight.
        leader_id: Option<u64>,
    },

    /// The write reached a member that does not run its group's version: it
    /// is read-only. Nothing was applied. Names both versions and the leader
    /// when known.
    #[error("this member is read-only: {0}")]
    Mismatched(coordinode_core::version::Mismatch),

    /// A snapshot read (`ReadConcern.at_timestamp`) older than the MVCC
    /// retention horizon. History that old may already be collected, so the
    /// read is refused; the same read succeeds at any timestamp from
    /// `oldest_readable` on. The horizon moves with the clock and
    /// `retention_window_secs` (`Database::set_retention_window`).
    #[error(
        "snapshot timestamp {requested} is older than the MVCC retention horizon: \
         history is readable from timestamp {oldest_readable} on"
    )]
    OutsideRetention {
        /// The timestamp the read asked for.
        requested: u64,
        /// The oldest timestamp still readable when the read ran.
        oldest_readable: u64,
    },

    #[error("{0}")]
    Other(String),
}

impl From<coordinode_modality::StoreError> for DatabaseError {
    fn from(e: coordinode_modality::StoreError) -> Self {
        DatabaseError::Execution(e.into())
    }
}

impl From<coordinode_query::frontend::FrontendError> for DatabaseError {
    fn from(e: coordinode_query::frontend::FrontendError) -> Self {
        use coordinode_query::frontend::FrontendError as FE;
        match e {
            FE::Parse(p) => DatabaseError::Parse(p),
            FE::Plan(p) => DatabaseError::Plan(p),
            FE::Semantic(errors) => DatabaseError::Semantic(
                errors
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
            ),
            FE::Message(m) => DatabaseError::Semantic(m),
        }
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Index builds are the one background task a Database can leave
        // behind: each holds an `Arc` on the engine and would go on inserting
        // into an index nobody can reach any more, against storage that is
        // closing under it, and keep the storage locked for the next
        // opening. Stop and join them here, as the oplog worker does
        // through its own Drop. A stopped key-shaped build stays recorded
        // and the next opening finishes it.
        self.vector_index_registry.cancel_all_builds();
        self.index_builds.shutdown();
    }
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
    /// Used by the server binary in cluster mode: the server creates
    /// a shared `StorageEngine` + `TimestampOracle` for the `RaftNode`, then
    /// passes the same engine + a `RaftProposalPipeline` here. The DrainBuffer
    /// and TTL reaper submit mutations through Raft for replication.
    pub fn from_engine(
        path: impl AsRef<Path>,
        engine: Arc<StorageEngine>,
        oracle: Arc<TimestampOracle>,
        pipeline: Arc<dyn coordinode_core::txn::proposal::ProposalPipeline>,
    ) -> Result<Self, DatabaseError> {
        // The engine stamps log entries with a bound read from its clock, and
        // the database takes commit timestamps from `oracle`: two clocks would
        // let a bound pass over a commit the other one just took.
        if engine
            .oracle()
            .is_some_and(|clock| !Arc::ptr_eq(&clock, &oracle))
        {
            return Err(DatabaseError::Other(
                "the database's timestamp oracle is not the one its engine was opened with"
                    .to_string(),
            ));
        }
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
    /// Returns a [`RepairReport`](crate::repair::RepairReport).
    /// `report.is_clean()` is `false` only if
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
        // Whether a Raft state machine applies the writes, so the vector
        // indexes follow the applied entries through a worker. Embedded mode
        // updates HNSW inline on the write path and applies no Raft entries.
        follow_raft_applies: bool,
    ) -> Result<Self, DatabaseError> {
        // Auto-repair on open. For an embedded engine with a retained
        // oplog journal, if a checkpoint exists, scrub and rebuild any corrupt
        // partition from that checkpoint + oplog replay BEFORE any state is read
        // below. Cluster engines (no journal) and journal-less in-memory engines
        // skip this; deployments that never checkpoint pay no scrub cost. A
        // repair failure is logged, not fatal — the caller can re-run
        // `verify_and_repair()` to inspect.
        if engine.has_journal() {
            let root = crate::repair::checkpoint_root(engine.data_dir());
            // An interrupted rebuild is repaired, or reported, even when its
            // checkpoint has gone since.
            if crate::repair::latest_checkpoint(&root).is_some()
                || !engine.pending_rebuilds()?.is_empty()
            {
                match crate::repair::verify_and_repair(&engine, &root) {
                    Ok(report) if !report.repaired.is_empty() => {
                        tracing::warn!(
                            repaired = ?report.repaired,
                            clean = report.clean_after,
                            "auto-repaired corrupt partitions on open"
                        );
                    }
                    Ok(report) if !report.is_clean() => {
                        tracing::error!(
                            corrupt = ?report.corrupt_partitions,
                            "partitions need a rebuild but no checkpoint exists; \
                             restore from an off-device backup"
                        );
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::error!(error = %e, "auto-repair on open failed; continuing")
                    }
                }
            }
        }

        // The index catalog, before anything in this open writes: a definition
        // this build cannot read refuses the open, rather than serving the
        // database without that index and the uniqueness it enforces.
        let index_registry = Arc::new(coordinode_query::index::IndexRegistry::new());
        index_registry.load_all(&engine)?;

        let proposal_id_gen = Arc::new(ProposalIdGenerator::with_base(fresh_proposal_id_base()));

        // The verified field dictionary, before anything reads stored data:
        // a damaged or missing one refuses the open rather than serving
        // stored properties as absent.
        let fields = Arc::new(fields::FieldDictionary::open(
            Arc::clone(&engine),
            Arc::clone(&pipeline),
            Arc::clone(&proposal_id_gen),
            Arc::clone(&oracle),
        )?);
        let opened_view = fields.current()?;

        // Follow the applied commits (Raft entries, or the local commits of a
        // store without Raft) from before the rebuild below, so no commit
        // falls between what the rebuild reads and what the worker receives.
        // Commits arriving during the rebuild are the rebuild's to fold; the
        // worker's copy of them is a harmless upsert. The worker is also what
        // takes a committed deletion out of the graph.
        // The events stay until the worker has folded them: a vector search
        // answers the nodes they wrote from the store instead of waiting.
        let applied = engine.subscribe_applied_retained(Partition::Node, APPLIED_QUEUE_CAPACITY);
        let vector_coverage = Arc::new(coordinode_query::index::IndexCoverage::new(
            applied.position(),
        ));

        // Load vector index definitions from schema: partition and rebuild
        // HNSW graphs from stored vectors (eager rebuild). The registry is
        // tier-backed: every index it registers persists f32 to LSM, which
        // stays the source of truth for reranking. The rebuild resolves the
        // ids its data was written with and registers none.
        let vector_index_registry = Arc::new(Self::load_vector_indexes(
            engine.clone(),
            &opened_view,
            1, /* shard_id */
        )?);

        // The rebuild above covered what the store held; the worker keeps the
        // indexes current with every entry applied from here on.
        vector_index_registry.set_coverage(Arc::clone(&vector_coverage));
        let vector_worker = crate::vector_worker::VectorIndexWorker::spawn(
            Arc::clone(&engine),
            applied,
            Arc::clone(&vector_index_registry),
            Arc::clone(&fields) as Arc<dyn FieldRegistrar>,
            vector_coverage,
            1, /* shard_id */
        );

        // The text indexes follow the applied commits the same way: the
        // subscription opens before they are rebuilt from the store, so no
        // commit falls between the two, and searches answer the writes the
        // worker has not folded from the store.
        let text_applied =
            engine.subscribe_applied_retained(Partition::Node, APPLIED_QUEUE_CAPACITY);
        let text_coverage = Arc::new(coordinode_query::index::IndexCoverage::new(
            text_applied.position(),
        ));
        let text_index_base = path.join("text_indexes");
        let text_index_registry = Arc::new(Self::load_text_indexes(
            &engine,
            &opened_view,
            1, /* shard_id */
            &text_index_base,
        )?);
        text_index_registry.set_coverage(Arc::clone(&text_coverage));
        let text_worker = crate::text_worker::TextIndexWorker::spawn(
            Arc::clone(&engine),
            text_applied,
            Arc::clone(&text_index_registry),
            Arc::clone(&fields) as Arc<dyn FieldRegistrar>,
            text_coverage,
            1, /* shard_id */
        );
        let index_builds = coordinode_query::index::IndexBuildService::new(
            Arc::new(index_builds::DatabaseBuilds {
                engine: Arc::clone(&engine),
                oracle: Arc::clone(&oracle),
                pipeline: Arc::clone(&pipeline),
                proposal_id_gen: Arc::clone(&proposal_id_gen),
                fields: Arc::clone(&fields),
                registry: Arc::clone(&index_registry),
                text_registry: Arc::clone(&text_index_registry),
                shard_id: 1,
            }),
            coordinode_query::index::IndexBuildConfig::default(),
        );

        // Every NodeId comes from a lease the log granted, taken on the first
        // CREATE and then kept one lease ahead in the background, so a crash
        // or a change of leader never reissues one (CE single-shard: hint 0).
        let reserver = id_lease::PrefetchingReserver::start(id_lease::LogLeaseReserver::new(
            Arc::clone(&engine),
            Arc::clone(&pipeline),
            Arc::clone(&proposal_id_gen),
            Arc::clone(&oracle),
        ))
        .map_err(|e| DatabaseError::Other(format!("start the NodeId lease worker: {e}")))?;
        let allocator = NodeIdAllocator::leased(0, Arc::new(reserver));

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
                // A fresh timestamp, held until the entry is drained: bounds
                // handed out since the crash may have passed the old one, and
                // the entry was never in the log, only in this node's
                // volatile state.
                let (commit_ts, hold) =
                    engine.pending_commits().obligate(|| oracle.next().as_raw());
                let entry = entry.restamped(Timestamp::from_raw(commit_ts), Box::new(hold));
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
                Arc::clone(&fields) as Arc<dyn FieldRegistrar>,
                Arc::clone(&oracle),
                Arc::clone(&pipeline),
                Arc::clone(&proposal_id_gen),
            ))
        } else {
            None
        };

        // The engine resolves a node from its id alone when it decides an
        // invariant claim, and node keys carry the shard ahead of the id. This
        // is where the two halves meet: the handle knows the shard, the engine
        // does the lookup.
        engine.set_node_shard(1);

        let db = Self {
            engine,
            fields,
            allocator,
            shard_id: 1,
            operations: None,
            query_registry: Arc::new(QueryRegistry::new()),
            nplus1_detector: Arc::new(NPlus1Detector::new()),
            dismissed: Arc::new(DismissedSet::new()),
            oracle,
            proposal_id_gen,
            pipeline,
            vector_consistency: None,
            vector_build_wait: DEFAULT_VECTOR_BUILD_WAIT,
            read_concern: coordinode_core::txn::read_concern::ReadConcernLevel::default(),
            snapshot_read_ts: None,
            cached_stats: Mutex::new(None),
            stats_generation: AtomicU64::new(0),
            stats_ttl: Duration::from_secs(STATS_CACHE_TTL_SECS),
            #[cfg(test)]
            stats_computations: AtomicU64::new(0),
            write_concern: coordinode_core::txn::write_concern::WriteConcern::default(),
            index_registry,
            index_builds,
            vector_index_registry,
            _vector_worker: vector_worker,
            text_index_registry,
            _text_worker: text_worker,
            extension_registry: ExtensionRegistry::new(),
            procedure_registry: ProcedureRegistry::with_builtins(),
            adaptive_config: AdaptiveConfig::default(),
            feedback_cache: FeedbackCache::default(),
            drain_buffer,
            nvme_write_buffer,
            _drain_handle: drain_handle,
            _ttl_reaper_handle: ttl_reaper_handle,
            cluster_mode: follow_raft_applies,
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
            interactive_begun: None,
        };
        // A store that owns its log takes up its unfinished index builds now;
        // a cluster member does it once it leads, since the builds are
        // written through the log.
        if !follow_raft_applies {
            db.resume_interrupted_index_builds()?;
        }
        Ok(db)
    }

    /// Load persisted vector index definitions from `schema:idx:*` and
    /// rebuild HNSW graphs by scanning stored vectors in the `node:` partition.
    ///
    /// Called during `Database::open()` for eager HNSW rebuild.
    fn load_vector_indexes(
        engine: Arc<StorageEngine>,
        fields: &FieldInterner,
        shard_id: u16,
    ) -> Result<coordinode_query::index::VectorIndexRegistry, DatabaseError> {
        use coordinode_query::index::IndexType;

        // Tier-backed registry: every registered HNSW index gets a
        // VectorTierHandle scoped to its `(label_id, property_id)`.
        let registry =
            coordinode_query::index::VectorIndexRegistry::with_vector_tier(engine.clone());

        // A definition the listing cannot read refuses the open: the
        // database is not served without one of its indexes.
        let hnsw_defs: Vec<_> = coordinode_query::index::ops::list_index_definitions(&engine)?
            .into_iter()
            .filter(|def| def.index_type == IndexType::Hnsw && def.vector_config.is_some())
            .collect();
        if hnsw_defs.is_empty() {
            return Ok(registry);
        }

        Self::register_and_populate_hnsw(
            &registry,
            fields,
            &engine,
            shard_id,
            &hnsw_defs,
            PopulateMode::Blocking,
        );
        Ok(registry)
    }

    /// Discover vector index definitions that were replicated into the
    /// Schema partition after this Database opened (a follower applying
    /// a leader's CREATE VECTOR INDEX) and bring them live: register in
    /// the in-memory registry and rebuild the local HNSW from stored
    /// nodes. Returns the number of indexes brought up. Cluster
    /// deployments call this alongside [`Self::refresh_btree_indexes`]
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
            &self.fields.current()?,
            &self.engine,
            self.shard_id,
            &new_defs,
            PopulateMode::Background,
        );
        Ok(new_defs.len())
    }

    /// Bring this member's text indexes in line with the definitions the
    /// Schema partition holds: register and rebuild from the store each one
    /// replicated after the database opened (a follower applying a leader's
    /// CREATE TEXT INDEX), and drop each one whose definition is gone.
    /// Returns how many indexes it registered or dropped. Cluster
    /// deployments call this whenever the applied index advances, beside
    /// [`Self::refresh_vector_indexes`].
    pub fn refresh_text_indexes(&self) -> Result<usize, DatabaseError> {
        use coordinode_query::index::IndexType;
        let stored: Vec<_> = coordinode_query::index::ops::list_index_definitions(&self.engine)?
            .into_iter()
            .filter(|d| d.index_type == IndexType::Text && d.text_config.is_some())
            .collect();
        let registered = self.text_index_registry.definitions();

        let mut changed = 0usize;
        for def in &registered {
            if !stored.iter().any(|d| d.id == def.id) {
                self.text_index_registry
                    .unregister(&def.label, def.property());
                changed += 1;
            }
        }
        let new_defs: Vec<_> = stored
            .into_iter()
            .filter(|d| !registered.iter().any(|r| r.id == d.id))
            .collect();
        if !new_defs.is_empty() {
            Self::populate_text_indexes(
                &self.text_index_registry,
                &self.engine,
                &self.fields.current()?,
                self.shard_id,
                &new_defs,
            );
            changed += new_defs.len();
        }
        Ok(changed)
    }

    /// Register the given HNSW definitions in `registry` and build them
    /// beside whatever writes are landing (shared by the open-time loader
    /// and the cluster refresh path).
    ///
    /// On open the build blocks until the indexes are whole, one scan for
    /// all of them. A replica bringing up an index another member defined
    /// builds each one in the background, owned by the registry so a drop
    /// cancels it; the caller runs on the async runtime and must not block.
    fn register_and_populate_hnsw(
        registry: &coordinode_query::index::VectorIndexRegistry,
        fields: &FieldInterner,
        engine: &Arc<StorageEngine>,
        shard_id: u16,
        hnsw_defs: &[coordinode_query::index::IndexDefinition],
        mode: PopulateMode,
    ) {
        // Resolve (label, property) to the ids the definition was published
        // with, build per-index tier handles, then register each HNSW with
        // its tier bound. A rebuild only resolves: it registers nothing, so
        // it can never give a name an id its data was not written with.
        let mut resolved = Vec::with_capacity(hnsw_defs.len());
        for def in hnsw_defs {
            let (Some(label_id), Some(property_id)) =
                (fields.lookup(&def.label), fields.lookup(def.property()))
            else {
                tracing::error!(
                    index = %def,
                    label = %def.label,
                    property = %def.property(),
                    "vector index names have no field binding; the index stays offline"
                );
                continue;
            };
            let tier = registry.tier_handle(label_id, property_id);
            registry.register_for_build(def.clone(), tier);
            resolved.push((def, property_id));
        }

        let mut members = Vec::with_capacity(resolved.len());
        for (def, field_id) in resolved {
            let (Some(hnsw), Some(health)) = (
                registry.get(&def.label, def.property()),
                registry.health_handle(&def.label, def.property()),
            ) else {
                tracing::warn!(index = %def, "registered vector index has no graph to build");
                continue;
            };
            members.push((def, field_id, hnsw, health));
        }

        match mode {
            PopulateMode::Blocking => {
                let targets: Vec<coordinode_query::index::BuildTarget<'_>> = members
                    .iter()
                    .map(
                        |(def, field_id, hnsw, health)| coordinode_query::index::BuildTarget {
                            hnsw: hnsw.as_ref(),
                            health: health.as_ref(),
                            label: &def.label,
                            field_id: *field_id,
                        },
                    )
                    .collect();
                let token = registry.new_build_token();
                // On its own thread: the build may pause the Raft applies,
                // which blocks, and the caller can be on the async runtime.
                let outcome = std::thread::scope(|scope| {
                    scope
                        .spawn(|| {
                            coordinode_query::index::VectorBuild {
                                engine,
                                token: &token,
                                shard_id,
                                targets: &targets,
                            }
                            .run()
                        })
                        .join()
                });
                match outcome {
                    Ok(Ok(outcome)) => tracing::info!(
                        indexes = targets.len(),
                        ?outcome,
                        "rebuilt HNSW indexes on open"
                    ),
                    Ok(Err(reason)) => {
                        for (def, _, _, health) in &members {
                            tracing::warn!(index = %def, %reason, "HNSW rebuild on open failed");
                            health.mark_offline(reason.clone());
                        }
                    }
                    Err(_) => {
                        for (def, _, _, health) in &members {
                            tracing::warn!(index = %def, "HNSW rebuild on open panicked");
                            health.mark_offline("panic in the rebuild on open".to_string());
                        }
                    }
                }
            }
            PopulateMode::Background => {
                for (def, field_id, hnsw, health) in members {
                    Self::spawn_replica_build(
                        registry, engine, shard_id, def, field_id, hnsw, health,
                    );
                }
            }
        }
    }

    /// Build `def` on a replica in the background, the way a `CREATE VECTOR
    /// INDEX` does on the member that ran it: the registry owns the thread,
    /// so dropping the index cancels and joins it.
    fn spawn_replica_build(
        registry: &coordinode_query::index::VectorIndexRegistry,
        engine: &Arc<StorageEngine>,
        shard_id: u16,
        def: &coordinode_query::index::IndexDefinition,
        field_id: u32,
        hnsw: Arc<std::sync::RwLock<coordinode_vector::hnsw::HnswIndex>>,
        health: Arc<coordinode_vector::health::HealthSignal>,
    ) {
        let token = registry.new_build_token();
        let build_token = token.clone();
        let engine = Arc::clone(engine);
        let name = def.to_string();
        let label = def.label.clone();
        // The index was registered rebuilding; without a build it would stay
        // so, and a blocked reader would wait on it until its timeout.
        let unbuilt = Arc::clone(&health);
        let spawned = std::thread::Builder::new()
            .name(format!("vec-replica-{name}"))
            .spawn(move || {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    coordinode_query::index::VectorBuild {
                        engine: engine.as_ref(),
                        token: &build_token,
                        shard_id,
                        targets: &[coordinode_query::index::BuildTarget {
                            hnsw: hnsw.as_ref(),
                            health: health.as_ref(),
                            label: &label,
                            field_id,
                        }],
                    }
                    .run()
                }));
                match outcome {
                    Ok(Ok(outcome)) => {
                        tracing::info!(index = %name, ?outcome, "vector index built on replica");
                    }
                    Ok(Err(reason)) => {
                        tracing::warn!(index = %name, %reason, "vector index build on replica failed");
                        health.mark_offline(reason);
                    }
                    Err(_) => {
                        tracing::warn!(index = %name, "vector index build on replica panicked");
                        health.mark_offline("panic in the replica build".to_string());
                    }
                }
            });
        match spawned {
            Ok(thread) => {
                registry.register_build(def, &token, thread);
            }
            Err(e) => {
                tracing::warn!(index = %def, error = %e, "could not spawn the replica build");
                unbuilt.mark_offline(format!("could not start the build: {e}"));
            }
        }
    }

    /// Load the persisted text index definitions and rebuild their tantivy
    /// indexes by scanning the stored nodes in the `node:` partition.
    fn load_text_indexes(
        engine: &StorageEngine,
        interner: &FieldInterner,
        shard_id: u16,
        base_dir: &Path,
    ) -> Result<coordinode_query::index::TextIndexRegistry, DatabaseError> {
        use coordinode_query::index::IndexType;

        let registry = coordinode_query::index::TextIndexRegistry::new(base_dir);

        // A definition the listing cannot read refuses the open: the
        // database is not served without one of its indexes.
        let text_defs: Vec<_> = coordinode_query::index::ops::list_index_definitions(engine)?
            .into_iter()
            .filter(|def| def.index_type == IndexType::Text && def.text_config.is_some())
            .collect();

        Self::populate_text_indexes(&registry, engine, interner, shard_id, &text_defs);
        Ok(registry)
    }

    /// Register `text_defs` in `registry` and rebuild each from the nodes
    /// stored on `shard_id`. A rebuild replaces what the index held, so a
    /// document of a node deleted while the process was down does not
    /// survive it.
    fn populate_text_indexes(
        registry: &coordinode_query::index::TextIndexRegistry,
        engine: &StorageEngine,
        interner: &FieldInterner,
        shard_id: u16,
        text_defs: &[coordinode_query::index::IndexDefinition],
    ) {
        let mut total_docs = 0usize;
        for def in text_defs {
            if let Err(e) = registry.register(def.clone()) {
                tracing::warn!("failed to register text index {def}: {e}");
                continue;
            }
            for property in &def.properties {
                let rebuilt = registry.rebuild_index(&def.label, property, || {
                    coordinode_query::index::text_registry::stored_texts(
                        engine,
                        shard_id,
                        interner,
                        &def.label,
                        property,
                        coordinode_query::executor::runner::wall_clock_us(),
                    )
                });
                match rebuilt {
                    Ok(docs) => total_docs += docs,
                    Err(e) => {
                        tracing::warn!("failed to rebuild text index {def} on {property}: {e}")
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
    }

    /// Snapshot the current session defaults into a [`QuerySession`].
    ///
    /// Consumes any one-shot `snapshot_read_ts` so a second query in
    /// the same session sees the default (latest) read timestamp.
    fn capture_session(&mut self) -> QuerySession {
        QuerySession {
            read_concern: self.read_concern,
            snapshot_read_ts: self.snapshot_read_ts.take(),
            write_concern: self.write_concern,
            vector_consistency: self.vector_consistency,
            vector_build_wait: self.vector_build_wait,
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
        match Self::parse_session_set(query) {
            Some(SessionSetting::VectorConsistency(mode)) => {
                self.vector_consistency = Some(mode);
                true
            }
            Some(SessionSetting::VectorBuildWait(wait)) => {
                self.vector_build_wait = wait;
                true
            }
            None => false,
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

    /// Add a procedure `CALL` can run and `dbms.procedures()` lists. Called at
    /// setup by an enterprise layer or an embedder.
    ///
    /// # Errors
    ///
    /// [`ProcedureError::DuplicateName`] when a procedure of that name is
    /// already registered, built-in or not; the existing one stays.
    pub fn register_procedure(
        &mut self,
        procedure: Arc<dyn Procedure>,
    ) -> Result<(), ProcedureError> {
        self.procedure_registry.register(procedure)
    }

    /// The procedures `CALL` dispatches to.
    pub fn procedures(&self) -> &ProcedureRegistry {
        &self.procedure_registry
    }

    /// Execute a Cypher query and return result rows.
    ///
    /// Automatically tracks query fingerprint and execution time in the
    /// query advisor registry for performance analysis.
    pub fn execute_cypher(&mut self, query: &str) -> Result<Vec<Row>, DatabaseError> {
        if self.try_apply_session_set(query) {
            return Ok(Vec::new());
        }
        let session = self.capture_session();
        self.execute_cypher_impl(query, None, None, &session, TxnMode::AutoCommit, &mut None)
            .map(|(rows, _, _)| rows)
    }

    /// Execute a SQL statement (`SELECT` / `INSERT`) against the relational
    /// TABLE modality.
    ///
    /// SQL is parsed and lowered natively into the same neutral `LogicalPlan`
    /// as Cypher via the [`SqlFrontend`](coordinode_query::sql::SqlFrontend),
    /// then run through the identical dialect-agnostic execute path. The lowered
    /// plan is seeded into the plan cache so that shared path picks it up
    /// instead of cypher-parsing the text.
    pub fn execute_sql(&mut self, query: &str) -> Result<Vec<Row>, DatabaseError> {
        use coordinode_query::frontend::QueryFrontend;
        let parsed = coordinode_query::sql::SqlFrontend::new().parse(query)?;
        self.plan_cache.put(
            query.to_string(),
            Arc::new(CachedPlan {
                canonical: parsed.canonical,
                fingerprint: parsed.fingerprint,
                plan: parsed.plan,
                vector_consistency_hinted: parsed.vector_consistency_hinted,
                vector_build_wait: parsed.vector_build_wait,
            }),
        );
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
    /// applied only to this call's query session; session defaults
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
        if let Some(wc) = write_concern {
            session.write_concern = wc;
        }

        let params = params.filter(|p| !p.is_empty());
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

    /// A fresh read timestamp for a statement, pinned in the same step it is
    /// allocated. The pin goes into `hold`, which the caller keeps for the
    /// statement; pinned later, at the first read, the watermark can already
    /// have passed it and the read would be refused.
    fn fresh_pinned_read_ts(
        &self,
        hold: &mut Option<coordinode_storage::engine::coordinator::SnapshotPin>,
    ) -> Timestamp {
        let (seqno, pin) = self.engine.pin_new_snapshot(|| self.oracle.next().as_raw());
        *hold = pin;
        Timestamp::from_raw(seqno)
    }

    /// Begin an interactive multi-statement transaction.
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
        // The view is the engine's complete snapshot, not the timestamp just
        // allocated: a fresh number from the clock can already cover a commit
        // that has not applied, and a transaction starting there would see
        // neither that write nor any sign of it. `read_ts` stays the
        // allocated value because it identifies this attempt.
        let mut txn = coordinode_storage::engine::transaction::Transaction::begin(
            &self.engine,
            Some(&self.oracle),
            read_ts,
        );
        let state = txn.take_state();
        self.interactive_txns
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id, (state, Instant::now()));
        if let Some(begun) = &self.interactive_begun {
            begun();
        }
        id
    }

    /// Call `begun` after every interactive transaction opens. The server's
    /// idle reaper uses it to sleep while no transaction is open.
    pub fn set_interactive_begun_hook(&mut self, begun: Arc<dyn Fn() + Send + Sync>) {
        self.interactive_begun = Some(begun);
    }

    /// Run one statement of an interactive transaction.
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
        self.execute_in_transaction_with(txn_id, query, params, &StatementOptions::default())
    }

    /// [`Self::execute_in_transaction`] with the vector settings in `options`
    /// in place of the database's. Its read and write concerns are the
    /// transaction's, fixed when it began and when it commits, so those
    /// fields of `options` are not read here.
    pub fn execute_in_transaction_with(
        &self,
        txn_id: u64,
        query: &str,
        params: Option<std::collections::HashMap<String, coordinode_core::graph::types::Value>>,
        options: &StatementOptions,
    ) -> Result<Vec<Row>, DatabaseError> {
        let state = {
            let mut reg = self
                .interactive_txns
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            match reg.remove(&txn_id) {
                Some((state, _touched)) => state,
                None => return Err(DatabaseError::UnknownTransaction(txn_id)),
            }
        };
        let session = QuerySession {
            read_concern: self.read_concern,
            snapshot_read_ts: None,
            write_concern: self.write_concern,
            vector_consistency: options.vector_consistency.or(self.vector_consistency),
            vector_build_wait: options.vector_build_wait.unwrap_or(self.vector_build_wait),
            after_commit_generation: 0,
        };
        let params = params.filter(|p| !p.is_empty());
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
                return Err(DatabaseError::TransactionTooLarge {
                    id: txn_id,
                    buffered,
                    limit: self.max_interactive_txn_bytes,
                });
            }
            self.interactive_txns
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(txn_id, (state, Instant::now()));
        }
        Ok(rows)
    }

    /// Commit an interactive transaction: validate the accumulated
    /// write set, assign `commit_ts`, and persist every buffered mutation in a
    /// single proposal. The handle is consumed (removed from the registry)
    /// whether commit succeeds or fails; on a write conflict the client
    /// retries the whole transaction from `begin`. Returns the
    /// [`CommitReceipt`](coordinode_core::txn::transaction::CommitReceipt):
    /// `commit_ts` (the HLC commit timestamp every mutation landed at: the
    /// `AS OF TIMESTAMP` anchor for this write, and a changed-keys scan from it
    /// includes this commit) plus the committed Raft index in cluster mode
    /// (`None` in embedded mode, where there is no Raft log).
    pub fn commit_transaction(
        &self,
        txn_id: u64,
    ) -> Result<coordinode_core::txn::transaction::CommitReceipt, DatabaseError> {
        let state = self
            .interactive_txns
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&txn_id)
            .map(|(s, _)| s);
        let Some(state) = state else {
            return Err(DatabaseError::UnknownTransaction(txn_id));
        };
        let mut txn = coordinode_storage::engine::transaction::Transaction::resume(
            &self.engine,
            Some(&self.oracle),
            state,
        );
        // Constraints judge each node as the whole transaction leaves it,
        // not as one of its statements did.
        let fields = self.fields.current()?;
        coordinode_query::executor::runner::check_post_state(&mut txn, &self.engine, &fields)?;
        let wc = self.write_concern;
        let commit_ctx = coordinode_storage::engine::transaction::CommitContext {
            write_concern: &wc,
            pipeline: Some(self.pipeline.as_ref()),
            id_gen: Some(&self.proposal_id_gen),
            drain_buffer: Some(&self.drain_buffer),
            nvme_write_buffer: self.nvme_write_buffer.as_deref(),
        };
        // Split by variant, because the three failures call for opposite
        // reactions. Only an OCC conflict is worth retrying: the read-set was
        // invalidated by a concurrent writer and re-running the transaction
        // resolves it. A storage or pipeline failure retried immediately just
        // fails again, and telling the client otherwise would be a lie with a
        // retry storm attached.
        use coordinode_storage::engine::transaction::CommitError;
        let outcome = txn.commit(&commit_ctx).map_err(|e| match e {
            CommitError::Conflict(detail) => DatabaseError::TransactionConflict {
                id: txn_id,
                source_message: detail,
            },
            CommitError::Storage(s) => DatabaseError::Storage(s),
            CommitError::Serialization(detail) => {
                DatabaseError::Other(format!("commit failed: {detail}"))
            }
            // Retryable like a conflict, but with a delay: the engine sheds
            // writes until compaction drains its debt.
            CommitError::Backpressure => DatabaseError::WriteBackpressure,
            // Retryable at a different address: the leader.
            CommitError::NotLeader { leader_id } => DatabaseError::NotLeader { leader_id },
            // Retryable at the leader, or here once this member is updated.
            CommitError::Mismatched(m) => DatabaseError::Mismatched(m),
            // Not retryable: the same statement stages the same deltas.
            CommitError::CounterOverflow { key } => DatabaseError::Other(format!(
                "counter '{key}' would leave the i64 range; nothing was written"
            )),
            // Not retryable either: the transaction has to change fewer entries.
            e @ CommitError::IndexFanOut { .. } => DatabaseError::Other(e.to_string()),
            // Retryable from begin, like a conflict, but for a different
            // reason: the condition is re-evaluated against the state the
            // retry reads.
            CommitError::InvariantRefused { reason } => {
                DatabaseError::InvariantRefused { id: txn_id, reason }
            }
            // Not retryable: the value is held, by the node named. The
            // statements that claimed it are gone, so the index is found by
            // the generation the refusal names.
            CommitError::UniqueValueHeld {
                generation,
                values,
                holder,
            } => match self
                .index_registry
                .all()
                .into_iter()
                .find(|index| index.generation == generation)
            {
                Some(index) => DatabaseError::Execution(
                    coordinode_query::index::UniqueViolation::new(&index, &values, holder).into(),
                ),
                None => DatabaseError::InvariantRefused {
                    id: txn_id,
                    reason: format!(
                        "a unique value of {generation} is held by node {}",
                        holder.to_element_id()
                    ),
                },
            },
            // Retryable once the build is done.
            CommitError::UniquenessUnresolved { generation, limit } => {
                let index = self
                    .index_registry
                    .all()
                    .into_iter()
                    .find(|index| index.generation == generation)
                    .map_or_else(|| generation.to_string(), |index| index.to_string());
                DatabaseError::Execution(
                    coordinode_query::executor::runner::ExecutionError::UniquenessUnresolved {
                        index,
                        limit,
                    },
                )
            }
            // Retryable, but whether retrying is the right answer is the
            // caller's to decide: the record moved, and what that means
            // depends on what it was writing. The new version travels with
            // the refusal so that decision needs no second read.
            CommitError::RevisionMismatch { expected, current } => {
                DatabaseError::RevisionMismatch {
                    id: txn_id,
                    expected,
                    current,
                }
            }
        })?;
        // An interactive transaction is always opened against the oracle
        // (`begin_transaction`), so the storage commit is never on the legacy
        // no-oracle path and always carries a commit timestamp. A missing one
        // is an engine invariant violation, surfaced as an error rather than
        // a made-up timestamp the host would then trust as a cursor.
        let commit_ts = outcome.commit_ts.ok_or_else(|| {
            DatabaseError::Other(format!(
                "transaction {txn_id} committed without a commit timestamp"
            ))
        })?;
        Ok(coordinode_core::txn::transaction::CommitReceipt {
            commit_ts,
            applied_index: outcome.applied_index,
        })
    }

    /// Roll back an interactive transaction: discard all buffered
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
            Err(DatabaseError::UnknownTransaction(txn_id))
        }
    }

    /// The version of a node: the timestamp of the commit that last wrote it,
    /// or `None` when there is no such node.
    ///
    /// This is the number a conditional write is stated against. It is the
    /// commit timestamp rather than a counter of the node's own, so the same
    /// value a write receipt already returns is the one a later write
    /// conditions on.
    pub fn node_version(
        &self,
        node_id: coordinode_core::graph::node::NodeId,
    ) -> Result<Option<u64>, DatabaseError> {
        let key = coordinode_core::graph::node::encode_node_key(self.shard_id, node_id);
        Ok(self
            .engine
            .record_version(coordinode_storage::engine::partition::Partition::Node, &key)?)
    }

    /// Commit this transaction only while the node is still at `expected`.
    ///
    /// `None` requires the node not to exist, which is the create-if-absent
    /// form of the same condition. The condition is checked when the
    /// transaction commits, against the state at that moment, and a mismatch
    /// refuses the whole transaction with the version that is there instead.
    ///
    /// Stating it does not read the node or write anything by itself: it is
    /// the caller saying what its statements were built on. A caller that
    /// wants the ordinary read-modify-write pairs this with
    /// [`Self::node_version`] before the statements that change the node.
    pub fn expect_node_version(
        &self,
        txn_id: u64,
        node_id: coordinode_core::graph::node::NodeId,
        expected: Option<u64>,
    ) -> Result<(), DatabaseError> {
        let key = coordinode_core::graph::node::encode_node_key(self.shard_id, node_id);
        let mut reg = self
            .interactive_txns
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let Some((state, touched)) = reg.get_mut(&txn_id) else {
            return Err(DatabaseError::UnknownTransaction(txn_id));
        };
        *touched = Instant::now();
        state.expect_version(
            coordinode_storage::engine::partition::Partition::Node,
            &key,
            expected,
        )?;
        Ok(())
    }

    /// Drop interactive transactions idle longer than `timeout`. An open
    /// transaction pins an MVCC snapshot and buffers writes in memory, so an
    /// abandoned one would leak retention and leader memory. Called
    /// opportunistically on `begin`; the server's reaper also runs it at the
    /// returned instant, so reaping does not wait for the next `begin`.
    ///
    /// Returns when the earliest transaction still open becomes idle for
    /// `timeout`, or `None` when none is open.
    pub fn reap_idle_transactions(&self, timeout: Duration) -> Option<Instant> {
        let now = Instant::now();
        let mut txns = self
            .interactive_txns
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        txns.retain(|_, (_, touched)| now.duration_since(*touched) < timeout);
        txns.values().map(|(_, touched)| *touched + timeout).min()
    }

    /// Run one pass of the COMPUTED TTL reaper now, as the background reaper
    /// runs it: expired records are deleted by committed transactions, which
    /// replicate and reach every follower of the applied commits (the vector
    /// indexes among them). For a host that wants expiry applied without
    /// waiting for the reaper's next interval.
    ///
    /// # Errors
    ///
    /// The field dictionary could not be read; nothing was reaped.
    pub fn reap_expired(
        &self,
    ) -> Result<coordinode_query::index::ttl_reaper::ComputedTtlReapResult, DatabaseError> {
        coordinode_query::index::ttl_reaper::reap_pass(
            &self.engine,
            self.shard_id,
            coordinode_query::index::ttl_reaper::TtlReaperConfig::default().batch_size,
            self.fields.as_ref(),
            &self.oracle,
            self.pipeline.as_ref(),
            &self.proposal_id_gen,
        )
        .map_err(|e| DatabaseError::Other(format!("field dictionary: {e}")))
    }

    /// Default idle timeout for an open interactive transaction.
    pub const DEFAULT_INTERACTIVE_TXN_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

    /// Default max buffered bytes per interactive transaction (256 MiB).
    pub const DEFAULT_MAX_INTERACTIVE_TXN_BYTES: usize = 256 * 1024 * 1024;

    /// Vector-index builds running on this node right now, with live progress.
    ///
    /// Each entry is `(index, label, property, health)`, where `health` is the
    /// same `Rebuilding { progress, eta_ms, indexed_hlc }` the search path and
    /// the metrics exporter read. An empty result means no build is in flight,
    /// which after a [`Self::cancel_index_build`] or a `DROP VECTOR INDEX` is
    /// a guarantee, not a sample: both join the build before returning.
    ///
    /// This is one node's view. An index partitioned across nodes is built per
    /// node, and the cluster-wide picture is the coordinator's aggregation of
    /// these per-node answers.
    pub fn index_builds(
        &self,
    ) -> Vec<(
        String,
        String,
        String,
        Option<coordinode_vector::health::IndexHealthState>,
    )> {
        self.vector_index_registry.build_progress()
    }

    /// Cancel the index build `operation` (the generation it fills, as a
    /// creating statement and [`Self::index_build_status`] report it).
    /// Returns whether a build was stopped; `false` when it already had an
    /// outcome or no build has the operation.
    ///
    /// A key-shaped build is cancelled through its record, in the commit
    /// that also withdraws the index it was creating with the constraint
    /// that owns it, or keeps an index it was rebuilding, failed. It cannot
    /// undo a publication that committed first. A vector build stops and is
    /// joined: the index keeps what the build inserted (an insert is an
    /// upsert, so a later build re-covers those nodes once) and the build
    /// writes nothing afterwards.
    ///
    /// # Errors
    ///
    /// The cancellation could not be committed.
    pub fn cancel_index_build(
        &self,
        operation: coordinode_query::index::GenerationId,
    ) -> Result<bool, DatabaseError> {
        if self
            .index_builds
            .cancel(operation)
            .map_err(DatabaseError::Other)?
        {
            return Ok(true);
        }
        Ok(self.vector_index_registry.cancel_build(operation))
    }

    /// Every index build the catalog records or an executor of this process
    /// holds: the record (index, generation, what it does on failure, its
    /// state) and, while this process runs it, whether it waits for a seat
    /// or for older transactions, or how far it has indexed.
    ///
    /// # Errors
    ///
    /// The records could not be read.
    pub fn index_build_status(
        &self,
    ) -> Result<Vec<coordinode_query::index::BuildStatus>, DatabaseError> {
        Ok(self.index_builds.builds()?)
    }

    /// The index build `operation`, after waiting up to `wait` for its
    /// outcome; `None` when no build has the operation. Waiting cancels
    /// nothing: a waiter that gives up leaves the build running.
    ///
    /// # Errors
    ///
    /// The record could not be read.
    pub fn index_build(
        &self,
        operation: coordinode_query::index::GenerationId,
        wait: Duration,
    ) -> Result<Option<coordinode_query::index::BuildStatus>, DatabaseError> {
        Ok(self.index_builds.status(operation, wait)?)
    }

    /// The duplicates the index build `operation` repaired (`ON DUPLICATE
    /// RENAME`), in node order: each node, property, and the value replaced
    /// by which. Every one is a committed change of the node, kept whatever
    /// the build's outcome.
    ///
    /// # Errors
    ///
    /// The records could not be read.
    pub fn index_build_repairs(
        &self,
        operation: coordinode_query::index::GenerationId,
    ) -> Result<Vec<coordinode_query::index::DuplicateRepairRecord>, DatabaseError> {
        use coordinode_modality::IndexStore as _;
        Ok(coordinode_modality::LocalIndexStore::new(&self.engine)
            .list_repairs(operation)
            .map_err(coordinode_query::executor::runner::ExecutionError::from)?)
    }

    /// How index builds run: how many fill indexes at once and how long a
    /// backfill waits for the transactions opened before its index.
    pub fn index_build_config(&self) -> coordinode_query::index::IndexBuildConfig {
        self.index_builds.config()
    }

    /// Retune index builds while the database runs (server config wiring).
    pub fn set_index_build_config(&self, config: coordinode_query::index::IndexBuildConfig) {
        self.index_builds.set_config(config);
    }

    /// Set the interactive-transaction idle timeout (server config wiring).
    pub fn set_interactive_idle_timeout(&mut self, timeout: Duration) {
        self.interactive_idle_timeout = timeout;
    }

    /// Set the per-interactive-transaction buffered-bytes ceiling (server
    /// config wiring).
    pub fn set_max_interactive_txn_bytes(&mut self, bytes: usize) {
        self.max_interactive_txn_bytes = bytes;
    }

    /// Set the MVCC time-travel retention window at runtime: how far back
    /// `AS OF TIMESTAMP` / `at_timestamp` reads stay answerable. The GC
    /// watermark moves with the next commit or compaction; narrowing the
    /// window releases history and the storage that held it, widening it
    /// cannot bring back history an earlier compaction already released.
    /// Configured at open via `StorageConfig::retention_window_secs`
    /// (default seven days).
    pub fn set_retention_window(&self, window: Duration) {
        self.engine.set_retention_window(window);
    }

    /// The configured MVCC time-travel retention window.
    pub fn retention_window(&self) -> Duration {
        self.engine.retention_window()
    }

    /// The oldest timestamp a time-travel read can currently ask for. Moves
    /// forward with the clock; a read older than this is refused with
    /// [`DatabaseError::OutsideRetention`] (or its executor counterpart for
    /// `AS OF TIMESTAMP`).
    ///
    /// A read at timestamp `T` is served from the snapshot at `T + 1`
    /// (inclusive `AS OF`), so `T` is readable exactly when `T + 1` reaches the
    /// engine's first readable seqno — one below it. The clamp at zero is the
    /// bottom of the timestamp domain, not an overflow guard: a horizon of zero
    /// means nothing has been collected, and no timestamp sits below zero to
    /// report.
    pub fn oldest_readable_timestamp(&self) -> Timestamp {
        Timestamp::from_raw(self.engine.oldest_readable_seqno().saturating_sub(1))
    }

    /// Set session-level vector consistency mode.
    ///
    /// Equivalent to `SET vector_consistency = 'snapshot'` in Cypher. It
    /// replaces the mode a query's `read_consistency` implies; a query that
    /// names its own mode in a hint keeps it.
    pub fn set_vector_consistency(&mut self, mode: VectorConsistencyMode) {
        self.vector_consistency = Some(mode);
    }

    /// Get current session-level vector consistency mode; `current` when the
    /// session has not set one.
    pub fn vector_consistency(&self) -> VectorConsistencyMode {
        self.vector_consistency.unwrap_or_default()
    }

    /// Set how long a query waits for a vector index still being built,
    /// under the `block` policy, before it fails.
    ///
    /// Equivalent to `SET vector_build_wait = '5s'` in Cypher; a query that
    /// names its own bound in a `vector_build_wait` hint keeps it. Takes
    /// effect from the next query. Default [`DEFAULT_VECTOR_BUILD_WAIT`].
    pub fn set_vector_build_wait(&mut self, wait: Duration) {
        self.vector_build_wait = wait;
    }

    /// How long a query waits for a vector index still being built, when it
    /// names no bound of its own.
    pub fn vector_build_wait(&self) -> Duration {
        self.vector_build_wait
    }

    /// Set the retired-memory budget of every vector index: the bytes of
    /// replaced neighbour lists an index lets wait for reclamation before
    /// its inserts and removals hold off until the searches that may still
    /// read them finish. Bounds that memory under a stalled search by
    /// slowing writers, never by dropping a write. Takes effect on the next
    /// write, for the indexes created later too. Default
    /// [`coordinode_vector::hnsw::DEFAULT_RETIRED_BYTES_BUDGET`].
    pub fn set_vector_retired_bytes_budget(&self, bytes: usize) {
        self.vector_index_registry.set_retired_bytes_budget(bytes);
    }

    /// The retired-memory budget of the vector indexes.
    pub fn vector_retired_bytes_budget(&self) -> usize {
        self.vector_index_registry.retired_bytes_budget()
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

    /// Set the session-level write concern (`w`, `journal` and timeout).
    pub fn set_write_concern(&mut self, wc: coordinode_core::txn::write_concern::WriteConcern) {
        self.write_concern = wc;
    }

    /// Override the AFTER COMMIT trigger dispatcher knobs (cascade-depth cap +
    /// default retry policy). The server calls this once at startup from
    /// `coordinode.conf` / CLI flags; embedded callers may tune it directly.
    pub fn set_trigger_dispatch_config(&mut self, cfg: TriggerDispatchConfig) {
        self.trigger_dispatch_config = cfg;
    }

    /// Get the current session-level write concern.
    pub fn write_concern(&self) -> coordinode_core::txn::write_concern::WriteConcern {
        self.write_concern
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
    /// Builds a per-call query session from the Database's session
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
        self.execute_cypher_shared_with(
            query,
            params,
            source,
            &StatementOptions {
                read_concern: read_concern.cloned(),
                write_concern: write_concern.copied(),
                ..StatementOptions::default()
            },
        )
    }

    /// [`Self::execute_cypher_shared`] with every per-statement setting in
    /// `options`: what a server runs a client session's statement with, so
    /// the session's settings apply to its own statements alone.
    pub fn execute_cypher_shared_with(
        &self,
        query: &str,
        params: Option<std::collections::HashMap<String, coordinode_core::graph::types::Value>>,
        source: Option<&SourceContext>,
        options: &StatementOptions,
    ) -> Result<CypherResult, DatabaseError> {
        if Self::parse_session_set(query).is_some() {
            return Err(DatabaseError::Semantic(
                "session SET commands require exclusive Database access; \
                 use execute_cypher_full"
                    .to_string(),
            ));
        }

        let mut session = QuerySession {
            read_concern: self.read_concern,
            snapshot_read_ts: None,
            write_concern: self.write_concern,
            vector_consistency: options.vector_consistency.or(self.vector_consistency),
            vector_build_wait: options.vector_build_wait.unwrap_or(self.vector_build_wait),
            after_commit_generation: 0,
        };
        if let Some(rc) = &options.read_concern {
            rc.validate()
                .map_err(|e| DatabaseError::Semantic(e.to_string()))?;
            session.read_concern = rc.level;
            session.snapshot_read_ts = rc.at_timestamp;
        }
        if let Some(wc) = options.write_concern {
            session.write_concern = wc;
        }

        let params = params.filter(|p| !p.is_empty());
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
            write_concern: self.write_concern,
            vector_consistency: self.vector_consistency,
            vector_build_wait: self.vector_build_wait,
            after_commit_generation: 0,
        };
        let params = params.filter(|p| !p.is_empty());
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
        let Ok(parsed) = CypherFrontend::new().parse(query) else {
            return false;
        };
        Self::spine_is_keyset_pageable(&parsed.plan.root)
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

    /// Inject the live session registry so `SHOW SESSIONS` / `SHOW TRANSACTIONS`
    /// can render operational introspection. The server wires this once at
    /// startup with the same registry its session binding updates; embedded
    /// callers leave it unset and those statements return no rows.
    pub fn set_operations_view(
        &mut self,
        operations: Arc<dyn coordinode_core::operations::OperationsView>,
    ) {
        self.operations = Some(operations);
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
        let (canonical, fp, mut plan, hinted, hinted_build_wait) = match self.plan_cache.get(query)
        {
            Some(cached) => (
                cached.canonical.clone(),
                cached.fingerprint,
                cached.plan.clone(),
                cached.vector_consistency_hinted,
                cached.vector_build_wait,
            ),
            None => {
                // Parse, validate, lower, and fingerprint through the query
                // frontend. The frontend is the only dialect-aware component;
                // everything below the returned plan is language-neutral.
                let parsed = CypherFrontend::new().parse(query)?;
                self.plan_cache.put(
                    query.to_string(),
                    Arc::new(CachedPlan {
                        canonical: parsed.canonical.clone(),
                        fingerprint: parsed.fingerprint,
                        plan: parsed.plan.clone(),
                        vector_consistency_hinted: parsed.vector_consistency_hinted,
                        vector_build_wait: parsed.vector_build_wait,
                    }),
                );
                (
                    parsed.canonical,
                    parsed.fingerprint,
                    parsed.plan,
                    parsed.vector_consistency_hinted,
                    parsed.vector_build_wait,
                )
            }
        };
        apply_session_vector_consistency(&mut plan, hinted, session.vector_consistency);
        self.run_plan(
            Statement {
                plan,
                canonical,
                fingerprint: fp,
                build_wait: hinted_build_wait,
            },
            source,
            params,
            session,
            txn_mode,
            scan_paging,
        )
    }

    /// Run a lowered plan as one statement: index selection, parameter
    /// binding, snapshot, execution, commit and the advisor record. Every
    /// statement goes through here, whatever produced its plan.
    fn run_plan(
        &self,
        statement: Statement,
        source: Option<&SourceContext>,
        params: Option<std::collections::HashMap<String, coordinode_core::graph::types::Value>>,
        session: &QuerySession,
        txn_mode: TxnMode,
        scan_paging: &mut Option<ScanPaging>,
    ) -> Result<
        (
            Vec<Row>,
            WriteStats,
            Option<coordinode_storage::engine::transaction::TransactionState>,
        ),
        DatabaseError,
    > {
        let Statement {
            plan,
            canonical,
            fingerprint: fp,
            build_wait: hinted_build_wait,
        } = statement;
        // The statistics are computed only when a push-down decision needs
        // them: every write invalidates them, so computing them for each
        // statement would put a storage read behind one shared lock on the
        // write path.
        let graph_stats = core::cell::OnceCell::new();
        let combined_stats = core::cell::OnceCell::new();
        let stats = || {
            combined_stats
                .get_or_init(|| {
                    graph_stats
                        .get_or_init(|| self.compute_stats())
                        .as_ref()
                        .map(|g| CombinedStats {
                            graph: g,
                            vector: &self.vector_index_registry,
                        })
                })
                .as_ref()
                .map(|c| c as &dyn coordinode_core::graph::stats::StorageStats)
        };
        let mut plan = self.plan_for_execution(plan, &stats);

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
        // GC-watermark pin for an explicit historical snapshot, held for the
        // statement so compaction cannot collect the history it reads.
        let mut retention_pin = None;
        // The timestamp a snapshot read concern names, which the executor
        // treats like `AS OF TIMESTAMP`.
        let mut named_read_ts: Option<i64> = None;
        let read_ts = match &txn_mode {
            // Interactive transaction: every statement reuses the pinned
            // start_ts so all reads resolve against the same snapshot
            // (repeatable read across the transaction).
            TxnMode::Interactive(state) => state.read_ts(),
            TxnMode::AutoCommit if session.read_concern == ReadConcernLevel::Snapshot => {
                // One-shot snapshot read; already captured into the
                // session (Database.snapshot_read_ts was taken when
                // the session was built).
                if let Some(ts) = session.snapshot_read_ts {
                    // `at_timestamp = T` is inclusive, like `AS OF TIMESTAMP
                    // T`: it sees every commit with commit_ts <= T, so a
                    // commit receipt's commit_ts pins its own write. A storage
                    // snapshot at S sees seqnos strictly below S, hence T + 1.
                    // Saturating by design: at u64::MAX there is nothing above
                    // to include, so the top snapshot is the right bound, not
                    // an overflow.
                    let seqno = ts.saturating_add(1);
                    // A named timestamp is preserved, not lowered, so the
                    // only way to make it complete is to let the commits
                    // under it land. Answering before they do would read a
                    // state missing a write the caller's own timestamp
                    // covers; answering at another timestamp would silently
                    // give them a different read than the one they asked for.
                    if let Err(blocking) = self
                        .engine
                        .pending_commits()
                        .await_complete_at(ts, READ_TIMEOUT)
                    {
                        return Err(DatabaseError::Other(format!(
                            "read at timestamp {ts} timed out waiting for the commit at \
                             {blocking} to land; it is covered by this timestamp and the \
                             read cannot be answered without it"
                        )));
                    }
                    // Refused when the seqno is already below the GC
                    // watermark: that history may be collected, and a read
                    // there would answer from whatever survived.
                    let Some(pin) = self.engine.pin_snapshot_at(seqno) else {
                        return Err(DatabaseError::OutsideRetention {
                            requested: ts,
                            // One below the first readable seqno, since a read
                            // at T is served from snapshot T + 1. Clamped at
                            // the bottom of the timestamp domain, not against
                            // overflow: horizon zero means nothing collected.
                            oldest_readable: self.engine.oldest_readable_seqno().saturating_sub(1),
                        });
                    };
                    retention_pin = Some(pin);
                    // A timestamp past i64::MAX is later than every commit, and
                    // so is i64::MAX itself: both name the read of everything
                    // committed, so the executor gets the largest value it holds.
                    named_read_ts = Some(i64::try_from(ts).unwrap_or(i64::MAX));
                    Timestamp::from_raw(seqno)
                } else {
                    self.fresh_pinned_read_ts(&mut retention_pin)
                }
            }
            TxnMode::AutoCommit => self.fresh_pinned_read_ts(&mut retention_pin),
        };
        let _retention_pin = retention_pin.take();
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
        // The statement's view of the field dictionary: every binding
        // applied before it started, refreshed here when one has landed
        // since (a registration, a replica apply, a replay, a snapshot).
        // Names the statement introduces are registered through the
        // pipeline and join this view only; no lock is held while it runs.
        let mut fields_view = self.fields.current()?;
        let vector_loader =
            StorageVectorLoader::new(Arc::clone(&self.engine), fields_view.clone(), self.shard_id);
        // The statement's valid-time NOW. A cursor page keeps the instant of
        // the cursor's first page, the time its snapshot was pinned at, so
        // pages read one timeline projection however long the client takes.
        let valid_now = match (scan_paging.as_ref(), session.snapshot_read_ts) {
            (Some(_), Some(pinned)) => i64::try_from(pinned).unwrap_or(i64::MAX),
            _ => coordinode_query::executor::runner::wall_clock_us(),
        };
        let mut ctx = ExecutionContext {
            engine: &self.engine,
            interner: &mut fields_view,
            field_registrar: Some(self.fields.as_ref()),
            id_allocator: &self.allocator,
            shard_id: self.shard_id,
            scan_paging: scan_paging.clone(),
            operations: self.operations.as_deref(),
            adaptive: self.adaptive_config.clone(),
            dedup_varlen_targets: false,
            snapshot_ts: named_read_ts,
            valid_now,
            temporal_instants: Vec::new(),
            snapshot_pin: None,
            warnings: Vec::new(),
            write_stats: WriteStats::default(),
            key_claims: Default::default(),
            text_index: None,
            text_index_registry: Some(&self.text_index_registry),
            vector_indexes: Some(coordinode_query::executor::runner::VectorIndexes {
                registry: &self.vector_index_registry,
                engine: &self.engine,
                // The query's own bound wins over the session's.
                build_wait: hinted_build_wait.unwrap_or(session.vector_build_wait),
            }),
            btree_index_registry: Some(self.index_registry.as_ref()),
            index_builds: Some(&self.index_builds),
            // Extension-op handlers for this Database (empty by default). An
            // enterprise layer / integration test populates it via
            // Database::register_extension so SHARDED-BY-style extension ops
            // dispatch; an empty registry means none are dispatchable.
            extensions: Some(&self.extension_registry),
            vector_loader: Some(&vector_loader),
            mvcc_oracle: Some(&self.oracle),
            mvcc_read_ts: read_ts,
            procedures: Some(&self.procedure_registry),
            advisor: Some(AdvisorContext {
                registry: Arc::clone(&self.query_registry),
                nplus1: Arc::clone(&self.nplus1_detector),
                dismissed: Arc::clone(&self.dismissed),
            }),
            txn,
            vector_consistency: plan.vector_consistency,
            vector_overfetch_factor: 1.2,
            vector_mvcc_stats: None,
            // The injected pipeline (Raft in cluster mode) — NOT a local
            // engine-applying one. Writing past it breaks replication.
            proposal_pipeline: Some(self.pipeline.as_ref()),
            proposal_id_gen: Some(&self.proposal_id_gen),
            read_concern: session.read_concern,
            write_concern: session.write_concern,
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
            label_schema_cache: std::collections::HashMap::new(),
            applied_watermark: None,
            read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode::default(
            ),
            read_timeout: READ_TIMEOUT,
            params: std::collections::HashMap::new(),
        };

        let start = Instant::now();
        // Auto-commit flushes the statement's writes; an interactive statement
        // leaves them buffered on the transaction for COMMIT to flush later.
        let results = if interactive {
            execute_no_commit(&plan, &mut ctx)?
        } else {
            execute(&plan, &mut ctx)?
        };
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

        let write_stats = ctx.write_stats.clone();
        let had_mutations = write_stats.has_mutations();
        // Hand the executor-updated keyset state (last_key + exhausted) back to
        // the caller through the in/out channel. `None` stays `None` for a
        // non-paged execution; the cursor path reads this to build the next
        // page's resume token.
        *scan_paging = ctx.scan_paging.clone();
        drop(ctx);

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

    /// Publish exactly the bindings `bytes` carries (a dictionary written by
    /// [`FieldInterner::to_bytes`]), for data restored already encoded with
    /// them. Must run before that data is installed.
    ///
    /// # Errors
    ///
    /// The bytes are not a dictionary, or a binding contradicts one this
    /// database already holds; nothing of the batch is published then.
    pub fn adopt_field_bindings(&self, bytes: &[u8]) -> Result<(), DatabaseError> {
        let bindings = FieldInterner::from_bytes(bytes).map_err(ExecutionError::from)?;
        self.fields
            .adopt(&bindings)
            .map_err(|e| DatabaseError::Execution(e.into()))
    }

    /// Turn a lowered plan into the one that runs: B-tree index selection,
    /// vector index annotation, the HNSW access path and graph-predicate
    /// push-down. Execution and every EXPLAIN go through here, so EXPLAIN
    /// shows the plan that runs.
    fn plan_for_execution<'s>(
        &self,
        mut plan: planner::logical::LogicalPlan,
        stats: &dyn Fn() -> Option<&'s dyn coordinode_core::graph::stats::StorageStats>,
    ) -> planner::logical::LogicalPlan {
        // Filter(NodeScan) becomes IndexScan when a matching B-tree index is
        // registered.
        plan.root = planner::optimize_index_selection(plan.root, &self.index_registry);
        // VectorTopK carries the resolved HNSW index name into execution.
        plan.root = planner::annotate_vector_top_k(
            plan.root,
            &self.vector_index_registry,
            plan.vector_consistency,
        );
        // Pure vector top-K reads through the index as its row source and
        // fetches only the k result nodes; filtered queries keep VectorTopK.
        plan.root = planner::apply_hnsw_scan_access_path(
            plan.root,
            &self.vector_index_registry,
            plan.vector_consistency,
        );
        // text_match over one label reads its matches from the text index
        // instead of scanning the label to keep them.
        plan.root =
            planner::apply_text_index_scan_access_path(plan.root, &self.text_index_registry);
        // count over one bare label reads the label's node counter instead
        // of scanning the label, where the catalog lets the counter stand
        // for the scan. A catalog that cannot be read keeps the scan.
        plan.root = planner::apply_node_count_from_counter(plan.root, &|label| {
            coordinode_query::executor::runner::node_counter_answers_scan(&self.engine, label)
                .unwrap_or(false)
        });
        // Every VectorFilter after a Traverse gets its strategy (graph_first /
        // acorn_filtered / vector_first) from the push-down cost model.
        plan.root = planner::optimize_push_down_lazy(plan.root, stats);
        plan
    }

    /// The plan a Cypher query runs as, with the session's vector
    /// consistency applied, for EXPLAIN. `stats` are this database's
    /// statistics from [`Self::compute_stats`], computed once by the caller.
    pub fn explain_plan(
        &self,
        query: &str,
        stats: Option<&coordinode_storage::engine::stats::StorageStatsComputer>,
    ) -> Result<planner::logical::LogicalPlan, DatabaseError> {
        let parsed = CypherFrontend::new().parse(query)?;
        let mut plan = parsed.plan;
        apply_session_vector_consistency(
            &mut plan,
            parsed.vector_consistency_hinted,
            self.vector_consistency,
        );
        let combined = stats.map(|g| CombinedStats {
            graph: g,
            vector: &self.vector_index_registry,
        });
        let stats = || {
            combined
                .as_ref()
                .map(|c| c as &dyn coordinode_core::graph::stats::StorageStats)
        };
        Ok(self.plan_for_execution(plan, &stats))
    }

    /// Return EXPLAIN plan text for a Cypher query.
    ///
    /// Uses real storage statistics (node counts, fan-out) for
    /// more accurate cost estimates than hardcoded defaults.
    pub fn explain_cypher(&self, query: &str) -> Result<String, DatabaseError> {
        let stats = self.compute_stats();
        let plan = self.explain_plan(query, stats.as_ref())?;
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
            let shown = def.to_string();
            if !explain.contains(&shown) {
                continue;
            }
            if let Some(state) = self
                .vector_index_registry
                .health_snapshot(&def.label, def.property())
            {
                health_lines.push(format!("  {shown}: {}", describe_index_health(&state)));
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
        let stats = self.compute_stats();
        let plan = self.explain_plan(query, stats.as_ref())?;
        Ok(self.suggest_for(&plan, stats.as_ref()))
    }

    /// EXPLAIN SUGGEST for a plan from [`Self::explain_plan`]: its text and
    /// the suggestions for it, against `stats` and this database's indexes.
    pub fn suggest_for(
        &self,
        plan: &planner::logical::LogicalPlan,
        stats: Option<&coordinode_storage::engine::stats::StorageStatsComputer>,
    ) -> coordinode_query::advisor::ExplainSuggestResult {
        let stats_ref = stats.map(|s| s as &dyn coordinode_core::graph::stats::StorageStats);
        plan.explain_suggest_with_stats(stats_ref, Some(&self.index_registry))
    }

    /// The stored nodes per label, every label that has any, as of the
    /// latest commit. Read from counters the write path keeps on the same
    /// transaction as the nodes, so it costs a few reads however many nodes
    /// there are; `MATCH (n:L) RETURN count(n)` scans them. A temporal node
    /// counts once per stored version.
    ///
    /// # Errors
    ///
    /// A counter that does not decode, or a failed read.
    pub fn label_counts(&self) -> Result<std::collections::HashMap<String, u64>, DatabaseError> {
        Ok(coordinode_storage::engine::stats::label_counts(
            &self.engine,
        )?)
    }

    /// Compute storage statistics for the cost estimator (with TTL cache).
    ///
    /// Returns a cached snapshot if it is younger than `stats_ttl`.
    /// Otherwise recomputes from MVCC storage and caches the result.
    ///
    /// Returns `None` when the statistics cannot be read, and the planner then
    /// uses its defaults: they steer the choice of plan, never its result. The
    /// failure itself is logged as an error, because the usual cause is a
    /// damaged counter or adjacency list, which is worth an operator's look.
    pub fn compute_stats(&self) -> Option<coordinode_storage::engine::stats::StorageStatsComputer> {
        // Read before computing: a write that lands during the computation
        // moves the generation past this one, so its result is not reused.
        let generation = self.stats_generation.load(Ordering::Acquire);
        let mut guard = self.cached_stats.lock().ok()?;
        if let Some((ref stats, computed_at, computed_for)) = *guard {
            if computed_for == generation && computed_at.elapsed() < self.stats_ttl {
                return stats.clone();
            }
        }
        // Cache miss or expired — recompute.
        #[cfg(test)]
        self.stats_computations.fetch_add(1, Ordering::Relaxed);
        let fresh =
            match coordinode_storage::engine::stats::StorageStatsComputer::compute(&self.engine) {
                Ok(fresh) => Some(fresh),
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "planner statistics unavailable, planning with defaults"
                    );
                    None
                }
            };
        *guard = Some((fresh.clone(), Instant::now(), generation));
        fresh
    }

    /// Invalidate the cached storage statistics.
    ///
    /// The next `explain_cypher()` or `explain_suggest()` call will
    /// trigger a fresh scan.  Useful after bulk imports or schema changes.
    pub fn invalidate_stats_cache(&self) {
        self.stats_generation.fetch_add(1, Ordering::Release);
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

    /// Commit the catalog change `stage` makes in a transaction of its own,
    /// through the write pipeline, so the conditions it states (record
    /// versions, names that must be free) are decided at its commit.
    fn commit_catalog<E>(
        &self,
        stage: impl FnOnce(
            &mut coordinode_storage::engine::transaction::Transaction<'_>,
        ) -> Result<(), E>,
    ) -> Result<(), DatabaseError>
    where
        DatabaseError: From<E>,
    {
        let mut txn = self.begin_catalog_txn();
        stage(&mut txn)?;
        self.commit_catalog_txn(txn)
    }

    /// A transaction for one catalog change.
    fn begin_catalog_txn(&self) -> coordinode_storage::engine::transaction::Transaction<'_> {
        coordinode_storage::engine::transaction::Transaction::begin(
            &self.engine,
            Some(&self.oracle),
            self.oracle.next(),
        )
    }

    /// Commit catalog change `txn` through the write pipeline.
    fn commit_catalog_txn(
        &self,
        mut txn: coordinode_storage::engine::transaction::Transaction<'_>,
    ) -> Result<(), DatabaseError> {
        let wc = self.write_concern;
        let commit_ctx = coordinode_storage::engine::transaction::CommitContext {
            write_concern: &wc,
            pipeline: Some(self.pipeline.as_ref()),
            id_gen: Some(&self.proposal_id_gen),
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        txn.note_schema_change();
        txn.commit(&commit_ctx)
            .map_err(coordinode_query::executor::runner::catalog_commit_error)?;
        Ok(())
    }

    /// Create a vector (HNSW) index on a label's vector property.
    ///
    /// After creation, queries using `vector_similarity(n.prop, $q)` will
    /// use the HNSW index instead of brute-force distance computation.
    /// Call `populate_vector_index` to backfill existing vectors.
    ///
    /// # Errors
    ///
    /// The label or property name could not be registered, or the
    /// definition could not be published; the index is not created then.
    pub fn create_vector_index(
        &mut self,
        name: impl Into<String>,
        label: impl Into<String>,
        property: impl Into<String>,
        config: coordinode_query::index::VectorIndexConfig,
    ) -> Result<(), DatabaseError> {
        use coordinode_modality::{IndexStore as _, LocalIndexStore};
        let descriptor =
            coordinode_query::index::IndexDescriptor::hnsw(name, label, property, config);

        // The names the index is keyed by are bound before its definition is
        // published, so every member that sees the definition resolves them.
        let ids = self
            .fields
            .register(&[&descriptor.label, descriptor.property()])
            .map_err(ExecutionError::from)?;
        // Published in a catalog commit of its own, which gives the index
        // its identities and binds its name.
        let store = LocalIndexStore::new(&self.engine);
        let mut published = None;
        self.commit_catalog(|txn| {
            published = Some(store.publish_definition_txn(txn, descriptor)?);
            Ok::<(), coordinode_modality::StoreError>(())
        })?;
        let def = published.ok_or_else(|| {
            DatabaseError::Other("the vector index publication staged no definition".into())
        })?;

        // Register in both registries: VectorIndexRegistry holds the live HNSW
        // graph for query acceleration; IndexRegistry mirrors the definition so
        // advisors and planners can see all indexes (scalar + vector) through
        // a single source of truth.
        let tier = self.vector_index_registry.tier_handle(ids[0], ids[1]);
        self.vector_index_registry
            .register_with_tier(def.clone(), tier);
        self.index_registry.register_in_memory(def);
        Ok(())
    }

    /// Rebuild the B-tree index `def` from the nodes already stored, into a
    /// new generation of the same index, on the engine's build executor.
    ///
    /// The definition is published as building in the new generation, with
    /// the entries of the generation it served from removed and the build
    /// admitted, in one catalog commit conditioned on the record it
    /// replaces; writers maintain the new generation from then on. A build
    /// the stored data refuses keeps the index, marked failed, so its
    /// constraint still holds for new writes while lookups stop using it.
    /// Returns the number of nodes indexed.
    fn build_btree_index(
        &self,
        mut def: coordinode_query::index::IndexDefinition,
    ) -> Result<u64, DatabaseError> {
        use coordinode_modality::{ENTRY_LAYOUT, IndexStore as _, LocalIndexStore};
        use coordinode_query::index::{BuildFailure, IndexBuildRecord, IndexState};
        let store = LocalIndexStore::new(&self.engine);
        def.layout = ENTRY_LAYOUT;
        def.state = IndexState::Building {
            written: 0,
            estimated_total: 0,
        };
        let replaced = store.definition_version(def.id)?;
        let retired = def.generation;
        self.commit_catalog(|txn| {
            store.expect_definition_txn(txn, def.id, replaced)?;
            def.generation = store.allocate_generation_txn(txn)?;
            store.clear_txn(txn, retired)?;
            store.put_definition_txn(txn, &def)?;
            store.put_build_txn(
                txn,
                &IndexBuildRecord::accepted(def.id, def.generation, BuildFailure::Keep),
                None,
            )
        })?;
        self.index_registry
            .register_published(&self.engine, def.clone())?;
        self.index_builds
            .submit(def.generation)
            .map_err(DatabaseError::Other)?;
        self.build_outcome(&def)
    }

    /// Wait for the build of `def` and turn its outcome into what the caller
    /// is told: the nodes indexed, or why the index was not published.
    fn build_outcome(
        &self,
        def: &coordinode_query::index::IndexDefinition,
    ) -> Result<u64, DatabaseError> {
        use coordinode_query::index::{BuildError, IndexBuildOutcome};
        match self.index_builds.wait(def.generation, None)? {
            Some(IndexBuildOutcome::Published { indexed }) => Ok(indexed.unwrap_or(0)),
            Some(IndexBuildOutcome::Failed(BuildError::Duplicate(v))) => {
                Err(DatabaseError::Execution(v.into()))
            }
            Some(IndexBuildOutcome::Failed(BuildError::Other(reason))) => {
                Err(DatabaseError::Other(reason))
            }
            Some(IndexBuildOutcome::Cancelled) => Err(DatabaseError::Other(format!(
                "the build of index '{def}' was cancelled"
            ))),
            None => Err(DatabaseError::Other(format!(
                "the build of index '{def}' has no outcome yet"
            ))),
        }
    }

    /// Take up every index build an earlier process left without an
    /// outcome, on this process's executor, and wait for them: a build of a
    /// new index whose statement never returned is filled and published, its
    /// owning constraint active with it, or, refused by the stored data,
    /// withdrawn with that constraint, which the interrupted statement never
    /// acknowledged; a rebuild refused by the data keeps its index failed.
    /// Returns how many were published.
    ///
    /// # Errors
    ///
    /// The build records could not be read, or an executor could not start.
    pub fn resume_interrupted_index_builds(&self) -> Result<usize, DatabaseError> {
        use coordinode_query::index::IndexBuildOutcome;
        let resumed = self.index_builds.resume().map_err(DatabaseError::Other)?;
        let mut published = 0;
        for generation in resumed {
            match self.index_builds.wait(generation, None)? {
                Some(IndexBuildOutcome::Published { .. }) => {
                    published += 1;
                    tracing::info!(
                        generation = generation.as_raw(),
                        "finished an interrupted index build"
                    );
                }
                outcome => tracing::error!(
                    generation = generation.as_raw(),
                    ?outcome,
                    "an interrupted index build ended without its index"
                ),
            }
        }
        self.refresh_btree_indexes()?;
        Ok(published)
    }

    /// Reload the index definitions from the schema partition, so a member
    /// that applied another member's CREATE or DROP INDEX maintains and uses
    /// the same indexes. Cluster deployments call this whenever the applied
    /// index advances.
    pub fn refresh_btree_indexes(&self) -> Result<(), DatabaseError> {
        self.index_registry
            .load_all(&self.engine)
            .map_err(DatabaseError::Storage)
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
        use coordinode_modality::{IndexStore as _, LocalIndexStore};
        let label = label.into();
        let property = property.into();
        let descriptor = coordinode_query::index::IndexDescriptor::text(
            name,
            &label,
            vec![property.clone()],
            config,
        );

        // Publish the index definition before the local index exists, in a
        // catalog commit that gives it its identities and binds its name.
        let store = LocalIndexStore::new(&self.engine);
        let mut published = None;
        self.commit_catalog(|txn| {
            published = Some(store.publish_definition_txn(txn, descriptor)?);
            Ok::<(), coordinode_modality::StoreError>(())
        })?;
        let def = published.ok_or_else(|| {
            DatabaseError::Other("the text index publication staged no definition".into())
        })?;

        // Register in the text index registry (creates empty tantivy index).
        let generation = def.generation;
        self.text_index_registry
            .register(def.clone())
            .map_err(DatabaseError::Other)?;

        // The backfill is this member's build of the index, run by the
        // engine's executor; waiting for it cancels nothing.
        self.index_builds
            .run_local(
                generation,
                coordinode_query::index::lifecycle::text_build(label, vec![property]),
            )
            .map_err(DatabaseError::Other)?;
        let count = self.build_outcome(&def)?;
        if count > 0 {
            tracing::info!("backfilled text index with {count} document(s)");
        }
        Ok(())
    }

    /// Get a reference to the text index registry.
    pub fn text_index_registry(&self) -> &coordinode_query::index::TextIndexRegistry {
        &self.text_index_registry
    }

    /// Search the text index of `(label, property)` as the store stands now:
    /// the index together with the committed writes it has not folded yet,
    /// read from the store, and temporal nodes at their state valid now.
    /// `Ok(None)` when there is no such index.
    ///
    /// # Errors
    ///
    /// The field dictionary or the written nodes could not be read, or the
    /// query not run.
    pub fn text_search(
        &self,
        label: &str,
        property: &str,
        request: coordinode_search::tantivy::multi_lang::TextRequest<'_>,
        matches: coordinode_search::tantivy::pending::Matches,
    ) -> Result<Option<Vec<coordinode_search::tantivy::HighlightedResult>>, DatabaseError> {
        let interner = self.fields.current()?;
        // The latest applied state, as the text worker reads it.
        let read = coordinode_storage::engine::transaction::Transaction::new(
            &self.engine,
            None,
            Timestamp::ZERO,
            None,
        );
        self.text_index_registry
            .find(
                label,
                property,
                &read,
                self.shard_id,
                &interner,
                coordinode_query::index::IndexDelta::Nodes(Default::default()),
                coordinode_query::executor::runner::wall_clock_us(),
                request,
                matches,
            )
            .map_err(DatabaseError::Other)
    }

    /// The verified field dictionary as it stands: every binding applied so
    /// far. The view is immutable and cheap to clone; holding it blocks
    /// nothing.
    ///
    /// # Errors
    ///
    /// The stored dictionary is inconsistent.
    pub fn interner(&self) -> Result<FieldInterner, DatabaseError> {
        Ok(self.fields.current()?)
    }

    /// The authority new property names are registered with, for a writer
    /// outside the query path (a restore, an import) that encodes data
    /// itself.
    pub fn field_registrar(&self) -> Arc<dyn FieldRegistrar> {
        Arc::clone(&self.fields) as Arc<dyn FieldRegistrar>
    }

    /// Restore a logical backup or import into this database, keeping every
    /// node identifier of the input.
    ///
    /// The whole input is checked before anything is written: an identifier
    /// this database already issued refuses it, since writing it would
    /// replace a live node or hand the identifier out twice. The identifier
    /// lease is then raised above the input's identifiers through the log, so
    /// no node created later takes one. A crash midway leaves a record of the
    /// load, and restoring the same input again finishes it.
    ///
    /// # Errors
    ///
    /// [`RestoreError::IdentifiersIssued`](crate::backup::restore::RestoreError::IdentifiersIssued)
    /// when an identifier is already issued here;
    /// [`RestoreError::UnfinishedLoad`](crate::backup::restore::RestoreError::UnfinishedLoad)
    /// when an interrupted restore of a different input holds part of this
    /// database; [`RestoreError::Unsupported`](crate::backup::restore::RestoreError::Unsupported)
    /// for a snapshot; otherwise a malformed or
    /// incompatible input, or a storage or lease failure.
    pub fn restore(
        &self,
        format: crate::backup::BackupFormat,
        source: &dyn crate::backup::restore::RestoreSource,
        options: &crate::backup::restore::RestoreOptions<'_>,
    ) -> Result<crate::backup::restore::RestoreStats, crate::backup::restore::RestoreError> {
        let leases = id_lease::LogLeaseReserver::new(
            Arc::clone(&self.engine),
            Arc::clone(&self.pipeline),
            Arc::clone(&self.proposal_id_gen),
            Arc::clone(&self.oracle),
        );
        let raise = |base: u64, target: u64, token| {
            leases
                .raise_from(base, target, token)
                .map_err(|e| e.to_string())
        };
        let build_indexes = || self.build_indexes_over_stored_nodes();
        let check_constraints = || self.check_constraints_over_stored_nodes();
        let target = crate::backup::restore::RestoreTarget {
            engine: &self.engine,
            fields: self.fields.as_ref(),
            raise_lease: &raise,
            build_indexes: &build_indexes,
            check_constraints: &check_constraints,
        };
        crate::backup::restore::run(&target, format, source, options)
    }

    /// Build every declared index again from the nodes in the store: the
    /// records a restore writes reach no index on their way in. Index
    /// definitions the load itself brought are taken up first.
    fn build_indexes_over_stored_nodes(&self) -> Result<(), String> {
        use coordinode_query::index::IndexType;

        let defs = coordinode_query::index::ops::list_index_definitions(&self.engine)
            .map_err(|e| format!("read the index definitions: {e}"))?;
        self.refresh_btree_indexes().map_err(|e| e.to_string())?;
        for def in defs.iter().filter(|d| d.index_type == IndexType::BTree) {
            self.build_btree_index(def.clone())
                .map_err(|e| format!("build index '{def}': {e}"))?;
        }

        let fields = self.fields.current().map_err(|e| e.to_string())?;
        let hnsw: Vec<_> = defs
            .iter()
            .filter(|d| d.index_type == IndexType::Hnsw && d.vector_config.is_some())
            .cloned()
            .collect();
        if !hnsw.is_empty() {
            for def in &hnsw {
                self.vector_index_registry
                    .unregister(&def.label, def.property());
            }
            Self::register_and_populate_hnsw(
                &self.vector_index_registry,
                &fields,
                &self.engine,
                self.shard_id,
                &hnsw,
                PopulateMode::Blocking,
            );
        }

        let text: Vec<_> = defs
            .iter()
            .filter(|d| d.index_type == IndexType::Text && d.text_config.is_some())
            .cloned()
            .collect();
        for def in &text {
            self.text_index_registry
                .unregister(&def.label, def.property());
        }
        Self::populate_text_indexes(
            &self.text_index_registry,
            &self.engine,
            &fields,
            self.shard_id,
            &text,
        );
        Ok(())
    }

    /// Check every stored node against the presence and type constraints of
    /// its label, and record the name of every constraint a schema holds:
    /// the records and schemas a restore writes reach no check on their way
    /// in. Uniqueness is checked by the index build that runs first.
    fn check_constraints_over_stored_nodes(&self) -> Result<(), String> {
        use coordinode_core::schema::definition::{NodeConstraint, encode_constraint_name_key};
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        use coordinode_storage::engine::partition::Partition;

        let labels = LocalSchemaStore::new(&self.engine)
            .list_labels()
            .map_err(|e| format!("read the label schemas: {e}"))?;
        for schema in &labels {
            for constraint in schema.constraints() {
                let key = encode_constraint_name_key(&constraint.name);
                match self
                    .engine
                    .get(Partition::Schema, &key)
                    .map_err(|e| e.to_string())?
                {
                    Some(holder) if holder.as_ref() == schema.name.as_bytes() => {}
                    Some(holder) => {
                        return Err(format!(
                            "constraint '{}' of :{} is also held by :{}",
                            constraint.name,
                            schema.name,
                            String::from_utf8_lossy(&holder)
                        ));
                    }
                    None => self
                        .engine
                        .put(Partition::Schema, &key, schema.name.as_bytes())
                        .map_err(|e| e.to_string())?,
                }
            }
            if !schema
                .constraints()
                .iter()
                .any(NodeConstraint::checks_each_node)
            {
                continue;
            }
            let staged = std::collections::HashMap::new();
            if let Some(violation) =
                coordinode_storage::engine::claims::evaluate::first_label_schema_violation(
                    &self.engine,
                    schema,
                    &staged,
                )
                .map_err(|e| e.to_string())?
            {
                return Err(format!(
                    "node {} of :{} ({})",
                    violation.node.to_element_id(),
                    schema.name,
                    violation.reason
                ));
            }
        }
        Ok(())
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

    /// The session setting `query` changes, when it is a session SET command.
    ///
    /// Supports `SET vector_consistency = 'mode'` and
    /// `SET vector_build_wait = '5s'`. `None` for anything else, including a
    /// value the setting cannot take: the text then goes to the Cypher parser,
    /// which refuses it. Run on every statement, so a statement that is not a
    /// SET costs a prefix comparison and no allocation.
    pub fn parse_session_set(query: &str) -> Option<SessionSetting> {
        let trimmed = query.trim();
        if !trimmed
            .get(..4)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("set "))
        {
            return None;
        }

        let rest = trimmed[4..].trim();
        let (name, after_eq) = rest.split_once('=')?;
        let value = after_eq.trim();

        // Strip quotes (single or double)
        let unquoted = if value.len() >= 2
            && ((value.starts_with('\'') && value.ends_with('\''))
                || (value.starts_with('"') && value.ends_with('"')))
        {
            &value[1..value.len() - 1]
        } else {
            value
        };

        let name = name.trim();
        if name.eq_ignore_ascii_case("vector_consistency") {
            VectorConsistencyMode::from_str_opt(unquoted).map(SessionSetting::VectorConsistency)
        } else if name.eq_ignore_ascii_case("vector_build_wait") {
            coordinode_query::cypher::parse_wait(unquoted).map(SessionSetting::VectorBuildWait)
        } else {
            None
        }
    }
}

mod after_commit;
mod catalog;
mod fields;
mod id_lease;
mod index_builds;
pub use after_commit::{AfterCommitDispatchReport, TriggerDispatchConfig};
pub use catalog::{ConstraintDeclaration, LabelConstraint};
pub use fields::FieldDictionary;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
