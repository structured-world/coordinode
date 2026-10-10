//! Physical query executor: runs logical plan operators against storage.
//!
//! Each operator produces a `Vec<Row>` from its input.
//! Future optimization: streaming iterator model.

use core::time::Duration;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use coordinode_core::graph::edge::{AdjDirection, AdjKeyParts, PostingList, decode_edge_props};
use coordinode_core::graph::intern::{DictionaryError, FieldInterner};
use coordinode_core::graph::node::NodeIdAllocator;
use coordinode_core::graph::node::{NodeId, NodeRecord, decode_temporal_node_key};
use coordinode_core::graph::types::{Value, VectorConsistencyMode, VectorMvccStats};
use coordinode_core::schema::definition::{
    EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType, SchemaMode,
};
use coordinode_core::schema::validation::validate_one;
use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::core::StorageEngine;
// Storage-partition guard: the only legitimate Partition users in this
// crate are the partition-parameterised transaction primitives below (which
// take it by argument) and test fixtures. Production execution goes through the
// typed Layer-4 stores. See the crate-level `#![deny(clippy::disallowed_types)]`.
use coordinode_storage::engine::StorageSnapshot;
#[allow(clippy::disallowed_types)]
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use super::eval::{EvalError, eval_binary_op, eval_unary_op, is_truthy};
use super::eval_neutral::eval_neutral;
use super::row::Row;
use crate::index::{IndexDelta, IndexState, OnlineDuringBuild};
use crate::plan::ViolationMode;
use crate::plan::{Direction, LengthBound, Pattern, PatternElement};
use crate::planner::logical::*;

/// Default maximum hops for unbounded variable-length paths.
/// Prevents exponential fan-out on `*` or `*1..` patterns.
const DEFAULT_MAX_HOPS: u64 = 10;

/// Key-value pair returned by MVCC prefix scan: (user_key, value).
type KvPair = (Vec<u8>, Vec<u8>);

/// The kind of a named catalog object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogObject {
    Label,
    EdgeType,
    Constraint,
    Index,
}

impl std::fmt::Display for CatalogObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Label => "label",
            Self::EdgeType => "edge type",
            Self::Constraint => "constraint",
            Self::Index => "index",
        })
    }
}

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

    /// Arithmetic with no answer: division or modulo by an integer zero, or an
    /// integer operation whose exact result leaves the `i64` range. Carries the
    /// message verbatim, so a driver sees the same text it would elsewhere.
    #[error("{0}")]
    Arithmetic(#[from] super::eval::EvalError),

    #[error("serialization error: {0}")]
    Serialization(String),

    /// No NodeId could be handed out for a created node: the log granted no
    /// lease (this member does not lead, or the grant failed) or the shard's
    /// identifier space is used up. Nothing was written.
    #[error("{0}")]
    NodeIdLease(#[from] coordinode_core::graph::node::IdLeaseError),

    /// A property name could not be given an id, or stored data carries an
    /// id the dictionary cannot name. Nothing encoded with the missing
    /// binding was written or served.
    #[error("{0}")]
    FieldDictionary(#[from] coordinode_core::graph::intern::DictionaryError),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// A `CALL` refused before the procedure ran: an unknown name, arguments
    /// that do not match the signature, or a YIELD of a column it lacks.
    #[error("{0}")]
    Procedure(#[from] crate::procedure::ProcedureError),

    #[error("write conflict: {0}")]
    Conflict(String),

    /// A condition the statement's result depends on no longer holds, or
    /// another statement in flight holds an incompatible one. Nothing was
    /// applied; re-running the statement re-reads the state it depends on.
    ///
    /// Kept apart from `Conflict` for the same reason the engine keeps them
    /// apart: two statements can write disjoint keys and still break a graph
    /// condition together, and calling that a write conflict names the wrong
    /// cause to whoever reads the message.
    #[error("invariant refused the write: {0}")]
    InvariantRefused(String),

    /// A write conditioned on a record's version found a different one.
    /// Nothing was applied. The version that is there now travels with the
    /// error so the caller can decide without reading the record again.
    #[error(
        "record version mismatch: expected {expected:?}, found {current:?}; \
         nothing was applied"
    )]
    RevisionMismatch {
        /// The version the write was conditioned on; `None` required absence.
        expected: Option<u64>,
        /// The version the record has now; `None` means it is absent.
        current: Option<u64>,
    },

    /// The engine rejected the commit's writes because storage is over its
    /// compaction-debt stop threshold. Nothing was applied; retryable after
    /// a short delay.
    #[error(
        "write rejected: storage is over its compaction-debt stop threshold; \
         retry after compaction catches up"
    )]
    Backpressure,

    /// The write reached a node that is not the leader, so it could not be
    /// replicated. Nothing was applied; the same write succeeds at the leader.
    #[error("not the leader{}", match leader_id {
        Some(id) => format!("; leader is node {id}"),
        None => String::from("; no leader known yet"),
    })]
    NotLeader {
        /// The node the cluster last named leader, if any. `None` while an
        /// election is in flight.
        leader_id: Option<u64>,
    },

    /// The write reached a member that does not run its group's version.
    /// Nothing was applied; the same write succeeds at the leader named in
    /// the refusal, when one is known.
    #[error("this member is read-only: {0}")]
    Mismatched(coordinode_core::version::Mismatch),

    /// An `AS OF TIMESTAMP` older than the MVCC retention horizon. History
    /// that old may already be collected, so the read is refused instead of
    /// answering from whatever survived. The horizon moves with the clock
    /// and `retention_window_secs`; the same query succeeds at any timestamp
    /// from `oldest_readable` on.
    #[error(
        "AS OF TIMESTAMP {requested} is older than the MVCC retention horizon: \
         history is readable from timestamp {oldest_readable} on \
         (retention_window_secs)"
    )]
    OutsideRetention {
        /// The timestamp the query asked for.
        requested: i64,
        /// The oldest timestamp still readable when the query ran.
        oldest_readable: u64,
    },

    /// Schema mode violation: STRICT label rejected an undeclared property, or
    /// a write attempted to SET a COMPUTED (read-only) property.
    #[error("schema violation: {0}")]
    SchemaViolation(String),

    /// An insert named a key that a row of the table already holds. Nothing
    /// was written; the existing row is unchanged.
    #[error(
        "table `{table}` already has a row with key {key} (row {element_id}): key already exists"
    )]
    DuplicateKey {
        /// The table.
        table: String,
        /// The key, rendered.
        key: String,
        /// The elementId of the row that holds the key.
        element_id: String,
    },

    /// A write would give a value of a unique index a second holder. Nothing
    /// was written.
    #[error(
        "unique constraint violated on index `{index}`: property `{property}` already has \
         value {value} (node {element_id})"
    )]
    UniqueViolation {
        /// The unique index.
        index: String,
        /// The indexed properties, comma-separated.
        property: String,
        /// The value, rendered.
        value: String,
        /// The elementId of the node that holds the value.
        element_id: String,
    },

    /// A write takes a value of a unique index whose build has not reached
    /// every stored node yet, and proving no stored node holds the value
    /// would read more than the configured limit. Nothing was written; the
    /// value may well be free, and the same write succeeds once the build
    /// is done.
    #[error(
        "unique index `{index}` is still being built and proving the value free would read \
         more than {limit} stored nodes; retry once the build is done"
    )]
    UniquenessUnresolved {
        /// The unique index being built.
        index: String,
        /// The most stored node rows one proof reads.
        limit: u64,
    },

    /// A write would leave a node breaking a constraint of its label: a
    /// required property missing or null, or a value of the wrong type.
    /// Nothing was written.
    #[error(
        "constraint `{constraint}` ({kind}) violated on :{label}: property `{property}` of node \
         {element_id} {}",
        constraint_problem(kind)
    )]
    ConstraintViolation {
        /// The constraint.
        constraint: String,
        /// What it requires: presence (`NOT NULL`, `NODE KEY`) or a type.
        /// Boxed: a property type is large, and every result carries this
        /// error's size.
        kind: Box<coordinode_core::schema::definition::ConstraintKind>,
        /// The constrained label.
        label: String,
        /// The property that breaks it.
        property: String,
        /// The elementId of the node.
        element_id: String,
    },

    /// A statement tried to change the key of an existing row. A row's key is
    /// its identity and does not change; delete the row and insert a new one.
    #[error(
        "column `{column}` is part of the key of table `{table}` and cannot be changed; \
         delete the row and insert a new one"
    )]
    KeyImmutable {
        /// The table.
        table: String,
        /// The key column the statement targeted.
        column: String,
    },

    /// A catalog object of the name already exists. Nothing changed.
    #[error("{object} '{name}' already exists")]
    CatalogObjectExists {
        /// What kind of object holds the name.
        object: CatalogObject,
        /// The name.
        name: String,
    },

    /// No catalog object of the name exists. Nothing changed.
    #[error("{object} '{name}' not found")]
    CatalogObjectMissing {
        /// What kind of object was looked for.
        object: CatalogObject,
        /// The name.
        name: String,
    },

    /// A catalog change the current catalog refuses: a definition its
    /// constraints cannot hold under, an object still being validated, a
    /// dependency that forbids it. Nothing changed.
    #[error("catalog change refused: {0}")]
    CatalogRefused(String),

    /// L1 cycle protection trip: cumulative trigger cascade depth
    /// for the current originating mutation exceeded its limit. `chain` lists
    /// the trigger names that fired, in firing order, to help diagnose the
    /// runaway cascade.
    #[error("trigger cascade depth exceeded: current={current}, limit={limit}, chain={chain:?}")]
    CascadeOverflow {
        current: u32,
        limit: u32,
        chain: Vec<String>,
    },

    /// L2 cycle protection trip: a single trigger fired more times
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

/// Unique keys a statement claimed, kept until it commits.
#[derive(Debug, Default)]
pub struct KeyClaims {
    /// Table keys, as (table, key, row).
    pub tables: Vec<(String, Vec<Value>, NodeId)>,
    /// Values of unique indexes.
    pub indexes: Vec<crate::index::UniqueClaim>,
}

impl KeyClaims {
    /// Whether the statement claimed nothing.
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty() && self.indexes.is_empty()
    }
}

impl From<crate::index::UniqueViolation> for ExecutionError {
    fn from(v: crate::index::UniqueViolation) -> Self {
        unique_violation(v)
    }
}

/// The refusal of a write that leaves `node` breaking a constraint of
/// `label`. Any other validation error is a schema violation.
fn constraint_violation(
    e: coordinode_core::schema::validation::ValidationError,
    label: &str,
    node: NodeId,
) -> ExecutionError {
    use coordinode_core::schema::validation::ValidationError;
    match e {
        ValidationError::ConstraintViolation {
            constraint,
            kind,
            property,
        } => ExecutionError::ConstraintViolation {
            constraint,
            kind: Box::new(kind),
            label: label.to_string(),
            property,
            element_id: node.to_element_id(),
        },
        other => ExecutionError::SchemaViolation(other.to_string()),
    }
}

/// Check every node `txn` recorded for a post-state check against the
/// presence and type constraints of its primary label, as the transaction
/// leaves it: its buffered record with its pending document deltas applied.
/// Called once, right before the commit, so a node built up over several
/// writes or statements is judged as it lands, never mid-way. Reading each
/// label's schema binds the commit to the revision checked against.
///
/// # Errors
///
/// The first node that breaks a constraint, or a read that failed.
pub fn check_post_state(
    txn: &mut coordinode_storage::engine::transaction::Transaction<'_>,
    engine: &StorageEngine,
    interner: &FieldInterner,
) -> Result<(), ExecutionError> {
    use coordinode_core::graph::node::{decode_node_key, decode_temporal_node_key};
    use coordinode_modality::{LocalNodeStore, LocalSchemaStore, SchemaStore as _};
    let mut keys = txn.take_post_state_checks();
    if keys.is_empty() {
        return Ok(());
    }
    keys.sort_unstable();
    keys.dedup();
    let store = LocalSchemaStore::new(engine);
    let mut schemas: HashMap<String, Option<LabelSchema>> = HashMap::new();
    for key in keys {
        let Some(record) = LocalNodeStore::post_state(txn, &key)? else {
            continue;
        };
        let label = record.primary_label();
        if label.is_empty() {
            continue;
        }
        if !schemas.contains_key(label) {
            let schema = store.load_label_txn(txn, label)?;
            schemas.insert(label.to_string(), schema);
        }
        let Some(Some(schema)) = schemas.get(label) else {
            continue;
        };
        let lookup = crate::index::registry::record_lookup(&record, interner);
        if let Err(e) = coordinode_core::schema::validation::check_node_constraints(schema, &lookup)
        {
            let node = decode_node_key(&key)
                .map(|(_, node)| node)
                .or_else(|| decode_temporal_node_key(&key).map(|(_, node, _)| node))
                .ok_or_else(|| {
                    ExecutionError::Serialization("a recorded node key does not decode".into())
                })?;
            return Err(constraint_violation(e, label, node));
        }
    }
    Ok(())
}

/// What is wrong with a property that breaks a constraint of `kind`.
fn constraint_problem(kind: &coordinode_core::schema::definition::ConstraintKind) -> &'static str {
    match kind {
        coordinode_core::schema::definition::ConstraintKind::Type(_) => {
            "has a value of another type"
        }
        _ => "is missing or null",
    }
}

/// The refusal of a write that breaks a unique index.
fn unique_violation(v: crate::index::UniqueViolation) -> ExecutionError {
    ExecutionError::UniqueViolation {
        value: match &v.value {
            Value::Array(values) => render_key(values),
            one => render_key(std::slice::from_ref(one)),
        },
        index: v.index_name,
        property: v.property,
        element_id: v.holder.to_element_id(),
    }
}

fn index_write_error(e: crate::index::IndexWriteError) -> ExecutionError {
    match e {
        crate::index::IndexWriteError::Unique(v) => unique_violation(v),
        crate::index::IndexWriteError::Store(e) => e.into(),
    }
}

/// The refusal of an insert whose key `holder` already holds in `table`.
fn duplicate_key(table: &str, key: &[Value], holder: NodeId) -> ExecutionError {
    ExecutionError::DuplicateKey {
        table: table.to_string(),
        key: render_key(key),
        element_id: holder.to_element_id(),
    }
}

/// A key as a reader writes it: the value alone, or a parenthesised tuple for
/// a compound key.
fn render_key(key: &[Value]) -> String {
    fn one(value: &Value) -> String {
        match value {
            Value::String(s) => format!("'{}'", s.replace('\'', "\\'")),
            Value::Int(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Timestamp(t) => format!("timestamp {t}"),
            Value::Binary(b) => format!("binary of {} bytes", b.len()),
            other => format!("{other:?}"),
        }
    }
    match key {
        [single] => one(single),
        _ => format!("({})", key.iter().map(one).collect::<Vec<_>>().join(", ")),
    }
}

/// The key of `record` under `columns`, or `None` when the record lacks one
/// of them.
fn row_key(
    columns: &[String],
    record: &NodeRecord,
    interner: &FieldInterner,
) -> Option<Vec<Value>> {
    columns
        .iter()
        .map(|column| {
            let id = interner.lookup(column)?;
            record.props.get(&id).cloned()
        })
        .collect()
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
        Self::new(100_000) // default feedback_cache_size
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
    /// The timestamp this statement's writes landed at, for a statement that
    /// committed on its own. `None` for a read, and for a statement of an
    /// interactive transaction, whose timestamp belongs to the commit that
    /// ends it rather than to any one statement inside it.
    ///
    /// It is the same number the commit receipt of an interactive transaction
    /// returns, and the same number a record's version is stated against, so
    /// a caller that wrote a record already holds the version to condition
    /// its next write on rather than having to read it back.
    pub commit_ts: Option<u64>,
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

/// Keyset-paging state for a server-side cursor over a single-`NodeScan` source.
///
/// `resume` / `limit` are inputs (the previous page's last key, and the cap on
/// storage keys read per page); `last_key` / `exhausted` are written back by the
/// scan as the next page's resume point and end-of-prefix flag. Present only for
/// keyset-pageable plans; the normal path leaves [`ExecutionContext::scan_paging`]
/// `None` and scans the whole prefix.
#[derive(Default, Clone)]
pub struct ScanPaging {
    /// The previous page's last key; `None` starts at the prefix.
    pub resume: Option<Vec<u8>>,
    /// Cap on storage keys read this page.
    pub limit: usize,
    /// Last key read this page: the next page's resume point.
    pub last_key: Option<Vec<u8>>,
    /// True when the scanned prefix has no rows beyond this page.
    pub exhausted: bool,
}

/// The vector indexes a statement searches and maintains, together with the
/// engine a `CREATE VECTOR INDEX` builds from in the background.
///
/// Held as one value so a context that can define a vector index can always
/// build it: the build outlives the statement and needs an owned engine.
#[derive(Clone, Copy)]
pub struct VectorIndexes<'a> {
    /// The registered indexes.
    pub registry: &'a crate::index::VectorIndexRegistry,
    /// The engine the indexes are built from, the one the context reads.
    pub engine: &'a Arc<StorageEngine>,
    /// How long a reader under the `block` policy waits for an index still
    /// being built before it fails: the query's hint, else the session's.
    pub build_wait: std::time::Duration,
}

pub struct ExecutionContext<'a> {
    pub engine: &'a StorageEngine,
    /// This statement's view of the field dictionary: the verified bindings
    /// when it started, plus those it registered since.
    pub interner: &'a mut FieldInterner,
    /// The authority new property names are registered with before data
    /// encoded with them is written. `None` when `interner` is itself the
    /// only authority (a context over a bare engine, as in tests).
    pub field_registrar: Option<&'a dyn coordinode_core::graph::intern::FieldRegistrar>,
    /// Node ID allocator for CREATE operations.
    pub id_allocator: &'a NodeIdAllocator,
    /// Default shard ID for single-node deployment.
    pub shard_id: u16,
    /// Keyset-paging state for a server-side cursor (see [`ScanPaging`]). `None`
    /// on the normal path: the single `NodeScan` source scans the whole prefix.
    pub scan_paging: Option<ScanPaging>,
    /// Live session registry for operational introspection, injected by the
    /// server. `SHOW SESSIONS` / `SHOW TRANSACTIONS` read a snapshot through it;
    /// `None` (embedded / tests) makes those statements report no sessions.
    pub operations: Option<&'a dyn coordinode_core::operations::OperationsView>,
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
    /// The statement's valid-time NOW, in microseconds since the epoch: the
    /// instant a temporal node read projects its timeline at unless the query
    /// names another. Bound once, so every read and write of one statement
    /// sees the same instant; independent of `snapshot_ts`.
    pub valid_now: i64,
    /// Valid-time instants a query named for a node variable (a
    /// `temporal_active_at(n, t)` conjunct), innermost last. Reads of that
    /// variable project at the named instant instead of `valid_now`.
    pub temporal_instants: Vec<(String, i64)>,
    /// GC-watermark pin for an `AS OF TIMESTAMP` read, held for the
    /// statement so compaction cannot collect the history it reads. `None`
    /// for reads at the current snapshot (never below the watermark).
    pub snapshot_pin: Option<coordinode_storage::engine::coordinator::SnapshotPin>,
    /// Warnings collected during execution (e.g., fan-out capping).
    pub warnings: Vec<String>,
    /// Write statistics accumulated during this statement.
    pub write_stats: WriteStats,
    /// Unique keys this statement claimed. A commit that loses a race for one
    /// of them is reported as the key existing, not as a bare write conflict.
    pub key_claims: KeyClaims,
    /// Optional full-text search index for text_match()/text_score() queries.
    /// Uses MultiLanguageTextIndex which wraps TextIndex with per-language support.
    /// 2-arg text_match(field, query) uses default language; 3-arg adds explicit language.
    /// When `text_index_registry` is set, this field is ignored in favor of
    /// registry-based lookup by (label, property).
    pub text_index: Option<&'a coordinode_search::tantivy::multi_lang::MultiLanguageTextIndex>,
    /// Text index registry for automatic full-text index management.
    /// When set, `execute_text_filter` resolves the text index by (label, property).
    /// Writes do not touch it: the text indexes follow the committed records,
    /// and a search waits for them to cover the store.
    pub text_index_registry: Option<&'a crate::index::TextIndexRegistry>,
    /// Vector indexes for HNSW-accelerated vector search and `CREATE VECTOR
    /// INDEX`. When set, VectorFilter checks for applicable HNSW indexes
    /// before falling back to brute-force distance computation; read the
    /// registry through [`Self::vector_index_registry`].
    pub vector_indexes: Option<VectorIndexes<'a>>,
    /// B-tree index registry for unique constraint enforcement.
    /// When set, `execute_create_node` calls `on_node_created` to check
    /// unique constraints and maintain B-tree index entries.
    pub btree_index_registry: Option<&'a crate::index::IndexRegistry>,
    /// The engine's executor of index builds. CREATE INDEX and CREATE
    /// CONSTRAINT admit a build with the index's publication and wait for
    /// its outcome here; the build itself does not belong to the statement.
    pub index_builds: Option<&'a crate::index::IndexBuildService>,
    /// Registry of extension-op handlers (EE-populated, CE default empty).
    /// Reached only by the [`LogicalOp::Extension`] dispatch arm; `None` /
    /// empty in pure-CE contexts means no extension ops are dispatchable.
    pub extensions: Option<&'a ExtensionRegistry>,
    /// Optional VectorLoader for disk-backed f32 reranking.
    /// When HNSW indexes have `offload_vectors` enabled, this loader provides
    /// f32 vectors from storage for exact reranking of SQ8 candidates.
    pub vector_loader: Option<&'a dyn coordinode_vector::VectorLoader>,
    /// MVCC timestamp oracle. When set, enables MVCC-versioned reads/writes.
    pub mvcc_oracle: Option<&'a TimestampOracle>,
    /// Per-shard `maxAssigned` watermark handle.
    ///
    /// Readers under `read_consistency = 'snapshot'` call
    /// `applied_watermark.wait_for(snapshot_ts, read_timeout)` before
    /// dispatching the read, so every modality on this shard observes the
    /// fully-applied state at `snapshot_ts`. `None` in legacy /
    /// single-writer test contexts — the executor then skips the wait and
    /// reads "current" state.
    pub applied_watermark:
        Option<std::sync::Arc<coordinode_core::txn::watermark::MaxAssignedWatermark>>,
    /// Cross-modality read consistency mode for this statement.
    /// Set by the planner from `LogicalPlan::read_consistency` (hint or
    /// auto-promotion). Default `Current` preserves the single-modality
    /// fast path.
    pub read_consistency: coordinode_core::txn::read_consistency::ReadConsistencyMode,
    /// Timeout for `applied_watermark.wait_for(snapshot_ts, …)` under
    /// `Snapshot` / `Exact` consistency. Default 2s, the documented
    /// `read_timeout`.
    pub read_timeout: std::time::Duration,
    /// MVCC read timestamp (start_ts). Allocated from oracle at statement start.
    /// All reads see a consistent snapshot at this timestamp.
    pub mvcc_read_ts: Timestamp,
    /// Layer-3 transaction context: owns the MVCC read snapshot,
    /// the read-your-own-writes write buffer, and the OCC read-set. The
    /// executor routes primitive `(partition, key)` reads/writes through it
    /// (`mvcc_get` / `mvcc_put` / `mvcc_delete` / `mvcc_prefix_scan` delegate
    /// here); modality-specific RYOW (node merge deltas) and the commit path
    /// still live on the executor and drive the transaction via its
    /// `pub(crate)` surface until they migrate too.
    pub txn: Transaction<'a>,
    /// The procedures `CALL` dispatches to. `None` refuses every call.
    pub procedures: Option<&'a crate::procedure::ProcedureRegistry>,
    /// Advisor state the `db.advisor.*` procedures read and reset; `None`
    /// makes them unavailable.
    pub advisor: Option<crate::advisor::AdvisorContext>,

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
    /// MVCC snapshot for point-in-time reads (native seqno MVCC).
    ///
    /// When MVCC is enabled, this snapshot is created at `mvcc_read_ts` via
    /// `engine.snapshot_at(seqno)`. All reads (mvcc_get, mvcc_prefix_scan)
    /// go through this snapshot for O(1) lookups instead of key-suffix prefix
    /// scanning. Also used for adj: partition reads (replaces adj_snapshot).
    ///
    /// When None (legacy mode), reads go directly through engine.get().
    pub mvcc_snapshot: Option<StorageSnapshot>,
    /// L1 cascade depth counter. Shared across all triggers in one
    /// originating user mutation. Incremented before each trigger body is
    /// executed ([`Self::cascade_enter`]), decremented when the body returns
    /// ([`Self::cascade_exit`]). When `cascade_depth > cascade_depth_limit`
    /// the firing is rejected with `ExecutionError::CascadeOverflow`.
    pub cascade_depth: u32,
    /// Cluster-default cap on `cascade_depth`. Per-trigger `CASCADE_LIMIT n`
    /// in the trigger definition tightens this further when present.
    /// Source: cluster setting `triggers.max_cascade_depth` (default 10).
    pub cascade_depth_limit: u32,
    /// L2 unique-trigger fanout map. Keyed by trigger name; value is the
    /// number of times that trigger has fired within the current cascade
    /// root. When `cascade_fire_counts[name] > cascade_fanout_limit` the
    /// firing is rejected with `ExecutionError::CascadeFanoutOverflow`.
    pub cascade_fire_counts: HashMap<String, u32>,
    /// Cluster-default cap on per-trigger fanout. Per-trigger `CASCADE_FANOUT n`
    /// tightens this further when present. Source: cluster setting
    /// `triggers.max_cascade_fanout` (default 100).
    pub cascade_fanout_limit: u32,
    /// Ordered chain of trigger names that have fired within the current
    /// cascade — used to populate the `cascade_chain` diagnostic field in
    /// dead-letter records when L1/L2 trip.
    pub cascade_chain: Vec<String>,
    /// Async AFTER COMMIT cascade generation of the statement being executed
    /// (the async counterpart of L1). `0` for a user statement; set to
    /// the queued event's `generation` when the dispatcher runs a trigger body,
    /// so any AFTER COMMIT events that body enqueues are stamped `generation + 1`
    /// and the dispatcher can bound the async cascade depth.
    pub after_commit_generation: u32,
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
    /// Per-statement cache: label → its schema, `None` for a label without
    /// one. Read on the first node write of a label, which binds the
    /// statement's commit to that schema revision; dropped for a label whose
    /// schema the statement changes.
    pub label_schema_cache: HashMap<String, Option<LabelSchema>>,
    /// Named query parameters bound to this statement.
    ///
    /// Populated before `execute()` is called. Accessible inside aggregate
    /// functions such as `percentileCont(x, $p)` where `$p` is a bound parameter.
    /// Keys omit the `$` prefix (e.g., `"p"` for `$p`).
    pub params: HashMap<String, coordinode_core::graph::types::Value>,
}

/// Ids of the engine-managed fields a temporal version carries:
/// `[valid_from, valid_to, __ingestion_ts__]`, registered together.
fn temporal_field_ids(ctx: &mut ExecutionContext<'_>) -> Result<[u32; 3], ExecutionError> {
    let ids = ctx.field_ids(&["valid_from", "valid_to", "__ingestion_ts__"])?;
    Ok([ids[0], ids[1], ids[2]])
}

impl<'a> ExecutionContext<'a> {
    /// The id of `name` for data this statement writes, registering the name
    /// first when it has none. The binding is durable before the id is
    /// returned, so no write encoded with it can outlive its meaning.
    ///
    /// # Errors
    ///
    /// The registration was refused or could not be published.
    #[inline]
    pub fn field_id(&mut self, name: &str) -> Result<u32, ExecutionError> {
        if let Some(id) = self.interner.lookup(name) {
            return Ok(id);
        }
        Ok(self.register_fields(&[name])?[0])
    }

    /// The ids of `names`, in order, registering the missing ones in one
    /// batch: a write that introduces several names pays one registration.
    ///
    /// # Errors
    ///
    /// As [`Self::field_id`].
    pub fn field_ids(&mut self, names: &[&str]) -> Result<Vec<u32>, ExecutionError> {
        if let Some(ids) = names.iter().map(|n| self.interner.lookup(n)).collect() {
            return Ok(ids);
        }
        self.register_fields(names)
    }

    #[cold]
    fn register_fields(&mut self, names: &[&str]) -> Result<Vec<u32>, ExecutionError> {
        let Some(registrar) = self.field_registrar else {
            return Ok(names.iter().map(|n| self.interner.intern(n)).collect());
        };
        let ids = registrar.register(names)?;
        // Callers index the result by position.
        if ids.len() != names.len() {
            return Err(DictionaryError::Registration(format!(
                "{} ids returned for {} names",
                ids.len(),
                names.len()
            ))
            .into());
        }
        for (name, &id) in names.iter().zip(&ids) {
            self.interner.insert_binding(name, id)?;
        }
        Ok(ids)
    }

    /// The vector index registry, when this context has vector indexes.
    pub fn vector_index_registry(&self) -> Option<&'a crate::index::VectorIndexRegistry> {
        self.vector_indexes.map(|v| v.registry)
    }

    /// L1+L2 cascade entry. Call before executing a trigger body.
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

    /// Load all enabled triggers that match a single
    /// `(target_segment, event)` mutation. The lookup is O(matching_triggers)
    /// — it reads exactly one index key + N definition keys, never scans the
    /// trigger table. Called at trigger firing time.
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
    /// executor's current values. The executor owns `mvcc_snapshot` /
    /// `mvcc_read_ts` (assigned post-construction and
    /// read pervasively), so we refresh the transaction's copies at the point
    /// of each read. Called at the entry of every read primitive; the
    /// transaction's OCC scope is created lazily at the first tracked read,
    /// pinned at the then-current `read_ts` (matching prior behaviour).
    fn sync_txn_state(&mut self) {
        self.txn.set_oracle(self.mvcc_oracle);
        self.txn.set_snapshot(self.mvcc_snapshot);
        self.txn.set_read_ts(self.mvcc_read_ts);
    }

    /// Re-pin this statement's read snapshot to the present.
    ///
    /// For the narrow case where the statement itself has just made an
    /// asynchronous writer stop: writes that landed between the statement's
    /// original snapshot and the moment the writer was joined are real history,
    /// not a concurrent transaction, and conflict detection must see them as
    /// such. Only sound once the writer is provably finished, and only before
    /// the statement has read anything it intends to keep — the moved snapshot
    /// would otherwise break repeatable-read within the statement.
    fn refresh_read_snapshot(&mut self) {
        let Some(oracle) = self.mvcc_oracle else {
            return;
        };
        // Allocated and pinned in one step: pinned later, the watermark can
        // have passed it by the first read.
        let (now, pin) = self.engine.pin_new_snapshot(|| oracle.next().as_raw());
        self.mvcc_read_ts = Timestamp::from_raw(now);
        self.mvcc_snapshot = Some(now);
        self.txn.adopt_snapshot(now, pin);
        self.sync_txn_state();
    }

    /// MVCC-aware read: write buffer → snapshot O(1) → legacy fallback.
    ///
    /// 1. Check write buffer (read-your-own-writes within this statement)
    /// 2. If MVCC snapshot set: snapshot.get() — O(1) native seqno MVCC
    /// 3. If no snapshot: direct engine.get() — legacy mode
    ///
    /// Generic test-access primitive used by this crate's and downstream
    /// crates' tests. Production execution reads through the typed Layer-4
    /// stores (no `Partition` in production query paths).
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
        // transaction.
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
    /// `schema:label:<name>:<revision>`. The schema partition is
    /// revision-prefixed: DDL such as `ALTER LABEL ... SET SCHEMA` writes a
    /// new revision and moves the pointer, so readers always go through it.
    pub fn load_current_label_schema(
        &mut self,
        name: &str,
    ) -> Result<Option<LabelSchema>, ExecutionError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        Ok(LocalSchemaStore::new(self.engine).load_label_txn(&mut self.txn, name)?)
    }

    /// Read the label schema a DDL statement writes the next revision of,
    /// conditioning the statement's commit on it still being the current
    /// one: two statements changing one label's schema cannot both build on
    /// the same revision.
    pub fn load_label_schema_for_update(
        &mut self,
        name: &str,
    ) -> Result<Option<LabelSchema>, ExecutionError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        Ok(LocalSchemaStore::new(self.engine).load_label_for_update_txn(&mut self.txn, name)?)
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
        self.label_schema_cache.remove(&schema.name);
        Ok(LocalSchemaStore::new(self.engine).save_label_txn(&mut self.txn, schema)?)
    }

    /// Drop a label by tombstoning its current-revision pointer (DROP TABLE).
    /// The label resolves to "not declared" from the next read.
    pub fn drop_current_label_schema(&mut self, name: &str) -> Result<(), ExecutionError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        self.label_schema_cache.remove(name);
        Ok(LocalSchemaStore::new(self.engine).drop_label_txn(&mut self.txn, name)?)
    }

    /// The label holding the constraint `name`, or `None` when no label has
    /// a constraint of that name.
    pub fn constraint_label(&mut self, name: &str) -> Result<Option<String>, ExecutionError> {
        use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
        self.sync_txn_state();
        Ok(LocalSchemaStore::new(self.engine).constraint_label_txn(&mut self.txn, name)?)
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

    /// Claim `key` of `table` for the row `node_id`. A key another row holds,
    /// committed or claimed earlier in this statement, is refused.
    pub fn claim_table_key(
        &mut self,
        table: &str,
        key: Vec<Value>,
        node_id: NodeId,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{LocalTableKeyStore, TableKeyStore as _};
        self.sync_txn_state();
        if let Some(holder) = LocalTableKeyStore.lookup(&mut self.txn, table, &key)? {
            return Err(duplicate_key(table, &key, holder));
        }
        LocalTableKeyStore.claim(&mut self.txn, table, &key, node_id)?;
        self.key_claims
            .tables
            .push((table.to_string(), key, node_id));
        Ok(())
    }

    /// State the unique values claimed since the `from`th claim whose index
    /// entries cannot prove them free as conditions of the commit, decided
    /// by reading the stored nodes as they stand when it is decided. An
    /// index still being built has not reached every stored node, so the
    /// commit reads the nodes past the key the backfill covered through; an
    /// index whose entries were found disagreeing with their records proves
    /// nothing, so the commit reads every node of the label.
    fn claim_building_uniques(&mut self, from: usize) {
        use coordinode_core::index::derive::tuples;
        use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope, UncoveredSource};
        let limit = self.index_builds.map_or(
            crate::index::DEFAULT_UNIQUE_ADMISSION_READ_LIMIT,
            |builds| builds.config().unique_admission_read_limit,
        );
        let revision = self.txn.schema_generation();
        let interner: &FieldInterner = self.interner;
        let field_of = |name: &str| interner.lookup(name);
        let mut stated = Vec::new();
        for claim in &self.key_claims.indexes[from..] {
            let building = matches!(claim.index.state, IndexState::Building { .. });
            if !building && !claim.needs_source_proof {
                continue;
            }
            let generation = claim.index.generation;
            let covered_through = if claim.needs_source_proof {
                None
            } else {
                self.index_builds
                    .and_then(|builds| builds.covered_through(generation))
            };
            let source = UncoveredSource {
                shard_id: self.shard_id,
                label: claim.index.label.clone(),
                interpretation: claim.index.interpretation(&field_of),
                covered_through,
                read_limit: limit,
            };
            for tuple in tuples(&claim.values) {
                stated.push(Claim::new(
                    ClaimScope::UniqueValue { generation, tuple },
                    ClaimPredicate::UniqueHolder {
                        node: claim.node_id,
                        uncovered: Some(Box::new(source.clone())),
                    },
                    revision,
                ));
            }
        }
        for claim in stated {
            self.txn.claim(claim);
        }
    }

    /// Stage the B-tree index entries of `record`, a node being created, or
    /// the version of a temporal node starting at `valid_from`. A unique
    /// value another node holds refuses the write.
    pub fn index_node_created(
        &mut self,
        node_id: NodeId,
        valid_from: Option<i64>,
        record: &NodeRecord,
    ) -> Result<(), ExecutionError> {
        let Some(registry) = self.btree_index_registry else {
            return Ok(());
        };
        let label = record.primary_label();
        if label.is_empty() || !registry.has_btree_for(label) {
            return Ok(());
        }
        self.sync_txn_state();
        let from = self.key_claims.indexes.len();
        {
            let interner: &FieldInterner = self.interner;
            let lookup = crate::index::registry::record_lookup(record, interner);
            let field_of = |name: &str| interner.lookup(name);
            registry
                .on_node_created(
                    self.engine,
                    &mut self.txn,
                    self.shard_id,
                    &crate::index::registry::NodeState {
                        node_id,
                        valid_from,
                        label,
                        value_of: &lookup,
                    },
                    &field_of,
                    &mut self.key_claims.indexes,
                )
                .map_err(index_write_error)?;
        }
        self.claim_building_uniques(from);
        Ok(())
    }

    /// Move the B-tree index entries of the version of temporal node
    /// `node_id` starting at `valid_from` as its record changes in place
    /// from `before` to `after` (closing it, reopening it). Each version has
    /// entries of its own, so only this version's move.
    pub fn index_version_changed(
        &mut self,
        node_id: NodeId,
        valid_from: i64,
        before: &NodeRecord,
        after: &NodeRecord,
    ) -> Result<(), ExecutionError> {
        let Some(registry) = self.btree_index_registry else {
            return Ok(());
        };
        let label = before.primary_label();
        if !registry.has_btree_for(label) {
            return Ok(());
        }
        self.sync_txn_state();
        let from = self.key_claims.indexes.len();
        {
            let interner: &FieldInterner = self.interner;
            let changed: Vec<&str> = before
                .props
                .keys()
                .chain(after.props.keys())
                .filter(|field| before.props.get(field) != after.props.get(field))
                .filter_map(|field| interner.resolve(*field))
                .collect();
            if changed.is_empty() {
                return Ok(());
            }
            let before_of = crate::index::registry::record_lookup(before, interner);
            let after_of = crate::index::registry::record_lookup(after, interner);
            let field_of = |name: &str| interner.lookup(name);
            registry
                .on_property_changed(
                    self.engine,
                    &mut self.txn,
                    self.shard_id,
                    &crate::index::PropertyChange {
                        node_id,
                        valid_from: Some(valid_from),
                        label,
                        properties: &changed,
                        before: &before_of,
                        after: &after_of,
                    },
                    &field_of,
                    &mut self.key_claims.indexes,
                )
                .map_err(index_write_error)?;
        }
        self.claim_building_uniques(from);
        Ok(())
    }

    /// Move the B-tree index entries of `record` as its `property` changes to
    /// `new_value` (`None`: the property is removed). Called before the
    /// record changes; a unique value another node holds refuses the write.
    pub fn index_property_changed(
        &mut self,
        node_id: NodeId,
        record: &NodeRecord,
        property: &str,
        new_value: Option<&Value>,
    ) -> Result<(), ExecutionError> {
        let Some(registry) = self.btree_index_registry else {
            return Ok(());
        };
        let label = record.primary_label();
        if !registry.has_btree_for(label) {
            return Ok(());
        }
        self.sync_txn_state();
        let from = self.key_claims.indexes.len();
        {
            let interner: &FieldInterner = self.interner;
            let before = crate::index::registry::record_lookup(record, interner);
            let after = |name: &str| {
                if name == property {
                    new_value.cloned()
                } else {
                    before(name)
                }
            };
            let field_of = |name: &str| interner.lookup(name);
            registry
                .on_property_changed(
                    self.engine,
                    &mut self.txn,
                    self.shard_id,
                    &crate::index::PropertyChange {
                        node_id,
                        valid_from: None,
                        label,
                        properties: &[property],
                        before: &before,
                        after: &after,
                    },
                    &field_of,
                    &mut self.key_claims.indexes,
                )
                .map_err(index_write_error)?;
        }
        self.claim_building_uniques(from);
        Ok(())
    }

    /// Whether a node of `label` has B-tree index entries to keep, so a
    /// write must know its record before the change.
    pub fn indexes_label(&self, label: &str) -> bool {
        self.btree_index_registry
            .is_some_and(|registry| registry.has_btree_for(label))
    }

    /// Move the B-tree index entries of a node whose `properties` go from
    /// their values in `before` to those in `after`, as a map SET changes
    /// several at once. Called before the change is written; a unique value
    /// another node holds refuses the write.
    pub fn index_record_changed(
        &mut self,
        node_id: NodeId,
        before: &NodeRecord,
        after: &NodeRecord,
        properties: &[&str],
    ) -> Result<(), ExecutionError> {
        let Some(registry) = self.btree_index_registry else {
            return Ok(());
        };
        let label = before.primary_label();
        if properties.is_empty() || !registry.has_btree_for(label) {
            return Ok(());
        }
        self.sync_txn_state();
        let from = self.key_claims.indexes.len();
        {
            let interner: &FieldInterner = self.interner;
            let before_of = crate::index::registry::record_lookup(before, interner);
            let after_of = crate::index::registry::record_lookup(after, interner);
            let field_of = |name: &str| interner.lookup(name);
            registry
                .on_property_changed(
                    self.engine,
                    &mut self.txn,
                    self.shard_id,
                    &crate::index::PropertyChange {
                        node_id,
                        valid_from: None,
                        label,
                        properties,
                        before: &before_of,
                        after: &after_of,
                    },
                    &field_of,
                    &mut self.key_claims.indexes,
                )
                .map_err(index_write_error)?;
        }
        self.claim_building_uniques(from);
        Ok(())
    }

    /// Move the B-tree index entries of a node whose declared properties go
    /// from `old` to `new` (keyed by field id), as a node merge rewrites its
    /// target. A unique value another node holds refuses the write.
    pub fn index_fields_changed(
        &mut self,
        node_id: NodeId,
        label: &str,
        old: &HashMap<u32, Value>,
        new: &HashMap<u32, Value>,
    ) -> Result<(), ExecutionError> {
        let Some(registry) = self.btree_index_registry else {
            return Ok(());
        };
        if !registry.has_btree_for(label) {
            return Ok(());
        }
        self.sync_txn_state();
        let interner: &FieldInterner = self.interner;
        let changed: Vec<&str> = old
            .keys()
            .chain(new.keys())
            .filter(|field| old.get(field) != new.get(field))
            .filter_map(|field| interner.resolve(*field))
            .collect();
        if changed.is_empty() {
            return Ok(());
        }
        let before = |name: &str| interner.lookup(name).and_then(|f| old.get(&f).cloned());
        let after = |name: &str| interner.lookup(name).and_then(|f| new.get(&f).cloned());
        let field_of = |name: &str| interner.lookup(name);
        let from = self.key_claims.indexes.len();
        registry
            .on_property_changed(
                self.engine,
                &mut self.txn,
                self.shard_id,
                &crate::index::PropertyChange {
                    node_id,
                    valid_from: None,
                    label,
                    properties: &changed,
                    before: &before,
                    after: &after,
                },
                &field_of,
                &mut self.key_claims.indexes,
            )
            .map_err(index_write_error)?;
        self.claim_building_uniques(from);
        Ok(())
    }

    /// The nodes whose entry in the B-tree index `id` holds exactly `value`,
    /// as this statement sees the index. `None` when the index cannot
    /// answer: it is not active here (dropped since the plan was built), not
    /// fully built, its entries were found disagreeing with their records,
    /// or `value` has no key.
    pub fn index_lookup(
        &mut self,
        id: crate::index::IndexId,
        value: &Value,
    ) -> Result<Option<Vec<NodeId>>, ExecutionError> {
        use coordinode_modality::{ENTRY_LAYOUT, IndexStore as _, LocalIndexStore};
        let Some(registry) = self.btree_index_registry else {
            return Ok(None);
        };
        let Some(index) = registry.get_by_id(id) else {
            return Ok(None);
        };
        if index.index_type != crate::index::IndexType::BTree
            || index.state != IndexState::Ready
            || index.layout != ENTRY_LAYOUT
            || registry.is_suspect(index.generation)
        {
            return Ok(None);
        }
        self.sync_txn_state();
        Ok(LocalIndexStore::new(self.engine).scan_exact(
            &mut self.txn,
            &index,
            std::slice::from_ref(value),
        )?)
    }

    /// Whether the entry of the B-tree index `id` under `value` that named
    /// `node` is one the index had no reason to hold: `record` is the node
    /// as this statement sees it (`None`: no such node), and the index's own
    /// interpretation of it (label, sparse and partial rules, each list
    /// element, each version of a temporal node) yields no such entry. A
    /// legitimate candidate the query's own comparison rejects (a list
    /// holding the value, a value a temporal node held before) is not one.
    /// A disagreement marks the generation suspect.
    ///
    /// # Errors
    ///
    /// A storage failure reading a temporal node's versions.
    pub fn index_entry_disagrees(
        &mut self,
        id: crate::index::IndexId,
        value: &Value,
        node: NodeId,
        record: Option<&NodeRecord>,
    ) -> Result<bool, ExecutionError> {
        use coordinode_core::index::derive::tuples;
        let Some(registry) = self.btree_index_registry else {
            return Ok(false);
        };
        let Some(index) = registry.get_by_id(id) else {
            return Ok(false);
        };
        let wanted = tuples(std::slice::from_ref(value));
        let interner: &FieldInterner = self.interner;
        let field_of = |name: &str| interner.lookup(name);
        let interpretation = index.interpretation(&field_of);
        let holds = |record: &NodeRecord| {
            record.primary_label() == index.label
                && interpretation
                    .record_membership(record)
                    .is_some_and(|held| tuples(&held).iter().any(|t| wanted.contains(t)))
        };
        if record.is_some_and(holds) {
            return Ok(false);
        }
        self.sync_txn_state();
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        if LocalNodeStore
            .versions(&self.txn, self.shard_id, node)?
            .iter()
            .any(|(_, version)| holds(version))
        {
            return Ok(false);
        }
        for tuple in wanted {
            registry.report_mismatch(
                &index,
                crate::index::Mismatch::Extra {
                    node: node.as_raw(),
                    valid_from: None,
                    tuple,
                },
            );
        }
        Ok(true)
    }

    /// Stage the removal of the B-tree index entries of `record`, a node
    /// being deleted.
    pub fn index_node_deleted(
        &mut self,
        node_id: NodeId,
        record: &NodeRecord,
    ) -> Result<(), ExecutionError> {
        let Some(registry) = self.btree_index_registry else {
            return Ok(());
        };
        let label = record.primary_label();
        if !registry.has_btree_for(label) {
            return Ok(());
        }
        self.sync_txn_state();
        let interner: &FieldInterner = self.interner;
        let lookup = crate::index::registry::record_lookup(record, interner);
        let field_of = |name: &str| interner.lookup(name);
        Ok(registry.on_node_deleted(
            self.engine,
            &mut self.txn,
            &crate::index::registry::NodeState {
                node_id,
                valid_from: None,
                label,
                value_of: &lookup,
            },
            &field_of,
        )?)
    }

    /// Stage and commit one catalog change in a transaction of its own,
    /// through this statement's commit path, outside the statement
    /// transaction: DDL whose record is written on the condition of its
    /// version, so two concurrent changes cannot both build on one state.
    pub fn commit_catalog_change<E>(
        &mut self,
        stage: impl FnOnce(
            &mut coordinode_storage::engine::transaction::Transaction<'_>,
        ) -> Result<(), E>,
    ) -> Result<(), ExecutionError>
    where
        ExecutionError: From<E>,
    {
        use coordinode_storage::engine::transaction::{CommitContext, Transaction};
        let mut txn = match self.mvcc_oracle {
            Some(oracle) => Transaction::begin(self.engine, Some(oracle), oracle.next()),
            None => Transaction::new(self.engine, None, Timestamp::ZERO, None),
        };
        stage(&mut txn)?;
        // Every writer of a catalogued object conditions its commit on the
        // record a change rewrites, so a change refused by them would lose
        // under steady writes. It waits for the ones already admitted (a
        // registration lasts from validation to apply) and makes later ones
        // queue behind it.
        txn.wait_for_overlapping_commits(CATALOG_ADMISSION_WAIT);
        let write_concern = self.write_concern;
        let outcome = txn
            .commit(&CommitContext {
                write_concern: &write_concern,
                pipeline: self.proposal_pipeline,
                id_gen: self.proposal_id_gen,
                drain_buffer: self.drain_buffer,
                nvme_write_buffer: self.nvme_write_buffer,
            })
            .map_err(catalog_commit_error)?;
        if let Some(index) = outcome.applied_index {
            self.write_stats.applied_index = self.write_stats.applied_index.max(Some(index));
        }
        Ok(())
    }

    /// Commit `mutations` as one log entry of their own, outside the
    /// statement transaction: DDL whose effects land together or not at all,
    /// such as an index definition and the range tombstone clearing the
    /// entries left under its name.
    pub fn propose_mutations(
        &mut self,
        mutations: Vec<coordinode_core::txn::proposal::Mutation>,
    ) -> Result<(), ExecutionError> {
        let (Some(pipeline), Some(id_gen)) = (self.proposal_pipeline, self.proposal_id_gen) else {
            // A context without a log (unit tests) has nothing to replicate to.
            use coordinode_modality::{IndexStore as _, LocalIndexStore};
            return Ok(LocalIndexStore::new(self.engine).apply_unreplicated(&mutations)?);
        };
        // Held as not yet logged until the entry is handed over, so no
        // closed bound passes over it meanwhile.
        let (commit_ts, _held) = match self.mvcc_oracle {
            Some(oracle) => {
                let (ts, held) = self
                    .engine
                    .pending_commits()
                    .obligate(|| oracle.next().as_raw());
                (Timestamp::from_raw(ts), Some(held))
            }
            None => (self.mvcc_read_ts, None),
        };
        let proposal = coordinode_core::txn::proposal::RaftProposal {
            id: id_gen.next(),
            mutations,
            commit_ts,
            start_ts: self.mvcc_read_ts,
            bypass_rate_limiter: false,
        };
        let outcome = pipeline
            .propose_and_wait(&proposal)
            .map_err(|e| ExecutionError::Unsupported(format!("commit DDL: {e}")))?;
        // A causal read after the statement fences past this entry too.
        self.write_stats.applied_index = self.write_stats.applied_index.max(outcome.applied_index);
        Ok(())
    }

    /// Free the key of `record`, a row being deleted, when its label is a
    /// table with declared key columns.
    pub fn release_table_key(&mut self, record: &NodeRecord) -> Result<(), ExecutionError> {
        use coordinode_core::schema::definition::TableKey;
        use coordinode_modality::{LocalTableKeyStore, TableKeyStore as _};
        let table = record.primary_label().to_string();
        let Some(schema) = self.load_current_label_schema(&table)? else {
            return Ok(());
        };
        let Some(TableKey::Columns(columns)) = schema.table_key() else {
            return Ok(());
        };
        let Some(key) = row_key(columns, record, self.interner) else {
            return Ok(());
        };
        self.sync_txn_state();
        LocalTableKeyStore.release(&mut self.txn, &table, &key)?;
        Ok(())
    }

    /// Free every key of `table` (DROP TABLE).
    pub fn release_all_table_keys(&mut self, table: &str) -> Result<(), ExecutionError> {
        use coordinode_modality::{LocalTableKeyStore, TableKeyStore as _};
        self.sync_txn_state();
        LocalTableKeyStore.release_all(&mut self.txn, table)?;
        Ok(())
    }

    /// Name the index and the value of a commit refused over a unique value
    /// this statement claimed: a holder found becomes
    /// [`ExecutionError::UniqueViolation`], an unresolved proof
    /// [`ExecutionError::UniquenessUnresolved`]. Any other refusal is handed
    /// back.
    fn explain_unique_refusal(
        &self,
        error: coordinode_storage::engine::transaction::CommitError,
    ) -> Result<ExecutionError, coordinode_storage::engine::transaction::CommitError> {
        use coordinode_storage::engine::transaction::CommitError;
        let explained = match &error {
            CommitError::UniqueValueHeld {
                generation,
                values,
                holder,
            } => self
                .key_claims
                .indexes
                .iter()
                .find(|claim| claim.index.generation == *generation)
                .map(|claim| {
                    unique_violation(crate::index::UniqueViolation::new(
                        &claim.index,
                        values,
                        *holder,
                    ))
                }),
            CommitError::UniquenessUnresolved { generation, limit } => self
                .key_claims
                .indexes
                .iter()
                .find(|claim| claim.index.generation == *generation)
                .map(|claim| ExecutionError::UniquenessUnresolved {
                    index: claim.index.to_string(),
                    limit: *limit,
                }),
            _ => None,
        };
        explained.ok_or(error)
    }

    /// Turn a commit refused for contention into [`ExecutionError::DuplicateKey`]
    /// when a key this statement claimed is now held by another row: two
    /// writers inserted one key and this one lost, so the key exists.
    fn explain_lost_key_race(&self, error: ExecutionError) -> ExecutionError {
        use coordinode_modality::{
            IndexStore as _, LocalIndexStore, LocalTableKeyStore, TableKeyStore as _,
        };
        if !matches!(error, ExecutionError::Conflict(_)) || self.key_claims.is_empty() {
            return error;
        }
        // The winner may have taken its timestamp and not applied yet; let the
        // commits already numbered land before asking who holds the key. This
        // is a read of the key's outcome, so it waits as long as a read does.
        if let Some(oracle) = self.mvcc_oracle {
            if self
                .engine
                .pending_commits()
                .await_complete_at(oracle.current().as_raw(), self.read_timeout)
                .is_err()
            {
                return error;
            }
        }
        for (table, key, claimed_for) in &self.key_claims.tables {
            if let Ok(Some(holder)) = LocalTableKeyStore.committed_holder(self.engine, table, key) {
                if holder != *claimed_for {
                    return duplicate_key(table, key, holder);
                }
            }
        }
        let indexes = LocalIndexStore::new(self.engine);
        // The latest committed records, read directly: the holder an entry
        // names is a duplicate only while its own record holds the value.
        let latest = coordinode_storage::engine::transaction::Transaction::new(
            self.engine,
            None,
            Timestamp::ZERO,
            None,
        );
        let interner: &FieldInterner = self.interner;
        let field_of = |name: &str| interner.lookup(name);
        for claim in &self.key_claims.indexes {
            let Ok(Some(holder)) =
                indexes.committed_conflict(&claim.index, &claim.values, claim.node_id)
            else {
                continue;
            };
            if crate::index::registry::holds_values(
                &latest,
                self.shard_id,
                &claim.index,
                &field_of,
                holder,
                &claim.values,
            )
            .unwrap_or(false)
            {
                return unique_violation(crate::index::UniqueViolation::new(
                    &claim.index,
                    &claim.values,
                    holder,
                ));
            }
        }
        error
    }

    /// Publish a new index as a catalog commit of its own, outside the
    /// statement transaction: the catalog gives it its identities and binds
    /// its name in that commit, so the index exists for every member before
    /// anything is built for it. A name another index holds is refused as
    /// [`ExecutionError::CatalogObjectExists`].
    pub fn publish_index_in_catalog(
        &mut self,
        descriptor: crate::index::IndexDescriptor,
    ) -> Result<crate::index::IndexDefinition, ExecutionError> {
        use coordinode_modality::{IndexStore as _, LocalIndexStore};
        let store = LocalIndexStore::new(self.engine);
        let mut published = None;
        self.commit_catalog_change(|txn| {
            published = Some(store.publish_definition_txn(txn, descriptor)?);
            Ok::<(), coordinode_modality::StoreError>(())
        })
        .map_err(name_taken)?;
        published.ok_or_else(|| {
            ExecutionError::Unsupported("the index publication staged no definition".into())
        })
    }

    /// Persist an existing index definition transactionally.
    pub fn mvcc_put_index_def(
        &mut self,
        def: &crate::index::IndexDefinition,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{IndexStore as _, LocalIndexStore};
        self.sync_txn_state();
        Ok(LocalIndexStore::new(self.engine).put_definition_txn(&mut self.txn, def)?)
    }

    /// Delete an index definition and its name binding transactionally
    /// (DROP INDEX DDL).
    pub fn mvcc_delete_index_def(
        &mut self,
        def: &crate::index::IndexDefinition,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{IndexStore as _, LocalIndexStore};
        self.sync_txn_state();
        Ok(LocalIndexStore::new(self.engine).delete_definition_txn(&mut self.txn, def)?)
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
        // the OCC-tracked read; the query layer keeps the decode +
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

    /// Stage the planner-statistics counter deltas for a node row created
    /// with `record`'s labels (total +1, each label +1). Called at every
    /// site that also counts `write_stats.nodes_created`, so the counters
    /// commit atomically with the write they describe.
    pub fn stat_node_created(&mut self, record: &NodeRecord) {
        use coordinode_modality::{LocalStatsStore, StatsStore as _};
        LocalStatsStore.node_created(&mut self.txn, record.labels.iter().map(String::as_str));
    }

    /// Counterpart of [`Self::stat_node_created`] for a deleted node row
    /// (total -1, each label -1); paired with `write_stats.nodes_deleted`.
    pub fn stat_node_deleted(&mut self, record: &NodeRecord) {
        use coordinode_modality::{LocalStatsStore, StatsStore as _};
        LocalStatsStore.node_deleted(&mut self.txn, record.labels.iter().map(String::as_str));
    }

    /// MVCC-aware typed node write: buffers the put on the transaction for
    /// atomic flush and RYOW visibility. Every node write passes here or
    /// through its temporal and columnar twins, which is where a node of a
    /// constrained label is recorded for the check of the state the
    /// transaction leaves it in.
    pub fn mvcc_put_node(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
        record: &NodeRecord,
    ) -> Result<(), ExecutionError> {
        // Delegate to Layer-4 LocalNodeStore (owns key encoding + msgpack);
        // the put buffers on the transaction for atomic flush.
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        if self.label_checks_nodes(record.primary_label())? {
            self.txn
                .note_post_state_check(&coordinode_core::graph::node::encode_node_key(
                    shard_id, node_id,
                ));
        }
        self.sync_txn_state();
        Ok(LocalNodeStore.put(&mut self.txn, shard_id, node_id, record)?)
    }

    /// Whether `label` has a presence or type constraint, which every node
    /// written under it is checked against before the commit. Reading the
    /// label's schema binds this statement's commit to that revision, so a
    /// constraint enabled meanwhile refuses the commit instead of missing
    /// the write.
    pub fn label_checks_nodes(&mut self, label: &str) -> Result<bool, ExecutionError> {
        use coordinode_core::schema::definition::NodeConstraint;
        if label.is_empty() {
            return Ok(false);
        }
        if !self.label_schema_cache.contains_key(label) {
            let schema = self.load_current_label_schema(label)?;
            self.label_schema_cache.insert(label.to_string(), schema);
        }
        Ok(matches!(
            self.label_schema_cache.get(label),
            Some(Some(schema)) if schema.constraints().iter().any(NodeConstraint::checks_each_node)
        ))
    }

    /// Refuse `record`, the state node `node_id` is stored in by a write that
    /// lands at once, outside the transaction, when it breaks a presence or
    /// type constraint of its primary label.
    fn check_constraints_now(
        &mut self,
        node_id: NodeId,
        record: &NodeRecord,
    ) -> Result<(), ExecutionError> {
        let label = record.primary_label();
        if !self.label_checks_nodes(label)? {
            return Ok(());
        }
        let Some(Some(schema)) = self.label_schema_cache.get(label) else {
            return Ok(());
        };
        let lookup = crate::index::registry::record_lookup(record, self.interner);
        coordinode_core::schema::validation::check_node_constraints(schema, &lookup)
            .map_err(|e| constraint_violation(e, label, node_id))
    }

    /// Write a node record into a `STORAGE COLUMNAR` table's own tree.
    ///
    /// Columnar tables live outside the Partition-based transaction, so the
    /// write goes straight to the table's columnar tree at a fresh oracle
    /// timestamp (the engine reconstructs rows from columnar blocks on read).
    /// Visible to subsequent statements (read-committed across statements), not
    /// snapshot-isolated within an open transaction. The write is journalled to
    /// the retained oplog at its commit_ts before it touches the tree, so an
    /// un-flushed row is replayed on the next open (crash recovery); the
    /// per-statement flush bounds how much must be replayed.
    pub fn columnar_put_node(
        &mut self,
        label: &str,
        node_id: NodeId,
        record: &NodeRecord,
    ) -> Result<(), ExecutionError> {
        self.check_constraints_now(node_id, record)?;
        let key = coordinode_core::graph::node::encode_node_key(self.shard_id, node_id);
        let bytes = record.to_msgpack().map_err(|e| {
            ExecutionError::Serialization(format!("columnar node {} encode: {e}", node_id.as_raw()))
        })?;
        let seqno = self
            .engine
            .oracle()
            .map(|o| o.next().as_raw())
            .unwrap_or_else(|| self.engine.snapshot().saturating_add(1));
        self.engine.columnar_insert(label, key, bytes, seqno)?;
        Ok(())
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

    /// The field ids a timeline projection reads. Looked up, never
    /// registered: a read does not add names to the dictionary.
    pub(crate) fn timeline_fields(&self) -> crate::executor::temporal_read::TimelineFields {
        crate::executor::temporal_read::TimelineFields {
            valid_to: self.interner.lookup("valid_to"),
            deleted: self.interner.lookup("__deleted__"),
        }
    }

    /// The valid-time instant reads of node variable `variable` project at:
    /// the innermost instant the query named for it, else `valid_now`.
    pub(crate) fn instant_for(&self, variable: &str) -> i64 {
        self.temporal_instants
            .iter()
            .rev()
            .find(|(v, _)| v == variable)
            .map_or(self.valid_now, |(_, at)| *at)
    }

    /// The state of temporal node `node_id`'s timeline at `at`, read from all
    /// of its versions under the statement's snapshot (OCC-tracked).
    pub(crate) fn temporal_node_state(
        &mut self,
        node_id: NodeId,
        at: i64,
    ) -> Result<crate::executor::temporal_read::StateAt, ExecutionError> {
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        let prefix = LocalNodeStore.version_prefix(self.shard_id, node_id);
        self.sync_txn_state();
        let scanned = LocalNodeStore.prefix_scan_tracked(&mut self.txn, &prefix)?;
        let versions = decode_versions(&scanned)?;
        Ok(crate::executor::temporal_read::state_at(
            versions,
            at,
            self.timeline_fields(),
        ))
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
        if self.label_checks_nodes(record.primary_label())? {
            self.txn.note_post_state_check(
                &coordinode_core::graph::node::encode_temporal_node_key(
                    shard_id,
                    node_id,
                    valid_from_ms,
                ),
            );
        }
        self.sync_txn_state();
        Ok(LocalNodeStore.put_temporal(&mut self.txn, shard_id, node_id, valid_from_ms, record)?)
    }

    /// Close the version of temporal node `node_id` that starts at
    /// `valid_from`: write `record`, that version, with its `valid_to` set to
    /// `valid_to`, and move the version's index entries with it.
    pub fn close_temporal_version(
        &mut self,
        node_id: NodeId,
        valid_from: i64,
        record: &mut NodeRecord,
        valid_to: i64,
    ) -> Result<(), ExecutionError> {
        let vt_fid = self.field_id("valid_to")?;
        // The record before the close, kept only when an index may read it.
        let open = if self.indexes_label(record.primary_label()) {
            Some(record.clone())
        } else {
            None
        };
        record.set(vt_fid, Value::Int(valid_to));
        self.mvcc_put_node_temporal(self.shard_id, node_id, valid_from, record)?;
        match open {
            Some(open) => self.index_version_changed(node_id, valid_from, &open, record),
            None => Ok(()),
        }
    }

    /// Write `record` as the new version of temporal node `node_id` that
    /// starts at `valid_from`, with the version's index entries.
    pub fn open_temporal_version(
        &mut self,
        node_id: NodeId,
        valid_from: i64,
        record: &NodeRecord,
    ) -> Result<(), ExecutionError> {
        self.mvcc_put_node_temporal(self.shard_id, node_id, valid_from, record)?;
        self.index_node_created(node_id, Some(valid_from), record)
    }

    /// MVCC-aware typed delete of a temporal node version (tombstone
    /// at the specific 25-byte temporal key). Reserved for the
    /// privileged erase operation that removes individual versions
    /// physically; the standard bitemporal delete path uses
    /// close-current + tombstone-version inserts via
    /// [`Self::mvcc_put_node_temporal`] instead. No production caller
    /// in coordinode-query yet — added alongside the read/write pair so
    /// the typed surface stays symmetric.
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

    /// Re-point an edge's properties from `(old_src, old_tgt)` to
    /// `(new_src, new_tgt)`, moving every per-version body for a temporal edge
    /// type (preserving each `valid_from`) or the single body otherwise. The
    /// adjacency move is identical for both — only the edgeprop store differs.
    pub fn mvcc_move_edge_props_versioned(
        &mut self,
        edge_type: &str,
        old_src: NodeId,
        old_tgt: NodeId,
        new_src: NodeId,
        new_tgt: NodeId,
        temporal: bool,
    ) -> Result<(), ExecutionError> {
        if temporal {
            self.mvcc_transfer_all_edge_prop_versions(edge_type, old_src, old_tgt, new_src, new_tgt)
        } else {
            self.mvcc_move_edge_props(edge_type, old_src, old_tgt, new_src, new_tgt)
        }
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
    ///
    /// A document delta makes its root property a document whatever it held
    /// before, so a node of a constrained label is recorded for the check of
    /// the state the transaction leaves it in, as a whole write is.
    pub fn mvcc_merge_node_delta(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
        operand: Vec<u8>,
    ) -> Result<(), ExecutionError> {
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        if let Some(label) = self.schema_label_for_node(shard_id, node_id)? {
            if self.label_checks_nodes(&label)? {
                self.txn
                    .note_post_state_check(&coordinode_core::graph::node::encode_node_key(
                        shard_id, node_id,
                    ));
            }
        }
        LocalNodeStore.buffer_node_delta(&mut self.txn, shard_id, node_id, operand);
        Ok(())
    }

    /// The node as this transaction leaves it, through a tracked read that
    /// keeps its pending document deltas pending: the read a writer makes
    /// between the deltas it stages, which [`Self::mvcc_get_node`] would
    /// turn into a whole write of the record.
    pub fn mvcc_node_post_state(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
    ) -> Result<Option<NodeRecord>, ExecutionError> {
        Ok(self
            .mvcc_node_post_state_sized(shard_id, node_id)?
            .map(|(record, _)| record))
    }

    /// [`Self::mvcc_node_post_state`] with the size of the stored record the
    /// pending deltas apply to.
    pub fn mvcc_node_post_state_sized(
        &mut self,
        shard_id: u16,
        node_id: NodeId,
    ) -> Result<Option<(NodeRecord, usize)>, ExecutionError> {
        use coordinode_modality::LocalNodeStore;
        self.sync_txn_state();
        let key = LocalNodeStore::record_key(shard_id, node_id);
        Ok(LocalNodeStore::post_state_tracked(&mut self.txn, &key)?)
    }

    /// Whether a property change of `delta_bytes` to the node whose stored
    /// record is `stored_bytes` long is written as a delta rather than as the
    /// record.
    ///
    /// A delta saves rewriting a large record, but every read of the key
    /// unpacks and folds every delta written since its last whole write,
    /// until a compaction reaches its base. A small record is cheaper to
    /// rewrite than to leave a chain behind, a change carrying a large share
    /// of the record saves little and costs every read its size, and a long
    /// chain is ended by a whole write (the reason RocksDB has
    /// `max_successive_merges`).
    fn writes_property_delta(
        &self,
        shard_id: u16,
        node_id: NodeId,
        stored_bytes: usize,
        delta_bytes: usize,
    ) -> bool {
        /// Records smaller than this are rewritten whole.
        const MIN_DELTA_RECORD_BYTES: usize = 1024;
        /// A change larger than this share of the record is written whole.
        const MAX_DELTA_SHARE_DIVISOR: usize = 4;
        /// Deltas in a row on one key before the next write is whole.
        const MAX_SUCCESSIVE_DELTAS: u32 = 32;
        if stored_bytes < MIN_DELTA_RECORD_BYTES
            || delta_bytes > stored_bytes / MAX_DELTA_SHARE_DIVISOR
        {
            return false;
        }
        let key = coordinode_modality::LocalNodeStore::record_key(shard_id, node_id);
        self.engine.node_delta_run(&key) < MAX_SUCCESSIVE_DELTAS
    }

    /// Note how the property change of `node_id` was written, for the next
    /// writer's choice.
    fn note_property_write(&self, shard_id: u16, node_id: NodeId, as_delta: bool) {
        let key = coordinode_modality::LocalNodeStore::record_key(shard_id, node_id);
        if as_delta {
            self.engine.note_node_delta(&key);
        } else {
            self.engine.note_node_whole_write(&key);
        }
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
        // locus: OCC validation, commit_ts assignment, write-concern
        // fan-out, and the Raft proposal pipeline all live in `Transaction`.
        self.sync_txn_state();
        // Constraints judge each node as the statement leaves it.
        check_post_state(&mut self.txn, self.engine, self.interner)?;
        // Asked before the commit drains the buffers, which is the only
        // moment the answer exists.
        let wrote = !self.txn.write_buffer_is_empty() || self.txn.has_pending_merges();
        let write_concern = self.write_concern;
        let ctx = coordinode_storage::engine::transaction::CommitContext {
            write_concern: &write_concern,
            pipeline: self.proposal_pipeline,
            id_gen: self.proposal_id_gen,
            drain_buffer: self.drain_buffer,
            nvme_write_buffer: self.nvme_write_buffer,
        };
        let outcome = match self.txn.commit(&ctx) {
            Ok(outcome) => outcome,
            Err(e) => {
                let e = match self.explain_unique_refusal(e) {
                    Ok(explained) => return Err(explained),
                    Err(e) => e,
                };
                return Err(self.explain_lost_key_race(commit_err_to_execution(e)));
            }
        };
        // operationTime spans every Raft entry the statement produced — `max`,
        // not assign: a statement may have issued an earlier in-execute
        // proposal (e.g. CREATE VECTOR INDEX schema persist) whose index must
        // remain covered. (`Option` orders `None < Some`.)
        self.write_stats.applied_index = self.write_stats.applied_index.max(outcome.applied_index);
        // The timestamp travels with the statistics rather than only as a
        // return value, because the caller that needs it is two layers up and
        // reads the statistics anyway: a statement that wrote a record holds
        // the record's new version without reading it back.
        //
        // Only when there was something to commit. A read-only commit answers
        // with the timestamp it read at, which is the right answer to "when
        // did this transaction see the world" and the wrong one to "what
        // version is this record now": reporting it would hand a caller a
        // number that no write ever landed at, and a conditional write
        // against it would be refused for a reason nobody could explain.
        if wrote {
            self.write_stats.commit_ts = outcome.commit_ts.map(|ts| ts.as_raw());
        }
        Ok(outcome.commit_ts)
    }

    /// Materialize pending node merge deltas for a key into the write buffer.
    ///
    /// Called lazily from the generic `mvcc_get()` primitive. Production node
    /// reads materialise deltas inside `LocalNodeStore`.
    fn materialize_node_deltas(&mut self, node_key: &[u8]) -> Result<(), ExecutionError> {
        // Node-delta read-your-own-writes lives in Layer-4 LocalNodeStore
        // (node-modality concern); the modality-agnostic transaction does not
        // own it.
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
    // key or `Partition::Adj` in the query layer.

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

        // This transaction's own staged operands, replayed in the order they
        // were staged (read-your-own-writes). An add and a remove of one
        // member do not commute, so the order is what makes this the state the
        // commit will produce rather than a different one.
        self.txn.apply_staged_adj(adj_key, &mut plist);

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
    // Adjacency stays off the OCC path by construction: edge add/remove
    // goes through the posting-list merge operator, which is commutative.

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

    /// State that this statement observed the pair as not adjacent, and built
    /// its result on that.
    ///
    /// Only a statement whose decision turned on the observation says this.
    /// A plain CREATE does not: it attaches regardless of what was there, so
    /// claiming an observation it never made would refuse writers that agree
    /// with each other.
    pub fn claim_pair_observed_absent(&mut self, edge_type: &str, source: NodeId, target: NodeId) {
        use coordinode_core::txn::invariant::{Adjacency, Claim, ClaimPredicate, ClaimScope};
        let generation = self.txn.schema_generation();
        self.txn.claim(Claim::new(
            ClaimScope::Pair {
                source,
                target,
                edge_type: edge_type.to_string(),
            },
            ClaimPredicate::PairAdjacency {
                observed: Adjacency::Absent,
            },
            generation,
        ));
    }

    /// State that this statement enumerated the whole incident set of the
    /// scope and its result depends on nothing having joined or left it.
    ///
    /// A detach, a node merge or a redirect is only correct if it saw every
    /// member. The rows it read prove nothing about the one that arrives
    /// while it is reading: that member and this statement write different
    /// keys, so first-committer-wins never compares them.
    pub fn claim_incident_set_complete(&mut self, edge_type: &str, node: NodeId, outgoing: bool) {
        use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope, Direction};
        let generation = self.txn.schema_generation();
        self.txn.claim(Claim::new(
            ClaimScope::Incident {
                node,
                edge_type: edge_type.to_string(),
                direction: if outgoing {
                    Direction::Outgoing
                } else {
                    Direction::Incoming
                },
            },
            ClaimPredicate::IncidentSetComplete,
            generation,
        ));
    }

    /// State that this statement enumerated every incident edge of `node`,
    /// of every type and in both directions, including types it did not know
    /// of when it read. Used where no type filter bounds what was promised.
    pub fn claim_all_incident_sets_complete(&mut self, node: NodeId) {
        use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope};
        let generation = self.txn.schema_generation();
        self.txn.claim(Claim::new(
            ClaimScope::Node(node),
            ClaimPredicate::IncidentSetComplete,
            generation,
        ));
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
    let crate::plan::expr::Expr::Variable(target_var) = &item.arg else {
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
            | LogicalOp::TextIndexScan { .. }
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
/// transactions call [`execute_no_commit`] directly and commit the
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
/// auto-commit; interactive transactions run N statements through
/// this entry against one shared `ctx.txn` and commit once at the end.
pub fn execute_no_commit(
    plan: &LogicalPlan,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Take MVCC snapshot at mvcc_read_ts for native seqno reads.
    // All reads (node, schema, edgeprop) go through this snapshot for O(1) lookups.
    // When oracle is set, snapshot_at(seqno) pins the LSM tree at read_ts.
    // For legacy mode (no oracle), mvcc_snapshot stays None → direct engine reads.
    //
    // A transaction that already chose its view (an interactive one, begun
    // at the engine's complete snapshot and pinned there) keeps it. Its
    // `read_ts` was allocated before that view was taken and can sit below
    // it: reading there would lose the view's completeness, and re-pinning
    // it can be refused once the watermark has followed the view.
    if ctx.mvcc_snapshot.is_none() && ctx.mvcc_oracle.is_some() {
        ctx.mvcc_snapshot = ctx.txn.snapshot().or_else(|| {
            ctx.engine
                .snapshot_at(ctx.mvcc_read_ts.as_raw())
                .or_else(|| Some(ctx.engine.snapshot()))
        });
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
    // Since commit_ts = seqno (OracleSeqnoGenerator), the timestamp value
    // is directly usable as a snapshot seqno for both node and adj partitions.
    if let Some(ref ts_expr) = plan.snapshot_ts {
        let ts_val = eval_neutral(ts_expr, &Row::new())?;
        let resolved_ts: Option<i64> = match ts_val {
            Value::Timestamp(ts) => Some(ts),
            // The HLC value is wall-clock microseconds since the Unix epoch,
            // so an RFC 3339 instant (RFC 3339 §5.6, zone offset required)
            // names the same point on it.
            Value::String(ref s) => Some(
                chrono::DateTime::parse_from_rfc3339(s)
                    .map_err(|e| {
                        ExecutionError::Unsupported(format!(
                            "AS OF TIMESTAMP '{s}' is not an RFC 3339 timestamp \
                             (for example 2026-03-15T10:00:00Z): {e}"
                        ))
                    })?
                    .timestamp_micros(),
            ),
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
            //
            // `AS OF TIMESTAMP T` is inclusive: it sees every commit with
            // `commit_ts <= T`, so a commit receipt's `commit_ts` is a valid
            // anchor for its own write. A storage snapshot at seqno S sees
            // versions with seqno STRICTLY below S, and a commit lands at
            // exactly its commit_ts (one seqno per proposal), hence T + 1.
            #[allow(clippy::cast_sign_loss)]
            let seqno = (ts as u64).checked_add(1).ok_or_else(|| {
                ExecutionError::Unsupported("AS OF TIMESTAMP value out of range".into())
            })?;
            // Pin the snapshot for the statement. The pin is refused when the
            // seqno is already below the GC watermark: that history may be
            // collected, and a read there would answer from whatever survived
            // (a newer version, or nothing). Refusing under the watermark's
            // own lock means a granted pin always protects live history.
            let Some(pin) = ctx.engine.pin_snapshot_at(seqno) else {
                return Err(ExecutionError::OutsideRetention {
                    requested: ts,
                    // A read at T is served from snapshot T + 1, so the oldest
                    // readable timestamp is one below the first readable seqno.
                    // The clamp is the bottom of the timestamp domain, not an
                    // overflow guard: a horizon of zero means nothing has been
                    // collected and there is no timestamp below zero to name.
                    oldest_readable: ctx.engine.oldest_readable_seqno().saturating_sub(1),
                });
            };
            ctx.snapshot_pin = Some(pin);
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

    // Propagate the plan's cross-modality consistency decision into
    // the execution context so downstream operators (VectorFilter, etc.)
    // observe it without a separate parameter. The planner has already
    // applied auto-promotion and the narrower `vector_consistency` override.
    ctx.read_consistency = plan.read_consistency;
    ctx.vector_consistency = plan.vector_consistency;

    // Cross-modality snapshot wait. When `read_consistency` is
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

        LogicalOp::NodeCountFromCounter {
            label,
            column,
            fallback,
        } => match counted_nodes(label, ctx)? {
            Some(n) => {
                let mut row = Row::new();
                row.insert(column.clone(), Value::Int(n));
                Ok(vec![row])
            }
            None => execute_op(fallback, ctx),
        },

        LogicalOp::IndexScan {
            variable,
            label,
            index,
            index_name: _,
            property,
            value_expr,
        } => execute_btree_index_scan(variable, label, *index, property, value_expr, ctx),

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

        // Index access path for text_match over one label: the index's
        // matches are the row source; only those nodes are fetched.
        LogicalOp::TextIndexScan {
            label,
            property,
            binding,
            query_string,
            language,
        } => execute_text_index_scan(
            label,
            property,
            binding,
            query_string,
            language.as_deref(),
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

            // Wildcard relationship pattern `MATCH (n)-[r]->(m)` — no type filter.
            // When edge_types is empty, expand over all schema-registered edge types.
            // This scans `schema:edge_type:<name>` keys (written on every CREATE edge)
            // plus any uncommitted registrations in the current transaction's write buffer.
            // Hoist temporal flag per edge type once per traversal. A wildcard
            // passes over the types no statement writes edges of; a named
            // one is refused.
            let resolved_types: Vec<String>;
            let mut edge_temporal: Vec<bool>;
            let effective_types: &[String] = if edge_types.is_empty() {
                let all = ctx.list_edge_types()?;
                let mut kept = Vec::with_capacity(all.len());
                edge_temporal = Vec::with_capacity(all.len());
                for et in all {
                    if let Ok(temporal) = edge_type_shape(&et, ctx)? {
                        edge_temporal.push(temporal);
                        kept.push(et);
                    }
                }
                resolved_types = kept;
                &resolved_types
            } else {
                edge_temporal = Vec::with_capacity(edge_types.len());
                for et in edge_types {
                    edge_temporal.push(lookup_edge_type_temporal(et, ctx)?);
                }
                edge_types
            };

            // Edge properties are only materialised under an edge binding, so
            // an anonymous relationship with an inline property map
            // (`-[:R {w: 5}]->`) binds one no query can name, filters through
            // it, and drops its columns before the rows leave this operator.
            let anonymous_edge_filtered = edge_variable.is_none() && !edge_filters.is_empty();
            let params = TraverseParams {
                source,
                edge_types: effective_types,
                direction: *direction,
                target_variable,
                target_labels,
                length: *length,
                edge_variable: if anonymous_edge_filtered {
                    Some(ANONYMOUS_EDGE_BINDING)
                } else {
                    edge_variable.as_deref()
                },
                target_filters,
                edge_filters,
                edge_temporal: &edge_temporal,
                temporal_filter: temporal_filter.as_ref(),
                path_variable: path_variable.as_deref(),
            };
            let mut rows = execute_traverse(&input_rows, &params, ctx)?;
            if anonymous_edge_filtered {
                for row in &mut rows {
                    row.retain(|k, _| {
                        k.strip_prefix(ANONYMOUS_EDGE_BINDING)
                            .is_none_or(|rest| !rest.is_empty() && !rest.starts_with('.'))
                    });
                }
            }
            Ok(rows)
        }

        LogicalOp::Filter { input, predicate } => {
            // `temporal_active_at(n, t)` names the instant the read binding
            // `n` (a scan, an index lookup, a traversal) projects temporal
            // timelines at; without it `n` is read at the statement's NOW
            // and the predicate could only reject.
            let scope = ctx.temporal_instants.len();
            named_instants(predicate, &mut ctx.temporal_instants);
            let rows = execute_filter_input(input, predicate, ctx);
            ctx.temporal_instants.truncate(scope);
            let rows = rows?;
            let corr = ctx.correlated_row.clone();
            if neutral_contains_subplan(predicate) {
                // Storage-aware path: correlated subplans need edge lookups.
                let mut result = Vec::new();
                for row in rows {
                    let effective_row = if let Some(ref outer) = corr {
                        let mut merged = outer.clone();
                        merged.extend(row.iter().map(|(k, v)| (k.clone(), v.clone())));
                        merged
                    } else {
                        row.clone()
                    };
                    let val = eval_neutral_with_storage(predicate, &effective_row, ctx)?;
                    if is_truthy(&val) {
                        result.push(row);
                    }
                }
                Ok(result)
            } else {
                // A loop rather than `filter`, because evaluating a predicate
                // can fail and a filter closure has nowhere to put the failure.
                let mut kept = Vec::with_capacity(rows.len());
                for row in rows {
                    let keep = if let Some(ref outer) = corr {
                        // Correlated OPTIONAL MATCH: merge outer-scope variables
                        // so predicates like `c.age > a.age` can resolve `a`.
                        // Current row takes precedence over outer scope.
                        let mut merged = outer.clone();
                        merged.extend(row.iter().map(|(k, v)| (k.clone(), v.clone())));
                        is_truthy(&eval_neutral(predicate, &merged)?)
                    } else {
                        is_truthy(&eval_neutral(predicate, &row)?)
                    };
                    if keep {
                        kept.push(row);
                    }
                }
                Ok(kept)
            }
        }

        LogicalOp::Project {
            input,
            items,
            distinct,
        } => {
            let rows = execute_op(input, ctx)?;

            // text_score guard: `text_score()` relies on `__text_score__`
            // populated by an upstream `TextFilter`. If a projection references
            // `text_score` but TextFilter never ran (missing FT index, or no
            // paired `text_match(...)` in WHERE), we must fail with a clear
            // error rather than silently returning 0.0. See regression test
            // `text_score_without_text_match_errors`.
            let score_reqs: crate::executor::eval::ScoreRequirements = items
                .iter()
                .map(|it| crate::executor::eval::expr_score_requirements_neutral(&it.expr))
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
                    if item.expr == crate::plan::expr::Expr::Star {
                        // Star: copy all columns
                        out.extend(row.clone());
                    } else {
                        // A column already carrying this expression's own
                        // rendered name was computed upstream from bindings
                        // this row no longer has: an aggregate emits each
                        // grouping key that way, and by the time projection
                        // runs, `n` and `r` are gone. Re-deriving it there
                        // yields NULL, so `type(r) AS rel, count(*)` grouped
                        // correctly and then could not name the group. Take
                        // the materialised value; a bare property resolves to
                        // the same column either way.
                        let materialised = match &item.expr {
                            crate::plan::expr::Expr::Variable(_)
                            | crate::plan::expr::Expr::Star => None,
                            expr => row.get(&expr_display_name_neutral(expr)).cloned(),
                        };
                        let val = match materialised {
                            Some(v) => v,
                            None if neutral_contains_subplan(&item.expr) => {
                                eval_neutral_with_storage(&item.expr, &row, ctx)?
                            }
                            None => eval_neutral(&item.expr, &row)?,
                        };
                        let key = item
                            .alias
                            .clone()
                            .unwrap_or_else(|| expr_display_name_neutral(&item.expr));
                        out.insert(key.clone(), val);

                        // Variable passthrough: when a projection item is
                        // bare `Variable(x)` (and is left unrenamed, OR is
                        // aliased — in which case we re-bind the alias's
                        // property columns), also propagate every `x.prop`
                        // and `x.__*__` auxiliary column from the input
                        // row. Without this, `MATCH (a) WITH a RETURN
                        // a.prop` would lose all property bindings at the
                        // WITH barrier and `a.prop` would resolve to NULL.
                        if let crate::plan::expr::Expr::Variable(var_name) = &item.expr {
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
            let rows = execute_op(input, ctx)?;

            // Same text_score guard as Project: an `ORDER BY text_score(...)` (bare
            // or inside arithmetic) without an upstream TextFilter would sort
            // by silent zeros. Error instead.
            let score_reqs: crate::executor::eval::ScoreRequirements = items
                .iter()
                .map(|it| crate::executor::eval::expr_score_requirements_neutral(&it.expr))
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

            // Evaluate each sort key once per row, then sort on the results. A
            // comparator has nowhere to report a failed evaluation, and it is
            // called O(n log n) times, so evaluating inside it also re-computed
            // every key on every comparison.
            //
            // The keys live in one flat buffer, row-major, rather than a vector
            // per row: this is a hot path and a per-row allocation is exactly
            // the cost worth not paying. Rows travel with their index instead
            // of with their keys, so the whole sort allocates a fixed three
            // times regardless of how many rows arrive.
            let width = items.len();
            let mut keys: Vec<Value> = Vec::with_capacity(rows.len() * width);
            for row in &rows {
                for item in items {
                    keys.push(eval_neutral(&item.expr, row)?);
                }
            }
            let mut indexed: Vec<(usize, Row)> = rows.into_iter().enumerate().collect();
            indexed.sort_by(|(a, _), (b, _)| {
                for (idx, item) in items.iter().enumerate() {
                    let cmp = compare_values(&keys[a * width + idx], &keys[b * width + idx]);
                    let cmp = if item.ascending { cmp } else { cmp.reverse() };
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                std::cmp::Ordering::Equal
            });
            Ok(indexed.into_iter().map(|(_, row)| row).collect())
        }

        LogicalOp::Limit { input, count } => {
            let rows = execute_op(input, ctx)?;
            let n = eval_neutral(count, &Row::new())?;
            if let Value::Int(limit) = n {
                Ok(rows.into_iter().take(limit.max(0) as usize).collect())
            } else {
                Ok(rows)
            }
        }

        LogicalOp::Skip { input, count } => {
            let rows = execute_op(input, ctx)?;
            let n = eval_neutral(count, &Row::new())?;
            if let Value::Int(skip) = n {
                Ok(rows.into_iter().skip(skip.max(0) as usize).collect())
            } else {
                Ok(rows)
            }
        }

        LogicalOp::CartesianProduct { left, right } => {
            let left_rows = execute_op(left, ctx)?;

            // MERGE (src)-[r:TYPE]->(tgt) in a CartesianProduct context requires
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
            push_down,
        } => {
            let rows = execute_op(input, ctx)?;
            let mode = ctx.vector_consistency;

            let score_params = VectorScoreParams {
                function,
                less_than: *less_than,
                threshold: *threshold,
                decay_field: decay_field.as_ref(),
            };

            // A threshold predicate defines a SET ("every row whose score
            // passes τ"), not a ranking, so it is evaluated exactly over the
            // materialised candidate set. An ANN index cannot serve it: HNSW
            // answers top-k, and any k-bounded pre-filter drops candidates
            // that pass τ but rank below k globally — a different answer, not
            // a lower recall. Graph-predicate push-down forbids the same
            // thing: never materialise `C` and then run an unfiltered HNSW
            // scan ignoring `C`.
            //
            // Nothing is lost by evaluating exactly: each surviving row's
            // score is recomputed from the row's own vector anyway, so the
            // index never saved a distance computation here — it only skipped
            // rows, and for a candidate set smaller than the index the graph
            // search costs more than the distances it skips. Index-accelerated
            // vector access lives in the top-k operators (`VectorTopK`,
            // `HnswScan`), where approximation is part of the contract.
            //
            // The planner's push-down decision is carried for EXPLAIN and
            // recorded here; it selects among physical strategies for top-k
            // access, and `graph_first` (exact scoring over `C`) is the only
            // one that preserves threshold semantics.
            tracing::trace!(
                strategy = push_down.as_ref().map(|d| d.strategy.as_wire_str()),
                candidates = rows.len(),
                "vector_filter: exact evaluation over the materialised candidate set"
            );
            let result = execute_vector_filter(&rows, vector_expr, query_vector, &score_params)?;
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
            materialize_own_node_writes(ctx)?;
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
            input,
            procedure,
            args,
            yields,
            filter,
            standalone,
        } => execute_procedure_call(
            input,
            procedure,
            args,
            yields.as_deref(),
            filter.as_ref(),
            *standalone,
            ctx,
        ),

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
            maintenance,
            on_duplicate_rename,
        } => execute_create_btree_index(
            name,
            label,
            property,
            *unique,
            *sparse,
            filter.as_ref(),
            *maintenance,
            on_duplicate_rename.as_deref(),
            ctx,
        ),

        LogicalOp::DropIndex { name } => execute_drop_btree_index(name, ctx),

        LogicalOp::CreateConstraint {
            name,
            if_not_exists,
            label,
            properties,
            kind,
            wait,
            on_duplicate_rename,
            scope,
        } => execute_create_constraint(
            name.as_deref(),
            *if_not_exists,
            label,
            properties,
            kind,
            OwnedIndexShape {
                wait: *wait,
                on_duplicate_rename: on_duplicate_rename.clone(),
                scope: scope.clone(),
                ..OwnedIndexShape::default()
            },
            ctx,
        ),

        LogicalOp::DropConstraint { name, if_exists } => {
            execute_drop_constraint(name, *if_exists, ctx)
        }

        LogicalOp::AlterIndexMaintenance { name, profile } => {
            execute_alter_index_maintenance(name, *profile, ctx)
        }

        LogicalOp::Reindex { name, label } => execute_reindex(name, label.as_deref(), ctx),

        LogicalOp::SetNamespaceIndexDefault { profile } => {
            execute_set_namespace_index_default(*profile, ctx)
        }

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
            discriminated_by,
        } => execute_create_edge_type(
            name,
            *temporal,
            properties,
            discriminated_by.as_deref(),
            ctx,
        ),

        LogicalOp::CreateNodeType {
            name,
            temporal,
            properties,
        } => execute_create_node_type(name, *temporal, properties, ctx),

        LogicalOp::CreateTable {
            name,
            columns,
            primary_key,
            columnar,
        } => execute_create_table(name, columns, primary_key, *columnar, ctx),

        LogicalOp::DropTable { name } => execute_drop_table(name, ctx),

        LogicalOp::CreateTrigger { clause } => execute_create_trigger(clause, ctx),
        LogicalOp::DropTrigger { name } => execute_drop_trigger(name, ctx),
        LogicalOp::ShowTriggers => execute_show_triggers(ctx),
        LogicalOp::ShowSessions => execute_show_sessions(ctx),
        LogicalOp::ShowTransactions => execute_show_transactions(ctx),
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
                let elems = match eval_neutral(list, row) {
                    Ok(Value::Array(items)) => items,
                    // FOREACH over NULL is a no-op (matches Cypher).
                    Ok(Value::Null) => continue,
                    Ok(other) => {
                        ctx.foreach_scope = prev_scope;
                        return Err(ExecutionError::Unsupported(format!(
                            "FOREACH expects a list, got {other:?}"
                        )));
                    }
                    // Restore the enclosing scope on the way out, as the
                    // wrong-type arm above does: a bare `?` here would leave
                    // the loop variable's scope installed.
                    Err(e) => {
                        ctx.foreach_scope = prev_scope;
                        return Err(e.into());
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
/// Whether the node counter of `label` equals what a scan of the label
/// counts, by the catalog as committed now. A counter counts stored node
/// rows, and a temporal node keeps one row per version under every label it
/// carries (a temporal label in any position makes the node temporal), so no
/// counter stands for a scan while any temporal label exists. A columnar
/// label's rows live in its table, outside the node rows the counters follow.
///
/// # Errors
///
/// The catalog could not be read.
pub fn node_counter_answers_scan(
    engine: &coordinode_storage::engine::core::StorageEngine,
    label: &str,
) -> Result<bool, ExecutionError> {
    use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
    let labels = LocalSchemaStore::new(engine).list_labels()?;
    Ok(!labels
        .iter()
        .any(|schema| schema.temporal || (schema.name == label && schema.is_columnar())))
}

/// The number of nodes carrying `label` as this statement reads them, from
/// the label's counter at the statement's snapshot plus what its own
/// transaction staged, or `None` when the counter cannot stand for a scan:
/// a read at a past timestamp (the catalog of then is not the one asked),
/// a catalog [`node_counter_answers_scan`] refuses, or a counter below zero,
/// which no sequence of committed writes leaves and so is not trusted.
fn counted_nodes(
    label: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Option<i64>, ExecutionError> {
    if ctx.snapshot_ts.is_some() || !node_counter_answers_scan(ctx.engine, label)? {
        return Ok(None);
    }
    let key = coordinode_core::graph::stats::label_count_key(label);
    ctx.sync_txn_state();
    let stored = match ctx.txn.get(Partition::Counter, &key)? {
        Some(bytes) => coordinode_storage::engine::merge::decode_counter(&bytes).map_err(|e| {
            ExecutionError::Serialization(format!("node counter of label {label}: {e}"))
        })?,
        None => 0,
    };
    let count = stored.checked_add(ctx.txn.pending_counter_delta(&key));
    match count {
        Some(n) if n >= 0 => Ok(Some(n)),
        _ => {
            tracing::warn!(label, stored, "node counter out of range; counting by scan");
            Ok(None)
        }
    }
}

fn execute_node_scan(
    variable: &str,
    labels: &[String],
    property_filters: &[(String, crate::plan::expr::Expr)],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    if let Some(rows) =
        scan_by_table_key(variable, labels, property_filters, property_filters, ctx)?
    {
        return Ok(rows);
    }
    let mut results = Vec::new();

    // A COLUMNAR table's rows live in its own columnar tree, not the node path;
    // scan there at the read snapshot. The (key, value) shape matches the node
    // prefix scan, so the decode loop below is shared verbatim.
    let columnar_label = labels.first().and_then(|l| {
        ctx.load_current_label_schema(l)
            .ok()
            .flatten()
            .filter(|s| s.is_columnar())
            .map(|_| l.clone())
    });

    // Scan all nodes in the shard using prefix scan. The Layer-4 store owns
    // the node-key shape; the query layer just runs the tracked prefix scan.
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let prefix_bytes = LocalNodeStore.shard_scan_prefix(ctx.shard_id);
    ctx.sync_txn_state();
    // Keyset-paged source (server-side cursor) when `scan_paging` is set;
    // otherwise the whole-prefix scan. The decode loop below is identical.
    let scan_results = if let Some(label) = columnar_label {
        ctx.engine
            .columnar_scan(&label, ctx.mvcc_read_ts.as_raw())?
    } else {
        match ctx
            .scan_paging
            .as_ref()
            .map(|p| (p.resume.clone(), p.limit))
        {
            Some((resume, limit)) => {
                let mut page = LocalNodeStore.prefix_scan_paged_tracked(
                    &mut ctx.txn,
                    &prefix_bytes,
                    resume.as_deref(),
                    limit,
                )?;
                // A page that ends inside one temporal node's versions would
                // project a partial timeline: read that node's versions whole
                // and resume after the last of them.
                if !page.exhausted {
                    let cut = page
                        .rows
                        .last()
                        .and_then(|(key, _)| decode_temporal_node_key(key));
                    if let Some((shard, node_id, _)) = cut {
                        let all = LocalNodeStore.prefix_scan_tracked(
                            &mut ctx.txn,
                            &LocalNodeStore.version_prefix(shard, node_id),
                        )?;
                        page.rows.retain(|(key, _)| {
                            decode_temporal_node_key(key).is_none_or(|(_, id, _)| id != node_id)
                        });
                        page.last_key = all.last().map(|(key, _)| key.clone());
                        page.rows.extend(all);
                    }
                }
                if let Some(paging) = ctx.scan_paging.as_mut() {
                    paging.last_key = page.last_key;
                    paging.exhausted = page.exhausted;
                }
                page.rows
            }
            None => LocalNodeStore.prefix_scan_tracked(&mut ctx.txn, &prefix_bytes)?,
        }
    };

    // A temporal node's versions sort together under its id; they are
    // gathered and the node contributes the state valid at the instant this
    // variable reads at, or nothing.
    let at = ctx.instant_for(variable);
    let fields = ctx.timeline_fields();
    let mut versions: Vec<(i64, NodeRecord)> = Vec::new();
    let mut versions_of: Option<NodeId> = None;
    let decode = |value_bytes: &[u8]| {
        NodeRecord::from_msgpack(value_bytes)
            .map_err(|e| ExecutionError::Serialization(format!("node deserialization error: {e}")))
    };
    for (key_bytes, value_bytes) in &scan_results {
        let temporal = decode_temporal_node_key(key_bytes);
        // A record of another label is passed over by its labels alone,
        // read before the properties: most of a shard scan for one label is
        // other labels' records, whose properties need not be decoded.
        if temporal.is_none()
            && !labels.is_empty()
            && NodeRecord::labels_from_msgpack(value_bytes)
                .is_ok_and(|stored| !labels.iter().all(|l| stored.contains(l)))
        {
            continue;
        }
        let record = decode(value_bytes)?;
        if let Some((_, node_id, valid_from)) = temporal {
            if versions_of != Some(node_id) {
                if let Some(previous) = versions_of.replace(node_id) {
                    push_temporal_state(
                        &mut results,
                        previous,
                        std::mem::take(&mut versions),
                        (at, fields),
                        (variable, labels, property_filters),
                        ctx,
                    )?;
                }
            }
            versions.push((valid_from, record));
            continue;
        }
        if let Some(previous) = versions_of.take() {
            push_temporal_state(
                &mut results,
                previous,
                std::mem::take(&mut versions),
                (at, fields),
                (variable, labels, property_filters),
                ctx,
            )?;
        }
        let node_id = decode_node_id_from_key(key_bytes);
        if let Some(row) =
            node_row_if_matching(variable, labels, node_id, &record, property_filters, ctx)?
        {
            results.push(row);
        }
    }
    if let Some(previous) = versions_of {
        push_temporal_state(
            &mut results,
            previous,
            versions,
            (at, fields),
            (variable, labels, property_filters),
            ctx,
        )?;
    }

    Ok(results)
}

/// Add the row for one temporal node's gathered `versions`: its state valid
/// at the read instant, when that state is live and matches the pattern.
fn push_temporal_state(
    results: &mut Vec<Row>,
    node_id: NodeId,
    versions: Vec<(i64, NodeRecord)>,
    (at, fields): (i64, crate::executor::temporal_read::TimelineFields),
    (variable, labels, property_filters): (&str, &[String], &[(String, crate::plan::expr::Expr)]),
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    let Some((_, record)) =
        crate::executor::temporal_read::state_at(versions, at, fields).positive()
    else {
        return Ok(());
    };
    if let Some(row) = node_row_if_matching(
        variable,
        labels,
        node_id.as_raw(),
        &record,
        property_filters,
        ctx,
    )? {
        results.push(row);
    }
    Ok(())
}

/// Append to `out` the instant each `temporal_active_at(n, t)` conjunct of
/// `predicate` names for node variable `n`, when `t` is a value that does not
/// depend on the row. A disjunct, or an instant read from the row, names
/// nothing: the predicate then filters the rows projected at NOW.
fn named_instants(predicate: &crate::plan::expr::Expr, out: &mut Vec<(String, i64)>) {
    use crate::plan::expr::{BinOp, Expr};
    match predicate {
        Expr::Binary {
            left,
            op: BinOp::And,
            right,
        } => {
            named_instants(left, out);
            named_instants(right, out);
        }
        Expr::Call { name, args, .. } if name.eq_ignore_ascii_case("temporal_active_at") => {
            if let [Expr::Variable(v), at] = args.as_slice() {
                if let Ok(Value::Int(t) | Value::Timestamp(t)) = eval_neutral(at, &Row::new()) {
                    out.push((v.clone(), t));
                }
            }
        }
        _ => {}
    }
}

/// The `(valid_from, record)` versions among scanned node rows, in key order.
fn decode_versions(
    scanned: &[(Vec<u8>, Vec<u8>)],
) -> Result<Vec<(i64, NodeRecord)>, ExecutionError> {
    let mut out = Vec::with_capacity(scanned.len());
    for (key, bytes) in scanned {
        let Some((_, _, valid_from)) = decode_temporal_node_key(key) else {
            continue;
        };
        let record = NodeRecord::from_msgpack(bytes).map_err(|e| {
            ExecutionError::Serialization(format!("temporal node deserialization error: {e}"))
        })?;
        out.push((valid_from, record));
    }
    Ok(out)
}

/// Bind `var`'s label columns for `record` and return its primary label:
/// `var.__label__` always, and `var.__labels__` with every label when the
/// node has more than one (what `labels()` reads; a single-label node pays
/// no extra column).
fn insert_label_columns(row: &mut Row, var: &str, record: &NodeRecord) -> String {
    let primary = record.primary_label().to_string();
    row.insert(format!("{var}.__label__"), Value::String(primary.clone()));
    if record.labels.len() > 1 {
        row.insert(
            format!("{var}.__labels__"),
            Value::Array(record.labels.iter().cloned().map(Value::String).collect()),
        );
    }
    primary
}

/// [`insert_label_columns`] for a row that already binds `var` to a node
/// whose labels just changed: a list left from before is dropped first.
fn refresh_label_columns(row: &mut Row, var: &str, record: &NodeRecord) {
    row.remove(&format!("{var}.__labels__"));
    insert_label_columns(row, var, record);
}

/// Whether `record` carries every label of a pattern: `(n:A:B)` is a node
/// labelled both A and B. An empty pattern list matches any node.
fn has_all_labels(record: &NodeRecord, labels: &[String]) -> bool {
    labels.iter().all(|l| record.has_label(l))
}

/// The row a node scan produces for `record`, or `None` when the record
/// lacks one of `labels` or fails one of the inline `property_filters`.
fn node_row_if_matching(
    variable: &str,
    labels: &[String],
    node_id: u64,
    record: &NodeRecord,
    property_filters: &[(String, crate::plan::expr::Expr)],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Option<Row>, ExecutionError> {
    if !has_all_labels(record, labels) {
        return Ok(None);
    }

    let mut row = Row::new();
    row.insert(variable.to_string(), Value::Int(node_id as i64));
    for (field_id, value) in &record.props {
        if let Some(field_name) = ctx.interner.resolve(*field_id) {
            row.insert(format!("{variable}.{field_name}"), value.clone());
        }
    }
    // Overflow properties (undeclared properties of a VALIDATED label).
    if let Some(extra) = &record.extra {
        for (name, value) in extra {
            row.insert(format!("{variable}.{name}"), value.clone());
        }
    }
    let primary_label = insert_label_columns(&mut row, variable, record);
    inject_computed_properties(&mut row, variable, &primary_label, ctx)?;

    // Inline property filters. Inside a correlated join (e.g. `UNWIND ... AS
    // e MATCH (a {p: e.x})`) a filter value can reference outer bindings, so
    // it is evaluated against the correlated row extended with this node's:
    // an inline `{p: e.x}` means the same as `WHERE a.p = e.x`.
    for (prop_name, filter_expr) in property_filters {
        let actual = row
            .get(&format!("{variable}.{prop_name}"))
            .cloned()
            .unwrap_or(Value::Null);
        let expected = match &ctx.correlated_row {
            Some(corr) => {
                let mut eval_row = corr.clone();
                eval_row.extend(row.clone());
                eval_neutral(filter_expr, &eval_row)?
            }
            None => eval_neutral(filter_expr, &row)?,
        };
        if actual != expected {
            return Ok(None);
        }
    }
    Ok(Some(row))
}

/// Serve a scan of one row-stored table whose key columns `key_filters` all
/// pin to values independent of the row: the key index names the one row that
/// can match, so one read replaces a scan of the shard. `property_filters`
/// still apply to that row. `None` when the scan is not such a lookup, and
/// then the caller scans.
///
/// A value whose type differs from its key column's declared type is left to
/// the scan, which compares as the query language does; the key index only
/// holds values of the declared types.
fn scan_by_table_key(
    variable: &str,
    labels: &[String],
    key_filters: &[(String, crate::plan::expr::Expr)],
    property_filters: &[(String, crate::plan::expr::Expr)],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Option<Vec<Row>>, ExecutionError> {
    use coordinode_core::schema::definition::TableKey;
    use coordinode_modality::{LocalTableKeyStore, StoreError, TableKeyStore as _};

    // A server-side cursor pages through the scan's own key order.
    if ctx.scan_paging.is_some() {
        return Ok(None);
    }
    let [label] = labels else {
        return Ok(None);
    };
    let Some(schema) = ctx.load_current_label_schema(label)? else {
        return Ok(None);
    };
    if schema.is_columnar() || schema.temporal {
        return Ok(None);
    }
    let Some(TableKey::Columns(columns)) = schema.table_key() else {
        return Ok(None);
    };
    let mut key = Vec::with_capacity(columns.len());
    for column in columns {
        let Some((_, expr)) = key_filters.iter().find(|(p, _)| p == column) else {
            return Ok(None);
        };
        if crate::planner::builder::expr_references_var(expr, variable) {
            return Ok(None);
        }
        let value = match &ctx.correlated_row {
            Some(corr) => eval_neutral(expr, corr)?,
            None => eval_neutral(expr, &Row::new())?,
        };
        let declared = schema.get_property(column);
        if declared.is_none_or(|def| validate_one(column, &value, def).is_err()) {
            return Ok(None);
        }
        key.push(value);
    }

    ctx.sync_txn_state();
    let holder = match LocalTableKeyStore.lookup(&mut ctx.txn, label, &key) {
        Ok(holder) => holder,
        Err(StoreError::UnsupportedKey(_)) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let Some(node_id) = holder else {
        return Ok(Some(Vec::new()));
    };
    let Some(record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? else {
        return Ok(Some(Vec::new()));
    };
    Ok(Some(
        node_row_if_matching(
            variable,
            labels,
            node_id.as_raw(),
            &record,
            property_filters,
            ctx,
        )?
        .into_iter()
        .collect(),
    ))
}

/// The table and key columns of the node `variable` binds in `row`, when it
/// is a row of a table with declared key columns.
fn bound_table_key(
    row: &Row,
    variable: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Option<(String, Vec<String>)>, ExecutionError> {
    use coordinode_core::schema::definition::TableKey;
    if !matches!(row.get(variable), Some(Value::Int(_))) {
        return Ok(None);
    }
    let Some(Value::String(label)) = row.get(&format!("{variable}.__label__")) else {
        return Ok(None);
    };
    let label = label.clone();
    let Some(schema) = ctx.load_current_label_schema(&label)? else {
        return Ok(None);
    };
    Ok(match schema.table_key() {
        Some(TableKey::Columns(columns)) => Some((label, columns.to_vec())),
        _ => None,
    })
}

/// Refuse a table label given to, or taken from, an existing node: a node
/// becomes a table row only by being inserted into the table.
fn refuse_table_label_change(
    label: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    if ctx
        .load_current_label_schema(label)?
        .is_some_and(|s| s.is_table())
    {
        return Err(ExecutionError::Unsupported(format!(
            "`{label}` is a table: a row joins or leaves it only by insert or delete, \
             not by adding or removing the label"
        )));
    }
    Ok(())
}

/// Refuse SET items that would change the key of a table row bound in `row`.
/// A row's key is its identity and does not change.
fn refuse_key_changes_in_set(
    row: &Row,
    items: &[crate::plan::SetItem],
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use crate::plan::SetItem;
    for item in items {
        let (variable, touched): (&str, Option<&str>) = match item {
            SetItem::Property {
                variable, property, ..
            } => (variable, Some(property)),
            SetItem::PropertyPath { variable, path, .. }
            | SetItem::DocFunction { variable, path, .. } => {
                (variable, path.first().map(String::as_str))
            }
            SetItem::ReplaceProperties { variable, .. }
            | SetItem::MergeProperties { variable, .. } => (variable, None),
            SetItem::AddLabel { label, .. } => {
                refuse_table_label_change(label, ctx)?;
                continue;
            }
        };
        let Some((table, columns)) = bound_table_key(row, variable, ctx)? else {
            continue;
        };
        let changed = match (item, touched) {
            (_, Some(column)) => columns.iter().find(|c| c.as_str() == column),
            // Replacing every property replaces the key with them.
            (SetItem::ReplaceProperties { .. }, None) => columns.first(),
            (SetItem::MergeProperties { expr, .. }, None) => match eval_neutral(expr, row)? {
                Value::Map(map) => columns.iter().find(|c| map.contains_key(c.as_str())),
                Value::Document(rmpv::Value::Map(entries)) => columns
                    .iter()
                    .find(|c| entries.iter().any(|(k, _)| k.as_str() == Some(c.as_str()))),
                _ => None,
            },
            _ => None,
        };
        if let Some(column) = changed {
            return Err(ExecutionError::KeyImmutable {
                table,
                column: column.clone(),
            });
        }
    }
    Ok(())
}

/// Refuse REMOVE items that would take the key from a table row bound in
/// `row`, or take a table label from a node.
fn refuse_key_changes_in_remove(
    row: &Row,
    items: &[crate::plan::RemoveItem],
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use crate::plan::RemoveItem;
    for item in items {
        let (variable, column) = match item {
            RemoveItem::Property { variable, property } => (variable, property.as_str()),
            RemoveItem::PropertyPath { variable, path } => match path.first() {
                Some(first) => (variable, first.as_str()),
                None => continue,
            },
            RemoveItem::Label { label, .. } => {
                refuse_table_label_change(label, ctx)?;
                continue;
            }
        };
        if let Some((table, columns)) = bound_table_key(row, variable, ctx)? {
            if columns.iter().any(|c| c == column) {
                return Err(ExecutionError::KeyImmutable {
                    table,
                    column: column.to_string(),
                });
            }
        }
    }
    Ok(())
}

/// The rows a Filter's `input` gives for `predicate` to filter. A node scan
/// that the predicate pins to one node id (`n = $id`, `id(n) = $id`) reads
/// that node alone; one pinned to a keyed table row reads the row its key
/// index names. Either way the full predicate is still applied by the caller.
fn execute_filter_input(
    input: &LogicalOp,
    predicate: &crate::plan::expr::Expr,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    if let LogicalOp::NodeScan {
        variable,
        labels,
        property_filters,
    } = input
    {
        // A server-side cursor pages through the scan's own key order, and a
        // COLUMNAR table keeps its rows outside the node records.
        if ctx.scan_paging.is_none() {
            if let Some(id) = node_id_filter(predicate, variable) {
                let columnar = match labels.first() {
                    Some(label) => ctx
                        .load_current_label_schema(label)?
                        .is_some_and(|s| s.is_columnar()),
                    None => false,
                };
                if !columnar {
                    return scan_by_node_id(variable, labels, id, property_filters, ctx);
                }
            }
        }
        let mut key_filters = property_filters.clone();
        equality_filters(predicate, variable, &mut key_filters);
        if let Some(rows) =
            scan_by_table_key(variable, labels, &key_filters, property_filters, ctx)?
        {
            return Ok(rows);
        }
    }
    execute_op(input, ctx)
}

/// The node id an `n = v` or `id(n) = v` conjunct of `predicate` pins node
/// variable `n` to, when `v` is an integer that does not depend on the row.
fn node_id_filter(predicate: &crate::plan::expr::Expr, variable: &str) -> Option<i64> {
    use crate::plan::expr::{BinOp, Expr};
    let Expr::Binary { left, op, right } = predicate else {
        return None;
    };
    match op {
        BinOp::And => node_id_filter(left, variable).or_else(|| node_id_filter(right, variable)),
        BinOp::Eq => [(left, right), (right, left)]
            .into_iter()
            .find_map(|(side, value)| {
                let names_node = match side.as_ref() {
                    Expr::Variable(v) => v == variable,
                    Expr::Call { name, args, .. } if name.eq_ignore_ascii_case("id") => {
                        matches!(args.as_slice(), [Expr::Variable(v)] if v == variable)
                    }
                    _ => false,
                };
                if !names_node {
                    return None;
                }
                match eval_neutral(value, &Row::new()).ok()? {
                    Value::Int(id) => Some(id),
                    _ => None,
                }
            }),
        _ => None,
    }
}

/// Serve a node scan pinned to one node id with a point read: the node's
/// record, or for a temporal node its state at the instant `variable` reads
/// at, when it carries `labels` and passes `property_filters`.
fn scan_by_node_id(
    variable: &str,
    labels: &[String],
    id: i64,
    property_filters: &[(String, crate::plan::expr::Expr)],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // A negative value names no node.
    let Ok(raw) = u64::try_from(id) else {
        return Ok(Vec::new());
    };
    let node_id = NodeId::from_raw(raw);
    let record = match ctx.mvcc_get_node(ctx.shard_id, node_id)? {
        Some(record) => Some(record),
        None => {
            let at = ctx.instant_for(variable);
            ctx.temporal_node_state(node_id, at)?
                .positive()
                .map(|(_, record)| record)
        }
    };
    let Some(record) = record else {
        return Ok(Vec::new());
    };
    Ok(
        node_row_if_matching(variable, labels, raw, &record, property_filters, ctx)?
            .into_iter()
            .collect(),
    )
}

/// The `variable.column = value` conjuncts of `predicate`, as scan filters.
fn equality_filters(
    predicate: &crate::plan::expr::Expr,
    variable: &str,
    out: &mut Vec<(String, crate::plan::expr::Expr)>,
) {
    use crate::plan::expr::{BinOp, Expr};
    let Expr::Binary { left, op, right } = predicate else {
        return;
    };
    match op {
        BinOp::And => {
            equality_filters(left, variable, out);
            equality_filters(right, variable, out);
        }
        BinOp::Eq => {
            for (side, value) in [(left, right), (right, left)] {
                if let Some(column) =
                    crate::planner::builder::extract_index_property(side, variable)
                {
                    out.push((column, (**value).clone()));
                }
            }
        }
        _ => {}
    }
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
    index: crate::index::IndexId,
    property: &str,
    value_expr: &crate::plan::expr::Expr,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let temporal = ctx
        .load_current_label_schema(label)?
        .is_some_and(|s| s.temporal);

    // Evaluate the lookup value. A correlated key (e.g. `WHERE a.pid = e.s`
    // driven per outer row) resolves against `correlated_row`; a literal /
    // parameter key ignores the row, so the empty-row fallback is equivalent.
    let lookup_val = match &ctx.correlated_row {
        Some(corr) => eval_neutral(value_expr, corr)?,
        None => eval_neutral(value_expr, &Row::new())?,
    };

    // An equality with NULL is never true.
    if lookup_val.is_null() {
        return Ok(Vec::new());
    }

    // An index that cannot answer (not built, its entries found wrong, or a
    // value with no key) leaves the equality to a scan of the label's
    // records, which applies the query's own comparison at this statement's
    // snapshot with its own writes.
    let scan_records = |ctx: &mut ExecutionContext<'_>, lookup_val: Value| {
        execute_node_scan(
            variable,
            &[label.to_string()],
            &[(
                property.to_string(),
                crate::plan::expr::Expr::Literal(lookup_val),
            )],
            ctx,
        )
    };

    // The entries as this statement sees them: its snapshot, its own writes.
    let Some(ids) = ctx.index_lookup(index, &lookup_val)? else {
        return scan_records(ctx, lookup_val);
    };

    // A temporal node's entries are the union of the values its versions
    // ever held, so a candidate resolves to its state valid at the instant
    // this variable reads at; one with no live state there is no match.
    let records: Vec<Option<NodeRecord>> = if temporal {
        let at = ctx.instant_for(variable);
        let mut states = Vec::with_capacity(ids.len());
        for id in &ids {
            states.push(
                ctx.temporal_node_state(*id, at)?
                    .positive()
                    .map(|(_, record)| record),
            );
        }
        states
    } else {
        use coordinode_modality::NodeStore as _;
        // One batched multi_get (single version snapshot + batched bloom/SST
        // traversal) rather than a per-id lookup loop.
        coordinode_modality::LocalNodeStore.get_many(&ctx.txn, ctx.shard_id, &ids)?
    };

    let mut results = Vec::with_capacity(ids.len());
    let labels = [label.to_string()];
    for (id, record_opt) in ids.into_iter().zip(records) {
        // The index finds a list by each of its elements, and a temporal
        // node by any value it ever held; the equality the query asked holds
        // only for the value the record carries itself. A candidate the
        // index had no reason to hold (no node, or a node whose own entries
        // do not include the value) shows the index wrong: its other
        // entries prove nothing either, and the records answer instead.
        let held = record_opt.as_ref().and_then(|record| {
            crate::index::registry::record_lookup(record, ctx.interner)(property)
        });
        if held.as_ref() != Some(&lookup_val) {
            if ctx.index_entry_disagrees(index, &lookup_val, id, record_opt.as_ref())? {
                return scan_records(ctx, lookup_val);
            }
            continue;
        }
        let Some(record) = record_opt else {
            continue;
        };
        // The label check guards against entries of relabeled nodes.
        if let Some(row) = node_row_if_matching(variable, &labels, id.as_raw(), &record, &[], ctx)?
        {
            results.push(row);
        }
    }

    Ok(results)
}

/// The nodes a read through an index of the current state answers itself,
/// beyond `pending` (what the index's worker has not folded): on a read at a
/// named timestamp T, every node written after T, whose state then the index
/// no longer holds; on a current read, the transaction's own uncommitted
/// writes. Each is evaluated exactly from the store as the statement reads it.
fn read_delta(
    pending: IndexDelta,
    ctx: &ExecutionContext<'_>,
) -> Result<IndexDelta, ExecutionError> {
    match ctx.snapshot_ts {
        Some(at) => {
            // A read at T sees the commits at or below T: the nodes changed
            // since are those written from seqno T + 1 on.
            let since = u64::try_from(at)
                .ok()
                .and_then(|at| at.checked_add(1))
                .ok_or_else(|| {
                    ExecutionError::Unsupported("AS OF TIMESTAMP value out of range".into())
                })?;
            Ok(pending.union(IndexDelta::written_since(ctx.engine, ctx.shard_id, since)?))
        }
        None => Ok(pending.with_nodes(own_written_nodes(ctx))),
    }
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
    query_vector: &crate::plan::expr::Expr,
    k: usize,
    function: &str,
    distance_alias: Option<&str>,
    index_name: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let Some(indexes) = ctx.vector_indexes else {
        return Err(ExecutionError::Unsupported(format!(
            "HnswScan({index_name}) requires vector indexes in ExecutionContext"
        )));
    };
    let registry = indexes.registry;
    // Honour the online-during-build policy exactly like the
    // scan-then-rank path does.
    gate_vector_index_read(indexes, label, property)?;

    let qv_val = eval_neutral(query_vector, &Row::new())?;
    let Some(qv) = coerce_value_to_vec(&qv_val) else {
        return Err(ExecutionError::Unsupported(format!(
            "HnswScan({index_name}): query vector did not evaluate to a vector"
        )));
    };

    use coordinode_modality::NodeStore as _;
    let nodes = coordinode_modality::LocalNodeStore;

    // The commits the index has not folded yet, and the transaction's own
    // writes or (at a named timestamp) the nodes written since, are answered
    // from the store as this statement reads it; the index answers for every
    // other node.
    materialize_own_node_writes(ctx)?;
    let delta = read_delta(registry.delta(ctx.shard_id), ctx)?;
    // (node, record, index score, exact score of the ORDER BY function)
    let mut candidates: Vec<(u64, NodeRecord, f32, Option<f64>)> = Vec::new();
    if delta != IndexDelta::Unknown {
        let superseded = match &delta {
            IndexDelta::Nodes(nodes) => nodes.len(),
            IndexDelta::Unknown => 0,
        };
        // A deleted node stays in the graph until a rebuild and a hit of a
        // node the delta names is answered from the store, so some hits do
        // not count. Ask for more until k of them do or the index runs out,
        // instead of answering LIMIT k with fewer rows than exist.
        let mut want = k.checked_add(superseded).ok_or_else(|| {
            ExecutionError::Unsupported(format!("HnswScan({index_name}): k={k} is too large"))
        })?;
        loop {
            let Some(hits) = registry.search(label, property, &qv, want) else {
                // Index disappeared between planning and execution (concurrent
                // DROP). Empty result keeps the read path total; the planner
                // will not pick HnswScan on the next statement.
                return Ok(Vec::new());
            };
            let kept: Vec<_> = hits
                .iter()
                .filter(|h| !delta.contains(NodeId::from_raw(h.id)))
                .collect();
            // Hydrate the kept hits in one batched multi_get; the input order
            // is preserved, so the rows stay in similarity order.
            let ids: Vec<NodeId> = kept.iter().map(|h| NodeId::from_raw(h.id)).collect();
            let records = nodes.get_many(&ctx.txn, ctx.shard_id, &ids)?;
            let found: Vec<_> = kept
                .iter()
                .zip(records)
                .filter_map(|(hit, record)| {
                    let record = record?;
                    record
                        .has_label(label)
                        .then_some((hit.id, record, hit.score, None))
                })
                .collect();
            if found.len() >= k || hits.len() < want {
                candidates = found;
                break;
            }
            want = want.checked_mul(2).ok_or_else(|| {
                ExecutionError::Unsupported(format!("HnswScan({index_name}): k={k} is too large"))
            })?;
        }
    }
    if !delta.is_empty() {
        // One scale for the index's hits and the written nodes: the function
        // the query orders by, computed exactly on each vector.
        let field = ctx.interner.lookup(property);
        candidates.extend(
            written_vector_nodes(&delta, label, field, qv.len(), ctx)?
                .into_iter()
                .map(|(id, record)| (id, record, f32::NAN, None)),
        );
        let mut scored: Vec<_> = candidates
            .into_iter()
            .filter_map(|(id, record, index_score, _)| {
                let vector = record_vector(&record, field?)?;
                let exact = vector_function_score(function, &qv, &vector)?;
                Some((id, record, index_score, Some(exact)))
            })
            .collect();
        let ascending = vector_function_ascending(function);
        scored.sort_by(|a, b| {
            let (x, y) = (a.3.unwrap_or(f64::NAN), b.3.unwrap_or(f64::NAN));
            let order = if ascending {
                x.total_cmp(&y)
            } else {
                y.total_cmp(&x)
            };
            order.then(a.0.cmp(&b.0))
        });
        candidates = scored;
    }
    candidates.truncate(k);
    let mut results = Vec::with_capacity(candidates.len());

    for (id, record, index_score, exact) in candidates {
        let mut row = Row::new();
        row.insert(binding.to_string(), Value::Int(id as i64));
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
        let primary_label = insert_label_columns(&mut row, binding, &record);
        inject_computed_properties(&mut row, binding, &primary_label, ctx)?;

        if let Some(alias) = distance_alias {
            let score = exact.or_else(|| {
                let node_vec = row
                    .get(&format!("{binding}.{property}"))
                    .and_then(coerce_value_to_vec)?;
                vector_function_score(function, &qv, &node_vec)
            });
            row.insert(
                alias.to_string(),
                Value::Float(score.unwrap_or(index_score as f64)),
            );
        }
        results.push(row);
    }
    Ok(results)
}

/// Execute the `TextIndexScan` access path: every node the text index of
/// `(label, property)` matches for `query`, as this statement reads the store
/// (the same view `TextFilter` uses), fetched in one batch and shaped as the
/// `NodeScan` rows it replaces, in node-id order, each carrying its score for
/// `text_score()`.
fn execute_text_index_scan(
    label: &str,
    property: &str,
    binding: &str,
    query: &str,
    language: Option<&str>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_modality::NodeStore as _;

    let Some(registry) = ctx.text_index_registry else {
        return Err(text_match_missing_index_error(Some(label), Some(property)));
    };
    materialize_own_node_writes(ctx)?;
    let matches = text_index_matches(registry, (binding, label, property), query, language, ctx)
        .map_err(|e| ExecutionError::Unsupported(format!("text search error: {e}")))?
        // Dropped between planning and execution: refused as TextFilter does.
        .ok_or_else(|| text_match_missing_index_error(Some(label), Some(property)))?;

    let mut ids: Vec<NodeId> = matches.keys().map(|id| NodeId::from_raw(*id)).collect();
    ids.sort_unstable();
    let records = coordinode_modality::LocalNodeStore.get_many(&ctx.txn, ctx.shard_id, &ids)?;
    let at = ctx.instant_for(binding);
    let mut rows = Vec::with_capacity(ids.len());
    for (id, record) in ids.into_iter().zip(records) {
        // A temporal node has no row of its own: its state at the instant
        // the binding reads at, as the label scan projects it.
        let record = match record {
            Some(record) => Some(record),
            None => ctx
                .temporal_node_state(id, at)?
                .positive()
                .map(|(_, record)| record),
        };
        let Some(record) = record.filter(|r| r.has_label(label)) else {
            continue;
        };
        let mut row = Row::new();
        row.insert(binding.to_string(), Value::Int(id.as_raw() as i64));
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
        let primary_label = insert_label_columns(&mut row, binding, &record);
        inject_computed_properties(&mut row, binding, &primary_label, ctx)?;
        let score = matches.get(&id.as_raw()).copied().unwrap_or_default();
        row.insert("__text_score__".to_string(), Value::Float(f64::from(score)));
        rows.push(row);
    }
    Ok(rows)
}

/// The scalar a vector ORDER BY function computes for `a` against `b`;
/// `None` for an unknown function or vectors of different widths.
fn vector_function_score(function: &str, a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    Some(match function {
        "vector_distance" => coordinode_vector::metrics::euclidean_distance(a, b) as f64,
        "vector_similarity" => coordinode_vector::metrics::cosine_similarity(a, b) as f64,
        "vector_dot" => coordinode_vector::metrics::dot_product(a, b) as f64,
        "vector_manhattan" => coordinode_vector::metrics::manhattan_distance(a, b) as f64,
        _ => return None,
    })
}

/// Whether a lower value of `function` ranks first (distances) rather than a
/// higher one (similarities).
fn vector_function_ascending(function: &str) -> bool {
    matches!(function, "vector_distance" | "vector_manhattan")
}

/// The vector `record` holds in `field`.
fn record_vector(record: &NodeRecord, field: u32) -> Option<Vec<f32>> {
    coerce_value_to_vec(record.props.get(&field)?)
}

/// The nodes `delta` names whose primary label is `label` (what a vector
/// index of `label` holds) and that carry a `dims`-wide vector in `field`,
/// as this statement reads them; every such node of the shard when the delta
/// is unknown.
fn written_vector_nodes(
    delta: &IndexDelta,
    label: &str,
    field: Option<u32>,
    dims: usize,
    ctx: &ExecutionContext<'_>,
) -> Result<Vec<(u64, NodeRecord)>, ExecutionError> {
    use coordinode_modality::NodeStore as _;
    let Some(field) = field else {
        // No node has ever carried the property.
        return Ok(Vec::new());
    };
    let indexed = |record: &NodeRecord| {
        record.primary_label() == label
            && record_vector(record, field).is_some_and(|v| v.len() == dims)
    };
    let nodes = coordinode_modality::LocalNodeStore;
    let mut written = Vec::new();
    match delta {
        IndexDelta::Nodes(ids) => {
            let mut ids: Vec<NodeId> = ids.iter().copied().collect();
            ids.sort_unstable();
            let records = nodes.get_many(&ctx.txn, ctx.shard_id, &ids)?;
            for (id, record) in ids.into_iter().zip(records) {
                if let Some(record) = record.filter(|r| indexed(r)) {
                    written.push((id.as_raw(), record));
                }
            }
        }
        IndexDelta::Unknown => {
            nodes.for_each_in_shard(&ctx.txn, ctx.shard_id, &mut |id, record| {
                if indexed(&record) {
                    written.push((id.as_raw(), record));
                }
                Ok(())
            })?;
        }
    }
    Ok(written)
}

/// Edge binding a traversal uses for an anonymous relationship that carries an
/// inline property map. The leading NUL keeps it out of reach of any Cypher
/// identifier, so it never shadows or correlates with a query variable.
const ANONYMOUS_EDGE_BINDING: &str = "\u{0}edge";

/// Parameters for edge traversal.
struct TraverseParams<'a> {
    source: &'a str,
    edge_types: &'a [String],
    direction: Direction,
    target_variable: &'a str,
    target_labels: &'a [String],
    length: Option<LengthBound>,
    edge_variable: Option<&'a str>,
    target_filters: &'a [(String, crate::plan::expr::Expr)],
    /// Inline edge property filters from pattern (e.g., `[r:TYPE {prop: val}]`).
    edge_filters: &'a [(String, crate::plan::expr::Expr)],
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
    // Traversal into a temporal target lands on the target's state valid at
    // the instant its variable reads at (the statement's NOW unless a
    // `temporal_active_at` conjunct names one), as a label-scoped MATCH
    // does; a target with no live state there gives no row. Detected per
    // target at row-build time in `build_target_rows`.

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

/// The state of temporal node `target_id` at the instant `variable` reads at,
/// with its `valid_from`; empty when the node has no live state then.
fn temporal_target_state(
    ctx: &mut ExecutionContext<'_>,
    target_id: NodeId,
    variable: &str,
) -> Result<Vec<(NodeRecord, Option<i64>)>, ExecutionError> {
    let at = ctx.instant_for(variable);
    Ok(ctx
        .temporal_node_state(target_id, at)?
        .positive()
        .map(|(valid_from, record)| (record, Some(valid_from)))
        .into_iter()
        .collect())
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

    // A temporal target is read as its timeline's state at the instant the
    // target variable reads at; any other target by a point read. Label
    // schemas do not change within one query, so the per-row check is cheap.
    let target_is_temporal = target_labels.iter().any(|lbl| {
        ctx.load_current_label_schema(lbl)
            .ok()
            .flatten()
            .is_some_and(|s| s.temporal)
    });

    // At most one (record, valid_from) pair: the target's state, if any.
    let target_records: Vec<(NodeRecord, Option<i64>)> = if target_is_temporal {
        temporal_target_state(ctx, target_id, target_variable)?
    } else {
        match ctx.mvcc_get_node(ctx.shard_id, target_id)? {
            Some(rec) => vec![(rec, None)],
            // No plain row: the target may be a temporal node the pattern
            // does not name by label, which lives under per-version keys
            // only. It is read the same way a labelled one is; a dangling
            // edge finds no version either.
            None => temporal_target_state(ctx, target_id, target_variable)?,
        }
    };
    if target_records.is_empty() {
        return Ok(Vec::new());
    }

    // For each version (or the single non-temporal record), build a
    // base out_row carrying the target's properties + version axes,
    // then apply the edge-property fan-out + filter pipeline below.
    let mut materialised_rows: Vec<Row> = Vec::with_capacity(target_records.len());
    for (target_record, valid_from_opt) in target_records {
        if !has_all_labels(&target_record, target_labels) {
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

        let target_label = insert_label_columns(&mut out_row, target_variable, &target_record);

        // Re-surface valid_from from the key suffix so callers
        // always see a non-null binding even if the stored property map
        // happens to omit it (defensive — write path requires it).
        if let Some(vf) = valid_from_opt {
            out_row.insert(format!("{target_variable}.valid_from"), Value::Int(vf));
        }

        // Inject COMPUTED property values from schema.
        inject_computed_properties(&mut out_row, target_variable, &target_label, ctx)?;

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
            let expected = eval_neutral(filter_expr, &row)?;
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
                let expected = eval_neutral(filter_expr, &row)?;
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
    /// Same bound evaluation instant as the sequential execution path.
    evaluation_time: i64,
    /// Unique schema dependencies discovered by chunk-local caches. Merged
    /// into the owning transaction once, rather than locking for each row.
    schema_reads: parking_lot::Mutex<HashMap<String, u64>>,
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
///
/// A target with no plain record is pushed to `unresolved` rather than
/// dropped: it may be a temporal node the pattern does not label, whose
/// versions only the sequential path reads.
fn process_targets_parallel(
    neighbors: &[(u64, u64, usize)],
    input_row: &Row,
    params: &TraverseParams<'_>,
    pctx: &ParallelCtx<'_>,
    unresolved: &Mutex<Vec<(u64, u64, usize)>>,
) -> Result<Vec<Row>, ExecutionError> {
    let target_variable = params.target_variable;
    let target_labels = params.target_labels;
    let target_filters = params.target_filters;
    let edge_variable = params.edge_variable;
    let direction = params.direction;
    let source = params.source;

    neighbors
        .par_chunks(pctx.chunk_size.max(1))
        .flat_map_iter(|chunk| {
            let mut schemas = HashMap::<String, Option<LabelSchema>>::new();
            let rows: Vec<_> = chunk
                .iter()
                .filter_map(|(src_uid, tgt_uid, et_idx)| {
                    let target_id = NodeId::from_raw(*tgt_uid);

                    // Stateless snapshot read through the Layer-4 store: the
                    // snapshot seqno (a Copy value) is the shared unit across
                    // rayon workers, so no &mut Transaction crosses threads. The
                    // store owns key encoding and hands the key back for OCC.
                    use coordinode_modality::{LocalNodeStore, NodeStore as _};
                    let (target_key, target_record) = LocalNodeStore
                        .read_record_at_snapshot(
                            pctx.engine,
                            pctx.mvcc_snapshot,
                            pctx.shard_id,
                            target_id,
                        )
                        .ok()?;
                    let Some(mut target_record) = target_record else {
                        if let Ok(mut guard) = unresolved.lock() {
                            guard.push((*src_uid, *tgt_uid, *et_idx));
                        }
                        return None;
                    };

                    // Track the node key in the OCC read-set: each worker records
                    // into the shared accumulator; merged into the statement's
                    // read-set after the parallel section.
                    if let Some(ref keys) = pctx.occ_read_keys {
                        if let Ok(mut guard) = keys.node.lock() {
                            guard.push(target_key);
                        }
                    }

                    if !has_all_labels(&target_record, target_labels) {
                        return None;
                    }

                    let mut out_row = input_row.clone();
                    // Update source in row so a deeper hop looks its edge
                    // properties up from the actual source, not the start node.
                    out_row.insert(source.to_string(), Value::Int(*src_uid as i64));
                    out_row.insert(target_variable.to_string(), Value::Int(*tgt_uid as i64));

                    // Resolve property names from interner (read-only, thread-safe)
                    // Move the decoded payloads; draining retains the record's
                    // labels so metadata keeps its original precedence below.
                    for (field_id, value) in target_record.props.drain() {
                        if let Some(field_name) = pctx.interner.resolve(field_id) {
                            let col_name = format!("{target_variable}.{field_name}");
                            out_row.insert(col_name, value);
                        }
                    }
                    if let Some(extra) = target_record.extra.take() {
                        for (name, value) in extra {
                            let col_name = format!("{target_variable}.{name}");
                            out_row.insert(col_name, value);
                        }
                    }

                    let target_label =
                        insert_label_columns(&mut out_row, target_variable, &target_record);

                    if !schemas.contains_key(&target_label) {
                        let Some(snapshot) = pctx.mvcc_snapshot else {
                            return Some(Err(ExecutionError::Unsupported(
                                "parallel schema evaluation requires a bound snapshot".into(),
                            )));
                        };
                        let schema = match coordinode_modality::LocalSchemaStore::new(pctx.engine)
                            .load_label_at(snapshot, &target_label)
                        {
                            Ok(schema) => schema,
                            Err(error) => return Some(Err(error.into())),
                        };
                        schemas.insert(target_label.clone(), schema);
                    }
                    if let Some(Some(schema)) = schemas.get(&target_label) {
                        inject_computed_from_schema(
                            &mut out_row,
                            target_variable,
                            schema,
                            pctx.evaluation_time,
                        );
                    }

                    // Edge variable: type + edge properties
                    let edge_type = params.edge_types.get(*et_idx).map(|s| s.as_str());
                    if let Some(ev) = edge_variable {
                        if let Some(et) = edge_type {
                            out_row.insert(format!("{ev}.__type__"), Value::String(et.to_string()));
                            // Store the relationship variable itself for count(r) support
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
                                            if let Some(field_name) =
                                                pctx.interner.resolve(field_id)
                                            {
                                                out_row.insert(format!("{ev}.{field_name}"), value);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Apply target inline property filters. A failed evaluation is
                    // yielded as an item rather than swallowed, so the collect
                    // below turns it into the whole call's error; `None` keeps its
                    // one meaning, "this row does not match".
                    for (prop_name, filter_expr) in target_filters {
                        let actual = out_row
                            .get(&format!("{target_variable}.{prop_name}"))
                            .cloned()
                            .unwrap_or(Value::Null);
                        let expected = match eval_neutral(filter_expr, &out_row) {
                            Ok(v) => v,
                            Err(e) => return Some(Err(e.into())),
                        };
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
                            let expected = match eval_neutral(filter_expr, &out_row) {
                                Ok(v) => v,
                                Err(e) => return Some(Err(e.into())),
                            };
                            if actual != expected {
                                return None;
                            }
                        }
                    }

                    Some(Ok(out_row))
                })
                .collect();
            // One merge per chunk, including an absent schema's revision 0.
            // A read-only attempt has no validation cost at commit; a writer
            // must retain exactly the interpretation these rows used.
            let mut reads = pctx.schema_reads.lock();
            for (label, schema) in schemas {
                reads.insert(label, schema.map_or(0, |schema| schema.schema_revision));
            }
            rows
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
        // Temporal target labels need per-version prefix scans
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
            && !ctx.txn.has_schema_changes()
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
                // The engine-only executor also binds a statement adjacency
                // snapshot at entry. Use that same view for target records
                // and schema; never choose a fresh view inside a worker.
                mvcc_snapshot: ctx.mvcc_snapshot.or(ctx.txn.adj_snapshot()),
                chunk_size: ctx.adaptive.parallel_chunk_size,
                evaluation_time: ctx.valid_now,
                schema_reads: parking_lot::Mutex::new(HashMap::new()),
                // Read keys are not collected: the default conflict level
                // validates writes only, so commit never consults a read set.
                // FOR UPDATE / the serializable level re-enable collection
                // selectively when they land; until then filling these vectors
                // was per-row work feeding a set nobody read.
                occ_read_keys: None,
            };
            let src_raw = source_id.as_raw();
            let with_src: Vec<(u64, u64, usize)> = neighbors
                .iter()
                .map(|&(tgt, et_idx)| (src_raw, tgt, et_idx))
                .collect();
            let unresolved = Mutex::new(Vec::new());
            let parallel_rows =
                process_targets_parallel(&with_src, row, params, &pctx, &unresolved)?;
            for (label, revision) in pctx.schema_reads.lock().iter() {
                ctx.txn.note_label_schema_read(label, *revision);
            }
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
            // Targets with no plain record (temporal nodes the pattern does
            // not label, or dangling edges) take the sequential path, which
            // reads their versions.
            for (_, target_uid, et_idx) in unresolved.into_inner().unwrap_or_default() {
                let trp = TargetRowParams {
                    input_row: row,
                    target_uid,
                    edge_type: params.edge_types.get(et_idx).map(|s| s.as_str()),
                    edge_is_temporal: params.edge_temporal.get(et_idx).copied().unwrap_or(false),
                };
                results.extend(build_target_rows(&trp, params, ctx)?);
            }
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
                && !ctx.txn.has_schema_changes()
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
                    // Both executor modes have already bound the statement
                    // adjacency view; reuse it for records and schema.
                    mvcc_snapshot: ctx.mvcc_snapshot.or(ctx.txn.adj_snapshot()),
                    chunk_size: ctx.adaptive.parallel_chunk_size,
                    evaluation_time: ctx.valid_now,
                    schema_reads: parking_lot::Mutex::new(HashMap::new()),
                    // See the single-hop site: reads are not conflict-tracked
                    // at the default level, so nothing collects here either.
                    occ_read_keys: None,
                };
                let unresolved = Mutex::new(Vec::new());
                let parallel_rows =
                    process_targets_parallel(&depth_neighbors, row, params, &pctx, &unresolved)?;
                for (label, revision) in pctx.schema_reads.lock().iter() {
                    ctx.txn.note_label_schema_read(label, *revision);
                }
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
                // As at the single-hop site: targets with no plain record are
                // read sequentially, versions included. The parallel path runs
                // only without a path variable, so no route is bound here.
                for (src_uid, tgt_uid, et_idx) in unresolved.into_inner().unwrap_or_default() {
                    let mut hop_row = row.clone();
                    hop_row.insert(params.source.to_string(), Value::Int(src_uid as i64));
                    let trp = TargetRowParams {
                        input_row: &hop_row,
                        target_uid: tgt_uid,
                        edge_type: params.edge_types.get(et_idx).map(|s| s.as_str()),
                        edge_is_temporal: params
                            .edge_temporal
                            .get(et_idx)
                            .copied()
                            .unwrap_or(false),
                    };
                    results.extend(build_target_rows(&trp, params, ctx)?);
                }
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

/// Parameters for vector score filtering (threshold comparison + optional decay).
struct VectorScoreParams<'a> {
    function: &'a str,
    less_than: bool,
    threshold: f64,
    decay_field: Option<&'a crate::plan::expr::Expr>,
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
    vector_expr: &crate::plan::expr::Expr,
    query_vector_expr: &crate::plan::expr::Expr,
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

    // `exact` asks for every vector to be evaluated, never the index.
    if ctx.vector_consistency == VectorConsistencyMode::Exact {
        return Ok(None);
    }

    let Some(indexes) = ctx.vector_indexes else {
        return Ok(None);
    };
    let registry = indexes.registry;

    // Extract variable name and property from vector_expr (e.g. n.embedding).
    let (variable, property) = match vector_expr {
        crate::plan::expr::Expr::Property { base, key } => match base.as_ref() {
            crate::plan::expr::Expr::Variable(var) => (var.as_str(), key.as_str()),
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

    // Refused before the size threshold below, so whether a read at a named
    // timestamp is answered does not depend on how many rows it sees.
    // Small input: brute force is cheaper than HNSW + intersection overhead.
    // This covers the typical hybrid_search case where traversal narrows the
    // candidate set to a handful of nodes per query.
    if rows.len() < VECTOR_TOP_K_BRUTE_FORCE_THRESHOLD {
        return Ok(None);
    }

    // Evaluate the query vector (constant across all rows).
    let query_val = eval_neutral(query_vector_expr, &rows[0])?;
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

    gate_vector_index_read(indexes, &label_str, &property_str)?;
    // The commits the index has not folded yet, and the transaction's own
    // writes or (at a named timestamp) the nodes written since, are answered
    // from the rows, which hold this statement's view of every node; with no
    // record of which nodes the commits touched, the rows are ranked exactly.
    let delta = read_delta(registry.delta(ctx.shard_id), ctx)?;
    if delta == IndexDelta::Unknown {
        return Ok(None);
    }

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
    // A hit of a node the delta names is replaced by that node's row below.
    let mut intersected: Vec<(f32, &Row)> = Vec::new();
    for result in &search_results {
        if delta.contains(NodeId::from_raw(result.id)) {
            continue;
        }
        if let Some(row) = row_by_id.get(&result.id) {
            intersected.push((result.score, row));
        }
    }
    if let IndexDelta::Nodes(written) = &delta {
        for id in written {
            if let Some(row) = row_by_id.get(&id.as_raw()) {
                intersected.push((f32::NAN, row));
            }
        }
    }

    // If intersection is smaller than k, fall back to brute force. The HNSW
    // top-`overfetch` missed too many candidates that are in `rows` — meaning
    // the upstream filter was very restrictive and our overfetch was too small.
    if intersected.len() < k.min(rows.len()) {
        return Ok(None);
    }

    if !delta.is_empty() {
        // The written nodes have no index score: rank everything by the
        // function the query orders by, computed exactly on each row.
        let mut scored = Vec::with_capacity(intersected.len());
        for (_, row) in intersected {
            if let Some(exact) = recompute_score_for_row(vector_expr, &query_vec, function, row)? {
                scored.push((exact, row));
            }
        }
        let ascending = vector_function_ascending(function);
        scored.sort_by(|a, b| {
            if ascending {
                a.0.total_cmp(&b.0)
            } else {
                b.0.total_cmp(&a.0)
            }
        });
        let mut result_rows = Vec::with_capacity(k);
        for (exact, row) in scored.into_iter().take(k) {
            let mut cloned = row.clone();
            if let Some(alias) = distance_alias {
                cloned.insert(alias.to_string(), Value::Float(exact));
            }
            result_rows.push(cloned);
        }
        return Ok(Some(result_rows));
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
            let recomputed = recompute_score_for_row(vector_expr, &query_vec, function, &cloned)?;
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
    vector_expr: &crate::plan::expr::Expr,
    query_vec: &[f32],
    function: &str,
    row: &Row,
) -> Result<Option<f64>, EvalError> {
    // `Ok(None)` stays "this row has no score"; an evaluation that fails is an
    // error rather than a missing score, so the two do not get conflated.
    let vec_val = eval_neutral(vector_expr, row)?;
    let Some(a) = coerce_value_to_vec(&vec_val) else {
        return Ok(None);
    };
    if a.len() != query_vec.len() {
        return Ok(None);
    }
    let score = match function {
        "vector_distance" => coordinode_vector::metrics::euclidean_distance(&a, query_vec) as f64,
        "vector_similarity" => coordinode_vector::metrics::cosine_similarity(&a, query_vec) as f64,
        "vector_dot" => coordinode_vector::metrics::dot_product(&a, query_vec) as f64,
        "vector_manhattan" => coordinode_vector::metrics::manhattan_distance(&a, query_vec) as f64,
        _ => return Ok(None),
    };
    Ok(Some(score))
}

/// Brute-force top-K: compute distance per row, sort, take first K.
///
/// Used as fallback when `try_hnsw_vector_top_k` returns `None` (no index,
/// non-NodeScan input rows, or insufficient HNSW intersection).
fn execute_vector_top_k_brute_force(
    rows: Vec<Row>,
    vector_expr: &crate::plan::expr::Expr,
    query_vector_expr: &crate::plan::expr::Expr,
    function: &str,
    k: usize,
    distance_alias: Option<&str>,
) -> Result<Vec<Row>, ExecutionError> {
    if rows.is_empty() || k == 0 {
        return Ok(Vec::new());
    }

    // Evaluate query vector once (constant across rows).
    let query_val = eval_neutral(query_vector_expr, &rows[0])?;
    let query_vec = match coerce_value_to_vec(&query_val) {
        Some(v) => v,
        None => return Ok(Vec::new()),
    };

    // Compute (score, row) pairs, skipping rows with missing/mismatched vectors.
    let mut scored: Vec<(f64, Row)> = Vec::with_capacity(rows.len());
    for row in rows {
        let vec_val = eval_neutral(vector_expr, &row)?;
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
fn apply_decay_multiplier(
    raw_score: f64,
    decay_field: Option<&crate::plan::expr::Expr>,
    row: &Row,
) -> Result<f64, EvalError> {
    Ok(if let Some(decay_expr) = decay_field {
        let decay_val = eval_neutral(decay_expr, row)?;
        let decay_factor = match decay_val {
            Value::Float(f) => f,
            Value::Int(i) => i as f64,
            _ => 1.0, // missing decay → no attenuation
        };
        raw_score * decay_factor
    } else {
        raw_score
    })
}

fn execute_vector_filter(
    rows: &[Row],
    vector_expr: &crate::plan::expr::Expr,
    query_vector: &crate::plan::expr::Expr,
    params: &VectorScoreParams<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let mut results = Vec::new();

    for row in rows {
        let vec_val = eval_neutral(vector_expr, row)?;
        let query_val = eval_neutral(query_vector, row)?;

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

        let score = apply_decay_multiplier(raw_score, params.decay_field, row)?;

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
fn extract_method_ident(expr: &crate::plan::expr::Expr) -> Option<(String, String)> {
    match expr {
        crate::plan::expr::Expr::Property { base, key } => {
            if let crate::plan::expr::Expr::Variable(var) = base.as_ref() {
                return Some((var.clone(), key.clone()));
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
    method_expr: &crate::plan::expr::Expr,
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
        if let Some(reg) = ctx.vector_index_registry() {
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
            if reg.has_index(label, &property) {
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
    // `any` short-circuits, so a row that fails to evaluate stops the scan with
    // that failure rather than being read as "not a vector".
    let mut any_vector = false;
    for row in rows.iter() {
        if matches!(eval_neutral(method_expr, row)?, Value::Vector(_)) {
            any_vector = true;
            break;
        }
    }
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
    methods: &[crate::plan::expr::Expr],
    query_vector: Option<&crate::plan::expr::Expr>,
    query_text: Option<&crate::plan::expr::Expr>,
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
    // Text methods read the transaction's own writes from the store.
    materialize_own_node_writes(ctx)?;

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
    let qv_value = match query_vector {
        Some(e) => eval_neutral(e, &Row::new())?,
        None => Value::Null,
    };
    let qt_value = match query_text {
        Some(e) => eval_neutral(e, &Row::new())?,
        None => Value::Null,
    };

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
            )?,
            RankFuseMethodKind::VectorBruteForce => score_vector_method(
                &rows,
                method_expr,
                qv_slice,
                None, // default: cosine similarity
                kind.desc(),
                &variable_for_method(method_expr),
            )?,
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

    // Apply shard overfetch cap (set by a distributed plan; None in CE).
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

fn variable_for_method(expr: &crate::plan::expr::Expr) -> String {
    match expr {
        crate::plan::expr::Expr::Property { base: inner, .. } => {
            if let crate::plan::expr::Expr::Variable(v) = inner.as_ref() {
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
    method_expr: &crate::plan::expr::Expr,
    query_vec: &[f32],
    metric: Option<coordinode_core::graph::types::VectorMetric>,
    desc: bool,
    variable: &str,
) -> Result<Vec<Option<usize>>, EvalError> {
    // (row_idx, score, node_id) for matched rows.
    let mut scored: Vec<(usize, f64, u64)> = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let val = eval_neutral(method_expr, row)?;
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

    Ok(assign_competition_ranks(rows.len(), &mut scored, desc))
}

/// Score rows by a text (BM25) method via `TextIndexRegistry`. Missing FT-index
/// for the (label, property) is a hard error, like the `text_score()` guard.
fn score_text_method(
    rows: &[Row],
    method_expr: &crate::plan::expr::Expr,
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
    let scores = text_index_matches(registry, (variable, label, property), query_text, None, ctx)
        .map_err(|e| ExecutionError::Unsupported(format!("rrf_score(): text search error: {e}")))?
        .ok_or_else(|| {
            ExecutionError::Unsupported(format!(
                "rrf_score(): text method {variable}.{property} on :{label} requires a \
                 full-text index; create one with `CREATE TEXT INDEX … ON :{label}({property})`"
            ))
        })?;

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
    methods: &[crate::plan::expr::Expr],
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
                raw_scores_vector_method(rows, method_expr, qv_slice, Some(*metric), kind.desc())?
            }
            RankFuseMethodKind::VectorBruteForce => {
                raw_scores_vector_method(rows, method_expr, qv_slice, None, kind.desc())?
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
    method_expr: &crate::plan::expr::Expr,
    query_vec: &[f32],
    metric: Option<coordinode_core::graph::types::VectorMetric>,
    desc: bool,
) -> Result<Vec<Option<f64>>, EvalError> {
    let mut out: Vec<Option<f64>> = vec![None; rows.len()];
    for (i, row) in rows.iter().enumerate() {
        let val = eval_neutral(method_expr, row)?;
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
    Ok(out)
}

/// Sibling of `score_text_method` returning raw BM25 scores per row.
#[allow(clippy::too_many_arguments)]
fn raw_scores_text_method(
    rows: &[Row],
    method_expr: &crate::plan::expr::Expr,
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
    let scores = text_index_matches(registry, (variable, label, property), query_text, None, ctx)
        .map_err(|e| ExecutionError::Unsupported(format!("hybrid fusion: text search: {e}")))?
        .ok_or_else(|| {
            ExecutionError::Unsupported(format!(
                "hybrid fusion: no text index on :{label}({property})"
            ))
        })?;

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
fn eval_weight(expr: &crate::plan::expr::Expr, default: f64) -> Result<f64, EvalError> {
    let v = eval_neutral(expr, &Row::new())?;
    Ok(match v {
        Value::Float(f) if f.is_finite() => f,
        Value::Int(i) => i as f64,
        _ => default,
    })
}

/// Execute `LogicalOp::DocScore`: per doc row, traverse outward HAS_CHUNK,
/// read chunk nodes, score each chunk against the query vector via cosine
/// similarity, compute `α·max + β·avg + γ·coverage`, write it to
/// `__doc_score__` on the output row.
#[allow(clippy::too_many_arguments)]
fn execute_doc_score(
    rows: Vec<Row>,
    doc_variable: &str,
    query_vector: &crate::plan::expr::Expr,
    alpha: &crate::plan::expr::Expr,
    beta: &crate::plan::expr::Expr,
    gamma: &crate::plan::expr::Expr,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let alpha_f = eval_weight(alpha, 0.5)?;
    let beta_f = eval_weight(beta, 0.3)?;
    let gamma_f = eval_weight(gamma, 0.2)?;
    let query_val = eval_neutral(query_vector, &Row::new())?;
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
    doc_expr: &crate::plan::expr::Expr,
    query_expr: &crate::plan::expr::Expr,
    k: usize,
    score_alias: Option<&str>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let rows = execute_op(input, ctx)?;
    if k == 0 || rows.is_empty() {
        return Ok(Vec::new());
    }

    let query_val = eval_neutral(query_expr, &Row::new())?;
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
        let doc_val = eval_neutral(doc_expr, row)?;
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

/// Build the hard-fail error message for a missing full-text index.
///
/// Consistent with the `text_score()` guard and the RankFuse text-method
/// guard: `text_match()` must not silently pass every row when the index is
/// missing, which would turn `WHERE text_match(...)` into a no-op filter.
/// Tells the user how to fix it.
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

/// Fold the pending node merge operands of the statement's transaction into
/// its buffered records, so a store read sees the nodes it wrote as it does.
fn materialize_own_node_writes(ctx: &mut ExecutionContext<'_>) -> Result<(), ExecutionError> {
    if ctx.txn.node_deltas().is_empty() {
        return Ok(());
    }
    let mut keys: Vec<Vec<u8>> = ctx
        .txn
        .node_deltas()
        .iter()
        .map(|(key, _)| key.clone())
        .collect();
    keys.sort_unstable();
    keys.dedup();
    for key in keys {
        coordinode_modality::LocalNodeStore::materialize_pending_deltas(&mut ctx.txn, &key)?;
    }
    Ok(())
}

/// The nodes of this shard the statement's transaction has written and not
/// committed: no index holds them, so a search reads them through the
/// transaction.
fn own_written_nodes(ctx: &ExecutionContext<'_>) -> Vec<NodeId> {
    let buffered = ctx
        .txn
        .write_buffer()
        .keys()
        .filter(|(partition, _)| *partition == Partition::Node)
        .map(|(_, key)| key.as_slice());
    let merged = ctx.txn.node_deltas().iter().map(|(key, _)| key.as_slice());
    buffered
        .chain(merged)
        .filter_map(coordinode_core::graph::node::decode_written_node)
        .filter(|(shard, _)| *shard == ctx.shard_id)
        .map(|(_, id)| id)
        .collect()
}

/// Every node the text index of `(label, property)` matches for `query`
/// (tokenized by `language`, the index's default when `None`) with its BM25
/// score, as this statement reads the store: the writes the index has not
/// folded, the transaction's own (fold them first with
/// [`materialize_own_node_writes`]) and at a named timestamp the nodes
/// written since are evaluated from it. `Ok(None)` when there is no such
/// index.
fn text_index_matches(
    registry: &crate::index::TextIndexRegistry,
    (variable, label, property): (&str, &str, &str),
    query: &str,
    language: Option<&str>,
    ctx: &ExecutionContext<'_>,
) -> Result<Option<HashMap<u64, f32>>, String> {
    use coordinode_search::tantivy::multi_lang::TextRequest;
    use coordinode_search::tantivy::pending::Matches;

    let also = read_delta(IndexDelta::Nodes(Default::default()), ctx).map_err(|e| e.to_string())?;
    let Some(hits) = registry.find(
        label,
        property,
        &ctx.txn,
        ctx.shard_id,
        ctx.interner,
        also,
        // A temporal node matches by its state at the instant this variable
        // reads at, as a scan of the label projects it.
        ctx.instant_for(variable),
        TextRequest::Terms {
            query,
            language,
            snippets: false,
        },
        Matches::All,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(
        hits.into_iter()
            .map(|hit| (hit.node_id, hit.score))
            .collect(),
    ))
}

/// TextFilter: search TextIndex for matching documents, filter rows.
///
/// For each row, evaluates `text_expr` to get the text content of a node field,
/// then searches the TextIndex for matches. Keeps rows whose node_id appears
/// in the search results.
fn execute_text_filter(
    rows: &[Row],
    text_expr: &crate::plan::expr::Expr,
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
    // A legacy index is asked for as many matches as it holds documents: the
    // predicate is membership, which a top-K cutoff would truncate.
    let limit = ctx
        .text_index
        .map_or(1, |index| {
            usize::try_from(index.num_docs()).unwrap_or(usize::MAX)
        })
        .max(1);
    // Extract property name from text_expr (PropertyAccess { var, property }).
    let property = match text_expr {
        crate::plan::expr::Expr::Property { key, .. } => Some(key.as_str()),
        _ => None,
    };
    // Extract label from the first row's __label__ field, if the variable is
    // bound and carries its label.
    let label_owned: Option<String> = match text_expr {
        crate::plan::expr::Expr::Property { base, .. } => match base.as_ref() {
            crate::plan::expr::Expr::Variable(var) => rows.first().and_then(|r| {
                r.get(&format!("{var}.__label__"))
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
            }),
            _ => None,
        },
        _ => None,
    };
    let label = label_owned.as_deref();
    let variable = match text_expr {
        crate::plan::expr::Expr::Property { base, .. } => match base.as_ref() {
            crate::plan::expr::Expr::Variable(var) => var.as_str(),
            _ => "",
        },
        _ => "",
    };

    let search_results: Vec<coordinode_search::tantivy::TextSearchResult> = if let Some(registry) =
        ctx.text_index_registry
    {
        if let (Some(l), Some(p)) = (label, property) {
            // Every match, not a top-K: the predicate is membership.
            match text_index_matches(registry, (variable, l, p), query_string, language, ctx)
                .map_err(|e| ExecutionError::Unsupported(format!("text search error: {e}")))?
            {
                Some(matches) => matches
                    .into_iter()
                    .map(
                        |(node_id, score)| coordinode_search::tantivy::TextSearchResult {
                            node_id,
                            score,
                        },
                    )
                    .collect(),
                // Registry is wired but has no index for (label, property) —
                // hard-fail, don't silently pass every row through.
                None => return Err(text_match_missing_index_error(Some(l), Some(p))),
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
            // Neither registry lookup worked nor legacy index present.
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
        // No registry and no legacy index — hard-fail.
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
            crate::plan::expr::Expr::Property { base, .. } => {
                if let crate::plan::expr::Expr::Variable(var) = base.as_ref() {
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
    field_expr: &crate::plan::expr::Expr,
    token_expr: &crate::plan::expr::Expr,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_search::encrypted::{EncryptedIndex, SearchToken};

    if rows.is_empty() {
        return Ok(Vec::new());
    }

    // Evaluate token from expression (parameter or literal).
    // Use the first row for context (token expression is typically a parameter,
    // independent of row data).
    let token_val = eval_neutral(token_expr, &rows[0])?;
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
        crate::plan::expr::Expr::Property { base, key } => {
            if let crate::plan::expr::Expr::Variable(var) = base.as_ref() {
                (var.clone(), key.clone())
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
fn execute_unwind(
    rows: &[Row],
    expr: &crate::plan::expr::Expr,
    variable: &str,
) -> Result<Vec<Row>, ExecutionError> {
    let mut results = Vec::new();

    for row in rows {
        let val = eval_neutral(expr, row)?;
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
fn collect_expr_vars(expr: &crate::plan::expr::Expr, vars: &mut Vec<String>) {
    use crate::plan::expr::{Expr as PExpr, MapProjItem};
    match expr {
        PExpr::Variable(name) => vars.push(name.clone()),
        PExpr::Property { base, .. } => collect_expr_vars(base, vars),
        PExpr::Binary { left, right, .. } => {
            collect_expr_vars(left, vars);
            collect_expr_vars(right, vars);
        }
        PExpr::Unary { operand, .. } => collect_expr_vars(operand, vars),
        PExpr::Call { args, .. } => {
            for arg in args {
                collect_expr_vars(arg, vars);
            }
        }
        PExpr::List(items) => {
            for item in items {
                collect_expr_vars(item, vars);
            }
        }
        PExpr::Map(entries) => {
            for (_, val) in entries {
                collect_expr_vars(val, vars);
            }
        }
        PExpr::MapProjection { base, items } => {
            collect_expr_vars(base, vars);
            for item in items {
                if let MapProjItem::Computed(_, e) = item {
                    collect_expr_vars(e, vars);
                }
            }
        }
        PExpr::In { item, list } => {
            collect_expr_vars(item, vars);
            collect_expr_vars(list, vars);
        }
        PExpr::IsNull { operand, .. } | PExpr::IsTyped { operand, .. } => {
            collect_expr_vars(operand, vars)
        }
        PExpr::StringMatch { value, pattern, .. } => {
            collect_expr_vars(value, vars);
            collect_expr_vars(pattern, vars);
        }
        PExpr::Case {
            operand,
            branches,
            otherwise,
        } => {
            if let Some(op) = operand {
                collect_expr_vars(op, vars);
            }
            for (cond, then) in branches {
                collect_expr_vars(cond, vars);
                collect_expr_vars(then, vars);
            }
            if let Some(el) = otherwise {
                collect_expr_vars(el, vars);
            }
        }
        PExpr::Subscript { base, index } => {
            collect_expr_vars(base, vars);
            collect_expr_vars(index, vars);
        }
        PExpr::Slice { base, start, end } => {
            collect_expr_vars(base, vars);
            if let Some(s) = start {
                collect_expr_vars(s, vars);
            }
            if let Some(e) = end {
                collect_expr_vars(e, vars);
            }
        }
        PExpr::Reduce {
            acc,
            init,
            var,
            list,
            step,
        } => {
            collect_expr_vars(init, vars);
            collect_expr_vars(list, vars);
            // acc and var are bound locally inside the fold; exclude them from
            // the outer variables the step expression depends on.
            let mut inner = Vec::new();
            collect_expr_vars(step, &mut inner);
            vars.extend(inner.into_iter().filter(|v| v != acc && v != var));
        }
        PExpr::ListQuantifier {
            var,
            list,
            predicate,
            ..
        } => {
            collect_expr_vars(list, vars);
            // var is bound locally; exclude it from the predicate's outer deps.
            let mut inner = Vec::new();
            collect_expr_vars(predicate, &mut inner);
            vars.extend(inner.into_iter().filter(|v| v != var));
        }
        PExpr::ListComprehension {
            var,
            list,
            filter,
            map,
        } => {
            collect_expr_vars(list, vars);
            let mut inner = Vec::new();
            if let Some(p) = filter {
                collect_expr_vars(p, &mut inner);
            }
            if let Some(m) = map {
                collect_expr_vars(m, &mut inner);
            }
            vars.extend(inner.into_iter().filter(|v| v != var));
        }
        // Embedded subplans bind their own variables; any outer-correlation
        // variables they reference are already provisioned by the outer clauses,
        // so they contribute no extra outer dependencies here.
        PExpr::ExistsSubplan(_)
        | PExpr::CountSubplan(_)
        | PExpr::CollectSubplan { .. }
        | PExpr::PatternComprehension { .. } => {}
        // Literal, Parameter, Star — no variable references.
        PExpr::Literal(_) | PExpr::Parameter(_) | PExpr::Star => {}
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

/// Parameters for shortest path computation.
struct ShortestPathParams<'a> {
    source: &'a str,
    target: &'a str,
    edge_types: &'a [String],
    direction: Direction,
    max_depth: u64,
    path_variable: &'a str,
}

/// BFS shortest path between two bound nodes.
///
/// Finds the shortest (unweighted) route from source to target over the
/// given edge types and binds it to the path variable as a path value, or
/// NULL when the target is not reachable within the depth bound.
fn execute_shortest_path(
    rows: &[Row],
    sp: &ShortestPathParams<'_>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use std::collections::VecDeque;

    let max_d = sp.max_depth.min(DEFAULT_MAX_HOPS) as usize;
    let mut results = Vec::new();

    // An untyped pattern names no type, and the frontier expands per named
    // type, so without this the BFS explores nothing and calls the target
    // unreachable. Resolved once for the whole call, not per row.
    let resolved_types: Vec<String>;
    let edge_types: &[String] = if sp.edge_types.is_empty() {
        resolved_types = ctx.list_edge_types()?;
        &resolved_types
    } else {
        sp.edge_types
    };

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
            let neighbors = expand_one_hop(nid, edge_types, sp.direction, ctx)?;

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
                let edge_type = edge_types.get(et_idx).cloned().unwrap_or_default();
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
    expr: &crate::plan::expr::Expr,
    params: &HashMap<String, coordinode_core::graph::types::Value>,
) -> Option<f64> {
    use crate::plan::expr::Expr as PExpr;
    match expr {
        PExpr::Literal(Value::Float(f)) => Some(*f),
        PExpr::Literal(Value::Int(i)) => Some(*i as f64),
        PExpr::Parameter(name) => params.get(name).and_then(|v| match v {
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
    group_by: &[crate::plan::expr::Expr],
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
            let key: Vec<Value> = group_by
                .iter()
                .map(|e| eval_neutral(e, row))
                .collect::<Result<_, _>>()?;
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
            let col = expr_display_name_neutral(expr);
            out.insert(col, val);
        }

        // Compute aggregates
        for agg in aggregates {
            let val = compute_aggregate(agg, group_rows, params)?;
            let col = agg.alias.clone().unwrap_or_else(|| agg.function.clone());
            out.insert(col, val);
        }

        results.push(out);
    }

    Ok(results)
}

/// Evaluate aggregate argument for all rows, applying DISTINCT dedup if needed.
/// Returns non-null values only.
fn eval_aggregate_values(agg: &AggregateItem, rows: &[&Row]) -> Result<Vec<Value>, EvalError> {
    let mut values: Vec<Value> = Vec::with_capacity(rows.len());
    for r in rows.iter() {
        let v = eval_neutral(&agg.arg, r)?;
        if !v.is_null() {
            values.push(v);
        }
    }

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

    Ok(values)
}

/// Compute a single aggregate function over a group of rows.
fn compute_aggregate(
    agg: &AggregateItem,
    rows: &[&Row],
    params: &HashMap<String, coordinode_core::graph::types::Value>,
) -> Result<Value, EvalError> {
    use crate::function::AggregateFn;
    let function = AggregateFn::resolve(&agg.function)
        .ok_or_else(|| EvalError::UnknownFunction(agg.function.clone()))?;
    Ok(match function {
        AggregateFn::Count => {
            if agg.arg == crate::plan::expr::Expr::Star {
                // count(*) ignores DISTINCT — counts all rows
                Value::Int(rows.len() as i64)
            } else {
                let values = eval_aggregate_values(agg, rows)?;
                Value::Int(values.len() as i64)
            }
        }
        AggregateFn::Sum => {
            let values = eval_aggregate_values(agg, rows)?;
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
        AggregateFn::Avg => {
            let values = eval_aggregate_values(agg, rows)?;
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
        AggregateFn::Min => {
            let values = eval_aggregate_values(agg, rows)?;
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
        AggregateFn::Max => {
            let values = eval_aggregate_values(agg, rows)?;
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
        AggregateFn::Collect => {
            let values = eval_aggregate_values(agg, rows)?;
            Value::Array(values)
        }
        AggregateFn::PercentileCont | AggregateFn::PercentileDisc => {
            let agg_values = eval_aggregate_values(agg, rows)?;
            let mut values: Vec<f64> = agg_values
                .iter()
                .filter_map(|v| match v {
                    Value::Int(n) => Some(*n as f64),
                    Value::Float(f) => Some(*f),
                    _ => None,
                })
                .collect();

            if values.is_empty() {
                return Ok(Value::Null);
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

            if function == AggregateFn::PercentileDisc {
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
        AggregateFn::StDev | AggregateFn::StDevP => {
            let agg_values = eval_aggregate_values(agg, rows)?;
            let values: Vec<f64> = agg_values
                .iter()
                .filter_map(|v| match v {
                    Value::Int(n) => Some(*n as f64),
                    Value::Float(f) => Some(*f),
                    _ => None,
                })
                .collect();

            if values.is_empty() {
                return Ok(Value::Null);
            }

            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>();

            if function == AggregateFn::StDevP {
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
    })
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
    on_match: &[crate::plan::SetItem],
    on_create: &[crate::plan::SetItem],
    multi: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // MERGE (src)-[r:TYPE]->(tgt) — relationship pattern with correlated bindings.
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

    // Standalone MERGE ALL — Cartesian product across all matching src × tgt nodes.
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

    // Standalone relationship MERGE — no correlated_row (no preceding MATCH).
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
    on_match: &[crate::plan::SetItem],
    on_create: &[crate::plan::SetItem],
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
        LogicalOp::ProcedureCall { input, yields, .. } => {
            collect_bound_vars(input, out);
            out.extend(yields.iter().flatten().map(|y| y.variable.clone()));
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
        // A call whose arguments read an outer variable runs per outer row.
        LogicalOp::ProcedureCall {
            input,
            args,
            filter,
            ..
        } => {
            scan_filter_references_outside(input, bound)
                || args.iter().any(|a| expr_references_outside(a, bound))
                || filter
                    .as_ref()
                    .is_some_and(|f| expr_references_outside(f, bound))
        }
        LogicalOp::CartesianProduct { left, right } | LogicalOp::LeftOuterJoin { left, right } => {
            scan_filter_references_outside(left, bound)
                || scan_filter_references_outside(right, bound)
        }
        _ => false,
    }
}

/// True if `expr` references any `Variable` not present in `bound`.
fn expr_references_outside(
    expr: &crate::plan::expr::Expr,
    bound: &std::collections::HashSet<String>,
) -> bool {
    use crate::plan::expr::Expr as PExpr;
    match expr {
        PExpr::Variable(name) => !bound.contains(name),
        PExpr::Property { base, .. } => expr_references_outside(base, bound),
        PExpr::Unary { operand, .. } => expr_references_outside(operand, bound),
        PExpr::Binary { left, right, .. } => {
            expr_references_outside(left, bound) || expr_references_outside(right, bound)
        }
        PExpr::Call { args, .. } | PExpr::List(args) => {
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
            )));
        }
    };
    let target_id = match correlated.get(target_variable.as_str()) {
        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
        _ => {
            return Err(ExecutionError::Unsupported(format!(
                "MERGE relationship: target variable '{target_variable}' not bound in scope"
            )));
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
        // If edge_filters are specified, also verify that the stored edge properties
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
            // A loop rather than `all`, so a filter that cannot be evaluated
            // stops here instead of reading as "does not match".
            let mut filters_match = true;
            for (prop_name, filter_expr) in edge_filters.iter() {
                let actual = stored.get(prop_name).cloned().unwrap_or(Value::Null);
                if actual != eval_neutral(filter_expr, correlated)? {
                    filters_match = false;
                    break;
                }
            }
            if !filters_match {
                // This edge exists but has different property values — treat as no match.
                continue;
            }
        }

        // Edge found and all property filters match.
        let mut row = correlated.clone();
        if let Some(ev) = edge_variable {
            let (ep_src, ep_tgt) = match direction {
                Direction::Outgoing | Direction::Both => (source_id, target_id),
                Direction::Incoming => (target_id, source_id),
            };
            row.insert(format!("{ev}.__type__"), Value::String(et.clone()));
            row.insert(ev.clone(), Value::String(et.clone()));
            // The endpoints a later SET needs to find the edgeprop key. Without
            // them every edge SET in the statement locates nothing and returns
            // quietly, so `MERGE (a)-[r:T]->(b) SET r.x = 1` answered with the
            // new value and stored none of it.
            row.insert(format!("{ev}.__src__"), Value::Int(ep_src.as_raw() as i64));
            row.insert(format!("{ev}.__tgt__"), Value::Int(ep_tgt.as_raw() as i64));
            // Populate stored edge properties into the result row so ON MATCH SET can
            // reference them via `r.prop_name`.
            if !edge_filters.is_empty() {
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

// ----------------------------------------------------------------------------
// Neutral storage-aware evaluation.
//
// The neutral expression IR carries correlated subqueries as an already-lowered
// `Box<LogicalPlan>` (the pattern + its filter planned at build time), so the
// neutral evaluator runs that subplan directly instead of re-planning a cypher
// MatchClause per row. This is the language-neutral analogue of
// `eval_predicate_with_storage` and is the path the planner converges on.
// ----------------------------------------------------------------------------

/// Does a neutral expression contain a correlated-subplan form that needs the
/// storage engine? Mirrors `expr_contains_pattern_predicate`: only top-level,
/// unary, and binary positions are inspected (a subplan buried inside another
/// node is evaluated by the pure path, which yields `Null`/`false` for it, just
/// as the cypher evaluator does).
fn neutral_contains_subplan(expr: &crate::plan::expr::Expr) -> bool {
    use crate::plan::expr::Expr as PExpr;
    match expr {
        PExpr::ExistsSubplan(_)
        | PExpr::CountSubplan(_)
        | PExpr::CollectSubplan { .. }
        | PExpr::PatternComprehension { .. } => true,
        // A comprehension or quantifier binds each element of its list to a
        // variable, and an element drawn from `nodes(p)` / `relationships(p)`
        // is only an identity until its properties are read from storage.
        // Evaluating those on the pure path leaves every `x.prop` NULL.
        PExpr::ListComprehension { .. } | PExpr::ListQuantifier { .. } => true,
        PExpr::Unary { operand, .. } => neutral_contains_subplan(operand),
        PExpr::Binary { left, right, .. } => {
            neutral_contains_subplan(left) || neutral_contains_subplan(right)
        }
        _ => false,
    }
}

/// Bind one element of a path list under `var`, with its properties.
///
/// `nodes(p)` yields node ids and `relationships(p)` yields `{type, source,
/// target}` maps, neither of which carries properties. A MATCH binding for the
/// same node looks like `x` plus a `x.<prop>` column per property, so this
/// writes that shape: the element keeps its identity (so `id(x)` still reads an
/// id) and gains the columns a predicate or projection expects to find.
fn bind_path_element(
    scratch: &mut Row,
    var: &str,
    item: &Value,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    // The scratch row is reused for every element, so last element's columns
    // have to go first. Kept, they answer for an element that does not carry
    // the property: a path whose middle node has no name would report the
    // name of the node before it.
    let prefix = format!("{var}.");
    scratch.retain(|column, _| !column.starts_with(&prefix));
    scratch.insert(var.to_string(), item.clone());
    match item {
        Value::Int(raw) => {
            let node_id = NodeId::from_raw(*raw as u64);
            if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                insert_label_columns(scratch, var, &record);
                for (field_id, value) in &record.props {
                    if let Some(name) = ctx.interner.resolve(*field_id) {
                        scratch.insert(format!("{var}.{name}"), value.clone());
                    }
                }
            }
        }
        Value::Map(map) => {
            let (Some(Value::String(edge_type)), Some(Value::Int(src)), Some(Value::Int(tgt))) =
                (map.get("type"), map.get("source"), map.get("target"))
            else {
                return Ok(());
            };
            let (edge_type, src, tgt) = (
                edge_type.clone(),
                NodeId::from_raw(*src as u64),
                NodeId::from_raw(*tgt as u64),
            );
            // Non-temporal key only: a path hop does not carry the valid_from
            // that selects one version of a temporal edge.
            if let Some(props) = ctx.mvcc_get_edge_props_either(&edge_type, src, tgt, None)? {
                for (field_id, value) in &props {
                    if let Some(name) = ctx.interner.resolve(*field_id) {
                        scratch.insert(format!("{var}.{name}"), value.clone());
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Evaluate a neutral expression that may contain correlated subplans, with
/// storage access for the embedded subqueries. Subplan-free nodes fall through
/// to the pure neutral evaluator.
fn eval_neutral_with_storage(
    expr: &crate::plan::expr::Expr,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Value, ExecutionError> {
    use crate::plan::expr::{BinOp, Expr as PExpr};
    match expr {
        PExpr::ExistsSubplan(subplan) => exists_subplan_matches(subplan, row, ctx),
        PExpr::CountSubplan(subplan) => {
            let rows = correlated_subplan_rows(subplan, row, ctx)?;
            Ok(Value::Int(i64::try_from(rows.len()).unwrap_or(i64::MAX)))
        }
        PExpr::CollectSubplan {
            subplan,
            projection,
        } => {
            let rows = correlated_subplan_rows(subplan, row, ctx)?;
            Ok(Value::Array(
                rows.iter()
                    .map(|er| eval_neutral(projection, er))
                    .collect::<Result<_, _>>()?,
            ))
        }
        PExpr::PatternComprehension { subplan, map } => {
            let rows = correlated_subplan_rows(subplan, row, ctx)?;
            Ok(Value::Array(
                rows.iter()
                    .map(|er| eval_neutral(map, er))
                    .collect::<Result<_, _>>()?,
            ))
        }
        PExpr::ListComprehension {
            var,
            list,
            filter,
            map,
        } => {
            let Value::Array(items) = eval_neutral_with_storage(list, row, ctx)? else {
                return Ok(Value::Null);
            };
            let mut scratch = row.clone();
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                bind_path_element(&mut scratch, var, &item, ctx)?;
                let keep = match filter {
                    Some(p) => matches!(
                        eval_neutral_with_storage(p, &scratch, ctx)?,
                        Value::Bool(true)
                    ),
                    None => true,
                };
                if keep {
                    out.push(match map {
                        Some(m) => eval_neutral_with_storage(m, &scratch, ctx)?,
                        None => item,
                    });
                }
            }
            Ok(Value::Array(out))
        }
        PExpr::ListQuantifier {
            kind,
            var,
            list,
            predicate,
        } => {
            let Value::Array(items) = eval_neutral_with_storage(list, row, ctx)? else {
                return Ok(Value::Null);
            };
            let total = items.len();
            let mut scratch = row.clone();
            let mut true_count = 0usize;
            for item in items {
                bind_path_element(&mut scratch, var, &item, ctx)?;
                if matches!(
                    eval_neutral_with_storage(predicate, &scratch, ctx)?,
                    Value::Bool(true)
                ) {
                    true_count += 1;
                }
            }
            Ok(Value::Bool(match kind {
                crate::plan::expr::Quantifier::All => true_count == total,
                crate::plan::expr::Quantifier::Any => true_count > 0,
                crate::plan::expr::Quantifier::None => true_count == 0,
                crate::plan::expr::Quantifier::Single => true_count == 1,
            }))
        }
        PExpr::Unary { op, operand } => {
            let v = eval_neutral_with_storage(operand, row, ctx)?;
            Ok(eval_unary_op(*op, &v))
        }
        PExpr::Binary { left, op, right } => {
            let lv = eval_neutral_with_storage(left, row, ctx)?;
            // Short-circuit AND/OR before evaluating the right side.
            match op {
                BinOp::And if !is_truthy(&lv) => return Ok(Value::Bool(false)),
                BinOp::Or if is_truthy(&lv) => return Ok(Value::Bool(true)),
                _ => {}
            }
            let rv = eval_neutral_with_storage(right, row, ctx)?;
            Ok(eval_binary_op(&lv, *op, &rv)?)
        }
        other => Ok(eval_neutral(other, row)?),
    }
}

/// Neutral `EXISTS`: true when the embedded subplan yields at least one row
/// consistent with the outer bindings on shared variables.
fn exists_subplan_matches(
    subplan: &crate::planner::logical::LogicalPlan,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Value, ExecutionError> {
    let rows = execute_op(&subplan.root, ctx)?;
    let any = rows
        .iter()
        .any(|rr| rr.iter().all(|(k, v)| row.get(k).is_none_or(|ov| ov == v)));
    Ok(Value::Bool(any))
}

/// Execute an embedded subplan correlated with the outer `row`: keep only rows
/// that agree with the outer bindings on shared variables, each merged with the
/// outer bindings. Shared by neutral `COUNT`/`COLLECT`/pattern-comprehension.
fn correlated_subplan_rows(
    subplan: &crate::planner::logical::LogicalPlan,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let rows = execute_op(&subplan.root, ctx)?;
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

/// Create a relationship edge described by `traverse` between the source and target
/// nodes bound in `correlated`. Called from `execute_merge` when no existing edge
/// was found by `execute_merge_relationship_check`.
///
/// Edge properties from `edge_filters` are stored in the EdgeProp partition
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
            )));
        }
    };
    let target_id = match correlated.get(target_variable.as_str()) {
        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
        _ => {
            return Err(ExecutionError::Unsupported(format!(
                "MERGE relationship: target variable '{target_variable}' not bound in scope"
            )));
        }
    };

    // Use the first (and typically only) edge type for creation.
    let et = &edge_types[0];

    // Direction determines which node is "from" vs "to" in the adjacency lists.
    let (from_id, to_id) = match direction {
        Direction::Outgoing | Direction::Both => (source_id, target_id),
        Direction::Incoming => (target_id, source_id),
    };
    // This branch runs because the check found nothing to match. What was
    // observed depends on the pattern: with no property filters it is the
    // plain absence of the pair, and saying so protects the creation against
    // a concurrent erase that writes different keys and so passes
    // first-committer-wins. With filters the check answered a narrower
    // question ("no edge matching these properties"), which a pair claim does
    // not state; an edge can be there and still not match. Claiming absence
    // there would assert an observation nobody made.
    if edge_filters.is_empty() {
        ctx.claim_pair_observed_absent(et, from_id, to_id);
    }
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
        // Same endpoints the edgeprop key was written under, so a SET later in
        // the statement can find the edge this MERGE just created.
        row.insert(format!("{ev}.__src__"), Value::Int(from_id.as_raw() as i64));
        row.insert(format!("{ev}.__tgt__"), Value::Int(to_id.as_raw() as i64));
    }

    // Store edge properties (from pattern `[r:TYPE {prop: val}]`) in EdgeProp partition.
    // Key: edgeprop:<TYPE>:<from_id BE>:<to_id BE>  (same format as CREATE clause).
    // Value: MessagePack Vec<(field_id, Value)>.
    // Note: if this edge already existed with different properties, the new values
    // overwrite the old (upsert semantics within a single-edge-per-type data model).
    let mut resolved_props: Vec<(String, Value)> = Vec::with_capacity(edge_filters.len());
    if !edge_filters.is_empty() {
        let names: Vec<&str> = edge_filters.iter().map(|(n, _)| n.as_str()).collect();
        let field_ids = ctx.field_ids(&names)?;
        let mut prop_map: Vec<(u32, Value)> = Vec::with_capacity(edge_filters.len());
        for ((prop_name, expr), field_id) in edge_filters.iter().zip(field_ids) {
            let value = eval_neutral(expr, correlated)?;
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

/// Create a complete relationship pattern for standalone MERGE (no preceding MATCH).
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
/// "Find or create" semantics:
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
            )));
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
            )));
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
fn execute_upsert(
    pattern: &LogicalOp,
    on_match: &[crate::plan::SetItem],
    on_create_patterns: &[Pattern],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Step 1: MATCH — find existing nodes
    let matches = execute_op(pattern, ctx)?;

    if !matches.is_empty() {
        // Step 2: ON MATCH — apply SET items
        if on_match.is_empty() {
            return Ok(matches);
        }

        // Concurrent-modification detection is owned by Layer-3 OCC:
        // every `mvcc_get` issued during MATCH (above) tracked the
        // node key in the per-transaction `occ_scope`; at commit time
        // `mvcc_flush` calls `coordinator.validate_occ` which probes
        // `has_write_after` on each tracked key and surfaces
        // `ExecutionError::Conflict` if any concurrent writer landed
        // since `mvcc_read_ts`. No byte-level CAS here: it would be
        // redundant and strictly less safe (it tolerates ABA writes,
        // OCC does not).
        execute_update(&matches, on_match, &ViolationMode::Fail, ctx)
    } else {
        // Step 3: ON CREATE — create nodes and edges from patterns (two-pass)
        // Safe-reject for temporal labels: UPSERT ON CREATE uses
        // `encode_node_key` (16-byte form) directly and would silently
        // bypass the per-version 25-byte key on a temporal label, also
        // skipping `__ingestion_ts__` auto-population and the
        // `valid_from`-required guard.
        for create_pattern in on_create_patterns {
            for element in &create_pattern.elements {
                if let PatternElement::Node(np) = element {
                    for lbl in &np.labels {
                        if let Ok(Some(s)) = ctx.load_current_label_schema(lbl) {
                            if s.temporal {
                                return Err(ExecutionError::Unsupported(format!(
                                    "UPSERT ON CREATE into temporal label '{lbl}' is not \
                                     yet supported. Use an explicit CREATE clause for \
                                     temporal labels."
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
                        // UPSERT's ON CREATE branch is a CREATE of the
                        // pattern node: every label, the schema checks, the
                        // index entries and the CREATE triggers of each label.
                        let created = execute_create_node(
                            std::slice::from_ref(&current_row),
                            Some(np.variable.as_deref().unwrap_or("_")),
                            &np.labels,
                            &np.properties,
                            ctx,
                        )?;
                        if let Some(created_row) = created.into_iter().next() {
                            current_row = created_row;
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
            // Safe-reject for temporal labels: this code path (used by
            // MERGE's create branch and by `MERGE (a)-[:E]->(b)` endpoint
            // synthesis) writes via the 16-byte non-temporal key
            // unconditionally. A temporal target would silently land in
            // non-temporal storage, so MERGE/UPSERT on temporal labels is
            // refused until it writes per-version keys.
            for lbl in labels {
                if let Ok(Some(s)) = ctx.load_current_label_schema(lbl) {
                    if s.temporal {
                        return Err(ExecutionError::Unsupported(format!(
                            "MERGE / UPSERT into temporal label '{lbl}' is not yet \
                             supported. Use explicit CREATE for temporal labels."
                        )));
                    }
                }
            }

            // The node MERGE invents is a CREATE of its pattern node: every
            // label, the schema checks, the index entries, the vector and
            // text index writes, and the CREATE triggers of each label.
            execute_create_node(&[Row::new()], Some(variable), labels, property_filters, ctx)
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

/// Gate a vector search against this member's build of the index and its
/// online-during-build policy. Returns `Ok(())` when the caller can use the
/// in-memory HNSW handle, `Err(...)` when the caller must abort.
///
/// The graph is this member's own, built from the data it holds, so its
/// readiness is the graph's health signal, never the replicated definition.
/// Under `Block` the reader waits at most `indexes.build_wait`, the bound its
/// caller chose (query hint, else session setting).
///
/// Cost: under `PartialRecall` this is a single map lookup, the dominant
/// common case.
fn gate_vector_index_read(
    indexes: VectorIndexes<'_>,
    label: &str,
    property: &str,
) -> Result<(), ExecutionError> {
    let registry = indexes.registry;
    let Some(def) = registry.get_definition(label, property) else {
        // No registered def — caller will fall back to brute-force or
        // return an empty result. Not our gate to enforce.
        return Ok(());
    };

    // PartialRecall: search whatever the graph holds right now.
    if def.online_during_build == OnlineDuringBuild::PartialRecall {
        return Ok(());
    }
    // A graph without a health signal is not being built.
    let Some(health) = registry.health_handle(label, property) else {
        return Ok(());
    };

    // The health turns ready once the build has handed maintenance to the
    // worker and folded what the worker left to it during the scan, which is
    // where the build ends: a reader waits for the scan and that fold.
    let wait = indexes.build_wait;
    // A wait too long to add to the clock is no bound at all.
    let deadline = std::time::Instant::now().checked_add(wait);
    let poll_step = std::time::Duration::from_millis(25);
    loop {
        let state = health.snapshot();
        if state.is_ready() {
            return Ok(());
        }
        if state.is_offline() {
            return Err(ExecutionError::Unsupported(format!(
                "vector index '{def}' failed to build on this node"
            )));
        }
        match def.online_during_build {
            OnlineDuringBuild::Offline => {
                return Err(ExecutionError::Unsupported(format!(
                    "vector index '{def}' is offline during build"
                )));
            }
            OnlineDuringBuild::Block => {
                let remaining = match deadline {
                    Some(deadline) => deadline.saturating_duration_since(std::time::Instant::now()),
                    None => poll_step,
                };
                if remaining.is_zero() {
                    return Err(ExecutionError::Unsupported(format!(
                        "vector index '{def}' still building after {wait:?}; wait longer \
                         with /*+ vector_build_wait('...') */ or the session's \
                         vector_build_wait"
                    )));
                }
                // Never sleep past the caller's bound.
                std::thread::sleep(poll_step.min(remaining));
            }
            // Returned above.
            OnlineDuringBuild::PartialRecall => return Ok(()),
        }
    }
}

/// CREATE node: allocate ID, build record, write to storage.
fn execute_create_node(
    input_rows: &[Row],
    variable: Option<&str>,
    labels: &[String],
    properties: &[(String, crate::plan::expr::Expr)],
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

    // Reserved-name guard at CREATE time: the engine's temporal fields
    // (`__ingestion_ts__`, `__deleted__`) are written by the engine into
    // every version of a temporal node, user-immutable. Rejecting user-supplied values up
    // front prevents accidental shadowing and matches the symmetric DDL-time
    // reserved-name diagnostic in `execute_create_node_type`.
    for (prop_name, _) in properties {
        if coordinode_core::schema::definition::TEMPORAL_ENGINE_FIELDS.contains(&prop_name.as_str())
        {
            return Err(ExecutionError::Unsupported(format!(
                "property name '{prop_name}' is reserved for engine-internal use \
                 (the engine writes it into every version of a temporal node); \
                 it cannot be assigned in CREATE"
            )));
        }
    }

    // Write-time guard: temporal node types require `valid_from` on every
    // CREATE (mirror of the edge-side enforcement at `execute_create_edge`).
    // The per-version node key carries the i64 BE valid_from suffix, so a
    // CREATE on a TEMPORAL label without `valid_from` is rejected at write
    // time rather than writing a record the key encoder cannot place.
    //
    // Multi-label case: scan EVERY label, not just `labels.first()`. A node
    // declared `CREATE (n:Foo:Bar)` where any of Foo / Bar carries the
    // TEMPORAL flag must satisfy the bitemporal contract. Reports the first
    // temporal label name to the user.
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

    // Every row, of a table or not, is a node with an id from the lease. A
    // table with declared key columns also holds each row's key in its key
    // index, which refuses a key another row holds. A ROW table writes the
    // node on the node path; a COLUMNAR table writes it to its own columnar
    // tree (`table_columnar`).
    let (key_columns, table_columnar): (Vec<String>, bool) = match schema.as_ref() {
        Some(s) if s.is_table() => (s.key_columns().to_vec(), s.is_columnar()),
        _ => (Vec::new(), false),
    };
    let table_label = labels.first().cloned().unwrap_or_default();

    for input_row in input_rows {
        let node_id = ctx.id_allocator.next()?;
        let mut row_key = Vec::with_capacity(key_columns.len());
        for column in &key_columns {
            let Some((_, expr)) = properties.iter().find(|(n, _)| n == column) else {
                return Err(ExecutionError::SchemaViolation(format!(
                    "key column '{column}' is required to insert into table '{table_label}'"
                )));
            };
            row_key.push(eval_neutral(expr, input_row)?);
        }

        let mut record = NodeRecord::with_labels(labels.to_vec());
        // Every name stored under an id is registered in one batch up front.
        // An undeclared property of a VALIDATED label is stored by name in the
        // overflow map, so it takes no id.
        let takes_id = |name: &str| {
            !matches!(mode, SchemaMode::Validated)
                || schema
                    .as_ref()
                    .is_some_and(|s| s.get_property(name).is_some())
        };
        let registered: Vec<&str> = properties
            .iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| takes_id(n))
            .collect();
        let mut registered_ids = ctx.field_ids(&registered)?.into_iter();
        // Aligned with `properties`; the reserved id marks a name stored by
        // name, whose branch never reads it.
        let field_ids: Vec<u32> = properties
            .iter()
            .map(|(n, _)| {
                if takes_id(n) {
                    registered_ids.next().unwrap_or(FieldInterner::RESERVED_ID)
                } else {
                    FieldInterner::RESERVED_ID
                }
            })
            .collect();
        for ((prop_name, expr), field_id) in properties.iter().zip(field_ids) {
            // Map literals → Document for full dot-notation support in storage.
            let val = eval_neutral(expr, input_row)?.map_to_document();

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
                            // Declared non-computed property → validate type, then set.
                            validate_one(prop_name, &val, def)
                                .map_err(|e| ExecutionError::SchemaViolation(e.to_string()))?;
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
                            // Declared non-computed property → validate type, then set.
                            validate_one(prop_name, &val, def)
                                .map_err(|e| ExecutionError::SchemaViolation(e.to_string()))?;
                            record.set(field_id, val.clone());
                        }
                    }
                }
                SchemaMode::Flexible => {
                    // No schema enforcement: set unconditionally.
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

        // Temporal storage path: when the (primary or any) label is
        // TEMPORAL, extract `valid_from` from the supplied properties and
        // emit the per-version key. Auto-populate `__ingestion_ts__` from
        // the current HLC commit timestamp so the bitemporal system-axis is
        // queryable without an additional lookup. Mirror of the temporal-
        // edge write path in `execute_create_edge`.
        let valid_from_for_key: Option<i64> = if let Some(ref tlabel) = temporal_label {
            let mut vf: Option<i64> = None;
            for (prop_name, expr) in properties {
                if prop_name == "valid_from" {
                    let val = eval_neutral(expr, input_row)?;
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
                                 TIMESTAMP (epoch microseconds, as `now()` returns; \
                                 `timestamp()` is milliseconds and will not compare \
                                 against engine-assigned versions), got {other:?}"
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
                    let val = eval_neutral(expr, input_row)?;
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
            let field_id = ctx.field_id("__ingestion_ts__")?;
            record.set(field_id, Value::Int(ingestion_us));
        }

        // Index entries, in the statement transaction, from the values as
        // stored, under the version for a temporal label; a unique value
        // another node holds refuses the write.
        ctx.index_node_created(node_id, valid_from_for_key, &record)?;

        if !row_key.is_empty() {
            ctx.claim_table_key(&table_label, row_key, node_id)?;
        }

        if table_columnar {
            // COLUMNAR table row → its own columnar tree (read back by the
            // columnar scan branch of NodeScan), not the node path.
            ctx.columnar_put_node(&table_label, node_id, &record)?;
        } else {
            // `valid_from_for_key` is `Some(_)` for temporal labels and
            // `None` otherwise — the typed dispatch helper picks the
            // 25-byte temporal key or the 16-byte non-temporal key.
            ctx.mvcc_put_node_either(ctx.shard_id, node_id, valid_from_for_key, &record)?;
        }
        ctx.write_stats.nodes_created += 1;
        ctx.stat_node_created(&record);
        ctx.write_stats.properties_set += properties.len() as u64;

        // Fire BEFORE COMMIT triggers registered on any of the new node's
        // labels for the CREATE event. Triggers fire after the node write
        // is staged in the MVCC buffer so a trigger body can MATCH the new
        // node via read-your-own-writes. On a trigger error (ON ERROR
        // PROPAGATE — the BEFORE-COMMIT default) the surrounding error
        // path drops the buffer and the node is never persisted. AFTER
        // COMMIT triggers are skipped here; they fire through the async
        // oplog-consumer path once that lands.
        if !labels.is_empty() {
            let mut props_map: std::collections::HashMap<String, Value> =
                std::collections::HashMap::with_capacity(properties.len());
            for (name, expr) in properties.iter() {
                let val = eval_neutral(expr, input_row)?.map_to_document();
                props_map.insert(name.clone(), val);
            }
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
        insert_label_columns(&mut row, var_name, &record);
        for (prop_name, expr) in properties {
            let val = eval_neutral(expr, input_row)?;
            row.insert(format!("{var_name}.{prop_name}"), val);
        }

        results.push(row);
    }

    // Persist a columnar table's rows once per statement (not per row): the
    // runtime columnar tree is not registered with the background flush
    // manager, so flush its memtable here to survive a clean restart.
    if table_columnar {
        ctx.engine.flush_columnar_table(&table_label)?;
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
    properties: &[(String, crate::plan::expr::Expr)],
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

    // An end that names no column was never bound by this statement: nothing
    // upstream matched or created it. Skipping the row would report success
    // with the nodes written and the relationship missing.
    let unbound = |end: &str| {
        ExecutionError::Unsupported(format!(
            "CREATE relationship of type '{edge_type}': end '{end}' is not bound to a node; \
             bind it with MATCH, or give it a label or properties so CREATE makes it"
        ))
    };

    for row in input_rows {
        // Get source and target node IDs from the row. A bound end holding no
        // node (an OPTIONAL MATCH that found nothing) adds no edge for the row.
        let source_id = match row.get(source) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            Some(_) => continue,
            None => return Err(unbound(source)),
        };
        let target_id = match row.get(target) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            Some(_) => continue,
            None => return Err(unbound(target)),
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
            // Reject writes to reserved metadata field names that would
            // otherwise collide with engine-internal row columns, before any
            // name of the edge is registered.
            if let Some((prop_name, _)) = properties
                .iter()
                .find(|(n, _)| matches!(n.as_str(), "__src__" | "__tgt__" | "__type__"))
            {
                return Err(ExecutionError::Unsupported(format!(
                    "edge property name '{prop_name}' is reserved for engine-internal \
                     metadata; choose a different name"
                )));
            }
            let names: Vec<&str> = properties.iter().map(|(n, _)| n.as_str()).collect();
            let field_ids = ctx.field_ids(&names)?;
            for ((prop_name, expr), field_id) in properties.iter().zip(field_ids) {
                let value = eval_neutral(expr, row)?.map_to_document();
                if is_temporal && prop_name == "valid_from" {
                    // Accept both Int and Timestamp, both read as epoch
                    // microseconds: the value goes into the key unconverted, so
                    // the caller's scale is the stored scale.
                    // Reject Null (explicit null violates the temporal contract)
                    // and any other type.
                    valid_from_value = match &value {
                        Value::Int(us) => Some(*us),
                        Value::Timestamp(us) => Some(*us),
                        Value::Null => {
                            return Err(ExecutionError::Unsupported(format!(
                                "temporal edge '{edge_type}': valid_from must not be NULL"
                            )));
                        }
                        other => {
                            return Err(ExecutionError::Unsupported(format!(
                                "temporal edge '{edge_type}': valid_from must be INT or \
                                 TIMESTAMP (epoch microseconds, as `now()` returns; \
                                 `timestamp()` is milliseconds and will not compare \
                                 against microsecond versions), got {other:?}"
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
        let mut resolved_props: Vec<(String, Value)> = Vec::with_capacity(properties.len());
        for (name, expr) in properties.iter() {
            resolved_props.push((name.clone(), eval_neutral(expr, row)?.map_to_document()));
        }
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
                let value = eval_neutral(expr, row)?;
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
    items: &[crate::plan::SetItem],
    violation_mode: &ViolationMode,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let skip_on_violation = matches!(violation_mode, ViolationMode::Skip);
    for row in input_rows {
        refuse_key_changes_in_set(row, items, ctx)?;
    }

    let mut results = Vec::new();

    // Temporal-node SET routing. For each SET item targeting a node
    // on a temporal label, classify by property:
    //
    //   * `valid_from` → reject (immutable storage-key suffix).
    //   * `valid_to`   → mutate in place at the matched per-version key
    //     (the close-version path).
    //   * Other property / label mutations → write a NEW version row at
    //     `valid_from = NOW` (the close+open path: close current, open new).
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
                crate::plan::SetItem::Property { variable, .. }
                | crate::plan::SetItem::PropertyPath { variable, .. }
                | crate::plan::SetItem::DocFunction { variable, .. }
                | crate::plan::SetItem::ReplaceProperties { variable, .. }
                | crate::plan::SetItem::MergeProperties { variable, .. }
                | crate::plan::SetItem::AddLabel { variable, .. } => variable,
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
                crate::plan::SetItem::Property { property, .. } if property == "valid_from" => {
                    return Err(ExecutionError::Unsupported(format!(
                        "SET {var}.valid_from is rejected on temporal label '{primary}': \
                         valid_from is the version-key suffix and is immutable. To \
                         re-key a version, DELETE the row and CREATE a new one with \
                         the desired valid_from."
                    )));
                }
                crate::plan::SetItem::Property { property, .. } if property == "valid_to" => {
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

        // Close+open processing for temporal nodes mutated by non-`valid_to`
        // SET items. This block fires
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
                // The new version opens at the statement's NOW, the instant
                // its reads saw the matched version valid at, and strictly
                // after that version's valid_from so its interval is valid.
                // A version that starts at or after NOW (backfill / replay)
                // is followed 1 µs later.
                let new_valid_from = if ctx.valid_now > current_valid_from {
                    ctx.valid_now
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
                let label = closing_record.primary_label().to_string();
                let label_schema = ctx.load_current_label_schema(&label)?;
                // The same admission as a SET of a non-temporal node, before
                // anything is written: a refused item leaves the node with
                // exactly the versions it had.
                let violation = |property: &str, value: Option<&Value>| {
                    refuse_engine_temporal_field(property).map(|()| {
                        set_property_violation(label_schema.as_ref(), &label, property, value)
                    })
                };

                // Step 2: apply each relevant SET item to the new record.
                // Mark each item's index in `processed_temporal_items` so
                // the main SET loop skips it for this row.
                for (item_idx, item) in items.iter().enumerate() {
                    let item_var = match item {
                        crate::plan::SetItem::Property { variable, .. }
                        | crate::plan::SetItem::PropertyPath { variable, .. }
                        | crate::plan::SetItem::DocFunction { variable, .. }
                        | crate::plan::SetItem::ReplaceProperties { variable, .. }
                        | crate::plan::SetItem::MergeProperties { variable, .. }
                        | crate::plan::SetItem::AddLabel { variable, .. } => variable.as_str(),
                    };
                    if item_var != var.as_str() {
                        continue;
                    }
                    // Skip valid_to (in-place) and valid_from (rejected at pre-scan).
                    if let crate::plan::SetItem::Property { property, .. } = item {
                        if property == "valid_to" || property == "valid_from" {
                            continue;
                        }
                    }
                    match item {
                        crate::plan::SetItem::Property { property, expr, .. } => {
                            let val = eval_neutral(expr, &out_row)?.map_to_document();
                            if let Some(err) = violation(property, Some(&val))? {
                                if skip_on_violation {
                                    continue 'row_loop;
                                }
                                return Err(err);
                            }
                            let by_name = stored_by_name(label_schema.as_ref(), property);
                            store_node_property(&mut new_record, property, val, by_name, ctx)?;
                        }
                        crate::plan::SetItem::AddLabel { label, .. } => {
                            new_record.add_label(label.clone());
                        }
                        crate::plan::SetItem::ReplaceProperties { expr, .. } => {
                            let val = eval_neutral(expr, &out_row)?;
                            if let Value::Map(map) = val {
                                for (k, v) in &map {
                                    if let Some(err) = violation(k, Some(v))? {
                                        if skip_on_violation {
                                            continue 'row_loop;
                                        }
                                        return Err(err);
                                    }
                                }
                                // Clear existing user props, keep engine-managed
                                // fields (__ingestion_ts__, valid_from, valid_to
                                // are reapplied below).
                                new_record.props.clear();
                                new_record.extra = None;
                                register_stored_ids(label_schema.as_ref(), &map, ctx)?;
                                for (name, v) in map {
                                    let by_name = stored_by_name(label_schema.as_ref(), &name);
                                    store_node_property(
                                        &mut new_record,
                                        &name,
                                        v.map_to_document(),
                                        by_name,
                                        ctx,
                                    )?;
                                }
                            }
                        }
                        crate::plan::SetItem::MergeProperties { expr, .. } => {
                            let val = eval_neutral(expr, &out_row)?;
                            if let Value::Map(map) = val {
                                for (k, v) in &map {
                                    if let Some(err) = violation(k, Some(v))? {
                                        if skip_on_violation {
                                            continue 'row_loop;
                                        }
                                        return Err(err);
                                    }
                                }
                                register_stored_ids(label_schema.as_ref(), &map, ctx)?;
                                for (name, v) in map {
                                    let by_name = stored_by_name(label_schema.as_ref(), &name);
                                    store_node_property(
                                        &mut new_record,
                                        &name,
                                        v.map_to_document(),
                                        by_name,
                                        ctx,
                                    )?;
                                }
                            }
                        }
                        crate::plan::SetItem::PropertyPath { path, expr, .. } => {
                            // Nested PropertyPath SET on temporal. Build
                            // the same DocDelta the
                            // non-temporal path queues as a merge operand,
                            // but apply it in-memory to `new_record` so the
                            // close+open writes carry the post-delta state.
                            let val = eval_neutral(expr, &out_row)?.map_to_document();
                            if path.is_empty() {
                                return Err(ExecutionError::Unsupported(format!(
                                    "SET on temporal node `{var}`: empty property path"
                                )));
                            }
                            if let Some(err) = violation(&path[0], None)? {
                                if skip_on_violation {
                                    continue 'row_loop;
                                }
                                return Err(err);
                            }
                            let (target, sub_path) =
                                document_target(label_schema.as_ref(), path, ctx)?;
                            let delta = coordinode_core::graph::doc_delta::DocDelta::SetPath {
                                target,
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
                        crate::plan::SetItem::DocFunction {
                            function,
                            path,
                            value_expr,
                            ..
                        } => {
                            // doc_push / doc_pull / doc_add_to_set /
                            // doc_inc on temporal nodes.
                            // Same construction as the non-temporal path,
                            // but applied in-memory to `new_record`.
                            let val = eval_neutral(value_expr, &out_row)?;
                            // A bare variable names a property of its own name.
                            let full_path = if path.is_empty() {
                                std::slice::from_ref(var)
                            } else {
                                path.as_slice()
                            };
                            if let Some(err) = violation(&full_path[0], None)? {
                                if skip_on_violation {
                                    continue 'row_loop;
                                }
                                return Err(err);
                            }
                            let (target, sub_path) =
                                document_target(label_schema.as_ref(), full_path, ctx)?;
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
                let [vf_fid, vt_fid, its_fid] = temporal_field_ids(ctx)?;
                new_record.set(vf_fid, Value::Int(new_valid_from));
                new_record.props.remove(&vt_fid);
                new_record.set(its_fid, Value::Int(now_us));

                // Step 3: close-current at its per-version key.
                ctx.close_temporal_version(
                    node_id,
                    current_valid_from,
                    &mut closing_record,
                    new_valid_from,
                )?;

                // Step 4: open-new at the fresh per-version key for NOW.
                ctx.open_temporal_version(node_id, new_valid_from, &new_record)?;
                ctx.write_stats.nodes_created += 1;
                // One new version ROW: the statistics counters track stored
                // rows (what a partition scan would count), so a temporal
                // version bump increments the label counts like a create.
                ctx.stat_node_created(&new_record);

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
            // Skip items already processed in the close+open
            // pass above. Their snapshot is implicitly the pre-mutation
            // record (which still exists at the matched per-version key
            // until we rewrote its valid_to). For UPDATE-trigger purposes
            // the closing snapshot is captured by the close+open block.
            if processed_temporal_items.contains(&snap_item_idx) {
                continue;
            }
            let variable = match item {
                crate::plan::SetItem::Property { variable, .. }
                | crate::plan::SetItem::PropertyPath { variable, .. }
                | crate::plan::SetItem::DocFunction { variable, .. }
                | crate::plan::SetItem::ReplaceProperties { variable, .. }
                | crate::plan::SetItem::MergeProperties { variable, .. }
                | crate::plan::SetItem::AddLabel { variable, .. } => variable,
            };
            // Edge mutation flows through `SetItem::Property` and the two
            // map assignments. The remaining variants (`PropertyPath`,
            // `DocFunction`, `AddLabel`) require `Value::Int` and no-op on
            // edge variables; snapshotting those would fire an UPDATE trigger
            // with `$before == $after`, semantically wrong (no mutation
            // happened) and a needless round trip through the trigger body.
            let is_property_mutation = matches!(
                item,
                crate::plan::SetItem::Property { .. }
                    | crate::plan::SetItem::MergeProperties { .. }
                    | crate::plan::SetItem::ReplaceProperties { .. }
            );
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
            // Skip items already handled by the close+open
            // path at the top of this row's processing. Their property
            // values are already in the new version's record and the
            // output row was updated to reflect that state.
            if processed_temporal_items.contains(&item_idx) {
                continue;
            }
            match item {
                crate::plan::SetItem::Property {
                    variable,
                    property,
                    expr,
                } => {
                    // Reserved-name guard at SET time: the engine's temporal
                    // fields are engine-owned on temporal labels (user-immutable).
                    // Reject before any storage mutation so the bitemporal
                    // contract cannot be subverted via `SET n.__ingestion_ts__
                    // = ...`. Edge metadata names
                    // (`__src__`/`__tgt__`/`__type__`) are rejected separately
                    // in `update_edge_property` for the edge SET path.
                    refuse_engine_temporal_field(property)?;

                    // Map literals → Document for nested property storage.
                    let val = eval_neutral(expr, &out_row)?.map_to_document();

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
                        out_row.insert(format!("{variable}.{property}"), eval_neutral(expr, row)?);
                        continue;
                    }

                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    // Temporal node close-version path: when the label
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
                        match new_valid_to {
                            Some(vt) => {
                                ctx.close_temporal_version(node_id, valid_from, &mut record, vt)?;
                            }
                            None => {
                                // Re-open a closed version: drop the
                                // `valid_to` field entirely so the version
                                // becomes open again.
                                let field_id = ctx.field_id("valid_to")?;
                                let closed = ctx
                                    .indexes_label(record.primary_label())
                                    .then(|| record.clone());
                                record.props.remove(&field_id);
                                ctx.mvcc_put_node_temporal(
                                    ctx.shard_id,
                                    node_id,
                                    valid_from,
                                    &record,
                                )?;
                                if let Some(closed) = closed {
                                    ctx.index_version_changed(
                                        node_id, valid_from, &closed, &record,
                                    )?;
                                }
                            }
                        }
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

                    // Read the node as this statement leaves it so far, with
                    // the property deltas of earlier items still pending.
                    if let Some((record, stored_bytes)) =
                        ctx.mvcc_node_post_state_sized(ctx.shard_id, node_id)?
                    {
                        // Enforce schema mode before writing the property.
                        // Load the label schema for the node's primary label and
                        // check STRICT/VALIDATED constraints on the property name.
                        // Returns None if no schema exists (schemaless node → always allowed).
                        let label = record.primary_label().to_string();
                        let label_schema = ctx.load_current_label_schema(&label)?;

                        // Collect potential schema violation into Option<ExecutionError> so we can
                        // choose between skip (ON VIOLATION SKIP) and fail (default) after the check.
                        let schema_err = set_property_violation(
                            label_schema.as_ref(),
                            &label,
                            property,
                            Some(&val),
                        );

                        // ON VIOLATION SKIP: silently drop this row and move to next.
                        // Default (Fail): propagate the error immediately.
                        if let Some(err) = schema_err {
                            if skip_on_violation {
                                continue 'row_loop;
                            }
                            return Err(err);
                        }

                        // Move the node's index entries from the old value to
                        // the new one before the record changes; a unique
                        // value another node holds refuses the write.
                        ctx.index_property_changed(node_id, &record, property, Some(&val))?;

                        let by_name = stored_by_name(label_schema.as_ref(), property);
                        write_property_changes(
                            ctx,
                            node_id,
                            &record,
                            stored_bytes,
                            &[(property.as_str(), val.clone(), by_name)],
                        )?;
                        ctx.write_stats.properties_set += 1;

                        // Reflect the new value in the output row only when the
                        // write was actually applied. If mvcc_get returned None
                        // (e.g. node was DELETEd earlier in the same query), the
                        // `if let Some` block above is skipped and we must not
                        // expose the unapplied value through RETURN.
                        out_row.insert(format!("{variable}.{property}"), val);
                    }
                }
                crate::plan::SetItem::PropertyPath {
                    variable,
                    path,
                    expr,
                } => {
                    let val = eval_neutral(expr, &out_row)?;
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
                    let label = ctx.schema_label_for_node(ctx.shard_id, node_id)?;
                    let label_schema = match &label {
                        Some(label) => ctx.load_current_label_schema(label)?,
                        None => None,
                    };
                    let schema_err = set_property_violation(
                        label_schema.as_ref(),
                        label.as_deref().unwrap_or_default(),
                        &path[0],
                        None,
                    );
                    if let Some(err) = schema_err {
                        if skip_on_violation {
                            continue 'row_loop;
                        }
                        return Err(err);
                    }

                    // O(1) write via merge operand: an overflow root by its
                    // full path in the overflow map, any other by its id and
                    // the path below it.
                    let (target, sub_path) = document_target(label_schema.as_ref(), path, ctx)?;
                    let delta = coordinode_core::graph::doc_delta::DocDelta::SetPath {
                        target,
                        path: sub_path,
                        value: val.to_rmpv(),
                    };
                    let operand = delta.encode().map_err(|e| {
                        ExecutionError::Serialization(format!("DocDelta encode: {e}"))
                    })?;
                    ctx.mvcc_merge_node_delta(ctx.shard_id, node_id, operand)?;
                    ctx.write_stats.properties_set += 1;

                    let path_str = path.join(".");
                    out_row.insert(format!("{variable}.{path_str}"), val);
                }
                crate::plan::SetItem::DocFunction {
                    function,
                    variable,
                    path,
                    value_expr,
                } => {
                    let val = eval_neutral(value_expr, &out_row)?;
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
                    let label = ctx.schema_label_for_node(ctx.shard_id, node_id)?;
                    let label_schema = match &label {
                        Some(label) => ctx.load_current_label_schema(label)?,
                        None => None,
                    };
                    let schema_err = set_property_violation(
                        label_schema.as_ref(),
                        label.as_deref().unwrap_or_default(),
                        root_prop,
                        None,
                    );
                    if let Some(err) = schema_err {
                        if skip_on_violation {
                            continue 'row_loop;
                        }
                        return Err(err);
                    }

                    // path[0] is the root property, path[1..] the nested path;
                    // a bare variable (doc_push(n, "x")) names a property of
                    // its own name.
                    let full_path = if path.is_empty() {
                        std::slice::from_ref(variable)
                    } else {
                        path.as_slice()
                    };
                    let (target, sub_path) =
                        document_target(label_schema.as_ref(), full_path, ctx)?;

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
                    ctx.mvcc_merge_node_delta(ctx.shard_id, node_id, operand)?;
                    ctx.write_stats.properties_set += 1;
                }
                crate::plan::SetItem::ReplaceProperties { variable, expr } => {
                    let map_val = eval_neutral(expr, &out_row)?;
                    if let Some(Value::String(edge_type)) = out_row.get(variable).cloned() {
                        update_edge_properties_from_map(
                            variable,
                            &edge_type,
                            &map_val,
                            EdgeMapAssign::Replace,
                            &mut out_row,
                            ctx,
                        )?;
                        continue;
                    }
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    if let Some(mut record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                        // Schema validation: SET n = {map} replaces ALL properties.
                        // In STRICT mode every key must be declared; VALIDATED checks declared keys only.
                        let label = record.primary_label().to_string();
                        let label_schema = ctx.load_current_label_schema(&label)?;
                        if let Value::Map(ref map) = map_val {
                            let schema_err = map.iter().find_map(|(k, v)| {
                                set_property_violation(label_schema.as_ref(), &label, k, Some(v))
                            });
                            if let Some(err) = schema_err {
                                if skip_on_violation {
                                    continue 'row_loop;
                                }
                                return Err(err);
                            }
                        }

                        // Every property goes, the overflow ones too; the
                        // map's are stored where the schema keeps each.
                        let before = ctx
                            .indexes_label(record.primary_label())
                            .then(|| record.clone());
                        record.props.clear();
                        record.extra = None;
                        if let Value::Map(ref map) = map_val {
                            register_stored_ids(label_schema.as_ref(), map, ctx)?;
                            for (name, v) in map {
                                let by_name = stored_by_name(label_schema.as_ref(), name);
                                store_node_property(&mut record, name, v.clone(), by_name, ctx)?;
                            }
                        }
                        if let Some(before) = before {
                            // Every property the node had or now has may move
                            // an index entry.
                            let mut changed = property_names(&before, ctx.interner);
                            if let Value::Map(ref map) = map_val {
                                changed.extend(map.keys().cloned());
                            }
                            changed.sort_unstable();
                            changed.dedup();
                            let changed: Vec<&str> = changed.iter().map(String::as_str).collect();
                            ctx.index_record_changed(node_id, &before, &record, &changed)?;
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
                    }
                }
                crate::plan::SetItem::MergeProperties { variable, expr } => {
                    let map_val = eval_neutral(expr, &out_row)?;
                    // Relationship variables bind to Value::String(edge_type),
                    // never to a node id, so they have to leave here: the node
                    // path below would drop the write on the floor.
                    if let Some(Value::String(edge_type)) = out_row.get(variable).cloned() {
                        update_edge_properties_from_map(
                            variable,
                            &edge_type,
                            &map_val,
                            EdgeMapAssign::Merge,
                            &mut out_row,
                            ctx,
                        )?;
                        continue;
                    }
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    // The node as the statement leaves it so far, with earlier
                    // property deltas still pending.
                    if let Some((record, stored_bytes)) =
                        ctx.mvcc_node_post_state_sized(ctx.shard_id, node_id)?
                    {
                        // Schema validation: SET n += {map} merges new properties.
                        // STRICT: every key in map must be declared; VALIDATED: checks declared keys.
                        let label = record.primary_label().to_string();
                        let label_schema = ctx.load_current_label_schema(&label)?;
                        if let Value::Map(ref map) = map_val {
                            let schema_err = map.iter().find_map(|(k, v)| {
                                set_property_violation(label_schema.as_ref(), &label, k, Some(v))
                            });
                            if let Some(err) = schema_err {
                                if skip_on_violation {
                                    continue 'row_loop;
                                }
                                return Err(err);
                            }
                        }

                        // The keys of the map are written; the properties it
                        // does not name are left as they are.
                        if let Value::Map(ref map) = map_val {
                            register_stored_ids(label_schema.as_ref(), map, ctx)?;
                            if ctx.indexes_label(record.primary_label()) {
                                let mut after = record.clone();
                                for (name, v) in map {
                                    let by_name = stored_by_name(label_schema.as_ref(), name);
                                    store_node_property(&mut after, name, v.clone(), by_name, ctx)?;
                                }
                                let changed: Vec<&str> = map.keys().map(String::as_str).collect();
                                ctx.index_record_changed(node_id, &record, &after, &changed)?;
                            }
                            let changes: Vec<(&str, Value, bool)> = map
                                .iter()
                                .map(|(name, v)| {
                                    (
                                        name.as_str(),
                                        v.clone(),
                                        stored_by_name(label_schema.as_ref(), name),
                                    )
                                })
                                .collect();
                            write_property_changes(ctx, node_id, &record, stored_bytes, &changes)?;
                        }

                        // Update out_row so RETURN clauses see the merged values.
                        // MergeProperties adds/overwrites; existing untouched props stay.
                        if let Value::Map(ref map) = map_val {
                            for (k, v) in map {
                                out_row.insert(format!("{variable}.{k}"), v.clone());
                            }
                        }
                    }
                }
                crate::plan::SetItem::AddLabel { variable, label } => {
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    if let Some(mut record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                        let newly_added = !record.has_label(label);
                        record.add_label(label.clone());
                        ctx.write_stats.labels_added += 1;
                        // Statistics counters: only a label the node did not
                        // already carry changes its cardinality.
                        if newly_added {
                            use coordinode_modality::{LocalStatsStore, StatsStore as _};
                            LocalStatsStore.label_added(&mut ctx.txn, label);
                        }

                        ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;

                        refresh_label_columns(&mut out_row, variable, &record);
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
    items: &[crate::plan::RemoveItem],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // REMOVE on a temporal node is a *new version* — the same close+open
    // path as a temporal SET. Removing a property
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
    //   * REMOVE `n.<path.to.nested>` (PropertyPath) → close current + open
    //     new, with the path removed by a DocDelta applied in memory to
    //     the new record (the merge-operand path is keyed on the
    //     non-temporal key).
    for row in input_rows {
        refuse_key_changes_in_remove(row, items, ctx)?;
    }
    let mut temporal_remove_vars: std::collections::HashSet<(usize, String)> =
        std::collections::HashSet::new();
    for (row_idx, row) in input_rows.iter().enumerate() {
        for item in items {
            let var = match item {
                crate::plan::RemoveItem::Property { variable, .. }
                | crate::plan::RemoveItem::PropertyPath { variable, .. }
                | crate::plan::RemoveItem::Label { variable, .. } => variable,
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
                crate::plan::RemoveItem::Property { property, .. } => {
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
                crate::plan::RemoveItem::PropertyPath { .. } => {
                    // Classify; the delta is built and
                    // applied to `new_record` in the close+open block.
                    temporal_remove_vars.insert((row_idx, var.clone()));
                }
                crate::plan::RemoveItem::Label { .. } => {
                    temporal_remove_vars.insert((row_idx, var.clone()));
                }
            }
        }
    }

    let mut results = Vec::new();

    for (row_idx, row) in input_rows.iter().enumerate() {
        let mut out_row = row.clone();

        // Close+open processing for temporal nodes whose REMOVE pre-scan
        // classified as new-version. Mirrors the temporal SET block — same
        // shape, same invariants. Items
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
                // Opens at the statement's NOW, as the temporal SET does.
                let new_valid_from = if ctx.valid_now > current_valid_from {
                    ctx.valid_now
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
                let remove_label_schema =
                    ctx.load_current_label_schema(closing_record.primary_label())?;

                for (item_idx, item) in items.iter().enumerate() {
                    let item_var = match item {
                        crate::plan::RemoveItem::Property { variable, .. }
                        | crate::plan::RemoveItem::PropertyPath { variable, .. }
                        | crate::plan::RemoveItem::Label { variable, .. } => variable.as_str(),
                    };
                    if item_var != var.as_str() {
                        continue;
                    }
                    match item {
                        crate::plan::RemoveItem::Property { property, .. } => {
                            remove_node_property(&mut new_record, property, ctx.interner);
                            ctx.write_stats.properties_removed += 1;
                        }
                        crate::plan::RemoveItem::Label { label, .. } => {
                            new_record.remove_label(label);
                            ctx.write_stats.labels_removed += 1;
                        }
                        crate::plan::RemoveItem::PropertyPath { path, .. } => {
                            // Nested REMOVE on temporal —
                            // build DeletePath DocDelta, apply in-memory.
                            if path.is_empty() {
                                return Err(ExecutionError::Unsupported(format!(
                                    "REMOVE on temporal node `{var}`: empty property path"
                                )));
                            }
                            if path.len() == 1 {
                                remove_node_property(&mut new_record, &path[0], ctx.interner);
                            } else if let Some((target, sub_path)) =
                                removal_target(remove_label_schema.as_ref(), path, ctx.interner)
                            {
                                // A name with no binding is on no record:
                                // nothing to remove, and nothing registered.
                                let delta =
                                    coordinode_core::graph::doc_delta::DocDelta::DeletePath {
                                        target,
                                        path: sub_path,
                                    };
                                coordinode_storage::engine::merge::apply_doc_deltas_to_record(
                                    &mut new_record,
                                    &[delta],
                                );
                            }
                            ctx.write_stats.properties_removed += 1;
                        }
                    }
                    processed_temporal_items.insert(item_idx);
                }

                let [vf_fid, vt_fid, its_fid] = temporal_field_ids(ctx)?;
                new_record.set(vf_fid, Value::Int(new_valid_from));
                new_record.props.remove(&vt_fid);
                new_record.set(its_fid, Value::Int(now_us));

                ctx.close_temporal_version(
                    node_id,
                    current_valid_from,
                    &mut closing_record,
                    new_valid_from,
                )?;
                ctx.open_temporal_version(node_id, new_valid_from, &new_record)?;
                // One new version ROW (row-count semantics, same as the SET
                // close-current + open-new path).
                ctx.stat_node_created(&new_record);

                // Surface the new version's valid_from on out_row so any
                // downstream RETURN sees the latest version, symmetric with
                // the temporal SET path.
                out_row.insert(format!("{var}.valid_from"), Value::Int(new_valid_from));
                if new_record.labels != closing_record.labels {
                    refresh_label_columns(&mut out_row, var, &new_record);
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
                crate::plan::RemoveItem::Property { variable, .. }
                | crate::plan::RemoveItem::PropertyPath { variable, .. }
                | crate::plan::RemoveItem::Label { variable, .. } => variable,
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
                crate::plan::RemoveItem::Property { variable, property } => {
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    if let Some(mut record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                        let stored = ctx
                            .interner
                            .lookup(property)
                            .is_some_and(|field_id| record.props.contains_key(&field_id))
                            || record.get_extra(property).is_some();
                        if stored {
                            // The node's index entries move to the property's
                            // absence: the old value's entry goes, and an
                            // index that keeps missing values gets one.
                            ctx.index_property_changed(node_id, &record, property, None)?;
                            remove_node_property(&mut record, property, ctx.interner);
                            ctx.write_stats.properties_removed += 1;
                        }

                        ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;
                    }

                    out_row.remove(&format!("{variable}.{property}"));
                }
                crate::plan::RemoveItem::PropertyPath { variable, path } => {
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    // O(1) delete via merge operand; the schema says where the
                    // root is stored. A name with no binding is on no record,
                    // so there is nothing to delete and nothing is registered.
                    let label_schema = match ctx.schema_label_for_node(ctx.shard_id, node_id)? {
                        Some(label) => ctx.load_current_label_schema(&label)?,
                        None => None,
                    };
                    if let Some((target, sub_path)) =
                        removal_target(label_schema.as_ref(), path, ctx.interner)
                    {
                        let delta = coordinode_core::graph::doc_delta::DocDelta::DeletePath {
                            target,
                            path: sub_path,
                        };
                        let operand = delta.encode().map_err(|e| {
                            ExecutionError::Serialization(format!("DocDelta encode: {e}"))
                        })?;
                        ctx.mvcc_merge_node_delta(ctx.shard_id, node_id, operand)?;
                    }
                    ctx.write_stats.properties_removed += 1;

                    let path_str = path.join(".");
                    out_row.remove(&format!("{variable}.{path_str}"));
                }
                crate::plan::RemoveItem::Label { variable, label } => {
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    if let Some(mut record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                        let was_present = record.remove_label(label);
                        ctx.write_stats.labels_removed += 1;
                        // Statistics counters: only an actually-removed label
                        // changes its cardinality.
                        if was_present {
                            use coordinode_modality::{LocalStatsStore, StatsStore as _};
                            LocalStatsStore.label_removed(&mut ctx.txn, label);
                        }

                        ctx.mvcc_put_node(ctx.shard_id, node_id, &record)?;

                        refresh_label_columns(&mut out_row, variable, &record);
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

    // DELETE on temporal nodes is a *positive bitemporal fact* (the XTDB
    // pattern). Instead of hard-deleting the
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
    // Hard erasure across ALL versions is a separate privileged erase
    // operation, never a modifier on DELETE: a tombstone is the only
    // DELETE mode on temporal labels.
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
            // The tombstone opens at the statement's NOW, as the temporal
            // SET's new version does.
            let new_valid_from = if ctx.valid_now > *current_valid_from {
                ctx.valid_now
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
            let [vf_fid, vt_fid, its_fid] = temporal_field_ids(ctx)?;
            let deleted_fid = ctx.field_id("__deleted__")?;
            let was_open = !closing_record.props.contains_key(&vt_fid);
            if was_open {
                ctx.close_temporal_version(
                    *node_id,
                    *current_valid_from,
                    &mut closing_record,
                    new_valid_from,
                )?;
            }

            // Step 2: write the tombstone row at NOW. Carries the same
            // labels as the closing record (so downstream MATCH still
            // sees the row under the right label), no user properties,
            // `__deleted__: true`, refreshed `__ingestion_ts__`. The
            // tombstone has `valid_from = NOW`, `valid_to = NULL` —
            // the deletion is "current" from NOW onward.
            let mut tombstone = NodeRecord::with_labels(closing_record.labels.clone());
            tombstone.set(deleted_fid, Value::Bool(true));
            tombstone.set(vf_fid, Value::Int(new_valid_from));
            tombstone.set(its_fid, Value::Int(now_us));
            ctx.mvcc_put_node_temporal(ctx.shard_id, *node_id, new_valid_from, &tombstone)?;
            ctx.write_stats.nodes_deleted += 1;
            // The statistics counters track stored ROWS (scan cost), and a
            // temporal delete ADDS a tombstone row carrying the labels, so
            // the label counts go up by one row here, not down.
            ctx.stat_node_created(&tombstone);

            tombstoned_temporal_rows.insert((*row_idx, var.clone()));
        }
    }

    for (row_idx, row) in input_rows.iter().enumerate() {
        for var in variables {
            // If this (row, var) was tombstoned above as a temporal-node
            // positive bitemporal fact, skip the hard-delete path entirely
            // — the tombstone IS the delete. Edges connected to a temporal
            // node are intentionally left intact: at past valid times the
            // node still existed, so its edges are still part of history.
            // DETACH DELETE on a temporal node does not cascade to them.
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

            // The node's B-tree entries go with it, so its unique values are
            // free for a new node. Vector and text indexes follow the
            // committed deletion on their own.
            if ctx.btree_index_registry.is_some() {
                if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                    ctx.index_node_deleted(node_id, &record)?;
                }
            }
            if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
                ctx.release_table_key(&record)?;
            }
            ctx.mvcc_delete_node(ctx.shard_id, node_id)?;
            ctx.write_stats.nodes_deleted += 1;
            // Statistics counters: decrement by the pre-delete labels. A
            // delete of an absent node removes no row, so no decrement.
            if let Some((labels, _)) = &pre_snapshot {
                use coordinode_modality::{LocalStatsStore, StatsStore as _};
                LocalStatsStore.node_deleted(&mut ctx.txn, labels.iter().map(String::as_str));
            }

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
// MERGE NODES
// =====================================================================

/// Execute a `MERGE NODES (a, b) INTO target` clause for each input row.
///
/// Collapses the non-surviving source into the surviving target within a
/// single MVCC transaction. The semantic contract is documented with
/// `MERGE NODES` in `docs/cypher/extensions.md`.
#[allow(clippy::too_many_arguments)]
fn execute_merge_nodes(
    input_rows: &[Row],
    source_a: &str,
    source_b: &str,
    target: &str,
    conflict: &crate::plan::MergeNodesConflictStrategy,
    transfer_edges: Option<&crate::plan::TransferEdgesEndpoints>,
    duplicate: &crate::plan::MergeNodesDuplicateStrategy,
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

    // Safe-reject for temporal labels: MERGE NODES reads source nodes via
    // 16-byte `encode_node_key`, mutates the target record, and deletes
    // the non-survivor via `detach_delete_node` — none of these paths are
    // temporal-aware. Merging two temporal nodes (whether the intent is
    // "fold version histories" or "treat all versions as one logical
    // node") needs a per-version write path.
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
                        "MERGE NODES on temporal label '{primary}' is not yet supported: \
                         merging needs to fold per-version histories."
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
        // Stale columns were all dropped above, so the plain insert suffices.
        insert_label_columns(&mut row_with_refresh, target_var, &target_rec);

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
    set_items: &[crate::plan::SetItem],
    as_of: Option<&crate::plan::expr::Expr>,
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

        // The statement's NOW, in the epoch microseconds of every valid-time
        // value: the instant the clone's first version opens at and the
        // source's state is read at by default.
        let now_us = ctx.valid_now;
        // `AS OF <ts>` selects the source's state valid at that valid-time
        // instant; default is the state valid now.
        let as_of_us: Option<i64> = match as_of {
            Some(expr) => match eval_neutral(expr, row)? {
                Value::Int(us) => Some(us),
                Value::Timestamp(us) => Some(us),
                other => {
                    return Err(ExecutionError::Unsupported(format!(
                        "CLONE NODE AS OF requires an INT or TIMESTAMP (epoch \
                         microseconds), got {other:?}"
                    )));
                }
            },
            None => None,
        };
        // Non-temporal nodes read through the point key; temporal nodes have no
        // point key (per-version layout), so read the state valid at the
        // requested instant. A deleted or not-yet-valid source has none.
        let source_rec = match ctx.mvcc_get_node(ctx.shard_id, a_id)? {
            Some(r) if as_of_us.is_none() => r,
            _ => ctx
                .temporal_node_state(a_id, as_of_us.unwrap_or(now_us))?
                .positive()
                .map(|(_, record)| record)
                .ok_or_else(|| {
                    let suffix = as_of_us
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
        let mut properties: Vec<(String, crate::plan::expr::Expr)> = if with_properties {
            let mut props = Vec::new();
            for (field_id, val) in &source_rec.props {
                let Some(name) = ctx.interner.resolve(*field_id) else {
                    continue;
                };
                if matches!(name, "__ingestion_ts__" | "valid_from" | "valid_to") {
                    continue;
                }
                props.push((
                    name.to_string(),
                    crate::plan::expr::Expr::Literal(val.clone()),
                ));
            }
            if let Some(extra) = &source_rec.extra {
                for (k, v) in extra {
                    if matches!(k.as_str(), "__ingestion_ts__" | "valid_from" | "valid_to") {
                        continue;
                    }
                    props.push((k.clone(), crate::plan::expr::Expr::Literal(v.clone())));
                }
            }
            props
        } else {
            Vec::new()
        };

        // valid_from = NOW for the clone's first version. An explicit
        // `SET b.valid_from = ...` (applied after create) back-dates it.
        if is_temporal {
            properties.push((
                "valid_from".to_string(),
                crate::plan::expr::Expr::Literal(Value::Int(now_us)),
            ));
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
            execute_update(&created, set_items, &crate::plan::ViolationMode::Fail, ctx)?
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
        // Cloning a temporal node's CURRENT incident edges (never the version
        // history) onto the clone requires the cloned edge to be injected into
        // the per-version edgeprop store AND any vector/secondary index on that
        // edge type — the same bitemporal index-maintenance mechanism as a
        // version-transition (close-old / open-new active version). That
        // mechanism is pending, so for now reject WITH EDGES when the source
        // has incident edges of a temporal type rather than writing an
        // adjacency with no per-version edgeprop (and a stale index).
        // A type identified by its own discriminator holds no edge a
        // statement wrote; one there would be refused, not copied as single.
        let shape = edge_type_shape(et, ctx)?;
        if shape != Ok(false) {
            let has_incident =
                ctx.adj_get_fwd(et, a_id)?.is_some() || ctx.adj_get_rev(et, a_id)?.is_some();
            if has_incident {
                return Err(match shape {
                    Err(column) => independently_identified(et, &column),
                    Ok(_) => ExecutionError::Unsupported(format!(
                        "CLONE NODE WITH EDGES does not yet clone temporal edge type '{et}' \
                         (pending bitemporal index-maintenance); the node and its non-temporal \
                         edges clone normally."
                    )),
                });
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
    /// Whether this edge type is temporal — selects per-version vs single-body
    /// edgeprop transfer when re-pointing (the adjacency move is identical).
    temporal: bool,
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
    direction: crate::plan::RedirectDirection,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use crate::plan::RedirectDirection;

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

        // The redirect moves every edge of `a` it finds; an edge attached to
        // `a` after this read is one it never moved, and the two write
        // different keys. Without a type filter the promise covers types not
        // known yet as well, so the claim covers the whole node.
        if edge_types.is_some() {
            for et in &types {
                if do_out {
                    ctx.claim_incident_set_complete(et, a_id, true);
                }
                if do_in {
                    ctx.claim_incident_set_complete(et, a_id, false);
                }
            }
        } else {
            ctx.claim_all_incident_sets_complete(a_id);
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
                    temporal: lookup_edge_type_temporal(et, ctx)?,
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
                    ctx.mvcc_move_edge_props_versioned(
                        et,
                        a_id,
                        a_id,
                        b_id,
                        new_tgt,
                        snap.temporal,
                    )?;
                } else {
                    ctx.adj_merge_remove_fwd(et, a_id, x.as_raw());
                    ctx.adj_merge_remove_rev(et, x, a_id.as_raw());
                    ctx.adj_merge_add_fwd(et, b_id, x.as_raw());
                    ctx.adj_merge_add_rev(et, x, b_id.as_raw());
                    ctx.mvcc_move_edge_props_versioned(et, a_id, x, b_id, x, snap.temporal)?;
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
                    ctx.mvcc_move_edge_props_versioned(et, a_id, a_id, a_id, b_id, snap.temporal)?;
                } else {
                    ctx.adj_merge_remove_fwd(et, x, a_id.as_raw());
                    ctx.adj_merge_remove_rev(et, a_id, x.as_raw());
                    ctx.adj_merge_add_fwd(et, x, b_id.as_raw());
                    ctx.adj_merge_add_rev(et, b_id, x.as_raw());
                    ctx.mvcc_move_edge_props_versioned(et, x, a_id, x, b_id, snap.temporal)?;
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
    conflict: &crate::plan::MergeNodesConflictStrategy,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use crate::plan::MergeNodesConflictStrategy as S;
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
    item: &crate::plan::SetItem,
    row: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use crate::plan::SetItem;
    match item {
        SetItem::Property { property, expr, .. } => {
            let val = eval_neutral(expr, row)?.map_to_document();
            let field_id = ctx.field_id(property)?;
            target.set(field_id, val);
        }
        SetItem::PropertyPath { path, expr, .. } => {
            // Nested path SET: walk into `extra` (top-level path segment as
            // string key), creating a document along the way. For first
            // implementation, only support single-segment paths via the
            // declared props path. Multi-segment paths fall through to extra
            // with a dotted string key — matches existing executor behavior.
            let val = eval_neutral(expr, row)?.map_to_document();
            if path.len() == 1 {
                let field_id = ctx.field_id(&path[0])?;
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

/// Move the B-tree entries of the surviving target node from its values
/// before the merge (`old`) to those after it (`new`), so unique constraints
/// stay in sync.
fn notify_indexes_for_target_change(
    target_id: NodeId,
    label: &str,
    old: &HashMap<u32, Value>,
    new: &HashMap<u32, Value>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    // B-tree entries move from the old values to the new ones in one pass;
    // a unique value another node holds refuses the merge. Vector and text
    // indexes follow the committed record on their own.
    ctx.index_fields_changed(target_id, label, old, new)
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
    duplicate: &crate::plan::MergeNodesDuplicateStrategy,
    transfer_edge_properties: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use crate::plan::MergeNodesDuplicateStrategy as D;

    let edge_types = ctx.list_edge_types()?;
    for edge_type in &edge_types {
        // A type identified by its own discriminator holds no edge a
        // statement wrote, so it is refused only when the node has one.
        let shape = edge_type_shape(edge_type, ctx)?;
        for direction in [AdjDirection::Out, AdjDirection::In] {
            let Some(plist) = (match direction {
                AdjDirection::Out => ctx.adj_get_fwd(edge_type, source_id),
                AdjDirection::In => ctx.adj_get_rev(edge_type, source_id),
            })?
            else {
                continue;
            };
            let temporal = shape
                .as_ref()
                .map(|t| *t)
                .map_err(|column| independently_identified(edge_type, column))?;

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
    // The node's B-tree entries go before primary storage drops the node, or
    // unique constraints would still reject re-creation with the same value.
    // Vector and text indexes follow the committed deletion on their own.
    // Snapshot pre-mutation node state once for trigger firing — re-used after
    // the node is deleted to populate `$before` for the BEFORE COMMIT DELETE
    // trigger. Mirrors the `execute_delete` / `cascade_delete_source_node`
    // wiring so that MERGE NODES' source-cleanup fires the same triggers as
    // DETACH DELETE on the same node.
    let node_pre_snapshot: Option<(Vec<String>, std::collections::BTreeMap<String, Value>)> = ctx
        .mvcc_get_node(ctx.shard_id, node_id)?
        .map(|rec| snapshot_node_record(&rec, ctx));
    if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
        ctx.index_node_deleted(node_id, &record)?;
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
            // The detach is built on this posting list being the whole set:
            // every peer it names is unhooked and the node's own list is
            // purged. An edge attached while this ran would survive both
            // steps and point at a node that is gone.
            ctx.claim_incident_set_complete(
                edge_type,
                node_id,
                matches!(direction, AdjDirection::Out),
            );
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
    // Drop the primary node record, and its table key if it is a table row.
    if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, node_id)? {
        ctx.release_table_key(&record)?;
    }
    ctx.mvcc_delete_node(ctx.shard_id, node_id)?;
    ctx.write_stats.nodes_deleted += 1;
    // Statistics counters: decrement by the pre-delete labels (no row
    // removed when the node was already absent).
    if let Some((labels, _)) = &node_pre_snapshot {
        use coordinode_modality::{LocalStatsStore, StatsStore as _};
        LocalStatsStore.node_deleted(&mut ctx.txn, labels.iter().map(String::as_str));
    }

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
// DETACH DOCUMENT
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
    edge_direction: crate::plan::EdgeFromSource,
    transfer: Option<&crate::plan::TransferEdgesSpec>,
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

    // DETACH DOCUMENT on a temporal source: the source-mutation side
    // ("remove property from source") routes through the close+open
    // dance — read source's current per-version record, build a new
    // version with the property removed via DocDelta (applied in-memory
    // by `apply_doc_deltas_to_record`), close current at valid_to = NOW.
    //
    // TRANSFER EDGES on a temporal source is rejected: the new version
    // of the source has a different temporal identity than the closed
    // version, and nothing yet decides which version owns the
    // transferred edges. Without that decision we'd silently re-point
    // edges to an ambiguous (node_id, valid_from) pair.

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
                 (var `{source_variable}`) is not yet supported: edges \
                 would need an owner per (node_id, valid_from) version. \
                 Use plain DETACH DOCUMENT \
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
        let literal_props: Vec<(String, crate::plan::expr::Expr)> = props
            .iter()
            .map(|(k, v)| (k.clone(), crate::plan::expr::Expr::Literal(v.clone())))
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
            crate::plan::EdgeFromSource::Incoming => (target_id, source_id),
            crate::plan::EdgeFromSource::Outgoing => (source_id, target_id),
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
            // Opens at the statement's NOW, as the temporal SET does.
            let new_valid_from = if ctx.valid_now > current_valid_from {
                ctx.valid_now
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
                match field_id_opt {
                    Some(fid) => DocDelta::DeletePath {
                        target: PathTarget::PropField(fid),
                        path: property_path[1..].to_vec(),
                    },
                    None => DocDelta::DeletePath {
                        target: PathTarget::Extra,
                        path: property_path.to_vec(),
                    },
                }
            };
            coordinode_storage::engine::merge::apply_doc_deltas_to_record(
                &mut new_record,
                &[delta],
            );

            // Refresh bitemporal axis fields on the new record (the closing
            // one gets its valid_to as it is written).
            let [vf_fid, vt_fid, its_fid] = temporal_field_ids(ctx)?;
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
            ctx.close_temporal_version(source_id, current_vf, &mut closing_record, new_valid_from)?;
            ctx.open_temporal_version(source_id, new_valid_from, &new_record)?;
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
    // A name can be interned for another label and still be stored in this
    // node's overflow map, so both places are looked at.
    let by_id = interner
        .lookup(first)
        .and_then(|fid| record.props.get(&fid).map(|v| (Some(fid), v.clone())));
    let (field_id, root): (Option<u32>, Value) =
        match by_id.or_else(|| record.get_extra(first).map(|v| (None, v.clone()))) {
            Some(found) => found,
            None => {
                return Err(ExecutionError::Unsupported(format!(
                    "DETACH DOCUMENT: property `{first}` not found on node"
                )));
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
/// `Value::Document`: promotion is shallow, only the top level becomes
/// properties.
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
        // Nested path: an interned prop is addressed below its id; the
        // overflow map is one document, addressed by the full path.
        match field_id_opt {
            Some(fid) => DocDelta::DeletePath {
                target: PathTarget::PropField(fid),
                path: path[1..].to_vec(),
            },
            None => DocDelta::DeletePath {
                target: PathTarget::Extra,
                path: path.to_vec(),
            },
        }
    };

    let operand = delta
        .encode()
        .map_err(|e| ExecutionError::Serialization(format!("DocDelta encode: {e}")))?;
    ctx.mvcc_merge_node_delta(ctx.shard_id, node_id, operand)?;
    ctx.write_stats.properties_removed += 1;
    Ok(())
}

/// Attempt to extract the edge-type list from a `TRANSFER EDGES WHERE` predicate.
///
/// Supported shapes, where `r` is [`crate::plan::TRANSFER_EDGE_VARIABLE`]:
///   - `type(r) IN ['T1', 'T2', ...]`
///   - `type(r) = 'T1'`
///
/// Anything more complex returns an error; we prefer explicit scope to a
/// misinterpreted transfer.
fn extract_transfer_edge_types(
    predicate: &crate::plan::expr::Expr,
) -> Result<Vec<String>, ExecutionError> {
    use crate::plan::expr::{BinOp, Expr as NExpr};

    fn is_type_of_r(expr: &NExpr) -> bool {
        matches!(
            expr,
            NExpr::Call { name, args, .. }
                if name.eq_ignore_ascii_case("type")
                    && args.len() == 1
                    && matches!(
                        &args[0],
                        NExpr::Variable(v) if v == crate::plan::TRANSFER_EDGE_VARIABLE
                    )
        )
    }

    fn lit_string(e: &NExpr) -> Option<String> {
        if let NExpr::Literal(Value::String(s)) = e {
            Some(s.clone())
        } else {
            None
        }
    }

    match predicate {
        NExpr::In { item, list } if is_type_of_r(item) => {
            let NExpr::List(items) = list.as_ref() else {
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
        NExpr::Binary {
            left,
            op: BinOp::Eq,
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
// ATTACH DOCUMENT
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
///     matching edges via posting-list merges (the DETACH DOCUMENT helper).
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
    edge_direction: crate::plan::EdgeFromSource,
    target_property_path: &[String],
    transfer: Option<&crate::plan::TransferEdgesSpec>,
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

    // ATTACH DOCUMENT supports a temporal *target* — the target's
    // matched per-version record is read via its `<target>.valid_from`
    // binding and the property added via the close+open dance
    // (DocDelta::SetPath applied in-memory to the new-version clone).
    // Temporal *source* is rejected: ATTACH cascade-deletes the source,
    // which on a temporal label needs the positive-bitemporal-fact
    // tombstone composition + cross-version edge cleanup.
    for row in input_rows {
        if let Some(Value::String(primary)) = row.get(&format!("{source_variable}.__label__")) {
            if let Ok(Some(s)) = ctx.load_current_label_schema(primary) {
                if s.temporal {
                    return Err(ExecutionError::Unsupported(format!(
                        "ATTACH DOCUMENT on a temporal *source* (var \
                         `{source_variable}`, label '{primary}') is not yet \
                         supported: ATTACH cascade-deletes the source, which on \
                         a temporal label requires the positive-bitemporal-fact \
                         tombstone composition + cross-version edge cleanup. \
                         ATTACH onto a temporal *target* is supported."
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
            // Opens at the statement's NOW, as the temporal SET does.
            let new_valid_from = if ctx.valid_now > current_valid_from {
                ctx.valid_now
            } else {
                current_valid_from + 1
            };
            let mut closing_record = target_record.clone();
            let mut new_record = target_record.clone();

            // Build SetPath delta — same shape `emit_attach_set_path`
            // produces, but applied in-memory.
            use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};
            let first = &target_property_path[0];
            let field_id = ctx.field_id(first)?;
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

            let [vf_fid, vt_fid, its_fid] = temporal_field_ids(ctx)?;
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
            ctx.close_temporal_version(target_id, current_vf, &mut closing_record, new_valid_from)?;
            ctx.open_temporal_version(target_id, new_valid_from, &new_record)?;
            ctx.write_stats.properties_set += 1;
        } else {
            emit_attach_set_path(target_id, target_property_path, doc, ctx)?;
        }

        // --- 4. Delete the connecting edge (both directions) ---
        let (edge_src, edge_tgt) = match edge_direction {
            crate::plan::EdgeFromSource::Outgoing => (source_id, target_id),
            crate::plan::EdgeFromSource::Incoming => (target_id, source_id),
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

/// Whether property `name` of a node under `schema` is stored by name in the
/// overflow map: the undeclared properties of a VALIDATED label are, every
/// other property is stored under its interned id.
fn stored_by_name(schema: Option<&LabelSchema>, name: &str) -> bool {
    schema
        .is_some_and(|s| matches!(s.mode, SchemaMode::Validated) && s.get_property(name).is_none())
}

/// What refuses a SET of `property` on a node of `label` under `schema`: an
/// undeclared property of a STRICT label, a computed property, or a `value`
/// of another type than declared. `value` is `None` for a write below the
/// property (a path or a document function), which only the declaration of
/// the property itself answers for. `None` when the write is admitted.
fn set_property_violation(
    schema: Option<&LabelSchema>,
    label: &str,
    property: &str,
    value: Option<&Value>,
) -> Option<ExecutionError> {
    let schema = schema?;
    let def = match (&schema.mode, schema.get_property(property)) {
        (SchemaMode::Flexible, _) | (SchemaMode::Validated, None) => return None,
        (SchemaMode::Strict, None) => {
            return Some(ExecutionError::SchemaViolation(format!(
                "unknown property '{property}' for strict label '{label}'"
            )));
        }
        (_, Some(def)) => def,
    };
    if def.is_computed() {
        return Some(ExecutionError::SchemaViolation(format!(
            "cannot SET computed property '{property}'"
        )));
    }
    value
        .and_then(|v| validate_one(property, v, def).err())
        .map(|e| ExecutionError::SchemaViolation(e.to_string()))
}

/// Refuse a user write of a field the engine keeps in every version of a
/// temporal node.
fn refuse_engine_temporal_field(property: &str) -> Result<(), ExecutionError> {
    if coordinode_core::schema::definition::TEMPORAL_ENGINE_FIELDS.contains(&property) {
        return Err(ExecutionError::Unsupported(format!(
            "SET on '{property}' is reserved: this field is engine-managed on temporal \
             labels and cannot be assigned by SET"
        )));
    }
    Ok(())
}

/// Set property `name` of `record` where its schema keeps it, dropping a copy
/// left in the other place: a record holds one value per name, so no read
/// can see an older one.
fn store_node_property(
    record: &mut NodeRecord,
    name: &str,
    value: Value,
    by_name: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    if by_name {
        if let Some(field_id) = ctx.interner.lookup(name) {
            record.props.remove(&field_id);
        }
        record.set_extra(name, value);
    } else {
        let field_id = ctx.field_id(name)?;
        record.remove_extra(name);
        record.set(field_id, value);
    }
    Ok(())
}

/// Write `changes` (name, value, stored by name) to the properties of
/// `node_id`, whose record as the statement leaves it is `record`, stored in
/// `stored_bytes`: as property deltas when that leaves the record unwritten
/// at a cost reads can bear, else as the whole record.
fn write_property_changes(
    ctx: &mut ExecutionContext<'_>,
    node_id: NodeId,
    record: &NodeRecord,
    stored_bytes: usize,
    changes: &[(&str, Value, bool)],
) -> Result<(), ExecutionError> {
    let shard_id = ctx.shard_id;
    // Encoded first: their size decides whether they are written at all.
    let mut operands = Vec::with_capacity(changes.len());
    for (name, value, by_name) in changes {
        for delta in property_set_deltas(record, name, value.clone(), *by_name, ctx)? {
            operands.push(
                delta
                    .encode()
                    .map_err(|e| ExecutionError::Serialization(format!("property delta: {e}")))?,
            );
        }
    }
    let delta_bytes = operands.iter().map(Vec::len).sum();
    if ctx.writes_property_delta(shard_id, node_id, stored_bytes, delta_bytes) {
        for operand in operands {
            ctx.mvcc_merge_node_delta(shard_id, node_id, operand)?;
        }
        ctx.note_property_write(shard_id, node_id, true);
        return Ok(());
    }
    // The tracked read folds the statement's pending deltas of the node into
    // its buffered record, so the whole write below is the only change left
    // for it and no delta is applied twice.
    let Some(mut whole) = ctx.mvcc_get_node(shard_id, node_id)? else {
        return Ok(());
    };
    for (name, value, by_name) in changes {
        store_node_property(&mut whole, name, value.clone(), *by_name, ctx)?;
    }
    ctx.mvcc_put_node(shard_id, node_id, &whole)?;
    ctx.note_property_write(shard_id, node_id, false);
    Ok(())
}

/// The deltas that do what [`store_node_property`] does to `record`: set
/// property `name` where its schema keeps it, and remove a copy `record`
/// holds in the other place, so no read can see an older value.
fn property_set_deltas(
    record: &NodeRecord,
    name: &str,
    value: Value,
    by_name: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<coordinode_core::graph::doc_delta::DocDelta>, ExecutionError> {
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};
    let mut deltas = Vec::with_capacity(2);
    if by_name {
        if let Some(field_id) = ctx.interner.lookup(name) {
            if record.props.contains_key(&field_id) {
                deltas.push(DocDelta::RemoveProperty {
                    target: PathTarget::PropField(field_id),
                    key: None,
                });
            }
        }
        deltas.push(DocDelta::SetProperty {
            target: PathTarget::Extra,
            key: Some(name.to_string()),
            value,
        });
    } else {
        let field_id = ctx.field_id(name)?;
        if record.get_extra(name).is_some() {
            deltas.push(DocDelta::RemoveProperty {
                target: PathTarget::Extra,
                key: Some(name.to_string()),
            });
        }
        deltas.push(DocDelta::SetProperty {
            target: PathTarget::PropField(field_id),
            key: None,
            value,
        });
    }
    Ok(deltas)
}

/// Register in one batch the names of `map` stored under an id, so storing
/// them one by one registers nothing more.
fn register_stored_ids(
    schema: Option<&LabelSchema>,
    map: &std::collections::BTreeMap<String, Value>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    let names: Vec<&str> = map
        .keys()
        .map(String::as_str)
        .filter(|name| !stored_by_name(schema, name))
        .collect();
    ctx.field_ids(&names)?;
    Ok(())
}

/// The names of every property `record` holds, under an id or by name.
fn property_names(record: &NodeRecord, interner: &FieldInterner) -> Vec<String> {
    record
        .props
        .keys()
        .filter_map(|field_id| interner.resolve(*field_id).map(str::to_string))
        .chain(record.extra.iter().flat_map(|extra| extra.keys().cloned()))
        .collect()
}

/// Remove property `name` from `record` wherever it is stored, returning the
/// value it had.
fn remove_node_property(
    record: &mut NodeRecord,
    name: &str,
    interner: &FieldInterner,
) -> Option<Value> {
    let by_id = interner
        .lookup(name)
        .and_then(|field_id| record.props.remove(&field_id));
    record.remove_extra(name).or(by_id)
}

/// The merge target and in-target path of property path `path` of a node
/// under `schema`: an overflow property is addressed by its full path inside
/// the overflow map, any other by its id and the path below it.
fn document_target(
    schema: Option<&LabelSchema>,
    path: &[String],
    ctx: &mut ExecutionContext<'_>,
) -> Result<(coordinode_core::graph::doc_delta::PathTarget, Vec<String>), ExecutionError> {
    use coordinode_core::graph::doc_delta::PathTarget;
    if stored_by_name(schema, &path[0]) {
        Ok((PathTarget::Extra, path.to_vec()))
    } else {
        Ok((
            PathTarget::PropField(ctx.field_id(&path[0])?),
            path[1..].to_vec(),
        ))
    }
}

/// The merge target and in-target path for deleting property path `path` of
/// a node under `schema`, or `None` when the root has no id and so is stored
/// on no record. Registers nothing.
fn removal_target(
    schema: Option<&LabelSchema>,
    path: &[String],
    interner: &FieldInterner,
) -> Option<(coordinode_core::graph::doc_delta::PathTarget, Vec<String>)> {
    use coordinode_core::graph::doc_delta::PathTarget;
    if stored_by_name(schema, &path[0]) {
        return Some((PathTarget::Extra, path.to_vec()));
    }
    interner
        .lookup(&path[0])
        .map(|field_id| (PathTarget::PropField(field_id), path[1..].to_vec()))
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
    let field_id = ctx.field_id(first)?;
    let sub: Vec<String> = path[1..].to_vec();

    let delta = DocDelta::SetPath {
        target: PathTarget::PropField(field_id),
        path: sub,
        value: doc,
    };
    let operand = delta
        .encode()
        .map_err(|e| ExecutionError::Serialization(format!("DocDelta encode: {e}")))?;
    ctx.mvcc_merge_node_delta(ctx.shard_id, target_id, operand)?;
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
/// rare (a mistaken insert, a privileged erase) but must be supported.
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

    // Delete the node record itself (B-tree / vector indexes left to the
    // standard delete path; ATTACH DOCUMENT's source is typically a
    // short-lived node so we keep this tight — cleanup mirrors DETACH DELETE
    // behaviour in `execute_delete`).
    // Snapshot the pre-mutation node record once: re-used for both the
    // BEFORE COMMIT DELETE trigger firing and (where present) index cleanup.
    let pre_snapshot: Option<(Vec<String>, std::collections::BTreeMap<String, Value>)> = ctx
        .mvcc_get_node(ctx.shard_id, source_id)?
        .map(|rec| snapshot_node_record(&rec, ctx));
    if ctx.btree_index_registry.is_some() {
        if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, source_id)? {
            ctx.index_node_deleted(source_id, &record)?;
        }
    }
    if let Some(record) = ctx.mvcc_get_node(ctx.shard_id, source_id)? {
        ctx.release_table_key(&record)?;
    }
    ctx.mvcc_delete_node(ctx.shard_id, source_id)?;
    ctx.write_stats.nodes_deleted += 1;
    // Statistics counters: decrement by the pre-delete labels (no row
    // removed when the node was already absent).
    if let Some((labels, _)) = &pre_snapshot {
        use coordinode_modality::{LocalStatsStore, StatsStore as _};
        LocalStatsStore.node_deleted(&mut ctx.txn, labels.iter().map(String::as_str));
    }

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

/// Build the Schema-partition key for a given edge type name (test fixtures
/// that plant/inspect raw edge-type markers). Production code goes through
/// [`coordinode_modality::SchemaStore`].
///
/// Format: `schema:edge_type:<name>:1`
#[cfg(test)]
fn edge_type_schema_key(edge_type: &str) -> Vec<u8> {
    coordinode_core::schema::definition::encode_edge_type_schema_key(edge_type, 1)
}

/// Whether instances of an edge type are identified by their `valid_from`
/// (the start-identified temporal shorthand) rather than by the endpoints
/// alone, read from the type's resolved discriminator.
///
/// An edge type without a definition, or with only the existence marker an
/// edge create leaves, is single-edge. A type identified by any other
/// property is refused: reading or writing it as single-edge or by
/// `valid_from` would merge or split its instances.
fn lookup_edge_type_temporal(
    edge_type: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<bool, ExecutionError> {
    edge_type_shape(edge_type, ctx)?.map_err(|column| independently_identified(edge_type, &column))
}

/// The refusal of a statement that reaches edges of a type identified by
/// its own discriminator.
fn independently_identified(edge_type: &str, column: &str) -> ExecutionError {
    ExecutionError::Unsupported(format!(
        "edge type '{edge_type}' identifies its instances by '{column}'; queries over it are \
         not supported yet"
    ))
}

/// Whether instances of an edge type are start-identified (`Ok(true)`) or
/// single (`Ok(false)`), or the property they are identified by otherwise
/// (`Err`). No statement writes an edge of the last kind, so a statement
/// ranging over every type, rather than naming this one, passes it over
/// without missing anything.
fn edge_type_shape(
    edge_type: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Result<bool, String>, ExecutionError> {
    let Some(schema) = ctx.load_current_edge_type_schema(edge_type)? else {
        return Ok(Ok(false));
    };
    Ok(match schema.discriminator() {
        None => Ok(false),
        Some(_) if schema.is_start_identified() => Ok(true),
        Some(d) => Err(d.column.clone()),
    })
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
    let field_id = ctx.field_id(property)?;
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

/// Whether a map assignment keeps the properties it does not name.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeMapAssign {
    /// `SET r += {map}`: named keys are written, the rest survive.
    Merge,
    /// `SET r = {map}`: the map becomes the whole property set.
    Replace,
}

/// Apply `SET r += {map}` / `SET r = {map}` to a relationship.
///
/// Reads the edgeprop map once, applies every key, writes once, rather than
/// paying a read-modify-write per key through [`update_edge_property`]. The
/// guards are the same ones that path enforces: `valid_from` is part of the
/// storage key on a temporal edge, and the metadata names are engine-owned.
///
/// A non-map right-hand side is a no-op: the parser only produces these items
/// for map expressions, and a parameter that resolves to something else is
/// the caller's mistake to see in the row, not a reason to corrupt the edge.
fn update_edge_properties_from_map(
    edge_variable: &str,
    edge_type: &str,
    map_val: &Value,
    mode: EdgeMapAssign,
    row: &mut Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    let Value::Map(map) = map_val else {
        return Ok(());
    };

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
    for key in map.keys() {
        if is_temporal && key == "valid_from" {
            return Err(ExecutionError::Unsupported(format!(
                "SET {edge_variable} with a map containing 'valid_from' is not \
                 allowed on temporal edges: valid_from is part of the storage \
                 key. DELETE the version and CREATE a new one with the updated \
                 timestamp."
            )));
        }
        if matches!(key.as_str(), "__src__" | "__tgt__" | "__type__") {
            return Err(ExecutionError::Unsupported(format!(
                "SET {edge_variable} with a map containing '{key}' is reserved: \
                 '{key}' is engine-internal row metadata and cannot be assigned"
            )));
        }
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

    let previous = ctx
        .mvcc_get_edge_props_either(edge_type, src, tgt, valid_from_for_key)?
        .unwrap_or_default();

    let mut prop_map = match mode {
        EdgeMapAssign::Merge => previous.clone(),
        EdgeMapAssign::Replace => Vec::with_capacity(map.len()),
    };
    let names: Vec<&str> = map.keys().map(String::as_str).collect();
    let field_ids = ctx.field_ids(&names)?;
    for (value, field_id) in map.values().zip(field_ids) {
        match prop_map.iter_mut().find(|entry| entry.0 == field_id) {
            Some(entry) => entry.1 = value.clone(),
            None => prop_map.push((field_id, value.clone())),
        }
        ctx.write_stats.properties_set += 1;
    }
    ctx.mvcc_put_edge_props_either(edge_type, src, tgt, valid_from_for_key, &prop_map)?;

    // Keep the row in step so a RETURN in the same statement reads what was
    // written. A replace drops the keys the map omits, so they have to read
    // back as NULL rather than as their pre-write value.
    if mode == EdgeMapAssign::Replace {
        for (field_id, _) in &previous {
            if let Some(name) = ctx.interner.resolve(*field_id) {
                row.insert(format!("{edge_variable}.{name}"), Value::Null);
            }
        }
    }
    for (key, value) in map {
        row.insert(format!("{edge_variable}.{key}"), value.clone());
    }

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
/// Resolve each declaration (including absence) once in the statement's
/// transactional view, then evaluate at its already bound instant. No fresh
/// engine snapshot is acquired while materializing an individual row.
fn inject_computed_properties(
    row: &mut Row,
    variable: &str,
    label: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    if !ctx.label_schema_cache.contains_key(label) {
        let schema = ctx.load_current_label_schema(label)?;
        ctx.label_schema_cache.insert(label.to_string(), schema);
    }
    if let Some(Some(schema)) = ctx.label_schema_cache.get(label) {
        inject_computed_from_schema(row, variable, schema, ctx.valid_now);
    }
    Ok(())
}

/// Pure enrichment shared by sequential and parallel readers. The caller
/// supplies the same view-bound interpretation and evaluation instant.
fn inject_computed_from_schema(row: &mut Row, variable: &str, schema: &LabelSchema, now_us: i64) {
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

/// Neutral-IR twin of [`expr_display_name`]: derives the default projection
/// column name for a neutral expression carried by `ProjectItem`.
fn expr_display_name_neutral(expr: &crate::plan::expr::Expr) -> String {
    use crate::plan::expr::Expr as PExpr;
    match expr {
        PExpr::Variable(name) => name.clone(),
        PExpr::Property { base, key } => {
            let parent = expr_display_name_neutral(base);
            format!("{parent}.{key}")
        }
        PExpr::Call { name, .. } => name.clone(),
        PExpr::Star => "*".to_string(),
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
        .load_label_schema_for_update(label)?
        .unwrap_or_else(|| LabelSchema::new_node_id(label));

    schema.set_mode(mode);
    schema.schema_revision = next_revision(&schema)?;
    if let Some((name, conflict)) = schema.constraints().iter().find_map(|c| {
        schema
            .constraint_conflict(c)
            .map(|why| (c.name.clone(), why))
    }) {
        return Err(ExecutionError::CatalogRefused(format!(
            "label '{label}' cannot be set to {mode}: constraint '{name}' could not hold: \
             {conflict}"
        )));
    }

    // The new mode governs the nodes already stored. The commit decides this
    // authoritatively with no write under the old mode beside it; checking
    // here as well names the node that refuses it.
    let staged = std::collections::HashMap::new();
    if let Some(violation) =
        coordinode_storage::engine::claims::evaluate::first_label_schema_violation(
            ctx.engine, &schema, &staged,
        )?
    {
        return Err(ExecutionError::SchemaViolation(format!(
            "label '{label}' cannot be set to {mode}: node {} breaks it ({})",
            violation.node.as_raw(),
            violation.reason
        )));
    }

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

/// Execute `CREATE NODE TYPE <name> [TEMPORAL] [WITH (...)]`. Mirror of
/// `execute_create_edge_type` for node labels.
///
/// Persists a new `LabelSchema` with the bitemporal flag set as declared and
/// the user-supplied property declarations. Rejects:
///   - Duplicate label (label already has a current-revision pointer)
///   - Reserved engine-internal property names (`__ingestion_ts__`,
///     `__deleted__`, `__src__`, `__tgt__`, `__type__`): these are populated/owned by the
///     engine; user declarations would shadow them. `valid_from` and
///     `valid_to` are user-supplied and may be declared.
///   - Unsupported property type spellings
///
/// The TEMPORAL flag is immutable from this point forward: changing it
/// requires creating a new label type and copying data.
fn execute_create_node_type(
    name: &str,
    temporal: bool,
    properties: &[crate::plan::PropertyDecl],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_core::schema::definition::{
        LabelSchema, PlacementPolicy, PropertyDef, PropertyType,
    };

    // Reject if the label already has a schema; a concurrent definition of
    // the same label refuses this commit.
    if ctx.load_label_schema_for_update(name)?.is_some() {
        return Err(ExecutionError::CatalogObjectExists {
            object: CatalogObject::Label,
            name: name.to_string(),
        });
    }

    // Reject reserved engine-internal property names. `__ingestion_ts__`
    // (HLC commit-ts on temporal labels) is fully engine-owned — declaring
    // it would shadow the canonical engine value. `__src__` / `__tgt__` /
    // `__type__` are edge-row metadata and have no meaning on a node label;
    // rejecting them keeps the reserved-name surface symmetric and prevents
    // accidental confusion. `valid_from` / `valid_to` are NOT in this list
    // by design: they are user-supplied bitemporal interval fields on
    // temporal labels — engine validates
    // their type and invariants at write time but does not own the values.
    for decl in properties {
        if coordinode_core::schema::definition::TEMPORAL_ENGINE_FIELDS.contains(&decl.name.as_str())
            || matches!(decl.name.as_str(), "__src__" | "__tgt__" | "__type__")
        {
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

/// Resolve a `CREATE TABLE` lexical column type (SQL or cypher spelling) to a
/// `PropertyType`. Returns `None` for an unknown type name.
fn resolve_table_column_type(
    type_name: &str,
) -> Option<coordinode_core::schema::definition::PropertyType> {
    coordinode_core::schema::definition::PropertyType::from_type_name(type_name)
}

/// CREATE TABLE: declare a relational TABLE label. Persists a
/// `LabelSchema` carrying the declared primary key and storage layout, and for
/// a columnar table opens the per-table columnar tree.
fn execute_create_table(
    name: &str,
    columns: &[crate::plan::TableColumn],
    primary_key: &[String],
    columnar: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_core::schema::definition::{
        LabelSchema, PlacementPolicy, PropertyDef, StorageLayout,
    };

    if ctx.load_label_schema_for_update(name)?.is_some() {
        return Err(ExecutionError::CatalogObjectExists {
            object: CatalogObject::Label,
            name: name.to_string(),
        });
    }
    // Uniqueness of a column is a constraint the table would have to own, not
    // a flag of the column, and a TABLE cannot own one yet: the declaration is
    // refused before anything of the table is written.
    if let Some(col) = columns.iter().find(|c| c.unique) {
        return Err(ExecutionError::Unsupported(format!(
            "UNIQUE on column '{}' of table '{name}' is not supported; the table was not created",
            col.name
        )));
    }

    let mut schema = LabelSchema::new(name, PlacementPolicy::NodeId);
    for col in columns {
        let Some(ptype) = resolve_table_column_type(&col.type_name) else {
            return Err(ExecutionError::Unsupported(format!(
                "unsupported column type '{}' for table '{name}'",
                col.type_name
            )));
        };
        let mut prop = PropertyDef::new(&col.name, ptype);
        // Primary-key columns are implicitly NOT NULL.
        if col.not_null || primary_key.iter().any(|pk| pk == &col.name) {
            prop = prop.not_null();
        }
        schema.add_property(prop);
    }
    // Every primary-key column must be a declared column.
    for pk in primary_key {
        if !columns.iter().any(|c| &c.name == pk) {
            return Err(ExecutionError::Unsupported(format!(
                "PRIMARY KEY column '{pk}' is not declared in table '{name}'"
            )));
        }
    }
    // Without declared key columns the table is keyed by row id.
    schema.make_table(primary_key.to_vec());
    if columnar {
        schema.set_storage_layout(StorageLayout::Columnar);
    }

    ctx.save_current_label_schema(&schema)?;

    // A columnar table is backed by its own columnar-mode tree; open it now so
    // it exists before any write. Row tables stay on the node path.
    if columnar {
        ctx.engine.create_columnar_table(name)?;
    }

    let mut row = Row::new();
    row.insert("name".to_string(), Value::String(name.to_string()));
    row.insert(
        "storage".to_string(),
        Value::String(if columnar { "COLUMNAR" } else { "ROW" }.to_string()),
    );
    row.insert("columns".to_string(), Value::Int(columns.len() as i64));
    Ok(vec![row])
}

/// DROP TABLE: drop a relational TABLE label. Tombstones the schema
/// pointer and, for a columnar table, drops its per-table columnar tree.
fn execute_drop_table(
    name: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let Some(schema) = ctx.load_label_schema_for_update(name)? else {
        return Err(ExecutionError::CatalogObjectMissing {
            object: CatalogObject::Label,
            name: name.to_string(),
        });
    };
    // A constraint depends on the table: dropping the table would leave its
    // name taken and its index maintained with nothing to hold.
    if let Some(constraint) = schema.constraints().first() {
        return Err(ExecutionError::CatalogRefused(format!(
            "table '{name}' has constraint '{}'; drop its constraints first",
            constraint.name
        )));
    }
    if !schema.is_table() {
        return Err(ExecutionError::Unsupported(format!(
            "'{name}' is not a table; DROP TABLE applies only to relational tables"
        )));
    }
    let columnar = schema.is_columnar();

    // A row table's rows are nodes on the node path: they go with the table,
    // their edges with them, or a table created again under the name would
    // find them.
    if !columnar {
        let rows = execute_node_scan("__row", &[name.to_string()], &[], ctx)?;
        for row in &rows {
            if let Some(Value::Int(id)) = row.get("__row") {
                detach_delete_node(NodeId::from_raw(*id as u64), ctx)?;
            }
        }
    }
    ctx.release_all_table_keys(name)?;
    ctx.drop_current_label_schema(name)?;
    if columnar {
        ctx.engine.drop_columnar_table(name)?;
    }

    let mut row = Row::new();
    row.insert("name".to_string(), Value::String(name.to_string()));
    row.insert("status".to_string(), Value::String("dropped".to_string()));
    Ok(vec![row])
}

/// Execute CREATE EDGE TYPE: register an edge-type schema in the Schema partition.
///
/// Persists an `EdgeTypeSchema` keyed by `schema:edge_type:<name>` via the MVCC
/// write buffer. Subsequent edge writes that name this type can be validated
/// against the declared properties; if `temporal == true`, the write path
/// requires `valid_from` and stores per-version edgeprop entries.
///
/// Returns one row: `{ name, temporal, discriminator, version, properties }`,
/// `discriminator` being the property instances are identified by (null for
/// a single-edge type).
fn execute_create_edge_type(
    name: &str,
    temporal: bool,
    properties: &[crate::plan::PropertyDecl],
    discriminated_by: Option<&str>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Reject if the edge type was registered before (either via explicit DDL,
    // which writes a pointer, or implicitly by a prior edge create, which
    // writes a revisioned existence marker). Probe both:
    //  1. current_revision pointer → set by explicit CREATE EDGE TYPE
    //  2. legacy unparametrised existence marker at revision 1 → set by
    //     implicit edge registration in `create_edges_for_correlated_row`.
    if ctx.mvcc_edge_type_exists(name)? {
        return Err(ExecutionError::CatalogObjectExists {
            object: CatalogObject::EdgeType,
            name: name.to_string(),
        });
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
    schema
        .resolve_identity(discriminated_by)
        .map_err(ExecutionError::CatalogRefused)?;

    // Persist new version + pointer atomically.
    ctx.save_current_edge_type_schema(&schema)?;

    let mut row = Row::new();
    row.insert("name".to_string(), Value::String(name.to_string()));
    row.insert("temporal".to_string(), Value::Bool(temporal));
    row.insert(
        "discriminator".to_string(),
        schema
            .discriminator()
            .map_or(Value::Null, |d| Value::String(d.column.clone())),
    );
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
// Trigger DDL executors
// ======================================================================

fn current_hlc_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// The wall clock in microseconds since the epoch, the valid-time unit: what
/// a statement binds as its [`ExecutionContext::valid_now`].
pub fn wall_clock_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_micros()).unwrap_or(i64::MAX))
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
    c: &crate::plan::TriggerDef,
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

    let target_segment = c.target.index_key_segment();
    let events_schema = c.events;

    let schema = TriggerSchema {
        name: c.name.clone(),
        target: c.target.clone(),
        events: events_schema,
        timing: c.timing,
        body_source: c.body_source.clone(),
        cascade_limit: c.cascade_limit,
        cascade_fanout: c.cascade_fanout,
        on_error: c.on_error.clone(),
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

/// `SHOW SESSIONS`: one row per live client session on this node: its id, peer,
/// age, in-flight request count, and number of open transactions. Reads the
/// injected session registry; with none wired (embedded / tests) the result is
/// empty.
fn execute_show_sessions(ctx: &mut ExecutionContext<'_>) -> Result<Vec<Row>, ExecutionError> {
    let Some(ops) = ctx.operations else {
        return Ok(Vec::new());
    };
    let mut rows: Vec<Row> = Vec::new();
    for s in ops.sessions() {
        let mut row = Row::new();
        row.insert("session".into(), Value::String(s.session_id));
        row.insert("peer".into(), Value::String(s.peer));
        row.insert("age_ms".into(), Value::Int(s.age_ms as i64));
        row.insert("in_flight".into(), Value::Int(s.in_flight as i64));
        row.insert(
            "open_transactions".into(),
            Value::Int(s.transactions.len() as i64),
        );
        rows.push(row);
    }
    Ok(rows)
}

/// `SHOW TRANSACTIONS`: one row per open interactive transaction across all
/// sessions on this node: its handle, owning session, ordering mode, age, and
/// the milliseconds left before the idle reaper auto-aborts it. Reads the
/// injected session registry; with none wired (embedded / tests) the result is
/// empty.
fn execute_show_transactions(ctx: &mut ExecutionContext<'_>) -> Result<Vec<Row>, ExecutionError> {
    let Some(ops) = ctx.operations else {
        return Ok(Vec::new());
    };
    let mut rows: Vec<Row> = Vec::new();
    for s in ops.sessions() {
        for t in s.transactions {
            let mut row = Row::new();
            row.insert("txid".into(), Value::Int(t.txid as i64));
            row.insert("session".into(), Value::String(s.session_id.clone()));
            row.insert("peer".into(), Value::String(s.peer.clone()));
            row.insert("ordering".into(), Value::String(t.ordering.as_str().into()));
            row.insert("age_ms".into(), Value::Int(t.age_ms as i64));
            row.insert(
                "auto_abort_in_ms".into(),
                Value::Int(t.auto_abort_in_ms as i64),
            );
            rows.push(row);
        }
    }
    Ok(rows)
}

fn execute_alter_trigger(
    c: &crate::plan::AlterTriggerDef,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use crate::plan::AlterTriggerAction;
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
            schema.on_error = Some(pol.clone());
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
/// BEFORE COMMIT behave as PROPAGATE — synchronous retry inside the same
/// transaction would deadlock against write locks, and dead-lettering
/// inside an aborting transaction is paradoxical (the dead-letter write
/// would itself be rolled back).
///
/// AFTER COMMIT triggers in the matched list are enqueued — not run inline.
/// Each one writes a durable [`PendingTriggerEvent`](coordinode_core::schema::triggers::PendingTriggerEvent)
/// into the SAME transaction as the originating mutation (atomic enqueue), and
/// the out-of-band dispatcher (`Database::dispatch_after_commit_triggers`)
/// executes the body afterwards with the parameters captured here. The
/// enqueued `generation` bounds async cascade
/// depth.
pub(crate) fn fire_before_commit_triggers(
    matched: &[coordinode_core::schema::triggers::TriggerSchema],
    params: &std::collections::HashMap<String, Value>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use coordinode_core::schema::triggers::TriggerTimingSchema;
    for trigger in matched {
        match trigger.timing {
            TriggerTimingSchema::BeforeCommit => {
                ctx.cascade_enter(&trigger.name, trigger.cascade_limit, trigger.cascade_fanout)?;
                let result = execute_trigger_body_inline(trigger, params, ctx);
                ctx.cascade_exit();
                result?;
            }
            TriggerTimingSchema::AfterCommit => {
                enqueue_after_commit_trigger(trigger, params, ctx)?;
            }
        }
    }
    Ok(())
}

/// Persist a queued AFTER COMMIT event for `trigger` into the current
/// transaction. The parameters are snapshotted now (the dispatcher must not
/// reconstruct `$before`/`$after` after the fact); the body and `ON ERROR`
/// policy are re-read live from the definition at dispatch time.
fn enqueue_after_commit_trigger(
    trigger: &coordinode_core::schema::triggers::TriggerSchema,
    params: &std::collections::HashMap<String, Value>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<(), ExecutionError> {
    use coordinode_core::schema::triggers::PendingTriggerEvent;
    use coordinode_modality::{LocalTriggerStore, TriggerStore as _};

    // Monotonic enqueue stamp: each call to the oracle advances the HLC, so
    // distinct events (even within one statement) get distinct, ordered keys.
    // Without an oracle (legacy non-MVCC mode) fall back to the wall clock —
    // the worst case is a key collision that overwrites a sibling enqueue,
    // which the legacy mode never exercises (triggers require MVCC).
    let seq = match ctx.mvcc_oracle {
        Some(oracle) => oracle.next().as_raw(),
        None => current_hlc_us(),
    };
    let event = PendingTriggerEvent {
        trigger_name: trigger.name.clone(),
        params: params.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        attempt: 0,
        generation: ctx.after_commit_generation.saturating_add(1),
        first_seen_us: current_hlc_us(),
        next_attempt_us: 0,
    };
    let bytes = rmp_serde::to_vec(&event).map_err(|e| {
        ExecutionError::Serialization(format!(
            "after-commit event for `{}` encode: {e}",
            trigger.name
        ))
    })?;
    ctx.sync_txn_state();
    LocalTriggerStore.put_pending(&mut ctx.txn, &trigger.name, seq, &bytes)?;
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
    fields: &[crate::plan::TextIndexFieldSpec],
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
    // Publish the definition as its own catalog commit, which gives it its
    // identities, so every member sees the index before anything is built.
    let def = ctx.publish_index_in_catalog(crate::index::IndexDescriptor::text(
        name,
        label,
        properties.clone(),
        config,
    ))?;
    let generation = def.generation;

    // Register in text index registry (creates tantivy directory + empty index).
    registry
        .register(def)
        .map_err(|e| ExecutionError::Unsupported(format!("register text index: {e}")))?;

    // The backfill is this member's build of the index, run by the engine's
    // executor; the statement waits for it, and waiting cancels nothing.
    let builds = ctx.index_builds.ok_or_else(|| {
        ExecutionError::Unsupported("building an index requires the engine's index builds".into())
    })?;
    builds
        .run_local(
            generation,
            crate::index::lifecycle::text_build(label.to_string(), properties.clone()),
        )
        .map_err(ExecutionError::Unsupported)?;
    let count = match builds.wait(generation, None)? {
        Some(crate::index::IndexBuildOutcome::Published { indexed }) => indexed.unwrap_or(0),
        Some(crate::index::IndexBuildOutcome::Failed(e)) => {
            return Err(ExecutionError::Unsupported(e.to_string()));
        }
        other => {
            return Err(ExecutionError::Unsupported(format!(
                "the build of text index '{name}' ended without its index: {other:?}"
            )));
        }
    };

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
    let def = registry
        .definitions()
        .into_iter()
        .find(|d| d.name.as_deref() == Some(name));

    let Some(def) = def else {
        return Err(ExecutionError::Unsupported(format!(
            "text index '{name}' not found"
        )));
    };

    // Remove from registry.
    registry.unregister(&def.label, def.property());

    // Remove the definition transactionally through the index store.
    ctx.mvcc_delete_index_def(&def)?;

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
/// 1. Validates the context carries vector indexes.
/// 2. Builds a `VectorIndexConfig` from the OPTIONS.
/// 3. Persists the `IndexDefinition` to the `Schema` partition.
/// 4. Registers the empty HNSW graph in the registry, rebuilding.
/// 5. Starts the background build that fills it, owned by the registry.
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
    let Some(VectorIndexes {
        registry,
        engine: engine_arc,
        ..
    }) = ctx.vector_indexes
    else {
        return Err(ExecutionError::Unsupported(
            "CREATE VECTOR INDEX requires vector indexes in ExecutionContext".into(),
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
    let mut descriptor = crate::index::IndexDescriptor::hnsw(name, label, property, config);
    descriptor.online_during_build = online_during_build;

    // The names the index is keyed by are bound before its definition is
    // published, so every member that sees the definition can resolve them.
    let ids = ctx.field_ids(&[label, property])?;
    let (label_id, property_id) = (ids[0], ids[1]);

    // Publish the definition as its own catalog commit, before the build
    // starts: the catalog gives it its identities and binds its name in that
    // commit, and replicas discover the index by observing it in their
    // applied stream and run their own local backfill (the HNSW graph itself
    // is never replicated, only the data is). The commit's log index is
    // folded into operationTime, so a causal read after a CREATE VECTOR INDEX
    // fences past the definition's replication.
    let def = ctx.publish_index_in_catalog(descriptor)?;

    // Register the empty HNSW graph in memory with its tier handle,
    // keyed by the ids bound above.
    let tier = registry.tier_handle(label_id, property_id);
    registry.register_for_build(def.clone(), tier);

    let shard_id = ctx.shard_id;

    // The live HNSW handle, `Arc<RwLock<HnswIndex>>`: cheap to clone and
    // valid for the lifetime of the registry entry the build fills.
    let hnsw = registry.get(label, property).ok_or_else(|| {
        ExecutionError::Unsupported(format!(
            "vector index '{name}' was not registered: its definition carries no vector config"
        ))
    })?;

    // The build runs on its own thread, owned by the registry, and the
    // statement returns once it has started. Its readiness is this member's
    // own: the graph's health signal carries it, and every member builds
    // its graph from the data it holds; the replicated definition says only
    // that the index exists. A member that restarts rebuilds the graph on
    // open.
    let engine = Arc::clone(engine_arc);
    let label_owned = label.to_string();
    let name_owned = name.to_string();
    let token = registry.new_build_token();
    let build_token = token.clone();
    // The build publishes through the index's health signal, not
    // the registry: the signal is atomic and `Arc`-shared, so the
    // thread needs neither a borrow of the registry nor a lock on
    // the search path to report where it is.
    let health = registry.health_handle(label, property).ok_or_else(|| {
        ExecutionError::Unsupported(format!(
            "vector index '{name}' has no health signal to publish build progress on"
        ))
    })?;
    let unbuilt = Arc::clone(&health);

    let thread = std::thread::Builder::new()
        .name(format!("vec-backfill-{name}"))
        .spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::index::VectorBuild {
                    engine: engine.as_ref(),
                    token: &build_token,
                    shard_id,
                    targets: &[crate::index::BuildTarget {
                        hnsw: hnsw.as_ref(),
                        health: health.as_ref(),
                        label: &label_owned,
                        field_id: property_id,
                    }],
                }
                .run()
            }));
            match outcome {
                // The build marked the index ready after its last fold, only
                // from the handed-over state; marking it again here would
                // overrule a rebuild started since.
                Ok(Ok(crate::index::BuildOutcome::Complete { scanned })) => {
                    tracing::info!(
                        index = %name_owned,
                        scanned,
                        "vector index backfill complete"
                    );
                }
                // Cancelled: the index is being dropped or replaced by
                // whoever cancelled us, who owns it from here.
                Ok(Ok(crate::index::BuildOutcome::Cancelled)) => {}
                Ok(Err(reason)) => health.mark_offline(reason),
                Err(panic) => {
                    let reason = panic
                        .downcast_ref::<&'static str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "panic in backfill thread".to_string());
                    health.mark_offline(reason);
                }
            }
        })
        .map_err(|e| {
            // Registered rebuilding with no build behind it, the
            // index would hold a blocked reader until its timeout.
            unbuilt.mark_offline(format!("could not start the build: {e}"));
            ExecutionError::Unsupported(format!("spawn backfill thread: {e}"))
        })?;
    registry.register_build(&def, &token, thread);

    // The build has only started: nothing is indexed yet, and the state
    // reported is the one persisted above.
    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("label".to_string(), Value::String(label.to_string()));
    row.insert("property".to_string(), Value::String(property.to_string()));
    row.insert("nodes_indexed".to_string(), Value::Int(0));
    row.insert("state".to_string(), Value::String("building".to_string()));
    Ok(vec![row])
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
    let Some(registry) = ctx.vector_index_registry() else {
        return Err(ExecutionError::Unsupported(
            "DROP VECTOR INDEX requires vector indexes in ExecutionContext".into(),
        ));
    };

    // Find the definition by name to get (label, property).
    let def = registry
        .all_definitions()
        .into_iter()
        .find(|d| d.name.as_deref() == Some(name));

    let Some(def) = def else {
        return Err(ExecutionError::Unsupported(format!(
            "vector index '{name}' not found"
        )));
    };

    let label = def.label.clone();
    let property = def.property().to_string();

    // Stop the backfill before touching the definition. A build still running
    // owns the index's persisted state; dropping out from under it would race
    // its writes. Cancelling joins the thread, so once this returns the
    // definition has exactly one writer left: this statement.
    if registry.cancel_build(def.generation) {
        // The cancelled build may have written state after this statement drew
        // its read snapshot, which conflict detection would (correctly) read as
        // a write from the future. The build is now joined and can write no
        // more, so re-pinning the snapshot to now puts every write it ever made
        // in the past — the delete below is then judged against a snapshot that
        // actually contains them.
        ctx.refresh_read_snapshot();
    }

    // Tombstone the definition transactionally through the index store.
    ctx.mvcc_delete_index_def(&def)?;

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
/// The definition is published as building, with any entries a dropped
/// index of the same name left removed, in one catalog commit. Writers
/// maintain the index from then on; the backfill fills in the nodes already
/// stored; the definition is then published as ready, and only then answers
/// lookups. Stored data that breaks a unique index fails the statement and
/// withdraws the index.
#[allow(clippy::too_many_arguments)]
fn execute_create_btree_index(
    name: &str,
    label: &str,
    property: &str,
    unique: bool,
    sparse: bool,
    filter: Option<&crate::index::definition::PartialFilter>,
    maintenance: Option<crate::index::IndexProfile>,
    on_duplicate_rename: Option<&str>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    if on_duplicate_rename.is_some() && !unique {
        return Err(ExecutionError::CatalogRefused(
            "ON DUPLICATE RENAME repairs duplicates of a unique index; this index is not unique"
                .into(),
        ));
    }
    // Indexes and constraints share one namespace: a uniqueness constraint's
    // index carries the constraint's name.
    if ctx.constraint_label(name)?.is_some() {
        return Err(ExecutionError::CatalogObjectExists {
            object: CatalogObject::Constraint,
            name: name.to_string(),
        });
    }
    // A unique index declares the invariant a uniqueness constraint does, so
    // it is that constraint, owning the index: the constraint catalog shows
    // every uniqueness the engine enforces, and a constraint declared later
    // over the same property finds it. A partial one is a uniqueness among
    // the nodes its predicate admits: the constraint carries that scope.
    if unique {
        let constrained = execute_create_constraint(
            Some(name),
            false,
            label,
            &[property.to_string()],
            &coordinode_core::schema::definition::ConstraintKind::Unique,
            OwnedIndexShape {
                sparse: Some(sparse),
                maintenance,
                wait: None,
                on_duplicate_rename: on_duplicate_rename.map(str::to_string),
                scope: filter.cloned(),
            },
            ctx,
        )?;
        let mut row = Row::new();
        row.insert("index".to_string(), Value::String(name.to_string()));
        row.insert("label".to_string(), Value::String(label.to_string()));
        row.insert("property".to_string(), Value::String(property.to_string()));
        row.insert("unique".to_string(), Value::Bool(true));
        row.insert("sparse".to_string(), Value::Bool(sparse));
        // The constraint's build is the index's: the same operation, the
        // index ready exactly when the constraint is active.
        if let Some(constrained) = constrained.first() {
            for column in ["operation", "nodes_indexed"] {
                row.insert(
                    column.to_string(),
                    constrained.get(column).cloned().unwrap_or(Value::Null),
                );
            }
            let state = match constrained.get("state") {
                Some(Value::String(s)) if s == "ACTIVE" => "READY",
                _ => "BUILDING",
            };
            row.insert("state".to_string(), Value::String(state.into()));
        }
        if let Some(def) = ctx.btree_index_registry.and_then(|r| r.get(name)) {
            insert_maintenance(&mut row, &def.maintenance);
        }
        return Ok(vec![row]);
    }
    let mut descriptor = crate::index::IndexDescriptor::btree(name, label, property);
    if sparse {
        descriptor = descriptor.sparse();
    }
    if let Some(f) = filter {
        descriptor = descriptor.with_filter(f.clone());
    }
    let def = publish_index_build(descriptor, maintenance, None, ctx, |_| Ok(()));
    ctx.label_schema_cache.remove(label);
    let def = def?;
    let waited = await_index_build(&def, None, ctx)?;

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("label".to_string(), Value::String(label.to_string()));
    row.insert("property".to_string(), Value::String(property.to_string()));
    row.insert("unique".to_string(), Value::Bool(false));
    row.insert("sparse".to_string(), Value::Bool(sparse));
    insert_build(&mut row, &def, waited);
    row.insert("state".to_string(), index_state(waited));
    insert_maintenance(&mut row, &def.maintenance);
    Ok(vec![row])
}

/// The error a catalog commit that lost the name race fails with: the
/// index exists, as when the name was found taken before the commit.
fn name_taken(e: ExecutionError) -> ExecutionError {
    match e {
        ExecutionError::Modality(coordinode_modality::StoreError::IndexNameTaken(name)) => {
            ExecutionError::CatalogObjectExists {
                object: CatalogObject::Index,
                name,
            }
        }
        other => other,
    }
}

/// The repair `ON DUPLICATE RENAME target` lets the build of a unique index
/// over `properties` of `label` make, refused before anything is published
/// unless the build can make it: the target is one of the properties, and a
/// string the label lets be rewritten in place.
fn duplicate_repair(
    target: Option<&str>,
    label: &str,
    properties: &[String],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Option<crate::index::DuplicateRepair>, ExecutionError> {
    let Some(target) = target else {
        return Ok(None);
    };
    let refused = |why: String| {
        Err(ExecutionError::CatalogRefused(format!(
            "ON DUPLICATE RENAME {target}: {why}"
        )))
    };
    if !properties.iter().any(|p| p == target) {
        return refused(format!(
            "the repaired property must be one the uniqueness covers ({})",
            properties.join(", ")
        ));
    }
    if let Some(schema) = ctx.load_current_label_schema(label)? {
        if schema.temporal {
            // Every version a temporal node held keeps its value, so a
            // rewrite of the current one leaves the duplicate in the history
            // the index covers.
            return refused(format!(
                ":{label} is temporal; a repair cannot rewrite the versions it keeps"
            ));
        }
        if schema.is_columnar() {
            return refused(format!(":{label} is a COLUMNAR table"));
        }
        match schema.properties.get(target).map(|p| &p.property_type) {
            None | Some(PropertyType::String) => {}
            Some(PropertyType::Computed(_)) => {
                return refused(format!("`{target}` of :{label} is computed and read-only"));
            }
            Some(other) => {
                return refused(format!(
                    "`{target}` of :{label} is declared {other}; a repair appends a suffix to \
                     a string"
                ));
            }
        }
    }
    Ok(Some(crate::index::DuplicateRepair {
        property: target.to_string(),
    }))
}

/// Publish B-tree index `descriptor` as building, with its build admitted,
/// in one catalog commit together with `with`: the catalog gives it a new
/// identity and generation and binds its name, on the condition that no
/// live index holds the name when it commits. The generation is new, so no
/// entry of a dropped index can be under it. Writers maintain the index from
/// this commit on; the build belongs to the engine's executor.
fn publish_index_build(
    mut descriptor: crate::index::IndexDescriptor,
    maintenance: Option<crate::index::IndexProfile>,
    repair: Option<crate::index::DuplicateRepair>,
    ctx: &mut ExecutionContext<'_>,
    mut with: impl FnMut(
        &mut coordinode_storage::engine::transaction::Transaction<'_>,
    ) -> Result<(), coordinode_modality::StoreError>,
) -> Result<crate::index::IndexDefinition, ExecutionError> {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    let Some(registry) = ctx.btree_index_registry else {
        return Err(ExecutionError::Unsupported(
            "CREATE INDEX requires btree_index_registry in ExecutionContext".into(),
        ));
    };
    if let Some(name) = &descriptor.name {
        if registry.get(name).is_some() {
            return Err(ExecutionError::CatalogObjectExists {
                object: CatalogObject::Index,
                name: name.clone(),
            });
        }
    }

    let engine = ctx.engine;
    let store = LocalIndexStore::new(engine);
    // The binding is resolved once, here, and recorded: a later change of
    // the namespace default does not reinterpret this index.
    let (policy, _) = store.index_policy()?;
    descriptor.maintenance = policy.resolve(maintenance, 1);
    descriptor.state = IndexState::Building {
        written: 0,
        estimated_total: 0,
    };

    let mut published = None;
    ctx.commit_catalog_change(|txn| {
        let def = store.publish_definition_txn(txn, descriptor)?;
        store.put_build_txn(
            txn,
            &crate::index::IndexBuildRecord::accepted(
                def.id,
                def.generation,
                crate::index::BuildFailure::Withdraw,
            )
            .repairing(repair.clone()),
            None,
        )?;
        published = Some(def);
        with(txn)
    })
    .map_err(name_taken)?;
    let def = published.ok_or_else(|| {
        ExecutionError::Unsupported("the index publication staged no definition".into())
    })?;
    registry.register_published(engine, def.clone())?;
    Ok(def)
}

/// How a statement's wait for the build of the index it created ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuildWait {
    /// The index was published ready, with this many nodes indexed.
    Published(u64),
    /// The wait ran out first. The build goes on without the statement and
    /// is inspected, awaited or cancelled through its operation.
    Running,
}

/// Run the admitted build of `def` on the engine's executor and wait up to
/// `wait` (`None`: the engine's statement wait) for its outcome. A build
/// that fails or is cancelled within the wait is the error the statement
/// fails with. The executor publishes the index ready, or withdraws it with
/// the constraint that owns it, itself, whether anyone still waits or not.
fn await_index_build(
    def: &crate::index::IndexDefinition,
    wait: Option<Duration>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<BuildWait, ExecutionError> {
    use crate::index::{BuildError, IndexBuildOutcome};
    let builds = ctx.index_builds.ok_or_else(|| {
        ExecutionError::Unsupported("building an index requires the engine's index builds".into())
    })?;
    // The statement's own transaction is open while it waits. Having staged
    // nothing, it leaves the backfill's wait, which would otherwise wait for
    // the statement waiting for it; one of an interactive transaction that
    // already staged node writes stays, and the build ends after it does.
    ctx.txn.release_from_schema_waits();
    builds
        .submit(def.generation)
        .map_err(ExecutionError::Unsupported)?;
    let wait = wait.unwrap_or_else(|| builds.config().statement_wait);
    match builds.wait(def.generation, Some(wait))? {
        Some(IndexBuildOutcome::Published { indexed }) => {
            Ok(BuildWait::Published(indexed.unwrap_or(0)))
        }
        Some(IndexBuildOutcome::Failed(BuildError::Duplicate(v))) => Err(unique_violation(v)),
        Some(IndexBuildOutcome::Failed(BuildError::Other(reason))) => {
            Err(ExecutionError::Unsupported(reason))
        }
        Some(IndexBuildOutcome::Cancelled) => Err(ExecutionError::Unsupported(format!(
            "the build of index '{def}' was cancelled"
        ))),
        None => Ok(BuildWait::Running),
    }
}

/// The result columns naming the build of an index a statement created:
/// its operation, and the nodes it indexed once published (`Null` while it
/// still runs).
fn insert_build(row: &mut Row, def: &crate::index::IndexDefinition, wait: BuildWait) {
    row.insert(
        "operation".to_string(),
        Value::Int(operation_value(def.generation)),
    );
    let indexed = match wait {
        BuildWait::Published(indexed) => Value::Int(count_value(indexed)),
        BuildWait::Running => Value::Null,
    };
    row.insert("nodes_indexed".to_string(), indexed);
}

/// The `state` column of an index a statement created.
fn index_state(wait: BuildWait) -> Value {
    Value::String(
        match wait {
            BuildWait::Published(_) => "READY",
            BuildWait::Running => "BUILDING",
        }
        .into(),
    )
}

/// A build operation as a query value: its generation.
pub(crate) fn operation_value(generation: crate::index::GenerationId) -> i64 {
    // Generations are allocated one at a time from one catalog counter
    // starting at 0, so they stay far below `i64::MAX` for the life of a
    // database.
    i64::try_from(generation.as_raw()).unwrap_or(i64::MAX)
}

/// A count as a query value.
fn count_value(n: u64) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// The maintenance binding of an index as result columns: the effective
/// profile, where it comes from, and its epoch.
fn insert_maintenance(row: &mut Row, maintenance: &crate::index::IndexMaintenance) {
    let profile = match maintenance.profile {
        crate::index::IndexProfile::Resolved => "RESOLVED",
        crate::index::IndexProfile::Derived => "DERIVED",
    };
    let source = match maintenance.source {
        crate::index::ProfileSource::Override => "OVERRIDE".to_string(),
        crate::index::ProfileSource::Namespace { revision } => {
            format!("NAMESPACE@{revision}")
        }
    };
    row.insert("maintenance".to_string(), Value::String(profile.into()));
    row.insert("maintenance_source".to_string(), Value::String(source));
    row.insert(
        "maintenance_epoch".to_string(),
        Value::Int(i64::try_from(maintenance.epoch).unwrap_or(i64::MAX)),
    );
}

/// Execute `ALTER INDEX idx SET MAINTENANCE ...`: an explicit transition to
/// `profile`, or to the namespace default when `None`, under a new policy
/// epoch. The entries' layout is the same in both profiles, so no entry is
/// rewritten: effects sealed under the old epoch keep their own binding, and
/// a writer whose effects were staged under it is refused at commit and
/// retried under the new one.
fn execute_alter_index_maintenance(
    name: &str,
    profile: Option<crate::index::IndexProfile>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    let Some(registry) = ctx.btree_index_registry else {
        return Err(ExecutionError::Unsupported(
            "ALTER INDEX requires btree_index_registry in ExecutionContext".into(),
        ));
    };
    let store = LocalIndexStore::new(ctx.engine);
    let (mut def, version) = stored_definition(name, ctx.engine)?
        .ok_or_else(|| ExecutionError::Unsupported(format!("index '{name}' not found")))?;
    if def.index_type != crate::index::IndexType::BTree {
        return Err(ExecutionError::Unsupported(format!(
            "index '{name}' is not key-shaped: its maintenance follows its own class"
        )));
    }
    if def.state != IndexState::Ready {
        return Err(ExecutionError::Unsupported(format!(
            "index '{name}' is still being built; change its maintenance once it is ready"
        )));
    }
    let (policy, _) = store.index_policy()?;
    let epoch = def.maintenance.epoch.checked_add(1).ok_or_else(|| {
        ExecutionError::Unsupported(format!("index '{name}' has no maintenance epoch left"))
    })?;
    let from = def.maintenance;
    def.maintenance = policy.resolve(profile, epoch);
    let staged = def.clone();
    ctx.commit_catalog_change(|txn| {
        store.expect_definition_txn(txn, staged.id, version)?;
        store.put_definition_txn(txn, &staged)
    })?;
    let to = def.maintenance;
    registry.register_published(ctx.engine, def)?;

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert(
        "previous_epoch".to_string(),
        Value::Int(i64::try_from(from.epoch).unwrap_or(i64::MAX)),
    );
    insert_maintenance(&mut row, &to);
    Ok(vec![row])
}

/// Execute `ALTER NAMESPACE SET INDEX MAINTENANCE ...`: the default for
/// indexes created from now on, at a new policy revision. Existing indexes
/// keep their binding until their own transition.
fn execute_set_namespace_index_default(
    profile: crate::index::IndexProfile,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    let store = LocalIndexStore::new(ctx.engine);
    let (current, version) = store.index_policy()?;
    let revision = current.revision.checked_add(1).ok_or_else(|| {
        ExecutionError::Unsupported("the namespace index policy has no revision left".into())
    })?;
    let policy = crate::index::NamespaceIndexPolicy {
        default: profile,
        revision,
    };
    ctx.commit_catalog_change(|txn| store.put_index_policy_txn(txn, &policy, version))?;

    let mut row = Row::new();
    let name = match profile {
        crate::index::IndexProfile::Resolved => "RESOLVED",
        crate::index::IndexProfile::Derived => "DERIVED",
    };
    row.insert(
        "index_maintenance_default".to_string(),
        Value::String(name.into()),
    );
    row.insert(
        "revision".to_string(),
        Value::Int(i64::try_from(revision).unwrap_or(i64::MAX)),
    );
    Ok(vec![row])
}

/// Execute `REINDEX idx [ON :Label]`: rebuild the B-tree index from its
/// records into a fresh generation, and wait for the build as `CREATE INDEX`
/// does. Writers maintain the new generation at once; lookups answer from
/// the records until it is built, and its unique values are proved free
/// from them. A constraint the index enforces stays enforced throughout.
fn execute_reindex(
    name: &str,
    label: Option<&str>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let builds = ctx.index_builds.ok_or_else(|| {
        ExecutionError::Unsupported("REINDEX requires the engine's index builds".into())
    })?;
    let (def, _) = stored_definition(name, ctx.engine)?.ok_or_else(|| {
        ExecutionError::CatalogObjectMissing {
            object: CatalogObject::Index,
            name: name.to_string(),
        }
    })?;
    if let Some(label) = label {
        if def.label != label {
            return Err(ExecutionError::CatalogRefused(format!(
                "index '{name}' is on :{}, not :{label}",
                def.label
            )));
        }
    }
    if def.index_type != crate::index::IndexType::BTree {
        return Err(ExecutionError::CatalogRefused(format!(
            "index '{name}' is not a B-tree index; REINDEX rebuilds B-tree indexes"
        )));
    }
    let def = builds.rebuild(def).map_err(ExecutionError::Unsupported)?;
    let wait = await_index_build(&def, None, ctx)?;
    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("label".to_string(), Value::String(def.label.clone()));
    row.insert("state".to_string(), index_state(wait));
    insert_build(&mut row, &def, wait);
    Ok(vec![row])
}

/// Execute `DROP INDEX idx`: the definition and every entry go in one
/// catalog commit, on the condition that the definition is still the one
/// inspected here, and the index stops being maintained once it is durable.
/// The index a constraint enforces goes only with the constraint.
fn execute_drop_btree_index(
    name: &str,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    let Some(registry) = ctx.btree_index_registry else {
        return Err(ExecutionError::Unsupported(
            "DROP INDEX requires btree_index_registry in ExecutionContext".into(),
        ));
    };
    let engine = ctx.engine;
    let store = LocalIndexStore::new(engine);
    let (def, version) =
        stored_definition(name, engine)?.ok_or_else(|| ExecutionError::CatalogObjectMissing {
            object: CatalogObject::Index,
            name: name.to_string(),
        })?;
    if let Some(constraint) = &def.owner {
        return Err(ExecutionError::CatalogRefused(format!(
            "index '{name}' belongs to constraint '{constraint}'; drop the constraint instead"
        )));
    }

    let label = def.label.clone();
    let property = def.property().to_string();
    // Bound to the object the name resolved to: a statement that resolved
    // the name before it was rebound drops nothing of the later index.
    // Its finished build records go with it; a build still running has its
    // pages fenced by the deleted definition and removes its own record. Its
    // integrity records go too, which ends a check still running on it.
    ctx.commit_catalog_change(|txn| {
        store.expect_definition_txn(txn, def.id, version)?;
        store.delete_definition_txn(txn, &def)?;
        store.delete_finished_builds_txn(txn, def.id)?;
        store.delete_integrity_txn(txn, def.id)?;
        store.clear_txn(txn, def.generation)
    })?;
    registry.unregister(def.id);

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
    row.insert("label".to_string(), Value::String(label));
    row.insert("property".to_string(), Value::String(property));
    row.insert("dropped".to_string(), Value::Bool(true));
    Ok(vec![row])
}

/// The stored definition of the index the name `name` binds and the version
/// of its record, read as one: a name rebound or a definition replaced
/// between the reads is reported as a concurrent change rather than paired
/// with another object or another version.
fn stored_definition(
    name: &str,
    engine: &StorageEngine,
) -> Result<Option<(crate::index::IndexDefinition, Option<u64>)>, ExecutionError> {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    let store = LocalIndexStore::new(engine);
    let binding = store.name_version(name)?;
    let Some(id) = store.resolve_name(name)? else {
        return Ok(None);
    };
    let version = store.definition_version(id)?;
    let def = store.load_definition(id)?;
    if store.name_version(name)? != binding || store.definition_version(id)? != version {
        return Err(ExecutionError::Unsupported(format!(
            "the definition of index '{name}' changed concurrently; retry the statement"
        )));
    }
    Ok(def.map(|d| (d, version)))
}

/// The name a constraint created without one gets: its label, properties and
/// kind, so repeating the statement names the same constraint.
fn derived_constraint_name(
    label: &str,
    properties: &[String],
    kind: &coordinode_core::schema::definition::ConstraintKind,
) -> String {
    use coordinode_core::schema::definition::ConstraintKind;
    let suffix = match kind {
        ConstraintKind::Unique => "unique",
        ConstraintKind::NotNull => "not_null",
        ConstraintKind::NodeKey => "node_key",
        ConstraintKind::Type(_) => "type",
    };
    format!("{label}_{}_{suffix}", properties.join("_"))
}

/// The result row of constraint DDL.
fn constraint_row(
    constraint: &coordinode_core::schema::definition::NodeConstraint,
    label: &str,
    changed: (&str, bool),
) -> Row {
    let mut row = Row::new();
    row.insert(
        "constraint".to_string(),
        Value::String(constraint.name.clone()),
    );
    row.insert("label".to_string(), Value::String(label.to_string()));
    row.insert(
        "properties".to_string(),
        Value::Array(
            constraint
                .properties
                .iter()
                .map(|p| Value::String(p.clone()))
                .collect(),
        ),
    );
    row.insert(
        "kind".to_string(),
        Value::String(constraint.kind.to_string()),
    );
    row.insert(
        "state".to_string(),
        Value::String(constraint.state.to_string()),
    );
    row.insert(
        "scope".to_string(),
        constraint
            .scope
            .as_ref()
            .map_or(Value::Null, |s| Value::String(s.to_string())),
    );
    row.insert(changed.0.to_string(), Value::Bool(changed.1));
    row
}

/// How the index a uniqueness or key constraint owns is built, and how long
/// the statement waits for that build.
#[derive(Debug, Clone, Default)]
struct OwnedIndexShape {
    /// Leave nodes missing a value out of the index; `None` takes the
    /// constraint kind's own choice.
    sparse: Option<bool>,
    /// The index's maintenance profile; `None` takes the namespace default.
    maintenance: Option<crate::index::IndexProfile>,
    /// How long the statement waits for the build; `None` takes the
    /// engine's statement wait.
    wait: Option<Duration>,
    /// `ON DUPLICATE RENAME prop`: the property the build may change to
    /// repair a stored duplicate.
    on_duplicate_rename: Option<String>,
    /// The nodes a uniqueness holds among (`CREATE UNIQUE INDEX ... WHERE`):
    /// the constraint's scope and the filter of the index it owns.
    scope: Option<crate::index::definition::PartialFilter>,
}

/// The build operation of the index constraint `name` owns while it is
/// validating, as the result column `operation`; `Null` when no index of
/// the name is published.
fn owned_build_operation(name: &str, ctx: &ExecutionContext<'_>) -> Value {
    ctx.btree_index_registry
        .and_then(|r| r.get(name))
        .filter(|d| d.owner.as_deref() == Some(name))
        .map_or(Value::Null, |d| Value::Int(operation_value(d.generation)))
}

/// Execute `CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR (n:Label) REQUIRE ...`.
///
/// A presence or type constraint lands as a new revision of the label's
/// schema whose commit checks every stored node of the label against it,
/// with writers validated under the old revision held out; those committing
/// later are refused and retried under the new one.
///
/// A uniqueness or key constraint lands the same way, as validating, in the
/// one commit that publishes the unique index it owns and admits its build:
/// writers enforce it from that commit on while the build checks the stored
/// nodes, and the build's last commit, bound to the index definition the
/// first published, makes both active, or withdraws both when stored nodes
/// break it. The statement waits for that outcome up to its bound; past it,
/// it returns the constraint validating with the build's operation, and the
/// build goes on without it. Asking again for the same constraint while it
/// validates (`IF NOT EXISTS`, as after a lost reply) returns that same
/// constraint and operation; no second build is admitted.
fn execute_create_constraint(
    name: Option<&str>,
    if_not_exists: bool,
    label: &str,
    properties: &[String],
    kind: &coordinode_core::schema::definition::ConstraintKind,
    shape: OwnedIndexShape,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_core::schema::definition::{ConstraintKind, ConstraintState, NodeConstraint};
    use coordinode_modality::{LocalSchemaStore, SchemaStore as _, StoreError};

    let constraint = NodeConstraint {
        name: name.map_or_else(
            || derived_constraint_name(label, properties, kind),
            str::to_string,
        ),
        properties: properties.to_vec(),
        kind: kind.clone(),
        state: ConstraintState::Validating,
        scope: shape.scope.clone(),
    };
    // A scope narrows which nodes are compared with each other; a key also
    // requires its values of every node, which a scope cannot narrow.
    if constraint.scope.is_some() && constraint.kind != ConstraintKind::Unique {
        return Err(ExecutionError::CatalogRefused(format!(
            "only a uniqueness can hold among the nodes a predicate admits, not an IS {kind} \
             constraint"
        )));
    }
    if shape.on_duplicate_rename.is_some() && !constraint.owns_index() {
        return Err(ExecutionError::CatalogRefused(format!(
            "ON DUPLICATE RENAME repairs duplicates of a UNIQUE or NODE KEY constraint, not of \
             an IS {kind} one"
        )));
    }
    let repair = duplicate_repair(shape.on_duplicate_rename.as_deref(), label, properties, ctx)?;

    if let Some(holder) = ctx.constraint_label(&constraint.name)? {
        if if_not_exists {
            let existing = ctx
                .load_current_label_schema(&holder)?
                .and_then(|s| s.constraint(&constraint.name).cloned());
            let existing = existing.as_ref().unwrap_or(&constraint);
            let mut row = constraint_row(existing, &holder, ("created", false));
            row.insert(
                "operation".to_string(),
                owned_build_operation(&existing.name, ctx),
            );
            return Ok(vec![row]);
        }
        return Err(ExecutionError::CatalogObjectExists {
            object: CatalogObject::Constraint,
            name: constraint.name,
        });
    }
    let schema = ctx.load_current_label_schema(label)?;
    if let Some(existing) = schema.as_ref().and_then(|s| {
        s.constraints()
            .iter()
            .find(|c| c.same_requirement(&constraint))
    }) {
        if if_not_exists {
            let mut row = constraint_row(existing, label, ("created", false));
            row.insert(
                "operation".to_string(),
                owned_build_operation(&existing.name, ctx),
            );
            return Ok(vec![row]);
        }
        return Err(ExecutionError::CatalogObjectExists {
            object: CatalogObject::Constraint,
            name: existing.name.clone(),
        });
    }
    // The rows of a COLUMNAR table are written outside the transaction a
    // schema revision binds, so nothing could hold them to the constraint
    // while it is enabled.
    if schema.as_ref().is_some_and(LabelSchema::is_columnar) {
        return Err(ExecutionError::CatalogRefused(format!(
            "constraints on the COLUMNAR table '{label}' are not supported"
        )));
    }
    if let Some(conflict) = schema
        .as_ref()
        .and_then(|s| s.constraint_conflict(&constraint))
    {
        return Err(ExecutionError::CatalogRefused(format!(
            "constraint '{}' cannot hold: {conflict}",
            constraint.name
        )));
    }
    if constraint.owns_index()
        && ctx
            .btree_index_registry
            .is_some_and(|r| r.get(&constraint.name).is_some())
    {
        return Err(ExecutionError::CatalogObjectExists {
            object: CatalogObject::Index,
            name: constraint.name,
        });
    }

    let read_revision = schema.as_ref().map(|s| s.schema_revision);
    let mut next = match schema {
        Some(mut s) => {
            s.schema_revision = next_revision(&s)?;
            s
        }
        // A label without a schema is enforced as FLEXIBLE; the schema the
        // constraint creates keeps that.
        None => {
            let mut s = LabelSchema::new_node_id(label);
            s.set_mode(SchemaMode::Flexible);
            s
        }
    };
    let mut published = constraint.clone();
    if !constraint.owns_index() {
        published.state = ConstraintState::Active;
    }
    next.add_constraint(published.clone());

    // The commit decides this authoritatively; checking here as well names
    // the node that refuses it.
    if constraint.checks_each_node() {
        let staged = std::collections::HashMap::new();
        if let Some(violation) =
            coordinode_storage::engine::claims::evaluate::first_label_schema_violation(
                ctx.engine, &next, &staged,
            )?
        {
            return Err(ExecutionError::SchemaViolation(format!(
                "cannot create constraint '{}': node {} breaks it ({})",
                constraint.name,
                violation.node.to_element_id(),
                violation.reason
            )));
        }
    }

    let engine = ctx.engine;
    // The revision this statement checked must still be the one in force
    // when its successor lands, and the name must still be free.
    let mut stage_constraint =
        |txn: &mut coordinode_storage::engine::transaction::Transaction<'_>| {
            let store = LocalSchemaStore::new(engine);
            let current = store.load_label_for_update_txn(txn, label)?;
            if current.map(|s| s.schema_revision) != read_revision {
                return Err(StoreError::Invariant(format!(
                    "the schema of :{label} changed while the constraint was being created; \
                     retry the statement"
                )));
            }
            store.save_label_txn(txn, &next)?;
            store.claim_constraint_name_txn(txn, &constraint.name, label)
        };

    if !constraint.owns_index() {
        let committed = ctx.commit_catalog_change(stage_constraint);
        ctx.label_schema_cache.remove(label);
        committed?;
        // Checked on each node in the commit itself: no build, no operation.
        let mut row = constraint_row(&published, label, ("created", true));
        row.insert("operation".to_string(), Value::Null);
        return Ok(vec![row]);
    }

    let mut descriptor =
        crate::index::IndexDescriptor::compound(&constraint.name, label, properties.to_vec())
            .unique()
            .owned_by(&constraint.name);
    // A node missing a value is not constrained by uniqueness; a key
    // requires the values, which its per-node part enforces.
    if shape
        .sparse
        .unwrap_or(constraint.kind == ConstraintKind::Unique)
    {
        descriptor = descriptor.sparse();
    }
    if let Some(scope) = &constraint.scope {
        descriptor = descriptor.with_filter(scope.clone());
    }
    let def = publish_index_build(
        descriptor,
        shape.maintenance,
        repair,
        ctx,
        &mut stage_constraint,
    );
    ctx.label_schema_cache.remove(label);
    let def = def?;

    // The executor activates the constraint with the index, or withdraws it
    // with the index, in the commit that ends the build.
    let waited = await_index_build(&def, shape.wait, ctx);
    ctx.label_schema_cache.remove(label);
    let waited = waited?;

    if matches!(waited, BuildWait::Published(_)) {
        published.state = ConstraintState::Active;
    }
    let mut row = constraint_row(&published, label, ("created", true));
    insert_build(&mut row, &def, waited);
    Ok(vec![row])
}

/// Stage, in the catalog commit that publishes a validated index as ready,
/// the activation of constraint `name` of `label` that owns it: a new
/// revision of the label's latest schema with the constraint active. A
/// constraint already active (its index rebuilt) stays as it is: a rebuild
/// never suspends it. Fails when the constraint is gone (dropped meanwhile).
///
/// # Errors
///
/// The constraint is gone, or the schema could not be read or staged.
pub fn stage_constraint_activation(
    engine: &StorageEngine,
    txn: &mut coordinode_storage::engine::transaction::Transaction<'_>,
    label: &str,
    name: &str,
) -> Result<(), coordinode_modality::StoreError> {
    use coordinode_core::schema::definition::ConstraintState;
    use coordinode_modality::{LocalSchemaStore, SchemaStore as _, StoreError};
    let store = LocalSchemaStore::new(engine);
    let mut latest = store
        .load_label_for_update_txn(txn, label)?
        .ok_or_else(|| {
            StoreError::Invariant(format!("the schema of :{label} was dropped meanwhile"))
        })?;
    let Some(constraint) = latest.constraint_mut(name) else {
        return Err(StoreError::Invariant(format!(
            "constraint '{name}' was dropped while it was being validated"
        )));
    };
    if constraint.state == ConstraintState::Active {
        return Ok(());
    }
    constraint.state = ConstraintState::Active;
    latest.schema_revision = next_revision(&latest).map_err(store_error)?;
    store.save_label_admitting_txn(txn, &latest)
}

/// Stage, in the catalog commit that withdraws an index whose validation
/// failed, the withdrawal of constraint `name` of `label` that owns it: a
/// new revision of the label's latest schema without it, and its name
/// released.
///
/// # Errors
///
/// The schema or the name record could not be read or staged.
pub fn stage_constraint_withdrawal(
    engine: &StorageEngine,
    txn: &mut coordinode_storage::engine::transaction::Transaction<'_>,
    label: &str,
    name: &str,
) -> Result<(), coordinode_modality::StoreError> {
    use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
    let store = LocalSchemaStore::new(engine);
    if let Some(mut latest) = store.load_label_for_update_txn(txn, label)? {
        if latest.remove_constraint(name).is_some() {
            latest.schema_revision = next_revision(&latest).map_err(store_error)?;
            store.save_label_admitting_txn(txn, &latest)?;
        }
    }
    store.release_constraint_name_txn(txn, name)
}

/// The revision a change of `schema` is published at.
fn next_revision(
    schema: &coordinode_core::schema::definition::LabelSchema,
) -> Result<u64, ExecutionError> {
    schema.schema_revision.checked_add(1).ok_or_else(|| {
        ExecutionError::Unsupported(format!(
            "label '{}' has no schema revision left",
            schema.name
        ))
    })
}

/// An execution error raised while staging a catalog change.
fn store_error(e: ExecutionError) -> coordinode_modality::StoreError {
    coordinode_modality::StoreError::Invariant(e.to_string())
}

/// Execute `DROP CONSTRAINT name [IF EXISTS]`: one catalog commit removes the
/// constraint from a new revision of the label's schema, releases its name,
/// and removes the index it owns with every entry, on the condition that
/// the index definition is still the one inspected here. Writers stop
/// maintaining the index once that commit is durable.
fn execute_drop_constraint(
    name: &str,
    if_exists: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use coordinode_modality::{
        IndexStore as _, LocalIndexStore, LocalSchemaStore, SchemaStore as _, StoreError,
    };

    let Some(label) = ctx.constraint_label(name)? else {
        if if_exists {
            let mut row = Row::new();
            row.insert("constraint".to_string(), Value::String(name.to_string()));
            row.insert("dropped".to_string(), Value::Bool(false));
            return Ok(vec![row]);
        }
        return Err(ExecutionError::CatalogObjectMissing {
            object: CatalogObject::Constraint,
            name: name.to_string(),
        });
    };
    let schema = ctx.load_current_label_schema(&label)?;
    let read_revision = schema.as_ref().map(|s| s.schema_revision);
    let Some(mut next) = schema else {
        return Err(ExecutionError::Unsupported(format!(
            "constraint '{name}' names :{label}, which has no schema"
        )));
    };
    let Some(removed) = next.remove_constraint(name) else {
        return Err(ExecutionError::Unsupported(format!(
            "constraint '{name}' names :{label}, whose schema does not hold it"
        )));
    };
    next.schema_revision = next_revision(&next)?;
    // The index this constraint owns, as stored now: the commit removes it
    // only while it is still that record.
    let index = if removed.owns_index() {
        stored_definition(name, ctx.engine)?.filter(|(def, _)| def.owner.as_deref() == Some(name))
    } else {
        None
    };

    let engine = ctx.engine;
    let dropped = ctx.commit_catalog_change(|txn| {
        let store = LocalSchemaStore::new(engine);
        let current = store.load_label_for_update_txn(txn, &label)?;
        if current.map(|s| s.schema_revision) != read_revision {
            return Err(StoreError::Invariant(format!(
                "the schema of :{label} changed while the constraint was being dropped; \
                 retry the statement"
            )));
        }
        store.save_label_admitting_txn(txn, &next)?;
        store.release_constraint_name_txn(txn, name)?;
        if let Some((def, version)) = &index {
            let indexes = LocalIndexStore::new(engine);
            indexes.expect_definition_txn(txn, def.id, *version)?;
            indexes.delete_definition_txn(txn, def)?;
            indexes.delete_finished_builds_txn(txn, def.id)?;
            indexes.delete_integrity_txn(txn, def.id)?;
            indexes.clear_txn(txn, def.generation)?;
        }
        Ok(())
    });
    ctx.label_schema_cache.remove(&label);
    dropped?;
    if let Some((def, _)) = &index {
        if let Some(registry) = ctx.btree_index_registry {
            registry.unregister(def.id);
        }
    }
    Ok(vec![constraint_row(&removed, &label, ("dropped", true))])
}

/// Execute a procedure call: once per input row, with the arguments evaluated
/// against that row and checked against the signature; each output row
/// extends the input row with the yielded variables. A procedure without
/// outputs passes each input row through once.
fn execute_procedure_call(
    input: &LogicalOp,
    procedure: &str,
    args: &[crate::plan::expr::Expr],
    yields: Option<&[crate::planner::logical::YieldColumn]>,
    filter: Option<&crate::plan::expr::Expr>,
    standalone: bool,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    use crate::procedure::{ProcedureError, bind_arguments, bind_yields};

    let registry = ctx.procedures.ok_or_else(|| {
        ExecutionError::Unsupported("no procedure catalog is available here".into())
    })?;
    let callee = registry
        .get(procedure)
        .ok_or_else(|| ProcedureError::Unknown {
            procedure: procedure.to_string(),
        })?;
    let signature = callee.signature();
    // Resolved before any row runs, so a bad YIELD fails without side effects.
    let bindings = bind_yields(signature, yields, standalone)?;

    let input_rows = execute_op(input, ctx)?;
    let mut out = Vec::new();
    for row in input_rows {
        let mut arg_values = Vec::with_capacity(args.len());
        for arg in args {
            arg_values.push(eval_neutral_with_storage(arg, &row, ctx)?);
        }
        let results = callee.call(ctx, bind_arguments(signature, arg_values)?)?;
        if signature.outputs.is_empty() {
            out.push(row);
            continue;
        }
        for result in results {
            if result.len() != signature.outputs.len() {
                return Err(ProcedureError::OutputShape {
                    procedure: procedure.to_string(),
                    declared: signature.outputs.len(),
                    found: result.len(),
                }
                .into());
            }
            let mut joined = row.clone();
            for (index, variable) in &bindings {
                joined.insert(variable.clone(), result[*index].clone());
            }
            if let Some(filter) = filter {
                if !is_truthy(&eval_neutral_with_storage(filter, &joined, ctx)?) {
                    continue;
                }
            }
            out.push(joined);
        }
    }
    Ok(out)
}

/// How long a catalog change waits for the commits already admitted on its
/// keys. They hold their registration from validation to apply, including a
/// majority write, so this covers a slow replica rather than a stuck one; past
/// it the change is refused as before.
const CATALOG_ADMISSION_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// The error a catalog change fails with when its commit is refused: a record
/// it was conditioned on, or a key it wrote, changed concurrently, which a
/// retry re-reads; anything else as any commit reports it.
pub fn catalog_commit_error(
    err: coordinode_storage::engine::transaction::CommitError,
) -> ExecutionError {
    use coordinode_storage::engine::transaction::CommitError;
    match err {
        CommitError::RevisionMismatch { .. } | CommitError::Conflict(_) => {
            ExecutionError::Conflict(
                "the catalog record changed concurrently; retry the statement".into(),
            )
        }
        other => commit_err_to_execution(other),
    }
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
        CommitError::Backpressure => ExecutionError::Backpressure,
        CommitError::NotLeader { leader_id } => ExecutionError::NotLeader { leader_id },
        CommitError::Mismatched(m) => ExecutionError::Mismatched(m),
        // A caller error, not a transient one: retrying the same statement
        // stages the same deltas and is refused again.
        CommitError::CounterOverflow { key } => ExecutionError::Serialization(format!(
            "counter '{key}' would leave the i64 range; nothing was written"
        )),
        // Also the same on retry: the statement has to change fewer entries.
        e @ CommitError::IndexFanOut { .. } => ExecutionError::Serialization(e.to_string()),
        CommitError::InvariantRefused { reason } => ExecutionError::InvariantRefused(reason),
        // Named by generation here; an executor that holds the claim names
        // the index and the value before this is reached.
        e @ CommitError::UniqueValueHeld { .. } => ExecutionError::InvariantRefused(e.to_string()),
        CommitError::UniquenessUnresolved { generation, limit } => {
            ExecutionError::UniquenessUnresolved {
                index: generation.to_string(),
                limit,
            }
        }
        CommitError::RevisionMismatch { expected, current } => {
            ExecutionError::RevisionMismatch { expected, current }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod fusion_kernel_tests;
