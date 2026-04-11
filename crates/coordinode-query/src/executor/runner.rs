//! Physical query executor: runs logical plan operators against storage.
//!
//! Each operator produces a `Vec<Row>` from its input.
//! Future optimization: streaming iterator model.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use rayon::prelude::*;

use coordinode_core::graph::edge::{
    decode_adj_key, encode_adj_key_forward, encode_adj_key_reverse, encode_edgeprop_key,
    AdjDirection, PostingList,
};
use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::NodeIdAllocator;
use coordinode_core::graph::node::{encode_node_key, NodeId, NodeRecord};
use coordinode_core::graph::types::{Value, VectorConsistencyMode, VectorMvccStats};
use coordinode_core::schema::definition::{encode_label_schema_key, LabelSchema, SchemaMode};
use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::merge::{encode_add_batch, encode_remove};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::StorageSnapshot;
use coordinode_storage::Guard;

use super::eval::{eval_expr, is_truthy};
use super::row::Row;
use crate::cypher::ast::{Direction, Expr, LengthBound, Pattern, PatternElement};
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

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("write conflict: {0}")]
    Conflict(String),
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
pub struct ExecutionContext<'a> {
    pub engine: &'a StorageEngine,
    pub interner: &'a mut FieldInterner,
    /// Node ID allocator for CREATE operations.
    pub id_allocator: &'a NodeIdAllocator,
    /// Default shard ID for single-node deployment.
    pub shard_id: u16,
    /// Adaptive query plan configuration.
    pub adaptive: AdaptiveConfig,
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
    /// Optional VectorLoader for disk-backed f32 reranking (G009).
    /// When HNSW indexes have `offload_vectors` enabled, this loader provides
    /// f32 vectors from storage for exact reranking of SQ8 candidates.
    pub vector_loader: Option<&'a dyn coordinode_vector::VectorLoader>,
    /// MVCC timestamp oracle. When set, enables MVCC-versioned reads/writes.
    pub mvcc_oracle: Option<&'a TimestampOracle>,
    /// MVCC read timestamp (start_ts). Allocated from oracle at statement start.
    /// All reads see a consistent snapshot at this timestamp.
    pub mvcc_read_ts: Timestamp,
    /// In-memory write buffer for MVCC: (partition, user_key) → value.
    /// Writes are buffered here during execution and flushed atomically
    /// at commit time with a commit_ts from the oracle.
    /// Reads check this buffer first (read-your-own-writes).
    #[allow(clippy::type_complexity)]
    pub mvcc_write_buffer: HashMap<(Partition, Vec<u8>), Option<Vec<u8>>>,
    /// Procedure context for CALL statements. When set, enables
    /// `db.advisor.*` procedures in the executor.
    pub procedure_ctx: Option<crate::advisor::procedures::ProcedureContext>,

    /// OCC read-set: keys read from storage during this transaction.
    ///
    /// At commit time, each key in the read-set is checked for writes
    /// committed after `mvcc_read_ts`. If any such write exists, the
    /// transaction has a read-write conflict and must be aborted.
    ///
    /// Keys written to `mvcc_write_buffer` (read-your-own-writes) are
    /// NOT added to the read-set — they are part of our own transaction.
    ///
    /// `adj:` partition keys are excluded from conflict checking when
    /// merge operators are in use — merge writes are commutative
    /// and conflict-free by construction.
    #[allow(clippy::type_complexity)]
    pub mvcc_read_set: HashSet<(Partition, Vec<u8>)>,
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
    /// Edge creates accumulate UIDs here instead of read-modify-write.
    /// At flush, each entry becomes a `Mutation::Merge` with `encode_add_batch`.
    /// Also checked for read-your-own-writes during edge traversal.
    pub merge_adj_adds: HashMap<Vec<u8>, Vec<u64>>,
    /// Pending merge-remove UIDs per adj key.
    ///
    /// Edge deletes (specific edge, not DETACH) accumulate here.
    /// At flush, each UID becomes a `Mutation::Merge` with `encode_remove`.
    pub merge_adj_removes: HashMap<Vec<u8>, Vec<u64>>,
    /// MVCC snapshot for point-in-time reads (ADR-016: native seqno MVCC).
    ///
    /// When MVCC is enabled, this snapshot is created at `mvcc_read_ts` via
    /// `engine.snapshot_at(seqno)`. All reads (mvcc_get, mvcc_prefix_scan)
    /// go through this snapshot for O(1) lookups instead of key-suffix prefix
    /// scanning. Also used for adj: partition reads (replaces adj_snapshot).
    ///
    /// When None (legacy mode), reads go directly through engine.get().
    pub mvcc_snapshot: Option<StorageSnapshot>,
    /// storage snapshot for adj: partition time-travel reads.
    ///
    /// When set (typically from AS OF TIMESTAMP or statement-level snapshot),
    /// `adj_get()` reads through this snapshot instead of raw engine.get().
    /// This ensures merge operands written after the snapshot are invisible,
    /// providing consistent edge visibility for time-travel queries.
    ///
    /// Taken at statement start from `engine.snapshot()`.
    pub adj_snapshot: Option<StorageSnapshot>,
    /// Pending DocDelta merge operands for node: partition.
    ///
    /// Path-targeted SET/REMOVE on DOCUMENT properties via merge operands.
    /// Each entry: (node_key, encoded DocDelta with PathTarget::PropField).
    /// At flush, each becomes `Mutation::Merge` on `Partition::Node`.
    pub merge_node_deltas: Vec<(Vec<u8>, Vec<u8>)>,
    /// Outer-scope row for correlated OPTIONAL MATCH execution.
    ///
    /// When set, the Filter operator merges these variables into each row
    /// before predicate evaluation. This allows correlated patterns like
    /// `OPTIONAL MATCH (b)-[:R]->(c) WHERE c.x = a.y` where `a` comes
    /// from the outer MATCH scope.
    pub correlated_row: Option<Row>,
}

/// Convert storage `Partition` to serializable `PartitionId`.
fn partition_to_id(p: Partition) -> PartitionId {
    match p {
        Partition::Node => PartitionId::Node,
        Partition::Adj => PartitionId::Adj,
        Partition::EdgeProp => PartitionId::EdgeProp,
        Partition::Blob => PartitionId::Blob,
        Partition::BlobRef => PartitionId::BlobRef,
        Partition::Schema => PartitionId::Schema,
        Partition::Idx => PartitionId::Idx,
        Partition::Raft => unreachable!("Raft partition is not exposed to the query layer"),
        Partition::Counter => PartitionId::Counter,
    }
}

impl<'a> ExecutionContext<'a> {
    /// MVCC-aware read: write buffer → snapshot O(1) → legacy fallback.
    ///
    /// 1. Check write buffer (read-your-own-writes within this statement)
    /// 2. If MVCC snapshot set: snapshot.get() — O(1) native seqno MVCC (ADR-016)
    /// 3. If no snapshot: direct engine.get() — legacy mode
    pub fn mvcc_get(
        &mut self,
        part: Partition,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, ExecutionError> {
        // RYOW for pending node merge deltas: materialize into write buffer
        // so subsequent reads see the correct state.
        if part == Partition::Node && self.merge_node_deltas.iter().any(|(k, _)| k == key) {
            self.materialize_node_deltas(key)?;
        }

        // Check write buffer first (read-your-own-writes — not tracked in read-set)
        let buf_key = (part, key.to_vec());
        if let Some(buffered) = self.mvcc_write_buffer.get(&buf_key) {
            return Ok(buffered.clone());
        }

        // Track this key in the OCC read-set (for conflict detection at commit).
        // Only in MVCC mode — legacy mode doesn't have conflict detection.
        if self.mvcc_oracle.is_some() {
            self.mvcc_read_set.insert((part, key.to_vec()));
        }

        // Native seqno MVCC read via snapshot_at (ADR-016) or legacy read
        if let Some(ref snap) = self.mvcc_snapshot {
            Ok(self
                .engine
                .snapshot_get(snap, part, key)?
                .map(|b| b.to_vec()))
        } else {
            Ok(self.engine.get(part, key)?.map(|b| b.to_vec()))
        }
    }

    /// MVCC-aware write: buffers in write_buffer for atomic flush at commit.
    ///
    /// When MVCC is disabled (legacy mode), writes directly to engine.
    pub fn mvcc_put(
        &mut self,
        part: Partition,
        key: &[u8],
        value: &[u8],
    ) -> Result<(), ExecutionError> {
        if self.mvcc_oracle.is_some() {
            self.mvcc_write_buffer
                .insert((part, key.to_vec()), Some(value.to_vec()));
            Ok(())
        } else {
            Ok(self.engine.put(part, key, value)?)
        }
    }

    /// MVCC-aware delete: buffers tombstone in write_buffer for atomic flush.
    ///
    /// When MVCC is disabled (legacy mode), deletes directly from engine.
    pub fn mvcc_delete(&mut self, part: Partition, key: &[u8]) -> Result<(), ExecutionError> {
        if self.mvcc_oracle.is_some() {
            self.mvcc_write_buffer.insert((part, key.to_vec()), None);
            Ok(())
        } else {
            Ok(self.engine.delete(part, key)?)
        }
    }

    /// Flush MVCC write buffer to storage with a commit timestamp.
    ///
    /// Called at the end of statement execution. Assigns commit_ts from
    /// the oracle, performs OCC conflict detection against the read-set,
    /// and writes all buffered mutations with versioned keys.
    ///
    /// ## OCC Conflict Detection
    ///
    /// Before flushing writes, checks every key in `mvcc_read_set` for
    /// versions committed after `mvcc_read_ts` (our snapshot). If any
    /// such version exists, another transaction modified a key we read
    /// → read-write conflict → `ErrConflict`.
    ///
    /// `adj:` partition keys are excluded from conflict checking because
    /// posting list operations are commutative (use merge operators).
    ///
    /// Returns the commit_ts used, or `ErrConflict` if a conflict is detected.
    pub fn mvcc_flush(&mut self) -> Result<Option<Timestamp>, ExecutionError> {
        // Flush adj merge buffers even in legacy (no MVCC) mode.
        // Legacy puts write directly to engine, but merge adds are buffered.
        if self.mvcc_oracle.is_none() {
            for (key, uids) in self.merge_adj_adds.drain() {
                self.engine
                    .merge(Partition::Adj, &key, &encode_add_batch(&uids))
                    .map_err(ExecutionError::Storage)?;
            }
            for (key, uids) in self.merge_adj_removes.drain() {
                for uid in uids {
                    self.engine
                        .merge(Partition::Adj, &key, &encode_remove(uid))
                        .map_err(ExecutionError::Storage)?;
                }
            }
            for (key, operand) in self.merge_node_deltas.drain(..) {
                self.engine
                    .merge(Partition::Node, &key, &operand)
                    .map_err(ExecutionError::Storage)?;
            }
            return Ok(None); // Legacy mode — writes already applied
        }
        // SAFETY: checked is_none() above and returned early.
        let oracle = match self.mvcc_oracle {
            Some(o) => o,
            None => return Ok(None),
        };

        let has_merge_ops = !self.merge_adj_adds.is_empty()
            || !self.merge_adj_removes.is_empty()
            || !self.merge_node_deltas.is_empty();
        if self.mvcc_write_buffer.is_empty() && !has_merge_ops {
            return Ok(Some(self.mvcc_read_ts)); // Read-only — no commit needed
        }

        let commit_ts = oracle.next();

        // OCC conflict detection (ADR-016: native seqno-based).
        //
        // For each key in our read-set, check if the latest version's seqno
        // is greater than our start_ts. If so, another transaction committed
        // a write after we started — abort with ErrConflict.
        //
        // Uses lsm-tree's get_internal_entry to inspect seqno metadata
        // without full value deserialization. Detects all writes including
        // ABA (write + revert to same value).
        for (part, key) in &self.mvcc_read_set {
            // Exclude adj: partition — posting list writes are commutative
            // and conflict-free (merge operators).
            if *part == Partition::Adj {
                continue;
            }

            if self
                .engine
                .has_write_after(*part, key, self.mvcc_read_ts.as_raw())?
            {
                return Err(ExecutionError::Conflict(format!(
                    "OCC conflict: key in {part:?} partition was modified by another \
                     transaction after start_ts={}. Retry the transaction.",
                    self.mvcc_read_ts.as_raw()
                )));
            }
        }

        // Resolve effective write concern (j:true upgrades W0 → W1).
        use coordinode_core::txn::write_concern::WriteConcernLevel;
        let effective_level = self.write_concern.effective_level();

        // Write concern W0 (fire-and-forget): apply directly to local storage
        // without going through the proposal pipeline. No durability guarantee.
        // Data visible locally but NOT replicated. Lost on crash.
        //
        // ADR-016: writes use plain engine.put()/delete() — no versioned key
        // encoding. LSM seqno from OracleSeqnoGenerator provides native MVCC.
        if effective_level == WriteConcernLevel::W0 {
            for ((part, key), value) in self.mvcc_write_buffer.drain() {
                match value {
                    Some(v) => self.engine.put(part, &key, &v)?,
                    None => self.engine.delete(part, &key)?,
                }
            }
            // Apply adj merge operands directly to StorageEngine (raw keys).
            for (key, uids) in self.merge_adj_adds.drain() {
                self.engine
                    .merge(Partition::Adj, &key, &encode_add_batch(&uids))
                    .map_err(ExecutionError::Storage)?;
            }
            for (key, uids) in self.merge_adj_removes.drain() {
                for uid in uids {
                    self.engine
                        .merge(Partition::Adj, &key, &encode_remove(uid))
                        .map_err(ExecutionError::Storage)?;
                }
            }
            for (key, operand) in self.merge_node_deltas.drain(..) {
                self.engine
                    .merge(Partition::Node, &key, &operand)
                    .map_err(ExecutionError::Storage)?;
            }
            return Ok(Some(commit_ts));
        }

        // Write concern Memory/Cache (volatile with drain):
        // 1. Apply locally for immediate read visibility (same as W0)
        // 2. Buffer mutations in DrainBuffer for background Raft replication
        // 3. Return immediately — drain thread handles durability
        //
        // Crash before drain = data lost (explicit contract).
        // Drained entries preserve original commit_ts for CDC fidelity.
        if effective_level.is_volatile() {
            // Step 1: Apply locally for read visibility.
            for ((part, key), value) in &self.mvcc_write_buffer {
                match value {
                    Some(v) => self.engine.put(*part, key, v)?,
                    None => self.engine.delete(*part, key)?,
                }
            }
            for (key, uids) in &self.merge_adj_adds {
                self.engine
                    .merge(Partition::Adj, key, &encode_add_batch(uids))
                    .map_err(ExecutionError::Storage)?;
            }
            for (key, uids) in &self.merge_adj_removes {
                for uid in uids {
                    self.engine
                        .merge(Partition::Adj, key, &encode_remove(*uid))
                        .map_err(ExecutionError::Storage)?;
                }
            }
            for (key, operand) in &self.merge_node_deltas {
                self.engine
                    .merge(Partition::Node, key, operand)
                    .map_err(ExecutionError::Storage)?;
            }

            // Step 2: Buffer for drain (if drain buffer is available).
            if let Some(drain_buf) = self.drain_buffer {
                let mut mutations: Vec<Mutation> = self
                    .mvcc_write_buffer
                    .drain()
                    .map(|((part, key), value)| match value {
                        Some(v) => Mutation::Put {
                            partition: partition_to_id(part),
                            key,
                            value: v,
                        },
                        None => Mutation::Delete {
                            partition: partition_to_id(part),
                            key,
                        },
                    })
                    .collect();

                for (key, uids) in self.merge_adj_adds.drain() {
                    mutations.push(Mutation::Merge {
                        partition: PartitionId::Adj,
                        key,
                        operand: encode_add_batch(&uids),
                    });
                }
                for (key, uids) in self.merge_adj_removes.drain() {
                    for uid in uids {
                        mutations.push(Mutation::Merge {
                            partition: PartitionId::Adj,
                            key: key.clone(),
                            operand: encode_remove(uid),
                        });
                    }
                }
                for (key, operand) in self.merge_node_deltas.drain(..) {
                    mutations.push(Mutation::Merge {
                        partition: PartitionId::Node,
                        key,
                        operand,
                    });
                }

                let entry = coordinode_core::txn::drain::DrainEntry::new(
                    mutations,
                    commit_ts,
                    self.mvcc_read_ts,
                );

                // w:cache: persist to NVMe before ACK for process-crash recovery.
                // w:memory skips this — data loss on crash is the explicit contract.
                if effective_level == WriteConcernLevel::Cache {
                    if let Some(wb) = self.nvme_write_buffer {
                        wb.append(&entry).map_err(|e| {
                            ExecutionError::Serialization(format!("w:cache NVMe write failed: {e}"))
                        })?;
                    }
                }

                drain_buf.append(entry).map_err(|e| {
                    ExecutionError::Serialization(format!("volatile write backpressure: {e}"))
                })?;
            } else {
                // No drain buffer — clear write buffers (local writes already applied).
                self.mvcc_write_buffer.clear();
                self.merge_adj_adds.clear();
                self.merge_adj_removes.clear();
                self.merge_node_deltas.clear();
            }

            return Ok(Some(commit_ts));
        }

        // W1 / Majority: apply through proposal pipeline (or direct write).
        //
        // When a pipeline is configured, mutations are packaged into a
        // RaftProposal and sent through the pipeline for durable application.
        // In single-node mode (W1 and Majority are equivalent), the pipeline
        // applies directly to CoordiNode storage. In cluster mode, Majority replicates
        // via Raft first while W1 returns after leader WAL fsync.
        //
        // When no pipeline is configured (legacy/test mode), mutations are
        // written directly to MvccEngine.
        if let (Some(pipeline), Some(id_gen)) = (self.proposal_pipeline, self.proposal_id_gen) {
            let mut mutations: Vec<Mutation> = self
                .mvcc_write_buffer
                .drain()
                .map(|((part, key), value)| match value {
                    Some(v) => Mutation::Put {
                        partition: partition_to_id(part),
                        key,
                        value: v,
                    },
                    None => Mutation::Delete {
                        partition: partition_to_id(part),
                        key,
                    },
                })
                .collect();

            // Adj merge operands: bypass MVCC, raw keys.
            for (key, uids) in self.merge_adj_adds.drain() {
                mutations.push(Mutation::Merge {
                    partition: PartitionId::Adj,
                    key,
                    operand: encode_add_batch(&uids),
                });
            }
            for (key, uids) in self.merge_adj_removes.drain() {
                for uid in uids {
                    mutations.push(Mutation::Merge {
                        partition: PartitionId::Adj,
                        key: key.clone(),
                        operand: encode_remove(uid),
                    });
                }
            }
            for (key, operand) in self.merge_node_deltas.drain(..) {
                mutations.push(Mutation::Merge {
                    partition: PartitionId::Node,
                    key,
                    operand,
                });
            }

            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations,
                commit_ts,
                start_ts: self.mvcc_read_ts,
                bypass_rate_limiter: false,
            };

            // Apply write concern timeout if configured (wtimeout > 0).
            //
            // propose_with_timeout uses true async timeout in cluster mode
            // (RaftProposalPipeline wraps propose_async with tokio::time::timeout).
            // In embedded/single-node mode, the default impl delegates to
            // propose_and_wait (proposals complete in µs, timeout irrelevant).
            //
            // Per MongoDB spec: "On timeout, data is NOT rolled back."
            // The proposal may still commit after timeout fires.
            let timeout_ms = self.write_concern.timeout_ms;
            if timeout_ms > 0 {
                let timeout = std::time::Duration::from_millis(u64::from(timeout_ms));
                pipeline
                    .propose_with_timeout(&proposal, timeout)
                    .map_err(|e| {
                        ExecutionError::Serialization(format!("proposal pipeline error: {e}"))
                    })?;
            } else {
                pipeline.propose_and_wait(&proposal).map_err(|e| {
                    ExecutionError::Serialization(format!("proposal pipeline error: {e}"))
                })?;
            }
        } else {
            // Legacy direct-write path (no pipeline configured).
            // ADR-016: plain engine.put()/delete() — oracle auto-stamps seqno.
            for ((part, key), value) in self.mvcc_write_buffer.drain() {
                match value {
                    Some(v) => self.engine.put(part, &key, &v)?,
                    None => self.engine.delete(part, &key)?,
                }
            }
            // Apply adj merge operands directly to StorageEngine (raw keys).
            for (key, uids) in self.merge_adj_adds.drain() {
                self.engine
                    .merge(Partition::Adj, &key, &encode_add_batch(&uids))
                    .map_err(ExecutionError::Storage)?;
            }
            for (key, uids) in self.merge_adj_removes.drain() {
                for uid in uids {
                    self.engine
                        .merge(Partition::Adj, &key, &encode_remove(uid))
                        .map_err(ExecutionError::Storage)?;
                }
            }
            for (key, operand) in self.merge_node_deltas.drain(..) {
                self.engine
                    .merge(Partition::Node, &key, &operand)
                    .map_err(ExecutionError::Storage)?;
            }
        }

        // Journal gate (j:true): force WAL fsync after commit.
        // With FlushPolicy::SyncPerBatch this is already done by WriteBatch,
        // but with Periodic/Manual policies, j:true forces an explicit persist.
        if self.write_concern.journal {
            self.engine
                .persist()
                .map_err(|e| ExecutionError::Serialization(format!("journal fsync failed: {e}")))?;
        }

        Ok(Some(commit_ts))
    }

    /// Materialize pending node merge deltas for a key into the write buffer.
    ///
    /// Called lazily from `mvcc_get()` to ensure RYOW correctness: if a
    /// transaction wrote merge operands (SET n.config.x = y) and later reads
    /// the same node, the pending deltas must be visible.
    fn materialize_node_deltas(&mut self, node_key: &[u8]) -> Result<(), ExecutionError> {
        // Collect matching deltas.
        let matching: Vec<Vec<u8>> = self
            .merge_node_deltas
            .iter()
            .filter(|(k, _)| k == node_key)
            .map(|(_, op)| op.clone())
            .collect();

        if matching.is_empty() {
            return Ok(());
        }

        // Remove materialized deltas from the buffer.
        self.merge_node_deltas.retain(|(k, _)| k != node_key);

        // Read current node value (from write buffer or storage).
        let buf_key = (Partition::Node, node_key.to_vec());
        let current = if let Some(buffered) = self.mvcc_write_buffer.get(&buf_key) {
            buffered.clone()
        } else if let Some(ref snap) = self.mvcc_snapshot {
            self.engine
                .snapshot_get(snap, Partition::Node, node_key)?
                .map(|b| b.to_vec())
        } else {
            self.engine
                .get(Partition::Node, node_key)?
                .map(|b| b.to_vec())
        };

        let mut record = match current {
            Some(ref bytes) => NodeRecord::from_msgpack(bytes)
                .map_err(|e| ExecutionError::Serialization(format!("RYOW node decode: {e}")))?,
            None => NodeRecord::new(""),
        };

        // Apply each pending delta.
        for operand in &matching {
            if let Ok(delta) = coordinode_core::graph::doc_delta::DocDelta::decode(&operand[1..]) {
                match delta.target() {
                    coordinode_core::graph::doc_delta::PathTarget::PropField(field_id) => {
                        let mut doc = match record.props.get(field_id) {
                            Some(v) => v.to_rmpv(),
                            None => rmpv::Value::Map(Vec::new()),
                        };
                        delta.apply(&mut doc);
                        record.set(
                            *field_id,
                            coordinode_core::graph::types::Value::Document(doc),
                        );
                    }
                    coordinode_core::graph::doc_delta::PathTarget::Extra => {
                        // Extra-targeted deltas handled by merge function, not here.
                    }
                }
            }
        }

        let new_bytes = record
            .to_msgpack()
            .map_err(|e| ExecutionError::Serialization(format!("RYOW node encode: {e}")))?;
        self.mvcc_write_buffer.insert(buf_key, Some(new_bytes));
        Ok(())
    }

    /// MVCC-aware prefix scan: returns deduplicated (user_key, value) pairs
    /// visible at the current snapshot timestamp.
    ///
    /// In legacy mode, returns raw prefix scan results as (key, value) pairs.
    pub fn mvcc_prefix_scan(
        &mut self,
        part: Partition,
        prefix: &[u8],
    ) -> Result<Vec<KvPair>, ExecutionError> {
        // RYOW for pending node merge deltas: materialize all matching keys
        // into write buffer so the scan sees up-to-date values.
        if part == Partition::Node && !self.merge_node_deltas.is_empty() {
            let matching_keys: Vec<Vec<u8>> = self
                .merge_node_deltas
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

        // Check write buffer for matching entries (read-your-own-writes)
        let mut buffer_matches: Vec<KvPair> = Vec::new();
        for ((p, key), value) in &self.mvcc_write_buffer {
            if *p == part && key.starts_with(prefix) {
                if let Some(v) = value {
                    buffer_matches.push((key.clone(), v.clone()));
                }
            }
        }

        if let Some(ref snap) = self.mvcc_snapshot {
            // Native seqno MVCC scan via snapshot (ADR-016)
            let scan_results = self.engine.snapshot_prefix_scan(snap, part, prefix)?;
            let mut results: Vec<KvPair> = scan_results
                .into_iter()
                .map(|(k, v)| (k, v.to_vec()))
                .collect();

            // Track all scanned keys in OCC read-set.
            // Keys from the write buffer are our own writes — NOT tracked.
            let buffer_keys: HashSet<Vec<u8>> =
                buffer_matches.iter().map(|(k, _)| k.clone()).collect();
            for (k, _) in &results {
                if !buffer_keys.contains(k) {
                    self.mvcc_read_set.insert((part, k.clone()));
                }
            }

            // Merge buffer matches (buffer takes priority)
            results.retain(|(k, _)| !buffer_keys.contains(k));
            results.extend(buffer_matches);
            Ok(results)
        } else {
            let iter = self.engine.prefix_scan(part, prefix)?;
            let mut results: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            for guard in iter {
                let (k, v) = guard
                    .into_inner()
                    .map_err(|e| ExecutionError::Storage(e.into()))?;
                results.push((k.to_vec(), v.to_vec()));
            }
            // Add buffer matches
            results.extend(buffer_matches);
            Ok(results)
        }
    }

    // ── Adj partition: raw merge operators (no MVCC key versioning) ──

    /// Record an edge add via merge operator (no read required).
    ///
    /// Buffers the UID for batch encoding at flush time. Also handles
    /// read-your-own-writes: subsequent traversals in this transaction
    /// will see the added edge.
    pub fn adj_merge_add(&mut self, adj_key: &[u8], uid: u64) {
        self.merge_adj_adds
            .entry(adj_key.to_vec())
            .or_default()
            .push(uid);
    }

    /// Record an edge remove via merge operator (no read required).
    pub fn adj_merge_remove(&mut self, adj_key: &[u8], uid: u64) {
        self.merge_adj_removes
            .entry(adj_key.to_vec())
            .or_default()
            .push(uid);
    }

    /// Read an adjacency posting list (raw key, no MVCC timestamp).
    ///
    /// Reads the latest merged value from StorageEngine, then applies
    /// pending adds/removes from this transaction's merge buffer
    /// (read-your-own-writes).
    pub fn adj_get(&self, adj_key: &[u8]) -> Result<Option<PostingList>, ExecutionError> {
        // Read base posting list from storage (raw key, merge-resolved).
        // When adj_snapshot is set (AS OF TIMESTAMP), read through the snapshot
        // so merge operands written after the snapshot are invisible.
        let raw = if let Some(snap) = &self.adj_snapshot {
            self.engine.snapshot_get(snap, Partition::Adj, adj_key)?
        } else {
            self.engine.get(Partition::Adj, adj_key)?
        };
        let mut plist = match raw {
            Some(bytes) => PostingList::from_bytes(&bytes)
                .map_err(|e| ExecutionError::Serialization(format!("posting list: {e}")))?,
            None => PostingList::new(),
        };

        // Apply pending adds from this transaction (read-your-own-writes).
        if let Some(adds) = self.merge_adj_adds.get(adj_key) {
            for &uid in adds {
                plist.insert(uid);
            }
        }
        // Apply pending removes from this transaction.
        if let Some(removes) = self.merge_adj_removes.get(adj_key) {
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

    /// Raw prefix scan on adj: partition (no MVCC timestamp filtering).
    ///
    /// Returns (raw_key, raw_value) pairs.
    pub fn adj_prefix_scan(&self, prefix: &[u8]) -> Result<Vec<KvPair>, ExecutionError> {
        // When adj_snapshot is set (AS OF TIMESTAMP or statement-level snapshot),
        // read through the snapshot so merge operands written after it are invisible.
        if let Some(snap) = &self.adj_snapshot {
            let entries = self
                .engine
                .snapshot_prefix_scan(snap, Partition::Adj, prefix)?;
            Ok(entries.into_iter().map(|(k, v)| (k, v.to_vec())).collect())
        } else {
            let iter = self.engine.prefix_scan(Partition::Adj, prefix)?;
            let mut results = Vec::new();
            for guard in iter {
                let (k, v) = guard
                    .into_inner()
                    .map_err(|e| ExecutionError::Storage(e.into()))?;
                results.push((k.to_vec(), v.to_vec()));
            }
            Ok(results)
        }
    }

    /// Return all edge type names registered in the Schema partition.
    ///
    /// Used by DETACH DELETE to build targeted adj key lookups instead of
    /// scanning every adj: key in the database.
    pub(crate) fn list_edge_types(&self) -> Result<Vec<String>, ExecutionError> {
        const PREFIX: &[u8] = b"schema:edge_type:";
        let iter = self.engine.prefix_scan(Partition::Schema, PREFIX)?;
        let mut types = Vec::new();
        for guard in iter {
            let (k, _) = guard
                .into_inner()
                .map_err(|e| ExecutionError::Storage(e.into()))?;
            // Also include any buffered (uncommitted) edge type registrations from
            // this transaction, so DETACH DELETE sees same-tx edge creations.
            if let Ok(name) = std::str::from_utf8(&k[PREFIX.len()..]) {
                types.push(name.to_string());
            }
        }
        // Include edge types registered in this transaction's write buffer
        // (not yet flushed to the engine — covers same-tx CREATE + DETACH DELETE).
        for (part, key) in self.mvcc_write_buffer.keys() {
            if *part == Partition::Schema && key.starts_with(PREFIX) {
                if let Ok(name) = std::str::from_utf8(&key[PREFIX.len()..]) {
                    if !types.contains(&name.to_string()) {
                        types.push(name.to_string());
                    }
                }
            }
        }
        Ok(types)
    }
}

/// Execute a logical plan against storage, returning result rows.
pub fn execute(
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
    if ctx.adj_snapshot.is_none() {
        if let Some(ref snap) = ctx.mvcc_snapshot {
            ctx.adj_snapshot = Some(*snap);
        } else {
            ctx.adj_snapshot = Some(ctx.engine.snapshot());
        }
    }

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
                ctx.adj_snapshot = Some(snap);
            }
        }
    }

    let result = execute_op(&plan.root, ctx)?;

    // Flush MVCC write buffer: assign commit_ts and persist all buffered writes.
    // In legacy mode (mvcc_oracle: None), this is a no-op.
    ctx.mvcc_flush()?;

    Ok(result)
}

/// Execute a single logical operator recursively.
fn execute_op(op: &LogicalOp, ctx: &mut ExecutionContext<'_>) -> Result<Vec<Row>, ExecutionError> {
    match op {
        LogicalOp::NodeScan {
            variable,
            labels,
            property_filters,
        } => execute_node_scan(variable, labels, property_filters, ctx),

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
            };
            execute_traverse(&input_rows, &params, ctx)
        }

        LogicalOp::Filter { input, predicate } => {
            let rows = execute_op(input, ctx)?;
            let corr = ctx.correlated_row.clone();
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

        LogicalOp::Project {
            input,
            items,
            distinct,
        } => {
            let rows = execute_op(input, ctx)?;
            let mut result: Vec<Row> = rows
                .into_iter()
                .map(|row| {
                    let mut out = Row::new();
                    for item in items {
                        if item.expr == Expr::Star {
                            // Star: copy all columns
                            out.extend(row.clone());
                        } else {
                            let val = eval_expr(&item.expr, &row);
                            let key = item
                                .alias
                                .clone()
                                .unwrap_or_else(|| expr_display_name(&item.expr));
                            out.insert(key, val);
                        }
                    }
                    out
                })
                .collect();

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
            execute_aggregate(&rows, group_by, aggregates)
        }

        LogicalOp::Sort { input, items } => {
            let mut rows = execute_op(input, ctx)?;
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
            if is_relationship_merge(right) {
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

        LogicalOp::VectorFilter {
            input,
            vector_expr,
            query_vector,
            function,
            less_than,
            threshold,
            decay_field,
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
                            let node_key = coordinode_core::graph::node::encode_node_key(
                                ctx.shard_id,
                                coordinode_core::graph::node::NodeId::from_raw(id as u64),
                            );
                            match ctx.engine.snapshot_get(snap, Partition::Node, &node_key) {
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
        } => {
            let rows = execute_op(input, ctx)?;

            // Try HNSW-accelerated top-K path.
            let result = if let Some(hnsw_result) = try_hnsw_vector_top_k(
                &rows,
                vector_expr,
                query_vector,
                function,
                *k,
                distance_alias.as_deref(),
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

        LogicalOp::Empty => Ok(vec![Row::new()]),

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

        LogicalOp::Update { input, items } => {
            let input_rows = execute_op(input, ctx)?;
            execute_update(&input_rows, items, ctx)
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
        } => execute_merge(pattern, on_match, on_create, ctx),

        LogicalOp::Upsert {
            pattern,
            on_match,
            on_create_patterns,
        } => execute_upsert(pattern, on_match, on_create_patterns, ctx),
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

    // Scan all nodes in the shard using prefix scan.
    let prefix = encode_node_key(ctx.shard_id, NodeId::from_raw(0));
    // Use just the shard prefix (everything before the node ID)
    let prefix_bytes = &prefix[..8];

    let scan_results = ctx.mvcc_prefix_scan(Partition::Node, prefix_bytes)?;

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

        // Apply inline property filters from pattern
        let mut matches = true;
        for (prop_name, filter_expr) in property_filters {
            let actual = row
                .get(&format!("{variable}.{prop_name}"))
                .cloned()
                .unwrap_or(Value::Null);
            let expected = eval_expr(filter_expr, &row);
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
}

/// Traverse edges from source nodes to target nodes.
///
/// Dispatches to single-hop or variable-length BFS based on `params.length`.
fn execute_traverse(
    input_rows: &[Row],
    params: &TraverseParams<'_>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
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

    for (et_idx, edge_type) in edge_types.iter().enumerate() {
        let adj_key = match direction {
            Direction::Outgoing | Direction::Both => encode_adj_key_forward(edge_type, src_id),
            Direction::Incoming => encode_adj_key_reverse(edge_type, src_id),
        };

        if let Some(posting_list) = ctx.adj_get(&adj_key)? {
            let fan_out = posting_list.len();
            if ctx.adaptive.enabled && fan_out > ctx.adaptive.parallel_threshold {
                tracing::info!(
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
            let rev_key = encode_adj_key_reverse(edge_type, src_id);
            if let Some(posting_list) = ctx.adj_get(&rev_key)? {
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
}

/// Fetch a target node and build an output row if it passes label/property filters.
///
/// Returns `None` if the node doesn't exist or fails filters.
fn build_target_row(
    trp: &TargetRowParams<'_>,
    params: &TraverseParams<'_>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Option<Row>, ExecutionError> {
    let target_uid = trp.target_uid;
    let target_variable = params.target_variable;
    let target_labels = params.target_labels;
    let target_filters = params.target_filters;
    let edge_variable = params.edge_variable;
    let edge_type = trp.edge_type;

    let target_id = NodeId::from_raw(target_uid);
    let target_key = encode_node_key(ctx.shard_id, target_id);
    let target_record = match ctx.mvcc_get(Partition::Node, &target_key)? {
        Some(bytes) => NodeRecord::from_msgpack(&bytes).map_err(|e| {
            ExecutionError::Serialization(format!("target node deserialization error: {e}"))
        })?,
        None => return Ok(None),
    };

    // Label filter
    if !target_labels.is_empty() && !target_labels.iter().any(|l| target_record.has_label(l)) {
        return Ok(None);
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

    // Inject COMPUTED property values from schema (R082).
    inject_computed_properties(&mut out_row, target_variable, &target_label, ctx);

    if let Some(ev) = edge_variable {
        if let Some(et) = edge_type {
            out_row.insert(format!("{ev}.__type__"), Value::String(et.to_string()));
            // G070: also store the relationship variable itself (not only `r.__type__`)
            // so that aggregate expressions like `count(r)` evaluate to a non-null value.
            // Without this, eval_expr("r") returns Null → eval_aggregate_values filters it
            // out → count(r) = 0. Storing the edge type string as the variable value makes
            // count(r) behave identically to count(r.__type__).
            out_row.insert(ev.to_string(), Value::String(et.to_string()));

            // Load edge properties from EdgeProp partition.
            // Source ID comes from the input row (the node we traversed from).
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
                let ep_key = encode_edgeprop_key(et, ep_src, ep_tgt);
                if let Some(ep_bytes) = ctx.mvcc_get(Partition::EdgeProp, &ep_key)? {
                    if let Ok(prop_map) = rmp_serde::from_slice::<Vec<(u32, Value)>>(&ep_bytes) {
                        for (field_id, value) in prop_map {
                            if let Some(field_name) = ctx.interner.resolve(field_id) {
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
            return Ok(None);
        }
    }

    // Apply inline edge property filters (e.g., [r:TYPE {rating: 5}])
    if let Some(ev) = edge_variable {
        for (prop_name, filter_expr) in params.edge_filters {
            let actual = out_row
                .get(&format!("{ev}.{prop_name}"))
                .cloned()
                .unwrap_or(Value::Null);
            let expected = eval_expr(filter_expr, &out_row);
            if actual != expected {
                return Ok(None);
            }
        }
    }

    Ok(Some(out_row))
}

/// Collected OCC read-set keys from parallel processing (G067).
type OccReadKeys = Mutex<Vec<(Partition, Vec<u8>)>>;

/// Read-only context extracted from `ExecutionContext` for parallel processing.
/// All fields are `Send + Sync`, enabling safe rayon parallelism.
struct ParallelCtx<'a> {
    engine: &'a StorageEngine,
    interner: &'a FieldInterner,
    shard_id: u16,
    mvcc_snapshot: Option<StorageSnapshot>,
    chunk_size: usize,
    /// OCC read-set keys collected during parallel processing (G067).
    /// When `Some`, parallel workers push `(Partition, key)` for each storage
    /// read so that the caller can merge them into `ExecutionContext::mvcc_read_set`
    /// after the parallel block. `None` when MVCC oracle is inactive (legacy mode).
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
/// OCC read-set tracking (G067): when `pctx.occ_read_keys` is `Some`, all
/// read keys are collected into the `Mutex<Vec>` for the caller to merge into
/// `ExecutionContext::mvcc_read_set` after the parallel block completes.
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
                let target_key = encode_node_key(pctx.shard_id, target_id);

                // Read node record (direct engine access, thread-safe)
                let bytes = if let Some(snap) = pctx.mvcc_snapshot {
                    pctx.engine
                        .snapshot_get(&snap, Partition::Node, &target_key)
                        .ok()?
                } else {
                    pctx.engine.get(Partition::Node, &target_key).ok()?
                };
                let bytes = bytes?;

                // Track Node key in OCC read-set (G067)
                if let Some(ref keys) = pctx.occ_read_keys {
                    if let Ok(mut guard) = keys.lock() {
                        guard.push((Partition::Node, target_key.to_vec()));
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
                            let ep_key = encode_edgeprop_key(et, ep_src, ep_tgt);
                            let ep_bytes = if let Some(snap) = pctx.mvcc_snapshot {
                                pctx.engine
                                    .snapshot_get(&snap, Partition::EdgeProp, &ep_key)
                                    .ok()
                                    .flatten()
                            } else {
                                pctx.engine.get(Partition::EdgeProp, &ep_key).ok().flatten()
                            };
                            // Track EdgeProp key in OCC read-set (G067)
                            if let Some(ref keys) = pctx.occ_read_keys {
                                if let Ok(mut guard) = keys.lock() {
                                    guard.push((Partition::EdgeProp, ep_key));
                                }
                            }
                            if let Some(ep_bytes) = ep_bytes {
                                if let Ok(prop_map) =
                                    rmp_serde::from_slice::<Vec<(u32, Value)>>(&ep_bytes)
                                {
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

        // Switch to parallel when fan-out exceeds threshold
        if use_parallel && neighbors.len() >= ctx.adaptive.parallel_threshold {
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
                    Some(Mutex::new(Vec::new()))
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
            // Merge OCC read keys from parallel workers (G067)
            if let Some(ref keys_mutex) = pctx.occ_read_keys {
                if let Ok(keys) = keys_mutex.lock() {
                    ctx.mvcc_read_set.extend(keys.iter().cloned());
                }
            }
            results.extend(parallel_rows);
        } else {
            // Sequential path for normal fan-out
            for (target_uid, et_idx) in neighbors {
                let edge_type = params.edge_types.get(et_idx).map(|s| s.as_str());
                let trp = TargetRowParams {
                    input_row: row,
                    target_uid,
                    edge_type,
                };
                if let Some(out_row) = build_target_row(&trp, params, ctx)? {
                    results.push(out_row);
                }
            }
        }
    }

    Ok(results)
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

    let mut results = Vec::new();

    for row in input_rows {
        let source_id = match row.get(params.source) {
            Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
            _ => continue,
        };

        // Edge-level cycle detection: (source_uid, target_uid, edge_type_idx).
        // Prevents traversing the same relationship twice per BFS invocation.
        let mut visited_edges: HashSet<(u64, u64, usize)> = HashSet::new();

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

            for &src_uid in &frontier {
                let src_nid = NodeId::from_raw(src_uid);
                let neighbors = expand_one_hop(src_nid, params.edge_types, params.direction, ctx)?;

                for (tgt_uid, et_idx) in neighbors {
                    if !visited_edges.insert((src_uid, tgt_uid, et_idx)) {
                        continue;
                    }
                    edges_processed += 1;
                    next_frontier.push(tgt_uid);
                    if depth >= min_hops {
                        depth_neighbors.push((src_uid, tgt_uid, et_idx));
                    }
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

            // Build result rows: parallel if enough neighbors, sequential otherwise
            let use_parallel =
                ctx.adaptive.enabled && depth_neighbors.len() >= ctx.adaptive.parallel_threshold;

            if use_parallel {
                // depth_neighbors already has (src, tgt, et_idx) — pass directly
                let pctx = ParallelCtx {
                    engine: ctx.engine,
                    interner: ctx.interner,
                    shard_id: ctx.shard_id,
                    mvcc_snapshot: ctx.mvcc_snapshot,
                    chunk_size: ctx.adaptive.parallel_chunk_size,
                    occ_read_keys: if ctx.mvcc_oracle.is_some() {
                        Some(Mutex::new(Vec::new()))
                    } else {
                        None
                    },
                };
                let parallel_rows = process_targets_parallel(&depth_neighbors, row, params, &pctx);
                // Merge OCC read keys from parallel workers (G067)
                if let Some(ref keys_mutex) = pctx.occ_read_keys {
                    if let Ok(keys) = keys_mutex.lock() {
                        ctx.mvcc_read_set.extend(keys.iter().cloned());
                    }
                }
                results.extend(parallel_rows);
            } else {
                for &(src_uid, tgt_uid, et_idx) in &depth_neighbors {
                    // For depth > 1, update source in the row so edge property lookups
                    // use the correct intermediate node (G066 fix).
                    let mut hop_row = row.clone();
                    hop_row.insert(params.source.to_string(), Value::Int(src_uid as i64));

                    let edge_type = params.edge_types.get(et_idx).map(|s| s.as_str());
                    let trp = TargetRowParams {
                        input_row: &hop_row,
                        target_uid: tgt_uid,
                        edge_type,
                    };
                    if let Some(out_row) = build_target_row(&trp, params, ctx)? {
                        results.push(out_row);
                    }
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
/// 2. Get label from `rows[0]["variable.__label__"]`
/// 3. Check `registry.has_index(label, property)`
/// 4. Evaluate the query vector
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
fn try_hnsw_vector_top_k(
    rows: &[Row],
    vector_expr: &Expr,
    query_vector_expr: &Expr,
    function: &str,
    k: usize,
    distance_alias: Option<&str>,
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

    // Determine the label from the first row's __label__ field.
    let label_key = format!("{variable}.__label__");
    let label = match rows[0].get(&label_key) {
        Some(Value::String(l)) => l.as_str(),
        _ => return Ok(None),
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

    // Overfetch strategy — request enough HNSW candidates to cover:
    // - k * 4: baseline margin for ordering stability
    // - rows.len() * 2: double the filter-reduced subset size, so intersection
    //   likely contains at least k actual rows even when HNSW's globally-nearest
    //   are not in our filtered subset
    // - 100: floor for very small k
    // Capped at 10_000 to prevent excessive memory for massive result sets.
    let overfetch = (k * 4).max(rows.len() * 2).clamp(100, 10_000);

    let search_results = match registry.search_with_loader(
        label,
        property,
        &query_vec,
        overfetch,
        ctx.vector_loader,
    ) {
        Some(r) => r,
        None => return Ok(None),
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
            results.push(row.clone());
        }
    }

    Ok(results)
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
fn execute_text_filter(
    rows: &[Row],
    text_expr: &Expr,
    query_string: &str,
    language: Option<&str>,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // Try text_index_registry first (automatic mode), fallback to legacy text_index.
    // The registry is keyed by (label, property) — extract property from text_expr.
    let limit = rows.len().max(1000);
    let search_results = if let Some(registry) = ctx.text_index_registry {
        // Extract property name from text_expr (PropertyAccess { var, property }).
        let property = match text_expr {
            Expr::PropertyAccess { property, .. } => Some(property.as_str()),
            _ => None,
        };
        // Extract label from the first row's __label__ field.
        let label = if property.is_some() {
            if let Expr::PropertyAccess { expr, .. } = text_expr {
                if let Expr::Variable(var) = expr.as_ref() {
                    rows.first().and_then(|r| {
                        r.get(&format!("{var}.__label__"))
                            .and_then(|v| v.as_str().map(|s| s.to_string()))
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let (Some(label), Some(property)) = (&label, property) {
            if let Some(handle) = registry.get(label, property) {
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
                // No index for this (label, property) — graceful degradation.
                ctx.warnings.push(format!(
                    "text_match() called but no text index for ({label}, {property}). \
                     Create one with CREATE TEXT INDEX."
                ));
                return Ok(rows.to_vec());
            }
        } else {
            // Could not determine label/property — fall through to legacy.
            if let Some(text_index) = ctx.text_index {
                if let Some(lang) = language {
                    text_index
                        .search_with_language(query_string, limit, lang)
                        .map_err(|e| {
                            ExecutionError::Unsupported(format!("text search error: {e}"))
                        })?
                } else {
                    text_index.search(query_string, limit).map_err(|e| {
                        ExecutionError::Unsupported(format!("text search error: {e}"))
                    })?
                }
            } else {
                ctx.warnings
                    .push("text_match() called but no text index available.".to_string());
                return Ok(rows.to_vec());
            }
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
        // No text index available — all rows pass (graceful degradation).
        ctx.warnings.push(
            "text_match() called but no text index available. \
             Create a text index with CREATE TEXT INDEX."
                .to_string(),
        );
        return Ok(rows.to_vec());
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

    // Create storage-backed SSE index on the fly (cheap — just stores engine ref + strings).
    let index = EncryptedIndex::new(ctx.engine, &label, &property);
    let matching_ids = index
        .search(&search_token)
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

        // BFS from src_uid to tgt_uid
        let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
        let mut visited: HashSet<u64> = HashSet::new();

        queue.push_back((src_uid, 0));
        visited.insert(src_uid);

        let mut found_depth: Option<usize> = None;

        while let Some((uid, depth)) = queue.pop_front() {
            if uid == tgt_uid {
                found_depth = Some(depth);
                break;
            }
            if depth >= max_d {
                continue;
            }

            let nid = NodeId::from_raw(uid);
            let neighbors = expand_one_hop(nid, sp.edge_types, sp.direction, ctx)?;

            for (neighbor_uid, _et_idx) in neighbors {
                if visited.insert(neighbor_uid) {
                    queue.push_back((neighbor_uid, depth + 1));
                }
            }
        }

        let mut out = row.clone();
        match found_depth {
            Some(d) => {
                out.insert(sp.path_variable.to_string(), Value::Int(d as i64));
            }
            None => {
                out.insert(sp.path_variable.to_string(), Value::Null);
            }
        }
        results.push(out);
    }

    Ok(results)
}

/// Execute aggregation: group rows and compute aggregate functions.
fn execute_aggregate(
    rows: &[Row],
    group_by: &[Expr],
    aggregates: &[AggregateItem],
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
            let val = compute_aggregate(agg, group_rows);
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
fn compute_aggregate(agg: &AggregateItem, rows: &[&Row]) -> Value {
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

            // Default percentile = 0.5 (median)
            let percentile = 0.5_f64;

            if agg.function == "percentileDisc" {
                // Nearest rank method
                let idx =
                    ((percentile * values.len() as f64).ceil() as usize).min(values.len()) - 1;
                Value::Float(values[idx.min(values.len() - 1)])
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
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    // G072: MERGE (src)-[r:TYPE]->(tgt) — relationship pattern with correlated bindings.
    // When the pattern is a Traverse and ctx.correlated_row has src/tgt bound, use the
    // targeted match+create path instead of the generic execute_op + execute_create_from_pattern.
    // The generic path scans all nodes and fails to create edges from Traverse patterns.
    if let (Some(traverse), Some(correlated)) =
        (as_traverse_op(pattern), ctx.correlated_row.clone())
    {
        let matches = execute_merge_relationship_check(traverse, &correlated, ctx)?;
        if !matches.is_empty() {
            if on_match.is_empty() {
                return Ok(matches);
            }
            return execute_update(&matches, on_match, ctx);
        }
        let created = execute_merge_relationship_create(traverse, &correlated, ctx)?;
        if on_create.is_empty() {
            return Ok(created);
        }
        return execute_update(&created, on_create, ctx);
    }

    // Phase 1: Try to find existing matches
    let matches = execute_op(pattern, ctx)?;

    if !matches.is_empty() {
        // Pattern found — apply ON MATCH SET items
        if on_match.is_empty() {
            return Ok(matches);
        }
        execute_update(&matches, on_match, ctx)
    } else {
        // Pattern not found — create new node from pattern, then apply ON CREATE SET
        let created = execute_create_from_pattern(pattern, ctx)?;
        if on_create.is_empty() {
            return Ok(created);
        }
        execute_update(&created, on_create, ctx)
    }
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
    let (source, edge_types, direction, target_variable, edge_variable) = match traverse {
        LogicalOp::Traverse {
            source,
            edge_types,
            direction,
            target_variable,
            edge_variable,
            ..
        } => (
            source,
            edge_types,
            direction,
            target_variable,
            edge_variable,
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
    for et in effective_types {
        let neighbors = expand_one_hop(source_id, std::slice::from_ref(et), *direction, ctx)?;
        if neighbors.iter().any(|(tgt, _)| *tgt == target_raw) {
            let mut row = correlated.clone();
            if let Some(ev) = edge_variable {
                row.insert(format!("{ev}.__type__"), Value::String(et.clone()));
                row.insert(ev.clone(), Value::String(et.clone()));
            }
            return Ok(vec![row]);
        }
    }

    Ok(vec![])
}

/// Create a relationship edge described by `traverse` between the source and target
/// nodes bound in `correlated`. Called from `execute_merge` when no existing edge
/// was found by `execute_merge_relationship_check`.
fn execute_merge_relationship_create(
    traverse: &LogicalOp,
    correlated: &Row,
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let (source, edge_types, direction, target_variable, edge_variable) = match traverse {
        LogicalOp::Traverse {
            source,
            edge_types,
            direction,
            target_variable,
            edge_variable,
            ..
        } => (
            source,
            edge_types,
            direction,
            target_variable,
            edge_variable,
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
    let fwd_key = encode_adj_key_forward(et, from_id);
    ctx.adj_merge_add(&fwd_key, to_id.as_raw());
    let rev_key = encode_adj_key_reverse(et, to_id);
    ctx.adj_merge_add(&rev_key, from_id.as_raw());
    ctx.write_stats.edges_created += 1;

    // Register edge type in schema (idempotent).
    let et_key = edge_type_schema_key(et);
    if !ctx
        .mvcc_write_buffer
        .contains_key(&(Partition::Schema, et_key.clone()))
    {
        ctx.mvcc_put(Partition::Schema, &et_key, b"")?;
    }

    let mut row = correlated.clone();
    if let Some(ev) = edge_variable {
        row.insert(format!("{ev}.__type__"), Value::String(et.clone()));
        row.insert(ev.clone(), Value::String(et.clone()));
    }

    Ok(vec![row])
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

        // Capture current state of matched nodes for CAS verification.
        // For each variable that maps to a node ID, read the raw bytes now.
        let mut cas_snapshots: Vec<Vec<(String, Vec<u8>)>> = Vec::new();
        for row in &matches {
            let mut snapshot = Vec::new();
            for item in on_match {
                if let crate::cypher::ast::SetItem::Property { variable, .. } = item {
                    if let Some(Value::Int(id)) = row.get(variable) {
                        let node_id = NodeId::from_raw(*id as u64);
                        let key = encode_node_key(ctx.shard_id, node_id);
                        if let Some(bytes) = ctx.mvcc_get(Partition::Node, &key)? {
                            // Dedup: don't snapshot same variable twice
                            if !snapshot.iter().any(|(v, _)| v == variable) {
                                snapshot.push((variable.clone(), bytes.to_vec()));
                            }
                        }
                    }
                }
            }
            cas_snapshots.push(snapshot);
        }

        // CAS check: verify nodes haven't changed since MATCH read.
        // In single-node synchronous mode this is inherently safe, but the
        // check is present for correctness when concurrent executors exist
        // (server mode with multiple connections, or future Raft leader).
        for (row_idx, snapshot) in cas_snapshots.iter().enumerate() {
            for (variable, original_bytes) in snapshot {
                if let Some(Value::Int(id)) = matches[row_idx].get(variable) {
                    let node_id = NodeId::from_raw(*id as u64);
                    let key = encode_node_key(ctx.shard_id, node_id);
                    let current_bytes = ctx.mvcc_get(Partition::Node, &key)?;
                    match current_bytes {
                        Some(ref bytes) if bytes.as_slice() != original_bytes.as_slice() => {
                            return Err(ExecutionError::Conflict(format!(
                                "UPSERT conflict: node {variable}(id={id}) was modified \
                                 between MATCH and SET. Retry the UPSERT."
                            )));
                        }
                        None => {
                            return Err(ExecutionError::Conflict(format!(
                                "UPSERT conflict: node {variable}(id={id}) was deleted \
                                 between MATCH and SET."
                            )));
                        }
                        _ => {} // bytes match — no conflict
                    }
                }
            }
        }

        execute_update(&matches, on_match, ctx)
    } else {
        // Phase 3: ON CREATE — create nodes and edges from patterns (two-pass)
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

                        let key = encode_node_key(ctx.shard_id, node_id);
                        let bytes = record.to_msgpack().map_err(|e| {
                            ExecutionError::Serialization(format!("node serialize: {e}"))
                        })?;
                        ctx.mvcc_put(Partition::Node, &key, &bytes)?;
                        ctx.write_stats.nodes_created += 1;

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

                        // Forward posting list (merge operator, no read needed).
                        let fwd_key = encode_adj_key_forward(&edge_type, source_id);
                        ctx.adj_merge_add(&fwd_key, target_id.as_raw());

                        // Reverse posting list (merge operator, no read needed).
                        let rev_key = encode_adj_key_reverse(&edge_type, target_id);
                        ctx.adj_merge_add(&rev_key, source_id.as_raw());
                        ctx.write_stats.edges_created += 1;

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
            let node_id = ctx.id_allocator.next();
            let label = labels.first().cloned().unwrap_or_default();

            let mut record = NodeRecord::new(&label);
            let empty_row = Row::new();
            for (prop_name, expr) in property_filters {
                let val = eval_expr(expr, &empty_row);
                let field_id = ctx.interner.intern(prop_name);
                record.set(field_id, val);
            }

            let key = encode_node_key(ctx.shard_id, node_id);
            let bytes = record
                .to_msgpack()
                .map_err(|e| ExecutionError::Serialization(format!("node serialize: {e}")))?;
            ctx.mvcc_put(Partition::Node, &key, &bytes)?;

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
fn try_extract_vector(val: &Value) -> Option<Vec<f32>> {
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
        let schema_key = encode_label_schema_key(primary);
        match ctx.mvcc_get(Partition::Schema, &schema_key) {
            Ok(Some(bytes)) => LabelSchema::from_msgpack(&bytes).ok(),
            _ => None,
        }
    } else {
        None
    };

    let is_validated = schema
        .as_ref()
        .is_some_and(|s| s.mode == SchemaMode::Validated);

    for input_row in input_rows {
        let node_id = ctx.id_allocator.next();

        let mut record = NodeRecord::with_labels(labels.to_vec());
        for (prop_name, expr) in properties {
            // Map literals → Document for full dot-notation support in storage.
            let val = eval_expr(expr, input_row).map_to_document();

            if is_validated {
                // VALIDATED: declared properties → interned props,
                // undeclared → extra overflow map (string keys).
                let Some(schema_ref) = schema.as_ref() else {
                    unreachable!()
                };
                if schema_ref.get_property(prop_name).is_some() {
                    let field_id = ctx.interner.intern(prop_name);
                    record.set(field_id, val.clone());
                } else {
                    record.set_extra(prop_name, val.clone());
                }
            } else {
                // STRICT / FLEXIBLE / no schema: all properties interned.
                let field_id = ctx.interner.intern(prop_name);
                record.set(field_id, val.clone());
            }
        }

        let key = encode_node_key(ctx.shard_id, node_id);
        let bytes = record
            .to_msgpack()
            .map_err(|e| ExecutionError::Serialization(format!("node serialization: {e}")))?;

        ctx.mvcc_put(Partition::Node, &key, &bytes)?;
        ctx.write_stats.nodes_created += 1;
        ctx.write_stats.properties_set += properties.len() as u64;

        // Notify vector index registry of any vector properties.
        if let Some(registry) = ctx.vector_index_registry {
            if let Some(primary_label) = labels.first() {
                for (prop_name, expr) in properties {
                    let val = eval_expr(expr, input_row);
                    if let Some(vec_data) = try_extract_vector(&val) {
                        registry.on_vector_written(primary_label, node_id, prop_name, &vec_data);
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

        // Forward posting list: source -> targets (merge operator, no read needed).
        let fwd_key = encode_adj_key_forward(edge_type, source_id);
        ctx.adj_merge_add(&fwd_key, target_id.as_raw());

        // Reverse posting list: target <- sources (merge operator, no read needed).
        let rev_key = encode_adj_key_reverse(edge_type, target_id);
        ctx.adj_merge_add(&rev_key, source_id.as_raw());
        ctx.write_stats.edges_created += 1;

        // Register edge type in schema (idempotent marker).
        // This enables O(edge_types) targeted lookup in DETACH DELETE instead of
        // O(all_edges) full scan. Deduplicated within this transaction via write buffer.
        let et_key = edge_type_schema_key(edge_type);
        if !ctx
            .mvcc_write_buffer
            .contains_key(&(Partition::Schema, et_key.clone()))
        {
            ctx.mvcc_put(Partition::Schema, &et_key, b"")?;
        }

        // Store edge properties (facets) in EdgeProp partition.
        // Key: edgeprop:<TYPE>:<src BE>:<tgt BE>
        // Value: MessagePack map of field_id → Value (same as node properties)
        if !properties.is_empty() {
            let mut prop_map: Vec<(u32, Value)> = Vec::with_capacity(properties.len());
            for (prop_name, expr) in properties {
                let field_id = ctx.interner.intern(prop_name);
                let value = eval_expr(expr, row).map_to_document();
                prop_map.push((field_id, value));
            }
            let prop_bytes = rmp_serde::to_vec(&prop_map)
                .map_err(|e| ExecutionError::Serialization(format!("edge prop encode: {e}")))?;
            let ep_key = encode_edgeprop_key(edge_type, source_id, target_id);
            ctx.mvcc_put(Partition::EdgeProp, &ep_key, &prop_bytes)?;
            ctx.write_stats.properties_set += properties.len() as u64;
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
fn execute_update(
    input_rows: &[Row],
    items: &[crate::cypher::ast::SetItem],
    ctx: &mut ExecutionContext<'_>,
) -> Result<Vec<Row>, ExecutionError> {
    let mut results = Vec::new();

    for row in input_rows {
        let mut out_row = row.clone();

        for item in items {
            match item {
                crate::cypher::ast::SetItem::Property {
                    variable,
                    property,
                    expr,
                } => {
                    // Map literals → Document for nested property storage.
                    let val = eval_expr(expr, &out_row).map_to_document();
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    // Read current node record
                    let key = encode_node_key(ctx.shard_id, node_id);
                    if let Some(bytes) = ctx.mvcc_get(Partition::Node, &key)? {
                        let mut record = NodeRecord::from_msgpack(&bytes).map_err(|e| {
                            ExecutionError::Serialization(format!("node decode: {e}"))
                        })?;

                        let field_id = ctx.interner.intern(property);
                        record.set(field_id, val.clone());

                        let new_bytes = record.to_msgpack().map_err(|e| {
                            ExecutionError::Serialization(format!("node encode: {e}"))
                        })?;
                        ctx.mvcc_put(Partition::Node, &key, &new_bytes)?;
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
                    }

                    out_row.insert(format!("{variable}.{property}"), val);
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

                    // O(1) write via merge operand — no read required.
                    // path[0] = property name → resolved to field_id via interner
                    // path[1..] = nested path within the DOCUMENT value
                    let field_id = ctx.interner.intern(&path[0]);
                    let sub_path = &path[1..];
                    let key = encode_node_key(ctx.shard_id, node_id);

                    let delta = coordinode_core::graph::doc_delta::DocDelta::SetPath {
                        target: coordinode_core::graph::doc_delta::PathTarget::PropField(field_id),
                        path: sub_path.to_vec(),
                        value: val.to_rmpv(),
                    };
                    let operand = delta.encode().map_err(|e| {
                        ExecutionError::Serialization(format!("DocDelta encode: {e}"))
                    })?;
                    ctx.merge_node_deltas.push((key, operand));
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

                    let key = encode_node_key(ctx.shard_id, node_id);
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
                    ctx.merge_node_deltas.push((key, operand));
                    ctx.write_stats.properties_set += 1;
                }
                crate::cypher::ast::SetItem::ReplaceProperties { variable, expr } => {
                    let map_val = eval_expr(expr, &out_row);
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    let key = encode_node_key(ctx.shard_id, node_id);
                    if let Some(bytes) = ctx.mvcc_get(Partition::Node, &key)? {
                        let mut record = NodeRecord::from_msgpack(&bytes).map_err(|e| {
                            ExecutionError::Serialization(format!("node decode: {e}"))
                        })?;

                        // Clear existing props and set new ones from map
                        record.props.clear();
                        if let Value::Map(ref map) = map_val {
                            for (k, v) in map {
                                let field_id = ctx.interner.intern(k);
                                record.set(field_id, v.clone());
                            }
                        }

                        let new_bytes = record.to_msgpack().map_err(|e| {
                            ExecutionError::Serialization(format!("node encode: {e}"))
                        })?;
                        ctx.mvcc_put(Partition::Node, &key, &new_bytes)?;

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

                    let key = encode_node_key(ctx.shard_id, node_id);
                    if let Some(bytes) = ctx.mvcc_get(Partition::Node, &key)? {
                        let mut record = NodeRecord::from_msgpack(&bytes).map_err(|e| {
                            ExecutionError::Serialization(format!("node decode: {e}"))
                        })?;

                        if let Value::Map(ref map) = map_val {
                            for (k, v) in map {
                                let field_id = ctx.interner.intern(k);
                                record.set(field_id, v.clone());
                            }
                        }

                        let new_bytes = record.to_msgpack().map_err(|e| {
                            ExecutionError::Serialization(format!("node encode: {e}"))
                        })?;
                        ctx.mvcc_put(Partition::Node, &key, &new_bytes)?;

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

                    let key = encode_node_key(ctx.shard_id, node_id);
                    if let Some(bytes) = ctx.mvcc_get(Partition::Node, &key)? {
                        let mut record = NodeRecord::from_msgpack(&bytes).map_err(|e| {
                            ExecutionError::Serialization(format!("node decode: {e}"))
                        })?;

                        record.add_label(label.clone());
                        ctx.write_stats.labels_added += 1;

                        let new_bytes = record.to_msgpack().map_err(|e| {
                            ExecutionError::Serialization(format!("node encode: {e}"))
                        })?;
                        ctx.mvcc_put(Partition::Node, &key, &new_bytes)?;

                        out_row.insert(
                            format!("{variable}.__label__"),
                            Value::String(record.primary_label().to_string()),
                        );
                    }
                }
            }
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
    let mut results = Vec::new();

    for row in input_rows {
        let mut out_row = row.clone();

        for item in items {
            match item {
                crate::cypher::ast::RemoveItem::Property { variable, property } => {
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    let key = encode_node_key(ctx.shard_id, node_id);
                    if let Some(bytes) = ctx.mvcc_get(Partition::Node, &key)? {
                        let mut record = NodeRecord::from_msgpack(&bytes).map_err(|e| {
                            ExecutionError::Serialization(format!("node decode: {e}"))
                        })?;

                        if let Some(field_id) = ctx.interner.lookup(property) {
                            // Notify vector index if removing a vector property.
                            if let Some(registry) = ctx.vector_index_registry {
                                if record
                                    .props
                                    .get(&field_id)
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

                        let new_bytes = record.to_msgpack().map_err(|e| {
                            ExecutionError::Serialization(format!("node encode: {e}"))
                        })?;
                        ctx.mvcc_put(Partition::Node, &key, &new_bytes)?;
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
                    let key = encode_node_key(ctx.shard_id, node_id);

                    let delta = coordinode_core::graph::doc_delta::DocDelta::DeletePath {
                        target: coordinode_core::graph::doc_delta::PathTarget::PropField(field_id),
                        path: sub_path.to_vec(),
                    };
                    let operand = delta.encode().map_err(|e| {
                        ExecutionError::Serialization(format!("DocDelta encode: {e}"))
                    })?;
                    ctx.merge_node_deltas.push((key, operand));
                    ctx.write_stats.properties_removed += 1;

                    let path_str = path.join(".");
                    out_row.remove(&format!("{variable}.{path_str}"));
                }
                crate::cypher::ast::RemoveItem::Label { variable, label } => {
                    let node_id = match out_row.get(variable) {
                        Some(Value::Int(id)) => NodeId::from_raw(*id as u64),
                        _ => continue,
                    };

                    let key = encode_node_key(ctx.shard_id, node_id);
                    if let Some(bytes) = ctx.mvcc_get(Partition::Node, &key)? {
                        let mut record = NodeRecord::from_msgpack(&bytes).map_err(|e| {
                            ExecutionError::Serialization(format!("node decode: {e}"))
                        })?;

                        record.remove_label(label);
                        ctx.write_stats.labels_removed += 1;

                        let new_bytes = record.to_msgpack().map_err(|e| {
                            ExecutionError::Serialization(format!("node encode: {e}"))
                        })?;
                        ctx.mvcc_put(Partition::Node, &key, &new_bytes)?;

                        out_row.insert(
                            format!("{variable}.__label__"),
                            Value::String(record.primary_label().to_string()),
                        );
                    }
                }
            }
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
    for row in input_rows {
        for var in variables {
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
            let mut edges_to_delete = Vec::new();

            for edge_type in &edge_types {
                for adj_key in [
                    encode_adj_key_forward(edge_type, node_id),
                    encode_adj_key_reverse(edge_type, node_id),
                ] {
                    if ctx.adj_get(&adj_key)?.is_some() {
                        edge_count += 1;
                        if detach {
                            edges_to_delete.push(adj_key);
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
            for edge_key in &edges_to_delete {
                if let Some(parts) = decode_adj_key(edge_key) {
                    // Read posting list to find connected nodes
                    if let Some(plist) = ctx.adj_get(edge_key)? {
                        for peer_uid in plist.iter() {
                            let peer_id = NodeId::from_raw(peer_uid);
                            let counterpart_key = match parts.direction {
                                // This key is adj:TYPE:out:NODE → counterpart is adj:TYPE:in:PEER
                                AdjDirection::Out => {
                                    encode_adj_key_reverse(&parts.edge_type, peer_id)
                                }
                                // This key is adj:TYPE:in:NODE → counterpart is adj:TYPE:out:PEER
                                AdjDirection::In => {
                                    encode_adj_key_forward(&parts.edge_type, peer_id)
                                }
                            };
                            ctx.adj_merge_remove(&counterpart_key, node_id.as_raw());

                            // Clean up edge properties for this edge
                            let (ep_src, ep_tgt) = match parts.direction {
                                AdjDirection::Out => (node_id, peer_id),
                                AdjDirection::In => (peer_id, node_id),
                            };
                            let ep_key = encode_edgeprop_key(&parts.edge_type, ep_src, ep_tgt);
                            ctx.mvcc_delete(Partition::EdgeProp, &ep_key)?;
                        }
                    }
                }

                ctx.engine
                    .delete(Partition::Adj, edge_key)
                    .map_err(ExecutionError::Storage)?;
                ctx.write_stats.edges_deleted += 1;
            }
            // Also clear any pending merge adds for deleted keys.
            for edge_key in &edges_to_delete {
                ctx.merge_adj_adds.remove(edge_key);
                ctx.merge_adj_removes.remove(edge_key);
            }

            // Delete the node record.
            // Notify vector and text index registries of deleted properties.
            let key = encode_node_key(ctx.shard_id, node_id);
            if ctx.vector_index_registry.is_some() || ctx.text_index_registry.is_some() {
                if let Some(node_bytes) = ctx.mvcc_get(Partition::Node, &key)? {
                    if let Ok(record) = NodeRecord::from_msgpack(&node_bytes) {
                        let label = record.primary_label().to_string();
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
            ctx.mvcc_delete(Partition::Node, &key)?;
            ctx.write_stats.nodes_deleted += 1;
        }
    }

    // DELETE returns the input rows (so downstream RETURN can reference them)
    Ok(input_rows.to_vec())
}

/// Build the Schema-partition key for a given edge type name.
///
/// Format: `schema:edge_type:<name>`
fn edge_type_schema_key(edge_type: &str) -> Vec<u8> {
    const PREFIX: &[u8] = b"schema:edge_type:";
    let mut k = Vec::with_capacity(PREFIX.len() + edge_type.len());
    k.extend_from_slice(PREFIX);
    k.extend_from_slice(edge_type.as_bytes());
    k
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
    let schema_key = encode_label_schema_key(label);
    let schema = match engine.get(Partition::Schema, &schema_key) {
        Ok(Some(bytes)) => match LabelSchema::from_msgpack(&bytes) {
            Ok(s) => s,
            Err(_) => return,
        },
        _ => return,
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

    let key = encode_label_schema_key(label);

    // Load existing schema or create a new one.
    let mut schema = match ctx.mvcc_get(Partition::Schema, &key) {
        Ok(Some(bytes)) => LabelSchema::from_msgpack(&bytes).map_err(|e| {
            ExecutionError::Unsupported(format!("corrupt schema for label '{label}': {e}"))
        })?,
        Ok(None) => LabelSchema::new(label),
        Err(e) => {
            return Err(ExecutionError::Unsupported(format!(
                "storage error reading schema: {e}"
            )));
        }
    };

    schema.set_mode(mode);
    schema.version += 1;

    // Persist via MVCC write buffer → proposal pipeline.
    let value = schema
        .to_msgpack()
        .map_err(|e| ExecutionError::Unsupported(format!("failed to serialize schema: {e}")))?;
    ctx.mvcc_write_buffer
        .insert((Partition::Schema, key.clone()), Some(value));

    // Return result row with label info.
    let mut row = Row::new();
    row.insert("label".to_string(), Value::String(label.to_string()));
    row.insert("mode".to_string(), Value::String(format!("{mode}")));
    row.insert("version".to_string(), Value::Int(schema.version as i64));
    Ok(vec![row])
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

    // Persist index definition to schema: partition.
    let schema_key = def.schema_key();
    let bytes = rmp_serde::to_vec(&def)
        .map_err(|e| ExecutionError::Unsupported(format!("serialize index def: {e}")))?;
    ctx.mvcc_write_buffer
        .insert((Partition::Schema, schema_key), Some(bytes));

    // Register in text index registry (creates tantivy directory + empty index).
    registry
        .register(def)
        .map_err(|e| ExecutionError::Unsupported(format!("register text index: {e}")))?;

    // Backfill: scan existing nodes with this label and any indexed property.
    let shard_id = ctx.shard_id;
    let node_prefix = {
        let mut p = Vec::with_capacity(5 + 2 + 1);
        p.extend_from_slice(b"node:");
        p.extend_from_slice(&shard_id.to_be_bytes());
        p.push(b':');
        p
    };

    let mut count = 0u64;
    if let Ok(iter) = ctx.engine.prefix_scan(Partition::Node, &node_prefix) {
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
            let node_id = match coordinode_core::graph::node::decode_node_key(&_key) {
                Some((_shard, nid)) => nid,
                None => continue,
            };
            // Index all matching properties for this node.
            let mut indexed = false;
            for prop in &properties {
                if let Some(field_id) = ctx.interner.lookup(prop) {
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
    }

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

    // Remove from schema: partition.
    let schema_key = def.schema_key();
    ctx.mvcc_write_buffer
        .insert((Partition::Schema, schema_key), None);

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
    // Store the index definition keyed by name for easy DROP lookup.
    let schema_key = format!("encrypted_index:{name}");
    let meta = format!("{{\"name\":\"{name}\",\"label\":\"{label}\",\"property\":\"{property}\"}}");
    ctx.mvcc_write_buffer.insert(
        (Partition::Schema, schema_key.into_bytes()),
        Some(meta.into_bytes()),
    );

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
    // Scan for the schema key matching this index name.
    // Since we don't have a registry for encrypted indexes yet, we tombstone
    // all schema keys that end with the index name.
    // For now, write a tombstone for the known key pattern.
    // A full implementation would scan the schema partition.
    let tombstone_key = format!("encrypted_index:{name}");
    ctx.mvcc_write_buffer
        .insert((Partition::Schema, tombstone_key.into_bytes()), None);

    let mut row = Row::new();
    row.insert("index".to_string(), Value::String(name.to_string()));
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::cypher::ast::BinaryOperator;
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_storage::engine::config::StorageConfig;

    /// Create a test engine in a temp directory.
    fn test_engine(dir: &std::path::Path) -> StorageEngine {
        let config = StorageConfig::new(dir);
        StorageEngine::open(&config).expect("open engine")
    }

    /// Insert a test node into storage.
    fn insert_node(
        engine: &StorageEngine,
        shard_id: u16,
        node_id: u64,
        label: &str,
        props: &[(&str, Value)],
        interner: &mut FieldInterner,
    ) {
        let mut record = NodeRecord::new(label);
        for (name, value) in props {
            let field_id = interner.intern(name);
            record.set(field_id, value.clone());
        }
        let key = encode_node_key(shard_id, NodeId::from_raw(node_id));
        let bytes = record.to_msgpack().expect("serialize");
        engine.put(Partition::Node, &key, &bytes).expect("put node");
    }

    /// Insert an edge via merge operator (both forward and reverse posting list).
    /// Also registers the edge type in Schema so DETACH DELETE can use targeted lookup.
    fn insert_edge(engine: &StorageEngine, edge_type: &str, source_id: u64, target_id: u64) {
        use coordinode_storage::engine::merge::encode_add;

        // Register edge type in schema (required for targeted DETACH DELETE lookup).
        let et_key = edge_type_schema_key(edge_type);
        engine
            .put(Partition::Schema, &et_key, b"")
            .expect("register edge type in schema");

        // Forward posting list: merge add (no read needed)
        let fwd_key = encode_adj_key_forward(edge_type, NodeId::from_raw(source_id));
        engine
            .merge(Partition::Adj, &fwd_key, &encode_add(target_id))
            .expect("merge fwd");

        // Reverse posting list: merge add (no read needed)
        let rev_key = encode_adj_key_reverse(edge_type, NodeId::from_raw(target_id));
        engine
            .merge(Partition::Adj, &rev_key, &encode_add(source_id))
            .expect("merge rev");
    }

    fn setup_test_graph() -> (tempfile::TempDir, StorageEngine, FieldInterner) {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        // Create test nodes
        insert_node(
            &engine,
            1,
            1,
            "User",
            &[
                ("name", Value::String("Alice".into())),
                ("age", Value::Int(30)),
            ],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "User",
            &[
                ("name", Value::String("Bob".into())),
                ("age", Value::Int(25)),
            ],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            3,
            "User",
            &[
                ("name", Value::String("Charlie".into())),
                ("age", Value::Int(35)),
            ],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            4,
            "Movie",
            &[("title", Value::String("Matrix".into()))],
            &mut interner,
        );

        // Create edges: Alice->Bob, Alice->Charlie, Bob->Charlie
        insert_edge(&engine, "KNOWS", 1, 2);
        insert_edge(&engine, "KNOWS", 1, 3);
        insert_edge(&engine, "KNOWS", 2, 3);

        // Alice likes Matrix
        insert_edge(&engine, "LIKES", 1, 4);

        (dir, engine, interner)
    }

    fn make_ctx<'a>(
        engine: &'a StorageEngine,
        interner: &'a mut FieldInterner,
        allocator: &'a NodeIdAllocator,
    ) -> ExecutionContext<'a> {
        ExecutionContext {
            engine,
            interner,
            id_allocator: allocator,
            shard_id: 1,
            adaptive: AdaptiveConfig::default(),
            snapshot_ts: None,
            retention_window_us: 7 * 24 * 3600 * 1_000_000, // 7 days in micros
            warnings: Vec::new(),
            write_stats: WriteStats::default(),
            text_index: None,
            text_index_registry: None,
            vector_index_registry: None,
            vector_loader: None,
            mvcc_oracle: None,
            mvcc_read_ts: coordinode_core::txn::timestamp::Timestamp::ZERO,
            procedure_ctx: None,
            mvcc_write_buffer: std::collections::HashMap::new(),
            mvcc_read_set: std::collections::HashSet::new(),
            vector_consistency: VectorConsistencyMode::default(),
            vector_overfetch_factor: 1.2,
            vector_mvcc_stats: None,
            proposal_pipeline: None,
            proposal_id_gen: None,
            read_concern: coordinode_core::txn::read_concern::ReadConcernLevel::Local,
            write_concern: coordinode_core::txn::write_concern::WriteConcern::majority(),
            drain_buffer: None,
            nvme_write_buffer: None,
            merge_adj_adds: std::collections::HashMap::new(),
            merge_adj_removes: std::collections::HashMap::new(),
            mvcc_snapshot: None,
            adj_snapshot: None,
            merge_node_deltas: Vec::new(),
            correlated_row: None,
            feedback_cache: None,
        }
    }

    // -- NodeScan --

    #[test]
    fn node_scan_all() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec![],
                    property_filters: vec![],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::Star,
                    alias: None,
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 4); // Alice, Bob, Charlie, Matrix
    }

    #[test]
    fn node_scan_by_label() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::Star,
                    alias: None,
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 3); // Alice, Bob, Charlie
    }

    #[test]
    fn node_scan_with_property_filter() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::Star,
                    alias: None,
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].get("n.name"),
            Some(&Value::String("Alice".into()))
        );
    }

    // -- Traverse --

    #[test]
    fn traverse_outgoing() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // MATCH (a:User {name: 'Alice'})-[:KNOWS]->(b) RETURN b.name
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::Traverse {
                    input: Box::new(LogicalOp::NodeScan {
                        variable: "a".into(),
                        labels: vec!["User".into()],
                        property_filters: vec![(
                            "name".into(),
                            Expr::Literal(Value::String("Alice".into())),
                        )],
                    }),
                    source: "a".into(),
                    edge_types: vec!["KNOWS".into()],
                    direction: Direction::Outgoing,
                    target_variable: "b".into(),
                    target_labels: vec![],
                    length: None,
                    edge_variable: None,
                    target_filters: vec![],
                    edge_filters: vec![],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("b".into())),
                        property: "name".into(),
                    },
                    alias: Some("name".into()),
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 2); // Alice knows Bob and Charlie

        let names: Vec<&Value> = result.iter().filter_map(|r| r.get("name")).collect();
        assert!(names.contains(&&Value::String("Bob".into())));
        assert!(names.contains(&&Value::String("Charlie".into())));
    }

    // -- Filter --

    #[test]
    fn filter_by_age() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // MATCH (n:User) WHERE n.age > 28 RETURN n.name
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::Filter {
                    input: Box::new(LogicalOp::NodeScan {
                        variable: "n".into(),
                        labels: vec!["User".into()],
                        property_filters: vec![],
                    }),
                    predicate: Expr::BinaryOp {
                        left: Box::new(Expr::PropertyAccess {
                            expr: Box::new(Expr::Variable("n".into())),
                            property: "age".into(),
                        }),
                        op: BinaryOperator::Gt,
                        right: Box::new(Expr::Literal(Value::Int(28))),
                    },
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "name".into(),
                    },
                    alias: Some("name".into()),
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 2); // Alice (30) and Charlie (35)
    }

    // -- Aggregate --

    #[test]
    fn aggregate_count() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::Aggregate {
                    input: Box::new(LogicalOp::NodeScan {
                        variable: "n".into(),
                        labels: vec!["User".into()],
                        property_filters: vec![],
                    }),
                    group_by: vec![],
                    aggregates: vec![AggregateItem {
                        function: "count".into(),
                        arg: Expr::Star,
                        distinct: false,
                        alias: Some("cnt".into()),
                    }],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::Variable("cnt".into()),
                    alias: None,
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("cnt"), Some(&Value::Int(3)));
    }

    #[test]
    fn aggregate_sum_avg() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // sum(n.age) and avg(n.age) for User nodes (ages: 30, 25, 35)
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::Aggregate {
                    input: Box::new(LogicalOp::NodeScan {
                        variable: "n".into(),
                        labels: vec!["User".into()],
                        property_filters: vec![],
                    }),
                    group_by: vec![],
                    aggregates: vec![
                        AggregateItem {
                            function: "sum".into(),
                            arg: Expr::PropertyAccess {
                                expr: Box::new(Expr::Variable("n".into())),
                                property: "age".into(),
                            },
                            distinct: false,
                            alias: Some("total".into()),
                        },
                        AggregateItem {
                            function: "avg".into(),
                            arg: Expr::PropertyAccess {
                                expr: Box::new(Expr::Variable("n".into())),
                                property: "age".into(),
                            },
                            distinct: false,
                            alias: Some("average".into()),
                        },
                    ],
                }),
                items: vec![
                    crate::planner::logical::ProjectItem {
                        expr: Expr::Variable("total".into()),
                        alias: None,
                    },
                    crate::planner::logical::ProjectItem {
                        expr: Expr::Variable("average".into()),
                        alias: None,
                    },
                ],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("total"), Some(&Value::Int(90))); // 30+25+35 (all Int)
        assert_eq!(result[0].get("average"), Some(&Value::Float(30.0))); // 90/3
    }

    #[test]
    fn aggregate_min_max() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![
                    AggregateItem {
                        function: "min".into(),
                        arg: Expr::PropertyAccess {
                            expr: Box::new(Expr::Variable("n".into())),
                            property: "age".into(),
                        },
                        distinct: false,
                        alias: Some("youngest".into()),
                    },
                    AggregateItem {
                        function: "max".into(),
                        arg: Expr::PropertyAccess {
                            expr: Box::new(Expr::Variable("n".into())),
                            property: "age".into(),
                        },
                        distinct: false,
                        alias: Some("oldest".into()),
                    },
                ],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("youngest"), Some(&Value::Int(25)));
        assert_eq!(result[0].get("oldest"), Some(&Value::Int(35)));
    }

    #[test]
    fn aggregate_collect() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![AggregateItem {
                    function: "collect".into(),
                    arg: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "name".into(),
                    },
                    distinct: false,
                    alias: Some("names".into()),
                }],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        if let Some(Value::Array(names)) = result[0].get("names") {
            assert_eq!(names.len(), 3);
            // Names should include Alice, Bob, Charlie (order may vary)
            let name_strings: Vec<&str> = names.iter().filter_map(|v| v.as_str()).collect();
            assert!(name_strings.contains(&"Alice"));
            assert!(name_strings.contains(&"Bob"));
            assert!(name_strings.contains(&"Charlie"));
        } else {
            panic!("expected Array for collect()");
        }
    }

    #[test]
    fn aggregate_percentile_cont() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // percentileCont(n.age, 0.5) — median of [25, 30, 35] = 30.0
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![AggregateItem {
                    function: "percentileCont".into(),
                    arg: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    },
                    distinct: false,
                    alias: Some("median".into()),
                }],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        // Median of [25, 30, 35] = 30.0
        assert_eq!(result[0].get("median"), Some(&Value::Float(30.0)));
    }

    #[test]
    fn aggregate_stdev() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![AggregateItem {
                    function: "stDev".into(),
                    arg: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    },
                    distinct: false,
                    alias: Some("sd".into()),
                }],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        // stDev of [25, 30, 35] = 5.0 (sample)
        if let Some(Value::Float(sd)) = result[0].get("sd") {
            assert!((sd - 5.0).abs() < 0.01, "expected ~5.0, got {sd}");
        } else {
            panic!("expected Float for stDev");
        }
    }

    #[test]
    fn aggregate_empty_returns_null() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // sum/avg on empty set
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["NonExistent".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![
                    AggregateItem {
                        function: "sum".into(),
                        arg: Expr::PropertyAccess {
                            expr: Box::new(Expr::Variable("n".into())),
                            property: "age".into(),
                        },
                        distinct: false,
                        alias: Some("total".into()),
                    },
                    AggregateItem {
                        function: "count".into(),
                        arg: Expr::Star,
                        distinct: false,
                        alias: Some("cnt".into()),
                    },
                ],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        // count(*) on empty set = 0
        assert_eq!(result[0].get("cnt"), Some(&Value::Int(0)));
        // sum on empty set = null
        assert_eq!(result[0].get("total"), Some(&Value::Null));
    }

    // -- Sort + Limit --

    #[test]
    fn sort_and_limit() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Limit {
                input: Box::new(LogicalOp::Sort {
                    input: Box::new(LogicalOp::Project {
                        input: Box::new(LogicalOp::NodeScan {
                            variable: "n".into(),
                            labels: vec!["User".into()],
                            property_filters: vec![],
                        }),
                        items: vec![
                            crate::planner::logical::ProjectItem {
                                expr: Expr::PropertyAccess {
                                    expr: Box::new(Expr::Variable("n".into())),
                                    property: "name".into(),
                                },
                                alias: Some("name".into()),
                            },
                            crate::planner::logical::ProjectItem {
                                expr: Expr::PropertyAccess {
                                    expr: Box::new(Expr::Variable("n".into())),
                                    property: "age".into(),
                                },
                                alias: Some("age".into()),
                            },
                        ],
                        distinct: false,
                    }),
                    items: vec![crate::cypher::ast::SortItem {
                        expr: Expr::Variable("age".into()),
                        ascending: false,
                    }],
                }),
                count: Expr::Literal(Value::Int(2)),
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 2);
        // Sorted by age DESC: Charlie (35), Alice (30)
        assert_eq!(
            result[0].get("name"),
            Some(&Value::String("Charlie".into()))
        );
        assert_eq!(result[1].get("name"), Some(&Value::String("Alice".into())));
    }

    // -- Empty result --

    #[test]
    fn empty_scan() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["NonExistent".into()],
                    property_filters: vec![],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::Star,
                    alias: None,
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert!(result.is_empty());
    }

    // ====== Write operations ======

    #[test]
    fn create_node_writes_to_storage() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::new();
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::CreateNode {
                input: None,
                variable: Some("n".into()),
                labels: vec!["User".into()],
                properties: vec![
                    ("name".into(), Expr::Literal(Value::String("Alice".into()))),
                    ("age".into(), Expr::Literal(Value::Int(30))),
                ],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains_key("n")); // Should have node ID

        // Verify node was written to storage
        let node_id = result[0]
            .get("n")
            .and_then(|v| v.as_int())
            .expect("node id");
        let key = encode_node_key(1, NodeId::from_raw(node_id as u64));
        let bytes = engine
            .get(Partition::Node, &key)
            .expect("get")
            .expect("node exists");
        let record = NodeRecord::from_msgpack(&bytes).expect("decode");
        assert_eq!(record.primary_label(), "User");
    }

    #[test]
    fn create_and_return() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::new();
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::CreateNode {
                    input: None,
                    variable: Some("n".into()),
                    labels: vec!["User".into()],
                    properties: vec![("name".into(), Expr::Literal(Value::String("Bob".into())))],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "name".into(),
                    },
                    alias: Some("name".into()),
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("name"), Some(&Value::String("Bob".into())));
    }

    #[test]
    fn set_property_updates_storage() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // MATCH (n:User {name: 'Alice'}) SET n.name = 'Alicia'
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Update {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                items: vec![crate::cypher::ast::SetItem::Property {
                    variable: "n".into(),
                    property: "name".into(),
                    expr: Expr::Literal(Value::String("Alicia".into())),
                }],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);

        // Verify the update persisted in storage
        let node_id = result[0].get("n").and_then(|v| v.as_int()).expect("id");
        let key = encode_node_key(1, NodeId::from_raw(node_id as u64));
        let bytes = engine
            .get(Partition::Node, &key)
            .expect("get")
            .expect("exists");
        let record = NodeRecord::from_msgpack(&bytes).expect("decode");
        let name_id = interner.lookup("name").expect("field id");
        assert_eq!(record.get(name_id), Some(&Value::String("Alicia".into())));
    }

    #[test]
    fn delete_removes_from_storage() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // First, verify Alice exists
        let key = encode_node_key(1, NodeId::from_raw(1));
        assert!(engine.get(Partition::Node, &key).expect("get").is_some());

        // DETACH DELETE node 1 (Alice) — she has edges, so DETACH is required.
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Delete {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                variables: vec!["n".into()],
                detach: true,
            },
        };

        execute(&plan, &mut ctx).expect("execute");

        // Verify Alice was deleted
        assert!(engine.get(Partition::Node, &key).expect("get").is_none());
    }

    #[test]
    fn remove_property_from_node() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // MATCH (n:User {name: 'Alice'}) REMOVE n.age
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::RemoveOp {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                items: vec![crate::cypher::ast::RemoveItem::Property {
                    variable: "n".into(),
                    property: "age".into(),
                }],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);

        // Verify age was removed
        let node_id = result[0].get("n").and_then(|v| v.as_int()).expect("id");
        let key = encode_node_key(1, NodeId::from_raw(node_id as u64));
        let bytes = engine
            .get(Partition::Node, &key)
            .expect("get")
            .expect("exists");
        let record = NodeRecord::from_msgpack(&bytes).expect("decode");
        let age_id = interner.lookup("age").expect("field id");
        assert!(record.get(age_id).is_none());
    }

    #[test]
    fn create_multiple_nodes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::new();
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // Create first node
        let plan1 = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::CreateNode {
                input: None,
                variable: Some("a".into()),
                labels: vec!["User".into()],
                properties: vec![("name".into(), Expr::Literal(Value::String("X".into())))],
            },
        };
        execute(&plan1, &mut ctx).expect("create a");

        // Create second node
        let plan2 = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::CreateNode {
                input: None,
                variable: Some("b".into()),
                labels: vec!["User".into()],
                properties: vec![("name".into(), Expr::Literal(Value::String("Y".into())))],
            },
        };
        execute(&plan2, &mut ctx).expect("create b");

        // Verify both exist with unique IDs
        assert_eq!(allocator.current().as_raw(), 2);
    }

    // ====== MERGE / UPSERT ======

    #[test]
    fn merge_creates_when_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::new();
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // MERGE (n:User {email: 'alice@test.com'}) ON CREATE SET n.name = 'Alice'
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Merge {
                pattern: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "email".into(),
                        Expr::Literal(Value::String("alice@test.com".into())),
                    )],
                }),
                on_match: vec![],
                on_create: vec![crate::cypher::ast::SetItem::Property {
                    variable: "n".into(),
                    property: "name".into(),
                    expr: Expr::Literal(Value::String("Alice".into())),
                }],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        // Node should have been created
        assert!(result[0].contains_key("n"));
    }

    #[test]
    fn merge_updates_when_found() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // MERGE (n:User {name: 'Alice'}) ON MATCH SET n.age = 31
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Merge {
                pattern: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                on_match: vec![crate::cypher::ast::SetItem::Property {
                    variable: "n".into(),
                    property: "age".into(),
                    expr: Expr::Literal(Value::Int(31)),
                }],
                on_create: vec![],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);

        // Verify age was updated
        let node_id = result[0].get("n").and_then(|v| v.as_int()).expect("id");
        let key = encode_node_key(1, NodeId::from_raw(node_id as u64));
        let bytes = engine
            .get(Partition::Node, &key)
            .expect("get")
            .expect("exists");
        let record = NodeRecord::from_msgpack(&bytes).expect("decode");
        let age_id = interner.lookup("age").expect("field id");
        assert_eq!(record.get(age_id), Some(&Value::Int(31)));
    }

    #[test]
    fn merge_no_duplicate_on_existing() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // MERGE on existing node should NOT create a new node
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Merge {
                pattern: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                on_match: vec![],
                on_create: vec![],
            },
        };

        execute(&plan, &mut ctx).expect("execute");

        // Allocator should NOT have advanced (no new node created)
        assert_eq!(allocator.current().as_raw(), 100);
    }

    #[test]
    fn upsert_creates_when_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::new();
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // UPSERT MATCH (u:User {email: 'bob@test.com'})
        // ON CREATE CREATE (u:User {email: 'bob@test.com', name: 'Bob'})
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Upsert {
                pattern: Box::new(LogicalOp::NodeScan {
                    variable: "u".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "email".into(),
                        Expr::Literal(Value::String("bob@test.com".into())),
                    )],
                }),
                on_match: vec![],
                on_create_patterns: vec![Pattern {
                    elements: vec![PatternElement::Node(crate::cypher::ast::NodePattern {
                        variable: Some("u".into()),
                        labels: vec!["User".into()],
                        properties: vec![
                            (
                                "email".into(),
                                Expr::Literal(Value::String("bob@test.com".into())),
                            ),
                            ("name".into(), Expr::Literal(Value::String("Bob".into()))),
                        ],
                    })],
                }],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("u.name"), Some(&Value::String("Bob".into())));
    }

    #[test]
    fn upsert_updates_when_found() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // UPSERT MATCH (n:User {name: 'Alice'})
        // ON MATCH SET n.age = 31
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Upsert {
                pattern: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                on_match: vec![crate::cypher::ast::SetItem::Property {
                    variable: "n".into(),
                    property: "age".into(),
                    expr: Expr::Literal(Value::Int(31)),
                }],
                on_create_patterns: vec![],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);

        // Verify no new node was created
        assert_eq!(allocator.current().as_raw(), 100);

        // Verify age was updated
        let node_id = result[0].get("n").and_then(|v| v.as_int()).expect("id");
        let key = encode_node_key(1, NodeId::from_raw(node_id as u64));
        let bytes = engine
            .get(Partition::Node, &key)
            .expect("get")
            .expect("exists");
        let record = NodeRecord::from_msgpack(&bytes).expect("decode");
        let age_id = interner.lookup("age").expect("field id");
        assert_eq!(record.get(age_id), Some(&Value::Int(31)));
    }

    // ====== UPSERT CAS conflict ======

    #[test]
    fn upsert_cas_conflict_on_external_modification() {
        // Test CAS conflict detection: modify node externally between
        // two sequential UPSERTs that target the same node.
        // The first UPSERT changes the node, the second should still succeed
        // because CAS reads fresh bytes before comparing.
        //
        // To test actual conflict detection, we'd need to inject a
        // modification between the MATCH and SET phases inside execute_upsert.
        // Since execute_upsert is private, we verify the mechanism indirectly:
        // two sequential UPSERTs both succeed (no false positives).
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));

        // First UPSERT: set age=50
        {
            let mut ctx = make_ctx(&engine, &mut interner, &allocator);
            let plan = LogicalPlan {
                snapshot_ts: None,
                vector_consistency: VectorConsistencyMode::default(),
                root: LogicalOp::Upsert {
                    pattern: Box::new(LogicalOp::NodeScan {
                        variable: "n".into(),
                        labels: vec!["User".into()],
                        property_filters: vec![(
                            "name".into(),
                            Expr::Literal(Value::String("Alice".into())),
                        )],
                    }),
                    on_match: vec![crate::cypher::ast::SetItem::Property {
                        variable: "n".into(),
                        property: "age".into(),
                        expr: Expr::Literal(Value::Int(50)),
                    }],
                    on_create_patterns: vec![],
                },
            };
            execute(&plan, &mut ctx).expect("first upsert");
        }

        // Externally modify Alice's age directly via storage (simulates
        // concurrent modification from another connection)
        let alice_id = NodeId::from_raw(1);
        let key = encode_node_key(1, alice_id);
        {
            let bytes = engine.get(Partition::Node, &key).unwrap().unwrap();
            let mut record = NodeRecord::from_msgpack(&bytes).unwrap();
            let age_id = interner.lookup("age").unwrap();
            record.set(age_id, Value::Int(999)); // external modification
            let new_bytes = record.to_msgpack().unwrap();
            engine.put(Partition::Node, &key, &new_bytes).unwrap();
        }

        // Second UPSERT: this reads fresh bytes in CAS snapshot,
        // then CAS re-reads and compares — they match because CAS
        // snapshot was taken after the external modification.
        // So this should succeed (CAS reads at the same moment).
        {
            let mut ctx = make_ctx(&engine, &mut interner, &allocator);
            let plan = LogicalPlan {
                snapshot_ts: None,
                vector_consistency: VectorConsistencyMode::default(),
                root: LogicalOp::Upsert {
                    pattern: Box::new(LogicalOp::NodeScan {
                        variable: "n".into(),
                        labels: vec!["User".into()],
                        property_filters: vec![(
                            "name".into(),
                            Expr::Literal(Value::String("Alice".into())),
                        )],
                    }),
                    on_match: vec![crate::cypher::ast::SetItem::Property {
                        variable: "n".into(),
                        property: "age".into(),
                        expr: Expr::Literal(Value::Int(60)),
                    }],
                    on_create_patterns: vec![],
                },
            };
            let result = execute(&plan, &mut ctx).expect("second upsert should succeed");
            assert_eq!(result.len(), 1);
        }

        // Verify final value is 60 (second UPSERT applied)
        let final_bytes = engine.get(Partition::Node, &key).unwrap().unwrap();
        let record = NodeRecord::from_msgpack(&final_bytes).unwrap();
        let age_id = interner.lookup("age").unwrap();
        assert_eq!(record.get(age_id), Some(&Value::Int(60)));
    }

    #[test]
    fn upsert_errconflict_variant_exists() {
        // Verify ExecutionError::Conflict variant exists and formats correctly
        let err = ExecutionError::Conflict("test conflict".into());
        let msg = format!("{err}");
        assert!(msg.contains("write conflict"));
        assert!(msg.contains("test conflict"));
    }

    // ====== Adaptive query plans ======

    #[test]
    fn adaptive_parallel_on_super_node() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::new();

        // Create source node
        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("name", Value::String("Hub".into()))],
            &mut interner,
        );

        // Create 20 target nodes and edges (simulate high fan-out)
        for i in 2..=21u64 {
            insert_node(
                &engine,
                1,
                i,
                "User",
                &[("name", Value::String(format!("T{i}")))],
                &mut interner,
            );
            insert_edge(&engine, "FOLLOWS", 1, i);
        }

        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        // Set low threshold to trigger parallel processing on 20 edges
        ctx.adaptive.parallel_threshold = 5;

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::Traverse {
                    input: Box::new(LogicalOp::NodeScan {
                        variable: "a".into(),
                        labels: vec!["User".into()],
                        property_filters: vec![(
                            "name".into(),
                            Expr::Literal(Value::String("Hub".into())),
                        )],
                    }),
                    source: "a".into(),
                    edge_types: vec!["FOLLOWS".into()],
                    direction: Direction::Outgoing,
                    target_variable: "b".into(),
                    target_labels: vec![],
                    length: None,
                    edge_variable: None,
                    target_filters: vec![],
                    edge_filters: vec![],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::Star,
                    alias: None,
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        // ALL 20 results returned (no truncation — parallel processes all edges)
        assert_eq!(result.len(), 20);
        // Should have a warning about parallel activation
        assert!(
            ctx.warnings.iter().any(|w| w.contains("parallel")),
            "expected parallel activation warning, got: {:?}",
            ctx.warnings,
        );
    }

    #[test]
    fn adaptive_disabled_no_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::new();

        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("name", Value::String("Hub".into()))],
            &mut interner,
        );
        for i in 2..=11u64 {
            insert_node(
                &engine,
                1,
                i,
                "User",
                &[("name", Value::String(format!("T{i}")))],
                &mut interner,
            );
            insert_edge(&engine, "FOLLOWS", 1, i);
        }

        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        ctx.adaptive.enabled = false; // Disable adaptive

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Hub".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["FOLLOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: None,
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        // All 10 results should be returned (no cap)
        assert_eq!(result.len(), 10);
        assert!(ctx.warnings.is_empty());
    }

    #[test]
    fn adaptive_normal_fan_out_no_warning() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        // Default max_fan_out is 10_000, our test graph has 2-3 edges per node

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["KNOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: None,
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 2); // Alice knows Bob and Charlie
        assert!(ctx.warnings.is_empty()); // No super-node warning
    }

    #[test]
    fn feedback_cache_records_super_node() {
        let cache = FeedbackCache::new(100);
        assert!(cache.lookup(42).is_none());

        cache.record(42, 50_000);
        assert_eq!(cache.lookup(42), Some(50_000));

        // Overwrite with new degree
        cache.record(42, 75_000);
        assert_eq!(cache.lookup(42), Some(75_000));
    }

    #[test]
    fn feedback_cache_evicts_when_full() {
        let cache = FeedbackCache::new(10);
        for i in 0..10 {
            cache.record(i, 1000 + i as usize);
        }
        assert_eq!(cache.lookup(0), Some(1000));
        assert_eq!(cache.lookup(9), Some(1009));

        // Adding one more should trigger eviction of half (5 entries)
        cache.record(100, 9999);
        // After eviction, some early entries should be gone
        let remaining: usize = (0..10).filter(|i| cache.lookup(*i).is_some()).count();
        assert!(
            remaining < 10,
            "expected eviction, but {remaining}/10 entries remain"
        );
        // New entry should be present
        assert_eq!(cache.lookup(100), Some(9999));
    }

    #[test]
    fn adaptive_parallel_correctness_matches_sequential() {
        // Verify parallel path produces the same results as sequential
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::new();

        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("name", Value::String("Hub".into()))],
            &mut interner,
        );
        for i in 2..=11u64 {
            insert_node(
                &engine,
                1,
                i,
                "User",
                &[("name", Value::String(format!("T{i}")))],
                &mut interner,
            );
            insert_edge(&engine, "FOLLOWS", 1, i);
        }

        // Run with parallel (threshold = 5, so 10 edges triggers parallel)
        let mut ctx_par = make_ctx(&engine, &mut interner, &allocator);
        ctx_par.adaptive.parallel_threshold = 5;

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Hub".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["FOLLOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: None,
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
            },
        };

        let result_par = execute(&plan, &mut ctx_par).expect("parallel execute");

        // Run without parallel (disabled)
        let mut ctx_seq = make_ctx(&engine, &mut interner, &allocator);
        ctx_seq.adaptive.enabled = false;

        let result_seq = execute(&plan, &mut ctx_seq).expect("sequential execute");

        // Same number of results
        assert_eq!(result_par.len(), result_seq.len());
        assert_eq!(result_par.len(), 10);

        // Same node IDs (order may differ due to parallel execution)
        let mut par_ids: Vec<i64> = result_par
            .iter()
            .filter_map(|r| {
                r.get("b").and_then(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
            })
            .collect();
        let mut seq_ids: Vec<i64> = result_seq
            .iter()
            .filter_map(|r| {
                r.get("b").and_then(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
            })
            .collect();
        par_ids.sort();
        seq_ids.sort();
        assert_eq!(par_ids, seq_ids);
    }

    // ====== AS OF TIMESTAMP ======

    #[test]
    fn as_of_timestamp_sets_snapshot() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // Use a recent timestamp (within retention window)
        let recent_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0)
            - 3600 * 1_000_000; // 1 hour ago

        let plan = LogicalPlan {
            snapshot_ts: Some(Expr::Literal(Value::Int(recent_ts))),
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::Star,
                    alias: None,
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert!(!result.is_empty());
        // Snapshot timestamp should be set
        assert_eq!(ctx.snapshot_ts, Some(recent_ts));
    }

    #[test]
    fn as_of_timestamp_rejects_expired() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // Use a very old timestamp (30 days ago — outside 7-day retention)
        let old_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0)
            - 30 * 24 * 3600 * 1_000_000; // 30 days ago

        let plan = LogicalPlan {
            snapshot_ts: Some(Expr::Literal(Value::Timestamp(old_ts))),
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::Star,
                    alias: None,
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("retention window"));
    }

    #[test]
    fn as_of_timestamp_string_warning() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: Some(Expr::Literal(Value::String("2025-06-15T10:00:00Z".into()))),
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::Star,
                    alias: None,
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert!(!result.is_empty());
        // Should have a warning about string parsing
        assert!(ctx.warnings.iter().any(|w| w.contains("AS OF TIMESTAMP")));
    }

    #[test]
    fn no_timestamp_leaves_snapshot_none() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                items: vec![crate::planner::logical::ProjectItem {
                    expr: Expr::Star,
                    alias: None,
                }],
                distinct: false,
            },
        };

        execute(&plan, &mut ctx).expect("execute");
        assert!(ctx.snapshot_ts.is_none());
    }

    // -- Variable-length path traversal --

    /// Build a longer test graph for multi-hop tests:
    /// A(1)→B(2)→C(3)→D(5)→E(6), plus A→C, B→C (already in setup)
    fn setup_varlen_graph() -> (tempfile::TempDir, StorageEngine, FieldInterner) {
        let (dir, engine, mut interner) = setup_test_graph();

        // Add nodes D and E
        insert_node(
            &engine,
            1,
            5,
            "User",
            &[("name", Value::String("Dave".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            6,
            "User",
            &[("name", Value::String("Eve".into()))],
            &mut interner,
        );

        // Extend chain: Charlie(3)→Dave(5), Dave(5)→Eve(6)
        insert_edge(&engine, "KNOWS", 3, 5);
        insert_edge(&engine, "KNOWS", 5, 6);

        (dir, engine, interner)
    }

    #[test]
    fn varlen_traverse_exact_2_hops() {
        // Alice -[:KNOWS*2..2]-> ? should find Charlie (via Alice→Bob→Charlie)
        // and Dave (via Alice→Charlie→Dave)
        let (_dir, engine, mut interner) = setup_varlen_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["KNOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: Some(LengthBound {
                    min: Some(2),
                    max: Some(2),
                }),
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
            },
        };

        let results = execute(&plan, &mut ctx).expect("execute");
        let target_ids: Vec<i64> = results
            .iter()
            .filter_map(|r| match r.get("b") {
                Some(Value::Int(id)) => Some(*id),
                _ => None,
            })
            .collect();
        // At 2 hops from Alice: Charlie (via Bob) and Dave (via Charlie)
        assert!(target_ids.contains(&3), "should reach Charlie at 2 hops");
        assert!(target_ids.contains(&5), "should reach Dave at 2 hops");
    }

    #[test]
    fn varlen_traverse_range_1_to_3() {
        // Alice -[:KNOWS*1..3]-> ? should find Bob(1hop), Charlie(1hop+2hop), Dave(2hop+3hop), Eve(3hop)
        let (_dir, engine, mut interner) = setup_varlen_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["KNOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: Some(LengthBound {
                    min: Some(1),
                    max: Some(3),
                }),
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
            },
        };

        let results = execute(&plan, &mut ctx).expect("execute");
        let target_ids: Vec<i64> = results
            .iter()
            .filter_map(|r| match r.get("b") {
                Some(Value::Int(id)) => Some(*id),
                _ => None,
            })
            .collect();
        // Should reach: Bob(1), Charlie(1+2), Dave(2+3), Eve(3)
        assert!(target_ids.contains(&2), "should reach Bob");
        assert!(target_ids.contains(&3), "should reach Charlie");
        assert!(target_ids.contains(&5), "should reach Dave");
        assert!(target_ids.contains(&6), "should reach Eve");
    }

    #[test]
    fn varlen_traverse_cycle_detection() {
        // Create a cycle: A→B→C→A
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("name", Value::String("A".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "User",
            &[("name", Value::String("B".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            3,
            "User",
            &[("name", Value::String("C".into()))],
            &mut interner,
        );

        insert_edge(&engine, "KNOWS", 1, 2); // A→B
        insert_edge(&engine, "KNOWS", 2, 3); // B→C
        insert_edge(&engine, "KNOWS", 3, 1); // C→A (cycle!)

        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // Traverse *1..10 — should NOT loop forever thanks to edge-uniqueness
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("A".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["KNOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: Some(LengthBound {
                    min: Some(1),
                    max: Some(10),
                }),
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
            },
        };

        let results = execute(&plan, &mut ctx).expect("execute should not hang");
        // With edge-uniqueness: 3 edges total (A→B, B→C, C→A), so max 3 results
        assert!(
            results.len() <= 3,
            "cycle detection should cap at 3 edges, got {}",
            results.len()
        );
    }

    #[test]
    fn varlen_traverse_min_greater_than_max_yields_empty() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["KNOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: Some(LengthBound {
                    min: Some(5),
                    max: Some(2),
                }),
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
            },
        };

        let results = execute(&plan, &mut ctx).expect("execute");
        assert!(results.is_empty(), "min > max should yield no results");
    }

    // -- UNWIND --

    #[test]
    fn unwind_list_expansion() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // UNWIND [1, 2, 3] AS x → 3 rows
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Unwind {
                input: Box::new(LogicalOp::Empty),
                expr: Expr::List(vec![
                    Expr::Literal(Value::Int(1)),
                    Expr::Literal(Value::Int(2)),
                    Expr::Literal(Value::Int(3)),
                ]),
                variable: "x".into(),
            },
        };

        let results = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(results.len(), 3, "UNWIND [1,2,3] should produce 3 rows");
        assert_eq!(results[0].get("x"), Some(&Value::Int(1)));
        assert_eq!(results[1].get("x"), Some(&Value::Int(2)));
        assert_eq!(results[2].get("x"), Some(&Value::Int(3)));
    }

    #[test]
    fn unwind_null_produces_empty() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Unwind {
                input: Box::new(LogicalOp::Empty),
                expr: Expr::Literal(Value::Null),
                variable: "x".into(),
            },
        };

        let results = execute(&plan, &mut ctx).expect("execute");
        assert!(results.is_empty(), "UNWIND NULL should produce zero rows");
    }

    #[test]
    fn unwind_scalar_single_row() {
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Unwind {
                input: Box::new(LogicalOp::Empty),
                expr: Expr::Literal(Value::Int(42)),
                variable: "x".into(),
            },
        };

        let results = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(results.len(), 1, "UNWIND scalar should produce 1 row");
        assert_eq!(results[0].get("x"), Some(&Value::Int(42)));
    }

    // -- OPTIONAL MATCH (LeftOuterJoin) --

    #[test]
    fn optional_match_with_results() {
        // Alice has KNOWS edges → should return real results
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::LeftOuterJoin {
                left: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                right: Box::new(LogicalOp::NodeScan {
                    variable: "m".into(),
                    labels: vec!["Movie".into()],
                    property_filters: vec![],
                }),
            },
        };

        let results = execute(&plan, &mut ctx).expect("execute");
        // Alice × Movie(Matrix) → at least 1 row
        assert!(!results.is_empty(), "should have results from Movie scan");
        assert!(results[0].contains_key("a"), "left variable should be set");
        assert!(results[0].contains_key("m"), "right variable should be set");
    }

    #[test]
    fn optional_match_no_results_nulls() {
        // Scan for non-existent label → right side empty → NULLs
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::LeftOuterJoin {
                left: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                right: Box::new(LogicalOp::NodeScan {
                    variable: "x".into(),
                    labels: vec!["NonExistent".into()],
                    property_filters: vec![],
                }),
            },
        };

        let results = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(results.len(), 1, "should have 1 row (left row with NULLs)");
        assert!(results[0].contains_key("a"), "left variable should be set");
        assert_eq!(
            results[0].get("x"),
            Some(&Value::Null),
            "right var should be NULL"
        );
    }

    // -- Shortest Path --

    #[test]
    fn shortest_path_direct() {
        // Alice→Bob is 1 hop
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let mut input_row = Row::new();
        input_row.insert("a".into(), Value::Int(1)); // Alice
        input_row.insert("b".into(), Value::Int(2)); // Bob

        let sp = ShortestPathParams {
            source: "a",
            target: "b",
            edge_types: &["KNOWS".into()],
            direction: Direction::Outgoing,
            max_depth: 10,
            path_variable: "p",
        };

        let results = execute_shortest_path(&[input_row], &sp, &mut ctx).expect("sp");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("p"), Some(&Value::Int(1)));
    }

    #[test]
    fn shortest_path_two_hops() {
        // Alice→Bob→Charlie is 2 hops; Alice→Charlie is 1 hop (direct)
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let mut input_row = Row::new();
        input_row.insert("a".into(), Value::Int(1)); // Alice
        input_row.insert("c".into(), Value::Int(3)); // Charlie

        let sp = ShortestPathParams {
            source: "a",
            target: "c",
            edge_types: &["KNOWS".into()],
            direction: Direction::Outgoing,
            max_depth: 10,
            path_variable: "p",
        };

        let results = execute_shortest_path(&[input_row], &sp, &mut ctx).expect("sp");
        assert_eq!(results.len(), 1);
        // Alice→Charlie is direct (1 hop), should find shortest
        assert_eq!(results[0].get("p"), Some(&Value::Int(1)));
    }

    #[test]
    fn shortest_path_unreachable() {
        // Matrix(4) has no KNOWS edges to anyone
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let mut input_row = Row::new();
        input_row.insert("a".into(), Value::Int(4)); // Matrix
        input_row.insert("b".into(), Value::Int(1)); // Alice

        let sp = ShortestPathParams {
            source: "a",
            target: "b",
            edge_types: &["KNOWS".into()],
            direction: Direction::Outgoing,
            max_depth: 10,
            path_variable: "p",
        };

        let results = execute_shortest_path(&[input_row], &sp, &mut ctx).expect("sp");
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].get("p"),
            Some(&Value::Null),
            "unreachable → NULL"
        );
    }

    #[test]
    fn shortest_path_same_node() {
        // Alice→Alice is 0 hops
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let mut input_row = Row::new();
        input_row.insert("a".into(), Value::Int(1));
        input_row.insert("b".into(), Value::Int(1));

        let sp = ShortestPathParams {
            source: "a",
            target: "b",
            edge_types: &["KNOWS".into()],
            direction: Direction::Outgoing,
            max_depth: 10,
            path_variable: "p",
        };

        let results = execute_shortest_path(&[input_row], &sp, &mut ctx).expect("sp");
        assert_eq!(results[0].get("p"), Some(&Value::Int(0)));
    }

    // -- Aggregation: DISTINCT --

    #[test]
    fn aggregate_count_distinct() {
        // Create rows with duplicate age values
        let (_dir, engine, mut interner) = setup_test_graph();

        // Add another User with age=30 (same as Alice) — before ctx borrow
        insert_node(
            &engine,
            1,
            10,
            "User",
            &[
                ("name", Value::String("Frank".into())),
                ("age", Value::Int(30)),
            ],
            &mut interner,
        );

        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // count(DISTINCT n.age) on all Users should be 3 (30, 25, 35) not 4
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![
                    AggregateItem {
                        function: "count".into(),
                        arg: Expr::PropertyAccess {
                            expr: Box::new(Expr::Variable("n".into())),
                            property: "age".into(),
                        },
                        distinct: true,
                        alias: Some("unique_ages".into()),
                    },
                    AggregateItem {
                        function: "count".into(),
                        arg: Expr::PropertyAccess {
                            expr: Box::new(Expr::Variable("n".into())),
                            property: "age".into(),
                        },
                        distinct: false,
                        alias: Some("total_ages".into()),
                    },
                ],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("unique_ages"), Some(&Value::Int(3))); // 30, 25, 35
        assert_eq!(result[0].get("total_ages"), Some(&Value::Int(4))); // 30, 25, 35, 30
    }

    #[test]
    fn aggregate_collect_distinct() {
        let (_dir, engine, mut interner) = setup_test_graph();

        // Add user with duplicate age — before ctx borrow
        insert_node(
            &engine,
            1,
            10,
            "User",
            &[
                ("name", Value::String("Frank".into())),
                ("age", Value::Int(30)),
            ],
            &mut interner,
        );

        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![AggregateItem {
                    function: "collect".into(),
                    arg: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    },
                    distinct: true,
                    alias: Some("ages".into()),
                }],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        if let Some(Value::Array(ages)) = result[0].get("ages") {
            assert_eq!(ages.len(), 3, "collect(DISTINCT) should have 3 unique ages");
        } else {
            panic!("expected Array for collect(DISTINCT)");
        }
    }

    #[test]
    fn aggregate_sum_int_preserves_type() {
        // sum() on all-Int values should return Int, not Float
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![AggregateItem {
                    function: "sum".into(),
                    arg: Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "age".into(),
                    },
                    distinct: false,
                    alias: Some("total".into()),
                }],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        // All ages are Int, so sum should be Int
        assert!(matches!(result[0].get("total"), Some(Value::Int(_))));
    }

    #[test]
    fn aggregate_group_by_with_null() {
        // GROUP BY should create separate group for NULL values
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        insert_node(
            &engine,
            1,
            1,
            "Item",
            &[("category", Value::String("A".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "Item",
            &[("category", Value::String("A".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            3,
            "Item",
            &[("category", Value::String("B".into()))],
            &mut interner,
        );
        insert_node(&engine, 1, 4, "Item", &[], &mut interner); // No category → NULL group

        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["Item".into()],
                    property_filters: vec![],
                }),
                group_by: vec![Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "category".into(),
                }],
                aggregates: vec![AggregateItem {
                    function: "count".into(),
                    arg: Expr::Star,
                    distinct: false,
                    alias: Some("cnt".into()),
                }],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        // Should have 3 groups: A (2 items), B (1 item), NULL (1 item)
        assert_eq!(result.len(), 3, "should have 3 groups (A, B, NULL)");
    }

    #[test]
    fn aggregate_empty_input() {
        // Aggregation on empty input: count=0, sum/avg=NULL
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Aggregate {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["NonExistent".into()],
                    property_filters: vec![],
                }),
                group_by: vec![],
                aggregates: vec![
                    AggregateItem {
                        function: "count".into(),
                        arg: Expr::Star,
                        distinct: false,
                        alias: Some("cnt".into()),
                    },
                    AggregateItem {
                        function: "sum".into(),
                        arg: Expr::Literal(Value::Int(1)),
                        distinct: false,
                        alias: Some("total".into()),
                    },
                    AggregateItem {
                        function: "avg".into(),
                        arg: Expr::Literal(Value::Int(1)),
                        distinct: false,
                        alias: Some("mean".into()),
                    },
                ],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1, "aggregate on empty → 1 row with defaults");
        assert_eq!(result[0].get("cnt"), Some(&Value::Int(0)));
        assert_eq!(result[0].get("total"), Some(&Value::Null));
        assert_eq!(result[0].get("mean"), Some(&Value::Null));
    }

    #[test]
    fn aggregate_pipeline_with_then_group_by() {
        // Test multi-stage: scan → aggregate → project (WITH) → uses alias
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // Simulate: WITH count(*) AS cnt RETURN cnt
        // This is Aggregate → Project → Project
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Project {
                input: Box::new(LogicalOp::Project {
                    input: Box::new(LogicalOp::Aggregate {
                        input: Box::new(LogicalOp::NodeScan {
                            variable: "n".into(),
                            labels: vec!["User".into()],
                            property_filters: vec![],
                        }),
                        group_by: vec![],
                        aggregates: vec![AggregateItem {
                            function: "count".into(),
                            arg: Expr::Star,
                            distinct: false,
                            alias: Some("cnt".into()),
                        }],
                    }),
                    items: vec![ProjectItem {
                        expr: Expr::Variable("cnt".into()),
                        alias: Some("cnt".into()),
                    }],
                    distinct: false,
                }),
                items: vec![ProjectItem {
                    expr: Expr::Variable("cnt".into()),
                    alias: Some("total".into()),
                }],
                distinct: false,
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("total"), Some(&Value::Int(3))); // 3 User nodes
    }

    // -- DETACH DELETE reverse posting list cleanup (G050) --

    #[test]
    fn detach_delete_cleans_reverse_posting_lists() {
        // Setup: Alice(1)->Bob(2) via KNOWS, Alice(1)->Charlie(3) via KNOWS,
        //        Bob(2)->Charlie(3) via KNOWS, Alice(1)->Matrix(4) via LIKES.
        let (_dir, engine, mut interner) = setup_test_graph();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // Verify Bob's incoming KNOWS posting list contains Alice(1) before delete
        let bob_in_key = encode_adj_key_reverse("KNOWS", NodeId::from_raw(2));
        let plist = ctx.adj_get(&bob_in_key).expect("read").expect("non-empty");
        assert!(
            plist.contains(1),
            "Bob's incoming KNOWS should contain Alice(1)"
        );

        // Verify Charlie's incoming KNOWS posting list contains Alice(1) and Bob(2)
        let charlie_in_key = encode_adj_key_reverse("KNOWS", NodeId::from_raw(3));
        let plist = ctx
            .adj_get(&charlie_in_key)
            .expect("read")
            .expect("non-empty");
        assert!(
            plist.contains(1),
            "Charlie's incoming KNOWS should contain Alice(1)"
        );
        assert!(
            plist.contains(2),
            "Charlie's incoming KNOWS should contain Bob(2)"
        );

        // Verify Matrix's incoming LIKES posting list contains Alice(1)
        let matrix_in_key = encode_adj_key_reverse("LIKES", NodeId::from_raw(4));
        let plist = ctx
            .adj_get(&matrix_in_key)
            .expect("read")
            .expect("non-empty");
        assert!(
            plist.contains(1),
            "Matrix's incoming LIKES should contain Alice(1)"
        );

        // DETACH DELETE Alice (node 1)
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Delete {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                variables: vec!["n".into()],
                detach: true,
            },
        };

        execute(&plan, &mut ctx).expect("execute DETACH DELETE");

        // Flush pending merge removes to storage
        // merge removes already flushed by execute() → mvcc_flush()

        // Verify Alice's node is gone
        let alice_key = encode_node_key(1, NodeId::from_raw(1));
        assert!(engine
            .get(Partition::Node, &alice_key)
            .expect("get")
            .is_none());

        // Verify Alice's own adj: keys are gone
        let alice_out_knows = encode_adj_key_forward("KNOWS", NodeId::from_raw(1));
        assert!(engine
            .get(Partition::Adj, &alice_out_knows)
            .expect("get")
            .is_none());

        // KEY CHECK: Bob's incoming KNOWS should NO LONGER contain Alice(1)
        let bob_in_plist = match engine.get(Partition::Adj, &bob_in_key).expect("get") {
            Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
            None => PostingList::new(),
        };
        assert!(
            !bob_in_plist.contains(1),
            "Bob's incoming KNOWS must not contain Alice(1) after DETACH DELETE"
        );

        // KEY CHECK: Charlie's incoming KNOWS should NO LONGER contain Alice(1)
        // but should STILL contain Bob(2)
        let charlie_in_plist = match engine.get(Partition::Adj, &charlie_in_key).expect("get") {
            Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
            None => PostingList::new(),
        };
        assert!(
            !charlie_in_plist.contains(1),
            "Charlie's incoming KNOWS must not contain Alice(1) after DETACH DELETE"
        );
        assert!(
            charlie_in_plist.contains(2),
            "Charlie's incoming KNOWS must still contain Bob(2)"
        );

        // KEY CHECK: Matrix's incoming LIKES should NO LONGER contain Alice(1)
        let matrix_in_plist = match engine.get(Partition::Adj, &matrix_in_key).expect("get") {
            Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
            None => PostingList::new(),
        };
        assert!(
            !matrix_in_plist.contains(1),
            "Matrix's incoming LIKES must not contain Alice(1) after DETACH DELETE"
        );
    }

    #[test]
    fn detach_delete_cleans_incoming_edge_counterparts() {
        // Test: node with incoming edges only.
        // Bob(2)->Alice(1), Charlie(3)->Alice(1) via FOLLOWS.
        // DETACH DELETE Alice should remove Alice from Bob's and Charlie's outgoing FOLLOWS.
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        // Create nodes
        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("name", Value::String("Alice".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "User",
            &[("name", Value::String("Bob".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            3,
            "User",
            &[("name", Value::String("Charlie".into()))],
            &mut interner,
        );

        // Bob->Alice and Charlie->Alice via FOLLOWS
        insert_edge(&engine, "FOLLOWS", 2, 1);
        insert_edge(&engine, "FOLLOWS", 3, 1);

        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(100));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // DETACH DELETE Alice
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Delete {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                variables: vec!["n".into()],
                detach: true,
            },
        };

        execute(&plan, &mut ctx).expect("execute DETACH DELETE");
        // merge removes already flushed by execute() → mvcc_flush()

        // Bob's outgoing FOLLOWS should not contain Alice(1)
        let bob_out_key = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(2));
        let bob_plist = match engine.get(Partition::Adj, &bob_out_key).expect("get") {
            Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
            None => PostingList::new(),
        };
        assert!(
            !bob_plist.contains(1),
            "Bob's outgoing FOLLOWS must not contain Alice(1)"
        );

        // Charlie's outgoing FOLLOWS should not contain Alice(1)
        let charlie_out_key = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(3));
        let charlie_plist = match engine.get(Partition::Adj, &charlie_out_key).expect("get") {
            Some(bytes) => PostingList::from_bytes(&bytes).expect("decode"),
            None => PostingList::new(),
        };
        assert!(
            !charlie_plist.contains(1),
            "Charlie's outgoing FOLLOWS must not contain Alice(1)"
        );
    }

    // ── list_edge_types unit tests ──────────────────────────────────────────

    /// Empty schema → empty edge type list.
    #[test]
    fn list_edge_types_empty_schema() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
        let ctx = make_ctx(&engine, &mut interner, &allocator);

        let types = ctx.list_edge_types().expect("list_edge_types");
        assert!(types.is_empty(), "no edge types in empty schema");
    }

    /// Registering edge types via insert_edge makes list_edge_types return them.
    #[test]
    fn list_edge_types_returns_registered() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());

        insert_edge(&engine, "FOLLOWS", 1, 2);
        insert_edge(&engine, "LIKES", 1, 3);
        // Duplicate registration for FOLLOWS is idempotent.
        insert_edge(&engine, "FOLLOWS", 2, 3);

        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
        let ctx = make_ctx(&engine, &mut interner, &allocator);

        let mut types = ctx.list_edge_types().expect("list_edge_types");
        types.sort();
        assert_eq!(types, vec!["FOLLOWS", "LIKES"]);
    }

    /// Edge type registered in mvcc_write_buffer (same-tx CREATE edge) is visible.
    #[test]
    fn list_edge_types_includes_write_buffer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // Write edge type directly into write buffer (simulates execute_create_edge).
        let et_key = edge_type_schema_key("WORKS_AT");
        ctx.mvcc_write_buffer
            .insert((Partition::Schema, et_key), Some(b"".to_vec()));

        let types = ctx.list_edge_types().expect("list_edge_types");
        assert!(
            types.contains(&"WORKS_AT".to_string()),
            "write-buffer edge type must be visible"
        );
    }

    /// DETACH DELETE with two edge types uses targeted lookup: cleans up both.
    #[test]
    fn detach_delete_multi_edge_type_targeted_lookup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let mut interner = FieldInterner::new();

        // Node 1 (Alice): has FOLLOWS and LIKES edges
        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("name", Value::String("Alice".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            2,
            "User",
            &[("name", Value::String("Bob".into()))],
            &mut interner,
        );
        insert_node(
            &engine,
            1,
            3,
            "User",
            &[("name", Value::String("Carol".into()))],
            &mut interner,
        );

        // Alice -[FOLLOWS]-> Bob
        insert_edge(&engine, "FOLLOWS", 1, 2);
        // Alice -[LIKES]-> Carol
        insert_edge(&engine, "LIKES", 1, 3);
        // Bob -[FOLLOWS]-> Carol  (should not be affected)
        insert_edge(&engine, "FOLLOWS", 2, 3);

        let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
        let mut ctx = make_ctx(&engine, &mut interner, &allocator);

        // DETACH DELETE Alice (node_id = 1)
        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Delete {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "n".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Alice".into())),
                    )],
                }),
                variables: vec!["n".into()],
                detach: true,
            },
        };

        execute(&plan, &mut ctx).expect("DETACH DELETE Alice");

        // Bob's FOLLOWS posting list must not contain Alice(1).
        let bob_follows_out = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(2));
        let plist = engine.get(Partition::Adj, &bob_follows_out).expect("get");
        if let Some(bytes) = plist {
            let pl = PostingList::from_bytes(&bytes).expect("decode");
            assert!(
                !pl.contains(1),
                "Bob's FOLLOWS must not contain Alice after delete"
            );
        }

        // Carol's LIKES incoming posting list must not contain Alice(1).
        let carol_likes_in = encode_adj_key_reverse("LIKES", NodeId::from_raw(3));
        let plist = engine.get(Partition::Adj, &carol_likes_in).expect("get");
        if let Some(bytes) = plist {
            let pl = PostingList::from_bytes(&bytes).expect("decode");
            assert!(
                !pl.contains(1),
                "Carol's LIKES-in must not contain Alice after delete"
            );
        }

        // Bob-[FOLLOWS]->Carol edge must be untouched.
        let bob_carol = encode_adj_key_forward("FOLLOWS", NodeId::from_raw(2));
        let plist = engine.get(Partition::Adj, &bob_carol).expect("get");
        if let Some(bytes) = plist {
            let pl = PostingList::from_bytes(&bytes).expect("decode");
            assert!(pl.contains(3), "Bob's FOLLOWS must still contain Carol(3)");
        }
    }

    // ── Correlated OPTIONAL MATCH detection (G004) ─────────────────────

    #[test]
    fn needs_correlated_non_correlated() {
        // OPTIONAL MATCH (a)-[:KNOWS]->(b) — right side introduces a, b.
        // Filter predicate only references b.age (right-side variable).
        // → NOT correlated.
        let right = LogicalOp::Filter {
            input: Box::new(LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["Person".into()],
                    property_filters: vec![],
                }),
                source: "a".into(),
                edge_types: vec!["KNOWS".into()],
                target_variable: "b".into(),
                target_labels: vec![],
                edge_variable: None,
                direction: Direction::Outgoing,
                length: None,
                target_filters: vec![],
                edge_filters: vec![],
            }),
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("b".into())),
                    property: "age".into(),
                }),
                op: BinaryOperator::Gt,
                right: Box::new(Expr::Literal(Value::Int(30))),
            },
        };
        assert!(
            !super::needs_correlated_execution(&right),
            "b.age > 30 references only right-side variable b"
        );
    }

    #[test]
    fn needs_correlated_yes_cross_scope() {
        // OPTIONAL MATCH (b:Person) WHERE b.age > a.age
        // Right side introduces "b". Predicate references "a" (not introduced).
        // → IS correlated.
        let right = LogicalOp::Filter {
            input: Box::new(LogicalOp::NodeScan {
                variable: "b".into(),
                labels: vec!["Person".into()],
                property_filters: vec![],
            }),
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("b".into())),
                    property: "age".into(),
                }),
                op: BinaryOperator::Gt,
                right: Box::new(Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("a".into())),
                    property: "age".into(),
                }),
            },
        };
        assert!(
            super::needs_correlated_execution(&right),
            "a.age references left-scope variable a"
        );
    }

    #[test]
    fn needs_correlated_no_filter() {
        // OPTIONAL MATCH (a)-[:KNOWS]->(b) — no filter at all.
        // → NOT correlated.
        let right = LogicalOp::Traverse {
            input: Box::new(LogicalOp::NodeScan {
                variable: "a".into(),
                labels: vec!["Person".into()],
                property_filters: vec![],
            }),
            source: "a".into(),
            edge_types: vec!["KNOWS".into()],
            target_variable: "b".into(),
            target_labels: vec![],
            edge_variable: None,
            direction: Direction::Outgoing,
            length: None,
            target_filters: vec![],
            edge_filters: vec![],
        };
        assert!(
            !super::needs_correlated_execution(&right),
            "no filter = no correlation"
        );
    }

    #[test]
    fn collect_expr_vars_covers_in_and_is_null() {
        // Verify collect_expr_vars extracts variables from In and IsNull.
        let expr = Expr::In {
            expr: Box::new(Expr::Variable("x".into())),
            list: Box::new(Expr::List(vec![Expr::Variable("y".into())])),
        };
        let mut vars = Vec::new();
        super::collect_expr_vars(&expr, &mut vars);
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));

        let expr2 = Expr::IsNull {
            expr: Box::new(Expr::Variable("z".into())),
            negated: false,
        };
        let mut vars2 = Vec::new();
        super::collect_expr_vars(&expr2, &mut vars2);
        assert!(vars2.contains(&"z".to_string()));
    }

    #[test]
    fn collect_expr_vars_covers_string_match() {
        let expr = Expr::StringMatch {
            expr: Box::new(Expr::Variable("a".into())),
            op: crate::cypher::ast::StringOp::StartsWith,
            pattern: Box::new(Expr::Variable("b".into())),
        };
        let mut vars = Vec::new();
        super::collect_expr_vars(&expr, &mut vars);
        assert!(vars.contains(&"a".to_string()));
        assert!(vars.contains(&"b".to_string()));
    }

    // -- G067: Parallel path OCC read-set tracking --

    #[test]
    fn g067_parallel_traversal_populates_occ_read_set() {
        // Verify that parallel traversal collects read keys into mvcc_read_set
        // so OCC conflict detection works for write transactions on super-nodes.
        let dir = tempfile::tempdir().expect("tempdir");
        let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let engine = StorageEngine::open_with_oracle(
            &coordinode_storage::engine::config::StorageConfig::new(dir.path()),
            oracle.clone(),
        )
        .expect("open with oracle");
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::new();

        // Hub node with 10 targets (threshold=5 triggers parallel)
        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("name", Value::String("Hub".into()))],
            &mut interner,
        );
        for i in 2..=11u64 {
            insert_node(
                &engine,
                1,
                i,
                "User",
                &[("name", Value::String(format!("T{i}")))],
                &mut interner,
            );
            insert_edge(&engine, "FOLLOWS", 1, i);
        }

        // Allocate a read timestamp and take a snapshot
        let read_ts = oracle.next();
        let snap = engine.snapshot_at(read_ts.as_raw());

        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        ctx.mvcc_oracle = Some(&oracle);
        ctx.mvcc_read_ts = read_ts;
        ctx.mvcc_snapshot = snap;
        ctx.adaptive.parallel_threshold = 5; // trigger parallel on 10 edges

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Hub".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["FOLLOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: None,
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 10, "should return all 10 targets");

        // Verify parallel was triggered
        assert!(
            ctx.warnings.iter().any(|w| w.contains("parallel")),
            "expected parallel activation, got: {:?}",
            ctx.warnings,
        );

        // Verify OCC read-set contains Node keys for target nodes.
        // The source node (Hub, id=1) is read via sequential mvcc_get in NodeScan,
        // and 10 target nodes are read via parallel path — all should be tracked.
        let node_read_keys: Vec<_> = ctx
            .mvcc_read_set
            .iter()
            .filter(|(part, _)| *part == Partition::Node)
            .collect();

        // At least 10 target Node keys must be in read-set (from parallel path)
        // plus 1 for the Hub node (from sequential NodeScan)
        assert!(
            node_read_keys.len() >= 10,
            "expected ≥10 Node keys in OCC read-set (parallel targets), got {}",
            node_read_keys.len(),
        );

        // Verify specific target keys are tracked
        for target_id in 2..=11u64 {
            let target_key = encode_node_key(1, NodeId::from_raw(target_id));
            assert!(
                ctx.mvcc_read_set
                    .contains(&(Partition::Node, target_key.to_vec())),
                "target node {target_id} should be in OCC read-set",
            );
        }
    }

    #[test]
    fn g067_parallel_occ_detects_conflict_on_target_node() {
        // End-to-end: parallel traversal reads target nodes, concurrent write
        // modifies one target, OCC conflict detection catches it.
        let dir = tempfile::tempdir().expect("tempdir");
        let oracle = std::sync::Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let engine = StorageEngine::open_with_oracle(
            &coordinode_storage::engine::config::StorageConfig::new(dir.path()),
            oracle.clone(),
        )
        .expect("open with oracle");
        let mut interner = FieldInterner::new();
        let allocator = NodeIdAllocator::new();

        // Hub + 10 targets
        insert_node(
            &engine,
            1,
            1,
            "User",
            &[("name", Value::String("Hub".into()))],
            &mut interner,
        );
        for i in 2..=11u64 {
            insert_node(
                &engine,
                1,
                i,
                "User",
                &[("name", Value::String(format!("T{i}")))],
                &mut interner,
            );
            insert_edge(&engine, "FOLLOWS", 1, i);
        }

        // T1: Start read transaction
        let read_ts = oracle.next();
        let snap = engine.snapshot_at(read_ts.as_raw());

        let mut ctx = make_ctx(&engine, &mut interner, &allocator);
        ctx.mvcc_oracle = Some(&oracle);
        ctx.mvcc_read_ts = read_ts;
        ctx.mvcc_snapshot = snap;
        ctx.adaptive.parallel_threshold = 5;

        let plan = LogicalPlan {
            snapshot_ts: None,
            vector_consistency: VectorConsistencyMode::default(),
            root: LogicalOp::Traverse {
                input: Box::new(LogicalOp::NodeScan {
                    variable: "a".into(),
                    labels: vec!["User".into()],
                    property_filters: vec![(
                        "name".into(),
                        Expr::Literal(Value::String("Hub".into())),
                    )],
                }),
                source: "a".into(),
                edge_types: vec!["FOLLOWS".into()],
                direction: Direction::Outgoing,
                target_variable: "b".into(),
                target_labels: vec![],
                length: None,
                edge_variable: None,
                target_filters: vec![],
                edge_filters: vec![],
            },
        };

        let result = execute(&plan, &mut ctx).expect("execute");
        assert_eq!(result.len(), 10);

        // T2: Concurrent transaction modifies target node 5 (after T1's read_ts)
        let _write_ts = oracle.next();
        let mut modified_record = NodeRecord::new("User");
        // Use field_id 0 directly — "name" was interned first by insert_node,
        // so it has id 0. Avoids borrowing interner while ctx is alive.
        modified_record.set(0, Value::String("T5-modified".into()));
        let target5_key = encode_node_key(1, NodeId::from_raw(5));
        engine
            .put(
                Partition::Node,
                &target5_key,
                &modified_record.to_msgpack().expect("serialize"),
            )
            .expect("concurrent write");

        // T1: Add a dummy write so mvcc_flush doesn't skip conflict check
        // (read-only transactions return early without OCC check)
        let dummy_key = encode_node_key(1, NodeId::from_raw(999));
        ctx.mvcc_write_buffer
            .insert((Partition::Node, dummy_key), Some(b"dummy".to_vec()));

        // T1: OCC conflict check via mvcc_flush should detect the write to target 5
        let conflict = ctx.mvcc_flush();
        assert!(
            conflict.is_err(),
            "OCC should detect conflict on target node 5 modified after read_ts",
        );
        let err_msg = format!("{}", conflict.unwrap_err());
        assert!(
            err_msg.contains("OCC conflict"),
            "expected OCC conflict error, got: {err_msg}",
        );
    }
}
