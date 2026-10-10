//! CoordiNode LSM storage engine — the core KV layer for CoordiNode.
//!
//! Each logical partition maps to an `lsm_tree::AnyTree` (Tree or BlobTree)
//! opened with a shared `SharedSequenceNumberGenerator`. All trees share one
//! `lsm_tree::Cache` instance and the same seqno counter, ensuring
//! cross-partition monotonic ordering for MVCC reads.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicU64};
use std::sync::{Arc, Mutex};

use coordinode_core::txn::proposal::Mutation;
use lsm_tree::{AbstractTree, Guard};
use tracing::info;

use super::{MAX_DERIVED_EFFECTS, SeekableStorageIter, StorageIter};
use crate::cache::access::AccessTracker;
use crate::cache::tiered::TieredCache;
use crate::engine::batch::WriteBatch;
use crate::engine::compaction::CompactionScheduler;
use crate::engine::config::EndpointConfig;
use crate::engine::config::{FlushPolicy, StorageConfig};
use crate::engine::coordinator::{LocalMultiModalCoordinator, MultiModalCoordinator, SnapshotPin};
use crate::engine::coverage::{self, Coverage, Domain, TreeCoverage};
use crate::engine::flush::FlushManager;
use crate::engine::oplog_journal::{
    EmbeddedOplog, OplogJournalConfig, apply_oplog_ops_at, op_partition,
};
use crate::engine::partition::Partition;
use crate::engine::routing::PartitionRouting;
use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::{OplogEntry, OplogOp};

/// Newtype wrapper that bridges coordinode-core's `TimestampOracle` to
/// lsm-tree's `SequenceNumberGenerator` trait. Makes every write's LSM
/// seqno equal to the HLC timestamp for native MVCC.
#[derive(Debug)]
pub struct OracleSeqnoGenerator(
    pub std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>,
);

/// A retention window in seqno units. Seqnos are HLC microseconds, so the
/// conversion is exact up to `u64::MAX` µs (~584,942 years); anything larger
/// saturates to "retain forever", the only sensible reading of such a window.
fn retention_window_to_us(window: std::time::Duration) -> u64 {
    u64::try_from(window.as_micros()).unwrap_or(u64::MAX)
}

impl lsm_tree::SequenceNumberGenerator for OracleSeqnoGenerator {
    fn next(&self) -> lsm_tree::SeqNo {
        self.0.next().as_raw()
    }

    fn get(&self) -> lsm_tree::SeqNo {
        // TimestampOracle::next() returns the NEW value (post-increment),
        // so current() == last_written_seqno. The LSM seqno_filter uses
        // strict `<`, meaning reads at snapshot S see seqnos < S. To include
        // the last write in reads, we must return current() + 1 — matching
        // SequenceNumberCounter::get() semantics where get() is always
        // strictly greater than the last value returned by next().
        self.0.current().as_raw() + 1
    }

    fn set(&self, value: lsm_tree::SeqNo) {
        self.0
            .advance_to(coordinode_core::txn::timestamp::Timestamp::from_raw(value));
    }

    fn fetch_max(&self, value: lsm_tree::SeqNo) {
        self.0
            .advance_to(coordinode_core::txn::timestamp::Timestamp::from_raw(value));
    }
}

/// The CoordiNode storage engine.
///
/// Owns 8 `lsm_tree::AnyTree` instances (one per [`Partition`]), a shared
/// `Cache`, and a shared `SequenceNumberGenerator`. All mutations go directly
/// to the appropriate tree; there is no intermediate transaction layer.
///
/// For atomic multi-partition writes use [`WriteBatch`]. Crash safety is
/// provided by the LSM WAL (each tree has its own WAL segment).
///
/// # Drop ordering
///
/// Fields are dropped in declaration order:
/// 1. `flush_manager` — stops flush workers before trees are touched
/// 2. `compaction_scheduler` — stops compaction workers before trees drop
/// 3. `trees` — AnyTree handles released last
pub struct StorageEngine {
    /// Background flush worker pool. Dropped first (before trees) to ensure
    /// worker threads are joined before the tree Arc refs are released.
    flush_manager: Option<FlushManager>,
    /// Background compaction worker pool. Dropped second (after flush, before trees).
    compaction_scheduler: Option<CompactionScheduler>,
    /// Layer-3 multi-partition coordinator: owns the partition tree map,
    /// shared seqno generator, block cache, and MVCC GC watermark. Every
    /// partition-keyed read/write delegates here. See
    /// [`crate::engine::coordinator`].
    coordinator: LocalMultiModalCoordinator,
    /// Invariant claims of the attempts running against this engine. Shared
    /// by every transaction, because a condition is only protected if the
    /// attempt that would break it is looking at the same table.
    claim_registry: crate::engine::claims::ClaimRegistry,
    /// Counts the schema changes this process has applied. A claim carries the
    /// value it read, so a predicate evaluated before a definition changed is
    /// not taken as evidence about the graph after it.
    ///
    /// Process-local on purpose: it is only ever compared between attempts
    /// running at the same time, and no attempt outlives the process. Giving
    /// it durable identity would buy nothing and cost a write on every DDL.
    schema_generation: AtomicU64,
    /// Per node key, the property deltas written since its last whole write:
    /// a read of a key with pending deltas folds every one of them, so a
    /// writer bounds the run by writing the record whole once it is long
    /// enough (the reason RocksDB has `max_successive_merges`). A hint for
    /// the choice of write form only, so process-local and bounded: a lost
    /// or reset count costs one early whole write.
    node_delta_runs: parking_lot::Mutex<rustc_hash::FxHashMap<Vec<u8>, u32>>,
    /// Held while a metadata command is decided and applied, so each decision
    /// reads every earlier one; on the journalled path it also spans the
    /// journal append, which makes the decisions' order the journal's order.
    /// Field dictionary reads and snapshot installs take it too, so a reader
    /// never sees a binding's two records half written.
    metadata_decisions: parking_lot::Mutex<()>,
    /// Counts changes to the field dictionary this process has seen applied
    /// or installed. A view of the dictionary taken at one value covers every
    /// binding applied before it, so a reader compares it once per query.
    field_dictionary_generation: AtomicU64,
    /// Counts the dictionary changes that can bind ids below the frontier
    /// (adopted bindings, installed snapshots): a view extended only above
    /// its frontier stays correct while this holds still.
    field_dictionary_epoch: AtomicU64,
    /// Commits admitted here and not yet applied. Validation reads committed
    /// state, which is exactly what these are not part of yet, so they are
    /// consulted beside it.
    pending_commits: Arc<crate::engine::pending::PendingCommits>,
    /// The closed bound of the Raft log applied here: every commit timestamp
    /// below it is in an entry applied on this node. `0` while no entry
    /// carrying one has applied. Mirrors [`CLOSURE_KEY`].
    closure_frontier: AtomicU64,
    /// How long a snapshot waits for those commits before stepping behind
    /// them. Runtime-settable: it trades read latency against freshness, and
    /// which side a deployment wants is not known at compile time.
    snapshot_wait: std::sync::atomic::AtomicU64,
    /// The shard whose node rows this engine holds, for the one lookup inside
    /// the engine that starts from a node id rather than a key. Settable at
    /// runtime because the layer that knows it is built after the engine.
    node_shard: std::sync::atomic::AtomicU16,
    flush_policy: FlushPolicy,
    /// The operator's oplog settings, shared by the embedded journal and the
    /// Raft log opened over this engine.
    oplog_config: OplogJournalConfig,
    /// Optional tiered block cache (DRAM → NVMe → SSD cascade).
    tiered_cache: Option<TieredCache>,
    /// Per-key access tracker for cache eviction and heat map.
    access_tracker: AccessTracker,
    /// Root data directory — exposed so subsystems (e.g. Raft oplog) can
    /// derive their own sub-directories without re-reading the config.
    data_dir: PathBuf,
    /// Configured endpoints — cloned from `StorageConfig.endpoints` at
    /// open time so subsystems (WAL/oplog placement, tier routing,
    /// hard-limit enforcement) can resolve target endpoints without
    /// retaining a reference to the original config.
    endpoints: Vec<crate::engine::config::EndpointConfig>,
    /// Per-endpoint capacity tracker — atomic `used_bytes` snapshots,
    /// hard-limit thresholds, `is_writable` gating flag. Populated at
    /// engine open from the endpoint config; refreshed by the
    /// background scanner.
    capacity: Arc<crate::engine::capacity::CapacityTracker>,
    /// Background scanner that periodically re-runs
    /// `refresh_capacity()`. `None` only when capacity tracking is
    /// disabled (every endpoint has `hard_limit_bytes == 0`) or when
    /// the engine is in an in-memory test mode that opts out — the
    /// default path always spawns it. Drop order: scanner is dropped
    /// before `trees` (declared earlier in the struct) so the thread
    /// stops accessing tree handles before they are released.
    capacity_scanner: Option<crate::engine::capacity::CapacityScanner>,
    /// Per-partition resolved L0 endpoint id (the endpoint that
    /// receives newly flushed SSTs for this partition). Cached at
    /// engine open so the pre-write capacity gate is a single
    /// HashMap lookup on the hot path. Schema partition is mapped to
    /// the primary endpoint id (single-tier bootstrap).
    partition_l0_endpoint: HashMap<Partition, String>,
    /// Optional embedded oplog journal (oracle-backed, no-Raft mode).
    ///
    /// `Some` when opened via [`StorageEngine::open_with_oracle`] against a
    /// persistent endpoint. The retained oplog drives crash recovery and
    /// WAL-replay-repair (rebuild a corrupt partition from a checkpoint then
    /// replay forward). In cluster mode this is `None` — the Raft log is the
    /// equivalent retained oplog.
    oplog: Option<Arc<Mutex<EmbeddedOplog>>>,
    /// Which journal indices are applied, and the last fold of them into the
    /// partition trees' coverage records. `Some` exactly when `oplog` is.
    coverage: Option<Coverage>,
    /// The timestamp oracle this engine stamps writes with. `Some` only
    /// when opened via [`StorageEngine::open_with_oracle`]. Exposed via
    /// [`StorageEngine::oracle`] so subsystems applying externally
    /// stamped writes (the Raft state machine on followers) can advance
    /// the SAME oracle that local MVCC readers draw snapshots from;
    /// without that, follower reads pin a stale snapshot and observe
    /// none of the replicated data.
    oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
    /// `STORAGE COLUMNAR` table trees, keyed by table id. Each columnar table
    /// owns one columnar-mode tree (the whole-tree columnar layout cannot share
    /// the row-mode partition trees); see [`crate::columnar`]. Shares this
    /// engine's seqno generator and block cache. Re-opened from disk at engine
    /// open so a restart recovers the tables.
    #[cfg(feature = "columnar")]
    columnar_tables: crate::columnar::ColumnarTableRegistry,
    /// Partitions whose tree failed to open with positively-identified
    /// structural damage and was repaired (manifest rebuild, block salvage)
    /// during THIS open. Consumed by the repair orchestrator: a lossy
    /// [`replay_scope`](OpenRepair::replay_scope) means the partition owes a
    /// checkpoint + oplog rebuild even though its tree now opens cleanly.
    open_repairs: Vec<OpenRepair>,
    /// Cached write-pressure tier (0/1/2 = None/Slowdown/Stop), latched by
    /// the compaction monitor each poll cycle so the commit-path check is a
    /// single relaxed atomic load.
    write_pressure: Arc<AtomicU8>,
    /// Pauses this node's Raft applies, when a Raft state machine runs over
    /// the engine; read by whatever replaces or copies a partition at an
    /// exact log position. Written once per state machine open.
    raft_fence: parking_lot::RwLock<Option<Arc<dyn raft_coverage::RaftApplyFence>>>,
    /// The lowest Raft log index the latest local checkpoint's trees may
    /// lack, which a partition rebuild from that checkpoint replays from;
    /// the log keeps it. `u64::MAX` when no checkpoint needs any.
    raft_log_keep_from: AtomicU64,
    /// Partition captures taken for copies to other nodes, naming each
    /// one's directory.
    partition_captures: AtomicU64,
    /// Consumers watching the writes a partition receives (see
    /// [`Self::tap_writes`]).
    write_taps: Arc<crate::engine::tap::WriteTaps>,
    /// Consumers following the Raft entries as they apply (see
    /// [`Self::subscribe_applied`]).
    applied_feed: Arc<crate::engine::applied::AppliedFeed>,
    /// How many applied commits a derived-index worker's subscription holds
    /// ([`StorageConfig::index_feed_capacity`]).
    index_feed_capacity: usize,
    /// The transactions open here (see [`Self::await_transactions_through`]).
    open_transactions: Arc<crate::engine::open_txns::OpenTransactions>,
    /// Refuses new writes while the disk under a durable endpoint is below
    /// its free-space reserve; shared with the consensus layer, which checks
    /// it before anything reaches its log.
    space: Arc<crate::engine::space::SpaceGuard>,
}

/// An inclusive `[min, max]` user-key range, as reported by a lossy open
/// repair ([`OpenRepair::scoped_ranges`]) and consumed by the range-scoped
/// rebuild ([`StorageEngine::repair_partition_ranges_from_checkpoint`]).
pub type KeyRange = (Vec<u8>, Vec<u8>);

/// Owned `(key, value)` rows, as a rebuild exports them from a checkpoint.
type Rows = Vec<(Vec<u8>, Vec<u8>)>;

/// One `STORAGE COLUMNAR` table as a whole-store snapshot carries it: its id
/// and its `(key, value)` rows in key order.
pub type ColumnarTable = (String, Vec<(Vec<u8>, Vec<u8>)>);

/// What an applied metadata command did to the field dictionary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DictionaryChange {
    /// Nothing.
    None,
    /// Bindings above the frontier were added.
    Extended,
    /// Bindings may have landed anywhere, below the frontier included.
    Replaced,
}

/// The engine's cached write-pressure verdict, refreshed by the compaction
/// monitor each poll cycle from the worst per-partition backpressure signal
/// (L0 height, pending-compaction byte debt).
///
/// Nothing in the engine sleeps on this: `Slowdown` raises compaction
/// priority and is exported to metrics, and `Stop` is consulted once per
/// commit ([`crate::engine::transaction::Transaction::commit`]) to reject
/// new client writes with a retryable error until compaction catches up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePressure {
    /// Every partition is within its target shape.
    None,
    /// Some partition crossed a slowdown threshold: compaction is being
    /// prioritised; writes still proceed at full rate.
    Slowdown,
    /// Some partition crossed a stop threshold: new client writes should be
    /// rejected (retryable) until the debt drains.
    Stop,
}

impl WritePressure {
    pub(crate) fn from_u8(tier: u8) -> Self {
        match tier {
            0 => Self::None,
            1 => Self::Slowdown,
            _ => Self::Stop,
        }
    }
}

/// One partition tree repaired during engine open.
///
/// Produced when the tree's `open` failed structurally (lost manifest
/// pointer, corrupt version log) and the block-salvaging repair rebuilt it
/// from the SSTs on disk. [`replay_scope`](Self::replay_scope) is the replay
/// obligation the repair left behind: `TailOnly` when every block survived,
/// lossy otherwise.
#[derive(Debug)]
pub struct OpenRepair {
    /// The partition whose tree was repaired.
    pub partition: Partition,
    /// How far back a retained-oplog replay must reach to restore what the
    /// repair could not recover.
    pub replay_scope: lsm_tree::WalReplayScope,
    /// The full repair outcome (recovered / salvaged / unreadable counts,
    /// lost key coverage) for logging and diagnostics.
    pub report: lsm_tree::RepairReport,
}

impl OpenRepair {
    /// `true` when the repair could not recover everything: block salvage
    /// dropped key ranges (or a table's coverage never parsed), so the
    /// partition owes a rebuild from a checkpoint plus oplog replay even
    /// though its tree now reads checksum-clean.
    #[must_use]
    pub fn is_lossy(&self) -> bool {
        !matches!(self.replay_scope, lsm_tree::WalReplayScope::TailOnly)
    }

    /// The inclusive `[min, max]` key ranges the repair lost, when every loss
    /// is key-scopable — the precondition for a range-scoped rebuild
    /// ([`StorageEngine::repair_partition_ranges_from_checkpoint`]) instead
    /// of a full one. `None` when any excluded table's coverage never parsed
    /// (`unknowable_losses`): no key bound can prove a retained row is
    /// unaffected, so only a full rebuild is sound. Also `None` when a loss
    /// reaches the partition's apply-coverage record: without it the open
    /// could not tell which journal entries the rest of the tree holds.
    #[must_use]
    pub fn scoped_ranges(&self) -> Option<Vec<KeyRange>> {
        if !self.report.unknowable_losses.is_empty() {
            return None;
        }
        if self
            .report
            .lost_coverage
            .iter()
            .any(|(_, min, _, _)| coverage::is_reserved(min))
        {
            return None;
        }
        Some(
            self.report
                .lost_coverage
                .iter()
                .map(|(_, min, max, _)| (min.to_vec(), max.to_vec()))
                .collect(),
        )
    }
}

impl StorageEngine {
    /// Open or create a CoordiNode storage engine at the configured path.
    ///
    /// Creates all 8 partition trees if they don't exist.
    pub fn open(config: &StorageConfig) -> StorageResult<Self> {
        let gc_watermark = Arc::new(AtomicU64::new(0));
        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(lsm_tree::SequenceNumberCounter::default());
        Self::finish_open(config, seqno, gc_watermark, None, None)
    }

    /// Open with a custom `TimestampOracle` as the seqno generator.
    ///
    /// Every write's LSM seqno equals the oracle's timestamp, enabling
    /// native MVCC via `snapshot_at(ts)`. The oracle must be shared with
    /// the transaction layer via `Arc`.
    pub fn open_with_oracle(
        config: &StorageConfig,
        oracle: std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>,
    ) -> StorageResult<Self> {
        let gc_watermark = Arc::new(AtomicU64::new(0));
        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(OracleSeqnoGenerator(Arc::clone(&oracle)));
        Self::finish_open(config, seqno, gc_watermark, Some(oracle), None)
    }

    /// Open an embedded (no-Raft) engine backed by the oracle plus a RETAINED
    /// oplog journal.
    ///
    /// Like [`open_with_oracle`](Self::open_with_oracle), every write's LSM
    /// seqno equals the oracle timestamp — but in addition every proposal is
    /// journalled to a retained oplog (at the oplog-eligible endpoint), so the
    /// engine survives a crash by replaying the un-flushed tail and can repair
    /// a corrupt partition from a checkpoint plus oplog replay
    /// ([`Self::repair_partition_from_checkpoint`]).
    ///
    /// In-memory configs (every endpoint `Volatile`) get no journal — there is
    /// no durable place to keep it; durability there is best-effort by design.
    pub fn open_embedded(
        config: &StorageConfig,
        oracle: std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>,
    ) -> StorageResult<Self> {
        let gc_watermark = Arc::new(AtomicU64::new(0));
        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(OracleSeqnoGenerator(Arc::clone(&oracle)));
        Self::finish_open(
            config,
            seqno,
            gc_watermark,
            Some(oracle),
            Some(OplogJournalConfig::from(config)),
        )
    }

    /// [`open_embedded`](Self::open_embedded) with an explicit oplog journal
    /// configuration (retention window, segment rotation thresholds).
    pub fn open_embedded_with_journal(
        config: &StorageConfig,
        oracle: std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>,
        journal: OplogJournalConfig,
    ) -> StorageResult<Self> {
        let gc_watermark = Arc::new(AtomicU64::new(0));
        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(OracleSeqnoGenerator(Arc::clone(&oracle)));
        Self::finish_open(config, seqno, gc_watermark, Some(oracle), Some(journal))
    }

    fn finish_open(
        config: &StorageConfig,
        seqno: lsm_tree::SharedSequenceNumberGenerator,
        gc_watermark: Arc<AtomicU64>,
        oracle: Option<std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
        journal_config: Option<OplogJournalConfig>,
    ) -> StorageResult<Self> {
        // Before anything reads the directory: a directory of another engine
        // format is migrated or refused here, never opened as it is.
        let format = crate::format::prepare(config)?;

        // Built here rather than inside the coordinator: the capacity scanner
        // is spawned below, before the coordinator exists, and its cascade
        // eviction compacts — so it needs to publish and read the same
        // watermark.
        // Shared with the watermark controller: a snapshot stops below the
        // oldest commit in flight, and the watermark must not pass it.
        let pending_commits = Arc::new(crate::engine::pending::PendingCommits::new(
            config.max_commits_in_flight,
        ));
        let gc_controller = LocalMultiModalCoordinator::build_gc_controller(
            Arc::clone(&gc_watermark),
            seqno.clone(),
            Arc::clone(&pending_commits),
            oracle.is_some(),
        );

        // When built with `--features io-uring` on Linux and no explicit
        // filesystem backend was configured, default every partition tree to a
        // single shared io_uring ring. Falls back to StdFs if the running
        // kernel lacks io_uring (pre-5.6 or restricted). An explicit
        // `StorageConfig::with_fs` always wins. On non-Linux targets or without
        // the feature this block does not exist and `config` is the argument
        // unchanged (byte-identical to a build without io-uring).
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        let _io_uring_backing;
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        let config = if config.fs.is_none() {
            match lsm_tree::fs::IoUringFs::new() {
                Ok(fs) => {
                    _io_uring_backing = config
                        .clone()
                        .with_fs(Arc::new(fs) as Arc<dyn lsm_tree::fs::Fs>);
                    &_io_uring_backing
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "io-uring feature enabled but io_uring is unavailable; using StdFs"
                    );
                    config
                }
            }
        } else {
            config
        };

        // Shared block cache across all partition trees.
        let cache = Arc::new(lsm_tree::Cache::with_capacity_bytes(
            config.block_cache_bytes,
        ));

        // Two-pass open (per-LSM-level endpoint routing):
        //
        // **Pass 1** — open the Schema partition with no per-level routing.
        // Schema holds the routing metadata for every other partition, so
        // it has to be reachable before we can decide where other partitions'
        // SSTs should land. Schema itself stays single-tier on the primary
        // endpoint by design; metadata is small and the bootstrap problem
        // (routing-needed-to-open-the-routing-store) does not arise.
        //
        // **Pass 2** — for each non-Schema partition, load the persisted
        // [`PartitionRouting`] from Schema or initialise (compute + persist)
        // on first open against this endpoint set. Open the partition tree
        // with `level_routes` derived from the routing.
        let mut trees = HashMap::with_capacity(Partition::all().len());
        let mut open_repairs: Vec<OpenRepair> = Vec::new();

        // Structural open failures (lost `current` pointer, corrupt version
        // log) are repaired in place with block salvage rather than failing
        // the whole engine open: recovery must always yield an openable
        // engine, with any loss REPORTED (via `open_repairs`) for the repair
        // orchestrator to replay, never a dead end. Config-level mistakes
        // (wrong comparator, missing dictionary) still propagate as errors.
        let repair_policy = lsm_tree::RepairPolicy::default().salvage(true);

        // Pass 1: Schema partition (single-tier, no routing).
        let schema_config = config
            .to_tree_config(Partition::Schema, Arc::clone(&seqno))
            .use_cache(Arc::clone(&cache));
        let (schema_tree, schema_repair) = schema_config.open_or_repair(repair_policy)?;
        if let Some(report) = schema_repair {
            tracing::warn!(
                partition = Partition::Schema.name(),
                recovered = report.recovered,
                salvaged = report.salvaged,
                scope = ?report.wal_replay_scope(),
                "partition tree structurally repaired at open"
            );
            open_repairs.push(OpenRepair {
                partition: Partition::Schema,
                replay_scope: report.wal_replay_scope(),
                report,
            });
        }
        trees.insert(Partition::Schema, schema_tree.clone());

        // Restore seqno from Schema BEFORE reading routing — the routing
        // get() needs a fresh seqno bound so it observes prior persisted
        // writes, and one above the tree's retention floor so it is served at
        // all. The two differ most here: schema rows are as old as the
        // database while the floor is as recent as the last compaction, so a
        // clock that restarts behind that compaction would otherwise read
        // below the floor and fail the open. Same seeding as the whole-engine
        // restore below, for the one tree read before it.
        {
            use lsm_tree::AbstractTree;
            if let Some(max) = schema_tree.get_highest_seqno() {
                seqno.fetch_max(max + 1);
            }
            if let Some(first_servable) = schema_tree.retention_floor().checked_add(1) {
                seqno.fetch_max(first_servable);
            }
        }

        // Pass 2: load/initialise routing for each non-Schema partition,
        // then open it with `level_routes` wired. Also cache each
        // partition's L0 endpoint id for the pre-write capacity gate.
        let primary_endpoint_id = config.endpoints[0].id.clone();
        let mut partition_l0_endpoint: HashMap<Partition, String> = HashMap::new();
        partition_l0_endpoint.insert(Partition::Schema, primary_endpoint_id.clone());
        for &part in Partition::all() {
            if part == Partition::Schema {
                continue;
            }
            let routing = load_or_init_partition_routing(
                &schema_tree,
                &seqno,
                &config.endpoints,
                part,
                config.relocated,
            )?;
            let l0_endpoint = routing
                .levels
                .get(&0)
                .cloned()
                .unwrap_or_else(|| primary_endpoint_id.clone());
            partition_l0_endpoint.insert(part, l0_endpoint);
            let tree_config = config
                .to_tree_config_with_routing(part, Arc::clone(&seqno), Some(&routing))
                .use_cache(Arc::clone(&cache));
            let (tree, repair) = tree_config.open_or_repair(repair_policy)?;
            if let Some(report) = repair {
                tracing::warn!(
                    partition = part.name(),
                    recovered = report.recovered,
                    salvaged = report.salvaged,
                    scope = ?report.wal_replay_scope(),
                    "partition tree structurally repaired at open"
                );
                open_repairs.push(OpenRepair {
                    partition: part,
                    replay_scope: report.wal_replay_scope(),
                    report,
                });
            }
            trees.insert(part, tree);
        }

        // ── Embedded oplog journal: open + crash recovery ────────────────────
        // Oracle-backed standalone engines journal every proposal to a RETAINED
        // oplog. Each partition tree records, in-band, which journal indices it
        // physically holds (`engine::coverage`); on open, entry i is replayed
        // into partition p iff p's record lacks i, at seqno = entry.ts, with
        // its marker. A commit's ts is its MVCC version, not its position on
        // disk: a late-finalized commit can sit in a memtable behind a flushed
        // newer one, so no seqno comparison can stand in for the record. The
        // replay precedes the seqno restore below so the restored watermark
        // covers the replayed entries.
        //
        // Columnar tables live outside the partition trees and their registry is
        // built further below, so columnar ops are collected here and replayed
        // in a second pass once the registry exists.
        #[cfg(feature = "columnar")]
        let mut columnar_replay: Vec<ColumnarReplay> = Vec::new();
        // `Some(next)` when a journal exists and its store carries no coverage
        // record yet (a fresh store): the base covering every index below
        // `next` is written once the seqno is restored.
        let mut establish_coverage: Option<u64> = None;
        let oplog = match journal_config {
            Some(jcfg) => match config.select_oplog_endpoint(0) {
                Ok(endpoint) => {
                    let dir = endpoint.path.join("oplog").join("0");
                    let mut journal = EmbeddedOplog::open(&dir, 0, &jcfg)?;
                    let entries = journal.read_all()?;
                    let mut covered: HashMap<Partition, TreeCoverage> =
                        HashMap::with_capacity(trees.len());
                    for (&part, tree) in &trees {
                        covered.insert(part, TreeCoverage::read(tree, Domain::Journal)?);
                    }
                    if !covered.values().any(TreeCoverage::has_record) {
                        if !entries.is_empty() {
                            return Err(StorageError::CoverageUnprovable {
                                path: dir.display().to_string(),
                                entries: entries.len(),
                            });
                        }
                        establish_coverage = Some(journal.next_index());
                    }
                    // An index a tree records as covered is never handed out
                    // again, even when the segments that held it are gone.
                    journal.advance_next_index(
                        covered
                            .values()
                            .map(TreeCoverage::next_uncovered)
                            .max()
                            .unwrap_or(0),
                    );
                    // A tree whose rebuild was cut short is left alone: its
                    // record no longer says what it holds, and the repair that
                    // follows rebuilds it from the checkpoint and this journal.
                    let rebuilding = read_rebuild_intents(config.data_dir())?;
                    let mut replayed = 0usize;
                    for entry in &entries {
                        // Group the entry's data ops per partition so each
                        // partition receives them as one batch at the entry's
                        // ts, with the entry's marker last. DERIVED work is
                        // derived first, from the whole entry: the data
                        // partition it reads may already hold the entry.
                        let ops = crate::oplog::convert::resolve_entry_ops(&entry.ops)?;
                        let mut per_partition: HashMap<Partition, Vec<&OplogOp>> = HashMap::new();
                        for op in ops.iter() {
                            #[cfg(feature = "columnar")]
                            if let OplogOp::ColumnarInsert {
                                table_id,
                                key,
                                value,
                            } = op
                            {
                                columnar_replay.push(ColumnarReplay {
                                    table_id: table_id.clone(),
                                    key: key.clone(),
                                    value: value.clone(),
                                    ts: entry.ts,
                                    index: entry.index,
                                });
                                continue;
                            }
                            let Some(part) = op_partition(op) else {
                                continue;
                            };
                            per_partition.entry(part).or_default().push(op);
                        }
                        for (part, ops) in per_partition {
                            let tree = trees.get(&part).ok_or_else(|| {
                                StorageError::PartitionNotFound {
                                    name: part.name().to_string(),
                                }
                            })?;
                            let skip = rebuilding.contains(&part)
                                || covered
                                    .get(&part)
                                    .is_some_and(|c| c.contains(entry.index, 0));
                            if !skip {
                                apply_oplog_ops_at(tree, &ops, entry.ts, entry.index)?;
                                replayed += ops.len();
                            }
                        }
                    }
                    if replayed > 0 {
                        tracing::info!(
                            replayed,
                            "oplog: replayed un-flushed entries from journal on open"
                        );
                        for tree in trees.values() {
                            tree.flush_active_memtable(0)?;
                        }
                    }
                    Some(journal)
                }
                // No oplog-eligible (Durable/Degraded) endpoint — in-memory /
                // no-persistence config. Durability is best-effort; no journal.
                Err(_) => None,
            },
            None => None,
        };

        // Restore seqno counter to (max_persisted_seqno + 1) so reads see all
        // data written in a previous session. seqno_filter uses strict <, so
        // `get()` must be > the highest written seqno for those writes to be visible.
        let max_persisted = trees
            .values()
            .filter_map(|t| {
                use lsm_tree::AbstractTree;
                t.get_highest_seqno()
            })
            .max();
        if let Some(max) = max_persisted {
            seqno.fetch_max(max + 1);
        }

        // The data's highest seqno is not the whole story. A compaction that
        // collected history records a retention floor and zeroes the seqnos of
        // the rows it settled, so the highest seqno left in the tables can sit
        // far below that floor. Reads at or below the floor are refused
        // (`lsm_tree::Error::SnapshotBelowRetention`), so a counter seeded only
        // from the data would reopen the engine unable to read itself. Seed
        // above the floor as well.
        let max_floor = trees
            .values()
            .map(|t| {
                use lsm_tree::AbstractTree;
                t.retention_floor()
            })
            .max();
        if let Some(floor) = max_floor {
            // Checked, not clamped: `SeqNo::MAX` is the read-latest sentinel
            // and never a real floor, but if it ever appeared there would be
            // no servable seqno to seed and silently wrapping to 0 would seed
            // the one value guaranteed to be refused.
            if let Some(first_servable) = floor.checked_add(1) {
                seqno.fetch_max(first_servable);
            }
        }

        // A fresh journalled store gets its coverage record before it accepts
        // a write, and the record is made durable at once: a store with
        // journal entries and no record is one whose coverage cannot be
        // proven, and is refused on the next open.
        if let Some(next) = establish_coverage {
            let at = seqno.next();
            for tree in trees.values() {
                coverage::write_fold(tree, Domain::Journal, next, next, &[], at);
            }
            for tree in trees.values() {
                tree.flush_active_memtable(0)?;
            }
        }
        let (oplog, coverage) = match oplog {
            Some(journal) => {
                let coverage = Coverage::new(journal.next_index());
                (Some(Arc::new(Mutex::new(journal))), Some(coverage))
            }
            None => (None, None),
        };

        // Open tiered cache if configured.
        let tiered_cache = if config.cache.is_enabled() {
            match TieredCache::open(&config.cache) {
                Ok(cache) => Some(cache),
                Err(e) => {
                    tracing::warn!(error = %e, "tiered cache failed to open, continuing without");
                    None
                }
            }
        } else {
            None
        };

        // Start background flush manager. Its monitor sleeps until the write
        // path's trigger, a finished flush or a memtable's age wakes it; the
        // compaction monitor sleeps until a flush or a compaction wakes it.
        let flush_wake = Arc::new(coordinode_core::txn::wake::Wake::default());
        let compaction_wake = Arc::new(coordinode_core::txn::wake::Wake::default());
        let flush_trigger = Arc::new(crate::engine::flush::FlushTrigger::new(
            Arc::clone(&flush_wake),
            config.max_write_buffer_bytes,
        ));
        let flush_manager = FlushManager::start(
            &trees,
            gc_controller.watermarks(),
            config.max_write_buffer_bytes,
            config.max_sealed_memtables,
            config.flush_workers,
            config.max_memtable_age_secs,
            flush_wake,
            Arc::clone(&compaction_wake),
        )?;

        // Feed the backpressure thresholds to every partition tree so the
        // monitor's per-tree verdict has something to compute against
        // (thresholds default to OFF inside lsm-tree).
        let thresholds = config.backpressure.to_thresholds();
        for tree in trees.values() {
            let inner = match tree {
                lsm_tree::AnyTree::Standard(t) => t,
                lsm_tree::AnyTree::Blob(bt) => &bt.index,
            };
            inner.update_runtime_config(|cfg| cfg.backpressure = thresholds)?;
        }

        // Start background compaction scheduler. It latches the worst
        // per-partition backpressure verdict into `write_pressure` each poll
        // cycle; the commit path reads it as a single atomic load.
        let write_pressure = Arc::new(AtomicU8::new(0));
        let compaction_scheduler = CompactionScheduler::start(
            &trees,
            gc_controller.watermarks(),
            config.compaction_workers,
            config.compaction_l0_urgent_threshold,
            config.backpressure.bytes_slowdown,
            Arc::clone(&write_pressure),
            compaction_wake,
        )?;

        info!(
            path = %config.data_dir().display(),
            endpoints = config.endpoints.len(),
            partitions = trees.len(),
            cache_layers = config.cache.layers.len(),
            flush_workers = config.flush_workers,
            compaction_workers = config.compaction_workers,
            "storage engine opened"
        );

        // Build the capacity tracker + warm-load persisted snapshots
        // BEFORE spawning the scanner so the first scan tick sees the
        // hydrated state.
        let capacity_arc = {
            let tracker = crate::engine::capacity::CapacityTracker::new(&config.endpoints);
            for ep in &config.endpoints {
                let persisted = load_persisted_capacity(&schema_tree, &seqno, &ep.id);
                if persisted > 0 {
                    if let Some(usage) = tracker.get(&ep.id) {
                        usage
                            .used_bytes
                            .store(persisted, std::sync::atomic::Ordering::Release);
                        use crate::engine::capacity::CapacitySeverity;
                        let sev = CapacitySeverity::for_usage(persisted, usage.hard_limit_bytes);
                        let writable = !matches!(sev, CapacitySeverity::Full);
                        usage
                            .is_writable
                            .store(writable, std::sync::atomic::Ordering::Release);
                    }
                }
            }
            Arc::new(tracker)
        };

        // Spawn the background scanner. It samples every partition's
        // footprint on every deployment; the capacity scan, its persisted
        // usage and cascade eviction run only where an endpoint has a
        // `hard_limit_bytes` to enforce. The closure captures cheap snapshots
        // (AnyTree is internally Arc'd) so no circular reference with `Self`.
        let capacity_scanner = {
            let tracked = config.endpoints.iter().any(|ep| ep.hard_limit_bytes > 0);
            let tracker_c = Arc::clone(&capacity_arc);
            let endpoints_c = config.endpoints.clone();
            let trees_c = trees.clone();
            let seqno_c = Arc::clone(&seqno);
            let gc_controller_c = Arc::clone(&gc_controller);
            let interval = std::time::Duration::from_secs(5);
            Some(
                crate::engine::capacity::CapacityScanner::start(interval, move || {
                    publish_label_counts(&trees_c, &seqno_c);
                    if !tracked {
                        publish_footprint(&trees_c);
                        return;
                    }
                    run_capacity_refresh(&tracker_c, &endpoints_c, &trees_c, &seqno_c, |id| {
                        run_cascade_evict(
                            &endpoints_c,
                            &trees_c,
                            &seqno_c,
                            &|part| gc_controller_c.maintenance_compaction_threshold_for(part),
                            id,
                        )
                    });
                })
                .map_err(|e| StorageError::InvalidConfig(format!("spawn capacity scanner: {e}")))?,
            )
        };

        // Per-table columnar trees live under `<primary endpoint>/tables`,
        // sharing this engine's seqno generator and block cache. Constructed
        // before the cache moves into the coordinator; re-opens any table tree
        // already on disk.
        #[cfg(feature = "columnar")]
        let columnar_tables = {
            // Use the engine's configured filesystem backend (MemFs for
            // in-memory engines) so columnar table trees never touch the host
            // filesystem when the engine is virtual.
            let fs: Arc<dyn lsm_tree::fs::Fs> = config
                .fs
                .clone()
                .unwrap_or_else(|| Arc::new(lsm_tree::fs::StdFs));
            crate::columnar::ColumnarTableRegistry::open(
                config.endpoints[0].path.join("tables"),
                fs,
                Arc::clone(&seqno),
                Arc::clone(&cache),
                config.oplog_sync_method.tree_sync_mode(),
            )?
        };

        // Restore the shared seqno past any persisted columnar write (the
        // partition-only restore above misses a columnar-only seqno line), then
        // replay the journalled columnar rows that did not reach an SST — the
        // mirror of the partition replay loop, routed to the table registry.
        #[cfg(feature = "columnar")]
        {
            use lsm_tree::AbstractTree;
            if let Some(max) = columnar_tables.max_highest_seqno() {
                seqno.fetch_max(max + 1);
            }
            // Each table's own coverage record decides, like a partition's. A
            // table that is gone was dropped: its rows are not brought back,
            // and a table created later under the same name starts its record
            // past every entry journalled before it (`open_columnar_table`).
            let mut covered: HashMap<String, TreeCoverage> = HashMap::new();
            if let Some(coverage) = &coverage {
                for table_id in columnar_tables.table_ids() {
                    if let Some(tree) = columnar_tables.get(&table_id) {
                        let c = TreeCoverage::read(&tree, Domain::Journal)?;
                        // Markers already on disk are folded by the next fold.
                        for index in c.marked_indices() {
                            coverage.note_table_marker(&table_id, index);
                        }
                        covered.insert(table_id, c);
                    }
                }
            }
            let mut touched: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut replayed = 0usize;
            for ColumnarReplay {
                table_id,
                key,
                value,
                ts,
                index,
            } in columnar_replay
            {
                let (Some(tree), Some(c)) =
                    (columnar_tables.get(&table_id), covered.get(&table_id))
                else {
                    continue;
                };
                if !c.contains(index, 0) {
                    apply_columnar_row(&tree, key, value, ts, Some(index))?;
                    if let Some(coverage) = &coverage {
                        coverage.note_table_marker(&table_id, index);
                    }
                    touched.insert(table_id);
                    replayed += 1;
                }
            }
            for table_id in &touched {
                if let Some(tree) = columnar_tables.get(table_id) {
                    tree.flush_active_memtable(0)?;
                }
            }
            if replayed > 0 {
                tracing::info!(
                    replayed,
                    "oplog: replayed un-flushed columnar rows from journal on open"
                );
            }
        }

        // A capture a crash left behind belongs to no copy.
        let captures = config.data_dir().join(raft_coverage::PARTITION_CAPTURE_DIR);
        match std::fs::remove_dir_all(&captures) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(StorageError::Io(format!("clear {captures:?}: {e}"))),
        }

        let coordinator = LocalMultiModalCoordinator::new(
            trees,
            Arc::clone(&seqno),
            cache,
            gc_watermark,
            gc_controller,
            flush_trigger,
        );
        coordinator.set_retention_window_us(retention_window_to_us(
            std::time::Duration::from_secs(config.retention_window_secs),
        ));
        // The open succeeded: a migrated directory now records its format.
        // An open refused above leaves the old marker, so the store it
        // refused stays as its writer left it.
        crate::format::settle(&format)?;
        Ok(Self {
            flush_manager: Some(flush_manager),
            compaction_scheduler: Some(compaction_scheduler),
            coordinator,
            claim_registry: crate::engine::claims::ClaimRegistry::new(config.max_invariant_claims),
            schema_generation: AtomicU64::new(0),
            node_delta_runs: parking_lot::Mutex::new(rustc_hash::FxHashMap::default()),
            metadata_decisions: parking_lot::Mutex::new(()),
            field_dictionary_generation: AtomicU64::new(0),
            field_dictionary_epoch: AtomicU64::new(0),
            pending_commits,
            closure_frontier: AtomicU64::new(0),
            snapshot_wait: std::sync::atomic::AtomicU64::new(config.snapshot_wait_ms),
            node_shard: std::sync::atomic::AtomicU16::new(config.node_shard),
            flush_policy: config.flush_policy,
            oplog_config: OplogJournalConfig::from(config),
            tiered_cache,
            access_tracker: AccessTracker::new(),
            data_dir: config.data_dir().to_path_buf(),
            endpoints: config.endpoints.clone(),
            capacity: capacity_arc,
            capacity_scanner,
            partition_l0_endpoint,
            oplog,
            coverage,
            oracle,
            #[cfg(feature = "columnar")]
            columnar_tables,
            open_repairs,
            write_pressure,
            raft_fence: parking_lot::RwLock::new(None),
            raft_log_keep_from: AtomicU64::new(u64::MAX),
            partition_captures: AtomicU64::new(0),
            write_taps: Arc::new(crate::engine::tap::WriteTaps::default()),
            applied_feed: Arc::new(crate::engine::applied::AppliedFeed::default()),
            index_feed_capacity: config.index_feed_capacity.max(1),
            open_transactions: Arc::new(crate::engine::open_txns::OpenTransactions::default()),
            space: Arc::new(crate::engine::space::SpaceGuard::new(config)),
        })
    }

    /// The free-space guard: new writes are refused while it is paused, and
    /// a layer that writes ahead of the engine (a consensus log) checks it
    /// first.
    pub fn space(&self) -> &Arc<crate::engine::space::SpaceGuard> {
        &self.space
    }

    /// The cached write-pressure verdict (one relaxed atomic load), latched
    /// by the compaction monitor from the worst per-partition backpressure
    /// signal. `Stop` means new client writes should be rejected (retryable)
    /// until compaction drains the debt; nothing in the engine sleeps on it.
    #[must_use]
    pub fn write_pressure(&self) -> WritePressure {
        WritePressure::from_u8(
            self.write_pressure
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Test-only override of the cached pressure tier (the production value
    /// is latched by the compaction monitor, which is timing-dependent).
    #[cfg(test)]
    pub(crate) fn force_write_pressure(&self, tier: u8) {
        self.write_pressure
            .store(tier, std::sync::atomic::Ordering::Relaxed);
    }

    /// Partition trees that were structurally repaired during this open, with
    /// the replay obligation each repair left behind. Empty on a clean open.
    ///
    /// A caller running WAL-replay-repair (the embedded repair orchestrator)
    /// must treat any entry whose scope is not
    /// [`TailOnly`](lsm_tree::WalReplayScope::TailOnly) as a partition to
    /// rebuild from a checkpoint plus oplog replay: its tree opens and reads
    /// cleanly, but block salvage dropped the key ranges in
    /// [`OpenRepair::report`]`.lost_coverage`.
    #[must_use]
    pub fn open_repairs(&self) -> &[OpenRepair] {
        &self.open_repairs
    }

    /// Create (or open if it already exists) the columnar tree for a
    /// `STORAGE COLUMNAR` table, returning a handle to it. Idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the tree cannot be opened.
    #[cfg(feature = "columnar")]
    pub fn create_columnar_table(&self, table_id: &str) -> StorageResult<lsm_tree::AnyTree> {
        self.open_columnar_table(table_id)
    }

    /// The tree of `table_id`, created on first use. A table created here
    /// holds none of the entries journalled before it existed (those belonged
    /// to a dropped table of the same name, if any), so on a journalled engine
    /// it gets a coverage base at the journal's next index, made durable
    /// before any row can reach it.
    #[cfg(feature = "columnar")]
    fn open_columnar_table(&self, table_id: &str) -> StorageResult<lsm_tree::AnyTree> {
        self.columnar_tables.create_or_open_with(table_id, |tree| {
            let Some(oplog) = &self.oplog else {
                return Ok(());
            };
            // Every row for this table is journalled only after its writer got
            // the tree, which the registry lock held here prevents, so each of
            // them gets an index at or above `next`. Other commits appending
            // meanwhile touch other trees.
            let next = oplog
                .lock()
                .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?
                .next_index();
            tree.insert(
                Domain::Journal.base_key(),
                coverage::encode_base(next, &[]),
                self.next_seqno(),
            );
            tree.flush_active_memtable(0)?;
            Ok(())
        })
    }

    /// Handle to an existing `STORAGE COLUMNAR` table tree, or `None` if no such
    /// table is registered on this node.
    #[cfg(feature = "columnar")]
    pub fn columnar_table_tree(&self, table_id: &str) -> Option<lsm_tree::AnyTree> {
        self.columnar_tables.get(table_id)
    }

    /// Drop a `STORAGE COLUMNAR` table: release its tree handle and delete its
    /// on-disk directory. Idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the directory removal fails.
    #[cfg(feature = "columnar")]
    pub fn drop_columnar_table(&self, table_id: &str) -> StorageResult<()> {
        self.columnar_tables.drop_table(table_id)
    }

    /// Insert a row `(key, value)` into a `STORAGE COLUMNAR` table's tree at
    /// `seqno`, creating the tree on first use. The engine transposes rows to
    /// columnar blocks at flush and reconstructs them on read.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the tree cannot be opened.
    #[cfg(feature = "columnar")]
    pub fn columnar_insert(
        &self,
        table_id: &str,
        key: Vec<u8>,
        value: Vec<u8>,
        seqno: lsm_tree::SeqNo,
    ) -> StorageResult<()> {
        // Journal the row at its commit_ts BEFORE applying it to the tree
        // memtable, so a write that reached the retained oplog survives a crash
        // and is replayed on the next open. Columnar rows live outside the
        // Partition keyspace, so they carry a table-id-tagged ColumnarInsert op
        // routed back to the table registry on recovery, and the table's own
        // tree records its coverage. No journal (cluster mode / plain open /
        // in-memory) → durability follows the tree's own flush.
        let tree = self.open_columnar_table(table_id)?;
        let index = match &self.oplog {
            Some(oplog) => {
                let mut guard = oplog
                    .lock()
                    .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
                Some(guard.append_columnar(table_id, &key, &value, seqno)?)
            }
            None => None,
        };
        apply_columnar_row(&tree, key, value, seqno, index)?;
        if let (Some(index), Some(coverage)) = (index, &self.coverage) {
            coverage.note_table_marker(table_id, index);
            self.note_applied(coverage, index);
        }
        Ok(())
    }

    /// Flush a `STORAGE COLUMNAR` table's active memtable to an on-disk SST, so
    /// its rows survive a clean restart (the runtime table tree is not yet
    /// registered with the background flush manager). Called once per write
    /// statement, not per row. A no-op if the table has no tree.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the flush fails.
    #[cfg(feature = "columnar")]
    pub fn flush_columnar_table(&self, table_id: &str) -> StorageResult<()> {
        use lsm_tree::AbstractTree;
        if let Some(tree) = self.columnar_tables.get(table_id) {
            tree.flush_active_memtable(0)?;
        }
        Ok(())
    }

    /// Scan every row of a `STORAGE COLUMNAR` table visible at `snapshot`,
    /// returned as `(key, value)` byte pairs in key order. Empty if the table
    /// has no tree yet.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] on a block read / decode failure.
    #[cfg(feature = "columnar")]
    pub fn columnar_scan(
        &self,
        table_id: &str,
        snapshot: lsm_tree::SeqNo,
    ) -> StorageResult<Vec<(Vec<u8>, Vec<u8>)>> {
        use lsm_tree::table::columnar::{COL_SEQNO, COL_USER_KEY, COL_VALUE, COL_VALUE_TYPE};

        let Some(tree) = self.columnar_tables.get(table_id) else {
            return Ok(Vec::new());
        };
        // Tree-level projected columnar scan: segment selection, MVCC
        // visibility, delete-bitmap masking, and cross-segment merge run
        // inside the engine on the vectorized batch path. Full-row readback
        // projects all four intrinsic columns; a projected/predicate-pushed
        // variant narrows this list instead of decoding whole rows.
        let mut out = Vec::new();
        // Unbounded, which keeps the engine's zero-copy path for an
        // all-visible segment; the table's coverage record is dropped by key.
        for batch in tree.columnar_scan(
            &[COL_USER_KEY, COL_SEQNO, COL_VALUE_TYPE, COL_VALUE],
            None,
            snapshot,
            ..,
        )? {
            let batch = batch?;
            out.extend(
                crate::columnar::columnar_batch_rows(&batch)?
                    .into_iter()
                    .filter(|(key, _)| !coverage::is_reserved(key)),
            );
        }
        Ok(out)
    }

    /// The rows [`Self::columnar_scan`] returns, handed to `visit` in key
    /// order one engine batch at a time and never collected beyond a batch.
    /// Each batch's rows are charged to `budget` while they are visited, and
    /// each row counts one unit of its work.
    ///
    /// # Errors
    ///
    /// A block read or decode failure, the budget's refusal, or the first
    /// error of `visit`.
    #[cfg(feature = "columnar")]
    pub fn columnar_for_each<E>(
        &self,
        table_id: &str,
        snapshot: lsm_tree::SeqNo,
        budget: &coordinode_core::budget::QueryBudget,
        mut visit: impl FnMut(&[u8], &[u8]) -> Result<(), E>,
    ) -> Result<(), E>
    where
        E: From<StorageError> + From<coordinode_core::budget::BudgetStop>,
    {
        use lsm_tree::table::columnar::{COL_SEQNO, COL_USER_KEY, COL_VALUE, COL_VALUE_TYPE};

        let Some(tree) = self.columnar_tables.get(table_id) else {
            return Ok(());
        };
        for batch in tree
            .columnar_scan(
                &[COL_USER_KEY, COL_SEQNO, COL_VALUE_TYPE, COL_VALUE],
                None,
                snapshot,
                ..,
            )
            .map_err(StorageError::from)?
        {
            let batch = batch.map_err(StorageError::from)?;
            // A batch holds the engine's fixed row count; its decoded rows
            // are charged before any of them is handed on.
            let rows = crate::columnar::columnar_batch_rows(&batch)?;
            let _batch = budget.reserve(
                rows.iter()
                    .map(|(key, value)| (key.capacity() + value.capacity()) as u64)
                    .sum(),
            )?;
            for (key, value) in &rows {
                if coverage::is_reserved(key) {
                    continue;
                }
                budget.work(1)?;
                visit(key, value)?;
            }
        }
        Ok(())
    }

    /// Every `STORAGE COLUMNAR` table with its rows visible at `snapshot`, in
    /// table-id order: the columnar half of a whole-store snapshot. Empty in
    /// a build without columnar support.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] on a block read / decode failure.
    pub fn columnar_tables_at(
        &self,
        snapshot: lsm_tree::SeqNo,
    ) -> StorageResult<Vec<ColumnarTable>> {
        #[cfg(feature = "columnar")]
        {
            let mut tables = Vec::new();
            for table_id in self.columnar_tables.table_ids() {
                let rows = self.columnar_scan(&table_id, snapshot)?;
                tables.push((table_id, rows));
            }
            Ok(tables)
        }
        #[cfg(not(feature = "columnar"))]
        {
            let _ = snapshot;
            Ok(Vec::new())
        }
    }

    /// Make the `STORAGE COLUMNAR` tables exactly `tables`: a table not
    /// listed is dropped, a listed one is recreated holding only its listed
    /// rows (sorted by key, as [`Self::columnar_tables_at`] returns them).
    /// The installing half of a whole-store snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if a tree cannot be dropped, created
    /// or written, and [`StorageError::InvalidConfig`] when rows arrive for a
    /// build without columnar support.
    pub fn replace_columnar_tables(&self, tables: Vec<ColumnarTable>) -> StorageResult<()> {
        #[cfg(feature = "columnar")]
        {
            let keep: std::collections::HashSet<&str> =
                tables.iter().map(|(id, _)| id.as_str()).collect();
            for table_id in self.columnar_tables.table_ids() {
                if !keep.contains(table_id.as_str()) {
                    self.columnar_tables.drop_table(&table_id)?;
                }
            }
            for (table_id, rows) in &tables {
                self.columnar_tables.drop_table(table_id)?;
                let tree = self.open_columnar_table(table_id)?;
                if !rows.is_empty() {
                    let rows: Vec<crate::columnar::ColumnarRow<'_>> = rows
                        .iter()
                        .map(|(key, value)| crate::columnar::ColumnarRow { key, value })
                        .collect();
                    crate::columnar::write_columnar_rows(&tree, &rows)?;
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "columnar"))]
        {
            if tables.is_empty() {
                Ok(())
            } else {
                Err(StorageError::InvalidConfig(
                    "snapshot carries STORAGE COLUMNAR tables; this build has no columnar support"
                        .into(),
                ))
            }
        }
    }

    /// The timestamp oracle this engine stamps writes with, when opened
    /// via [`StorageEngine::open_with_oracle`]. Subsystems that apply
    /// externally stamped writes (the Raft state machine on followers)
    /// must advance this oracle so local MVCC readers observe them.
    pub fn oracle(&self) -> Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>> {
        self.oracle.clone()
    }

    /// Get a tree handle by logical partition.
    pub fn tree(&self, part: Partition) -> StorageResult<&lsm_tree::AnyTree> {
        self.coordinator
            .trees()
            .get(&part)
            .ok_or_else(|| StorageError::PartitionNotFound {
                name: part.name().to_string(),
            })
    }

    /// Borrow the Layer-3 coordinator. Replicated-writer and the
    /// seqno-consumer registry plug in at this seam — see
    /// [`LocalMultiModalCoordinator`] doc for the wire-in contract.
    pub fn coordinator(&self) -> &LocalMultiModalCoordinator {
        &self.coordinator
    }

    /// Root data directory for this engine.
    ///
    /// Subsystems that need their own on-disk sub-directories may derive
    /// paths from this for the **single-endpoint** baseline; with
    /// multi-endpoint placement they instead consult [`Self::endpoints`] /
    /// [`Self::select_oplog_endpoint`].
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Configured endpoints (the storage-stack endpoint layer).
    /// Subsystems consult this to resolve WAL/oplog/SST placement
    /// targets. Returned by reference — endpoints are config-time immutable
    /// after engine open.
    pub fn endpoints(&self) -> &[crate::engine::config::EndpointConfig] {
        &self.endpoints
    }

    /// Create a consistent on-disk checkpoint of the whole database in
    /// `target` (which must not yet exist). Each partition tree is
    /// hard-link checkpointed into `target/<partition>/` (zero-copy on a
    /// single filesystem, falling back to byte-copy across volumes), and
    /// the oplog directory is copied alongside. The result is a complete,
    /// independently-openable database in one directory, whatever the
    /// endpoint layout it came from: [`Self::open_checkpoint`] opens it.
    ///
    /// The field interner and schema metadata live in the Schema partition
    /// tree, so they are captured by that tree's checkpoint — no separate
    /// handling needed.
    ///
    /// # Errors
    ///
    /// - `target` already exists, or its parent cannot be created
    /// - any partition tree's checkpoint fails (see lsm `create_checkpoint`)
    /// - the oplog directory cannot be copied
    pub fn create_checkpoint(&self, target: &Path) -> StorageResult<CheckpointSummary> {
        claim_checkpoint_dir(target)?;
        let mut summary = CheckpointSummary::default();

        // Fold the applied prefix into every tree first, so the checkpoint's
        // bases, and the journal a rebuild from it needs, are as recent as
        // they can be.
        if let Some(coverage) = &self.coverage {
            let mut folded = coverage.lock_folded();
            self.fold_coverage(coverage, &mut folded);
        }

        // The journal is copied before the trees. A segment the purge drops
        // meanwhile held only entries already persisted in the live trees,
        // which the trees' checkpoint below then carries; copied after the
        // trees instead, a purge could drop entries that reached the trees
        // only after their checkpoint, and the copy would hold them nowhere.
        if let Ok(endpoint) = self.select_oplog_endpoint(0) {
            let src_oplog = endpoint.path.join("oplog");
            if src_oplog.exists() {
                summary.oplog_bytes = capture_journal(&src_oplog, &target.join("oplog"))?;
            }
        }

        self.checkpoint_trees(target, &mut summary)?;
        Ok(summary)
    }

    /// Capture the store's data as it stands, without its journal: every
    /// partition tree and `STORAGE COLUMNAR` table, flushed and hard-linked
    /// into `target` (which must not exist), opened with
    /// [`Self::open_checkpoint`].
    ///
    /// Taken while nothing is applied, the capture holds exactly the applied
    /// prefix, which no MVCC snapshot does: an entry applies at its own
    /// commit timestamp, so one applied later can carry a timestamp below a
    /// snapshot taken earlier and show through it.
    ///
    /// # Errors
    ///
    /// `target` exists or cannot be created, or a tree's checkpoint fails.
    pub fn capture(&self, target: &Path) -> StorageResult<CheckpointSummary> {
        claim_checkpoint_dir(target)?;
        let mut summary = CheckpointSummary::default();
        self.checkpoint_trees(target, &mut summary)?;
        Ok(summary)
    }

    /// Flush and hard-link every partition tree and columnar table into
    /// `target`.
    fn checkpoint_trees(
        &self,
        target: &Path,
        summary: &mut CheckpointSummary,
    ) -> StorageResult<()> {
        use lsm_tree::AbstractTree;

        for &part in Partition::all() {
            let tree = self.tree(part)?;
            // Flush the active memtable to an on-disk segment first: a
            // checkpoint snapshots persisted segments, so recent writes still
            // resident in memory would otherwise be missing from the backup.
            // This also makes the per-partition checkpoint directory appear
            // deterministically (previously it depended on whether a
            // background flush happened to have run before the checkpoint).
            tree.flush_active_memtable(0).map_err(|e| {
                StorageError::Io(format!("flush before checkpoint {}: {e}", part.name()))
            })?;
            let info = tree
                .create_checkpoint(&target.join(part.name()))
                .map_err(|e| {
                    StorageError::Io(format!("checkpoint partition {}: {e}", part.name()))
                })?;
            summary.partitions += 1;
            summary.total_bytes += info.total_bytes;
            summary.max_seqno = summary.max_seqno.max(info.seqno);
        }

        // `STORAGE COLUMNAR` tables live outside the partition trees, under
        // the same `tables` directory the registry reopens them from.
        #[cfg(feature = "columnar")]
        {
            let tables = target.join("tables");
            std::fs::create_dir_all(&tables)
                .map_err(|e| StorageError::Io(format!("create {tables:?}: {e}")))?;
            for table_id in self.columnar_tables.table_ids() {
                let Some(tree) = self.columnar_tables.get(&table_id) else {
                    continue;
                };
                tree.flush_active_memtable(0)?;
                let info = tree
                    .create_checkpoint(&tables.join(&table_id))
                    .map_err(|e| {
                        StorageError::Io(format!("checkpoint columnar table {table_id}: {e}"))
                    })?;
                summary.total_bytes += info.total_bytes;
                summary.max_seqno = summary.max_seqno.max(info.seqno);
            }
        }
        // The copy is a directory of this engine's format, opened as one.
        crate::format::write_marker(target, coordinode_core::version::engine_format_version())
    }

    /// Open a checkpoint written by [`Self::create_checkpoint`] as a plain
    /// engine over its one directory. The store's own per-level routing
    /// names the endpoints it was written under, so it is replaced by the
    /// single-directory default (see [`StorageConfig::relocated()`]) under
    /// the endpoint id a single-directory server uses, `default`: once
    /// opened here, the checkpoint also opens as that server's data
    /// directory.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::open`].
    pub fn open_checkpoint(checkpoint_dir: &Path) -> StorageResult<Self> {
        use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
        Self::open(
            &StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                checkpoint_dir,
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )])
            .relocated(),
        )
    }

    /// Rebuild a corrupt partition from a checkpoint plus oplog replay
    /// (WAL-replay-repair, repair path 2). Used when no healthy replica can
    /// serve the partition — the single-node / embedded / RF=1 case.
    ///
    /// Opens `checkpoint_dir` read-only, exports the partition's base
    /// key-values, physically drops the live (corrupt) partition tables,
    /// reinstalls the base, then replays the entries of `oplog_since` the
    /// base lacks to roll the partition forward to its current state.
    /// Returns the number of base entries reinstalled.
    ///
    /// Same-disk checkpoints only protect against localized corruption;
    /// whole-device loss requires an off-device backup (PITR).
    pub fn repair_partition_from_checkpoint(
        &self,
        checkpoint_dir: &Path,
        oplog_since: &[OplogEntry],
        partition: Partition,
    ) -> StorageResult<usize> {
        // 1. Open the checkpoint read-only and export the partition base, with
        //    the record of which journal entries the base holds. The
        //    checkpoint engine is dropped before we mutate the live engine.
        let (base, held): (Rows, TreeCoverage) = {
            let ckpt = StorageEngine::open_checkpoint(checkpoint_dir)?;
            let snapshot = ckpt.snapshot();
            let prefix = format!("{}:", partition.name());
            let rows = ckpt
                .snapshot_prefix_scan(&snapshot, partition, prefix.as_bytes())?
                .into_iter()
                .map(|(k, v)| (k, v.to_vec()))
                .collect();
            (
                rows,
                TreeCoverage::read(ckpt.tree(partition)?, Domain::Journal)?,
            )
        };

        // 2. Physically clear the live (corrupt) tables, reinstall the base.
        //    `clear_partition` deletes the table files without reading their
        //    blocks; a `drop_range` here would instead run a compaction that
        //    re-reads the corrupt block we are repairing and abort with a
        //    ChecksumMismatch. The clear is on disk at once while the rebuilt
        //    data is not, so the intent goes first.
        self.begin_rebuild(partition)?;
        self.clear_partition(partition)?;
        let base_len = base.len();
        for (key, value) in &base {
            self.put(partition, key, value)?;
        }

        // 3. Replay the granular oplog ops the base lacks for this partition,
        //    rolling it forward to the current state. What the base holds is
        //    its own record, not a journal position: a commit landing while
        //    the checkpoint was taken can be in the base or not.
        for entry in oplog_since {
            if held.contains(entry.index, 0) {
                continue;
            }
            // A rebuilt index partition derives its DERIVED entries from the
            // journal entry's own data ops, which it does not replay itself.
            let ops = crate::oplog::convert::resolve_entry_ops(&entry.ops)?;
            for op in ops.iter() {
                if op_partition(op) != Some(partition) {
                    continue;
                }
                self.apply_repair_op(partition, op)?;
            }
        }
        self.rebind_coverage(partition)?;
        self.finish_rebuild(partition)?;
        Ok(base_len)
    }

    /// After a rebuild replayed the journal into `partition` up to its end,
    /// record in the partition's tree that it holds every applied entry. The
    /// clear removed its old record, and the replayed ops carry none.
    ///
    /// Precondition: no commit is in flight, so every journal entry is
    /// applied and the rebuild replayed all of them.
    fn rebind_coverage(&self, partition: Partition) -> StorageResult<()> {
        let Some(coverage) = &self.coverage else {
            return Ok(());
        };
        let tree = self.tree(partition)?;
        let (next, above) = coverage.applied_snapshot();
        // Written after the rebuilt data, so a persisted record implies the
        // data it covers is persisted.
        let at = self.next_seqno();
        let domain = Domain::Journal;
        tree.insert(domain.base_key(), coverage::encode_base(next, &[]), at);
        for index in above {
            tree.insert(domain.marker_key(index, 0).as_slice(), &[][..], at);
        }
        self.coordinator.flush_trigger().wrote_unmeasured();
        Ok(())
    }

    /// The partitions whose rebuild started and did not finish: a crash
    /// between clearing a tree and persisting its rebuilt data. Such a tree
    /// holds an unknown part of its data and of its coverage record, so only
    /// a new full rebuild restores it.
    ///
    /// # Errors
    ///
    /// The intent directory cannot be read.
    pub fn pending_rebuilds(&self) -> StorageResult<Vec<Partition>> {
        read_rebuild_intents(&self.data_dir)
    }

    /// Record, durably, that `partition` is about to be rebuilt. Written
    /// before the tree is cleared and removed by [`Self::finish_rebuild`]
    /// once the rebuilt tree is on disk.
    fn begin_rebuild(&self, partition: Partition) -> StorageResult<()> {
        let dir = self.data_dir.join(REBUILD_INTENT_DIR);
        let io = |what: &str, e: std::io::Error| {
            StorageError::Io(format!(
                "rebuild intent for {}: {what}: {e}",
                partition.name()
            ))
        };
        // no-std: StoragePort trait, caller-injected
        std::fs::create_dir_all(&dir).map_err(|e| io("create dir", e))?;
        let file =
            std::fs::File::create(dir.join(partition.name())).map_err(|e| io("create", e))?;
        file.sync_all().map_err(|e| io("sync", e))?;
        sync_dir(&dir)?;
        sync_dir(&self.data_dir)
    }

    /// Persist the rebuilt tree of `partition`, then drop its intent.
    fn finish_rebuild(&self, partition: Partition) -> StorageResult<()> {
        use lsm_tree::AbstractTree;
        self.tree(partition)?.flush_active_memtable(0)?;
        if partition == Partition::Schema {
            // The dictionary was rebuilt: views read before it reread it.
            self.note_field_dictionary_change();
        }
        let dir = self.data_dir.join(REBUILD_INTENT_DIR);
        let io = |what: &str, e: std::io::Error| {
            StorageError::Io(format!(
                "rebuild intent for {}: {what}: {e}",
                partition.name()
            ))
        };
        std::fs::remove_file(dir.join(partition.name())).map_err(|e| io("remove", e))?;
        sync_dir(&dir)
    }

    /// Apply one journal op during a checkpoint rebuild (shared by the full
    /// and range-scoped repair paths). Ops that target no partition (Noop,
    /// Raft bookkeeping, columnar rows) are no-ops here; the callers filter
    /// them out by partition before calling.
    fn apply_repair_op(&self, partition: Partition, op: &OplogOp) -> StorageResult<()> {
        match op {
            OplogOp::Insert { key, value, .. } => self.put(partition, key, value),
            OplogOp::Delete { key, .. } => self.delete(partition, key),
            OplogOp::Merge { key, operand, .. } => self.merge(partition, key, operand),
            OplogOp::RemoveRange { start, end, .. } => self.remove_range(partition, start, end),
            OplogOp::Noop
            | OplogOp::RaftEntry { .. }
            | OplogOp::RaftTruncation { .. }
            | OplogOp::ColumnarInsert { .. } => Ok(()),
            OplogOp::Derive { .. } | OplogOp::Unit { .. } => Err(unresolved_unit()),
        }
    }

    /// Range-scoped WAL-replay-repair: rebuild ONLY the inclusive `[min, max]`
    /// key ranges of `partition` from a checkpoint plus oplog replay, leaving
    /// every SST outside those ranges untouched. Used after a lossy open-time
    /// salvage whose losses are key-scopable
    /// ([`OpenRepair::scoped_ranges`]); the cost is proportional to the lost
    /// data plus the post-checkpoint oplog window, not the partition size.
    ///
    /// Exactly-once per range: each range is cleared with a range tombstone
    /// (safe here — salvage already rewrote the tree checksum-clean, so no
    /// corrupt block can be re-read), the checkpoint base rows inside it are
    /// reinstalled, and the post-checkpoint ops whose keys fall inside it are
    /// replayed in order. A replayed `RemoveRange` op is CLIPPED to its
    /// intersection with the lost ranges: outside them the live tree already
    /// holds the op's final effect, and re-applying the delete at a fresh
    /// seqno would erase writes that originally landed after it.
    ///
    /// Returns the number of checkpoint base entries reinstalled.
    pub fn repair_partition_ranges_from_checkpoint(
        &self,
        checkpoint_dir: &Path,
        oplog_since: &[OplogEntry],
        partition: Partition,
        ranges: &[KeyRange],
    ) -> StorageResult<usize> {
        // The exclusive upper bound one past an inclusive `max` key.
        fn succ(max: &[u8]) -> Vec<u8> {
            let mut s = Vec::with_capacity(max.len() + 1);
            s.extend_from_slice(max);
            s.push(0);
            s
        }
        let in_ranges = |key: &[u8]| {
            ranges
                .iter()
                .any(|(min, max)| key >= &min[..] && key <= &max[..])
        };

        // 1. Open the checkpoint read-only and export the base rows inside
        //    each lost range, with the record of which journal entries the
        //    base holds. The checkpoint engine is dropped before the live
        //    engine is mutated.
        let (base, held): (Rows, TreeCoverage) = {
            let ckpt = StorageEngine::open_checkpoint(checkpoint_dir)?;
            let mut rows = Vec::new();
            for (min, max) in ranges {
                for guard in ckpt.range_scan(partition, min, max)? {
                    let (k, v) = guard.into_inner()?;
                    rows.push((k.to_vec(), v.to_vec()));
                }
            }
            (
                rows,
                TreeCoverage::read(ckpt.tree(partition)?, Domain::Journal)?,
            )
        };

        // 2. Clear the lost ranges on the live tree, reinstall the base. A
        //    crash part way leaves the ranges' state unknown while the
        //    coverage record still claims the entries, so the intent turns
        //    the next open's repair into a full rebuild.
        self.begin_rebuild(partition)?;
        for (min, max) in ranges {
            self.remove_range(partition, min, &succ(max))?;
        }
        let base_len = base.len();
        for (key, value) in &base {
            self.put(partition, key, value)?;
        }

        // 3. Replay the ops the base lacks that intersect the lost ranges, in
        //    journal order, DERIVED work derived from the whole entry first.
        for entry in oplog_since {
            if held.contains(entry.index, 0) {
                continue;
            }
            let ops = crate::oplog::convert::resolve_entry_ops(&entry.ops)?;
            for op in ops.iter() {
                if op_partition(op) != Some(partition) {
                    continue;
                }
                match op {
                    OplogOp::Insert { key, .. }
                    | OplogOp::Delete { key, .. }
                    | OplogOp::Merge { key, .. } => {
                        if in_ranges(key) {
                            self.apply_repair_op(partition, op)?;
                        }
                    }
                    OplogOp::RemoveRange { start, end, .. } => {
                        for (min, max) in ranges {
                            let s = if start[..] > min[..] {
                                start.clone()
                            } else {
                                min.clone()
                            };
                            let bound = succ(max);
                            let e = if end[..] < bound[..] {
                                end.clone()
                            } else {
                                bound
                            };
                            if s < e {
                                self.remove_range(partition, &s, &e)?;
                            }
                        }
                    }
                    OplogOp::Noop
                    | OplogOp::RaftEntry { .. }
                    | OplogOp::RaftTruncation { .. }
                    | OplogOp::ColumnarInsert { .. } => {}
                    OplogOp::Derive { .. } | OplogOp::Unit { .. } => {
                        return Err(unresolved_unit());
                    }
                }
            }
        }
        self.finish_rebuild(partition)?;
        Ok(base_len)
    }

    /// Read journal entries with `index >= from_index` from the embedded oplog,
    /// or `None` when no journal is active. Used by the repair orchestrator to
    /// collect the entries to replay forward from a checkpoint cursor.
    pub fn oplog_read_since(&self, from_index: u64) -> StorageResult<Option<Vec<OplogEntry>>> {
        match &self.oplog {
            None => Ok(None),
            Some(oplog) => {
                let mut guard = oplog
                    .lock()
                    .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
                Ok(Some(guard.read_since(from_index)?))
            }
        }
    }

    /// The journal index a rebuild from `checkpoint_dir` replays from: every
    /// entry below it is in every tree of the checkpoint, by the trees' own
    /// coverage bases. The repair orchestrator feeds this to
    /// [`oplog_read_since`], and the journal purge keeps everything at or
    /// above it. `0` for a checkpoint without coverage records.
    ///
    /// [`oplog_read_since`]: Self::oplog_read_since
    pub fn checkpoint_replay_floor(checkpoint_dir: &Path) -> StorageResult<u64> {
        let ckpt = StorageEngine::open_checkpoint(checkpoint_dir)?;
        let mut floor = u64::MAX;
        for tree in ckpt.coordinator.trees().values() {
            let held = TreeCoverage::read(tree, Domain::Journal)?;
            floor = floor.min(held.base().map_or(0, |(next, _)| next));
        }
        Ok(if floor == u64::MAX { 0 } else { floor })
    }

    /// Purge journal segments outside the retention window. No-op when no
    /// journal is active. Returns the number of segments removed. Called
    /// periodically by the embedded checkpoint scheduler so the journal does
    /// not grow without bound.
    /// `keep_from_index` is the WAL-replay-repair floor: the latest
    /// checkpoint's oplog cursor. Segments holding entries at or above it are
    /// kept regardless of age, so a checkpoint rebuild can always replay
    /// forward. Pass `u64::MAX` when no checkpoint exists (only the time
    /// window and the durability guard apply).
    ///
    /// Independently of the time window and the index floor, a segment
    /// holding any entry whose write is not yet persisted in its partition
    /// tree (the journal is its only copy) always survives the purge:
    /// dropping it would silently lose the write on the next crash. Durable
    /// means covered by a coverage base that is itself on disk: the purge
    /// folds the applied prefix into every tree and flushes them first.
    pub fn oplog_purge_expired(&self, now_secs: u64, keep_from_index: u64) -> StorageResult<usize> {
        match (&self.oplog, &self.coverage) {
            (Some(oplog), Some(coverage)) => {
                let durable_below = {
                    let mut folded = coverage.lock_folded();
                    let next = self.fold_coverage(coverage, &mut folded);
                    // The flush persists every memtable sealed so far, and
                    // with it the base just written.
                    self.persist()?;
                    next
                };
                // The fold reached every partition tree and every columnar
                // table tree, so one index bound covers them all. A build
                // without columnar support cannot have applied a columnar row
                // and keeps any entry that carries one.
                let is_durable = |entry: &OplogEntry| -> bool {
                    entry.index < durable_below
                        && (cfg!(feature = "columnar")
                            || !entry
                                .ops
                                .iter()
                                .any(|op| matches!(op, OplogOp::ColumnarInsert { .. })))
                };
                let mut guard = oplog
                    .lock()
                    .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
                guard.purge_expired(now_secs, keep_from_index, &is_durable)
            }
            _ => Ok(0),
        }
    }

    /// Resolve the oplog target endpoint for a given shard (the
    /// endpoint-to-journal placement rule). Convenience accessor that re-runs the per-shard
    /// round-robin selection logic from [`crate::engine::config::StorageConfig::select_oplog_endpoint`]
    /// against the stored endpoint list. Returns an `Io` error if no
    /// endpoint qualifies — typical caller is `LogStore::open` (Raft),
    /// which surfaces this as a configuration error.
    pub fn select_oplog_endpoint(
        &self,
        shard_id: u32,
    ) -> StorageResult<&crate::engine::config::EndpointConfig> {
        let eligible: Vec<&crate::engine::config::EndpointConfig> = self
            .endpoints
            .iter()
            .filter(|ep| ep.is_oplog_eligible())
            .collect();
        if eligible.is_empty() {
            return Err(StorageError::Io(
                "no oplog-eligible endpoint configured (need Durable or Degraded \
                 durability: the oplog must survive a process restart)"
                    .to_string(),
            ));
        }
        // SAFETY: eligible non-empty checked above.
        Ok(eligible[(shard_id as usize) % eligible.len()])
    }

    /// All oplog-eligible endpoints — used for cross-endpoint recovery scan.
    /// On engine open, callers (`LogStore::open` for Raft) inspect every
    /// oplog-eligible endpoint's `oplog/<shard_id>/` directory to
    /// discover sealed segments that may have been written under a
    /// previous config-driven endpoint routing.
    pub fn all_oplog_eligible_endpoints(&self) -> Vec<&crate::engine::config::EndpointConfig> {
        self.endpoints
            .iter()
            .filter(|ep| ep.is_oplog_eligible())
            .collect()
    }

    /// Advance the seqno by one and return the new value.
    ///
    /// Used by [`WriteBatch`] to assign a single seqno to an entire batch.
    pub(crate) fn next_seqno(&self) -> lsm_tree::SeqNo {
        self.coordinator.next_seqno()
    }

    /// Update the GC watermark for the seqno-based retention filter.
    ///
    /// Versions with `seqno <= watermark` become eligible for removal
    /// during LSM compaction.
    pub fn set_gc_watermark(&self, watermark: u64) {
        self.coordinator.set_gc_watermark(watermark);
    }

    /// Pin a read snapshot at the current seqno, holding the GC watermark at or
    /// below it until the returned guard drops. A reader that holds the guard
    /// for the lifetime of its snapshot reads is guaranteed that compaction
    /// will not fold or collect any state it must still observe. Returns the
    /// pinned seqno to read at.
    pub fn pin_snapshot(&self) -> (lsm_tree::SeqNo, SnapshotPin) {
        self.coordinator.pin_snapshot()
    }

    /// Pin a read snapshot at an explicit seqno (a time-travel statement, a
    /// long-lived backup / CDC consumer). Holds the GC watermark at or below
    /// `seqno` until the guard drops. `None` when `seqno` is already below the
    /// watermark: that history may be collected and no pin can protect it;
    /// read at or above [`Self::gc_watermark`] instead.
    pub fn pin_snapshot_at(&self, seqno: lsm_tree::SeqNo) -> Option<SnapshotPin> {
        self.coordinator.pin_snapshot_at(seqno)
    }

    /// The latest complete snapshot, already pinned.
    pub fn pin_latest_snapshot(&self) -> (lsm_tree::SeqNo, Option<SnapshotPin>) {
        self.pin_new_snapshot(|| self.snapshot())
    }

    /// Record a transaction as open until the returned marker drops.
    /// [`Self::await_transactions_through`] waits for the ones opened before
    /// a boundary.
    pub(crate) fn open_transaction(&self) -> crate::engine::open_txns::OpenTransaction {
        use crate::engine::coordinator::MultiModalCoordinator as _;
        self.open_transactions
            .open(|| self.coordinator.current_seqno())
    }

    /// Choose a snapshot that is not behind the present and pin it in the same
    /// step: the latest complete snapshot, or a timestamp freshly allocated
    /// from the clock.
    ///
    /// Choosing and then calling [`Self::pin_snapshot_at`] is two steps, and
    /// commits can land between them: the snapshot floor rises past the
    /// choice, the next pin released anywhere publishes it as the watermark,
    /// and the pin is refused. So the watermark is held where it stands first.
    /// The watermark never exceeds the snapshot floor, the floor only rises,
    /// and a fresh timestamp is above it, so the choice made under the hold is
    /// at or above the held watermark and pinning it cannot be refused.
    ///
    /// `choose` must not return a seqno older than the moment it is called;
    /// a historical snapshot is pinned with [`Self::pin_snapshot_at`], which
    /// refuses it when its history may be gone.
    pub fn pin_new_snapshot(
        &self,
        choose: impl FnOnce() -> lsm_tree::SeqNo,
    ) -> (lsm_tree::SeqNo, Option<SnapshotPin>) {
        let hold = self.coordinator.hold_watermark();
        let snapshot = choose();
        let pin = self.coordinator.pin_snapshot_at(snapshot);
        debug_assert!(
            pin.is_some(),
            "a snapshot chosen under a watermark hold was refused its pin"
        );
        drop(hold);
        (snapshot, pin)
    }

    /// The oldest seqno a time-travel read can still be served at.
    ///
    /// Two boundaries, and the horizon is the later of them:
    ///
    /// - **Policy** — the GC watermark, which the configured time-travel
    ///   window, the consumer floor and live snapshot pins drive together.
    /// - **Physical** — each partition tree's persisted retention floor. A
    ///   compaction that collected below watermark `w` records `w - 1`; a
    ///   `drop_range` or a `clear` records its own install seqno, which prunes
    ///   history independently of the window, so this can sit above the
    ///   watermark. A read at or below the floor is refused by the tree itself
    ///   (`lsm_tree::Error::SnapshotBelowRetention`), hence the `+ 1`.
    ///
    /// The system partitions keep no history of their own (see
    /// [`Partition::is_system`]), so their floors run ahead of the window and
    /// do not bound a time-travel read of data.
    pub fn oldest_readable_seqno(&self) -> lsm_tree::SeqNo {
        use lsm_tree::AbstractTree as _;
        let mut horizon = self.coordinator.gc_watermark_value();
        for part in Partition::all().iter().filter(|part| !part.is_system()) {
            if let Ok(tree) = self.tree(*part) {
                // A tree serves strictly above its floor, so the first
                // readable seqno is one past it. Plain arithmetic: the floor
                // is a watermark or an install seqno, both from the seqno
                // generator (HLC microseconds or a counter), and is never
                // `SeqNo::MAX`, the read-latest sentinel a caller passes to a
                // read. The `debug_assert` fails loudly if that ever stops
                // holding, which is where it would need fixing.
                let oldest_retained = tree.oldest_retained_seqno();
                debug_assert!(
                    oldest_retained < lsm_tree::SeqNo::MAX,
                    "retention floor at the read-latest sentinel"
                );
                horizon = horizon.max(oldest_retained + 1);
            }
        }
        horizon
    }

    /// Refuse a snapshot read below the retention horizon. Version history
    /// there may already be collected; answering from what survived would
    /// return a wrong (newer or missing) version, and once the history is
    /// pruned the tree has no retained version to serve it from at all.
    fn check_snapshot_retained(&self, snapshot: lsm_tree::SeqNo) -> StorageResult<()> {
        let watermark = self.oldest_readable_seqno();
        if snapshot < watermark {
            return Err(StorageError::SnapshotOutsideRetention {
                snapshot,
                watermark,
            });
        }
        Ok(())
    }

    /// Advance the GC watermark toward the current seqno when no read snapshot
    /// is pinned. Lets compaction fold merge operands and collect old versions
    /// during quiescent periods; a no-op while any snapshot is pinned.
    pub fn advance_gc_watermark(&self) {
        self.coordinator.advance_gc_watermark();
    }

    /// Publish the consumer-registry retention floor (the seqno-space feed):
    /// `min(consumer_checkpoints, MVCC time-travel window)`, supplied by the
    /// `SeqnoConsumerRegistry` in `coordinode-replicate`. The effective GC
    /// watermark becomes `min(live snapshot pin / current seqno, floor)` — a
    /// lagging CDC / backup consumer or the retention window holds old MVCC
    /// versions back (CockroachDB protected-timestamp / TiDB service-safe-point
    /// shape) without ever overriding a live reader's pin. `u64::MAX` clears it.
    pub fn set_consumer_retention_floor(&self, floor: u64) {
        self.coordinator.set_consumer_retention_floor(floor);
    }

    /// The current GC watermark value: the seqno below which a time-travel
    /// read may already find its versions collected. `AS OF TIMESTAMP T` is
    /// exact for every `T >= watermark` (the version current at the watermark
    /// survives compaction as the base) and refused below it.
    pub fn gc_watermark(&self) -> u64 {
        self.coordinator.gc_watermark_value()
    }

    /// What the retention window holds on disk for `part` beyond its live
    /// version: the tables compactions inside the window consumed, kept for
    /// the snapshots that still see them. See
    /// [`crate::engine::retention_stats`].
    pub fn retained_history(
        &self,
        part: Partition,
    ) -> StorageResult<crate::engine::retention_stats::RetainedHistory> {
        crate::engine::retention_stats::retained_history(self.tree(part)?)
    }

    /// [`Self::retained_history`] for every partition, in `Partition::all()`
    /// order.
    pub fn retained_history_all(
        &self,
    ) -> StorageResult<Vec<(Partition, crate::engine::retention_stats::RetainedHistory)>> {
        let mut out = Vec::with_capacity(Partition::all().len());
        for &part in Partition::all() {
            out.push((part, self.retained_history(part)?));
        }
        Ok(out)
    }

    /// Set the MVCC time-travel retention window at runtime and republish the
    /// GC watermark. Takes effect for the next compaction; widening the window
    /// cannot bring back versions an earlier compaction already collected.
    /// Inert on an engine opened without an oracle (seqnos are not a clock).
    pub fn set_retention_window(&self, window: std::time::Duration) {
        self.coordinator
            .set_retention_window_us(retention_window_to_us(window));
    }

    /// The configured MVCC time-travel retention window.
    pub fn retention_window(&self) -> std::time::Duration {
        std::time::Duration::from_micros(self.coordinator.retention_window_us())
    }

    /// Read a value by key from the given partition.
    ///
    /// Read path with tiered cache:
    ///   1. Tiered cache (NVMe/SSD layers) → hit → return
    ///   2. LSM storage (DRAM block cache → persistent storage) → populate cache → return
    ///   3. Miss → None
    pub fn get(&self, part: Partition, key: &[u8]) -> StorageResult<Option<bytes::Bytes>> {
        // Check tiered cache first. A miss takes the key's ticket before the
        // tree read, so the value is cached only if no write was invalidated
        // since.
        let ticket = match &self.tiered_cache {
            Some(cache) => {
                if let Some(value) = cache.get(part, key) {
                    self.access_tracker.record(part, key);
                    return Ok(Some(value));
                }
                Some(cache.ticket(part, key))
            }
            None => None,
        };

        // Fall through to LSM storage.
        let tree = self.tree(part)?;
        let value = tree.get(key, self.coordinator.current_seqno())?;

        match value {
            Some(v) => {
                let bytes = bytes::Bytes::copy_from_slice(&v);
                if let (Some(cache), Some(ticket)) = (&self.tiered_cache, ticket) {
                    let weight = Self::resolve_cache_weight(cache, part, &bytes);
                    cache.fill(part, key, &bytes, weight, ticket);
                }
                self.access_tracker.record(part, key);
                Ok(Some(bytes))
            }
            None => Ok(None),
        }
    }

    /// Batch point lookup: resolve many keys in one call, returning a value
    /// (or `None`) per input key in the same order.
    ///
    /// Cache-resident keys are served from the tiered cache; the remaining
    /// keys are fetched from the LSM tree through a single
    /// [`lsm_tree::AbstractTree::multi_get`], which acquires the version
    /// snapshot once and batches the bloom-filter + SST traversal for the whole
    /// set — materially cheaper than calling [`Self::get`] in a loop (each of
    /// which re-pins the snapshot and descends independently). Use this whenever
    /// a known set of keys is resolved together (e.g. materializing the node
    /// records behind an index or vector-search result set).
    pub fn multi_get(
        &self,
        part: Partition,
        keys: &[&[u8]],
    ) -> StorageResult<Vec<Option<bytes::Bytes>>> {
        let mut out: Vec<Option<bytes::Bytes>> = vec![None; keys.len()];

        // Split cache hits from misses; only the misses go to the tree, each
        // with its ticket taken before the read (see `get`).
        let mut miss_idx: Vec<usize> = Vec::new();
        let mut miss_keys: Vec<&[u8]> = Vec::new();
        let mut tickets = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            if let Some(cache) = &self.tiered_cache {
                if let Some(value) = cache.get(part, key) {
                    self.access_tracker.record(part, key);
                    out[i] = Some(value);
                    continue;
                }
                tickets.push(cache.ticket(part, key));
            }
            miss_idx.push(i);
            miss_keys.push(key);
        }

        if miss_keys.is_empty() {
            return Ok(out);
        }

        let tree = self.tree(part)?;
        let seqno = self.coordinator.current_seqno();
        let values = tree.multi_get(miss_keys.iter().copied(), seqno)?;

        for (n, (slot, value)) in miss_idx.into_iter().zip(values).enumerate() {
            if let Some(v) = value {
                let bytes = bytes::Bytes::copy_from_slice(&v);
                if let (Some(cache), Some(ticket)) = (&self.tiered_cache, tickets.get(n)) {
                    let weight = Self::resolve_cache_weight(cache, part, &bytes);
                    cache.fill(part, keys[slot], &bytes, weight, *ticket);
                }
                self.access_tracker.record(part, keys[slot]);
                out[slot] = Some(bytes);
            }
        }

        Ok(out)
    }

    /// Write a key-value pair to the given partition.
    /// Invalidates any cached entry for this key.
    ///
    /// Pre-write capacity gate: if the partition's resolved L0
    /// endpoint is currently non-writable (its `used_bytes` is at or
    /// above `hard_limit_bytes` per the latest capacity scan), the
    /// call returns [`StorageError::CapacityExhausted`] without
    /// inserting. The coordinator may retry on a different endpoint
    /// or surface the error to the client.
    pub fn put(&self, part: Partition, key: &[u8], value: &[u8]) -> StorageResult<()> {
        self.check_partition_capacity(part)?;
        self.coordinator.put_no_capacity_check(part, key, value)?;
        self.write_taps.wrote(part, [key]);
        if let Some(cache) = &self.tiered_cache {
            cache.remove(part, key);
        }
        Ok(())
    }

    /// Pre-write capacity gate for a partition. Looks up the
    /// partition's L0 endpoint in the cached routing and consults the
    /// capacity tracker's `is_writable` flag. Returns
    /// [`StorageError::CapacityExhausted`] when the endpoint has
    /// crossed the 100% threshold; `Ok(())` otherwise.
    ///
    /// `Schema` and `Raft` partitions are always permitted — these
    /// hold engine-internal metadata that must remain accessible even
    /// when user-data endpoints are full (otherwise the operator
    /// could not read the metrics that prove the endpoint is full).
    pub fn check_partition_capacity(&self, part: Partition) -> StorageResult<()> {
        if matches!(part, Partition::Schema | Partition::Raft) {
            return Ok(());
        }
        let Some(endpoint_id) = self.partition_l0_endpoint.get(&part) else {
            return Ok(());
        };
        let Some(usage) = self.capacity.get(endpoint_id) else {
            return Ok(());
        };
        if !usage.is_writable() {
            return Err(StorageError::CapacityExhausted {
                endpoint_id: usage.id.clone(),
                used_bytes: usage.used(),
                hard_limit_bytes: usage.hard_limit_bytes,
            });
        }
        Ok(())
    }

    /// Capacity tracker handle — exposed so background scanners,
    /// Prometheus exporters, and admin RPCs can read per-endpoint
    /// usage state without going through `put`/`get` paths.
    pub fn capacity(&self) -> &Arc<crate::engine::capacity::CapacityTracker> {
        &self.capacity
    }

    /// Refresh the capacity tracker by scanning every configured
    /// endpoint's per-partition `tables/` directory and recomputing
    /// `used_bytes`. Side effects: severity-transition logs,
    /// `is_writable` flag flips at the 100% threshold, and (when the
    /// endpoint strategy is `CascadeEvict`) a cascade-eviction fire
    /// at the 95% emergency threshold.
    ///
    /// Synchronous — caller decides cadence. A background polling
    /// loop wrapper lives in the engine's open path.
    pub fn refresh_capacity(&self) {
        run_capacity_refresh(
            &self.capacity,
            &self.endpoints,
            self.coordinator.trees(),
            self.coordinator.seqno_generator(),
            |id| self.cascade_evict_endpoint(id),
        );
    }

    /// Delete a key from the given partition.
    ///
    /// Pre-write capacity gate applies: a delete writes a tombstone
    /// that still consumes memtable bytes (and eventually SST bytes
    /// after flush), so the hard-limit gate must fire here too;
    /// operators evict via cascade or by reducing `hard_limit_bytes`,
    /// not by stuffing more tombstones onto a Full endpoint.
    pub fn delete(&self, part: Partition, key: &[u8]) -> StorageResult<()> {
        self.check_partition_capacity(part)?;
        self.coordinator.delete(part, key)?;
        self.write_taps.wrote(part, [key]);
        if let Some(cache) = &self.tiered_cache {
            cache.remove(part, key);
        }
        Ok(())
    }

    /// Delete the half-open range `[start, end)` of a partition with one MVCC
    /// range tombstone. Snapshot-aware and seqno'd, so it replicates and
    /// PITR-replays correctly — unlike [`drop_range`](Self::drop_range), which is
    /// an eager, non-MVCC table-level drop. Invalidates the partition's cache:
    /// the tombstone leaves the shadowed keys physically present, so a stale
    /// cache hit would otherwise return a deleted value, and the cache cannot
    /// range-query to invalidate precisely.
    ///
    /// Not capacity-gated — deleting frees space.
    pub fn remove_range(&self, part: Partition, start: &[u8], end: &[u8]) -> StorageResult<()> {
        self.coordinator.remove_range(part, start, end)?;
        self.write_taps.replaced(part);
        if let Some(cache) = &self.tiered_cache {
            cache.clear_partition(part);
        }
        Ok(())
    }

    /// Store a merge operand for the given key in the specified partition.
    ///
    /// The operand is lazily combined with the existing value during reads
    /// and compaction, using the partition's registered merge operator.
    ///
    /// For the `Adj` partition, operands are posting list deltas encoded
    /// via `crate::engine::merge::encode_add` / `encode_remove` / `encode_add_batch`.
    ///
    /// Invalidates any cached entry for this key (stale after merge).
    pub fn merge(&self, part: Partition, key: &[u8], operand: &[u8]) -> StorageResult<()> {
        self.check_partition_capacity(part)?;
        self.coordinator
            .merge_no_capacity_check(part, key, operand)?;
        self.write_taps.wrote(part, [key]);
        if let Some(cache) = &self.tiered_cache {
            cache.remove(part, key);
        }
        Ok(())
    }

    /// Apply a single proposal [`Mutation`] to its target partition.
    ///
    /// Maps the partition-agnostic [`PartitionId`] the mutation carries to the
    /// physical [`Partition`] and dispatches to [`Self::put`] / [`Self::delete`]
    /// / [`Self::merge`]. Put/Delete write plain keys (the seqno oracle
    /// auto-stamps them); Merge writes the raw operand. This lets
    /// callers above the storage layer (background maintenance, proposal
    /// pipelines) apply mutations without naming a partition or key encoder.
    ///
    /// [`PartitionId`]: coordinode_core::txn::proposal::PartitionId
    pub fn apply_mutation(&self, mutation: &Mutation) -> StorageResult<()> {
        match mutation {
            Mutation::Put {
                partition,
                key,
                value,
            } => self.put(Partition::from(*partition), key, value),
            Mutation::Delete { partition, key } => self.delete(Partition::from(*partition), key),
            Mutation::Merge {
                partition,
                key,
                operand,
            } => self.merge(Partition::from(*partition), key, operand),
            Mutation::RemoveRange {
                partition,
                start,
                end,
            } => self.remove_range(Partition::from(*partition), start, end),
            // Both are resolved by the unit they belong to.
            Mutation::Command(_) | Mutation::Derive(_) => {
                self.apply_proposal_at(std::slice::from_ref(mutation), 0)
            }
        }
    }

    /// `mutations` with its metadata command replaced by the effects it has
    /// against the state held now, and whether those effects publish field
    /// bindings. Runs under `metadata_decisions`.
    ///
    /// A proposal carries at most one command: a second one in the same
    /// proposal would decide against state that lacks the first one's
    /// effects, so it is refused (it publishes nothing), the same way on
    /// every member.
    fn decide_commands(
        &self,
        mutations: &[Mutation],
    ) -> StorageResult<(Vec<Mutation>, DictionaryChange)> {
        use coordinode_core::txn::proposal::MetadataCommand;
        let mut effects = Vec::with_capacity(mutations.len() + 1);
        let mut decided = false;
        let mut dictionary = DictionaryChange::None;
        for mutation in mutations {
            let Mutation::Command(command) = mutation else {
                effects.push(mutation.clone());
                continue;
            };
            if decided {
                tracing::warn!(
                    ?command,
                    "a second metadata command in one proposal is refused"
                );
                continue;
            }
            decided = true;
            let decided_effects = crate::engine::metadata::decide(self, command)?;
            if !decided_effects.is_empty() {
                dictionary = match command {
                    MetadataCommand::RegisterFields { .. } => DictionaryChange::Extended,
                    MetadataCommand::AdoptFields { .. } => DictionaryChange::Replaced,
                    MetadataCommand::GrantNodeLease { .. }
                    | MetadataCommand::RecordGroupPair { .. } => DictionaryChange::None,
                };
            }
            effects.extend(decided_effects);
        }
        Ok((effects, dictionary))
    }

    /// Publish a dictionary change that has landed.
    fn note_dictionary(&self, change: DictionaryChange) {
        use std::sync::atomic::Ordering::AcqRel;
        match change {
            DictionaryChange::None => {}
            DictionaryChange::Extended => {
                self.field_dictionary_generation.fetch_add(1, AcqRel);
            }
            // The epoch moves first, so a reader that sees the new
            // generation also sees that it must reread everything.
            DictionaryChange::Replaced => {
                self.field_dictionary_epoch.fetch_add(1, AcqRel);
                self.field_dictionary_generation.fetch_add(1, AcqRel);
            }
        }
    }

    /// The field dictionary generation: it moves whenever a binding is
    /// applied here or a snapshot replaces the dictionary. A view of the
    /// dictionary read at one value covers every binding applied before it.
    pub fn field_dictionary_generation(&self) -> u64 {
        self.field_dictionary_generation
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// The field dictionary epoch: it moves when bindings may have landed
    /// below the frontier, which a view can only pick up by rereading all.
    pub fn field_dictionary_epoch(&self) -> u64 {
        self.field_dictionary_epoch
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Hold off metadata commands: none is decided or applied, and no field
    /// dictionary is read, until the guard drops. The two records of a binding
    /// land key by key, so a reader running beside their writer can find one
    /// without the other.
    pub(crate) fn metadata_exclusive(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.metadata_decisions.lock()
    }

    /// Run `f` with metadata commands held off, for a writer that replaces
    /// field dictionary records outside an applied command (a snapshot
    /// install): no dictionary read sees the replacement half done.
    pub fn with_metadata_exclusive<R>(&self, f: impl FnOnce() -> R) -> R {
        let _exclusive = self.metadata_exclusive();
        f()
    }

    /// Record that the stored field dictionary changed other than through an
    /// applied command (a snapshot or restore installed records directly).
    pub fn note_field_dictionary_change(&self) {
        self.note_dictionary(DictionaryChange::Replaced);
    }

    /// Apply every mutation of one committed proposal at a single seqno: its
    /// commit timestamp (`seqno == commit_ts`). This is the one write
    /// path for a transaction's commit in every deployment mode: the embedded
    /// pipeline, the Raft state machine applying a replicated entry, and the
    /// direct no-pipeline commit all land here, so a snapshot read at
    /// `commit_ts` sees the whole transaction and a read at `commit_ts - 1`
    /// sees none of it.
    ///
    /// `commit_ts` is honoured verbatim on an oracle-backed engine
    /// ([`StorageEngine::open_with_oracle`] and the embedded variants): its
    /// seqno generator IS the oracle the timestamp was allocated from, and the
    /// generator is advanced to `commit_ts` afterwards so a follower applying
    /// a leader's timestamp never hands out an older one. An engine opened
    /// without an oracle has no relation between its seqnos and the caller's
    /// timestamps, so it allocates a fresh seqno for the batch instead; the
    /// zero timestamp (the "no timestamp" sentinel) does the same.
    ///
    /// # Errors
    ///
    /// A full endpoint rejects the whole proposal before any mutation lands.
    /// Two different operation kinds on one key inside a single proposal are
    /// rejected by the tree (`MixedOperationBatch`): the transaction layer
    /// canonicalises each key to one final operation, so this marks an
    /// upstream bug rather than silently ordering the pair.
    pub fn apply_proposal_at(&self, mutations: &[Mutation], commit_ts: u64) -> StorageResult<()> {
        // A local commit has no log position; its subscribers follow it by
        // commit timestamp. Staged before the write lands, so a reader that
        // sees it is told the derived indexes may not hold it yet.
        let staged = self.applied_feed.stage(0, commit_ts, mutations);
        self.apply_proposal_covered(mutations, commit_ts, None)?;
        staged.publish();
        Ok(())
    }

    /// Journal one committed proposal, then apply it at `commit_ts` together
    /// with its coverage marker: the write path of an engine opened with a
    /// retained journal ([`Self::open_embedded`]). The journal append is
    /// ordered and fsynced; the apply runs outside the journal lock, so
    /// proposals may reach the memtables in a different order than their
    /// indices, which is exactly what per-index coverage makes safe.
    ///
    /// # Errors
    ///
    /// [`StorageError::InvalidConfig`] when the engine has no journal; the
    /// errors of [`Self::apply_proposal_at`] otherwise. An entry that was
    /// journalled but failed to apply stays uncovered and is replayed on the
    /// next open.
    pub fn commit_journaled(&self, mutations: &[Mutation], commit_ts: u64) -> StorageResult<()> {
        // Refused before the journal append, which is the write a full disk
        // would fail in.
        self.space.admit()?;
        if mutations.iter().any(|m| matches!(m, Mutation::Command(_))) {
            // The decision, the journal append and the apply happen under one
            // lock: the journal then records each decision's effects in the
            // order they were decided, and replay reproduces them exactly.
            let _decisions = self.metadata_exclusive();
            let (effects, dictionary) = self.decide_commands(mutations)?;
            // Nothing decided (names already bound, or a refusal) leaves the
            // state as it is, so there is nothing to journal or replay.
            if effects.is_empty() {
                return Ok(());
            }
            self.journal_and_apply(&effects, commit_ts)?;
            self.note_dictionary(dictionary);
            return Ok(());
        }
        self.journal_and_apply(mutations, commit_ts)
    }

    /// Journal `mutations`, then apply them at `commit_ts` with their coverage
    /// marker. None of them is a metadata command.
    fn journal_and_apply(&self, mutations: &[Mutation], commit_ts: u64) -> StorageResult<()> {
        let (Some(oplog), Some(coverage)) = (&self.oplog, &self.coverage) else {
            return Err(StorageError::InvalidConfig(
                "commit_journaled on an engine without a retained journal".into(),
            ));
        };
        let index = {
            let mut guard = oplog
                .lock()
                .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
            guard.append(mutations, commit_ts)?
        };
        let mark = coverage::Mark {
            domain: Domain::Journal,
            index,
            sub: 0,
        };
        // The journal keeps the sealed DERIVED work and its inputs; the trees
        // receive the entries derived from them, under the same marker.
        let resolved =
            coordinode_core::index::derive::resolve_unit(mutations, MAX_DERIVED_EFFECTS)?;
        // Subscribers of an embedded engine follow its commits the way a
        // cluster member's follow its applied log entries, staged before the
        // write lands.
        let staged = self.applied_feed.stage(index, commit_ts, mutations);
        self.apply_effects_covered(&resolved, commit_ts, Some(mark))?;
        self.note_applied(coverage, index);
        staged.publish();
        Ok(())
    }

    /// Record journal entry `index` as applied everywhere it lands, folding
    /// the accumulated markers when enough have built up. The fold runs on
    /// the committing thread unless another one is already folding.
    fn note_applied(&self, coverage: &Coverage, index: u64) {
        if coverage.mark_applied(index) {
            if let Some(mut folded) = coverage.try_lock_folded() {
                self.fold_coverage(coverage, &mut folded);
            }
        }
    }

    /// Fold every applied index below the applied prefix into each partition
    /// tree's coverage base, removing the markers it replaces. Returns the
    /// base now written to every tree.
    fn fold_coverage(&self, coverage: &Coverage, folded: &mut coverage::FoldGuard<'_>) -> u64 {
        let next = coverage.applied_prefix();
        let from = folded.folded();
        if next <= from {
            return from;
        }
        // Above every marker below `next`: each of those entries drew its
        // commit_ts from this generator before it was applied.
        let at = self.next_seqno();
        for tree in self.coordinator.trees().values() {
            coverage::write_fold(tree, Domain::Journal, from, next, &[], at);
        }
        // A fold can land in a memtable a rotation just emptied.
        self.coordinator.flush_trigger().wrote_unmeasured();
        #[cfg(feature = "columnar")]
        for (table_id, markers) in coverage.take_table_markers_below(next) {
            // A dropped table took its markers with it.
            if let Some(tree) = self.columnar_tables.get(&table_id) {
                coverage::write_table_fold(&tree, Domain::Journal, next, &markers, at);
            }
        }
        folded.set_folded(next);
        next
    }

    fn apply_proposal_covered(
        &self,
        mutations: &[Mutation],
        commit_ts: u64,
        cover: Option<coverage::Mark>,
    ) -> StorageResult<()> {
        // DERIVED entries are derived from the unit itself and land in the
        // same batch, under the same coverage marker, as the data they index.
        let mutations =
            coordinode_core::index::derive::resolve_unit(mutations, MAX_DERIVED_EFFECTS)?;
        let mutations = mutations.as_ref();
        if mutations.iter().any(|m| matches!(m, Mutation::Command(_))) {
            let _decisions = self.metadata_exclusive();
            let (effects, dictionary) = self.decide_commands(mutations)?;
            self.apply_effects_covered(&effects, commit_ts, cover)?;
            self.note_dictionary(dictionary);
            return Ok(());
        }
        self.apply_effects_covered(mutations, commit_ts, cover)
    }

    /// Apply `mutations`, none of them a metadata command, as one batch at
    /// `commit_ts` together with `cover`.
    fn apply_effects_covered(
        &self,
        mutations: &[Mutation],
        commit_ts: u64,
        cover: Option<coverage::Mark>,
    ) -> StorageResult<()> {
        if mutations.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::with_capacity(self, mutations.len());
        for mutation in mutations {
            match mutation {
                Mutation::Put {
                    partition,
                    key,
                    value,
                } => batch.put(Partition::from(*partition), key.clone(), value.clone()),
                Mutation::Delete { partition, key } => {
                    batch.delete(Partition::from(*partition), key.clone());
                }
                Mutation::Merge {
                    partition,
                    key,
                    operand,
                } => batch.merge(Partition::from(*partition), key.clone(), operand.clone()),
                Mutation::RemoveRange {
                    partition,
                    start,
                    end,
                } => batch.remove_range(Partition::from(*partition), start.clone(), end.clone()),
                Mutation::Command(_) => {
                    return Err(StorageError::InvalidConfig(
                        "a metadata command reached the write batch undecided".into(),
                    ));
                }
                Mutation::Derive(_) => {
                    return Err(StorageError::InvalidConfig(
                        "derived index work reached the write batch unresolved".into(),
                    ));
                }
            }
        }
        let seqno = match (&self.oracle, commit_ts) {
            (Some(_), ts) if ts > 0 => ts,
            _ => self.next_seqno(),
        };
        batch.commit_covered(seqno, cover)?;
        if let Some(oracle) = &self.oracle {
            oracle.advance_to(coordinode_core::txn::timestamp::Timestamp::from_raw(seqno));
            // The commit path is the retention window's clock tick: with the
            // oracle moved past this commit the window floor moves too, so a
            // compaction that follows sees the current horizon without a
            // background timer. One uncontended lock per proposal.
            self.coordinator.advance_gc_watermark();
        }
        // Only now does the engine's read point include `seqno` (an entry
        // applied on a follower can carry a commit timestamp ahead of its
        // clock): invalidated earlier, a read in between would cache the
        // values this commit replaced.
        batch.forget_cached();
        Ok(())
    }

    /// Bulk-delete all keys in a range by dropping entire LSM tables.
    ///
    /// This is a table-level operation — far more efficient than individual
    /// deletes for large contiguous key ranges. Tables fully contained within
    /// the range are dropped; partially overlapping tables are untouched
    /// (their keys remain until compaction handles them).
    ///
    /// Use cases: TTL cleanup, cascading graph deletes, index drop.
    ///
    /// Invalidates tiered cache for the entire partition (range unknown to cache layer).
    pub fn drop_range<K: AsRef<[u8]>, R: std::ops::RangeBounds<K>>(
        &self,
        part: Partition,
        range: R,
    ) -> StorageResult<()> {
        let tree = self.tree(part)?;
        tree.drop_range(range)?;
        self.write_taps.replaced(part);
        // A read is answered from the cache before the tree, so a dropped
        // key's cached value would outlive the drop.
        if let Some(cache) = &self.tiered_cache {
            cache.clear_partition(part);
        }
        Ok(())
    }

    /// Physically reset a partition to empty: install a fresh empty version and
    /// delete every obsolete table / blob file WITHOUT reading their blocks.
    ///
    /// Unlike [`drop_range`](Self::drop_range) over the full range — which runs a
    /// drop-range *compaction* that reads the tables it rewrites — `clear` never
    /// touches block contents. That distinction is load-bearing for repair: when
    /// a partition is being rebuilt because one of its SSTs is corrupt, a
    /// compaction would re-read (and trip on) the very block we are discarding.
    /// Use this before reinstalling a known-good base.
    pub fn clear_partition(&self, part: Partition) -> StorageResult<()> {
        let tree = self.tree(part)?;
        tree.clear()?;
        self.write_taps.replaced(part);
        if let Some(cache) = &self.tiered_cache {
            cache.clear_partition(part);
        }
        Ok(())
    }

    /// The invariant-claim table shared by every attempt on this engine.
    ///
    /// Shared rather than per-transaction: a condition is only protected if
    /// the attempt that would break it consults the same table.
    pub fn claim_registry(&self) -> &crate::engine::claims::ClaimRegistry {
        &self.claim_registry
    }

    /// The commits admitted on this node and not yet applied.
    ///
    /// A writer registers its scope here before it validates, so no interval
    /// exists in which another writer sees neither its effect nor the
    /// obligation to account for it. A reader consults the same table to know
    /// whether a snapshot is complete.
    pub fn pending_commits(&self) -> &Arc<crate::engine::pending::PendingCommits> {
        &self.pending_commits
    }

    /// The schema generation as it stands now. An attempt reads this once and
    /// stamps it on every claim it makes.
    pub fn schema_generation(&self) -> u64 {
        self.schema_generation
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Count one more property delta written to `node_key` and return the
    /// run it makes since the key's last whole write.
    pub fn note_node_delta(&self, node_key: &[u8]) -> u32 {
        /// Keys tracked at once; past it the table starts over, which only
        /// makes some runs look shorter than they are.
        const MAX_TRACKED: usize = 1 << 16;
        let mut runs = self.node_delta_runs.lock();
        if runs.len() >= MAX_TRACKED && !runs.contains_key(node_key) {
            runs.clear();
        }
        let run = runs.entry(node_key.to_vec()).or_insert(0);
        // Bounded far below u32::MAX: a writer ends the run with a whole
        // write once it reaches its limit.
        *run += 1;
        *run
    }

    /// The property deltas written to `node_key` since its last whole write.
    pub fn node_delta_run(&self, node_key: &[u8]) -> u32 {
        self.node_delta_runs
            .lock()
            .get(node_key)
            .copied()
            .unwrap_or(0)
    }

    /// `node_key` was written whole: its run of deltas is over.
    pub fn note_node_whole_write(&self, node_key: &[u8]) {
        self.node_delta_runs.lock().remove(node_key);
    }

    /// Record that a schema definition changed, so that predicates evaluated
    /// before the change stop counting as evidence about the graph after it.
    ///
    /// Called when the change lands, not when it is staged: a definition that
    /// never commits never invalidated anything.
    pub fn note_schema_change(&self) {
        self.schema_generation
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    /// The shard whose node rows this engine holds.
    pub fn node_shard(&self) -> u16 {
        self.node_shard.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Tell the engine which shard's node rows it holds.
    ///
    /// Called by the handle above once it is constructed; the configured value
    /// stands until then.
    pub fn set_node_shard(&self, shard: u16) {
        self.node_shard
            .store(shard, std::sync::atomic::Ordering::Release);
    }

    /// Check if a key exists in the given partition.
    pub fn contains_key(&self, part: Partition, key: &[u8]) -> StorageResult<bool> {
        let tree = self.tree(part)?;
        let value = tree.get(key, self.coordinator.current_seqno())?;
        Ok(value.is_some())
    }

    /// Force a major compaction on the named partition tree.
    ///
    /// Drives the lsm-tree compactor to push all SSTs down to the
    /// bottom level. Under multi-endpoint per-LSM-level routing this
    /// physically moves data from the upper-level (hot-tier) endpoints
    /// to the bottom-level (cold-tier) endpoint — the underlying
    /// mechanism behind cascade eviction.
    ///
    /// Blocks the caller until compaction finishes. Intended for
    /// operator-driven maintenance, capacity-pressure cascade eviction,
    /// and end-to-end tests; not used on the steady-state write path.
    pub fn major_compact(&self, part: Partition) -> StorageResult<()> {
        let tree = self.tree(part)?;
        // Target table size of 64 MiB is the lsm-tree default — picked
        // here explicitly so the major compaction's output SST size is
        // independent of any per-partition tuning we may layer on later.
        //
        // `seqno_threshold` is the GC watermark: everything BELOW it may be
        // folded, collected and have its seqno zeroed. Passing `SeqNo::MAX`
        // here (as this did until 2026-09-08) reads as "keep everything" but
        // means the opposite — it collects every superseded version whatever
        // the retention policy says, zeroes the bottommost seqnos, and, since
        // coordinode-lsm-tree 5.8.6, records a retention floor at the
        // compaction's install seqno that refuses every snapshot at or below
        // it after a reopen.
        //
        // What "as much as is legitimate" means depends on whether the engine
        // owes anyone a time-travel contract; see
        // `GcWatermarkController::maintenance_compaction_threshold`.
        let threshold = self.coordinator.maintenance_compaction_threshold_for(part);
        tree.major_compact(64 * 1024 * 1024, threshold)
            .map_err(|e| StorageError::Io(format!("major compact {}: {e}", part.name())))?;
        Ok(())
    }

    /// Report from a single cascade-eviction invocation.
    ///
    /// `compacted_partitions` counts the partition trees whose routing
    /// referenced the saturated endpoint and for which a major
    /// compaction was therefore triggered.
    ///
    /// Trigger a cascade eviction for the named endpoint.
    ///
    /// For each non-Schema partition whose persisted routing references
    /// `endpoint_id`, fire a major compaction on its tree. Major
    /// compaction pushes SSTs down through the LSM level hierarchy,
    /// and because per-LSM-level routing places the bottom levels on
    /// cooler endpoints, the net effect is to move data off the
    /// saturated endpoint and onto its next-cooler neighbour.
    ///
    /// This is the **mechanism** consumed by the capacity-tracking
    /// layer (the auto-trigger at 95% of `hard_limit_bytes`); the
    /// caller wires the detection loop.
    ///
    /// Returns `Ok(report)` with the number of partitions touched,
    /// even when zero (the named endpoint may not host any partition's
    /// data).
    pub fn cascade_evict_endpoint(&self, endpoint_id: &str) -> StorageResult<CascadeReport> {
        run_cascade_evict(
            &self.endpoints,
            self.coordinator.trees(),
            self.coordinator.seqno_generator(),
            &|part| self.coordinator.maintenance_compaction_threshold_for(part),
            endpoint_id,
        )
    }

    /// Flushes the active memtable of every partition tree to an SST file.
    /// SST files are written atomically (atomic rename), so this provides
    /// crash safety without requiring a separate WAL fsync.
    ///
    /// When a standalone WAL is active, a WAL checkpoint (rotation) is
    /// performed after the SST flush.  This keeps the WAL small: all data
    /// that was in the WAL is now in SST, so the journal can be truncated.
    pub fn persist(&self) -> StorageResult<()> {
        for tree in self.coordinator.trees().values() {
            tree.flush_active_memtable(0)?;
        }
        #[cfg(feature = "columnar")]
        for table_id in self.columnar_tables.table_ids() {
            if let Some(tree) = self.columnar_tables.get(&table_id) {
                tree.flush_active_memtable(0)?;
            }
        }
        // The retained oplog journal is NOT truncated on flush — it must survive
        // for WAL-replay-repair, and crash recovery skips the entries each
        // partition's coverage record already holds (see `open_embedded`).
        Ok(())
    }

    /// [`persist`](Self::persist) for one partition: every write to `part`
    /// made before the call is durable when it returns. For state that has
    /// no journal behind it, such as a Raft vote.
    ///
    /// # Errors
    ///
    /// The partition does not exist or its flush fails.
    pub fn persist_partition(&self, part: Partition) -> StorageResult<()> {
        self.tree(part)?.flush_active_memtable(0)?;
        Ok(())
    }

    /// Journal a proposal without applying it, leaving the journal as its
    /// only copy. Returns `Some(index)` if a record was written, `None` when
    /// no journal is configured.
    #[cfg(test)]
    pub(crate) fn oplog_append(
        &self,
        mutations: &[Mutation],
        commit_ts: u64,
    ) -> StorageResult<Option<u64>> {
        match &self.oplog {
            None => Ok(None),
            Some(oplog) => {
                let mut guard = oplog
                    .lock()
                    .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
                let index = guard.append(mutations, commit_ts)?;
                Ok(Some(index))
            }
        }
    }

    /// Return `true` if a retained embedded oplog journal is active.
    pub fn has_journal(&self) -> bool {
        self.oplog.is_some()
    }

    /// The engine's current sequence number: every write already applied has a
    /// seqno at or below it, and every later write a higher one.
    ///
    /// Read it before starting work that must later fold in "everything since",
    /// then feed it to [`Self::changed_keys_since`]. Saves callers from
    /// importing the Layer-3 coordinator trait to reach the same value.
    pub fn current_seqno(&self) -> lsm_tree::SeqNo {
        use crate::engine::coordinator::MultiModalCoordinator as _;
        self.coordinator.current_seqno()
    }

    /// Approximate live item count in a partition.
    ///
    /// Read straight off the LSM metadata, so it costs no scan — but it counts
    /// every version and tombstone the levels still hold, and knows nothing of
    /// labels or shards. It is an upper bound suitable for progress bars and
    /// ETAs, never for an answer a caller might present as a row count.
    pub fn approximate_len(&self, part: Partition) -> StorageResult<usize> {
        Ok(self.tree(part)?.approximate_len())
    }

    /// Get approximate disk space used by the engine in bytes.
    pub fn disk_space(&self) -> StorageResult<u64> {
        Ok(self
            .coordinator
            .trees()
            .values()
            .map(|t| t.disk_space())
            .sum())
    }

    /// Get the configured flush policy.
    pub fn flush_policy(&self) -> FlushPolicy {
        self.flush_policy
    }

    /// The oplog settings (segment rotation, retention, sync method) this
    /// engine was opened with: a log opened over the engine uses them.
    pub fn oplog_config(&self) -> &OplogJournalConfig {
        &self.oplog_config
    }

    /// Get the shared block cache.
    pub fn cache(&self) -> &Arc<lsm_tree::Cache> {
        self.coordinator.cache()
    }

    /// Force a major compaction on a specific partition.
    ///
    /// Flushes the memtable to SST first (compaction only sees SST data), then
    /// runs a major compaction that folds versions below the GC watermark.
    ///
    /// In production, compaction runs automatically in the background.
    /// This method is primarily for testing and manual maintenance.
    pub fn force_compaction(&self, part: Partition) -> StorageResult<()> {
        let tree = self.tree(part)?;
        // Flush memtable so compaction sees the latest data.
        tree.flush_active_memtable(0)?;
        // Compact/GC up to the CURRENT watermark only — never advance it here.
        // The watermark is the seqno below which no reader needs history; an
        // unpinned time-travel / AS OF read holds an older seqno the controller
        // cannot see, so forcing the watermark forward would collect state it
        // still observes. `seqno_threshold` is the fold/GC boundary: versions
        // below it may go, everything at or above it is preserved.
        //
        // A system partition has no such reader, so its threshold is brought
        // up to the live pins first: its own writes (this node's Raft state)
        // do not pass through the commit path that republishes it.
        if part.is_system() {
            self.coordinator.advance_gc_watermark();
        }
        let watermark = self.coordinator.gc_watermark_value_for(part);
        tree.major_compact(u64::MAX, watermark)?;
        // Operand fold for commutative partitions is time-travel safe: it
        // rewrites each key's merged value with `put`, adding a new version on
        // top while leaving the older operands intact for reads at older
        // seqnos. It collapses the per-read operand cost without GC-ing history.
        if part.is_commutative() {
            self.collapse_merge_operands(part)?;
        }
        Ok(())
    }

    /// Fold accumulated merge operands in a commutative partition into one
    /// stored value per key in a single pass, returning the number of keys
    /// rewritten.
    ///
    /// Each `merge()` write appends an operand; a key touched `N` times carries
    /// `N` operands the merge operator re-applies on *every* read (`O(N)` per
    /// read). The background compactor folds these as the GC watermark advances,
    /// but convergence takes several passes; this reads each key once (folding
    /// the operands) and writes the folded value back with `put`, collapsing the
    /// chain to a single base value immediately. Use after a bulk load.
    pub fn collapse_merge_operands(&self, part: Partition) -> StorageResult<usize> {
        // Snapshot the merged state first; writing while the scan iterator is
        // live would re-read keys this pass has already rewritten.
        let mut folded: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for guard in self.prefix_scan(part, b"")? {
            let (key, value) = guard.into_inner()?;
            folded.push((key.to_vec(), value.to_vec()));
        }
        let rewritten = folded.len();
        for (key, value) in folded {
            self.put(part, &key, &value)?;
        }
        Ok(rewritten)
    }

    /// Scan all key-value pairs in a partition whose keys start with the given prefix.
    ///
    /// Returns an iterator of `IterGuardImpl` items. Use `guard.into_inner()`
    /// to get `(UserKey, UserValue)`.
    pub fn prefix_scan(&self, part: Partition, prefix: &[u8]) -> StorageResult<StorageIter> {
        let tree = self.tree(part)?;
        let seqno = self.coordinator.current_seqno();
        Ok(coverage::user_prefix(tree, prefix, seqno))
    }

    /// Whether this store holds any data of its own: one live key under any
    /// partition's [user-data prefix](Partition::user_data_prefix).
    ///
    /// Consensus state, partition routing and internal bookkeeping do not
    /// count, so a node that has only ever been started answers `false`
    /// however many times it was restarted. This is the question a node has
    /// to answer before it joins a group that already holds data: only the
    /// member a group is formed around brings data into it, and a joining
    /// node that brought its own would have to lose it silently.
    pub fn holds_user_data(&self) -> StorageResult<bool> {
        for &part in Partition::all() {
            let Some(prefix) = part.user_data_prefix() else {
                continue;
            };
            if self.prefix_scan(part, prefix)?.next().is_some() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Prefix scan in descending key order (high to low) — the reverse-iteration
    /// counterpart of [`Self::prefix_scan`], walking the same double-ended LSM
    /// iterator from its high end. A "latest" / "last N within a prefix"
    /// consumer reads from the top and stops early instead of scanning the whole
    /// prefix and sorting.
    pub fn prefix_scan_rev(&self, part: Partition, prefix: &[u8]) -> StorageResult<StorageIter> {
        let tree = self.tree(part)?;
        let seqno = self.coordinator.current_seqno();
        Ok(Box::new(coverage::user_prefix(tree, prefix, seqno).rev()))
    }

    /// Keys touched (written, merged, or deleted) at or after `since_seqno`
    /// (inclusive), deduplicated and sorted. Inclusive pairs with the snapshot
    /// convention (a snapshot at `S` sees versions strictly below `S`): the
    /// seqno a snapshot or cursor was taken at is exactly the first one it has
    /// not seen, so resuming from it misses nothing; a consumer that has
    /// processed a commit landed at `T` resumes from `T + 1`. O(delta): the
    /// lsm-tree surfaces only the keys whose version history reached
    /// `since_seqno`, instead of scanning the whole partition twice and
    /// diffing.
    ///
    /// Values are intentionally NOT returned — the caller re-reads the merged
    /// current value per key, so a key that accumulated merge operands (adj
    /// posting list, counter delta) is captured as its resolved state, not raw
    /// operands. Dispatches over `AnyTree`: KV-separated (blob) partitions use
    /// the blob scan path that resolves indirected values.
    pub fn changed_keys_since(
        &self,
        part: Partition,
        since_seqno: u64,
    ) -> StorageResult<Vec<Vec<u8>>> {
        use lsm_tree::{AnyTree, ScanSinceEvent};

        let mut keys: Vec<Vec<u8>> = Vec::new();
        let mut collect = |ev: ScanSinceEvent| -> StorageResult<()> {
            match ev {
                // A weak (single-delete) tombstone still means the key's
                // history advanced past `since_seqno`; for a changed-keys
                // consumer it is indistinguishable from a regular delete,
                // since the caller re-reads the merged current value per key.
                ScanSinceEvent::Insert { key, .. }
                | ScanSinceEvent::MergeOperand { key, .. }
                | ScanSinceEvent::PointTombstone { key, .. }
                | ScanSinceEvent::WeakTombstone { key, .. } => {
                    if !coverage::is_reserved(&key) {
                        keys.push(key.to_vec());
                    }
                    Ok(())
                }
                // The coverage fold's tombstone lies wholly in the reserved
                // namespace and touches no user key.
                ScanSinceEvent::RangeTombstone { end_key, .. }
                    if end_key.as_ref() <= coverage::USER_KEYSPACE_START =>
                {
                    Ok(())
                }
                // CoordiNode's Mutation set is Put / Delete / Merge only — it
                // never issues range deletes, so this is unreachable for our
                // data. Surface it loudly rather than silently miss keys.
                ScanSinceEvent::RangeTombstone { .. } => Err(StorageError::Io(
                    "range tombstone in scan_since_seqno: CoordiNode issues no range deletes"
                        .to_string(),
                )),
            }
        };
        match self.tree(part)? {
            AnyTree::Standard(t) => {
                for ev in t.scan_since_seqno(since_seqno).map_err(|e| {
                    StorageError::Io(format!("scan_since_seqno {}: {e}", part.name()))
                })? {
                    collect(ev)?;
                }
            }
            AnyTree::Blob(bt) => {
                for ev in bt.scan_since_seqno(since_seqno).map_err(|e| {
                    StorageError::Io(format!("scan_since_seqno {}: {e}", part.name()))
                })? {
                    collect(ev)?;
                }
            }
        }
        keys.sort_unstable();
        keys.dedup();
        Ok(keys)
    }

    /// Scan key-value pairs visible at a specific sequence number.
    ///
    /// Like [`Self::prefix_scan`], but reads at an arbitrary point-in-time.
    pub fn prefix_scan_at(
        &self,
        part: Partition,
        prefix: &[u8],
        seqno: lsm_tree::SeqNo,
    ) -> StorageResult<StorageIter> {
        self.check_snapshot_retained(seqno)?;
        let tree = self.tree(part)?;
        Ok(coverage::user_prefix(tree, prefix, seqno))
    }

    /// Inclusive-bounded range scan: yields entries with keys `K` such
    /// that `start ≤ K ≤ end`. Convenience wrapper over
    /// [`MultiModalCoordinator::range_scan`]; used by callers that have
    /// decomposed a query into disjoint key intervals (e.g. spatial
    /// Z-curve subrange decomposition).
    pub fn range_scan(
        &self,
        part: Partition,
        start: &[u8],
        end: &[u8],
    ) -> StorageResult<StorageIter> {
        self.coordinator.range_scan(part, start, end)
    }

    /// Inclusive-bounded range scan in descending key order (high to low) — the
    /// reverse-iteration counterpart of [`Self::range_scan`]. A
    /// `descending … LIMIT n` consumer takes `n` from the high end and stops,
    /// avoiding a full forward scan + in-memory sort. Same `[start, end]`
    /// inclusive bounds.
    pub fn range_scan_rev(
        &self,
        part: Partition,
        start: &[u8],
        end: &[u8],
    ) -> StorageResult<StorageIter> {
        self.coordinator.range_scan_rev(part, start, end)
    }

    /// Seekable range scan over `[start, end]` at `seqno`. The returned iterator
    /// can `seek_to` an arbitrary key mid-walk, so one open iterator skips the
    /// dead bytes between disjoint subranges without reopening per-SST readers
    /// (spatial Z-curve skip-scan). `seqno` pins the read snapshot.
    pub fn range_seekable(
        &self,
        part: Partition,
        start: &[u8],
        end: &[u8],
        seqno: lsm_tree::SeqNo,
    ) -> StorageResult<SeekableStorageIter> {
        self.check_snapshot_retained(seqno)?;
        self.coordinator.range_seekable(part, start, end, seqno)
    }

    /// Get the tiered cache, if enabled.
    pub fn tiered_cache(&self) -> Option<&TieredCache> {
        self.tiered_cache.as_ref()
    }

    /// Get the access tracker.
    pub fn access_tracker(&self) -> &AccessTracker {
        &self.access_tracker
    }

    /// Create a new write batch for atomic, crash-safe mutations.
    pub fn write_batch(&self) -> WriteBatch<'_> {
        WriteBatch::new(self)
    }

    /// Take a snapshot of the current sequence number.
    ///
    /// A snapshot that is complete: everything below it has landed.
    ///
    /// The engine's own counter is not that. A commit takes its timestamp
    /// before it applies, so between the two there is a number the counter has
    /// already passed and whose effect is nowhere to be seen. A reader handed
    /// that number sees neither the write nor any sign that one is coming, and
    /// a writer built on such a read validates against a state that is missing
    /// the very commit it is racing: it finds nothing written, commits, and
    /// replaces an update that was never visible to it.
    ///
    /// So the snapshot stops below the oldest commit still in flight. Reads
    /// see sequence numbers strictly below it, so that commit is excluded
    /// rather than half-present, and a writer starting there is told about the
    /// conflict when it validates. The cost is that a snapshot taken during a
    /// commit lags by that commit, which is the definition of complete rather
    /// than a delay added on top of it.
    pub fn snapshot(&self) -> lsm_tree::SeqNo {
        self.pending_commits.complete_snapshot(
            || self.coordinator.snapshot(),
            std::time::Duration::from_millis(
                self.snapshot_wait
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        )
    }

    /// The last commit timestamp whose every write, and every write below
    /// it, is applied here: a cut a derived structure that has folded what
    /// it took can claim. `None` while nothing is known to be complete.
    ///
    /// A standalone store learns of every commit before it lands, so its
    /// [`Self::snapshot`] is such a cut. A Raft member also applies commits
    /// stamped by a leader in log order, which need not be timestamp order:
    /// a commit stamped lower can still be on its way behind one stamped
    /// higher. Its cut is bounded by the closed bound of the entries applied
    /// here, below which every commit is already in the log before them.
    pub fn complete_cut(&self) -> Option<u64> {
        let snapshot = self.snapshot();
        let bound = if self.raft_fence().is_some() {
            snapshot.min(self.closure_frontier())
        } else {
            snapshot
        };
        bound.checked_sub(1)
    }

    /// The closed bound of the Raft entries applied here; see
    /// [`Self::complete_cut`].
    pub fn closure_frontier(&self) -> u64 {
        self.closure_frontier
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Raise the closed bound to `below` once the entry carrying it has
    /// applied, its [`CLOSURE_KEY`] record included. Notify every partition's
    /// applied consumers afterwards, even if the entry changed no keys there.
    pub fn raise_closure_frontier(&self, below: u64, index: u64) {
        debug_assert!(below > 0);
        let previous = self
            .closure_frontier
            .fetch_max(below, std::sync::atomic::Ordering::AcqRel);
        if below > previous {
            self.applied_feed.closed(index, below - 1);
        }
    }

    /// Read the closed bound from its stored record: when the store opens
    /// and when a snapshot replaced it. A store whose applied entries carried
    /// none knows of no bound.
    ///
    /// # Errors
    ///
    /// The record cannot be read or is malformed.
    pub fn reload_closure_frontier(&self) -> StorageResult<()> {
        let below = match self.get(Partition::Schema, CLOSURE_KEY)? {
            Some(bytes) => decode_closure(&bytes)?,
            None => 0,
        };
        self.closure_frontier
            .store(below, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Open a tap on `partition` and return it with a snapshot: every write
    /// to the partition is either visible at the snapshot or delivered by
    /// the tap, whatever timestamp it lands at.
    ///
    /// A write finishing before the tap opens is at or below the returned
    /// seqno. On a Raft store an apply lands at the leader's timestamp,
    /// which can be above this node's clock until the apply advances it, so
    /// the snapshot is read while no entry applies.
    ///
    /// Pin the snapshot (with [`Self::pin_snapshot_at`]) before reading at
    /// it. Blocks on a Raft store; call it off the async runtime.
    ///
    /// # Errors
    ///
    /// The errors of pausing the Raft applies.
    pub fn tap_writes(
        &self,
        partition: Partition,
    ) -> StorageResult<(crate::engine::tap::WriteTap, lsm_tree::SeqNo)> {
        let tap = self.write_taps.open(partition);
        let at = self.tap_snapshot()?;
        Ok((tap, at))
    }

    /// Follow the commits applied to `partition` from now on, queueing at
    /// most `capacity` events; see [`crate::engine::applied`].
    ///
    /// Reported, after they are in the store: the entries the Raft state
    /// machine applies through [`Self::apply_raft_proposal`] (with their log
    /// index), and the local commits of an engine without Raft through
    /// [`Self::commit_journaled`] (with their journal index) or
    /// [`Self::apply_proposal_at`] (with index 0).
    pub fn subscribe_applied(
        &self,
        partition: Partition,
        capacity: usize,
    ) -> crate::engine::applied::AppliedSubscription {
        self.applied_feed.subscribe(partition, capacity, false)
    }

    /// [`Self::subscribe_applied`] whose events stay after they are handed
    /// out until the consumer releases them
    /// ([`crate::engine::applied::AppliedPosition::release`]), counting
    /// toward `capacity` meanwhile: readers of the consumer's state learn
    /// which keys it has not folded yet from
    /// [`crate::engine::applied::AppliedPosition::pending`].
    pub fn subscribe_applied_retained(
        &self,
        partition: Partition,
        capacity: usize,
    ) -> crate::engine::applied::AppliedSubscription {
        self.applied_feed.subscribe(partition, capacity, true)
    }

    /// How many applied commits a derived-index worker's retained
    /// subscription holds ([`StorageConfig::index_feed_capacity`]).
    pub fn index_feed_capacity(&self) -> usize {
        self.index_feed_capacity
    }

    /// Run `hook` between each entry's store write and its publication to
    /// the subscribers, so a test can look at what a reader sees then.
    #[cfg(test)]
    pub(crate) fn pause_applies_after_write(&self, hook: Arc<dyn Fn() + Send + Sync>) {
        *self.applied_feed.between_write_and_publish.lock() = Some(hook);
    }

    /// Start `tap` over after it reported
    /// [`Tapped::Replaced`](crate::engine::tap::Tapped::Replaced): discard
    /// what it holds and return a fresh snapshot with the guarantee of
    /// [`Self::tap_writes`].
    ///
    /// # Errors
    ///
    /// The errors of pausing the Raft applies.
    pub fn rebase_tap(&self, tap: &crate::engine::tap::WriteTap) -> StorageResult<lsm_tree::SeqNo> {
        tap.reset();
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        self.tap_snapshot()
    }

    fn tap_snapshot(&self) -> StorageResult<lsm_tree::SeqNo> {
        use crate::engine::coordinator::MultiModalCoordinator as _;
        let Some(fence) = self.raft_fence() else {
            return Ok(self.coordinator.snapshot());
        };
        let mut at = 0;
        fence.with_applies_paused(&mut |_, _| {
            at = self.coordinator.snapshot();
            Ok(())
        })?;
        Ok(at)
    }

    /// Allocate a seqno and return it: every snapshot taken before this
    /// call is at or below it, every one taken after is above it. Pair with
    /// [`Self::await_transactions_through`] to wait for the transactions
    /// already running when something changed.
    pub fn snapshot_boundary(&self) -> lsm_tree::SeqNo {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let boundary = self.next_seqno();
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        boundary
    }

    /// Wait until no transaction opened at or before `boundary` (a value of
    /// [`Self::snapshot_boundary`]) is still open, polling every `poll`, for
    /// at most `timeout`. Returns how many are still open when the time ran
    /// out. A caller waiting from inside a transaction of its own leaves the
    /// count first ([`Transaction::release_from_schema_waits`](crate::engine::transaction::Transaction::release_from_schema_waits)).
    ///
    /// Only transactions are waited for: a long-lived reader (a backup, a
    /// CDC consumer) writes nothing and is not one.
    pub fn await_transactions_through(
        &self,
        boundary: lsm_tree::SeqNo,
        poll: std::time::Duration,
        timeout: std::time::Duration,
    ) -> Result<(), usize> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let open = self.open_transactions.open_through(boundary);
            if open == 0 {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return Err(open);
            }
            std::thread::sleep(poll);
        }
    }

    /// The taps open on this engine, for the write paths outside this file.
    pub(crate) fn write_taps(&self) -> &crate::engine::tap::WriteTaps {
        &self.write_taps
    }

    /// How many subscriptions to the applies are open.
    #[cfg(test)]
    pub(crate) fn applied_subscriptions(&self) -> usize {
        self.applied_feed.open()
    }

    /// How long a snapshot waits for the commits still landing.
    pub fn snapshot_wait_ms(&self) -> u64 {
        self.snapshot_wait
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Change that wait without a restart.
    ///
    /// Zero makes a snapshot step behind an unapplied commit immediately
    /// rather than waiting for it: still complete, but older, and a writer
    /// starting from it can conflict with its own last commit.
    pub fn set_snapshot_wait_ms(&self, ms: u64) {
        self.snapshot_wait
            .store(ms, std::sync::atomic::Ordering::Relaxed);
    }

    /// Creates a point-in-time snapshot at a specific sequence number.
    ///
    /// Returns `Some(seqno)` always — lsm-tree handles future seqnos by
    /// returning the latest visible version of each key. Reads through a
    /// snapshot below [`Self::gc_watermark`] fail with
    /// [`StorageError::SnapshotOutsideRetention`]; pin the seqno first
    /// ([`Self::pin_snapshot_at`]) to hold the watermark for a longer read.
    pub fn snapshot_at(&self, seqno: lsm_tree::SeqNo) -> Option<lsm_tree::SeqNo> {
        Some(seqno)
    }

    /// Inspect a value borrowed from its LSM owner at the selected snapshot.
    /// The owner remains alive until `inspect` returns; its borrow cannot
    /// escape in `R`. No payload is copied into an intermediate byte buffer.
    /// Retention and visibility checks are the same as [`Self::snapshot_get`].
    pub fn with_snapshot_value<R>(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        key: &[u8],
        inspect: impl FnOnce(Option<&[u8]>) -> R,
    ) -> StorageResult<R> {
        self.check_snapshot_retained(*snapshot)?;
        let value = self.tree(part)?.get(key, *snapshot)?;
        Ok(inspect(value.as_deref()))
    }

    /// Read a value through a previously taken snapshot.
    ///
    /// Returns the value as it was at the snapshot seqno — writes after
    /// the snapshot are invisible.
    pub fn snapshot_get(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<bytes::Bytes>> {
        self.check_snapshot_retained(*snapshot)?;
        let tree = self.tree(part)?;
        let value = tree.get(key, *snapshot)?;
        Ok(value.map(|v| bytes::Bytes::copy_from_slice(&v)))
    }

    /// Batch point lookup through a previously taken snapshot — the
    /// snapshot-pinned counterpart of [`Self::multi_get`]. Returns a value (or
    /// `None`) per input key in order, fetched with one
    /// [`lsm_tree::AbstractTree::multi_get`] at `snapshot`. Bypasses the tiered
    /// cache (the cache tracks the live seqno, not historical snapshots), so
    /// MVCC reads stay snapshot-consistent.
    pub fn snapshot_multi_get(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        keys: &[&[u8]],
    ) -> StorageResult<Vec<Option<bytes::Bytes>>> {
        self.check_snapshot_retained(*snapshot)?;
        let tree = self.tree(part)?;
        let values = tree.multi_get(keys.iter().copied(), *snapshot)?;
        Ok(values
            .into_iter()
            .map(|v| v.map(|b| bytes::Bytes::copy_from_slice(&b)))
            .collect())
    }

    /// Scan keys by prefix through a previously taken snapshot.
    ///
    /// Returns a vector of (key, value) pairs visible at the snapshot seqno.
    pub fn snapshot_prefix_scan(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        prefix: &[u8],
    ) -> StorageResult<Vec<(Vec<u8>, bytes::Bytes)>> {
        let mut results = Vec::new();
        for guard in self.snapshot_prefix_iter(snapshot, part, prefix)? {
            let (key, value) = guard.into_inner()?;
            results.push((key.to_vec(), bytes::Bytes::copy_from_slice(&value)));
        }
        Ok(results)
    }

    /// Lazy counterpart of [`Self::snapshot_prefix_scan`]: entries are read as
    /// the caller pulls them, so a caller that stops early reads only what it
    /// consumed instead of the whole prefix.
    pub fn snapshot_prefix_iter(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        prefix: &[u8],
    ) -> StorageResult<StorageIter> {
        self.check_snapshot_retained(*snapshot)?;
        let tree = self.tree(part)?;
        Ok(coverage::user_prefix(tree, prefix, *snapshot))
    }

    /// The version of a record: the timestamp of the commit that last wrote
    /// it, or `None` when no live row is there.
    ///
    /// This is the value a caller compares against when it writes on the
    /// condition that nobody else has. It is the commit timestamp rather than
    /// a counter of its own, because the commit that wrote the row is exactly
    /// what a conditional write is asking about, and inventing a second
    /// number would mean keeping two things in step for no gain.
    ///
    /// A commutative partition has no version to give: adjacency and counter
    /// rows are folded from operands rather than replaced, so there is no
    /// single commit that last wrote one. Asking is a caller error rather
    /// than an answer of `None`, which would read as "nothing is there".
    pub fn record_version(
        &self,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<lsm_tree::SeqNo>> {
        if part.is_commutative() {
            return Err(StorageError::InvalidConfig(format!(
                "the {part:?} partition is merge-composed: its rows are folded from \
                 operands, so no single commit is their version"
            )));
        }
        let tree = self.tree(part)?;
        Ok(tree
            .get_internal_entry(key, lsm_tree::SeqNo::MAX)?
            .map(|e| e.key.seqno))
    }

    /// A record's latest value together with its [`Self::record_version`],
    /// both of one write: what a read-modify-write states as its condition.
    /// Reading the value and the version separately could pair a version
    /// with an older value when a commit lands between the two reads, and a
    /// write conditioned on that version would then pass on stale data.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::record_version`].
    pub fn get_versioned(
        &self,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<(bytes::Bytes, lsm_tree::SeqNo)>> {
        // The latest value, read between two reads of the version: equal
        // versions mean no write landed between, so the value is that
        // write's. The value is read at the present, never at the version
        // itself: a record last written before the retention horizon is still
        // the latest state, and compaction may have settled its version to
        // zero, so a read at the version would be a refused read in the past.
        let tree = self.tree(part)?;
        loop {
            let Some(version) = self.record_version(part, key)? else {
                return Ok(None);
            };
            let value = tree.get(key, lsm_tree::SeqNo::MAX)?;
            if self.record_version(part, key)? != Some(version) {
                continue;
            }
            return Ok(value.map(|value| (bytes::Bytes::copy_from_slice(&value), version)));
        }
    }

    /// Whether `key` was written by anything this snapshot does not already
    /// include.
    ///
    /// The bound is inclusive, unlike [`Self::has_write_after`], because a
    /// snapshot names the first seqno it does *not* see: a write landing
    /// immediately after the snapshot was taken carries exactly that number,
    /// and a strict comparison would call the one write the view certainly
    /// missed invisible. Callers that hold an oracle timestamp rather than a
    /// snapshot want the strict form; callers holding a snapshot want this.
    pub fn written_since_snapshot(
        &self,
        part: Partition,
        key: &[u8],
        snapshot: lsm_tree::SeqNo,
    ) -> StorageResult<bool> {
        let tree = self.tree(part)?;
        match tree.get_internal_entry(key, lsm_tree::SeqNo::MAX)? {
            Some(e) if e.key.seqno >= snapshot => Ok(true),
            Some(_) => Ok(false),
            // Nothing live now: it was taken away since the view if the view
            // could still see it.
            None => Ok(self.snapshot_get(&snapshot, part, key)?.is_some()),
        }
    }

    /// Check if a key has been written after the given sequence number.
    ///
    /// Returns `true` if the latest version of `key` in `part` has a seqno
    /// strictly greater than `after_seqno`. Used by OCC conflict detection:
    /// if another transaction committed a write after our start_ts,
    /// our read is stale and the transaction must abort.
    ///
    /// For deleted keys (tombstones), falls back to snapshot comparison:
    /// if key existed at `after_seqno` but is gone now, a delete happened.
    pub fn has_write_after(
        &self,
        part: Partition,
        key: &[u8],
        after_seqno: lsm_tree::SeqNo,
    ) -> StorageResult<bool> {
        let tree = self.tree(part)?;
        let entry = tree.get_internal_entry(key, lsm_tree::SeqNo::MAX)?;

        match entry {
            // Live entry with newer seqno → write detected.
            Some(e) if e.key.seqno > after_seqno => Ok(true),
            // Live entry at or before our seqno → no conflict.
            Some(_) => Ok(false),
            // No live entry (tombstone or never existed).
            // If key existed at after_seqno, a delete happened since our read.
            None => {
                let old_val = self.snapshot_get(&after_seqno, part, key)?;
                Ok(old_val.is_some())
            }
        }
    }

    /// Resolve cache eviction weight for a value in a given partition.
    fn resolve_cache_weight(cache: &TieredCache, part: Partition, value: &[u8]) -> f32 {
        if part != Partition::Node {
            return 1.0;
        }
        if cache.label_weights_empty() {
            return 1.0;
        }
        if let Ok(record) = coordinode_core::graph::node::NodeRecord::from_msgpack(value) {
            if let Some(label) = record.labels.first() {
                return cache.resolve_weight(label);
            }
        }
        1.0
    }
}

/// Free function form of `StorageEngine::cascade_evict_endpoint` —
/// shared between the engine method and the background scanner's
/// closure. See `StorageEngine::cascade_evict_endpoint` for the
/// contract.
fn run_cascade_evict(
    endpoints: &[crate::engine::config::EndpointConfig],
    trees: &HashMap<Partition, lsm_tree::AnyTree>,
    seqno: &lsm_tree::SharedSequenceNumberGenerator,
    gc_watermark: &dyn Fn(Partition) -> u64,
    endpoint_id: &str,
) -> StorageResult<CascadeReport> {
    use lsm_tree::AbstractTree;

    if !endpoints.iter().any(|e| e.id == endpoint_id) {
        return Err(StorageError::Io(format!(
            "cascade_evict_endpoint: unknown endpoint id {endpoint_id:?}"
        )));
    }

    let schema_tree =
        trees
            .get(&Partition::Schema)
            .ok_or_else(|| StorageError::PartitionNotFound {
                name: Partition::Schema.name().to_string(),
            })?;
    let read_seqno = seqno.get();

    let mut compacted_partitions = 0u32;
    for &part in Partition::all() {
        if part == Partition::Schema || part == Partition::Raft {
            continue;
        }
        let key = routing_key_for(part);
        let bytes = match schema_tree
            .get(&key, read_seqno)
            .map_err(|e| StorageError::Io(format!("schema get routing {}: {e}", part.name())))?
        {
            Some(b) => b,
            None => continue,
        };
        let routing: crate::engine::routing::PartitionRouting = rmp_serde::from_slice(&bytes)
            .map_err(|e| StorageError::Io(format!("decode routing for {}: {e}", part.name())))?;
        if !routing.endpoints_used().iter().any(|id| *id == endpoint_id) {
            continue;
        }
        tracing::info!(
            endpoint = endpoint_id,
            partition = part.name(),
            "cascade eviction: triggering major compaction"
        );
        let tree = trees
            .get(&part)
            .ok_or_else(|| StorageError::PartitionNotFound {
                name: part.name().to_string(),
            })?;
        // The engine's GC watermark for this partition, not `SeqNo::MAX`:
        // eviction moves data between endpoints and must not collect history
        // the retention policy still holds. See `StorageEngine::major_compact`
        // for what the threshold means.
        tree.major_compact(64 * 1024 * 1024, gc_watermark(part))
            .map_err(|e| StorageError::Io(format!("major compact {}: {e}", part.name())))?;
        compacted_partitions += 1;
    }

    Ok(CascadeReport {
        compacted_partitions,
    })
}

/// Publish each partition's footprint: the live version (in-window key
/// versions included, so this is where the retention window shows) next to
/// what is on disk beside it. A folder walk, sampled on the capacity-scan
/// cadence whether or not any endpoint has a capacity limit.
fn publish_footprint(trees: &HashMap<Partition, lsm_tree::AnyTree>) {
    for (part, tree) in trees {
        match crate::engine::retention_stats::retained_history(tree) {
            Ok(history) => {
                metrics::gauge!("coordinode_storage_live_bytes", "partition" => part.name())
                    .set(history.live_bytes as f64);
                metrics::gauge!(
                    "coordinode_storage_retained_history_bytes",
                    "partition" => part.name()
                )
                .set(history.retained_bytes as f64);
            }
            Err(e) => {
                tracing::warn!(partition = part.name(), error = %e, "retained-history scan failed");
            }
        }
    }
}

/// Publish the stored nodes per label (`coordinode_graph_nodes_total`), from
/// the counters the write path keeps on the same transaction as the nodes:
/// one read per label, sampled on the capacity-scan cadence. A counter below
/// zero is not published; no sequence of committed writes leaves one.
fn publish_label_counts(
    trees: &HashMap<Partition, lsm_tree::AnyTree>,
    seqno: &lsm_tree::SharedSequenceNumberGenerator,
) {
    use coordinode_core::graph::stats::LABEL_KEY_PREFIX;
    use lsm_tree::{AbstractTree, Guard as _};

    let Some(tree) = trees.get(&Partition::Counter) else {
        return;
    };
    for guard in tree.prefix(LABEL_KEY_PREFIX, seqno.get(), None) {
        let (key, value) = match guard.into_inner() {
            Ok(kv) => kv,
            Err(e) => {
                tracing::warn!(error = %e, "label counter scan failed");
                return;
            }
        };
        let Ok(label) = std::str::from_utf8(&key[LABEL_KEY_PREFIX.len()..]) else {
            continue;
        };
        match crate::engine::merge::decode_counter(&value) {
            Ok(count) if count >= 0 => {
                metrics::gauge!("coordinode_graph_nodes_total", "label" => label.to_owned())
                    .set(count as f64);
            }
            Ok(count) => {
                tracing::warn!(label, count, "label counter below zero; not published");
            }
            Err(e) => {
                tracing::warn!(label, error = %e, "label counter does not decode");
            }
        }
    }
}

/// Free function form of `StorageEngine::refresh_capacity` — the
/// scan + persist + auto-cascade pipeline. Pulled out of the method
/// so the background `CapacityScanner` can drive it via a closure
/// without needing an `Arc<StorageEngine>` (which would create a
/// reference cycle with the scanner field).
///
/// `cascade_fn` is the cascade-eviction callback. The engine method
/// passes `|id| self.cascade_evict_endpoint(id)`; the scanner closure
/// passes a snapshot-based callback (see `finish_open`).
fn run_capacity_refresh<F>(
    capacity: &crate::engine::capacity::CapacityTracker,
    endpoints: &[crate::engine::config::EndpointConfig],
    trees: &HashMap<Partition, lsm_tree::AnyTree>,
    seqno: &lsm_tree::SharedSequenceNumberGenerator,
    mut cascade_fn: F,
) where
    F: FnMut(&str) -> StorageResult<CascadeReport>,
{
    let endpoint_paths: std::collections::BTreeMap<String, PathBuf> = endpoints
        .iter()
        .map(|ep| (ep.id.clone(), ep.path.clone()))
        .collect();
    let partition_names: Vec<&str> = Partition::all()
        .iter()
        .filter(|p| **p != Partition::Raft)
        .map(|p| p.name())
        .collect();
    capacity.refresh(&endpoint_paths, &partition_names);
    publish_footprint(trees);

    // Persist used_bytes snapshots to Schema for warm-load on the
    // next engine open, only when the value moved: an unchanged
    // snapshot rewritten every scan keeps the Schema memtable from
    // ever emptying, which on an idle engine means a flush, a table and
    // a compaction per memtable age for nothing.
    if let Some(schema_tree) = trees.get(&Partition::Schema) {
        use lsm_tree::AbstractTree;
        for (_id, usage) in capacity.iter() {
            let key = capacity_key_for(&usage.id);
            let Ok(encoded) = rmp_serde::to_vec(&usage.used()) else {
                continue;
            };
            let stored = schema_tree.get(&key, lsm_tree::SeqNo::MAX).ok().flatten();
            if stored.as_deref() != Some(encoded.as_slice()) {
                schema_tree.insert(&key, &encoded, seqno.next());
            }
        }
    }

    // Auto-cascade pass: any endpoint at Emergency severity with
    // `CascadeEvict` strategy triggers a cascade eviction. The
    // tracing log of the threshold crossing (emitted from
    // `CapacityTracker::refresh`) immediately precedes the eviction's
    // tracing log because both fire in this single function call.
    for (id, usage) in capacity.iter() {
        use crate::engine::capacity::CapacitySeverity;
        use crate::engine::config::HardLimitStrategy;
        // Auto-cascade fires at ≥95% per the storage-stack hard-limit
        // table: Emergency (95-99%) AND Full (100%+) both warrant
        // eviction. Limiting to Emergency-only would miss the common
        // case where writes burst past 100% within one scan interval
        // and severity jumps Normal → Full directly without a
        // visible Emergency band.
        if matches!(usage.strategy, HardLimitStrategy::CascadeEvict)
            && matches!(
                usage.severity(),
                CapacitySeverity::Emergency | CapacitySeverity::Full
            )
        {
            let id = id.to_string();
            metrics::counter!(
                "endpoint_cascade_events_total",
                "endpoint_id" => id.clone(),
            )
            .increment(1);
            if let Err(e) = cascade_fn(&id) {
                tracing::warn!(
                    endpoint = %id,
                    error = %e,
                    "auto cascade eviction failed",
                );
            }
        }
    }
}

/// Encode a capacity-snapshot key for the Schema partition.
///
/// Key format: `meta:capacity:<endpoint_id>` — colon-prefixed to live
/// in the engine-metadata namespace alongside `meta:routing:*`. Value
/// is a MessagePack-encoded `u64` for the most-recent `used_bytes`
/// observation. Lets a fresh engine open warm the capacity tracker
/// to last-known values rather than running every gate against
/// zero-used until the first scan completes.
fn capacity_key_for(endpoint_id: &str) -> Vec<u8> {
    format!("meta:capacity:{endpoint_id}").into_bytes()
}

/// Load the last-known `used_bytes` for an endpoint from Schema. Used
/// at engine open to seed the capacity tracker before the first scan
/// runs. Returns `0` (== "treat as fresh") when no snapshot exists —
/// not an error: a partition with no prior persisted snapshot is the
/// fresh-engine case.
fn load_persisted_capacity(
    schema_tree: &lsm_tree::AnyTree,
    seqno: &lsm_tree::SharedSequenceNumberGenerator,
    endpoint_id: &str,
) -> u64 {
    use lsm_tree::AbstractTree;
    let key = capacity_key_for(endpoint_id);
    match schema_tree.get(&key, seqno.get()) {
        Ok(Some(bytes)) => rmp_serde::from_slice(&bytes).unwrap_or(0),
        _ => 0,
    }
}

/// Outcome of one cascade-eviction invocation.
///
/// `compacted_partitions` is the number of partition trees on which
/// a major compaction was fired in response to the eviction request.
/// A value of zero means the named endpoint did not host any
/// partition's data, which is a valid no-op (e.g. the operator named
/// a hot-only endpoint but no partition had data flushed to it yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CascadeReport {
    /// Count of partition trees whose major compaction was triggered.
    pub compacted_partitions: u32,
}

/// Encode a routing-metadata key for the Schema partition.
///
/// Key format: `meta:routing:<partition_name>` — colon-prefixed so the
/// existing `schema:`-prefixed keys do not collide with routing
/// metadata. Schema partition's docstring reserves the `schema:`
/// namespace; `meta:` is the canonical namespace for engine-level
/// configuration metadata.
fn routing_key_for(partition: Partition) -> Vec<u8> {
    format!("meta:routing:{}", partition.name()).into_bytes()
}

/// Load the persisted [`PartitionRouting`] for a partition from
/// the Schema tree, or initialise (compute default + persist) on first
/// open against this endpoint set.
///
/// **Validation:** persisted routings are validated against the current
/// `endpoints` list. If a previously-referenced endpoint id is missing,
/// returns [`StorageError::Io`] wrapping a `RoutingError::UnknownEndpoint` —
/// the operator removed an endpoint that still hosts SSTs, and continuing
/// would orphan that data when lsm-tree's recovery scan misses it.
///
/// **Persistence format:** MessagePack via `rmp_serde::to_vec` /
/// `from_slice`, matching the rest of the storage layer's serialisation
/// conventions (oplog, WAL records, document snapshots).
///
/// `relocated`: the store's tables were gathered into `endpoints` from
/// another layout, so the persisted routing is replaced by the default.
fn load_or_init_partition_routing(
    schema_tree: &lsm_tree::AnyTree,
    seqno: &lsm_tree::SharedSequenceNumberGenerator,
    endpoints: &[EndpointConfig],
    partition: Partition,
    relocated: bool,
) -> StorageResult<PartitionRouting> {
    use lsm_tree::AbstractTree;
    let key = routing_key_for(partition);
    let read_seqno = seqno.get();
    let persisted = if relocated {
        None
    } else {
        schema_tree.get(&key, read_seqno).map_err(|e| {
            StorageError::Io(format!("schema get routing for {}: {e}", partition.name()))
        })?
    };
    match persisted {
        Some(bytes) => {
            // Existing routing — decode and validate against current
            // endpoint set.
            let routing: PartitionRouting = rmp_serde::from_slice(&bytes).map_err(|e| {
                StorageError::Io(format!(
                    "decode persisted routing for {}: {e}",
                    partition.name()
                ))
            })?;
            routing
                .validate(endpoints)
                .map_err(|e| StorageError::Io(e.to_string()))?;
            Ok(routing)
        }
        None => {
            // First open against this endpoint set, or a relocated store:
            // compute default, persist, return.
            let routing = PartitionRouting::default_for_endpoints(endpoints);
            let encoded = rmp_serde::to_vec(&routing).map_err(|e| {
                StorageError::Io(format!(
                    "encode default routing for {}: {e}",
                    partition.name()
                ))
            })?;
            let write_seqno = seqno.next();
            schema_tree.insert(&key, &encoded, write_seqno);
            tracing::info!(
                partition = partition.name(),
                endpoints = ?routing.endpoints_used(),
                "initialised default per-LSM-level routing"
            );
            Ok(routing)
        }
    }
}

/// Flush all active memtables to SST on clean shutdown.
///
/// coordinode-lsm-tree has no WAL: data in the active memtable is lost if
/// the process exits without flushing. `Drop` performs a best-effort flush
/// so that data written without an explicit `persist()` call still survives
/// a clean shutdown (drop at end of scope, e.g. in tests or graceful server
/// restart). Errors are silently ignored — this is a best-effort safety net,
/// not a crash-recovery guarantee.
///
/// # Drop ordering
///
/// 1. Stop `FlushManager` — joins monitor + worker threads.
/// 2. Stop `CompactionScheduler` — joins monitor + worker threads.
/// 3. Flush active memtables — best-effort final flush.
/// 4. Remaining fields (`trees`, `cache`, etc.) drop naturally after this.
impl Drop for StorageEngine {
    fn drop(&mut self) {
        // Step 0: stop the background capacity scanner. Joins the
        // scanner thread BEFORE other background workers — the
        // scanner only reads tree handles, but ordering is cheap
        // and keeps the shutdown sequence symmetric.
        drop(self.capacity_scanner.take());

        // Step 1: stop background flush workers before touching trees.
        drop(self.flush_manager.take());

        // Step 2: stop background compaction workers.
        drop(self.compaction_scheduler.take());

        // Step 3: best-effort final flush of any remaining active memtable data.
        for tree in self.coordinator.trees().values() {
            let _ = tree.flush_active_memtable(0);
        }
    }
}

/// Outcome of [`StorageEngine::create_checkpoint`].
#[derive(Debug, Default, Clone)]
pub struct CheckpointSummary {
    /// Number of partition trees checkpointed.
    pub partitions: usize,
    /// Sum of `CheckpointInfo.total_bytes` across partition trees: near
    /// zero for an all-hard-link checkpoint, large when cross-fs copy
    /// fall-back fired.
    pub total_bytes: u64,
    /// Bytes copied for the oplog directory: the active segments. Sealed
    /// segments are hard-linked and count only when linking fell back to a
    /// copy across volumes.
    pub oplog_bytes: u64,
    /// Highest captured lsm seqno across partitions — the checkpoint's
    /// logical position, used by PITR to bound oplog replay.
    pub max_seqno: lsm_tree::SeqNo,
}

/// Capture the journal directory `src` into `dst`, returning the bytes
/// copied. A sealed segment is never written again, so it is hard-linked
/// (copied only where the link fails, across volumes); anything else, the
/// active segment above all, is copied. A file that disappears between the
/// listing and its capture is skipped: the journal purge removed it, and it
/// held only entries already in the trees.
fn capture_journal(src: &Path, dst: &Path) -> StorageResult<u64> {
    std::fs::create_dir_all(dst)
        .map_err(|e| StorageError::Io(format!("create dir {dst:?}: {e}")))?;
    let mut bytes = 0u64;
    let entries =
        std::fs::read_dir(src).map_err(|e| StorageError::Io(format!("read dir {src:?}: {e}")))?;
    for entry in entries {
        let entry = entry.map_err(|e| StorageError::Io(format!("dir entry in {src:?}: {e}")))?;
        let file_type = entry
            .file_type()
            .map_err(|e| StorageError::Io(format!("file type {:?}: {e}", entry.path())))?;
        let to = dst.join(entry.file_name());
        if file_type.is_dir() {
            bytes += capture_journal(&entry.path(), &to)?;
        } else {
            let from = entry.path();
            match crate::oplog::segment::is_sealed(&from) {
                Ok(true) => {
                    if std::fs::hard_link(&from, &to).is_ok() {
                        continue;
                    }
                }
                Ok(false) => {}
                Err(_) if !from.exists() => continue,
                Err(e) => return Err(e),
            }
            match std::fs::copy(&from, &to) {
                Ok(copied) => bytes += copied,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(StorageError::Io(format!("copy {from:?}: {e}")));
                }
            }
        }
    }
    Ok(bytes)
}

/// Create the directory a checkpoint is written into, refusing one that
/// already exists.
fn claim_checkpoint_dir(target: &Path) -> StorageResult<()> {
    if target.exists() {
        return Err(StorageError::Io(format!(
            "checkpoint target {target:?} already exists; refusing to overwrite"
        )));
    }
    std::fs::create_dir_all(target)
        .map_err(|e| StorageError::Io(format!("create checkpoint dir {target:?}: {e}")))
}

/// Directory under the data dir holding one empty file per partition whose
/// rebuild is in progress, named after the partition.
const REBUILD_INTENT_DIR: &str = "rebuild";

/// Schema record of the Raft log's closed bound as applied here, written by
/// the apply of the entry carrying it. It is replicated data, so a snapshot
/// carries it and a restart finds it with the entries its tree holds.
pub const CLOSURE_KEY: &[u8] = b"closure:below";

/// The stored form of a closed bound.
#[must_use]
pub fn encode_closure(below: u64) -> Vec<u8> {
    below.to_be_bytes().to_vec()
}

fn decode_closure(bytes: &[u8]) -> StorageResult<u64> {
    let bytes: [u8; 8] = bytes.try_into().map_err(|_| {
        StorageError::Serialization(format!(
            "the closed bound record holds {} bytes, not 8",
            bytes.len()
        ))
    })?;
    Ok(u64::from_be_bytes(bytes))
}

/// A journalled op applied without deriving its entry first: the entry's
/// DERIVED work would otherwise be skipped silently.
fn unresolved_unit() -> StorageError {
    StorageError::InvalidConfig("a journal entry replayed without its unit resolved".into())
}

/// The partitions with a rebuild intent under `data_dir`.
fn read_rebuild_intents(data_dir: &Path) -> StorageResult<Vec<Partition>> {
    let dir = data_dir.join(REBUILD_INTENT_DIR);
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(StorageError::Io(format!("read {dir:?}: {e}"))),
    };
    let mut pending = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| StorageError::Io(format!("entry in {dir:?}: {e}")))?;
        let name = entry.file_name();
        let part = Partition::all()
            .iter()
            .copied()
            .find(|p| name.as_os_str() == p.name())
            .ok_or_else(|| {
                StorageError::Io(format!("unknown rebuild intent {name:?} in {dir:?}"))
            })?;
        pending.push(part);
    }
    Ok(pending)
}

/// Make the entries of `dir` durable.
fn sync_dir(dir: &Path) -> StorageResult<()> {
    lsm_tree::fs::Fs::sync_directory(&lsm_tree::fs::StdFs, dir)
        .map_err(|e| StorageError::Io(format!("sync {dir:?}: {e}")))
}

/// A journalled `STORAGE COLUMNAR` row, held from the journal read until the
/// table registry exists to replay it into.
#[cfg(feature = "columnar")]
struct ColumnarReplay {
    table_id: String,
    key: Vec<u8>,
    value: Vec<u8>,
    ts: u64,
    index: u64,
}

/// Write one `STORAGE COLUMNAR` row at `seqno`, with the coverage marker of
/// its journal entry `index` in the same lsm batch, so the table tree holds
/// both or neither.
#[cfg(feature = "columnar")]
fn apply_columnar_row(
    tree: &lsm_tree::AnyTree,
    key: Vec<u8>,
    value: Vec<u8>,
    seqno: lsm_tree::SeqNo,
    index: Option<u64>,
) -> StorageResult<()> {
    let mut batch = lsm_tree::WriteBatch::with_capacity(2);
    batch.insert(key, value);
    if let Some(index) = index {
        batch.insert(Domain::Journal.marker_key(index, 0).as_slice(), &[][..]);
    }
    tree.apply_batch(batch, seqno)?;
    Ok(())
}

mod raft_coverage;
pub use raft_coverage::{
    PartitionCopy, RaftApplyFence, RaftApplyState, RaftCoverage, RaftHeld, RaftPosition,
    is_node_local,
};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod checkpoint_tests;

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod oplog_journal_recovery_tests;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod merge_tests;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod retention_tests;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod cache_tests;

#[cfg(all(test, feature = "columnar"))]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod columnar_table_tests;
