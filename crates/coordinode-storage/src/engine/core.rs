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
use coordinode_core::txn::write_concern::{Journal, WriteConcern};
use lsm_tree::AbstractTree;
use tracing::info;

use super::{MAX_DERIVED_EFFECTS, SeekableStorageIter, StorageIter};
use crate::cache::access::AccessTracker;
use crate::cache::tiered::TieredCache;
use crate::engine::batch::WriteBatch;
use crate::engine::compaction::CompactionScheduler;
use crate::engine::config::EndpointConfig;
use crate::engine::config::{FlushPolicy, StorageConfig};
use crate::engine::coordinator::{LocalMultiModalCoordinator, SnapshotPin};
use crate::engine::coverage::{self, Coverage, Domain, TreeCoverage};
use crate::engine::flush::FlushManager;
use crate::engine::open_sets::{OpenSets, SetTrees};
use crate::engine::oplog_journal::{
    EmbeddedOplog, OplogJournalConfig, apply_oplog_ops_at, op_partition,
};
use crate::engine::partition::Partition;
use crate::engine::rebuild::{self, Rebuild, RebuildRefusal};
use crate::engine::routing::PartitionRouting;
use crate::engine::scope::{NamespaceId, Scope, TreeId};
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

/// A test hook run with a partition once a checkpoint has pinned its tree.
#[cfg(test)]
pub(crate) type TreePinnedHook = Box<dyn FnMut(Partition) + Send>;

/// How long a rebuild from a checkpoint waits, in milliseconds, for the
/// operations holding the set of trees to finish before it is refused and
/// tried again later.
const REBUILD_TREE_SET_WAIT_MS: u64 = 30_000;
/// [`REBUILD_TREE_SET_WAIT_MS`] as a duration.
const REBUILD_TREE_SET_WAIT: std::time::Duration =
    std::time::Duration::from_millis(REBUILD_TREE_SET_WAIT_MS);

/// How long a change cursor taken at the end of the journal waits for the
/// entries before the last one applied to finish applying.
pub const CURSOR_APPLY_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// One deployment's store: the system set of trees and every namespace's
/// set, with what they share (the sequence line, the log or journal, the
/// block cache, the flush and compaction pools, the maintenance budget and
/// the endpoints' footprints).
///
/// It reads and writes no namespace's data itself: that goes through a
/// [`StorageEngine`] bound to one namespace, which reaches everything here
/// besides. Applying a logged unit routes each row to the namespace its
/// unit names.
///
/// # Drop ordering
///
/// Fields are dropped in declaration order:
/// 1. `flush_manager` — stops flush workers before trees are touched
/// 2. `compaction_scheduler` — stops compaction workers before trees drop
/// 3. `trees` — AnyTree handles released last
pub struct Deployment {
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
    /// The trees under repair, and of them those whose repair runs now. A
    /// tree is under repair from before the first destructive step until the
    /// repaired tree is installed; a repair that fails leaves it so, with its
    /// intent on disk, until a repair of the tree finishes. A running repair
    /// is held by its [`PartitionRepair`]: a second one of the same tree is
    /// refused while it runs. Read by a cut's page and a capture, never per
    /// key. no-std: spin::Mutex
    repairs: parking_lot::Mutex<RepairMarks>,
    /// Held while a metadata command is decided and applied, so each decision
    /// reads every earlier one; on the journalled path it also spans the
    /// journal append, which makes the decisions' order the journal's order.
    /// Field dictionary reads and snapshot installs take it too, so a reader
    /// never sees a binding's two records half written.
    metadata_decisions: parking_lot::Mutex<()>,
    /// Held, per issuer stripe, while an operation command is decided and
    /// applied (and, on the journalled path, journalled), so a decision about
    /// an operation reads every earlier one about it. Separate from
    /// `metadata_decisions`: client writes of different issuers decide in
    /// parallel and never hold off dictionary reads.
    operation_decisions: crate::engine::operations::DecisionLocks,
    /// Held shared while a unit's principal is checked and the unit applied
    /// (and, on the journalled path, journalled), and exclusively while a
    /// change to the grant catalog is decided and applied: a write is checked
    /// against every grant change applied before it and none after, on a
    /// path whose commits are not ordered by one log. Units with no principal
    /// do not take it.
    access_decisions: parking_lot::RwLock<()>,
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
    /// carrying one has applied. Mirrors the ledger's head record.
    closure_frontier: AtomicU64,
    /// The highest commit timestamp of a committed Raft entry whose effects
    /// may be visible here: raised before an entry's first write lands, so a
    /// read that sees a write is never ahead of it. `0` before any. Every
    /// such commit is in the log, so a closed bound reaches it once the
    /// leader covers it; it says nothing about the apply having finished.
    raft_effects_ceiling: AtomicU64,
    /// The closed bound and the commit ceilings of the Raft log applied
    /// here, as its head and breakpoint records hold them in the Schema
    /// tree's reserved namespace; see [`Self::commit_ceiling_through`].
    // no-std: spin::Mutex; held for one update or one search.
    closure: parking_lot::Mutex<ClosureState>,
    /// Each partition's memtable flush threshold, taken again when a hard
    /// limit or reserve changes.
    flush_thresholds: Arc<crate::engine::flush::FlushThresholds>,
    /// How many breakpoint records the ledger holds, readable without the
    /// `closure` lock; `NO_CLOSURE_LEDGER` while the store holds none.
    /// Mirrors the ledger in `closure`.
    closure_held: AtomicU64,
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
    /// The filesystem every tree writes through, whose device caches a
    /// durable flush empties under each endpoint.
    fs: Arc<dyn lsm_tree::fs::Fs>,
    /// How the trees sync their files: under a barrier a flush is ordered,
    /// and durable once the device caches are flushed.
    tree_sync: lsm_tree::fs::SyncMode,
    /// The version of the deployment tree whose blocks the open verified
    /// before it read the tree, when it did.
    deployment_verified: Option<u64>,
    /// The deployment tree is owed a rebuild (begun here, or cut short before
    /// this open): the closure ledger in memory is not the tree's, so no
    /// commit ceiling is answered until the rebuild installs one.
    ledger_owed: std::sync::atomic::AtomicBool,
    /// The budget every compaction of every tree and the scrub read and
    /// write under, retuned live (see [`Self::set_maintenance_rate`]).
    maintenance: Arc<lsm_tree::rate_limiter::RateLimiter>,
    /// The rate client writes are held to while a partition is past its
    /// slowdown threshold (see [`Self::set_write_slowdown_rate`]).
    write_slowdown: Arc<lsm_tree::rate_limiter::RateLimiter>,
    /// Wakes the compaction monitor: a flush the engine runs itself adds
    /// tables, as the flush manager's do.
    compaction_wake: Arc<coordinode_core::txn::wake::Wake>,
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
    /// The thread that reads free space for `space`; dropped with the
    /// scanner, first on close.
    space_monitor: Option<crate::engine::space::SpaceMonitor>,
    /// Optional embedded oplog journal (oracle-backed, no-Raft mode).
    ///
    /// `Some` when opened via [`StorageEngine::open_with_oracle`] against a
    /// persistent endpoint. The retained oplog drives crash recovery and
    /// WAL-replay-repair (rebuild a corrupt partition from a checkpoint then
    /// replay forward). In cluster mode this is `None` — the Raft log is the
    /// equivalent retained oplog.
    oplog: Option<Arc<Mutex<EmbeddedOplog>>>,
    /// Syncs what volatile commits left unsynced in `oplog`. `Some` exactly
    /// when `oplog` is.
    journal_flusher: Option<crate::engine::oplog_journal::JournalFlusher>,
    /// This member's own recovery records when `oplog` is `None`: the store's
    /// log is a consensus log, which never carries them.
    local_journal: Option<crate::engine::local_journal::LocalJournal>,
    /// Syncs what deferred local records left unsynced. `Some` exactly when
    /// `local_journal` is.
    local_flusher: Option<crate::engine::oplog_journal::JournalFlusher>,
    /// Memtables written to tables on demand ([`Self::flush_memtables`]):
    /// each costs a table, a manifest and a directory sync.
    memtable_flushes: AtomicU64,
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
    /// Each live columnar table's write-pressure series; empty in a build
    /// without columnar support.
    table_metrics: Arc<crate::engine::owned_metrics::TableMetrics>,
    /// The stored nodes of each label that has any, sampled by the capacity
    /// scanner.
    node_counts: Arc<crate::engine::owned_metrics::NodeCounts>,
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
    /// The memtable axis of the same verdict, latched by the flush monitor
    /// as writes fill the memtables and flushes drain them.
    memtable_pressure: Arc<AtomicU8>,
    /// Pauses this node's Raft applies, when a Raft state machine runs over
    /// the engine; read by whatever replaces or copies a partition at an
    /// exact log position. Written once per state machine open.
    raft_fence: parking_lot::RwLock<Option<Arc<dyn raft_coverage::RaftApplyFence>>>,
    /// The group whose log the member running over the engine applies, set
    /// when its node opens: with the deployment, the history its log
    /// positions count entries of.
    log_group: parking_lot::RwLock<Option<coordinode_core::group::GroupId>>,
    /// The lowest Raft log index the latest local checkpoint's trees may
    /// lack, which a partition rebuild from that checkpoint replays from;
    /// the log keeps it. `u64::MAX` when no checkpoint needs any.
    raft_log_keep_from: AtomicU64,
    /// Partition captures taken for copies to other nodes, naming each
    /// one's directory.
    partition_captures: AtomicU64,
    /// Serializes writes of the second copy of the consensus records, so a
    /// refresh that read older records never lands after a newer one, and
    /// holds what this process proved about the copy: only what it wrote
    /// durably is known to survive a power cut, not bytes merely read back.
    promise_copy: parking_lot::Mutex<crate::engine::promise::CopyWriter>,
    /// Consumers watching the writes a partition receives (see
    /// [`Self::tap_writes`]).
    write_taps: Arc<crate::engine::tap::WriteTaps>,
    /// Consumers following the Raft entries as they apply (see
    /// [`Self::subscribe_applied`]).
    applied_feed: Arc<crate::engine::applied::AppliedFeed>,
    /// How many applied commits a derived-index worker's subscription holds
    /// ([`StorageConfig::index_feed_capacity`]).
    index_feed_capacity: usize,
    /// [`StorageConfig::text_index_batch`], at least one.
    text_index_batch: usize,
    /// [`StorageConfig::text_index_retry_ms`], at least one.
    text_index_retry: std::time::Duration,
    /// The transactions open here (see [`Self::await_transactions_through`]).
    open_transactions: Arc<crate::engine::open_txns::OpenTransactions>,
    /// Held by operations that change the trees outside one commit, alone by
    /// a capture or a repair (see [`crate::engine::tree_set`]).
    tree_set: Arc<crate::engine::tree_set::TreeSetGate>,
    /// The catalog's list of tables, set by the layer that owns the catalog;
    /// `None` until it is, and then no capture checks its trees against it.
    catalog: parking_lot::Mutex<Option<Arc<dyn crate::engine::tree_set::CatalogInventory>>>,
    /// Refuses new writes while the disk under a durable endpoint is below
    /// its free-space reserve; shared with the consensus layer, which checks
    /// it before anything reaches its log.
    space: Arc<crate::engine::space::SpaceGuard>,
    /// The owned footprint of every endpoint with a hard limit: every byte
    /// the store's files take there is charged before it is written, and a
    /// client write is admitted against it before it is journaled.
    footprints: crate::engine::footprint::Footprints,
    /// One reclamation of the footprint at a time: refusals arriving while
    /// one runs wait for its outcome rather than start another.
    footprint_reclaim: parking_lot::Mutex<()>,
    /// Notified by the flush workers each time a table lands: the journal it
    /// covers may go ([`Self::start_journal_reclaim`]).
    table_landed: Arc<coordinode_core::txn::wake::Wake>,
    /// Test hook: reclamation fails after its flushes, before any removal.
    #[cfg(test)]
    pub(crate) fail_journal_purge: std::sync::atomic::AtomicBool,
    /// Test hook: run in [`Self::fixed_cut`] once `L` is read and `B` pinned.
    #[cfg(any(test, feature = "test-faults"))]
    pub(crate) on_cut_bound: parking_lot::Mutex<Option<Box<dyn FnMut() + Send>>>,
    /// Test hook: run with a partition once a checkpoint has pinned its tree,
    /// before the next tree is pinned.
    #[cfg(test)]
    pub(crate) on_tree_pinned: parking_lot::Mutex<Option<TreePinnedHook>>,
    /// Test hook: run once a table's rebuild from a checkpoint has cleared
    /// the live tree, before its history goes back in.
    #[cfg(any(test, feature = "test-faults"))]
    on_rebuild_cleared: parking_lot::Mutex<Option<Box<dyn FnMut() + Send>>>,
    /// Test hook: run once a drop within an operation has removed a table's
    /// tree, before the operation goes on to its catalog change.
    #[cfg(any(test, feature = "test-faults"))]
    on_table_dropped: parking_lot::Mutex<Option<Box<dyn FnMut() + Send>>>,
    /// Test hook: run once a `STORAGE COLUMNAR` row has landed in its tree.
    #[cfg(any(test, feature = "test-faults"))]
    on_columnar_row: parking_lot::Mutex<Option<Box<dyn FnMut() + Send>>>,
    /// Test hook: run once a creation within an operation has its table's
    /// tree and stamp, the registry's lock given back, before it returns.
    #[cfg(any(test, feature = "test-faults"))]
    on_table_created: parking_lot::Mutex<Option<Box<dyn FnMut() + Send>>>,
    /// Test hook: run once a checkpoint has pinned its trees and given the
    /// set of trees back, before it copies anything.
    #[cfg(any(test, feature = "test-faults"))]
    pub(crate) on_trees_pinned: parking_lot::Mutex<Option<Box<dyn FnMut() + Send>>>,
    /// Test hook: run with an entry's index once it is journalled, before
    /// it is applied. Called outside its lock, so a hook holding one entry
    /// back leaves other commits free to journal and apply.
    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    pub(crate) on_journalled: parking_lot::Mutex<Option<Arc<dyn Fn(u64) + Send + Sync>>>,
    /// Test hook: run with the segment a cut's continuation read found to
    /// hold its next entry, before it opens it.
    #[cfg(test)]
    #[allow(clippy::type_complexity)]
    pub(crate) on_segment_found:
        parking_lot::Mutex<Option<Box<dyn FnMut(&std::path::Path) + Send>>>,
    /// The lowest journal index a rebuild from the latest local checkpoint
    /// replays from; reclamation keeps it. `u64::MAX` when no checkpoint
    /// needs any.
    journal_keep_from: AtomicU64,
    /// The lowest journal index a registered consumer still needs, published
    /// by its registry; `u64::MAX` when none needs any.
    consumer_journal_floor: AtomicU64,
    /// Set at open while the consumer registry's keyspace holds registrations
    /// whose floors no registry has published yet; collection is held until
    /// one does.
    protections_pending: std::sync::atomic::AtomicBool,
    /// Journal indices held while a consumer registering at them publishes
    /// the floor that keeps them.
    journal_holds: crate::engine::oplog_journal::JournalHolds,
    /// The decoded entries of each log the store's change feeds read, by the
    /// log's directories: one per log, shared by its readers.
    // no-std: spin::Mutex; taken once per read to find the log.
    decoded_logs: parking_lot::Mutex<
        rustc_hash::FxHashMap<Vec<PathBuf>, Arc<crate::engine::decoded_log::DecodedLog>>,
    >,
    /// Where each log a writer opened over this store starts, and its cut
    /// count, as the writer publishes them, by directory.
    // no-std: spin::Mutex; taken when a log opens and a feed first reads it.
    log_floors: parking_lot::Mutex<rustc_hash::FxHashMap<PathBuf, crate::oplog::RetainedFloor>>,
    /// The memory budget of every log's decoded entries.
    decoded_memory: Arc<crate::engine::decoded_log::DecodedMemory>,
    /// Whether the store holds a consensus group's history (its log is bound
    /// to a group, or its trees hold entries of one), as found at open.
    group_history: bool,
    /// This deployment as shared, set once when the open hands it out: what
    /// a [`StorageEngine`] bound to one of its namespaces holds.
    // no-std: once_cell::race::OnceBox
    me: std::sync::OnceLock<std::sync::Weak<Deployment>>,
    /// What opening a namespace's set takes, for one created after the open.
    opener: namespace_open::TreeOpener,
    /// Held while a namespace's set is opened or closed, so two first uses
    /// of one namespace open its trees once and no use opens them while
    /// they close. Holds the closing signals of the trees of each namespace
    /// closed for being idle, which its next open waits for: until then the
    /// closed trees may still touch their directories. no-std: spin::Mutex
    namespace_opening:
        parking_lot::Mutex<rustc_hash::FxHashMap<NamespaceId, Vec<lsm_tree::TreeClose>>>,
    /// The holds on the data directories. Last, so they are released only
    /// once every tree above is closed.
    _directory_locks: crate::format::DirectoryLocks,
}

/// A deployment's store bound to one namespace: what a session, a query, a
/// modality store or a transaction reads and writes through.
///
/// Every partition-addressed read and write here reaches the namespace's own
/// tree of that partition, or the system set's for a partition only the
/// deployment holds ([`Partition::is_deployment`]); a unit it commits names
/// the namespace. Everything the namespaces share is reached through the
/// [`Deployment`] it dereferences to. Cloning it is two reference counts.
///
/// An engine is bound to its namespace when it is made, except a member's
/// engine ([`Deployment::member_engine`]): a member that joins a group holds
/// no namespace until its group's catalog arrives, so its engine binds to
/// the catalog's one namespace on its first use after that. The deployment's
/// own partitions are reached without a binding.
#[derive(Clone)]
pub struct StorageEngine {
    deployment: Arc<Deployment>,
    /// Set at construction for an engine of a named namespace, on first use
    /// for a member's. no-std: once_cell::race::OnceBox
    bound: std::sync::OnceLock<Bound>,
}

/// The namespace an engine is bound to, with its set of trees.
#[derive(Clone)]
struct Bound {
    namespace: NamespaceId,
    set: Arc<SetTrees>,
}

/// What [`StorageEngine::namespace`] answers for a member's engine bound to
/// no namespace yet. The catalog hands out ids from 1, so no namespace has
/// it, and a unit naming it is refused.
const NO_NAMESPACE: NamespaceId = NamespaceId::from_raw(0);

impl core::ops::Deref for StorageEngine {
    type Target = Deployment;

    #[inline]
    fn deref(&self) -> &Deployment {
        &self.deployment
    }
}

impl core::fmt::Debug for StorageEngine {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StorageEngine")
            .field("data_dir", &self.deployment.data_dir)
            .field("namespace", &self.bound.get().map(|bound| bound.namespace))
            .finish_non_exhaustive()
    }
}

/// History entries a replacement import writes between two admissions
/// ([`StorageEngine::import_generation_history`]): admission is an atomic load
/// almost always, so this bounds how far an import runs past a refusal, not
/// what it costs.
pub const IMPORT_ADMISSION_STRIDE: usize = 1024;

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

/// A unit's hold on grant changes, kept through its check and apply.
pub(crate) enum AccessHold<'a> {
    /// The unit neither acts for a principal nor changes grants.
    None,
    /// The unit acts for a principal: grant changes wait for it.
    Shared(#[expect(dead_code, reason = "held for its drop")] parking_lot::RwLockReadGuard<'a, ()>),
    /// The unit changes grants: units acting for principals wait for it.
    Exclusive(
        #[expect(dead_code, reason = "held for its drop")] parking_lot::RwLockWriteGuard<'a, ()>,
    ),
}

/// The partition a resolved mutation writes, `None` for one that writes none.
fn mutation_partition(mutation: &Mutation) -> Option<Partition> {
    match mutation {
        Mutation::Put { partition, .. }
        | Mutation::Delete { partition, .. }
        | Mutation::Merge { partition, .. }
        | Mutation::RemoveRange { partition, .. } => Some(Partition::from(*partition)),
        Mutation::Command(_)
        | Mutation::Derive(_)
        | Mutation::Operation(_)
        | Mutation::Effects(_)
        | Mutation::Namespace(_)
        | Mutation::Actor(_) => None,
    }
}

/// One write to a node-local record ([`StorageEngine::commit_local`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalWrite {
    /// Set `key` in `partition` to `value`.
    Put {
        partition: Partition,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    /// Remove `key` from `partition`.
    Delete { partition: Partition, key: Vec<u8> },
}

impl LocalWrite {
    fn into_mutation(self) -> StorageResult<Mutation> {
        let carried = |partition: Partition| {
            partition.proposal_id().ok_or_else(|| {
                StorageError::InvalidConfig(format!(
                    "the journal carries no node-local record of {}",
                    partition.name()
                ))
            })
        };
        Ok(match self {
            Self::Put {
                partition,
                key,
                value,
            } => Mutation::Put {
                partition: carried(partition)?,
                key,
                value,
            },
            Self::Delete { partition, key } => Mutation::Delete {
                partition: carried(partition)?,
                key,
            },
        })
    }
}

/// Whose a journal entry is: the store's, which subscribers follow, or this
/// node's own recovery records, which they do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryScope {
    Shared,
    NodeLocal,
}

/// How a crash test ends a store: see [`Deployment::journal_crash_point`].
#[cfg(any(test, feature = "test-faults"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashKind {
    /// Power lost: every journal entry not synced is gone.
    PowerCut,
    /// The process killed: what it handed to the operating system stays.
    Kill,
}

/// A journal stopped by [`Deployment::hold_journal`]; it answers again
/// once this is dropped.
#[cfg(any(test, feature = "test-faults"))]
#[must_use = "the journal answers again once this is dropped"]
pub struct JournalStall<'a> {
    _held: std::sync::MutexGuard<'a, crate::engine::oplog_journal::EmbeddedOplog>,
}

/// When a [`StorageEngine::commit_local`] call returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalSync {
    /// Once the records are durable: they must survive a crash before the
    /// use they guard begins.
    BeforeUse,
    /// Once they are applied; the next sync covers them, and recovery repairs
    /// their loss.
    Deferred,
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
    /// The tree that was repaired: a partition's, of the system set or of a
    /// namespace's.
    pub tree: TreeId,
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
    /// `deployment` bound to namespace `namespace`, whose set it opens when
    /// it is not open yet.
    ///
    /// # Errors
    ///
    /// The namespace's trees cannot be opened.
    pub fn bind(deployment: Arc<Deployment>, namespace: NamespaceId) -> StorageResult<Self> {
        let set = deployment.namespace_set(namespace)?;
        Ok(Self {
            deployment,
            bound: std::sync::OnceLock::from(Bound { namespace, set }),
        })
    }

    /// The deployment this engine is a namespace of.
    #[inline]
    #[must_use]
    pub fn deployment(&self) -> &Arc<Deployment> {
        &self.deployment
    }

    /// The namespace this engine is bound to, binding a member's engine
    /// first.
    ///
    /// # Errors
    ///
    /// A member's engine whose catalog holds no namespace
    /// ([`StorageError::NoNamespace`]), or the namespace's trees cannot be
    /// opened.
    #[inline]
    fn bound(&self) -> StorageResult<&Bound> {
        match self.bound.get() {
            Some(bound) => Ok(bound),
            None => self.bind_anchor(),
        }
    }

    /// Bind a member's engine to its catalog's anchor namespace
    /// ([`Deployment::anchor_namespace`]).
    #[cold]
    fn bind_anchor(&self) -> StorageResult<&Bound> {
        let namespace = self.deployment.anchor_namespace()?;
        let set = self.deployment.namespace_set(namespace)?;
        Ok(self.bound.get_or_init(|| Bound { namespace, set }))
    }

    /// Whether this engine is bound to a namespace, binding a member's engine
    /// first when its catalog now holds one.
    #[must_use]
    pub fn is_bound(&self) -> bool {
        self.bound().is_ok()
    }

    /// Bind a member's engine once its catalog holds a namespace: `true` once
    /// bound, `false` while the catalog holds none yet.
    ///
    /// # Errors
    ///
    /// The catalog or the namespace's trees cannot be read.
    pub fn try_bind(&self) -> StorageResult<bool> {
        match self.bound() {
            Ok(_) => Ok(true),
            Err(StorageError::NoNamespace { .. }) => Ok(false),
            Err(err) => Err(err),
        }
    }

    /// The namespace this engine reads and writes. A member's engine bound
    /// to none yet, its catalog holding none, answers an id no namespace
    /// has, under which nothing is written or read.
    #[inline]
    #[must_use]
    pub fn namespace(&self) -> NamespaceId {
        self.bound().map_or(NO_NAMESPACE, |bound| bound.namespace)
    }

    /// The namespace's set of trees.
    ///
    /// # Errors
    ///
    /// As [`Self::bound`].
    #[inline]
    pub(crate) fn set(&self) -> StorageResult<&Arc<SetTrees>> {
        self.bound().map(|bound| &bound.set)
    }

    /// The set a write batch of this engine starts in: the namespace's, or
    /// the system set for a member's engine bound to none, whose batches
    /// write only the deployment's partitions.
    pub(crate) fn batch_set(&self) -> Arc<SetTrees> {
        match self.bound() {
            Ok(bound) => Arc::clone(&bound.set),
            Err(_) => Arc::clone(self.coordinator.sets().system()),
        }
    }

    /// The address of the tree of `part` this engine reaches: the
    /// namespace's, or the system set's for a partition only the deployment
    /// holds.
    #[must_use]
    pub fn tree_id(&self, part: Partition) -> TreeId {
        if part.is_deployment() {
            TreeId::system(part)
        } else {
            TreeId::namespace(self.namespace(), part)
        }
    }

    /// [`Deployment::commit_local`] of records of this namespace's trees
    /// beside the deployment's own.
    ///
    /// # Errors
    ///
    /// As [`Deployment::commit_local`].
    pub fn commit_local(&self, writes: Vec<LocalWrite>, sync: LocalSync) -> StorageResult<()> {
        let namespace = match self.bound() {
            Ok(bound) => Some(bound.namespace),
            Err(_) => None,
        };
        self.deployment.commit_local_in(namespace, writes, sync)
    }

    /// Open the store at `config` as a store of one namespace: the host's
    /// choice of its own store, which initializes a fresh one with its one
    /// namespace ([`Deployment::single_store`]).
    ///
    /// # Errors
    ///
    /// As [`Deployment::open`] and [`Deployment::single_store`].
    pub fn open(config: &StorageConfig) -> StorageResult<Self> {
        Deployment::open(config)?.single_store()
    }

    /// [`Deployment::open_with_oracle`], as a store of one namespace.
    ///
    /// # Errors
    ///
    /// As [`Deployment::open_with_oracle`] and [`Deployment::single_store`].
    pub fn open_with_oracle(
        config: &StorageConfig,
        oracle: Arc<coordinode_core::txn::timestamp::TimestampOracle>,
    ) -> StorageResult<Self> {
        Deployment::open_with_oracle(config, oracle)?.single_store()
    }

    /// [`Deployment::open_embedded`], as a store of one namespace.
    ///
    /// # Errors
    ///
    /// As [`Deployment::open_embedded`] and [`Deployment::single_store`].
    pub fn open_embedded(
        config: &StorageConfig,
        oracle: Arc<coordinode_core::txn::timestamp::TimestampOracle>,
    ) -> StorageResult<Self> {
        Deployment::open_embedded(config, oracle)?.single_store()
    }

    /// [`Deployment::open_embedded_with_journal`], as a store of one
    /// namespace.
    ///
    /// # Errors
    ///
    /// As [`Deployment::open_embedded_with_journal`] and
    /// [`Deployment::single_store`].
    pub fn open_embedded_with_journal(
        config: &StorageConfig,
        oracle: Arc<coordinode_core::txn::timestamp::TimestampOracle>,
        journal: OplogJournalConfig,
    ) -> StorageResult<Self> {
        Deployment::open_embedded_with_journal(config, oracle, journal)?.single_store()
    }

    /// [`Deployment::open_restoring`], as a store of one namespace.
    ///
    /// # Errors
    ///
    /// As [`Deployment::open_restoring`] and [`Deployment::single_store`].
    pub fn open_restoring(
        config: &StorageConfig,
        oracle: Arc<coordinode_core::txn::timestamp::TimestampOracle>,
        claim: crate::format::RestoreClaim,
    ) -> StorageResult<Self> {
        Deployment::open_restoring(config, oracle, claim)?.single_store()
    }
}

impl Deployment {
    /// This deployment bound to namespace `namespace`.
    ///
    /// # Errors
    ///
    /// The deployment is being dropped, or the namespace's set cannot be
    /// opened.
    pub fn engine(&self, namespace: NamespaceId) -> StorageResult<StorageEngine> {
        let me = self
            .me
            .get()
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| StorageError::Io("the deployment is closing".into()))?;
        StorageEngine::bind(me, namespace)
    }

    /// Hand the opened deployment out shared.
    fn shared(self) -> Arc<Self> {
        let shared = Arc::new(self);
        // Set once, here, before anyone else holds it.
        shared.me.get_or_init(|| Arc::downgrade(&shared));
        shared
    }

    /// Open or create a deployment's store at the configured path.
    pub fn open(config: &StorageConfig) -> StorageResult<Arc<Self>> {
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
    ) -> StorageResult<Arc<Self>> {
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
    /// ([`StorageEngine::repair_partition_from_checkpoint`]).
    ///
    /// In-memory configs (every endpoint `Volatile`) get no journal — there is
    /// no durable place to keep it; durability there is best-effort by design.
    pub fn open_embedded(
        config: &StorageConfig,
        oracle: std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>,
    ) -> StorageResult<Arc<Self>> {
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
    ) -> StorageResult<Arc<Self>> {
        let gc_watermark = Arc::new(AtomicU64::new(0));
        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(OracleSeqnoGenerator(Arc::clone(&oracle)));
        Self::finish_open(config, seqno, gc_watermark, Some(oracle), Some(journal))
    }

    /// Bring a store the non-replicated build wrote to the cut a replicated
    /// group starts from.
    ///
    /// Its journal's entries are replayed and flushed into the partition
    /// trees, which then hold every one of them, and the journal's segments
    /// are removed: the consensus log that follows lives where the journal
    /// did, and an entry of the old journal read as one of the new log is
    /// refused. Interrupted at any point, it is run again on the next open:
    /// entries the trees already hold are not replayed twice. Returns whether
    /// there was a journal to retire; a store without one, or whose log is
    /// already a consensus log, is left as it is.
    ///
    /// Must run with no engine open on the store.
    ///
    /// # Errors
    ///
    /// The journal cannot be read or replayed, the flush fails, or a segment
    /// cannot be removed.
    pub fn retire_embedded_journal(config: &StorageConfig) -> StorageResult<bool> {
        use crate::oplog::OplogOp;
        use crate::oplog::manager::OplogManager;

        let Ok(endpoint) = config.select_oplog_endpoint(0) else {
            return Ok(false);
        };
        let dir = endpoint.path.join("oplog").join("0");
        if !dir.exists() {
            return Ok(false);
        }
        let journal = OplogJournalConfig::from(config);
        let open_log = || {
            OplogManager::open(
                &dir,
                0,
                journal.max_segment_bytes,
                journal.max_segment_entries,
                journal.retention_secs,
            )
        };
        // A consensus log frames every entry; a journal entry carries the
        // unit itself. The newest entry tells which this directory holds.
        let consensus = |op: &OplogOp| {
            matches!(
                op,
                OplogOp::RaftEntry { .. } | OplogOp::RaftTruncation { .. }
            )
        };
        match open_log()?.recover_last_entry()? {
            None => return Ok(false),
            Some(last) if last.ops.iter().any(consensus) => return Ok(false),
            Some(_) => {}
        }
        {
            let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
            let engine = Self::open_embedded(config, oracle)?;
            // Every entry is in a tree's memtable after the replay. The base
            // goes into every tree, as before a purge, so a removal cut short
            // below leaves a journal whose missing prefix every tree covers.
            let covered = engine.persist_coverage_base()?;
            let next = match &engine.oplog {
                Some(log) => log
                    .lock()
                    .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?
                    .next_index(),
                None => 0,
            };
            if covered < next {
                return Err(StorageError::Io(format!(
                    "the journal holds entries up to {next} but the trees cover only below \
                     {covered}; not retiring it"
                )));
            }
        }
        open_log()?.truncate_all()?;
        // The consensus log starts in this directory next: the removals are
        // made durable first, so a power cut cannot bring old journal
        // segments back beside it.
        sync_dir(&dir)?;
        tracing::info!(dir = %dir.display(), "embedded journal retired into the store");
        Ok(true)
    }

    /// Open the directories `claim` holds for the restore that claimed them,
    /// as [`Self::open_with_oracle`] does: the claim's holds pass to the
    /// engine, and the unfinished restore it is for does not refuse the
    /// open. The restore installs its image through this engine and ends
    /// with [`crate::format::finish_restore`] before dropping it.
    ///
    /// # Errors
    ///
    /// As [`Self::open_with_oracle`];
    /// [`StorageError::RestoreTarget`] when `config` names other
    /// directories than the claim covers.
    pub fn open_restoring(
        config: &StorageConfig,
        oracle: std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>,
        claim: crate::format::RestoreClaim,
    ) -> StorageResult<Arc<Self>> {
        let gc_watermark = Arc::new(AtomicU64::new(0));
        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(OracleSeqnoGenerator(Arc::clone(&oracle)));
        // No batch of a restore needs to be durable on its own: its claim
        // keeps a cut-short install unserved and the rerun starts it over,
        // and the restore persists once before it ends the claim. Full
        // memtables still flush in the background.
        let mut config = config.clone();
        config.flush_policy = FlushPolicy::Manual;
        Self::finish_open_with(
            &config,
            seqno,
            gc_watermark,
            Some(oracle),
            None,
            Some(claim),
        )
    }

    fn finish_open(
        config: &StorageConfig,
        seqno: lsm_tree::SharedSequenceNumberGenerator,
        gc_watermark: Arc<AtomicU64>,
        oracle: Option<std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
        journal_config: Option<OplogJournalConfig>,
    ) -> StorageResult<Arc<Self>> {
        Self::finish_open_with(config, seqno, gc_watermark, oracle, journal_config, None)
    }

    fn finish_open_with(
        config: &StorageConfig,
        seqno: lsm_tree::SharedSequenceNumberGenerator,
        gc_watermark: Arc<AtomicU64>,
        oracle: Option<std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
        journal_config: Option<OplogJournalConfig>,
        claim: Option<crate::format::RestoreClaim>,
    ) -> StorageResult<Arc<Self>> {
        // Before anything reads the directory: the directories are taken,
        // and one of another engine format is migrated or refused here,
        // never opened as it is.
        let crate::format::Prepared {
            format,
            locks: directory_locks,
        } = match claim {
            Some(claim) => crate::format::prepare_restoring(config, claim)?,
            None => crate::format::prepare(config)?,
        };

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
            retention_window_to_us(std::time::Duration::from_secs(config.retention_window_secs)),
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

        // Every file the trees grow under an endpoint with a hard limit is
        // charged to its footprint before it is written; the footprints start
        // with what the endpoints already hold.
        let footprints = crate::engine::footprint::Footprints::open(&config.endpoints)
            .map_err(|e| StorageError::Io(format!("measure the endpoints' footprint: {e}")))?;
        // A store that does not fit its limit still opens; what it grows
        // while opening is bounded by the phase, held until this returns.
        let _opening = footprints.open_phase();
        let _metered;
        let config = if footprints.is_empty() {
            config
        } else {
            let inner = config
                .fs
                .clone()
                .unwrap_or_else(|| Arc::new(lsm_tree::fs::StdFs));
            _metered = config
                .clone()
                .with_fs(Arc::new(crate::engine::metered_fs::MeteredFs::new(
                    inner,
                    footprints.clone(),
                )) as Arc<dyn lsm_tree::fs::Fs>);
            &_metered
        };

        // Shared block cache across all partition trees.
        let cache = Arc::new(lsm_tree::Cache::with_capacity_bytes(
            config.block_cache_bytes,
        ));

        // Two-pass open (per-LSM-level endpoint routing):
        //
        // **Pass 1**: open the deployment's records tree with no per-level
        // routing. It holds the routing of every other tree, so it has to be
        // reachable before we can decide where other trees' SSTs should
        // land. It stays single-tier on the primary endpoint by design: its
        // records are small and the bootstrap problem
        // (routing-needed-to-open-the-routing-store) does not arise.
        //
        // **Pass 2**: for each other tree, load the persisted
        // [`PartitionRouting`] from the deployment's records or initialise
        // (compute + persist) it on first open against this endpoint set, and
        // open the tree with `level_routes` derived from it. A namespace's
        // Schema tree stays single-tier like the deployment's: metadata.
        let mut trees = HashMap::with_capacity(Partition::all().len());
        let mut open_repairs: Vec<OpenRepair> = Vec::new();

        // Structural open failures (lost `current` pointer, corrupt version
        // log) are repaired in place with block salvage rather than failing
        // the whole engine open: recovery must always yield an openable
        // engine, with any loss REPORTED (via `open_repairs`) for the repair
        // orchestrator to replay, never a dead end. Config-level mistakes
        // (wrong comparator, missing dictionary) still propagate as errors.
        let repair_policy = lsm_tree::RepairPolicy::default().salvage(true);

        // One write buffer for every partition tree: the memtable axis of the
        // write-pressure verdict bounds their unflushed bytes together.
        let write_buffer = Arc::new(lsm_tree::WriteBufferBudget::default());
        // See `StorageEngine::closure_held`.
        let closure_held = AtomicU64::new(NO_CLOSURE_LEDGER);
        // One maintenance budget for every tree's compaction and the scrub:
        // the trees share the disks the foreground needs.
        let maintenance = new_maintenance_budget(
            config.maintenance_bytes_per_sec,
            config.maintenance_backoff_ms,
        );
        // One account of the room compactions in progress hold, for every
        // tree: they share the endpoints' volumes, so each reserves against
        // what the others still have to write rather than the same free bytes.
        let capacity_pool = Arc::new(lsm_tree::CapacityPool::default());

        // Pass 1: the deployment's records (single-tier, no routing).
        let deployment_config = config
            .to_tree_config(TreeId::system(Partition::Deployment), Arc::clone(&seqno))
            .use_cache(Arc::clone(&cache))
            .write_buffer_budget(Arc::clone(&write_buffer))
            .compaction_rate_limiter(Arc::clone(&maintenance))
            .compaction_capacity(Arc::clone(&capacity_pool));
        let (deployment_tree, deployment_repair) =
            deployment_config.open_or_repair(repair_policy)?;
        let deployment_repair = deployment_repair.map(|report| {
            tracing::warn!(
                partition = Partition::Deployment.name(),
                recovered = report.recovered,
                salvaged = report.salvaged,
                scope = ?report.wal_replay_scope(),
                "partition tree structurally repaired at open"
            );
            OpenRepair {
                tree: TreeId::system(Partition::Deployment),
                replay_scope: report.wal_replay_scope(),
                report,
            }
        });
        trees.insert(Partition::Deployment, deployment_tree.clone());

        // Restore seqno from the deployment's records BEFORE reading routing:
        // the routing get() needs a fresh seqno bound so it observes prior
        // persisted writes, and one above the tree's retention floor so it is
        // served at all. The two differ most here: these rows are as old as
        // the database while the floor is as recent as the last compaction,
        // so a clock that restarts behind that compaction would otherwise
        // read below the floor and fail the open. Same seeding as the
        // whole-engine restore below, for the one tree read before it.
        {
            use lsm_tree::AbstractTree;
            if let Some(max) = deployment_tree.get_highest_seqno() {
                seqno.fetch_max(max + 1);
            }
            if let Some(first_servable) = deployment_tree.retention_floor().checked_add(1) {
                seqno.fetch_max(first_servable);
            }
        }
        // Nothing reads the deployment's records as an answer until they are
        // proved; a rebuild the tree is owed runs once the journal is read.
        let settled = deployment_bootstrap::Bootstrap {
            config,
            records: &deployment_tree,
            repair: deployment_repair.as_ref(),
            seqno: &seqno,
            journaled: journal_config.is_some() && config.select_oplog_endpoint(0).is_ok(),
            footprints: &footprints,
            maintenance: &maintenance,
        }
        .settle()?;
        let deployment_verified = settled.verified;
        let mut deployment_rebuild = settled.rebuild;
        let mut deployment_rebuilt = false;
        {
            use lsm_tree::AbstractTree;
            let head = deployment_tree
                .get(crate::engine::closure_ledger::HEAD_KEY, seqno.get())
                .map_err(|e| StorageError::Io(format!("schema get closed bound: {e}")))?;
            if let Some(head) = head {
                closure_held.store(
                    crate::engine::closure_ledger::ClosureLedger::stored_held(&head)?,
                    std::sync::atomic::Ordering::Release,
                );
                // A capture of the store reads at its closed bound: nothing
                // the open or its maintenance collects goes past it.
                gc_controller.hold_closed_bound(
                    crate::engine::closure_ledger::ClosureLedger::stored_below(&head)?,
                );
            }
        }

        // Pass 2: load/initialise routing for each other partition. The
        // deployment's other trees open here with `level_routes` wired; a
        // namespace's take the same routing when its set opens. The node
        // partition's routing also places the columnar tables' levels.
        // Where each partition's flushes land: the endpoint of its level 0.
        // The Schema trees stay on the primary endpoint.
        let mut flush_landing: HashMap<Partition, std::path::PathBuf> = HashMap::new();
        for part in [Partition::Deployment, Partition::Schema] {
            flush_landing.insert(part, config.data_dir().to_path_buf());
        }
        // Intents live as files of the data directory itself: a store whose
        // directory is not one on disk (a memory filesystem) keeps none.
        let durable_intents = config.data_dir().is_dir();
        let mut data_routing: HashMap<Partition, PartitionRouting> = HashMap::new();
        for &part in Partition::all() {
            if part == Partition::Deployment {
                continue;
            }
            // A namespace's Schema tree stays single-tier, like the
            // deployment's: metadata is small and read first.
            let routing = if part == Partition::Schema {
                None
            } else {
                Some(load_or_init_partition_routing(
                    &deployment_tree,
                    &seqno,
                    &config.endpoints,
                    part,
                    config.relocated,
                )?)
            };
            if let Some(endpoint) = routing
                .as_ref()
                .and_then(|routing| routing.levels.get(&0))
                .and_then(|id| config.endpoints.iter().find(|e| &e.id == id))
            {
                flush_landing.insert(part, endpoint.path.clone());
            }
            if !part.is_deployment() {
                if let Some(routing) = routing {
                    data_routing.insert(part, routing);
                }
                continue;
            }
            let tree_config = config
                .to_tree_config_with_routing(
                    TreeId::system(part),
                    Arc::clone(&seqno),
                    routing.as_ref(),
                )
                .use_cache(Arc::clone(&cache))
                .write_buffer_budget(Arc::clone(&write_buffer))
                .compaction_rate_limiter(Arc::clone(&maintenance))
                .compaction_capacity(Arc::clone(&capacity_pool));
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
                    tree: TreeId::system(part),
                    replay_scope: report.wal_replay_scope(),
                    report,
                });
            }
            trees.insert(part, tree);
        }
        // Each live columnar table's write-pressure series, of every
        // namespace.
        let table_metrics = Arc::new(crate::engine::owned_metrics::TableMetrics::default());
        let opener = namespace_open::TreeOpener {
            config: config.clone(),
            seqno: Arc::clone(&seqno),
            cache: Arc::clone(&cache),
            write_buffer: Arc::clone(&write_buffer),
            maintenance: Arc::clone(&maintenance),
            capacity: Arc::clone(&capacity_pool),
            repair_policy,
            routing: data_routing,
            #[cfg(feature = "columnar")]
            table_metrics: Arc::clone(&table_metrics),
        };
        // Every namespace the catalog holds live, and every one whose trees
        // lie on disk, opens with the store: the journal's replay and the
        // proof of what the trees cover below reach any of them, also while
        // the catalog itself is owed a rebuild.
        let mut namespace_sets: std::collections::BTreeMap<NamespaceId, SetTrees> =
            std::collections::BTreeMap::new();
        let mut to_open = namespace_open::namespaces_on_disk(&config.endpoints);
        for (id, record) in namespace_open::catalog_at(&deployment_tree, seqno.get())? {
            if record.state == coordinode_core::namespace::NamespaceState::Live {
                to_open.insert(id);
            } else {
                // What a dropped namespace's removal left, a crash or an open
                // file in its way, goes now: nothing reads it again.
                to_open.remove(&id);
                remove_namespace_dirs(&namespace_dirs(&config.endpoints, id), &footprints);
            }
        }
        for id in to_open {
            let (set, repairs) = opener.open_namespace(id, &deployment_tree)?;
            open_repairs.extend(repairs);
            namespace_sets.insert(id, set);
        }

        // Registrations a consumer registry recorded in an earlier run hold
        // history this engine cannot read the bounds of: their registry
        // publishes them once it recovers. Until then nothing is collected,
        // neither MVCC history nor journal, so no reclamation this open
        // starts runs ahead of the protections the store owes.
        // Records only: the apply coverage markers every member keeps in
        // each tree sit below the user keyspace.
        let protections_pending = {
            use lsm_tree::AbstractTree as _;
            trees.get(&Partition::Registry).is_some_and(|tree| {
                tree.range(
                    crate::engine::coverage::USER_KEYSPACE_START..,
                    lsm_tree::SeqNo::MAX,
                    None,
                )
                .next()
                .is_some()
            })
        };
        if protections_pending {
            gc_controller.set_external_floor(0);
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
        let oplog = match journal_config {
            Some(mut jcfg) => match config.select_oplog_endpoint(0) {
                Ok(endpoint) => {
                    jcfg.max_segment_bytes = crate::engine::footprint::journal_segment_bytes(
                        endpoint.hard_limit_bytes,
                        jcfg.max_segment_bytes,
                    );
                    // A store whose history is a group's log is refused
                    // before its log is opened as a journal: an entry written
                    // here would never reach the group.
                    if let Some(evidence) = group_history_evidence(&endpoint.path, &trees)? {
                        return Err(StorageError::GroupMemberStore {
                            path: config.data_dir().display().to_string(),
                            evidence,
                        });
                    }
                    let dir = endpoint.path.join("oplog").join("0");
                    let mut journal = EmbeddedOplog::open(
                        &dir,
                        0,
                        &jcfg,
                        pending_commits.durable_ceiling_cell(),
                        footprints.clone(),
                    )?;
                    let entries = journal.read_all()?;
                    // A rebuild owed already keeps its intent as it is: the
                    // Schema tree's holds the records the rebuild puts back.
                    let owed = read_rebuild_intents(config.data_dir())?.trees;
                    let mut covered: HashMap<TreeId, TreeCoverage> = HashMap::new();
                    for (&part, tree) in &trees {
                        note_journal_coverage(
                            TreeId::system(part),
                            tree,
                            &owed,
                            config.data_dir(),
                            &mut covered,
                        )?;
                    }
                    for set in namespace_sets.values() {
                        for (id, tree) in set.iter() {
                            note_journal_coverage(
                                id,
                                tree,
                                &owed,
                                config.data_dir(),
                                &mut covered,
                            )?;
                        }
                    }
                    // A tree without a coverage record is replayed every entry
                    // that touches it. No entry's effect can be durable in it:
                    // an entry's effects land in a tree in one batch with its
                    // marker, so whatever made them durable made the marker (or
                    // the base that folded it) durable too, and tables written
                    // outside the journal carry no entry. The replay is exact
                    // as long as the journal still holds every entry from the
                    // first. A purge is preceded by a fold into every tree, so a
                    // tree with no record behind a purged journal lost it (a
                    // lost tree, a lost manifest), and part of its history.
                    // The same holds for any index the journal no longer has:
                    // a purged prefix or a segment lost from the middle. Every
                    // tree must cover each such index itself, or the replay
                    // would skip an entry it never received.
                    //
                    // A tree whose rebuild was cut short is left alone: its
                    // record no longer says what it holds, and the repair that
                    // follows rebuilds it from the checkpoint and this journal.
                    let rebuilding = read_rebuild_intents(config.data_dir())?.trees;
                    if !crate::engine::local_journal::holes_covered(
                        &entries,
                        journal.next_index(),
                        &covered,
                        |id: TreeId, _| rebuilding.contains(&id),
                    ) {
                        return Err(StorageError::CoverageUnprovable {
                            path: dir.display().to_string(),
                            entries: entries.len(),
                        });
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
                    // The deployment records' rebuild, before the replay:
                    // every entry below this index is applied, as the
                    // journal and every other tree say, and the rebuilt tree
                    // holds them all. The replay below leaves the tree to it.
                    if let Some(rebuild) = deployment_rebuild.take() {
                        rebuild.run(deployment_bootstrap::RebuildInputs {
                            config,
                            records: crate::engine::local_journal::tree_of(
                                &trees,
                                Partition::Deployment,
                            )?,
                            repair: deployment_repair.as_ref(),
                            seqno: &seqno,
                            footprints: &footprints,
                            repair_policy,
                            entries: &entries,
                            next: journal.next_index(),
                        })?;
                        deployment_rebuilt = true;
                    }
                    let mut replayed = 0usize;
                    for entry in &entries {
                        // Group the entry's data ops per tree so each tree
                        // receives them as one batch at the entry's ts, with
                        // the entry's marker last. DERIVED work is derived
                        // first, from the whole entry: the data partition it
                        // reads may already hold the entry.
                        let ops = crate::oplog::convert::resolve_entry_ops(&entry.ops)?;
                        let mut per_tree: HashMap<TreeId, Vec<&OplogOp>> = HashMap::new();
                        // The namespace the rows after the last marker belong
                        // to; `Some(None)` for one no longer live, whose rows
                        // are left out as its apply leaves them out.
                        let mut namespace: Option<Option<NamespaceId>> = None;
                        for op in ops.iter() {
                            if let OplogOp::Namespace { id } = op {
                                let id = NamespaceId::from_raw(*id);
                                // A namespace an earlier entry created opens
                                // now: its record is replayed by then.
                                if !namespace_sets.contains_key(&id)
                                    && namespace_open::live_at(
                                        &deployment_tree,
                                        id,
                                        lsm_tree::SeqNo::MAX,
                                    )?
                                {
                                    let (set, repairs) =
                                        opener.open_namespace(id, &deployment_tree)?;
                                    open_repairs.extend(repairs);
                                    for (tree_id, tree) in set.iter() {
                                        note_journal_coverage(
                                            tree_id,
                                            tree,
                                            &owed,
                                            config.data_dir(),
                                            &mut covered,
                                        )?;
                                    }
                                    namespace_sets.insert(id, set);
                                }
                                namespace = Some(namespace_sets.contains_key(&id).then_some(id));
                                continue;
                            }
                            #[cfg(feature = "columnar")]
                            if let OplogOp::ColumnarInsert {
                                table_id,
                                key,
                                value,
                            } = op
                            {
                                // A row of a namespace dropped since is not
                                // kept, as its apply would not keep it.
                                if let Some(Some(namespace)) = namespace {
                                    columnar_replay.push(ColumnarReplay {
                                        namespace,
                                        table_id: table_id.clone(),
                                        key: key.clone(),
                                        value: value.clone(),
                                        ts: entry.ts,
                                        index: entry.index,
                                    });
                                }
                                continue;
                            }
                            let Some(part) = op_partition(op) else {
                                continue;
                            };
                            let id = if part.is_deployment() {
                                TreeId::system(part)
                            } else {
                                match namespace {
                                    Some(Some(ns)) => TreeId::namespace(ns, part),
                                    // A namespace dropped since: nothing of
                                    // it is kept.
                                    Some(None) => continue,
                                    None => {
                                        return Err(StorageError::Io(format!(
                                            "journal entry {} writes the {} partition \
                                             without naming its namespace",
                                            entry.index,
                                            part.name()
                                        )));
                                    }
                                }
                            };
                            per_tree.entry(id).or_default().push(op);
                        }
                        for (id, ops) in &per_tree {
                            let (tree, installations) = match id.scope {
                                Scope::System => (trees.get(&id.part), None),
                                Scope::Namespace(ns) => {
                                    let set = namespace_sets.get(&ns);
                                    (
                                        set.and_then(|set| set.get(id.part)),
                                        set.and_then(|set| set.installations_of(id.part)),
                                    )
                                }
                            };
                            let tree = tree.ok_or_else(|| StorageError::PartitionNotFound {
                                name: id.name(),
                            })?;
                            let skip = rebuilding.contains(id)
                                || covered.get(id).is_some_and(|c| c.contains(entry.index, 0));
                            if !skip {
                                apply_oplog_ops_at(
                                    id.part,
                                    tree,
                                    installations,
                                    ops,
                                    entry.ts,
                                    entry.index,
                                    Domain::Journal,
                                )?;
                                replayed += ops.len();
                            }
                        }
                    }
                    // Not flushed here: the journal keeps every replayed entry,
                    // and the markers the replay wrote are what a later crash
                    // replays them again against, each once.
                    if replayed > 0 {
                        tracing::info!(
                            replayed,
                            "oplog: replayed un-flushed entries from journal on open"
                        );
                    }
                    Some(journal)
                }
                // No oplog-eligible (Durable/Degraded) endpoint — in-memory /
                // no-persistence config. Durability is best-effort; no journal.
                Err(_) => None,
            },
            None => None,
        };
        // A rebuilt deployment tree's repair is met; any other stays owed.
        if let Some(repair) = deployment_repair
            && !deployment_rebuilt
        {
            open_repairs.push(repair);
        }

        // ── Member-local journal: open + crash recovery ─────────────────────
        // A store whose log is a consensus log keeps this member's own records
        // in a journal of their own, replayed here, before anything reads them.
        // A store with its own journal records them there instead; what an
        // open without a journal left (an offline tool, never a member: those
        // are refused above) is replayed too, and folded into the trees once
        // the store is open (`drain_local_journal`).
        let local_journal = match config.select_oplog_endpoint(0) {
            Ok(endpoint) => {
                let dir = crate::engine::local_journal::LocalJournal::dir(&endpoint.path);
                if oplog.is_none() || dir.exists() {
                    // The local records' trees: the deployment's, and those
                    // of the one namespace a store of one namespace holds.
                    let mut local_trees = trees.clone();
                    let mut local_installations = None;
                    if let [set] = namespace_sets.values().collect::<Vec<_>>()[..] {
                        local_trees.extend(set.iter().map(|(id, tree)| (id.part, tree.clone())));
                        local_installations = set.installations();
                    }
                    Some(crate::engine::local_journal::LocalJournal::open(
                        &dir,
                        &OplogJournalConfig::from(config),
                        &local_trees,
                        local_installations,
                        footprints.clone(),
                    )?)
                } else {
                    None
                }
            }
            // A store kept in memory has nothing to recover its records
            // from; they live in its trees alone.
            Err(_) => None,
        };
        // Every tree the open holds, for what is read of all of them alike.
        let every_tree = || {
            trees.values().chain(
                namespace_sets
                    .values()
                    .flat_map(|set| set.iter().map(|(_, tree)| tree)),
            )
        };

        // Restore seqno counter to (max_persisted_seqno + 1) so reads see all
        // data written in a previous session. seqno_filter uses strict <, so
        // `get()` must be > the highest written seqno for those writes to be visible.
        let max_persisted = every_tree()
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
        let max_floor = every_tree()
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

        // A replacement cut short before its contents were complete is
        // dropped, and a retired copy still holding entries is cleared, at
        // seqnos above everything the store holds.
        for set in namespace_sets.values() {
            if let (Some(idx), Some(installations)) = (set.get(Partition::Idx), set.installations())
            {
                installations.settle(idx, seqno.next())?;
            }
        }
        // A namespace the catalog marks dropping once its records are whole
        // (a drop the replay brought back) is not served: its set closes and
        // its trees go.
        for (id, record) in namespace_open::catalog_at(&deployment_tree, lsm_tree::SeqNo::MAX)? {
            if record.state != coordinode_core::namespace::NamespaceState::Live
                && namespace_sets.remove(&id).is_some()
            {
                remove_namespace_dirs(&namespace_dirs(&config.endpoints, id), &footprints);
            }
        }

        // No coverage base is written at open, and nothing is flushed for one:
        // a tree with no record is replayed exactly from a journal that holds
        // every entry from the first (above), and its first flush carries the
        // markers of what it took.
        let (oplog, journal_flusher, coverage) = match oplog {
            Some(journal) => {
                let coverage = Coverage::new(journal.next_index());
                let journal = Arc::new(Mutex::new(journal));
                let flusher = crate::engine::oplog_journal::JournalFlusher::start(
                    &journal,
                    std::time::Duration::from_millis(config.volatile_flush_interval_ms),
                )?;
                (Some(journal), Some(flusher), Some(coverage))
            }
            None => (None, None, None),
        };
        let local_flusher = match &local_journal {
            Some(local) => Some(crate::engine::oplog_journal::JournalFlusher::start(
                local.log(),
                std::time::Duration::from_millis(config.volatile_flush_interval_ms),
            )?),
            None => None,
        };
        // The journal a flush syncs before it writes a table: a table never
        // holds an effect of an entry that journal could still lose. A store
        // keeps one of the two; one with its own journal drains the local
        // one at open, before any flush.
        let flush_journal = oplog
            .clone()
            .or_else(|| local_journal.as_ref().map(|local| Arc::clone(local.log())));

        // Open tiered cache if configured. A layer under a capped endpoint is
        // reserved first; one that does not fit leaves the cache off, which
        // costs reads their cache and loses nothing.
        let tiered_cache = if config.cache.is_enabled() {
            match reserve_cache_layers(&config.cache, &footprints) {
                Err(e) => {
                    tracing::warn!(error = %e, "tiered cache does not fit a hard limit, continuing without");
                    None
                }
                Ok(()) => match TieredCache::open(&config.cache) {
                    Ok(cache) => Some(cache),
                    Err(e) => {
                        release_cache_layers(&config.cache, &footprints);
                        tracing::warn!(error = %e, "tiered cache failed to open, continuing without");
                        None
                    }
                },
            }
        } else {
            None
        };

        // Start background flush manager. Its monitor sleeps until the write
        // path's trigger, a finished flush or a memtable's age wakes it; the
        // compaction monitor sleeps until a flush or a compaction wakes it.
        let flush_wake = Arc::new(coordinode_core::txn::wake::Wake::default());
        let compaction_wake = Arc::new(coordinode_core::txn::wake::Wake::default());
        let flush_thresholds = Arc::new(crate::engine::flush::FlushThresholds::new(
            config.max_write_buffer_bytes,
            |part| {
                flush_landing
                    .get(&part)
                    .and_then(|path| footprints.for_path(path))
                    .cloned()
            },
        ));
        let flush_trigger = Arc::new(crate::engine::flush::FlushTrigger::new(
            Arc::clone(&flush_wake),
            Arc::clone(&flush_thresholds),
        ));
        // The memtable axis of the write-pressure verdict, latched by the
        // flush monitor, which writes and finished flushes wake.
        let memtable_pressure = Arc::new(AtomicU8::new(0));

        // From here on the trees are found by set: the background services
        // list them afresh on every pass.
        let sets = open_sets_of(&trees, namespace_sets);
        // Every namespace's columnar tables, listed afresh on each pass of
        // the flush and compaction monitors.
        #[cfg(feature = "columnar")]
        let table_source = || -> Option<crate::engine::flush::TableSource> {
            let sets = Arc::clone(&sets);
            Some(Box::new(move || sets.table_trees()))
        };
        #[cfg(not(feature = "columnar"))]
        let table_source = || -> Option<crate::engine::flush::TableSource> { None };
        #[cfg(not(feature = "columnar"))]
        let _ = &capacity_pool;
        let flushed_tables = table_source();
        let table_landed = Arc::new(coordinode_core::txn::wake::Wake::default());
        let flush_manager = FlushManager::start(
            Arc::clone(&sets),
            gc_controller.watermarks(),
            Arc::clone(&flush_thresholds),
            config.max_sealed_memtables,
            config.flush_workers,
            config.max_memtable_age_secs,
            flush_wake,
            Arc::clone(&compaction_wake),
            flush_journal,
            crate::engine::flush::MemtablePressure {
                budget: write_buffer,
                thresholds: config.backpressure.memtable_thresholds(),
                tier: Arc::clone(&memtable_pressure),
            },
            flushed_tables,
            footprints.clone(),
            Arc::clone(&table_landed),
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
            Arc::clone(&sets),
            gc_controller.watermarks(),
            config.compaction_workers,
            config.compaction_l0_urgent_threshold,
            config.backpressure.bytes_slowdown,
            Arc::clone(&write_pressure),
            Arc::clone(&compaction_wake),
            config.compaction_strategy,
            table_source(),
        )?;
        // The rate client writes are held to while a partition is past its
        // slowdown threshold.
        let write_slowdown = Arc::new(lsm_tree::rate_limiter::RateLimiter::new(
            config.write_slowdown_bytes_per_sec,
        ));

        info!(
            path = %config.data_dir().display(),
            endpoints = config.endpoints.len(),
            partitions = trees.len(),
            cache_layers = config.cache.layers.len(),
            flush_workers = config.flush_workers,
            compaction_workers = config.compaction_workers,
            "storage engine opened"
        );

        // The capacity tracker reports the footprints, which were measured
        // from the endpoints at open: it starts from them, before the first
        // scan tick.
        let capacity_arc = {
            let tracker = crate::engine::capacity::CapacityTracker::new(&config.endpoints);
            for footprint in footprints.iter() {
                if let Some(usage) = tracker.get(footprint.id()) {
                    usage
                        .used_bytes
                        .store(footprint.charged(), std::sync::atomic::Ordering::Release);
                }
            }
            Arc::new(tracker)
        };

        // Spawn the background scanner. It samples every partition's
        // footprint on every deployment; the capacity report and cascade
        // eviction run only where an endpoint has a `hard_limit_bytes`. The
        // closure captures cheap snapshots (AnyTree is internally Arc'd) so
        // no circular reference with `Self`.
        let node_counts = Arc::new(crate::engine::owned_metrics::NodeCounts::default());
        let capacity_scanner = {
            let node_counts_c = Arc::clone(&node_counts);
            let tracked = config.endpoints.iter().any(|ep| ep.hard_limit_bytes > 0);
            let tracker_c = Arc::clone(&capacity_arc);
            let endpoints_c = config.endpoints.clone();
            let sets_c = Arc::clone(&sets);
            let seqno_c = Arc::clone(&seqno);
            let footprints_c = footprints.clone();
            let gc_controller_c = Arc::clone(&gc_controller);
            let interval = std::time::Duration::from_secs(5);
            Some(
                crate::engine::capacity::CapacityScanner::start(interval, move || {
                    publish_label_counts(&sets_c, &seqno_c, &node_counts_c);
                    let trees_c = sets_c.list();
                    #[cfg(feature = "lsm-metrics")]
                    publish_tree_metrics(&trees_c);
                    if !tracked {
                        publish_footprint(&trees_c);
                        return;
                    }
                    run_capacity_refresh(&tracker_c, &endpoints_c, &trees_c, &footprints_c, |id| {
                        run_cascade_evict(
                            &endpoints_c,
                            &sets_c,
                            &seqno_c,
                            &|part| gc_controller_c.maintenance_compaction_threshold_for(part),
                            id,
                        )
                    });
                })
                .map_err(|e| StorageError::InvalidConfig(format!("spawn capacity scanner: {e}")))?,
            )
        };
        // Free space is read here, never on a write: see `SpaceGuard`.
        let space = Arc::new(
            crate::engine::space::SpaceGuard::new(config).with_footprints(footprints.clone()),
        );
        let space_monitor = Some(space.start_monitor()?);

        // Restore the shared seqno past any persisted columnar write (the
        // partition-only restore above misses a columnar-only seqno line), then
        // replay the journalled columnar rows that did not reach an SST — the
        // mirror of the partition replay loop, routed to the table registry.
        #[cfg(feature = "columnar")]
        {
            use lsm_tree::AbstractTree;
            let registries: HashMap<NamespaceId, Arc<crate::columnar::ColumnarTableRegistry>> =
                sets.namespace_sets()
                    .iter()
                    .filter_map(|set| match (set.scope(), set.columnar()) {
                        (Scope::Namespace(ns), Some(registry)) => Some((ns, Arc::clone(registry))),
                        _ => None,
                    })
                    .collect();
            for registry in registries.values() {
                if let Some(max) = registry.max_highest_seqno() {
                    seqno.fetch_max(max + 1);
                }
            }
            // Each table's own coverage record decides, like a partition's. A
            // table that is gone was dropped: its rows are not brought back,
            // and a table created later under the same name starts its record
            // past every entry journalled before it (`open_columnar_table`).
            let mut covered: HashMap<(NamespaceId, String), TreeCoverage> = HashMap::new();
            if let Some(coverage) = &coverage {
                for (&ns, registry) in &registries {
                    for table_id in registry.table_ids() {
                        if let Some(tree) = registry.get(&table_id) {
                            let c = TreeCoverage::read(&tree, Domain::Journal)?;
                            // Markers already on disk are folded by the next
                            // fold.
                            for index in c.marked_indices() {
                                coverage.note_table_marker(ns, &table_id, index);
                            }
                            covered.insert((ns, table_id), c);
                        }
                    }
                }
            }
            let mut touched: std::collections::HashSet<(NamespaceId, String)> =
                std::collections::HashSet::new();
            let mut replayed = 0usize;
            for ColumnarReplay {
                namespace,
                table_id,
                key,
                value,
                ts,
                index,
            } in columnar_replay
            {
                let Some(registry) = registries.get(&namespace) else {
                    continue;
                };
                let table = (namespace, table_id);
                let (Some(tree), Some(c)) = (registry.get(&table.1), covered.get(&table)) else {
                    continue;
                };
                if !c.contains(index, 0) {
                    apply_columnar_row(&tree, key, value, ts, Some(index))?;
                    if let Some(coverage) = &coverage {
                        coverage.note_table_marker(namespace, &table.1, index);
                    }
                    touched.insert(table);
                    replayed += 1;
                }
            }
            for (namespace, table_id) in &touched {
                if let Some(tree) = registries
                    .get(namespace)
                    .and_then(|registry| registry.get(table_id))
                {
                    tree.flush_active_memtable(0)?;
                    tree.persist_flushes()?;
                }
            }
            if replayed > 0 {
                tracing::info!(
                    replayed,
                    "oplog: replayed un-flushed columnar rows from journal on open"
                );
            }
        }

        let fs: Arc<dyn lsm_tree::fs::Fs> = config
            .fs
            .clone()
            .unwrap_or_else(|| Arc::new(lsm_tree::fs::StdFs));
        // A capture a crash left behind belongs to no copy.
        remove_capture_with(
            &*fs,
            &config.data_dir().join(raft_coverage::PARTITION_CAPTURE_DIR),
        )?;

        // Durable evidence, not a running state machine: a member's store
        // opened offline holds its group's history all the same.
        let group_history = match config.select_oplog_endpoint(0) {
            Ok(endpoint) => group_history_evidence(&endpoint.path, &trees)?.is_some(),
            Err(_) => false,
        };
        // A rebuild of the deployment's records still owed after the open (a
        // transfer from a peer cut short) leaves the tree without the ledger
        // it is to install.
        let schema_owed = durable_intents
            && read_rebuild_intents(config.data_dir())?
                .trees
                .contains(&TreeId::system(Partition::Deployment));
        let coordinator = LocalMultiModalCoordinator::new(
            sets,
            Arc::clone(&seqno),
            cache,
            gc_watermark,
            gc_controller,
            flush_trigger,
        );
        // Brought up to the clock the open moved, the window held all along.
        coordinator.advance_gc_watermark();
        // The open succeeded: a migrated directory now records its format.
        // An open refused above leaves the old marker, so the store it
        // refused stays as its writer left it.
        crate::format::settle(&format, &footprints)?;
        let mut engine = Self {
            flush_manager: Some(flush_manager),
            compaction_scheduler: Some(compaction_scheduler),
            coordinator,
            claim_registry: crate::engine::claims::ClaimRegistry::new(config.max_invariant_claims),
            schema_generation: AtomicU64::new(0),
            repairs: parking_lot::Mutex::new(RepairMarks::default()),
            metadata_decisions: parking_lot::Mutex::new(()),
            operation_decisions: crate::engine::operations::DecisionLocks::default(),
            access_decisions: parking_lot::RwLock::new(()),
            field_dictionary_generation: AtomicU64::new(0),
            field_dictionary_epoch: AtomicU64::new(0),
            pending_commits,
            closure_frontier: AtomicU64::new(0),
            raft_effects_ceiling: AtomicU64::new(0),
            closure: parking_lot::Mutex::new(ClosureState::default()),
            flush_thresholds,
            closure_held,
            snapshot_wait: std::sync::atomic::AtomicU64::new(config.snapshot_wait_ms),
            node_shard: std::sync::atomic::AtomicU16::new(config.node_shard),
            flush_policy: config.flush_policy,
            oplog_config: OplogJournalConfig::from(config),
            tiered_cache,
            access_tracker: AccessTracker::new(),
            data_dir: config.data_dir().to_path_buf(),
            fs,
            tree_sync: config.oplog_sync_method.tree_sync_mode(),
            deployment_verified,
            ledger_owed: std::sync::atomic::AtomicBool::new(schema_owed),
            maintenance,
            write_slowdown,
            compaction_wake,
            endpoints: config.endpoints.clone(),
            capacity: capacity_arc,
            capacity_scanner,
            space_monitor,
            oplog,
            journal_flusher,
            local_journal,
            local_flusher,
            memtable_flushes: AtomicU64::new(0),
            coverage,
            oracle,
            table_metrics,
            node_counts,
            open_repairs,
            write_pressure,
            memtable_pressure,
            raft_fence: parking_lot::RwLock::new(None),
            log_group: parking_lot::RwLock::new(None),
            raft_log_keep_from: AtomicU64::new(u64::MAX),
            partition_captures: AtomicU64::new(0),
            promise_copy: parking_lot::Mutex::new(crate::engine::promise::CopyWriter::default()),
            write_taps: Arc::new(crate::engine::tap::WriteTaps::default()),
            applied_feed: Arc::new(crate::engine::applied::AppliedFeed::default()),
            index_feed_capacity: config.index_feed_capacity.max(1),
            text_index_batch: config.text_index_batch.max(1),
            text_index_retry: std::time::Duration::from_millis(config.text_index_retry_ms.max(1)),
            open_transactions: Arc::new(crate::engine::open_txns::OpenTransactions::default()),
            tree_set: Arc::new(crate::engine::tree_set::TreeSetGate::default()),
            catalog: parking_lot::Mutex::new(None),
            space,
            footprints,
            footprint_reclaim: parking_lot::Mutex::new(()),
            table_landed,
            #[cfg(test)]
            fail_journal_purge: std::sync::atomic::AtomicBool::new(false),
            #[cfg(any(test, feature = "test-faults"))]
            on_cut_bound: parking_lot::Mutex::new(None),
            #[cfg(test)]
            on_tree_pinned: parking_lot::Mutex::new(None),
            #[cfg(any(test, feature = "test-faults"))]
            on_rebuild_cleared: parking_lot::Mutex::new(None),
            #[cfg(any(test, feature = "test-faults"))]
            on_table_dropped: parking_lot::Mutex::new(None),
            #[cfg(any(test, feature = "test-faults"))]
            on_columnar_row: parking_lot::Mutex::new(None),
            #[cfg(any(test, feature = "test-faults"))]
            on_table_created: parking_lot::Mutex::new(None),
            #[cfg(any(test, feature = "test-faults"))]
            on_trees_pinned: parking_lot::Mutex::new(None),
            #[cfg(test)]
            on_journalled: parking_lot::Mutex::new(None),
            #[cfg(test)]
            on_segment_found: parking_lot::Mutex::new(None),
            journal_keep_from: AtomicU64::new(u64::MAX),
            consumer_journal_floor: AtomicU64::new(if protections_pending { 0 } else { u64::MAX }),
            protections_pending: std::sync::atomic::AtomicBool::new(protections_pending),
            journal_holds: Default::default(),
            decoded_logs: parking_lot::Mutex::new(rustc_hash::FxHashMap::default()),
            log_floors: parking_lot::Mutex::new(rustc_hash::FxHashMap::default()),
            decoded_memory: crate::engine::decoded_log::DecodedMemory::new(
                crate::engine::decoded_log::DEFAULT_DECODED_BYTES,
            ),
            group_history,
            me: std::sync::OnceLock::new(),
            opener,
            namespace_opening: parking_lot::Mutex::new(rustc_hash::FxHashMap::default()),
            _directory_locks: directory_locks,
        };
        engine.drain_local_journal()?;
        // A repair that lost nothing owes only the tail the journal or the
        // log replays, which the open above has done: the trees stop owing
        // its report. A lossy one stays owed until a rebuild restores it.
        for repair in &engine.open_repairs {
            if !repair.is_lossy() {
                engine.acknowledge_open_repair(repair.tree)?;
            }
        }
        #[cfg(feature = "columnar")]
        for set in engine.coordinator.sets().namespace_sets() {
            let settled: Vec<String> = set
                .table_repairs()
                .lock()
                .iter()
                .filter(|repair| !repair.is_lossy())
                .map(|repair| repair.table.clone())
                .collect();
            for table in settled {
                acknowledge_table_repair(&set, &table)?;
            }
        }
        Ok(engine.shared())
    }

    /// Each live `STORAGE COLUMNAR` table's write-pressure series, for an
    /// operations endpoint to render beside the process's other metrics. A
    /// table's series exists exactly while the table does.
    #[must_use]
    pub fn table_metrics(&self) -> &crate::engine::owned_metrics::TableMetrics {
        &self.table_metrics
    }

    /// The stored nodes of each label that has any, for an operations
    /// endpoint to render. A label whose last node is gone has no series.
    #[must_use]
    pub fn node_counts(&self) -> &crate::engine::owned_metrics::NodeCounts {
        &self.node_counts
    }

    /// The columnar tables of every open namespace whose tree an open
    /// repaired with loss and which still owe a rebuild from a checkpoint
    /// plus the journal, with their namespace. Empty in a build without
    /// columnar support.
    #[must_use]
    pub fn lossy_tables(&self) -> Vec<(NamespaceId, String)> {
        #[cfg(feature = "columnar")]
        {
            let mut out = Vec::new();
            for set in self.coordinator.sets().namespace_sets() {
                let Scope::Namespace(namespace) = set.scope() else {
                    continue;
                };
                out.extend(
                    set.table_repairs()
                        .lock()
                        .iter()
                        .map(|repair| (namespace, repair.table.clone())),
                );
            }
            out
        }
        #[cfg(not(feature = "columnar"))]
        Vec::new()
    }

    /// Let tree `id` stop owing the report of the repair an open ran on it,
    /// once its replay obligation is met: without it every later open hands
    /// the same report out again. `false` when no repair of the tree is owed.
    ///
    /// # Errors
    ///
    /// The tree cannot install the version without the report.
    fn acknowledge_open_repair(&self, id: TreeId) -> StorageResult<bool> {
        use lsm_tree::AbstractTree as _;
        let Some(repair) = self.open_repairs.iter().find(|r| r.tree == id) else {
            return Ok(false);
        };
        Ok(self
            .tree_of(id)?
            .acknowledge_repair_report(repair.report.id)?)
    }

    /// On a store with its own journal, fold what a member-local journal
    /// left (an open without a journal wrote it) into the trees, flush them
    /// and retire that journal: from here on the store's journal carries
    /// these records.
    fn drain_local_journal(&mut self) -> StorageResult<()> {
        if self.oplog.is_none() {
            return Ok(());
        }
        let Some(local) = self.local_journal.take() else {
            return Ok(());
        };
        drop(self.local_flusher.take());
        // The trees' flushes sync only the store's journal: the local one is
        // synced first, so no table takes an entry it could still lose.
        local.sync()?;
        let at = self.next_seqno();
        let store = self.local_record_set();
        local.compact(&self.coordinator.partition_trees(&store), at, |tree| {
            self.flush_memtables(tree)
        })?;
        // Every tree now covers it durably, so a later open has nothing to
        // replay and no journal to drain again.
        local.retire()
    }

    /// The free-space guard: new writes are refused while it is paused, and
    /// a layer that writes ahead of the engine (a consensus log) checks it
    /// first.
    pub fn space(&self) -> &Arc<crate::engine::space::SpaceGuard> {
        &self.space
    }

    /// The owned footprint of every endpoint with a hard limit.
    pub fn footprints(&self) -> &crate::engine::footprint::Footprints {
        &self.footprints
    }

    /// Change the hard limit of endpoint `endpoint_id` without a restart, and
    /// its recovery reserve (`None`:
    /// [`default_recovery_reserve`](crate::engine::config::default_recovery_reserve)). A raised
    /// limit admits writes at once. A limit below what the endpoint holds
    /// deletes nothing: it is pending, client writes are refused and the
    /// store's own growth is held to what it holds plus the reserve until
    /// reclamation brings it under the limit.
    ///
    /// # Errors
    ///
    /// [`StorageError::InvalidConfig`] for an unknown endpoint, a limit of
    /// `0` or below [`crate::engine::config::MIN_HARD_LIMIT_BYTES`], or an
    /// endpoint opened without a hard limit: the files it writes are not
    /// charged, so a limit there takes a restart with it configured.
    pub fn set_hard_limit(
        &self,
        endpoint_id: &str,
        limit: u64,
        reserve: Option<u64>,
    ) -> StorageResult<()> {
        if limit == 0 {
            return Err(StorageError::InvalidConfig(
                "a hard limit of 0 removes the limit, which takes a restart".into(),
            ));
        }
        if limit < crate::engine::config::MIN_HARD_LIMIT_BYTES {
            return Err(StorageError::InvalidConfig(format!(
                "a hard limit of {limit} bytes is below the supported minimum of {} bytes",
                crate::engine::config::MIN_HARD_LIMIT_BYTES
            )));
        }
        let footprint = self.footprints.get(endpoint_id).ok_or_else(|| {
            StorageError::InvalidConfig(format!(
                "endpoint {endpoint_id:?} has no hard limit to change: one is set in its \
                 configuration and takes effect at the next open"
            ))
        })?;
        let reserve = reserve.unwrap_or(crate::engine::config::default_recovery_reserve(limit));
        footprint.set_limit(limit, reserve);
        // The flushes that land on it fit a new reserve; the monitor judges
        // the memtables against it at once.
        self.flush_thresholds.refresh();
        self.coordinator.flush_trigger().wrote_unmeasured();
        tracing::info!(
            endpoint = endpoint_id,
            hard_limit_bytes = limit,
            recovery_reserve_bytes = footprint.reserve(),
            pending = footprint.pending_limit().is_some(),
            "endpoint hard limit changed"
        );
        Ok(())
    }

    /// Admit a client write of `mutations` before anything of it is written:
    /// the free-space reserve of the volumes, then the owned footprint of
    /// every endpoint with a hard limit, which must hold its journal entry. A
    /// refusal on the footprint starts
    /// the reclamation that can make room: a flush of the memtables and the
    /// removal of journal segments the trees then cover.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutOfSpace`] below the free-space reserve,
    /// [`StorageError::CapacityExhausted`] when the write does not fit the
    /// footprint.
    pub fn admit_write(
        &self,
        mutations: &[Mutation],
    ) -> StorageResult<crate::engine::footprint::Admitted> {
        self.admit_proposal(mutations, crate::engine::footprint::Admit::Client)
    }

    /// Admit `mutations` as a write of `class` before it is proposed to the
    /// group's log: the free-space reserve, then the owned footprint of every
    /// endpoint with a hard limit, reclaiming once before a refusal. The
    /// returned admission holds the room until it drops, which the leader
    /// keeps until the entry is in its log: a log append that met a limit
    /// inside its write would stop consensus on this member.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutOfSpace`] below the free-space reserve,
    /// [`StorageError::CapacityExhausted`] when the write does not fit the
    /// footprint.
    pub fn admit_proposal(
        &self,
        mutations: &[Mutation],
        class: crate::engine::footprint::Admit,
    ) -> StorageResult<crate::engine::footprint::Admitted> {
        let class = class.on_this_thread();
        match class {
            crate::engine::footprint::Admit::Client => self.space.admit_free_space()?,
            crate::engine::footprint::Admit::Recovery => {
                self.space.admit_free_space_recovery()?;
            }
        }
        if self.footprints.is_empty() {
            return Ok(crate::engine::footprint::Admitted::default());
        }
        // Sized only under a hard limit, which needs it anyway: the size
        // also spends the free-space headroom then.
        let bytes = crate::engine::footprint::write_footprint(mutations);
        self.space.spend(bytes);
        self.admit_footprint(bytes, class)
    }

    /// Admit, on the owned footprint, a write of the partitions `parts` that
    /// adds about `bytes`: the store's own metadata (Schema, Raft) as a
    /// recovery record, which may use the reserve, anything else as a client
    /// write.
    ///
    /// # Errors
    ///
    /// [`StorageError::CapacityExhausted`] when the write does not fit.
    pub(crate) fn admit_partition_write(
        &self,
        parts: impl IntoIterator<Item = Partition, IntoIter: Clone>,
        bytes: u64,
    ) -> StorageResult<crate::engine::footprint::Admitted> {
        use crate::engine::footprint::Admit;
        if self.footprints.is_empty() {
            return Ok(crate::engine::footprint::Admitted::default());
        }
        let class = if parts.into_iter().all(|part| {
            matches!(
                part,
                Partition::Schema | Partition::Deployment | Partition::Raft
            )
        }) {
            Admit::Recovery
        } else {
            Admit::Client.on_this_thread()
        };
        self.admit_footprint(bytes, class)
    }

    /// Admit `bytes` on every capped endpoint as `class`, reclaiming once
    /// when they do not fit.
    fn admit_footprint(
        &self,
        bytes: u64,
        class: crate::engine::footprint::Admit,
    ) -> StorageResult<crate::engine::footprint::Admitted> {
        if let Ok(admitted) = self.footprints.admit(bytes, class) {
            return Ok(admitted);
        }
        self.reclaim_footprint();
        // What the reclamation freed admits the write at once; what it could
        // not leaves the refusal standing, with the footprint as it is now.
        self.footprints.admit(bytes, class)
    }

    /// The cached write-pressure verdict (two relaxed atomic loads): the
    /// worse of the per-partition table verdict the compaction monitor
    /// latches and the memtable verdict the flush monitor latches. `Stop`
    /// means new client writes should be rejected (retryable) until
    /// compaction or flushes drain; nothing in the engine sleeps on it.
    #[must_use]
    pub fn write_pressure(&self) -> WritePressure {
        let tables = self
            .write_pressure
            .load(std::sync::atomic::Ordering::Relaxed);
        let memtables = self
            .memtable_pressure
            .load(std::sync::atomic::Ordering::Relaxed);
        WritePressure::from_u8(tables.max(memtables))
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
}

impl StorageEngine {
    /// The namespace's `STORAGE COLUMNAR` table registry.
    #[cfg(feature = "columnar")]
    fn columnar(&self) -> StorageResult<&Arc<crate::columnar::ColumnarTableRegistry>> {
        let bound = self.bound()?;
        bound.set.columnar().ok_or_else(|| {
            StorageError::InvalidConfig(format!("{} holds no table registry", bound.namespace))
        })
    }

    /// The write-pressure verdict a write to `STORAGE COLUMNAR` table
    /// `table_id` meets: the worse of the table's own (its L0 height and
    /// compaction debt, latched by the compaction monitor) and the memtables'
    /// shared one. The partitions' verdict does not hold the table's writes
    /// back, nor does the table's hold theirs. `Stop` means its writes should
    /// be rejected (retryable) until compaction or flushes drain.
    #[cfg(feature = "columnar")]
    #[must_use]
    pub fn columnar_write_pressure(&self, table_id: &str) -> WritePressure {
        let table = self
            .columnar()
            .map_or(0, |registry| registry.pressure(table_id));
        let memtables = self
            .memtable_pressure
            .load(std::sync::atomic::Ordering::Relaxed);
        WritePressure::from_u8(table.max(memtables))
    }

    /// The columnar tables whose tree an open repaired with loss and which
    /// still owe a rebuild from a checkpoint plus the journal, by id. Empty
    /// in a build without columnar support.
    #[must_use]
    pub fn lossy_table_repairs(&self) -> Vec<String> {
        #[cfg(feature = "columnar")]
        {
            self.set().map_or_else(
                |_| Vec::new(),
                |set| {
                    set.table_repairs()
                        .lock()
                        .iter()
                        .map(|repair| repair.table.clone())
                        .collect()
                },
            )
        }
        #[cfg(not(feature = "columnar"))]
        Vec::new()
    }

    /// Let `table`'s tree stop owing the report of the repair an open ran on
    /// it, once what it lost is restored. `false` when none is owed.
    #[cfg(feature = "columnar")]
    fn acknowledge_table_repair(&self, table: &str) -> StorageResult<bool> {
        acknowledge_table_repair(self.set()?, table)
    }

    /// Create (or open if it already exists) the columnar tree for a
    /// `STORAGE COLUMNAR` table, returning a handle to it. Idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the tree cannot be opened.
    #[cfg(feature = "columnar")]
    pub fn create_columnar_table(&self, table_id: &str) -> StorageResult<lsm_tree::AnyTree> {
        let _held = self.hold_tree_set();
        self.open_columnar_table(table_id).map(|(tree, _, _)| tree)
    }

    /// [`Self::create_columnar_table`] within an operation that holds the
    /// set of trees as `_held` from before the tree is made until its
    /// catalog change commits. Returns the stamp of the tree it got, read
    /// with the tree under the registry's lock: the timestamp it was created
    /// at in this process, 0 for a table the store held when it opened. A
    /// transaction that creates the table keeps the stamp, so its rows go
    /// into that tree and into no other made under the name. No handle of
    /// the tree is kept: a drop of the table waits for every one.
    ///
    /// # Errors
    ///
    /// As [`Self::create_columnar_table`].
    #[cfg(feature = "columnar")]
    pub fn create_columnar_table_held(
        &self,
        table_id: &str,
        _held: &crate::engine::tree_set::TreeSetHold,
    ) -> StorageResult<lsm_tree::SeqNo> {
        let (_, _, created_at) = self.open_columnar_table(table_id)?;
        // Taken once, and called with its lock given back: what it runs may
        // create a table itself.
        #[cfg(any(test, feature = "test-faults"))]
        {
            let hook = self.on_table_created.lock().take();
            if let Some(mut hook) = hook {
                hook();
            }
        }
        Ok(created_at)
    }

    /// The tree of `table_id`, created on first use, with the table's own
    /// write-pressure tier and the tree's stamp. A table created here holds none of the entries
    /// journalled before it existed (those belonged to a dropped table of the
    /// same name, if any), so on a journalled engine it gets a coverage base
    /// at the journal's next index, made durable before any row can reach it.
    #[cfg(feature = "columnar")]
    fn open_columnar_table(&self, table_id: &str) -> StorageResult<(lsm_tree::AnyTree, u8, u64)> {
        self.columnar()?.open_judged(
            table_id,
            || self.next_seqno(),
            |tree| {
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
                tree.persist_flushes()?;
                Ok(())
            },
        )
    }

    /// Handle to an existing `STORAGE COLUMNAR` table tree, or `None` if no such
    /// table is registered on this node.
    #[cfg(feature = "columnar")]
    pub fn columnar_table_tree(&self, table_id: &str) -> Option<lsm_tree::AnyTree> {
        self.columnar().ok()?.get(table_id)
    }

    /// Drop a `STORAGE COLUMNAR` table: release its tree handle and delete its
    /// on-disk directory. Idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the directory removal fails.
    #[cfg(feature = "columnar")]
    pub fn drop_columnar_table(&self, table_id: &str) -> StorageResult<()> {
        let held = self.hold_tree_set();
        self.drop_columnar_table_held(table_id, &held)
    }

    /// [`Self::drop_columnar_table`] within an operation that holds the set
    /// of trees as `_held` from before the tree is removed until its catalog
    /// change commits.
    ///
    /// # Errors
    ///
    /// As [`Self::drop_columnar_table`].
    #[cfg(feature = "columnar")]
    pub fn drop_columnar_table_held(
        &self,
        table_id: &str,
        _held: &crate::engine::tree_set::TreeSetHold,
    ) -> StorageResult<()> {
        self.columnar()?.drop_table(table_id)?;
        #[cfg(any(test, feature = "test-faults"))]
        if let Some(hook) = self.on_table_dropped.lock().as_mut() {
            hook();
        }
        Ok(())
    }

    /// Insert a row `(key, value)` into a `STORAGE COLUMNAR` table's tree at
    /// `seqno`, creating the tree on first use. The engine transposes rows to
    /// columnar blocks at flush and reconstructs them on read. The call is
    /// one operation on the set of trees; an operation writing many rows
    /// holds the set once and writes them with [`Self::columnar_insert_held`].
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
        let held = self.hold_tree_set();
        self.columnar_insert_held(&held, table_id, key, value, seqno)
    }

    /// [`Self::columnar_insert`] within an operation that holds the set of
    /// trees as `_held` from its first row until its last.
    ///
    /// # Errors
    ///
    /// As [`Self::columnar_insert`].
    #[cfg(feature = "columnar")]
    pub fn columnar_insert_held(
        &self,
        _held: &crate::engine::tree_set::TreeSetHold,
        table_id: &str,
        key: Vec<u8>,
        value: Vec<u8>,
        seqno: lsm_tree::SeqNo,
    ) -> StorageResult<()> {
        let (tree, table_tier, _) = self.open_columnar_table(table_id)?;
        self.insert_columnar_row(table_id, &tree, table_tier, key, value, seqno)
    }

    /// [`Self::columnar_insert_held`] into a table whose tree exists: a statement
    /// writes rows into the tree its table's creation made, never into a new
    /// one standing in for a tree a drop removed, so a tree under a table's
    /// name is the one that table was created with. `read_at` is the
    /// snapshot the statement read the table's definition at, and
    /// `created_here` the stamp of the tree its own transaction created
    /// under the name, if it did ([`Self::create_columnar_table_held`]): a
    /// tree newer than the snapshot and not that one belongs to a table
    /// created again under the name since, and the row is refused before it
    /// is written.
    ///
    /// # Errors
    ///
    /// [`StorageError::TableTreeMissing`] when the table has no tree;
    /// [`StorageError::TableCreatedSinceRead`] when its tree is newer than
    /// `read_at` and not the one `created_here` names; as
    /// [`Self::columnar_insert`] otherwise.
    #[cfg(feature = "columnar")]
    #[allow(clippy::too_many_arguments)] // one row, and what it is checked against
    pub fn columnar_insert_existing(
        &self,
        _held: &crate::engine::tree_set::TreeSetHold,
        read_at: lsm_tree::SeqNo,
        created_here: Option<lsm_tree::SeqNo>,
        table_id: &str,
        key: Vec<u8>,
        value: Vec<u8>,
        seqno: lsm_tree::SeqNo,
    ) -> StorageResult<()> {
        let (tree, table_tier, created_at) =
            self.columnar()?.get_stamped(table_id).ok_or_else(|| {
                StorageError::TableTreeMissing {
                    table: table_id.to_owned(),
                }
            })?;
        if created_at > read_at && created_here != Some(created_at) {
            return Err(StorageError::TableCreatedSinceRead {
                table: table_id.to_owned(),
            });
        }
        self.insert_columnar_row(table_id, &tree, table_tier, key, value, seqno)
    }

    /// [`Self::columnar_insert`] into `tree`, the table's tree already open,
    /// with its write-pressure tier `table_tier`.
    #[cfg(feature = "columnar")]
    fn insert_columnar_row(
        &self,
        table_id: &str,
        tree: &lsm_tree::AnyTree,
        table_tier: u8,
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
        // Under slowdown a row waits its turn at the write rate, as a commit
        // does; the stop tier refuses the statement before its first row.
        let memtable_tier = self
            .memtable_pressure
            .load(std::sync::atomic::Ordering::Relaxed);
        if WritePressure::from_u8(table_tier.max(memtable_tier)) == WritePressure::Slowdown {
            let delay = self.write_slowdown_delay((key.len() + value.len()) as u64);
            if !delay.is_zero() {
                std::thread::sleep(delay);
            }
        }
        let index = match &self.oplog {
            Some(oplog) => {
                let mut guard = oplog
                    .lock()
                    .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
                Some(guard.append_columnar(self.namespace(), table_id, &key, &value, seqno)?)
            }
            None => None,
        };
        let (added, memtable) = apply_columnar_row(tree, key, value, seqno, index)?;
        // A table's tree lands where the node partition's does.
        self.coordinator
            .flush_trigger()
            .wrote(Partition::Node, added, memtable);
        if let (Some(index), Some(coverage)) = (index, &self.coverage) {
            coverage.note_table_marker(self.namespace(), table_id, index);
            self.note_applied(coverage, index);
        }
        #[cfg(any(test, feature = "test-faults"))]
        if let Some(hook) = self.on_columnar_row.lock().as_mut() {
            hook();
        }
        Ok(())
    }

    /// End a write statement on a `STORAGE COLUMNAR` table: its rows are
    /// durable when this returns. With a journal they are already, as any
    /// commit's, and the flush manager flushes the table's memtable under the
    /// partitions' triggers. Without one the table's memtable is flushed now,
    /// the only copy its rows have. A no-op if the table has no tree.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the flush fails.
    #[cfg(feature = "columnar")]
    pub fn finish_columnar_statement(&self, table_id: &str) -> StorageResult<()> {
        if self.oplog.is_some() {
            return Ok(());
        }
        self.flush_columnar_table(table_id)
    }

    /// Flush one `STORAGE COLUMNAR` table's memtables durably, the journal
    /// synced first, while the table lives. A no-op if the table has no tree
    /// or its drop has taken it.
    #[cfg(feature = "columnar")]
    pub(crate) fn flush_columnar_table(&self, table_id: &str) -> StorageResult<()> {
        self.columnar()?
            .with_live(table_id, |tree| self.flush_memtables(tree))
            .unwrap_or(Ok(()))
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

        let Some(tree) = self.columnar()?.get(table_id) else {
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

        let Some(tree) = self.columnar()?.get(table_id) else {
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
            for table_id in self.columnar()?.table_ids() {
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
            let registry = self.columnar()?;
            let keep: std::collections::HashSet<&str> =
                tables.iter().map(|(id, _)| id.as_str()).collect();
            for table_id in registry.table_ids() {
                if !keep.contains(table_id.as_str()) {
                    registry.drop_table(&table_id)?;
                }
            }
            for (table_id, rows) in &tables {
                registry.drop_table(table_id)?;
                let (tree, _, _) = self.open_columnar_table(table_id)?;
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

    /// Drop every `STORAGE COLUMNAR` table: the start of installing a
    /// whole-store image table by table ([`Self::reset_columnar_table`],
    /// [`Self::append_columnar_rows`]).
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if a tree cannot be dropped.
    pub fn clear_columnar_tables(&self) -> StorageResult<()> {
        #[cfg(feature = "columnar")]
        {
            let registry = self.columnar()?;
            for table_id in registry.table_ids() {
                registry.drop_table(&table_id)?;
            }
        }
        Ok(())
    }

    /// Create `table_id` empty, dropping a table of that name: a table of a
    /// whole-store image, its rows following through
    /// [`Self::append_columnar_rows`].
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the tree cannot be dropped or
    /// created, and [`StorageError::InvalidConfig`] in a build without
    /// columnar support.
    pub fn reset_columnar_table(&self, table_id: &str) -> StorageResult<()> {
        #[cfg(feature = "columnar")]
        {
            self.columnar()?.drop_table(table_id)?;
            self.open_columnar_table(table_id).map(drop)
        }
        #[cfg(not(feature = "columnar"))]
        {
            Err(StorageError::InvalidConfig(format!(
                "the image carries STORAGE COLUMNAR table {table_id}; this build has no \
                 columnar support"
            )))
        }
    }

    /// Write `rows` (sorted by key, each above the last written) into
    /// `table_id`, which [`Self::reset_columnar_table`] created.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the tree cannot be opened or
    /// written, and [`StorageError::InvalidConfig`] in a build without
    /// columnar support.
    pub fn append_columnar_rows(
        &self,
        table_id: &str,
        rows: &[(Vec<u8>, Vec<u8>)],
    ) -> StorageResult<()> {
        #[cfg(feature = "columnar")]
        {
            let (tree, _, _) = self.open_columnar_table(table_id)?;
            let rows: Vec<crate::columnar::ColumnarRow<'_>> = rows
                .iter()
                .map(|(key, value)| crate::columnar::ColumnarRow { key, value })
                .collect();
            crate::columnar::write_columnar_rows(&tree, &rows)
        }
        #[cfg(not(feature = "columnar"))]
        {
            let _ = rows;
            Err(StorageError::InvalidConfig(format!(
                "the image carries STORAGE COLUMNAR table {table_id}; this build has no \
                 columnar support"
            )))
        }
    }
}

impl Deployment {
    /// The budget background maintenance reads and writes under: every
    /// tree's compactions and the scrub.
    pub fn maintenance_budget(&self) -> &Arc<lsm_tree::rate_limiter::RateLimiter> {
        &self.maintenance
    }

    /// Retune the maintenance budget to `bytes_per_sec` (`0`: unthrottled).
    /// Work already waiting on it is held to the new rate at once.
    pub fn set_maintenance_rate(&self, bytes_per_sec: u64) {
        self.maintenance.set_rate(bytes_per_sec);
    }

    /// Back the maintenance budget off, down to 5% of its rate, while the
    /// smoothed read latency of maintenance stays above `ceiling`; `None`
    /// keeps the budget at its rate.
    pub fn set_maintenance_backoff(&self, ceiling: Option<std::time::Duration>) {
        self.maintenance
            .set_latency_backoff(ceiling.map(lsm_tree::rate_limiter::LatencyBackoff::new));
    }

    /// Hold client writes to `bytes_per_sec` while a partition is past its
    /// slowdown threshold (`0`: full speed until the stop threshold).
    pub fn set_write_slowdown_rate(&self, bytes_per_sec: u64) {
        self.write_slowdown.set_rate(bytes_per_sec);
    }

    /// How long a commit of `bytes` waits under slowdown: its turn at the
    /// write rate, after every byte debited before it. Zero without a rate.
    pub(crate) fn write_slowdown_delay(&self, bytes: u64) -> std::time::Duration {
        if self.write_slowdown.rate() == 0 {
            return std::time::Duration::ZERO;
        }
        self.write_slowdown.reserve(bytes)
    }

    /// The timestamp oracle this engine stamps writes with, when opened
    /// via [`StorageEngine::open_with_oracle`]. Subsystems that apply
    /// externally stamped writes (the Raft state machine on followers)
    /// must advance this oracle so local MVCC readers observe them.
    pub fn oracle(&self) -> Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>> {
        self.oracle.clone()
    }

    /// The tree of a partition only the deployment holds
    /// ([`Partition::is_deployment`]).
    ///
    /// # Errors
    ///
    /// [`StorageError::PartitionNotFound`] for a namespace's partition.
    pub fn system_tree(&self, part: Partition) -> StorageResult<&lsm_tree::AnyTree> {
        self.coordinator.system_tree(part)
    }

    /// The tree `id`, when its set is open.
    ///
    /// # Errors
    ///
    /// [`StorageError::PartitionNotFound`] when the tree is not open.
    pub(crate) fn tree_of(&self, id: TreeId) -> StorageResult<lsm_tree::AnyTree> {
        self.coordinator
            .sets()
            .tree(id)
            .ok_or_else(|| StorageError::PartitionNotFound { name: id.name() })
    }

    /// Namespace `namespace`'s set of trees, opened now when the catalog
    /// holds the namespace live and its set is not open yet.
    ///
    /// # Errors
    ///
    /// [`StorageError::UnknownNamespace`] when the catalog holds no live
    /// namespace of that id; its trees cannot be opened.
    pub(crate) fn namespace_set(&self, namespace: NamespaceId) -> StorageResult<Arc<SetTrees>> {
        if let Some(set) = self.coordinator.sets().namespace(namespace) {
            set.touch();
            return Ok(set);
        }
        let mut opening = self.namespace_opening.lock();
        if let Some(set) = self.coordinator.sets().namespace(namespace) {
            set.touch();
            return Ok(set);
        }
        if !namespace_open::live_at(
            self.system_tree(Partition::Deployment)?,
            namespace,
            lsm_tree::SeqNo::MAX,
        )? {
            return Err(StorageError::UnknownNamespace { namespace });
        }
        // Trees closed for being idle are opened again only once nothing of
        // the closed handles touches their directories.
        if let Some(closing) = opening.remove(&namespace) {
            for tree in closing {
                tree.wait();
            }
        }
        let (set, repairs) = self
            .opener
            .open_namespace(namespace, self.system_tree(Partition::Deployment)?)?;
        for repair in repairs {
            // A namespace that opens after the store is one created since:
            // a repair of its trees has nothing earlier to lose.
            tracing::warn!(
                tree = %repair.tree,
                lossy = repair.is_lossy(),
                "a namespace's tree opened after the store was repaired"
            );
        }
        if let (Some(idx), Some(installations)) = (set.get(Partition::Idx), set.installations()) {
            installations.settle(idx, self.next_seqno())?;
        }
        let set = Arc::new(set);
        self.coordinator.sets().insert(Arc::clone(&set));
        // Brought up to the clock the settle moved, the window held all
        // along, as the store's own open does.
        self.coordinator.advance_gc_watermark();
        Ok(set)
    }
}

impl StorageEngine {
    /// Get a tree handle by logical partition: the namespace's own, or the
    /// system set's for a partition only the deployment holds.
    ///
    /// The tree holds keys as stored: in the index tree a generation key
    /// sits under this member's installation of the generation, not under
    /// the generation itself. Read and write through the engine's methods,
    /// which translate.
    pub fn tree(&self, part: Partition) -> StorageResult<&lsm_tree::AnyTree> {
        // The deployment's own trees need no namespace: a member's engine
        // reaches them before its namespace arrives.
        if part.is_deployment() {
            return self.coordinator.system_tree(part);
        }
        self.coordinator.tree_in(self.set()?, part)
    }

    /// The index tree's installation catalog, when `part` is the index
    /// partition.
    #[inline]
    pub(crate) fn installations_of(
        &self,
        part: Partition,
    ) -> Option<&crate::engine::installation::Installations> {
        self.set().ok()?.installations_of(part)
    }

    /// The namespace's index tree installation catalog.
    #[inline]
    fn installations(&self) -> StorageResult<&crate::engine::installation::Installations> {
        self.set()?
            .installations()
            .ok_or_else(|| StorageError::PartitionNotFound {
                name: Partition::Idx.name().to_string(),
            })
    }

    /// This member's published installation of `generation` in the index
    /// partition: the identity of the local copy reads of the generation
    /// resolve to now. `None` when the generation has no copy here.
    pub fn installation_of(
        &self,
        generation: coordinode_core::index::identity::GenerationId,
    ) -> Option<u64> {
        self.set()
            .ok()?
            .installations()?
            .current()
            .installation(generation.as_raw())
    }

    /// This member's installation of `generation`, bound now when it has
    /// none yet: the copy a guard taken before the generation's first write
    /// protects is the one that write lands in.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the installation ids are
    /// exhausted; a write failure.
    pub fn bind_installation(
        &self,
        generation: coordinode_core::index::identity::GenerationId,
    ) -> StorageResult<u64> {
        self.installations()?.bind(
            self.tree(Partition::Idx)?,
            generation.as_raw(),
            self.next_seqno(),
        )
    }

    /// The installation being prepared to replace this member's copy of
    /// `generation` ([`Self::stage_generation`]), if one is.
    pub fn staging_installation_of(
        &self,
        generation: coordinode_core::index::identity::GenerationId,
    ) -> Option<u64> {
        self.set()
            .ok()?
            .installations()?
            .current()
            .staging(generation.as_raw())
    }

    /// Begin replacing this member's copy of `generation` in the index
    /// partition: a new installation is registered beside the published one,
    /// and from this call on every write of the generation reaches both. A
    /// history taken after this call ([`Self::export_generation_history`],
    /// here or on a peer) fills it.
    ///
    /// The replacement is filled with [`Self::import_generation_history`],
    /// completed with [`Self::finish_generation_import`] and swapped in with
    /// [`Self::publish_generation`], or dropped with
    /// [`Self::abandon_generation`]. A crash before it is completed drops it
    /// at the next open.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no copy
    /// here or already has a replacement in preparation; a write failure.
    pub fn stage_generation(
        &self,
        generation: coordinode_core::index::identity::GenerationId,
    ) -> StorageResult<()> {
        self.admit_replacement_write()?;
        self.installations()?.stage(
            self.tree(Partition::Idx)?,
            self.coverage_domain(),
            generation.as_raw(),
            self.next_seqno(),
        )?;
        Ok(())
    }

    /// A log position every effect below which the index tree `idx` holds,
    /// read while no batch applies. The embedded journal applies proposals
    /// concurrently, so its applied prefix is the bound. A Raft state machine
    /// applies its entries in order, and an entry with no index effects
    /// leaves no marker in the tree, so one past the highest marker is.
    fn applied_position(&self, idx: &lsm_tree::AnyTree) -> StorageResult<u64> {
        match &self.coverage {
            Some(coverage) if self.coverage_domain() == Domain::Journal => {
                Ok(coverage.applied_prefix())
            }
            _ => Ok(TreeCoverage::read(idx, self.coverage_domain())?.next_uncovered()),
        }
    }

    /// The log whose positions this engine's applies are recorded in: the
    /// Raft log when a state machine runs over the engine, the embedded
    /// journal otherwise.
    fn coverage_domain(&self) -> Domain {
        if self.raft_fence().is_some() {
            Domain::Raft
        } else {
            Domain::Journal
        }
    }

    /// Write `entries`, versions of `generation`'s history in logical keys,
    /// into its replacement at their own seqnos. A version the replacement
    /// already holds is written again unchanged. Returns how many were
    /// written.
    ///
    /// The writes are admitted like any other, every [`IMPORT_ADMISSION_STRIDE`]
    /// entries: refused below the free-space reserve or at the endpoint's
    /// hard limit, which the reserve exists to keep for reclaiming space,
    /// not for building copies. A refused import has written a prefix of
    /// `entries`; running it again, or the whole history, is safe, since a
    /// version written twice is unchanged.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no
    /// replacement in preparation or an entry lies outside it;
    /// [`StorageError::OutOfSpace`] or [`StorageError::CapacityExhausted`]
    /// when the write is not admitted.
    pub fn import_generation_history(
        &self,
        generation: coordinode_core::index::identity::GenerationId,
        entries: &[crate::engine::installation::HistoryEntry],
    ) -> StorageResult<usize> {
        let written = self.installations()?.import(
            self.tree(Partition::Idx)?,
            generation.as_raw(),
            entries,
            IMPORT_ADMISSION_STRIDE,
            || self.admit_replacement_write(),
        );
        self.coordinator.flush_trigger().wrote_unmeasured();
        written
    }

    /// Admit a write that prepares a replacement copy of an index generation,
    /// as a client write to the index partition is admitted.
    fn admit_replacement_write(&self) -> StorageResult<()> {
        self.space.admit_free_space()?;
        // A stride of entries must fit; once written they are owed by the
        // memtable, so the admission need not outlive the check.
        self.admit_partition_write(
            [Partition::Idx],
            crate::engine::footprint::footprint_of(0, IMPORT_ADMISSION_STRIDE),
        )
        .map(drop)
    }

    /// Record that `generation`'s replacement holds its complete imported
    /// history. The record is written to the tree after the imported
    /// versions, and a flush persists the tree's writes as a prefix, so it
    /// reaches disk no earlier than they do; one lost to a crash leaves the
    /// replacement to be dropped at the next open. `covers_through` and
    /// `history_from` are the imported history's bounds
    /// ([`GenerationHistory`](crate::engine::installation::GenerationHistory)):
    /// a history taken after [`Self::stage_generation`] returned qualifies,
    /// one taken before does not, and once published the copy refuses reads
    /// at snapshots below `history_from`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::import_generation_history`];
    /// [`StorageError::PositionBehind`] for a history that ends before the
    /// replacement's registration.
    pub fn finish_generation_import(
        &self,
        generation: coordinode_core::index::identity::GenerationId,
        covers_through: u64,
        history_from: lsm_tree::SeqNo,
    ) -> StorageResult<()> {
        self.installations()?.mark_imported(
            self.tree(Partition::Idx)?,
            generation.as_raw(),
            covers_through,
            history_from,
            self.next_seqno(),
        )
    }

    /// Swap `generation`'s completed replacement in for its copy: from this
    /// call on every read of the generation reads the replacement, and the
    /// old copy's entries are deleted. A reader already holding the old copy
    /// at an earlier snapshot still reads it.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no
    /// completed replacement; a write or flush failure.
    pub fn publish_generation(
        &self,
        generation: coordinode_core::index::identity::GenerationId,
    ) -> StorageResult<()> {
        let tree = self.tree(Partition::Idx)?;
        self.installations()?.publish(
            tree,
            generation.as_raw(),
            self.next_seqno(),
            self.next_seqno(),
        )?;
        self.flush_memtables(tree)?;
        // Cached values were read from the old copy.
        self.write_taps.replaced(self.tree_id(Partition::Idx));
        if let Some(cache) = &self.tiered_cache {
            cache.clear_partition(self.tree_id(Partition::Idx));
        }
        Ok(())
    }

    /// Drop `generation`'s replacement; the generation keeps its copy.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no
    /// replacement in preparation; a write failure.
    pub fn abandon_generation(
        &self,
        generation: coordinode_core::index::identity::GenerationId,
    ) -> StorageResult<()> {
        self.installations()?.abandon(
            self.tree(Partition::Idx)?,
            generation.as_raw(),
            self.next_seqno(),
            self.next_seqno(),
        )
    }

    /// The history this member's copy of `generation` holds, in logical
    /// keys and seqno order: the portable form another member, or a
    /// replacement here, is filled from.
    ///
    /// # Errors
    ///
    /// [`StorageError::InstallationCatalog`] when the generation has no copy
    /// here or holds a version a history entry cannot carry; a read failure.
    pub fn export_generation_history(
        &self,
        generation: coordinode_core::index::identity::GenerationId,
    ) -> StorageResult<crate::engine::installation::GenerationHistory> {
        self.installations()?.export(
            self.tree(Partition::Idx)?,
            |idx| self.applied_position(idx),
            generation.as_raw(),
            self.oldest_readable_seqno(),
        )
    }

    /// Where the logical `key` of `part` is stored, for a read: as written,
    /// or under its generation's installation (written into `buf`). `None`
    /// when its generation has no installation here: nothing is stored
    /// under it.
    #[inline]
    fn stored_key<'a>(
        &self,
        part: Partition,
        key: &'a [u8],
        buf: &'a mut Vec<u8>,
    ) -> Option<&'a [u8]> {
        use crate::engine::installation::Located;
        if part != Partition::Idx {
            return Some(key);
        }
        match self.set().ok()?.installations()?.current().locate(key, buf) {
            Located::Raw(key) | Located::Bound(key) => Some(key),
            Located::Unbound => None,
        }
    }

    /// Scan the keys of `part` starting with the logical `prefix` at
    /// `seqno`, giving keys back as addressed.
    fn scan_prefix_at(
        &self,
        part: Partition,
        prefix: &[u8],
        seqno: lsm_tree::SeqNo,
    ) -> StorageResult<StorageIter> {
        let tree = self.tree(part)?;
        Ok(match part {
            Partition::Idx => {
                let bindings = self.installations()?.current();
                bindings.check_prefix_history(prefix, seqno)?;
                crate::engine::installation::scan_prefix(part, tree, &bindings, prefix, seqno)
            }
            _ => coverage::user_prefix(part, tree, prefix, seqno),
        })
    }

    /// Scan the logical range `(lo, hi)` of `part` at `seqno`, never into
    /// the reserved namespace, giving keys back as addressed.
    fn scan_range_at(
        &self,
        part: Partition,
        lo: std::ops::Bound<&[u8]>,
        hi: std::ops::Bound<&[u8]>,
        seqno: lsm_tree::SeqNo,
    ) -> StorageResult<StorageIter> {
        use std::ops::Bound;
        let tree = self.tree(part)?;
        let lo = match lo {
            Bound::Included(k) => Bound::Included(coverage::clamp_user_start(k)),
            Bound::Excluded(k) if k < coverage::USER_KEYSPACE_START => {
                Bound::Included(coverage::USER_KEYSPACE_START)
            }
            Bound::Excluded(k) => Bound::Excluded(k),
            Bound::Unbounded => Bound::Included(coverage::USER_KEYSPACE_START),
        };
        Ok(match part {
            Partition::Idx => {
                let bindings = self.installations()?.current();
                bindings.check_history(lo, hi, seqno)?;
                crate::engine::installation::scan(part, tree, &bindings, lo, hi, seqno)
            }
            _ => {
                let owned = |b: Bound<&[u8]>| match b {
                    Bound::Included(k) => Bound::Included(k.to_vec()),
                    Bound::Excluded(k) => Bound::Excluded(k.to_vec()),
                    Bound::Unbounded => Bound::Unbounded,
                };
                Box::new(
                    tree.range((owned(lo), owned(hi)), seqno, None)
                        .map(move |inner| crate::engine::StorageGuard::raw(part, inner)),
                )
            }
        })
    }

    /// Point lookups of the logical `keys` of `part` at `seqno` through one
    /// tree descent, one result per key, each value as stored: its revision
    /// stamp included. Index generation keys are translated into one shared
    /// buffer; a key of a generation with no installation here finds
    /// nothing.
    fn stored_multi_get_stamped(
        &self,
        part: Partition,
        keys: &[&[u8]],
        seqno: lsm_tree::SeqNo,
    ) -> StorageResult<Vec<Option<lsm_tree::UserValue>>> {
        use crate::engine::installation::Located;
        let tree = self.tree(part)?;
        if part != Partition::Idx {
            return Ok(tree.multi_get(keys.iter().copied(), seqno)?);
        }
        let bindings = self.installations()?.current();
        let mut stored = Vec::with_capacity(keys.iter().map(|k| k.len()).sum());
        // Per key: where its stored form sits in `stored`, or `None`.
        let mut spans: Vec<Option<(usize, usize)>> = Vec::with_capacity(keys.len());
        let mut buf = Vec::new();
        for key in keys {
            let at = stored.len();
            match bindings.locate(key, &mut buf) {
                Located::Raw(key) | Located::Bound(key) => {
                    stored.extend_from_slice(key);
                    spans.push(Some((at, stored.len())));
                }
                Located::Unbound => spans.push(None),
            }
        }
        let present: Vec<&[u8]> = spans
            .iter()
            .flatten()
            .map(|&(from, to)| &stored[from..to])
            .collect();
        let mut found = tree.multi_get(present, seqno)?.into_iter();
        Ok(spans
            .iter()
            .map(|span| span.and_then(|_| found.next().flatten()))
            .collect())
    }

    /// Addressing for one write to `part`: binds a new generation in the
    /// index tree on its first write; as written in every other tree.
    fn addressing(
        &self,
        part: Partition,
    ) -> StorageResult<crate::engine::installation::Addressing<'_>> {
        Ok(crate::engine::installation::Addressing::new(
            self.installations_of(part),
            self.tree(part)?,
            self.coordinator.seqno_generator().get(),
        ))
    }

    /// The Layer-3 coordinator reading and writing this namespace.
    ///
    /// Its key-addressed methods read keys as stored; an index generation
    /// key is read through the engine's own methods, which translate it.
    #[must_use]
    pub fn scoped_coordinator(&self) -> crate::engine::coordinator::ScopedCoordinator<'_> {
        // A member's engine bound to no namespace yet reaches the
        // deployment's trees only.
        match self.set() {
            Ok(set) => self.coordinator.scoped(set),
            Err(_) => self.coordinator.scoped(self.coordinator.sets().system()),
        }
    }
}

impl Deployment {
    /// Borrow the Layer-3 coordinator. Replicated-writer and the
    /// seqno-consumer registry plug in at this seam — see
    /// [`LocalMultiModalCoordinator`] doc for the wire-in contract.
    ///
    /// Its key-addressed methods read keys as stored; an index generation
    /// key is read through the engine's own methods, which translate it.
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
    /// `target` (which must not yet exist). Each partition tree is captured
    /// as it stands, its memtables included, without a flush: its files are
    /// hard-linked into `target/<partition>/` (copied across volumes) and
    /// what its memtables hold is written there. The oplog directory is
    /// copied alongside. The result is a complete, independently-openable
    /// database in one directory, whatever the endpoint layout it came from:
    /// [`Self::open_checkpoint`] opens it. Nothing grows under the store's
    /// own endpoints for a `target` outside them.
    ///
    /// Blocks: call it off the async runtime. On a store a Raft state
    /// machine runs over, the trees are captured with the applies paused;
    /// otherwise at one cut of the commit clock, each commit below it waited
    /// for and each one past it left out, so every commit is in the trees
    /// whole or not at all (a copied journal replays the ones past it).
    ///
    /// The field interner and schema metadata live in the Schema partition
    /// tree, so they are captured by that tree's checkpoint — no separate
    /// handling needed.
    ///
    /// # Errors
    ///
    /// - `target` already exists, or its parent cannot be created
    /// - a tree cannot be captured or rebuilt in `target`
    /// - [`StorageError::CutNotReady`]: a commit below the cut did not land
    ///   in time; the same checkpoint can be taken again
    /// - the oplog directory cannot be copied
    pub fn create_checkpoint(&self, target: &Path) -> StorageResult<CheckpointSummary> {
        claim_checkpoint_dir(target)?;
        let taken = self.take_checkpoint(target);
        if taken.is_err() {
            self.discard_checkpoint(target);
        }
        taken
    }

    /// Remove the unsealed checkpoint in `target`: it is no checkpoint, and
    /// its links would keep tables a compaction replaced and its journal copy
    /// its bytes. What the removal leaves on disk stays charged.
    fn discard_checkpoint(&self, target: &Path) {
        match std::fs::remove_dir_all(target) {
            Ok(()) => self.footprints.removed_tree(target),
            Err(e) => {
                tracing::warn!(?target, error = %e, "removing a failed checkpoint");
                self.footprints.removed_partly(target);
            }
        }
    }

    /// The steps of [`Self::create_checkpoint`] after `target` is claimed.
    fn take_checkpoint(&self, target: &Path) -> StorageResult<CheckpointSummary> {
        let mut summary = CheckpointSummary::default();
        // No reclamation removes journal segments while the checkpoint is
        // taken: the floor its rebuild replays from is kept once it exists.
        let _no_reclaim = self.footprint_reclaim.lock();

        // Fold the applied prefix into every tree first, so the checkpoint's
        // bases, and the journal a rebuild from it needs, are as recent as
        // they can be.
        let replay_floor = self.coverage.as_ref().map(|coverage| {
            let mut folded = coverage.lock_folded();
            self.fold_coverage(coverage, &mut folded, coverage::FoldScope::All)
        });

        // The journal is copied before the trees. A segment the purge drops
        // meanwhile held only entries already persisted in the live trees,
        // which the trees' checkpoint below then carries; copied after the
        // trees instead, a purge could drop entries that reached the trees
        // only after their checkpoint, and the copy would hold them nowhere.
        let src_oplog = self
            .select_oplog_endpoint(0)
            .ok()
            .map(|endpoint| endpoint.path.join("oplog"))
            .filter(|dir| dir.exists());
        if let Some(src) = &src_oplog {
            summary.oplog_bytes = capture_journal(src, &target.join("oplog"), &self.footprints)?;
        }

        let end = self.checkpoint_trees(target, src_oplog.as_deref(), &mut summary)?;
        // And once more after the trees, up to where the journal ended when
        // they were pinned: entries written before reached the trees of the
        // partitions they touch, but not the first copy of the journal.
        // Without them the copy's next entry would follow a gap the other
        // trees do not cover, and the copy would not open again. Entries
        // written after are past the trees and stay out: replayed, they
        // would bring part of an operation that started after the pins.
        if let (Some(src), Some(end)) = (&src_oplog, &end) {
            summary.oplog_bytes +=
                capture_journal_tail(src, &target.join("oplog"), end, &self.footprints)?;
        }
        self.seal_checkpoint(target, replay_floor.unwrap_or(0))?;
        // The newest checkpoint is the one a rebuild starts from: its trees
        // hold every entry below the folded base, and the journal keeps the
        // rest for it.
        if let Some(floor) = replay_floor {
            self.set_journal_keep_from(floor);
        }
        Ok(summary)
    }

    /// The error a checkpoint step `e` ends the checkpoint with, naming the
    /// step. A refusal for room keeps its kind, so a caller can retry it
    /// once room comes back ([`StorageError::is_out_of_room`]): the typed
    /// refusal as it came, a refused file growth as the same storage-full
    /// error with the step added. Where it was refused is only what the
    /// error itself says; no endpoint is inferred from current counters.
    fn checkpoint_failure(e: StorageError, step: impl FnOnce() -> String) -> StorageError {
        match e {
            StorageError::CapacityExhausted { .. } => e,
            StorageError::Engine(lsm_tree::Error::Io(io))
                if io.kind() == lsm_tree::io::ErrorKind::StorageFull =>
            {
                StorageError::Engine(lsm_tree::Error::Io(lsm_tree::io::Error::new(
                    lsm_tree::io::ErrorKind::StorageFull,
                    format!("{}: {io}", step()),
                )))
            }
            other => StorageError::Io(format!("{}: {other}", step())),
        }
    }

    /// Build every partition tree and columnar table in `target` from pinned
    /// versions ([`Self::checkpoint_pinned`]): nothing is flushed, so taking
    /// the checkpoint needs no room under the store's own endpoints.
    fn checkpoint_trees(
        &self,
        target: &Path,
        journal: Option<&Path>,
        summary: &mut CheckpointSummary,
    ) -> StorageResult<Option<JournalEnd>> {
        self.checkpoint_pinned(target, journal, summary)
            .map_err(|e| match e {
                // Refusals the caller acts on keep their kind.
                StorageError::TreeSetBusy { .. }
                | StorageError::CutNotReady { .. }
                | StorageError::CatalogMismatch { .. } => e,
                other => Self::checkpoint_failure(other, || "checkpoint the trees".into()),
            })
    }

    /// Where the journal under `src` ends now. The store's own journal tells
    /// it under its append lock, held only for that: its sealed segments
    /// whole and the active one up to its last whole entry, that entry's
    /// bytes handed to the operating system; the files are measured after
    /// the lock is given back. A log this store does not append to itself
    /// (a group's) is measured as its files stand. A checkpoint reads the
    /// end with its trees pinned and copies the journal up to it, never past.
    ///
    /// # Errors
    ///
    /// The journal's lock is poisoned or its buffer cannot be flushed, or
    /// its files cannot be measured.
    pub(crate) fn journal_end(&self, src: &Path) -> StorageResult<JournalEnd> {
        let Some(oplog) = &self.oplog else {
            return journal_lengths(src);
        };
        let crate::oplog::manager::LogEnd { sealed, active } = oplog
            .lock()
            .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?
            .end_to_os()?;
        let rel = |path: &Path| {
            path.strip_prefix(src).map(Path::to_path_buf).map_err(|_| {
                StorageError::Io(format!("journal segment {path:?} lies outside {src:?}"))
            })
        };
        let mut end = JournalEnd::default();
        for path in sealed {
            let len = std::fs::metadata(&path)
                .map_err(|e| StorageError::Io(format!("measure {path:?}: {e}")))?
                .len();
            end.0.insert(rel(&path)?, len);
        }
        if let Some((path, len)) = active {
            end.0.insert(rel(&path)?, len);
        }
        Ok(end)
    }

    /// The file system every tree writes through.
    pub(crate) fn tree_fs(&self) -> &Arc<dyn lsm_tree::fs::Fs> {
        &self.fs
    }

    /// Carry out `read` through the file system the trees read through:
    /// `io_uring` where the build and the kernel have it, so the segments are
    /// read in one submission. The calling thread waits for it; see
    /// [`PlannedRead::read`](crate::oplog::PlannedRead::read).
    ///
    /// # Errors
    ///
    /// As [`PlannedRead::read`](crate::oplog::PlannedRead::read).
    pub fn read_journal(
        &self,
        read: crate::oplog::PlannedRead,
    ) -> StorageResult<Option<Vec<crate::oplog::OplogEntry>>> {
        read.read(self.fs.as_ref())
    }

    /// How the trees sync their files.
    pub(crate) fn tree_sync(&self) -> lsm_tree::fs::SyncMode {
        self.tree_sync
    }

    /// Make the checkpoint in `target` durable, with the journal index a
    /// rebuild from it replays from, and name it a store.
    fn seal_checkpoint(&self, target: &Path, replay_floor: u64) -> StorageResult<()> {
        use std::io::Write as _;
        // Read back without opening the checkpoint, which would change it.
        let path = target.join(REPLAY_FLOOR_FILE);
        let io = |e: std::io::Error| StorageError::Io(format!("write {path:?}: {e}"));
        // no-std: StoragePort trait, caller-injected
        // The floor counts this store's journal: it names the deployment the
        // store's history belongs to, so another store's is not taken for it.
        let mut payload = replay_floor.to_be_bytes().to_vec();
        payload.extend_from_slice(&deployment_tag(crate::engine::metadata::bound_deployment(
            self,
        )?));
        let record = seal_record(REPLAY_FLOOR_MAGIC, &payload)?;
        crate::engine::promise::charge_record(&self.footprints, &path, record.len() as u64)?;
        let mut file = std::fs::File::create(&path).map_err(io)?;
        file.write_all(&record).map_err(io)?;
        file.sync_all().map_err(io)?;
        // A checkpoint is a base the Raft log is kept for: durable, both the
        // flushed trees under the endpoints and the copy, before its marker
        // names it a store.
        self.sync_endpoint_devices()?;
        if self.tree_sync == lsm_tree::fs::SyncMode::Barrier {
            self.fs.sync_device(target).map_err(|e| {
                StorageError::Io(format!("flush the device caches under {target:?}: {e}"))
            })?;
        }
        // The copy is a directory of this engine's format, opened as one.
        crate::format::write_marker(
            target,
            coordinode_core::version::engine_format_version(),
            &self.footprints,
        )
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
    pub fn open_checkpoint(checkpoint_dir: &Path) -> StorageResult<Arc<Self>> {
        use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
        let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            checkpoint_dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )])
        .relocated();
        // A repair base is read, never maintained: no compaction rewrites
        // it, and it serves every snapshot it physically holds. Its seqnos
        // are the clock that wrote them, so it opens on a clock, as every
        // open of a directory must.
        config.compaction_workers = 0;
        config.retention_window_secs = u64::MAX;
        let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
        Self::open_with_oracle(&config, oracle)
    }
}

impl StorageEngine {
    /// [`Deployment::open_checkpoint`] as a store of one namespace: bound to
    /// the checkpoint's one namespace, and reaching only the deployment's own
    /// trees when it holds none or several. Opening creates nothing.
    ///
    /// # Errors
    ///
    /// As [`Deployment::open_checkpoint`].
    pub fn open_checkpoint(checkpoint_dir: &Path) -> StorageResult<Self> {
        Ok(Deployment::open_checkpoint(checkpoint_dir)?.member_engine())
    }

    /// The checkpoint at `checkpoint_dir` opened read-only and bound to this
    /// engine's namespace: `None` when the checkpoint holds no such namespace.
    /// Another namespace's trees may hold the same keys and table names, so a
    /// rebuild only ever reads its own namespace's copy.
    ///
    /// # Errors
    ///
    /// The checkpoint cannot be opened, or its catalog cannot be read.
    fn checkpoint_view(&self, checkpoint_dir: &Path) -> StorageResult<Option<StorageEngine>> {
        let checkpoint = Deployment::open_checkpoint(checkpoint_dir)?;
        // A member's engine bound to no namespace rebuilds the deployment's
        // trees only, which the checkpoint's own member engine reaches.
        let Ok(bound) = self.bound() else {
            return Ok(Some(checkpoint.member_engine()));
        };
        match checkpoint.engine(bound.namespace) {
            Ok(view) => Ok(Some(view)),
            Err(StorageError::UnknownNamespace { .. }) => Ok(None),
            Err(err) => Err(err),
        }
    }

    /// Rebuild a corrupt partition from a checkpoint plus oplog replay
    /// (WAL-replay-repair, repair path 2). Used when no healthy replica can
    /// serve the partition — the single-node / embedded / RF=1 case.
    ///
    /// Opens `checkpoint_dir` read-only, physically drops the live (corrupt)
    /// partition tables, reinstalls every version, merge operand and deletion
    /// the checkpoint's copy holds, each at its original seqno, then replays
    /// the entries of `oplog_since` that copy lacks, each at its own commit
    /// timestamp, to roll the partition forward to its current state.
    ///
    /// The rebuild is refused, changing nothing, unless it can prove the
    /// partition's state: the engine keeps a journal, and every applied
    /// journal entry is in the checkpoint's copy or in `oplog_since`.
    ///
    /// Same-disk checkpoints only protect against localized corruption;
    /// whole-device loss requires an off-device backup (PITR).
    ///
    /// # Errors
    ///
    /// The checkpoint cannot be opened or read, the journal suffix does not
    /// decode, or the partition cannot be cleared or written.
    pub fn repair_partition_from_checkpoint(
        &self,
        checkpoint_dir: &Path,
        oplog_since: &[OplogEntry],
        partition: Partition,
    ) -> StorageResult<Rebuild> {
        let Some(coverage) = self.coverage.as_ref().filter(|_| self.oplog.is_some()) else {
            return Ok(Rebuild::Refused(RebuildRefusal::NoJournal));
        };

        // 1. Prove the state before touching the live partition, against the
        //    checkpoint's copy of this namespace's tree.
        let Some(ckpt) = self.checkpoint_view(checkpoint_dir)? else {
            return Ok(Rebuild::Refused(RebuildRefusal::MissingFromCheckpoint));
        };
        if ckpt
            .open_repairs()
            .iter()
            .any(|r| r.tree == ckpt.tree_id(partition) && r.is_lossy())
        {
            return Ok(Rebuild::Refused(RebuildRefusal::CheckpointDamaged));
        }
        let source = ckpt.tree(partition)?.clone();
        let held = TreeCoverage::read(&source, Domain::Journal)?;
        // Read once: the coverage written in step 4 is exactly what was proved.
        let (next, above) = coverage.applied_snapshot();
        if let Some(refusal) = rebuild::uncovered_entry(&held, oplog_since, (next, &above)) {
            return Ok(Rebuild::Refused(refusal));
        }
        let source_floor = source.retention_floor();
        // The copy's entries are stored under its own installations.
        let source_bindings = ckpt.installations_of(partition).map(|i| i.current());

        // The clear below removes this node's own records with the rest of
        // the tree, so they are read first and put back after the base. A
        // checkpoint's copy of them is no proof of what they are now (a route
        // or a member record changed since), so a live tree that cannot give
        // them leaves the rebuild refused. The consensus state (vote, log) is
        // not taken back from an older copy either: an older vote would let
        // this node vote twice in one term, so none is reinstated. A rebuild
        // a crash cut short after its clear kept them in its intent, which
        // is read in place of the cleared tree.
        let local = match read_kept_rows(&self.data_dir, self.tree_id(partition))
            .transpose()
            .unwrap_or_else(|| self.node_local_rows(partition))
        {
            Ok(local) => local,
            Err(e) if partition == Partition::Raft => {
                tracing::warn!(
                    partition = partition.name(),
                    error = %e,
                    "this node's consensus records could not be read from the damaged tree"
                );
                Rows::new()
            }
            Err(e) => {
                tracing::error!(
                    partition = partition.name(),
                    error = %e,
                    "this node's own records could not be read from the damaged tree"
                );
                return Ok(Rebuild::Refused(RebuildRefusal::LocalStateLost));
            }
        };

        // 2. Physically clear the live (corrupt) tables, reinstall the base.
        //    `clear_partition` deletes the table files without reading their
        //    blocks; a range tombstone here would leave them to a compaction
        //    that re-reads the corrupt block we are repairing and aborts with a
        //    ChecksumMismatch. The clear is on disk at once while the rebuilt
        //    data is not, so the intent goes first, with the records kept.
        let repair = self.begin_rebuild_keeping(self.tree_id(partition), &local)?;
        self.clear_partition_tree(partition, self.tree(partition)?)?;
        let mut versions = 0;
        for event in rebuild::tree_history(&source)? {
            if self.reinstall_event(partition, source_bindings.as_deref(), event)? {
                versions += 1;
            }
        }
        drop(source);
        drop(ckpt);
        for (key, value) in &local {
            self.put_stored(partition, key, value)?;
        }

        // 3. Replay the journal entries the copy lacks for this partition at
        //    their own commit timestamps, each with its coverage marker,
        //    rolling it forward to the current state. What the copy holds is
        //    its own record, not a journal position: a commit landing while
        //    the checkpoint was taken can be in it or not.
        let tree = self.tree(partition)?;
        let installations = self.installations_of(partition);
        let mut replayed = 0;
        for entry in oplog_since {
            if held.contains(entry.index, 0) {
                continue;
            }
            // A rebuilt index partition derives its DERIVED entries from the
            // journal entry's own data ops, which it does not replay itself.
            let ops = crate::oplog::convert::resolve_entry_ops(&entry.ops)?;
            let ops = ops_of_tree(&ops, self.namespace(), partition);
            if ops.is_empty() {
                continue;
            }
            apply_oplog_ops_at(
                partition,
                tree,
                installations,
                &ops,
                entry.ts,
                entry.index,
                Domain::Journal,
            )?;
            replayed += 1;
        }
        self.coordinator.flush_trigger().wrote_unmeasured();
        self.rebind_coverage(partition, next, &above)?;
        // 4. With the history on disk and the intent still in place, serve
        //    it.
        self.flush_memtables(tree)?;
        self.restore_retention_floor(tree, partition, source_floor)?;
        self.finish_rebuild(repair)?;
        // The rebuilt tree is on disk: whatever an open-time repair of it
        // lost is restored.
        self.acknowledge_open_repair(self.tree_id(partition))?;
        Ok(Rebuild::Rebuilt { versions, replayed })
    }

    /// Lower the retention floor of `tree`, rebuilt as `partition`'s and
    /// raised by the rebuild's clear, to what its reinstalled history serves:
    /// the checkpoint copy's own floor `source_floor`, or just below the GC
    /// watermark the rebuilt tree has been flushed under, whichever is
    /// higher, since a flush may have collected history below that
    /// watermark. A floor already at or below it is left alone.
    ///
    /// Precondition: the reinstalled history is on disk, and the rebuild's
    /// intent is still in place, so a crash before this call leaves the clear's
    /// floor and a rebuild still owed.
    fn restore_retention_floor(
        &self,
        tree: &lsm_tree::AnyTree,
        partition: Partition,
        source_floor: lsm_tree::SeqNo,
    ) -> StorageResult<()> {
        // A snapshot at or above the watermark W reads intact history: the
        // collection below W keeps each key's newest version below it. The
        // clamp is the rule itself: a watermark of 0 collected nothing, so
        // nothing is refused.
        let collected = self
            .coordinator
            .gc_watermark_value_for(partition)
            .saturating_sub(1);
        let floor = source_floor.max(collected);
        if floor < tree.retention_floor() {
            tree.lower_retention_floor(floor)?;
        }
        Ok(())
    }

    /// Write one event of a checkpoint copy's history into the rebuilt
    /// `partition` at its original seqno, translated from the copy's
    /// installations (`source_bindings`, for the index partition) to this
    /// engine's. Returns whether the event belongs to the rebuilt tree: the
    /// reserved keys do not (the coverage record is written once the rebuild
    /// is proved, and the index catalog stays this member's), nor do this
    /// node's own records (taken from the live tree) or entries under a
    /// retired installation.
    fn reinstall_event(
        &self,
        partition: Partition,
        source_bindings: Option<&crate::engine::installation::Bindings>,
        event: lsm_tree::ScanSinceEvent,
    ) -> StorageResult<bool> {
        use lsm_tree::ScanSinceEvent;

        let tree = self.tree(partition)?;
        let mut addressing = crate::engine::installation::Addressing::new(
            self.installations_of(partition),
            tree,
            event.seqno(),
        );
        let written = match event {
            ScanSinceEvent::RangeTombstone {
                start_key,
                end_key,
                seqno,
            } => {
                let start = coverage::clamp_user_start(&start_key);
                // The consensus partition holds only this node's own records.
                if start >= end_key.as_ref() || partition == Partition::Raft {
                    return Ok(false);
                }
                let logical = match source_bindings {
                    Some(bindings) => bindings.logical_ranges_until(start, end_key.to_vec()),
                    None => vec![(start.to_vec(), end_key.to_vec())],
                };
                let mut added = 0;
                for (start, end) in logical {
                    let stored = addressing.range(&start, &end);
                    let staging = addressing.staging_range(&start, &end);
                    for (start, end) in stored.into_iter().chain(staging) {
                        if start < end {
                            added += tree.remove_range(start, end, seqno);
                        }
                    }
                }
                Some((added, tree.active_memtable().size()))
            }
            ScanSinceEvent::Insert { key, value, seqno } => {
                self.reinstall_point(partition, source_bindings, &mut addressing, &key, |key| {
                    tree.insert(key, value.clone(), seqno)
                })?
            }
            ScanSinceEvent::MergeOperand {
                key,
                operand,
                seqno,
            } => {
                self.reinstall_point(partition, source_bindings, &mut addressing, &key, |key| {
                    tree.merge(key, operand.clone(), seqno)
                })?
            }
            ScanSinceEvent::PointTombstone { key, seqno } => {
                self.reinstall_point(partition, source_bindings, &mut addressing, &key, |key| {
                    tree.remove(key, seqno)
                })?
            }
            ScanSinceEvent::WeakTombstone { key, seqno } => {
                self.reinstall_point(partition, source_bindings, &mut addressing, &key, |key| {
                    tree.remove_weak(key, seqno)
                })?
            }
        };
        let Some((added, memtable)) = written else {
            return Ok(false);
        };
        self.coordinator
            .flush_trigger()
            .wrote(partition, added, memtable);
        Ok(true)
    }

    /// The point write of [`Self::reinstall_event`]: `write` applied to the
    /// stored key of the copy's physical `key` and to its replacement's, or
    /// `None` when the key does not belong to the rebuilt tree.
    fn reinstall_point(
        &self,
        partition: Partition,
        source_bindings: Option<&crate::engine::installation::Bindings>,
        addressing: &mut crate::engine::installation::Addressing<'_>,
        key: &[u8],
        write: impl Fn(Vec<u8>) -> (u64, u64),
    ) -> StorageResult<Option<(u64, u64)>> {
        if coverage::is_reserved(key) {
            return Ok(None);
        }
        let logical = match source_bindings {
            Some(bindings) => match bindings.logical_key(key) {
                Some(logical) => logical,
                None => return Ok(None),
            },
            None => key.to_vec(),
        };
        if raft_coverage::is_node_local(partition, &logical) {
            return Ok(None);
        }
        let stored = addressing.point(&logical)?.to_vec();
        let staging = addressing.staging_point(&logical).map(<[u8]>::to_vec);
        let (mut added, mut memtable) = write(stored);
        if let Some(staging) = staging {
            let (more, size) = write(staging);
            added += more;
            memtable = size;
        }
        Ok(Some((added, memtable)))
    }

    /// Rebuild `STORAGE COLUMNAR` table `table_id` from a checkpoint plus the
    /// journal, as [`Self::repair_partition_from_checkpoint`] rebuilds a
    /// partition: after an open-time salvage lost part of its tree, or a
    /// rebuild cut short. The table's tree is replaced by every version,
    /// merge operand and deletion the checkpoint's copy holds, each at its
    /// original seqno, the entries of `oplog_since` that copy lacks are
    /// replayed into it at their own timestamps, and its coverage record is
    /// bound to the journal.
    ///
    /// The rebuild installs only a state it can prove, and refuses otherwise,
    /// changing nothing: the checkpoint must hold the table, every applied
    /// journal entry must be in the checkpoint's copy or in `oplog_since`, and
    /// no entry of `oplog_since` past the checkpoint may write one of
    /// `catalog_keys` (the schema partition keys of the table's catalog
    /// entry), since a DROP and a re-create under the same name would make the
    /// checkpoint's table a different one.
    ///
    /// History the checkpoint's copy no longer holds (versions a compaction
    /// collected before it was taken) stays collected.
    ///
    /// Precondition, as for a partition: no write to the table is in flight.
    ///
    /// # Errors
    ///
    /// The checkpoint cannot be opened or read, the journal suffix does not
    /// decode, or the table's tree cannot be replaced or written.
    #[cfg(feature = "columnar")]
    pub fn repair_table_from_checkpoint(
        &self,
        checkpoint_dir: &Path,
        oplog_since: &[OplogEntry],
        table_id: &str,
        catalog_keys: &[Vec<u8>],
    ) -> StorageResult<Rebuild> {
        let Some(coverage) = self.coverage.as_ref().filter(|_| self.oplog.is_some()) else {
            return Ok(Rebuild::Refused(RebuildRefusal::NoJournal));
        };

        // 1. Prove the state before touching the live table. A catalog change
        //    the checkpoint does not hold may be a DROP and a re-create: the
        //    checkpoint's table and the live one are then different tables
        //    under one name.
        // The checkpoint's copy of this namespace's table: the same name may
        // be a table of another namespace too.
        let Some(ckpt) = self.checkpoint_view(checkpoint_dir)? else {
            return Ok(Rebuild::Refused(RebuildRefusal::MissingFromCheckpoint));
        };
        let Some(source) = ckpt.columnar_table_tree(table_id) else {
            return Ok(Rebuild::Refused(RebuildRefusal::MissingFromCheckpoint));
        };
        if ckpt.lossy_table_repairs().iter().any(|t| t == table_id) {
            return Ok(Rebuild::Refused(RebuildRefusal::CheckpointDamaged));
        }
        let held = TreeCoverage::read(&source, Domain::Journal)?;
        let catalog_held = TreeCoverage::read(ckpt.tree(Partition::Schema)?, Domain::Journal)?;
        // Read once: the coverage written in step 4 is exactly what was proved.
        let (next, above) = coverage.applied_snapshot();
        if let Some(refusal) = rebuild::catalog_change(&catalog_held, oplog_since, catalog_keys)?
            .or_else(|| rebuild::uncovered_entry(&held, oplog_since, (next, &above)))
        {
            return Ok(Rebuild::Refused(refusal));
        }
        let source_floor = source.retention_floor();

        // 2. Replace the live tree with the checkpoint's copy, version by
        //    version. The drop is on disk at once while the rebuilt tree is
        //    not, so the intent goes first. The new tree is cleared before
        //    the history goes in: its floor then refuses every older snapshot
        //    until the whole history is back.
        let target = RebuildTarget::Table(self.namespace(), table_id);
        // Held alone until the rebuilt table is installed (the end of this
        // call): no capture pins it dropped or half filled.
        let _alone = self.rebuild_alone()?;
        // The rebuilt tree stands for the same table: it keeps the stamp the
        // table was created with (none for one the store held at its open), so
        // a transaction that read the table before a later creation under its
        // name is still refused a row in it.
        let registry = self.columnar()?;
        let created_at = registry
            .get_stamped(table_id)
            .map_or(0, |(_, _, created_at)| created_at);
        write_rebuild_intent(&self.data_dir, target)?;
        registry.drop_table(table_id)?;
        let tree = registry.create_or_open_with(table_id, created_at, |_| Ok(()))?;
        tree.clear()?;
        #[cfg(any(test, feature = "test-faults"))]
        if let Some(hook) = self.on_rebuild_cleared.lock().as_mut() {
            hook();
        }
        let mut versions = 0;
        for event in rebuild::tree_history(&source)? {
            let (added, memtable) = rebuild::replay_event(&tree, event);
            self.coordinator
                .flush_trigger()
                .wrote(Partition::Node, added, memtable);
            versions += 1;
        }
        drop(source);
        drop(ckpt);

        // 3. Replay the table's rows the copy lacks, each at its entry's ts
        //    with its marker, rolling the table forward to the journal's end.
        let mut replayed = 0;
        let own = self.namespace().as_raw();
        for entry in oplog_since {
            if held.contains(entry.index, 0) {
                continue;
            }
            // A row belongs to the namespace whose marker precedes it.
            let mut namespace = None;
            for op in crate::oplog::convert::resolve_entry_ops(&entry.ops)?.iter() {
                if let OplogOp::Namespace { id } = op {
                    namespace = Some(*id);
                    continue;
                }
                if let OplogOp::ColumnarInsert {
                    table_id: table,
                    key,
                    value,
                } = op
                    && namespace == Some(own)
                    && table == table_id
                {
                    let (added, memtable) = apply_columnar_row(
                        &tree,
                        key.clone(),
                        value.clone(),
                        entry.ts,
                        Some(entry.index),
                    )?;
                    self.coordinator
                        .flush_trigger()
                        .wrote(Partition::Node, added, memtable);
                    replayed += 1;
                }
            }
        }

        // 4. The tree holds every entry proved in step 1: record it, after
        //    the rebuilt rows, then persist the tree, serve its history and
        //    drop the intent. A table keeps the versions the node partition
        //    keeps, under its watermark.
        let domain = Domain::Journal;
        let at = self.next_seqno();
        tree.insert(domain.base_key(), coverage::encode_base(next, &[]), at);
        for index in above {
            tree.insert(domain.marker_key(index, 0).as_slice(), &[][..], at);
        }
        self.flush_memtables(&tree)?;
        self.restore_retention_floor(&tree, Partition::Node, source_floor)?;
        remove_rebuild_intent(&self.data_dir, target, &self.footprints)?;
        // What an open-time salvage of the table lost is restored.
        self.acknowledge_table_repair(table_id)?;
        Ok(Rebuild::Rebuilt { versions, replayed })
    }

    /// This node's own records in `partition` (see
    /// [`raft_coverage::is_node_local`]), read from the live tree as stored,
    /// revisions included: what [`Self::put_stored`] writes back.
    fn node_local_rows(&self, partition: Partition) -> StorageResult<Rows> {
        let prefixes: &[&[u8]] = match partition {
            Partition::Raft => &[b""],
            _ => raft_coverage::node_local_prefixes(partition),
        };
        let mut rows = Rows::new();
        for prefix in prefixes {
            for guard in self.prefix_scan(partition, prefix)? {
                let (key, value) = guard.into_stored()?;
                rows.push((key, value.to_vec()));
            }
        }
        Ok(rows)
    }

    /// After a rebuild replayed the journal into `partition` up to its end,
    /// record in the partition's tree that it holds the applied set it proved
    /// it holds: every index below `next`, and `above`. The clear removed its
    /// old record.
    fn rebind_coverage(&self, partition: Partition, next: u64, above: &[u64]) -> StorageResult<()> {
        let tree = self.tree(partition)?;
        // Written after the rebuilt data, so a persisted record implies the
        // data it covers is persisted.
        let at = self.next_seqno();
        let domain = Domain::Journal;
        tree.insert(domain.base_key(), coverage::encode_base(next, &[]), at);
        for &index in above {
            tree.insert(domain.marker_key(index, 0).as_slice(), &[][..], at);
        }
        self.coordinator.flush_trigger().wrote_unmeasured();
        Ok(())
    }
}

impl Deployment {
    /// The partitions whose rebuild started and did not finish: a crash
    /// between clearing a tree and persisting its rebuilt data. Such a tree
    /// holds an unknown part of its data and of its coverage record, so only
    /// a new full rebuild restores it.
    ///
    /// # Errors
    ///
    /// The intent directory cannot be read.
    pub fn pending_rebuilds(&self) -> StorageResult<Vec<TreeId>> {
        Ok(read_rebuild_intents(&self.data_dir)?.trees)
    }

    /// Whether the open verified every block of `partition`'s tree, before
    /// it read it, at the version the tree still has: a verification now
    /// would read the same blocks for the same answer. Only the deployment's
    /// records are verified that early, where the open goes on to verify the
    /// trees a checkpoint backs.
    #[must_use]
    pub fn verified_at_open(&self, partition: Partition) -> bool {
        use lsm_tree::AbstractTree as _;
        partition == Partition::Deployment
            && self.deployment_verified.is_some_and(|version| {
                self.system_tree(partition)
                    .is_ok_and(|tree| tree.current_version().id() == version)
            })
    }

    /// The `STORAGE COLUMNAR` tables whose rebuild started and did not
    /// finish, with their namespace: like a tree's, only a new rebuild
    /// restores them.
    ///
    /// # Errors
    ///
    /// The intent directory cannot be read.
    pub fn pending_table_rebuilds(&self) -> StorageResult<Vec<(NamespaceId, String)>> {
        Ok(read_rebuild_intents(&self.data_dir)?.tables)
    }

    /// Record, durably, that tree `id` is about to be rebuilt. Written
    /// before the tree is cleared and removed by [`StorageEngine::finish_rebuild`]
    /// once the rebuilt tree is on disk.
    fn begin_rebuild(&self, id: TreeId) -> StorageResult<PartitionRepair<'_>> {
        let alone = self.rebuild_alone()?;
        // Published before the intent lands and before any tree is touched:
        // an observation from here on sees the repair, whatever step it is at.
        let mut active = self.enter_repair(id)?;
        active._alone = Some(alone);
        write_rebuild_intent(&self.data_dir, RebuildTarget::Tree(id))?;
        Ok(active)
    }

    /// The set of trees alone, for a rebuild from a checkpoint: from its
    /// first destructive step until its result is installed no capture pins
    /// its tree. Its intent keeps a capture away after a failure too.
    ///
    /// # Errors
    ///
    /// [`StorageError::TreeSetBusy`] when operations hold the set for longer
    /// than [`REBUILD_TREE_SET_WAIT`].
    fn rebuild_alone(&self) -> StorageResult<crate::engine::tree_set::TreeSetExclusive<'_>> {
        self.tree_set
            .exclusive(REBUILD_TREE_SET_WAIT)
            .ok_or(StorageError::TreeSetBusy {
                operation: "repair",
                waited_ms: REBUILD_TREE_SET_WAIT_MS,
            })
    }

    /// [`Self::begin_rebuild`] for a rebuild that puts `kept`, the tree's own
    /// records read before its clear, back after it: the intent holds them,
    /// so an open after a crash in between still has them.
    fn begin_rebuild_keeping(&self, id: TreeId, kept: &Rows) -> StorageResult<PartitionRepair<'_>> {
        let alone = self.rebuild_alone()?;
        let mut active = self.enter_repair(id)?;
        active._alone = Some(alone);
        self.record_kept(id, kept)?;
        Ok(active)
    }

    /// Write the rebuild intent of tree `id`, under a repair taken, with the
    /// records of its own `kept` that the rebuild puts back after its clear.
    /// Only the Schema trees have such records outside the consensus
    /// partition, which no copy rebuilds.
    fn record_kept(&self, id: TreeId, kept: &Rows) -> StorageResult<()> {
        let partition = id.part;
        if partition == Partition::Deployment {
            // From the clear on, the ledger in memory is not the tree's: a
            // stored record read for it from here on answers nothing.
            let mut state = self.closure.lock();
            state.installation += 1;
            self.ledger_owed
                .store(true, std::sync::atomic::Ordering::Release);
        }
        let kept = if matches!(partition, Partition::Deployment | Partition::Schema) {
            encode_kept_rows(kept)?
        } else {
            Vec::new()
        };
        write_rebuild_intent_with(
            &self.data_dir,
            RebuildTarget::Tree(id),
            &kept,
            &self.footprints,
        )
    }

    /// Take the repair of tree `id` and mark it under repair until
    /// [`Self::leave_repair`].
    ///
    /// # Errors
    ///
    /// Another repair of the tree is running.
    pub(crate) fn enter_repair(&self, id: TreeId) -> StorageResult<PartitionRepair<'_>> {
        let mut marks = self.repairs.lock();
        if !marks.running.insert(id) {
            return Err(StorageError::InvalidConfig(format!(
                "a repair of the {id} tree is already running"
            )));
        }
        marks.under.insert(id);
        Ok(PartitionRepair {
            engine: self,
            tree: id,
            _alone: None,
        })
    }

    /// Whether tree `id` is under repair: it may be cleared or hold a copy
    /// only partly installed.
    pub(crate) fn is_under_repair(&self, id: TreeId) -> bool {
        self.repairs.lock().under.contains(&id)
    }

    /// A tree under repair, if any is.
    pub(crate) fn any_under_repair(&self) -> Option<TreeId> {
        self.repairs.lock().under.iter().next().copied()
    }

    /// End the repair of tree `id`.
    fn leave_repair(&self, id: TreeId) {
        self.repairs.lock().under.remove(&id);
    }

    /// Refuse a repair taken on another store.
    pub(crate) fn check_repair(&self, repair: &PartitionRepair<'_>) -> StorageResult<()> {
        if core::ptr::eq(repair.engine, self) {
            Ok(())
        } else {
            Err(StorageError::InvalidConfig(format!(
                "the repair of the {} tree was taken on another store",
                repair.tree
            )))
        }
    }
}

impl StorageEngine {
    /// Persist the tree `repair` rebuilt, then drop its intent and end the
    /// repair. Only the repair's holder can finish it.
    fn finish_rebuild(&self, repair: PartitionRepair<'_>) -> StorageResult<()> {
        self.finish_rebuild_with(repair, || Ok(()))
    }

    /// [`Self::finish_rebuild`] running `last` (the repair's own finishing
    /// effects, such as qualifying a transferred index) once the tree is
    /// persisted and while the repair still holds the partition: nothing
    /// else repairs it, and it is not ready, until `last` is done.
    ///
    /// # Errors
    ///
    /// `repair` was taken on another store, which is refused before anything
    /// changes here; a flush, `last` or the intent removal failed, leaving
    /// the partition owed a repair.
    fn finish_rebuild_with(
        &self,
        repair: PartitionRepair<'_>,
        last: impl FnOnce() -> StorageResult<()>,
    ) -> StorageResult<()> {
        self.check_repair(&repair)?;
        let id = repair.tree;
        let partition = id.part;
        self.flush_memtables(&self.tree_of(id)?)?;
        if partition == Partition::Schema {
            // The dictionary was rebuilt: views read before it reread it.
            self.note_field_dictionary_change();
        }
        last()?;
        if partition == Partition::Deployment {
            // The rebuilt tree's ledger, whichever way it was rebuilt, is the
            // one answered from here on.
            self.reload_closure_frontier()?;
        }
        remove_rebuild_intent(&self.data_dir, RebuildTarget::Tree(id), &self.footprints)?;
        if partition == Partition::Deployment {
            self.ledger_owed
                .store(false, std::sync::atomic::Ordering::Release);
        }
        self.leave_repair(id);
        drop(repair);
        Ok(())
    }

    /// What a partition replaced from a copy establishes before its repair
    /// ends ([`Self::finish_rebuild_with`]'s last step): the index tree's
    /// transferred copies qualified against `index_inventory`. (The Schema
    /// tree's closure ledger is installed by the finish of any of its
    /// rebuilds.) A failure leaves the partition owed its repair.
    pub(crate) fn establish_replaced(
        &self,
        partition: Partition,
        index_inventory: Option<&[u8]>,
    ) -> StorageResult<()> {
        if partition == Partition::Idx {
            self.index_transfer_installed(index_inventory)
        } else {
            Ok(())
        }
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
            | OplogOp::ColumnarInsert { .. }
            | OplogOp::Namespace { .. } => Ok(()),
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
    /// The restored rows are written above the clearing range tombstone, so
    /// at fresh seqnos. Refused, changing nothing, as a full rebuild is when
    /// the checkpoint and the journal cannot prove the partition's state.
    ///
    /// # Errors
    ///
    /// The checkpoint cannot be opened or read, the journal suffix does not
    /// decode, or the partition cannot be written.
    pub fn repair_partition_ranges_from_checkpoint(
        &self,
        checkpoint_dir: &Path,
        oplog_since: &[OplogEntry],
        partition: Partition,
        ranges: &[KeyRange],
    ) -> StorageResult<Rebuild> {
        use std::ops::Bound;

        let Some(coverage) = self.coverage.as_ref().filter(|_| self.oplog.is_some()) else {
            return Ok(Rebuild::Refused(RebuildRefusal::NoJournal));
        };
        // The lost ranges are ranges of the tree as stored; everything below
        // works in the keys the application addresses, `[start, end)`. In the
        // index tree a stored range maps to one logical range per installation
        // it crosses.
        let ranges: Vec<(Vec<u8>, Vec<u8>)> = match partition {
            Partition::Idx => {
                let bindings = self.installations()?.current();
                ranges
                    .iter()
                    .flat_map(|(min, max)| bindings.logical_ranges(min, max))
                    .collect()
            }
            _ => ranges
                .iter()
                .map(|(min, max)| {
                    let mut end = max.to_vec();
                    end.push(0);
                    (min.to_vec(), end)
                })
                .collect(),
        };
        let in_ranges = |key: &[u8]| {
            ranges
                .iter()
                .any(|(start, end)| key >= &start[..] && key < &end[..])
        };

        // 1. Open the checkpoint read-only and export the base rows inside
        //    each lost range, with the record of which journal entries the
        //    base holds. The checkpoint engine is dropped before the live
        //    engine is mutated.
        let (base, held): (Rows, TreeCoverage) = {
            let Some(ckpt) = self.checkpoint_view(checkpoint_dir)? else {
                return Ok(Rebuild::Refused(RebuildRefusal::MissingFromCheckpoint));
            };
            if ckpt
                .open_repairs()
                .iter()
                .any(|r| r.tree == ckpt.tree_id(partition) && r.is_lossy())
            {
                return Ok(Rebuild::Refused(RebuildRefusal::CheckpointDamaged));
            }
            let mut rows = Vec::new();
            for (start, end) in &ranges {
                for guard in ckpt.scan_range_at(
                    partition,
                    Bound::Included(start),
                    Bound::Excluded(end),
                    ckpt.snapshot(),
                )? {
                    let (k, v) = guard.into_owned()?;
                    rows.push((k, v.to_vec()));
                }
            }
            (
                rows,
                TreeCoverage::read(ckpt.tree(partition)?, Domain::Journal)?,
            )
        };
        let (next, above) = coverage.applied_snapshot();
        if let Some(refusal) = rebuild::uncovered_entry(&held, oplog_since, (next, &above)) {
            return Ok(Rebuild::Refused(refusal));
        }

        // 2. Clear the lost ranges on the live tree, reinstall the base. A
        //    crash part way leaves the ranges' state unknown while the
        //    coverage record still claims the entries, so the intent turns
        //    the next open's repair into a full rebuild.
        let repair = self.begin_rebuild(self.tree_id(partition))?;
        for (start, end) in &ranges {
            self.remove_range(partition, start, end)?;
        }
        let base_len = base.len();
        for (key, value) in &base {
            self.put(partition, key, value)?;
        }

        // 3. Replay the ops the base lacks that intersect the lost ranges, in
        //    journal order, DERIVED work derived from the whole entry first.
        let mut replayed = 0;
        for entry in oplog_since {
            if held.contains(entry.index, 0) {
                continue;
            }
            replayed += 1;
            let ops = crate::oplog::convert::resolve_entry_ops(&entry.ops)?;
            for op in ops_of_tree(&ops, self.namespace(), partition) {
                match op {
                    OplogOp::Insert { key, .. }
                    | OplogOp::Delete { key, .. }
                    | OplogOp::Merge { key, .. } => {
                        if in_ranges(key) {
                            self.apply_repair_op(partition, op)?;
                        }
                    }
                    OplogOp::RemoveRange { start, end, .. } => {
                        for (lost_start, lost_end) in &ranges {
                            let s = start.max(lost_start);
                            let e = end.min(lost_end);
                            if s < e {
                                self.remove_range(partition, s, e)?;
                            }
                        }
                    }
                    OplogOp::Noop
                    | OplogOp::RaftEntry { .. }
                    | OplogOp::RaftTruncation { .. }
                    | OplogOp::ColumnarInsert { .. }
                    | OplogOp::Namespace { .. } => {}
                    OplogOp::Derive { .. } | OplogOp::Unit { .. } => {
                        return Err(unresolved_unit());
                    }
                }
            }
        }
        self.finish_rebuild(repair)?;
        // The lost ranges are restored and on disk.
        self.acknowledge_open_repair(self.tree_id(partition))?;
        Ok(Rebuild::Rebuilt {
            versions: base_len,
            replayed,
        })
    }
}

impl Deployment {
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
    /// entry below it is in every tree of the checkpoint, folded into each
    /// tree's coverage base before the trees were captured. The repair
    /// orchestrator feeds this to [`oplog_read_since`], and the journal purge
    /// keeps everything at or above it. `0` for a store without a journal.
    /// Read from the record the checkpoint was sealed with: the checkpoint is
    /// not opened, so it stays as it was.
    ///
    /// # Errors
    ///
    /// The checkpoint holds no such record, or it cannot be read.
    ///
    /// [`oplog_read_since`]: Self::oplog_read_since
    pub fn checkpoint_replay_floor(&self, checkpoint_dir: &Path) -> StorageResult<u64> {
        let path = checkpoint_dir.join(REPLAY_FLOOR_FILE);
        // no-std: StoragePort trait, caller-injected
        let bytes =
            std::fs::read(&path).map_err(|e| StorageError::Io(format!("read {path:?}: {e}")))?;
        let not_ours =
            |why: &str| StorageError::Io(format!("{path:?} is not a replay floor: {why}"));
        let payload = open_record(REPLAY_FLOOR_MAGIC, &bytes).map_err(not_ours)?;
        let (floor, deployment) = payload
            .split_first_chunk::<8>()
            .ok_or_else(|| not_ours("too short"))?;
        if deployment != deployment_tag(crate::engine::metadata::bound_deployment(self)?) {
            return Err(not_ours("sealed by a store of another deployment"));
        }
        Ok(u64::from_be_bytes(*floor))
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
        match &self.oplog {
            Some(oplog) => {
                let durable_below = self.persist_coverage_base()?;
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
                // Read under the journal lock a hold is taken under.
                let keep = keep_from_index.min(self.consumer_journal_needs());
                guard.purge_expired(now_secs, keep, &is_durable)
            }
            _ => Ok(0),
        }
    }

    /// The lowest journal index a consumer needs kept: its registry's
    /// published floor, or an index held while a registration publishes it.
    /// Read with the journal locked, the lock [`Self::hold_journal_from`]
    /// takes its hold under.
    fn consumer_journal_needs(&self) -> u64 {
        let held = self
            .journal_holds
            .lock()
            .keys()
            .next()
            .copied()
            .unwrap_or(u64::MAX);
        held.min(
            self.consumer_journal_floor
                .load(std::sync::atomic::Ordering::Acquire),
        )
    }

    /// Keep the journal from `index` on for the consumers registered to read
    /// it; `u64::MAX` when none needs any. Expiry and reclamation under a
    /// footprint limit never remove it: a store at its limit then refuses new
    /// writes rather than drop history a consumer was promised.
    pub fn set_consumer_journal_floor(&self, index: u64) {
        self.consumer_journal_floor
            .store(index, std::sync::atomic::Ordering::Release);
    }

    /// Hold the journal from `index` on until the hold drops, if the journal
    /// still holds `index`; `Ok(None)` when it no longer does. The check and
    /// the hold are made under the lock every purge takes, so a purge either
    /// ran before (and the index is gone) or keeps it.
    ///
    /// # Errors
    ///
    /// The engine keeps no journal of its own, or its lock is poisoned.
    pub fn hold_journal_from(
        &self,
        index: u64,
    ) -> StorageResult<Option<crate::engine::oplog_journal::JournalHold>> {
        let oplog = self
            .oplog
            .as_ref()
            .ok_or_else(|| StorageError::InvalidConfig("the store keeps no journal".into()))?;
        let journal = oplog
            .lock()
            .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
        if index < journal.first_index() {
            return Ok(None);
        }
        Ok(Some(crate::engine::oplog_journal::JournalHold::take(
            &self.journal_holds,
            index,
        )))
    }

    /// What a reader of this store's own journal needs to know of it: the
    /// directory of its segments and where they start, kept current. `None`
    /// when the store keeps no journal.
    ///
    /// # Errors
    ///
    /// The journal lock is poisoned.
    pub fn journal_location(
        &self,
    ) -> StorageResult<Option<(std::path::PathBuf, crate::oplog::RetainedFloor)>> {
        let Some(oplog) = &self.oplog else {
            return Ok(None);
        };
        let journal = oplog
            .lock()
            .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
        Ok(Some((
            journal.dir().to_path_buf(),
            journal.retained_floor(),
        )))
    }

    /// The index the journal's next entry receives: one past the newest a
    /// reader of it can be given. `None` when the store keeps no journal.
    ///
    /// # Errors
    ///
    /// The journal lock is poisoned.
    pub fn journal_next_index(&self) -> StorageResult<Option<u64>> {
        let Some(oplog) = &self.oplog else {
            return Ok(None);
        };
        let journal = oplog
            .lock()
            .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
        Ok(Some(journal.next_index()))
    }

    /// Take a fixed cut of this store's history: a view pinned below a fresh
    /// bound `B`, the first journal position `L` its continuation replays,
    /// and the join end `H` (see [`crate::engine::fixed_cut`]). Cold work:
    /// nothing is added to writes, and the journal is not searched.
    ///
    /// `L` is read first; `B` is then drawn from the commit clock, above
    /// every commit journalled before `L`. The view is qualified once every
    /// commit below `B` has finished and is durable. `H` is read after that,
    /// and every entry before it is then waited for and made durable: a late
    /// entry below `B` falls inside the join, not after it.
    ///
    /// # Errors
    ///
    /// [`StorageError::InvalidConfig`] for a store without a journal of its
    /// own; [`StorageError::CutNotReady`] when a commit is still in flight or
    /// not durable after `timeout`; a journal sync failure.
    fn fixed_cut_of(
        &self,
        scope: crate::engine::fixed_cut::CutScope,
        timeout: std::time::Duration,
    ) -> StorageResult<crate::engine::fixed_cut::FixedCut> {
        let Some(oplog) = &self.oplog else {
            if self.raft_fence().is_some() || self.group_history {
                return self.fixed_cut_of_group(scope, timeout);
            }
            return Err(StorageError::InvalidConfig(
                "a fixed cut follows the store's own journal or its group's log, and this \
                 store has neither"
                    .into(),
            ));
        };
        let journal = || {
            oplog
                .lock()
                .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))
        };
        let not_ready = |reason: String| StorageError::CutNotReady { reason };
        let first = journal()?.next_index();
        // Drawn after `first` from the clock every commit takes its timestamp
        // from, and pinned in the same step: above every journalled commit
        // before `first`.
        let (bound, pin) = self.pin_new_snapshot(|| self.next_seqno());
        let pin = pin.ok_or_else(|| not_ready("the view could not be pinned".into()))?;
        #[cfg(any(test, feature = "test-faults"))]
        if let Some(hook) = self.on_cut_bound.lock().as_mut() {
            hook();
        }
        let deadline = std::time::Instant::now() + timeout;
        // A deadline already passed leaves no time: none, not an error.
        let left = || deadline.saturating_duration_since(std::time::Instant::now());
        // A drawn timestamp is above the clock's zero.
        let below = |ts: u64| {
            ts.checked_sub(1)
                .ok_or_else(|| StorageError::Io("the commit clock drew 0".into()))
        };
        // Kept at `bound`, never lowered: the commits below it finish.
        self.pending_commits
            .await_complete_at(below(bound)?, left())
            .map_err(|ts| not_ready(format!("a commit at {ts} below the cut is in flight")))?;
        crate::engine::oplog_journal::sync_journal(oplog)?;
        if self.pending_commits.durable_floor(bound) < bound {
            return Err(not_ready("a commit below the cut is not durable".into()));
        }
        let join_end = journal()?.next_index();
        // Every entry before `join_end` drew its timestamp before this one.
        let after = self.next_seqno();
        self.pending_commits
            .await_complete_at(below(after)?, left())
            .map_err(|ts| not_ready(format!("a commit at {ts} of the join is in flight")))?;
        crate::engine::oplog_journal::sync_journal(oplog)?;
        Ok(crate::engine::fixed_cut::FixedCut {
            first,
            bound,
            join_end,
            history: self.known_cut_history()?,
            scope,
            _pin: pin,
        })
    }

    /// The last entry of its group's log this member applied and accounts
    /// for; `0` before any. A reader of the log stops after it: an entry past
    /// it may never commit.
    pub fn group_applied_through(&self) -> u64 {
        self.closure.lock().ledger.last_index()
    }

    /// [`Self::fixed_cut`] of a member applying its group's log. `L` is the
    /// applied prefix, and the commit ceiling of that prefix is the lowest
    /// bound the view may take. The bound and the join end come from one
    /// reading of the applied closure: every commit below its closed bound
    /// is in an entry up to the last one it accounts for. A member that
    /// applies only what the group made durable needs no sync.
    fn fixed_cut_of_group(
        &self,
        scope: crate::engine::fixed_cut::CutScope,
        timeout: std::time::Duration,
    ) -> StorageResult<crate::engine::fixed_cut::FixedCut> {
        let not_ready = |reason: String| StorageError::CutNotReady { reason };
        // With no applies running nothing moves the closed bound: it is the
        // one the Schema tree records, read as a state machine opening over
        // the store reads it.
        if self.raft_fence().is_none() {
            self.reload_closure_frontier()?;
        }
        let applied = self.closure.lock().ledger.last_index();
        let first = applied
            .checked_add(1)
            .ok_or_else(|| StorageError::Io("the applied log index overflowed".into()))?;
        let lower = if applied == 0 {
            0
        } else {
            self.commit_ceiling_through(applied)?
        };
        if !self.await_complete_through(lower, timeout) {
            return Err(not_ready(format!(
                "the group's closed bound has not passed {lower}"
            )));
        }
        // One reading: the bound and the entry it was closed by.
        let (bound, last) = {
            let state = self.closure.lock();
            (state.ledger.below(), state.ledger.last_index())
        };
        if bound <= lower {
            // A bound of none proves no empty history.
            return Err(not_ready(format!(
                "the group's closed bound {bound} is not above {lower}"
            )));
        }
        let pin = self.pin_readable_snapshot_at(bound).map_err(|oldest| {
            not_ready(format!(
                "the view at {bound} is no longer served; the oldest is {oldest}"
            ))
        })?;
        let join_end = last
            .checked_add(1)
            .ok_or_else(|| StorageError::Io("the applied log index overflowed".into()))?;
        self.qualify_group_cut(&scope, join_end)?;
        Ok(crate::engine::fixed_cut::FixedCut {
            first,
            bound,
            join_end,
            history: self.known_cut_history()?,
            scope,
            _pin: pin,
        })
    }

    /// Refuse a group member's cut of `scope` whose view the scope's tree
    /// does not physically hold as the closure says: every entry below
    /// `join_end` applied, none beyond installed into it. The closure ledger
    /// lives in the Schema tree, which can stand further along than another
    /// tree after a reopen, and a partition installed from a peer ahead of
    /// the applies holds the entries up to its copy's position as rows below
    /// every commit, which a view at the bound would show whatever their
    /// commit timestamps.
    ///
    /// A running member is asked with its applies paused; a store with none
    /// running is qualified by the scope tree's durable coverage alone.
    ///
    /// # Errors
    ///
    /// [`StorageError::CutNotReady`] when it is not held so, yet.
    fn qualify_group_cut(
        &self,
        scope: &crate::engine::fixed_cut::CutScope,
        join_end: u64,
    ) -> StorageResult<()> {
        let partition = scope.partition();
        let tree = scope.tree()?;
        let not_ready = |reason: String| StorageError::CutNotReady { reason };
        if self.is_under_repair(tree) {
            return Err(not_ready(format!(
                "the {} partition is under repair",
                partition.name()
            )));
        }
        let Some(fence) = self.raft_fence() else {
            return match self.raft_coverage()?.base_of(tree) {
                Some(next) if next == join_end => Ok(()),
                held => Err(not_ready(format!(
                    "the {} tree is shown to hold its group's log through {}, not through \
                     entry {} the closure accounts for; open it as a member until its trees \
                     hold the entries",
                    partition.name(),
                    held.unwrap_or(0),
                    join_end - 1
                ))),
            };
        };
        let mut verdict = Ok(());
        fence.with_applies_paused(&mut |applies, _| {
            if applies.next() < join_end {
                verdict = Err(not_ready(format!(
                    "the applies stand at entry {}, behind the closure's {}",
                    applies.next(),
                    join_end
                )));
            } else if applies.floor(tree) > applies.next() {
                verdict = Err(not_ready(format!(
                    "the {} partition was installed at entry {}, ahead of the applies at {}",
                    partition.name(),
                    applies.floor(tree),
                    applies.next()
                )));
            }
            Ok(())
        })?;
        verdict
    }

    /// Read `cut`'s continuation from this store's own journal at `from`:
    /// at most `max_entries` entries, within `budget`, only of entries
    /// applied and durable. An entry not yet applied may still be refused by
    /// its operation binding, and one not yet durable may be lost: neither
    /// is handed over until it is settled.
    ///
    /// The journal is held from `from` for the read only, and its lock only
    /// while the hold and the readable end are taken: the segments are read
    /// without it.
    ///
    /// # Errors
    ///
    /// [`StorageError::RetentionLost`] when `from` is no longer held;
    /// [`StorageError::InvalidConfig`] for a store without a journal; see
    /// [`Self::cut_read_log`].
    pub fn cut_read<'b>(
        &self,
        cut: &crate::engine::fixed_cut::FixedCut,
        from: u64,
        max_entries: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
    ) -> StorageResult<(
        crate::engine::fixed_cut::CutRead,
        coordinode_core::budget::MemoryCharge<'b>,
    )> {
        self.held_cut(cut)?;
        let (Some(oplog), Some(coverage)) = (&self.oplog, &self.coverage) else {
            return Err(StorageError::InvalidConfig(
                "the store keeps no journal".into(),
            ));
        };
        let (_hold, dir, until) = {
            let journal = oplog
                .lock()
                .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
            let first_retained = journal.first_index();
            if from < first_retained {
                return Err(StorageError::RetentionLost {
                    requested: from,
                    first_retained,
                });
            }
            let hold = crate::engine::oplog_journal::JournalHold::take(&self.journal_holds, from);
            // Applied in a contiguous prefix and durable: every entry below
            // both is settled.
            let until = journal.durable_below().min(coverage.applied_prefix());
            (hold, journal.dir().to_path_buf(), until)
        };
        self.cut_read_log(cut, &[dir], from, until, max_entries, budget)
    }

    /// Read the committed units of this store's own journal from `from`, for
    /// a change feed: at most `max_entries` entries, within `budget`, only of
    /// entries applied and durable, every unit whole with all its
    /// operations. An entry not yet applied may still be refused by its
    /// operation binding, and one not yet durable may be lost: neither is
    /// handed over until it is settled. The units keep their memory charged
    /// to `budget` through the returned charge.
    ///
    /// The journal is held from `from` for the read only; a reader's position
    /// does not keep it.
    ///
    /// # Errors
    ///
    /// [`StorageError::RetentionLost`] when `from` is no longer held;
    /// [`StorageError::InvalidConfig`] for a store without a journal;
    /// [`StorageError::Budget`] when the first entry does not fit the budget
    /// on its own; a confirmed entry that cannot be read.
    pub fn change_read<'b>(
        &self,
        from: u64,
        max_entries: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
    ) -> StorageResult<(
        crate::engine::change_feed::ChangeRead,
        coordinode_core::budget::MemoryCharge<'b>,
    )> {
        self.journal_change_read(from, max_entries, budget, None)
    }

    /// [`Self::change_read`] with the selections `follower`'s interest made
    /// of the entries read, when a follower is given.
    fn journal_change_read<'b>(
        &self,
        from: u64,
        max_entries: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
        follower: Option<&crate::engine::decoded_log::ChangeFollower>,
    ) -> StorageResult<(
        crate::engine::change_feed::ChangeRead,
        coordinode_core::budget::MemoryCharge<'b>,
    )> {
        let (Some(oplog), Some(coverage)) = (&self.oplog, &self.coverage) else {
            return Err(StorageError::InvalidConfig(
                "the store keeps no journal".into(),
            ));
        };
        let (_hold, log, until) = {
            let journal = oplog
                .lock()
                .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
            let first_retained = journal.first_index();
            if from < first_retained {
                return Err(StorageError::RetentionLost {
                    requested: from,
                    first_retained,
                });
            }
            let hold = crate::engine::oplog_journal::JournalHold::take(&self.journal_holds, from);
            let until = journal.durable_below().min(coverage.applied_prefix());
            let log = self.decoded_log(
                &[journal.dir().to_path_buf()],
                Some(journal.retained_floor()),
            );
            (hold, log, until)
        };
        self.change_read_of(&log, from, until, max_entries, budget, follower)
    }

    /// The decoded entries of the log in `dirs`, shared by every change-feed
    /// reader of it: made on first use, with `floor` when its writer is this
    /// store's own journal, or with what the log's writer registered
    /// ([`Self::register_log`]).
    pub(crate) fn decoded_log(
        &self,
        dirs: &[PathBuf],
        floor: Option<crate::oplog::RetainedFloor>,
    ) -> Arc<crate::engine::decoded_log::DecodedLog> {
        let mut key = dirs.to_vec();
        key.sort();
        let mut logs = self.decoded_logs.lock();
        if let Some(log) = logs.get(&key) {
            return Arc::clone(log);
        }
        let floor = floor.or_else(|| {
            let floors = self.log_floors.lock();
            dirs.iter().find_map(|dir| floors.get(dir).cloned())
        });
        let log = Arc::new(crate::engine::decoded_log::DecodedLog::new(
            key.clone(),
            floor,
            Arc::clone(&self.decoded_memory),
        ));
        logs.insert(key, Arc::clone(&log));
        log
    }

    /// Record that a writer of this store keeps a log in `dirs` and publishes
    /// `floor` for it: where the log starts and how often it was cut back.
    /// Change feeds reading the log then know a segment rewritten by a cut
    /// from the one it replaced. Call it before any feed reads the log.
    pub fn register_log(&self, dirs: &[PathBuf], floor: crate::oplog::RetainedFloor) {
        let mut floors = self.log_floors.lock();
        for dir in dirs {
            floors.insert(dir.clone(), floor.clone());
        }
    }

    /// Set how many bytes of decoded log entries the store holds for its
    /// change feeds, across its logs (`cdc_buffer_bytes`). A lower target
    /// lets go of the oldest held entries now; entries readers still hold
    /// stay charged until they let go.
    pub fn set_change_feed_cache_bytes(&self, bytes: usize) {
        self.decoded_memory.set_target(bytes);
        for log in self.decoded_logs.lock().values() {
            log.trim();
        }
    }

    /// The target of the decoded entries held for change feeds, and what
    /// they take now, those only readers still hold included.
    pub fn change_feed_cache_bytes(&self) -> (usize, usize) {
        (self.decoded_memory.target(), self.decoded_memory.held())
    }

    /// The decoded entries of `log` from `from` below `until`, at most
    /// `max_entries`: shared with every reader of the log, none charged to
    /// `budget` but those the store cannot hold, which the read decodes
    /// within what `budget` has left and keeps charged through the returned
    /// charge.
    pub(crate) fn change_read_of<'b>(
        &self,
        log: &Arc<crate::engine::decoded_log::DecodedLog>,
        from: u64,
        until: u64,
        max_entries: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
        follower: Option<&crate::engine::decoded_log::ChangeFollower>,
    ) -> StorageResult<(
        crate::engine::change_feed::ChangeRead,
        coordinode_core::budget::MemoryCharge<'b>,
    )> {
        if follower.is_some_and(|f| !f.follows_log(log)) {
            return Err(StorageError::InvalidConfig(
                "the change follower follows another log than the read's".into(),
            ));
        }
        let mut charge = budget.empty_charge();
        let mut selections = Vec::new();
        let (entries, next) = if from < until && max_entries > 0 {
            // What the read itself holds while it runs, as a read of the log
            // always did: a segment's read buffer, the one path it holds, and
            // a point read of the deployment's records deciding a unit's
            // operation binding (a marker of an empty value under a short key).
            let marker = crate::engine::operations::skipped_unit_key(
                coordinode_core::txn::operation::LogPosition { index: 0, sub: 0 },
            )
            .len() as u64;
            let (fixed, per_record_byte) =
                scan_hold_terms(self.system_tree(Partition::Deployment)?);
            // Runs are bounded by the tree's levels: far below u64::MAX.
            let working = crate::oplog::segment::FRAME_CURSOR_BUFFER as u64
                + READER_PATH_ROOM
                + fixed
                + per_record_byte * marker;
            charge.grow(working).map_err(StorageError::Budget)?;
            let mut reader = crate::engine::decoded_log::Reader {
                budget,
                charge: &mut charge,
            };
            let read = log.read(self, from, until, max_entries, &mut reader);
            // The selection a followed interest made of each entry, made once
            // for all its followers.
            let read = read.and_then(|(entries, next)| {
                if let Some(follower) = follower {
                    selections = log.select(follower, &entries, &mut reader)?;
                }
                Ok((entries, next))
            });
            charge.shrink(working);
            read?
        } else {
            (Vec::new(), from)
        };
        Ok((
            crate::engine::change_feed::ChangeRead {
                entries,
                next,
                selections,
            },
            charge,
        ))
    }

    /// A follower of `interest` on this store's journal: while some
    /// follower of an interest lives, reads of it given the follower select
    /// each unit once for all of them. Drop it when the reader leaves.
    ///
    /// # Errors
    ///
    /// [`StorageError::InvalidConfig`] for a store without a journal.
    pub fn follow_changes(
        &self,
        interest: &crate::engine::change_feed::Interest,
    ) -> StorageResult<crate::engine::decoded_log::ChangeFollower> {
        let oplog = self
            .oplog
            .as_ref()
            .ok_or_else(|| StorageError::InvalidConfig("the store keeps no journal".into()))?;
        let (dir, floor) = {
            let journal = oplog
                .lock()
                .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
            (journal.dir().to_path_buf(), journal.retained_floor())
        };
        Ok(self.decoded_log(&[dir], Some(floor)).follow(interest))
    }

    /// [`Self::follow_changes`] on the log segments in `dirs`: the consensus
    /// log of the group this store applies.
    pub fn follow_changes_log(
        &self,
        dirs: &[PathBuf],
        interest: &crate::engine::change_feed::Interest,
    ) -> crate::engine::decoded_log::ChangeFollower {
        self.decoded_log(dirs, None).follow(interest)
    }

    /// [`Self::change_read`] over the log segments in `dirs`, from `from` up
    /// to (not including) `until`, the end of what is confirmed: the
    /// consensus log of the group this store applies, read up to the last
    /// entry it applied.
    ///
    /// # Errors
    ///
    /// As [`Self::change_read`].
    pub fn change_read_log<'b>(
        &self,
        dirs: &[std::path::PathBuf],
        from: u64,
        until: u64,
        max_entries: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
    ) -> StorageResult<(
        crate::engine::change_feed::ChangeRead,
        coordinode_core::budget::MemoryCharge<'b>,
    )> {
        let log = self.decoded_log(dirs, None);
        self.change_read_of(&log, from, until, max_entries, budget, None)
    }

    /// A change cursor at `position` of this store's journal, for
    /// `interest`: where a reader that holds everything before `position`
    /// continues, as a registered consumer's acknowledged position says.
    ///
    /// # Errors
    ///
    /// [`StorageError::InvalidConfig`] while the store's history is not
    /// known.
    pub fn change_cursor_at(
        &self,
        position: u64,
        interest: crate::engine::change_feed::Interest,
    ) -> StorageResult<crate::engine::change_feed::ChangeCursor> {
        let history = self.cut_history()?.ok_or_else(|| {
            StorageError::InvalidConfig("the store's history is not known yet".into())
        })?;
        Ok(crate::engine::change_feed::ChangeCursor::new(
            history, interest, position,
        ))
    }

    /// A change cursor after every entry of this store's journal applied now:
    /// a reader starting here gets what commits from now on. Entries apply
    /// out of journal order, so an earlier entry may still be applying when a
    /// later one has applied and returned to its writer; the cursor waits for
    /// those and comes after them all. The journal is synced first, so an
    /// entry applied under a volatile level is behind the cursor too.
    ///
    /// # Errors
    ///
    /// As [`Self::change_cursor_at`]; [`StorageError::InvalidConfig`] for a
    /// store without a journal; [`StorageError::Io`] when the journal cannot
    /// be synced, or an earlier entry does not finish applying within
    /// [`CURSOR_APPLY_WAIT`].
    pub fn change_cursor_at_end(
        &self,
        interest: crate::engine::change_feed::Interest,
    ) -> StorageResult<crate::engine::change_feed::ChangeCursor> {
        let (Some(oplog), Some(coverage)) = (&self.oplog, &self.coverage) else {
            return Err(StorageError::InvalidConfig(
                "the store keeps no journal".into(),
            ));
        };
        crate::engine::oplog_journal::sync_journal(oplog)?;
        let applied = coverage.prefix_past_applied(CURSOR_APPLY_WAIT).ok_or_else(|| {
            StorageError::Io(format!(
                "an entry before the cursor did not finish applying within {CURSOR_APPLY_WAIT:?}"
            ))
        })?;
        let end = {
            let journal = oplog
                .lock()
                .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
            journal.durable_below().min(applied)
        };
        self.change_cursor_at(end, interest)
    }

    /// Read the feed from `cursor`: the units of at most `max_entries`
    /// journal entries, within `budget`, each whole with the events of it
    /// the cursor's interest takes, and the cursor to continue from. A
    /// reader's position keeps nothing of the journal: one that falls
    /// behind what it holds is told so.
    ///
    /// # Errors
    ///
    /// [`StorageError::ChangeCursor`] for a cursor of another history;
    /// [`StorageError::RetentionLost`] when the journal no longer holds the
    /// cursor's position; [`StorageError::ChangeFeedUnsupported`] for a unit
    /// the feed cannot project; see [`Self::change_read`]. On any of them
    /// nothing is handed over and the cursor stays valid.
    ///
    /// A reader that reads again and again passes the `follower` it holds
    /// for the cursor's interest ([`Self::follow_changes`]): every follower
    /// of the interest then shares what it selects of each unit.
    /// [`StorageError::InvalidConfig`] for a follower of another interest
    /// or log.
    pub fn read_changes<'b>(
        &self,
        cursor: &crate::engine::change_feed::ChangeCursor,
        max_entries: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
        follower: Option<&crate::engine::decoded_log::ChangeFollower>,
    ) -> StorageResult<(
        crate::engine::change_feed::ChangeBatch,
        coordinode_core::budget::MemoryCharge<'b>,
    )> {
        self.check_cursor_history(cursor)?;
        Self::check_follower(cursor, follower)?;
        let (read, mut charge) =
            self.journal_change_read(cursor.position(), max_entries, budget, follower)?;
        let batch = Self::batch_of(
            cursor,
            &read,
            &mut charge,
            &mut crate::engine::change_feed::Unencoded,
        )?;
        Ok((batch, charge))
    }

    /// [`Self::read_changes`] over the log segments in `dirs`, up to (not
    /// including) `until`: the consensus log of the group this store
    /// applies, read up to the last entry it applied.
    ///
    /// # Errors
    ///
    /// As [`Self::read_changes`].
    pub fn read_changes_log<'b>(
        &self,
        cursor: &crate::engine::change_feed::ChangeCursor,
        dirs: &[std::path::PathBuf],
        until: u64,
        max_entries: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
        follower: Option<&crate::engine::decoded_log::ChangeFollower>,
    ) -> StorageResult<(
        crate::engine::change_feed::ChangeBatch,
        coordinode_core::budget::MemoryCharge<'b>,
    )> {
        self.read_changes_log_encoded(
            cursor,
            dirs,
            until,
            max_entries,
            budget,
            &mut crate::engine::change_feed::Unencoded,
            follower,
        )
    }

    /// [`Self::read_changes_log`] for a reader that holds its own form of
    /// the batch with it, as `encoding` says what each unit and event of
    /// that form takes: the batch stops where the events and their form no
    /// longer fit the budget together, and the returned charge holds both.
    /// The entries are decoded once for every reader of the log and shared;
    /// only this reader's events and their form are charged to `budget`.
    ///
    /// # Errors
    ///
    /// As [`Self::read_changes_log`].
    #[allow(clippy::too_many_arguments)]
    pub fn read_changes_log_encoded<'b>(
        &self,
        cursor: &crate::engine::change_feed::ChangeCursor,
        dirs: &[std::path::PathBuf],
        until: u64,
        max_entries: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
        encoding: &mut impl crate::engine::change_feed::ChangeEncoding,
        follower: Option<&crate::engine::decoded_log::ChangeFollower>,
    ) -> StorageResult<(
        crate::engine::change_feed::ChangeBatch,
        coordinode_core::budget::MemoryCharge<'b>,
    )> {
        self.check_cursor_history(cursor)?;
        Self::check_follower(cursor, follower)?;
        let log = self.decoded_log(dirs, None);
        let (read, mut charge) = self.change_read_of(
            &log,
            cursor.position(),
            until,
            max_entries,
            budget,
            follower,
        )?;
        let batch = Self::batch_of(cursor, &read, &mut charge, encoding)?;
        Ok((batch, charge))
    }

    /// Refuse a follower of another interest than the cursor's.
    fn check_follower(
        cursor: &crate::engine::change_feed::ChangeCursor,
        follower: Option<&crate::engine::decoded_log::ChangeFollower>,
    ) -> StorageResult<()> {
        if follower.is_some_and(|f| f.interest() != cursor.interest()) {
            return Err(StorageError::InvalidConfig(
                "the change follower follows another interest than the cursor".into(),
            ));
        }
        Ok(())
    }

    /// Refuse a cursor whose positions count another log than this store's.
    fn check_cursor_history(
        &self,
        cursor: &crate::engine::change_feed::ChangeCursor,
    ) -> StorageResult<()> {
        if self.cut_history()? != Some(cursor.history()) {
            return Err(crate::engine::change_feed::CursorError::OtherHistory.into());
        }
        Ok(())
    }

    /// The batch `read` hands a reader at `cursor`, its events charged to
    /// `charge`: it stops inside a unit when the charge refuses an event, and
    /// its cursor continues there.
    fn batch_of(
        cursor: &crate::engine::change_feed::ChangeCursor,
        read: &crate::engine::change_feed::ChangeRead,
        charge: &mut coordinode_core::budget::MemoryCharge<'_>,
        encoding: &mut impl crate::engine::change_feed::ChangeEncoding,
    ) -> StorageResult<crate::engine::change_feed::ChangeBatch> {
        use crate::engine::change_feed::{ChangeBatch, ChangeCursor, ReadPlace, unit_changes_from};
        let (units, stopped) = unit_changes_from(
            read.selected_units(),
            cursor.interest(),
            ReadPlace {
                entry: cursor.position(),
                within: cursor.within_entry(),
            },
            charge,
            encoding,
        )?;
        let next = match stopped {
            Some(place) => ChangeCursor::within(
                cursor.history(),
                cursor.interest().clone(),
                place.entry,
                place.within,
            ),
            None => ChangeCursor::new(cursor.history(), cursor.interest().clone(), read.next),
        };
        Ok(ChangeBatch::read_from(cursor.clone(), units, next))
    }

    /// The units of one entry for a change feed, read in place and appended
    /// to `units`, each decoded within what is left of `pool`. A data change
    /// recorded outside a unit carries no effect context and cannot be
    /// projected, so it fails the read rather than vanish from the feed.
    pub(crate) fn change_units_of_entry(
        &self,
        entry: &crate::oplog::OplogEntryRef<'_>,
        pool: &mut crate::engine::fixed_cut::EntryPool,
        units: &mut Vec<crate::engine::change_feed::ChangeUnit>,
    ) -> Result<(), EntryOutcome> {
        use crate::oplog::OplogOpRef;
        use coordinode_core::txn::operation::LogPosition;
        use coordinode_core::txn::proposal::Mutation;
        let unsupported = |what: &str| {
            EntryOutcome::Failed(StorageError::ChangeFeedUnsupported {
                reason: format!("entry {} carries {what}", entry.index),
            })
        };
        let mut sub = 0u32;
        for op in &entry.ops {
            let frame = match op {
                OplogOpRef::Unit { frame } => *frame,
                OplogOpRef::Insert { .. }
                | OplogOpRef::Delete { .. }
                | OplogOpRef::Merge { .. }
                | OplogOpRef::RemoveRange { .. } => return Err(unsupported("a raw row change")),
                OplogOpRef::ColumnarInsert { .. } => return Err(unsupported("a columnar insert")),
                // Index entries and framing: nothing a feed projects.
                OplogOpRef::Derive { .. }
                | OplogOpRef::Noop
                | OplogOpRef::RaftTruncation { .. }
                | OplogOpRef::RaftEntry { .. }
                | OplogOpRef::Namespace { .. } => continue,
            };
            let position = LogPosition {
                index: entry.index,
                sub,
            };
            sub = sub.checked_add(1).ok_or_else(|| {
                EntryOutcome::Failed(StorageError::Serialization(format!(
                    "entry {} carries over 2^32 units",
                    entry.index
                )))
            })?;
            let (unit, expanded) = match coordinode_core::txn::frame::decode_proposal_rows(
                frame,
                &pool.decode_limits(),
            ) {
                Ok(decoded) => decoded,
                Err(coordinode_core::txn::frame::FrameError::OverLimit { .. }) => {
                    return Err(EntryOutcome::NoRoom);
                }
                Err(e) => return Err(EntryOutcome::Failed(e.into())),
            };
            let held = (unit.mutations.capacity() * core::mem::size_of::<Mutation>()) as u64
                + expanded as u64;
            if !pool.retain(held) {
                return Err(EntryOutcome::NoRoom);
            }
            let refused = coordinode_core::txn::proposal::decided_at_apply(&unit.mutations)
                && crate::engine::operations::unit_skipped(self, position)
                    .map_err(EntryOutcome::Failed)?;
            // The unit's namespace, by its marker. A unit naming several
            // would mix their keys in one projection, which names no
            // namespace per event.
            let mut namespace = None;
            for m in &unit.mutations {
                if let Mutation::Namespace(named) = m {
                    match namespace {
                        Some(known) if known != *named => {
                            return Err(unsupported("a unit of several namespaces"));
                        }
                        _ => namespace = Some(*named),
                    }
                }
            }
            let mut mutations = unit.mutations;
            if refused {
                mutations.clear();
            }
            if units.len() == units.capacity() {
                let more = units.capacity().max(4);
                if !pool.grow(
                    (more * core::mem::size_of::<crate::engine::change_feed::ChangeUnit>()) as u64,
                ) {
                    return Err(EntryOutcome::NoRoom);
                }
                units.reserve_exact(more);
            }
            units.push(crate::engine::change_feed::ChangeUnit {
                position,
                commit_ts: unit.commit_ts.as_raw(),
                refused,
                namespace,
                mutations,
            });
        }
        Ok(())
    }

    /// Read `cut`'s continuation from the log segments in `dirs`, from
    /// `from` up to (not including) `until`, the end of what is confirmed:
    /// at most `max_entries` entries, within `budget`. The units handed over
    /// keep their memory charged to `budget` through the returned charge.
    ///
    /// Frames are read one at a time, each frame's length checked against
    /// the budget before its payload is read and each unit decoded within
    /// what is left of it, so the read holds no more than it hands over plus
    /// one frame. A read stops before an entry that does not fit, which a
    /// later read with more room takes. Every confirmed entry is either
    /// handed over or accounted for: one the log no longer holds is
    /// [`StorageError::RetentionLost`], one it holds but cannot read an
    /// error, never an empty read as if nothing were there yet.
    ///
    /// # Errors
    ///
    /// [`StorageError::CutLost`] for a cut this store does not hold;
    /// [`StorageError::RetentionLost`] when the log no longer holds the next
    /// entry; [`StorageError::Budget`] when the first entry does not fit the
    /// budget on its own; a confirmed entry that cannot be read; see
    /// [`Self::cut_units_of`]. On any of them the read hands nothing over.
    pub fn cut_read_log<'b>(
        &self,
        cut: &crate::engine::fixed_cut::FixedCut,
        dirs: &[std::path::PathBuf],
        from: u64,
        until: u64,
        max_entries: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
    ) -> StorageResult<(
        crate::engine::fixed_cut::CutRead,
        coordinode_core::budget::MemoryCharge<'b>,
    )> {
        self.check_cut_history(cut)?;
        let mut units = Vec::new();
        let (next, charge) = self.read_log_units(
            dirs,
            from,
            until,
            max_entries,
            budget,
            &mut units,
            |entry, pool, units| self.units_of_entry(cut, entry, pool, units),
        )?;
        Ok((crate::engine::fixed_cut::CutRead { units, next }, charge))
    }

    /// Read the log segments in `dirs` from `from` up to (not including)
    /// `until`: at most `max_entries` entries, within `budget`, each entry
    /// turned into units by `units_of` and appended to `units`. Returns the
    /// position after the last entry read and the charge the units keep.
    ///
    /// Frames are read one at a time, each frame's length checked against
    /// the budget before its payload is read and each unit decoded within
    /// what is left of it, so the read holds no more than it hands over plus
    /// one frame. A read stops before an entry that does not fit, which a
    /// later read with more room takes. Every confirmed entry is either
    /// handed over or accounted for: one the log no longer holds is
    /// [`StorageError::RetentionLost`], one it holds but cannot read an
    /// error, never an empty read as if nothing were there yet.
    #[allow(clippy::too_many_arguments)]
    fn read_log_units<'b, U>(
        &self,
        dirs: &[std::path::PathBuf],
        from: u64,
        until: u64,
        max_entries: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
        units: &mut Vec<U>,
        mut units_of: impl FnMut(
            &crate::oplog::OplogEntryRef<'_>,
            &mut crate::engine::fixed_cut::EntryPool,
            &mut Vec<U>,
        ) -> Result<(), EntryOutcome>,
    ) -> StorageResult<(u64, coordinode_core::budget::MemoryCharge<'b>)> {
        use crate::engine::fixed_cut::EntryPool;
        use crate::oplog::segment::{FRAME_CURSOR_BUFFER, FrameCursor, Peeked};
        let mut charge = budget.empty_charge();
        let mut next = from;
        let mut taken = 0usize;
        let stop = || coordinode_core::budget::BudgetStop::Memory {
            requested: 0,
            used: budget.memory_used(),
            limit: budget.memory_limit(),
        };
        // A segment's read buffer, the one path a search holds, and a point
        // read of the deployment's records deciding a unit's operation
        // binding: a marker of an empty value under a short key.
        let marker = crate::engine::operations::skipped_unit_key(
            coordinode_core::txn::operation::LogPosition { index: 0, sub: 0 },
        )
        .len() as u64;
        let (fixed, per_record_byte) = scan_hold_terms(self.system_tree(Partition::Deployment)?);
        // Runs are bounded by the tree's levels: far below u64::MAX.
        let reader =
            FRAME_CURSOR_BUFFER as u64 + READER_PATH_ROOM + fixed + per_record_byte * marker;
        charge.grow(reader).map_err(StorageError::Budget)?;
        'log: while next < until && taken < max_entries {
            let at = next;
            let (held_from, holding) = crate::oplog::tailer::segment_holding(dirs, next)?;
            let Some((_, path)) = holding else {
                charge.shrink(reader);
                return Err(StorageError::RetentionLost {
                    requested: next,
                    first_retained: held_from.unwrap_or(until),
                });
            };
            #[cfg(test)]
            if let Some(hook) = self.on_segment_found.lock().as_mut() {
                hook(&path);
            }
            let mut frames = match FrameCursor::open(&path) {
                Ok(frames) => frames,
                // Purged since it was found: whether the entry is gone is
                // read from the log as it stands now.
                Err(_) if !path.exists() => continue,
                Err(e) => {
                    charge.shrink(reader);
                    return Err(e);
                }
            };
            loop {
                let (len, index, ops) = match frames.peek()? {
                    Peeked::End => break,
                    Peeked::Frame { len, index, ops } => (len, index, ops),
                };
                if index < next {
                    frames.skip();
                    continue;
                }
                if index > next || next >= until || taken >= max_entries {
                    break;
                }
                // The entry's working memory, taken before anything of it is
                // read: what the budget has left, its payload and its list
                // of operations out of it first. What is not spent returns.
                debug_assert!(budget.memory_used() <= budget.memory_limit());
                // A budget refuses every reservation past its limit.
                let room = budget.memory_limit() - budget.memory_used();
                let envelope = ops
                    .checked_mul(core::mem::size_of::<crate::oplog::OplogOpRef<'_>>() as u64)
                    .and_then(|bytes| bytes.checked_add(len));
                let Some(fixed) = envelope.filter(|fixed| *fixed <= room) else {
                    if taken == 0 {
                        charge.shrink(reader);
                        return Err(StorageError::Budget(stop()));
                    }
                    break 'log;
                };
                if let Err(refused) = charge.grow(room) {
                    // Taken by another part of the budget meanwhile.
                    if taken == 0 {
                        charge.shrink(reader);
                        return Err(StorageError::Budget(refused));
                    }
                    break 'log;
                }
                let mut pool = EntryPool::new(room - fixed);
                let start = units.len();
                let outcome = match frames.read_payload()? {
                    Some(payload) => match crate::oplog::OplogEntryRef::decode(&payload) {
                        Ok(entry) if entry.index == index => units_of(&entry, &mut pool, units),
                        _ => Err(EntryOutcome::Unreadable),
                    },
                    None => Err(EntryOutcome::Unreadable),
                };
                match outcome {
                    Ok(()) => {
                        // The payload and the operation list are gone; the
                        // units and the room they grew the batch by stay.
                        charge.shrink(room - pool.kept());
                        next = index + 1;
                        taken += 1;
                    }
                    Err(failed) => {
                        units.truncate(start);
                        // Only the batch's grown room stays.
                        charge.shrink(room - pool.grown());
                        match failed {
                            EntryOutcome::NoRoom if taken == 0 => {
                                charge.shrink(reader);
                                return Err(StorageError::Budget(stop()));
                            }
                            EntryOutcome::NoRoom => break 'log,
                            EntryOutcome::Unreadable => break,
                            EntryOutcome::Failed(e) => {
                                charge.shrink(reader);
                                return Err(e);
                            }
                        }
                    }
                }
            }
            if next == at {
                charge.shrink(reader);
                // The segment that should hold the next confirmed entry ends
                // before it: purged meanwhile, or unreadable.
                let (first, holding) = crate::oplog::tailer::segment_holding(dirs, next)?;
                return Err(if holding.is_some() {
                    StorageError::Io(format!(
                        "the log holds entry {next}, below its confirmed end {until}, but it \
                         cannot be read"
                    ))
                } else {
                    StorageError::RetentionLost {
                        requested: next,
                        first_retained: first.unwrap_or(until),
                    }
                });
            }
        }
        charge.shrink(reader);
        Ok((next, charge))
    }
}

impl StorageEngine {
    /// Take a fixed cut of this store's history for `scope`'s rows in this
    /// engine's namespace: a view pinned below a fresh bound `B`, the first
    /// journal position `L` its continuation replays, and the join end `H`
    /// (see [`crate::engine::fixed_cut`]). Cold work: nothing is added to
    /// writes, and the journal is not searched. The view and the units of
    /// the continuation are of this namespace only.
    ///
    /// `L` is read first; `B` is then drawn from the commit clock, above
    /// every commit journalled before `L`. The view is qualified once every
    /// commit below `B` has finished and is durable. `H` is read after that,
    /// and every entry before it is then waited for and made durable: a late
    /// entry below `B` falls inside the join, not after it.
    ///
    /// # Errors
    ///
    /// [`StorageError::InvalidConfig`] for a store without a journal of its
    /// own; [`StorageError::CutNotReady`] when a commit is still in flight or
    /// not durable after `timeout`; a journal sync failure.
    pub fn fixed_cut(
        &self,
        scope: crate::engine::fixed_cut::CutScope,
        timeout: std::time::Duration,
    ) -> StorageResult<crate::engine::fixed_cut::FixedCut> {
        self.fixed_cut_of(scope.in_namespace(self.namespace()), timeout)
    }

    /// The position `resume` names in this store's log, where reading of
    /// `scope`'s rows in this engine's namespace goes on after a cut's join.
    ///
    /// # Errors
    ///
    /// [`StorageError::InvalidConfig`] for a position of another history or
    /// reached for another scope or namespace: it counts entries of another
    /// log, or the consumer's state covers other rows.
    pub fn resume_position(
        &self,
        resume: &crate::engine::fixed_cut::CutResume,
        scope: &crate::engine::fixed_cut::CutScope,
    ) -> StorageResult<u64> {
        if self.cut_history()? != Some(resume.history) {
            return Err(StorageError::InvalidConfig(
                "the position is of another history".into(),
            ));
        }
        if resume.scope != scope.clone().in_namespace(self.namespace()) {
            return Err(StorageError::InvalidConfig(
                "the position was reached for another scope".into(),
            ));
        }
        Ok(resume.next)
    }

    /// Refuse `cut` taken in another namespace than this engine's.
    ///
    /// # Errors
    ///
    /// [`StorageError::InvalidConfig`].
    fn cut_of_namespace(&self, cut: &crate::engine::fixed_cut::FixedCut) -> StorageResult<()> {
        if cut.scope.namespace() != Some(self.namespace()) {
            return Err(StorageError::InvalidConfig(
                "the cut was taken in another namespace".into(),
            ));
        }
        Ok(())
    }

    /// One page of `cut`'s view: the rows of its scope in key order, past
    /// `after` when given, at most `max_rows` of them, within `budget`; the
    /// rows keep their memory charged to it through the returned charge.
    /// The read starts at `after`, not at the scope's start, so a page costs
    /// what it hands over, and a row's size is checked before it is copied.
    ///
    /// # Errors
    ///
    /// [`StorageError::CutLost`] for a cut this store does not hold;
    /// [`StorageError::Budget`] for a first row that does not fit the budget
    /// on its own; the view no longer readable, or a read that fails.
    pub fn cut_page<'b>(
        &self,
        cut: &crate::engine::fixed_cut::FixedCut,
        after: Option<&[u8]>,
        max_rows: usize,
        budget: &'b coordinode_core::budget::QueryBudget,
    ) -> StorageResult<(
        crate::engine::fixed_cut::CutRows,
        coordinode_core::budget::MemoryCharge<'b>,
    )> {
        use std::ops::Bound;
        self.held_cut(cut)?;
        self.cut_of_namespace(cut)?;
        self.scope_view_intact(cut)?;
        self.check_snapshot_retained(cut.scope.partition(), cut.bound)?;
        let prefix = cut.scope.prefix();
        let lo = match after {
            // A key at or past the prefix's start: the page goes on after it.
            Some(after) if after >= prefix => Bound::Excluded(after),
            _ => Bound::Included(prefix),
        };
        let hi = cut.scope.end().map_or(Bound::Unbounded, Bound::Excluded);
        let mut page = crate::engine::fixed_cut::CutRows::default();
        let mut charge = budget.empty_charge();
        let refused = || {
            StorageError::Budget(coordinode_core::budget::BudgetStop::Memory {
                requested: 0,
                used: budget.memory_used(),
                limit: budget.memory_limit(),
            })
        };
        // What the scan holds while the page is read, for the tree's runs as
        // they stand and the longest row the page can then still take.
        debug_assert!(budget.memory_used() <= budget.memory_limit());
        // A budget refuses every reservation past its limit.
        let room = budget.memory_limit() - budget.memory_used();
        let Some((scan, longest)) = self.scan_room(cut.scope.partition(), room)? else {
            return Err(refused());
        };
        charge.grow(scan).map_err(StorageError::Budget)?;
        for guard in self.scan_range_at(cut.scope.partition(), lo, hi, cut.bound)? {
            if page.len() >= max_rows {
                break;
            }
            // Shared with the block they were read from: nothing is copied
            // until the row is known to fit.
            let (key, value) = guard.into_inner()?;
            // Slices of memory that exists: their lengths fit in u64.
            let row = (key.len() + value.len()) as u64;
            // A row longer than the scan was given room for is not taken.
            if row > longest {
                if page.is_empty() {
                    return Err(refused());
                }
                break;
            }
            // Copied into the page's blocks; the budget is asked only when a
            // block or the index has to grow.
            if let Err(stop) = page.push(&key, &value, &mut charge, budget) {
                if page.is_empty() {
                    return Err(StorageError::Budget(stop));
                }
                break;
            }
        }
        charge.shrink(scan);
        // A repair that replaced the tree while the page was read leaves
        // rows of another copy in it: the page is not handed over.
        self.scope_view_intact(cut)?;
        Ok((page, charge))
    }

    /// Refuse `cut` once its scope's tree no longer holds the view: a repair
    /// of the partition began (the tree is cleared or holds a copy partly
    /// installed), or one replaced it, raising the tree's floor past the
    /// view.
    ///
    /// # Errors
    ///
    /// [`StorageError::CutLost`].
    fn scope_view_intact(&self, cut: &crate::engine::fixed_cut::FixedCut) -> StorageResult<()> {
        use lsm_tree::AbstractTree as _;
        let partition = cut.scope.partition();
        if self.is_under_repair(self.tree_id(partition))
            || self.tree(partition)?.oldest_retained_seqno() >= cut.bound
        {
            return Err(StorageError::CutLost { bound: cut.bound });
        }
        Ok(())
    }
}

impl Deployment {
    /// Refuse a cut whose view this store does not hold: one taken of a
    /// store closed since. Its bound may still be readable here, but nothing
    /// kept it, so serving it would hide that the view was lost.
    ///
    /// # Errors
    ///
    /// [`StorageError::CutLost`].
    pub fn held_cut(&self, cut: &crate::engine::fixed_cut::FixedCut) -> StorageResult<()> {
        if self.coordinator.holds(&cut._pin) {
            return Ok(());
        }
        Err(StorageError::CutLost { bound: cut.bound })
    }

    /// The units `entries` carry, entries of this store's own journal or of
    /// the consensus log it applies, read in log order: each unit with its
    /// own commit timestamp, whether `cut`'s view already holds it, and its
    /// data changes inside the cut's scope. A unit its operation binding
    /// refused carries no data changes.
    ///
    /// # Errors
    ///
    /// [`StorageError::CutLost`] for a cut this store does not hold;
    /// [`StorageError::InvalidConfig`] for a cut of another history;
    /// [`StorageError::CutUnsupported`] for an effect inside the scope the
    /// cut cannot carry; a frame or decision that cannot be read.
    pub fn cut_units_of(
        &self,
        cut: &crate::engine::fixed_cut::FixedCut,
        entries: &[crate::oplog::OplogEntry],
    ) -> StorageResult<Vec<crate::engine::fixed_cut::CutUnit>> {
        self.check_cut_history(cut)?;
        let mut units = Vec::new();
        for entry in entries {
            let payload = entry
                .encode()
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            let entry = crate::oplog::OplogEntryRef::decode(&payload)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            let mut pool = crate::engine::fixed_cut::EntryPool::new(u64::MAX);
            self.units_of_entry(cut, &entry, &mut pool, &mut units)
                .map_err(|failed| match failed {
                    EntryOutcome::Failed(e) => e,
                    EntryOutcome::NoRoom | EntryOutcome::Unreadable => {
                        StorageError::Serialization(format!("entry {} cannot be read", entry.index))
                    }
                })?;
        }
        Ok(units)
    }

    /// Refuse a cut this store does not hold, or one of another history.
    fn check_cut_history(&self, cut: &crate::engine::fixed_cut::FixedCut) -> StorageResult<()> {
        self.held_cut(cut)?;
        if self.cut_history()? != Some(cut.history) {
            return Err(StorageError::InvalidConfig(
                "the cut was taken of another history".into(),
            ));
        }
        Ok(())
    }

    /// The units of one entry, read in place, appended to `units`: each
    /// unit decoded within what is left of `pool`, which its decoded form
    /// and the room the batch grows by are kept from.
    fn units_of_entry(
        &self,
        cut: &crate::engine::fixed_cut::FixedCut,
        entry: &crate::oplog::OplogEntryRef<'_>,
        pool: &mut crate::engine::fixed_cut::EntryPool,
        units: &mut Vec<crate::engine::fixed_cut::CutUnit>,
    ) -> Result<(), EntryOutcome> {
        use crate::oplog::OplogOpRef;
        use coordinode_core::txn::operation::LogPosition;
        use coordinode_core::txn::proposal::Mutation;
        let scope = &cut.scope;
        let unsupported = |what: &str| {
            EntryOutcome::Failed(StorageError::CutUnsupported {
                reason: format!("entry {} carries {what} inside the scope", entry.index),
            })
        };
        let mut sub = 0u32;
        // The namespace the raw ops after a marker are of.
        let mut raw_namespace = None;
        for op in &entry.ops {
            let frame = match op {
                OplogOpRef::Unit { frame } => *frame,
                // A raw data op carries no commit timestamp of its own to
                // decide against the view by: inside the scope it cannot be
                // carried, outside it touches nothing handed over.
                OplogOpRef::Insert { partition, key, .. }
                | OplogOpRef::Delete { partition, key }
                | OplogOpRef::Merge { partition, key, .. } => {
                    match crate::placement::partition_from_wire_tag(*partition) {
                        Some(p) if p != scope.partition() => continue,
                        Some(_) if scope.names_other(raw_namespace) => continue,
                        Some(_) if !scope.holds_key(key) => continue,
                        _ => return Err(unsupported("a raw row write")),
                    }
                }
                OplogOpRef::RemoveRange {
                    partition,
                    start,
                    end,
                } => match crate::placement::partition_from_wire_tag(*partition) {
                    Some(p) if p != scope.partition() => continue,
                    Some(_) if scope.names_other(raw_namespace) => continue,
                    Some(_) if !scope.meets_range(start, end) => continue,
                    _ => return Err(unsupported("a raw range delete")),
                },
                // Index entries: a row scope never holds the index partition.
                // Columnar rows live in their tables' own trees, outside every
                // partition.
                OplogOpRef::Derive { .. } | OplogOpRef::ColumnarInsert { .. } => continue,
                // Framing: a gap fill, a log truncation, and a group log entry's
                // envelope, which holds its closing bound or a membership
                // change; its proposals follow it as units.
                OplogOpRef::Noop
                | OplogOpRef::RaftTruncation { .. }
                | OplogOpRef::RaftEntry { .. } => continue,
                // Names the namespace of the raw ops after it, which carry
                // none of their own.
                OplogOpRef::Namespace { id } => {
                    raw_namespace = Some(coordinode_core::namespace::NamespaceId::from_raw(*id));
                    continue;
                }
            };
            let position = LogPosition {
                index: entry.index,
                sub,
            };
            sub = sub.checked_add(1).ok_or_else(|| {
                EntryOutcome::Failed(StorageError::Serialization(format!(
                    "entry {} carries over 2^32 units",
                    entry.index
                )))
            })?;
            // A reader of rows: the unit's metadata commands and DERIVED work
            // are read to their ends and never built.
            let (unit, expanded) = match coordinode_core::txn::frame::decode_proposal_rows(
                frame,
                &pool.decode_limits(),
            ) {
                Ok(decoded) => decoded,
                Err(coordinode_core::txn::frame::FrameError::OverLimit { .. }) => {
                    return Err(EntryOutcome::NoRoom);
                }
                Err(e) => return Err(EntryOutcome::Failed(e.into())),
            };
            // A unit of another namespace holds nothing of the scope's.
            let mut unit_namespace = None;
            for m in &unit.mutations {
                if let Mutation::Namespace(named) = m {
                    match unit_namespace {
                        Some(known) if known != *named => {
                            return Err(unsupported("a unit of several namespaces"));
                        }
                        _ => unit_namespace = Some(*named),
                    }
                }
            }
            if !scope.holds_namespace(unit_namespace) {
                continue;
            }
            // What the decoded unit holds: its operations, of the capacity
            // decoded, and the bytes they own. Within the decode bounds.
            let held = (unit.mutations.capacity() * core::mem::size_of::<Mutation>()) as u64
                + expanded as u64;
            if !pool.retain(held) {
                return Err(EntryOutcome::NoRoom);
            }
            let commit_ts = unit.commit_ts.as_raw();
            let refused = coordinode_core::txn::proposal::decided_at_apply(&unit.mutations)
                && crate::engine::operations::unit_skipped(self, position)
                    .map_err(EntryOutcome::Failed)?;
            let mut mutations = unit.mutations;
            if refused {
                mutations.clear();
            }
            // The unit's events in the scope, numbered over its whole
            // projection before the scope chooses its rows below.
            // Only the events in the scope are built, each kept within the
            // pool as it is handed over.
            let mut events: Vec<crate::engine::change_feed::ChangeEvent> = Vec::new();
            let mut no_room = false;
            // The ordinal after the last event handed over: short of the
            // unit's count when its working state ran out of room.
            let mut through = 0u32;
            let projection =
                crate::engine::change_feed::project_within(&mutations, 0, pool, |event, pool| {
                    use crate::engine::change_feed::Take;
                    through = event.ordinal + 1;
                    if !scope.holds_target(&event.target) {
                        return Take::More;
                    }
                    // The list's room and the event's, kept with the unit,
                    // taken before they are allocated.
                    if events.len() == events.capacity() {
                        let slot =
                            core::mem::size_of::<crate::engine::change_feed::ChangeEvent>() as u64;
                        let old = events.capacity();
                        let new = old.max(2) * 2;
                        if !pool.retain(new as u64 * slot) {
                            no_room = true;
                            return Take::Enough;
                        }
                        events.reserve_exact(new - events.len());
                        pool.unretain(old as u64 * slot);
                    }
                    if !pool.retain(crate::engine::change_feed::event_heap_bytes(&event)) {
                        no_room = true;
                        return Take::Enough;
                    }
                    events.push(crate::engine::change_feed::owned_event(
                        position,
                        &event,
                        crate::engine::change_feed::Delivery::Apply,
                    ));
                    Take::More
                });
            let events = match projection {
                _ if no_room => return Err(EntryOutcome::NoRoom),
                Ok(projected) if through < projected => return Err(EntryOutcome::NoRoom),
                Ok(projected) => Some(crate::engine::fixed_cut::CutEvents { events, projected }),
                Err(crate::engine::change_feed::ProjectStop::NoRoom) => {
                    return Err(EntryOutcome::NoRoom);
                }
                // Rows without the context their meaning is read by: a
                // consumer of rows applies them; one of events is told.
                Err(crate::engine::change_feed::ProjectStop::Projection(_)) => None,
            };
            // Kept in place: no second list is built.
            let mut no_room = false;
            mutations.retain_mut(|m| match m {
                Mutation::Put { partition, key, .. }
                | Mutation::Delete { partition, key }
                | Mutation::Merge { partition, key, .. } => {
                    scope.holds_partition(*partition) && scope.holds_key(key)
                }
                // Only the part inside the scope: the rest deletes rows the
                // consumer does not hold. Bounds that change are copied once
                // their room is taken from the pool: a clipped bound can be
                // longer than the one it replaces.
                Mutation::RemoveRange {
                    partition,
                    start,
                    end,
                } => {
                    if !scope.holds_partition(*partition) {
                        return false;
                    }
                    let Some((lo, hi)) = scope.clip_range(start, end) else {
                        return false;
                    };
                    if (lo, hi) == (start.as_slice(), end.as_slice()) {
                        return true;
                    }
                    if no_room || !pool.retain((lo.len() + hi.len()) as u64) {
                        no_room = true;
                        return false;
                    }
                    let (lo, hi) = (lo.to_vec(), hi.to_vec());
                    *start = lo;
                    *end = hi;
                    true
                }
                // Its effects land in the schema partition, which no row
                // scope is.
                Mutation::Command(_) => false,
                // Index entries, in the index partition.
                Mutation::Derive(_) => false,
                // Decides the unit, already read as `refused`.
                Mutation::Operation(_) | Mutation::Actor(_) => false,
                // What the rows mean, not a row: a copy of rows applies none.
                Mutation::Effects(_) => false,
                // Names the namespace of the rows after it: kept with them.
                Mutation::Namespace(_) => true,
            });
            if no_room {
                return Err(EntryOutcome::NoRoom);
            }
            if units.len() == units.capacity() {
                let more = units.capacity().max(4);
                if !pool
                    .grow((more * core::mem::size_of::<crate::engine::fixed_cut::CutUnit>()) as u64)
                {
                    return Err(EntryOutcome::NoRoom);
                }
                units.reserve_exact(more);
            }
            units.push(crate::engine::fixed_cut::CutUnit {
                position,
                commit_ts,
                mutations,
                refused,
                in_view: cut.in_view(position, commit_ts),
                events,
            });
        }
        Ok(())
    }

    /// Record the group whose log the member running over the engine
    /// applies, when its node opens.
    pub fn set_log_group(&self, group: coordinode_core::group::GroupId) {
        *self.log_group.write() = Some(group);
    }

    /// The history this store's log positions count entries of: its own
    /// journal's, named by its deployment, or its group's log. `None` while
    /// the deployment, or the group of a member, is not known.
    ///
    /// # Errors
    ///
    /// The deployment record cannot be read.
    pub fn cut_history(&self) -> StorageResult<Option<crate::engine::fixed_cut::CutHistory>> {
        use crate::engine::fixed_cut::CutHistory;
        let Some(deployment) = crate::engine::metadata::bound_deployment(self)? else {
            return Ok(None);
        };
        if self.oplog.is_some() {
            return Ok(Some(CutHistory::Journal(deployment)));
        }
        Ok((*self.log_group.read()).map(|group| {
            CutHistory::Group(coordinode_core::txn::read_concern::HistoryId { deployment, group })
        }))
    }

    /// [`Self::cut_history`] for a cut, which names its history.
    fn known_cut_history(&self) -> StorageResult<crate::engine::fixed_cut::CutHistory> {
        self.cut_history()?
            .ok_or_else(|| StorageError::CutNotReady {
                reason: "the history of the store's log is not known yet".into(),
            })
    }

    /// Keep the journal from `index` on for a rebuild from the latest local
    /// checkpoint ([`Self::checkpoint_replay_floor`]); `u64::MAX` when none
    /// needs it. Reclamation under a footprint limit never removes it.
    pub fn set_journal_keep_from(&self, index: u64) {
        self.journal_keep_from
            .store(index, std::sync::atomic::Ordering::Release);
    }

    /// Make room in the owned footprint: flush the memtables, which turns
    /// what they owe into tables, then remove the journal segments every tree
    /// then covers durably. What a rebuild from the latest checkpoint replays
    /// is kept, and so is every entry whose writes are not yet in a table.
    /// Covered segments go even inside the journal's retention window: the
    /// trees hold their effects, and the window is a time bound on history,
    /// not an obligation the limit may be broken for.
    ///
    /// One runs at a time; a call arriving while one runs waits for it and
    /// returns, its outcome being the room there is.
    pub fn reclaim_footprint(&self) {
        let Some(_one) = self.footprint_reclaim.try_lock() else {
            drop(self.footprint_reclaim.lock());
            return;
        };
        self.reclaim_counted();
    }

    /// One reclamation under the held lock; whether it failed.
    fn reclaim_counted(&self) -> bool {
        metrics::counter!("coordinode_footprint_reclamations_total").increment(1);
        match self.reclaim_journal() {
            Ok(()) => false,
            Err(e) => {
                tracing::warn!(error = %e, "footprint reclamation failed");
                true
            }
        }
    }

    /// Start reclaiming, in the background, the journal segments the trees
    /// cover as flushes land, rather than first when a write is refused for
    /// room. Only a store with its own journal and a hard limit has any; a
    /// consensus log keeps its own bounds. Runs until the returned handle
    /// drops.
    ///
    /// # Errors
    ///
    /// The thread cannot be spawned.
    pub fn start_journal_reclaim(
        self: &Arc<Self>,
    ) -> StorageResult<Option<crate::engine::journal_reclaim::JournalReclaimer>> {
        if self.oplog.is_none() || self.footprints.is_empty() {
            return Ok(None);
        }
        self.footprints
            .set_reclaim_wake(Arc::clone(&self.table_landed));
        crate::engine::journal_reclaim::JournalReclaimer::start(
            Arc::downgrade(self),
            Arc::clone(&self.table_landed),
        )
        .map(Some)
    }

    /// The reclaimer's wake, for a test that runs its rounds by hand.
    #[cfg(test)]
    pub(crate) fn table_landed(&self) -> Arc<coordinode_core::txn::wake::Wake> {
        Arc::clone(&self.table_landed)
    }

    /// One background reclamation, run only when it may remove something the
    /// last could not ([`crate::engine::journal_reclaim::Round::due`]). It
    /// takes its own turn after any reclamation running, and records what it
    /// read before it ran, so grounds that arrive meanwhile are not taken as
    /// checked. A round left with nothing new to do does not run again.
    pub(crate) fn journal_reclaim_round(
        &self,
        last: &mut Option<crate::engine::journal_reclaim::Round>,
    ) {
        use crate::engine::journal_reclaim::{Grounds, Round};
        let Some(oplog) = &self.oplog else {
            return;
        };
        let grounds = || {
            oplog.lock().ok().map(|log| Grounds {
                newest_sealed: log.newest_sealed(),
                floor: self
                    .journal_keep_from
                    .load(std::sync::atomic::Ordering::Acquire)
                    .min(self.consumer_journal_needs()),
            })
        };
        let _one = self.footprint_reclaim.lock();
        let Some(checked) = grounds() else {
            return;
        };
        if checked.newest_sealed.is_none() || last.is_some_and(|round| !round.due(checked)) {
            return;
        }
        // Begun before the round: room coming back while it runs is seen if
        // it then fails, and its own failed output wakes nothing.
        let attempt = self.footprints.flush_attempt();
        let failed = self.reclaim_counted();
        if failed {
            attempt.failed();
        } else {
            // The segments it removed are room a stalled flush waits for,
            // which the attempt kept from counting as its own cleanup.
            drop(attempt);
            self.footprints.room_given_back();
        }
        *last = Some(Round { checked, failed });
    }

    fn reclaim_journal(&self) -> StorageResult<()> {
        let Some(oplog) = &self.oplog else {
            return self.persist();
        };
        // Every entry below `covered` is in a table of every tree.
        let covered = self.persist_coverage_base()?;
        #[cfg(test)]
        if self
            .fail_journal_purge
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Err(StorageError::Io("injected journal purge failure".into()));
        }
        let mut log = oplog
            .lock()
            .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
        // What consumers need is read under the journal lock a hold is taken
        // under.
        let below = covered
            .min(
                self.journal_keep_from
                    .load(std::sync::atomic::Ordering::Acquire),
            )
            .min(self.consumer_journal_needs());
        // A segment is removed only once sealed: the active one is sealed when
        // all it holds is covered, and the next entry starts a new one.
        if log.next_index() <= below {
            log.rotate()?;
        }
        let removed = log.purge_before(below)?;
        tracing::debug!(removed, below, "footprint reclamation purged the journal");
        Ok(())
    }

    /// Record, durably, that this store's log is a consensus log of the store
    /// incarnation `incarnation`: from here on the store opens only as a
    /// member of its group, never with a journal of its own. Called before
    /// the log's first entry is written, and again to give a marker written
    /// before markers carried an incarnation its one; recording the same
    /// incarnation again does nothing.
    ///
    /// # Errors
    ///
    /// The engine was opened with a journal of its own, which a consensus
    /// log cannot share the store with; no endpoint can hold the log; the
    /// marker names another incarnation; the record cannot be written.
    pub fn bind_consensus_log(
        &self,
        incarnation: &crate::engine::promise::Incarnation,
    ) -> StorageResult<()> {
        if self.oplog.is_some() {
            return Err(StorageError::InvalidConfig(
                "a consensus log cannot start on a store opened with a journal of its own".into(),
            ));
        }
        match self.consensus_binding()? {
            ConsensusBinding::Bound(bound) if bound == *incarnation => Ok(()),
            ConsensusBinding::Bound(_) => Err(StorageError::InvalidConfig(
                "the consensus log is bound to another store incarnation".into(),
            )),
            ConsensusBinding::Unbound | ConsensusBinding::Unnamed => {
                crate::engine::promise::replace_file(
                    &*self.fs,
                    &self.select_oplog_endpoint(0)?.path,
                    CONSENSUS_LOG_MARKER,
                    crate::engine::promise::format_incarnation(incarnation).as_bytes(),
                    &self.footprints,
                )
            }
        }
    }

    /// Whether this store's log is bound to a consensus group, and the store
    /// incarnation the binding names.
    ///
    /// # Errors
    ///
    /// No endpoint can hold the log, or the marker cannot be read or names
    /// no incarnation.
    pub fn consensus_binding(&self) -> StorageResult<ConsensusBinding> {
        let marker = self
            .select_oplog_endpoint(0)?
            .path
            .join(CONSENSUS_LOG_MARKER);
        let Some(bytes) = crate::engine::promise::read_file(&*self.fs, &marker)? else {
            return Ok(ConsensusBinding::Unbound);
        };
        Ok(
            match crate::engine::promise::parse_incarnation(&String::from_utf8_lossy(&bytes))? {
                Some(incarnation) => ConsensusBinding::Bound(incarnation),
                None => ConsensusBinding::Unnamed,
            },
        )
    }

    /// The second copy of this member's consensus promise, kept beside the
    /// consensus log out of reach of every tree repair.
    ///
    /// # Errors
    ///
    /// No endpoint can hold the log, or the file exists and cannot be read.
    pub fn read_promise_copy(&self) -> StorageResult<crate::engine::promise::PromiseCopy> {
        crate::engine::promise::read_copy(&*self.fs, &self.select_oplog_endpoint(0)?.path)
    }

    /// Write `payload`, for the store incarnation `incarnation`, as the
    /// newest record of the second copy of this member's consensus records:
    /// durable when this returns. Written even when the copy already reads
    /// the same: what an earlier process left may never have reached the
    /// disk.
    ///
    /// # Errors
    ///
    /// No endpoint can hold the log, the endpoint has no room for the file,
    /// or a write or sync fails. Nothing is promised then: the next write
    /// makes the copy durable again.
    pub fn write_promise_copy(
        &self,
        incarnation: &crate::engine::promise::Incarnation,
        payload: &[u8],
    ) -> StorageResult<()> {
        let mut writer = self.promise_copy.lock();
        crate::engine::promise::write_copy(
            &*self.fs,
            &self.select_oplog_endpoint(0)?.path,
            &mut writer,
            incarnation,
            payload,
            &self.footprints,
        )
    }

    /// Bring the second copy of the consensus records up to date with
    /// `current`, which reads them from where they are kept live: `current`
    /// runs under the same lock as every other write, so the copy never goes
    /// back to records older than what an earlier write left. Records this
    /// process already wrote durably are not written again; anything else,
    /// a copy that only reads the same included, is.
    ///
    /// # Errors
    ///
    /// The errors of `current`, or of writing the copy.
    pub fn refresh_promise_copy(
        &self,
        incarnation: &crate::engine::promise::Incarnation,
        current: impl FnOnce() -> StorageResult<Vec<u8>>,
    ) -> StorageResult<()> {
        let mut writer = self.promise_copy.lock();
        let payload = current()?;
        if writer.holds(incarnation, &payload) {
            return Ok(());
        }
        crate::engine::promise::write_copy(
            &*self.fs,
            &self.select_oplog_endpoint(0)?.path,
            &mut writer,
            incarnation,
            &payload,
            &self.footprints,
        )
    }

    /// Fold the journal's applied prefix into every tree, including the ones
    /// no entry reached, and persist them: every index below the returned
    /// bound is covered durably in every tree, so journal entries below it
    /// can go without leaving a hole a later open cannot account for. 0 on a
    /// store without a journal.
    ///
    /// # Errors
    ///
    /// The flush fails.
    pub(crate) fn persist_coverage_base(&self) -> StorageResult<u64> {
        let Some(coverage) = &self.coverage else {
            return Ok(0);
        };
        let mut folded = coverage.lock_folded();
        let next = self.fold_coverage(coverage, &mut folded, coverage::FoldScope::All);
        // The flush persists every memtable sealed so far, and with it the
        // base just written.
        self.persist()?;
        Ok(next)
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

    /// Pin a snapshot at `seqno` only if this store still serves reads there,
    /// or the oldest seqno it does serve.
    ///
    /// The watermark alone is not that bound: a partition cleared, or the
    /// tables a range tombstone spans dropped, stops serving below its own
    /// floor whatever the watermark says (see [`Self::oldest_readable_seqno`]).
    /// The floors are checked under the pin: from then on no collection can
    /// raise one past `seqno`, so a pin granted here protects history that is
    /// there.
    pub fn pin_readable_snapshot_at(
        &self,
        seqno: lsm_tree::SeqNo,
    ) -> Result<SnapshotPin, lsm_tree::SeqNo> {
        let Some(pin) = self.coordinator.pin_snapshot_at(seqno) else {
            return Err(self.oldest_readable_seqno());
        };
        let horizon = self.oldest_readable_seqno();
        if seqno < horizon {
            return Err(horizon);
        }
        Ok(pin)
    }

    /// The latest complete snapshot, already pinned.
    #[cfg_attr(feature = "stage-timing", track_caller)]
    pub fn pin_latest_snapshot(&self) -> (lsm_tree::SeqNo, Option<SnapshotPin>) {
        #[cfg(feature = "stage-timing")]
        let (caller, start) = (std::panic::Location::caller(), std::time::Instant::now());
        let pinned = self.pin_new_snapshot(|| self.snapshot());
        #[cfg(feature = "stage-timing")]
        crate::stage_timing::record_from("snapshot.pin_latest", caller, start);
        pinned
    }

    /// The cut an offline capture of the whole store reads every partition
    /// at (strictly below it), pinned for as long as the capture reads.
    ///
    /// A store outside any group learns of every commit before it lands, so
    /// its latest complete snapshot is such a cut. A store holding a group's
    /// history applied commits in log order, which need not be timestamp
    /// order: its cut is the log's closed bound as its Schema tree records
    /// it, below which every commit is in an entry applied before. That
    /// bound only holds where every partition physically holds every entry
    /// it accounts for, which the trees' coverage bases must show; a power
    /// cut that kept the bound and lost an entry behind it in another tree
    /// is refused, not captured as if the bound held. Every included tree,
    /// the consumer registry among them although it keeps no history of its
    /// own, must still serve the cut under the pin, or the capture is
    /// refused rather than read a later state.
    ///
    /// # Errors
    ///
    /// [`StorageError::CaptureRefused`] when no cut qualifies; a read of the
    /// closed bound or of a coverage record fails.
    pub fn pin_capture_cut(&self) -> StorageResult<(lsm_tree::SeqNo, Option<SnapshotPin>)> {
        use lsm_tree::AbstractTree as _;
        let refuse = |reason: String| StorageError::CaptureRefused { reason };
        let (cut, pin) = if self.group_history {
            let ledger = self.stored_closure_ledger()?;
            let coverage = self.raft_coverage()?;
            // Every tree of every namespace but the node-local consensus one.
            let mut included: Vec<TreeId> = coverage
                .trees()
                .filter(|id| id.part != Partition::Raft)
                .collect();
            included.sort();
            for id in included {
                match coverage.base_of(id) {
                    Some(next) if next > ledger.last_index() => {}
                    held => {
                        return Err(refuse(format!(
                            "its closed bound accounts for its group's log through entry {}, \
                             but its {id} tree is shown to hold that log only below {}; open \
                             it as a member until its trees hold the entries, then capture",
                            ledger.last_index(),
                            held.unwrap_or(0)
                        )));
                    }
                }
            }
            #[cfg(feature = "columnar")]
            if !self.coordinator.sets().table_trees().is_empty() {
                return Err(refuse(
                    "its STORAGE COLUMNAR tables record no position in its group's log, so \
                     no closed cut of them can be shown complete"
                        .into(),
                ));
            }
            let cut = self.snapshot().min(ledger.below());
            if cut == 0 {
                return Err(refuse(
                    "no commit of its group's log is known closed".into(),
                ));
            }
            let pin = self.pin_readable_snapshot_at(cut).map_err(|horizon| {
                refuse(format!(
                    "its closed cut {cut} is below the oldest state it still serves, {horizon}"
                ))
            })?;
            (cut, Some(pin))
        } else {
            self.pin_latest_snapshot()
        };
        // Under the pin, no collection raises a floor past the cut any more.
        for (id, tree) in self.coordinator.trees() {
            if id.part == Partition::Raft {
                continue;
            }
            let floor = tree.oldest_retained_seqno();
            if cut <= floor {
                return Err(refuse(format!(
                    "its {id} tree no longer holds its state at the cut {cut} (collected \
                     through {floor})"
                )));
            }
        }
        Ok((cut, pin))
    }

    /// [`Self::pin_latest_snapshot`] for a read that must not see a write a
    /// crash can still take back: the latest complete snapshot, lowered to
    /// [`Self::durable_view`].
    #[cfg_attr(feature = "stage-timing", track_caller)]
    pub fn pin_durable_snapshot(&self) -> (lsm_tree::SeqNo, Option<SnapshotPin>) {
        #[cfg(feature = "stage-timing")]
        let (caller, start) = (std::panic::Location::caller(), std::time::Instant::now());
        let pinned = self.pin_new_snapshot(|| self.durable_view(self.snapshot()));
        #[cfg(feature = "stage-timing")]
        crate::stage_timing::record_from("snapshot.pin_durable", caller, start);
        pinned
    }

    /// Hold the set of trees for an operation that changes them outside one
    /// commit, until the returned hold drops: a capture or repair waits for
    /// it (see [`crate::engine::tree_set`]). Waits while one has the set.
    pub fn hold_tree_set(&self) -> crate::engine::tree_set::TreeSetHold {
        self.tree_set.hold()
    }

    /// The gate over the set of trees.
    pub(crate) fn tree_set_gate(&self) -> &crate::engine::tree_set::TreeSetGate {
        &self.tree_set
    }

    /// Name the catalog's list of tables: from here on a capture of a store
    /// no Raft log orders is refused while the catalog and the table trees
    /// disagree ([`StorageError::CatalogMismatch`]).
    pub fn set_catalog_inventory(
        &self,
        inventory: Arc<dyn crate::engine::tree_set::CatalogInventory>,
    ) {
        *self.catalog.lock() = Some(inventory);
    }

    /// The catalog's list of tables, if one is named.
    pub(crate) fn catalog_inventory(
        &self,
    ) -> Option<Arc<dyn crate::engine::tree_set::CatalogInventory>> {
        self.catalog.lock().clone()
    }

    /// Shared holds of the set of trees granted so far, for tests that count
    /// them.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn tree_set_holds_granted(&self) -> u64 {
        self.tree_set.holds_granted()
    }

    /// Record a transaction as open until the returned marker drops.
    /// [`Self::await_transactions_through`] waits for the ones opened before
    /// a boundary.
    pub(crate) fn open_transaction(&self) -> crate::engine::open_txns::OpenTransaction {
        self.open_transactions
            .open(|| self.coordinator.current_seqno())
    }

    /// The write concern for a commit no answer depends on (field-dictionary
    /// registration, index-build pages and bookkeeping). On a store with its
    /// own journal it is logged and applied like any commit and left to the
    /// next sync: an answer that promises durability, a table flush or the
    /// journal's flusher covers it, and a crash before that loses only what
    /// recovery redoes (the log keeps it ahead of every entry that relies on
    /// it). A group keeps majority: consensus syncs its log by itself.
    pub fn bookkeeping_concern(&self) -> WriteConcern {
        if self.oplog.is_some() {
            WriteConcern {
                w: coordinode_core::txn::write_concern::WriteAck::LEADER,
                journal: Journal::Cache,
                timeout_ms: 0,
            }
        } else {
            WriteConcern::majority()
        }
    }

    /// Memtables this store has written to tables on demand since it opened
    /// (persist, partition persist, checkpoint, close); background flushes
    /// by size are not counted. For observing what a workload costs.
    pub fn memtable_flushes(&self) -> u64 {
        self.memtable_flushes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Device flushes this store's journal has completed since it opened:
    /// its own journal, or the member-local one when its log is a consensus
    /// log; `None` for a store kept in memory. For observing what a workload
    /// costs in syncs.
    pub fn journal_syncs(&self) -> Option<u64> {
        let guard = self.journal_in_use()?.lock().ok()?;
        Some(guard.deferred().syncs())
    }

    /// The highest snapshot at or below `view` that holds only durable
    /// effects: below every commit still in flight and below every journal
    /// entry not yet synced. A read that must not see a write a crash can
    /// still take back (`majority`, `snapshot`, `linearizable`) reads here;
    /// `local` reads `view`. Chosen from what memory already holds: no sync,
    /// no wait. A store whose log is a consensus log applies only what is
    /// durable, so for it the answer is `view`.
    pub fn durable_view(&self, view: lsm_tree::SeqNo) -> lsm_tree::SeqNo {
        if self.oplog.is_none() {
            return view;
        }
        self.pending_commits.durable_floor(view)
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
    ///   `clear` records its own install seqno, and dropping the tables a
    ///   range tombstone spans records up to its seqno, which prunes
    ///   history independently of the window, so this can sit above the
    ///   watermark. A read at or below the floor is refused by the tree itself
    ///   (`lsm_tree::Error::SnapshotBelowRetention`), hence the `+ 1`.
    ///
    /// The system partitions keep no history of their own (see
    /// [`Partition::is_system`]), so their floors run ahead of the window and
    /// do not bound a time-travel read of data.
    pub fn oldest_readable_seqno(&self) -> lsm_tree::SeqNo {
        let mut horizon = self.coordinator.gc_watermark_value();
        for (id, tree) in self.coordinator.trees() {
            if !id.part.is_system() {
                horizon = horizon.max(first_readable_in(&tree));
            }
        }
        horizon
    }
}

/// What one scan of `tree` holds, as a fixed part and a part per byte of its
/// longest record: [`StorageEngine::scan_hold_bytes`] is their sum.
fn scan_hold_terms(tree: &lsm_tree::AnyTree) -> (u64, u64) {
    use lsm_tree::AbstractTree as _;
    // The read's own state and memtable iterators.
    const READS_FIXED: u64 = 16 << 10;
    // The trees' data block size: the LSM default, which the engine keeps.
    const DATA_BLOCK: u64 = 4096;
    // A table read's iterator beside the block it stands in.
    const TABLE_READ: u64 = 1024 + DATA_BLOCK;
    // A range read opens a run's first table as a source of its own and the
    // rest as a second, one table at a time.
    const READS_PER_RUN: u64 = 2;
    // Each of the first level's runs and the one of each deeper level; the
    // memtables count as one more.
    let runs: u64 = tree
        .current_version()
        .iter_levels()
        .map(|level| level.run_count() as u64)
        .sum::<u64>()
        + 1;
    // A data block holds at most one record past its size.
    (
        READS_FIXED + READS_PER_RUN * runs * TABLE_READ,
        READS_PER_RUN * runs,
    )
}

/// The first seqno `tree` still serves: one past its persisted retention
/// floor.
fn first_readable_in(tree: &lsm_tree::AnyTree) -> lsm_tree::SeqNo {
    use lsm_tree::AbstractTree as _;
    // A tree serves strictly above its floor, so the first readable seqno is
    // one past it. Plain arithmetic: the floor is a watermark or an install
    // seqno, both from the seqno generator (HLC microseconds or a counter),
    // and is never `SeqNo::MAX`, the read-latest sentinel a caller passes to
    // a read. The `debug_assert` fails loudly if that ever stops holding,
    // which is where it would need fixing.
    let oldest_retained = tree.oldest_retained_seqno();
    debug_assert!(
        oldest_retained < lsm_tree::SeqNo::MAX,
        "retention floor at the read-latest sentinel"
    );
    oldest_retained + 1
}

impl StorageEngine {
    /// Refuse a snapshot read of `part` below the retention horizon: the GC
    /// watermark, or `part`'s own retention floor where it is higher. Version
    /// history there may already be collected; answering from what survived
    /// would return a wrong (newer or missing) version, and once the history
    /// is pruned the tree has no retained version to serve it from at all.
    ///
    /// Only the tree read takes part: another partition's floor says nothing
    /// about this one's history, and reading every tree's floor on each read
    /// takes each tree's version lock for nothing. The check stands before
    /// the read because a key whose index installation is gone is answered
    /// "absent" without reaching the tree, and that answer must hold at the
    /// snapshot too. The tree checks its own floor again as it reads, which
    /// covers a floor that rises in between. A system partition keeps no
    /// history of its own (see [`Partition::is_system`]), so only the
    /// watermark bounds a read of it, as for [`Self::oldest_readable_seqno`].
    fn check_snapshot_retained(
        &self,
        part: Partition,
        snapshot: lsm_tree::SeqNo,
    ) -> StorageResult<()> {
        let mut watermark = self.coordinator.gc_watermark_value();
        if !part.is_system()
            && let Ok(tree) = self.tree(part)
        {
            watermark = watermark.max(first_readable_in(tree));
        }
        if snapshot < watermark {
            return Err(StorageError::SnapshotOutsideRetention {
                snapshot,
                watermark,
            });
        }
        Ok(())
    }
}

impl Deployment {
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
        self.protections_pending
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Whether this store opened over consumer registrations whose floors no
    /// registry has published yet: collection, of MVCC history and of the
    /// journal, is held until one does.
    pub fn consumer_protections_pending(&self) -> bool {
        self.protections_pending
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// The current GC watermark value: the seqno below which a time-travel
    /// read may already find its versions collected. `AS OF TIMESTAMP T` is
    /// exact for every `T >= watermark` (the version current at the watermark
    /// survives compaction as the base) and refused below it.
    pub fn gc_watermark(&self) -> u64 {
        self.coordinator.gc_watermark_value()
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

    /// Capacity tracker handle — exposed so background scanners,
    /// Prometheus exporters, and admin RPCs can read per-endpoint
    /// usage state without going through `put`/`get` paths.
    pub fn capacity(&self) -> &Arc<crate::engine::capacity::CapacityTracker> {
        &self.capacity
    }

    /// Refresh the capacity report from the owned footprints (an endpoint
    /// without a hard limit by walking its folder). Side effects:
    /// severity-transition logs and gauges, and (when the endpoint strategy
    /// is `CascadeEvict`) a cascade-eviction fire at the 95% emergency
    /// threshold. Enforcement is the footprint's, not this report's.
    ///
    /// Synchronous — caller decides cadence. A background polling
    /// loop wrapper lives in the engine's open path.
    pub fn refresh_capacity(&self) {
        run_capacity_refresh(
            &self.capacity,
            &self.endpoints,
            &self.coordinator.trees(),
            &self.footprints,
            |id| self.cascade_evict_endpoint(id),
        );
    }

    /// The value of `key` in the system set's `part` now.
    ///
    /// # Errors
    ///
    /// `part` has no tree in the system set, or the read fails.
    pub fn system_get(&self, part: Partition, key: &[u8]) -> StorageResult<Option<bytes::Bytes>> {
        let value = self
            .system_tree(part)?
            .get(key, self.coordinator.current_seqno())?;
        Ok(crate::engine::revision::user_bytes(part, key, value)?)
    }

    /// Write `key` of the system set's `part` directly, at the next seqno,
    /// admitted on the owned footprint like any direct write.
    ///
    /// # Errors
    ///
    /// `part` has no tree in the system set, the write is not admitted, or
    /// the tree refuses it.
    pub fn system_put(&self, part: Partition, key: &[u8], value: &[u8]) -> StorageResult<()> {
        self.system_write(part, key, (key.len() + value.len()) as u64, |seqno| {
            let mut batch = lsm_tree::WriteBatch::with_capacity(1);
            batch.insert(
                key,
                crate::engine::revision::stored(part, key, value, seqno).as_ref(),
            );
            batch
        })
    }

    /// Delete `key` of the system set's `part` directly, as
    /// [`Self::system_put`] writes.
    ///
    /// # Errors
    ///
    /// As [`Self::system_put`].
    pub fn system_delete(&self, part: Partition, key: &[u8]) -> StorageResult<()> {
        self.system_write(part, key, key.len() as u64, |_| {
            let mut batch = lsm_tree::WriteBatch::with_capacity(1);
            batch.remove(key);
            batch
        })
    }

    /// Apply the batch `build` makes for the seqno the write lands at, one
    /// direct write of `key` carrying `bytes`, to the system set's `part`.
    fn system_write(
        &self,
        part: Partition,
        key: &[u8],
        bytes: u64,
        build: impl FnOnce(lsm_tree::SeqNo) -> lsm_tree::WriteBatch,
    ) -> StorageResult<()> {
        let tree = self.system_tree(part)?;
        let _admitted =
            self.admit_partition_write([part], crate::engine::footprint::footprint_of(bytes, 1))?;
        // Held until the data is in: no complete snapshot may cover its seqno
        // before.
        let (seqno, _landing) = self.pending_commits.admit_direct(|| self.next_seqno());
        let (added, memtable) = tree.apply_batch(build(seqno), seqno)?;
        self.coordinator
            .flush_trigger()
            .wrote(part, added, memtable);
        self.write_taps.wrote(TreeId::system(part), [key]);
        if let Some(cache) = &self.tiered_cache {
            cache.remove(TreeId::system(part), key);
        }
        Ok(())
    }

    /// The store bound to its one namespace: what a store of one namespace
    /// reads and writes through. A fresh store, one whose catalog never
    /// held a namespace, is initialized as its own: it creates the one
    /// namespace [`DEFAULT_NAMESPACE`](coordinode_core::namespace::DEFAULT_NAMESPACE).
    /// One whose namespaces were all dropped is not: its own namespace is
    /// never created again behind its catalog.
    ///
    /// # Errors
    ///
    /// [`StorageError::NoNamespace`] for a store whose namespaces were all
    /// dropped; [`StorageError::SeveralNamespaces`] for one holding several;
    /// the creation's errors.
    pub fn single_store(&self) -> StorageResult<StorageEngine> {
        let live: Vec<(NamespaceId, coordinode_core::namespace::NamespaceRecord)> =
            crate::engine::metadata::namespaces(self)?
                .into_iter()
                .filter(|(_, record)| {
                    record.state == coordinode_core::namespace::NamespaceState::Live
                })
                .collect();
        match live.as_slice() {
            [(id, _)] => self.engine(*id),
            [] if crate::engine::metadata::namespace_allocator(self)? == 1 => {
                let id = self.create_namespace(coordinode_core::namespace::DEFAULT_NAMESPACE)?;
                self.engine(id)
            }
            [] => Err(StorageError::NoNamespace {
                path: self.data_dir.display().to_string(),
            }),
            many => Err(StorageError::SeveralNamespaces {
                path: self.data_dir.display().to_string(),
                names: many.iter().map(|(_, record)| record.name.clone()).collect(),
            }),
        }
    }

    /// This store bound to its anchor namespace ([`Self::anchor_namespace`])
    /// when the catalog holds one, else as [`Self::single_store`] makes a
    /// store's first: what a host serving the namespaces of its own store
    /// opens, the others reached by name.
    ///
    /// # Errors
    ///
    /// As [`Self::single_store`]; the anchor's trees cannot be opened.
    pub fn anchored_store(&self) -> StorageResult<StorageEngine> {
        if self.live_namespaces()?.is_empty() {
            return self.single_store();
        }
        self.engine(self.anchor_namespace()?)
    }

    /// The catalog's one live namespace.
    ///
    /// # Errors
    ///
    /// [`StorageError::NoNamespace`] when the catalog holds none,
    /// [`StorageError::SeveralNamespaces`] when it holds more than one, or the
    /// catalog cannot be read.
    pub fn sole_namespace(&self) -> StorageResult<NamespaceId> {
        let live = self.live_namespaces()?;
        match live.as_slice() {
            [(id, _)] => Ok(*id),
            [] => Err(StorageError::NoNamespace {
                path: self.data_dir.display().to_string(),
            }),
            many => Err(StorageError::SeveralNamespaces {
                path: self.data_dir.display().to_string(),
                names: many.iter().map(|(_, name)| name.clone()).collect(),
            }),
        }
    }

    /// The catalog's anchor namespace: the live namespace of the lowest id,
    /// which a member's own database handle works in. Every other namespace
    /// is reached by a session naming it.
    ///
    /// # Errors
    ///
    /// [`StorageError::NoNamespace`] when the catalog holds none, or the
    /// catalog cannot be read.
    pub fn anchor_namespace(&self) -> StorageResult<NamespaceId> {
        self.live_namespaces()?
            .into_iter()
            .map(|(id, _)| id)
            .min()
            .ok_or_else(|| StorageError::NoNamespace {
                path: self.data_dir.display().to_string(),
            })
    }

    /// The engine of a member of a group: bound to the catalog's anchor
    /// namespace when it holds one, or, for a member whose namespaces arrive
    /// with its group's catalog, on its first use once they have. Creates no
    /// namespace: a store that joins a group brings none of its own.
    #[must_use]
    pub fn member_engine(self: &Arc<Self>) -> StorageEngine {
        let engine = StorageEngine {
            deployment: Arc::clone(self),
            bound: std::sync::OnceLock::new(),
        };
        // Bound now when it can be: a member restarting over its namespace.
        drop(engine.bound());
        engine
    }

    /// Create namespace `name` in this store's catalog and return its id.
    /// For a store that is not a member of a group: a member's catalog
    /// changes through its group's log.
    ///
    /// # Errors
    ///
    /// [`StorageError::NamespaceRefused`] for a name no namespace may have,
    /// one a live namespace holds, or a store holding a group's history; a
    /// write failure.
    pub fn create_namespace(&self, name: &str) -> StorageResult<NamespaceId> {
        self.create_namespace_with(
            name,
            coordinode_core::namespace::NamespaceStorage::default(),
        )
    }

    /// [`Self::create_namespace`] with the data-block storage settings
    /// `storage`; unset ones inherit the deployment's.
    ///
    /// # Errors
    ///
    /// As [`Self::create_namespace`].
    pub fn create_namespace_with(
        &self,
        name: &str,
        storage: coordinode_core::namespace::NamespaceStorage,
    ) -> StorageResult<NamespaceId> {
        let refused = |reason: String| StorageError::NamespaceRefused {
            name: name.to_owned(),
            reason,
        };
        coordinode_core::namespace::validate_namespace_name(name)
            .map_err(|e| refused(e.to_string()))?;
        self.refuse_local_catalog_change()
            .map_err(|e| refused(e.to_string()))?;
        let token = crate::engine::metadata::namespace_token();
        self.commit_command(
            coordinode_core::txn::proposal::MetadataCommand::CreateNamespace {
                name: name.to_owned(),
                token,
                storage,
            },
        )?;
        let taken = || refused("a live namespace has the name".into());
        let id = crate::engine::metadata::namespace_by_name(self, name)?.ok_or_else(taken)?;
        match crate::engine::metadata::namespace_record(self, id)? {
            Some(record) if record.token == token => Ok(id),
            _ => Err(taken()),
        }
    }

    /// Drop the live namespace `name` from this store's catalog and return
    /// the id it had: from here on nothing reads or writes it, and its trees
    /// are removed. For a store that is not a member of a group.
    ///
    /// # Errors
    ///
    /// [`StorageError::NoSuchNamespace`] when no live namespace has the
    /// name; [`StorageError::NamespaceRefused`] on a store holding a group's
    /// history; a write failure.
    pub fn drop_namespace(&self, name: &str) -> StorageResult<NamespaceId> {
        let id = crate::engine::metadata::namespace_by_name(self, name)?.ok_or_else(|| {
            StorageError::NoSuchNamespace {
                name: name.to_owned(),
            }
        })?;
        self.refuse_local_catalog_change()
            .map_err(|e| StorageError::NamespaceRefused {
                name: name.to_owned(),
                reason: e.to_string(),
            })?;
        self.commit_command(
            coordinode_core::txn::proposal::MetadataCommand::DropNamespace {
                name: name.to_owned(),
            },
        )?;
        Ok(id)
    }

    /// Change the storage settings `change` names of the live namespace
    /// `name`, keeping the others; its trees write with them from their next
    /// flush or compaction. For a store that is not a member of a group.
    ///
    /// # Errors
    ///
    /// [`StorageError::NoSuchNamespace`] when no live namespace has the
    /// name; [`StorageError::NamespaceRefused`] on a store holding a group's
    /// history; a write failure.
    pub fn set_namespace_storage(
        &self,
        name: &str,
        change: coordinode_core::namespace::NamespaceStorage,
    ) -> StorageResult<()> {
        self.namespace_id(name)?;
        self.refuse_local_catalog_change()
            .map_err(|e| StorageError::NamespaceRefused {
                name: name.to_owned(),
                reason: e.to_string(),
            })?;
        self.commit_command(
            coordinode_core::txn::proposal::MetadataCommand::SetNamespaceStorage {
                name: name.to_owned(),
                change,
            },
        )
    }

    /// The storage settings of the live namespace called `name`.
    ///
    /// # Errors
    ///
    /// [`StorageError::NoSuchNamespace`] when none is; the catalog cannot be
    /// read.
    pub fn namespace_storage(
        &self,
        name: &str,
    ) -> StorageResult<coordinode_core::namespace::NamespaceStorage> {
        let id = self.namespace_id(name)?;
        Ok(crate::engine::metadata::namespace_record(self, id)?
            .map(|record| record.storage)
            .unwrap_or_default())
    }

    /// The live namespace called `name`.
    ///
    /// # Errors
    ///
    /// [`StorageError::NoSuchNamespace`] when none is; the catalog cannot be
    /// read.
    pub fn namespace_id(&self, name: &str) -> StorageResult<NamespaceId> {
        crate::engine::metadata::namespace_by_name(self, name)?.ok_or_else(|| {
            StorageError::NoSuchNamespace {
                name: name.to_owned(),
            }
        })
    }

    /// The live namespaces, with their names, in id order.
    ///
    /// # Errors
    ///
    /// The catalog cannot be read.
    pub fn live_namespaces(&self) -> StorageResult<Vec<(NamespaceId, String)>> {
        Ok(crate::engine::metadata::namespaces(self)?
            .into_iter()
            .filter(|(_, record)| record.state == coordinode_core::namespace::NamespaceState::Live)
            .map(|(id, record)| (id, record.name))
            .collect())
    }

    /// This store bound to the live namespace called `name`.
    ///
    /// # Errors
    ///
    /// As [`Self::namespace_id`]; the namespace's trees cannot be opened.
    pub fn engine_named(&self, name: &str) -> StorageResult<StorageEngine> {
        self.engine(self.namespace_id(name)?)
    }

    /// Refuse a catalog change decided here for a store whose order is a
    /// group's log, which the change would bypass.
    fn refuse_local_catalog_change(&self) -> StorageResult<()> {
        if self.group_history || self.raft_fence().is_some() {
            return Err(StorageError::InvalidConfig(
                "this store holds a group's history: its catalog changes through the group's log"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Decide and apply `command` in this store's own order, logged in its
    /// journal when it keeps one.
    fn commit_command(
        &self,
        command: coordinode_core::txn::proposal::MetadataCommand,
    ) -> StorageResult<()> {
        let mutations = [Mutation::Command(command)];
        let (commit_ts, _held) = self.pending_commits.obligate(|| self.next_seqno());
        if self.oplog.is_some() {
            return self.commit_journaled(&mutations, commit_ts);
        }
        self.apply_proposal_at(&mutations, commit_ts)?;
        // No journal keeps it: the catalog's records are made durable before
        // the change is answered, as a namespace outlives any write into it.
        self.flush_memtables(self.system_tree(Partition::Deployment)?)
    }

    /// The set a member-local record of a namespace's tree lands in: the one
    /// namespace of a store holding one, else the system set alone.
    fn local_record_set(&self) -> Arc<SetTrees> {
        match self.coordinator.sets().namespace_ids()[..] {
            [only] => self
                .coordinator
                .sets()
                .namespace(only)
                .unwrap_or_else(|| Arc::clone(self.coordinator.sets().system())),
            _ => Arc::clone(self.coordinator.sets().system()),
        }
    }

    /// See through the catalog changes `effects`, just applied, make: the
    /// set of a namespace they mark dropping is taken out of the engine and
    /// its trees removed once nothing holds them.
    fn note_catalog(&self, effects: &[Mutation]) {
        use coordinode_core::namespace::{
            NamespaceRecord, NamespaceState, decode_namespace_record_key,
        };
        for effect in effects {
            let Mutation::Put {
                partition: coordinode_core::txn::proposal::PartitionId::Deployment,
                key,
                value,
            } = effect
            else {
                continue;
            };
            let Some(id) = decode_namespace_record_key(key) else {
                continue;
            };
            match NamespaceRecord::decode(value) {
                Some(record) if record.state == NamespaceState::Dropping => {
                    self.retire_namespace(id);
                }
                Some(record) => self.apply_namespace_storage(id, record.storage),
                None => {}
            }
        }
    }

    /// Have namespace `id`'s open trees write with `storage` from their next
    /// flush or compaction on, and tables opened later too: a committed
    /// change applies where it lands, without a rewrite. Data already
    /// written keeps the codec its tables recorded until compaction rewrites
    /// it. A namespace not open takes the settings when it opens.
    fn apply_namespace_storage(
        &self,
        id: NamespaceId,
        storage: coordinode_core::namespace::NamespaceStorage,
    ) {
        let Some(set) = self.coordinator.sets().namespace(id) else {
            return;
        };
        set.set_storage(storage);
        let retune = |part: Partition, tree: &lsm_tree::AnyTree| {
            let policy = match self
                .opener
                .config
                .namespace_compression_policy(part, &storage)
            {
                Ok(policy) => policy,
                Err(e) => {
                    tracing::warn!(%id, error = %e, "namespace codecs not applied");
                    return;
                }
            };
            let inner = match tree {
                lsm_tree::AnyTree::Standard(t) => t,
                lsm_tree::AnyTree::Blob(bt) => &bt.index,
            };
            if let Err(e) = inner
                .update_runtime_config(|cfg| cfg.data_block_compression_policy = policy.clone())
            {
                tracing::warn!(%id, error = %e, "namespace codecs not applied to a tree");
            }
        };
        for (tree_id, tree) in set.iter() {
            retune(tree_id.part, tree);
        }
        #[cfg(feature = "columnar")]
        if let Some(registry) = set.columnar() {
            for table_id in registry.table_ids() {
                registry.with_live(&table_id, |tree| retune(Partition::Node, tree));
            }
        }
    }

    /// Take namespace `id`'s set out of the engine and have its trees, on
    /// every endpoint, removed once the last holder lets go of them.
    fn retire_namespace(&self, id: NamespaceId) {
        let dirs = namespace_dirs(&self.endpoints, id);
        let closing = self.namespace_opening.lock().remove(&id);
        match (self.coordinator.sets().remove(id), closing) {
            (Some(set), _) => set.retire_on_drop(dirs, self.footprints.clone()),
            // A namespace closed for being idle: its directories go once the
            // last of its closed trees no longer touches them, without the
            // apply waiting for that.
            (None, closing) => remove_namespace_dirs_once_closed(
                closing.unwrap_or_default(),
                dirs,
                self.footprints.clone(),
            ),
        }
    }

    /// Close the trees of every namespace that nothing holds and nothing
    /// reached since the previous call: a namespace's trees hold memtables,
    /// open files and compaction bookkeeping, which grow with the namespaces
    /// open rather than with those that exist. Each is flushed first and
    /// opens again from disk on its next use, which waits for the closed
    /// trees to let go of their directories. The number of namespaces
    /// closed.
    ///
    /// # Errors
    ///
    /// A flush failure; the namespace it struck stays open.
    pub fn close_idle_namespaces(&self) -> StorageResult<usize> {
        let mut opening = self.namespace_opening.lock();
        let mut closed = 0;
        for namespace in self.coordinator.sets().namespace_ids() {
            let Some(set) = self.coordinator.sets().namespace(namespace) else {
                continue;
            };
            // Held by a handle (a session, a write batch, a monitor pass),
            // or reached since the last pass: in use.
            if set.take_touched() || Arc::strong_count(&set) > 2 {
                continue;
            }
            #[cfg(feature = "columnar")]
            if set.columnar().is_some_and(|r| Arc::strong_count(r) > 1) {
                continue;
            }
            // Out of the map first: from here no use reaches the set but
            // through the open, which waits on this lock.
            let Some(taken) = self.coordinator.sets().remove(namespace) else {
                continue;
            };
            drop(set);
            let set = match Arc::try_unwrap(taken) {
                Ok(set) => set,
                // A handle taken between the look and the removal.
                Err(shared) => {
                    self.coordinator.sets().insert(shared);
                    continue;
                }
            };
            // The trees hold what reached them in memtables only: written
            // to tables before they close, or the next open would lose it.
            let flushed = set
                .iter()
                .try_for_each(|(_, tree)| self.flush_memtables_in_order(tree))
                .and_then(|()| self.flush_tables_of(&set));
            if let Err(e) = flushed {
                self.coordinator.sets().insert(Arc::new(set));
                return Err(e);
            }
            self.sync_endpoint_devices()?;
            opening.entry(namespace).or_default().extend(set.close());
            tracing::debug!(%namespace, "idle namespace closed");
            closed += 1;
        }
        Ok(closed)
    }

    /// Flush the memtables of every table of `set`'s namespace.
    fn flush_tables_of(&self, set: &SetTrees) -> StorageResult<()> {
        #[cfg(feature = "columnar")]
        if let Some(registry) = set.columnar() {
            for table_id in registry.table_ids() {
                registry
                    .with_live(&table_id, |tree| self.flush_memtables_in_order(tree))
                    .unwrap_or(Ok(()))?;
            }
        }
        #[cfg(not(feature = "columnar"))]
        let _ = set;
        Ok(())
    }

    /// Retire every open namespace the catalog no longer holds live: what a
    /// catalog replaced whole (a snapshot of another member installed) leaves
    /// to settle, since no drop command passed through this store for it.
    /// A namespace the new catalog holds and this store has not opened opens
    /// on first use, as any created since the open.
    ///
    /// # Errors
    ///
    /// The catalog cannot be read.
    pub fn adopt_catalog(&self) -> StorageResult<()> {
        let live: std::collections::BTreeSet<NamespaceId> = self
            .live_namespaces()?
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        for namespace in self.coordinator.sets().namespace_ids() {
            if !live.contains(&namespace) {
                self.retire_namespace(namespace);
            }
        }
        Ok(())
    }

    /// The value of `key` in the system set's `part` at `snapshot`.
    ///
    /// # Errors
    ///
    /// As [`Self::system_get`]; the snapshot is no longer retained.
    pub fn system_snapshot_get(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<bytes::Bytes>> {
        let value = self.system_tree(part)?.get(key, *snapshot)?;
        Ok(crate::engine::revision::user_bytes(part, key, value)?)
    }

    /// The records of the system set's `part` under `prefix` now, in key
    /// order.
    ///
    /// # Errors
    ///
    /// `part` has no tree in the system set.
    pub fn system_prefix_scan(&self, part: Partition, prefix: &[u8]) -> StorageResult<StorageIter> {
        Ok(coverage::user_prefix(
            part,
            self.system_tree(part)?,
            prefix,
            self.coordinator.current_seqno(),
        ))
    }

    /// [`Self::system_prefix_scan`] newest key first.
    ///
    /// # Errors
    ///
    /// As [`Self::system_prefix_scan`].
    pub fn system_prefix_scan_rev(
        &self,
        part: Partition,
        prefix: &[u8],
    ) -> StorageResult<StorageIter> {
        Ok(Box::new(
            coverage::user_prefix(
                part,
                self.system_tree(part)?,
                prefix,
                self.coordinator.current_seqno(),
            )
            .rev(),
        ))
    }
}

impl StorageEngine {
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

    /// Read a value by key from the given partition.
    ///
    /// Read path with tiered cache:
    ///   1. Tiered cache (NVMe/SSD layers) → hit → return
    ///   2. LSM storage (DRAM block cache → persistent storage) → populate cache → return
    ///   3. Miss → None
    pub fn get(&self, part: Partition, key: &[u8]) -> StorageResult<Option<bytes::Bytes>> {
        // Check tiered cache first. A miss takes the key's ticket before the
        // tree read, so the value is cached only if no write was invalidated
        // since. Entries are this namespace's tree's: the same key of
        // another namespace is another entry.
        let id = self.tree_id(part);
        let ticket = match &self.tiered_cache {
            Some(cache) => {
                if let Some(value) = cache.get(id, key) {
                    self.access_tracker.record(id, key);
                    return Ok(Some(value));
                }
                Some(cache.ticket(id, key))
            }
            None => None,
        };

        // Fall through to LSM storage.
        let tree = self.tree(part)?;
        let mut buf = Vec::new();
        let Some(stored) = self.stored_key(part, key, &mut buf) else {
            return Ok(None);
        };
        let value = tree.get(stored, self.coordinator.current_seqno())?;

        match crate::engine::revision::user_bytes(part, key, value)? {
            Some(bytes) => {
                if let (Some(cache), Some(ticket)) = (&self.tiered_cache, ticket) {
                    let weight = self.resolve_cache_weight(cache, part, &bytes);
                    cache.fill(id, key, &bytes, weight, ticket);
                }
                self.access_tracker.record(id, key);
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
        self.multi_get_each(part, keys, None, |slot, value: Option<StoredValue<'_>>| {
            out[slot] = value.map(StoredValue::to_bytes);
            Ok::<(), StorageError>(())
        })?;
        Ok(out)
    }

    /// [`Self::multi_get`] handing each key's value to `each` as the store
    /// holds it, instead of copying it out: `each` gets the key's index and
    /// its value or `None`, once per key, in no promised order, and keeps
    /// what it wants. A reader holding to a memory budget charges a value
    /// before it copies or decodes it. A value the tiered cache serves is the
    /// cache's; one read from the tree is a view into its block, copied only
    /// into the cache when the cache takes it. Under `budget`, the lists this
    /// read keeps aside for the keys are charged before they exist; the index
    /// partition's translation of generation keys keeps buffers it does not
    /// charge.
    ///
    /// # Errors
    ///
    /// A storage failure, the budget's refusal, or what `each` returns.
    pub fn multi_get_each<E: From<StorageError> + From<coordinode_core::budget::BudgetStop>>(
        &self,
        part: Partition,
        keys: &[&[u8]],
        budget: Option<&coordinode_core::budget::QueryBudget>,
        mut each: impl FnMut(usize, Option<StoredValue<'_>>) -> Result<(), E>,
    ) -> Result<(), E> {
        let ticket = match &self.tiered_cache {
            Some(_) => core::mem::size_of::<crate::cache::tiered::FillTicket>(),
            None => 0,
        };
        let _scratch = reserve_slots(
            budget,
            keys.len(),
            core::mem::size_of::<usize>() + core::mem::size_of::<&[u8]>() + ticket,
        )?;
        // Split cache hits from misses; only the misses go to the tree, each
        // with its ticket taken before the read (see `get`). Each list holds
        // room for exactly the keys.
        let mut miss_idx: Vec<usize> = Vec::with_capacity(keys.len());
        let mut miss_keys: Vec<&[u8]> = Vec::with_capacity(keys.len());
        let mut tickets = Vec::with_capacity(if ticket == 0 { 0 } else { keys.len() });
        let id = self.tree_id(part);
        for (i, key) in keys.iter().enumerate() {
            if let Some(cache) = &self.tiered_cache {
                if let Some(value) = cache.get(id, key) {
                    self.access_tracker.record(id, key);
                    each(i, Some(StoredValue::Cached(&value)))?;
                    continue;
                }
                tickets.push(cache.ticket(id, key));
            }
            miss_idx.push(i);
            miss_keys.push(key);
        }

        if miss_keys.is_empty() {
            return Ok(());
        }

        let values =
            self.stored_multi_get_stamped(part, &miss_keys, self.coordinator.current_seqno())?;

        for (n, (slot, value)) in miss_idx.into_iter().zip(values).enumerate() {
            let Some(stored) = value else {
                each(slot, None)?;
                continue;
            };
            // A view of the payload inside the tree's buffer: no copy.
            let v = if crate::engine::revision::stamped(part, keys[slot]) {
                crate::engine::revision::payload_of(&stored).map_err(StorageError::from)?
            } else {
                &stored
            };
            self.access_tracker.record(id, keys[slot]);
            match (&self.tiered_cache, tickets.get(n)) {
                (Some(cache), Some(ticket)) => {
                    let bytes = bytes::Bytes::copy_from_slice(v);
                    let weight = self.resolve_cache_weight(cache, part, &bytes);
                    cache.fill(id, keys[slot], &bytes, weight, *ticket);
                    each(slot, Some(StoredValue::Cached(&bytes)))?;
                }
                _ => each(slot, Some(StoredValue::Tree(v)))?,
            }
        }
        Ok(())
    }

    /// Write a key-value pair to the given partition.
    /// Invalidates any cached entry for this key.
    ///
    /// Admitted on the owned footprint first: a write that does not fit
    /// beside what the endpoints hold and owe returns
    /// [`StorageError::CapacityExhausted`] without inserting.
    pub fn put(&self, part: Partition, key: &[u8], value: &[u8]) -> StorageResult<()> {
        let _admitted = self.admit_partition_write(
            [part],
            crate::engine::footprint::footprint_of((key.len() + value.len()) as u64, 1),
        )?;
        let mut addressing = self.addressing(part)?;
        let point = addressing.point(key)?.to_vec();
        let staging = addressing.staging_point(key).map(<[u8]>::to_vec);
        self.apply_single(part, |seqno| {
            let value = crate::engine::revision::stored(part, key, value, seqno);
            let mut batch = lsm_tree::WriteBatch::with_capacity(3);
            batch.insert(point.as_slice(), value.as_ref());
            if let Some(staging) = &staging {
                batch.insert(staging.as_slice(), value.as_ref());
            }
            batch
        })?;
        let id = self.tree_id(part);
        self.write_taps.wrote(id, [key]);
        if let Some(cache) = &self.tiered_cache {
            cache.remove(id, key);
        }
        Ok(())
    }

    /// Move the record at `key` from partition `from` to partition `to` as
    /// it is stored, its revision included: the same write, kept, now read
    /// from `to`. Returns whether there was a record to move.
    ///
    /// # Errors
    ///
    /// A partition has no tree here, or a read or write fails.
    pub fn move_record(&self, from: Partition, to: Partition, key: &[u8]) -> StorageResult<bool> {
        let mut buf = Vec::new();
        let Some(stored) = self.stored_key(from, key, &mut buf) else {
            return Ok(false);
        };
        let Some(value) = self
            .tree(from)?
            .get(stored, self.coordinator.current_seqno())?
        else {
            return Ok(false);
        };
        self.put_stored(to, key, &value)?;
        self.delete(from, key)?;
        Ok(true)
    }

    /// Write back a row read as stored ([`Self::snapshot_prefix_scan_stored`]):
    /// its value already carries its revision, which the write keeps.
    pub(crate) fn put_stored(
        &self,
        part: Partition,
        key: &[u8],
        stored: &[u8],
    ) -> StorageResult<()> {
        let _admitted = self.admit_partition_write(
            [part],
            crate::engine::footprint::footprint_of((key.len() + stored.len()) as u64, 1),
        )?;
        let mut addressing = self.addressing(part)?;
        let mut batch = lsm_tree::WriteBatch::with_capacity(3);
        batch.insert(addressing.point(key)?, stored);
        if let Some(staging) = addressing.staging_point(key) {
            batch.insert(staging, stored);
        }
        self.apply_single(part, |_| batch)?;
        let id = self.tree_id(part);
        self.write_taps.wrote(id, [key]);
        if let Some(cache) = &self.tiered_cache {
            cache.remove(id, key);
        }
        Ok(())
    }

    /// Leave the live record at `key` here with its payload and no known
    /// commit, as a store brought from a release that kept none holds a
    /// record its evidence does not prove; nothing else changes. Whether a
    /// record was there. For tests of stores whose members prove different
    /// commits.
    ///
    /// # Errors
    ///
    /// A storage failure.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn forget_revision(&self, part: Partition, key: &[u8]) -> StorageResult<bool> {
        let Some(payload) = self.get(part, key)? else {
            return Ok(false);
        };
        let unknown =
            crate::engine::revision::stored(part, key, &payload, crate::engine::revision::UNKNOWN);
        self.put_stored(part, key, &unknown)?;
        Ok(true)
    }

    /// Apply one direct write, the batch `build` makes for the seqno it
    /// lands at (its key and a replacement's copy), at the next seqno.
    fn apply_single(
        &self,
        part: Partition,
        build: impl FnOnce(lsm_tree::SeqNo) -> lsm_tree::WriteBatch,
    ) -> StorageResult<()> {
        let tree = self.tree(part)?;
        // Held until the data is in: no complete snapshot may cover its
        // seqno before.
        let (seqno, _landing) = self.pending_commits.admit_direct(|| self.next_seqno());
        let (added, memtable) = tree.apply_batch(build(seqno), seqno)?;
        self.coordinator
            .flush_trigger()
            .wrote(part, added, memtable);
        Ok(())
    }

    /// Delete a key from the given partition.
    ///
    /// Admitted on the owned footprint like a put: a tombstone takes memtable
    /// bytes, then table bytes, until a compaction drops what it covers.
    pub fn delete(&self, part: Partition, key: &[u8]) -> StorageResult<()> {
        let _admitted = self.admit_partition_write(
            [part],
            crate::engine::footprint::footprint_of(key.len() as u64, 1),
        )?;
        let mut buf = Vec::new();
        // A generation with no installation here holds nothing to delete.
        if let Some(stored) = self.stored_key(part, key, &mut buf) {
            let mut batch = lsm_tree::WriteBatch::with_capacity(3);
            batch.remove(stored);
            let mut addressing = self.addressing(part)?;
            if let Some(staging) = addressing.staging_point(key) {
                batch.remove(staging);
            }
            self.apply_single(part, |_| batch)?;
        }
        let id = self.tree_id(part);
        self.write_taps.wrote(id, [key]);
        if let Some(cache) = &self.tiered_cache {
            cache.remove(id, key);
        }
        Ok(())
    }

    /// Delete the half-open range `[start, end)` of a partition with one MVCC
    /// range tombstone. Snapshot-aware and seqno'd, so it replicates and
    /// PITR-replays correctly; a compaction above the tombstone's seqno
    /// reclaims the tables it spans whole. Invalidates the partition's cache:
    /// the tombstone leaves the shadowed keys physically present, so a stale
    /// cache hit would otherwise return a deleted value, and the cache cannot
    /// range-query to invalidate precisely.
    ///
    /// Not capacity-gated — deleting frees space.
    pub fn remove_range(&self, part: Partition, start: &[u8], end: &[u8]) -> StorageResult<()> {
        // Every piece at one seqno: a snapshot sees the whole delete or none
        // of it, however many installations it spans.
        let addressing = self.addressing(part)?;
        let tree = self.tree(part)?;
        // Held across the tombstones: separate memtable writes, which no
        // complete snapshot may see in part.
        let (seqno, _landing) = self.pending_commits.admit_direct(|| self.next_seqno());
        let start = coverage::clamp_user_start(start);
        let ranges = addressing.range(start, end);
        let staging = addressing.staging_range(start, end);
        for (start, end) in ranges.into_iter().chain(staging) {
            if start < end {
                tree.remove_range(start, end, seqno);
            }
        }
        self.coordinator.flush_trigger().wrote_unmeasured();
        drop(_landing);
        let id = self.tree_id(part);
        self.write_taps.replaced(id);
        if let Some(cache) = &self.tiered_cache {
            cache.clear_partition(id);
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
        let _admitted = self.admit_partition_write(
            [part],
            crate::engine::footprint::footprint_of((key.len() + operand.len()) as u64, 1),
        )?;
        let mut addressing = self.addressing(part)?;
        let point = addressing.point(key)?.to_vec();
        let staging = addressing.staging_point(key).map(<[u8]>::to_vec);
        self.apply_single(part, |seqno| {
            let operand = crate::engine::revision::stored(part, key, operand, seqno);
            let mut batch = lsm_tree::WriteBatch::with_capacity(3);
            batch.merge(point.as_slice(), operand.as_ref());
            if let Some(staging) = &staging {
                batch.merge(staging.as_slice(), operand.as_ref());
            }
            batch
        })?;
        let id = self.tree_id(part);
        self.write_taps.wrote(id, [key]);
        if let Some(cache) = &self.tiered_cache {
            cache.remove(id, key);
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
            // Each is resolved by the unit it belongs to: one of this
            // namespace for a namespace's work.
            Mutation::Command(_) | Mutation::Derive(_) | Mutation::Operation(_) => {
                if mutation.requires_namespace() {
                    self.apply_proposal_at(
                        &coordinode_core::txn::proposal::in_namespace(
                            self.namespace(),
                            [mutation.clone()],
                        ),
                        0,
                    )
                } else {
                    self.apply_proposal_at(std::slice::from_ref(mutation), 0)
                }
            }
            // Write nothing; an acting principal alone has no write to
            // authorize.
            Mutation::Effects(_) | Mutation::Namespace(_) | Mutation::Actor(_) => Ok(()),
        }
    }
}

impl Deployment {
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
        commit_ts: u64,
        actor: Option<&coordinode_core::access::PrincipalId>,
    ) -> StorageResult<(Vec<Mutation>, DictionaryChange)> {
        use coordinode_core::txn::proposal::MetadataCommand;
        let mut effects = Vec::with_capacity(mutations.len() + 1);
        let mut decided = false;
        let mut dictionary = DictionaryChange::None;
        // The namespace the rows and the command after the last marker
        // belong to.
        let mut namespace = None;
        for (at, mutation) in mutations.iter().enumerate() {
            let command = match mutation {
                Mutation::Command(command) => command,
                Mutation::Namespace(named) => {
                    namespace = Some(*named);
                    effects.push(mutation.clone());
                    continue;
                }
                _ => {
                    effects.push(mutation.clone());
                    continue;
                }
            };
            if decided {
                tracing::warn!(
                    ?command,
                    "a second metadata command in one proposal is refused"
                );
                continue;
            }
            decided = true;
            // A namespace's command decides against that namespace's records;
            // one of the deployment's against the deployment's alone.
            let decided_effects = if mutation.requires_namespace() {
                let namespace = namespace.ok_or(coordinode_core::txn::proposal::Unscoped { at })?;
                let scope = self.engine(namespace)?;
                crate::engine::metadata::decide(self, Some(&scope), command, commit_ts, actor)?
            } else {
                crate::engine::metadata::decide(self, None, command, commit_ts, actor)?
            };
            if !decided_effects.is_empty() {
                dictionary = match command {
                    MetadataCommand::RegisterFields { .. } => DictionaryChange::Extended,
                    MetadataCommand::AdoptFields { .. } => DictionaryChange::Replaced,
                    MetadataCommand::BindLabels { .. }
                    | MetadataCommand::GrantNodeLease { .. }
                    | MetadataCommand::RecordGroupPair { .. }
                    | MetadataCommand::RecordDeployment { .. }
                    | MetadataCommand::ConditionalRegistryWrite { .. }
                    | MetadataCommand::CreateNamespace { .. }
                    | MetadataCommand::DropNamespace { .. }
                    | MetadataCommand::SetNamespaceStorage { .. }
                    | MetadataCommand::GrantAccess { .. }
                    | MetadataCommand::RevokeAccess { .. }
                    | MetadataCommand::RevokeGrant { .. }
                    | MetadataCommand::TransferAccess { .. }
                    | MetadataCommand::SetDeploymentRights { .. }
                    | MetadataCommand::CreateRole { .. }
                    | MetadataCommand::DropRole { .. }
                    | MetadataCommand::SetRoleEligible { .. }
                    | MetadataCommand::GrantRole { .. }
                    | MetadataCommand::RevokeRole { .. }
                    | MetadataCommand::TransferRole { .. } => DictionaryChange::None,
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
        // The check and the apply hold off grant changes together.
        let _access = self.access_guard(mutations);
        let (actor, mutations) = self.authorized(mutations)?;
        if let Some((command, rest)) = crate::engine::operations::unit_command(mutations)? {
            // The decision and the apply hold the issuer's lock together, so
            // the next decision about the operation reads this one's record.
            let _decision = self.operation_decisions.lock(&command.issuer());
            let decided = crate::engine::operations::decide(
                self,
                command,
                rest,
                commit_ts,
                coordinode_core::txn::operation::LogPosition::LOCAL,
                actor,
            )?;
            if let Some(refusal) = decided.refused {
                return Err(StorageError::AccessRefused(refusal));
            }
            if decided.effects.is_empty() {
                return Ok(());
            }
            let staged = self.applied_feed.stage(0, commit_ts, &decided.effects);
            self.apply_proposal_covered(&decided.effects, commit_ts, None, actor)?;
            staged.publish();
            return Ok(());
        }
        // A local commit has no log position; its subscribers follow it by
        // commit timestamp. Staged before the write lands, so a reader that
        // sees it is told the derived indexes may not hold it yet.
        let staged = self.applied_feed.stage(0, commit_ts, mutations);
        self.apply_proposal_covered(mutations, commit_ts, None, actor)?;
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
        self.commit_logged(mutations, commit_ts, Journal::Journal)
    }

    /// [`Self::commit_journaled`] with the journal entry taken only as far as
    /// `level` before the commit returns: a volatile level leaves its sync to
    /// the journal's flusher, a journaled commit or a table flush, whichever
    /// comes first. The commit is applied and visible either way.
    ///
    /// # Errors
    ///
    /// As [`Self::commit_journaled`].
    pub fn commit_logged(
        &self,
        mutations: &[Mutation],
        commit_ts: u64,
        level: Journal,
    ) -> StorageResult<()> {
        // Refused before the journal append, which is the write a full disk
        // would fail in; the admission is held until the write is in the
        // journal and the memtables.
        let _admitted = self.admit_write(mutations)?;
        // The check, the journal append and the apply hold off grant changes
        // together; a refused unit journals nothing, so a replay has nothing
        // of it to decide again.
        let _access = self.access_guard(mutations);
        let (actor, mutations) = self.authorized(mutations)?;
        if let Some((command, rest)) = crate::engine::operations::unit_command(mutations)? {
            // The decision, the journal append and the apply hold the
            // issuer's lock together: the journal records each decision's
            // effects in the order they were decided, and replay reproduces
            // them without deciding again.
            let _decision = self.operation_decisions.lock(&command.issuer());
            let decided = crate::engine::operations::decide(
                self,
                command,
                rest,
                commit_ts,
                coordinode_core::txn::operation::LogPosition::LOCAL,
                actor,
            )?;
            // Refused for its issuer's ownership: nothing is journaled.
            if let Some(refusal) = decided.refused {
                return Err(StorageError::AccessRefused(refusal));
            }
            if decided.effects.is_empty() {
                // The caller is answered from the record this decision read,
                // which a volatile write may have left unsynced.
                return self.sync_for(level);
            }
            return self.journal_and_apply(&decided.effects, commit_ts, level);
        }
        if mutations.iter().any(|m| matches!(m, Mutation::Command(_))) {
            // The decision, the journal append and the apply happen under one
            // lock: the journal then records each decision's effects in the
            // order they were decided, and replay reproduces them exactly.
            let _decisions = self.metadata_exclusive();
            let (effects, dictionary) = self.decide_commands(mutations, commit_ts, actor)?;
            // Nothing decided (names already bound, or a refusal) leaves the
            // state as it is, so there is nothing to journal or replay; the
            // bindings it found may still be unsynced. The namespace markers
            // the unit carried change nothing on their own.
            if effects.iter().all(|m| matches!(m, Mutation::Namespace(_))) {
                return self.sync_for(level);
            }
            self.journal_and_apply(&effects, commit_ts, level)?;
            self.note_dictionary(dictionary);
            self.note_catalog(&effects);
            return Ok(());
        }
        self.journal_and_apply(mutations, commit_ts, level)
    }

    /// The hold on grant changes a unit takes for its check and apply: the
    /// exclusive one for a unit that changes the grant catalog or the
    /// namespaces it grants on, the shared one for a unit acting for a
    /// principal, none otherwise.
    pub(crate) fn access_guard(&self, mutations: &[Mutation]) -> AccessHold<'_> {
        if mutations
            .iter()
            .any(|m| matches!(m, Mutation::Command(c) if c.changes_access()))
        {
            AccessHold::Exclusive(self.access_decisions.write())
        } else if coordinode_core::txn::proposal::actor_of(mutations).is_some() {
            AccessHold::Shared(self.access_decisions.read())
        } else {
            AccessHold::None
        }
    }

    /// `mutations` as they apply once their principal's grants are checked,
    /// with the principal they act for: without it, or as they are when they
    /// name none.
    ///
    /// # Errors
    ///
    /// [`StorageError::AccessRefused`] when the principal lacks a right the
    /// unit needs; a catalog read failure.
    fn authorized<'m>(
        &self,
        mutations: &'m [Mutation],
    ) -> StorageResult<(
        Option<&'m coordinode_core::access::PrincipalId>,
        &'m [Mutation],
    )> {
        crate::engine::access::check_unit(self, mutations)?
            .applying()
            .map_err(StorageError::AccessRefused)
    }

    /// Make the journal as durable as `level` promises for a commit that
    /// appended nothing: under `journal`, every entry it may have read from.
    fn sync_for(&self, level: Journal) -> StorageResult<()> {
        match (&self.oplog, level) {
            (Some(oplog), Journal::Journal) => crate::engine::oplog_journal::sync_journal(oplog),
            _ => Ok(()),
        }
    }

    /// Write node-local recovery records (index use guards and marks): never
    /// replicated, logged in this node's journal and applied at once. Under
    /// [`LocalSync::BeforeUse`] the call returns once they are durable, for a
    /// record that must survive a crash before what it guards happens; under
    /// [`LocalSync::Deferred`] the next sync covers them, for a record whose
    /// loss recovery repairs (a mark not saved leaves its guard armed, a
    /// qualification not saved is verified again). A journal append replaces
    /// the whole-tree flush such a record used to cost.
    ///
    /// # Errors
    ///
    /// The journal cannot take or sync the entry, or a write names a
    /// partition the journal cannot carry.
    pub fn commit_local(&self, writes: Vec<LocalWrite>, sync: LocalSync) -> StorageResult<()> {
        self.commit_local_in(None, writes, sync)
    }

    /// [`Self::commit_local`] of records of `namespace`'s trees beside the
    /// deployment's own.
    ///
    /// # Errors
    ///
    /// As [`Self::commit_local`]; a namespace's record without `namespace`.
    fn commit_local_in(
        &self,
        namespace: Option<NamespaceId>,
        writes: Vec<LocalWrite>,
        sync: LocalSync,
    ) -> StorageResult<()> {
        if writes.is_empty() {
            return Ok(());
        }
        // The consensus state (vote, commit and purge points) is admitted
        // neither against free space nor against the footprint: a refusal is
        // a storage error the consensus engine stops the group on, and the
        // records are a few fixed keys rewritten in place, kept for entries
        // already admitted; the journal growth they cause is still held to
        // the footprint's limit.
        let consensus = writes.iter().all(|write| match write {
            LocalWrite::Put { partition, .. } | LocalWrite::Delete { partition, .. } => {
                *partition == Partition::Raft
            }
        });
        // Any other record goes past the admission of client writes, but not
        // past the part of the reserve that retiring it needs.
        if !consensus {
            self.space.admit_free_space_recovery()?;
        }
        // Inside the footprint's recovery reserve, never past its limit.
        let _admitted = if self.footprints.is_empty() || consensus {
            crate::engine::footprint::Admitted::default()
        } else {
            let data = writes
                .iter()
                .map(|write| match write {
                    LocalWrite::Put { key, value, .. } => (key.len() + value.len()) as u64,
                    LocalWrite::Delete { key, .. } => key.len() as u64,
                })
                .sum();
            self.admit_footprint(
                crate::engine::footprint::footprint_of(data, writes.len()),
                crate::engine::footprint::Admit::Recovery,
            )?
        };
        if self.oplog.is_none() {
            return self.commit_member_local(namespace, writes, sync);
        }
        let mutations = writes
            .into_iter()
            .map(LocalWrite::into_mutation)
            .collect::<StorageResult<Vec<_>>>()?;
        // A record of a namespace's tree (an index guard) names the namespace
        // first; the consensus state is the deployment's and names none.
        let mutations = match mutations.iter().position(Mutation::requires_namespace) {
            None => mutations,
            Some(at) => coordinode_core::txn::proposal::in_namespace(
                namespace.ok_or(coordinode_core::txn::proposal::Unscoped { at })?,
                mutations,
            ),
        };
        let level = match sync {
            LocalSync::BeforeUse => Journal::Journal,
            LocalSync::Deferred => Journal::Cache,
        };
        let (commit_ts, _held) = self.pending_commits.obligate(|| self.next_seqno());
        // Straight to the journal, past the admission of client writes:
        // recovery records may use the space reserve a full disk leaves.
        self.journal_and_apply_as(&mutations, commit_ts, level, EntryScope::NodeLocal)
    }

    /// [`Self::commit_local`] on a store whose log is a consensus log: the
    /// records go to the member-local journal, one append (synced under
    /// [`LocalSync::BeforeUse`]) in place of a flush of every tree they
    /// touch. A store kept in memory has no journal and nothing to recover
    /// from, so its records are only written.
    fn commit_member_local(
        &self,
        namespace: Option<NamespaceId>,
        writes: Vec<LocalWrite>,
        sync: LocalSync,
    ) -> StorageResult<()> {
        // The trees the records land in: the namespace's and the system
        // set's, or the system set's alone for the deployment's records.
        let set = match namespace {
            Some(namespace) => self.namespace_set(namespace)?,
            None => Arc::clone(self.coordinator.sets().system()),
        };
        let Some(local) = &self.local_journal else {
            let engine = match namespace {
                Some(namespace) => Some(self.engine(namespace)?),
                None => None,
            };
            for write in writes {
                let (partition, key) = match &write {
                    LocalWrite::Put { partition, key, .. }
                    | LocalWrite::Delete { partition, key } => (*partition, key),
                };
                match (&write, partition.is_deployment(), &engine) {
                    (LocalWrite::Put { value, .. }, true, _) => {
                        self.system_put(partition, key, value)?;
                    }
                    (LocalWrite::Put { value, .. }, false, Some(engine)) => {
                        engine.put(partition, key, value)?;
                    }
                    (LocalWrite::Delete { .. }, false, Some(engine)) => {
                        engine.delete(partition, key)?;
                    }
                    (LocalWrite::Delete { .. }, true, _) => {
                        self.system_delete(partition, key)?;
                    }
                    (_, false, None) => {
                        return Err(StorageError::InvalidConfig(format!(
                            "a record of the {} partition names no namespace",
                            partition.name()
                        )));
                    }
                }
            }
            return Ok(());
        };
        let trees = self.coordinator.partition_trees(&set);
        let compact_due = {
            // The record stays on this member: nothing reaches the log for a
            // closing entry to cover once it is written.
            let (seqno, _held) = self.pending_commits.obligate_local(|| self.next_seqno());
            local.commit(
                crate::engine::local_journal::Targets {
                    trees: &trees,
                    installations: set.installations(),
                },
                writes,
                sync,
                seqno,
                || self.next_seqno(),
            )?
        };
        // Written past the coordinator, so the flush monitor is told.
        self.coordinator.flush_trigger().wrote_unmeasured();
        if compact_due {
            local.compact(&trees, self.next_seqno(), |tree| self.flush_memtables(tree))?;
        }
        Ok(())
    }

    /// Make every write applied here durable before an answer that promises
    /// it, such as a final operation outcome, without writing anything. A
    /// store whose log is a consensus log applies only what is durable, so
    /// for it this does nothing.
    ///
    /// # Errors
    ///
    /// The journal's sync fails, now or earlier.
    pub fn sync_applied(&self) -> StorageResult<()> {
        self.sync_for(Journal::Journal)
    }

    /// Journal `mutations` as far as `level` asks, then apply them at
    /// `commit_ts` with their coverage marker. None of them is a metadata
    /// command.
    fn journal_and_apply(
        &self,
        mutations: &[Mutation],
        commit_ts: u64,
        level: Journal,
    ) -> StorageResult<()> {
        self.journal_and_apply_as(mutations, commit_ts, level, EntryScope::Shared)
    }

    /// [`Self::journal_and_apply`] for an entry of `scope`: a node-local
    /// entry reaches the applied feed without its writes, so no subscriber
    /// derives anything from this node's recovery records.
    fn journal_and_apply_as(
        &self,
        mutations: &[Mutation],
        commit_ts: u64,
        level: Journal,
        scope: EntryScope,
    ) -> StorageResult<()> {
        let (Some(oplog), Some(coverage)) = (&self.oplog, &self.coverage) else {
            return Err(StorageError::InvalidConfig(
                "commit_journaled on an engine without a retained journal".into(),
            ));
        };
        let index = {
            let mut guard = oplog
                .lock()
                .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
            guard.append(mutations, commit_ts, level)?
        };
        #[cfg(test)]
        {
            let hook = self.on_journalled.lock().clone();
            if let Some(hook) = hook {
                hook(index);
            }
        }
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
        let published: &[Mutation] = match scope {
            EntryScope::Shared => mutations,
            EntryScope::NodeLocal => &[],
        };
        let staged = self.applied_feed.stage(index, commit_ts, published);
        // The trees the entry reaches: a namespace's rows the tree of the
        // namespace their last marker names.
        let mut namespace = None;
        for mutation in resolved.iter() {
            if let Mutation::Namespace(named) = mutation {
                namespace = Some(*named);
                continue;
            }
            if let Some(partition) = mutation_partition(mutation) {
                let tree = match namespace {
                    Some(ns) if !partition.is_deployment() => TreeId::namespace(ns, partition),
                    _ => TreeId::system(partition),
                };
                coverage.note_touched(tree, index);
            }
        }
        self.apply_effects_covered(&resolved, commit_ts, Some(mark))?;
        self.note_applied(coverage, index);
        staged.publish();
        Ok(())
    }

    /// Record journal entry `index` as applied everywhere it lands, folding
    /// the accumulated markers when enough have built up. The fold runs on
    /// the committing thread unless another one is already folding.
    fn note_applied(&self, coverage: &Coverage, index: u64) {
        if coverage.mark_applied(index)
            && let Some(mut folded) = coverage.try_lock_folded()
        {
            self.fold_coverage(coverage, &mut folded, coverage::FoldScope::Touched);
        }
    }

    /// Fold every applied index below the applied prefix into the coverage
    /// base of the partition trees `scope` names, removing the markers it
    /// replaces. Returns the base now written; under [`coverage::FoldScope::All`]
    /// every tree carries it.
    fn fold_coverage(
        &self,
        coverage: &Coverage,
        folded: &mut coverage::FoldGuard<'_>,
        scope: coverage::FoldScope,
    ) -> u64 {
        let next = coverage.applied_prefix();
        let from = folded.folded();
        if next <= from && scope == coverage::FoldScope::Touched {
            return from;
        }
        // Above every marker below `next`: each of those entries drew its
        // commit_ts from this generator before it was applied.
        let at = self.next_seqno();
        for (id, tree) in &self.coordinator.trees() {
            let id = *id;
            let based = coverage.based(id);
            // A tree no entry since the last fold reached holds no marker
            // there; skipping it leaves its old base, which a later fold of
            // every tree (before a purge or a checkpoint) brings up. Replay
            // stays exact either way: a marker or a base covers each entry
            // that reached the tree.
            let due = match scope {
                coverage::FoldScope::Touched => coverage.touched_since(id, from),
                coverage::FoldScope::All => true,
            };
            if !due || based >= next {
                continue;
            }
            coverage::write_fold(tree, Domain::Journal, based, next, &[], at);
            coverage.note_based(id, next);
        }
        // A fold can land in a memtable a rotation just emptied.
        self.coordinator.flush_trigger().wrote_unmeasured();
        #[cfg(feature = "columnar")]
        for ((namespace, table_id), markers) in coverage.take_table_markers_below(next) {
            // A dropped table, or a dropped namespace, took its markers with
            // it.
            if let Some(tree) = self
                .coordinator
                .sets()
                .namespace(namespace)
                .and_then(|set| set.columnar()?.get(&table_id))
            {
                coverage::write_table_fold(&tree, Domain::Journal, next, &markers, at);
            }
        }
        folded.set_folded(next);
        next
    }

    /// `actor` is the principal the unit acts for, its grants already
    /// checked: a command it carries is decided on its behalf.
    fn apply_proposal_covered(
        &self,
        mutations: &[Mutation],
        commit_ts: u64,
        cover: Option<coverage::Mark>,
        actor: Option<&coordinode_core::access::PrincipalId>,
    ) -> StorageResult<()> {
        // DERIVED entries are derived from the unit itself and land in the
        // same batch, under the same coverage marker, as the data they index.
        let mutations =
            coordinode_core::index::derive::resolve_unit(mutations, MAX_DERIVED_EFFECTS)?;
        let mutations = mutations.as_ref();
        if mutations.iter().any(|m| matches!(m, Mutation::Command(_))) {
            let _decisions = self.metadata_exclusive();
            let (effects, dictionary) = self.decide_commands(mutations, commit_ts, actor)?;
            self.apply_effects_covered(&effects, commit_ts, cover)?;
            self.note_dictionary(dictionary);
            self.note_catalog(&effects);
            return Ok(());
        }
        self.apply_effects_covered(mutations, commit_ts, cover)
    }

    /// Apply `mutations`, none of them a metadata command, as one batch at
    /// `commit_ts` together with `cover`.
    ///
    /// The rows of a namespace the catalog does not hold live (dropped while
    /// the unit was on its way) are left out, and the rest applies: nothing
    /// of that namespace is kept to observe them by, and a replay of the
    /// unit leaves out the same rows.
    fn apply_effects_covered(
        &self,
        mutations: &[Mutation],
        commit_ts: u64,
        cover: Option<coverage::Mark>,
    ) -> StorageResult<()> {
        if mutations.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::for_deployment(self, mutations.len());
        let mut named = false;
        // The rows after the last marker are of a namespace no longer live.
        let mut gone = false;
        for (at, mutation) in mutations.iter().enumerate() {
            if mutation.requires_namespace() {
                if !named {
                    return Err(coordinode_core::txn::proposal::Unscoped { at }.into());
                }
                if gone {
                    continue;
                }
            }
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
                Mutation::Operation(_) => {
                    return Err(StorageError::InvalidConfig(
                        "an operation command reached the write batch undecided".into(),
                    ));
                }
                // The unit's grants are checked, and the principal left out,
                // before anything is written: a principal here was never
                // checked, and its writes are refused rather than written.
                Mutation::Actor(_) => {
                    return Err(StorageError::InvalidConfig(
                        "an acting principal reached the write batch unauthorized".into(),
                    ));
                }
                // What the unit's rows mean for the change feed; it writes
                // nothing of its own.
                Mutation::Effects(_) => {}
                // The rows after it are of this namespace.
                Mutation::Namespace(namespace) => {
                    named = true;
                    gone = match self.namespace_set(*namespace) {
                        Ok(set) => {
                            batch.write_to(set)?;
                            false
                        }
                        Err(StorageError::UnknownNamespace { .. }) => {
                            tracing::debug!(
                                %namespace,
                                "rows of a namespace no longer live are left out"
                            );
                            true
                        }
                        Err(e) => return Err(e),
                    };
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
}

impl StorageEngine {
    /// The most bytes one scan of `part` opened now holds besides the records
    /// it hands out, when none of its records is longer than
    /// `longest_record` key and value bytes. A caller charging its memory
    /// reserves it before it opens the scan and keeps it until the scan is
    /// dropped; the tree's runs are read now, so it is taken again for each
    /// scan.
    ///
    /// # Errors
    ///
    /// The partition's tree is not open.
    pub fn scan_hold_bytes(&self, part: Partition, longest_record: u64) -> StorageResult<u64> {
        let (fixed, per_record_byte) = scan_hold_terms(self.tree(part)?);
        // Runs are bounded by the tree's levels, a record by the engine's
        // key and value limits: far below u64::MAX.
        Ok(fixed + per_record_byte * longest_record)
    }

    /// For a scan of `part` given `room` bytes in all: what the scan holds
    /// and the longest record it can then still hand over a copy of, the
    /// two together within `room`. `None` when `room` does not cover the
    /// scan with a record of one byte.
    ///
    /// # Errors
    ///
    /// The partition's tree is not open.
    pub fn scan_room(&self, part: Partition, room: u64) -> StorageResult<Option<(u64, u64)>> {
        let (fixed, per_record_byte) = scan_hold_terms(self.tree(part)?);
        // The scan holds `per_record_byte` bytes per byte of the record, and
        // the copy handed over one more.
        let longest = room
            .checked_sub(fixed)
            .map_or(0, |left| left / (per_record_byte + 1));
        Ok((longest > 0).then(|| (fixed + per_record_byte * longest, longest)))
    }

    /// Physically reset a partition to empty: install a fresh empty version and
    /// delete every obsolete table / blob file WITHOUT reading their blocks.
    ///
    /// Unlike a [`remove_range`](Self::remove_range) over the full range, whose
    /// tables a compaction later reclaims, `clear` never touches block
    /// contents and leaves no tombstone. That distinction is load-bearing for repair: when
    /// a partition is being rebuilt because one of its SSTs is corrupt, a
    /// compaction would re-read (and trip on) the very block we are discarding.
    /// Use this before reinstalling a known-good base.
    pub fn clear_partition(&self, part: Partition) -> StorageResult<()> {
        let tree = self.tree(part)?;
        // A repair of its own: refused while another one runs, and ready
        // again only when the clear succeeded. Under a durable intent, so a
        // clear that failed part way, or a crash in it, leaves the partition
        // owed a repair, also after a reopen.
        self.clear_partition_with(part, tree, || Ok(()))
    }

    /// [`Self::clear_partition`] running `cleared` right after the tree is
    /// cleared, before the repair finishes: where a test fails it there.
    pub(crate) fn clear_partition_with(
        &self,
        part: Partition,
        tree: &lsm_tree::AnyTree,
        cleared: impl FnOnce() -> StorageResult<()>,
    ) -> StorageResult<()> {
        // A Schema tree's own records (the deployment's routes and member
        // records, a namespace's records of this node about its indexes) are
        // the store's, not the partition's data: read under the repair, kept
        // in its intent, and put back after the clear.
        let id = self.tree_id(part);
        let repair = self.enter_repair(id)?;
        let kept = if matches!(part, Partition::Schema | Partition::Deployment) {
            self.node_local_rows(part)?
        } else {
            Rows::new()
        };
        self.record_kept(id, &kept)?;
        self.clear_partition_tree(part, tree)?;
        cleared()?;
        for (key, value) in &kept {
            self.put_stored(part, key, value)?;
        }
        self.finish_rebuild(repair)
    }

    /// The clear a rebuild holding the partition's repair makes of its tree.
    pub(crate) fn clear_partition_tree(
        &self,
        part: Partition,
        tree: &lsm_tree::AnyTree,
    ) -> StorageResult<()> {
        tree.clear()?;
        // The installations of the index tree outlive its entries: one handed
        // out before the clear is never handed out again.
        if let Some(installations) = self.installations_of(part) {
            installations.rewrite(tree, self.next_seqno())?;
        }
        let id = self.tree_id(part);
        self.write_taps.replaced(id);
        if let Some(cache) = &self.tiered_cache {
            cache.clear_partition(id);
        }
        Ok(())
    }

    /// Count one more property delta written to `node_key` and return the
    /// run it makes since the key's last whole write.
    pub fn note_node_delta(&self, node_key: &[u8]) -> u32 {
        // A hint only: an engine bound to no namespace writes no node.
        self.set().map_or(0, |set| set.note_node_delta(node_key))
    }

    /// The property deltas written to `node_key` since its last whole write.
    pub fn node_delta_run(&self, node_key: &[u8]) -> u32 {
        self.set().map_or(0, |set| set.node_delta_run(node_key))
    }

    /// `node_key` was written whole: its run of deltas is over.
    pub fn note_node_whole_write(&self, node_key: &[u8]) {
        if let Ok(set) = self.set() {
            set.note_node_whole_write(node_key);
        }
    }

    /// Check if a key exists in the given partition.
    pub fn contains_key(&self, part: Partition, key: &[u8]) -> StorageResult<bool> {
        let tree = self.tree(part)?;
        let mut buf = Vec::new();
        let Some(stored) = self.stored_key(part, key, &mut buf) else {
            return Ok(false);
        };
        let value = tree.get(stored, self.coordinator.current_seqno())?;
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
        // Under a hard limit the compaction's output takes only the room
        // left beside what is held and promised: the journal the tables
        // already cover goes first, so a full store can still compact.
        if !self.footprints.is_empty() {
            self.reclaim_footprint();
        }
        // Target table size of 64 MiB is the lsm-tree default — picked
        // here explicitly so the major compaction's output SST size is
        // independent of any per-partition tuning we may layer on later.
        //
        // `seqno_threshold` is the GC watermark: everything BELOW it may be
        // folded, collected and have its seqno zeroed. Passing `SeqNo::MAX`
        // means the opposite of "keep everything": it collects every
        // superseded version whatever the retention policy says, zeroes the
        // bottommost seqnos, and records a retention floor at the
        // compaction's install seqno that refuses every snapshot at or below
        // it after a reopen.
        //
        // What "as much as is legitimate" means depends on whether the engine
        // owes anyone a time-travel contract; see
        // `GcWatermarkController::maintenance_compaction_threshold`.
        let threshold = self.coordinator.maintenance_compaction_threshold_for(part);
        let _maintenance = crate::engine::footprint::MaintenanceScope::enter();
        tree.major_compact(64 * 1024 * 1024, threshold)
            .map_err(|e| StorageError::Io(format!("major compact {}: {e}", part.name())))?;
        Ok(())
    }

    /// [`persist`](Deployment::persist) for one partition: every write to
    /// `part` made before the call is durable when it returns. For state
    /// that has no journal behind it, such as a Raft vote.
    ///
    /// # Errors
    ///
    /// The partition does not exist or its flush fails.
    pub fn persist_partition(&self, part: Partition) -> StorageResult<()> {
        self.flush_memtables(self.tree(part)?)
    }
}

impl Deployment {
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
            self.coordinator.sets(),
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
        for (_, tree) in self.coordinator.trees() {
            self.flush_memtables_in_order(&tree)?;
        }
        // A table dropped meanwhile has nothing left to persist.
        #[cfg(feature = "columnar")]
        for set in self.coordinator.sets().namespace_sets() {
            let Some(registry) = set.columnar() else {
                continue;
            };
            for table_id in registry.table_ids() {
                registry
                    .with_live(&table_id, |tree| self.flush_memtables_in_order(tree))
                    .unwrap_or(Ok(()))?;
            }
        }
        // Once per device, not once per tree: every tree lies under the
        // endpoints.
        self.sync_endpoint_devices()?;
        // The retained oplog journal is NOT truncated on flush — it must survive
        // for WAL-replay-repair, and crash recovery skips the entries each
        // partition's coverage record already holds (see `open_embedded`).
        Ok(())
    }

    /// Flush every partition tree's memtables, durably: the flushes in
    /// order, then each endpoint's device caches once.
    ///
    /// # Errors
    ///
    /// A flush or device flush failure.
    pub(crate) fn flush_partition_trees(&self) -> StorageResult<()> {
        for (_, tree) in self.coordinator.trees() {
            self.flush_memtables_in_order(&tree)?;
        }
        self.sync_endpoint_devices()
    }

    /// Make durable what the trees synced only in order: flush the device
    /// caches under every endpoint, once each. Nothing to do unless the trees
    /// sync with barriers; under a full sync each flush is durable already.
    fn sync_endpoint_devices(&self) -> StorageResult<()> {
        if self.tree_sync != lsm_tree::fs::SyncMode::Barrier {
            return Ok(());
        }
        for endpoint in &self.endpoints {
            self.fs.sync_device(&endpoint.path).map_err(|e| {
                StorageError::Io(format!(
                    "flush the device caches under {:?}: {e}",
                    endpoint.path
                ))
            })?;
        }
        Ok(())
    }

    /// Write `tree`'s memtables to tables, the journal synced first: the
    /// memtables are sealed, then every journal entry they can hold (each
    /// was appended before it was applied) is made durable, and only then
    /// are the tables written. A table never holds an effect of an entry a
    /// crash could still take from the journal, so a crash loses whole
    /// entries from the end of the log and never part of one.
    ///
    /// The tables are durable when this returns, on every device the tree
    /// lies on: what releases a journal or promises a write relies on it.
    pub(crate) fn flush_memtables<T: AbstractTree + ?Sized>(&self, tree: &T) -> StorageResult<()> {
        self.flush_memtables_in_order(tree)?;
        if self.tree_sync == lsm_tree::fs::SyncMode::Barrier {
            tree.sync_devices()?;
        }
        Ok(())
    }

    /// [`Self::flush_memtables`] without the device flush: the tables are
    /// synced in the trees' mode, durable under a full sync, ordered under a
    /// barrier until [`Self::sync_endpoint_devices`].
    fn flush_memtables_in_order<T: AbstractTree + ?Sized>(&self, tree: &T) -> StorageResult<()> {
        let lock = tree.get_flush_lock();
        if tree.rotate_memtable().is_some() {
            self.memtable_flushes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(oplog) = &self.oplog {
            crate::engine::oplog_journal::sync_journal(oplog)?;
        }
        if let Some(local) = &self.local_journal {
            local.sync()?;
        }
        if tree.flush(&lock, 0)?.is_some() {
            // A new table may need compacting.
            self.compaction_wake.notify();
        }
        // The barrier: this flush and every one the flush workers left
        // unsynced reach the manifest before a caller drops the journal they
        // hold or promises a write. A tree whose flushes are synced has
        // nothing pending.
        tree.persist_flushes()?;
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
                let index = guard.append(mutations, commit_ts, Journal::Journal)?;
                Ok(Some(index))
            }
        }
    }

    /// Where `crash` now cuts the journal's active segment: its path and the
    /// length it is left at. `None` when the cut would take nothing. For
    /// crash tests, which cut the file there once the engine is gone.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn journal_crash_point(&self, crash: CrashKind) -> Option<(PathBuf, u64)> {
        let guard = self.journal_in_use()?.lock().ok()?;
        match crash {
            CrashKind::PowerCut => guard.power_cut_point(),
            CrashKind::Kill => guard.kill_point(),
        }
    }

    /// Fail the journal's sync, as a disk that refused it would: from here
    /// on the journal takes no write and no table is flushed until the store
    /// is reopened. For fault tests.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn fail_journal_syncs(&self) {
        if let Some(mut guard) = self
            .journal_in_use()
            .and_then(|journal| journal.lock().ok())
        {
            let through = guard.next_index();
            // The error is the point: the journal records the failure.
            let failed = guard.note_sync(through, Err(StorageError::Io("injected".into())));
            debug_assert!(failed.is_err());
        }
    }

    /// Show `partition`'s tree holding its group's log only below `next`,
    /// its rows left as they are: what a power cut that lost the tree's
    /// newer applies and kept another tree's leaves. For fault tests.
    ///
    /// # Errors
    ///
    /// The tree is not open, or the record cannot be made durable.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn lose_raft_coverage(&self, tree: TreeId, next: u64) -> StorageResult<()> {
        let tree = self.tree_of(tree)?;
        coverage::write_fold(&tree, Domain::Raft, 0, next, &[], self.next_seqno());
        self.flush_memtables(&tree)
    }

    /// Run `hook` in the next fixed cuts of this store's own journal once the
    /// continuation's start is read and the view's bound pinned, before the
    /// join's end is: a write it makes lands inside the join. `None` removes
    /// it. The hook must not set another. For tests of the join.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn set_cut_bound_hook(&self, hook: Option<Box<dyn FnMut() + Send>>) {
        *self.on_cut_bound.lock() = hook;
    }

    /// Run `hook` once a table's rebuild from a checkpoint has cleared the
    /// live tree, before its history goes back in. For tests of what a
    /// capture sees of a rebuild under way.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn set_rebuild_cleared_hook(&self, hook: Option<Box<dyn FnMut() + Send>>) {
        *self.on_rebuild_cleared.lock() = hook;
    }

    /// Run `hook` once a drop within an operation
    /// ([`StorageEngine::drop_columnar_table_held`]) has removed a table's tree. For
    /// tests of what a capture sees of a drop under way.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn set_table_dropped_hook(&self, hook: Option<Box<dyn FnMut() + Send>>) {
        *self.on_table_dropped.lock() = hook;
    }

    /// Run `hook` once each `STORAGE COLUMNAR` row has landed in its tree.
    /// For tests of what a capture sees of a statement writing several.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn set_columnar_row_hook(&self, hook: Option<Box<dyn FnMut() + Send>>) {
        *self.on_columnar_row.lock() = hook;
    }

    /// Run `hook` once a creation within an operation
    /// ([`StorageEngine::create_columnar_table_held`]) has its tree and stamp, before
    /// the operation records them. For tests of a table dropped and created
    /// again in between.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn set_table_created_hook(&self, hook: Option<Box<dyn FnMut() + Send>>) {
        *self.on_table_created.lock() = hook;
    }

    /// Run `hook` once a checkpoint has pinned its trees and given the set
    /// of trees back, before it copies anything. For tests of what a
    /// checkpoint takes of an operation that starts after its pins.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn set_trees_pinned_hook(&self, hook: Option<Box<dyn FnMut() + Send>>) {
        *self.on_trees_pinned.lock() = hook;
    }

    /// Stop the journal, as a disk that does not answer would: every write
    /// and sync of it waits until the returned hold is dropped. `None` for a
    /// store kept in memory. For fault tests.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn hold_journal(&self) -> Option<JournalStall<'_>> {
        Some(JournalStall {
            _held: self.journal_in_use()?.lock().ok()?,
        })
    }

    /// The journal this store writes: its own, or the member-local one when
    /// its log is a consensus log. A store keeps one of the two once open.
    fn journal_in_use(&self) -> Option<&Arc<Mutex<EmbeddedOplog>>> {
        self.oplog
            .as_ref()
            .or_else(|| self.local_journal.as_ref().map(|local| local.log()))
    }

    /// Return `true` if a retained embedded oplog journal is active.
    pub fn has_journal(&self) -> bool {
        self.oplog.is_some()
    }

    /// The engine's current sequence number: every write already applied has a
    /// seqno at or below it, and every later write a higher one.
    ///
    /// Read it before starting work that must later fold in "everything since",
    /// then feed it to [`StorageEngine::changed_keys_since`]. Saves callers from
    /// importing the Layer-3 coordinator trait to reach the same value.
    pub fn current_seqno(&self) -> lsm_tree::SeqNo {
        self.coordinator.current_seqno()
    }

    /// Get approximate disk space used by the engine in bytes.
    pub fn disk_space(&self) -> StorageResult<u64> {
        Ok(self
            .coordinator
            .trees()
            .iter()
            .map(|(_, t)| t.disk_space())
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

    /// Whether this store holds any data of its own: one live key under any
    /// partition's [user-data prefix](Partition::user_data_prefix), in any
    /// namespace.
    ///
    /// Consensus state, partition routing and internal bookkeeping do not
    /// count, so a node that has only ever been started answers `false`
    /// however many times it was restarted. This is the question a node has
    /// to answer before it joins a group that already holds data: only the
    /// member a group is formed around brings data into it, and a joining
    /// node that brought its own would have to lose it silently.
    pub fn holds_user_data(&self) -> StorageResult<bool> {
        for &part in Partition::all().iter().filter(|part| part.is_deployment()) {
            let Some(prefix) = part.user_data_prefix() else {
                continue;
            };
            if self.system_prefix_scan(part, prefix)?.next().is_some() {
                return Ok(true);
            }
        }
        for namespace in self.coordinator.sets().namespace_ids() {
            if self.engine(namespace)?.holds_own_data()? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether this store has a history of its own a group it joined would
    /// have to lose: data of its own ([`Self::holds_user_data`]) or a
    /// namespace it created, even one that is empty or dropped since. Only
    /// the member a group is formed around brings either into it.
    ///
    /// # Errors
    ///
    /// The catalog or a tree cannot be read.
    pub fn holds_local_state(&self) -> StorageResult<bool> {
        // The catalog hands out ids from 1: an allocator past it means some
        // namespace was created here.
        Ok(crate::engine::metadata::namespace_allocator(self)? != 1 || self.holds_user_data()?)
    }
}

impl StorageEngine {
    /// Approximate live item count in a partition.
    ///
    /// Read straight off the LSM metadata, so it costs no scan — but it counts
    /// every version and tombstone the levels still hold, and knows nothing of
    /// labels or shards. It is an upper bound suitable for progress bars and
    /// ETAs, never for an answer a caller might present as a row count.
    pub fn approximate_len(&self, part: Partition) -> StorageResult<usize> {
        Ok(self.tree(part)?.approximate_len())
    }

    /// Whether the namespace holds one live key under any of its partitions'
    /// [user-data prefixes](Partition::user_data_prefix).
    fn holds_own_data(&self) -> StorageResult<bool> {
        for &part in Partition::all().iter().filter(|part| !part.is_deployment()) {
            let Some(prefix) = part.user_data_prefix() else {
                continue;
            };
            if self.prefix_scan(part, prefix)?.next().is_some() {
                return Ok(true);
            }
        }
        Ok(false)
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
        self.flush_memtables(tree)?;
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
        {
            let _maintenance = crate::engine::footprint::MaintenanceScope::enter();
            tree.major_compact(u64::MAX, watermark)?;
        }
        // The tree's levels changed: the monitor judges its write pressure
        // again, which it otherwise does only after a later flush.
        self.compaction_wake.notify();
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
    /// Returns an iterator of [`StorageGuard`](crate::engine::StorageGuard)
    /// items. Use `guard.into_inner()` to get `(UserKey, UserValue)`.
    pub fn prefix_scan(&self, part: Partition, prefix: &[u8]) -> StorageResult<StorageIter> {
        self.scan_prefix_at(part, prefix, self.coordinator.current_seqno())
    }

    /// Prefix scan in descending key order (high to low) — the reverse-iteration
    /// counterpart of [`Self::prefix_scan`], walking the same double-ended LSM
    /// iterator from its high end. A "latest" / "last N within a prefix"
    /// consumer reads from the top and stops early instead of scanning the whole
    /// prefix and sorting.
    pub fn prefix_scan_rev(&self, part: Partition, prefix: &[u8]) -> StorageResult<StorageIter> {
        Ok(Box::new(
            self.scan_prefix_at(part, prefix, self.coordinator.current_seqno())?
                .rev(),
        ))
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

        // The index tree's keys go back in the form callers address them.
        let bindings = match part {
            Partition::Idx => Some(self.installations()?.current()),
            _ => None,
        };
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
                        let mut key = key.to_vec();
                        if let Some(bindings) = &bindings
                            && let Some(generation) = bindings.logical(&key)?
                        {
                            crate::engine::installation::put_generation(&mut key, generation);
                        }
                        keys.push(key);
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
        self.check_snapshot_retained(part, seqno)?;
        self.scan_prefix_at(part, prefix, seqno)
    }

    /// Inclusive-bounded range scan: yields entries with keys `K` such
    /// that `start ≤ K ≤ end`. Convenience wrapper over
    /// [`MultiModalCoordinator::range_scan`](crate::engine::coordinator::MultiModalCoordinator::range_scan);
    /// used by callers that have
    /// decomposed a query into disjoint key intervals (e.g. spatial
    /// Z-curve subrange decomposition).
    pub fn range_scan(
        &self,
        part: Partition,
        start: &[u8],
        end: &[u8],
    ) -> StorageResult<StorageIter> {
        use std::ops::Bound;
        self.scan_range_at(
            part,
            Bound::Included(start),
            Bound::Included(end),
            self.coordinator.current_seqno(),
        )
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
        use std::ops::Bound;
        Ok(Box::new(
            self.scan_range_at(
                part,
                Bound::Included(start),
                Bound::Included(end),
                self.coordinator.current_seqno(),
            )?
            .rev(),
        ))
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
        use std::ops::Bound;
        self.check_snapshot_retained(part, seqno)?;
        let tree = self.tree(part)?;
        let bindings = match part {
            Partition::Idx => Some(self.installations()?.current()),
            _ => None,
        };
        if let Some(bindings) = &bindings {
            bindings.check_history(
                Bound::Included(coverage::clamp_user_start(start)),
                Bound::Included(end),
                seqno,
            )?;
        }
        Ok(Box::new(crate::engine::installation::Seekable::open(
            part,
            tree,
            bindings.as_deref(),
            Bound::Included(coverage::clamp_user_start(start)),
            Bound::Included(end),
            seqno,
        )))
    }

    /// Create a new write batch for atomic, crash-safe mutations.
    pub fn write_batch(&self) -> WriteBatch<'_> {
        WriteBatch::new(self)
    }
}

impl Deployment {
    /// Get the tiered cache, if enabled.
    pub fn tiered_cache(&self) -> Option<&TieredCache> {
        self.tiered_cache.as_ref()
    }

    /// Get the access tracker.
    pub fn access_tracker(&self) -> &AccessTracker {
        &self.access_tracker
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
    #[cfg_attr(feature = "stage-timing", track_caller)]
    pub fn snapshot(&self) -> lsm_tree::SeqNo {
        #[cfg(feature = "stage-timing")]
        let (caller, start) = (std::panic::Location::caller(), std::time::Instant::now());
        let snapshot = self.pending_commits.complete_snapshot(
            || self.coordinator.current_seqno(),
            std::time::Duration::from_millis(
                self.snapshot_wait
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        );
        #[cfg(feature = "stage-timing")]
        crate::stage_timing::record_from("snapshot.call", caller, start);
        snapshot
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
    #[cfg_attr(feature = "stage-timing", track_caller)]
    pub fn complete_cut(&self) -> Option<u64> {
        let snapshot = self.snapshot();
        let bound = if self.raft_fence().is_some() {
            snapshot.min(self.closure_frontier())
        } else {
            snapshot
        };
        bound.checked_sub(1)
    }

    /// Wait, for at most `timeout`, until every commit at or below `through`
    /// is known complete here ([`Self::complete_cut`]); whether it is. Woken
    /// by the applies and by the closed bounds they raise, not polled.
    pub fn await_complete_through(&self, through: u64, timeout: std::time::Duration) -> bool {
        let reached = || self.complete_cut().is_some_and(|cut| cut >= through);
        if reached() {
            return true;
        }
        // Subscribed before the check that follows, so a bound raised between
        // the two still wakes the wait.
        let feed = self
            .applied_feed
            .subscribe(TreeId::system(Partition::Registry), 16, false);
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if reached() {
                return true;
            }
            let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
                return false;
            };
            if left.is_zero() {
                return false;
            }
            let _ = feed.next(Some(left));
        }
    }

    /// How many breakpoint records the closure ledger keeps in the store;
    /// `None` while the store holds no ledger.
    pub fn closure_records_held(&self) -> Option<u64> {
        match self.closure_held.load(std::sync::atomic::Ordering::Acquire) {
            NO_CLOSURE_LEDGER => None,
            held => Some(held),
        }
    }

    /// The closed bound of the Raft entries applied here; see
    /// [`Self::complete_cut`].
    pub fn closure_frontier(&self) -> u64 {
        self.closure_frontier
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// The Schema writes entry `index` makes after its proposals, with the
    /// timestamp to write them at, when the entry changes the ledger: the
    /// ledger's head with the closed bound `below` it carries (`0` for
    /// none), a breakpoint record for each commit ceiling its proposals
    /// raised, and the removal of breakpoints no view the store serves still
    /// needs, a bounded batch of them. `None` when nothing changes. The plan
    /// is installed by [`Self::raise_closure_frontier`] once they landed.
    ///
    /// `replaying`: the entry's writes are already in the store, which a
    /// restart's replay skips; it retires nothing then, since its removals
    /// would be skipped too.
    ///
    /// The timestamp is the larger of the commit ceiling and the bound less
    /// one: no earlier than the entry's own writes, and below every commit
    /// the bound leaves open.
    ///
    /// # Errors
    ///
    /// The breakpoint records could not be read.
    pub fn closure_record(
        &self,
        below: u64,
        index: u64,
        replaying: bool,
    ) -> StorageResult<Option<(Vec<coordinode_core::txn::proposal::Mutation>, u64)>> {
        let retained = if replaying {
            0
        } else {
            self.oldest_readable_seqno()
        };
        let mut state = self.closure.lock();
        let pending = std::mem::take(&mut state.pending);
        let plan = state
            .ledger
            .plan(&pending, below, index, retained, &StoredBreakpoints(self))?;
        let writes = closure_writes(&state.ledger, &plan, below);
        state.prepared = Some(plan);
        Ok(writes)
    }

    /// The ledger the Schema tree holds now, for a rebuild that replays the
    /// closure records of the entries its source cut lacks
    /// ([`Self::replay_closure_record`]). Read with the applies paused, after
    /// the source's rows are installed.
    ///
    /// # Errors
    ///
    /// The head or the breakpoint records cannot be read, or the tree holds
    /// the ledger where an earlier layout kept it.
    pub fn closure_replay(&self) -> StorageResult<ClosureReplay> {
        Ok(ClosureReplay(self.stored_closure_ledger()?))
    }

    /// Write the closure records entry `index` makes, as the apply of the
    /// entry made them, into the Schema tree a rebuild is filling: `commits`
    /// are the `(index, commit_ts)` of its proposals, `below` the bound it
    /// carries (`0` for none), `sub` its closure record's position in the
    /// entry. Records the rebuild's source already holds (`held`) are not
    /// written again, and the replay retires nothing, as a restart's replay
    /// does. Call with the applies paused.
    ///
    /// # Errors
    ///
    /// The breakpoint records cannot be read or the records cannot be
    /// written.
    pub fn replay_closure_record(
        &self,
        replay: &mut ClosureReplay,
        commits: &[(u64, u64)],
        below: u64,
        index: u64,
        sub: u32,
        held: bool,
    ) -> StorageResult<()> {
        let plan = replay
            .0
            .plan(commits, below, index, 0, &StoredBreakpoints(self))?;
        if !held && let Some((records, at)) = closure_writes(&replay.0, &plan, below) {
            self.apply_raft_proposal(&records, at, index, sub, |tree| {
                tree.part != Partition::Deployment
            })?;
        }
        replay.0.install(&plan);
        Ok(())
    }

    /// Account for entry `index` once the writes [`Self::closure_record`]
    /// planned for it landed: install its breakpoints and retirements, and
    /// raise the closed bound to `below` (`0` for none). Notify every
    /// partition's applied consumers when it rose, even if the entry changed
    /// no keys there.
    ///
    /// # Errors
    ///
    /// No plan was made for the entry, or the breakpoint records it needs
    /// could not be read.
    pub fn raise_closure_frontier(&self, below: u64, index: u64) -> StorageResult<()> {
        let mut state = self.closure.lock();
        let plan = match state.prepared.take() {
            Some(plan) => {
                // The records just written applied as the entry's last
                // proposal: bookkeeping, not a commit of the log.
                state.pending.retain(|&(at, _)| at != index);
                plan
            }
            // An entry that adds no breakpoint needs no write; one that does
            // must have planned it, or a reader would be answered from a
            // breakpoint the store does not hold.
            None => {
                let pending = std::mem::take(&mut state.pending);
                let plan =
                    state
                        .ledger
                        .plan(&pending, below, index, 0, &StoredBreakpoints(self))?;
                if !plan.writes_no_breakpoint() {
                    return Err(StorageError::Io(format!(
                        "entry {index} raised a commit ceiling without its closure record"
                    )));
                }
                plan
            }
        };
        let raised = state.ledger.install(&plan);
        self.closure_held
            .store(state.ledger.held(), std::sync::atomic::Ordering::Release);
        if !raised {
            return Ok(());
        }
        let below = state.ledger.below();
        drop(state);
        let previous = self
            .closure_frontier
            .fetch_max(below, std::sync::atomic::Ordering::AcqRel);
        if below > previous {
            self.coordinator.hold_closed_bound(below);
            self.applied_feed.closed(index, below - 1);
        }
        Ok(())
    }

    /// The latest commit a read at snapshot `view` can have seen that a
    /// member applying the same log can also be brought to see: on a Raft
    /// store, the larger of the view's complete cut and the highest commit
    /// applied here, which the leader's closed bound covers once it is
    /// handed out; `None` for a store without Raft, whose own clock bounds
    /// every commit it holds.
    ///
    /// A read's causal receipt stops here rather than at its snapshot
    /// timestamp: a snapshot taken from the clock can be later than any
    /// commit, and an idle group never closes a bound up to it, so a later
    /// read bounded by it would never be ready.
    ///
    /// The cut is the view's own: below every commit at or below `view`
    /// still in flight and below the closed bound, read as it stands. A
    /// commit stamped after the view is no part of what the read saw, and
    /// the receipt neither waits for it nor for one below the view. A Raft
    /// entry whose write the read may have seen is covered by the effects
    /// ceiling, raised before that write lands; a commit only admitted here,
    /// not yet in the log, never is.
    pub fn observed_commit_ceiling(&self, view: u64) -> Option<u64> {
        self.raft_fence()?;
        let applied = self
            .raft_effects_ceiling
            .load(std::sync::atomic::Ordering::Acquire);
        let complete = self
            .pending_commits
            .snapshot_floor(|| view)
            .min(self.closure_frontier());
        Some(
            complete
                .checked_sub(1)
                .map_or(applied, |cut| cut.max(applied)),
        )
    }

    /// A scheduling hint for the closer: a committed Raft entry reaching
    /// this node may carry a commit the closed bound does not cover yet. A
    /// leader that took over from one lost before it covered its last
    /// commits closes them: nothing else would, while no new write comes.
    /// It is raised before the entry finishes applying, so it never proves
    /// the entry ready; the closer checks its own state.
    pub fn effects_past_closure(&self) -> bool {
        self.raft_effects_ceiling
            .load(std::sync::atomic::Ordering::Acquire)
            >= self.closure_frontier()
    }

    /// Raise the effects ceiling to `commit_ts`, a committed Raft entry's,
    /// before its first write lands here.
    pub(crate) fn note_raft_effects(&self, commit_ts: u64) {
        self.raft_effects_ceiling
            .fetch_max(commit_ts, std::sync::atomic::Ordering::AcqRel);
    }

    /// Record that the Raft entry at `index` carrying `commit_ts` applied
    /// here. Entries apply in log order, so the prefix's ceiling only grows;
    /// the entry's closure record makes it a breakpoint readers see.
    pub(crate) fn note_raft_applied(&self, index: u64, commit_ts: u64) {
        self.closure.lock().pending.push((index, commit_ts));
    }

    /// The commit timestamp a read must reach to hold every commit of the
    /// Raft log through `index`, an index this member has applied: the
    /// highest commit of that prefix exactly, never one of a later prefix, so
    /// a read at an old position waits on no later writer. Once no read the
    /// store still serves can tell apart a run of old prefixes, it answers
    /// with the last of their ceilings, which every such read reaches.
    ///
    /// # Errors
    ///
    /// The breakpoint records could not be read.
    pub fn commit_ceiling_through(&self, index: u64) -> StorageResult<u64> {
        match self.commit_ceiling_at(index)? {
            CommitCeiling::Known(ceiling) => Ok(ceiling),
            CommitCeiling::Stored(stored) => self.read_stored_ceiling(stored),
        }
    }

    /// [`Self::commit_ceiling_through`] as far as memory answers it, without
    /// a read of the store: a position older than the breakpoints kept in
    /// memory is answered by [`Self::read_stored_ceiling`], a read of a
    /// stored record that holds up nothing else here while it runs.
    ///
    /// # Errors
    ///
    /// The Schema tree is owed a rebuild: the ledger it is to install is not
    /// in place, and none other answers for it.
    pub fn commit_ceiling_at(&self, index: u64) -> StorageResult<CommitCeiling> {
        let state = self.closure.lock();
        if self.ledger_owed.load(std::sync::atomic::Ordering::Acquire) {
            return Err(StorageError::InvalidConfig(
                "the Schema partition is owed a rebuild; its closure ledger is not in place".into(),
            ));
        }
        Ok(match state.ledger.ceiling_at(index) {
            crate::engine::closure_ledger::Ceiling::Known(ceiling) => CommitCeiling::Known(ceiling),
            crate::engine::closure_ledger::Ceiling::Stored(stored) => {
                CommitCeiling::Stored(stored.installed_in(state.installation))
            }
        })
    }

    /// Read a commit ceiling [`Self::commit_ceiling_at`] left to the stored
    /// records. Blocks on the read; it holds no lock the applies or other
    /// lookups take.
    ///
    /// # Errors
    ///
    /// The breakpoint records could not be read, or
    /// [`StorageError::ClosureLedgerReplaced`]: a Schema rebuild replaced the
    /// ledger `stored` was selected from while it was read.
    pub fn read_stored_ceiling(&self, stored: StoredCeiling) -> StorageResult<u64> {
        let found = stored.find(&StoredBreakpoints(self))?;
        // Checked once the read is done: a rebuild raises the installation
        // before it clears the tree, under this lock, so a read that met any
        // of its writes sees the installation moved.
        let state = self.closure.lock();
        if state.installation != stored.installation() {
            return Err(StorageError::ClosureLedgerReplaced);
        }
        Ok(found.unwrap_or_else(|| {
            // Found empty: perhaps retired under the read, which the ledger
            // answers from memory.
            state
                .ledger
                .ceiling_unstored(stored, state.prepared.as_ref())
        }))
    }

    /// Read the closed bound and the commit ceilings of the log applied here
    /// from their stored record: when the store opens and when a snapshot
    /// replaced it. A store whose applied entries carried no commit and no
    /// bound has none.
    ///
    /// # Errors
    ///
    /// The record cannot be read or is malformed.
    pub fn reload_closure_frontier(&self) -> StorageResult<()> {
        let ledger = self.stored_closure_ledger()?;
        self.closure_frontier
            .store(ledger.below(), std::sync::atomic::Ordering::Release);
        self.coordinator.hold_closed_bound(ledger.below());
        self.raft_effects_ceiling.store(
            ledger.applied_ceiling(),
            std::sync::atomic::Ordering::Release,
        );
        self.closure_held
            .store(ledger.held(), std::sync::atomic::Ordering::Release);
        let mut state = self.closure.lock();
        let installation = state.installation + 1;
        *state = ClosureState {
            ledger,
            installation,
            ..ClosureState::default()
        };
        Ok(())
    }

    /// The ledger the deployment tree holds: its head with the newest
    /// breakpoints, read from the top (never the whole window), and the
    /// first one after its anchor; empty when the tree holds no head.
    fn stored_closure_ledger(&self) -> StorageResult<crate::engine::closure_ledger::ClosureLedger> {
        use crate::engine::closure_ledger::{
            BREAKPOINT_PREFIX, Breakpoint, ClosureLedger, HEAD_KEY,
        };
        use lsm_tree::{AbstractTree, Guard as _};
        let tree = self.system_tree(Partition::Deployment)?;
        let Some(bytes) = tree.get(HEAD_KEY, lsm_tree::SeqNo::MAX)? else {
            return Ok(ClosureLedger::default());
        };
        ClosureLedger::decode(
            &bytes,
            tree.prefix(BREAKPOINT_PREFIX, lsm_tree::SeqNo::MAX, None)
                .rev()
                .map(|guard| {
                    let (key, value) = guard.into_inner()?;
                    Breakpoint::decode(&key, &value)
                }),
            &StoredBreakpoints(self),
        )
    }

    /// The ledger's records as the deployment tree holds them at `seqno`:
    /// what a copy of the partition carries beside its rows, from the same
    /// cut. Only these of the reserved records: coverage and installation
    /// records keep their own rules.
    pub(crate) fn closure_rows_at(&self, seqno: lsm_tree::SeqNo) -> StorageResult<Rows> {
        use crate::engine::closure_ledger::{BREAKPOINT_PREFIX, HEAD_KEY};
        use lsm_tree::{AbstractTree, Guard as _};
        let tree = self.system_tree(Partition::Deployment)?;
        let mut rows = Vec::new();
        if let Some(head) = tree.get(HEAD_KEY, seqno)? {
            rows.push((HEAD_KEY.to_vec(), head.to_vec()));
        }
        for guard in tree.prefix(BREAKPOINT_PREFIX, seqno, None) {
            let (key, value) = guard.into_inner()?;
            rows.push((key.to_vec(), value.to_vec()));
        }
        Ok(rows)
    }
}

impl StorageEngine {
    /// Open a tap on this namespace's `partition` and return it with a
    /// snapshot: every write to the namespace's tree is either visible at the
    /// snapshot or delivered by the tap, whatever timestamp it lands at.
    /// Another namespace's writes to the same partition reach no tap of this
    /// one.
    ///
    /// A write finishing before the tap opens is at or below the returned
    /// seqno. On a Raft store an apply lands at the leader's timestamp,
    /// which can be above this node's clock until the apply advances it, so
    /// the snapshot is read while no entry applies.
    ///
    /// Pin the snapshot (with [`Deployment::pin_snapshot_at`]) before reading at
    /// it. Blocks on a Raft store; call it off the async runtime.
    ///
    /// # Errors
    ///
    /// The errors of pausing the Raft applies.
    pub fn tap_writes(
        &self,
        partition: Partition,
    ) -> StorageResult<(crate::engine::tap::WriteTap, lsm_tree::SeqNo)> {
        let tap = self.write_taps.open(self.tree_id(partition));
        let at = self.tap_snapshot()?;
        Ok((tap, at))
    }

    /// Follow the commits applied to this namespace's `partition` (the
    /// system set's for a deployment partition) from now on, queueing at
    /// most `capacity` events; see [`crate::engine::applied`]. Another
    /// namespace's commits to the same partition are not reported.
    ///
    /// Reported, after they are in the store: the entries the Raft state
    /// machine applies through [`Deployment::apply_raft_proposal`] (with
    /// their log index), and the local commits of an engine without Raft
    /// through [`Deployment::commit_journaled`] (with their journal index) or
    /// [`Deployment::apply_proposal_at`] (with index 0).
    pub fn subscribe_applied(
        &self,
        partition: Partition,
        capacity: usize,
    ) -> crate::engine::applied::AppliedSubscription {
        self.applied_feed
            .subscribe(self.tree_id(partition), capacity, false)
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
        self.applied_feed
            .subscribe(self.tree_id(partition), capacity, true)
    }
}

impl Deployment {
    /// How many applied commits a derived-index worker's retained
    /// subscription holds ([`StorageConfig::index_feed_capacity`]).
    pub fn index_feed_capacity(&self) -> usize {
        self.index_feed_capacity
    }

    /// Applied commits the text index worker folds in one step, and rows one
    /// step of its passes writes ([`StorageConfig::text_index_batch`]).
    pub fn text_index_batch(&self) -> usize {
        self.text_index_batch
    }

    /// How often the text index worker retries what it could not finish
    /// ([`StorageConfig::text_index_retry_ms`]).
    pub fn text_index_retry(&self) -> std::time::Duration {
        self.text_index_retry
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
    /// [`StorageEngine::tap_writes`].
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
        let Some(fence) = self.raft_fence() else {
            return Ok(self.coordinator.current_seqno());
        };
        let mut at = 0;
        fence.with_applies_paused(&mut |_, _| {
            at = self.coordinator.current_seqno();
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
}

impl StorageEngine {
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
        self.check_snapshot_retained(part, *snapshot)?;
        self.check_history_of(part, key, *snapshot)?;
        let mut buf = Vec::new();
        let Some(stored) = self.stored_key(part, key, &mut buf) else {
            return Ok(inspect(None));
        };
        let value = self.tree(part)?.get(stored, *snapshot)?;
        let payload = match &value {
            Some(stored) if crate::engine::revision::stamped(part, key) => {
                Some(crate::engine::revision::payload_of(stored)?)
            }
            Some(stored) => Some(&stored[..]),
            None => None,
        };
        Ok(inspect(payload))
    }

    /// Refuse a read of the index generation `key` belongs to at a snapshot
    /// below the history its copy here holds.
    fn check_history_of(
        &self,
        part: Partition,
        key: &[u8],
        seqno: lsm_tree::SeqNo,
    ) -> StorageResult<()> {
        if part == Partition::Idx {
            self.installations()?
                .current()
                .check_key_history(key, seqno)?;
        }
        Ok(())
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
        self.check_snapshot_retained(part, *snapshot)?;
        self.snapshot_get_unchecked(snapshot, part, key)
    }

    /// [`Self::snapshot_get`] without the retention check, for a visibility
    /// probe whose caller pinned or bounded the snapshot itself.
    pub(crate) fn snapshot_get_unchecked(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<bytes::Bytes>> {
        let tree = self.tree(part)?;
        self.check_history_of(part, key, *snapshot)?;
        let mut buf = Vec::new();
        let Some(stored) = self.stored_key(part, key, &mut buf) else {
            return Ok(None);
        };
        let value = tree.get(stored, *snapshot)?;
        Ok(crate::engine::revision::user_bytes(part, key, value)?)
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
        let mut out: Vec<Option<bytes::Bytes>> = vec![None; keys.len()];
        self.snapshot_multi_get_each(snapshot, part, keys, None, |slot, value| {
            out[slot] = value.map(StoredValue::to_bytes);
            Ok::<(), StorageError>(())
        })?;
        Ok(out)
    }

    /// [`Self::snapshot_multi_get`] handing each key's value to `each` as a
    /// view into the tree's block, as [`Self::multi_get_each`] does.
    ///
    /// # Errors
    ///
    /// A storage failure (a snapshot no longer retained among them), or what
    /// `each` returns. Outside the index partition this read keeps nothing
    /// aside for the keys; the index partition's translation of generation
    /// keys keeps buffers `budget` does not charge.
    pub fn snapshot_multi_get_each<
        E: From<StorageError> + From<coordinode_core::budget::BudgetStop>,
    >(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        keys: &[&[u8]],
        _budget: Option<&coordinode_core::budget::QueryBudget>,
        mut each: impl FnMut(usize, Option<StoredValue<'_>>) -> Result<(), E>,
    ) -> Result<(), E> {
        self.check_snapshot_retained(part, *snapshot)?;
        for key in keys {
            self.check_history_of(part, key, *snapshot)?;
        }
        let values = self.stored_multi_get_stamped(part, keys, *snapshot)?;
        for (slot, (key, value)) in keys.iter().zip(&values).enumerate() {
            let Some(stored) = value else {
                each(slot, None)?;
                continue;
            };
            // A view of the payload inside the tree's buffer: no copy.
            let payload = if crate::engine::revision::stamped(part, key) {
                crate::engine::revision::payload_of(stored).map_err(StorageError::from)?
            } else {
                stored
            };
            each(slot, Some(StoredValue::Tree(payload)))?;
        }
        Ok(())
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
            let (key, value) = guard.into_owned()?;
            results.push((key, bytes::Bytes::copy_from_slice(&value)));
        }
        Ok(results)
    }

    /// [`Self::snapshot_prefix_scan`] with each value as stored, its revision
    /// stamp included: the rows a copy of the partition carries, so the copy
    /// keeps every record's revision.
    pub(crate) fn snapshot_prefix_scan_stored(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        prefix: &[u8],
    ) -> StorageResult<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut results = Vec::new();
        for guard in self.snapshot_prefix_iter(snapshot, part, prefix)? {
            let (key, value) = guard.into_stored()?;
            results.push((key, value.to_vec()));
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
        self.check_snapshot_retained(part, *snapshot)?;
        self.scan_prefix_at(part, prefix, *snapshot)
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
    /// The revision is the stamp the record's value carries, so it survives
    /// every rewrite of the
    /// record by storage: compaction, a copy of the partition installed
    /// elsewhere, a reopen.
    ///
    /// A commutative partition has no version to give: adjacency and counter
    /// rows are folded from operands rather than replaced, so there is no
    /// single commit that last wrote one. Asking is a caller error rather
    /// than an answer of `None`, which would read as "nothing is there".
    ///
    /// # Errors
    ///
    /// [`StorageError::RevisionUnavailable`] for a live record whose original
    /// commit timestamp is not known; [`StorageError::InvalidConfig`] for a
    /// commutative partition; a read failure.
    pub fn record_version(
        &self,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<lsm_tree::SeqNo>> {
        let Some(value) = self.get_versioned_at_rest(part, key)? else {
            return Ok(None);
        };
        Ok(Some(Self::revision_of(part, &value)?.1))
    }

    /// The version of a record as a read at `snapshot` saw it: the revision
    /// of the value visible there, so a value read at `snapshot` and this
    /// version are of one write. `None` when no live row is visible there.
    ///
    /// # Errors
    ///
    /// As [`Self::record_version`]; the snapshot is no longer retained.
    pub fn record_version_at(
        &self,
        snapshot: &lsm_tree::SeqNo,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<lsm_tree::SeqNo>> {
        Self::check_versioned(part, key)?;
        self.check_snapshot_retained(part, *snapshot)?;
        self.check_history_of(part, key, *snapshot)?;
        let mut buf = Vec::new();
        let Some(stored) = self.stored_key(part, key, &mut buf) else {
            return Ok(None);
        };
        let Some(value) = self.tree(part)?.get(stored, *snapshot)? else {
            return Ok(None);
        };
        Ok(Some(Self::revision_of(part, &value)?.1))
    }

    /// A record's latest value together with its [`Self::record_version`],
    /// both read from one stored value, so both are of one write: what a
    /// read-modify-write states as its condition.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::record_version`].
    pub fn get_versioned(
        &self,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<(bytes::Bytes, lsm_tree::SeqNo)>> {
        let Some(value) = self.get_versioned_at_rest(part, key)? else {
            return Ok(None);
        };
        let (payload, revision) = Self::revision_of(part, &value)?;
        Ok(Some((bytes::Bytes::copy_from_slice(payload), revision)))
    }

    /// The condition a write on this record names, for the engine's own
    /// catalog writers: its version, or zero for a live record whose commit
    /// is not known, which a condition matches for as long as nothing writes
    /// the record (every write gives it a known version). `None` when no live
    /// row is there. Never a version to hand a caller as the record's.
    ///
    /// # Errors
    ///
    /// [`StorageError::InvalidConfig`] for a commutative partition; a read
    /// failure.
    pub fn record_condition(
        &self,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<lsm_tree::SeqNo>> {
        let Some(value) = self.get_versioned_at_rest(part, key)? else {
            return Ok(None);
        };
        Ok(Some(Self::condition_of(&value)?.1))
    }

    /// A record's latest value with its [`Self::record_condition`], both of
    /// one stored value: what a catalog read-modify-write states.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::record_condition`].
    pub fn get_conditioned(
        &self,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<(bytes::Bytes, lsm_tree::SeqNo)>> {
        let Some(value) = self.get_versioned_at_rest(part, key)? else {
            return Ok(None);
        };
        let (payload, condition) = Self::condition_of(&value)?;
        Ok(Some((bytes::Bytes::copy_from_slice(payload), condition)))
    }

    /// A stored record's payload and its revision, zero when not known.
    fn condition_of(value: &[u8]) -> StorageResult<(&[u8], lsm_tree::SeqNo)> {
        Ok(
            crate::engine::revision::split(value).ok_or(lsm_tree::Error::InvalidHeader(
                "a stored record is shorter than its revision stamp",
            ))?,
        )
    }

    /// The latest stored value of a record, its revision stamp included. It
    /// is read at the present, never at the revision: a record last written
    /// before the retention horizon is still the latest state.
    fn get_versioned_at_rest(
        &self,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<lsm_tree::UserValue>> {
        Self::check_versioned(part, key)?;
        let tree = self.tree(part)?;
        let mut buf = Vec::new();
        let Some(stored) = self.stored_key(part, key, &mut buf) else {
            return Ok(None);
        };
        Ok(tree.get(stored, lsm_tree::SeqNo::MAX)?)
    }

    /// Refuse a version asked of a record that carries none: a row of a
    /// commutative partition, or a key of the engine's own.
    fn check_versioned(part: Partition, key: &[u8]) -> StorageResult<()> {
        if !part.carries_revision() {
            return Err(StorageError::InvalidConfig(format!(
                "the {part:?} partition is merge-composed: its rows are folded from \
                 operands, so no single commit is their version"
            )));
        }
        if !crate::engine::revision::stamped(part, key) {
            return Err(StorageError::InvalidConfig(
                "a key of the engine's reserved namespace has no version".into(),
            ));
        }
        Ok(())
    }

    /// A stored record's payload and its known revision.
    ///
    /// # Errors
    ///
    /// [`StorageError::RevisionUnavailable`] when the record's original
    /// commit timestamp is not known; the value holds no stamp.
    fn revision_of(part: Partition, value: &[u8]) -> StorageResult<(&[u8], lsm_tree::SeqNo)> {
        let (payload, revision) = crate::engine::revision::split(value).ok_or(
            lsm_tree::Error::InvalidHeader("a stored record is shorter than its revision stamp"),
        )?;
        if revision == crate::engine::revision::UNKNOWN {
            return Err(StorageError::RevisionUnavailable {
                partition: part.name(),
            });
        }
        Ok((payload, revision))
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
        let mut buf = Vec::new();
        let Some(stored) = self.stored_key(part, key, &mut buf) else {
            return Ok(false);
        };
        match tree.get_internal_entry(stored, lsm_tree::SeqNo::MAX)? {
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
        let mut buf = Vec::new();
        let Some(stored) = self.stored_key(part, key, &mut buf) else {
            return Ok(false);
        };
        let entry = tree.get_internal_entry(stored, lsm_tree::SeqNo::MAX)?;

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

    /// Resolve cache eviction weight for a value in a given partition: a
    /// node record weighs what its type label's name is configured to.
    fn resolve_cache_weight(&self, cache: &TieredCache, part: Partition, value: &[u8]) -> f32 {
        if part != Partition::Node {
            return 1.0;
        }
        if cache.label_weights_empty() {
            return 1.0;
        }
        let Some(label) = coordinode_core::graph::node::NodeRecord::labels_from_msgpack(value)
            .ok()
            .and_then(|labels| labels.first().copied())
        else {
            return 1.0;
        };
        cache.resolve_weight(label, |id| {
            let key = coordinode_core::schema::label_id::label_id_key(id);
            let name = self.get(Partition::Schema, &key).ok()??;
            String::from_utf8(name.to_vec()).ok()
        })
    }
}

/// Free function form of `StorageEngine::cascade_evict_endpoint` —
/// shared between the engine method and the background scanner's
/// closure. See `StorageEngine::cascade_evict_endpoint` for the
/// contract.
fn run_cascade_evict(
    endpoints: &[crate::engine::config::EndpointConfig],
    sets: &OpenSets,
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

    let records = sets.system().get(Partition::Deployment).ok_or_else(|| {
        StorageError::PartitionNotFound {
            name: Partition::Deployment.name().to_string(),
        }
    })?;
    let read_seqno = seqno.get();

    let mut compacted_partitions = 0u32;
    for (id, tree) in sets.list() {
        let part = id.part;
        // The Schema trees stay single-tier, with no routing of their own,
        // and the consensus log is never evicted.
        if matches!(
            part,
            Partition::Schema | Partition::Deployment | Partition::Raft
        ) {
            continue;
        }
        let key = routing_key_for(part);
        let bytes = match records
            .get(&key, read_seqno)
            .map_err(|e| StorageError::Io(format!("get routing {}: {e}", part.name())))?
        {
            Some(b) => b,
            None => continue,
        };
        let bytes = crate::engine::revision::payload_of(&bytes)?;
        let routing: crate::engine::routing::PartitionRouting = rmp_serde::from_slice(bytes)
            .map_err(|e| StorageError::Io(format!("decode routing for {}: {e}", part.name())))?;
        if !routing.endpoints_used().iter().any(|id| *id == endpoint_id) {
            continue;
        }
        tracing::info!(
            endpoint = endpoint_id,
            tree = %id,
            "cascade eviction: triggering major compaction"
        );
        // The engine's GC watermark for this partition, not `SeqNo::MAX`:
        // eviction moves data between endpoints and must not collect history
        // the retention policy still holds. See `StorageEngine::major_compact`
        // for what the threshold means.
        let _maintenance = crate::engine::footprint::MaintenanceScope::enter();
        tree.major_compact(64 * 1024 * 1024, gc_watermark(part))
            .map_err(|e| StorageError::Io(format!("major compact {}: {e}", part.name())))?;
        compacted_partitions += 1;
    }

    Ok(CascadeReport {
        compacted_partitions,
    })
}

/// The sets an open put up: the deployment's own trees in the system set,
/// and each namespace's set it opened.
fn open_sets_of(
    trees: &HashMap<Partition, lsm_tree::AnyTree>,
    namespaces: std::collections::BTreeMap<NamespaceId, SetTrees>,
) -> Arc<OpenSets> {
    let mut system = SetTrees::new(Scope::System);
    for (&part, tree) in trees {
        system.insert(part, tree.clone());
    }
    Arc::new(OpenSets::new(
        system,
        namespaces.into_values().map(Arc::new),
    ))
}

/// The ops of one journal entry's `ops` that write namespace `namespace`'s
/// `partition` tree: a namespace partition's write follows its namespace's
/// marker, a deployment partition's is the deployment's wherever it stands.
fn ops_of_tree(ops: &[OplogOp], namespace: NamespaceId, partition: Partition) -> Vec<&OplogOp> {
    let deployment = partition.is_deployment();
    let mut current = None;
    let mut out = Vec::new();
    for op in ops {
        if let OplogOp::Namespace { id } = op {
            current = Some(*id);
            continue;
        }
        if op_partition(op) == Some(partition)
            && (deployment || current == Some(namespace.as_raw()))
        {
            out.push(op);
        }
    }
    out
}

/// Let table `table` of `set`'s namespace stop owing the report of the repair
/// an open ran on its tree, once what it lost is restored. `false` when none
/// is owed.
///
/// # Errors
///
/// The tree cannot install the version without the report.
#[cfg(feature = "columnar")]
fn acknowledge_table_repair(set: &SetTrees, table: &str) -> StorageResult<bool> {
    use lsm_tree::AbstractTree as _;
    let mut repairs = set.table_repairs().lock();
    let Some(at) = repairs.iter().position(|repair| repair.table == table) else {
        return Ok(false);
    };
    let Some(tree) = set.columnar().and_then(|registry| registry.get(table)) else {
        // A table dropped since owes nothing.
        repairs.remove(at);
        return Ok(false);
    };
    let acknowledged = tree.acknowledge_repair_report(repairs[at].report.id)?;
    repairs.remove(at);
    Ok(acknowledged)
}

/// Namespace `id`'s directory under each of `endpoints`: where its trees,
/// on whichever tier they were placed, and its tables lie.
fn namespace_dirs(endpoints: &[EndpointConfig], id: NamespaceId) -> Vec<PathBuf> {
    let set = Scope::Namespace(id).dir();
    endpoints
        .iter()
        .map(|endpoint| endpoint.path.join(&set))
        .collect()
}

/// Remove `dirs`, a dropped namespace's directories, uncharging what they
/// held. A directory that cannot be removed now (a file still open on a
/// platform that keeps it) is left for the next open, which removes what a
/// dropping namespace left.
pub(crate) fn remove_namespace_dirs(
    dirs: &[PathBuf],
    footprints: &crate::engine::footprint::Footprints,
) {
    for dir in dirs {
        match std::fs::remove_dir_all(dir) {
            Ok(()) => footprints.removed_tree(dir),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::warn!(?dir, error = %e, "a dropped namespace's directory stays until the next open");
                footprints.removed_partly(dir);
            }
        }
    }
}

/// Remove `dirs`, a dropped namespace's directories, once every one of
/// `closing`, its closed trees, no longer touches them: a tree's background
/// work (its file pool, its deleter, a handle a pass still holds) writes in
/// its directory until then, and a removal racing it leaves files behind.
/// Nothing waits meanwhile; the last tree to close removes them.
pub(crate) fn remove_namespace_dirs_once_closed(
    closing: Vec<lsm_tree::TreeClose>,
    dirs: Vec<PathBuf>,
    footprints: crate::engine::footprint::Footprints,
) {
    if closing.is_empty() {
        remove_namespace_dirs(&dirs, &footprints);
        return;
    }
    let left = Arc::new(std::sync::atomic::AtomicUsize::new(closing.len()));
    let removal = Arc::new((dirs, footprints));
    for tree in closing {
        let (left, removal) = (Arc::clone(&left), Arc::clone(&removal));
        tree.on_closed(move || {
            // AcqRel: the closing that brings the count to zero sees every
            // other tree's closing before it removes the directories.
            if left.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                remove_namespace_dirs(&removal.0, &removal.1);
            }
        });
    }
}

/// Add tree `id`'s journal coverage record to `covered`. A tree whose
/// record is damaged no longer says what it holds: like a rebuild cut
/// short, it is left out (and so out of the replay) and owed a rebuild from
/// the latest checkpoint, under an intent written unless `owed` holds one.
///
/// # Errors
///
/// The record cannot be read for another reason, or the intent cannot be
/// written.
fn note_journal_coverage(
    id: TreeId,
    tree: &lsm_tree::AnyTree,
    owed: &[TreeId],
    data_dir: &Path,
    covered: &mut HashMap<TreeId, TreeCoverage>,
) -> StorageResult<()> {
    match TreeCoverage::read(tree, Domain::Journal) {
        Ok(record) => {
            covered.insert(id, record);
            Ok(())
        }
        Err(e) if e.is_data_damage() => {
            tracing::warn!(
                tree = %id,
                error = %e,
                "a tree's journal coverage record is damaged; the tree is rebuilt from the \
                 latest checkpoint"
            );
            if !owed.contains(&id) {
                write_rebuild_intent(data_dir, RebuildTarget::Tree(id))?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Publish each partition's footprint: the live version (in-window key
/// versions included, so this is where the retention window shows) next to
/// what is on disk beside it. A folder walk, sampled on the capacity-scan
/// cadence whether or not any endpoint has a capacity limit.
fn publish_footprint(trees: &[(TreeId, lsm_tree::AnyTree)]) {
    // One series per partition, summed over the sets holding one, so the
    // label set does not grow with the namespaces.
    let mut by_partition: HashMap<Partition, (u64, u64)> = HashMap::new();
    for (id, tree) in trees {
        match crate::engine::retention_stats::retained_history(tree) {
            Ok(history) => {
                let sums = by_partition.entry(id.part).or_insert((0, 0));
                // Bytes on the endpoints' disks: far below u64::MAX.
                sums.0 += history.live_bytes;
                sums.1 += history.retained_bytes;
            }
            Err(e) => {
                tracing::warn!(tree = %id, error = %e, "retained-history scan failed");
            }
        }
    }
    for (part, (live, retained)) in by_partition {
        metrics::gauge!("coordinode_storage_live_bytes", "partition" => part.name())
            .set(live as f64);
        metrics::gauge!(
            "coordinode_storage_retained_history_bytes",
            "partition" => part.name()
        )
        .set(retained as f64);
    }
}

/// Publish what each partition tree's reads and compactions cost, as its own
/// metrics count it: the bytes reads load, decode and copy, the useful and
/// incidental payload of projected scans, the block cache hit ratio, and the
/// bytes compactions read, wrote, carried without decoding and relocated.
#[cfg(feature = "lsm-metrics")]
fn publish_tree_metrics(trees: &[(TreeId, lsm_tree::AnyTree)]) {
    use lsm_tree::AbstractTree;

    // Summed over the sets holding a partition, so the label set does not
    // grow with the namespaces; the hit ratio is the one of the partition's
    // busiest tree.
    let mut sums: HashMap<(&'static str, &'static str), u64> = HashMap::new();
    let mut hit_ratio: HashMap<&'static str, (u64, f64)> = HashMap::new();
    for (id, tree) in trees {
        let m = tree.metrics();
        let name = id.part.name();
        let work = m.compaction_work_measured();
        let read = m.bytes_read();
        let busiest = hit_ratio.entry(name).or_insert((0, 0.0));
        if read >= busiest.0 {
            *busiest = (read, m.block_cache_hit_rate());
        }
        for (metric, value) in [
            ("coordinode_storage_read_bytes_total", m.bytes_read()),
            ("coordinode_storage_decoded_bytes_total", m.bytes_decoded()),
            ("coordinode_storage_copied_bytes_total", m.bytes_copied()),
            (
                "coordinode_storage_scan_useful_bytes_total",
                m.payload_bytes_useful(),
            ),
            (
                "coordinode_storage_scan_incidental_bytes_total",
                m.payload_bytes_incidental(),
            ),
            ("coordinode_storage_compaction_read_bytes_total", work.read),
            (
                "coordinode_storage_compaction_written_bytes_total",
                work.written,
            ),
            (
                "coordinode_storage_compaction_carried_bytes_total",
                work.carried,
            ),
            (
                "coordinode_storage_blob_relocated_bytes_total",
                m.blob_bytes_relocated(),
            ),
        ] {
            // Counts of bytes: far below u64::MAX summed.
            *sums.entry((metric, name)).or_insert(0) += value;
        }
    }
    for ((metric, name), value) in sums {
        metrics::counter!(metric, "partition" => name).absolute(value);
    }
    for (name, (_, ratio)) in hit_ratio {
        metrics::gauge!("coordinode_storage_block_cache_hit_ratio", "partition" => name).set(ratio);
    }
}

/// Publish the stored nodes per label (`coordinode_graph_nodes_total`), from
/// the counters the write path keeps on the same transaction as the nodes:
/// one read per label of each namespace, summed per label name so the series
/// do not grow with the namespaces, sampled on the capacity-scan cadence.
/// The sample replaces the last one, and a label with no node left is not in
/// it. A counter below zero is not published; no sequence of committed
/// writes leaves one. A scan that fails keeps the last sample.
fn publish_label_counts(
    sets: &OpenSets,
    seqno: &lsm_tree::SharedSequenceNumberGenerator,
    published: &crate::engine::owned_metrics::NodeCounts,
) {
    use coordinode_core::graph::stats::LABEL_KEY_PREFIX;
    use lsm_tree::{AbstractTree, Guard as _};

    let mut counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for namespace in sets.namespace_ids() {
        let Some(tree) = sets.tree(TreeId::namespace(namespace, Partition::Counter)) else {
            continue;
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
                // A label with no node left has no series.
                Ok(0) => {}
                Ok(count) if count > 0 => {
                    // Positive, so the cast is exact; nodes of a store sum far
                    // below u64::MAX.
                    *counts.entry(label.to_owned()).or_insert(0) += count as u64;
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
    published.replace(counts);
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
    trees: &[(TreeId, lsm_tree::AnyTree)],
    footprints: &crate::engine::footprint::Footprints,
    mut cascade_fn: F,
) where
    F: FnMut(&str) -> StorageResult<CascadeReport>,
{
    let endpoint_paths: std::collections::BTreeMap<String, PathBuf> = endpoints
        .iter()
        .map(|ep| (ep.id.clone(), ep.path.clone()))
        .collect();
    capacity.refresh(&endpoint_paths, footprints);
    publish_footprint(trees);

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

/// Charge `budget`, when there is one, for `slots` slots of `slot` bytes
/// about to be allocated, held while the returned charge lives. A count past
/// memory asks for more than any limit, and is refused.
pub(crate) fn reserve_slots(
    budget: Option<&coordinode_core::budget::QueryBudget>,
    slots: usize,
    slot: usize,
) -> Result<Option<coordinode_core::budget::MemoryCharge<'_>>, coordinode_core::budget::BudgetStop>
{
    let Some(budget) = budget else {
        return Ok(None);
    };
    let bytes = slots
        .checked_mul(slot)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(u64::MAX);
    budget.reserve(bytes).map(Some)
}

/// A value as the store holds it, handed to a reader that decides what to
/// copy of it: the tiered cache's own, or a view into a block the tree read.
/// Neither is the reader's memory; what the reader keeps, it copies.
#[derive(Debug, Clone, Copy)]
pub enum StoredValue<'a> {
    /// Held by the tiered cache.
    Cached(&'a bytes::Bytes),
    /// A view into the block the tree read it from.
    Tree(&'a [u8]),
}

impl<'a> StoredValue<'a> {
    /// The value's bytes, in place.
    pub fn as_bytes(self) -> &'a [u8] {
        match self {
            Self::Cached(bytes) => bytes,
            Self::Tree(value) => value,
        }
    }

    /// The value as bytes of its own: the cache's shared, a tree value
    /// copied.
    pub fn to_bytes(self) -> bytes::Bytes {
        match self {
            Self::Cached(bytes) => bytes.clone(),
            Self::Tree(value) => bytes::Bytes::copy_from_slice(value),
        }
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
/// the deployment's records, or initialise (compute default + persist) on
/// first open against this endpoint set.
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
    records: &lsm_tree::AnyTree,
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
        records
            .get(&key, read_seqno)
            .map_err(|e| StorageError::Io(format!("get routing for {}: {e}", partition.name())))?
    };
    match persisted {
        Some(bytes) => {
            // Existing routing — decode and validate against current
            // endpoint set.
            let bytes = crate::engine::revision::payload_of(&bytes)?;
            let routing: PartitionRouting = rmp_serde::from_slice(bytes).map_err(|e| {
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
            let mut batch = lsm_tree::WriteBatch::with_capacity(1);
            batch.insert(
                key.as_slice(),
                crate::engine::revision::stored(Partition::Deployment, &key, &encoded, write_seqno)
                    .as_ref(),
            );
            records.apply_batch(batch, write_seqno).map_err(|e| {
                StorageError::Io(format!("persist routing for {}: {e}", partition.name()))
            })?;
            tracing::info!(
                partition = partition.name(),
                endpoints = ?routing.endpoints_used(),
                "initialised default per-LSM-level routing"
            );
            Ok(routing)
        }
    }
}

/// A running repair of one partition, from the step that announces it to
/// its finish ([`StorageEngine::begin_partition_rebuild`],
/// [`StorageEngine::finish_raft_rebuild`]): held while it runs, so no other
/// repair or clear of the partition starts, and released when it ends.
/// Dropped without its finish, the repair failed: the partition stays owed
/// a repair, and the next one may start.
#[must_use = "the repair runs while this is held, and ends with its finish"]
pub struct PartitionRepair<'e> {
    engine: &'e Deployment,
    tree: TreeId,
    /// The set of trees, held alone by a rebuild from a checkpoint until its
    /// result is installed: no capture pins its tree half rebuilt.
    _alone: Option<crate::engine::tree_set::TreeSetExclusive<'e>>,
}

impl PartitionRepair<'_> {
    /// The partition under repair.
    pub fn partition(&self) -> Partition {
        self.tree.part
    }

    /// The tree under repair.
    pub fn tree(&self) -> TreeId {
        self.tree
    }
}

impl Drop for PartitionRepair<'_> {
    fn drop(&mut self) {
        self.engine.repairs.lock().running.remove(&self.tree);
    }
}

/// The trees under repair, and those whose repair runs now
/// ([`Deployment::enter_repair`]).
#[derive(Debug, Default)]
struct RepairMarks {
    under: rustc_hash::FxHashSet<TreeId>,
    running: rustc_hash::FxHashSet<TreeId>,
}

/// Why one log entry of a cut's continuation was not taken.
pub(crate) enum EntryOutcome {
    /// It does not fit what is left of the read's budget.
    NoRoom,
    /// Its frame does not hold together: a purge or a rewrite raced the read,
    /// or the bytes are damaged.
    Unreadable,
    /// Reading it failed.
    Failed(StorageError),
}

/// The path of the segment a continuation read holds while it reads it: no
/// longer than a path the operating system takes.
const READER_PATH_ROOM: u64 = 4 << 10;

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
/// 3. Stop the journal flusher, which syncs the journal's tail.
/// 4. Flush active memtables: best-effort final flush.
/// 5. Remaining fields (`trees`, `cache`, etc.) drop naturally after this.
impl Drop for Deployment {
    fn drop(&mut self) {
        // Step 0: stop the background capacity scanner. Joins the
        // scanner thread BEFORE other background workers — the
        // scanner only reads tree handles, but ordering is cheap
        // and keeps the shutdown sequence symmetric.
        drop(self.capacity_scanner.take());
        drop(self.space_monitor.take());

        // Step 1: stop background flush workers before touching trees.
        drop(self.flush_manager.take());

        // Step 2: stop background compaction workers.
        drop(self.compaction_scheduler.take());

        // Step 3: a clean shutdown keeps every volatile commit.
        drop(self.journal_flusher.take());
        drop(self.local_flusher.take());

        // Step 4: best-effort final flush of any remaining active memtable
        // data. A tree whose flush fails (the journal could not be synced
        // among others) keeps its data in the journal for the next open.
        for (_, tree) in self.coordinator.trees() {
            if let Err(e) = self.flush_memtables(&tree) {
                tracing::debug!(error = %e, "final memtable flush skipped");
            }
        }
    }
}

/// Outcome of [`Deployment::create_checkpoint`].
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

/// Capture the journal directory `src` into `dst`, which must not exist,
/// returning the bytes copied. A sealed segment is never written again, so
/// it is hard-linked (copied only where the link fails, across volumes);
/// anything else, the active segment above all, is copied. Nothing is ever
/// written into an existing file: a name already taken in `dst` may be a
/// hard link to a live segment, and a copy onto it would truncate the
/// journal itself. A file that disappears between the listing and its
/// capture is skipped: the journal purge removed it, and it held only
/// entries already in the trees. A link is the segment's bytes under a
/// second name, charged once; a copy is charged to `footprints` before it is
/// written.
fn capture_journal(
    src: &Path,
    dst: &Path,
    footprints: &crate::engine::footprint::Footprints,
) -> StorageResult<u64> {
    create_dir_new(dst)?;
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
            bytes += capture_journal(&entry.path(), &to, footprints)?;
        } else {
            let from = entry.path();
            match crate::oplog::segment::is_sealed(&from) {
                Ok(true) => match std::fs::hard_link(&from, &to) {
                    Ok(()) => {
                        footprints
                            .linked(&from, &to)
                            .map_err(|e| StorageError::Io(format!("link {to:?}: {e}")))?;
                        continue;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        return Err(StorageError::TargetExists {
                            path: to.display().to_string(),
                        });
                    }
                    // Across volumes, or a file system without links.
                    Err(_) => {}
                },
                Ok(false) => {}
                Err(_) if !from.exists() => continue,
                Err(e) => return Err(e),
            }
            match copy_new(&from, &to, footprints) {
                Ok(copied) => bytes += copied,
                Err(StorageError::Io(_)) if !from.exists() => {}
                Err(e) => return Err(e),
            }
        }
    }
    Ok(bytes)
}

/// Where a journal ended: the length of each of its files, by path under
/// the journal's directory. A file not listed did not exist yet.
#[derive(Debug, Default)]
pub(crate) struct JournalEnd(std::collections::HashMap<PathBuf, u64>);

/// The length of every file under the journal directory `src` now. Read
/// under the journal's append lock ([`StorageEngine::journal_end`]), each
/// ends at a whole entry.
///
/// # Errors
///
/// A directory or a file's length cannot be read.
pub(crate) fn journal_lengths(src: &Path) -> StorageResult<JournalEnd> {
    let mut end = JournalEnd::default();
    let mut pending = vec![PathBuf::new()];
    while let Some(rel) = pending.pop() {
        let dir = src.join(&rel);
        let entries = std::fs::read_dir(&dir)
            .map_err(|e| StorageError::Io(format!("read dir {dir:?}: {e}")))?;
        for entry in entries {
            let entry =
                entry.map_err(|e| StorageError::Io(format!("dir entry in {dir:?}: {e}")))?;
            let meta = entry
                .metadata()
                .map_err(|e| StorageError::Io(format!("measure {:?}: {e}", entry.path())))?;
            let name = rel.join(entry.file_name());
            if meta.is_dir() {
                pending.push(name);
            } else {
                end.0.insert(name, meta.len());
            }
        }
    }
    Ok(end)
}

/// Bring the journal copy in `dst`, taken by [`capture_journal`], up to
/// where the journal in `src` ended at `end`, never past it: a segment the
/// copy lacks is linked or copied as it was then, a segment that grew since
/// its copy is captured again up to its length then, and a segment made
/// after `end` is left out. A segment the copy holds whole is left as it
/// is, and one the purge removed since stays in the copy. Returns the bytes
/// copied.
fn capture_journal_tail(
    src: &Path,
    dst: &Path,
    end: &JournalEnd,
    footprints: &crate::engine::footprint::Footprints,
) -> StorageResult<u64> {
    let mut bytes = 0u64;
    for (name, &ended) in &end.0 {
        let from = src.join(name);
        let to = dst.join(name);
        if let Some(parent) = to.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| StorageError::Io(format!("create dir {parent:?}: {e}")))?;
        }
        let len = |path: &Path| std::fs::metadata(path).map(|m| m.len());
        match len(&to) {
            // A segment never shrinks: a copy as long as the end is the copy.
            Ok(copied) if copied == ended => continue,
            Ok(_) => {
                // The copy is a stale copy of the active segment, never a
                // link: a linked segment is sealed and keeps its length.
                std::fs::remove_file(&to)
                    .map_err(|e| StorageError::Io(format!("replace {to:?}: {e}")))?;
                footprints.removed(&to);
            }
            Err(_) => {}
        }
        if len(&from).is_ok_and(|now| now == ended)
            && crate::oplog::segment::is_sealed(&from).unwrap_or(false)
            && std::fs::hard_link(&from, &to).is_ok()
        {
            footprints
                .linked(&from, &to)
                .map_err(|e| StorageError::Io(format!("link {to:?}: {e}")))?;
            continue;
        }
        // The journal's segments are kept for the whole checkpoint: one
        // shorter than its end, or gone, would leave the copy without
        // entries the trees rely on.
        let copied = copy_prefix(&from, &to, Some(ended), footprints)?;
        if copied != ended {
            return Err(StorageError::Io(format!(
                "copy {from:?}: {copied} bytes of the {ended} the journal ended at"
            )));
        }
        bytes += copied;
    }
    Ok(bytes)
}

/// Copy `from` into a new file `to`, refusing an existing one, unlike
/// `std::fs::copy`, which truncates whatever `to` names (the source itself,
/// when `to` is a hard link to it). The copy is charged to `footprints`
/// before it is made.
fn copy_new(
    from: &Path,
    to: &Path,
    footprints: &crate::engine::footprint::Footprints,
) -> StorageResult<u64> {
    copy_prefix(from, to, None, footprints)
}

/// [`copy_new`] of the first `limit` bytes of `from` at most, all of it
/// without one.
fn copy_prefix(
    from: &Path,
    to: &Path,
    limit: Option<u64>,
    footprints: &crate::engine::footprint::Footprints,
) -> StorageResult<u64> {
    // The source first: one the purge removed leaves no empty copy behind.
    let mut source =
        std::fs::File::open(from).map_err(|e| StorageError::Io(format!("open {from:?}: {e}")))?;
    if let Some(charge) = footprints.file(to) {
        let now = source
            .metadata()
            .map_err(|e| StorageError::Io(format!("measure {from:?}: {e}")))?
            .len();
        let len = limit.map_or(now, |limit| limit.min(now));
        // An active segment may grow while it is copied; the copy is held to
        // what was charged for it.
        charge.charge_to(len)?;
        let mut target = std::fs::File::create_new(to).map_err(|e| {
            footprints.removed(to);
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                StorageError::TargetExists {
                    path: to.display().to_string(),
                }
            } else {
                StorageError::Io(format!("create {to:?}: {e}"))
            }
        })?;
        return std::io::copy(&mut std::io::Read::take(&mut source, len), &mut target)
            .map_err(|e| StorageError::Io(format!("copy {from:?} to {to:?}: {e}")));
    }
    let mut target = std::fs::File::create_new(to).map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            StorageError::TargetExists {
                path: to.display().to_string(),
            }
        } else {
            StorageError::Io(format!("create {to:?}: {e}"))
        }
    })?;
    let copied = match limit {
        Some(limit) => std::io::copy(&mut std::io::Read::take(&mut source, limit), &mut target),
        None => std::io::copy(&mut source, &mut target),
    };
    copied.map_err(|e| StorageError::Io(format!("copy {from:?} to {to:?}: {e}")))
}

/// Create directory `dir`, whose parent exists, refusing one that already
/// exists: the creation itself decides, so of two concurrent callers
/// exactly one gets it.
fn create_dir_new(dir: &Path) -> StorageResult<()> {
    std::fs::create_dir(dir).map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            StorageError::TargetExists {
                path: dir.display().to_string(),
            }
        } else {
            StorageError::Io(format!("create dir {dir:?}: {e}"))
        }
    })
}

/// Create the directory a checkpoint is written into, refusing one that
/// already exists, a concurrent checkpoint's included.
pub(crate) fn claim_checkpoint_dir(target: &Path) -> StorageResult<()> {
    if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| StorageError::Io(format!("create checkpoint parent {parent:?}: {e}")))?;
    }
    create_dir_new(target)
}

/// The maintenance budget an engine opens with: `bytes_per_sec` (`0`:
/// unthrottled), backing off while reads take longer than `backoff_ms`
/// (`0`: never).
fn new_maintenance_budget(
    bytes_per_sec: u64,
    backoff_ms: u64,
) -> Arc<lsm_tree::rate_limiter::RateLimiter> {
    let budget = Arc::new(lsm_tree::rate_limiter::RateLimiter::new(bytes_per_sec));
    if backoff_ms > 0 {
        budget.set_latency_backoff(Some(lsm_tree::rate_limiter::LatencyBackoff::new(
            std::time::Duration::from_millis(backoff_ms),
        )));
    }
    budget
}

/// Directory under the data dir holding one empty file per partition whose
/// rebuild is in progress, named after the partition.
const REBUILD_INTENT_DIR: &str = "rebuild";

/// The ledger a rebuild replays closure records against: the one its source
/// cut installed, advanced entry by entry; see
/// [`Deployment::replay_closure_record`].
pub struct ClosureReplay(crate::engine::closure_ledger::ClosureLedger);

/// The writes to the deployment's records `plan` makes against `ledger`, with the timestamp to
/// write them at, or `None` when it changes nothing: the head as the plan
/// leaves it, a record per breakpoint it adds and the removal of each it
/// retires, all in the reserved namespace. The timestamp is the larger of the
/// commit ceiling and the bound less one: no earlier than the entry's own
/// writes, and below every commit the bound leaves open.
fn closure_writes(
    ledger: &crate::engine::closure_ledger::ClosureLedger,
    plan: &crate::engine::closure_ledger::ClosurePlan,
    below: u64,
) -> Option<(Vec<coordinode_core::txn::proposal::Mutation>, u64)> {
    use crate::engine::closure_ledger::HEAD_KEY;
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    let head = ledger.encode_after(plan);
    if head == ledger.encode() && plan.writes_no_breakpoint() {
        return None;
    }
    let mut mutations = Vec::with_capacity(1 + plan.added().len() + plan.retired().len());
    mutations.push(Mutation::Put {
        partition: PartitionId::Deployment,
        key: HEAD_KEY.to_vec(),
        value: head,
    });
    mutations.extend(plan.added().iter().map(|b| Mutation::Put {
        partition: PartitionId::Deployment,
        key: b.key(),
        value: b.value().to_vec(),
    }));
    mutations.extend(plan.retired().iter().map(|b| Mutation::Delete {
        partition: PartitionId::Deployment,
        key: b.key(),
    }));
    let ceiling = plan
        .added()
        .last()
        .map_or(ledger.applied_ceiling(), |b| b.ceiling);
    let bound = below.max(ledger.below());
    let at = bound.checked_sub(1).map_or(ceiling, |b| b.max(ceiling));
    Some((mutations, at))
}

/// The closure ledger's held count while the store keeps no ledger: a store
/// that applies no consensus log owes it nothing.
const NO_CLOSURE_LEDGER: u64 = u64::MAX;

/// The closure ledger and what is staged for it.
#[derive(Default)]
struct ClosureState {
    /// What readers are answered from: only what the store holds.
    ledger: crate::engine::closure_ledger::ClosureLedger,
    /// Commits applied since the last entry's plan, `(index, commit_ts)`.
    pending: Vec<(u64, u64)>,
    /// The plan of the entry whose records are being written.
    prepared: Option<crate::engine::closure_ledger::ClosurePlan>,
    /// Which installation of the ledger this is: raised when a rebuild of the
    /// deployment's records clears the tree and when a ledger is loaded into
    /// memory, so a stored record read for one installation is not answered
    /// by another.
    installation: u64,
}

/// The breakpoint records in the deployment tree's reserved key space, read
/// by bounded seeks of the tree itself: the application's scans never reach
/// that key space.
struct StoredBreakpoints<'a>(&'a Deployment);

impl crate::engine::closure_ledger::Breakpoints for StoredBreakpoints<'_> {
    fn predecessor(
        &self,
        from: u64,
        through: u64,
    ) -> StorageResult<Option<crate::engine::closure_ledger::Breakpoint>> {
        use crate::engine::closure_ledger::{Breakpoint, breakpoint_key};
        use lsm_tree::{AbstractTree, Guard as _};
        if from > through {
            return Ok(None);
        }
        #[cfg(any(test, feature = "test-faults"))]
        breakpoint_read_gate::pass();
        self.0
            .system_tree(Partition::Deployment)?
            .range(
                breakpoint_key(from)..=breakpoint_key(through),
                lsm_tree::SeqNo::MAX,
                None,
            )
            .next_back()
            .map(|guard| {
                let (key, value) = guard.into_inner()?;
                Breakpoint::decode(&key, &value)
            })
            .transpose()
    }

    fn following(
        &self,
        after: u64,
        limit: usize,
    ) -> StorageResult<Vec<crate::engine::closure_ledger::Breakpoint>> {
        use crate::engine::closure_ledger::{Breakpoint, breakpoint_key};
        use lsm_tree::{AbstractTree, Guard as _};
        let Some(from) = after.checked_add(1) else {
            return Ok(Vec::new());
        };
        self.0
            .system_tree(Partition::Deployment)?
            .range(
                breakpoint_key(from)..=breakpoint_key(u64::MAX),
                lsm_tree::SeqNo::MAX,
                None,
            )
            .take(limit)
            .map(|guard| {
                let (key, value) = guard.into_inner()?;
                Breakpoint::decode(&key, &value)
            })
            .collect()
    }
}

/// A journalled op applied without deriving its entry first: the entry's
/// DERIVED work would otherwise be skipped silently.
fn unresolved_unit() -> StorageError {
    StorageError::InvalidConfig("a journal entry replayed without its unit resolved".into())
}

/// The partitions with a rebuild intent under `data_dir`.
/// Remove `dir` and everything under it through `fs`; nothing to remove is
/// fine.
fn remove_capture_with(fs: &dyn lsm_tree::fs::Fs, dir: &Path) -> StorageResult<()> {
    match fs.remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == lsm_tree::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(StorageError::Io(format!("remove {dir:?}: {e}"))),
    }
}

/// Holds every read of a stored closure breakpoint until released, as a disk
/// that stopped answering would, for the process the test runs in.
#[cfg(any(test, feature = "test-faults"))]
pub mod breakpoint_read_gate {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static HELD: parking_lot::Mutex<bool> = parking_lot::Mutex::new(false);
    static RELEASED: parking_lot::Condvar = parking_lot::Condvar::new();
    static WAITING: AtomicUsize = AtomicUsize::new(0);

    /// Hold the reads from now on.
    pub fn hold() {
        *HELD.lock() = true;
    }

    /// Let every held read and every later one through.
    pub fn release() {
        *HELD.lock() = false;
        RELEASED.notify_all();
    }

    /// How many reads are held now.
    pub fn waiting() -> usize {
        WAITING.load(Ordering::Acquire)
    }

    pub(super) fn pass() {
        let mut held = HELD.lock();
        if !*held {
            return;
        }
        WAITING.fetch_add(1, Ordering::AcqRel);
        while *held {
            RELEASED.wait(&mut held);
        }
        WAITING.fetch_sub(1, Ordering::AcqRel);
    }
}

pub use crate::engine::closure_ledger::StoredCeiling;

/// A log position's commit ceiling, as [`Deployment::commit_ceiling_at`]
/// answers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitCeiling {
    /// Answered from memory.
    Known(u64),
    /// To read from the stored records ([`Deployment::read_stored_ceiling`]).
    Stored(StoredCeiling),
}

/// The file that marks a store whose log is a consensus log, in the root of
/// the endpoint that holds the log. It holds the store incarnation in hex.
const CONSENSUS_LOG_MARKER: &str = "consensus-log";

/// What the consensus log marker of a store says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusBinding {
    /// No marker: the store's log was never bound to a group.
    Unbound,
    /// A marker written before markers carried an incarnation.
    Unnamed,
    /// Bound, by the store incarnation the marker names.
    Bound(crate::engine::promise::Incarnation),
}

/// What shows that the store at `endpoint`, with `trees`, keeps a replicated
/// group's history, or `None` when nothing does. The consensus log is bound
/// to the store before its first entry is written, and every entry a member
/// applies leaves Raft coverage in the trees it reaches.
fn group_history_evidence(
    endpoint: &Path,
    trees: &HashMap<Partition, lsm_tree::AnyTree>,
) -> StorageResult<Option<&'static str>> {
    if endpoint.join(CONSENSUS_LOG_MARKER).exists() {
        return Ok(Some("its log is bound to a consensus group"));
    }
    for (part, tree) in trees {
        let raft = match TreeCoverage::read(tree, Domain::Raft) {
            Ok(raft) => raft,
            // A damaged tree is rebuilt from this store's own checkpoint and
            // journal, which the other trees and the log marker vouch for.
            Err(e) if e.is_data_damage() => {
                tracing::warn!(
                    partition = part.name(),
                    error = %e,
                    "a partition's group coverage record is damaged"
                );
                continue;
            }
            Err(e) => return Err(e),
        };
        if raft.has_record() || raft.marked_indices().next().is_some() {
            return Ok(Some("its trees hold entries of a group's log"));
        }
    }
    Ok(None)
}

/// Reserve, on the footprint of a capped endpoint holding it, the room each
/// cache layer's file may reach: its `max_bytes`, twice over, since the
/// layer compacts into a fresh file beside the old one. The cache writes its
/// files directly, so the reservation is its whole charge. On a refusal
/// nothing stays reserved.
///
/// # Errors
///
/// [`StorageError::CapacityExhausted`] for a layer that does not fit;
/// [`StorageError::InvalidConfig`] for a size past what a file can hold.
fn reserve_cache_layers(
    cache: &crate::cache::config::TieredCacheConfig,
    footprints: &crate::engine::footprint::Footprints,
) -> StorageResult<()> {
    for layer in &cache.layers {
        let Some(charge) = footprints.file(&layer.path) else {
            continue;
        };
        let reserved = layer
            .max_bytes
            .checked_mul(2)
            .ok_or_else(|| {
                StorageError::InvalidConfig(format!(
                    "cache layer {:?}: max_bytes {} past what a file can hold",
                    layer.path, layer.max_bytes
                ))
            })
            .and_then(|bytes| charge.charge_to(bytes));
        if let Err(e) = reserved {
            release_cache_layers(cache, footprints);
            return Err(e);
        }
    }
    Ok(())
}

/// Bring each cache layer's charge back to what its file holds.
fn release_cache_layers(
    cache: &crate::cache::config::TieredCacheConfig,
    footprints: &crate::engine::footprint::Footprints,
) {
    for layer in &cache.layers {
        if let Some(charge) = footprints.file(&layer.path) {
            charge.settle(crate::engine::footprint::allocated_at(&layer.path).unwrap_or(0));
        }
    }
}

/// The file of a checkpoint holding the journal index a rebuild from it
/// replays from, as a sealed record ([`seal_record`]).
const REPLAY_FLOOR_FILE: &str = "replay-floor";

/// The kind of a checkpoint's replay floor record.
const REPLAY_FLOOR_MAGIC: &[u8; 8] = b"CNFLOOR1";

/// The kind of the records a Schema rebuild keeps in its intent.
const KEPT_ROWS_MAGIC: &[u8; 8] = b"CNKEPT01";

/// A deployment binding as a record holds it: a presence byte and the id.
fn deployment_tag(deployment: Option<coordinode_core::group::DeploymentId>) -> [u8; 17] {
    let mut tag = [0u8; 17];
    if let Some(id) = deployment {
        tag[0] = 1;
        tag[1..].copy_from_slice(&id.0);
    }
    tag
}

/// `payload` as a record that reads back only whole and as the kind it was
/// written as: `magic`, the payload's length, the payload, and a CRC-32C of
/// all of it.
fn seal_record(magic: &[u8; 8], payload: &[u8]) -> StorageResult<Vec<u8>> {
    let len = u32::try_from(payload.len())
        .map_err(|_| StorageError::InvalidConfig(format!("a record of {} bytes", payload.len())))?;
    let mut bytes = Vec::with_capacity(magic.len() + 4 + payload.len() + 4);
    bytes.extend_from_slice(magic);
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(payload);
    let sum = crc32c::crc32c(&bytes);
    bytes.extend_from_slice(&sum.to_le_bytes());
    Ok(bytes)
}

/// The payload of a record [`seal_record`] wrote with `magic`, or why
/// `bytes` is not one.
fn open_record<'a>(magic: &[u8; 8], bytes: &'a [u8]) -> Result<&'a [u8], &'static str> {
    if bytes.len() < magic.len() + 8 {
        return Err("shorter than its header");
    }
    let (body, sum) = bytes.split_at(bytes.len() - 4);
    let mut stored = [0u8; 4];
    stored.copy_from_slice(sum);
    if crc32c::crc32c(body) != u32::from_le_bytes(stored) {
        return Err("checksum mismatch");
    }
    if &body[..magic.len()] != magic {
        return Err("a record of another kind");
    }
    let mut len = [0u8; 4];
    len.copy_from_slice(&body[magic.len()..magic.len() + 4]);
    let payload = &body[magic.len() + 4..];
    if usize::try_from(u32::from_le_bytes(len)).ok() != Some(payload.len()) {
        return Err("length does not match its payload");
    }
    Ok(payload)
}

/// The rebuild intent file of a `STORAGE COLUMNAR` table is its namespace's
/// directory name and its id behind this prefix, so a table never names a
/// tree's.
const TABLE_INTENT_PREFIX: &str = "table.";

/// A tree a rebuild intent names.
#[derive(Debug, Clone, Copy)]
enum RebuildTarget<'a> {
    /// A partition's tree of the system set or of a namespace's.
    Tree(TreeId),
    /// A `STORAGE COLUMNAR` table of a namespace.
    #[cfg_attr(not(feature = "columnar"), allow(dead_code))]
    Table(NamespaceId, &'a str),
}

impl RebuildTarget<'_> {
    /// The intent's file name: the tree's name with `.` between its parts
    /// (neither a partition's name nor a namespace's directory holds one), or
    /// the table's behind [`TABLE_INTENT_PREFIX`].
    fn file_name(self) -> String {
        match self {
            Self::Tree(id) => id.name().replace('/', "."),
            Self::Table(namespace, table) => {
                format!("{TABLE_INTENT_PREFIX}{}.{table}", namespace.dir_name())
            }
        }
    }
}

/// Record, durably, that `target` of the store at `data_dir` holds an
/// unknown part of its data and of its coverage record, and must be rebuilt.
fn write_rebuild_intent(data_dir: &Path, target: RebuildTarget<'_>) -> StorageResult<()> {
    put_rebuild_intent(data_dir, target, &[])
}

/// [`write_rebuild_intent`] holding `kept`: what the rebuild keeps of the
/// tree it clears, durable before the clear, so a rebuild cut short after
/// it still has them. The file is charged to `footprints` before it is
/// written.
fn write_rebuild_intent_with(
    data_dir: &Path,
    target: RebuildTarget<'_>,
    kept: &[u8],
    footprints: &crate::engine::footprint::Footprints,
) -> StorageResult<()> {
    let path = data_dir.join(REBUILD_INTENT_DIR).join(target.file_name());
    crate::engine::promise::charge_record(footprints, &path, kept.len() as u64)?;
    put_rebuild_intent(data_dir, target, kept)
}

/// Write the intent file of `target` holding `kept`, durably.
fn put_rebuild_intent(
    data_dir: &Path,
    target: RebuildTarget<'_>,
    kept: &[u8],
) -> StorageResult<()> {
    use std::io::Write as _;
    let dir = data_dir.join(REBUILD_INTENT_DIR);
    let name = target.file_name();
    let io = |what: &str, e: std::io::Error| {
        StorageError::Io(format!("rebuild intent for {name}: {what}: {e}"))
    };
    // no-std: StoragePort trait, caller-injected
    std::fs::create_dir_all(&dir).map_err(|e| io("create dir", e))?;
    let mut file = std::fs::File::create(dir.join(&name)).map_err(|e| io("create", e))?;
    file.write_all(kept).map_err(|e| io("write", e))?;
    file.sync_all().map_err(|e| io("sync", e))?;
    sync_dir(&dir)?;
    sync_dir(data_dir)
}

/// The node's own records of tree `id` its pending rebuild keeps
/// ([`write_rebuild_intent_with`]): `None` when no rebuild of it is pending,
/// or its intent holds none to prove them by.
fn read_kept_rows(data_dir: &Path, id: TreeId) -> StorageResult<Option<Rows>> {
    let path = data_dir
        .join(REBUILD_INTENT_DIR)
        .join(RebuildTarget::Tree(id).file_name());
    // no-std: StoragePort trait, caller-injected
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(StorageError::Io(format!("read {path:?}: {e}"))),
    };
    if bytes.is_empty() {
        return Ok(None);
    }
    // Records that do not read back whole prove nothing.
    let Ok(payload) = open_record(KEPT_ROWS_MAGIC, &bytes) else {
        tracing::error!(path = %path.display(), "the records a rebuild kept are damaged");
        return Ok(None);
    };
    rmp_serde::from_slice(payload)
        .map(Some)
        .map_err(|e| StorageError::Io(format!("decode the records kept in {path:?}: {e}")))
}

/// `kept` as a rebuild intent holds them.
fn encode_kept_rows(kept: &Rows) -> StorageResult<Vec<u8>> {
    let payload = rmp_serde::to_vec(kept)
        .map_err(|e| StorageError::Io(format!("encode the records a rebuild keeps: {e}")))?;
    seal_record(KEPT_ROWS_MAGIC, &payload)
}

/// Drop the rebuild intent of `target`, its rebuilt tree on disk, and what
/// `footprints` was charged for it.
fn remove_rebuild_intent(
    data_dir: &Path,
    target: RebuildTarget<'_>,
    footprints: &crate::engine::footprint::Footprints,
) -> StorageResult<()> {
    let dir = data_dir.join(REBUILD_INTENT_DIR);
    let name = target.file_name();
    let path = dir.join(&name);
    // no-std: StoragePort trait, caller-injected
    std::fs::remove_file(&path)
        .map_err(|e| StorageError::Io(format!("rebuild intent for {name}: remove: {e}")))?;
    footprints.removed(&path);
    sync_dir(&dir)
}

/// The trees whose rebuild started and did not finish.
#[derive(Debug, Default)]
struct RebuildIntents {
    trees: Vec<TreeId>,
    tables: Vec<(NamespaceId, String)>,
}

fn read_rebuild_intents(data_dir: &Path) -> StorageResult<RebuildIntents> {
    let dir = data_dir.join(REBUILD_INTENT_DIR);
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(RebuildIntents::default());
        }
        Err(e) => return Err(StorageError::Io(format!("read {dir:?}: {e}"))),
    };
    let mut pending = RebuildIntents::default();
    for entry in entries {
        let entry = entry.map_err(|e| StorageError::Io(format!("entry in {dir:?}: {e}")))?;
        let name = entry.file_name();
        let unknown = || StorageError::Io(format!("unknown rebuild intent {name:?} in {dir:?}"));
        let name = name.to_str().ok_or_else(unknown)?;
        if let Some(rest) = name.strip_prefix(TABLE_INTENT_PREFIX) {
            let (namespace, table) = rest
                .split_once('.')
                .and_then(|(dir, table)| Some((NamespaceId::from_dir_name(dir)?, table)))
                .ok_or_else(unknown)?;
            pending.tables.push((namespace, table.to_string()));
            continue;
        }
        let id = TreeId::parse(&name.replace('.', "/")).ok_or_else(unknown)?;
        pending.trees.push(id);
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
    namespace: NamespaceId,
    table_id: String,
    key: Vec<u8>,
    value: Vec<u8>,
    ts: u64,
    index: u64,
}

/// Write one `STORAGE COLUMNAR` row at `seqno`, with the coverage marker of
/// its journal entry `index` in the same lsm batch, so the table tree holds
/// both or neither. Returns the bytes the write added and the memtable's size
/// after it, as the flush trigger takes them.
#[cfg(feature = "columnar")]
fn apply_columnar_row(
    tree: &lsm_tree::AnyTree,
    key: Vec<u8>,
    value: Vec<u8>,
    seqno: lsm_tree::SeqNo,
    index: Option<u64>,
) -> StorageResult<(u64, u64)> {
    let mut batch = lsm_tree::WriteBatch::with_capacity(2);
    batch.insert(key, value);
    if let Some(index) = index {
        batch.insert(Domain::Journal.marker_key(index, 0).as_slice(), &[][..]);
    }
    Ok(tree.apply_batch(batch, seqno)?)
}

mod deployment_bootstrap;
mod namespace_open;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod namespace_tests;
mod raft_coverage;
#[cfg(any(test, feature = "test-faults"))]
pub use deployment_bootstrap::CUT_AFTER_CLEAR as DEPLOYMENT_REBUILD_CUT_AFTER_CLEAR;
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
mod closure_tests;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod deployment_bootstrap_tests;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod cache_tests;

#[cfg(test)]
#[allow(clippy::expect_used)]
mod maintenance_tests;

#[cfg(all(test, feature = "columnar"))]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod columnar_table_tests;
