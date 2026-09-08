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

use super::{SeekableStorageIter, StorageIter};
use crate::cache::access::AccessTracker;
use crate::cache::tiered::TieredCache;
use crate::engine::batch::WriteBatch;
use crate::engine::compaction::CompactionScheduler;
use crate::engine::config::EndpointConfig;
use crate::engine::config::{FlushPolicy, StorageConfig};
use crate::engine::coordinator::{LocalMultiModalCoordinator, MultiModalCoordinator, SnapshotPin};
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
    flush_policy: FlushPolicy,
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
    /// equivalent retained oplog (ADR-017).
    oplog: Option<Arc<Mutex<EmbeddedOplog>>>,
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
}

/// An inclusive `[min, max]` user-key range, as reported by a lossy open
/// repair ([`OpenRepair::scoped_ranges`]) and consumed by the range-scoped
/// rebuild ([`StorageEngine::repair_partition_ranges_from_checkpoint`]).
pub type KeyRange = (Vec<u8>, Vec<u8>);

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
    /// unaffected, so only a full rebuild is sound.
    #[must_use]
    pub fn scoped_ranges(&self) -> Option<Vec<KeyRange>> {
        if !self.report.unknowable_losses.is_empty() {
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
    /// (`SegmentInstaller::wal_replay_repair`).
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
            Some(OplogJournalConfig::default()),
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
        // Built here rather than inside the coordinator: the capacity scanner
        // is spawned below, before the coordinator exists, and its cascade
        // eviction compacts — so it needs to publish and read the same
        // watermark.
        let gc_controller = LocalMultiModalCoordinator::build_gc_controller(
            Arc::clone(&gc_watermark),
            seqno.clone(),
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
            .to_tree_config(Partition::Schema, Arc::clone(&seqno), &gc_watermark)
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
        // writes.
        {
            use lsm_tree::AbstractTree;
            if let Some(max) = schema_tree.get_highest_seqno() {
                seqno.fetch_max(max + 1);
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
            let routing =
                load_or_init_partition_routing(&schema_tree, &seqno, &config.endpoints, part)?;
            let l0_endpoint = routing
                .levels
                .get(&0)
                .cloned()
                .unwrap_or_else(|| primary_endpoint_id.clone());
            partition_l0_endpoint.insert(part, l0_endpoint);
            let tree_config = config
                .to_tree_config_with_routing(
                    part,
                    Arc::clone(&seqno),
                    &gc_watermark,
                    Some(&routing),
                )
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
        // oplog (replacing the legacy per-batch persist()). On open, replay the
        // entries that are not yet durable in their partition — entry.ts > that
        // partition's highest seqno — applying each at seqno = entry.ts. Because
        // the oracle makes seqno == commit_ts == entry.ts, this is correct even
        // when the background flush worker persisted memtables on its own (a
        // fixed index cursor would lag those flushes and could double-apply a
        // non-idempotent merge). The replay must precede the seqno restore below
        // so the restored watermark covers the replayed entries.
        //
        // Columnar tables live outside the partition trees and their registry is
        // built further below, so columnar ops are collected here and replayed
        // in a second pass once the registry exists.
        #[cfg(feature = "columnar")]
        let mut columnar_replay: Vec<(String, Vec<u8>, Vec<u8>, u64)> = Vec::new();
        let oplog = match journal_config {
            Some(jcfg) => match config.select_oplog_endpoint(0) {
                Ok(endpoint) => {
                    use lsm_tree::AbstractTree;
                    let dir = endpoint.path.join("oplog").join("0");
                    let mut journal = EmbeddedOplog::open(&dir, 0, &jcfg)?;
                    let entries = journal.read_all()?;
                    let mut replayed = 0usize;
                    for entry in &entries {
                        // Group the entry's data ops per partition so each
                        // partition receives them as ONE batch at the entry's
                        // ts: the watermark test below is per partition, and
                        // the batch apply cannot straddle a memtable rotation,
                        // so a partition either holds the whole entry or none
                        // of it (the invariant `is_durable` relies on).
                        let mut per_partition: HashMap<Partition, Vec<&OplogOp>> = HashMap::new();
                        for op in &entry.ops {
                            #[cfg(feature = "columnar")]
                            if let OplogOp::ColumnarInsert {
                                table_id,
                                key,
                                value,
                            } = op
                            {
                                columnar_replay.push((
                                    table_id.clone(),
                                    key.clone(),
                                    value.clone(),
                                    entry.ts,
                                ));
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
                            if entry.ts > tree.get_highest_seqno().unwrap_or(0) {
                                apply_oplog_ops_at(tree, &ops, entry.ts)?;
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
                    Some(Arc::new(Mutex::new(journal)))
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

        // Start background flush manager.
        let flush_manager = FlushManager::start(
            &trees,
            Arc::clone(&gc_watermark),
            config.max_write_buffer_bytes,
            config.max_sealed_memtables,
            config.flush_workers,
            config.flush_poll_interval_ms,
            config.max_memtable_age_secs,
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
            Arc::clone(&gc_watermark),
            config.compaction_workers,
            config.compaction_l0_urgent_threshold,
            config.compaction_poll_interval_ms,
            config.backpressure.bytes_slowdown,
            Arc::clone(&write_pressure),
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

        // Spawn the background scanner. Skip when every endpoint has
        // `hard_limit_bytes == 0` (untracked deployment) — no point
        // spinning a thread that does nothing. The closure captures
        // cheap snapshots (AnyTree is internally Arc'd) so no
        // circular reference with `Self`.
        let capacity_scanner = if config.endpoints.iter().any(|ep| ep.hard_limit_bytes > 0) {
            let tracker_c = Arc::clone(&capacity_arc);
            let endpoints_c = config.endpoints.clone();
            let trees_c = trees.clone();
            let seqno_c = Arc::clone(&seqno);
            let gc_controller_c = Arc::clone(&gc_controller);
            let interval = std::time::Duration::from_secs(5);
            Some(
                crate::engine::capacity::CapacityScanner::start(interval, move || {
                    run_capacity_refresh(&tracker_c, &endpoints_c, &trees_c, &seqno_c, |id| {
                        run_cascade_evict(
                            &endpoints_c,
                            &trees_c,
                            &seqno_c,
                            gc_controller_c.maintenance_compaction_threshold(),
                            id,
                        )
                    });
                })
                .map_err(|e| StorageError::InvalidConfig(format!("spawn capacity scanner: {e}")))?,
            )
        } else {
            None
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
            let mut touched: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut replayed = 0usize;
            for (table_id, key, value, ts) in columnar_replay {
                let tree = columnar_tables.create_or_open(&table_id)?;
                if ts > tree.get_highest_seqno().unwrap_or(0) {
                    tree.insert(key, value, ts);
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

        let coordinator = LocalMultiModalCoordinator::new(
            trees,
            Arc::clone(&seqno),
            cache,
            gc_watermark,
            gc_controller,
        );
        coordinator.set_retention_window_us(retention_window_to_us(
            std::time::Duration::from_secs(config.retention_window_secs),
        ));
        Ok(Self {
            flush_manager: Some(flush_manager),
            compaction_scheduler: Some(compaction_scheduler),
            coordinator,
            flush_policy: config.flush_policy,
            tiered_cache,
            access_tracker: AccessTracker::new(),
            data_dir: config.data_dir().to_path_buf(),
            endpoints: config.endpoints.clone(),
            capacity: capacity_arc,
            capacity_scanner,
            partition_l0_endpoint,
            oplog,
            oracle,
            #[cfg(feature = "columnar")]
            columnar_tables,
            open_repairs,
            write_pressure,
        })
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
        self.columnar_tables.create_or_open(table_id)
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
        use lsm_tree::AbstractTree;
        // Journal the row at its commit_ts BEFORE applying it to the tree
        // memtable, so a write that reached the retained oplog survives a crash
        // and is replayed on the next open. Mirrors the partition write path
        // (`oplog_append`): columnar rows live outside the Partition keyspace,
        // so they carry a table-id-tagged ColumnarInsert op routed back to the
        // table registry on recovery. No journal (cluster mode / plain open /
        // in-memory) → durability follows the tree's own flush, as before.
        if let Some(oplog) = &self.oplog {
            let mut guard = oplog
                .lock()
                .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
            guard.append_columnar(table_id, &key, &value, seqno)?;
        }
        let tree = self.columnar_tables.create_or_open(table_id)?;
        tree.insert(key, value, seqno);
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
        for batch in tree.columnar_scan(
            &[COL_USER_KEY, COL_SEQNO, COL_VALUE_TYPE, COL_VALUE],
            None,
            snapshot,
            ..,
        )? {
            let batch = batch?;
            out.extend(crate::columnar::columnar_batch_rows(&batch)?);
        }
        Ok(out)
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
    /// independently-openable database: restore is simply
    /// `StorageEngine::open` against `target` (or a copy of it).
    ///
    /// The field interner and schema metadata live in the Schema partition
    /// tree, so they are captured by that tree's checkpoint — no separate
    /// handling needed.
    ///
    /// Scope: single-endpoint (single-node CE) layout, where every
    /// partition and the oplog live under one `data_dir`. Multi-endpoint
    /// (tiered placement across volumes) is rejected rather than silently
    /// producing a checkpoint whose restored routing points at absent
    /// source paths.
    ///
    /// # Errors
    ///
    /// - `target` already exists, or its parent cannot be created
    /// - any partition tree's checkpoint fails (see lsm `create_checkpoint`)
    /// - the engine uses more than one endpoint (multi-endpoint deferred)
    /// - the oplog directory cannot be copied
    pub fn create_checkpoint(&self, target: &Path) -> StorageResult<CheckpointSummary> {
        use lsm_tree::AbstractTree;

        if self.endpoints.len() > 1 {
            return Err(StorageError::Io(format!(
                "checkpoint of a multi-endpoint engine is not supported yet \
                 ({} endpoints configured); single-node CE backup expects one data_dir",
                self.endpoints.len()
            )));
        }
        if target.exists() {
            return Err(StorageError::Io(format!(
                "checkpoint target {target:?} already exists; refusing to overwrite"
            )));
        }
        std::fs::create_dir_all(target)
            .map_err(|e| StorageError::Io(format!("create checkpoint dir {target:?}: {e}")))?;

        let mut summary = CheckpointSummary::default();
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

        // Copy the oplog directory verbatim — sealed segments are
        // append-only, so a plain recursive byte copy is a consistent
        // snapshot for replay during restore.
        let src_oplog = self.data_dir.join("oplog");
        if src_oplog.exists() {
            summary.oplog_bytes = copy_dir_recursive(&src_oplog, &target.join("oplog"))?;
        }

        Ok(summary)
    }

    /// Rebuild a corrupt partition from a checkpoint plus oplog replay
    /// (WAL-replay-repair, repair path 2). Used when no healthy replica can
    /// serve the partition — the single-node / embedded / RF=1 case.
    ///
    /// Opens `checkpoint_dir` read-only, exports the partition's base
    /// key-values, physically drops the live (corrupt) partition tables,
    /// reinstalls the base, then replays `oplog_since` (the journal entries
    /// recorded after the checkpoint's cursor) to roll the partition forward to
    /// its current state. Returns the number of base entries reinstalled.
    ///
    /// The checkpoint persists routing under the original single-endpoint id
    /// `"default"`, so it is reopened with that id. Same-disk checkpoints only
    /// protect against localized corruption; whole-device loss requires an
    /// off-device backup (PITR).
    pub fn repair_partition_from_checkpoint(
        &self,
        checkpoint_dir: &Path,
        oplog_since: &[OplogEntry],
        partition: Partition,
    ) -> StorageResult<usize> {
        use crate::engine::config::{Durability, EndpointConfig, Media, Tier};

        // 1. Open the checkpoint read-only and export the partition base. The
        //    checkpoint engine is dropped before we mutate the live engine.
        let base: Vec<(Vec<u8>, Vec<u8>)> = {
            let ckpt_cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                checkpoint_dir,
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]);
            let ckpt = StorageEngine::open(&ckpt_cfg)?;
            let snapshot = ckpt.snapshot();
            let prefix = format!("{}:", partition.name());
            ckpt.snapshot_prefix_scan(&snapshot, partition, prefix.as_bytes())?
                .into_iter()
                .map(|(k, v)| (k, v.to_vec()))
                .collect()
        };

        // 2. Physically clear the live (corrupt) tables, reinstall the base.
        //    `clear_partition` deletes the table files without reading their
        //    blocks; a `drop_range` here would instead run a compaction that
        //    re-reads the corrupt block we are repairing and abort with a
        //    ChecksumMismatch.
        self.clear_partition(partition)?;
        let base_len = base.len();
        for (key, value) in &base {
            self.put(partition, key, value)?;
        }

        // 3. Replay the granular oplog ops since the checkpoint for this
        //    partition, rolling the base forward to the current state.
        for entry in oplog_since {
            for op in &entry.ops {
                if op_partition(op) != Some(partition) {
                    continue;
                }
                self.apply_repair_op(partition, op)?;
            }
        }
        Ok(base_len)
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
        use crate::engine::config::{Durability, EndpointConfig, Media, Tier};

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
        //    each lost range. The checkpoint engine is dropped before the
        //    live engine is mutated.
        let base: Vec<(Vec<u8>, Vec<u8>)> = {
            let ckpt_cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                checkpoint_dir,
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]);
            let ckpt = StorageEngine::open(&ckpt_cfg)?;
            let mut rows = Vec::new();
            for (min, max) in ranges {
                for guard in ckpt.range_scan(partition, min, max)? {
                    let (k, v) = guard.into_inner()?;
                    rows.push((k.to_vec(), v.to_vec()));
                }
            }
            rows
        };

        // 2. Clear the lost ranges on the live tree, reinstall the base.
        for (min, max) in ranges {
            self.remove_range(partition, min, &succ(max))?;
        }
        let base_len = base.len();
        for (key, value) in &base {
            self.put(partition, key, value)?;
        }

        // 3. Replay post-checkpoint ops intersecting the lost ranges, in
        //    journal order.
        for entry in oplog_since {
            for op in &entry.ops {
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
                }
            }
        }
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

    /// The journal index to start replaying from for a checkpoint: one past the
    /// last entry copied into `checkpoint_dir`'s oplog, or `0` if it has none.
    /// The repair orchestrator feeds this to [`oplog_read_since`] to gather the
    /// entries recorded after the checkpoint.
    ///
    /// [`oplog_read_since`]: Self::oplog_read_since
    pub fn checkpoint_oplog_cursor(checkpoint_dir: &Path) -> StorageResult<u64> {
        let oplog_dir = checkpoint_dir.join("oplog").join("0");
        let last = crate::engine::oplog_journal::last_index_in_dir(&oplog_dir)?;
        Ok(last.map(|i| i + 1).unwrap_or(0))
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
    /// dropping it would silently lose the write on the next crash.
    pub fn oplog_purge_expired(&self, now_secs: u64, keep_from_index: u64) -> StorageResult<usize> {
        use lsm_tree::AbstractTree;

        match &self.oplog {
            None => Ok(0),
            Some(oplog) => {
                // Per-partition persisted watermarks: everything at or below
                // a tree's highest persisted seqno is durable in its SSTs (a
                // flush persists the whole memtable prefix). A partition with
                // data only in its memtable answers 0, keeping every one of
                // its journal entries.
                let mut persisted: HashMap<Partition, u64> = HashMap::new();
                for &part in Partition::all() {
                    let tree = self.tree(part)?;
                    persisted.insert(part, tree.get_highest_persisted_seqno().unwrap_or(0));
                }
                let is_durable = |entry: &OplogEntry| -> bool {
                    entry.ops.iter().all(|op| match op_partition(op) {
                        Some(part) => entry.ts <= persisted.get(&part).copied().unwrap_or(0),
                        // Columnar rows live outside the partition trees:
                        // check the table's own tree when the registry knows
                        // it, otherwise keep the entry (fail-safe).
                        #[cfg(feature = "columnar")]
                        None => match op {
                            OplogOp::ColumnarInsert { table_id, .. } => self
                                .columnar_table_tree(table_id)
                                .and_then(|t| {
                                    t.get_highest_persisted_seqno().map(|w| entry.ts <= w)
                                })
                                .unwrap_or(false),
                            // Noop / Raft bookkeeping carries no partition
                            // data; nothing is lost by purging it.
                            _ => true,
                        },
                        #[cfg(not(feature = "columnar"))]
                        None => !matches!(op, OplogOp::ColumnarInsert { .. }),
                    })
                };
                let mut guard = oplog
                    .lock()
                    .map_err(|_| StorageError::Io("oplog journal mutex poisoned".into()))?;
                guard.purge_expired(now_secs, keep_from_index, &is_durable)
            }
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
                 durability — INV-D1: oplog must survive process restart)"
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

    /// The oldest seqno a time-travel read can still be served at.
    ///
    /// Two boundaries, and the horizon is the later of them:
    ///
    /// - **Policy** — the GC watermark, which the configured time-travel
    ///   window, the consumer floor and live snapshot pins drive together.
    /// - **Physical** — the oldest version each partition tree still retains.
    ///   A `drop_range`, a `clear` or a filtering compaction prunes history
    ///   independently of the window, so this can sit above the watermark; a
    ///   read below it is refused by the tree itself
    ///   (`lsm_tree::Error::SnapshotBelowRetention`). A tree is servable
    ///   strictly above `oldest_retained_seqno`, hence the `+ 1`.
    pub fn oldest_readable_seqno(&self) -> lsm_tree::SeqNo {
        use lsm_tree::AbstractTree as _;
        let mut horizon = self.coordinator.gc_watermark_value();
        for part in Partition::all() {
            if let Ok(tree) = self.tree(*part) {
                // A tree serves strictly above its oldest retained version, so
                // the first readable seqno is one past it. Checked rather than
                // clamped: `SeqNo::MAX` is the read-latest sentinel, never a
                // real install seqno, so this arm is unreachable — but if it
                // were ever reached, refusing every historical read is the safe
                // answer, and a silent wrap to 0 would admit every read
                // instead, which is the failure this whole guard exists to
                // prevent.
                let Some(first_servable) = tree.oldest_retained_seqno().checked_add(1) else {
                    return lsm_tree::SeqNo::MAX;
                };
                horizon = horizon.max(first_servable);
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

    /// Publish the consumer-registry retention floor (ADR-028 feed a):
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
        // Check tiered cache first.
        if let Some(cache) = &self.tiered_cache {
            if let Some(value) = cache.get(part, key) {
                self.access_tracker.record(part, key);
                return Ok(Some(value));
            }
        }

        // Fall through to LSM storage.
        let tree = self.tree(part)?;
        let value = tree.get(key, self.coordinator.current_seqno())?;

        match value {
            Some(v) => {
                let bytes = bytes::Bytes::copy_from_slice(&v);
                if let Some(cache) = &self.tiered_cache {
                    let weight = Self::resolve_cache_weight(cache, part, &bytes);
                    cache.put_weighted(part, key, &bytes, weight);
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

        // Split cache hits from misses; only the misses go to the tree.
        let mut miss_idx: Vec<usize> = Vec::new();
        let mut miss_keys: Vec<&[u8]> = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            if let Some(cache) = &self.tiered_cache {
                if let Some(value) = cache.get(part, key) {
                    self.access_tracker.record(part, key);
                    out[i] = Some(value);
                    continue;
                }
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

        for (slot, value) in miss_idx.into_iter().zip(values) {
            if let Some(v) = value {
                let bytes = bytes::Bytes::copy_from_slice(&v);
                if let Some(cache) = &self.tiered_cache {
                    let weight = Self::resolve_cache_weight(cache, part, &bytes);
                    cache.put_weighted(part, keys[slot], &bytes, weight);
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
    /// after flush). Under INV-D3 the gate must fire here too;
    /// operators evict via cascade or by reducing `hard_limit_bytes`,
    /// not by stuffing more tombstones onto a Full endpoint.
    pub fn delete(&self, part: Partition, key: &[u8]) -> StorageResult<()> {
        self.check_partition_capacity(part)?;
        self.coordinator.delete(part, key)?;
        if let Some(cache) = &self.tiered_cache {
            cache.remove(part, key);
        }
        Ok(())
    }

    /// Delete the half-open range `[start, end)` of a partition with one MVCC
    /// range tombstone (G096). Snapshot-aware and seqno'd, so it replicates and
    /// PITR-replays correctly — unlike [`drop_range`](Self::drop_range), which is
    /// an eager, non-MVCC table-level drop. Invalidates the partition's cache:
    /// the tombstone leaves the shadowed keys physically present, so a stale
    /// cache hit would otherwise return a deleted value, and the cache cannot
    /// range-query to invalidate precisely.
    ///
    /// Not capacity-gated — deleting frees space.
    pub fn remove_range(&self, part: Partition, start: &[u8], end: &[u8]) -> StorageResult<()> {
        self.coordinator.remove_range(part, start, end)?;
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
    /// auto-stamps under ADR-016); Merge writes the raw operand. This lets
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
        }
    }

    /// Apply every mutation of one committed proposal at a single seqno: its
    /// commit timestamp (ADR-016, `seqno == commit_ts`). This is the one write
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
            }
        }
        let seqno = match (&self.oracle, commit_ts) {
            (Some(_), ts) if ts > 0 => ts,
            _ => self.next_seqno(),
        };
        batch.commit_at(seqno)?;
        if let Some(oracle) = &self.oracle {
            oracle.advance_to(coordinode_core::txn::timestamp::Timestamp::from_raw(seqno));
            // The commit path is the retention window's clock tick: with the
            // oracle moved past this commit the window floor moves too, so a
            // compaction that follows sees the current horizon without a
            // background timer. One uncontended lock per proposal.
            self.coordinator.advance_gc_watermark();
        }
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
        // Note: tiered cache entries for dropped keys become stale.
        // They will miss on next read (key gone from tree) and naturally evict.
        // Per-partition cache clear is not implemented — drop_range is rare
        // and cache staleness is harmless (read returns None, cache evicts).
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
        if let Some(cache) = &self.tiered_cache {
            cache.clear_partition(part);
        }
        Ok(())
    }

    /// Check if a key exists in the given partition.
    pub fn contains_key(&self, part: Partition, key: &[u8]) -> StorageResult<bool> {
        let tree = self.tree(part)?;
        let value = tree.get(key, self.coordinator.current_seqno())?;
        Ok(value.is_some())
    }

    /// Flush all pending writes to durable storage.
    ///
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
        let threshold = self.coordinator.maintenance_compaction_threshold();
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
            self.coordinator.maintenance_compaction_threshold(),
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
        // The retained oplog journal is NOT truncated on flush — it must survive
        // for WAL-replay-repair, and crash recovery skips already-durable entries
        // via the seqno watermark (see `open_embedded`).
        Ok(())
    }

    /// Append a proposal's mutations to the retained oplog journal stamped at
    /// `commit_ts`, before they are applied to the memtable.
    ///
    /// Called by `OwnedLocalProposalPipeline` on engines opened via
    /// [`open_embedded`](Self::open_embedded). The journal is retained (not
    /// truncated on flush) so it drives both crash recovery and
    /// WAL-replay-repair. Returns `Some(index)` if a record was written, `None`
    /// when no journal is configured (cluster mode, plain `open`, or in-memory).
    pub fn oplog_append(
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

    /// Smallest "data durably on disk" watermark across partitions that
    /// currently have un-flushed memtable mutations.
    ///
    /// For each partition we compare the highest seqno that touched it at all
    /// (memtable + SST) against the highest seqno that has actually been
    /// flushed to an SST file:
    ///
    /// - If a partition has never been written, it is excluded from the min.
    /// - If a partition's memtable is fully persisted (`persisted >= highest`),
    ///   it is excluded from the min — it cannot lose data on crash.
    /// - Otherwise the partition is "lagging": its persisted seqno is the
    ///   ceiling beyond which mutations sit in volatile memory only.
    ///
    /// The min across all lagging partitions is the largest commit_ts T such
    /// that every mutation with `commit_ts ≤ T` is guaranteed to survive an
    /// unclean shutdown. When no partition is lagging the engine has no
    /// in-memory state to protect and `u64::MAX` is returned — the oplog
    /// purge gate is effectively open.
    ///
    /// Used by the Raft log store to gate oplog purging: an entry whose
    /// commit_ts exceeds this watermark must be retained because replay from
    /// the oplog is still the only way to reconstruct mutations sitting in
    /// some partition's memtable.
    pub fn min_partition_flushed_seqno(&self) -> u64 {
        self.coordinator
            .trees()
            .values()
            .filter_map(|t| {
                let highest = t.get_highest_seqno()?;
                let persisted = t.get_highest_persisted_seqno().unwrap_or(0);
                if persisted >= highest {
                    None
                } else {
                    Some(persisted)
                }
            })
            .min()
            .unwrap_or(u64::MAX)
    }

    /// Get the configured flush policy.
    pub fn flush_policy(&self) -> FlushPolicy {
        self.flush_policy
    }

    /// Get the shared block cache.
    pub fn cache(&self) -> &Arc<lsm_tree::Cache> {
        self.coordinator.cache()
    }

    /// Force a major compaction on a specific partition.
    ///
    /// Flushes the memtable to SST first (compaction filters only run on SST
    /// data), then triggers major compaction which invokes the MVCC GC filter.
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
        let watermark = self.coordinator.gc_watermark_value();
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
        Ok(Box::new(tree.prefix(prefix, seqno, None)))
    }

    /// Prefix scan in descending key order (high to low) — the reverse-iteration
    /// counterpart of [`Self::prefix_scan`], walking the same double-ended LSM
    /// iterator from its high end. A "latest" / "last N within a prefix"
    /// consumer reads from the top and stops early instead of scanning the whole
    /// prefix and sorting.
    pub fn prefix_scan_rev(&self, part: Partition, prefix: &[u8]) -> StorageResult<StorageIter> {
        let tree = self.tree(part)?;
        let seqno = self.coordinator.current_seqno();
        Ok(Box::new(tree.prefix(prefix, seqno, None).rev()))
    }

    /// Keys touched (written, merged, or deleted) at or after `since_seqno`
    /// (inclusive), deduplicated and sorted. Inclusive pairs with the snapshot
    /// convention (a snapshot at `S` sees versions strictly below `S`): the
    /// seqno a snapshot or cursor was taken at is exactly the first one it has
    /// not seen, so resuming from it misses nothing; a consumer that has
    /// processed a commit landed at `T` resumes from `T + 1`. The O(delta)
    /// basis for incremental snapshots: the lsm-tree surfaces only the keys
    /// whose version history reached `since_seqno`, instead of scanning the
    /// whole partition twice and diffing.
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
                    keys.push(key.to_vec());
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
    /// Used by incremental snapshots to compare old vs current state.
    pub fn prefix_scan_at(
        &self,
        part: Partition,
        prefix: &[u8],
        seqno: lsm_tree::SeqNo,
    ) -> StorageResult<StorageIter> {
        self.check_snapshot_retained(seqno)?;
        let tree = self.tree(part)?;
        Ok(Box::new(tree.prefix(prefix, seqno, None)))
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
    /// Returns the current LSM seqno. Reads with this seqno see all writes
    /// up to and including this point; later writes are invisible.
    pub fn snapshot(&self) -> lsm_tree::SeqNo {
        self.coordinator.snapshot()
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
        self.check_snapshot_retained(*snapshot)?;
        let tree = self.tree(part)?;
        let mut results = Vec::new();
        for guard in tree.prefix(prefix, *snapshot, None) {
            let (key, value) = guard.into_inner()?;
            results.push((key.to_vec(), bytes::Bytes::copy_from_slice(&value)));
        }
        Ok(results)
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
    gc_watermark: u64,
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
        // The engine's GC watermark, not `SeqNo::MAX`: eviction moves data
        // between endpoints and must not collect history the retention policy
        // still holds. See `StorageEngine::major_compact` for what the
        // threshold means.
        tree.major_compact(64 * 1024 * 1024, gc_watermark)
            .map_err(|e| StorageError::Io(format!("major compact {}: {e}", part.name())))?;
        compacted_partitions += 1;
    }

    Ok(CascadeReport {
        compacted_partitions,
    })
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

    // Persist used_bytes snapshots to Schema for warm-load on the
    // next engine open. Each snapshot is a tiny u64 (MessagePack
    // ≈ 9 bytes including header); writing one per endpoint per
    // refresh tick is negligible overhead.
    if let Some(schema_tree) = trees.get(&Partition::Schema) {
        use lsm_tree::AbstractTree;
        for (_id, usage) in capacity.iter() {
            let key = capacity_key_for(&usage.id);
            if let Ok(encoded) = rmp_serde::to_vec(&usage.used()) {
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
fn load_or_init_partition_routing(
    schema_tree: &lsm_tree::AnyTree,
    seqno: &lsm_tree::SharedSequenceNumberGenerator,
    endpoints: &[EndpointConfig],
    partition: Partition,
) -> StorageResult<PartitionRouting> {
    use lsm_tree::AbstractTree;
    let key = routing_key_for(partition);
    let read_seqno = seqno.get();
    match schema_tree.get(&key, read_seqno).map_err(|e| {
        StorageError::Io(format!("schema get routing for {}: {e}", partition.name()))
    })? {
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
            // First open against this endpoint set — compute default,
            // persist, return.
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
    /// Bytes copied for the oplog directory.
    pub oplog_bytes: u64,
    /// Highest captured lsm seqno across partitions — the checkpoint's
    /// logical position, used by PITR to bound oplog replay.
    pub max_seqno: lsm_tree::SeqNo,
}

/// Recursively copy `src` into `dst`, returning total bytes copied.
/// Plain file copy (no symlink following needed for oplog segments).
fn copy_dir_recursive(src: &Path, dst: &Path) -> StorageResult<u64> {
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
            bytes += copy_dir_recursive(&entry.path(), &to)?;
        } else {
            bytes += std::fs::copy(entry.path(), &to)
                .map_err(|e| StorageError::Io(format!("copy {:?}: {e}", entry.path())))?;
        }
    }
    Ok(bytes)
}

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

#[cfg(all(test, feature = "columnar"))]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod columnar_table_tests;
