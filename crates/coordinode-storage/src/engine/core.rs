//! CoordiNode LSM storage engine — the core KV layer for CoordiNode.
//!
//! Each logical partition maps to an `lsm_tree::AnyTree` (Tree or BlobTree)
//! opened with a shared `SharedSequenceNumberGenerator`. All trees share one
//! `lsm_tree::Cache` instance and the same seqno counter, ensuring
//! cross-partition monotonic ordering for MVCC reads.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
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
use crate::engine::partition::Partition;
use crate::engine::routing::PartitionRouting;
use crate::error::{StorageError, StorageResult};
use crate::wal::{StandaloneWal, WalConfig};

/// Newtype wrapper that bridges coordinode-core's `TimestampOracle` to
/// lsm-tree's `SequenceNumberGenerator` trait. Makes every write's LSM
/// seqno equal to the HLC timestamp for native MVCC.
#[derive(Debug)]
pub struct OracleSeqnoGenerator(
    pub std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>,
);

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
    /// Optional standalone WAL (embedded / no-Raft mode only).
    ///
    /// `Some` only when opened via `open_with_wal`. In cluster mode this is
    /// always `None` — Raft log is the crash-recovery mechanism (ADR-017).
    wal: Option<Arc<Mutex<StandaloneWal>>>,
    /// The timestamp oracle this engine stamps writes with. `Some` only
    /// when opened via [`StorageEngine::open_with_oracle`]. Exposed via
    /// [`StorageEngine::oracle`] so subsystems applying externally
    /// stamped writes (the Raft state machine on followers) can advance
    /// the SAME oracle that local MVCC readers draw snapshots from;
    /// without that, follower reads pin a stale snapshot and observe
    /// none of the replicated data.
    oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
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

    /// Open with a standalone WAL for crash durability (embedded / no-Raft mode).
    ///
    /// Pass `wal_config = None` to use the WAL with default settings:
    /// - Path: `<data_dir>/standalone.wal`
    /// - Sync: `SyncPerRecord` (full crash safety)
    ///
    /// Pass `wal_config = Some(WalConfig { … })` to customise path or sync
    /// policy (e.g. `NoSync` for test environments).
    ///
    /// In cluster mode, pass `None` for `wal_config` to `StorageEngine::open`
    /// directly — WAL is not used in cluster mode (Raft log = recovery, ADR-017).
    ///
    /// # Recovery
    ///
    /// If `standalone.wal` exists on open (crash recovery), all valid records
    /// are replayed into the memtable, then a `persist()` flushes them to SST.
    /// The WAL is then deleted and a fresh journal is started for new writes.
    pub fn open_with_wal(
        config: &StorageConfig,
        wal_config: Option<WalConfig>,
    ) -> StorageResult<Self> {
        let gc_watermark = Arc::new(AtomicU64::new(0));
        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(lsm_tree::SequenceNumberCounter::default());
        let wal_config = wal_config.unwrap_or_default();
        Self::finish_open(config, seqno, gc_watermark, Some(wal_config), None)
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
        Self::finish_open(config, seqno, gc_watermark, None, Some(oracle))
    }

    fn finish_open(
        config: &StorageConfig,
        seqno: lsm_tree::SharedSequenceNumberGenerator,
        gc_watermark: Arc<AtomicU64>,
        wal_config: Option<WalConfig>,
        oracle: Option<std::sync::Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
    ) -> StorageResult<Self> {
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

        // Pass 1: Schema partition (single-tier, no routing).
        let schema_config = config
            .to_tree_config(Partition::Schema, Arc::clone(&seqno), &gc_watermark)
            .use_cache(Arc::clone(&cache));
        let schema_tree = schema_config.open()?;
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
            let tree = tree_config.open()?;
            trees.insert(part, tree);
        }

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

        // Start background compaction scheduler.
        let compaction_scheduler = CompactionScheduler::start(
            &trees,
            Arc::clone(&gc_watermark),
            config.compaction_workers,
            config.compaction_l0_urgent_threshold,
            config.compaction_poll_interval_ms,
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

        // ── Standalone WAL setup ──────────────────────────────────────────────
        // Open and replay WAL if requested.  Must happen AFTER the lsm-tree
        // partitions are open so we can apply replayed mutations directly.
        let wal = if let Some(wal_cfg) = wal_config {
            // Standalone WAL lands at <wal_eligible_endpoint>/wal/standalone.wal
            // by default. Caller can override with an explicit
            // `wal_cfg.path` for tests or unusual deployments. Recovery
            // additionally scans every WAL-eligible endpoint to discover
            // orphaned WAL files left over by a previous config change.
            let wal_path = if let Some(p) = wal_cfg.path.clone() {
                p
            } else {
                let endpoint = config.select_wal_endpoint().map_err(|e| {
                    StorageError::Io(format!(
                        "standalone WAL requested but {e} — add at least one \
                         non-volatile NVMe/SSD/Hot-tier endpoint to StorageConfig"
                    ))
                })?;
                endpoint.path.join("wal").join("standalone.wal")
            };
            // Ensure parent dir exists (the endpoint path may exist as a
            // tempdir without the wal/ subdir).
            if let Some(parent) = wal_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| StorageError::Io(format!("create wal dir {parent:?}: {e}")))?;
            }
            // Scan other WAL-eligible endpoints for orphaned WAL
            // files that a previous config might have left behind.
            // Surfacing them as a warning lets the operator clean up
            // manually; we do NOT auto-replay foreign WAL files because
            // their lineage relative to current partition state is
            // ambiguous.
            for ep in config.all_wal_eligible_endpoints() {
                let orphan_candidate = ep.path.join("wal").join("standalone.wal");
                if orphan_candidate != wal_path && orphan_candidate.exists() {
                    tracing::warn!(
                        active_wal = %wal_path.display(),
                        orphan = %orphan_candidate.display(),
                        endpoint_id = %ep.id,
                        "orphaned standalone WAL found on a different endpoint — \
                         left in place for operator inspection (not replayed)"
                    );
                }
            }
            let sync = wal_cfg.sync;

            let (mut standalone_wal, replay_records) = StandaloneWal::open(wal_path.clone(), sync)?;

            if !replay_records.is_empty() {
                // Replay WAL records into the lsm-tree partitions.
                tracing::info!(
                    count = replay_records.len(),
                    "WAL: applying replay records to memtable"
                );
                for record in &replay_records {
                    for mutation in &record.mutations {
                        use coordinode_core::txn::proposal::Mutation as CoreMutation;
                        let part_seqno = seqno.next();
                        match mutation {
                            CoreMutation::Put {
                                partition,
                                key,
                                value,
                            } => {
                                let part = Partition::from(*partition);
                                let tree = trees.get(&part).ok_or_else(|| {
                                    StorageError::PartitionNotFound {
                                        name: part.name().to_string(),
                                    }
                                })?;
                                tree.insert(key, value, part_seqno);
                            }
                            CoreMutation::Delete { partition, key } => {
                                let part = Partition::from(*partition);
                                let tree = trees.get(&part).ok_or_else(|| {
                                    StorageError::PartitionNotFound {
                                        name: part.name().to_string(),
                                    }
                                })?;
                                tree.remove(key, part_seqno);
                            }
                            CoreMutation::Merge {
                                partition,
                                key,
                                operand,
                            } => {
                                let part = Partition::from(*partition);
                                let tree = trees.get(&part).ok_or_else(|| {
                                    StorageError::PartitionNotFound {
                                        name: part.name().to_string(),
                                    }
                                })?;
                                tree.merge(key, operand, part_seqno);
                            }
                            CoreMutation::RemoveRange {
                                partition,
                                start,
                                end,
                            } => {
                                let part = Partition::from(*partition);
                                let tree = trees.get(&part).ok_or_else(|| {
                                    StorageError::PartitionNotFound {
                                        name: part.name().to_string(),
                                    }
                                })?;
                                tree.remove_range(start.clone(), end.clone(), part_seqno);
                            }
                        }
                    }
                }

                // Flush replayed memtable data to SST for durability.
                tracing::info!("WAL: flushing replayed data to SST");
                for tree in trees.values() {
                    tree.flush_active_memtable(0)?;
                }

                // WAL replay complete — checkpoint (rotate) to start fresh.
                tracing::info!(path = %wal_path.display(), "WAL: checkpoint after recovery");
                standalone_wal.checkpoint()?;
            }

            Some(Arc::new(Mutex::new(standalone_wal)))
        } else {
            None
        };

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
            let interval = std::time::Duration::from_secs(5);
            Some(
                crate::engine::capacity::CapacityScanner::start(interval, move || {
                    run_capacity_refresh(&tracker_c, &endpoints_c, &trees_c, &seqno_c, |id| {
                        run_cascade_evict(&endpoints_c, &trees_c, &seqno_c, id)
                    });
                })
                .map_err(|e| StorageError::InvalidConfig(format!("spawn capacity scanner: {e}")))?,
            )
        } else {
            None
        };

        let coordinator =
            LocalMultiModalCoordinator::new(trees, Arc::clone(&seqno), cache, gc_watermark);
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
            wal,
            oracle,
        })
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

    /// Configured endpoints ([storage-stack.md](../../arch/core/storage-stack.md)
    /// Layer 1). Subsystems consult this to resolve WAL/oplog/SST placement
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

    /// Resolve the oplog target endpoint for a given shard
    /// ([storage-stack.md](../../arch/core/storage-stack.md) Layer 1↔2,
    /// INV-D1). Convenience accessor that re-runs the per-shard
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

    /// Pin a read snapshot at an explicit seqno (a statement reading at its
    /// allocated `read_ts`, or a long-lived backup / CDC consumer). Holds the
    /// GC watermark at or below `seqno` until the guard drops.
    pub fn pin_snapshot_at(&self, seqno: lsm_tree::SeqNo) -> SnapshotPin {
        self.coordinator.pin_snapshot_at(seqno)
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

    /// The current GC watermark value. Observability + test hook.
    pub fn gc_watermark(&self) -> u64 {
        self.coordinator.gc_watermark_value()
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
        // `seqno_threshold = SeqNo::MAX` means "drop nothing for
        // retention" — major compaction here is for placement, not GC.
        tree.major_compact(64 * 1024 * 1024, lsm_tree::SeqNo::MAX)
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
        // Checkpoint WAL after successful SST flush.
        if let Some(wal) = &self.wal {
            let mut guard = wal
                .lock()
                .map_err(|_| StorageError::Io("WAL mutex poisoned".into()))?;
            guard.checkpoint()?;
        }
        Ok(())
    }

    /// Append mutations to the standalone WAL before applying them to the memtable.
    ///
    /// Called by `OwnedLocalProposalPipeline` when a WAL is configured.
    /// Replaces the per-proposal `persist()` call: instead of flushing the
    /// entire memtable to SST on every write, a lightweight WAL record
    /// (≤4 KB typical, fsync ~0.1–0.5 ms) is written and memtable follows.
    ///
    /// Returns `Some(lsn)` if a WAL record was written, `None` when no WAL
    /// is configured (cluster mode or plain `open()`).
    pub fn wal_append(&self, mutations: &[Mutation]) -> StorageResult<Option<u64>> {
        match &self.wal {
            None => Ok(None),
            Some(wal) => {
                let mut guard = wal
                    .lock()
                    .map_err(|_| StorageError::Io("WAL mutex poisoned".into()))?;
                let lsn = guard.append(mutations)?;
                Ok(Some(lsn))
            }
        }
    }

    /// Return `true` if a standalone WAL is active.
    pub fn has_wal(&self) -> bool {
        self.wal.is_some()
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

    /// Keys touched (written, merged, or deleted) strictly after
    /// `since_seqno`, deduplicated and sorted. The O(delta) basis for
    /// incremental snapshots: the lsm-tree surfaces only the keys whose
    /// version history advanced past `since_seqno`, instead of scanning the
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
                ScanSinceEvent::Insert { key, .. }
                | ScanSinceEvent::MergeOperand { key, .. }
                | ScanSinceEvent::PointTombstone { key, .. } => {
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
    /// returning the latest visible version of each key. Keep snapshots
    /// short-lived to avoid blocking compaction.
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
        let tree = self.tree(part)?;
        let value = tree.get(key, *snapshot)?;
        Ok(value.map(|v| bytes::Bytes::copy_from_slice(&v)))
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
        tree.major_compact(64 * 1024 * 1024, lsm_tree::SeqNo::MAX)
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
mod wal_integration_tests;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod merge_tests;
