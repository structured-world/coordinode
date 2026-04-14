//! CoordiNode LSM storage engine — the core KV layer for CoordiNode.
//!
//! Each logical partition maps to an `lsm_tree::AnyTree` (Tree or BlobTree)
//! opened with a shared `SharedSequenceNumberGenerator`. All trees share one
//! `lsm_tree::Cache` instance and the same seqno counter, ensuring
//! cross-partition monotonic ordering for MVCC reads.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use coordinode_core::txn::proposal::Mutation;
use lsm_tree::{AbstractTree, Guard};
use tracing::info;

use super::StorageIter;
use crate::cache::access::AccessTracker;
use crate::cache::tiered::TieredCache;
use crate::engine::batch::WriteBatch;
use crate::engine::compaction::CompactionScheduler;
use crate::engine::config::{FlushPolicy, StorageConfig};
use crate::engine::flush::FlushManager;
use crate::engine::partition::Partition;
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
    trees: HashMap<Partition, lsm_tree::AnyTree>,
    seqno: lsm_tree::SharedSequenceNumberGenerator,
    cache: Arc<lsm_tree::Cache>,
    flush_policy: FlushPolicy,
    /// Optional tiered block cache (DRAM → NVMe → SSD cascade).
    tiered_cache: Option<TieredCache>,
    /// Per-key access tracker for cache eviction and heat map.
    access_tracker: AccessTracker,
    /// Shared GC watermark for the seqno-based retention compaction filter.
    /// Versions with seqno <= this value are eligible for GC during compaction.
    gc_watermark: Arc<AtomicU64>,
    /// Root data directory — exposed so subsystems (e.g. Raft oplog) can
    /// derive their own sub-directories without re-reading the config.
    data_dir: PathBuf,
    /// Optional standalone WAL (embedded / no-Raft mode only).
    ///
    /// `Some` only when opened via `open_with_wal`. In cluster mode this is
    /// always `None` — Raft log is the crash-recovery mechanism (ADR-017).
    wal: Option<Arc<Mutex<StandaloneWal>>>,
}

impl StorageEngine {
    /// Open or create a CoordiNode storage engine at the configured path.
    ///
    /// Creates all 8 partition trees if they don't exist.
    pub fn open(config: &StorageConfig) -> StorageResult<Self> {
        let gc_watermark = Arc::new(AtomicU64::new(0));
        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(lsm_tree::SequenceNumberCounter::default());
        Self::finish_open(config, seqno, gc_watermark, None)
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
        Self::finish_open(config, seqno, gc_watermark, Some(wal_config))
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
        let seqno: lsm_tree::SharedSequenceNumberGenerator = Arc::new(OracleSeqnoGenerator(oracle));
        Self::finish_open(config, seqno, gc_watermark, None)
    }

    fn finish_open(
        config: &StorageConfig,
        seqno: lsm_tree::SharedSequenceNumberGenerator,
        gc_watermark: Arc<AtomicU64>,
        wal_config: Option<WalConfig>,
    ) -> StorageResult<Self> {
        // Shared block cache across all partition trees.
        let cache = Arc::new(lsm_tree::Cache::with_capacity_bytes(
            config.block_cache_bytes,
        ));

        let mut trees = HashMap::with_capacity(Partition::all().len());
        for &part in Partition::all() {
            let tree_config = config
                .to_tree_config(part, Arc::clone(&seqno), &gc_watermark)
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
            path = %config.data_dir.display(),
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
            let wal_path = wal_cfg
                .path
                .unwrap_or_else(|| config.data_dir.join("standalone.wal"));
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

        Ok(Self {
            flush_manager: Some(flush_manager),
            compaction_scheduler: Some(compaction_scheduler),
            trees,
            seqno,
            cache,
            flush_policy: config.flush_policy,
            tiered_cache,
            access_tracker: AccessTracker::new(),
            gc_watermark,
            data_dir: config.data_dir.clone(),
            wal,
        })
    }

    /// Get a tree handle by logical partition.
    pub fn tree(&self, part: Partition) -> StorageResult<&lsm_tree::AnyTree> {
        self.trees
            .get(&part)
            .ok_or_else(|| StorageError::PartitionNotFound {
                name: part.name().to_string(),
            })
    }

    /// Root data directory for this engine.
    ///
    /// Subsystems that need their own on-disk sub-directories (e.g. the Raft
    /// oplog at `data_dir/raft_oplog/`) derive their paths from this.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Advance the seqno by one and return the new value.
    ///
    /// Used by [`WriteBatch`] to assign a single seqno to an entire batch.
    pub(crate) fn next_seqno(&self) -> lsm_tree::SeqNo {
        self.seqno.next()
    }

    /// Update the GC watermark for the seqno-based retention filter.
    ///
    /// Versions with `seqno <= watermark` become eligible for removal
    /// during LSM compaction.
    pub fn set_gc_watermark(&self, watermark: u64) {
        self.gc_watermark.store(watermark, Ordering::Release);
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
        let value = tree.get(key, self.seqno.get())?;

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
    pub fn put(&self, part: Partition, key: &[u8], value: &[u8]) -> StorageResult<()> {
        let tree = self.tree(part)?;
        tree.insert(key, value, self.seqno.next());
        if let Some(cache) = &self.tiered_cache {
            cache.remove(part, key);
        }
        Ok(())
    }

    /// Delete a key from the given partition.
    pub fn delete(&self, part: Partition, key: &[u8]) -> StorageResult<()> {
        let tree = self.tree(part)?;
        tree.remove(key, self.seqno.next());
        if let Some(cache) = &self.tiered_cache {
            cache.remove(part, key);
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
        let tree = self.tree(part)?;
        tree.merge(key, operand, self.seqno.next());
        if let Some(cache) = &self.tiered_cache {
            cache.remove(part, key);
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

    /// Check if a key exists in the given partition.
    pub fn contains_key(&self, part: Partition, key: &[u8]) -> StorageResult<bool> {
        let tree = self.tree(part)?;
        let value = tree.get(key, self.seqno.get())?;
        Ok(value.is_some())
    }

    /// Flush all pending writes to durable storage.
    ///
    /// Flushes the active memtable of every partition tree to an SST file.
    /// SST files are written atomically (atomic rename), so this provides
    /// crash safety without requiring a separate WAL fsync.
    ///
    /// When a standalone WAL is active, a WAL checkpoint (rotation) is
    /// performed after the SST flush.  This keeps the WAL small: all data
    /// that was in the WAL is now in SST, so the journal can be truncated.
    pub fn persist(&self) -> StorageResult<()> {
        for tree in self.trees.values() {
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
        Ok(self.trees.values().map(|t| t.disk_space()).sum())
    }

    /// Get the configured flush policy.
    pub fn flush_policy(&self) -> FlushPolicy {
        self.flush_policy
    }

    /// Get the shared block cache.
    pub fn cache(&self) -> &Arc<lsm_tree::Cache> {
        &self.cache
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
        // Major compaction: compact all SST data.
        tree.major_compact(u64::MAX, 0)?;
        Ok(())
    }

    /// Scan all key-value pairs in a partition whose keys start with the given prefix.
    ///
    /// Returns an iterator of `IterGuardImpl` items. Use `guard.into_inner()`
    /// to get `(UserKey, UserValue)`.
    pub fn prefix_scan(&self, part: Partition, prefix: &[u8]) -> StorageResult<StorageIter> {
        let tree = self.tree(part)?;
        let seqno = self.seqno.get();
        Ok(Box::new(tree.prefix(prefix, seqno, None)))
    }

    /// Scan key-value pairs visible at a specific sequence number.
    ///
    /// Like [`prefix_scan`], but reads at an arbitrary point-in-time.
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
        self.seqno.get()
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
        // Step 1: stop background flush workers before touching trees.
        drop(self.flush_manager.take());

        // Step 2: stop background compaction workers.
        drop(self.compaction_scheduler.take());

        // Step 3: best-effort final flush of any remaining active memtable data.
        for tree in self.trees.values() {
            let _ = tree.flush_active_memtable(0);
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Counter for unique MemFs paths across parallel tests.
    static MEMFS_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Create a test engine. Uses MemFs when `COORDINODE_TEST_MEMFS=1` env var is set,
    /// otherwise falls back to tempfile on disk.
    ///
    /// MemFs is ~10x faster (no disk I/O) but has limitations:
    /// - No oplog support (oplog uses std::fs directly)
    /// - Compaction finalization may fail on some code paths (lsm-tree known limitation)
    /// - TieredCache layers not supported (filesystem-based)
    ///
    /// Returns `(StorageEngine, Option<TempDir>)` — TempDir is None for MemFs.
    fn test_engine() -> (StorageEngine, Option<TempDir>) {
        if std::env::var("COORDINODE_TEST_MEMFS").as_deref() == Ok("1") {
            let engine = test_engine_memfs();
            (engine, None)
        } else {
            let dir = TempDir::new().expect("failed to create temp dir");
            let config = StorageConfig::new(dir.path());
            let engine = StorageEngine::open(&config).expect("failed to open engine");
            (engine, Some(dir))
        }
    }

    fn test_engine_memfs() -> StorageEngine {
        let id = MEMFS_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let config = StorageConfig::with_memfs(format!("/memfs/test_{id}"));
        StorageEngine::open(&config).expect("failed to open memfs engine")
    }

    #[test]
    fn memfs_engine_basic_kv() {
        let engine = test_engine_memfs();
        engine
            .put(Partition::Node, b"node:00:00000001", b"hello")
            .expect("put");
        let val = engine
            .get(Partition::Node, b"node:00:00000001")
            .expect("get");
        assert_eq!(val.as_deref(), Some(b"hello".as_slice()));
    }

    #[test]
    fn memfs_engine_all_partitions() {
        let engine = test_engine_memfs();
        for &part in Partition::all() {
            assert!(engine.tree(part).is_ok(), "partition {:?} missing", part);
        }
    }

    #[test]
    fn memfs_engine_multi_partition_writes() {
        let engine = test_engine_memfs();

        // Write to multiple partitions
        engine
            .put(Partition::Node, b"node:00:00000001", b"alice")
            .expect("put node");
        engine
            .put(Partition::Schema, b"schema:label:User", b"schema_data")
            .expect("put schema");
        engine
            .put(
                Partition::EdgeProp,
                b"edgeprop:KNOWS:00000001:00000002",
                b"props",
            )
            .expect("put edgeprop");

        // Read back from each partition
        assert_eq!(
            engine
                .get(Partition::Node, b"node:00:00000001")
                .expect("get")
                .as_deref(),
            Some(b"alice".as_slice())
        );
        assert_eq!(
            engine
                .get(Partition::Schema, b"schema:label:User")
                .expect("get")
                .as_deref(),
            Some(b"schema_data".as_slice())
        );
        assert_eq!(
            engine
                .get(Partition::EdgeProp, b"edgeprop:KNOWS:00000001:00000002")
                .expect("get")
                .as_deref(),
            Some(b"props".as_slice())
        );

        // Non-existent key returns None
        assert!(engine
            .get(Partition::Node, b"node:00:99999999")
            .expect("get")
            .is_none());
    }

    #[test]
    fn memfs_engine_merge_operator_adj() {
        use crate::engine::merge::encode_add_batch;

        let engine = test_engine_memfs();

        // Merge operator on Adj partition (PostingListMerge)
        let key = b"adj:KNOWS:out:00000001";
        let delta1 = encode_add_batch(&[100, 200]);
        let delta2 = encode_add_batch(&[300]);

        engine.merge(Partition::Adj, key, &delta1).expect("merge 1");
        engine.merge(Partition::Adj, key, &delta2).expect("merge 2");

        // Read merged posting list
        let val = engine.get(Partition::Adj, key).expect("get adj");
        assert!(val.is_some(), "merged adj key should exist");
        let bytes = val.expect("adj bytes");
        let plist = coordinode_core::graph::edge::PostingList::from_bytes(&bytes)
            .expect("decode posting list");
        let uids: Vec<u64> = plist.iter().collect();
        assert_eq!(uids, vec![100, 200, 300]);
    }

    #[test]
    fn memfs_engine_snapshot_read() {
        let engine = test_engine_memfs();

        // Write version 1
        engine
            .put(Partition::Node, b"node:00:00000001", b"v1")
            .expect("put v1");

        // Take snapshot: next_seqno() returns the seqno for the next write.
        // Snapshot at this value sees all writes with seqno < snap (i.e., v1).
        let snap = engine.next_seqno();

        // Write version 2
        engine
            .put(Partition::Node, b"node:00:00000001", b"v2")
            .expect("put v2");

        // Current read sees v2
        let current = engine
            .get(Partition::Node, b"node:00:00000001")
            .expect("get current");
        assert_eq!(current.as_deref(), Some(b"v2".as_slice()));

        // Snapshot read sees v1
        let historical = engine
            .snapshot_get(&snap, Partition::Node, b"node:00:00000001")
            .expect("snapshot get");
        assert_eq!(historical.as_deref(), Some(b"v1".as_slice()));
    }

    #[test]
    fn open_creates_all_partitions() {
        let (engine, _dir) = test_engine();
        for &part in Partition::all() {
            assert!(engine.tree(part).is_ok(), "partition {:?} missing", part);
        }
    }

    #[test]
    fn put_get_delete_round_trip() {
        let (engine, _dir) = test_engine();
        let key = b"test_key";
        let value = b"test_value";

        // Initially missing
        assert!(engine
            .get(Partition::Node, key)
            .expect("get failed")
            .is_none());
        assert!(!engine
            .contains_key(Partition::Node, key)
            .expect("contains_key failed"));

        // Put
        engine.put(Partition::Node, key, value).expect("put failed");

        // Get
        let result = engine.get(Partition::Node, key).expect("get failed");
        assert_eq!(result.as_deref(), Some(value.as_slice()));
        assert!(engine
            .contains_key(Partition::Node, key)
            .expect("contains_key failed"));

        // Delete
        engine.delete(Partition::Node, key).expect("delete failed");
        assert!(engine
            .get(Partition::Node, key)
            .expect("get failed")
            .is_none());
    }

    #[test]
    fn partitions_are_isolated() {
        let (engine, _dir) = test_engine();
        let key = b"shared_key";

        engine
            .put(Partition::Node, key, b"node_val")
            .expect("put failed");
        engine
            .put(Partition::Schema, key, b"schema_val")
            .expect("put failed");

        let node_val = engine.get(Partition::Node, key).expect("get failed");
        let schema_val = engine.get(Partition::Schema, key).expect("get failed");
        let adj_val = engine.get(Partition::Adj, key).expect("get failed");

        assert_eq!(node_val.as_deref(), Some(b"node_val".as_slice()));
        assert_eq!(schema_val.as_deref(), Some(b"schema_val".as_slice()));
        assert!(adj_val.is_none());
    }

    #[test]
    fn overwrite_existing_key() {
        let (engine, _dir) = test_engine();
        let key = b"overwrite_me";

        engine.put(Partition::Node, key, b"v1").expect("put failed");
        engine.put(Partition::Node, key, b"v2").expect("put failed");

        let result = engine.get(Partition::Node, key).expect("get failed");
        assert_eq!(result.as_deref(), Some(b"v2".as_slice()));
    }

    #[test]
    fn delete_nonexistent_key_is_ok() {
        let (engine, _dir) = test_engine();
        engine
            .delete(Partition::Node, b"no_such_key")
            .expect("delete should succeed");
    }

    #[test]
    fn empty_value_nonempty_key() {
        let (engine, _dir) = test_engine();
        engine
            .put(Partition::Adj, b"k", b"")
            .expect("put empty value failed");

        let result = engine.get(Partition::Adj, b"k").expect("get failed");
        assert_eq!(result.as_deref(), Some(b"".as_slice()));
    }

    #[test]
    fn large_value() {
        let (engine, _dir) = test_engine();
        let key = b"big";
        // 1MB value — tests KV separation threshold behavior.
        let value = vec![0xABu8; 1024 * 1024];

        engine
            .put(Partition::Blob, key, &value)
            .expect("put large failed");

        let result = engine.get(Partition::Blob, key).expect("get large failed");
        assert_eq!(result.as_deref(), Some(value.as_slice()));
    }

    #[test]
    fn persist_and_reopen() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let config = StorageConfig::new(dir.path());

        {
            let engine = StorageEngine::open(&config).expect("open failed");
            engine
                .put(Partition::Node, b"persist_key", b"persist_val")
                .expect("put failed");
            engine.persist().expect("persist failed");
        }

        {
            let engine = StorageEngine::open(&config).expect("reopen failed");
            let result = engine
                .get(Partition::Node, b"persist_key")
                .expect("get failed");
            assert_eq!(result.as_deref(), Some(b"persist_val".as_slice()));
        }
    }

    #[test]
    fn disk_space_reports_nonzero_after_writes() {
        let (engine, _dir) = test_engine();

        for i in 0..1000u32 {
            engine
                .put(Partition::Node, &i.to_be_bytes(), &[0xFFu8; 1024])
                .expect("put failed");
        }
        engine.persist().expect("persist failed");

        let space = engine.disk_space().expect("disk_space failed");
        assert!(
            space > 0,
            "disk space should be nonzero after 1MB of writes"
        );
    }

    #[test]
    fn engine_with_no_compression() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let mut config = StorageConfig::new(dir.path());
        config.compression = crate::engine::config::CompressionConfig {
            hot_codec: crate::engine::config::CompressionCodec::None,
            cold_codec: crate::engine::config::CompressionCodec::None,
            cold_level_threshold: 4,
        };
        let engine = StorageEngine::open(&config).expect("open with no compression");
        engine
            .put(Partition::Node, b"k1", b"v1")
            .expect("put failed");
        let result = engine.get(Partition::Node, b"k1").expect("get failed");
        assert_eq!(result.as_deref(), Some(b"v1".as_slice()));
    }

    #[test]
    fn engine_with_partition_compression_overrides() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let mut config = StorageConfig::new(dir.path());
        config.partition_compression = Some(vec![(
            Partition::Blob,
            crate::engine::config::CompressionCodec::None,
        )]);
        let engine = StorageEngine::open(&config).expect("open with overrides");

        engine
            .put(Partition::Node, b"n1", b"node_data")
            .expect("put node");
        engine
            .put(Partition::Blob, b"b1", b"blob_data")
            .expect("put blob");

        assert_eq!(
            engine
                .get(Partition::Node, b"n1")
                .expect("get node")
                .as_deref(),
            Some(b"node_data".as_slice())
        );
        assert_eq!(
            engine
                .get(Partition::Blob, b"b1")
                .expect("get blob")
                .as_deref(),
            Some(b"blob_data".as_slice())
        );
    }

    #[test]
    fn engine_compressed_data_survives_reopen() {
        let dir = TempDir::new().expect("failed to create temp dir");
        let config = StorageConfig::new(dir.path());

        {
            let engine = StorageEngine::open(&config).expect("open");
            for i in 0..100u32 {
                let key = format!("key_{i:06}");
                let value = format!("value_{i:06}_payload_with_some_extra_data_for_compression");
                engine
                    .put(Partition::Node, key.as_bytes(), value.as_bytes())
                    .expect("put");
            }
            engine.persist().expect("persist");
        }

        {
            let engine = StorageEngine::open(&config).expect("reopen");
            for i in 0..100u32 {
                let key = format!("key_{i:06}");
                let expected = format!("value_{i:06}_payload_with_some_extra_data_for_compression");
                let result = engine.get(Partition::Node, key.as_bytes()).expect("get");
                assert_eq!(
                    result.as_deref(),
                    Some(expected.as_bytes()),
                    "data mismatch at key {i}"
                );
            }
        }
    }

    #[test]
    fn all_partitions_support_crud() {
        let (engine, _dir) = test_engine();

        for &part in Partition::all() {
            let key = format!("key_{:?}", part);
            let value = format!("val_{:?}", part);

            engine
                .put(part, key.as_bytes(), value.as_bytes())
                .expect("put failed");
            let result = engine.get(part, key.as_bytes()).expect("get failed");
            assert_eq!(
                result.as_deref(),
                Some(value.as_bytes()),
                "CRUD failed for partition {:?}",
                part
            );

            engine.delete(part, key.as_bytes()).expect("delete failed");
            assert!(engine
                .get(part, key.as_bytes())
                .expect("get failed")
                .is_none());
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod wal_integration_tests {
    //! Integration tests for standalone WAL crash recovery.
    //!
    //! These tests validate the full round-trip:
    //!   open_with_wal → wal_append → drop (simulated crash) → open_with_wal → verify recovery.
    //!
    //! Unlike unit tests in `wal/mod.rs` (which test the WAL file directly),
    //! these tests go through `StorageEngine::open_with_wal` and verify that
    //! data written to the WAL is visible after engine reopen.

    use super::*;
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use tempfile::TempDir;

    /// Returns a `StorageConfig` pointing at `dir` with WAL enabled (WAL path = `dir/standalone.wal`).
    fn cfg_with_wal(dir: &TempDir) -> (StorageConfig, WalConfig) {
        let config = StorageConfig::new(dir.path());
        let wal_config = WalConfig {
            path: Some(dir.path().join("standalone.wal")),
            sync: crate::wal::WalSyncPolicy::SyncPerRecord,
        };
        (config, wal_config)
    }

    #[test]
    fn wal_recovery_put_survives_crash() {
        // Verifies that a Put mutation written to the WAL (but not yet flushed to SST)
        // is recovered after the engine is dropped (simulating a crash) and reopened.
        let dir = TempDir::new().expect("temp dir");
        let (config, wal_config) = cfg_with_wal(&dir);

        // Phase 1: write to WAL, then "crash" (drop without persist).
        {
            let engine =
                StorageEngine::open_with_wal(&config, Some(wal_config.clone())).expect("open");
            assert!(engine.has_wal());

            // Write through WAL — must NOT call persist() so data stays only in WAL.
            let mutations = vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:00000042".to_vec(),
                value: b"crashed-payload".to_vec(),
            }];
            engine.wal_append(&mutations).expect("wal_append");

            // Apply to memtable (mirrors what OwnedLocalProposalPipeline does).
            engine
                .put(Partition::Node, b"node:00:00000042", b"crashed-payload")
                .expect("put");

            // Drop engine WITHOUT calling persist() — simulates crash.
            // Data is in memtable + WAL but NOT in any SST.
        }

        // Phase 2: reopen with WAL — recovery must replay the WAL record.
        {
            let engine = StorageEngine::open_with_wal(&config, Some(wal_config)).expect("reopen");

            let val = engine
                .get(Partition::Node, b"node:00:00000042")
                .expect("get after recovery");

            assert_eq!(
                val.as_deref(),
                Some(b"crashed-payload".as_slice()),
                "WAL recovery must restore Put written before crash"
            );
        }
    }

    #[test]
    fn wal_recovery_delete_survives_crash() {
        // Verifies that a Delete mutation in the WAL (after a prior Put in SST)
        // is replayed correctly: the key must be absent after recovery.
        let dir = TempDir::new().expect("temp dir");
        let (config, wal_config) = cfg_with_wal(&dir);

        // Phase 1: persist a key to SST.
        {
            let engine =
                StorageEngine::open_with_wal(&config, Some(wal_config.clone())).expect("open");
            engine
                .put(Partition::Node, b"node:00:deadbeef", b"to-be-deleted")
                .expect("put");
            engine.persist().expect("persist");
            // persist() also checkpoints the WAL — WAL is now clean.
        }

        // Phase 2: delete via WAL, then crash.
        {
            let engine =
                StorageEngine::open_with_wal(&config, Some(wal_config.clone())).expect("open2");

            let mutations = vec![Mutation::Delete {
                partition: PartitionId::Node,
                key: b"node:00:deadbeef".to_vec(),
            }];
            engine.wal_append(&mutations).expect("wal_append");
            engine
                .delete(Partition::Node, b"node:00:deadbeef")
                .expect("delete");
            // Crash — no persist().
        }

        // Phase 3: reopen — delete must be replayed, key must be gone.
        {
            let engine = StorageEngine::open_with_wal(&config, Some(wal_config)).expect("reopen");

            let val = engine
                .get(Partition::Node, b"node:00:deadbeef")
                .expect("get after recovery");

            assert!(
                val.is_none(),
                "WAL recovery must replay Delete: key must be absent after crash"
            );
        }
    }

    #[test]
    fn wal_recovery_multiple_mutations_across_two_writes() {
        // Writes multiple WAL records (two separate wal_append calls), crashes,
        // reopens, and verifies ALL records are recovered in order.
        let dir = TempDir::new().expect("temp dir");
        let (config, wal_config) = cfg_with_wal(&dir);

        {
            let engine =
                StorageEngine::open_with_wal(&config, Some(wal_config.clone())).expect("open");

            // First WAL record: two puts.
            let batch1 = vec![
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:00:00000001".to_vec(),
                    value: b"alpha".to_vec(),
                },
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:00:00000002".to_vec(),
                    value: b"beta".to_vec(),
                },
            ];
            engine.wal_append(&batch1).expect("wal_append batch1");
            for m in &batch1 {
                if let Mutation::Put {
                    partition,
                    key,
                    value,
                } = m
                {
                    engine
                        .put(Partition::from(*partition), key, value)
                        .expect("put");
                }
            }

            // Second WAL record: one more put.
            let batch2 = vec![Mutation::Put {
                partition: PartitionId::Schema,
                key: b"schema:label:Person".to_vec(),
                value: b"{}".to_vec(),
            }];
            engine.wal_append(&batch2).expect("wal_append batch2");
            engine
                .put(Partition::Schema, b"schema:label:Person", b"{}")
                .expect("put schema");

            // Crash.
        }

        // Reopen and verify all three keys recovered.
        {
            let engine = StorageEngine::open_with_wal(&config, Some(wal_config)).expect("reopen");

            let v1 = engine
                .get(Partition::Node, b"node:00:00000001")
                .expect("get 1");
            let v2 = engine
                .get(Partition::Node, b"node:00:00000002")
                .expect("get 2");
            let v3 = engine
                .get(Partition::Schema, b"schema:label:Person")
                .expect("get 3");

            assert_eq!(
                v1.as_deref(),
                Some(b"alpha".as_slice()),
                "key 1 must recover"
            );
            assert_eq!(
                v2.as_deref(),
                Some(b"beta".as_slice()),
                "key 2 must recover"
            );
            assert_eq!(v3.as_deref(), Some(b"{}".as_slice()), "key 3 must recover");
        }
    }

    #[test]
    fn wal_checkpoint_on_persist_clears_wal_file() {
        // Verifies that calling persist() on an engine with WAL:
        //   1. Flushes memtable to SST.
        //   2. Checkpoints (rotates) the WAL so the file is empty.
        // After reopen, no replay happens (WAL is clean) and data is still readable from SST.
        let dir = TempDir::new().expect("temp dir");
        let (config, wal_config) = cfg_with_wal(&dir);

        {
            let engine =
                StorageEngine::open_with_wal(&config, Some(wal_config.clone())).expect("open");

            let mutations = vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:persisted".to_vec(),
                value: b"safe".to_vec(),
            }];
            engine.wal_append(&mutations).expect("wal_append");
            engine
                .put(Partition::Node, b"node:00:persisted", b"safe")
                .expect("put");

            // persist() must flush to SST AND checkpoint the WAL.
            engine.persist().expect("persist");

            // WAL file must be empty (or minimal header) after checkpoint.
            let wal_path = dir.path().join("standalone.wal");
            let wal_size = std::fs::metadata(&wal_path)
                .expect("wal file must exist after persist")
                .len();
            assert_eq!(
                wal_size, 0,
                "WAL must be empty after checkpoint via persist()"
            );
        }

        // Reopen: no WAL replay needed (SST has the data), key still readable.
        {
            let engine = StorageEngine::open_with_wal(&config, Some(wal_config)).expect("reopen");

            let val = engine
                .get(Partition::Node, b"node:00:persisted")
                .expect("get");
            assert_eq!(
                val.as_deref(),
                Some(b"safe".as_slice()),
                "data persisted to SST must survive WAL checkpoint + reopen"
            );
        }
    }

    #[test]
    fn no_wal_engine_wal_append_returns_none() {
        // Verifies that wal_append() on a plain open() engine (no WAL) returns None
        // without error — caller can use this to branch between WAL and legacy paths.
        let dir = TempDir::new().expect("temp dir");
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open without wal");

        assert!(!engine.has_wal(), "plain open must have no WAL");

        let mutations = vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:00:noop".to_vec(),
            value: b"x".to_vec(),
        }];
        let result = engine
            .wal_append(&mutations)
            .expect("wal_append must not error without wal");
        assert!(result.is_none(), "wal_append without WAL must return None");
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod merge_tests {
    use super::*;
    use crate::engine::merge::{encode_add, encode_add_batch, encode_remove};
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
    use coordinode_core::graph::edge::PostingList;
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;
    use tempfile::TempDir;

    fn test_engine() -> (StorageEngine, TempDir) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("failed to open engine");
        (engine, dir)
    }

    #[test]
    fn merge_add_produces_sorted_posting_list() {
        let (engine, _dir) = test_engine();
        let key = b"adj:FOLLOWS:out:42";

        engine
            .merge(Partition::Adj, key, &encode_add(30))
            .expect("merge 30");
        engine
            .merge(Partition::Adj, key, &encode_add(10))
            .expect("merge 10");
        engine
            .merge(Partition::Adj, key, &encode_add(20))
            .expect("merge 20");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get failed")
            .expect("key should exist");
        let plist = PostingList::from_bytes(&data).expect("decode failed");
        assert_eq!(plist.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn merge_add_and_remove_interleaved() {
        let (engine, _dir) = test_engine();
        let key = b"adj:KNOWS:out:1";

        engine
            .merge(Partition::Adj, key, &encode_add(100))
            .expect("add 100");
        engine
            .merge(Partition::Adj, key, &encode_add(200))
            .expect("add 200");
        engine
            .merge(Partition::Adj, key, &encode_add(300))
            .expect("add 300");
        engine
            .merge(Partition::Adj, key, &encode_remove(200))
            .expect("remove 200");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.as_slice(), &[100, 300]);
    }

    #[test]
    fn merge_batch_add() {
        let (engine, _dir) = test_engine();
        let key = b"adj:RATED:out:5";

        engine
            .merge(Partition::Adj, key, &encode_add_batch(&[50, 10, 30]))
            .expect("batch add");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.as_slice(), &[10, 30, 50]);
    }

    #[test]
    fn merge_on_existing_put_base() {
        let (engine, _dir) = test_engine();
        let key = b"adj:FOLLOWS:out:7";

        let base = PostingList::from_sorted(vec![1, 5, 10]);
        engine
            .put(Partition::Adj, key, &base.to_bytes().expect("encode"))
            .expect("put base");

        engine
            .merge(Partition::Adj, key, &encode_add(7))
            .expect("merge add");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.as_slice(), &[1, 5, 7, 10]);
    }

    #[test]
    fn merge_survives_persist_and_reopen() {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let key = b"adj:FOLLOWS:out:99";

        {
            let engine = StorageEngine::open(&config).expect("open");
            engine
                .merge(Partition::Adj, key, &encode_add(1))
                .expect("add 1");
            engine
                .merge(Partition::Adj, key, &encode_add(2))
                .expect("add 2");
            engine.persist().expect("persist");
        }

        {
            let engine = StorageEngine::open(&config).expect("reopen");
            let data = engine
                .get(Partition::Adj, key)
                .expect("get")
                .expect("should exist after reopen");
            let plist = PostingList::from_bytes(&data).expect("decode");
            assert_eq!(plist.as_slice(), &[1, 2]);
        }
    }

    #[test]
    fn merge_many_sequential_adds() {
        let (engine, _dir) = test_engine();
        let key = b"adj:FOLLOWS:out:hub";

        for uid in 0..1000u64 {
            engine
                .merge(Partition::Adj, key, &encode_add(uid))
                .expect("merge");
        }

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.len(), 1000);
        let slice = plist.as_slice();
        for i in 1..slice.len() {
            assert!(
                slice[i - 1] < slice[i],
                "not sorted at index {i}: {} >= {}",
                slice[i - 1],
                slice[i]
            );
        }
    }

    // -- R010b: edge cases + concurrent tests --

    #[test]
    fn merge_concurrent_multithreaded_adds() {
        use std::sync::Arc;
        use std::thread;

        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let key = b"adj:FOLLOWS:out:celebrity";

        let num_threads = 8;
        let uids_per_thread = 500;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let engine = Arc::clone(&engine);
                thread::spawn(move || {
                    for i in 0..uids_per_thread {
                        let uid = (t * uids_per_thread + i) as u64;
                        engine
                            .merge(Partition::Adj, key, &encode_add(uid))
                            .expect("merge failed");
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        engine.persist().expect("persist");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");

        let total = (num_threads * uids_per_thread) as usize;
        assert_eq!(
            plist.len(),
            total,
            "expected {total} UIDs, got {}",
            plist.len()
        );

        let slice = plist.as_slice();
        for i in 1..slice.len() {
            assert!(
                slice[i - 1] < slice[i],
                "not sorted at index {i}: {} >= {}",
                slice[i - 1],
                slice[i]
            );
        }
    }

    #[test]
    fn merge_compaction_correctness() {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open");
        let key = b"adj:RATED:out:99";

        for uid in (0..100u64).rev() {
            engine
                .merge(Partition::Adj, key, &encode_add(uid))
                .expect("merge");
        }

        engine
            .force_compaction(Partition::Adj)
            .expect("force compaction");

        for uid in 100..200u64 {
            engine
                .merge(Partition::Adj, key, &encode_add(uid))
                .expect("merge phase 2");
        }

        engine
            .merge(Partition::Adj, key, &encode_remove(50))
            .expect("remove 50");
        engine
            .merge(Partition::Adj, key, &encode_remove(75))
            .expect("remove 75");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");

        assert_eq!(plist.len(), 198);
        assert!(!plist.contains(50));
        assert!(!plist.contains(75));
        assert!(plist.contains(0));
        assert!(plist.contains(199));

        let slice = plist.as_slice();
        for i in 1..slice.len() {
            assert!(slice[i - 1] < slice[i]);
        }
    }

    #[test]
    fn merge_compaction_reopen_correctness() {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let key = b"adj:KNOWS:out:77";

        {
            let engine = StorageEngine::open(&config).expect("open");
            engine
                .merge(Partition::Adj, key, &encode_add(10))
                .expect("add 10");
            engine
                .merge(Partition::Adj, key, &encode_add(20))
                .expect("add 20");
            engine
                .merge(Partition::Adj, key, &encode_add(30))
                .expect("add 30");

            engine
                .force_compaction(Partition::Adj)
                .expect("force compaction");

            engine
                .merge(Partition::Adj, key, &encode_add(5))
                .expect("add 5");
            engine
                .merge(Partition::Adj, key, &encode_remove(20))
                .expect("remove 20");
            engine.persist().expect("persist");
        }

        {
            let engine = StorageEngine::open(&config).expect("reopen");
            let data = engine
                .get(Partition::Adj, key)
                .expect("get")
                .expect("should exist");
            let plist = PostingList::from_bytes(&data).expect("decode");
            assert_eq!(plist.as_slice(), &[5, 10, 30]);
        }
    }

    #[test]
    fn merge_empty_base_remove_is_noop() {
        let (engine, _dir) = test_engine();
        let key = b"adj:X:out:1";

        engine
            .merge(Partition::Adj, key, &encode_remove(42))
            .expect("remove on empty");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert!(
            plist.is_empty(),
            "remove on empty base should produce empty list"
        );
    }

    #[test]
    fn merge_add_remove_same_uid_in_sequence() {
        let (engine, _dir) = test_engine();
        let key = b"adj:Y:out:2";

        engine
            .merge(Partition::Adj, key, &encode_add(100))
            .expect("add");
        engine
            .merge(Partition::Adj, key, &encode_remove(100))
            .expect("remove");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert!(plist.is_empty());
    }

    #[test]
    fn merge_remove_then_add_same_uid() {
        let (engine, _dir) = test_engine();
        let key = b"adj:Z:out:3";

        engine
            .merge(Partition::Adj, key, &encode_remove(42))
            .expect("remove first");
        engine
            .merge(Partition::Adj, key, &encode_add(42))
            .expect("add after");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.as_slice(), &[42]);
    }

    #[test]
    fn merge_batch_with_duplicates_is_idempotent() {
        let (engine, _dir) = test_engine();
        let key = b"adj:W:out:4";

        engine
            .merge(Partition::Adj, key, &encode_add_batch(&[5, 3, 5, 1, 3, 1]))
            .expect("batch with dupes");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.as_slice(), &[1, 3, 5]);
    }

    #[test]
    fn merge_double_compaction_partial_re_merge() {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open");
        let key = b"adj:FOLLOWS:out:hub";

        engine
            .merge(Partition::Adj, key, &encode_add(10))
            .expect("add 10");
        engine
            .merge(Partition::Adj, key, &encode_add(20))
            .expect("add 20");

        engine
            .force_compaction(Partition::Adj)
            .expect("first compaction");

        engine
            .merge(Partition::Adj, key, &encode_add(15))
            .expect("add 15");

        engine
            .force_compaction(Partition::Adj)
            .expect("second compaction");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.as_slice(), &[10, 15, 20]);
    }

    // ================================================================
    // R010d: Merge operator stress + time-travel tests
    // ================================================================

    #[test]
    fn merge_stress_concurrent_writers_zero_conflict() {
        use std::sync::Arc;
        use std::thread;

        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let key = b"adj:FOLLOWS:out:supernode";

        let num_writers = 50;
        let edges_per_writer = 1_000;
        let total_uids = num_writers * edges_per_writer;

        let handles: Vec<_> = (0..num_writers)
            .map(|writer_id| {
                let engine = Arc::clone(&engine);
                thread::spawn(move || {
                    let base = writer_id * edges_per_writer;
                    let uids: Vec<u64> =
                        (base..base + edges_per_writer).map(|x| x as u64).collect();
                    engine
                        .merge(Partition::Adj, key, &encode_add_batch(&uids))
                        .expect("merge should never fail — commutative ops");
                })
            })
            .collect();

        for h in handles {
            h.join().expect("writer thread panicked");
        }

        engine.persist().expect("persist");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("posting list should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");

        assert_eq!(
            plist.len(),
            total_uids,
            "expected {} UIDs, got {} — some merge operands lost",
            total_uids,
            plist.len()
        );

        let slice = plist.as_slice();
        for i in 1..slice.len() {
            assert!(
                slice[i - 1] < slice[i],
                "not sorted at index {}: {} >= {}",
                i,
                slice[i - 1],
                slice[i]
            );
        }

        assert_eq!(slice[0], 0);
        assert_eq!(slice[slice.len() - 1], (total_uids - 1) as u64);
    }

    #[test]
    fn merge_time_travel_snapshot_correctness() {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open");
        let key = b"adj:RATES:out:user42";

        let batches = 10;
        let uids_per_batch = 100;
        let mut snapshots: Vec<lsm_tree::SeqNo> = Vec::with_capacity(batches);
        let mut expected_uids_at_snapshot: Vec<Vec<u64>> = Vec::with_capacity(batches);

        for batch_idx in 0..batches {
            let base = batch_idx * uids_per_batch;
            for i in 0..uids_per_batch {
                let uid = (base + i) as u64;
                engine
                    .merge(Partition::Adj, key, &encode_add(uid))
                    .expect("merge");
            }

            engine.persist().expect("persist");

            // Take a seqno snapshot after this batch.
            let snap = engine.snapshot();
            snapshots.push(snap);

            let expected: Vec<u64> = (0..((batch_idx + 1) * uids_per_batch) as u64).collect();
            expected_uids_at_snapshot.push(expected);
        }

        // Write more operands AFTER the last snapshot.
        for uid in 1000..1050u64 {
            engine
                .merge(Partition::Adj, key, &encode_add(uid))
                .expect("merge post-snapshot");
        }
        engine.persist().expect("persist");

        // Verify each snapshot sees only the UIDs written before it.
        for (snap_idx, snap) in snapshots.iter().enumerate() {
            let data = engine
                .snapshot_get(snap, Partition::Adj, key)
                .expect("snapshot get")
                .expect("posting list should exist in snapshot");
            let plist = PostingList::from_bytes(&data).expect("decode");

            assert_eq!(
                plist.as_slice(),
                expected_uids_at_snapshot[snap_idx].as_slice(),
                "snapshot {} sees wrong UIDs: expected {} UIDs, got {}",
                snap_idx,
                expected_uids_at_snapshot[snap_idx].len(),
                plist.len()
            );
        }

        // Current read sees ALL UIDs (1000 + 50 post-snapshot).
        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.len(), 1050, "current read should see all 1050 UIDs");
    }

    #[test]
    fn merge_compaction_preserves_time_travel() {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open");
        let key = b"adj:KNOWS:out:node7";

        // Phase 1: Write 200 UIDs in 2 batches, snapshot after each.
        for uid in 0..100u64 {
            engine
                .merge(Partition::Adj, key, &encode_add(uid))
                .expect("merge batch 1");
        }
        engine.persist().expect("persist");
        let snap_after_100 = engine.snapshot();

        for uid in 100..200u64 {
            engine
                .merge(Partition::Adj, key, &encode_add(uid))
                .expect("merge batch 2");
        }
        engine.persist().expect("persist");
        let snap_after_200 = engine.snapshot();

        // Phase 2: Force compaction.
        engine.force_compaction(Partition::Adj).expect("compaction");

        // Phase 3: Write 50 more UIDs after compaction.
        for uid in 200..250u64 {
            engine
                .merge(Partition::Adj, key, &encode_add(uid))
                .expect("merge batch 3");
        }
        engine.persist().expect("persist");
        let snap_after_250 = engine.snapshot();

        // Verify pre-compaction snapshots.
        let data_100 = engine
            .snapshot_get(&snap_after_100, Partition::Adj, key)
            .expect("get snap 100")
            .expect("should exist");
        let plist_100 = PostingList::from_bytes(&data_100).expect("decode");
        assert_eq!(
            plist_100.len(),
            100,
            "snap_after_100 should see 100 UIDs, got {}",
            plist_100.len()
        );

        let data_200 = engine
            .snapshot_get(&snap_after_200, Partition::Adj, key)
            .expect("get snap 200")
            .expect("should exist");
        let plist_200 = PostingList::from_bytes(&data_200).expect("decode");
        assert_eq!(
            plist_200.len(),
            200,
            "snap_after_200 should see 200 UIDs, got {}",
            plist_200.len()
        );

        // Post-compaction snapshot sees compacted base + new batch.
        let data_250 = engine
            .snapshot_get(&snap_after_250, Partition::Adj, key)
            .expect("get snap 250")
            .expect("should exist");
        let plist_250 = PostingList::from_bytes(&data_250).expect("decode");
        assert_eq!(
            plist_250.len(),
            250,
            "snap_after_250 should see 250 UIDs, got {}",
            plist_250.len()
        );

        // Current read sees everything.
        let data_all = engine
            .get(Partition::Adj, key)
            .expect("get current")
            .expect("should exist");
        let plist_all = PostingList::from_bytes(&data_all).expect("decode");
        assert_eq!(plist_all.len(), 250, "current should see 250 UIDs");

        // Verify all posting lists are sorted.
        for (name, plist) in [
            ("snap_100", &plist_100),
            ("snap_200", &plist_200),
            ("snap_250", &plist_250),
            ("current", &plist_all),
        ] {
            let slice = plist.as_slice();
            for i in 1..slice.len() {
                assert!(
                    slice[i - 1] < slice[i],
                    "{name}: not sorted at {i}: {} >= {}",
                    slice[i - 1],
                    slice[i]
                );
            }
        }
    }

    #[test]
    fn merge_stress_interleaved_add_remove_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let key = b"adj:LIKES:out:user99";

        let batch: Vec<u64> = (0..1000).collect();
        engine
            .merge(Partition::Adj, key, &encode_add_batch(&batch))
            .expect("batch add");
        engine.persist().expect("persist");

        let num_threads = 20;
        let ops_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let engine = Arc::clone(&engine);
                thread::spawn(move || {
                    if t % 2 == 0 {
                        let base = 1000 + (t / 2) * ops_per_thread;
                        let uids: Vec<u64> =
                            (base..base + ops_per_thread).map(|x| x as u64).collect();
                        engine
                            .merge(Partition::Adj, key, &encode_add_batch(&uids))
                            .expect("batch add");
                    } else {
                        let base = (t / 2) * ops_per_thread;
                        let end = std::cmp::min(base + ops_per_thread, 1000);
                        for uid in base..end {
                            engine
                                .merge(Partition::Adj, key, &encode_remove(uid as u64))
                                .expect("remove");
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        engine.persist().expect("persist");

        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");

        let slice = plist.as_slice();
        for i in 1..slice.len() {
            assert!(
                slice[i - 1] < slice[i],
                "not sorted at {i}: {} >= {}",
                slice[i - 1],
                slice[i]
            );
        }

        let added_count = (num_threads / 2) * ops_per_thread;
        assert_eq!(
            plist.len(),
            added_count,
            "expected {} UIDs (original 1000 removed, {} added)",
            added_count,
            added_count
        );
    }

    // === G049: UidPack format verification ===

    #[test]
    fn merge_stores_uidpack_format_on_disk() {
        // Verify that after merge operations, the stored bytes are
        // valid UidPack (not raw Vec<u64> MessagePack).
        let (engine, _dir) = test_engine();

        let key = b"adj:TEST:out:\x00\x00\x00\x00\x00\x00\x00\x01";

        // Write 300 UIDs via merge (forces multiple UidBlocks at 256/block).
        let uids: Vec<u64> = (1..=300).collect();
        let operand = crate::engine::merge::encode_add_batch(&uids);
        engine
            .merge(Partition::Adj, key, &operand)
            .expect("merge 300 UIDs");

        // Read back raw bytes and verify UidPack structure.
        let raw = engine
            .get(Partition::Adj, key)
            .expect("get failed")
            .expect("should exist");
        let pack: coordinode_core::graph::codec::UidPack =
            rmp_serde::from_slice(&raw).expect("stored bytes must be valid UidPack");
        assert_eq!(pack.total_uids(), 300);
        assert!(
            pack.blocks.len() >= 2,
            "300 UIDs should produce ≥2 blocks, got {}",
            pack.blocks.len()
        );

        // Verify UIDs are correct via PostingList decode.
        let plist = PostingList::from_bytes(&raw).expect("decode");
        assert_eq!(plist.len(), 300);
        assert_eq!(plist.as_slice()[0], 1);
        assert_eq!(plist.as_slice()[299], 300);

        // Verify compression: UidPack should be smaller than raw Vec<u64>.
        let raw_vec_size = rmp_serde::to_vec(&uids).expect("raw").len();
        assert!(
            raw.len() < raw_vec_size,
            "UidPack ({} bytes) should be smaller than raw Vec<u64> ({} bytes)",
            raw.len(),
            raw_vec_size
        );
    }

    // === R061: snapshot_at integration tests ===

    #[test]
    fn snapshot_at_reads_historical_value() {
        let (engine, _dir) = test_engine();
        let key = b"snap_test_key";

        engine.put(Partition::Node, key, b"v1").expect("put v1");
        engine.persist().expect("persist");
        let seqno_after_v1 = engine.snapshot();

        engine.put(Partition::Node, key, b"v2").expect("put v2");
        engine.persist().expect("persist");

        // Current read sees v2.
        let current = engine.get(Partition::Node, key).expect("get");
        assert_eq!(current.as_deref(), Some(b"v2".as_ref()));

        // Historical snapshot at seqno_after_v1 sees v1.
        let historical = engine
            .snapshot_get(&seqno_after_v1, Partition::Node, key)
            .expect("snap get");
        assert_eq!(historical.as_deref(), Some(b"v1".as_ref()));
    }

    #[test]
    fn snapshot_at_future_returns_some() {
        // snapshot_at always returns Some in the native seqno design.
        let (engine, _dir) = test_engine();
        assert!(engine.snapshot_at(u64::MAX).is_some());
    }

    #[test]
    fn snapshot_at_isolation_from_concurrent_writes() {
        let (engine, _dir) = test_engine();

        for i in 0..10u32 {
            engine
                .put(Partition::Node, format!("k{i}").as_bytes(), b"old")
                .expect("put");
        }
        engine.persist().expect("persist");

        let snap_seqno = engine.snapshot();

        for i in 0..10u32 {
            engine
                .put(Partition::Node, format!("k{i}").as_bytes(), b"new")
                .expect("put");
        }
        engine.persist().expect("persist");

        // Snapshot reads must see "old", not "new".
        for i in 0..10u32 {
            let val = engine
                .snapshot_get(&snap_seqno, Partition::Node, format!("k{i}").as_bytes())
                .expect("snap get");
            assert_eq!(
                val.as_deref(),
                Some(b"old".as_ref()),
                "snapshot isolation violated for k{i}"
            );
        }

        // Current reads see "new".
        for i in 0..10u32 {
            let val = engine
                .get(Partition::Node, format!("k{i}").as_bytes())
                .expect("get");
            assert_eq!(val.as_deref(), Some(b"new".as_ref()));
        }
    }

    // === R064: TimestampOracle as SequenceNumberGenerator ===

    #[test]
    fn open_with_oracle_writes_use_oracle_seqno() {
        use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
        use std::sync::Arc;

        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());

        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
        let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");

        for i in 0..10u32 {
            engine
                .put(Partition::Node, format!("k{i}").as_bytes(), b"val")
                .expect("put");
        }
        engine.persist().expect("persist");

        let current = oracle.current().as_raw();
        assert!(
            current >= 1010,
            "oracle should be ≥1010 after 10 writes, got {current}"
        );

        let snap = engine.snapshot();
        assert!(snap >= 1010, "snapshot seqno should be ≥1010, got {snap}");
    }

    #[test]
    fn oracle_snapshot_at_matches_write_timestamp() {
        use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
        use std::sync::Arc;

        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");

        engine.put(Partition::Node, b"key", b"v1").expect("put");
        engine.persist().expect("persist");
        let seqno_v1 = engine.snapshot();

        engine.put(Partition::Node, b"key", b"v2").expect("put");
        engine.persist().expect("persist");

        // Snapshot at seqno_v1 should see v1.
        let val = engine
            .snapshot_get(&seqno_v1, Partition::Node, b"key")
            .expect("snap get");
        assert_eq!(val.as_deref(), Some(b"v1".as_ref()));

        assert!(
            seqno_v1 > 100,
            "seqno should be oracle-driven (>100), got {seqno_v1}"
        );

        // Current read sees v2.
        let current = engine.get(Partition::Node, b"key").expect("get");
        assert_eq!(current.as_deref(), Some(b"v2".as_ref()));
    }

    // === R066: has_write_after — seqno-based OCC conflict detection ===

    #[test]
    fn has_write_after_detects_newer_write() {
        use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
        use std::sync::Arc;

        let dir = TempDir::new().expect("tempdir");
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");

        engine
            .put(Partition::Node, b"node:1:1", b"v1")
            .expect("put");
        let write_seqno = engine.snapshot();

        assert!(
            engine
                .has_write_after(Partition::Node, b"node:1:1", 99)
                .expect("check"),
            "should detect write after seqno 99"
        );

        assert!(
            !engine
                .has_write_after(Partition::Node, b"node:1:1", write_seqno)
                .expect("check"),
            "should not detect write at or before current seqno"
        );
    }

    #[test]
    fn has_write_after_nonexistent_key_returns_false() {
        use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
        use std::sync::Arc;

        let dir = TempDir::new().expect("tempdir");
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open_with_oracle(&config, oracle).expect("open");

        assert!(
            !engine
                .has_write_after(Partition::Node, b"nonexistent", 0)
                .expect("check"),
            "nonexistent key should return false"
        );
    }

    #[test]
    fn has_write_after_detects_delete_tombstone() {
        use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
        use std::sync::Arc;

        let dir = TempDir::new().expect("tempdir");
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");

        engine
            .put(Partition::Node, b"node:1:2", b"data")
            .expect("put");
        let seqno_after_put = engine.snapshot();

        engine.delete(Partition::Node, b"node:1:2").expect("delete");

        assert!(
            engine
                .has_write_after(Partition::Node, b"node:1:2", seqno_after_put)
                .expect("check"),
            "should detect delete tombstone as a write"
        );
    }

    #[test]
    fn has_write_after_different_partitions_independent() {
        use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
        use std::sync::Arc;

        let dir = TempDir::new().expect("tempdir");
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open_with_oracle(&config, oracle).expect("open");

        engine
            .put(Partition::Node, b"node:1:1", b"data")
            .expect("put");

        assert!(
            !engine
                .has_write_after(Partition::Schema, b"node:1:1", 0)
                .expect("check"),
            "write in Node partition should not affect Schema partition"
        );
    }

    // === R163: Document merge operator integration tests ===

    #[test]
    fn doc_merge_through_storage_engine() {
        // Verify that DocDelta merge operands written via engine.merge()
        // are correctly combined when read back via engine.get().
        use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
        use coordinode_core::graph::node::NodeRecord;
        use coordinode_core::graph::types::Value;

        let (engine, _dir) = test_engine();
        let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01";

        // Write base NodeRecord with 0x00 prefix via PUT.
        let mut rec = NodeRecord::new("Device");
        rec.set_extra("config", Value::Document(rmpv::Value::Map(vec![])));
        let base_msgpack = rec.to_msgpack().expect("encode");
        let mut base = Vec::with_capacity(1 + base_msgpack.len());
        base.push(PREFIX_NODE_RECORD);
        base.extend_from_slice(&base_msgpack);
        engine.put(Partition::Node, key, &base).expect("put base");

        // Write DocDelta merge operand: set config.ssid = "home".
        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["config".into(), "ssid".into()],
            value: rmpv::Value::String("home".into()),
        };
        let operand = delta.encode().expect("encode delta");
        engine
            .merge(Partition::Node, key, &operand)
            .expect("merge delta");

        // Read back: engine should merge base + delta transparently.
        let data = engine
            .get(Partition::Node, key)
            .expect("get")
            .expect("should exist");

        // Decode the merged result (strip 0x00 prefix).
        assert_eq!(data[0], PREFIX_NODE_RECORD);
        let merged = NodeRecord::from_msgpack(&data[1..]).expect("decode merged");

        // Labels preserved.
        assert!(merged.has_label("Device"));

        // DocDelta applied: config.ssid = "home".
        let config = merged.get_extra("config").expect("config key");
        if let Value::Document(doc) = config {
            let ssid = coordinode_core::graph::document::extract_at_path(doc, &["ssid"]);
            assert_eq!(ssid, rmpv::Value::String("home".into()));
        } else {
            panic!("expected Document, got {config:?}");
        }
    }

    #[test]
    fn doc_merge_multiple_deltas_through_engine() {
        // Multiple merge operands on same key — all applied in order.
        use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
        use coordinode_core::graph::node::NodeRecord;
        use coordinode_core::graph::types::Value;

        let (engine, _dir) = test_engine();
        let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02";

        // No base — merge operands create record from scratch.
        let deltas = vec![
            DocDelta::SetPath {
                target: PathTarget::Extra,
                path: vec!["name".into()],
                value: rmpv::Value::String("sensor-1".into()),
            },
            DocDelta::SetPath {
                target: PathTarget::Extra,
                path: vec!["status".into()],
                value: rmpv::Value::String("active".into()),
            },
            DocDelta::Increment {
                target: PathTarget::Extra,
                path: vec!["readings".into()],
                amount: 1.0,
            },
            DocDelta::Increment {
                target: PathTarget::Extra,
                path: vec!["readings".into()],
                amount: 1.0,
            },
            DocDelta::Increment {
                target: PathTarget::Extra,
                path: vec!["readings".into()],
                amount: 1.0,
            },
        ];

        for delta in &deltas {
            let operand = delta.encode().expect("encode");
            engine.merge(Partition::Node, key, &operand).expect("merge");
        }

        let data = engine
            .get(Partition::Node, key)
            .expect("get")
            .expect("should exist");

        assert_eq!(data[0], PREFIX_NODE_RECORD);
        let rec = NodeRecord::from_msgpack(&data[1..]).expect("decode");

        assert_eq!(
            rec.get_extra("name"),
            Some(&Value::String("sensor-1".into()))
        );
        assert_eq!(
            rec.get_extra("status"),
            Some(&Value::String("active".into()))
        );
        assert_eq!(rec.get_extra("readings"), Some(&Value::Int(3)));
    }

    #[test]
    fn doc_merge_survives_persist_and_reopen() {
        // Write merge operands → persist → reopen → verify merged result.
        use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
        use coordinode_core::graph::node::NodeRecord;
        use coordinode_core::graph::types::Value;

        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03";

        {
            let engine = StorageEngine::open(&config).expect("open");

            // Write base + delta.
            let rec = NodeRecord::new("Config");
            let base_msgpack = rec.to_msgpack().expect("encode");
            let mut base = Vec::with_capacity(1 + base_msgpack.len());
            base.push(PREFIX_NODE_RECORD);
            base.extend_from_slice(&base_msgpack);
            engine.put(Partition::Node, key, &base).expect("put");

            let delta = DocDelta::SetPath {
                target: PathTarget::Extra,
                path: vec!["version".into()],
                value: rmpv::Value::Integer(42.into()),
            };
            engine
                .merge(Partition::Node, key, &delta.encode().expect("enc"))
                .expect("merge");

            engine.persist().expect("persist");
        }

        // Reopen and verify.
        {
            let engine = StorageEngine::open(&config).expect("reopen");
            let data = engine
                .get(Partition::Node, key)
                .expect("get")
                .expect("should exist after reopen");

            assert_eq!(data[0], PREFIX_NODE_RECORD);
            let rec = NodeRecord::from_msgpack(&data[1..]).expect("decode");
            assert!(rec.has_label("Config"));
            assert_eq!(rec.get_extra("version"), Some(&Value::Int(42)));
        }
    }

    #[test]
    fn doc_merge_concurrent_different_paths_no_conflict() {
        // Multiple threads writing merge operands to different paths on same key.
        // All operands should be merged correctly — no OCC conflicts.
        use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
        use coordinode_core::graph::node::NodeRecord;
        use coordinode_core::graph::types::Value;
        use std::sync::Arc;

        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x04";

        let num_threads = 8;
        let ops_per_thread = 50;

        let mut handles = Vec::new();
        for thread_id in 0..num_threads {
            let engine = Arc::clone(&engine);
            let key = key.to_vec();
            handles.push(std::thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let path_name = format!("t{thread_id}_field{i}");
                    let delta = DocDelta::SetPath {
                        target: PathTarget::Extra,
                        path: vec![path_name],
                        value: rmpv::Value::Integer((thread_id * 1000 + i).into()),
                    };
                    let operand = delta.encode().expect("encode");
                    engine
                        .merge(Partition::Node, &key, &operand)
                        .expect("merge");
                }
            }));
        }

        for h in handles {
            h.join().expect("thread join");
        }

        let data = engine
            .get(Partition::Node, key)
            .expect("get")
            .expect("should exist");

        assert_eq!(data[0], PREFIX_NODE_RECORD);
        let rec = NodeRecord::from_msgpack(&data[1..]).expect("decode");

        // Verify all fields from all threads are present.
        let extra = rec.extra.as_ref().expect("extra should exist");
        let expected_count = num_threads * ops_per_thread;
        assert_eq!(
            extra.len(),
            expected_count as usize,
            "expected {} fields, got {}",
            expected_count,
            extra.len()
        );

        // Spot check: thread 0, field 0.
        assert_eq!(rec.get_extra("t0_field0"), Some(&Value::Int(0)));
        // Spot check: last thread, last field.
        let last_t = num_threads - 1;
        let last_f = ops_per_thread - 1;
        assert_eq!(
            rec.get_extra(&format!("t{last_t}_field{last_f}")),
            Some(&Value::Int((last_t * 1000 + last_f) as i64))
        );
    }

    #[test]
    fn doc_merge_concurrent_increment_same_path() {
        // Multiple threads incrementing the same counter via merge operands.
        // Increment is commutative — all increments should be summed correctly.
        use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
        use coordinode_core::graph::node::NodeRecord;
        use coordinode_core::graph::types::Value;
        use std::sync::Arc;

        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x05";

        let num_threads = 10;
        let increments_per_thread = 100;

        let mut handles = Vec::new();
        for _ in 0..num_threads {
            let engine = Arc::clone(&engine);
            let key = key.to_vec();
            handles.push(std::thread::spawn(move || {
                for _ in 0..increments_per_thread {
                    let delta = DocDelta::Increment {
                        target: PathTarget::Extra,
                        path: vec!["counter".into()],
                        amount: 1.0,
                    };
                    let operand = delta.encode().expect("encode");
                    engine
                        .merge(Partition::Node, &key, &operand)
                        .expect("merge");
                }
            }));
        }

        for h in handles {
            h.join().expect("thread join");
        }

        let data = engine
            .get(Partition::Node, key)
            .expect("get")
            .expect("should exist");

        assert_eq!(data[0], PREFIX_NODE_RECORD);
        let rec = NodeRecord::from_msgpack(&data[1..]).expect("decode");

        let expected = (num_threads * increments_per_thread) as i64;
        assert_eq!(
            rec.get_extra("counter"),
            Some(&Value::Int(expected)),
            "expected counter={expected} after {num_threads}×{increments_per_thread} increments"
        );
    }

    #[test]
    fn doc_merge_legacy_node_record_without_prefix() {
        // Pre-R163 data: NodeRecord stored without 0x00 prefix.
        // engine.put() writes bare msgpack. After merge with DocDelta,
        // the result should have the 0x00 prefix and contain both
        // original data and delta.
        use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
        use coordinode_core::graph::node::NodeRecord;
        use coordinode_core::graph::types::Value;

        let (engine, _dir) = test_engine();
        let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x06";

        // Write bare NodeRecord (legacy format, no prefix).
        let mut rec = NodeRecord::new("Legacy");
        rec.set_extra("old_field", Value::String("preserved".into()));
        let bare = rec.to_msgpack().expect("encode");
        engine.put(Partition::Node, key, &bare).expect("put legacy");

        // Apply a DocDelta.
        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["new_field".into()],
            value: rmpv::Value::Integer(99.into()),
        };
        engine
            .merge(Partition::Node, key, &delta.encode().expect("enc"))
            .expect("merge");

        let data = engine
            .get(Partition::Node, key)
            .expect("get")
            .expect("should exist");

        // Result now has 0x00 prefix (merge function normalizes).
        assert_eq!(data[0], PREFIX_NODE_RECORD);
        let merged = NodeRecord::from_msgpack(&data[1..]).expect("decode");

        assert!(merged.has_label("Legacy"));
        assert_eq!(
            merged.get_extra("old_field"),
            Some(&Value::String("preserved".into()))
        );
        assert_eq!(merged.get_extra("new_field"), Some(&Value::Int(99)));
    }

    /// G064: Concurrent threads SET different paths on the same node via merge operands.
    /// All changes must be applied — no data loss, no conflict.
    #[test]
    fn doc_merge_concurrent_different_prop_field_paths() {
        use std::sync::Arc;

        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));

        // Create base node with a DOCUMENT property at field_id=10.
        let mut base_rec = NodeRecord::new("Device");
        base_rec.set(10, Value::Document(rmpv::Value::Map(vec![])));
        let mut base = vec![PREFIX_NODE_RECORD];
        base.extend_from_slice(&base_rec.to_msgpack().expect("enc"));
        let key = b"node:0:1";
        engine.put(Partition::Node, key, &base).expect("put base");

        let num_threads = 8;
        let mut handles = Vec::new();

        for i in 0..num_threads {
            let eng = Arc::clone(&engine);
            let path_key = format!("field_{i}");
            handles.push(std::thread::spawn(move || {
                let delta = DocDelta::SetPath {
                    target: PathTarget::PropField(10),
                    path: vec![path_key],
                    value: rmpv::Value::Integer((i as i64).into()),
                };
                eng.merge(Partition::Node, key, &delta.encode().expect("enc"))
                    .expect("merge");
            }));
        }

        for h in handles {
            h.join().expect("thread join");
        }

        // Read back — all 8 fields must be present.
        let data = engine
            .get(Partition::Node, key)
            .expect("get")
            .expect("should exist");
        let merged = NodeRecord::from_msgpack(&data).expect("decode");

        if let Some(Value::Document(doc)) = merged.props.get(&10) {
            for i in 0..num_threads {
                let val = coordinode_core::graph::document::extract_at_path(
                    doc,
                    &[&format!("field_{i}")],
                );
                assert_eq!(
                    val,
                    rmpv::Value::Integer((i as i64).into()),
                    "field_{i} missing or wrong after concurrent merge"
                );
            }
        } else {
            panic!(
                "expected Document at props[10], got: {:?}",
                merged.props.get(&10)
            );
        }
    }

    /// R165: 100 concurrent writers each push to the same array via ArrayPush merge operands.
    /// All pushes must be applied — no data loss. Order is seqno-based.
    #[test]
    fn doc_merge_concurrent_100_writers_array_push() {
        use std::sync::Arc;

        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));

        // Create base node with an empty array at field_id=20.
        let mut base_rec = NodeRecord::new("Bag");
        base_rec.set(
            20,
            Value::Document(rmpv::Value::Map(vec![(
                rmpv::Value::String("items".into()),
                rmpv::Value::Array(vec![]),
            )])),
        );
        let mut base = vec![PREFIX_NODE_RECORD];
        base.extend_from_slice(&base_rec.to_msgpack().expect("enc"));
        let key = b"node:0:1";
        engine.put(Partition::Node, key, &base).expect("put base");

        let num_writers = 100;
        let mut handles = Vec::new();

        for i in 0..num_writers {
            let eng = Arc::clone(&engine);
            handles.push(std::thread::spawn(move || {
                let delta = DocDelta::ArrayPush {
                    target: PathTarget::PropField(20),
                    path: vec!["items".into()],
                    value: rmpv::Value::Integer((i as i64).into()),
                };
                eng.merge(Partition::Node, key, &delta.encode().expect("enc"))
                    .expect("merge");
            }));
        }

        for h in handles {
            h.join().expect("thread join");
        }

        // Read back — array must contain exactly 100 elements.
        let data = engine
            .get(Partition::Node, key)
            .expect("get")
            .expect("should exist");
        let merged = NodeRecord::from_msgpack(&data).expect("decode");

        if let Some(Value::Document(doc)) = merged.props.get(&20) {
            let items = coordinode_core::graph::document::extract_at_path(doc, &["items"]);
            if let rmpv::Value::Array(arr) = items {
                assert_eq!(
                    arr.len(),
                    num_writers,
                    "expected {num_writers} items, got {}",
                    arr.len()
                );
                // Verify all values 0..99 are present (order may vary by seqno).
                let mut values: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
                values.sort();
                let expected: Vec<i64> = (0..num_writers as i64).collect();
                assert_eq!(values, expected, "all 100 values must be present");
            } else {
                panic!("expected array at items, got: {items:?}");
            }
        } else {
            panic!("expected Document at props[20]");
        }
    }
}
