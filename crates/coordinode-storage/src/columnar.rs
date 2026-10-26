//! Native columnar block storage for `STORAGE COLUMNAR` tables.
//!
//! The columnar layout is a whole-tree property of the storage engine: a tree
//! is row-mode or columnar-mode, never a mix of block types. Each
//! `STORAGE COLUMNAR` table therefore owns its own columnar-mode tree (the
//! `table -> tree` registry that manages those trees is built on top of these
//! primitives). This module is the thin seam over the engine's columnar batch
//! ingest and projected scan: it turns CoordiNode rows into a column batch,
//! writes the batch as a columnar block, and scans columnar blocks back with
//! per-column projection and predicate push-down.
//!
//! Gated by the `columnar` feature (the lsm-tree columnar API is itself
//! feature-gated).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use lsm_tree::config::{ColumnEncoding, ColumnEncodingPolicy};
use lsm_tree::fs::Fs;
use lsm_tree::rate_limiter::RateLimiter;
use lsm_tree::table::columnar::{ColumnBatch, column_batch_to_entries, entries_to_column_batch};
use lsm_tree::{AnyTree, Cache, CapacityPool, Config, InternalValue, ValueType, WriteBufferBudget};

use crate::engine::flush::{TableLife, TableTree};
use crate::error::{StorageError, StorageResult};

/// A single row to write into a columnar tree: its key and its opaque value
/// bytes. (Per-field sub-column transpose, which lets a projected scan skip
/// unreferenced fields, is layered on top once the table schema is known; the
/// base primitive stores the value as one column.)
pub struct ColumnarRow<'a> {
    /// Row key (the primary-key encoding). Must be strictly increasing across
    /// the batch in the tree's comparator order.
    pub key: &'a [u8],
    /// Opaque MessagePack-encoded value bytes for the row.
    pub value: &'a [u8],
}

/// Enable the columnar layout on a freshly-opened standard tree.
///
/// A columnar-mode tree transposes its rows to columnar blocks at flush and
/// accepts pre-transposed batches via [`write_columnar_rows`]. The flag is a
/// runtime-config property, set once right after opening the tree and before
/// any write. Idempotent.
///
/// # Errors
///
/// Returns [`StorageError::Engine`] if the tree is a blob tree (columnar ingest
/// does not support KV separation) or the runtime-config update fails.
pub fn enable_columnar(tree: &AnyTree) -> StorageResult<()> {
    match tree {
        AnyTree::Standard(t) => {
            t.update_runtime_config(|cfg| cfg.columnar = true)?;
            Ok(())
        }
        AnyTree::Blob(_) => Err(StorageError::Engine(lsm_tree::Error::FeatureUnsupported(
            "columnar layout is not supported on a blob tree",
        ))),
    }
}

/// Write `rows` to a columnar-mode tree as one columnar block.
///
/// The rows must be sorted by key in the tree's comparator order and strictly
/// increasing (no duplicate keys within the batch), and the first key must
/// follow any previously written data. Each row is written at seqno `0`; the
/// engine assigns the atomic global sequence number when the ingestion is
/// finished.
///
/// # Errors
///
/// Returns [`StorageError::Engine`] if the batch shape is invalid, the keys are
/// not strictly increasing, the tree is not columnar, or a block write fails.
pub fn write_columnar_rows(tree: &AnyTree, rows: &[ColumnarRow<'_>]) -> StorageResult<()> {
    let entries: Vec<InternalValue> = rows
        .iter()
        .map(|r| InternalValue::from_components(r.key, r.value, 0, ValueType::Value))
        .collect();
    let batch = entries_to_column_batch(&entries)?;
    let mut ingestion = tree.ingestion()?;
    ingestion.write_columnar_batch(&batch)?;
    ingestion.finish()?;
    Ok(())
}

/// Decode a stored column batch back into its `(key, value)` rows. Used by the
/// readback path and tests; the projected scan operator decodes only the
/// columns it needs instead.
///
/// # Errors
///
/// Returns [`StorageError::Engine`] if the batch is malformed.
pub fn columnar_batch_rows(batch: &ColumnBatch) -> StorageResult<Vec<(Vec<u8>, Vec<u8>)>> {
    let entries = column_batch_to_entries(batch)?;
    Ok(entries
        .into_iter()
        .map(|e| (e.key.user_key.to_vec(), e.value.to_vec()))
        .collect())
}

/// The directory, under every endpoint, that holds the table trees.
pub(crate) const TABLES_DIR: &str = "tables";

/// A registry of per-table columnar trees, keyed by table id.
///
/// Each `STORAGE COLUMNAR` table owns one columnar-mode tree under
/// `<endpoint>/tables/<table_id>`. The registry opens those trees lazily,
/// creating one on first use (`CREATE TABLE`), handing back a cheap `AnyTree`
/// clone for reads/writes, and dropping the tree + its directories on
/// `DROP TABLE`. On construction it re-opens every table tree already on disk
/// so a restart recovers them, repairing one whose open fails structurally.
///
/// The open-handle map is per-process: it is a cache of `AnyTree` handles, not
/// the authoritative table set. The catalogue of which tables exist is schema
/// state (replicated); each node rebuilds its handle map from disk on open. All
/// trees share the engine's sequence-number generator and block cache so their
/// MVCC seqno line and cache budget match the rest of the engine.
pub struct ColumnarTableRegistry {
    /// The engine's filesystem backend. All directory operations and the table
    /// trees go through this, so a `MemFs`-backed engine never touches the host
    /// filesystem (a real `std::fs` write to the engine's virtual path would
    /// hit the host root and fail).
    fs: Arc<dyn Fs>,
    settings: TableTreeSettings,
    trees: Mutex<HashMap<String, TableEntry>>,
}

/// An open table: its tree, its write-pressure verdict and its life.
struct TableEntry {
    tree: AnyTree,
    pressure: Arc<AtomicU8>,
    alive: TableLife,
}

/// A table tree repaired while the registry opened it: its open failed
/// structurally and the block-salvaging repair rebuilt it from its tables.
#[derive(Debug)]
pub(crate) struct TableRepair {
    /// The table whose tree was repaired.
    pub(crate) table: String,
    /// The full repair outcome, lost key coverage included.
    pub(crate) report: lsm_tree::RepairReport,
}

impl TableRepair {
    /// `true` when the salvage dropped data, which a rebuild from a
    /// checkpoint plus the journal restores.
    #[must_use]
    pub(crate) fn is_lossy(&self) -> bool {
        !matches!(
            self.report.wal_replay_scope(),
            lsm_tree::WalReplayScope::TailOnly
        )
    }
}

/// What every table tree shares with the engine's partition trees.
pub(crate) struct TableTreeSettings {
    /// The tree config of a table, built as the engine builds a partition's
    /// (compression, level routing, sync mode, filesystem, page ECC) on the
    /// engine's sequence-number generator.
    pub(crate) config: Box<dyn Fn(&str) -> Config + Send + Sync>,
    /// Every endpoint's directory of table trees, the primary one (where a
    /// table's manifest lives) first: a table's levels can lie under each.
    pub(crate) roots: Vec<PathBuf>,
    /// The block cache every tree shares.
    pub(crate) cache: Arc<Cache>,
    /// The maintenance budget the table trees' compactions draw from.
    pub(crate) maintenance: Arc<RateLimiter>,
    /// The write buffer the table trees' unflushed bytes count toward.
    pub(crate) write_buffer: Arc<WriteBufferBudget>,
    /// The account of the room the table trees' compactions hold.
    pub(crate) capacity: Arc<CapacityPool>,
    /// How each level encodes the column pages.
    pub(crate) column_encoding: ColumnEncodingPolicy,
    /// The thresholds each table's write-pressure verdict is judged against.
    pub(crate) backpressure: lsm_tree::BackpressureThresholds,
    /// The tables' write-pressure series: a table's goes with its drop.
    pub(crate) metrics: Arc<crate::engine::owned_metrics::TableMetrics>,
}

impl TableTreeSettings {
    /// Plain column pages in the levels above `level`, read most while warm,
    /// and pages encoded per page from `level` down, mostly stored. A level
    /// past the last one a tree has never encodes.
    pub(crate) fn column_encoding_from(level: u8) -> ColumnEncodingPolicy {
        if level >= 7 {
            return ColumnEncodingPolicy::all(ColumnEncoding::Plain);
        }
        let mut policy = vec![ColumnEncoding::Plain; usize::from(level)];
        policy.push(ColumnEncoding::Auto);
        ColumnEncodingPolicy::new(policy)
    }
}

impl ColumnarTableRegistry {
    /// Open the registry on `fs`, re-opening every table tree already present
    /// on disk (restart recovery) with the trees `settings` describes. A tree
    /// whose open fails structurally (a lost manifest pointer, a corrupt
    /// version log) is repaired by block salvage instead of failing the open;
    /// the repairs are returned with the registry.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the primary directory cannot be
    /// created or read, or an existing table tree fails to open and cannot be
    /// repaired.
    pub(crate) fn open(
        fs: Arc<dyn Fs>,
        settings: TableTreeSettings,
    ) -> StorageResult<(Self, Vec<TableRepair>)> {
        let primary = settings
            .roots
            .first()
            .cloned()
            .ok_or_else(|| StorageError::InvalidConfig("no table directory".into()))?;
        fs.create_dir_all(&primary).map_err(lsm_tree::Error::from)?;
        let registry = Self {
            fs,
            settings,
            trees: Mutex::new(HashMap::new()),
        };
        let mut trees = HashMap::new();
        let mut repairs = Vec::new();
        for entry in registry
            .fs
            .read_dir(&primary)
            .map_err(lsm_tree::Error::from)?
        {
            if !entry.is_dir {
                continue;
            }
            let (tree, repair) = registry.open_tree(&entry.file_name)?;
            if let Some(report) = repair {
                tracing::warn!(
                    table = %entry.file_name,
                    recovered = report.recovered,
                    salvaged = report.salvaged,
                    scope = ?report.wal_replay_scope(),
                    "columnar table tree structurally repaired at open"
                );
                repairs.push(TableRepair {
                    table: entry.file_name.clone(),
                    report,
                });
            }
            trees.insert(entry.file_name, TableEntry::new(tree));
        }
        *registry.trees.lock().unwrap_or_else(|e| e.into_inner()) = trees;
        Ok((registry, repairs))
    }

    /// Open (creating if absent) the columnar-mode tree of `table_id` with
    /// the table settings, repairing it by block salvage if its open fails
    /// structurally.
    fn open_tree(
        &self,
        table_id: &str,
    ) -> StorageResult<(AnyTree, Option<lsm_tree::RepairReport>)> {
        let settings = &self.settings;
        let (tree, repair) = (settings.config)(table_id)
            .use_cache(Arc::clone(&settings.cache))
            .compaction_rate_limiter(Arc::clone(&settings.maintenance))
            .write_buffer_budget(Arc::clone(&settings.write_buffer))
            .compaction_capacity(Arc::clone(&settings.capacity))
            .column_encoding_policy(settings.column_encoding.clone())
            .open_or_repair(lsm_tree::RepairPolicy::default().salvage(true))?;
        enable_columnar(&tree)?;
        if let AnyTree::Standard(t) = &tree {
            t.update_runtime_config(|cfg| cfg.backpressure = settings.backpressure)?;
        }
        Ok((tree, repair))
    }

    /// Return the columnar tree for `table_id`, creating + opening it on first
    /// use. Subsequent calls hand back a clone of the same handle.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the tree cannot be opened.
    #[cfg(test)]
    pub(crate) fn create_or_open(&self, table_id: &str) -> StorageResult<AnyTree> {
        self.create_or_open_with(table_id, |_| Ok(()))
    }

    /// Open or create the table's tree, running `on_create` on a tree it had to
    /// create, before any other caller can reach the table: the registry lock
    /// is held throughout, so whatever `on_create` records about the new table
    /// is in place before its first write.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the tree cannot be opened, or the
    /// error `on_create` returns (the table is then not registered).
    pub(crate) fn create_or_open_with(
        &self,
        table_id: &str,
        on_create: impl FnOnce(&AnyTree) -> StorageResult<()>,
    ) -> StorageResult<AnyTree> {
        self.open_judged(table_id, on_create).map(|(tree, _)| tree)
    }

    /// [`Self::create_or_open_with`], with the table's write-pressure tier
    /// (see [`Self::pressure`]) read under the same lock: what a write to
    /// the table needs, for one lookup.
    pub(crate) fn open_judged(
        &self,
        table_id: &str,
        on_create: impl FnOnce(&AnyTree) -> StorageResult<()>,
    ) -> StorageResult<(AnyTree, u8)> {
        let mut trees = self.trees.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = trees.get(table_id) {
            return Ok((entry.tree.clone(), entry.pressure.load(Ordering::Relaxed)));
        }
        // A new table's directory holds no tree to repair, and the monitor
        // has not judged it.
        let (tree, _) = self.open_tree(table_id)?;
        on_create(&tree)?;
        trees.insert(table_id.to_owned(), TableEntry::new(tree.clone()));
        Ok((tree, 0))
    }

    /// Return the open columnar tree for `table_id`, or `None` if no such table
    /// is registered.
    pub fn get(&self, table_id: &str) -> Option<AnyTree> {
        self.trees
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(table_id)
            .map(|entry| entry.tree.clone())
    }

    /// The write-pressure tier (0/1/2 = none/slowdown/stop) the compaction
    /// monitor last latched for `table_id`; 0 for a table it has not judged.
    pub(crate) fn pressure(&self, table_id: &str) -> u8 {
        self.trees
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(table_id)
            .map_or(0, |entry| entry.pressure.load(Ordering::Relaxed))
    }

    /// Drop a table: release the tree handle and delete its directory under
    /// every endpoint. A flush or compaction of its tree that is running is
    /// waited for, and one still queued finds the table gone and does
    /// nothing, so none writes into a directory the drop removed.
    /// Idempotent — dropping an unknown table removes nothing and succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if a directory removal fails.
    pub fn drop_table(&self, table_id: &str) -> StorageResult<()> {
        let entry = self
            .trees
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(table_id);
        if let Some(entry) = entry {
            let mut alive = entry.alive.write();
            *alive = false;
            // Under the life it ends: a monitor pass of this table that was
            // about to set its series has done so or never will.
            self.settings.metrics.remove(table_id);
        }
        for root in &self.settings.roots {
            match self.fs.remove_dir_all(&root.join(table_id)) {
                Ok(()) => {}
                // A table that was never written there has no directory.
                Err(e) if e.kind() == lsm_tree::io::ErrorKind::NotFound => {}
                Err(e) => return Err(StorageError::Engine(e.into())),
            }
        }
        Ok(())
    }

    /// Highest LSM seqno across every registered columnar tree, or `None` if no
    /// table has any persisted data. Used at open time to bump the engine's
    /// shared sequence-number generator past columnar writes (the partition-tree
    /// restore alone would miss a columnar-only workload's seqno line).
    pub fn max_highest_seqno(&self) -> Option<u64> {
        use lsm_tree::AbstractTree;
        self.trees
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(|entry| entry.tree.get_highest_seqno())
            .max()
    }

    /// Every registered table with its tree handle and verdict, unordered.
    pub(crate) fn trees(&self) -> Vec<TableTree> {
        self.trees
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(id, entry)| TableTree {
                id: id.clone(),
                tree: entry.tree.clone(),
                pressure: Arc::clone(&entry.pressure),
                alive: Arc::clone(&entry.alive),
                metrics: Arc::clone(&self.settings.metrics),
            })
            .collect()
    }

    /// Sorted list of currently-registered table ids.
    pub fn table_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .trees
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect();
        ids.sort();
        ids
    }
}

impl TableEntry {
    /// A live table the compaction monitor has not judged yet.
    fn new(tree: AnyTree) -> Self {
        Self {
            tree,
            pressure: Arc::new(AtomicU8::new(0)),
            alive: Arc::new(parking_lot::RwLock::new(true)),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
