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
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lsm_tree::fs::Fs;
use lsm_tree::table::columnar::{column_batch_to_entries, entries_to_column_batch, ColumnBatch};
use lsm_tree::{AnyTree, Cache, Config, InternalValue, SharedSequenceNumberGenerator, ValueType};

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

/// A registry of per-table columnar trees, keyed by table id.
///
/// Each `STORAGE COLUMNAR` table owns one columnar-mode tree under
/// `<base_dir>/<table_id>`. The registry opens those trees lazily, creating one
/// on first use (`CREATE TABLE`), handing back a cheap `AnyTree` clone for
/// reads/writes, and dropping the tree + its directory on `DROP TABLE`. On
/// construction it re-opens every table tree already on disk so a restart
/// recovers them.
///
/// The open-handle map is per-process: it is a cache of `AnyTree` handles, not
/// the authoritative table set. The catalogue of which tables exist is schema
/// state (replicated); each node rebuilds its handle map from disk on open. All
/// trees share the engine's sequence-number generator and block cache so their
/// MVCC seqno line and cache budget match the rest of the engine.
pub struct ColumnarTableRegistry {
    base_dir: PathBuf,
    /// The engine's filesystem backend. All directory operations and the table
    /// trees go through this, so a `MemFs`-backed engine never touches the host
    /// filesystem (a real `std::fs` write to the engine's virtual path would
    /// hit the host root and fail).
    fs: Arc<dyn Fs>,
    seqno: SharedSequenceNumberGenerator,
    cache: Arc<Cache>,
    trees: Mutex<HashMap<String, AnyTree>>,
}

impl ColumnarTableRegistry {
    /// Open the registry rooted at `base_dir` on `fs`, re-opening every table
    /// tree already present on disk (restart recovery). Trees share `fs`,
    /// `seqno`, and `cache` with the engine.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the base directory cannot be created
    /// or an existing table tree fails to re-open.
    pub fn open(
        base_dir: PathBuf,
        fs: Arc<dyn Fs>,
        seqno: SharedSequenceNumberGenerator,
        cache: Arc<Cache>,
    ) -> StorageResult<Self> {
        fs.create_dir_all(&base_dir)
            .map_err(lsm_tree::Error::from)?;
        let mut trees = HashMap::new();
        for entry in fs.read_dir(&base_dir).map_err(lsm_tree::Error::from)? {
            if !entry.is_dir {
                continue;
            }
            let tree = open_columnar_tree(&entry.path, &fs, &seqno, &cache)?;
            trees.insert(entry.file_name, tree);
        }
        Ok(Self {
            base_dir,
            fs,
            seqno,
            cache,
            trees: Mutex::new(trees),
        })
    }

    /// Return the columnar tree for `table_id`, creating + opening it on first
    /// use. Subsequent calls hand back a clone of the same handle.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the tree cannot be opened.
    pub fn create_or_open(&self, table_id: &str) -> StorageResult<AnyTree> {
        let mut trees = self.trees.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tree) = trees.get(table_id) {
            return Ok(tree.clone());
        }
        let tree = open_columnar_tree(
            &self.base_dir.join(table_id),
            &self.fs,
            &self.seqno,
            &self.cache,
        )?;
        trees.insert(table_id.to_owned(), tree.clone());
        Ok(tree)
    }

    /// Return the open columnar tree for `table_id`, or `None` if no such table
    /// is registered.
    pub fn get(&self, table_id: &str) -> Option<AnyTree> {
        self.trees
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(table_id)
            .cloned()
    }

    /// Drop a table: release the tree handle and delete its directory.
    /// Idempotent — dropping an unknown table removes nothing and succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Engine`] if the directory removal fails.
    pub fn drop_table(&self, table_id: &str) -> StorageResult<()> {
        self.trees
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(table_id);
        let dir = self.base_dir.join(table_id);
        match self.fs.remove_dir_all(&dir) {
            Ok(()) => {}
            // A table that was never written to disk has no directory yet.
            Err(e) if e.kind() == lsm_tree::io::ErrorKind::NotFound => {}
            Err(e) => return Err(StorageError::Engine(e.into())),
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
            .filter_map(|t| t.get_highest_seqno())
            .max()
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

/// Open (creating if absent) a columnar-mode tree at `dir` on `fs`, sharing the
/// engine seqno generator and block cache.
fn open_columnar_tree(
    dir: &Path,
    fs: &Arc<dyn Fs>,
    seqno: &SharedSequenceNumberGenerator,
    cache: &Arc<Cache>,
) -> StorageResult<AnyTree> {
    // visible_seqno = seqno: every write is immediately visible, matching the
    // partition trees opened by the engine. with_shared_fs keeps the tree on
    // the engine's filesystem backend (e.g. MemFs for in-memory engines).
    let tree = Config::new_with_generators(dir, seqno.clone(), seqno.clone())
        .with_shared_fs(Arc::clone(fs))
        .use_cache(Arc::clone(cache))
        .open()?;
    enable_columnar(&tree)?;
    Ok(tree)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
