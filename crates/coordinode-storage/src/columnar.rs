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

use lsm_tree::table::columnar::{column_batch_to_entries, entries_to_column_batch, ColumnBatch};
use lsm_tree::{AnyTree, InternalValue, ValueType};

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
