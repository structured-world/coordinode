//! Rebuilding a damaged tree from a checkpoint plus the journal.
//!
//! A rebuild installs only a state it can prove: the history the checkpoint's
//! copy of the tree holds, each version at its original seqno, and the
//! journal entries that copy lacks, each at its own commit timestamp. When the
//! checkpoint and the journal cannot prove the tree's state, the rebuild is
//! refused and the tree is left as it was.

use std::path::{Path, PathBuf};

use lsm_tree::{AbstractTree as _, AnyTree, ScanSinceEvent};

use crate::engine::coverage::TreeCoverage;
use crate::engine::oplog_journal::op_partition;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::{OplogEntry, OplogOp};

/// The name of a checkpoint directory is its index behind this prefix,
/// zero-padded, so name order is age order.
pub const CHECKPOINT_PREFIX: &str = "ckpt-";

/// Where a store under `data_dir` keeps its checkpoints: the bases a rebuild
/// starts from, at the store's open as later.
#[must_use]
pub fn checkpoint_root(data_dir: &Path) -> PathBuf {
    data_dir.join("checkpoints")
}

/// The checkpoint directories under `root`, oldest first, with their index.
///
/// # Errors
///
/// `root` exists and cannot be listed.
pub fn checkpoint_dirs(root: &Path) -> StorageResult<Vec<(u64, PathBuf)>> {
    // no-std: StoragePort trait, caller-injected
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(StorageError::Io(format!(
                "read checkpoint root {root:?}: {e}"
            )));
        }
    };
    let mut dirs: Vec<(u64, PathBuf)> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let index = path
                .file_name()?
                .to_str()?
                .strip_prefix(CHECKPOINT_PREFIX)?
                .parse()
                .ok()?;
            Some((index, path))
        })
        .collect();
    dirs.sort_by_key(|(index, _)| *index);
    Ok(dirs)
}

/// The newest checkpoint under `root`, if any.
///
/// # Errors
///
/// `root` exists and cannot be listed.
pub fn latest_checkpoint(root: &Path) -> StorageResult<Option<PathBuf>> {
    Ok(checkpoint_dirs(root)?.pop().map(|(_, path)| path))
}

/// The outcome of rebuilding a partition or a `STORAGE COLUMNAR` table from a
/// checkpoint plus the journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rebuild {
    /// The tree was rebuilt.
    Rebuilt {
        /// Versions, merge operands and deletions copied from the
        /// checkpoint at their original seqnos.
        versions: usize,
        /// Journal entries replayed into the tree at their own timestamps.
        replayed: usize,
    },
    /// Nothing was changed: the checkpoint and the journal cannot prove the
    /// tree's state, and an unproved state is never installed.
    Refused(RebuildRefusal),
}

/// Why a rebuild was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildRefusal {
    /// The checkpoint holds no copy of the table. A table created after the
    /// checkpoint and a table whose section is missing look the same, and the
    /// second must not become an empty table.
    MissingFromCheckpoint,
    /// The engine keeps no journal, so what was applied after the checkpoint
    /// cannot be replayed.
    NoJournal,
    /// The checkpoint's copy of the tree opened only by a repair that dropped
    /// data: what survived of it is not the copy.
    CheckpointDamaged,
    /// Journal entry `index` is applied, but neither the checkpoint's copy of
    /// the tree nor the journal suffix holds it.
    UncoveredEntry {
        /// The first such entry.
        index: u64,
    },
    /// The journal suffix changes the table's catalog entry at entry `index`:
    /// a DROP and a re-create under the same name may stand between the
    /// checkpoint's table and the live one.
    CatalogChanged {
        /// The first entry changing it.
        index: u64,
    },
    /// The records this member keeps of its own in the tree (its routes,
    /// its consensus state, its closure ledger) were lost, or cannot be
    /// shown intact, and nothing else holds them.
    LocalStateLost,
    /// The checkpoint belongs to a store of another deployment than the
    /// records this member keeps name.
    OtherDeployment,
}

impl core::fmt::Display for RebuildRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingFromCheckpoint => f.write_str("the checkpoint holds no copy of the table"),
            Self::NoJournal => f.write_str("no journal to replay the entries after the checkpoint"),
            Self::CheckpointDamaged => {
                f.write_str("the checkpoint's copy opened only by a repair that dropped data")
            }
            Self::UncoveredEntry { index } => write!(
                f,
                "journal entry {index} is applied but neither the checkpoint nor the journal holds it"
            ),
            Self::CatalogChanged { index } => write!(
                f,
                "journal entry {index} changes the table's catalog entry after the checkpoint"
            ),
            Self::OtherDeployment => {
                f.write_str("the checkpoint belongs to a store of another deployment")
            }
            Self::LocalStateLost => f.write_str(
                "the member's own records in the tree cannot be shown intact and nothing else \
                 holds them",
            ),
        }
    }
}

/// The first applied journal entry that neither the checkpoint's copy of a
/// tree (coverage `held`) nor the journal `suffix` holds. `applied` is the
/// engine's applied set: its contiguous prefix and the indices above it. A
/// purged or unread stretch of the journal is lost data, not an empty one.
pub(crate) fn uncovered_entry(
    held: &TreeCoverage,
    suffix: &[OplogEntry],
    applied: (u64, &[u64]),
) -> Option<RebuildRefusal> {
    let mut in_suffix: Vec<u64> = suffix.iter().map(|e| e.index).collect();
    in_suffix.sort_unstable();
    let (next, above) = applied;
    let from = held.base().map_or(0, |(base, _)| base);
    let covered = |index: u64| held.contains(index, 0) || in_suffix.binary_search(&index).is_ok();
    (from..next)
        .chain(above.iter().copied())
        .find(|&index| !covered(index))
        .map(|index| RebuildRefusal::UncoveredEntry { index })
}

/// The first entry of `suffix` the checkpoint's schema partition (coverage
/// `catalog_held`) does not hold that writes one of `catalog_keys`.
///
/// # Errors
///
/// An entry's ops do not decode.
pub(crate) fn catalog_change(
    catalog_held: &TreeCoverage,
    suffix: &[OplogEntry],
    catalog_keys: &[Vec<u8>],
) -> StorageResult<Option<RebuildRefusal>> {
    for entry in suffix {
        if catalog_held.contains(entry.index, 0) {
            continue;
        }
        for op in crate::oplog::convert::resolve_entry_ops(&entry.ops)?.iter() {
            if op_partition(op) != Some(Partition::Schema) {
                continue;
            }
            let touched = match op {
                OplogOp::Insert { key, .. }
                | OplogOp::Delete { key, .. }
                | OplogOp::Merge { key, .. } => catalog_keys.iter().any(|k| k == key),
                OplogOp::RemoveRange { start, end, .. } => catalog_keys
                    .iter()
                    .any(|k| start.as_slice() <= k.as_slice() && k.as_slice() < end.as_slice()),
                _ => false,
            };
            if touched {
                return Ok(Some(RebuildRefusal::CatalogChanged { index: entry.index }));
            }
        }
    }
    Ok(None)
}

/// Every version, merge operand and deletion `tree` still holds, oldest seqno
/// first; at one seqno, in the order the tree resolves them.
///
/// # Errors
///
/// The tree's index or a data block cannot be read.
pub(crate) fn tree_history(
    tree: &AnyTree,
) -> StorageResult<Box<dyn Iterator<Item = ScanSinceEvent>>> {
    Ok(match tree {
        AnyTree::Standard(t) => Box::new(t.scan_since_seqno(0)?),
        AnyTree::Blob(t) => Box::new(t.scan_since_seqno(0)?),
    })
}

/// Write one event of a tree's history into `tree` at its original seqno.
/// Returns the bytes the write added and the memtable's size after it.
pub(crate) fn replay_event(tree: &AnyTree, event: ScanSinceEvent) -> (u64, u64) {
    match event {
        ScanSinceEvent::Insert { key, value, seqno } => tree.insert(key, value, seqno),
        ScanSinceEvent::MergeOperand {
            key,
            operand,
            seqno,
        } => tree.merge(key, operand, seqno),
        ScanSinceEvent::PointTombstone { key, seqno } => tree.remove(key, seqno),
        ScanSinceEvent::WeakTombstone { key, seqno } => tree.remove_weak(key, seqno),
        ScanSinceEvent::RangeTombstone {
            start_key,
            end_key,
            seqno,
        } => {
            let added = tree.remove_range(start_key, end_key, seqno);
            (added, tree.active_memtable().size())
        }
    }
}
