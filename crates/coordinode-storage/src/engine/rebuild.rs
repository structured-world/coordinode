//! Rebuilding a damaged tree from a checkpoint plus the journal.
//!
//! A rebuild installs only a state it can prove: the history the checkpoint's
//! copy of the tree holds, each version at its original seqno, and the
//! journal entries that copy lacks, each at its own commit timestamp. When the
//! checkpoint and the journal cannot prove the tree's state, the rebuild is
//! refused and the tree is left as it was.

use std::path::{Path, PathBuf};

#[cfg(feature = "columnar")]
use lsm_tree::AbstractTree as _;
use lsm_tree::{AnyTree, ScanSinceEvent};

use crate::engine::coverage::TreeCoverage;
#[cfg(feature = "columnar")]
use crate::engine::oplog_journal::op_partition;
#[cfg(feature = "columnar")]
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::OplogEntry;
#[cfg(feature = "columnar")]
use crate::oplog::entry::OplogOp;

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
#[cfg(feature = "columnar")]
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

/// One version of a key, or one merge operand of it, a tree holds: what
/// [`rewrite_tree_values`] hands its caller.
#[derive(Debug, Clone, Copy)]
pub struct StoredVersion<'a> {
    /// The key as stored.
    pub key: &'a [u8],
    /// The value or operand as stored.
    pub value: &'a [u8],
    /// The sequence number the tree holds it at.
    pub seqno: lsm_tree::SeqNo,
    /// Whether it is a merge operand rather than a value.
    pub operand: bool,
}

/// The memtable size past which [`rewrite_tree_values`] writes what it
/// holds to a table, so a large tree is not held in memory whole.
const REWRITE_FLUSH_BYTES: u64 = 64 * 1024 * 1024;

/// Write the tree of partition `part` at the directory `from` under the
/// endpoints of `config` anew at `to`: every version, merge operand and
/// deletion it holds, each at its own sequence number, every value and
/// operand through `rewrite`, and its retention floor. Both trees are closed
/// to everything else; `to` must not hold a tree. The new tree is durable
/// when this returns; putting it in place of the old one is the caller's.
/// Returns the number of values and operands rewritten.
///
/// # Errors
///
/// A tree cannot be opened, read or written, or `rewrite` refuses a value.
pub fn rewrite_tree_values(
    config: &crate::engine::config::StorageConfig,
    part: crate::engine::partition::Partition,
    (from, to): (&Path, &Path),
    routing: Option<&crate::engine::routing::PartitionRouting>,
    rewrite: &mut dyn FnMut(StoredVersion<'_>) -> StorageResult<Vec<u8>>,
) -> StorageResult<u64> {
    use lsm_tree::AbstractTree as _;
    let counter = || {
        lsm_tree::SharedSequenceNumberGenerator::from(lsm_tree::SequenceNumberCounter::default())
    };
    let source = config
        .tree_config_at(part, from, counter(), routing)
        .open()?;
    if let Some(taken) = config
        .endpoints
        .iter()
        .map(|endpoint| endpoint.path.join(to))
        .find(|path| path.exists())
    {
        return Err(StorageError::TargetExists {
            path: taken.display().to_string(),
        });
    }
    // The new tree starts past every seqno the source holds, and its clear
    // refuses every snapshot below that until the history is in and its
    // floor lowered to the source's: no read of it, now or after a crash
    // mid-way, is answered from history the source did not serve.
    let past = source
        .get_highest_seqno()
        .map_or(0, |highest| highest + 1)
        .max(source.retention_floor() + 1);
    let target = config
        .tree_config_at(
            part,
            to,
            lsm_tree::SharedSequenceNumberGenerator::from(lsm_tree::SequenceNumberCounter::new(
                past,
            )),
            routing,
        )
        .open()?;
    target.clear()?;
    let flush = |tree: &AnyTree| -> StorageResult<()> {
        let lock = tree.get_flush_lock();
        tree.rotate_memtable();
        // Nothing below a watermark of zero is collected: every version
        // the source holds is written as it is.
        tree.flush(&lock, 0)?;
        tree.persist_flushes()?;
        Ok(())
    };
    let mut rewritten = 0u64;
    for event in tree_history(&source)? {
        let (_, memtable) = match event {
            ScanSinceEvent::Insert { key, value, seqno } => {
                let value = rewrite(StoredVersion {
                    key: &key,
                    value: &value,
                    seqno,
                    operand: false,
                })?;
                rewritten += 1;
                target.insert(key, value, seqno)
            }
            ScanSinceEvent::MergeOperand {
                key,
                operand,
                seqno,
            } => {
                let operand = rewrite(StoredVersion {
                    key: &key,
                    value: &operand,
                    seqno,
                    operand: true,
                })?;
                rewritten += 1;
                target.merge(key, operand, seqno)
            }
            ScanSinceEvent::PointTombstone { key, seqno } => target.remove(key, seqno),
            ScanSinceEvent::WeakTombstone { key, seqno } => target.remove_weak(key, seqno),
            ScanSinceEvent::RangeTombstone {
                start_key,
                end_key,
                seqno,
            } => {
                target.remove_range(start_key, end_key, seqno);
                (0, target.active_memtable().size())
            }
        };
        if memtable > REWRITE_FLUSH_BYTES {
            flush(&target)?;
        }
    }
    flush(&target)?;
    target.sync_devices()?;
    // The clear left the floor at `past`, at or above the source's: lowered
    // to exactly the source's, whichever the source's is.
    target.lower_retention_floor(source.retention_floor())?;
    Ok(rewritten)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;

/// Write one event of a tree's history into `tree` at its original seqno.
/// Returns the bytes the write added and the memtable's size after it.
#[cfg(feature = "columnar")]
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
