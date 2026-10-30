//! The member-local journal: this member's own recovery records (index use
//! guards and marks, the consensus vote and log bounds) on a store whose log
//! is a consensus log, which must never carry them. A record becomes durable
//! with one append to this journal, not with a flush of the tree that holds
//! it; the trees take it at once, under a coverage marker of this journal's
//! own index space, and the journal is replayed on open like the store's.
//!
//! A tree's table never holds the effect of a local entry the journal could
//! still lose: every flush syncs this journal first. A crash therefore loses
//! whole entries from the end of the journal and never leaves a tree covering
//! an index the journal no longer has unless that index is behind a purge.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use coordinode_core::txn::write_concern::Journal;
use lsm_tree::AnyTree;

use crate::engine::core::{LocalSync, LocalWrite};
use crate::engine::coverage::{self, Coverage, Domain, TreeCoverage};
use crate::engine::installation::Installations;
use crate::engine::oplog_journal::{
    EmbeddedOplog, OplogJournalConfig, apply_oplog_ops_at, op_partition, sync_journal,
};
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::{OplogEntry, OplogOp};
use crate::placement::partition_wire_tag;

/// See the module docs.
pub(crate) struct LocalJournal {
    log: Arc<Mutex<EmbeddedOplog>>,
    coverage: Coverage,
    dir: PathBuf,
    /// Entries appended since the last compaction.
    since_compaction: AtomicU64,
    /// Local entries between compactions: a compaction folds the base of
    /// every tree holding local records, flushes those trees and drops the
    /// covered segments. Local records are few (an election, an index first
    /// used), so the default bounds the journal without its flushes becoming
    /// a cost of their own.
    compact_every: u64,
}

impl LocalJournal {
    /// Where the journal of a store whose primary endpoint is `endpoint`
    /// lives: beside the consensus log, never inside its directory.
    pub(crate) fn dir(endpoint: &Path) -> PathBuf {
        endpoint.join("local-journal").join("0")
    }

    /// Open the journal in `dir` and replay into `trees` every entry they do
    /// not cover.
    ///
    /// # Errors
    ///
    /// The journal cannot be read, a tree that holds local records lacks an
    /// index the journal no longer holds ([`StorageError::CoverageUnprovable`]),
    /// or a replay write fails.
    pub(crate) fn open(
        dir: &Path,
        cfg: &OplogJournalConfig,
        trees: &HashMap<Partition, AnyTree>,
        installations: &Installations,
        footprints: crate::engine::footprint::Footprints,
    ) -> StorageResult<Self> {
        // Its own ceiling, never read: local records do not bound a read view.
        let mut log =
            EmbeddedOplog::open(dir, 0, cfg, Arc::new(AtomicU64::new(u64::MAX)), footprints)?;
        let entries = log.read_all()?;
        let mut covered: HashMap<Partition, TreeCoverage> = HashMap::with_capacity(trees.len());
        for (&part, tree) in trees {
            covered.insert(part, TreeCoverage::read(tree, Domain::Local)?);
        }
        // A purge first gives every tree that took an entry a durable base
        // (see `compact`), so a tree with no local record took none of the
        // purged prefix (the hole from index 0) and has nothing to prove
        // there. A hole above it is a loss, not a purge: every tree must
        // cover it.
        let excused = |part: Partition, hole_start: u64| {
            hole_start == 0 && covered.get(&part).is_none_or(|c| !holds_local_records(c))
        };
        if !holes_covered(&entries, log.next_index(), &covered, excused) {
            return Err(StorageError::CoverageUnprovable {
                path: dir.display().to_string(),
                entries: entries.len(),
            });
        }
        // An index a tree covers is never handed out again, even when the
        // entry that held it was purged.
        log.advance_next_index(
            covered
                .values()
                .map(TreeCoverage::next_uncovered)
                .max()
                .unwrap_or(0),
        );
        let coverage = Coverage::new(log.next_index());
        // Every tree that holds local records is folded and flushed by the
        // next compaction, so its base keeps up with the purge.
        for (&part, c) in &covered {
            if let Some((base, _)) = c.base() {
                coverage.note_based(part, base);
            }
            if let Some(last) = c.marked_indices().last() {
                coverage.note_touched(part, last);
            }
        }
        for entry in &entries {
            for (part, ops) in by_partition(&entry.ops) {
                if covered
                    .get(&part)
                    .is_some_and(|c| c.contains(entry.index, 0))
                {
                    continue;
                }
                let tree = tree_of(trees, part)?;
                coverage.note_touched(part, entry.index);
                apply_oplog_ops_at(
                    tree,
                    (part == Partition::Idx).then_some(installations),
                    &ops,
                    entry.ts,
                    entry.index,
                    Domain::Local,
                )?;
            }
        }
        if !entries.is_empty() {
            tracing::info!(dir = %dir.display(), entries = entries.len(), "local journal replayed");
        }
        Ok(Self {
            log: Arc::new(Mutex::new(log)),
            coverage,
            dir: dir.to_path_buf(),
            since_compaction: AtomicU64::new(0),
            compact_every: cfg.local_journal_compact_entries,
        })
    }

    /// The journal itself: the flushes that must sync it first, and its
    /// flusher.
    pub(crate) fn log(&self) -> &Arc<Mutex<EmbeddedOplog>> {
        &self.log
    }

    /// Journal `writes` and apply them to `trees` at `seqno`, durable before
    /// returning under [`LocalSync::BeforeUse`]; `fold_seqno` hands out a
    /// seqno above every one applied so far, for a fold. Returns whether a
    /// compaction is due.
    ///
    /// # Errors
    ///
    /// The append or its sync fails, or a write names a missing partition.
    pub(crate) fn commit(
        &self,
        trees: &HashMap<Partition, AnyTree>,
        installations: &Installations,
        writes: Vec<LocalWrite>,
        sync: LocalSync,
        seqno: u64,
        fold_seqno: impl Fn() -> u64,
    ) -> StorageResult<bool> {
        let ops: Vec<OplogOp> = writes.into_iter().map(into_op).collect();
        let bytes = ops.iter().map(op_bytes).sum();
        let level = match sync {
            LocalSync::BeforeUse => Journal::Journal,
            LocalSync::Deferred => Journal::Cache,
        };
        // The append takes the ops and the apply after it needs them: an
        // apply before the append could reach a table ahead of its entry.
        // Local records are a few keys, so the copy is small.
        let applied = ops.clone();
        let index = self
            .log
            .lock()
            .map_err(|_| StorageError::Io("local journal mutex poisoned".into()))?
            .append_ops(ops, seqno, level, bytes)?;
        for (part, ops) in by_partition(&applied) {
            let tree = tree_of(trees, part)?;
            self.coverage.note_touched(part, index);
            apply_oplog_ops_at(
                tree,
                (part == Partition::Idx).then_some(installations),
                &ops,
                seqno,
                index,
                Domain::Local,
            )?;
        }
        if self.coverage.mark_applied(index)
            && let Some(mut folded) = self.coverage.try_lock_folded()
        {
            self.fold(trees, &mut folded, fold_seqno());
        }
        // Counts entries, far below u64::MAX.
        Ok(self.since_compaction.fetch_add(1, Ordering::Relaxed) + 1 >= self.compact_every)
    }

    /// Fold the applied prefix into the base of every tree that holds local
    /// records; returns the base now written to all of them. `at` exceeds the
    /// seqno of every marker below it.
    fn fold(
        &self,
        trees: &HashMap<Partition, AnyTree>,
        folded: &mut coverage::FoldGuard<'_>,
        at: u64,
    ) -> u64 {
        let next = self.coverage.applied_prefix();
        for (&part, tree) in trees {
            let based = self.coverage.based(part);
            if !self.holds_records(part) || based >= next {
                continue;
            }
            coverage::write_fold(tree, Domain::Local, based, next, &[], at);
            self.coverage.note_based(part, next);
        }
        folded.set_folded(next);
        next
    }

    /// Whether `part`'s tree took a local entry, now or before this open.
    fn holds_records(&self, part: Partition) -> bool {
        self.coverage.based(part) > 0 || self.coverage.touched_since(part, 0)
    }

    /// Fold the base of every tree that holds local records, flush those
    /// trees through `flush`, and drop the segments they now cover durably.
    ///
    /// # Errors
    ///
    /// A flush, the seal of the active segment or a removal fails.
    pub(crate) fn compact(
        &self,
        trees: &HashMap<Partition, AnyTree>,
        at: u64,
        flush: impl Fn(&AnyTree) -> StorageResult<()>,
    ) -> StorageResult<()> {
        let below = {
            let mut folded = self.coverage.lock_folded();
            let next = self.fold(trees, &mut folded, at);
            for (&part, tree) in trees {
                if self.holds_records(part) {
                    flush(tree)?;
                }
            }
            next
        };
        let mut log = self
            .log
            .lock()
            .map_err(|_| StorageError::Io("local journal mutex poisoned".into()))?;
        log.rotate()?;
        let removed = log.purge_before(below)?;
        drop(log);
        self.since_compaction.store(0, Ordering::Relaxed);
        tracing::debug!(dir = %self.dir.display(), removed, below, "local journal compacted");
        Ok(())
    }

    /// Remove the journal's directory once [`Self::compact`] left every
    /// entry covered durably by the trees. The trees keep their coverage, so
    /// a journal opened here later hands out indices above it.
    ///
    /// # Errors
    ///
    /// The directory cannot be removed.
    pub(crate) fn retire(self) -> StorageResult<()> {
        let Self { log, dir, .. } = self;
        drop(log);
        std::fs::remove_dir_all(&dir)
            .map_err(|e| StorageError::Io(format!("remove local journal {dir:?}: {e}")))
    }

    /// Make every local record appended so far durable.
    ///
    /// # Errors
    ///
    /// The sync fails, now or earlier.
    pub(crate) fn sync(&self) -> StorageResult<()> {
        sync_journal(&self.log)
    }
}

/// Whether a tree carries any local coverage: a base or a marker.
fn holds_local_records(coverage: &TreeCoverage) -> bool {
    coverage.has_record() || coverage.marked_indices().next().is_some()
}

fn into_op(write: LocalWrite) -> OplogOp {
    match write {
        LocalWrite::Put {
            partition,
            key,
            value,
        } => OplogOp::Insert {
            partition: partition_wire_tag(partition),
            key,
            value,
        },
        LocalWrite::Delete { partition, key } => OplogOp::Delete {
            partition: partition_wire_tag(partition),
            key,
        },
    }
}

/// The ops of one entry, grouped by the tree each lands in.
fn by_partition(ops: &[OplogOp]) -> HashMap<Partition, Vec<&OplogOp>> {
    let mut grouped: HashMap<Partition, Vec<&OplogOp>> = HashMap::new();
    for op in ops {
        if let Some(part) = op_partition(op) {
            grouped.entry(part).or_default().push(op);
        }
    }
    grouped
}

fn tree_of(trees: &HashMap<Partition, AnyTree>, part: Partition) -> StorageResult<&AnyTree> {
    trees
        .get(&part)
        .ok_or_else(|| StorageError::PartitionNotFound {
            name: part.name().to_string(),
        })
}

/// The bytes an op carries, for the journal's unsynced-tail accounting.
fn op_bytes(op: &OplogOp) -> u64 {
    match op {
        OplogOp::Insert { key, value, .. } => (key.len() + value.len()) as u64,
        OplogOp::Delete { key, .. } => key.len() as u64,
        _ => 0,
    }
}

/// Whether every index below `next` that `entries` lack (a purged prefix, a
/// segment lost from the middle) is covered by each tree in `covered`, except
/// where `skip(tree, hole_start)` excuses the tree from that hole. A tree
/// without coverage for such an index would never be replayed the entry it
/// may have missed.
pub(crate) fn holes_covered(
    entries: &[OplogEntry],
    next: u64,
    covered: &HashMap<Partition, TreeCoverage>,
    skip: impl Fn(Partition, u64) -> bool,
) -> bool {
    let mut holes: Vec<(u64, u64)> = Vec::new();
    let mut expected = 0u64;
    for entry in entries {
        if entry.index > expected {
            holes.push((expected, entry.index));
        }
        // Indices are far below u64::MAX: one per commit.
        expected = expected.max(entry.index + 1);
    }
    if next > expected {
        holes.push((expected, next));
    }
    covered.iter().all(|(&part, c)| {
        let base = c.base().map_or(0, |(base, _)| base);
        holes.iter().all(|&(start, end)| {
            skip(part, start) || (start.max(base)..end).all(|i| c.contains(i, 0))
        })
    })
}
