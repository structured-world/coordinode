//! Write batch: one seqno for a set of mutations, applied to every partition
//! tree under the memtable-rotation guard.
//!
//! A `WriteBatch` groups multiple mutations into a single logical unit. Every
//! mutation in the batch lands at the SAME seqno, so an MVCC snapshot either
//! sees the whole batch or none of it (ADR-016: seqno == commit_ts). Each
//! partition group is handed to the tree as one lsm-tree batch, which holds
//! the version-history guard for the whole insert: a concurrent memtable
//! rotation cannot seal a memtable holding only a prefix of the batch, which
//! is what keeps the oplog replay watermark ("a partition holds an entry iff
//! its highest persisted seqno is at least the entry ts") sound.
//!
//! ## Crash Safety Invariant
//!
//! After `commit()` returns `Ok(())`:
//! - All writes in the batch are durable when the engine uses
//!   `FlushPolicy::SyncPerBatch` (survived power loss)
//! - Partial batches are never visible (single seqno)
//!
//! If the process crashes mid-batch (before `commit()`):
//! - No writes from the batch are visible after restart
//! - The retained oplog / Raft log ensures consistency

use std::collections::HashMap;

use lsm_tree::AbstractTree;
use rayon::prelude::*;

use crate::engine::config::FlushPolicy;
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::StorageResult;

/// Minimum number of mutations required to engage the parallel memtable-write
/// path in [`WriteBatch::commit_at`].
///
/// Below this threshold the rayon work-stealing overhead (task spawn + thread
/// wake-up) exceeds the time saved by concurrent memtable inserts, so the
/// single-threaded path is used instead.  The threshold also requires at least
/// two distinct partitions (if everything targets one partition there is no
/// parallelism to exploit).
///
/// Set from `benches/write_batch_parallel.rs::threshold_sweep` (4 partitions,
/// 64-byte values, one lsm batch per partition): the serial path costs about
/// 0.7 µs per mutation and is flat, the rayon path costs 46 µs at 16
/// mutations and stays 5-7× slower per mutation up to 256; the two meet
/// only around 1024 mutations (0.50 µs per mutation parallel). The old value
/// of 16 made every mid-size multi-partition commit pay the rayon wake-up.
const PARALLEL_THRESHOLD: usize = 1024;

/// A mutation to be applied in a write batch.
#[derive(Debug)]
pub(crate) enum Mutation {
    Put {
        partition: Partition,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    Delete {
        partition: Partition,
        key: Vec<u8>,
    },
    /// Merge operand for partitions with a registered merge operator (e.g., Adj).
    ///
    /// Multiple merge operands for the same key within a batch are all applied.
    /// Use `encode_add_batch` to combine multiple UIDs into one operand when
    /// adding many edges to the same posting list in a single batch write.
    Merge {
        partition: Partition,
        key: Vec<u8>,
        operand: Vec<u8>,
    },
    /// One MVCC range tombstone over the half-open key range `[start, end)`.
    RemoveRange {
        partition: Partition,
        start: Vec<u8>,
        end: Vec<u8>,
    },
}

impl Mutation {
    /// Return the partition this mutation targets.
    fn partition(&self) -> Partition {
        match self {
            Self::Put { partition, .. }
            | Self::Delete { partition, .. }
            | Self::Merge { partition, .. }
            | Self::RemoveRange { partition, .. } => *partition,
        }
    }
}

/// An atomic write batch with crash safety guarantees.
///
/// Accumulates mutations, then applies them all at one seqno on `commit()`
/// (allocating the seqno) or, inside the crate, `commit_at` (caller-supplied
/// seqno, the commit timestamp of a transaction).
pub struct WriteBatch<'a> {
    engine: &'a StorageEngine,
    mutations: Vec<Mutation>,
}

impl<'a> WriteBatch<'a> {
    /// Create a new empty write batch.
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self {
            engine,
            mutations: Vec::new(),
        }
    }

    /// Create an empty write batch sized for `capacity` mutations.
    pub(crate) fn with_capacity(engine: &'a StorageEngine, capacity: usize) -> Self {
        Self {
            engine,
            mutations: Vec::with_capacity(capacity),
        }
    }

    /// Stage a put operation.
    pub fn put(
        &mut self,
        partition: Partition,
        key: impl Into<Vec<u8>>,
        value: impl Into<Vec<u8>>,
    ) {
        self.mutations.push(Mutation::Put {
            partition,
            key: key.into(),
            value: value.into(),
        });
    }

    /// Stage a delete operation.
    pub fn delete(&mut self, partition: Partition, key: impl Into<Vec<u8>>) {
        self.mutations.push(Mutation::Delete {
            partition,
            key: key.into(),
        });
    }

    /// Stage a merge operand for a partition with a registered merge operator.
    ///
    /// The operand is lazily combined with the existing value during reads
    /// and compaction. For the `Adj` partition, use `encode_add` / `encode_remove`
    /// / `encode_add_batch` from `crate::engine::merge` to encode operands.
    ///
    /// Use `encode_add_batch` to combine multiple UIDs into a single operand
    /// when adding multiple edges to the same adjacency list.
    pub fn merge(
        &mut self,
        partition: Partition,
        key: impl Into<Vec<u8>>,
        operand: impl Into<Vec<u8>>,
    ) {
        self.mutations.push(Mutation::Merge {
            partition,
            key: key.into(),
            operand: operand.into(),
        });
    }

    /// Stage one MVCC range tombstone over `[start, end)`.
    pub fn remove_range(
        &mut self,
        partition: Partition,
        start: impl Into<Vec<u8>>,
        end: impl Into<Vec<u8>>,
    ) {
        self.mutations.push(Mutation::RemoveRange {
            partition,
            start: start.into(),
            end: end.into(),
        });
    }

    /// Number of staged mutations.
    pub fn len(&self) -> usize {
        self.mutations.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }

    /// Commit all staged mutations at a freshly allocated seqno.
    ///
    /// With `FlushPolicy::SyncPerBatch`, the memtable is flushed to an SST
    /// file (crash-safe atomic rename) before returning.
    pub fn commit(self) -> StorageResult<()> {
        if self.mutations.is_empty() {
            return Ok(());
        }
        let engine = self.engine;
        // Capacity is gated inside `commit_at` BEFORE the seqno is consumed
        // only if we allocate after the gate; allocate here and let the gate
        // reject before any mutation lands (a consumed seqno with no writes
        // is harmless: seqnos only need to be monotonic, not dense).
        let seqno = engine.next_seqno();
        self.commit_at(seqno)?;
        if engine.flush_policy() == FlushPolicy::SyncPerBatch {
            engine.persist()?;
        }
        Ok(())
    }

    /// Apply every staged mutation at `seqno`, without allocating one and
    /// without persisting. This is the commit path of a transaction: `seqno`
    /// is its commit timestamp, so `snapshot_at(seqno)` sees the whole batch
    /// and `snapshot_at(seqno - 1)` sees none of it.
    ///
    /// ## Parallel memtable writes
    ///
    /// When the batch contains at least `PARALLEL_THRESHOLD` mutations that
    /// target at least two distinct partitions, mutations are grouped by
    /// partition and each group is applied concurrently on a rayon thread.
    /// This is safe because each partition maps to a separate tree and the
    /// tree's batch apply takes `&self`.
    ///
    /// ## Errors
    ///
    /// A partition on a full endpoint rejects the whole batch before any
    /// mutation lands. A batch carrying two different operation kinds on the
    /// same key (a put and a delete, a put and a merge) is rejected by the
    /// tree: at one seqno such a pair has no defined order, and the
    /// transaction layer canonicalises every key to one final operation
    /// before it gets here, so the error marks a bug upstream, never a
    /// silent overwrite.
    pub(crate) fn commit_at(self, seqno: lsm_tree::SeqNo) -> StorageResult<()> {
        if self.mutations.is_empty() {
            return Ok(());
        }

        let WriteBatch { engine, mutations } = self;

        // Pre-write capacity gate: atomicity contract, no mutation lands if any
        // partition in the batch targets a full L0 endpoint. `Schema` and
        // `Raft` bypass the gate (engine-internal metadata). O(distinct
        // partitions), not O(mutations).
        let mut checked: std::collections::HashSet<Partition> =
            std::collections::HashSet::with_capacity(Partition::all().len());
        for m in &mutations {
            let part = m.partition();
            if checked.insert(part) {
                engine.check_partition_capacity(part)?;
            }
        }

        // Group by partition, preserving staging order within each group so
        // repeated merge operands on one key resolve in the order they were
        // staged.
        let mut groups: HashMap<Partition, Vec<&Mutation>> =
            HashMap::with_capacity(Partition::all().len());
        for m in &mutations {
            groups.entry(m.partition()).or_default().push(m);
        }

        if groups.len() >= 2 && mutations.len() >= PARALLEL_THRESHOLD {
            groups
                .par_iter()
                .try_for_each(|(&part, group)| apply_group(engine, part, group, seqno))?;
        } else {
            for (&part, group) in &groups {
                apply_group(engine, part, group, seqno)?;
            }
        }

        // Invalidate cache for all mutated keys to prevent stale reads. A range
        // tombstone leaves the shadowed keys physically present, so the whole
        // partition's cache goes.
        if let Some(cache) = engine.tiered_cache() {
            for mutation in &mutations {
                match mutation {
                    Mutation::Put { partition, key, .. }
                    | Mutation::Delete { partition, key }
                    | Mutation::Merge { partition, key, .. } => cache.remove(*partition, key),
                    Mutation::RemoveRange { partition, .. } => cache.clear_partition(*partition),
                }
            }
        }

        Ok(())
    }
}

/// Apply one partition's group of a batch at `seqno`: the point operations go
/// to the tree as a single lsm-tree batch (one version-history guard, one
/// size accounting, no memtable rotation mid-batch); range tombstones follow
/// at the same seqno.
fn apply_group(
    engine: &StorageEngine,
    part: Partition,
    group: &[&Mutation],
    seqno: lsm_tree::SeqNo,
) -> StorageResult<()> {
    let tree = engine.tree(part)?;
    let mut batch = lsm_tree::WriteBatch::with_capacity(group.len());
    for mutation in group {
        match mutation {
            Mutation::Put { key, value, .. } => batch.insert(key.as_slice(), value.as_slice()),
            Mutation::Delete { key, .. } => batch.remove(key.as_slice()),
            Mutation::Merge { key, operand, .. } => {
                batch.merge(key.as_slice(), operand.as_slice());
            }
            Mutation::RemoveRange { .. } => {}
        }
    }
    tree.apply_batch(batch, seqno)?;
    for mutation in group {
        if let Mutation::RemoveRange { start, end, .. } = mutation {
            tree.remove_range(start.clone(), end.clone(), seqno);
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
