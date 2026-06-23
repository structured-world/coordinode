//! Write batch with fsync-per-batch crash safety guarantee.
//!
//! A `WriteBatch` groups multiple mutations into a single atomic unit.
//! On `commit()`, all writes are applied to the storage and — if the engine
//! uses `FlushPolicy::SyncPerBatch` — fsynced to disk before returning.
//!
//! ## Crash Safety Invariant
//!
//! After `commit()` returns `Ok(())`:
//! - All writes in the batch are durable (survived power loss)
//! - Partial batches are never visible (atomic commit via storage transaction)
//!
//! If the process crashes mid-batch (before `commit()`):
//! - No writes from the batch are visible after restart
//! - The storage WAL ensures consistency

use std::collections::HashMap;

use lsm_tree::AbstractTree;
use rayon::prelude::*;

use crate::engine::config::FlushPolicy;
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::StorageResult;

/// Minimum number of mutations required to engage the parallel memtable-write
/// path in [`WriteBatch::commit`].
///
/// Below this threshold the rayon work-stealing overhead (task spawn + thread
/// wake-up) exceeds the time saved by concurrent memtable inserts, so the
/// single-threaded path is used instead.  The threshold also requires at least
/// two distinct partitions (if everything targets one partition there is no
/// parallelism to exploit).
const PARALLEL_THRESHOLD: usize = 16;

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
}

impl Mutation {
    /// Return the partition this mutation targets.
    fn partition(&self) -> Partition {
        match self {
            Self::Put { partition, .. }
            | Self::Delete { partition, .. }
            | Self::Merge { partition, .. } => *partition,
        }
    }
}

/// An atomic write batch with crash safety guarantees.
///
/// Accumulates mutations, then applies them all atomically on `commit()`.
/// Uses the storage engine's `WriteTransaction` for atomicity and optional fsync for
/// durability.
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

    /// Number of staged mutations.
    pub fn len(&self) -> usize {
        self.mutations.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }

    /// Commit all staged mutations atomically.
    ///
    /// All mutations share a single seqno, providing logical atomicity for
    /// MVCC reads.  With `FlushPolicy::SyncPerBatch`, the memtable is flushed
    /// to an SST file (crash-safe atomic rename) before returning.
    ///
    /// ## Parallel memtable writes
    ///
    /// When the batch contains at least `PARALLEL_THRESHOLD` mutations that
    /// target at least two distinct partitions, mutations are grouped by
    /// partition and each group is applied concurrently on a rayon thread.
    /// This is safe because:
    ///
    /// - Each partition maps to a separate `AnyTree` (no shared mutable state
    ///   between groups).
    /// - `Tree::insert / remove / merge` take `&self` and use a concurrent
    ///   skip-list internally — safe to call from multiple threads with the
    ///   same seqno.
    /// - The `StorageEngine` `HashMap<Partition, AnyTree>` is read-only during
    ///   `commit()` (trees are only inserted at `open()` time).
    ///
    /// Below the threshold the single-threaded path avoids the rayon
    /// work-stealing overhead.
    pub fn commit(self) -> StorageResult<()> {
        if self.mutations.is_empty() {
            return Ok(());
        }

        // Destructure to satisfy borrow checker: `engine` and `mutations` are
        // used independently in closures below.
        let WriteBatch { engine, mutations } = self;

        // Pre-write capacity gate — atomicity contract: if any
        // partition in the batch targets a Full L0 endpoint, abort
        // the entire commit before any seqno is consumed or any
        // mutation lands in a memtable. Partition::Schema and
        // Partition::Raft bypass the gate via `check_partition_capacity`
        // (engine-internal metadata path), matching the single-write
        // gate behaviour. Using a set keeps the check O(distinct
        // partitions) instead of O(mutations).
        let mut checked: std::collections::HashSet<Partition> =
            std::collections::HashSet::with_capacity(Partition::all().len());
        for m in &mutations {
            let part = m.partition();
            if checked.insert(part) {
                engine.check_partition_capacity(part)?;
            }
        }

        // Single seqno for the entire batch — logical atomicity.
        let seqno = engine.next_seqno();

        // Group mutations by partition.  Preserves insertion order within each
        // group so that multiple mutations on the same key are applied in the
        // order they were staged.
        let mut groups: HashMap<Partition, Vec<&Mutation>> =
            HashMap::with_capacity(Partition::all().len());
        for m in &mutations {
            groups.entry(m.partition()).or_default().push(m);
        }

        if groups.len() >= 2 && mutations.len() >= PARALLEL_THRESHOLD {
            // Parallel path: scatter each partition group to a rayon thread.
            groups
                .par_iter()
                .try_for_each(|(&part, group)| -> StorageResult<()> {
                    let tree = engine.tree(part)?;
                    for mutation in group {
                        match mutation {
                            Mutation::Put { key, value, .. } => {
                                tree.insert(key, value, seqno);
                            }
                            Mutation::Delete { key, .. } => {
                                tree.remove(key, seqno);
                            }
                            Mutation::Merge { key, operand, .. } => {
                                tree.merge(key, operand, seqno);
                            }
                        }
                    }
                    Ok(())
                })?;
        } else {
            // Serial path: small batch or single-partition — skip rayon overhead.
            for (&part, group) in &groups {
                let tree = engine.tree(part)?;
                for mutation in group {
                    match mutation {
                        Mutation::Put { key, value, .. } => {
                            tree.insert(key, value, seqno);
                        }
                        Mutation::Delete { key, .. } => {
                            tree.remove(key, seqno);
                        }
                        Mutation::Merge { key, operand, .. } => {
                            tree.merge(key, operand, seqno);
                        }
                    }
                }
            }
        }

        // Invalidate cache for all mutated keys to prevent stale reads.
        if let Some(cache) = engine.tiered_cache() {
            for mutation in &mutations {
                let (partition, key) = match mutation {
                    Mutation::Put { partition, key, .. }
                    | Mutation::Delete { partition, key }
                    | Mutation::Merge { partition, key, .. } => (partition, key),
                };
                cache.remove(*partition, key);
            }
        }

        // If SyncPerBatch, flush memtable to SST immediately after commit.
        if engine.flush_policy() == FlushPolicy::SyncPerBatch {
            engine.persist()?;
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
