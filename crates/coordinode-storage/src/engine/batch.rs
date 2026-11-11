//! Write batch: one seqno for a set of mutations, applied to every partition
//! tree under the memtable-rotation guard.
//!
//! A `WriteBatch` groups multiple mutations into a single logical unit. Every
//! mutation in the batch lands at the SAME seqno, so an MVCC snapshot either
//! sees the whole batch or none of it (seqno == commit_ts). Each
//! partition group's point writes are handed to the tree as one lsm-tree
//! batch, which holds the version-history guard for the whole insert: a
//! concurrent memtable rotation cannot seal a memtable holding only a prefix
//! of it. A journalled commit puts its coverage marker in that batch, which is
//! what lets recovery tell exactly which entries a partition holds.
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
use crate::engine::coverage;
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

/// Bytes one entry of a tree batch holds beyond its key and value: the
/// batch's entry and the entry it is materialized into before the memtable
/// takes it.
const TREE_ENTRY_SLOT: u64 = 2 * core::mem::size_of::<lsm_tree::Slice>() as u64
    + core::mem::size_of::<u64>() as u64
    + core::mem::size_of::<lsm_tree::InternalValue>() as u64
    + SEEN_SLOT;

/// An entry's share of the table the tree batch checks its keys in for
/// mixed operations as it materializes: a key reference and an operation
/// kind, with the slot's control byte. The table holds up to 8/7 of the
/// next power of two above its entries: up to 16/7 slots per entry, of
/// which the doubling in [`STAGED_SLOTS`] pays two, and this the 8/7.
const SEEN_SLOT: u64 =
    (core::mem::size_of::<(&[u8], lsm_tree::ValueType)>() as u64 + 1).div_ceil(7) * 8;

/// What one such table holds once, whatever its entries: the trailing
/// group of control bytes past its slots.
const SEEN_GROUP: u64 = 16;

/// Bytes every staged mutation holds beyond the bytes it carries: its own
/// slot, its slot in the commit's partition group, its tally slot and the
/// two tree batch entries it becomes (the write and its unit record), each
/// doubled for the growth of the container holding it.
const STAGED_SLOTS: u64 = 2
    * (core::mem::size_of::<Mutation>() as u64
        + core::mem::size_of::<&Mutation>() as u64
        + crate::engine::material::COUNTED_SLOT
        + 2 * TREE_ENTRY_SLOT);

/// The most bytes a put of `key_len` and `value_len` bytes staged in a
/// [`WriteBatch`] holds at once until [`WriteBatch::commit`] hands it to
/// the memtables: the staged key and value, their copy in the tree batch,
/// and the unit record that copies the key once more, with their slots.
/// What the memtables keep afterwards is the store's, not the caller's. A
/// store preparing an index replacement writes an index entry twice more;
/// an offline restore's fresh store prepares none.
#[must_use]
pub fn staged_put_bytes(key_len: u64, value_len: u64) -> u64 {
    3 * key_len + 2 * value_len + crate::engine::material::UNIT_RECORD_FIXED + STAGED_SLOTS
}

/// [`staged_put_bytes`] for a delete of a `key_len`-byte key.
#[must_use]
pub fn staged_delete_bytes(key_len: u64) -> u64 {
    3 * key_len + crate::engine::material::UNIT_RECORD_FIXED + STAGED_SLOTS
}

/// The bytes a commit holds once per batch, besides its mutations: the
/// buffer translating an index key and the one encoding unit records, each
/// as long as the longest key, the batch's tally record, its partition
/// groups and the key tables their tree batches check.
#[must_use]
pub fn staged_batch_bytes(longest_key: u64) -> u64 {
    let groups = Partition::all().len() as u64
        * 2
        * (core::mem::size_of::<(Partition, Vec<&Mutation>)>() as u64 + 16 + SEEN_GROUP);
    2 * longest_key + crate::engine::material::UNIT_RECORD_FIXED + 32 + groups
}

/// The header fields a unit's compact frame carries once (its timestamps,
/// id, flags and operation count), each a varint of at most ten bytes.
const FRAME_UNIT_HEAD: u64 = 64;

/// The fields a frame carries per operation: its kind, partition and the
/// lengths of its key and value, each a varint of at most ten bytes.
const FRAME_OP_HEAD: u64 = 32;

/// What the journal's entry adds around a frame: its index, term,
/// timestamp, shard and op framing.
const JOURNAL_ENTRY_HEAD: u64 = 64;

/// The most bytes a journaled node-local write of one record of `key_len`
/// and `value_len` bytes ([`StorageEngine::commit_local`]) holds at once
/// beside what the memtables keep: the record as handed in and as resolved
/// for the trees, each staged ([`staged_put_bytes`]) with the batch's own
/// buffers; its compact frame and the journal entry encoding it, each
/// doubled for the growth of its buffer; and the bounds of the markers a
/// coverage fold removes, a pair in every tree.
#[must_use]
pub fn local_record_bytes(key_len: u64, value_len: u64) -> u64 {
    let frame = key_len + value_len + FRAME_UNIT_HEAD + FRAME_OP_HEAD;
    let fold = Partition::all().len() as u64 * 2 * 2 * crate::engine::coverage::MARKER_LEN as u64;
    2 * staged_put_bytes(key_len, value_len)
        + staged_batch_bytes(key_len)
        + 2 * frame
        + 2 * (frame + JOURNAL_ENTRY_HEAD)
        + fold
}

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

    /// The key and value bytes this mutation writes.
    fn data_bytes(&self) -> u64 {
        match self {
            Self::Put { key, value, .. } => (key.len() + value.len()) as u64,
            Self::Delete { key, .. } => key.len() as u64,
            Self::Merge { key, operand, .. } => (key.len() + operand.len()) as u64,
            Self::RemoveRange { start, end, .. } => (start.len() + end.len()) as u64,
        }
    }

    /// The key bytes this mutation's unit record holds: its key, or a
    /// range's bounds.
    fn unit_key_bytes(&self) -> u64 {
        match self {
            Self::Put { key, .. } | Self::Delete { key, .. } | Self::Merge { key, .. } => {
                key.len() as u64
            }
            Self::RemoveRange { start, end, .. } => (start.len() + end.len()) as u64,
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
        // Admitted whole on the owned footprint before any mutation lands:
        // a batch that does not fit writes nothing.
        let _admitted = engine.admit_partition_write(
            self.mutations.iter().map(Mutation::partition),
            crate::engine::footprint::footprint_of(
                self.mutations.iter().map(Mutation::data_bytes).sum(),
                self.mutations.len(),
            ) + crate::engine::material::record_allowance(
                self.mutations.len(),
                self.mutations.iter().map(Mutation::unit_key_bytes).sum(),
            ),
        )?;
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
        self.commit_covered(seqno, None)?;
        // The engine's read point includes `seqno` once it was allocated, so
        // the writes are visible now.
        self.forget_cached();
        Ok(())
    }

    /// [`Self::commit_at`] that also writes the coverage marker `cover` into
    /// every partition the batch touches, atomically with its effects. The
    /// tiered cache is left as it was: the caller calls
    /// [`forget_cached`](Self::forget_cached) once the writes are visible to
    /// reads, or a read between the two could cache what they replaced.
    pub(crate) fn commit_covered(
        &self,
        seqno: lsm_tree::SeqNo,
        cover: Option<coverage::Mark>,
    ) -> StorageResult<()> {
        if self.mutations.is_empty() {
            return Ok(());
        }

        let WriteBatch { engine, mutations } = self;

        // Group by partition, preserving staging order within each group so
        // repeated merge operands on one key resolve in the order they were
        // staged.
        let mut groups: HashMap<Partition, Vec<&Mutation>> =
            HashMap::with_capacity(Partition::all().len());
        for m in mutations {
            groups.entry(m.partition()).or_default().push(m);
        }

        if groups.len() >= 2 && mutations.len() >= PARALLEL_THRESHOLD {
            groups
                .par_iter()
                .try_for_each(|(&part, group)| apply_group(engine, part, group, seqno, cover))?;
        } else {
            for (&part, group) in &groups {
                apply_group(engine, part, group, seqno, cover)?;
            }
        }

        Ok(())
    }

    /// Invalidate the tiered cache for every key the batch wrote. Call once
    /// the writes are visible to reads. A range tombstone leaves the shadowed
    /// keys physically present, so the whole partition's cache goes.
    pub(crate) fn forget_cached(&self) {
        let Some(cache) = self.engine.tiered_cache() else {
            return;
        };
        for mutation in &self.mutations {
            match mutation {
                Mutation::Put { partition, key, .. }
                | Mutation::Delete { partition, key }
                | Mutation::Merge { partition, key, .. } => cache.remove(*partition, key),
                Mutation::RemoveRange { partition, .. } => cache.clear_partition(*partition),
            }
        }
    }
}

/// Apply one partition's group of a batch at `seqno`. Range tombstones go
/// first; the point operations and the coverage marker for `cover` follow as
/// one lsm-tree batch (one version-history guard, no memtable rotation
/// mid-batch). Each tombstone is its own memtable insert, so a rotation can
/// separate it from the point batch; with the marker last, the marker's
/// memtable is sealed no earlier than any effect's, and a flush, which
/// persists memtables as a prefix in seal order, never persists the marker
/// without them. A re-applied tombstone at the same seqno is idempotent.
fn apply_group(
    engine: &StorageEngine,
    part: Partition,
    group: &[&Mutation],
    seqno: lsm_tree::SeqNo,
    cover: Option<coverage::Mark>,
) -> StorageResult<()> {
    let tree = engine.tree(part)?;
    let mut addressing =
        crate::engine::installation::Addressing::new(engine.installations_of(part), tree, seqno);
    let mut batch = lsm_tree::WriteBatch::with_capacity(group.len() + 2);
    // The copy a replacement receives is the same logical write: counted once.
    let mut tally = crate::engine::material::Tally::default();
    // A replacement being prepared for a generation receives the same write
    // in the same batch, so the batch's coverage marker covers it too.
    for mutation in group {
        match mutation {
            Mutation::Put { key, value, .. } => {
                tally.put(key, value);
                batch.insert(addressing.point(key)?, value.as_slice());
                if let Some(staging) = addressing.staging_point(key) {
                    batch.insert(staging, value.as_slice());
                }
            }
            Mutation::Delete { key, .. } => {
                tally.delete(key);
                batch.remove(addressing.point(key)?);
                if let Some(staging) = addressing.staging_point(key) {
                    batch.remove(staging);
                }
            }
            Mutation::Merge { key, operand, .. } => {
                tally.merge(key, operand);
                batch.merge(addressing.point(key)?, operand.as_slice());
                if let Some(staging) = addressing.staging_point(key) {
                    batch.merge(staging, operand.as_slice());
                }
            }
            Mutation::RemoveRange { start, end, .. } => {
                let start = coverage::clamp_user_start(start);
                tally.range(start, end);
                let ranges = addressing.range(start, end);
                let staging = addressing.staging_range(start, end);
                for (start, end) in ranges.into_iter().chain(staging) {
                    if start < end {
                        tree.remove_range(start, end, seqno);
                        engine.coordinator().flush_trigger().wrote_unmeasured();
                    }
                }
            }
        }
    }
    tally.record(part, seqno, &mut batch);
    if let Some(mark) = cover {
        batch.insert(mark.key().as_slice(), &[][..]);
    }
    let (added, memtable) = tree.apply_batch(batch, seqno)?;
    engine.coordinator().flush_trigger().wrote(added, memtable);
    engine.write_taps().applied(part, group);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
