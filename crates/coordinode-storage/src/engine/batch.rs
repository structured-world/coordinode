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
    /// When the batch contains at least [`PARALLEL_THRESHOLD`] mutations that
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
mod tests {
    use super::*;
    use crate::engine::config::StorageConfig;
    use tempfile::TempDir;

    fn test_engine_with_policy(policy: FlushPolicy) -> (StorageEngine, TempDir) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let mut config = StorageConfig::new(dir.path());
        config.flush_policy = policy;
        let engine = StorageEngine::open(&config).expect("failed to open engine");
        (engine, dir)
    }

    #[test]
    fn empty_batch_commits_ok() {
        let (engine, _dir) = test_engine_with_policy(FlushPolicy::SyncPerBatch);
        let batch = WriteBatch::new(&engine);
        assert!(batch.is_empty());
        batch.commit().expect("empty batch should commit");
    }

    #[test]
    fn batch_put_is_atomic() {
        let (engine, _dir) = test_engine_with_policy(FlushPolicy::SyncPerBatch);

        let mut batch = WriteBatch::new(&engine);
        batch.put(Partition::Node, b"k1".to_vec(), b"v1".to_vec());
        batch.put(Partition::Schema, b"k2".to_vec(), b"v2".to_vec());
        assert_eq!(batch.len(), 2);

        batch.commit().expect("batch commit failed");

        // Both writes visible after commit
        let v1 = engine.get(Partition::Node, b"k1").expect("get failed");
        let v2 = engine.get(Partition::Schema, b"k2").expect("get failed");
        assert_eq!(v1.as_deref(), Some(b"v1".as_slice()));
        assert_eq!(v2.as_deref(), Some(b"v2".as_slice()));
    }

    #[test]
    fn batch_delete_works() {
        let (engine, _dir) = test_engine_with_policy(FlushPolicy::SyncPerBatch);

        // Pre-populate
        engine
            .put(Partition::Node, b"to_delete", b"val")
            .expect("put failed");

        let mut batch = WriteBatch::new(&engine);
        batch.delete(Partition::Node, b"to_delete".to_vec());
        batch.commit().expect("batch commit failed");

        assert!(engine
            .get(Partition::Node, b"to_delete")
            .expect("get failed")
            .is_none());
    }

    #[test]
    fn batch_mixed_put_and_delete() {
        let (engine, _dir) = test_engine_with_policy(FlushPolicy::SyncPerBatch);

        engine
            .put(Partition::Node, b"old", b"old_val")
            .expect("put failed");

        let mut batch = WriteBatch::new(&engine);
        batch.put(Partition::Node, b"new", b"new_val".to_vec());
        batch.delete(Partition::Node, b"old".to_vec());
        batch.commit().expect("batch commit failed");

        assert!(engine
            .get(Partition::Node, b"old")
            .expect("get failed")
            .is_none());
        assert_eq!(
            engine
                .get(Partition::Node, b"new")
                .expect("get failed")
                .as_deref(),
            Some(b"new_val".as_slice())
        );
    }

    #[test]
    fn sync_per_batch_survives_reopen() {
        let dir = TempDir::new().expect("create temp dir");
        let mut config = StorageConfig::new(dir.path());
        config.flush_policy = FlushPolicy::SyncPerBatch;

        // Write batch with SyncPerBatch
        {
            let engine = StorageEngine::open(&config).expect("open failed");
            let mut batch = WriteBatch::new(&engine);
            batch.put(Partition::Node, b"durable_key", b"durable_val".to_vec());
            batch.commit().expect("batch commit failed");
            // No explicit persist() — SyncPerBatch does it in commit()
        }

        // Reopen and verify data survived
        {
            let engine = StorageEngine::open(&config).expect("reopen failed");
            let result = engine
                .get(Partition::Node, b"durable_key")
                .expect("get failed");
            assert_eq!(result.as_deref(), Some(b"durable_val".as_slice()));
        }
    }

    #[test]
    fn uncommitted_batch_not_visible() {
        let dir = TempDir::new().expect("create temp dir");
        let mut config = StorageConfig::new(dir.path());
        config.flush_policy = FlushPolicy::SyncPerBatch;

        // Create batch but drop without committing
        {
            let engine = StorageEngine::open(&config).expect("open failed");
            let mut batch = WriteBatch::new(&engine);
            batch.put(Partition::Node, b"ghost", b"should_not_exist".to_vec());
            // batch dropped without commit
        }

        // Reopen — "ghost" key should not exist
        {
            let engine = StorageEngine::open(&config).expect("reopen failed");
            let result = engine.get(Partition::Node, b"ghost").expect("get failed");
            assert!(
                result.is_none(),
                "uncommitted batch data should not be visible"
            );
        }
    }

    #[test]
    fn periodic_flush_policy_works() {
        let (engine, _dir) = test_engine_with_policy(FlushPolicy::Periodic(100));

        let mut batch = WriteBatch::new(&engine);
        batch.put(Partition::Node, b"periodic", b"val".to_vec());
        batch.commit().expect("batch commit failed");

        let result = engine
            .get(Partition::Node, b"periodic")
            .expect("get failed");
        assert_eq!(result.as_deref(), Some(b"val".as_slice()));
    }

    #[test]
    fn manual_flush_policy_works() {
        let (engine, _dir) = test_engine_with_policy(FlushPolicy::Manual);

        let mut batch = WriteBatch::new(&engine);
        batch.put(Partition::Node, b"manual", b"val".to_vec());
        batch.commit().expect("batch commit failed");

        let result = engine.get(Partition::Node, b"manual").expect("get failed");
        assert_eq!(result.as_deref(), Some(b"val".as_slice()));
    }

    #[test]
    fn batch_merge_produces_sorted_posting_list() {
        use crate::engine::merge::encode_add;
        use coordinode_core::graph::edge::PostingList;

        let (engine, _dir) = test_engine_with_policy(FlushPolicy::SyncPerBatch);

        // Atomic batch: PUT node + MERGE two different adj keys.
        let mut batch = WriteBatch::new(&engine);
        batch.put(Partition::Node, b"node:0:42", b"props".to_vec());
        batch.merge(
            Partition::Adj,
            b"adj:FOLLOWS:out:42".to_vec(),
            encode_add(100),
        );
        batch.merge(
            Partition::Adj,
            b"adj:FOLLOWS:in:100".to_vec(),
            encode_add(42),
        );
        batch.commit().expect("batch with merge failed");

        // Verify node was written.
        let node = engine
            .get(Partition::Node, b"node:0:42")
            .expect("get node")
            .expect("node should exist");
        assert_eq!(&*node, b"props");

        // Verify forward adj posting list.
        let fwd = engine
            .get(Partition::Adj, b"adj:FOLLOWS:out:42")
            .expect("get fwd")
            .expect("fwd should exist");
        let plist = PostingList::from_bytes(&fwd).expect("decode fwd");
        assert_eq!(plist.as_slice(), &[100]);

        // Verify reverse adj posting list.
        let rev = engine
            .get(Partition::Adj, b"adj:FOLLOWS:in:100")
            .expect("get rev")
            .expect("rev should exist");
        let plist = PostingList::from_bytes(&rev).expect("decode rev");
        assert_eq!(plist.as_slice(), &[42]);
    }

    #[test]
    fn batch_merge_same_key_uses_batch_operand() {
        // Use encode_add_batch to add multiple UIDs to the same adj key in one batch.
        use crate::engine::merge::encode_add_batch;
        use coordinode_core::graph::edge::PostingList;

        let (engine, _dir) = test_engine_with_policy(FlushPolicy::SyncPerBatch);

        let mut batch = WriteBatch::new(&engine);
        batch.merge(
            Partition::Adj,
            b"adj:KNOWS:out:1".to_vec(),
            encode_add_batch(&[10, 20, 30]),
        );
        batch.commit().expect("batch merge failed");

        let data = engine
            .get(Partition::Adj, b"adj:KNOWS:out:1")
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.as_slice(), &[10, 20, 30]);
    }

    /// PARALLEL_THRESHOLD mutations across multiple partitions trigger the
    /// parallel commit path.  All writes must be visible after commit — same
    /// correctness guarantee as the serial path.
    #[test]
    fn parallel_commit_all_writes_visible() {
        let (engine, _dir) = test_engine_with_policy(FlushPolicy::Manual);

        let mut batch = WriteBatch::new(&engine);
        // Spread 8 keys across Node and Schema (2 distinct partitions) to
        // guarantee the parallel threshold of 16 total mutations is reached.
        for i in 0..8u8 {
            batch.put(Partition::Node, format!("node:{i}").into_bytes(), vec![i]);
            batch.put(
                Partition::Schema,
                format!("schema:{i}").into_bytes(),
                vec![i + 100],
            );
        }
        assert!(
            batch.len() >= PARALLEL_THRESHOLD,
            "test precondition: must be >= PARALLEL_THRESHOLD for parallel path"
        );
        batch.commit().expect("parallel commit failed");

        // Verify every key is visible with the correct value.
        for i in 0..8u8 {
            let v = engine
                .get(Partition::Node, format!("node:{i}").as_bytes())
                .expect("get node")
                .expect("node should exist");
            assert_eq!(&*v, &[i], "wrong value for node:{i}");

            let v = engine
                .get(Partition::Schema, format!("schema:{i}").as_bytes())
                .expect("get schema")
                .expect("schema should exist");
            assert_eq!(&*v, &[i + 100], "wrong value for schema:{i}");
        }
    }

    /// Parallel path with Delete and Merge operations (not just Put).
    /// Verifies all three mutation kinds are applied correctly when the
    /// parallel threshold is crossed.
    #[test]
    fn parallel_commit_delete_and_merge() {
        use crate::engine::merge::encode_add;
        use coordinode_core::graph::edge::PostingList;

        let (engine, _dir) = test_engine_with_policy(FlushPolicy::Manual);

        // Pre-populate keys that will be deleted/merged via the parallel path.
        for i in 0..8u8 {
            engine
                .put(Partition::Node, format!("del:{i}").as_bytes(), b"old")
                .expect("pre-put");
        }

        let mut batch = WriteBatch::new(&engine);
        // 8 Deletes on Node + 8 Merges on Adj = 16 mutations, 2 partitions → parallel path.
        for i in 0..8u8 {
            batch.delete(Partition::Node, format!("del:{i}").into_bytes());
            batch.merge(
                Partition::Adj,
                format!("adj:{i}").into_bytes(),
                encode_add(u64::from(i)),
            );
        }
        assert!(
            batch.len() >= PARALLEL_THRESHOLD,
            "test precondition: must trigger parallel path"
        );
        batch.commit().expect("parallel delete+merge failed");

        // All deleted keys must be gone.
        for i in 0..8u8 {
            assert!(
                engine
                    .get(Partition::Node, format!("del:{i}").as_bytes())
                    .expect("get")
                    .is_none(),
                "node del:{i} should be deleted"
            );
        }

        // All adj merge results must be visible.
        for i in 0..8u8 {
            let data = engine
                .get(Partition::Adj, format!("adj:{i}").as_bytes())
                .expect("get adj")
                .expect("adj should exist");
            let plist = PostingList::from_bytes(&data).expect("decode posting list");
            assert_eq!(plist.as_slice(), &[u64::from(i)]);
        }
    }

    /// 16 mutations but all on one partition → serial path (no parallel benefit).
    /// Ensures the single-partition gate works and results are still correct.
    #[test]
    fn single_partition_large_batch_uses_serial_path() {
        let (engine, _dir) = test_engine_with_policy(FlushPolicy::Manual);

        let mut batch = WriteBatch::new(&engine);
        for i in 0..16u8 {
            batch.put(Partition::Node, format!("k:{i}").into_bytes(), vec![i]);
        }
        // Same batch size as PARALLEL_THRESHOLD but only one partition.
        assert_eq!(batch.len(), PARALLEL_THRESHOLD);
        batch.commit().expect("large single-partition batch failed");

        for i in 0..16u8 {
            let v = engine
                .get(Partition::Node, format!("k:{i}").as_bytes())
                .expect("get")
                .expect("should exist");
            assert_eq!(&*v, &[i]);
        }
    }

    /// 15 mutations across 2 partitions → serial path (below threshold).
    /// Ensures the mutation-count gate prevents premature rayon dispatch.
    #[test]
    fn below_threshold_multi_partition_uses_serial_path() {
        let (engine, _dir) = test_engine_with_policy(FlushPolicy::Manual);

        let mut batch = WriteBatch::new(&engine);
        // 7 Node + 8 Schema = 15 mutations, 2 partitions, below PARALLEL_THRESHOLD.
        for i in 0..7u8 {
            batch.put(Partition::Node, format!("n:{i}").into_bytes(), vec![i]);
        }
        for i in 0..8u8 {
            batch.put(
                Partition::Schema,
                format!("s:{i}").into_bytes(),
                vec![i + 50],
            );
        }
        assert_eq!(batch.len(), 15);
        assert!(batch.len() < PARALLEL_THRESHOLD);
        batch.commit().expect("below-threshold batch failed");

        for i in 0..7u8 {
            let v = engine
                .get(Partition::Node, format!("n:{i}").as_bytes())
                .expect("get")
                .expect("should exist");
            assert_eq!(&*v, &[i]);
        }
        for i in 0..8u8 {
            let v = engine
                .get(Partition::Schema, format!("s:{i}").as_bytes())
                .expect("get")
                .expect("should exist");
            assert_eq!(&*v, &[i + 50]);
        }
    }

    #[test]
    fn batch_merge_mixed_put_and_merge() {
        // Mix of PUT on Node partition and MERGE on Adj partition — atomic.
        use crate::engine::merge::encode_add;
        use coordinode_core::graph::edge::PostingList;

        let (engine, _dir) = test_engine_with_policy(FlushPolicy::SyncPerBatch);

        // Pre-populate adj with some edges via direct merge.
        engine
            .merge(Partition::Adj, b"adj:R:out:1", &encode_add(50))
            .expect("pre-merge");

        // Atomic batch: delete old node, create new node, add edge.
        let mut batch = WriteBatch::new(&engine);
        batch.delete(Partition::Node, b"node:0:1".to_vec());
        batch.put(Partition::Node, b"node:0:2", b"new_node".to_vec());
        batch.merge(Partition::Adj, b"adj:R:out:1".to_vec(), encode_add(60));
        batch.commit().expect("mixed batch failed");

        // Verify: old node gone, new node present, adj has both UIDs.
        assert!(engine
            .get(Partition::Node, b"node:0:1")
            .expect("get")
            .is_none());
        assert_eq!(
            engine
                .get(Partition::Node, b"node:0:2")
                .expect("get")
                .as_deref(),
            Some(b"new_node".as_slice())
        );
        let data = engine
            .get(Partition::Adj, b"adj:R:out:1")
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.as_slice(), &[50, 60]);
    }
}
