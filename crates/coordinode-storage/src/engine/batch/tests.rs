use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use tempfile::TempDir;

fn test_engine_with_policy(policy: FlushPolicy) -> (StorageEngine, TempDir) {
    let dir = TempDir::new().expect("failed to create temp dir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
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

    assert!(
        engine
            .get(Partition::Node, b"to_delete")
            .expect("get failed")
            .is_none()
    );
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

    assert!(
        engine
            .get(Partition::Node, b"old")
            .expect("get failed")
            .is_none()
    );
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
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
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
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
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
    // Spread the keys across Node and Schema (2 distinct partitions) so the
    // parallel threshold is reached whatever its value.
    let per_partition = PARALLEL_THRESHOLD / 2;
    for i in 0..per_partition {
        batch.put(
            Partition::Node,
            format!("node:{i}").into_bytes(),
            i.to_le_bytes().to_vec(),
        );
        batch.put(
            Partition::Schema,
            format!("schema:{i}").into_bytes(),
            (i + 100).to_le_bytes().to_vec(),
        );
    }
    assert!(
        batch.len() >= PARALLEL_THRESHOLD,
        "test precondition: must be >= PARALLEL_THRESHOLD for parallel path"
    );
    batch.commit().expect("parallel commit failed");

    // Verify every key is visible with the correct value.
    for i in 0..per_partition {
        let v = engine
            .get(Partition::Node, format!("node:{i}").as_bytes())
            .expect("get node")
            .expect("node should exist");
        assert_eq!(&*v, &i.to_le_bytes(), "wrong value for node:{i}");

        let v = engine
            .get(Partition::Schema, format!("schema:{i}").as_bytes())
            .expect("get schema")
            .expect("schema should exist");
        assert_eq!(&*v, &(i + 100).to_le_bytes(), "wrong value for schema:{i}");
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
    let per_partition = PARALLEL_THRESHOLD / 2;
    for i in 0..per_partition {
        engine
            .put(Partition::Node, format!("del:{i}").as_bytes(), b"old")
            .expect("pre-put");
    }

    let mut batch = WriteBatch::new(&engine);
    // Deletes on Node + Merges on Adj, 2 partitions, at the threshold → parallel path.
    for i in 0..per_partition {
        batch.delete(Partition::Node, format!("del:{i}").into_bytes());
        batch.merge(
            Partition::Adj,
            format!("adj:{i}").into_bytes(),
            encode_add(i as u64),
        );
    }
    assert!(
        batch.len() >= PARALLEL_THRESHOLD,
        "test precondition: must trigger parallel path"
    );
    batch.commit().expect("parallel delete+merge failed");

    // All deleted keys must be gone.
    for i in 0..per_partition {
        assert!(
            engine
                .get(Partition::Node, format!("del:{i}").as_bytes())
                .expect("get")
                .is_none(),
            "node del:{i} should be deleted"
        );
    }

    // All adj merge results must be visible.
    for i in 0..per_partition {
        let data = engine
            .get(Partition::Adj, format!("adj:{i}").as_bytes())
            .expect("get adj")
            .expect("adj should exist");
        let plist = PostingList::from_bytes(&data).expect("decode posting list");
        assert_eq!(plist.as_slice(), &[i as u64]);
    }
}

/// PARALLEL_THRESHOLD mutations but all on one partition → serial path (no
/// parallel benefit). Ensures the single-partition gate works and results are
/// still correct.
#[test]
fn single_partition_large_batch_uses_serial_path() {
    let (engine, _dir) = test_engine_with_policy(FlushPolicy::Manual);

    let mut batch = WriteBatch::new(&engine);
    for i in 0..PARALLEL_THRESHOLD {
        batch.put(
            Partition::Node,
            format!("k:{i}").into_bytes(),
            i.to_le_bytes().to_vec(),
        );
    }
    // Same batch size as PARALLEL_THRESHOLD but only one partition.
    assert_eq!(batch.len(), PARALLEL_THRESHOLD);
    batch.commit().expect("large single-partition batch failed");

    for i in 0..PARALLEL_THRESHOLD {
        let v = engine
            .get(Partition::Node, format!("k:{i}").as_bytes())
            .expect("get")
            .expect("should exist");
        assert_eq!(&*v, &i.to_le_bytes());
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
    assert!(
        engine
            .get(Partition::Node, b"node:0:1")
            .expect("get")
            .is_none()
    );
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

#[test]
fn commit_at_lands_every_mutation_at_the_given_seqno() {
    // Every mutation of the batch is visible at a snapshot one past the
    // batch seqno and none of them one snapshot earlier: the batch is
    // atomic for MVCC readers. (A storage snapshot at S sees seqnos < S.)
    use crate::engine::merge::encode_add;
    use coordinode_core::graph::edge::PostingList;

    let (engine, _dir) = test_engine_with_policy(FlushPolicy::Manual);
    engine
        .put(Partition::Node, b"node:0:9", b"old")
        .expect("seed");
    let seqno = engine.next_seqno();

    let mut batch = WriteBatch::new(&engine);
    batch.put(Partition::Node, b"node:0:1", b"a".to_vec());
    batch.put(Partition::Schema, b"schema:x", b"b".to_vec());
    batch.delete(Partition::Node, b"node:0:9".to_vec());
    batch.merge(Partition::Adj, b"adj:R:out:1".to_vec(), encode_add(7));
    batch.commit_at(seqno).expect("commit_at");

    let before = seqno;
    let after = seqno + 1;
    assert_eq!(
        engine
            .snapshot_get(&before, Partition::Node, b"node:0:1")
            .expect("get"),
        None
    );
    assert_eq!(
        engine
            .snapshot_get(&before, Partition::Schema, b"schema:x")
            .expect("get"),
        None
    );
    assert_eq!(
        engine
            .snapshot_get(&before, Partition::Node, b"node:0:9")
            .expect("get")
            .as_deref(),
        Some(b"old".as_slice()),
        "the delete is not visible before the batch seqno"
    );
    assert_eq!(
        engine
            .snapshot_get(&after, Partition::Node, b"node:0:1")
            .expect("get")
            .as_deref(),
        Some(b"a".as_slice())
    );
    assert_eq!(
        engine
            .snapshot_get(&after, Partition::Schema, b"schema:x")
            .expect("get")
            .as_deref(),
        Some(b"b".as_slice())
    );
    assert_eq!(
        engine
            .snapshot_get(&after, Partition::Node, b"node:0:9")
            .expect("get"),
        None,
        "the delete is visible at the batch seqno"
    );
    let adj = engine
        .snapshot_get(&after, Partition::Adj, b"adj:R:out:1")
        .expect("get")
        .expect("posting list");
    assert_eq!(
        PostingList::from_bytes(&adj).expect("decode").as_slice(),
        &[7]
    );
}

#[test]
fn commit_at_parallel_path_is_atomic_at_the_seqno() {
    // Above PARALLEL_THRESHOLD across several partitions the groups are
    // applied on rayon threads; the batch must still be one seqno for every
    // partition: nothing visible one snapshot early, everything visible one
    // snapshot late, on every partition. Guards the parallel branch now that
    // the threshold is high enough that small tests never reach it.
    let (engine, _dir) = test_engine_with_policy(FlushPolicy::Manual);
    let parts = [
        Partition::Node,
        Partition::Adj,
        Partition::Schema,
        Partition::Idx,
    ];
    let per_partition = PARALLEL_THRESHOLD / parts.len() + 1;
    let seqno = engine.next_seqno();

    let mut batch = WriteBatch::new(&engine);
    for (p, part) in parts.iter().enumerate() {
        for i in 0..per_partition {
            batch.put(*part, format!("{p}:{i}").into_bytes(), b"v".to_vec());
        }
    }
    assert!(
        batch.len() >= PARALLEL_THRESHOLD,
        "must take the rayon path"
    );
    batch.commit_at(seqno).expect("commit_at");

    for (p, part) in parts.iter().enumerate() {
        for i in [0, per_partition / 2, per_partition - 1] {
            let key = format!("{p}:{i}").into_bytes();
            assert_eq!(
                engine.snapshot_get(&seqno, *part, &key).expect("get"),
                None,
                "{}:{i} invisible before the batch seqno",
                part.name()
            );
            assert!(
                engine
                    .snapshot_get(&(seqno + 1), *part, &key)
                    .expect("get")
                    .is_some(),
                "{}:{i} visible at the batch seqno",
                part.name()
            );
        }
    }
}

#[test]
fn commit_at_rejects_two_operation_kinds_on_one_key() {
    // A put and a delete of the same key at one seqno have no defined order;
    // the tree refuses the batch instead of picking one silently, and nothing
    // from the batch lands.
    let (engine, _dir) = test_engine_with_policy(FlushPolicy::Manual);
    let seqno = engine.next_seqno();

    let mut batch = WriteBatch::new(&engine);
    batch.put(Partition::Node, b"node:0:1", b"a".to_vec());
    batch.put(Partition::Node, b"node:0:2", b"b".to_vec());
    batch.delete(Partition::Node, b"node:0:2".to_vec());
    let err = batch
        .commit_at(seqno)
        .expect_err("mixed operation kinds on one key must be rejected");
    assert!(
        matches!(
            err,
            crate::error::StorageError::Engine(lsm_tree::Error::MixedOperationBatch)
        ),
        "unexpected error: {err}"
    );
    assert_eq!(
        engine.get(Partition::Node, b"node:0:1").expect("get"),
        None,
        "a rejected batch lands nothing"
    );
}

#[test]
fn commit_at_repeated_merges_on_one_key_all_resolve() {
    // Several merge operands on the same key inside one batch are the normal
    // shape of an adjacency write (one operand per removed uid); all of them
    // resolve at the batch seqno.
    use crate::engine::merge::{encode_add, encode_add_batch, encode_remove};
    use coordinode_core::graph::edge::PostingList;

    let (engine, _dir) = test_engine_with_policy(FlushPolicy::Manual);
    engine
        .merge(
            Partition::Adj,
            b"adj:R:out:1",
            &encode_add_batch(&[1, 2, 3, 4]),
        )
        .expect("seed");
    let seqno = engine.next_seqno();

    let mut batch = WriteBatch::new(&engine);
    batch.merge(Partition::Adj, b"adj:R:out:1".to_vec(), encode_remove(2));
    batch.merge(Partition::Adj, b"adj:R:out:1".to_vec(), encode_remove(4));
    batch.merge(Partition::Adj, b"adj:R:out:1".to_vec(), encode_add(9));
    batch.commit_at(seqno).expect("commit_at");

    let adj = engine
        .snapshot_get(&(seqno + 1), Partition::Adj, b"adj:R:out:1")
        .expect("get")
        .expect("posting list");
    assert_eq!(
        PostingList::from_bytes(&adj).expect("decode").as_slice(),
        &[1, 3, 9]
    );
}
