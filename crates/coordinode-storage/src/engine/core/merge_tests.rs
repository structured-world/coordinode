use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use crate::engine::merge::{encode_add, encode_add_batch, encode_remove};
use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
use coordinode_core::graph::edge::PostingList;
use coordinode_core::graph::node::NodeRecord;
use coordinode_core::graph::types::Value;
use tempfile::TempDir;

fn test_engine() -> (StorageEngine, TempDir) {
    let dir = TempDir::new().expect("failed to create temp dir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("failed to open engine");
    (engine, dir)
}

#[test]
fn merge_add_produces_sorted_posting_list() {
    let (engine, _dir) = test_engine();
    let key = b"adj:FOLLOWS:out:42";

    engine
        .merge(Partition::Adj, key, &encode_add(30))
        .expect("merge 30");
    engine
        .merge(Partition::Adj, key, &encode_add(10))
        .expect("merge 10");
    engine
        .merge(Partition::Adj, key, &encode_add(20))
        .expect("merge 20");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get failed")
        .expect("key should exist");
    let plist = PostingList::from_bytes(&data).expect("decode failed");
    assert_eq!(plist.as_slice(), &[10, 20, 30]);
}

#[test]
fn merge_add_and_remove_interleaved() {
    let (engine, _dir) = test_engine();
    let key = b"adj:KNOWS:out:1";

    engine
        .merge(Partition::Adj, key, &encode_add(100))
        .expect("add 100");
    engine
        .merge(Partition::Adj, key, &encode_add(200))
        .expect("add 200");
    engine
        .merge(Partition::Adj, key, &encode_add(300))
        .expect("add 300");
    engine
        .merge(Partition::Adj, key, &encode_remove(200))
        .expect("remove 200");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");
    assert_eq!(plist.as_slice(), &[100, 300]);
}

#[test]
fn merge_batch_add() {
    let (engine, _dir) = test_engine();
    let key = b"adj:RATED:out:5";

    engine
        .merge(Partition::Adj, key, &encode_add_batch(&[50, 10, 30]))
        .expect("batch add");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");
    assert_eq!(plist.as_slice(), &[10, 30, 50]);
}

#[test]
fn merge_on_existing_put_base() {
    let (engine, _dir) = test_engine();
    let key = b"adj:FOLLOWS:out:7";

    let base = PostingList::from_sorted(vec![1, 5, 10]);
    engine
        .put(Partition::Adj, key, &base.to_bytes().expect("encode"))
        .expect("put base");

    engine
        .merge(Partition::Adj, key, &encode_add(7))
        .expect("merge add");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");
    assert_eq!(plist.as_slice(), &[1, 5, 7, 10]);
}

#[test]
fn merge_survives_persist_and_reopen() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let key = b"adj:FOLLOWS:out:99";

    {
        let engine = StorageEngine::open(&config).expect("open");
        engine
            .merge(Partition::Adj, key, &encode_add(1))
            .expect("add 1");
        engine
            .merge(Partition::Adj, key, &encode_add(2))
            .expect("add 2");
        engine.persist().expect("persist");
    }

    {
        let engine = StorageEngine::open(&config).expect("reopen");
        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist after reopen");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.as_slice(), &[1, 2]);
    }
}

#[test]
fn merge_many_sequential_adds() {
    let (engine, _dir) = test_engine();
    let key = b"adj:FOLLOWS:out:hub";

    for uid in 0..1000u64 {
        engine
            .merge(Partition::Adj, key, &encode_add(uid))
            .expect("merge");
    }

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");
    assert_eq!(plist.len(), 1000);
    let slice = plist.as_slice();
    for i in 1..slice.len() {
        assert!(
            slice[i - 1] < slice[i],
            "not sorted at index {i}: {} >= {}",
            slice[i - 1],
            slice[i]
        );
    }
}

// -- R010b: edge cases + concurrent tests --

#[test]
fn merge_concurrent_multithreaded_adds() {
    use std::sync::Arc;
    use std::thread;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let key = b"adj:FOLLOWS:out:celebrity";

    let num_threads = 8;
    let uids_per_thread = 500;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let engine = Arc::clone(&engine);
            thread::spawn(move || {
                for i in 0..uids_per_thread {
                    let uid = (t * uids_per_thread + i) as u64;
                    engine
                        .merge(Partition::Adj, key, &encode_add(uid))
                        .expect("merge failed");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    engine.persist().expect("persist");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");

    let total = (num_threads * uids_per_thread) as usize;
    assert_eq!(
        plist.len(),
        total,
        "expected {total} UIDs, got {}",
        plist.len()
    );

    let slice = plist.as_slice();
    for i in 1..slice.len() {
        assert!(
            slice[i - 1] < slice[i],
            "not sorted at index {i}: {} >= {}",
            slice[i - 1],
            slice[i]
        );
    }
}

#[test]
fn merge_compaction_correctness() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let key = b"adj:RATED:out:99";

    for uid in (0..100u64).rev() {
        engine
            .merge(Partition::Adj, key, &encode_add(uid))
            .expect("merge");
    }

    engine
        .force_compaction(Partition::Adj)
        .expect("force compaction");

    for uid in 100..200u64 {
        engine
            .merge(Partition::Adj, key, &encode_add(uid))
            .expect("merge phase 2");
    }

    engine
        .merge(Partition::Adj, key, &encode_remove(50))
        .expect("remove 50");
    engine
        .merge(Partition::Adj, key, &encode_remove(75))
        .expect("remove 75");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");

    assert_eq!(plist.len(), 198);
    assert!(!plist.contains(50));
    assert!(!plist.contains(75));
    assert!(plist.contains(0));
    assert!(plist.contains(199));

    let slice = plist.as_slice();
    for i in 1..slice.len() {
        assert!(slice[i - 1] < slice[i]);
    }
}

#[test]
fn merge_compaction_reopen_correctness() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let key = b"adj:KNOWS:out:77";

    {
        let engine = StorageEngine::open(&config).expect("open");
        engine
            .merge(Partition::Adj, key, &encode_add(10))
            .expect("add 10");
        engine
            .merge(Partition::Adj, key, &encode_add(20))
            .expect("add 20");
        engine
            .merge(Partition::Adj, key, &encode_add(30))
            .expect("add 30");

        engine
            .force_compaction(Partition::Adj)
            .expect("force compaction");

        engine
            .merge(Partition::Adj, key, &encode_add(5))
            .expect("add 5");
        engine
            .merge(Partition::Adj, key, &encode_remove(20))
            .expect("remove 20");
        engine.persist().expect("persist");
    }

    {
        let engine = StorageEngine::open(&config).expect("reopen");
        let data = engine
            .get(Partition::Adj, key)
            .expect("get")
            .expect("should exist");
        let plist = PostingList::from_bytes(&data).expect("decode");
        assert_eq!(plist.as_slice(), &[5, 10, 30]);
    }
}

#[test]
fn merge_empty_base_remove_is_noop() {
    let (engine, _dir) = test_engine();
    let key = b"adj:X:out:1";

    engine
        .merge(Partition::Adj, key, &encode_remove(42))
        .expect("remove on empty");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");
    assert!(
        plist.is_empty(),
        "remove on empty base should produce empty list"
    );
}

#[test]
fn merge_add_remove_same_uid_in_sequence() {
    let (engine, _dir) = test_engine();
    let key = b"adj:Y:out:2";

    engine
        .merge(Partition::Adj, key, &encode_add(100))
        .expect("add");
    engine
        .merge(Partition::Adj, key, &encode_remove(100))
        .expect("remove");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");
    assert!(plist.is_empty());
}

#[test]
fn merge_remove_then_add_same_uid() {
    let (engine, _dir) = test_engine();
    let key = b"adj:Z:out:3";

    engine
        .merge(Partition::Adj, key, &encode_remove(42))
        .expect("remove first");
    engine
        .merge(Partition::Adj, key, &encode_add(42))
        .expect("add after");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");
    assert_eq!(plist.as_slice(), &[42]);
}

#[test]
fn merge_batch_with_duplicates_is_idempotent() {
    let (engine, _dir) = test_engine();
    let key = b"adj:W:out:4";

    engine
        .merge(Partition::Adj, key, &encode_add_batch(&[5, 3, 5, 1, 3, 1]))
        .expect("batch with dupes");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");
    assert_eq!(plist.as_slice(), &[1, 3, 5]);
}

#[test]
fn merge_double_compaction_partial_re_merge() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let key = b"adj:FOLLOWS:out:hub";

    engine
        .merge(Partition::Adj, key, &encode_add(10))
        .expect("add 10");
    engine
        .merge(Partition::Adj, key, &encode_add(20))
        .expect("add 20");

    engine
        .force_compaction(Partition::Adj)
        .expect("first compaction");

    engine
        .merge(Partition::Adj, key, &encode_add(15))
        .expect("add 15");

    engine
        .force_compaction(Partition::Adj)
        .expect("second compaction");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");
    assert_eq!(plist.as_slice(), &[10, 15, 20]);
}

// ================================================================
// R010d: Merge operator stress + time-travel tests
// ================================================================

#[test]
fn merge_stress_concurrent_writers_zero_conflict() {
    use std::sync::Arc;
    use std::thread;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let key = b"adj:FOLLOWS:out:supernode";

    let num_writers = 50;
    let edges_per_writer = 1_000;
    let total_uids = num_writers * edges_per_writer;

    let handles: Vec<_> = (0..num_writers)
        .map(|writer_id| {
            let engine = Arc::clone(&engine);
            thread::spawn(move || {
                let base = writer_id * edges_per_writer;
                let uids: Vec<u64> = (base..base + edges_per_writer).map(|x| x as u64).collect();
                engine
                    .merge(Partition::Adj, key, &encode_add_batch(&uids))
                    .expect("merge should never fail — commutative ops");
            })
        })
        .collect();

    for h in handles {
        h.join().expect("writer thread panicked");
    }

    engine.persist().expect("persist");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("posting list should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");

    assert_eq!(
        plist.len(),
        total_uids,
        "expected {} UIDs, got {} — some merge operands lost",
        total_uids,
        plist.len()
    );

    let slice = plist.as_slice();
    for i in 1..slice.len() {
        assert!(
            slice[i - 1] < slice[i],
            "not sorted at index {}: {} >= {}",
            i,
            slice[i - 1],
            slice[i]
        );
    }

    assert_eq!(slice[0], 0);
    assert_eq!(slice[slice.len() - 1], (total_uids - 1) as u64);
}

#[test]
fn merge_time_travel_snapshot_correctness() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let key = b"adj:RATES:out:user42";

    let batches = 10;
    let uids_per_batch = 100;
    let mut snapshots: Vec<lsm_tree::SeqNo> = Vec::with_capacity(batches);
    let mut expected_uids_at_snapshot: Vec<Vec<u64>> = Vec::with_capacity(batches);

    for batch_idx in 0..batches {
        let base = batch_idx * uids_per_batch;
        for i in 0..uids_per_batch {
            let uid = (base + i) as u64;
            engine
                .merge(Partition::Adj, key, &encode_add(uid))
                .expect("merge");
        }

        engine.persist().expect("persist");

        // Take a seqno snapshot after this batch.
        let snap = engine.snapshot();
        snapshots.push(snap);

        let expected: Vec<u64> = (0..((batch_idx + 1) * uids_per_batch) as u64).collect();
        expected_uids_at_snapshot.push(expected);
    }

    // Write more operands AFTER the last snapshot.
    for uid in 1000..1050u64 {
        engine
            .merge(Partition::Adj, key, &encode_add(uid))
            .expect("merge post-snapshot");
    }
    engine.persist().expect("persist");

    // Verify each snapshot sees only the UIDs written before it.
    for (snap_idx, snap) in snapshots.iter().enumerate() {
        let data = engine
            .snapshot_get(snap, Partition::Adj, key)
            .expect("snapshot get")
            .expect("posting list should exist in snapshot");
        let plist = PostingList::from_bytes(&data).expect("decode");

        assert_eq!(
            plist.as_slice(),
            expected_uids_at_snapshot[snap_idx].as_slice(),
            "snapshot {} sees wrong UIDs: expected {} UIDs, got {}",
            snap_idx,
            expected_uids_at_snapshot[snap_idx].len(),
            plist.len()
        );
    }

    // Current read sees ALL UIDs (1000 + 50 post-snapshot).
    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");
    assert_eq!(plist.len(), 1050, "current read should see all 1050 UIDs");
}

#[test]
fn merge_compaction_preserves_time_travel() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    let key = b"adj:KNOWS:out:node7";

    // Phase 1: Write 200 UIDs in 2 batches, snapshot after each.
    for uid in 0..100u64 {
        engine
            .merge(Partition::Adj, key, &encode_add(uid))
            .expect("merge batch 1");
    }
    engine.persist().expect("persist");
    let snap_after_100 = engine.snapshot();

    for uid in 100..200u64 {
        engine
            .merge(Partition::Adj, key, &encode_add(uid))
            .expect("merge batch 2");
    }
    engine.persist().expect("persist");
    let snap_after_200 = engine.snapshot();

    // Phase 2: Force compaction.
    engine.force_compaction(Partition::Adj).expect("compaction");

    // Phase 3: Write 50 more UIDs after compaction.
    for uid in 200..250u64 {
        engine
            .merge(Partition::Adj, key, &encode_add(uid))
            .expect("merge batch 3");
    }
    engine.persist().expect("persist");
    let snap_after_250 = engine.snapshot();

    // Verify pre-compaction snapshots.
    let data_100 = engine
        .snapshot_get(&snap_after_100, Partition::Adj, key)
        .expect("get snap 100")
        .expect("should exist");
    let plist_100 = PostingList::from_bytes(&data_100).expect("decode");
    assert_eq!(
        plist_100.len(),
        100,
        "snap_after_100 should see 100 UIDs, got {}",
        plist_100.len()
    );

    let data_200 = engine
        .snapshot_get(&snap_after_200, Partition::Adj, key)
        .expect("get snap 200")
        .expect("should exist");
    let plist_200 = PostingList::from_bytes(&data_200).expect("decode");
    assert_eq!(
        plist_200.len(),
        200,
        "snap_after_200 should see 200 UIDs, got {}",
        plist_200.len()
    );

    // Post-compaction snapshot sees compacted base + new batch.
    let data_250 = engine
        .snapshot_get(&snap_after_250, Partition::Adj, key)
        .expect("get snap 250")
        .expect("should exist");
    let plist_250 = PostingList::from_bytes(&data_250).expect("decode");
    assert_eq!(
        plist_250.len(),
        250,
        "snap_after_250 should see 250 UIDs, got {}",
        plist_250.len()
    );

    // Current read sees everything.
    let data_all = engine
        .get(Partition::Adj, key)
        .expect("get current")
        .expect("should exist");
    let plist_all = PostingList::from_bytes(&data_all).expect("decode");
    assert_eq!(plist_all.len(), 250, "current should see 250 UIDs");

    // Verify all posting lists are sorted.
    for (name, plist) in [
        ("snap_100", &plist_100),
        ("snap_200", &plist_200),
        ("snap_250", &plist_250),
        ("current", &plist_all),
    ] {
        let slice = plist.as_slice();
        for i in 1..slice.len() {
            assert!(
                slice[i - 1] < slice[i],
                "{name}: not sorted at {i}: {} >= {}",
                slice[i - 1],
                slice[i]
            );
        }
    }
}

#[test]
fn merge_stress_interleaved_add_remove_concurrent() {
    use std::sync::Arc;
    use std::thread;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let key = b"adj:LIKES:out:user99";

    let batch: Vec<u64> = (0..1000).collect();
    engine
        .merge(Partition::Adj, key, &encode_add_batch(&batch))
        .expect("batch add");
    engine.persist().expect("persist");

    let num_threads = 20;
    let ops_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let engine = Arc::clone(&engine);
            thread::spawn(move || {
                if t % 2 == 0 {
                    let base = 1000 + (t / 2) * ops_per_thread;
                    let uids: Vec<u64> = (base..base + ops_per_thread).map(|x| x as u64).collect();
                    engine
                        .merge(Partition::Adj, key, &encode_add_batch(&uids))
                        .expect("batch add");
                } else {
                    let base = (t / 2) * ops_per_thread;
                    let end = std::cmp::min(base + ops_per_thread, 1000);
                    for uid in base..end {
                        engine
                            .merge(Partition::Adj, key, &encode_remove(uid as u64))
                            .expect("remove");
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    engine.persist().expect("persist");

    let data = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("should exist");
    let plist = PostingList::from_bytes(&data).expect("decode");

    let slice = plist.as_slice();
    for i in 1..slice.len() {
        assert!(
            slice[i - 1] < slice[i],
            "not sorted at {i}: {} >= {}",
            slice[i - 1],
            slice[i]
        );
    }

    let added_count = (num_threads / 2) * ops_per_thread;
    assert_eq!(
        plist.len(),
        added_count,
        "expected {} UIDs (original 1000 removed, {} added)",
        added_count,
        added_count
    );
}

// === G049: UidPack format verification ===

#[test]
fn merge_stores_uidpack_format_on_disk() {
    // Verify that after merge operations, the stored bytes are
    // valid UidPack (not raw Vec<u64> MessagePack).
    let (engine, _dir) = test_engine();

    let key = b"adj:TEST:out:\x00\x00\x00\x00\x00\x00\x00\x01";

    // Write 300 UIDs via merge (forces multiple UidBlocks at 256/block).
    let uids: Vec<u64> = (1..=300).collect();
    let operand = crate::engine::merge::encode_add_batch(&uids);
    engine
        .merge(Partition::Adj, key, &operand)
        .expect("merge 300 UIDs");

    // Read back raw bytes and verify UidPack structure.
    let raw = engine
        .get(Partition::Adj, key)
        .expect("get failed")
        .expect("should exist");
    let pack: coordinode_core::graph::codec::UidPack =
        rmp_serde::from_slice(&raw).expect("stored bytes must be valid UidPack");
    assert_eq!(pack.total_uids(), 300);
    assert!(
        pack.blocks.len() >= 2,
        "300 UIDs should produce ≥2 blocks, got {}",
        pack.blocks.len()
    );

    // Verify UIDs are correct via PostingList decode.
    let plist = PostingList::from_bytes(&raw).expect("decode");
    assert_eq!(plist.len(), 300);
    assert_eq!(plist.as_slice()[0], 1);
    assert_eq!(plist.as_slice()[299], 300);

    // Verify compression: UidPack should be smaller than raw Vec<u64>.
    let raw_vec_size = rmp_serde::to_vec(&uids).expect("raw").len();
    assert!(
        raw.len() < raw_vec_size,
        "UidPack ({} bytes) should be smaller than raw Vec<u64> ({} bytes)",
        raw.len(),
        raw_vec_size
    );
}

// === R061: snapshot_at integration tests ===

#[test]
fn snapshot_at_reads_historical_value() {
    let (engine, _dir) = test_engine();
    let key = b"snap_test_key";

    engine.put(Partition::Node, key, b"v1").expect("put v1");
    engine.persist().expect("persist");
    let seqno_after_v1 = engine.snapshot();

    engine.put(Partition::Node, key, b"v2").expect("put v2");
    engine.persist().expect("persist");

    // Current read sees v2.
    let current = engine.get(Partition::Node, key).expect("get");
    assert_eq!(current.as_deref(), Some(b"v2".as_ref()));

    // Historical snapshot at seqno_after_v1 sees v1.
    let historical = engine
        .snapshot_get(&seqno_after_v1, Partition::Node, key)
        .expect("snap get");
    assert_eq!(historical.as_deref(), Some(b"v1".as_ref()));
}

#[test]
fn snapshot_at_future_returns_some() {
    // snapshot_at always returns Some in the native seqno design.
    let (engine, _dir) = test_engine();
    assert!(engine.snapshot_at(u64::MAX).is_some());
}

#[test]
fn snapshot_at_isolation_from_concurrent_writes() {
    let (engine, _dir) = test_engine();

    for i in 0..10u32 {
        engine
            .put(Partition::Node, format!("k{i}").as_bytes(), b"old")
            .expect("put");
    }
    engine.persist().expect("persist");

    let snap_seqno = engine.snapshot();

    for i in 0..10u32 {
        engine
            .put(Partition::Node, format!("k{i}").as_bytes(), b"new")
            .expect("put");
    }
    engine.persist().expect("persist");

    // Snapshot reads must see "old", not "new".
    for i in 0..10u32 {
        let val = engine
            .snapshot_get(&snap_seqno, Partition::Node, format!("k{i}").as_bytes())
            .expect("snap get");
        assert_eq!(
            val.as_deref(),
            Some(b"old".as_ref()),
            "snapshot isolation violated for k{i}"
        );
    }

    // Current reads see "new".
    for i in 0..10u32 {
        let val = engine
            .get(Partition::Node, format!("k{i}").as_bytes())
            .expect("get");
        assert_eq!(val.as_deref(), Some(b"new".as_ref()));
    }
}

// === R064: TimestampOracle as SequenceNumberGenerator ===

#[test]
fn open_with_oracle_writes_use_oracle_seqno() {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use std::sync::Arc;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);

    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");

    for i in 0..10u32 {
        engine
            .put(Partition::Node, format!("k{i}").as_bytes(), b"val")
            .expect("put");
    }
    engine.persist().expect("persist");

    let current = oracle.current().as_raw();
    assert!(
        current >= 1010,
        "oracle should be ≥1010 after 10 writes, got {current}"
    );

    let snap = engine.snapshot();
    assert!(snap >= 1010, "snapshot seqno should be ≥1010, got {snap}");
}

#[test]
fn oracle_snapshot_at_matches_write_timestamp() {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use std::sync::Arc;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");

    engine.put(Partition::Node, b"key", b"v1").expect("put");
    engine.persist().expect("persist");
    let seqno_v1 = engine.snapshot();

    engine.put(Partition::Node, b"key", b"v2").expect("put");
    engine.persist().expect("persist");

    // Snapshot at seqno_v1 should see v1.
    let val = engine
        .snapshot_get(&seqno_v1, Partition::Node, b"key")
        .expect("snap get");
    assert_eq!(val.as_deref(), Some(b"v1".as_ref()));

    assert!(
        seqno_v1 > 100,
        "seqno should be oracle-driven (>100), got {seqno_v1}"
    );

    // Current read sees v2.
    let current = engine.get(Partition::Node, b"key").expect("get");
    assert_eq!(current.as_deref(), Some(b"v2".as_ref()));
}

// === R066: has_write_after — seqno-based OCC conflict detection ===

#[test]
fn has_write_after_detects_newer_write() {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use std::sync::Arc;

    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");

    engine
        .put(Partition::Node, b"node:1:1", b"v1")
        .expect("put");
    let write_seqno = engine.snapshot();

    assert!(
        engine
            .has_write_after(Partition::Node, b"node:1:1", 99)
            .expect("check"),
        "should detect write after seqno 99"
    );

    assert!(
        !engine
            .has_write_after(Partition::Node, b"node:1:1", write_seqno)
            .expect("check"),
        "should not detect write at or before current seqno"
    );
}

#[test]
fn has_write_after_nonexistent_key_returns_false() {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use std::sync::Arc;

    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle).expect("open");

    assert!(
        !engine
            .has_write_after(Partition::Node, b"nonexistent", 0)
            .expect("check"),
        "nonexistent key should return false"
    );
}

#[test]
fn has_write_after_detects_delete_tombstone() {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use std::sync::Arc;

    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");

    engine
        .put(Partition::Node, b"node:1:2", b"data")
        .expect("put");
    let seqno_after_put = engine.snapshot();

    engine.delete(Partition::Node, b"node:1:2").expect("delete");

    assert!(
        engine
            .has_write_after(Partition::Node, b"node:1:2", seqno_after_put)
            .expect("check"),
        "should detect delete tombstone as a write"
    );
}

#[test]
fn has_write_after_different_partitions_independent() {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use std::sync::Arc;

    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle).expect("open");

    engine
        .put(Partition::Node, b"node:1:1", b"data")
        .expect("put");

    assert!(
        !engine
            .has_write_after(Partition::Schema, b"node:1:1", 0)
            .expect("check"),
        "write in Node partition should not affect Schema partition"
    );
}

// === R163: Document merge operator integration tests ===

#[test]
fn doc_merge_through_storage_engine() {
    // Verify that DocDelta merge operands written via engine.merge()
    // are correctly combined when read back via engine.get().
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;

    let (engine, _dir) = test_engine();
    let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01";

    // Write base NodeRecord with 0x00 prefix via PUT.
    let mut rec = NodeRecord::new("Device");
    rec.set_extra("config", Value::Document(rmpv::Value::Map(vec![])));
    let base_msgpack = rec.to_msgpack().expect("encode");
    let mut base = Vec::with_capacity(1 + base_msgpack.len());
    base.push(PREFIX_NODE_RECORD);
    base.extend_from_slice(&base_msgpack);
    engine.put(Partition::Node, key, &base).expect("put base");

    // Write DocDelta merge operand: set config.ssid = "home".
    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["config".into(), "ssid".into()],
        value: rmpv::Value::String("home".into()),
    };
    let operand = delta.encode().expect("encode delta");
    engine
        .merge(Partition::Node, key, &operand)
        .expect("merge delta");

    // Read back: engine should merge base + delta transparently.
    let data = engine
        .get(Partition::Node, key)
        .expect("get")
        .expect("should exist");

    // Decode the merged result (strip 0x00 prefix).
    assert_eq!(data[0], PREFIX_NODE_RECORD);
    let merged = NodeRecord::from_msgpack(&data[1..]).expect("decode merged");

    // Labels preserved.
    assert!(merged.has_label("Device"));

    // DocDelta applied: config.ssid = "home".
    let config = merged.get_extra("config").expect("config key");
    if let Value::Document(doc) = config {
        let ssid = coordinode_core::graph::document::extract_at_path(doc, &["ssid"]);
        assert_eq!(ssid, rmpv::Value::String("home".into()));
    } else {
        panic!("expected Document, got {config:?}");
    }
}

#[test]
fn doc_merge_multiple_deltas_through_engine() {
    // Multiple merge operands on same key — all applied in order.
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;

    let (engine, _dir) = test_engine();
    let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02";

    // No base — merge operands create record from scratch.
    let deltas = vec![
        DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["name".into()],
            value: rmpv::Value::String("sensor-1".into()),
        },
        DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["status".into()],
            value: rmpv::Value::String("active".into()),
        },
        DocDelta::Increment {
            target: PathTarget::Extra,
            path: vec!["readings".into()],
            amount: 1.0,
        },
        DocDelta::Increment {
            target: PathTarget::Extra,
            path: vec!["readings".into()],
            amount: 1.0,
        },
        DocDelta::Increment {
            target: PathTarget::Extra,
            path: vec!["readings".into()],
            amount: 1.0,
        },
    ];

    for delta in &deltas {
        let operand = delta.encode().expect("encode");
        engine.merge(Partition::Node, key, &operand).expect("merge");
    }

    let data = engine
        .get(Partition::Node, key)
        .expect("get")
        .expect("should exist");

    assert_eq!(data[0], PREFIX_NODE_RECORD);
    let rec = NodeRecord::from_msgpack(&data[1..]).expect("decode");

    assert_eq!(
        rec.get_extra("name"),
        Some(&Value::String("sensor-1".into()))
    );
    assert_eq!(
        rec.get_extra("status"),
        Some(&Value::String("active".into()))
    );
    assert_eq!(rec.get_extra("readings"), Some(&Value::Int(3)));
}

#[test]
fn doc_merge_survives_persist_and_reopen() {
    // Write merge operands → persist → reopen → verify merged result.
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03";

    {
        let engine = StorageEngine::open(&config).expect("open");

        // Write base + delta.
        let rec = NodeRecord::new("Config");
        let base_msgpack = rec.to_msgpack().expect("encode");
        let mut base = Vec::with_capacity(1 + base_msgpack.len());
        base.push(PREFIX_NODE_RECORD);
        base.extend_from_slice(&base_msgpack);
        engine.put(Partition::Node, key, &base).expect("put");

        let delta = DocDelta::SetPath {
            target: PathTarget::Extra,
            path: vec!["version".into()],
            value: rmpv::Value::Integer(42.into()),
        };
        engine
            .merge(Partition::Node, key, &delta.encode().expect("enc"))
            .expect("merge");

        engine.persist().expect("persist");
    }

    // Reopen and verify.
    {
        let engine = StorageEngine::open(&config).expect("reopen");
        let data = engine
            .get(Partition::Node, key)
            .expect("get")
            .expect("should exist after reopen");

        assert_eq!(data[0], PREFIX_NODE_RECORD);
        let rec = NodeRecord::from_msgpack(&data[1..]).expect("decode");
        assert!(rec.has_label("Config"));
        assert_eq!(rec.get_extra("version"), Some(&Value::Int(42)));
    }
}

#[test]
fn doc_merge_concurrent_different_paths_no_conflict() {
    // Multiple threads writing merge operands to different paths on same key.
    // All operands should be merged correctly — no OCC conflicts.
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;
    use std::sync::Arc;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x04";

    let num_threads = 8;
    let ops_per_thread = 50;

    let mut handles = Vec::new();
    for thread_id in 0..num_threads {
        let engine = Arc::clone(&engine);
        let key = key.to_vec();
        handles.push(std::thread::spawn(move || {
            for i in 0..ops_per_thread {
                let path_name = format!("t{thread_id}_field{i}");
                let delta = DocDelta::SetPath {
                    target: PathTarget::Extra,
                    path: vec![path_name],
                    value: rmpv::Value::Integer((thread_id * 1000 + i).into()),
                };
                let operand = delta.encode().expect("encode");
                engine
                    .merge(Partition::Node, &key, &operand)
                    .expect("merge");
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }

    let data = engine
        .get(Partition::Node, key)
        .expect("get")
        .expect("should exist");

    assert_eq!(data[0], PREFIX_NODE_RECORD);
    let rec = NodeRecord::from_msgpack(&data[1..]).expect("decode");

    // Verify all fields from all threads are present.
    let extra = rec.extra.as_ref().expect("extra should exist");
    let expected_count = num_threads * ops_per_thread;
    assert_eq!(
        extra.len(),
        expected_count as usize,
        "expected {} fields, got {}",
        expected_count,
        extra.len()
    );

    // Spot check: thread 0, field 0.
    assert_eq!(rec.get_extra("t0_field0"), Some(&Value::Int(0)));
    // Spot check: last thread, last field.
    let last_t = num_threads - 1;
    let last_f = ops_per_thread - 1;
    assert_eq!(
        rec.get_extra(&format!("t{last_t}_field{last_f}")),
        Some(&Value::Int((last_t * 1000 + last_f) as i64))
    );
}

#[test]
fn doc_merge_concurrent_increment_same_path() {
    // Multiple threads incrementing the same counter via merge operands.
    // Increment is commutative — all increments should be summed correctly.
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;
    use std::sync::Arc;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x05";

    let num_threads = 10;
    let increments_per_thread = 100;

    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let engine = Arc::clone(&engine);
        let key = key.to_vec();
        handles.push(std::thread::spawn(move || {
            for _ in 0..increments_per_thread {
                let delta = DocDelta::Increment {
                    target: PathTarget::Extra,
                    path: vec!["counter".into()],
                    amount: 1.0,
                };
                let operand = delta.encode().expect("encode");
                engine
                    .merge(Partition::Node, &key, &operand)
                    .expect("merge");
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }

    let data = engine
        .get(Partition::Node, key)
        .expect("get")
        .expect("should exist");

    assert_eq!(data[0], PREFIX_NODE_RECORD);
    let rec = NodeRecord::from_msgpack(&data[1..]).expect("decode");

    let expected = (num_threads * increments_per_thread) as i64;
    assert_eq!(
        rec.get_extra("counter"),
        Some(&Value::Int(expected)),
        "expected counter={expected} after {num_threads}×{increments_per_thread} increments"
    );
}

#[test]
fn doc_merge_legacy_node_record_without_prefix() {
    // Pre-R163 data: NodeRecord stored without 0x00 prefix.
    // engine.put() writes bare msgpack. After merge with DocDelta,
    // the result should have the 0x00 prefix and contain both
    // original data and delta.
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget, PREFIX_NODE_RECORD};
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::graph::types::Value;

    let (engine, _dir) = test_engine();
    let key = b"node:\x00\x01\x00\x00\x00\x00\x00\x00\x00\x06";

    // Write bare NodeRecord (legacy format, no prefix).
    let mut rec = NodeRecord::new("Legacy");
    rec.set_extra("old_field", Value::String("preserved".into()));
    let bare = rec.to_msgpack().expect("encode");
    engine.put(Partition::Node, key, &bare).expect("put legacy");

    // Apply a DocDelta.
    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["new_field".into()],
        value: rmpv::Value::Integer(99.into()),
    };
    engine
        .merge(Partition::Node, key, &delta.encode().expect("enc"))
        .expect("merge");

    let data = engine
        .get(Partition::Node, key)
        .expect("get")
        .expect("should exist");

    // Result now has 0x00 prefix (merge function normalizes).
    assert_eq!(data[0], PREFIX_NODE_RECORD);
    let merged = NodeRecord::from_msgpack(&data[1..]).expect("decode");

    assert!(merged.has_label("Legacy"));
    assert_eq!(
        merged.get_extra("old_field"),
        Some(&Value::String("preserved".into()))
    );
    assert_eq!(merged.get_extra("new_field"), Some(&Value::Int(99)));
}

/// G064: Concurrent threads SET different paths on the same node via merge operands.
/// All changes must be applied — no data loss, no conflict.
#[test]
fn doc_merge_concurrent_different_prop_field_paths() {
    use std::sync::Arc;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));

    // Create base node with a DOCUMENT property at field_id=10.
    let mut base_rec = NodeRecord::new("Device");
    base_rec.set(10, Value::Document(rmpv::Value::Map(vec![])));
    let mut base = vec![PREFIX_NODE_RECORD];
    base.extend_from_slice(&base_rec.to_msgpack().expect("enc"));
    let key = b"node:0:1";
    engine.put(Partition::Node, key, &base).expect("put base");

    let num_threads = 8;
    let mut handles = Vec::new();

    for i in 0..num_threads {
        let eng = Arc::clone(&engine);
        let path_key = format!("field_{i}");
        handles.push(std::thread::spawn(move || {
            let delta = DocDelta::SetPath {
                target: PathTarget::PropField(10),
                path: vec![path_key],
                value: rmpv::Value::Integer((i as i64).into()),
            };
            eng.merge(Partition::Node, key, &delta.encode().expect("enc"))
                .expect("merge");
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }

    // Read back — all 8 fields must be present.
    let data = engine
        .get(Partition::Node, key)
        .expect("get")
        .expect("should exist");
    let merged = NodeRecord::from_msgpack(&data).expect("decode");

    if let Some(Value::Document(doc)) = merged.props.get(&10) {
        for i in 0..num_threads {
            let val =
                coordinode_core::graph::document::extract_at_path(doc, &[&format!("field_{i}")]);
            assert_eq!(
                val,
                rmpv::Value::Integer((i as i64).into()),
                "field_{i} missing or wrong after concurrent merge"
            );
        }
    } else {
        panic!(
            "expected Document at props[10], got: {:?}",
            merged.props.get(&10)
        );
    }
}

/// R165: 100 concurrent writers each push to the same array via ArrayPush merge operands.
/// All pushes must be applied — no data loss. Order is seqno-based.
#[test]
fn doc_merge_concurrent_100_writers_array_push() {
    use std::sync::Arc;

    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));

    // Create base node with an empty array at field_id=20.
    let mut base_rec = NodeRecord::new("Bag");
    base_rec.set(
        20,
        Value::Document(rmpv::Value::Map(vec![(
            rmpv::Value::String("items".into()),
            rmpv::Value::Array(vec![]),
        )])),
    );
    let mut base = vec![PREFIX_NODE_RECORD];
    base.extend_from_slice(&base_rec.to_msgpack().expect("enc"));
    let key = b"node:0:1";
    engine.put(Partition::Node, key, &base).expect("put base");

    let num_writers = 100;
    let mut handles = Vec::new();

    for i in 0..num_writers {
        let eng = Arc::clone(&engine);
        handles.push(std::thread::spawn(move || {
            let delta = DocDelta::ArrayPush {
                target: PathTarget::PropField(20),
                path: vec!["items".into()],
                value: rmpv::Value::Integer((i as i64).into()),
            };
            eng.merge(Partition::Node, key, &delta.encode().expect("enc"))
                .expect("merge");
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }

    // Read back — array must contain exactly 100 elements.
    let data = engine
        .get(Partition::Node, key)
        .expect("get")
        .expect("should exist");
    let merged = NodeRecord::from_msgpack(&data).expect("decode");

    if let Some(Value::Document(doc)) = merged.props.get(&20) {
        let items = coordinode_core::graph::document::extract_at_path(doc, &["items"]);
        if let rmpv::Value::Array(arr) = items {
            assert_eq!(
                arr.len(),
                num_writers,
                "expected {num_writers} items, got {}",
                arr.len()
            );
            // Verify all values 0..99 are present (order may vary by seqno).
            let mut values: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
            values.sort();
            let expected: Vec<i64> = (0..num_writers as i64).collect();
            assert_eq!(values, expected, "all 100 values must be present");
        } else {
            panic!("expected array at items, got: {items:?}");
        }
    } else {
        panic!("expected Document at props[20]");
    }
}

#[test]
fn apply_mutation_dispatches_to_partition() {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    let (engine, _dir) = test_engine();

    // Put through apply_mutation lands in the mapped physical partition.
    engine
        .apply_mutation(&Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:0:1".to_vec(),
            value: b"alice".to_vec(),
        })
        .expect("put");
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:1")
            .expect("get")
            .as_deref(),
        Some(b"alice".as_ref()),
    );

    // Delete through apply_mutation tombstones the same key.
    engine
        .apply_mutation(&Mutation::Delete {
            partition: PartitionId::Node,
            key: b"node:0:1".to_vec(),
        })
        .expect("delete");
    assert!(engine
        .get(Partition::Node, b"node:0:1")
        .expect("get")
        .is_none());
}
