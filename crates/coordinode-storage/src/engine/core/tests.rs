use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use tempfile::TempDir;

/// With `--features io-uring` on Linux, a disk engine opened with no
/// explicit filesystem override runs on the shared `IoUringFs` ring.
/// Confirm the host kernel supports io_uring, then prove a write/read
/// roundtrip through the io_uring path. The whole disk-backed suite also
/// exercises io_uring under this feature; this is its focused marker.
#[cfg(all(target_os = "linux", feature = "io-uring"))]
#[test]
fn io_uring_backend_opens_and_roundtrips() {
    assert!(
        lsm_tree::fs::is_io_uring_available(),
        "io-uring feature build requires an io_uring-capable kernel (Linux 5.6+)",
    );
    let dir = TempDir::new().expect("failed to create temp dir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open io_uring-backed engine");
    engine
        .put(Partition::Schema, b"io_uring_key", b"io_uring_val")
        .expect("put via io_uring backend");
    let got = engine
        .get(Partition::Schema, b"io_uring_key")
        .expect("get via io_uring backend");
    assert_eq!(got.as_deref(), Some(&b"io_uring_val"[..]));
}

/// Counter for unique MemFs paths across parallel tests.
static MEMFS_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Create a test engine. Uses MemFs when `COORDINODE_TEST_MEMFS=1` env var is set,
/// otherwise falls back to tempfile on disk.
///
/// MemFs is ~10x faster (no disk I/O) but has limitations:
/// - No oplog support (oplog uses std::fs directly)
/// - Compaction finalization may fail on some code paths (lsm-tree known limitation)
/// - TieredCache layers not supported (filesystem-based)
///
/// Returns `(StorageEngine, Option<TempDir>)` — TempDir is None for MemFs.
fn test_engine() -> (StorageEngine, Option<TempDir>) {
    if std::env::var("COORDINODE_TEST_MEMFS").as_deref() == Ok("1") {
        let engine = test_engine_memfs();
        (engine, None)
    } else {
        let dir = TempDir::new().expect("failed to create temp dir");
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("failed to open engine");
        (engine, Some(dir))
    }
}

/// `remove_range` deletes exactly the keys in `[start, end)` and leaves keys
/// outside the range intact — the MVCC range-tombstone path (G096).
#[test]
fn remove_range_deletes_only_keys_in_range() {
    let engine = test_engine_memfs();
    let key = |i: u64| {
        let mut k = b"node:0:".to_vec();
        k.extend_from_slice(&i.to_be_bytes());
        k
    };
    for i in 0..10u64 {
        engine.put(Partition::Node, &key(i), b"v").expect("put");
    }

    // Delete [k3, k7): removes 3,4,5,6; 0-2 and 7-9 survive.
    engine
        .remove_range(Partition::Node, &key(3), &key(7))
        .expect("remove_range");

    for i in 0..10u64 {
        let got = engine.get(Partition::Node, &key(i)).expect("get");
        if (3..7).contains(&i) {
            assert!(got.is_none(), "key {i} inside [3,7) must be deleted");
        } else {
            assert!(got.is_some(), "key {i} outside [3,7) must survive");
        }
    }
}

fn test_engine_memfs() -> StorageEngine {
    use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
    let id = MEMFS_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let config = StorageConfig::with_endpoints_no_persistence(vec![EndpointConfig::new(
        "default-memfs",
        format!("/memfs/test_{id}"),
        Media::Ram,
        Durability::Volatile,
        Tier::Memory,
    )])
    .with_fs(Arc::new(lsm_tree::fs::MemFs::new()));
    StorageEngine::open(&config).expect("failed to open memfs engine")
}

/// `changed_keys_since` returns exactly the keys whose version
/// history advanced past the boundary seqno — new writes, deletes, and
/// merge-partition operands — and excludes untouched pre-boundary keys.
/// Exercises both the standard-tree path (Node/Counter) and the
/// KV-separated path (Blob).
#[test]
fn changed_keys_since_captures_post_boundary_only() {
    let (engine, _d) = test_engine();

    engine
        .put(Partition::Node, b"node:1:untouched", b"v0")
        .expect("put untouched");
    engine
        .put(Partition::Node, b"node:1:doomed", b"v0")
        .expect("put doomed");
    let boundary = engine.snapshot();

    // After the boundary: a fresh key, a delete of a pre-boundary key, a
    // merge operand (Counter), and a blob-tree write.
    engine
        .put(Partition::Node, b"node:1:fresh", b"v1")
        .expect("put fresh");
    engine
        .delete(Partition::Node, b"node:1:doomed")
        .expect("delete doomed");
    engine
        .merge(Partition::Counter, b"counter:deg:1", &5i64.to_le_bytes())
        .expect("merge counter");
    engine
        .put(Partition::Blob, b"blob:abc", &[7u8; 64])
        .expect("put blob");

    let node = engine
        .changed_keys_since(Partition::Node, boundary)
        .expect("changed node");
    assert!(node.contains(&b"node:1:fresh".to_vec()), "new key captured");
    assert!(
        node.contains(&b"node:1:doomed".to_vec()),
        "deleted key captured (tombstone)"
    );
    assert!(
        !node.contains(&b"node:1:untouched".to_vec()),
        "untouched pre-boundary key excluded"
    );

    let counter = engine
        .changed_keys_since(Partition::Counter, boundary)
        .expect("changed counter");
    assert!(
        counter.contains(&b"counter:deg:1".to_vec()),
        "merge-partition key captured via MergeOperand event"
    );

    let blob = engine
        .changed_keys_since(Partition::Blob, boundary)
        .expect("changed blob");
    assert!(
        blob.contains(&b"blob:abc".to_vec()),
        "KV-separated (blob) scan path captures the key"
    );

    // A scan from "now" sees nothing new.
    let now = engine.snapshot();
    assert!(
        engine
            .changed_keys_since(Partition::Node, now)
            .expect("changed at now")
            .is_empty(),
        "no keys changed at/after the current seqno"
    );
}

#[test]
fn memfs_engine_basic_kv() {
    let engine = test_engine_memfs();
    engine
        .put(Partition::Node, b"node:00:00000001", b"hello")
        .expect("put");
    let val = engine
        .get(Partition::Node, b"node:00:00000001")
        .expect("get");
    assert_eq!(val.as_deref(), Some(b"hello".as_slice()));
}

#[test]
fn memfs_engine_all_partitions() {
    let engine = test_engine_memfs();
    for &part in Partition::all() {
        assert!(engine.tree(part).is_ok(), "partition {:?} missing", part);
    }
}

#[test]
fn memfs_engine_multi_partition_writes() {
    let engine = test_engine_memfs();

    // Write to multiple partitions
    engine
        .put(Partition::Node, b"node:00:00000001", b"alice")
        .expect("put node");
    engine
        .put(Partition::Schema, b"schema:label:User", b"schema_data")
        .expect("put schema");
    engine
        .put(
            Partition::EdgeProp,
            b"edgeprop:KNOWS:00000001:00000002",
            b"props",
        )
        .expect("put edgeprop");

    // Read back from each partition
    assert_eq!(
        engine
            .get(Partition::Node, b"node:00:00000001")
            .expect("get")
            .as_deref(),
        Some(b"alice".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::Schema, b"schema:label:User")
            .expect("get")
            .as_deref(),
        Some(b"schema_data".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::EdgeProp, b"edgeprop:KNOWS:00000001:00000002")
            .expect("get")
            .as_deref(),
        Some(b"props".as_slice())
    );

    // Non-existent key returns None
    assert!(engine
        .get(Partition::Node, b"node:00:99999999")
        .expect("get")
        .is_none());
}

#[test]
fn memfs_engine_merge_operator_adj() {
    use crate::engine::merge::encode_add_batch;

    let engine = test_engine_memfs();

    // Merge operator on Adj partition (PostingListMerge)
    let key = b"adj:KNOWS:out:00000001";
    let delta1 = encode_add_batch(&[100, 200]);
    let delta2 = encode_add_batch(&[300]);

    engine.merge(Partition::Adj, key, &delta1).expect("merge 1");
    engine.merge(Partition::Adj, key, &delta2).expect("merge 2");

    // Read merged posting list
    let val = engine.get(Partition::Adj, key).expect("get adj");
    assert!(val.is_some(), "merged adj key should exist");
    let bytes = val.expect("adj bytes");
    let plist =
        coordinode_core::graph::edge::PostingList::from_bytes(&bytes).expect("decode posting list");
    let uids: Vec<u64> = plist.iter().collect();
    assert_eq!(uids, vec![100, 200, 300]);
}

#[test]
fn force_compaction_folds_adjacency_operands_preserving_value() {
    use crate::engine::merge::encode_add_batch;
    use coordinode_core::graph::edge::PostingList;

    let engine = test_engine_memfs();
    let key = b"adj:KNOWS:out:00000007";

    // Build a hub via many single-neighbour merge operands, the shape a
    // super-node takes after an incremental bulk load.
    let degree: u64 = 64;
    for t in 0..degree {
        engine
            .merge(Partition::Adj, key, &encode_add_batch(&[1000 + t]))
            .expect("merge");
    }
    let before_uids: Vec<u64> =
        PostingList::from_bytes(&engine.get(Partition::Adj, key).expect("get").expect("hub"))
            .expect("decode")
            .iter()
            .collect();
    assert_eq!(before_uids.len(), degree as usize);

    // force_compaction advances the watermark (nothing is pinned) and folds
    // the operand chain through the major compaction, preserving the value.
    engine.force_compaction(Partition::Adj).expect("compact");

    let after_uids: Vec<u64> =
        PostingList::from_bytes(&engine.get(Partition::Adj, key).expect("get").expect("hub"))
            .expect("decode")
            .iter()
            .collect();
    assert_eq!(after_uids, before_uids, "fold must preserve neighbours");

    // A later merge still composes correctly on top of the folded base.
    engine
        .merge(Partition::Adj, key, &encode_add_batch(&[9999]))
        .expect("merge after fold");
    let extended: Vec<u64> =
        PostingList::from_bytes(&engine.get(Partition::Adj, key).expect("get").expect("hub"))
            .expect("decode")
            .iter()
            .collect();
    assert!(extended.contains(&9999), "new edge visible after fold");
    assert_eq!(extended.len(), degree as usize + 1);
}

#[test]
fn gc_watermark_tracks_oldest_pin() {
    let engine = test_engine_memfs();

    // Writes advance the shared seqno.
    for i in 0..5u64 {
        engine
            .put(Partition::Node, format!("node:00:{i:08}").as_bytes(), b"v")
            .expect("put");
    }
    // No pins: the watermark advances to the current seqno so compaction
    // may fold / collect everything.
    engine.advance_gc_watermark();
    assert!(
        engine.gc_watermark() > 0,
        "watermark advances to current seqno when nothing is pinned"
    );

    // Pin a snapshot: the watermark must not exceed it.
    let (pinned, pin) = engine.pin_snapshot();
    assert!(
        engine.gc_watermark() <= pinned,
        "watermark capped at the pinned seqno"
    );

    // Newer writes move the current seqno well past the pin.
    for i in 5..15u64 {
        engine
            .put(Partition::Node, format!("node:00:{i:08}").as_bytes(), b"v")
            .expect("put");
    }
    engine.advance_gc_watermark();
    assert!(
        engine.gc_watermark() <= pinned,
        "an active pin holds the watermark back despite newer writes"
    );

    // Releasing the pin lets the watermark advance past the old snapshot.
    drop(pin);
    let after = engine.gc_watermark();
    assert!(
        after > pinned,
        "watermark advances after the pin is released (got {after}, pin {pinned})"
    );
}

#[test]
fn gc_watermark_holds_at_minimum_of_two_pins() {
    let engine = test_engine_memfs();
    for i in 0..3u64 {
        engine
            .put(Partition::Node, format!("node:00:{i:08}").as_bytes(), b"v")
            .expect("put");
    }
    let (first, pin_first) = engine.pin_snapshot();
    for i in 3..8u64 {
        engine
            .put(Partition::Node, format!("node:00:{i:08}").as_bytes(), b"v")
            .expect("put");
    }
    let (second, pin_second) = engine.pin_snapshot();
    assert!(second >= first, "later pin captures a newer seqno");
    // Watermark must sit at the OLDER pin while both are live.
    assert!(
        engine.gc_watermark() <= first,
        "watermark held at the oldest live pin"
    );
    // Dropping the newer pin alone must not advance past the older one.
    drop(pin_second);
    assert!(
        engine.gc_watermark() <= first,
        "older pin still holds the watermark after the newer one drops"
    );
    drop(pin_first);
    assert!(
        engine.gc_watermark() > first,
        "watermark advances only once the oldest pin drops"
    );
}

#[test]
fn memfs_engine_snapshot_read() {
    let engine = test_engine_memfs();

    // Write version 1
    engine
        .put(Partition::Node, b"node:00:00000001", b"v1")
        .expect("put v1");

    // Take snapshot: next_seqno() returns the seqno for the next write.
    // Snapshot at this value sees all writes with seqno < snap (i.e., v1).
    let snap = engine.next_seqno();

    // Write version 2
    engine
        .put(Partition::Node, b"node:00:00000001", b"v2")
        .expect("put v2");

    // Current read sees v2
    let current = engine
        .get(Partition::Node, b"node:00:00000001")
        .expect("get current");
    assert_eq!(current.as_deref(), Some(b"v2".as_slice()));

    // Snapshot read sees v1
    let historical = engine
        .snapshot_get(&snap, Partition::Node, b"node:00:00000001")
        .expect("snapshot get");
    assert_eq!(historical.as_deref(), Some(b"v1".as_slice()));
}

#[test]
fn open_creates_all_partitions() {
    let (engine, _dir) = test_engine();
    for &part in Partition::all() {
        assert!(engine.tree(part).is_ok(), "partition {:?} missing", part);
    }
}

#[test]
fn put_get_delete_round_trip() {
    let (engine, _dir) = test_engine();
    let key = b"test_key";
    let value = b"test_value";

    // Initially missing
    assert!(engine
        .get(Partition::Node, key)
        .expect("get failed")
        .is_none());
    assert!(!engine
        .contains_key(Partition::Node, key)
        .expect("contains_key failed"));

    // Put
    engine.put(Partition::Node, key, value).expect("put failed");

    // Get
    let result = engine.get(Partition::Node, key).expect("get failed");
    assert_eq!(result.as_deref(), Some(value.as_slice()));
    assert!(engine
        .contains_key(Partition::Node, key)
        .expect("contains_key failed"));

    // Delete
    engine.delete(Partition::Node, key).expect("delete failed");
    assert!(engine
        .get(Partition::Node, key)
        .expect("get failed")
        .is_none());
}

#[test]
fn partitions_are_isolated() {
    let (engine, _dir) = test_engine();
    let key = b"shared_key";

    engine
        .put(Partition::Node, key, b"node_val")
        .expect("put failed");
    engine
        .put(Partition::Schema, key, b"schema_val")
        .expect("put failed");

    let node_val = engine.get(Partition::Node, key).expect("get failed");
    let schema_val = engine.get(Partition::Schema, key).expect("get failed");
    let adj_val = engine.get(Partition::Adj, key).expect("get failed");

    assert_eq!(node_val.as_deref(), Some(b"node_val".as_slice()));
    assert_eq!(schema_val.as_deref(), Some(b"schema_val".as_slice()));
    assert!(adj_val.is_none());
}

#[test]
fn overwrite_existing_key() {
    let (engine, _dir) = test_engine();
    let key = b"overwrite_me";

    engine.put(Partition::Node, key, b"v1").expect("put failed");
    engine.put(Partition::Node, key, b"v2").expect("put failed");

    let result = engine.get(Partition::Node, key).expect("get failed");
    assert_eq!(result.as_deref(), Some(b"v2".as_slice()));
}

#[test]
fn delete_nonexistent_key_is_ok() {
    let (engine, _dir) = test_engine();
    engine
        .delete(Partition::Node, b"no_such_key")
        .expect("delete should succeed");
}

#[test]
fn empty_value_nonempty_key() {
    let (engine, _dir) = test_engine();
    engine
        .put(Partition::Adj, b"k", b"")
        .expect("put empty value failed");

    let result = engine.get(Partition::Adj, b"k").expect("get failed");
    assert_eq!(result.as_deref(), Some(b"".as_slice()));
}

#[test]
fn large_value() {
    let (engine, _dir) = test_engine();
    let key = b"big";
    // 1MB value — tests KV separation threshold behavior.
    let value = vec![0xABu8; 1024 * 1024];

    engine
        .put(Partition::Blob, key, &value)
        .expect("put large failed");

    let result = engine.get(Partition::Blob, key).expect("get large failed");
    assert_eq!(result.as_deref(), Some(value.as_slice()));
}

#[test]
fn persist_and_reopen() {
    let dir = TempDir::new().expect("failed to create temp dir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);

    {
        let engine = StorageEngine::open(&config).expect("open failed");
        engine
            .put(Partition::Node, b"persist_key", b"persist_val")
            .expect("put failed");
        engine.persist().expect("persist failed");
    }

    {
        let engine = StorageEngine::open(&config).expect("reopen failed");
        let result = engine
            .get(Partition::Node, b"persist_key")
            .expect("get failed");
        assert_eq!(result.as_deref(), Some(b"persist_val".as_slice()));
    }
}

#[test]
fn disk_space_reports_nonzero_after_writes() {
    let (engine, _dir) = test_engine();

    for i in 0..1000u32 {
        engine
            .put(Partition::Node, &i.to_be_bytes(), &[0xFFu8; 1024])
            .expect("put failed");
    }
    engine.persist().expect("persist failed");

    let space = engine.disk_space().expect("disk_space failed");
    assert!(
        space > 0,
        "disk space should be nonzero after 1MB of writes"
    );
}

#[test]
fn engine_with_no_compression() {
    let dir = TempDir::new().expect("failed to create temp dir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.compression = crate::engine::config::CompressionConfig {
        hot_codec: crate::engine::config::CompressionCodec::None,
        cold_codec: crate::engine::config::CompressionCodec::None,
        cold_level_threshold: 4,
    };
    let engine = StorageEngine::open(&config).expect("open with no compression");
    engine
        .put(Partition::Node, b"k1", b"v1")
        .expect("put failed");
    let result = engine.get(Partition::Node, b"k1").expect("get failed");
    assert_eq!(result.as_deref(), Some(b"v1".as_slice()));
}

#[test]
fn engine_with_partition_compression_overrides() {
    let dir = TempDir::new().expect("failed to create temp dir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.partition_compression = Some(vec![(
        Partition::Blob,
        crate::engine::config::CompressionCodec::None,
    )]);
    let engine = StorageEngine::open(&config).expect("open with overrides");

    engine
        .put(Partition::Node, b"n1", b"node_data")
        .expect("put node");
    engine
        .put(Partition::Blob, b"b1", b"blob_data")
        .expect("put blob");

    assert_eq!(
        engine
            .get(Partition::Node, b"n1")
            .expect("get node")
            .as_deref(),
        Some(b"node_data".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::Blob, b"b1")
            .expect("get blob")
            .as_deref(),
        Some(b"blob_data".as_slice())
    );
}

#[test]
fn engine_compressed_data_survives_reopen() {
    let dir = TempDir::new().expect("failed to create temp dir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);

    {
        let engine = StorageEngine::open(&config).expect("open");
        for i in 0..100u32 {
            let key = format!("key_{i:06}");
            let value = format!("value_{i:06}_payload_with_some_extra_data_for_compression");
            engine
                .put(Partition::Node, key.as_bytes(), value.as_bytes())
                .expect("put");
        }
        engine.persist().expect("persist");
    }

    {
        let engine = StorageEngine::open(&config).expect("reopen");
        for i in 0..100u32 {
            let key = format!("key_{i:06}");
            let expected = format!("value_{i:06}_payload_with_some_extra_data_for_compression");
            let result = engine.get(Partition::Node, key.as_bytes()).expect("get");
            assert_eq!(
                result.as_deref(),
                Some(expected.as_bytes()),
                "data mismatch at key {i}"
            );
        }
    }
}

#[test]
fn all_partitions_support_crud() {
    let (engine, _dir) = test_engine();

    for &part in Partition::all() {
        let key = format!("key_{:?}", part);
        let value = format!("val_{:?}", part);

        engine
            .put(part, key.as_bytes(), value.as_bytes())
            .expect("put failed");
        let result = engine.get(part, key.as_bytes()).expect("get failed");
        assert_eq!(
            result.as_deref(),
            Some(value.as_bytes()),
            "CRUD failed for partition {:?}",
            part
        );

        engine.delete(part, key.as_bytes()).expect("delete failed");
        assert!(engine
            .get(part, key.as_bytes())
            .expect("get failed")
            .is_none());
    }
}
