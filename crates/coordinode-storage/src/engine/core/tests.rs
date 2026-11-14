use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use tempfile::TempDir;

/// The inspector borrows the exact LSM buffer, not a Bytes/Vec copy. It
/// must select the same old value after a later write and handle absence.
#[test]
fn snapshot_inspector_borrows_the_lsm_value_without_copying() {
    let (engine, _dir) = test_engine();
    let key = b"borrowed-value";
    let value = vec![7; 4096];
    engine.put(Partition::Node, key, &value).expect("put");
    let (snapshot, _pin) = engine.pin_snapshot();
    let source = engine
        .tree(Partition::Node)
        .expect("tree")
        .get(key, snapshot)
        .expect("get")
        .expect("value");
    engine.put(Partition::Node, key, b"new").expect("overwrite");
    engine
        .with_snapshot_value(&snapshot, Partition::Node, key, |bytes| {
            let bytes = bytes.expect("old value");
            assert_eq!(bytes, value);
            assert_eq!(bytes.as_ptr(), source.as_ptr(), "payload was copied");
        })
        .expect("inspect");
    assert!(
        engine
            .with_snapshot_value(&snapshot, Partition::Node, b"missing", |bytes| bytes
                .is_none())
            .expect("absence")
    );
}

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
/// outside the range intact — the MVCC range-tombstone path.
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

/// `multi_get` returns one slot per input key in order: present keys carry
/// their value, absent keys carry `None`, and a cache-resident key resolves
/// identically to a cache-miss key. Mirrors the per-key `get` contract.
#[test]
fn multi_get_preserves_order_and_absence() {
    let engine = test_engine_memfs();
    engine.put(Partition::Node, b"node:k1", b"v1").expect("put");
    engine.put(Partition::Node, b"node:k2", b"v2").expect("put");
    engine.put(Partition::Node, b"node:k3", b"v3").expect("put");

    // Warm the cache for one key so the batch mixes a cache hit with misses.
    let _ = engine.get(Partition::Node, b"node:k2").expect("get");

    let keys: Vec<&[u8]> = vec![b"node:k1", b"absent", b"node:k3", b"node:k2"];
    let got = engine.multi_get(Partition::Node, &keys).expect("multi_get");

    assert_eq!(got.len(), 4);
    assert_eq!(got[0].as_deref(), Some(&b"v1"[..]));
    assert_eq!(got[1], None, "absent key must be None");
    assert_eq!(got[2].as_deref(), Some(&b"v3"[..]));
    assert_eq!(got[3].as_deref(), Some(&b"v2"[..]), "cache-hit slot");

    // Each multi_get slot agrees with an individual get on the same key.
    for k in &keys {
        let single = engine.get(Partition::Node, k).expect("get");
        let pos = keys.iter().position(|x| x == k).expect("key in set");
        let batch = got[pos].clone();
        assert_eq!(single, batch);
    }
}

/// Empty key set is a no-op returning an empty vector (no tree access).
#[test]
fn multi_get_empty_is_empty() {
    let engine = test_engine_memfs();
    let got = engine.multi_get(Partition::Node, &[]).expect("multi_get");
    assert!(got.is_empty());
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
    assert!(
        engine
            .get(Partition::Node, b"node:00:99999999")
            .expect("get")
            .is_none()
    );
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
    assert!(
        engine
            .get(Partition::Node, key)
            .expect("get failed")
            .is_none()
    );
    assert!(
        !engine
            .contains_key(Partition::Node, key)
            .expect("contains_key failed")
    );

    // Put
    engine.put(Partition::Node, key, value).expect("put failed");

    // Get
    let result = engine.get(Partition::Node, key).expect("get failed");
    assert_eq!(result.as_deref(), Some(value.as_slice()));
    assert!(
        engine
            .contains_key(Partition::Node, key)
            .expect("contains_key failed")
    );

    // Delete
    engine.delete(Partition::Node, key).expect("delete failed");
    assert!(
        engine
            .get(Partition::Node, key)
            .expect("get failed")
            .is_none()
    );
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

/// The default build carries zstd and uses it where the architecture puts it:
/// lz4 on the hot levels, zstd from the cold threshold down.
#[cfg(feature = "zstd")]
#[test]
fn default_compression_is_lz4_hot_and_zstd_cold() {
    use crate::engine::config::{CompressionCodec, CompressionConfig};

    let config = CompressionConfig::default();
    assert_eq!(config.hot_codec, CompressionCodec::Lz4);
    assert_eq!(config.cold_codec, CompressionCodec::Zstd(3));
    assert_eq!(config.cold_level_threshold, 4);
}

/// A partition compressed with zstd alone is written, flushed to a table and
/// read back after a reopen, so the codec is really linked and really used.
#[cfg(feature = "zstd")]
#[test]
fn zstd_compressed_data_survives_flush_and_reopen() {
    let dir = TempDir::new().expect("failed to create temp dir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.partition_compression = Some(vec![(
        Partition::Node,
        crate::engine::config::CompressionCodec::Zstd(9),
    )]);
    let value = |i: u32| format!("value_{i:06}_payload_with_some_extra_data_for_compression");

    {
        let engine = StorageEngine::open(&config).expect("open");
        for i in 0..200u32 {
            engine
                .put(
                    Partition::Node,
                    format!("key_{i:06}").as_bytes(),
                    value(i).as_bytes(),
                )
                .expect("put");
        }
        engine
            .tree(Partition::Node)
            .expect("tree")
            .flush_active_memtable(0)
            .expect("flush to a table");
        engine.persist().expect("persist");
    }

    let engine = StorageEngine::open(&config).expect("reopen");
    for i in 0..200u32 {
        let got = engine
            .get(Partition::Node, format!("key_{i:06}").as_bytes())
            .expect("get");
        assert_eq!(got.as_deref(), Some(value(i).as_bytes()), "key {i}");
    }
}

/// The configured codecs are the ones the tables are really written with: a
/// table flushed under zstd 22 on every level records zstd 22 for its data
/// blocks, a partition override records its own codec, and both read back
/// after a reopen.
#[cfg(feature = "zstd")]
#[test]
fn configured_codecs_are_written_into_the_tables() {
    use crate::engine::config::{CompressionCodec, CompressionConfig};

    let dir = TempDir::new().expect("failed to create temp dir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.compression = CompressionConfig {
        hot_codec: CompressionCodec::zstd(22).expect("level"),
        cold_codec: CompressionCodec::zstd(22).expect("level"),
        cold_level_threshold: 4,
    };
    config.partition_compression = Some(vec![(Partition::Idx, CompressionCodec::Lz4)]);
    let value = |i: u32| format!("value_{i:06}_payload_with_some_extra_data_for_compression");

    {
        let engine = StorageEngine::open(&config).expect("open");
        for part in [Partition::Node, Partition::Idx] {
            for i in 0..200u32 {
                engine
                    .put(part, format!("key_{i:06}").as_bytes(), value(i).as_bytes())
                    .expect("put");
            }
            engine
                .tree(part)
                .expect("tree")
                .flush_active_memtable(0)
                .expect("flush to a table");
        }
        engine.persist().expect("persist");
    }

    let codecs_of = |part: Partition| -> Vec<lsm_tree::CompressionType> {
        let mut found = Vec::new();
        let mut dirs = vec![dir.path().join(part.name())];
        while let Some(at) = dirs.pop() {
            for entry in std::fs::read_dir(&at).expect("read dir").flatten() {
                let path = entry.path();
                if path.is_dir() {
                    dirs.push(path);
                } else if let Ok(props) = lsm_tree::inspect::read_table_properties(&path) {
                    found.push(props.data_block_compression);
                }
            }
        }
        found
    };
    let node = codecs_of(Partition::Node);
    assert!(!node.is_empty(), "the node partition has a table");
    assert!(
        node.iter()
            .all(|c| *c == lsm_tree::CompressionType::Zstd(22)),
        "{node:?}"
    );
    let idx = codecs_of(Partition::Idx);
    assert!(!idx.is_empty(), "the idx partition has a table");
    assert!(
        idx.iter().all(|c| *c == lsm_tree::CompressionType::Lz4),
        "{idx:?}"
    );

    let engine = StorageEngine::open(&config).expect("reopen");
    for part in [Partition::Node, Partition::Idx] {
        for i in 0..200u32 {
            let got = engine
                .get(part, format!("key_{i:06}").as_bytes())
                .expect("get");
            assert_eq!(got.as_deref(), Some(value(i).as_bytes()), "{part:?} {i}");
        }
    }
}

/// A zstd level outside what the table format can store is refused when the
/// configuration is built, not when the first block is written.
#[cfg(feature = "zstd")]
#[test]
fn zstd_level_out_of_range_is_rejected() {
    use crate::engine::config::CompressionCodec;

    assert!(CompressionCodec::zstd(22).is_ok());
    assert!(CompressionCodec::zstd(23).is_err());
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
        assert!(
            engine
                .get(part, key.as_bytes())
                .expect("get failed")
                .is_none()
        );
    }
}

/// Remove every manifest snapshot (`v{id}`) and edit log (`edits-{id}`) of
/// the tree at `tree_dir`, its tables left in place: the manifest loss an
/// open repairs.
pub(super) fn lose_manifest(tree_dir: &std::path::Path) {
    let numbered = |rest: Option<&str>| {
        rest.is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()))
    };
    let mut lost = 0;
    for entry in std::fs::read_dir(tree_dir).expect("read tree dir") {
        let entry = entry.expect("dir entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if numbered(name.strip_prefix('v')) || numbered(name.strip_prefix("edits-")) {
            std::fs::remove_file(entry.path()).expect("remove manifest file");
            lost += 1;
        }
    }
    assert!(
        lost > 0,
        "layout changed: no manifest file in {}",
        tree_dir.display()
    );
}

/// A power cut mid-append leaves a torn last record in a tree's manifest edit
/// log. That record was never acknowledged, so the open rolls it back and
/// opens at the durable prefix: no repair (which would rebuild the manifest
/// from the tables and forget the tree's retention floor), every written row
/// still there.
#[test]
fn a_torn_manifest_tail_opens_without_a_repair() {
    use lsm_tree::AbstractTree;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let tree_dir = dir.path().join(Partition::Node.name());
    // The edit logs of the tree with their sizes.
    let logs = || -> Vec<(std::path::PathBuf, u64)> {
        std::fs::read_dir(&tree_dir)
            .expect("read tree dir")
            .map(|entry| entry.expect("dir entry").path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .and_then(|name| name.strip_prefix("edits-"))
                    .is_some_and(|id| id.bytes().all(|b| b.is_ascii_digit()))
            })
            .map(|path| {
                let len = std::fs::metadata(&path).expect("metadata").len();
                (path, len)
            })
            .collect()
    };
    let (log, durable) = {
        let engine = StorageEngine::open(&config).expect("initial open");
        let tree = engine.tree(Partition::Node).expect("tree");
        engine
            .put(Partition::Node, b"node:torn-tail", b"survives")
            .expect("put");
        // A store with a journal leaves its flushes' durability to a
        // barrier: each barrier writes one manifest record.
        tree.flush_active_memtable(0).expect("flush");
        tree.persist_flushes().expect("barrier");
        let before = logs();
        // A second barrier appends one record: the one the power cut tears.
        engine
            .put(Partition::Node, b"node:torn-away", b"unacknowledged")
            .expect("put");
        tree.flush_active_memtable(0).expect("flush");
        tree.persist_flushes().expect("barrier");
        logs()
            .into_iter()
            .find_map(|(path, len)| {
                let was = before.iter().find(|(p, _)| p == &path)?.1;
                (len > was).then_some((path, was))
            })
            .expect("the second flush appended to an edit log")
    };
    // Five bytes of the torn record reached the disk.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&log)
        .expect("open edit log")
        .set_len(durable + 5)
        .expect("tear the tail");

    let engine = StorageEngine::open(&config).expect("open after a torn tail");
    assert!(
        engine.open_repairs().is_empty(),
        "a torn tail was repaired: {:?}",
        engine.open_repairs()
    );
    assert_eq!(
        engine
            .get(Partition::Node, b"node:torn-tail")
            .expect("get")
            .as_deref(),
        Some(&b"survives"[..])
    );
}

/// Losing a partition's manifest (its tables intact) must not make the whole
/// engine unopenable: open repairs the tree from its SSTs, reports the repair
/// so the caller can decide whether a replay is owed, and once the open has
/// replayed what it owed the tree stops reporting it.
#[test]
fn open_repairs_partition_with_lost_manifest() {
    use lsm_tree::AbstractTree;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);

    // Seed a flushed SST in the Node partition, then close the engine.
    {
        let engine = StorageEngine::open(&config).expect("initial open");
        engine
            .put(Partition::Node, b"node:manifest-loss", b"survives")
            .expect("put");
        engine
            .tree(Partition::Node)
            .expect("tree")
            .flush_active_memtable(0)
            .expect("flush");
    }

    // Manifest loss: tables present, every snapshot and edit log gone.
    lose_manifest(&dir.path().join(Partition::Node.name()));

    // Reopen: the engine must come up (repairing the Node tree), the data
    // must still be readable, and the repair must be reported.
    {
        let engine = StorageEngine::open(&config).expect("open after manifest loss");
        assert_eq!(
            engine
                .get(Partition::Node, b"node:manifest-loss")
                .expect("get")
                .as_deref(),
            Some(&b"survives"[..])
        );
        let repairs = engine.open_repairs();
        assert_eq!(repairs.len(), 1, "exactly one partition was repaired");
        assert_eq!(repairs[0].partition, Partition::Node);
        // Nothing was lost (every SST intact), so no replay is owed.
        assert!(matches!(
            repairs[0].replay_scope,
            lsm_tree::WalReplayScope::TailOnly
        ));
    }

    // The tree keeps a report until it is acknowledged; the open settled
    // this one, so the next open reports nothing.
    let engine = StorageEngine::open(&config).expect("reopen");
    assert!(
        engine.open_repairs().is_empty(),
        "a lossless repair is not reported again: {:?}",
        engine.open_repairs()
    );
}

/// Repair at open makes every recovered table a run of its own in L0, so a
/// partition of more than 255 tables comes back as a level wider than a
/// manifest snapshot used to count. The repaired engine must open, keep
/// writing on top of that level, and open again afterwards.
#[test]
fn open_repairs_partition_with_more_than_255_tables() {
    use lsm_tree::AbstractTree;

    const TABLES: usize = 300;

    let dir = TempDir::new().expect("tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    // No compaction worker, so every flush stays a table of its own.
    config.compaction_workers = 0;

    {
        let engine = StorageEngine::open(&config).expect("initial open");
        for i in 0..TABLES {
            let key = format!("k{i:05}");
            engine
                .put(Partition::Node, key.as_bytes(), key.as_bytes())
                .expect("put");
            engine
                .tree(Partition::Node)
                .expect("tree")
                .flush_active_memtable(0)
                .expect("flush");
        }
    }

    lose_manifest(&dir.path().join(Partition::Node.name()));

    {
        let engine = StorageEngine::open(&config).expect("open after manifest loss");
        assert_eq!(engine.open_repairs().len(), 1, "the Node tree was repaired");
        assert!(
            engine.tree(Partition::Node).expect("tree").l0_run_count() > 255,
            "repair must have rebuilt a level wider than a byte can count",
        );
        engine
            .put(Partition::Node, b"after-repair", b"written")
            .expect("put after repair");
        engine.persist().expect("flush after repair");
    }

    let engine = StorageEngine::open(&config).expect("reopen after repair");
    assert!(
        engine.open_repairs().is_empty(),
        "the repaired tree opens without another repair",
    );
    for i in 0..TABLES {
        let key = format!("k{i:05}");
        assert_eq!(
            engine
                .get(Partition::Node, key.as_bytes())
                .expect("get")
                .as_deref(),
            Some(key.as_bytes()),
            "{key} must survive the repair",
        );
    }
    assert_eq!(
        engine
            .get(Partition::Node, b"after-repair")
            .expect("get")
            .as_deref(),
        Some(&b"written"[..]),
    );
}

/// Fail-closed: an open failure that is NOT positively-identified structural
/// damage (here: the directory lock held by a live engine) must propagate as
/// an error, never trigger an in-place repair. Repairing on an ambiguous
/// error could rebuild the manifest around a misconfigured view and drop
/// healthy data.
#[test]
fn open_does_not_repair_on_held_directory_lock() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("first open");
    engine
        .put(Partition::Node, b"node:locked", b"v")
        .expect("put");

    // Second open against the same directory: the held lock is an
    // environment error, not damage; it must surface, and the live
    // engine's data must remain untouched by any repair attempt.
    let second = StorageEngine::open(&config);
    assert!(second.is_err(), "double open must fail, not repair");
    assert_eq!(
        engine
            .get(Partition::Node, b"node:locked")
            .expect("get")
            .as_deref(),
        Some(&b"v"[..])
    );
    assert!(engine.open_repairs().is_empty());
}

/// The oplog trim contract: a record may be dropped only once its write is
/// durable in the partition trees (and not needed by the checkpoint floor).
/// Time-based retention alone must never discard a sealed segment whose
/// entries still exist only in memtables — that would lose them on crash.
#[test]
fn oplog_purge_never_drops_entries_not_yet_durable() {
    use crate::engine::oplog_journal::OplogJournalConfig;
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use coordinode_core::txn::timestamp::TimestampOracle;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    // Rotate every 2 entries so the appends below land in SEALED segments
    // (the purge never touches the active one); 1-second retention makes
    // every sealed segment time-expired immediately.
    let journal = OplogJournalConfig {
        retention_secs: 1,
        max_segment_entries: 2,
        ..OplogJournalConfig::default()
    };
    let engine = StorageEngine::open_embedded_with_journal(&config, oracle, journal)
        .expect("open_embedded_with_journal");

    for i in 0..6u64 {
        let key = format!("node:purge:{i}");
        engine
            .oplog_append(
                &[Mutation::Put {
                    partition: PartitionId::Node,
                    key: key.into_bytes(),
                    value: b"journal-only".to_vec(),
                }],
                i + 1,
            )
            .expect("oplog_append")
            .expect("journal active");
        // Deliberately NOT applied to the tree: the journal is the only copy.
    }

    // Far-future clock: every sealed segment is far outside the time window.
    let far_future = 4_000_000_000u64;
    engine
        .oplog_purge_expired(far_future, u64::MAX)
        .expect("purge");

    let entries = engine
        .oplog_read_since(0)
        .expect("read_since")
        .expect("journal active");
    let seen: Vec<u64> = entries.iter().map(|e| e.ts).collect();
    assert!(
        (1..=6).all(|ts| seen.contains(&ts)),
        "non-durable journal entries were purged; surviving ts = {seen:?}"
    );
}

/// The other direction of the durability guard: once the writes ARE persisted
/// (and no checkpoint floor holds them), time-expired sealed segments must be
/// collected — the guard may not turn into unbounded retention.
#[test]
fn oplog_purge_collects_durable_expired_segments() {
    use crate::engine::oplog_journal::OplogJournalConfig;
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use coordinode_core::txn::timestamp::TimestampOracle;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let journal = OplogJournalConfig {
        retention_secs: 1,
        max_segment_entries: 2,
        ..OplogJournalConfig::default()
    };
    let engine = StorageEngine::open_embedded_with_journal(&config, oracle.clone(), journal)
        .expect("open_embedded_with_journal");

    for i in 0..6u64 {
        let key = format!("node:purge:{i}");
        // Journalled and applied: the purge itself folds and persists the
        // coverage that makes these entries durable.
        engine
            .commit_journaled(
                &[Mutation::Put {
                    partition: PartitionId::Node,
                    key: key.into_bytes(),
                    value: b"applied".to_vec(),
                }],
                oracle.next().as_raw(),
            )
            .expect("commit_journaled");
    }

    let far_future = 4_000_000_000u64;
    let purged = engine
        .oplog_purge_expired(far_future, u64::MAX)
        .expect("purge");
    assert!(
        purged > 0,
        "durable time-expired sealed segments must be collected"
    );
}

/// A store whose journal rotates every two entries and expires at once, with
/// six durable entries written: every sealed segment is collectable unless
/// something keeps it.
fn journal_with_six_collectable_entries() -> (StorageEngine, TempDir) {
    use crate::engine::oplog_journal::OplogJournalConfig;
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use coordinode_core::txn::timestamp::TimestampOracle;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let journal = OplogJournalConfig {
        retention_secs: 1,
        max_segment_entries: 2,
        ..OplogJournalConfig::default()
    };
    let engine = StorageEngine::open_embedded_with_journal(&config, oracle.clone(), journal)
        .expect("open_embedded_with_journal");
    for i in 0..6u64 {
        engine
            .commit_journaled(
                &[Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:kept:{i}").into_bytes(),
                    value: b"applied".to_vec(),
                }],
                oracle.next().as_raw(),
            )
            .expect("commit_journaled");
    }
    (engine, dir)
}

/// The lowest index the journal still holds.
fn first_journal_index(engine: &StorageEngine) -> u64 {
    engine
        .oplog_read_since(0)
        .expect("read")
        .expect("journal")
        .first()
        .map_or(u64::MAX, |e| e.index)
}

/// Neither expiry nor footprint reclamation removes journal entries a
/// consumer's published floor needs, and lowering nothing else, both remove
/// what lies below it.
#[test]
fn journal_purges_keep_what_a_consumer_floor_needs() {
    let (engine, _dir) = journal_with_six_collectable_entries();
    engine.set_consumer_journal_floor(2);
    engine
        .oplog_purge_expired(4_000_000_000, u64::MAX)
        .expect("purge");
    assert!(
        first_journal_index(&engine) <= 2,
        "expiry removed entries the consumer floor keeps"
    );
    engine.reclaim_footprint();
    assert!(
        first_journal_index(&engine) <= 2,
        "reclamation removed entries the consumer floor keeps"
    );

    engine.set_consumer_journal_floor(u64::MAX);
    engine
        .oplog_purge_expired(4_000_000_000, u64::MAX)
        .expect("purge");
    assert!(
        first_journal_index(&engine) > 2,
        "with the floor released the expired entries go"
    );
}

/// A hold keeps the journal from its index on while it lives, and a hold is
/// refused below what the journal still holds: a purge either ran before the
/// hold was asked for, or it keeps what the hold names.
#[test]
fn a_journal_hold_keeps_its_index_and_is_refused_below_the_journal() {
    let (engine, _dir) = journal_with_six_collectable_entries();
    let hold = engine
        .hold_journal_from(1)
        .expect("hold")
        .expect("index 1 is still held");
    assert_eq!(hold.index(), 1);
    engine
        .oplog_purge_expired(4_000_000_000, u64::MAX)
        .expect("purge");
    engine.reclaim_footprint();
    assert!(first_journal_index(&engine) <= 1, "a purge passed a hold");

    drop(hold);
    engine
        .oplog_purge_expired(4_000_000_000, u64::MAX)
        .expect("purge");
    let first = first_journal_index(&engine);
    assert!(first > 1, "the dropped hold still kept the journal");
    assert!(
        engine.hold_journal_from(0).expect("hold").is_none(),
        "a hold below what the journal holds was granted"
    );
    assert!(
        engine.hold_journal_from(first).expect("hold").is_some(),
        "a hold at the journal's first entry was refused"
    );
}

/// A store keeping no journal of its own refuses a journal hold rather than
/// pretend to keep anything.
#[test]
fn a_store_without_a_journal_refuses_a_journal_hold() {
    let dir = TempDir::new().expect("tempdir");
    let engine = StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open");
    assert!(matches!(
        engine.hold_journal_from(0),
        Err(StorageError::InvalidConfig(_))
    ));
    assert!(engine.journal_location().expect("location").is_none());
    assert!(engine.journal_next_index().expect("next").is_none());
}

/// End-to-end power-loss recovery under the crash-simulator filesystem: the
/// trees roll back to their last-fsynced content, and reopening the engine
/// replays the journal so every acknowledged write survives. This exercises
/// the real durability contract (journal fsyncs on append, trees flush
/// lazily) instead of assuming a graceful shutdown.
#[test]
fn power_loss_recovery_replays_journal_over_rolled_back_trees() {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use coordinode_core::txn::timestamp::TimestampOracle;
    use lsm_tree::fs::{CrashFs, StdFs};

    let dir = TempDir::new().expect("tempdir");
    let crash_fs = Arc::new(CrashFs::new(StdFs));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
    .with_fs(crash_fs.clone());

    let write = |engine: &StorageEngine, oracle: &TimestampOracle, i: u64, value: &[u8]| {
        let key = format!("node:crash:{i}");
        engine
            .commit_journaled(
                &[Mutation::Put {
                    partition: PartitionId::Node,
                    key: key.into_bytes(),
                    value: value.to_vec(),
                }],
                oracle.next().as_raw(),
            )
            .expect("commit_journaled");
    };

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&config, oracle.clone()).expect("open before crash");
        for i in 0..4u64 {
            write(&engine, &oracle, i, b"flushed");
        }
        engine.persist().expect("persist first half");
        for i in 4..8u64 {
            write(&engine, &oracle, i, b"memtable-only");
        }
        // No persist for the second half: those rows exist only in memtables
        // and the fsynced journal when the "power" goes out.
    }
    crash_fs.crash();

    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&config, oracle).expect("open after crash");
    for i in 0..8u64 {
        let key = format!("node:crash:{i}");
        let expected: &[u8] = if i < 4 { b"flushed" } else { b"memtable-only" };
        assert_eq!(
            engine
                .get(Partition::Node, key.as_bytes())
                .expect("get")
                .as_deref(),
            Some(expected),
            "row {i} lost across the simulated power loss"
        );
    }
}

/// A store nobody has written to holds no data of its own, and one written to
/// holds it in the partition it was written to.
///
/// This is the question a node answers before joining a group that already has
/// data, so the false answer has to be exact: state the engine keeps about
/// itself must not read as the operator's data, or a fresh node could never
/// join, and the operator's data must not read as engine state, or a join
/// would silently replace it.
/// An order-sensitive merge chain means the same thing after every stage the
/// bytes pass through: in the memtable, after a flush, after a compaction and
/// after the directory is closed and opened again.
///
/// The operands of a posting list do not commute, and each stage is a chance
/// to reorder them: a flush writes them out, a compaction folds a prefix or
/// keeps it, and a reopen replays what the log and the tables hold. The
/// answer is read at every stage rather than only at the end, so a stage that
/// changed it is named by the assertion that fails.
#[test]
fn a_merge_chain_reads_the_same_through_flush_compaction_and_reopen() {
    use coordinode_core::graph::edge::PostingList;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);

    let key = b"adj:FOLLOWS:out:42";
    let read = |engine: &StorageEngine| -> Vec<u64> {
        match engine.get(Partition::Adj, key).expect("get") {
            Some(bytes) => PostingList::from_bytes(&bytes)
                .expect("decode")
                .as_slice()
                .to_vec(),
            None => Vec::new(),
        }
    };

    {
        let engine = StorageEngine::open(&config).expect("open");

        // A chain whose meaning depends on its order: 2 is added, removed and
        // added again, 3 is added and removed. Only the order says that 2
        // survives and 3 does not.
        for operand in [
            crate::engine::merge::encode_add(1),
            crate::engine::merge::encode_add(2),
            crate::engine::merge::encode_add(3),
            crate::engine::merge::encode_remove(2),
            crate::engine::merge::encode_remove(3),
            crate::engine::merge::encode_add(2),
        ] {
            engine.merge(Partition::Adj, key, &operand).expect("merge");
        }

        assert_eq!(read(&engine), vec![1, 2], "in the memtable");

        engine.persist().expect("flush");
        assert_eq!(read(&engine), vec![1, 2], "after a flush");

        engine
            .major_compact(Partition::Adj)
            .expect("compact the partition");
        assert_eq!(read(&engine), vec![1, 2], "after a compaction");
    }

    // Closed and opened again: what the directory holds is what it said.
    let reopened = StorageEngine::open(&config).expect("reopen");
    assert_eq!(read(&reopened), vec![1, 2], "after a reopen");
}

/// A scan resolves a merge chain the same way a point read does, forwards,
/// backwards and at a snapshot.
///
/// The point read and the scan are different code paths over the same
/// operands, and a chain that a `get` folds correctly can still reach a scan
/// unfolded: an iterator that returned the raw operand, or the base without
/// the operands above it, would answer with a value no caller could tell from
/// a real one. The chain here is order-sensitive on purpose, so a path that
/// merges the operands in the wrong order fails rather than passing on a sum
/// that happens to commute.
#[test]
fn every_read_path_resolves_a_merge_chain() {
    use coordinode_core::graph::edge::PostingList;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    let key = b"adj:FOLLOWS:out:7";
    for operand in [
        crate::engine::merge::encode_add(1),
        crate::engine::merge::encode_add(2),
        crate::engine::merge::encode_remove(1),
        crate::engine::merge::encode_add(3),
    ] {
        engine.merge(Partition::Adj, key, &operand).expect("merge");
    }
    let expected = vec![2u64, 3];

    let decode = |bytes: &[u8]| {
        PostingList::from_bytes(bytes)
            .expect("decode")
            .as_slice()
            .to_vec()
    };

    // The point read, as the baseline every other path must match.
    let point = engine
        .get(Partition::Adj, key)
        .expect("get")
        .expect("present");
    assert_eq!(decode(&point), expected, "point read");

    // A forward prefix scan.
    let mut seen = Vec::new();
    for guard in engine.prefix_scan(Partition::Adj, b"adj:").expect("scan") {
        let (k, v) = guard.into_inner().expect("entry");
        if k.as_ref() == key {
            seen.push(decode(&v));
        }
    }
    assert_eq!(seen, vec![expected.clone()], "forward scan");

    // The reverse scan, which walks the same entries the other way.
    let mut seen_rev = Vec::new();
    for guard in engine
        .prefix_scan_rev(Partition::Adj, b"adj:")
        .expect("reverse scan")
    {
        let (k, v) = guard.into_inner().expect("entry");
        if k.as_ref() == key {
            seen_rev.push(decode(&v));
        }
    }
    assert_eq!(seen_rev, vec![expected.clone()], "reverse scan");

    // A snapshot scan taken after the whole chain is written sees all of it.
    let snapshot = engine.snapshot();
    let at_snapshot = engine
        .snapshot_prefix_scan(&snapshot, Partition::Adj, b"adj:")
        .expect("snapshot scan");
    let found: Vec<Vec<u64>> = at_snapshot
        .into_iter()
        .filter(|(k, _)| k.as_slice() == key)
        .map(|(_, v)| decode(&v))
        .collect();
    assert_eq!(found, vec![expected.clone()], "snapshot scan");

    // And the same after a flush, where the operands are in tables rather
    // than in the memtable the iterator merged them from.
    engine.persist().expect("flush");
    let mut after_flush = Vec::new();
    for guard in engine.prefix_scan(Partition::Adj, b"adj:").expect("scan") {
        let (k, v) = guard.into_inner().expect("entry");
        if k.as_ref() == key {
            after_flush.push(decode(&v));
        }
    }
    assert_eq!(after_flush, vec![expected], "forward scan after a flush");
}

#[test]
fn a_fresh_store_holds_no_user_data() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    let occupied: Vec<&str> = Partition::all()
        .iter()
        .filter(|p| {
            p.user_data_prefix().is_some_and(|prefix| {
                engine
                    .prefix_scan(**p, prefix)
                    .expect("scan")
                    .next()
                    .is_some()
            })
        })
        .map(|p| p.name())
        .collect();
    assert!(
        occupied.is_empty(),
        "a fresh store must hold nothing of the operator's, found keys in {occupied:?}"
    );
    assert!(!engine.holds_user_data().expect("read the fresh store"));

    // The engine's own state in the shared Schema tree stays invisible to the
    // question: routing metadata is written at open, and the Raft state
    // machine keeps its applied id beside it.
    engine
        .put(Partition::Schema, b"raft:sm:applied", b"7")
        .expect("put");
    assert!(
        !engine.holds_user_data().expect("read"),
        "state the deployment keeps about itself is not the operator's data"
    );

    engine
        .put(Partition::Schema, b"schema:label:Person", b"{}")
        .expect("put");
    assert!(
        engine.holds_user_data().expect("read"),
        "a label the operator defined is theirs"
    );

    let dir2 = TempDir::new().expect("tempdir");
    let engine2 = StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir2.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open");
    engine2
        .put(Partition::Node, b"node:00:00000001", b"mine")
        .expect("put");
    assert!(engine2.holds_user_data().expect("read the written store"));
}

/// `persist_partition` promises that every write to the partition before the
/// call is durable when it returns. A sync failing inside one call fails it;
/// the next call on a healthy disk writes what the failed one did not, and a
/// power cut after it keeps the write. Swept over every sync a persist makes,
/// for the node-local partition the Raft vote uses and for the schema
/// partition.
#[test]
fn a_persist_after_a_failed_one_keeps_the_write_across_a_power_cut() {
    use lsm_tree::fs::{CrashFs, Fault, FaultFs, FaultInjector, FaultOp, FaultRule, StdFs};
    let refuse = Fault::Error(lsm_tree::io::ErrorKind::Other);
    let mut reached_by = Vec::new();
    for (partition, failed) in [Partition::Raft, Partition::Schema]
        .into_iter()
        .flat_map(|p| {
            [
                FaultOp::SyncAll,
                FaultOp::SyncData,
                FaultOp::Open,
                FaultOp::Write,
                FaultOp::Rename,
            ]
            .map(|op| (p, op))
        })
    {
        let mut reached = 0;
        for k in 0.. {
            let dir = TempDir::new().expect("temp dir");
            let crash = Arc::new(CrashFs::new(StdFs));
            let faults = Arc::new(FaultInjector::new());
            let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
                "default",
                dir.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            )]);
            config.fs = Some(Arc::new(FaultFs::with_injector(
                CrashFs::clone(&crash),
                Arc::clone(&faults),
            )));
            let engine = StorageEngine::open(&config).expect("open");
            let key = b"meta:probe";
            engine.put(partition, key, b"v").expect("put");

            // Every matching operation on the partition's files from the k-th
            // on fails, not just one: a one-shot rule could be taken by a
            // background thread's operation on the same files, leaving the
            // persist unfailed. Background work failing too only leaves more
            // for the retry to write.
            faults.arm(
                FaultRule::new(failed, refuse)
                    .on_path(partition.name())
                    .skip(k),
            );
            let first = engine.persist_partition(partition);
            if first.is_ok() {
                // The operation to fail lies past the last one a persist makes.
                break;
            }
            reached += 1;
            // A healthy disk from here; the second call must write again. The
            // proof that it did is the value surviving the power cut below.
            faults.clear();
            let retry = engine.persist_partition(partition);
            assert!(
                retry.is_ok(),
                "{partition:?} {failed:?} {k}: retry: {retry:?}"
            );

            // Power cut: nothing written from here on is durable, and every
            // file falls back to its last sync.
            for op in [
                FaultOp::Open,
                FaultOp::Write,
                FaultOp::SyncAll,
                FaultOp::SyncData,
                FaultOp::Rename,
            ] {
                faults.arm(FaultRule::new(op, refuse));
            }
            drop(engine);
            faults.clear();
            crash.crash();

            let engine = StorageEngine::open(&config).expect("reopen");
            assert_eq!(
                engine.get(partition, key).expect("read").as_deref(),
                Some(&b"v"[..]),
                "{partition:?} {failed:?} {k}: the write acknowledged by the retry was lost"
            );
        }
        reached_by.push((partition, failed, reached));
    }
    // Every partition's persist opens, writes and syncs a table; it need not
    // rename one. Which sync it is depends on the platform and the file:
    // the data alone (fdatasync) or the data with its metadata.
    let reached = |partition: Partition, op: FaultOp| {
        reached_by
            .iter()
            .find(|(p, o, _)| *p == partition && *o == op)
            .map_or(0, |(_, _, n)| *n)
    };
    for partition in [Partition::Raft, Partition::Schema] {
        for op in [FaultOp::Open, FaultOp::Write] {
            assert!(
                reached(partition, op) > 0,
                "{partition:?}: no {op:?} of a persist was failed: {reached_by:?}"
            );
        }
        assert!(
            reached(partition, FaultOp::SyncAll) + reached(partition, FaultOp::SyncData) > 0,
            "{partition:?}: no sync of a persist was failed: {reached_by:?}"
        );
    }
}

/// The device flush a persist ends with reaches every endpoint, not only the
/// one holding the journal: a barrier orders writes on its own device alone,
/// so a tree level on a second disk is durable only once that disk's cache
/// is flushed too.
#[test]
fn a_persist_flushes_the_device_of_every_endpoint() {
    use lsm_tree::fs::SyncKind;

    let (hot, cold) = (TempDir::new().expect("hot"), TempDir::new().expect("cold"));
    let injector = Arc::new(lsm_tree::fs::FaultInjector::new());
    let mut config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    config.fs = Some(Arc::new(lsm_tree::fs::FaultFs::with_injector(
        lsm_tree::fs::StdFs,
        Arc::clone(&injector),
    )));
    let engine = StorageEngine::open(&config).expect("open");
    engine
        .put(Partition::Node, b"node:00:00000001", b"row")
        .expect("put");
    // Only what the persist below syncs counts.
    drop(injector.take_syncs());
    engine.persist().expect("flush");

    let syncs = injector.take_syncs();
    for endpoint in [hot.path(), cold.path()] {
        assert!(
            syncs
                .iter()
                .any(|(path, kind)| *kind == SyncKind::Device && path == endpoint),
            "no device flush under {endpoint:?}: {syncs:?}"
        );
    }
}

/// A table flushed before its oplog entries are purged must be as durable as
/// those entries. Under a full-flush oplog the trees sync their files with
/// barriers, which keep them in order, and a persist then flushes the device
/// caches under the endpoint once, after every file sync: without that last
/// step a power cut on macOS after a purge could lose acknowledged writes.
/// Under a plain-fsync oplog the trees sync plainly and nothing flushes the
/// device.
#[test]
fn partition_trees_sync_as_durably_as_the_oplog() {
    use crate::engine::config::SyncMethod;
    use lsm_tree::fs::{SyncKind, SyncMode};

    for (method, expected) in [
        (SyncMethod::Full, SyncMode::Barrier),
        (SyncMethod::OpenDatasync, SyncMode::Barrier),
        (SyncMethod::Fsync, SyncMode::Normal),
    ] {
        let dir = TempDir::new().expect("temp dir");
        let injector = Arc::new(lsm_tree::fs::FaultInjector::new());
        let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        config.fs = Some(Arc::new(lsm_tree::fs::FaultFs::with_injector(
            lsm_tree::fs::StdFs,
            Arc::clone(&injector),
        )));
        config.oplog_sync_method = method;
        let engine = StorageEngine::open(&config).expect("open");
        engine
            .put(Partition::Node, b"node:00:00000001", b"row")
            .expect("put");
        // Only what the persist below syncs counts.
        drop(injector.take_syncs());
        engine.persist().expect("flush");

        let modes = injector.sync_modes_for("node");
        assert!(!modes.is_empty(), "{method:?}: the flush synced nothing");
        assert!(
            modes.iter().all(|m| *m == expected),
            "{method:?}: tree files synced at {modes:?}, expected {expected:?}"
        );
        // A compaction the flush starts may sync its own manifest edit after
        // the persist returns; what the persist promises is the flush's table
        // and manifest edit, synced before the device flush.
        let syncs = injector.take_syncs();
        let devices: Vec<usize> = syncs
            .iter()
            .enumerate()
            .filter(|(_, (path, kind))| *kind == SyncKind::Device && path == dir.path())
            .map(|(at, _)| at)
            .collect();
        if expected == SyncMode::Barrier {
            assert!(
                !devices.is_empty(),
                "{method:?}: no device flush of the endpoint: {syncs:?}"
            );
            let device = devices[0];
            // Separators normalized: Windows paths name `node\tables\`.
            let synced_before = |part: &str| {
                syncs[..device]
                    .iter()
                    .any(|(path, _)| path.to_string_lossy().replace('\\', "/").contains(part))
            };
            assert!(
                synced_before("node/tables/") && synced_before("node/edits-"),
                "{method:?}: the flushed table and its manifest edit come after the device flush: {syncs:?}"
            );
        } else {
            assert!(
                devices.is_empty(),
                "{method:?}: a device flush the method did not ask for: {syncs:?}"
            );
        }
    }
}

/// A durable store in `dir`.
fn test_config(dir: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// Apply Raft entry `index` as the apply path does: its commits, then the
/// closure records it plans, with its coverage, then the plan installed.
fn apply_closure_entry(engine: &StorageEngine, index: u64, commits: &[u64], below: u64) {
    for &ts in commits {
        engine.note_raft_applied(index, ts);
    }
    if let Some((records, at)) = engine
        .closure_record(below, index, false)
        .expect("plan the closure records")
    {
        engine
            .apply_raft_proposal(&records, at, index, 0, |_| false)
            .expect("write the closure records");
    }
    engine
        .raise_closure_frontier(below, index)
        .expect("install the plan");
}

/// A log position closed long ago keeps a ceiling of its own commits however
/// many entries applied after it: never the highest commit applied since,
/// which would make a read at that position wait on later writers.
#[test]
fn an_old_closed_position_keeps_its_own_ceiling() {
    let engine = test_engine_memfs();
    apply_closure_entry(&engine, 1, &[10], 0);
    apply_closure_entry(&engine, 2, &[], 11);
    for index in 3..=2_000 {
        apply_closure_entry(&engine, index, &[10_000 + index], 0);
    }
    assert_eq!(engine.commit_ceiling_through(1).expect("ceiling"), 10);
}

/// Breakpoints piled up past the cache while the bound stalls, then a bound
/// closing an early prefix: every position answers its own ceiling, the
/// closed ones too, before and after a restart, which reads the head and
/// the newest breakpoints and seeks the rest. Before, the oldest open
/// positions merged, and a closed position 1 needed commit 20 still open.
#[test]
fn every_position_keeps_its_ceiling_past_the_cache_and_a_restart() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let positions =
        u64::try_from(crate::engine::closure_ledger::CACHED_BREAKPOINTS * 3).expect("small") + 1;
    {
        let engine = StorageEngine::open(&test_config(dir.path())).expect("open");
        for index in 1..=positions {
            apply_closure_entry(&engine, index, &[index * 10], 0);
        }
        apply_closure_entry(&engine, positions + 1, &[], 11);
        assert_eq!(engine.commit_ceiling_through(1).expect("ceiling"), 10);
        for index in 1..=positions {
            assert_eq!(
                engine.commit_ceiling_through(index).expect("ceiling"),
                index * 10,
                "position {index}"
            );
        }
        engine.persist().expect("persist");
    }
    let engine = StorageEngine::open(&test_config(dir.path())).expect("reopen");
    engine.reload_closure_frontier().expect("reload");
    for index in 1..=positions {
        assert_eq!(
            engine.commit_ceiling_through(index).expect("ceiling"),
            index * 10,
            "position {index} after a restart"
        );
    }
}

/// The breakpoint records the store holds, counted in the Schema tree's
/// reserved namespace.
fn stored_breakpoints(engine: &StorageEngine) -> u64 {
    use lsm_tree::AbstractTree;
    engine
        .tree(Partition::Schema)
        .expect("schema tree")
        .prefix(
            crate::engine::closure_ledger::BREAKPOINT_PREFIX,
            lsm_tree::SeqNo::MAX,
            None,
        )
        .count() as u64
}

/// How many breakpoint records the ledger counts as held.
fn ledger_held(engine: &StorageEngine) -> u64 {
    engine
        .closure_held
        .load(std::sync::atomic::Ordering::Acquire)
}

/// A clock-backed store in `dir` under a hard limit of `limit` bytes.
fn clocked_capped(
    dir: &std::path::Path,
    limit: u64,
    reserve: Option<u64>,
) -> (
    StorageEngine,
    Arc<coordinode_core::txn::timestamp::TimestampOracle>,
) {
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let mut endpoint =
        EndpointConfig::new("default", dir, Media::Hdd, Durability::Durable, Tier::Warm)
            .with_hard_limit_bytes(limit);
    if let Some(reserve) = reserve {
        endpoint = endpoint.with_recovery_reserve_bytes(reserve);
    }
    let engine = StorageEngine::open_with_oracle(
        &StorageConfig::with_endpoints(vec![endpoint]),
        Arc::clone(&oracle),
    )
    .expect("open");
    (engine, oracle)
}

/// Commit one client write as entry `index` the way a consensus member
/// does: admitted first (its breakpoint and that breakpoint's removal
/// included), applied, its closure records written, the admission given
/// back once the entry is applied.
fn admitted_write(
    engine: &StorageEngine,
    oracle: &coordinode_core::txn::timestamp::TimestampOracle,
    index: u64,
    value_len: usize,
) -> bool {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    let write = [Mutation::Put {
        partition: PartitionId::Node,
        key: format!("node:1:w{index}").into_bytes(),
        value: vec![7u8; value_len],
    }];
    let Ok(admitted) = engine.admit_proposal(&write, crate::engine::footprint::Admit::Client)
    else {
        return false;
    };
    let ts = oracle.next().as_raw();
    engine
        .apply_raft_proposal(&write, ts, index, 0, |_| false)
        .expect("apply");
    if let Some((records, at)) = engine
        .closure_record(0, index, false)
        .expect("plan the closure records")
    {
        engine
            .apply_raft_proposal(&records, at, index, 1, |_| false)
            .expect("write the closure records");
    }
    engine.raise_closure_frontier(0, index).expect("install");
    drop(admitted);
    true
}

/// A report of `engine`'s grant after `last_index` as a member sends it,
/// once the raise it asked for (if any) is stored: the pool topped up.
fn grant_report(
    engine: &StorageEngine,
    last_index: u64,
) -> crate::engine::footprint::grants::GrantReport {
    let footprints = engine.footprints();
    footprints.top_up_grant();
    let (report, raise) = footprints.grant_report(last_index);
    let Some(raised) = raise else {
        return report;
    };
    let stored = engine.store_client_grant(raised).is_ok();
    footprints.grant_horizon_stored(raised, stored);
    footprints.grant_report(last_index).0
}

/// Open `engine`'s client grant as a member does: the log applied through
/// `applied`, the group's applied client bytes `counted` (through, total),
/// and `entries` the client entries the log holds past `applied`.
fn open_grant(
    engine: &StorageEngine,
    applied: u64,
    counted: (u64, u64),
    entries: &[crate::engine::footprint::grants::GrantedEntry],
) -> u64 {
    let epoch = engine
        .open_client_grant(applied, counted.0, counted.1)
        .expect("open the grant");
    engine
        .footprints()
        .recover_granted(entries)
        .expect("hold the entries");
    engine
        .footprints()
        .recover_granted_owed()
        .expect("hold what is owed");
    epoch
}

/// What a member promised its leader is what it reported, not what it is
/// configured to grant: near its limit, a grant larger than the room left
/// promises only what it reported, and a restart holds that again (it fits:
/// the restart freed the pool that held it). Before, the open stored the
/// configured grant as the promise and the next open demanded all of it,
/// more than the endpoint could hold.
#[test]
fn a_restart_holds_what_was_reported_not_what_was_configured() {
    const LIMIT: u64 = crate::engine::config::MIN_HARD_LIMIT_BYTES * 2;
    let dir = tempfile::TempDir::new().expect("tempdir");
    let reported = {
        let (engine, _) = clocked_capped(dir.path(), LIMIT, None);
        engine.set_client_grant(64 << 20);
        assert_eq!(open_grant(&engine, 0, (0, 0), &[]), 1);
        let report = grant_report(&engine, 0);
        assert!(report.available > 0 && report.available < 64 << 20);
        assert_eq!(engine.footprints().grant_record().horizon, report.available);
        report.available
    };
    let (engine, _) = clocked_capped(dir.path(), LIMIT, None);
    engine.set_client_grant(64 << 20);
    let footprint = engine.footprints().get("default").expect("a limit");
    let before = footprint.admitted();
    assert_eq!(
        open_grant(&engine, 0, (0, 0), &[]),
        2,
        "reopens near the limit"
    );
    assert_eq!(
        footprint.admitted() - before,
        reported,
        "the room reported is held again before anything else, no more"
    );
}

/// A grant reported and spent on an entry applied before a crash (or held
/// by a snapshot installed in its place), with no report after it, owes
/// nothing on reopen: the group's applied total counts the entry. Before,
/// the reopen read only unapplied entries and held the whole report again.
#[test]
fn a_spent_grant_owes_nothing_after_a_restart() {
    use crate::engine::footprint::grants::GrantedEntry;
    const LIMIT: u64 = 64 << 20;
    let spent = GrantedEntry {
        index: 1,
        term: 1,
        bytes: 1 << 20,
    };
    let dir = tempfile::TempDir::new().expect("tempdir");
    {
        let (engine, _) = clocked_capped(dir.path(), LIMIT, None);
        // A batch of twice the target: 1 MiB, spent by the entry below.
        engine.set_client_grant(1 << 19);
        open_grant(&engine, 0, (0, 0), &[]);
        assert_eq!(grant_report(&engine, 0).available, 1 << 20);
        engine.footprints().stage_granted(&[spent]).expect("stage");
        engine.footprints().granted_appended(1, 1, 1 << 20);
        engine.footprints().applied(1, 1, 1 << 20);
        // Crash: no report after the entry.
    }
    let (engine, _) = clocked_capped(dir.path(), LIMIT, None);
    engine.set_client_grant(0);
    let footprint = engine.footprints().get("default").expect("a limit");
    let before = footprint.admitted();
    open_grant(&engine, 1, (1, 1 << 20), &[]);
    assert_eq!(
        footprint.admitted(),
        before,
        "a spent promise holds nothing"
    );
}

/// Breakpoints held past the cache owe their removal to the footprint once
/// their writes' admissions are given back; a restart owes it from the
/// open, before the ledger is loaded and before any other write is
/// admitted. At a limit where client writes are already refused, entries
/// that only close (admitted for nothing) retire bounded batches and
/// persist without the store passing its limit, with the default recovery
/// reserve and with none. Before, a commit's admission covered one removal
/// and was given back at its apply: held breakpoints owed nothing, and a
/// closing entry's cleanup and the Schema memtable it opened came out of
/// room nobody reserved.
#[test]
fn closing_entries_clean_up_within_the_limit_client_writes_are_refused_at() {
    closing_entries_clean_up_within_the_limit(None);
}

/// As above with no recovery reserve: the ledger's cleanup, its Schema
/// memtable included, is owed in full, not taken from a reserve.
#[test]
fn closing_entries_clean_up_within_the_limit_with_no_recovery_reserve() {
    closing_entries_clean_up_within_the_limit(Some(0));
}

fn closing_entries_clean_up_within_the_limit(reserve: Option<u64>) {
    use crate::engine::closure_ledger::RETIRE_BATCH;
    use crate::engine::footprint::{Admit, closure_debt};
    use coordinode_core::txn::proposal::{Mutation, PartitionId};

    // The smallest limit a store takes.
    const LIMIT: u64 = crate::engine::config::MIN_HARD_LIMIT_BYTES;
    let dir = tempfile::TempDir::new().expect("tempdir");
    let mut index = 0;
    {
        let (engine, oracle) = clocked_capped(dir.path(), LIMIT, reserve);
        engine.set_retention_window(std::time::Duration::from_secs(3_600));
        // More breakpoints than the cache keeps, then writes until the store
        // refuses the next one.
        for _ in 0..crate::engine::closure_ledger::CACHED_BREAKPOINTS + 44 {
            index += 1;
            assert!(admitted_write(&engine, &oracle, index, 1_024));
        }
        while admitted_write(&engine, &oracle, index + 1, 16 << 10) {
            index += 1;
        }
        engine.persist().expect("persist");
    }
    let (engine, oracle) = clocked_capped(dir.path(), LIMIT, reserve);
    // Owed from the open, before the ledger is loaded.
    let held = stored_breakpoints(&engine);
    assert!(held > crate::engine::closure_ledger::CACHED_BREAKPOINTS as u64);
    let footprint = engine.footprints().get("default").expect("a limit");
    assert!(
        footprint.debt() >= closure_debt(held),
        "an open owes {} for {held} breakpoints before the ledger loads, not their cleanup {}",
        footprint.debt(),
        closure_debt(held)
    );
    engine.set_retention_window(std::time::Duration::from_secs(3_600));
    engine.reload_closure_frontier().expect("reload");
    assert_eq!(ledger_held(&engine), held);
    // What the flush gave back is taken by writes until clients are refused.
    while admitted_write(&engine, &oracle, index + 1, 16 << 10) {
        index += 1;
    }
    let held = ledger_held(&engine);
    let limit = footprint.limit();
    let probe = [Mutation::Put {
        partition: PartitionId::Node,
        key: b"node:1:probe".to_vec(),
        value: b"v".to_vec(),
    }];
    assert!(
        engine.admit_proposal(&probe, Admit::Client).is_err(),
        "client writes are refused at the limit"
    );
    assert!(
        footprint.committed() <= footprint.limit(),
        "client writes passed the limit: {} over {} (charged, admitted, debt {:?})",
        footprint.committed(),
        footprint.limit(),
        (footprint.charged(), footprint.admitted(), footprint.debt())
    );

    engine.set_retention_window(std::time::Duration::ZERO);
    let mut index = index + 1;
    while ledger_held(&engine) > 0 {
        let before = ledger_held(&engine);
        let parts = (footprint.charged(), footprint.admitted(), footprint.debt());
        apply_closure_entry(&engine, index, &[], oracle.next().as_raw() + 1);
        let retired = before - ledger_held(&engine);
        assert!(retired > 0 && retired <= RETIRE_BATCH as u64);
        assert_eq!(stored_breakpoints(&engine), ledger_held(&engine));
        assert!(
            footprint.committed() <= footprint.limit(),
            "the store passed its limit: {} over {} (charged, admitted, debt before {parts:?}, after {:?})",
            footprint.committed(),
            footprint.limit(),
            (footprint.charged(), footprint.admitted(), footprint.debt())
        );
        index += 1;
    }
    // The cleanup's memtable is written out within the limit too.
    engine
        .persist()
        .expect("the cleanup persists within the limit");
    assert!(
        footprint.committed() <= footprint.limit(),
        "the store passed its limit once persisted: {} over {}",
        footprint.committed(),
        footprint.limit()
    );
    assert_eq!(
        footprint.limit(),
        limit,
        "the limit stayed where it was set"
    );
    // Persisted and cleaned up, the ledger owes only its next head and the
    // Schema memtable that starts.
    assert_eq!(
        footprint.debt(),
        closure_debt(0) + crate::engine::footprint::memtable_start(),
        "the debt fell with the cleanup from {}",
        closure_debt(held)
    );
}

/// Cleanup interrupted at either side of its records landing leaves a store
/// whose restart serves every position as before and completes the cleanup:
/// a plan whose records never landed retires nothing on reopen; records that
/// landed before the plan was installed are the state reopen reads; an entry
/// applied again over records the store already holds retires nothing, even
/// with the served snapshot moved on, so the ledger never counts records the
/// store still holds as removed.
#[test]
fn interrupted_cleanup_reopens_exact_and_completes() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let positions =
        u64::try_from(crate::engine::closure_ledger::CACHED_BREAKPOINTS).expect("small") + 44;
    let (first, closing) = {
        let engine = StorageEngine::open(&test_config(dir.path())).expect("open");
        let first = engine.snapshot();
        let pinned = engine
            .pin_readable_snapshot_at(first)
            .expect("the current snapshot is served");
        for index in 1..=positions {
            apply_closure_entry(&engine, index, &[first + index], 0);
        }
        // Everything below the served snapshot may go now.
        drop(pinned);
        for _ in 0..positions + 10 {
            engine
                .put(Partition::Node, b"node:1:later", b"v")
                .expect("write");
        }
        let closing = positions + 1;
        // Stopped before landing: a plan that removes breakpoints is made,
        // and its records are never written.
        let (records, _) = engine
            .closure_record(first + positions + 1, closing, false)
            .expect("plan")
            .expect("a cleanup batch");
        assert!(
            records
                .iter()
                .any(|m| matches!(m, coordinode_core::txn::proposal::Mutation::Delete { .. })),
            "the interrupted plan removes breakpoints"
        );
        engine.persist().expect("persist");
        (first, closing)
    };
    let check_exact = |engine: &StorageEngine, from: u64| {
        for index in from..=positions {
            assert_eq!(
                engine.commit_ceiling_through(index).expect("ceiling"),
                first + index,
                "position {index}"
            );
        }
    };
    {
        let engine = StorageEngine::open(&test_config(dir.path())).expect("reopen");
        engine.reload_closure_frontier().expect("reload");
        assert_eq!(ledger_held(&engine), positions, "nothing was retired");
        assert_eq!(stored_breakpoints(&engine), positions);
        check_exact(&engine, 1);
        // Stopped after landing, before the plan is installed.
        let (records, at) = engine
            .closure_record(first + positions + 1, closing, false)
            .expect("plan")
            .expect("a cleanup batch");
        engine
            .apply_raft_proposal(&records, at, closing, 0, |_| false)
            .expect("records land");
        engine.persist().expect("persist");
    }
    let engine = StorageEngine::open(&test_config(dir.path())).expect("reopen");
    engine.reload_closure_frontier().expect("reload");
    let after_batch = ledger_held(&engine);
    assert!(
        after_batch < positions,
        "the landed batch is the state read back"
    );
    assert_eq!(stored_breakpoints(&engine), after_batch);
    // The same entry applied again over records the Schema tree holds.
    if let Some((records, at)) = engine
        .closure_record(first + positions + 1, closing, true)
        .expect("plan")
    {
        engine
            .apply_raft_proposal(&records, at, closing, 0, |part| part == Partition::Schema)
            .expect("skipped");
    }
    engine
        .raise_closure_frontier(first + positions + 1, closing)
        .expect("install");
    assert_eq!(
        ledger_held(&engine),
        after_batch,
        "a replay retires nothing"
    );
    assert_eq!(stored_breakpoints(&engine), ledger_held(&engine));
    // The positions not retired keep their exact ceilings.
    check_exact(&engine, positions - after_batch + 1);
    let mut index = closing + 1;
    while ledger_held(&engine) > 0 {
        apply_closure_entry(&engine, index, &[], first + positions + 1);
        assert_eq!(stored_breakpoints(&engine), ledger_held(&engine));
        index += 1;
    }
    let one = engine.commit_ceiling_through(1).expect("ceiling");
    assert!(
        one > first,
        "position 1 never reads below its commit: {one}"
    );
}

/// Closing positions 1 and 2 (commits 10 and 20) keeps position 1 at 10
/// while a snapshot at 10 is still served: a read there after position 1
/// stays ready. The closed bound alone does not retire the distinction.
#[test]
fn a_served_snapshot_keeps_a_closed_positions_ceiling() {
    let engine = test_engine_memfs();
    let first = engine.snapshot();
    let held = engine
        .pin_readable_snapshot_at(first)
        .expect("the current snapshot is served");
    // This store's seqnos are a counter: commits one apart, as its writes.
    apply_closure_entry(&engine, 1, &[first], 0);
    apply_closure_entry(&engine, 2, &[first + 1], 0);
    apply_closure_entry(&engine, 3, &[], first + 2);
    assert_eq!(engine.commit_ceiling_through(1).expect("ceiling"), first);
    assert_eq!(
        engine.commit_ceiling_through(2).expect("ceiling"),
        first + 1
    );

    // Once nothing at or below the second commit is served, the two closed
    // prefixes are one for every read the store can answer.
    drop(held);
    for _ in 0..5 {
        engine
            .put(Partition::Node, b"node:1:later", b"v")
            .expect("write");
    }
    apply_closure_entry(&engine, 4, &[], first + 2);
    let served = engine.oldest_readable_seqno();
    assert!(served > first + 1, "the store moved on: {served}");
    let one = engine.commit_ceiling_through(1).expect("ceiling");
    assert!(one >= first && one < served, "position 1 reads {one}");
}
