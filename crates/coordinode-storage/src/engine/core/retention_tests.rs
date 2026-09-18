//! MVCC retention-window tests: the engine-owned time-travel window floor,
//! which snapshots stay readable across compaction, and what is refused.

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use std::time::Duration;
use tempfile::TempDir;

/// Oracle-backed engine (seqno == commit_ts) whose oracle starts at `base`.
fn oracle_engine(base: u64) -> (StorageEngine, Arc<TimestampOracle>, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(base)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open");
    (engine, oracle, dir)
}

/// A partition under test with keys of its own shape.
#[derive(Debug, Clone, Copy)]
struct Subject {
    part: Partition,
    pid: PartitionId,
    key: &'static [u8],
    other_key: &'static [u8],
    prefix: &'static [u8],
}

/// `Node` carries a merge operator and `EdgeProp` does not: the two run
/// different compaction configurations and owe the same retention contract.
const SUBJECTS: [Subject; 2] = [
    Subject {
        part: Partition::Node,
        pid: PartitionId::Node,
        key: b"node:00:00000001",
        other_key: b"node:00:00000002",
        prefix: b"node:",
    },
    Subject {
        part: Partition::EdgeProp,
        pid: PartitionId::EdgeProp,
        key: b"edgeprop:T:00000001:00000002",
        other_key: b"edgeprop:T:00000001:00000003",
        prefix: b"edgeprop:",
    },
];

fn put_in(engine: &StorageEngine, pid: PartitionId, key: &[u8], value: &[u8], commit_ts: u64) {
    engine
        .apply_proposal_at(
            &[Mutation::Put {
                partition: pid,
                key: key.to_vec(),
                value: value.to_vec(),
            }],
            commit_ts,
        )
        .expect("apply");
}

fn delete_in(engine: &StorageEngine, pid: PartitionId, key: &[u8], commit_ts: u64) {
    engine
        .apply_proposal_at(
            &[Mutation::Delete {
                partition: pid,
                key: key.to_vec(),
            }],
            commit_ts,
        )
        .expect("apply");
}

fn read_in(
    engine: &StorageEngine,
    part: Partition,
    key: &[u8],
    snapshot: u64,
) -> StorageResult<Option<Vec<u8>>> {
    engine
        .snapshot_get(&snapshot, part, key)
        .map(|v| v.map(|b| b.to_vec()))
}

/// Seal the current memtable into its own table so the next compaction has
/// several tables to merge (a lone table is never rewritten).
fn flush_in(engine: &StorageEngine, part: Partition) {
    engine
        .tree(part)
        .expect("tree")
        .flush_active_memtable(0)
        .expect("flush");
}

fn put_at(engine: &StorageEngine, key: &[u8], value: &[u8], commit_ts: u64) {
    put_in(engine, PartitionId::Node, key, value, commit_ts);
}

fn read_at(engine: &StorageEngine, key: &[u8], snapshot: u64) -> StorageResult<Option<Vec<u8>>> {
    read_in(engine, Partition::Node, key, snapshot)
}

fn flush(engine: &StorageEngine) {
    flush_in(engine, Partition::Node);
}

/// Wall-clock microseconds now plus a wide margin: opening an engine
/// allocates timestamps, which re-anchors the oracle to the wall clock, so a
/// test that drives the clock itself must start in the future.
fn future_base() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros();
    u64::try_from(now).expect("fits") + 1_000_000_000_000
}

/// Inside the window, time travel is exact across compaction: a key written
/// before a point and rewritten after it still reads its older version at
/// that point, a key never rewritten keeps its value, and the live version
/// is unaffected. Every snapshot here is at or above the watermark, so the
/// tree serves each one from the version that was current at it.
#[test]
fn time_travel_inside_the_window_survives_compaction() {
    for s in SUBJECTS {
        let base = future_base();
        let (engine, _oracle, _dir) = oracle_engine(base);
        engine.set_retention_window(Duration::from_secs(3_600));

        put_in(&engine, s.pid, s.key, b"v1", base + 1_000);
        flush_in(&engine, s.part);
        put_in(&engine, s.pid, s.key, b"v2", base + 3_000);
        flush_in(&engine, s.part);
        put_in(&engine, s.pid, s.other_key, b"lonely", base + 1_500);

        engine.force_compaction(s.part).expect("compact");

        assert_eq!(
            read_in(&engine, s.part, s.key, base + 4_000).expect("read"),
            Some(b"v2".to_vec()),
            "{s:?}"
        );
        assert_eq!(
            read_in(&engine, s.part, s.key, base + 2_500).expect("read"),
            Some(b"v1".to_vec()),
            "{s:?}: the version current between the two writes is readable after compaction"
        );
        assert_eq!(
            read_in(&engine, s.part, s.other_key, base + 4_000).expect("read"),
            Some(b"lonely".to_vec()),
            "{s:?}"
        );
        assert_eq!(
            read_in(&engine, s.part, s.other_key, base + 1_200).expect("read"),
            None,
            "{s:?}: a snapshot before the write does not see it"
        );
    }
}

/// Inside the window the superseded key version survives in the compaction
/// output, so the old snapshot is served from the latest tables while the
/// consumed inputs are already gone. Once the clock moves the watermark past
/// it, a compaction folds the version away and a read there is refused
/// rather than answered from what survived.
#[test]
fn history_below_the_watermark_is_released_and_refused() {
    for s in SUBJECTS {
        history_below_the_watermark(s);
    }
}

fn history_below_the_watermark(s: Subject) {
    let base = future_base();
    let (engine, oracle, _dir) = oracle_engine(base);
    engine.set_retention_window(Duration::from_secs(1));

    put_in(&engine, s.pid, s.key, b"v1", base + 1_000);
    // A key written once and never again: the fold below must leave it alone
    // while it collects the rewritten key's history beside it.
    put_in(&engine, s.pid, s.other_key, b"cold", base + 1_500);
    flush_in(&engine, s.part);
    put_in(&engine, s.pid, s.key, b"v2", base + 3_000);
    flush_in(&engine, s.part);
    engine.force_compaction(s.part).expect("compact");
    // The window is paid for in key versions, not in tables: the inputs the
    // compaction consumed are released at its install, and the older version
    // is still readable because the output kept it.
    assert_eq!(
        engine
            .retained_history(s.part)
            .expect("stats")
            .retained_bytes,
        0,
        "{s:?}: no table is held only for history"
    );
    assert_eq!(
        read_in(&engine, s.part, s.key, base + 2_500).expect("read"),
        Some(b"v1".to_vec()),
        "{s:?}"
    );

    // Ten seconds pass; the next compaction folds what sits below the new
    // watermark.
    oracle.advance_to(Timestamp::from_raw(base + 10_000_000));
    engine.advance_gc_watermark();
    let watermark = engine.gc_watermark();
    assert_eq!(watermark, base + 10_000_001 - 1_000_000);
    engine.force_compaction(s.part).expect("compact");

    // Below the watermark: refused, never a panic or a wrong answer.
    match read_in(&engine, s.part, s.key, base + 2_500) {
        Err(StorageError::SnapshotOutsideRetention {
            snapshot,
            watermark: w,
        }) => {
            assert_eq!(snapshot, base + 2_500);
            assert_eq!(w, watermark);
        }
        other => panic!("{s:?}: expected SnapshotOutsideRetention, got {other:?}"),
    }
    assert!(
        engine.pin_snapshot_at(base + 2_500).is_none(),
        "{s:?}: a pin cannot protect history that is already collected"
    );
    let err = engine
        .snapshot_prefix_scan(&(base + 2_500), s.part, s.prefix)
        .expect_err("scan below the watermark is refused");
    assert!(matches!(err, StorageError::SnapshotOutsideRetention { .. }));
    let err = engine
        .prefix_scan_at(s.part, s.prefix, base + 2_500)
        .err()
        .expect("prefix_scan_at below the watermark is refused");
    assert!(matches!(err, StorageError::SnapshotOutsideRetention { .. }));

    // At the watermark and above: served, and exact. Both keys have every
    // version below the watermark, and each keeps its own newest one.
    assert_eq!(
        read_in(&engine, s.part, s.key, watermark).expect("read"),
        Some(b"v2".to_vec()),
        "{s:?}"
    );
    assert_eq!(
        read_in(&engine, s.part, s.other_key, watermark).expect("read"),
        Some(b"cold".to_vec()),
        "{s:?}: a key's only version is its newest and survives the fold"
    );
    let pin = engine
        .pin_snapshot_at(watermark)
        .expect("a pin at the watermark is granted");
    assert_eq!(pin.seqno(), watermark);
    drop(pin);

    // A write after the fold: the folded key now straddles the watermark
    // again, and the next compaction keeps both sides of it.
    put_in(&engine, s.pid, s.key, b"v3", base + 10_000_500);
    flush_in(&engine, s.part);
    engine.force_compaction(s.part).expect("compact");
    assert_eq!(
        read_in(&engine, s.part, s.key, base + 10_000_400).expect("read"),
        Some(b"v2".to_vec()),
        "{s:?}: the newest version below the watermark serves a snapshot above it"
    );
    assert_eq!(
        read_in(&engine, s.part, s.key, base + 20_000_000).expect("read"),
        Some(b"v3".to_vec()),
        "{s:?}"
    );
}

/// A delete is a version like any other. Inside the window a snapshot before
/// it still reads the value and one after it reads nothing; once the
/// tombstone is the newest version below the watermark, the fold must not
/// bring the value under it back, in this process or after a reopen.
#[test]
fn a_delete_below_the_watermark_does_not_resurrect_the_value() {
    for s in SUBJECTS {
        delete_does_not_resurrect(s);
    }
}

fn delete_does_not_resurrect(s: Subject) {
    let base = future_base();
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);

    {
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(base)));
        let engine = StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open");
        engine.set_retention_window(Duration::from_secs(1));

        put_in(&engine, s.pid, s.key, b"v1", base + 1_000);
        put_in(&engine, s.pid, s.other_key, b"kept", base + 1_200);
        flush_in(&engine, s.part);
        delete_in(&engine, s.pid, s.key, base + 2_000);
        flush_in(&engine, s.part);
        engine.force_compaction(s.part).expect("compact");

        assert_eq!(
            read_in(&engine, s.part, s.key, base + 1_500).expect("read"),
            Some(b"v1".to_vec()),
            "{s:?}: a snapshot before the delete still reads the value"
        );
        assert_eq!(
            read_in(&engine, s.part, s.key, base + 2_500).expect("read"),
            None,
            "{s:?}: a snapshot after the delete reads nothing"
        );

        oracle.advance_to(Timestamp::from_raw(base + 10_000_000));
        engine.advance_gc_watermark();
        let watermark = engine.gc_watermark();
        engine.force_compaction(s.part).expect("compact");

        assert_eq!(
            read_in(&engine, s.part, s.key, watermark).expect("read"),
            None,
            "{s:?}: the tombstone is the newest version below the watermark"
        );
        assert_eq!(engine.get(s.part, s.key).expect("get"), None, "{s:?}");
        assert!(
            matches!(
                read_in(&engine, s.part, s.key, base + 1_500),
                Err(StorageError::SnapshotOutsideRetention { .. })
            ),
            "{s:?}: the value under the tombstone is refused, not served"
        );
        assert_eq!(
            read_in(&engine, s.part, s.other_key, watermark).expect("read"),
            Some(b"kept".to_vec()),
            "{s:?}: the neighbouring key is untouched by the delete"
        );
        engine.persist().expect("persist");
    }

    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(
        base + 10_000_500,
    )));
    let engine = StorageEngine::open_with_oracle(&config, oracle).expect("reopen");
    engine.set_retention_window(Duration::from_secs(1));
    assert_eq!(
        engine.get(s.part, s.key).expect("get"),
        None,
        "{s:?}: still deleted after a reopen"
    );
    assert_eq!(
        engine
            .get(s.part, s.other_key)
            .expect("get")
            .map(|b| b.to_vec()),
        Some(b"kept".to_vec()),
        "{s:?}"
    );
}

/// The window's disk cost is observable per partition, and it is paid in key
/// versions, not in tables: inside the window the compaction output carries
/// both versions of the key (so it is larger than one version) and nothing
/// is held beside it; once the watermark passes, the next compaction folds
/// the superseded version and the live figure shrinks. A partition never
/// written reports zero on both sides.
#[test]
fn retained_history_follows_the_window() {
    let base = future_base();
    let (engine, oracle, _dir) = oracle_engine(base);
    engine.set_retention_window(Duration::from_secs(1));
    let key = b"node:00:00000001";

    let untouched = engine.retained_history(Partition::Adj).expect("stats");
    assert_eq!(untouched.live_bytes, 0);
    assert_eq!(untouched.retained_bytes, 0);
    assert_eq!(untouched.retained_ratio(), 0.0);

    put_at(&engine, key, &[b'a'; 4096], base + 1_000);
    flush(&engine);
    put_at(&engine, key, &[b'b'; 4096], base + 3_000);
    flush(&engine);
    engine.force_compaction(Partition::Node).expect("compact");

    let inside = engine.retained_history(Partition::Node).expect("stats");
    assert!(inside.live_bytes > 0, "the compacted output is live");
    assert_eq!(
        inside.retained_bytes, 0,
        "the consumed inputs are released at install, not held for the window"
    );
    assert_eq!(inside.retained_ratio(), 0.0);
    assert_eq!(
        read_at(&engine, key, base + 2_500).expect("read"),
        Some(vec![b'a'; 4096]),
        "the superseded version is served from the latest tables"
    );
    // Live agrees with the tree's own live accounting.
    let tree = engine.tree(Partition::Node).expect("tree");
    assert_eq!(
        inside.live_bytes,
        lsm_tree::AbstractTree::storage_stats(tree)
            .expect("storage_stats")
            .used_bytes
    );
    // The whole-engine view carries the same figure for this partition.
    let all = engine.retained_history_all().expect("all");
    let (_, from_all) = all
        .iter()
        .find(|(p, _)| *p == Partition::Node)
        .expect("Node present");
    assert_eq!(from_all.live_bytes, inside.live_bytes);

    oracle.advance_to(Timestamp::from_raw(base + 10_000_000));
    engine.advance_gc_watermark();
    engine.force_compaction(Partition::Node).expect("compact");

    let released = engine.retained_history(Partition::Node).expect("stats");
    assert!(released.live_bytes > 0);
    assert_eq!(released.retained_bytes, 0);
    // The window's cost was the superseded version inside the output; folding
    // it is what shrinks the partition (by less than its 4 KiB, since a run
    // of one byte compresses to almost nothing).
    assert!(
        released.live_bytes < inside.live_bytes,
        "the superseded version left the live tables: {} -> {}",
        inside.live_bytes,
        released.live_bytes
    );
}

/// The read boundary a process sees must be the one it sees after a restart:
/// compact inside the window, reopen, and every snapshot the window still
/// covers is served with the value it had, while a snapshot below the
/// horizon is refused on both sides of the reopen. A compaction that dropped
/// the newest version below its threshold used to turn the first half into
/// "absent" after a reopen, and a filtering install used to turn it into a
/// refusal.
#[test]
fn a_reopen_serves_the_same_window_the_process_served() {
    for s in SUBJECTS {
        reopen_serves_the_window(s);
    }
}

fn reopen_serves_the_window(s: Subject) {
    let part = s.part;
    let put = |engine: &StorageEngine, value: &[u8], commit_ts: u64| {
        put_in(engine, s.pid, s.key, value, commit_ts);
    };
    let read = |engine: &StorageEngine, snapshot: u64| read_in(engine, s.part, s.key, snapshot);
    let seal = |engine: &StorageEngine| flush_in(engine, s.part);
    let base = future_base();
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    // A snapshot inside the window, after the two old versions and before the
    // newest one: it resolves to `v1`, the newest version below the watermark.
    let inside = base + 500_000;

    let watermark = {
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(base)));
        let engine = StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open");
        engine.set_retention_window(Duration::from_secs(1));

        // Versions straddling the watermark: v0 and v1 end up below it, v2
        // above. The fold has to collect v0, keep v1 (a snapshot at or above
        // the watermark can still resolve to it) and leave v2 alone.
        put(&engine, b"v0", base + 1_000);
        seal(&engine);
        put(&engine, b"v1", base + 2_000);
        seal(&engine);
        put(&engine, b"v2", base + 1_003_000);
        seal(&engine);
        let watermark = engine.gc_watermark();
        assert_eq!(watermark, base + 1_003_001 - 1_000_000);
        assert!(watermark <= inside && inside < base + 1_003_000);

        engine.force_compaction(part).expect("compact");
        assert_eq!(
            read(&engine, inside).expect("read before the reopen"),
            Some(b"v1".to_vec()),
            "{part:?}"
        );
        engine.persist().expect("persist");
        watermark
    };

    // A fresh process: nothing in memory, only what the manifest recorded.
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(
        base + 1_003_500,
    )));
    let engine = StorageEngine::open_with_oracle(&config, oracle).expect("reopen");
    engine.set_retention_window(Duration::from_secs(1));

    assert!(
        engine.oldest_readable_seqno() <= inside,
        "{part:?}: the recorded floor ({}) must still admit a snapshot the window covers \
         ({inside}); the compaction ran at watermark {watermark}",
        engine.oldest_readable_seqno()
    );
    assert_eq!(
        read(&engine, inside).expect("read inside the window after the reopen"),
        Some(b"v1".to_vec()),
        "{part:?}: the newest version below the watermark survives compaction and reopen"
    );
    assert_eq!(
        read(&engine, base + 1_003_200).expect("read"),
        Some(b"v2".to_vec())
    );
    // Below the horizon the collected version is refused, not answered with
    // whatever survived.
    match read(&engine, base + 1_500) {
        Err(StorageError::SnapshotOutsideRetention { snapshot, .. }) => {
            assert_eq!(snapshot, base + 1_500);
        }
        other => panic!("expected SnapshotOutsideRetention below the horizon, got {other:?}"),
    }
}

/// Sum of the regular files directly under `dir`, `0` when the directory
/// does not exist: the test-side mirror of what `retained_history` scans,
/// computed independently through `std::fs`.
fn files_in(dir: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// Physical bytes of a partition across every endpoint root it may have
/// been routed to: `<root>/<partition>/tables` on each plus `blobs` on the
/// primary.
fn partition_bytes(roots: &[&std::path::Path], part: Partition) -> u64 {
    let mut total = 0;
    for root in roots {
        total += files_in(&root.join(part.name()).join("tables"));
    }
    total + files_in(&roots[0].join(part.name()).join("blobs"))
}

/// With per-level routing the partition's tables live under several
/// endpoint roots; the scan must cover every routed folder, or a bottom
/// level on the cold endpoint would be invisible and history there would
/// pass as free. Checked against an independent `std::fs` walk of all
/// three roots: live plus retained is exactly what is on disk.
#[test]
fn retained_history_spans_level_routed_endpoints() {
    let hot = TempDir::new().expect("hot");
    let warm = TempDir::new().expect("warm");
    let cold = TempDir::new().expect("cold");
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "ep-warm",
            warm.path(),
            Media::Ssd,
            Durability::Durable,
            Tier::Warm,
        ),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");
    for i in 0..2000u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine
        .major_compact(Partition::Node)
        .expect("major compact");

    let cold_tables = cold.path().join(Partition::Node.name()).join("tables");
    assert!(
        files_in(&cold_tables) > 0,
        "precondition: the bottom level landed on the cold endpoint"
    );

    let history = engine.retained_history(Partition::Node).expect("stats");
    let tree = engine.tree(Partition::Node).expect("tree");
    assert_eq!(
        history.live_bytes,
        lsm_tree::AbstractTree::storage_stats(tree)
            .expect("storage_stats")
            .used_bytes
    );
    assert_eq!(
        history.live_bytes + history.retained_bytes,
        partition_bytes(&[hot.path(), warm.path(), cold.path()], Partition::Node),
        "the scan covers the primary and every routed tables folder"
    );
    // Only the hot root would miss the cold level entirely.
    assert!(
        partition_bytes(&[hot.path()], Partition::Node) < history.live_bytes,
        "the primary folder alone does not hold the live version"
    );
}

/// A KV-separated partition keeps its values in blob files next to the
/// tables; the scan counts both, so live plus retained is what the two
/// folders hold, and the blob folder is where most of the bytes are.
#[test]
fn retained_history_of_a_blob_partition_counts_blob_files() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    // Values above the KV-separation threshold go to blob files.
    let value = vec![0x5Au8; 16 * 1024];
    for i in 0..64u32 {
        let key = format!("blob:{i:08}");
        engine
            .put(Partition::Blob, key.as_bytes(), &value)
            .expect("put");
    }
    engine.persist().expect("persist");

    let blobs = dir.path().join(Partition::Blob.name()).join("blobs");
    assert!(
        files_in(&blobs) > 0,
        "precondition: blob files were written"
    );

    let history = engine.retained_history(Partition::Blob).expect("stats");
    let tree = engine.tree(Partition::Blob).expect("tree");
    assert_eq!(
        history.live_bytes,
        lsm_tree::AbstractTree::storage_stats(tree)
            .expect("storage_stats")
            .used_bytes
    );
    assert_eq!(
        history.live_bytes + history.retained_bytes,
        partition_bytes(&[dir.path()], Partition::Blob)
    );
    assert!(
        history.live_bytes > files_in(&dir.path().join(Partition::Blob.name()).join("tables")),
        "the live figure includes the blob files, not the index tables alone"
    );
}

/// An in-memory engine has no folders until something is flushed: the scan
/// reports zero rather than failing on the missing directories, and after a
/// flush the `MemFs` listing is read like a real one.
#[test]
fn retained_history_on_a_memory_engine_reads_memfs_folders() {
    let config = StorageConfig::with_endpoints_no_persistence(vec![EndpointConfig::new(
        "memfs",
        "/memfs/retention",
        Media::Ram,
        Durability::Volatile,
        Tier::Memory,
    )])
    .with_fs(Arc::new(lsm_tree::fs::MemFs::new()));
    let engine = StorageEngine::open(&config).expect("open");

    let empty = engine.retained_history(Partition::Node).expect("stats");
    assert_eq!(
        empty,
        crate::engine::retention_stats::RetainedHistory::default()
    );

    engine
        .put(Partition::Node, b"node:00:00000001", b"v")
        .expect("put");
    flush(&engine);
    let flushed = engine.retained_history(Partition::Node).expect("stats");
    assert!(
        flushed.live_bytes > 0,
        "the flushed table is visible through MemFs"
    );
    assert_eq!(flushed.retained_bytes, 0);
}

/// A tables folder the process cannot list is an error, not a zero: a zero
/// would read as "no history" on an endpoint whose history is simply
/// unreadable, and the capacity refresh would publish it as such.
#[cfg(unix)]
#[test]
fn retained_history_reports_an_unreadable_tables_folder() {
    use std::os::unix::fs::PermissionsExt;

    /// Restores the folder's permissions on drop, so a failed assertion
    /// does not leave an unreadable directory behind in the tempdir.
    struct Restore(std::path::PathBuf);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
        }
    }

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    engine
        .put(Partition::Node, b"node:00:00000001", b"v")
        .expect("put");
    flush(&engine);

    let tables = dir.path().join(Partition::Node.name()).join("tables");
    let _restore = Restore(tables.clone());
    std::fs::set_permissions(&tables, std::fs::Permissions::from_mode(0o000)).expect("chmod");

    let err = engine
        .retained_history(Partition::Node)
        .expect_err("an unlistable tables folder must surface as an error");
    assert!(
        matches!(err, StorageError::Engine(_) | StorageError::Io(_)),
        "unexpected error kind: {err:?}"
    );
}

/// The capacity refresh publishes the two gauges per partition with the
/// values `retained_history` reports, under the `partition` label.
#[test]
fn capacity_refresh_publishes_retained_history_gauges() {
    use metrics::{Gauge, GaugeFn, Key, KeyName, Metadata, Recorder, SharedString, Unit};
    use std::sync::Mutex;

    /// One observed gauge `set`: metric name, labels, value.
    type Observed = (String, Vec<(String, String)>, f64);
    /// Records every gauge `set` with its key and labels.
    struct Capture(Arc<Mutex<Vec<Observed>>>);
    struct Handle {
        key: Key,
        sink: Arc<Mutex<Vec<Observed>>>,
    }
    impl GaugeFn for Handle {
        fn increment(&self, _: f64) {}
        fn decrement(&self, _: f64) {}
        fn set(&self, value: f64) {
            let labels = self
                .key
                .labels()
                .map(|l| (l.key().to_owned(), l.value().to_owned()))
                .collect();
            self.sink
                .lock()
                .expect("sink")
                .push((self.key.name().to_owned(), labels, value));
        }
    }
    impl Recorder for Capture {
        fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn register_counter(&self, _: &Key, _: &Metadata<'_>) -> metrics::Counter {
            metrics::Counter::noop()
        }
        fn register_gauge(&self, key: &Key, _: &Metadata<'_>) -> Gauge {
            Gauge::from_arc(Arc::new(Handle {
                key: key.clone(),
                sink: Arc::clone(&self.0),
            }))
        }
        fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> metrics::Histogram {
            metrics::Histogram::noop()
        }
    }

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    engine
        .put(Partition::Node, b"node:00:00000001", b"v")
        .expect("put");
    flush(&engine);
    let expected = engine.retained_history(Partition::Node).expect("stats");

    let sink = Arc::new(Mutex::new(Vec::new()));
    let recorder = Capture(Arc::clone(&sink));
    metrics::with_local_recorder(&recorder, || engine.refresh_capacity());

    let seen = sink.lock().expect("sink");
    let find = |name: &str| {
        seen.iter()
            .find(|(n, labels, _)| {
                n == name
                    && labels
                        .iter()
                        .any(|(k, v)| k == "partition" && v == Partition::Node.name())
            })
            .map(|(_, _, value)| *value)
    };
    assert_eq!(
        find("coordinode_storage_live_bytes"),
        Some(expected.live_bytes as f64)
    );
    assert_eq!(
        find("coordinode_storage_retained_history_bytes"),
        Some(expected.retained_bytes as f64)
    );
    assert!(
        seen.iter()
            .filter(|(n, _, _)| n == "coordinode_storage_live_bytes")
            .count()
            >= Partition::all().len() - 1,
        "every user-data partition is published"
    );
}

/// The configured window becomes a GC floor on an oracle-backed engine:
/// `watermark == now_seqno - window`, refreshed as the clock advances and
/// on every runtime change of the window.
#[test]
fn retention_window_holds_the_watermark_back_on_oracle_engines() {
    let base = future_base();
    let (engine, oracle, _dir) = oracle_engine(base);
    engine.set_retention_window(Duration::from_secs(1));
    let window_us = 1_000_000;
    assert_eq!(engine.retention_window(), Duration::from_secs(1));

    engine.advance_gc_watermark();
    let snap = engine.snapshot();
    assert!(snap > base, "oracle sits at or past the future base");
    assert_eq!(engine.gc_watermark(), snap - window_us);

    // Time passes: the floor follows the oracle.
    oracle.advance_to(Timestamp::from_raw(base + 5_000_000));
    engine.advance_gc_watermark();
    let snap = engine.snapshot();
    assert_eq!(snap, base + 5_000_001);
    assert_eq!(engine.gc_watermark(), snap - window_us);

    // Widening the window at runtime moves the floor back immediately.
    engine.set_retention_window(Duration::from_secs(3));
    assert_eq!(engine.gc_watermark(), snap - 3 * window_us);

    // A window reaching past the epoch saturates at "keep everything".
    engine.set_retention_window(Duration::from_secs(100 * 365 * 24 * 3600));
    assert_eq!(engine.gc_watermark(), 0);
}

/// The commit path ticks the window: after a proposal the watermark already
/// reflects the oracle position that commit advanced to, no background
/// timer needed.
#[test]
fn commit_path_refreshes_the_window_floor() {
    let base = future_base();
    let (engine, _oracle, _dir) = oracle_engine(base);
    engine.set_retention_window(Duration::from_secs(1));
    engine.advance_gc_watermark();
    let before = engine.gc_watermark();

    put_at(&engine, b"node:00:00000001", b"v", base + 5_000_000);
    assert_eq!(
        engine.gc_watermark(),
        base + 5_000_001 - 1_000_000,
        "the proposal's commit_ts moved the oracle and, with it, the floor"
    );
    assert!(engine.gc_watermark() > before);
}

/// A live snapshot pin below the window floor still wins: the watermark is
/// the minimum of every input, a pinned reader is never collected under.
#[test]
fn snapshot_pin_below_the_window_floor_wins() {
    let base = future_base();
    let (engine, _oracle, _dir) = oracle_engine(base);
    engine.set_retention_window(Duration::from_secs(1));

    // Pin inside the window (half a second back): the window floor is still
    // the lower of the two.
    let target = engine.snapshot() - 500_000;
    let pin = engine.pin_snapshot_at(target).expect("inside the window");
    assert_eq!(engine.gc_watermark(), engine.snapshot() - 1_000_000);
    // Narrowing the window to 1 ms would lift the floor over the pin; the
    // pin holds it.
    engine.set_retention_window(Duration::from_millis(1));
    assert_eq!(engine.gc_watermark(), target);
    drop(pin);
    assert_eq!(engine.gc_watermark(), engine.snapshot() - 1_000);
}

/// The consumer floor (registry) and the window floor combine by `min`: a
/// consumer lagging beyond the window extends retention for itself, a
/// consumer ahead of the window does not shrink it.
#[test]
fn consumer_floor_below_the_window_floor_wins() {
    let base = future_base();
    let (engine, _oracle, _dir) = oracle_engine(base);
    engine.set_retention_window(Duration::from_secs(1));

    engine.set_consumer_retention_floor(base - 2_000_000);
    assert_eq!(engine.gc_watermark(), base - 2_000_000);
    engine.set_consumer_retention_floor(engine.snapshot());
    assert_eq!(engine.gc_watermark(), engine.snapshot() - 1_000_000);
    engine.set_consumer_retention_floor(u64::MAX);
    assert_eq!(engine.gc_watermark(), engine.snapshot() - 1_000_000);
}

/// Without an oracle the seqno is a plain counter, not a clock, so the
/// window is meaningless and imposes no floor: the watermark tracks the
/// current seqno as before (compaction folds everything).
#[test]
fn retention_window_is_inert_without_an_oracle() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    for i in 0..10u64 {
        engine
            .put(Partition::Node, format!("node:00:{i:08}").as_bytes(), b"v")
            .expect("put");
    }
    engine.set_retention_window(Duration::from_secs(7 * 24 * 3600));
    engine.advance_gc_watermark();
    assert_eq!(engine.gc_watermark(), engine.snapshot());
    assert_eq!(
        engine.retention_window(),
        Duration::from_secs(7 * 24 * 3600)
    );
}

/// The window is read from `StorageConfig` at open and applies immediately.
#[test]
fn retention_window_comes_from_storage_config() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(
        future_base(),
    )));
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.retention_window_secs = 2;
    let engine = StorageEngine::open_with_oracle(&config, oracle).expect("open");
    assert_eq!(engine.retention_window(), Duration::from_secs(2));
    assert_eq!(engine.gc_watermark(), engine.snapshot() - 2_000_000);
}

/// The default window is seven days, matching the documented default.
#[test]
fn default_retention_window_is_seven_days() {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        "/nonexistent",
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    assert_eq!(config.retention_window_secs, 7 * 24 * 3600);
}

/// The engine's own guard is a policy boundary; the tree keeps a physical one
/// and prunes history for reasons the window knows nothing about. `drop_range`
/// is one: it raises the tree's retention floor to its own install seqno, so a
/// snapshot the window still permits can become unservable.
///
/// The read must then be refused with the same typed error the policy guard
/// raises — before the fix in coordinode-lsm-tree 5.8.6 the tree panicked, and
/// without the `From` mapping it would surface as an opaque engine error that
/// gRPC reports as INTERNAL instead of OUT_OF_RANGE.
#[test]
fn read_below_the_physical_floor_is_refused_not_panicked() {
    let base = future_base();
    let (engine, _oracle, _dir) = oracle_engine(base);
    // A window wide enough that policy alone would permit every read below.
    engine.set_retention_window(Duration::from_secs(7 * 24 * 3600));

    let key = b"node:00:00000001";
    put_at(&engine, key, b"v1", base + 1_000);
    flush(&engine);
    put_at(&engine, key, b"v2", base + 3_000);
    flush(&engine);

    let old_snapshot = base + 1_500;
    assert_eq!(
        read_at(&engine, key, old_snapshot).expect("readable before the reset"),
        Some(b"v1".to_vec())
    );
    assert!(
        engine.oldest_readable_seqno() <= old_snapshot,
        "policy horizon must still permit the snapshot before the reset"
    );

    // `clear_partition` installs a fresh empty version and drops the history
    // behind it — the repair path does this before reinstalling a known-good
    // base. The window knows nothing about it.
    engine
        .clear_partition(Partition::Node)
        .expect("clear partition");

    let horizon = engine.oldest_readable_seqno();
    assert!(
        horizon > old_snapshot,
        "clearing a partition must raise the horizon past the old snapshot \
         (horizon {horizon}, snapshot {old_snapshot})"
    );

    match read_at(&engine, key, old_snapshot) {
        Err(StorageError::SnapshotOutsideRetention { snapshot, .. }) => {
            assert_eq!(snapshot, old_snapshot);
        }
        other => panic!("expected SnapshotOutsideRetention, got {other:?}"),
    }
    // At and above the horizon the read is served — the partition is empty now,
    // so the answer is "no such key", which is a result and not an error.
    assert_eq!(read_at(&engine, key, horizon).expect("served"), None);
}
