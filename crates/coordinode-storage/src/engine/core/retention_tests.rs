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

fn put_at(engine: &StorageEngine, key: &[u8], value: &[u8], commit_ts: u64) {
    engine
        .apply_proposal_at(
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: key.to_vec(),
                value: value.to_vec(),
            }],
            commit_ts,
        )
        .expect("apply");
}

fn read_at(engine: &StorageEngine, key: &[u8], snapshot: u64) -> StorageResult<Option<Vec<u8>>> {
    engine
        .snapshot_get(&snapshot, Partition::Node, key)
        .map(|v| v.map(|b| b.to_vec()))
}

/// Seal the current memtable into its own table so the next compaction has
/// several tables to merge (a lone table is never rewritten).
fn flush(engine: &StorageEngine) {
    engine
        .tree(Partition::Node)
        .expect("tree")
        .flush_active_memtable(0)
        .expect("flush");
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

fn count_files(dir: &std::path::Path) -> usize {
    let mut n = 0;
    for entry in std::fs::read_dir(dir).expect("read_dir") {
        let entry = entry.expect("entry");
        if entry.path().is_dir() {
            n += count_files(&entry.path());
        } else {
            n += 1;
        }
    }
    n
}

/// Inside the window, time travel is exact across compaction: a key written
/// before a point and rewritten after it still reads its older version at
/// that point, a key never rewritten keeps its value, and the live version
/// is unaffected. Every snapshot here is at or above the watermark, so the
/// tree serves each one from the version that was current at it.
#[test]
fn time_travel_inside_the_window_survives_compaction() {
    let base = future_base();
    let (engine, _oracle, _dir) = oracle_engine(base);
    engine.set_retention_window(Duration::from_secs(3_600));
    let key = b"node:00:00000001";

    put_at(&engine, key, b"v1", base + 1_000);
    flush(&engine);
    put_at(&engine, key, b"v2", base + 3_000);
    flush(&engine);
    put_at(&engine, b"node:00:00000002", b"lonely", base + 1_500);

    engine.force_compaction(Partition::Node).expect("compact");

    assert_eq!(
        read_at(&engine, key, base + 4_000).expect("read"),
        Some(b"v2".to_vec())
    );
    assert_eq!(
        read_at(&engine, key, base + 2_500).expect("read"),
        Some(b"v1".to_vec()),
        "the version current between the two writes is readable after compaction"
    );
    assert_eq!(
        read_at(&engine, b"node:00:00000002", base + 4_000).expect("read"),
        Some(b"lonely".to_vec())
    );
    assert_eq!(
        read_at(&engine, b"node:00:00000002", base + 1_200).expect("read"),
        None,
        "a snapshot before the write does not see it"
    );
}

/// Once the clock moves the watermark past a compaction, the history below
/// it is released (the retained tree versions and their tables go) and a
/// read there is refused rather than answered from what survived.
#[test]
fn history_below_the_watermark_is_released_and_refused() {
    let base = future_base();
    let (engine, oracle, dir) = oracle_engine(base);
    engine.set_retention_window(Duration::from_secs(1));
    let key = b"node:00:00000001";

    put_at(&engine, key, b"v1", base + 1_000);
    flush(&engine);
    put_at(&engine, key, b"v2", base + 3_000);
    flush(&engine);
    engine.force_compaction(Partition::Node).expect("compact");
    // The compaction inputs stay on disk: the watermark (1 s behind the
    // clock) is still below the versions that reference them.
    let retained = count_files(dir.path());
    assert_eq!(
        read_at(&engine, key, base + 2_500).expect("read"),
        Some(b"v1".to_vec())
    );

    // Ten seconds pass; the next compaction lets the tree prune the versions
    // below the new watermark and delete the tables only they referenced.
    oracle.advance_to(Timestamp::from_raw(base + 10_000_000));
    engine.advance_gc_watermark();
    let watermark = engine.gc_watermark();
    assert_eq!(watermark, base + 10_000_001 - 1_000_000);
    engine.force_compaction(Partition::Node).expect("compact");
    assert!(
        count_files(dir.path()) < retained,
        "tables referenced only by pruned versions are released"
    );

    // Below the watermark: refused, never a panic or a wrong answer.
    match read_at(&engine, key, base + 2_500) {
        Err(StorageError::SnapshotOutsideRetention {
            snapshot,
            watermark: w,
        }) => {
            assert_eq!(snapshot, base + 2_500);
            assert_eq!(w, watermark);
        }
        other => panic!("expected SnapshotOutsideRetention, got {other:?}"),
    }
    assert!(
        engine.pin_snapshot_at(base + 2_500).is_none(),
        "a pin cannot protect history that is already collected"
    );
    let err = engine
        .snapshot_prefix_scan(&(base + 2_500), Partition::Node, b"node:")
        .expect_err("scan below the watermark is refused");
    assert!(matches!(err, StorageError::SnapshotOutsideRetention { .. }));
    let err = engine
        .prefix_scan_at(Partition::Node, b"node:", base + 2_500)
        .err()
        .expect("prefix_scan_at below the watermark is refused");
    assert!(matches!(err, StorageError::SnapshotOutsideRetention { .. }));

    // At the watermark and above: served, and exact.
    assert_eq!(
        read_at(&engine, key, watermark).expect("read"),
        Some(b"v2".to_vec())
    );
    let pin = engine
        .pin_snapshot_at(watermark)
        .expect("a pin at the watermark is granted");
    assert_eq!(pin.seqno(), watermark);
    assert_eq!(
        read_at(&engine, key, base + 20_000_000).expect("read"),
        Some(b"v2".to_vec())
    );
}

/// The window's disk cost is observable per partition: inside the window a
/// compaction's consumed inputs stay on disk as retained history next to
/// the live output; once the watermark passes the compaction and the next
/// install prunes the versions, the consumed inputs are released and what
/// remains retained is exactly the newest version below the watermark (the
/// previous live version), kept so a read at the watermark has a version
/// to be served from. A partition never written reports zero on both sides.
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
    assert!(
        inside.retained_bytes > 0,
        "the two consumed input tables are retained inside the window"
    );
    assert!(inside.retained_ratio() > 0.0);
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
    // The version history keeps the newest version below the watermark so a
    // read at exactly the watermark is served from the version current at
    // it: what remains retained is that one version's tables, which is the
    // previous live version (the first compaction's output) and nothing
    // older. The two consumed inputs are gone.
    assert_eq!(
        released.retained_bytes, inside.live_bytes,
        "only the newest version below the watermark is still held"
    );
    assert!(
        released.retained_bytes < inside.retained_bytes + inside.live_bytes,
        "the inputs consumed inside the window were released"
    );
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
