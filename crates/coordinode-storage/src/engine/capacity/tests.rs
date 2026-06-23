use super::*;
use crate::engine::config::{Durability, Media, Tier};

fn ep(id: &str, hard_limit: u64) -> EndpointConfig {
    let mut e = EndpointConfig::new(
        id,
        format!("/{id}"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    );
    e.hard_limit_bytes = hard_limit;
    e
}

#[test]
fn severity_thresholds_map_to_expected_ranges() {
    let limit = 100u64;
    assert_eq!(
        CapacitySeverity::for_usage(0, limit),
        CapacitySeverity::Normal
    );
    assert_eq!(
        CapacitySeverity::for_usage(79, limit),
        CapacitySeverity::Normal
    );
    assert_eq!(
        CapacitySeverity::for_usage(80, limit),
        CapacitySeverity::Warning
    );
    assert_eq!(
        CapacitySeverity::for_usage(89, limit),
        CapacitySeverity::Warning
    );
    assert_eq!(
        CapacitySeverity::for_usage(90, limit),
        CapacitySeverity::Critical
    );
    assert_eq!(
        CapacitySeverity::for_usage(94, limit),
        CapacitySeverity::Critical
    );
    assert_eq!(
        CapacitySeverity::for_usage(95, limit),
        CapacitySeverity::Emergency
    );
    assert_eq!(
        CapacitySeverity::for_usage(99, limit),
        CapacitySeverity::Emergency
    );
    assert_eq!(
        CapacitySeverity::for_usage(100, limit),
        CapacitySeverity::Full
    );
    assert_eq!(
        CapacitySeverity::for_usage(200, limit),
        CapacitySeverity::Full,
        "above 100% still maps to Full",
    );
}

#[test]
fn severity_with_no_hard_limit_is_normal() {
    // `hard_limit_bytes == 0` = "no limit configured" — never
    // alerts, never blocks writes. Untracked endpoint.
    assert_eq!(
        CapacitySeverity::for_usage(u64::MAX, 0),
        CapacitySeverity::Normal,
    );
}

#[test]
fn severity_labels_match_prometheus_convention() {
    assert_eq!(CapacitySeverity::Normal.label(), "normal");
    assert_eq!(CapacitySeverity::Warning.label(), "warning");
    assert_eq!(CapacitySeverity::Critical.label(), "critical");
    assert_eq!(CapacitySeverity::Emergency.label(), "emergency");
    assert_eq!(CapacitySeverity::Full.label(), "full");
}

#[test]
fn tracker_starts_writable_with_zero_usage() {
    let usages = vec![ep("a", 1000), ep("b", 0)];
    let tracker = CapacityTracker::new(&usages);
    let a = tracker.get("a").expect("a tracked");
    assert_eq!(a.used(), 0);
    assert!(a.is_writable(), "fresh endpoint is writable");
    assert_eq!(a.severity(), CapacitySeverity::Normal);
    let b = tracker.get("b").expect("b tracked");
    assert_eq!(b.hard_limit_bytes, 0, "no-limit endpoint preserved");
}

#[test]
#[should_panic(expected = "duplicate endpoint id")]
fn tracker_duplicate_id_panics() {
    let a = ep("dup", 100);
    let b = ep("dup", 200);
    let _ = CapacityTracker::new(&[a, b]);
}

#[test]
fn tracker_iter_returns_stable_btree_order() {
    let tracker = CapacityTracker::new(&[ep("c", 1), ep("a", 1), ep("b", 1)]);
    let ids: Vec<&str> = tracker.iter().map(|(id, _)| id).collect();
    assert_eq!(ids, vec!["a", "b", "c"]);
}

#[test]
fn dir_size_sums_file_lengths_recursively() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(tmp.path().join("inner")).unwrap();
    std::fs::write(tmp.path().join("a.bin"), vec![0u8; 500]).unwrap();
    std::fs::write(tmp.path().join("inner").join("b.bin"), vec![0u8; 1500]).unwrap();
    assert_eq!(dir_size(tmp.path()), 2000);
}

#[test]
fn dir_size_missing_root_returns_zero() {
    // Recovery scan must tolerate missing dirs without panicking.
    assert_eq!(dir_size(Path::new("/nonexistent/abs/path")), 0);
}

#[test]
fn refresh_updates_used_and_severity() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ep_root = tmp.path().to_path_buf();
    let part_dir = ep_root.join("node").join("tables");
    std::fs::create_dir_all(&part_dir).unwrap();
    // Plant 800 B of fake SST data — should be 80% of a 1000 B limit.
    std::fs::write(part_dir.join("000.sst"), vec![0u8; 800]).unwrap();

    let mut ep_config = ep("only", 1000);
    ep_config.path = ep_root.clone();
    let tracker = CapacityTracker::new(&[ep_config]);

    let mut paths = BTreeMap::new();
    paths.insert("only".to_string(), ep_root);
    tracker.refresh(&paths, &["node"]);

    let usage = tracker.get("only").unwrap();
    assert_eq!(usage.used(), 800);
    assert_eq!(usage.severity(), CapacitySeverity::Warning);
    assert!(
        usage.is_writable(),
        "Warning severity still writable — only Full flips the flag",
    );
}

#[test]
fn refresh_full_severity_flips_is_writable_off() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ep_root = tmp.path().to_path_buf();
    let part_dir = ep_root.join("node").join("tables");
    std::fs::create_dir_all(&part_dir).unwrap();
    // 1100 B > 1000 B hard_limit → Full severity.
    std::fs::write(part_dir.join("000.sst"), vec![0u8; 1100]).unwrap();

    let mut ep_config = ep("only", 1000);
    ep_config.path = ep_root.clone();
    let tracker = CapacityTracker::new(&[ep_config]);

    let mut paths = BTreeMap::new();
    paths.insert("only".to_string(), ep_root);
    tracker.refresh(&paths, &["node"]);

    let usage = tracker.get("only").unwrap();
    assert_eq!(usage.severity(), CapacitySeverity::Full);
    assert!(
        !usage.is_writable(),
        "Full severity must flip is_writable off — writes targeting this endpoint reject",
    );
}

#[test]
fn refresh_recovery_flips_is_writable_back_on() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ep_root = tmp.path().to_path_buf();
    let part_dir = ep_root.join("node").join("tables");
    std::fs::create_dir_all(&part_dir).unwrap();

    let mut ep_config = ep("only", 1000);
    ep_config.path = ep_root.clone();
    let tracker = CapacityTracker::new(&[ep_config]);
    let mut paths = BTreeMap::new();
    paths.insert("only".to_string(), ep_root.clone());

    // Step 1: full.
    let big_path = part_dir.join("big.sst");
    std::fs::write(&big_path, vec![0u8; 1500]).unwrap();
    tracker.refresh(&paths, &["node"]);
    let usage = tracker.get("only").unwrap();
    assert!(!usage.is_writable());

    // Step 2: cascade-eviction analog — operator deletes the big
    // SST. Next refresh observes a lower usage and must flip the
    // writable flag back on.
    std::fs::remove_file(&big_path).unwrap();
    tracker.refresh(&paths, &["node"]);
    assert!(
        usage.is_writable(),
        "recovery below threshold must re-enable writes",
    );
    assert_eq!(usage.severity(), CapacitySeverity::Normal);
}

#[test]
fn refresh_missing_endpoint_path_skipped_not_errored() {
    let tracker = CapacityTracker::new(&[ep("missing", 1000)]);
    // No path entry for "missing" → refresh skips, no panic, no
    // mutation of state.
    tracker.refresh(&BTreeMap::new(), &["node"]);
    assert_eq!(tracker.get("missing").unwrap().used(), 0);
}

#[test]
fn scanner_fires_refresh_then_shuts_down() {
    // The scanner thread must invoke refresh_fn at least once per
    // interval and must exit promptly after Drop.
    let count = Arc::new(AtomicU64::new(0));
    let cnt_clone = Arc::clone(&count);
    let scanner = CapacityScanner::start(Duration::from_millis(50), move || {
        cnt_clone.fetch_add(1, Ordering::Relaxed);
    })
    .expect("scanner spawn");

    // Sleep long enough to observe at least 2 ticks. The scanner
    // defers its first refresh by `interval` (see
    // `first_tick_is_deferred_by_interval`), and the loop's
    // shutdown-check granularity is 100 ms, so the effective
    // cadence on a 50 ms interval is ~100 ms — yielding ≥2 ticks
    // in 350 ms (first at ~100 ms, second at ~200 ms, third ~300).
    std::thread::sleep(Duration::from_millis(350));
    let observed = count.load(Ordering::Relaxed);
    assert!(
        observed >= 2,
        "expected scanner to fire at least 2 times in 350 ms with 50 ms interval, got {observed}",
    );

    // Drop → shutdown → join. Should return within
    // tick_granularity (100 ms) + epsilon. We bound the wait with
    // a timeout via std::thread::scope to fail loudly if join
    // hangs.
    let start = std::time::Instant::now();
    drop(scanner);
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(1),
        "scanner shutdown should be near-instant (<1 s), took {elapsed:?}",
    );
}

#[test]
fn scanner_shutdown_is_idempotent() {
    // Calling shutdown() multiple times must not panic and must
    // not affect drop semantics. Guards against a double-drop
    // pattern (e.g., explicit close in error-handling path
    // followed by RAII drop).
    let scanner = CapacityScanner::start(Duration::from_secs(60), || {}).expect("spawn");
    scanner.shutdown();
    scanner.shutdown();
    scanner.shutdown();
    // Drop must still join cleanly.
    drop(scanner);
}

/// Regression test for the warm-load race: the scanner's first tick
/// MUST be deferred by `interval`, not fire immediately on spawn.
///
/// Why: at engine open, `load_persisted_capacity` warm-hydrates the
/// tracker from Schema. The user thread then starts writes + calls
/// `engine.refresh_capacity()` to publish a fresh measurement to
/// Schema. If the scanner also fires its first tick concurrently,
/// its slow disk scan can finish AFTER the user's manual refresh,
/// grab a higher seqno, and write a stale (or empty) measurement
/// that wins LSM merge on the next reopen — clobbering the value
/// the user just committed.
///
/// Repro on the unwanted-immediate-tick code:
/// - spawn scanner with `interval = 1s`, counter `Arc<AtomicU32>`
/// - sleep 200 ms (well under interval)
/// - assert counter == 0 (no tick yet)
///
/// Before the fix this assertion FAILS (counter == 1 because the
/// loop body calls refresh_fn() immediately).
#[test]
fn first_tick_is_deferred_by_interval() {
    use std::sync::atomic::AtomicU32;
    use std::thread;
    let count = Arc::new(AtomicU32::new(0));
    let count_c = Arc::clone(&count);
    let scanner = CapacityScanner::start(Duration::from_secs(1), move || {
        count_c.fetch_add(1, Ordering::Release);
    })
    .expect("spawn");
    // Sleep well under `interval` so no scheduled tick is due yet.
    thread::sleep(Duration::from_millis(200));
    let observed = count.load(Ordering::Acquire);
    scanner.shutdown();
    drop(scanner);
    assert_eq!(
        observed, 0,
        "scanner fired its first tick before `interval` elapsed (warm-load race)",
    );
}
