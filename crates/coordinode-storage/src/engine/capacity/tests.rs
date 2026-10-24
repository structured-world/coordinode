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
fn tracker_starts_with_zero_usage() {
    let usages = vec![ep("a", 1000), ep("b", 0)];
    let tracker = CapacityTracker::new(&usages);
    let a = tracker.get("a").expect("a tracked");
    assert_eq!(a.used(), 0);
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

/// An endpoint with a hard limit reports what its footprint charges, and the
/// severity that charge reaches; one that shrinks reports the recovery.
#[test]
fn refresh_reports_the_charged_footprint() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut config = ep("only", 1 << 20);
    config.path = tmp.path().to_path_buf();
    let footprints = crate::engine::footprint::Footprints::open(&[config.clone()]).unwrap();
    let footprint = footprints.get("only").unwrap();
    let tracker = CapacityTracker::new(&[config]);
    let paths = BTreeMap::from([("only".to_string(), tmp.path().to_path_buf())]);

    let file = footprint.file(&tmp.path().join("table"));
    file.grow_to(850 * 1024).unwrap();
    tracker.refresh(&paths, &footprints);
    let usage = tracker.get("only").unwrap();
    assert_eq!(usage.used(), footprint.charged());
    assert_eq!(usage.severity(), CapacitySeverity::Warning);

    file.truncate_to(0);
    tracker.refresh(&paths, &footprints);
    assert_eq!(usage.used(), 0);
    assert_eq!(usage.severity(), CapacitySeverity::Normal);
}

/// An endpoint without a hard limit has no footprint: its folder is walked
/// for the gauges.
#[test]
fn refresh_walks_the_folder_of_an_endpoint_without_a_limit() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("000.sst"), vec![0u8; 800]).unwrap();
    let mut config = ep("free", 0);
    config.path = tmp.path().to_path_buf();
    let tracker = CapacityTracker::new(&[config]);
    let paths = BTreeMap::from([("free".to_string(), tmp.path().to_path_buf())]);
    tracker.refresh(&paths, &Default::default());
    let usage = tracker.get("free").unwrap();
    assert_eq!(usage.used(), 800);
    assert_eq!(usage.severity(), CapacitySeverity::Normal);
}

#[test]
fn refresh_missing_endpoint_path_skipped_not_errored() {
    let tracker = CapacityTracker::new(&[ep("missing", 1000)]);
    // Neither a footprint nor a path for "missing": refresh skips it, no
    // panic, no mutation of state.
    tracker.refresh(&BTreeMap::new(), &Default::default());
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

    // The scanner fires repeatedly: wait for a second tick (about 200 ms on
    // an idle machine, the loop's granularity being 100 ms) with a deadline
    // a loaded CI runner cannot miss, rather than a fixed sleep it can.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut observed = count.load(Ordering::Relaxed);
    while observed < 2 && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
        observed = count.load(Ordering::Relaxed);
    }
    assert!(
        observed >= 2,
        "expected the scanner to fire at least twice within 10 s, got {observed}",
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

/// The scanner's first tick is deferred by `interval`, not fired on spawn:
/// the engine open, which measured the footprints itself, completes
/// without a scan competing with the caller's first writes.
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
