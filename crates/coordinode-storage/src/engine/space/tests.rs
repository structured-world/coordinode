use super::*;
use crate::engine::config::{EndpointConfig, Media, Tier};

fn guard_over(dir: &std::path::Path) -> SpaceGuard {
    SpaceGuard::new(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
}

/// With the default reserve on a disk that has room, writes are admitted
/// and the reading reports the real free space.
#[test]
fn a_disk_with_room_admits_writes() {
    let dir = tempfile::tempdir().unwrap();
    let guard = guard_over(dir.path());
    let free = fs4::available_space(dir.path()).unwrap();
    if free < DEFAULT_RESUME_FREE_BYTES {
        // A host this full would pause by design; nothing to check here.
        return;
    }
    assert!(guard.admit().is_ok());
    assert!(!guard.is_paused());
    assert!(guard.available_bytes() <= free + (64 << 20));
}

/// A reserve above the free space pauses writes at once, the refusal names
/// the path and both numbers, and lowering the reserve resumes them.
#[test]
fn a_reserve_above_the_free_space_refuses_writes() {
    let dir = tempfile::tempdir().unwrap();
    let guard = guard_over(dir.path());
    guard.set_reserve(u64::MAX, u64::MAX);
    assert!(guard.is_paused());
    match guard.admit() {
        Err(StorageError::OutOfSpace {
            path,
            available_bytes,
            min_free_bytes,
        }) => {
            assert_eq!(path, dir.path().display().to_string());
            assert_eq!(min_free_bytes, u64::MAX);
            assert!(available_bytes < u64::MAX);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    guard.set_reserve(0, 0);
    assert!(guard.admit().is_ok(), "space above the resume level again");
}

/// Once paused, writes stay refused until free space reaches the resume
/// level, not merely the reserve: no flip-flopping at the boundary.
#[test]
fn writes_resume_only_above_the_resume_level() {
    let dir = tempfile::tempdir().unwrap();
    let guard = guard_over(dir.path());
    let free = fs4::available_space(dir.path()).unwrap();
    guard.set_reserve(u64::MAX, u64::MAX);
    assert!(guard.is_paused());
    // The reserve drops below the free space but the resume level stays
    // above it: still paused.
    guard.set_reserve(free / 2, u64::MAX);
    assert!(guard.is_paused());
    guard.set_reserve(free / 2, free / 2);
    assert!(!guard.is_paused());
    // Not paused, the reserve alone decides again.
    guard.set_reserve(free / 2, u64::MAX);
    assert!(!guard.is_paused());
}

/// A resume level below the reserve is taken as the reserve.
#[test]
fn the_resume_level_is_at_least_the_reserve() {
    let dir = tempfile::tempdir().unwrap();
    let guard = guard_over(dir.path());
    guard.set_reserve(10, 5);
    assert_eq!(guard.resume_free_bytes(), 10);
}

/// How many times `guard` read free space.
fn readings(guard: &SpaceGuard) -> u64 {
    guard.readings.load(Ordering::Relaxed)
}

/// Wait up to 5 s for `guard` to read free space more than `seen` times.
fn read_again(guard: &SpaceGuard, seen: u64) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if readings(guard) > seen {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

/// A write never reads free space: admissions, reservations and the
/// paused checks answer from the monitor's last reading. Before, a write
/// read the filesystem itself whenever the reading was 250 ms old.
#[test]
fn admitting_writes_never_reads_free_space() {
    let dir = tempfile::tempdir().unwrap();
    let guard = guard_over(dir.path());
    guard.set_reserve(0, 0);
    let seen = readings(&guard);
    let until = std::time::Instant::now() + Duration::from_millis(400);
    while std::time::Instant::now() < until {
        guard.admit().unwrap();
        guard.admit_free_space().unwrap();
        guard.admit_free_space_recovery().unwrap();
        drop(guard.reserve(1, Admit::Client).unwrap());
        assert!(!guard.is_paused());
        assert!(!guard.is_paused_by_free_space());
    }
    assert_eq!(readings(&guard), seen, "a write read free space");
}

/// Writes that spend the headroom the last reading left wake the monitor
/// at once, long before its interval.
#[test]
fn spending_the_headroom_wakes_the_monitor() {
    let dir = tempfile::tempdir().unwrap();
    let guard = Arc::new(guard_over(dir.path()));
    guard.set_reserve(0, 0);
    guard.set_check_interval_ms(3_600_000);
    let _monitor = guard.start_monitor().unwrap();
    // The interval change woke the monitor before it ran: let it take that
    // reading first.
    std::thread::sleep(Duration::from_millis(100));
    let seen = readings(&guard);
    let headroom = u64::try_from(guard.headroom.load(Ordering::Relaxed)).unwrap();
    assert!(headroom > 0, "an empty reserve leaves headroom");
    guard.spend(headroom / 2);
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(readings(&guard), seen, "half the headroom wakes nothing");
    guard.spend(headroom);
    assert!(
        read_again(&guard, seen),
        "the spent headroom woke no reading"
    );
}

/// With nothing waking it, the monitor reads free space on its interval:
/// other programs fill the disk too.
#[test]
fn the_monitor_reads_on_its_interval() {
    let dir = tempfile::tempdir().unwrap();
    let guard = Arc::new(guard_over(dir.path()));
    guard.set_reserve(0, 0);
    guard.set_check_interval_ms(50);
    let _monitor = guard.start_monitor().unwrap();
    let seen = readings(&guard);
    assert!(read_again(&guard, seen + 2), "no readings on the interval");
}

/// While writes are paused the monitor reads every 250 ms, whatever its
/// interval, so space freed by anyone admits them again promptly.
#[test]
fn a_paused_guard_is_read_often() {
    let dir = tempfile::tempdir().unwrap();
    let guard = Arc::new(guard_over(dir.path()));
    guard.set_check_interval_ms(3_600_000);
    guard.set_reserve(u64::MAX, u64::MAX);
    assert!(guard.is_paused());
    let _monitor = guard.start_monitor().unwrap();
    let seen = readings(&guard);
    assert!(read_again(&guard, seen + 2), "a paused guard was not read");
}

/// Volatile endpoints hold no durable writes to protect: an all-volatile
/// store never pauses on their account.
#[test]
fn volatile_endpoints_are_not_measured() {
    let config = StorageConfig::with_endpoints_no_persistence(vec![EndpointConfig::new(
        "ram",
        "memory",
        Media::Ram,
        Durability::Volatile,
        Tier::Memory,
    )]);
    let guard = SpaceGuard::new(&config);
    guard.set_reserve(u64::MAX, u64::MAX);
    assert!(guard.admit().is_ok());
}
