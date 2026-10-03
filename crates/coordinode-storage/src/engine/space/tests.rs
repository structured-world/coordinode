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
