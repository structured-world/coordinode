use super::*;

#[test]
fn snapshot_read_ts() {
    let ts = Timestamp::from_raw(42);
    let snap = Snapshot::at(ts);
    assert_eq!(snap.read_ts().as_raw(), 42);
}

#[test]
fn retention_default_is_7_days() {
    let policy = RetentionPolicy::default();
    assert_eq!(policy.window(), Duration::from_secs(7 * 24 * 3600));
}

#[test]
fn retention_old_version_is_expired() {
    let policy = RetentionPolicy::new(Duration::from_secs(3600)); // 1 hour
                                                                  // Timestamp from 2 hours ago (in microseconds)
    let two_hours_ago = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_micros() as u64
        - 2 * 3600 * 1_000_000;

    assert!(policy.is_expired(Timestamp::from_raw(two_hours_ago)));
}

#[test]
fn retention_recent_version_not_expired() {
    let policy = RetentionPolicy::new(Duration::from_secs(3600));
    let now_micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_micros() as u64;

    assert!(!policy.is_expired(Timestamp::from_raw(now_micros)));
}
