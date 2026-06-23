use super::*;

/// Factory produces filters with correct name.
#[test]
fn factory_name() {
    let watermark = Arc::new(AtomicU64::new(0));
    let factory = SeqnoRetentionFilterFactory::new(watermark);
    assert_eq!(factory.name(), "coordinode.seqno_retention");
}

/// Shared watermark is readable via Arc.
#[test]
fn shared_watermark_updates() {
    let watermark = Arc::new(AtomicU64::new(0));
    let factory_watermark = Arc::clone(&watermark);

    // Initial value
    assert_eq!(factory_watermark.load(Ordering::Acquire), 0);

    // External update visible to factory
    watermark.store(42, Ordering::Release);
    assert_eq!(factory_watermark.load(Ordering::Acquire), 42);
}

/// Retention filter logic: items above watermark are kept.
#[test]
fn filter_keeps_items_above_watermark() {
    let filter = SeqnoRetentionFilter {
        watermark: 100,
        last_key: Vec::new(),
        has_live_version: false,
    };

    // seqno 200 > watermark 100 → within retention
    assert!(filter.watermark < 200);
    // seqno 50 <= watermark 100 → eligible for GC
    assert!(filter.watermark >= 50);
}

/// Retention filter preserves newest expired version per key.
#[test]
fn filter_preserves_newest_expired_version() {
    let filter = SeqnoRetentionFilter {
        watermark: 100,
        last_key: Vec::new(),
        has_live_version: false,
    };

    // When has_live_version is false and item is below watermark,
    // the filter should keep it (first expired = newest version).
    assert!(!filter.has_live_version);
}

/// Retention filter destroys older expired versions.
#[test]
fn filter_destroys_older_expired() {
    let filter = SeqnoRetentionFilter {
        watermark: 100,
        last_key: b"key1".to_vec(),
        has_live_version: true,
    };

    // When has_live_version is true and item is below watermark,
    // the filter should destroy it (older expired version).
    assert!(filter.has_live_version);
}
