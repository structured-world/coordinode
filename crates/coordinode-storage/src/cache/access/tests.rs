use super::*;
use crate::engine::partition::Partition;
use crate::engine::scope::NamespaceId;

/// The node tree of namespace 1, the tree most tests track keys of.
const NODE: TreeId = TreeId::namespace(NamespaceId::from_raw(1), Partition::Node);
/// The adjacency tree of namespace 1.
const ADJ: TreeId = TreeId::namespace(NamespaceId::from_raw(1), Partition::Adj);

#[test]
fn record_and_get() {
    let tracker = AccessTracker::new();
    tracker.record(NODE, b"key1");
    tracker.record(NODE, b"key1");
    tracker.record(NODE, b"key1");

    let stats = tracker.get(NODE, b"key1").expect("should exist");
    assert_eq!(stats.count, 3);
    assert!(stats.last_access > 0);
}

#[test]
fn different_keys_independent() {
    let tracker = AccessTracker::new();
    tracker.record(NODE, b"key1");
    tracker.record(NODE, b"key1");
    tracker.record(NODE, b"key2");

    let s1 = tracker.get(NODE, b"key1").expect("key1");
    let s2 = tracker.get(NODE, b"key2").expect("key2");
    assert_eq!(s1.count, 2);
    assert_eq!(s2.count, 1);
}

#[test]
fn different_partitions_independent() {
    let tracker = AccessTracker::new();
    tracker.record(NODE, b"key");
    tracker.record(ADJ, b"key");

    assert!(tracker.get(NODE, b"key").is_some());
    assert!(tracker.get(ADJ, b"key").is_some());
    assert_eq!(tracker.len(), 2);
}

/// The same partition and key of two namespaces are two tracked keys.
#[test]
fn different_namespaces_independent() {
    let tracker = AccessTracker::new();
    let other = TreeId::namespace(NamespaceId::from_raw(2), Partition::Node);
    tracker.record(NODE, b"key");
    tracker.record(NODE, b"key");
    tracker.record(other, b"key");

    assert_eq!(tracker.get(NODE, b"key").expect("node").count, 2);
    assert_eq!(tracker.get(other, b"key").expect("other").count, 1);
    assert_ne!(
        compute_cache_key(NODE, b"key"),
        compute_cache_key(other, b"key")
    );
}

#[test]
fn missing_key_returns_none() {
    let tracker = AccessTracker::new();
    assert!(tracker.get(NODE, b"missing").is_none());
}

#[test]
fn remove_key() {
    let tracker = AccessTracker::new();
    tracker.record(NODE, b"key");
    assert_eq!(tracker.len(), 1);

    let cache_key = compute_cache_key(NODE, b"key");
    tracker.remove(cache_key);
    assert!(tracker.is_empty());
}

#[test]
fn coldest_returns_least_accessed() {
    let tracker = AccessTracker::new();

    // key1: 10 accesses
    for _ in 0..10 {
        tracker.record(NODE, b"hot");
    }
    // key2: 1 access
    tracker.record(NODE, b"cold");

    let coldest = tracker.coldest(1);
    assert_eq!(coldest.len(), 1);
    assert_eq!(coldest[0], compute_cache_key(NODE, b"cold"));
}

#[test]
fn aggregate_stats() {
    let tracker = AccessTracker::new();
    for _ in 0..5 {
        tracker.record(NODE, b"hot");
    }
    tracker.record(NODE, b"cold");

    let agg = tracker.aggregate();
    assert_eq!(agg.unique_keys, 2);
    assert_eq!(agg.total_accesses, 6);
    assert_eq!(agg.hottest_count, 5);
    assert_eq!(agg.hottest_key, Some(compute_cache_key(NODE, b"hot")));
}

#[test]
fn cache_key_deterministic() {
    let k1 = compute_cache_key(NODE, b"test");
    let k2 = compute_cache_key(NODE, b"test");
    assert_eq!(k1, k2);

    // Different partition = different key
    let k3 = compute_cache_key(ADJ, b"test");
    assert_ne!(k1, k3);
}

#[test]
fn empty_tracker() {
    let tracker = AccessTracker::new();
    assert!(tracker.is_empty());
    assert_eq!(tracker.len(), 0);
    assert!(tracker.coldest(10).is_empty());
}
