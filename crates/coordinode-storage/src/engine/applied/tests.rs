use std::time::Duration;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use tempfile::TempDir;

use super::AppliedEvent;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;

fn engine(dir: &TempDir) -> StorageEngine {
    StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open")
}

fn put(partition: PartitionId, key: &[u8]) -> Mutation {
    Mutation::Put {
        partition,
        key: key.to_vec(),
        value: b"v".to_vec(),
    }
}

const NO_WAIT: Duration = Duration::from_millis(0);

/// An applied entry reaches the subscriber with its index, its commit
/// timestamp and the keys it wrote in the subscribed partition, and nothing
/// of the partitions the subscriber does not follow.
#[test]
fn an_applied_entry_reports_its_keys_in_the_partition() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied(Partition::Node, 16);

    engine
        .apply_raft_proposal(
            &[
                put(PartitionId::Node, b"a"),
                put(PartitionId::EdgeProp, b"elsewhere"),
                Mutation::Delete {
                    partition: PartitionId::Node,
                    key: b"b".to_vec(),
                },
            ],
            42,
            7,
            0,
            |_| false,
        )
        .expect("apply");
    engine
        .apply_raft_proposal(&[put(PartitionId::EdgeProp, b"x")], 43, 8, 0, |_| false)
        .expect("apply");

    assert_eq!(
        sub.next(NO_WAIT),
        Some(AppliedEvent::Keys {
            index: 7,
            commit_ts: 42,
            keys: vec![b"a".to_vec(), b"b".to_vec()],
        })
    );
    assert_eq!(sub.try_next(), None, "entry 8 wrote nothing in Node");
}

/// What a partition the entry skips (because it already holds it) receives
/// is not reported: the entry did not write there.
#[test]
fn a_skipped_partition_reports_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied(Partition::Node, 16);

    engine
        .apply_raft_proposal(&[put(PartitionId::Node, b"a")], 5, 1, 0, |p| {
            p == Partition::Node
        })
        .expect("apply");

    assert_eq!(sub.try_next(), None);
}

/// Only the Raft applies are reported: a write outside them is not an entry
/// of the log.
#[test]
fn a_write_outside_the_applies_is_not_reported() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied(Partition::Node, 16);

    engine.put(Partition::Node, b"a", b"v").expect("put");

    assert_eq!(sub.try_next(), None);
}

/// A range removed from the partition cannot be listed key by key.
#[test]
fn a_range_removed_reports_the_partition_replaced() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied(Partition::Node, 16);

    engine
        .apply_raft_proposal(
            &[Mutation::RemoveRange {
                partition: PartitionId::Node,
                start: b"a".to_vec(),
                end: b"z".to_vec(),
            }],
            9,
            3,
            0,
            |_| false,
        )
        .expect("apply");

    assert_eq!(sub.next(NO_WAIT), Some(AppliedEvent::Replaced));
}

/// A snapshot installed in place of the applies replaces every partition;
/// a partition rebuilt from a peer or a checkpoint replaces that one only.
#[test]
fn a_snapshot_or_a_rebuilt_partition_reports_the_partition_replaced() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(&dir);
    let node = engine.subscribe_applied(Partition::Node, 16);
    let edges = engine.subscribe_applied(Partition::EdgeProp, 16);

    engine.reset_raft_coverage(10, &[]).expect("reset");
    assert_eq!(node.next(NO_WAIT), Some(AppliedEvent::Replaced));
    assert_eq!(edges.next(NO_WAIT), Some(AppliedEvent::Replaced));

    engine
        .begin_partition_rebuild(Partition::Node, &Vec::new())
        .expect("begin");
    engine
        .finish_raft_rebuild(Partition::Node, 10, &[])
        .expect("finish");
    assert_eq!(node.next(NO_WAIT), Some(AppliedEvent::Replaced));
    assert_eq!(edges.try_next(), None, "EdgeProp was not rebuilt");
}

/// The applies never wait on a subscriber: a full queue drops the event and
/// the subscriber is told once to read the store afresh, then receives what
/// was queued and what comes after.
#[test]
fn a_full_queue_is_reported_as_replaced_without_blocking_the_applies() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied(Partition::Node, 1);

    for index in 1..=3u64 {
        engine
            .apply_raft_proposal(&[put(PartitionId::Node, b"k")], index, index, 0, |_| false)
            .expect("apply never waits on the subscriber");
    }

    assert_eq!(sub.next(NO_WAIT), Some(AppliedEvent::Replaced));
    assert!(matches!(
        sub.next(NO_WAIT),
        Some(AppliedEvent::Keys { index: 1, .. })
    ));
    assert_eq!(sub.try_next(), None);

    engine
        .apply_raft_proposal(&[put(PartitionId::Node, b"k")], 4, 4, 0, |_| false)
        .expect("apply");
    assert!(matches!(
        sub.next(NO_WAIT),
        Some(AppliedEvent::Keys { index: 4, .. })
    ));
}

/// A dropped subscription stops receiving, and the applies go back to one
/// load and no work.
#[test]
fn a_dropped_subscription_is_forgotten() {
    let dir = TempDir::new().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied(Partition::Node, 16);
    drop(sub);
    assert_eq!(engine.applied_subscriptions(), 0);
    engine
        .apply_raft_proposal(&[put(PartitionId::Node, b"a")], 1, 1, 0, |_| false)
        .expect("apply with no subscriber");
}
