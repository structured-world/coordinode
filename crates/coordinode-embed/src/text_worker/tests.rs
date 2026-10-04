use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_query::index::TextReadiness;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

fn engine(dir: &tempfile::TempDir) -> StorageEngine {
    StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]))
    .expect("open")
}

/// Apply one Node write as entry `index`, so the subscription is offered
/// one more event.
fn apply(engine: &StorageEngine, index: u64) {
    engine
        .apply_raft_proposal(
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:k{index}").into_bytes(),
                value: b"v".to_vec(),
            }],
            10 + index,
            index,
            0,
            |_| false,
        )
        .expect("apply");
}

/// With nothing applied, or everything applied folded, a search goes ahead
/// at once.
#[test]
fn a_search_over_covered_indexes_does_not_wait() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied(Partition::Node, 16);
    let readiness = TextReadiness::new(sub.position(), Duration::from_secs(5));

    let started = Instant::now();
    assert_eq!(readiness.await_covered(), Ok(()));
    apply(&engine, 1);
    readiness.advance(1);
    assert_eq!(readiness.await_covered(), Ok(()));
    assert!(started.elapsed() < Duration::from_secs(1));
}

/// An applied entry the worker has not folded keeps a search waiting; past
/// the wait it fails naming what the indexes hold and what it needed,
/// instead of answering without the entry.
#[test]
fn a_search_ahead_of_the_indexes_fails_after_its_wait() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied(Partition::Node, 16);
    let readiness = TextReadiness::new(sub.position(), Duration::from_millis(50));
    apply(&engine, 1);
    apply(&engine, 2);
    readiness.advance(1);

    let err = readiness
        .await_covered()
        .expect_err("the indexes lack entry 2");
    assert_eq!((err.folded, err.needed), (1, 2));
    assert!(err.waited_ms >= 50, "{err:?}");
}

/// A search waiting on the indexes goes ahead as soon as the worker folds
/// what it needs, without waiting out its bound.
#[test]
fn a_waiting_search_goes_ahead_when_the_worker_catches_up() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied(Partition::Node, 16);
    let readiness = Arc::new(TextReadiness::new(sub.position(), Duration::from_secs(30)));
    apply(&engine, 1);

    let worker = {
        let readiness = Arc::clone(&readiness);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            readiness.advance(1);
        })
    };
    let started = Instant::now();
    assert_eq!(readiness.await_covered(), Ok(()));
    assert!(started.elapsed() < Duration::from_secs(10));
    worker.join().expect("worker");
}

/// The wait can be changed on a live readiness; zero refuses a search ahead
/// of the indexes at once.
#[test]
fn the_wait_is_retunable_and_zero_refuses_at_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = engine(&dir);
    let sub = engine.subscribe_applied(Partition::Node, 16);
    let readiness = TextReadiness::new(sub.position(), Duration::from_secs(30));
    apply(&engine, 1);
    readiness.set_wait(Duration::ZERO);

    let started = Instant::now();
    assert!(readiness.await_covered().is_err());
    assert!(started.elapsed() < Duration::from_secs(1));
}
