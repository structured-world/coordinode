use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use coordinode_core::txn::proposal::{Mutation as Proposed, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use tempfile::TempDir;

use super::Tapped;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;

fn config(dir: &TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn keys(tapped: Tapped) -> Vec<Vec<u8>> {
    match tapped {
        Tapped::Keys(mut keys) => {
            keys.sort();
            keys
        }
        Tapped::Replaced => panic!("expected keys, the partition was reported replaced"),
    }
}

fn put(key: &[u8]) -> Proposed {
    Proposed::Put {
        partition: PartitionId::Node,
        key: key.to_vec(),
        value: b"v".to_vec(),
    }
}

/// A tap delivers the writes that land after it opens and none from before,
/// each key once however often it was written, and only for its partition.
#[test]
fn a_tap_delivers_each_key_written_after_it_opened_once() {
    let dir = TempDir::new().expect("tempdir");
    let engine = StorageEngine::open(&config(&dir)).expect("open");
    engine.put(Partition::Node, b"before", b"v").expect("put");

    let (tap, _) = engine.tap_writes(Partition::Node).expect("tap");
    engine.put(Partition::Node, b"a", b"1").expect("put");
    engine.put(Partition::Node, b"a", b"2").expect("put");
    engine.delete(Partition::Node, b"b").expect("delete");
    engine
        .put(Partition::EdgeProp, b"other", b"v")
        .expect("put");

    assert_eq!(keys(tap.take()), vec![b"a".to_vec(), b"b".to_vec()]);
    assert_eq!(
        keys(tap.take()),
        Vec::<Vec<u8>>::new(),
        "take leaves it empty"
    );
}

/// The batch path, which transactions and Raft applies take, is tapped too.
#[test]
fn a_tap_delivers_the_keys_of_an_applied_proposal() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config(&dir), Arc::clone(&oracle)).expect("open");
    let (tap, _) = engine.tap_writes(Partition::Node).expect("tap");
    engine
        .apply_proposal_at(&[put(b"x"), put(b"y")], oracle.next().as_raw())
        .expect("apply");
    assert_eq!(keys(tap.take()), vec![b"x".to_vec(), b"y".to_vec()]);
}

/// The hole a timestamp cannot close: a write committed with a timestamp
/// below the tap's snapshot, applied after the tap opened. The snapshot does
/// not show it, so the tap has to.
#[test]
fn a_write_landing_below_the_tap_snapshot_is_delivered() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config(&dir), Arc::clone(&oracle)).expect("open");
    let early = oracle.next().as_raw();
    let (tap, at) = engine.tap_writes(Partition::Node).expect("tap");
    assert!(
        early < at,
        "the commit timestamp was taken before the snapshot"
    );

    engine
        .apply_proposal_at(&[put(b"late")], early)
        .expect("apply");

    assert_eq!(keys(tap.take()), vec![b"late".to_vec()]);
}

/// A clear, a range tombstone and a range drop cannot be listed key by key,
/// so the tap reports the partition replaced and the consumer starts over;
/// a rebase empties it and keys flow again.
#[test]
fn replacing_the_partition_is_reported_and_a_rebase_resumes() {
    let dir = TempDir::new().expect("tempdir");
    let engine = StorageEngine::open(&config(&dir)).expect("open");
    let (tap, _) = engine.tap_writes(Partition::Node).expect("tap");

    engine.put(Partition::Node, b"a", b"v").expect("put");
    engine
        .remove_range(Partition::Node, b"a", b"z")
        .expect("range");
    engine.put(Partition::Node, b"b", b"v").expect("put");
    assert_eq!(tap.take(), Tapped::Replaced);

    engine.rebase_tap(&tap).expect("rebase");
    engine.put(Partition::Node, b"c", b"v").expect("put");
    assert_eq!(keys(tap.take()), vec![b"c".to_vec()]);

    engine.clear_partition(Partition::Node).expect("clear");
    assert_eq!(tap.take(), Tapped::Replaced);

    engine.rebase_tap(&tap).expect("rebase");
    engine
        .drop_range::<&[u8], _>(Partition::Node, ..)
        .expect("drop range");
    assert_eq!(tap.take(), Tapped::Replaced);
}

/// A dropped tap stops collecting, and the writes after it cost only the
/// check that no tap is open.
#[test]
fn a_dropped_tap_collects_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let engine = StorageEngine::open(&config(&dir)).expect("open");
    let (first, _) = engine.tap_writes(Partition::Node).expect("tap");
    let (second, _) = engine.tap_writes(Partition::Node).expect("tap");
    drop(first);
    engine.put(Partition::Node, b"a", b"v").expect("put");
    assert_eq!(keys(second.take()), vec![b"a".to_vec()]);
    drop(second);
    assert_eq!(engine.write_taps().open.load(Ordering::SeqCst), 0);
}

/// Pauses nothing, but counts how often it was asked to.
struct CountingFence {
    pauses: std::sync::atomic::AtomicUsize,
}

impl crate::engine::core::RaftApplyFence for CountingFence {
    fn with_applies_paused(
        &self,
        work: &mut dyn FnMut(
            &mut crate::engine::core::RaftApplyState,
            &[u8],
        ) -> crate::error::StorageResult<()>,
    ) -> crate::error::StorageResult<()> {
        self.pauses.fetch_add(1, Ordering::SeqCst);
        work(&mut crate::engine::core::RaftApplyState::new(0), &[])
    }
}

/// On a Raft store an apply lands at the leader's timestamp and advances
/// this node's clock only afterwards, so a snapshot read mid-apply can sit
/// below a write that already missed the tap. The snapshot is therefore read
/// with the applies paused, on opening and on every rebase.
#[test]
fn on_a_raft_store_the_tap_snapshot_is_read_with_the_applies_paused() {
    let dir = TempDir::new().expect("tempdir");
    let engine = StorageEngine::open(&config(&dir)).expect("open");
    let fence = Arc::new(CountingFence {
        pauses: std::sync::atomic::AtomicUsize::new(0),
    });
    engine.register_raft_fence(Arc::clone(&fence) as Arc<dyn crate::engine::core::RaftApplyFence>);

    let (tap, _) = engine.tap_writes(Partition::Node).expect("tap");
    assert_eq!(fence.pauses.load(Ordering::SeqCst), 1);
    engine.rebase_tap(&tap).expect("rebase");
    assert_eq!(fence.pauses.load(Ordering::SeqCst), 2);
}

/// Writers run while the tap opens: every key they wrote is either visible
/// at the tap's snapshot or delivered by the tap, never neither.
#[test]
fn concurrent_writes_are_visible_at_the_snapshot_or_delivered() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(
        StorageEngine::open_with_oracle(&config(&dir), Arc::clone(&oracle)).expect("open"),
    );
    let stop = Arc::new(AtomicBool::new(false));
    let writers: Vec<_> = (0..4u8)
        .map(|w| {
            let engine = Arc::clone(&engine);
            let oracle = Arc::clone(&oracle);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut written = Vec::new();
                let mut i = 0u32;
                while !stop.load(Ordering::Relaxed) {
                    let key = format!("w{w}-{i:08}").into_bytes();
                    let ts = oracle.next().as_raw();
                    engine.apply_proposal_at(&[put(&key)], ts).expect("apply");
                    written.push(key);
                    i += 1;
                }
                written
            })
        })
        .collect();

    std::thread::sleep(std::time::Duration::from_millis(20));
    let (tap, at) = engine.tap_writes(Partition::Node).expect("tap");
    let _pin = engine.pin_snapshot_at(at).expect("pin");
    std::thread::sleep(std::time::Duration::from_millis(20));
    stop.store(true, Ordering::Relaxed);
    let written: Vec<Vec<u8>> = writers
        .into_iter()
        .flat_map(|h| h.join().expect("writer"))
        .collect();

    let delivered: std::collections::HashSet<Vec<u8>> = keys(tap.take()).into_iter().collect();
    let mut missing = Vec::new();
    for key in &written {
        let visible = engine
            .snapshot_get(&at, Partition::Node, key)
            .expect("read")
            .is_some();
        if !visible && !delivered.contains(key) {
            missing.push(String::from_utf8_lossy(key).into_owned());
        }
    }
    assert!(
        missing.is_empty(),
        "{} of {} writes neither visible at the snapshot nor delivered: {:?}",
        missing.len(),
        written.len(),
        &missing[..missing.len().min(5)]
    );
}
