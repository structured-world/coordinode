use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use super::*;

/// Build a minimal in-memory partition tree map for testing.
fn make_test_trees() -> (HashMap<Partition, lsm_tree::AnyTree>, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let mut trees = HashMap::new();
    let tree = lsm_tree::Config::new_with_generators(
        dir.path().join("node"),
        Arc::clone(&seqno),
        Arc::clone(&seqno),
    )
    .open()
    .expect("open tree");
    trees.insert(Partition::Node, tree);
    (trees, dir)
}

#[test]
fn flush_manager_starts_and_stops() {
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    let mgr = FlushManager::start(
        &trees,
        Arc::clone(&gc_watermark),
        64 * 1024 * 1024, // 64MB threshold (won't trigger in this test)
        4,
        1,  // 1 worker
        50, // poll interval
        0,  // age trigger disabled
    )
    .expect("start FlushManager");

    // Brief sleep to let threads spin up.
    std::thread::sleep(Duration::from_millis(120));

    // Drop manager: should join all threads cleanly without hanging.
    drop(mgr);
}

#[test]
fn flush_manager_flushes_when_sealed_count_exceeded() {
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let tree = trees.get(&Partition::Node).expect("node tree").clone();

    // Write a small value so the memtable is non-empty before sealing.
    tree.insert(b"key1", b"value1", seqno.next());

    // Seal to produce 1 sealed memtable (below threshold of 4).
    tree.rotate_memtable();
    assert_eq!(tree.sealed_memtable_count(), 1, "one sealed before start");

    // FlushManager with max_sealed=0 so ANY sealed count triggers flush.
    let mgr = FlushManager::start(
        &trees,
        Arc::clone(&gc_watermark),
        u64::MAX, // size threshold: never triggers
        0,        // max_sealed=0: flush immediately when any sealed memtable exists
        1,
        20, // fast poll for test
        0,  // age trigger disabled — isolating the sealed-count gate
    )
    .expect("start FlushManager");

    // Wait up to 500ms for the flush to complete.
    let mut flushed = false;
    for _ in 0..25 {
        std::thread::sleep(Duration::from_millis(20));
        if tree.sealed_memtable_count() == 0 {
            flushed = true;
            break;
        }
    }

    drop(mgr);
    assert!(
        flushed,
        "FlushManager should have flushed the sealed memtable"
    );
}

#[test]
fn flush_manager_flushes_when_size_threshold_exceeded() {
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let tree = trees.get(&Partition::Node).expect("node tree").clone();

    // Write enough data to exceed a tiny threshold (1 byte).
    tree.insert(b"key1", b"value1_some_data", seqno.next());

    // FlushManager with threshold=1 byte so the active memtable exceeds it.
    let mgr = FlushManager::start(
        &trees,
        Arc::clone(&gc_watermark),
        1,   // 1 byte threshold — will always trigger
        100, // high sealed count so only size trigger fires
        1,
        20,
        0, // age trigger disabled — isolating the size gate
    )
    .expect("start FlushManager");

    // Wait up to 500ms for rotate + flush to complete.
    let mut flushed = false;
    for _ in 0..25 {
        std::thread::sleep(Duration::from_millis(20));
        // After flush: sealed=0 AND active size reset to 0.
        if tree.sealed_memtable_count() == 0 && tree.active_memtable().size() == 0 {
            flushed = true;
            break;
        }
    }

    drop(mgr);
    assert!(
        flushed,
        "FlushManager should have rotated and flushed the memtable"
    );
}

#[test]
fn flush_manager_multiple_workers_no_panic() {
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    // Start with 4 workers, rapid flush to exercise concurrency.
    let mgr = FlushManager::start(
        &trees,
        Arc::clone(&gc_watermark),
        1,  // 1 byte: always trigger
        0,  // max_sealed=0: always trigger
        4,  // 4 workers
        10, // fast poll
        0,  // age trigger disabled — concurrency comes from size+sealed
    )
    .expect("start FlushManager");

    std::thread::sleep(Duration::from_millis(150));
    drop(mgr); // must not panic or deadlock
}

#[test]
fn flush_manager_age_trigger_rotates_idle_memtable() {
    // R076b: a memtable with even one byte of data must roll over to SST
    // after `max_memtable_age_secs`, independent of the size threshold.
    // Without this, light-load workloads could sit in volatile memory
    // for hours; combined with the R076a oplog purge gate, oplog
    // retention would grow without bound waiting for size-based flush.
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let tree = trees.get(&Partition::Node).expect("node tree").clone();

    // One tiny write — nowhere near the 64MB size threshold.
    tree.insert(b"k", b"v", seqno.next());
    assert!(
        tree.active_memtable().size() > 0,
        "precondition: data lives in the active memtable"
    );

    // age trigger = 1 second, size & sealed thresholds effectively off.
    let mgr = FlushManager::start(
        &trees,
        Arc::clone(&gc_watermark),
        u64::MAX, // size: never triggers
        usize::MAX,
        1,
        20, // poll every 20ms
        1,  // age trigger after 1s
    )
    .expect("start FlushManager");

    // The age trigger needs both wall time AND a poll tick to fire. Give
    // it 2× the age budget; flush worker then handles the sealed memtable.
    let mut flushed = false;
    for _ in 0..150 {
        std::thread::sleep(Duration::from_millis(20));
        if tree.sealed_memtable_count() == 0 && tree.active_memtable().size() == 0 {
            flushed = true;
            break;
        }
    }

    drop(mgr);
    assert!(
        flushed,
        "age trigger should have rotated the lone memtable entry to SST"
    );
}

#[test]
fn flush_manager_age_zero_disables_time_based_trigger() {
    // max_memtable_age_secs == 0 must preserve the pre-R076b behavior:
    // size and sealed-count gates alone, no implicit time rotation.
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let tree = trees.get(&Partition::Node).expect("node tree").clone();

    tree.insert(b"k", b"v", seqno.next());

    let mgr = FlushManager::start(
        &trees,
        Arc::clone(&gc_watermark),
        u64::MAX, // size: never triggers
        usize::MAX,
        1,
        20,
        0, // age trigger DISABLED
    )
    .expect("start FlushManager");

    // Even after several poll cycles the active memtable must still hold
    // its byte — nothing else has been touched to push it out.
    std::thread::sleep(Duration::from_millis(400));
    let stayed = tree.active_memtable().size() > 0 && tree.sealed_memtable_count() == 0;
    drop(mgr);
    assert!(
        stayed,
        "with age trigger off, idle memtable must remain in memory"
    );
}
