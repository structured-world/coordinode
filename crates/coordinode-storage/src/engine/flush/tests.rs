use std::sync::Arc;
use std::sync::atomic::AtomicU64;
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

/// Start a manager with fresh wakes; returns the monitor's wake so a test can
/// act as the write path.
fn start(
    trees: &HashMap<Partition, lsm_tree::AnyTree>,
    gc_watermark: &Arc<AtomicU64>,
    threshold: u64,
    max_sealed: usize,
    workers: usize,
    max_age_secs: u64,
) -> (FlushManager, Arc<Wake>) {
    let wake = Arc::new(Wake::default());
    let mgr = FlushManager::start(
        trees,
        GcWatermarks::uniform(gc_watermark),
        threshold,
        max_sealed,
        workers,
        max_age_secs,
        Arc::clone(&wake),
        Arc::new(Wake::default()),
    )
    .expect("start FlushManager");
    (mgr, wake)
}

#[test]
fn flush_manager_starts_and_stops() {
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    // 64MB threshold (won't trigger in this test), age trigger disabled.
    let (mgr, _wake) = start(&trees, &gc_watermark, 64 * 1024 * 1024, 4, 1, 0);

    // Brief sleep to let threads spin up.
    std::thread::sleep(Duration::from_millis(120));

    // Drop manager: should join all threads cleanly without hanging, though
    // the monitor is parked with no deadline.
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

    // max_sealed=0 so ANY sealed count triggers flush; size never triggers,
    // age trigger disabled, isolating the sealed-count gate.
    let (mgr, _wake) = start(&trees, &gc_watermark, u64::MAX, 0, 1, 0);

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

    // 1 byte threshold, so it always triggers; high sealed count so only the
    // size trigger fires; age trigger disabled.
    let (mgr, _wake) = start(&trees, &gc_watermark, 1, 100, 1, 0);

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

/// A write that crosses the threshold after the monitor went to sleep wakes
/// it through the trigger: the monitor has no timer that would find the
/// write otherwise.
#[test]
fn a_write_through_the_trigger_wakes_the_sleeping_monitor() {
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));
    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let tree = trees.get(&Partition::Node).expect("node tree").clone();

    let (mgr, wake) = start(&trees, &gc_watermark, 8, 100, 1, 0);
    let trigger = FlushTrigger::new(wake, 8);
    // Let the monitor find nothing and park with no deadline.
    std::thread::sleep(Duration::from_millis(100));

    let (added, memtable) = tree.insert(b"key1", b"a value longer than eight bytes", seqno.next());
    trigger.wrote(added, memtable);

    let mut flushed = false;
    for _ in 0..50 {
        std::thread::sleep(Duration::from_millis(20));
        if tree.sealed_memtable_count() == 0 && tree.active_memtable().size() == 0 {
            flushed = true;
            break;
        }
    }
    drop(mgr);
    assert!(flushed, "the triggered write was not flushed");
}

#[test]
fn flush_manager_multiple_workers_no_panic() {
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    // 4 workers, every trigger on, to exercise concurrency.
    let (mgr, _wake) = start(&trees, &gc_watermark, 1, 0, 4, 0);

    std::thread::sleep(Duration::from_millis(150));
    drop(mgr); // must not panic or deadlock
}

#[test]
fn flush_manager_age_trigger_rotates_idle_memtable() {
    // A memtable with even one byte of data must roll over to SST
    // after `max_memtable_age_secs`, independent of the size threshold.
    // Without this, light-load workloads could sit in volatile memory
    // for hours; combined with the oplog purge gate, oplog
    // retention would grow without bound waiting for size-based flush.
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let tree = trees.get(&Partition::Node).expect("node tree").clone();

    // One tiny write, nowhere near the size threshold.
    tree.insert(b"k", b"v", seqno.next());
    assert!(
        tree.active_memtable().size() > 0,
        "precondition: data lives in the active memtable"
    );

    // age trigger = 1 second, size & sealed thresholds effectively off. The
    // monitor sleeps until the age deadline, with no other wakeup.
    let (mgr, _wake) = start(&trees, &gc_watermark, u64::MAX, usize::MAX, 1, 1);

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
    // max_memtable_age_secs == 0 disables the age trigger: size and
    // sealed-count gates alone, no implicit time rotation.
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let tree = trees.get(&Partition::Node).expect("node tree").clone();

    tree.insert(b"k", b"v", seqno.next());

    let (mgr, _wake) = start(&trees, &gc_watermark, u64::MAX, usize::MAX, 1, 0);

    // The active memtable must still hold its byte: nothing else has been
    // touched to push it out.
    std::thread::sleep(Duration::from_millis(400));
    let stayed = tree.active_memtable().size() > 0 && tree.sealed_memtable_count() == 0;
    drop(mgr);
    assert!(
        stayed,
        "with age trigger off, idle memtable must remain in memory"
    );
}
