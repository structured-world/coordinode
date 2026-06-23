use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use super::*;

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
fn compaction_scheduler_starts_and_stops() {
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    let sched = CompactionScheduler::start(
        &trees,
        Arc::clone(&gc_watermark),
        1,  // 1 worker
        8,  // l0_urgent_threshold
        50, // poll interval ms
    )
    .expect("start CompactionScheduler");

    std::thread::sleep(Duration::from_millis(120));
    drop(sched);
}

#[test]
fn compaction_priority_rules() {
    // Adj is High by default.
    assert_eq!(
        compaction_priority(Partition::Adj, 0, 8),
        CompactionPriority::High,
    );
    // Blob is Low by default.
    assert_eq!(
        compaction_priority(Partition::Blob, 0, 8),
        CompactionPriority::Low,
    );
    // Node is Normal by default.
    assert_eq!(
        compaction_priority(Partition::Node, 0, 8),
        CompactionPriority::Normal,
    );
    // L0 above threshold → Urgent, regardless of partition.
    assert_eq!(
        compaction_priority(Partition::Node, 9, 8),
        CompactionPriority::Urgent,
    );
    assert_eq!(
        compaction_priority(Partition::Adj, 9, 8),
        CompactionPriority::Urgent,
    );
    assert_eq!(
        compaction_priority(Partition::Blob, 9, 8),
        CompactionPriority::Urgent,
    );
    // L0 exactly at threshold → not Urgent.
    assert_eq!(
        compaction_priority(Partition::Node, 8, 8),
        CompactionPriority::Normal,
    );
}

#[test]
fn compaction_priority_ordering() {
    // Urgent < High < Normal < Low (lower value = higher priority via Ord).
    assert!(CompactionPriority::Urgent < CompactionPriority::High);
    assert!(CompactionPriority::High < CompactionPriority::Normal);
    assert!(CompactionPriority::Normal < CompactionPriority::Low);
}

#[test]
fn compaction_scheduler_no_panic_with_l0_data() {
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let tree = trees.get(&Partition::Node).expect("node tree").clone();

    // Write + flush to produce an L0 SST file.
    for i in 0_u64..20 {
        tree.insert(
            format!("key{i:04}").as_bytes(),
            format!("value{i}").as_bytes(),
            seqno.next(),
        );
    }
    tree.rotate_memtable();
    let lock = tree.get_flush_lock();
    let _ = tree.flush(&lock, 0);

    let sched = CompactionScheduler::start(
        &trees,
        Arc::clone(&gc_watermark),
        1,
        8,
        20, // fast poll for test
    )
    .expect("start CompactionScheduler");

    std::thread::sleep(Duration::from_millis(300));
    drop(sched); // must not panic or deadlock
}

#[test]
fn compaction_scheduler_multiple_workers_no_panic() {
    let (trees, _dir) = make_test_trees();
    let gc_watermark = Arc::new(AtomicU64::new(0));

    let sched = CompactionScheduler::start(
        &trees,
        Arc::clone(&gc_watermark),
        4, // 4 workers
        8,
        10, // fast poll
    )
    .expect("start CompactionScheduler");

    std::thread::sleep(Duration::from_millis(150));
    drop(sched);
}
