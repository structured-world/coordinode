use std::sync::Arc;
use std::sync::atomic::AtomicU64;
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
        GcWatermarks::uniform(&gc_watermark),
        1,                       // 1 worker
        8,                       // l0_urgent_threshold
        64 * 1024 * 1024 * 1024, // debt_urgent_bytes
        Arc::new(std::sync::atomic::AtomicU8::new(0)),
        Arc::new(Wake::default()),
    )
    .expect("start CompactionScheduler");

    std::thread::sleep(Duration::from_millis(120));
    drop(sched);
}

#[test]
fn compaction_priority_rules() {
    const GIB: u64 = 1024 * 1024 * 1024;
    // Adj is High by default.
    assert_eq!(
        compaction_priority(Partition::Adj, 0, 8, 0, 64 * GIB),
        CompactionPriority::High,
    );
    // Blob is Low by default.
    assert_eq!(
        compaction_priority(Partition::Blob, 0, 8, 0, 64 * GIB),
        CompactionPriority::Low,
    );
    // Node is Normal by default.
    assert_eq!(
        compaction_priority(Partition::Node, 0, 8, 0, 64 * GIB),
        CompactionPriority::Normal,
    );
    // L0 above threshold → Urgent, regardless of partition.
    assert_eq!(
        compaction_priority(Partition::Node, 9, 8, 0, 64 * GIB),
        CompactionPriority::Urgent,
    );
    assert_eq!(
        compaction_priority(Partition::Adj, 9, 8, 0, 64 * GIB),
        CompactionPriority::Urgent,
    );
    assert_eq!(
        compaction_priority(Partition::Blob, 9, 8, 0, 64 * GIB),
        CompactionPriority::Urgent,
    );
    // L0 exactly at threshold → not Urgent.
    assert_eq!(
        compaction_priority(Partition::Node, 8, 8, 0, 64 * GIB),
        CompactionPriority::Normal,
    );
    // Pending-compaction byte debt at or above the urgent level → Urgent,
    // even with a short L0 (heavy-overwrite workloads accumulate debt in
    // the mid levels without a tall L0).
    assert_eq!(
        compaction_priority(Partition::Node, 0, 8, 64 * GIB, 64 * GIB),
        CompactionPriority::Urgent,
    );
    // Just below the debt level → the partition's default priority.
    assert_eq!(
        compaction_priority(Partition::Node, 0, 8, 64 * GIB - 1, 64 * GIB),
        CompactionPriority::Normal,
    );
    // A zero debt threshold disables the debt axis entirely.
    assert_eq!(
        compaction_priority(Partition::Node, 0, 8, u64::MAX, 0),
        CompactionPriority::Normal,
    );
}

/// A compaction that did work is counted for its partition, with its
/// duration. Two requests reach the worker, the second finds the merge done.
#[test]
fn a_compaction_that_did_work_is_counted_for_its_partition() {
    use metrics::{CounterFn, HistogramFn, Key, KeyName, Metadata, Recorder, SharedString, Unit};
    use std::sync::Mutex;

    /// A metric's name and its labels.
    type Recorded = (String, Vec<(String, String)>);
    /// What a recorder saw: each counter increment and histogram sample.
    #[derive(Default)]
    struct Seen(Mutex<Vec<Recorded>>);
    struct Handle(Key, Arc<Seen>);
    impl CounterFn for Handle {
        fn increment(&self, _: u64) {
            self.record();
        }
        fn absolute(&self, _: u64) {}
    }
    impl HistogramFn for Handle {
        fn record(&self, _: f64) {
            Handle::record(self);
        }
    }
    impl Handle {
        fn record(&self) {
            let labels = self
                .0
                .labels()
                .map(|l| (l.key().to_owned(), l.value().to_owned()))
                .collect();
            self.1
                .0
                .lock()
                .expect("seen")
                .push((self.0.name().to_owned(), labels));
        }
    }
    struct Capture(Arc<Seen>);
    impl Recorder for Capture {
        fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn register_counter(&self, key: &Key, _: &Metadata<'_>) -> metrics::Counter {
            metrics::Counter::from_arc(Arc::new(Handle(key.clone(), Arc::clone(&self.0))))
        }
        fn register_gauge(&self, _: &Key, _: &Metadata<'_>) -> metrics::Gauge {
            metrics::Gauge::noop()
        }
        fn register_histogram(&self, key: &Key, _: &Metadata<'_>) -> metrics::Histogram {
            metrics::Histogram::from_arc(Arc::new(Handle(key.clone(), Arc::clone(&self.0))))
        }
    }

    let (trees, _dir) = make_test_trees();
    let tree = trees.get(&Partition::Node).expect("node tree").clone();
    let seqno = lsm_tree::SequenceNumberCounter::default();
    // Overlapping L0 tables, enough for the leveled strategy to merge them.
    for round in 0_u64..8 {
        for i in 0_u64..20 {
            tree.insert(
                format!("key{i:04}").as_bytes(),
                round.to_le_bytes(),
                seqno.next(),
            );
        }
        tree.rotate_memtable();
        let lock = tree.get_flush_lock();
        tree.flush(&lock, 0).expect("flush");
    }

    let seen = Arc::new(Seen::default());
    let recorder = Capture(Arc::clone(&seen));
    let (sender, receiver) = flume::bounded(2);
    for _ in 0..2 {
        sender
            .send(CompactionRequest {
                tree: tree.clone(),
                partition: Partition::Node,
                priority: CompactionPriority::Urgent,
                gc_watermark: seqno.get(),
            })
            .expect("queue");
    }
    drop(sender);
    let compacted = Wake::default();
    metrics::with_local_recorder(&recorder, || compaction_worker_loop(receiver, &compacted));

    let seen = seen.0.lock().expect("seen");
    let node = |name: &str| {
        seen.iter()
            .filter(|(n, labels)| {
                n == name && labels.iter().any(|(k, v)| k == "partition" && v == "node")
            })
            .count()
    };
    let total = node("coordinode_storage_compaction_total");
    assert!(total >= 1, "a compaction that merged the tables: {seen:?}");
    assert_eq!(
        node("coordinode_storage_compaction_duration_seconds"),
        total
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
        GcWatermarks::uniform(&gc_watermark),
        1,
        8,
        64 * 1024 * 1024 * 1024,
        Arc::new(std::sync::atomic::AtomicU8::new(0)),
        Arc::new(Wake::default()),
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
        GcWatermarks::uniform(&gc_watermark),
        4, // 4 workers
        8,
        64 * 1024 * 1024 * 1024,
        Arc::new(std::sync::atomic::AtomicU8::new(0)),
        Arc::new(Wake::default()),
    )
    .expect("start CompactionScheduler");

    std::thread::sleep(Duration::from_millis(150));
    drop(sched);
}
