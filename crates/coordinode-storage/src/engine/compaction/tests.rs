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
        CompactionStrategyKind::Leveled,
        None,
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
                source: CompactionSource::Partition(Partition::Node),
                strategy: CompactionStrategyKind::Leveled.for_tree(),
                priority: CompactionPriority::Urgent,
                gc_watermark: seqno.get(),
                life: None,
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
        CompactionStrategyKind::Leveled,
        None,
    )
    .expect("start CompactionScheduler");

    std::thread::sleep(Duration::from_millis(300));
    drop(sched); // must not panic or deadlock
}

/// Under the planned strategy the scheduler compacts too. A planned tree
/// keeps small runs in its top level until a unit's worth has gathered, so a
/// stream of tiny flushes is sent down by the L0 table count instead: past
/// 500 tables, the guard it shares with the leveled strategy.
#[test]
fn the_planned_strategy_compacts_a_tree() {
    let (trees, _dir) = make_test_trees();
    let tree = trees.get(&Partition::Node).expect("node tree").clone();
    let seqno = lsm_tree::SequenceNumberCounter::default();
    for round in 0_u64..510 {
        tree.insert(b"key", round.to_le_bytes(), seqno.next());
        tree.rotate_memtable();
        let lock = tree.get_flush_lock();
        tree.flush(&lock, 0).expect("flush");
    }
    let runs = tree.l0_run_count();
    let gc_watermark = Arc::new(AtomicU64::new(0));
    let sched = CompactionScheduler::start(
        &trees,
        GcWatermarks::uniform(&gc_watermark),
        1,
        8,
        64 * 1024 * 1024 * 1024,
        Arc::new(std::sync::atomic::AtomicU8::new(0)),
        Arc::new(Wake::default()),
        CompactionStrategyKind::Planned,
        None,
    )
    .expect("start CompactionScheduler");

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while tree.l0_run_count() >= runs {
        assert!(
            std::time::Instant::now() < deadline,
            "{runs} overlapping L0 runs were never compacted"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(sched);
}

/// The monitor judges each table on its own and latches its verdict where
/// that table's writes read it, not into the partitions' verdict, and the
/// tables reach the metrics as one series under `partition="tables"` (the
/// worst tier) whatever their number, with no label per table.
#[test]
fn the_monitor_judges_each_table_into_its_own_verdict_and_one_series() {
    use metrics::{GaugeFn, Key, KeyName, Metadata, Recorder, SharedString, Unit};
    use std::sync::Mutex;

    /// Each gauge's labels and last value, by name.
    type Gauges = Mutex<Vec<(String, Vec<(String, String)>, f64)>>;
    struct Handle(Key, Arc<Gauges>);
    impl GaugeFn for Handle {
        fn increment(&self, _: f64) {}
        fn decrement(&self, _: f64) {}
        fn set(&self, value: f64) {
            let labels = self
                .0
                .labels()
                .map(|l| (l.key().to_owned(), l.value().to_owned()))
                .collect();
            self.1
                .lock()
                .expect("gauges")
                .push((self.0.name().to_owned(), labels, value));
        }
    }
    struct Capture(Arc<Gauges>);
    impl Recorder for Capture {
        fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn register_counter(&self, _: &Key, _: &Metadata<'_>) -> metrics::Counter {
            metrics::Counter::noop()
        }
        fn register_gauge(&self, key: &Key, _: &Metadata<'_>) -> metrics::Gauge {
            metrics::Gauge::from_arc(Arc::new(Handle(key.clone(), Arc::clone(&self.0))))
        }
        fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> metrics::Histogram {
            metrics::Histogram::noop()
        }
    }

    let (trees, dir) = make_test_trees();
    let seqno: lsm_tree::SharedSequenceNumberGenerator =
        Arc::new(lsm_tree::SequenceNumberCounter::default());
    let table_metrics = Arc::new(crate::engine::owned_metrics::TableMetrics::default());
    let table = |name: &str, runs: u64, alive: bool| {
        let tree = lsm_tree::Config::new_with_generators(
            dir.path().join(name),
            Arc::clone(&seqno),
            Arc::clone(&seqno),
        )
        .open()
        .expect("open table tree");
        if let lsm_tree::AnyTree::Standard(t) = &tree {
            t.update_runtime_config(|cfg| {
                cfg.backpressure = lsm_tree::BackpressureThresholds {
                    l0_stop: Some(2),
                    ..lsm_tree::BackpressureThresholds::OFF
                };
            })
            .expect("thresholds");
        }
        // Overlapping runs: each rewrites the same key.
        for round in 0..runs {
            tree.insert(b"key", round.to_le_bytes(), seqno.next());
            tree.rotate_memtable();
            let lock = tree.get_flush_lock();
            tree.flush(&lock, 0).expect("flush");
        }
        TableTree {
            id: name.to_string(),
            tree,
            pressure: Arc::new(std::sync::atomic::AtomicU8::new(0)),
            alive: Arc::new(parking_lot::RwLock::new(alive)),
            metrics: Arc::clone(&table_metrics),
        }
    };
    // `dropped` was listed before its drop took its life.
    let tables = vec![
        table("behind", 3, true),
        table("idle", 0, true),
        table("dropped", 3, false),
    ];
    let (behind, idle) = (
        Arc::clone(&tables[0].pressure),
        Arc::clone(&tables[1].pressure),
    );

    let gauges = Arc::new(Gauges::default());
    let (sender, _receiver) = flume::bounded(64);
    let shutdown = Arc::new(AtomicBool::new(false));
    let write_pressure = Arc::new(std::sync::atomic::AtomicU8::new(0));
    let wake = Arc::new(Wake::default());
    let monitor = {
        let recorder = Capture(Arc::clone(&gauges));
        let monitored = trees
            .iter()
            .map(|(&p, t)| MonitoredTree {
                tree: t.clone(),
                source: CompactionSource::Partition(p),
                strategy: CompactionStrategyKind::Leveled.for_tree(),
                table: None,
            })
            .collect();
        let cfg = MonitorConfig {
            sender,
            gc_watermarks: GcWatermarks::uniform(&Arc::new(AtomicU64::new(0))),
            l0_urgent_threshold: 8,
            debt_urgent_bytes: 64 * 1024 * 1024 * 1024,
            write_pressure: Arc::clone(&write_pressure),
            shutdown: Arc::clone(&shutdown),
            strategy: CompactionStrategyKind::Leveled,
            tables: Some(Box::new(move || tables.clone())),
        };
        let wake = Arc::clone(&wake);
        std::thread::spawn(move || {
            metrics::with_local_recorder(&recorder, || {
                compaction_monitor_loop(monitored, cfg, &wake)
            });
        })
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while behind.load(Ordering::Relaxed) != 2 {
        assert!(
            std::time::Instant::now() < deadline,
            "the table was never judged"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    shutdown.store(true, Ordering::Relaxed);
    wake.notify();
    monitor.join().expect("monitor");

    assert_eq!(
        idle.load(Ordering::Relaxed),
        0,
        "a table is judged on its own tree"
    );
    assert_eq!(
        write_pressure.load(Ordering::Relaxed),
        0,
        "a table's verdict is not the partitions'"
    );
    let gauges = gauges.lock().expect("gauges");
    let last = |name: &str, partition: &str| {
        gauges
            .iter()
            .rev()
            .find(|(n, labels, _)| {
                n == name
                    && labels
                        .iter()
                        .any(|(k, v)| k == "partition" && v == partition)
            })
            .map(|(_, _, value)| *value)
    };
    assert_eq!(
        last("coordinode_storage_backpressure_tier", "tables"),
        Some(2.0)
    );
    assert!(last("coordinode_storage_compaction_debt_bytes", "tables").is_some());
    assert_eq!(
        last("coordinode_storage_backpressure_tier", "node"),
        Some(0.0)
    );
    assert!(
        gauges
            .iter()
            .all(|(_, labels, _)| labels.iter().all(|(k, _)| k == "partition")),
        "the facade series carry no label per table: {gauges:?}"
    );
    // Each live table has its own series; a dropped one has none.
    let mut rendered = String::new();
    table_metrics.render(&mut rendered);
    assert!(
        rendered.contains("coordinode_storage_table_backpressure_tier{table=\"behind\"} 2\n"),
        "{rendered}"
    );
    assert!(
        rendered.contains("coordinode_storage_table_backpressure_tier{table=\"idle\"} 0\n"),
        "{rendered}"
    );
    assert!(!rendered.contains("dropped"), "{rendered}");
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
        CompactionStrategyKind::Leveled,
        None,
    )
    .expect("start CompactionScheduler");

    std::thread::sleep(Duration::from_millis(150));
    drop(sched);
}
