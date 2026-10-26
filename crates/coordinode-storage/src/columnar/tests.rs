use super::*;
use lsm_tree::Config;
use lsm_tree::SharedSequenceNumberGenerator;
use lsm_tree::fs::SyncMode;

/// The engine's defaults for the table trees under `roots` (the first the
/// primary) on `fs`, syncing at `sync`.
fn settings(roots: Vec<PathBuf>, fs: Arc<dyn Fs>, sync: SyncMode) -> TableTreeSettings {
    let seqno: SharedSequenceNumberGenerator =
        std::sync::Arc::new(lsm_tree::SequenceNumberCounter::default());
    let primary = roots[0].clone();
    TableTreeSettings {
        config: Box::new(move |table_id| {
            Config::new_with_generators(primary.join(table_id), seqno.clone(), seqno.clone())
                .with_shared_fs(Arc::clone(&fs))
                .sync_mode(sync)
        }),
        roots,
        cache: std::sync::Arc::new(lsm_tree::Cache::with_capacity_bytes(8 * 1024 * 1024)),
        maintenance: std::sync::Arc::new(RateLimiter::new(0)),
        write_buffer: std::sync::Arc::new(lsm_tree::WriteBufferBudget::default()),
        capacity: std::sync::Arc::new(lsm_tree::CapacityPool::default()),
        column_encoding: TableTreeSettings::column_encoding_from(3),
        backpressure: lsm_tree::BackpressureThresholds::default(),
        metrics: Arc::default(),
    }
}

fn open_columnar_tree() -> (tempfile::TempDir, AnyTree) {
    let dir = tempfile::tempdir().unwrap();
    let tree = Config::new(dir.path(), Default::default(), Default::default())
        .open()
        .unwrap();
    enable_columnar(&tree).unwrap();
    (dir, tree)
}

#[test]
fn batch_rows_round_trip_in_memory() {
    let entries = vec![
        InternalValue::from_components(b"k1".to_vec(), b"v1".to_vec(), 0, ValueType::Value),
        InternalValue::from_components(b"k2".to_vec(), b"v2".to_vec(), 0, ValueType::Value),
    ];
    let batch = entries_to_column_batch(&entries).unwrap();
    let rows = columnar_batch_rows(&batch).unwrap();
    assert_eq!(
        rows,
        vec![
            (b"k1".to_vec(), b"v1".to_vec()),
            (b"k2".to_vec(), b"v2".to_vec()),
        ]
    );
}

#[test]
fn write_columnar_rows_succeeds_on_columnar_tree() {
    let (_dir, tree) = open_columnar_tree();
    let rows = vec![
        ColumnarRow {
            key: b"k1",
            value: b"v1",
        },
        ColumnarRow {
            key: b"k2",
            value: b"v2",
        },
    ];
    write_columnar_rows(&tree, &rows).unwrap();
}

fn new_registry(base: &std::path::Path) -> ColumnarTableRegistry {
    let fs: std::sync::Arc<dyn lsm_tree::fs::Fs> = std::sync::Arc::new(lsm_tree::fs::StdFs);
    let (registry, repairs) = ColumnarTableRegistry::open(
        Arc::clone(&fs),
        settings(vec![base.to_path_buf()], fs, SyncMode::Full),
    )
    .unwrap();
    assert!(repairs.is_empty(), "{repairs:?}");
    registry
}

#[test]
fn registry_create_get_and_list() {
    let dir = tempfile::tempdir().unwrap();
    let reg = new_registry(dir.path());
    assert!(reg.table_ids().is_empty());

    let tree = reg.create_or_open("trades").unwrap();
    write_columnar_rows(
        &tree,
        &[ColumnarRow {
            key: b"pk1",
            value: b"row1",
        }],
    )
    .unwrap();

    // Second call returns the same registered table.
    assert!(reg.get("trades").is_some());
    assert_eq!(reg.table_ids(), vec!["trades".to_string()]);
    // create_or_open is idempotent for an existing table.
    reg.create_or_open("trades").unwrap();
    assert_eq!(reg.table_ids(), vec!["trades".to_string()]);
}

#[test]
fn registry_drop_removes_table() {
    let dir = tempfile::tempdir().unwrap();
    let reg = new_registry(dir.path());
    reg.create_or_open("t").unwrap();
    assert!(reg.get("t").is_some());

    reg.drop_table("t").unwrap();
    assert!(reg.get("t").is_none());
    assert!(reg.table_ids().is_empty());
    assert!(!dir.path().join("t").exists());
    // Dropping an unknown table is a no-op.
    reg.drop_table("missing").unwrap();
}

/// A drop waits for the maintenance running on the table's tree before its
/// directory goes, and maintenance that reaches the tree after the drop finds
/// it gone and does nothing: no flush or compaction writes into a directory
/// being removed.
#[test]
fn a_drop_waits_for_running_maintenance_and_turns_away_the_rest() {
    use crate::engine::flush::while_alive;

    let dir = tempfile::tempdir().unwrap();
    let reg = Arc::new(new_registry(dir.path()));
    reg.create_or_open("t").unwrap();
    let table = reg.trees().pop().expect("the table");
    let (started_tx, started) = std::sync::mpsc::channel();
    let (finish, finish_rx) = std::sync::mpsc::channel::<()>();
    let life = Arc::clone(&table.alive);
    let maintenance = std::thread::spawn(move || {
        while_alive(Some(&life), || {
            started_tx.send(()).unwrap();
            finish_rx.recv().unwrap();
        })
    });
    started.recv().unwrap();

    let dropper = {
        let reg = Arc::clone(&reg);
        std::thread::spawn(move || reg.drop_table("t"))
    };
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert!(
        !dropper.is_finished(),
        "the drop did not wait for the maintenance"
    );
    assert!(dir.path().join("t").exists());
    finish.send(()).unwrap();
    assert_eq!(maintenance.join().unwrap(), Some(()));
    dropper.join().unwrap().unwrap();
    assert!(!dir.path().join("t").exists());

    assert_eq!(
        while_alive(Some(&table.alive), || ()),
        None,
        "maintenance queued before the drop ran after it"
    );
}

/// Hold `table`'s life the way running maintenance does, until the returned
/// sender is used (or dropped).
fn hold_life(
    table: &crate::engine::flush::TableTree,
) -> (
    std::sync::mpsc::Sender<()>,
    std::thread::JoinHandle<Option<()>>,
) {
    use crate::engine::flush::while_alive;

    let (started_tx, started) = std::sync::mpsc::channel();
    let (finish, finish_rx) = std::sync::mpsc::channel::<()>();
    let life = Arc::clone(&table.alive);
    let maintenance = std::thread::spawn(move || {
        while_alive(Some(&life), || {
            started_tx.send(()).unwrap();
            // The test releases the life by sending or by dropping `finish`.
            let _released = finish_rx.recv();
        })
    });
    started.recv().unwrap();
    (finish, maintenance)
}

/// Wait until the registry's retirement state of `name` satisfies `reached`.
fn wait_for_state(
    reg: &ColumnarTableRegistry,
    name: &str,
    what: &str,
    reached: impl Fn(bool, usize) -> bool,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let (running, waiting) = reg.retirement_state(name);
        if reached(running, waiting) {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "never reached: {what}"
        );
        std::thread::yield_now();
    }
}

/// A table created again under a name whose drop is still running waits for
/// the drop to finish: the drop's removal of the old table's directory and
/// series, by name, never reaches the new table.
#[test]
fn a_recreate_waits_for_the_drop_of_its_name() {
    let dir = tempfile::tempdir().unwrap();
    let reg = Arc::new(new_registry(dir.path()));
    reg.create_or_open("t").unwrap();
    let table = reg.trees().pop().expect("the table");
    let (finish, maintenance) = hold_life(&table);

    // The drop has published the name as retiring and waits for the life.
    let dropper = {
        let reg = Arc::clone(&reg);
        std::thread::spawn(move || reg.drop_table("t"))
    };
    wait_for_state(&reg, "t", "the drop running", |running, _| running);
    // The re-create waits on the drop.
    let creator = {
        let reg = Arc::clone(&reg);
        std::thread::spawn(move || reg.create_or_open("t").map(|_| ()))
    };
    wait_for_state(&reg, "t", "the re-create waiting", |running, waiting| {
        running && waiting == 1
    });
    assert!(reg.get("t").is_none(), "nothing is served under the name");

    finish.send(()).unwrap();
    maintenance.join().unwrap();
    dropper.join().unwrap().unwrap();
    creator.join().unwrap().unwrap();
    let new = reg.trees().pop().expect("the new table");
    assert!(
        !Arc::ptr_eq(&new.alive, &table.alive),
        "the new table has a life of its own"
    );
    assert!(*new.alive.read(), "the new table lives");
    assert!(
        dir.path().join("t").exists(),
        "the drop removed the new table's directory"
    );
}

/// Work on a table through `with_live` (a flush, a checkpoint capture) holds
/// the table's drop off its folder until it is done, and work asked for once
/// the drop has taken the table does not run at all.
#[test]
fn work_on_a_live_table_holds_its_drop_and_none_runs_after() {
    let dir = tempfile::tempdir().unwrap();
    let reg = Arc::new(new_registry(dir.path()));
    reg.create_or_open("t").unwrap();

    let (started_tx, started) = std::sync::mpsc::channel();
    let (finish, finish_rx) = std::sync::mpsc::channel::<()>();
    let worker = {
        let reg = Arc::clone(&reg);
        std::thread::spawn(move || {
            reg.with_live("t", |_tree| {
                started_tx.send(()).unwrap();
                let _released = finish_rx.recv();
            })
        })
    };
    started.recv().unwrap();
    let dropper = {
        let reg = Arc::clone(&reg);
        std::thread::spawn(move || reg.drop_table("t"))
    };
    wait_for_state(&reg, "t", "the drop running", |running, _| running);
    assert!(
        dir.path().join("t").exists(),
        "the drop removed the folder under running work"
    );
    finish.send(()).unwrap();
    assert_eq!(worker.join().unwrap(), Some(()));
    dropper.join().unwrap().unwrap();
    assert!(!dir.path().join("t").exists());

    assert_eq!(
        reg.with_live("t", |_tree| ()),
        None,
        "no work runs on a dropped table"
    );
}

/// The series rendered for `table`, as `(tier, debt)` lines.
fn series_of(reg: &ColumnarTableRegistry, table: &str) -> Vec<String> {
    let mut out = String::new();
    reg.settings.metrics.render(&mut out);
    out.lines()
        .filter(|line| line.contains(&format!("{{table=\"{table}\"}}")))
        .map(str::to_owned)
        .collect()
}

/// Two drops of one table and a create under its name meet, in the order
/// that matters: the first drop finishes, the create registers the new table
/// and publishes its series, and only then does the second drop, which
/// waited on the first, go on. It drops the same, old table, so it returns
/// and the new table, its directory and its series survive; the old table's
/// series went with the first drop.
#[test]
fn a_second_drop_of_a_table_never_drops_the_one_created_after() {
    use crate::engine::flush::while_alive;

    let dir = tempfile::tempdir().unwrap();
    let reg = Arc::new(new_registry(dir.path()));
    reg.create_or_open("t").unwrap();
    let old = reg.trees().pop().expect("the table");
    while_alive(Some(&old.alive), || reg.settings.metrics.set("t", 2, 100));
    assert_eq!(series_of(&reg, "t").len(), 2, "the old table's series");
    let (finish, maintenance) = hold_life(&old);

    let first = {
        let reg = Arc::clone(&reg);
        std::thread::spawn(move || reg.drop_table("t"))
    };
    wait_for_state(&reg, "t", "the first drop running", |running, _| running);
    // The second drop stops, once the first is done, until the test lets it on.
    let (paused_tx, paused) = std::sync::mpsc::channel::<()>();
    let (resume, resume_rx) = std::sync::mpsc::channel::<()>();
    *reg.pause_after_drop_wait.lock().unwrap() = Some(Box::new(move || {
        paused_tx.send(()).unwrap();
        resume_rx.recv().unwrap();
    }));
    let second = {
        let reg = Arc::clone(&reg);
        std::thread::spawn(move || reg.drop_table("t"))
    };
    wait_for_state(&reg, "t", "the second drop waiting", |_, waiting| {
        waiting == 1
    });

    finish.send(()).unwrap();
    maintenance.join().unwrap();
    first.join().unwrap().unwrap();
    paused.recv().unwrap();
    assert!(
        series_of(&reg, "t").is_empty(),
        "the old table's series went with its drop"
    );

    // The new table registers and its monitor publishes its series while the
    // second drop is held after its wait.
    reg.create_or_open("t").unwrap();
    let new = reg.trees().pop().expect("the new table");
    assert!(!Arc::ptr_eq(&new.alive, &old.alive));
    while_alive(Some(&new.alive), || reg.settings.metrics.set("t", 0, 7));
    resume.send(()).unwrap();
    second.join().unwrap().unwrap();

    assert!(reg.get("t").is_some(), "the new table is still registered");
    assert!(*new.alive.read(), "the new table lives");
    assert!(dir.path().join("t").exists(), "its directory is in place");
    assert_eq!(
        series_of(&reg, "t"),
        vec![
            "coordinode_storage_table_backpressure_tier{table=\"t\"} 0".to_owned(),
            "coordinode_storage_table_compaction_debt_bytes{table=\"t\"} 7".to_owned(),
        ],
        "its series is in place"
    );
}

/// A drop that fails before its directories are gone leaves the name refused:
/// a create would open what is left of the dropped table. A second drop
/// finishes the removal, and the name is free again.
#[test]
fn a_name_whose_drop_failed_is_refused_until_a_drop_finishes() {
    use lsm_tree::fs::{Fault, FaultInjector, FaultOp, FaultRule};

    let dir = tempfile::tempdir().unwrap();
    let injector = Arc::new(FaultInjector::new());
    let fs: Arc<dyn Fs> = Arc::new(lsm_tree::fs::FaultFs::with_injector(
        lsm_tree::fs::StdFs,
        Arc::clone(&injector),
    ));
    let reg = ColumnarTableRegistry::open(
        Arc::clone(&fs),
        settings(vec![dir.path().to_path_buf()], fs, SyncMode::Normal),
    )
    .unwrap()
    .0;
    reg.create_or_open("t").unwrap();

    injector.arm(
        FaultRule::new(
            FaultOp::RemoveDirAll,
            Fault::Error(lsm_tree::io::ErrorKind::PermissionDenied),
        )
        .once(),
    );
    assert!(reg.drop_table("t").is_err(), "the armed removal fails");
    assert!(reg.get("t").is_none(), "the dropped table is not served");
    assert!(
        reg.create_or_open("t").is_err(),
        "a create over the unfinished drop is refused"
    );

    reg.drop_table("t").unwrap();
    assert!(!dir.path().join("t").exists());
    reg.create_or_open("t").unwrap();
    assert!(reg.get("t").is_some());
}

#[test]
fn registry_recovers_tables_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let reg = new_registry(dir.path());
        reg.create_or_open("orders").unwrap();
        reg.create_or_open("trades").unwrap();
    }
    // A fresh registry over the same base dir re-opens the on-disk tables.
    let reopened = new_registry(dir.path());
    assert_eq!(
        reopened.table_ids(),
        vec!["orders".to_string(), "trades".to_string()]
    );
}

/// A columnar table's tree syncs at the registry's mode, the same as the
/// engine's partition trees, both on first creation and on reopen.
#[test]
fn table_trees_sync_at_the_registry_mode() {
    for mode in [SyncMode::Normal, SyncMode::Full] {
        let dir = tempfile::tempdir().unwrap();
        let injector = std::sync::Arc::new(lsm_tree::fs::FaultInjector::new());
        let open = || {
            let fs: std::sync::Arc<dyn lsm_tree::fs::Fs> =
                std::sync::Arc::new(lsm_tree::fs::FaultFs::with_injector(
                    lsm_tree::fs::StdFs,
                    std::sync::Arc::clone(&injector),
                ));
            ColumnarTableRegistry::open(
                Arc::clone(&fs),
                settings(vec![dir.path().to_path_buf()], fs, mode),
            )
            .unwrap()
            .0
        };

        let tree = open().create_or_open("orders").unwrap();
        let rows = [ColumnarRow {
            key: b"k1",
            value: b"v1",
        }];
        write_columnar_rows(&tree, &rows).unwrap();
        drop(tree);
        let reopened = open().get("orders").expect("reopened table");
        let rows = [ColumnarRow {
            key: b"k2",
            value: b"v2",
        }];
        write_columnar_rows(&reopened, &rows).unwrap();

        let modes = injector.sync_modes_for("orders");
        assert!(!modes.is_empty(), "{mode:?}: nothing was synced");
        assert!(
            modes.iter().all(|m| *m == mode),
            "{mode:?}: table files synced at {modes:?}"
        );
    }
}
