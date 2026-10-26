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
