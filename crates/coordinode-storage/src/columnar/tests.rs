use super::*;
use lsm_tree::Config;

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
    let seqno: SharedSequenceNumberGenerator =
        std::sync::Arc::new(lsm_tree::SequenceNumberCounter::default());
    let cache = std::sync::Arc::new(lsm_tree::Cache::with_capacity_bytes(8 * 1024 * 1024));
    ColumnarTableRegistry::open(base.to_path_buf(), seqno, cache).unwrap()
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
