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
