use crate::columnar::{write_columnar_rows, ColumnarRow};
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;

fn persistent_config(dir: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

#[test]
fn create_write_and_get_columnar_table() {
    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();

    let tree = engine.create_columnar_table("trades").unwrap();
    write_columnar_rows(
        &tree,
        &[
            ColumnarRow {
                key: b"pk1",
                value: b"r1",
            },
            ColumnarRow {
                key: b"pk2",
                value: b"r2",
            },
        ],
    )
    .unwrap();

    assert!(engine.columnar_table_tree("trades").is_some());
    assert!(engine.columnar_table_tree("absent").is_none());
}

#[test]
fn columnar_table_recovers_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();
        engine.create_columnar_table("orders").unwrap();
    }
    // A fresh engine over the same directory re-opens the on-disk table tree.
    let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();
    assert!(engine.columnar_table_tree("orders").is_some());
}

#[test]
fn drop_columnar_table_removes_it() {
    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();
    engine.create_columnar_table("scratch").unwrap();
    assert!(engine.columnar_table_tree("scratch").is_some());

    engine.drop_columnar_table("scratch").unwrap();
    assert!(engine.columnar_table_tree("scratch").is_none());
    // Idempotent.
    engine.drop_columnar_table("scratch").unwrap();
}
