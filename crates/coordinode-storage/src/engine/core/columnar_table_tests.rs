use crate::columnar::{ColumnarRow, write_columnar_rows};
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

/// The tables of the flush manager: a columnar table whose memtable grows
/// past the flush threshold is flushed in the background, like a partition,
/// with no statement asking for it.
#[test]
fn the_flush_manager_flushes_a_columnar_table_past_its_threshold() {
    use lsm_tree::AbstractTree;

    let dir = tempfile::tempdir().unwrap();
    let mut config = persistent_config(dir.path());
    config.max_write_buffer_bytes = 16 * 1024;
    let engine = StorageEngine::open(&config).unwrap();
    for i in 0..64u32 {
        engine
            .columnar_insert(
                "metrics",
                format!("row:{i:04}").into_bytes(),
                vec![b'x'; 1024],
                engine.next_seqno(),
            )
            .unwrap();
    }
    let tree = engine.columnar_table_tree("metrics").unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while tree.table_count() == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "64 KiB of rows past a 16 KiB threshold were never flushed"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(engine.columnar_scan("metrics", u64::MAX).unwrap().len(), 64);
}

/// A statement's end makes a columnar table's rows durable: a journal holds
/// them already, so nothing is flushed and the scan reads them from the
/// memtable; without a journal the table is flushed then.
#[test]
fn a_statement_flushes_a_columnar_table_only_without_a_journal() {
    use lsm_tree::AbstractTree;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let journaled =
        StorageEngine::open_embedded(&persistent_config(dir.path()), Arc::clone(&oracle)).unwrap();
    journaled
        .columnar_insert(
            "trades",
            b"row:1".to_vec(),
            b"v".to_vec(),
            oracle.next().as_raw(),
        )
        .unwrap();
    // A new table's coverage record is on disk from its creation.
    let tree = journaled.columnar_table_tree("trades").unwrap();
    let tables = tree.table_count();
    journaled.finish_columnar_statement("trades").unwrap();
    assert_eq!(
        tree.table_count(),
        tables,
        "a journaled statement flushed its table"
    );
    assert_eq!(
        journaled.columnar_scan("trades", u64::MAX).unwrap(),
        vec![(b"row:1".to_vec(), b"v".to_vec())]
    );
    drop(journaled);

    let plain_dir = tempfile::tempdir().unwrap();
    let plain = StorageEngine::open(&persistent_config(plain_dir.path())).unwrap();
    plain
        .columnar_insert(
            "trades",
            b"row:1".to_vec(),
            b"v".to_vec(),
            plain.next_seqno(),
        )
        .unwrap();
    let tree = plain.columnar_table_tree("trades").unwrap();
    let tables = tree.table_count();
    plain.finish_columnar_statement("trades").unwrap();
    assert!(
        tree.table_count() > tables,
        "a statement without a journal left its rows in the memtable"
    );
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
