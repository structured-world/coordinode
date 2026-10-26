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

/// The tables of the compaction scheduler: a columnar table whose flushes
/// pile up overlapping runs in L0 is compacted in the background, like a
/// partition, and every row still reads.
#[test]
fn the_compaction_scheduler_compacts_a_columnar_table() {
    use lsm_tree::AbstractTree;

    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();
    for round in 0..12u32 {
        for i in 0..16u32 {
            engine
                .columnar_insert(
                    "metrics",
                    format!("row:{i:04}").into_bytes(),
                    round.to_le_bytes().to_vec(),
                    engine.next_seqno(),
                )
                .unwrap();
        }
        engine.flush_columnar_table("metrics").unwrap();
    }
    let tree = engine.columnar_table_tree("metrics").unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while tree.l0_run_count() >= 12 {
        assert!(
            std::time::Instant::now() < deadline,
            "twelve overlapping L0 runs were never compacted"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let rows = engine.columnar_scan("metrics", u64::MAX).unwrap();
    assert_eq!(rows.len(), 16);
    assert!(rows.iter().all(|(_, v)| v == &11u32.to_le_bytes().to_vec()));
}

/// A table's write-pressure series lives exactly as long as the table: the
/// compaction monitor publishes it once the table is judged, `DROP TABLE`
/// takes it away at once, and a table created again under the name gets its
/// own once judged.
#[test]
fn a_dropped_table_takes_its_series_with_it() {
    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();
    let rendered = || {
        let mut out = String::new();
        engine.table_metrics().render(&mut out);
        out
    };
    let write_and_flush = || {
        engine
            .columnar_insert(
                "trades",
                b"row:0001".to_vec(),
                b"v".to_vec(),
                engine.next_seqno(),
            )
            .unwrap();
        // A flush wakes the compaction monitor, which judges the table.
        engine.flush_columnar_table("trades").unwrap();
    };
    let wait_for_series = || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !rendered().contains("table=\"trades\"") {
            assert!(
                std::time::Instant::now() < deadline,
                "the table was never judged"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    };

    write_and_flush();
    wait_for_series();
    engine.drop_columnar_table("trades").unwrap();
    assert!(!rendered().contains("trades"), "{}", rendered());

    write_and_flush();
    wait_for_series();
}

/// A flush does not change what a table reads at any snapshot: a segment
/// holding several versions of a key answers an older snapshot with the
/// version that snapshot sees, not with nothing.
#[test]
fn a_flush_keeps_every_snapshots_read() {
    use lsm_tree::AbstractTree as _;

    let dir = tempfile::tempdir().unwrap();
    let mut config = persistent_config(dir.path());
    config.compaction_workers = 0;
    let engine = StorageEngine::open(&config).unwrap();
    let mut snapshots = Vec::new();
    for (key, value) in [
        (&b"row:a"[..], &b"a-v1"[..]),
        (b"row:b", b"b-live"),
        (b"row:c", b"c-live"),
    ] {
        engine
            .columnar_insert("t", key.to_vec(), value.to_vec(), engine.next_seqno())
            .unwrap();
    }
    let overwrite = engine.next_seqno();
    snapshots.push(overwrite);
    engine
        .columnar_insert("t", b"row:a".to_vec(), b"a-v2".to_vec(), overwrite)
        .unwrap();
    let tree = engine.columnar_table_tree("t").unwrap();
    let deleted = engine.next_seqno();
    snapshots.push(deleted);
    tree.remove(b"row:b".to_vec(), deleted);
    snapshots.push(engine.next_seqno());
    snapshots.push(u64::MAX);

    let before: Vec<_> = snapshots
        .iter()
        .map(|&at| engine.columnar_scan("t", at).unwrap())
        .collect();
    let row = |k: &[u8], v: &[u8]| (k.to_vec(), v.to_vec());
    assert_eq!(
        before[0],
        vec![
            row(b"row:a", b"a-v1"),
            row(b"row:b", b"b-live"),
            row(b"row:c", b"c-live")
        ],
        "before the overwrite"
    );
    assert_eq!(
        before[1],
        vec![
            row(b"row:a", b"a-v2"),
            row(b"row:b", b"b-live"),
            row(b"row:c", b"c-live")
        ],
        "before the delete"
    );
    engine.flush_columnar_table("t").unwrap();
    for (at, before) in snapshots.iter().zip(&before) {
        assert_eq!(
            &engine.columnar_scan("t", *at).unwrap(),
            before,
            "snapshot {at} reads differently after the flush"
        );
    }
}

/// A table's history copied event by event into a new tree reads the same
/// at every snapshot, however the source's versions are laid out over its
/// memtable and flushed segments.
#[test]
fn a_copied_table_history_reads_the_same_at_every_snapshot() {
    use lsm_tree::AbstractTree as _;

    // Flush after each of these writes (by index), or never.
    for flush_after in [
        vec![],
        vec![0, 1, 2, 3, 4, 5, 6],
        vec![6],
        vec![3],
        vec![4, 5],
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut config = persistent_config(dir.path());
        config.compaction_workers = 0;
        let engine = StorageEngine::open(&config).unwrap();
        let mut snapshots = Vec::new();
        let mut step = 0;
        let mut after = |engine: &StorageEngine| {
            if flush_after.contains(&step) {
                engine.flush_columnar_table("source").unwrap();
            }
            step += 1;
        };
        for (key, value) in [
            (&b"row:a"[..], &b"a-v1"[..]),
            (b"row:b", b"b-live"),
            (b"row:c", b"c-live"),
            (b"row:d", b"d-live"),
        ] {
            engine
                .columnar_insert("source", key.to_vec(), value.to_vec(), engine.next_seqno())
                .unwrap();
            after(&engine);
        }
        let overwrite = engine.next_seqno();
        snapshots.push(overwrite);
        engine
            .columnar_insert("source", b"row:a".to_vec(), b"a-v2".to_vec(), overwrite)
            .unwrap();
        after(&engine);
        let source = engine.columnar_table_tree("source").unwrap();
        let deleted = engine.next_seqno();
        snapshots.push(deleted);
        source.remove(b"row:b".to_vec(), deleted);
        after(&engine);
        let ranged = engine.next_seqno();
        snapshots.push(ranged);
        source.remove_range(b"row:c".to_vec(), b"row:e".to_vec(), ranged);
        after(&engine);
        snapshots.push(engine.next_seqno());
        snapshots.push(u64::MAX);

        let copy = engine.create_columnar_table("copy").unwrap();
        for event in crate::engine::rebuild::tree_history(&source).unwrap() {
            crate::engine::rebuild::replay_event(&copy, event);
        }
        for flush_copy in [false, true] {
            if flush_copy {
                engine.flush_columnar_table("copy").unwrap();
            }
            for &at in &snapshots {
                assert_eq!(
                    engine.columnar_scan("copy", at).unwrap(),
                    engine.columnar_scan("source", at).unwrap(),
                    "flushes after {flush_after:?}, copy flushed {flush_copy}, snapshot {at}"
                );
            }
        }
    }
}

/// A columnar table's tree encodes its column pages from the configured level
/// down, plain above it.
#[test]
fn a_columnar_table_encodes_its_pages_from_the_configured_level() {
    use lsm_tree::config::ColumnEncoding;

    let policy = crate::columnar::TableTreeSettings::column_encoding_from(2);
    assert_eq!(
        &*policy,
        &[
            ColumnEncoding::Plain,
            ColumnEncoding::Plain,
            ColumnEncoding::Auto
        ][..]
    );
    let never = crate::columnar::TableTreeSettings::column_encoding_from(7);
    assert_eq!(&*never, &[ColumnEncoding::Plain][..]);
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

/// Insert row `i` into `table`, rewrite its anchor row and flush: one more
/// L0 run, overlapping every earlier one through the anchor (tables of
/// disjoint keys would line up as one run). The table then holds `i + 1`
/// rows of its own plus the anchor.
fn flushed_row(engine: &StorageEngine, table: &str, i: u32) {
    for key in [format!("row:{i:06}"), "row:anchor".to_string()] {
        engine
            .columnar_insert(
                table,
                key.into_bytes(),
                i.to_le_bytes().to_vec(),
                engine.next_seqno(),
            )
            .unwrap();
    }
    engine.flush_columnar_table(table).unwrap();
}

/// Wait up to ten seconds for `done`, failing with `what`.
fn eventually(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !done() {
        assert!(std::time::Instant::now() < deadline, "{what}");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// A table whose L0 passes the stop threshold is judged on its own: its
/// writes meet `Stop`, while another table's and the partitions' commits are
/// judged by their own trees and meet nothing. Without the table's verdict
/// its writes were never held back at all.
#[test]
fn a_table_past_its_stop_threshold_is_stopped_alone() {
    use crate::engine::core::WritePressure;
    use crate::engine::partition::Partition;

    let dir = tempfile::tempdir().unwrap();
    let mut config = persistent_config(dir.path());
    config.backpressure.l0_slowdown = 3;
    config.backpressure.l0_stop = 5;
    // Nothing compacts the runs away.
    config.compaction_workers = 0;
    let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config, std::sync::Arc::clone(&oracle)).unwrap();
    engine.create_columnar_table("cold").unwrap();
    for i in 0..3 {
        flushed_row(&engine, "hot", i);
    }
    eventually("three runs never slowed the table", || {
        engine.columnar_write_pressure("hot") == WritePressure::Slowdown
    });
    for i in 3..6 {
        flushed_row(&engine, "hot", i);
    }
    eventually("six runs never stopped the table", || {
        engine.columnar_write_pressure("hot") == WritePressure::Stop
    });
    assert_eq!(engine.columnar_write_pressure("cold"), WritePressure::None);
    assert_eq!(engine.write_pressure(), WritePressure::None);
    // A partition commit is not held back by the table.
    commit_node_write(&engine, &oracle, b"node:after-stop", b"v");
    assert_eq!(
        engine
            .get(Partition::Node, b"node:after-stop")
            .unwrap()
            .as_deref(),
        Some(&b"v"[..])
    );
}

/// A table past its slowdown threshold holds its rows to the write rate, as
/// a slowed commit is held: past the limiter's burst a megabyte at a megabyte
/// a second waits. The table's slowdown does not hold the partitions.
#[test]
fn a_slowed_table_holds_its_rows_to_the_write_rate() {
    use crate::engine::core::WritePressure;

    let dir = tempfile::tempdir().unwrap();
    let mut config = persistent_config(dir.path());
    config.backpressure.l0_slowdown = 2;
    config.compaction_workers = 0;
    let oracle = std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config, std::sync::Arc::clone(&oracle)).unwrap();
    for i in 0..2 {
        flushed_row(&engine, "slow", i);
    }
    eventually("two runs never slowed the table", || {
        engine.columnar_write_pressure("slow") == WritePressure::Slowdown
    });
    assert_eq!(engine.write_pressure(), WritePressure::None);
    engine.set_write_slowdown_rate(1024 * 1024);
    let value = vec![7u8; 1024 * 1024];
    let insert = |key: &[u8]| {
        let started = std::time::Instant::now();
        engine
            .columnar_insert("slow", key.to_vec(), value.clone(), engine.next_seqno())
            .unwrap();
        started.elapsed()
    };
    insert(b"row:big-1");
    let held = insert(b"row:big-2");
    assert!(
        held >= std::time::Duration::from_millis(500),
        "a megabyte past the burst at a megabyte a second took {held:?}"
    );
    let started = std::time::Instant::now();
    commit_node_write(&engine, &oracle, b"node:free", &value);
    assert!(
        started.elapsed() < std::time::Duration::from_millis(500),
        "the table's slowdown held a partition commit"
    );
}

/// Commit a write of `value` at `key` to the node partition through the
/// commit path, where the partitions' verdict gates it.
fn commit_node_write(
    engine: &StorageEngine,
    oracle: &coordinode_core::txn::timestamp::TimestampOracle,
    key: &[u8],
    value: &[u8],
) {
    let snap = engine.snapshot();
    let mut txn = crate::engine::transaction::Transaction::new(
        engine,
        Some(oracle),
        coordinode_core::txn::timestamp::Timestamp::from_raw(snap),
        Some(snap),
    );
    txn.put(crate::engine::partition::Partition::Node, key, value)
        .unwrap();
    let concern = coordinode_core::txn::write_concern::WriteConcern::default();
    txn.commit(&crate::engine::transaction::CommitContext {
        write_concern: &concern,
        pipeline: None,
        id_gen: None,
    })
    .unwrap();
}

/// A table written and flushed statement by statement, as one without a
/// journal is, piles up an L0 run per statement; the compaction scheduler
/// keeps it from reaching the slowdown threshold under the sustained load and
/// brings it back to the urgent threshold once the load stops, every row
/// still reading.
#[test]
fn a_table_written_statement_by_statement_keeps_its_l0_low() {
    use lsm_tree::AbstractTree;

    let dir = tempfile::tempdir().unwrap();
    let config = persistent_config(dir.path());
    let engine = StorageEngine::open(&config).unwrap();
    let mut highest = 0;
    for i in 0..200 {
        flushed_row(&engine, "trades", i);
        let tree = engine.columnar_table_tree("trades").unwrap();
        highest = highest.max(tree.l0_run_count());
    }
    assert!(
        highest < config.backpressure.l0_slowdown,
        "the table's L0 reached {highest} runs under the load"
    );
    let tree = engine.columnar_table_tree("trades").unwrap();
    eventually("the table's L0 stayed above the urgent threshold", || {
        tree.l0_run_count() <= config.compaction_l0_urgent_threshold
    });
    assert_eq!(engine.columnar_scan("trades", u64::MAX).unwrap().len(), 201);
}

/// Dropping a table whose flushes and compactions are queued and running
/// leaves none of them writing into it: its directories are gone under every
/// endpoint once the drop returns and stay gone, a new table of the same name
/// starts empty and is compacted in its turn, and a reopen finds only it.
#[test]
fn a_dropped_table_leaves_no_maintenance_behind() {
    use lsm_tree::AbstractTree;

    let hot = tempfile::tempdir().unwrap();
    let cold = tempfile::tempdir().unwrap();
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "hot",
            hot.path(),
            Media::Ssd,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let dirs = || {
        [hot.path(), cold.path()].map(|root| root.join(crate::columnar::TABLES_DIR).join("scratch"))
    };
    {
        let engine = StorageEngine::open(&config).unwrap();
        for round in 0..3u32 {
            for i in 0..40 {
                flushed_row(&engine, "scratch", round * 40 + i);
            }
            // The flushes woke the compaction monitor: its requests are
            // queued or running when the table goes.
            engine.drop_columnar_table("scratch").unwrap();
            for dir in dirs() {
                assert!(!dir.exists(), "{dir:?} outlived the drop");
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        for dir in dirs() {
            assert!(!dir.exists(), "{dir:?} came back after the drop");
        }
        for i in 0..12 {
            flushed_row(&engine, "scratch", i);
        }
        let tree = engine.columnar_table_tree("scratch").unwrap();
        eventually("the new table of the name was never compacted", || {
            tree.l0_run_count() < 12
        });
        assert_eq!(engine.columnar_scan("scratch", u64::MAX).unwrap().len(), 13);
    }
    let engine = StorageEngine::open(&config).unwrap();
    assert_eq!(engine.columnar_scan("scratch", u64::MAX).unwrap().len(), 13);
}

/// A reopen gives every table back to both schedulers: an existing table's
/// L0 piled up after the reopen is compacted, and its verdict is judged
/// against the configured thresholds.
#[test]
fn a_reopen_gives_every_table_back_to_the_schedulers() {
    use crate::engine::core::WritePressure;
    use lsm_tree::AbstractTree;

    let dir = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();
        flushed_row(&engine, "orders", 0);
        flushed_row(&engine, "trades", 0);
    }
    {
        let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();
        for i in 1..13 {
            flushed_row(&engine, "orders", i);
        }
        let tree = engine.columnar_table_tree("orders").unwrap();
        eventually("a reopened table was never compacted", || {
            tree.l0_run_count() < 13
        });
    }
    let mut config = persistent_config(dir.path());
    config.backpressure.l0_slowdown = 1;
    config.backpressure.l0_stop = 2;
    config.compaction_workers = 0;
    let engine = StorageEngine::open(&config).unwrap();
    for i in 1..3 {
        flushed_row(&engine, "trades", i);
    }
    eventually("a reopened table was never judged", || {
        engine.columnar_write_pressure("trades") == WritePressure::Stop
    });
}

/// A table whose manifest is lost is repaired from its tables at open, as a
/// partition is, instead of failing the engine: its rows read, the repair
/// lost nothing so nothing is owed, and the next open repairs nothing.
#[test]
fn a_table_with_a_lost_manifest_is_salvaged_at_open() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();
        for i in 0..5 {
            flushed_row(&engine, "trades", i);
        }
    }
    super::tests::lose_manifest(&dir.path().join(crate::columnar::TABLES_DIR).join("trades"));

    let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();
    assert_eq!(engine.columnar_scan("trades", u64::MAX).unwrap().len(), 6);
    assert!(
        engine.lossy_table_repairs().is_empty(),
        "a repair with every table intact owes nothing"
    );
    drop(engine);
    let engine = StorageEngine::open(&persistent_config(dir.path())).unwrap();
    assert!(engine.lossy_table_repairs().is_empty());
    assert_eq!(engine.columnar_scan("trades", u64::MAX).unwrap().len(), 6);
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
