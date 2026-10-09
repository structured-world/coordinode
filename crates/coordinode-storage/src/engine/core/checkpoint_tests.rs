use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use tempfile::TempDir;

fn disk_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open engine")
}

/// A `page_ecc = ForceOn` endpoint makes
/// `to_tree_config_with_routing` request `Config::page_ecc(true)`.
/// Verify the engine opens (no `PageEccUnsupported`), and a value
/// survives a flush-to-SST + reopen — i.e. it round-trips through
/// the Reed-Solomon-trailered block codec. Only meaningful with the
/// `page_ecc` feature compiled, so the test is gated on it.
#[cfg(feature = "page_ecc")]
#[test]
fn page_ecc_force_on_round_trips_through_sst() {
    use crate::engine::config::PageEccPolicy;

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ecc",
            dir.path(),
            Media::Hdd,
            Durability::Durable, // ForceOn overrides Durable's Auto-off
            Tier::Warm,
        )
        .with_page_ecc(PageEccPolicy::ForceOn),
    ]);

    let engine = StorageEngine::open(&config).expect("open engine with page_ecc");
    engine
        .put(Partition::Node, b"k-ecc", b"v-ecc")
        .expect("put");
    // Flush memtable → SST so the page-ECC block codec runs on disk.
    engine.persist().expect("persist to SST");
    drop(engine);

    // Reopen and read back through the ECC-decoded SST block path.
    let reopened = StorageEngine::open(&config).expect("reopen engine with page_ecc");
    assert_eq!(
        reopened
            .get(Partition::Node, b"k-ecc")
            .expect("get")
            .as_deref(),
        Some(b"v-ecc".as_ref()),
        "value must round-trip through page-ECC SST blocks",
    );
}

/// A checkpoint is documented as a complete database: `STORAGE COLUMNAR`
/// tables live outside the partition trees, so they must be captured too, or
/// a store served from the checkpoint silently loses every columnar row.
#[cfg(feature = "columnar")]
#[test]
fn checkpoint_carries_the_columnar_tables() {
    let src_dir = TempDir::new().expect("src tempdir");
    let engine = disk_engine(src_dir.path());
    let at = engine.next_seqno();
    engine
        .columnar_insert("metrics", b"row-1".to_vec(), b"v1".to_vec(), at)
        .expect("insert");

    let ckpt_parent = TempDir::new().expect("ckpt parent");
    let target = ckpt_parent.path().join("snap");
    engine.create_checkpoint(&target).expect("checkpoint");
    drop(engine);

    let restored = disk_engine(&target);
    assert_eq!(
        restored
            .columnar_scan("metrics", restored.snapshot())
            .expect("scan"),
        vec![(b"row-1".to_vec(), b"v1".to_vec())]
    );
}

/// A store spread over several endpoints checkpoints into one directory, and
/// the checkpoint opens there: its persisted per-level routing names
/// endpoints the checkpoint does not have.
#[test]
fn checkpoint_of_a_multi_endpoint_store_opens_in_one_directory() {
    let hot = TempDir::new().expect("hot");
    let cold = TempDir::new().expect("cold");
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "hot",
            hot.path(),
            Media::Nvme,
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
    let engine = StorageEngine::open(&config).expect("open");
    for i in 0..64u32 {
        engine
            .put(Partition::Node, format!("node:0:{i:04}").as_bytes(), b"v")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine
        .force_compaction(Partition::Node)
        .expect("compact to the cold level");

    let ckpt_parent = TempDir::new().expect("ckpt parent");
    let target = ckpt_parent.path().join("snap");
    engine.create_checkpoint(&target).expect("checkpoint");
    drop(engine);

    let restored = StorageEngine::open_checkpoint(&target).expect("open checkpoint");
    assert_eq!(
        restored
            .prefix_scan(Partition::Node, b"node:0:")
            .expect("scan")
            .count(),
        64
    );
}

#[test]
fn checkpoint_round_trips_all_partitions() {
    let src_dir = TempDir::new().expect("src tempdir");
    let engine = disk_engine(src_dir.path());

    // Seed a couple of partitions, including Schema (which carries the
    // interner) and Node (the main data partition).
    engine
        .put(Partition::Node, b"k-node", b"v-node")
        .expect("put node");
    engine
        .put(Partition::Schema, b"k-schema", b"v-schema")
        .expect("put schema");
    engine
        .put(Partition::Adj, b"adj:T:out:x", b"posting")
        .expect("put adj");

    // Checkpoint into a fresh sibling dir.
    let ckpt_parent = TempDir::new().expect("ckpt parent");
    let target = ckpt_parent.path().join("snap1");
    let summary = engine.create_checkpoint(&target).expect("checkpoint");
    assert_eq!(
        summary.partitions,
        Partition::all().len(),
        "every partition tree must be checkpointed"
    );
    // Partition directories are named by `Partition::name()`, which is
    // lower-case ("node", "schema"). Assert the exact names: a capitalised
    // path silently passes on case-insensitive filesystems (macOS) and
    // fails on case-sensitive ones (Linux CI).
    assert!(
        target.join(Partition::Node.name()).exists(),
        "Node partition dir in checkpoint"
    );
    assert!(
        target.join(Partition::Schema.name()).exists(),
        "Schema partition dir in checkpoint"
    );

    // Drop the source engine, then open a brand-new engine against the
    // checkpoint and confirm every written key is present and correct.
    drop(engine);
    let restored = disk_engine(&target);
    assert_eq!(
        restored
            .get(Partition::Node, b"k-node")
            .expect("get node")
            .as_deref(),
        Some(b"v-node".as_ref()),
        "node value must survive the checkpoint"
    );
    assert_eq!(
        restored
            .get(Partition::Schema, b"k-schema")
            .expect("get schema")
            .as_deref(),
        Some(b"v-schema".as_ref()),
        "schema value must survive the checkpoint"
    );
    assert_eq!(
        restored
            .get(Partition::Adj, b"adj:T:out:x")
            .expect("get adj")
            .as_deref(),
        Some(b"posting".as_ref()),
        "adjacency value must survive the checkpoint"
    );
}

/// A sealed journal segment never changes again, so a checkpoint shares it
/// with the store instead of copying it: every checkpoint used to carry a
/// full copy of the journal. The active segment is still being written and
/// is copied. Both read back whole from the checkpoint.
#[test]
fn checkpoint_links_sealed_journal_segments_and_copies_the_active_one() {
    use crate::engine::config::SyncMethod;
    use crate::oplog::entry::{OplogEntry, OplogOp};
    use crate::oplog::segment::{SegmentReader, SegmentWriter};

    let entry = |index: u64| OplogEntry {
        ts: 1000 + index,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 1,
            key: format!("node:k{index}").into_bytes(),
            value: vec![7u8; 512],
        }],
        is_migration: false,
        pre_images: None,
    };

    let src_dir = TempDir::new().expect("src tempdir");
    let engine = disk_engine(src_dir.path());
    let journal = src_dir.path().join("oplog").join("0");
    std::fs::create_dir_all(&journal).expect("journal dir");
    let sealed_name = "oplog-00000000000000000001.bin";
    let active_name = "oplog-00000000000000000009.bin";
    let mut sealed =
        SegmentWriter::create(&journal.join(sealed_name), 0, 1, SyncMethod::Full).expect("sealed");
    for i in 1..9 {
        sealed.append(&entry(i)).expect("append");
    }
    sealed.seal().expect("seal");
    let mut active =
        SegmentWriter::create(&journal.join(active_name), 0, 9, SyncMethod::Full).expect("active");
    active.append(&entry(9)).expect("append");
    active.flush_and_sync().expect("sync");

    let ckpt_parent = TempDir::new().expect("ckpt parent");
    let target = ckpt_parent.path().join("snap");
    let summary = engine.create_checkpoint(&target).expect("checkpoint");
    let captured = target.join("oplog").join("0");

    let active_len = std::fs::metadata(journal.join(active_name))
        .expect("active len")
        .len();
    assert_eq!(
        summary.oplog_bytes, active_len,
        "only the active segment is copied"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        let ino = |p: &std::path::Path| std::fs::metadata(p).expect("metadata").ino();
        assert_eq!(
            ino(&captured.join(sealed_name)),
            ino(&journal.join(sealed_name)),
            "the sealed segment is the same file"
        );
        assert_ne!(
            ino(&captured.join(active_name)),
            ino(&journal.join(active_name)),
            "the active segment is a copy"
        );
    }

    // Appending to the store's active segment does not reach the copy.
    active.append(&entry(10)).expect("append");
    active.flush_and_sync().expect("sync");
    drop(active);
    let read = SegmentReader::open(&captured.join(sealed_name)).expect("read sealed");
    assert_eq!(read.entries().len(), 8);
    let tail = SegmentReader::open_active(&captured.join(active_name)).expect("read active");
    assert_eq!(tail.entries().len(), 1);
}

#[test]
fn checkpoint_refuses_existing_target() {
    let src_dir = TempDir::new().expect("src tempdir");
    let engine = disk_engine(src_dir.path());
    let existing = TempDir::new().expect("existing target");
    match engine.create_checkpoint(existing.path()) {
        Err(e) => assert!(
            format!("{e}").contains("already exists"),
            "must refuse to overwrite an existing target, got: {e}"
        ),
        Ok(_) => panic!("checkpoint must refuse an existing target"),
    }
}

/// Flushes that outpace compaction leave more than 255 runs in L0 (the L0
/// backpressure wall only rejects client writes, and an operator may raise
/// it). The storage engine used to write such a level into a manifest
/// snapshot with its run count wrapped to a byte, so a checkpoint of it did
/// not open. Both the checkpoint and the source must open with every run.
#[test]
fn checkpoint_of_more_than_255_l0_runs_opens_with_every_run() {
    use lsm_tree::AbstractTree;

    const RUNS: usize = 300;

    let src_dir = TempDir::new().expect("src tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        src_dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    // No compaction worker, so nothing folds L0 while it grows.
    config.compaction_workers = 0;
    let engine = StorageEngine::open(&config).expect("open engine");

    for i in 0..RUNS {
        let key = format!("k{i:05}");
        engine
            .put(Partition::Node, key.as_bytes(), key.as_bytes())
            .expect("put key");
        // Every flush also rewrites `zzz`, so each table overlaps every other
        // one and each flush stays a run of its own.
        engine
            .put(Partition::Node, b"zzz", key.as_bytes())
            .expect("put zzz");
        engine
            .tree(Partition::Node)
            .expect("node tree")
            .flush_active_memtable(0)
            .expect("flush");
    }
    assert!(
        engine
            .tree(Partition::Node)
            .expect("node tree")
            .l0_run_count()
            > 255,
        "the scenario needs a level wider than a byte can count",
    );

    let ckpt_parent = TempDir::new().expect("ckpt parent");
    let target = ckpt_parent.path().join("wide");
    engine.create_checkpoint(&target).expect("checkpoint");
    drop(engine);

    let newest = format!("k{:05}", RUNS - 1);
    for (what, dir) in [("checkpoint", target.as_path()), ("source", src_dir.path())] {
        let reopened = disk_engine(dir);
        for i in 0..RUNS {
            let key = format!("k{i:05}");
            assert_eq!(
                reopened
                    .get(Partition::Node, key.as_bytes())
                    .expect("get")
                    .as_deref(),
                Some(key.as_bytes()),
                "{key} must read back from the {what}",
            );
        }
        // The newest run still shadows the older ones: run order survived.
        assert_eq!(
            reopened
                .get(Partition::Node, b"zzz")
                .expect("get zzz")
                .as_deref(),
            Some(newest.as_bytes()),
            "run order must survive in the {what}",
        );
    }
}
