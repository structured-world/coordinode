//! Tests for embedded checkpoint + WAL-replay-repair.

use std::path::Path;
use std::sync::Arc;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use tempfile::TempDir;

use super::*;

/// Remove every manifest snapshot (`v{id}`) and edit log (`edits-{id}`) of
/// the tree at `tree_dir`, its tables left in place: the structural damage
/// that makes the next open repair the tree.
fn lose_manifest(tree_dir: &Path) {
    let numbered = |rest: Option<&str>| {
        rest.is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()))
    };
    let mut lost = 0;
    for entry in std::fs::read_dir(tree_dir).expect("read tree dir") {
        let entry = entry.expect("dir entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if numbered(name.strip_prefix('v')) || numbered(name.strip_prefix("edits-")) {
            std::fs::remove_file(entry.path()).expect("remove manifest file");
            lost += 1;
        }
    }
    assert!(
        lost > 0,
        "layout changed: no manifest file in {}",
        tree_dir.display()
    );
}

fn durable_cfg(dir: &TempDir) -> StorageConfig {
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    // Fault injection addresses on-disk tables; a background rewrite must not
    // replace the victim between its selection, corruption and verification.
    config.compaction_workers = 0;
    config
}

fn open(dir: &TempDir) -> Arc<StorageEngine> {
    let oracle = Arc::new(TimestampOracle::new());
    Arc::new(StorageEngine::open_embedded(&durable_cfg(dir), oracle).expect("open_embedded"))
}

/// Journal a Put at `commit_ts`, then apply it at that ts (the embedded
/// pipeline's single call). The explicit ts lets a test tell entries apart by
/// the ts it gave them.
fn put(engine: &StorageEngine, commit_ts: u64, part: PartitionId, key: &[u8], value: &[u8]) {
    engine
        .commit_journaled(
            &[Mutation::Put {
                partition: part,
                key: key.to_vec(),
                value: value.to_vec(),
            }],
            commit_ts,
        )
        .expect("commit_journaled");
}

/// Damage every non-checkpoint table's first data-block header, located through
/// its archive directory. Blob files are checked by their whole-file digest.
fn corrupt_post_checkpoint_tables(tables_dir: &Path, checkpoint_root: &Path) -> usize {
    let ckpt_tables = checkpoint_table_names(tables_dir, checkpoint_root);

    let mut corrupted = 0;
    let mut stack = vec![tables_dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            let meta = entry.metadata().expect("metadata");
            if meta.is_dir() {
                stack.push(p);
                continue;
            }
            if tables_dir.file_name().is_some_and(|name| name == "tables")
                && !matches!(
                    lsm_tree::file::TableDirEntry::classify(&entry.file_name().to_string_lossy()),
                    lsm_tree::file::TableDirEntry::Table(_)
                )
            {
                continue;
            }
            if ckpt_tables.contains(&entry.file_name()) {
                continue;
            }
            let mut bytes = std::fs::read(&p).expect("read table");
            let n = bytes.len();
            if n < 16 {
                continue;
            }
            let offset = if tables_dir.file_name().is_some_and(|name| name == "tables") {
                let archive = lsm_tree::sfa::Reader::from_reader(&mut std::io::Cursor::new(&bytes))
                    .expect("read table layout before corruption");
                let data = archive.toc().section(b"data").expect("table data section");
                assert!(data.len() > 0, "the victim contains a data block");
                usize::try_from(data.pos()).expect("data position")
            } else {
                n / 2
            };
            bytes[offset] ^= 0xFF;
            std::fs::write(&p, &bytes).expect("corrupt table");
            corrupted += 1;
        }
    }
    corrupted
}

/// The names of the tables some checkpoint under `checkpoint_root` holds for
/// the partition whose live tables are in `tables_dir`. A checkpoint
/// hard-links a table under the name it has in the live tree, and a tree
/// never reuses a table id, so a live table with one of these names shares
/// its blocks with a checkpoint. Portable where inode numbers are not.
fn checkpoint_table_names(
    tables_dir: &Path,
    checkpoint_root: &Path,
) -> std::collections::HashSet<std::ffi::OsString> {
    let partition = tables_dir
        .parent()
        .and_then(Path::file_name)
        .expect("tables dir inside a partition dir")
        .to_os_string();
    // `tables` or `blobs`: a checkpoint links both under their live names.
    let folder = tables_dir
        .file_name()
        .expect("a named files dir")
        .to_os_string();
    let mut names = std::collections::HashSet::new();
    let mut stack = vec![checkpoint_root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        let in_tables = d.file_name().is_some_and(|n| n == folder)
            && d.parent()
                .and_then(Path::file_name)
                .is_some_and(|n| n == partition);
        for entry in rd.flatten() {
            if entry.file_type().expect("file type").is_dir() {
                stack.push(entry.path());
            } else if in_tables {
                names.insert(entry.file_name());
            }
        }
    }
    names
}

/// Even a tiny SST must receive damage in its data section, rather than only
/// in unrelated metadata whose size dominates the file.
#[test]
fn the_corruption_fixture_hits_data_in_a_small_table() {
    use lsm_tree::AbstractTree;

    let dir = TempDir::new().expect("tempdir");
    let engine = open(&dir);
    put(&engine, 1, PartitionId::Node, b"node:0:0001", b"v");
    engine.persist().expect("persist");
    let tables = dir.path().join(Partition::Node.name()).join("tables");
    let version = engine
        .tree(Partition::Node)
        .expect("node tree")
        .current_version();
    let before: Vec<_> = version
        .iter_tables()
        .map(|table| {
            let path = tables.join(table.id().to_string());
            let bytes = std::fs::read(&path).expect("read published table");
            (path, bytes)
        })
        .collect();
    assert!(!before.is_empty(), "a durable row must produce a table");
    corrupt_post_checkpoint_tables(&tables, &checkpoint_root(dir.path()));
    for (path, bytes) in before {
        let archive = lsm_tree::sfa::Reader::from_reader(&mut std::io::Cursor::new(&bytes))
            .expect("read published table layout");
        let data = archive.toc().section(b"data").expect("table data section");
        let start = usize::try_from(data.pos()).expect("data position");
        let end = start + usize::try_from(data.len()).expect("data length");
        assert!(end > start, "the table contains a data block");
        let after = std::fs::read(path).expect("read damaged table");
        assert_ne!(
            &bytes[start..end],
            &after[start..end],
            "the fault injector must damage data even when metadata dominates"
        );
    }
    let report = coordinode_storage::scrub::scrub_all(
        &engine,
        &coordinode_storage::scrub::ScrubConfig::default(),
    )
    .expect("scrub damaged data");
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.partition == Partition::Node),
        "the injected data fault must be reported: {report:?}"
    );
}

#[test]
fn checkpoint_create_latest_and_prune() {
    let dir = TempDir::new().expect("tempdir");
    let engine = open(&dir);
    put(&engine, 1, PartitionId::Node, b"node:0:0001", b"a");
    let root = checkpoint_root(dir.path());

    let c0 = create_checkpoint(&engine, &root).expect("checkpoint 0");
    let c1 = create_checkpoint(&engine, &root).expect("checkpoint 1");
    let c2 = create_checkpoint(&engine, &root).expect("checkpoint 2");
    assert_ne!(c0, c1);
    assert_eq!(latest_checkpoint(&root), Some(c2.clone()));

    // Keep newest 2 → c0 pruned, c1 + c2 remain.
    let removed = prune_checkpoints(&engine, &root, 2).expect("prune");
    assert_eq!(removed, 1);
    assert!(!c0.exists());
    assert!(c1.exists());
    assert!(c2.exists());
    assert_eq!(latest_checkpoint(&root), Some(c2));
}

/// Every file under `dir` with its bytes, by path.
fn files_with_bytes(dir: &Path) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("read dir").flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.insert(p.clone(), std::fs::read(&p).expect("read"));
            }
        }
    }
    out
}

/// Two checkpoints taken at once under one root each get a directory of
/// their own, and neither writes into the live journal: both name the next
/// checkpoint the same way, and the second used to copy its sealed segments
/// onto the first one's hard links, truncating the journal itself.
#[test]
fn concurrent_checkpoints_under_one_root_each_get_their_own() {
    use coordinode_storage::engine::oplog_journal::OplogJournalConfig;
    let dir = TempDir::new().expect("tempdir");
    let journal = OplogJournalConfig {
        max_segment_entries: 2,
        ..OplogJournalConfig::default()
    };
    let engine = StorageEngine::open_embedded_with_journal(
        &durable_cfg(&dir),
        Arc::new(TimestampOracle::new()),
        journal,
    )
    .expect("open");
    for i in 0..9u64 {
        put(
            &engine,
            i + 1,
            PartitionId::Node,
            format!("node:0:{i:04}").as_bytes(),
            b"v",
        );
    }
    engine.sync_applied().expect("sync");
    let oplog = dir.path().join("oplog");
    let before = files_with_bytes(&oplog);
    let root = checkpoint_root(dir.path());
    let barrier = std::sync::Barrier::new(2);
    let taken: Vec<std::path::PathBuf> = std::thread::scope(|s| {
        let racers: Vec<_> = (0..2)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    create_checkpoint(&engine, &root)
                })
            })
            .collect();
        racers
            .into_iter()
            .map(|r| r.join().expect("racer").expect("checkpoint"))
            .collect()
    });
    assert_ne!(taken[0], taken[1], "both checkpoints took one directory");
    let after = files_with_bytes(&oplog);
    for (path, bytes) in &before {
        if let Some(now) = after.get(path) {
            assert!(
                now.starts_with(bytes),
                "{} lost bytes to a checkpoint",
                path.display()
            );
        }
    }
}

#[test]
fn verify_and_repair_is_clean_on_healthy_engine() {
    let dir = TempDir::new().expect("tempdir");
    let engine = open(&dir);
    put(&engine, 1, PartitionId::Node, b"node:0:0001", b"a");
    engine.persist().expect("persist");
    let root = checkpoint_root(dir.path());

    let report = verify_and_repair(&engine, &root).expect("verify");
    assert!(report.is_clean(), "healthy engine must report clean");
    assert!(report.repaired.is_empty());
    assert!(report.corrupt_partitions.is_empty());
}

#[test]
fn repair_rebuilds_corrupt_partition_from_checkpoint_plus_oplog() {
    let dir = TempDir::new().expect("tempdir");
    let engine = open(&dir);
    let root = checkpoint_root(dir.path());

    // A: 100 Node rows, durable BEFORE the checkpoint — they land in the
    // checkpoint base and make the Node SST the largest on-disk file.
    for i in 0..100u32 {
        let key = format!("node:0:{i:08}");
        put(
            &engine,
            u64::from(i) + 1,
            PartitionId::Node,
            key.as_bytes(),
            b"base-A",
        );
    }
    engine.persist().expect("persist A");
    create_checkpoint(&engine, &root).expect("checkpoint");

    // B: durable AFTER the checkpoint — only in the live SSTs + the retained
    // oplog (the checkpoint's oplog copy stops before it). A bulk of rows makes
    // the post-checkpoint table large enough that a corrupted byte reliably lands
    // in a data block; the sentinel key is asserted on after repair.
    for i in 0..64u32 {
        let key = format!("node:0:5000{i:04}");
        put(
            &engine,
            1000 + u64::from(i),
            PartitionId::Node,
            key.as_bytes(),
            b"bulk-B",
        );
    }
    put(
        &engine,
        2000,
        PartitionId::Node,
        b"node:0:99999999",
        b"tail-B",
    );
    engine.persist().expect("persist B");

    // Corrupt every post-checkpoint Node table (not the checkpoint-shared base),
    // so the scrub flags Node while the repair base A stays a clean source.
    let corrupted = corrupt_post_checkpoint_tables(
        &dir.path().join(Partition::Node.name()).join("tables"),
        &root,
    );
    assert!(
        corrupted > 0,
        "test must corrupt a post-checkpoint Node table"
    );

    // Repair: base A from the checkpoint + replay B from the oplog.
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert!(
        report.corrupt_partitions.contains(&Partition::Node),
        "scrub must flag the corrupt Node partition"
    );
    assert!(
        report.repaired.contains(&Partition::Node),
        "Node must be rebuilt: {report:?}"
    );
    assert!(
        report.is_clean(),
        "re-scrub after repair must be clean: {report:?}"
    );

    // Both the checkpoint base (A) and the post-checkpoint oplog tail (B) are
    // present — proving the rebuild used checkpoint + oplog replay, not just one.
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:00000042")
            .expect("get A")
            .as_deref(),
        Some(b"base-A".as_slice()),
        "checkpoint base row must be restored"
    );
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:99999999")
            .expect("get B")
            .as_deref(),
        Some(b"tail-B".as_slice()),
        "post-checkpoint oplog tail must be replayed"
    );
}

/// A partition keeping its large values in blob files is rebuilt when one of
/// them rots, as a table-only partition is: the scrub reads the blob files,
/// and the rebuild restores both the checkpoint base and the journal tail.
#[test]
fn repair_rebuilds_a_partition_with_a_corrupt_blob_file() {
    let dir = TempDir::new().expect("tempdir");
    let engine = open(&dir);
    let root = checkpoint_root(dir.path());
    let big = |tag: u8| vec![tag; 16 * 1024];

    for i in 0..8u64 {
        put(
            &engine,
            i + 1,
            PartitionId::Blob,
            format!("blob:base:{i}").as_bytes(),
            &big(0xA1),
        );
    }
    engine.persist().expect("persist base");
    create_checkpoint(&engine, &root).expect("checkpoint");
    for i in 0..8u64 {
        put(
            &engine,
            100 + i,
            PartitionId::Blob,
            format!("blob:tail:{i}").as_bytes(),
            &big(0xB2),
        );
    }
    engine.persist().expect("persist tail");

    let corrupted = corrupt_post_checkpoint_tables(
        &dir.path().join(Partition::Blob.name()).join("blobs"),
        &root,
    );
    assert!(
        corrupted > 0,
        "test must corrupt a post-checkpoint blob file"
    );

    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert!(
        report.repaired.contains(&Partition::Blob),
        "the blob partition must be rebuilt: {report:?}"
    );
    assert!(report.is_clean(), "re-scrub after repair: {report:?}");
    for (key, value) in [("blob:base:3", big(0xA1)), ("blob:tail:5", big(0xB2))] {
        assert_eq!(
            engine
                .get(Partition::Blob, key.as_bytes())
                .expect("read after repair")
                .as_deref(),
            Some(value.as_slice()),
            "{key} after the repair"
        );
    }
}

#[test]
fn database_open_auto_repairs_corrupt_partition() {
    use crate::Database;
    let dir = TempDir::new().expect("tempdir");

    // Step 1: 100 rows checkpointed (the repair base), then a bulk of
    // post-checkpoint rows flushed to their own SSTs (recorded in the oplog past
    // the checkpoint cursor), then close. Those post-checkpoint tables are what we
    // corrupt: not hard-linked into the checkpoint, so the base stays clean.
    {
        let mut db = Database::open(dir.path()).expect("open");
        for i in 0..100u32 {
            db.execute_cypher(&format!("CREATE (n:Big {{k: {i}}})"))
                .expect("create");
        }
        db.checkpoint(3).expect("checkpoint");
        for k in 100..150u32 {
            db.execute_cypher(&format!("CREATE (n:Big {{k: {k}}})"))
                .expect("create post-checkpoint");
        }
        db.persist()
            .expect("flush post-checkpoint rows to their own SSTs");
    }

    // Step 2: corrupt every post-checkpoint Node table (excluding the
    // checkpoint-shared base), so the scrub flags Node while the repair base
    // stays a clean source for the rebuild.
    let root = checkpoint_root(dir.path());
    let corrupted = corrupt_post_checkpoint_tables(
        &dir.path().join(Partition::Node.name()).join("tables"),
        &root,
    );
    assert!(
        corrupted > 0,
        "test must corrupt a post-checkpoint Node table"
    );

    // Step 3: reopen via Database::open — auto-on-open repair must heal the
    // partition from the checkpoint base (100 rows) plus oplog replay (the
    // post-checkpoint rows) before serving, with no explicit call.
    let mut db = Database::open(dir.path()).expect("reopen auto-repairs");
    let rows = db
        .execute_cypher("MATCH (n:Big) RETURN n.k")
        .expect("match after auto-repair");
    assert_eq!(
        rows.len(),
        150,
        "every row (checkpoint base + post-checkpoint oplog) must survive auto-repair"
    );
    assert!(
        db.verify_and_repair().expect("verify").is_clean(),
        "a re-scrub after auto-repair must be clean"
    );
}

/// The operation records are rebuilt with the rest of the Schema partition:
/// a repair from the checkpoint base plus the journal keeps every outcome and
/// fence, those written after the checkpoint included, so a retry after the
/// repair still runs nothing and an excluded number still admits nothing.
#[test]
fn a_repaired_schema_keeps_operation_outcomes_and_fences() {
    use crate::Database;
    use crate::db::StatementOptions;
    use crate::operation::{IssuerId, OperationOutcome, Resolve};

    let dir = TempDir::new().expect("tempdir");
    let bound = |op| StatementOptions {
        operation: Some(op),
        ..StatementOptions::default()
    };
    let (before, after, excluded) = {
        let db = Database::open(dir.path()).expect("open");
        let issuer = db.enroll_issuer(IssuerId([0x33; 16])).expect("enroll");
        let before = issuer.next().expect("number");
        db.execute_cypher_shared_with("CREATE (:Op {k: 1})", None, None, &bound(before))
            .expect("bound write");
        db.checkpoint(3).expect("checkpoint");
        let after = issuer.next().expect("number");
        db.execute_cypher_shared_with("CREATE (:Op {k: 2})", None, None, &bound(after))
            .expect("bound write after the checkpoint");
        let excluded = issuer.next().expect("number");
        assert_eq!(
            db.resolve_operation(&excluded, Resolve::Exclude)
                .expect("exclude"),
            OperationOutcome::Aborted
        );
        db.persist()
            .expect("flush the records written after the checkpoint");
        (before, after, excluded)
    };

    let root = checkpoint_root(dir.path());
    let corrupted = corrupt_post_checkpoint_tables(
        &dir.path().join(Partition::Schema.name()).join("tables"),
        &root,
    );
    assert!(
        corrupted > 0,
        "test must corrupt a post-checkpoint Schema table"
    );

    let mut db = Database::open(dir.path()).expect("reopen auto-repairs");
    for op in [&before, &after] {
        assert!(matches!(
            db.resolve_operation(op, Resolve::Lookup).expect("lookup"),
            OperationOutcome::Committed { .. }
        ));
    }
    assert_eq!(
        db.resolve_operation(&excluded, Resolve::Lookup)
            .expect("lookup"),
        OperationOutcome::Aborted
    );
    for (op, k) in [(after, 2), (excluded, 3)] {
        assert!(matches!(
            db.execute_cypher_shared_with(
                &format!("CREATE (:Op {{k: {k}}})"),
                None,
                None,
                &bound(op)
            ),
            Err(crate::DatabaseError::OperationSettled { .. })
        ));
    }
    assert_eq!(
        db.execute_cypher("MATCH (n:Op) RETURN n")
            .expect("count")
            .len(),
        2
    );
}

#[test]
fn scheduler_starts_and_stops_cleanly() {
    let dir = TempDir::new().expect("tempdir");
    let engine = open(&dir);
    put(&engine, 1, PartitionId::Node, b"node:0:0001", b"a");
    let scheduler = CheckpointScheduler::start(
        Arc::clone(&engine),
        dir.path().to_path_buf(),
        CheckpointSchedulerConfig {
            interval: Duration::from_millis(50),
            keep: 2,
        },
    );
    // Give the scheduler a couple of ticks to take at least one checkpoint.
    std::thread::sleep(Duration::from_millis(400));
    drop(scheduler); // Drop joins the thread without hanging.
    assert!(
        latest_checkpoint(&checkpoint_root(dir.path())).is_some(),
        "scheduler should have produced a checkpoint"
    );
}

/// A partition tree that was structurally repaired AT OPEN with data loss
/// (lost `current` pointer + corrupt blocks the salvage had to drop) reads
/// checksum-clean afterwards, so the scrub cannot see the loss. The
/// open-repair record is the only evidence: `verify_and_repair` must treat a
/// lossy open repair as corruption and rebuild the partition from the
/// checkpoint plus oplog replay.
#[test]
fn lossy_open_repair_forces_checkpoint_rebuild() {
    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());

    // Base A before the checkpoint, tail B after it (same shape as
    // repair_rebuilds_corrupt_partition_from_checkpoint_plus_oplog).
    {
        let engine = open(&dir);
        for i in 0..100u32 {
            let key = format!("node:0:{i:08}");
            put(
                &engine,
                u64::from(i) + 1,
                PartitionId::Node,
                key.as_bytes(),
                b"base-A",
            );
        }
        engine.persist().expect("persist A");
        create_checkpoint(&engine, &root).expect("checkpoint");
        for i in 0..64u32 {
            let key = format!("node:0:5000{i:04}");
            put(
                &engine,
                1000 + u64::from(i),
                PartitionId::Node,
                key.as_bytes(),
                b"bulk-B",
            );
        }
        put(
            &engine,
            2000,
            PartitionId::Node,
            b"node:0:99999999",
            b"tail-B",
        );
        engine.persist().expect("persist B");
    } // engine closed: the damage below must hit a quiesced tree

    // Structural damage + content damage: the lost manifest makes the next
    // open run a repair, and the corrupt post-checkpoint tables force that
    // repair's salvage to drop blocks (a lossy replay scope).
    let node_dir = dir.path().join(Partition::Node.name());
    let corrupted = corrupt_post_checkpoint_tables(&node_dir.join("tables"), &root);
    assert!(corrupted > 0, "test must corrupt a post-checkpoint table");
    lose_manifest(&node_dir);

    // Reopen: open succeeds by structural repair, and the repair is lossy.
    let engine = open(&dir);
    let lossy = engine
        .open_repairs()
        .iter()
        .find(|r| r.partition == Partition::Node)
        .expect("Node open repair must be recorded");
    assert!(
        lossy.is_lossy(),
        "salvage over corrupt tables must report a lossy scope: {:?}",
        lossy.replay_scope
    );

    // The orchestrator must rebuild Node even though the salvaged tree now
    // scrubs clean, and the rebuild must restore both A and B in full.
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert!(
        report.repaired.contains(&Partition::Node),
        "lossy open repair must force a rebuild: {report:?}"
    );
    assert!(report.is_clean(), "post-rebuild re-scrub: {report:?}");
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:00000042")
            .expect("get A")
            .as_deref(),
        Some(b"base-A".as_slice()),
        "checkpoint base row must be restored"
    );
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:99999999")
            .expect("get B")
            .as_deref(),
        Some(b"tail-B".as_slice()),
        "post-checkpoint oplog tail must be replayed"
    );

    // The rebuild settled the repair: the tree no longer reports it.
    drop(engine);
    let engine = open(&dir);
    assert!(
        engine.open_repairs().is_empty(),
        "a rebuilt partition is not reported again: {:?}",
        engine.open_repairs()
    );
}

/// A columnar table whose tree an open repaired with loss (a lost manifest
/// over corrupt post-checkpoint tables) is rebuilt as a partition is: the
/// orchestrator restores the checkpoint's rows and replays the journal's,
/// the table reads whole, and neither the repair nor a rebuild is owed after.
#[test]
fn a_lossy_table_repair_rebuilds_the_table_from_the_checkpoint() {
    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());
    let oracle = Arc::new(TimestampOracle::new());
    let open = || {
        StorageEngine::open_embedded(&durable_cfg(&dir), Arc::clone(&oracle))
            .expect("open_embedded")
    };
    let row = |engine: &StorageEngine, i: u32, value: &[u8]| {
        engine
            .columnar_insert(
                "trades",
                format!("row:{i:06}").into_bytes(),
                value.to_vec(),
                oracle.next().as_raw(),
            )
            .expect("columnar insert");
    };
    {
        let engine = open();
        for i in 0..50 {
            row(&engine, i, b"base-A");
        }
        create_checkpoint(&engine, &root).expect("checkpoint");
        for i in 50..80 {
            row(&engine, i, b"tail-B");
        }
        engine.persist().expect("persist");
    }

    let table_dir = dir.path().join("tables").join("trades");
    let corrupted = corrupt_post_checkpoint_tables(&table_dir.join("tables"), &root);
    assert!(corrupted > 0, "test must corrupt a post-checkpoint table");
    lose_manifest(&table_dir);

    let engine = open();
    assert_eq!(engine.lossy_table_repairs(), vec!["trades".to_string()]);
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert_eq!(
        report.corrupt_tables,
        vec!["trades".to_string()],
        "{report:?}"
    );
    assert_eq!(
        report.repaired_tables,
        vec!["trades".to_string()],
        "{report:?}"
    );
    assert!(report.is_clean(), "{report:?}");
    let rows = engine
        .columnar_scan("trades", u64::MAX)
        .expect("scan the rebuilt table");
    assert_eq!(rows.len(), 80);
    assert!(rows[..50].iter().all(|(_, v)| v == b"base-A"));
    assert!(rows[50..].iter().all(|(_, v)| v == b"tail-B"));
    assert!(engine.lossy_table_repairs().is_empty());
    assert!(engine.pending_table_rebuilds().expect("intents").is_empty());

    drop(engine);
    let engine = open();
    assert!(engine.lossy_table_repairs().is_empty());
    assert_eq!(
        engine
            .columnar_scan("trades", u64::MAX)
            .expect("scan")
            .len(),
        80
    );
}

/// A table rebuild a crash cut short leaves its intent, and only that says
/// the table holds an unknown part of its rows: the next pass rebuilds it
/// again from the checkpoint and the journal, and drops the intent. The
/// table's name may be a partition's without being taken for it.
#[test]
fn a_table_rebuild_cut_short_is_rebuilt_again() {
    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());
    let oracle = Arc::new(TimestampOracle::new());
    let open = || {
        StorageEngine::open_embedded(&durable_cfg(&dir), Arc::clone(&oracle))
            .expect("open_embedded")
    };
    {
        let engine = open();
        for i in 0..10u32 {
            engine
                .columnar_insert(
                    "node",
                    format!("row:{i:06}").into_bytes(),
                    b"kept".to_vec(),
                    oracle.next().as_raw(),
                )
                .expect("columnar insert");
        }
        create_checkpoint(&engine, &root).expect("checkpoint");
    }
    // The intent a rebuild writes before it clears the table.
    let intents = dir.path().join("rebuild");
    std::fs::create_dir_all(&intents).expect("intent dir");
    std::fs::write(intents.join("table.node"), b"").expect("intent");

    let engine = open();
    assert_eq!(
        engine.pending_table_rebuilds().expect("intents"),
        vec!["node".to_string()]
    );
    assert!(engine.pending_rebuilds().expect("intents").is_empty());
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert_eq!(
        report.repaired_tables,
        vec!["node".to_string()],
        "{report:?}"
    );
    assert!(report.repaired.is_empty(), "{report:?}");
    assert!(report.is_clean(), "{report:?}");
    assert!(engine.pending_table_rebuilds().expect("intents").is_empty());
    assert_eq!(
        engine.columnar_scan("node", u64::MAX).expect("scan").len(),
        10
    );
}

/// The database open finishes a table rebuild a crash cut short, as it does a
/// partition's, before it serves anything: the intent alone says the table is
/// incomplete.
#[test]
fn database_open_finishes_an_interrupted_table_rebuild() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let data_dir = {
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), Arc::clone(&oracle))
            .expect("open_embedded");
        for i in 0..4u32 {
            engine
                .columnar_insert(
                    "trades",
                    format!("row:{i:06}").into_bytes(),
                    b"kept".to_vec(),
                    oracle.next().as_raw(),
                )
                .expect("columnar insert");
        }
        let root = checkpoint_root(engine.data_dir());
        create_checkpoint(&engine, &root).expect("checkpoint");
        engine.data_dir().to_path_buf()
    };
    let intents = data_dir.join("rebuild");
    std::fs::create_dir_all(&intents).expect("intent dir");
    std::fs::write(intents.join("table.trades"), b"").expect("intent");

    let db = crate::Database::open(&data_dir).expect("open repairs");
    assert!(
        db.engine()
            .pending_table_rebuilds()
            .expect("intents")
            .is_empty()
    );
    assert_eq!(
        db.engine()
            .columnar_scan("trades", u64::MAX)
            .expect("scan")
            .len(),
        4
    );
}

/// A table rebuild cut short with no checkpoint to finish it from is reported
/// by the database open, not passed over: the open names the table in its
/// error, and the intent stays for the operator's restore.
#[test]
fn database_open_reports_a_table_rebuild_it_cannot_finish() {
    use tracing_subscriber::layer::SubscriberExt as _;

    /// Every error event's fields, rendered.
    struct Errors(Arc<parking_lot::Mutex<Vec<String>>>);
    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Errors {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _: tracing_subscriber::layer::Context<'_, S>,
        ) {
            struct Fields(String);
            impl tracing::field::Visit for Fields {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    self.0.push_str(&format!("{}={value:?} ", field.name()));
                }
            }
            if *event.metadata().level() == tracing::Level::ERROR {
                let mut fields = Fields(String::new());
                event.record(&mut fields);
                self.0.lock().push(fields.0);
            }
        }
    }

    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let data_dir = {
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), Arc::clone(&oracle))
            .expect("open_embedded");
        engine
            .columnar_insert(
                "trades",
                b"row:000000".to_vec(),
                b"kept".to_vec(),
                oracle.next().as_raw(),
            )
            .expect("columnar insert");
        engine.data_dir().to_path_buf()
    };
    let intents = data_dir.join("rebuild");
    std::fs::create_dir_all(&intents).expect("intent dir");
    std::fs::write(intents.join("table.trades"), b"").expect("intent");

    let errors = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(Errors(Arc::clone(&errors)));
    let db = tracing::subscriber::with_default(subscriber, || crate::Database::open(&data_dir))
        .expect("the open serves what it has");
    let errors = errors.lock();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("corrupt_tables=[\"trades\"]")),
        "the open did not report the table: {errors:?}"
    );
    assert_eq!(
        db.engine().pending_table_rebuilds().expect("intents"),
        vec!["trades".to_string()],
        "the intent is kept until a restore finishes it"
    );
}

/// A lossy table repair with no checkpoint to rebuild from is reported, not
/// absorbed: the table is listed, nothing claims a rebuild, and the next open
/// reports the same loss.
#[test]
fn a_lossy_table_repair_without_a_checkpoint_reports_not_clean() {
    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());
    let oracle = Arc::new(TimestampOracle::new());
    let open = || {
        StorageEngine::open_embedded(&durable_cfg(&dir), Arc::clone(&oracle))
            .expect("open_embedded")
    };
    {
        let engine = open();
        for i in 0..40u32 {
            engine
                .columnar_insert(
                    "trades",
                    format!("row:{i:06}").into_bytes(),
                    b"unbacked".to_vec(),
                    oracle.next().as_raw(),
                )
                .expect("columnar insert");
        }
        engine.persist().expect("persist");
    }
    let table_dir = dir.path().join("tables").join("trades");
    assert!(corrupt_post_checkpoint_tables(&table_dir.join("tables"), &root) > 0);
    lose_manifest(&table_dir);

    let engine = open();
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert_eq!(
        report.corrupt_tables,
        vec!["trades".to_string()],
        "{report:?}"
    );
    assert!(report.repaired_tables.is_empty(), "{report:?}");
    assert!(!report.is_clean(), "{report:?}");
    drop(engine);
    assert_eq!(open().lossy_table_repairs(), vec!["trades".to_string()]);
}

/// A table rebuilt from a checkpoint keeps the history the checkpoint's copy
/// holds, each version at its own seqno: a snapshot taken before an overwrite,
/// a point deletion or a range deletion reads the table as it read before the
/// loss. Rewriting the copy's current rows at fresh seqnos would answer every
/// older snapshot with nothing.
#[test]
fn a_table_rebuild_keeps_the_checkpoints_history() {
    use lsm_tree::AbstractTree as _;

    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());
    let oracle = Arc::new(TimestampOracle::new());
    let open = || {
        StorageEngine::open_embedded(&durable_cfg(&dir), Arc::clone(&oracle))
            .expect("open_embedded")
    };
    let row = |engine: &StorageEngine, key: &[u8], value: &[u8]| -> u64 {
        let ts = oracle.next().as_raw();
        engine
            .columnar_insert("trades", key.to_vec(), value.to_vec(), ts)
            .expect("columnar insert");
        ts
    };
    // Snapshot seqnos to read at: each sees the versions strictly below it.
    let mut snapshots = Vec::new();
    let expected = {
        let engine = open();
        row(&engine, b"row:a", b"a-v1");
        row(&engine, b"row:b", b"b-live");
        row(&engine, b"row:c", b"c-live");
        row(&engine, b"row:d", b"d-live");
        snapshots.push(row(&engine, b"row:a", b"a-v2"));
        let tree = engine.columnar_table_tree("trades").expect("table tree");
        let deleted = oracle.next().as_raw();
        snapshots.push(deleted);
        tree.remove(b"row:b".to_vec(), deleted);
        let ranged = oracle.next().as_raw();
        snapshots.push(ranged);
        tree.remove_range(b"row:c".to_vec(), b"row:e".to_vec(), ranged);
        snapshots.push(oracle.next().as_raw());
        engine.persist().expect("persist");
        create_checkpoint(&engine, &root).expect("checkpoint");
        for i in 0..30u32 {
            row(&engine, format!("tail:{i:06}").as_bytes(), b"tail");
        }
        engine.persist().expect("persist");
        snapshots.push(u64::MAX);
        snapshots
            .iter()
            .map(|&at| engine.columnar_scan("trades", at).expect("scan before"))
            .collect::<Vec<_>>()
    };
    // The reads differ, or the test would not tell history from the latest.
    assert_ne!(expected[0], expected[1]);
    assert_ne!(expected[1], expected[2]);
    assert_ne!(expected[2], expected[3]);

    let table_dir = dir.path().join("tables").join("trades");
    assert!(corrupt_post_checkpoint_tables(&table_dir.join("tables"), &root) > 0);
    lose_manifest(&table_dir);

    let engine = open();
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert_eq!(
        report.repaired_tables,
        vec!["trades".to_string()],
        "{report:?}"
    );
    assert!(report.is_clean(), "{report:?}");
    for (at, before) in snapshots.iter().zip(&expected) {
        assert_eq!(
            &engine.columnar_scan("trades", *at).expect("scan after"),
            before,
            "the rebuilt table reads differently at snapshot {at}"
        );
    }
}

/// A table the checkpoint holds no copy of is not rebuilt as an empty table:
/// a table created after the checkpoint and one whose section went missing
/// cannot be told apart, so the rebuild refuses and changes nothing, and the
/// loss stays reported.
#[test]
fn a_table_missing_from_the_checkpoint_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());
    let oracle = Arc::new(TimestampOracle::new());
    let open = || {
        StorageEngine::open_embedded(&durable_cfg(&dir), Arc::clone(&oracle))
            .expect("open_embedded")
    };
    {
        let engine = open();
        create_checkpoint(&engine, &root).expect("checkpoint");
        for i in 0..40u32 {
            engine
                .columnar_insert(
                    "late",
                    format!("row:{i:06}").into_bytes(),
                    b"after".to_vec(),
                    oracle.next().as_raw(),
                )
                .expect("columnar insert");
        }
        engine.persist().expect("persist");
    }
    let table_dir = dir.path().join("tables").join("late");
    assert!(corrupt_post_checkpoint_tables(&table_dir.join("tables"), &root) > 0);
    lose_manifest(&table_dir);

    let engine = open();
    let salvaged = engine.columnar_scan("late", u64::MAX).expect("scan");
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert_eq!(
        report.refused_tables,
        vec![("late".to_string(), RebuildRefusal::MissingFromCheckpoint)],
        "{report:?}"
    );
    assert!(report.repaired_tables.is_empty(), "{report:?}");
    assert!(!report.is_clean(), "{report:?}");
    assert_eq!(
        engine.columnar_scan("late", u64::MAX).expect("scan"),
        salvaged,
        "a refused rebuild changes nothing"
    );
    drop(engine);
    assert_eq!(open().lossy_table_repairs(), vec!["late".to_string()]);
}

/// A catalog change to the table after the checkpoint may be a DROP and a
/// re-create under the same name: the checkpoint's rows then belong to a
/// table that no longer exists, so the rebuild refuses rather than bring them
/// back.
#[test]
fn a_table_rebuild_across_a_catalog_change_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());
    let oracle = Arc::new(TimestampOracle::new());
    let open = || {
        StorageEngine::open_embedded(&durable_cfg(&dir), Arc::clone(&oracle))
            .expect("open_embedded")
    };
    let pointer = encode_label_current_revision_key("trades");
    {
        let engine = open();
        put(
            &engine,
            oracle.next().as_raw(),
            PartitionId::Schema,
            &pointer,
            &1u64.to_be_bytes(),
        );
        for i in 0..20u32 {
            engine
                .columnar_insert(
                    "trades",
                    format!("row:{i:06}").into_bytes(),
                    b"dropped".to_vec(),
                    oracle.next().as_raw(),
                )
                .expect("columnar insert");
        }
        create_checkpoint(&engine, &root).expect("checkpoint");
        // The table is dropped and created again under its name.
        engine.drop_columnar_table("trades").expect("drop");
        put(
            &engine,
            oracle.next().as_raw(),
            PartitionId::Schema,
            &pointer,
            &2u64.to_be_bytes(),
        );
        for i in 0..40u32 {
            engine
                .columnar_insert(
                    "trades",
                    format!("new:{i:06}").into_bytes(),
                    b"current".to_vec(),
                    oracle.next().as_raw(),
                )
                .expect("columnar insert");
        }
        engine.persist().expect("persist");
    }
    // The re-created tree numbers its tables from the start again, under the
    // names the checkpoint's copy has: every one of them is the new table's.
    let no_checkpoint = TempDir::new().expect("tempdir");
    let table_dir = dir.path().join("tables").join("trades");
    assert!(corrupt_post_checkpoint_tables(&table_dir.join("tables"), no_checkpoint.path()) > 0);
    lose_manifest(&table_dir);

    let engine = open();
    let salvaged = engine.columnar_scan("trades", u64::MAX).expect("scan");
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert!(
        matches!(
            report.refused_tables.as_slice(),
            [(table, RebuildRefusal::CatalogChanged { .. })] if table == "trades"
        ),
        "{report:?}"
    );
    assert!(!report.is_clean(), "{report:?}");
    assert_eq!(
        engine.columnar_scan("trades", u64::MAX).expect("scan"),
        salvaged,
        "a refused rebuild brings no row of the checkpoint's table back"
    );
}

/// An applied journal entry that neither the checkpoint's copy nor the
/// journal suffix holds is lost data: the rebuild refuses instead of binding
/// the table's coverage past it.
#[test]
fn a_table_rebuild_missing_part_of_the_journal_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&durable_cfg(&dir), Arc::clone(&oracle))
        .expect("open_embedded");
    let row = |key: String| {
        engine
            .columnar_insert(
                "trades",
                key.into_bytes(),
                b"v".to_vec(),
                oracle.next().as_raw(),
            )
            .expect("columnar insert");
    };
    for i in 0..10u32 {
        row(format!("row:{i:06}"));
    }
    let checkpoint = create_checkpoint(&engine, &root).expect("checkpoint");
    for i in 10..20u32 {
        row(format!("row:{i:06}"));
    }
    let from = StorageEngine::checkpoint_replay_floor(&checkpoint).expect("floor");
    let mut suffix = engine
        .oplog_read_since(from)
        .expect("read journal")
        .expect("journal");
    let lost = suffix.pop().expect("an entry after the checkpoint").index;
    let before = engine.columnar_scan("trades", u64::MAX).expect("scan");

    let outcome = engine
        .repair_table_from_checkpoint(&checkpoint, &suffix, "trades", &[])
        .expect("rebuild");
    assert_eq!(
        outcome,
        Rebuild::Refused(RebuildRefusal::UncoveredEntry { index: lost })
    );
    assert_eq!(
        engine.columnar_scan("trades", u64::MAX).expect("scan"),
        before,
        "a refused rebuild changes nothing"
    );
}

/// The reported dead end: a lossy open repair with NO checkpoint to rebuild
/// from. The loss must be REPORTED (partition listed, not clean), never
/// silently absorbed, and nothing may claim a repair happened.
#[test]
fn lossy_open_repair_without_checkpoint_reports_not_clean() {
    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());

    {
        let engine = open(&dir);
        assert!(
            engine.open_repairs().is_empty(),
            "a clean open must record no repairs"
        );
        for i in 0..64u32 {
            let key = format!("node:0:{i:08}");
            put(
                &engine,
                u64::from(i) + 1,
                PartitionId::Node,
                key.as_bytes(),
                b"unbacked",
            );
        }
        engine.persist().expect("persist");
    }

    // No checkpoint exists; damage the tree structurally + corrupt its
    // tables so the open-time salvage is lossy.
    let node_dir = dir.path().join(Partition::Node.name());
    let corrupted = corrupt_post_checkpoint_tables(&node_dir.join("tables"), &root);
    assert!(corrupted > 0, "test must corrupt a table");
    lose_manifest(&node_dir);

    let engine = open(&dir);
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert!(
        report.corrupt_partitions.contains(&Partition::Node),
        "the loss must be reported: {report:?}"
    );
    assert!(
        report.repaired.is_empty(),
        "nothing can be rebuilt without a checkpoint: {report:?}"
    );
    assert!(
        !report.is_clean(),
        "a lossy repair without a rebuild source must not read clean: {report:?}"
    );
    assert!(report.checkpoint_used.is_none());

    // Nothing restored the loss, so it is not forgotten across a restart:
    // the next open reports the same repair, lossy as before.
    drop(engine);
    let engine = open(&dir);
    let again = engine
        .open_repairs()
        .iter()
        .find(|r| r.partition == Partition::Node)
        .expect("the unrestored loss is reported again");
    assert!(again.is_lossy(), "{:?}", again.replay_scope);
}

/// The repair floor: sealed, durable, time-expired oplog segments that lie at
/// or above the latest checkpoint's replay cursor must survive the purge —
/// dropping them would leave the checkpoint base with nothing to roll
/// forward, silently breaking WAL-replay-repair.
#[test]
fn oplog_purge_keeps_entries_needed_by_checkpoint_replay() {
    use coordinode_storage::engine::oplog_journal::OplogJournalConfig;

    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());
    let oracle = Arc::new(TimestampOracle::new());
    // Rotate every 2 entries (sealed segments) and expire immediately.
    let journal = OplogJournalConfig {
        retention_secs: 1,
        max_segment_entries: 2,
        ..OplogJournalConfig::default()
    };
    let engine = Arc::new(
        StorageEngine::open_embedded_with_journal(&durable_cfg(&dir), oracle, journal)
            .expect("open_embedded_with_journal"),
    );

    // Base before the checkpoint.
    for i in 0..4u32 {
        let key = format!("node:0:{i:04}");
        put(
            &engine,
            u64::from(i) + 1,
            PartitionId::Node,
            key.as_bytes(),
            b"base",
        );
    }
    engine.persist().expect("persist base");
    create_checkpoint(&engine, &root).expect("checkpoint");
    let cursor = StorageEngine::checkpoint_replay_floor(
        &latest_checkpoint(&root).expect("checkpoint exists"),
    )
    .expect("cursor");

    // Post-checkpoint tail: durable in the trees AND sealed in the journal.
    for i in 0..4u32 {
        let key = format!("node:0:9{i:03}");
        put(
            &engine,
            100 + u64::from(i),
            PartitionId::Node,
            key.as_bytes(),
            b"tail",
        );
    }
    engine.persist().expect("persist tail");

    // Aggressive purge with the checkpoint floor (what the scheduler passes):
    // everything is durable and time-expired, but entries at or above the
    // cursor must stay replayable.
    engine
        .oplog_purge_expired(4_000_000_000, cursor)
        .expect("purge");
    let replayable = engine
        .oplog_read_since(cursor)
        .expect("read_since")
        .expect("journal active");
    assert!(
        replayable.iter().any(|e| e.ts >= 100),
        "post-checkpoint tail must survive the purge for replay; got {} entries",
        replayable.len()
    );
}

/// Copy the directory tree at `from` to `to`.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create dir");
    for entry in std::fs::read_dir(from).expect("read dir").flatten() {
        let target = to.join(entry.file_name());
        if entry.file_type().expect("file type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("copy");
        }
    }
}

/// A full rebuild of the index partition reinstalls every family the
/// checkpoint holds there (index generation entries, table keys, spatial
/// points), not only keys spelled after the partition's name, and replays the
/// entries the checkpoint lacks.
#[test]
fn a_rebuild_of_the_index_partition_keeps_every_family() {
    use coordinode_core::index::encoding::encode_unique_entry_key;
    use coordinode_core::index::identity::GenerationId;
    let dir = TempDir::new().expect("tempdir");
    let engine = open(&dir);
    let root = checkpoint_root(dir.path());
    let entry = encode_unique_entry_key(GenerationId::from_raw(7), b"\x40a\x00");
    let late = encode_unique_entry_key(GenerationId::from_raw(7), b"\x40b\x00");
    put(&engine, 1, PartitionId::Idx, &entry, b"holder");
    put(&engine, 2, PartitionId::Idx, b"tkey:1", b"table");
    put(&engine, 3, PartitionId::Idx, b"idx:spatial:1", b"point");
    engine.persist().expect("persist");
    let ckpt = create_checkpoint(&engine, &root).expect("checkpoint");
    put(&engine, 50, PartitionId::Idx, &late, b"late");

    let from = StorageEngine::checkpoint_replay_floor(&ckpt).expect("floor");
    let since = engine
        .oplog_read_since(from)
        .expect("read")
        .expect("journal");
    let outcome = engine
        .repair_partition_from_checkpoint(&ckpt, &since, Partition::Idx)
        .expect("rebuild");
    assert!(matches!(outcome, Rebuild::Rebuilt { .. }), "{outcome:?}");
    for (key, value) in [
        (entry.as_slice(), &b"holder"[..]),
        (b"tkey:1", b"table"),
        (b"idx:spatial:1", b"point"),
        (late.as_slice(), b"late"),
    ] {
        assert_eq!(
            engine.get(Partition::Idx, key).expect("get").as_deref(),
            Some(value),
            "{key:?}"
        );
    }
}

/// A commit that lands while a checkpoint is being taken can reach the
/// checkpoint's journal copy without reaching its trees. The rebuild must
/// replay what the checkpoint's tree lacks, not start after the last entry
/// its journal copy happens to hold.
#[test]
fn a_rebuild_replays_what_the_checkpoint_tree_lacks() {
    let dir = TempDir::new().expect("tempdir");
    let engine = open(&dir);
    let root = checkpoint_root(dir.path());
    for i in 0..4u32 {
        let key = format!("node:0:{i:04}");
        put(
            &engine,
            u64::from(i) + 1,
            PartitionId::Node,
            key.as_bytes(),
            b"base",
        );
    }
    engine.persist().expect("persist");
    let ckpt = create_checkpoint(&engine, &root).expect("checkpoint");
    // Committed after the Node tree was captured, before the journal was.
    put(&engine, 50, PartitionId::Node, b"node:0:late", b"late");
    std::fs::remove_dir_all(ckpt.join("oplog")).expect("drop journal copy");
    copy_tree(&dir.path().join("oplog"), &ckpt.join("oplog"));

    let from = StorageEngine::checkpoint_replay_floor(&ckpt).expect("floor");
    let since = engine
        .oplog_read_since(from)
        .expect("read")
        .expect("journal");
    let outcome = engine
        .repair_partition_from_checkpoint(&ckpt, &since, Partition::Node)
        .expect("rebuild");
    assert!(matches!(outcome, Rebuild::Rebuilt { .. }), "{outcome:?}");
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:late")
            .expect("get")
            .as_deref(),
        Some(b"late".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:0002")
            .expect("get")
            .as_deref(),
        Some(b"base".as_slice())
    );
}

/// A rebuilt partition keeps every version at the seqno it was committed at:
/// the checkpoint's copy is reinstalled version by version and the journal
/// replayed at each entry's commit timestamp. A change-feed cursor past the
/// last commit sees nothing new, and one just before the post-checkpoint
/// commit sees exactly its key; rewriting the rows at fresh seqnos would hand
/// every consumer the whole partition again as changed.
#[test]
fn a_rebuilt_partition_keeps_its_commit_timestamps() {
    let dir = TempDir::new().expect("tempdir");
    let engine = open(&dir);
    let root = checkpoint_root(dir.path());
    put(&engine, 10, PartitionId::Node, b"node:0:a", b"a-v1");
    put(&engine, 20, PartitionId::Node, b"node:0:a", b"a-v2");
    put(&engine, 30, PartitionId::Node, b"node:0:b", b"b");
    engine.persist().expect("persist");
    let ckpt = create_checkpoint(&engine, &root).expect("checkpoint");
    put(&engine, 40, PartitionId::Node, b"node:0:c", b"c");

    let from = StorageEngine::checkpoint_replay_floor(&ckpt).expect("floor");
    let since = engine
        .oplog_read_since(from)
        .expect("read")
        .expect("journal");
    let outcome = engine
        .repair_partition_from_checkpoint(&ckpt, &since, Partition::Node)
        .expect("rebuild");
    assert!(matches!(outcome, Rebuild::Rebuilt { .. }), "{outcome:?}");

    assert_eq!(
        engine
            .changed_keys_since(Partition::Node, 41)
            .expect("changed since the last commit"),
        Vec::<Vec<u8>>::new()
    );
    assert_eq!(
        engine
            .changed_keys_since(Partition::Node, 31)
            .expect("changed since the checkpoint"),
        vec![b"node:0:c".to_vec()]
    );
    assert_eq!(
        engine
            .changed_keys_since(Partition::Node, 11)
            .expect("changed since the first commit"),
        vec![
            b"node:0:a".to_vec(),
            b"node:0:b".to_vec(),
            b"node:0:c".to_vec()
        ]
    );
    for (key, value) in [
        (&b"node:0:a"[..], &b"a-v2"[..]),
        (b"node:0:b", b"b"),
        (b"node:0:c", b"c"),
    ] {
        assert_eq!(
            engine.get(Partition::Node, key).expect("get").as_deref(),
            Some(value)
        );
    }
}

/// A partition rebuild whose journal suffix misses an applied entry is
/// refused and leaves the partition as it was: binding its coverage past the
/// entry would certify data the rebuild never had.
#[test]
fn a_partition_rebuild_missing_part_of_the_journal_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let engine = open(&dir);
    let root = checkpoint_root(dir.path());
    put(&engine, 10, PartitionId::Node, b"node:0:a", b"a");
    engine.persist().expect("persist");
    let ckpt = create_checkpoint(&engine, &root).expect("checkpoint");
    put(&engine, 20, PartitionId::Node, b"node:0:b", b"b");
    put(&engine, 30, PartitionId::Node, b"node:0:c", b"c");

    let from = StorageEngine::checkpoint_replay_floor(&ckpt).expect("floor");
    let mut since = engine
        .oplog_read_since(from)
        .expect("read")
        .expect("journal");
    let lost = since.pop().expect("an entry after the checkpoint").index;
    let outcome = engine
        .repair_partition_from_checkpoint(&ckpt, &since, Partition::Node)
        .expect("rebuild");
    assert_eq!(
        outcome,
        Rebuild::Refused(RebuildRefusal::UncoveredEntry { index: lost })
    );
    assert!(engine.pending_rebuilds().expect("intents").is_empty());
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:c")
            .expect("get")
            .as_deref(),
        Some(b"c".as_slice()),
        "a refused rebuild changes nothing"
    );
}

/// Run a full rebuild of Node with the disk refusing `op` after the first
/// `skip` of them, cut the power, reopen and repair. Returns whether the
/// fault stopped the rebuild; once it does not, every point inside the
/// rebuild has been cut.
///
/// The journal below the checkpoint is purged first, as the scheduler does,
/// so the checkpoint is the only copy of the base: a reopen that replayed the
/// journal into a half-rebuilt tree could not bring it back.
fn rebuild_cut_at(op: lsm_tree::fs::FaultOp, skip: u64) -> bool {
    use coordinode_storage::engine::oplog_journal::OplogJournalConfig;
    use coordinode_test_fixtures::PowerRig;

    let journal = OplogJournalConfig {
        retention_secs: 1,
        max_segment_entries: 2,
        ..OplogJournalConfig::default()
    };
    let open_rig = |rig: &PowerRig| {
        Arc::new(
            StorageEngine::open_embedded_with_journal(
                &rig.config(),
                Arc::new(TimestampOracle::new()),
                journal.clone(),
            )
            .expect("open"),
        )
    };
    let base = |i: u32| format!("node:0:{i:04}");
    let tail = |i: u32| format!("node:0:9{i:03}");

    let rig = PowerRig::new();
    let engine = open_rig(&rig);
    let root = checkpoint_root(engine.data_dir());
    for i in 0..8u32 {
        put(
            &engine,
            u64::from(i) + 1,
            PartitionId::Node,
            base(i).as_bytes(),
            b"base",
        );
    }
    engine.persist().expect("persist base");
    let ckpt = create_checkpoint(&engine, &root).expect("checkpoint");
    let cursor = StorageEngine::checkpoint_replay_floor(&ckpt).expect("floor");
    for i in 0..4u32 {
        put(
            &engine,
            100 + u64::from(i),
            PartitionId::Node,
            tail(i).as_bytes(),
            b"tail",
        );
    }
    engine.persist().expect("persist tail");
    engine
        .oplog_purge_expired(4_000_000_000, cursor)
        .expect("purge");
    let since = engine
        .oplog_read_since(cursor)
        .expect("read")
        .expect("journal");

    rig.fail_from(op, skip);
    let stopped = engine
        .repair_partition_from_checkpoint(&ckpt, &since, Partition::Node)
        .is_err();
    rig.cut(engine);

    let engine = open_rig(&rig);
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert!(report.is_clean(), "{op:?} after {skip}: {report:?}");
    for (key, value) in (0..8u32)
        .map(|i| (base(i), b"base"))
        .chain((0..4u32).map(|i| (tail(i), b"tail")))
    {
        assert_eq!(
            engine
                .get(Partition::Node, key.as_bytes())
                .expect("get")
                .as_deref(),
            Some(value.as_slice()),
            "{key} after a cut at {op:?} {skip}"
        );
    }
    stopped
}

/// The database open finishes a rebuild a crash cut short, before it serves
/// anything: the intent alone says the tree is incomplete.
#[test]
fn database_open_finishes_an_interrupted_rebuild() {
    use coordinode_storage::engine::oplog_journal::OplogJournalConfig;
    use coordinode_test_fixtures::PowerRig;

    let rig = PowerRig::new();
    let engine = Arc::new(
        StorageEngine::open_embedded_with_journal(
            &rig.config(),
            Arc::new(TimestampOracle::new()),
            OplogJournalConfig::default(),
        )
        .expect("open"),
    );
    let data_dir = engine.data_dir().to_path_buf();
    let root = checkpoint_root(&data_dir);
    put(&engine, 1, PartitionId::Node, b"node:0:0001", b"base");
    engine.persist().expect("persist");
    let ckpt = create_checkpoint(&engine, &root).expect("checkpoint");
    put(&engine, 2, PartitionId::Node, b"node:0:0002", b"tail");
    engine.persist().expect("persist");
    let since = engine.oplog_read_since(0).expect("read").expect("journal");

    // The first write of the rebuild fails: the tree is cleared, nothing
    // rebuilt reaches the disk.
    rig.fail_from(lsm_tree::fs::FaultOp::Write, 0);
    engine
        .repair_partition_from_checkpoint(&ckpt, &since, Partition::Node)
        .expect_err("the cut stops the rebuild");
    rig.cut(engine);

    let db = crate::Database::open(&data_dir).expect("open repairs");
    for (key, value) in [
        (&b"node:0:0001"[..], &b"base"[..]),
        (b"node:0:0002", b"tail"),
    ] {
        assert_eq!(
            db.engine()
                .get(Partition::Node, key)
                .expect("get")
                .as_deref(),
            Some(value)
        );
    }
    let pending = db.engine().pending_rebuilds().expect("intents");
    assert!(
        pending.is_empty(),
        "the open left {pending:?} unrebuilt; a repair now gives {:?}",
        db.verify_and_repair()
    );
}

/// A full rebuild clears the tree on disk before the rebuilt data reaches
/// it. A power cut in between must not leave a tree the next open fills
/// from the journal alone: the base lives only in the checkpoint, and the
/// cleared tree scrubs clean, so nothing else would bring it back.
#[test]
fn a_power_cut_inside_a_full_rebuild_loses_nothing() {
    use lsm_tree::fs::FaultOp;
    for op in [FaultOp::Write, FaultOp::SyncAll, FaultOp::SyncData] {
        let mut skip = 0;
        while rebuild_cut_at(op, skip) {
            skip += 1;
        }
    }
}

/// A lossy open repair whose lost span reaches the partition's apply-coverage
/// record must take the FULL rebuild, even though the span is known. Every
/// table a journalled commit flushes starts with its coverage markers, so a
/// lost table takes some of them with it; the open then cannot tell which
/// journal entries the surviving tables still hold, and replaying on a guess
/// can apply a merge twice outside the lost span. Only clearing the partition
/// and rebuilding it from checkpoint + oplog is exact. The full rebuild
/// clears the partition's history, so the original-snapshot read below is
/// refused where a scoped repair would still serve it.
#[test]
fn lossy_open_repair_losing_the_coverage_record_rebuilds_in_full() {
    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());

    // The MVCC snapshot covering everything written below, captured before
    // the engine closed: the scoped-vs-full discriminator at the bottom.
    let snap_before_damage = {
        let engine = open(&dir);
        // Base A at ts 1..=100, checkpointed: lives in a checkpoint-shared
        // SST that the corruption below never touches.
        for i in 0..100u32 {
            let key = format!("node:0:{i:08}");
            put(
                &engine,
                u64::from(i) + 1,
                PartitionId::Node,
                key.as_bytes(),
                b"base-A",
            );
        }
        engine.persist().expect("persist A");
        create_checkpoint(&engine, &root).expect("checkpoint");
        // Tail B after the checkpoint: this table gets corrupted, so the
        // lost coverage spans only B's key range.
        for i in 0..64u32 {
            let key = format!("node:0:5000{i:04}");
            put(
                &engine,
                1000 + u64::from(i),
                PartitionId::Node,
                key.as_bytes(),
                b"bulk-B",
            );
        }
        put(
            &engine,
            2000,
            PartitionId::Node,
            b"node:0:99999999",
            b"tail-B",
        );
        engine.persist().expect("persist B");
        engine.snapshot()
    };

    let node_dir = dir.path().join(Partition::Node.name());
    let corrupted = corrupt_post_checkpoint_tables(&node_dir.join("tables"), &root);
    assert!(corrupted > 0, "test must corrupt a post-checkpoint table");
    lose_manifest(&node_dir);

    let engine = open(&dir);
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert!(
        report.repaired.contains(&Partition::Node),
        "lossy open repair must be repaired: {report:?}"
    );
    assert!(
        !report.scoped.contains(&Partition::Node),
        "a span that lost coverage markers must not be repaired range-scoped: {report:?}"
    );
    assert!(report.is_clean(), "{report:?}");

    // Current values: both halves fully restored.
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:00000042")
            .expect("get A")
            .as_deref(),
        Some(b"base-A".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:99999999")
            .expect("get B")
            .as_deref(),
        Some(b"tail-B".as_slice())
    );

    // The discriminator: a full rebuild clears the partition's history, so
    // the pre-damage snapshot is no longer servable at all.
    assert!(
        matches!(
            engine.prefix_scan_at(Partition::Node, b"node:0:00000042", snap_before_damage),
            Err(coordinode_storage::error::StorageError::SnapshotOutsideRetention { .. })
        ),
        "a full rebuild must have replaced the partition's history"
    );
}

/// The scoped-repair fallback: when a lost table's coverage is unknowable
/// (its metadata never parsed), no key bound can prove a retained row is
/// unaffected, so the partition must take the FULL rebuild, and the report
/// must not claim a scoped repair.
#[test]
fn unscopable_loss_falls_back_to_full_rebuild() {
    let dir = TempDir::new().expect("tempdir");
    let root = checkpoint_root(dir.path());

    {
        let engine = open(&dir);
        for i in 0..50u32 {
            let key = format!("node:0:{i:08}");
            put(
                &engine,
                u64::from(i) + 1,
                PartitionId::Node,
                key.as_bytes(),
                b"base-A",
            );
        }
        engine.persist().expect("persist A");
        create_checkpoint(&engine, &root).expect("checkpoint");
        put(
            &engine,
            2000,
            PartitionId::Node,
            b"node:0:99999999",
            b"tail-B",
        );
        engine.persist().expect("persist B");
    }

    // Obliterate every post-checkpoint table wholesale: no readable
    // metadata, no coverage, an unknowable loss.
    let node_dir = dir.path().join(Partition::Node.name());
    let tables_dir = node_dir.join("tables");
    let ckpt = latest_checkpoint(&root).expect("checkpoint exists");
    let mut obliterated = 0usize;
    {
        let ckpt_tables = checkpoint_table_names(&tables_dir, &ckpt);
        let mut stack = vec![tables_dir.clone()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else {
                continue;
            };
            for entry in rd.flatten() {
                let p = entry.path();
                let meta = entry.metadata().expect("metadata");
                if meta.is_dir() {
                    stack.push(p);
                    continue;
                }
                if ckpt_tables.contains(&entry.file_name()) {
                    continue;
                }
                let len = meta.len() as usize;
                std::fs::write(&p, vec![0xFFu8; len]).expect("obliterate table");
                obliterated += 1;
            }
        }
    }
    assert!(
        obliterated > 0,
        "test must obliterate a post-checkpoint table"
    );
    lose_manifest(&node_dir);

    let engine = open(&dir);
    let report = verify_and_repair(&engine, &root).expect("verify_and_repair");
    assert!(
        report.repaired.contains(&Partition::Node),
        "unscopable loss must still be repaired: {report:?}"
    );
    assert!(
        !report.scoped.contains(&Partition::Node),
        "an unknowable loss must not claim a scoped repair: {report:?}"
    );
    assert!(report.is_clean(), "{report:?}");
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:00000042")
            .expect("get A")
            .as_deref(),
        Some(b"base-A".as_slice())
    );
    assert_eq!(
        engine
            .get(Partition::Node, b"node:0:99999999")
            .expect("get B")
            .as_deref(),
        Some(b"tail-B".as_slice())
    );
}
