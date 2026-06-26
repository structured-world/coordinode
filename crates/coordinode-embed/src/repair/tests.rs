//! Tests for embedded checkpoint + WAL-replay-repair (G111).

use std::path::Path;
use std::sync::Arc;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use tempfile::TempDir;

use super::*;

fn durable_cfg(dir: &TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn open(dir: &TempDir) -> Arc<StorageEngine> {
    let oracle = Arc::new(TimestampOracle::new());
    Arc::new(StorageEngine::open_embedded(&durable_cfg(dir), oracle).expect("open_embedded"))
}

/// Journal a Put at a fresh commit_ts, then apply it (mirrors the embedded
/// pipeline: oplog-first, then memtable).
fn put(engine: &StorageEngine, oracle_ts: u64, part: PartitionId, key: &[u8], value: &[u8]) {
    engine
        .oplog_append(
            &[Mutation::Put {
                partition: part,
                key: key.to_vec(),
                value: value.to_vec(),
            }],
            oracle_ts,
        )
        .expect("oplog_append")
        .expect("journal active");
    engine.put(Partition::from(part), key, value).expect("put");
}

/// The largest live SST beneath `search_dir` (scope it to one partition's data
/// directory) whose blocks are NOT shared with the checkpoint.
///
/// Checkpoints hard-link SSTs (O(1), no disk doubling, the intended design), so
/// a file hard-linked into the checkpoint shares its physical blocks with the
/// repair base. Corrupting such a file corrupts the base too, which no
/// single-node repair can recover (that needs a healthy replica, or an
/// off-device PITR backup). Picking the victim by inode makes the test honest
/// about what it breaks: genuine *post-checkpoint* on-disk data, the scenario
/// checkpoint+oplog repair actually recovers. Picking by size alone is
/// non-deterministic across
/// platforms (compaction timing differs) and can land on a checkpoint-shared
/// SST, which was the original cross-platform CI flake.
fn largest_post_checkpoint_file(search_dir: &Path, checkpoint_root: &Path) -> std::path::PathBuf {
    use std::os::unix::fs::MetadataExt;

    // Inodes the checkpoint holds via its hard links.
    let mut ckpt_inodes = std::collections::HashSet::new();
    let mut stack = vec![checkpoint_root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in rd.flatten() {
            let meta = entry.metadata().expect("metadata");
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                ckpt_inodes.insert(meta.ino());
            }
        }
    }

    // Largest live SST whose inode the checkpoint does not also hold.
    let mut best: Option<(u64, std::path::PathBuf)> = None;
    let mut stack = vec![search_dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("read_dir").flatten() {
            let p = entry.path();
            let meta = entry.metadata().expect("metadata");
            if meta.is_dir() {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if matches!(name, "checkpoints" | "oplog" | "text_indexes") {
                    continue;
                }
                stack.push(p);
            } else if !ckpt_inodes.contains(&meta.ino())
                && best.as_ref().map(|(s, _)| meta.len() > *s).unwrap_or(true)
            {
                best = Some((meta.len(), p));
            }
        }
    }
    best.expect("a post-checkpoint live SST to corrupt").1
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
    let removed = prune_checkpoints(&root, 2).expect("prune");
    assert_eq!(removed, 1);
    assert!(!c0.exists());
    assert!(c1.exists());
    assert!(c2.exists());
    assert_eq!(latest_checkpoint(&root), Some(c2));
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
    // oplog (the checkpoint's oplog copy stops before it).
    put(
        &engine,
        1000,
        PartitionId::Node,
        b"node:0:99999999",
        b"tail-B",
    );
    engine.persist().expect("persist B");

    // Corrupt a post-checkpoint Node SST (B's data) and confirm scrub catches
    // it. Scoped to the Node partition dir (so scrub flags Node, not some other
    // partition's larger post-checkpoint file) and to a non-checkpoint-shared
    // inode (so the repair base A, hard-linked in the checkpoint, stays clean).
    let victim = largest_post_checkpoint_file(&dir.path().join(Partition::Node.name()), &root);
    let mut bytes = std::fs::read(&victim).expect("read sst");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&victim, &bytes).expect("corrupt sst");

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

#[test]
fn database_open_auto_repairs_corrupt_partition() {
    use crate::Database;
    let dir = TempDir::new().expect("tempdir");

    // Phase 1: 100 rows checkpointed (the repair base), then ONE post-checkpoint
    // row flushed to its own SST (recorded in the oplog past the checkpoint
    // cursor), then close. The post-checkpoint SST is what we corrupt: it is not
    // hard-linked into the checkpoint, so the base stays a clean repair source.
    {
        let mut db = Database::open(dir.path()).expect("open");
        for i in 0..100u32 {
            db.execute_cypher(&format!("CREATE (n:Big {{k: {i}}})"))
                .expect("create");
        }
        db.checkpoint(3).expect("checkpoint");
        db.execute_cypher("CREATE (n:Big {k: 100})")
            .expect("create post-checkpoint");
        db.persist()
            .expect("flush post-checkpoint row to its own SST");
    }

    // Phase 2: corrupt a post-checkpoint Node SST — one whose inode the
    // checkpoint does not also hold (corrupting a checkpoint-shared SST would
    // corrupt the repair base, which single-node repair cannot recover).
    let root = checkpoint_root(dir.path());
    let victim = largest_post_checkpoint_file(&dir.path().join(Partition::Node.name()), &root);
    let mut bytes = std::fs::read(&victim).expect("read sst");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&victim, &bytes).expect("corrupt sst");

    // Phase 3: reopen via Database::open — auto-on-open repair must heal the
    // partition from the checkpoint base (100 rows) plus oplog replay (the
    // post-checkpoint row) before serving, with no explicit call.
    let mut db = Database::open(dir.path()).expect("reopen auto-repairs");
    let rows = db
        .execute_cypher("MATCH (n:Big) RETURN n.k")
        .expect("match after auto-repair");
    assert_eq!(
        rows.len(),
        101,
        "every row (checkpoint base + post-checkpoint oplog) must survive auto-repair"
    );
    assert!(
        db.verify_and_repair().expect("verify").is_clean(),
        "a re-scrub after auto-repair must be clean"
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
