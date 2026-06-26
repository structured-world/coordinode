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

/// XOR-corrupt several spread-out bytes in every SST table file beneath
/// `tables_dir` whose inode the checkpoint does not share. Returns the number of
/// files corrupted.
///
/// Three properties make this robust where a single-file heuristic was not (it
/// flaked across platforms because compaction timing and SST byte-layout
/// differ):
///
/// - **All non-checkpoint tables, not the largest one.** The live
///   post-checkpoint table is hit regardless of which file it is or whether a
///   compaction left obsolete tables around (the scrub ignores obsolete ones, so
///   corrupting them is harmless).
/// - **Excludes checkpoint-shared inodes.** Checkpoints hard-link SSTs (O(1), the
///   intended design); a file hard-linked into the checkpoint shares its physical
///   blocks with the repair base, so corrupting it would corrupt the base too —
///   which no single-node repair can recover (that needs a healthy replica or an
///   off-device PITR backup). Leaving those inodes intact keeps the base clean.
/// - **Several offsets per file.** A single mid-byte can land in the index /
///   footer (which the block scrub does not checksum) on a small table; flipping
///   bytes at 1/8, 1/4, 1/2, 3/4 guarantees at least one hits a data block.
fn corrupt_post_checkpoint_tables(tables_dir: &Path, checkpoint_root: &Path) -> usize {
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
            if ckpt_inodes.contains(&meta.ino()) {
                continue;
            }
            let mut bytes = std::fs::read(&p).expect("read table");
            let n = bytes.len();
            if n < 16 {
                continue;
            }
            for off in [n / 8, n / 4, n / 2, n * 3 / 4] {
                bytes[off] ^= 0xFF;
            }
            std::fs::write(&p, &bytes).expect("corrupt table");
            corrupted += 1;
        }
    }
    corrupted
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

#[test]
fn database_open_auto_repairs_corrupt_partition() {
    use crate::Database;
    let dir = TempDir::new().expect("tempdir");

    // Phase 1: 100 rows checkpointed (the repair base), then a bulk of
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

    // Phase 2: corrupt every post-checkpoint Node table (excluding the
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

    // Phase 3: reopen via Database::open — auto-on-open repair must heal the
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
