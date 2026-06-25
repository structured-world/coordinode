//! Integration tests for embedded oplog-journal crash recovery (G111).
//!
//! These validate the full round-trip an embedded engine performs:
//!   open_embedded → oplog_append + apply → drop (simulated crash) →
//!   open_embedded → verify the un-flushed tail is replayed from the journal.
//!
//! They mimic `OwnedLocalProposalPipeline`: a commit_ts is drawn from the
//! oracle, the mutations are journalled at that ts, then applied to the
//! memtable (which re-stamps via the oracle — exactly as `engine.put` does in
//! the real pipeline). The recovery rule under test is `entry.ts >
//! partition.highest_persisted_seqno`, which must replay only the entries that
//! did not reach an SST.

use std::sync::Arc;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};

/// A persistent (oplog-eligible) single-endpoint config rooted at `dir`.
fn durable_cfg(dir: &TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// Write a batch the way the embedded pipeline does: draw a commit_ts from the
/// oracle, journal the mutations at that ts, then apply each to the memtable.
fn write_batch(engine: &StorageEngine, oracle: &Arc<TimestampOracle>, mutations: &[Mutation]) {
    let commit_ts = oracle.next().as_raw();
    engine
        .oplog_append(mutations, commit_ts)
        .expect("oplog_append")
        .expect("journal active");
    for m in mutations {
        match m {
            Mutation::Put {
                partition,
                key,
                value,
            } => {
                engine
                    .put(Partition::from(*partition), key, value)
                    .expect("put");
            }
            Mutation::Delete { partition, key } => {
                engine
                    .delete(Partition::from(*partition), key)
                    .expect("delete");
            }
            Mutation::Merge {
                partition,
                key,
                operand,
            } => {
                engine
                    .merge(Partition::from(*partition), key, operand)
                    .expect("merge");
            }
            Mutation::RemoveRange {
                partition,
                start,
                end,
            } => {
                engine
                    .remove_range(Partition::from(*partition), start, end)
                    .expect("remove_range");
            }
        }
    }
}

#[test]
fn embedded_engine_has_journal_on_durable_endpoint() {
    let dir = TempDir::new().expect("temp dir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("open");
    assert!(
        engine.has_journal(),
        "a Durable endpoint must get a retained oplog journal"
    );
}

#[test]
fn put_survives_crash_via_journal_replay() {
    let dir = TempDir::new().expect("temp dir");

    // Phase 1: journal + apply, then "crash" (drop without persist).
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0001".to_vec(),
                value: b"survives".to_vec(),
            }],
        );
        // Drop without persist() — memtable lost, only the oplog has the write.
    }

    // Phase 2: reopen — recovery must replay the journalled Put.
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
        let val = engine.get(Partition::Node, b"node:00:0001").expect("get");
        assert_eq!(
            val.as_deref(),
            Some(b"survives".as_slice()),
            "journal replay must restore an un-flushed Put"
        );
    }
}

#[test]
fn delete_survives_crash_via_journal_replay() {
    let dir = TempDir::new().expect("temp dir");

    // Phase 1: write a value and make it durable (flushed to SST).
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0002".to_vec(),
                value: b"to-delete".to_vec(),
            }],
        );
        engine.persist().expect("persist");

        // Now journal a Delete but do NOT persist — it lives only in the oplog.
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Delete {
                partition: PartitionId::Node,
                key: b"node:00:0002".to_vec(),
            }],
        );
        // Crash.
    }

    // Phase 2: reopen — the Delete must be replayed over the durable Put.
    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
        let val = engine.get(Partition::Node, b"node:00:0002").expect("get");
        assert_eq!(
            val, None,
            "journal replay must restore an un-flushed Delete over flushed data"
        );
    }
}

#[test]
fn already_flushed_entries_are_not_replayed_over_newer_state() {
    // The critical no-double-apply guarantee: an entry whose data is already
    // durable in SST (entry.ts <= partition persisted seqno) must be SKIPPED on
    // recovery, while a later un-flushed entry to the same key IS replayed.
    let dir = TempDir::new().expect("temp dir");

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");

        // A: durable (flushed). Its journal entry stays retained.
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0003".to_vec(),
                value: b"v1-durable".to_vec(),
            }],
        );
        engine.persist().expect("persist A");

        // B: overwrite the same key, NOT flushed — only in the oplog.
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:0003".to_vec(),
                value: b"v2-journalled".to_vec(),
            }],
        );
        // Crash.
    }

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
        let val = engine.get(Partition::Node, b"node:00:0003").expect("get");
        // Recovery skips the already-durable A (ts <= persisted) and replays
        // only B, so the final value is B — never a stale resurrection of A.
        assert_eq!(
            val.as_deref(),
            Some(b"v2-journalled".as_slice()),
            "recovery must skip flushed entries and replay only the un-flushed tail"
        );
    }
}

#[test]
fn merge_replayed_once_not_doubled() {
    // A merge that is durable must be skipped on recovery (not re-applied),
    // while an un-flushed merge must be replayed exactly once. Posting-list
    // merges on Adj are additive, so a double-apply would be observable as a
    // duplicated edge — here we assert the recovered value byte-matches a
    // single clean application.
    let dir = TempDir::new().expect("temp dir");
    let operand = crate::engine::merge::encode_add(9);

    // Reference: a clean single application (no crash) for byte comparison.
    let reference = {
        let ref_dir = TempDir::new().expect("temp dir");
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&ref_dir), oracle.clone()).expect("open");
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Merge {
                partition: PartitionId::Adj,
                key: b"adj:00:0008".to_vec(),
                operand: operand.clone(),
            }],
        );
        engine.persist().expect("persist");
        engine
            .get(Partition::Adj, b"adj:00:0008")
            .expect("get")
            .map(|v| v.to_vec())
    };

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_embedded(&durable_cfg(&dir), oracle.clone()).expect("open");
        write_batch(
            &engine,
            &oracle,
            &[Mutation::Merge {
                partition: PartitionId::Adj,
                key: b"adj:00:0008".to_vec(),
                operand: operand.clone(),
            }],
        );
        // Crash without persist — the merge lives only in the oplog.
    }

    {
        let oracle = Arc::new(TimestampOracle::new());
        let engine = StorageEngine::open_embedded(&durable_cfg(&dir), oracle).expect("reopen");
        let recovered = engine
            .get(Partition::Adj, b"adj:00:0008")
            .expect("get")
            .map(|v| v.to_vec());
        assert_eq!(
            recovered, reference,
            "a replayed merge must apply exactly once (byte-identical to a single clean apply)"
        );
    }
}

#[test]
fn in_memory_engine_has_no_journal() {
    // A fully volatile (in-memory) config has no oplog-eligible endpoint, so
    // open_with_oracle / open_embedded must not create a journal.
    let config = StorageConfig::with_endpoints_no_persistence(vec![EndpointConfig::new(
        "mem",
        std::path::Path::new("/coordinode-test-in-memory"),
        Media::Ram,
        Durability::Volatile,
        Tier::Memory,
    )])
    .with_fs(Arc::new(lsm_tree::fs::MemFs::new()) as Arc<dyn lsm_tree::fs::Fs>);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&config, oracle).expect("open in-memory");
    assert!(
        !engine.has_journal(),
        "a fully volatile config must not create a disk journal"
    );
}
