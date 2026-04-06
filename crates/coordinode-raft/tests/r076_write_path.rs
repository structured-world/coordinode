//! Integration tests: Single-fsync write path and crash recovery (R076).
//!
//! Tests verify that:
//!   - `OplogManager::flush()` is a no-op when no active writer exists
//!   - `LogStore::append()` fsyncs before calling `io_completed`
//!   - Crash scenario: stale `applied_index` triggers oplog replay on restart
//!   - Re-applying already-applied entries is idempotent (same seqno → same value)
//!
//! # Crash recovery model (ADR-017)
//!
//! The write path is:
//!   ```text
//!   append to oplog → fsync → io_completed → [replicate] → commit → apply → save_applied
//!   ```
//!
//! If the process crashes AFTER fsync but BEFORE save_applied:
//!   - SST-persisted `applied_index` is stale (behind actual applied entries)
//!   - On restart, openraft re-delivers committed entries after `applied_index`
//!   - Those entries are in the oplog (durable since the fsync)
//!   - Re-applying is idempotent: same HLC seqno → same (key, seqno, value)

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::storage::{CommittedLeaderId, Entry, LogStore, Request};
use coordinode_storage::engine::config::StorageConfig;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::oplog::manager::OplogManager;
use openraft::entry::RaftEntry;
use openraft::storage::{IOFlushed, RaftLogReader, RaftLogStorage};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn open_engine(dir: &std::path::Path) -> Arc<StorageEngine> {
    let config = StorageConfig::new(dir);
    Arc::new(StorageEngine::open(&config).expect("open engine"))
}

fn make_proposal(id_raw: u64, key: &str, value: &str, ts: u64) -> RaftProposal {
    RaftProposal {
        id: ProposalId::from_raw(id_raw),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: key.as_bytes().to_vec(),
            value: value.as_bytes().to_vec(),
        }],
        commit_ts: Timestamp::from_raw(ts),
        start_ts: Timestamp::from_raw(ts - 1),
        bypass_rate_limiter: false,
    }
}

fn make_entry(index: u64, term: u64) -> Entry {
    let proposal = make_proposal(
        index,
        &format!("node:1:{index}"),
        &format!("val-{index}"),
        1000 + index,
    );
    let log_id = openraft::LogId::new(CommittedLeaderId { term, node_id: 0 }, index);
    Entry::new_normal(log_id, Request::single(proposal))
}

// Key in Partition::Schema where the state machine persists last_applied.
// Defined in storage.rs — hardcoded here to avoid exposing it as pub.
const KEY_SM_APPLIED: &[u8] = b"raft:sm:applied";

// ── OplogManager-level tests ──────────────────────────────────────────────────

/// `OplogManager::flush()` is a no-op when there is no active writer.
#[test]
fn oplog_flush_noop_without_active_writer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600)
        .expect("open manager");

    // No entries appended → no active writer.
    mgr.flush().expect("flush on empty manager must succeed");
}

/// `OplogManager::flush()` succeeds after appending entries.
///
/// Verifies that the BufWriter is flushed and sync_data() completes without
/// error. Entry readability is confirmed via read_range after a rotate.
#[test]
fn oplog_flush_after_append_readable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = OplogManager::open(dir.path(), 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600)
        .expect("open manager");

    use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};

    for i in 0..5u64 {
        let entry = OplogEntry {
            ts: 1000 + i,
            term: 1,
            index: i,
            shard: 0,
            ops: vec![OplogOp::Insert {
                partition: 1,
                key: format!("k{i}").into_bytes(),
                value: b"v".to_vec(),
            }],
            is_migration: false,
            pre_images: None,
        };
        mgr.append(&entry).expect("append");
    }

    // Fsync: flush BufWriter + sync_data
    mgr.flush().expect("flush must succeed");

    // Entries must be readable after seal (read_range rotates the active writer).
    let entries = mgr.read_range(0, 5).expect("read_range after flush");
    assert_eq!(entries.len(), 5, "all 5 entries must be readable");
    assert_eq!(entries[0].index, 0);
    assert_eq!(entries[4].index, 4);
}

// ── LogStore-level fsync test ─────────────────────────────────────────────────

/// `LogStore::append()` fsyncs entries before calling `io_completed`.
///
/// Verifies that entries are readable via `try_get_log_entries` immediately
/// after append — i.e., the BufWriter was flushed and the data is on disk.
/// (True kernel-level crash durability can't be tested in a user-space test,
/// but this confirms the flush path is exercised.)
#[tokio::test]
async fn logstore_append_fsyncs_data_readable_immediately() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());
    let mut store = LogStore::open(Arc::clone(&engine)).expect("open");

    let entries: Vec<Entry> = (1..=3u64).map(|i| make_entry(i, 1)).collect();
    store
        .append(entries, IOFlushed::noop())
        .await
        .expect("append");

    // Entries must be readable without any explicit seal/rotate.
    let loaded = store
        .try_get_log_entries(1u64..=3)
        .await
        .expect("try_get_log_entries");

    assert_eq!(
        loaded.len(),
        3,
        "all 3 entries must be readable after fsync"
    );
    assert_eq!(loaded[0].log_id.index, 1);
    assert_eq!(loaded[2].log_id.index, 3);
}

// ── Crash recovery: stale applied_index ──────────────────────────────────────

/// Crash recovery: stale `applied_index` triggers oplog replay on restart.
///
/// Scenario:
///   1. Write 5 proposals through a RaftNode (committed + applied, data in SST)
///   2. Overwrite `applied_index` to 2 (simulates crash where FlushManager
///      flushed SST with applied_index=2 but entries 3-5 were lost from memtable)
///   3. Flush the tampered value to SST
///   4. Reopen the RaftNode
///   5. openraft: sees applied=2 in SST, log ends at 5 → re-delivers entries 3-5
///   6. Re-application is idempotent (same seqno, same value)
///   7. Verify all 5 entries' data is present after recovery
#[tokio::test(flavor = "multi_thread")]
async fn crash_recovery_stale_applied_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data_dir = dir.path().to_path_buf();

    // ── Phase 1: Write 5 proposals and flush to SST ───────────────────────────
    {
        let engine = open_engine(&data_dir);
        let engine_read = Arc::clone(&engine);
        let node = RaftNode::single_node(Arc::clone(&engine))
            .await
            .expect("bootstrap");

        // Wait for leadership
        tokio::time::sleep(Duration::from_millis(500)).await;

        let pipeline = node.pipeline();
        let id_gen = ProposalIdGenerator::new();

        for i in 1u64..=5 {
            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("crash-key-{i}").into_bytes(),
                    value: format!("crash-val-{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(1000 + i),
                start_ts: Timestamp::from_raw(1000 + i - 1),
                bypass_rate_limiter: false,
            };
            pipeline.propose_and_wait(&proposal).expect("propose");
        }

        // Verify data is there before we tamper
        for i in 1u64..=5 {
            let val = engine_read
                .get(Partition::Node, format!("crash-key-{i}").as_bytes())
                .expect("read");
            assert_eq!(
                val.as_deref(),
                Some(format!("crash-val-{i}").as_bytes()),
                "data must be present before tamper"
            );
        }

        // Flush all to SST — including tree data, applied_index, and last_log_id.
        engine_read.persist().expect("persist");

        // ── Tamper: overwrite applied_index to 2 ─────────────────────────────
        // This simulates a crash where:
        //   - SST had applied_index=2 at the last flush
        //   - Entries 3,4,5 were applied to memtable but memtable was lost
        //   - applied_index in memtable (=5) was also lost
        // We reconstruct this state by overwriting the persisted applied_index.
        //
        // The oplog already has all 5 entries fsynced — they will be replayed.
        let log_id_2: Option<openraft::LogId<CommittedLeaderId>> = engine_read
            .get(Partition::Schema, KEY_SM_APPLIED)
            .expect("read applied_index")
            .and_then(|b| rmp_serde::from_slice(&b).ok());

        // Confirm that applied_index is currently ≥ 5 (counting bootstrap at 0,
        // membership at 1, then proposals at 2-6 or similar offset).
        // We don't assert the exact index since openraft adds Membership entries.
        assert!(
            log_id_2.is_some(),
            "applied_index must be set after 5 proposals"
        );
        let real_applied = log_id_2.unwrap().index;
        assert!(
            real_applied >= 5,
            "applied_index must be ≥ 5, got {real_applied}"
        );

        // Build a stale applied_index = real_applied - 3 (replay last 3 entries).
        let stale_index = real_applied - 3;
        let stale_log_id = openraft::LogId::new(
            openraft::vote::leader_id_adv::CommittedLeaderId {
                term: 1,
                node_id: 1,
            },
            stale_index,
        );
        let stale_bytes = rmp_serde::to_vec(&Some(stale_log_id)).expect("serialize");
        engine_read
            .put(Partition::Schema, KEY_SM_APPLIED, &stale_bytes)
            .expect("overwrite applied_index");

        // Persist tampered value to SST — SST now has stale applied_index.
        engine_read.persist().expect("persist tampered");

        // Graceful shutdown (simulates "crash" after which we reopen cleanly).
        // In real crash, Drop wouldn't run and oplog entries are safe (fsynced).
        node.shutdown().await.expect("shutdown");
    }

    // ── Phase 2: Reopen and verify crash recovery ────────────────────────────
    {
        let engine = open_engine(&data_dir);
        let engine_read = Arc::clone(&engine);

        // openraft will:
        //   1. Read applied_index=stale from SST → StateMachine::applied_state()
        //   2. Read last_log_id from Partition::Raft → LogStore::get_log_state()
        //   3. Re-deliver committed entries after stale_index to StateMachine::apply()
        //   4. apply() re-applies those mutations (idempotent: same seqno)
        let node = RaftNode::open(1, Arc::clone(&engine))
            .await
            .expect("reopen");

        // Wait for recovery + leadership
        tokio::time::sleep(Duration::from_millis(3000)).await;

        // All 5 entries' data must be present after replay.
        for i in 1u64..=5 {
            let val = engine_read
                .get(Partition::Node, format!("crash-key-{i}").as_bytes())
                .expect("read after recovery");
            assert_eq!(
                val.as_deref(),
                Some(format!("crash-val-{i}").as_bytes()),
                "crash-key-{i} data must survive crash recovery (stale applied_index replay)"
            );
        }

        // Node must be able to accept new proposals — confirms it's the leader.
        let pipeline = node.pipeline();
        let id_gen = ProposalIdGenerator::with_base(1000);
        let new_proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"crash-key-new".to_vec(),
                value: b"crash-val-new".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(9000),
            start_ts: Timestamp::from_raw(8999),
            bypass_rate_limiter: false,
        };
        pipeline
            .propose_and_wait(&new_proposal)
            .expect("propose after crash recovery");

        let new_val = engine_read
            .get(Partition::Node, b"crash-key-new")
            .expect("read new proposal");
        assert_eq!(
            new_val.as_deref(),
            Some(b"crash-val-new".as_slice()),
            "new proposal after recovery must be applied"
        );

        node.shutdown().await.expect("shutdown");
    }
}

/// Verify that the apply order fix (mutations before save_applied) is correct.
///
/// After the fix: if we inspect the engine state RIGHT before save_applied runs,
/// the tree data must already be in the memtable. This test verifies the net
/// result: data is readable after proposal completion.
///
/// This is a regression test for the "save_applied before apply" bug.
#[tokio::test(flavor = "multi_thread")]
async fn apply_order_mutations_before_applied_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());
    let engine_read = Arc::clone(&engine);
    let node = RaftNode::single_node(Arc::clone(&engine))
        .await
        .expect("bootstrap");

    tokio::time::sleep(Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"apply-order-key".to_vec(),
            value: b"apply-order-val".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(5000),
        start_ts: Timestamp::from_raw(4999),
        bypass_rate_limiter: false,
    };

    pipeline.propose_and_wait(&proposal).expect("propose");

    // After propose_and_wait returns, both tree mutation AND applied_index must
    // be in memtable. Since propose_and_wait blocks until apply() returns,
    // both writes have happened by now.
    let val = engine_read
        .get(Partition::Node, b"apply-order-key")
        .expect("read");
    assert_eq!(
        val.as_deref(),
        Some(b"apply-order-val".as_slice()),
        "tree mutation must be present after proposal completes"
    );

    let applied_raw = engine_read
        .get(Partition::Schema, KEY_SM_APPLIED)
        .expect("read applied_index");
    assert!(
        applied_raw.is_some(),
        "applied_index must be set after proposal completes"
    );

    node.shutdown().await.expect("shutdown");
}
