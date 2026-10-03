//! Integration tests: single-fsync write path and crash recovery.
//!
//! Tests verify that:
//!   - `OplogManager::flush()` is a no-op when no active writer exists
//!   - `LogStore::append()` fsyncs before calling `io_completed`
//!   - A resume point behind the applied entries re-delivers them on restart
//!   - Every applied proposal is recorded in the trees it touched
//!
//! # Crash recovery model
//!
//! The write path is:
//!   ```text
//!   append to oplog → fsync → io_completed → [replicate] → commit → apply
//!   ```
//!
//! Each apply writes, with its effects, a coverage marker into every tree it
//! touches. On restart the state machine resumes from the lowest prefix every
//! tree durably covers; openraft re-delivers the committed entries after it
//! from the oplog (durable since the fsync), and a tree that already holds a
//! proposal is skipped.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::storage::{CommittedLeaderId, Entry, LogStore, Request};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::oplog::manager::OplogManager;
use openraft::entry::RaftEntry;
use openraft::storage::{IOFlushed, RaftLogReader, RaftLogStorage};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn open_engine(dir: &std::path::Path) -> Arc<StorageEngine> {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
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

/// Crash recovery: a resume point behind the applied entries makes openraft
/// re-deliver them on restart.
///
/// Scenario:
///   1. Write 5 proposals through a RaftNode (committed + applied, data in SST)
///   2. Rewind every tree's coverage base by three entries, as if they had
///      flushed only up to there
///   3. Reopen the RaftNode
///   4. openraft sees the rewound position and re-delivers the last entries
///   5. Verify all 5 entries' data is present after recovery
#[tokio::test(flavor = "multi_thread")]
async fn crash_recovery_resumes_from_the_covered_prefix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data_dir = dir.path().to_path_buf();

    // ── Step 1: Write 5 proposals and flush to SST ────────────────────────────
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

        // Flush all to SST: tree data, coverage markers and last_log_id.
        engine_read.persist().expect("persist");

        // ── Rewind the resume point by three entries ─────────────────────────
        // As if every tree had last flushed three entries ago: the coverage
        // base moves back and the markers above it go. The oplog holds all
        // entries fsynced, so they will be re-delivered.
        let applied_up_to = engine_read
            .raft_coverage()
            .expect("read coverage")
            .skip_until();
        assert!(
            applied_up_to >= 5,
            "5 proposals must be recorded as applied, got up to {applied_up_to}"
        );
        let rewound = applied_up_to - 3;
        let rewound_id = openraft::LogId::new(
            openraft::vote::leader_id_adv::CommittedLeaderId {
                term: 1,
                node_id: 1,
            },
            rewound - 1,
        );
        engine_read
            .reset_raft_coverage(rewound, &rmp_serde::to_vec(&rewound_id).expect("serialize"))
            .expect("rewind coverage");

        // Graceful shutdown (simulates "crash" after which we reopen cleanly).
        // In real crash, Drop wouldn't run and oplog entries are safe (fsynced).
        node.shutdown().await.expect("shutdown");
        // Once shutdown returns nothing but the node and this test holds
        // the engine, or the reopen below finds the directory locked.
        drop(node);
        assert_eq!(
            Arc::strong_count(&engine_read),
            2,
            "something still holds the engine after shutdown"
        );
    }

    // ── Step 2: Reopen and verify crash recovery ─────────────────────────────
    {
        let engine = open_engine(&data_dir);
        let engine_read = Arc::clone(&engine);

        // openraft will:
        //   1. Read the rewound position → StateMachine::applied_state()
        //   2. Read last_log_id from Partition::Raft → LogStore::get_log_state()
        //   3. Re-deliver committed entries after it to StateMachine::apply()
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
                "crash-key-{i} data must survive crash recovery (re-delivery from the covered prefix)"
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

/// A stopped node leaves its directory free for the next open in the same
/// process. The snapshot trigger holds the engine while it sizes the log; a
/// shutdown that only asked it to stop returned while it still did, and the
/// reopen found the directory locked. A trigger probing every millisecond is
/// mid-probe at almost any shutdown.
#[tokio::test(flavor = "multi_thread")]
async fn a_stopped_node_leaves_its_directory_free_while_the_trigger_probes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let id_gen = ProposalIdGenerator::new();
    for round in 0..8u64 {
        let engine = open_engine(dir.path());
        let node = RaftNode::open_with_oracle_and_options(
            1,
            Arc::clone(&engine),
            None,
            coordinode_raft::cluster::NodeOptions {
                snapshots: coordinode_raft::cluster::SnapshotTriggerConfig {
                    check_interval: Duration::from_millis(1),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await
        .expect("open the node");
        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: format!("probe-{round}").into_bytes(),
                value: b"v".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(10_000 + 2 * round + 1),
            start_ts: Timestamp::from_raw(10_000 + 2 * round),
            bypass_rate_limiter: false,
        };
        node.pipeline()
            .propose_and_wait(&proposal)
            .expect("propose");
        // Past the first probe interval, so the trigger is sizing the log.
        tokio::time::sleep(Duration::from_millis(5)).await;
        node.shutdown().await.expect("shutdown");
        drop(node);
        drop(engine);
    }
    // The last round's directory opens too.
    drop(open_engine(dir.path()));
}

/// A completed proposal is in its tree together with the record saying so:
/// the data is readable and the tree's coverage marks the proposal.
#[tokio::test(flavor = "multi_thread")]
async fn an_applied_proposal_is_recorded_in_its_tree() {
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

    // propose_and_wait blocks until apply() returns, so the mutation and its
    // coverage marker are both in the Node tree by now.
    let val = engine_read
        .get(Partition::Node, b"apply-order-key")
        .expect("read");
    assert_eq!(
        val.as_deref(),
        Some(b"apply-order-val".as_slice()),
        "tree mutation must be present after proposal completes"
    );

    let coverage = engine_read.raft_coverage().expect("read coverage");
    let last = coverage.skip_until();
    assert!(last > 0, "the applied proposal must be recorded");
    assert!(
        coverage.holds(Partition::Node, last - 1, 0),
        "the Node tree records the proposal it holds"
    );

    node.shutdown().await.expect("shutdown");
}
