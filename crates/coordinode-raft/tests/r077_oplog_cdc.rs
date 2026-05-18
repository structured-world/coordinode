//! Integration tests: Oplog CDC consumer API (R077).
//!
//! Tests verify that:
//!   - `OplogTailer` streams entries from sealed segments
//!   - Resume token resumes the stream from the correct position
//!   - `is_migration` filter excludes/includes entries correctly
//!   - `edge_types` filter only delivers entries touching the requested adj keys
//!   - After RaftNode writes proposals, OplogTailer delivers the committed log entries
//!   - Multiple segments across rotation boundaries are read in order

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::cluster::RaftNode;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::manager::OplogManager;
use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer, ResumeToken};

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

fn make_entry(index: u64, ts: u64, is_migration: bool) -> OplogEntry {
    OplogEntry {
        ts,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 1,
            key: format!("node:k{index}").into_bytes(),
            value: b"v".to_vec(),
        }],
        is_migration,
        pre_images: None,
    }
}

fn open_manager(dir: &std::path::Path) -> OplogManager {
    OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open manager")
}

// ── OplogTailer unit-level integration tests ──────────────────────────────────

/// After writing 5 entries to the oplog and sealing, OplogTailer delivers all 5.
#[test]
fn tailer_delivers_all_entries_from_oplog() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    mgr.rotate().expect("seal");

    let token = ResumeToken::from_start(0);
    let mut tailer = OplogTailer::new(dir.path(), token);
    let batch = tailer.read_next(100, &CdcFilters::default()).expect("read");

    assert_eq!(batch.len(), 5, "all 5 entries must be delivered");
    for (i, (entry, _)) in batch.iter().enumerate() {
        assert_eq!(entry.index, i as u64, "entries in index order");
    }
}

/// Resume token correctly resumes after partial consumption.
#[test]
fn tailer_resume_continues_from_last_position() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    for i in 0..8u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    mgr.rotate().expect("seal");

    // First consumer: read 3 entries
    let mut tailer = OplogTailer::new(dir.path(), ResumeToken::from_start(0));
    let first = tailer.read_next(3, &CdcFilters::default()).expect("read");
    assert_eq!(first.len(), 3);
    let resume = first[2].1.clone();
    assert_eq!(resume.entry_offset, 3, "token should point past 3rd entry");

    // Second consumer from resume: should get entries 3-7 (5 remaining)
    let mut tailer2 = OplogTailer::new(dir.path(), resume);
    let second = tailer2
        .read_next(100, &CdcFilters::default())
        .expect("read");
    assert_eq!(second.len(), 5, "resume delivers remaining 5 entries");
    assert_eq!(second[0].0.index, 3, "first remaining entry is index 3");
    assert_eq!(second[4].0.index, 7);
}

/// `is_migration=Some(false)` filter skips migration entries.
#[test]
fn tailer_filter_skips_migration_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    mgr.append(&make_entry(0, 1000, false)).expect("append"); // normal
    mgr.append(&make_entry(1, 1001, true)).expect("append"); // migration
    mgr.append(&make_entry(2, 1002, false)).expect("append"); // normal
    mgr.append(&make_entry(3, 1003, true)).expect("append"); // migration
    mgr.rotate().expect("seal");

    let filters = CdcFilters {
        is_migration: Some(false),
        ..Default::default()
    };
    let mut tailer = OplogTailer::new(dir.path(), ResumeToken::from_start(0));
    let batch = tailer.read_next(100, &filters).expect("read");

    assert_eq!(batch.len(), 2, "only 2 non-migration entries");
    assert!(batch.iter().all(|(e, _)| !e.is_migration));
    assert_eq!(batch[0].0.index, 0);
    assert_eq!(batch[1].0.index, 2);
}

/// `edge_types` filter only delivers entries touching adj keys for that type.
#[test]
fn tailer_filter_edge_type_delivers_only_matching() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    // FOLLOWS adj entry
    mgr.append(&OplogEntry {
        ts: 1000,
        term: 1,
        index: 0,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 2,
            key: b"adj:FOLLOWS:out:\x00\x00\x00\x00\x00\x00\x00\x01".to_vec(),
            value: b"pl".to_vec(),
        }],
        is_migration: false,
        pre_images: None,
    })
    .expect("append");

    // LIKES adj entry
    mgr.append(&OplogEntry {
        ts: 1001,
        term: 1,
        index: 1,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 2,
            key: b"adj:LIKES:out:\x00\x00\x00\x00\x00\x00\x00\x01".to_vec(),
            value: b"pl".to_vec(),
        }],
        is_migration: false,
        pre_images: None,
    })
    .expect("append");

    // Node entry — not an adj key, should not match edge_types filter
    mgr.append(&make_entry(2, 1002, false)).expect("append");

    mgr.rotate().expect("seal");

    let filters = CdcFilters {
        edge_types: vec!["FOLLOWS".to_string()],
        ..Default::default()
    };
    let mut tailer = OplogTailer::new(dir.path(), ResumeToken::from_start(0));
    let batch = tailer.read_next(100, &filters).expect("read");

    assert_eq!(batch.len(), 1, "only FOLLOWS entry delivered");
    assert_eq!(batch[0].0.index, 0);
}

/// Tailer reads across segment rotation boundaries in order.
#[test]
fn tailer_reads_across_rotation_boundary() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut mgr = open_manager(dir.path());

    // Segment 1: entries 0-4
    for i in 0..5u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    mgr.rotate().expect("seal segment 1");

    // Segment 2: entries 5-9
    for i in 5..10u64 {
        mgr.append(&make_entry(i, 1000 + i, false)).expect("append");
    }
    mgr.rotate().expect("seal segment 2");

    let mut tailer = OplogTailer::new(dir.path(), ResumeToken::from_start(0));
    let all = tailer.read_next(100, &CdcFilters::default()).expect("read");

    assert_eq!(all.len(), 10, "all 10 entries across 2 segments");
    assert_eq!(all[0].0.index, 0);
    assert_eq!(all[9].0.index, 9);
}

// ── RaftNode-level integration test ───────────────────────────────────────────

/// After committing proposals through RaftNode, OplogTailer delivers the
/// corresponding oplog entries.
///
/// The oplog for shard 0 lives at `data_dir/oplog/0/`. After write proposals
/// are committed, the active segment is sealed (either via rotation or on
/// shutdown). The tailer then reads back all entries.
#[tokio::test(flavor = "multi_thread")]
async fn cdc_tailer_delivers_raft_proposals() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data_dir = dir.path().to_path_buf();

    let engine = open_engine(&data_dir);
    let node = RaftNode::single_node(Arc::clone(&engine))
        .await
        .expect("bootstrap");

    // Wait for leadership.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    let n_proposals = 5u64;
    for i in 1..=n_proposals {
        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: format!("cdc-key-{i}").into_bytes(),
                value: format!("cdc-val-{i}").into_bytes(),
            }],
            commit_ts: Timestamp::from_raw(2000 + i),
            start_ts: Timestamp::from_raw(2000 + i - 1),
            bypass_rate_limiter: false,
        };
        pipeline.propose_and_wait(&proposal).expect("propose");
    }

    // Shutdown flushes and seals the active oplog segment.
    node.shutdown().await.expect("shutdown");

    // The oplog dir for shard 0.
    let oplog_dir = data_dir.join("oplog").join("0");

    assert!(
        oplog_dir.exists(),
        "oplog/0 directory must exist after Raft writes"
    );

    // Tail the oplog from the start.
    let mut tailer = OplogTailer::new(&oplog_dir, ResumeToken::from_start(0));
    let batch = tailer
        .read_next(1000, &CdcFilters::default())
        .expect("read oplog entries");

    assert!(
        !batch.is_empty(),
        "tailer must deliver at least 1 entry (bootstrap + proposals)"
    );

    // The Raft oplog stores entries as OplogOp::RaftEntry { data } — each entry
    // carries the serialized openraft log entry (proposal + Raft metadata).
    // CDC consumers decode RaftEntry ops to extract the mutations.
    //
    // Verify that the expected number of RaftEntry ops is present (at least
    // n_proposals entries, plus openraft bootstrap/membership entries).
    let raft_entry_count = batch
        .iter()
        .filter(|(e, _)| {
            e.ops
                .iter()
                .any(|op| matches!(op, OplogOp::RaftEntry { .. }))
        })
        .count();

    // openraft adds: 1 bootstrap (blank) + 1+ membership entries.
    // We wrote n_proposals, so total should be at least n_proposals + 1.
    assert!(
        raft_entry_count >= n_proposals as usize,
        "expected ≥{n_proposals} RaftEntry ops in oplog, got {raft_entry_count}"
    );

    // All entries must have monotonically increasing Raft indices.
    let indices: Vec<u64> = batch.iter().map(|(e, _)| e.index).collect();
    for w in indices.windows(2) {
        assert!(w[0] < w[1], "oplog entries must be in index order");
    }
}

/// CDC resume: read first half via tailer, then resume and get second half.
#[tokio::test(flavor = "multi_thread")]
async fn cdc_tailer_resume_after_partial_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data_dir = dir.path().to_path_buf();

    let engine = open_engine(&data_dir);
    let node = RaftNode::single_node(Arc::clone(&engine))
        .await
        .expect("bootstrap");

    tokio::time::sleep(Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    for i in 1u64..=10 {
        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: format!("resume-key-{i}").into_bytes(),
                value: format!("resume-val-{i}").into_bytes(),
            }],
            commit_ts: Timestamp::from_raw(3000 + i),
            start_ts: Timestamp::from_raw(3000 + i - 1),
            bypass_rate_limiter: false,
        };
        pipeline.propose_and_wait(&proposal).expect("propose");
    }

    node.shutdown().await.expect("shutdown");

    let oplog_dir = data_dir.join("oplog").join("0");

    // First read: get all entries and find a midpoint.
    let mut tailer = OplogTailer::new(&oplog_dir, ResumeToken::from_start(0));
    let all = tailer
        .read_next(1000, &CdcFilters::default())
        .expect("full read");
    assert!(!all.is_empty(), "must have entries");

    let mid = all.len() / 2;
    let resume_token = all[mid - 1].1.clone();

    // Second read from resume: must get exactly the second half.
    let mut tailer2 = OplogTailer::new(&oplog_dir, resume_token);
    let second_half = tailer2
        .read_next(1000, &CdcFilters::default())
        .expect("resume read");

    assert_eq!(
        second_half.len(),
        all.len() - mid,
        "resume must deliver exactly the second half"
    );

    // Entries in second half must match the tail of the full read.
    for (i, (entry, _)) in second_half.iter().enumerate() {
        assert_eq!(
            entry.index,
            all[mid + i].0.index,
            "resumed entry at position {i} must match full read"
        );
    }
}

// ── G058: CDC edge_type filter works through Raft path ──────────────

#[tokio::test(flavor = "multi_thread")]
async fn cdc_filter_edge_type_through_raft() {
    // Regression test for G058: Raft proposals produce decoded
    // OplogOp::Insert/Delete/Merge alongside RaftEntry, enabling
    // server-side CDC filtering by edge_type.
    let dir = tempfile::tempdir().expect("tempdir");
    let data_dir = dir.path().to_path_buf();

    let engine = open_engine(&data_dir);
    let node = RaftNode::single_node(Arc::clone(&engine))
        .await
        .expect("bootstrap");

    tokio::time::sleep(Duration::from_millis(500)).await;

    let pipeline = node.pipeline();
    let id_gen = ProposalIdGenerator::new();

    // Proposal 1: edge mutation on adj:FOLLOWS key
    pipeline
        .propose_and_wait(&RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Merge {
                partition: PartitionId::Adj,
                key: b"adj:FOLLOWS:out:00000001".to_vec(),
                operand: b"add:42".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(3001),
            start_ts: Timestamp::from_raw(3000),
            bypass_rate_limiter: false,
        })
        .expect("propose FOLLOWS edge");

    // Proposal 2: edge mutation on adj:LIKES key
    pipeline
        .propose_and_wait(&RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Merge {
                partition: PartitionId::Adj,
                key: b"adj:LIKES:out:00000001".to_vec(),
                operand: b"add:99".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(3003),
            start_ts: Timestamp::from_raw(3002),
            bypass_rate_limiter: false,
        })
        .expect("propose LIKES edge");

    // Proposal 3: node mutation (not an edge)
    pipeline
        .propose_and_wait(&RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:0:1".to_vec(),
                value: b"data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(3005),
            start_ts: Timestamp::from_raw(3004),
            bypass_rate_limiter: false,
        })
        .expect("propose node write");

    node.shutdown().await.expect("shutdown");

    let oplog_dir = data_dir.join("oplog").join("0");

    // Filter: only FOLLOWS edges
    let mut tailer = OplogTailer::new(&oplog_dir, ResumeToken::from_start(0));
    let follows_only = tailer
        .read_next(
            1000,
            &CdcFilters {
                edge_types: vec!["FOLLOWS".to_string()],
                ..Default::default()
            },
        )
        .expect("read with FOLLOWS filter");

    // Should find exactly 1 entry with FOLLOWS adj key
    let follows_count = follows_only
        .iter()
        .filter(|(e, _)| {
            e.ops.iter().any(|op| match op {
                OplogOp::Merge { key, .. } => key.starts_with(b"adj:FOLLOWS:"),
                _ => false,
            })
        })
        .count();
    assert_eq!(
        follows_count, 1,
        "filter should deliver exactly 1 FOLLOWS entry, got {follows_count}. Entries: {follows_only:?}"
    );

    // The LIKES and Node entries should NOT be in the filtered result
    let has_likes = follows_only.iter().any(|(e, _)| {
        e.ops.iter().any(|op| match op {
            OplogOp::Merge { key, .. } => key.starts_with(b"adj:LIKES:"),
            _ => false,
        })
    });
    assert!(!has_likes, "LIKES edges should be filtered out");
}
