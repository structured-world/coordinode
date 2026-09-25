//! Integration test: the Raft log keeps every entry some partition tree has
//! not durably applied, whatever openraft asks to purge.
//!
//! Scenario: entries are applied through the state machine but no tree has
//! flushed them. openraft, which considers them applied, asks the log store
//! to purge. The purge must stop at the entries every tree durably records
//! as applied (none yet), so that after a power cut the log still holds the
//! entries the trees lost, the state machine resumes before them, and their
//! re-delivery restores the data.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use coordinode_core::txn::proposal::{Mutation, PartitionId, ProposalId, RaftProposal};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_raft::storage::{
    CommittedLeaderId, CoordinodeStateMachine, Entry, LogStore, Request,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::PowerRig;
use openraft::entry::RaftEntry;
use openraft::storage::{IOFlushed, RaftLogReader, RaftLogStorage, RaftStateMachine};

fn open(rig: &PowerRig) -> (Arc<StorageEngine>, Arc<TimestampOracle>) {
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(
        StorageEngine::open_with_oracle(&rig.config(), Arc::clone(&oracle)).expect("open engine"),
    );
    (engine, oracle)
}

fn entry(index: u64, commit_ts: u64) -> Entry {
    let proposal = RaftProposal {
        id: ProposalId::from_raw(index),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:1:{index}").into_bytes(),
            value: format!("v{index}").into_bytes(),
        }],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    };
    Entry::new_normal(
        openraft::LogId::new(
            CommittedLeaderId {
                term: 1,
                node_id: 0,
            },
            index,
        ),
        Request::single(proposal),
    )
}

async fn apply(sm: &mut CoordinodeStateMachine, entries: Vec<Entry>) {
    let stream = futures_util::stream::iter(entries.into_iter().map(|e| Ok((e, None))));
    sm.apply(stream).await.expect("apply");
}

#[tokio::test]
async fn purge_keeps_entries_no_tree_has_flushed_and_replay_restores_them() {
    let rig = PowerRig::new();
    let base_ts;
    {
        let (engine, oracle) = open(&rig);
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()))
            .expect("open state machine");
        base_ts = oracle.current().as_raw() + 1_000;
        let entries: Vec<Entry> = (1..=5).map(|i| entry(i, base_ts + i)).collect();
        log.append(entries.clone(), IOFlushed::noop())
            .await
            .expect("append");
        apply(&mut sm, entries).await;

        // openraft would now purge everything it applied.
        log.purge(openraft::LogId::new(
            CommittedLeaderId {
                term: 1,
                node_id: 0,
            },
            5,
        ))
        .await
        .expect("purge");
        let kept = log.try_get_log_entries(1..=5).await.expect("read log");
        assert_eq!(kept.len(), 5, "no entry is durable in every tree yet");

        drop(sm);
        drop(log);
        rig.cut(engine);
    }

    let (engine, oracle) = open(&rig);
    let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle))
        .expect("reopen state machine");
    let (applied, _) = sm.applied_state().await.expect("applied state");
    assert_eq!(
        applied, None,
        "the state machine resumes before every lost entry"
    );

    // What openraft re-delivers after the restart.
    let redelivered = log.try_get_log_entries(1..=5).await.expect("read log");
    assert_eq!(redelivered.len(), 5);
    apply(&mut sm, redelivered).await;
    for i in 1..=5u64 {
        assert_eq!(
            engine
                .get(Partition::Node, format!("node:1:{i}").as_bytes())
                .expect("get")
                .as_deref(),
            Some(format!("v{i}").as_bytes()),
            "entry {i} restored from the retained log"
        );
    }
}
