//! Integration tests: Oplog as Raft log storage (R075).
//!
//! Tests verify that:
//!   - `LogStore::open()` creates `data_dir/raft_oplog/`
//!   - `append()` writes to oplog; `get_log_state()` returns O(1) `last_log_id`
//!   - `last_log_id` and `last_purged` survive close + reopen (Partition::Raft)
//!   - `purge()` updates `last_purged` and filters entries in `try_get_log_entries`
//!   - `truncate_after()` keeps only the requested prefix
//!   - `get_log_reader()` returns a clone that reads the same entries

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use coordinode_core::txn::proposal::{Mutation, PartitionId, ProposalId, RaftProposal};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_raft::storage::{CommittedLeaderId, Entry, LogId, LogStore, Request, TypeConfig};
use coordinode_storage::engine::config::StorageConfig;
use coordinode_storage::engine::core::StorageEngine;
use openraft::entry::RaftEntry;
use openraft::storage::{IOFlushed, LogState, RaftLogReader, RaftLogStorage};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn open_engine(dir: &std::path::Path) -> Arc<StorageEngine> {
    let config = StorageConfig::new(dir);
    Arc::new(StorageEngine::open(&config).expect("open engine"))
}

fn make_entry(index: u64, term: u64) -> Entry {
    let proposal = RaftProposal {
        id: ProposalId::from_raw(index),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:1:{index}").into_bytes(),
            value: format!("val-{index}").into_bytes(),
        }],
        commit_ts: Timestamp::from_raw(1000 + index),
        start_ts: Timestamp::from_raw(1000 + index - 1),
        bypass_rate_limiter: false,
    };
    let log_id = openraft::LogId::new(CommittedLeaderId { term, node_id: 0 }, index);
    Entry::new_normal(log_id, Request::single(proposal))
}

fn make_log_id(index: u64, term: u64) -> LogId {
    openraft::LogId::new(CommittedLeaderId { term, node_id: 0 }, index)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// `LogStore::open()` creates `data_dir/raft_oplog/` on first open.
#[test]
fn logstore_creates_oplog_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());

    LogStore::open(Arc::clone(&engine)).expect("open logstore");

    let oplog_dir = engine.data_dir().join("raft_oplog");
    assert!(oplog_dir.exists(), "raft_oplog directory must be created");
    assert!(oplog_dir.is_dir(), "raft_oplog must be a directory");
}

/// Appended entries are readable and `get_log_state()` returns `last_log_id`.
#[tokio::test]
async fn logstore_append_and_read_back() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());
    let mut store = LogStore::open(Arc::clone(&engine)).expect("open");

    let entries: Vec<Entry> = (1..=5u64).map(|i| make_entry(i, 1)).collect();
    store
        .append(entries, IOFlushed::noop())
        .await
        .expect("append");

    let state: LogState<TypeConfig> = store.get_log_state().await.expect("get_log_state");
    assert_eq!(state.last_log_id.expect("last_log_id").index, 5);

    let loaded = store
        .try_get_log_entries(1u64..=5)
        .await
        .expect("try_get_log_entries");
    assert_eq!(loaded.len(), 5);
    assert_eq!(loaded[0].log_id.index, 1);
    assert_eq!(loaded[4].log_id.index, 5);
}

/// `last_log_id` is persisted in `Partition::Raft` and recovered after reopen — O(1).
#[tokio::test]
async fn logstore_last_log_id_survives_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    // First session: append 10 entries.
    {
        let engine = open_engine(dir.path());
        let mut store = LogStore::open(Arc::clone(&engine)).expect("open first");
        let entries: Vec<Entry> = (1..=10u64).map(|i| make_entry(i, 1)).collect();
        store
            .append(entries, IOFlushed::noop())
            .await
            .expect("append");
    }

    // Second session: last_log_id must be recovered from Partition::Raft.
    {
        let engine = open_engine(dir.path());
        let mut store = LogStore::open(Arc::clone(&engine)).expect("open second");
        let state: LogState<TypeConfig> = store.get_log_state().await.expect("get_log_state");
        assert_eq!(
            state.last_log_id.expect("last_log_id after reopen").index,
            10,
            "last_log_id must be 10 after reopen"
        );
    }
}

/// `purge()` updates `last_purged`; `try_get_log_entries` filters out purged entries.
#[tokio::test]
async fn logstore_purge_filters_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());
    let mut store = LogStore::open(Arc::clone(&engine)).expect("open");

    let entries: Vec<Entry> = (1..=5u64).map(|i| make_entry(i, 1)).collect();
    store
        .append(entries, IOFlushed::noop())
        .await
        .expect("append");

    // Purge up to index 3 (inclusive).
    store.purge(make_log_id(3, 1)).await.expect("purge");

    let state: LogState<TypeConfig> = store.get_log_state().await.expect("get_log_state");
    assert_eq!(
        state.last_purged_log_id.expect("last_purged").index,
        3,
        "last_purged_log_id must be 3"
    );

    // Only entries 4 and 5 must be visible.
    let visible = store
        .try_get_log_entries(0u64..=10)
        .await
        .expect("try_get_log_entries after purge");
    assert_eq!(visible.len(), 2, "only entries 4 and 5 should be visible");
    assert_eq!(visible[0].log_id.index, 4);
    assert_eq!(visible[1].log_id.index, 5);
}

/// `last_purged` is persisted and survives a reopen.
#[tokio::test]
async fn logstore_last_purged_survives_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let engine = open_engine(dir.path());
        let mut store = LogStore::open(Arc::clone(&engine)).expect("open first");
        let entries: Vec<Entry> = (1..=5u64).map(|i| make_entry(i, 1)).collect();
        store
            .append(entries, IOFlushed::noop())
            .await
            .expect("append");
        store.purge(make_log_id(3, 1)).await.expect("purge");
    }

    {
        let engine = open_engine(dir.path());
        let mut store = LogStore::open(Arc::clone(&engine)).expect("open second");
        let state: LogState<TypeConfig> = store.get_log_state().await.expect("get_log_state");
        assert_eq!(
            state
                .last_purged_log_id
                .expect("last_purged after reopen")
                .index,
            3,
            "last_purged must be 3 after reopen"
        );
    }
}

/// `truncate_after(Some(log_id))` keeps only entries up to the given index.
#[tokio::test]
async fn logstore_truncate_after_keeps_prefix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());
    let mut store = LogStore::open(Arc::clone(&engine)).expect("open");

    let entries: Vec<Entry> = (1..=6u64).map(|i| make_entry(i, 1)).collect();
    store
        .append(entries, IOFlushed::noop())
        .await
        .expect("append");

    // Keep 1, 2, 3 — delete 4, 5, 6.
    store
        .truncate_after(Some(make_log_id(3, 1)))
        .await
        .expect("truncate_after");

    let state: LogState<TypeConfig> = store.get_log_state().await.expect("get_log_state");
    assert_eq!(
        state.last_log_id.expect("last_log_id after truncate").index,
        3
    );

    let remaining = store
        .try_get_log_entries(0u64..=10)
        .await
        .expect("try_get_log_entries after truncate");
    assert_eq!(remaining.len(), 3, "only entries 1-3 should remain");
    assert_eq!(remaining[0].log_id.index, 1);
    assert_eq!(remaining[2].log_id.index, 3);
}

/// `truncate_after(None)` removes all entries.
#[tokio::test]
async fn logstore_truncate_after_none_clears_all() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());
    let mut store = LogStore::open(Arc::clone(&engine)).expect("open");

    let entries: Vec<Entry> = (1..=4u64).map(|i| make_entry(i, 1)).collect();
    store
        .append(entries, IOFlushed::noop())
        .await
        .expect("append");

    store
        .truncate_after(None)
        .await
        .expect("truncate_after None");

    let state: LogState<TypeConfig> = store.get_log_state().await.expect("get_log_state");
    assert!(
        state.last_log_id.is_none(),
        "last_log_id must be None after full truncation"
    );

    let remaining = store
        .try_get_log_entries(0u64..=10)
        .await
        .expect("try_get_log_entries after full truncate");
    assert!(
        remaining.is_empty(),
        "all entries must be gone after truncate_after(None)"
    );
}

/// `get_log_reader()` returns a cheap clone sharing the same state.
#[tokio::test]
async fn logstore_get_log_reader_shares_state() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());
    let mut store = LogStore::open(Arc::clone(&engine)).expect("open");

    let entries: Vec<Entry> = (1..=3u64).map(|i| make_entry(i, 1)).collect();
    store
        .append(entries, IOFlushed::noop())
        .await
        .expect("append");

    let mut reader = store.get_log_reader().await;

    let loaded = reader
        .try_get_log_entries(1u64..=3)
        .await
        .expect("reader try_get_log_entries");
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded[0].log_id.index, 1);
    assert_eq!(loaded[2].log_id.index, 3);
}

/// Appending then re-reading across the full range works with partial range queries.
#[tokio::test]
async fn logstore_partial_range_query() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(dir.path());
    let mut store = LogStore::open(Arc::clone(&engine)).expect("open");

    let entries: Vec<Entry> = (1..=10u64).map(|i| make_entry(i, 1)).collect();
    store
        .append(entries, IOFlushed::noop())
        .await
        .expect("append");

    let mid = store
        .try_get_log_entries(4u64..=7)
        .await
        .expect("partial range");
    assert_eq!(mid.len(), 4);
    assert_eq!(mid[0].log_id.index, 4);
    assert_eq!(mid[3].log_id.index, 7);
}
