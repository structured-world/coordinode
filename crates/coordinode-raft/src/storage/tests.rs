use super::*;
use coordinode_core::txn::proposal::PartitionId;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

fn test_engine() -> (tempfile::TempDir, Arc<StorageEngine>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    (dir, engine)
}

fn log_id(term: u64, index: u64) -> openraft::type_config::alias::LogIdOf<TypeConfig> {
    openraft::LogId::new(CommittedLeaderId { term, node_id: 0 }, index)
}

// -- LogStore --

#[tokio::test]
async fn log_store_save_and_read_vote() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();

    // No vote initially
    assert!(store.read_vote().await.unwrap().is_none());

    // Save a vote: term=1, node_id=42
    let vote = Vote::new(1, 42);
    store.save_vote(&vote).await.unwrap();

    // Read it back
    let loaded = store.read_vote().await.unwrap().unwrap();
    assert_eq!(loaded, vote);
}

#[tokio::test]
async fn log_store_append_and_read_entries() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();

    // Create entries
    let entries = vec![
        make_entry(1, 1, "first"),
        make_entry(2, 1, "second"),
        make_entry(3, 1, "third"),
    ];

    store.append(entries, IOFlushed::noop()).await.unwrap();

    // Read back
    let loaded = store.try_get_log_entries(1..=3).await.unwrap();
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded[0].log_id.index, 1);
    assert_eq!(loaded[2].log_id.index, 3);
}

#[tokio::test]
async fn log_store_get_log_state() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();

    // Empty state
    let state = store.get_log_state().await.unwrap();
    assert!(state.last_log_id.is_none());

    // Add entries
    let entries = vec![make_entry(1, 1, "a"), make_entry(2, 1, "b")];
    store.append(entries, IOFlushed::noop()).await.unwrap();

    let state = store.get_log_state().await.unwrap();
    assert_eq!(state.last_log_id.unwrap().index, 2);
}

#[tokio::test]
async fn log_store_truncate_after() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();

    let entries = vec![
        make_entry(1, 1, "a"),
        make_entry(2, 1, "b"),
        make_entry(3, 1, "c"),
    ];
    store.append(entries, IOFlushed::noop()).await.unwrap();

    // Truncate after index 1 (keep 1, delete 2 and 3)
    store.truncate_after(Some(log_id(1, 1))).await.unwrap();

    let remaining = store.try_get_log_entries(0..=10).await.unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].log_id.index, 1);
}

/// A change stream reports each entry's commit timestamp and the term of the
/// leader that wrote it; both come from the oplog record, so a record that
/// left them at zero sent every CDC event with `ts = 0, term = 0`.
#[tokio::test]
async fn oplog_record_carries_commit_ts_and_term() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();
    store
        .append(vec![make_entry(5, 3, "a")], IOFlushed::noop())
        .await
        .unwrap();

    let records = store.oplog.lock().unwrap().read_range(5, 6).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].ts, 1005, "the proposal's commit timestamp");
    assert_eq!(records[0].term, 3, "the term of the entry's leader");
}

#[tokio::test]
async fn log_store_purge() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(Arc::clone(&engine)).unwrap();

    let entries = vec![
        make_entry(1, 1, "a"),
        make_entry(2, 1, "b"),
        make_entry(3, 1, "c"),
    ];
    store.append(entries, IOFlushed::noop()).await.unwrap();
    // Every tree durably holds entries 0..=2.
    engine.reset_raft_coverage(3, &[]).unwrap();

    // Purge up to index 2 (delete 1 and 2, keep 3)
    store.purge(log_id(1, 2)).await.unwrap();

    let remaining = store.try_get_log_entries(0..=10).await.unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].log_id.index, 3);
}

#[tokio::test]
async fn log_store_purge_stops_at_what_every_tree_holds() {
    // openraft may ask to forget an entry a tree has not flushed yet; the
    // entry is then its only copy, so the purge stops below it and records
    // the entry it actually stopped at.
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(Arc::clone(&engine)).unwrap();
    let entries = vec![
        make_entry(1, 1, "a"),
        make_entry(2, 1, "b"),
        make_entry(3, 1, "c"),
    ];
    store.append(entries, IOFlushed::noop()).await.unwrap();
    engine.reset_raft_coverage(2, &[]).unwrap();

    store.purge(log_id(1, 3)).await.unwrap();

    let remaining = store.try_get_log_entries(0..=10).await.unwrap();
    assert_eq!(
        remaining.iter().map(|e| e.log_id.index).collect::<Vec<_>>(),
        vec![2, 3],
        "entries at or above the durable floor stay"
    );
    let state = store.get_log_state().await.unwrap();
    assert_eq!(state.last_purged_log_id, Some(log_id(1, 1)));
}

#[tokio::test]
async fn log_store_purge_keeps_what_the_latest_checkpoint_lacks() {
    // A partition rebuild from the latest checkpoint replays the log from the
    // oldest entry the checkpoint's trees lack; purging past it would leave
    // the rebuild nothing to replay, even though every live tree holds it.
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(Arc::clone(&engine)).unwrap();
    let entries = vec![
        make_entry(1, 1, "a"),
        make_entry(2, 1, "b"),
        make_entry(3, 1, "c"),
    ];
    store.append(entries, IOFlushed::noop()).await.unwrap();
    engine.reset_raft_coverage(4, &[]).unwrap();
    engine.set_raft_log_keep_from(2);

    store.purge(log_id(1, 3)).await.unwrap();

    let remaining = store.try_get_log_entries(0..=10).await.unwrap();
    assert_eq!(
        remaining.iter().map(|e| e.log_id.index).collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[tokio::test]
async fn a_checkpoint_raft_floor_is_the_lowest_base_of_its_trees() {
    let (_dir, engine) = test_engine();
    engine.reset_raft_coverage(7, &[]).unwrap();
    let ckpt_root = tempfile::tempdir().unwrap();
    let ckpt = ckpt_root.path().join("c");
    engine.create_checkpoint(&ckpt).unwrap();
    assert_eq!(StorageEngine::checkpoint_raft_floor(&ckpt).unwrap(), 7);
}

#[tokio::test]
async fn log_store_purge_with_nothing_durable_keeps_everything() {
    // A fresh record covers nothing, so no entry may go.
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(Arc::clone(&engine)).unwrap();
    let entries = vec![make_entry(1, 1, "a"), make_entry(2, 1, "b")];
    store.append(entries, IOFlushed::noop()).await.unwrap();
    engine.reset_raft_coverage(0, &[]).unwrap();

    store.purge(log_id(1, 2)).await.unwrap();

    assert_eq!(store.try_get_log_entries(0..=10).await.unwrap().len(), 2);
    let state = store.get_log_state().await.unwrap();
    assert_eq!(state.last_purged_log_id, None);
}

#[tokio::test]
async fn log_store_committed_roundtrip() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();

    // No committed initially
    assert!(store.read_committed().await.unwrap().is_none());

    store.save_committed(Some(log_id(1, 5))).await.unwrap();

    let loaded = store.read_committed().await.unwrap().unwrap();
    assert_eq!(loaded.index, 5);
}

// -- CoordinodeStateMachine --

#[tokio::test]
async fn state_machine_initial_state() {
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(engine).expect("open state machine");

    let (applied, membership) = sm.applied_state().await.unwrap();
    assert!(applied.is_none());
    assert!(membership.log_id().is_none());
}

#[tokio::test]
async fn state_machine_resumes_from_the_lowest_covered_prefix() {
    // The applied position comes from the trees' coverage bases: the entry
    // every tree durably holds, not a key one of them happened to flush.
    let (_dir, engine) = test_engine();
    let resume = log_id(3, 9);
    engine
        .reset_raft_coverage(10, &rmp_serde::to_vec(&resume).unwrap())
        .unwrap();

    let mut sm = CoordinodeStateMachine::new(engine).expect("open state machine");
    let (applied, _) = sm.applied_state().await.unwrap();
    assert_eq!(applied, Some(resume));
    assert_eq!(sm.applied_index(), 9);
}

#[test]
fn a_store_that_applied_entries_without_coverage_is_refused() {
    // A release before apply coverage kept only its own applied-index key.
    // Nothing proves which entries each tree holds, so the open refuses
    // rather than guessing.
    let (_dir, engine) = test_engine();
    engine
        .put(
            Partition::Schema,
            KEY_SM_APPLIED,
            &rmp_serde::to_vec(&log_id(1, 42)).unwrap(),
        )
        .unwrap();

    let err = CoordinodeStateMachine::new(Arc::clone(&engine))
        .err()
        .expect("legacy store must be refused");
    assert!(
        err.to_string().contains("without an apply-coverage record"),
        "got: {err}"
    );
    assert!(
        !engine.raft_coverage().unwrap().has_record(),
        "the refused store is left untouched"
    );
}

/// Readers waiting on the applied watermark (causal reads with an
/// `after_index`, the change stream's bound) must see an installed snapshot
/// at once: on an idle cluster no later apply would ever move it.
#[tokio::test]
async fn installing_a_snapshot_publishes_its_applied_index() {
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(engine).expect("open state machine");
    let mut applied = sm.subscribe_applied();
    assert_eq!(*applied.borrow_and_update(), 0);

    let meta = SnapshotMeta {
        last_log_id: Some(log_id(2, 20)),
        last_membership: openraft::StoredMembership::default(),
    };
    sm.install_snapshot(&meta, std::io::Cursor::new(Vec::new()))
        .await
        .unwrap();

    assert!(applied.has_changed().unwrap(), "waiters are woken");
    assert_eq!(*applied.borrow(), 20);
    assert_eq!(sm.applied_index(), 20);
}

#[tokio::test]
async fn installing_a_snapshot_rebinds_every_tree_to_it() {
    // After the install every tree holds exactly the snapshot, so the
    // resume point is the snapshot and no older marker survives.
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open state machine");
    sm.apply_proposal(&node_put(1, 1000, b"pre-snapshot"), 5, 0)
        .unwrap();

    let snapshot_id = log_id(2, 20);
    let meta = SnapshotMeta {
        last_log_id: Some(snapshot_id),
        last_membership: openraft::StoredMembership::default(),
    };
    sm.install_snapshot(&meta, std::io::Cursor::new(Vec::new()))
        .await
        .unwrap();

    let coverage = engine.raft_coverage().unwrap();
    assert_eq!(
        coverage.resume_point(),
        Some((21, rmp_serde::to_vec(&snapshot_id).unwrap().as_slice()))
    );
    assert!(coverage.holds(Partition::Node, 5, 0), "below the snapshot");
    assert_eq!(coverage.skip_until(), 21, "the pre-snapshot marker is gone");

    let mut reopened = CoordinodeStateMachine::new(engine).expect("reopen");
    assert_eq!(reopened.applied_state().await.unwrap().0, Some(snapshot_id));
}

// -- Dedup tests --

fn node_put(id: u64, commit_ts: u64, value: &[u8]) -> RaftProposal {
    RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(id),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:1:{id}").into_bytes(),
            value: value.to_vec(),
        }],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    }
}

#[test]
fn dedup_skips_duplicate_proposal() {
    // Apply same proposal twice — second should be detected as duplicate
    let (_dir, engine) = test_engine();
    let sm = CoordinodeStateMachine::new(engine).expect("open state machine");

    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(42),
        mutations: vec![Mutation::Put {
            partition: coordinode_core::txn::proposal::PartitionId::Node,
            key: b"node:1:100".to_vec(),
            value: b"data".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(1000),
        start_ts: Timestamp::from_raw(999),
        bypass_rate_limiter: false,
    };

    // First apply: should return 1 mutation applied
    let r1 = sm.apply_proposal(&proposal, 1, 0).unwrap();
    assert_eq!(r1.mutations_applied, 1);

    // Second apply (same id + same size), carried by a later entry: should
    // return 0 (dedup)
    let r2 = sm.apply_proposal(&proposal, 2, 0).unwrap();
    assert_eq!(r2.mutations_applied, 0);
}

#[test]
fn dedup_allows_different_size_same_id() {
    // Same proposal ID but different payload size should re-apply
    // (represents a retry with modified payload)
    let (_dir, engine) = test_engine();
    let sm = CoordinodeStateMachine::new(engine).expect("open state machine");

    let proposal_v1 = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(42),
        mutations: vec![Mutation::Put {
            partition: coordinode_core::txn::proposal::PartitionId::Node,
            key: b"node:1:100".to_vec(),
            value: b"short".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(1000),
        start_ts: Timestamp::from_raw(999),
        bypass_rate_limiter: false,
    };

    let proposal_v2 = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(42),
        mutations: vec![Mutation::Put {
            partition: coordinode_core::txn::proposal::PartitionId::Node,
            key: b"node:1:100".to_vec(),
            value: b"much-longer-value-that-changes-size".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(1001),
        start_ts: Timestamp::from_raw(1000),
        bypass_rate_limiter: false,
    };

    // First version applied
    let r1 = sm.apply_proposal(&proposal_v1, 1, 0).unwrap();
    assert_eq!(r1.mutations_applied, 1);

    // Second version with different size: should NOT be deduped
    let r2 = sm.apply_proposal(&proposal_v2, 2, 0).unwrap();
    assert_eq!(r2.mutations_applied, 1);
}

#[test]
fn dedup_gc_removes_old_entries() {
    // Verify that dedup GC cleans old entries
    let (_dir, engine) = test_engine();
    let sm = CoordinodeStateMachine::new(engine).expect("open state machine");

    sm.apply_proposal(&node_put(1, 100, b"data"), 1, 0).unwrap();

    // Verify dedup map has 1 entry
    let dedup_len = sm.dedup.lock().unwrap().len();
    assert_eq!(dedup_len, 1, "dedup map should have 1 entry");

    // Manually set the entry's `seen` to old time to trigger GC
    {
        let mut dedup = sm.dedup.lock().unwrap();
        if let Some(entry) = dedup.get_mut(&1u64) {
            entry.seen = Instant::now() - Duration::from_secs(DEDUP_MAX_AGE_SECS + 1);
        }
        // Force last_gc to be old too
        *sm.last_dedup_gc.lock().unwrap() =
            Instant::now() - Duration::from_secs(DEDUP_GC_INTERVAL_SECS + 1);
    }

    // Apply another proposal to trigger GC
    sm.apply_proposal(&node_put(2, 200, b"data2"), 2, 0)
        .unwrap();

    // Old entry should have been GC'd, new one remains
    let dedup = sm.dedup.lock().unwrap();
    assert!(!dedup.contains_key(&1u64), "old entry should be GC'd");
    assert!(dedup.contains_key(&2u64), "new entry should remain");
}

// -- Crash recovery --

/// One Raft entry whose single proposal merges into two trees, both
/// non-idempotent: an edge on Adj and a degree delta on Counter.
fn two_tree_merge_entry(index: u64, commit_ts: u64) -> Entry {
    use openraft::entry::RaftEntry;
    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(index),
        mutations: vec![
            Mutation::Merge {
                partition: PartitionId::Adj,
                key: b"adj:R:out:1".to_vec(),
                operand: coordinode_storage::engine::merge::encode_add(7),
            },
            Mutation::Merge {
                partition: PartitionId::Counter,
                key: b"counter:degree:1".to_vec(),
                operand: coordinode_storage::engine::merge::encode_counter_delta(1),
            },
        ],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    };
    Entry::new_normal(log_id(1, index), Request::single(proposal))
}

async fn apply_entries(sm: &mut CoordinodeStateMachine, entries: Vec<Entry>) {
    let stream = futures_util::stream::iter(entries.into_iter().map(|e| Ok((e, None))));
    sm.apply(stream).await.expect("apply");
}

/// Install a peer's Counter copy (six increments, standing at entry 7) with
/// the disk refusing `op` after `skip` of them, cut the power, reopen, and
/// install the copy again. Returns whether the fault stopped the install.
async fn install_cut_at(op: lsm_tree::fs::FaultOp, skip: u64) -> bool {
    use coordinode_storage::engine::core::{PartitionCopy, RaftPosition};
    let rig = coordinode_test_fixtures::PowerRig::new();
    let key = b"counter:degree:1".to_vec();
    let copy = Arc::new(PartitionCopy {
        rows: vec![(key.clone(), 6i64.to_le_bytes().to_vec())],
        position: Some(RaftPosition {
            next: 7,
            payload: rmp_serde::to_vec(&log_id(1, 6)).expect("payload"),
        }),
    });
    let install = |engine: &Arc<StorageEngine>| {
        let (engine, copy) = (Arc::clone(engine), Arc::clone(&copy));
        tokio::task::spawn_blocking(move || {
            engine.install_partition(Partition::Counter, &copy, true)
        })
    };

    let (engine, oracle) = open_rig_engine(&rig);
    let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle))
        .expect("open state machine");
    apply_entries(&mut sm, vec![two_tree_merge_entry(1, 1000)]).await;
    engine.persist().expect("persist");
    rig.fail_from(op, skip);
    let stopped = install(&engine).await.expect("install task").is_err();
    drop(sm);
    rig.cut(engine);

    let (engine, oracle) = open_rig_engine(&rig);
    let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle))
        .expect("a store with an interrupted install opens");
    if stopped {
        assert!(
            engine
                .pending_rebuilds()
                .expect("intents")
                .contains(&Partition::Counter),
            "{op:?} {skip}: the interrupted install is known"
        );
    }
    install(&engine)
        .await
        .expect("install task")
        .expect("the copy installs again");
    assert_eq!(
        engine
            .get(Partition::Counter, &key)
            .expect("get")
            .as_deref(),
        Some(6i64.to_le_bytes().as_slice()),
        "{op:?} {skip}: exactly the copy"
    );
    assert!(engine.pending_rebuilds().expect("intents").is_empty());
    assert!(
        engine
            .raft_coverage()
            .expect("coverage")
            .holds(Partition::Counter, 6, 0)
    );
    drop(sm);
    stopped
}

#[tokio::test(flavor = "multi_thread")]
async fn a_power_cut_inside_a_partition_install_is_repaired_by_installing_again() {
    use lsm_tree::fs::FaultOp;
    for op in [FaultOp::Write, FaultOp::SyncAll, FaultOp::SyncData] {
        let mut skip = 0;
        while install_cut_at(op, skip).await {
            skip += 1;
        }
    }
}

#[test]
fn captures_a_crash_left_behind_are_cleared_on_open() {
    // A capture lives only as long as the copy or snapshot build that made
    // it; one a crash interrupted would otherwise hold its hard links, and
    // the disk space of every table they pin, for good.
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let partition_capture = dir.path().join("partition-capture").join("node-1");
    let snapshot_capture = dir.path().join("snapshot-capture").join("1");
    std::fs::create_dir_all(&partition_capture).expect("mkdir");
    std::fs::create_dir_all(&snapshot_capture).expect("mkdir");

    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    assert!(!partition_capture.exists());
    let _sm = CoordinodeStateMachine::new(engine).expect("open state machine");
    assert!(!snapshot_capture.exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rebuild_the_log_cannot_complete_changes_nothing() {
    // The checkpoint's Counter tree lacks entries 1..=3, and the log no
    // longer holds them. Rebuilding would lose them: the rebuild refuses
    // before it clears anything.
    let (_dir, engine) = test_engine();
    let store = LogStore::open(Arc::clone(&engine)).unwrap();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open");
    let ckpt_root = tempfile::tempdir().unwrap();
    let ckpt = ckpt_root.path().join("c");
    engine.create_checkpoint(&ckpt).unwrap();
    apply_entries(
        &mut sm,
        (1..=3).map(|i| two_tree_merge_entry(i, 1000 * i)).collect(),
    )
    .await;
    let key = b"counter:degree:1";
    let before = engine.get(Partition::Counter, key).unwrap();

    let log = Arc::clone(&store.oplog);
    let rebuilder = Arc::clone(&engine);
    let err = tokio::task::spawn_blocking(move || {
        rebuild_partition_from_checkpoint(&rebuilder, &log, &ckpt, Partition::Counter)
    })
    .await
    .expect("rebuild task")
    .expect_err("the log lacks what the checkpoint needs");
    assert!(err.to_string().contains("no longer holds"), "{err}");
    assert_eq!(engine.get(Partition::Counter, key).unwrap(), before);
    assert!(engine.pending_rebuilds().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_partition_installed_ahead_is_left_alone_until_the_applies_pass_it() {
    // A peer's copy of Counter standing at entry 6 replaces the local one
    // while the applies stand at 1. Entries 2..=6 are already in the copy:
    // applying them again would count them twice, and a fold at 5 must not
    // lower the copy's base below 7.
    use coordinode_storage::engine::core::{PartitionCopy, RaftPosition};
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open");
    apply_entries(&mut sm, vec![two_tree_merge_entry(1, 1000)]).await;

    let key = b"counter:degree:1".to_vec();
    let copy = PartitionCopy {
        rows: vec![(key.clone(), 6i64.to_le_bytes().to_vec())],
        position: Some(RaftPosition {
            next: 7,
            payload: rmp_serde::to_vec(&log_id(1, 6)).expect("payload"),
        }),
    };
    let installer = Arc::clone(&engine);
    tokio::task::spawn_blocking(move || {
        installer.install_partition(Partition::Counter, &copy, true)
    })
    .await
    .expect("install task")
    .expect("install");

    apply_entries(
        &mut sm,
        (2..=8).map(|i| two_tree_merge_entry(i, 1000 * i)).collect(),
    )
    .await;
    assert_eq!(
        engine
            .get(Partition::Counter, &key)
            .expect("get")
            .as_deref(),
        Some(8i64.to_le_bytes().as_slice()),
        "six from the copy, then entries 7 and 8"
    );
    let coverage = engine.raft_coverage().expect("coverage");
    assert!(
        coverage.holds(Partition::Counter, 6, 0),
        "the fold at 5 kept the copy's base"
    );
    assert!(coverage.holds(Partition::Counter, 8, 0));
}

#[tokio::test]
async fn a_snapshot_holds_exactly_the_entries_up_to_its_log_id() {
    // openraft builds a snapshot after `get_snapshot_builder` returns, while
    // the state machine keeps applying. A follower installs the snapshot and
    // then receives every entry past its log id, so an effect the data
    // already carried from a later entry would be applied twice.
    let (_dir, leader) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&leader)).expect("open");
    apply_entries(&mut sm, vec![two_tree_merge_entry(1, 1000)]).await;
    let mut builder = sm.get_snapshot_builder().await;
    apply_entries(&mut sm, vec![two_tree_merge_entry(2, 2000)]).await;
    let snapshot = builder.build_snapshot().await.expect("build");
    assert_eq!(snapshot.meta.last_log_id, Some(log_id(1, 1)));

    let (_dir2, follower) = test_engine();
    let mut fsm = CoordinodeStateMachine::new(Arc::clone(&follower)).expect("open follower");
    fsm.install_snapshot(&snapshot.meta, snapshot.snapshot)
        .await
        .expect("install");
    apply_entries(&mut fsm, vec![two_tree_merge_entry(2, 2000)]).await;
    let key = b"counter:degree:1";
    assert_eq!(
        follower.get(Partition::Counter, key).expect("follower get"),
        leader.get(Partition::Counter, key).expect("leader get"),
        "the follower counts each entry once"
    );
}

fn open_rig_engine(
    rig: &coordinode_test_fixtures::PowerRig,
) -> (
    Arc<StorageEngine>,
    Arc<coordinode_core::txn::timestamp::TimestampOracle>,
) {
    let oracle = Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new());
    let engine = Arc::new(
        StorageEngine::open_with_oracle(&rig.config(), Arc::clone(&oracle)).expect("open engine"),
    );
    (engine, oracle)
}

#[tokio::test]
async fn an_entry_one_tree_flushed_is_replayed_only_into_the_other() {
    // The entry reaches an SST in Adj while Counter's memtable is lost. On
    // restart openraft re-delivers it: Counter must get the delta, Adj must
    // not get the edge a second time. Resuming from a single applied-index
    // key gets one of the two wrong whichever tree flushed it.
    use lsm_tree::AbstractTree;

    let reference = {
        let rig = coordinode_test_fixtures::PowerRig::new();
        let (engine, oracle) = open_rig_engine(&rig);
        let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()))
            .expect("open");
        let ts = oracle.current().as_raw() + 1_000;
        apply_entries(&mut sm, vec![two_tree_merge_entry(1, ts)]).await;
        (
            engine.get(Partition::Adj, b"adj:R:out:1").unwrap(),
            engine.get(Partition::Counter, b"counter:degree:1").unwrap(),
        )
    };

    let rig = coordinode_test_fixtures::PowerRig::new();
    let ts;
    {
        let (engine, oracle) = open_rig_engine(&rig);
        let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()))
            .expect("open");
        ts = oracle.current().as_raw() + 1_000;
        apply_entries(&mut sm, vec![two_tree_merge_entry(1, ts)]).await;
        engine
            .tree(Partition::Adj)
            .unwrap()
            .flush_active_memtable(0)
            .unwrap();
        drop(sm);
        rig.cut(engine);
    }

    let (engine, oracle) = open_rig_engine(&rig);
    let mut sm =
        CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle)).expect("reopen");
    let (applied, _) = sm.applied_state().await.unwrap();
    assert_eq!(applied, None, "nothing is covered in every tree yet");
    apply_entries(&mut sm, vec![two_tree_merge_entry(1, ts)]).await;
    assert_eq!(
        (
            engine.get(Partition::Adj, b"adj:R:out:1").unwrap(),
            engine.get(Partition::Counter, b"counter:degree:1").unwrap(),
        ),
        reference,
        "each tree holds the entry exactly once after the re-delivery"
    );
}

// -- Crash campaign --

const RAFT_CAMPAIGN_ENTRIES: u64 = 24;

/// One campaign entry: a put naming it, an edge carrying its index and a +1
/// on a shared counter, across three trees.
fn raft_campaign_entry(index: u64, commit_ts: u64) -> Entry {
    use openraft::entry::RaftEntry;
    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(index),
        mutations: vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:00:c{index:04}").into_bytes(),
                value: b"x".to_vec(),
            },
            Mutation::Merge {
                partition: PartitionId::Adj,
                key: b"adj:C:out:1".to_vec(),
                operand: coordinode_storage::engine::merge::encode_add(index),
            },
            Mutation::Merge {
                partition: PartitionId::Counter,
                key: b"counter:campaign".to_vec(),
                operand: coordinode_storage::engine::merge::encode_counter_delta(1),
            },
        ],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    };
    Entry::new_normal(log_id(1, index), Request::single(proposal))
}

/// Append and apply entries one by one until the first failure, with a full
/// flush, a flush of one tree alone, folds (every few entries under test) and
/// log purges on the way. Returns the entries applied.
async fn run_raft_campaign(
    engine: &Arc<StorageEngine>,
    log: &mut LogStore,
    sm: &mut CoordinodeStateMachine,
    base_ts: u64,
) -> Vec<u64> {
    use lsm_tree::AbstractTree;
    let mut applied = Vec::new();
    for i in 1..=RAFT_CAMPAIGN_ENTRIES {
        let entry = raft_campaign_entry(i, base_ts + i);
        if log
            .append(vec![entry.clone()], IOFlushed::noop())
            .await
            .is_err()
        {
            return applied;
        }
        let stream = futures_util::stream::iter(std::iter::once(Ok((entry, None))));
        if sm.apply(stream).await.is_err() {
            return applied;
        }
        applied.push(i);
        let step = match i {
            7 => engine.persist().map_err(|e| e.to_string()),
            11 => engine
                .tree(Partition::Adj)
                .map_err(|e| e.to_string())
                .and_then(|t| {
                    t.flush_active_memtable(0)
                        .map_err(|e| e.to_string())
                        .map(|_| ())
                }),
            13 | 19 => log.purge(log_id(1, i)).await.map_err(|e| e.to_string()),
            _ => Ok(()),
        };
        if step.is_err() {
            return applied;
        }
    }
    applied
}

fn check_raft_campaign(engine: &StorageEngine, applied: &[u64], label: &str) {
    let present: Vec<u64> = (1..=RAFT_CAMPAIGN_ENTRIES)
        .filter(|i| {
            engine
                .get(Partition::Node, format!("node:00:c{i:04}").as_bytes())
                .unwrap()
                .is_some()
        })
        .collect();
    for i in applied {
        assert!(present.contains(i), "{label}: applied entry {i} lost");
    }
    let counter = engine
        .get(Partition::Counter, b"counter:campaign")
        .unwrap()
        .map(|v| coordinode_storage::engine::merge::decode_counter(&v).unwrap())
        .unwrap_or(0);
    assert_eq!(
        counter,
        present.len() as i64,
        "{label}: each present entry counted once (present {present:?})"
    );
    let edges: Vec<u64> = engine
        .get(Partition::Adj, b"adj:C:out:1")
        .unwrap()
        .map(|v| {
            coordinode_core::graph::edge::PostingList::from_bytes(&v)
                .unwrap()
                .iter()
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(edges, present, "{label}: the edges are the present entries");
}

/// Reopen after the cut the way openraft does: resume at the state
/// machine's applied position and re-deliver every log entry after it.
async fn recover_raft(rig: &coordinode_test_fixtures::PowerRig) -> Arc<StorageEngine> {
    let (engine, oracle) = open_rig_engine(rig);
    let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle))
        .expect("reopen state machine");
    let (applied, _) = sm.applied_state().await.unwrap();
    let from = applied.map_or(0, |id| id.index + 1);
    let redelivered = log.try_get_log_entries(from..).await.unwrap();
    let stream = futures_util::stream::iter(redelivered.into_iter().map(|e| Ok((e, None))));
    sm.apply(stream).await.expect("re-apply");
    drop(sm);
    drop(log);
    engine
}

async fn raft_cut_at(op: lsm_tree::fs::FaultOp, k: u64) -> bool {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let applied;
    {
        let (engine, oracle) = open_rig_engine(&rig);
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        let mut sm =
            CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(Arc::clone(&oracle)))
                .expect("open state machine");
        let base_ts = oracle.current().as_raw() + 1_000;
        rig.fail_from(op, k);
        applied = run_raft_campaign(&engine, &mut log, &mut sm, base_ts).await;
        drop(sm);
        drop(log);
        rig.cut(engine);
    }
    for round in 0..2 {
        let engine = recover_raft(&rig).await;
        check_raft_campaign(
            &engine,
            &applied,
            &format!("{op:?} cut at {k}, open {round}"),
        );
        rig.cut(engine);
    }
    (applied.len() as u64) < RAFT_CAMPAIGN_ENTRIES
}

/// A power cut at every operation of one kind the Raft workload performs,
/// cut points run concurrently in windows. Returns how many were reached.
async fn raft_cut_at_every(op: lsm_tree::fs::FaultOp) -> u64 {
    let window = std::thread::available_parallelism().map_or(4, |n| n.get() as u64);
    let mut reached = 0;
    let mut start = 0;
    loop {
        let handles: Vec<_> = (start..start + window)
            .map(|k| tokio::spawn(raft_cut_at(op, k)))
            .collect();
        let mut window_hits = 0;
        for handle in handles {
            if handle.await.expect("cut point task") {
                window_hits += 1;
            }
        }
        reached += window_hits;
        if window_hits < window {
            return reached;
        }
        start += window;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_power_cut_at_any_sync_leaves_every_raft_entry_once() {
    // Log appends, table and manifest publishes, folds and purges all end in
    // a full sync, so this walks the Raft path's whole durability order.
    assert!(raft_cut_at_every(lsm_tree::fs::FaultOp::SyncAll).await > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_power_cut_at_any_write_leaves_every_raft_entry_once() {
    assert!(raft_cut_at_every(lsm_tree::fs::FaultOp::Write).await > 0);
}

// -- Config --

#[test]
fn default_config_values() {
    let config = default_raft_config();
    // default_raft_config honours the COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS
    // escape hatch (set in CI); assert the branch matching the current env.
    if std::env::var_os("COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS").is_some() {
        assert_eq!(config.heartbeat_interval, 150);
        assert_eq!(config.election_timeout_min, 1500);
        assert_eq!(config.election_timeout_max, 3000);
    } else {
        assert_eq!(config.heartbeat_interval, 150);
        assert_eq!(config.election_timeout_min, 300);
        assert_eq!(config.election_timeout_max, 600);
    }
    assert_eq!(config.max_payload_entries, 300);
}

// -- Purge persistence tests --

#[tokio::test]
async fn log_store_purge_persists_last_purged_log_id() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(Arc::clone(&engine)).unwrap();

    let entries = vec![
        make_entry(1, 1, "a"),
        make_entry(2, 1, "b"),
        make_entry(3, 1, "c"),
    ];
    store.append(entries, IOFlushed::noop()).await.unwrap();
    engine.reset_raft_coverage(3, &[]).unwrap();

    // Initially no purge
    let state = store.get_log_state().await.unwrap();
    assert!(state.last_purged_log_id.is_none(), "no purge initially");

    // Purge up to index 2
    store.purge(log_id(1, 2)).await.unwrap();

    // Verify purge tracked in-session
    let state = store.get_log_state().await.unwrap();
    assert_eq!(
        state.last_purged_log_id.unwrap().index,
        2,
        "last_purged_log_id should be 2 after purge"
    );
}

#[tokio::test]
async fn log_store_purge_survives_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    let config = || {
        StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            &path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )])
    };

    // Step 1: write + purge
    {
        let engine = Arc::new(StorageEngine::open(&config()).expect("open"));
        let mut store = LogStore::open(Arc::clone(&engine)).unwrap();

        let entries = vec![
            make_entry(1, 1, "a"),
            make_entry(2, 1, "b"),
            make_entry(3, 1, "c"),
        ];
        store.append(entries, IOFlushed::noop()).await.unwrap();
        engine.reset_raft_coverage(3, &[]).unwrap();
        store.purge(log_id(1, 2)).await.unwrap();
    }

    // Step 2: reopen, verify purge state persisted
    {
        let engine = Arc::new(StorageEngine::open(&config()).expect("reopen"));
        let mut store = LogStore::open(engine).unwrap();

        let state = store.get_log_state().await.unwrap();
        assert_eq!(
            state.last_purged_log_id.unwrap().index,
            2,
            "last_purged_log_id should survive reopen"
        );
        // Only entry 3 should remain
        assert_eq!(
            state.last_log_id.unwrap().index,
            3,
            "entry 3 should survive purge"
        );
    }
}

// -- Snapshot persistence tests --

#[tokio::test]
async fn snapshot_build_persists_to_storage() {
    let (_dir, engine) = test_engine();

    // Write some data
    engine
        .put(Partition::Node, b"node:0:1", b"alice")
        .expect("put");

    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open state machine");

    // Set applied state so snapshot has a valid log_id
    *sm.last_applied.lock().unwrap() = Some(log_id(1, 5));

    // Build snapshot via the SnapshotBuilder
    let mut builder = sm.get_snapshot_builder().await;
    let snap = builder.build_snapshot().await.unwrap();

    assert!(snap.meta.last_log_id.is_some());
    assert_eq!(snap.meta.last_log_id.unwrap().index, 5);

    // Verify snapshot was persisted to CoordiNode storage (build_snapshot writes it)
    let snap_meta = engine.get(Partition::Schema, KEY_SNAPSHOT_META).unwrap();
    assert!(snap_meta.is_some(), "snapshot meta should be persisted");

    let snap_data = engine.get(Partition::Schema, KEY_SNAPSHOT_DATA).unwrap();
    assert!(snap_data.is_some(), "snapshot data should be persisted");
    let data = snap_data.unwrap();
    assert!(data.len() > 10, "snapshot data should be non-empty");
    assert_eq!(&data[..4], b"CNSN", "snapshot data should have CNSN magic");
}

#[tokio::test]
async fn snapshot_survives_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    let config = || {
        StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            &path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )])
    };

    // Step 1: build snapshot
    {
        let engine = Arc::new(StorageEngine::open(&config()).expect("open"));
        engine
            .put(Partition::Node, b"node:0:1", b"alice")
            .expect("put");

        let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open state machine");
        *sm.last_applied.lock().unwrap() = Some(log_id(1, 10));

        let mut builder = sm.get_snapshot_builder().await;
        let _snap = builder.build_snapshot().await.unwrap();
    }

    // Step 2: reopen, verify snapshot is still there
    {
        let engine = Arc::new(StorageEngine::open(&config()).expect("reopen"));
        let mut sm = CoordinodeStateMachine::new(engine).expect("reopen state machine");

        let snap = sm.get_current_snapshot().await.unwrap();
        assert!(snap.is_some(), "snapshot should survive reopen");

        let snap = snap.unwrap();
        assert_eq!(
            snap.meta.last_log_id.unwrap().index,
            10,
            "snapshot last_log_id should be 10"
        );
        assert_eq!(
            snap.meta.last_log_id.unwrap().committed_leader_id().term,
            1,
            "snapshot last_log_id term should be 1"
        );

        let data = snap.snapshot.into_inner();
        assert!(
            data.len() > 10,
            "snapshot data should be non-empty after reopen"
        );
        assert_eq!(&data[..4], b"CNSN", "snapshot data magic after reopen");
    }
}

fn make_entry(index: u64, term: u64, title: &str) -> Entry {
    use openraft::entry::RaftEntry;

    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(index),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:1:{index}").into_bytes(),
            value: title.as_bytes().to_vec(),
        }],
        commit_ts: Timestamp::from_raw(1000 + index),
        start_ts: Timestamp::from_raw(1000 + index - 1),
        bypass_rate_limiter: false,
    };

    Entry::new_normal(log_id(term, index), Request::single(proposal))
}

// -- oracle.advance_to() during Raft apply --

#[test]
fn apply_advances_oracle_to_commit_ts() {
    use coordinode_core::txn::timestamp::TimestampOracle;

    let dir = tempfile::TempDir::new().unwrap();
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
    let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()))
        .expect("open state machine");

    // Apply proposal with commit_ts=500
    let result = sm.apply_proposal(&node_put(1, 500, b"data"), 1, 0).unwrap();
    assert_eq!(result.mutations_applied, 1);

    // Oracle should have advanced to at least 500
    let next_ts = oracle.next();
    assert!(
        next_ts.as_raw() > 500,
        "oracle should have advanced past 500, got {}",
        next_ts.as_raw()
    );
}

#[test]
fn apply_100_entries_oracle_monotonic() {
    use coordinode_core::txn::timestamp::TimestampOracle;

    let dir = tempfile::TempDir::new().unwrap();
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
    let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()))
        .expect("open state machine");

    // Apply 100 proposals with increasing commit_ts, one per entry
    for i in 1..=100u64 {
        let proposal = node_put(i, 1000 + i, format!("v{i}").as_bytes());
        sm.apply_proposal(&proposal, i, 0).unwrap();
    }

    // Oracle should be at least at 1100 (last commit_ts)
    let final_ts = oracle.next();
    assert!(
        final_ts.as_raw() > 1100,
        "oracle should be past 1100 after 100 entries, got {}",
        final_ts.as_raw()
    );

    // Verify all 100 writes are readable
    for i in 1..=100u64 {
        let val = engine
            .get(Partition::Node, format!("node:1:{i}").as_bytes())
            .unwrap();
        assert_eq!(
            val.as_deref(),
            Some(format!("v{i}").as_bytes()),
            "mismatch at i={i}"
        );
    }
}

#[test]
fn apply_without_oracle_still_works() {
    // State machine without oracle applies normally
    let (_dir, engine) = test_engine();
    let sm = CoordinodeStateMachine::new(engine.clone()).expect("open state machine");

    let result = sm.apply_proposal(&node_put(1, 500, b"data"), 1, 0).unwrap();
    assert_eq!(result.mutations_applied, 1);

    let val = engine.get(Partition::Node, b"node:1:1").unwrap();
    assert_eq!(val.as_deref(), Some(b"data".as_slice()));
}

#[test]
fn apply_seqnos_match_commit_ts_with_oracle() {
    use coordinode_core::txn::timestamp::TimestampOracle;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };

    // Create engine WITH oracle so seqno = oracle timestamp
    let dir = tempfile::TempDir::new().unwrap();
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
    let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()))
        .expect("open state machine");
    // Opening re-anchored the oracle to the wall clock; the leader's commit
    // timestamps below are real HLC values inside the retention window (a
    // contrived tiny timestamp would sit below the horizon and be refused).
    let base = oracle.current().as_raw() + 1_000;

    // Apply at commit_ts=base+500
    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:1".to_vec(),
            value: b"first".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(base + 500),
        start_ts: Timestamp::from_raw(base + 499),
        bypass_rate_limiter: false,
    };
    sm.apply_proposal(&proposal, 1, 0).unwrap();

    // Apply at commit_ts=base+700
    let proposal2 = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(2),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:1".to_vec(),
            value: b"second".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(base + 700),
        start_ts: Timestamp::from_raw(base + 699),
        bypass_rate_limiter: false,
    };
    sm.apply_proposal(&proposal2, 2, 0).unwrap();

    // Each entry is applied at exactly its commit_ts, so time travel by
    // commit_ts is exact in both directions: at +500 the first write is
    // there, at +499 nothing is, at +700 the second supersedes it and at
    // +699 the first is still the visible version. A storage snapshot at S
    // sees seqnos strictly below S, so "as of commit_ts T" reads at T + 1.
    let at = |offset: u64| {
        engine
            .snapshot_get(&(base + offset + 1), Partition::Node, b"node:1:1")
            .unwrap()
    };
    assert_eq!(at(499), None, "nothing visible before the first commit_ts");
    assert_eq!(at(500).as_deref(), Some(b"first".as_ref()));
    assert_eq!(at(699).as_deref(), Some(b"first".as_ref()));
    assert_eq!(at(700).as_deref(), Some(b"second".as_ref()));
    let current_snap = engine.snapshot();
    let val_current = engine
        .snapshot_get(&current_snap, Partition::Node, b"node:1:1")
        .unwrap();
    assert_eq!(
        val_current.as_deref(),
        Some(b"second".as_ref()),
        "current snapshot should see last write"
    );

    // Verify oracle advanced past the last applied commit_ts
    let final_ts = oracle.next();
    assert!(
        final_ts.as_raw() > base + 700,
        "oracle should be past {}, got {}",
        base + 700,
        final_ts.as_raw()
    );
}

#[test]
fn multi_mutation_entry_is_atomic_at_its_commit_ts() {
    use coordinode_core::txn::timestamp::TimestampOracle;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };

    // A proposal carrying several mutations must be visible as a whole at its
    // commit_ts and invisible as a whole one tick earlier. Applying the
    // mutations at commit_ts, commit_ts + 1, ... (one seqno per op) passed
    // the single-mutation tests above while exposing a torn transaction to
    // any snapshot reader inside the range.
    let dir = tempfile::TempDir::new().unwrap();
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
    let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()))
        .expect("open state machine");
    // A real HLC commit timestamp inside the retention window (see
    // `apply_seqnos_match_commit_ts_with_oracle`).
    let commit_ts = oracle.current().as_raw() + 1_500;

    let keys: Vec<Vec<u8>> = (1..=5u64)
        .map(|i| format!("node:1:{i}").into_bytes())
        .collect();
    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
        mutations: keys
            .iter()
            .map(|k| Mutation::Put {
                partition: PartitionId::Node,
                key: k.clone(),
                value: b"v".to_vec(),
            })
            .collect(),
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    };
    let applied = sm.apply_proposal(&proposal, 1, 0).unwrap();
    assert_eq!(applied.mutations_applied, 5);

    // A storage snapshot at S sees seqnos strictly below S: "as of
    // commit_ts - 1" is a snapshot at commit_ts, "as of commit_ts" a snapshot
    // at commit_ts + 1.
    for key in &keys {
        assert_eq!(
            engine
                .snapshot_get(&commit_ts, Partition::Node, key)
                .unwrap(),
            None,
            "no key of the entry is visible before its commit_ts"
        );
        assert_eq!(
            engine
                .snapshot_get(&(commit_ts + 1), Partition::Node, key)
                .unwrap()
                .as_deref(),
            Some(b"v".as_ref()),
            "every key of the entry is visible at its commit_ts"
        );
    }
    // The oracle hands out strictly newer timestamps after the apply.
    assert!(oracle.next().as_raw() > commit_ts);
}

// ── Regression: unclean shutdown restart ─────────────────────────────────
//
// Bug: CoordiNode 0.3.17 crashed on restart with:
//   "create segment /data/oplog/0/oplog-00000000000000000000.bin: File exists (os error 17)"
//
// Root cause: after crash between oplog.flush() and put(KEY_LAST_LOG_ID),
// the LSM key was absent but the segment file existed. On restart,
// last_log_id=None → openraft called initialize() → SegmentWriter::create
// with create_new(true) on the existing segment → EEXIST.
//
// Fix: LogStore::open() now recovers last_log_id from oplog segments when
// the LSM key is missing.

/// Simulate crash between fsync and LSM key write: segment file exists but
/// KEY_LAST_LOG_ID was never persisted.  On re-open, last_log_id must be
/// reconstructed from the segment (not None), preventing the EEXIST crash.
#[tokio::test]
async fn restart_after_crash_recovers_last_log_id_from_oplog() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);

    // ── First "run": write entries, then simulate crash (delete LSM key) ──
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("open engine"));
        let mut store = LogStore::open(Arc::clone(&engine)).expect("open store");

        let entries = vec![
            make_entry(1, 1, "alpha"),
            make_entry(2, 1, "beta"),
            make_entry(3, 1, "gamma"),
        ];
        store
            .append(entries, IOFlushed::noop())
            .await
            .expect("append");

        // State should have last_log_id=3 after a normal append.
        let state = store.get_log_state().await.expect("get_log_state");
        assert_eq!(
            state.last_log_id.expect("last_log_id after append").index,
            3
        );

        // Simulate crash: remove the LSM key that was persisted by append().
        // The oplog segment files (in <oplog_endpoint>/oplog/<shard>/) remain on disk.
        engine
            .delete(Partition::Raft, KEY_LAST_LOG_ID)
            .expect("delete LSM key to simulate crash");

        // engine and store drop here — on a real crash the process dies instead.
    }

    // ── Restart: re-open the same data directory ──────────────────────────
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("re-open engine"));
        let mut store = LogStore::open(engine).expect("re-open store");

        let state = store
            .get_log_state()
            .await
            .expect("get_log_state after restart");

        // last_log_id must be recovered from the oplog segment, not None.
        // Without the fix this would be None → openraft calls initialize() →
        // SegmentWriter::create on existing segment → EEXIST crash.
        let recovered = state.last_log_id.expect(
            "last_log_id must be recovered from oplog after simulated crash; \
                 None would cause EEXIST crash on restart",
        );
        assert_eq!(
            recovered.index, 3,
            "recovered index must match last appended entry"
        );
    }
}

/// Variant: crash on a sealed segment (proper footer present).
/// After normal rotation the last segment has a footer → SegmentReader::open
/// succeeds on the fast path.  Recovery must still work.
#[tokio::test]
async fn restart_after_crash_with_sealed_segment_recovers_last_log_id() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);

    {
        let engine = Arc::new(StorageEngine::open(&config).expect("open engine"));
        let mut store = LogStore::open(Arc::clone(&engine)).expect("open store");

        let entries = vec![make_entry(1, 1, "x"), make_entry(2, 1, "y")];
        store
            .append(entries, IOFlushed::noop())
            .await
            .expect("append");

        // Force segment rotation so the active writer gets a footer.
        store.oplog.lock().expect("lock").rotate().expect("rotate");

        // Simulate crash: delete LSM key after rotation.
        engine
            .delete(Partition::Raft, KEY_LAST_LOG_ID)
            .expect("delete LSM key");
    }

    {
        let engine = Arc::new(StorageEngine::open(&config).expect("re-open engine"));
        let mut store = LogStore::open(engine).expect("re-open store");

        let state = store.get_log_state().await.expect("get_log_state");
        let recovered = state
            .last_log_id
            .expect("last_log_id must be recovered from sealed segment");
        assert_eq!(recovered.index, 2);
    }
}

// ─────────────────────────────────────────────────────────────────────
// MaxAssignedWatermark integration with apply_proposal.
// ─────────────────────────────────────────────────────────────────────

#[allow(clippy::panic)]
mod snap2_watermark_integration {
    use super::*;
    use coordinode_core::txn::watermark::{MaxAssignedWatermark, WaitError};
    use std::time::Duration;

    fn make_proposal(id: u64, commit_ts: u64) -> RaftProposal {
        RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(id),
            mutations: vec![Mutation::Put {
                partition: coordinode_core::txn::proposal::PartitionId::Node,
                key: format!("node:1:{id}").into_bytes(),
                value: b"data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(commit_ts),
            start_ts: Timestamp::from_raw(commit_ts - 1),
            bypass_rate_limiter: false,
        }
    }

    fn open_with_watermark(
        engine: Arc<StorageEngine>,
        wm: &Arc<MaxAssignedWatermark>,
    ) -> CoordinodeStateMachine {
        CoordinodeStateMachine::with_oracle_and_watermark(engine, None, Some(Arc::clone(wm)))
            .expect("open state machine")
    }

    #[tokio::test]
    async fn advance_applies_after_successful_proposal() {
        // Writer bursts commit_ts=1000, reader at T=800 returns immediately
        // after the apply has advanced the watermark past T.
        let (_dir, engine) = test_engine();
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let sm = open_with_watermark(engine, &wm);

        // Apply a single proposal with commit_ts = 1000.
        sm.apply_proposal(&make_proposal(42, 1000), 1, 0)
            .expect("apply ok");

        // Watermark must have advanced to commit_ts.
        assert_eq!(wm.current().as_raw(), 1000);

        // A wait at T=800 returns immediately.
        let got = wm
            .wait_for(Timestamp::from_raw(800), Duration::from_millis(50))
            .await
            .expect("fast path");
        assert_eq!(got.as_raw(), 1000);
    }

    #[tokio::test]
    async fn multiple_proposals_advance_to_latest() {
        // Applying proposals in increasing commit_ts order — watermark
        // always equals the latest.
        let (_dir, engine) = test_engine();
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let sm = open_with_watermark(engine, &wm);

        for (i, ts) in [(1u64, 100u64), (2, 200), (3, 350)] {
            sm.apply_proposal(&make_proposal(i, ts), i, 0)
                .expect("apply ok");
            assert_eq!(wm.current().as_raw(), ts);
        }
    }

    #[tokio::test]
    async fn out_of_order_commit_ts_is_monotonic_no_regression() {
        // Defensive: HLC guarantees monotonic commit_ts per shard, but
        // if a stale proposal somehow arrives (e.g. test, or replay
        // edge case), the watermark must not regress.
        let (_dir, engine) = test_engine();
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let sm = open_with_watermark(engine, &wm);

        sm.apply_proposal(&make_proposal(1, 500), 1, 0).expect("ok");
        assert_eq!(wm.current().as_raw(), 500);

        // Older commit_ts proposal — watermark must NOT go back.
        sm.apply_proposal(&make_proposal(2, 300), 2, 0).expect("ok");
        assert_eq!(wm.current().as_raw(), 500);
    }

    #[tokio::test]
    async fn no_watermark_configured_is_noop() {
        // The watermark is optional — paths that don't need cross-modality
        // snapshots still work without one.
        let (_dir, engine) = test_engine();
        let sm = CoordinodeStateMachine::new(engine).expect("open state machine");

        // Apply should succeed even without a watermark wired in.
        sm.apply_proposal(&make_proposal(1, 100), 1, 0)
            .expect("apply ok without watermark");

        // max_assigned() getter returns None.
        assert!(sm.max_assigned().is_none());
    }

    #[tokio::test]
    async fn reader_unblocks_when_applier_catches_up() {
        // Reader at latest T blocks, applier advances watermark, reader
        // unblocks promptly.
        let (_dir, engine) = test_engine();
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let sm = Arc::new(open_with_watermark(engine, &wm));

        let wm_reader = Arc::clone(&wm);
        let reader = tokio::spawn(async move {
            wm_reader
                .wait_for(Timestamp::from_raw(777), Duration::from_millis(500))
                .await
        });

        // Applier delay — ensure reader is actually blocked.
        tokio::time::sleep(Duration::from_millis(20)).await;
        sm.apply_proposal(&make_proposal(1, 777), 1, 0)
            .expect("apply");

        let got = reader.await.expect("task").expect("reader unblocks");
        assert_eq!(got.as_raw(), 777);
    }

    #[tokio::test]
    async fn reader_times_out_when_no_apply_arrives() {
        // Timeout path returns ErrReadTimeout (not stale Ok). Apply never
        // happens; reader must time out with the final observed watermark
        // value.
        let (_dir, engine) = test_engine();
        let wm = MaxAssignedWatermark::new(Timestamp::from_raw(100));
        let _sm = open_with_watermark(engine, &wm);

        let err = wm
            .wait_for(Timestamp::from_raw(500), Duration::from_millis(50))
            .await
            .expect_err("must time out");
        match err {
            WaitError::Timeout {
                target, current, ..
            } => {
                assert_eq!(target, 500);
                assert_eq!(current, 100);
            }
            other => panic!("expected Timeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn commit_ts_zero_proposal_does_not_advance() {
        // Defensive: commit_ts=0 is a sentinel for "no timestamp"
        // (shouldn't happen in practice) — must not advance the
        // watermark away from its initial value.
        let (_dir, engine) = test_engine();
        let wm = MaxAssignedWatermark::new(Timestamp::from_raw(50));
        let sm = open_with_watermark(engine, &wm);

        // commit_ts = 0 — should be ignored by the guard.
        let proposal = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
            mutations: vec![Mutation::Put {
                partition: coordinode_core::txn::proposal::PartitionId::Node,
                key: b"x".to_vec(),
                value: b"y".to_vec(),
            }],
            commit_ts: Timestamp::ZERO,
            start_ts: Timestamp::ZERO,
            bypass_rate_limiter: false,
        };
        sm.apply_proposal(&proposal, 1, 0).expect("apply");

        // Watermark unchanged.
        assert_eq!(wm.current().as_raw(), 50);
    }
}
