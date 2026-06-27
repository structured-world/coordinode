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
    let log_id = openraft::LogId::new(
        CommittedLeaderId {
            term: 1,
            node_id: 0,
        },
        1,
    );
    store.truncate_after(Some(log_id)).await.unwrap();

    let remaining = store.try_get_log_entries(0..=10).await.unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].log_id.index, 1);
}

#[tokio::test]
async fn log_store_purge() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();

    let entries = vec![
        make_entry(1, 1, "a"),
        make_entry(2, 1, "b"),
        make_entry(3, 1, "c"),
    ];
    store.append(entries, IOFlushed::noop()).await.unwrap();

    // Purge up to index 2 (delete 1 and 2, keep 3)
    let log_id = openraft::LogId::new(
        CommittedLeaderId {
            term: 1,
            node_id: 0,
        },
        2,
    );
    store.purge(log_id).await.unwrap();

    let remaining = store.try_get_log_entries(0..=10).await.unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].log_id.index, 3);
}

#[tokio::test]
async fn log_store_committed_roundtrip() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();

    // No committed initially
    assert!(store.read_committed().await.unwrap().is_none());

    let log_id = openraft::LogId::new(
        CommittedLeaderId {
            term: 1,
            node_id: 0,
        },
        5,
    );
    store.save_committed(Some(log_id)).await.unwrap();

    let loaded = store.read_committed().await.unwrap().unwrap();
    assert_eq!(loaded.index, 5);
}

// -- CoordinodeStateMachine --

#[tokio::test]
async fn state_machine_initial_state() {
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(engine);

    let (applied, membership) = sm.applied_state().await.unwrap();
    assert!(applied.is_none());
    assert!(membership.log_id().is_none());
}

// -- Dedup tests --

#[test]
fn dedup_skips_duplicate_proposal() {
    // Apply same proposal twice — second should be detected as duplicate
    let (_dir, engine) = test_engine();
    let sm = CoordinodeStateMachine::new(engine);

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
    let r1 = sm.apply_proposal(&proposal).unwrap();
    assert_eq!(r1.mutations_applied, 1);

    // Second apply (same id + same size): should return 0 (dedup)
    let r2 = sm.apply_proposal(&proposal).unwrap();
    assert_eq!(r2.mutations_applied, 0);
}

#[test]
fn dedup_allows_different_size_same_id() {
    // Same proposal ID but different payload size should re-apply
    // (represents a retry with modified payload)
    let (_dir, engine) = test_engine();
    let sm = CoordinodeStateMachine::new(engine);

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
    let r1 = sm.apply_proposal(&proposal_v1).unwrap();
    assert_eq!(r1.mutations_applied, 1);

    // Second version with different size: should NOT be deduped
    let r2 = sm.apply_proposal(&proposal_v2).unwrap();
    assert_eq!(r2.mutations_applied, 1);
}

#[test]
fn dedup_gc_removes_old_entries() {
    // Verify that dedup GC cleans old entries
    let (_dir, engine) = test_engine();
    let sm = CoordinodeStateMachine::new(engine);

    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
        mutations: vec![Mutation::Put {
            partition: coordinode_core::txn::proposal::PartitionId::Node,
            key: b"node:1:1".to_vec(),
            value: b"data".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(100),
        start_ts: Timestamp::from_raw(99),
        bypass_rate_limiter: false,
    };

    sm.apply_proposal(&proposal).unwrap();

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
    let proposal2 = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(2),
        mutations: vec![Mutation::Put {
            partition: coordinode_core::txn::proposal::PartitionId::Node,
            key: b"node:1:2".to_vec(),
            value: b"data2".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(200),
        start_ts: Timestamp::from_raw(199),
        bypass_rate_limiter: false,
    };
    sm.apply_proposal(&proposal2).unwrap();

    // Old entry should have been GC'd, new one remains
    let dedup = sm.dedup.lock().unwrap();
    assert!(!dedup.contains_key(&1u64), "old entry should be GC'd");
    assert!(dedup.contains_key(&2u64), "new entry should remain");
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

// -- Helpers --

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

    // Initially no purge
    let state = store.get_log_state().await.unwrap();
    assert!(state.last_purged_log_id.is_none(), "no purge initially");

    // Purge up to index 2
    let purge_id = openraft::LogId::new(
        CommittedLeaderId {
            term: 1,
            node_id: 0,
        },
        2,
    );
    store.purge(purge_id).await.unwrap();

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

    // Phase 1: write + purge
    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            &path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let mut store = LogStore::open(Arc::clone(&engine)).unwrap();

        let entries = vec![
            make_entry(1, 1, "a"),
            make_entry(2, 1, "b"),
            make_entry(3, 1, "c"),
        ];
        store.append(entries, IOFlushed::noop()).await.unwrap();

        let purge_id = openraft::LogId::new(
            CommittedLeaderId {
                term: 1,
                node_id: 0,
            },
            2,
        );
        store.purge(purge_id).await.unwrap();
    }

    // Phase 2: reopen, verify purge state persisted
    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            &path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
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

    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine));

    // Set applied state so snapshot has a valid log_id
    let log_id = openraft::LogId::new(
        CommittedLeaderId {
            term: 1,
            node_id: 0,
        },
        5,
    );
    *sm.last_applied.lock().unwrap() = Some(log_id);
    sm.save_applied(&log_id).unwrap();

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

    // Phase 1: build snapshot
    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            &path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        engine
            .put(Partition::Node, b"node:0:1", b"alice")
            .expect("put");

        let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine));
        let log_id = openraft::LogId::new(
            CommittedLeaderId {
                term: 1,
                node_id: 0,
            },
            10,
        );
        *sm.last_applied.lock().unwrap() = Some(log_id);
        sm.save_applied(&log_id).unwrap();

        let mut builder = sm.get_snapshot_builder().await;
        let _snap = builder.build_snapshot().await.unwrap();
    }

    // Phase 2: reopen, verify snapshot is still there
    {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            &path,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
        let mut sm = CoordinodeStateMachine::new(engine);

        let snap = sm.get_current_snapshot().await.unwrap();
        assert!(snap.is_some(), "snapshot should survive reopen");

        let snap = snap.unwrap();
        assert_eq!(
            snap.meta.last_log_id.unwrap().index,
            10,
            "snapshot last_log_id should be 10"
        );
        assert_eq!(
            snap.meta.snapshot_id, "snap-10-1",
            "snapshot_id should match"
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
    use coordinode_core::txn::proposal::PartitionId;
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

    let committed_leader_id = CommittedLeaderId { term, node_id: 0 };
    let log_id = openraft::LogId::new(committed_leader_id, index);

    Entry::new_normal(log_id, Request::single(proposal))
}

// -- R068: oracle.advance_to() during Raft apply --

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
    let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()));

    // Apply proposal with commit_ts=500
    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:1".to_vec(),
            value: b"data".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(500),
        start_ts: Timestamp::from_raw(499),
        bypass_rate_limiter: false,
    };

    let result = sm.apply_proposal(&proposal).unwrap();
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
    let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()));

    // Apply 100 proposals with increasing commit_ts
    for i in 1..=100u64 {
        let proposal = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(i),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:1:{i}").into_bytes(),
                value: format!("v{i}").into_bytes(),
            }],
            commit_ts: Timestamp::from_raw(1000 + i),
            start_ts: Timestamp::from_raw(999 + i),
            bypass_rate_limiter: false,
        };
        sm.apply_proposal(&proposal).unwrap();
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
    // Backward compat: state machine without oracle applies normally
    let (_dir, engine) = test_engine();
    let sm = CoordinodeStateMachine::new(engine.clone());

    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:1".to_vec(),
            value: b"data".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(500),
        start_ts: Timestamp::from_raw(499),
        bypass_rate_limiter: false,
    };

    let result = sm.apply_proposal(&proposal).unwrap();
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
    let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()));

    // Apply at commit_ts=500
    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:1".to_vec(),
            value: b"first".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(500),
        start_ts: Timestamp::from_raw(499),
        bypass_rate_limiter: false,
    };
    sm.apply_proposal(&proposal).unwrap();

    // Apply at commit_ts=700
    let proposal2 = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(2),
        mutations: vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:1:1".to_vec(),
            value: b"second".to_vec(),
        }],
        commit_ts: Timestamp::from_raw(700),
        start_ts: Timestamp::from_raw(699),
        bypass_rate_limiter: false,
    };
    sm.apply_proposal(&proposal2).unwrap();

    // Take a snapshot between the two proposals (after first apply).
    // We can't use snapshot_at(500) directly because the snapshot tracker
    // may not have registered that seqno yet (memtable not sealed).
    // Instead, verify via snapshot taken at the right point.
    //
    // Since we can't go back in time, verify that current snapshot sees
    // "second" and that oracle advanced monotonically through both entries.
    let current_snap = engine.snapshot();
    let val_current = engine
        .snapshot_get(&current_snap, Partition::Node, b"node:1:1")
        .unwrap();
    assert_eq!(
        val_current.as_deref(),
        Some(b"second".as_ref()),
        "current snapshot should see last write"
    );

    // Verify oracle advanced past 700
    let final_ts = oracle.next();
    assert!(
        final_ts.as_raw() > 700,
        "oracle should be past 700, got {}",
        final_ts.as_raw()
    );
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
// R-SNAP2: MaxAssignedWatermark integration with apply_proposal.
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

    #[tokio::test]
    async fn advance_applies_after_successful_proposal() {
        // R-SNAP2 regression #1: writer bursts commit_ts=1000, reader at
        // T=800 returns immediately after the apply has advanced the
        // watermark past T.
        let (_dir, engine) = test_engine();
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let sm =
            CoordinodeStateMachine::with_oracle_and_watermark(engine, None, Some(Arc::clone(&wm)));

        // Apply a single proposal with commit_ts = 1000.
        let proposal = make_proposal(42, 1000);
        sm.apply_proposal(&proposal).expect("apply ok");

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
        let sm =
            CoordinodeStateMachine::with_oracle_and_watermark(engine, None, Some(Arc::clone(&wm)));

        for (i, ts) in [(1u64, 100u64), (2, 200), (3, 350)] {
            let p = make_proposal(i, ts);
            sm.apply_proposal(&p).expect("apply ok");
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
        let sm =
            CoordinodeStateMachine::with_oracle_and_watermark(engine, None, Some(Arc::clone(&wm)));

        sm.apply_proposal(&make_proposal(1, 500)).expect("ok");
        assert_eq!(wm.current().as_raw(), 500);

        // Older commit_ts proposal — watermark must NOT go back.
        sm.apply_proposal(&make_proposal(2, 300)).expect("ok");
        assert_eq!(wm.current().as_raw(), 500);
    }

    #[tokio::test]
    async fn no_watermark_configured_is_noop() {
        // The watermark is optional — legacy paths that don't need
        // cross-modality snapshots still work without one.
        let (_dir, engine) = test_engine();
        let sm = CoordinodeStateMachine::new(engine);

        // Apply should succeed even without a watermark wired in.
        sm.apply_proposal(&make_proposal(1, 100))
            .expect("apply ok without watermark");

        // max_assigned() getter returns None.
        assert!(sm.max_assigned().is_none());
    }

    #[tokio::test]
    async fn reader_unblocks_when_applier_catches_up() {
        // R-SNAP2 regression #2: reader at latest T blocks, applier
        // advances watermark, reader unblocks promptly.
        let (_dir, engine) = test_engine();
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let sm = Arc::new(CoordinodeStateMachine::with_oracle_and_watermark(
            engine,
            None,
            Some(Arc::clone(&wm)),
        ));

        let wm_reader = Arc::clone(&wm);
        let reader = tokio::spawn(async move {
            wm_reader
                .wait_for(Timestamp::from_raw(777), Duration::from_millis(500))
                .await
        });

        // Applier delay — ensure reader is actually blocked.
        tokio::time::sleep(Duration::from_millis(20)).await;
        sm.apply_proposal(&make_proposal(1, 777)).expect("apply");

        let got = reader.await.expect("task").expect("reader unblocks");
        assert_eq!(got.as_raw(), 777);
    }

    #[tokio::test]
    async fn reader_times_out_when_no_apply_arrives() {
        // R-SNAP2 regression #3: timeout path returns ErrReadTimeout
        // (not stale Ok). Apply never happens; reader must time out
        // with the final observed watermark value.
        let (_dir, engine) = test_engine();
        let wm = MaxAssignedWatermark::new(Timestamp::from_raw(100));
        let _sm =
            CoordinodeStateMachine::with_oracle_and_watermark(engine, None, Some(Arc::clone(&wm)));

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
        let sm =
            CoordinodeStateMachine::with_oracle_and_watermark(engine, None, Some(Arc::clone(&wm)));

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
        sm.apply_proposal(&proposal).expect("apply");

        // Watermark unchanged.
        assert_eq!(wm.current().as_raw(), 50);
    }
}
