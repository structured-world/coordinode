use std::io::{Read, Write};

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

/// A snapshot carrying metadata only, staged where a received one would be.
fn empty_snapshot(engine: &StorageEngine) -> SnapshotFile {
    SnapshotFile::stage(&crate::snapshot::snapshot_dir(engine)).unwrap()
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

/// Only a write under a user-data prefix counts as the user's data in the
/// log: a node whose log carries its own bookkeeping, removals and metadata
/// commands still joins, while one that acknowledged a write does not, even
/// if that write reached no tree before a crash.
#[tokio::test]
async fn only_user_writes_in_the_log_count_as_data() {
    use openraft::entry::RaftEntry;

    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();

    let bookkeeping = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
        mutations: vec![
            Mutation::Command(
                coordinode_core::txn::proposal::MetadataCommand::RegisterFields {
                    names: vec!["name".to_string()],
                },
            ),
            Mutation::Put {
                partition: PartitionId::Schema,
                key: b"meta:routing".to_vec(),
                value: b"x".to_vec(),
            },
            Mutation::Delete {
                partition: PartitionId::Node,
                key: b"node:1:9".to_vec(),
            },
        ],
        commit_ts: Timestamp::from_raw(1001),
        start_ts: Timestamp::from_raw(1000),
        bypass_rate_limiter: false,
    };
    store
        .append(
            vec![Entry::new_normal(
                log_id(1, 1),
                Request::single(bookkeeping),
            )],
            IOFlushed::noop(),
        )
        .await
        .unwrap();
    assert!(!store.writes_user_data_from(0).unwrap());

    store
        .append(vec![make_entry(2, 1, "mine")], IOFlushed::noop())
        .await
        .unwrap();
    assert!(store.writes_user_data_from(0).unwrap());
    assert!(store.writes_user_data_from(2).unwrap());
    // Past the write: replay starts after it, so the trees answer for it.
    assert!(!store.writes_user_data_from(3).unwrap());
}

/// The Raft log takes its segment rotation from the engine's configuration:
/// it used fixed constants, so an operator's oplog settings reached the
/// embedded journal and not the log of a replicated node.
#[tokio::test]
async fn the_raft_log_rotates_by_the_engine_configuration() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.oplog_segment_max_entries = 3;
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let mut store = LogStore::open(engine).expect("log store");

    let entries: Vec<Entry> = (1..=10).map(|i| make_entry(i, 1, "row")).collect();
    store
        .append(entries, IOFlushed::noop())
        .await
        .expect("append");

    let sealed = store.oplog.lock().expect("lock").sealed_segment_count();
    assert_eq!(sealed, 3, "10 entries at 3 per segment seal three segments");
    let read = store.try_get_log_entries(1..=10).await.expect("read");
    assert_eq!(read.len(), 10);
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

/// A log record holds each proposal once, as its unit frame after a small
/// envelope, and reads back as the entry that was appended. The record
/// carried the proposals twice before: inside the envelope and again as
/// operations for change streams.
#[tokio::test]
async fn a_log_record_holds_each_proposal_once_as_its_frame() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();
    let proposals = vec![
        node_put(1, 1000, &[7u8; 256]),
        node_put(2, 1001, &[9u8; 256]),
    ];
    let entry = Entry::new_normal(log_id(1, 4), Request::batch(proposals.clone()));
    store
        .append(vec![entry.clone()], IOFlushed::noop())
        .await
        .unwrap();

    let record = store
        .oplog
        .lock()
        .unwrap()
        .read_range(4, 5)
        .unwrap()
        .pop()
        .expect("the record");
    let expected: Vec<OplogOp> = proposals
        .iter()
        .map(|p| OplogOp::Unit {
            frame: encode_proposal(p).unwrap(),
        })
        .collect();
    assert!(
        matches!(record.ops.first(), Some(OplogOp::RaftEntry { data }) if data.len() < 64),
        "a small envelope comes first: {:?}",
        record.ops.first()
    );
    assert_eq!(
        record.ops[1..],
        expected[..],
        "each proposal follows once, as its frame"
    );

    let read = store.try_get_log_entries(4..=4).await.unwrap();
    assert_eq!(read, vec![entry]);
}

/// A record written before proposals were stored as unit frames, with the
/// proposals in its envelope and their operations after it, still reads as
/// the entry it recorded: retained log tails outlive the upgrade.
#[tokio::test]
async fn a_log_record_from_before_unit_frames_still_reads() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();
    let entry = Entry::new_normal(log_id(1, 1), Request::single(node_put(1, 1000, b"v")));
    let old = OplogEntry {
        ts: 1000,
        term: 1,
        index: 1,
        shard: 0,
        ops: vec![
            OplogOp::RaftEntry {
                data: rmp_serde::to_vec(&entry).unwrap(),
            },
            OplogOp::Insert {
                partition: 0,
                key: b"node:1:1".to_vec(),
                value: b"v".to_vec(),
            },
        ],
        is_migration: false,
        pre_images: None,
    };
    store.oplog.lock().unwrap().append(&old).unwrap();

    let read = store.try_get_log_entries(1..=1).await.unwrap();
    assert_eq!(read, vec![entry]);
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
async fn a_purge_below_what_is_already_purged_is_a_no_op() {
    // A later checkpoint can lower the floor below what an earlier purge
    // already removed. The purge must then remove nothing and succeed: an
    // error from purge stops consensus on a node that has lost nothing.
    // One entry per segment, so a purge really deletes the entries.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.oplog_segment_max_entries = 1;
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let mut store = LogStore::open(Arc::clone(&engine)).unwrap();
    let entries = vec![
        make_entry(1, 1, "a"),
        make_entry(2, 1, "b"),
        make_entry(3, 1, "c"),
        make_entry(4, 1, "d"),
        make_entry(5, 1, "e"),
    ];
    store.append(entries, IOFlushed::noop()).await.unwrap();
    engine.reset_raft_coverage(6, &[]).unwrap();
    store.purge(log_id(1, 3)).await.unwrap();

    engine.set_raft_log_keep_from(2);
    store
        .purge(log_id(1, 5))
        .await
        .expect("a floor below the purged entries removes nothing");

    let remaining = store.try_get_log_entries(0..=10).await.unwrap();
    assert_eq!(
        remaining.iter().map(|e| e.log_id.index).collect::<Vec<_>>(),
        vec![4, 5],
        "nothing past the floor goes, nothing purged comes back"
    );
    let state = store.get_log_state().await.unwrap();
    assert_eq!(state.last_purged_log_id, Some(log_id(1, 3)));
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

/// The applied membership that does not decode fails the open. It was taken
/// as the initial, empty membership: the member forgot its group and could
/// elect itself alone.
#[test]
fn an_unreadable_applied_membership_refuses_the_open() {
    let (_dir, engine) = test_engine();
    engine
        .put(Partition::Schema, KEY_SM_MEMBERSHIP, b"not-msgpack-bytes")
        .unwrap();

    let err = CoordinodeStateMachine::new(Arc::clone(&engine))
        .err()
        .expect("an unreadable membership must not open as an empty one");
    assert!(err.to_string().contains("raft:sm:membership"), "got: {err}");
}

/// The purge point that does not decode fails the log's open. It was taken
/// as no purge at all, so the log claimed to start at entries it no longer
/// holds.
#[test]
fn an_unreadable_purge_point_refuses_the_log_open() {
    let (_dir, engine) = test_engine();
    engine
        .put(Partition::Raft, KEY_PURGED, b"not-msgpack-bytes")
        .unwrap();

    let err = LogStore::open(engine)
        .err()
        .expect("an unreadable purge point must not open as none");
    assert!(err.to_string().contains("raft:purged"), "got: {err}");
}

/// Readers waiting on the applied watermark (causal reads with an
/// `after_index`, the change stream's bound) must see an installed snapshot
/// at once: on an idle cluster no later apply would ever move it.
#[tokio::test]
async fn installing_a_snapshot_publishes_its_applied_index() {
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open state machine");
    let mut applied = sm.subscribe_applied();
    assert_eq!(*applied.borrow_and_update(), 0);

    let meta = SnapshotMeta {
        last_log_id: Some(log_id(2, 20)),
        last_membership: openraft::StoredMembership::default(),
    };
    sm.install_snapshot(&meta, empty_snapshot(&engine))
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
    sm.install_snapshot(&meta, empty_snapshot(&engine))
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

/// A snapshot standing at entry 2 (`node:1:a` = `a1`, `node:1:b` = `b1`):
/// its metadata and its bytes.
async fn snapshot_image() -> (SnapshotMeta, Vec<u8>) {
    use std::io::Seek;
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open");
    apply_entries(
        &mut sm,
        vec![
            node_key_entry(1, 1000, b"node:1:a", b"a1"),
            node_key_entry(2, 2000, b"node:1:b", b"b1"),
        ],
    )
    .await;
    let mut builder = sm.get_snapshot_builder().await;
    let mut snapshot = builder.build_snapshot().await.expect("build");
    snapshot.snapshot.rewind().expect("rewind");
    let mut bytes = Vec::new();
    snapshot
        .snapshot
        .read_to_end(&mut bytes)
        .expect("read the snapshot");
    (snapshot.meta, bytes)
}

/// `bytes` staged as a received snapshot for `engine`.
fn staged_snapshot(engine: &StorageEngine, bytes: &[u8]) -> SnapshotFile {
    use std::io::Seek;
    let mut file = empty_snapshot(engine);
    file.write_all(bytes).expect("stage the snapshot");
    file.rewind().expect("rewind");
    file
}

/// Install the snapshot of [`snapshot_image`] over a store holding a row the
/// snapshot lacks, with the disk refusing `op` after `skip` of them; cut the
/// power, reopen and install it again. Returns whether the fault stopped the
/// first install.
async fn snapshot_install_cut_at(
    op: lsm_tree::fs::FaultOp,
    skip: u64,
    meta: &SnapshotMeta,
    bytes: &[u8],
) -> bool {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let (engine, clock) = open_rig_engine(&rig);
    let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(clock.clone()))
        .expect("open state machine");
    apply_entries(
        &mut sm,
        vec![node_key_entry(
            1,
            clock.next().as_raw(),
            b"node:1:stale",
            b"x",
        )],
    )
    .await;
    engine.persist().expect("persist");
    let snapshot = staged_snapshot(&engine, bytes);
    rig.fail_from(op, skip);
    let stopped = sm.install_snapshot(meta, snapshot).await.is_err();
    drop(sm);
    rig.cut(engine);

    let (engine, clock) = open_rig_engine(&rig);
    let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(clock))
        .expect("a store with an interrupted snapshot install opens");
    sm.install_snapshot(meta, staged_snapshot(&engine, bytes))
        .await
        .expect("the snapshot installs again");
    let read = |key: &[u8]| engine.get(Partition::Node, key).expect("get");
    assert_eq!(
        read(b"node:1:a").as_deref(),
        Some(b"a1".as_slice()),
        "{op:?} {skip}"
    );
    assert_eq!(
        read(b"node:1:b").as_deref(),
        Some(b"b1".as_slice()),
        "{op:?} {skip}"
    );
    assert_eq!(
        read(b"node:1:stale"),
        None,
        "{op:?} {skip}: exactly the snapshot"
    );
    assert!(engine.pending_rebuilds().expect("intents").is_empty());
    assert_eq!(
        engine
            .raft_coverage()
            .expect("coverage")
            .resume_point()
            .map(|(next, _)| next),
        Some(3),
        "{op:?} {skip}: every tree stands at the snapshot"
    );
    drop(sm);
    stopped
}

#[tokio::test(flavor = "multi_thread")]
async fn a_power_cut_inside_a_snapshot_install_is_repaired_by_installing_again() {
    use lsm_tree::fs::FaultOp;
    let (meta, bytes) = snapshot_image().await;
    for op in [FaultOp::Write, FaultOp::SyncAll, FaultOp::SyncData] {
        let mut skip = 0;
        while snapshot_install_cut_at(op, skip, &meta, &bytes).await {
            skip += 1;
        }
    }
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

/// An entry past the snapshot that the leader committed before the follower
/// installed the snapshot must win over the snapshot's older value for the
/// same key: the follower reads what the leader reads.
#[tokio::test]
async fn an_entry_after_the_snapshot_wins_over_the_installed_value() {
    use openraft::entry::RaftEntry;
    let put = |index: u64, ts: u64, value: &[u8]| {
        Entry::new_normal(
            log_id(1, index),
            Request::single(RaftProposal {
                id: coordinode_core::txn::proposal::ProposalId::from_raw(index),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:1:k".to_vec(),
                    value: value.to_vec(),
                }],
                commit_ts: Timestamp::from_raw(ts),
                start_ts: Timestamp::from_raw(ts - 1),
                bypass_rate_limiter: false,
            }),
        )
    };

    let leader_rig = coordinode_test_fixtures::PowerRig::new();
    let (leader, leader_clock) = open_rig_engine(&leader_rig);
    let mut sm =
        CoordinodeStateMachine::with_oracle(Arc::clone(&leader), Some(leader_clock.clone()))
            .expect("open leader");
    let t1 = leader_clock.current().as_raw() + 1;
    apply_entries(&mut sm, vec![put(1, t1, b"v1")]).await;
    let mut builder = sm.get_snapshot_builder().await;
    let snapshot = builder.build_snapshot().await.expect("build");
    let t2 = leader_clock.current().as_raw() + 1;
    apply_entries(&mut sm, vec![put(2, t2, b"v2")]).await;

    // The snapshot reaches the follower after entry 2 was committed.
    std::thread::sleep(std::time::Duration::from_millis(20));
    let follower_rig = coordinode_test_fixtures::PowerRig::new();
    let (follower, follower_clock) = open_rig_engine(&follower_rig);
    let mut fsm = CoordinodeStateMachine::with_oracle(Arc::clone(&follower), Some(follower_clock))
        .expect("open follower");
    fsm.install_snapshot(&snapshot.meta, snapshot.snapshot)
        .await
        .expect("install");
    apply_entries(&mut fsm, vec![put(2, t2, b"v2")]).await;

    assert_eq!(
        follower
            .get(Partition::Node, b"node:1:k")
            .expect("follower get")
            .as_deref(),
        Some(b"v2".as_slice()),
        "the follower must read the entry committed after the snapshot"
    );
}

/// An entry putting `value` under `key` in the Node partition at `ts`.
fn node_key_entry(index: u64, ts: u64, key: &[u8], value: &[u8]) -> Entry {
    use openraft::entry::RaftEntry;
    Entry::new_normal(
        log_id(1, index),
        Request::single(RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(index),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: key.to_vec(),
                value: value.to_vec(),
            }],
            commit_ts: Timestamp::from_raw(ts),
            start_ts: Timestamp::from_raw(ts - 1),
            bypass_rate_limiter: false,
        }),
    )
}

/// A state machine reopened after a snapshot was built stands at or past the
/// snapshot: every tree held its entries when it was captured. Standing
/// below it makes openraft install the snapshot over a store that already
/// holds it and more.
#[tokio::test]
async fn a_reopened_state_machine_stands_at_or_past_its_snapshot() {
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open");
    apply_entries(
        &mut sm,
        (1..=3)
            .map(|i| node_key_entry(i, 1000 * i, format!("node:1:{i}").as_bytes(), b"v"))
            .collect(),
    )
    .await;
    let mut builder = sm.get_snapshot_builder().await;
    let snapshot = builder.build_snapshot().await.expect("build");
    assert_eq!(snapshot.meta.last_log_id, Some(log_id(1, 3)));
    drop(builder);
    drop(sm);

    let mut reopened = CoordinodeStateMachine::new(engine).expect("reopen");
    let applied = reopened.applied_state().await.expect("applied").0;
    assert!(
        applied >= Some(log_id(1, 3)),
        "the reopened state machine stands at {applied:?}, below the snapshot at 3"
    );
}

/// A snapshot installed over a store that already applied entries past it,
/// followed by those entries again (what openraft does when the applies
/// stand below its snapshot), leaves the store at its latest state: a key the
/// later entries created is there, and a key they changed holds the change.
#[tokio::test]
async fn entries_reapplied_after_a_snapshot_install_are_visible() {
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open");
    let early = vec![
        node_key_entry(1, 1000, b"node:1:a", b"a1"),
        node_key_entry(2, 2000, b"node:1:b", b"b1"),
    ];
    let late = vec![
        node_key_entry(3, 3000, b"node:1:a", b"a2"),
        node_key_entry(4, 4000, b"node:1:c", b"c1"),
    ];
    apply_entries(&mut sm, early).await;
    let mut builder = sm.get_snapshot_builder().await;
    let snapshot = builder.build_snapshot().await.expect("build");
    assert_eq!(snapshot.meta.last_log_id, Some(log_id(1, 2)));
    apply_entries(&mut sm, late.clone()).await;

    sm.install_snapshot(&snapshot.meta, snapshot.snapshot)
        .await
        .expect("install");
    apply_entries(&mut sm, late).await;
    engine.persist().expect("persist");
    engine.major_compact(Partition::Node).expect("compact");

    let read = |key: &[u8]| engine.get(Partition::Node, key).expect("get");
    assert_eq!(
        read(b"node:1:a").as_deref(),
        Some(b"a2".as_slice()),
        "changed"
    );
    assert_eq!(read(b"node:1:b").as_deref(), Some(b"b1".as_slice()), "kept");
    assert_eq!(
        read(b"node:1:c").as_deref(),
        Some(b"c1".as_slice()),
        "created"
    );
}

/// The same after a restart: the store reopened with its seqno past every
/// applied entry, the snapshot installed, the entries after it replayed at
/// their own commit timestamps, which sit below that seqno.
#[tokio::test]
async fn entries_replayed_after_a_snapshot_install_on_reopen_are_visible() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let (engine, clock) = open_rig_engine(&rig);
    let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(clock.clone()))
        .expect("open");
    let ts = || clock.next().as_raw();
    let early = vec![
        node_key_entry(1, ts(), b"node:1:a", b"a1"),
        node_key_entry(2, ts(), b"node:1:b", b"b1"),
    ];
    let late = vec![
        node_key_entry(3, ts(), b"node:1:a", b"a2"),
        node_key_entry(4, ts(), b"node:1:c", b"c1"),
    ];
    apply_entries(&mut sm, early).await;
    let mut builder = sm.get_snapshot_builder().await;
    builder.build_snapshot().await.expect("build");
    apply_entries(&mut sm, late.clone()).await;
    drop(builder);
    drop(sm);
    // A clean stop: every applied entry is on disk, and the reopened seqno
    // sits past all of them.
    engine.persist().expect("persist");
    drop(engine);

    let (engine, clock) = open_rig_engine(&rig);
    let mut sm =
        CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(clock)).expect("reopen");
    let snapshot = sm
        .get_current_snapshot()
        .await
        .expect("read the snapshot")
        .expect("the built snapshot is current");
    assert_eq!(snapshot.meta.last_log_id, Some(log_id(1, 2)));
    sm.install_snapshot(&snapshot.meta, snapshot.snapshot)
        .await
        .expect("install");
    apply_entries(&mut sm, late).await;
    // Merged on disk, the versions of a key are ordered by seqno alone; a
    // read of the memtable first would hide a replay buried under a newer
    // installed row until a compaction.
    engine.persist().expect("persist");
    engine.major_compact(Partition::Node).expect("compact");

    let read = |key: &[u8]| engine.get(Partition::Node, key).expect("get");
    assert_eq!(read(b"node:1:a").as_deref(), Some(b"a2".as_slice()));
    assert_eq!(read(b"node:1:b").as_deref(), Some(b"b1".as_slice()));
    assert_eq!(read(b"node:1:c").as_deref(), Some(b"c1".as_slice()));
}

/// A vote is durable once `save_vote` returns: openraft answers the
/// candidate right after, and a node that forgot its vote in a crash could
/// grant a second one in the same term, electing two leaders.
#[tokio::test]
async fn a_saved_vote_survives_a_crash() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let vote = Vote::new(7, 3);
    {
        let (engine, _) = open_rig_engine(&rig);
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        log.save_vote(&vote).await.expect("save vote");
        drop(log);
        rig.cut(engine);
    }
    let (engine, _) = open_rig_engine(&rig);
    let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    assert_eq!(log.read_vote().await.expect("read vote"), Some(vote));
}

/// Append `entries` and wait until the log reports them durable, as openraft
/// does before it counts on them.
async fn append_durable(log: &mut LogStore, entries: Vec<Entry>) {
    let (tx, rx) = <TypeConfig as openraft::type_config::TypeConfigExt>::oneshot();
    log.append(entries, IOFlushed::signal(tx))
        .await
        .expect("append");
    rx.await.expect("answered").expect("durable");
}

/// The end of the log after a crash is where the log on disk ends, never an
/// older end recorded beside it: a vote persists the Raft metadata with the
/// last log id of its time, and a log that reopened there would drop every
/// acknowledged entry after it.
#[tokio::test]
async fn the_log_reopens_at_its_durable_end_after_a_crash() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    {
        let (engine, _) = open_rig_engine(&rig);
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        append_durable(
            &mut log,
            (1..=3).map(|i| make_entry(i, 1, "before")).collect(),
        )
        .await;
        log.save_vote(&Vote::new(1, 1)).await.expect("vote");
        append_durable(
            &mut log,
            (4..=6).map(|i| make_entry(i, 1, "after")).collect(),
        )
        .await;
        drop(log);
        rig.cut(engine);
    }
    let (engine, _) = open_rig_engine(&rig);
    let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    let state = log.get_log_state().await.expect("log state");
    assert_eq!(
        state.last_log_id.map(|id| id.index),
        Some(6),
        "the log ends at its last durable entry"
    );
}

/// A purge is recorded durably before it drops a segment: after a crash the
/// log reopens purged where it was, never promising entries already gone.
#[tokio::test]
async fn a_purge_survives_a_crash() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    {
        let (engine, _) = open_rig_engine(&rig);
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        append_durable(&mut log, (1..=5).map(|i| make_entry(i, 1, "x")).collect()).await;
        engine.reset_raft_coverage(5, &[]).expect("coverage");
        log.purge(log_id(1, 3)).await.expect("purge");
        drop(log);
        rig.cut(engine);
    }
    let (engine, _) = open_rig_engine(&rig);
    let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    let state = log.get_log_state().await.expect("log state");
    assert_eq!(
        state.last_purged_log_id.map(|id| id.index),
        Some(3),
        "the purge point survives the crash"
    );
    assert_eq!(state.last_log_id.map(|id| id.index), Some(5));
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

/// A field registration committed to the log but lost from the memtable is
/// decided again when openraft re-delivers it, and the re-decision binds the
/// same ids: it reads only what earlier entries bound, and an entry whose
/// bindings already reached disk decides nothing new. A registration that
/// re-decided against a different frontier would give a name an id other
/// data already means something else by.
#[tokio::test]
async fn a_registration_lost_before_flush_replays_to_the_same_ids() {
    use coordinode_core::txn::proposal::MetadataCommand;
    use coordinode_storage::engine::metadata::load_field_dictionary;
    use lsm_tree::AbstractTree;
    use openraft::entry::RaftEntry;

    let register = |index: u64, ts: u64, names: &[&str]| {
        Entry::new_normal(
            log_id(1, index),
            Request::single(RaftProposal {
                id: coordinode_core::txn::proposal::ProposalId::from_raw(index),
                mutations: vec![Mutation::Command(MetadataCommand::RegisterFields {
                    names: names.iter().map(|n| (*n).to_owned()).collect(),
                })],
                commit_ts: Timestamp::from_raw(ts),
                start_ts: Timestamp::from_raw(ts - 1),
                bypass_rate_limiter: false,
            }),
        )
    };
    let bindings = |engine: &StorageEngine| {
        let dictionary = load_field_dictionary(engine).expect("a consistent dictionary");
        ["b", "a", "c", "d"].map(|n| dictionary.lookup(n))
    };

    let rig = coordinode_test_fixtures::PowerRig::new();
    let (ts1, ts2);
    let before = {
        let (engine, oracle) = open_rig_engine(&rig);
        let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()))
            .expect("open");
        ts1 = oracle.current().as_raw() + 1_000;
        ts2 = ts1 + 1;
        apply_entries(&mut sm, vec![register(1, ts1, &["b", "a"])]).await;
        // Entry 1 reaches disk; entry 2 stays in the memtable the cut loses.
        engine
            .tree(Partition::Schema)
            .unwrap()
            .flush_active_memtable(0)
            .unwrap();
        apply_entries(&mut sm, vec![register(2, ts2, &["a", "c", "d"])]).await;
        let before = bindings(&engine);
        drop(sm);
        rig.cut(engine);
        before
    };
    assert_eq!(before, [Some(1), Some(2), Some(3), Some(4)]);

    let (engine, oracle) = open_rig_engine(&rig);
    let mut sm =
        CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle)).expect("reopen");
    assert_eq!(
        bindings(&engine),
        [Some(1), Some(2), None, None],
        "the cut lost entry 2's bindings"
    );
    apply_entries(
        &mut sm,
        vec![
            register(1, ts1, &["b", "a"]),
            register(2, ts2, &["a", "c", "d"]),
        ],
    )
    .await;
    assert_eq!(
        bindings(&engine),
        before,
        "the re-delivered registrations bind what they bound before the cut"
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
        // openraft applies an entry only once the log reports it durable.
        let (durable_tx, durable) = <TypeConfig as openraft::type_config::TypeConfigExt>::oneshot();
        if log
            .append(vec![entry.clone()], IOFlushed::signal(durable_tx))
            .await
            .is_err()
            || !matches!(durable.await, Ok(Ok(())))
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

    // The metadata is recorded in the store; the build keeps the capture it
    // was made from and serializes nothing, so its cost does not grow with
    // the data.
    let snap_meta = engine.get(Partition::Schema, KEY_SNAPSHOT_META).unwrap();
    assert!(snap_meta.is_some(), "snapshot meta should be persisted");
    drop(snap);
    let files = snapshot_files(&engine);
    assert_eq!(files.len(), 1, "one published snapshot: {files:?}");
    assert!(files[0].is_dir(), "kept as the capture: {files:?}");

    // Read, the capture serializes into a snapshot holding the data, which
    // installs on another store.
    let (current_meta, mut current) = sm.snapshots.current().unwrap().expect("a current snapshot");
    assert_eq!(current_meta.last_log_id, Some(log_id(1, 5)));
    let mut data = Vec::new();
    current.read_to_end(&mut data).unwrap();
    assert_eq!(&data[..4], b"CNSN", "snapshot data should have CNSN magic");
    let (_other_dir, other) = test_engine();
    crate::snapshot::install_full_snapshot_from_reader(&other, &mut std::io::Cursor::new(&data))
        .unwrap();
    assert_eq!(
        other.get(Partition::Node, b"node:0:1").unwrap().as_deref(),
        Some(&b"alice"[..])
    );
    drop(current);
    assert_eq!(
        snapshot_files(&engine),
        files,
        "reading leaves only the published capture"
    );
}

/// A capture the record names but a crash lost reads as no snapshot, which
/// openraft answers by building a new one, never as an error that would stop
/// the node from starting.
#[tokio::test]
async fn a_lost_snapshot_capture_reads_as_no_snapshot() {
    let (_dir, engine) = test_engine();
    engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open state machine");
    *sm.last_applied.lock().unwrap() = Some(log_id(1, 5));
    drop(
        sm.get_snapshot_builder()
            .await
            .build_snapshot()
            .await
            .unwrap(),
    );

    let files = snapshot_files(&engine);
    assert_eq!(files.len(), 1, "{files:?}");
    std::fs::remove_dir_all(&files[0]).unwrap();

    assert!(sm.snapshots.current().unwrap().is_none());
    assert!(sm.get_current_snapshot().await.unwrap().is_none());
    assert!(
        snapshot_files(&engine).is_empty(),
        "no half-made copy is left behind"
    );
}

/// Every file in `engine`'s snapshot directory.
fn snapshot_files(engine: &StorageEngine) -> Vec<std::path::PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(crate::snapshot::snapshot_dir(engine))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    files.sort();
    files
}

/// A new build replaces the file of the one before it, so the directory
/// holds one snapshot however many were built.
#[tokio::test]
async fn a_new_snapshot_removes_the_file_it_replaces() {
    let (_dir, engine) = test_engine();
    engine.put(Partition::Node, b"node:0:1", b"alice").unwrap();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open state machine");

    let mut published = Vec::new();
    for index in [5, 6] {
        *sm.last_applied.lock().unwrap() = Some(log_id(1, index));
        let mut builder = sm.get_snapshot_builder().await;
        builder.build_snapshot().await.unwrap();
        let files = snapshot_files(&engine);
        assert_eq!(files.len(), 1, "after the build at {index}: {files:?}");
        published.push(files[0].clone());
    }
    assert_ne!(published[0], published[1], "the second build is a new file");
}

/// A snapshot older than the current one never replaces it: a build that
/// finishes after a newer install leaves the newer snapshot current.
#[tokio::test]
async fn an_older_snapshot_does_not_replace_a_newer_one() {
    let (_dir, engine) = test_engine();
    let store = SnapshotStore::open(Arc::clone(&engine)).unwrap();
    let meta_at = |index| SnapshotMeta {
        last_log_id: Some(log_id(1, index)),
        last_membership: openraft::StoredMembership::default(),
    };

    let mut newer = store.stage().unwrap();
    newer.write_all(b"newer").unwrap();
    assert!(store.publish(&meta_at(9), &mut newer).unwrap());
    let mut older = store.stage().unwrap();
    older.write_all(b"older").unwrap();
    assert!(!store.publish(&meta_at(4), &mut older).unwrap());
    drop(older);

    let (meta, mut current) = store.current().unwrap().expect("a current snapshot");
    assert_eq!(meta.last_log_id, Some(log_id(1, 9)));
    let mut data = Vec::new();
    current.read_to_end(&mut data).unwrap();
    assert_eq!(data, b"newer");
    assert_eq!(snapshot_files(&engine).len(), 1, "the refused file is gone");
}

/// A received snapshot that never installs, and anything a crash left in
/// the directory, is gone once the store reopens; the current one stays.
#[tokio::test]
async fn reopening_keeps_only_the_current_snapshot() {
    let (_dir, engine) = test_engine();
    let dir = crate::snapshot::snapshot_dir(&engine);
    {
        let store = SnapshotStore::open(Arc::clone(&engine)).unwrap();
        let mut current = store.stage().unwrap();
        current.write_all(b"current").unwrap();
        let meta = SnapshotMeta {
            last_log_id: Some(log_id(1, 3)),
            last_membership: openraft::StoredMembership::default(),
        };
        assert!(store.publish(&meta, &mut current).unwrap());

        let received = SnapshotFile::stage(&dir).unwrap();
        drop(received);
        assert_eq!(
            snapshot_files(&engine).len(),
            1,
            "a dropped staged file goes"
        );
    }
    std::fs::write(dir.join("left-by-a-crash.part"), b"half").unwrap();

    let store = SnapshotStore::open(Arc::clone(&engine)).unwrap();
    assert_eq!(
        snapshot_files(&engine).len(),
        1,
        "{:?}",
        snapshot_files(&engine)
    );
    let (_, mut current) = store.current().unwrap().expect("a current snapshot");
    let mut data = Vec::new();
    current.read_to_end(&mut data).unwrap();
    assert_eq!(data, b"current");
}

/// A store an earlier build wrote kept the snapshot bytes as a Schema value;
/// opening it moves them to a file and drops the value, which every later
/// snapshot would otherwise carry.
#[tokio::test]
async fn a_snapshot_kept_in_the_store_moves_to_a_file() {
    let (_dir, engine) = test_engine();
    let meta = SnapshotMeta {
        last_log_id: Some(log_id(2, 7)),
        last_membership: openraft::StoredMembership::default(),
    };
    engine
        .put(
            Partition::Schema,
            KEY_SNAPSHOT_META,
            &rmp_serde::to_vec(&meta).unwrap(),
        )
        .unwrap();
    engine
        .put(Partition::Schema, b"raft:snapshot:data", b"CNSN-legacy")
        .unwrap();

    let store = SnapshotStore::open(Arc::clone(&engine)).unwrap();
    assert!(
        engine
            .get(Partition::Schema, b"raft:snapshot:data")
            .unwrap()
            .is_none()
    );
    let (current_meta, mut current) = store.current().unwrap().expect("a current snapshot");
    assert_eq!(current_meta.last_log_id, meta.last_log_id);
    let mut data = Vec::new();
    current.read_to_end(&mut data).unwrap();
    assert_eq!(data, b"CNSN-legacy");
}

/// A build serializes the store, and the store must not hold the previous
/// snapshot: otherwise every snapshot carries all the ones before it and
/// grows with each build over unchanged data.
#[tokio::test]
async fn a_snapshot_does_not_carry_the_previous_one() {
    let (_dir, engine) = test_engine();
    engine
        .put(Partition::Node, b"node:0:1", &vec![7u8; 64 * 1024])
        .expect("put");
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open state machine");

    let mut sizes = Vec::new();
    for index in [5, 6] {
        *sm.last_applied.lock().unwrap() = Some(log_id(1, index));
        let mut builder = sm.get_snapshot_builder().await;
        let mut snap = builder.build_snapshot().await.unwrap();
        let mut data = Vec::new();
        snap.snapshot.read_to_end(&mut data).unwrap();
        sizes.push(data.len());
    }

    assert!(
        sizes[1] < sizes[0] + 4096,
        "the second snapshot of unchanged data grew from {} to {} bytes",
        sizes[0],
        sizes[1]
    );
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

        let mut snapshot = snap.snapshot;
        let mut data = Vec::new();
        snapshot.read_to_end(&mut data).unwrap();
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
// Root cause: the end of the log was read from a key recorded beside it,
// absent after the crash while the segment file existed. On restart,
// last_log_id=None → openraft called initialize() → SegmentWriter::create
// with create_new(true) on the existing segment → EEXIST.
//
// Fix: LogStore::open() reads the end of the log from its segments.

/// On re-open, last_log_id is read from the segment the entries went to
/// (not None), preventing the EEXIST crash.
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

    // ── First "run": write entries ──
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
