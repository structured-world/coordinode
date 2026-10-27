use std::io::Read;

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

/// A snapshot carrying metadata only.
fn empty_snapshot() -> SnapshotData {
    SnapshotData::Empty
}

/// The snapshot `sm` would send a follower now: its store pinned at its
/// applies.
async fn pinned(sm: &mut CoordinodeStateMachine) -> Snapshot {
    sm.get_current_snapshot()
        .await
        .expect("pin the store")
        .expect("a snapshot once something applied")
}

/// The metadata and the bytes of a pinned snapshot, as the network sends them.
fn sent(snapshot: Snapshot) -> (SnapshotMeta, Vec<u8>) {
    let mut transfer = match snapshot.snapshot {
        SnapshotData::Pinned { transfer, .. } => Some(transfer),
        _ => None,
    }
    .expect("a snapshot to send is the pinned store");
    let mut bytes = Vec::new();
    transfer.read_to_end(&mut bytes).expect("stream the store");
    assert_eq!(bytes.len() as u64, transfer.len());
    (snapshot.meta, bytes)
}

/// `bytes` received by `engine`'s member, where the network receives them.
fn received(engine: &StorageEngine, bytes: &[u8]) -> SnapshotData {
    let dir = crate::snapshot::receive_target(&crate::snapshot::snapshot_dir(engine));
    SnapshotData::Received(
        coordinode_storage::engine::store_transfer::receive_store(&mut &bytes[..], &dir)
            .expect("receive the snapshot"),
    )
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

/// Truncating the log after an index leaves the segments holding the kept
/// entries in place: the kept prefix is durable at every step, so a crash in
/// the middle of a truncation cannot take entries openraft relies on. Only
/// the segments wholly above the cut go, and the one the cut falls in is cut
/// short; the log reopens with the prefix and takes new entries after it.
#[cfg(unix)]
#[tokio::test]
async fn a_truncation_never_removes_the_kept_prefix() {
    use std::os::unix::fs::MetadataExt;

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
    let mut store = LogStore::open(Arc::clone(&engine)).expect("log store");
    let entries: Vec<Entry> = (1..=10).map(|i| make_entry(i, 1, "row")).collect();
    store
        .append(entries, IOFlushed::noop())
        .await
        .expect("append");

    let first_segment = {
        let oplog = store.oplog.lock().expect("lock");
        oplog.sealed_segment_paths()[0].to_path_buf()
    };
    let inode = std::fs::metadata(&first_segment).expect("stat").ino();

    store
        .truncate_after(Some(log_id(1, 5)))
        .await
        .expect("truncate");
    assert_eq!(
        std::fs::metadata(&first_segment)
            .expect("the kept segment is still there")
            .ino(),
        inode,
        "the segment of the kept prefix was removed and written again"
    );
    let kept = store.try_get_log_entries(0..=20).await.expect("read");
    assert_eq!(
        kept.iter().map(|e| e.log_id.index).collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5]
    );
    drop(store);

    let mut store = LogStore::open(engine).expect("reopen");
    let kept = store.try_get_log_entries(0..=20).await.expect("read");
    assert_eq!(kept.len(), 5);
    store
        .append(vec![make_entry(6, 2, "after")], IOFlushed::noop())
        .await
        .expect("append after the cut");
    let all = store.try_get_log_entries(0..=20).await.expect("read");
    assert_eq!(
        all.last()
            .map(|e| (e.log_id.index, e.log_id.leader_id.term)),
        Some((6, 2))
    );
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
    sm.install_snapshot(&meta, empty_snapshot()).await.unwrap();

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
    sm.install_snapshot(&meta, empty_snapshot()).await.unwrap();

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
    sent(pinned(&mut sm).await)
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
    let snapshot = received(&engine, bytes);
    rig.fail_from(op, skip);
    let stopped = sm.install_snapshot(meta, snapshot).await.is_err();
    drop(sm);
    rig.cut(engine);

    let (engine, clock) = open_rig_engine(&rig);
    let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(clock))
        .expect("a store with an interrupted snapshot install opens");
    sm.install_snapshot(meta, received(&engine, bytes))
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
    // A partition capture lives only as long as the copy that made it, and
    // a received snapshot only until it installs; one a crash interrupted
    // would otherwise hold its files, and their disk space, for good.
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    // Captures are left in a directory an engine already opened.
    drop(StorageEngine::open(&config).expect("first open"));
    let partition_capture = dir.path().join("partition-capture").join("node-1");
    // A snapshot received from a leader and never installed.
    let snapshot_capture = dir.path().join("raft-snapshot").join("1.recv");
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

/// A snapshot sent while a partition stands ahead of the applies carries
/// that partition's own position: the follower leaves it alone until its
/// applies pass it, as the leader does, rather than counting the entries the
/// copy already holds a second time.
#[tokio::test(flavor = "multi_thread")]
async fn a_partition_ahead_of_the_applies_travels_with_its_position() {
    use coordinode_storage::engine::core::{PartitionCopy, RaftPosition};
    let (_dir, leader) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&leader)).expect("open");
    apply_entries(&mut sm, vec![two_tree_merge_entry(1, 1000)]).await;
    let key = b"counter:degree:1".to_vec();
    let copy = PartitionCopy {
        rows: vec![(key.clone(), 6i64.to_le_bytes().to_vec())],
        position: Some(RaftPosition {
            next: 7,
            payload: rmp_serde::to_vec(&log_id(1, 6)).expect("payload"),
        }),
    };
    let installer = Arc::clone(&leader);
    tokio::task::spawn_blocking(move || {
        installer.install_partition(Partition::Counter, &copy, true)
    })
    .await
    .expect("install task")
    .expect("install");

    let (meta, bytes) = sent(pinned(&mut sm).await);
    assert_eq!(meta.last_log_id, Some(log_id(1, 1)));
    let (_dir2, follower) = test_engine();
    let mut fsm = CoordinodeStateMachine::new(Arc::clone(&follower)).expect("open follower");
    fsm.install_snapshot(&meta, received(&follower, &bytes))
        .await
        .expect("install");

    let later: Vec<Entry> = (2..=8).map(|i| two_tree_merge_entry(i, 1000 * i)).collect();
    apply_entries(&mut sm, later.clone()).await;
    apply_entries(&mut fsm, later).await;
    let counter = |engine: &StorageEngine| engine.get(Partition::Counter, &key).expect("get");
    assert_eq!(
        counter(&follower).as_deref(),
        Some(8i64.to_le_bytes().as_slice()),
        "six from the copy, then entries 7 and 8"
    );
    assert_eq!(counter(&follower), counter(&leader));
}

#[tokio::test]
async fn a_snapshot_holds_exactly_the_entries_up_to_its_log_id() {
    // A pinned snapshot is streamed while the state machine keeps applying.
    // A follower installs it and then receives every entry past its log id,
    // so an effect the data already carried from a later entry would be
    // applied twice.
    let (_dir, leader) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&leader)).expect("open");
    apply_entries(&mut sm, vec![two_tree_merge_entry(1, 1000)]).await;
    let snapshot = pinned(&mut sm).await;
    apply_entries(&mut sm, vec![two_tree_merge_entry(2, 2000)]).await;
    leader.persist().expect("flush the later entry");
    let (meta, bytes) = sent(snapshot);
    assert_eq!(meta.last_log_id, Some(log_id(1, 1)));

    let (_dir2, follower) = test_engine();
    let mut fsm = CoordinodeStateMachine::new(Arc::clone(&follower)).expect("open follower");
    fsm.install_snapshot(&meta, received(&follower, &bytes))
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
    let (meta, bytes) = sent(pinned(&mut sm).await);
    let t2 = leader_clock.current().as_raw() + 1;
    apply_entries(&mut sm, vec![put(2, t2, b"v2")]).await;

    // The snapshot reaches the follower after entry 2 was committed.
    std::thread::sleep(std::time::Duration::from_millis(20));
    let follower_rig = coordinode_test_fixtures::PowerRig::new();
    let (follower, follower_clock) = open_rig_engine(&follower_rig);
    let mut fsm = CoordinodeStateMachine::with_oracle(Arc::clone(&follower), Some(follower_clock))
        .expect("open follower");
    fsm.install_snapshot(&meta, received(&follower, &bytes))
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

/// An entry putting `value` under the index entry `rest` of generation 40 at
/// `ts`.
fn index_entry(index: u64, ts: u64, rest: &[u8], value: &[u8]) -> Entry {
    use openraft::entry::RaftEntry;
    Entry::new_normal(
        log_id(1, index),
        Request::single(RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(index),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Idx,
                key: generation_key(rest),
                value: value.to_vec(),
            }],
            commit_ts: Timestamp::from_raw(ts),
            start_ts: Timestamp::from_raw(ts - 1),
            bypass_rate_limiter: false,
        }),
    )
}

/// The logical key of entry `rest` of generation 40.
fn generation_key(rest: &[u8]) -> Vec<u8> {
    let mut key = vec![coordinode_core::index::encoding::GENERATION_TAGS[0]];
    key.extend_from_slice(&40u64.to_be_bytes());
    key.extend_from_slice(rest);
    key
}

/// Two replicas applying one Raft log: one replaces its copy of an index
/// generation with the other's history. A history taken before the replica
/// registered its replacement stands behind it in the Raft log and is
/// refused; one taken after completes it, and the published copy answers the
/// latest value, an earlier snapshot, and an entry applied while it was
/// prepared.
#[tokio::test]
async fn a_replica_replaces_a_generation_from_another_replicas_raft_history() {
    use coordinode_core::index::identity::GenerationId;
    use coordinode_core::txn::timestamp::TimestampOracle;
    let generation = GenerationId::from_raw(40);
    // Each replica applies an entry at its commit timestamp, as a cluster
    // node does: the histories line up version for version.
    let replica = |dir: &std::path::Path| {
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
        let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle))
            .expect("open state machine");
        (engine, sm)
    };
    let a_dir = tempfile::tempdir().expect("tempdir");
    let b_dir = tempfile::tempdir().expect("tempdir");
    let (a, mut sm_a) = replica(a_dir.path());
    let (b, mut sm_b) = replica(b_dir.path());
    // Commit timestamps inside the retention window, which runs back from now.
    let now = TimestampOracle::new().next().as_raw();
    let log = [
        index_entry(1, now + 1000, b"k", b"v1"),
        index_entry(2, now + 2000, b"k", b"v2"),
        index_entry(3, now + 3000, b"j", b"j1"),
        index_entry(4, now + 4000, b"m", b"m1"),
    ];
    apply_entries(&mut sm_a, log[..2].to_vec()).await;
    let early = a.export_generation_history(generation).expect("export");
    apply_entries(&mut sm_a, log[2..3].to_vec()).await;
    apply_entries(&mut sm_b, log[..3].to_vec()).await;

    b.stage_generation(generation).expect("stage");
    b.import_generation_history(generation, &early.entries)
        .expect("import");
    assert!(
        matches!(
            b.finish_generation_import(generation, early.covers_through, early.history_from),
            Err(coordinode_storage::error::StorageError::PositionBehind { .. })
        ),
        "a history short of entry 3 cannot complete the replacement"
    );

    let current = a.export_generation_history(generation).expect("export");
    apply_entries(&mut sm_b, log[3..].to_vec()).await;
    b.import_generation_history(generation, &current.entries)
        .expect("import");
    b.finish_generation_import(generation, current.covers_through, current.history_from)
        .expect("finish");
    b.publish_generation(generation).expect("publish");

    let get = |key: &[u8]| {
        b.get(Partition::Idx, &generation_key(key))
            .expect("get")
            .map(|v| v.to_vec())
    };
    assert_eq!(get(b"k"), Some(b"v2".to_vec()));
    assert_eq!(get(b"j"), Some(b"j1".to_vec()));
    assert_eq!(get(b"m"), Some(b"m1".to_vec()), "applied while prepared");
    assert_eq!(
        b.snapshot_get(&(now + 1500), Partition::Idx, &generation_key(b"k"))
            .expect("historical read")
            .map(|v| v.to_vec()),
        Some(b"v1".to_vec())
    );
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
    let (meta, bytes) = sent(pinned(&mut sm).await);
    assert_eq!(meta.last_log_id, Some(log_id(1, 2)));
    apply_entries(&mut sm, late.clone()).await;

    sm.install_snapshot(&meta, received(&engine, &bytes))
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
    // A leader's snapshot at entry 2, arriving after this member applied
    // further and restarted.
    let (meta, bytes) = sent(pinned(&mut sm).await);
    assert_eq!(meta.last_log_id, Some(log_id(1, 2)));
    apply_entries(&mut sm, late.clone()).await;
    drop(sm);
    // A clean stop: every applied entry is on disk, and the reopened seqno
    // sits past all of them.
    engine.persist().expect("persist");
    drop(engine);

    let (engine, clock) = open_rig_engine(&rig);
    let mut sm =
        CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(clock)).expect("reopen");
    sm.install_snapshot(&meta, received(&engine, &bytes))
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

/// A store whose member-local journal compacts after every entry, so a saved
/// vote lives in the Raft tree alone once `save_vote` returns.
fn compacting_config(dir: &std::path::Path) -> StorageConfig {
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.local_journal_compact_entries = 1;
    config
}

/// Save `vote` on a fresh member at `dir`, then lose the Raft tree the way a
/// damaged disk does: the tree's directory is gone and the engine opens it
/// empty.
async fn member_whose_raft_tree_was_lost(dir: &std::path::Path, vote: Vote) -> StorageConfig {
    let config = compacting_config(dir);
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        log.save_vote(&vote).await.expect("save vote");
    }
    std::fs::remove_dir_all(dir.join(Partition::Raft.name())).expect("lose the Raft tree");
    config
}

/// A member that lost its Raft tree still holds its promise: the separate copy
/// of the vote survives the tree, and the reopened member reads the vote it
/// granted, never the engine's initial one, which would let it vote again in
/// the same term.
#[tokio::test]
async fn a_vote_survives_the_loss_of_the_raft_tree() {
    let dir = tempfile::tempdir().expect("tempdir");
    let vote = Vote::new(7, 3);
    let config = member_whose_raft_tree_was_lost(dir.path(), vote).await;
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    assert_eq!(
        log.read_vote().await.expect("read vote"),
        Some(vote),
        "the promise survives the tree"
    );
}

/// With both copies of its vote gone, an existing member does not open as a
/// fresh one: its log is bound to a group, so an absent vote is a lost one,
/// and the open is refused with the store left as it was.
#[tokio::test]
async fn a_member_that_lost_every_copy_of_its_vote_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = member_whose_raft_tree_was_lost(dir.path(), Vote::new(7, 3)).await;
    std::fs::remove_file(
        dir.path()
            .join(coordinode_storage::engine::promise::PROMISE_FILE),
    )
    .expect("lose the second copy");
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    let err = LogStore::open(Arc::clone(&engine))
        .err()
        .expect("a member without its promise must not open");
    assert!(
        err.to_string().contains("vote"),
        "names the lost vote: {err}"
    );
}

/// The second copy's file in the store at `dir`.
fn promise_file(dir: &std::path::Path) -> std::path::PathBuf {
    dir.join(coordinode_storage::engine::promise::PROMISE_FILE)
}

/// The store incarnation `engine`'s consensus log marker names.
fn bound_incarnation(engine: &StorageEngine) -> Incarnation {
    match engine.consensus_binding().expect("binding") {
        ConsensusBinding::Bound(incarnation) => Some(incarnation),
        ConsensusBinding::Unbound | ConsensusBinding::Unnamed => None,
    }
    .expect("the marker names an incarnation")
}

/// A member opened for the first time records that it has not voted in both
/// places that matter: the second copy holds the explicit "no vote yet", and
/// the consensus log marker names the store's incarnation. Opened again it
/// is the same member, not a lost one.
#[tokio::test]
async fn a_new_member_records_its_promise_before_it_starts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = compacting_config(dir.path());
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
    assert_eq!(log.read_vote().await.expect("read"), None);
    let incarnation = bound_incarnation(&engine);
    assert_eq!(
        engine.read_promise_copy().expect("copy"),
        PromiseCopy::Held {
            incarnation,
            payload: rmp_serde::to_vec(&None::<Vote>).expect("encode"),
        }
    );
    drop(log);
    drop(engine);
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    assert_eq!(log.read_vote().await.expect("read"), None);
    assert_eq!(
        engine.consensus_binding().expect("binding"),
        ConsensusBinding::Bound(incarnation),
        "the incarnation is drawn once"
    );
}

/// A damaged second copy is rewritten from the tree, and a tree that lost its
/// record takes it back from the copy: after one open both hold the vote
/// again, so the next loss of either still leaves one.
#[tokio::test]
async fn a_damaged_copy_is_restored_from_the_other() {
    let dir = tempfile::tempdir().expect("tempdir");
    let vote = Vote::new(4, 2);
    let config = compacting_config(dir.path());
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        log.save_vote(&vote).await.expect("save");
    }
    let file = promise_file(dir.path());
    let mut bytes = std::fs::read(&file).expect("read copy");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&file, &bytes).expect("damage the copy");
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
        assert_eq!(log.read_vote().await.expect("read"), Some(vote));
    }
    // The copy was rewritten: now the tree may go too.
    std::fs::remove_dir_all(dir.path().join(Partition::Raft.name())).expect("lose the tree");
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    assert_eq!(log.read_vote().await.expect("read"), Some(vote));
    assert_eq!(
        engine
            .get(Partition::Raft, KEY_VOTE)
            .expect("tree")
            .map(|b| b.to_vec()),
        Some(rmp_serde::to_vec(&vote).expect("encode")),
        "the tree took the vote back from the copy"
    );
}

/// Of two copies that disagree, the greater by the consensus engine's vote
/// order is the promise: a copy can lag the other only by an update that did
/// not finish, and the newer promise is kept, never the older.
#[tokio::test]
async fn the_greater_of_two_copies_is_the_promise() {
    let dir = tempfile::tempdir().expect("tempdir");
    let older = Vote::new(4, 2);
    let newer = Vote::new_committed(5, 1);
    let config = compacting_config(dir.path());
    let incarnation = {
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        log.save_vote(&older).await.expect("save");
        let incarnation = bound_incarnation(&engine);
        // An update the tree took and the copy did not.
        engine
            .commit_local(
                vec![LocalWrite::Put {
                    partition: Partition::Raft,
                    key: KEY_VOTE.to_vec(),
                    value: rmp_serde::to_vec(&newer).expect("encode"),
                }],
                LocalSync::BeforeUse,
            )
            .expect("tree only");
        incarnation
    };
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
        assert_eq!(log.read_vote().await.expect("read"), Some(newer));
        assert_eq!(
            engine.read_promise_copy().expect("copy"),
            PromiseCopy::Held {
                incarnation,
                payload: rmp_serde::to_vec(&Some(newer)).expect("encode"),
            },
            "the lagging copy caught up"
        );
    }
    // The other way round: the copy ahead of a tree that lost its newest
    // record to a damaged block.
    let ahead = Vote::new(9, 3);
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
        engine
            .write_promise_copy(
                &incarnation,
                &rmp_serde::to_vec(&Some(ahead)).expect("encode"),
            )
            .expect("copy only");
    }
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    assert_eq!(log.read_vote().await.expect("read"), Some(ahead));
}

/// A second copy that belongs to another store (a file carried over from
/// another member's directory) is no promise of this one: it never outranks
/// the tree, it is replaced, and with the tree gone the member is refused
/// rather than opened on someone else's vote.
#[tokio::test]
async fn another_stores_copy_is_never_taken_for_the_promise() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mine = Vote::new(3, 1);
    let config = compacting_config(dir.path());
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        log.save_vote(&mine).await.expect("save");
        engine
            .write_promise_copy(
                &[0xee; 16],
                &rmp_serde::to_vec(&Some(Vote::new(50, 9))).expect("encode"),
            )
            .expect("foreign copy");
    }
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
        assert_eq!(log.read_vote().await.expect("read"), Some(mine));
        engine
            .write_promise_copy(
                &[0xee; 16],
                &rmp_serde::to_vec(&Some(Vote::new(50, 9))).expect("encode"),
            )
            .expect("foreign copy again");
    }
    std::fs::remove_dir_all(dir.path().join(Partition::Raft.name())).expect("lose the tree");
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    let err = LogStore::open(Arc::clone(&engine))
        .err()
        .expect("another store's copy must not open this member");
    assert!(PromiseUnqualified::of(&err).is_some(), "typed: {err}");
    assert!(err.to_string().contains("another store"), "{err}");
}

/// A store bound to its group before the marker carried an incarnation
/// keeps its vote: the tree's record becomes the second copy and the marker
/// gets an incarnation. One that already holds a group's history and has no
/// vote anywhere has lost it, and is refused.
#[tokio::test]
async fn a_store_bound_before_incarnations_is_converted_or_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let vote = Vote::new(6, 1);
    let config = compacting_config(dir.path());
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        log.save_vote(&vote).await.expect("save");
        append_durable(&mut log, vec![make_entry(1, 6, "x")]).await;
    }
    // The layout an earlier build left: an empty marker and no second copy.
    std::fs::write(dir.path().join("consensus-log"), b"").expect("empty marker");
    std::fs::remove_file(promise_file(dir.path())).expect("no copy");
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
        assert_eq!(log.read_vote().await.expect("read"), Some(vote));
        assert!(matches!(
            engine.consensus_binding().expect("binding"),
            ConsensusBinding::Bound(_)
        ));
        assert!(matches!(
            engine.read_promise_copy().expect("copy"),
            PromiseCopy::Held { .. }
        ));
    }
    // The same layout with the tree's record lost as well.
    std::fs::write(dir.path().join("consensus-log"), b"").expect("empty marker");
    std::fs::remove_file(promise_file(dir.path())).expect("no copy");
    std::fs::remove_dir_all(dir.path().join(Partition::Raft.name())).expect("lose the tree");
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    let err = LogStore::open(Arc::clone(&engine))
        .err()
        .expect("a member with a log and no vote anywhere lost it");
    assert!(PromiseUnqualified::of(&err).is_some(), "typed: {err}");
}

/// An unchanged vote costs no write: openraft may save the vote it already
/// holds, and only a change needs a sync.
#[tokio::test]
async fn saving_the_same_vote_again_writes_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = compacting_config(dir.path());
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
    let vote = Vote::new(2, 1);
    log.save_vote(&vote).await.expect("save");
    let file = promise_file(dir.path());
    std::fs::remove_file(&file).expect("remove the copy to see a rewrite");
    log.save_vote(&vote).await.expect("save the same");
    assert!(
        !file.exists(),
        "an unchanged vote must not be written again"
    );
    log.save_vote(&Vote::new(3, 1))
        .await
        .expect("save a change");
    assert!(file.exists(), "a changed vote is written");
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
    // a sync: a full one, or a data sync where the trees order their writes
    // with a barrier. Cutting at both walks the Raft path's whole durability
    // order.
    let full = raft_cut_at_every(lsm_tree::fs::FaultOp::SyncAll).await;
    let data = raft_cut_at_every(lsm_tree::fs::FaultOp::SyncData).await;
    assert!(full + data > 0, "the workload reached no sync");
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

/// A build compacts the log against the applied store: it copies and
/// serializes nothing, and leaves nothing beside the store. The data is the
/// store itself, pinned when a follower needs it, and installs on another
/// member.
#[tokio::test]
async fn a_build_keeps_nothing_and_a_pin_serves_the_data() {
    let (_dir, engine) = test_engine();
    engine
        .put(Partition::Node, b"node:0:1", b"alice")
        .expect("put");
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open state machine");
    *sm.last_applied.lock().unwrap() = Some(log_id(1, 5));

    let mut builder = sm.get_snapshot_builder().await;
    let built = builder.build_snapshot().await.unwrap();
    assert_eq!(built.meta.last_log_id, Some(log_id(1, 5)));
    assert!(matches!(built.snapshot, SnapshotData::Empty));
    assert!(
        snapshot_files(&engine).is_empty(),
        "nothing kept beside the store: {:?}",
        snapshot_files(&engine)
    );

    let (meta, bytes) = sent(pinned(&mut sm).await);
    assert_eq!(meta.last_log_id, Some(log_id(1, 5)));
    let (_other_dir, other) = test_engine();
    let mut other_sm = CoordinodeStateMachine::new(Arc::clone(&other)).expect("open other");
    other_sm
        .install_snapshot(&meta, received(&other, &bytes))
        .await
        .expect("install");
    assert_eq!(
        other.get(Partition::Node, b"node:0:1").unwrap().as_deref(),
        Some(&b"alice"[..])
    );
    assert!(
        snapshot_files(&other).is_empty(),
        "the received copy goes once installed: {:?}",
        snapshot_files(&other)
    );
}

/// Before anything applied there is no snapshot to send.
#[tokio::test]
async fn nothing_applied_is_no_snapshot() {
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(engine).expect("open state machine");
    assert!(sm.get_current_snapshot().await.unwrap().is_none());
}

/// Every file in `engine`'s snapshot directory; none when it does not exist.
fn snapshot_files(engine: &StorageEngine) -> Vec<std::path::PathBuf> {
    let Ok(dir) = std::fs::read_dir(crate::snapshot::snapshot_dir(engine)) else {
        return Vec::new();
    };
    let mut files: Vec<_> = dir.map(|entry| entry.unwrap().path()).collect();
    files.sort();
    files
}

/// A snapshot is the store, and the store must not hold the previous
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
        builder.build_snapshot().await.unwrap();
        let (_, data) = sent(pinned(&mut sm).await);
        sizes.push(data.len());
    }

    // A build flushes every tree, so the second pin holds a few more small
    // tables until compaction merges them; carrying the first snapshot would
    // add at least the first one's whole size.
    assert!(
        sizes[1] < 2 * sizes[0],
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

        let (_, data) = sent(snap);
        assert_eq!(&data[..4], b"CNST", "the store streams after reopen");
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

/// A log of entries 1..=6 in two sealed segments (1..=3, 4..=6) with
/// `committed` recorded, whose first segment then lost its footer and part of
/// entry 3 the way a disk shows writes it lost after a failed fsync.
async fn log_with_a_hole_at_3(committed: u64) -> (tempfile::TempDir, StorageConfig) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open engine"));
    let active = raft_oplog_dirs(&engine, 0).expect("oplog dirs").active;
    {
        let mut store = LogStore::open(Arc::clone(&engine)).expect("open store");
        append_durable(&mut store, (1..=3).map(|i| make_entry(i, 1, "x")).collect()).await;
        store.oplog.lock().expect("lock").rotate().expect("rotate");
        append_durable(&mut store, (4..=6).map(|i| make_entry(i, 1, "x")).collect()).await;
        store.oplog.lock().expect("lock").rotate().expect("rotate");
        store
            .save_committed(Some(log_id(1, committed)))
            .await
            .expect("committed");
        engine
            .persist_partition(Partition::Raft)
            .expect("persist committed");
    }
    drop(engine);
    let first = active.join(format!("oplog-{:020}.bin", 1));
    let mut bytes = std::fs::read(&first).expect("read segment");
    let at = bytes.len() - (coordinode_storage::oplog::FOOTER_SIZE as usize + 8);
    bytes[at..].fill(0);
    std::fs::write(&first, &bytes).expect("write segment");
    (dir, config)
}

/// Committed entries are never cut away. When writes the disk lost leave a
/// hole at or below the committed index, the log is refused rather than
/// reopened shorter than what was acknowledged, and its segments stay as they
/// were.
#[tokio::test]
async fn a_hole_below_the_committed_index_refuses_the_log() {
    let (_dir, config) = log_with_a_hole_at_3(5).await;
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen engine"));
    let err = LogStore::open(Arc::clone(&engine))
        .err()
        .expect("a hole below committed must refuse the log");
    assert!(
        err.to_string().contains("from index 3"),
        "names the first lost index: {err}"
    );
}

/// A member whose peers still hold what its disk lost opens instead: the log
/// ends before the hole, the committed index comes down to that end so the
/// state machine is not asked to replay entries that are gone, and the lost
/// committed entry is recorded durably, so a restart before the member has
/// caught up keeps it out of elections all the same.
#[tokio::test]
async fn a_member_whose_peers_hold_the_lost_entries_opens_to_catch_up() {
    let (_dir, config) = log_with_a_hole_at_3(5).await;
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen engine"));
        let (mut store, through) =
            LogStore::open_with(Arc::clone(&engine), LostCommitted::CatchUp).expect("open");
        assert_eq!(through, Some(log_id(1, 5)), "the lost committed entry");
        let state = store.get_log_state().await.expect("log state");
        assert_eq!(state.last_log_id.map(|id| id.index), Some(2));
        assert_eq!(
            store.read_committed().await.expect("committed"),
            Some(log_id(1, 2)),
            "committed comes down to what the log holds"
        );
    }
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen engine"));
    let (_, through) =
        LogStore::open_with(Arc::clone(&engine), LostCommitted::CatchUp).expect("open again");
    assert_eq!(through, Some(log_id(1, 5)), "the loss survives a restart");

    // Without peers to catch up from, a recorded loss refuses the member.
    drop(engine);
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen engine"));
    let err = LogStore::open(Arc::clone(&engine))
        .err()
        .expect("a member alone cannot recover what it lost");
    assert!(err.to_string().contains("index 5"), "{err}");

    // Caught up, the record goes, and the member opens like any other.
    clear_catch_up(&engine).expect("clear");
    let (_, through) =
        LogStore::open_with(Arc::clone(&engine), LostCommitted::CatchUp).expect("open caught up");
    assert_eq!(through, None);
}

/// A hole above the committed index lost only entries never acknowledged:
/// the log reopens ending right before it, and the entries past it are not
/// part of the log.
#[tokio::test]
async fn a_hole_above_the_committed_index_ends_the_log_before_it() {
    let (_dir, config) = log_with_a_hole_at_3(2).await;
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen engine"));
    let mut store = LogStore::open(Arc::clone(&engine)).expect("reopen store");
    let state = store.get_log_state().await.expect("log state");
    assert_eq!(state.last_log_id.map(|id| id.index), Some(2));
    let entries = store.try_get_log_entries(1..7).await.expect("read");
    assert_eq!(
        entries.iter().map(|e| e.log_id.index).collect::<Vec<_>>(),
        vec![1, 2]
    );
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
        let reader = std::thread::spawn(move || {
            wm_reader.wait_for(Timestamp::from_raw(777), Duration::from_millis(500))
        });

        // Applier delay — ensure reader is actually blocked.
        std::thread::sleep(Duration::from_millis(20));
        sm.apply_proposal(&make_proposal(1, 777), 1, 0)
            .expect("apply");

        let got = reader.join().expect("thread").expect("reader unblocks");
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

        let WaitError::Timeout {
            target, current, ..
        } = wm
            .wait_for(Timestamp::from_raw(500), Duration::from_millis(50))
            .expect_err("must time out");
        assert_eq!(target, 500);
        assert_eq!(current, 100);
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
