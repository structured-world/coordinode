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
    // The writer thread puts the entries in the segments.
    store.sync.wait_idle();

    let sealed = store.oplog.lock().expect("lock").sealed_segment_count();
    assert_eq!(sealed, 3, "10 entries at 3 per segment seal three segments");
    let read = store.try_get_log_entries(1..=10).await.expect("read");
    assert_eq!(read.len(), 10);
}

/// An append does no file I/O: with the segments held by another thread (a
/// write of the segment waits out a full sync of the same file, as macOS
/// holds it), it returns at once and the entry reads back. The Raft core
/// appends; when it wrote the segment itself it waited out every sync of
/// the log and handed over one batch per sync.
#[tokio::test(flavor = "multi_thread")]
async fn an_append_returns_while_the_segments_are_held() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(engine).unwrap();
    let oplog = Arc::clone(&store.oplog);
    let (held_tx, held_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        let _held = oplog.lock().expect("lock");
        held_tx.send(()).expect("held");
        release_rx.recv().ok();
    });
    held_rx.recv().expect("the segments are held");

    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        store.append(vec![make_entry(1, 1, "a")], IOFlushed::noop()),
    )
    .await
    .expect("the append waited on the segments")
    .expect("append");
    let read = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        store.try_get_log_entries(1..=1),
    )
    .await
    .expect("the read waited on the segments")
    .expect("read");
    assert_eq!(read.len(), 1, "the entry reads back before it is written");

    release_tx.send(()).expect("release");
    holder.join().expect("holder");
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
    // The writer thread puts the entries in the segments.
    store.sync.wait_idle();

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
    store.sync.wait_idle();

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
    store.sync.wait_idle();

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
    append_durable(&mut store, entries).await;
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
    // openraft purges only applied entries, which are on disk by then.
    append_durable(&mut store, entries).await;
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
    append_durable(&mut store, entries).await;
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
    append_durable(&mut store, entries).await;
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
    append_durable(&mut store, entries).await;
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

    let log = store.oplog_handle();
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

/// The Schema tree rebuilt from a checkpoint holds the closure records of
/// the entries the rebuild replays from the log, not only the checkpoint's:
/// after the rebuild and a restart every position answers its own commit
/// ceiling. Replaying only the proposals left the replayed positions
/// answered with the checkpoint's last ceiling.
#[tokio::test(flavor = "multi_thread")]
async fn a_schema_rebuilt_from_a_checkpoint_keeps_the_replayed_ceilings() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_dir(dir.path());
    let mut log = LogStore::open(Arc::clone(&engine)).unwrap();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open");
    // A group's history starts with its initial membership at entry 0, as
    // openraft initializes it; the data entries follow from 1. The rebuild
    // reads the log from the checkpoint's base, 0.
    use openraft::entry::RaftEntry;
    let membership = openraft::Membership::new(
        vec![std::collections::BTreeSet::from([1])],
        std::collections::BTreeMap::from([(1, MemberNode::new("http://node-1".into(), None))]),
    )
    .expect("a uniform configuration");
    let initial = Entry::new_membership(log_id(1, 0), membership);
    let first: Vec<Entry> = std::iter::once(initial)
        .chain((1..=3).map(|i| two_tree_merge_entry(i, 1000 * i)))
        .collect();
    append_durable(&mut log, first.clone()).await;
    apply_entries(&mut sm, first).await;
    let ckpt_root = tempfile::tempdir().unwrap();
    let ckpt = ckpt_root.path().join("c");
    engine.create_checkpoint(&ckpt).unwrap();
    let later: Vec<Entry> = (4..=6).map(|i| two_tree_merge_entry(i, 1000 * i)).collect();
    append_durable(&mut log, later.clone()).await;
    apply_entries(&mut sm, later).await;
    // The empty prefix through the membership commits nothing.
    let ceilings: Vec<u64> = (0..=6)
        .map(|i| engine.commit_ceiling_through(i).expect("ceiling"))
        .collect();
    assert_eq!(ceilings, [0, 1000, 2000, 3000, 4000, 5000, 6000]);

    rebuild_schema(&engine, log.oplog_handle(), &ckpt).await;
    for (i, ceiling) in (0..=6).zip(&ceilings) {
        assert_eq!(
            engine.commit_ceiling_through(i).expect("ceiling"),
            *ceiling,
            "position {i} after the rebuild"
        );
    }

    // Every owner of the store closes; the reopened store reads the ledger
    // from the rebuilt tree alone.
    drop(sm);
    drop(log);
    engine.persist().expect("persist");
    drop(engine);
    let engine = open_dir(dir.path());
    let _log = LogStore::open(Arc::clone(&engine)).unwrap();
    let _sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("reopen");
    for (i, ceiling) in (0..=6).zip(&ceilings) {
        assert_eq!(
            engine.commit_ceiling_through(i).expect("ceiling"),
            *ceiling,
            "position {i} after the store reopened"
        );
    }
}

/// A proposal bound to a client operation: a counter increment bound to
/// operation 1 of incarnation 1 with intent `intent`, or (intent 0) that
/// incarnation's enrollment.
fn bound_entry(index: u64, id: u64, commit_ts: u64, intent: u8) -> Entry {
    use coordinode_core::txn::operation::{
        DEFAULT_ADMISSION_WINDOW, INTENT_DIGEST_LEN, ISSUER_ID_LEN, IntentDigest, IssuerId,
        OperationBinding, OperationCommand, OperationId,
    };
    use openraft::entry::RaftEntry;
    let issuer = IssuerId([0x5a; ISSUER_ID_LEN]);
    let mutations = if intent == 0 {
        vec![Mutation::Operation(OperationCommand::Enroll {
            issuer,
            incarnation: 1,
            window: DEFAULT_ADMISSION_WINDOW,
        })]
    } else {
        vec![
            Mutation::Merge {
                partition: PartitionId::Counter,
                key: b"counter:bound".to_vec(),
                operand: coordinode_storage::engine::merge::encode_counter_delta(1),
            },
            Mutation::Operation(OperationCommand::Bind(OperationBinding {
                op: OperationId {
                    issuer,
                    incarnation: 1,
                    sequence: 1,
                },
                intent: IntentDigest([intent; INTENT_DIGEST_LEN]),
                attempt: 100,
            })),
        ]
    };
    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(id),
        mutations,
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: false,
    };
    Entry::new_normal(log_id(1, index), Request::single(proposal))
}

/// The application rows of the Schema tree a rebuild compares: every key
/// but this member's own (`raft:`, `meta:`).
fn schema_rows(engine: &StorageEngine) -> Vec<(Vec<u8>, Vec<u8>)> {
    engine
        .prefix_scan(Partition::Schema, &[])
        .expect("scan")
        .map(|guard| {
            let (key, value) = guard.into_inner().expect("row");
            (key.to_vec(), value.to_vec())
        })
        .filter(|(key, _)| !key.starts_with(b"raft:") && !key.starts_with(b"meta:"))
        .collect()
}

/// A retransmitted proposal bound to a client operation (same id, same
/// size, a later commit) is decided again by its binding when a rebuild
/// replays it, as its apply decided it: the rebuilt Schema tree holds the
/// same records, the durable mark of the refused retransmission included,
/// and every position keeps its ceiling, the retransmission's commit too,
/// after the rebuild and after a restart. The rebuild's filter of repeated
/// proposals dropped it unmarked and left its commit out of the ledger.
#[tokio::test(flavor = "multi_thread")]
async fn a_rebuild_decides_a_retransmitted_bound_proposal_as_its_apply_did() {
    use openraft::entry::RaftEntry;
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_dir(dir.path());
    let mut log = LogStore::open(Arc::clone(&engine)).unwrap();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open");
    let membership = openraft::Membership::new(
        vec![std::collections::BTreeSet::from([1])],
        std::collections::BTreeMap::from([(1, MemberNode::new("http://node-1".into(), None))]),
    )
    .expect("a uniform configuration");
    let first = vec![
        Entry::new_membership(log_id(1, 0), membership),
        bound_entry(1, 900, 1_000, 0),
        bound_entry(2, 901, 2_000, 7),
    ];
    append_durable(&mut log, first.clone()).await;
    apply_entries(&mut sm, first).await;
    let ckpt_root = tempfile::tempdir().unwrap();
    let ckpt = ckpt_root.path().join("c");
    engine.create_checkpoint(&ckpt).unwrap();
    // Another intent for the same operation, same id and size: the binding
    // refuses it and marks the unit.
    let later = vec![
        bound_entry(3, 901, 3_000, 8),
        two_tree_merge_entry(4, 4_000),
    ];
    append_durable(&mut log, later.clone()).await;
    apply_entries(&mut sm, later).await;
    let refused = coordinode_core::txn::operation::LogPosition { index: 3, sub: 0 };
    let marked = |engine: &StorageEngine| {
        coordinode_storage::engine::operations::unit_skipped(engine, refused).expect("mark")
    };
    let ceilings_of = |engine: &StorageEngine, through: u64| -> Vec<u64> {
        (0..=through)
            .map(|i| engine.commit_ceiling_through(i).expect("ceiling"))
            .collect()
    };
    assert!(
        marked(&engine),
        "the apply marked the refused retransmission"
    );
    let rows = schema_rows(&engine);
    assert_eq!(ceilings_of(&engine, 4), [0, 1_000, 2_000, 3_000, 4_000]);

    rebuild_schema(&engine, log.oplog_handle(), &ckpt).await;
    assert_eq!(schema_rows(&engine), rows, "the rebuilt Schema rows");
    assert!(marked(&engine), "the rebuild kept the mark");
    assert_eq!(
        ceilings_of(&engine, 4),
        [0, 1_000, 2_000, 3_000, 4_000],
        "after the rebuild"
    );

    // The state machine that ran across the rebuild goes on.
    let five = two_tree_merge_entry(5, 5_000);
    append_durable(&mut log, vec![five.clone()]).await;
    apply_entries(&mut sm, vec![five]).await;
    let ceilings = [0, 1_000, 2_000, 3_000, 4_000, 5_000];
    assert_eq!(ceilings_of(&engine, 5), ceilings, "after the next entry");
    let rows = schema_rows(&engine);

    // Every owner of the store closes; the directory opens again.
    drop(sm);
    drop(log);
    engine.persist().expect("persist");
    drop(engine);
    let engine = open_dir(dir.path());
    let _log = LogStore::open(Arc::clone(&engine)).unwrap();
    let _sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("reopen");
    assert_eq!(
        schema_rows(&engine),
        rows,
        "the Schema rows after the store reopened"
    );
    assert!(marked(&engine), "the mark after the store reopened");
    assert_eq!(
        ceilings_of(&engine, 5),
        ceilings,
        "after the store reopened"
    );
}

/// One client or rate-limit-bypassing write at `commit_ts`, to the Node and
/// Schema trees: the Schema tree records the proposal's position too.
fn granted_entry(index: u64, commit_ts: u64, bypass: bool) -> Entry {
    use openraft::entry::RaftEntry;
    let proposal = RaftProposal {
        id: coordinode_core::txn::proposal::ProposalId::from_raw(10_000 + index),
        mutations: vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:1:g{index}").into_bytes(),
                value: vec![7u8; 100 + index as usize],
            },
            Mutation::Put {
                partition: PartitionId::Schema,
                key: format!("test:g{index}").into_bytes(),
                value: vec![1u8; 8],
            },
        ],
        commit_ts: Timestamp::from_raw(commit_ts),
        start_ts: Timestamp::from_raw(commit_ts - 1),
        bypass_rate_limiter: bypass,
    };
    Entry::new_normal(log_id(1, index), Request::single(proposal))
}

/// The client bytes members pay for `entry`.
fn entry_room(entry: &Entry) -> u64 {
    match &entry.payload {
        openraft::entry::EntryPayload::Normal(request) => request
            .proposals
            .iter()
            .filter(|p| !p.bypass_rate_limiter)
            .map(crate::cluster::grants::proposal_room)
            .sum(),
        _ => 0,
    }
}

/// The store at `dir`, opened as [`test_engine`] opens it.
fn open_dir(dir: &std::path::Path) -> Arc<StorageEngine> {
    Arc::new(
        StorageEngine::open(&StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]))
        .expect("open"),
    )
}

/// Rebuild the Schema tree of `engine` from the checkpoint at `ckpt` and the
/// log `oplog`.
async fn rebuild_schema(engine: &Arc<StorageEngine>, oplog: RaftLogHandle, ckpt: &std::path::Path) {
    let engine = Arc::clone(engine);
    let ckpt = ckpt.to_path_buf();
    tokio::task::spawn_blocking(move || {
        rebuild_partition_from_checkpoint(&engine, &oplog, &ckpt, Partition::Schema)
    })
    .await
    .expect("rebuild task")
    .expect("rebuild");
}

/// The Schema tree rebuilt from a checkpoint holds the group's applied
/// client bytes of the entries the rebuild replays, counted as their apply
/// counted them: client entries only, closing and bypassing ones adding
/// nothing, whatever order their commit timestamps came in. The checkpoint
/// is taken inside entry 4, after its proposal and before its granted-bytes
/// record, so it holds the one position and lacks the other. The total is
/// the applied one after the rebuild and after rebuilding again; the state
/// machine that ran across the rebuild adds the next entry's bytes once;
/// after every owner of the store closed and the directory reopened, the
/// reopened member reads the same total and adds the entry after once.
/// Replaying only the proposals left the member with the checkpoint's
/// total.
#[tokio::test(flavor = "multi_thread")]
async fn a_schema_rebuilt_from_a_checkpoint_keeps_the_granted_bytes() {
    use openraft::entry::RaftEntry;
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_dir(dir.path());
    let mut log = LogStore::open(Arc::clone(&engine)).unwrap();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open");
    let ckpt_root = tempfile::tempdir().unwrap();
    let ckpt = ckpt_root.path().join("c");
    {
        let (engine, ckpt) = (Arc::clone(&engine), ckpt.clone());
        sm.after_proposals = Some(Box::new(move |index| {
            if index == 4 {
                engine.create_checkpoint(&ckpt).expect("checkpoint");
            }
        }));
    }
    let membership = openraft::Membership::new(
        vec![std::collections::BTreeSet::from([1])],
        std::collections::BTreeMap::from([(1, MemberNode::new("http://node-1".into(), None))]),
    )
    .expect("a uniform configuration");
    let entries = vec![
        Entry::new_membership(log_id(1, 0), membership),
        granted_entry(1, 5_000, false),
        granted_entry(2, 4_000, true),
        granted_entry(3, 6_000, false),
        granted_entry(4, 3_000, false),
        Entry::new_normal(log_id(1, 5), Request::closing(2_000)),
        granted_entry(6, 9_000, true),
        granted_entry(7, 2_500, false),
    ];
    append_durable(&mut log, entries.clone()).await;
    apply_entries(&mut sm, entries).await;
    sm.after_proposals = None;
    let total = GrantedBytes::load(&engine).expect("total");
    assert_eq!(total.through, 7);
    assert_eq!(total, sm.granted_bytes);

    // The checkpoint holds entry 4's proposal and not its granted bytes.
    let (_, held) =
        StorageEngine::checkpoint_raft_partition(&ckpt, Partition::Schema).expect("checkpoint");
    assert!(held.holds(4, 0), "the checkpoint holds the proposal");
    assert!(!held.holds(4, 1), "the checkpoint lacks the granted bytes");

    rebuild_schema(&engine, log.oplog_handle(), &ckpt).await;
    assert_eq!(
        GrantedBytes::load(&engine).expect("total"),
        total,
        "after the rebuild"
    );

    // The state machine that ran across the rebuild goes on.
    let eight = granted_entry(8, 1_000, false);
    let room8 = entry_room(&eight);
    assert!(room8 > 0, "entry 8 is a client write");
    append_durable(&mut log, vec![eight.clone()]).await;
    apply_entries(&mut sm, vec![eight]).await;
    let total8 = GrantedBytes {
        through: 8,
        total: total.total + room8,
    };
    assert_eq!(GrantedBytes::load(&engine).expect("total"), total8);
    assert_eq!(sm.granted_bytes, total8);
    rebuild_schema(&engine, log.oplog_handle(), &ckpt).await;
    assert_eq!(
        GrantedBytes::load(&engine).expect("total"),
        total8,
        "after rebuilding again"
    );

    // Every owner of the store closes; the directory opens again.
    drop(sm);
    drop(log);
    engine.persist().expect("persist");
    drop(engine);
    let engine = open_dir(dir.path());
    let mut log = LogStore::open(Arc::clone(&engine)).unwrap();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("reopen");
    assert_eq!(sm.granted_bytes, total8, "after the store reopened");
    let nine = granted_entry(9, 1_500, false);
    let room9 = entry_room(&nine);
    append_durable(&mut log, vec![nine.clone()]).await;
    apply_entries(&mut sm, vec![nine]).await;
    assert_eq!(
        GrantedBytes::load(&engine).expect("total"),
        GrantedBytes {
            through: 9,
            total: total8.total + room9,
        }
    );
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

/// The group's applied client bytes as `engine`'s applied state records
/// them.
fn granted(engine: &StorageEngine) -> GrantedBytes {
    GrantedBytes::load(engine).expect("read the granted bytes")
}

/// Entries apply in log order but may carry commit timestamps out of order:
/// the later entry, stamped earlier, still records the current total, read
/// back as such at once, after the store is closed and opened again, on a
/// follower that installs a snapshot of it, and counted on from by the next
/// entry. Before, the total was one key whose versions were ordered by the
/// entries' timestamps, so the earlier-stamped later total hid under the
/// earlier one.
#[tokio::test(flavor = "multi_thread")]
async fn the_applied_client_bytes_follow_the_log_not_the_timestamps() {
    let rig = coordinode_test_fixtures::PowerRig::new();
    let (engine, oracle) = open_rig_engine(&rig);
    let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()))
        .expect("open");
    let (early, late) = (oracle.next().as_raw(), oracle.next().as_raw());
    let first = node_key_entry(1, late, b"node:1:a", b"a1");
    let second = node_key_entry(2, early, b"node:1:b", b"b1");
    let room = |entry: &Entry| crate::cluster::grants::entry_room(entry).1;
    let total = GrantedBytes {
        through: 2,
        total: room(&first) + room(&second),
    };
    let entries = vec![first, second];
    apply_entries(&mut sm, entries.clone()).await;
    assert_eq!(granted(&engine), total, "read back at once");

    // A clean close and a real reopen of the store; openraft delivers the
    // entries past the folded coverage again.
    drop(sm);
    drop(engine);
    let (engine, oracle) = open_rig_engine(&rig);
    let mut sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()))
        .expect("reopen");
    assert_eq!(granted(&engine), total, "after a reopen");
    assert_eq!(sm.granted_bytes, total);
    apply_entries(&mut sm, entries).await;
    assert_eq!(granted(&engine), total, "counted once when delivered again");

    let (meta, bytes) = sent(pinned(&mut sm).await);
    let follower_rig = coordinode_test_fixtures::PowerRig::new();
    let (follower, follower_oracle) = open_rig_engine(&follower_rig);
    let mut fsm = CoordinodeStateMachine::with_oracle(Arc::clone(&follower), Some(follower_oracle))
        .expect("open follower");
    fsm.install_snapshot(&meta, received(&follower, &bytes))
        .await
        .expect("install");
    assert_eq!(granted(&follower), total, "carried by a snapshot");

    // Stamped before the totals already written, as an entry may be.
    let next = node_key_entry(3, early, b"node:1:c", b"c1");
    let after = GrantedBytes {
        through: 3,
        total: total.total + room(&next),
    };
    apply_entries(&mut sm, vec![next.clone()]).await;
    apply_entries(&mut fsm, vec![next]).await;
    assert_eq!(granted(&engine), after);
    assert_eq!(granted(&follower), after, "the follower counts on");
    assert_eq!((granted_keys(&engine), granted_keys(&follower)), (1, 1));
}

/// The keys holding the group's applied client bytes in `engine`.
fn granted_keys(engine: &StorageEngine) -> usize {
    engine
        .prefix_scan(Partition::Schema, b"group:granted-bytes")
        .expect("scan")
        .count()
}

/// Client entries without a commit timestamp (a store without an oracle
/// stamps none) keep the group's total exact and its state bounded: one
/// record whatever the number of entries, through applies, a purge-free
/// reopen with redelivery, a snapshot and the entries after it. Before, every
/// total was a key of its own, removed only by an entry stamped after it,
/// which an unstamped entry never is: one key per entry, kept for ever.
#[tokio::test]
async fn unstamped_client_entries_keep_one_total() {
    let (_dir, engine) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("open");
    let entries: Vec<Entry> = (1..=50u64)
        .map(|i| node_key_entry(i, 0, format!("node:1:{i}").as_bytes(), b"v"))
        .collect();
    let total = GrantedBytes {
        through: 50,
        total: entries
            .iter()
            .map(|entry| crate::cluster::grants::entry_room(entry).1)
            .sum(),
    };
    apply_entries(&mut sm, entries.clone()).await;
    assert_eq!(granted(&engine), total);
    assert_eq!(granted_keys(&engine), 1, "one record for 50 entries");

    drop(sm);
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine)).expect("reopen");
    apply_entries(&mut sm, entries).await;
    assert_eq!(granted(&engine), total, "counted once when delivered again");
    assert_eq!(granted_keys(&engine), 1);

    let (meta, bytes) = sent(pinned(&mut sm).await);
    let (_dir2, follower) = test_engine();
    let mut fsm = CoordinodeStateMachine::new(Arc::clone(&follower)).expect("open follower");
    fsm.install_snapshot(&meta, received(&follower, &bytes))
        .await
        .expect("install");
    assert_eq!(granted(&follower), total);
    let next = node_key_entry(51, 0, b"node:1:51", b"v");
    let after = GrantedBytes {
        through: 51,
        total: total.total + crate::cluster::grants::entry_room(&next).1,
    };
    apply_entries(&mut fsm, vec![next]).await;
    assert_eq!(granted(&follower), after);
    assert_eq!(granted_keys(&follower), 1);
}

/// The group's applied client bytes are part of the applied state: written
/// with each client entry, counted once when an entry is delivered again,
/// and carried by a snapshot, so a follower that installs one in place of
/// entries it never received reads the leader's total at the same index.
/// Without it the follower knows nothing of what those entries spent of
/// the room it granted.
#[tokio::test]
async fn the_applied_client_bytes_travel_with_a_snapshot() {
    let (_dir, leader) = test_engine();
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&leader)).expect("open");
    let entries = vec![
        node_key_entry(1, 1000, b"node:1:a", b"a1"),
        node_key_entry(2, 2000, b"node:1:b", b"b1"),
    ];
    let expected: u64 = entries
        .iter()
        .map(|entry| crate::cluster::grants::entry_room(entry).1)
        .sum();
    assert!(expected > 0, "client entries carry bytes");
    apply_entries(&mut sm, entries.clone()).await;
    let total = GrantedBytes {
        through: 2,
        total: expected,
    };
    assert_eq!(granted(&leader), total);

    // Delivered again after a reopen: counted already.
    drop(sm);
    let mut sm = CoordinodeStateMachine::new(Arc::clone(&leader)).expect("reopen");
    apply_entries(&mut sm, entries).await;
    assert_eq!(granted(&leader), total);

    let (meta, bytes) = sent(pinned(&mut sm).await);
    let (_dir2, follower) = test_engine();
    let mut fsm = CoordinodeStateMachine::new(Arc::clone(&follower)).expect("open follower");
    assert_eq!(granted(&follower), GrantedBytes::default());
    fsm.install_snapshot(&meta, received(&follower, &bytes))
        .await
        .expect("install");
    assert_eq!(granted(&follower), total);
    assert_eq!(fsm.granted_bytes, total, "the state machine reads it");

    // The next entry counts on from the snapshot's total on both.
    let next = node_key_entry(3, 3000, b"node:1:c", b"c1");
    let room = crate::cluster::grants::entry_room(&next).1;
    apply_entries(&mut sm, vec![next.clone()]).await;
    apply_entries(&mut fsm, vec![next]).await;
    let after = GrantedBytes {
        through: 3,
        total: expected + room,
    };
    assert_eq!(granted(&leader), after);
    assert_eq!(granted(&follower), after);
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
            // An unstamped entry (no oracle) carries no start either.
            start_ts: Timestamp::from_raw(if ts == 0 { 0 } else { ts - 1 }),
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
    lose_the_copy(dir.path());
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    let err = LogStore::open(Arc::clone(&engine))
        .err()
        .expect("a member without its promise must not open");
    assert!(PromiseUnqualified::of(&err).is_some(), "typed: {err}");
    assert!(
        !copy_exists(dir.path()),
        "the refusal wrote no promise of its own"
    );
}

/// The slot files of the second copy of the store at `dir`.
fn copy_slots(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    coordinode_storage::engine::promise::copy_files(&lsm_tree::fs::StdFs, dir)
        .expect("list the copy")
        .into_iter()
        .map(|(_, path)| path)
        .collect()
}

/// Lose every slot of the second copy of the store at `dir`.
fn lose_the_copy(dir: &std::path::Path) {
    for slot in copy_slots(dir) {
        std::fs::remove_file(slot).expect("lose a slot of the copy");
    }
}

/// Whether any slot of the second copy of the store at `dir` exists.
fn copy_exists(dir: &std::path::Path) -> bool {
    !copy_slots(dir).is_empty()
}

/// Damage the last byte of every slot of the second copy at `dir`.
fn damage_the_copy(dir: &std::path::Path) {
    for slot in copy_slots(dir) {
        let mut bytes = std::fs::read(&slot).expect("read a slot");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&slot, &bytes).expect("damage a slot");
    }
}

/// Damage the Raft tree of the store at `dir` so the next open salvages it
/// with loss: its manifest is gone, which makes the open rebuild the tree
/// from its tables, and every table lost its second half, which that
/// salvage cannot read.
fn salvage_the_raft_tree_with_loss(dir: &std::path::Path) {
    let tree = dir.join(Partition::Raft.name());
    for entry in std::fs::read_dir(&tree).expect("read the tree") {
        let path = entry.expect("entry").path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let manifest = name
            .strip_prefix('v')
            .or_else(|| name.strip_prefix("edits-"))
            .is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()));
        if manifest {
            std::fs::remove_file(&path).expect("lose a manifest file");
        }
    }
    let tables = tree.join("tables");
    for entry in std::fs::read_dir(&tables).expect("read the tables") {
        let path = entry.expect("entry").path();
        let len = std::fs::metadata(&path).expect("stat").len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open a table")
            .set_len(len / 2)
            .expect("lose its second half");
    }
}

/// The open-time salvage of a damaged Raft tree reads back clean but
/// incomplete: the vote's record is gone though nothing errs. That readable
/// absence is not taken for a member that never voted; the second copy
/// gives the vote back.
#[tokio::test]
async fn a_lossy_salvage_of_the_raft_tree_keeps_the_vote() {
    let dir = tempfile::tempdir().expect("tempdir");
    let vote = Vote::new(8, 2);
    let config = compacting_config(dir.path());
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
        log.save_vote(&vote).await.expect("save");
        engine
            .persist_partition(Partition::Raft)
            .expect("the vote in a table");
    }
    salvage_the_raft_tree_with_loss(dir.path());
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    assert!(
        engine
            .open_repairs()
            .iter()
            .any(|r| r.partition == Partition::Raft),
        "the open salvaged the Raft tree: {:?}",
        engine.open_repairs()
    );
    assert_eq!(
        engine.get(Partition::Raft, KEY_VOTE).expect("readable"),
        None,
        "the salvaged tree lost the vote's record without an error"
    );
    let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    assert_eq!(log.read_vote().await.expect("read"), Some(vote));
}

/// Lose the Raft tree of the store at `dir` and its member-local journal:
/// every record the journal no longer holds is gone with the tree, as after
/// a fold and a purge.
fn lose_the_tree_and_its_journal(dir: &std::path::Path) {
    std::fs::remove_dir_all(dir.join(Partition::Raft.name())).expect("lose the tree");
    std::fs::remove_dir_all(dir.join("local-journal")).expect("lose the local journal");
}

/// The purge point goes back to the tree with the vote: a member that lost
/// its Raft tree does not read entries its log no longer holds as if
/// nothing were purged.
#[tokio::test]
async fn the_purge_point_survives_the_loss_of_the_raft_tree() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = compacting_config(dir.path());
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        let mut store = LogStore::open(Arc::clone(&engine)).expect("open log");
        store.save_vote(&Vote::new(1, 1)).await.expect("vote");
        append_durable(
            &mut store,
            vec![
                make_entry(1, 1, "a"),
                make_entry(2, 1, "b"),
                make_entry(3, 1, "c"),
            ],
        )
        .await;
        engine.reset_raft_coverage(3, &[]).expect("coverage");
        store.purge(log_id(1, 2)).await.expect("purge");
    }
    lose_the_tree_and_its_journal(dir.path());
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    let mut store = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    let state = store.get_log_state().await.expect("state");
    assert_eq!(
        state.last_purged_log_id,
        Some(log_id(1, 2)),
        "the purge point came back from the second copy"
    );
    assert_eq!(
        store.read_vote().await.expect("vote"),
        Some(Vote::new(1, 1))
    );
}

/// A recorded loss of committed entries goes back to the tree with the
/// vote: a member that lost its Raft tree while catching up stays out of
/// elections until it holds the lost entries again.
#[tokio::test]
async fn a_recorded_loss_survives_the_loss_of_the_raft_tree() {
    let (dir, config) = log_with_a_hole_at_3(5).await;
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen engine"));
        let (_, through) =
            LogStore::open_with(Arc::clone(&engine), LostCommitted::CatchUp).expect("open");
        assert_eq!(through, Some(log_id(1, 5)));
    }
    lose_the_tree_and_its_journal(dir.path());
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen engine"));
    let err = LogStore::open(Arc::clone(&engine))
        .err()
        .expect("alone, the recorded loss still refuses the member");
    assert!(err.to_string().contains("index 5"), "{err}");
    let (_, through) =
        LogStore::open_with(Arc::clone(&engine), LostCommitted::CatchUp).expect("open");
    assert_eq!(
        through,
        Some(log_id(1, 5)),
        "the loss came back from the copy"
    );
}

/// The second copy's payload for a member that holds `vote` and has neither
/// purged its log nor lost committed entries.
fn promises_of(vote: Option<Vote>) -> Vec<u8> {
    rmp_serde::to_vec(&Promises {
        member: None,
        vote,
        purged: None,
        catch_up: None,
    })
    .expect("encode")
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
            payload: promises_of(None),
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
    damage_the_copy(dir.path());
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
                payload: promises_of(Some(newer)),
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
            .write_promise_copy(&incarnation, &promises_of(Some(ahead)))
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
            .write_promise_copy(&[0xee; 16], &promises_of(Some(Vote::new(50, 9))))
            .expect("foreign copy");
    }
    {
        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
        assert_eq!(log.read_vote().await.expect("read"), Some(mine));
        engine
            .write_promise_copy(&[0xee; 16], &promises_of(Some(Vote::new(50, 9))))
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
    lose_the_copy(dir.path());
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
    lose_the_copy(dir.path());
    std::fs::remove_dir_all(dir.path().join(Partition::Raft.name())).expect("lose the tree");
    let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
    let err = LogStore::open(Arc::clone(&engine))
        .err()
        .expect("a member with a log and no vote anywhere lost it");
    assert!(PromiseUnqualified::of(&err).is_some(), "typed: {err}");
}

/// A conversion cut short after the second copy was written and before the
/// marker took its incarnation is finished by the next open under the
/// copy's incarnation, not a newly drawn one; the copy alone still proves
/// the vote when the tree's record is lost as well.
#[tokio::test]
async fn a_conversion_cut_short_before_the_marker_is_finished_by_the_next_open() {
    let incarnation = [0x5a; 16];
    let vote = Vote::new(6, 1);
    for lose_the_tree in [false, true] {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = compacting_config(dir.path());
        {
            let engine = Arc::new(StorageEngine::open(&config).expect("open"));
            let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
            log.save_vote(&vote).await.expect("save");
            append_durable(&mut log, vec![make_entry(1, 6, "x")]).await;
        }
        // The earlier layout, then the conversion's copy write and nothing
        // after it: the marker is still empty.
        std::fs::write(dir.path().join("consensus-log"), b"").expect("empty marker");
        lose_the_copy(dir.path());
        {
            let engine = StorageEngine::open(&config).expect("reopen");
            engine
                .write_promise_copy(&incarnation, &promises_of(Some(vote)))
                .expect("the conversion's copy");
        }
        if lose_the_tree {
            std::fs::remove_dir_all(dir.path().join(Partition::Raft.name()))
                .expect("lose the tree");
        }

        let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
        let mut log = LogStore::open(Arc::clone(&engine)).expect("reopen log");
        assert_eq!(
            log.read_vote().await.expect("read"),
            Some(vote),
            "tree lost: {lose_the_tree}"
        );
        assert!(
            matches!(
                engine.consensus_binding().expect("binding"),
                ConsensusBinding::Bound(bound) if bound == incarnation
            ),
            "tree lost: {lose_the_tree}: the marker must take the copy's incarnation"
        );
        assert!(
            matches!(
                engine.read_promise_copy().expect("copy"),
                PromiseCopy::Held { incarnation: held, .. } if held == incarnation
            ),
            "tree lost: {lose_the_tree}"
        );
    }
}

/// An unchanged vote costs no write in either copy: openraft may save the
/// vote it already holds, and only a change needs a sync. The tree's copy
/// is made durable by a member-local journal sync, so a rewrite shows in
/// the journal's syncs even where the second copy is left alone.
#[tokio::test]
async fn saving_the_same_vote_again_writes_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = compacting_config(dir.path());
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));
    let mut log = LogStore::open(Arc::clone(&engine)).expect("open log");
    let vote = Vote::new(2, 1);
    log.save_vote(&vote).await.expect("save");
    let syncs = engine.journal_syncs().expect("a member-local journal");
    lose_the_copy(dir.path());
    log.save_vote(&vote).await.expect("save the same");
    assert!(
        !copy_exists(dir.path()),
        "an unchanged vote must not be written again"
    );
    assert_eq!(
        engine.journal_syncs(),
        Some(syncs),
        "an unchanged vote must not sync the local journal again"
    );
    log.save_vote(&Vote::new(3, 1))
        .await
        .expect("save a change");
    assert!(copy_exists(dir.path()), "a changed vote is written");
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
        let adj = engine.tree(Partition::Adj).unwrap();
        adj.flush_active_memtable(0).unwrap();
        adj.persist_flushes().unwrap();
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
        let schema = engine.tree(Partition::Schema).unwrap();
        schema.flush_active_memtable(0).unwrap();
        schema.persist_flushes().unwrap();
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
            // One tree's flush made durable alone.
            9 => engine
                .tree(Partition::Adj)
                .map_err(|e| e.to_string())
                .and_then(|t| {
                    t.flush_active_memtable(0).map_err(|e| e.to_string())?;
                    t.persist_flushes().map_err(|e| e.to_string())
                }),
            // One tree's flush left to the next barrier, as the flush workers
            // leave it: the log purge below must not pass it.
            11 => engine
                .tree(Partition::Node)
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

/// A Raft log whose write fails, as on a full disk, appends nothing more:
/// the batch that failed is refused, and so is every batch after it, so no
/// entry is written or acknowledged past one that may be lost. Reopened,
/// the log holds an unbroken run of entries ending at what is durable.
#[tokio::test]
async fn a_failed_log_write_stops_the_raft_log() {
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
    {
        let engine = Arc::new(StorageEngine::open(&config()).expect("open"));
        let mut store = LogStore::open(Arc::clone(&engine)).expect("open log");
        // The answer openraft gets for an append: durable, or the failure.
        async fn appended(store: &mut LogStore, entry: Entry) -> Result<(), io::Error> {
            let (tx, rx) = <TypeConfig as openraft::type_config::TypeConfigExt>::oneshot();
            store
                .append(vec![entry], IOFlushed::signal(tx))
                .await
                .expect("an append is taken; the writer answers it");
            rx.await.expect("answered")
        }
        appended(&mut store, make_entry(1, 1, "a"))
            .await
            .expect("1 durable");
        appended(&mut store, make_entry(2, 1, "b"))
            .await
            .expect("2 durable");
        store.oplog.lock().expect("oplog").fail_writes();
        assert!(
            appended(&mut store, make_entry(3, 1, "c")).await.is_err(),
            "the write fails"
        );
        assert!(
            appended(&mut store, make_entry(4, 1, "d")).await.is_err(),
            "nothing is acknowledged past the failure"
        );
    }
    let engine = Arc::new(StorageEngine::open(&config()).expect("reopen"));
    let mut store = LogStore::open(Arc::clone(&engine)).expect("reopen log");
    let entries = store.try_get_log_entries(1..).await.expect("read");
    let indexes: Vec<u64> = entries.iter().map(|e| e.log_id.index).collect();
    assert!(
        indexes == [1, 2] || indexes == [1, 2, 3],
        "an unbroken log ending at what is durable: {indexes:?}"
    );
}

#[tokio::test]
async fn log_store_purge_persists_last_purged_log_id() {
    let (_dir, engine) = test_engine();
    let mut store = LogStore::open(Arc::clone(&engine)).unwrap();

    let entries = vec![
        make_entry(1, 1, "a"),
        make_entry(2, 1, "b"),
        make_entry(3, 1, "c"),
    ];
    append_durable(&mut store, entries).await;
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
        append_durable(&mut store, entries).await;
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
