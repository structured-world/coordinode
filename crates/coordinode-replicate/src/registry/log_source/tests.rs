use super::*;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::manager::OplogManager;

/// Microseconds per second: entry timestamps are HLC microseconds.
const US_PER_SEC: u64 = 1_000_000;

fn endpoint(dir: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// A fresh engine without a journal of its own.
fn open_engine() -> (Arc<StorageEngine>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_with_oracle(&endpoint(dir.path()), oracle).expect("open"));
    (engine, dir)
}

/// The floor of a log that holds no segment.
fn empty_floor() -> RetainedFloor {
    let dir = tempfile::tempdir().expect("tempdir");
    OplogManager::open(dir.path(), 0, 1 << 20, 1_000, 3_600)
        .expect("open oplog")
        .retained_floor()
}

/// Write entries `0..count` into `dir`, sealing a segment after each index in
/// `seal_after`; entry `i` carries timestamp `1_000_000 * (i + 1)` µs.
fn write_log(dir: &std::path::Path, count: u64, seal_after: &[u64]) -> OplogManager {
    let mut mgr =
        OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open oplog");
    for index in 0..count {
        mgr.append(&OplogEntry {
            ts: US_PER_SEC * (index + 1),
            term: 1,
            index,
            shard: 0,
            ops: vec![OplogOp::Insert {
                partition: 0,
                key: format!("node:{index}").into_bytes(),
                value: vec![0u8; 64],
            }],
            is_migration: false,
            pre_images: None,
        })
        .expect("append");
        if seal_after.contains(&index) {
            mgr.rotate().expect("seal");
        }
    }
    mgr.flush().expect("flush");
    mgr
}

/// Write one entry per element of `stamps` into `dir`, entry `i` carrying
/// `stamps[i]` as its timestamp (0: an entry with no commit, such as the
/// closing entry a leader appends after a burst of commits).
fn write_stamped_log(dir: &std::path::Path, stamps: &[u64]) -> OplogManager {
    let mut mgr =
        OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open oplog");
    for (index, &ts) in (0u64..).zip(stamps) {
        mgr.append(&OplogEntry {
            ts,
            term: 1,
            index,
            shard: 0,
            ops: vec![OplogOp::Noop],
            is_migration: false,
            pre_images: None,
        })
        .expect("append");
    }
    mgr.flush().expect("flush");
    mgr
}

/// The source answers for the log from the segments it holds: the oldest
/// retained index after a purge, the head, the write time of an entry in
/// milliseconds, and the bytes a consumer at a position keeps on disk, which
/// shrink as it advances.
#[test]
fn the_source_measures_the_log_it_holds() {
    let (engine, _engine_dir) = open_engine();
    let log_dir = tempfile::tempdir().expect("log dir");
    let mut mgr = write_log(log_dir.path(), 9, &[2, 5]);
    assert_eq!(mgr.purge_before(3).expect("purge"), 1);
    let source = LogRetentionSource::new(
        Arc::clone(&engine),
        vec![log_dir.path().to_path_buf()],
        mgr.retained_floor(),
        Arc::new(|| 9),
    );
    let kind = ConsumerKind::OplogEvents;

    assert!(source.accounts(kind));
    assert_eq!(source.head(kind), 9);
    assert_eq!(source.first_retained(kind), 3);
    assert_eq!(source.produced_at_ms(kind, 4), Ok(Some(5_000)));
    assert_eq!(
        source.produced_at_ms(kind, 9),
        Ok(None),
        "nothing written there yet"
    );

    let from_oldest = source.retained_bytes_from(kind, 3).expect("bytes");
    let from_last_segment = source.retained_bytes_from(kind, 6).expect("bytes");
    assert!(
        from_oldest > from_last_segment,
        "{from_oldest} vs {from_last_segment}"
    );
    assert!(from_last_segment > 0, "the open segment is still kept");
}

/// The registry's bound sweep asks the age of every lagging BOUNDED
/// consumer's checkpoint twice a sweep, every second. Each ask used to open a
/// fresh reader that read and decoded the whole segment holding the
/// checkpoint: tens of MB/s on a busy log with two consumers. Asking the same
/// checkpoint again must read nothing, and a moved one only near it.
#[test]
fn the_source_answers_an_unmoved_checkpoint_without_reading_the_log() {
    let (engine, _engine_dir) = open_engine();
    let log_dir = tempfile::tempdir().expect("log dir");
    let _mgr = write_log(log_dir.path(), 3_000, &[]);
    let source = LogRetentionSource::new(
        Arc::clone(&engine),
        vec![log_dir.path().to_path_buf()],
        empty_floor(),
        Arc::new(|| 3_000),
    );
    let kind = ConsumerKind::OplogEvents;

    assert_eq!(source.produced_at_ms(kind, 2_999), Ok(Some(3_000_000)));
    let scanned = source.timeline.bytes_read();
    assert!(
        scanned > 0,
        "answered from the kept timeline, not a fresh read"
    );
    for _ in 0..50 {
        assert_eq!(source.produced_at_ms(kind, 100), Ok(Some(101_000)));
    }
    let first_ask = source.timeline.bytes_read() - scanned;
    assert!(first_ask < 256 * 1024, "one ask read {first_ask} bytes");
    assert_eq!(source.produced_at_ms(kind, 100), Ok(Some(101_000)));
    assert_eq!(
        source.timeline.bytes_read() - scanned,
        first_ask,
        "an unmoved checkpoint reads nothing more"
    );

    // Released below a higher floor, the old checkpoint's answer goes and
    // the segment holding the floor stays scanned.
    source.release_log_below(2_000);
    let before = source.timeline.bytes_read();
    assert_eq!(source.produced_at_ms(kind, 2_500), Ok(Some(2_501_000)));
    assert!(source.timeline.bytes_read() - before < 256 * 1024);
}

/// An entry that carries no commit (a leader's closing entry, its first
/// entry of a term, a membership change) is no work a consumer must
/// acknowledge: the age of what a consumer owes is the write time of the
/// first committed entry from its checkpoint on, and with none up to the
/// head it owes nothing. Aging such an entry from timestamp 0 made every
/// BOUNDED consumer whose checkpoint reached a closing entry exceed any
/// progress lag at once.
#[test]
fn entries_without_a_commit_owe_no_progress() {
    let (engine, _engine_dir) = open_engine();
    let log_dir = tempfile::tempdir().expect("log dir");
    let t = |s: u64| US_PER_SEC * s;
    let _mgr = write_stamped_log(log_dir.path(), &[t(1), t(2), 0, t(4), 0, 0]);
    let source = LogRetentionSource::new(
        Arc::clone(&engine),
        vec![log_dir.path().to_path_buf()],
        empty_floor(),
        Arc::new(|| 6),
    );
    let kind = ConsumerKind::OplogEvents;

    assert_eq!(source.produced_at_ms(kind, 1), Ok(Some(2_000)));
    assert_eq!(
        source.produced_at_ms(kind, 2),
        Ok(Some(4_000)),
        "a closing entry ages from the commit after it"
    );
    assert_eq!(
        source.produced_at_ms(kind, 4),
        Ok(None),
        "nothing but closing entries up to the head is no outstanding work"
    );
}

/// A log with no segment keeps nothing below the head, so a new consumer
/// cannot claim a position under it.
#[test]
fn a_log_without_segments_retains_nothing_below_the_head() {
    let (engine, _engine_dir) = open_engine();
    let log_dir = tempfile::tempdir().expect("log dir");
    let mgr = OplogManager::open(log_dir.path(), 0, 1 << 20, 1_000, 3_600).expect("open oplog");
    let source = LogRetentionSource::new(
        Arc::clone(&engine),
        vec![log_dir.path().to_path_buf()],
        mgr.retained_floor(),
        Arc::new(|| 4),
    );
    assert_eq!(source.first_retained(ConsumerKind::OplogEvents), 4);
    assert_eq!(
        source
            .hold_log_from(3)
            .expect_err("below the log is refused"),
        4
    );
    assert!(
        source.hold_log_from(4).expect("held").is_none(),
        "a group's log takes no holds"
    );
}

/// A pinned snapshot's material is every version visible at its cut, which
/// the store does not account: BOUNDED is refused for it rather than
/// enforced on numbers that do not exist.
#[test]
fn the_source_does_not_account_snapshot_pins() {
    let (engine, _engine_dir) = open_engine();
    let source = LogRetentionSource::new(
        Arc::clone(&engine),
        Vec::new(),
        empty_floor(),
        Arc::new(|| 0),
    );
    for kind in [ConsumerKind::MvccSnapshotPin, ConsumerKind::Ephemeral] {
        assert!(kind.is_seqno_space());
        assert!(!source.accounts(kind));
        assert_eq!(source.head(kind), engine.snapshot());
        assert_eq!(source.first_retained(kind), engine.gc_watermark());
        assert_eq!(
            source.produced_at_ms(kind, 1),
            Err(Unmeasured::NotAccounted)
        );
        assert_eq!(
            source.retained_bytes_from(kind, 1),
            Err(Unmeasured::NotAccounted)
        );
    }
}

/// A state reader's work and material from its checkpoint: an idle store
/// owes it nothing, however long it waits; its first write starts the age at
/// that write's commit time, not at the checkpoint; a commit applied out of
/// timestamp order still counts from its own time; and the bytes are the
/// logical size of every commit from the checkpoint on.
#[test]
fn a_state_reader_owes_the_commits_from_its_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let start = 4_000_000_000_000_000;
    let oracle = Arc::new(TimestampOracle::resume_from(
        coordinode_core::txn::timestamp::Timestamp::from_raw(start),
    ));
    let engine =
        Arc::new(StorageEngine::open_with_oracle(&endpoint(dir.path()), oracle).expect("open"));
    let source = LogRetentionSource::new(
        Arc::clone(&engine),
        Vec::new(),
        empty_floor(),
        Arc::new(|| 0),
    );
    let kind = ConsumerKind::LsmStateDelta;
    assert!(source.accounts(kind));
    let checkpoint = start + 1_000_000;
    assert_eq!(source.produced_at_ms(kind, checkpoint), Ok(None), "idle");
    assert_eq!(source.retained_bytes_from(kind, checkpoint), Ok(0));

    let put = |key: &[u8], len: usize, ts: u64| {
        engine
            .apply_proposal_at(
                &[coordinode_core::txn::proposal::Mutation::Put {
                    partition: coordinode_core::txn::proposal::PartitionId::Node,
                    key: key.to_vec(),
                    value: vec![0; len],
                }],
                ts,
            )
            .expect("apply");
    };
    // The first write lands long after the checkpoint.
    put(b"node:late", 10, checkpoint + 60_000_000);
    assert_eq!(
        source.produced_at_ms(kind, checkpoint),
        Ok(Some((checkpoint + 60_000_000) / 1_000))
    );
    // An earlier commit applied after it is older work.
    put(b"node:early", 20, checkpoint + 5_000_000);
    assert_eq!(
        source.produced_at_ms(kind, checkpoint),
        Ok(Some((checkpoint + 5_000_000) / 1_000))
    );
    assert_eq!(
        source.retained_bytes_from(kind, checkpoint),
        Ok((9 + 10) + (10 + 20))
    );
    // From between the two, only the later one is owed.
    assert_eq!(
        source.retained_bytes_from(kind, checkpoint + 5_000_001),
        Ok(9 + 10)
    );
}

/// Over a store's own journal the source reports the journal's head and
/// start, and a hold it gives keeps the journal from its position while it
/// lives; a store without a journal gives no such source.
#[test]
fn over_its_own_journal_the_source_holds_what_it_admits() {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use coordinode_storage::engine::oplog_journal::OplogJournalConfig;

    let (plain, _plain_dir) = open_engine();
    assert!(
        LogRetentionSource::over_own_journal(plain)
            .expect("inspect")
            .is_none()
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(
        StorageEngine::open_embedded_with_journal(
            &endpoint(dir.path()),
            Arc::clone(&oracle),
            OplogJournalConfig {
                retention_secs: 1,
                max_segment_entries: 2,
                ..OplogJournalConfig::default()
            },
        )
        .expect("open embedded"),
    );
    for i in 0..6u64 {
        engine
            .commit_journaled(
                &[Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:j:{i}").into_bytes(),
                    value: b"v".to_vec(),
                }],
                oracle.next().as_raw(),
            )
            .expect("commit");
    }
    let source = LogRetentionSource::over_own_journal(Arc::clone(&engine))
        .expect("inspect")
        .expect("a journal");
    let kind = ConsumerKind::OplogEvents;
    let head = source.head(kind);
    assert!(head >= 6, "the head is past the six entries, got {head}");
    let hold = source
        .hold_log_from(source.first_retained(kind))
        .expect("held")
        .expect("the store's own journal takes a hold");
    engine
        .oplog_purge_expired(4_000_000_000, u64::MAX)
        .expect("purge");
    assert!(
        source.first_retained(kind) <= hold.index(),
        "a purge passed the hold"
    );
    drop(hold);
    engine
        .oplog_purge_expired(4_000_000_000, u64::MAX)
        .expect("purge");
    let first = source.first_retained(kind);
    assert!(first > 0, "the released journal was purged");
    assert_eq!(
        source
            .hold_log_from(0)
            .expect_err("below the purged journal is refused"),
        first
    );
}
