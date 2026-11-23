use std::path::PathBuf;
use std::sync::{Arc, Barrier};

use coordinode_core::budget::QueryBudget;
use coordinode_core::graph::node::{NodeId, NodeRecord, encode_node_key};
use coordinode_core::txn::proposal::{Mutation, NodeBasis, PartitionId, UnitEffects};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};

use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::error::StorageError;
use crate::oplog::entry::{OplogEntry, OplogOp};
use crate::oplog::manager::OplogManager;

/// A store without a journal of its own, as a member of a group is: its
/// change feeds read a log a writer keeps beside it.
fn open_engine(dir: &std::path::Path) -> StorageEngine {
    StorageEngine::open_with_oracle(
        &StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]),
        Arc::new(TimestampOracle::new()),
    )
    .expect("open engine")
}

/// The entry at `index` carrying one unit writing node `node`, committed at
/// `ts`, as a commit writes it.
fn unit_entry(index: u64, ts: u64, node: u64) -> OplogEntry {
    let key = encode_node_key(0, NodeId::from_raw(node));
    let mut effects = UnitEffects::new();
    effects.nodes = vec![NodeBasis {
        key: key.clone(),
        labels: Vec::new(),
    }];
    let frame = coordinode_core::txn::frame::encode_unit(
        &crate::engine::scope::store_unit([
            Mutation::Put {
                partition: PartitionId::Node,
                key,
                value: NodeRecord::with_labels(Vec::new())
                    .to_msgpack()
                    .expect("record"),
            },
            Mutation::Effects(effects),
        ]),
        Timestamp::from_raw(ts),
    )
    .expect("frame");
    OplogEntry {
        ts,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Unit { frame }],
        is_migration: false,
        pre_images: None,
    }
}

/// A log in `dir` whose segments hold `per_segment` entries each, its
/// writer registered with `engine` as a group's log writer registers.
fn log_of(engine: &StorageEngine, dir: &std::path::Path, per_segment: u32) -> OplogManager {
    let log = OplogManager::open(dir, 0, 64 << 20, per_segment, 7 * 24 * 3600).expect("open log");
    engine.register_log(&[dir.to_path_buf()], log.retained_floor());
    log
}

fn append(log: &mut OplogManager, indices: std::ops::Range<u64>, ts_base: u64) {
    for index in indices {
        log.append(&unit_entry(index, ts_base + index, index + 1))
            .expect("append");
    }
    log.flush().expect("flush");
}

/// The commit timestamps of the units of `from..until` as one reader reads
/// them, `page` entries a read.
fn read_all(
    engine: &StorageEngine,
    dirs: &[PathBuf],
    from: u64,
    until: u64,
    page: usize,
) -> Vec<u64> {
    let budget = QueryBudget::new(64 << 20);
    let mut at = from;
    let mut seen = Vec::new();
    while at < until {
        let (read, _charge) = engine
            .change_read_log(dirs, at, until, page, &budget)
            .expect("read");
        assert!(read.next > at, "the read moved on from {at}");
        seen.extend(read.units().map(|unit| unit.commit_ts));
        at = read.next;
    }
    seen
}

/// Readers starting together on the same cold range, catch-up readers among
/// them, decode each entry once between them: every reader gets every unit
/// in order, and a reader reading the range again once the others passed it
/// decodes nothing.
#[test]
fn readers_of_one_range_decode_each_entry_once() {
    const ENTRIES: u64 = 200;
    for readers in [1usize, 100, 1_000] {
        let data = tempfile::tempdir().unwrap();
        let logs = tempfile::tempdir().unwrap();
        let engine = Arc::new(open_engine(data.path()));
        let mut log = log_of(&engine, logs.path(), 64);
        append(&mut log, 0..ENTRIES, 1_000);
        let dirs = vec![logs.path().to_path_buf()];
        let expected: Vec<u64> = (0..ENTRIES).map(|i| 1_000 + i).collect();

        let gate = Arc::new(Barrier::new(readers));
        let handles: Vec<_> = (0..readers)
            .map(|reader| {
                let (engine, dirs, gate) = (Arc::clone(&engine), dirs.clone(), Arc::clone(&gate));
                std::thread::spawn(move || {
                    gate.wait();
                    // Pages of several sizes, as readers of differing credit read.
                    read_all(&engine, &dirs, 0, ENTRIES, 1 + reader % 17)
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().expect("reader"), expected);
        }
        let decoded = engine.decoded_log(&dirs, None);
        assert_eq!(
            decoded.decodes(),
            ENTRIES as usize,
            "{readers} readers decoded each entry once"
        );
        assert_eq!(read_all(&engine, &dirs, 0, ENTRIES, 32), expected);
        assert_eq!(decoded.decodes(), ENTRIES as usize, "a hit decodes nothing");
    }
}

/// A log cut back and written again under the same segment name and the
/// same indices is read as it stands after the cut, not as it was decoded
/// before: the new contents, as a cold reader reads them.
#[test]
fn a_cut_log_is_read_as_written_again() {
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = open_engine(data.path());
    let mut log = log_of(&engine, logs.path(), 1_000);
    append(&mut log, 0..10, 1_000);
    let dirs = vec![logs.path().to_path_buf()];
    assert_eq!(read_all(&engine, &dirs, 0, 10, 4).len(), 10);

    log.truncate_from(0).expect("cut the whole log");
    append(&mut log, 0..10, 5_000);
    let again = read_all(&engine, &dirs, 0, 10, 4);
    assert_eq!(again, (0..10).map(|i| 5_000 + i).collect::<Vec<_>>());

    let cold_data = tempfile::tempdir().unwrap();
    let cold = open_engine(cold_data.path());
    assert_eq!(read_all(&cold, &dirs, 0, 10, 10), again, "as a cold reader");
}

/// A position the log let go of is refused as lost, though its entry is
/// still held decoded.
#[test]
fn a_held_entry_does_not_make_a_lost_position_readable() {
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = open_engine(data.path());
    let mut log = log_of(&engine, logs.path(), 5);
    append(&mut log, 0..20, 1_000);
    let dirs = vec![logs.path().to_path_buf()];
    assert_eq!(read_all(&engine, &dirs, 0, 20, 20).len(), 20);

    log.purge_before(10).expect("purge");
    let budget = QueryBudget::new(64 << 20);
    match engine.change_read_log(&dirs, 2, 20, 4, &budget) {
        Err(StorageError::RetentionLost { requested, .. }) => assert_eq!(requested, 2),
        other => panic!("a purged position must be lost, got {other:?}"),
    }
}

/// When room is needed, the segment written first stops being held first,
/// whatever order the segments were read in and however recently it was
/// read again.
#[test]
fn the_oldest_segment_is_let_go_first() {
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = open_engine(data.path());
    let mut log = log_of(&engine, logs.path(), 10);
    append(&mut log, 0..40, 1_000);
    let dirs = vec![logs.path().to_path_buf()];
    let decoded = engine.decoded_log(&dirs, None);

    // Out of write order: the second segment, the fourth, then the first.
    read_all(&engine, &dirs, 10, 20, 10);
    read_all(&engine, &dirs, 30, 40, 10);
    read_all(&engine, &dirs, 0, 10, 10);
    // The first again: a hit does not make it younger.
    read_all(&engine, &dirs, 0, 10, 10);
    assert_eq!(decoded.held_segments(), vec![0, 10, 30]);

    // Room for about three segments: the third one's fill lets go of the
    // first segment, not of a later one.
    let (_, held) = engine.change_feed_cache_bytes();
    engine.set_change_feed_cache_bytes(held);
    read_all(&engine, &dirs, 20, 30, 10);
    let segments = decoded.held_segments();
    assert!(
        !segments.contains(&0),
        "the first written goes first: {segments:?}"
    );
    assert!(
        segments.contains(&20),
        "the one read now is held: {segments:?}"
    );
    let (target, held) = engine.change_feed_cache_bytes();
    assert!(held <= target, "held {held} of a {target}-byte target");
}

/// An entry a reader still holds stays charged after the target shrinks
/// below it, and its bytes fund no other entry; once the reader lets go,
/// nothing stays charged.
#[test]
fn a_borrowed_entry_stays_charged_until_its_reader_lets_go() {
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = open_engine(data.path());
    let mut log = log_of(&engine, logs.path(), 100);
    append(&mut log, 0..20, 1_000);
    let dirs = vec![logs.path().to_path_buf()];
    let budget = QueryBudget::new(64 << 20);
    let (held_read, _charge) = engine
        .change_read_log(&dirs, 0, 20, 20, &budget)
        .expect("read");
    let (_, before) = engine.change_feed_cache_bytes();
    assert!(before > 0);

    engine.set_change_feed_cache_bytes(0);
    let decoded = engine.decoded_log(&dirs, None);
    assert!(
        decoded.held_segments().is_empty(),
        "nothing held over a zero target"
    );
    let (_, borrowed) = engine.change_feed_cache_bytes();
    assert_eq!(borrowed, before, "the reader's entries stay charged");
    // The reader still reads them, as they were.
    assert_eq!(held_read.units().count(), 20);

    drop(held_read);
    let (_, after) = engine.change_feed_cache_bytes();
    assert_eq!(after, 0, "released with the last holder");
}

/// With no room to hold anything, a reader still reads every entry, each
/// within its own budget; a frame larger than that budget is refused as the
/// budget's, the reader's position unchanged.
#[test]
fn a_log_larger_than_the_target_reads_through_the_reader() {
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = open_engine(data.path());
    let mut log = log_of(&engine, logs.path(), 100);
    append(&mut log, 0..30, 1_000);
    let dirs = vec![logs.path().to_path_buf()];
    engine.set_change_feed_cache_bytes(0);

    assert_eq!(
        read_all(&engine, &dirs, 0, 30, 4),
        (0..30).map(|i| 1_000 + i).collect::<Vec<_>>()
    );
    let decoded = engine.decoded_log(&dirs, None);
    assert!(decoded.held_segments().is_empty());
    assert_eq!(
        engine.change_feed_cache_bytes().1,
        0,
        "nothing left charged"
    );

    let tiny = QueryBudget::new(16);
    match engine.change_read_log(&dirs, 3, 30, 4, &tiny) {
        Err(StorageError::Budget(_)) => {}
        other => panic!("a frame past the reader's budget is the budget's, got {other:?}"),
    }
}
