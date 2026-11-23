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

/// A fill that began before the log was cut back and written again under
/// the same segment name and indices, and ends after it, keeps nothing of
/// what it read: its reader reads the entries written again, as a cold
/// reader does, and so does every reader after it.
#[test]
fn a_fill_late_across_a_cut_keeps_nothing_it_read() {
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = open_engine(data.path());
    let log = Arc::new(parking_lot::Mutex::new(log_of(&engine, logs.path(), 1_000)));
    append(&mut log.lock(), 0..10, 1_000);
    let dirs = vec![logs.path().to_path_buf()];
    let decoded = engine.decoded_log(&dirs, None);
    decoded.before_next_publish({
        let log = Arc::clone(&log);
        move || {
            let mut log = log.lock();
            log.truncate_from(0).expect("cut the whole log");
            append(&mut log, 0..10, 5_000);
        }
    });
    let rewritten: Vec<u64> = (0..10).map(|i| 5_000 + i).collect();
    assert_eq!(read_all(&engine, &dirs, 0, 10, 10), rewritten);
    assert_eq!(
        read_all(&engine, &dirs, 0, 10, 4),
        rewritten,
        "a later reader"
    );
    let cold_data = tempfile::tempdir().unwrap();
    let cold = open_engine(cold_data.path());
    assert_eq!(
        read_all(&cold, &dirs, 0, 10, 10),
        rewritten,
        "as a cold reader"
    );
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

/// The entry at `index` writing node `node` labelled `labels` before and
/// after, committed at `ts`.
fn labelled_entry(index: u64, ts: u64, node: u64, labels: &[u32]) -> OplogEntry {
    use coordinode_core::schema::label_id::LabelId;
    let labels: Vec<LabelId> = labels
        .iter()
        .map(|&l| LabelId::new(l).expect("a label id"))
        .collect();
    let key = encode_node_key(0, NodeId::from_raw(node));
    let mut effects = UnitEffects::new();
    effects.nodes = vec![NodeBasis {
        key: key.clone(),
        labels: labels.clone(),
    }];
    // The node's types imply nothing.
    effects.implications = labels
        .iter()
        .map(|&label| coordinode_core::txn::proposal::Implication {
            label,
            implies: Vec::new(),
        })
        .collect();
    let frame = coordinode_core::txn::frame::encode_unit(
        &crate::engine::scope::store_unit([
            Mutation::Put {
                partition: PartitionId::Node,
                key,
                value: NodeRecord::with_labels(labels)
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

/// Nodes of label 1 at even indices, of label 3 at odd ones.
fn append_labelled(log: &mut OplogManager, indices: std::ops::Range<u64>) {
    for index in indices {
        let label = if index % 2 == 0 { 1 } else { 3 };
        log.append(&labelled_entry(index, 1_000 + index, index + 1, &[label]))
            .expect("append");
    }
    log.flush().expect("flush");
}

fn labels(ids: &[u32]) -> Option<Vec<coordinode_core::schema::label_id::LabelId>> {
    Some(
        ids.iter()
            .map(|&l| coordinode_core::schema::label_id::LabelId::new(l).expect("a label id"))
            .collect(),
    )
}

/// Every unit of `from..until` as a reader of `interest` receives it, `page`
/// entries a read, following with `follower` when given.
fn deliveries(
    engine: &StorageEngine,
    dirs: &[PathBuf],
    interest: &crate::engine::change_feed::Interest,
    follower: Option<&super::ChangeFollower>,
    (from, until): (u64, u64),
    page: usize,
) -> Vec<crate::engine::change_feed::UnitChanges> {
    use crate::engine::change_feed::{ReadPlace, Unencoded, Within, unit_changes_from};
    let log = engine.decoded_log(dirs, None);
    let budget = QueryBudget::new(64 << 20);
    let mut place = ReadPlace {
        entry: from,
        within: Within::default(),
    };
    let mut out = Vec::new();
    while place.entry < until {
        let (read, mut charge) = engine
            .change_read_of(&log, place.entry, until, page, &budget, follower)
            .expect("read");
        let (units, stopped) = unit_changes_from(
            read.selected_units(),
            interest,
            place,
            &mut charge,
            &mut Unencoded,
        )
        .expect("deliver");
        out.extend(units);
        place = stopped.unwrap_or(ReadPlace {
            entry: read.next,
            within: Within::default(),
        });
    }
    out
}

/// Readers of one interest, its selectors spelled in different orders and
/// with repeats, select each unit once between them, at 1, 100 and 1 000
/// readers starting together on a cold range; a second interest selects the
/// range once more, apart. Every reader receives what a reader alone does.
#[test]
fn followers_of_one_interest_select_each_unit_once() {
    use crate::engine::change_feed::Interest;
    const ENTRIES: u64 = 120;
    for readers in [1usize, 100, 1_000] {
        let data = tempfile::tempdir().unwrap();
        let logs = tempfile::tempdir().unwrap();
        let engine = Arc::new(open_engine(data.path()));
        let mut log = log_of(&engine, logs.path(), 32);
        append_labelled(&mut log, 0..ENTRIES);
        let dirs = vec![logs.path().to_path_buf()];
        let spellings = [
            Interest::new(labels(&[2, 1, 1]), None),
            Interest::new(labels(&[1, 2]), None),
        ];
        let other = Interest::new(labels(&[3]), None);
        let alone =
            |interest: &Interest| deliveries(&engine, &dirs, interest, None, (0, ENTRIES), 7);
        let (expected, expected_other) = (alone(&spellings[0]), alone(&other));
        assert!(expected.iter().any(|u| !u.events.is_empty()));

        let decoded = engine.decoded_log(&dirs, None);
        let gate = Arc::new(Barrier::new(readers + 1));
        let handles: Vec<_> = (0..=readers)
            .map(|reader| {
                let (engine, dirs, gate) = (Arc::clone(&engine), dirs.clone(), Arc::clone(&gate));
                // The last reader reads the other interest.
                let interest = if reader == readers {
                    other.clone()
                } else {
                    spellings[reader % 2].clone()
                };
                let follower = decoded.follow(&interest);
                std::thread::spawn(move || {
                    gate.wait();
                    let got = deliveries(
                        &engine,
                        &dirs,
                        &interest,
                        Some(&follower),
                        (0, ENTRIES),
                        1 + reader % 13,
                    );
                    (reader, got, follower)
                })
            })
            .collect();
        let mut followers = Vec::new();
        for handle in handles {
            let (reader, got, follower) = handle.join().expect("reader");
            let want = if reader == readers {
                &expected_other
            } else {
                &expected
            };
            assert_eq!(
                &got, want,
                "reader {reader} of {readers} received what one alone does"
            );
            followers.push(follower);
        }
        assert_eq!(
            decoded.selects(),
            2 * ENTRIES as usize,
            "{readers} readers of one interest and one of another selected each unit once per interest"
        );
        drop(followers);
    }
}

/// What an interest selected stays while some follower of it does, and goes
/// with the last one: nothing of it is held or charged afterwards, and a
/// later follower selects again.
#[test]
fn the_last_follower_takes_its_selections_with_it() {
    use crate::engine::change_feed::Interest;
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = open_engine(data.path());
    let mut log = log_of(&engine, logs.path(), 16);
    append_labelled(&mut log, 0..40);
    let dirs = vec![logs.path().to_path_buf()];
    let interest = Interest::new(labels(&[1]), None);
    let decoded = engine.decoded_log(&dirs, None);
    // The entries alone, held for later readers.
    deliveries(&engine, &dirs, &interest, None, (0, 40), 40);
    let (_, base) = engine.change_feed_cache_bytes();

    let (first, second) = (decoded.follow(&interest), decoded.follow(&interest));
    deliveries(&engine, &dirs, &interest, Some(&first), (0, 40), 8);
    assert_eq!(decoded.held_selections(), 1);
    let (_, with) = engine.change_feed_cache_bytes();
    assert!(
        with > base,
        "the selections are charged to the store's budget"
    );
    drop(first);
    assert_eq!(decoded.held_selections(), 1, "another follower keeps them");
    deliveries(&engine, &dirs, &interest, Some(&second), (0, 40), 8);
    assert_eq!(decoded.selects(), 40, "the second follower's read hits");

    drop(second);
    assert_eq!(decoded.held_selections(), 0, "the last follower took them");
    assert_eq!(
        engine.change_feed_cache_bytes().1,
        base,
        "nothing of them charged"
    );
    let third = decoded.follow(&interest);
    deliveries(&engine, &dirs, &interest, Some(&third), (0, 40), 8);
    assert_eq!(decoded.selects(), 80, "a new follower selects again");
}

/// An explicitly empty selector takes no event, every unit still examined
/// whole and delivered complete with its count; an absent one takes every
/// event. Both share nothing with each other.
#[test]
fn an_empty_selector_completes_units_without_events() {
    use crate::engine::change_feed::Interest;
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = open_engine(data.path());
    let mut log = log_of(&engine, logs.path(), 16);
    append_labelled(&mut log, 0..20);
    let dirs = vec![logs.path().to_path_buf()];
    let decoded = engine.decoded_log(&dirs, None);
    let (none, all) = (
        Interest::new(labels(&[]), Some(Vec::new())),
        Interest::new(None, None),
    );
    let (fnone, fall) = (decoded.follow(&none), decoded.follow(&all));
    let empty = deliveries(&engine, &dirs, &none, Some(&fnone), (0, 20), 6);
    let every = deliveries(&engine, &dirs, &all, Some(&fall), (0, 20), 6);
    assert_eq!(empty.len(), 20);
    for (unit, whole) in empty.iter().zip(&every) {
        assert!(unit.events.is_empty() && unit.is_whole_unit(), "{unit:?}");
        assert_eq!(unit.projected, whole.projected);
        assert_eq!(whole.events.len(), whole.projected as usize);
    }
    assert_eq!(
        decoded.selects(),
        40,
        "each interest selected each unit once"
    );
}

/// A segment let go takes the selections of it with it: none is looked up
/// afterwards while their follower lives, and the next read selects again
/// and still delivers what a reader alone receives.
#[test]
fn a_segment_let_go_takes_its_selections_with_it() {
    use crate::engine::change_feed::Interest;
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = open_engine(data.path());
    let mut log = log_of(&engine, logs.path(), 10);
    append_labelled(&mut log, 0..30);
    let dirs = vec![logs.path().to_path_buf()];
    let decoded = engine.decoded_log(&dirs, None);
    let interest = Interest::new(labels(&[3]), None);
    let alone = deliveries(&engine, &dirs, &interest, None, (0, 30), 30);
    let follower = decoded.follow(&interest);
    assert_eq!(
        deliveries(&engine, &dirs, &interest, Some(&follower), (0, 30), 5),
        alone
    );
    assert_eq!(decoded.held_selections(), 1);

    engine.set_change_feed_cache_bytes(0);
    assert_eq!(decoded.held_selections(), 0, "let go with its segments");
    assert_eq!(
        engine.change_feed_cache_bytes().1,
        0,
        "nothing left charged"
    );
    engine.set_change_feed_cache_bytes(super::DEFAULT_DECODED_BYTES);
    assert_eq!(
        deliveries(&engine, &dirs, &interest, Some(&follower), (0, 30), 5),
        alone
    );
    assert_eq!(
        decoded.selects(),
        60,
        "selected again after the segments went"
    );
}

/// A reader that fails while it fills lets its claim go: the next reader
/// fills and reads instead of waiting for the failed fill for good, for the
/// entries and for a followed interest's selections alike.
#[test]
fn a_reader_failing_inside_a_fill_lets_the_others_read() {
    use crate::engine::change_feed::Interest;
    for selection in [false, true] {
        let data = tempfile::tempdir().unwrap();
        let logs = tempfile::tempdir().unwrap();
        let engine = Arc::new(open_engine(data.path()));
        let mut log = log_of(&engine, logs.path(), 32);
        append_labelled(&mut log, 0..20);
        let dirs = vec![logs.path().to_path_buf()];
        let interest = Interest::new(labels(&[1]), None);
        let cold = tempfile::tempdir().unwrap();
        let alone = deliveries(
            &open_engine(cold.path()),
            &dirs,
            &interest,
            None,
            (0, 20),
            20,
        );
        let decoded = engine.decoded_log(&dirs, None);
        let follower = decoded.follow(&interest);

        decoded.panic_next_fill(selection);
        let failed = {
            let (engine, dirs, interest, follower) = (
                Arc::clone(&engine),
                dirs.clone(),
                interest.clone(),
                follower.clone(),
            );
            std::thread::spawn(move || {
                deliveries(&engine, &dirs, &interest, Some(&follower), (0, 20), 20)
            })
            .join()
        };
        assert!(failed.is_err(), "the first reader failed inside its fill");

        let (tx, rx) = std::sync::mpsc::channel();
        {
            let (engine, dirs, interest, follower) = (
                Arc::clone(&engine),
                dirs.clone(),
                interest.clone(),
                follower.clone(),
            );
            std::thread::spawn(move || {
                tx.send(deliveries(
                    &engine,
                    &dirs,
                    &interest,
                    Some(&follower),
                    (0, 20),
                    20,
                ))
                .ok();
            });
        }
        let got = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the next reader is not held by the failed fill");
        assert_eq!(got, alone, "selection fill failed: {selection}");
    }
}

/// Entries appended, a segment sealed and a new one begun while readers
/// fill: what they read below the end they were given is what a cold reader
/// reads, and a reader at the tail then reads the new entries without
/// decoding the ones it passed again.
#[test]
fn appends_and_rotations_during_fills_leave_reads_whole() {
    use crate::engine::change_feed::Interest;
    const FIRST: u64 = 60;
    const MORE: u64 = 60;
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = Arc::new(open_engine(data.path()));
    let mut log = log_of(&engine, logs.path(), 25);
    append_labelled(&mut log, 0..FIRST);
    let dirs = vec![logs.path().to_path_buf()];
    let interest = Interest::new(labels(&[3]), None);
    let decoded = engine.decoded_log(&dirs, None);

    let writer = std::thread::spawn(move || {
        for index in FIRST..FIRST + MORE {
            let label = if index % 2 == 0 { 1 } else { 3 };
            log.append(&labelled_entry(index, 1_000 + index, index + 1, &[label]))
                .expect("append");
            log.flush().expect("flush");
            if index % 7 == 0 {
                log.rotate().expect("rotate");
            }
        }
        log
    });
    let readers: Vec<_> = (0..8)
        .map(|reader| {
            let (engine, dirs, interest, follower) = (
                Arc::clone(&engine),
                dirs.clone(),
                interest.clone(),
                decoded.follow(&interest),
            );
            std::thread::spawn(move || {
                deliveries(
                    &engine,
                    &dirs,
                    &interest,
                    Some(&follower),
                    (0, FIRST),
                    1 + reader,
                )
            })
        })
        .collect();
    let got: Vec<_> = readers
        .into_iter()
        .map(|r| r.join().expect("reader"))
        .collect();
    let log = writer.join().expect("writer");

    let cold = tempfile::tempdir().unwrap();
    let cold = open_engine(cold.path());
    let expected = deliveries(&cold, &dirs, &interest, None, (0, FIRST), FIRST as usize);
    for read in &got {
        assert_eq!(read, &expected);
    }
    let decoded_before = decoded.decodes();
    let follower = decoded.follow(&interest);
    let tail = deliveries(
        &engine,
        &dirs,
        &interest,
        Some(&follower),
        (FIRST, FIRST + MORE),
        9,
    );
    assert_eq!(
        tail,
        deliveries(&cold, &dirs, &interest, None, (FIRST, FIRST + MORE), 9)
    );
    assert!(
        decoded.decodes() - decoded_before <= MORE as usize,
        "the tail decoded only its own entries"
    );
    drop(log);
}

/// A fill that cannot read its frame fails its reader and every reader
/// waiting on it, none of them held for good, and none of them moved past
/// the frame.
#[test]
fn a_failed_fill_wakes_its_waiters() {
    let data = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let engine = Arc::new(open_engine(data.path()));
    let mut log = log_of(&engine, logs.path(), 100);
    append(&mut log, 0..20, 1_000);
    log.rotate().expect("seal");
    let dirs = vec![logs.path().to_path_buf()];
    let segment = std::fs::read_dir(logs.path())
        .expect("list")
        .map(|e| e.expect("entry").path())
        .find(|p| p.extension().is_some_and(|x| x == "bin"))
        .expect("a segment");
    // The frame of entry 0 damaged: its checksum fails.
    let mut bytes = std::fs::read(&segment).expect("read");
    let at = usize::try_from(crate::oplog::segment::HEADER_SIZE).expect("small") + 3;
    bytes[at] ^= 0xff;
    std::fs::write(&segment, &bytes).expect("damage");

    let gate = Arc::new(Barrier::new(8));
    let (tx, rx) = std::sync::mpsc::channel();
    for _ in 0..8 {
        let (engine, dirs, gate, tx) = (
            Arc::clone(&engine),
            dirs.clone(),
            Arc::clone(&gate),
            tx.clone(),
        );
        std::thread::spawn(move || {
            gate.wait();
            let budget = QueryBudget::new(64 << 20);
            tx.send(
                engine
                    .change_read_log(&dirs, 0, 20, 4, &budget)
                    .map(|(r, _)| r.next),
            )
            .ok();
        });
    }
    for _ in 0..8 {
        let read = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("no reader is held by a failed fill");
        assert!(read.is_err(), "a damaged frame is not read past: {read:?}");
    }
}

/// A small first page decodes the same frames however long the segment's
/// suffix after it is: the page and its bounded read-ahead.
#[test]
fn a_small_first_page_decodes_no_more_for_a_longer_suffix() {
    let decoded_for = |entries: u64| {
        let data = tempfile::tempdir().unwrap();
        let logs = tempfile::tempdir().unwrap();
        let engine = open_engine(data.path());
        let mut log = log_of(&engine, logs.path(), 100_000);
        append(&mut log, 0..entries, 1_000);
        let dirs = vec![logs.path().to_path_buf()];
        let budget = QueryBudget::new(64 << 20);
        engine
            .change_read_log(&dirs, 0, entries, 2, &budget)
            .expect("read");
        engine.decoded_log(&dirs, None).decodes()
    };
    let (short, long) = (decoded_for(200), decoded_for(2_000));
    assert_eq!(short, long, "the suffix length changes nothing");
    assert!(
        short <= 2 + super::READ_AHEAD,
        "{short} frames for a 2-entry page"
    );
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
