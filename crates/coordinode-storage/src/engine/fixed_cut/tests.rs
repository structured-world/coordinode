use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::Journal;

use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;

const COUNTER: &[u8] = b"counter:degree:7";

fn config(dir: &tempfile::TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn open(dir: &tempfile::TempDir) -> Arc<StorageEngine> {
    open_with(&config(dir))
}

fn open_with(config: &StorageConfig) -> Arc<StorageEngine> {
    let engine = Arc::new(
        StorageEngine::open_embedded(config, Arc::new(TimestampOracle::new())).expect("open"),
    );
    // A cut names the history it is of: the store's deployment.
    engine
        .put(
            Partition::Schema,
            coordinode_core::group::ENROLLED_DEPLOYMENT_KEY,
            &[7; 16],
        )
        .expect("deployment");
    engine
}

/// A budget no test read reaches.
fn unlimited() -> coordinode_core::budget::QueryBudget {
    coordinode_core::budget::QueryBudget::new(u64::MAX)
}

/// Up to `entries` entries of `cut`'s continuation from `from`.
fn read(
    engine: &StorageEngine,
    cut: &super::FixedCut,
    from: u64,
    entries: usize,
) -> super::CutRead {
    engine
        .cut_read(cut, from, entries, &unlimited())
        .expect("read")
        .0
}

/// The counter rows the tests count with.
fn counters() -> super::CutScope {
    super::CutScope::rows(Partition::Counter, b"counter:".to_vec()).expect("scope")
}

/// Cut the counters of `engine`.
fn cut(engine: &StorageEngine) -> super::FixedCut {
    engine
        .fixed_cut(counters(), Duration::from_secs(10))
        .expect("cut")
}

/// One consensus log entry batching two units, one stamped below the cut
/// and one above it: each is decided by its own commit timestamp, not by
/// its entry's highest one, so the first is in the view and the second is
/// applied.
#[test]
fn units_of_one_entry_on_both_sides_of_the_cut_are_decided_apart() {
    use coordinode_core::txn::timestamp::Timestamp;
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    let during = Arc::clone(&engine);
    let mut once = true;
    // One entry inside the join, so `[L, H)` holds a position.
    *engine.on_cut_bound.lock() = Some(Box::new(move || {
        if std::mem::take(&mut once) {
            during
                .commit_logged(&add(1), during.next_seqno(), Journal::Journal)
                .expect("inside the join");
        }
    }));
    let cut = cut(&engine);
    *engine.on_cut_bound.lock() = None;
    assert!(cut.join_end() > cut.first(), "the join holds a position");
    let below = cut.bound() - 1;
    let above = cut.bound() + 1;
    let frame = |delta, ts| {
        coordinode_core::txn::frame::encode_unit(&add(delta), Timestamp::from_raw(ts))
            .expect("frame")
    };
    let entry = crate::oplog::OplogEntry {
        ts: above,
        term: 1,
        index: cut.first(),
        shard: 0,
        ops: vec![
            crate::oplog::OplogOp::Unit {
                frame: frame(5, below),
            },
            crate::oplog::OplogOp::Unit {
                frame: frame(7, above),
            },
        ],
        is_migration: false,
        pre_images: None,
    };
    let units = engine.cut_units_of(&cut, &[entry]).expect("units");
    assert_eq!(units.len(), 2);
    assert_eq!((units[0].commit_ts, units[0].in_view), (below, true));
    assert_eq!((units[1].commit_ts, units[1].in_view), (above, false));
    assert_eq!(units[1].position.sub, 1, "its place in the entry is kept");
}

/// A commit below the cut journalled without a sync is made durable before
/// the cut is handed off: the view holds no effect a crash can take back.
#[test]
fn a_deferred_commit_below_the_cut_is_durable_before_the_handoff() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    engine
        .commit_logged(&add(1), engine.next_seqno(), Journal::Memory)
        .expect("deferred commit");
    let cut = cut(&engine);
    assert!(
        engine.pending_commits().durable_floor(cut.bound()) >= cut.bound(),
        "every commit below the cut is durable"
    );
}

/// The continuation is not held for the whole copy: journal removed past
/// `L` while the view is copied is RetentionLost, never a partial join.
#[test]
fn a_continuation_pruned_during_the_copy_is_retention_lost() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = config(&dir);
    config.oplog_segment_max_entries = 2;
    config.oplog_retention_secs = 1;
    let engine = open_with(&config);
    for _ in 0..2 {
        engine
            .commit_logged(&add(1), engine.next_seqno(), Journal::Journal)
            .expect("commit");
    }
    let cut = cut(&engine);
    for _ in 0..6 {
        engine
            .commit_logged(&add(1), engine.next_seqno(), Journal::Journal)
            .expect("commit");
    }
    let later = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_secs()
        + 3600;
    let removed = engine.oplog_purge_expired(later, u64::MAX).expect("purge");
    assert!(removed > 0, "the copy outlived the journal");
    let lost = engine
        .cut_read(&cut, cut.first(), 16, &unlimited())
        .map(|(read, _)| read);
    assert!(
        matches!(lost, Err(crate::error::StorageError::RetentionLost { .. })),
        "the join is lost, not partial: {lost:?}"
    );
}

/// Units of `read` at journal entry `index`.
fn at(read: &super::CutRead, index: u64) -> usize {
    read.units
        .iter()
        .filter(|u| u.position.index == index)
        .count()
}

/// An entry journalled but not yet applied may still be refused by its
/// operation binding: the continuation stops before it, and hands it over
/// once it is applied.
#[test]
fn an_entry_journalled_and_not_applied_is_not_handed_over() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    let cut = cut(&engine);
    let (journalled, appended) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel::<()>();
    let released = parking_lot::Mutex::new(released);
    *engine.on_journalled.lock() = Some(Box::new(move |index| {
        journalled.send(index).expect("signal");
        released.lock().recv().expect("release");
    }));
    let writer = Arc::clone(&engine);
    let commit = std::thread::spawn(move || {
        writer
            .commit_logged(&add(1), writer.next_seqno(), Journal::Journal)
            .expect("commit");
    });
    let index = appended.recv().expect("journalled");
    let first = read(&engine, &cut, cut.first(), 64);
    assert!(
        first.next <= index,
        "the read stops before an unapplied entry"
    );
    assert_eq!(at(&first, index), 0, "nothing of it is handed over");
    release.send(()).expect("release");
    commit.join().expect("writer");
    *engine.on_journalled.lock() = None;
    let later = read(&engine, &cut, first.next, 64);
    assert_eq!(at(&later, index), 1, "applied, it is handed over");
}

/// An entry applied but not yet durable can still be lost with the unsynced
/// tail: the continuation stops before it until it is synced.
#[test]
fn an_entry_not_yet_durable_is_not_handed_over() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = config(&dir);
    // The flusher must not sync the tail while the test looks at it.
    config.volatile_flush_interval_ms = 3_600_000;
    let engine = open_with(&config);
    let cut = cut(&engine);
    let index = engine
        .journal_next_index()
        .expect("journal")
        .expect("a journal");
    // Handed to the operating system: in the segment file, not synced.
    engine
        .commit_logged(&add(1), engine.next_seqno(), Journal::Cache)
        .expect("deferred commit");
    let first = read(&engine, &cut, cut.first(), 64);
    assert!(
        first.next <= index,
        "the read stops before an unsynced entry"
    );
    engine.sync_applied().expect("sync");
    let later = read(&engine, &cut, first.next, 64);
    assert_eq!(at(&later, index), 1, "durable, it is handed over");
}

/// A read keeps within its budget while the segment holds far more than it:
/// the units handed over stay charged until the read is dropped, the entry
/// that does not fit stays for the next read, and an entry larger than the
/// whole budget is refused.
#[test]
fn a_continuation_read_keeps_to_its_budget() {
    use coordinode_core::budget::QueryBudget;
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    let cut = cut(&engine);
    for _ in 0..200 {
        engine
            .commit_logged(&add(1), engine.next_seqno(), Journal::Journal)
            .expect("commit");
    }
    let whole = read(&engine, &cut, cut.first(), usize::MAX);
    assert_eq!(whole.units.len(), 200);
    let small = QueryBudget::new(1024);
    let (part, charge) = engine
        .cut_read(&cut, cut.first(), usize::MAX, &small)
        .expect("part");
    assert!(
        (1..200).contains(&part.units.len()),
        "a part of the segment: {}",
        part.units.len()
    );
    assert!(small.memory_peak() <= 1024, "within the budget");
    assert_eq!(small.memory_used(), charge.bytes(), "held by the result");
    drop(charge);
    assert_eq!(small.memory_used(), 0, "returned once the read is dropped");
    let rest = read(&engine, &cut, part.next, usize::MAX);
    assert_eq!(part.units.len() + rest.units.len(), 200, "nothing skipped");
    let tiny = QueryBudget::new(4);
    let refused = engine.cut_read(&cut, cut.first(), usize::MAX, &tiny);
    assert!(
        matches!(refused, Err(crate::error::StorageError::Budget(_))),
        "an entry past the whole budget is refused: {:?}",
        refused.map(|(read, _)| read)
    );
}

/// A cut, then entries over several journal segments, and a copy of the
/// journal's directory to read the continuation from.
fn copied_journal(
    dir: &tempfile::TempDir,
) -> (Arc<StorageEngine>, super::FixedCut, std::path::PathBuf, u64) {
    let mut config = config(dir);
    config.oplog_segment_max_entries = 2;
    let engine = open_with(&config);
    let cut = cut(&engine);
    for _ in 0..6 {
        engine
            .commit_logged(&add(1), engine.next_seqno(), Journal::Journal)
            .expect("commit");
    }
    let until = engine
        .journal_next_index()
        .expect("journal")
        .expect("a journal");
    let (journal, _) = engine
        .journal_location()
        .expect("location")
        .expect("a journal");
    let copy = dir.path().join("copy");
    std::fs::create_dir(&copy).expect("copy");
    for file in std::fs::read_dir(&journal).expect("journal") {
        let file = file.expect("file");
        std::fs::copy(file.path(), copy.join(file.file_name())).expect("copy");
    }
    (engine, cut, copy, until)
}

/// The copied journal's segments, by first index.
fn segments_of(dir: &std::path::Path) -> Vec<(u64, std::path::PathBuf)> {
    crate::oplog::tailer::list_segments(&[dir.to_path_buf()]).expect("segments")
}

/// Confirmed entries are handed over or accounted for, never answered as an
/// empty read: with every segment gone, or the one holding the next entry
/// purged while later ones remain, the read is RetentionLost; a segment
/// whose header is damaged is an error.
#[test]
fn confirmed_entries_the_log_cannot_give_are_not_an_empty_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, cut, copy, until) = copied_journal(&dir);
    let read_copy = || {
        engine
            .cut_read_log(
                &cut,
                std::slice::from_ref(&copy),
                cut.first(),
                until,
                usize::MAX,
                &unlimited(),
            )
            .map(|(read, _)| read)
    };
    let whole = read_copy().expect("the copy reads whole");
    assert_eq!(whole.next, until);
    assert!(segments_of(&copy).len() > 2, "several segments");

    // The segment holding the next entry purged, later ones kept.
    let held = segments_of(&copy);
    let (_, first) = held
        .iter()
        .rev()
        .find(|(start, _)| *start <= cut.first())
        .expect("a segment holds the first entry");
    let saved = std::fs::read(first).expect("segment");
    std::fs::remove_file(first).expect("purge");
    let lost = read_copy();
    assert!(
        matches!(lost, Err(crate::error::StorageError::RetentionLost { .. })),
        "{lost:?}"
    );

    // Its header damaged: an error, not a wait for entries.
    let mut damaged = saved;
    damaged[0] ^= 0xFF;
    std::fs::write(first, &damaged).expect("damage");
    let unreadable = read_copy();
    assert!(
        matches!(&unreadable, Err(e) if !matches!(e, crate::error::StorageError::RetentionLost { .. })),
        "{unreadable:?}"
    );

    // Every segment gone.
    for (_, path) in segments_of(&copy) {
        std::fs::remove_file(path).expect("purge");
    }
    let gone = read_copy();
    assert!(
        matches!(gone, Err(crate::error::StorageError::RetentionLost { .. })),
        "{gone:?}"
    );
}

/// A small frame that expands to many changes is refused by what it decodes
/// to, not admitted by its encoded size: the budget bounds the decoding
/// before it runs.
#[test]
fn a_frame_expanding_past_the_budget_is_refused_before_it_is_decoded() {
    use coordinode_core::budget::QueryBudget;
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    let cut = cut(&engine);
    // Keys sharing a long prefix encode to a few bytes each.
    let many: Vec<Mutation> = (0..2000u32)
        .map(|i| Mutation::Delete {
            partition: PartitionId::Counter,
            key: format!("counter:{:0>64}", i).into_bytes(),
        })
        .collect();
    let frame = coordinode_core::txn::frame::encode_unit(
        &many,
        coordinode_core::txn::timestamp::Timestamp::from_raw(1),
    )
    .expect("frame")
    .len() as u64;
    engine
        .commit_logged(&many, engine.next_seqno(), Journal::Journal)
        .expect("commit");
    // Room for the frame several times over, not for what it decodes to.
    let budget = QueryBudget::new(frame * 4);
    let refused = engine.cut_read(&cut, cut.first(), usize::MAX, &budget);
    assert!(
        matches!(refused, Err(crate::error::StorageError::Budget(_))),
        "refused by what it decodes to: {:?}",
        refused.map(|(read, _)| read.units.len())
    );
    assert!(budget.memory_peak() <= frame * 4, "within the budget");
}

/// A page refuses a row larger than its whole budget, and keeps within it
/// otherwise.
#[test]
fn a_page_keeps_to_its_budget() {
    use coordinode_core::budget::QueryBudget;
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    engine
        .commit_logged(
            &[Mutation::Put {
                partition: PartitionId::Counter,
                key: b"counter:big".to_vec(),
                value: vec![0; 4096],
            }],
            engine.next_seqno(),
            Journal::Journal,
        )
        .expect("row");
    let cut = cut(&engine);
    let small = QueryBudget::new(1024);
    let refused = engine.cut_page(&cut, None, 16, &small);
    assert!(
        matches!(refused, Err(crate::error::StorageError::Budget(_))),
        "a row past the whole budget is refused"
    );
    assert_eq!(small.memory_peak(), 0, "refused before it is copied");
    let (page, _) = engine
        .cut_page(&cut, None, 16, &QueryBudget::new(1 << 20))
        .expect("page");
    assert_eq!(page.len(), 1);
}

/// Units carry only their changes inside the scope; an effect outside it
/// is dropped, and a raw row write inside it, which carries no commit
/// timestamp of its own, is refused rather than skipped.
#[test]
fn a_cut_hands_over_its_scope_and_refuses_what_it_cannot_carry() {
    use coordinode_core::txn::timestamp::Timestamp;
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    let cut = cut(&engine);
    let mut both = add(3);
    both.push(Mutation::Put {
        partition: PartitionId::Node,
        key: b"node:1".to_vec(),
        value: b"n".to_vec(),
    });
    let entry = |ops| crate::oplog::OplogEntry {
        ts: cut.bound() + 1,
        term: 0,
        index: cut.join_end(),
        shard: 0,
        ops,
        is_migration: false,
        pre_images: None,
    };
    let unit = crate::oplog::OplogOp::Unit {
        frame: coordinode_core::txn::frame::encode_unit(
            &both,
            Timestamp::from_raw(cut.bound() + 1),
        )
        .expect("frame"),
    };
    let outside = crate::oplog::OplogOp::Insert {
        partition: crate::placement::partition_wire_tag(Partition::Node),
        key: b"node:2".to_vec(),
        value: b"n".to_vec(),
    };
    let columnar = crate::oplog::OplogOp::ColumnarInsert {
        table_id: "t".into(),
        key: b"k".to_vec(),
        value: b"v".to_vec(),
    };
    let units = engine
        .cut_units_of(&cut, &[entry(vec![unit, outside, columnar])])
        .expect("units");
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].mutations, add(3), "only the scope's changes");
    let inside = crate::oplog::OplogOp::Insert {
        partition: crate::placement::partition_wire_tag(Partition::Counter),
        key: COUNTER.to_vec(),
        value: b"1".to_vec(),
    };
    let refused = engine.cut_units_of(&cut, &[entry(vec![inside])]);
    assert!(
        matches!(
            refused,
            Err(crate::error::StorageError::CutUnsupported { .. })
        ),
        "{refused:?}"
    );
    for partition in [Partition::Idx, Partition::Schema, Partition::Raft] {
        assert!(
            super::CutScope::rows(partition, Vec::new()).is_err(),
            "{partition:?} is not a row scope"
        );
    }
}

/// A page starts where the last one ended: the rows it passes over do not
/// grow with how far into the scope it is.
#[test]
fn a_far_page_reads_only_from_where_the_last_ended() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    let rows: Vec<Mutation> = (0..64u32)
        .map(|i| Mutation::Put {
            partition: PartitionId::Counter,
            key: format!("counter:{i:04}").into_bytes(),
            value: crate::engine::merge::encode_counter_delta(1),
        })
        .collect();
    engine
        .commit_logged(&rows, engine.next_seqno(), Journal::Journal)
        .expect("rows");
    let cut = cut(&engine);
    let budget = unlimited();
    let (page, _) = engine
        .cut_page(&cut, Some(b"counter:0059"), 2, &budget)
        .expect("page");
    let keys: Vec<&[u8]> = page.iter().map(|(k, _)| k.as_slice()).collect();
    assert_eq!(
        keys,
        [b"counter:0060".as_slice(), b"counter:0061".as_slice()]
    );
    let mut seen = 0;
    let mut after: Option<Vec<u8>> = None;
    loop {
        let (page, _) = engine
            .cut_page(&cut, after.as_deref(), 5, &budget)
            .expect("page");
        let Some((last, _)) = page.last() else { break };
        seen += page.len();
        after = Some(last.clone());
    }
    assert_eq!(seen, 64, "every row once");
}

/// A range delete reaching past the scope on both sides is handed over
/// only for the part inside it: applied by the consumer, it removes the
/// scope's rows in range and nothing outside the prefix.
#[test]
fn a_range_delete_is_clipped_to_the_scope() {
    use coordinode_core::txn::timestamp::Timestamp;
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    let scope = super::CutScope::rows(Partition::Counter, b"counter:m".to_vec()).expect("scope");
    let cut = engine
        .fixed_cut(scope, Duration::from_secs(10))
        .expect("cut");
    let delete = vec![Mutation::RemoveRange {
        partition: PartitionId::Counter,
        start: b"counter:a".to_vec(),
        end: b"counter:z".to_vec(),
    }];
    let entry = crate::oplog::OplogEntry {
        ts: cut.bound() + 1,
        term: 0,
        index: cut.join_end(),
        shard: 0,
        ops: vec![crate::oplog::OplogOp::Unit {
            frame: coordinode_core::txn::frame::encode_unit(
                &delete,
                Timestamp::from_raw(cut.bound() + 1),
            )
            .expect("frame"),
        }],
        is_migration: false,
        pre_images: None,
    };
    let units = engine.cut_units_of(&cut, &[entry]).expect("units");
    let [unit] = units.as_slice() else {
        panic!("one unit: {units:?}")
    };
    let [Mutation::RemoveRange { start, end, .. }] = unit.mutations.as_slice() else {
        panic!("one range delete: {:?}", unit.mutations)
    };
    assert_eq!(
        (start.as_slice(), end.as_slice()),
        (b"counter:m".as_slice(), b"counter:n".as_slice())
    );
    // Applied to a destination holding rows on both sides of the prefix.
    let rows = [
        b"counter:b".as_slice(),
        b"counter:m1",
        b"counter:mz",
        b"counter:y",
    ];
    let left: Vec<&[u8]> = rows
        .iter()
        .copied()
        .filter(|k| !(*k >= start.as_slice() && *k < end.as_slice()))
        .collect();
    assert_eq!(
        left,
        [b"counter:b".as_slice(), b"counter:y"],
        "nothing outside is deleted"
    );
}

/// Clipping keeps an open top for a prefix with no successor: the empty
/// prefix and an all-`0xFF` one reach the end of the key space.
#[test]
fn a_scope_without_a_successor_clips_to_the_end_of_the_key_space() {
    let all = super::CutScope::rows(Partition::Counter, Vec::new()).expect("scope");
    assert_eq!(
        all.clip_range(b"a", b"z"),
        Some((b"a".to_vec(), b"z".to_vec()))
    );
    let top = super::CutScope::rows(Partition::Counter, vec![0xFF]).expect("scope");
    assert_eq!(
        top.clip_range(&[0x00], &[0xFF, 0x01]),
        Some((vec![0xFF], vec![0xFF, 0x01]))
    );
    assert_eq!(
        top.clip_range(&[0x00], &[0xFE]),
        None,
        "nothing of it is inside"
    );
}

fn add(delta: i64) -> Vec<Mutation> {
    vec![Mutation::Merge {
        partition: PartitionId::Counter,
        key: COUNTER.to_vec(),
        operand: crate::engine::merge::encode_counter_delta(delta),
    }]
}

fn counter(value: Option<bytes::Bytes>) -> i64 {
    value.map_or(0, |v| {
        crate::engine::merge::decode_counter(&v).expect("counter")
    })
}

/// A commit that drew its timestamp below the cut but reached the journal
/// after `L` is in the view, and the join says so: a consumer that copies
/// the view and applies only the units the view does not hold ends with the
/// source's state, a non-idempotent merge counted once. The journal order is
/// not the timestamp order: a later-stamped commit sits before it.
#[test]
fn a_late_commit_below_the_cut_is_in_the_view_and_counted_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    let commit = |m: &[Mutation], ts: u64| {
        engine
            .commit_logged(m, ts, Journal::Journal)
            .expect("commit");
    };
    commit(&add(1), engine.next_seqno());
    // Drawn below the cut, journalled only after the cut read `L`.
    let (late, held) = engine.pending_commits().obligate(|| engine.next_seqno());
    let during = Arc::clone(&engine);
    let mut held = Some(held);
    *engine.on_cut_bound.lock() = Some(Box::new(move || {
        let Some(held) = held.take() else { return };
        // Stamped above the cut, journalled first.
        during
            .commit_logged(&add(10), during.next_seqno(), Journal::Journal)
            .expect("above");
        during
            .commit_logged(&add(100), late, Journal::Journal)
            .expect("late");
        drop(held);
    }));
    let cut = cut(&engine);
    *engine.on_cut_bound.lock() = None;
    assert!(late < cut.bound(), "the late commit is below the cut");

    let mut destination = counter(
        engine
            .snapshot_get(&cut.bound(), Partition::Counter, COUNTER)
            .expect("view"),
    );
    assert_eq!(destination, 101, "the view holds the late commit");
    let span = usize::try_from(cut.join_end() - cut.first()).expect("span");
    let joined = read(&engine, &cut, cut.first(), span);
    assert_eq!(joined.next, cut.join_end(), "the read ends at the join");
    let units = joined.units;
    let late_unit = units
        .iter()
        .find(|u| u.commit_ts == late)
        .expect("the late unit is in the join");
    assert!(late_unit.in_view, "the view already holds it");
    for unit in units.iter().filter(|u| !u.in_view) {
        for m in &unit.mutations {
            if let Mutation::Merge { operand, .. } = m {
                destination += crate::engine::merge::decode_counter(operand).expect("delta");
            }
        }
    }
    let source = counter(engine.get(Partition::Counter, COUNTER).expect("source"));
    assert_eq!(source, 111);
    assert_eq!(destination, source, "every unit counted once");
}
