//! The logical material each commit adds, checked against an independent
//! oracle: the test keeps its own list of every unit it wrote (seqno and
//! logical bytes) and compares the engine's answer from every checkpoint.

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use tempfile::TempDir;

fn config(dir: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn oracle_engine(dir: &std::path::Path, base: u64) -> StorageEngine {
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(base)));
    StorageEngine::open_with_oracle(&config(dir), oracle).expect("open")
}

/// Far above the wall clock, so the test picks every commit timestamp.
fn future_base() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros();
    u64::try_from(now).expect("fits") + 1_000_000_000_000
}

/// What the test wrote: logical bytes per commit timestamp.
#[derive(Default)]
struct Oracle(std::collections::BTreeMap<u64, u64>);

impl Oracle {
    fn wrote(&mut self, ts: u64, key: &[u8], written: &[u8]) {
        *self.0.entry(ts).or_insert(0) += (key.len() + written.len()) as u64;
    }

    fn from(&self, s: u64) -> crate::engine::material::Material {
        let mut first = None;
        let mut bytes = 0;
        for (&ts, &b) in self.0.range(s..) {
            first.get_or_insert(ts);
            bytes += b;
        }
        crate::engine::material::Material { first, bytes }
    }
}

fn apply(engine: &StorageEngine, oracle: &mut Oracle, ts: u64, mutations: Vec<Mutation>) {
    for m in &mutations {
        match m {
            Mutation::Put { key, value, .. } => oracle.wrote(ts, key, value),
            Mutation::Delete { key, .. } => oracle.wrote(ts, key, &[]),
            Mutation::Merge { key, operand, .. } => oracle.wrote(ts, key, operand),
            Mutation::RemoveRange { start, end, .. } => oracle.wrote(ts, start, end),
            other => panic!("not a data mutation: {other:?}"),
        }
    }
    engine.apply_proposal_at(&mutations, ts).expect("apply");
}

/// The engine's answer equals the oracle's from every checkpoint, below,
/// between and past the commits.
fn check(engine: &StorageEngine, oracle: &Oracle, base: u64, span: u64) {
    for s in base..base + span {
        assert_eq!(
            engine.material_between(s, None).expect("material"),
            oracle.from(s),
            "from {s}"
        );
    }
}

/// Every kind of unit, commits applied out of timestamp order, several
/// units and partitions in one commit, a merge chain charged by its
/// operands (not by the short value it resolves to), a range deletion, the
/// consensus log and the registry left out; the same answer from the
/// memtable, after a flush, and after a reopen.
#[test]
fn material_from_every_checkpoint_matches_what_was_written() {
    let dir = TempDir::new().expect("tempdir");
    // Opening writes the engine's own records at the timestamps right after
    // the oracle's start; the test's commits come well after them, and every
    // checkpoint checked is past them.
    let start = future_base();
    let base = start + 10_000;
    let mut oracle = Oracle::default();
    {
        let engine = oracle_engine(dir.path(), start);
        // A later commit applied before an earlier one.
        apply(
            &engine,
            &mut oracle,
            base + 40,
            vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:a".to_vec(),
                value: vec![7; 10],
            }],
        );
        apply(
            &engine,
            &mut oracle,
            base + 20,
            vec![
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:b".to_vec(),
                    value: vec![1; 20],
                },
                Mutation::Put {
                    partition: PartitionId::EdgeProp,
                    key: b"edgeprop:x".to_vec(),
                    value: vec![2; 5],
                },
            ],
        );
        // Ten counter operands resolve to one 8-byte value; the material is
        // the ten operands.
        for i in 0..10 {
            apply(
                &engine,
                &mut oracle,
                base + 50 + i,
                vec![Mutation::Merge {
                    partition: PartitionId::Counter,
                    key: b"counter:c".to_vec(),
                    operand: crate::engine::merge::encode_counter_delta(1),
                }],
            );
        }
        apply(
            &engine,
            &mut oracle,
            base + 70,
            vec![Mutation::Delete {
                partition: PartitionId::Node,
                key: b"node:a".to_vec(),
            }],
        );
        apply(
            &engine,
            &mut oracle,
            base + 80,
            vec![Mutation::RemoveRange {
                partition: PartitionId::EdgeProp,
                start: b"edgeprop:".to_vec(),
                end: b"edgeprop;".to_vec(),
            }],
        );
        // The registry's own records are no source material.
        engine
            .apply_proposal_at(
                &[Mutation::Put {
                    partition: PartitionId::Registry,
                    key: b"registry:consumer".to_vec(),
                    value: vec![3; 30],
                }],
                base + 90,
            )
            .expect("apply registry record");

        check(&engine, &oracle, base, 100);
        for &part in Partition::all() {
            engine
                .tree(part)
                .expect("tree")
                .flush_active_memtable(0)
                .expect("flush");
        }
        check(&engine, &oracle, base, 100);
    }
    let engine = oracle_engine(dir.path(), base + 1_000);
    check(&engine, &oracle, base, 100);
    // Opening again wrote nothing below the oracle's new start.
    assert!(
        engine
            .material_between(base + 100, Some(base + 1_000))
            .expect("material")
            .first
            .is_none()
    );
}

/// Records below the readable floor are dropped once the floor has moved a
/// window past the last retirement, and only those: a checkpoint at or above
/// the floor still reads every record it needs.
#[test]
fn records_below_the_readable_floor_are_retired() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let engine = oracle_engine(dir.path(), start);
    // History is kept one retirement window back from the newest commit, so
    // the floor falls between the two commits below.
    engine.set_retention_window(std::time::Duration::from_micros(
        crate::engine::material::RETIRE_WINDOW,
    ));
    let base = start + 10_000;
    let mut oracle = Oracle::default();
    let put = |key: &[u8]| Mutation::Put {
        partition: PartitionId::Node,
        key: key.to_vec(),
        value: vec![1; 4],
    };
    apply(&engine, &mut oracle, base, vec![put(b"node:old")]);
    // A commit far enough ahead moves the floor (no retention window, no
    // consumer) past the first one and triggers a retirement.
    let later = base + 2 * crate::engine::material::RETIRE_WINDOW;
    apply(&engine, &mut oracle, later, vec![put(b"node:new")]);
    let floor = engine.oldest_readable_seqno();
    assert!(
        floor > base && floor <= later,
        "the floor lies between the commits: {base} < {floor} <= {later}"
    );
    assert_eq!(
        engine
            .material_between(0, Some(floor))
            .expect("material")
            .first,
        None,
        "nothing is kept below the floor"
    );
    assert_eq!(
        engine.material_between(later, None).expect("material"),
        oracle.from(later)
    );
}

/// An observation counts only what lies at or below the store's complete
/// cut: a commit applied while an earlier timestamp is still in flight is
/// not counted yet (the raw records already show it), and is once the
/// earlier one has gone. A commit applied in part above the cut can thus
/// never be aged or sized.
#[test]
fn an_observation_stops_at_the_complete_cut() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let engine = oracle_engine(dir.path(), start);
    let base = start + 10_000;
    let mut oracle = Oracle::default();
    // An earlier commit holds its timestamp, not yet applied.
    let (held_ts, held) = engine
        .pending_commits()
        .admit_allocated(
            || base,
            Vec::new(),
            Vec::new(),
            crate::engine::pending::Material::Changes,
            None,
        )
        .expect("admit");
    apply(
        &engine,
        &mut oracle,
        base + 50,
        vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:after".to_vec(),
            value: vec![1; 9],
        }],
    );
    assert_eq!(
        engine.material_between(base, None).expect("raw").bytes,
        19,
        "the raw records already show the later commit"
    );
    // A checkpoint at the cut itself, so the read does reach the records.
    let checkpoint = held_ts - 1;
    engine.fold_material_to_cut().expect("fold");
    match engine.observe_material(checkpoint).expect("observe") {
        Observation::Observed {
            through, material, ..
        } => {
            assert_eq!(through, checkpoint, "the cut stops below the held commit");
            assert_eq!(material, crate::engine::material::Material::default());
        }
        other => panic!("observed nothing: {other:?}"),
    }
    drop(held);
    engine.fold_material_to_cut().expect("fold");
    match engine.observe_material(checkpoint).expect("observe") {
        Observation::Observed { material, .. } => assert_eq!(material, oracle.from(checkpoint)),
        other => panic!("observed nothing: {other:?}"),
    }
}

/// While a tree's rebuild from a checkpoint is owed its records are not all
/// there: an observation says so instead of counting what it finds.
#[test]
fn an_observation_during_a_rebuild_is_no_measure() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let engine = oracle_engine(dir.path(), future_base());
    let _repair = engine.begin_rebuild(Partition::Node).expect("intent");
    assert_eq!(
        engine.observe_material(0).expect("observe"),
        Observation::Rebuilding
    );
}

/// A rebuild that begins after the observation checked for one, and before
/// it read, leaves no observation: what it would have read can mix the
/// tree's states.
#[test]
fn a_rebuild_begun_during_an_observation_voids_it() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let engine = oracle_engine(dir.path(), start);
    engine.fold_material_to_cut().expect("fold");
    let observed = engine
        .observe_material_with(start, || {
            // Begun and left running: the repair holds on past the read.
            core::mem::forget(engine.begin_rebuild(Partition::Node).expect("intent"));
        })
        .expect("observe");
    assert_eq!(observed, Observation::Rebuilding);
}

/// The public split repair (a partition installed from a copy, the log
/// replayed into it, then finished) holds the partition from its begin to
/// its finish: after the begin has returned, a clear or a second rebuild of
/// the partition is refused, observations stay unavailable, and the
/// installed rows are still there when it finishes. Before, the repair
/// released the partition when the begin returned, and a clear could wipe
/// the installed base before the finish recorded its coverage.
#[test]
fn a_split_repair_holds_its_partition_until_it_finishes() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let engine = oracle_engine(dir.path(), start);
    let rows: Rows = vec![(b"node:installed".to_vec(), b"base".to_vec())];
    let repair = engine
        .begin_partition_rebuild(Partition::Node, &rows)
        .expect("begin");
    assert!(
        engine.clear_partition(Partition::Node).is_err(),
        "a clear ran during the split repair"
    );
    assert!(
        engine
            .begin_partition_rebuild(Partition::Node, &Vec::new())
            .is_err(),
        "a second rebuild ran during the split repair"
    );
    assert_eq!(
        engine.observe_material(start).expect("observe"),
        Observation::Rebuilding
    );
    engine.finish_raft_rebuild(repair, 1, &[]).expect("finish");
    assert_eq!(
        engine
            .get(Partition::Node, b"node:installed")
            .expect("read")
            .as_deref(),
        Some(&b"base"[..]),
        "the installed rows were wiped before the finish"
    );
}

/// A transfer into the index partition refused because another repair of it
/// runs (one that is no transfer) leaves the transfer status as it was:
/// marking a transfer pending would take every index copy as unqualified.
#[test]
fn a_refused_index_transfer_leaves_the_transfer_status_alone() {
    let dir = TempDir::new().expect("tempdir");
    let engine = oracle_engine(dir.path(), future_base());
    let running = engine.begin_rebuild(Partition::Idx).expect("a repair");
    assert!(
        engine
            .begin_partition_rebuild(Partition::Idx, &Vec::new())
            .is_err()
    );
    assert!(
        !engine.index_transfer_pending().expect("read"),
        "the refused transfer marked one pending"
    );
    drop(running);
}

/// The intent a repair's first step writes is durable: a store reopened
/// with it still owes the repair.
#[test]
fn a_repair_intent_voids_observations_after_reopen() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    {
        let engine = oracle_engine(dir.path(), start);
        // The clear's first step, then the process is gone.
        core::mem::forget(engine.begin_rebuild(Partition::Node).expect("intent"));
    }
    let engine = oracle_engine(dir.path(), start + 1_000);
    assert_eq!(
        engine.observe_material(start + 1_000).expect("observe"),
        Observation::Rebuilding
    );
}

/// A repair is finished only by the store it was taken on: handed to another
/// store, whose own repair of the partition runs, it is refused before that
/// store changes anything, and the other store's repair goes on.
#[test]
fn a_repair_finishes_only_on_its_own_store() {
    use crate::engine::material::Observation;
    let (dir_a, dir_b) = (TempDir::new().expect("a"), TempDir::new().expect("b"));
    let start = future_base();
    let a = oracle_engine(dir_a.path(), start);
    let b = oracle_engine(dir_b.path(), start);
    let rows: Rows = vec![(b"node:b".to_vec(), b"base".to_vec())];
    let of_a = a
        .begin_partition_rebuild(Partition::Node, &Vec::new())
        .expect("a's repair");
    let of_b = b
        .begin_partition_rebuild(Partition::Node, &rows)
        .expect("b's repair");
    assert!(
        b.finish_raft_rebuild(of_a, 1, &[]).is_err(),
        "a repair taken on another store finished here"
    );
    assert_eq!(
        b.observe_material(start).expect("observe"),
        Observation::Rebuilding,
        "b's repair was ended by a's"
    );
    assert!(
        b.clear_partition(Partition::Node).is_err(),
        "b's repair no longer holds its partition"
    );
    b.finish_raft_rebuild(of_b, 1, &[]).expect("b finishes");
    assert_eq!(
        b.get(Partition::Node, b"node:b").expect("read").as_deref(),
        Some(&b"base"[..])
    );
}

/// What qualifies a transferred index runs while the transfer's repair
/// still holds the partition: a repair begun from inside it is refused, so
/// no next transfer's marker or installation is taken for this one's.
#[test]
fn an_index_transfer_is_qualified_under_its_repair() {
    struct Probe(std::sync::Mutex<Option<bool>>);
    impl crate::engine::index_copies::IndexCopyQualifier for Probe {
        fn capture(&self, _: &StorageEngine) -> StorageResult<Vec<u8>> {
            Ok(Vec::new())
        }
        fn seal(&self, _: &StorageEngine, _: &[u8]) -> StorageResult<Vec<u8>> {
            Ok(Vec::new())
        }
        fn installed(&self, engine: &StorageEngine, _: Option<&[u8]>) -> StorageResult<()> {
            let second = engine.begin_partition_rebuild(Partition::Idx, &Vec::new());
            *self.0.lock().expect("probe") = Some(second.is_err());
            Ok(())
        }
        fn copy_proof(&self, _: &StorageEngine, _: u64) -> StorageResult<Option<(Vec<u8>, u64)>> {
            Ok(None)
        }
    }
    let dir = TempDir::new().expect("tempdir");
    let engine = oracle_engine(dir.path(), future_base());
    let probe = Arc::new(Probe(std::sync::Mutex::new(None)));
    engine.set_index_copy_qualifier(Arc::clone(&probe) as _);
    let repair = engine
        .begin_partition_rebuild(Partition::Idx, &Vec::new())
        .expect("begin");
    engine.finish_raft_rebuild(repair, 1, &[]).expect("finish");
    assert_eq!(
        *probe.0.lock().expect("probe"),
        Some(true),
        "a repair began while the transfer was being qualified"
    );
}

/// A transferred index whose qualification fails leaves the partition owed
/// a repair, also after a reopen, and lets the next repair take it.
#[test]
fn a_failed_index_qualification_leaves_the_partition_owed() {
    use crate::engine::material::Observation;
    struct Failing;
    impl crate::engine::index_copies::IndexCopyQualifier for Failing {
        fn capture(&self, _: &StorageEngine) -> StorageResult<Vec<u8>> {
            Ok(Vec::new())
        }
        fn seal(&self, _: &StorageEngine, _: &[u8]) -> StorageResult<Vec<u8>> {
            Ok(Vec::new())
        }
        fn installed(&self, _: &StorageEngine, _: Option<&[u8]>) -> StorageResult<()> {
            Err(StorageError::Io("qualification failed".into()))
        }
        fn copy_proof(&self, _: &StorageEngine, _: u64) -> StorageResult<Option<(Vec<u8>, u64)>> {
            Ok(None)
        }
    }
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    {
        let engine = oracle_engine(dir.path(), start);
        engine.set_index_copy_qualifier(Arc::new(Failing));
        let repair = engine
            .begin_partition_rebuild(Partition::Idx, &Vec::new())
            .expect("begin");
        assert!(engine.finish_raft_rebuild(repair, 1, &[]).is_err());
        assert_eq!(
            engine.observe_material(start).expect("observe"),
            Observation::Rebuilding,
            "a failed qualification passed as finished"
        );
        let next = engine
            .begin_rebuild(Partition::Idx)
            .expect("the failed repair still holds the partition");
        drop(next);
    }
    let engine = oracle_engine(dir.path(), start + 1_000);
    assert_eq!(
        engine.observe_material(start + 1_000).expect("observe"),
        Observation::Rebuilding,
        "a failed qualification passed as finished after a reopen"
    );
    assert!(
        engine.index_transfer_pending().expect("read"),
        "the transfer was taken as qualified"
    );
}

/// A clear of a populated partition that fails after its destructive step
/// leaves the partition owed a repair, also after a reopen.
#[test]
fn a_clear_failing_after_it_cleared_leaves_the_partition_owed() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    {
        let engine = oracle_engine(dir.path(), start);
        engine
            .put(Partition::Node, b"node:gone", b"value")
            .expect("populate");
        let tree = engine.tree(Partition::Node).expect("tree");
        let failed = engine.clear_partition_with(Partition::Node, tree, || {
            Err(StorageError::Io("cut short after the clear".into()))
        });
        assert!(failed.is_err());
        assert_eq!(
            engine.observe_material(start).expect("observe"),
            Observation::Rebuilding
        );
    }
    let engine = oracle_engine(dir.path(), start + 1_000);
    assert_eq!(
        engine.observe_material(start + 1_000).expect("observe"),
        Observation::Rebuilding,
        "a clear cut short passed as finished after a reopen"
    );
}

/// A repair is visible from its first step, before its intent is written:
/// an observation taken then, with the generation already moved and no
/// intent on disk, is no measure.
#[test]
fn a_repair_before_its_intent_lands_voids_an_observation() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let engine = oracle_engine(dir.path(), start);
    let _repair = engine.enter_repair(Partition::Node).expect("repair");
    assert_eq!(
        engine.observe_material(start).expect("observe"),
        Observation::Rebuilding
    );
}

/// Two repairs of one partition never run at once: the second is refused
/// while the first runs, so it cannot end the first's unavailable interval
/// early; once the first is gone (finished or failed), the next one starts.
/// A clear that fails leaves the partition unavailable.
#[test]
fn a_partition_takes_one_repair_at_a_time() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let engine = oracle_engine(dir.path(), start);
    let first = engine.begin_rebuild(Partition::Node).expect("first");
    assert!(
        engine.clear_partition(Partition::Node).is_err(),
        "a clear joined a running rebuild"
    );
    assert!(
        engine.begin_rebuild(Partition::Node).is_err(),
        "a second rebuild joined a running one"
    );
    assert_eq!(
        engine.observe_material(start).expect("observe"),
        Observation::Rebuilding,
        "the running rebuild is still unavailable"
    );
    // The first fails (its guard goes without finishing): still owed.
    drop(first);
    assert_eq!(
        engine.observe_material(start).expect("observe"),
        Observation::Rebuilding,
        "a failed repair passed as finished"
    );
    // The next repair takes it and finishes.
    let next = engine.begin_rebuild(Partition::Node).expect("next");
    engine.finish_rebuild(next).expect("finish");
    engine.fold_material_to_cut().expect("fold");
    assert!(matches!(
        engine.observe_material(start).expect("observe"),
        Observation::Observed { .. }
    ));
}

/// A rebuild left owed by an earlier run (its intent on disk, the process
/// gone before it finished) still voids observations after a reopen.
#[test]
fn an_unfinished_rebuild_voids_observations_after_reopen() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    {
        let engine = oracle_engine(dir.path(), start);
        let _repair = engine.begin_rebuild(Partition::Node).expect("intent");
    }
    let engine = oracle_engine(dir.path(), start + 1_000);
    assert_eq!(
        engine.observe_material(start + 1_000).expect("observe"),
        Observation::Rebuilding
    );
}

/// The readable floor rising during an observation (a consumer moved on,
/// records below the floor retired) takes none of the records it reads:
/// they are held from the checkpoint until the read is done.
#[test]
fn records_from_the_checkpoint_stay_through_an_observation() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let engine = oracle_engine(dir.path(), start);
    engine.set_retention_window(std::time::Duration::from_secs(3_600));
    let base = start + 10_000;
    let mut oracle = Oracle::default();
    apply(
        &engine,
        &mut oracle,
        base,
        vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:kept".to_vec(),
            value: vec![1; 4],
        }],
    );
    let later = base + 2 * crate::engine::material::RETIRE_WINDOW;
    engine.fold_material_to_cut().expect("fold");
    let observed = engine
        .observe_material_with(base, || {
            // Meanwhile nothing keeps the window back: a commit far ahead
            // moves the floor past the checkpoint and retires below it.
            engine.set_retention_window(std::time::Duration::from_micros(1));
            apply(
                &engine,
                &mut Oracle::default(),
                later,
                vec![Mutation::Put {
                    partition: PartitionId::EdgeProp,
                    key: b"edgeprop:later".to_vec(),
                    value: vec![2; 4],
                }],
            );
            // And collection runs over the records before the read reaches
            // them.
            let tree = engine.tree(Partition::Accounting).expect("tree");
            tree.flush_active_memtable(0).expect("flush");
            engine
                .force_compaction(Partition::Accounting)
                .expect("compact");
        })
        .expect("observe");
    match observed {
        Observation::Observed { material, .. } => {
            assert_eq!(
                material.first,
                Some(base),
                "the checkpoint's work was retired"
            );
        }
        other => panic!("observed nothing: {other:?}"),
    }
}

/// A direct write outside any log holds the complete cut below its seqno
/// until all of it has landed, its accounting in its own tree included: its
/// data is in the data tree and its accounting not yet in the accounting
/// tree, and no observation taken in between counts the write, in part or
/// as empty.
#[test]
fn a_direct_write_under_way_holds_the_cut_below_it() {
    use crate::engine::material::Observation;
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let engine = oracle_engine(dir.path(), start);
    let mut batch = lsm_tree::WriteBatch::with_capacity(2);
    batch.insert(b"node:direct".as_slice(), [1u8; 4].as_slice());
    let mut tally = crate::engine::material::Tally::default();
    tally.put(b"node:direct", &[1; 4]);
    let mut seen = None;
    engine
        .apply_single_with(Partition::Node, batch, tally, |seqno| {
            let data = engine
                .tree(Partition::Node)
                .expect("tree")
                .get(b"node:direct", lsm_tree::SeqNo::MAX)
                .expect("read")
                .is_some();
            let cut = engine.complete_cut();
            let observed = engine.observe_material(start).expect("observe");
            seen = Some((seqno, data, cut, observed));
        })
        .expect("write");
    let (seqno, data, cut, observed) = seen.expect("landing ran");
    assert!(data, "the landing ran before the data was in its tree");
    assert!(
        cut.is_none_or(|cut| cut < seqno),
        "the cut {cut:?} covered the write at {seqno} between its two trees"
    );
    assert!(
        !matches!(observed, Observation::Observed { through, .. } if through >= seqno),
        "an observation between the two trees covered the write: {observed:?}"
    );
    assert!(engine.complete_cut().is_some_and(|cut| cut >= seqno));
    engine.fold_material_to_cut().expect("fold");
    match engine.observe_material(seqno).expect("observe") {
        Observation::Observed { material, .. } => {
            assert_eq!(material.first, Some(seqno), "the write is the first work");
        }
        other => panic!("observed nothing once the write landed: {other:?}"),
    }
}

/// A direct write outside any commit (background maintenance) is material
/// too, at the seqno it takes.
#[test]
fn a_direct_write_is_material() {
    let dir = TempDir::new().expect("tempdir");
    let engine = oracle_engine(dir.path(), future_base());
    let before = engine.material_between(0, None).expect("material");
    engine
        .put(Partition::Node, b"node:direct", &[9; 12])
        .expect("put");
    engine
        .merge(
            Partition::Counter,
            b"counter:direct",
            &crate::engine::merge::encode_counter_delta(2),
        )
        .expect("merge");
    engine
        .delete(Partition::Node, b"node:direct")
        .expect("delete");
    let after = engine.material_between(0, None).expect("material");
    let added = (11 + 12) + (14 + 8) + 11;
    assert_eq!(after.bytes - before.bytes, added);
}
