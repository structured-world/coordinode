//! The folded material through the engine, checked against the closure
//! definition computed by the test from what it committed: for every key
//! changed since the continuation, its last value, deletion or range
//! deletion and every operand since, each unit once.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use crate::engine::material::{Material, Observation, unit_bytes};
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

fn engine_at(dir: &std::path::Path, base: u64) -> StorageEngine {
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

/// What the test committed, per commit timestamp, in apply order within it.
#[derive(Default, Clone)]
struct Committed(BTreeMap<u64, Vec<Mutation>>);

impl Committed {
    fn commit(&mut self, engine: &StorageEngine, ts: u64, mutations: Vec<Mutation>) {
        engine.apply_proposal_at(&mutations, ts).expect("apply");
        self.0.entry(ts).or_default().extend(mutations);
    }

    /// The closure charge from `s` through every commit below `through`.
    fn charge_from(&self, s: u64, through: u64) -> Material {
        let (keys, tombs) = self.replay(through);
        let first = self.0.range(s..through).next().map(|(&ts, _)| ts);
        let changed: Vec<&Key> = keys.values().filter(|k| k.newest >= s).collect();
        let own: u64 = changed.iter().map(|k| k.own).sum();
        let shared: BTreeSet<_> = changed.iter().filter_map(|k| k.rests_on.clone()).collect();
        Material {
            first,
            bytes: own + shared.iter().map(|t| tombs[t]).sum::<u64>(),
        }
    }

    /// What a fixed snapshot at `s` needs, by definition: every key with a
    /// value below `s`, with its closure; a deleted key nothing; a range
    /// deletion once while a key with a value rests on it.
    fn snapshot_at(&self, s: u64) -> u64 {
        let (keys, tombs) = self.replay(s);
        let visible: Vec<&Key> = keys.values().filter(|k| k.visible).collect();
        let own: u64 = visible.iter().map(|k| k.own).sum();
        let shared: BTreeSet<_> = visible.iter().filter_map(|k| k.rests_on.clone()).collect();
        own + shared.iter().map(|t| tombs[t]).sum::<u64>()
    }

    /// Each key's state after every commit below `through`, and the range
    /// deletions' lengths.
    fn replay(&self, through: u64) -> Replayed {
        let mut keys: BTreeMap<(String, Vec<u8>), Key> = BTreeMap::new();
        let mut tombs: BTreeMap<(String, u64, usize), u64> = BTreeMap::new();
        for (&ts, mutations) in self.0.range(..through) {
            let mut ordinals: BTreeMap<String, usize> = BTreeMap::new();
            for m in mutations {
                let part = match m {
                    Mutation::Put { partition, .. }
                    | Mutation::Delete { partition, .. }
                    | Mutation::Merge { partition, .. }
                    | Mutation::RemoveRange { partition, .. } => format!("{partition:?}"),
                    other => panic!("not a data mutation: {other:?}"),
                };
                let ordinal = ordinals.entry(part.clone()).or_insert(0);
                let this = *ordinal;
                *ordinal += 1;
                match m {
                    Mutation::Put { key, value, .. } => {
                        keys.insert(
                            (part, key.clone()),
                            Key {
                                newest: ts,
                                own: unit_bytes(key, value),
                                rests_on: None,
                                visible: true,
                            },
                        );
                    }
                    Mutation::Delete { key, .. } => {
                        keys.insert(
                            (part, key.clone()),
                            Key {
                                newest: ts,
                                own: unit_bytes(key, &[]),
                                rests_on: None,
                                visible: false,
                            },
                        );
                    }
                    Mutation::Merge { key, operand, .. } => {
                        let k = keys.entry((part, key.clone())).or_insert(Key {
                            newest: ts,
                            own: 0,
                            rests_on: None,
                            visible: true,
                        });
                        k.newest = ts;
                        k.own += unit_bytes(key, operand);
                        k.visible = true;
                    }
                    Mutation::RemoveRange { start, end, .. } => {
                        let id = (part.clone(), ts, this);
                        tombs.insert(id.clone(), unit_bytes(start, end));
                        let lo = (part.clone(), start.clone());
                        let hi = (part, end.clone());
                        for (_, k) in keys.range_mut(lo..hi) {
                            k.newest = ts;
                            k.own = 0;
                            k.rests_on = Some(id.clone());
                            k.visible = false;
                        }
                    }
                    other => panic!("not a data mutation: {other:?}"),
                }
            }
        }
        (keys, tombs)
    }
}

/// A key as the oracle replays it.
struct Key {
    newest: u64,
    own: u64,
    rests_on: Option<(String, u64, usize)>,
    visible: bool,
}

type Replayed = (
    BTreeMap<(String, Vec<u8>), Key>,
    BTreeMap<(String, u64, usize), u64>,
);

fn observed(engine: &StorageEngine, s: u64) -> (u64, Material) {
    match engine.observe_material(s).expect("observe") {
        Observation::Observed {
            through, material, ..
        } => (through, material),
        other => panic!("observed nothing: {other:?}"),
    }
}

fn put(part: PartitionId, key: &str, len: usize) -> Mutation {
    Mutation::Put {
        partition: part,
        key: key.as_bytes().to_vec(),
        value: vec![7; len],
    }
}

/// Commits over several trees applied out of timestamp order, several units
/// of one tree in one commit, overwrites, deletions, counter operands, range
/// deletions over part of the keys and keys rewritten after them: the
/// observation from every continuation equals the closure definition, and
/// equals it again after a flush and after a reopen.
#[test]
fn the_observed_charge_matches_the_closure_of_what_was_committed() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let mut committed = Committed::default();
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut plan: Vec<(u64, Vec<Mutation>)> = Vec::new();
    for i in 0..120u64 {
        let ts = base + 10 + i * 3;
        let mut muts = Vec::new();
        if next() % 12 == 0 {
            // A range deletion commits alone: a batch does not mix it with
            // point writes.
            let k = next() % 5;
            muts.push(Mutation::RemoveRange {
                partition: PartitionId::EdgeProp,
                start: format!("edgeprop:{k}").into_bytes(),
                end: format!("edgeprop:{}", k + 4).into_bytes(),
            });
            plan.push((ts, muts));
            continue;
        }
        // Distinct keys in one commit: two kinds of operation on one key in
        // one proposal are refused.
        let first_key = next() % 10;
        for n in 0..(1 + next() % 3) {
            let k = (first_key + n * 3) % 10;
            muts.push(match next() % 8 {
                0 => Mutation::Delete {
                    partition: PartitionId::Node,
                    key: format!("node:{k}").into_bytes(),
                },
                1 | 2 => Mutation::Merge {
                    partition: PartitionId::Counter,
                    key: format!("counter:{}", k % 3).into_bytes(),
                    operand: crate::engine::merge::encode_counter_delta(1),
                },
                3 => put(
                    PartitionId::EdgeProp,
                    &format!("edgeprop:{k}"),
                    (next() % 40) as usize,
                ),
                _ => put(
                    PartitionId::Node,
                    &format!("node:{k}"),
                    (next() % 200) as usize,
                ),
            });
        }
        plan.push((ts, muts));
    }
    // Applied out of timestamp order, in pairs swapped.
    for pair in plan.chunks(2).rev() {
        for (ts, muts) in pair.iter().rev() {
            // Recorded in apply order per commit; one commit per timestamp.
            committed.0.insert(*ts, muts.clone());
        }
    }
    {
        let engine = engine_at(dir.path(), start);
        {
            // Applied here without holding their timestamps in flight, so
            // nothing may fold until all of them are in: the commit path's
            // rounds wait on this.
            let _no_fold = engine.material_fold.lock();
            for (ts, muts) in plan.iter().rev() {
                engine.apply_proposal_at(muts, *ts).expect("apply");
            }
        }
        engine.fold_material_to_cut().expect("fold");
        let end = plan.last().map(|(ts, _)| ts + 1).expect("commits");
        let check = |engine: &StorageEngine| {
            // What the open wrote below the test's commits is in every
            // snapshot alike.
            let below = engine
                .snapshot_material(base)
                .expect("snapshot")
                .expect("folded");
            for s in (base..end + 2).step_by(1) {
                let (through, material) = observed(engine, s);
                assert_eq!(material, committed.charge_from(s, through + 1), "from {s}");
                assert_eq!(
                    engine
                        .snapshot_material(s)
                        .expect("snapshot")
                        .map(|b| b - below),
                    Some(committed.snapshot_at(s)),
                    "a fixed snapshot at {s}"
                );
            }
        };
        check(&engine);
        for &part in Partition::all() {
            engine
                .tree(part)
                .expect("tree")
                .flush_active_memtable(0)
                .expect("flush");
        }
        check(&engine);
    }
    let engine = engine_at(dir.path(), base + 100_000);
    engine.fold_material_to_cut().expect("fold");
    for s in base..base + 400 {
        let (through, material) = observed(&engine, s);
        assert_eq!(
            material,
            committed.charge_from(s, through + 1),
            "after reopen, from {s}"
        );
    }
}

/// A key overwritten many times since the continuation costs its last value,
/// not what was written in all.
#[test]
fn overwrites_since_the_continuation_cost_the_last_value() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let engine = engine_at(dir.path(), start);
    let mut committed = Committed::default();
    for i in 0..200u64 {
        committed.commit(
            &engine,
            base + i,
            vec![put(PartitionId::Node, "node:hot", 1_000)],
        );
    }
    engine.fold_material_to_cut().expect("fold");
    let (_, material) = observed(&engine, base);
    assert_eq!(material.bytes, unit_bytes(b"node:hot", &[7; 1_000]));
    assert_eq!(
        material.first,
        Some(base),
        "the oldest work still dates the consumer"
    );
}

/// The newest commit of a source that went idle is folded: the round's bound
/// is one past the complete cut, the last commit included.
#[test]
fn the_last_commit_of_an_idle_source_is_folded() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let engine = engine_at(dir.path(), start);
    let mut committed = Committed::default();
    committed.commit(&engine, base, vec![put(PartitionId::Node, "node:last", 30)]);
    engine.fold_material_to_cut().expect("fold");
    let (through, material) = observed(&engine, base);
    assert!(through >= base);
    assert_eq!(material, committed.charge_from(base, through + 1));
    assert!(material.bytes > 0);
}

/// A commit applied twice at its timestamp (a replay) is folded once, both
/// when it is replayed before the fold and after.
#[test]
fn a_replayed_commit_is_folded_once() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let engine = engine_at(dir.path(), start);
    let mut committed = Committed::default();
    let muts = vec![
        Mutation::Merge {
            partition: PartitionId::Counter,
            key: b"counter:x".to_vec(),
            operand: crate::engine::merge::encode_counter_delta(1),
        },
        Mutation::Merge {
            partition: PartitionId::Counter,
            key: b"counter:x".to_vec(),
            operand: crate::engine::merge::encode_counter_delta(1),
        },
    ];
    committed.commit(&engine, base, muts.clone());
    engine
        .apply_proposal_at(&muts, base)
        .expect("replay before the fold");
    engine.fold_material_to_cut().expect("fold");
    engine
        .apply_proposal_at(&muts, base)
        .expect("replay after the fold");
    engine.fold_material_to_cut().expect("fold");
    let (through, material) = observed(&engine, base);
    assert_eq!(material, committed.charge_from(base, through + 1));
    assert_eq!(
        material.bytes,
        2 * unit_bytes(b"counter:x", &crate::engine::merge::encode_counter_delta(1)),
        "two equal operands at one commit are two units, each once"
    );
}

/// A commit with an earlier timestamp still in flight bounds the round:
/// what was applied above it is not folded until it lands, and then the
/// earlier commit is folded before the later one.
#[test]
fn a_commit_in_flight_bounds_the_round() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let engine = engine_at(dir.path(), start);
    let mut committed = Committed::default();
    let (held_ts, held) = engine
        .pending_commits()
        .admit_allocated(
            || base,
            Vec::new(),
            Vec::new(),
            crate::engine::pending::Material::Changes,
        )
        .expect("admit");
    committed.commit(
        &engine,
        base + 50,
        vec![put(PartitionId::Node, "node:k", 500)],
    );
    engine.fold_material_to_cut().expect("fold");
    let (through, material) = observed(&engine, held_ts - 1);
    assert!(through < held_ts, "the round passed a commit in flight");
    assert_eq!(material, Material::default());
    // The earlier commit lands: a small value under the same key, older.
    engine
        .apply_proposal_at(&[put(PartitionId::Node, "node:k", 1)], held_ts)
        .expect("apply");
    committed
        .0
        .insert(held_ts, vec![put(PartitionId::Node, "node:k", 1)]);
    drop(held);
    engine.fold_material_to_cut().expect("fold");
    let (through, material) = observed(&engine, held_ts);
    assert_eq!(material, committed.charge_from(held_ts, through + 1));
    assert_eq!(
        material.bytes,
        unit_bytes(b"node:k", &[7; 500]),
        "the later value is the newest"
    );
}

/// A round cut short by its budget leaves every tree's cursor where its
/// last batch left it: after a reopen the fold resumes from there and no
/// unit is folded twice.
#[test]
fn a_round_cut_short_resumes_without_folding_twice() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let mut committed = Committed::default();
    {
        let engine = engine_at(dir.path(), start);
        // Held through the commits: the commit path folds none of them.
        let mut folder = engine.material_fold.lock();
        for i in 0..600u64 {
            committed.commit(
                &engine,
                base + i,
                vec![Mutation::Merge {
                    partition: PartitionId::Counter,
                    key: format!("counter:{}", i % 7).into_bytes(),
                    operand: crate::engine::merge::encode_counter_delta(1),
                }],
            );
        }
        let target = engine.complete_cut().expect("cut") + 1;
        assert!(
            engine
                .fold_round(&mut folder, target, 300, true)
                .expect("round")
                == super::material_fold::Progress::Short,
            "a round of 300 units finished 600"
        );
    }
    let engine = engine_at(dir.path(), base + 100_000);
    engine.fold_material_to_cut().expect("fold");
    for s in [
        base,
        base + 1,
        base + 299,
        base + 300,
        base + 599,
        base + 600,
    ] {
        let (through, material) = observed(&engine, s);
        assert_eq!(material, committed.charge_from(s, through + 1), "from {s}");
    }
}

/// A partition installed from a copy is folded again from what was put
/// back: an operand merged onto an installed row after the installation is
/// charged with that row as its base. Until the fold has gone through the
/// installed rows no round is published, so nothing reads the cleared state
/// as the material.
#[test]
fn an_installed_partition_is_folded_from_what_was_put_back() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let engine = engine_at(dir.path(), start);
    let mut committed = Committed::default();
    committed.commit(
        &engine,
        base,
        vec![put(PartitionId::Counter, "counter:a", 40)],
    );
    engine.fold_material_to_cut().expect("fold");
    let installed: Rows = vec![(b"counter:a".to_vec(), vec![1; 8])];
    let repair = engine
        .begin_partition_rebuild(Partition::Counter, &installed)
        .expect("begin");
    assert!(
        engine.material_round().expect("round").is_none(),
        "the round outlived the clear"
    );
    engine
        .fold_material_to_cut()
        .expect("fold during the repair");
    assert!(
        engine.material_round().expect("round").is_none(),
        "a round was published over a tree under repair"
    );
    engine.finish_raft_rebuild(repair, 1, &[]).expect("finish");
    let operand = crate::engine::merge::encode_counter_delta(1);
    engine
        .apply_proposal_at(
            &[Mutation::Merge {
                partition: PartitionId::Counter,
                key: b"counter:a".to_vec(),
                operand: operand.clone(),
            }],
            base + 100,
        )
        .expect("merge");
    engine.fold_material_to_cut().expect("fold");
    let (_, material) = observed(&engine, base + 100);
    assert_eq!(
        material.bytes,
        unit_bytes(b"counter:a", &[1; 8]) + unit_bytes(b"counter:a", &operand),
        "the installed row is the operand's base"
    );
}

/// A range deletion over more keys than a round's budget spends that budget
/// on its keys and stops: the round folds no more keys than it may, leaves
/// the range's next key in the cursor and publishes nothing; the rounds
/// after it, also after a reopen, finish the range with the exact charge.
#[test]
fn a_range_past_the_budget_stops_the_round_and_resumes() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let mut committed = Committed::default();
    const KEYS: u64 = 1_000;
    const BUDGET: usize = 300;
    {
        let engine = engine_at(dir.path(), start);
        let mut folder = engine.material_fold.lock();
        for i in 0..KEYS {
            committed.commit(
                &engine,
                base + i,
                vec![put(PartitionId::Node, &format!("node:{i:05}"), 10)],
            );
        }
        let target = engine.complete_cut().expect("cut") + 1;
        while engine
            .fold_round(&mut folder, target, usize::MAX, true)
            .expect("fold the keys")
            == super::material_fold::Progress::Short
        {}
        let range = vec![Mutation::RemoveRange {
            partition: PartitionId::Node,
            start: b"node:".to_vec(),
            end: b"node;".to_vec(),
        }];
        committed.commit(&engine, base + KEYS + 10, range);
        let round = engine.material_round().expect("round");
        let target = engine.complete_cut().expect("cut") + 1;
        assert_eq!(
            engine
                .fold_round(&mut folder, target, BUDGET, true)
                .expect("round"),
            super::material_fold::Progress::Short,
            "a range of {KEYS} keys finished in a budget of {BUDGET}"
        );
        assert_eq!(
            engine.material_round().expect("round"),
            round,
            "a round left short was published"
        );
        let tree = engine.tree(Partition::Node).expect("tree");
        let rested = (0..KEYS)
            .filter(|i| {
                crate::engine::material::fold::read(tree, format!("node:{i:05}").as_bytes())
                    .expect("descriptor")
                    .is_some_and(|d| d.rests_on.is_some())
            })
            .count();
        assert!(
            rested <= BUDGET,
            "a round of {BUDGET} folded {rested} keys of the range"
        );
        assert!(rested > 0, "the round made no progress on the range");
    }
    let engine = engine_at(dir.path(), base + 100_000);
    while !engine.fold_material_to_cut().expect("fold") {}
    for s in [base, base + KEYS, base + KEYS + 10, base + KEYS + 11] {
        let (through, material) = observed(&engine, s);
        assert_eq!(material, committed.charge_from(s, through + 1), "from {s}");
    }
}

/// A round holds no more memory than it may: with room for a few units, the
/// first round stops short after folding those and publishes nothing; the
/// rounds after it, each as small, finish with the exact charge, no unit
/// folded twice or lost.
#[test]
fn a_round_holds_no_more_memory_than_it_may() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let engine = engine_at(dir.path(), start);
    let mut committed = Committed::default();
    let mut folder = engine.material_fold.lock();
    const UNITS: u64 = 200;
    for i in 0..UNITS {
        let key = format!("node:{i:03}:{}", "k".repeat(2_000));
        committed.commit(&engine, base + i, vec![put(PartitionId::Node, &key, 100)]);
    }
    let target = engine.complete_cut().expect("cut") + 1;
    // Room for a handful of units of 2 KB keys: each reserves its key's
    // copies and its summary moves at their most, a range page the longest
    // key it could resume from, and the fold its reads at their longest
    // through the tree's runs.
    const MEMORY: u64 = 5 << 18;
    let round = engine.material_round().expect("round");
    assert_eq!(
        engine
            .fold_round_within(&mut folder, target, usize::MAX, MEMORY, true)
            .expect("round"),
        super::material_fold::Progress::Short,
        "{UNITS} units of 2 KB keys fitted in {MEMORY} bytes"
    );
    assert_eq!(engine.material_round().expect("round"), round);
    let mut rounds = 1;
    while engine
        .fold_round_within(&mut folder, target, usize::MAX, MEMORY, true)
        .expect("round")
        == super::material_fold::Progress::Short
    {
        rounds += 1;
        assert!(rounds < 1_000, "the rounds made no progress");
    }
    assert!(
        rounds > 2,
        "{rounds} rounds folded {UNITS} units in {MEMORY} bytes"
    );

    // A range deletion over those keys walks no more of them per round than
    // the fold's memory holds: its pages read descriptors, not units.
    committed.commit(
        &engine,
        base + 300,
        vec![Mutation::RemoveRange {
            partition: PartitionId::Node,
            start: b"node:".to_vec(),
            end: b"node;".to_vec(),
        }],
    );
    let target = engine.complete_cut().expect("cut") + 1;
    assert_eq!(
        engine
            .fold_round_within(&mut folder, target, usize::MAX, MEMORY, true)
            .expect("round"),
        super::material_fold::Progress::Short
    );
    let tree = engine.tree(Partition::Node).expect("tree");
    let rested = (0..UNITS)
        .filter(|i| {
            let key = format!("node:{i:03}:{}", "k".repeat(2_000));
            crate::engine::material::fold::read(tree, key.as_bytes())
                .expect("descriptor")
                .is_some_and(|d| d.rests_on.is_some())
        })
        .count();
    assert!(
        (1..=8).contains(&rested),
        "a round in {MEMORY} bytes folded {rested} keys of the range"
    );
    while engine
        .fold_round_within(&mut folder, target, usize::MAX, MEMORY, true)
        .expect("round")
        == super::material_fold::Progress::Short
    {}
    drop(folder);
    for s in [
        base,
        base + 30,
        base + UNITS - 1,
        base + UNITS,
        base + 300,
        base + 301,
    ] {
        let (through, material) = observed(&engine, s);
        assert_eq!(material, committed.charge_from(s, through + 1), "from {s}");
    }
}

/// A fold that writes into a tree whose memtable a flush just emptied starts
/// a memtable the flush then owes: the room for that is admitted before the
/// write, so the store stays within its limit, and once there is room the
/// fold goes on to the same result.
#[test]
fn a_fold_into_a_flushed_tree_admits_the_flush_it_starts() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )
        .with_hard_limit_bytes(1 << 30),
    ]);
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(start)));
    let engine = StorageEngine::open_with_oracle(&config, oracle).expect("open");
    let mut committed = Committed::default();
    let mut folder = engine.material_fold.lock();
    for i in 0..20u64 {
        committed.commit(
            &engine,
            base + i,
            vec![put(PartitionId::Node, &format!("node:{i}"), 50)],
        );
    }
    engine.persist().expect("flush");
    let tree = engine.tree(Partition::Node).expect("tree");
    assert!(tree.active_memtable_is_empty(), "the flush left a memtable");
    let footprint = engine.footprints().get("default").expect("a limit");
    let (limit, reserve) = (footprint.limit(), footprint.reserve());
    // Room for the fold's records, not for the memtable they start.
    let tight = footprint.committed() + crate::engine::footprint::MEMTABLE_START_DEBT / 2;
    footprint.set_limit(tight, 0);
    assert_eq!(
        footprint.pending_limit(),
        None,
        "the store is over its limit"
    );
    let target = engine.complete_cut().expect("cut") + 1;
    let tight_round = engine.fold_round(&mut folder, target, usize::MAX, true);
    assert!(
        footprint.committed() <= footprint.limit(),
        "a fold took the store past its limit: {} over {} ({tight_round:?})",
        footprint.committed(),
        footprint.limit()
    );
    footprint.set_limit(limit, reserve);
    while engine
        .fold_round(&mut folder, target, usize::MAX, true)
        .expect("fold")
        == super::material_fold::Progress::Short
    {}
    drop(folder);
    for s in [base, base + 10, base + 20] {
        let (through, material) = observed(&engine, s);
        assert_eq!(material, committed.charge_from(s, through + 1), "from {s}");
    }
}

/// A round that only has a commit to complete (its units folded by an
/// earlier round that stopped at its end) needs memory for that too: with
/// none it writes nothing and leaves the cursor where it was; with room it
/// completes the commit, exactly.
#[test]
fn completing_a_commit_without_memory_writes_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let engine = engine_at(dir.path(), start);
    let mut committed = Committed::default();
    let mut folder = engine.material_fold.lock();
    committed.commit(
        &engine,
        base,
        (0..10u64)
            .map(|i| put(PartitionId::Node, &format!("node:{i}"), 50))
            .collect(),
    );
    let target = engine.complete_cut().expect("cut") + 1;
    // Exactly the commit's units: the round stops at its end, open.
    assert_eq!(
        engine
            .fold_round_within(&mut folder, target, 10, 16 << 20, true)
            .expect("round"),
        super::material_fold::Progress::Short
    );
    let tree = engine.tree(Partition::Node).expect("tree");
    let cursor = || {
        tree.get(
            super::material_fold::CURSOR_KEY.as_slice(),
            lsm_tree::SeqNo::MAX,
        )
        .expect("read")
        .map(|v| v.to_vec())
    };
    let open = cursor();
    assert!(open.is_some(), "the units were folded");
    for memory in [0u64, 1, 4 << 10] {
        assert_eq!(
            engine
                .fold_round_within(&mut folder, target, usize::MAX, memory, true)
                .expect("round"),
            super::material_fold::Progress::Short,
            "completed in {memory} bytes"
        );
        assert_eq!(cursor(), open, "a round in {memory} bytes moved the cursor");
    }
    assert_eq!(
        engine
            .fold_round_within(&mut folder, target, usize::MAX, 16 << 20, true)
            .expect("round"),
        super::material_fold::Progress::Reached
    );
    drop(folder);
    let (through, material) = observed(&engine, base);
    assert_eq!(material, committed.charge_from(base, through + 1));
}

/// A fold batch the footprint refuses changes nothing: no cursor moves, no
/// snapshot record is written, and once there is room the fold writes the
/// same result it would have.
#[test]
fn a_fold_refused_room_changes_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let engine = engine_at(dir.path(), start);
    let mut committed = Committed::default();
    let mut folder = engine.material_fold.lock();
    for i in 0..20u64 {
        committed.commit(
            &engine,
            base + i,
            vec![put(PartitionId::Node, &format!("node:{i}"), 50)],
        );
    }
    let tree = engine.tree(Partition::Node).expect("tree");
    let target = engine.complete_cut().expect("cut") + 1;
    // The footprint refuses the fold's batches room.
    super::material_fold::REFUSE_ROOM.with(|refuse| refuse.set(true));
    assert!(
        engine
            .fold_round(&mut folder, target, usize::MAX, true)
            .is_err(),
        "a fold was written without room"
    );
    assert!(
        tree.get(
            super::material_fold::CURSOR_KEY.as_slice(),
            lsm_tree::SeqNo::MAX
        )
        .expect("read")
        .is_none(),
        "a refused fold moved the cursor"
    );
    assert!(
        tree.get(
            super::material_fold::VISIBLE_KEY.as_slice(),
            lsm_tree::SeqNo::MAX
        )
        .expect("read")
        .is_none(),
        "a refused fold wrote a snapshot record"
    );
    super::material_fold::REFUSE_ROOM.with(|refuse| refuse.set(false));
    while engine
        .fold_round(&mut folder, target, usize::MAX, true)
        .expect("fold")
        == super::material_fold::Progress::Short
    {}
    drop(folder);
    let below = engine
        .snapshot_material(base)
        .expect("snapshot")
        .expect("folded");
    for s in [base, base + 10, base + 20] {
        let (through, material) = observed(&engine, s);
        assert_eq!(material, committed.charge_from(s, through + 1), "from {s}");
        assert_eq!(
            engine
                .snapshot_material(s)
                .expect("snapshot")
                .map(|b| b - below),
            Some(committed.snapshot_at(s)),
            "a fixed snapshot at {s}"
        );
    }
}

/// Keys of the longest length a tree takes are folded like any other: their
/// descriptors and their entries in a range deletion's dependents index fit
/// an LSM key, two of them sharing every byte but the last stay apart, and
/// the value is read back as written, also after a reopen.
#[test]
fn keys_of_the_longest_length_are_folded() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let mut committed = Committed::default();
    let longest = |last: u8| {
        let mut key = b"counter:".to_vec();
        key.resize(u16::MAX as usize - 1, b'k');
        key.push(last);
        key
    };
    let (a, b) = (longest(b'a'), longest(b'b'));
    let operand = crate::engine::merge::encode_counter_delta(1);
    {
        let engine = engine_at(dir.path(), start);
        for (i, key) in [&a, &b].into_iter().enumerate() {
            committed.commit(
                &engine,
                base + i as u64,
                vec![Mutation::Put {
                    partition: PartitionId::Counter,
                    key: key.clone(),
                    value: 5i64.to_le_bytes().to_vec(),
                }],
            );
        }
        committed.commit(
            &engine,
            base + 10,
            vec![Mutation::RemoveRange {
                partition: PartitionId::Counter,
                start: b"counter:".to_vec(),
                end: b"counter;".to_vec(),
            }],
        );
        committed.commit(
            &engine,
            base + 20,
            vec![Mutation::Merge {
                partition: PartitionId::Counter,
                key: a.clone(),
                operand: operand.clone(),
            }],
        );
        while !engine.fold_material_to_cut().expect("fold") {}
        for s in [base, base + 10, base + 20, base + 21] {
            let (through, material) = observed(&engine, s);
            assert_eq!(material, committed.charge_from(s, through + 1), "from {s}");
        }
        assert!(
            engine.get(Partition::Counter, &a).expect("read").is_some(),
            "the longest key's value is read back"
        );
    }
    let engine = engine_at(dir.path(), base + 100_000);
    while !engine.fold_material_to_cut().expect("fold") {}
    let (through, material) = observed(&engine, base + 20);
    assert_eq!(material, committed.charge_from(base + 20, through + 1));
    assert!(engine.get(Partition::Counter, &b).expect("read").is_none());
}

/// A fold write cut between its snapshot records and its batch (a crash
/// between the two writes) leaves records the next fold writes again with
/// the same values, from the cursor the batch did not move; after a reopen
/// the delta and every fixed snapshot are charged exactly.
#[test]
fn a_fold_cut_between_its_marks_and_its_batch_is_rewritten_alike() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let mut committed = Committed::default();
    {
        let engine = engine_at(dir.path(), start);
        for i in 0..20u64 {
            committed.commit(
                &engine,
                base + i,
                vec![put(
                    PartitionId::Node,
                    &format!("node:{}", i % 7),
                    40 + i as usize,
                )],
            );
        }
        super::material_fold::STOP_AFTER_MARKS.with(|s| s.set(true));
        let cut = engine.fold_material_to_cut();
        super::material_fold::STOP_AFTER_MARKS.with(|s| s.set(false));
        assert!(cut.is_err(), "the write was not cut");
        let tree = engine.tree(Partition::Node).expect("tree");
        assert!(
            tree.get(
                super::material_fold::CURSOR_KEY.as_slice(),
                lsm_tree::SeqNo::MAX
            )
            .expect("read")
            .is_none(),
            "the cut write moved the cursor"
        );
        engine.persist().expect("persist what was written");
    }
    let engine = engine_at(dir.path(), start);
    engine.fold_material_to_cut().expect("fold again");
    let below = engine
        .snapshot_material(base)
        .expect("snapshot")
        .expect("folded");
    for s in [base, base + 5, base + 19, base + 20] {
        let (through, material) = observed(&engine, s);
        assert_eq!(material, committed.charge_from(s, through + 1), "from {s}");
        assert_eq!(
            engine
                .snapshot_material(s)
                .expect("snapshot")
                .map(|b| b - below),
            Some(committed.snapshot_at(s)),
            "a fixed snapshot at {s}"
        );
    }
}

/// A retirement refused for room is not lost: once there is room, the next
/// round retires the records with no new write to bring it on, and an open
/// after the refusal retires them too.
#[test]
fn a_refused_retirement_is_retried_without_new_writes() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let later = base + 2 * crate::engine::material::RETIRE_WINDOW;
    let below = later - later % crate::engine::material::RETIRE_WINDOW;
    let retained = |engine: &StorageEngine| {
        let tree = engine.tree(Partition::Node).expect("tree");
        crate::engine::material::units_from(tree, (0, 0), below, lsm_tree::SeqNo::MAX, 8, None)
            .expect("units")
            .0
            .len()
    };
    let refuse = |on: bool| super::material_fold::REFUSE_RETIRE.with(|r| r.set(on));
    let mut committed = Committed::default();
    {
        let engine = engine_at(dir.path(), start);
        engine.set_retention_window(std::time::Duration::from_secs(10 * 3_600));
        committed.commit(&engine, base, vec![put(PartitionId::Node, "node:a", 10)]);
        committed.commit(&engine, later, vec![put(PartitionId::Node, "node:b", 10)]);
        refuse(true);
        engine.fold_material_to_cut().expect("fold");
        assert!(retained(&engine) > 0, "retired without room");
        // Room again, and no write since: the next round retires them.
        refuse(false);
        engine.fold_material_to_cut().expect("fold");
        assert_eq!(retained(&engine), 0, "a refused retirement was not retried");
        let (through, material) = observed(&engine, base);
        assert_eq!(material, committed.charge_from(base, through + 1));
    }
    // Refused, then the store closes before a retry: its open retires them.
    let dir = TempDir::new().expect("tempdir");
    let mut committed = Committed::default();
    {
        let engine = engine_at(dir.path(), start);
        engine.set_retention_window(std::time::Duration::from_secs(10 * 3_600));
        committed.commit(&engine, base, vec![put(PartitionId::Node, "node:a", 10)]);
        committed.commit(&engine, later, vec![put(PartitionId::Node, "node:b", 10)]);
        refuse(true);
        engine.fold_material_to_cut().expect("fold");
        assert!(retained(&engine) > 0, "retired without room");
        refuse(false);
        engine.persist().expect("persist");
    }
    let engine = engine_at(dir.path(), start);
    assert_eq!(retained(&engine), 0, "the open did not retire them");
}

/// Unit records the published rounds folded are dropped a window at a
/// time, and the observation stays exact without them; the material records
/// that date the work stay.
#[test]
fn folded_unit_records_are_retired_a_window_at_a_time() {
    let dir = TempDir::new().expect("tempdir");
    let start = future_base();
    let base = start + 10_000;
    let engine = engine_at(dir.path(), start);
    engine.set_retention_window(std::time::Duration::from_secs(10 * 3_600));
    let mut committed = Committed::default();
    committed.commit(&engine, base, vec![put(PartitionId::Node, "node:a", 10)]);
    let later = base + 2 * crate::engine::material::RETIRE_WINDOW;
    committed.commit(&engine, later, vec![put(PartitionId::Node, "node:b", 10)]);
    engine.fold_material_to_cut().expect("fold");
    let tree = engine.tree(Partition::Node).expect("tree");
    let below = later - later % crate::engine::material::RETIRE_WINDOW;
    assert!(
        crate::engine::material::units_from(tree, (0, 0), below, lsm_tree::SeqNo::MAX, 8, None)
            .expect("units")
            .0
            .is_empty(),
        "folded unit records were kept"
    );
    let (through, material) = observed(&engine, base);
    assert_eq!(material, committed.charge_from(base, through + 1));
}
