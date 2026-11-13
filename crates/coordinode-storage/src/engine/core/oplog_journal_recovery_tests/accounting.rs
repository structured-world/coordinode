//! The accounting tree and the data trees reach the disk independently: a
//! crash can leave any one of them holding a journal entry the other lacks.
//! Replay derives an entry's accounting from the whole entry, whichever data
//! trees already hold their part, and never applies a held operand again.
//! Each case checks the exact logical bytes against the definition and the
//! unit multiplicity (one unit per write, the same operand twice twice).

use super::*;
use crate::engine::material::{self, unit_bytes};

/// Commits per case.
const COMMITS: u64 = 40;

/// The logical bytes the commits of [`commit_all`] add.
fn expected_bytes() -> u64 {
    (0..COMMITS)
        .flat_map(|n| two_tree_commit(&format!("k{n}"), n))
        .map(|m| match m {
            Mutation::Put { key, value, .. } => unit_bytes(&key, &value),
            Mutation::Merge { key, operand, .. } => unit_bytes(&key, &operand),
            _ => 0,
        })
        .sum()
}

/// Commit [`COMMITS`] two-tree entries, each journaled durably; returns the
/// first commit timestamp.
fn commit_all(engine: &StorageEngine, oracle: &TimestampOracle) -> u64 {
    let first = oracle.next().as_raw();
    engine
        .commit_logged(&two_tree_commit("k0", 0), first, Journal::Journal)
        .expect("commit");
    for n in 1..COMMITS {
        commit_at(
            engine,
            oracle,
            &two_tree_commit(&format!("k{n}"), n),
            Journal::Journal,
        );
    }
    first
}

/// The unit records `part`'s scope holds from `from` on.
fn units(engine: &StorageEngine, part: Partition, from: u64) -> usize {
    let heads = *engine.material_heads();
    let scope = engine.material_scope(&heads, part).expect("scope");
    material::units_from(
        &scope,
        (from, 0),
        lsm_tree::SeqNo::MAX,
        lsm_tree::SeqNo::MAX,
        usize::MAX,
        None,
    )
    .expect("units")
    .0
    .len()
}

/// The logical bytes the node and adjacency scopes hold from `from` on: the
/// trees the commits write. The Schema scope also takes the routing an open
/// writes again when a cut took it.
fn bytes(engine: &StorageEngine, from: u64) -> u64 {
    let heads = *engine.material_heads();
    let mut total = material::Material::default();
    for part in [Partition::Node, Partition::Adj] {
        let scope = engine.material_scope(&heads, part).expect("scope");
        material::read_into(&scope, from, None, lsm_tree::SeqNo::MAX, &mut total).expect("read");
    }
    total.bytes
}

/// The store after the cut holds every commit once, its data and its
/// accounting both exact.
fn assert_exact(engine: &StorageEngine, first: u64, case: &str) {
    assert_eq!(
        two_tree_state(engine, &format!("k{}", COMMITS - 1)),
        (true, (0..COMMITS).collect()),
        "{case}: the data"
    );
    assert_eq!(
        bytes(engine, first),
        expected_bytes(),
        "{case}: the logical bytes"
    );
    assert_eq!(
        units(engine, Partition::Node, first),
        COMMITS as usize,
        "{case}: the node units"
    );
    assert_eq!(
        units(engine, Partition::Adj, first),
        COMMITS as usize,
        "{case}: the operands, each its own unit"
    );
    engine.fold_material_to_cut().expect("fold");
    assert!(
        matches!(
            engine.observe_material(first).expect("observe"),
            material::Observation::Observed { .. }
        ),
        "{case}: no observation"
    );
}

/// Reopen after a cut with the retirement of folded unit records refused:
/// the fold at the open retires the units below the retirement window its
/// cursor reaches, which the commits' timestamps cross whenever the case
/// runs over a window boundary of the clock, and the unit multiplicity this
/// file checks is the replay's, read before any retirement.
fn reopen(rig: &PowerRig) -> StorageEngine {
    crate::engine::core::material_fold::REFUSE_RETIRE.with(|refuse| refuse.set(true));
    rig.open().0
}

/// Persist `parts` alone, then lose power.
fn persist_then_cut(rig: &PowerRig, engine: StorageEngine, parts: &[Partition]) {
    for &part in parts {
        engine
            .flush_memtables(engine.tree(part).expect("tree"))
            .expect("flush");
    }
    rig.cut(engine);
}

/// The data trees hold the entries and the accounting tree lost them: the
/// replay restores the accounting alone, from the whole entries, without
/// applying a held operand again.
#[test]
fn accounting_lost_beside_durable_data_is_replayed_from_the_entries() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    let first = commit_all(&engine, &oracle);
    persist_then_cut(&rig, engine, &[Partition::Node, Partition::Adj]);
    let engine = reopen(&rig);
    assert_exact(&engine, first, "data durable, accounting lost");
}

/// The accounting tree holds the entries and the data trees lost them: the
/// replay restores the data alone, and the accounting is not counted twice.
#[test]
fn data_lost_beside_durable_accounting_is_replayed_once() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    let first = commit_all(&engine, &oracle);
    persist_then_cut(&rig, engine, &[Partition::Accounting]);
    let engine = reopen(&rig);
    assert_exact(&engine, first, "accounting durable, data lost");
}

/// One data tree holds the entries, the other and the accounting lost them:
/// the replay restores the missing data tree's part and the whole entries'
/// accounting.
#[test]
fn an_entry_held_by_one_of_its_trees_is_accounted_whole() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    let first = commit_all(&engine, &oracle);
    persist_then_cut(&rig, engine, &[Partition::Node]);
    let engine = reopen(&rig);
    assert_exact(
        &engine,
        first,
        "node durable, adjacency and accounting lost",
    );
}

/// Everything durable but the one data tree: no part is replayed twice.
#[test]
fn an_entry_missing_from_one_data_tree_only_replays_that_part() {
    let rig = PowerRig::new();
    let (engine, oracle) = rig.open();
    let first = commit_all(&engine, &oracle);
    persist_then_cut(&rig, engine, &[Partition::Node, Partition::Accounting]);
    let engine = reopen(&rig);
    assert_exact(&engine, first, "adjacency lost alone");
    // A second cut and open replay nothing more.
    rig.cut(engine);
    let engine = reopen(&rig);
    assert_exact(&engine, first, "a second open");
}
