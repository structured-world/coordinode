//! The memory a change feed read holds at once, counted by the allocator:
//! a read keeps within the budget it is given, whatever its units project
//! to and whatever the reader stops short of. A binary of its own, since it
//! replaces the global allocator.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use coordinode_core::budget::QueryBudget;
use coordinode_core::graph::node::{NodeId, NodeRecord, encode_node_key};
use coordinode_core::schema::label_id::LabelId;
use coordinode_core::txn::proposal::{
    EFFECTS_INTERPRETATION, Implication, Mutation, NodeBasis, PartitionId, UnitEffects,
};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::Journal;
use coordinode_storage::engine::change_feed::{ChangeCursor, Interest};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::alloc_peak;

#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

/// Nodes the large unit of these tests writes.
const NODES: u64 = 20_000;

fn label(raw: u32) -> LabelId {
    LabelId::new(raw).expect("nonzero")
}

/// An engine with a journal and a deployment, and its clock.
fn open_engine(dir: &std::path::Path) -> (StorageEngine, Arc<TimestampOracle>) {
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(
        &StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]),
        Arc::clone(&oracle),
    )
    .unwrap();
    engine
        .put(
            Partition::Deployment,
            coordinode_core::group::ENROLLED_DEPLOYMENT_KEY,
            &[7; 16],
        )
        .unwrap();
    (engine, oracle)
}

/// An engine whose journal holds one unit writing [`NODES`] nodes of a type
/// implying many labels, each with the labels it had before, and a cursor
/// before it: what a large import commits.
fn large_unit(dir: &std::path::Path) -> (StorageEngine, ChangeCursor) {
    let mut unit: Vec<Mutation> = (1..=NODES).map(put).collect();
    // Many implied labels: each event routes by far more than its row
    // stores, so a read holds events heavier than the rows it decoded.
    unit.push(context(1..=NODES, Vec::new(), 40));
    journal_with(dir, &unit)
}

/// A node record of the type label 3.
fn put(id: u64) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Node,
        key: encode_node_key(0, NodeId::from_raw(id)),
        value: NodeRecord::with_labels(vec![label(3)])
            .to_msgpack()
            .expect("record"),
    }
}

/// The effect context of a unit changing `nodes`, each of type 3 before
/// it, with the run of deletes at `runs`, type 3 implying `implied` labels.
fn context(nodes: core::ops::RangeInclusive<u64>, runs: Vec<u32>, implied: u32) -> Mutation {
    Mutation::Effects(UnitEffects {
        interpretation: EFFECTS_INTERPRETATION,
        nodes: nodes
            .map(|id| NodeBasis {
                key: encode_node_key(0, NodeId::from_raw(id)),
                labels: vec![label(3)],
            })
            .collect(),
        runs,
        implications: vec![Implication {
            label: label(3),
            implies: (4..4 + implied).map(label).collect(),
        }],
        ..UnitEffects::new()
    })
}

/// An engine whose journal holds `unit`, in the store's namespace, and a
/// cursor before it.
fn journal_with(dir: &std::path::Path, unit: &[Mutation]) -> (StorageEngine, ChangeCursor) {
    let (engine, oracle) = open_engine(dir);
    let start = engine.change_cursor_at_end(Interest::all()).unwrap();
    let unit = coordinode_storage::engine::scope::store_unit(unit.iter().cloned());
    engine
        .commit_logged(&unit, oracle.next().as_raw(), Journal::Journal)
        .unwrap();
    (engine, start)
}

/// Read from `start` within each of `limits`, refused or not, and assert
/// the read held no more than its budget at any moment when the store holds
/// nothing for later readers, every entry then decoded within the read's own
/// budget; and no more than its budget and the store's own when the store
/// holds decoded entries for later readers within a budget of that size.
fn holds_within(
    engine: &StorageEngine,
    start: &ChangeCursor,
    limits: impl IntoIterator<Item = u64>,
) {
    for limit in limits {
        engine.set_change_feed_cache_bytes(0);
        let budget = QueryBudget::new(limit);
        let (_, peak) =
            alloc_peak::peak_of(|| engine.read_changes(start, 16, &budget, None).map(|_| ()));
        assert!(
            peak as u64 <= limit,
            "held {peak} bytes at once within a {limit}-byte budget"
        );

        let shared = usize::try_from(limit).expect("a test budget");
        engine.set_change_feed_cache_bytes(shared);
        let budget = QueryBudget::new(limit);
        let (_, peak) =
            alloc_peak::peak_of(|| engine.read_changes(start, 16, &budget, None).map(|_| ()));
        assert!(
            peak as u64 <= 2 * limit,
            "held {peak} bytes at once within a {limit}-byte budget and a {shared}-byte one \
             for later readers"
        );
        // A follower's read also keeps the interest's selection of each
        // unit for later readers, within the same store budget.
        engine.set_change_feed_cache_bytes(0);
        engine.set_change_feed_cache_bytes(shared);
        let follower = engine.follow_changes(start.interest()).expect("follow");
        let budget = QueryBudget::new(limit);
        let (_, peak) = alloc_peak::peak_of(|| {
            engine
                .read_changes(start, 16, &budget, Some(&follower))
                .map(|_| ())
        });
        assert!(
            peak as u64 <= 2 * limit,
            "a follower held {peak} bytes at once within a {limit}-byte budget and a \
             {shared}-byte one for later readers"
        );
        drop(follower);
        // Nothing held for the next limit's reads.
        engine.set_change_feed_cache_bytes(0);
    }
}

/// Nodes the units changing nodes again write.
const AGAIN: u64 = 200;

/// Labels type 3 implies in the units changing nodes again: the lists a
/// node's later change starts from are large.
const IMPLIED: u32 = 400;

/// A node changed again later in its unit keeps the labels its first change
/// left for that later change: a read keeping them, each copy taken from
/// its budget before it is made, holds its budget at every limit around
/// where it starts to fit.
#[test]
fn a_node_changed_again_holds_its_budget() {
    let dir = tempfile::tempdir().unwrap();
    let mut unit: Vec<Mutation> = (1..=AGAIN).flat_map(|id| [put(id), put(id)]).collect();
    unit.push(context(1..=AGAIN, Vec::new(), IMPLIED));
    let (engine, start) = journal_with(dir.path(), &unit);
    // Budgets from well below to past the least one the read succeeds in
    // (210 976 bytes): fixed, so a change of the read does not move them.
    holds_within(&engine, &start, (0..96).map(|step| 146_000 + step * 1024));
}

/// A run of deletes whose keys a later put of the unit changes again: the
/// run's keys are copied only within the budget, kept for the later put.
#[test]
fn a_run_key_changed_again_holds_its_budget() {
    let dir = tempfile::tempdir().unwrap();
    let key = |id| encode_node_key(0, NodeId::from_raw(id));
    let mut unit = vec![Mutation::RemoveRange {
        partition: PartitionId::Node,
        start: key(1),
        end: key(AGAIN + 1),
    }];
    unit.extend((1..=AGAIN).map(put));
    unit.push(context(1..=AGAIN, vec![0], IMPLIED));
    let (engine, start) = journal_with(dir.path(), &unit);
    // Budgets from well below to past the least one the read succeeds in
    // (126 600 bytes): fixed, so a change of the read does not move them.
    holds_within(&engine, &start, (0..96).map(|step| 62_000 + step * 1024));
}

/// A unit far larger than the read's budget is read in fragments, and the
/// read that stops after the first events of it holds no more than its
/// budget: the events it stops short of, and the working state for the
/// unit's other nodes, are not built past what it holds, while the unit's
/// whole count is still told.
#[test]
fn a_read_stopping_inside_a_large_unit_holds_its_budget() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, start) = large_unit(dir.path());
    let read = |budget: &QueryBudget| {
        engine
            .read_changes(&start, 16, budget, None)
            .map(|(batch, _)| {
                let unit = &batch.units[0];
                (unit.events.len(), unit.projected)
            })
    };
    // A first read sets up what the process keeps for every later one (its
    // log statements' registrations): not memory any read holds.
    read(&QueryBudget::new(u64::MAX)).expect("warm-up read");
    // Near the least budget the unit is read in, in part, at all: fixed, so
    // a change of the read does not move the condition it is held to.
    let limit: u64 = 9_222_996;
    let budget = QueryBudget::new(limit);
    let (read, peak) = alloc_peak::peak_of(|| read(&budget));
    let (events, projected) = read.expect("read");
    assert!(
        events > 0 && (events as u64) < NODES,
        "a fragment: {events} events"
    );
    assert_eq!(u64::from(projected), NODES, "the unit's whole count");
    assert!(
        peak as u64 <= limit,
        "held {peak} bytes at once within a {limit}-byte budget"
    );
}

/// The units a read decodes keep charged at least the memory they hold, to
/// the store's budget of decoded entries every reader of the log shares:
/// every list of their effect context, its slots and what each slot owns.
#[test]
fn decoded_units_are_charged_what_they_hold() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, start) = large_unit(dir.path());
    let budget = QueryBudget::new(64 << 20);
    let (read, _charge) = engine.change_read(start.position(), 16, &budget).unwrap();
    let units: Vec<_> = read.units().collect();
    assert_eq!(units.len(), 1);
    let held = held_by(units.iter().copied());
    let (_, charged) = engine.change_feed_cache_bytes();
    assert!(
        held <= charged as u64,
        "the units hold {held} bytes, charged {charged}"
    );
}

/// The heap a read's units hold: their list, each unit's operations, and
/// what each operation and its effect context own, by capacity.
fn held_by<'u>(
    units: impl IntoIterator<Item = &'u coordinode_storage::engine::change_feed::ChangeUnit>,
) -> u64 {
    let labels = |l: &Vec<LabelId>| l.capacity() * std::mem::size_of::<LabelId>();
    let owned = |m: &Mutation| match m {
        Mutation::Put { key, value, .. } => key.capacity() + value.capacity(),
        Mutation::Delete { key, .. } => key.capacity(),
        Mutation::Effects(e) => {
            e.nodes.capacity() * std::mem::size_of::<NodeBasis>()
                + e.nodes
                    .iter()
                    .map(|n| n.key.capacity() + labels(&n.labels))
                    .sum::<usize>()
                + e.removed_edges.capacity() * std::mem::size_of::<Vec<u8>>()
                + e.runs.capacity() * std::mem::size_of::<u32>()
                + e.implications.capacity() * std::mem::size_of::<Implication>()
                + e.implications
                    .iter()
                    .map(|i| labels(&i.implies))
                    .sum::<usize>()
        }
        // The namespace the rows belong to owns no bytes.
        Mutation::Namespace(_) => 0,
        other => panic!("the test writes no {other:?}"),
    };
    let bytes = units
        .into_iter()
        .map(|u| {
            std::mem::size_of_val(u)
                + u.mutations.capacity() * std::mem::size_of::<Mutation>()
                + u.mutations.iter().map(owned).sum::<usize>()
        })
        .sum::<usize>();
    bytes as u64
}
