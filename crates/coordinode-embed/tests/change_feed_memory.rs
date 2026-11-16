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
            Partition::Schema,
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
    let (engine, oracle) = open_engine(dir);
    let start = engine.change_cursor_at_end(Interest::all()).unwrap();
    let record = NodeRecord::with_labels(vec![label(3)])
        .to_msgpack()
        .expect("record");
    let mut unit: Vec<Mutation> = (1..=NODES)
        .map(|id| Mutation::Put {
            partition: PartitionId::Node,
            key: encode_node_key(0, NodeId::from_raw(id)),
            value: record.clone(),
        })
        .collect();
    unit.push(Mutation::Effects(UnitEffects {
        interpretation: EFFECTS_INTERPRETATION,
        nodes: (1..=NODES)
            .map(|id| NodeBasis {
                key: encode_node_key(0, NodeId::from_raw(id)),
                labels: vec![label(3)],
            })
            .collect(),
        // Many implied labels: each event routes by far more than its row
        // stores, so a read holds events heavier than the rows it decoded.
        implications: vec![Implication {
            label: label(3),
            implies: (4..44).map(label).collect(),
        }],
        ..UnitEffects::new()
    }));
    engine
        .commit_logged(&unit, oracle.next().as_raw(), Journal::Journal)
        .unwrap();
    (engine, start)
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
        engine.read_changes(&start, 16, budget).map(|(batch, _)| {
            let unit = &batch.units[0];
            (unit.events.len(), unit.projected)
        })
    };
    // The least budget the unit is read in, in part, at all.
    let mut limit: u64 = 64 << 10;
    while read(&QueryBudget::new(limit)).is_err() {
        limit += limit / 8;
    }
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

/// The units a read decodes keep charged at least the memory they hold:
/// every list of their effect context, its slots and what each slot owns.
#[test]
fn decoded_units_are_charged_what_they_hold() {
    let dir = tempfile::tempdir().unwrap();
    let (engine, start) = large_unit(dir.path());
    let budget = QueryBudget::new(64 << 20);
    let (read, charge) = engine.change_read(start.position(), 16, &budget).unwrap();
    assert_eq!(read.units.len(), 1);
    let held = held_by(&read.units);
    assert!(
        held <= charge.bytes(),
        "the units hold {held} bytes, charged {}",
        charge.bytes()
    );
}

/// The heap a read's units hold: their list, each unit's operations, and
/// what each operation and its effect context own, by capacity.
fn held_by(units: &[coordinode_storage::engine::change_feed::ChangeUnit]) -> u64 {
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
        other => panic!("the test writes no {other:?}"),
    };
    let bytes = std::mem::size_of_val(units)
        + units
            .iter()
            .map(|u| {
                u.mutations.capacity() * std::mem::size_of::<Mutation>()
                    + u.mutations.iter().map(owned).sum::<usize>()
            })
            .sum::<usize>();
    bytes as u64
}
