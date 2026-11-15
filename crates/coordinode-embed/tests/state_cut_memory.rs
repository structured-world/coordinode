//! The memory a state cut's reads hold at once, counted by the allocator:
//! a read keeps within the budget it is given, whatever an entry or a page
//! expands to. A binary of its own, since it replaces the global allocator.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::budget::QueryBudget;
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::Journal;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::fixed_cut::{CutScope, FixedCut};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::oplog::{OplogEntry, OplogOp};

#[path = "../benches/support/alloc_peak.rs"]
mod alloc_peak;

#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

/// An engine with a journal and a deployment, its clock, and a cut of its
/// counters.
fn cut_engine(dir: &std::path::Path) -> (StorageEngine, Arc<TimestampOracle>, FixedCut) {
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
    let scope = CutScope::rows(Partition::Counter, b"counter:".to_vec()).unwrap();
    let cut = engine.fixed_cut(scope, Duration::from_secs(10)).unwrap();
    (engine, oracle, cut)
}

/// `n` deletes of keys sharing a long prefix: a few encoded bytes each, far
/// more once decoded.
fn deletes(n: u32, from: u32) -> Vec<Mutation> {
    (from..from + n)
        .map(|i| Mutation::Delete {
            partition: PartitionId::Counter,
            key: format!("counter:{i:0>64}").into_bytes(),
        })
        .collect()
}

/// One log entry carrying several units, each of whose decoding fits the
/// read's budget on its own and all of which together do not, as a group's
/// log entry batching proposals does: the read is refused, and holds no
/// more than its budget at any moment while it is.
#[test]
fn units_of_one_entry_are_bounded_together() {
    alloc_peak::install();
    let dir = tempfile::tempdir().unwrap();
    let (engine, _, cut) = cut_engine(dir.path());
    let log = dir.path().join("group-log");
    let mut manager =
        coordinode_storage::oplog::manager::OplogManager::open(&log, 0, 64 << 20, 1 << 20, 0)
            .unwrap();
    let units: Vec<OplogOp> = (0..8u32)
        .map(|u| OplogOp::Unit {
            frame: coordinode_core::txn::frame::encode_unit(
                &deletes(2_000, u * 2_000),
                Timestamp::from_raw(cut.bound() + 1),
            )
            .unwrap(),
        })
        .collect();
    manager
        .append(&OplogEntry {
            ts: cut.bound() + 1,
            term: 1,
            index: cut.first(),
            shard: 0,
            ops: units,
            is_migration: false,
            pre_images: None,
        })
        .unwrap();
    manager.flush().unwrap();
    drop(manager);

    let limit = 1 << 20;
    let budget = QueryBudget::new(limit);
    let (refused, peak) = alloc_peak::peak_of(|| {
        engine
            .cut_read_log(
                &cut,
                std::slice::from_ref(&log),
                cut.first(),
                cut.first() + 1,
                16,
                &budget,
            )
            .map(|(read, _)| read.units.len())
    });
    assert!(
        matches!(
            refused,
            Err(coordinode_storage::error::StorageError::Budget(_))
        ),
        "eight units do not fit together: {refused:?}"
    );
    assert!(
        peak <= limit,
        "held {peak} bytes at once within a {limit}-byte budget"
    );
}

/// A frame expanding far past the budget is refused before it is decoded:
/// its payload, its read in place and its decoding together stay within
/// the budget.
#[test]
fn a_frame_expanding_past_the_budget_is_refused_within_it() {
    alloc_peak::install();
    let dir = tempfile::tempdir().unwrap();
    let (engine, oracle, cut) = cut_engine(dir.path());
    let many = deletes(20_000, 0);
    engine
        .commit_logged(&many, oracle.next().as_raw(), Journal::Journal)
        .unwrap();
    let limit = 512 << 10;
    let budget = QueryBudget::new(limit);
    let (refused, peak) = alloc_peak::peak_of(|| {
        engine
            .cut_read(&cut, cut.first(), usize::MAX, &budget)
            .map(|(read, _)| read.units.len())
    });
    assert!(refused.is_err(), "refused by its budget");
    assert!(
        peak <= limit,
        "held {peak} bytes at once within a {limit}-byte budget"
    );
}

/// A unit near both of its decode bounds at once, as many operations as
/// the budget holds operations and as many key bytes as it holds bytes:
/// bounded apart, its decoding would hold about twice the budget; bounded
/// together with its payload and read, it stays within.
#[test]
fn a_unit_at_both_decode_bounds_stays_within_the_budget() {
    alloc_peak::install();
    let dir = tempfile::tempdir().unwrap();
    let (engine, oracle, cut) = cut_engine(dir.path());
    let limit: u64 = 1 << 20;
    let slot = std::mem::size_of::<Mutation>() as u64;
    let ops = limit * 8 / 10 / slot;
    // Each key as long as an operation slot: the keys fill the budget too.
    let width = usize::try_from(slot).unwrap() - "counter:".len();
    let many: Vec<Mutation> = (0..ops)
        .map(|i| Mutation::Delete {
            partition: PartitionId::Counter,
            key: format!("counter:{i:0>width$}").into_bytes(),
        })
        .collect();
    engine
        .commit_logged(&many, oracle.next().as_raw(), Journal::Journal)
        .unwrap();
    drop(many);
    let budget = QueryBudget::new(limit);
    let (read, peak) = alloc_peak::peak_of(|| {
        engine
            .cut_read(&cut, cut.first(), usize::MAX, &budget)
            .map(|(read, _)| read.units.len())
    });
    assert!(read.is_err(), "it does not fit: {read:?}");
    assert!(
        peak <= limit,
        "held {peak} bytes at once within a {limit}-byte budget"
    );
}

/// A read that succeeds keeps what it hands over within its budget, its
/// batch's growth included, and its charge covers it until it is dropped.
#[test]
fn a_successful_read_holds_no_more_than_its_budget() {
    alloc_peak::install();
    let dir = tempfile::tempdir().unwrap();
    let (engine, oracle, cut) = cut_engine(dir.path());
    for i in 0..20 {
        engine
            .commit_logged(
                &deletes(50, i * 50),
                oracle.next().as_raw(),
                Journal::Journal,
            )
            .unwrap();
    }
    let limit = 2 << 20;
    let budget = QueryBudget::new(limit);
    let (read, peak) =
        alloc_peak::peak_of(|| engine.cut_read(&cut, cut.first(), usize::MAX, &budget));
    let (read, charge) = read.expect("read");
    assert_eq!(read.units.len(), 20);
    assert!(
        peak <= limit,
        "held {peak} bytes at once within a {limit}-byte budget"
    );
    // The batch's room for units, not only the units, is charged.
    let room = read.units.capacity()
        * std::mem::size_of::<coordinode_storage::engine::fixed_cut::CutUnit>();
    assert!(
        charge.bytes() >= room as u64,
        "charged {} for a batch of room {room}",
        charge.bytes()
    );
}

/// A small page that succeeds, its rows growing the page's room, holds no
/// more than its budget.
#[test]
fn a_small_page_holds_no_more_than_its_budget() {
    alloc_peak::install();
    let dir = tempfile::tempdir().unwrap();
    let (engine, oracle, _) = cut_engine(dir.path());
    let rows: Vec<Mutation> = (0..3u32)
        .map(|i| Mutation::Put {
            partition: PartitionId::Counter,
            key: format!("counter:{i}").into_bytes(),
            value: vec![0; 100],
        })
        .collect();
    engine
        .commit_logged(&rows, oracle.next().as_raw(), Journal::Journal)
        .unwrap();
    let scope = CutScope::rows(Partition::Counter, b"counter:".to_vec()).unwrap();
    let cut = engine.fixed_cut(scope, Duration::from_secs(10)).unwrap();
    let limit = 128 << 10;
    let budget = QueryBudget::new(limit);
    let (page, peak) = alloc_peak::peak_of(|| engine.cut_page(&cut, None, 16, &budget));
    let (rows, charge) = page.expect("page");
    assert_eq!(rows.len(), 3);
    assert!(
        peak <= limit,
        "held {peak} bytes at once within a {limit}-byte budget"
    );
    // The page's room, not only its rows, is charged while it is held.
    let held = rows.capacity() * std::mem::size_of::<(Vec<u8>, Vec<u8>)>()
        + rows
            .iter()
            .map(|(k, v)| k.capacity() + v.capacity())
            .sum::<usize>();
    assert!(
        charge.bytes() >= held as u64,
        "charged {} for a page holding {held}",
        charge.bytes()
    );
}
