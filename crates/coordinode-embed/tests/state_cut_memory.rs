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
    let (engine, oracle) = open_engine(dir);
    let scope = CutScope::rows(Partition::Counter, b"counter:".to_vec()).unwrap();
    let cut = engine.fixed_cut(scope, Duration::from_secs(10)).unwrap();
    (engine, oracle, cut)
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

/// A group's log in `dir` holding one entry at `index` with `units`, each
/// stamped `ts`.
fn group_log(dir: &std::path::Path, index: u64, ts: u64, units: &[Vec<Mutation>]) {
    let mut manager =
        coordinode_storage::oplog::manager::OplogManager::open(dir, 0, 64 << 20, 1 << 20, 0)
            .unwrap();
    let ops = units
        .iter()
        .map(|unit| OplogOp::Unit {
            frame: coordinode_core::txn::frame::encode_unit(unit, Timestamp::from_raw(ts)).unwrap(),
        })
        .collect();
    manager
        .append(&OplogEntry {
            ts,
            term: 1,
            index,
            shard: 0,
            ops,
            is_migration: false,
            pre_images: None,
        })
        .unwrap();
    manager.flush().unwrap();
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
    let held = rows.held();
    assert!(
        charge.bytes() >= held as u64,
        "charged {} for a page holding {held}",
        charge.bytes()
    );
}

/// The memory units of a read hold: the batch's room, each unit's
/// operations and the bytes they own.
fn units_held(units: &[coordinode_storage::engine::fixed_cut::CutUnit]) -> u64 {
    let slot = std::mem::size_of::<Mutation>();
    let owned = |m: &Mutation| match m {
        Mutation::Put { key, value, .. } => key.capacity() + value.capacity(),
        Mutation::Delete { key, .. } => key.capacity(),
        Mutation::Merge { key, operand, .. } => key.capacity() + operand.capacity(),
        Mutation::RemoveRange { start, end, .. } => start.capacity() + end.capacity(),
        other => panic!("a row read carries no {other:?}"),
    };
    let bytes: usize = units
        .iter()
        .map(|unit| {
            unit.mutations.capacity() * slot + unit.mutations.iter().map(owned).sum::<usize>()
        })
        .sum();
    bytes as u64
}

/// A short range delete over a scope of a long prefix is clipped to bounds
/// as long as the prefix: the read takes their room before it copies them,
/// keeps it charged with the batch, and is refused where only the short
/// bounds would fit.
#[test]
fn clipped_bounds_are_charged_before_they_are_copied() {
    alloc_peak::install();
    let dir = tempfile::tempdir().unwrap();
    let (engine, _) = open_engine(dir.path());
    let short = b"counter:m".to_vec();
    let long = [short.as_slice(), &[b'x'; 4096]].concat();
    let cut_of = |prefix: &[u8]| {
        engine
            .fixed_cut(
                CutScope::rows(Partition::Counter, prefix.to_vec()).unwrap(),
                Duration::from_secs(10),
            )
            .unwrap()
    };
    let (short_cut, long_cut) = (cut_of(&short), cut_of(&long));
    let wide = vec![Mutation::RemoveRange {
        partition: PartitionId::Counter,
        start: b"counter:a".to_vec(),
        end: b"counter:z".to_vec(),
    }];
    let read = |cut: &FixedCut, budget: &QueryBudget| {
        let log = dir.path().join(format!("log-{}", cut.first()));
        if !log.exists() {
            group_log(
                &log,
                cut.first(),
                cut.bound() + 1,
                std::slice::from_ref(&wide),
            );
        }
        engine
            .cut_read_log(cut, &[log], cut.first(), cut.first() + 1, 16, budget)
            .map(|(read, charge)| (units_held(&read.units), charge.bytes(), read.units))
    };

    let ample = QueryBudget::new(4 << 20);
    let (held, charged, units) = read(&long_cut, &ample).expect("read");
    let Mutation::RemoveRange { start, end, .. } = &units[0].mutations[0] else {
        panic!("a range delete")
    };
    assert_eq!(start.len(), long.len(), "clipped to the long prefix");
    assert!(end.len() >= long.len());
    assert!(
        charged >= held,
        "charged {charged} for units holding {held}"
    );

    // The least budget the short bounds are read within.
    let mut limit = 16 << 10;
    while read(&short_cut, &QueryBudget::new(limit)).is_err() {
        limit += 256;
    }
    let refused = read(&long_cut, &QueryBudget::new(limit)).map(|(held, _, _)| held);
    assert!(
        matches!(
            refused,
            Err(coordinode_storage::error::StorageError::Budget(_))
        ),
        "the long bounds do not fit where the short ones did: {refused:?}"
    );
}

/// A unit's large metadata command, left out by a row scope, is never built:
/// the read takes the unit's row change within a budget far below what the
/// command decodes to.
#[test]
fn a_command_left_out_by_the_scope_is_never_built() {
    alloc_peak::install();
    let dir = tempfile::tempdir().unwrap();
    let (engine, _, cut) = cut_engine(dir.path());
    let unit = vec![
        Mutation::Put {
            partition: PartitionId::Counter,
            key: b"counter:row".to_vec(),
            value: b"v".to_vec(),
        },
        // Empty names: a byte each encoded, a string each decoded.
        Mutation::Command(
            coordinode_core::txn::proposal::MetadataCommand::RegisterFields {
                names: vec![String::new(); 400_000],
            },
        ),
    ];
    let log = dir.path().join("group-log");
    group_log(&log, cut.first(), cut.bound() + 1, &[unit]);
    let limit = 2 << 20;
    let budget = QueryBudget::new(limit);
    let (read, peak) = alloc_peak::peak_of(|| {
        engine
            .cut_read_log(
                &cut,
                std::slice::from_ref(&log),
                cut.first(),
                cut.first() + 1,
                16,
                &budget,
            )
            .map(|(read, _)| read.units.iter().map(|u| u.mutations.len()).sum::<usize>())
    });
    assert_eq!(read.expect("read"), 1, "the row change is taken");
    assert!(
        peak <= limit,
        "held {peak} bytes at once within a {limit}-byte budget"
    );
}

/// A page read cold, over several runs of the tree and a large record,
/// holds no more than its budget: its scan is charged for the runs as they
/// stand, not a fixed allowance.
#[test]
fn a_cold_page_over_several_runs_keeps_its_budget() {
    alloc_peak::install();
    let dir = tempfile::tempdir().unwrap();
    {
        let (engine, oracle) = open_engine(dir.path());
        for run in 0..4u32 {
            let mut rows: Vec<Mutation> = (0..50u32)
                .map(|i| Mutation::Put {
                    partition: PartitionId::Counter,
                    key: format!("counter:{run}:{i:04}").into_bytes(),
                    value: vec![1; 64],
                })
                .collect();
            rows.push(Mutation::Put {
                partition: PartitionId::Counter,
                key: format!("counter:{run}:big").into_bytes(),
                value: vec![2; 48 << 10],
            });
            engine
                .commit_logged(&rows, oracle.next().as_raw(), Journal::Journal)
                .unwrap();
            // Each flush lands as a run of its own.
            engine.persist().unwrap();
        }
    }
    // Reopened: nothing of the tree is cached.
    let (engine, _) = open_engine(dir.path());
    let scope = CutScope::rows(Partition::Counter, b"counter:".to_vec()).unwrap();
    let cut = engine.fixed_cut(scope, Duration::from_secs(10)).unwrap();
    // Less than the rows take beside the scan: the page fills its budget.
    let limit = 200 << 10;
    let budget = QueryBudget::new(limit);
    let (page, peak) = alloc_peak::peak_of(|| {
        engine
            .cut_page(&cut, None, 1000, &budget)
            .map(|(rows, _)| rows.len())
    });
    assert!(page.expect("page") > 0);
    assert!(
        peak <= limit,
        "held {peak} bytes at once within a {limit}-byte budget"
    );
}
