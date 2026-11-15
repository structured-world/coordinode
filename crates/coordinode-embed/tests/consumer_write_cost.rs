//! What registered change-stream consumers cost an ordinary write: nothing.
//! A write records its own data and coverage; it neither writes nor folds
//! anything per key for consumers, nor looks at their registrations. A
//! binary of its own, since it replaces the global allocator.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_embed::Database;
use coordinode_embed::consumer::{
    ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy, InitialSeqno, TopologyScope,
};

#[path = "../benches/support/alloc_peak.rs"]
mod alloc_peak;

#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

/// The writes measured.
const WRITES: u32 = 200;

/// The bytes `WRITES` ordinary writes allocate on the writing thread, in a
/// database with `ordinary` registered consumers holding no history and
/// `strict` ones holding it.
fn cost_of_writes(ordinary: u32, strict: u32) -> u64 {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let register = |id: String, retention| ConsumerRegistration {
        consumer_id: id,
        kind: ConsumerKind::OplogEvents,
        scope: TopologyScope::Shard(0),
        initial_seqno: InitialSeqno::FromNow,
        retention,
    };
    for i in 0..ordinary {
        db.register_consumer(register(
            format!("ordinary-{i}"),
            ConsumerRetentionPolicy::Ordinary,
        ))
        .unwrap();
    }
    for i in 0..strict {
        db.register_consumer(register(
            format!("strict-{i}"),
            ConsumerRetentionPolicy::Strict,
        ))
        .unwrap();
    }
    // Caches and the field dictionary settle before the measured writes.
    for i in 0..WRITES {
        db.execute_cypher(&format!("CREATE (:Warm {{n: {i}}})"))
            .unwrap();
    }
    let ((), _) = alloc_peak::peak_of(|| {
        for i in 0..WRITES {
            db.execute_cypher(&format!("CREATE (:User {{n: {i}}})"))
                .unwrap();
        }
    });
    alloc_peak::sample_allocated()
}

/// Ordinary writes allocate the same with no consumer, with many consumers
/// holding no history and with consumers holding it: no work on a write
/// grows with the registrations.
#[test]
fn an_ordinary_write_costs_the_same_with_registered_consumers() {
    alloc_peak::install();
    let none = cost_of_writes(0, 0);
    let many = cost_of_writes(256, 8);
    // Allowance for what differs between two databases: their identifiers'
    // lengths, their caches' timing.
    let allowance = none / 20;
    assert!(
        many <= none + allowance,
        "{WRITES} writes allocated {many} bytes with 264 consumers, {none} with none"
    );
}
