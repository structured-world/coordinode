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

/// The writes measured in one round.
const WRITES: u32 = 200;

/// Rounds measured per database; the least of them is its cost. A background
/// flush that rotates a memtable inside a round makes that round's writes
/// take arena segments the write itself does not need; a cost every write
/// carries shows in every round.
const ROUNDS: u32 = 3;

/// The least bytes `WRITES` ordinary writes allocate on the writing thread
/// over `ROUNDS` rounds, in a database with `ordinary` registered consumers
/// holding no history and `strict` ones holding it.
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
    let mut least = u64::MAX;
    for round in 0..ROUNDS {
        // Every store starts the measured writes on empty memtables. A
        // memtable takes an arena segment (1.5 MiB) when it starts and when
        // one fills, on the write that gets there; where that falls depends
        // on the bytes written before and on the store's background work, not
        // on the write measured.
        db.persist().unwrap();
        // The fresh memtables' first arena segments go to whichever thread
        // writes into them first: this write, unmeasured, or background work
        // of the store. Taken here, no measured write pays for one, whatever
        // the threads' timing on a loaded host.
        db.execute_cypher(&format!("CREATE (:User {{round: {round}, n: -1}})"))
            .unwrap();
        let ((), _) = alloc_peak::peak_of(|| {
            for i in 0..WRITES {
                db.execute_cypher(&format!("CREATE (:User {{round: {round}, n: {i}}})"))
                    .unwrap();
            }
        });
        least = least.min(alloc_peak::sample_allocated());
    }
    least
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
