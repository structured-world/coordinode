//! The memory a state cut's continuation read allocates, counted by the
//! allocator: the read is bounded by its budget before it decodes, not only
//! refused by what it decoded. A binary of its own, since it replaces the
//! global allocator.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::alloc::System;
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::budget::QueryBudget;
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::Journal;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::fixed_cut::CutScope;
use coordinode_storage::engine::partition::Partition;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Deletes of keys sharing a long prefix: a few encoded bytes each, far more
/// once decoded.
const DELETES: u32 = 20_000;

/// A frame that expands past the read's budget is refused before it is
/// decoded: the read allocates about its budget, not what the frame expands
/// to.
#[test]
fn a_read_refused_by_its_budget_does_not_decode_past_it() {
    let dir = tempfile::tempdir().unwrap();
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(
        &StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
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
    let many: Vec<Mutation> = (0..DELETES)
        .map(|i| Mutation::Delete {
            partition: PartitionId::Counter,
            key: format!("counter:{i:0>64}").into_bytes(),
        })
        .collect();
    let frame = coordinode_core::txn::frame::encode_unit(
        &many,
        coordinode_core::txn::timestamp::Timestamp::from_raw(1),
    )
    .unwrap()
    .len() as u64;
    engine
        .commit_logged(&many, oracle.next().as_raw(), Journal::Journal)
        .unwrap();
    // What the frame decodes to: each delete and its key.
    let decoded = u64::from(DELETES) * (std::mem::size_of::<Mutation>() as u64 + 72);
    let limit = frame * 4;
    assert!(decoded > limit * 8, "the frame expands far past the budget");

    let budget = QueryBudget::new(limit);
    let region = Region::new(GLOBAL);
    let refused = engine.cut_read(&cut, cut.first(), usize::MAX, &budget);
    let allocated = region.change().bytes_allocated as u64;
    assert!(refused.is_err(), "refused by its budget");
    drop(refused);
    assert!(
        allocated < decoded / 2,
        "allocated {allocated} bytes for a {limit}-byte budget; decoding would take {decoded}"
    );
}
