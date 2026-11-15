use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_core::txn::write_concern::Journal;

use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;

const COUNTER: &[u8] = b"counter:degree:7";

fn config(dir: &tempfile::TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn open(dir: &tempfile::TempDir) -> Arc<StorageEngine> {
    open_with(&config(dir))
}

fn open_with(config: &StorageConfig) -> Arc<StorageEngine> {
    Arc::new(StorageEngine::open_embedded(config, Arc::new(TimestampOracle::new())).expect("open"))
}

/// A commit below the cut journalled without a sync is made durable before
/// the cut is handed off: the view holds no effect a crash can take back.
#[test]
fn a_deferred_commit_below_the_cut_is_durable_before_the_handoff() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    engine
        .commit_logged(&add(1), engine.next_seqno(), Journal::Memory)
        .expect("deferred commit");
    let cut = engine.fixed_cut(Duration::from_secs(10)).expect("cut");
    assert!(
        engine.pending_commits().durable_floor(cut.bound()) >= cut.bound(),
        "every commit below the cut is durable"
    );
}

/// The continuation is not held for the whole copy: journal removed past
/// `L` while the view is copied is RetentionLost, never a partial join.
#[test]
fn a_continuation_pruned_during_the_copy_is_retention_lost() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = config(&dir);
    config.oplog_segment_max_entries = 2;
    config.oplog_retention_secs = 1;
    let engine = open_with(&config);
    for _ in 0..2 {
        engine
            .commit_logged(&add(1), engine.next_seqno(), Journal::Journal)
            .expect("commit");
    }
    let cut = engine.fixed_cut(Duration::from_secs(10)).expect("cut");
    for _ in 0..6 {
        engine
            .commit_logged(&add(1), engine.next_seqno(), Journal::Journal)
            .expect("commit");
    }
    let later = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_secs()
        + 3600;
    let removed = engine.oplog_purge_expired(later, u64::MAX).expect("purge");
    assert!(removed > 0, "the copy outlived the journal");
    let lost = engine.fixed_cut_units(&cut, cut.first(), 16);
    assert!(
        matches!(lost, Err(crate::error::StorageError::RetentionLost { .. })),
        "the join is lost, not partial: {lost:?}"
    );
}

fn add(delta: i64) -> Vec<Mutation> {
    vec![Mutation::Merge {
        partition: PartitionId::Counter,
        key: COUNTER.to_vec(),
        operand: crate::engine::merge::encode_counter_delta(delta),
    }]
}

fn counter(value: Option<bytes::Bytes>) -> i64 {
    value.map_or(0, |v| {
        crate::engine::merge::decode_counter(&v).expect("counter")
    })
}

/// A commit that drew its timestamp below the cut but reached the journal
/// after `L` is in the view, and the join says so: a consumer that copies
/// the view and applies only the units the view does not hold ends with the
/// source's state, a non-idempotent merge counted once. The journal order is
/// not the timestamp order: a later-stamped commit sits before it.
#[test]
fn a_late_commit_below_the_cut_is_in_the_view_and_counted_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open(&dir);
    let commit = |m: &[Mutation], ts: u64| {
        engine
            .commit_logged(m, ts, Journal::Journal)
            .expect("commit");
    };
    commit(&add(1), engine.next_seqno());
    // Drawn below the cut, journalled only after the cut read `L`.
    let (late, held) = engine.pending_commits().obligate(|| engine.next_seqno());
    let during = Arc::clone(&engine);
    let mut held = Some(held);
    *engine.on_cut_bound.lock() = Some(Box::new(move || {
        let Some(held) = held.take() else { return };
        // Stamped above the cut, journalled first.
        during
            .commit_logged(&add(10), during.next_seqno(), Journal::Journal)
            .expect("above");
        during
            .commit_logged(&add(100), late, Journal::Journal)
            .expect("late");
        drop(held);
    }));
    let cut = engine.fixed_cut(Duration::from_secs(10)).expect("cut");
    *engine.on_cut_bound.lock() = None;
    assert!(late < cut.bound(), "the late commit is below the cut");

    let mut destination = counter(
        engine
            .snapshot_get(&cut.bound(), Partition::Counter, COUNTER)
            .expect("view"),
    );
    assert_eq!(destination, 101, "the view holds the late commit");
    let units = engine
        .fixed_cut_units(&cut, cut.first(), cut.join_end() - cut.first())
        .expect("join");
    let late_unit = units
        .iter()
        .find(|u| u.commit_ts == late)
        .expect("the late unit is in the join");
    assert!(late_unit.in_view, "the view already holds it");
    for unit in units.iter().filter(|u| !u.in_view) {
        for m in &unit.mutations {
            if let Mutation::Merge { operand, .. } = m {
                destination += crate::engine::merge::decode_counter(operand).expect("delta");
            }
        }
    }
    let source = counter(engine.get(Partition::Counter, COUNTER).expect("source"));
    assert_eq!(source, 111);
    assert_eq!(destination, source, "every unit counted once");
}
