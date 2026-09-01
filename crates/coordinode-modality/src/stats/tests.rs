use std::sync::Arc;

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::{CommitContext, Transaction};

use super::*;

fn engine() -> (StorageEngine, Arc<TimestampOracle>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().to_string_lossy().as_ref(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap();
    (engine, oracle, dir)
}

fn counter(engine: &StorageEngine, key: &[u8]) -> i64 {
    engine
        .get(Partition::Counter, key)
        .unwrap()
        .map(|v| i64::from_le_bytes(v.as_ref().try_into().unwrap()))
        .unwrap_or(0)
}

fn commit(txn: &mut Transaction<'_>) {
    let wc = coordinode_core::txn::write_concern::WriteConcern::default();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).unwrap();
}

/// Create / delete / label-change deltas commit atomically with the
/// transaction and fold through the counter merge operator; a full
/// create+delete cycle nets to zero (no drift).
#[test]
fn counters_track_create_delete_and_label_changes() {
    let (engine, oracle, _d) = engine();
    fn mvcc<'a>(e: &'a StorageEngine, o: &'a TimestampOracle) -> Transaction<'a> {
        let snap = e.snapshot();
        Transaction::new(e, Some(o), Timestamp::from_raw(snap), Some(snap))
    }

    // Two nodes: one with two labels, one with one.
    let mut txn = mvcc(&engine, &oracle);
    LocalStatsStore.node_created(&mut txn, ["User", "Admin"]);
    LocalStatsStore.node_created(&mut txn, ["User"]);
    commit(&mut txn);
    assert_eq!(counter(&engine, NODES_TOTAL_KEY), 2);
    assert_eq!(counter(&engine, &label_count_key("User")), 2);
    assert_eq!(counter(&engine, &label_count_key("Admin")), 1);

    // SET :Admin on the second node, REMOVE :Admin from the first.
    let mut txn = mvcc(&engine, &oracle);
    LocalStatsStore.label_added(&mut txn, "Admin");
    LocalStatsStore.label_removed(&mut txn, "Admin");
    commit(&mut txn);
    assert_eq!(counter(&engine, &label_count_key("Admin")), 1);

    // Delete both; everything nets to zero.
    let mut txn = mvcc(&engine, &oracle);
    LocalStatsStore.node_deleted(&mut txn, ["User", "Admin"]);
    LocalStatsStore.node_deleted(&mut txn, ["User"]);
    commit(&mut txn);
    assert_eq!(counter(&engine, NODES_TOTAL_KEY), 0);
    assert_eq!(counter(&engine, &label_count_key("User")), 0);
    assert_eq!(counter(&engine, &label_count_key("Admin")), 0);
}

/// A rolled-back (dropped) transaction leaves no counter residue: deltas are
/// buffered on the transaction, not applied at staging time.
#[test]
fn dropped_transaction_stages_no_counter_deltas() {
    let (engine, oracle, _d) = engine();
    {
        let snap = engine.snapshot();
        let mut txn = Transaction::new(
            &engine,
            Some(&oracle),
            Timestamp::from_raw(snap),
            Some(snap),
        );
        LocalStatsStore.node_created(&mut txn, ["Ghost"]);
        // Dropped without commit.
    }
    assert_eq!(counter(&engine, &label_count_key("Ghost")), 0);
    assert_eq!(counter(&engine, NODES_TOTAL_KEY), 0);
}
