use super::*;
use coordinode_core::graph::intern::FieldInterner;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::partition::Partition;

fn test_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open engine")
}

fn insert_node_with_timestamp(
    engine: &StorageEngine,
    shard_id: u16,
    node_id: u64,
    label: &str,
    timestamp_us: i64,
    interner: &mut FieldInterner,
) {
    let mut record = NodeRecord::new(label);
    let ts_field = interner.intern("created_at");
    record.set(ts_field, Value::Timestamp(timestamp_us));
    seed_node_record(engine, shard_id, NodeId::from_raw(node_id), &record);
}

/// Commit a built node record in its own MVCC transaction.
fn seed_node_record(engine: &StorageEngine, shard_id: u16, node_id: NodeId, record: &NodeRecord) {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    LocalNodeStore
        .put(&mut txn, shard_id, node_id, record)
        .expect("put node");
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).expect("commit node");
}

/// Read a node at the latest committed snapshot via an MVCC transaction.
fn read_node(engine: &StorageEngine, shard_id: u16, node_id: NodeId) -> Option<NodeRecord> {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::Transaction;
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    LocalNodeStore
        .get(&txn, shard_id, node_id)
        .expect("get node")
}

fn create_ttl_index_entry(
    engine: &StorageEngine,
    index: &IndexDefinition,
    node_id: u64,
    timestamp_us: i64,
) {
    let value = Value::Timestamp(timestamp_us);
    let key = coordinode_core::index::encoding::encode_index_key(&index.name, &value, node_id);
    engine.put(Partition::Idx, &key, &[]).expect("put index");
}

#[test]
fn reap_expired_nodes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let idx = IndexDefinition::btree("session_ttl", "Session", "created_at").with_ttl(3600); // 1 hour TTL

    let now_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);

    // Node 1: created 2 hours ago (expired)
    let two_hours_ago = now_us - 2 * 3600 * 1_000_000;
    insert_node_with_timestamp(&engine, 1, 1, "Session", two_hours_ago, &mut interner);
    create_ttl_index_entry(&engine, &idx, 1, two_hours_ago);

    // Node 2: created 30 minutes ago (NOT expired)
    let thirty_min_ago = now_us - 30 * 60 * 1_000_000;
    insert_node_with_timestamp(&engine, 1, 2, "Session", thirty_min_ago, &mut interner);
    create_ttl_index_entry(&engine, &idx, 2, thirty_min_ago);

    // Run reaper
    let result = reap_ttl_index(&engine, &idx, 1);
    assert_eq!(result.checked, 2);
    assert_eq!(result.deleted, 1); // Only node 1 expired

    // Verify node 1 was deleted
    assert!(read_node(&engine, 1, NodeId::from_raw(1)).is_none());

    // Verify node 2 still exists
    assert!(read_node(&engine, 1, NodeId::from_raw(2)).is_some());
}

#[test]
fn reap_no_ttl_returns_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    // Index without TTL
    let idx = IndexDefinition::btree("user_email", "User", "email");
    let result = reap_ttl_index(&engine, &idx, 1);
    assert_eq!(result.checked, 0);
    assert_eq!(result.deleted, 0);
}

#[test]
fn reap_cleans_orphaned_index_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx = IndexDefinition::btree("session_ttl", "Session", "created_at").with_ttl(3600);

    let old_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
        - 2 * 3600 * 1_000_000;

    // Create index entry WITHOUT a corresponding node (orphan)
    create_ttl_index_entry(&engine, &idx, 99, old_ts);

    let result = reap_ttl_index(&engine, &idx, 1);
    assert_eq!(result.checked, 1);
    assert_eq!(result.deleted, 1); // Orphaned entry cleaned up
}

#[test]
fn reap_all_ttl_indexes_runs_multiple() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx1 = IndexDefinition::btree("session_ttl", "Session", "created_at").with_ttl(3600);
    let idx2 = IndexDefinition::btree("user_email", "User", "email"); // Not TTL

    let results = reap_all_ttl_indexes(&engine, &[&idx1, &idx2], 1);
    // Only idx1 is TTL, so only 1 result
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "session_ttl");
}
