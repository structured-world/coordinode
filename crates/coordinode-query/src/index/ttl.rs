//! TTL (Time-To-Live) index reaper: deletes expired nodes.
//!
//! A TTL index is a B-tree index on a Timestamp property with an
//! `expire_after` duration. The reaper periodically scans TTL indexes
//! and deletes nodes where `timestamp + ttl < now()`.

use std::time::{SystemTime, UNIX_EPOCH};

use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_modality::{IndexStore as _, LocalIndexStore, LocalNodeStore, NodeStore};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;

use super::definition::IndexDefinition;

/// Result of a TTL reap pass.
#[derive(Debug, Default)]
pub struct ReapResult {
    /// Number of nodes checked.
    pub checked: usize,
    /// Number of expired nodes deleted.
    pub deleted: usize,
    /// Errors encountered (non-fatal, processing continues).
    pub errors: Vec<String>,
}

/// Run a single TTL reap pass for one index.
///
/// Scans the index for all entries, checks if the indexed timestamp
/// value has expired (timestamp_us + ttl_seconds * 1_000_000 < now_us),
/// and deletes expired nodes.
pub fn reap_ttl_index(
    engine: &StorageEngine,
    index: &IndexDefinition,
    shard_id: u16,
) -> ReapResult {
    let mut result = ReapResult::default();

    let ttl_seconds = match index.ttl_seconds {
        Some(ttl) => ttl,
        None => return result, // Not a TTL index
    };

    let ttl_us = ttl_seconds as i64 * 1_000_000;
    let now_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);

    let cutoff_us = now_us - ttl_us;
    let nodes = LocalNodeStore;
    // TTL reaping is a bulk background sweep — cheap direct-mode transaction
    // (no snapshot/OCC): reads latest, node deletes apply immediately to the
    // engine, like the direct deletes this replaces.
    let mut reaper_txn = Transaction::new(engine, None, Timestamp::ZERO, None);

    // Full-index walk through the Layer-4 store — returns (raw key, node id)
    // pairs with the node id already decoded from the entry suffix.
    let index_store = LocalIndexStore::new(engine);
    let entries = match index_store.scan_all(&index.name) {
        Ok(entries) => entries,
        Err(e) => {
            result.errors.push(format!("scan error: {e}"));
            return result;
        }
    };

    // Collect expired node IDs
    let mut expired_nodes: Vec<(u64, Vec<u8>)> = Vec::new(); // (node_id, index_key)

    for (key, node) in entries {
        result.checked += 1;
        // Read the node to check its timestamp value
        match nodes.get(&reaper_txn, shard_id, node) {
            Ok(Some(record)) => {
                // Check if the timestamp property has expired
                if is_node_expired(&record, index.property(), cutoff_us) {
                    expired_nodes.push((node.as_raw(), key));
                }
            }
            Ok(None) => {
                // Node already deleted — clean up orphaned index entry
                expired_nodes.push((node.as_raw(), key));
            }
            Err(e) => {
                result.errors.push(format!("node read error: {e}"));
            }
        }
    }

    // Delete expired nodes and their index entries
    for (node_id, index_key) in &expired_nodes {
        // Delete the index entry
        if let Err(e) = index_store.delete_raw(index_key) {
            result
                .errors
                .push(format!("index delete error for node {node_id}: {e}"));
            continue;
        }

        // Delete the node record (buffered on the reaper transaction).
        if let Err(e) = nodes.delete(&mut reaper_txn, shard_id, NodeId::from_raw(*node_id)) {
            result
                .errors
                .push(format!("node delete error for node {node_id}: {e}"));
            continue;
        }

        result.deleted += 1;
    }

    result
}

/// Check if a node's timestamp property has expired.
fn is_node_expired(record: &NodeRecord, _property: &str, cutoff_us: i64) -> bool {
    // Check all timestamp properties — the TTL applies to the indexed field.
    // Without the interner, we check all Timestamp values (the indexed one
    // will be among them).
    for value in record.props.values() {
        if let Value::Timestamp(ts) = value {
            if *ts < cutoff_us {
                return true;
            }
        }
    }
    false
}

/// Run reap passes for all TTL indexes in the registry.
pub fn reap_all_ttl_indexes(
    engine: &StorageEngine,
    indexes: &[&IndexDefinition],
    shard_id: u16,
) -> Vec<(String, ReapResult)> {
    indexes
        .iter()
        .filter(|idx| idx.ttl_seconds.is_some())
        .map(|idx| {
            let result = reap_ttl_index(engine, idx, shard_id);
            (idx.name.clone(), result)
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
// Tests plant raw index fixtures via the storage partition.
#[allow(clippy::disallowed_types)]
mod tests {
    use super::*;
    use coordinode_core::graph::intern::FieldInterner;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
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
    fn seed_node_record(
        engine: &StorageEngine,
        shard_id: u16,
        node_id: NodeId,
        record: &NodeRecord,
    ) {
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
}
