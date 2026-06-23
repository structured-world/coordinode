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
mod tests;
