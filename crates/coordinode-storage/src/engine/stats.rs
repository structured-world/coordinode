//! Storage statistics computed from the CoordiNode storage engine.
//!
//! Provides real node counts, label cardinality, and edge fan-out
//! statistics for the query cost estimator, replacing hardcoded defaults.

use std::collections::HashMap;

use lsm_tree::Guard;

use coordinode_core::graph::edge::PostingList;
use coordinode_core::graph::node::NodeRecord;
use coordinode_core::graph::stats::StorageStats;

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::StorageResult;

/// Pre-computed storage statistics snapshot.
///
/// Computed by scanning the `node:` and `adj:` partitions. The caller
/// is responsible for caching and refreshing (e.g., on a timer or after
/// a configurable number of writes).
#[derive(Clone)]
pub struct StorageStatsComputer {
    total_nodes: u64,
    label_counts: HashMap<String, u64>,
    edge_type_fan_outs: HashMap<String, f64>,
    overall_avg_fan_out: f64,
    num_labels: u64,
}

/// Maximum number of adjacency entries to sample for fan-out estimation.
/// Sampling avoids full-scan cost on large databases.
const FAN_OUT_SAMPLE_LIMIT: usize = 1000;

impl StorageStatsComputer {
    /// Compute statistics by scanning raw (non-MVCC) storage.
    ///
    /// Use this when writing directly to StorageEngine (tests, bulk import).
    /// For MVCC-enabled databases (normal operation), use [`Self::compute_mvcc`].
    pub fn compute(engine: &StorageEngine) -> StorageResult<Self> {
        let (total_nodes, label_counts) = Self::count_nodes(engine)?;
        let (edge_type_fan_outs, overall_avg_fan_out) = Self::sample_fan_out(engine)?;
        let num_labels = label_counts.len() as u64;

        Ok(Self {
            total_nodes,
            label_counts,
            edge_type_fan_outs,
            overall_avg_fan_out,
            num_labels,
        })
    }

    /// Compute statistics from MVCC-versioned storage (ADR-016: native seqno snapshot).
    ///
    /// Uses a current snapshot for consistent reads across all partitions.
    /// This is the method used by Database and Server for EXPLAIN cost estimation.
    pub fn compute_mvcc(engine: &StorageEngine) -> StorageResult<Self> {
        let snapshot = engine.snapshot();

        let (total_nodes, label_counts) = Self::count_nodes_snapshot(engine, &snapshot)?;
        let (edge_type_fan_outs, overall_avg_fan_out) =
            Self::sample_fan_out_snapshot(engine, &snapshot)?;
        let num_labels = label_counts.len() as u64;

        Ok(Self {
            total_nodes,
            label_counts,
            edge_type_fan_outs,
            overall_avg_fan_out,
            num_labels,
        })
    }

    /// Count nodes per label by scanning the `node:` partition.
    fn count_nodes(engine: &StorageEngine) -> StorageResult<(u64, HashMap<String, u64>)> {
        let mut total: u64 = 0;
        let mut label_counts: HashMap<String, u64> = HashMap::new();

        let iter = engine.prefix_scan(Partition::Node, b"node:")?;
        for guard in iter {
            let Ok((_key, value)) = guard.into_inner() else {
                continue;
            };
            total += 1;

            // Decode NodeRecord to extract labels
            if let Ok(record) = NodeRecord::from_msgpack(&value) {
                for label in &record.labels {
                    *label_counts.entry(label.clone()).or_insert(0) += 1;
                }
            }
        }

        Ok((total, label_counts))
    }

    /// Sample adjacency posting lists to estimate average fan-out per edge type.
    ///
    /// Only scans outgoing (`adj:*:out:*`) keys to avoid double-counting.
    /// Samples up to `FAN_OUT_SAMPLE_LIMIT` entries for efficiency.
    fn sample_fan_out(engine: &StorageEngine) -> StorageResult<(HashMap<String, f64>, f64)> {
        let mut type_total_edges: HashMap<String, u64> = HashMap::new();
        let mut type_entry_count: HashMap<String, u64> = HashMap::new();
        let mut global_total_edges: u64 = 0;
        let mut global_entry_count: u64 = 0;

        let iter = engine.prefix_scan(Partition::Adj, b"adj:")?;
        let mut sampled = 0;

        for guard in iter {
            if sampled >= FAN_OUT_SAMPLE_LIMIT {
                break;
            }

            let Ok((key, value)) = guard.into_inner() else {
                continue;
            };

            // Parse key: adj:<TYPE>:out:<id> or adj:<TYPE>:in:<id>
            // Only count outgoing to avoid double-counting
            let key_bytes: &[u8] = &key;
            let key_str = match std::str::from_utf8(key_bytes) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Skip reverse (incoming) keys
            if !key_str.contains(":out:") {
                continue;
            }

            // Extract edge type: between first "adj:" and ":out:"
            let after_adj = match key_str.strip_prefix("adj:") {
                Some(rest) => rest,
                None => continue,
            };
            let edge_type = match after_adj.find(":out:") {
                Some(pos) => &after_adj[..pos],
                None => continue,
            };

            // Count UIDs in posting list
            let uid_count = match PostingList::from_bytes(&value) {
                Ok(pl) => pl.len() as u64,
                Err(_) => continue,
            };

            *type_total_edges.entry(edge_type.to_string()).or_insert(0) += uid_count;
            *type_entry_count.entry(edge_type.to_string()).or_insert(0) += 1;
            global_total_edges += uid_count;
            global_entry_count += 1;
            sampled += 1;
        }

        // Compute per-type averages
        let mut type_fan_outs = HashMap::new();
        for (edge_type, total) in &type_total_edges {
            let count = type_entry_count[edge_type];
            if count > 0 {
                type_fan_outs.insert(edge_type.clone(), *total as f64 / count as f64);
            }
        }

        let overall = if global_entry_count > 0 {
            global_total_edges as f64 / global_entry_count as f64
        } else {
            0.0
        };

        Ok((type_fan_outs, overall))
    }

    /// Count nodes per label using snapshot-based scan (ADR-016: native seqno MVCC).
    fn count_nodes_snapshot(
        engine: &StorageEngine,
        snapshot: &lsm_tree::SeqNo,
    ) -> StorageResult<(u64, HashMap<String, u64>)> {
        let mut total: u64 = 0;
        let mut label_counts: HashMap<String, u64> = HashMap::new();

        let entries = engine.snapshot_prefix_scan(snapshot, Partition::Node, b"node:")?;
        for (_key, value) in entries {
            total += 1;
            if let Ok(record) = NodeRecord::from_msgpack(&value) {
                for label in &record.labels {
                    *label_counts.entry(label.clone()).or_insert(0) += 1;
                }
            }
        }

        Ok((total, label_counts))
    }

    /// Sample fan-out from snapshot-based adjacency data (ADR-016).
    fn sample_fan_out_snapshot(
        engine: &StorageEngine,
        snapshot: &lsm_tree::SeqNo,
    ) -> StorageResult<(HashMap<String, f64>, f64)> {
        let mut type_total_edges: HashMap<String, u64> = HashMap::new();
        let mut type_entry_count: HashMap<String, u64> = HashMap::new();
        let mut global_total_edges: u64 = 0;
        let mut global_entry_count: u64 = 0;

        let entries = engine.snapshot_prefix_scan(snapshot, Partition::Adj, b"adj:")?;
        let mut sampled = 0;

        for (key, value) in entries {
            if sampled >= FAN_OUT_SAMPLE_LIMIT {
                break;
            }

            let key_str = match std::str::from_utf8(&key) {
                Ok(s) => s,
                Err(_) => continue,
            };

            if !key_str.contains(":out:") {
                continue;
            }

            let after_adj = match key_str.strip_prefix("adj:") {
                Some(rest) => rest,
                None => continue,
            };
            let edge_type = match after_adj.find(":out:") {
                Some(pos) => &after_adj[..pos],
                None => continue,
            };

            let uid_count = match PostingList::from_bytes(&value) {
                Ok(pl) => pl.len() as u64,
                Err(_) => continue,
            };

            *type_total_edges.entry(edge_type.to_string()).or_insert(0) += uid_count;
            *type_entry_count.entry(edge_type.to_string()).or_insert(0) += 1;
            global_total_edges += uid_count;
            global_entry_count += 1;
            sampled += 1;
        }

        let mut type_fan_outs = HashMap::new();
        for (edge_type, total) in &type_total_edges {
            let count = type_entry_count[edge_type];
            if count > 0 {
                type_fan_outs.insert(edge_type.clone(), *total as f64 / count as f64);
            }
        }

        let overall = if global_entry_count > 0 {
            global_total_edges as f64 / global_entry_count as f64
        } else {
            0.0
        };

        Ok((type_fan_outs, overall))
    }
}

impl StorageStats for StorageStatsComputer {
    fn total_node_count(&self) -> u64 {
        self.total_nodes
    }

    fn node_count_for_label(&self, label: &str) -> Option<u64> {
        self.label_counts.get(label).copied()
    }

    fn avg_fan_out_for_type(&self, edge_type: &str) -> Option<f64> {
        self.edge_type_fan_outs.get(edge_type).copied()
    }

    fn avg_fan_out(&self) -> f64 {
        self.overall_avg_fan_out
    }

    fn label_count(&self) -> u64 {
        self.num_labels
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
