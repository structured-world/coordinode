//! Storage statistics computed from the CoordiNode storage engine.
//!
//! Provides real node counts, label cardinality, and edge fan-out
//! statistics for the query cost estimator, replacing hardcoded defaults.
//! Node and label cardinalities are read from the incrementally-maintained
//! counters in the counter partition (staged by the write executors on the
//! same transaction as the data writes); fan-out remains a bounded sample
//! over the adjacency partition. A refresh therefore costs a handful of
//! counter reads plus the sample, not a node-partition scan.
//!
//! Damaged bytes are reported, never read as a plausible number: a counter
//! read as zero or a list left out of the sample would price plans against a
//! graph that is not the one on disk, and nothing downstream could tell.

use std::collections::HashMap;

use lsm_tree::Guard;

use coordinode_core::graph::edge::{AdjDirection, PostingList, decode_adj_key};
use coordinode_core::graph::stats::{LABEL_KEY_PREFIX, NODES_TOTAL_KEY, StorageStats};

use crate::engine::core::StorageEngine;
use crate::engine::merge;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// Pre-computed storage statistics snapshot.
///
/// Node/label cardinalities come from the counter partition's incremental
/// statistics counters; fan-out from a bounded adjacency sample. The caller
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

/// Maximum number of forward adjacency lists sampled per edge type.
/// Sampling avoids full-scan cost on large databases; the cap is per type so
/// that every type gets an estimate, not only the first ones in key order.
const FAN_OUT_SAMPLE_PER_TYPE: u64 = 1000;

/// Key prefix every adjacency key starts with (see `encode_adj_key_forward`).
const ADJ_KEY_PREFIX: &[u8] = b"adj:";

/// Inclusive upper bound for a range over [`ADJ_KEY_PREFIX`]: the prefix with
/// its last byte incremented (`:` + 1 = `;`) sorts after every key under it.
const ADJ_KEY_RANGE_END: &[u8] = b"adj;";

impl StorageStatsComputer {
    /// Compute statistics at a complete snapshot of the engine.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Serialization`] naming the key when a counter,
    /// an adjacency key or a posting list does not decode, and any error the
    /// reads themselves return.
    pub fn compute(engine: &StorageEngine) -> StorageResult<Self> {
        // Pinned so the watermark cannot pass the snapshot mid-walk.
        let (snapshot, _pin) = engine.pin_latest_snapshot();

        // A counter below zero (drift from a doubled decrement) is well formed
        // but counts nothing; the planner reads it as none.
        let total_nodes =
            match engine.snapshot_get(&snapshot, Partition::Counter, NODES_TOTAL_KEY)? {
                Some(value) => u64::try_from(decode_counter(NODES_TOTAL_KEY, &value)?).unwrap_or(0),
                None => 0,
            };

        let mut label_counts: HashMap<String, u64> = HashMap::new();
        for guard in engine.snapshot_prefix_iter(&snapshot, Partition::Counter, LABEL_KEY_PREFIX)? {
            let (key, value) = guard.into_inner()?;
            // Zero or below means every row with the label is gone, and a
            // label with no rows is absent, as a scan of the rows reports it.
            let Ok(count @ 1..) = u64::try_from(decode_counter(&key, &value)?) else {
                continue;
            };
            let label = std::str::from_utf8(&key[LABEL_KEY_PREFIX.len()..])
                .map_err(|_| corrupt(Partition::Counter, &key, "label name is not UTF-8"))?;
            label_counts.insert(label.to_owned(), count);
        }

        let (edge_type_fan_outs, overall_avg_fan_out) = sample_fan_out(engine, snapshot)?;
        let num_labels = label_counts.len() as u64;

        Ok(Self {
            total_nodes,
            label_counts,
            edge_type_fan_outs,
            overall_avg_fan_out,
            num_labels,
        })
    }
}

/// Decode a statistics counter, reporting a value of the wrong width.
fn decode_counter(key: &[u8], value: &[u8]) -> StorageResult<i64> {
    merge::decode_counter(value).map_err(|_| {
        corrupt(
            Partition::Counter,
            key,
            &format!("counter holds {} bytes, expected 8", value.len()),
        )
    })
}

/// A value or key that no writer of this partition produces.
fn corrupt(part: Partition, key: &[u8], what: &str) -> StorageError {
    StorageError::Serialization(format!("{part:?} key {}: {what}", key.escape_ascii()))
}

/// Sample forward posting lists to estimate the average fan-out per edge type.
///
/// Reverse lists would count every edge twice, and for each type they all sort
/// before the forward ones (`in` < `out`), so the walk seeks past them rather
/// than reading each one; a type that reaches its sample cap is skipped the
/// same way. The cost is therefore the sampled lists plus a seek per type.
fn sample_fan_out(
    engine: &StorageEngine,
    snapshot: lsm_tree::SeqNo,
) -> StorageResult<(HashMap<String, f64>, f64)> {
    // Per type: (edges in the sampled lists, lists sampled).
    let mut per_type: HashMap<String, (u64, u64)> = HashMap::new();
    let mut iter =
        engine.range_seekable(Partition::Adj, ADJ_KEY_PREFIX, ADJ_KEY_RANGE_END, snapshot)?;

    while let Some(guard) = iter.next() {
        let (key, value) = guard.into_inner()?;
        if !key.starts_with(ADJ_KEY_PREFIX) {
            // The inclusive bound itself; nothing under the prefix is left.
            break;
        }
        let parts = decode_adj_key(&key)
            .ok_or_else(|| corrupt(Partition::Adj, &key, "not an adjacency key"))?;
        if matches!(parts.direction, AdjDirection::In) {
            iter.seek_to(&type_key(&parts.edge_type, ":out:"));
            continue;
        }
        let list = PostingList::from_bytes(&value)
            .map_err(|e| corrupt(Partition::Adj, &key, &format!("not a posting list: {e}")))?;
        let edges = list.len() as u64;
        let sampled = match per_type.get_mut(&parts.edge_type) {
            Some(entry) => {
                entry.0 += edges;
                entry.1 += 1;
                entry.1
            }
            None => {
                per_type.insert(parts.edge_type.clone(), (edges, 1));
                1
            }
        };
        if sampled >= FAN_OUT_SAMPLE_PER_TYPE {
            // `;` follows `:`, so this sorts after every forward list of the type.
            iter.seek_to(&type_key(&parts.edge_type, ":out;"));
        }
    }

    let (mut edges, mut lists) = (0u64, 0u64);
    let mut type_fan_outs = HashMap::with_capacity(per_type.len());
    for (edge_type, (type_edges, type_lists)) in per_type {
        edges += type_edges;
        lists += type_lists;
        // Every entry holds at least the list that created it.
        type_fan_outs.insert(edge_type, type_edges as f64 / type_lists as f64);
    }
    let overall = if lists > 0 {
        edges as f64 / lists as f64
    } else {
        0.0
    };
    Ok((type_fan_outs, overall))
}

/// `adj:<edge_type><suffix>`: a seek target inside one type's key range.
fn type_key(edge_type: &str, suffix: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(ADJ_KEY_PREFIX.len() + edge_type.len() + suffix.len());
    key.extend_from_slice(ADJ_KEY_PREFIX);
    key.extend_from_slice(edge_type.as_bytes());
    key.extend_from_slice(suffix.as_bytes());
    key
}

/// Recompute the planner's node and label counters from the node rows.
///
/// The counters are staged by the write executor on the transaction that
/// writes the node, so a path that writes node rows directly leaves them
/// untouched: a bulk restore fills the graph and the counters keep saying
/// what they said before, which for a fresh database is nothing. The cost
/// estimator then prices every plan against an empty graph. This reads the
/// rows that are actually there and writes the counters to match.
///
/// Written with `put` rather than with deltas because this establishes the
/// value rather than moving it; it is a rebuild from the truth on disk, not
/// an increment, so it must not be composed with what was there before.
///
/// One row counts once, so a temporal node counts once per stored version,
/// which is what the counters mean and what a scan of the partition sees.
///
/// # Errors
///
/// A row that is not a node record is reported and no counter is written: a
/// total that left it out would durably claim a smaller graph than the one
/// on disk.
pub fn rebuild_node_counters(engine: &StorageEngine) -> StorageResult<()> {
    use coordinode_core::graph::node::NODE_KEY_PREFIX;
    use coordinode_core::graph::stats::label_count_key;

    use crate::engine::merge::decode_node_record;

    let mut total: i64 = 0;
    let mut label_counts: HashMap<String, i64> = HashMap::new();

    for guard in engine.prefix_scan(Partition::Node, NODE_KEY_PREFIX)? {
        let (key, value) = guard.into_inner()?;
        let record = decode_node_record(&value)
            .map_err(|_| corrupt(Partition::Node, &key, "not a node record"))?;
        total += 1;
        for label in &record.labels {
            *label_counts.entry(label.clone()).or_insert(0) += 1;
        }
    }

    engine.put(Partition::Counter, NODES_TOTAL_KEY, &total.to_le_bytes())?;
    for (label, count) in label_counts {
        engine.put(
            Partition::Counter,
            &label_count_key(&label),
            &count.to_le_bytes(),
        )?;
    }

    Ok(())
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
