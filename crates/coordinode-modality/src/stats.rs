//! Statistics store — incremental planner-statistics counters in
//! [`Partition::Counter`].
//!
//! Label cardinalities and the total node count were previously derived by a
//! full scan + decode of the node partition on every statistics refresh.
//! This store maintains them incrementally instead: every node create /
//! delete / label change stages commutative counter deltas on the SAME
//! transaction as the data write, so the counters commit (and replicate,
//! and replay after a crash) atomically with the change they describe.
//!
//! Key shapes (all in the `counter:` partition, whose merge operator sums
//! i64 little-endian deltas):
//!
//! - `stat:nodes:total` — one row per stored node row (a temporal label
//!   counts one per VERSION, matching what a partition scan would count).
//! - `stat:label:<label>` — rows carrying `<label>`.
//!
//! Deltas are commutative merge operands: they carry no OCC surface, cannot
//! conflict, and fold at compaction.

use coordinode_storage::engine::transaction::Transaction;

// The key shapes live in coordinode-core (`graph::stats`) so the storage
// layer's statistics reader shares them; re-exported here for Layer-5
// callers that already import this module.
pub use coordinode_core::graph::stats::{
    LABEL_KEY_PREFIX, NODES_TOTAL_KEY, counter_delta_operand, label_count_key,
};

/// Layer 4 statistics store: stages planner-statistics counter deltas on the
/// transaction alongside the data writes they describe.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not maintain planner-statistics counters",
    label = "the statistics store is `LocalStatsStore`",
    note = "use `coordinode_modality::LocalStatsStore`, the CE implementation \
            backed by the counter partition's merge operator"
)]
pub trait StatsStore {
    /// Record a node row created with `labels`: total +1, each label +1.
    fn node_created<'a, I: IntoIterator<Item = &'a str>>(&self, txn: &mut Transaction, labels: I);

    /// Record a node row deleted that carried `labels`: total -1, each
    /// label -1.
    fn node_deleted<'a, I: IntoIterator<Item = &'a str>>(&self, txn: &mut Transaction, labels: I);

    /// Record one label added to an existing node row (`SET n:Label`).
    fn label_added(&self, txn: &mut Transaction, label: &str);

    /// Record one label removed from an existing node row (`REMOVE n:Label`).
    fn label_removed(&self, txn: &mut Transaction, label: &str);
}

/// CE statistics store over the counter partition's merge operator.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalStatsStore;

impl StatsStore for LocalStatsStore {
    fn node_created<'a, I: IntoIterator<Item = &'a str>>(&self, txn: &mut Transaction, labels: I) {
        txn.push_counter_delta(NODES_TOTAL_KEY, 1);
        for label in labels {
            txn.push_counter_delta(&label_count_key(label), 1);
        }
    }

    fn node_deleted<'a, I: IntoIterator<Item = &'a str>>(&self, txn: &mut Transaction, labels: I) {
        txn.push_counter_delta(NODES_TOTAL_KEY, -1);
        for label in labels {
            txn.push_counter_delta(&label_count_key(label), -1);
        }
    }

    fn label_added(&self, txn: &mut Transaction, label: &str) {
        txn.push_counter_delta(&label_count_key(label), 1);
    }

    fn label_removed(&self, txn: &mut Transaction, label: &str) {
        txn.push_counter_delta(&label_count_key(label), -1);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
