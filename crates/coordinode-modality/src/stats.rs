//! Statistics store: incremental planner-statistics counters in the storage
//! engine's counter partition.
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
//! Deltas are commutative merge operands: they carry no OCC surface and
//! cannot conflict. They fold only where the engine can prove what they
//! apply to, which above the last level it cannot, so between compactions
//! that reach it the chain on a key grows with the write volume and every
//! read of the counter resolves all of it.

use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use crate::error::{StoreError, StoreResult};

// The key shapes live in coordinode-core (`graph::stats`) so the storage
// layer's statistics reader shares them; re-exported here for Layer-5
// callers that already import this module.
pub use coordinode_core::graph::stats::{
    LABEL_KEY_PREFIX, NODES_TOTAL_KEY, counter_delta_operand, label_count_key, pending_count_key,
};
use coordinode_core::schema::label_id::LabelId;

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

    /// Record that change `operation`, while it is certified, will add
    /// `delta` to the count of label `label` once put in force: a carrier
    /// it gains placed (+1) or taken out (-1), one it retires placed (-1)
    /// or taken out (+1). The counter of the label itself keeps the facts in
    /// force until then.
    fn pending_moved(&self, txn: &mut Transaction, operation: u64, label: LabelId, delta: i64);

    /// What change `operation` adds to the count of `label` once put in
    /// force, as `txn` reads it.
    ///
    /// # Errors
    ///
    /// A storage failure or a counter that does not decode.
    fn pending_count(
        &self,
        txn: &mut Transaction,
        operation: u64,
        label: LabelId,
    ) -> StoreResult<i64>;

    /// Settle what change `operation` left pending for `label` (named
    /// `name`) once it ended: added to the label's count when it was put in
    /// force (`in_force`), dropped either way. Nothing adds to it after the
    /// change ended, as every writer bound to the change while it was
    /// certified is refused at its commit after.
    ///
    /// # Errors
    ///
    /// As [`Self::pending_count`].
    fn settle_pending(
        &self,
        txn: &mut Transaction,
        change: (u64, LabelId, &str),
        in_force: bool,
    ) -> StoreResult<()>;
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

    fn pending_moved(&self, txn: &mut Transaction, operation: u64, label: LabelId, delta: i64) {
        if delta != 0 {
            txn.push_counter_delta(&pending_count_key(operation, label.get()), delta);
        }
    }

    fn pending_count(
        &self,
        txn: &mut Transaction,
        operation: u64,
        label: LabelId,
    ) -> StoreResult<i64> {
        let key = pending_count_key(operation, label.get());
        let stored = match txn.get(Partition::Counter, &key)? {
            Some(bytes) => {
                coordinode_storage::engine::merge::decode_counter(&bytes).map_err(|e| {
                    StoreError::Decode {
                        kind: "pending count",
                        message: format!("label {label} under change {operation}: {e}"),
                    }
                })?
            }
            None => 0,
        };
        stored
            .checked_add(txn.pending_counter_delta(&key))
            .ok_or_else(|| {
                StoreError::Invariant(format!(
                    "the pending count of label {label} under change {operation} is out of range"
                ))
            })
    }

    fn settle_pending(
        &self,
        txn: &mut Transaction,
        (operation, label, name): (u64, LabelId, &str),
        in_force: bool,
    ) -> StoreResult<()> {
        let pending = self.pending_count(txn, operation, label)?;
        if in_force && pending != 0 {
            txn.push_counter_delta(&label_count_key(name), pending);
        }
        txn.delete(
            Partition::Counter,
            &pending_count_key(operation, label.get()),
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
