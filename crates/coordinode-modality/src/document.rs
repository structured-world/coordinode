//! Document store — path-targeted partial updates on DOCUMENT
//! properties of [`NodeRecord`] via the [`DocumentMerge`] operator
//! (ADR-015).
//!
//! Document-typed properties (Mongo-like nested maps + arrays) are
//! mutated through commutative [`DocDelta`] operands rather than
//! read-modify-write. Each [`DocDelta`] is encoded with the
//! `PREFIX_DOC_DELTA` byte and buffered on the transaction; the
//! merge function replays operands in seqno order against the base
//! `NodeRecord` during reads and compaction.
//!
//! ## Transaction threading (ADR-041)
//!
//! Every method takes an explicit `&mut Transaction`. The encoded
//! [`DocDelta`] operand is buffered via
//! [`Transaction::push_node_delta`] and applied at
//! [`Transaction::commit`] through the node-partition merge path, so a
//! set of document mutations lands atomically (or not at all). Reads in
//! the same transaction surface the pending operands via
//! [`NodeStore::get_raw_tracked`](crate::NodeStore::get_raw_tracked),
//! which materialises buffered deltas before the OCC-tracked read.
//!
//! ## What this store exposes
//!
//! Every method is a typed wrapper around
//! [`Transaction::push_node_delta`] that hides:
//!
//! - Operand framing (the `PREFIX_DOC_DELTA` prefix byte).
//! - MessagePack encoding of the [`DocDelta`] enum.
//! - The node-key shape (`encode_node_key(shard, id)`).
//!
//! The caller chooses [`PathTarget`] (Extra map vs. interned
//! PropField) and supplies the path + value. The store does not
//! materialise the post-merge `NodeRecord` for inspection — for that,
//! callers use [`NodeStore::get`](crate::NodeStore::get), which
//! triggers the merge transparently.
//!
//! ## Read side
//!
//! Reading a node that has accumulated [`DocDelta`] operands goes
//! through the existing [`NodeStore`](crate::NodeStore) — the merge
//! operator collapses the delta history during `get`. This store is
//! write-only on purpose: no read API exists at the doc level, only
//! at the node level. That matches how the merge operator works (one
//! materialisation point per read).
//!
//! [`DocumentMerge`]: coordinode_storage::engine::merge::DocumentMerge
//! [`NodeRecord`]: coordinode_core::graph::node::NodeRecord

use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};
use coordinode_core::graph::node::{encode_node_key, NodeId};
use coordinode_storage::engine::transaction::Transaction;

use crate::error::{StoreError, StoreResult};

/// Layer 4 document store: typed buffering of [`DocDelta`] operands
/// against DOCUMENT-typed properties of a node. Every method buffers a
/// merge operand on the supplied [`Transaction`]; the operands apply
/// atomically at [`Transaction::commit`].
pub trait DocumentStore {
    /// Set a value at a dotted path on the node's document property.
    /// Intermediate maps are created as needed.
    fn set_path(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Delete a value at a dotted path. Idempotent on missing path.
    fn delete_path(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
    ) -> StoreResult<()>;

    /// Append a value to an array at path. Creates the array if
    /// missing.
    fn array_push(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Remove the first occurrence of a value from an array at path.
    /// Idempotent.
    fn array_pull(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Add value to array only if not already present.
    fn array_add_to_set(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()>;

    /// Numeric increment at path (commutative).
    fn increment(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        amount: f64,
    ) -> StoreResult<()>;

    /// Remove a top-level property from the node's record. For
    /// `PathTarget::Extra` the caller supplies the key; for
    /// `PathTarget::PropField(_)` the field id in the target is used.
    fn remove_property(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        key: Option<String>,
    ) -> StoreResult<()>;
}

/// CE single-shard implementation of [`DocumentStore`]. Stateless — all
/// storage access flows through the [`Transaction`] passed to each
/// method (ADR-041).
pub struct LocalDocumentStore;

impl LocalDocumentStore {
    /// Encode `delta` and buffer it as a node merge operand on `txn`.
    /// The operand carries the `PREFIX_DOC_DELTA` framing byte and is
    /// applied at commit via the node-partition merge path.
    fn write_delta(
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        delta: DocDelta,
    ) -> StoreResult<()> {
        let operand = delta.encode().map_err(|e| StoreError::Decode {
            kind: "doc delta",
            message: format!("encode: {e}"),
        })?;
        txn.push_node_delta(encode_node_key(shard_id, node_id), operand);
        Ok(())
    }
}

impl DocumentStore for LocalDocumentStore {
    fn set_path(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
            shard_id,
            node_id,
            DocDelta::SetPath {
                target,
                path,
                value,
            },
        )
    }

    fn delete_path(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
            shard_id,
            node_id,
            DocDelta::DeletePath { target, path },
        )
    }

    fn array_push(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
            shard_id,
            node_id,
            DocDelta::ArrayPush {
                target,
                path,
                value,
            },
        )
    }

    fn array_pull(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
            shard_id,
            node_id,
            DocDelta::ArrayPull {
                target,
                path,
                value,
            },
        )
    }

    fn array_add_to_set(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        value: rmpv::Value,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
            shard_id,
            node_id,
            DocDelta::ArrayAddToSet {
                target,
                path,
                value,
            },
        )
    }

    fn increment(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        path: Vec<String>,
        amount: f64,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
            shard_id,
            node_id,
            DocDelta::Increment {
                target,
                path,
                amount,
            },
        )
    }

    fn remove_property(
        &self,
        txn: &mut Transaction,
        shard_id: u16,
        node_id: NodeId,
        target: PathTarget,
        key: Option<String>,
    ) -> StoreResult<()> {
        Self::write_delta(
            txn,
            shard_id,
            node_id,
            DocDelta::RemoveProperty { target, key },
        )
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
