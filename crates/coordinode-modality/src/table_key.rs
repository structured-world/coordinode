//! Table key store: the unique index from a relational table's declared key
//! to the NodeId of the row that holds it.
//!
//! Every operation threads the statement [`Transaction`], so the entry is
//! written atomically with its row, a statement reads its own earlier claims,
//! and two transactions claiming one key conflict at commit instead of both
//! succeeding.
//!
//! An entry is `tkey:<table length u32 BE><table><key>` → NodeId (u64 BE) in
//! [`Partition::Idx`]. The key encoding is injective, so two distinct keys
//! never share an entry, and it preserves order within a table, so a range of
//! keys is a range of entries.

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{Unindexable, encode_element};
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use crate::error::{StoreError, StoreResult};

const PREFIX: &[u8] = b"tkey:";

/// Layer 4 store for the unique index of a table's declared key.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not store table keys",
    label = "a table key store is required here",
    note = "use `LocalTableKeyStore`, the CE implementation over a statement transaction"
)]
pub trait TableKeyStore {
    /// The row holding `key` in `table`, as the transaction sees it (its own
    /// earlier claims included).
    ///
    /// # Errors
    ///
    /// A key value of a type that cannot be a key, or a storage failure.
    fn lookup(
        &self,
        txn: &mut Transaction,
        table: &str,
        key: &[Value],
    ) -> StoreResult<Option<NodeId>>;

    /// Record that `node_id` holds `key` in `table`. The caller has checked
    /// the key is free; a concurrent claim of the same key fails one of the
    /// two transactions at commit.
    ///
    /// # Errors
    ///
    /// As [`TableKeyStore::lookup`].
    fn claim(
        &self,
        txn: &mut Transaction,
        table: &str,
        key: &[Value],
        node_id: NodeId,
    ) -> StoreResult<()>;

    /// Free `key` in `table` (its row is being deleted).
    ///
    /// # Errors
    ///
    /// As [`TableKeyStore::lookup`].
    fn release(&self, txn: &mut Transaction, table: &str, key: &[Value]) -> StoreResult<()>;

    /// Free every key of `table` (the table is being dropped).
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn release_all(&self, txn: &mut Transaction, table: &str) -> StoreResult<()>;

    /// The mutation that frees `key` in `table`, for a writer that deletes
    /// rows by submitting mutations directly rather than through a
    /// transaction (the TTL reaper).
    ///
    /// # Errors
    ///
    /// A key value of a type that cannot be a key.
    fn release_mutation(&self, table: &str, key: &[Value]) -> StoreResult<Mutation>;

    /// The row holding `key` in `table` in the latest committed state, outside
    /// any transaction. Tells a transaction that lost a race for a key which
    /// row won it.
    ///
    /// # Errors
    ///
    /// As [`TableKeyStore::lookup`].
    fn committed_holder(
        &self,
        engine: &StorageEngine,
        table: &str,
        key: &[Value],
    ) -> StoreResult<Option<NodeId>>;
}

/// CE implementation of [`TableKeyStore`].
///
/// # Examples
///
/// ```no_run
/// use coordinode_modality::{LocalTableKeyStore, TableKeyStore};
/// use coordinode_core::graph::{node::NodeId, types::Value};
/// # use coordinode_storage::engine::{config::*, core::StorageEngine, transaction::Transaction};
/// # use coordinode_core::txn::timestamp::Timestamp;
/// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
/// #     "ep", std::path::Path::new("/tmp/x"),
/// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
/// # let engine = StorageEngine::open(&cfg)?;
/// # let mut txn = Transaction::begin(&engine, None, Timestamp::from_raw(1));
/// let key = [Value::Int(1001)];
/// let store = LocalTableKeyStore;
/// assert_eq!(store.lookup(&mut txn, "Trade", &key)?, None);
/// store.claim(&mut txn, "Trade", &key, NodeId::from_raw(7))?;
/// assert_eq!(store.lookup(&mut txn, "Trade", &key)?, Some(NodeId::from_raw(7)));
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalTableKeyStore;

impl TableKeyStore for LocalTableKeyStore {
    fn lookup(
        &self,
        txn: &mut Transaction,
        table: &str,
        key: &[Value],
    ) -> StoreResult<Option<NodeId>> {
        let entry = entry_key(table, key)?;
        txn.get(Partition::Idx, &entry)?
            .as_deref()
            .map(decode_holder)
            .transpose()
    }

    fn claim(
        &self,
        txn: &mut Transaction,
        table: &str,
        key: &[Value],
        node_id: NodeId,
    ) -> StoreResult<()> {
        let entry = entry_key(table, key)?;
        txn.put(Partition::Idx, &entry, &node_id.as_raw().to_be_bytes())?;
        Ok(())
    }

    fn release(&self, txn: &mut Transaction, table: &str, key: &[Value]) -> StoreResult<()> {
        let entry = entry_key(table, key)?;
        txn.delete(Partition::Idx, &entry)?;
        Ok(())
    }

    fn release_all(&self, txn: &mut Transaction, table: &str) -> StoreResult<()> {
        let prefix = table_prefix(table)?;
        for (key, _) in txn.prefix_scan(Partition::Idx, &prefix)? {
            txn.delete(Partition::Idx, &key)?;
        }
        Ok(())
    }

    fn release_mutation(&self, table: &str, key: &[Value]) -> StoreResult<Mutation> {
        Ok(Mutation::Delete {
            partition: PartitionId::Idx,
            key: entry_key(table, key)?,
        })
    }

    fn committed_holder(
        &self,
        engine: &StorageEngine,
        table: &str,
        key: &[Value],
    ) -> StoreResult<Option<NodeId>> {
        let entry = entry_key(table, key)?;
        engine
            .get(Partition::Idx, &entry)?
            .as_deref()
            .map(decode_holder)
            .transpose()
    }
}

fn decode_holder(bytes: &[u8]) -> StoreResult<NodeId> {
    let raw = <[u8; 8]>::try_from(bytes).map_err(|_| StoreError::Decode {
        kind: "table key entry",
        message: format!("{} bytes, expected 8", bytes.len()),
    })?;
    Ok(NodeId::from_raw(u64::from_be_bytes(raw)))
}

fn table_prefix(table: &str) -> StoreResult<Vec<u8>> {
    let len = u32::try_from(table.len())
        .map_err(|_| StoreError::Invariant(format!("table name of {} bytes", table.len())))?;
    let mut out = Vec::with_capacity(PREFIX.len() + 4 + table.len());
    out.extend_from_slice(PREFIX);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(table.as_bytes());
    Ok(out)
}

/// The table prefix followed by the key's tuple encoding (injective and
/// order-preserving, see [`coordinode_core::index::encoding`]). A key value
/// cannot be NULL: a row is found by its key, and NULL equals nothing.
fn entry_key(table: &str, key: &[Value]) -> StoreResult<Vec<u8>> {
    let mut out = table_prefix(table)?;
    for value in key {
        if value.is_null() {
            return Err(StoreError::UnsupportedKey("NULL"));
        }
        encode_element(value, &mut out).map_err(unindexable)?;
    }
    Ok(out)
}

pub(crate) fn unindexable(what: Unindexable) -> StoreError {
    StoreError::UnsupportedKey(match what {
        Unindexable::NaN => "NaN",
        Unindexable::Kind(kind) => kind,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
