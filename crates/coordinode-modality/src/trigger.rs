//! Trigger store — DDL state for triggers, living in [`Partition::Schema`]
//! alongside label/edge-type schemas.
//!
//! Two key families, both under `schema:trigger`:
//!
//! - **Definitions** (`schema:trigger:<name>`): one serialized
//!   [`TriggerSchema`](coordinode_core::schema::triggers::TriggerSchema) per
//!   trigger.
//! - **Index** (`schema:trigger_index:<target>:<event>`): a `Vec<String>` of
//!   trigger names registered for a `(target, event)` pair, so the mutation
//!   path can look up matching triggers without scanning every definition.
//!
//! This store owns ONLY the key encoding and the `Partition::Schema` binding;
//! it hands raw bytes back and forth so the query layer keeps the value codec
//! (`rmp_serde`) and its diagnostic error messages (ADR-041 raw-bytes pattern).

use coordinode_core::schema::triggers::{
    encode_trigger_index_key, encode_trigger_key, trigger_scan_prefix,
};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::{KvPair, Transaction};

use crate::error::StoreResult;

/// Layer 4 trigger store over a [`Transaction`]. All reads are OCC-tracked
/// (a trigger a mutation consults must be conflict-checked); writes buffer for
/// atomic commit.
pub trait TriggerStore {
    /// Raw definition bytes for `name`, or `None` if no such trigger.
    fn get_definition(&self, txn: &mut Transaction, name: &str) -> StoreResult<Option<Vec<u8>>>;

    /// Buffer a trigger definition write (caller-encoded bytes).
    fn put_definition(&self, txn: &mut Transaction, name: &str, bytes: &[u8]) -> StoreResult<()>;

    /// Tombstone a trigger definition. Idempotent on a missing trigger.
    fn delete_definition(&self, txn: &mut Transaction, name: &str) -> StoreResult<()>;

    /// Raw index bytes (encoded `Vec<String>` of names) for a `(target, event)`
    /// pair, or `None` when no triggers are registered for it.
    fn get_index(
        &self,
        txn: &mut Transaction,
        target_segment: &str,
        event_segment: &str,
    ) -> StoreResult<Option<Vec<u8>>>;

    /// Buffer an index write (caller-encoded `Vec<String>` bytes).
    fn put_index(
        &self,
        txn: &mut Transaction,
        target_segment: &str,
        event_segment: &str,
        bytes: &[u8],
    ) -> StoreResult<()>;

    /// Tombstone an index entry (the list became empty).
    fn delete_index(
        &self,
        txn: &mut Transaction,
        target_segment: &str,
        event_segment: &str,
    ) -> StoreResult<()>;

    /// Scan every trigger definition: `(key, raw bytes)` pairs. The index
    /// family (`schema:trigger_index:…`) is excluded — the scan prefix
    /// (`schema:trigger:`) ends in `:`, while index keys have `_` at that byte.
    /// Callers still skip the bare-prefix key defensively.
    fn scan_definitions(&self, txn: &mut Transaction) -> StoreResult<Vec<KvPair>>;
}

/// CE single-shard implementation of [`TriggerStore`].
pub struct LocalTriggerStore;

impl TriggerStore for LocalTriggerStore {
    fn get_definition(&self, txn: &mut Transaction, name: &str) -> StoreResult<Option<Vec<u8>>> {
        Ok(txn.get(Partition::Schema, &encode_trigger_key(name))?)
    }

    fn put_definition(&self, txn: &mut Transaction, name: &str, bytes: &[u8]) -> StoreResult<()> {
        txn.put(Partition::Schema, &encode_trigger_key(name), bytes)?;
        Ok(())
    }

    fn delete_definition(&self, txn: &mut Transaction, name: &str) -> StoreResult<()> {
        txn.delete(Partition::Schema, &encode_trigger_key(name))?;
        Ok(())
    }

    fn get_index(
        &self,
        txn: &mut Transaction,
        target_segment: &str,
        event_segment: &str,
    ) -> StoreResult<Option<Vec<u8>>> {
        let key = encode_trigger_index_key(target_segment, event_segment);
        Ok(txn.get(Partition::Schema, &key)?)
    }

    fn put_index(
        &self,
        txn: &mut Transaction,
        target_segment: &str,
        event_segment: &str,
        bytes: &[u8],
    ) -> StoreResult<()> {
        let key = encode_trigger_index_key(target_segment, event_segment);
        txn.put(Partition::Schema, &key, bytes)?;
        Ok(())
    }

    fn delete_index(
        &self,
        txn: &mut Transaction,
        target_segment: &str,
        event_segment: &str,
    ) -> StoreResult<()> {
        let key = encode_trigger_index_key(target_segment, event_segment);
        txn.delete(Partition::Schema, &key)?;
        Ok(())
    }

    fn scan_definitions(&self, txn: &mut Transaction) -> StoreResult<Vec<KvPair>> {
        let prefix = trigger_scan_prefix();
        let scanned = txn.prefix_scan(Partition::Schema, prefix)?;
        Ok(scanned
            .into_iter()
            .filter(|(k, _)| k.starts_with(prefix) && k.len() != prefix.len())
            .collect())
    }
}
