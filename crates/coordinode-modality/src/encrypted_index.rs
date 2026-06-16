//! Encrypted (blind) index catalog.
//!
//! Stores the metadata for an encrypted-search index — a serializable
//! catalog record keyed by `encrypted_index:<name>` in [`Partition::Schema`].
//! Like the other catalog stores ([`crate::IndexStore`],
//! [`crate::SchemaStore`]), the query layer issues logical CREATE/DROP
//! ENCRYPTED INDEX DDL and this store owns the keyspace and encoding — Layer 5
//! names neither.
//!
//! This is intentionally minimal: the record carries the index identity
//! (`name`, `label`, `property`). The token scheme, key material, and a
//! listing/registry are part of the encrypted-search feature build-out and
//! are added when that lands; the catalog typing here exists so the DDL path
//! conforms to the storage layering today.

use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use serde::{Deserialize, Serialize};

use crate::error::{StoreError, StoreResult};

/// Catalog record describing an encrypted (blind) index: which property of
/// which label is searchable via token-based equality without exposing
/// plaintext to the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncryptedIndexDefinition {
    /// Index name (unique per database).
    pub name: String,
    /// Node label this index applies to.
    pub label: String,
    /// Indexed property name.
    pub property: String,
}

impl EncryptedIndexDefinition {
    /// Build a definition for `name` over `label(property)`.
    pub fn new(
        name: impl Into<String>,
        label: impl Into<String>,
        property: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            property: property.into(),
        }
    }

    /// Schema-catalog storage key for this definition: `encrypted_index:<name>`.
    pub fn schema_key(&self) -> Vec<u8> {
        encrypted_index_key(&self.name)
    }
}

fn encrypted_index_key(name: &str) -> Vec<u8> {
    const PREFIX: &[u8] = b"encrypted_index:";
    let mut key = Vec::with_capacity(PREFIX.len() + name.len());
    key.extend_from_slice(PREFIX);
    key.extend_from_slice(name.as_bytes());
    key
}

/// Layer 4 store for the encrypted-index catalog: typed CRUD over the
/// `encrypted_index:` keyspace in [`Partition::Schema`].
pub trait EncryptedIndexStore {
    /// Persist a definition through a statement [`Transaction`] (OCC-tracked,
    /// applied atomically at commit) — the CREATE ENCRYPTED INDEX DDL path.
    /// Mirrors [`crate::IndexStore::put_definition_txn`].
    fn put_definition_txn(
        &self,
        txn: &mut Transaction,
        def: &EncryptedIndexDefinition,
    ) -> StoreResult<()>;

    /// Delete a definition by name through a statement [`Transaction`] — the
    /// DROP ENCRYPTED INDEX DDL path. Tombstone semantics (no error when
    /// absent).
    fn delete_definition_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<()>;

    /// Load a persisted definition by name (latest committed state).
    /// `Ok(None)` when no definition is stored under that name.
    fn load_definition(&self, name: &str) -> StoreResult<Option<EncryptedIndexDefinition>>;
}

/// CE single-shard implementation of [`EncryptedIndexStore`].
pub struct LocalEncryptedIndexStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalEncryptedIndexStore<'a> {
    /// Wrap a storage engine for encrypted-index catalog operations.
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }
}

impl EncryptedIndexStore for LocalEncryptedIndexStore<'_> {
    fn put_definition_txn(
        &self,
        txn: &mut Transaction,
        def: &EncryptedIndexDefinition,
    ) -> StoreResult<()> {
        let value = rmp_serde::to_vec(def).map_err(|e| {
            StoreError::Invariant(format!("encrypted index definition serialize: {e}"))
        })?;
        txn.put(Partition::Schema, &def.schema_key(), &value)?;
        Ok(())
    }

    fn delete_definition_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<()> {
        txn.delete(Partition::Schema, &encrypted_index_key(name))?;
        Ok(())
    }

    fn load_definition(&self, name: &str) -> StoreResult<Option<EncryptedIndexDefinition>> {
        match self
            .engine
            .get(Partition::Schema, &encrypted_index_key(name))?
        {
            Some(bytes) => {
                let def = rmp_serde::from_slice(&bytes).map_err(|e| StoreError::Decode {
                    kind: "encrypted index definition",
                    message: e.to_string(),
                })?;
                Ok(Some(def))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_storage::engine::transaction::CommitContext;

    fn open_engine() -> coordinode_test_fixtures::EngineFixture {
        coordinode_test_fixtures::engine_for_logic()
    }

    #[test]
    fn schema_key_format_is_stable() {
        // The DDL key contract is `encrypted_index:<name>` — integration
        // tests and any future scanner depend on this exact prefix.
        let def = EncryptedIndexDefinition::new("idx_ssn", "Patient", "ssn");
        assert_eq!(def.schema_key(), b"encrypted_index:idx_ssn");
    }

    #[test]
    fn definition_txn_round_trip() {
        let fx = open_engine();
        let engine = &fx.engine;
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let store = LocalEncryptedIndexStore::new(engine);

        let commit = |t: &mut Transaction| {
            let wc = WriteConcern::majority();
            let ctx = CommitContext {
                write_concern: &wc,
                pipeline: None,
                id_gen: None,
                drain_buffer: None,
                nvme_write_buffer: None,
            };
            t.commit(&ctx).expect("commit");
        };

        let def = EncryptedIndexDefinition::new("idx_email", "User", "email");

        // CREATE: persist through a statement transaction.
        let read_ts = oracle.next();
        let mut t = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        store.put_definition_txn(&mut t, &def).expect("put txn");
        commit(&mut t);
        let loaded = store
            .load_definition("idx_email")
            .expect("load")
            .expect("present after commit");
        assert_eq!(loaded, def);

        // DROP: delete through a statement transaction.
        let read_ts = oracle.next();
        let mut t = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        store
            .delete_definition_txn(&mut t, "idx_email")
            .expect("delete txn");
        commit(&mut t);
        assert!(store.load_definition("idx_email").expect("load").is_none());
    }
}
