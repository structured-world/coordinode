//! Index store — secondary B-tree-style entries in [`Partition::Idx`]
//! plus the index-definition catalog in [`Partition::Schema`].
//!
//! Entries take the form `idx:<name>:<sortable_value>:<node_id>`
//! (value-bytes encoded by [`coordinode_core::index::encoding`] for
//! correct lexicographic ordering). Supports point lookup
//! ([`IndexStore::scan_exact`]) and full-index walk
//! ([`IndexStore::scan_all`]).
//!
//! The store also owns the index-DEFINITION catalog: the serializable
//! [`IndexDefinition`] records keyed by `schema:idx:<name>`. This
//! mirrors how mature engines place the schema/index catalog below the
//! query engine (the query layer issues logical DDL and reads the
//! catalog for planning, but does not own the definition keyspace or
//! its encoding). Definition CRUD lives in
//! [`IndexStore::put_definition`] / [`IndexStore::load_definition`] /
//! [`IndexStore::list_definitions`] / [`IndexStore::delete_definition`].
//!
//! ## Single-column vs compound
//!
//! Both layouts share the same key shape (`idx:name:encoded:id`);
//! compound uses [`encode_compound_value`] to pack multiple [`Value`]s
//! with a separator byte. The store exposes both via
//! [`IndexStore::put_entry`] (slice of values) so the caller doesn't
//! need to branch on arity.

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_core::index::encoding::{
    decode_node_id_from_index_key, encode_compound_index_key, encode_compound_value,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_storage::Guard;

use crate::error::{StoreError, StoreResult};
use crate::index_def::{IndexDefinition, IndexState};

/// Layer 4 index store: entry-level B-tree index ops over
/// [`Partition::Idx`].
pub trait IndexStore {
    /// Insert a (values → node) entry under the named index. Both
    /// single-column (slice of 1) and compound (slice of N) work.
    /// Idempotent: re-putting the same `(name, values, node_id)` is a
    /// no-op semantically.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalIndexStore, IndexStore};
    /// # use coordinode_core::graph::{node::NodeId, types::Value};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalIndexStore::new(&engine);
    /// store.put_entry("by_name", &[Value::String("alice".into())], NodeId::from_raw(1))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn put_entry(&self, name: &str, values: &[Value], node_id: NodeId) -> StoreResult<()>;

    /// Remove a specific entry. Returns Ok even if the entry was
    /// already absent (matches storage tombstone semantics).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalIndexStore, IndexStore};
    /// # use coordinode_core::graph::{node::NodeId, types::Value};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalIndexStore::new(&engine);
    /// store.delete_entry("by_name", &[Value::String("alice".into())], NodeId::from_raw(1))?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn delete_entry(&self, name: &str, values: &[Value], node_id: NodeId) -> StoreResult<()>;

    /// Return all node ids whose entry has the exact given value(s).
    /// Empty `Vec` means "no matches".
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalIndexStore, IndexStore};
    /// # use coordinode_core::graph::types::Value;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalIndexStore::new(&engine);
    /// let hits = store.scan_exact("by_name", &[Value::String("alice".into())])?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn scan_exact(&self, name: &str, values: &[Value]) -> StoreResult<Vec<NodeId>>;

    /// Return all (sortable bytes, node id) pairs in the named index,
    /// without value filtering. Useful for full-index walks (TTL
    /// reaper, index rebuild, count). For large indexes the caller
    /// should prefer a streaming form once Layer 4 grows one — for
    /// PR-scope simplicity this materialises into a `Vec`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalIndexStore, IndexStore};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalIndexStore::new(&engine);
    /// let _all = store.scan_all("by_name")?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn scan_all(&self, name: &str) -> StoreResult<Vec<(Vec<u8>, NodeId)>>;

    /// Delete every entry under the named index prefix. Returns the
    /// number of entries removed. Used for DROP INDEX and the abort
    /// path of an index build (roll back partially-written entries
    /// after a unique-constraint violation).
    fn clear(&self, name: &str) -> StoreResult<usize>;

    /// Delete a single index entry by its raw key — the opaque bytes
    /// handed back from [`IndexStore::scan_all`]. Lets a full-index walk
    /// (e.g. the TTL reaper) selectively drop the entries it decided are
    /// expired without reconstructing the `(values, node_id)` tuple. The
    /// caller never builds the key; it only passes back one it scanned.
    fn delete_raw(&self, raw_key: &[u8]) -> StoreResult<()>;

    /// Return every entry's node id under the named index in key order,
    /// without value filtering. Backs full-index walks (range queries,
    /// SHOW INDEX). Distinct from [`IndexStore::scan_all`] in that it
    /// does not allocate a key buffer per entry.
    fn scan_entry_ids(&self, name: &str) -> StoreResult<Vec<NodeId>>;

    /// Persist an index definition into the schema catalog (keyed by
    /// `schema:idx:<name>`). Overwrites any existing definition with the
    /// same name. The store owns both the catalog keyspace and the
    /// MessagePack encoding — callers pass the typed definition only.
    fn put_definition(&self, def: &IndexDefinition) -> StoreResult<()>;

    /// Load a persisted index definition by name. `Ok(None)` when no
    /// definition is stored under that name.
    fn load_definition(&self, name: &str) -> StoreResult<Option<IndexDefinition>>;

    /// List every persisted index definition in `schema:idx:` key order.
    /// A definition whose stored bytes fail to decode is skipped (with a
    /// tracing warning) rather than aborting the whole listing — one
    /// corrupt record must not take down registry rebuild on open.
    fn list_definitions(&self) -> StoreResult<Vec<IndexDefinition>>;

    /// Delete a persisted index definition by name. Returns `Ok(())`
    /// even when no definition was stored (tombstone semantics).
    fn delete_definition(&self, name: &str) -> StoreResult<()>;

    /// Update only the build `state` of a persisted definition, leaving
    /// every other field intact. Returns `Ok(false)` when no definition
    /// is stored under `name` (the caller handles the race). Used by the
    /// backfill task to publish progress / terminal states.
    fn set_definition_state(&self, name: &str, state: IndexState) -> StoreResult<bool>;

    /// Persist an index definition through a statement [`Transaction`]
    /// (OCC-tracked, read-your-own-writes) — the CREATE INDEX DDL path. Same
    /// `schema:idx:<name>` keyspace and encoding as [`Self::put_definition`],
    /// but the write buffers on the transaction and applies atomically at
    /// commit, so a CREATE INDEX racing a conflicting schema change is
    /// detected like any other write. Mirrors the `SchemaStore::*_txn` family.
    fn put_definition_txn(&self, txn: &mut Transaction, def: &IndexDefinition) -> StoreResult<()>;

    /// Delete a persisted index definition by name through a statement
    /// [`Transaction`] — the DROP INDEX DDL path. Tombstone semantics (no error
    /// when absent).
    fn delete_definition_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<()>;
}

/// CE single-shard implementation of [`IndexStore`].
pub struct LocalIndexStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalIndexStore<'a> {
    /// Wrap a storage engine for index-store operations.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use coordinode_modality::{LocalIndexStore, IndexStore};
    /// use coordinode_core::graph::node::NodeId;
    /// use coordinode_core::graph::types::Value;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/store"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm,
    /// # )]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// let store = LocalIndexStore::new(&engine);
    /// let key = [Value::String("alice".into())];
    /// store.put_entry("by_name", &key, NodeId::from_raw(1))?;
    /// let hits = store.scan_exact("by_name", &key)?;
    /// assert_eq!(hits, vec![NodeId::from_raw(1)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }
}

fn index_prefix(name: &str) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + name.len() + 1);
    p.extend_from_slice(b"idx:");
    p.extend_from_slice(name.as_bytes());
    p.push(b':');
    p
}

fn definition_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(11 + name.len());
    key.extend_from_slice(b"schema:idx:");
    key.extend_from_slice(name.as_bytes());
    key
}

fn index_value_prefix(name: &str, values: &[Value]) -> Vec<u8> {
    let encoded = encode_compound_value(values);
    let mut p = Vec::with_capacity(4 + name.len() + 1 + encoded.len() + 1);
    p.extend_from_slice(b"idx:");
    p.extend_from_slice(name.as_bytes());
    p.push(b':');
    p.extend_from_slice(&encoded);
    p.push(b':');
    p
}

impl IndexStore for LocalIndexStore<'_> {
    fn put_entry(&self, name: &str, values: &[Value], node_id: NodeId) -> StoreResult<()> {
        let key = encode_compound_index_key(name, values, node_id.as_raw());
        self.engine.put(Partition::Idx, &key, &[])?;
        Ok(())
    }

    fn delete_entry(&self, name: &str, values: &[Value], node_id: NodeId) -> StoreResult<()> {
        let key = encode_compound_index_key(name, values, node_id.as_raw());
        self.engine.delete(Partition::Idx, &key)?;
        Ok(())
    }

    fn scan_exact(&self, name: &str, values: &[Value]) -> StoreResult<Vec<NodeId>> {
        let prefix = index_value_prefix(name, values);
        let iter = self.engine.prefix_scan(Partition::Idx, &prefix)?;
        let mut out = Vec::new();
        for guard in iter {
            let (key, _) = guard.into_inner()?;
            if let Some(id) = decode_node_id_from_index_key(&key) {
                out.push(NodeId::from_raw(id));
            }
        }
        Ok(out)
    }

    fn scan_all(&self, name: &str) -> StoreResult<Vec<(Vec<u8>, NodeId)>> {
        let prefix = index_prefix(name);
        let iter = self.engine.prefix_scan(Partition::Idx, &prefix)?;
        let mut out = Vec::new();
        for guard in iter {
            let (key, _) = guard.into_inner()?;
            let Some(id) = decode_node_id_from_index_key(&key) else {
                continue;
            };
            out.push((key.to_vec(), NodeId::from_raw(id)));
        }
        Ok(out)
    }

    fn clear(&self, name: &str) -> StoreResult<usize> {
        let prefix = index_prefix(name);
        let iter = self.engine.prefix_scan(Partition::Idx, &prefix)?;
        // Collect first, then delete: deleting while holding the scan
        // iterator over the same partition is not guaranteed safe across
        // engine implementations.
        let mut keys: Vec<Vec<u8>> = Vec::new();
        for guard in iter {
            let (key, _) = guard.into_inner()?;
            keys.push(key.to_vec());
        }
        let removed = keys.len();
        for key in &keys {
            self.engine.delete(Partition::Idx, key)?;
        }
        Ok(removed)
    }

    fn delete_raw(&self, raw_key: &[u8]) -> StoreResult<()> {
        self.engine.delete(Partition::Idx, raw_key)?;
        Ok(())
    }

    fn scan_entry_ids(&self, name: &str) -> StoreResult<Vec<NodeId>> {
        let prefix = index_prefix(name);
        let iter = self.engine.prefix_scan(Partition::Idx, &prefix)?;
        let mut out = Vec::new();
        for guard in iter {
            let (key, _) = guard.into_inner()?;
            if let Some(id) = decode_node_id_from_index_key(&key) {
                out.push(NodeId::from_raw(id));
            }
        }
        Ok(out)
    }

    fn put_definition(&self, def: &IndexDefinition) -> StoreResult<()> {
        let key = def.schema_key();
        let value = rmp_serde::to_vec(def)
            .map_err(|e| StoreError::Invariant(format!("index definition serialize: {e}")))?;
        self.engine.put(Partition::Schema, &key, &value)?;
        Ok(())
    }

    fn load_definition(&self, name: &str) -> StoreResult<Option<IndexDefinition>> {
        let key = definition_key(name);
        match self.engine.get(Partition::Schema, &key)? {
            Some(bytes) => {
                let def = rmp_serde::from_slice(&bytes).map_err(|e| StoreError::Decode {
                    kind: "index definition",
                    message: e.to_string(),
                })?;
                Ok(Some(def))
            }
            None => Ok(None),
        }
    }

    fn list_definitions(&self) -> StoreResult<Vec<IndexDefinition>> {
        let iter = self.engine.prefix_scan(Partition::Schema, b"schema:idx:")?;
        let mut out = Vec::new();
        for guard in iter {
            let (_key, value) = guard.into_inner()?;
            match rmp_serde::from_slice::<IndexDefinition>(&value) {
                Ok(def) => out.push(def),
                Err(e) => {
                    tracing::warn!("list_definitions: skipping corrupt index def: {e}");
                    continue;
                }
            }
        }
        Ok(out)
    }

    fn delete_definition(&self, name: &str) -> StoreResult<()> {
        let key = definition_key(name);
        self.engine.delete(Partition::Schema, &key)?;
        Ok(())
    }

    fn set_definition_state(&self, name: &str, state: IndexState) -> StoreResult<bool> {
        let Some(mut def) = self.load_definition(name)? else {
            return Ok(false);
        };
        def.state = state;
        self.put_definition(&def)?;
        Ok(true)
    }

    fn put_definition_txn(&self, txn: &mut Transaction, def: &IndexDefinition) -> StoreResult<()> {
        let value = rmp_serde::to_vec(def)
            .map_err(|e| StoreError::Invariant(format!("index definition serialize: {e}")))?;
        txn.put(Partition::Schema, &def.schema_key(), &value)?;
        Ok(())
    }

    fn delete_definition_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<()> {
        txn.delete(Partition::Schema, &definition_key(name))?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
