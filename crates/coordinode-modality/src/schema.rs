//! Schema store — DDL state for labels, edge types, and per-label
//! placement metadata.
//!
//! All schema state lives in [`Partition::Schema`]. The store hides:
//!
//! - Key format (`schema:label:<name>:<revision>`,
//!   `schema:current_revision:label:<name>`, mirrors for edge types).
//! - Revision indirection: callers read "current schema" without
//!   knowing the revision pointer trick.
//! - Atomic write composition: saving a schema writes the body AND
//!   updates the current-revision pointer in a single batch, so
//!   readers never observe a pointer that names a missing revision.
//!
//! Schema DDL is Raft-replicated above this layer; the store assumes
//! the caller already holds the appropriate write authority (leader,
//! valid revision number, …).

use coordinode_core::schema::definition::{
    encode_edge_type_current_revision_key, encode_edge_type_schema_key,
    encode_label_current_revision_key, encode_label_schema_key, EdgeTypeSchema, LabelSchema,
};
use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_storage::Guard;

use crate::error::{StoreError, StoreResult};

/// Layer 4 schema store: typed read/write of label and edge type
/// schemas, hiding revision indirection and partition keys.
pub trait SchemaStore {
    /// Load the current revision of a label schema by name. Returns
    /// `None` if the label is not declared.
    ///
    /// Resolves the `schema:current_revision:label:<name>` pointer,
    /// then loads the revision-suffixed body. Returns `Decode` if the
    /// pointer is corrupt (not 8 bytes) or the named revision is
    /// missing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalSchemaStore, SchemaStore};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalSchemaStore::new(&engine);
    /// let _label = store.load_label("User")?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn load_label(&self, name: &str) -> StoreResult<Option<LabelSchema>>;

    /// Persist a label schema as the current revision. Body and
    /// pointer are written in a single atomic batch — readers never
    /// observe a pointer naming a missing revision.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalSchemaStore, SchemaStore};
    /// # use coordinode_core::schema::definition::LabelSchema;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalSchemaStore::new(&engine);
    /// # let schema: LabelSchema = unimplemented!();
    /// store.save_label(&schema)?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn save_label(&self, schema: &LabelSchema) -> StoreResult<()>;

    /// Load the current revision of an edge type schema by name.
    /// Symmetric to [`Self::load_label`]. Returns `None` for missing
    /// edge type OR for legacy zero-length idempotent existence
    /// markers (predates DDL revisioning).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalSchemaStore, SchemaStore};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalSchemaStore::new(&engine);
    /// let _edge_type = store.load_edge_type("KNOWS")?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn load_edge_type(&self, name: &str) -> StoreResult<Option<EdgeTypeSchema>>;

    /// Persist an edge type schema as the current revision. Same
    /// atomicity contract as [`Self::save_label`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalSchemaStore, SchemaStore};
    /// # use coordinode_core::schema::definition::EdgeTypeSchema;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalSchemaStore::new(&engine);
    /// let schema = EdgeTypeSchema::new("KNOWS");
    /// store.save_edge_type(&schema)?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn save_edge_type(&self, schema: &EdgeTypeSchema) -> StoreResult<()>;

    /// List every declared label schema at its current revision.
    /// Returns one [`LabelSchema`] per active label, in arbitrary
    /// order (callers must sort if they need determinism). Labels
    /// whose pointer references a missing body are skipped with a
    /// `tracing::warn!` rather than aborting the whole listing.
    ///
    /// Used by background catalog work (TTL reaper, schema cache
    /// build-up, observability endpoints) that need to enumerate
    /// every label without knowing the names up front. Unlocks
    /// migration of `coordinode-query::ttl_reaper::discover_ttl_targets`
    /// off raw `schema:label:` prefix scans.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use coordinode_modality::{LocalSchemaStore, SchemaStore};
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/x"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// # let store = LocalSchemaStore::new(&engine);
    /// for schema in store.list_labels()? {
    ///     println!("label {}", schema.name);
    /// }
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn list_labels(&self) -> StoreResult<Vec<LabelSchema>>;

    /// List every declared edge type schema at its current revision.
    /// Symmetric to [`Self::list_labels`]. Returns one
    /// [`EdgeTypeSchema`] per active edge type, in arbitrary order.
    fn list_edge_types(&self) -> StoreResult<Vec<EdgeTypeSchema>>;

    // ── Transaction-threaded variants (query-execution path) ───────────────
    // The query layer reads/writes schema inside a statement transaction:
    // pointer + body resolve through the transaction (RYOW + OCC tracking in
    // MVCC mode). Same revision-indirection contract as the engine variants.

    /// Load the current label schema through a transaction (tracked read).
    fn load_label_txn(&self, txn: &mut Transaction, name: &str)
        -> StoreResult<Option<LabelSchema>>;

    /// Load the current edge type schema through a transaction (tracked read).
    fn load_edge_type_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
    ) -> StoreResult<Option<EdgeTypeSchema>>;

    /// Persist a label schema (body + current-revision pointer) on the
    /// transaction's write buffer; applied atomically at commit.
    fn save_label_txn(&self, txn: &mut Transaction, schema: &LabelSchema) -> StoreResult<()>;

    /// Persist an edge type schema (body + pointer) on the transaction.
    fn save_edge_type_txn(&self, txn: &mut Transaction, schema: &EdgeTypeSchema)
        -> StoreResult<()>;

    /// Idempotently write the implicit edge-type existence marker (empty body
    /// at revision 1) — checks the write buffer (RYOW) then the OCC-tracked
    /// snapshot before writing, so a concurrent `CREATE EDGE TYPE` body is
    /// never clobbered. Edge-create paths call this for first-seen types.
    fn register_edge_type_marker(&self, txn: &mut Transaction, name: &str) -> StoreResult<()>;

    /// Whether an edge type exists: explicit current-revision pointer OR the
    /// implicit existence marker (revision 1). Both reads are OCC-tracked.
    fn edge_type_exists(&self, txn: &mut Transaction, name: &str) -> StoreResult<bool>;

    /// Distinct names of every edge type registered in the schema partition,
    /// including types created in this transaction's (not yet flushed) write
    /// buffer. Enumerates the `schema:edge_type:<name>:<version>` family and
    /// strips the version suffix.
    fn list_edge_type_names(&self, txn: &mut Transaction) -> StoreResult<Vec<String>>;

    /// Like [`Self::list_edge_type_names`] but reads straight from the engine at
    /// the latest committed state, with no [`Transaction`]. For background
    /// scanners (TTL reaper) that enumerate edge types without an MVCC
    /// transaction. Includes existence markers (zero-length bodies): a
    /// registered type with only a marker still has adjacency a reaper must
    /// clean, so unlike [`Self::list_edge_types`] this does not drop them.
    fn list_edge_type_names_engine(&self) -> StoreResult<Vec<String>>;
}

/// Extract the edge-type name from a `schema:edge_type:<name>:<version>` key.
/// Names cannot contain ':' (DDL grammar), so the rightmost ':' splits name
/// from version.
fn edge_type_name_from_key(key: &[u8]) -> Option<String> {
    const PREFIX: &[u8] = b"schema:edge_type:";
    let suffix = key.get(PREFIX.len()..)?;
    let suffix_str = std::str::from_utf8(suffix).ok()?;
    let (name, _version) = suffix_str.rsplit_once(':')?;
    Some(name.to_string())
}

/// CE single-shard implementation of [`SchemaStore`]. Operates
/// directly on a [`StorageEngine`]. Reads use point gets; writes use
/// a two-op [`WriteBatch`] for revision-body + pointer atomicity.
pub struct LocalSchemaStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalSchemaStore<'a> {
    /// Wrap a storage engine for schema-store operations. Cheap: the
    /// store carries only a borrow.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use coordinode_modality::{LocalSchemaStore, SchemaStore};
    /// use coordinode_core::schema::definition::LabelSchema;
    /// # use coordinode_storage::engine::{config::*, core::StorageEngine};
    /// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
    /// #     "ep", std::path::Path::new("/tmp/store"),
    /// #     Media::Hdd, Durability::Durable, Tier::Warm,
    /// # )]);
    /// # let engine = StorageEngine::open(&cfg)?;
    /// let store = LocalSchemaStore::new(&engine);
    /// assert!(store.load_label("User")?.is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }

    fn load_revision_pointer(&self, key: &[u8], kind: &'static str) -> StoreResult<Option<u64>> {
        let Some(bytes) = self.engine.get(Partition::Schema, key)? else {
            return Ok(None);
        };
        let array: [u8; 8] = bytes.as_ref().try_into().map_err(|_| StoreError::Decode {
            kind,
            message: format!("revision pointer expected 8 bytes, got {}", bytes.len()),
        })?;
        Ok(Some(u64::from_be_bytes(array)))
    }

    /// Transaction-threaded revision pointer read (tracked).
    fn load_revision_pointer_txn(
        txn: &mut Transaction,
        key: &[u8],
        kind: &'static str,
    ) -> StoreResult<Option<u64>> {
        let Some(bytes) = txn.get(Partition::Schema, key)? else {
            return Ok(None);
        };
        let array: [u8; 8] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| StoreError::Decode {
                kind,
                message: format!("revision pointer expected 8 bytes, got {}", bytes.len()),
            })?;
        Ok(Some(u64::from_be_bytes(array)))
    }
}

impl SchemaStore for LocalSchemaStore<'_> {
    fn load_label(&self, name: &str) -> StoreResult<Option<LabelSchema>> {
        let pointer_key = encode_label_current_revision_key(name);
        let Some(revision) = self.load_revision_pointer(&pointer_key, "label revision pointer")?
        else {
            return Ok(None);
        };
        let schema_key = encode_label_schema_key(name, revision);
        let Some(schema_bytes) = self.engine.get(Partition::Schema, &schema_key)? else {
            return Err(StoreError::Decode {
                kind: "label schema",
                message: format!("pointer for '{name}' references missing revision {revision}"),
            });
        };
        LabelSchema::from_msgpack(&schema_bytes)
            .map(Some)
            .map_err(|e| StoreError::Decode {
                kind: "label schema",
                message: format!("decode failed for '{name}' rev {revision}: {e}"),
            })
    }

    fn save_label(&self, schema: &LabelSchema) -> StoreResult<()> {
        let body = schema.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "label schema",
            message: format!("encode '{}': {e}", schema.name),
        })?;
        let mut batch = WriteBatch::new(self.engine);
        batch.put(
            Partition::Schema,
            encode_label_schema_key(&schema.name, schema.schema_revision),
            body,
        );
        batch.put(
            Partition::Schema,
            encode_label_current_revision_key(&schema.name),
            schema.schema_revision.to_be_bytes().to_vec(),
        );
        batch.commit()?;
        Ok(())
    }

    fn load_edge_type(&self, name: &str) -> StoreResult<Option<EdgeTypeSchema>> {
        let pointer_key = encode_edge_type_current_revision_key(name);
        let Some(revision) =
            self.load_revision_pointer(&pointer_key, "edge type revision pointer")?
        else {
            return Ok(None);
        };
        let schema_key = encode_edge_type_schema_key(name, revision);
        let Some(schema_bytes) = self.engine.get(Partition::Schema, &schema_key)? else {
            return Err(StoreError::Decode {
                kind: "edge type schema",
                message: format!("pointer for '{name}' references missing revision {revision}"),
            });
        };
        // Legacy zero-length idempotent existence marker predates DDL
        // revisioning — surface as "no schema declared" to callers.
        if schema_bytes.is_empty() {
            return Ok(None);
        }
        EdgeTypeSchema::from_msgpack(&schema_bytes)
            .map(Some)
            .map_err(|e| StoreError::Decode {
                kind: "edge type schema",
                message: format!("decode failed for '{name}' rev {revision}: {e}"),
            })
    }

    fn save_edge_type(&self, schema: &EdgeTypeSchema) -> StoreResult<()> {
        let body = schema.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "edge type schema",
            message: format!("encode '{}': {e}", schema.name),
        })?;
        let mut batch = WriteBatch::new(self.engine);
        batch.put(
            Partition::Schema,
            encode_edge_type_schema_key(&schema.name, schema.schema_revision),
            body,
        );
        batch.put(
            Partition::Schema,
            encode_edge_type_current_revision_key(&schema.name),
            schema.schema_revision.to_be_bytes().to_vec(),
        );
        batch.commit()?;
        Ok(())
    }

    fn list_labels(&self) -> StoreResult<Vec<LabelSchema>> {
        // Enumerate via the current-revision pointer prefix —
        // pointers point at the active revision body, so this
        // surfaces exactly one schema per declared label (vs the
        // bare `schema:label:` prefix which yields one row per
        // historical revision).
        const POINTER_PREFIX: &[u8] = b"schema:current_revision:label:";
        let iter = self.engine.prefix_scan(Partition::Schema, POINTER_PREFIX)?;
        let mut out = Vec::new();
        for guard in iter {
            let (key, _) = guard.into_inner()?;
            let Some(name) = std::str::from_utf8(&key[POINTER_PREFIX.len()..]).ok() else {
                tracing::warn!(
                    "SchemaStore::list_labels: skipping non-UTF8 label name in pointer key",
                );
                continue;
            };
            // `load_label` does the pointer-resolve + body decode in
            // one shot. Errors propagate (a corrupt body is a real
            // failure); a missing body would have been an Err inside
            // load_label too.
            match self.load_label(name) {
                Ok(Some(schema)) => out.push(schema),
                Ok(None) => {
                    // Pointer existed but resolved to None — race
                    // against a concurrent DROP LABEL. Skip silently.
                }
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }

    fn list_edge_types(&self) -> StoreResult<Vec<EdgeTypeSchema>> {
        const POINTER_PREFIX: &[u8] = b"schema:current_revision:edge_type:";
        let iter = self.engine.prefix_scan(Partition::Schema, POINTER_PREFIX)?;
        let mut out = Vec::new();
        for guard in iter {
            let (key, _) = guard.into_inner()?;
            let Some(name) = std::str::from_utf8(&key[POINTER_PREFIX.len()..]).ok() else {
                tracing::warn!(
                    "SchemaStore::list_edge_types: skipping non-UTF8 edge type name in pointer key",
                );
                continue;
            };
            match self.load_edge_type(name) {
                Ok(Some(schema)) => out.push(schema),
                Ok(None) => {
                    // Either a pre-DDL idempotent existence marker
                    // (zero-length body — `load_edge_type` returns
                    // `None`) or a concurrent DROP. Skip.
                }
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }

    fn load_label_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
    ) -> StoreResult<Option<LabelSchema>> {
        let pointer_key = encode_label_current_revision_key(name);
        let Some(revision) = Self::load_revision_pointer_txn(txn, &pointer_key, "label")? else {
            return Ok(None);
        };
        let schema_key = encode_label_schema_key(name, revision);
        let Some(schema_bytes) = txn.get(Partition::Schema, &schema_key)? else {
            return Err(StoreError::Decode {
                kind: "label schema",
                message: format!("pointer for '{name}' references missing revision {revision}"),
            });
        };
        LabelSchema::from_msgpack(&schema_bytes)
            .map(Some)
            .map_err(|e| StoreError::Decode {
                kind: "label schema",
                message: format!("decode failed for '{name}' rev {revision}: {e}"),
            })
    }

    fn load_edge_type_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
    ) -> StoreResult<Option<EdgeTypeSchema>> {
        let pointer_key = encode_edge_type_current_revision_key(name);
        let Some(revision) = Self::load_revision_pointer_txn(txn, &pointer_key, "edge type")?
        else {
            return Ok(None);
        };
        let schema_key = encode_edge_type_schema_key(name, revision);
        let Some(schema_bytes) = txn.get(Partition::Schema, &schema_key)? else {
            return Err(StoreError::Decode {
                kind: "edge type schema",
                message: format!("pointer for '{name}' references missing revision {revision}"),
            });
        };
        // Legacy zero-length idempotent existence marker → "no schema".
        if schema_bytes.is_empty() {
            return Ok(None);
        }
        EdgeTypeSchema::from_msgpack(&schema_bytes)
            .map(Some)
            .map_err(|e| StoreError::Decode {
                kind: "edge type schema",
                message: format!("decode failed for '{name}' rev {revision}: {e}"),
            })
    }

    fn save_label_txn(&self, txn: &mut Transaction, schema: &LabelSchema) -> StoreResult<()> {
        let body = schema.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "label schema",
            message: format!("encode '{}': {e}", schema.name),
        })?;
        let schema_key = encode_label_schema_key(&schema.name, schema.schema_revision);
        txn.put(Partition::Schema, &schema_key, &body)?;
        let pointer_key = encode_label_current_revision_key(&schema.name);
        txn.put(
            Partition::Schema,
            &pointer_key,
            &schema.schema_revision.to_be_bytes(),
        )?;
        Ok(())
    }

    fn save_edge_type_txn(
        &self,
        txn: &mut Transaction,
        schema: &EdgeTypeSchema,
    ) -> StoreResult<()> {
        let body = schema.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "edge type schema",
            message: format!("encode '{}': {e}", schema.name),
        })?;
        let schema_key = encode_edge_type_schema_key(&schema.name, schema.schema_revision);
        txn.put(Partition::Schema, &schema_key, &body)?;
        let pointer_key = encode_edge_type_current_revision_key(&schema.name);
        txn.put(
            Partition::Schema,
            &pointer_key,
            &schema.schema_revision.to_be_bytes(),
        )?;
        Ok(())
    }

    fn register_edge_type_marker(&self, txn: &mut Transaction, name: &str) -> StoreResult<()> {
        let key = encode_edge_type_schema_key(name, 1);
        let already = txn.buffered(Partition::Schema, &key).is_some()
            || txn.get(Partition::Schema, &key)?.is_some();
        if !already {
            txn.put(Partition::Schema, &key, b"")?;
        }
        Ok(())
    }

    fn edge_type_exists(&self, txn: &mut Transaction, name: &str) -> StoreResult<bool> {
        let pointer_key = encode_edge_type_current_revision_key(name);
        if txn.get(Partition::Schema, &pointer_key)?.is_some() {
            return Ok(true);
        }
        let marker_key = encode_edge_type_schema_key(name, 1);
        Ok(txn.get(Partition::Schema, &marker_key)?.is_some())
    }

    fn list_edge_type_names(&self, txn: &mut Transaction) -> StoreResult<Vec<String>> {
        const PREFIX: &[u8] = b"schema:edge_type:";
        let mut types: Vec<String> = Vec::new();
        for (k, _) in txn.prefix_scan(Partition::Schema, PREFIX)? {
            if let Some(name) = edge_type_name_from_key(&k) {
                if !types.contains(&name) {
                    types.push(name);
                }
            }
        }
        Ok(types)
    }

    fn list_edge_type_names_engine(&self) -> StoreResult<Vec<String>> {
        const PREFIX: &[u8] = b"schema:edge_type:";
        let mut types: Vec<String> = Vec::new();
        for guard in self.engine.prefix_scan(Partition::Schema, PREFIX)? {
            let (k, _) = guard.into_inner()?;
            if let Some(name) = edge_type_name_from_key(&k) {
                if !types.contains(&name) {
                    types.push(name);
                }
            }
        }
        Ok(types)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
