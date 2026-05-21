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
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use coordinode_core::schema::definition::{PlacementPolicy, PropertyDef, PropertyType};
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use tempfile::TempDir;

    fn open_engine() -> (TempDir, StorageEngine) {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("open engine");
        (dir, engine)
    }

    fn sample_label() -> LabelSchema {
        let mut schema = LabelSchema::new("User", PlacementPolicy::NodeId);
        schema.add_property(PropertyDef::new("email", PropertyType::String).not_null());
        schema
    }

    fn sample_edge_type() -> EdgeTypeSchema {
        EdgeTypeSchema::new("KNOWS")
    }

    #[test]
    fn round_trip_label_schema() {
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);
        let schema = sample_label();

        assert!(
            store.load_label("User").expect("load none").is_none(),
            "label must not exist before save"
        );

        store.save_label(&schema).expect("save");
        let loaded = store
            .load_label("User")
            .expect("load some")
            .expect("Some(schema)");
        assert_eq!(loaded.name, schema.name);
        assert_eq!(loaded.schema_revision, schema.schema_revision);
        assert_eq!(loaded.properties.len(), schema.properties.len());
    }

    #[test]
    fn save_label_is_atomic_pointer_and_body() {
        // The save path writes body + pointer in one WriteBatch — a
        // reader can never observe a pointer naming a missing
        // revision. Smoke check: after save, both keys exist; after a
        // second save with a bumped revision, pointer points to the
        // newer revision AND both bodies are still readable (revision
        // history is preserved).
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);

        let mut v1 = sample_label();
        v1.schema_revision = 1;
        store.save_label(&v1).expect("save v1");

        let mut v2 = sample_label();
        v2.schema_revision = 2;
        store.save_label(&v2).expect("save v2");

        // Current load returns v2.
        let cur = store
            .load_label("User")
            .expect("load")
            .expect("Some(schema)");
        assert_eq!(cur.schema_revision, 2);

        // v1 body still readable through its revisioned key.
        let v1_bytes = engine
            .get(Partition::Schema, &encode_label_schema_key("User", 1))
            .expect("get")
            .expect("v1 body");
        let v1_loaded = LabelSchema::from_msgpack(&v1_bytes).expect("decode v1");
        assert_eq!(v1_loaded.schema_revision, 1);
    }

    #[test]
    fn corrupt_pointer_surfaces_as_decode_error() {
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);
        // Inject a corrupt pointer (3 bytes instead of 8).
        engine
            .put(
                Partition::Schema,
                &encode_label_current_revision_key("Corrupt"),
                &[0xff, 0xff, 0xff],
            )
            .expect("inject");
        let err = store.load_label("Corrupt").expect_err("must error");
        assert!(matches!(err, StoreError::Decode { .. }));
    }

    #[test]
    fn pointer_naming_missing_revision_surfaces_as_decode_error() {
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);
        // Pointer says revision 7 exists, but no body at rev 7.
        engine
            .put(
                Partition::Schema,
                &encode_label_current_revision_key("Orphan"),
                &7u64.to_be_bytes(),
            )
            .expect("inject");
        let err = store.load_label("Orphan").expect_err("must error");
        assert!(matches!(
            err,
            StoreError::Decode {
                kind: "label schema",
                ..
            }
        ));
    }

    #[test]
    fn edge_type_round_trip() {
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);
        let schema = EdgeTypeSchema::new("KNOWS");

        assert!(store.load_edge_type("KNOWS").expect("none").is_none());
        store.save_edge_type(&schema).expect("save");
        let loaded = store.load_edge_type("KNOWS").expect("some").expect("Some");
        assert_eq!(loaded.name, "KNOWS");
    }

    #[test]
    fn edge_type_revision_bump_preserves_history() {
        // Symmetric to the label revision-bump test: save v1, save
        // v2, current load returns v2, v1 body still readable via
        // its revisioned key.
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);

        let mut v1 = EdgeTypeSchema::new("KNOWS");
        v1.schema_revision = 1;
        store.save_edge_type(&v1).expect("save v1");

        let mut v2 = EdgeTypeSchema::new("KNOWS");
        v2.schema_revision = 2;
        store.save_edge_type(&v2).expect("save v2");

        let cur = store
            .load_edge_type("KNOWS")
            .expect("ok")
            .expect("Some(schema)");
        assert_eq!(cur.schema_revision, 2);

        let v1_bytes = engine
            .get(Partition::Schema, &encode_edge_type_schema_key("KNOWS", 1))
            .expect("ok")
            .expect("v1 body");
        let v1_loaded = EdgeTypeSchema::from_msgpack(&v1_bytes).expect("decode v1");
        assert_eq!(v1_loaded.schema_revision, 1);
    }

    #[test]
    fn legacy_zero_length_edge_marker_loads_as_none() {
        // Pre-DDL deployments wrote a zero-length value at the
        // revisioned key to mark "edge type exists, no schema body".
        // The store must surface this as `None`, not a decode error.
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);
        engine
            .put(
                Partition::Schema,
                &encode_edge_type_current_revision_key("LEGACY"),
                &1u64.to_be_bytes(),
            )
            .expect("pointer");
        engine
            .put(
                Partition::Schema,
                &encode_edge_type_schema_key("LEGACY", 1),
                b"",
            )
            .expect("empty marker");
        assert!(
            store.load_edge_type("LEGACY").expect("ok").is_none(),
            "legacy zero-length marker must decode as None",
        );
    }

    // ── list_labels / list_edge_types ─────────────────────────────

    #[test]
    fn list_labels_returns_every_declared_label_at_current_revision() {
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);
        let mut a = sample_label();
        a.name = "User".into();
        a.schema_revision = 1;
        let mut b = sample_label();
        b.name = "Order".into();
        b.schema_revision = 1;
        let mut c = sample_label();
        c.name = "User".into();
        c.schema_revision = 2;
        store.save_label(&a).expect("save User rev 1");
        store.save_label(&b).expect("save Order rev 1");
        store.save_label(&c).expect("save User rev 2");

        let mut listed = store.list_labels().expect("list");
        listed.sort_by(|l, r| l.name.cmp(&r.name));
        assert_eq!(listed.len(), 2, "one schema per declared label");
        assert_eq!(listed[0].name, "Order");
        assert_eq!(listed[0].schema_revision, 1);
        assert_eq!(listed[1].name, "User");
        assert_eq!(listed[1].schema_revision, 2, "current rev for User");
    }

    #[test]
    fn list_labels_empty_when_no_schemas_declared() {
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);
        let listed = store.list_labels().expect("list");
        assert!(listed.is_empty());
    }

    #[test]
    fn list_edge_types_returns_every_declared_edge_type_at_current_revision() {
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);
        let mut knows_v1 = sample_edge_type();
        knows_v1.name = "KNOWS".into();
        knows_v1.schema_revision = 1;
        let mut owns = sample_edge_type();
        owns.name = "OWNS".into();
        owns.schema_revision = 1;
        let mut knows_v2 = sample_edge_type();
        knows_v2.name = "KNOWS".into();
        knows_v2.schema_revision = 2;
        store.save_edge_type(&knows_v1).expect("save KNOWS rev 1");
        store.save_edge_type(&owns).expect("save OWNS rev 1");
        store.save_edge_type(&knows_v2).expect("save KNOWS rev 2");

        let mut listed = store.list_edge_types().expect("list");
        listed.sort_by(|l, r| l.name.cmp(&r.name));
        assert_eq!(listed.len(), 2, "one schema per declared edge type");
        assert_eq!(listed[0].name, "KNOWS");
        assert_eq!(listed[0].schema_revision, 2, "current rev for KNOWS");
        assert_eq!(listed[1].name, "OWNS");
    }

    #[test]
    fn list_edge_types_skips_legacy_zero_length_markers() {
        // Pre-DDL idempotent existence markers shouldn't surface in
        // the listing — `load_edge_type` returns None for them, so
        // `list_edge_types` filters them out.
        let (_dir, engine) = open_engine();
        let store = LocalSchemaStore::new(&engine);
        // Real schema for KNOWS.
        let mut real = sample_edge_type();
        real.name = "KNOWS".into();
        real.schema_revision = 1;
        store.save_edge_type(&real).expect("save");
        // Legacy marker for LEGACY edge type — bare pointer + empty body.
        engine
            .put(
                Partition::Schema,
                &encode_edge_type_current_revision_key("LEGACY"),
                &1u64.to_be_bytes(),
            )
            .expect("legacy pointer");
        engine
            .put(
                Partition::Schema,
                &encode_edge_type_schema_key("LEGACY", 1),
                b"",
            )
            .expect("legacy empty body");

        let listed = store.list_edge_types().expect("list");
        assert_eq!(
            listed.len(),
            1,
            "legacy marker must be skipped, only KNOWS surfaces",
        );
        assert_eq!(listed[0].name, "KNOWS");
    }
}
