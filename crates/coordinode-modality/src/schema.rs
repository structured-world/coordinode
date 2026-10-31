//! Schema store — DDL state for labels, edge types, and per-label
//! placement metadata.
//!
//! All schema state lives in [`Partition::Schema`]. The store hides:
//!
//! - Key format (`schema:label:<id>:<revision>`,
//!   `schema:current_revision:label:<name>`, mirrors for edge types).
//! - Label identity: a name is bound to a catalog identity
//!   ([`coordinode_core::schema::label_id`]), allocated on the label's first
//!   use or declaration and retired when it is dropped; definitions live
//!   under the identity.
//! - Revision indirection: callers read "current schema" without
//!   knowing the revision pointer trick.
//! - Atomic write composition: saving a schema writes the body AND
//!   updates the name binding in a single batch, so readers never
//!   observe a binding that names a missing revision.
//!
//! Schema DDL is Raft-replicated above this layer; the store assumes
//! the caller already holds the appropriate write authority (leader,
//! valid revision number, …).

use coordinode_core::schema::definition::{
    EDGE_TYPE_SCHEMA_KEY_PREFIX, EdgeTypeSchema, LabelSchema, decode_edge_type_schema_key_name,
    decode_label_binding_history_commit, encode_constraint_name_key,
    encode_edge_type_current_revision_key, encode_edge_type_schema_key,
    encode_label_binding_history_prefix, encode_label_current_revision_key,
    encode_label_schema_key,
};
use coordinode_core::schema::label_id::{BindingState, LabelBinding, LabelId, label_id_key};
use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

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
    /// observe a pointer naming a missing revision. The batch commits at no
    /// transaction timestamp, so it keeps no binding history: a retained
    /// system time reports the binding as uncovered rather than reading it.
    /// Writers whose history matters stage the definition in a transaction
    /// ([`Self::save_label_txn`]).
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

    /// Load the label schema as committed now, to write its next revision
    /// in `txn`: the commit is conditioned on that schema still being the
    /// current one, so a change committed in between refuses it rather than
    /// being overwritten.
    fn load_label_for_update_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
    ) -> StoreResult<Option<LabelSchema>>;

    /// Load the current edge type schema through a transaction (tracked read).
    fn load_edge_type_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
    ) -> StoreResult<Option<EdgeTypeSchema>>;

    /// Persist a label schema (body + current-revision pointer) on the
    /// transaction's write buffer; applied atomically at commit.
    fn save_label_txn(&self, txn: &mut Transaction, schema: &LabelSchema) -> StoreResult<()>;

    /// Persist a revision that requires nothing of a node the current one
    /// does not (a constraint removed, or one moving from validating to
    /// active), without checking the stored nodes against it: they satisfy
    /// it because they satisfy the current one. A writer validated under the
    /// current revision is still refused at its commit, the pointer having
    /// moved; a concurrent change of the label's schema conflicts on it.
    fn save_label_admitting_txn(
        &self,
        txn: &mut Transaction,
        schema: &LabelSchema,
    ) -> StoreResult<()>;

    /// Persist a revision carrying a candidate to prove over the stored
    /// nodes, without checking them at this commit: the proof does, after
    /// it. Writers validated under the current revision that are still on
    /// their way to the store are waited for or refused at this commit, so
    /// the proof reads every node they wrote; later ones read this revision
    /// and are held to the candidate.
    fn publish_candidate_txn(&self, txn: &mut Transaction, schema: &LabelSchema)
    -> StoreResult<()>;

    /// Drop a label (DROP TABLE / DROP label) by retiring its identity on the
    /// transaction. The body revisions are left in place but become
    /// unreachable, so the name resolves to "not declared" from the next read,
    /// and a label created under it later gets a new identity. Applied
    /// atomically at commit.
    fn drop_label_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<()>;

    /// The identity `name` is bound to, read through the transaction; when the
    /// name has no live binding there, the one the store holds now; `None`
    /// when it has neither. Never allocates one.
    ///
    /// The fallback is sound for any view: no record the view holds carries
    /// the name under another identity, since none was live there, and a
    /// write of this transaction that first used the label bound it outside
    /// the transaction, after its view was taken.
    fn label_id_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<Option<LabelId>>;

    /// The identity `name` is bound to, allocating one in the transaction
    /// when it has none: for a catalog transaction declaring a label. Two
    /// attempts allocating for one name, or an allocation racing a label
    /// binding applied meanwhile, both write its binding, so one commits and
    /// the other runs again and finds it. A data write binds a label it uses
    /// for the first time outside its transaction instead.
    fn bind_label_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<LabelId>;

    /// The name identity `id` was allocated for; `None` when no label has
    /// that id. An identity record is written once, so the store's latest
    /// names an identity bound after the transaction's view as truly. A
    /// dropped label's identity keeps its name, so a record written before
    /// the drop still reads with it.
    fn label_name_txn(&self, txn: &mut Transaction, id: LabelId) -> StoreResult<Option<String>>;

    /// [`Self::label_id_txn`] at the latest committed state, for readers
    /// without a transaction.
    fn label_id(&self, name: &str) -> StoreResult<Option<LabelId>>;

    /// [`Self::label_name_txn`] at the latest committed state, for readers
    /// without a transaction.
    fn label_name(&self, id: LabelId) -> StoreResult<Option<String>>;

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

    /// The label that holds the constraint `name`, read through the
    /// transaction; `None` when no label has a constraint of that name.
    fn constraint_label_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
    ) -> StoreResult<Option<String>>;

    /// Record that `label` holds the constraint `name`, on the condition that
    /// no label holds it when the transaction commits: of two statements
    /// taking one name, one commits.
    fn claim_constraint_name_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
        label: &str,
    ) -> StoreResult<()>;

    /// Release the constraint name `name`, on the condition that its record
    /// is still the one read now when the transaction commits.
    fn release_constraint_name_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<()>;
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

    /// Resolve a declaration in an already pinned read view. Does not choose
    /// a fresh snapshot or wait for unrelated commits. The caller retains
    /// the view and records schema dependencies if it later writes.
    pub fn load_label_at(
        &self,
        snapshot: coordinode_storage::engine::StorageSnapshot,
        name: &str,
    ) -> StoreResult<Option<LabelSchema>> {
        let pointer_key = encode_label_current_revision_key(name);
        let Some(binding) = self
            .engine
            .snapshot_get(&snapshot, Partition::Schema, &pointer_key)?
            .map(|bytes| decode_binding(name, &bytes))
            .transpose()?
        else {
            return Ok(None);
        };
        let revision = binding.declared_revision();
        if revision == 0 {
            return Ok(None);
        }
        let schema_key = encode_label_schema_key(binding.id, revision);
        let Some(schema_bytes) =
            self.engine
                .snapshot_get(&snapshot, Partition::Schema, &schema_key)?
        else {
            return Err(missing_revision(name, revision));
        };
        decode_label_body(name, revision, &schema_bytes).map(Some)
    }

    /// The identity `name` is bound to as `snapshot` reads it; when it has no
    /// live binding there, the one the store holds now (see
    /// [`SchemaStore::label_id_txn`]); `None` when it has neither.
    pub fn label_id_at(
        &self,
        snapshot: coordinode_storage::engine::StorageSnapshot,
        name: &str,
    ) -> StoreResult<Option<LabelId>> {
        let at = self
            .engine
            .snapshot_get(
                &snapshot,
                Partition::Schema,
                &encode_label_current_revision_key(name),
            )?
            .map(|bytes| decode_binding(name, &bytes))
            .transpose()?
            .and_then(|b| b.live_id());
        match at {
            Some(id) => Ok(Some(id)),
            None => self.label_id(name),
        }
    }

    /// Label `id` as `snapshot` reads the catalog: the name it was allocated
    /// for, and the definition that name is bound to while the binding is
    /// still `id` (`None` for an implicit, retired or replaced identity).
    ///
    /// # Errors
    ///
    /// `Decode` for an id without an identity record: a stored record names
    /// only allocated labels.
    pub fn identity_at(
        &self,
        snapshot: coordinode_storage::engine::StorageSnapshot,
        id: LabelId,
    ) -> StoreResult<(String, Option<LabelSchema>)> {
        // An identity record is written once, so the latest one is the one
        // any view sees.
        let Some(name) = self.label_name(id)? else {
            return Err(StoreError::Decode {
                kind: "label identity",
                message: format!("label {id} has no identity record"),
            });
        };
        let schema = match self.label_id_at(snapshot, &name)? {
            Some(bound) if bound == id => self.load_label_at(snapshot, &name)?,
            _ => None,
        };
        Ok((name, schema))
    }

    /// What label `id` makes its nodes carry as `snapshot` reads the
    /// catalog: the name it was allocated for and, while that name is bound
    /// to `id` with a definition, its kind and implied labels, in force or
    /// with its candidate in force (`candidate`); as
    /// [`Self::identity_at`] would give them, without the rest of the
    /// definition. Every byte decoded is passed to `charge` before it is
    /// decoded: the name record's, and a bound of what the implications take.
    ///
    /// # Errors
    ///
    /// A storage or decode failure, an id without an identity record, or
    /// what `charge` returns.
    pub fn implication_facts_at(
        &self,
        snapshot: coordinode_storage::engine::StorageSnapshot,
        id: LabelId,
        candidate: bool,
        charge: &mut dyn FnMut(usize) -> StoreResult<()>,
    ) -> StoreResult<(
        String,
        Option<coordinode_core::schema::definition::ImplicationFacts>,
    )> {
        let Some(name) = self.engine.get(Partition::Schema, &label_id_key(id))? else {
            return Err(StoreError::Decode {
                kind: "label identity",
                message: format!("label {id} has no identity record"),
            });
        };
        charge(name.len())?;
        let name = decode_label_name(id, name.to_vec())?;
        if self.label_id_at(snapshot, &name)? != Some(id) {
            return Ok((name, None));
        }
        let pointer_key = encode_label_current_revision_key(&name);
        let Some(binding) = self
            .engine
            .snapshot_get(&snapshot, Partition::Schema, &pointer_key)?
            .map(|bytes| decode_binding(&name, &bytes))
            .transpose()?
        else {
            return Ok((name, None));
        };
        let revision = binding.declared_revision();
        if revision == 0 {
            return Ok((name, None));
        }
        let Some(body) = self.engine.snapshot_get(
            &snapshot,
            Partition::Schema,
            &encode_label_schema_key(binding.id, revision),
        )?
        else {
            return Err(missing_revision(&name, revision));
        };
        let bound =
            LabelSchema::implication_facts_bound(&body).map_err(|e| StoreError::Decode {
                kind: "label schema",
                message: format!("decode failed for '{name}' rev {revision}: {e}"),
            })?;
        charge(bound)?;
        let facts = LabelSchema::implication_facts_from_msgpack(&body, candidate).map_err(|e| {
            StoreError::Decode {
                kind: "label schema",
                message: format!("decode failed for '{name}' rev {revision}: {e}"),
            }
        })?;
        Ok((name, Some(facts)))
    }

    /// What label name `name` was bound to at system time `known_at`: the
    /// last binding its history holds from a commit at or before it; `None`
    /// when no commit had bound the name by then.
    ///
    /// One seek from `known_at` down, at the newest view: neither the bindings
    /// before the one found nor those after `known_at` are read.
    ///
    /// # Errors
    ///
    /// A storage failure, or a history record that does not decode.
    pub fn binding_known_at(
        &self,
        name: &str,
        known_at: coordinode_core::txn::timestamp::Timestamp,
    ) -> StoreResult<Option<LabelBinding>> {
        let prefix = encode_label_binding_history_prefix(name);
        let mut last = prefix.clone();
        last.extend_from_slice(&known_at.as_raw().to_be_bytes());
        let Some(guard) = self
            .engine
            .range_scan_rev(Partition::Schema, &prefix, &last)?
            .next()
        else {
            return Ok(None);
        };
        let (key, value) = guard
            .into_inner()
            .map_err(coordinode_storage::error::StorageError::from)?;
        let undecodable = || StoreError::Decode {
            kind: "label binding history",
            message: format!("a binding of '{name}' in its history does not decode"),
        };
        decode_label_binding_history_commit(name, &key).ok_or_else(undecodable)?;
        LabelBinding::decode(&value)
            .map(Some)
            .ok_or_else(undecodable)
    }

    /// The definition label name `name` was bound to at system time
    /// `known_at`, read from its binding history and the revision that names:
    /// `None` when the name was bound to no definition then (an implicit or
    /// retired binding), and `Ok(None)` too when no commit had bound it.
    ///
    /// # Errors
    ///
    /// A storage failure, a history record or body that does not decode, or
    /// a binding naming a revision whose body is missing.
    pub fn load_label_known_at(
        &self,
        name: &str,
        known_at: coordinode_core::txn::timestamp::Timestamp,
    ) -> StoreResult<Option<LabelSchema>> {
        let Some(binding) = self.binding_known_at(name, known_at)? else {
            return Ok(None);
        };
        let revision = binding.declared_revision();
        if revision == 0 {
            return Ok(None);
        }
        // A revision's body is written once and kept.
        let Some(body) = self.engine.get(
            Partition::Schema,
            &encode_label_schema_key(binding.id, revision),
        )?
        else {
            return Err(missing_revision(name, revision));
        };
        decode_label_body(name, revision, &body).map(Some)
    }

    /// [`Self::implication_facts_at`] as the catalog stood at system time
    /// `known_at`, read from the retained binding history and the revision it
    /// names rather than from an operational snapshot: it answers however
    /// long ago `known_at` is. The facts in force only: a candidate published
    /// then was not yet the interpretation.
    ///
    /// # Errors
    ///
    /// As [`Self::implication_facts_at`]; and
    /// [`StoreError::HistoryUncovered`] when the name of `id` had no binding
    /// in its history by `known_at` although `id` was allocated for it: the
    /// interpretation then was never retained, and is not guessed.
    pub fn implication_facts_known_at(
        &self,
        known_at: coordinode_core::txn::timestamp::Timestamp,
        id: LabelId,
        charge: &mut dyn FnMut(usize) -> StoreResult<()>,
    ) -> StoreResult<(
        String,
        Option<coordinode_core::schema::definition::ImplicationFacts>,
    )> {
        let Some(name) = self.engine.get(Partition::Schema, &label_id_key(id))? else {
            return Err(StoreError::Decode {
                kind: "label identity",
                message: format!("label {id} has no identity record"),
            });
        };
        charge(name.len())?;
        let name = decode_label_name(id, name.to_vec())?;
        let Some(binding) = self.binding_known_at(&name, known_at)? else {
            return Err(StoreError::HistoryUncovered {
                what: format!("the binding of label '{name}' at {}", known_at.as_raw()),
            });
        };
        if binding.live_id() != Some(id) {
            return Ok((name, None));
        }
        let revision = binding.declared_revision();
        if revision == 0 {
            return Ok((name, None));
        }
        // A revision's body is written once and kept, so the newest view
        // holds the one bound then.
        let Some(body) = self.engine.get(
            Partition::Schema,
            &encode_label_schema_key(binding.id, revision),
        )?
        else {
            return Err(missing_revision(&name, revision));
        };
        let decode_error = |e: rmp_serde::decode::Error| StoreError::Decode {
            kind: "label schema",
            message: format!("decode failed for '{name}' rev {revision}: {e}"),
        };
        charge(LabelSchema::implication_facts_bound(&body).map_err(decode_error)?)?;
        let facts =
            LabelSchema::implication_facts_from_msgpack(&body, false).map_err(decode_error)?;
        Ok((name, Some(facts)))
    }

    /// The identity `name` is bound to, binding an implicit one when it has
    /// none, committed at once: for an engine written without a proposal
    /// pipeline (a test, a tool, a direct-mode statement), deciding exactly as
    /// the ordered application of a label binding does.
    pub fn bind_label(&self, name: &str) -> StoreResult<LabelId> {
        if let Some(id) = self.binding(name)?.and_then(|b| b.live_id()) {
            return Ok(id);
        }
        // Written at the engine's latest version, never below a retirement.
        let effects = coordinode_storage::engine::metadata::label_binding_effects(
            self.engine,
            &[name.to_owned()],
            u64::MAX,
        )?;
        let mut batch = WriteBatch::new(self.engine);
        for effect in effects {
            if let coordinode_core::txn::proposal::Mutation::Put { key, value, .. } = effect {
                batch.put(Partition::Schema, key, value);
            }
        }
        batch.commit()?;
        self.binding(name)?
            .and_then(|b| b.live_id())
            .ok_or_else(|| {
                StoreError::Invariant(format!(
                    "label '{name}' was not bound: the binding was refused"
                ))
            })
    }

    /// Every label identity allocated so far with the name it was allocated
    /// for, in id order: what a background pass that names many records'
    /// labels reads once instead of once per label.
    pub fn label_names(&self) -> StoreResult<Vec<(LabelId, String)>> {
        let mut out = Vec::new();
        for guard in self.engine.prefix_scan(
            Partition::Schema,
            coordinode_core::schema::label_id::LABEL_ID_KEY_PREFIX,
        )? {
            let (key, value) = guard.into_inner()?;
            let Some(id) = coordinode_core::schema::label_id::decode_label_id_key(&key) else {
                return Err(StoreError::Decode {
                    kind: "label identity",
                    message: format!("{} bytes past the identity prefix", key.len()),
                });
            };
            out.push((id, decode_label_name(id, value.to_vec())?));
        }
        Ok(out)
    }

    /// The binding of `name` at the latest committed state.
    fn binding(&self, name: &str) -> StoreResult<Option<LabelBinding>> {
        self.engine
            .get(Partition::Schema, &encode_label_current_revision_key(name))?
            .map(|bytes| decode_binding(name, &bytes))
            .transpose()
    }

    /// The revision the pointer at `key` names, as of `snapshot`.
    fn load_revision_pointer(
        &self,
        snapshot: &coordinode_storage::engine::StorageSnapshot,
        key: &[u8],
        kind: &'static str,
    ) -> StoreResult<Option<u64>> {
        let Some(bytes) = self.engine.snapshot_get(snapshot, Partition::Schema, key)? else {
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

/// State that `txn` makes `revision` the schema of `label` (`0` removes it).
fn claim_activation(txn: &mut Transaction, label: &str, revision: u64) {
    use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope};
    let generation = txn.schema_generation();
    txn.claim(Claim::new(
        ClaimScope::LabelSchema(label.to_string()),
        ClaimPredicate::SchemaActivated { revision },
        generation,
    ));
}

/// The binding of `name` from its stored bytes.
fn decode_binding(name: &str, bytes: &[u8]) -> StoreResult<LabelBinding> {
    LabelBinding::decode(bytes).ok_or_else(|| StoreError::Decode {
        kind: "label binding",
        message: format!("binding of '{name}' is not one ({} bytes)", bytes.len()),
    })
}

/// A binding naming a definition revision that is not stored.
fn missing_revision(name: &str, revision: u64) -> StoreError {
    StoreError::Decode {
        kind: "label schema",
        message: format!("binding of '{name}' references missing revision {revision}"),
    }
}

/// A stored label definition.
fn decode_label_body(name: &str, revision: u64, bytes: &[u8]) -> StoreResult<LabelSchema> {
    LabelSchema::from_msgpack(bytes).map_err(|e| StoreError::Decode {
        kind: "label schema",
        message: format!("decode failed for '{name}' rev {revision}: {e}"),
    })
}

/// The encoded definition `schema`.
fn encode_label_body(schema: &LabelSchema) -> StoreResult<Vec<u8>> {
    schema.to_msgpack().map_err(|e| StoreError::Decode {
        kind: "label schema",
        message: format!("encode '{}': {e}", schema.name),
    })
}

/// The binding of `name` as `txn` sees it.
fn binding_txn(txn: &mut Transaction, name: &str) -> StoreResult<Option<LabelBinding>> {
    txn.get(Partition::Schema, &encode_label_current_revision_key(name))?
        .map(|bytes| decode_binding(name, &bytes))
        .transpose()
}

/// Allocate a new identity for `name` on `txn`, a catalog transaction: the
/// identity after the newest issued, as the store holds it now, and its
/// write-once record. An attempt racing another allocation, or a label
/// binding applied meanwhile, writes the same record or the same binding,
/// so it is a write conflict its catalog commit retries, then finding the
/// identity bound.
fn allocate_label_id_txn(
    engine: &StorageEngine,
    txn: &mut Transaction,
    name: &str,
) -> StoreResult<LabelId> {
    use coordinode_storage::engine::metadata::{label_frontier, next_label_id};
    let mut id = next_label_id(label_frontier(engine)?)?;
    // Identities this transaction allocated itself are past the store's
    // newest record: skip them.
    while txn.get(Partition::Schema, &label_id_key(id))?.is_some() {
        id = id
            .next()
            .ok_or_else(|| StoreError::Invariant("the label id space is exhausted".into()))?;
    }
    txn.put(Partition::Schema, &label_id_key(id), name.as_bytes())?;
    Ok(id)
}

/// Stage `schema` as the current definition of its label on `txn`: under
/// the identity its name is bound to, or a new one when the name has none
/// or was dropped.
fn stage_label_definition(
    engine: &StorageEngine,
    txn: &mut Transaction,
    schema: &LabelSchema,
) -> StoreResult<()> {
    let body = encode_label_body(schema)?;
    // The binding is read as allocation reads the newest identity: past the
    // transaction's view when the name was bound since (a first use binds it
    // outside any transaction). Reading the two at different points could
    // give one name a second identity.
    let live = match binding_txn(txn, &schema.name)?.and_then(|b| b.live_id()) {
        Some(id) => Some(id),
        None => LocalSchemaStore::new(engine).label_id(&schema.name)?,
    };
    let id = match live {
        Some(id) => id,
        None => allocate_label_id_txn(engine, txn, &schema.name)?,
    };
    txn.put(
        Partition::Schema,
        &encode_label_schema_key(id, schema.schema_revision),
        &body,
    )?;
    let binding = LabelBinding {
        id,
        revision: schema.schema_revision,
        state: BindingState::Declared,
    };
    bind_name_txn(txn, &schema.name, binding)
}

/// Stage `binding` as what label name `name` is bound to, and keep it in the
/// name's binding history under the commit: the interpretation a retained
/// system time reads is the one bound then, not the one bound now.
fn bind_name_txn(txn: &mut Transaction, name: &str, binding: LabelBinding) -> StoreResult<()> {
    let encoded = binding.encode();
    txn.put(
        Partition::Schema,
        &encode_label_current_revision_key(name),
        &encoded,
    )?;
    txn.put_stamped(
        Partition::Schema,
        &encode_label_binding_history_prefix(name),
        &encoded,
    )?;
    Ok(())
}

/// The name an identity record holds.
fn decode_label_name(id: LabelId, bytes: Vec<u8>) -> StoreResult<String> {
    String::from_utf8(bytes).map_err(|e| StoreError::Decode {
        kind: "label identity",
        message: format!("name of label {id}: {e}"),
    })
}

/// The stored body of edge type `schema`. A temporal type always resolves
/// to some discriminator; one without was handed over before
/// [`EdgeTypeSchema::resolve_identity`] ran, and its writers would have no
/// key shape to agree on, so every writer of the definition refuses it here.
fn edge_type_body(schema: &EdgeTypeSchema) -> StoreResult<Vec<u8>> {
    if schema.temporal && schema.discriminator().is_none() {
        return Err(StoreError::Invariant(format!(
            "edge type '{}' is temporal but its identity was never resolved",
            schema.name
        )));
    }
    schema.to_msgpack().map_err(|e| StoreError::Decode {
        kind: "edge type schema",
        message: format!("encode '{}': {e}", schema.name),
    })
}

/// The writes that keep edge type `schema`'s cardinality profile beside its
/// definition: the profile to store (`None` to remove it), and the
/// constraints the previous profile kept counts for that this definition
/// no longer declares, whose counts and coverage go with it. A constraint
/// declared again later starts from a rebuild, never from counts that
/// stopped changing while it was gone.
struct ProfileWrites {
    profile: Option<Vec<u8>>,
    withdrawn: Vec<coordinode_core::graph::cardinality::CardinalityDescriptor>,
}

fn profile_writes(previous: Option<&[u8]>, schema: &EdgeTypeSchema) -> StoreResult<ProfileWrites> {
    use coordinode_core::graph::cardinality::CardinalityProfile;

    let previous = match previous {
        Some(bytes) => {
            Some(
                CardinalityProfile::from_msgpack(bytes).map_err(|e| StoreError::Decode {
                    kind: "edge type cardinality profile",
                    message: format!("decode '{}': {e}", schema.name),
                })?,
            )
        }
        None => None,
    };
    let profile = schema.cardinality_profile();
    // A count belongs to its descriptor's key, not to its bound: changing
    // only the bound keeps the counts.
    let kept: Vec<Vec<u8>> = profile
        .iter()
        .flat_map(|p| p.descriptors.iter().map(|d| d.counter_prefix()))
        .collect();
    let withdrawn = previous
        .map(|p| p.descriptors)
        .unwrap_or_default()
        .into_iter()
        .filter(|d| !kept.contains(&d.counter_prefix()))
        .collect();
    let profile = profile
        .map(|p| {
            p.to_msgpack().map_err(|e| StoreError::Decode {
                kind: "edge type cardinality profile",
                message: format!("encode '{}': {e}", schema.name),
            })
        })
        .transpose()?;
    Ok(ProfileWrites { profile, withdrawn })
}

/// The exclusive end of the counts under `prefix`: each is the prefix and an
/// eight-byte node, so nine 0xFF bytes past the prefix are beyond them all.
fn counts_end(prefix: &[u8]) -> Vec<u8> {
    let mut end = prefix.to_vec();
    end.extend_from_slice(&[0xFF; 9]);
    end
}

impl SchemaStore for LocalSchemaStore<'_> {
    fn load_label(&self, name: &str) -> StoreResult<Option<LabelSchema>> {
        // The pointer and the body it names are read from one complete
        // snapshot: a revision committing between two latest-state reads
        // could show its pointer and not yet its body.
        self.load_label_at(self.engine.snapshot(), name)
    }

    fn save_label(&self, schema: &LabelSchema) -> StoreResult<()> {
        let body = encode_label_body(schema)?;
        let mut batch = WriteBatch::new(self.engine);
        let id = match self.binding(&schema.name)?.and_then(|b| b.live_id()) {
            Some(id) => id,
            None => {
                // The batch commits unconditionally: this path serves a
                // single writer (a test, a tool) and is not raced.
                use coordinode_storage::engine::metadata::{label_frontier, next_label_id};
                let id = next_label_id(label_frontier(self.engine)?)?;
                batch.put(
                    Partition::Schema,
                    label_id_key(id).to_vec(),
                    schema.name.as_bytes().to_vec(),
                );
                id
            }
        };
        batch.put(
            Partition::Schema,
            encode_label_schema_key(id, schema.schema_revision),
            body,
        );
        let binding = LabelBinding {
            id,
            revision: schema.schema_revision,
            state: BindingState::Declared,
        };
        batch.put(
            Partition::Schema,
            encode_label_current_revision_key(&schema.name),
            binding.encode().to_vec(),
        );
        batch.commit()?;
        Ok(())
    }

    fn load_edge_type(&self, name: &str) -> StoreResult<Option<EdgeTypeSchema>> {
        // One snapshot for the pointer and the body, as in `load_label`.
        let snapshot = self.engine.snapshot();
        let pointer_key = encode_edge_type_current_revision_key(name);
        let Some(revision) =
            self.load_revision_pointer(&snapshot, &pointer_key, "edge type revision pointer")?
        else {
            return Ok(None);
        };
        let schema_key = encode_edge_type_schema_key(name, revision);
        let Some(schema_bytes) =
            self.engine
                .snapshot_get(&snapshot, Partition::Schema, &schema_key)?
        else {
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
        use coordinode_core::graph::cardinality::encode_cardinality_profile_key;

        let body = edge_type_body(schema)?;
        let profile_key = encode_cardinality_profile_key(&schema.name);
        let previous = self.engine.get(Partition::Schema, &profile_key)?;
        let profile = profile_writes(previous.as_deref(), schema)?;
        let mut batch = WriteBatch::new(self.engine);
        for withdrawn in &profile.withdrawn {
            let prefix = withdrawn.counter_prefix();
            batch.remove_range(Partition::Counter, prefix.clone(), counts_end(&prefix));
            batch.delete(Partition::Schema, withdrawn.coverage_key());
        }
        match profile.profile {
            Some(bytes) => batch.put(Partition::Schema, profile_key, bytes),
            None => batch.delete(Partition::Schema, profile_key),
        }
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
        // A schema this attempt staged itself is covered by its own
        // activation; one it read from the store is what its writes are
        // validated under, and has to still be there when they land.
        // A binding this attempt allocated itself conflicts with any other
        // writer of it, a declaration included.
        let own_pointer = txn.buffered(Partition::Schema, &pointer_key).is_some();
        let binding = binding_txn(txn, name)?;
        let revision = binding.map_or(0, |b| b.declared_revision());
        let Some(binding) = binding.filter(|_| revision != 0) else {
            if !own_pointer {
                txn.note_label_schema_read(name, 0);
            }
            return Ok(None);
        };
        let schema_key = encode_label_schema_key(binding.id, revision);
        if !own_pointer && txn.buffered(Partition::Schema, &schema_key).is_none() {
            txn.note_label_schema_read(name, revision);
        }
        let Some(schema_bytes) = txn.get(Partition::Schema, &schema_key)? else {
            return Err(missing_revision(name, revision));
        };
        let schema = decode_label_body(name, revision, &schema_bytes)?;
        // A write under a schema whose candidate is being proved is held to
        // the candidate as well, since the proof never sees it.
        if !own_pointer && schema.candidate().is_some() {
            txn.note_label_candidate_read(name, revision);
        }
        Ok(Some(schema))
    }

    fn load_label_for_update_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
    ) -> StoreResult<Option<LabelSchema>> {
        let pointer_key = encode_label_current_revision_key(name);
        // Every change of a label's schema writes its pointer. The version is
        // read before the schema: a change landing between the two reads
        // refuses the commit instead of passing under the older schema.
        let version = txn.record_version(Partition::Schema, &pointer_key)?;
        let current = self.load_label(name)?;
        txn.expect_version(Partition::Schema, &pointer_key, version)?;
        Ok(current)
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
        stage_label_definition(self.engine, txn, schema)?;
        // A definition changed, so predicates other attempts evaluated under
        // the old one stop counting as evidence once this lands.
        txn.note_schema_change();
        // The new schema governs the nodes already stored, and the commit
        // checks them against it with no write under the old one beside it.
        claim_activation(txn, &schema.name, schema.schema_revision);
        Ok(())
    }

    fn save_label_admitting_txn(
        &self,
        txn: &mut Transaction,
        schema: &LabelSchema,
    ) -> StoreResult<()> {
        stage_label_definition(self.engine, txn, schema)?;
        txn.note_schema_change();
        Ok(())
    }

    fn publish_candidate_txn(
        &self,
        txn: &mut Transaction,
        schema: &LabelSchema,
    ) -> StoreResult<()> {
        use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope};
        self.save_label_admitting_txn(txn, schema)?;
        let generation = txn.schema_generation();
        txn.claim(Claim::new(
            ClaimScope::LabelSchema(schema.name.clone()),
            ClaimPredicate::CandidatePublished {
                revision: schema.schema_revision,
            },
            generation,
        ));
        Ok(())
    }

    fn drop_label_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<()> {
        if let Some(binding) = binding_txn(txn, name)?.filter(|b| b.live_id().is_some()) {
            let retired = LabelBinding {
                state: BindingState::Retired,
                ..binding
            };
            bind_name_txn(txn, name, retired)?;
        }
        txn.note_schema_change();
        claim_activation(txn, name, 0);
        Ok(())
    }

    fn label_id_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<Option<LabelId>> {
        match binding_txn(txn, name)?.and_then(|b| b.live_id()) {
            Some(id) => Ok(Some(id)),
            None => self.label_id(name),
        }
    }

    fn bind_label_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<LabelId> {
        if let Some(id) = self.label_id_txn(txn, name)? {
            return Ok(id);
        }
        // An implicit binding declares nothing, so it does not move the
        // schema generation: it changes no predicate another attempt holds.
        let id = allocate_label_id_txn(self.engine, txn, name)?;
        let binding = LabelBinding {
            id,
            revision: 0,
            state: BindingState::Implicit,
        };
        bind_name_txn(txn, name, binding)?;
        Ok(id)
    }

    fn label_name_txn(&self, txn: &mut Transaction, id: LabelId) -> StoreResult<Option<String>> {
        match txn.get(Partition::Schema, &label_id_key(id))? {
            Some(bytes) => decode_label_name(id, bytes).map(Some),
            // Written once, so the latest record names an identity bound
            // after the transaction's view just as truly.
            None => self.label_name(id),
        }
    }

    fn label_id(&self, name: &str) -> StoreResult<Option<LabelId>> {
        Ok(self.binding(name)?.and_then(|b| b.live_id()))
    }

    fn label_name(&self, id: LabelId) -> StoreResult<Option<String>> {
        self.engine
            .get(Partition::Schema, &label_id_key(id))?
            .map(|bytes| decode_label_name(id, bytes.to_vec()))
            .transpose()
    }

    fn save_edge_type_txn(
        &self,
        txn: &mut Transaction,
        schema: &EdgeTypeSchema,
    ) -> StoreResult<()> {
        use coordinode_core::graph::cardinality::encode_cardinality_profile_key;

        let body = edge_type_body(schema)?;
        let profile_key = encode_cardinality_profile_key(&schema.name);
        let previous = txn.get(Partition::Schema, &profile_key)?;
        let profile = profile_writes(previous.as_deref(), schema)?;
        for withdrawn in &profile.withdrawn {
            let prefix = withdrawn.counter_prefix();
            txn.remove_range(Partition::Counter, &prefix, &counts_end(&prefix))?;
            txn.delete(Partition::Schema, &withdrawn.coverage_key())?;
        }
        match profile.profile {
            Some(bytes) => txn.put(Partition::Schema, &profile_key, &bytes)?,
            None => txn.delete(Partition::Schema, &profile_key)?,
        }
        let schema_key = encode_edge_type_schema_key(&schema.name, schema.schema_revision);
        txn.put(Partition::Schema, &schema_key, &body)?;
        let pointer_key = encode_edge_type_current_revision_key(&schema.name);
        txn.put(
            Partition::Schema,
            &pointer_key,
            &schema.schema_revision.to_be_bytes(),
        )?;
        txn.note_schema_change();
        Ok(())
    }

    /// Records that a type has been seen, which declares nothing about the
    /// shapes it admits, so it deliberately does not move the schema
    /// generation: invalidating other attempts' evidence for a marker that
    /// constrains nothing would cost refusals and buy nothing.
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
        let mut types: Vec<String> = Vec::new();
        for (k, _) in txn.prefix_scan(Partition::Schema, EDGE_TYPE_SCHEMA_KEY_PREFIX)? {
            if let Some(name) = decode_edge_type_schema_key_name(&k)
                && !types.iter().any(|t| t == name)
            {
                types.push(name.to_string());
            }
        }
        Ok(types)
    }

    fn list_edge_type_names_engine(&self) -> StoreResult<Vec<String>> {
        let mut types: Vec<String> = Vec::new();
        for guard in self
            .engine
            .prefix_scan(Partition::Schema, EDGE_TYPE_SCHEMA_KEY_PREFIX)?
        {
            let (k, _) = guard.into_inner()?;
            if let Some(name) = decode_edge_type_schema_key_name(&k)
                && !types.iter().any(|t| t == name)
            {
                types.push(name.to_string());
            }
        }
        Ok(types)
    }

    fn constraint_label_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
    ) -> StoreResult<Option<String>> {
        let Some(bytes) = txn.get(Partition::Schema, &encode_constraint_name_key(name))? else {
            return Ok(None);
        };
        String::from_utf8(bytes)
            .map(Some)
            .map_err(|e| StoreError::Decode {
                kind: "constraint name",
                message: format!("label of constraint '{name}': {e}"),
            })
    }

    fn claim_constraint_name_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
        label: &str,
    ) -> StoreResult<()> {
        let key = encode_constraint_name_key(name);
        txn.expect_version(Partition::Schema, &key, None)?;
        txn.put(Partition::Schema, &key, label.as_bytes())?;
        Ok(())
    }

    fn release_constraint_name_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<()> {
        let key = encode_constraint_name_key(name);
        let version = self.engine.record_version(Partition::Schema, &key)?;
        txn.expect_version(Partition::Schema, &key, version)?;
        txn.delete(Partition::Schema, &key)?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
