//! Index registry: the B-tree index definitions in force, and the
//! maintenance of their entries as nodes are created, changed and deleted.
//!
//! Every entry is staged in the writing statement's transaction (see
//! [`coordinode_modality::IndexStore`]), so it commits, replicates and rolls
//! back with the data it indexes.

use std::sync::Arc;

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_core::index::derive::EntryOwner;
use coordinode_modality::{IndexStore as _, LocalIndexStore, StoreError};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_storage::error::StorageError;
use rustc_hash::FxHashMap;

use super::definition::{
    GenerationId, IndexDefinition, IndexId, IndexIntegrityRecord, IndexType, Integrity, Mismatch,
};

/// Registry of active indexes.
///
/// Interior mutability lets DDL executed inside a statement update the live
/// registry through the shared reference the execution context holds.
pub struct IndexRegistry {
    indexes: parking_lot::RwLock<Catalog>,
    /// Generations whose entries were found disagreeing with the records
    /// they name: none of them proves a value free or a result complete, so
    /// lookups answer from the records and every unique value taken in them
    /// is proved free against the whole label.
    integrity: parking_lot::Mutex<IntegrityView>,
    /// Whether any generation is suspect: the one check a healthy lookup
    /// pays.
    any_suspect: core::sync::atomic::AtomicBool,
    /// Signalled when a disagreement is found, for the maintenance that
    /// records it.
    reported: parking_lot::Condvar,
}

/// The suspect generations as this process knows them.
#[derive(Default)]
struct IntegrityView {
    /// Suspect per the catalog's integrity records.
    recorded: rustc_hash::FxHashSet<GenerationId>,
    /// Found suspect here and not recorded in the catalog yet: a member
    /// whose report cannot be recorded (one that takes no writes) keeps
    /// answering from the records.
    local: rustc_hash::FxHashSet<GenerationId>,
    /// Verified generations that were once damaged, with the earliest
    /// snapshot they answer for.
    trusted_from: FxHashMap<GenerationId, u64>,
    /// Disagreements found here, waiting to be recorded.
    reports: Vec<IntegrityReport>,
}

/// A disagreement between an entry and a record, found by a read or a write
/// and waiting to be recorded in the catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityReport {
    /// The logical index.
    pub index: IndexId,
    /// The generation whose entry disagrees.
    pub generation: GenerationId,
    /// What was found.
    pub found: Mismatch,
}

/// The most reports waiting to be recorded; past it new ones are dropped,
/// the generation stays suspect here, and the check its first report
/// starts finds the rest.
const MAX_PENDING_REPORTS: usize = 1024;

/// The indexes in force, by identity, and the names that bind them.
#[derive(Default)]
struct Catalog {
    by_id: FxHashMap<IndexId, Registered>,
    by_name: FxHashMap<String, IndexId>,
}

impl Catalog {
    fn insert(&mut self, registered: Registered) {
        let id = registered.def.id;
        if let Some(old) = self.by_id.insert(id, registered) {
            self.unbind(&old.def);
        }
        if let Some(name) = &self.by_id[&id].def.name {
            self.by_name.insert(name.clone(), id);
        }
    }

    fn remove(&mut self, id: IndexId) {
        if let Some(old) = self.by_id.remove(&id) {
            self.unbind(&old.def);
        }
    }

    /// Drop `def`'s name binding while it is still `def`'s.
    fn unbind(&mut self, def: &IndexDefinition) {
        if let Some(name) = &def.name {
            if self.by_name.get(name) == Some(&def.id) {
                self.by_name.remove(name);
            }
        }
    }
}

/// A definition in force, with the version of its stored record when this
/// member read it; `None` for one registered without a stored record.
/// Shared, so the maintenance a write runs takes the definitions it needs
/// without copying them.
#[derive(Debug, Clone)]
struct Registered {
    def: Arc<IndexDefinition>,
    version: Option<u64>,
}

/// A unique value a statement claimed. Kept until the statement commits, so
/// a claim that loses a race can name the node that won it.
#[derive(Debug, Clone)]
pub struct UniqueClaim {
    /// The unique index.
    pub index: IndexDefinition,
    /// The values claimed, one per indexed property.
    pub values: Vec<Value>,
    /// The node that claimed them.
    pub node_id: NodeId,
    /// The index's entries cannot prove the values free: the commit reads
    /// the label's stored nodes for another holder instead.
    pub needs_source_proof: bool,
}

/// Properties of one node changing value: what the index maintenance of a
/// SET, REMOVE or node merge needs to know.
pub struct PropertyChange<'a> {
    /// The node.
    pub node_id: NodeId,
    /// The `valid_from` of the temporal node's version that changes; `None`
    /// for a node that is not temporal.
    pub valid_from: Option<i64>,
    /// Its primary label.
    pub label: &'a str,
    /// The properties that change.
    pub properties: &'a [&'a str],
    /// The node's properties before the change.
    pub before: &'a dyn Fn(&str) -> Option<Value>,
    /// The node's properties after the change.
    pub after: &'a dyn Fn(&str) -> Option<Value>,
}

/// A node, or one version of a temporal node, as it is created, or as it was
/// before it is deleted.
pub struct NodeState<'a> {
    /// The node.
    pub node_id: NodeId,
    /// The version's `valid_from` for a temporal node; `None` for a node
    /// that is not temporal.
    pub valid_from: Option<i64>,
    /// Its primary label.
    pub label: &'a str,
    /// Its properties.
    pub value_of: &'a dyn Fn(&str) -> Option<Value>,
}

impl NodeState<'_> {
    /// Whose entries the state holds: the node, or its version.
    fn owner(&self) -> EntryOwner {
        EntryOwner {
            node_id: self.node_id.as_raw(),
            valid_from: self.valid_from,
        }
    }
}

/// The field id a property name is bound to now, which a DERIVED effect is
/// sealed with.
pub type FieldOf<'a> = &'a dyn Fn(&str) -> Option<u32>;

/// A write that would give a unique index's value a second holder.
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "unique constraint violated on index `{index_name}`: property `{property}` already has \
     value {value:?} (node {})",
    .holder.to_element_id()
)]
pub struct UniqueViolation {
    /// The unique index.
    pub index_name: String,
    /// The indexed properties, comma-separated.
    pub property: String,
    /// The value, or the list of values of a compound index.
    pub value: Value,
    /// The node that holds the value.
    pub holder: NodeId,
}

/// Why maintaining an index failed.
#[derive(Debug, thiserror::Error)]
pub enum IndexWriteError {
    /// The write would break a unique index.
    #[error(transparent)]
    Unique(#[from] UniqueViolation),
    /// The index store failed.
    #[error(transparent)]
    Store(#[from] StoreError),
}

impl UniqueViolation {
    /// The violation of `index` by `values`, held by `holder`.
    pub fn new(index: &IndexDefinition, values: &[Value], holder: NodeId) -> Self {
        Self {
            index_name: index.to_string(),
            property: index.properties.join(","),
            value: match values {
                [one] => one.clone(),
                many => Value::Array(many.to_vec()),
            },
            holder,
        }
    }
}

/// The values `index` holds for a node whose properties `value_of` answers,
/// or `None` when the node has no entry in it: a sparse index skips a node
/// missing any indexed property, a partial index one its filter rejects.
fn entry_values(
    index: &IndexDefinition,
    value_of: &dyn Fn(&str) -> Option<Value>,
) -> Option<Vec<Value>> {
    let values: Vec<Value> = index
        .properties
        .iter()
        .map(|p| value_of(p).unwrap_or(Value::Null))
        .collect();
    if index.sparse && values.iter().any(Value::is_null) {
        return None;
    }
    if let Some(filter) = &index.filter {
        let property = filter.property();
        let tested = [(
            property.to_string(),
            value_of(property).unwrap_or(Value::Null),
        )];
        if !filter.matches(&tested) {
            return None;
        }
    }
    Some(values)
}

/// A property lookup over a list of `(name, value)` pairs.
pub fn props_lookup(props: &[(String, Value)]) -> impl Fn(&str) -> Option<Value> + '_ {
    move |name| {
        props
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    }
}

impl IndexRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            indexes: parking_lot::RwLock::new(Catalog::default()),
            integrity: parking_lot::Mutex::new(IntegrityView::default()),
            any_suspect: core::sync::atomic::AtomicBool::new(false),
            reported: parking_lot::Condvar::new(),
        }
    }

    /// Record that an entry of `index` was found disagreeing with the record
    /// it names, as `found` says: from now on its generation proves no
    /// lookup complete and no unique value free in this process, and the
    /// report waits for the maintenance that records it in the catalog and
    /// checks the generation. A read or a write that finds one reports it;
    /// neither changes an entry or a record for it.
    pub fn report_mismatch(&self, index: &IndexDefinition, found: Mismatch) {
        let mut view = self.integrity.lock();
        if view.local.insert(index.generation) {
            tracing::warn!(
                index = %index,
                generation = index.generation.as_raw(),
                ?found,
                "an index entry disagrees with the record it names; lookups in this \
                 index generation answer from the stored records until it is checked"
            );
        }
        let report = IntegrityReport {
            index: index.id,
            generation: index.generation,
            found,
        };
        if view.reports.len() < MAX_PENDING_REPORTS && !view.reports.contains(&report) {
            view.reports.push(report);
        }
        self.any_suspect
            .store(true, core::sync::atomic::Ordering::Release);
        self.reported.notify_all();
    }

    /// Whether an entry of `generation` is known to disagree with its record,
    /// here or per the catalog.
    pub fn is_suspect(&self, generation: GenerationId) -> bool {
        if !self.any_suspect.load(core::sync::atomic::Ordering::Acquire) {
            return false;
        }
        let view = self.integrity.lock();
        view.recorded.contains(&generation) || view.local.contains(&generation)
    }

    /// The reports waiting to be recorded, taken from the queue, waiting up
    /// to `timeout` for one when there is none.
    pub fn take_reports(&self, timeout: core::time::Duration) -> Vec<IntegrityReport> {
        let mut view = self.integrity.lock();
        if view.reports.is_empty() {
            self.reported.wait_for(&mut view, timeout);
        }
        core::mem::take(&mut view.reports)
    }

    /// Wake whoever waits in [`Self::take_reports`], as a shutdown does.
    pub fn wake_reporters(&self) {
        let _view = self.integrity.lock();
        self.reported.notify_all();
    }

    /// Bring the suspect set in line with the catalog's integrity
    /// `records`: a generation recorded suspect stays so until a check
    /// verifies it. A suspicion found here is about this member's copy and
    /// is left alone: a verified record proves the copy of the member that
    /// checked.
    pub fn apply_integrity(&self, records: &[IndexIntegrityRecord]) {
        let mut view = self.integrity.lock();
        view.recorded = records
            .iter()
            .filter(|r| r.integrity == Integrity::Suspect)
            .map(|r| r.generation)
            .collect();
        view.trusted_from = records
            .iter()
            .filter(|r| r.integrity == Integrity::Verified)
            .filter_map(|r| r.trusted_from.map(|ts| (r.generation, ts)))
            .collect();
        self.refresh_any_suspect(&view);
    }

    /// Generations this member stored as found disagreeing in its own copy,
    /// as a restart reads them back.
    pub fn found_unfit_here(&self, generations: &[GenerationId]) {
        if generations.is_empty() {
            return;
        }
        let mut view = self.integrity.lock();
        view.local.extend(generations.iter().copied());
        self.refresh_any_suspect(&view);
    }

    /// A check that ran on this member verified `generation` here: the
    /// suspicion this member found in its copy is lifted.
    pub fn verified_here(&self, generation: GenerationId) {
        let mut view = self.integrity.lock();
        view.local.remove(&generation);
        self.refresh_any_suspect(&view);
    }

    /// Forget suspicions found here in generations other than `live`: a
    /// generation that left the catalog was replaced or dropped, and its
    /// copy with it.
    pub fn retain_generations(&self, live: &[GenerationId]) {
        let mut view = self.integrity.lock();
        view.local.retain(|g| live.contains(g));
        self.refresh_any_suspect(&view);
    }

    fn refresh_any_suspect(&self, view: &IntegrityView) {
        self.any_suspect.store(
            !view.recorded.is_empty() || !view.local.is_empty() || !view.trusted_from.is_empty(),
            core::sync::atomic::Ordering::Release,
        );
    }

    /// Whether `generation` proves a lookup complete for a read at
    /// `read_ts`: not suspect, and, when it was once damaged and verified
    /// since, not read at a snapshot older than its verification.
    pub fn answers_at(&self, generation: GenerationId, read_ts: u64) -> bool {
        if !self.any_suspect.load(core::sync::atomic::Ordering::Acquire) {
            return true;
        }
        let view = self.integrity.lock();
        if view.recorded.contains(&generation) || view.local.contains(&generation) {
            return false;
        }
        view.trusted_from
            .get(&generation)
            .is_none_or(|from| read_ts >= *from)
    }

    /// Make `index` active in this process without a stored record to bind
    /// writers to: for a context that has none.
    pub fn register_in_memory(&self, index: IndexDefinition) {
        self.insert(index, None);
    }

    /// Make `index` active in this process as its stored record now stands,
    /// after the statement that published it: writers bind their effects to
    /// that record's version.
    ///
    /// # Errors
    ///
    /// A storage failure reading the record's version.
    pub fn register_published(
        &self,
        engine: &StorageEngine,
        index: IndexDefinition,
    ) -> Result<(), StorageError> {
        let version = engine.record_version(
            coordinode_storage::engine::partition::Partition::Schema,
            &index.schema_key(),
        )?;
        self.insert(index, version);
        Ok(())
    }

    fn insert(&self, def: IndexDefinition, version: Option<u64>) {
        self.indexes.write().insert(Registered {
            def: Arc::new(def),
            version,
        });
    }

    /// Stop maintaining the index `id` in this process.
    pub fn unregister(&self, id: IndexId) {
        self.indexes.write().remove(id);
    }

    /// Replace the active set with the definitions stored in the schema
    /// partition, and the suspect generations with the catalog's integrity
    /// records. A member that applied another member's CREATE or DROP INDEX,
    /// or a check's outcome, picks it up here. Every type is listed, so
    /// planning and advice see every index; only B-tree indexes have entries
    /// maintained here.
    pub fn load_all(&self, engine: &StorageEngine) -> Result<(), StorageError> {
        let store = LocalIndexStore::new(engine);
        let storage = |e| match e {
            StoreError::Storage(e) => e,
            other => StorageError::Serialization(other.to_string()),
        };
        let records = store.list_integrity().map_err(storage)?;
        self.apply_integrity(&records);
        let defs = super::ops::list_index_definitions(engine)?;
        let live: Vec<GenerationId> = defs.iter().map(|d| d.generation).collect();
        // This member's own marks survive a restart; those of generations
        // that left the catalog went with their copies.
        let (kept, gone): (Vec<_>, Vec<_>) = store
            .list_unfit_here()
            .map_err(storage)?
            .into_iter()
            .partition(|g| live.contains(g));
        store.clear_unfit_here(&gone).map_err(storage)?;
        self.found_unfit_here(&kept);
        self.retain_generations(&live);
        let mut loaded = Catalog::default();
        for def in defs {
            let version = engine.record_version(
                coordinode_storage::engine::partition::Partition::Schema,
                &def.schema_key(),
            )?;
            loaded.insert(Registered {
                def: Arc::new(def),
                version,
            });
        }
        *self.indexes.write() = loaded;
        Ok(())
    }

    /// The B-tree indexes on `label`: the ones whose entries this registry
    /// maintains.
    fn btree_for_label(&self, label: &str) -> Vec<Registered> {
        self.indexes
            .read()
            .by_id
            .values()
            .filter(|r| r.def.label == label && r.def.index_type == IndexType::BTree)
            .cloned()
            .collect()
    }

    /// The indexes matching `keep` (owned clones).
    fn defs_where(&self, keep: impl Fn(&IndexDefinition) -> bool) -> Vec<IndexDefinition> {
        self.indexes
            .read()
            .by_id
            .values()
            .filter(|r| keep(&r.def))
            .map(|r| IndexDefinition::clone(&r.def))
            .collect()
    }

    /// Get all indexes for a specific label (returns owned clones).
    pub fn indexes_for_label(&self, label: &str) -> Vec<IndexDefinition> {
        self.defs_where(|idx| idx.label == label)
    }

    /// Get all indexes that cover a specific label + property (returns owned clones).
    pub fn indexes_for_property(&self, label: &str, property: &str) -> Vec<IndexDefinition> {
        self.defs_where(|idx| idx.label == label && idx.properties.iter().any(|p| p == property))
    }

    /// The index the name `name` binds (owned clone).
    pub fn get(&self, name: &str) -> Option<IndexDefinition> {
        let catalog = self.indexes.read();
        let id = catalog.by_name.get(name)?;
        catalog
            .by_id
            .get(id)
            .map(|r| IndexDefinition::clone(&r.def))
    }

    /// The index `id` (owned clone).
    pub fn get_by_id(&self, id: IndexId) -> Option<IndexDefinition> {
        self.indexes
            .read()
            .by_id
            .get(&id)
            .map(|r| IndexDefinition::clone(&r.def))
    }

    /// Every active index (owned clones).
    pub fn all(&self) -> Vec<IndexDefinition> {
        self.defs_where(|_| true)
    }

    /// Whether any B-tree index on `label` reads `property`, as an indexed
    /// value or as its partial filter.
    pub fn reads_property(&self, label: &str, property: &str) -> bool {
        self.btree_for_label(label).iter().any(|r| {
            r.def.properties.iter().any(|p| p == property)
                || r.def
                    .filter
                    .as_ref()
                    .is_some_and(|f| f.property() == property)
        })
    }

    /// Check if any index exists for a label.
    pub fn has_indexes_for(&self, label: &str) -> bool {
        self.indexes
            .read()
            .by_id
            .values()
            .any(|r| r.def.label == label)
    }

    /// Whether any B-tree index, whose entries a write maintains, exists for
    /// a label. The check a write makes before any other index work.
    pub fn has_btree_for(&self, label: &str) -> bool {
        self.indexes
            .read()
            .by_id
            .values()
            .any(|r| r.def.label == label && r.def.index_type == IndexType::BTree)
    }

    /// Number of registered indexes.
    pub fn len(&self) -> usize {
        self.indexes.read().by_id.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.indexes.read().by_id.is_empty()
    }

    /// Stage the entries of a node of shard `shard_id` being created. A
    /// unique value another node holds refuses the write; each unique value
    /// claimed is appended to `claims`.
    pub fn on_node_created(
        &self,
        engine: &StorageEngine,
        txn: &mut Transaction,
        shard_id: u16,
        node: &NodeState<'_>,
        field_of: FieldOf<'_>,
        claims: &mut Vec<UniqueClaim>,
    ) -> Result<(), IndexWriteError> {
        for Registered {
            def: index,
            version,
        } in self.btree_for_label(node.label)
        {
            let Some(values) = entry_values(&index, node.value_of) else {
                continue;
            };
            bind(txn, &index, version)?;
            let staging = Staging {
                engine,
                shard_id,
                field_of,
                registry: Some(self),
            };
            stage(
                &staging,
                txn,
                &index,
                node.owner(),
                None,
                Some(values),
                claims,
            )?;
        }
        Ok(())
    }

    /// Move the entries of a node whose property changes as `change`
    /// describes. Only the indexes that read the property are touched.
    pub fn on_property_changed(
        &self,
        engine: &StorageEngine,
        txn: &mut Transaction,
        shard_id: u16,
        change: &PropertyChange<'_>,
        field_of: FieldOf<'_>,
        claims: &mut Vec<UniqueClaim>,
    ) -> Result<(), IndexWriteError> {
        for Registered {
            def: index,
            version,
        } in self.btree_for_label(change.label)
        {
            let changed = |p: &str| change.properties.contains(&p);
            let reads = index.properties.iter().any(|p| changed(p))
                || index.filter.as_ref().is_some_and(|f| changed(f.property()));
            if !reads {
                continue;
            }
            let old = entry_values(&index, change.before);
            let new = entry_values(&index, change.after);
            if old == new {
                continue;
            }
            bind(txn, &index, version)?;
            let owner = EntryOwner {
                node_id: change.node_id.as_raw(),
                valid_from: change.valid_from,
            };
            let staging = Staging {
                engine,
                shard_id,
                field_of,
                registry: Some(self),
            };
            stage(&staging, txn, &index, owner, old, new, claims)?;
        }
        Ok(())
    }

    /// Stage the removal of the entries of a node being deleted.
    pub fn on_node_deleted(
        &self,
        engine: &StorageEngine,
        txn: &mut Transaction,
        node: &NodeState<'_>,
        field_of: FieldOf<'_>,
    ) -> Result<(), StoreError> {
        let store = LocalIndexStore::new(engine);
        for Registered {
            def: index,
            version,
        } in self.btree_for_label(node.label)
        {
            if let Some(values) = entry_values(&index, node.value_of) {
                bind(txn, &index, version)?;
                store.stage_membership(txn, &index, field_of, node.owner(), Some(&values), None)?;
            }
        }
        Ok(())
    }
}

/// Bind the writing transaction's effects in `index` to the definition
/// record this member read, when it read one: a transition, drop or rebuild
/// of the index before the commit refuses the write, to be retried under
/// the binding in force.
fn bind(
    txn: &mut Transaction,
    index: &IndexDefinition,
    version: Option<u64>,
) -> Result<(), StoreError> {
    if version.is_some() {
        txn.bind_index_definition(&index.schema_key(), version)?;
    }
    Ok(())
}

/// Stage the entry `index` holds for `owner` (a node, or one version of a
/// temporal node, of shard `shard_id`) whose properties `value_of` answers,
/// if it has one there, and say whether it did. A unique value another node
/// holds refuses the write; a unique value claimed is appended to `claims`.
#[allow(clippy::too_many_arguments)]
pub fn stage_node_entry(
    engine: &StorageEngine,
    txn: &mut Transaction,
    shard_id: u16,
    index: &IndexDefinition,
    owner: EntryOwner,
    value_of: &dyn Fn(&str) -> Option<Value>,
    field_of: FieldOf<'_>,
    claims: &mut Vec<UniqueClaim>,
) -> Result<bool, IndexWriteError> {
    match entry_values(index, value_of) {
        Some(values) => stage(
            &Staging {
                engine,
                shard_id,
                field_of,
                registry: None,
            },
            txn,
            index,
            owner,
            None,
            Some(values),
            claims,
        ),
        None => Ok(false),
    }
}

/// A property lookup over a stored node: declared properties through the
/// field dictionary, then the undeclared ones kept by name.
pub fn record_lookup<'r>(
    record: &'r coordinode_core::graph::node::NodeRecord,
    interner: &'r coordinode_core::graph::intern::FieldInterner,
) -> impl Fn(&str) -> Option<Value> + 'r {
    move |name| {
        interner
            .lookup(name)
            .and_then(|field| record.get(field).cloned())
            .or_else(|| record.get_extra(name).cloned())
    }
}

/// What staging an entry reads besides the entry: the store, the shard its
/// nodes live in, the field dictionary, and the registry that keeps which
/// generations are suspect (`None` for a build filling a generation no
/// reader uses yet).
struct Staging<'a> {
    engine: &'a StorageEngine,
    shard_id: u16,
    field_of: FieldOf<'a>,
    registry: Option<&'a IndexRegistry>,
}

/// Stage `owner`'s membership in `index` moving from `old` to `new`,
/// refusing a unique value another node holds, and say whether an entry was
/// put.
///
/// A unique entry names the value's holder, and the holder is checked
/// against its own record before the value is refused: an entry naming a
/// node that does not hold the value is wrong. That marks the generation
/// suspect and leaves the value to be proved free from the stored nodes
/// when the statement commits; a wrong entry is never taken as proof that
/// the value is free, nor as a duplicate of the node it names.
fn stage(
    staging: &Staging<'_>,
    txn: &mut Transaction,
    index: &IndexDefinition,
    owner: EntryOwner,
    old: Option<Vec<Value>>,
    new: Option<Vec<Value>>,
    claims: &mut Vec<UniqueClaim>,
) -> Result<bool, IndexWriteError> {
    let node_id = NodeId::from_raw(owner.node_id);
    let store = LocalIndexStore::new(staging.engine);
    let mut needs_source_proof = false;
    if index.unique {
        if let Some(values) = &new {
            needs_source_proof = staging
                .registry
                .is_some_and(|r| r.is_suspect(index.generation));
            if let Some(holder) = store.unique_conflict(txn, index, values, node_id)? {
                if holds_values(
                    txn,
                    staging.shard_id,
                    index,
                    staging.field_of,
                    holder,
                    values,
                )? {
                    return Err(UniqueViolation::new(index, values, holder).into());
                }
                if let Some(registry) = staging.registry {
                    for tuple in held_tuples(&store, txn, index, values, holder)? {
                        registry.report_mismatch(
                            index,
                            Mismatch::Extra {
                                node: holder.as_raw(),
                                valid_from: None,
                                tuple,
                            },
                        );
                    }
                }
                needs_source_proof = true;
            }
        }
    }
    let written = store.stage_membership(
        txn,
        index,
        staging.field_of,
        owner,
        old.as_deref(),
        new.as_deref(),
    )? > 0;
    if index.unique && written {
        if let Some(values) = new {
            claims.push(UniqueClaim {
                index: index.clone(),
                values,
                node_id,
                needs_source_proof,
            });
        }
    }
    Ok(written)
}

/// The tuples of `values` whose unique entry in `index` names `holder`, as
/// `txn` sees them.
fn held_tuples(
    store: &LocalIndexStore<'_>,
    txn: &Transaction,
    index: &IndexDefinition,
    values: &[Value],
    holder: NodeId,
) -> Result<Vec<Vec<u8>>, StoreError> {
    use coordinode_core::index::derive::tuples;
    let mut out = Vec::new();
    for tuple in tuples(values) {
        if store.unique_holder(txn, index, &tuple)? == Some(Some(holder)) {
            out.push(tuple);
        }
    }
    Ok(out)
}

/// Whether the node `holder` of shard `shard_id` holds one of the entries
/// `values` take in `index`, as `txn` sees its record: the node's own
/// record, or any version of a temporal node, since a temporal node keeps
/// the values of its history. Checked against the index's own
/// interpretation (its label, sparse and partial rules, each list element),
/// not against any query.
///
/// # Errors
///
/// A storage failure or an undecodable record.
pub fn holds_values(
    txn: &Transaction,
    shard_id: u16,
    index: &IndexDefinition,
    field_of: FieldOf<'_>,
    holder: NodeId,
    values: &[Value],
) -> Result<bool, StoreError> {
    use coordinode_core::index::derive::tuples;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    let wanted = tuples(values);
    let interpretation = index.interpretation(field_of);
    let holds = |record: &coordinode_core::graph::node::NodeRecord| {
        record.primary_label() == index.label
            && interpretation
                .record_membership(record)
                .is_some_and(|held| tuples(&held).iter().any(|t| wanted.contains(t)))
    };
    if LocalNodeStore
        .get(txn, shard_id, holder)?
        .is_some_and(|record| holds(&record))
    {
        return Ok(true);
    }
    Ok(LocalNodeStore
        .versions(txn, shard_id, holder)?
        .iter()
        .any(|(_, record)| holds(record)))
}

impl Default for IndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
