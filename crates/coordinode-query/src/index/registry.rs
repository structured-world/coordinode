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
    /// Orders the stored marks of this member's unfit copies: a mark written
    /// for a new report never lands before the removal of an older one.
    marks: parking_lot::Mutex<()>,
    /// Runs inside [`Self::verified_here`] right after the stored mark is
    /// removed, so a test can make a finding at that point.
    #[cfg(test)]
    after_mark_removed: parking_lot::Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}

/// The suspect generations as this process knows them.
#[derive(Default)]
struct IntegrityView {
    /// Suspect per the catalog's integrity records.
    recorded: rustc_hash::FxHashSet<GenerationId>,
    /// Generations whose copy on this member was found disagreeing with the
    /// records: they answer from the records here until a check verifies
    /// this copy, whatever the catalog says about another member's.
    local: FxHashMap<GenerationId, LocalMark>,
    /// Source of [`LocalMark::revision`].
    last_revision: u64,
    /// Verified generations that were once damaged, with the earliest
    /// snapshot they answer for.
    trusted_from: FxHashMap<GenerationId, u64>,
    /// Disagreements found here, waiting to be recorded.
    reports: Vec<IntegrityReport>,
    /// A report was dropped at the bound: the maintenance finds the
    /// generations it would have named from the marks instead.
    overflowed: bool,
}

/// This member's own finding against one generation's copy.
#[derive(Debug, Clone, Copy)]
struct LocalMark {
    /// Moves with every disagreement found here, from one counter for all
    /// generations: a check proves the copy only if none arrived since the
    /// revision its pass started at.
    revision: u64,
    /// Where its stored copy stands.
    state: MarkState,
}

/// Whether a mark survives a restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkState {
    /// Not written: the disk was below its free-space reserve, so nothing
    /// reached it, and the write is safe to make once there is room.
    Pending,
    /// Written and flushed.
    Stored,
    /// A write or flush of it failed: whether anything reached the disk is
    /// unknown, and a retry proves nothing (a failed fsync may have dropped
    /// the pages it reports clean afterwards). Never retried; the generation
    /// stays unfit in this process.
    Failed,
}

/// A mark could not be made durable: the statement that found the
/// disagreement fails with it rather than report it kept.
#[derive(Debug, thiserror::Error)]
#[error(
    "the finding that index generation {generation} disagrees with its records could not be \
     stored durably on this member: {source}"
)]
pub struct MarkNotDurable {
    /// The generation found disagreeing.
    pub generation: u64,
    /// The write or flush that failed.
    #[source]
    pub source: StoreError,
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
/// the generation stays suspect here, and the maintenance admits the checks
/// they would have started from this member's marks.
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
    /// A disagreement found during the write could not be stored durably.
    #[error(transparent)]
    MarkNotDurable(#[from] MarkNotDurable),
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
            marks: parking_lot::Mutex::new(()),
            #[cfg(test)]
            after_mark_removed: parking_lot::Mutex::new(None),
        }
    }

    /// Record that an entry of `index` was found disagreeing with the record
    /// it names, as `found` says: from now on its generation proves no
    /// lookup complete and no unique value free on this member, and the
    /// report waits for the maintenance that records it in the catalog and
    /// checks the generation. A read or a write that finds one reports it;
    /// neither changes an entry or a record for it.
    ///
    /// When this returns `Ok`, the finding is stored in `engine`, so a
    /// restart keeps answering from the records; or the disk is below its
    /// free-space reserve, nothing was written, and the maintenance stores it
    /// once there is room ([`Self::store_pending_marks`]). The decision that
    /// a mark is stored is taken in the same order as a check lifting it, so
    /// a finding made while a check removes the mark returns only after the
    /// mark is stored again.
    ///
    /// # Errors
    ///
    /// [`MarkNotDurable`] when the write or flush of the mark failed: its
    /// durability is unknown and it is not retried. The generation stays
    /// unfit in this process; the statement that found it fails.
    pub fn report_mismatch(
        &self,
        engine: &StorageEngine,
        index: &IndexDefinition,
        found: Mismatch,
    ) -> Result<(), MarkNotDurable> {
        {
            let mut view = self.integrity.lock();
            view.last_revision += 1;
            let revision = view.last_revision;
            let mark = view.local.entry(index.generation).or_insert(LocalMark {
                revision,
                state: MarkState::Pending,
            });
            mark.revision = revision;
            if mark.state == MarkState::Pending {
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
            if view.reports.contains(&report) {
                // Already queued.
            } else if view.reports.len() < MAX_PENDING_REPORTS {
                view.reports.push(report);
            } else {
                view.overflowed = true;
            }
            self.any_suspect
                .store(true, core::sync::atomic::Ordering::Release);
            self.reported.notify_all();
        }
        self.store_mark(engine, index.generation)
    }

    /// Store the mark of `generation` if it waits to be. Taken in the order
    /// of [`Self::verified_here`]: what it finds is what a lift left.
    fn store_mark(
        &self,
        engine: &StorageEngine,
        generation: GenerationId,
    ) -> Result<(), MarkNotDurable> {
        let _order = self.marks.lock();
        let state = self
            .integrity
            .lock()
            .local
            .get(&generation)
            .map(|m| m.state);
        match state {
            Some(MarkState::Pending) => self.write_mark(engine, generation),
            Some(MarkState::Failed) => Err(MarkNotDurable {
                generation: generation.as_raw(),
                source: StoreError::Storage(StorageError::Io(
                    "an earlier write of this mark failed; its durability is unknown".into(),
                )),
            }),
            Some(MarkState::Stored) | None => Ok(()),
        }
    }

    /// Write and flush the mark of `generation`, under the order lock. A disk
    /// below its free-space reserve is not written to at all and the mark
    /// stays pending; any failure after that leaves it failed.
    fn write_mark(
        &self,
        engine: &StorageEngine,
        generation: GenerationId,
    ) -> Result<(), MarkNotDurable> {
        if engine.space().is_paused() {
            return Ok(());
        }
        let result = LocalIndexStore::new(engine).mark_unfit_here(generation);
        let state = if result.is_ok() {
            MarkState::Stored
        } else {
            MarkState::Failed
        };
        if let Some(mark) = self.integrity.lock().local.get_mut(&generation) {
            mark.state = state;
        }
        result.map_err(|source| {
            tracing::error!(
                generation = generation.as_raw(),
                error = %source,
                "could not store that this member's index copy disagrees with the records; \
                 its durability is unknown and it is not retried"
            );
            MarkNotDurable {
                generation: generation.as_raw(),
                source,
            }
        })
    }

    /// Store the marks the free-space reserve kept from being written, now
    /// that there may be room. A mark whose write failed is not retried.
    ///
    /// # Errors
    ///
    /// The first mark whose write or flush failed.
    pub fn store_pending_marks(&self, engine: &StorageEngine) -> Result<(), MarkNotDurable> {
        let pending: Vec<GenerationId> = self
            .integrity
            .lock()
            .local
            .iter()
            .filter(|(_, m)| m.state == MarkState::Pending)
            .map(|(g, _)| *g)
            .collect();
        for generation in pending {
            self.store_mark(engine, generation)?;
        }
        Ok(())
    }

    /// The generations this member found its copy of unfit, for the
    /// maintenance to admit their checks from when reports were dropped or
    /// the marks were read back after a restart.
    pub fn unfit_here(&self) -> Vec<GenerationId> {
        self.integrity.lock().local.keys().copied().collect()
    }

    /// The index that serves from `generation` now, if one does.
    pub fn index_of_generation(&self, generation: GenerationId) -> Option<IndexId> {
        self.indexes
            .read()
            .by_id
            .values()
            .find(|r| r.def.generation == generation)
            .map(|r| r.def.id)
    }

    /// Whether a report was dropped at the bound since the last call.
    pub fn take_overflow(&self) -> bool {
        core::mem::take(&mut self.integrity.lock().overflowed)
    }

    /// Whether an entry of `generation` is known to disagree with its record,
    /// here or per the catalog.
    pub fn is_suspect(&self, generation: GenerationId) -> bool {
        if !self.any_suspect.load(core::sync::atomic::Ordering::Acquire) {
            return false;
        }
        let view = self.integrity.lock();
        view.recorded.contains(&generation) || view.local.contains_key(&generation)
    }

    /// The revision of this member's latest finding against `generation`, or
    /// `0` when it has none: what a check's pass takes before it reads.
    pub fn local_revision(&self, generation: GenerationId) -> u64 {
        self.integrity
            .lock()
            .local
            .get(&generation)
            .map_or(0, |m| m.revision)
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
        for generation in generations {
            view.last_revision += 1;
            let revision = view.last_revision;
            view.local.entry(*generation).or_insert(LocalMark {
                revision,
                state: MarkState::Stored,
            });
        }
        self.refresh_any_suspect(&view);
    }

    /// A check's pass read this member's copy of `generation` from its start
    /// and found it agreeing with the records, after taking `revision`: the
    /// mark is lifted, in memory and in `engine`, unless a disagreement was
    /// found here since. Returns whether it was lifted.
    ///
    /// A finding made while the mark is being removed waits for this to
    /// finish (the same order lock decides whether a mark is stored), and
    /// this stores its mark again before letting it go on.
    ///
    /// # Errors
    ///
    /// The stored mark could not be removed (the copy stays unfit here), or a
    /// finding made during the removal could not be stored again.
    pub fn verified_here(
        &self,
        engine: &StorageEngine,
        generation: GenerationId,
        revision: u64,
    ) -> Result<bool, MarkNotDurable> {
        let _order = self.marks.lock();
        {
            let view = self.integrity.lock();
            // A failed mark's stored state is unknown: nothing about this copy
            // is proved by touching the disk again, so it stays unfit here.
            if view
                .local
                .get(&generation)
                .is_some_and(|m| m.revision != revision || m.state == MarkState::Failed)
            {
                return Ok(false);
            }
        }
        // Removed from storage first. A failure leaves the copy unfit, and
        // the stored mark of unknown state: the removal may have reached the
        // disk, so the mark no longer counts as stored and is never written
        // again.
        if let Err(source) = LocalIndexStore::new(engine).clear_unfit_here(&[generation]) {
            if let Some(mark) = self.integrity.lock().local.get_mut(&generation) {
                mark.state = MarkState::Failed;
            }
            tracing::error!(
                generation = generation.as_raw(),
                error = %source,
                "could not remove the mark of a verified index copy; its stored state is \
                 unknown and the copy stays unfit here"
            );
            return Err(MarkNotDurable {
                generation: generation.as_raw(),
                source,
            });
        }
        #[cfg(test)]
        if let Some(hook) = self.after_mark_removed.lock().as_ref() {
            hook();
        }
        let mut view = self.integrity.lock();
        match view.local.get_mut(&generation) {
            Some(mark) if mark.revision != revision => {
                // A finding arrived during the removal: the stored copy it
                // relied on is gone, so it is written again while the finding
                // still waits on the order lock.
                mark.state = MarkState::Pending;
                drop(view);
                self.write_mark(engine, generation)?;
                Ok(false)
            }
            _ => {
                view.local.remove(&generation);
                self.refresh_any_suspect(&view);
                Ok(true)
            }
        }
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
        if view.recorded.contains(&generation) || view.local.contains_key(&generation) {
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
        // This member's own marks survive a restart. A generation that left
        // the catalog keeps its mark: a reader pinned to an older snapshot,
        // plan or cursor may still reach its entries, and nothing records
        // when the last such reader is gone.
        self.found_unfit_here(&store.list_unfit_here().map_err(storage)?);
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
                            staging.engine,
                            index,
                            Mismatch::Extra {
                                node: holder.as_raw(),
                                valid_from: None,
                                tuple,
                            },
                        )?;
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
