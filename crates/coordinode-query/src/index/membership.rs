//! A node's label membership as the index layer judges it: the labels its
//! record stores and the ones its type implies, compiled once per type from
//! the catalog the pass reads, so a pass over many nodes reads no catalog
//! and walks no implication per node.

use coordinode_core::graph::node::NodeRecord;
use coordinode_core::schema::labels::{ImpliedCache, LabelFacts, Membership};
use coordinode_modality::{LocalSchemaStore, SchemaStore as _, StoreError};
use coordinode_storage::engine::StorageSnapshot;
use coordinode_storage::engine::core::StorageEngine;

/// The declared types that imply `label` in the latest catalog: a node of
/// one of them carries `label` without storing it. Read once per index or
/// pass, never per node.
///
/// # Errors
///
/// A catalog read failure, or an implication cycle the catalog holds.
pub fn types_implying(engine: &StorageEngine, label: &str) -> Result<Vec<String>, StoreError> {
    let store = LocalSchemaStore::new(engine);
    let mut cache = CatalogMembership::current(engine);
    let mut implying = Vec::new();
    for schema in store.list_labels()? {
        if schema.is_auxiliary() || schema.implies().is_empty() {
            continue;
        }
        let stored = [schema.name];
        if cache.of_stored(&stored)?.contains(label) {
            let [name] = stored;
            implying.push(name);
        }
    }
    Ok(implying)
}

/// Whether a node storing `stored` carries `label`, `implying` being the
/// types that imply it ([`types_implying`]).
pub fn carries_label(stored: &[String], label: &str, implying: &[String]) -> bool {
    stored.iter().any(|l| l == label)
        || stored
            .first()
            .is_some_and(|primary| implying.iter().any(|t| t == primary))
}

/// The membership of the records one pass reads, under one view of the
/// catalog: its latest state, or the state at a pinned snapshot.
pub struct CatalogMembership<'a> {
    store: LocalSchemaStore<'a>,
    snapshot: Option<StorageSnapshot>,
    implied: ImpliedCache,
}

impl<'a> CatalogMembership<'a> {
    /// Membership under the latest catalog.
    pub fn current(engine: &'a StorageEngine) -> Self {
        Self {
            store: LocalSchemaStore::new(engine),
            snapshot: None,
            implied: ImpliedCache::default(),
        }
    }

    /// Membership under the catalog as `snapshot` reads it.
    pub fn at(engine: &'a StorageEngine, snapshot: StorageSnapshot) -> Self {
        Self {
            store: LocalSchemaStore::new(engine),
            snapshot: Some(snapshot),
            implied: ImpliedCache::default(),
        }
    }

    /// Every label `record` carries: its stored ones and the ones its type
    /// implies.
    ///
    /// # Errors
    ///
    /// A catalog read failure, or an implication cycle the catalog holds.
    pub fn of<'r>(&'r mut self, record: &'r NodeRecord) -> Result<Membership<'r>, StoreError> {
        self.of_stored(&record.labels)
    }

    /// [`Self::of`] for a node whose stored labels are `stored`.
    ///
    /// # Errors
    ///
    /// As [`Self::of`].
    pub fn of_stored<'r>(&'r mut self, stored: &'r [String]) -> Result<Membership<'r>, StoreError> {
        if let Some(primary) = stored.first() {
            let (store, snapshot) = (&self.store, self.snapshot);
            self.implied
                .compile(primary, |label| {
                    let schema = match snapshot {
                        Some(snapshot) => store.load_label_at(snapshot, label)?,
                        None => store.load_label(label)?,
                    };
                    Ok::<_, StoreError>(schema.as_ref().map(LabelFacts::of).unwrap_or_default())
                })?
                .map_err(|e| StoreError::Invariant(e.to_string()))?;
        }
        Ok(self
            .implied
            .membership(stored)
            .unwrap_or_else(|| Membership::stored_only(stored)))
    }

    /// Whether `record` carries `label`.
    ///
    /// # Errors
    ///
    /// As [`Self::of`].
    pub fn carries(&mut self, record: &NodeRecord, label: &str) -> Result<bool, StoreError> {
        if record.labels.iter().any(|l| l == label) {
            return Ok(true);
        }
        Ok(self.of(record)?.contains(label))
    }
}
