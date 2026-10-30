//! A node's label membership as the index layer judges it: the labels its
//! record stores and the ones its type implies, compiled once per type from
//! the catalog the pass reads, so a pass over many nodes reads no catalog
//! and walks no implication per node.

use coordinode_core::graph::node::NodeRecord;
use coordinode_core::schema::label_id::LabelId;
use coordinode_core::schema::labels::{ImpliedCache, LabelFacts, Membership};
use coordinode_modality::{LocalSchemaStore, SchemaStore as _, StoreError};
use coordinode_storage::engine::StorageSnapshot;
use coordinode_storage::engine::core::StorageEngine;

/// The nodes that carry one label: those storing it, and those whose type is
/// one of the declared types implying it. Resolved once per index or pass,
/// never per node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelCarriers {
    /// The label's identity.
    pub label: LabelId,
    /// The declared types implying it.
    pub implying: Vec<LabelId>,
}

impl LabelCarriers {
    /// Carriers of `label` that no type implies.
    pub fn stored_only(label: LabelId) -> Self {
        Self {
            label,
            implying: Vec::new(),
        }
    }

    /// Whether a node storing `stored` carries the label.
    pub fn carried_by(&self, stored: &[LabelId]) -> bool {
        stored.contains(&self.label)
            || stored
                .first()
                .is_some_and(|primary| self.implying.contains(primary))
    }
}

/// The carriers of `label` in the latest catalog; `None` when the label has
/// no identity, and no node carries it.
///
/// # Errors
///
/// A catalog read failure, or an implication cycle the catalog holds.
pub fn label_carriers(
    engine: &StorageEngine,
    label: &str,
) -> Result<Option<LabelCarriers>, StoreError> {
    label_carriers_in(engine, label, &mut CatalogMembership::current(engine))
}

/// The carriers of `label` under the view of the catalog `cache` reads: a
/// snapshot's, or the one with every published candidate put in force.
///
/// # Errors
///
/// As [`label_carriers`].
pub fn label_carriers_in(
    engine: &StorageEngine,
    label: &str,
    cache: &mut CatalogMembership<'_>,
) -> Result<Option<LabelCarriers>, StoreError> {
    let store = LocalSchemaStore::new(engine);
    let Some(label) = cache.label_id(label)? else {
        return Ok(None);
    };
    let mut implying = Vec::new();
    for schema in store.list_labels()? {
        if schema.is_auxiliary() || schema.implies().is_empty() {
            continue;
        }
        let Some(id) = cache.label_id(&schema.name)? else {
            continue;
        };
        if cache.of_stored(&[id])?.contains(label) {
            implying.push(id);
        }
    }
    // In id order: two views holding the same interpretation compare equal.
    implying.sort_unstable();
    Ok(Some(LabelCarriers { label, implying }))
}

/// The membership of the records one pass reads, under one view of the
/// catalog: its latest state, or the state at a pinned snapshot.
pub struct CatalogMembership<'a> {
    engine: &'a StorageEngine,
    snapshot: Option<StorageSnapshot>,
    implied: ImpliedCache,
    ids: rustc_hash::FxHashMap<String, Option<LabelId>>,
    /// Whether a label's published candidate answers for it instead of the
    /// facts in force.
    candidates: bool,
}

impl<'a> CatalogMembership<'a> {
    /// Membership under the latest catalog.
    pub fn current(engine: &'a StorageEngine) -> Self {
        Self {
            engine,
            snapshot: None,
            implied: ImpliedCache::default(),
            ids: rustc_hash::FxHashMap::default(),
            candidates: false,
        }
    }

    /// Membership under the latest catalog with every published candidate
    /// put in force: what a node carries once the changes being proved are.
    pub fn candidates(engine: &'a StorageEngine) -> Self {
        Self {
            candidates: true,
            ..Self::current(engine)
        }
    }

    /// Membership under the catalog as `snapshot` reads it.
    pub fn at(engine: &'a StorageEngine, snapshot: StorageSnapshot) -> Self {
        Self {
            snapshot: Some(snapshot),
            ..Self::current(engine)
        }
    }

    /// The snapshot the catalog is read at: the pinned one, or the latest.
    fn view(&self) -> StorageSnapshot {
        self.snapshot.unwrap_or_else(|| self.engine.snapshot())
    }

    /// The identity label `name` is bound to under the view; `None` when it
    /// has none, which no node can carry.
    ///
    /// # Errors
    ///
    /// A catalog read failure.
    pub fn label_id(&mut self, name: &str) -> Result<Option<LabelId>, StoreError> {
        if let Some(id) = self.ids.get(name) {
            return Ok(*id);
        }
        let id = LocalSchemaStore::new(self.engine).label_id_at(self.view(), name)?;
        self.ids.insert(name.to_string(), id);
        Ok(id)
    }

    /// The name label `id` was allocated for.
    ///
    /// # Errors
    ///
    /// A catalog read failure, or an id without an identity record.
    pub fn name_of(&self, id: LabelId) -> Result<String, StoreError> {
        LocalSchemaStore::new(self.engine)
            .label_name(id)?
            .ok_or_else(|| StoreError::Decode {
                kind: "label identity",
                message: format!("label {id} has no identity record"),
            })
    }

    /// What the catalog says about label `id` under the view.
    fn facts(&mut self, id: LabelId) -> Result<LabelFacts, StoreError> {
        let (name, schema) = LocalSchemaStore::new(self.engine).identity_at(self.view(), id)?;
        let Some(mut schema) = schema else {
            return Ok(LabelFacts::default());
        };
        if self.candidates
            && let Some(candidate) = schema.candidate_schema()
        {
            schema = candidate;
        }
        let mut implies = Vec::with_capacity(schema.implies().len());
        for implied in schema.implies() {
            let Some(implied_id) = self.label_id(implied)? else {
                return Err(StoreError::Invariant(format!(
                    ":{name} implies :{implied}, which has no identity"
                )));
            };
            implies.push(implied_id);
        }
        Ok(LabelFacts {
            kind: schema.kind(),
            implies: implies.into(),
        })
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
    pub fn of_stored<'r>(
        &'r mut self,
        stored: &'r [LabelId],
    ) -> Result<Membership<'r>, StoreError> {
        if let Some(&primary) = stored.first() {
            let mut implied = core::mem::take(&mut self.implied);
            let compiled = implied.compile(primary, |id| self.facts(id));
            self.implied = implied;
            compiled?.map_err(|e| {
                StoreError::Invariant(format!("labels imply one another in a cycle: {e:?}"))
            })?;
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
    pub fn carries(&mut self, record: &NodeRecord, label: LabelId) -> Result<bool, StoreError> {
        if record.stores_label(label) {
            return Ok(true);
        }
        Ok(self.of(record)?.contains(label))
    }

    /// [`Self::carries`] for the label named `label`; a label without an
    /// identity is carried by no node.
    ///
    /// # Errors
    ///
    /// As [`Self::of`].
    pub fn carries_named(&mut self, record: &NodeRecord, label: &str) -> Result<bool, StoreError> {
        match self.label_id(label)? {
            Some(id) => self.carries(record, id),
            None => Ok(false),
        }
    }
}
