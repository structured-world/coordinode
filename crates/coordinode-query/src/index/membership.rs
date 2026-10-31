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
    // Every identity ever allocated, each judged by the view's own facts:
    // the latest catalog's types and implications are not the view's, which
    // may hold a type that implies nothing now, or a candidate's implication
    // the types in force lack. Identity records are never removed, so none
    // the view knows is missed; one allocated after it has no facts there.
    let mut implying = Vec::new();
    for (id, _) in store.label_names()? {
        if id != label && cache.of_stored(&[id])?.contains(label) {
            implying.push(id);
        }
    }
    // In id order: two views holding the same interpretation compare equal.
    implying.sort_unstable();
    Ok(Some(LabelCarriers { label, implying }))
}

/// The membership of the records one pass reads, under one view of the
/// catalog, pinned for the whole pass: every fact it judges by is of one
/// committed state, never a mix of several.
pub struct CatalogMembership<'a> {
    engine: &'a StorageEngine,
    snapshot: StorageSnapshot,
    /// Keeps the catalog of `snapshot` readable for the whole pass: no
    /// compaction collects a version it still judges by. `None` when the
    /// snapshot was already below the collection watermark, which the reads
    /// then report.
    _pin: Option<coordinode_storage::engine::coordinator::SnapshotPin>,
    implied: ImpliedCache,
    ids: rustc_hash::FxHashMap<String, Option<LabelId>>,
    /// Whether a label's published candidate answers for it instead of the
    /// facts in force.
    candidates: bool,
    /// Where what the pass learns of the catalog is charged before it is
    /// kept, when it holds to a memory budget.
    budget: Option<&'a coordinode_core::budget::QueryBudget>,
}

impl<'a> CatalogMembership<'a> {
    /// Membership under the catalog as it stands now.
    pub fn current(engine: &'a StorageEngine) -> Self {
        let (snapshot, pin) = engine.pin_latest_snapshot();
        Self::pinned(engine, snapshot, pin)
    }

    /// Membership under the catalog as it stands now with every published
    /// candidate put in force: what a node carries once the changes being
    /// proved are.
    pub fn candidates(engine: &'a StorageEngine) -> Self {
        let (snapshot, pin) = engine.pin_latest_snapshot();
        Self {
            candidates: true,
            ..Self::pinned(engine, snapshot, pin)
        }
    }

    /// Membership under the catalog as `snapshot` reads it.
    pub fn at(engine: &'a StorageEngine, snapshot: StorageSnapshot) -> Self {
        Self::pinned(engine, snapshot, engine.pin_snapshot_at(snapshot))
    }

    fn pinned(
        engine: &'a StorageEngine,
        snapshot: StorageSnapshot,
        pin: Option<coordinode_storage::engine::coordinator::SnapshotPin>,
    ) -> Self {
        Self {
            engine,
            snapshot,
            _pin: pin,
            implied: ImpliedCache::default(),
            ids: rustc_hash::FxHashMap::default(),
            candidates: false,
            budget: None,
        }
    }

    /// The same membership, charging `budget` for every catalog fact it
    /// keeps before keeping it, for as long as the budget lives: the names
    /// it binds, the implications it reads and the closures it compiles.
    /// A fact that does not fit fails the read with the budget's refusal.
    pub fn charged_to(self, budget: &'a coordinode_core::budget::QueryBudget) -> Self {
        Self {
            budget: Some(budget),
            ..self
        }
    }

    /// Charge `bytes` about to be kept to the budget, when there is one.
    fn charge(&self, bytes: usize) -> Result<(), StoreError> {
        if let Some(budget) = self.budget {
            // Bytes of memory about to exist: fits in u64.
            budget.reserve(bytes as u64)?.keep_until_query_ends();
        }
        Ok(())
    }

    /// [`Self::candidates`] as `snapshot` reads the catalog: with
    /// [`Self::at`] on the same snapshot, the facts in force and the
    /// candidates of one committed state.
    pub fn candidates_at(engine: &'a StorageEngine, snapshot: StorageSnapshot) -> Self {
        Self {
            candidates: true,
            ..Self::at(engine, snapshot)
        }
    }

    /// The snapshot the catalog is read at.
    fn view(&self) -> StorageSnapshot {
        self.snapshot
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
        // The name, and its entry three times over with its control byte:
        // a hash map holds a little over twice the room its entries take.
        self.charge(name.len() + 3 * (core::mem::size_of::<(String, Option<LabelId>)>() + 1))?;
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
        // The name and the implied names live while the facts are read:
        // charged before each is decoded and released after.
        let mut read = self.budget.map(|b| b.empty_charge());
        let (name, facts) = LocalSchemaStore::new(self.engine).implication_facts_at(
            self.view(),
            id,
            self.candidates,
            &mut |bytes| {
                if let Some(read) = read.as_mut() {
                    // Bytes of memory about to exist: fits in u64.
                    read.grow(bytes as u64)?;
                }
                Ok(())
            },
        )?;
        let Some(facts) = facts else {
            return Ok(LabelFacts::default());
        };
        // Kept for the rest of the pass, each `n` implications: the list
        // built here and its shared copy (2n); the walk's pending stack,
        // which takes each of them once and grows to at most twice what it
        // holds (2n); and the label's own slot in the closure, which also
        // grows to at most twice what it holds and is copied once into the
        // compiled slice (3). Every label the walk visits is read here once,
        // so this covers the whole walk.
        let n = facts.implies.len();
        self.charge(
            (4 * n + 3) * core::mem::size_of::<LabelId>() + 2 * core::mem::size_of::<usize>(),
        )?;
        let mut implies = Vec::with_capacity(n);
        for implied in &facts.implies {
            let Some(implied_id) = self.label_id(implied)? else {
                return Err(StoreError::Invariant(format!(
                    ":{name} implies :{implied}, which has no identity"
                )));
            };
            implies.push(implied_id);
        }
        drop(read);
        Ok(LabelFacts {
            kind: facts.kind,
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
            if self.implied.implied(primary).is_none() {
                // The compiled type's entry three times over with its control
                // byte, as a hash map holds it; its labels are charged as the
                // walk reads them.
                self.charge(
                    3 * (core::mem::size_of::<(
                        LabelId,
                        coordinode_core::schema::labels::ImpliedLabels,
                    )>() + 1),
                )?;
            }
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
