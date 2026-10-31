//! Which interpretation of its label a vector index answers a reader with.
//!
//! While a change of what a type implies is certified, an index keeps the
//! members of both interpretations in one graph and records the ones only
//! one of them carries. A reader is answered with its own: the other's
//! members still lead the traversal on to its own, but are never results.
//! A reader whose interpretation the index does not hold is answered from
//! the store instead.

use std::sync::atomic::Ordering;

use coordinode_search::tantivy::Scope;
use coordinode_vector::hnsw::SearchResult;
use rustc_hash::FxHashMap;

use super::{VectorIndexKey, VectorIndexRegistry};
use crate::index::membership::LabelCarriers;
use crate::index::selection::{Answer, Held, answer};

/// The interpretations an index holds of its label and the members only one
/// of them carries.
#[derive(Debug, Default)]
pub(super) struct Selection {
    held: Held,
    /// Which change of the record this is (see
    /// [`VectorIndexRegistry::held_at`]).
    revision: u64,
    /// Members only the candidate carries ([`Scope::Gained`]) or only the
    /// base ([`Scope::Retired`]); every other member is carried by both.
    moved: FxHashMap<u64, Scope>,
}

impl Selection {
    /// Whether a search answering with `scope` left out returns `id`.
    fn admits(&self, without: Option<Scope>, id: u64) -> bool {
        without.is_none_or(|scope| self.moved.get(&id) != Some(&scope))
    }
}

/// Whose interpretation of the label a search answers.
#[derive(Debug, Clone, Copy)]
pub enum Reader<'a> {
    /// A reader for whom the label's carriers are these, as its view of the
    /// catalog has them; `None` for a label without an identity.
    Of(Option<&'a LabelCarriers>),
    /// A caller with no view of the catalog, answered with every member.
    Any,
}

impl VectorIndexRegistry {
    fn next_revision(&self) -> u64 {
        self.revisions.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// The changes of what the indexes hold, for a caller waiting for one
    /// to hold an interpretation.
    pub fn held_changes(&self) -> &crate::index::selection::HeldChanges {
        &self.held_changes
    }

    /// The vector indexes with a record not yet answering readers of both
    /// the carriers in force and those with every published candidate put
    /// in force, each as `(label, property)`: `carriers` gives both for a
    /// label, read from one catalog state. An index without a record is
    /// filled by its own build, which reads the catalog as it then stands.
    ///
    /// # Errors
    ///
    /// What `carriers` fails with.
    pub fn unprepared<E>(
        &self,
        carriers: &mut impl FnMut(&str) -> Result<crate::index::selection::BothCarriers, E>,
    ) -> Result<Vec<(String, String)>, E> {
        let mut unprepared = Vec::new();
        for def in self.all_definitions() {
            let property = def.property();
            let Some(held) = self.held(&def.label, property) else {
                continue;
            };
            let (in_force, candidate) = carriers(&def.label)?;
            if !crate::index::selection::prepared(&held, &in_force, &candidate) {
                unprepared.push((def.label.clone(), property.to_string()));
            }
        }
        Ok(unprepared)
    }

    /// The interpretations the index of `(label, property)` holds; `None`
    /// when nothing filled it under a recorded one.
    pub fn held(&self, label: &str, property: &str) -> Option<Held> {
        self.held_at(label, property).map(|(held, _)| held)
    }

    /// [`Self::held`] with the revision of the record: a later change of
    /// the record carries another.
    pub fn held_at(&self, label: &str, property: &str) -> Option<(Held, u64)> {
        self.selections
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(label.to_string(), property.to_string()))
            .map(|s| (s.held.clone(), s.revision))
    }

    /// Record that the index of `(label, property)` was filled with the
    /// members `carriers` carry, and with nothing else.
    pub fn record_built(&self, label: &str, property: &str, carriers: Option<LabelCarriers>) {
        let revision = self.next_revision();
        self.selections
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                (label.to_string(), property.to_string()),
                Selection {
                    held: Held::built(carriers),
                    revision,
                    moved: FxHashMap::default(),
                },
            );
        self.held_changes.changed();
    }

    /// Forget what the index of `(label, property)` was filled with: a new
    /// index under the key holds nothing recorded yet.
    pub(super) fn forget_selection(&self, key: &VectorIndexKey) {
        self.selections
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(key);
        self.held_changes.changed();
    }

    /// Begin keeping the members of `candidate` beside those of `base`, the
    /// index's own, in the index of `(label, property)`: only while its
    /// record is still at `revision`. The revision the record now carries,
    /// or `None` when it had moved on.
    pub fn begin_change(
        &self,
        (label, property): (&str, &str),
        revision: u64,
        base: Option<LabelCarriers>,
        candidate: Option<LabelCarriers>,
    ) -> Option<u64> {
        let next = self.next_revision();
        let mut selections = self.selections.write().unwrap_or_else(|e| e.into_inner());
        let selection = selections.get_mut(&(label.to_string(), property.to_string()))?;
        if selection.revision != revision || selection.held.base.as_ref() != Some(&base) {
            return None;
        }
        // The members an earlier candidate alone carried leave first: kept,
        // they would count as carried by both.
        for (&id, &scope) in &selection.moved {
            if scope == Scope::Gained {
                self.remove_member(label, property, id);
            }
        }
        selection.held = Held {
            base: Some(base),
            candidate: Some((candidate, false)),
        };
        selection.revision = next;
        selection.moved.clear();
        drop(selections);
        self.held_changes.changed();
        Some(next)
    }

    /// Record which interpretation alone carries member `id` of the index of
    /// `(label, property)`, `None` for both, while its record is at
    /// `revision`. Recorded before the member is inserted and cleared after
    /// it is removed, so a search never takes it for one of both.
    pub fn mark(
        &self,
        (label, property): (&str, &str),
        revision: u64,
        id: u64,
        scope: Option<Scope>,
    ) -> bool {
        let mut selections = self.selections.write().unwrap_or_else(|e| e.into_inner());
        let Some(selection) = selections.get_mut(&(label.to_string(), property.to_string())) else {
            return false;
        };
        if selection.revision != revision {
            return false;
        }
        match scope {
            Some(scope) => selection.moved.insert(id, scope),
            None => selection.moved.remove(&id),
        };
        true
    }

    /// Record the index of `(label, property)` holding every member of the
    /// candidate it keeps, while its record is at `revision`.
    pub fn finish_change(&self, (label, property): (&str, &str), revision: u64) -> bool {
        let mut selections = self.selections.write().unwrap_or_else(|e| e.into_inner());
        let Some(selection) = selections.get_mut(&(label.to_string(), property.to_string())) else {
            return false;
        };
        let finished = match &mut selection.held.candidate {
            Some((_, prepared)) if selection.revision == revision => {
                *prepared = true;
                true
            }
            _ => false,
        };
        drop(selections);
        if finished {
            self.held_changes.changed();
        }
        finished
    }

    /// Bring the index of `(label, property)` back to one interpretation,
    /// `in_force`, once the change it followed ended, while its record is at
    /// `revision`: the members of the `dropped` scope leave the graph and
    /// every other one counts as carried by it. One step under the record's
    /// lock, so no search sees the graph without them and the record still
    /// holding the interpretation they belong to.
    pub fn settle_change(
        &self,
        (label, property): (&str, &str),
        revision: u64,
        dropped: Scope,
        in_force: Option<LabelCarriers>,
    ) -> bool {
        let next = self.next_revision();
        let mut selections = self.selections.write().unwrap_or_else(|e| e.into_inner());
        let Some(selection) = selections.get_mut(&(label.to_string(), property.to_string())) else {
            return false;
        };
        if selection.revision != revision {
            return false;
        }
        for (&id, &scope) in &selection.moved {
            if scope == dropped {
                self.remove_member(label, property, id);
            }
        }
        *selection = Selection {
            held: Held::built(in_force),
            revision: next,
            moved: FxHashMap::default(),
        };
        drop(selections);
        self.held_changes.changed();
        true
    }

    /// The members only one interpretation carries in the index of
    /// `(label, property)`, with the scope of each.
    pub fn moved(&self, label: &str, property: &str) -> Vec<(u64, Scope)> {
        self.selections
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(label.to_string(), property.to_string()))
            .map(|s| s.moved.iter().map(|(&id, &scope)| (id, scope)).collect())
            .unwrap_or_default()
    }

    /// Whether the index of `(label, property)` is still being prepared for
    /// `reader`'s interpretation of the label: until it holds it whole, a
    /// read answers to the index's build policy, as during a build.
    pub fn preparing_for(&self, (label, property): (&str, &str), reader: Reader<'_>) -> bool {
        let Reader::Of(carriers) = reader else {
            return false;
        };
        self.selections
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(label.to_string(), property.to_string()))
            .is_some_and(|s| answer(carriers, &s.held) == Answer::Preparing)
    }

    /// Run `search` with the admission `reader` is answered with in the index
    /// of `(label, property)`, its record read-locked throughout, so the
    /// graph and the members it leaves out are of one state. `Ok(None)` when
    /// the index does not hold the reader's interpretation.
    pub(super) fn with_admission<E>(
        &self,
        (label, property): (&str, &str),
        reader: Reader<'_>,
        search: impl FnOnce(Option<&dyn Fn(u64) -> bool>) -> Result<Option<Vec<SearchResult>>, E>,
    ) -> Result<Option<Vec<SearchResult>>, E> {
        let selections = self.selections.read().unwrap_or_else(|e| e.into_inner());
        let Some(selection) = selections.get(&(label.to_string(), property.to_string())) else {
            return search(None);
        };
        let without = match reader {
            Reader::Any => {
                return search(None);
            }
            Reader::Of(carriers) => match answer(carriers, &selection.held) {
                Answer::Whole => return search(None),
                Answer::Without(scope) => scope,
                // Still being prepared for the reader's interpretation:
                // the members only the facts it left carry are kept out,
                // and the ones it gains not inserted yet are what a reader
                // admitted under the index's build policy misses, as during
                // a build.
                Answer::Preparing => Scope::Retired,
                Answer::Unable => return Ok(None),
            },
        };
        let admits = |id: u64| selection.admits(Some(without), id);
        search(Some(&admits))
    }
}
