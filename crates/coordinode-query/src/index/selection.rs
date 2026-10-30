//! Which interpretation of a label an index maintained outside the store
//! (full-text, vector) answers with while a change of what a type implies
//! keeps the label's carriers apart.
//!
//! The label lookup's definition says, at any view, whether a change of the
//! label is being certified, was put in force or was withdrawn, until its
//! parts are folded; the change's certification record keeps the outcome
//! after that. Such an index keeps the documents of both interpretations
//! while it follows a change, each in its scope, and records the last change
//! it brought back to one interpretation. From the two, a reader learns
//! which scope to leave out, or that the index cannot answer for its view.

use coordinode_core::index::identity::GenerationId;
use coordinode_core::schema::label_id::LabelId;
use coordinode_modality::{BuildState, IndexStore as _, LocalIndexStore, PartState, StoreError};
use coordinode_search::tantivy::Scope;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;

/// Where the change of a label stands at one view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    /// No change of the label is kept apart.
    None,
    /// Change `operation` is being certified: the facts in force still hold.
    Proposed(GenerationId),
    /// Change `operation` was put in force.
    InForce(GenerationId),
    /// Change `operation` was withdrawn: the facts in force stay.
    Withdrawn(GenerationId),
}

/// How an index answers a reader of one view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    /// Every document it holds belongs to the reader's interpretation.
    Whole,
    /// Its documents of this scope do not belong to the reader's
    /// interpretation and are left out.
    Without(Scope),
    /// It does not hold the reader's interpretation yet: every node is read
    /// from the store instead.
    Unable,
}

/// What an index maintained outside the store keeps of the changes of its
/// label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Followed {
    /// The change whose documents it keeps in scopes, and whether every
    /// stored node has been written in its scope since.
    pub current: Option<(GenerationId, bool)>,
    /// The last change it brought back to one interpretation.
    pub settled: Option<GenerationId>,
}

/// Where the change of `label` stands as `read` sees the label lookup's
/// definition: at its snapshot, or the latest for a read without one.
///
/// # Errors
///
/// A catalog read failure.
pub fn change_in(
    engine: &StorageEngine,
    read: &Transaction<'_>,
    label: LabelId,
) -> Result<Change, StoreError> {
    let Some(snapshot) = read.snapshot() else {
        return change_now(engine, label);
    };
    let store = LocalIndexStore::new(engine);
    let Some(lookup) = store.label_lookup()? else {
        return Ok(Change::None);
    };
    let at_view = store.definition_at(snapshot, lookup.id)?;
    Ok(at_view.map_or(Change::None, |def| change_of(&def, label)))
}

/// Where the change of `label` stands in the latest catalog.
///
/// # Errors
///
/// A catalog read failure.
pub fn change_now(engine: &StorageEngine, label: LabelId) -> Result<Change, StoreError> {
    let store = LocalIndexStore::new(engine);
    Ok(store
        .label_lookup()?
        .map_or(Change::None, |def| change_of(&def, label)))
}

/// Where the change of `label` stands as the label lookup `lookup` keeps it:
/// one being certified first, as it is the one in progress.
fn change_of(lookup: &coordinode_modality::IndexDefinition, label: LabelId) -> Change {
    let mut parts = lookup.selection.iter().filter(|p| p.label == label);
    // Both roles of one change carry the same state.
    if let Some(proposed) = parts.clone().find(|p| p.state == PartState::Proposed) {
        return Change::Proposed(proposed.operation);
    }
    match parts.next() {
        None => Change::None,
        Some(p) if p.state == PartState::InForce => Change::InForce(p.operation),
        Some(p) => Change::Withdrawn(p.operation),
    }
}

/// How the certification `operation` ended for `label`, from its record:
/// put in force unless the label's candidate was refused; `None` while it
/// has not ended, or when no record of it is kept.
///
/// # Errors
///
/// A catalog read failure.
pub fn outcome(
    engine: &StorageEngine,
    operation: GenerationId,
    label: &str,
) -> Result<Option<Change>, StoreError> {
    let Some((record, _)) = LocalIndexStore::new(engine).load_certification(operation)? else {
        return Ok(None);
    };
    Ok(match record.state {
        BuildState::Published if record.refusal_of(label).is_none() => {
            Some(Change::InForce(operation))
        }
        BuildState::Published | BuildState::Failed { .. } | BuildState::Cancelled => {
            Some(Change::Withdrawn(operation))
        }
        BuildState::Accepted | BuildState::Running { .. } => None,
    })
}

/// How an index that keeps `followed` of `label`'s changes answers a reader
/// whose view sees `change`. A view whose parts are folded already asks the
/// change the index keeps of `concluded`, the outcome of a change by its
/// identity.
///
/// # Errors
///
/// What `concluded` fails with.
pub fn answer(
    change: Change,
    followed: Followed,
    concluded: impl FnOnce(GenerationId) -> Result<Option<Change>, StoreError>,
) -> Result<Answer, StoreError> {
    let change = match (change, followed.current) {
        (Change::None, None) => return Ok(Answer::Whole),
        // The parts are folded: the change the index keeps has ended, and
        // its record says how.
        (Change::None, Some((operation, _))) => match concluded(operation)? {
            Some(ended) => ended,
            None => return Ok(Answer::Unable),
        },
        (change, _) => change,
    };
    let operation = match change {
        Change::None => return Ok(Answer::Whole),
        Change::Proposed(op) | Change::InForce(op) | Change::Withdrawn(op) => op,
    };
    // Generations only grow: an index that settled this change or a later
    // one holds its outcome alone.
    if followed
        .settled
        .is_some_and(|settled| settled.as_raw() >= operation.as_raw())
    {
        return Ok(Answer::Whole);
    }
    Ok(match (followed.current, change) {
        (Some((kept, _)), Change::Proposed(_) | Change::Withdrawn(_)) if kept == operation => {
            Answer::Without(Scope::Gained)
        }
        (Some((kept, true)), Change::InForce(_)) if kept == operation => {
            Answer::Without(Scope::Retired)
        }
        // Keeping another change, or this one not yet prepared: the index
        // does not hold the reader's interpretation.
        (Some(_), _) => Answer::Unable,
        // Not following the change yet, the index holds the facts in force,
        // which hold until the change is put in force.
        (None, Change::Proposed(_) | Change::Withdrawn(_)) => Answer::Whole,
        (None, _) => Answer::Unable,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
