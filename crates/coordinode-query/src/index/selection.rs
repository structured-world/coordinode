//! Which interpretation of a label an index maintained outside the store
//! (full-text, vector) answers with while a change of what a type implies
//! keeps the label's carriers apart.
//!
//! Such an index records the interpretations its documents were written
//! under: the base its shared and retired documents hold, and while it
//! follows a change, the candidate its shared and gained documents hold,
//! once every stored node was written in its scope. A reader brings the
//! label's carriers as its own view of the catalog has them. The index
//! answers a reader whose carriers are one of those, leaving the other's
//! own documents out; any other reader, however its view relates in time,
//! is read from the store instead.

use coordinode_search::tantivy::Scope;

use super::membership::LabelCarriers;

/// How an index answers a reader of one view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    /// Every document it holds belongs to the reader's interpretation.
    Whole,
    /// Its documents of this scope do not belong to the reader's
    /// interpretation and are left out.
    Without(Scope),
    /// It does not hold the reader's interpretation: every node is read
    /// from the store instead.
    Unable,
}

/// The interpretations an index's documents were written under. `None`
/// inside for a label without an identity, which no node carries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Held {
    /// What the shared and retired documents hold; `None` until the index
    /// is first built, and while its retired documents are being removed.
    pub base: Option<Option<LabelCarriers>>,
    /// While the index follows a change: what the shared and gained
    /// documents hold, and whether every stored node has been written in
    /// its scope since, without which the gained documents are partial.
    pub candidate: Option<(Option<LabelCarriers>, bool)>,
}

impl Held {
    /// An index built under `base` alone.
    pub fn built(base: Option<LabelCarriers>) -> Self {
        Self {
            base: Some(base),
            candidate: None,
        }
    }
}

/// How an index holding `held` answers a reader for whom the label's
/// carriers are `reader`.
pub fn answer(reader: Option<&LabelCarriers>, held: &Held) -> Answer {
    let is = |carriers: &Option<LabelCarriers>| carriers.as_ref() == reader;
    match (&held.base, &held.candidate) {
        (Some(base), None) if is(base) => Answer::Whole,
        (Some(base), Some(_)) if is(base) => Answer::Without(Scope::Gained),
        (_, Some((candidate, true))) if is(candidate) => Answer::Without(Scope::Retired),
        _ => Answer::Unable,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
