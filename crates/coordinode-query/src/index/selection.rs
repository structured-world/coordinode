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
    /// It is being prepared for the reader's interpretation and does not
    /// hold it whole yet: the reader is refused as not ready, never
    /// answered by reading every node instead.
    Preparing,
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

/// A label's carriers in force and with every published candidate put in
/// force, of one catalog state.
pub type BothCarriers = (Option<LabelCarriers>, Option<LabelCarriers>);

/// Whether an index holding `held` answers readers of both `in_force` and
/// `candidate`, the label's carriers before and after a change of what a
/// type implies: always when the change leaves them alone, otherwise once
/// it holds the facts in force and every stored node was written in the
/// candidate's scope.
pub fn prepared(
    held: &Held,
    in_force: &Option<LabelCarriers>,
    candidate: &Option<LabelCarriers>,
) -> bool {
    in_force == candidate
        || (held.base.as_ref() == Some(in_force)
            && held
                .candidate
                .as_ref()
                .is_some_and(|(c, prepared)| *prepared && c == candidate))
}

/// Counts the changes of what a registry's indexes hold, so a caller
/// waiting for one to hold an interpretation sleeps until something
/// changed rather than asking again on a timer.
#[derive(Debug, Default)]
pub struct HeldChanges {
    count: parking_lot::Mutex<u64>,
    changed: parking_lot::Condvar,
}

impl HeldChanges {
    /// Record a change and wake every waiter.
    pub fn changed(&self) {
        let mut count = self.count.lock();
        // One per change of an index, far below u64::MAX.
        *count += 1;
        self.changed.notify_all();
    }

    /// How many changes were recorded: the mark a wait starts from.
    pub fn seen(&self) -> u64 {
        *self.count.lock()
    }

    /// Wait until a change after `seen` is recorded, or `limit` passed;
    /// returns the count then.
    pub fn wait_past(&self, seen: u64, limit: core::time::Duration) -> u64 {
        let mut count = self.count.lock();
        if *count == seen {
            self.changed.wait_for(&mut count, limit);
        }
        *count
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
        (_, Some((candidate, false))) if is(candidate) => Answer::Preparing,
        _ => Answer::Unable,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
