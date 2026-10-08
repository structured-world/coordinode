//! The state of a temporal node's timeline at one valid-time instant.
//!
//! A temporal node keeps one stored version per `valid_from`. A read at
//! instant V selects the version whose half-open interval `[valid_from,
//! valid_to)` contains V, with an absent `valid_to` meaning no end. The
//! choice depends only on the intervals: a version written later, an
//! open-ended future version or an older backfill never stands in for the
//! state valid at V. A deletion is a tombstone version, so a deleted node has
//! a state at V, just not a positive one.

use std::borrow::Borrow;

use coordinode_core::graph::node::NodeRecord;
use coordinode_core::graph::types::Value;

/// What a node's timeline holds at one instant; `R` is the record, owned or
/// borrowed from the versions read.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StateAt<R = NodeRecord> {
    /// A live state, with the start of the version that holds it.
    Positive { valid_from: i64, record: R },
    /// The node was deleted at or before the instant.
    Deleted,
    /// No version covers the instant: before the first one, after the last
    /// one ended, or in a gap between two.
    Absent,
}

impl<R> StateAt<R> {
    /// The live state, if there is one.
    pub(crate) fn positive(self) -> Option<(i64, R)> {
        match self {
            Self::Positive { valid_from, record } => Some((valid_from, record)),
            Self::Deleted | Self::Absent => None,
        }
    }
}

/// The field ids a timeline projection reads, `None` for a name no record
/// carries yet (then no version can set it).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TimelineFields {
    pub valid_to: Option<u32>,
    pub deleted: Option<u32>,
}

/// The state at `at` of the timeline `versions`, given as `(valid_from,
/// record)` in ascending `valid_from` order (the order their keys sort in).
pub(crate) fn state_at<R: Borrow<NodeRecord>>(
    versions: impl IntoIterator<Item = (i64, R)>,
    at: i64,
    fields: TimelineFields,
) -> StateAt<R> {
    state_span(versions, at, fields).state
}

/// A timeline's state at one instant and the half-open interval around it
/// over which the timeline holds that same state.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StateSpan<R = NodeRecord> {
    /// The state at the instant.
    pub state: StateAt<R>,
    /// Where the state starts; `None` when it holds since forever.
    pub from: Option<i64>,
    /// Where it ends, exclusive; `None` when it holds for good.
    pub until: Option<i64>,
}

/// [`state_at`] together with the interval the state holds over: the
/// covering version's interval cut at the next version's start, or, while no
/// version covers the instant, the gap between the versions around it.
pub(crate) fn state_span<R: Borrow<NodeRecord>>(
    versions: impl IntoIterator<Item = (i64, R)>,
    at: i64,
    fields: TimelineFields,
) -> StateSpan<R> {
    // The version starting latest at or before `at`: a later start
    // supersedes an earlier one at every instant they share.
    let mut covering = None;
    let mut next = None;
    for (valid_from, record) in versions {
        if valid_from > at {
            next = Some(valid_from);
            break;
        }
        covering = Some((valid_from, record));
    }
    let Some((valid_from, record)) = covering else {
        return StateSpan {
            state: StateAt::Absent,
            from: None,
            until: next,
        };
    };
    let props = &record.borrow().props;
    let valid_to = fields
        .valid_to
        .and_then(|id| props.get(&id))
        .and_then(|value| match value {
            Value::Int(t) | Value::Timestamp(t) => Some(*t),
            _ => None,
        });
    if let Some(ended) = valid_to.filter(|valid_to| *valid_to <= at) {
        return StateSpan {
            state: StateAt::Absent,
            from: Some(ended.max(valid_from)),
            until: next,
        };
    }
    let until = match (valid_to, next) {
        (Some(end), Some(next)) => Some(end.min(next)),
        (end, next) => end.or(next),
    };
    let deleted = fields
        .deleted
        .and_then(|id| props.get(&id))
        .is_some_and(|value| matches!(value, Value::Bool(true)));
    let state = if deleted {
        StateAt::Deleted
    } else {
        StateAt::Positive { valid_from, record }
    };
    StateSpan {
        state,
        from: Some(valid_from),
        until,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
