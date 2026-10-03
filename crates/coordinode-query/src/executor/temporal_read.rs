//! The state of a temporal node's timeline at one valid-time instant.
//!
//! A temporal node keeps one stored version per `valid_from`. A read at
//! instant V selects the version whose half-open interval `[valid_from,
//! valid_to)` contains V, with an absent `valid_to` meaning no end. The
//! choice depends only on the intervals: a version written later, an
//! open-ended future version or an older backfill never stands in for the
//! state valid at V. A deletion is a tombstone version, so a deleted node has
//! a state at V, just not a positive one.

use coordinode_core::graph::node::NodeRecord;
use coordinode_core::graph::types::Value;

/// What a node's timeline holds at one instant.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StateAt {
    /// A live state, with the start of the version that holds it.
    Positive { valid_from: i64, record: NodeRecord },
    /// The node was deleted at or before the instant.
    Deleted,
    /// No version covers the instant: before the first one, after the last
    /// one ended, or in a gap between two.
    Absent,
}

impl StateAt {
    /// The live state, if there is one.
    pub(crate) fn positive(self) -> Option<(i64, NodeRecord)> {
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
pub(crate) fn state_at(
    versions: impl IntoIterator<Item = (i64, NodeRecord)>,
    at: i64,
    fields: TimelineFields,
) -> StateAt {
    // The version starting latest at or before `at`: a later start
    // supersedes an earlier one at every instant they share.
    let mut covering = None;
    for (valid_from, record) in versions {
        if valid_from > at {
            break;
        }
        covering = Some((valid_from, record));
    }
    let Some((valid_from, record)) = covering else {
        return StateAt::Absent;
    };
    let ended = fields
        .valid_to
        .and_then(|id| record.props.get(&id))
        .and_then(|value| match value {
            Value::Int(t) | Value::Timestamp(t) => Some(*t),
            _ => None,
        })
        .is_some_and(|valid_to| valid_to <= at);
    if ended {
        return StateAt::Absent;
    }
    let deleted = fields
        .deleted
        .and_then(|id| record.props.get(&id))
        .is_some_and(|value| matches!(value, Value::Bool(true)));
    if deleted {
        return StateAt::Deleted;
    }
    StateAt::Positive { valid_from, record }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
