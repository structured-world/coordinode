//! Wire form of a follower's answer to one AppendEntries request.
//!
//! openraft's [`AppendError`] carries no serde on purpose: the transport owns
//! its bytes. This form keeps the conflict hint, which lets the leader skip
//! straight to the common prefix instead of probing back request by request.

use openraft::raft::{AppendError, AppendResult, ConflictHint};
use serde::{Deserialize, Serialize};

use crate::storage::{LogId, TypeConfig, Vote};

/// One AppendEntries outcome as it travels between members.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum AppendReply {
    /// The last log id the follower accepted from the request.
    Accepted(Option<LogId>),
    /// The follower's log does not hold the request's previous log id.
    Conflict {
        /// The previous log id of the rejected request.
        log_id: LogId,
        /// Samples of the follower's log after truncation, when it sent them.
        hint: Option<Vec<LogId>>,
    },
    /// The follower has seen a higher vote than the sender's.
    HigherVote(Vote),
}

impl From<AppendResult<TypeConfig>> for AppendReply {
    fn from(result: AppendResult<TypeConfig>) -> Self {
        match result {
            Ok(accepted) => Self::Accepted(accepted),
            Err(AppendError::Conflict { log_id, hint }) => Self::Conflict {
                log_id,
                hint: hint.map(|hint| hint.log_ids),
            },
            Err(AppendError::HigherVote(vote)) => Self::HigherVote(vote),
        }
    }
}

impl From<AppendReply> for AppendResult<TypeConfig> {
    fn from(reply: AppendReply) -> Self {
        match reply {
            AppendReply::Accepted(accepted) => Ok(accepted),
            AppendReply::Conflict { log_id, hint } => Err(AppendError::Conflict {
                log_id,
                hint: hint.map(ConflictHint::new),
            }),
            AppendReply::HigherVote(vote) => Err(AppendError::HigherVote(vote)),
        }
    }
}

impl AppendReply {
    /// The index of the last entry the follower accepted, if it accepted any.
    pub(crate) fn matched(&self) -> Option<u64> {
        match self {
            Self::Accepted(accepted) => accepted.map(|id| id.index),
            Self::Conflict { .. } | Self::HigherVote(_) => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
