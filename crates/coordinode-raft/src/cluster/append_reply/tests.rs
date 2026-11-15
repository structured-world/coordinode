use openraft::raft::{AppendError, AppendResult, ConflictHint};

use super::AppendReply;
use crate::storage::{CommittedLeaderId, LogId, TypeConfig, Vote};

fn log_id(term: u64, index: u64) -> LogId {
    LogId::new(CommittedLeaderId { term, node_id: 1 }, index)
}

fn over_the_wire(result: AppendResult<TypeConfig>) -> AppendResult<TypeConfig> {
    let bytes = rmp_serde::to_vec(&AppendReply::from(result)).unwrap();
    let reply: AppendReply = rmp_serde::from_slice(&bytes).unwrap();
    AppendResult::from(reply)
}

/// Every outcome a follower gives comes back unchanged on the leader,
/// including the conflict hint: dropping it would make the leader probe back
/// one request at a time instead of jumping to the common prefix.
#[test]
fn every_append_outcome_survives_the_wire() {
    let outcomes: Vec<AppendResult<TypeConfig>> = vec![
        Ok(None),
        Ok(Some(log_id(3, 17))),
        Err(AppendError::Conflict {
            log_id: log_id(3, 17),
            hint: None,
        }),
        Err(AppendError::Conflict {
            log_id: log_id(3, 17),
            hint: Some(ConflictHint::new(vec![log_id(1, 4), log_id(2, 9)])),
        }),
        Err(AppendError::HigherVote(Vote::new_committed(5, 2))),
    ];
    for outcome in outcomes {
        assert_eq!(over_the_wire(outcome.clone()), outcome);
    }
}
