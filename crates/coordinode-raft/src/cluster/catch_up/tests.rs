use super::*;
use crate::storage::{CommittedLeaderId, LogId, Vote};

fn append(
    term: u64,
    committed_term: u64,
    committed: u64,
) -> openraft::raft::AppendEntriesRequest<TypeConfig> {
    openraft::raft::AppendEntriesRequest {
        vote: Vote::new_committed(term, 1),
        prev_log_id: None,
        entries: Vec::new(),
        leader_commit: Some(LogId::new(
            CommittedLeaderId {
                term: committed_term,
                node_id: 1,
            },
            committed,
        )),
    }
}

/// The member's own restored state is no evidence: without a leader's report
/// it never counts as caught up, however far its commit point reads.
#[test]
fn a_restored_commit_point_alone_is_not_caught_up() {
    assert!(!caught_up(Some(1_000), None, 27));
}

/// Caught up takes a report at or past the lost entry and holding it.
#[test]
fn caught_up_takes_a_report_past_the_loss_and_holding_it() {
    assert!(
        !caught_up(Some(30), Some(20), 27),
        "a report short of the loss"
    );
    assert!(!caught_up(Some(29), Some(30), 27), "not holding the report");
    assert!(caught_up(Some(30), Some(30), 27));
}

/// A commit of an earlier term than the leader's own may still lack entries
/// committed on the member's lost copy; only the leader's own-term commit is
/// taken, and reports only rise.
#[test]
fn only_a_leaders_own_term_commit_is_taken() {
    let guard = CatchUp::new(Some(27));
    guard.note(&append(2, 1, 40));
    assert_eq!(guard.reported(), None, "a commit of an earlier term");
    guard.note(&append(2, 2, 35));
    assert_eq!(guard.reported(), Some(35));
    guard.note(&append(2, 2, 31));
    assert_eq!(guard.reported(), Some(35), "reports only rise");
}
