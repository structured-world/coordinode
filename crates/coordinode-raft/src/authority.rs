//! Whether a leader can still be the one to commit what it handed to its log.
//!
//! openraft refuses new proposals once a leader's quorum lease runs out, but
//! keeps the ones it accepted before waiting, since they may still commit.
//! A writer waiting on such a proposal is answered here as soon as this node
//! can no longer commit it: it stopped leading, its term moved on, or no
//! quorum acknowledged it for a whole lease. The entry may still commit under
//! another leader, so the answer is that its outcome is unknown, as a MongoDB
//! primary answers the writes waiting on it when it steps down.

use std::time::Duration;

use coordinode_core::txn::proposal::ProposalError;
use openraft::Instant as _;
use openraft::rt::watch::WatchReceiver as _;

use crate::storage::{CoordinodeStateMachine, TypeConfig};

type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// The term this node leads in now, or `None` when it does not lead.
pub(crate) fn leading_term(raft: &RaftInstance) -> Option<u64> {
    let metrics = raft.metrics();
    let m = metrics.borrow_watched();
    m.state.is_leader().then_some(m.current_term)
}

/// Resolve once this node can no longer commit what it handed to its log
/// while leading in `term`, with the error a writer waiting on it gets.
///
/// The lease is openraft's own, `election_timeout_max`: the span after which
/// it refuses new proposals for want of a quorum acknowledgement.
pub(crate) async fn lost(raft: &RaftInstance, term: u64) -> ProposalError {
    let lease = Duration::from_millis(raft.config().election_timeout_max);
    let mut metrics = raft.metrics();
    // A leader no quorum has acknowledged yet counts from the hand-over.
    let since = std::time::Instant::now();
    loop {
        let quiet = {
            let m = metrics.borrow_watched();
            if m.running_state.is_err() || !m.state.is_leader() || m.current_term != term {
                break;
            }
            // A leader that is its own quorum acknowledges itself: no other
            // member's silence can take its authority (openraft reports no
            // quorum ack time for it), only losing the leadership can.
            let self_quorum = m
                .membership_config
                .membership()
                .voter_ids()
                .all(|voter| voter == m.id);
            if self_quorum {
                None
            } else {
                let since_handover = since.elapsed();
                Some(
                    m.last_quorum_acked
                        .map_or(since_handover, |at| at.elapsed().min(since_handover)),
                )
            }
        };
        let left = match quiet {
            Some(quiet) => match lease.checked_sub(quiet).filter(|left| !left.is_zero()) {
                Some(left) => Some(left),
                None => break,
            },
            None => None,
        };
        tokio::select! {
            changed = metrics.changed() => {
                if changed.is_err() {
                    break;
                }
            }
            () = sleep_or_never(left) => {}
        }
    }
    metrics::counter!("coordinode_raft_proposals_total", "status" => "leadership_lost")
        .increment(1);
    ProposalError::LeadershipLost { term }
}

/// Sleep for `left`, or forever when there is no deadline.
async fn sleep_or_never(left: Option<Duration>) {
    match left {
        Some(left) => tokio::time::sleep(left).await,
        None => core::future::pending().await,
    }
}
