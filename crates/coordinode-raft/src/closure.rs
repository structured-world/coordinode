//! The closed bound a leader stamps on each entry it hands to the log.
//!
//! A commit takes its timestamp before its entry reaches the log, and two
//! commits can reach it in the other order, so a member applying the log
//! applies timestamps out of order. The bound an entry carries tells it what
//! it may rely on: every commit timestamp below it is in an earlier entry, or
//! never will be. A member that applied the entry knows every commit below
//! the bound is applied, which is what a cut through the log needs (a
//! derived index claiming freshness, a read waiting for a timestamp).
//!
//! The leader takes the bound from the commits it still has in flight. A
//! timestamp is registered from its allocation until its entry is handed to
//! the log ([`PendingCommits::obligate`] and the commit admission), and the
//! log takes entries in the order they are handed over, so a timestamp below
//! the bound was handed over before the stamped entry, or refused before
//! anything was. A new leader stamps nothing until it has applied an entry of
//! its own term: from then on every entry of an earlier term that will ever
//! commit is applied, its bound included, and its clock is past them.
//!
//! [`PendingCommits::obligate`]: coordinode_storage::engine::pending::PendingCommits::obligate

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::ProposalError;
use coordinode_storage::engine::core::StorageEngine;

use crate::storage::{CoordinodeStateMachine, Request, TypeConfig};

type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// How long a proposal waits for a new leader to apply its term's first
/// entry. A leader commits that entry as soon as it is elected, one round
/// to a quorum; a wait past this is a leader without its quorum.
pub const LEADER_READY_WAIT: Duration = Duration::from_secs(5);

/// Whether this node leads and has applied an entry of its own term.
fn ready(metrics: &openraft::RaftMetrics<TypeConfig>) -> bool {
    metrics.state.is_leader()
        && metrics.vote.is_committed()
        && metrics.last_applied.as_ref().is_some_and(|applied| {
            applied.committed_leader_id().term == metrics.vote.leader_id().term
        })
}

/// Stamp `request` with this leader's closed bound, once the leader may
/// vouch for its term's prefix.
///
/// A node that does not lead stamps nothing: the write it hands over is
/// refused or forwarded, and a leader stamps its own entries.
///
/// # Errors
///
/// [`ProposalError::LeaderNotReady`] when the leader has not applied an
/// entry of its term within [`LEADER_READY_WAIT`];
/// [`ProposalError::BelowClosedBound`] when a proposal's timestamp is below
/// the bound already applied here: an entry before promises no such commit
/// follows.
pub(crate) async fn stamp(
    raft: &RaftInstance,
    engine: &StorageEngine,
    request: &mut Request,
) -> Result<(), ProposalError> {
    use openraft::async_runtime::watch::WatchReceiver as _;

    let Some(oracle) = engine.oracle() else {
        return Ok(());
    };
    let leads = {
        let metrics = raft.metrics();
        let current = metrics.borrow_watched();
        current.state.is_leader()
    };
    if !leads {
        return Ok(());
    }
    wait_ready(raft).await?;
    let closed = engine.closure_frontier();
    if let Some(stale) = request
        .proposals
        .iter()
        .map(|p| p.commit_ts.as_raw())
        .find(|ts| *ts < closed)
    {
        return Err(ProposalError::BelowClosedBound {
            commit_ts: stale,
            closed_below: closed,
        });
    }
    request.closed_below = engine
        .pending_commits()
        .stamp_closed_below(|| oracle.current().as_raw());
    Ok(())
}

/// Wait until this node leads and has applied an entry of its own term.
async fn wait_ready(raft: &RaftInstance) -> Result<(), ProposalError> {
    let waited = raft
        .wait(Some(LEADER_READY_WAIT))
        .metrics(ready, "the leader applies its term's first entry")
        .await;
    waited
        .map(|_| ())
        .map_err(|_| ProposalError::LeaderNotReady {
            waited_ms: u64::try_from(LEADER_READY_WAIT.as_millis()).unwrap_or(u64::MAX),
        })
}

/// Append a closing entry whenever commits handed to the log are not yet
/// covered by a bound and nothing else is coming to cover them: the last
/// entry of a burst carries a bound at or below its own commit, so without
/// this an idle group never tells its members the commit landed.
///
/// Wakes on each withdrawal, waits `settle` for a following entry to cover
/// it, and appends one entry for all that is still uncovered. A closing
/// entry carries no timestamp, so it makes no further one due.
pub(crate) fn spawn_closer(
    raft: Arc<RaftInstance>,
    engine: Arc<StorageEngine>,
    settle: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut seen = 0u64;
        loop {
            // Only the table waits on the blocking pool, so an aborted closer
            // leaves no hold on the store behind it.
            let pending = Arc::clone(engine.pending_commits());
            let woke = tokio::task::spawn_blocking(move || {
                pending.await_withdrawal(seen, Duration::from_secs(1))
            })
            .await;
            let Ok(count) = woke else {
                return;
            };
            if count == seen {
                continue;
            }
            seen = count;
            tokio::time::sleep(settle).await;
            let Some(oracle) = engine.oracle() else {
                continue;
            };
            {
                use openraft::async_runtime::watch::WatchReceiver as _;
                let metrics = raft.metrics();
                if !ready(&metrics.borrow_watched()) {
                    continue;
                }
            }
            let due = engine
                .pending_commits()
                .closure_due(|| oracle.current().as_raw());
            if due.is_none() {
                continue;
            }
            let mut request = Request::closing(0);
            match stamp(&raft, &engine, &mut request).await {
                Ok(()) if request.closed_below > 0 => {
                    if let Err(e) = raft.client_write(request).await {
                        tracing::debug!(error = %e, "closing entry not appended");
                    }
                }
                Ok(()) => {}
                Err(e) => tracing::debug!(error = %e, "closing entry not stamped"),
            }
        }
    })
}
