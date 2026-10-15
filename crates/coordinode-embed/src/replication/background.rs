//! The work a member does only while it leads: finishing the index builds no
//! executor finishes, and dispatching AFTER COMMIT triggers. Both commit
//! through the log, so only the leader runs them, and each looks every time
//! the member becomes the leader, not only the first.

use std::sync::Arc;
use std::time::Duration;

use coordinode_raft::cluster::RaftNode;
use parking_lot::RwLock;

use crate::Database;

/// A failed attempt is tried again after this, not at the next apply.
const RETRY: Duration = Duration::from_secs(5);

/// Watch `raft_node`'s leadership and, each time this member becomes the
/// leader, finish the builds left unfinished: those a process left when it
/// stopped, and those a former leader held when it lost the lead. Ends when
/// the node is dropped.
pub(super) fn spawn_index_build_resumer(
    database: Arc<RwLock<Database>>,
    raft_node: Arc<RaftNode>,
) -> tokio::task::JoinHandle<()> {
    let mut applied = raft_node.subscribe_applied();
    tokio::spawn(async move {
        // Whether the work ran during this member's current lead.
        let mut done_this_lead = false;
        let mut look_now = true;
        loop {
            // A new leader's first entry is among the applied ones, so
            // looking at each catches every change of leader.
            if !std::mem::take(&mut look_now) && applied.changed().await.is_err() {
                break;
            }
            if raft_node.current_leader() != Some(raft_node.node_id()) {
                done_this_lead = false;
                continue;
            }
            if done_this_lead {
                continue;
            }
            let db = Arc::clone(&database);
            match tokio::task::spawn_blocking(move || db.read().resume_interrupted_index_builds())
                .await
            {
                Ok(Ok(resumed)) => {
                    if resumed.published > 0 {
                        tracing::info!(
                            resumed = resumed.published,
                            "interrupted index builds finished"
                        );
                    }
                    if resumed.untaken == 0 {
                        done_this_lead = true;
                        continue;
                    }
                    // Leading, yet the writes were refused: the lead is not
                    // usable for proposals yet. Look again at the next applied
                    // entry, or after a pause when none comes.
                    tracing::debug!(
                        untaken = resumed.untaken,
                        "unfinished index builds not taken yet"
                    );
                    tokio::select! {
                        changed = applied.changed() => if changed.is_err() { break },
                        () = tokio::time::sleep(RETRY) => {}
                    }
                    look_now = true;
                }
                Ok(Err(e)) => {
                    tracing::warn!(%e, "taking up unfinished index builds failed; retrying");
                    tokio::time::sleep(RETRY).await;
                    look_now = true;
                }
                Err(e) => {
                    tracing::warn!(%e, "index build resumer task join error");
                    tokio::time::sleep(RETRY).await;
                    look_now = true;
                }
            }
        }
    })
}

/// Drive AFTER COMMIT trigger dispatch on the leader.
///
/// The event queue is replicated, so every member sees the same backlog;
/// running it only on the leader means one member runs it at a time (the
/// bodies' writes go through the leader's pipeline anyway). A leader change
/// before an event is acknowledged runs it again on the new leader, so a
/// body runs at least once. Woken by each applied entry (a fresh enqueue, a
/// trigger enabled, the entry a new leader appends) and by the earliest retry
/// the last pass left scheduled; with nothing queued it sleeps. Passes are at
/// least `interval` apart. The blocking dispatch runs off the async runtime,
/// so a long body never stalls consensus.
pub(super) fn spawn_trigger_dispatch(
    database: Arc<RwLock<Database>>,
    raft_node: Arc<RaftNode>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    let mut applied = raft_node.subscribe_applied();
    tokio::spawn(async move {
        let mut due: Option<tokio::time::Instant> = None;
        loop {
            tokio::select! {
                changed = applied.changed() => {
                    if changed.is_err() {
                        break; // The node is gone: so is the worker.
                    }
                }
                _ = tokio::time::sleep_until(due.unwrap_or_else(tokio::time::Instant::now)), if due.is_some() => {}
            }
            due = None;
            if raft_node.current_leader() != Some(raft_node.node_id()) {
                continue;
            }
            let passed_at = tokio::time::Instant::now();
            let db = Arc::clone(&database);
            match tokio::task::spawn_blocking(move || db.read().dispatch_after_commit_triggers())
                .await
            {
                Ok(report) => {
                    for e in &report.errors {
                        tracing::warn!("after-commit trigger dispatch: {e}");
                    }
                    // A bookkeeping failure leaves its event due: try again.
                    let retry_in = if report.errors.is_empty() {
                        report.next_due_us.map(|at| {
                            // A retry already due waits for nothing: the
                            // floor at zero is the meaning, not a guard.
                            Duration::from_micros(at.saturating_sub(unix_now_us()))
                        })
                    } else {
                        Some(Duration::ZERO)
                    };
                    due = retry_in.map(|wait| passed_at + wait.max(interval));
                }
                Err(e) => {
                    tracing::warn!(%e, "after-commit dispatch task join error");
                    due = Some(passed_at + interval);
                }
            }
        }
    })
}

/// Wall-clock microseconds since the Unix epoch, the clock the AFTER COMMIT
/// queue schedules retries on.
fn unix_now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_micros() as u64)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
