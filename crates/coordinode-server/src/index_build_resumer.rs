//! Takes up the index builds no executor finishes: those a process left
//! when it stopped, and those a former leader held when it lost the lead.
//! Builds commit through the log, so only the leader runs them, and it
//! looks each time it becomes the leader, not only the first.

use std::sync::Arc;
use std::time::Duration;

use coordinode_embed::Database;
use coordinode_raft::cluster::RaftNode;

/// A failed attempt is tried again after this, not at the next apply.
const RETRY: Duration = Duration::from_secs(5);

/// Watch `raft_node`'s leadership and, each time this member becomes the
/// leader, finish the builds left unfinished. Ends when the node is dropped.
pub(crate) fn spawn(
    database: Arc<parking_lot::RwLock<Database>>,
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
                    if resumed > 0 {
                        tracing::info!(resumed, "interrupted index builds finished");
                    }
                    done_this_lead = true;
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
