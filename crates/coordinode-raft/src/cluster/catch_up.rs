//! A member whose disk lost log entries its group had committed stays out of
//! elections until a leader has sent them again.
//!
//! Its shorter log would let it vote for a candidate missing them, and a
//! leader elected that way overwrites them on every member. So it neither
//! campaigns nor grants a vote until it holds an entry that a leader, writing
//! to it since it reopened, reports committed in that leader's own term, at
//! or past the lost one.
//!
//! Such an entry is enough. A leader's committed entry of its own term
//! follows every entry committed before it (Leader Completeness, Raft §5.4).
//! And a leader reading this member's replies on a connection opened after
//! the restart has already counted every acknowledgement the member sent
//! before it, so no entry committed on a lost copy is left out of what that
//! leader reports. Nothing the member's restored state says counts: it holds
//! the vote and the commit point the member had before the loss.

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use coordinode_storage::engine::core::StorageEngine;
use openraft::async_runtime::watch::WatchReceiver as _;

use super::RaftInstance;
use crate::storage::{TypeConfig, clear_catch_up};

/// Stored when nothing is lost; no log index takes this value.
const NONE: u64 = u64::MAX;

struct Inner {
    /// The highest committed index the member lost, until it holds it again.
    lost: AtomicU64,
    /// One above the highest index a leader reported committed in its own
    /// term since the member reopened; zero before any report.
    reported: AtomicU64,
    /// Woken by a new report.
    reports: tokio::sync::Notify,
}

/// What this member lost and what its leaders report, shared by its consensus
/// handlers and the task that lifts it.
#[derive(Clone)]
pub(crate) struct CatchUp(Arc<Inner>);

impl CatchUp {
    /// A member that lost committed entries through `through`, or none.
    pub(crate) fn new(through: Option<u64>) -> Self {
        Self(Arc::new(Inner {
            lost: AtomicU64::new(through.unwrap_or(NONE)),
            reported: AtomicU64::new(0),
            reports: tokio::sync::Notify::new(),
        }))
    }

    /// Whether the member still lacks committed entries.
    pub(crate) fn pending(&self) -> bool {
        self.lost().is_some()
    }

    fn lost(&self) -> Option<u64> {
        match self.0.lost.load(Ordering::Acquire) {
            NONE => None,
            index => Some(index),
        }
    }

    fn reported(&self) -> Option<u64> {
        self.0.reported.load(Ordering::Acquire).checked_sub(1)
    }

    /// Take note of what a leader writing to this member reports committed.
    /// Only a commit of the leader's own term counts.
    pub(crate) fn note(&self, req: &openraft::raft::AppendEntriesRequest<TypeConfig>) {
        if !self.pending() {
            return;
        }
        let Some(committed) = &req.leader_commit else {
            return;
        };
        if committed.committed_leader_id().term != req.vote.leader_id().term {
            return;
        }
        // Held one above the index, so the empty cell (zero) sorts below
        // every report; a log index is a Raft index and never reaches the
        // top of the range.
        let above = committed.index + 1;
        if self.0.reported.fetch_max(above, Ordering::AcqRel) < above {
            self.0.reports.notify_one();
        }
    }

    fn clear(&self) {
        self.0.lost.store(NONE, Ordering::Release);
    }
}

/// Whether the member holds a commit reported to it at or past `lost`.
fn caught_up(local_committed: Option<u64>, reported: Option<u64>, lost: u64) -> bool {
    match (local_committed, reported) {
        (Some(held), Some(reported)) => reported >= lost && held >= reported,
        _ => false,
    }
}

/// Wait until the member has caught up, then forget the loss durably and let
/// it campaign again when `elect` says the node does. A restart before then
/// finds the loss still recorded. Ends without lifting when the consensus
/// stops first.
pub(crate) fn spawn_lift(
    raft: Arc<RaftInstance>,
    engine: Arc<StorageEngine>,
    work: crate::storage::EngineWork,
    guard: CatchUp,
    elect: bool,
) -> Option<tokio::task::JoinHandle<()>> {
    let lost = guard.lost()?;
    Some(tokio::spawn(async move {
        let mut metrics = raft.metrics();
        loop {
            let held = metrics
                .borrow_watched()
                .local_committed
                .as_ref()
                .map(|id| id.index);
            if caught_up(held, guard.reported(), lost) {
                break;
            }
            tokio::select! {
                changed = metrics.changed() => if changed.is_err() { return },
                () = guard.0.reports.notified() => {}
            }
        }
        // Forgotten on disk only now that the entries are back: a member that
        // stops before this finds the loss still recorded when it restarts.
        // A shutdown aborts this task but not the blocking write, which holds
        // the engine until it ends: counted as work the shutdown waits out.
        let running = work.start();
        let cleared = tokio::task::spawn_blocking(move || {
            // Locals drop in reverse: the engine before the guard.
            let _running = running;
            let engine = engine;
            clear_catch_up(&engine)
        })
        .await;
        match cleared {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::error!(error = %e, "raft: could not record that this member caught up");
                return;
            }
            Err(e) => {
                tracing::error!(error = %e, "raft: recording that this member caught up failed");
                return;
            }
        }
        guard.clear();
        raft.runtime_config().elect(elect);
        tracing::info!(
            lost_through = lost,
            "raft: this member holds the entries it lost again and takes part in elections"
        );
    }))
}

#[cfg(test)]
mod tests;
