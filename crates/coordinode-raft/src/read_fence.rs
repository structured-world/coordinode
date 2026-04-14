//! Read fence: consistency controls for follower reads.
//!
//! Implements the read-side consistency model described in
//! `arch/distribution/consistency.md`. Controls which node may serve a read
//! and what data the node must have applied before returning results.
//!
//! ## Read Preference
//!
//! Determines routing — which node is allowed to serve this read:
//! - `Primary`: leader only (default, always consistent)
//! - `PrimaryPreferred`: leader if available, else any follower
//! - `Secondary`: followers only (never leader)
//! - `SecondaryPreferred`: followers if available, else leader
//! - `Nearest`: lowest-latency node (CE: local node; EE: Vivaldi-based)
//!
//! ## Read Concern
//!
//! Determines the consistency guarantee of returned data:
//! - `Local`: whatever the node has applied (always safe in Raft — followers
//!   only apply committed entries)
//! - `Majority`: same as Local in CE Raft (all applied data is majority-committed);
//!   distinction matters with `after_index` (R142 causal sessions)
//! - `Linearizable`: leader-only lease read — guarantees reading after all prior
//!   commits. Adds ~1 heartbeat RTT
//! - `Snapshot`: MVCC snapshot at current applied index (CE = same as Majority)
//!
//! ## Staleness Exclusion (CE)
//!
//! A follower is excluded from serving reads if it is more than
//! `CE_STALENESS_THRESHOLD_ENTRIES` log entries behind its last applied state.
//! EE adds configurable timing-based `max_staleness`.

use std::sync::Arc;
use std::time::Duration;

use crate::storage::{CoordinodeStateMachine, TypeConfig};

type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// Maximum lag in log entries before a CE follower is considered stale.
///
/// Corresponds to ~10s of write traffic at 1,000 writes/second. Followers
/// exceeding this threshold return `ReadFenceError::StaleReplica`.
/// EE replaces this with configurable `max_staleness_seconds` per connection.
pub const CE_STALENESS_THRESHOLD_ENTRIES: u64 = 10_000;

/// Default timeout for blocking fence operations (majority wait, linearizable).
pub const READ_FENCE_TIMEOUT: Duration = Duration::from_secs(10);

/// Which node may serve this read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadPreference {
    /// Route to the Raft leader only. Consistent reads. Default.
    #[default]
    Primary,
    /// Route to leader if alive, else any follower.
    PrimaryPreferred,
    /// Route to followers only (offloads leader from read traffic).
    Secondary,
    /// Route to followers if available, else leader (standard scale-out mode).
    SecondaryPreferred,
    /// Lowest latency node (CE: always local node; EE: Vivaldi coordinate-based).
    Nearest,
}

impl ReadPreference {
    /// Convert from the proto enum value.
    ///
    /// `UNSPECIFIED` (0) maps to `Primary` (default).
    pub fn from_proto(v: i32) -> Self {
        // Matches coordinode.v1.replication.ReadPreference enum values.
        match v {
            1 => Self::Primary,
            2 => Self::PrimaryPreferred,
            3 => Self::Secondary,
            4 => Self::SecondaryPreferred,
            5 => Self::Nearest,
            _ => Self::Primary, // UNSPECIFIED → Primary
        }
    }
}

/// What data this read is allowed to observe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadConcern {
    /// Return whatever this node has applied. Fast, may lag leader.
    ///
    /// In Raft, followers only apply committed entries — "local" data is still
    /// majority-committed, but may not include the most recent writes.
    #[default]
    Local,
    /// Return data acknowledged by majority. In CE Raft, this is equivalent to
    /// `Local` (all applied data is already committed). Distinction matters when
    /// the client provides an `after_index` for causal consistency (R142).
    Majority,
    /// Strongest guarantee: node must be leader and confirm lease before reading.
    /// Returns data after all prior commits. Adds ~1 heartbeat RTT.
    /// Only valid on the Raft leader; returns `LinearizableRequiresLeader` on followers.
    Linearizable,
    /// MVCC point-in-time snapshot at current applied index.
    /// CE: equivalent to `Majority` (snapshot taken at `applied_index`).
    Snapshot,
}

impl ReadConcern {
    /// Convert from the proto enum value.
    pub fn from_proto(v: i32) -> Self {
        match v {
            1 => Self::Local,
            2 => Self::Majority,
            3 => Self::Linearizable,
            4 => Self::Snapshot,
            _ => Self::Local, // UNSPECIFIED → Local
        }
    }
}

/// Errors from read fence application.
#[derive(Debug, thiserror::Error)]
pub enum ReadFenceError {
    /// Caller requested `Secondary` but this node is the Raft leader.
    ///
    /// R151 (request router) will redirect to a follower node.
    #[error(
        "readPreference=SECONDARY requires a follower node. \
         This node is the Raft leader. Use SECONDARY_PREFERRED for automatic fallback."
    )]
    NotFollower,

    /// Caller requested `Primary` but this node is not the Raft leader.
    ///
    /// R151 (request router) will redirect to the leader.
    #[error(
        "readPreference=PRIMARY requires the Raft leader. \
         This node is a follower. Use PRIMARY_PREFERRED for automatic fallback."
    )]
    NotLeader,

    /// `readConcern=LINEARIZABLE` is only valid on the Raft leader.
    #[error(
        "readConcern=LINEARIZABLE is only valid on the Raft leader. \
         This node is a follower. Use MAJORITY for follower reads."
    )]
    LinearizableRequiresLeader,

    /// This follower is too far behind the leader's committed index.
    ///
    /// `lag` is the number of log entries between `last_applied` and `last_log_index`.
    /// Wait for the follower to catch up, or route to the leader.
    #[error(
        "This follower is {lag} log entries behind (CE threshold: {threshold}). \
         Wait for replication to catch up before issuing reads."
    )]
    StaleReplica { lag: u64, threshold: u64 },

    /// The majority wait-fence timed out before the applied index reached `target`.
    #[error(
        "Read fence timeout: applied index {current} did not reach target {target} \
         within the deadline"
    )]
    Timeout { target: u64, current: u64 },

    /// Underlying Raft error (e.g., not initialized, fatal state).
    #[error("Raft error during read fence: {0}")]
    Raft(String),
}

/// Per-request read fence handle.
///
/// Created via [`RaftNode::read_fence()`]. Apply before executing a query
/// to enforce read preference and concern guarantees.
///
/// ## Usage
///
/// ```ignore
/// let mut fence = raft_node.read_fence();
/// fence.apply(ReadPreference::SecondaryPreferred, ReadConcern::Local, timeout).await?;
/// // … execute query …
/// let applied = fence.applied_index();  // for response.applied_index
/// ```
pub struct ReadFence {
    /// Cloned receiver from the state machine's applied watermark.
    applied_rx: tokio::sync::watch::Receiver<u64>,
    /// Underlying openraft instance (for leader check + linearizable read).
    raft: Arc<RaftInstance>,
    /// Staleness threshold override. `None` → use `CE_STALENESS_THRESHOLD_ENTRIES`.
    ///
    /// EE sets this per-connection via `max_staleness_seconds` configuration.
    /// Tests use `with_staleness_threshold()` to inject a low value.
    staleness_threshold: Option<u64>,
    /// Inject a specific lag value instead of computing from openraft metrics.
    ///
    /// `None` → compute from live metrics (production path).
    /// `Some(n)` → return n from `staleness_entries()` (EE per-connection
    /// overrides and integration tests). Tests use this to trigger `StaleReplica`
    /// reliably without needing to write 10K+ log entries of real lag.
    staleness_lag_override: Option<u64>,
}

impl ReadFence {
    /// Create a new `ReadFence` from the node's applied watermark and raft instance.
    ///
    /// Called by [`RaftNode::read_fence()`].
    pub(crate) fn new(
        applied_rx: tokio::sync::watch::Receiver<u64>,
        raft: Arc<RaftInstance>,
    ) -> Self {
        Self {
            applied_rx,
            raft,
            staleness_threshold: None,
            staleness_lag_override: None,
        }
    }

    /// Inject a specific lag value for `staleness_entries()`.
    ///
    /// Used by integration tests to trigger `StaleReplica` without needing
    /// to write 10K+ entries of real lag. Also available to EE for
    /// per-connection staleness budget enforcement.
    pub fn with_staleness_lag(mut self, lag: u64) -> Self {
        self.staleness_lag_override = Some(lag);
        self
    }

    /// Override the staleness threshold for this fence.
    ///
    /// Used by EE to apply per-connection `max_staleness_entries` limits.
    /// In tests, set to a low value to force `StaleReplica` without needing
    /// to write 10K log entries.
    pub fn with_staleness_threshold(mut self, threshold: u64) -> Self {
        self.staleness_threshold = Some(threshold);
        self
    }

    /// Apply read preference and concern checks before executing a query.
    ///
    /// Checks:
    /// 1. Whether this node's role satisfies the requested `ReadPreference`
    /// 2. Whether the node's applied index satisfies the requested `ReadConcern`
    /// 3. Whether the node exceeds the CE staleness threshold (for follower reads)
    ///
    /// Returns `Ok(())` when all checks pass and the read may proceed.
    pub async fn apply(
        &mut self,
        preference: ReadPreference,
        concern: ReadConcern,
        // Reserved for R142 causal read timeout (majority wait with after_index).
        _timeout: Duration,
    ) -> Result<(), ReadFenceError> {
        let is_leader = self.check_is_leader().await;

        // --- Step 1: read preference role check ---
        match preference {
            ReadPreference::Primary => {
                if !is_leader {
                    return Err(ReadFenceError::NotLeader);
                }
            }
            ReadPreference::Secondary => {
                if is_leader {
                    return Err(ReadFenceError::NotFollower);
                }
                // Check staleness before committing to serve from this follower.
                self.check_staleness()?;
            }
            ReadPreference::PrimaryPreferred => {
                // Prefer leader, fallback to follower — always OK.
                // If follower, check staleness.
                if !is_leader {
                    self.check_staleness()?;
                }
            }
            ReadPreference::SecondaryPreferred => {
                // Prefer follower, fallback to leader — always OK.
                // If follower, check staleness.
                if !is_leader {
                    self.check_staleness()?;
                }
            }
            ReadPreference::Nearest => {
                // CE: serve from this node regardless of role.
                // Check staleness if follower.
                if !is_leader {
                    self.check_staleness()?;
                }
            }
        }

        // --- Step 2: read concern fence ---
        match concern {
            ReadConcern::Local => {
                // No fence — serve current applied state immediately.
            }
            ReadConcern::Majority => {
                // In Raft, all applied entries on any node are already majority-committed.
                // No blocking fence needed without an explicit `after_index` (R142).
                // CE: equivalent to Local.
            }
            ReadConcern::Snapshot => {
                // CE: MVCC snapshot at current applied index — equivalent to Majority.
                // No blocking fence needed.
            }
            ReadConcern::Linearizable => {
                // Must be leader; confirms leadership via heartbeat lease.
                if !is_leader {
                    return Err(ReadFenceError::LinearizableRequiresLeader);
                }
                self.raft
                    .ensure_linearizable(openraft::raft::ReadPolicy::LeaseRead)
                    .await
                    .map_err(|e| ReadFenceError::Raft(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Apply fence with the default CE timeout (`READ_FENCE_TIMEOUT`).
    pub async fn apply_default(
        &mut self,
        preference: ReadPreference,
        concern: ReadConcern,
    ) -> Result<(), ReadFenceError> {
        self.apply(preference, concern, READ_FENCE_TIMEOUT).await
    }

    /// Wait until the applied index reaches at least `target`.
    ///
    /// Used by R142 causal sessions: the client provides an `after_index`
    /// from a prior write, and the follower waits before returning results.
    pub async fn wait_for_index(
        &mut self,
        target: u64,
        timeout: Duration,
    ) -> Result<(), ReadFenceError> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let current = *self.applied_rx.borrow();
            if current >= target {
                return Ok(());
            }
            let changed = tokio::time::timeout_at(deadline, self.applied_rx.changed()).await;
            match changed {
                Ok(Ok(())) => continue,
                Ok(Err(_)) => {
                    // Sender dropped — state machine shut down.
                    return Err(ReadFenceError::Timeout {
                        target,
                        current: *self.applied_rx.borrow(),
                    });
                }
                Err(_) => {
                    return Err(ReadFenceError::Timeout {
                        target,
                        current: *self.applied_rx.borrow(),
                    });
                }
            }
        }
    }

    /// Return the current applied log index on this node.
    ///
    /// Include in query responses as `QueryStats.applied_index` so clients
    /// can use it for causal reads (R142 `after_index`).
    pub fn applied_index(&self) -> u64 {
        *self.applied_rx.borrow()
    }

    /// Staleness lag: entries received from leader but not yet applied.
    ///
    /// `last_log_index - last_applied.index`
    ///
    /// Positive value means the state machine is catching up.
    /// CE threshold: `CE_STALENESS_THRESHOLD_ENTRIES`.
    pub fn staleness_entries(&self) -> u64 {
        if let Some(lag) = self.staleness_lag_override {
            return lag;
        }

        use openraft::async_runtime::watch::WatchReceiver as _;

        let metrics = self.raft.metrics().borrow_watched().clone();
        let last_log = metrics.last_log_index.unwrap_or(0);
        let last_applied = metrics
            .last_applied
            .as_ref()
            .map(|lid| lid.index)
            .unwrap_or(0);
        last_log.saturating_sub(last_applied)
    }

    // --- Internal helpers ---

    async fn check_is_leader(&self) -> bool {
        self.raft
            .ensure_linearizable(openraft::raft::ReadPolicy::LeaseRead)
            .await
            .is_ok()
    }

    fn check_staleness(&self) -> Result<(), ReadFenceError> {
        let threshold = self
            .staleness_threshold
            .unwrap_or(CE_STALENESS_THRESHOLD_ENTRIES);
        let lag = self.staleness_entries();
        if lag > threshold {
            return Err(ReadFenceError::StaleReplica { lag, threshold });
        }
        Ok(())
    }
}
