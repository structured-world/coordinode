//! Read fence: consistency controls for follower reads.
//!
//! Implements the read-side consistency model: controls which node may serve
//! a read and what data the node must have applied before returning results.
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
//!   a causal bound on top of it is established by [`ReadFence::await_causal`]
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
    /// `Local` (all applied data is already committed). A causal bound is the
    /// read's own setting, checked by [`ReadFence::await_causal`].
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
    /// Returned to the client, which retries on a follower.
    #[error(
        "readPreference=SECONDARY requires a follower node. \
         This node is the Raft leader. Use SECONDARY_PREFERRED for automatic fallback."
    )]
    NotFollower,

    /// Caller requested `Primary` but this node is not the Raft leader.
    ///
    /// The server forwards such a read to the leader once instead of
    /// returning this error.
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

    /// The leader could not confirm its lease for a linearizable read within
    /// the fence timeout.
    #[error("Read fence timeout: leadership not confirmed within {timeout_ms} ms")]
    LeaseTimeout { timeout_ms: u64 },

    /// Underlying Raft error (e.g., not initialized, fatal state).
    #[error("Raft error during read fence: {0}")]
    Raft(String),

    /// This member does not run its group's version: it serves reads only as
    /// of what it last applied, so a read that needs proof of currency (the
    /// leader, a majority or linearizable concern, a causal wait on a later
    /// position) is refused.
    #[error("this member is read-only and cannot prove its reads current: {0}")]
    ReadOnly(coordinode_core::version::Mismatch),

    /// The read's causal bound is not met here: not every commit at or below
    /// `after_ts` is applied yet, or the log has not reached the read's
    /// position. Refused at once unless the read asked to wait, and after
    /// its wait otherwise; never answered at an earlier point.
    #[error(
        "this member is not ready for the read's causal bound: {missing}. \
         Retry, read on another member, or allow the read to wait"
    )]
    NotReady {
        /// What is missing, for the message.
        missing: String,
        /// Whether the read waited for it.
        waited: bool,
    },

    /// The read's causal position counts entries of another history than
    /// this member's (another deployment or group): it cannot be compared
    /// with this member's log.
    #[error(
        "the read's causal position is in another history ({requested}) than this \
         member's ({serving}); positions of different histories are not comparable"
    )]
    ForeignHistory {
        /// The history the read's position names.
        requested: String,
        /// This member's history, or what is unknown about it.
        serving: String,
    },
}

/// A history in text: its deployment and group.
fn history_text(history: &coordinode_core::txn::read_concern::HistoryId) -> String {
    format!("deployment {} {}", history.deployment, history.group)
}

/// Per-request read fence handle.
///
/// Created via [`RaftNode::read_fence()`](crate::cluster::RaftNode::read_fence). Apply before executing a query
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
    /// This member's version view; a member that does not run its group's
    /// version serves only reads that need no proof of currency.
    version: Option<Arc<crate::cluster::version::VersionGate>>,
    /// The store, whose closed bound tells which commits are all applied.
    engine: Option<crate::engine_hold::EngineHold>,
    /// The history this member's log belongs to; `None` before its
    /// deployment is known.
    history: Option<coordinode_core::txn::read_concern::HistoryId>,
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
            version: None,
            engine: None,
            history: None,
        }
    }

    /// Judge causal bounds against `engine`'s closed bound and this member's
    /// `history`.
    pub(crate) fn with_causal_source(
        mut self,
        engine: crate::engine_hold::EngineHold,
        history: Option<coordinode_core::txn::read_concern::HistoryId>,
    ) -> Self {
        self.engine = Some(engine);
        self.history = history;
        self
    }

    /// The last commit timestamp at or below which every commit is applied
    /// here; `None` while nothing is known complete.
    pub fn complete_through(&self) -> Option<u64> {
        self.engine
            .as_ref()
            .and_then(|engine| engine.complete_cut())
    }

    /// This member's history, `None` before its deployment is known.
    pub fn history(&self) -> Option<coordinode_core::txn::read_concern::HistoryId> {
        self.history
    }

    /// Hold reads to what `gate` says this member may serve.
    pub(crate) fn with_version_gate(
        mut self,
        gate: Arc<crate::cluster::version::VersionGate>,
    ) -> Self {
        self.version = Some(gate);
        self
    }

    /// Why this member is read-only, `None` while it runs its group's
    /// version.
    fn read_only(&self) -> Option<coordinode_core::version::Mismatch> {
        match self.version.as_ref()?.state() {
            crate::cluster::version::MemberState::Mismatched(m) => Some(m),
            crate::cluster::version::MemberState::Matched => None,
        }
    }

    /// The commit timestamp a read served here is as of, while this member
    /// is read-only; `None` while it runs its group's version and serves
    /// current reads.
    pub fn as_of(&self) -> Option<u64> {
        self.read_only().map(|m| m.as_of)
    }

    /// This member's locally observed role, for routing and response metadata.
    ///
    /// This is not proof of authority or freshness. Those checks belong to
    /// the requested read concern; asking them here would make even LOCAL
    /// reads wait for a committed entry that has not applied yet. A frozen
    /// member cannot serve as its group's leader.
    pub fn locally_leads(&self) -> bool {
        use openraft::async_runtime::watch::WatchReceiver as _;

        if self.read_only().is_some() {
            return false;
        }
        let receiver = self.raft.metrics();
        let metrics = receiver.borrow_watched();
        metrics.state.is_leader() && metrics.current_leader == Some(metrics.id)
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
        // Bound on the one blocking check here, the linearizable lease read.
        timeout: Duration,
    ) -> Result<(), ReadFenceError> {
        // A read-only member's consensus is a frozen view: what it believes
        // about leadership and lag says nothing current. It serves what it
        // holds, labelled as of (`as_of`), and refuses every read that needs
        // to be current.
        if let Some(m) = self.read_only() {
            let needs_currency = preference == ReadPreference::Primary
                || matches!(concern, ReadConcern::Majority | ReadConcern::Linearizable);
            if needs_currency {
                metrics::counter!("coordinode_version_refused_reads_total").increment(1);
                return Err(ReadFenceError::ReadOnly(m));
            }
            return Ok(());
        }
        let is_leader = self.locally_leads();

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
                // No blocking fence: a causal bound is checked on its own.
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
                tokio::time::timeout(
                    timeout,
                    self.raft
                        .ensure_linearizable(openraft::raft::ReadPolicy::LeaseRead),
                )
                .await
                .map_err(|_| ReadFenceError::LeaseTimeout {
                    timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
                })?
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

    /// Establish the read's causal bounds here: every commit at or below
    /// `after_ts` applied (the store's closed bound past it, which a later
    /// clock alone never gives), and the log applied through
    /// `after_position`, which must count entries of this member's history.
    ///
    /// A position is met once the log is applied through it and the closed
    /// bound covers every commit of that prefix: a view taken at the closed
    /// bound then holds the prefix's effects. It is answered as the commit
    /// timestamp bound it amounts to, which the read's view must reach.
    ///
    /// A bound not met is refused at once as not ready, unless `wait` asks
    /// the read to wait for it: then it waits up to `wait` as entries apply,
    /// and is refused the same way if it is still not met. It is never
    /// answered at an earlier point.
    ///
    /// # Errors
    ///
    /// [`ReadFenceError::ForeignHistory`] for a position of another history,
    /// [`ReadFenceError::NotReady`] for a bound not met (after the wait),
    /// [`ReadFenceError::ReadOnly`] on a member that receives nothing.
    pub async fn await_causal(
        &mut self,
        after_ts: Option<u64>,
        after_position: Option<coordinode_core::txn::read_concern::HistoryPosition>,
        wait: Option<Duration>,
    ) -> Result<Option<u64>, ReadFenceError> {
        if let Some(position) = after_position
            && self.history != Some(position.history)
        {
            return Err(ReadFenceError::ForeignHistory {
                requested: history_text(&position.history),
                serving: self
                    .history
                    .as_ref()
                    .map_or_else(|| "not yet known".to_owned(), history_text),
            });
        }
        let deadline = wait.map(|wait| tokio::time::Instant::now() + wait);
        loop {
            let missing = match self.causal_gap(after_ts, after_position) {
                Ok(bound) => {
                    // A position met, the read allowed a wait and the
                    // prefix is not closed yet: a view chosen at the closed
                    // bound would leave part of it out, so the wait goes on
                    // until it is closed too. Running out of it refuses
                    // nothing here: a latest view holds the prefix already,
                    // and a lower one is refused where it is chosen.
                    if let (Some(deadline), Some(ts)) = (deadline, bound)
                        && self.complete_through().is_none_or(|through| through < ts)
                        && matches!(
                            tokio::time::timeout_at(deadline, self.applied_rx.changed()).await,
                            Ok(Ok(()))
                        )
                    {
                        continue;
                    }
                    return Ok(bound);
                }
                Err(missing) => missing,
            };
            // A read-only member receives nothing, so a later point never
            // arrives: refuse by name instead of waiting out the bound.
            if let Some(m) = self.read_only() {
                metrics::counter!("coordinode_version_refused_reads_total").increment(1);
                return Err(ReadFenceError::ReadOnly(m));
            }
            let Some(deadline) = deadline else {
                return Err(ReadFenceError::NotReady {
                    missing,
                    waited: false,
                });
            };
            // A closed state machine applies nothing more, like a passed
            // deadline.
            match tokio::time::timeout_at(deadline, self.applied_rx.changed()).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) | Err(_) => {
                    return Err(ReadFenceError::NotReady {
                        missing,
                        waited: true,
                    });
                }
            }
        }
    }

    /// The commit timestamp bound the causal bounds amount to when they are
    /// met here, or what of them is not.
    fn causal_gap(
        &self,
        after_ts: Option<u64>,
        after_position: Option<coordinode_core::txn::read_concern::HistoryPosition>,
    ) -> Result<Option<u64>, String> {
        if let Some(ts) = after_ts {
            let complete = self.complete_through();
            if complete.is_none_or(|through| through < ts) {
                return Err(match complete {
                    Some(through) => format!(
                        "commits at or below {ts} are not all applied (complete through {through})"
                    ),
                    None => format!("commits at or below {ts} are not known to be applied"),
                });
            }
        }
        let mut bound = after_ts;
        if let Some(position) = after_position {
            let applied = *self.applied_rx.borrow();
            if applied < position.index {
                return Err(format!(
                    "the log is applied through {applied}, the read needs {}",
                    position.index
                ));
            }
            // Applied through the position, the prefix is in this member's
            // latest view. A view chosen lower (at the closed bound) holds it
            // only up to the prefix's commit ceiling, which the read carries
            // on as the timestamp its view must reach; it is not required
            // complete here, since a latest view needs no closed bound.
            let prefix = self
                .engine
                .as_ref()
                .map(|engine| engine.commit_ceiling_through(position.index))
                .filter(|ceiling| *ceiling > 0);
            bound = bound.max(prefix);
        }
        Ok(bound)
    }

    /// The commit timestamp bound a linearizable read amounts to once its
    /// authority barrier passed: the commits of the log applied here
    /// through the barrier, which the read's view must hold. `None` without
    /// a store to tell.
    pub fn barrier_bound(&self) -> Option<u64> {
        let applied = *self.applied_rx.borrow();
        self.engine
            .as_ref()
            .map(|engine| engine.commit_ceiling_through(applied))
    }

    /// Return the current applied log index on this node, a position in
    /// [`Self::history`].
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
