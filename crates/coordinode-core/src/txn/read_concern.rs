//! Read concern levels for controlling read consistency guarantees.
//!
//! Read concerns determine what data a query sees relative to the cluster's
//! replication state. Higher levels provide stronger guarantees at the cost
//! of latency.
//!
//! ## Levels
//!
//! - **Local**: Read latest data on current node. No replication guarantee.
//!   May see data that will be rolled back if leader crashes.
//! - **Majority**: Read data committed by a majority of Raft voters.
//!   Cannot be rolled back. Production default.
//! - **Linearizable**: Strictest. Leader writes a no-op Raft entry,
//!   waits for majority ACK, then reads. Guarantees real-time ordering.
//!   Primary-only.
//! - **Snapshot**: Read from a specific MVCC timestamp (pinned view).
//!   Used for multi-statement transactions and temporal queries.

/// Read concern level controlling consistency guarantees.
///
/// Maps to proto `ReadConcernLevel` in `consistency.proto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReadConcernLevel {
    /// Latest data on current node. May be stale or rolled back.
    /// Zero latency overhead. Suitable for caching, telemetry.
    #[default]
    Local,

    /// Data acknowledged by majority of Raft group.
    /// Cannot be rolled back after leader failure.
    /// ~1 RTT latency (wait for commit_index). Production default.
    Majority,

    /// Strictest guarantee: all operations in real-time order.
    /// Leader writes no-op to Raft log, waits for majority ACK.
    /// ~2 RTT latency. Primary-only (error on followers).
    Linearizable,

    /// Point-in-time MVCC snapshot, optionally at explicit timestamp.
    /// Used inside multi-statement transactions for repeatable reads.
    Snapshot,
}

impl ReadConcernLevel {
    /// Whether this level requires the read to be served by the leader.
    pub fn requires_leader(&self) -> bool {
        matches!(self, Self::Linearizable)
    }

    /// Whether this level guarantees no rollback (durable).
    pub fn is_durable(&self) -> bool {
        !matches!(self, Self::Local)
    }
}

/// The consensus history a log position belongs to: one group of one
/// deployment. Positions of different histories are not comparable, and a
/// deployment formed anew is a different history even for the same group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistoryId {
    /// The deployment the group belongs to.
    pub deployment: crate::group::DeploymentId,
    /// The group within it.
    pub group: crate::group::GroupId,
}

/// A position in one history's log: the history it is counted in and the
/// index of the entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HistoryPosition {
    /// The history the index counts entries of.
    pub history: HistoryId,
    /// The log index.
    pub index: u64,
}

/// What a completed operation tells its caller for later causal reads: the
/// timestamp its effects or its view are at, and where its own write landed
/// in its history's log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CausalReceipt {
    /// A write's own commit timestamp, or the timestamp a read's snapshot
    /// was taken at; zero when the operation has none (a write answered
    /// before it committed).
    pub operation_time: u64,
    /// Where the write's own entry is in its history's log; `None` outside a
    /// replicated group and for a read.
    pub position: Option<HistoryPosition>,
}

/// Read concern configuration for a query or transaction.
#[derive(Debug, Clone, Default)]
pub struct ReadConcern {
    /// Consistency level.
    pub level: ReadConcernLevel,

    /// Causal lower bound: the read is taken at a timestamp no earlier than
    /// this, with every commit at or below it applied where it is served.
    /// The largest operation time a causal session has observed.
    pub after_ts: Option<u64>,

    /// Causal position in one history: the read is served only where that
    /// history has applied this entry. A position of another history than
    /// the serving member's is refused.
    pub after_position: Option<HistoryPosition>,

    /// Optional explicit snapshot timestamp for `Snapshot` level.
    /// If set with `level = Snapshot`, reads pinned to this MVCC timestamp.
    /// Mutually exclusive with the causal bounds.
    pub at_timestamp: Option<u64>,

    /// How long the read may wait for a causal bound it cannot meet yet.
    /// `None` waits for nothing: a bound not met is refused as not ready.
    pub wait: Option<core::time::Duration>,
}

impl ReadConcern {
    /// Default: `ReadConcernLevel::Local` with no fences.
    pub fn local() -> Self {
        Self {
            level: ReadConcernLevel::Local,
            ..Default::default()
        }
    }

    /// Majority read concern.
    pub fn majority() -> Self {
        Self {
            level: ReadConcernLevel::Majority,
            ..Default::default()
        }
    }

    /// Linearizable read concern (leader-only).
    pub fn linearizable() -> Self {
        Self {
            level: ReadConcernLevel::Linearizable,
            ..Default::default()
        }
    }

    /// Snapshot read concern at explicit timestamp.
    pub fn snapshot_at(timestamp: u64) -> Self {
        Self {
            level: ReadConcernLevel::Snapshot,
            at_timestamp: Some(timestamp),
            ..Default::default()
        }
    }

    /// Whether the read carries a causal bound.
    pub fn is_causal(&self) -> bool {
        self.after_ts.is_some() || self.after_position.is_some()
    }

    /// Validate that the read concern configuration is consistent.
    /// Returns error message if invalid.
    pub fn validate(&self) -> Result<(), &'static str> {
        // afterClusterTime and atClusterTime are mutually exclusive
        if self.is_causal() && self.at_timestamp.is_some() {
            return Err("a causal bound and at_timestamp are mutually exclusive");
        }

        // linearizable cannot use afterClusterTime
        if self.level == ReadConcernLevel::Linearizable && self.is_causal() {
            return Err("afterClusterTime incompatible with linearizable");
        }

        // atClusterTime only valid with snapshot level
        if self.at_timestamp.is_some() && self.level != ReadConcernLevel::Snapshot {
            return Err("at_timestamp only valid with snapshot read concern");
        }

        Ok(())
    }
}

impl std::fmt::Display for ReadConcernLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => write!(f, "local"),
            Self::Majority => write!(f, "majority"),
            Self::Linearizable => write!(f, "linearizable"),
            Self::Snapshot => write!(f, "snapshot"),
        }
    }
}

#[cfg(test)]
mod tests;
