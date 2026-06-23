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

/// Read concern configuration for a query or transaction.
#[derive(Debug, Clone, Default)]
pub struct ReadConcern {
    /// Consistency level.
    pub level: ReadConcernLevel,

    /// Optional causal fence: wait until node has applied entries
    /// at least up to this Raft log index before reading.
    /// Used with causal consistency sessions (`afterClusterTime`).
    pub after_index: Option<u64>,

    /// Optional explicit snapshot timestamp for `Snapshot` level.
    /// If set with `level = Snapshot`, reads pinned to this MVCC timestamp.
    /// Mutually exclusive with `after_index`.
    pub at_timestamp: Option<u64>,
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

    /// Validate that the read concern configuration is consistent.
    /// Returns error message if invalid.
    pub fn validate(&self) -> Result<(), &'static str> {
        // afterClusterTime and atClusterTime are mutually exclusive
        if self.after_index.is_some() && self.at_timestamp.is_some() {
            return Err("after_index and at_timestamp are mutually exclusive");
        }

        // linearizable cannot use afterClusterTime
        if self.level == ReadConcernLevel::Linearizable && self.after_index.is_some() {
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
