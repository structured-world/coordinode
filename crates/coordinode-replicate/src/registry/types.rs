//! Public types for the [`SeqnoConsumerRegistry`](super::SeqnoConsumerRegistry).
//!
//! These describe what a consumer registers, how its retention reach is
//! scoped across the topology, and what the registry reports back. The
//! registration record replicates through the shard's Raft group, so the
//! wire-relevant types derive `Serialize`/`Deserialize`; their layout may
//! still change before the first public release.

use serde::{Deserialize, Serialize};

/// What kind of consumer a registration represents.
///
/// The kind selects which read API the consumer uses and which retention
/// floor its checkpoint feeds (see [`ConsumerKind::is_seqno_space`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsumerKind {
    /// Reads the ordered event stream via `OplogTailer::tail_from`
    /// (CDC sinks, materialised-view refresh, triggers).
    OplogEvents,
    /// Reads a seqno-pruned state diff via lsm-tree `scan_since_seqno`
    /// (Raft incremental snapshot, backup, incremental index rebuild).
    LsmStateDelta,
    /// Pins a read-at-seqno snapshot (time-travel queries, long scans).
    MvccSnapshotPin,
    /// Short-lived registration with no durable contract (e.g. a Raft
    /// snapshot builder during a single rebuild).
    Ephemeral,
}

impl ConsumerKind {
    /// Whether this kind's `checkpoint_seqno` lives in the **MVCC seqno**
    /// space (HLC commit-ts, microseconds) and therefore feeds the LSM GC
    /// watermark (feed a), versus the **oplog Raft-index** space which feeds
    /// oplog segment retention (feed b).
    ///
    /// The two are physically distinct counters — the MVCC oracle stamps
    /// wall-clock microseconds while the oplog `ResumeToken` is a
    /// Raft log index — so a single cluster-wide `min(checkpoint)` across
    /// both would be meaningless. Retention math is split by space; the
    /// `kind` selects which floor a registration contributes to.
    pub fn is_seqno_space(self) -> bool {
        match self {
            // Read at an MVCC seqno (scan_since_seqno / read-at-seqno / the
            // snapshot builder's pinned seqno).
            Self::LsmStateDelta | Self::MvccSnapshotPin | Self::Ephemeral => true,
            // Reads the oplog via a Raft-index `ResumeToken`.
            Self::OplogEvents => false,
        }
    }

    /// The kind's name as a metric label.
    pub fn label(self) -> &'static str {
        match self {
            Self::OplogEvents => "oplog_events",
            Self::LsmStateDelta => "lsm_state_delta",
            Self::MvccSnapshotPin => "mvcc_snapshot_pin",
            Self::Ephemeral => "ephemeral",
        }
    }

    /// Every kind, for per-kind gauges that must also report zero.
    pub const ALL: [Self; 4] = [
        Self::OplogEvents,
        Self::LsmStateDelta,
        Self::MvccSnapshotPin,
        Self::Ephemeral,
    ];
}

/// Where in the failure-domain hierarchy a registration takes effect.
///
/// A registration at scope `X` is materialised in every shard whose topology
/// contains `X` in its ancestor chain (`cluster ⊃ dc ⊃ rack ⊃ node ⊃ shard`).
/// DC and rack ids are operator-defined labels; node and shard ids are the
/// cluster's own identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyScope {
    /// Every shard in the cluster.
    Cluster,
    /// Every shard in the named data centre.
    Dc(String),
    /// Every shard in the named rack.
    Rack(String),
    /// Every shard hosted on the given node.
    Node(u64),
    /// A single shard.
    Shard(u16),
}

/// Where a freshly-registered consumer's checkpoint starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitialSeqno {
    /// Start at the shard's current open seqno — only changes from now on.
    FromNow,
    /// Start at the earliest seqno still retained on the shard (replay the
    /// whole available history).
    FromEarliestRetained,
    /// Start at an explicit seqno (resume from a persisted external offset).
    At(u64),
}

/// Finite limits a BOUNDED consumer accepts, validated at construction.
///
/// Crossing either limit permits terminating the registration. Both are
/// required and non-zero: a BOUNDED consumer without a finite bound would be
/// STRICT under another name. The liveness timeout is an additional,
/// separately chosen condition; `None` disables it and only it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatedRetentionBounds {
    max_progress_lag_ms: u64,
    max_retained_bytes: u64,
    liveness_timeout_ms: Option<u64>,
}

impl ValidatedRetentionBounds {
    /// Bounds of `max_progress_lag_ms` (age of the oldest source work the
    /// consumer has not acknowledged), `max_retained_bytes` (source material
    /// its checkpoint requires) and an optional liveness timeout.
    ///
    /// # Errors
    ///
    /// [`RegistryError::InvalidRetention`] when a limit is zero.
    pub fn new(
        max_progress_lag_ms: u64,
        max_retained_bytes: u64,
        liveness_timeout_ms: Option<u64>,
    ) -> Result<Self, RegistryError> {
        if max_progress_lag_ms == 0 {
            return Err(RegistryError::InvalidRetention(
                "a BOUNDED progress-lag limit must be above zero".to_string(),
            ));
        }
        if max_retained_bytes == 0 {
            return Err(RegistryError::InvalidRetention(
                "a BOUNDED retained-bytes limit must be above zero".to_string(),
            ));
        }
        if liveness_timeout_ms == Some(0) {
            return Err(RegistryError::InvalidRetention(
                "a liveness timeout must be above zero; omit it to disable it".to_string(),
            ));
        }
        Ok(Self {
            max_progress_lag_ms,
            max_retained_bytes,
            liveness_timeout_ms,
        })
    }

    /// Most age, in ms, of the oldest unacknowledged source work.
    pub fn max_progress_lag_ms(&self) -> u64 {
        self.max_progress_lag_ms
    }

    /// Most bytes of source material the checkpoint may require.
    pub fn max_retained_bytes(&self) -> u64 {
        self.max_retained_bytes
    }

    /// Longest gap between heartbeats, when the consumer chose one.
    pub fn liveness_timeout_ms(&self) -> Option<u64> {
        self.liveness_timeout_ms
    }
}

/// How long a consumer's source protection lasts. Every registration names
/// one; there is no default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsumerRetentionPolicy {
    /// Protection lasts until the consumer is cancelled. Missing heartbeats
    /// do not end it; new work that would need retention the shard cannot
    /// give is held back instead.
    Strict,
    /// Protection ends, durably and with a reason, once a declared bound is
    /// crossed.
    Bounded(ValidatedRetentionBounds),
}

/// Why a registration ended. Recorded with the terminal state, so a late
/// handle learns which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminalReason {
    /// Cancelled by its owner or an operator.
    Cancelled,
    /// A BOUNDED consumer went longer than its liveness timeout without a
    /// heartbeat.
    LivenessExpired,
    /// A BOUNDED consumer's oldest unacknowledged work grew older than its
    /// progress-lag limit.
    ProgressLagExceeded,
    /// A BOUNDED consumer's checkpoint required more source material than its
    /// retained-bytes limit.
    RetainedBytesExceeded,
}

/// A consumer's registration request.
///
/// `consumer_id` is a globally-unique opaque string; a topology-wide consumer
/// registers in each affected shard with the **same** id (cross-shard
/// ordering is the consumer SDK's concern via HLC merge).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsumerRegistration {
    /// Globally-unique, opaque consumer identifier.
    pub consumer_id: String,
    /// Which read API this consumer uses.
    pub kind: ConsumerKind,
    /// Failure-domain reach of the registration.
    pub scope: TopologyScope,
    /// Where the consumer's checkpoint starts.
    pub initial_seqno: InitialSeqno,
    /// How long its source protection lasts.
    pub retention: ConsumerRetentionPolicy,
}

/// Opaque handle returned by [`register`](super::SeqnoConsumerRegistry::register).
///
/// Addresses one incarnation of a registration: once that incarnation ends,
/// the handle is refused even if a consumer of the same id registers again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredHandle {
    consumer_id: String,
    incarnation: u64,
}

impl RegisteredHandle {
    /// A handle for incarnation `incarnation` of `consumer_id`.
    pub fn new(consumer_id: impl Into<String>, incarnation: u64) -> Self {
        Self {
            consumer_id: consumer_id.into(),
            incarnation,
        }
    }

    /// The consumer id this handle addresses.
    pub fn consumer_id(&self) -> &str {
        &self.consumer_id
    }

    /// The incarnation this handle addresses.
    pub fn incarnation(&self) -> u64 {
        self.incarnation
    }
}

/// Whether a registration still holds its protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegistrationState {
    /// Live: its checkpoint holds the source back.
    Live,
    /// Ended for `reason` at clock ms `at_ms`; it holds nothing.
    Terminated {
        /// Why it ended.
        reason: TerminalReason,
        /// When the ending was decided.
        at_ms: u64,
    },
}

/// A point-in-time view of one registration, for ops / debugging
/// ([`list_consumers`](super::SeqnoConsumerRegistry::list_consumers)).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsumerSnapshot {
    /// The consumer's globally-unique id.
    pub consumer_id: String,
    /// The registration incarnation this view is of.
    pub incarnation: u64,
    /// Which read API the consumer uses.
    pub kind: ConsumerKind,
    /// The scope this shard's entry was materialised at.
    pub scope: TopologyScope,
    /// Where the registration was originally placed (may be a broader scope
    /// than this shard's entry, e.g. a cluster-wide consumer seen locally).
    pub scope_origin: TopologyScope,
    /// How long its source protection lasts.
    pub retention: ConsumerRetentionPolicy,
    /// Live, or ended and why.
    pub state: RegistrationState,
    /// The consumer's last acknowledged checkpoint on this shard.
    pub checkpoint_seqno: u64,
    /// Wall-clock millis of the last heartbeat the leader committed.
    pub last_heartbeat_ts_ms: u64,
}

/// Errors from the [`SeqnoConsumerRegistry`](super::SeqnoConsumerRegistry).
#[derive(Debug, Clone, thiserror::Error)]
pub enum RegistryError {
    /// The registration was rejected because `consumer_id` is empty.
    #[error("consumer_id must be non-empty")]
    EmptyConsumerId,

    /// The registration used a topology scope this deployment does not
    /// support. `dc` / `rack` scopes require a multi-DC topology and are
    /// EE-only; the CE registry accepts only `cluster`, `node`, and `shard`.
    #[error("unsupported topology scope for this deployment: {0}")]
    UnsupportedScope(String),

    /// `checkpoint` / `heartbeat` / `unregister` referenced a consumer that was
    /// never registered on this shard.
    #[error("no registration found for consumer {0:?} on this shard")]
    UnknownConsumer(String),

    /// A registration of this id is live: a second one would take over its
    /// protection without its owner knowing. Resume it with its handle, or
    /// cancel it first.
    #[error("consumer {consumer_id:?} is already registered (incarnation {incarnation})")]
    AlreadyRegistered {
        /// The id asked for.
        consumer_id: String,
        /// The live incarnation holding it.
        incarnation: u64,
    },

    /// The handle's incarnation has ended: its protection is gone, and
    /// nothing done through the handle takes effect. The reason and the last
    /// accepted checkpoint come with it.
    #[error(
        "consumer {consumer_id:?} incarnation {incarnation} ended ({reason:?}) at checkpoint {checkpoint}"
    )]
    Terminated {
        /// The consumer id.
        consumer_id: String,
        /// The ended incarnation.
        incarnation: u64,
        /// Why it ended.
        reason: TerminalReason,
        /// The last checkpoint it had acknowledged.
        checkpoint: u64,
    },

    /// The handle names an incarnation other than the one registered now:
    /// the consumer it was minted for ended and another registered under the
    /// same id.
    #[error(
        "consumer {consumer_id:?}: handle is for incarnation {handle}, the registration is incarnation {current}"
    )]
    StaleIncarnation {
        /// The consumer id.
        consumer_id: String,
        /// The incarnation the handle names.
        handle: u64,
        /// The incarnation registered now.
        current: u64,
    },

    /// The retention policy cannot be admitted: a bound is not finite, or the
    /// source cannot measure what a BOUNDED limit needs for this consumer kind.
    #[error("retention policy refused: {0}")]
    InvalidRetention(String),

    /// The consumer's checkpoint fell behind the shard's retention floor
    /// (operator-forced GC bump). Reads via the consumer's API return this
    /// instead of silently losing data: `(checkpoint, current_floor)`.
    #[error("retention lost: checkpoint {checkpoint} is below shard floor {floor}")]
    RetentionLost { checkpoint: u64, floor: u64 },

    /// The shard is under write pressure: a new registration would add
    /// retention to storage that is already behind on reclaiming it. Nothing
    /// was registered; retry once the pressure eases.
    #[error("the shard is under write pressure; registration not admitted")]
    Backpressure,

    /// This member cannot decide the transition: it is not the leader of the
    /// shard's ordering group. Nothing changed; the leader can.
    #[error("not the leader{}", match leader_id {
        Some(id) => format!("; leader is node {id}"),
        None => String::from("; no leader known yet"),
    })]
    NotLeader {
        /// The member the group last named leader, if any.
        leader_id: Option<u64>,
    },

    /// The registry's Raft proposal failed to commit (replication error,
    /// timed out). Carries the underlying message.
    #[error("registry replication failed: {0}")]
    Replication(String),
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
