//! Public types for the [`SeqnoConsumerRegistry`](super::SeqnoConsumerRegistry).
//!
//! These describe what a consumer registers, how its retention reach is
//! scoped across the topology, and what the registry reports back. The
//! registration record replicates through the shard's Raft group, so the
//! wire-relevant types derive `Serialize`/`Deserialize`; their layout
//! stabilises in the pre-public-release window (ADR-028 consequence 8).

use serde::{Deserialize, Serialize};

/// What kind of consumer a registration represents.
///
/// The kind selects which read API the consumer uses; it does **not** affect
/// retention math (`shard_floor = min(checkpoint_seqno)` regardless of kind).
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
    /// wall-clock microseconds (ADR-007) while the oplog `ResumeToken` is a
    /// Raft log index — so a single cluster-wide `min(checkpoint)` across
    /// both would be meaningless. Retention math is split by space; the
    /// `kind` selects which floor a registration contributes to. (This
    /// refines ADR-028, which described a single kind-agnostic floor.)
    pub fn is_seqno_space(self) -> bool {
        match self {
            // Read at an MVCC seqno (scan_since_seqno / read-at-seqno / the
            // snapshot builder's pinned seqno).
            Self::LsmStateDelta | Self::MvccSnapshotPin | Self::Ephemeral => true,
            // Reads the oplog via a Raft-index `ResumeToken`.
            Self::OplogEvents => false,
        }
    }
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
    /// Auto-evict after this many milliseconds without a heartbeat.
    /// `0` = persistent (never auto-evicted).
    pub ttl_ms: u64,
}

/// Opaque handle returned by [`register`](super::SeqnoConsumerRegistry::register).
///
/// Required by `checkpoint` / `heartbeat` / `unregister` to address the
/// registration without re-sending the full record. Carries the
/// `consumer_id` it was minted for; it is not transferable between
/// consumers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredHandle {
    consumer_id: String,
}

impl RegisteredHandle {
    /// Mint a handle for an accepted registration.
    pub fn new(consumer_id: impl Into<String>) -> Self {
        Self {
            consumer_id: consumer_id.into(),
        }
    }

    /// The consumer id this handle addresses.
    pub fn consumer_id(&self) -> &str {
        &self.consumer_id
    }
}

/// A point-in-time view of one registration, for ops / debugging
/// ([`list_consumers`](super::SeqnoConsumerRegistry::list_consumers)).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsumerSnapshot {
    /// The consumer's globally-unique id.
    pub consumer_id: String,
    /// Which read API the consumer uses.
    pub kind: ConsumerKind,
    /// The scope this shard's entry was materialised at.
    pub scope: TopologyScope,
    /// Where the registration was originally placed (may be a broader scope
    /// than this shard's entry, e.g. a cluster-wide consumer seen locally).
    pub scope_origin: TopologyScope,
    /// The consumer's last acknowledged checkpoint on this shard.
    pub checkpoint_seqno: u64,
    /// Wall-clock millis of the last heartbeat the leader committed.
    pub last_heartbeat_ts_ms: u64,
    /// Auto-evict TTL in millis (`0` = persistent).
    pub ttl_ms: u64,
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

    /// `checkpoint` / `heartbeat` / `unregister` referenced a handle whose
    /// registration is not present on this shard (never registered, or
    /// already evicted / unregistered).
    #[error("no registration found for consumer {0:?} on this shard")]
    UnknownConsumer(String),

    /// The consumer's checkpoint fell behind the shard's retention floor
    /// (operator-forced GC bump). Reads via the consumer's API return this
    /// instead of silently losing data: `(checkpoint, current_floor)`.
    #[error("retention lost: checkpoint {checkpoint} is below shard floor {floor}")]
    RetentionLost { checkpoint: u64, floor: u64 },

    /// The registry's Raft proposal failed to commit (replication error,
    /// not leader, timed out). Carries the underlying message.
    #[error("registry replication failed: {0}")]
    Replication(String),
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn registration_roundtrips_through_msgpack() {
        // The record replicates through Raft as msgpack; encode/decode must
        // be lossless across every field, including the scope variant.
        let reg = ConsumerRegistration {
            consumer_id: "kafka-sink-eu".to_string(),
            kind: ConsumerKind::OplogEvents,
            scope: TopologyScope::Dc("eu-west".to_string()),
            initial_seqno: InitialSeqno::At(4096),
            ttl_ms: 30_000,
        };
        let bytes = rmp_serde::to_vec(&reg).expect("encode");
        let decoded: ConsumerRegistration = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(reg, decoded);
    }

    #[test]
    fn every_scope_variant_roundtrips() {
        for scope in [
            TopologyScope::Cluster,
            TopologyScope::Dc("dc1".into()),
            TopologyScope::Rack("r7".into()),
            TopologyScope::Node(42),
            TopologyScope::Shard(3),
        ] {
            let bytes = rmp_serde::to_vec(&scope).expect("encode");
            let decoded: TopologyScope = rmp_serde::from_slice(&bytes).expect("decode");
            assert_eq!(scope, decoded);
        }
    }

    #[test]
    fn every_kind_variant_roundtrips() {
        for kind in [
            ConsumerKind::OplogEvents,
            ConsumerKind::LsmStateDelta,
            ConsumerKind::MvccSnapshotPin,
            ConsumerKind::Ephemeral,
        ] {
            let bytes = rmp_serde::to_vec(&kind).expect("encode");
            let decoded: ConsumerKind = rmp_serde::from_slice(&bytes).expect("decode");
            assert_eq!(kind, decoded);
        }
    }

    #[test]
    fn handle_addresses_its_consumer_id() {
        let h = RegisteredHandle::new("c1");
        assert_eq!(h.consumer_id(), "c1");
        // Distinct ids produce distinct (non-interchangeable) handles.
        assert_ne!(h, RegisteredHandle::new("c2"));
    }

    #[test]
    fn retention_lost_error_carries_both_seqnos() {
        let e = RegistryError::RetentionLost {
            checkpoint: 10,
            floor: 25,
        };
        let msg = format!("{e}");
        assert!(msg.contains("10") && msg.contains("25"), "msg: {msg}");
    }
}
