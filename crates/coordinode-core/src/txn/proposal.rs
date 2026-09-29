//! Raft proposal pipeline: abstraction for replicating mutations.
//!
//! The proposal pipeline sits between OCC validation (executor) and storage
//! (CoordiNode storage). In single-node CE, the [`ProposalPipeline`] implementation applies
//! mutations directly. In distributed CE, it replicates via Raft before
//! applying.
//!
//! ## Flow
//!
//! ```text
//! Executor: OCC check → assign commit_ts → create RaftProposal
//!   → pipeline.propose_and_wait(proposal)
//!     → [Raft replication in cluster mode]
//!     → apply mutations to MvccEngine at commit_ts
//!     → ACK
//! ```
//!
//! ## Design decisions
//!
//! - OCC check happens BEFORE the proposal (no wasted Raft bandwidth on
//!   doomed transactions).
//! - `commit_ts` is assigned BEFORE the proposal so all replicas apply at the
//!   same timestamp.
//! - The trait is synchronous for now (single-node). Distributed mode will
//!   add async propose-and-wait with error channels.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use super::timestamp::Timestamp;

/// Unique proposal identifier for deduplication.
///
/// In single-node mode, dedup is not strictly needed but the ID generator
/// prepares the abstraction for Raft replay scenarios where
/// leader changes can cause proposal re-delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProposalId(u64);

impl ProposalId {
    /// Create from raw value (for testing / deserialization).
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Get the raw u64 value.
    pub fn as_raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ProposalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "prop:{}", self.0)
    }
}

/// Monotonic proposal ID generator.
///
/// Thread-safe, lock-free. Each call to `next()` returns a unique ID: a
/// monotonically increasing u64 counting up from the generator's base. A
/// process draws a random base, so two processes (or one restarted) do not
/// hand out the same ids.
pub struct ProposalIdGenerator {
    counter: AtomicU64,
}

impl ProposalIdGenerator {
    /// Create a new generator starting from 1.
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }

    /// Create a generator whose ids start just after `base`.
    ///
    /// Ids must not repeat across processes: the state machine drops a
    /// proposal whose id and size it has already applied, and it re-applies
    /// the log of earlier incarnations after a restart. A production
    /// generator therefore starts at a base drawn fresh for each process (a
    /// random 64-bit point, so two processes' ranges overlap only with
    /// probability ids-issued / 2^64); a fixed base is for tests.
    pub fn with_base(base: u64) -> Self {
        Self {
            counter: AtomicU64::new(base),
        }
    }

    /// Allocate the next proposal ID.
    pub fn next(&self) -> ProposalId {
        // A random base may sit anywhere in the range; wrapping past the top
        // keeps the sequence unique for all but the one wrap.
        ProposalId(self.counter.fetch_add(1, Ordering::SeqCst).wrapping_add(1))
    }
}

impl Default for ProposalIdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// A single mutation within a proposal.
///
/// Represents a versioned put or delete on a specific partition and key.
/// The value is already serialized (MessagePack for nodes, posting lists
/// for adj, etc.). The commit_ts from the parent proposal is used to
/// encode the versioned key at apply time.
///
/// `Merge` mutations bypass MVCC key versioning — they write raw merge
/// operands directly to StorageEngine, not through MvccEngine. The LSM
/// engine combines operands lazily during reads and compaction. Used for
/// adjacency posting lists (conflict-free concurrent edge writes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Mutation {
    /// Write a key-value pair at the proposal's commit_ts.
    Put {
        /// Storage partition (node, adj, edgeprop, etc.).
        partition: PartitionId,
        /// User key (without MVCC timestamp suffix).
        key: Vec<u8>,
        /// Serialized value.
        value: Vec<u8>,
    },
    /// Write a tombstone (delete marker) at the proposal's commit_ts.
    Delete {
        /// Storage partition.
        partition: PartitionId,
        /// User key (without MVCC timestamp suffix).
        key: Vec<u8>,
    },
    /// Store a merge operand (bypasses MVCC key versioning).
    ///
    /// Applied via `StorageEngine::merge()` — the LSM engine's merge operator
    /// combines operands with any existing base value. For the `Adj` partition,
    /// operands are encoded via `coordinode_storage::engine::merge::encode_*`.
    ///
    /// Merge writes use raw keys (no timestamp suffix). MVCC visibility is
    /// handled by LSM sequence numbers, not application-level timestamps.
    Merge {
        /// Storage partition (typically Adj).
        partition: PartitionId,
        /// Raw key (no MVCC timestamp suffix).
        key: Vec<u8>,
        /// Encoded merge operand.
        operand: Vec<u8>,
    },
    /// Delete every key in the half-open range `[start, end)` with one MVCC
    /// range tombstone. Produced by run-length coalescing a dense
    /// contiguous run of deleted keys, or by a whole-prefix DROP. The range
    /// covers only keys that are all being deleted — never across a gap holding a
    /// surviving key.
    RemoveRange {
        /// Storage partition.
        partition: PartitionId,
        /// Inclusive start bound.
        start: Vec<u8>,
        /// Exclusive end bound.
        end: Vec<u8>,
    },
    /// A metadata change whose effects the ordered application decides
    /// against the state it applies to, rather than the proposer against the
    /// state it last read. Its effects land in the Schema partition.
    Command(MetadataCommand),
    /// Entry maintenance of a DERIVED index: every member derives the
    /// entries from the sealed interpretation and exact inputs carried here,
    /// never from its current catalog or rows.
    Derive(DerivedIndexWork),
}

/// The maintenance binding a DERIVED effect was sealed under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexBinding {
    /// The index's maintenance-policy epoch when the effect was sealed.
    pub epoch: u64,
    /// Everything that decides the index's entries.
    pub interpretation: crate::index::derive::IndexInterpretation,
}

/// Where a DERIVED effect finds a node's new membership.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DerivedSource {
    /// The node record put by the operation at this position of the same
    /// unit: members extract the membership from that record.
    UnitRecord(u32),
    /// The membership itself, when the unit holds no whole record to extract
    /// it from (a merge-operand update), or `None` when the node leaves the
    /// index.
    Values(Option<Vec<crate::graph::types::Value>>),
}

/// One node's membership change in one DERIVED index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedIndexWork {
    /// The binding the change was sealed under.
    pub binding: IndexBinding,
    /// The node.
    pub node_id: u64,
    /// Its membership before the unit, or `None` when it had no entry.
    pub old: Option<Vec<crate::graph::types::Value>>,
    /// Its membership after the unit.
    pub new: DerivedSource,
}

/// A metadata change decided at ordered application.
///
/// Two proposers reading the same state would pick the same new id; applying
/// the decision in log order is what makes one of them win and the other see
/// the winner. Each command's effects are write-once records (one key per
/// binding, one per lease), so the order commit timestamps put them in never
/// matters, only the order they were applied in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataCommand {
    /// Get-or-create field bindings: each name without one gets the next id
    /// above the dictionary's frontier, in the order given. The whole batch
    /// is refused, publishing nothing, when it is invalid or the id space
    /// cannot hold it.
    RegisterFields {
        /// Names to bind, at most
        /// [`MAX_REGISTRATION_BATCH`](crate::graph::intern::MAX_REGISTRATION_BATCH).
        names: Vec<String>,
    },
    /// Publish exactly these bindings, for data already encoded with them.
    /// Refused as a whole when any contradicts a published binding.
    AdoptFields {
        /// `(name, id)` pairs to publish.
        bindings: Vec<(String, u32)>,
    },
    /// Grant the NodeId range `(base, ceiling]` to the proposer identified by
    /// `token`, only while the granted ceiling is still `base`.
    GrantNodeLease {
        /// The ceiling the proposer read.
        base: u64,
        /// The ceiling it asks for.
        ceiling: u64,
        /// Identifies the grant, so the proposer can tell it won.
        token: [u8; crate::graph::node::NODE_LEASE_TOKEN_LEN],
    },
}

impl MetadataCommand {
    /// Approximate encoded size, for rate limiting and dedup.
    pub fn size_estimate(&self) -> usize {
        match self {
            Self::RegisterFields { names } => names.iter().map(|n| n.len() + 2).sum(),
            Self::AdoptFields { bindings } => bindings.iter().map(|(n, _)| n.len() + 6).sum(),
            Self::GrantNodeLease { token, .. } => 16 + token.len(),
        }
    }
}

impl DerivedIndexWork {
    /// Approximate encoded size, for rate limiting and buffer accounting.
    pub fn size_estimate(&self) -> usize {
        // An indexed value is rarely more than a short string.
        const VALUE_ESTIMATE: usize = 16;
        let interpretation = &self.binding.interpretation;
        let names: usize = interpretation
            .properties
            .iter()
            .map(|p| p.name.len() + 5)
            .sum();
        let values = |v: &Option<Vec<crate::graph::types::Value>>| {
            v.as_ref().map_or(1, |v| v.len() * VALUE_ESTIMATE)
        };
        let new = match &self.new {
            DerivedSource::UnitRecord(_) => 5,
            DerivedSource::Values(v) => values(v),
        };
        16 + interpretation.name.len() + names + values(&self.old) + new
    }
}

impl Mutation {
    /// Approximate encoded size, for rate limiting, dedup and buffer
    /// accounting.
    pub fn size_estimate(&self) -> usize {
        match self {
            Self::Put { key, value, .. } => 1 + key.len() + value.len(),
            Self::Delete { key, .. } => 1 + key.len(),
            Self::Merge { key, operand, .. } => 1 + key.len() + operand.len(),
            Self::RemoveRange { start, end, .. } => 1 + start.len() + end.len(),
            Self::Command(command) => 1 + command.size_estimate(),
            Self::Derive(work) => 1 + work.size_estimate(),
        }
    }

    /// Build a delete mutation for the edge-property body of a
    /// specific `(edge_type, src, tgt)` triple.
    ///
    /// Encapsulates the raw `encode_edgeprop_key` call so callers
    /// don't have to reach into the byte-level key encoder — keeps
    /// the encoder-lockdown gate in `coordinode-query` clean. The
    /// returned variant is the same `Mutation::Delete` shape that
    /// flows through Raft replication; only the construction site
    /// changes.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_core::txn::proposal::Mutation;
    /// use coordinode_core::graph::node::NodeId;
    ///
    /// let m = Mutation::delete_edge_props(
    ///     "FOLLOWS",
    ///     NodeId::from_raw(1),
    ///     NodeId::from_raw(2),
    /// );
    /// // Mutation is the standard Delete variant; the typed
    /// // builder just hides the key-encoding step.
    /// match m {
    ///     Mutation::Delete { partition, key } => {
    ///         assert_eq!(
    ///             partition,
    ///             coordinode_core::txn::proposal::PartitionId::EdgeProp,
    ///         );
    ///         assert!(!key.is_empty());
    ///     }
    ///     _ => panic!("expected Delete"),
    /// }
    /// ```
    pub fn delete_edge_props(
        edge_type: &str,
        src: crate::graph::node::NodeId,
        tgt: crate::graph::node::NodeId,
    ) -> Self {
        Self::Delete {
            partition: PartitionId::EdgeProp,
            key: crate::graph::edge::encode_edgeprop_key(edge_type, src, tgt),
        }
    }
}

/// Partition identifier for serialization.
///
/// Mirrors `coordinode_storage::engine::partition::Partition` but without
/// the storage dependency. Converted at the pipeline boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PartitionId {
    Node,
    Adj,
    EdgeProp,
    Blob,
    BlobRef,
    Schema,
    Idx,
    Counter,
    /// `vec:` — f32 vector truth tier. Every vector property's
    /// full-precision bytes live here as the
    /// per-vector source of truth. Cross-shard rerank reads
    /// f32 directly from here (matches Qdrant / Weaviate / ES BBQ
    /// pattern — no intermediate quantized disk tier).
    VectorF32,

    /// `registry:` — per-shard consumer-retention registry.
    /// Holds `ConsumerRegistration` records replicated through this shard's
    /// Raft group; `min(checkpoint_seqno)` over the keyspace is the shard's
    /// retention floor consumed by LSM compaction, oplog retention, and the
    /// tiering validator.
    Registry,
}

/// A batch of mutations to be proposed through Raft.
///
/// Created after OCC validation succeeds. Contains all buffered writes
/// from a single transaction, the assigned commit_ts, and metadata for
/// deduplication.
///
/// ## Serialization
///
/// Serializes as one [compact frame](super::frame), the encoding the log,
/// replication and recovery share. Deserialization also reads the field-wise
/// form proposals were written in before frames, so a log tail written by an
/// earlier release replays; nothing writes that form any more.
#[derive(Debug, Clone, PartialEq)]
pub struct RaftProposal {
    /// Unique proposal ID for deduplication.
    pub id: ProposalId,
    /// All mutations in this transaction.
    pub mutations: Vec<Mutation>,
    /// Commit timestamp — all replicas apply at this exact ts.
    pub commit_ts: Timestamp,
    /// Start timestamp — for audit and debugging.
    pub start_ts: Timestamp,
    /// If true, skip rate limiter acquisition.
    ///
    /// Used for latency-sensitive proposals that must not be delayed:
    /// - Membership changes (add/remove node)
    /// - Delta proposals (commit/abort oracle decisions)
    ///
    /// Throttling these would stall the very commits that release the
    /// limiter's permits, so they skip it entirely.
    pub bypass_rate_limiter: bool,
}

impl Serialize for RaftProposal {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let frame = super::frame::encode_proposal(self).map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&frame)
    }
}

impl<'de> Deserialize<'de> for RaftProposal {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ProposalVisitor)
    }
}

/// The field-wise form of a proposal written before frames.
#[derive(Deserialize)]
struct FieldWiseProposal {
    id: ProposalId,
    mutations: Vec<Mutation>,
    commit_ts: Timestamp,
    start_ts: Timestamp,
    #[serde(default)]
    bypass_rate_limiter: bool,
}

impl From<FieldWiseProposal> for RaftProposal {
    fn from(p: FieldWiseProposal) -> Self {
        Self {
            id: p.id,
            mutations: p.mutations,
            commit_ts: p.commit_ts,
            start_ts: p.start_ts,
            bypass_rate_limiter: p.bypass_rate_limiter,
        }
    }
}

struct ProposalVisitor;

impl<'de> serde::de::Visitor<'de> for ProposalVisitor {
    type Value = RaftProposal;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a proposal frame")
    }

    fn visit_bytes<E: serde::de::Error>(self, frame: &[u8]) -> Result<RaftProposal, E> {
        super::frame::decode_proposal(frame, &super::frame::DecodeLimits::DEFAULT)
            .map_err(E::custom)
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, seq: A) -> Result<RaftProposal, A::Error> {
        FieldWiseProposal::deserialize(serde::de::value::SeqAccessDeserializer::new(seq))
            .map(Into::into)
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, map: A) -> Result<RaftProposal, A::Error> {
        FieldWiseProposal::deserialize(serde::de::value::MapAccessDeserializer::new(map))
            .map(Into::into)
    }
}

impl RaftProposal {
    /// Number of mutations in this proposal.
    pub fn mutation_count(&self) -> usize {
        self.mutations.len()
    }

    /// Approximate serialized size (for dedup and rate limiting).
    pub fn size_estimate(&self) -> usize {
        // id(8) + commit_ts(8) + start_ts(8)
        24 + self
            .mutations
            .iter()
            .map(Mutation::size_estimate)
            .sum::<usize>()
    }
}

/// Error from the proposal pipeline.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProposalError {
    /// Storage-level error during apply.
    #[error("storage error: {0}")]
    Storage(String),

    /// A targeted storage endpoint reached its hard limit and is no
    /// longer accepting writes. Carries the endpoint id and limit
    /// values so callers can route to a different endpoint, surface
    /// a meaningful client error (gRPC `RESOURCE_EXHAUSTED`), or
    /// trigger an operator-driven eviction.
    ///
    /// Distinct from the generic `Storage(String)` variant so that
    /// type information survives the propagation chain
    /// (`StorageError::CapacityExhausted` → `ProposalError` → query /
    /// gRPC layer) instead of being lost in a `to_string()` cast.
    #[error(
        "endpoint {endpoint_id:?} capacity exhausted (used={used_bytes}, hard_limit={hard_limit_bytes})"
    )]
    CapacityExhausted {
        endpoint_id: String,
        used_bytes: u64,
        hard_limit_bytes: u64,
    },

    /// Proposal was a duplicate (already applied). Not an error in
    /// normal operation — Raft replay can deliver the same proposal twice.
    #[error("duplicate proposal: {0}")]
    Duplicate(ProposalId),

    /// Pipeline is shutting down (node is stepping down as leader).
    #[error("pipeline shutting down")]
    ShuttingDown,

    /// The write concern cannot be satisfied by this group as it is: more
    /// acknowledging members were asked for than the group has.
    #[error("invalid write concern: {0}")]
    InvalidWriteConcern(String),

    /// All retries exhausted. Proposal was not committed within the
    /// timeout window (3 attempts: 4s, 8s, 16s).
    #[error("proposal timed out after {retries} retries")]
    Timeout { retries: u32 },

    /// This node is not the Raft leader. The proposal must be forwarded
    /// to the leader node. `leader_id` is `Some` if the leader is known.
    #[error("not leader, forward to {leader_id:?}")]
    NotLeader { leader_id: Option<u64> },

    /// Raft consensus error (e.g., network, quorum lost).
    #[error("raft error: {0}")]
    Raft(String),

    /// WriteConcern timeout: the client-specified `wtimeout` expired before
    /// the proposal was confirmed committed. The data may or may not have
    /// been written — it is NOT rolled back.
    ///
    /// Per MongoDB spec: "when wtimeout fires, the data is not rolled back —
    /// it may or may not replicate eventually. The client must handle this
    /// ambiguity (retry or read-back to verify)."
    ///
    /// Distinct from [`Timeout`](ProposalError::Timeout) which means all
    /// internal retry attempts were exhausted (Raft-level timeout).
    #[error("write concern timeout: {timeout_ms}ms exceeded")]
    WriteConcernTimeout { timeout_ms: u32 },

    /// The proposal cannot be written to the log: it exceeds a frame bound,
    /// or carries DERIVED work no member could derive. Nothing was proposed.
    #[error("proposal refused before the log: {0}")]
    Unencodable(#[from] crate::txn::frame::FrameError),
}

/// Outcome of a successfully applied proposal.
///
/// Carries the Raft committed log index of this specific proposal so the
/// write path can return a faithful causal `operationTime` token to the
/// client. Without it, callers fall back to sampling the node's current
/// applied index, which is not the index of *this* write (it is captured
/// around execution and may precede or overshoot the write's own commit) —
/// the root of the `operationTime` inaccuracy.
///
/// `applied_index` is `None` for the local / embedded pipeline, which has
/// no Raft log and therefore no cluster-wide commit index. Causal sessions
/// are a cluster concept; in embedded mode the absence is correct, not a
/// missing value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProposalOutcome {
    /// Raft committed log index of this proposal, or `None` when applied
    /// through a non-replicated (local / embedded) pipeline.
    pub applied_index: Option<u64>,
}

impl ProposalOutcome {
    /// Outcome for a replicated proposal committed at `index`.
    pub fn replicated(index: u64) -> Self {
        Self {
            applied_index: Some(index),
        }
    }

    /// Outcome for a locally-applied proposal (no Raft index).
    pub fn local() -> Self {
        Self {
            applied_index: None,
        }
    }
}

/// Abstraction for replicating and applying mutation proposals.
///
/// ## Implementations
///
/// - `LocalProposalPipeline` (coordinode-raft): applies directly to
///   MvccEngine. Used in single-node CE and embedded mode.
/// - `RaftProposalPipeline` (coordinode-raft, distributed mode): replicates via
///   openraft, applies after majority ACK.
///
/// ## Contract
///
/// - Caller has already performed OCC validation and assigned commit_ts.
/// - On `Ok(outcome)`, the mutations are durably applied at commit_ts;
///   `outcome.applied_index` is the Raft commit index when replicated.
/// - On `Err(ProposalError::Duplicate)`, the mutations were already applied
///   (safe to ignore).
/// - On `Err(ProposalError::Storage)`, the transaction should be retried.
pub trait ProposalPipeline: Send + Sync {
    /// Propose mutations and wait for durable application.
    ///
    /// In local mode: applies directly (synchronous).
    /// In cluster mode: replicates via Raft, waits for majority
    /// ACK, then applies. Returns after the mutations are durable, with the
    /// committed [`ProposalOutcome`].
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError>;

    /// Propose mutations with a client-specified write concern timeout.
    ///
    /// If `timeout` elapses before the proposal is confirmed committed,
    /// returns [`ProposalError::WriteConcernTimeout`]. The proposal is NOT
    /// cancelled — data may still be committed after timeout fires.
    ///
    /// ## Default implementation
    ///
    /// Delegates to [`propose_and_wait`](Self::propose_and_wait) (ignoring
    /// timeout). This is correct for `LocalProposalPipeline` where proposals
    /// complete in microseconds. `RaftProposalPipeline` overrides with true
    /// async timeout via `tokio::time::timeout`.
    fn propose_with_timeout(
        &self,
        proposal: &RaftProposal,
        timeout: std::time::Duration,
    ) -> Result<ProposalOutcome, ProposalError> {
        let _ = timeout; // default: ignore timeout (local proposals are instant)
        self.propose_and_wait(proposal)
    }

    /// Propose mutations and return once `ack` members hold them.
    ///
    /// `WriteAck::Majority` is [`propose_and_wait`](Self::propose_and_wait).
    /// `WriteAck::Acks(0)` returns as soon as the proposal is on its way
    /// through the leader; `Acks(1)` once the leader's log holds it;
    /// `Acks(n)` once `n` members do. A `timeout` bounds the wait for the
    /// acknowledgement, never the write itself.
    ///
    /// ## Default implementation
    ///
    /// Applies through [`propose_and_wait`](Self::propose_and_wait) /
    /// [`propose_with_timeout`](Self::propose_with_timeout): a local pipeline
    /// has one member, so every `ack` is satisfied by the local apply.
    fn propose_with_ack(
        &self,
        proposal: &RaftProposal,
        ack: crate::txn::write_concern::WriteAck,
        timeout: Option<std::time::Duration>,
    ) -> Result<ProposalOutcome, ProposalError> {
        let _ = ack;
        match timeout {
            Some(t) => self.propose_with_timeout(proposal, t),
            None => self.propose_and_wait(proposal),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
