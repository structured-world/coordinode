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
/// Thread-safe, lock-free. Each call to `next()` returns a unique ID.
/// Format follows Dgraph pattern: monotonically increasing u64.
/// In distributed mode, the format extends to include node_id
/// to guarantee cluster-wide uniqueness.
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

    /// Create a generator with a base value for cluster-wide uniqueness.
    ///
    /// The base is typically `(node_id as u64) << 48`, giving each node
    /// its own 48-bit counter space (~280 trillion IDs per node).
    /// This follows the Dgraph pattern of embedding node identity in
    /// proposal keys for O(1) dedup across the cluster.
    pub fn with_base(base: u64) -> Self {
        Self {
            counter: AtomicU64::new(base),
        }
    }

    /// Allocate the next proposal ID.
    pub fn next(&self) -> ProposalId {
        ProposalId(self.counter.fetch_add(1, Ordering::SeqCst) + 1)
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
}

/// A batch of mutations to be proposed through Raft.
///
/// Created after OCC validation succeeds. Contains all buffered writes
/// from a single transaction, the assigned commit_ts, and metadata for
/// deduplication.
///
/// ## Serialization
///
/// In distributed mode, the proposal is serialized as:
/// `[proposal_id: 8 bytes BE][bincode(RaftProposal)]`
/// The 8-byte prefix enables O(1) dedup lookup without deserializing
/// the full payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// - Delta proposals (commit/abort oracle decisions, R140+)
    ///
    /// Follows Dgraph pattern: delta proposals bypass the IOU-based
    /// rate limiter entirely (worker/proposal.go:287-290).
    #[serde(default)]
    pub bypass_rate_limiter: bool,
}

impl RaftProposal {
    /// Number of mutations in this proposal.
    pub fn mutation_count(&self) -> usize {
        self.mutations.len()
    }

    /// Approximate serialized size (for dedup and rate limiting).
    pub fn size_estimate(&self) -> usize {
        let mut size = 24; // id(8) + commit_ts(8) + start_ts(8)
        for m in &self.mutations {
            size += match m {
                Mutation::Put { key, value, .. } => 1 + key.len() + value.len(),
                Mutation::Delete { key, .. } => 1 + key.len(),
                Mutation::Merge { key, operand, .. } => 1 + key.len() + operand.len(),
            };
        }
        size
    }
}

/// Error from the proposal pipeline.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProposalError {
    /// Storage-level error during apply.
    #[error("storage error: {0}")]
    Storage(String),

    /// Proposal was a duplicate (already applied). Not an error in
    /// normal operation — Raft replay can deliver the same proposal twice.
    #[error("duplicate proposal: {0}")]
    Duplicate(ProposalId),

    /// Pipeline is shutting down (node is stepping down as leader).
    #[error("pipeline shutting down")]
    ShuttingDown,

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
/// - On `Ok(())`, the mutations are durably applied at commit_ts.
/// - On `Err(ProposalError::Duplicate)`, the mutations were already applied
///   (safe to ignore).
/// - On `Err(ProposalError::Storage)`, the transaction should be retried.
pub trait ProposalPipeline: Send + Sync {
    /// Propose mutations and wait for durable application.
    ///
    /// In local mode: applies directly (synchronous).
    /// In cluster mode: replicates via Raft, waits for majority
    /// ACK, then applies. Returns after the mutations are durable.
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<(), ProposalError>;
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn proposal_id_monotonic() {
        let gen = ProposalIdGenerator::new();
        let id1 = gen.next();
        let id2 = gen.next();
        let id3 = gen.next();
        assert_eq!(id1.as_raw(), 1);
        assert_eq!(id2.as_raw(), 2);
        assert_eq!(id3.as_raw(), 3);
    }

    #[test]
    fn proposal_id_concurrent() {
        use std::collections::BTreeSet;
        use std::sync::Arc;

        let gen = Arc::new(ProposalIdGenerator::new());
        let mut handles = Vec::new();

        for _ in 0..4 {
            let gen = Arc::clone(&gen);
            handles.push(std::thread::spawn(move || {
                (0..1000).map(|_| gen.next().as_raw()).collect::<Vec<_>>()
            }));
        }

        let mut all: BTreeSet<u64> = BTreeSet::new();
        for h in handles {
            for id in h.join().expect("thread panicked") {
                assert!(all.insert(id), "duplicate proposal ID: {id}");
            }
        }
        assert_eq!(all.len(), 4000);
    }

    #[test]
    fn mutation_size_estimate() {
        let proposal = RaftProposal {
            id: ProposalId::from_raw(1),
            mutations: vec![
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: vec![0; 10],
                    value: vec![0; 100],
                },
                Mutation::Delete {
                    partition: PartitionId::Adj,
                    key: vec![0; 20],
                },
            ],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };

        // 24 + (1 + 10 + 100) + (1 + 20) = 156
        assert_eq!(proposal.size_estimate(), 156);
    }

    #[test]
    fn empty_proposal() {
        let proposal = RaftProposal {
            id: ProposalId::from_raw(1),
            mutations: vec![],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };
        assert_eq!(proposal.mutation_count(), 0);
        assert_eq!(proposal.size_estimate(), 24);
    }

    #[test]
    fn proposal_id_display() {
        let id = ProposalId::from_raw(42);
        assert_eq!(format!("{id}"), "prop:42");
    }
}
