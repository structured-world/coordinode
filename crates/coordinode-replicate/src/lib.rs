//! `coordinode-replicate` — the replication-orchestration layer.
//!
//! Sits above `coordinode-storage` / `coordinode-raft` / `coordinode-query`
//! and below `coordinode-server`. It owns the cross-cutting replication
//! concerns that must not leak into the consensus engine or the query
//! engine, preserving the layering rule that no crate depends on
//! `coordinode-server` and that the consensus engine never depends on the
//! query engine.
//!
//! Current contents:
//!
//! - [`ReplicatedWriter`] — the write entrypoint the gRPC layer routes
//!   Cypher execution through. The leader materialises the write (the
//!   executor resolves `NOW()` / `RAND()` / trigger results during
//!   execution), the deterministic write-set is proposed through the
//!   injected proposal pipeline, and the committed Raft index is surfaced
//!   on the result as the causal `operationTime` token. Followers apply the
//!   same write-set through the state machine; standalone mode is a
//!   pass-through with no committed index.
//!
//! - [`SeqnoConsumerRegistry`] (ADR-028, R137a) — the per-shard,
//!   Raft-replicated retention registry that feeds the lsm-tree
//!   `gc_watermark`, oplog segment retention, and the tiering-DDL validator.
//!   It lives here because storage / oplog / tiering all consume its
//!   per-shard floor, and its registrations replicate through the same Raft
//!   group as data; co-locating it with [`ReplicatedWriter`] keeps the
//!   replication layer cohesive. The public surface (types + trait) is in
//!   [`registry`]; the Raft-backed implementation, heartbeat batching,
//!   eviction, and downstream feeds land incrementally.
//!
//! Tier: `std-only` — wraps the running database and a tokio-backed Raft
//! proposal pipeline; not a no-std target.

#![forbid(unsafe_code)]

pub mod registry;
pub mod transfer;
mod writer;

pub use registry::{
    BackgroundConfig, Clock, ConsumerKind, ConsumerRegistration, ConsumerSnapshot, InitialSeqno,
    RegisteredHandle, RegistryBackground, RegistryError, SeqnoConsumerRegistry,
    ShardConsumerRegistry, SystemClock, TopologyScope,
};
pub use writer::ReplicatedWriter;

// Re-exported so server-side callers can name the causal token type
// without reaching past the replication layer into core directly.
pub use coordinode_core::txn::proposal::ProposalOutcome;
