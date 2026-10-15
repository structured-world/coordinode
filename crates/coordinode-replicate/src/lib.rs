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
//! - [`SeqnoConsumerRegistry`] — the per-shard,
//!   Raft-replicated retention registry that feeds the lsm-tree
//!   `gc_watermark`, oplog segment retention, and the tiering-DDL validator.
//!   It lives here because storage / oplog / tiering all consume its
//!   per-shard floor, and its registrations replicate through the same Raft
//!   group as data. The public surface (types + trait) is in [`registry`].
//! - [`SegmentInstaller`] and the segment-transfer service: the inter-node
//!   bulk transfer behind repair, migration and resync.
//!
//! The embedded database consumes this crate when it is a member of a
//! replicated group, so this crate does not depend on it.
//!
//! Tier: `std-only`: wraps tokio-backed transfer and a Raft proposal
//! pipeline; not a no-std target.

#![forbid(unsafe_code)]

pub mod registry;
pub mod segment_store;
pub mod transfer;

pub use registry::{
    BackgroundConfig, Clock, ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy,
    ConsumerSnapshot, InitialSeqno, RegisteredHandle, RegistrationState, RegistrationWatch,
    RegistryBackground, RegistryError, RetentionSource, SeqnoConsumerRegistry,
    ShardConsumerRegistry, SystemClock, TerminalReason, TopologyScope, ValidatedRetentionBounds,
};
pub use segment_store::{RepairError, SegmentInstaller};
pub use transfer::PeerLinks;

// Re-exported so server-side callers can name the causal token type
// without reaching past the replication layer into core directly.
pub use coordinode_core::txn::proposal::ProposalOutcome;

// Re-exported so server-side callers (e.g. the background-scrub repair trigger)
// can name the wire piece encoding without depending on coordinode-swarm
// directly.
pub use coordinode_swarm::PieceEncoding;
