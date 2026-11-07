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
//! The embedded database consumes the registry always and the transfer when
//! it is a member of a replicated group, so this crate does not depend on it.
//!
//! Features: `transfer` (default) adds segment transfer and its gRPC stubs;
//! `async` adds awaiting a registration's change. Without them the registry
//! needs no async runtime.
//!
//! Tier: `std-only`: wraps a proposal pipeline, threads and, with
//! `transfer`, tokio-backed transfer; not a no-std target.

#![forbid(unsafe_code)]

pub mod registry;
#[cfg(feature = "transfer")]
pub mod segment_store;
#[cfg(feature = "transfer")]
pub mod transfer;

pub use registry::{
    BackgroundConfig, Clock, ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy,
    ConsumerSnapshot, InitialSeqno, LogRetentionSource, PriorIncarnation, RegisteredHandle,
    RegistrationState, RegistrationWatch, RegistryBackground, RegistryError, RetentionPressure,
    RetentionSource, RetentionSpace, RetentionSummary, SeqnoConsumerRegistry,
    ShardConsumerRegistry, SystemClock, TerminalReason, TopologyScope, ValidatedRetentionBounds,
};
#[cfg(feature = "transfer")]
pub use segment_store::{RepairError, SegmentInstaller};
#[cfg(feature = "transfer")]
pub use transfer::PeerLinks;

// Re-exported so server-side callers can name the causal token type
// without reaching past the replication layer into core directly.
pub use coordinode_core::txn::proposal::ProposalOutcome;

// Re-exported so server-side callers (e.g. the background-scrub repair trigger)
// can name the wire piece encoding without depending on coordinode-swarm
// directly.
#[cfg(feature = "transfer")]
pub use coordinode_swarm::PieceEncoding;
