//! `SeqnoConsumerRegistry` — the single source of truth for "what is the
//! oldest seqno still needed by any active consumer" on a shard (ADR-028).
//!
//! The registry unifies three previously-independent retention drivers (the
//! lsm-tree compaction `gc_watermark`, oplog segment retention, and the EE
//! tiering-DDL validator) behind one per-shard, Raft-replicated keyspace.
//! Consumers `register`, then `heartbeat` to stay alive and `checkpoint` to
//! advance their progress; the shard's effective retention floor is
//! `min(checkpoint_seqno)` over every live registration.
//!
//! Public surface (types + trait) plus the Raft-backed
//! [`ShardConsumerRegistry`] implementation: eager register / checkpoint /
//! unregister proposals, batched heartbeats + TTL eviction
//! ([`RegistryBackground`]), and the retention feeds — the MVCC GC watermark
//! (feed a, `min(seqno_floor, time-travel window)`) and the oplog retention
//! floor (feed b). Floors are split by consumer space (MVCC seqno vs oplog
//! Raft index); see [`ConsumerKind::is_seqno_space`].

mod entry;
mod shard;
mod types;

pub use shard::{BackgroundConfig, Clock, RegistryBackground, ShardConsumerRegistry, SystemClock};
pub use types::{
    ConsumerKind, ConsumerRegistration, ConsumerSnapshot, InitialSeqno, RegisteredHandle,
    RegistryError, TopologyScope,
};

/// Per-shard accounting of consumer retention checkpoints.
///
/// `heartbeat` is the only high-frequency call (batched at the leader);
/// `checkpoint` advances the consumer's progress and, transitively, the
/// shard floor (eagerly persisted). `shard_floor` is the canonical retention
/// bound the compaction filter, oplog manager, and tiering validator read.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a `SeqnoConsumerRegistry`",
    label = "this type cannot account for consumer retention",
    note = "use the Raft-backed registry in `coordinode-replicate`, or implement \
            `SeqnoConsumerRegistry` for a custom retention source"
)]
pub trait SeqnoConsumerRegistry {
    /// Register a consumer on this shard, returning a handle for subsequent
    /// `checkpoint` / `heartbeat` / `unregister` calls.
    ///
    /// # Errors
    /// [`RegistryError::EmptyConsumerId`] if `reg.consumer_id` is empty;
    /// [`RegistryError::Replication`] if the registration proposal does not
    /// commit.
    fn register(&self, reg: ConsumerRegistration) -> Result<RegisteredHandle, RegistryError>;

    /// Advance the consumer's checkpoint to `seqno` (eagerly persisted).
    ///
    /// # Errors
    /// [`RegistryError::UnknownConsumer`] if the handle has no live
    /// registration; [`RegistryError::Replication`] on commit failure.
    fn checkpoint(&self, handle: &RegisteredHandle, seqno: u64) -> Result<(), RegistryError>;

    /// Renew the consumer's liveness (batched at the leader).
    ///
    /// # Errors
    /// [`RegistryError::UnknownConsumer`] if the handle has no live
    /// registration.
    fn heartbeat(&self, handle: &RegisteredHandle) -> Result<(), RegistryError>;

    /// Remove the consumer's registration, freeing the retention it pinned.
    ///
    /// # Errors
    /// [`RegistryError::UnknownConsumer`] if already removed;
    /// [`RegistryError::Replication`] on commit failure.
    fn unregister(&self, handle: RegisteredHandle) -> Result<(), RegistryError>;

    /// The canonical retention floor: `min(checkpoint_seqno)` over every live
    /// registration on this shard. No live consumers → no floor constraint
    /// (returns `u64::MAX`, i.e. "retain nothing on behalf of consumers").
    fn shard_floor(&self) -> u64;

    /// Snapshot every live registration on this shard (ops / debugging).
    fn list_consumers(&self) -> Vec<ConsumerSnapshot>;
}
