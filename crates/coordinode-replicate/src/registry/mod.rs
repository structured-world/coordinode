//! `SeqnoConsumerRegistry`: the per-shard record of which consumers need the
//! shard's history, from where, and for how long.
//!
//! The registry feeds the retention of the history consumers read: the
//! lsm-tree GC watermark (MVCC seqno space) and oplog segment retention
//! (oplog index space). Each consumer registers with an explicit retention
//! policy: STRICT keeps its protection until it is cancelled, BOUNDED until
//! a declared bound is crossed. A shard's floor in each space is the minimum
//! checkpoint over its live registrations.
//!
//! Public surface (types and trait) plus the replicated
//! [`ShardConsumerRegistry`] implementation, whose every transition is a
//! transaction conditioned on the record it read, and its background service
//! ([`RegistryBackground`]): coalesced heartbeats and the sweep that ends a
//! BOUNDED registration once its bound is crossed.

mod entry;
mod shard;
mod source;
mod types;

pub use shard::{BackgroundConfig, Clock, RegistryBackground, ShardConsumerRegistry, SystemClock};
pub use source::RetentionSource;
pub use types::{
    ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy, ConsumerSnapshot, InitialSeqno,
    RegisteredHandle, RegistrationState, RegistryError, TerminalReason, TopologyScope,
    ValidatedRetentionBounds,
};

/// Per-shard accounting of consumer retention checkpoints.
///
/// `heartbeat` is the only high-frequency call (batched); `checkpoint`
/// advances the consumer's progress and, transitively, the shard floor.
/// Every call made through a handle is refused once the incarnation it names
/// has ended.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a `SeqnoConsumerRegistry`",
    label = "this type cannot account for consumer retention",
    note = "use the replicated registry in `coordinode-replicate`, or implement \
            `SeqnoConsumerRegistry` for a custom retention source"
)]
pub trait SeqnoConsumerRegistry {
    /// Register a consumer on this shard, returning a handle to a new
    /// incarnation.
    ///
    /// # Errors
    /// [`RegistryError::EmptyConsumerId`], [`RegistryError::UnsupportedScope`],
    /// [`RegistryError::InvalidRetention`] when the policy cannot be admitted,
    /// [`RegistryError::RetentionLost`] when the starting position is no
    /// longer held, [`RegistryError::AlreadyRegistered`] while another
    /// incarnation of the id is live, and the commit failures.
    fn register(&self, reg: ConsumerRegistration) -> Result<RegisteredHandle, RegistryError>;

    /// Advance the consumer's checkpoint to `seqno`; a lower one is ignored.
    ///
    /// # Errors
    /// The handle refusals ([`RegistryError::UnknownConsumer`],
    /// [`RegistryError::StaleIncarnation`], [`RegistryError::Terminated`]) and
    /// the commit failures.
    fn checkpoint(&self, handle: &RegisteredHandle, seqno: u64) -> Result<(), RegistryError>;

    /// Record that the consumer is alive. Proves liveness only: it never
    /// moves the checkpoint or resets progress age.
    ///
    /// # Errors
    /// The handle refusals when written at once; a buffered heartbeat of an
    /// ended incarnation is dropped when flushed.
    fn heartbeat(&self, handle: &RegisteredHandle) -> Result<(), RegistryError>;

    /// Cancel the registration: its incarnation ends and stops holding the
    /// source. The record stays, refusing the handle from now on.
    ///
    /// # Errors
    /// The handle refusals and the commit failures.
    fn unregister(&self, handle: RegisteredHandle) -> Result<(), RegistryError>;

    /// The MVCC-space floor: `min(checkpoint_seqno)` over live MVCC-space
    /// registrations, or `u64::MAX` when there is none.
    fn shard_floor(&self) -> u64;

    /// Every registration on this shard, live or ended (ops / debugging).
    fn list_consumers(&self) -> Vec<ConsumerSnapshot>;
}
