//! [`RetentionSource`]: what the registry needs to know about the source a
//! consumer reads to admit a registration and judge its bounds.
//!
//! The registry does not know how a source stores its history: an oplog
//! counts Raft indexes in segment files, the MVCC store counts commit
//! timestamps in trees. The node that owns the source answers for it.

use super::types::ConsumerKind;

/// The source behind one consumer position space on a shard.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a `RetentionSource`",
    label = "this type cannot describe the history a consumer reads",
    note = "the server implements it over the Raft log and the MVCC store; tests implement it over fixed positions"
)]
pub trait RetentionSource: Send + Sync {
    /// One past the newest position the source has produced in `kind`'s
    /// space: a consumer checkpointed here has no work outstanding.
    fn head(&self, kind: ConsumerKind) -> u64;

    /// The oldest position the source still holds in `kind`'s space. A
    /// consumer needing anything below it has lost that history.
    fn first_retained(&self, kind: ConsumerKind) -> u64;

    /// Whether the source measures progress age and required bytes for
    /// `kind`, which a BOUNDED limit needs to be judged at all.
    fn accounts(&self, kind: ConsumerKind) -> bool;

    /// Clock ms at which the work at `position` was produced, or `None` when
    /// nothing is there or the source cannot tell.
    fn produced_at_ms(&self, kind: ConsumerKind, position: u64) -> Option<u64>;

    /// Bytes of source material a checkpoint at `position` requires the
    /// source to keep, or `None` when the source cannot account them.
    fn retained_bytes_from(&self, kind: ConsumerKind, position: u64) -> Option<u64>;

    /// Whether the source can take on retention for a new `kind` consumer
    /// now. `false` while its storage is behind on reclaiming what it already
    /// holds: the registration is refused as retryable backpressure.
    fn admits(&self, kind: ConsumerKind) -> bool;
}
