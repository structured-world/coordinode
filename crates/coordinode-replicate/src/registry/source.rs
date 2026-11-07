//! [`RetentionSource`]: what the registry needs to know about the source a
//! consumer reads to admit a registration and judge its bounds.
//!
//! The registry does not know how a source stores its history: an oplog
//! counts Raft indexes in segment files, the MVCC store counts commit
//! timestamps in trees. The node that owns the source answers for it.

use coordinode_storage::engine::oplog_journal::JournalHold;

use super::types::{ConsumerKind, RetentionPressure};

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

    /// Clock ms at which the first work at or after `position` was produced,
    /// or `None` when there is none up to the head or the source cannot
    /// tell. Source entries that carry no work (nothing a consumer receives)
    /// are passed over: they are not owed.
    fn produced_at_ms(&self, kind: ConsumerKind, position: u64) -> Option<u64>;

    /// Bytes of source material a checkpoint at `position` requires the
    /// source to keep, or `None` when the source cannot account them.
    fn retained_bytes_from(&self, kind: ConsumerKind, position: u64) -> Option<u64>;

    /// Whether the source can take on retention for a new `kind` consumer
    /// now, or why not: the registration is then refused as retryable
    /// backpressure with that reason.
    ///
    /// # Errors
    ///
    /// The pressure that keeps it from taking on more.
    fn admits(&self, kind: ConsumerKind) -> Result<(), RetentionPressure>;

    /// The lowest checkpoint among live consumers in the log position space
    /// is now `position` (`u64::MAX` with none): nothing below it will be
    /// asked of [`Self::produced_at_ms`] again, so a source may drop what it
    /// keeps to answer for it. A source that keeps nothing ignores it.
    fn release_log_below(&self, _position: u64) {}

    /// Keep the log from `position` on while a registration starting there
    /// commits and publishes the floor that keeps it from then on.
    ///
    /// `Err(first)` when the log no longer holds `position`, `first` being
    /// where it starts now. A log whose purge consults the registry's floor
    /// returns a hold taken under the lock its purge takes; one whose purge
    /// is bounded by other means answers from [`Self::first_retained`] and
    /// holds nothing.
    fn hold_log_from(&self, position: u64) -> Result<Option<JournalHold>, u64> {
        let first = self.first_retained(ConsumerKind::OplogEvents);
        if position < first {
            Err(first)
        } else {
            Ok(None)
        }
    }
}
