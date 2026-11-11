//! [`RetentionSource`]: what the registry needs to know about the source a
//! consumer reads to admit a registration and judge its bounds.
//!
//! The registry does not know how a source stores its history: an oplog
//! counts Raft indexes in segment files, the MVCC store counts commit
//! timestamps in trees. The node that owns the source answers for it.

use coordinode_storage::engine::oplog_journal::JournalHold;

use super::types::{ConsumerKind, RetentionPressure, Unmeasured};

/// One observation of a checkpoint's work and material
/// ([`RetentionSource::observe`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Observed {
    /// Clock ms the oldest unseen work was produced, `None` when none is.
    pub produced_at_ms: Option<u64>,
    /// Bytes of source material the checkpoint requires.
    pub retained_bytes: u64,
    /// The source state the observation describes, when the source can
    /// change under it ([`RetentionSource::still_current`]); `None` for a
    /// source whose answers do not go stale that way.
    pub generation: Option<u64>,
}

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
    /// `Ok(None)` when the source has proved there is none up to the head.
    /// Source entries that carry no work (nothing a consumer receives) are
    /// passed over: they are not owed.
    ///
    /// # Errors
    ///
    /// [`Unmeasured`] when the source cannot tell: never a proof of no work.
    fn produced_at_ms(&self, kind: ConsumerKind, position: u64) -> Result<Option<u64>, Unmeasured>;

    /// Bytes of source material a checkpoint at `position` requires the
    /// source to keep.
    ///
    /// # Errors
    ///
    /// [`Unmeasured`] when the source cannot account them: never zero bytes.
    fn retained_bytes_from(&self, kind: ConsumerKind, position: u64) -> Result<u64, Unmeasured>;

    /// The work age and the bytes of a checkpoint at `position`, taken from
    /// one observation of the source, so a judgement never pairs an age and
    /// a size of two different states. A source whose two answers are
    /// always consistent may leave the default, which asks each of them.
    ///
    /// # Errors
    ///
    /// [`Unmeasured`] when either cannot be taken.
    fn observe(&self, kind: ConsumerKind, position: u64) -> Result<Observed, Unmeasured> {
        Ok(Observed {
            produced_at_ms: self.produced_at_ms(kind, position)?,
            retained_bytes: self.retained_bytes_from(kind, position)?,
            generation: None,
        })
    }

    /// Whether `observed` still describes the source: checked last before a
    /// registration is ended on it, so material replaced since (a rebuild,
    /// a cleared tree) ends nobody on numbers that no longer hold.
    fn still_current(&self, observed: &Observed) -> bool {
        let _ = observed;
        true
    }

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
