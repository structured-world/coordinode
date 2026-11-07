//! [`LogRetentionSource`]: the history a store holds for its consumers, its
//! log for log consumers and its MVCC store for the others.

use std::path::PathBuf;
use std::sync::Arc;

use coordinode_storage::engine::core::{StorageEngine, WritePressure};
use coordinode_storage::engine::oplog_journal::JournalHold;
use coordinode_storage::oplog::OplogTimeline;
use coordinode_storage::oplog::RetainedFloor;
use coordinode_storage::oplog::tailer::bytes_needed_from;

use super::source::RetentionSource;
use super::types::ConsumerKind;

/// The history a store holds for its consumers: a log (a group's consensus
/// log, or the store's own journal) for log consumers, the MVCC store for the
/// others.
pub struct LogRetentionSource {
    engine: Arc<StorageEngine>,
    /// Every directory holding segments of the log.
    log_dirs: Vec<PathBuf>,
    /// Where the log's segments start, published by the log itself: every
    /// change stream asks on each wake, so it must not list the directories.
    floor: RetainedFloor,
    /// Which log it is, and so where its head is and how it is purged.
    log: Log,
    /// When each log entry was written: the bound checks ask the age of the
    /// same acknowledged positions every sweep, and reading the segment that
    /// holds one would cost its size on every ask.
    timeline: OplogTimeline,
}

/// The log a [`LogRetentionSource`] answers for.
enum Log {
    /// A group's consensus log, purged as the group decides; its head (one
    /// past the last entry a consumer can be given) is what this reports.
    Group(Arc<dyn Fn() -> u64 + Send + Sync>),
    /// The store's own journal, whose purges keep what the registry's floor
    /// and a registration's hold need.
    OwnJournal,
}

impl LogRetentionSource {
    /// A source over `engine`, the log segments in `log_dirs` starting at
    /// `floor`, and the head `head` reports: a group's consensus log, whose
    /// purge is bounded by the group, not by the registry.
    pub fn new(
        engine: Arc<StorageEngine>,
        log_dirs: Vec<PathBuf>,
        floor: RetainedFloor,
        head: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Self {
        Self {
            engine,
            timeline: OplogTimeline::new(log_dirs.clone()),
            log_dirs,
            floor,
            log: Log::Group(head),
        }
    }

    /// A source over `engine` and the journal it keeps of its own: a store
    /// that is not a member of a replicated group. Its purges keep what the
    /// registry's floor names, so a registration holds its position there
    /// while it publishes that floor. `None` when the store keeps no journal.
    ///
    /// # Errors
    ///
    /// The journal could not be inspected.
    pub fn over_own_journal(
        engine: Arc<StorageEngine>,
    ) -> coordinode_storage::error::StorageResult<Option<Self>> {
        let Some((dir, floor)) = engine.journal_location()? else {
            return Ok(None);
        };
        Ok(Some(Self {
            engine,
            timeline: OplogTimeline::new(vec![dir.clone()]),
            log_dirs: vec![dir],
            floor,
            log: Log::OwnJournal,
        }))
    }

    /// One past the last log entry a consumer can be given.
    fn log_head(&self) -> u64 {
        match &self.log {
            Log::Group(head) => head(),
            Log::OwnJournal => match self.engine.journal_next_index() {
                Ok(Some(next)) => next,
                // The journal was there at construction and is never removed.
                Ok(None) => 0,
                Err(e) => {
                    // A poisoned journal takes no more writes: nothing past
                    // its start can be claimed, and a hold there is refused.
                    tracing::error!(error = %e, "registry: journal head unreadable");
                    0
                }
            },
        }
    }
}

impl RetentionSource for LogRetentionSource {
    fn head(&self, kind: ConsumerKind) -> u64 {
        if kind.is_seqno_space() {
            self.engine.snapshot()
        } else {
            self.log_head()
        }
    }

    fn first_retained(&self, kind: ConsumerKind) -> u64 {
        if kind.is_seqno_space() {
            return self.engine.gc_watermark();
        }
        match self.floor.get() {
            Some(first) => first,
            // No segment holds anything: nothing below the head is kept.
            None => self.head(kind),
        }
    }

    fn accounts(&self, kind: ConsumerKind) -> bool {
        // The log knows when each entry was written and how big its segments
        // are; the MVCC store keeps neither per version.
        !kind.is_seqno_space()
    }

    fn produced_at_ms(&self, kind: ConsumerKind, position: u64) -> Option<u64> {
        if kind.is_seqno_space() {
            return None;
        }
        // Work a consumer owes starts at the first committed entry from its
        // position: an entry with no commit carries no event and no time.
        match self.timeline.first_committed_at(position, self.head(kind)) {
            // An entry's timestamp is wall-clock microseconds.
            Ok(ts) => ts.map(|ts| ts / 1_000),
            Err(e) => {
                tracing::warn!(error = %e, position, "registry: log entry time unreadable");
                None
            }
        }
    }

    fn release_log_below(&self, position: u64) {
        self.timeline.release_below(position);
    }

    fn retained_bytes_from(&self, kind: ConsumerKind, position: u64) -> Option<u64> {
        if kind.is_seqno_space() {
            return None;
        }
        match bytes_needed_from(&self.log_dirs, position) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                tracing::warn!(error = %e, "registry: log size unreadable");
                None
            }
        }
    }

    fn admits(&self, _kind: ConsumerKind) -> bool {
        // The log and the MVCC store share the engine's endpoints, so one
        // verdict covers both: Stop means compaction is behind on reclaiming.
        !matches!(self.engine.write_pressure(), WritePressure::Stop)
    }

    fn hold_log_from(&self, position: u64) -> Result<Option<JournalHold>, u64> {
        if matches!(self.log, Log::Group(_)) {
            let first = self.first_retained(ConsumerKind::OplogEvents);
            return if position < first {
                Err(first)
            } else {
                Ok(None)
            };
        }
        match self.engine.hold_journal_from(position) {
            Ok(Some(hold)) => Ok(Some(hold)),
            Ok(None) => Err(self.first_retained(ConsumerKind::OplogEvents)),
            Err(e) => {
                // An unreadable journal holds nothing a consumer could rely on.
                tracing::warn!(error = %e, position, "registry: journal hold refused");
                Err(self.first_retained(ConsumerKind::OplogEvents))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
