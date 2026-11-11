//! [`LogRetentionSource`]: the history a store holds for its consumers, its
//! log for log consumers and its MVCC store for the others.

use std::path::PathBuf;
use std::sync::Arc;

use coordinode_storage::engine::core::{StorageEngine, WritePressure};
use coordinode_storage::engine::oplog_journal::JournalHold;
use coordinode_storage::oplog::OplogTimeline;
use coordinode_storage::oplog::RetainedFloor;
use coordinode_storage::oplog::tailer::bytes_needed_from;

use coordinode_storage::error::StorageError;

use super::source::{Observed, RetentionSource};
use super::types::{ConsumerKind, RetentionPressure, Unmeasured};

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
            return self.engine.oldest_readable_seqno();
        }
        match self.floor.get() {
            Some(first) => first,
            // No segment holds anything: nothing below the head is kept.
            None => self.head(kind),
        }
    }

    fn accounts(&self, kind: ConsumerKind) -> bool {
        // The log knows when each entry was written and how big its segments
        // are. The MVCC store records the logical material of each commit, at
        // its commit timestamp: that is a time only with an oracle, and it is
        // what a state reader keeps, not what a pinned snapshot does.
        match kind {
            ConsumerKind::OplogEvents => true,
            ConsumerKind::LsmStateDelta => self.engine.oracle().is_some(),
            ConsumerKind::MvccSnapshotPin | ConsumerKind::Ephemeral => false,
        }
    }

    fn produced_at_ms(&self, kind: ConsumerKind, position: u64) -> Result<Option<u64>, Unmeasured> {
        if kind.is_seqno_space() {
            return self.observe(kind, position).map(|o| o.produced_at_ms);
        }
        // Work a consumer owes starts at the first committed entry from its
        // position: an entry with no commit carries no event and no time.
        match self.timeline.first_committed_at(position, self.head(kind)) {
            // An entry's timestamp is wall-clock microseconds.
            Ok(ts) => Ok(ts.map(|ts| ts / 1_000)),
            Err(e) => {
                tracing::warn!(error = %e, position, "registry: log entry time unreadable");
                Err(Unmeasured::Unreadable)
            }
        }
    }

    fn release_log_below(&self, position: u64) {
        self.timeline.release_below(position);
    }

    fn retained_bytes_from(&self, kind: ConsumerKind, position: u64) -> Result<u64, Unmeasured> {
        if kind.is_seqno_space() {
            return self.observe(kind, position).map(|o| o.retained_bytes);
        }
        bytes_needed_from(&self.log_dirs, position).map_err(|e| {
            tracing::warn!(error = %e, "registry: log size unreadable");
            Unmeasured::Unreadable
        })
    }

    fn observe(&self, kind: ConsumerKind, position: u64) -> Result<Observed, Unmeasured> {
        use coordinode_storage::engine::material::Observation;
        if !kind.is_seqno_space() {
            return Ok(Observed {
                produced_at_ms: self.produced_at_ms(kind, position)?,
                retained_bytes: self.retained_bytes_from(kind, position)?,
                generation: None,
            });
        }
        if !self.accounts(kind) {
            return Err(Unmeasured::NotAccounted);
        }
        // The commits from the checkpoint through the store's complete cut:
        // the first is the oldest work the reader has not seen (its commit
        // timestamp is wall-clock microseconds), and their logical bytes
        // before compression are what it keeps from collection. A commit
        // above the cut is not counted yet: it is younger than the cut, and
        // counting it in part could age or size a write never wholly applied.
        match self.engine.observe_material(position) {
            Ok(Observation::Observed {
                material,
                generation,
                ..
            }) => Ok(Observed {
                produced_at_ms: material.first.map(|us| us / 1_000),
                retained_bytes: material.bytes,
                generation: Some(generation),
            }),
            // Nothing known complete yet, or a tree mid-rebuild: the evidence
            // is missing, which ends nothing and is no zero.
            Ok(Observation::NoCut | Observation::Rebuilding | Observation::Lost) => {
                Err(Unmeasured::Unreadable)
            }
            Err(e) => {
                tracing::warn!(error = %e, position, "registry: material records unreadable");
                Err(Unmeasured::Unreadable)
            }
        }
    }

    fn still_current(&self, observed: &Observed) -> bool {
        observed
            .generation
            .is_none_or(|generation| self.engine.material_generation() == generation)
    }

    fn admits(&self, _kind: ConsumerKind) -> Result<(), RetentionPressure> {
        // The log and the MVCC store share the engine's endpoints, so one
        // verdict covers both. A consumer's history is kept through
        // reclamation, so new retention is taken on only while compaction
        // keeps up, the volume is above its free-space reserve, and every
        // capped endpoint has room for client writes: what is left past that
        // is the reserve recovery work needs.
        if matches!(self.engine.write_pressure(), WritePressure::Stop) {
            return Err(RetentionPressure::WritePressure);
        }
        match self.engine.space().admit() {
            Ok(()) => Ok(()),
            Err(StorageError::OutOfSpace { .. }) => Err(RetentionPressure::FreeSpace),
            Err(StorageError::CapacityExhausted { endpoint_id, .. }) => {
                Err(RetentionPressure::Footprint { endpoint_id })
            }
            Err(other) => Err(RetentionPressure::Unknown(other.to_string())),
        }
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
