//! How far an index maintained from the applied commits covers the store, and
//! the wait a search makes for it.
//!
//! The full-text and vector indexes are maintained from the entries applied
//! to the store, each kind by a worker that follows them. A search must not
//! answer from an index that lacks a commit its snapshot includes, so before
//! reading it waits until the worker has folded every entry applied before
//! the search began, within a bound, and fails explicitly past it.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use coordinode_storage::engine::applied::AppliedPosition;

/// Default bound on a search's wait for its indexes to cover the store.
pub const DEFAULT_INDEX_READY_WAIT: Duration = Duration::from_millis(2000);

/// The kind of index a worker maintains from the applied commits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaintainedIndex {
    /// Full-text (Tantivy) indexes.
    Text,
    /// Vector (HNSW) indexes.
    Vector,
}

impl MaintainedIndex {
    /// The kind as an error metadata value names it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "full-text",
            Self::Vector => "vector",
        }
    }
}

impl fmt::Display for MaintainedIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A search found its indexes behind the store past its wait.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "the {kind} indexes have not caught up with the store within {waited_ms} ms \
     (folded through event {folded}, the store holds event {needed})"
)]
pub struct IndexBehind {
    /// Which indexes.
    pub kind: MaintainedIndex,
    /// The last applied event the indexes hold.
    pub folded: u64,
    /// The applied event the search needed them to hold.
    pub needed: u64,
    /// How long the search waited.
    pub waited_ms: u64,
}

/// The coverage of one kind of index: what the store has applied, what its
/// worker has folded, and the wait between the two.
#[derive(Debug)]
pub struct IndexReadiness {
    kind: MaintainedIndex,
    applied: AppliedPosition,
    /// The number of the last applied event folded into every index of the
    /// kind.
    folded: AtomicU64,
    /// Woken whenever `folded` advances.
    // no-std: spin-based wait or a caller-provided parking primitive.
    changed: parking_lot::Condvar,
    lock: parking_lot::Mutex<()>,
    wait: AtomicU64,
}

impl IndexReadiness {
    /// Coverage of the `kind` worker following `applied`, with searches
    /// waiting at most `wait`.
    pub fn new(kind: MaintainedIndex, applied: AppliedPosition, wait: Duration) -> Self {
        Self {
            kind,
            applied,
            folded: AtomicU64::new(0),
            changed: parking_lot::Condvar::new(),
            lock: parking_lot::Mutex::new(()),
            wait: AtomicU64::new(duration_ms(wait)),
        }
    }

    /// Record that every applied event up to `seq` is in the indexes.
    pub fn advance(&self, seq: u64) {
        let _guard = self.lock.lock();
        if self.folded.fetch_max(seq, Ordering::AcqRel) < seq {
            self.changed.notify_all();
        }
    }

    /// The number of the last applied event the indexes hold.
    pub fn folded(&self) -> u64 {
        self.folded.load(Ordering::Acquire)
    }

    /// The number of the last event applied to the store.
    pub fn applied(&self) -> u64 {
        self.applied.delivered()
    }

    /// Change how long a search waits for coverage.
    pub fn set_wait(&self, wait: Duration) {
        self.wait.store(duration_ms(wait), Ordering::Relaxed);
    }

    /// Wait until the indexes hold every entry applied before this call, or
    /// fail once the configured wait has passed.
    ///
    /// # Errors
    ///
    /// [`IndexBehind`] when the worker has not folded them in time.
    pub fn await_covered(&self) -> Result<(), IndexBehind> {
        let needed = self.applied.delivered();
        if self.folded() >= needed {
            return Ok(());
        }
        let wait = Duration::from_millis(self.wait.load(Ordering::Relaxed));
        let started = Instant::now();
        let deadline = started + wait;
        let mut guard = self.lock.lock();
        while self.folded() < needed {
            if self.changed.wait_until(&mut guard, deadline).timed_out() && self.folded() < needed {
                return Err(IndexBehind {
                    kind: self.kind,
                    folded: self.folded(),
                    needed,
                    waited_ms: duration_ms(started.elapsed()),
                });
            }
        }
        Ok(())
    }
}

/// `d` in whole milliseconds; a wait beyond u64 milliseconds outlasts any
/// process.
fn duration_ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}
