//! Background reclamation of the journal a store's tables cover.
//!
//! Under a hard limit a journal segment the trees already hold is a second
//! copy of the same data. Left until a write is refused for room, the copy
//! may have taken the room the reclamation itself needs. Each landed flush
//! wakes the reclaimer instead; it reclaims once a segment sealed since its
//! last round, and after a failed round only once room came back.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use coordinode_core::txn::wake::Wake;

use crate::engine::core::StorageEngine;
use crate::error::{StorageError, StorageResult};

/// What the last round left: the journal's first index and sealed segment
/// count, the bytes charged on the capped endpoints, and whether it failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Round {
    pub(crate) segments: (u64, usize),
    pub(crate) charged: u64,
    pub(crate) failed: bool,
}

impl Round {
    /// Whether another round may remove something this one could not: the
    /// journal moved (a segment sealed or went), or this one failed and
    /// room came back since.
    pub(crate) fn due(&self, segments: (u64, usize), charged: u64) -> bool {
        segments != self.segments || (self.failed && charged < self.charged)
    }
}

/// The running reclaimer; dropping it stops the thread.
pub struct JournalReclaimer {
    stop: Arc<AtomicBool>,
    wake: Arc<Wake>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for JournalReclaimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JournalReclaimer").finish_non_exhaustive()
    }
}

impl JournalReclaimer {
    /// Reclaim for `engine` each time `landed` is notified, until dropped.
    /// The engine is held weakly: the reclaimer does not keep it open.
    pub(crate) fn start(engine: Weak<StorageEngine>, landed: Arc<Wake>) -> StorageResult<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let wake = Arc::clone(&landed);
        let thread = std::thread::Builder::new()
            .name("coord-journal-reclaim".into())
            .spawn(move || {
                landed.bind();
                let mut last = None;
                loop {
                    landed.wait(None);
                    if stopped.load(Ordering::Acquire) {
                        return;
                    }
                    let Some(engine) = engine.upgrade() else {
                        return;
                    };
                    engine.journal_reclaim_round(&mut last);
                }
            })
            .map_err(|e| StorageError::InvalidConfig(format!("journal reclaimer spawn: {e}")))?;
        Ok(Self {
            stop,
            wake,
            thread: Some(thread),
        })
    }
}

impl Drop for JournalReclaimer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.wake.interrupt();
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            tracing::error!("the journal reclaimer panicked");
        }
    }
}

#[cfg(test)]
mod tests;
