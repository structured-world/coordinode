//! The Raft log's sync thread. An append returns once its entries are written
//! to the kernel; this thread makes them durable and only then tells openraft
//! and the writers waiting on them. One sync covers every append queued while
//! the previous one ran, so a slow sync (a full flush reaches the medium, not
//! the drive's cache) is paid once per group of appends, and the Raft core
//! keeps replicating while it runs.

use std::io;
use std::sync::Arc;
use std::thread::JoinHandle;

use coordinode_core::txn::proposal::ProposalId;
use coordinode_storage::oplog::SyncHandle;
use openraft::storage::IOFlushed;
use parking_lot::{Condvar, Mutex};

use crate::storage::{AppendNotifier, TypeConfig};

/// One append waiting for its entries to become durable.
pub(crate) struct Append {
    /// Told once the entries are durable.
    pub(crate) callback: IOFlushed<TypeConfig>,
    /// `(proposal id, log index)` of every proposal the append carries.
    pub(crate) proposals: Vec<(ProposalId, u64)>,
}

#[derive(Default)]
struct Queue {
    appends: Vec<Append>,
    /// Syncs the segment the latest append went to. An append that rotated
    /// to a new segment left the old one durable: sealing syncs it.
    handle: Option<SyncHandle>,
    syncing: bool,
    closed: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    /// Signalled when an append is queued or the log closes.
    queued: Condvar,
    /// Signalled when the queue has drained and no sync is running.
    idle: Condvar,
    notifier: Arc<AppendNotifier>,
}

/// Owns the sync thread of one Raft log; dropping it syncs and answers what
/// is still queued, then stops the thread.
pub(crate) struct LogSync {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl LogSync {
    /// Start the sync thread of a log, telling `notifier` of every proposal
    /// made durable.
    ///
    /// # Errors
    ///
    /// The thread cannot be spawned.
    pub(crate) fn start(notifier: Arc<AppendNotifier>) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            queued: Condvar::new(),
            idle: Condvar::new(),
            notifier,
        });
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("raft-log-sync".into())
            .spawn(move || worker.run())?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    /// Queue `append`, whose entries were all written before `handle` was
    /// taken from the segment they went to.
    pub(crate) fn submit(&self, handle: Option<SyncHandle>, append: Append) {
        let mut queue = self.shared.queue.lock();
        if handle.is_some() {
            queue.handle = handle;
        }
        queue.appends.push(append);
        self.shared.queued.notify_one();
    }

    /// Block until every queued append is durable and answered. A change
    /// that rewrites the log waits here first, so no append is answered
    /// after entries it covered were replaced.
    pub(crate) fn wait_idle(&self) {
        let mut queue = self.shared.queue.lock();
        while !queue.appends.is_empty() || queue.syncing {
            self.shared.idle.wait(&mut queue);
        }
    }
}

impl Drop for LogSync {
    fn drop(&mut self) {
        {
            let mut queue = self.shared.queue.lock();
            queue.closed = true;
            self.shared.queued.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::error!("the raft log sync thread panicked");
            }
        }
    }
}

impl Shared {
    fn run(&self) {
        loop {
            let (appends, handle) = {
                let mut queue = self.queue.lock();
                while queue.appends.is_empty() && !queue.closed {
                    self.queued.wait(&mut queue);
                }
                if queue.appends.is_empty() {
                    // Closed, and nothing is left to answer.
                    return;
                }
                queue.syncing = true;
                (std::mem::take(&mut queue.appends), queue.handle.clone())
            };
            let synced = match &handle {
                Some(handle) => handle.sync().map_err(|e| e.to_string()),
                None => Ok(()),
            };
            self.answer(appends, &synced);
            let mut queue = self.queue.lock();
            queue.syncing = false;
            if queue.appends.is_empty() {
                self.idle.notify_all();
            }
        }
    }

    /// Tell openraft and the writers how the sync of `appends` went. A failed
    /// sync answers every append of the group with the error.
    fn answer(&self, appends: Vec<Append>, synced: &Result<(), String>) {
        for append in appends {
            match synced {
                Ok(()) => {
                    for (id, index) in append.proposals {
                        self.notifier.notify(id, index);
                    }
                    append.callback.io_completed(Ok(()));
                }
                Err(e) => append
                    .callback
                    .io_completed(Err(io::Error::other(format!("sync the raft log: {e}")))),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
