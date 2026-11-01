//! The Raft log's writer thread. An append hands its entries to this thread
//! and returns: the Raft core does no file I/O. The thread writes every
//! queued append to the segments, hands them to the kernel, makes them
//! durable and only then tells openraft and the writers waiting on them.
//!
//! The core must not write the segment itself: a write waits while a full
//! sync of the same file runs (macOS holds the file through `F_FULLFSYNC`),
//! so a core that writes is held for the length of every sync and can hand
//! over one batch per sync. Here the core keeps appending while the thread
//! syncs, and one sync covers every append queued meanwhile.
//!
//! Entries waiting for the thread, and the latest written ones, stay in a
//! [`Tail`] the log reads from: replication reads what the core just
//! appended without waiting for it to reach the file.

use std::collections::VecDeque;
use std::io;
use std::sync::Arc;
use std::thread::JoinHandle;

use coordinode_core::txn::proposal::ProposalId;
use coordinode_storage::oplog::{OplogManager, SyncHandle};
use openraft::storage::IOFlushed;
use parking_lot::{Condvar, Mutex};

use crate::storage::{AppendNotifier, Entry, LogStore, TypeConfig};

/// Written entries the tail keeps for readers beside the unwritten ones:
/// enough for replication to a follower a little behind to read from memory.
pub(crate) const TAIL_WRITTEN_ENTRIES: usize = 4096;

/// One append waiting to be written and made durable.
pub(crate) struct Append {
    /// Told once the entries are durable.
    pub(crate) callback: IOFlushed<TypeConfig>,
    /// `(proposal id, log index)` of every proposal the append carries.
    pub(crate) proposals: Vec<(ProposalId, u64)>,
    /// The entries, in log order, shared with the [`Tail`].
    pub(crate) entries: Vec<Arc<Entry>>,
}

/// The latest entries of the log, contiguous: every entry not yet written to
/// the segments, and up to [`TAIL_WRITTEN_ENTRIES`] written ones before them.
/// An entry below [`Tail::first`] is in the segments.
#[derive(Default)]
pub(crate) struct Tail {
    entries: VecDeque<Arc<Entry>>,
}

impl Tail {
    /// The index of the first entry held, if any.
    pub(crate) fn first(&self) -> Option<u64> {
        self.entries.front().map(|e| e.log_id.index)
    }

    /// Add `entries`, which follow the held ones. An entry at an index
    /// already held replaces it and everything after it, as a log
    /// overwritten from there.
    pub(crate) fn push(&mut self, entries: &[Arc<Entry>]) {
        let Some(first) = entries.first() else {
            return;
        };
        let from = first.log_id.index;
        if let Some(held) = self.first() {
            if from < held || from > held + self.entries.len() as u64 {
                // Not contiguous with what is held: start over from here.
                self.entries.clear();
            } else {
                // In range: `from - held` is at most the length.
                self.entries.truncate((from - held) as usize);
            }
        }
        self.entries.extend(entries.iter().cloned());
    }

    /// Drop the entries at or above `index`.
    pub(crate) fn truncate_from(&mut self, index: u64) {
        if let Some(held) = self.first() {
            let keep = index.saturating_sub(held);
            self.entries
                .truncate(usize::try_from(keep).unwrap_or(usize::MAX));
        }
    }

    /// The held entries in `[start, end)`.
    pub(crate) fn range(&self, start: u64, end: u64) -> Vec<Arc<Entry>> {
        let Some(held) = self.first() else {
            return Vec::new();
        };
        let skip = usize::try_from(start.saturating_sub(held)).unwrap_or(usize::MAX);
        let take = usize::try_from(end.saturating_sub(start.max(held))).unwrap_or(usize::MAX);
        self.entries.iter().skip(skip).take(take).cloned().collect()
    }

    /// Forget written entries beyond the [`TAIL_WRITTEN_ENTRIES`] kept: those
    /// at or below `written` may go, oldest first.
    fn written(&mut self, written: u64) {
        let unwritten = self
            .entries
            .iter()
            .rev()
            .take_while(|e| e.log_id.index > written)
            .count();
        while self.entries.len() > unwritten + TAIL_WRITTEN_ENTRIES {
            self.entries.pop_front();
        }
    }
}

/// Records a consensus write (the committed index) where the store keeps it.
pub(crate) type RecordWrite =
    Box<dyn Fn(coordinode_storage::engine::core::LocalWrite) -> Result<(), String> + Send + Sync>;

#[derive(Default)]
struct Queue {
    appends: Vec<Append>,
    /// The latest consensus record waiting to be written: each replaces the
    /// one before it, so a burst of commits writes one record.
    record: Option<coordinode_storage::engine::core::LocalWrite>,
    /// The thread is writing or syncing a taken group.
    syncing: bool,
    closed: bool,
    /// The last index written to the segments.
    written: Option<u64>,
    /// The first write or sync that failed. A later sync can succeed without
    /// the bytes the failed one covered (the kernel may drop dirty pages
    /// after reporting the failure), so from here every append is answered
    /// with this failure, never as durable.
    failed: Option<String>,
}

struct Shared {
    queue: Mutex<Queue>,
    /// Signalled when an append is queued or the log closes.
    queued: Condvar,
    /// Signalled when a group was written, and when the queue has drained
    /// and no sync is running.
    progress: Condvar,
    notifier: Arc<AppendNotifier>,
    oplog: Arc<std::sync::Mutex<OplogManager>>,
    tail: Arc<Mutex<Tail>>,
    /// The last index in the segments, for async waiters; `u64::MAX` once
    /// the log failed or closed, so no waiter is left behind.
    written_tx: tokio::sync::watch::Sender<Option<u64>>,
    /// Writes a queued consensus record.
    record_write: RecordWrite,
}

/// Owns the writer thread of one Raft log; dropping it writes, syncs and
/// answers what is still queued, then stops the thread.
pub(crate) struct LogSync {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl LogSync {
    /// Start the writer thread of the log in `oplog`, whose segments end at
    /// `written`, keeping `tail`, telling `notifier` of every proposal
    /// written and made durable, and writing consensus records through
    /// `record_write`.
    ///
    /// # Errors
    ///
    /// The thread cannot be spawned.
    pub(crate) fn start(
        notifier: Arc<AppendNotifier>,
        oplog: Arc<std::sync::Mutex<OplogManager>>,
        tail: Arc<Mutex<Tail>>,
        written: Option<u64>,
        record_write: RecordWrite,
    ) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                written,
                ..Queue::default()
            }),
            queued: Condvar::new(),
            progress: Condvar::new(),
            notifier,
            oplog,
            tail,
            written_tx: tokio::sync::watch::Sender::new(written),
            record_write,
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

    /// Queue `append` for writing; its entries are already in the tail.
    pub(crate) fn submit(&self, append: Append) {
        let mut queue = self.shared.queue.lock();
        queue.appends.push(append);
        self.shared.queued.notify_one();
    }

    /// Queue a consensus record whose write the next sync covers, in place
    /// of one still waiting. The caller never waits on the file it lands in.
    pub(crate) fn save_record(&self, record: coordinode_storage::engine::core::LocalWrite) {
        let mut queue = self.shared.queue.lock();
        queue.record = Some(record);
        self.shared.queued.notify_one();
    }

    /// Block until every queued append is written, durable and answered, and
    /// a queued consensus record written. A change that rewrites the log
    /// waits here first, so no append is answered after entries it covered
    /// were replaced.
    pub(crate) fn wait_idle(&self) {
        let mut queue = self.shared.queue.lock();
        while !queue.appends.is_empty() || queue.record.is_some() || queue.syncing {
            self.shared.progress.wait(&mut queue);
        }
    }

    /// The log was cut to end below `keep_exclusive`, with nothing queued: the
    /// written mark comes down with it, so an entry appended again at a cut
    /// index is not taken for written.
    pub(crate) fn truncated(&self, keep_exclusive: u64) {
        let mut queue = self.shared.queue.lock();
        let below = keep_exclusive.checked_sub(1);
        if queue.written > below {
            queue.written = below;
            self.shared.written_tx.send_if_modified(|w| {
                // A failed or closed log keeps releasing every waiter.
                if w.is_some_and(|w| w != u64::MAX) {
                    *w = below;
                    true
                } else {
                    false
                }
            });
        }
    }

    /// The last index written to the segments, as it advances; `u64::MAX`
    /// once the log failed or closed.
    pub(crate) fn written(&self) -> tokio::sync::watch::Receiver<Option<u64>> {
        self.shared.written_tx.subscribe()
    }

    /// Block until the entry at `index` is in the segments, or the log
    /// failed or closed: a reader of the segments sees every entry up to it.
    pub(crate) fn wait_written(&self, index: u64) {
        let mut queue = self.shared.queue.lock();
        while queue.written.is_none_or(|w| w < index) && queue.failed.is_none() && !queue.closed {
            if queue.appends.is_empty() && !queue.syncing {
                // Nothing queued reaches it: the entry is not in this log.
                return;
            }
            self.shared.progress.wait(&mut queue);
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
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            tracing::error!("the raft log sync thread panicked");
        }
    }
}

impl Shared {
    fn run(&self) {
        loop {
            let (appends, record, failed) = {
                let mut queue = self.queue.lock();
                while queue.appends.is_empty() && queue.record.is_none() && !queue.closed {
                    self.queued.wait(&mut queue);
                }
                if queue.appends.is_empty() && queue.record.is_none() {
                    // Closed, and nothing is left to answer.
                    self.written_tx.send_replace(Some(u64::MAX));
                    return;
                }
                queue.syncing = true;
                (
                    std::mem::take(&mut queue.appends),
                    queue.record.take(),
                    queue.failed.clone(),
                )
            };
            // The consensus record goes to its own journal, off the Raft
            // core: a write there waits out that journal's syncs too.
            if let Some(record) = record
                && failed.is_none()
                && let Err(e) = (self.record_write)(record)
            {
                tracing::error!(error = %e, "raft consensus record write failed");
                let mut queue = self.queue.lock();
                queue.failed.get_or_insert(e);
                self.written_tx.send_replace(Some(u64::MAX));
            }
            if appends.is_empty() {
                let mut queue = self.queue.lock();
                queue.syncing = false;
                self.progress.notify_all();
                continue;
            }
            let failed = self.queue.lock().failed.clone();
            // What one sync covers and how long it took: the group commit's
            // batching and the medium's flush cost, the two halves of an
            // acknowledged write's latency here.
            metrics::histogram!("coordinode_raft_log_sync_appends").record(appends.len() as f64);
            metrics::histogram!("coordinode_raft_log_sync_proposals")
                .record(appends.iter().map(|a| a.proposals.len()).sum::<usize>() as f64);
            let written = match failed {
                Some(failed) => Err(failed),
                None => self.write(&appends),
            };
            if let Ok((_, last)) = &written {
                // Written to the kernel: a reader of the segments sees them,
                // and a writer under a volatile `j` is answered now.
                for append in &appends {
                    self.notifier.notify_written(&append.proposals);
                }
                if let Some(last) = *last {
                    self.tail.lock().written(last);
                    self.queue.lock().written = Some(last);
                    self.progress.notify_all();
                    self.written_tx.send_replace(Some(last));
                }
            }
            let started = std::time::Instant::now();
            let synced = written.and_then(|(handle, _)| match handle {
                Some(handle) => handle.sync().map_err(|e| e.to_string()),
                None => Ok(()),
            });
            metrics::histogram!("coordinode_raft_log_sync_seconds")
                .record(started.elapsed().as_secs_f64());
            self.answer(appends, &synced);
            let mut queue = self.queue.lock();
            if let Err(e) = &synced
                && queue.failed.is_none()
            {
                tracing::error!(
                    error = %e,
                    "raft log write or sync failed; no append is acknowledged from here until the log is reopened"
                );
                queue.failed = Some(e.clone());
                // Nothing more reaches the segments: release every waiter.
                self.written_tx.send_replace(Some(u64::MAX));
            }
            queue.syncing = false;
            self.progress.notify_all();
        }
    }

    /// Write the entries of `appends` to the segments and hand them to the
    /// kernel. Returns what syncs them and the last index written. A segment
    /// rotated out meanwhile was synced when sealed.
    fn write(&self, appends: &[Append]) -> Result<(Option<SyncHandle>, Option<u64>), String> {
        let mut oplog = self
            .oplog
            .lock()
            .map_err(|_| "raft log mutex poisoned".to_owned())?;
        let mut last = None;
        for entry in appends.iter().flat_map(|a| a.entries.iter()) {
            let record = LogStore::entry_to_oplog(entry).map_err(|e| e.to_string())?;
            oplog.append(&record).map_err(|e| e.to_string())?;
            last = Some(entry.log_id.index);
        }
        let handle = oplog.flush_to_os().map_err(|e| e.to_string())?;
        Ok((handle, last))
    }

    /// Tell openraft and the writers how the sync of `appends` went. A failed
    /// write or sync answers every append of the group with the error.
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
