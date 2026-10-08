//! FlushManager: background memtable → SST flush worker pool.
//!
//! Monitors all partition trees and flushes sealed memtables to SST when either:
//!   - active memtable size exceeds `flush_threshold_bytes`
//!   - sealed memtable count exceeds `max_sealed`
//!
//! Architecture: one monitor thread checks the trees when something may have
//! made one due, and submits [`FlushRequest`]s via a flume channel to N worker
//! threads. Workers call `get_flush_lock()` + `flush()` on the received tree
//! clone.
//!
//! Nothing here runs on a timer while the database is idle. The write path
//! tells the monitor when a memtable crossed its size threshold or received
//! its first entry ([`FlushTrigger`]); a finished flush tells it the sealed
//! backlog moved; the only deadline the monitor keeps is the age of a
//! memtable that holds data.
//!
//! Shutdown is automatic via [`Drop`]: the monitor exits when woken with the
//! shutdown flag set, then workers exit when all senders are dropped.

use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use lsm_tree::AbstractTree;

use crate::engine::coordinator::GcWatermarks;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};
use coordinode_core::txn::wake::Wake;

/// How the write path tells the flush monitor that a memtable may be due.
///
/// Held by every writer of a partition tree. A write that leaves its memtable
/// above the size threshold, or that is the memtable's first entry (which
/// starts its age), wakes the monitor; any other write does nothing beyond one
/// comparison.
#[derive(Debug)]
pub(crate) struct FlushTrigger {
    wake: Arc<Wake>,
    threshold_bytes: u64,
}

impl FlushTrigger {
    /// A trigger waking `wake` for memtables above `threshold_bytes`.
    pub(crate) fn new(wake: Arc<Wake>, threshold_bytes: u64) -> Self {
        Self {
            wake,
            threshold_bytes,
        }
    }

    /// A write of `added` bytes left its memtable at `memtable_bytes`.
    #[inline]
    pub(crate) fn wrote(&self, added: u64, memtable_bytes: u64) {
        if memtable_bytes > self.threshold_bytes || memtable_bytes == added {
            self.wake.notify();
        }
    }

    /// A write whose effect on the memtable's size is not reported (a range
    /// tombstone): wake the monitor rather than guess.
    pub(crate) fn wrote_unmeasured(&self) {
        self.wake.notify();
    }
}

/// Request to flush sealed memtables for a single partition tree.
struct FlushRequest {
    /// Clone of the partition tree handle (cheap: all fields are Arc internally).
    tree: lsm_tree::AnyTree,
    /// Partition tag — used for trace logging only.
    partition: Partition,
    /// GC watermark seqno: versions with seqno ≤ this value may be evicted.
    gc_watermark: u64,
    /// Set by the monitor when it queued this request, cleared by the worker
    /// once the flush returned, so the monitor asks again for a partition
    /// whose sealed memtables are still there and nobody is flushing.
    queued: Arc<AtomicBool>,
}

/// Background memtable → SST flush worker pool.
///
/// Started by `StorageEngine::finish_open` and dropped automatically when the
/// engine is dropped (field ordering ensures FlushManager drops before trees).
pub(crate) struct FlushManager {
    /// Worker thread handles.
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Monitor thread handle (None after drop).
    monitor: Option<std::thread::JoinHandle<()>>,
    /// Shutdown flag: set to `true` to stop all threads.
    shutdown: Arc<AtomicBool>,
    /// The monitor's wakeup, notified on shutdown so the monitor sees the flag.
    wake: Arc<Wake>,
    /// Sender clone held here so we can drop it explicitly before joining workers.
    sender: Option<flume::Sender<FlushRequest>>,
}

impl FlushManager {
    /// Start the flush manager.
    ///
    /// Spawns one monitor thread, woken through `wake` (the wake the write
    /// path's [`FlushTrigger`] notifies), and `num_workers` worker threads,
    /// which notify `compaction_wake` after every flush: a flush adds a table
    /// compaction may have to act on.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any background thread fails to spawn (OS resource limit).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start(
        trees: &HashMap<Partition, lsm_tree::AnyTree>,
        gc_watermarks: GcWatermarks,
        flush_threshold_bytes: u64,
        max_sealed: usize,
        num_workers: usize,
        max_memtable_age_secs: u64,
        wake: Arc<Wake>,
        compaction_wake: Arc<Wake>,
    ) -> StorageResult<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));

        // Bounded channel: capacity = workers × 4 so burst flushes don't stall the monitor.
        let capacity = (num_workers * 4).max(8);
        let (sender, receiver) = flume::bounded::<FlushRequest>(capacity);

        // Spawn N worker threads — each holds a receiver clone (flume supports multi-consumer).
        let mut workers = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let rx = receiver.clone();
            let flushed = Arc::clone(&wake);
            let compaction = Arc::clone(&compaction_wake);
            let handle = std::thread::Builder::new()
                .name(format!("coord-flush-worker-{i}"))
                .spawn(move || flush_worker_loop(rx, &flushed, &compaction))
                .map_err(|e| StorageError::InvalidConfig(format!("flush worker spawn: {e}")))?;
            workers.push(handle);
        }

        // Clone tree handles for the monitor (cheap: AnyTree is Clone via Arc).
        let monitored: Vec<(Partition, lsm_tree::AnyTree)> =
            trees.iter().map(|(&p, t)| (p, t.clone())).collect();

        // Spawn monitor thread.
        let tx = sender.clone();
        let shutdown_m = Arc::clone(&shutdown);
        let wake_m = Arc::clone(&wake);
        let monitor = std::thread::Builder::new()
            .name("coord-flush-monitor".to_string())
            .spawn(move || {
                flush_monitor_loop(
                    monitored,
                    FlushMonitorConfig {
                        sender: tx,
                        gc_watermarks,
                        flush_threshold_bytes,
                        max_sealed,
                        max_memtable_age_secs,
                        shutdown: shutdown_m,
                        wake: wake_m,
                    },
                );
            })
            .map_err(|e| StorageError::InvalidConfig(format!("flush monitor spawn: {e}")))?;

        Ok(Self {
            workers,
            monitor: Some(monitor),
            shutdown,
            wake,
            sender: Some(sender),
        })
    }
}

impl Drop for FlushManager {
    fn drop(&mut self) {
        // Signal all threads to stop, and wake the monitor so it sees it.
        self.shutdown.store(true, Ordering::Relaxed);
        self.wake.notify();

        // The monitor's sender clone is dropped when the monitor exits.
        if let Some(monitor) = self.monitor.take() {
            let _ = monitor.join();
        }

        // Drop our sender clone. Once ALL senders (monitor's + ours) are dropped,
        // the channel closes and workers receive `Disconnected` → exit their loops.
        drop(self.sender.take());

        // Join workers.
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Parameters bundled to keep the monitor signature within clippy's
/// `too_many_arguments` budget. Pure data — borrowed only by the spawned thread.
struct FlushMonitorConfig {
    sender: flume::Sender<FlushRequest>,
    gc_watermarks: GcWatermarks,
    flush_threshold_bytes: u64,
    max_sealed: usize,
    max_memtable_age_secs: u64,
    shutdown: Arc<AtomicBool>,
    wake: Arc<Wake>,
}

/// Monitor loop: checks all partition trees when woken and submits flush
/// requests where needed, then sleeps until the next event or the earliest
/// age deadline of a memtable holding data.
///
/// Three independent triggers can rotate a partition's active memtable:
///
/// 1. **Size threshold:** `active_memtable.size() > flush_threshold_bytes`.
///    Caps memory use under sustained write load.
/// 2. **Sealed backlog:** `sealed_memtable_count() > max_sealed`. Prevents
///    a slow flush worker from accumulating an unbounded queue.
/// 3. **Memtable age:** any non-empty memtable older than
///    `max_memtable_age_secs` (default 30s; `0` disables the trigger).
///    Without this, light or bursty workloads can leave mutations in the
///    memtable for hours; combined with the oplog purge gate that would
///    grow the oplog unbounded waiting for size-based flush to fire.
///    The clock starts at startup and resets on every rotation; an empty
///    active memtable is never rotated (no data to lose).
fn flush_monitor_loop(trees: Vec<(Partition, lsm_tree::AnyTree)>, cfg: FlushMonitorConfig) {
    cfg.wake.bind();
    let max_age = Duration::from_secs(cfg.max_memtable_age_secs);
    let start = Instant::now();
    let mut last_rotate: HashMap<Partition, Instant> =
        trees.iter().map(|(p, _)| (*p, start)).collect();
    let queued: Vec<Arc<AtomicBool>> = trees
        .iter()
        .map(|_| Arc::new(AtomicBool::new(false)))
        .collect();

    while !cfg.shutdown.load(Ordering::Relaxed) {
        let now = Instant::now();
        // The soonest a memtable that holds data and stays below the other
        // triggers becomes due by age. None while every memtable is empty: an
        // idle engine sleeps until a write wakes it.
        let mut next_due: Option<Duration> = None;

        for ((partition, tree), queued) in trees.iter().zip(&queued) {
            let active_bytes = tree.active_memtable().size();
            let age = now.saturating_duration_since(*last_rotate.get(partition).unwrap_or(&start));
            let age_triggered = cfg.max_memtable_age_secs > 0 && active_bytes > 0 && age >= max_age;

            let rotate = active_bytes > cfg.flush_threshold_bytes
                || tree.sealed_memtable_count() > cfg.max_sealed
                || age_triggered;

            if rotate {
                // Rotate: atomically seal the active memtable → it joins the sealed list.
                // A fresh empty memtable becomes the new active.
                tree.rotate_memtable();
                last_rotate.insert(*partition, now);
            } else if cfg.max_memtable_age_secs > 0 && active_bytes > 0 {
                let remaining = max_age.saturating_sub(age);
                next_due = Some(next_due.map_or(remaining, |due| due.min(remaining)));
            }

            // Every sealed memtable is flushed, not only one sealed just now:
            // a request turned away while the workers were busy leaves its
            // memtable sealed, and an idle partition never trips the gates
            // above again. One request per partition at a time, because a
            // flush takes every sealed memtable of its tree.
            if tree.sealed_memtable_count() > 0 && !queued.load(Ordering::Acquire) {
                let req = FlushRequest {
                    tree: tree.clone(),
                    partition: *partition,
                    gc_watermark: cfg.gc_watermarks.for_partition(*partition),
                    queued: Arc::clone(queued),
                };
                queued.store(true, Ordering::Release);
                // Non-blocking: with the workers busy the request waits for
                // the next pass, which a finished flush wakes.
                if cfg.sender.try_send(req).is_err() {
                    queued.store(false, Ordering::Release);
                }
            }
        }

        cfg.wake.wait(next_due);
    }
}

/// Worker loop: receives flush requests and flushes sealed memtables to SST.
///
/// Blocks on the channel with no timeout: the manager's drop closes the
/// channel, which is what ends the loop.
fn flush_worker_loop(receiver: flume::Receiver<FlushRequest>, flushed: &Wake, compaction: &Wake) {
    while let Ok(FlushRequest {
        tree,
        partition,
        gc_watermark,
        queued,
    }) = receiver.recv()
    {
        // get_flush_lock() is not Send: it is acquired and used in this thread.
        let flush_lock = tree.get_flush_lock();
        let result = tree.flush(&flush_lock, gc_watermark);
        drop(flush_lock);
        // Before the wake below, so the pass it starts may ask again.
        queued.store(false, Ordering::Release);
        match result {
            Ok(Some(bytes)) => {
                tracing::debug!(
                    partition = partition.name(),
                    flushed_bytes = bytes,
                    "memtable flushed to SST"
                );
                // A new table may need compacting.
                compaction.notify();
            }
            Ok(None) => {
                // Nothing to flush: another worker or the monitor already did it.
            }
            Err(e) => {
                tracing::error!(
                    partition = partition.name(),
                    error = %e,
                    "memtable flush failed"
                );
                // Retried on the monitor's next pass, which the next write or
                // flush starts; waking it now would retry a failing disk in a
                // loop.
                continue;
            }
        }
        // The sealed backlog moved; a request the monitor could not queue
        // while workers were busy can go now.
        flushed.notify();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
