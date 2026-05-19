//! FlushManager: background memtable → SST flush worker pool (R072).
//!
//! Monitors all partition trees and flushes sealed memtables to SST when either:
//!   - active memtable size exceeds `flush_threshold_bytes`
//!   - sealed memtable count exceeds `max_sealed`
//!
//! Architecture: one monitor thread polls all trees at a configurable interval
//! and submits [`FlushRequest`]s via a flume channel to N worker threads.
//! Workers call `get_flush_lock()` + `flush()` on the received tree clone.
//!
//! Shutdown is automatic via [`Drop`] — the monitor exits when the shutdown
//! flag is set, then workers exit when all senders are dropped.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use lsm_tree::AbstractTree;

use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// Request to flush sealed memtables for a single partition tree.
struct FlushRequest {
    /// Clone of the partition tree handle (cheap: all fields are Arc internally).
    tree: lsm_tree::AnyTree,
    /// Partition tag — used for trace logging only.
    partition: Partition,
    /// GC watermark seqno: versions with seqno ≤ this value may be evicted.
    gc_watermark: u64,
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
    /// Sender clone held here so we can drop it explicitly before joining workers.
    sender: Option<flume::Sender<FlushRequest>>,
}

impl FlushManager {
    /// Start the flush manager.
    ///
    /// Spawns one monitor thread and `num_workers` worker threads.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any background thread fails to spawn (OS resource limit).
    pub(crate) fn start(
        trees: &HashMap<Partition, lsm_tree::AnyTree>,
        gc_watermark: Arc<AtomicU64>,
        flush_threshold_bytes: u64,
        max_sealed: usize,
        num_workers: usize,
        poll_interval_ms: u64,
        max_memtable_age_secs: u64,
    ) -> StorageResult<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));

        // Bounded channel: capacity = workers × 4 so burst flushes don't stall the monitor.
        let capacity = (num_workers * 4).max(8);
        let (sender, receiver) = flume::bounded::<FlushRequest>(capacity);

        // Spawn N worker threads — each holds a receiver clone (flume supports multi-consumer).
        let mut workers = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let rx = receiver.clone();
            let shutdown_w = Arc::clone(&shutdown);
            let handle = std::thread::Builder::new()
                .name(format!("coord-flush-worker-{i}"))
                .spawn(move || flush_worker_loop(rx, shutdown_w))
                .map_err(|e| StorageError::InvalidConfig(format!("flush worker spawn: {e}")))?;
            workers.push(handle);
        }

        // Clone tree handles for the monitor (cheap: AnyTree is Clone via Arc).
        let monitored: Vec<(Partition, lsm_tree::AnyTree)> =
            trees.iter().map(|(&p, t)| (p, t.clone())).collect();

        // Spawn monitor thread.
        let tx = sender.clone();
        let shutdown_m = Arc::clone(&shutdown);
        let monitor = std::thread::Builder::new()
            .name("coord-flush-monitor".to_string())
            .spawn(move || {
                flush_monitor_loop(
                    monitored,
                    FlushMonitorConfig {
                        sender: tx,
                        gc_watermark,
                        flush_threshold_bytes,
                        max_sealed,
                        poll_interval_ms,
                        max_memtable_age_secs,
                        shutdown: shutdown_m,
                    },
                );
            })
            .map_err(|e| StorageError::InvalidConfig(format!("flush monitor spawn: {e}")))?;

        Ok(Self {
            workers,
            monitor: Some(monitor),
            shutdown,
            sender: Some(sender),
        })
    }
}

impl Drop for FlushManager {
    fn drop(&mut self) {
        // Signal all threads to stop.
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for monitor — it sleeps at most `poll_interval_ms` before checking shutdown.
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
    gc_watermark: Arc<AtomicU64>,
    flush_threshold_bytes: u64,
    max_sealed: usize,
    poll_interval_ms: u64,
    max_memtable_age_secs: u64,
    shutdown: Arc<AtomicBool>,
}

/// Monitor loop: polls all partition trees and submits flush requests when needed.
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
///    memtable for hours; combined with R076a's purge gate that would
///    grow the oplog unbounded waiting for size-based flush to fire.
///    The clock starts at startup and resets on every rotation; an empty
///    active memtable is never rotated (no data to lose).
fn flush_monitor_loop(trees: Vec<(Partition, lsm_tree::AnyTree)>, cfg: FlushMonitorConfig) {
    let max_age = Duration::from_secs(cfg.max_memtable_age_secs);
    let start = Instant::now();
    let mut last_rotate: HashMap<Partition, Instant> =
        trees.iter().map(|(p, _)| (*p, start)).collect();

    while !cfg.shutdown.load(Ordering::Relaxed) {
        let watermark = cfg.gc_watermark.load(Ordering::Relaxed);
        let now = Instant::now();

        for (partition, tree) in &trees {
            let active_bytes = tree.active_memtable().size();
            let sealed_count = tree.sealed_memtable_count();
            let age = now.saturating_duration_since(*last_rotate.get(partition).unwrap_or(&start));
            let age_triggered = cfg.max_memtable_age_secs > 0 && active_bytes > 0 && age >= max_age;

            let needs_flush = active_bytes > cfg.flush_threshold_bytes
                || sealed_count > cfg.max_sealed
                || age_triggered;

            if needs_flush {
                // Rotate: atomically seal the active memtable → it joins the sealed list.
                // A fresh empty memtable becomes the new active.
                tree.rotate_memtable();
                last_rotate.insert(*partition, now);

                let req = FlushRequest {
                    tree: tree.clone(),
                    partition: *partition,
                    gc_watermark: watermark,
                };

                // Non-blocking: if channel is full, workers are busy.
                // The sealed memtable stays in the sealed list until the next flush call.
                let _ = cfg.sender.try_send(req);
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(cfg.poll_interval_ms));
    }
}

/// Worker loop: receives flush requests and flushes sealed memtables to SST.
fn flush_worker_loop(receiver: flume::Receiver<FlushRequest>, shutdown: Arc<AtomicBool>) {
    loop {
        // Use a timeout so the shutdown flag is checked periodically even when idle.
        match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(FlushRequest {
                tree,
                partition,
                gc_watermark,
            }) => {
                // get_flush_lock() is not Send — must be acquired and used in this thread.
                let flush_lock = tree.get_flush_lock();
                match tree.flush(&flush_lock, gc_watermark) {
                    Ok(Some(bytes)) => {
                        tracing::debug!(
                            partition = partition.name(),
                            flushed_bytes = bytes,
                            "memtable flushed to SST"
                        );
                    }
                    Ok(None) => {
                        // Nothing to flush — another worker or the monitor already did it.
                    }
                    Err(e) => {
                        tracing::error!(
                            partition = partition.name(),
                            error = %e,
                            "memtable flush failed"
                        );
                    }
                }
            }
            Err(flume::RecvTimeoutError::Timeout) => {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
            }
            Err(flume::RecvTimeoutError::Disconnected) => {
                // All senders dropped — channel closed, exit gracefully.
                break;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    /// Build a minimal in-memory partition tree map for testing.
    fn make_test_trees() -> (HashMap<Partition, lsm_tree::AnyTree>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(lsm_tree::SequenceNumberCounter::default());
        let mut trees = HashMap::new();
        let tree = lsm_tree::Config::new_with_generators(
            dir.path().join("node"),
            Arc::clone(&seqno),
            Arc::clone(&seqno),
        )
        .open()
        .expect("open tree");
        trees.insert(Partition::Node, tree);
        (trees, dir)
    }

    #[test]
    fn flush_manager_starts_and_stops() {
        let (trees, _dir) = make_test_trees();
        let gc_watermark = Arc::new(AtomicU64::new(0));

        let mgr = FlushManager::start(
            &trees,
            Arc::clone(&gc_watermark),
            64 * 1024 * 1024, // 64MB threshold (won't trigger in this test)
            4,
            1,  // 1 worker
            50, // poll interval
            0,  // age trigger disabled
        )
        .expect("start FlushManager");

        // Brief sleep to let threads spin up.
        std::thread::sleep(Duration::from_millis(120));

        // Drop manager: should join all threads cleanly without hanging.
        drop(mgr);
    }

    #[test]
    fn flush_manager_flushes_when_sealed_count_exceeded() {
        let (trees, _dir) = make_test_trees();
        let gc_watermark = Arc::new(AtomicU64::new(0));

        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(lsm_tree::SequenceNumberCounter::default());
        let tree = trees.get(&Partition::Node).expect("node tree").clone();

        // Write a small value so the memtable is non-empty before sealing.
        tree.insert(b"key1", b"value1", seqno.next());

        // Seal to produce 1 sealed memtable (below threshold of 4).
        tree.rotate_memtable();
        assert_eq!(tree.sealed_memtable_count(), 1, "one sealed before start");

        // FlushManager with max_sealed=0 so ANY sealed count triggers flush.
        let mgr = FlushManager::start(
            &trees,
            Arc::clone(&gc_watermark),
            u64::MAX, // size threshold: never triggers
            0,        // max_sealed=0: flush immediately when any sealed memtable exists
            1,
            20, // fast poll for test
            0,  // age trigger disabled — isolating the sealed-count gate
        )
        .expect("start FlushManager");

        // Wait up to 500ms for the flush to complete.
        let mut flushed = false;
        for _ in 0..25 {
            std::thread::sleep(Duration::from_millis(20));
            if tree.sealed_memtable_count() == 0 {
                flushed = true;
                break;
            }
        }

        drop(mgr);
        assert!(
            flushed,
            "FlushManager should have flushed the sealed memtable"
        );
    }

    #[test]
    fn flush_manager_flushes_when_size_threshold_exceeded() {
        let (trees, _dir) = make_test_trees();
        let gc_watermark = Arc::new(AtomicU64::new(0));

        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(lsm_tree::SequenceNumberCounter::default());
        let tree = trees.get(&Partition::Node).expect("node tree").clone();

        // Write enough data to exceed a tiny threshold (1 byte).
        tree.insert(b"key1", b"value1_some_data", seqno.next());

        // FlushManager with threshold=1 byte so the active memtable exceeds it.
        let mgr = FlushManager::start(
            &trees,
            Arc::clone(&gc_watermark),
            1,   // 1 byte threshold — will always trigger
            100, // high sealed count so only size trigger fires
            1,
            20,
            0, // age trigger disabled — isolating the size gate
        )
        .expect("start FlushManager");

        // Wait up to 500ms for rotate + flush to complete.
        let mut flushed = false;
        for _ in 0..25 {
            std::thread::sleep(Duration::from_millis(20));
            // After flush: sealed=0 AND active size reset to 0.
            if tree.sealed_memtable_count() == 0 && tree.active_memtable().size() == 0 {
                flushed = true;
                break;
            }
        }

        drop(mgr);
        assert!(
            flushed,
            "FlushManager should have rotated and flushed the memtable"
        );
    }

    #[test]
    fn flush_manager_multiple_workers_no_panic() {
        let (trees, _dir) = make_test_trees();
        let gc_watermark = Arc::new(AtomicU64::new(0));

        // Start with 4 workers, rapid flush to exercise concurrency.
        let mgr = FlushManager::start(
            &trees,
            Arc::clone(&gc_watermark),
            1,  // 1 byte: always trigger
            0,  // max_sealed=0: always trigger
            4,  // 4 workers
            10, // fast poll
            0,  // age trigger disabled — concurrency comes from size+sealed
        )
        .expect("start FlushManager");

        std::thread::sleep(Duration::from_millis(150));
        drop(mgr); // must not panic or deadlock
    }

    #[test]
    fn flush_manager_age_trigger_rotates_idle_memtable() {
        // R076b: a memtable with even one byte of data must roll over to SST
        // after `max_memtable_age_secs`, independent of the size threshold.
        // Without this, light-load workloads could sit in volatile memory
        // for hours; combined with the R076a oplog purge gate, oplog
        // retention would grow without bound waiting for size-based flush.
        let (trees, _dir) = make_test_trees();
        let gc_watermark = Arc::new(AtomicU64::new(0));

        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(lsm_tree::SequenceNumberCounter::default());
        let tree = trees.get(&Partition::Node).expect("node tree").clone();

        // One tiny write — nowhere near the 64MB size threshold.
        tree.insert(b"k", b"v", seqno.next());
        assert!(
            tree.active_memtable().size() > 0,
            "precondition: data lives in the active memtable"
        );

        // age trigger = 1 second, size & sealed thresholds effectively off.
        let mgr = FlushManager::start(
            &trees,
            Arc::clone(&gc_watermark),
            u64::MAX, // size: never triggers
            usize::MAX,
            1,
            20, // poll every 20ms
            1,  // age trigger after 1s
        )
        .expect("start FlushManager");

        // The age trigger needs both wall time AND a poll tick to fire. Give
        // it 2× the age budget; flush worker then handles the sealed memtable.
        let mut flushed = false;
        for _ in 0..150 {
            std::thread::sleep(Duration::from_millis(20));
            if tree.sealed_memtable_count() == 0 && tree.active_memtable().size() == 0 {
                flushed = true;
                break;
            }
        }

        drop(mgr);
        assert!(
            flushed,
            "age trigger should have rotated the lone memtable entry to SST"
        );
    }

    #[test]
    fn flush_manager_age_zero_disables_time_based_trigger() {
        // max_memtable_age_secs == 0 must preserve the pre-R076b behavior:
        // size and sealed-count gates alone, no implicit time rotation.
        let (trees, _dir) = make_test_trees();
        let gc_watermark = Arc::new(AtomicU64::new(0));

        let seqno: lsm_tree::SharedSequenceNumberGenerator =
            Arc::new(lsm_tree::SequenceNumberCounter::default());
        let tree = trees.get(&Partition::Node).expect("node tree").clone();

        tree.insert(b"k", b"v", seqno.next());

        let mgr = FlushManager::start(
            &trees,
            Arc::clone(&gc_watermark),
            u64::MAX, // size: never triggers
            usize::MAX,
            1,
            20,
            0, // age trigger DISABLED
        )
        .expect("start FlushManager");

        // Even after several poll cycles the active memtable must still hold
        // its byte — nothing else has been touched to push it out.
        std::thread::sleep(Duration::from_millis(400));
        let stayed = tree.active_memtable().size() > 0 && tree.sealed_memtable_count() == 0;
        drop(mgr);
        assert!(
            stayed,
            "with age trigger off, idle memtable must remain in memory"
        );
    }
}
