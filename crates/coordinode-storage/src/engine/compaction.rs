//! CompactionScheduler: priority-based LSM compaction worker pool (R073).
//!
//! Monitor thread polls all partition trees every `poll_interval_ms` and
//! submits [`CompactionRequest`]s sorted by priority (Urgent → High →
//! Normal → Low) via a flume channel to N worker threads.
//! Workers call `tree.compact(Leveled::default(), gc_watermark)`.
//!
//! Priority rules (per partition per poll cycle):
//!   - **Urgent**: `l0_run_count > l0_urgent_threshold` — write stall imminent
//!   - **High**: Adj partition — posting lists benefit most from compaction
//!   - **Low**: Blob partition — large values, compaction is expensive
//!   - **Normal**: all other partitions (Node, EdgeProp, Schema, Idx, BlobRef, Raft)
//!
//! Sub-compaction parallelism is achieved by having N workers compact
//! different partitions concurrently from the shared priority queue.
//!
//! Shutdown is automatic via [`Drop`] — identical lifecycle to [`super::flush::FlushManager`].

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

use lsm_tree::AbstractTree;

use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// Compaction priority — lower numeric value = higher urgency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CompactionPriority {
    /// L0 run count exceeded threshold: write stall imminent.
    Urgent = 0,
    /// Adj partition: posting lists benefit most from compaction.
    High = 1,
    /// All other partitions (Node, EdgeProp, Schema, Idx, BlobRef, Raft).
    Normal = 2,
    /// Blob partition: large values make compaction expensive.
    Low = 3,
}

/// Request to compact a single partition tree.
struct CompactionRequest {
    tree: lsm_tree::AnyTree,
    partition: Partition,
    priority: CompactionPriority,
    gc_watermark: u64,
}

/// Background priority-based LSM compaction worker pool.
///
/// Started by [`crate::engine::core::StorageEngine::finish_open`] and dropped
/// automatically when the engine drops. Declared as the second field in
/// `StorageEngine` (after `flush_manager`, before `trees`) to ensure worker
/// threads are joined before tree handles are released.
pub(crate) struct CompactionScheduler {
    workers: Vec<std::thread::JoinHandle<()>>,
    monitor: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    /// Sender clone held here so it can be dropped explicitly before joining workers.
    sender: Option<flume::Sender<CompactionRequest>>,
}

impl CompactionScheduler {
    /// Start the compaction scheduler.
    ///
    /// Spawns one monitor thread and `num_workers` worker threads.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any background thread fails to spawn (OS resource limit).
    pub(crate) fn start(
        trees: &HashMap<Partition, lsm_tree::AnyTree>,
        gc_watermark: Arc<AtomicU64>,
        num_workers: usize,
        l0_urgent_threshold: usize,
        poll_interval_ms: u64,
    ) -> StorageResult<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));

        // Bounded channel: capacity = workers × 4 (one slot per partition per poll).
        let capacity = (num_workers * 4).max(8);
        let (sender, receiver) = flume::bounded::<CompactionRequest>(capacity);

        // Spawn N worker threads.
        let mut workers = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let rx = receiver.clone();
            let shutdown_w = Arc::clone(&shutdown);
            let handle = std::thread::Builder::new()
                .name(format!("coord-compact-worker-{i}"))
                .spawn(move || compaction_worker_loop(rx, shutdown_w))
                .map_err(|e| {
                    StorageError::InvalidConfig(format!("compaction worker spawn: {e}"))
                })?;
            workers.push(handle);
        }

        // Clone tree handles for the monitor.
        let monitored: Vec<(Partition, lsm_tree::AnyTree)> =
            trees.iter().map(|(&p, t)| (p, t.clone())).collect();

        let tx = sender.clone();
        let shutdown_m = Arc::clone(&shutdown);
        let monitor = std::thread::Builder::new()
            .name("coord-compact-monitor".to_string())
            .spawn(move || {
                compaction_monitor_loop(
                    monitored,
                    tx,
                    gc_watermark,
                    l0_urgent_threshold,
                    poll_interval_ms,
                    shutdown_m,
                );
            })
            .map_err(|e| StorageError::InvalidConfig(format!("compaction monitor spawn: {e}")))?;

        Ok(Self {
            workers,
            monitor: Some(monitor),
            shutdown,
            sender: Some(sender),
        })
    }
}

impl Drop for CompactionScheduler {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(monitor) = self.monitor.take() {
            let _ = monitor.join();
        }
        drop(self.sender.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Assign a [`CompactionPriority`] to a partition based on its current L0 state.
///
/// Exposed for unit testing. Used by the monitor thread each poll cycle.
pub(crate) fn compaction_priority(
    partition: Partition,
    l0_run_count: usize,
    urgent_threshold: usize,
) -> CompactionPriority {
    if l0_run_count > urgent_threshold {
        return CompactionPriority::Urgent;
    }
    match partition {
        Partition::Adj => CompactionPriority::High,
        Partition::Blob => CompactionPriority::Low,
        _ => CompactionPriority::Normal,
    }
}

/// Monitor loop: polls all partition trees, assigns priorities, submits requests.
fn compaction_monitor_loop(
    trees: Vec<(Partition, lsm_tree::AnyTree)>,
    sender: flume::Sender<CompactionRequest>,
    gc_watermark: Arc<AtomicU64>,
    l0_urgent_threshold: usize,
    poll_interval_ms: u64,
    shutdown: Arc<AtomicBool>,
) {
    while !shutdown.load(Ordering::Relaxed) {
        let watermark = gc_watermark.load(Ordering::Relaxed);

        // Build requests sorted by priority: Urgent (0) first, Low (3) last.
        let mut requests: Vec<CompactionRequest> = trees
            .iter()
            .map(|(partition, tree)| {
                let l0 = tree.l0_run_count();
                let priority = compaction_priority(*partition, l0, l0_urgent_threshold);
                CompactionRequest {
                    tree: tree.clone(),
                    partition: *partition,
                    priority,
                    gc_watermark: watermark,
                }
            })
            .collect();

        requests.sort_by_key(|r| r.priority);

        for req in requests {
            // Non-blocking: if the channel is full, workers are busy.
            // Skipped partitions will be retried on the next poll cycle.
            let _ = sender.try_send(req);
        }

        std::thread::sleep(std::time::Duration::from_millis(poll_interval_ms));
    }
}

/// Worker loop: receives compaction requests and executes Leveled compaction.
fn compaction_worker_loop(receiver: flume::Receiver<CompactionRequest>, shutdown: Arc<AtomicBool>) {
    loop {
        match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(CompactionRequest {
                tree,
                partition,
                priority,
                gc_watermark,
            }) => {
                let strategy = Arc::new(lsm_tree::compaction::Leveled::default());
                match tree.compact(strategy, gc_watermark) {
                    Ok(result)
                        if result.action != lsm_tree::compaction::CompactionAction::Nothing =>
                    {
                        tracing::debug!(
                            partition = partition.name(),
                            ?priority,
                            tables_in = result.tables_in,
                            tables_out = result.tables_out,
                            "compaction completed"
                        );
                    }
                    Ok(_) => {
                        // Nothing to compact — Leveled strategy found no work.
                    }
                    Err(e) => {
                        tracing::error!(
                            partition = partition.name(),
                            error = %e,
                            "compaction failed"
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
                break;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
