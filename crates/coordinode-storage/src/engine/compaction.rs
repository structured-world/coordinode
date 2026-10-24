//! CompactionScheduler: priority-based LSM compaction worker pool.
//!
//! Monitor thread checks all partition trees when something changed what
//! compaction has to do (a flush added a table, a compaction finished) and
//! submits [`CompactionRequest`]s sorted by priority (Urgent → High → Normal
//! → Low) via a flume channel to N worker threads. Workers call
//! `tree.compact(strategy, gc_watermark)` with the tree's own strategy of the
//! configured kind. The `STORAGE COLUMNAR` tables, created and dropped at run
//! time, are compacted with the partitions. Between events nothing
//! runs: an idle engine has nothing to compact, and asking every tree again
//! on a timer only costs wakeups.
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
    Arc,
    atomic::{AtomicBool, Ordering},
};

use lsm_tree::AbstractTree;

use crate::engine::coordinator::GcWatermarks;
use crate::engine::flush::TableSource;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};
use coordinode_core::txn::wake::Wake;

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

/// How the trees' compactions are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionStrategyKind {
    /// Leveled compaction: each level a fixed size ratio over the one above.
    #[default]
    Leveled,
    /// Merges planned for query throughput over a compaction round, sized to
    /// the maintenance budget (Wang et al., SIGMOD 2025).
    Planned,
}

impl CompactionStrategyKind {
    /// A strategy of this kind for one tree: a planned one keeps the plan it
    /// worked out for that tree, so trees never share one.
    pub(crate) fn for_tree(self) -> Arc<dyn lsm_tree::compaction::CompactionStrategy> {
        match self {
            Self::Leveled => Arc::new(lsm_tree::compaction::Leveled::default()),
            Self::Planned => Arc::new(lsm_tree::compaction::Planned::default()),
        }
    }
}

/// Which tree a compaction request is for.
#[derive(Clone)]
enum CompactionSource {
    Partition(Partition),
    Table,
}

impl CompactionSource {
    /// The `partition` label of its metrics and log lines: the columnar
    /// tables share one, so their number does not grow the label set.
    fn name(&self) -> &'static str {
        match self {
            Self::Partition(partition) => partition.name(),
            Self::Table => "tables",
        }
    }
}

/// Request to compact a single tree.
struct CompactionRequest {
    tree: lsm_tree::AnyTree,
    source: CompactionSource,
    strategy: Arc<dyn lsm_tree::compaction::CompactionStrategy>,
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
    /// The monitor's wakeup, notified on shutdown so the monitor sees the flag.
    wake: Arc<Wake>,
    /// Sender clone held here so it can be dropped explicitly before joining workers.
    sender: Option<flume::Sender<CompactionRequest>>,
}

impl CompactionScheduler {
    /// Start the compaction scheduler.
    ///
    /// Spawns one monitor thread, woken through `wake` (which the flush
    /// workers notify after each flush), and `num_workers` worker threads.
    /// Every tree compacts under a strategy of kind `strategy`; `tables` lists
    /// the trees created and dropped at run time, compacted with the
    /// partition trees.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any background thread fails to spawn (OS resource limit).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start(
        trees: &HashMap<Partition, lsm_tree::AnyTree>,
        gc_watermarks: GcWatermarks,
        num_workers: usize,
        l0_urgent_threshold: usize,
        debt_urgent_bytes: u64,
        write_pressure: Arc<std::sync::atomic::AtomicU8>,
        wake: Arc<Wake>,
        strategy: CompactionStrategyKind,
        tables: Option<TableSource>,
    ) -> StorageResult<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));

        // Bounded channel: capacity = workers × 4 (one slot per partition per poll).
        let capacity = (num_workers * 4).max(8);
        let (sender, receiver) = flume::bounded::<CompactionRequest>(capacity);

        // Spawn N worker threads.
        let mut workers = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let rx = receiver.clone();
            let compacted = Arc::clone(&wake);
            let handle = std::thread::Builder::new()
                .name(format!("coord-compact-worker-{i}"))
                .spawn(move || compaction_worker_loop(rx, &compacted))
                .map_err(|e| {
                    StorageError::InvalidConfig(format!("compaction worker spawn: {e}"))
                })?;
            workers.push(handle);
        }

        // Clone tree handles for the monitor, each with the strategy it
        // compacts under for as long as the engine runs.
        let monitored: Vec<MonitoredTree> = trees
            .iter()
            .map(|(&p, t)| MonitoredTree {
                tree: t.clone(),
                source: CompactionSource::Partition(p),
                strategy: strategy.for_tree(),
            })
            .collect();

        let tx = sender.clone();
        let shutdown_m = Arc::clone(&shutdown);
        let wake_m = Arc::clone(&wake);
        let monitor = std::thread::Builder::new()
            .name("coord-compact-monitor".to_string())
            .spawn(move || {
                compaction_monitor_loop(
                    monitored,
                    MonitorConfig {
                        sender: tx,
                        gc_watermarks,
                        l0_urgent_threshold,
                        debt_urgent_bytes,
                        write_pressure,
                        shutdown: shutdown_m,
                        strategy,
                        tables,
                    },
                    &wake_m,
                );
            })
            .map_err(|e| StorageError::InvalidConfig(format!("compaction monitor spawn: {e}")))?;

        Ok(Self {
            workers,
            monitor: Some(monitor),
            shutdown,
            wake,
            sender: Some(sender),
        })
    }
}

impl Drop for CompactionScheduler {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.wake.notify();
        if let Some(monitor) = self.monitor.take() {
            let _ = monitor.join();
        }
        drop(self.sender.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Assign a [`CompactionPriority`] to a partition based on its current L0
/// state and pending-compaction byte debt.
///
/// Either urgency axis alone is sufficient: a tall L0 spikes read
/// amplification even with little total debt, and a large byte debt starves
/// the size targets even when L0 itself is short (heavy overwrite workloads
/// accumulate debt in the mid levels).
///
/// Exposed for unit testing. Used by the monitor thread each poll cycle.
pub(crate) fn compaction_priority(
    partition: Partition,
    l0_run_count: usize,
    urgent_threshold: usize,
    debt_bytes: u64,
    debt_urgent_bytes: u64,
) -> CompactionPriority {
    if l0_run_count > urgent_threshold || (debt_urgent_bytes > 0 && debt_bytes >= debt_urgent_bytes)
    {
        return CompactionPriority::Urgent;
    }
    match partition {
        Partition::Adj => CompactionPriority::High,
        Partition::Blob => CompactionPriority::Low,
        _ => CompactionPriority::Normal,
    }
}

/// A tree the monitor watches, with the strategy it compacts under.
struct MonitoredTree {
    tree: lsm_tree::AnyTree,
    source: CompactionSource,
    strategy: Arc<dyn lsm_tree::compaction::CompactionStrategy>,
}

/// What the monitor runs with, bundled to keep its signature short.
struct MonitorConfig {
    sender: flume::Sender<CompactionRequest>,
    gc_watermarks: GcWatermarks,
    l0_urgent_threshold: usize,
    debt_urgent_bytes: u64,
    write_pressure: Arc<std::sync::atomic::AtomicU8>,
    shutdown: Arc<AtomicBool>,
    strategy: CompactionStrategyKind,
    tables: Option<TableSource>,
}

/// Monitor loop: polls all trees, assigns priorities, submits requests, and
/// latches the worst per-partition backpressure verdict into `write_pressure`
/// (0/1/2 = none/slowdown/stop) for the commit-path check.
fn compaction_monitor_loop(trees: Vec<MonitoredTree>, cfg: MonitorConfig, wake: &Wake) {
    wake.bind();
    // A table's strategy lives as long as the table, as a partition's does.
    let mut tables: HashMap<String, MonitoredTree> = HashMap::new();
    while !cfg.shutdown.load(Ordering::Relaxed) {
        if let Some(source) = &cfg.tables {
            let listed = source();
            tables.retain(|table, _| listed.iter().any(|(id, _)| id == table));
            for (table, tree) in listed {
                tables.entry(table).or_insert_with(|| MonitoredTree {
                    tree,
                    source: CompactionSource::Table,
                    strategy: cfg.strategy.for_tree(),
                });
            }
        }
        // Build requests sorted by priority: Urgent (0) first, Low (3) last.
        // Along the way, compute the worst backpressure tier across the
        // partitions: the commit path it gates writes no columnar table.
        let mut worst_tier: u8 = 0;
        let mut requests: Vec<CompactionRequest> = trees
            .iter()
            .chain(tables.values())
            .map(|monitored| {
                let MonitoredTree {
                    tree,
                    source,
                    strategy,
                } = monitored;
                let l0 = tree.l0_run_count();
                let debt = tree.compaction_debt(strategy.as_ref());
                // The tables are node rows kept apart: they keep the
                // versions the node partition keeps.
                let partition = match source {
                    CompactionSource::Partition(partition) => *partition,
                    CompactionSource::Table => Partition::Node,
                };
                let priority = compaction_priority(
                    partition,
                    l0,
                    cfg.l0_urgent_threshold,
                    debt,
                    cfg.debt_urgent_bytes,
                );
                if let CompactionSource::Partition(_) = source {
                    let tier = match tree.write_backpressure(strategy.as_ref()) {
                        lsm_tree::Backpressure::None => 0u8,
                        lsm_tree::Backpressure::Slowdown { .. } => 1,
                        lsm_tree::Backpressure::Stop => 2,
                    };
                    if tier == 2 {
                        // An operator seeing WRITE_BACKPRESSURE rejections
                        // needs to know which partition tripped it and on
                        // which axis.
                        tracing::warn!(
                            partition = source.name(),
                            l0_run_count = l0,
                            compaction_debt_bytes = debt,
                            "write-backpressure STOP: shedding client writes until \
                             compaction catches up"
                        );
                    }
                    worst_tier = worst_tier.max(tier);
                    metrics::gauge!(
                        "coordinode_storage_backpressure_tier",
                        "partition" => source.name()
                    )
                    .set(f64::from(tier));
                    metrics::gauge!(
                        "coordinode_storage_compaction_debt_bytes",
                        "partition" => source.name()
                    )
                    .set(debt as f64);
                }
                CompactionRequest {
                    tree: tree.clone(),
                    source: source.clone(),
                    strategy: Arc::clone(strategy),
                    priority,
                    gc_watermark: cfg.gc_watermarks.for_partition(partition),
                }
            })
            .collect();
        cfg.write_pressure.store(worst_tier, Ordering::Relaxed);

        requests.sort_by_key(|r| r.priority);

        for req in requests {
            // Non-blocking: if the channel is full, workers are busy. A
            // partition skipped here is asked again when a running compaction
            // finishes and wakes the monitor.
            let _ = cfg.sender.try_send(req);
        }

        wake.wait(None);
    }
}

/// Worker loop: receives compaction requests and runs each under its tree's
/// strategy.
///
/// Blocks on the channel with no timeout: the scheduler's drop closes the
/// channel, which is what ends the loop. A compaction that did work wakes the
/// monitor, because the result can make another level due.
fn compaction_worker_loop(receiver: flume::Receiver<CompactionRequest>, compacted: &Wake) {
    // The thread runs compactions only.
    let _maintenance = crate::engine::footprint::MaintenanceScope::enter();
    while let Ok(CompactionRequest {
        tree,
        source,
        strategy,
        priority,
        gc_watermark,
    }) = receiver.recv()
    {
        let started = std::time::Instant::now();
        match tree.compact(strategy, gc_watermark) {
            Ok(result) if result.action != lsm_tree::compaction::CompactionAction::Nothing => {
                metrics::counter!("coordinode_storage_compaction_total", "partition" => source.name())
                    .increment(1);
                metrics::histogram!(
                    "coordinode_storage_compaction_duration_seconds",
                    "partition" => source.name()
                )
                .record(started.elapsed().as_secs_f64());
                tracing::debug!(
                    partition = source.name(),
                    ?priority,
                    tables_in = result.tables_in,
                    tables_out = result.tables_out,
                    "compaction completed"
                );
                compacted.notify();
            }
            Ok(_) => {
                // Nothing to compact: the strategy found no work.
            }
            Err(e) => {
                tracing::error!(
                    partition = source.name(),
                    error = %e,
                    "compaction failed"
                );
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
