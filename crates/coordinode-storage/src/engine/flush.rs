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
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use lsm_tree::AbstractTree;

use crate::engine::coordinator::GcWatermarks;
use crate::engine::open_sets::OpenSets;
use crate::engine::oplog_journal::{EmbeddedOplog, sync_journal};
use crate::engine::partition::Partition;
use crate::engine::scope::TreeId;
use crate::error::{StorageError, StorageResult};
use coordinode_core::txn::wake::Wake;

/// The smallest memtable size an endpoint's recovery reserve brings the
/// flush threshold down to: below it a store would flush almost every write.
const MIN_LIMITED_FLUSH_BYTES: u64 = 64 * 1024;

/// The memtable size past which each partition's memtable is flushed: the
/// configured size, brought down for a partition whose flushes land on an
/// endpoint with a hard limit to half that endpoint's recovery reserve, so
/// a memtable becomes a table while the reserve, where flushes run, still
/// has room for it. Scheduling only: nothing is held for the flush. The
/// write path's trigger and the monitor read the same value.
#[derive(Debug)]
pub(crate) struct FlushThresholds {
    configured: u64,
    /// Per partition, by [`Partition::position`]; a columnar table's tree
    /// lands where the node partition's does.
    by_partition: Vec<AtomicU64>,
    /// The footprint of the endpoint each partition's flushes land on, when
    /// it has a hard limit.
    landing: Vec<Option<Arc<crate::engine::footprint::Footprint>>>,
}

impl FlushThresholds {
    /// `configured` for every partition, each brought down by the reserve of
    /// the limited endpoint `landing` names for it.
    pub(crate) fn new(
        configured: u64,
        landing: impl Fn(Partition) -> Option<Arc<crate::engine::footprint::Footprint>>,
    ) -> Self {
        let landing: Vec<_> = Partition::all().iter().map(|&p| landing(p)).collect();
        let thresholds = Self {
            configured,
            by_partition: Partition::all()
                .iter()
                .map(|_| AtomicU64::new(configured))
                .collect(),
            landing,
        };
        thresholds.refresh();
        thresholds
    }

    /// Take the endpoints' reserves again, after a limit or reserve changed.
    pub(crate) fn refresh(&self) {
        for (threshold, landing) in self.by_partition.iter().zip(&self.landing) {
            let value = match landing.as_ref().map(|f| f.reserve()) {
                // No limit, or no reserve to fit a flush into: the configured
                // size, never one that would flush every write.
                None | Some(0) => self.configured,
                Some(reserve) => self
                    .configured
                    .min((reserve / 2).max(MIN_LIMITED_FLUSH_BYTES)),
            };
            threshold.store(value, Ordering::Release);
        }
    }

    /// The threshold of `partition`'s memtable.
    #[inline]
    pub(crate) fn of(&self, partition: Partition) -> u64 {
        self.by_partition[partition.position()].load(Ordering::Acquire)
    }
}

/// How the write path tells the flush monitor that a memtable may be due.
///
/// Held by every writer of a partition tree. A write that leaves its memtable
/// above its partition's threshold, or that is the memtable's first entry
/// (which starts its age), wakes the monitor; any other write does nothing
/// beyond one comparison.
#[derive(Debug)]
pub(crate) struct FlushTrigger {
    wake: Arc<Wake>,
    thresholds: Arc<FlushThresholds>,
}

impl FlushTrigger {
    /// A trigger waking `wake` for memtables above their `thresholds`.
    pub(crate) fn new(wake: Arc<Wake>, thresholds: Arc<FlushThresholds>) -> Self {
        Self { wake, thresholds }
    }

    /// A write of `added` bytes left `partition`'s memtable at
    /// `memtable_bytes`.
    #[inline]
    pub(crate) fn wrote(&self, partition: Partition, added: u64, memtable_bytes: u64) {
        if memtable_bytes > self.thresholds.of(partition) || memtable_bytes == added {
            self.wake.notify();
        }
    }

    /// A write whose effect on the memtable's size is not reported (a range
    /// tombstone): wake the monitor rather than guess.
    pub(crate) fn wrote_unmeasured(&self) {
        self.wake.notify();
    }
}

/// The memtable axis of the engine's write-pressure verdict: the unflushed
/// bytes of every tree sharing `budget`, judged against `thresholds` each time
/// the monitor runs and latched into `tier` (0/1/2 = none/slowdown/stop).
pub(crate) struct MemtablePressure {
    pub(crate) budget: Arc<lsm_tree::WriteBufferBudget>,
    pub(crate) thresholds: lsm_tree::BackpressureThresholds,
    pub(crate) tier: Arc<AtomicU8>,
}

impl MemtablePressure {
    /// Judge the unflushed bytes now and latch the tier.
    fn latch(&self) {
        let unflushed = self.budget.unflushed_bytes();
        let verdict = lsm_tree::Backpressure::compute(
            &lsm_tree::BackpressureSignals {
                unflushed_memtable_bytes: unflushed,
                ..lsm_tree::BackpressureSignals::default()
            },
            &self.thresholds,
        );
        let tier = match verdict {
            lsm_tree::Backpressure::None => 0u8,
            lsm_tree::Backpressure::Slowdown { .. } => 1,
            lsm_tree::Backpressure::Stop => 2,
        };
        let was = self.tier.swap(tier, Ordering::Relaxed);
        if tier == 2 && was != 2 {
            // An operator seeing WRITE_BACKPRESSURE rejections needs to know
            // the flushes, not compaction, fell behind.
            tracing::warn!(
                unflushed_memtable_bytes = unflushed,
                "write-backpressure STOP: shedding client writes until the \
                 memtables are flushed"
            );
        }
        metrics::gauge!("coordinode_storage_unflushed_memtable_bytes").set(unflushed as f64);
    }
}

/// A tree that comes and goes while the engine runs: a `STORAGE COLUMNAR`
/// table's, with where its write-pressure verdict is latched.
#[derive(Clone)]
pub(crate) struct TableTree {
    /// The table's id.
    pub(crate) id: String,
    /// Its tree, held weakly: a dropped table closes without waiting for the
    /// monitors, and a closing one is not raised again.
    pub(crate) tree: lsm_tree::WeakTree,
    /// The table's own verdict (0/1/2 = none/slowdown/stop), latched by the
    /// compaction monitor and read by the table's writes. One per table, so
    /// it also tells a table from one created again under its id.
    pub(crate) pressure: Arc<AtomicU8>,
    /// Where the table's write-pressure series is set, while its tree is
    /// held: its drop removes the series once the tree has closed.
    pub(crate) metrics: Arc<crate::engine::owned_metrics::TableMetrics>,
}

/// The tree a flush or compaction is asked for: a partition's, which lives
/// as long as the engine, or a table's, held weakly so that a dropped table
/// closes without waiting for the requests queued for it.
#[derive(Clone)]
pub(crate) enum TreeRef {
    /// A partition's tree.
    Partition(lsm_tree::WeakTree),
    /// A table's tree.
    Table(lsm_tree::WeakTree),
}

impl TreeRef {
    /// The request for `tree`, weak: it holds open neither a table nor a
    /// namespace's tree, either of which may close (dropped, or idle).
    pub(crate) fn of(tree: &lsm_tree::AnyTree, table: bool) -> Self {
        if table {
            Self::Table(tree.downgrade())
        } else {
            Self::Partition(tree.downgrade())
        }
    }

    /// Run `work` on the tree while it is neither closed nor closing, whose
    /// closing then waits for the work. `None` when the tree is gone.
    pub(crate) fn with_live<T>(&self, work: impl FnOnce(&lsm_tree::AnyTree) -> T) -> Option<T> {
        match self {
            Self::Partition(tree) | Self::Table(tree) => tree.upgrade().map(|tree| work(&tree)),
        }
    }
}

/// The tables, listed afresh on every monitor pass.
pub(crate) type TableSource = Box<dyn Fn() -> Vec<TableTree> + Send>;

/// Which tree a flush request is for: the name its log lines carry.
enum FlushSource {
    Partition(TreeId),
    Table(String),
}

impl std::fmt::Display for FlushSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Partition(tree) => write!(f, "{tree}"),
            Self::Table(table) => f.write_str(table),
        }
    }
}

/// Request to flush sealed memtables for a single tree.
struct FlushRequest {
    /// The tree to flush.
    tree: TreeRef,
    /// Which tree it is, for trace logging only.
    source: FlushSource,
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
    ///
    /// `journal` is the engine's journal, when it keeps one: a worker syncs it
    /// before it writes a sealed memtable, so no table holds an effect of an
    /// entry the journal could still lose. `pressure` is the memtable axis
    /// of the write-pressure verdict the monitor latches on every pass.
    /// `tables` lists the trees created and dropped at run time, which the
    /// monitor flushes under the same triggers as the partition trees.
    /// `landed` is notified each time a flush writes a table: the journal it
    /// covers may then go ([`crate::engine::journal_reclaim`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start(
        trees: Arc<OpenSets>,
        gc_watermarks: GcWatermarks,
        thresholds: Arc<FlushThresholds>,
        max_sealed: usize,
        num_workers: usize,
        max_memtable_age_secs: u64,
        wake: Arc<Wake>,
        compaction_wake: Arc<Wake>,
        journal: Option<Arc<Mutex<EmbeddedOplog>>>,
        pressure: MemtablePressure,
        tables: Option<TableSource>,
        footprints: crate::engine::footprint::Footprints,
        landed: Arc<Wake>,
    ) -> StorageResult<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        footprints.set_flush_wake(Arc::clone(&wake));

        // Bounded channel: capacity = workers × 4 so burst flushes don't stall the monitor.
        let capacity = (num_workers * 4).max(8);
        let (sender, receiver) = flume::bounded::<FlushRequest>(capacity);

        // Spawn N worker threads — each holds a receiver clone (flume supports multi-consumer).
        let mut workers = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let rx = receiver.clone();
            let flushed = Arc::clone(&wake);
            let compaction = Arc::clone(&compaction_wake);
            let journal = journal.clone();
            let footprints = footprints.clone();
            let landed = Arc::clone(&landed);
            let handle = std::thread::Builder::new()
                .name(format!("coord-flush-worker-{i}"))
                .spawn(move || {
                    flush_worker_loop(
                        rx,
                        Wakes {
                            flushed: &flushed,
                            compaction: &compaction,
                            landed: &landed,
                        },
                        journal.as_deref(),
                        &footprints,
                    );
                })
                .map_err(|e| StorageError::InvalidConfig(format!("flush worker spawn: {e}")))?;
            workers.push(handle);
        }

        // Spawn monitor thread. It lists the trees afresh on every pass: a
        // namespace's set comes and goes while the engine runs.
        let tx = sender.clone();
        let shutdown_m = Arc::clone(&shutdown);
        let wake_m = Arc::clone(&wake);
        let monitor = std::thread::Builder::new()
            .name("coord-flush-monitor".to_string())
            .spawn(move || {
                flush_monitor_loop(
                    &trees,
                    FlushMonitorConfig {
                        sender: tx,
                        gc_watermarks,
                        thresholds,
                        max_sealed,
                        max_memtable_age_secs,
                        shutdown: shutdown_m,
                        wake: wake_m,
                        pressure,
                        tables,
                        footprints,
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
    /// Each partition's memtable size threshold; see [`FlushThresholds`].
    thresholds: Arc<FlushThresholds>,
    max_sealed: usize,
    max_memtable_age_secs: u64,
    shutdown: Arc<AtomicBool>,
    wake: Arc<Wake>,
    pressure: MemtablePressure,
    tables: Option<TableSource>,
    /// The endpoints' hard limits, whose refusals ask every memtable holding
    /// data to be flushed ([`Footprints::squeeze`]).
    footprints: crate::engine::footprint::Footprints,
}

/// What the monitor keeps per tree between passes.
struct TreeState {
    /// When the tree's memtable last rotated, or the monitor started.
    last_rotate: Instant,
    /// Whether a flush request for the tree is queued or running.
    queued: Arc<AtomicBool>,
}

impl TreeState {
    fn new(now: Instant) -> Self {
        Self {
            last_rotate: now,
            queued: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// One tree's turn in a monitor pass: rotate its memtable when a trigger
/// fires, queue a flush of its sealed memtables, and lower `next_due` to when
/// its memtable comes due by age.
#[allow(clippy::too_many_arguments)]
fn consider_tree(
    tree: &lsm_tree::AnyTree,
    lands_as: Partition,
    state: &mut TreeState,
    source: impl FnOnce() -> (FlushSource, u64),
    now: Instant,
    cfg: &FlushMonitorConfig,
    squeezed: bool,
    next_due: &mut Option<Duration>,
) {
    let max_age = Duration::from_secs(cfg.max_memtable_age_secs);
    let active_bytes = tree.active_memtable().size();
    let age = now.saturating_duration_since(state.last_rotate);
    let age_triggered = cfg.max_memtable_age_secs > 0 && active_bytes > 0 && age >= max_age;

    let rotate = active_bytes > cfg.thresholds.of(lands_as)
        || tree.sealed_memtable_count() > cfg.max_sealed
        || age_triggered
        || (squeezed && active_bytes > 0);

    if rotate {
        // Rotate: atomically seal the active memtable → it joins the sealed list.
        // A fresh empty memtable becomes the new active.
        tree.rotate_memtable();
        state.last_rotate = now;
    } else if cfg.max_memtable_age_secs > 0 && active_bytes > 0 {
        let remaining = max_age.saturating_sub(age);
        *next_due = Some(next_due.map_or(remaining, |due| due.min(remaining)));
    }

    // Every sealed memtable is flushed, not only one sealed just now: a
    // request turned away while the workers were busy leaves its memtable
    // sealed, and an idle tree never trips the gates above again. One request
    // per tree at a time, because a flush takes every sealed memtable of its
    // tree.
    if tree.sealed_memtable_count() > 0 && !state.queued.load(Ordering::Acquire) {
        let (source, gc_watermark) = source();
        let req = FlushRequest {
            tree: TreeRef::of(tree, matches!(source, FlushSource::Table(_))),
            source,
            gc_watermark,
            queued: Arc::clone(&state.queued),
        };
        state.queued.store(true, Ordering::Release);
        // Non-blocking: with the workers busy the request waits for the next
        // pass, which a finished flush wakes.
        if cfg.sender.try_send(req).is_err() {
            state.queued.store(false, Ordering::Release);
        }
    }
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
fn flush_monitor_loop(trees: &OpenSets, cfg: FlushMonitorConfig) {
    cfg.wake.bind();
    // A tree's state lives as long as its set or table: one dropped since
    // the last pass loses it, one opened starts its age at the pass that
    // finds it.
    let mut states: HashMap<TreeId, TreeState> = HashMap::new();
    let mut table_states: HashMap<String, TreeState> = HashMap::new();

    while !cfg.shutdown.load(Ordering::Relaxed) {
        let now = Instant::now();
        // The soonest a memtable that holds data and stays below the other
        // triggers becomes due by age. None while every memtable is empty: an
        // idle engine sleeps until a write wakes it.
        let mut next_due: Option<Duration> = None;
        // A hard limit refused a write: every memtable holding data goes.
        let squeezed = cfg.footprints.take_squeeze();

        let listed = trees.list();
        states.retain(|id, _| listed.iter().any(|(listed, _)| listed == id));
        for (id, tree) in &listed {
            let state = states.entry(*id).or_insert_with(|| TreeState::new(now));
            let source = || {
                (
                    FlushSource::Partition(*id),
                    cfg.gc_watermarks.for_partition(id.part),
                )
            };
            consider_tree(
                tree,
                id.part,
                state,
                source,
                now,
                &cfg,
                squeezed,
                &mut next_due,
            );
        }
        if let Some(tables) = &cfg.tables {
            let listed = tables();
            table_states.retain(|table, _| listed.iter().any(|t| &t.id == table));
            for TableTree {
                id: table, tree, ..
            } in listed
            {
                // Dropped since it was listed.
                let Some(tree) = tree.upgrade() else {
                    continue;
                };
                let state = table_states
                    .entry(table.clone())
                    .or_insert_with(|| TreeState::new(now));
                // The tables are node rows kept apart: they keep the
                // versions the node partition keeps.
                let source = || {
                    (
                        FlushSource::Table(table),
                        cfg.gc_watermarks.for_partition(Partition::Node),
                    )
                };
                consider_tree(
                    &tree,
                    Partition::Node,
                    state,
                    source,
                    now,
                    &cfg,
                    squeezed,
                    &mut next_due,
                );
            }
        }

        // After the rotations above, so a backlog this pass queued is judged
        // with it; a finished flush wakes the monitor to judge it again.
        cfg.pressure.latch();

        // The handles go before the sleep: a tree closing (its namespace
        // idle or dropped) waits for every holder, and a sleeping monitor
        // would hold it until its next pass.
        drop(listed);
        cfg.wake.wait(next_due);
    }
}

/// Who a flush worker tells what happened.
struct Wakes<'a> {
    /// The monitor: the sealed backlog moved.
    flushed: &'a Wake,
    /// The compaction monitor: a new table may need compacting.
    compaction: &'a Wake,
    /// The journal reclaimer: a table landed.
    landed: &'a Wake,
}

/// Worker loop: receives flush requests and flushes sealed memtables to SST.
///
/// Blocks on the channel with no timeout: the manager's drop closes the
/// channel, which is what ends the loop.
fn flush_worker_loop(
    receiver: flume::Receiver<FlushRequest>,
    Wakes {
        flushed,
        compaction,
        landed,
    }: Wakes<'_>,
    journal: Option<&Mutex<EmbeddedOplog>>,
    footprints: &crate::engine::footprint::Footprints,
) {
    while let Ok(FlushRequest {
        tree,
        source,
        gc_watermark,
        queued,
    }) = receiver.recv()
    {
        // Started before the flush: a change of room while it runs is seen
        // if it then fails.
        let attempt = footprints.flush_attempt();
        let result = tree.with_live(|tree| {
            // get_flush_lock() is not Send: it is acquired and used in this
            // thread.
            let flush_lock = tree.get_flush_lock();
            // Every entry a sealed memtable holds was journaled before it was
            // applied, so a sync now covers them all.
            match journal.map_or(Ok(()), sync_journal) {
                Ok(()) => tree.flush(&flush_lock, gc_watermark),
                Err(e) => Err(lsm_tree::Error::Io(lsm_tree::io::Error::other(
                    e.to_string(),
                ))),
            }
        });
        // Before the wake below, and before a failure is registered, so the
        // pass either starts may ask again.
        queued.store(false, Ordering::SeqCst);
        // A table dropped since the request was queued has nothing to flush.
        let Some(result) = result else {
            continue;
        };
        match result {
            Ok(Some(bytes)) => {
                tracing::debug!(
                    tree = %source,
                    flushed_bytes = bytes,
                    "memtable flushed to SST"
                );
                // A new table may need compacting, and the journal it covers
                // may go.
                compaction.notify();
                landed.notify();
            }
            Ok(None) => {
                // Nothing to flush: another worker or the monitor already did it.
            }
            Err(e) => {
                tracing::error!(
                    tree = %source,
                    error = %e,
                    "memtable flush failed"
                );
                // Retried on the monitor's next pass, which the next write or
                // flush starts, or room coming back on a capped endpoint
                // since the flush started ([`FlushAttempt::failed`], after
                // `queued` is clear); waking it now would retry a failing
                // disk in a loop.
                attempt.failed();
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
