//! [`ShardConsumerRegistry`]: the Raft-replicated, per-shard implementation of
//! [`SeqnoConsumerRegistry`].
//!
//! Register / checkpoint / unregister are eager `RaftProposal`s into
//! `Partition::Registry`; the shard's retention floor is `min(checkpoint_seqno)`
//! over the live (non-expired) records, cached in an `Arc<AtomicU64>` so the
//! downstream feeds read it without re-scanning.
//!
//! Heartbeats and eviction run through an optional background service
//! ([`RegistryBackground`]): heartbeats buffer in-memory on the leader and
//! flush as a single coalesced proposal per `heartbeat_window_ms` window the
//! first buffered heartbeat opens (≤ `1000 / window` proposals/sec regardless
//! of consumer count); expired registrations are swept and removed by an
//! eviction proposal when the earliest one can expire, and the floor is
//! refreshed whenever the registry keyspace changes. An idle registry runs
//! nothing. Without the background service, `heartbeat` writes eagerly (used
//! by tests).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::Guard;
use coordinode_storage::engine::applied::AppliedStop;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use parking_lot::Mutex;

use super::SeqnoConsumerRegistry;
use super::entry::{REGISTRY_KEY_PREFIX, RegistryEntry, encode_registry_key};
use super::types::{
    ConsumerRegistration, ConsumerSnapshot, InitialSeqno, RegisteredHandle, RegistryError,
    TopologyScope,
};

/// Wall-clock source for heartbeat / TTL accounting.
///
/// Injected (not `std::time::SystemTime` hard-wired) so tests drive expiry
/// deterministically and a future no-std consumer can supply its own clock.
// no-std: caller-provided Clock trait (this is exactly that seam).
pub trait Clock: Send + Sync {
    /// Milliseconds since the Unix epoch.
    fn now_ms(&self) -> u64;
}

/// `std`-backed wall clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        // no-std: caller-provided Clock trait.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// Tuning for the background heartbeat-flush + eviction-sweep service.
#[derive(Debug, Clone, Copy)]
pub struct BackgroundConfig {
    /// Drain window for buffered heartbeats: the first heartbeat buffered
    /// opens a window, and the heartbeats of the window go in one proposal
    /// regardless of consumer count. Nothing buffered, nothing runs.
    pub heartbeat_window_ms: u64,
    /// Shortest gap between two eviction sweeps. A sweep runs when the
    /// earliest registration can expire and when the registry keyspace
    /// changes, never on a timer of its own.
    pub eviction_interval_ms: u64,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self {
            heartbeat_window_ms: 1_000,
            eviction_interval_ms: 1_000,
        }
    }
}

/// Shared registry state — held by the facade and the background tasks.
struct RegistryCore {
    engine: Arc<StorageEngine>,
    pipeline: Arc<dyn ProposalPipeline>,
    id_gen: Arc<ProposalIdGenerator>,
    clock: Arc<dyn Clock>,
    /// Cached `min(checkpoint_seqno)` over **MVCC-seqno-space** consumers
    /// (`LsmStateDelta` / `MvccSnapshotPin` / `Ephemeral`); `u64::MAX` when
    /// none. Published to the engine as the consumer retention floor (feed
    /// a); the engine combines it by `min` with its own time-travel window
    /// and live snapshot pins, so a consumer can only ever extend retention.
    floor: Arc<AtomicU64>,
    /// Cached `min(checkpoint)` over **oplog-index-space** consumers
    /// (`OplogEvents`, whose `ResumeToken` is a Raft log index); `u64::MAX`
    /// when none. Drives oplog segment retention (feed b): a segment is kept
    /// iff `last_index >= this` OR it is within the time window.
    oplog_index_floor: Arc<AtomicU64>,
    /// Whether `dc` / `rack` scopes are accepted (EE multi-DC topologies).
    allow_topology_scopes: bool,
    /// `true` once a background service is running: `heartbeat` then buffers
    /// into `pending_hb` instead of writing eagerly.
    batching_on: AtomicBool,
    /// Buffered heartbeats awaiting the next coalesced flush:
    /// `consumer_id → latest heartbeat ts`.
    pending_hb: Mutex<HashMap<String, u64>>,
    /// Notified when a heartbeat is buffered, opening a flush window.
    hb_buffered: tokio::sync::Notify,
}

impl RegistryCore {
    fn check_scope(&self, scope: &TopologyScope) -> Result<(), RegistryError> {
        match scope {
            TopologyScope::Dc(_) | TopologyScope::Rack(_) if !self.allow_topology_scopes => {
                Err(RegistryError::UnsupportedScope(format!("{scope:?}")))
            }
            _ => Ok(()),
        }
    }

    fn resolve_initial(&self, initial: InitialSeqno) -> u64 {
        match initial {
            InitialSeqno::FromNow => self.engine.snapshot(),
            InitialSeqno::FromEarliestRetained => 0,
            InitialSeqno::At(s) => s,
        }
    }

    fn read_entry(&self, consumer_id: &str) -> Result<Option<RegistryEntry>, RegistryError> {
        let key = encode_registry_key(consumer_id);
        let raw = self
            .engine
            .get(Partition::Registry, &key)
            .map_err(|e| RegistryError::Replication(e.to_string()))?;
        match raw {
            Some(bytes) => RegistryEntry::decode(&bytes)
                .map(Some)
                .map_err(|e| RegistryError::Replication(format!("decode registry entry: {e}"))),
            None => Ok(None),
        }
    }

    /// Propose a batch of registry mutations through Raft as one entry.
    fn propose(&self, mutations: Vec<Mutation>) -> Result<(), RegistryError> {
        if mutations.is_empty() {
            return Ok(());
        }
        let proposal = RaftProposal {
            id: self.id_gen.next(),
            mutations,
            // Registry keys are last-write-wins on a plain key (the engine
            // auto-stamps the LSM seqno); commit_ts is not used for
            // versioned-key encoding here.
            commit_ts: Timestamp::from_raw(0),
            start_ts: Timestamp::from_raw(0),
            bypass_rate_limiter: false,
        };
        self.pipeline
            .propose_and_wait(&proposal)
            .map(|_| ())
            .map_err(|e| RegistryError::Replication(e.to_string()))
    }

    fn put_mutation(entry: &RegistryEntry) -> Result<Mutation, RegistryError> {
        Ok(Mutation::Put {
            partition: PartitionId::Registry,
            key: encode_registry_key(&entry.consumer_id),
            value: entry
                .encode()
                .map_err(|e| RegistryError::Replication(format!("encode registry entry: {e}")))?,
        })
    }

    /// Re-scan the keyspace, dropping expired records, refresh the two
    /// space-split consumer floors, and publish the engine GC watermark
    /// (feed a, combine B). Returns the seqno-space floor.
    ///
    /// Floors split by consumer space ([`ConsumerKind::is_seqno_space`]):
    /// MVCC-seqno consumers drive the GC watermark, oplog-index consumers
    /// drive oplog retention. Mixing them would compare a microsecond HLC
    /// against a Raft log index.
    fn recompute_floor(&self) -> Result<u64, RegistryError> {
        let now = self.clock.now_ms();
        let mut seqno_floor = u64::MAX;
        let mut oplog_floor = u64::MAX;
        let mut count = 0u64;
        for entry in self.scan_entries(now)? {
            count += 1;
            if entry.kind.is_seqno_space() {
                seqno_floor = seqno_floor.min(entry.checkpoint_seqno);
            } else {
                oplog_floor = oplog_floor.min(entry.checkpoint_seqno);
            }
        }
        self.floor.store(seqno_floor, Ordering::Release);
        self.oplog_index_floor.store(oplog_floor, Ordering::Release);
        metrics::gauge!("registry_consumer_count").set(count as f64);
        metrics::gauge!("registry_shard_floor_seqno").set(seqno_floor as f64);
        metrics::gauge!("registry_oplog_floor_index").set(oplog_floor as f64);

        // Feed (a): the consumer floor. The engine combines it by `min` with
        // its own time-travel window (`retention_window_secs`) and live
        // snapshot pins, so no consumers (`u64::MAX`) leaves the window in
        // force and a consumer lagging beyond the window lowers the
        // watermark below it, extending retention for that consumer
        // (CockroachDB protected-timestamp / TiDB service-safe-point).
        self.engine.set_consumer_retention_floor(seqno_floor);
        Ok(seqno_floor)
    }

    /// Collect every live registration: not expired at `now_ms`, or with a
    /// heartbeat still in the buffer, a sign of life the stored entry does
    /// not show yet.
    fn scan_entries(&self, now_ms: u64) -> Result<Vec<RegistryEntry>, RegistryError> {
        let mut out = Vec::new();
        let iter = self
            .engine
            .prefix_scan(Partition::Registry, REGISTRY_KEY_PREFIX)
            .map_err(|e| RegistryError::Replication(e.to_string()))?;
        for guard in iter {
            let (_, value) = guard
                .into_inner()
                .map_err(|e| RegistryError::Replication(e.to_string()))?;
            let entry = RegistryEntry::decode(&value)
                .map_err(|e| RegistryError::Replication(format!("decode registry entry: {e}")))?;
            if !entry.is_expired(now_ms) || self.pending_hb.lock().contains_key(&entry.consumer_id)
            {
                out.push(entry);
            }
        }
        Ok(out)
    }

    /// Write the buffered heartbeats in one coalesced proposal.
    /// Consumers that vanished since buffering are silently skipped. Each
    /// heartbeat stays in the buffer until its write lands: while the write is
    /// in flight the stored entry does not show it yet, and the buffer is then
    /// the only sign of life a floor or a listing can see. A failed write
    /// leaves all of them there for the next flush.
    fn flush_pending_heartbeats(&self) -> Result<(), RegistryError> {
        let taken: Vec<(String, u64)> = {
            let pending = self.pending_hb.lock();
            if pending.is_empty() {
                return Ok(());
            }
            pending.iter().map(|(id, ts)| (id.clone(), *ts)).collect()
        };
        self.write_heartbeats(&taken)?;
        let mut pending = self.pending_hb.lock();
        for (consumer_id, ts) in taken {
            // A heartbeat buffered during the write is newer and stays.
            if pending.get(&consumer_id) == Some(&ts) {
                pending.remove(&consumer_id);
            }
        }
        Ok(())
    }

    /// Persist `heartbeats` in one proposal.
    fn write_heartbeats(&self, heartbeats: &[(String, u64)]) -> Result<(), RegistryError> {
        let mut mutations = Vec::with_capacity(heartbeats.len());
        for (consumer_id, ts) in heartbeats {
            if let Some(mut entry) = self.read_entry(consumer_id)? {
                entry.last_heartbeat_ts_ms = entry.last_heartbeat_ts_ms.max(*ts);
                mutations.push(Self::put_mutation(&entry)?);
            }
        }
        // One coalesced proposal per drained window.
        metrics::counter!("registry_heartbeat_batches_total").increment(1);
        metrics::histogram!("registry_heartbeat_batch_size").record(mutations.len() as f64);
        self.propose(mutations)
    }

    /// Evict registrations past their TTL via one Delete proposal, then
    /// refresh the floor (a dead consumer must stop pinning retention).
    fn sweep_evictions(&self) -> Result<Sweep, RegistryError> {
        // A heartbeat still in the buffer is a sign of life the stored entry
        // does not show yet: persist it first, or a live consumer whose TTL is
        // close to the flush window is evicted between its heartbeat and the
        // next flush.
        self.flush_pending_heartbeats()?;
        let now = self.clock.now_ms();
        let mut to_evict = Vec::new();
        let mut next_expiry_ms: Option<u64> = None;
        let iter = self
            .engine
            .prefix_scan(Partition::Registry, REGISTRY_KEY_PREFIX)
            .map_err(|e| RegistryError::Replication(e.to_string()))?;
        for guard in iter {
            let (_, value) = guard
                .into_inner()
                .map_err(|e| RegistryError::Replication(e.to_string()))?;
            let mut entry = RegistryEntry::decode(&value)
                .map_err(|e| RegistryError::Replication(format!("decode registry entry: {e}")))?;
            // The flush above can outlast a TTL; a heartbeat that arrived
            // meanwhile is only in the buffer, and it is a sign of life.
            let buffered = self.pending_hb.lock().get(&entry.consumer_id).copied();
            if entry.is_expired(now) && buffered.is_none() {
                to_evict.push(Mutation::Delete {
                    partition: PartitionId::Registry,
                    key: encode_registry_key(&entry.consumer_id),
                });
                continue;
            }
            if let Some(ts) = buffered {
                entry.last_heartbeat_ts_ms = entry.last_heartbeat_ts_ms.max(ts);
            }
            if let Some(at) = entry.expires_at_ms() {
                next_expiry_ms = Some(next_expiry_ms.map_or(at, |next| next.min(at)));
            }
        }
        let evicted = to_evict.len();
        if evicted > 0 {
            self.propose(to_evict)?;
            metrics::counter!("registry_evictions_total").increment(evicted as u64);
        }
        // Always refresh the floor: an expired registration that another
        // node evicted, or a checkpoint advanced through a different handle,
        // must reach the engine even when this sweep evicted nothing.
        self.recompute_floor()?;
        Ok(Sweep {
            evicted,
            next_expiry_ms,
        })
    }
}

/// What one eviction sweep did and when the next one is due.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sweep {
    /// Registrations removed.
    evicted: usize,
    /// When the earliest registration left can expire (clock ms), or `None`
    /// when none can.
    next_expiry_ms: Option<u64>,
}

/// Per-shard consumer-retention registry backed by `Partition::Registry`.
///
/// Construct with [`ShardConsumerRegistry::new`] (CE: `cluster` / `node` /
/// `shard` scopes) — `dc` / `rack` scopes require a multi-DC topology and are
/// rejected unless enabled via [`with_topology_scopes`](Self::with_topology_scopes)
/// (EE). Call [`start_background`](Self::start_background) on the leader to run
/// batched heartbeats + TTL eviction.
#[derive(Clone)]
pub struct ShardConsumerRegistry {
    core: Arc<RegistryCore>,
}

impl ShardConsumerRegistry {
    /// Open a CE registry over the given engine + proposal pipeline.
    pub fn new(
        engine: Arc<StorageEngine>,
        pipeline: Arc<dyn ProposalPipeline>,
        id_gen: Arc<ProposalIdGenerator>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let core = Arc::new(RegistryCore {
            engine,
            pipeline,
            id_gen,
            clock,
            floor: Arc::new(AtomicU64::new(u64::MAX)),
            oplog_index_floor: Arc::new(AtomicU64::new(u64::MAX)),
            allow_topology_scopes: false,
            batching_on: AtomicBool::new(false),
            pending_hb: Mutex::new(HashMap::new()),
            hb_buffered: tokio::sync::Notify::new(),
        });
        // Recover the floor + publish the engine watermark from any
        // registrations persisted in a prior life.
        let _ = core.recompute_floor();
        Self { core }
    }

    /// EE: also accept `dc` / `rack` scopes (requires a multi-DC topology).
    ///
    /// Must be called on a freshly-constructed registry (before it is cloned
    /// or a background task is spawned), while the `Arc` is uniquely held.
    pub fn with_topology_scopes(mut self) -> Self {
        if let Some(core) = Arc::get_mut(&mut self.core) {
            core.allow_topology_scopes = true;
        }
        self
    }

    /// A cheaply-cloned handle to the cached MVCC-seqno floor (feed a / gc
    /// watermark observability). The engine watermark itself is published via
    /// `set_consumer_retention_floor`; this handle is the raw consumer floor.
    pub fn floor_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.core.floor)
    }

    /// The oplog-index retention floor (feed b): `min(checkpoint)` over
    /// `OplogEvents` consumers, or `u64::MAX` when none. The oplog manager
    /// keeps a segment iff its last index `>= ` this OR it is within the time
    /// window (logical OR — the time policy is the safety net for shards with
    /// no CDC consumer). Raft-index space, distinct from
    /// [`shard_floor`](SeqnoConsumerRegistry::shard_floor).
    pub fn oplog_retention_floor(&self) -> u64 {
        self.core.oplog_index_floor.load(Ordering::Acquire)
    }

    /// Lagging-consumer guard: verify the consumer's checkpoint has
    /// not fallen below what the engine has already GC'd. A seqno-space
    /// consumer's read path calls this before reading-from-checkpoint; it
    /// returns the safe checkpoint, or [`RegistryError::RetentionLost`] when
    /// the versions it needs were collected (operator-forced GC bump, or the
    /// consumer registered too late) — never a silent gap.
    ///
    /// # Errors
    /// [`RegistryError::UnknownConsumer`] if the handle has no live
    /// registration; [`RegistryError::RetentionLost`] if
    /// `checkpoint < engine gc watermark`.
    ///
    /// Oplog-index consumers are not checked here (their lost-detection needs
    /// the oplog purged index, surfaced when feed (b) is wired into
    /// `LogStore::purge`); they return `Ok(checkpoint)`.
    pub fn check_retention(&self, handle: &RegisteredHandle) -> Result<u64, RegistryError> {
        let entry = self
            .core
            .read_entry(handle.consumer_id())?
            .ok_or_else(|| RegistryError::UnknownConsumer(handle.consumer_id().to_string()))?;
        if entry.kind.is_seqno_space() {
            let floor = self.core.engine.gc_watermark();
            if entry.checkpoint_seqno < floor {
                return Err(RegistryError::RetentionLost {
                    checkpoint: entry.checkpoint_seqno,
                    floor,
                });
            }
        }
        Ok(entry.checkpoint_seqno)
    }

    /// Start the leader-side background service: coalesced heartbeat flush +
    /// TTL eviction. Switches `heartbeat` to buffered mode. Returns a handle
    /// whose [`shutdown`](RegistryBackground::shutdown) does a final flush.
    pub fn start_background(&self, cfg: BackgroundConfig) -> RegistryBackground {
        self.core.batching_on.store(true, Ordering::Release);
        let core = Arc::clone(&self.core);
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let stop = Arc::clone(&shutdown);
        // Registry writes wait on a proposal, an fsync at least: they run on
        // the blocking pool, or the runtime thread they would hold is the one
        // the consumers' streams heartbeat from.
        let blocking =
            |core: &Arc<RegistryCore>,
             work: fn(&RegistryCore) -> Result<Option<u64>, RegistryError>| {
                let core = Arc::clone(core);
                async move {
                    tokio::task::spawn_blocking(move || work(&core))
                        .await
                        .unwrap_or_else(|e| {
                            Err(RegistryError::Replication(format!("registry task: {e}")))
                        })
                }
            };
        let flush = |core: &RegistryCore| core.flush_pending_heartbeats().map(|()| None);
        let sweep = |core: &RegistryCore| {
            core.sweep_evictions().map(|sweep| {
                if sweep.evicted > 0 {
                    tracing::debug!(evicted = sweep.evicted, "registry TTL sweep");
                }
                sweep.next_expiry_ms
            })
        };

        // A registration another member wrote changes the floor this member
        // publishes, so a write applied to the registry keyspace asks for a
        // sweep. The feed is a blocking subscription; one parked thread
        // relays it.
        let changed = Arc::new(tokio::sync::Notify::new());
        let applied = self.core.engine.subscribe_applied(Partition::Registry, 16);
        let applied_stop = applied.stopper();
        let relay = Arc::clone(&changed);
        let relay_thread = std::thread::Builder::new()
            .name("registry-applied".to_string())
            .spawn(move || {
                while applied.next(None).is_some() {
                    relay.notify_one();
                }
            })
            .map_err(|e| tracing::error!(error = %e, "registry apply relay did not start"))
            .ok();

        let window = Duration::from_millis(cfg.heartbeat_window_ms);
        let gap = Duration::from_millis(cfg.eviction_interval_ms);
        let handle = tokio::spawn(async move {
            let mut flush_at: Option<tokio::time::Instant> = None;
            // The first sweep publishes the floor the stored registrations hold.
            let mut sweep_at: Option<tokio::time::Instant> = Some(tokio::time::Instant::now());
            let mut last_sweep: Option<tokio::time::Instant> = None;
            loop {
                tokio::select! {
                    _ = stop.notified() => break,
                    _ = core.hb_buffered.notified(), if flush_at.is_none() => {
                        flush_at = Some(tokio::time::Instant::now() + window);
                    }
                    _ = changed.notified() => {
                        let soonest = last_sweep.map_or_else(tokio::time::Instant::now, |at| at + gap);
                        sweep_at = Some(sweep_at.map_or(soonest, |due| due.min(soonest)));
                    }
                    _ = sleep_until(flush_at), if flush_at.is_some() => {
                        flush_at = None;
                        if let Err(e) = blocking(&core, flush).await {
                            tracing::warn!(error = %e, "registry heartbeat flush failed");
                            // The heartbeats stay buffered: try again a window on.
                            flush_at = Some(tokio::time::Instant::now() + window);
                        }
                    }
                    _ = sleep_until(sweep_at), if sweep_at.is_some() => {
                        let swept_at = tokio::time::Instant::now();
                        last_sweep = Some(swept_at);
                        sweep_at = match blocking(&core, sweep).await {
                            Ok(next) => next.map(|at_ms| {
                                let wait = at_ms.saturating_sub(core.clock.now_ms());
                                swept_at + Duration::from_millis(wait).max(gap)
                            }),
                            Err(e) => {
                                tracing::warn!(error = %e, "registry eviction sweep failed");
                                Some(swept_at + gap)
                            }
                        };
                    }
                }
            }
            // Final flush so no buffered heartbeat is lost on graceful stop.
            if let Err(e) = blocking(&core, flush).await {
                tracing::warn!(error = %e, "registry final heartbeat flush failed");
            }
        });
        RegistryBackground {
            shutdown,
            handle: Some(handle),
            applied_stop,
            relay_thread,
            config: cfg,
        }
    }
}

/// Sleep until `at`; never return when there is no `at`.
async fn sleep_until(at: Option<tokio::time::Instant>) {
    match at {
        Some(at) => tokio::time::sleep_until(at).await,
        None => std::future::pending().await,
    }
}

/// Handle to the running background service. Drop detaches the task; prefer
/// [`shutdown`](Self::shutdown) for a clean final flush.
pub struct RegistryBackground {
    shutdown: Arc<tokio::sync::Notify>,
    handle: Option<tokio::task::JoinHandle<()>>,
    /// Ends the thread relaying registry applies.
    applied_stop: AppliedStop,
    relay_thread: Option<std::thread::JoinHandle<()>>,
    config: BackgroundConfig,
}

impl RegistryBackground {
    /// The cadences this service runs with.
    pub fn config(&self) -> BackgroundConfig {
        self.config
    }

    /// Stop the service after a final heartbeat flush, awaiting the task.
    pub async fn shutdown(mut self) {
        self.shutdown.notify_one();
        self.applied_stop.stop();
        let relay = self.relay_thread.take();
        if let Some(relay) = relay {
            // The stop has ended its wait; the join is immediate.
            let _ = tokio::task::spawn_blocking(move || relay.join()).await;
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for RegistryBackground {
    fn drop(&mut self) {
        // A parked relay thread would outlive the service otherwise.
        self.applied_stop.stop();
    }
}

impl SeqnoConsumerRegistry for ShardConsumerRegistry {
    fn register(&self, reg: ConsumerRegistration) -> Result<RegisteredHandle, RegistryError> {
        if reg.consumer_id.is_empty() {
            return Err(RegistryError::EmptyConsumerId);
        }
        self.core.check_scope(&reg.scope)?;

        let entry = RegistryEntry {
            consumer_id: reg.consumer_id.clone(),
            kind: reg.kind,
            scope: reg.scope.clone(),
            scope_origin: reg.scope,
            checkpoint_seqno: self.core.resolve_initial(reg.initial_seqno),
            last_heartbeat_ts_ms: self.core.clock.now_ms(),
            ttl_ms: reg.ttl_ms,
        };
        self.core
            .propose(vec![RegistryCore::put_mutation(&entry)?])?;
        self.core.recompute_floor()?;
        Ok(RegisteredHandle::new(entry.consumer_id))
    }

    fn checkpoint(&self, handle: &RegisteredHandle, seqno: u64) -> Result<(), RegistryError> {
        let mut entry = self
            .core
            .read_entry(handle.consumer_id())?
            .ok_or_else(|| RegistryError::UnknownConsumer(handle.consumer_id().to_string()))?;
        // Checkpoints only advance — a stale retry must not rewind retention.
        entry.checkpoint_seqno = entry.checkpoint_seqno.max(seqno);
        entry.last_heartbeat_ts_ms = self.core.clock.now_ms();
        self.core
            .propose(vec![RegistryCore::put_mutation(&entry)?])?;
        self.core.recompute_floor()?;
        Ok(())
    }

    fn heartbeat(&self, handle: &RegisteredHandle) -> Result<(), RegistryError> {
        let now = self.core.clock.now_ms();
        if self.core.batching_on.load(Ordering::Acquire) {
            // Buffer; the background drain coalesces into one proposal.
            // Validation is deferred — a vanished consumer is skipped at flush.
            self.core
                .pending_hb
                .lock()
                .entry(handle.consumer_id().to_string())
                .and_modify(|t| *t = (*t).max(now))
                .or_insert(now);
            self.core.hb_buffered.notify_one();
            return Ok(());
        }
        // Eager path (no background service): validate + write immediately.
        let mut entry = self
            .core
            .read_entry(handle.consumer_id())?
            .ok_or_else(|| RegistryError::UnknownConsumer(handle.consumer_id().to_string()))?;
        entry.last_heartbeat_ts_ms = now;
        self.core.propose(vec![RegistryCore::put_mutation(&entry)?])
    }

    fn unregister(&self, handle: RegisteredHandle) -> Result<(), RegistryError> {
        if self.core.read_entry(handle.consumer_id())?.is_none() {
            return Err(RegistryError::UnknownConsumer(
                handle.consumer_id().to_string(),
            ));
        }
        self.core.propose(vec![Mutation::Delete {
            partition: PartitionId::Registry,
            key: encode_registry_key(handle.consumer_id()),
        }])?;
        self.core.recompute_floor()?;
        Ok(())
    }

    fn shard_floor(&self) -> u64 {
        self.core.floor.load(Ordering::Acquire)
    }

    fn list_consumers(&self) -> Vec<ConsumerSnapshot> {
        let now = self.core.clock.now_ms();
        // Per-consumer lag gauges are emitted here (ops-pull) rather than on
        // the hot path, so the high-cardinality `consumer_id` label is only
        // produced when an operator inspects the registry.
        let current_seqno = self.core.engine.snapshot();
        self.core
            .scan_entries(now)
            .unwrap_or_default()
            .into_iter()
            .map(|e| {
                let lag_seqno = if e.kind.is_seqno_space() {
                    current_seqno.saturating_sub(e.checkpoint_seqno)
                } else {
                    0
                };
                metrics::gauge!("registry_consumer_lag_seqno", "consumer_id" => e.consumer_id.clone())
                    .set(lag_seqno as f64);
                metrics::gauge!("registry_consumer_lag_age_seconds", "consumer_id" => e.consumer_id.clone())
                    .set(now.saturating_sub(e.last_heartbeat_ts_ms) as f64 / 1_000.0);
                ConsumerSnapshot {
                    consumer_id: e.consumer_id,
                    kind: e.kind,
                    scope: e.scope,
                    scope_origin: e.scope_origin,
                    checkpoint_seqno: e.checkpoint_seqno,
                    last_heartbeat_ts_ms: e.last_heartbeat_ts_ms,
                    ttl_ms: e.ttl_ms,
                }
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
