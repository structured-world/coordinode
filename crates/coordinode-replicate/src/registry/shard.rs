//! [`ShardConsumerRegistry`]: the Raft-replicated, per-shard implementation of
//! [`SeqnoConsumerRegistry`].
//!
//! Register / checkpoint / unregister are eager `RaftProposal`s into
//! `Partition::Registry`; the shard's retention floor is `min(checkpoint_seqno)`
//! over the live (non-expired) records, cached in an `Arc<AtomicU64>` so the
//! downstream feeds read it without re-scanning.
//!
//! Heartbeats and eviction run through an optional background service
//! ([`RegistryBackground`], S4b): heartbeats buffer in-memory on the leader and
//! flush as a single coalesced proposal every `heartbeat_window_ms`
//! (≤ `1000 / window` proposals/sec regardless of consumer count); expired
//! registrations are swept and removed by an eviction proposal on a timer.
//! Without the background service, `heartbeat` writes eagerly (used by tests).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;
use parking_lot::Mutex;

use super::entry::{encode_registry_key, RegistryEntry, REGISTRY_KEY_PREFIX};
use super::types::{
    ConsumerRegistration, ConsumerSnapshot, InitialSeqno, RegisteredHandle, RegistryError,
    TopologyScope,
};
use super::SeqnoConsumerRegistry;

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
    /// Drain window for buffered heartbeats. One coalesced proposal per
    /// window regardless of consumer count (ADR-028 S4b).
    pub heartbeat_window_ms: u64,
    /// How often to sweep and evict registrations past their TTL.
    pub eviction_interval_ms: u64,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self {
            heartbeat_window_ms: 100,
            eviction_interval_ms: 1_000,
        }
    }
}

/// Default MVCC time-travel retention window: 7 days, in microseconds.
/// Matches the documented MVCC retention window (default 7 days).
/// Our `commit_ts`/seqno is an HLC in wall-clock microseconds (ADR-007), so
/// `now_seqno - this` is the window floor directly in seqno space.
const DEFAULT_RETENTION_WINDOW_US: u64 = 7 * 24 * 3_600 * 1_000_000;

/// Shared registry state — held by the facade and the background tasks.
struct RegistryCore {
    engine: Arc<StorageEngine>,
    pipeline: Arc<dyn ProposalPipeline>,
    id_gen: Arc<ProposalIdGenerator>,
    clock: Arc<dyn Clock>,
    /// Cached `min(checkpoint_seqno)` over **MVCC-seqno-space** consumers
    /// (`LsmStateDelta` / `MvccSnapshotPin` / `Ephemeral`); `u64::MAX` when
    /// none. Drives the LSM GC watermark (feed a) combined with the
    /// time-travel window.
    floor: Arc<AtomicU64>,
    /// Cached `min(checkpoint)` over **oplog-index-space** consumers
    /// (`OplogEvents`, whose `ResumeToken` is a Raft log index); `u64::MAX`
    /// when none. Drives oplog segment retention (feed b): a segment is kept
    /// iff `last_index >= this` OR it is within the time window.
    oplog_index_floor: Arc<AtomicU64>,
    /// MVCC time-travel retention window in microseconds. The engine GC
    /// watermark is held back to at least `now_seqno - retention_window_us`
    /// so `AS OF TIMESTAMP` within the window always resolves, independent of
    /// consumers (bitemporal system axis, ADR-027).
    retention_window_us: u64,
    /// Whether `dc` / `rack` scopes are accepted (EE multi-DC topologies).
    allow_topology_scopes: bool,
    /// `true` once a background service is running: `heartbeat` then buffers
    /// into `pending_hb` instead of writing eagerly.
    batching_on: AtomicBool,
    /// Buffered heartbeats awaiting the next coalesced flush:
    /// `consumer_id → latest heartbeat ts`.
    pending_hb: Mutex<HashMap<String, u64>>,
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

        // Feed (a): GC watermark = min(seqno_floor, time-travel window).
        // No seqno consumers → seqno_floor == u64::MAX → the window dominates,
        // so the engine keeps `AS OF TIMESTAMP` history for the whole window
        // and never collapses to "GC everything". A consumer lagging beyond
        // the window lowers the floor below it, extending retention for that
        // consumer (CockroachDB protected-timestamp / TiDB service-safe-point).
        let time_window_floor = self
            .engine
            .snapshot()
            .saturating_sub(self.retention_window_us);
        self.engine
            .set_consumer_retention_floor(seqno_floor.min(time_window_floor));
        Ok(seqno_floor)
    }

    /// Collect every live (non-expired at `now_ms`) registration.
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
            if !entry.is_expired(now_ms) {
                out.push(entry);
            }
        }
        Ok(out)
    }

    /// Drain the buffered heartbeats into one coalesced proposal (S4b).
    /// Consumers that vanished since buffering are silently skipped.
    fn flush_pending_heartbeats(&self) -> Result<(), RegistryError> {
        let drained: Vec<(String, u64)> = {
            let mut pending = self.pending_hb.lock();
            if pending.is_empty() {
                return Ok(());
            }
            pending.drain().collect()
        };
        let mut mutations = Vec::with_capacity(drained.len());
        for (consumer_id, ts) in drained {
            if let Some(mut entry) = self.read_entry(&consumer_id)? {
                entry.last_heartbeat_ts_ms = entry.last_heartbeat_ts_ms.max(ts);
                mutations.push(Self::put_mutation(&entry)?);
            }
        }
        // S4b observability: one coalesced proposal per drained window.
        metrics::counter!("registry_heartbeat_batches_total").increment(1);
        metrics::histogram!("registry_heartbeat_batch_size").record(mutations.len() as f64);
        self.propose(mutations)
    }

    /// Evict registrations past their TTL via one Delete proposal, then
    /// refresh the floor (a dead consumer must stop pinning retention).
    fn sweep_evictions(&self) -> Result<usize, RegistryError> {
        let now = self.clock.now_ms();
        let mut to_evict = Vec::new();
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
            if entry.is_expired(now) {
                to_evict.push(Mutation::Delete {
                    partition: PartitionId::Registry,
                    key: encode_registry_key(&entry.consumer_id),
                });
            }
        }
        let count = to_evict.len();
        if count > 0 {
            self.propose(to_evict)?;
            metrics::counter!("registry_evictions_total").increment(count as u64);
        }
        // Always refresh the floor — even with no eviction the time-travel
        // window floor (`now_seqno - retention_window`) advances with the wall
        // clock, so the published GC watermark must move forward each sweep
        // (otherwise it freezes at its construction-time value).
        self.recompute_floor()?;
        Ok(count)
    }
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
            retention_window_us: DEFAULT_RETENTION_WINDOW_US,
            allow_topology_scopes: false,
            batching_on: AtomicBool::new(false),
            pending_hb: Mutex::new(HashMap::new()),
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

    /// Override the MVCC time-travel retention window (default 7 days). Mainly
    /// for tests + deployments tuning the `AS OF TIMESTAMP` horizon. Must be
    /// called before the registry is cloned / a background task is spawned.
    pub fn with_retention_window_us(mut self, window_us: u64) -> Self {
        if let Some(core) = Arc::get_mut(&mut self.core) {
            core.retention_window_us = window_us;
            let _ = core.recompute_floor();
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
    /// no CDC consumer). Raft-index space, distinct from [`shard_floor`].
    pub fn oplog_retention_floor(&self) -> u64 {
        self.core.oplog_index_floor.load(Ordering::Acquire)
    }

    /// Lagging-consumer guard (ADR-028): verify the consumer's checkpoint has
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
        let handle = tokio::spawn(async move {
            let mut hb = tokio::time::interval(Duration::from_millis(cfg.heartbeat_window_ms));
            let mut evict = tokio::time::interval(Duration::from_millis(cfg.eviction_interval_ms));
            loop {
                tokio::select! {
                    _ = stop.notified() => break,
                    _ = hb.tick() => {
                        if let Err(e) = core.flush_pending_heartbeats() {
                            tracing::warn!(error = %e, "registry heartbeat flush failed");
                        }
                    }
                    _ = evict.tick() => {
                        match core.sweep_evictions() {
                            Ok(n) if n > 0 => tracing::debug!(evicted = n, "registry TTL sweep"),
                            Ok(_) => {}
                            Err(e) => tracing::warn!(error = %e, "registry eviction sweep failed"),
                        }
                    }
                }
            }
            // Final flush so no buffered heartbeat is lost on graceful stop.
            let _ = core.flush_pending_heartbeats();
        });
        RegistryBackground { shutdown, handle }
    }
}

/// Handle to the running background service. Drop detaches the task; prefer
/// [`shutdown`](Self::shutdown) for a clean final flush.
pub struct RegistryBackground {
    shutdown: Arc<tokio::sync::Notify>,
    handle: tokio::task::JoinHandle<()>,
}

impl RegistryBackground {
    /// Stop the service after a final heartbeat flush, awaiting the task.
    pub async fn shutdown(self) {
        self.shutdown.notify_one();
        let _ = self.handle.await;
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
            // S4b: buffer; the background drain coalesces into one proposal.
            // Validation is deferred — a vanished consumer is skipped at flush.
            self.core
                .pending_hb
                .lock()
                .entry(handle.consumer_id().to_string())
                .and_modify(|t| *t = (*t).max(now))
                .or_insert(now);
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
