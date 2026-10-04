//! [`ShardConsumerRegistry`]: the replicated, per-shard implementation of
//! [`SeqnoConsumerRegistry`].
//!
//! Every transition of a registration (register, checkpoint, heartbeat,
//! cancel, a BOUNDED bound crossed) is one transaction over its record in
//! `Partition::Registry`, conditioned on the version of the record it read.
//! Two transitions racing over one record cannot both land: the second finds
//! the version moved, reads again and decides against what is there now. An
//! acknowledgement racing an expiry therefore either advances the live
//! incarnation before the expiry is judged, or finds it ended.
//!
//! An ended registration keeps its record. The incarnation it carries is
//! what refuses a handle of the ended registration, a heartbeat delayed past
//! the end, and numbers the next registration of the same id.
//!
//! The retention floors (MVCC seqno space, oplog index space) are the
//! minimum checkpoint over live registrations, cached so the downstream
//! feeds read them without scanning.
//!
//! Heartbeats buffer on the node and flush as one coalesced transaction per
//! window; BOUNDED bounds are judged by a background sweep that runs when the
//! earliest bound can be crossed and when the registry keyspace changes.
//!
//! A new registration is refused while the source cannot take on more
//! retention (the server's source says so under Stop write pressure). Every
//! other transition passes the write-pressure gate, since it releases or
//! records retention rather than taking on more.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::Guard;
use coordinode_storage::engine::applied::AppliedStop;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};
use parking_lot::Mutex;

use super::SeqnoConsumerRegistry;
use super::entry::{LEGACY_KEY_PREFIX, REGISTRY_KEY_PREFIX, RegistryEntry, encode_registry_key};
use super::source::RetentionSource;
use super::types::{
    ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy, ConsumerSnapshot, InitialSeqno,
    RegisteredHandle, RegistrationState, RegistryError, TerminalReason, TopologyScope,
};
use super::watch::{RegistrationWatch, Watches};
use coordinode_storage::engine::applied::AppliedEvent;

/// Attempts at one transition before contention is reported. Each attempt
/// reads afresh, so only writers landing between every read and its commit
/// exhaust them.
const TRANSITION_ATTEMPTS: u32 = 16;

/// Pause before attempt `attempt` (from 1): a lost attempt usually met a
/// commit to the same record still in flight, which lands within a write's
/// durability; retrying at once would only meet it again.
fn backoff(attempt: u32) -> Duration {
    Duration::from_millis(1u64 << attempt.min(6)).min(Duration::from_millis(50))
}

/// Wall-clock source for heartbeat and progress-age accounting.
///
/// Injected (not `std::time::SystemTime` hard-wired) so tests drive time
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

/// Tuning for the background heartbeat-flush and bound-sweep service.
#[derive(Debug, Clone, Copy)]
pub struct BackgroundConfig {
    /// Drain window for buffered heartbeats: the first heartbeat buffered
    /// opens a window, and the heartbeats of the window go in one transaction
    /// regardless of consumer count. Nothing buffered, nothing runs.
    pub heartbeat_window_ms: u64,
    /// Shortest gap between two sweeps. A sweep runs when the earliest bound
    /// can be crossed and when the registry keyspace changes; while a BOUNDED
    /// consumer is behind its source, at this cadence, since the source
    /// growing is what moves its retained bytes.
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

/// What a transition decided for one record.
enum Step<T> {
    /// Write this record, conditioned on the version read, then answer `T`.
    Write(RegistryEntry, T),
    /// Nothing to write; answer `T`.
    Keep(T),
}

/// Shared registry state, held by the facade and the background tasks.
struct RegistryCore {
    engine: Arc<StorageEngine>,
    pipeline: Arc<dyn ProposalPipeline>,
    id_gen: Arc<ProposalIdGenerator>,
    clock: Arc<dyn Clock>,
    source: Arc<dyn RetentionSource>,
    /// Cached `min(checkpoint_seqno)` over live **MVCC-seqno-space**
    /// registrations; `u64::MAX` when none. Published to the engine as the
    /// consumer retention floor, which the engine combines by `min` with its
    /// own time-travel window and live snapshot pins.
    floor: Arc<AtomicU64>,
    /// Cached `min(checkpoint)` over live **oplog-index-space** registrations;
    /// `u64::MAX` when none.
    oplog_index_floor: Arc<AtomicU64>,
    /// Whether `dc` / `rack` scopes are accepted (EE multi-DC topologies).
    allow_topology_scopes: bool,
    /// `true` once a background service is running: `heartbeat` then buffers
    /// into `pending_hb` instead of writing at once.
    batching_on: AtomicBool,
    /// Buffered heartbeats awaiting the next coalesced flush, by consumer id
    /// and the incarnation that sent them: a heartbeat of an ended
    /// incarnation must not keep its successor alive.
    pending_hb: Mutex<HashMap<(String, u64), u64>>,
    /// Notified when a heartbeat is buffered, opening a flush window.
    hb_buffered: tokio::sync::Notify,
    /// Readers' notices of writes to the records they watch.
    watches: Arc<Watches>,
    /// Runs once between a transition's read and its commit: what another
    /// writer does in that window.
    #[cfg(test)]
    before_commit: Mutex<Option<Box<dyn FnOnce() + Send>>>,
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

    /// Where a new registration's checkpoint starts, refused when the source
    /// no longer holds that position.
    fn resolve_initial(&self, reg: &ConsumerRegistration) -> Result<u64, RegistryError> {
        let first = self.source.first_retained(reg.kind);
        match reg.initial_seqno {
            InitialSeqno::FromNow => Ok(self.source.head(reg.kind)),
            InitialSeqno::FromEarliestRetained => Ok(first),
            InitialSeqno::At(at) if at < first => Err(RegistryError::RetentionLost {
                checkpoint: at,
                floor: first,
            }),
            InitialSeqno::At(at) => Ok(at),
        }
    }

    /// The record of `consumer_id` and the version it carries.
    fn read_entry(
        &self,
        consumer_id: &str,
    ) -> Result<Option<(RegistryEntry, Option<u64>)>, RegistryError> {
        let key = encode_registry_key(consumer_id);
        let version = self
            .engine
            .record_version(Partition::Registry, &key)
            .map_err(|e| RegistryError::Replication(e.to_string()))?;
        let raw = self
            .engine
            .get(Partition::Registry, &key)
            .map_err(|e| RegistryError::Replication(e.to_string()))?;
        match raw {
            Some(bytes) => RegistryEntry::decode(&bytes)
                .map(|entry| Some((entry, version)))
                .map_err(|e| RegistryError::Replication(format!("decode registry entry: {e}"))),
            None => Ok(None),
        }
    }

    /// Commit `writes` in one transaction, each conditioned on the version its
    /// record had when read (`None` for a record that did not exist). `Ok(false)`
    /// when another writer moved one of them: the caller reads again.
    fn commit_writes(
        &self,
        writes: &[(RegistryEntry, Option<u64>)],
        deletes: &[Vec<u8>],
    ) -> Result<bool, RegistryError> {
        if writes.is_empty() && deletes.is_empty() {
            return Ok(true);
        }
        let oracle = self.engine.oracle().ok_or_else(|| {
            RegistryError::Replication(
                "the registry needs an engine with a timestamp oracle".into(),
            )
        })?;
        let mut txn = Transaction::begin(&self.engine, Some(oracle.as_ref()), oracle.next());
        // Registry transitions release or record retention; holding them back
        // under storage pressure would keep the bytes that relieve it.
        txn.exempt_from_write_pressure();
        for (entry, version) in writes {
            let key = encode_registry_key(&entry.consumer_id);
            let value = entry
                .encode()
                .map_err(|e| RegistryError::Replication(format!("encode registry entry: {e}")))?;
            txn.expect_version(Partition::Registry, &key, *version)
                .map_err(|e| RegistryError::Replication(e.to_string()))?;
            txn.put(Partition::Registry, &key, &value)
                .map_err(|e| RegistryError::Replication(e.to_string()))?;
        }
        for key in deletes {
            txn.delete(Partition::Registry, key)
                .map_err(|e| RegistryError::Replication(e.to_string()))?;
        }
        let write_concern = WriteConcern::default();
        let ctx = CommitContext {
            write_concern: &write_concern,
            pipeline: Some(self.pipeline.as_ref()),
            id_gen: Some(&self.id_gen),
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        match txn.commit(&ctx) {
            Ok(_) => Ok(true),
            Err(CommitError::RevisionMismatch { .. } | CommitError::Conflict(_)) => Ok(false),
            Err(CommitError::NotLeader { leader_id }) => {
                Err(RegistryError::NotLeader { leader_id })
            }
            Err(e) => Err(RegistryError::Replication(e.to_string())),
        }
    }

    /// Run `decide` against the current record of `consumer_id` and commit
    /// what it chose, reading again whenever another writer got there first.
    fn transition<T>(
        &self,
        consumer_id: &str,
        mut decide: impl FnMut(Option<RegistryEntry>) -> Result<Step<T>, RegistryError>,
    ) -> Result<T, RegistryError> {
        for attempt in 0..TRANSITION_ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(backoff(attempt));
            }
            let (current, version) = match self.read_entry(consumer_id)? {
                Some((entry, version)) => (Some(entry), version),
                None => (None, None),
            };
            // What this record holds the floors at now, if anything.
            let held = current
                .as_ref()
                .filter(|entry| entry.is_live())
                .map(|entry| (entry.kind.is_seqno_space(), entry.checkpoint_seqno));
            match decide(current)? {
                Step::Keep(answer) => return Ok(answer),
                Step::Write(entry, answer) => {
                    #[cfg(test)]
                    {
                        // Taken first: the hook may run a transition itself.
                        let hook = self.before_commit.lock().take();
                        if let Some(hook) = hook {
                            hook();
                        }
                    }
                    // A write that can lower a floor publishes it before
                    // returning: history the new position needs must not be
                    // collected meanwhile. One that can only raise it (an
                    // acknowledgement, an end) leaves that to the background
                    // sweep its applied write triggers; a floor published
                    // late only keeps history longer. Without the background
                    // service nothing else would publish it.
                    let lowers = entry.is_live()
                        && held.is_none_or(|(seqno_space, checkpoint)| {
                            seqno_space != entry.kind.is_seqno_space()
                                || entry.checkpoint_seqno < checkpoint
                        });
                    if self.commit_writes(&[(entry, version)], &[])? {
                        if lowers || !self.batching_on.load(Ordering::Acquire) {
                            self.recompute_floor()?;
                        }
                        return Ok(answer);
                    }
                }
            }
        }
        Err(RegistryError::Replication(format!(
            "registration {consumer_id:?} kept changing under {TRANSITION_ATTEMPTS} attempts"
        )))
    }

    /// The live record a handle addresses, or why the handle is refused.
    fn live_for(
        entry: Option<RegistryEntry>,
        handle: &RegisteredHandle,
    ) -> Result<RegistryEntry, RegistryError> {
        let entry = entry
            .ok_or_else(|| RegistryError::UnknownConsumer(handle.consumer_id().to_string()))?;
        if entry.incarnation != handle.incarnation() {
            return Err(RegistryError::StaleIncarnation {
                consumer_id: entry.consumer_id,
                handle: handle.incarnation(),
                current: entry.incarnation,
            });
        }
        if let RegistrationState::Terminated { reason, .. } = entry.state {
            return Err(RegistryError::Terminated {
                consumer_id: entry.consumer_id,
                incarnation: entry.incarnation,
                reason,
                checkpoint: entry.checkpoint_seqno,
            });
        }
        Ok(entry)
    }

    /// Re-scan the keyspace, refresh the two space-split floors over live
    /// registrations, and publish the engine GC watermark. Returns the
    /// seqno-space floor.
    fn recompute_floor(&self) -> Result<u64, RegistryError> {
        let mut seqno_floor = u64::MAX;
        let mut oplog_floor = u64::MAX;
        let mut counts = [0u64; ConsumerKind::ALL.len()];
        for entry in self.scan_entries()? {
            if !entry.is_live() {
                continue;
            }
            if let Some(slot) = ConsumerKind::ALL.iter().position(|k| *k == entry.kind) {
                counts[slot] += 1;
            }
            if entry.kind.is_seqno_space() {
                seqno_floor = seqno_floor.min(entry.checkpoint_seqno);
            } else {
                oplog_floor = oplog_floor.min(entry.checkpoint_seqno);
            }
        }
        self.floor.store(seqno_floor, Ordering::Release);
        self.oplog_index_floor.store(oplog_floor, Ordering::Release);
        for (kind, count) in ConsumerKind::ALL.iter().zip(counts) {
            metrics::gauge!("registry_consumer_count", "kind" => kind.label()).set(count as f64);
        }
        metrics::gauge!("registry_shard_floor_seqno").set(seqno_floor as f64);
        metrics::gauge!("registry_oplog_floor_index").set(oplog_floor as f64);
        // MVCC history the consumers hold below the newest commit. No
        // consumer, or a floor at or past the newest commit, pins nothing:
        // the clamp at zero is that meaning, not a masked overflow.
        let pinned = match seqno_floor {
            u64::MAX => 0,
            floor => self.engine.snapshot().saturating_sub(floor),
        };
        metrics::gauge!("registry_gc_pinned_seqno_count").set(pinned as f64);
        // The engine combines this by `min` with its own time-travel window
        // and live snapshot pins: a consumer can only extend retention.
        self.engine.set_consumer_retention_floor(seqno_floor);
        Ok(seqno_floor)
    }

    /// Every registration record, live or ended.
    fn scan_entries(&self) -> Result<Vec<RegistryEntry>, RegistryError> {
        let mut out = Vec::new();
        let iter = self
            .engine
            .prefix_scan(Partition::Registry, REGISTRY_KEY_PREFIX)
            .map_err(|e| RegistryError::Replication(e.to_string()))?;
        for guard in iter {
            let (_, value) = guard
                .into_inner()
                .map_err(|e| RegistryError::Replication(e.to_string()))?;
            out.push(
                RegistryEntry::decode(&value).map_err(|e| {
                    RegistryError::Replication(format!("decode registry entry: {e}"))
                })?,
            );
        }
        Ok(out)
    }

    /// Keys of records in the format that predates retention policies.
    fn legacy_keys(&self) -> Result<Vec<Vec<u8>>, RegistryError> {
        let mut out = Vec::new();
        let iter = self
            .engine
            .prefix_scan(Partition::Registry, LEGACY_KEY_PREFIX)
            .map_err(|e| RegistryError::Replication(e.to_string()))?;
        for guard in iter {
            let (key, _) = guard
                .into_inner()
                .map_err(|e| RegistryError::Replication(e.to_string()))?;
            out.push(key.to_vec());
        }
        Ok(out)
    }

    /// Write the buffered heartbeats in one coalesced transaction.
    ///
    /// A heartbeat of an incarnation that has since ended or been replaced is
    /// dropped: it is no sign of life of anything registered now. Each
    /// heartbeat stays in the buffer until its write lands, so a sweep judging
    /// liveness meanwhile still sees it.
    fn flush_pending_heartbeats(&self) -> Result<(), RegistryError> {
        for attempt in 0..TRANSITION_ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(backoff(attempt));
            }
            let taken: Vec<((String, u64), u64)> = {
                let pending = self.pending_hb.lock();
                if pending.is_empty() {
                    return Ok(());
                }
                pending.iter().map(|(k, ts)| (k.clone(), *ts)).collect()
            };
            let mut writes = Vec::with_capacity(taken.len());
            let mut stale = Vec::new();
            for ((consumer_id, incarnation), ts) in &taken {
                match self.read_entry(consumer_id)? {
                    Some((mut entry, version))
                        if entry.incarnation == *incarnation && entry.is_live() =>
                    {
                        entry.last_heartbeat_ts_ms = entry.last_heartbeat_ts_ms.max(*ts);
                        writes.push((entry, version));
                    }
                    _ => stale.push((consumer_id.clone(), *incarnation)),
                }
            }
            match self.commit_writes(&writes, &[]) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(RegistryError::NotLeader { leader_id }) => {
                    // Only the leader records a heartbeat, and a stream on this
                    // member ends at its next acknowledgement, which is refused
                    // the same way: keeping these would only retry for nothing.
                    let mut pending = self.pending_hb.lock();
                    for (key, ts) in taken {
                        if pending.get(&key) == Some(&ts) {
                            pending.remove(&key);
                        }
                    }
                    tracing::debug!(?leader_id, "registry: heartbeats dropped on a follower");
                    return Ok(());
                }
                Err(e) => return Err(e),
            }
            metrics::counter!("registry_heartbeat_batches_total").increment(1);
            metrics::histogram!("registry_heartbeat_batch_size").record(writes.len() as f64);
            let mut pending = self.pending_hb.lock();
            for (key, ts) in taken {
                // A heartbeat buffered during the write is newer and stays.
                if pending.get(&key) == Some(&ts) {
                    pending.remove(&key);
                }
            }
            for key in stale {
                pending.remove(&key);
            }
            return Ok(());
        }
        Err(RegistryError::Replication(format!(
            "heartbeat flush kept conflicting under {TRANSITION_ATTEMPTS} attempts"
        )))
    }

    /// End every live BOUNDED registration whose bound is crossed, durably and
    /// one record at a time, delete records of the old format, then refresh
    /// the floors.
    fn sweep_evictions(&self) -> Result<Sweep, RegistryError> {
        // A buffered heartbeat is a sign of life the stored record does not
        // show yet: persist it before judging liveness.
        self.flush_pending_heartbeats()?;
        // A member that is not the leader decides no transition; it keeps the
        // deadlines so its sweep acts once it leads, and refreshes its floors.
        let mut leads = true;
        let legacy = self.legacy_keys()?;
        if !legacy.is_empty() {
            match self.commit_writes(&[], &legacy) {
                Err(RegistryError::NotLeader { .. }) => leads = false,
                other => {
                    other?;
                }
            }
        }
        let mut evicted = 0usize;
        let mut next_deadline_ms: Option<u64> = None;
        let mut behind = false;
        for entry in self.scan_entries()? {
            if !entry.is_live() {
                continue;
            }
            let consumer_id = entry.consumer_id.clone();
            let incarnation = entry.incarnation;
            if !leads {
                if let Some(at) = entry.next_deadline_ms(self.source.as_ref()) {
                    next_deadline_ms = Some(next_deadline_ms.map_or(at, |n| n.min(at)));
                }
                continue;
            }
            let ended = match self.transition(&consumer_id, |current| {
                let Some(mut current) = current else {
                    return Ok(Step::Keep(None));
                };
                if current.incarnation != incarnation || !current.is_live() {
                    return Ok(Step::Keep(None));
                }
                // A heartbeat buffered while the flush above was writing is
                // only in the buffer, and it is a sign of life.
                if let Some(ts) = self
                    .pending_hb
                    .lock()
                    .get(&(consumer_id.clone(), incarnation))
                {
                    current.last_heartbeat_ts_ms = current.last_heartbeat_ts_ms.max(*ts);
                }
                let now = self.clock.now_ms();
                match current.termination(now, self.source.as_ref()) {
                    Some(reason) => {
                        let checkpoint = current.checkpoint_seqno;
                        current.state = RegistrationState::Terminated { reason, at_ms: now };
                        Ok(Step::Write(current, Some((reason, checkpoint))))
                    }
                    None => Ok(Step::Keep(None)),
                }
            }) {
                Ok(ended) => ended,
                Err(RegistryError::NotLeader { .. }) => {
                    leads = false;
                    None
                }
                Err(e) => return Err(e),
            };
            match ended {
                Some((reason, checkpoint)) => {
                    evicted += 1;
                    tracing::info!(
                        consumer_id,
                        incarnation,
                        ?reason,
                        checkpoint,
                        "registry: bounded consumer terminated"
                    );
                }
                None => {
                    if let Some(at) = entry.next_deadline_ms(self.source.as_ref()) {
                        next_deadline_ms = Some(next_deadline_ms.map_or(at, |n| n.min(at)));
                    }
                    if leads
                        && matches!(entry.retention, ConsumerRetentionPolicy::Bounded(_))
                        && entry.checkpoint_seqno < self.source.head(entry.kind)
                    {
                        behind = true;
                    }
                }
            }
        }
        if evicted > 0 {
            metrics::counter!("registry_evictions_total").increment(evicted as u64);
        }
        // Always refresh: a registration ended or advanced through another
        // member must reach the engine even when this sweep ended nothing.
        self.recompute_floor()?;
        Ok(Sweep {
            evicted,
            next_deadline_ms,
            behind,
        })
    }
}

/// What one sweep did and when the next one is due.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sweep {
    /// Registrations ended.
    evicted: usize,
    /// When the earliest live bound can be crossed with nothing else
    /// happening (clock ms), or `None` when none can.
    next_deadline_ms: Option<u64>,
    /// Whether a live BOUNDED consumer is behind its source, whose growth
    /// moves the bytes it requires.
    behind: bool,
}

/// Per-shard consumer-retention registry backed by `Partition::Registry`.
///
/// Construct with [`ShardConsumerRegistry::new`] (CE: `cluster` / `node` /
/// `shard` scopes): `dc` / `rack` scopes require a multi-DC topology and are
/// rejected unless enabled via [`with_topology_scopes`](Self::with_topology_scopes)
/// (EE). Call [`start_background`](Self::start_background) to run batched
/// heartbeats and the bound sweep.
#[derive(Clone)]
pub struct ShardConsumerRegistry {
    core: Arc<RegistryCore>,
}

impl ShardConsumerRegistry {
    /// Open a CE registry over the given engine, proposal pipeline and the
    /// source its consumers read.
    pub fn new(
        engine: Arc<StorageEngine>,
        pipeline: Arc<dyn ProposalPipeline>,
        id_gen: Arc<ProposalIdGenerator>,
        clock: Arc<dyn Clock>,
        source: Arc<dyn RetentionSource>,
    ) -> Self {
        let core = Arc::new(RegistryCore {
            engine,
            pipeline,
            id_gen,
            clock,
            source,
            floor: Arc::new(AtomicU64::new(u64::MAX)),
            oplog_index_floor: Arc::new(AtomicU64::new(u64::MAX)),
            allow_topology_scopes: false,
            batching_on: AtomicBool::new(false),
            pending_hb: Mutex::new(HashMap::new()),
            hb_buffered: tokio::sync::Notify::new(),
            watches: Arc::new(Watches::default()),
            #[cfg(test)]
            before_commit: Mutex::new(None),
        });
        // Recover the floors and publish the engine watermark from any
        // registrations persisted in a prior life. A failure here leaves the
        // floors unconstrained until the first sweep, which retries it.
        if let Err(e) = core.recompute_floor() {
            tracing::warn!(error = %e, "registry: floors not recovered at open");
        }
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

    /// A cheaply-cloned handle to the cached MVCC-seqno floor. The engine
    /// watermark itself is published via `set_consumer_retention_floor`.
    pub fn floor_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.core.floor)
    }

    /// The oplog-index retention floor: `min(checkpoint)` over live
    /// `OplogEvents` registrations, or `u64::MAX` when none. Raft-index space,
    /// distinct from [`shard_floor`](SeqnoConsumerRegistry::shard_floor).
    pub fn oplog_retention_floor(&self) -> u64 {
        self.core.oplog_index_floor.load(Ordering::Acquire)
    }

    /// Take up a live registration again by its id and incarnation, as a
    /// consumer reconnecting after a disconnect does.
    ///
    /// # Errors
    ///
    /// [`RegistryError::UnknownConsumer`] when the id was never registered,
    /// [`RegistryError::StaleIncarnation`] when another incarnation holds it,
    /// [`RegistryError::Terminated`] when the incarnation has ended.
    pub fn resume(
        &self,
        consumer_id: &str,
        incarnation: u64,
    ) -> Result<RegisteredHandle, RegistryError> {
        let handle = RegisteredHandle::new(consumer_id, incarnation);
        let entry = self.core.read_entry(consumer_id)?.map(|(entry, _)| entry);
        RegistryCore::live_for(entry, &handle)?;
        Ok(handle)
    }

    /// Verify the source still holds what a live registration's checkpoint
    /// needs, and return that checkpoint. A reader calls this before reading
    /// from the checkpoint, so missing history is refused, never skipped.
    ///
    /// # Errors
    ///
    /// The handle refusals of [`Self::resume`], and
    /// [`RegistryError::RetentionLost`] when the source no longer holds the
    /// checkpoint: the MVCC store has collected below it, or the oplog has
    /// purged it.
    pub fn check_retention(&self, handle: &RegisteredHandle) -> Result<u64, RegistryError> {
        let entry = self.core.read_entry(handle.consumer_id())?.map(|(e, _)| e);
        let entry = RegistryCore::live_for(entry, handle)?;
        let floor = if entry.kind.is_seqno_space() {
            self.core.engine.gc_watermark()
        } else {
            self.core.source.first_retained(entry.kind)
        };
        if entry.checkpoint_seqno < floor {
            return Err(RegistryError::RetentionLost {
                checkpoint: entry.checkpoint_seqno,
                floor,
            });
        }
        Ok(entry.checkpoint_seqno)
    }

    /// Notices of writes to `handle`'s record, so a reader that must know
    /// whether its registration still stands calls
    /// [`check_retention`](Self::check_retention) only after one: an end, a
    /// newer incarnation, an acknowledgement. Its own reads detect history
    /// the source no longer holds.
    pub fn watch(&self, handle: &RegisteredHandle) -> RegistrationWatch {
        self.core
            .watches
            .watch(encode_registry_key(handle.consumer_id()))
    }

    /// Start the background service: coalesced heartbeat flush and the
    /// bound sweep. Switches `heartbeat` to buffered mode. Returns a handle
    /// whose [`shutdown`](RegistryBackground::shutdown) does a final flush.
    pub fn start_background(&self, cfg: BackgroundConfig) -> RegistryBackground {
        self.core.batching_on.store(true, Ordering::Release);
        let core = Arc::clone(&self.core);
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let stop = Arc::clone(&shutdown);
        // Registry writes wait on a commit, an fsync at least: they run on
        // the blocking pool, or the runtime thread they would hold is the one
        // the consumers' streams heartbeat from.
        let blocking =
            |core: &Arc<RegistryCore>,
             work: fn(&RegistryCore) -> Result<Option<Sweep>, RegistryError>| {
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
                    tracing::debug!(evicted = sweep.evicted, "registry bound sweep");
                }
                Some(sweep)
            })
        };

        // A registration another member wrote changes the floor this member
        // publishes, so a write applied to the registry keyspace asks for a
        // sweep, and tells the readers watching that record. The feed is a
        // blocking subscription; one parked thread relays it.
        let changed = Arc::new(tokio::sync::Notify::new());
        let applied = self.core.engine.subscribe_applied(Partition::Registry, 16);
        let applied_stop = applied.stopper();
        let relay = Arc::clone(&changed);
        let watches = Arc::clone(&self.core.watches);
        let relay_watches = Arc::clone(&watches);
        let relay_thread = std::thread::Builder::new()
            .name("registry-applied".to_string())
            .spawn(move || {
                while let Some(event) = applied.next(None) {
                    match event {
                        AppliedEvent::Keys { keys, .. } => relay_watches.applied(&keys),
                        AppliedEvent::Replaced => relay_watches.applied_unknown(),
                    }
                    relay.notify_one();
                }
            })
            .map_err(|e| tracing::error!(error = %e, "registry apply relay did not start"))
            .ok();
        // Subscribed before this, so no write applied from here on goes
        // unrelayed.
        watches.set_relayed(relay_thread.is_some());

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
                            Ok(Some(sweep)) => {
                                let deadline = sweep.next_deadline_ms.map(|at_ms| {
                                    let wait = at_ms.saturating_sub(core.clock.now_ms());
                                    swept_at + Duration::from_millis(wait).max(gap)
                                });
                                let periodic = sweep.behind.then_some(swept_at + gap);
                                match (deadline, periodic) {
                                    (Some(a), Some(b)) => Some(a.min(b)),
                                    (a, b) => a.or(b),
                                }
                            }
                            Ok(None) => None,
                            Err(e) => {
                                tracing::warn!(error = %e, "registry bound sweep failed");
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
            watches,
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
    /// Told the relay stopped, so watches stop relying on it.
    watches: Arc<Watches>,
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
        self.watches.set_relayed(false);
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
        self.watches.set_relayed(false);
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
        if matches!(reg.retention, ConsumerRetentionPolicy::Bounded(_))
            && !self.core.source.accounts(reg.kind)
        {
            return Err(RegistryError::InvalidRetention(format!(
                "this shard cannot measure progress age and required bytes for {:?} consumers, \
                 so a BOUNDED limit could not be judged; register it STRICT",
                reg.kind
            )));
        }
        // A registration is new retention; it is admitted only while storage
        // keeps up with reclaiming what it holds. Transitions that release or
        // record retention are never held back.
        if !self.core.source.admits(reg.kind) {
            return Err(RegistryError::Backpressure);
        }
        let checkpoint = self.core.resolve_initial(&reg)?;
        let now = self.core.clock.now_ms();
        let incarnation = self.core.transition(&reg.consumer_id, |current| {
            let incarnation = match current {
                Some(existing) if existing.is_live() => {
                    return Err(RegistryError::AlreadyRegistered {
                        consumer_id: existing.consumer_id,
                        incarnation: existing.incarnation,
                    });
                }
                Some(ended) => ended.incarnation.checked_add(1).ok_or_else(|| {
                    RegistryError::Replication(format!(
                        "consumer {:?} has used every incarnation",
                        reg.consumer_id
                    ))
                })?,
                None => 1,
            };
            let entry = RegistryEntry {
                consumer_id: reg.consumer_id.clone(),
                incarnation,
                kind: reg.kind,
                scope: reg.scope.clone(),
                scope_origin: reg.scope.clone(),
                retention: reg.retention,
                state: RegistrationState::Live,
                checkpoint_seqno: checkpoint,
                last_heartbeat_ts_ms: now,
            };
            Ok(Step::Write(entry, incarnation))
        })?;
        Ok(RegisteredHandle::new(reg.consumer_id, incarnation))
    }

    fn checkpoint(&self, handle: &RegisteredHandle, seqno: u64) -> Result<(), RegistryError> {
        let now = self.core.clock.now_ms();
        self.core.transition(handle.consumer_id(), |current| {
            let mut entry = RegistryCore::live_for(current, handle)?;
            // Checkpoints only advance: a stale retry must not rewind
            // retention. An acknowledgement is also a sign of life.
            if seqno <= entry.checkpoint_seqno && entry.last_heartbeat_ts_ms >= now {
                return Ok(Step::Keep(()));
            }
            entry.checkpoint_seqno = entry.checkpoint_seqno.max(seqno);
            entry.last_heartbeat_ts_ms = entry.last_heartbeat_ts_ms.max(now);
            Ok(Step::Write(entry, ()))
        })
    }

    fn heartbeat(&self, handle: &RegisteredHandle) -> Result<(), RegistryError> {
        let now = self.core.clock.now_ms();
        if self.core.batching_on.load(Ordering::Acquire) {
            // Buffer; the background flush coalesces into one transaction and
            // drops it there if the incarnation has ended meanwhile.
            self.core
                .pending_hb
                .lock()
                .entry((handle.consumer_id().to_string(), handle.incarnation()))
                .and_modify(|t| *t = (*t).max(now))
                .or_insert(now);
            self.core.hb_buffered.notify_one();
            return Ok(());
        }
        self.core.transition(handle.consumer_id(), |current| {
            let mut entry = RegistryCore::live_for(current, handle)?;
            entry.last_heartbeat_ts_ms = entry.last_heartbeat_ts_ms.max(now);
            Ok(Step::Write(entry, ()))
        })
    }

    fn unregister(&self, handle: RegisteredHandle) -> Result<(), RegistryError> {
        let now = self.core.clock.now_ms();
        self.core.transition(handle.consumer_id(), |current| {
            let mut entry = RegistryCore::live_for(current, &handle)?;
            entry.state = RegistrationState::Terminated {
                reason: TerminalReason::Cancelled,
                at_ms: now,
            };
            Ok(Step::Write(entry, ()))
        })
    }

    fn shard_floor(&self) -> u64 {
        self.core.floor.load(Ordering::Acquire)
    }

    fn list_consumers(&self) -> Vec<ConsumerSnapshot> {
        let now = self.core.clock.now_ms();
        let source = self.core.source.as_ref();
        // Per-consumer gauges are emitted here (ops-pull) rather than on the
        // hot path, so the high-cardinality `consumer_id` label is only
        // produced when an operator inspects the registry.
        let entries = match self.core.scan_entries() {
            Ok(entries) => entries,
            Err(e) => {
                tracing::warn!(error = %e, "registry: listing failed");
                return Vec::new();
            }
        };
        entries
            .into_iter()
            .map(|e| {
                if e.is_live() {
                    let head = source.head(e.kind);
                    let behind = e.checkpoint_seqno < head;
                    // Positions of one space: the work outstanding, zero
                    // for a consumer at or past the head.
                    metrics::gauge!(
                        "registry_consumer_lag_seqno",
                        "consumer_id" => e.consumer_id.clone(),
                        "kind" => e.kind.label()
                    )
                    .set(head.saturating_sub(e.checkpoint_seqno) as f64);
                    // Age of the oldest unacknowledged work; a consumer with
                    // none outstanding is not lagging, however long ago it
                    // last acknowledged.
                    let lag_age_ms = behind
                        .then(|| source.produced_at_ms(e.kind, e.checkpoint_seqno))
                        .flatten()
                        .map_or(0, |at| now.saturating_sub(at));
                    metrics::gauge!("registry_consumer_lag_age_seconds", "consumer_id" => e.consumer_id.clone())
                        .set(lag_age_ms as f64 / 1_000.0);
                    metrics::gauge!("registry_consumer_heartbeat_age_seconds", "consumer_id" => e.consumer_id.clone())
                        .set(now.saturating_sub(e.last_heartbeat_ts_ms) as f64 / 1_000.0);
                    if let Some(bytes) = source.retained_bytes_from(e.kind, e.checkpoint_seqno) {
                        metrics::gauge!("registry_consumer_retained_bytes", "consumer_id" => e.consumer_id.clone())
                            .set(bytes as f64);
                    }
                }
                ConsumerSnapshot {
                    consumer_id: e.consumer_id,
                    incarnation: e.incarnation,
                    kind: e.kind,
                    scope: e.scope,
                    scope_origin: e.scope_origin,
                    retention: e.retention,
                    state: e.state,
                    checkpoint_seqno: e.checkpoint_seqno,
                    last_heartbeat_ts_ms: e.last_heartbeat_ts_ms,
                }
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
