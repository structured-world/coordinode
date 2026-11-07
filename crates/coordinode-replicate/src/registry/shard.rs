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
use std::time::{Duration, Instant};

use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::applied::AppliedStop;
use coordinode_storage::engine::coordinator::SnapshotPin;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::oplog_journal::JournalHold;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};
use parking_lot::{Condvar, Mutex};

use super::SeqnoConsumerRegistry;
use super::entry::{
    HEARTBEAT_KEY_PREFIX, Heartbeat, REGISTRY_KEY_PREFIX, RegistryEntry, encode_heartbeat_key,
    encode_registry_key,
};
use super::source::RetentionSource;
use super::types::{
    ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy, ConsumerSnapshot, InitialSeqno,
    RegisteredHandle, RegistrationState, RegistryError, RetentionSpace, RetentionSummary,
    TerminalReason, TopologyScope,
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

/// What holds a new registration's starting position until its floor is
/// published. Dropping it releases the position to the floor; nothing reads
/// what it holds.
enum Held {
    /// An MVCC snapshot pinned in the engine.
    Snapshot { _pin: SnapshotPin },
    /// A log position held against purge, or nothing when the log's purge is
    /// bounded by other means.
    Log { _hold: Option<JournalHold> },
}

/// What the background service has been asked to do since it last looked.
#[derive(Debug, Default)]
struct WakeReasons {
    heartbeat_buffered: bool,
    registry_changed: bool,
    stop: bool,
}

/// Wakes the background service's thread. A reason stays set until the
/// service takes it, so one raised while the service is busy is not lost.
#[derive(Debug, Default)]
struct Wake {
    reasons: Mutex<WakeReasons>,
    raised: Condvar,
}

impl Wake {
    fn raise(&self, set: impl FnOnce(&mut WakeReasons)) {
        set(&mut self.reasons.lock());
        self.raised.notify_one();
    }
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
    /// Held across a floor scan and its publication: a scan taken before a
    /// registration committed must not publish after the scan that sees it,
    /// or the floor rises past what that registration holds.
    floor_publication: Mutex<()>,
    /// Whether `dc` / `rack` scopes are accepted (EE multi-DC topologies).
    allow_topology_scopes: bool,
    /// `true` once a background service is running: `heartbeat` then buffers
    /// into `pending_hb` instead of writing at once.
    batching_on: AtomicBool,
    /// Buffered heartbeats awaiting the next coalesced flush, by consumer id
    /// and the incarnation that sent them: a heartbeat of an ended
    /// incarnation must not keep its successor alive.
    pending_hb: Mutex<HashMap<(String, u64), u64>>,
    /// What wakes the background service: a heartbeat buffered, a registry
    /// write applied, a stop.
    wake: Wake,
    /// Readers' notices of writes to the records they watch.
    watches: Arc<Watches>,
    /// Runs once between a transition's read and its commit: what another
    /// writer does in that window.
    #[cfg(test)]
    before_commit: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// Runs once between a floor scan and its publication.
    #[cfg(test)]
    after_floor_scan: Mutex<Option<Box<dyn FnOnce() + Send>>>,
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

    /// Where a new registration's checkpoint starts, refused when that
    /// position is no longer held.
    ///
    /// MVCC history is held by the engine, which decides what it has
    /// collected: the position is pinned there as it is checked, under the
    /// lock that publishes the watermark, and the pin returned must outlive
    /// the floor the registration then publishes. Checking first and
    /// protecting later lets collection pass the position in between.
    fn resolve_initial(&self, reg: &ConsumerRegistration) -> Result<(u64, Held), RegistryError> {
        if reg.kind.is_seqno_space() {
            return self
                .pin_initial(reg)
                .map(|(at, pin)| (at, Held::Snapshot { _pin: pin }));
        }
        // The log may be purged between reading where it starts and holding
        // that position: the hold is refused then, and the next read is past
        // the purge.
        for _ in 0..TRANSITION_ATTEMPTS {
            let at = match reg.initial_seqno {
                InitialSeqno::FromNow => self.source.head(reg.kind),
                InitialSeqno::FromEarliestRetained => self.source.first_retained(reg.kind),
                InitialSeqno::At(at) => at,
            };
            match self.source.hold_log_from(at) {
                Ok(hold) => return Ok((at, Held::Log { _hold: hold })),
                Err(first) if matches!(reg.initial_seqno, InitialSeqno::At(_)) => {
                    return Err(RegistryError::RetentionLost {
                        checkpoint: at,
                        floor: first,
                    });
                }
                Err(_) => {}
            }
        }
        Err(RegistryError::Replication(format!(
            "the log of {:?} kept being purged under {TRANSITION_ATTEMPTS} attempts",
            reg.consumer_id
        )))
    }

    /// The MVCC starting position of `reg`, pinned in the engine.
    fn pin_initial(&self, reg: &ConsumerRegistration) -> Result<(u64, SnapshotPin), RegistryError> {
        let lost = |checkpoint| RegistryError::RetentionLost {
            checkpoint,
            floor: self.engine.gc_watermark(),
        };
        match reg.initial_seqno {
            InitialSeqno::At(at) => self
                .engine
                .pin_snapshot_at(at)
                .map(|pin| (at, pin))
                .ok_or_else(|| lost(at)),
            InitialSeqno::FromNow => {
                let at = self.source.head(reg.kind);
                self.engine
                    .pin_snapshot_at(at)
                    .map(|pin| (at, pin))
                    .ok_or_else(|| lost(at))
            }
            InitialSeqno::FromEarliestRetained => {
                // The watermark only moves forward: a pin refused at the value
                // just read means it moved, and the next read is past it.
                for _ in 0..TRANSITION_ATTEMPTS {
                    let at = self.engine.gc_watermark();
                    if let Some(pin) = self.engine.pin_snapshot_at(at) {
                        return Ok((at, pin));
                    }
                }
                Err(RegistryError::Replication(format!(
                    "the collected history of {:?} kept moving under {TRANSITION_ATTEMPTS} attempts",
                    reg.consumer_id
                )))
            }
        }
    }

    /// The record of `consumer_id`, with the liveness its heartbeat record
    /// shows, and the version of the registration record.
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
        let Some(bytes) = raw else {
            return Ok(None);
        };
        let entry = RegistryEntry::decode(&bytes)
            .map_err(|e| RegistryError::Replication(format!("decode registry entry: {e}")))?;
        let (heartbeat, _) = self.read_heartbeat(consumer_id)?;
        Ok(Some((entry.with_heartbeat(heartbeat), version)))
    }

    /// The heartbeat record of `consumer_id` and its version, the version
    /// read first: a heartbeat landing between the two reads moves the
    /// version past what a commit conditioned on it expects.
    fn read_heartbeat(
        &self,
        consumer_id: &str,
    ) -> Result<(Option<Heartbeat>, Option<u64>), RegistryError> {
        let key = encode_heartbeat_key(consumer_id);
        let version = self
            .engine
            .record_version(Partition::Registry, &key)
            .map_err(|e| RegistryError::Replication(e.to_string()))?;
        let raw = self
            .engine
            .get(Partition::Registry, &key)
            .map_err(|e| RegistryError::Replication(e.to_string()))?;
        let heartbeat = match raw {
            Some(bytes) => Some(Heartbeat::decode(&bytes).ok_or_else(|| {
                RegistryError::Replication(format!(
                    "decode heartbeat of {consumer_id:?}: {} bytes",
                    bytes.len()
                ))
            })?),
            None => None,
        };
        Ok((heartbeat, version))
    }

    /// Commit `writes` in one transaction, each conditioned on the version its
    /// record had when read (`None` for a record that did not exist), and on
    /// each record in `unchanged` still having the version given. `Ok(false)`
    /// when another writer moved one of them: the caller reads again.
    fn commit_writes(
        &self,
        writes: &[(RegistryEntry, Option<u64>)],
        unchanged: &[(Vec<u8>, Option<u64>)],
    ) -> Result<bool, RegistryError> {
        if writes.is_empty() {
            return Ok(true);
        }
        self.commit_staged(|txn| {
            for (entry, version) in writes {
                let key = encode_registry_key(&entry.consumer_id);
                let value = entry.encode().map_err(|e| {
                    RegistryError::Replication(format!("encode registry entry: {e}"))
                })?;
                txn.expect_version(Partition::Registry, &key, *version)
                    .map_err(|e| RegistryError::Replication(e.to_string()))?;
                txn.put(Partition::Registry, &key, &value)
                    .map_err(|e| RegistryError::Replication(e.to_string()))?;
            }
            for (key, version) in unchanged {
                txn.expect_version(Partition::Registry, key, *version)
                    .map_err(|e| RegistryError::Replication(e.to_string()))?;
            }
            Ok(())
        })
    }

    /// Commit what `stage` puts into one registry transaction. `Ok(false)`
    /// when a condition it stated no longer holds.
    fn commit_staged(
        &self,
        stage: impl FnOnce(&mut Transaction<'_>) -> Result<(), RegistryError>,
    ) -> Result<bool, RegistryError> {
        let oracle = self.engine.oracle().ok_or_else(|| {
            RegistryError::Replication(
                "the registry needs an engine with a timestamp oracle".into(),
            )
        })?;
        let mut txn = Transaction::begin(&self.engine, Some(oracle.as_ref()), oracle.next());
        // Registry transitions release or record retention; holding them back
        // under storage pressure would keep the bytes that relieve it.
        txn.exempt_from_write_pressure();
        stage(&mut txn)?;
        let write_concern = WriteConcern::default();
        let ctx = CommitContext {
            write_concern: &write_concern,
            pipeline: Some(self.pipeline.as_ref()),
            id_gen: Some(&self.id_gen),
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
    /// Heartbeats do not count as another writer: the record's liveness may
    /// be a heartbeat older than the commit, which only a decision on
    /// liveness cares about ([`Self::transition_on_liveness`]).
    fn transition<T>(
        &self,
        consumer_id: &str,
        decide: impl FnMut(Option<RegistryEntry>) -> Result<Step<T>, RegistryError>,
    ) -> Result<T, RegistryError> {
        self.transition_with(consumer_id, false, decide)
    }

    /// [`Self::transition`] for a decision that depends on how long ago the
    /// consumer was last heard from: a heartbeat landing before its commit
    /// makes it read again, so a consumer alive at the commit is never ended
    /// as silent.
    fn transition_on_liveness<T>(
        &self,
        consumer_id: &str,
        decide: impl FnMut(Option<RegistryEntry>) -> Result<Step<T>, RegistryError>,
    ) -> Result<T, RegistryError> {
        self.transition_with(consumer_id, true, decide)
    }

    fn transition_with<T>(
        &self,
        consumer_id: &str,
        on_liveness: bool,
        mut decide: impl FnMut(Option<RegistryEntry>) -> Result<Step<T>, RegistryError>,
    ) -> Result<T, RegistryError> {
        for attempt in 0..TRANSITION_ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(backoff(attempt));
            }
            // Read before the record, so the heartbeat merged into it is no
            // older than the version the commit then requires.
            let heard = if on_liveness {
                let (_, version) = self.read_heartbeat(consumer_id)?;
                vec![(encode_heartbeat_key(consumer_id), version)]
            } else {
                Vec::new()
            };
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
                    if self.commit_writes(&[(entry, version)], &heard)? {
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
        if let RegistrationState::Terminated {
            reason,
            retained_from,
            ..
        } = entry.state
        {
            return Err(RegistryError::Terminated {
                consumer_id: entry.consumer_id,
                incarnation: entry.incarnation,
                reason,
                checkpoint: entry.checkpoint_seqno,
                retained_from,
            });
        }
        Ok(entry)
    }

    /// The first position held in `kind`'s space: the engine's collection
    /// watermark for MVCC history, which the engine alone decides, and the
    /// source's first retained entry for a log.
    fn retained_from(&self, kind: ConsumerKind) -> u64 {
        if kind.is_seqno_space() {
            self.engine.gc_watermark()
        } else {
            self.source.first_retained(kind)
        }
    }

    /// `entry` ended for `reason` at `now`, recording where its history
    /// started then.
    fn terminated(
        &self,
        mut entry: RegistryEntry,
        reason: TerminalReason,
        now: u64,
    ) -> RegistryEntry {
        entry.state = RegistrationState::Terminated {
            reason,
            at_ms: now,
            retained_from: self.retained_from(entry.kind),
        };
        entry
    }

    /// Re-scan the keyspace, refresh the two space-split floors over live
    /// registrations, and publish the engine GC watermark. Returns the
    /// seqno-space floor.
    fn recompute_floor(&self) -> Result<u64, RegistryError> {
        let _publishing = self.floor_publication.lock();
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
        #[cfg(test)]
        {
            let hook = self.after_floor_scan.lock().take();
            if let Some(hook) = hook {
                hook();
            }
        }
        self.floor.store(seqno_floor, Ordering::Release);
        self.oplog_index_floor.store(oplog_floor, Ordering::Release);
        self.source.release_log_below(oplog_floor);
        // The store's own journal, when it keeps one, is the log its log
        // consumers read; a store whose log is a consensus log ignores it.
        self.engine.set_consumer_journal_floor(oplog_floor);
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

    /// Every registration record, live or ended, with the liveness its
    /// heartbeat record shows.
    fn scan_entries(&self) -> Result<Vec<RegistryEntry>, RegistryError> {
        let mut heartbeats: HashMap<Vec<u8>, Heartbeat> = HashMap::new();
        let iter = self
            .engine
            .prefix_scan(Partition::Registry, HEARTBEAT_KEY_PREFIX)
            .map_err(|e| RegistryError::Replication(e.to_string()))?;
        for guard in iter {
            let (key, value) = guard
                .into_inner()
                .map_err(|e| RegistryError::Replication(e.to_string()))?;
            let heartbeat = Heartbeat::decode(&value).ok_or_else(|| {
                RegistryError::Replication(format!(
                    "decode heartbeat {key:?}: {} bytes",
                    value.len()
                ))
            })?;
            let id = key.get(HEARTBEAT_KEY_PREFIX.len()..).unwrap_or_default();
            heartbeats.insert(id.to_vec(), heartbeat);
        }
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
            let heartbeat = heartbeats.get(entry.consumer_id.as_bytes()).copied();
            out.push(entry.with_heartbeat(heartbeat));
        }
        Ok(out)
    }

    /// A heartbeat of `handle` at `at`, written as the background flush
    /// writes it.
    #[cfg(test)]
    fn clock_heartbeat(&self, handle: &RegisteredHandle, at: u64) -> Result<(), RegistryError> {
        self.pending_hb
            .lock()
            .insert((handle.consumer_id().to_string(), handle.incarnation()), at);
        self.flush_pending_heartbeats()
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
            let mut beats = Vec::with_capacity(taken.len());
            let mut stale = Vec::new();
            for ((consumer_id, incarnation), ts) in &taken {
                let live = self
                    .read_entry(consumer_id)?
                    .is_some_and(|(entry, _)| entry.incarnation == *incarnation && entry.is_live());
                if !live {
                    stale.push((consumer_id.clone(), *incarnation));
                    continue;
                }
                // A time already stored for this incarnation is kept if later.
                let (stored, _) = self.read_heartbeat(consumer_id)?;
                let ts_ms = match stored {
                    Some(hb) if hb.incarnation == *incarnation => hb.ts_ms.max(*ts),
                    _ => *ts,
                };
                beats.push((
                    encode_heartbeat_key(consumer_id),
                    Heartbeat {
                        incarnation: *incarnation,
                        ts_ms,
                    },
                ));
            }
            // Unconditioned: a heartbeat writes only its own record, so a
            // transition of the same consumer neither refuses it nor is
            // refused by it.
            let written = if beats.is_empty() {
                Ok(true)
            } else {
                self.commit_staged(|txn| {
                    for (key, hb) in &beats {
                        txn.put(Partition::Registry, key, &hb.encode())
                            .map_err(|e| RegistryError::Replication(e.to_string()))?;
                    }
                    Ok(())
                })
            };
            match written {
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
            metrics::histogram!("registry_heartbeat_batch_size").record(beats.len() as f64);
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
    /// one record at a time, then refresh the floors.
    fn sweep_evictions(&self) -> Result<Sweep, RegistryError> {
        // A buffered heartbeat is a sign of life the stored record does not
        // show yet: persist it before judging liveness.
        self.flush_pending_heartbeats()?;
        // A member that is not the leader decides no transition; it keeps the
        // deadlines so its sweep acts once it leads, and refreshes its floors.
        let mut leads = true;
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
            let ended = match self.transition_on_liveness(&consumer_id, |current| {
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
                        Ok(Step::Write(
                            self.terminated(current, reason, now),
                            Some((reason, checkpoint)),
                        ))
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
            floor_publication: Mutex::new(()),
            allow_topology_scopes: false,
            batching_on: AtomicBool::new(false),
            pending_hb: Mutex::new(HashMap::new()),
            wake: Wake::default(),
            watches: Arc::new(Watches::default()),
            #[cfg(test)]
            before_commit: Mutex::new(None),
            #[cfg(test)]
            after_floor_scan: Mutex::new(None),
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
        let floor = self.core.retained_from(entry.kind);
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
    /// whose [`stop`](RegistryBackground::stop) does a final flush.
    ///
    /// The service is one thread of its own, not a task on an async runtime:
    /// its work waits on commits, and an embedded database without a runtime
    /// runs it too.
    ///
    /// # Errors
    ///
    /// The OS refused a thread for the service.
    pub fn start_background(
        &self,
        cfg: BackgroundConfig,
    ) -> Result<RegistryBackground, RegistryError> {
        self.core.batching_on.store(true, Ordering::Release);

        // A registration another member wrote changes the floor this member
        // publishes, so a write applied to the registry keyspace asks for a
        // sweep, and tells the readers watching that record. The feed is a
        // blocking subscription; one parked thread relays it.
        let applied = self.core.engine.subscribe_applied(Partition::Registry, 16);
        let applied_stop = applied.stopper();
        let relay = Arc::clone(&self.core);
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
                    relay.wake.raise(|r| r.registry_changed = true);
                }
            })
            .map_err(|e| tracing::error!(error = %e, "registry apply relay did not start"))
            .ok();
        // Subscribed before this, so no write applied from here on goes
        // unrelayed.
        watches.set_relayed(relay_thread.is_some());

        let core = Arc::clone(&self.core);
        let service = std::thread::Builder::new()
            .name("registry-service".to_string())
            .spawn(move || core.serve(cfg));
        let service = match service {
            Ok(service) => service,
            Err(e) => {
                watches.set_relayed(false);
                applied_stop.stop();
                self.core.batching_on.store(false, Ordering::Release);
                return Err(RegistryError::Replication(format!(
                    "registry service thread did not start: {e}"
                )));
            }
        };
        Ok(RegistryBackground {
            core: Arc::clone(&self.core),
            service: Some(service),
            applied_stop,
            relay_thread,
            watches,
            config: cfg,
        })
    }
}

impl RegistryCore {
    /// The background service's loop: flush heartbeats a window after the
    /// first one buffered, sweep when a bound can be crossed or the registry
    /// changed, until stopped; then a final flush.
    fn serve(&self, cfg: BackgroundConfig) {
        let window = Duration::from_millis(cfg.heartbeat_window_ms);
        let gap = Duration::from_millis(cfg.eviction_interval_ms);
        let mut flush_at: Option<Instant> = None;
        // The first sweep publishes the floor the stored registrations hold.
        let mut sweep_at: Option<Instant> = Some(Instant::now());
        let mut last_sweep: Option<Instant> = None;
        loop {
            let (heartbeat, changed, stop) = {
                let mut reasons = self.wake.reasons.lock();
                loop {
                    // A buffered heartbeat opens a window only when none is
                    // open: the window already open flushes it.
                    let opens = reasons.heartbeat_buffered && flush_at.is_none();
                    if reasons.stop || reasons.registry_changed || opens {
                        break;
                    }
                    let due = match (flush_at, sweep_at) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (a, b) => a.or(b),
                    };
                    match due {
                        Some(due) if due <= Instant::now() => break,
                        Some(due) => {
                            self.wake.raised.wait_until(&mut reasons, due);
                        }
                        None => self.wake.raised.wait(&mut reasons),
                    }
                }
                let heartbeat = flush_at.is_none() && reasons.heartbeat_buffered;
                if heartbeat {
                    reasons.heartbeat_buffered = false;
                }
                let changed = std::mem::take(&mut reasons.registry_changed);
                (heartbeat, changed, reasons.stop)
            };
            if stop {
                break;
            }
            let now = Instant::now();
            if heartbeat {
                flush_at = Some(now + window);
            }
            if changed {
                let soonest = last_sweep.map_or(now, |at| at + gap);
                sweep_at = Some(sweep_at.map_or(soonest, |due| due.min(soonest)));
            }
            if flush_at.is_some_and(|at| at <= now) {
                flush_at = None;
                if let Err(e) = self.flush_pending_heartbeats() {
                    tracing::warn!(error = %e, "registry heartbeat flush failed");
                    // The heartbeats stay buffered: try again a window on.
                    flush_at = Some(Instant::now() + window);
                }
            }
            if sweep_at.is_some_and(|at| at <= now) {
                last_sweep = Some(now);
                sweep_at = match self.sweep_evictions() {
                    Ok(sweep) => {
                        if sweep.evicted > 0 {
                            tracing::debug!(evicted = sweep.evicted, "registry bound sweep");
                        }
                        let deadline = sweep.next_deadline_ms.map(|at_ms| {
                            let wait = at_ms.saturating_sub(self.clock.now_ms());
                            now + Duration::from_millis(wait).max(gap)
                        });
                        let periodic = sweep.behind.then_some(now + gap);
                        match (deadline, periodic) {
                            (Some(a), Some(b)) => Some(a.min(b)),
                            (a, b) => a.or(b),
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "registry bound sweep failed");
                        Some(now + gap)
                    }
                };
            }
        }
        // Final flush so no buffered heartbeat is lost on a stop.
        if let Err(e) = self.flush_pending_heartbeats() {
            tracing::warn!(error = %e, "registry final heartbeat flush failed");
        }
    }
}

/// Handle to the running background service. Dropping it stops the service,
/// which finishes with a final flush on its own; [`stop`](Self::stop) also
/// waits for that.
pub struct RegistryBackground {
    core: Arc<RegistryCore>,
    service: Option<std::thread::JoinHandle<()>>,
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

    /// Stop the service and wait for its final heartbeat flush. Blocks on
    /// that flush's commit: an async caller runs it where blocking is allowed.
    pub fn stop(mut self) {
        self.signal_stop();
        for thread in [self.relay_thread.take(), self.service.take()]
            .into_iter()
            .flatten()
        {
            if thread.join().is_err() {
                tracing::error!("a registry service thread panicked");
            }
        }
    }

    fn signal_stop(&self) {
        // Heartbeats from now on are written at once: nothing will flush a
        // buffer once the service is gone.
        self.core.batching_on.store(false, Ordering::Release);
        self.watches.set_relayed(false);
        self.applied_stop.stop();
        self.core.wake.raise(|r| r.stop = true);
    }
}

impl Drop for RegistryBackground {
    fn drop(&mut self) {
        // The threads end on their own: the relay at the stop of its feed,
        // the service after its final flush.
        self.signal_stop();
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
        // A registration is new retention; it is admitted only while the
        // store has room to keep it beside what it already owes. Transitions
        // that release or record retention are never held back.
        self.core
            .source
            .admits(reg.kind)
            .map_err(RegistryError::Backpressure)?;
        // Dropped once the registration has published its floor (the
        // transition does that before returning): until then the pin is what
        // holds the history the new checkpoint needs.
        let (checkpoint, _held) = self.core.resolve_initial(&reg)?;
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
            self.core.wake.raise(|r| r.heartbeat_buffered = true);
            return Ok(());
        }
        // Unbatched: refuse a handle that no longer addresses a live
        // registration, then write its heartbeat record at once.
        let current = self
            .core
            .read_entry(handle.consumer_id())?
            .map(|(entry, _)| entry);
        RegistryCore::live_for(current, handle)?;
        self.core
            .pending_hb
            .lock()
            .entry((handle.consumer_id().to_string(), handle.incarnation()))
            .and_modify(|t| *t = (*t).max(now))
            .or_insert(now);
        self.core.flush_pending_heartbeats()
    }

    fn unregister(&self, handle: RegisteredHandle) -> Result<(), RegistryError> {
        let now = self.core.clock.now_ms();
        self.core.transition(handle.consumer_id(), |current| {
            let entry = RegistryCore::live_for(current, &handle)?;
            Ok(Step::Write(
                self.core.terminated(entry, TerminalReason::Cancelled, now),
                (),
            ))
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
        let floors = Floors::of(&entries);
        entries
            .into_iter()
            .map(|e| {
                let mut progress_lag_ms = None;
                let mut retained_bytes = None;
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
                    // last acknowledged. A source that does not account the
                    // space cannot tell.
                    progress_lag_ms = if !behind {
                        Some(0)
                    } else {
                        match source.produced_at_ms(e.kind, e.checkpoint_seqno) {
                            Some(at) => Some(now.saturating_sub(at)),
                            None => source.accounts(e.kind).then_some(0),
                        }
                    };
                    metrics::gauge!("registry_consumer_lag_age_seconds", "consumer_id" => e.consumer_id.clone())
                        .set(progress_lag_ms.unwrap_or(0) as f64 / 1_000.0);
                    metrics::gauge!("registry_consumer_heartbeat_age_seconds", "consumer_id" => e.consumer_id.clone())
                        .set(now.saturating_sub(e.last_heartbeat_ts_ms) as f64 / 1_000.0);
                    retained_bytes = source.retained_bytes_from(e.kind, e.checkpoint_seqno);
                    if let Some(bytes) = retained_bytes {
                        metrics::gauge!("registry_consumer_retained_bytes", "consumer_id" => e.consumer_id.clone())
                            .set(bytes as f64);
                    }
                }
                ConsumerSnapshot {
                    holds_floor: floors.held_by(&e),
                    progress_lag_ms,
                    retained_bytes,
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

/// The lowest live checkpoint of each space.
#[derive(Debug, Clone, Copy, Default)]
struct Floors {
    log: Option<u64>,
    mvcc: Option<u64>,
}

impl Floors {
    fn of(entries: &[RegistryEntry]) -> Self {
        let mut floors = Self::default();
        for e in entries.iter().filter(|e| e.is_live()) {
            let floor = floors.slot(RetentionSpace::of(e.kind));
            *floor = Some(floor.map_or(e.checkpoint_seqno, |f| f.min(e.checkpoint_seqno)));
        }
        floors
    }

    fn slot(&mut self, space: RetentionSpace) -> &mut Option<u64> {
        match space {
            RetentionSpace::Log => &mut self.log,
            RetentionSpace::Mvcc => &mut self.mvcc,
        }
    }

    fn get(&self, space: RetentionSpace) -> Option<u64> {
        match space {
            RetentionSpace::Log => self.log,
            RetentionSpace::Mvcc => self.mvcc,
        }
    }

    /// Whether `e` is live and at the floor of its space.
    fn held_by(&self, e: &RegistryEntry) -> bool {
        e.is_live() && self.get(RetentionSpace::of(e.kind)) == Some(e.checkpoint_seqno)
    }
}

impl ShardConsumerRegistry {
    /// What the live registrations hold of the source in each space, the
    /// bytes counted once, and which registrations hold it: one entry per
    /// space, log first.
    ///
    /// # Errors
    ///
    /// The registry's records could not be read.
    pub fn retention_summary(&self) -> Result<Vec<RetentionSummary>, RegistryError> {
        let entries = self.core.scan_entries()?;
        let floors = Floors::of(&entries);
        let source = self.core.source.as_ref();
        let summary = [
            (RetentionSpace::Log, ConsumerKind::OplogEvents),
            (RetentionSpace::Mvcc, ConsumerKind::LsmStateDelta),
        ]
        .into_iter()
        .map(|(space, kind)| {
            let floor = floors.get(space);
            let retained_bytes = floor.and_then(|at| source.retained_bytes_from(kind, at));
            metrics::gauge!("registry_retained_bytes", "space" => space.label())
                .set(retained_bytes.unwrap_or(0) as f64);
            RetentionSummary {
                space,
                floor,
                retained_bytes,
                held_by: entries
                    .iter()
                    .filter(|e| RetentionSpace::of(e.kind) == space && floors.held_by(e))
                    .map(|e| e.consumer_id.clone())
                    .collect(),
                refused: source.admits(kind).err(),
            }
        })
        .collect();
        Ok(summary)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
