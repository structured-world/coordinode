//! Storage-backed Raft log storage and state machine.
//!
//! Implements openraft's `RaftLogStorage` and `RaftStateMachine` traits
//! on top of the CoordiNode LSM storage engine.
//!
//! ## Partition & Key Namespace Convention
//!
//! Two partitions are used, with strict key-prefix namespaces:
//!
//! **`Partition::Raft`** — log entries, vote, purge tracking (raw writes, no MVCC):
//! - `raft:vote` — persisted vote (term + voted_for)
//! - `raft:log:{index:020}` — log entries (zero-padded for sorted iteration)
//! - `raft:committed` — last committed log id
//! - `raft:purged` — last purged log id
//! - `raft:oplog:last_log_id` — oplog-based last log id
//!
//! **`Partition::Schema`** — shared with user schema data (`schema:*` keys):
//! - `raft:sm:applied`: last applied log id (state machine)
//! - `raft:sm:membership`: last applied membership config
//! - `raft:snapshot:meta`: snapshot metadata
//! - `raft:snapshot:file`: the snapshot's file name; the bytes live in the
//!   engine's `raft-snapshot` directory, outside the store
//!
//! User schema uses `schema:label:*`, `schema:edge_type:*`, `schema:meta:*`
//! keys in the same `Partition::Schema`. No collision occurs because prefixes
//! don't overlap (`raft:` vs `schema:`) and write methods differ (Raft uses
//! raw puts, schema uses MVCC puts with timestamp suffix).

use std::collections::HashMap;
use std::fmt::Debug;
use std::io;
use std::ops::RangeBounds;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use coordinode_storage::oplog::{OplogEntry, OplogManager, OplogOp};

use futures_util::stream::StreamExt;
use openraft::entry::RaftEntry;
use openraft::storage::{IOFlushed, LogState, RaftLogStorage, RaftStateMachine};
use openraft::{OptionalSend, RaftLogReader, RaftSnapshotBuilder};
use serde::{Deserialize, Serialize};

use coordinode_core::txn::frame::{DecodeLimits, decode_proposal, encode_proposal};
use coordinode_core::txn::proposal::{MetadataCommand, Mutation, RaftProposal};
use coordinode_storage::engine::core::{
    LocalSync, LocalWrite, RaftApplyFence, RaftApplyState, RaftCoverage, StorageEngine,
};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::{StorageError, StorageResult};

use crate::snapshot::SnapshotData;

/// Maximum age for dedup entries before GC (10 minutes): far longer than
/// any proposal retry window, so an entry this old can no longer meet a
/// retry of its proposal.
const DEDUP_MAX_AGE_SECS: u64 = 600;

/// GC interval for dedup map (5 minutes = max age / 2), so no entry outlives
/// its age by more than half of it.
const DEDUP_GC_INTERVAL_SECS: u64 = 300;

// ── Type Configuration ──────────────────────────────────────────────

/// Application request data: one or more proposals batched into a single
/// Raft log entry.
///
/// Batching reduces the number of Raft round-trips for concurrent writers:
/// N individual `client_write()` calls become 1 batched entry.
/// [`WaitForMajorityService`](crate::wait_majority::WaitForMajorityService)
/// uses this to coalesce proposals from multiple writers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Request {
    pub proposals: Vec<RaftProposal>,
    /// The leader's closed bound when it handed this entry to the log: every
    /// commit timestamp below it is in an earlier entry or never will be.
    /// `0` carries none (an entry handed over before the leader could vouch
    /// for its term's prefix). Exclusive, so a reader at T may rely on it
    /// only for `T < closed_below`.
    pub closed_below: u64,
}

impl Request {
    /// Create a request from a single proposal (non-batched path).
    pub fn single(proposal: RaftProposal) -> Self {
        Self {
            proposals: vec![proposal],
            closed_below: 0,
        }
    }

    /// Create a request from a batch of proposals (coalesced path).
    pub fn batch(proposals: Vec<RaftProposal>) -> Self {
        Self {
            proposals,
            closed_below: 0,
        }
    }

    /// An entry carrying only a closed bound: what a leader appends when the
    /// last entries it handed over left timestamps uncovered and nothing
    /// else is coming to cover them.
    pub fn closing(closed_below: u64) -> Self {
        Self {
            proposals: Vec::new(),
            closed_below,
        }
    }
}

impl std::fmt::Display for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_mutations: usize = self.proposals.iter().map(|p| p.mutations.len()).sum();
        write!(
            f,
            "Request(proposals={}, mutations={})",
            self.proposals.len(),
            total_mutations
        )
    }
}

/// Application response after applying a proposal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Response {
    /// Number of mutations applied.
    pub mutations_applied: usize,
}

impl std::fmt::Display for Response {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Response(applied={})", self.mutations_applied)
    }
}

openraft::declare_raft_types!(
    /// CoordiNode Raft type configuration.
    pub TypeConfig:
        D = Request,
        R = Response,
        Node = MemberNode,
);

pub use crate::member::MemberNode;

// Type aliases for convenience
pub type CommittedLeaderId = openraft::vote::leader_id_adv::CommittedLeaderId<u64, u64>;
pub type LeaderId = openraft::impls::leader_id_adv::LeaderId<u64, u64>;
pub type LogId = openraft::LogId<CommittedLeaderId>;
pub type Vote = openraft::impls::Vote<LeaderId>;
pub type Entry = openraft::impls::Entry<
    CommittedLeaderId,
    openraft::impls::EntryPayload<Request, u64, MemberNode>,
>;
pub type SnapshotMeta = openraft::storage::SnapshotMeta<CommittedLeaderId, u64, MemberNode>;
pub type Snapshot = openraft::storage::Snapshot<CommittedLeaderId, u64, MemberNode, SnapshotData>;
pub type StoredMembership = openraft::StoredMembership<CommittedLeaderId, u64, MemberNode>;

// ── Key Prefixes ────────────────────────────────────────────────────

const KEY_VOTE: &[u8] = b"raft:vote";
const KEY_COMMITTED: &[u8] = b"raft:committed";
const KEY_SM_APPLIED: &[u8] = b"raft:sm:applied";
const KEY_SM_MEMBERSHIP: &[u8] = b"raft:sm:membership";
const KEY_PURGED: &[u8] = b"raft:purged";
/// The committed entry a member's disk lost and its peers still hold; present
/// until the member has caught up past it.
const KEY_CATCH_UP: &[u8] = b"raft:catch-up";

/// What opening the log does when the disk lost entries already committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LostCommitted {
    /// Refuse the open: no peer holds them, so nothing could restore them.
    Refuse,
    /// Open with the log ending before the hole and record the loss: the
    /// member's peers hold the entries and its leader sends them again.
    CatchUp,
}

/// Whether the membership the state machine last applied has a voter other
/// than `node_id`: one that may hold entries this member's disk lost.
///
/// # Errors
///
/// The stored membership cannot be read or decoded.
pub(crate) fn has_other_voters(engine: &StorageEngine, node_id: u64) -> Result<bool, io::Error> {
    let stored: Option<StoredMembership> =
        load_stored(engine, Partition::Schema, KEY_SM_MEMBERSHIP)?;
    Ok(stored.is_some_and(|m| m.membership().voter_ids().any(|id| id != node_id)))
}

/// Forget the committed entry recorded as lost, once the member holds it
/// again. Durable before it returns.
///
/// # Errors
///
/// The write or its flush fails.
pub fn clear_catch_up(engine: &StorageEngine) -> Result<(), io::Error> {
    // One synced append to the member-local journal replaces the flush of the
    // whole Raft tree.
    engine
        .commit_local(
            vec![LocalWrite::Delete {
                partition: Partition::Raft,
                key: KEY_CATCH_UP.to_vec(),
            }],
            LocalSync::BeforeUse,
        )
        .map_err(|e| io::Error::other(format!("persist the caught-up member: {e}")))
}

/// Rebuild `partition` from the checkpoint at `checkpoint_dir` and this
/// node's Raft log `log`, with the applies paused: the checkpoint's rows,
/// then every proposal of the log the checkpoint's tree lacks, up to where
/// the applies stand. The rebuilt tree then holds exactly what the other
/// trees do.
///
/// Proposals replay through the state machine's own rules: a repeated
/// proposal id with the same size is skipped, as far back as the replay
/// reaches (a restarted state machine starts its dedup empty the same way).
///
/// # Errors
///
/// No Raft state machine runs over `engine`, the log no longer holds an
/// entry the checkpoint lacks, or the checkpoint or a write fails. Nothing
/// is changed unless the log holds every entry needed.
pub fn rebuild_partition_from_checkpoint(
    engine: &StorageEngine,
    log: &Mutex<OplogManager>,
    checkpoint_dir: &std::path::Path,
    partition: Partition,
) -> Result<(), io::Error> {
    let fence = engine
        .raft_fence()
        .ok_or_else(|| io::Error::other("no Raft state machine runs over this store"))?;
    fence
        .with_applies_paused(&mut |applies, payload| {
            let (rows, held) = StorageEngine::checkpoint_raft_partition(checkpoint_dir, partition)?;
            let (from, to) = (held.base_next(), applies.next());
            let entries = if from < to {
                log.lock()
                    .map_err(|_| StorageError::Io("raft log mutex poisoned".into()))?
                    .read_range(from, to)?
            } else {
                Vec::new()
            };
            let complete = entries.first().is_none_or(|e| e.index == from)
                && u64::try_from(entries.len()).is_ok_and(|n| n == to.saturating_sub(from));
            if !complete {
                return Err(StorageError::Io(format!(
                    "the Raft log no longer holds entries {from}..{to} the checkpoint \
                     lacks; repair {} from a peer",
                    partition.name()
                )));
            }
            engine.begin_partition_rebuild(partition, &rows)?;
            let mut seen: HashMap<u64, usize> = HashMap::new();
            for oplog_entry in entries {
                let index = oplog_entry.index;
                let entry = LogStore::oplog_to_entry(oplog_entry)
                    .map_err(|e| StorageError::Io(e.to_string()))?;
                let openraft::entry::EntryPayload::Normal(request) = &entry.payload else {
                    continue;
                };
                for (sub, proposal) in request.proposals.iter().enumerate() {
                    let sub = u32::try_from(sub).map_err(|_| {
                        StorageError::Io(format!("entry {index} carries over 2^32 proposals"))
                    })?;
                    let size = proposal.size_estimate();
                    let repeated = seen.insert(proposal.id.as_raw(), size) == Some(size);
                    if repeated || held.holds(index, sub) {
                        continue;
                    }
                    engine.apply_raft_proposal(
                        &proposal.mutations,
                        proposal.commit_ts.as_raw(),
                        index,
                        sub,
                        |p| p != partition,
                    )?;
                }
            }
            engine.finish_raft_rebuild(partition, to, payload)
        })
        .map_err(|e| io::Error::other(format!("rebuild {}: {e}", partition.name())))
}

// ── Log Store ─────────────────────────────────────────────────

/// Tells a waiting writer at which log index its proposal reached this
/// member's log: written, or durable.
///
/// The proposal pipeline subscribes by proposal id before it submits. The log
/// store fires a [`LogReach::Written`] waiter once the batch holding the
/// proposal is handed to the operating system, and the log's sync thread
/// fires every remaining waiter once the batch is durable. This is what a
/// write concern of `w:1` waits for (written for a volatile `j`, durable for
/// `j: journal`), and what `w:N` counts from: the leader's own copy is the
/// first of the N.
#[derive(Default)]
pub struct AppendNotifier {
    waiters: Mutex<HashMap<coordinode_core::txn::proposal::ProposalId, Waiter>>,
    /// Waiters subscribed for [`LogReach::Written`], so an append with none
    /// skips the lookup. Counts registrations, so it can only overstate.
    written: std::sync::atomic::AtomicUsize,
}

/// How far a proposal must reach this member's log before its waiter fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogReach {
    /// Handed to the operating system, not yet synced: a volatile `j`.
    Written,
    /// Synced: `j: journal`.
    Durable,
}

struct Waiter {
    tx: tokio::sync::oneshot::Sender<u64>,
    reach: LogReach,
}

impl AppendNotifier {
    /// Wait for `id` to reach this member's log as far as `reach`; resolves
    /// to its log index. Subscribe before submitting the proposal, or the
    /// append may fire first and the receiver never resolves.
    pub fn subscribe(
        &self,
        id: coordinode_core::txn::proposal::ProposalId,
        reach: LogReach,
    ) -> tokio::sync::oneshot::Receiver<u64> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if let Ok(mut waiters) = self.waiters.lock() {
            if reach == LogReach::Written {
                self.written
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            waiters.insert(id, Waiter { tx, reach });
        }
        rx
    }

    /// `proposals` were written to the log: answer the waiters that asked
    /// for no more.
    pub(crate) fn notify_written(
        &self,
        proposals: &[(coordinode_core::txn::proposal::ProposalId, u64)],
    ) {
        if self.written.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return;
        }
        let Ok(mut waiters) = self.waiters.lock() else {
            return;
        };
        for &(id, index) in proposals {
            if waiters
                .get(&id)
                .is_some_and(|w| w.reach == LogReach::Written)
            {
                if let Some(waiter) = waiters.remove(&id) {
                    self.written
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    waiter.answer(index);
                }
            }
        }
    }

    /// `id` is durable in the log at `index`: answer its waiter, whatever it
    /// asked for.
    pub(crate) fn notify(&self, id: coordinode_core::txn::proposal::ProposalId, index: u64) {
        let waiter = match self.waiters.lock() {
            Ok(mut waiters) => waiters.remove(&id),
            Err(_) => None,
        };
        if let Some(waiter) = waiter {
            if waiter.reach == LogReach::Written {
                self.written
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
            waiter.answer(index);
        }
    }
}

impl Waiter {
    fn answer(self, index: u64) {
        if self.tx.send(index).is_err() {
            // The writer stopped waiting (its write concern timed out).
            tracing::trace!(index, "append notified after its waiter gave up");
        }
    }
}

/// Storage-backed Raft log storage.
///
/// Log entries are stored in **oplog segments** (one sealed file per segment),
/// not in the LSM KV store. This delivers:
/// - O(1) `last_log_id` via an in-memory cache (eliminates the O(N) scan)
/// - Segment-granular purge via `OplogManager::purge_before`
/// - Sequential I/O for `append` and `try_get_log_entries`
///
/// An append returns once its entries are written; a sync thread makes them
/// durable and then answers openraft, so the Raft core replicates while the
/// disk flushes and one sync covers every append queued meanwhile.
///
/// Raft metadata (vote, committed, purged, last_log_id) is stored in
/// `Partition::Raft` to separate it from application data.
///
/// All fields use `Arc` so `get_log_reader()` returns a cheap clone that
/// shares the same oplog handle and caches.
pub struct LogStore {
    engine: crate::engine_hold::EngineHold,
    /// Oplog manager for log entry segments.
    oplog: Arc<Mutex<OplogManager>>,
    /// Waiters for "my proposal is durable in this log", see [`AppendNotifier`].
    append_notifier: Arc<AppendNotifier>,
    /// Makes appends durable off the Raft core and answers them.
    sync: Arc<crate::log_sync::LogSync>,
    /// In-memory cache of the last appended log id. Updated on every
    /// `append()` and persisted to `Partition::Raft` so it survives restarts.
    last_log_id: Arc<Mutex<Option<LogId>>>,
    /// In-memory cache of the last purged log id. Updated on every `purge()`
    /// and persisted to `Partition::Raft`.
    last_purged: Arc<Mutex<Option<LogId>>>,
    /// This store's (or reader's) share of [`EngineWork`], since it holds
    /// the engine for as long as it lives.
    hold: EngineWorkGuard,
}

/// Where a shard's Raft log lives on disk.
pub struct RaftOplogDirs {
    /// The directory new segments are written to.
    pub active: std::path::PathBuf,
    /// Every directory that may hold segments of the log, `active` among
    /// them: a change of endpoint routing leaves older segments where they
    /// were written. Some may not exist.
    pub all: Vec<std::path::PathBuf>,
}

/// The directories of `shard_id`'s Raft log: `<endpoint>/oplog/<shard_id>/`
/// on the oplog endpoint chosen for the shard, and on every other
/// oplog-eligible endpoint for segments written under an earlier routing.
///
/// # Errors
///
/// No endpoint is eligible to hold the oplog.
pub fn raft_oplog_dirs(engine: &StorageEngine, shard_id: u32) -> Result<RaftOplogDirs, io::Error> {
    let shard_dir = |path: &std::path::Path| path.join("oplog").join(format!("{shard_id}"));
    let active = shard_dir(
        &engine
            .select_oplog_endpoint(shard_id)
            .map_err(|e| io::Error::other(e.to_string()))?
            .path,
    );
    let all = engine
        .all_oplog_eligible_endpoints()
        .iter()
        .map(|ep| shard_dir(&ep.path))
        .collect();
    Ok(RaftOplogDirs { active, all })
}

/// Bytes of `engine`'s Raft log segments on disk, over every directory that
/// may hold them. A directory that does not exist holds none.
///
/// # Errors
///
/// No endpoint is eligible to hold the log, or a directory cannot be read.
pub(crate) fn raft_log_bytes(engine: &StorageEngine) -> Result<u64, io::Error> {
    let mut bytes = 0u64;
    for dir in raft_oplog_dirs(engine, 0)?.all {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        };
        for entry in entries {
            let meta = entry?.metadata()?;
            if meta.is_file() {
                bytes += meta.len();
            }
        }
    }
    Ok(bytes)
}

impl LogStore {
    /// A shared handle to the Raft oplog manager. Cloned out before the
    /// `LogStore` is moved into `openraft::Raft` so a subsystem (e.g. WAL-replay
    /// repair) can read committed entries since a checkpoint without going
    /// through openraft. Reads serialize against appends via the inner `Mutex`.
    pub fn oplog_handle(&self) -> Arc<Mutex<OplogManager>> {
        Arc::clone(&self.oplog)
    }

    /// The local-append notifier the proposal pipeline waits on for `w:1` and
    /// `w:N`. Cloned out before the `LogStore` is moved into `openraft::Raft`.
    pub fn append_notifier(&self) -> Arc<AppendNotifier> {
        Arc::clone(&self.append_notifier)
    }

    /// The [`EngineWork`] this store and its readers count in. The state
    /// machine joins it through
    /// [`CoordinodeStateMachine::with_engine_work`], so a node waits on one
    /// handle for everything openraft holds the engine through.
    pub fn engine_work(&self) -> EngineWork {
        self.hold.work()
    }

    /// Open the LogStore, routing oplog segments to the oplog-eligible
    /// endpoint chosen for shard 0, so the log lands on media fit to carry
    /// it rather than wherever the data partitions live. Path layout:
    /// `<endpoint.path>/oplog/<shard_id>/`.
    /// Recovery scans every oplog-eligible endpoint's directory for sealed
    /// segments left over from a previous config-driven routing.
    ///
    /// The log's end is read from its segments, the only record of it that
    /// is durable exactly when the entries are; `last_purged` comes from
    /// `Partition::Raft`, recorded durably before any segment goes.
    ///
    /// # Errors
    ///
    /// The log's directories or its last segment cannot be read. A log that
    /// cannot be read is refused rather than taken as empty: an empty log
    /// would let the group start over on top of acknowledged entries. So is a
    /// log whose recovery would end it at or below its committed index, and
    /// one recorded as having lost committed entries: no peer could send
    /// them again.
    pub fn open(engine: Arc<StorageEngine>) -> Result<Self, io::Error> {
        Self::open_with(engine, LostCommitted::Refuse).map(|(store, _)| store)
    }

    /// Open the log as [`open`](Self::open) does, deciding by `lost` what a
    /// disk that lost committed entries leads to. With
    /// [`LostCommitted::CatchUp`] the log ends before the first entry it no
    /// longer holds, the committed index comes down to that end, and the
    /// highest committed entry lost is recorded durably until
    /// [`clear_catch_up`]; it is returned while recorded, for the member to
    /// stay out of elections until its leader has sent it back.
    ///
    /// # Errors
    ///
    /// As [`open`](Self::open), except for lost committed entries under
    /// [`LostCommitted::CatchUp`].
    pub fn open_with(
        engine: Arc<StorageEngine>,
        lost: LostCommitted,
    ) -> Result<(Self, Option<LogId>), io::Error> {
        // Shard 0 = single Raft log shard in CE. Multi-shard EE will
        // open one LogStore per shard, each routing to its own
        // select_oplog_endpoint(shard_id).
        let shard_id: u32 = 0;
        // Before the log's first entry: from here the store opens only as a
        // member, never with a journal of its own beside this log.
        engine
            .bind_consensus_log()
            .map_err(|e| io::Error::other(e.to_string()))?;
        let RaftOplogDirs {
            active: active_dir,
            all: recovery_dirs,
        } = raft_oplog_dirs(&engine, shard_id)?;
        let settings = engine.oplog_config();
        let committed: Option<LogId> = load_stored(&engine, Partition::Raft, KEY_COMMITTED)?;
        let recorded: Option<LogId> = load_stored(&engine, Partition::Raft, KEY_CATCH_UP)?;
        // Recovering the segments must not end the log below an entry
        // already committed (those were acknowledged) unless peers hold it.
        let must_hold = match lost {
            LostCommitted::Refuse => {
                if let Some(lost) = recorded {
                    return Err(io::Error::other(format!(
                        "this member lost committed log entries through index {} and no \
                         peer is left to send them again",
                        lost.index
                    )));
                }
                committed.map(|id| id.index)
            }
            LostCommitted::CatchUp => None,
        };
        let oplog = OplogManager::open_multi(
            &active_dir,
            &recovery_dirs,
            shard_id,
            settings.max_segment_bytes,
            settings.max_segment_entries,
            settings.retention_secs,
            must_hold,
        )
        .map_err(|e| io::Error::other(e.to_string()))?
        .with_sync_method(settings.sync_method);

        let last_purged = Self::load_log_id_from_partition(&engine, KEY_PURGED)?;
        // The last entry in the segments; its recovery reads the last segment
        // (which a crash may have left without a footer). A log whose entries
        // are all purged ends where it was purged, as openraft expects of an
        // empty log (`None` sorts below every id).
        let recovered = if oplog.has_segments() {
            Self::recover_last_log_id_from_oplog(&oplog)?
        } else {
            None
        };
        let last_log_id = recovered.max(last_purged);

        // Committed entries the log no longer holds: the leader sends them
        // again. Until then the replay openraft starts from must stop where
        // the log does, and the loss stays recorded across restarts.
        let mut catch_up = recorded;
        if let Some(lost) = committed.filter(|c| Some(*c) > last_log_id) {
            catch_up = catch_up.max(Some(lost));
            let put = |key: &[u8], id: &LogId| -> Result<LocalWrite, io::Error> {
                Ok(LocalWrite::Put {
                    partition: Partition::Raft,
                    key: key.to_vec(),
                    value: rmp_serde::to_vec(id).map_err(|e| io::Error::other(e.to_string()))?,
                })
            };
            let mut writes = Vec::with_capacity(2);
            if let Some(through) = &catch_up {
                writes.push(put(KEY_CATCH_UP, through)?);
            }
            writes.push(match &last_log_id {
                Some(end) => put(KEY_COMMITTED, end)?,
                None => LocalWrite::Delete {
                    partition: Partition::Raft,
                    key: KEY_COMMITTED.to_vec(),
                },
            });
            // Both records in one synced journal append, in place of a flush
            // of the whole Raft tree.
            engine
                .commit_local(writes, LocalSync::BeforeUse)
                .map_err(|e| io::Error::other(format!("persist the lost entries: {e}")))?;
            tracing::error!(
                lost_through = lost.index,
                log_ends_at = last_log_id.map(|id| id.index),
                "raft: the disk lost committed log entries; this member stays out of \
                 elections until its leader has sent them again"
            );
        }

        let append_notifier = Arc::new(AppendNotifier::default());
        let sync = Arc::new(crate::log_sync::LogSync::start(Arc::clone(
            &append_notifier,
        ))?);
        let store = Self {
            engine: crate::engine_hold::EngineHold::new(engine, "raft log store"),
            oplog: Arc::new(Mutex::new(oplog)),
            append_notifier,
            sync,
            last_log_id: Arc::new(Mutex::new(last_log_id)),
            last_purged: Arc::new(Mutex::new(last_purged)),
            hold: EngineWork::default().start(),
        };
        Ok((store, catch_up))
    }

    /// Reconstruct `last_log_id` from oplog segment files.
    ///
    /// Delegates the segment scan to `OplogManager::recover_last_entry`, which
    /// handles both properly sealed segments (footer present) and unsealed
    /// segments that survived an unclean shutdown (entries fsynced, no footer).
    ///
    /// Returns `Ok(Some(log_id))` if at least one valid entry is found,
    /// `Ok(None)` if all segments are empty, or `Err` on I/O failure.
    fn recover_last_log_id_from_oplog(oplog: &OplogManager) -> Result<Option<LogId>, io::Error> {
        let last_oplog_entry = oplog
            .recover_last_entry()
            .map_err(|e| io::Error::other(e.to_string()))?;

        match last_oplog_entry {
            Some(entry) => {
                let raft_entry = Self::oplog_to_entry(entry)?;
                Ok(Some(raft_entry.log_id))
            }
            None => Ok(None),
        }
    }

    /// The log id stored at `key`, `None` when nothing is. A value that cannot
    /// be read or decoded fails the open: taken as absent, a lost purge point
    /// would let the log be read from entries that are gone.
    fn load_log_id_from_partition(
        engine: &StorageEngine,
        key: &[u8],
    ) -> Result<Option<LogId>, io::Error> {
        load_stored(engine, Partition::Raft, key)
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, io::Error> {
        self.engine
            .get(Partition::Raft, key)
            .map(|opt| opt.map(|v| v.to_vec()))
            .map_err(|e| io::Error::other(e.to_string()))
    }

    fn put(&self, key: &[u8], value: &[u8]) -> Result<(), io::Error> {
        self.engine
            .put(Partition::Raft, key, value)
            .map_err(|e| io::Error::other(e.to_string()))
    }

    fn delete(&self, key: &[u8]) -> Result<(), io::Error> {
        self.engine
            .delete(Partition::Raft, key)
            .map_err(|e| io::Error::other(e.to_string()))
    }

    /// Encode a Raft Entry as an OplogEntry.
    ///
    /// The first op is the entry's envelope; a Normal entry's proposals follow
    /// it, one unit frame each, so every proposal's bytes are written once, in
    /// the encoding the embedded journal records a unit in. A change stream
    /// reads the operations out of those frames.
    fn entry_to_oplog(entry: &Entry) -> Result<OplogEntry, io::Error> {
        let (envelope, units) = match &entry.payload {
            openraft::entry::EntryPayload::Normal(request) => {
                let units = request
                    .proposals
                    .iter()
                    .map(|p| encode_proposal(p).map(|frame| OplogOp::Unit { frame }))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(io::Error::other)?;
                let envelope =
                    Entry::new_normal(entry.log_id, Request::closing(request.closed_below));
                (rmp_serde::to_vec(&envelope), units)
            }
            _ => (rmp_serde::to_vec(entry), Vec::new()),
        };
        let data = envelope.map_err(|e| io::Error::other(e.to_string()))?;
        // RaftEntry is always first: oplog_to_entry() relies on ops[0] being it.
        let mut ops = Vec::with_capacity(1 + units.len());
        ops.push(OplogOp::RaftEntry { data });
        ops.extend(units);
        // Proposals carry no migration flag yet.
        let is_migration = false;

        // A batched entry commits every proposal; its latest commit
        // timestamp is the entry's. Membership entries carry none.
        let ts = match &entry.payload {
            openraft::entry::EntryPayload::Normal(request) => request
                .proposals
                .iter()
                .map(|p| p.commit_ts.as_raw())
                .max()
                .unwrap_or(0),
            _ => 0,
        };

        Ok(OplogEntry {
            ts,
            term: entry.log_id.committed_leader_id().term,
            index: entry.log_id.index,
            shard: 0,
            ops,
            is_migration,
            pre_images: None,
        })
    }

    /// Decode an OplogEntry back into a Raft Entry.
    fn oplog_to_entry(oplog_entry: OplogEntry) -> Result<Entry, io::Error> {
        let mut ops = oplog_entry.ops.into_iter();
        let mut entry: Entry = match ops.next() {
            Some(OplogOp::RaftEntry { data }) => {
                rmp_serde::from_slice(&data).map_err(|e| io::Error::other(e.to_string()))?
            }
            Some(op) => {
                return Err(io::Error::other(format!(
                    "unexpected oplog op in Raft segment: expected RaftEntry, got {:?}",
                    std::mem::discriminant(&op)
                )));
            }
            None => return Err(io::Error::other("oplog entry has no ops in Raft segment")),
        };
        // An entry written before proposals were stored as unit frames holds
        // them in its envelope, followed by their operations for change
        // streams; the envelope alone is read then.
        if let openraft::entry::EntryPayload::Normal(request) = &mut entry.payload {
            if request.proposals.is_empty() {
                for op in ops {
                    if let OplogOp::Unit { frame } = op {
                        request.proposals.push(
                            decode_proposal(&frame, &DecodeLimits::DEFAULT)
                                .map_err(io::Error::other)?,
                        );
                    }
                }
            }
        }
        Ok(entry)
    }

    /// Whether an entry at index `from` or later writes data a user put
    /// there, by the measure of [`Partition::is_user_key`].
    ///
    /// Entries past the state machine's resume point exist only here until
    /// they are replayed: a crash drops what they applied to unflushed trees,
    /// so the trees alone understate what this store holds.
    ///
    /// # Errors
    ///
    /// The log cannot be read or an entry does not decode.
    pub fn writes_user_data_from(&self, from: u64) -> Result<bool, io::Error> {
        let entries = self
            .oplog
            .lock()
            .map_err(|_| io::Error::other("raft log mutex poisoned"))?
            .read_range(from, u64::MAX)
            .map_err(|e| io::Error::other(e.to_string()))?;
        for oplog_entry in entries {
            let entry = Self::oplog_to_entry(oplog_entry)?;
            let openraft::entry::EntryPayload::Normal(request) = &entry.payload else {
                continue;
            };
            let writes = request
                .proposals
                .iter()
                .flat_map(|proposal| &proposal.mutations)
                .any(|mutation| match mutation {
                    Mutation::Put { partition, key, .. }
                    | Mutation::Merge { partition, key, .. } => {
                        Partition::from(*partition).is_user_key(key)
                    }
                    // Removals, metadata and operation commands and index
                    // maintenance add nothing of the user's: an index entry
                    // follows a record the same unit or an earlier one wrote,
                    // and an operation record describes the unit's own data.
                    Mutation::Delete { .. }
                    | Mutation::RemoveRange { .. }
                    | Mutation::Command(_)
                    | Mutation::Derive(_)
                    | Mutation::Operation(_) => false,
                });
            if writes {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether an entry of this log from index `from` on changes the
    /// membership to one naming `node_id` beside another node: the log of a
    /// member of a group whose membership was not applied yet.
    pub fn names_member_from(&self, from: u64, node_id: u64) -> Result<bool, io::Error> {
        use openraft::entry::RaftEntry as _;
        let entries = self
            .oplog
            .lock()
            .map_err(|_| io::Error::other("raft log mutex poisoned"))?
            .read_range(from, u64::MAX)
            .map_err(|e| io::Error::other(e.to_string()))?;
        for oplog_entry in entries {
            let entry = Self::oplog_to_entry(oplog_entry)?;
            if entry
                .get_membership()
                .is_some_and(|m| names_beside_another(&m, node_id))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Whether `membership` names `node_id` and at least one other node.
pub(crate) fn names_beside_another(
    membership: &openraft::Membership<u64, MemberNode>,
    node_id: u64,
) -> bool {
    membership.nodes().any(|(id, _)| *id == node_id) && membership.nodes().count() > 1
}

impl RaftLogReader<TypeConfig> for LogStore {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + OptionalSend>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry>, io::Error> {
        let start = match range.start_bound() {
            std::ops::Bound::Included(&s) => s,
            std::ops::Bound::Excluded(&s) => s + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end_exclusive = match range.end_bound() {
            std::ops::Bound::Included(&e) => e + 1,
            std::ops::Bound::Excluded(&e) => {
                if e == 0 {
                    return Ok(Vec::new());
                }
                e
            }
            std::ops::Bound::Unbounded => u64::MAX,
        };

        if start >= end_exclusive {
            return Ok(Vec::new());
        }

        // First valid index: everything strictly below this has been purged.
        // When last_purged is None (no purge happened yet), min_valid_index = 0,
        // so no entries are skipped — including the Bootstrap entry at index 0.
        let min_valid_index: u64 = self
            .last_purged
            .lock()
            .map_err(|_| io::Error::other("last_purged mutex poisoned"))?
            .map(|id| id.index + 1)
            .unwrap_or(0);

        let oplog_entries = self
            .oplog
            .lock()
            .map_err(|_| io::Error::other("oplog mutex poisoned"))?
            .read_range(start, end_exclusive)
            .map_err(|e| io::Error::other(e.to_string()))?;

        let mut entries = Vec::with_capacity(oplog_entries.len());
        for oe in oplog_entries {
            // Skip entries that fall within the purged range (segment granularity
            // means a segment may still contain some pre-purge-boundary entries).
            if oe.index < min_valid_index {
                continue;
            }
            entries.push(Self::oplog_to_entry(oe)?);
        }

        Ok(entries)
    }

    async fn read_vote(&mut self) -> Result<Option<Vote>, io::Error> {
        match self.get(KEY_VOTE)? {
            Some(bytes) => {
                let vote: Vote =
                    rmp_serde::from_slice(&bytes).map_err(|e| io::Error::other(e.to_string()))?;
                Ok(Some(vote))
            }
            None => Ok(None),
        }
    }
}

impl RaftLogStorage<TypeConfig> for LogStore {
    type LogReader = LogStore;

    async fn get_log_state(&mut self) -> Result<LogState<TypeConfig>, io::Error> {
        // Both caches were loaded from Partition::Raft on open() — O(1).
        let last_purged_log_id = *self
            .last_purged
            .lock()
            .map_err(|_| io::Error::other("last_purged mutex poisoned"))?;
        let last_log_id = *self
            .last_log_id
            .lock()
            .map_err(|_| io::Error::other("last_log_id mutex poisoned"))?;

        // NOTE: openraft stores the Bootstrap (initial membership) entry at index 0.
        // Returning last_purged_log_id = None when no purge has happened is correct:
        // openraft will scan from index 0 to recover membership via try_get_log_entries.
        // The purge filter in try_get_log_entries uses `oe.index < min_valid_index`
        // (with min_valid_index = 0 when no purge) so entry 0 is never incorrectly skipped.

        Ok(LogState {
            last_purged_log_id,
            last_log_id,
        })
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        // Cheap clone: all fields are Arc-wrapped.
        LogStore {
            engine: self.engine.share("raft log reader"),
            oplog: Arc::clone(&self.oplog),
            last_log_id: Arc::clone(&self.last_log_id),
            last_purged: Arc::clone(&self.last_purged),
            append_notifier: Arc::clone(&self.append_notifier),
            sync: Arc::clone(&self.sync),
            hold: self.hold.another(),
        }
    }

    async fn save_vote(&mut self, vote: &Vote) -> Result<(), io::Error> {
        let bytes = rmp_serde::to_vec(vote).map_err(|e| io::Error::other(e.to_string()))?;
        // openraft answers the candidate once this returns; a vote lost in a
        // crash would let this node grant a second one in the same term. One
        // synced append to the member-local journal makes it durable, in place
        // of a flush of the whole Raft tree.
        self.engine
            .commit_local(
                vec![LocalWrite::Put {
                    partition: Partition::Raft,
                    key: KEY_VOTE.to_vec(),
                    value: bytes,
                }],
                LocalSync::BeforeUse,
            )
            .map_err(|e| io::Error::other(format!("persist the vote: {e}")))
    }

    async fn save_committed(
        &mut self,
        committed: Option<openraft::type_config::alias::LogIdOf<TypeConfig>>,
    ) -> Result<(), io::Error> {
        match committed {
            Some(log_id) => {
                let bytes =
                    rmp_serde::to_vec(&log_id).map_err(|e| io::Error::other(e.to_string()))?;
                self.put(KEY_COMMITTED, &bytes)
            }
            None => self.delete(KEY_COMMITTED),
        }
    }

    async fn read_committed(
        &mut self,
    ) -> Result<Option<openraft::type_config::alias::LogIdOf<TypeConfig>>, io::Error> {
        match self.get(KEY_COMMITTED)? {
            Some(bytes) => {
                let log_id =
                    rmp_serde::from_slice(&bytes).map_err(|e| io::Error::other(e.to_string()))?;
                Ok(Some(log_id))
            }
            None => Ok(None),
        }
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: IOFlushed<TypeConfig>,
    ) -> Result<(), io::Error>
    where
        I: IntoIterator<Item = Entry> + OptionalSend,
        I::IntoIter: OptionalSend,
    {
        let mut last: Option<LogId> = None;
        // (proposal id, log index) of every proposal in this batch, told to
        // the waiting writers only once the batch is durable.
        let mut appended: Vec<(coordinode_core::txn::proposal::ProposalId, u64)> = Vec::new();
        let handle = {
            let mut oplog = self
                .oplog
                .lock()
                .map_err(|_| io::Error::other("oplog mutex poisoned"))?;
            for entry in entries {
                let oplog_entry = Self::entry_to_oplog(&entry)?;
                if let openraft::entry::EntryPayload::Normal(request) = &entry.payload {
                    appended.extend(request.proposals.iter().map(|p| (p.id, entry.log_id.index)));
                }
                last = Some(entry.log_id);
                oplog
                    .append(&oplog_entry)
                    .map_err(|e| io::Error::other(e.to_string()))?;
            }
            // Written to the kernel, so a read sees the entries at once; the
            // sync thread makes them durable, and only then is openraft told
            // (the crash-safety boundary) and the writers answered. The Raft
            // core meanwhile replicates the batch instead of waiting on the
            // disk, and one sync covers every batch appended while it runs.
            oplog
                .flush_to_os()
                .map_err(|e| io::Error::other(e.to_string()))?
        };

        if let Some(log_id) = last {
            *self
                .last_log_id
                .lock()
                .map_err(|_| io::Error::other("last_log_id mutex poisoned"))? = Some(log_id);
        }
        // A writer under a volatile `j` is answered now, at the write; the
        // commit still waits for the sync below, as every commit does.
        self.append_notifier.notify_written(&appended);
        self.sync.submit(
            handle,
            crate::log_sync::Append {
                callback,
                proposals: appended,
            },
        );
        Ok(())
    }

    async fn truncate_after(
        &mut self,
        last_log_id: Option<openraft::type_config::alias::LogIdOf<TypeConfig>>,
    ) -> Result<(), io::Error> {
        let keep_exclusive = last_log_id.map_or(0, |id| id.index + 1);

        // No append may be answered after the entries it covered are gone.
        self.sync.wait_idle();

        // Cut the suffix in place: the kept entries are never rewritten, so a
        // crash midway cannot take any of them (openraft counts on a durable
        // prefix through every step of a truncation).
        self.oplog
            .lock()
            .map_err(|_| io::Error::other("oplog mutex poisoned"))?
            .truncate_from(keep_exclusive)
            .map_err(|e| io::Error::other(e.to_string()))?;

        *self
            .last_log_id
            .lock()
            .map_err(|_| io::Error::other("last_log_id mutex poisoned"))? = last_log_id;

        tracing::debug!(
            keep_through = last_log_id.map(|id| id.index),
            "raft log truncated after"
        );

        Ok(())
    }

    async fn purge(
        &mut self,
        log_id: openraft::type_config::alias::LogIdOf<TypeConfig>,
    ) -> Result<(), io::Error> {
        // openraft may forget an entry once applied, but the entry is the
        // only copy of whatever a tree has not flushed yet. Purge only below
        // the entries every tree durably records as applied, whatever openraft
        // asked for; the rest stays and a later purge takes it.
        let floor = self
            .engine
            .raft_durable_floor()
            .map_err(|e| io::Error::other(format!("raft coverage floor: {e}")))?;
        let below = floor.min(log_id.index + 1);
        // A floor at or below what an earlier purge already removed (a later
        // checkpoint can lower it) leaves nothing to remove: the purge point
        // never moves back, and the entry below it is gone already.
        let purged_below = self
            .last_purged
            .lock()
            .map_err(|_| io::Error::other("last_purged mutex poisoned"))?
            .map_or(0, |id| id.index + 1);
        if below <= purged_below {
            return Ok(());
        }
        let purged_to = if below == log_id.index + 1 {
            log_id
        } else {
            // The id of the last entry actually dropped, read before it goes.
            let last = self
                .oplog
                .lock()
                .map_err(|_| io::Error::other("oplog mutex poisoned"))?
                .read_range(below - 1, below)
                .map_err(|e| io::Error::other(e.to_string()))?
                .pop()
                .ok_or_else(|| io::Error::other(format!("raft log entry {} missing", below - 1)))?;
            Self::oplog_to_entry(last)?.log_id
        };

        // Recorded durably before any segment goes: a crash between the two
        // then leaves segments the record says are purged (skipped on read),
        // never a record that promises entries already deleted. A synced
        // append to the member-local journal makes it durable, in place of a
        // flush of the whole Raft tree.
        let bytes = rmp_serde::to_vec(&purged_to).map_err(|e| io::Error::other(e.to_string()))?;
        self.engine
            .commit_local(
                vec![LocalWrite::Put {
                    partition: Partition::Raft,
                    key: KEY_PURGED.to_vec(),
                    value: bytes,
                }],
                LocalSync::BeforeUse,
            )
            .map_err(|e| io::Error::other(format!("persist the purge point: {e}")))?;
        *self
            .last_purged
            .lock()
            .map_err(|_| io::Error::other("last_purged mutex poisoned"))? = Some(purged_to);

        let purged_segments = self
            .oplog
            .lock()
            .map_err(|_| io::Error::other("oplog mutex poisoned"))?
            .purge_before(below)
            .map_err(|e| io::Error::other(e.to_string()))?;

        tracing::debug!(
            requested = log_id.index,
            purged_up_to = purged_to.index,
            segments_removed = purged_segments,
            "raft log purged"
        );

        Ok(())
    }
}

// ── State Machine ───────────────────────────────────────────────────

/// Dedup entry for tracking applied proposals.
///
/// Stores the proposal size estimate and last-seen timestamp.
/// Used to detect duplicate proposals from Raft replay after leader change.
#[derive(Debug)]
struct DedupEntry {
    /// Approximate proposal size (for double-checking retried proposals).
    size: usize,
    /// When this proposal was last seen (for GC).
    seen: Instant,
}

/// The version pairs a group recorded, oldest first. A handful over the life
/// of a group: one per version move.
pub type GroupPairs = Arc<[coordinode_core::version::RecordedPair]>;

/// Storage-backed Raft state machine.
///
/// Applies committed Raft entries (mutations) to the database via StorageEngine.
/// Tracks last-applied log id and membership in the Schema partition.
/// Includes proposal dedup tracking to handle Raft replay idempotently.
///
/// ## Applied Watermark
///
/// The state machine broadcasts the applied log index via a `watch` channel
/// after each entry is applied. Readers can subscribe to wait for a specific
/// index, enabling linearizable reads (follower waits until `Applied >= readTs`)
/// and snapshot trigger decisions.
pub struct CoordinodeStateMachine {
    engine: crate::engine_hold::EngineHold,
    /// Timestamp oracle advanced during Raft apply.
    ///
    /// Every entry is applied at its `commit_ts` as one batch (the engine
    /// stamps the seqno; see `StorageEngine::apply_proposal_at`), and the
    /// oracle is advanced to that `commit_ts` afterwards. This ensures:
    /// - Raft replay produces identical seqnos as original application
    /// - snapshot_at(commit_ts) sees the whole entry, snapshot_at(commit_ts - 1) none of it
    /// - a follower promoted to leader never allocates a timestamp it already applied
    oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
    /// Last applied log id, cached in memory for fast access.
    last_applied: Mutex<Option<openraft::type_config::alias::LogIdOf<TypeConfig>>>,
    /// Last applied membership, cached in memory.
    last_membership: Mutex<StoredMembership>,
    /// Dedup map: proposal_id → (size, last_seen). Entries older than
    /// `DEDUP_MAX_AGE_SECS` are GC'd periodically. Serial access only
    /// (openraft calls `apply` serially).
    dedup: Mutex<HashMap<u64, DedupEntry>>,
    /// Last time the dedup map was GC'd.
    last_dedup_gc: Mutex<Instant>,
    /// Applied watermark: broadcasts the latest applied log index.
    /// Subscribers can wait for a specific index to be applied.
    applied_tx: tokio::sync::watch::Sender<u64>,
    /// Receiver side kept to prevent channel closure.
    applied_rx: tokio::sync::watch::Receiver<u64>,
    /// Every version pair the group recorded in its applied state, oldest
    /// first, republished whenever an applied entry or an installed snapshot
    /// moves it.
    group_pair_tx: tokio::sync::watch::Sender<GroupPairs>,
    /// The deployment's identity as the applied state records it, `None`
    /// before the founding leader's record applies here.
    deployment_tx: tokio::sync::watch::Sender<Option<coordinode_core::group::DeploymentId>>,
    /// The highest commit timestamp applied here: what a member that stops
    /// receiving entries serves its reads as of.
    applied_commit_ts: Arc<core::sync::atomic::AtomicU64>,
    /// Per-shard `maxAssigned` watermark over HLC commit_ts.
    /// Distinct from `applied_tx` (Raft log index). Advanced after every
    /// successful proposal apply so snapshot readers can `WaitForTs(T)`
    /// until every write with `commit_ts ≤ T` has been applied to every
    /// modality on this shard.
    ///
    /// Optional so legacy paths that don't need cross-modality snapshot
    /// semantics can leave it `None` (e.g. embedded-mode fast tests).
    max_assigned: Option<Arc<coordinode_core::txn::watermark::MaxAssignedWatermark>>,
    /// Count of snapshot builds performed by this state machine.
    /// Observability for the snapshot trigger: every build flushes every
    /// tree's memtables, so redundant rebuilds are worth monitoring.
    snapshot_builds: Arc<core::sync::atomic::AtomicU64>,
    /// What each partition tree held when the state machine opened. Entries
    /// below [`Self::skip_until`] are re-delivered from the lowest covered
    /// prefix, and a tree that already holds a proposal is left alone; past
    /// that point nothing needs checking and this is dropped.
    replay_skip: Option<RaftCoverage>,
    skip_until: u64,
    /// The coverage base last written to every tree: all entries below it.
    folded: u64,
    /// Where the applies stand, held while applying; registered with the
    /// engine as its Raft apply fence.
    gate: Arc<RaftApplyGate>,
    /// Snapshot work holding the engine off the async runtime, which a
    /// shutdown waits out.
    engine_work: EngineWork,
    /// This state machine's own share of `engine_work`: it holds the engine
    /// for as long as openraft keeps it. Declared last, so it is dropped after
    /// every field holding the engine: a shutdown waiting on it would
    /// otherwise return while `engine` still held the directory.
    _hold: EngineWorkGuard,
}

/// Everything that holds the engine on openraft's behalf: the log store and
/// each reader of it, the state machine, and the work the state machine runs
/// outside openraft's tasks (a snapshot capture on a blocking thread, a
/// snapshot build).
///
/// Shutting openraft down stops its core, but not everything it spawned: a
/// replication task waiting on a peer keeps its log reader until that call
/// returns, a blocking thread runs to its end, and a build task can outlive
/// the core. Until they finish they hold the engine, so the directory stays
/// locked and a restart in the same process cannot open it. The node waits
/// for this to go idle after the consensus stops.
#[derive(Clone, Default)]
pub struct EngineWork(Arc<EngineWorkInner>);

#[derive(Default)]
struct EngineWorkInner {
    running: core::sync::atomic::AtomicUsize,
    idle: tokio::sync::Notify,
}

/// One piece of [`EngineWork`] under way; finishes on drop.
pub struct EngineWorkGuard(Arc<EngineWorkInner>);

impl EngineWork {
    pub(crate) fn start(&self) -> EngineWorkGuard {
        self.0
            .running
            .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        EngineWorkGuard(Arc::clone(&self.0))
    }

    /// Wait until no work holds the engine, for at most `timeout`. `false`
    /// when some was still running at the deadline.
    pub async fn wait_idle(&self, timeout: std::time::Duration) -> bool {
        let wait = async {
            loop {
                let notified = self.0.idle.notified();
                if self.0.running.load(core::sync::atomic::Ordering::SeqCst) == 0 {
                    return;
                }
                notified.await;
            }
        };
        tokio::time::timeout(timeout, wait).await.is_ok()
    }
}

impl EngineWorkGuard {
    /// A second guard on the same work, for a clone of whatever holds this one.
    fn another(&self) -> Self {
        self.0
            .running
            .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        Self(Arc::clone(&self.0))
    }

    fn work(&self) -> EngineWork {
        EngineWork(Arc::clone(&self.0))
    }
}

impl Drop for EngineWorkGuard {
    fn drop(&mut self) {
        if self
            .0
            .running
            .fetch_sub(1, core::sync::atomic::Ordering::SeqCst)
            == 1
        {
            self.0.idle.notify_waiters();
        }
    }
}

/// This node's Raft apply position, locked by the state machine for every
/// batch it applies, so a partition can be copied or replaced at an exact
/// log position with the applies paused.
pub struct RaftApplyGate {
    state: tokio::sync::Mutex<GateState>,
}

struct GateState {
    applies: RaftApplyState,
    /// The last applied entry, whose log id a coverage base carries.
    last: Option<openraft::type_config::alias::LogIdOf<TypeConfig>>,
}

impl RaftApplyFence for RaftApplyGate {
    fn with_applies_paused(
        &self,
        work: &mut dyn FnMut(&mut RaftApplyState, &[u8]) -> StorageResult<()>,
    ) -> StorageResult<()> {
        let mut state = self.state.blocking_lock();
        let payload = match &state.last {
            Some(id) => {
                rmp_serde::to_vec(id).map_err(|e| StorageError::Serialization(e.to_string()))?
            }
            None => Vec::new(),
        };
        work(&mut state.applies, &payload)
    }
}

/// How long a snapshot build sleeps before it persists the store, so a test
/// can shut a node down while one is under way.
#[cfg(test)]
pub(crate) static BUILD_DELAY_MS: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);

/// The consensus state stored at `key` in `partition`, `None` when nothing
/// is. A read or decode failure is an error, never an absent value: the
/// state it holds (a purge point, the applied membership) cannot be guessed.
fn load_stored<T: serde::de::DeserializeOwned>(
    engine: &StorageEngine,
    partition: Partition,
    key: &[u8],
) -> Result<Option<T>, io::Error> {
    let what = String::from_utf8_lossy(key);
    let Some(bytes) = engine
        .get(partition, key)
        .map_err(|e| io::Error::other(format!("read {what}: {e}")))?
    else {
        return Ok(None);
    };
    rmp_serde::from_slice(&bytes)
        .map(Some)
        .map_err(|e| io::Error::other(format!("decode {what}: {e}")))
}

/// How many applied entries accumulate as coverage markers before the state
/// machine folds them into every tree's base. The crate's own tests fold
/// often, so their short workloads cross fold boundaries.
#[cfg(not(test))]
const RAFT_FOLD_EVERY: u64 = 4096;
#[cfg(test)]
const RAFT_FOLD_EVERY: u64 = 5;

impl CoordinodeStateMachine {
    /// Open the state machine over `engine`; see
    /// [`Self::with_oracle_and_watermark`] for the errors.
    pub fn new(engine: Arc<StorageEngine>) -> Result<Self, io::Error> {
        Self::with_oracle(engine, None)
    }

    /// Create with a timestamp oracle for seqno advancement.
    ///
    /// When oracle is set, `apply_proposal()` calls `oracle.advance_to(commit_ts)`
    /// after applying the entry at `commit_ts`, so later allocations are newer.
    pub fn with_oracle(
        engine: Arc<StorageEngine>,
        oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
    ) -> Result<Self, io::Error> {
        Self::with_oracle_and_watermark(engine, oracle, None)
    }

    /// Create with a timestamp oracle AND a `MaxAssignedWatermark`.
    ///
    /// When the watermark is set, `apply_proposal()` advances it to the
    /// proposal's `commit_ts` AFTER every mutation in that proposal has
    /// been persisted to the storage engine. Readers at snapshot_ts T can
    /// then `watermark.wait_for(T, timeout)` to block until every write
    /// with `commit_ts ≤ T` is visible on this shard.
    ///
    /// The applied position is read from the partition trees' Raft coverage
    /// records, not from a key of its own: each tree flushes on its own
    /// schedule, so the position every tree holds is the lowest of their
    /// bases, and openraft re-delivers from there.
    ///
    /// # Errors
    ///
    /// A store that applied Raft entries under a release without coverage
    /// records is refused: nothing in it proves which entries each tree
    /// holds. A failed coverage read, an unreadable applied membership or the
    /// first record's flush also fails the open.
    pub fn with_oracle_and_watermark(
        engine: Arc<StorageEngine>,
        oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
        max_assigned: Option<Arc<coordinode_core::txn::watermark::MaxAssignedWatermark>>,
    ) -> Result<Self, io::Error> {
        let coverage = engine
            .raft_coverage()
            .map_err(|e| io::Error::other(format!("read raft coverage: {e}")))?;
        let (last_applied, replay_skip) = match coverage.resume_point() {
            Some((_, payload)) => {
                let last_applied: Option<openraft::type_config::alias::LogIdOf<TypeConfig>> =
                    if payload.is_empty() {
                        None
                    } else {
                        Some(rmp_serde::from_slice(payload).map_err(|e| {
                            io::Error::other(format!("raft coverage base log id: {e}"))
                        })?)
                    };
                (last_applied, Some(coverage))
            }
            None if Self::load_log_id(&engine, KEY_SM_APPLIED)?.is_some() => {
                return Err(io::Error::other(
                    "this store applied Raft entries without an apply-coverage record, so \
                     nothing proves which of them each partition holds; it was written by an \
                     earlier release, which this one does not open",
                ));
            }
            None => {
                engine
                    .reset_raft_coverage(0, &[])
                    .map_err(|e| io::Error::other(format!("establish raft coverage: {e}")))?;
                (None, None)
            }
        };
        // The closed bound the trees hold; the entries replayed above the
        // resume point raise it again as they apply.
        engine
            .reload_closure_frontier()
            .map_err(|e| io::Error::other(format!("read the closed bound: {e}")))?;
        if let (Some(oracle), Some(at)) = (&oracle, engine.closure_frontier().checked_sub(1)) {
            oracle.advance_to(coordinode_core::txn::timestamp::Timestamp::from_raw(at));
        }
        // A snapshot a crash left half received installs nowhere.
        crate::snapshot::clear_snapshot_dir(&engine)
            .map_err(|e| io::Error::other(format!("clear the snapshot directory: {e}")))?;
        // The trees may still hold markers below their bases; the first fold
        // removes markers from index 0.
        let folded = 0;
        let skip_until = replay_skip.as_ref().map_or(0, RaftCoverage::skip_until);
        let last_membership = Self::load_membership(&engine)?;

        // Initialize applied watermark from the resume point.
        let initial_index = last_applied.map(|id| id.index).unwrap_or(0);
        let (applied_tx, applied_rx) = tokio::sync::watch::channel(initial_index);

        let gate = Arc::new(RaftApplyGate {
            state: tokio::sync::Mutex::new(GateState {
                applies: RaftApplyState::new(last_applied.map_or(0, |id| id.index + 1)),
                last: last_applied,
            }),
        });
        engine.register_raft_fence(Arc::clone(&gate) as Arc<dyn RaftApplyFence>);
        let engine_work = EngineWork::default();
        let group_pairs = coordinode_storage::engine::metadata::group_pair_records(&engine)
            .map_err(|e| io::Error::other(format!("read the group's version pairs: {e}")))?;
        let (group_pair_tx, _) = tokio::sync::watch::channel(Arc::from(group_pairs));
        let deployment = coordinode_storage::engine::metadata::recorded_deployment(&engine)
            .map_err(|e| io::Error::other(format!("read the deployment record: {e}")))?;
        let (deployment_tx, _) = tokio::sync::watch::channel(deployment);
        // Until an entry applies, what the store holds is as of the clock it
        // reopened at.
        let reopened_at = engine.oracle().map_or(0, |o| o.current().as_raw());

        Ok(Self {
            engine: crate::engine_hold::EngineHold::new(engine, "state machine"),
            oracle,
            last_applied: Mutex::new(last_applied),
            last_membership: Mutex::new(last_membership),
            dedup: Mutex::new(HashMap::new()),
            last_dedup_gc: Mutex::new(Instant::now()),
            applied_tx,
            applied_rx,
            group_pair_tx,
            deployment_tx,
            applied_commit_ts: Arc::new(core::sync::atomic::AtomicU64::new(reopened_at)),
            max_assigned,
            snapshot_builds: Arc::new(core::sync::atomic::AtomicU64::new(0)),
            replay_skip,
            skip_until,
            folded,
            gate,
            _hold: engine_work.start(),
            engine_work,
        })
    }

    /// Count this state machine and its snapshot work in `work`, the log
    /// store's (see [`LogStore::engine_work`]), so a node waits on one handle
    /// for everything openraft holds the engine through.
    pub fn with_engine_work(mut self, work: EngineWork) -> Self {
        self._hold = work.start();
        self.engine_work = work;
        self
    }

    /// Handle to the snapshot-build counter (increments on every full
    /// snapshot serialization). Cluster orchestration exposes it for
    /// metrics and tests.
    pub fn snapshot_builds_handle(&self) -> Arc<core::sync::atomic::AtomicU64> {
        Arc::clone(&self.snapshot_builds)
    }

    /// Handle to the snapshot work holding the engine, which a node's
    /// shutdown waits out.
    pub fn engine_work_handle(&self) -> EngineWork {
        self.engine_work.clone()
    }

    /// Handle to the per-shard `maxAssigned` watermark, if one was wired
    /// in at construction. Query executors hold this to drive
    /// `read_consistency='snapshot'` waits.
    pub fn max_assigned(
        &self,
    ) -> Option<Arc<coordinode_core::txn::watermark::MaxAssignedWatermark>> {
        self.max_assigned.clone()
    }

    /// Subscribe to the applied watermark.
    ///
    /// Returns a `watch::Receiver<u64>` that yields the latest applied
    /// log index whenever a new entry is applied. Use `changed().await`
    /// to wait for updates, or `borrow()` to read the current value.
    ///
    /// Used by:
    /// - Follower reads: wait until `Applied >= query.readTs`
    /// - Snapshot decisions: check entries since last snapshot
    /// - Health monitoring: detect how far behind this node is
    pub fn subscribe_applied(&self) -> tokio::sync::watch::Receiver<u64> {
        self.applied_rx.clone()
    }

    /// Get the current applied log index (non-blocking).
    pub fn applied_index(&self) -> u64 {
        *self.applied_rx.borrow()
    }

    /// Subscribe to the version pairs the group recorded, as applied here.
    pub fn subscribe_group_pair(&self) -> tokio::sync::watch::Receiver<GroupPairs> {
        self.group_pair_tx.subscribe()
    }

    /// Handle to the highest commit timestamp applied here.
    pub fn applied_commit_ts_handle(&self) -> Arc<core::sync::atomic::AtomicU64> {
        Arc::clone(&self.applied_commit_ts)
    }

    /// Subscribe to the deployment's identity, as applied here.
    pub fn subscribe_deployment(
        &self,
    ) -> tokio::sync::watch::Receiver<Option<coordinode_core::group::DeploymentId>> {
        self.deployment_tx.subscribe()
    }

    /// Reread the deployment record after the applied state may have set it.
    fn refresh_deployment(&self) -> Result<(), io::Error> {
        let recorded = coordinode_storage::engine::metadata::recorded_deployment(&self.engine)
            .map_err(|e| io::Error::other(format!("read the deployment record: {e}")))?;
        self.deployment_tx.send_if_modified(|current| {
            let moved = *current != recorded;
            *current = recorded;
            moved
        });
        Ok(())
    }

    /// Reread the group's recorded pair after the applied state moved it.
    fn refresh_group_pair(&self) -> Result<(), io::Error> {
        let pairs = coordinode_storage::engine::metadata::group_pair_records(&self.engine)
            .map_err(|e| io::Error::other(format!("read the group's version pairs: {e}")))?;
        self.group_pair_tx.send_if_modified(|current| {
            let moved = **current != *pairs;
            if moved {
                *current = Arc::from(pairs);
            }
            moved
        });
        Ok(())
    }

    fn load_log_id(
        engine: &StorageEngine,
        key: &[u8],
    ) -> Result<Option<openraft::type_config::alias::LogIdOf<TypeConfig>>, io::Error> {
        debug_assert!(
            key.starts_with(b"raft:"),
            "Raft state machine keys in Schema partition must use raft: prefix, got {:?}",
            String::from_utf8_lossy(key)
        );
        load_stored(engine, Partition::Schema, key)
    }

    /// The membership the state machine last applied, the initial one when
    /// none was. A stored one that cannot be read or decoded fails the open:
    /// taken as the initial membership, the member would forget its group.
    fn load_membership(engine: &StorageEngine) -> Result<StoredMembership, io::Error> {
        Ok(load_stored(engine, Partition::Schema, KEY_SM_MEMBERSHIP)?.unwrap_or_default())
    }

    /// Record every node identity that `previous` binds and `next` no longer
    /// does as retired: a member that left is never admitted again under
    /// it, whatever certificate for it is still about. Decided on applying
    /// the membership entry, so every member records the same retirements
    /// at the same point of the log.
    fn retire_removed<'a>(
        &self,
        previous: impl Iterator<Item = &'a crate::member::MemberNode>,
        next: impl Iterator<Item = &'a crate::member::MemberNode>,
    ) -> Result<(), io::Error> {
        let kept: Vec<&crate::member::MemberIdentity> =
            next.filter_map(|node| node.identity.as_ref()).collect();
        for identity in previous
            .filter_map(|node| node.identity.as_ref())
            .filter(|identity| !kept.contains(identity))
        {
            tracing::info!(%identity, "a member left: its node identity is retired");
            self.engine
                .put(
                    Partition::Schema,
                    &coordinode_core::group::retired_identity_key(&identity.canonical()),
                    &[],
                )
                .map_err(|e| io::Error::other(format!("retire {identity}: {e}")))?;
        }
        Ok(())
    }

    fn save_membership(&self, membership: &StoredMembership) -> Result<(), io::Error> {
        let bytes = rmp_serde::to_vec(membership).map_err(|e| io::Error::other(e.to_string()))?;
        self.engine
            .put(Partition::Schema, KEY_SM_MEMBERSHIP, &bytes)
            .map_err(|e| io::Error::other(e.to_string()))
    }

    /// Apply a single proposal's mutations to storage (native seqno MVCC).
    ///
    /// The whole proposal is applied as one batch at its `commit_ts` seqno.
    /// No versioned key encoding.
    ///
    /// Includes dedup check: if this proposal ID was already applied with
    /// the same size estimate, skip re-application (idempotent Raft replay).
    ///
    /// A partition `applies` says is already past `index` (installed from a
    /// peer further along) is left alone.
    fn apply_proposal_under(
        &self,
        proposal: &RaftProposal,
        index: u64,
        sub: u32,
        applies: &RaftApplyState,
    ) -> Result<Response, io::Error> {
        let proposal_key = proposal.id.as_raw();
        let proposal_size = proposal.size_estimate();
        // A unit bound to a client operation is decided by its binding, which
        // is durable and marks a refused unit for raw readers of the log; the
        // process-local filter below would drop a retransmission unmarked.
        let bound = matches!(proposal.mutations.last(), Some(Mutation::Operation(_)));

        // Dedup check: same key + same size = already applied
        if !bound {
            let mut dedup = self
                .dedup
                .lock()
                .map_err(|e| io::Error::other(format!("dedup mutex poisoned: {e}")))?;

            if let Some(entry) = dedup.get_mut(&proposal_key) {
                if entry.size == proposal_size {
                    // Exact duplicate — skip application, update timestamp
                    entry.seen = Instant::now();
                    tracing::debug!(
                        proposal_id = proposal_key,
                        "duplicate proposal detected, skipping apply"
                    );
                    return Ok(Response {
                        mutations_applied: 0,
                    });
                }
                // Different size = different retry payload, re-apply
            }
        }

        // Apply the whole entry at ONE seqno, its commit_ts: a
        // snapshot at commit_ts sees every mutation of the entry, a snapshot
        // one tick earlier sees none, on the leader and on every follower
        // alike. The engine advances its own generator past commit_ts; the
        // state machine's oracle is advanced too so a follower promoted to
        // leader never allocates a timestamp at or below what it applied.
        //
        // Each touched tree records `(index, sub)` in the same batch as the
        // proposal's effects. Inside the replay window a tree that already
        // holds the proposal is skipped, so a re-delivered merge is never
        // applied twice.
        let replay = self
            .replay_skip
            .as_ref()
            .filter(|_| index < self.skip_until);
        let count = self
            .engine
            .apply_raft_proposal(
                &proposal.mutations,
                proposal.commit_ts.as_raw(),
                index,
                sub,
                |part| {
                    applies.skips(part, index) || replay.is_some_and(|c| c.holds(part, index, sub))
                },
            )
            .map_err(|e| io::Error::other(e.to_string()))?;
        if let Some(ref oracle) = self.oracle {
            if proposal.commit_ts.as_raw() > 0 {
                oracle.advance_to(proposal.commit_ts);
            }
        }

        // Record in dedup map
        if !bound {
            let mut dedup = self
                .dedup
                .lock()
                .map_err(|e| io::Error::other(format!("dedup mutex poisoned: {e}")))?;
            dedup.insert(
                proposal_key,
                DedupEntry {
                    size: proposal_size,
                    seen: Instant::now(),
                },
            );
        }

        // Periodic dedup GC (every DEDUP_GC_INTERVAL_SECS)
        self.maybe_gc_dedup();

        // Advance the per-shard `maxAssigned` watermark AFTER every
        // mutation in this proposal has been persisted. Snapshot readers at
        // T ≤ commit_ts now see a fully-applied state on this shard.
        // Monotonic — a proposal whose commit_ts is below the current
        // watermark is a no-op (shouldn't happen in a healthy cluster; HLC
        // guarantees per-shard monotonicity — but defensive).
        if let Some(ref wm) = self.max_assigned {
            if proposal.commit_ts.as_raw() > 0 {
                wm.advance(proposal.commit_ts);
            }
        }

        Ok(Response {
            mutations_applied: count,
        })
    }

    /// Record `closed_below`, the bound entry `index` carries, after its
    /// proposals: every commit stamped below it applied here before it.
    ///
    /// The record lands as position `sub` of the entry, with its coverage
    /// marker like a proposal, so a replay skips it where the Schema tree
    /// holds it and a snapshot carries it. It is written at `closed_below -
    /// 1`, below every commit the bound leaves open and above every earlier
    /// bound. The oracle is raised to the same value, so this node, once it
    /// leads, stamps nothing at or below a bound a past leader handed out.
    fn apply_closure(
        &self,
        closed_below: u64,
        index: u64,
        sub: u32,
        applies: &RaftApplyState,
    ) -> Result<(), io::Error> {
        // `closed_below > 0` at the caller.
        let at = closed_below - 1;
        let replay = self
            .replay_skip
            .as_ref()
            .filter(|_| index < self.skip_until);
        self.engine
            .apply_raft_proposal(
                &[Mutation::Put {
                    partition: coordinode_core::txn::proposal::PartitionId::Schema,
                    key: coordinode_storage::engine::core::CLOSURE_KEY.to_vec(),
                    value: coordinode_storage::engine::core::encode_closure(closed_below),
                }],
                at,
                index,
                sub,
                |part| {
                    applies.skips(part, index) || replay.is_some_and(|c| c.holds(part, index, sub))
                },
            )
            .map_err(|e| io::Error::other(format!("record the closed bound: {e}")))?;
        if let Some(ref oracle) = self.oracle {
            oracle.advance_to(coordinode_core::txn::timestamp::Timestamp::from_raw(at));
        }
        self.engine.raise_closure_frontier(closed_below, index);
        Ok(())
    }

    /// The log index openraft re-delivers from when this state machine opens:
    /// one past the position every partition tree is known to hold.
    pub fn next_to_apply(&self) -> Result<u64, io::Error> {
        let last = *self
            .last_applied
            .lock()
            .map_err(|_| io::Error::other("last_applied mutex poisoned"))?;
        Ok(last.map_or(0, |id| id.index + 1))
    }

    /// Whether the membership this state machine applied names `node_id`
    /// beside another node: the store of a member of a group.
    pub fn applied_membership_names(&self, node_id: u64) -> Result<bool, io::Error> {
        let stored = self
            .last_membership
            .lock()
            .map_err(|_| io::Error::other("last_membership mutex poisoned"))?;
        Ok(names_beside_another(stored.membership(), node_id))
    }

    /// [`Self::apply_proposal_under`] with no partition installed ahead.
    #[cfg(test)]
    fn apply_proposal(
        &self,
        proposal: &RaftProposal,
        index: u64,
        sub: u32,
    ) -> Result<Response, io::Error> {
        self.apply_proposal_under(proposal, index, sub, &RaftApplyState::default())
    }

    /// Run dedup GC if enough time has passed since last run.
    /// Removes entries older than `DEDUP_MAX_AGE_SECS`.
    fn maybe_gc_dedup(&self) {
        let now = Instant::now();
        let gc_interval = Duration::from_secs(DEDUP_GC_INTERVAL_SECS);
        let max_age = Duration::from_secs(DEDUP_MAX_AGE_SECS);

        // Check if GC is due (non-blocking: skip if lock is contended)
        let should_gc = self
            .last_dedup_gc
            .lock()
            .map(|last| now.duration_since(*last) >= gc_interval)
            .unwrap_or(false);

        if !should_gc {
            return;
        }

        // Update GC timestamp
        if let Ok(mut last) = self.last_dedup_gc.lock() {
            *last = now;
        }

        // Evict old entries
        if let Ok(mut dedup) = self.dedup.lock() {
            let before = dedup.len();
            dedup.retain(|_, entry| now.duration_since(entry.seen) < max_age);
            let evicted = before - dedup.len();
            if evicted > 0 {
                tracing::debug!(evicted, remaining = dedup.len(), "dedup map GC complete");
            }
        }
    }
}

impl RaftStateMachine<TypeConfig> for CoordinodeStateMachine {
    // A snapshot is a file in the engine's snapshot directory, never a
    // buffer: it is as large as the store.
    type SnapshotData = SnapshotData;

    type SnapshotBuilder = CoordinodeSnapshotBuilder;

    async fn applied_state(
        &mut self,
    ) -> Result<
        (
            Option<openraft::type_config::alias::LogIdOf<TypeConfig>>,
            StoredMembership,
        ),
        io::Error,
    > {
        let applied = *self
            .last_applied
            .lock()
            .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))?;
        let membership = self
            .last_membership
            .lock()
            .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))?
            .clone();
        Ok((applied, membership))
    }

    async fn apply<Strm>(&mut self, entries: Strm) -> Result<(), io::Error>
    where
        Strm: futures_util::Stream<
                Item = Result<openraft::storage::EntryResponder<TypeConfig>, io::Error>,
            > + Unpin
            + OptionalSend,
    {
        let mut stream = entries;
        // Held for the batch: a partition copied or replaced with the applies
        // paused sees none of it half done.
        let gate = Arc::clone(&self.gate);
        let mut gate = gate.state.lock().await;
        while let Some(item) = stream.next().await {
            let (entry, responder) = item?;

            // Membership changes are applied before normal entries (pure metadata,
            // no tree mutations, no crash-consistency concern with ordering).
            if let Some(ref membership) = entry.get_membership() {
                let stored: StoredMembership =
                    openraft::StoredMembership::new(Some(entry.log_id), membership.clone());
                let previous = self
                    .last_membership
                    .lock()
                    .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))?
                    .clone();
                // Before the membership itself: one partition's writes land
                // in order, so a membership that survives a crash has its
                // retirements beside it.
                self.retire_removed(
                    previous.membership().nodes().map(|(_, node)| node),
                    membership.nodes().map(|(_, node)| node),
                )?;
                *self
                    .last_membership
                    .lock()
                    .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))? =
                    stored.clone();
                self.save_membership(&stored)?;
            }

            // Each proposal lands with its coverage marker in every tree it
            // touches; there is no separate "applied index" key, because a
            // key in one tree says nothing about what another tree flushed.
            // After a crash openraft re-delivers from the lowest covered
            // prefix and the markers keep every tree at exactly-once.
            let index = entry.log_id.index;
            let response = match &entry.payload {
                openraft::entry::EntryPayload::Normal(request) => {
                    let mut total = 0;
                    let mut records_pair = false;
                    let mut records_deployment = false;
                    for (sub, proposal) in request.proposals.iter().enumerate() {
                        let sub = u32::try_from(sub).map_err(|_| {
                            io::Error::other(format!("entry {index} carries over 2^32 proposals"))
                        })?;
                        let r = self.apply_proposal_under(proposal, index, sub, &gate.applies)?;
                        total += r.mutations_applied;
                        self.applied_commit_ts.fetch_max(
                            proposal.commit_ts.as_raw(),
                            core::sync::atomic::Ordering::Release,
                        );
                        records_pair |= proposal.mutations.iter().any(|m| {
                            matches!(
                                m,
                                Mutation::Command(MetadataCommand::RecordGroupPair { .. })
                            )
                        });
                        records_deployment |= proposal.mutations.iter().any(|m| {
                            matches!(
                                m,
                                Mutation::Command(MetadataCommand::RecordDeployment { .. })
                            )
                        });
                    }
                    if records_pair {
                        self.refresh_group_pair()?;
                    }
                    if records_deployment {
                        self.refresh_deployment()?;
                    }
                    if request.closed_below > 0 {
                        let sub = u32::try_from(request.proposals.len()).map_err(|_| {
                            io::Error::other(format!("entry {index} carries over 2^32 proposals"))
                        })?;
                        self.apply_closure(request.closed_below, index, sub, &gate.applies)?;
                    }
                    Response {
                        mutations_applied: total,
                    }
                }
                _ => Response {
                    mutations_applied: 0,
                },
            };

            *self
                .last_applied
                .lock()
                .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))? =
                Some(entry.log_id);
            gate.applies.applied(index);
            gate.last = Some(entry.log_id);
            let next = index + 1;
            debug_assert!(self.folded <= next, "fold ahead of the applied entry");
            if next >= self.skip_until {
                self.replay_skip = None;
            }
            if next - self.folded >= RAFT_FOLD_EVERY {
                let payload = rmp_serde::to_vec(&entry.log_id)
                    .map_err(|e| io::Error::other(e.to_string()))?;
                let applies = &gate.applies;
                self.engine
                    .fold_raft_coverage(self.folded, next, &payload, |part| {
                        applies.floor(part) >= next
                    });
                self.folded = next;
            }

            // Broadcast applied watermark (ignore send error — receivers may be dropped)
            let _ = self.applied_tx.send(entry.log_id.index);

            // Send response back to the caller
            if let Some(tx) = responder {
                openraft::storage::ApplyResponder::send(tx, response);
            }
        }

        Ok(())
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        // Mutex poisoning can only happen if another thread panicked
        // while holding the lock. In that case, the entire node is
        // in an unrecoverable state. Panicking here is acceptable.
        #[allow(clippy::unwrap_used)]
        let last_applied = *self.last_applied.lock().unwrap();
        #[allow(clippy::unwrap_used)]
        let last_membership = self.last_membership.lock().unwrap().clone();

        // Every tree holds the applied entries now, and the build flushes
        // them together with this fold before openraft may purge the log
        // below it. A store reopened after the build therefore resumes at or
        // past it, so the purged entries are never needed again.
        let folded = match last_applied {
            Some(log_id) if log_id.index + 1 > self.folded => {
                let next = log_id.index + 1;
                let gate = self.gate.state.lock().await;
                let fold = rmp_serde::to_vec(&log_id).map(|payload| {
                    self.engine
                        .fold_raft_coverage(self.folded, next, &payload, |part| {
                            gate.applies.floor(part) >= next
                        });
                });
                drop(gate);
                fold.map(|()| self.folded = next)
                    .map_err(|e| format!("encode the snapshot's coverage base: {e}"))
            }
            _ => Ok(()),
        };

        CoordinodeSnapshotBuilder {
            last_applied,
            last_membership,
            snapshot_builds: Arc::clone(&self.snapshot_builds),
            folded,
            engine: self.engine.share("snapshot build"),
            engine_work: self.engine_work.clone(),
            _work: self.engine_work.start(),
        }
    }

    async fn install_snapshot(
        &mut self,
        meta: &openraft::type_config::alias::SnapshotMetaOf<TypeConfig>,
        snapshot: SnapshotData,
    ) -> Result<(), io::Error> {
        let last_log_index = meta.last_log_id.map(|id| id.index);

        // No partition copy or replacement runs across the install; after
        // it, every partition stands at the snapshot.
        let gate = Arc::clone(&self.gate);
        let mut gate = gate.state.lock().await;

        // Every tree is left holding exactly the snapshot: the entries up to
        // its last log id, nothing above. That position is recorded durably
        // before openraft is told the install finished, together with the
        // installed data it covers.
        let (next, payload) = match meta.last_log_id {
            Some(log_id) => (
                log_id.index + 1,
                rmp_serde::to_vec(&log_id).map_err(|e| io::Error::other(e.to_string()))?,
            ),
            None => (0, Vec::new()),
        };

        // A snapshot without data carries metadata only and leaves the data
        // as it is. The install reads the received trees, off the async
        // runtime, and the received copy goes once it is in.
        let engine = self.engine.share("snapshot install");
        let work = self.engine_work.start();
        let positions = tokio::task::spawn_blocking(move || {
            // Locals drop in reverse: the engine before the guard.
            let _work = work;
            let engine = engine;
            tracing::info!(last_log_index, "installing snapshot");
            match snapshot {
                SnapshotData::Received(received) => {
                    crate::snapshot::install_received(&engine, &received, next, &payload)?;
                    Ok(received.positions().to_vec())
                }
                SnapshotData::Empty => {
                    engine.reset_raft_coverage(next, &payload).map_err(|e| {
                        io::Error::other(format!("rebind raft coverage to snapshot: {e}"))
                    })?;
                    Ok(Vec::new())
                }
                // openraft installs what a leader sent or this node kept; a
                // pinned store is what this node sends.
                SnapshotData::Pinned { .. } => Err(io::Error::other(
                    "a snapshot pinned for sending cannot be installed over its own store",
                )),
            }
        })
        .await
        .map_err(|e| io::Error::other(format!("snapshot install task: {e}")))??;

        // The proposals seen so far were applied to the store the snapshot
        // replaced; the entries after it apply again, as on a fresh node.
        self.dedup
            .lock()
            .map_err(|e| io::Error::other(format!("dedup mutex poisoned: {e}")))?
            .clear();
        self.replay_skip = None;
        self.skip_until = 0;
        self.folded = next;
        gate.applies.reset(next);
        // A partition the snapshot holds further along is left alone by the
        // applies until they pass its position.
        for (partition, position) in &positions {
            gate.applies.raise_floor(*partition, position.next);
        }
        gate.last = meta.last_log_id;
        self.refresh_group_pair()?;
        self.refresh_deployment()?;
        // The snapshot carries the bound of the entries it holds.
        self.engine
            .reload_closure_frontier()
            .map_err(|e| io::Error::other(format!("read the installed closed bound: {e}")))?;
        if let Some(ref oracle) = self.oracle {
            if let Some(at) = self.engine.closure_frontier().checked_sub(1) {
                oracle.advance_to(coordinode_core::txn::timestamp::Timestamp::from_raw(at));
            }
        }

        *self
            .last_applied
            .lock()
            .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))? = meta.last_log_id;
        // The snapshot's entries are applied now; waiters on the watermark
        // must not have to wait for a later entry that an idle cluster never
        // sends.
        if let Some(log_id) = meta.last_log_id {
            self.applied_tx.send_if_modified(|applied| {
                let raised = log_id.index > *applied;
                if raised {
                    *applied = log_id.index;
                }
                raised
            });
        }

        *self
            .last_membership
            .lock()
            .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))? =
            meta.last_membership.clone();
        self.save_membership(&meta.last_membership)?;

        // The installed store is the current snapshot from now on:
        // get_current_snapshot() pins it at the applies' position.
        tracing::info!("snapshot install complete");
        Ok(())
    }

    async fn get_current_snapshot(&mut self) -> Result<Option<Snapshot>, io::Error> {
        // The snapshot is the applied store itself, pinned here: openraft
        // asks between two applies, so the pin holds exactly the entries up
        // to `last_applied`. Nothing applied yet means no snapshot.
        let last_applied = *self
            .last_applied
            .lock()
            .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))?;
        if last_applied.is_none() {
            return Ok(None);
        }
        let meta = SnapshotMeta {
            last_log_id: last_applied,
            last_membership: self
                .last_membership
                .lock()
                .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))?
                .clone(),
        };
        // The pins hold the store's files until the snapshot is sent or
        // dropped: work a shutdown waits out.
        let work = self.engine_work.start();
        let engine = self.engine.share("snapshot pin");
        let transfer = tokio::task::spawn_blocking(move || engine.pin_store())
            .await
            .map_err(|e| io::Error::other(format!("snapshot pin task: {e}")))?
            .map_err(|e| io::Error::other(format!("pin the store for a snapshot: {e}")))?;
        Ok(Some(Snapshot {
            meta,
            snapshot: SnapshotData::Pinned {
                transfer,
                _work: work,
            },
        }))
    }
}

// ── Snapshot Builder ────────────────────────────────────────────────

/// Compacts the Raft log against the applied store.
///
/// The store itself is the snapshot: a build makes every entry up to
/// `last_applied` durable in the trees, with the coverage fold the state
/// machine wrote for it, so openraft may purge the log below it. Nothing is
/// copied or serialized; a follower that needs the data gets the store
/// pinned at send time (`get_current_snapshot`).
pub struct CoordinodeSnapshotBuilder {
    last_applied: Option<openraft::type_config::alias::LogIdOf<TypeConfig>>,
    last_membership: StoredMembership,
    /// Shared build counter from the owning state machine.
    snapshot_builds: Arc<core::sync::atomic::AtomicU64>,
    /// Whether the coverage fold at `last_applied` was written.
    folded: Result<(), String>,
    /// The store the build flushes.
    engine: crate::engine_hold::EngineHold,
    /// Where the build's blocking work is counted.
    engine_work: EngineWork,
    /// Counts this builder as work holding the engine. Declared last, so it
    /// is dropped after `engine`.
    _work: EngineWorkGuard,
}

impl RaftSnapshotBuilder<TypeConfig> for CoordinodeSnapshotBuilder {
    type SnapshotData = SnapshotData;

    async fn build_snapshot(&mut self) -> Result<Snapshot, io::Error> {
        let last_log_id = self.last_applied;
        tracing::info!(
            last_log_index = last_log_id.map(|id| id.index),
            last_log_term = last_log_id.map(|id| id.committed_leader_id().term),
            "compacting the log against the applied store"
        );
        self.folded.clone().map_err(io::Error::other)?;

        let work = self.engine_work.start();
        let engine = self.engine.share("snapshot build");
        tokio::task::spawn_blocking(move || {
            // Locals drop in reverse: the engine before the guard.
            let _work = work;
            let engine = engine;
            #[cfg(test)]
            std::thread::sleep(std::time::Duration::from_millis(
                BUILD_DELAY_MS.load(core::sync::atomic::Ordering::Relaxed),
            ));
            engine
                .persist()
                .map_err(|e| io::Error::other(format!("persist the store for a snapshot: {e}")))
        })
        .await
        .map_err(|e| io::Error::other(format!("snapshot build task: {e}")))??;

        self.snapshot_builds
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(Snapshot {
            meta: SnapshotMeta {
                last_log_id,
                last_membership: self.last_membership.clone(),
            },
            snapshot: SnapshotData::Empty,
        })
    }
}

// ── Raft Config Defaults ────────────────────────────────────────────

/// Create openraft Config with CoordiNode defaults from architecture spec.
///
/// - Heartbeat: 150ms
/// - Election timeout: 300-600ms
/// - Max payload entries: 300
/// - Snapshot policy: every 10,000 entries
pub fn default_raft_config() -> openraft::Config {
    // Test-only escape hatch. Shared CI runners are CPU-oversubscribed (several
    // workflows per host), which can starve heartbeats and provoke spurious
    // elections — and openraft has a debug-only leadership-transition assertion
    // that such churn can trip. When `COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS` is
    // set (only by CI / the test harness) use generous election timeouts so the
    // election-timing-sensitive cluster tests stay deterministic under load.
    // Production never sets it and keeps the 300-600ms values; the override only
    // lengthens timeouts, so a stray setting is harmless.
    // Only the election window is widened, not the heartbeat: replication and
    // catch-up timing stay identical (tests with fixed replication waits keep
    // working), while a starved follower tolerates many missed heartbeats before
    // calling a spurious election.
    let (heartbeat_interval, election_timeout_min, election_timeout_max) =
        if std::env::var_os("COORDINODE_TEST_RAFT_GENEROUS_TIMEOUTS").is_some() {
            (150, 1500, 3000)
        } else {
            (150, 300, 600)
        };
    openraft::Config {
        heartbeat_interval,
        election_timeout_min,
        election_timeout_max,
        max_payload_entries: 300,
        snapshot_policy: openraft::SnapshotPolicy::LogsSinceLast(10_000),
        // A member whose disk lost committed entries reopens with a shorter
        // log and stays out of elections until its leader has sent them
        // again; the leader replicates to it from where its log now ends.
        allow_log_reversion: Some(true),
        // A candidate first asks whether it could win, without raising its
        // term; a member cut off from its leader then cannot depose it on
        // return. The question travels as a real request to every peer.
        enable_pre_vote: Some(true),
        ..Default::default()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
