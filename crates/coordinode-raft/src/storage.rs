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
//! - `raft:sm:applied` — last applied log id (state machine)
//! - `raft:sm:membership` — last applied membership config
//! - `raft:snapshot:meta` — snapshot metadata
//! - `raft:snapshot:data` — snapshot data (serialized)
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

use coordinode_core::txn::proposal::{Mutation, PartitionId, RaftProposal};
use coordinode_storage::engine::core::{
    RaftApplyFence, RaftApplyState, RaftCoverage, StorageEngine,
};
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::{StorageError, StorageResult};

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
}

impl Request {
    /// Create a request from a single proposal (non-batched path).
    pub fn single(proposal: RaftProposal) -> Self {
        Self {
            proposals: vec![proposal],
        }
    }

    /// Create a request from a batch of proposals (coalesced path).
    pub fn batch(proposals: Vec<RaftProposal>) -> Self {
        Self { proposals }
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
);

// Type aliases for convenience
pub type CommittedLeaderId = openraft::vote::leader_id_adv::CommittedLeaderId<u64, u64>;
pub type LeaderId = openraft::impls::leader_id_adv::LeaderId<u64, u64>;
pub type LogId = openraft::LogId<CommittedLeaderId>;
pub type Vote = openraft::impls::Vote<LeaderId>;
pub type Entry = openraft::impls::Entry<
    CommittedLeaderId,
    openraft::impls::EntryPayload<Request, u64, openraft::impls::BasicNode>,
>;
pub type SnapshotMeta =
    openraft::storage::SnapshotMeta<CommittedLeaderId, u64, openraft::impls::BasicNode>;
pub type Snapshot = openraft::storage::Snapshot<
    CommittedLeaderId,
    u64,
    openraft::impls::BasicNode,
    std::io::Cursor<Vec<u8>>,
>;
pub type StoredMembership =
    openraft::StoredMembership<CommittedLeaderId, u64, openraft::impls::BasicNode>;

// ── Key Prefixes ────────────────────────────────────────────────────

const KEY_VOTE: &[u8] = b"raft:vote";
const KEY_COMMITTED: &[u8] = b"raft:committed";
const KEY_SM_APPLIED: &[u8] = b"raft:sm:applied";
const KEY_SM_MEMBERSHIP: &[u8] = b"raft:sm:membership";
const KEY_SNAPSHOT_META: &[u8] = b"raft:snapshot:meta";
const KEY_SNAPSHOT_DATA: &[u8] = b"raft:snapshot:data";
const KEY_PURGED: &[u8] = b"raft:purged";
/// Persisted last_log_id for O(1) recovery after restart.
const KEY_LAST_LOG_ID: &[u8] = b"raft:oplog:last_log_id";

/// OplogManager defaults for the Raft log oplog.
const RAFT_OPLOG_MAX_BYTES: u64 = 64 * 1024 * 1024; // 64 MB per segment
const RAFT_OPLOG_MAX_ENTRIES: u32 = 50_000;
const RAFT_OPLOG_RETENTION_SECS: u64 = 7 * 24 * 3600; // 7 days (index-based purge is primary)

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

/// Storage-backed Raft log storage.
///
/// Log entries are stored in **oplog segments** (one sealed file per segment),
/// not in the LSM KV store. This delivers:
/// - O(1) `last_log_id` via an in-memory cache (eliminates the O(N) scan)
/// - Segment-granular purge via `OplogManager::purge_before`
/// - Sequential I/O for `append` and `try_get_log_entries`
///
/// Raft metadata (vote, committed, purged, last_log_id) is stored in
/// `Partition::Raft` to separate it from application data.
///
/// All fields use `Arc` so `get_log_reader()` returns a cheap clone that
/// shares the same oplog handle and caches.
/// Tells a waiting writer at which log index its proposal became durable in
/// this member's log.
///
/// The proposal pipeline subscribes by proposal id before it submits; the log
/// store fires after the batch fsync in [`RaftLogStorage::append`]. This is
/// what a write concern of `w:1` waits for, and what `w:N` counts from: the
/// leader's own copy is the first of the N.
#[derive(Default)]
pub struct AppendNotifier {
    waiters: Mutex<
        HashMap<coordinode_core::txn::proposal::ProposalId, tokio::sync::oneshot::Sender<u64>>,
    >,
}

impl AppendNotifier {
    /// Wait for `id` to be durable in this member's log; resolves to its
    /// log index. Subscribe before submitting the proposal, or the append may
    /// fire first and the receiver never resolves.
    pub fn subscribe(
        &self,
        id: coordinode_core::txn::proposal::ProposalId,
    ) -> tokio::sync::oneshot::Receiver<u64> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if let Ok(mut waiters) = self.waiters.lock() {
            waiters.insert(id, tx);
        }
        rx
    }

    fn notify(&self, id: coordinode_core::txn::proposal::ProposalId, index: u64) {
        let waiter = match self.waiters.lock() {
            Ok(mut waiters) => waiters.remove(&id),
            Err(_) => None,
        };
        if let Some(tx) = waiter {
            // A receiver that gave up (timeout) is not an error here.
            let _ = tx.send(index);
        }
    }
}

pub struct LogStore {
    engine: Arc<StorageEngine>,
    /// Oplog manager for log entry segments.
    oplog: Arc<Mutex<OplogManager>>,
    /// Waiters for "my proposal is durable in this log", see [`AppendNotifier`].
    append_notifier: Arc<AppendNotifier>,
    /// In-memory cache of the last appended log id. Updated on every
    /// `append()` and persisted to `Partition::Raft` so it survives restarts.
    last_log_id: Arc<Mutex<Option<LogId>>>,
    /// In-memory cache of the last purged log id. Updated on every `purge()`
    /// and persisted to `Partition::Raft`.
    last_purged: Arc<Mutex<Option<LogId>>>,
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

    /// Open the LogStore, routing oplog segments to the oplog-eligible
    /// endpoint chosen for shard 0, so the log lands on media fit to carry
    /// it rather than wherever the data partitions live. Path layout:
    /// `<endpoint.path>/oplog/<shard_id>/`.
    /// Recovery scans every oplog-eligible endpoint's directory for sealed
    /// segments left over from a previous config-driven routing.
    ///
    /// Loads `last_log_id` and `last_purged` from `Partition::Raft` so
    /// `get_log_state()` is O(1) even after a restart.
    pub fn open(engine: Arc<StorageEngine>) -> Result<Self, io::Error> {
        // Shard 0 = single Raft log shard in CE. Multi-shard EE will
        // open one LogStore per shard, each routing to its own
        // select_oplog_endpoint(shard_id).
        let shard_id: u32 = 0;
        let RaftOplogDirs {
            active: active_dir,
            all: recovery_dirs,
        } = raft_oplog_dirs(&engine, shard_id)?;
        let oplog = OplogManager::open_multi(
            &active_dir,
            &recovery_dirs,
            shard_id,
            RAFT_OPLOG_MAX_BYTES,
            RAFT_OPLOG_MAX_ENTRIES,
            RAFT_OPLOG_RETENTION_SECS,
        )
        .map_err(|e| io::Error::other(e.to_string()))?;

        // Load persisted caches from Partition::Raft — O(1) recovery.
        let last_purged = Self::load_log_id_from_partition(&engine, KEY_PURGED);
        let mut last_log_id = Self::load_log_id_from_partition(&engine, KEY_LAST_LOG_ID);

        // Crash-recovery path: if the LSM key was not flushed before process
        // death but oplog segment files exist, reconstruct last_log_id by
        // scanning the last segment (which may lack a footer).
        //
        // Without this, openraft sees an empty log state and calls initialize(),
        // which tries to create oplog segment 0 — but the file already exists
        // → I/O error "File exists (os error 17)".
        if last_log_id.is_none() && oplog.has_segments() {
            match Self::recover_last_log_id_from_oplog(&oplog) {
                Ok(Some(recovered_id)) => {
                    tracing::warn!(
                        recovered_index = recovered_id.index,
                        "raft: last_log_id missing from LSM — recovered from oplog segment"
                    );
                    // Persist to avoid O(N) re-scan on the next restart.
                    if let Ok(bytes) = rmp_serde::to_vec(&recovered_id) {
                        let _ = engine.put(Partition::Raft, KEY_LAST_LOG_ID, &bytes);
                    }
                    last_log_id = Some(recovered_id);
                }
                Ok(None) => {
                    // Segments exist but contain no valid entries — leave last_log_id as None.
                    // openraft will re-initialize the Raft group cleanly.
                    tracing::warn!(
                        "raft: oplog segments found but no valid entries readable — treating log as empty"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "raft: failed to recover last_log_id from oplog — treating log as empty"
                    );
                }
            }
        }

        Ok(Self {
            engine,
            oplog: Arc::new(Mutex::new(oplog)),
            append_notifier: Arc::new(AppendNotifier::default()),
            last_log_id: Arc::new(Mutex::new(last_log_id)),
            last_purged: Arc::new(Mutex::new(last_purged)),
        })
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

    fn load_log_id_from_partition(engine: &StorageEngine, key: &[u8]) -> Option<LogId> {
        engine
            .get(Partition::Raft, key)
            .ok()
            .flatten()
            .and_then(|bytes| rmp_serde::from_slice(&bytes).ok())
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

    /// Map `PartitionId` (coordinode-core) to oplog u8 discriminant.
    fn partition_id_to_u8(id: PartitionId) -> u8 {
        match id {
            PartitionId::Node => 0,
            PartitionId::Adj => 1,
            PartitionId::EdgeProp => 2,
            PartitionId::Blob => 3,
            PartitionId::BlobRef => 4,
            PartitionId::Schema => 5,
            PartitionId::Idx => 6,
            PartitionId::Counter => 7,
            PartitionId::VectorF32 => 8,
            PartitionId::Registry => 9,
        }
    }

    /// Encode a Raft Entry as an OplogEntry.
    ///
    /// For Normal entries (application proposals), the mutations are decoded
    /// and stored as OplogOp::Insert/Delete/Merge alongside the original
    /// RaftEntry (kept for Raft log recovery via `oplog_to_entry`). This
    /// enables server-side CDC filtering by edge_type and is_migration.
    ///
    /// For Membership entries, only the RaftEntry op is stored (no mutations).
    fn entry_to_oplog(entry: &Entry) -> Result<OplogEntry, io::Error> {
        let data = rmp_serde::to_vec(entry).map_err(|e| io::Error::other(e.to_string()))?;
        let raft_op = OplogOp::RaftEntry { data };

        // Extract decoded mutation ops from Normal entries for CDC filtering.
        // Batched entries flatten all proposals' mutations into a single ops list.
        let (mut ops, is_migration) = match &entry.payload {
            openraft::entry::EntryPayload::Normal(request) => {
                let decoded: Vec<OplogOp> = request
                    .proposals
                    .iter()
                    .flat_map(|p| p.mutations.iter())
                    .map(|m| match m {
                        Mutation::Put {
                            partition,
                            key,
                            value,
                        } => OplogOp::Insert {
                            partition: Self::partition_id_to_u8(*partition),
                            key: key.clone(),
                            value: value.clone(),
                        },
                        Mutation::Delete { partition, key } => OplogOp::Delete {
                            partition: Self::partition_id_to_u8(*partition),
                            key: key.clone(),
                        },
                        Mutation::Merge {
                            partition,
                            key,
                            operand,
                        } => OplogOp::Merge {
                            partition: Self::partition_id_to_u8(*partition),
                            key: key.clone(),
                            operand: operand.clone(),
                        },
                        Mutation::RemoveRange {
                            partition,
                            start,
                            end,
                        } => OplogOp::RemoveRange {
                            partition: Self::partition_id_to_u8(*partition),
                            start: start.clone(),
                            end: end.clone(),
                        },
                    })
                    .collect();
                // is_migration could be derived from proposal metadata in the future;
                // for now, proposals don't carry migration flags.
                (decoded, false)
            }
            _ => (Vec::new(), false),
        };

        // RaftEntry is always first — oplog_to_entry() relies on ops[0] being RaftEntry.
        ops.insert(0, raft_op);

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
        match oplog_entry.ops.into_iter().next() {
            Some(OplogOp::RaftEntry { data }) => {
                rmp_serde::from_slice(&data).map_err(|e| io::Error::other(e.to_string()))
            }
            Some(op) => Err(io::Error::other(format!(
                "unexpected oplog op in Raft segment: expected RaftEntry, got {:?}",
                std::mem::discriminant(&op)
            ))),
            None => Err(io::Error::other("oplog entry has no ops in Raft segment")),
        }
    }
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
            engine: Arc::clone(&self.engine),
            oplog: Arc::clone(&self.oplog),
            last_log_id: Arc::clone(&self.last_log_id),
            last_purged: Arc::clone(&self.last_purged),
            append_notifier: Arc::clone(&self.append_notifier),
        }
    }

    async fn save_vote(&mut self, vote: &Vote) -> Result<(), io::Error> {
        let bytes = rmp_serde::to_vec(vote).map_err(|e| io::Error::other(e.to_string()))?;
        self.put(KEY_VOTE, &bytes)
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
        // the waiting writers only after the fsync below.
        let mut appended: Vec<(coordinode_core::txn::proposal::ProposalId, u64)> = Vec::new();
        {
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
            // ONE fsync per write batch: flush user-space buffer → kernel → storage.
            // All entries above are durable after this call. This is the crash-safety
            // boundary: a process killed after flush_and_sync() will NOT lose these entries.
            oplog.flush().map_err(|e| io::Error::other(e.to_string()))?;
        }
        for (id, index) in appended {
            self.append_notifier.notify(id, index);
        }

        // Update and persist the last_log_id cache.
        if let Some(log_id) = last {
            *self
                .last_log_id
                .lock()
                .map_err(|_| io::Error::other("last_log_id mutex poisoned"))? = Some(log_id);
            let bytes = rmp_serde::to_vec(&log_id).map_err(|e| io::Error::other(e.to_string()))?;
            self.put(KEY_LAST_LOG_ID, &bytes)?;
        }

        // Signal IO completion — entries are durable in the oplog (fsynced above).
        callback.io_completed(Ok(()));

        Ok(())
    }

    async fn truncate_after(
        &mut self,
        last_log_id: Option<openraft::type_config::alias::LogIdOf<TypeConfig>>,
    ) -> Result<(), io::Error> {
        let keep_exclusive = last_log_id.map_or(0, |id| id.index + 1);

        // Read entries to keep before wiping the oplog.
        let to_keep = if keep_exclusive > 0 {
            self.oplog
                .lock()
                .map_err(|_| io::Error::other("oplog mutex poisoned"))?
                .read_range(0, keep_exclusive)
                .map_err(|e| io::Error::other(e.to_string()))?
        } else {
            Vec::new()
        };

        // Delete all segments, then re-append the kept entries.
        {
            let mut oplog = self
                .oplog
                .lock()
                .map_err(|_| io::Error::other("oplog mutex poisoned"))?;
            oplog
                .truncate_all()
                .map_err(|e| io::Error::other(e.to_string()))?;
            for oe in to_keep {
                oplog
                    .append(&oe)
                    .map_err(|e| io::Error::other(e.to_string()))?;
            }
        }

        // Update and persist the last_log_id cache.
        *self
            .last_log_id
            .lock()
            .map_err(|_| io::Error::other("last_log_id mutex poisoned"))? = last_log_id;
        match last_log_id {
            Some(id) => {
                let bytes = rmp_serde::to_vec(&id).map_err(|e| io::Error::other(e.to_string()))?;
                self.put(KEY_LAST_LOG_ID, &bytes)?;
            }
            None => {
                self.delete(KEY_LAST_LOG_ID)?;
            }
        }

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
        if below == 0 {
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

        let purged_segments = self
            .oplog
            .lock()
            .map_err(|_| io::Error::other("oplog mutex poisoned"))?
            .purge_before(below)
            .map_err(|e| io::Error::other(e.to_string()))?;

        // Update and persist the last_purged cache.
        *self
            .last_purged
            .lock()
            .map_err(|_| io::Error::other("last_purged mutex poisoned"))? = Some(purged_to);
        let bytes = rmp_serde::to_vec(&purged_to).map_err(|e| io::Error::other(e.to_string()))?;
        self.put(KEY_PURGED, &bytes)?;

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
    engine: Arc<StorageEngine>,
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
    last_membership:
        Mutex<openraft::StoredMembership<CommittedLeaderId, u64, openraft::impls::BasicNode>>,
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
    /// Per-shard `maxAssigned` watermark over HLC commit_ts.
    /// Distinct from `applied_tx` (Raft log index). Advanced after every
    /// successful proposal apply so snapshot readers can `WaitForTs(T)`
    /// until every write with `commit_ts ≤ T` has been applied to every
    /// modality on this shard.
    ///
    /// Optional so legacy paths that don't need cross-modality snapshot
    /// semantics can leave it `None` (e.g. embedded-mode fast tests).
    max_assigned: Option<Arc<coordinode_core::txn::watermark::MaxAssignedWatermark>>,
    /// Count of full snapshot builds performed by this state machine.
    /// Observability for the snapshot trigger: every build serializes
    /// ALL partitions (hundreds of MB), so redundant rebuilds are a
    /// direct latency and leader-stability hazard worth monitoring.
    snapshot_builds: Arc<core::sync::atomic::AtomicU64>,
    /// What each partition tree held when the state machine opened. Entries
    /// below [`Self::skip_until`] are re-delivered from the lowest covered
    /// prefix, and a tree that already holds a proposal is left alone; past
    /// that point nothing needs checking and this is dropped.
    replay_skip: Option<RaftCoverage>,
    skip_until: u64,
    /// The coverage base last written to every tree: all entries below it.
    folded: u64,
    /// Captures taken for snapshot builds, naming each one's directory.
    captures: u64,
    /// Where the applies stand, held while applying; registered with the
    /// engine as its Raft apply fence.
    gate: Arc<RaftApplyGate>,
    /// Snapshot work holding the engine off the async runtime, which a
    /// shutdown waits out.
    engine_work: EngineWork,
}

/// Work the state machine runs outside openraft's tasks that holds the
/// engine: a snapshot capture on a blocking thread, a snapshot build.
///
/// Shutting openraft down stops its tasks, but a blocking thread they were
/// waiting on runs to its end, and a build task can outlive the core. Until
/// both finish they hold the engine, so the directory stays locked and a
/// restart in the same process cannot open it. The node waits for this to
/// go idle after the consensus stops.
#[derive(Clone, Default)]
pub struct EngineWork(Arc<EngineWorkInner>);

#[derive(Default)]
struct EngineWorkInner {
    running: core::sync::atomic::AtomicUsize,
    idle: tokio::sync::Notify,
}

/// One piece of [`EngineWork`] under way; finishes on drop.
pub(crate) struct EngineWorkGuard(Arc<EngineWorkInner>);

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

/// How long a snapshot capture sleeps before it starts, so a test can shut a
/// node down while one is under way.
#[cfg(test)]
pub(crate) static CAPTURE_DELAY_MS: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);

/// Where snapshot builds capture the store, under the engine's data
/// directory. Nothing under it outlives the build that made it; a crash
/// leaves it behind, and the next open clears it.
fn snapshot_capture_root(engine: &StorageEngine) -> std::path::PathBuf {
    engine.data_dir().join("snapshot-capture")
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
    /// holds. It is moved to this release by a dump and a restore. A failed
    /// coverage read or the first record's flush also fails the open.
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
            None if Self::load_log_id(&engine, KEY_SM_APPLIED).is_some() => {
                return Err(io::Error::other(
                    "this store applied Raft entries without an apply-coverage record, so \
                     nothing proves which of them each partition holds; dump it with the \
                     release that wrote it (`coordinode backup --format raft-snapshot`) and \
                     restore the dump with this one (`coordinode restore --format raft-snapshot`)",
                ));
            }
            None => {
                engine
                    .reset_raft_coverage(0, &[])
                    .map_err(|e| io::Error::other(format!("establish raft coverage: {e}")))?;
                (None, None)
            }
        };
        // A capture a crash left behind belongs to no build.
        match std::fs::remove_dir_all(snapshot_capture_root(&engine)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(io::Error::other(format!("clear snapshot captures: {e}"))),
        }
        // The trees may still hold markers below their bases; the first fold
        // removes markers from index 0.
        let folded = 0;
        let skip_until = replay_skip.as_ref().map_or(0, RaftCoverage::skip_until);
        let last_membership = Self::load_membership(&engine);

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

        Ok(Self {
            engine,
            oracle,
            last_applied: Mutex::new(last_applied),
            last_membership: Mutex::new(last_membership),
            dedup: Mutex::new(HashMap::new()),
            last_dedup_gc: Mutex::new(Instant::now()),
            applied_tx,
            applied_rx,
            max_assigned,
            snapshot_builds: Arc::new(core::sync::atomic::AtomicU64::new(0)),
            replay_skip,
            skip_until,
            folded,
            captures: 0,
            gate,
            engine_work: EngineWork::default(),
        })
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

    fn load_log_id(
        engine: &StorageEngine,
        key: &[u8],
    ) -> Option<openraft::type_config::alias::LogIdOf<TypeConfig>> {
        debug_assert!(
            key.starts_with(b"raft:"),
            "Raft state machine keys in Schema partition must use raft: prefix, got {:?}",
            String::from_utf8_lossy(key)
        );
        engine
            .get(Partition::Schema, key)
            .ok()
            .flatten()
            .and_then(|bytes| rmp_serde::from_slice(&bytes).ok())
    }

    fn load_membership(
        engine: &StorageEngine,
    ) -> openraft::StoredMembership<CommittedLeaderId, u64, openraft::impls::BasicNode> {
        engine
            .get(Partition::Schema, KEY_SM_MEMBERSHIP)
            .ok()
            .flatten()
            .and_then(|bytes| rmp_serde::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    fn save_membership(
        &self,
        membership: &openraft::StoredMembership<CommittedLeaderId, u64, openraft::impls::BasicNode>,
    ) -> Result<(), io::Error> {
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

        // Dedup check: same key + same size = already applied
        {
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
        {
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
    // Snapshot payloads are whole in-memory blobs, so an owned cursor is the
    // read/write handle. openraft 0.10 moved this out of the type config onto
    // the state machine.
    type SnapshotData = std::io::Cursor<Vec<u8>>;

    type SnapshotBuilder = CoordinodeSnapshotBuilder;

    async fn applied_state(
        &mut self,
    ) -> Result<
        (
            Option<openraft::type_config::alias::LogIdOf<TypeConfig>>,
            openraft::StoredMembership<CommittedLeaderId, u64, openraft::impls::BasicNode>,
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
                    for (sub, proposal) in request.proposals.iter().enumerate() {
                        let sub = u32::try_from(sub).map_err(|_| {
                            io::Error::other(format!("entry {index} carries over 2^32 proposals"))
                        })?;
                        let r = self.apply_proposal_under(proposal, index, sub, &gate.applies)?;
                        total += r.mutations_applied;
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

        // openraft builds the snapshot after this returns, while entries keep
        // applying, and the snapshot must hold exactly the entries up to
        // `last_applied`. No entry is applied while this runs, so a capture
        // taken now is that state; reading the live store later is not.
        self.captures += 1;
        let dir = snapshot_capture_root(&self.engine).join(format!("{:020}", self.captures));
        // Counted from before the capture to the builder's end, so the work
        // never reads idle between the two.
        let builder_work = self.engine_work.start();
        let engine = Arc::clone(&self.engine);
        let target = dir.clone();
        let work = self.engine_work.start();
        let capture = match tokio::task::spawn_blocking(move || {
            #[cfg(test)]
            std::thread::sleep(std::time::Duration::from_millis(
                CAPTURE_DELAY_MS.load(core::sync::atomic::Ordering::Relaxed),
            ));
            let captured = match target.parent() {
                Some(parent) => std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create {parent:?}: {e}"))
                    .and_then(|()| engine.capture(&target).map_err(|e| e.to_string())),
                None => engine.capture(&target).map_err(|e| e.to_string()),
            };
            // The engine is released before the work is counted done.
            drop(engine);
            drop(work);
            captured
        })
        .await
        {
            Ok(Ok(_)) => Ok(dir),
            Ok(Err(e)) => Err(format!("capture the store for a snapshot: {e}")),
            Err(e) => Err(format!("capture task: {e}")),
        };

        CoordinodeSnapshotBuilder {
            engine: Arc::clone(&self.engine),
            last_applied,
            last_membership,
            snapshot_builds: Arc::clone(&self.snapshot_builds),
            capture,
            engine_work: self.engine_work.clone(),
            _work: builder_work,
        }
    }

    async fn install_snapshot(
        &mut self,
        meta: &openraft::type_config::alias::SnapshotMetaOf<TypeConfig>,
        snapshot: std::io::Cursor<Vec<u8>>,
    ) -> Result<(), io::Error> {
        let data = snapshot.into_inner();

        tracing::info!(
            data_bytes = data.len(),
            last_log_index = meta.last_log_id.map(|id| id.index),
            "installing snapshot"
        );

        // No partition copy or replacement runs across the install; after
        // it, every partition stands at the snapshot.
        let gate = Arc::clone(&self.gate);
        let mut gate = gate.state.lock().await;

        // Apply snapshot data to storage partitions (if non-empty).
        // Empty snapshots are valid (metadata-only, e.g., from tests).
        if !data.is_empty() {
            crate::snapshot::install_full_snapshot(&self.engine, &data)?;
        }

        // Every tree now holds exactly the snapshot: the entries up to its
        // last log id, nothing above. Recorded durably before openraft is told
        // the install finished, together with the installed data it covers.
        let (next, payload) = match meta.last_log_id {
            Some(log_id) => (
                log_id.index + 1,
                rmp_serde::to_vec(&log_id).map_err(|e| io::Error::other(e.to_string()))?,
            ),
            None => (0, Vec::new()),
        };
        self.engine
            .reset_raft_coverage(next, &payload)
            .map_err(|e| io::Error::other(format!("rebind raft coverage to snapshot: {e}")))?;
        self.replay_skip = None;
        self.skip_until = 0;
        self.folded = next;
        gate.applies.reset(next);
        gate.last = meta.last_log_id;

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

        // Save snapshot data for get_current_snapshot()
        self.engine
            .put(Partition::Schema, KEY_SNAPSHOT_DATA, &data)
            .map_err(|e| io::Error::other(e.to_string()))?;

        // Save snapshot metadata
        let meta_bytes = rmp_serde::to_vec(meta).map_err(|e| io::Error::other(e.to_string()))?;
        self.engine
            .put(Partition::Schema, KEY_SNAPSHOT_META, &meta_bytes)
            .map_err(|e| io::Error::other(e.to_string()))?;

        tracing::info!("snapshot install complete");
        Ok(())
    }

    async fn get_current_snapshot(&mut self) -> Result<Option<Snapshot>, io::Error> {
        let meta_bytes = match self
            .engine
            .get(Partition::Schema, KEY_SNAPSHOT_META)
            .map_err(|e| io::Error::other(e.to_string()))?
        {
            Some(b) => b.to_vec(),
            None => return Ok(None),
        };

        let meta: SnapshotMeta =
            rmp_serde::from_slice(&meta_bytes).map_err(|e| io::Error::other(e.to_string()))?;

        let data = self
            .engine
            .get(Partition::Schema, KEY_SNAPSHOT_DATA)
            .map_err(|e| io::Error::other(e.to_string()))?
            .map(|b| b.to_vec())
            .unwrap_or_default();

        Ok(Some(Snapshot {
            meta,
            snapshot: std::io::Cursor::new(data),
        }))
    }
}

// ── Snapshot Builder ────────────────────────────────────────────────

/// Builds a full snapshot of all storage partitions for Raft log compaction.
///
/// The snapshot captures every KV pair of the replicated partitions (see
/// `snapshot::snapshot_partitions`) in a binary format with xxh3 checksum.
/// This data is then sent to followers via the Snapshot gRPC RPC.
///
/// openraft manages the snapshot metadata (index, term, membership); the
/// data itself travels in this binary format, never through the Raft log.
pub struct CoordinodeSnapshotBuilder {
    /// Engine reference for iterating all storage partitions.
    engine: Arc<StorageEngine>,
    last_applied: Option<openraft::type_config::alias::LogIdOf<TypeConfig>>,
    last_membership: openraft::StoredMembership<CommittedLeaderId, u64, openraft::impls::BasicNode>,
    /// Shared build counter from the owning state machine.
    snapshot_builds: Arc<core::sync::atomic::AtomicU64>,
    /// The store captured at `last_applied`, or why it could not be.
    capture: Result<std::path::PathBuf, String>,
    /// Where the build's blocking work is counted.
    engine_work: EngineWork,
    /// Counts this builder as work holding the engine. Declared last, so it
    /// is dropped after `engine`.
    _work: EngineWorkGuard,
}

impl Drop for CoordinodeSnapshotBuilder {
    fn drop(&mut self) {
        if let Ok(dir) = &self.capture {
            if let Err(e) = std::fs::remove_dir_all(dir) {
                if e.kind() != io::ErrorKind::NotFound {
                    tracing::warn!(?dir, %e, "could not remove a snapshot capture");
                }
            }
        }
    }
}

impl RaftSnapshotBuilder<TypeConfig> for CoordinodeSnapshotBuilder {
    type SnapshotData = std::io::Cursor<Vec<u8>>;

    async fn build_snapshot(&mut self) -> Result<Snapshot, io::Error> {
        let last_log_id = self.last_applied;

        tracing::info!(
            last_log_index = last_log_id.map(|id| id.index),
            last_log_term = last_log_id.map(|id| id.committed_leader_id().term),
            "building full storage snapshot"
        );

        self.snapshot_builds
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        // Serialize the captured store: every entry up to `last_log_id`, none
        // after it.
        let dir = self.capture.clone().map_err(io::Error::other)?;
        // The capture's tables are hard links into the store's; the build
        // counts as work on the store until it has closed them.
        let work = self.engine_work.start();
        let data = tokio::task::spawn_blocking(move || {
            let _work = work;
            let captured = StorageEngine::open_checkpoint(&dir)
                .map_err(|e| io::Error::other(format!("open the snapshot capture: {e}")))?;
            crate::snapshot::build_full_snapshot(&captured)
        })
        .await
        .map_err(|e| io::Error::other(format!("snapshot build task: {e}")))??;

        let meta = SnapshotMeta {
            last_log_id,
            last_membership: self.last_membership.clone(),
        };

        tracing::info!(snapshot_bytes = data.len(), "snapshot build complete");

        // Persist snapshot to storage so get_current_snapshot() can return it.
        // openraft's build_snapshot flow doesn't call install_snapshot()
        // on the leader — the built snapshot is kept in-memory for sending
        // to followers. We persist it here for durability and restart recovery.
        let meta_bytes = rmp_serde::to_vec(&meta).map_err(|e| io::Error::other(e.to_string()))?;
        self.engine
            .put(Partition::Schema, KEY_SNAPSHOT_META, &meta_bytes)
            .map_err(|e| io::Error::other(e.to_string()))?;
        self.engine
            .put(Partition::Schema, KEY_SNAPSHOT_DATA, &data)
            .map_err(|e| io::Error::other(e.to_string()))?;

        Ok(Snapshot {
            meta,
            snapshot: std::io::Cursor::new(data),
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
        ..Default::default()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
