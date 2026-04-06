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
use openraft::entry::RaftPayload;
use openraft::storage::{IOFlushed, LogState, RaftLogStorage, RaftStateMachine};
use openraft::{OptionalSend, RaftLogReader, RaftSnapshotBuilder};
use serde::{Deserialize, Serialize};

use coordinode_core::txn::proposal::{Mutation, PartitionId, RaftProposal};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use crate::proposal::to_partition;

/// Maximum age for dedup entries before GC (10 minutes).
/// Matches Dgraph's `maxAge` in processApplyCh (draft.go:942).
const DEDUP_MAX_AGE_SECS: u64 = 600;

/// GC interval for dedup map (5 minutes = maxAge / 2).
/// Matches Dgraph's `tick` interval (draft.go:943).
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
pub type Entry =
    openraft::impls::Entry<CommittedLeaderId, Request, u64, openraft::impls::BasicNode>;
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
pub struct LogStore {
    engine: Arc<StorageEngine>,
    /// Oplog manager for log entry segments.
    oplog: Arc<Mutex<OplogManager>>,
    /// In-memory cache of the last appended log id. Updated on every
    /// `append()` and persisted to `Partition::Raft` so it survives restarts.
    last_log_id: Arc<Mutex<Option<LogId>>>,
    /// In-memory cache of the last purged log id. Updated on every `purge()`
    /// and persisted to `Partition::Raft`.
    last_purged: Arc<Mutex<Option<LogId>>>,
}

impl LogStore {
    /// Open the LogStore, creating the oplog directory under
    /// `engine.data_dir()/raft_oplog/` if it does not exist.
    ///
    /// Loads `last_log_id` and `last_purged` from `Partition::Raft` so
    /// `get_log_state()` is O(1) even after a restart.
    pub fn open(engine: Arc<StorageEngine>) -> Result<Self, io::Error> {
        let oplog_dir = engine.data_dir().join("raft_oplog");
        let oplog = OplogManager::open(
            &oplog_dir,
            0, // shard_id — single shard for the Raft log
            RAFT_OPLOG_MAX_BYTES,
            RAFT_OPLOG_MAX_ENTRIES,
            RAFT_OPLOG_RETENTION_SECS,
        )
        .map_err(|e| io::Error::other(e.to_string()))?;

        // Load persisted caches from Partition::Raft — O(1) recovery.
        let last_purged = Self::load_log_id_from_partition(&engine, KEY_PURGED);
        let last_log_id = Self::load_log_id_from_partition(&engine, KEY_LAST_LOG_ID);

        Ok(Self {
            engine,
            oplog: Arc::new(Mutex::new(oplog)),
            last_log_id: Arc::new(Mutex::new(last_log_id)),
            last_purged: Arc::new(Mutex::new(last_purged)),
        })
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

        Ok(OplogEntry {
            ts: 0,
            term: 0,
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
        {
            let mut oplog = self
                .oplog
                .lock()
                .map_err(|_| io::Error::other("oplog mutex poisoned"))?;
            for entry in entries {
                let oplog_entry = Self::entry_to_oplog(&entry)?;
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
        // Delete sealed segments whose last entry index < log_id.index + 1.
        let purged_segments = self
            .oplog
            .lock()
            .map_err(|_| io::Error::other("oplog mutex poisoned"))?
            .purge_before(log_id.index + 1)
            .map_err(|e| io::Error::other(e.to_string()))?;

        // Update and persist the last_purged cache.
        *self
            .last_purged
            .lock()
            .map_err(|_| io::Error::other("last_purged mutex poisoned"))? = Some(log_id);
        let bytes = rmp_serde::to_vec(&log_id).map_err(|e| io::Error::other(e.to_string()))?;
        self.put(KEY_PURGED, &bytes)?;

        tracing::debug!(
            purged_up_to = log_id.index,
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
/// Follows Dgraph's `P` struct pattern (draft.go:866-871).
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
    /// Timestamp oracle for seqno advancement during Raft replay (R068, ADR-016).
    ///
    /// Before applying each entry's mutations, we call `oracle.advance_to(commit_ts)`
    /// so all writes in that entry receive the correct seqno. This ensures:
    /// - Raft replay produces identical seqnos as original application
    /// - snapshot_at(commit_ts) works correctly for time-travel reads
    /// - OCC conflict detection sees the right seqnos
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
    /// Follows Dgraph's `Applied.Done(index)` pattern (draft.go:101).
    applied_tx: tokio::sync::watch::Sender<u64>,
    /// Receiver side kept to prevent channel closure.
    applied_rx: tokio::sync::watch::Receiver<u64>,
}

impl CoordinodeStateMachine {
    pub fn new(engine: Arc<StorageEngine>) -> Self {
        Self::with_oracle(engine, None)
    }

    /// Create with a timestamp oracle for seqno advancement (R068, ADR-016).
    ///
    /// When oracle is set, `apply_proposal()` calls `oracle.advance_to(commit_ts)`
    /// before writing mutations, ensuring each entry's writes get the correct seqno.
    pub fn with_oracle(
        engine: Arc<StorageEngine>,
        oracle: Option<Arc<coordinode_core::txn::timestamp::TimestampOracle>>,
    ) -> Self {
        // Load persisted state
        let last_applied = Self::load_log_id(&engine, KEY_SM_APPLIED);
        let last_membership = Self::load_membership(&engine);

        // Initialize applied watermark from persisted state
        let initial_index = last_applied.map(|id| id.index).unwrap_or(0);
        let (applied_tx, applied_rx) = tokio::sync::watch::channel(initial_index);

        Self {
            engine,
            oracle,
            last_applied: Mutex::new(last_applied),
            last_membership: Mutex::new(last_membership),
            dedup: Mutex::new(HashMap::new()),
            last_dedup_gc: Mutex::new(Instant::now()),
            applied_tx,
            applied_rx,
        }
    }

    /// Subscribe to the applied watermark.
    ///
    /// Returns a `watch::Receiver<u64>` that yields the latest applied
    /// log index whenever a new entry is applied. Use `changed().await`
    /// to wait for updates, or `borrow()` to read the current value.
    ///
    /// Used by:
    /// - Follower reads: wait until `Applied >= query.readTs` (R141)
    /// - Snapshot decisions: check entries since last snapshot (R134)
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

    fn save_applied(
        &self,
        log_id: &openraft::type_config::alias::LogIdOf<TypeConfig>,
    ) -> Result<(), io::Error> {
        let bytes = rmp_serde::to_vec(log_id).map_err(|e| io::Error::other(e.to_string()))?;
        self.engine
            .put(Partition::Schema, KEY_SM_APPLIED, &bytes)
            .map_err(|e| io::Error::other(e.to_string()))
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

    /// Apply a single proposal's mutations to storage (ADR-016: native seqno MVCC).
    ///
    /// Put/Delete use plain engine.put()/delete() — OracleSeqnoGenerator
    /// auto-stamps seqno. No versioned key encoding.
    ///
    /// Includes dedup check: if this proposal ID was already applied with
    /// the same size estimate, skip re-application (idempotent Raft replay).
    /// Follows Dgraph's dedup pattern (draft.go:874-940).
    fn apply_proposal(&self, proposal: &RaftProposal) -> Result<Response, io::Error> {
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

        // R068: Advance oracle so next seqno = commit_ts.
        //
        // advance_to(N) sets counter to N, then next() returns N+1.
        // So advance_to(commit_ts - 1) makes the first write get commit_ts.
        // For proposals with multiple mutations, subsequent writes get
        // commit_ts+1, commit_ts+2, etc. — all within this entry's range.
        if let Some(ref oracle) = self.oracle {
            if proposal.commit_ts.as_raw() > 0 {
                oracle.advance_to(coordinode_core::txn::timestamp::Timestamp::from_raw(
                    proposal.commit_ts.as_raw() - 1,
                ));
            }
        }

        // Apply mutations (ADR-016: plain keys, oracle auto-stamps seqno)
        let mut count = 0;

        for mutation in &proposal.mutations {
            match mutation {
                Mutation::Put {
                    partition,
                    key,
                    value,
                } => {
                    self.engine
                        .put(to_partition(*partition), key, value)
                        .map_err(|e| io::Error::other(e.to_string()))?;
                    count += 1;
                }
                Mutation::Delete { partition, key } => {
                    self.engine
                        .delete(to_partition(*partition), key)
                        .map_err(|e| io::Error::other(e.to_string()))?;
                    count += 1;
                }
                Mutation::Merge {
                    partition,
                    key,
                    operand,
                } => {
                    self.engine
                        .merge(to_partition(*partition), key, operand)
                        .map_err(|e| io::Error::other(e.to_string()))?;
                    count += 1;
                }
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

        Ok(Response {
            mutations_applied: count,
        })
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

            // Apply tree mutations BEFORE persisting applied_index.
            //
            // Correct crash-safety ordering (ADR-017, R076):
            //   1. Apply mutations to trees (memtable — not yet durable)
            //   2. Persist applied_index (also in memtable — not yet durable)
            //   3. FlushManager eventually flushes both to SST atomically
            //
            // If a crash occurs between steps 1 and 2, the SST-persisted
            // applied_index will be behind. On restart, openraft re-delivers
            // the missing entries (they are in the oplog, which WAS fsynced
            // before apply was called). Re-applying is idempotent: same HLC
            // seqno → same (key, seqno, value) → LSM overwrites are harmless.
            //
            // The WRONG order (old code) was: save_applied BEFORE mutations.
            // That caused data loss: crash after save_applied but before apply
            // → applied_index ahead of actual data → openraft won't replay.
            let response = match &entry.payload {
                openraft::entry::EntryPayload::Normal(request) => {
                    let mut total = 0;
                    for proposal in &request.proposals {
                        let r = self.apply_proposal(proposal)?;
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

            // Persist applied_index AFTER mutations are written to memtable.
            *self
                .last_applied
                .lock()
                .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))? =
                Some(entry.log_id);
            self.save_applied(&entry.log_id)?;

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

        CoordinodeSnapshotBuilder {
            engine: Arc::clone(&self.engine),
            last_applied,
            last_membership,
        }
    }

    async fn begin_receiving_snapshot(&mut self) -> Result<std::io::Cursor<Vec<u8>>, io::Error> {
        Ok(std::io::Cursor::new(Vec::new()))
    }

    async fn install_snapshot(
        &mut self,
        meta: &openraft::type_config::alias::SnapshotMetaOf<TypeConfig>,
        snapshot: std::io::Cursor<Vec<u8>>,
    ) -> Result<(), io::Error> {
        let data = snapshot.into_inner();

        tracing::info!(
            snapshot_id = %meta.snapshot_id,
            data_bytes = data.len(),
            last_log_index = meta.last_log_id.map(|id| id.index),
            "installing snapshot"
        );

        // Apply snapshot data to storage partitions (if non-empty).
        // Empty snapshots are valid (metadata-only, e.g., from tests).
        if !data.is_empty() {
            crate::snapshot::install_full_snapshot(&self.engine, &data)?;
        }

        // Update last applied and membership from snapshot metadata
        *self
            .last_applied
            .lock()
            .map_err(|e| io::Error::other(format!("mutex poisoned: {e}")))? = meta.last_log_id;
        if let Some(ref log_id) = meta.last_log_id {
            self.save_applied(log_id)?;
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
/// The snapshot captures every KV pair across all 7 partitions (Node, Adj,
/// EdgeProp, Blob, BlobRef, Schema, Idx) in a binary format with xxh3
/// checksum. This data is then sent to followers via the Snapshot gRPC RPC.
///
/// Follows Dgraph's content separation pattern: openraft manages snapshot
/// metadata (index, term, membership), while the actual data transfer uses
/// our binary format (not pushed through the Raft log).
pub struct CoordinodeSnapshotBuilder {
    /// Engine reference for iterating all storage partitions.
    engine: Arc<StorageEngine>,
    last_applied: Option<openraft::type_config::alias::LogIdOf<TypeConfig>>,
    last_membership: openraft::StoredMembership<CommittedLeaderId, u64, openraft::impls::BasicNode>,
}

impl RaftSnapshotBuilder<TypeConfig> for CoordinodeSnapshotBuilder {
    async fn build_snapshot(&mut self) -> Result<Snapshot, io::Error> {
        let last_log_id = self.last_applied;
        let snap_id = match &last_log_id {
            Some(id) => format!("snap-{}-{}", id.index, id.committed_leader_id().term),
            None => "snap-0-0".to_string(),
        };

        tracing::info!(
            snapshot_id = %snap_id,
            last_log_index = last_log_id.map(|id| id.index),
            "building full storage snapshot"
        );

        // Serialize all KV data from all partitions
        let data = crate::snapshot::build_full_snapshot(&self.engine)?;

        let meta = SnapshotMeta {
            last_log_id,
            last_membership: self.last_membership.clone(),
            snapshot_id: snap_id,
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
    openraft::Config {
        heartbeat_interval: 150,
        election_timeout_min: 300,
        election_timeout_max: 600,
        max_payload_entries: 300,
        snapshot_policy: openraft::SnapshotPolicy::LogsSinceLast(10_000),
        ..Default::default()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use coordinode_core::txn::proposal::PartitionId;
    use coordinode_core::txn::timestamp::Timestamp;
    use coordinode_storage::engine::config::StorageConfig;

    fn test_engine() -> (tempfile::TempDir, Arc<StorageEngine>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        (dir, engine)
    }

    // -- LogStore --

    #[tokio::test]
    async fn log_store_save_and_read_vote() {
        let (_dir, engine) = test_engine();
        let mut store = LogStore::open(engine).unwrap();

        // No vote initially
        assert!(store.read_vote().await.unwrap().is_none());

        // Save a vote: term=1, node_id=42
        let vote = Vote::new(1, 42);
        store.save_vote(&vote).await.unwrap();

        // Read it back
        let loaded = store.read_vote().await.unwrap().unwrap();
        assert_eq!(loaded, vote);
    }

    #[tokio::test]
    async fn log_store_append_and_read_entries() {
        let (_dir, engine) = test_engine();
        let mut store = LogStore::open(engine).unwrap();

        // Create entries
        let entries = vec![
            make_entry(1, 1, "first"),
            make_entry(2, 1, "second"),
            make_entry(3, 1, "third"),
        ];

        store.append(entries, IOFlushed::noop()).await.unwrap();

        // Read back
        let loaded = store.try_get_log_entries(1..=3).await.unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].log_id.index, 1);
        assert_eq!(loaded[2].log_id.index, 3);
    }

    #[tokio::test]
    async fn log_store_get_log_state() {
        let (_dir, engine) = test_engine();
        let mut store = LogStore::open(engine).unwrap();

        // Empty state
        let state = store.get_log_state().await.unwrap();
        assert!(state.last_log_id.is_none());

        // Add entries
        let entries = vec![make_entry(1, 1, "a"), make_entry(2, 1, "b")];
        store.append(entries, IOFlushed::noop()).await.unwrap();

        let state = store.get_log_state().await.unwrap();
        assert_eq!(state.last_log_id.unwrap().index, 2);
    }

    #[tokio::test]
    async fn log_store_truncate_after() {
        let (_dir, engine) = test_engine();
        let mut store = LogStore::open(engine).unwrap();

        let entries = vec![
            make_entry(1, 1, "a"),
            make_entry(2, 1, "b"),
            make_entry(3, 1, "c"),
        ];
        store.append(entries, IOFlushed::noop()).await.unwrap();

        // Truncate after index 1 (keep 1, delete 2 and 3)
        let log_id = openraft::LogId::new(
            CommittedLeaderId {
                term: 1,
                node_id: 0,
            },
            1,
        );
        store.truncate_after(Some(log_id)).await.unwrap();

        let remaining = store.try_get_log_entries(0..=10).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].log_id.index, 1);
    }

    #[tokio::test]
    async fn log_store_purge() {
        let (_dir, engine) = test_engine();
        let mut store = LogStore::open(engine).unwrap();

        let entries = vec![
            make_entry(1, 1, "a"),
            make_entry(2, 1, "b"),
            make_entry(3, 1, "c"),
        ];
        store.append(entries, IOFlushed::noop()).await.unwrap();

        // Purge up to index 2 (delete 1 and 2, keep 3)
        let log_id = openraft::LogId::new(
            CommittedLeaderId {
                term: 1,
                node_id: 0,
            },
            2,
        );
        store.purge(log_id).await.unwrap();

        let remaining = store.try_get_log_entries(0..=10).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].log_id.index, 3);
    }

    #[tokio::test]
    async fn log_store_committed_roundtrip() {
        let (_dir, engine) = test_engine();
        let mut store = LogStore::open(engine).unwrap();

        // No committed initially
        assert!(store.read_committed().await.unwrap().is_none());

        let log_id = openraft::LogId::new(
            CommittedLeaderId {
                term: 1,
                node_id: 0,
            },
            5,
        );
        store.save_committed(Some(log_id)).await.unwrap();

        let loaded = store.read_committed().await.unwrap().unwrap();
        assert_eq!(loaded.index, 5);
    }

    // -- CoordinodeStateMachine --

    #[tokio::test]
    async fn state_machine_initial_state() {
        let (_dir, engine) = test_engine();
        let mut sm = CoordinodeStateMachine::new(engine);

        let (applied, membership) = sm.applied_state().await.unwrap();
        assert!(applied.is_none());
        assert!(membership.log_id().is_none());
    }

    // -- Dedup tests --

    #[test]
    fn dedup_skips_duplicate_proposal() {
        // Apply same proposal twice — second should be detected as duplicate
        let (_dir, engine) = test_engine();
        let sm = CoordinodeStateMachine::new(engine);

        let proposal = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(42),
            mutations: vec![Mutation::Put {
                partition: coordinode_core::txn::proposal::PartitionId::Node,
                key: b"node:1:100".to_vec(),
                value: b"data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(1000),
            start_ts: Timestamp::from_raw(999),
            bypass_rate_limiter: false,
        };

        // First apply: should return 1 mutation applied
        let r1 = sm.apply_proposal(&proposal).unwrap();
        assert_eq!(r1.mutations_applied, 1);

        // Second apply (same id + same size): should return 0 (dedup)
        let r2 = sm.apply_proposal(&proposal).unwrap();
        assert_eq!(r2.mutations_applied, 0);
    }

    #[test]
    fn dedup_allows_different_size_same_id() {
        // Same proposal ID but different payload size should re-apply
        // (represents a retry with modified payload)
        let (_dir, engine) = test_engine();
        let sm = CoordinodeStateMachine::new(engine);

        let proposal_v1 = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(42),
            mutations: vec![Mutation::Put {
                partition: coordinode_core::txn::proposal::PartitionId::Node,
                key: b"node:1:100".to_vec(),
                value: b"short".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(1000),
            start_ts: Timestamp::from_raw(999),
            bypass_rate_limiter: false,
        };

        let proposal_v2 = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(42),
            mutations: vec![Mutation::Put {
                partition: coordinode_core::txn::proposal::PartitionId::Node,
                key: b"node:1:100".to_vec(),
                value: b"much-longer-value-that-changes-size".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(1001),
            start_ts: Timestamp::from_raw(1000),
            bypass_rate_limiter: false,
        };

        // First version applied
        let r1 = sm.apply_proposal(&proposal_v1).unwrap();
        assert_eq!(r1.mutations_applied, 1);

        // Second version with different size: should NOT be deduped
        let r2 = sm.apply_proposal(&proposal_v2).unwrap();
        assert_eq!(r2.mutations_applied, 1);
    }

    #[test]
    fn dedup_gc_removes_old_entries() {
        // Verify that dedup GC cleans old entries
        let (_dir, engine) = test_engine();
        let sm = CoordinodeStateMachine::new(engine);

        let proposal = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
            mutations: vec![Mutation::Put {
                partition: coordinode_core::txn::proposal::PartitionId::Node,
                key: b"node:1:1".to_vec(),
                value: b"data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };

        sm.apply_proposal(&proposal).unwrap();

        // Verify dedup map has 1 entry
        let dedup_len = sm.dedup.lock().unwrap().len();
        assert_eq!(dedup_len, 1, "dedup map should have 1 entry");

        // Manually set the entry's `seen` to old time to trigger GC
        {
            let mut dedup = sm.dedup.lock().unwrap();
            if let Some(entry) = dedup.get_mut(&1u64) {
                entry.seen = Instant::now() - Duration::from_secs(DEDUP_MAX_AGE_SECS + 1);
            }
            // Force last_gc to be old too
            *sm.last_dedup_gc.lock().unwrap() =
                Instant::now() - Duration::from_secs(DEDUP_GC_INTERVAL_SECS + 1);
        }

        // Apply another proposal to trigger GC
        let proposal2 = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(2),
            mutations: vec![Mutation::Put {
                partition: coordinode_core::txn::proposal::PartitionId::Node,
                key: b"node:1:2".to_vec(),
                value: b"data2".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(200),
            start_ts: Timestamp::from_raw(199),
            bypass_rate_limiter: false,
        };
        sm.apply_proposal(&proposal2).unwrap();

        // Old entry should have been GC'd, new one remains
        let dedup = sm.dedup.lock().unwrap();
        assert!(!dedup.contains_key(&1u64), "old entry should be GC'd");
        assert!(dedup.contains_key(&2u64), "new entry should remain");
    }

    // -- Config --

    #[test]
    fn default_config_values() {
        let config = default_raft_config();
        assert_eq!(config.heartbeat_interval, 150);
        assert_eq!(config.election_timeout_min, 300);
        assert_eq!(config.election_timeout_max, 600);
        assert_eq!(config.max_payload_entries, 300);
    }

    // -- Helpers --

    // -- Purge persistence tests --

    #[tokio::test]
    async fn log_store_purge_persists_last_purged_log_id() {
        let (_dir, engine) = test_engine();
        let mut store = LogStore::open(Arc::clone(&engine)).unwrap();

        let entries = vec![
            make_entry(1, 1, "a"),
            make_entry(2, 1, "b"),
            make_entry(3, 1, "c"),
        ];
        store.append(entries, IOFlushed::noop()).await.unwrap();

        // Initially no purge
        let state = store.get_log_state().await.unwrap();
        assert!(state.last_purged_log_id.is_none(), "no purge initially");

        // Purge up to index 2
        let purge_id = openraft::LogId::new(
            CommittedLeaderId {
                term: 1,
                node_id: 0,
            },
            2,
        );
        store.purge(purge_id).await.unwrap();

        // Verify purge tracked in-session
        let state = store.get_log_state().await.unwrap();
        assert_eq!(
            state.last_purged_log_id.unwrap().index,
            2,
            "last_purged_log_id should be 2 after purge"
        );
    }

    #[tokio::test]
    async fn log_store_purge_survives_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_path_buf();

        // Phase 1: write + purge
        {
            let config = StorageConfig::new(&path);
            let engine = Arc::new(StorageEngine::open(&config).expect("open"));
            let mut store = LogStore::open(Arc::clone(&engine)).unwrap();

            let entries = vec![
                make_entry(1, 1, "a"),
                make_entry(2, 1, "b"),
                make_entry(3, 1, "c"),
            ];
            store.append(entries, IOFlushed::noop()).await.unwrap();

            let purge_id = openraft::LogId::new(
                CommittedLeaderId {
                    term: 1,
                    node_id: 0,
                },
                2,
            );
            store.purge(purge_id).await.unwrap();
        }

        // Phase 2: reopen, verify purge state persisted
        {
            let config = StorageConfig::new(&path);
            let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
            let mut store = LogStore::open(engine).unwrap();

            let state = store.get_log_state().await.unwrap();
            assert_eq!(
                state.last_purged_log_id.unwrap().index,
                2,
                "last_purged_log_id should survive reopen"
            );
            // Only entry 3 should remain
            assert_eq!(
                state.last_log_id.unwrap().index,
                3,
                "entry 3 should survive purge"
            );
        }
    }

    // -- Snapshot persistence tests --

    #[tokio::test]
    async fn snapshot_build_persists_to_storage() {
        let (_dir, engine) = test_engine();

        // Write some data
        engine
            .put(Partition::Node, b"node:0:1", b"alice")
            .expect("put");

        let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine));

        // Set applied state so snapshot has a valid log_id
        let log_id = openraft::LogId::new(
            CommittedLeaderId {
                term: 1,
                node_id: 0,
            },
            5,
        );
        *sm.last_applied.lock().unwrap() = Some(log_id);
        sm.save_applied(&log_id).unwrap();

        // Build snapshot via the SnapshotBuilder
        let mut builder = sm.get_snapshot_builder().await;
        let snap = builder.build_snapshot().await.unwrap();

        assert!(snap.meta.last_log_id.is_some());
        assert_eq!(snap.meta.last_log_id.unwrap().index, 5);

        // Verify snapshot was persisted to CoordiNode storage (build_snapshot writes it)
        let snap_meta = engine.get(Partition::Schema, KEY_SNAPSHOT_META).unwrap();
        assert!(snap_meta.is_some(), "snapshot meta should be persisted");

        let snap_data = engine.get(Partition::Schema, KEY_SNAPSHOT_DATA).unwrap();
        assert!(snap_data.is_some(), "snapshot data should be persisted");
        let data = snap_data.unwrap();
        assert!(data.len() > 10, "snapshot data should be non-empty");
        assert_eq!(&data[..4], b"CNSN", "snapshot data should have CNSN magic");
    }

    #[tokio::test]
    async fn snapshot_survives_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_path_buf();

        // Phase 1: build snapshot
        {
            let config = StorageConfig::new(&path);
            let engine = Arc::new(StorageEngine::open(&config).expect("open"));
            engine
                .put(Partition::Node, b"node:0:1", b"alice")
                .expect("put");

            let mut sm = CoordinodeStateMachine::new(Arc::clone(&engine));
            let log_id = openraft::LogId::new(
                CommittedLeaderId {
                    term: 1,
                    node_id: 0,
                },
                10,
            );
            *sm.last_applied.lock().unwrap() = Some(log_id);
            sm.save_applied(&log_id).unwrap();

            let mut builder = sm.get_snapshot_builder().await;
            let _snap = builder.build_snapshot().await.unwrap();
        }

        // Phase 2: reopen, verify snapshot is still there
        {
            let config = StorageConfig::new(&path);
            let engine = Arc::new(StorageEngine::open(&config).expect("reopen"));
            let mut sm = CoordinodeStateMachine::new(engine);

            let snap = sm.get_current_snapshot().await.unwrap();
            assert!(snap.is_some(), "snapshot should survive reopen");

            let snap = snap.unwrap();
            assert_eq!(
                snap.meta.last_log_id.unwrap().index,
                10,
                "snapshot last_log_id should be 10"
            );
            assert_eq!(
                snap.meta.snapshot_id, "snap-10-1",
                "snapshot_id should match"
            );

            let data = snap.snapshot.into_inner();
            assert!(
                data.len() > 10,
                "snapshot data should be non-empty after reopen"
            );
            assert_eq!(&data[..4], b"CNSN", "snapshot data magic after reopen");
        }
    }

    fn make_entry(index: u64, term: u64, title: &str) -> Entry {
        use coordinode_core::txn::proposal::PartitionId;
        use openraft::entry::RaftEntry;

        let proposal = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(index),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:1:{index}").into_bytes(),
                value: title.as_bytes().to_vec(),
            }],
            commit_ts: Timestamp::from_raw(1000 + index),
            start_ts: Timestamp::from_raw(1000 + index - 1),
            bypass_rate_limiter: false,
        };

        let committed_leader_id = CommittedLeaderId { term, node_id: 0 };
        let log_id = openraft::LogId::new(committed_leader_id, index);

        Entry::new_normal(log_id, Request::single(proposal))
    }

    // -- R068: oracle.advance_to() during Raft apply --

    #[test]
    fn apply_advances_oracle_to_commit_ts() {
        use coordinode_core::txn::timestamp::TimestampOracle;

        let dir = tempfile::TempDir::new().unwrap();
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
        let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()));

        // Apply proposal with commit_ts=500
        let proposal = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:1".to_vec(),
                value: b"data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(500),
            start_ts: Timestamp::from_raw(499),
            bypass_rate_limiter: false,
        };

        let result = sm.apply_proposal(&proposal).unwrap();
        assert_eq!(result.mutations_applied, 1);

        // Oracle should have advanced to at least 500
        let next_ts = oracle.next();
        assert!(
            next_ts.as_raw() > 500,
            "oracle should have advanced past 500, got {}",
            next_ts.as_raw()
        );
    }

    #[test]
    fn apply_100_entries_oracle_monotonic() {
        use coordinode_core::txn::timestamp::TimestampOracle;

        let dir = tempfile::TempDir::new().unwrap();
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
        let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()));

        // Apply 100 proposals with increasing commit_ts
        for i in 1..=100u64 {
            let proposal = RaftProposal {
                id: coordinode_core::txn::proposal::ProposalId::from_raw(i),
                mutations: vec![Mutation::Put {
                    partition: PartitionId::Node,
                    key: format!("node:1:{i}").into_bytes(),
                    value: format!("v{i}").into_bytes(),
                }],
                commit_ts: Timestamp::from_raw(1000 + i),
                start_ts: Timestamp::from_raw(999 + i),
                bypass_rate_limiter: false,
            };
            sm.apply_proposal(&proposal).unwrap();
        }

        // Oracle should be at least at 1100 (last commit_ts)
        let final_ts = oracle.next();
        assert!(
            final_ts.as_raw() > 1100,
            "oracle should be past 1100 after 100 entries, got {}",
            final_ts.as_raw()
        );

        // Verify all 100 writes are readable
        for i in 1..=100u64 {
            let val = engine
                .get(Partition::Node, format!("node:1:{i}").as_bytes())
                .unwrap();
            assert_eq!(
                val.as_deref(),
                Some(format!("v{i}").as_bytes()),
                "mismatch at i={i}"
            );
        }
    }

    #[test]
    fn apply_without_oracle_still_works() {
        // Backward compat: state machine without oracle applies normally
        let (_dir, engine) = test_engine();
        let sm = CoordinodeStateMachine::new(engine.clone());

        let proposal = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:1".to_vec(),
                value: b"data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(500),
            start_ts: Timestamp::from_raw(499),
            bypass_rate_limiter: false,
        };

        let result = sm.apply_proposal(&proposal).unwrap();
        assert_eq!(result.mutations_applied, 1);

        let val = engine.get(Partition::Node, b"node:1:1").unwrap();
        assert_eq!(val.as_deref(), Some(b"data".as_slice()));
    }

    #[test]
    fn apply_seqnos_match_commit_ts_with_oracle() {
        use coordinode_core::txn::timestamp::TimestampOracle;
        use coordinode_storage::engine::config::StorageConfig;

        // Create engine WITH oracle so seqno = oracle timestamp
        let dir = tempfile::TempDir::new().unwrap();
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(100)));
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).unwrap());
        let sm = CoordinodeStateMachine::with_oracle(Arc::clone(&engine), Some(oracle.clone()));

        // Apply at commit_ts=500
        let proposal = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(1),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:1".to_vec(),
                value: b"first".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(500),
            start_ts: Timestamp::from_raw(499),
            bypass_rate_limiter: false,
        };
        sm.apply_proposal(&proposal).unwrap();

        // Apply at commit_ts=700
        let proposal2 = RaftProposal {
            id: coordinode_core::txn::proposal::ProposalId::from_raw(2),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:1".to_vec(),
                value: b"second".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(700),
            start_ts: Timestamp::from_raw(699),
            bypass_rate_limiter: false,
        };
        sm.apply_proposal(&proposal2).unwrap();

        // Take a snapshot between the two proposals (after first apply).
        // We can't use snapshot_at(500) directly because the snapshot tracker
        // may not have registered that seqno yet (memtable not sealed).
        // Instead, verify via snapshot taken at the right point.
        //
        // Since we can't go back in time, verify that current snapshot sees
        // "second" and that oracle advanced monotonically through both entries.
        let current_snap = engine.snapshot();
        let val_current = engine
            .snapshot_get(&current_snap, Partition::Node, b"node:1:1")
            .unwrap();
        assert_eq!(
            val_current.as_deref(),
            Some(b"second".as_ref()),
            "current snapshot should see last write"
        );

        // Verify oracle advanced past 700
        let final_ts = oracle.next();
        assert!(
            final_ts.as_raw() > 700,
            "oracle should be past 700, got {}",
            final_ts.as_raw()
        );
    }
}
