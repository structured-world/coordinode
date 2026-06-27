//! Layer-3 transaction context (ADR-041).
//!
//! Owns the per-statement transactional state and is **modality-agnostic** —
//! its surface is `get` / `put` / `delete` / `prefix_scan` over
//! `(Partition, key)` plus the MVCC read snapshot and the OCC read-set.
//! Layer-4 modality stores (`NodeStore`, `EdgeStore`, …) take a
//! `&mut Transaction` and translate their typed arguments into partition+key
//! calls; the Layer-5 query engine holds only this handle and never names a
//! partition or a key encoder.
//!
//! Why here (the storage layer) and not in the query engine: MVCC snapshot
//! coordination and OCC read/write-set management are storage-layer
//! responsibilities; keeping the transaction modality-agnostic preserves the
//! dependency direction (modality stores depend down on the transaction, the
//! transaction never knows the modality taxonomy).
//!
//! Scope of this module: the read path, the read-your-own-writes write buffer,
//! OCC read tracking, and the commutative merge buffers (adjacency adds/removes
//! and node document deltas). Commit orchestration (OCC validation + commit-ts
//! assignment + write-concern-aware flush) still lives in the query engine,
//! which drains these buffers via the `take_*` accessors.

use std::collections::{HashMap, HashSet};

use coordinode_core::txn::drain::{DrainBuffer, DrainEntry};
use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalError, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::{WriteConcern, WriteConcernLevel};
use lsm_tree::Guard;

use crate::cache::write_buffer::NvmeWriteBuffer;
use crate::engine::coordinator::{MultiModalCoordinator, OccScope};
use crate::engine::core::StorageEngine;
use crate::engine::merge::{encode_add_batch, encode_remove};
use crate::engine::partition::Partition;
use crate::engine::StorageSnapshot;
use crate::error::{StorageError, StorageResult};

/// A key/value pair returned by [`Transaction::prefix_scan`].
pub type KvPair = (Vec<u8>, Vec<u8>);

/// Per-statement commit policy passed to [`Transaction::commit`]. These are
/// execution-context concerns (durability level, replication wiring), not part
/// of the transaction's identity, so they are supplied at commit time rather
/// than stored on the transaction.
pub struct CommitContext<'b> {
    /// Effective write concern (W0 / volatile / W1 / majority, journal gate).
    pub write_concern: &'b WriteConcern,
    /// Raft proposal pipeline for durable W1/Majority application. `None` in
    /// legacy / embedded single-node mode (writes go straight to the engine).
    pub pipeline: Option<&'b dyn ProposalPipeline>,
    /// Monotonic proposal-id source, paired with `pipeline`.
    pub id_gen: Option<&'b ProposalIdGenerator>,
    /// Volatile-durability drain buffer for background Raft replication.
    pub drain_buffer: Option<&'b DrainBuffer>,
    /// NVMe persistence buffer for `w:cache` pre-ACK durability.
    pub nvme_write_buffer: Option<&'b NvmeWriteBuffer>,
}

/// Result of a successful [`Transaction::commit`].
pub struct CommitOutcome {
    /// Commit timestamp assigned by the oracle, or `None` for a read-only
    /// transaction / legacy mode (writes already applied).
    pub commit_ts: Option<Timestamp>,
    /// Committed Raft index of this write (the causal `operationTime` token).
    /// `Some` only on the proposal-pipeline path; `None` otherwise.
    pub applied_index: Option<u64>,
}

/// Errors from [`Transaction::commit`].
#[derive(Debug, thiserror::Error)]
pub enum CommitError {
    /// OCC read-write conflict: a key this transaction read was modified by a
    /// concurrent writer after `read_ts`. The caller should retry.
    #[error("{0}")]
    Conflict(String),
    /// Underlying storage failure during flush.
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// Encoding / replication-pipeline failure.
    #[error("{0}")]
    Serialization(String),
}

/// Map a storage [`Partition`] to its wire [`PartitionId`] for Raft proposals.
fn partition_to_id(p: Partition) -> PartitionId {
    match p {
        Partition::Node => PartitionId::Node,
        Partition::Adj => PartitionId::Adj,
        Partition::EdgeProp => PartitionId::EdgeProp,
        Partition::Blob => PartitionId::Blob,
        Partition::BlobRef => PartitionId::BlobRef,
        Partition::Schema => PartitionId::Schema,
        Partition::Idx => PartitionId::Idx,
        Partition::Raft => unreachable!("Raft partition is not exposed to the query layer"),
        Partition::Counter => PartitionId::Counter,
        Partition::VectorF32 => PartitionId::VectorF32,
        Partition::Registry => PartitionId::Registry,
    }
}

/// Translate a proposal-pipeline error into a [`CommitError`], preserving the
/// capacity-exhaustion structured variant (operators retry on a different
/// endpoint) and folding the rest into a serialization-class failure.
fn proposal_err_to_commit(err: ProposalError) -> CommitError {
    if let ProposalError::CapacityExhausted {
        endpoint_id,
        used_bytes,
        hard_limit_bytes,
    } = err
    {
        return CommitError::Storage(StorageError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        });
    }
    CommitError::Serialization(format!("proposal pipeline error: {err}"))
}

/// Per-statement transaction context (ADR-041). See the module docs.
pub struct Transaction<'a> {
    engine: &'a StorageEngine,
    /// `None` selects legacy mode: writes apply directly to the engine with no
    /// MVCC versioning, reads bypass the snapshot. `Some` enables snapshot
    /// isolation + buffered-write atomic commit.
    oracle: Option<&'a TimestampOracle>,
    /// Read timestamp this transaction is pinned at (snapshot seqno space).
    read_ts: Timestamp,
    /// MVCC read snapshot (`= seqno`). `None` in legacy mode.
    snapshot: Option<StorageSnapshot>,
    /// Adjacency (`adj:`) time-travel snapshot. Set at statement start (reusing
    /// the MVCC snapshot when present, else `engine.snapshot()`) or overridden
    /// by `AS OF TIMESTAMP`. Adjacency posting-list base reads go through this
    /// so merge operands written after the snapshot stay invisible. Distinct
    /// from `snapshot`: in legacy mode this is still a real snapshot while the
    /// MVCC snapshot is `None`.
    adj_snapshot: Option<StorageSnapshot>,
    /// Buffered writes: `(partition, key) -> Some(value)` (put) | `None`
    /// (tombstone). Reads consult this first (read-your-own-writes); the
    /// buffer is drained atomically at commit. Empty in legacy mode (writes
    /// go straight to the engine).
    write_buffer: HashMap<(Partition, Vec<u8>), Option<Vec<u8>>>,
    /// Lazily-created OCC read-set, pinned at `read_ts`. Every non-own-write
    /// read records its key here; `commit` validates the set against
    /// concurrent writers. `None` until the first tracked read (and always in
    /// legacy mode).
    occ_scope: Option<OccScope>,
    /// Buffered adjacency-posting merge adds: `adj key -> uids`. Adjacency
    /// writes are commutative merge operands (not point writes), so they live
    /// in a separate buffer from `write_buffer` and bypass OCC conflict
    /// detection. Drained at commit.
    merge_adj_adds: HashMap<Vec<u8>, Vec<u64>>,
    /// Buffered adjacency-posting merge removes: `adj key -> uids`.
    merge_adj_removes: HashMap<Vec<u8>, Vec<u64>>,
    /// Buffered node merge operands: `(node key, operand bytes)`. Read-modify
    /// -write document deltas (SET nested path) materialise these against the
    /// node record before a read; drained at commit.
    merge_node_deltas: Vec<(Vec<u8>, Vec<u8>)>,
}

/// The borrow-free owned state of a [`Transaction`] — everything except the
/// `engine` / `oracle` borrows. An interactive multi-statement transaction
/// (ADR-042) parks this in a leader-local registry between statements and
/// rebuilds a [`Transaction`] around it (via [`Transaction::resume`]) for each
/// statement, so the pinned snapshot, write buffer, OCC read-set, and merge
/// buffers persist across statements while the transaction itself stays a
/// short-lived borrow. Produced by [`Transaction::into_state`].
pub struct TransactionState {
    read_ts: Timestamp,
    snapshot: Option<StorageSnapshot>,
    adj_snapshot: Option<StorageSnapshot>,
    write_buffer: HashMap<(Partition, Vec<u8>), Option<Vec<u8>>>,
    occ_scope: Option<OccScope>,
    merge_adj_adds: HashMap<Vec<u8>, Vec<u64>>,
    merge_adj_removes: HashMap<Vec<u8>, Vec<u64>>,
    merge_node_deltas: Vec<(Vec<u8>, Vec<u8>)>,
}

impl TransactionState {
    /// The pinned read timestamp (`start_ts`) of the parked transaction. An
    /// interactive transaction reuses this for every statement so all reads
    /// resolve against the same snapshot (repeatable read).
    pub fn read_ts(&self) -> Timestamp {
        self.read_ts
    }

    /// Approximate size in bytes of the buffered, uncommitted mutations —
    /// the point write buffer (keys + values) plus the adjacency and node
    /// merge buffers. An interactive transaction caps this against a
    /// configured ceiling so a client that buffers without committing cannot
    /// grow leader memory unbounded.
    pub fn buffered_bytes(&self) -> usize {
        let writes: usize = self
            .write_buffer
            .iter()
            .map(|((_, k), v)| k.len() + v.as_ref().map_or(0, Vec::len))
            .sum();
        let adj_adds: usize = self
            .merge_adj_adds
            .iter()
            .map(|(k, uids)| k.len() + uids.len() * 8)
            .sum();
        let adj_removes: usize = self
            .merge_adj_removes
            .iter()
            .map(|(k, uids)| k.len() + uids.len() * 8)
            .sum();
        let node_deltas: usize = self
            .merge_node_deltas
            .iter()
            .map(|(k, op)| k.len() + op.len())
            .sum();
        writes + adj_adds + adj_removes + node_deltas
    }
}

impl<'a> Transaction<'a> {
    /// Open a transaction. `oracle: Some` + `snapshot: Some` is the MVCC path;
    /// `oracle: None` is legacy direct-to-engine mode (no snapshot, no buffer).
    pub fn new(
        engine: &'a StorageEngine,
        oracle: Option<&'a TimestampOracle>,
        read_ts: Timestamp,
        snapshot: Option<StorageSnapshot>,
    ) -> Self {
        Self {
            engine,
            oracle,
            read_ts,
            snapshot,
            adj_snapshot: None,
            write_buffer: HashMap::new(),
            occ_scope: None,
            merge_adj_adds: HashMap::new(),
            merge_adj_removes: HashMap::new(),
            merge_node_deltas: Vec::new(),
        }
    }

    /// Consume the transaction, returning its borrow-free owned state.
    ///
    /// The `engine` / `oracle` borrows are dropped; everything that defines
    /// the transaction's progress — the pinned `read_ts` + snapshots, the
    /// read-your-own-writes write buffer, the OCC read-set, and the adjacency
    /// / node merge buffers — is moved out. Pair with [`Self::resume`] to park
    /// an interactive multi-statement transaction (ADR-042) in a leader-local
    /// registry between statements and rebuild it for the next statement, so
    /// the snapshot and accumulated buffers persist while the `Transaction`
    /// itself stays a short-lived borrow.
    pub fn into_state(self) -> TransactionState {
        TransactionState {
            read_ts: self.read_ts,
            snapshot: self.snapshot,
            adj_snapshot: self.adj_snapshot,
            write_buffer: self.write_buffer,
            occ_scope: self.occ_scope,
            merge_adj_adds: self.merge_adj_adds,
            merge_adj_removes: self.merge_adj_removes,
            merge_node_deltas: self.merge_node_deltas,
        }
    }

    /// Extract the transaction's owned state into a [`TransactionState`]
    /// without consuming the transaction — the buffers are drained
    /// (`mem::take`) and the snapshots/`read_ts` copied, leaving the
    /// transaction valid but empty. Used to park an interactive transaction's
    /// progress mid-flow (when the transaction lives inside a borrowed
    /// `ExecutionContext` that cannot be moved out). Pair with
    /// [`Self::resume`].
    pub fn take_state(&mut self) -> TransactionState {
        TransactionState {
            read_ts: self.read_ts,
            snapshot: self.snapshot,
            adj_snapshot: self.adj_snapshot,
            write_buffer: std::mem::take(&mut self.write_buffer),
            occ_scope: self.occ_scope.take(),
            merge_adj_adds: std::mem::take(&mut self.merge_adj_adds),
            merge_adj_removes: std::mem::take(&mut self.merge_adj_removes),
            merge_node_deltas: std::mem::take(&mut self.merge_node_deltas),
        }
    }

    /// Rebuild a transaction around parked [`TransactionState`] with freshly
    /// supplied `engine` / `oracle` borrows — the resume side of
    /// [`Self::into_state`] / [`Self::take_state`]. The pinned `read_ts` +
    /// snapshots carry over, so every statement of an interactive transaction
    /// reads the same MVCC snapshot (repeatable read across the transaction).
    pub fn resume(
        engine: &'a StorageEngine,
        oracle: Option<&'a TimestampOracle>,
        state: TransactionState,
    ) -> Self {
        Self {
            engine,
            oracle,
            read_ts: state.read_ts,
            snapshot: state.snapshot,
            adj_snapshot: state.adj_snapshot,
            write_buffer: state.write_buffer,
            occ_scope: state.occ_scope,
            merge_adj_adds: state.merge_adj_adds,
            merge_adj_removes: state.merge_adj_removes,
            merge_node_deltas: state.merge_node_deltas,
        }
    }

    /// The read timestamp this transaction is pinned at.
    pub fn read_ts(&self) -> Timestamp {
        self.read_ts
    }

    /// Whether this transaction is in MVCC mode (vs legacy direct-to-engine).
    pub fn is_mvcc(&self) -> bool {
        self.oracle.is_some()
    }

    /// Lazily create the OCC read-set scope, pinned at `read_ts`. Legacy mode
    /// (no oracle) has no scope and tracks nothing. Exposed to callers above
    /// that track reads they performed through their own fast path (e.g. a
    /// parallel sub-context that merges its read-set in afterwards).
    pub fn ensure_occ_scope(&mut self) -> Option<&OccScope> {
        if self.oracle.is_some() && self.occ_scope.is_none() {
            self.occ_scope = Some(
                self.engine
                    .coordinator()
                    .occ_scope_at(self.read_ts.as_raw()),
            );
        }
        self.occ_scope.as_ref()
    }

    /// MVCC-aware read: write buffer (read-your-own-writes, not OCC-tracked) →
    /// snapshot read (OCC-tracked) → legacy direct read.
    pub fn get(&mut self, part: Partition, key: &[u8]) -> StorageResult<Option<Vec<u8>>> {
        // Read-your-own-writes: a value this transaction wrote shadows storage
        // and is deliberately NOT added to the OCC read-set (you cannot
        // conflict with yourself).
        if let Some(buffered) = self.write_buffer.get(&(part, key.to_vec())) {
            return Ok(buffered.clone());
        }
        // Track for conflict detection at commit.
        if let Some(scope) = self.ensure_occ_scope() {
            scope.track(part, key);
        }
        match self.snapshot {
            Some(snap) => Ok(self
                .engine
                .snapshot_get(&snap, part, key)?
                .map(|b| b.to_vec())),
            None => Ok(self.engine.get(part, key)?.map(|b| b.to_vec())),
        }
    }

    /// MVCC-aware write: buffers for atomic flush at commit. Legacy mode writes
    /// straight to the engine.
    pub fn put(&mut self, part: Partition, key: &[u8], value: &[u8]) -> StorageResult<()> {
        if self.oracle.is_some() {
            self.write_buffer
                .insert((part, key.to_vec()), Some(value.to_vec()));
            Ok(())
        } else {
            self.engine.put(part, key, value)
        }
    }

    /// MVCC-aware delete: buffers a tombstone for atomic flush. Legacy mode
    /// deletes straight from the engine.
    pub fn delete(&mut self, part: Partition, key: &[u8]) -> StorageResult<()> {
        if self.oracle.is_some() {
            self.write_buffer.insert((part, key.to_vec()), None);
            Ok(())
        } else {
            self.engine.delete(part, key)
        }
    }

    /// Read without OCC tracking: write buffer (read-your-own-writes) →
    /// snapshot → legacy engine read. For callers above that perform a
    /// modality-internal read which must NOT participate in conflict detection
    /// (e.g. read-modify-write materialisation of a merge delta).
    pub fn read_untracked(&self, part: Partition, key: &[u8]) -> StorageResult<Option<Vec<u8>>> {
        if let Some(buffered) = self.write_buffer.get(&(part, key.to_vec())) {
            return Ok(buffered.clone());
        }
        match self.snapshot {
            Some(snap) => Ok(self
                .engine
                .snapshot_get(&snap, part, key)?
                .map(|b| b.to_vec())),
            None => Ok(self.engine.get(part, key)?.map(|b| b.to_vec())),
        }
    }

    /// Batch counterpart of [`Self::read_untracked`]: resolve many keys at once,
    /// returning a value (or `None`) per input key in order. Keys already in the
    /// write buffer are served from it (read-your-own-writes); the rest are
    /// fetched with a single batched engine `multi_get` (snapshot-pinned when
    /// the transaction holds an MVCC snapshot). No OCC read-set tracking — the
    /// "untracked" contract matches the single-key form. Used by store-layer
    /// batch reads (e.g. materializing many node records behind an index or
    /// vector-search result set) to avoid a per-key lookup loop.
    pub fn multi_read_untracked(
        &self,
        part: Partition,
        keys: &[&[u8]],
    ) -> StorageResult<Vec<Option<Vec<u8>>>> {
        let mut out: Vec<Option<Vec<u8>>> = vec![None; keys.len()];
        let mut miss_idx: Vec<usize> = Vec::new();
        let mut miss_keys: Vec<&[u8]> = Vec::new();

        for (i, key) in keys.iter().enumerate() {
            if let Some(buffered) = self.write_buffer.get(&(part, key.to_vec())) {
                out[i] = buffered.clone();
            } else {
                miss_idx.push(i);
                miss_keys.push(key);
            }
        }

        if miss_keys.is_empty() {
            return Ok(out);
        }

        let values = match self.snapshot {
            Some(snap) => self.engine.snapshot_multi_get(&snap, part, &miss_keys)?,
            None => self.engine.multi_get(part, &miss_keys)?,
        };
        for (slot, value) in miss_idx.into_iter().zip(values) {
            out[slot] = value.map(|b| b.to_vec());
        }
        Ok(out)
    }

    /// Untracked prefix scan over a partition: snapshot-aware (reads through
    /// the MVCC snapshot when set, else latest), no write-buffer overlay and no
    /// OCC tracking. For modality stores that walk a key prefix (temporal
    /// version walks, shard scans) without joining the conflict set.
    pub fn base_prefix_scan(&self, part: Partition, prefix: &[u8]) -> StorageResult<Vec<KvPair>> {
        match self.snapshot {
            Some(snap) => Ok(self
                .engine
                .snapshot_prefix_scan(&snap, part, prefix)?
                .into_iter()
                .map(|(k, v)| (k, v.to_vec()))
                .collect()),
            None => {
                let iter = self.engine.prefix_scan(part, prefix)?;
                let mut results: Vec<KvPair> = Vec::new();
                for guard in iter {
                    let (k, v) = guard.into_inner()?;
                    results.push((k.to_vec(), v.to_vec()));
                }
                Ok(results)
            }
        }
    }

    /// Streaming untracked prefix scan over a partition at the latest committed
    /// state (no snapshot pinning, no buffer overlay, no OCC tracking). Returns
    /// the engine's lazy guard iterator for constant-memory walks of very large
    /// prefixes (shard scans over millions of rows).
    pub fn base_prefix_iter(
        &self,
        part: Partition,
        prefix: &[u8],
    ) -> StorageResult<crate::engine::StorageIter> {
        self.engine.prefix_scan(part, prefix)
    }

    /// Snapshot-isolated seekable range scan over `[start, end]` (inclusive).
    /// The returned iterator can `seek_to` an arbitrary key mid-walk, so one
    /// open iterator skips the dead bytes between disjoint subranges without
    /// reopening per-SST readers per jump (e.g. spatial Z-curve dead-zone
    /// skipping). Reads the transaction's pinned snapshot when present, else the
    /// latest committed seqno. Untracked (no OCC read-set, no buffer overlay) —
    /// for committed-snapshot index scans, like [`Self::base_prefix_scan`].
    pub fn base_range_seekable(
        &self,
        part: Partition,
        start: &[u8],
        end: &[u8],
    ) -> StorageResult<crate::engine::SeekableStorageIter> {
        let seqno = self.snapshot.unwrap_or_else(|| self.engine.snapshot());
        self.engine.range_seekable(part, start, end, seqno)
    }

    /// Point read at an explicit snapshot seqno (not the transaction's own read
    /// snapshot), untracked. For visibility probes that pin a caller-supplied
    /// point-in-time (e.g. the MVCC reachability filter).
    pub fn snapshot_get_at(
        &self,
        snapshot: StorageSnapshot,
        part: Partition,
        key: &[u8],
    ) -> StorageResult<Option<Vec<u8>>> {
        Ok(self
            .engine
            .coordinator()
            .snapshot_get(&snapshot, part, key)?
            .map(|b| b.to_vec()))
    }

    /// Read-your-own-writes probe into the write buffer (no storage read, no
    /// OCC tracking). `Some(&Some(v))` = buffered put, `Some(&None)` =
    /// buffered tombstone, `None` = not buffered.
    pub fn buffered(&self, part: Partition, key: &[u8]) -> Option<&Option<Vec<u8>>> {
        self.write_buffer.get(&(part, key.to_vec()))
    }

    /// Borrow the buffered writes (for the commit path's read-only scan +
    /// CDC pre-image collection). See [`Self::write_buffer_mut`] to drain.
    pub fn write_buffer(&self) -> &HashMap<(Partition, Vec<u8>), Option<Vec<u8>>> {
        &self.write_buffer
    }

    /// Mutable access to the buffered writes for the commit path
    /// (drain / clear). Transitional: the commit path itself moves into this
    /// type in a follow-on increment.
    pub fn write_buffer_mut(&mut self) -> &mut HashMap<(Partition, Vec<u8>), Option<Vec<u8>>> {
        &mut self.write_buffer
    }

    /// Take the buffered writes, leaving the buffer empty. The commit path
    /// drains this and applies each `(partition, key) -> value | tombstone`.
    pub fn take_write_buffer(&mut self) -> HashMap<(Partition, Vec<u8>), Option<Vec<u8>>> {
        std::mem::take(&mut self.write_buffer)
    }

    /// Whether the write buffer holds no buffered mutations (commit read-only
    /// fast exit).
    pub fn write_buffer_is_empty(&self) -> bool {
        self.write_buffer.is_empty()
    }

    /// The OCC read-set accumulated so far, for commit-time conflict
    /// validation. `None` until the first tracked read (and in legacy mode).
    pub fn occ_scope(&self) -> Option<&OccScope> {
        self.occ_scope.as_ref()
    }

    /// Repin the read timestamp. Transitional: the executor reassigns its read
    /// ts at statement start (e.g. after resolving a causal-read watermark) and
    /// syncs it here so the transaction's snapshot reads agree.
    pub fn set_read_ts(&mut self, read_ts: Timestamp) {
        self.read_ts = read_ts;
    }

    /// Set (or clear) the MVCC read snapshot. Transitional: the executor opens
    /// the snapshot after building the context and syncs it here.
    pub fn set_snapshot(&mut self, snapshot: Option<StorageSnapshot>) {
        self.snapshot = snapshot;
    }

    /// The adjacency time-travel snapshot, if any. Adjacency base reads go
    /// through this so post-snapshot merge operands stay invisible.
    pub fn adj_snapshot(&self) -> Option<StorageSnapshot> {
        self.adj_snapshot
    }

    /// Set (or clear) the adjacency time-travel snapshot. Taken at statement
    /// start or overridden by `AS OF TIMESTAMP`.
    pub fn set_adj_snapshot(&mut self, snapshot: Option<StorageSnapshot>) {
        self.adj_snapshot = snapshot;
    }

    /// Set (or clear) the timestamp oracle. Transitional: some callers flip a
    /// context from legacy into MVCC mode after construction (tests, and the
    /// executor before a write statement); syncing the oracle keeps the
    /// buffer-vs-engine decision and OCC-scope creation in agreement. `None`
    /// is legacy mode.
    pub fn set_oracle(&mut self, oracle: Option<&'a TimestampOracle>) {
        self.oracle = oracle;
    }

    /// Buffer an adjacency add (commutative merge operand). Not OCC-tracked.
    pub fn merge_adj_add(&mut self, adj_key: &[u8], uid: u64) {
        self.merge_adj_adds
            .entry(adj_key.to_vec())
            .or_default()
            .push(uid);
    }

    /// Buffer an adjacency remove (commutative merge operand). Not OCC-tracked.
    pub fn merge_adj_remove(&mut self, adj_key: &[u8], uid: u64) {
        self.merge_adj_removes
            .entry(adj_key.to_vec())
            .or_default()
            .push(uid);
    }

    /// Buffer a node merge operand (pre-encoded document delta) at `node_key`.
    pub fn push_node_delta(&mut self, node_key: Vec<u8>, operand: Vec<u8>) {
        self.merge_node_deltas.push((node_key, operand));
    }

    /// Whether any merge operand (adjacency or node delta) is buffered. Used by
    /// the commit path's read-only fast exit.
    pub fn has_pending_merges(&self) -> bool {
        !self.merge_adj_adds.is_empty()
            || !self.merge_adj_removes.is_empty()
            || !self.merge_node_deltas.is_empty()
    }

    /// Borrow buffered adjacency adds (read-your-own-writes overlay on the
    /// posting-list read path; commit-path local-apply iteration).
    pub fn merge_adj_adds(&self) -> &HashMap<Vec<u8>, Vec<u64>> {
        &self.merge_adj_adds
    }

    /// Borrow buffered adjacency removes (see [`Self::merge_adj_adds`]).
    pub fn merge_adj_removes(&self) -> &HashMap<Vec<u8>, Vec<u64>> {
        &self.merge_adj_removes
    }

    /// Borrow buffered node merge operands (materialisation read + has-pending
    /// checks).
    pub fn node_deltas(&self) -> &[(Vec<u8>, Vec<u8>)] {
        &self.merge_node_deltas
    }

    /// Mutable access to buffered node merge operands (materialisation removes
    /// the deltas it has folded).
    pub fn node_deltas_mut(&mut self) -> &mut Vec<(Vec<u8>, Vec<u8>)> {
        &mut self.merge_node_deltas
    }

    /// Take the buffered adjacency adds, leaving the buffer empty.
    pub fn take_merge_adj_adds(&mut self) -> HashMap<Vec<u8>, Vec<u64>> {
        std::mem::take(&mut self.merge_adj_adds)
    }

    /// Take the buffered adjacency removes, leaving the buffer empty.
    pub fn take_merge_adj_removes(&mut self) -> HashMap<Vec<u8>, Vec<u64>> {
        std::mem::take(&mut self.merge_adj_removes)
    }

    /// Take the buffered node merge operands, leaving the buffer empty.
    pub fn take_node_deltas(&mut self) -> Vec<(Vec<u8>, Vec<u8>)> {
        std::mem::take(&mut self.merge_node_deltas)
    }

    /// Clear all buffered merge operands (commit path's volatile no-drain
    /// branch: local writes already applied, nothing to replicate).
    pub fn clear_merges(&mut self) {
        self.merge_adj_adds.clear();
        self.merge_adj_removes.clear();
        self.merge_node_deltas.clear();
    }

    /// Drop any buffered adjacency adds/removes for `adj_key`. Used when a
    /// node-delete cascade tombstones the posting list — pending merge
    /// operands must not resurrect it.
    pub fn drop_adj_merges(&mut self, adj_key: &[u8]) {
        self.merge_adj_adds.remove(adj_key);
        self.merge_adj_removes.remove(adj_key);
    }

    /// Base adjacency point read: reads `Partition::Adj` at the adjacency
    /// snapshot (AS-OF) if set, else latest. Bypasses the write buffer and OCC
    /// tracking — adjacency is commutative, so the edge layer overlays buffered
    /// merge operands (and any buffered point tombstone) on top of this itself.
    pub fn adj_base_get(&self, key: &[u8]) -> StorageResult<Option<Vec<u8>>> {
        match self.adj_snapshot {
            Some(snap) => Ok(self
                .engine
                .snapshot_get(&snap, Partition::Adj, key)?
                .map(|b| b.to_vec())),
            None => Ok(self.engine.get(Partition::Adj, key)?.map(|b| b.to_vec())),
        }
    }

    /// Base adjacency prefix scan, snapshot-aware (see [`Self::adj_base_get`]).
    /// Returns owned `(key, value)` pairs in key order, no buffer overlay.
    pub fn adj_base_prefix_scan(&self, prefix: &[u8]) -> StorageResult<Vec<KvPair>> {
        match self.adj_snapshot {
            Some(snap) => Ok(self
                .engine
                .snapshot_prefix_scan(&snap, Partition::Adj, prefix)?
                .into_iter()
                .map(|(k, v)| (k, v.to_vec()))
                .collect()),
            None => {
                let iter = self.engine.prefix_scan(Partition::Adj, prefix)?;
                let mut results: Vec<KvPair> = Vec::new();
                for guard in iter {
                    let (k, v) = guard.into_inner()?;
                    results.push((k.to_vec(), v.to_vec()));
                }
                Ok(results)
            }
        }
    }

    /// Flush the transaction: assign a commit timestamp, run OCC conflict
    /// detection against the read-set, and apply all buffered point writes +
    /// commutative merge operands under the effective write concern.
    ///
    /// This is the single Layer-3 commit locus (ADR-041): OCC validation,
    /// `commit_ts` assignment, write-concern fan-out, and the Raft proposal
    /// pipeline all live here. Adjacency (`adj:`) keys bypass conflict checking
    /// because posting-list operations are commutative merge operands.
    ///
    /// Returns the commit timestamp used (or `None` for a read-only / legacy
    /// transaction) plus the committed Raft index on the pipeline path.
    pub fn commit(&mut self, ctx: &CommitContext<'_>) -> Result<CommitOutcome, CommitError> {
        // Flush adj merge buffers even in legacy (no MVCC) mode.
        // Legacy puts write directly to engine, but merge adds are buffered.
        if self.oracle.is_none() {
            for (key, uids) in self.merge_adj_adds.drain() {
                self.engine
                    .merge(Partition::Adj, &key, &encode_add_batch(&uids))?;
            }
            for (key, uids) in self.merge_adj_removes.drain() {
                for uid in uids {
                    self.engine
                        .merge(Partition::Adj, &key, &encode_remove(uid))?;
                }
            }
            for (key, operand) in self.merge_node_deltas.drain(..) {
                self.engine.merge(Partition::Node, &key, &operand)?;
            }
            // Legacy mode — writes already applied.
            return Ok(CommitOutcome {
                commit_ts: None,
                applied_index: None,
            });
        }
        // SAFETY: checked is_none() above and returned early.
        let oracle = match self.oracle {
            Some(o) => o,
            None => {
                return Ok(CommitOutcome {
                    commit_ts: None,
                    applied_index: None,
                })
            }
        };

        let has_merge_ops = !self.merge_adj_adds.is_empty()
            || !self.merge_adj_removes.is_empty()
            || !self.merge_node_deltas.is_empty();
        if self.write_buffer.is_empty() && !has_merge_ops {
            // Read-only — no commit needed.
            return Ok(CommitOutcome {
                commit_ts: Some(self.read_ts),
                applied_index: None,
            });
        }

        let commit_ts = oracle.next();

        // OCC conflict detection (ADR-016: native seqno-based). The coordinator
        // walks the scope's tracked keys, skips commutative partitions, and
        // returns the first conflicting key. Detects all writes including ABA
        // (write + revert to same value) via lsm-tree seqno inspection.
        if let Some(scope) = self.occ_scope.as_ref() {
            if let Some(conflict) = self.engine.coordinator().validate_occ(scope)? {
                return Err(CommitError::Conflict(format!(
                    "OCC conflict: key in {:?} partition was modified by another \
                     transaction after start_ts={}. Retry the transaction.",
                    conflict.partition, conflict.read_ts,
                )));
            }
        }

        // Drain the buffered point writes; each write-concern branch below
        // applies / replicates them.
        let mut wb = std::mem::take(&mut self.write_buffer);

        // Resolve effective write concern (j:true upgrades W0 → W1).
        let effective_level = ctx.write_concern.effective_level();

        // Write concern W0 (fire-and-forget): apply directly to local storage
        // without going through the proposal pipeline. No durability guarantee.
        // Data visible locally but NOT replicated. Lost on crash.
        //
        // ADR-016: writes use plain engine.put()/delete() — no versioned key
        // encoding. LSM seqno from OracleSeqnoGenerator provides native MVCC.
        if effective_level == WriteConcernLevel::W0 {
            for ((part, key), value) in wb.drain() {
                match value {
                    Some(v) => self.engine.put(part, &key, &v)?,
                    None => self.engine.delete(part, &key)?,
                }
            }
            // Apply adj merge operands directly to StorageEngine (raw keys).
            for (key, uids) in self.merge_adj_adds.drain() {
                self.engine
                    .merge(Partition::Adj, &key, &encode_add_batch(&uids))?;
            }
            for (key, uids) in self.merge_adj_removes.drain() {
                for uid in uids {
                    self.engine
                        .merge(Partition::Adj, &key, &encode_remove(uid))?;
                }
            }
            for (key, operand) in self.merge_node_deltas.drain(..) {
                self.engine.merge(Partition::Node, &key, &operand)?;
            }
            return Ok(CommitOutcome {
                commit_ts: Some(commit_ts),
                applied_index: None,
            });
        }

        // Write concern Memory/Cache (volatile with drain):
        // 1. Apply locally for immediate read visibility (same as W0)
        // 2. Buffer mutations in DrainBuffer for background Raft replication
        // 3. Return immediately — drain thread handles durability
        //
        // Crash before drain = data lost (explicit contract).
        // Drained entries preserve original commit_ts for CDC fidelity.
        if effective_level.is_volatile() {
            // Step 1: Apply locally for read visibility.
            for ((part, key), value) in &wb {
                match value {
                    Some(v) => self.engine.put(*part, key, v)?,
                    None => self.engine.delete(*part, key)?,
                }
            }
            for (key, uids) in &self.merge_adj_adds {
                self.engine
                    .merge(Partition::Adj, key, &encode_add_batch(uids))?;
            }
            for (key, uids) in &self.merge_adj_removes {
                for uid in uids {
                    self.engine
                        .merge(Partition::Adj, key, &encode_remove(*uid))?;
                }
            }
            for (key, operand) in &self.merge_node_deltas {
                self.engine.merge(Partition::Node, key, operand)?;
            }

            // Step 2: Buffer for drain (if drain buffer is available).
            if let Some(drain_buf) = ctx.drain_buffer {
                let mut mutations: Vec<Mutation> = wb
                    .drain()
                    .map(|((part, key), value)| match value {
                        Some(v) => Mutation::Put {
                            partition: partition_to_id(part),
                            key,
                            value: v,
                        },
                        None => Mutation::Delete {
                            partition: partition_to_id(part),
                            key,
                        },
                    })
                    .collect();

                for (key, uids) in self.merge_adj_adds.drain() {
                    mutations.push(Mutation::Merge {
                        partition: PartitionId::Adj,
                        key,
                        operand: encode_add_batch(&uids),
                    });
                }
                for (key, uids) in self.merge_adj_removes.drain() {
                    for uid in uids {
                        mutations.push(Mutation::Merge {
                            partition: PartitionId::Adj,
                            key: key.clone(),
                            operand: encode_remove(uid),
                        });
                    }
                }
                for (key, operand) in self.merge_node_deltas.drain(..) {
                    mutations.push(Mutation::Merge {
                        partition: PartitionId::Node,
                        key,
                        operand,
                    });
                }

                let entry = DrainEntry::new(mutations, commit_ts, self.read_ts);

                // w:cache: persist to NVMe before ACK for process-crash recovery.
                // w:memory skips this — data loss on crash is the explicit contract.
                if effective_level == WriteConcernLevel::Cache {
                    if let Some(nvme) = ctx.nvme_write_buffer {
                        nvme.append(&entry).map_err(|e| {
                            CommitError::Serialization(format!("w:cache NVMe write failed: {e}"))
                        })?;
                    }
                }

                drain_buf.append(entry).map_err(|e| {
                    CommitError::Serialization(format!("volatile write backpressure: {e}"))
                })?;
            } else {
                // No drain buffer — clear write buffers (local writes already applied).
                wb.clear();
                self.merge_adj_adds.clear();
                self.merge_adj_removes.clear();
                self.merge_node_deltas.clear();
            }

            return Ok(CommitOutcome {
                commit_ts: Some(commit_ts),
                applied_index: None,
            });
        }

        // W1 / Majority: apply through proposal pipeline (or direct write).
        //
        // When a pipeline is configured, mutations are packaged into a
        // RaftProposal and sent through the pipeline for durable application.
        // In single-node mode (W1 and Majority are equivalent), the pipeline
        // applies directly to CoordiNode storage. In cluster mode, Majority
        // replicates via Raft first while W1 returns after leader WAL fsync.
        //
        // When no pipeline is configured (legacy/test mode), mutations are
        // written directly to the engine.
        let mut applied_index: Option<u64> = None;
        if let (Some(pipeline), Some(id_gen)) = (ctx.pipeline, ctx.id_gen) {
            let mut mutations: Vec<Mutation> = wb
                .drain()
                .map(|((part, key), value)| match value {
                    Some(v) => Mutation::Put {
                        partition: partition_to_id(part),
                        key,
                        value: v,
                    },
                    None => Mutation::Delete {
                        partition: partition_to_id(part),
                        key,
                    },
                })
                .collect();

            // Adj merge operands: bypass MVCC, raw keys.
            for (key, uids) in self.merge_adj_adds.drain() {
                mutations.push(Mutation::Merge {
                    partition: PartitionId::Adj,
                    key,
                    operand: encode_add_batch(&uids),
                });
            }
            for (key, uids) in self.merge_adj_removes.drain() {
                for uid in uids {
                    mutations.push(Mutation::Merge {
                        partition: PartitionId::Adj,
                        key: key.clone(),
                        operand: encode_remove(uid),
                    });
                }
            }
            for (key, operand) in self.merge_node_deltas.drain(..) {
                mutations.push(Mutation::Merge {
                    partition: PartitionId::Node,
                    key,
                    operand,
                });
            }

            // Coalesce dense runs of point deletes into range deletes before
            // proposing (G096): a bulk delete ("delete all relationships between
            // these nodes", DROP) replicates + PITR-logs as a few range ops
            // instead of N point tombstones. Non-deletes / short runs untouched.
            let mutations = coordinode_core::txn::coalesce::coalesce_delete_mutations(
                mutations,
                coordinode_core::txn::coalesce::DEFAULT_MIN_RUN,
            );

            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations,
                commit_ts,
                start_ts: self.read_ts,
                bypass_rate_limiter: false,
            };

            // Apply write concern timeout if configured (wtimeout > 0).
            //
            // propose_with_timeout uses true async timeout in cluster mode
            // (RaftProposalPipeline wraps propose_async with tokio::time::timeout).
            // In embedded/single-node mode, the default impl delegates to
            // propose_and_wait (proposals complete in µs, timeout irrelevant).
            //
            // Per MongoDB spec: "On timeout, data is NOT rolled back."
            // The proposal may still commit after timeout fires.
            let timeout_ms = ctx.write_concern.timeout_ms;
            let outcome = if timeout_ms > 0 {
                let timeout = std::time::Duration::from_millis(u64::from(timeout_ms));
                pipeline
                    .propose_with_timeout(&proposal, timeout)
                    .map_err(proposal_err_to_commit)?
            } else {
                pipeline
                    .propose_and_wait(&proposal)
                    .map_err(proposal_err_to_commit)?
            };
            // Record the committed Raft index of this write so the gRPC layer
            // can return it as the causal operationTime token. `None` in
            // local/embedded mode (no Raft log).
            applied_index = outcome.applied_index;
        } else {
            // Legacy direct-write path (no pipeline configured).
            // ADR-016: plain engine.put()/delete() — oracle auto-stamps seqno.
            for ((part, key), value) in wb.drain() {
                match value {
                    Some(v) => self.engine.put(part, &key, &v)?,
                    None => self.engine.delete(part, &key)?,
                }
            }
            // Apply adj merge operands directly to StorageEngine (raw keys).
            for (key, uids) in self.merge_adj_adds.drain() {
                self.engine
                    .merge(Partition::Adj, &key, &encode_add_batch(&uids))?;
            }
            for (key, uids) in self.merge_adj_removes.drain() {
                for uid in uids {
                    self.engine
                        .merge(Partition::Adj, &key, &encode_remove(uid))?;
                }
            }
            for (key, operand) in self.merge_node_deltas.drain(..) {
                self.engine.merge(Partition::Node, &key, &operand)?;
            }
        }

        // Journal gate (j:true): force WAL fsync after commit.
        // With FlushPolicy::SyncPerBatch this is already done by WriteBatch,
        // but with Periodic/Manual policies, j:true forces an explicit persist.
        if ctx.write_concern.journal {
            self.engine
                .persist()
                .map_err(|e| CommitError::Serialization(format!("journal fsync failed: {e}")))?;
        }

        Ok(CommitOutcome {
            commit_ts: Some(commit_ts),
            applied_index,
        })
    }

    /// MVCC-aware prefix scan: snapshot results overlaid with buffered writes
    /// (a buffered value replaces the storage row for that key). Snapshot keys
    /// other than our own buffered writes are OCC-tracked. Legacy mode scans
    /// the engine directly and overlays the (empty-in-legacy) buffer.
    ///
    /// Note: buffered tombstones (in-transaction deletes) do NOT hide a storage
    /// row from the scan — this matches the established executor semantics
    /// (read-your-own-writes covers point reads; scans surface the snapshot row
    /// for keys without a buffered *value*). Preserved deliberately for
    /// behavioural parity.
    pub fn prefix_scan(&mut self, part: Partition, prefix: &[u8]) -> StorageResult<Vec<KvPair>> {
        // Own writes (buffered values, NOT tombstones) that match the prefix.
        let buffer_matches: Vec<KvPair> = self
            .write_buffer
            .iter()
            .filter_map(|((p, k), v)| {
                if *p == part && k.starts_with(prefix) {
                    v.as_ref().map(|val| (k.clone(), val.clone()))
                } else {
                    None
                }
            })
            .collect();

        match self.snapshot {
            Some(snap) => {
                let mut results: Vec<KvPair> = self
                    .engine
                    .snapshot_prefix_scan(&snap, part, prefix)?
                    .into_iter()
                    .map(|(k, v)| (k, v.to_vec()))
                    .collect();
                // OCC-track every scanned storage key except our own writes.
                let buffer_keys: HashSet<Vec<u8>> =
                    buffer_matches.iter().map(|(k, _)| k.clone()).collect();
                if let Some(scope) = self.ensure_occ_scope() {
                    for (k, _) in &results {
                        if !buffer_keys.contains(k) {
                            scope.track(part, k);
                        }
                    }
                }
                // Buffer takes priority: drop storage rows shadowed by a
                // buffered value, then append the buffered values.
                results.retain(|(k, _)| !buffer_keys.contains(k));
                results.extend(buffer_matches);
                Ok(results)
            }
            None => {
                // Legacy mode: writes apply straight to the engine, so the
                // buffer is empty and `buffer_matches` overlays nothing.
                let iter = self.engine.prefix_scan(part, prefix)?;
                let mut results: Vec<KvPair> = Vec::new();
                for guard in iter {
                    let (k, v) = guard.into_inner()?;
                    results.push((k.to_vec(), v.to_vec()));
                }
                results.extend(buffer_matches);
                Ok(results)
            }
        }
    }

    /// Keyset-resumed page of a prefix scan, reading the transaction's pinned
    /// snapshot. Returns up to `limit` rows whose key carries `prefix`, starting
    /// strictly after `start_after` (or at the prefix start when `None`), plus
    /// the last key returned (the next page's resume point) and whether the
    /// prefix is exhausted. Each returned key is OCC-tracked.
    ///
    /// This is the cursor's memory-bounded source: it holds at most one page
    /// regardless of the prefix's total size. It reads the committed snapshot
    /// only and does NOT overlay this transaction's own buffered writes — the
    /// keyset cursor falls back to a materialised scan when the transaction has
    /// buffered writes over the prefix.
    pub fn prefix_scan_paged(
        &mut self,
        part: Partition,
        prefix: &[u8],
        start_after: Option<&[u8]>,
        limit: usize,
    ) -> StorageResult<PagedScan> {
        let start = match start_after {
            Some(after) => {
                // The smallest key strictly greater than `after`.
                let mut s = after.to_vec();
                s.push(0);
                s
            }
            None => prefix.to_vec(),
        };
        let end = prefix_upper_bound(prefix);
        let iter = self.base_range_seekable(part, &start, &end)?;

        let mut rows: Vec<KvPair> = Vec::with_capacity(limit);
        let mut exhausted = true;
        for guard in iter {
            let (key, value) = guard.into_inner()?;
            if !key.starts_with(prefix) {
                continue;
            }
            if rows.len() == limit {
                // At least one more matching row exists beyond this page.
                exhausted = false;
                break;
            }
            rows.push((key.to_vec(), value.to_vec()));
        }

        // OCC-track every key this page observed.
        if let Some(scope) = self.ensure_occ_scope() {
            for (key, _) in &rows {
                scope.track(part, key);
            }
        }

        let last_key = rows.last().map(|(k, _)| k.clone());
        Ok(PagedScan {
            rows,
            last_key,
            exhausted,
        })
    }
}

/// One page of a keyset-resumed prefix scan (see [`Transaction::prefix_scan_paged`]).
pub struct PagedScan {
    /// The page's rows, in key order.
    pub rows: Vec<KvPair>,
    /// Storage key of the last row returned: the resume point for the next page.
    /// `None` when the page is empty.
    pub last_key: Option<Vec<u8>>,
    /// True when the prefix has no rows beyond this page.
    pub exhausted: bool,
}

/// The smallest key that sorts strictly after every key carrying `prefix`, for
/// use as an inclusive range upper bound (callers still filter with
/// `starts_with`, since the bound itself may be a real, non-matching key).
fn prefix_upper_bound(prefix: &[u8]) -> Vec<u8> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.pop() {
        if last < 0xFF {
            end.push(last + 1);
            return end;
        }
    }
    // Empty or all-0xFF prefix: extend with 0xFF to cover the rest of the
    // keyspace (node-scan prefixes never reach this branch).
    let mut end = prefix.to_vec();
    end.push(0xFF);
    end
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
