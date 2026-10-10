//! Layer-3 transaction context.
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
//! and the commutative merge buffers (adjacency adds/removes and node document
//! deltas). Conflict detection is write-set first-committer-wins at commit;
//! reads are not tracked at the default level (the OCC scope exists for the
//! opt-in serializable level and FOR UPDATE, which pin keys selectively).
//! Commit orchestration (validation + commit-ts assignment +
//! write-concern-aware flush) still lives in the query engine, which drains
//! these buffers via the `take_*` accessors.

use std::collections::HashMap;

use coordinode_core::graph::edge::PostingList;
use coordinode_core::txn::drain::{DrainBuffer, DrainEntry};
use coordinode_core::txn::invariant::{Claim, ClaimSet};
use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalError, ProposalIdGenerator, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::{Journal, WriteConcern};
use lsm_tree::Guard;

use crate::cache::write_buffer::NvmeWriteBuffer;
use crate::engine::StorageSnapshot;
use crate::engine::coordinator::{MultiModalCoordinator, OccScope, SnapshotPin};
use crate::engine::core::StorageEngine;
use crate::engine::merge::{encode_add_batch, encode_counter_delta, encode_remove};
use crate::engine::open_txns::OpenTransaction;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// A key/value pair returned by [`Transaction::prefix_scan`].
pub type KvPair = (Vec<u8>, Vec<u8>);

/// Per-statement commit policy passed to [`Transaction::commit`]. These are
/// execution-context concerns (durability level, replication wiring), not part
/// of the transaction's identity, so they are supplied at commit time rather
/// than stored on the transaction.
pub struct CommitContext<'b> {
    /// Write concern: how many members hold the write (`w`) and in what state
    /// (`journal`) before the caller is answered.
    pub write_concern: &'b WriteConcern,
    /// Raft proposal pipeline for durable application. `None` in
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
#[derive(Debug)]
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
    /// Write-write conflict: a key this transaction WROTE was also written by
    /// a transaction that committed after `read_ts` (first-committer-wins).
    /// Nothing was applied; re-running the whole transaction is the intended
    /// response, and the only commit failure for which that helps.
    #[error("{0}")]
    Conflict(String),
    /// Underlying storage failure during flush.
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// Encoding / replication-pipeline failure.
    #[error("{0}")]
    Serialization(String),
    /// The engine is over its compaction-debt stop threshold: new client
    /// writes are rejected until compaction catches up. Nothing was applied;
    /// retry after a short delay.
    #[error(
        "write rejected: storage is over its compaction-debt stop threshold; \
         retry after compaction catches up"
    )]
    Backpressure,
    /// This node is not the leader, so it cannot replicate the write. Nothing
    /// was applied. `leader_id` is the node the cluster last named leader, when
    /// one is known: an election in flight leaves it `None` for a moment.
    ///
    /// Structured rather than folded into a message because the answer to it
    /// is mechanical: retry the same write at the leader. Every layer above
    /// needs the id to do that, and a string cannot carry it.
    #[error("not the leader{}", match leader_id {
        Some(id) => format!("; leader is node {id}"),
        None => String::from("; no leader known yet"),
    })]
    NotLeader {
        /// The node the cluster last named leader, if any.
        leader_id: Option<u64>,
    },
    /// This member does not run the version its group runs, so it takes no
    /// writes. Nothing was applied. Kept structured so every layer above can
    /// name both versions and the leader to retry at.
    #[error("this member is read-only: {0}")]
    Mismatched(coordinode_core::version::Mismatch),
    /// The deltas staged for one counter sum to a value outside `i64`.
    /// Nothing was applied. A counter operand carries no base, so a sum that
    /// cannot exist is only discoverable where it is assembled; refusing it
    /// here keeps it out of a compaction, which has no caller to answer and
    /// would stop making progress on that partition instead.
    #[error("counter '{key}' would leave the i64 range; nothing was written")]
    CounterOverflow {
        /// The counter key whose staged deltas overflowed.
        key: String,
    },
    /// The DERIVED index work this attempt seals derives more entry effects
    /// than a member derives for one unit. Nothing was applied: every member
    /// would refuse the unit when it derives it.
    #[error(
        "the transaction changes more than {limit} derived index entries; \
         nothing was written"
    )]
    IndexFanOut {
        /// The bound on entry effects per unit.
        limit: usize,
    },
    /// A write the transaction staged did not fit the budget of the
    /// statement that staged it, so it and everything after it was not
    /// staged. Nothing was applied.
    #[error("{0}; nothing was written")]
    Budget(coordinode_core::budget::BudgetStop),
    /// A record this attempt wrote on the condition of its version has a
    /// different one. Nothing was applied.
    ///
    /// The version that is there now is carried here so the caller can
    /// decide what to do without reading the record again, and so that a
    /// retry starts from a fact rather than from another race.
    #[error(
        "record version mismatch: expected {expected:?}, found {current:?}; \
         nothing was applied"
    )]
    RevisionMismatch {
        /// The version the caller wrote against. `None` means it required
        /// the record to be absent.
        expected: Option<u64>,
        /// The version the record has now. `None` means it is absent.
        current: Option<u64>,
    },
    /// A condition this attempt's result depends on no longer holds, or
    /// another attempt in flight holds an incompatible one. Nothing was
    /// applied.
    ///
    /// Separate from `Conflict`, which is the write set losing first-committer
    /// -wins: two attempts can write disjoint keys and still break a graph
    /// condition together, and telling a caller "write conflict" for that
    /// would name the wrong cause.
    #[error("invariant refused the commit: {reason}")]
    InvariantRefused {
        /// Which condition refused it, in the words of the condition.
        reason: String,
    },
    /// A value this attempt gives a node in a unique index is held by
    /// another node in the post-state. Nothing was applied.
    #[error("a unique value of index generation {generation} is held by node {}", holder.as_raw())]
    UniqueValueHeld {
        /// The index generation.
        generation: coordinode_core::index::identity::GenerationId,
        /// The values the holder has in the index.
        values: Vec<coordinode_core::graph::types::Value>,
        /// The node holding it.
        holder: coordinode_core::graph::node::NodeId,
    },
    /// Proving a value this attempt takes in a unique index still being
    /// built free would read more than `limit` stored rows. Nothing was
    /// applied; the value may be free, and the attempt can be retried once
    /// the build is done.
    #[error(
        "proving a unique value of index generation {generation} free would read more than \
         {limit} stored nodes; retry once the build is done"
    )]
    UniquenessUnresolved {
        /// The index generation.
        generation: coordinode_core::index::identity::GenerationId,
        /// The most stored rows one proof reads.
        limit: u64,
    },
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
    match err {
        ProposalError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        } => CommitError::Storage(StorageError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        }),
        // Kept structured all the way up: the caller's response is to retry at
        // the leader, and it needs the id to do that.
        ProposalError::NotLeader { leader_id } => CommitError::NotLeader { leader_id },
        ProposalError::Mismatched(m) => CommitError::Mismatched(m),
        // Nothing was proposed, and a new attempt takes a new timestamp: the
        // answer is the one to a conflict.
        e @ (ProposalError::BelowClosedBound { .. } | ProposalError::LeaderNotReady { .. }) => {
            CommitError::Conflict(e.to_string())
        }
        ProposalError::OutOfSpace {
            path,
            available_bytes,
            min_free_bytes,
        } => CommitError::Storage(StorageError::OutOfSpace {
            path,
            available_bytes,
            min_free_bytes,
        }),
        other => CommitError::Serialization(format!("proposal pipeline error: {other}")),
    }
}

/// Per-statement transaction context. See the module docs.
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
    /// Buffered adjacency-posting merge operands, in the order they were
    /// staged. Adjacency writes are merge operands (not point writes), so they
    /// live in a separate buffer from `write_buffer` and bypass OCC conflict
    /// detection. Drained at commit.
    ///
    /// The order is the buffer's whole job. An add and a remove of one member
    /// do not commute, and both carry the same commit timestamp once the
    /// transaction commits, so nothing downstream can recover an order this
    /// buffer did not keep: whatever order reaches the merge operator is the
    /// only order that key will ever have. A transaction that removes an edge
    /// and writes it again means it to be there at the end.
    merge_adj_ops: Vec<(Vec<u8>, AdjOp)>,
    /// Buffered node merge operands: `(node key, operand bytes)`. Read-modify
    /// -write document deltas (SET nested path) materialise these against the
    /// node record before a read; drained at commit.
    merge_node_deltas: Vec<(Vec<u8>, Vec<u8>)>,
    /// Buffered counter merge deltas: `counter key -> summed signed delta`.
    /// Commutative increments on the `counter:` partition (statistics
    /// counters, degree caches); bypass OCC like the other merge buffers.
    /// COALESCED per key: a bulk statement touching one counter N times
    /// stages one entry (and one commit-time operand), not N — the buffer
    /// and the replication proposal stay O(distinct counters), not O(ops).
    /// Drained at commit as `CounterMerge` operands; a sum that folded to
    /// zero is skipped at drain (nothing to apply).
    merge_counter_deltas: HashMap<Vec<u8>, i64>,
    /// The first counter key whose staged deltas left `i64`, if any. Held so
    /// that commit refuses instead of writing an operand no fold can apply.
    counter_overflow: Option<Vec<u8>>,
    /// The budget of the statement staging writes now, which each staged
    /// write is charged to before it is buffered; `None` charges nothing.
    write_budget: Option<std::sync::Arc<coordinode_core::budget::QueryBudget>>,
    /// The refusal that stopped staging: nothing is staged after it, and
    /// commit refuses.
    budget_refusal: Option<coordinode_core::budget::BudgetStop>,
    /// What this attempt's result depends on, stated by the writers as they
    /// go. Checked and reserved at commit: the write set alone cannot tell
    /// two attempts apart that each validated a condition the other breaks.
    claims: ClaimSet,
    /// The label schemas this attempt read, as the claims they become if it
    /// writes. Held apart from `claims` because a read alone depends on
    /// nothing a schema change can break: an attempt that only read commits
    /// whatever the schema did meanwhile.
    schema_reads: ClaimSet,
    /// Records this attempt will only write if their version is still what it
    /// read. `None` as the expected version means the record must not exist.
    ///
    /// Held apart from the write buffer because the condition is not a write:
    /// it is what makes the write admissible, and the two are checked at
    /// different points against different evidence.
    expected_versions: Vec<(Partition, Vec<u8>, Option<u64>)>,
    /// The schema generation as it stood when this attempt began. Every claim
    /// it makes is stamped with it, so a predicate evaluated here is not taken
    /// as evidence about a graph whose definitions have since changed.
    schema_generation: u64,
    /// Whether this attempt changes a schema definition. Set by the writer
    /// that stages the change, acted on once the commit lands.
    schema_changed: bool,
    /// GC-watermark pin at `snapshot`, held for the transaction's life (and
    /// parked with its state between interactive statements) so compaction
    /// never collects the history this transaction reads. `None` in legacy
    /// mode, or when the snapshot was already below the watermark when set
    /// (reads then fail with `SnapshotOutsideRetention` instead of guessing).
    snapshot_pin: Option<SnapshotPin>,
    /// The first sequence number whose writes this transaction may not have
    /// seen: its snapshot, or lower when the snapshot covered commits still
    /// in flight at the moment it was set. Those commits land below the
    /// snapshot but after the reads, so validating from the snapshot alone
    /// would forgive exactly the writes this transaction missed.
    validate_from: Option<StorageSnapshot>,
    /// This transaction's entry among the engine's open transactions, from
    /// creation until its last state drops. `None` only once
    /// [`Self::take_state`] has moved it to the parked state.
    open: Option<OpenTransaction>,
    /// DERIVED index work: entries staged in the write buffer that the unit
    /// logs as sealed work rather than as mutations.
    derived: derived::DerivedLedger,
    /// Key ranges this attempt removes, `(partition, start, end)`, applied in
    /// the commit's unit ahead of its point writes ([`Self::remove_range`]).
    range_removals: Vec<(Partition, Vec<u8>, Vec<u8>)>,
    /// Node keys whose state as this attempt leaves them its writer checks
    /// before the commit ([`Self::note_post_state_check`]).
    post_state_checks: Vec<Vec<u8>>,
    /// Commits even while storage sheds writes under pressure
    /// ([`Self::exempt_from_write_pressure`]).
    pressure_exempt: bool,
    /// How long admission waits for the commits in flight on its keys
    /// instead of refusing ([`Self::wait_for_overlapping_commits`]).
    overlap_wait: Option<std::time::Duration>,
    /// The changes to kept cardinality counts this commit derived from its
    /// staged edge effects. Derived anew by each commit attempt rather than
    /// added to [`Self::merge_counter_deltas`], so an attempt that fails
    /// after deriving leaves nothing a later one would count twice.
    kept_counts: rustc_hash::FxHashMap<Vec<u8>, i64>,
}

/// The borrow-free owned state of a [`Transaction`] — everything except the
/// `engine` / `oracle` borrows. An interactive multi-statement transaction
/// parks this in a leader-local registry between statements and
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
    merge_adj_ops: Vec<(Vec<u8>, AdjOp)>,
    merge_node_deltas: Vec<(Vec<u8>, Vec<u8>)>,
    merge_counter_deltas: HashMap<Vec<u8>, i64>,
    /// The attempt's stated conditions. Parked with the rest of its state,
    /// because an interactive transaction is one attempt across statements
    /// and a condition stated by the first still binds the last.
    claims: ClaimSet,
    /// Parked with the claims: a schema read by the first statement governs
    /// a write staged by a later one.
    schema_reads: ClaimSet,
    /// Parked for the same reason as the claims: a condition stated by one
    /// statement of an interactive transaction binds the commit of the last.
    expected_versions: Vec<(Partition, Vec<u8>, Option<u64>)>,
    /// Parked with the claims it stamps: the generation belongs to the
    /// attempt, and the attempt spans the statements.
    schema_generation: u64,
    schema_changed: bool,
    snapshot_pin: Option<SnapshotPin>,
    validate_from: Option<StorageSnapshot>,
    /// Parked with the rest: the transaction stays open between statements.
    open: Option<OpenTransaction>,
    /// Parked with the write buffer whose index entries it describes.
    derived: derived::DerivedLedger,
    /// Parked with the write buffer: the removals commit with its writes.
    range_removals: Vec<(Partition, Vec<u8>, Vec<u8>)>,
    /// Parked: a node one statement wrote is checked as the last one leaves
    /// it.
    post_state_checks: Vec<Vec<u8>>,
    /// Parked: a statement whose writes were cut short by its budget leaves
    /// the attempt unable to commit, whatever statement asks.
    budget_refusal: Option<coordinode_core::budget::BudgetStop>,
}

/// Bytes the write buffer holds for one staged key beside the key and value
/// themselves.
const STAGED_WRITE: usize =
    core::mem::size_of::<((Partition, Vec<u8>), Option<Vec<u8>>)>() + core::mem::size_of::<u64>();
/// Bytes one staged adjacency operand holds beside its key.
const STAGED_ADJ: usize = core::mem::size_of::<(Vec<u8>, AdjOp)>();
/// Bytes one staged node delta holds beside its key and operand.
const STAGED_DELTA: usize = core::mem::size_of::<(Vec<u8>, Vec<u8>)>();
/// Bytes one staged counter holds beside its key.
const STAGED_COUNTER: usize = core::mem::size_of::<(Vec<u8>, i64)>() + core::mem::size_of::<u64>();

/// One staged adjacency operand. Kept as a sequence rather than as two sets
/// because an add and a remove of the same member do not commute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdjOp {
    /// Add a member to the posting list.
    Add(u64),
    /// Remove a member from the posting list.
    Remove(u64),
}

/// Turn the staged operands into encoded merge operands, in order.
///
/// A run of consecutive adds on one key becomes one batch operand, which is
/// what the bulk paths stage and what keeps a large insert to one operand
/// rather than thousands. A remove ends the run it is part of, because putting
/// it anywhere else would change what the chain means.
fn encode_staged_adj(ops: &[(Vec<u8>, AdjOp)]) -> Vec<(&[u8], Vec<u8>)> {
    let mut out: Vec<(&[u8], Vec<u8>)> = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        let key = ops[i].0.as_slice();
        match ops[i].1 {
            AdjOp::Add(uid) => {
                let mut uids = vec![uid];
                let mut j = i + 1;
                while let Some((next_key, AdjOp::Add(next))) = ops.get(j).map(|(k, o)| (k, *o)) {
                    if next_key.as_slice() != key {
                        break;
                    }
                    uids.push(next);
                    j += 1;
                }
                out.push((key, encode_add_batch(&uids)));
                i = j;
            }
            AdjOp::Remove(uid) => {
                out.push((key, encode_remove(uid)));
                i += 1;
            }
        }
    }
    out
}

impl TransactionState {
    /// The pinned read timestamp (`start_ts`) of the parked transaction. An
    /// interactive transaction reuses this for every statement so all reads
    /// resolve against the same snapshot (repeatable read).
    pub fn read_ts(&self) -> Timestamp {
        self.read_ts
    }

    /// State a version condition on a transaction that is parked between
    /// statements.
    ///
    /// An interactive transaction spends most of its life here rather than as
    /// a `Transaction`, and a caller that wants to condition its commit has
    /// nowhere else to say so. The condition binds the commit, not the
    /// statement it was stated beside.
    pub fn expect_version(
        &mut self,
        part: Partition,
        key: &[u8],
        expected: Option<u64>,
    ) -> StorageResult<()> {
        if part.is_commutative() {
            return Err(StorageError::InvalidConfig(format!(
                "the {part:?} partition is merge-composed: its rows are folded from \
                 operands, so a write cannot be conditioned on their version"
            )));
        }
        self.expected_versions.push((part, key.to_vec(), expected));
        Ok(())
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
        let adj_ops: usize = self.merge_adj_ops.iter().map(|(k, _)| k.len() + 8).sum();
        let node_deltas: usize = self
            .merge_node_deltas
            .iter()
            .map(|(k, op)| k.len() + op.len())
            .sum();
        writes + adj_ops + node_deltas
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
        let pin = snapshot.and_then(|s| engine.pin_snapshot_at(s));
        Self::with_pin(engine, oracle, read_ts, snapshot, pin)
    }

    /// Open a transaction at the latest complete state, its snapshot pinned
    /// in the same step it is taken. See [`StorageEngine::pin_latest_snapshot`]
    /// for why the two cannot be separate calls.
    pub fn begin(
        engine: &'a StorageEngine,
        oracle: Option<&'a TimestampOracle>,
        read_ts: Timestamp,
    ) -> Self {
        let (snapshot, pin) = engine.pin_latest_snapshot();
        Self::with_pin(engine, oracle, read_ts, Some(snapshot), pin)
    }

    fn with_pin(
        engine: &'a StorageEngine,
        oracle: Option<&'a TimestampOracle>,
        read_ts: Timestamp,
        snapshot: Option<StorageSnapshot>,
        snapshot_pin: Option<SnapshotPin>,
    ) -> Self {
        Self {
            engine,
            oracle,
            read_ts,
            snapshot,
            adj_snapshot: None,
            write_buffer: HashMap::new(),
            occ_scope: None,
            merge_adj_ops: Vec::new(),
            merge_node_deltas: Vec::new(),
            merge_counter_deltas: HashMap::new(),
            counter_overflow: None,
            write_budget: None,
            budget_refusal: None,
            claims: ClaimSet::new(),
            schema_reads: ClaimSet::new(),
            expected_versions: Vec::new(),
            schema_generation: engine.schema_generation(),
            schema_changed: false,
            snapshot_pin,
            validate_from: snapshot.map(|s| Self::first_unseen(engine, s)),
            open: Some(engine.open_transaction()),
            derived: derived::DerivedLedger::default(),
            range_removals: Vec::new(),
            post_state_checks: Vec::new(),
            pressure_exempt: false,
            overlap_wait: None,
            kept_counts: rustc_hash::FxHashMap::default(),
        }
    }

    /// The change this transaction has staged to the counter `key` and not
    /// committed yet: what its own reads of the counter add to the stored
    /// value.
    pub fn pending_counter_delta(&self, key: &[u8]) -> i64 {
        self.merge_counter_deltas.get(key).copied().unwrap_or(0)
    }

    /// Have the commit wait up to `wait` for the commits already admitted on
    /// its keys to land, instead of being refused by them, and reserve its
    /// keys meanwhile so later commits queue behind it.
    ///
    /// For a catalog change: every writer of the catalogued object conditions
    /// its commit on the record the change rewrites, so under steady writes
    /// the change would lose to whichever writer came first, every time.
    /// Writers admitted before it land under the old record; writers arriving
    /// while it waits are refused and retry against the new one.
    pub fn wait_for_overlapping_commits(&mut self, wait: std::time::Duration) {
        self.overlap_wait = Some(wait);
    }

    /// Let this transaction commit while storage sheds writes under pressure.
    ///
    /// For metadata whose change is what relieves the pressure, such as
    /// ending a consumer registration that holds history back: refusing it
    /// would keep the bytes the refusal is meant to free.
    pub fn exempt_from_write_pressure(&mut self) {
        self.pressure_exempt = true;
    }

    /// Charge what this transaction stages from now on to `budget`, the
    /// running statement's: each staged write reserves its bytes, for the
    /// rest of the statement, before it is buffered. `None` charges nothing.
    pub fn charge_writes_to(
        &mut self,
        budget: Option<std::sync::Arc<coordinode_core::budget::QueryBudget>>,
    ) {
        self.write_budget = budget;
    }

    /// The refusal that stopped staging, when a write did not fit the
    /// statement's budget. Nothing was staged after it; commit refuses.
    pub fn budget_refusal(&self) -> Option<coordinode_core::budget::BudgetStop> {
        self.budget_refusal
    }

    /// Reserve `bytes` a write is about to stage. After a refusal every
    /// later write is refused as well, so nothing past it is staged.
    fn admit_staged(&mut self, bytes: usize) -> Result<(), coordinode_core::budget::BudgetStop> {
        if let Some(stop) = self.budget_refusal {
            return Err(stop);
        }
        let Some(budget) = &self.write_budget else {
            return Ok(());
        };
        // The size of memory about to exist fits in u64.
        match budget.reserve(bytes as u64) {
            Ok(charge) => {
                charge.keep_until_query_ends();
                Ok(())
            }
            Err(stop) => {
                self.budget_refusal = Some(stop);
                Err(stop)
            }
        }
    }

    /// The first sequence number a view at `snapshot` may have missed writes
    /// at: the oldest commit still in flight at or below it, else the
    /// snapshot itself. A commit in flight has its number but not its effect,
    /// so a read now cannot see it, and it lands below the snapshot later.
    fn first_unseen(engine: &StorageEngine, snapshot: StorageSnapshot) -> StorageSnapshot {
        engine.pending_commits().snapshot_floor(|| snapshot)
    }

    /// The snapshot this transaction reads at, if it reads at one.
    pub fn snapshot(&self) -> Option<StorageSnapshot> {
        self.snapshot
    }

    /// Whether the history at this transaction's snapshot is protected.
    #[cfg(test)]
    pub(crate) fn snapshot_pinned(&self) -> bool {
        self.snapshot_pin.is_some()
    }

    /// Consume the transaction, returning its borrow-free owned state.
    ///
    /// The `engine` / `oracle` borrows are dropped; everything that defines
    /// the transaction's progress — the pinned `read_ts` + snapshots, the
    /// read-your-own-writes write buffer, the OCC read-set, and the adjacency
    /// / node merge buffers — is moved out. Pair with [`Self::resume`] to park
    /// an interactive multi-statement transaction in a leader-local
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
            merge_adj_ops: self.merge_adj_ops,
            claims: self.claims,
            schema_reads: self.schema_reads,
            expected_versions: self.expected_versions,
            schema_generation: self.schema_generation,
            schema_changed: self.schema_changed,
            merge_node_deltas: self.merge_node_deltas,
            merge_counter_deltas: self.merge_counter_deltas,
            snapshot_pin: self.snapshot_pin,
            validate_from: self.validate_from,
            open: self.open,
            derived: self.derived,
            range_removals: self.range_removals,
            post_state_checks: self.post_state_checks,
            budget_refusal: self.budget_refusal,
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
            merge_adj_ops: std::mem::take(&mut self.merge_adj_ops),
            claims: std::mem::take(&mut self.claims),
            schema_reads: std::mem::take(&mut self.schema_reads),
            expected_versions: std::mem::take(&mut self.expected_versions),
            schema_generation: self.schema_generation,
            schema_changed: std::mem::take(&mut self.schema_changed),
            merge_node_deltas: std::mem::take(&mut self.merge_node_deltas),
            merge_counter_deltas: std::mem::take(&mut self.merge_counter_deltas),
            snapshot_pin: self.snapshot_pin.take(),
            validate_from: self.validate_from,
            open: self.open.take(),
            derived: std::mem::take(&mut self.derived),
            range_removals: std::mem::take(&mut self.range_removals),
            post_state_checks: std::mem::take(&mut self.post_state_checks),
            budget_refusal: self.budget_refusal.take(),
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
            merge_adj_ops: state.merge_adj_ops,
            counter_overflow: None,
            // Charged per statement: the next one names its own budget.
            write_budget: None,
            budget_refusal: state.budget_refusal,
            claims: state.claims,
            schema_reads: state.schema_reads,
            expected_versions: state.expected_versions,
            schema_generation: state.schema_generation,
            schema_changed: state.schema_changed,
            merge_node_deltas: state.merge_node_deltas,
            merge_counter_deltas: state.merge_counter_deltas,
            snapshot_pin: state.snapshot_pin,
            validate_from: state.validate_from,
            open: state.open,
            derived: state.derived,
            range_removals: state.range_removals,
            post_state_checks: state.post_state_checks,
            pressure_exempt: false,
            overlap_wait: None,
            kept_counts: rustc_hash::FxHashMap::default(),
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
        // Reads are NOT tracked for conflict detection: the default level
        // conflicts on writes only (write-set first-committer-wins at commit),
        // so recording every read here would be a per-read allocation into a
        // mutex-guarded set that commit never consults. Read-dependency
        // conflicts are the opt-in serializable level's and FOR UPDATE's job,
        // which re-introduce tracking selectively through `ensure_occ_scope`.
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
            self.admit_staged(STAGED_WRITE + key.len() + value.len())?;
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
            self.admit_staged(STAGED_WRITE + key.len())?;
            self.write_buffer.insert((part, key.to_vec()), None);
            Ok(())
        } else {
            self.engine.delete(part, key)
        }
    }

    /// Remove every key in `[start, end)` of `part` with this attempt's
    /// commit, in one unit with its other writes. Legacy mode removes
    /// straight from the engine.
    ///
    /// For DDL that drops a whole key family together with the catalog
    /// record that owns it. The attempt neither reads nor writes inside a
    /// range it removes: its reads would still see the range as stored.
    pub fn remove_range(&mut self, part: Partition, start: &[u8], end: &[u8]) -> StorageResult<()> {
        if self.oracle.is_some() {
            self.range_removals
                .push((part, start.to_vec(), end.to_vec()));
            Ok(())
        } else {
            self.engine.remove_range(part, start, end)
        }
    }

    /// Record that the writer checks node `key` as this attempt leaves it
    /// before the commit: a condition on the final state, which a check at
    /// each intermediate write would refuse wrongly. Kept across the
    /// statements of an interactive transaction. A key recorded twice is
    /// recorded twice: the bulk write path pays a push, not a search.
    pub fn note_post_state_check(&mut self, key: &[u8]) {
        self.post_state_checks.push(key.to_vec());
    }

    /// The node keys recorded by [`Self::note_post_state_check`], taken by
    /// the check that runs once, right before the commit; a key appears as
    /// often as it was recorded.
    pub fn take_post_state_checks(&mut self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.post_state_checks)
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
    /// the snapshot after building the context and syncs it here. The pin
    /// follows the snapshot: history at the new seqno is protected for the
    /// rest of the transaction, the old pin is released.
    ///
    /// Setting the snapshot it already holds keeps the pin it has: the executor
    /// syncs before every read, and re-pinning the same seqno each time would
    /// take the watermark lock per read for nothing.
    pub fn set_snapshot(&mut self, snapshot: Option<StorageSnapshot>) {
        if snapshot.is_some() && snapshot == self.snapshot && self.snapshot_pin.is_some() {
            return;
        }
        self.snapshot = snapshot;
        self.snapshot_pin = snapshot.and_then(|s| self.engine.pin_snapshot_at(s));
        self.validate_from = snapshot.map(|s| Self::first_unseen(self.engine, s));
    }

    /// Take a snapshot whose pin the caller already holds, from
    /// [`StorageEngine::pin_new_snapshot`], rather than pinning it here after
    /// the fact.
    pub fn adopt_snapshot(&mut self, snapshot: StorageSnapshot, pin: Option<SnapshotPin>) {
        self.snapshot = Some(snapshot);
        self.snapshot_pin = pin;
        self.validate_from = Some(Self::first_unseen(self.engine, snapshot));
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

    /// Buffer an adjacency add, after everything staged before it. Not
    /// OCC-tracked.
    pub fn merge_adj_add(&mut self, adj_key: &[u8], uid: u64) {
        // A refused write is recorded and fails the statement and the commit.
        if self.admit_staged(STAGED_ADJ + adj_key.len()).is_err() {
            return;
        }
        self.merge_adj_ops.push((adj_key.to_vec(), AdjOp::Add(uid)));
    }

    /// Buffer an adjacency remove, after everything staged before it. Not
    /// OCC-tracked.
    pub fn merge_adj_remove(&mut self, adj_key: &[u8], uid: u64) {
        if self.admit_staged(STAGED_ADJ + adj_key.len()).is_err() {
            return;
        }
        self.merge_adj_ops
            .push((adj_key.to_vec(), AdjOp::Remove(uid)));
    }

    /// Decide every stated condition against authoritative state and this
    /// attempt's own staged writes.
    ///
    /// A condition the evaluator cannot decide refuses the commit rather than
    /// passing it: an undecidable claim is not a satisfied one, and admitting
    /// it would mean the protection is absent exactly where the evidence is.
    ///
    /// The claims [`decided_before_admission`] are left to
    /// [`Self::decide_uncovered_uniques`].
    fn evaluate_claims(&self) -> Result<(), CommitError> {
        use crate::engine::claims::evaluate::{Evaluation, Verdict};

        if self.claims.claims().iter().all(decided_before_admission) {
            return Ok(());
        }
        // The attempt's view as a sequence number, which is what "written
        // since" is asked in. The pinned snapshot is that number by
        // construction; `read_ts` only coincides with it where the oracle and
        // the engine share one space, and reading it here would make the
        // check silently pass wherever they do not. A transaction without a
        // snapshot is the legacy direct path, which has no view to protect and
        // applies as it goes.
        let view = self.snapshot.unwrap_or_else(|| self.engine.snapshot());
        let evaluation =
            Evaluation::new(self.engine, &self.merge_adj_ops, &self.write_buffer, view)
                .with_node_deltas(&self.merge_node_deltas)
                .with_kept_counts(&self.kept_counts);
        for claim in self
            .claims
            .claims()
            .iter()
            .filter(|claim| !decided_before_admission(claim))
        {
            match evaluation.decide(claim)? {
                Verdict::Holds => {}
                Verdict::Broken => {
                    return Err(CommitError::InvariantRefused {
                        reason: format!(
                            "{:?} no longer holds on {:?}",
                            claim.predicate, claim.scope
                        ),
                    });
                }
                Verdict::Undecidable => {
                    return Err(CommitError::InvariantRefused {
                        reason: format!(
                            "{:?} on {:?} cannot be decided here, so it cannot be admitted",
                            claim.predicate, claim.scope
                        ),
                    });
                }
                Verdict::HeldBy { holder, values } => {
                    return Err(match &claim.scope {
                        coordinode_core::txn::invariant::ClaimScope::UniqueValue {
                            generation,
                            ..
                        } => CommitError::UniqueValueHeld {
                            generation: *generation,
                            values,
                            holder,
                        },
                        scope => CommitError::InvariantRefused {
                            reason: format!(
                                "{:?} on {scope:?} is held by node {}",
                                claim.predicate,
                                holder.as_raw()
                            ),
                        },
                    });
                }
                Verdict::OverLimit { limit } => {
                    return Err(match &claim.scope {
                        coordinode_core::txn::invariant::ClaimScope::UniqueValue {
                            generation,
                            ..
                        } => CommitError::UniquenessUnresolved {
                            generation: *generation,
                            limit,
                        },
                        scope => CommitError::InvariantRefused {
                            reason: format!(
                                "{:?} on {scope:?} would read more than {limit} rows to decide",
                                claim.predicate
                            ),
                        },
                    });
                }
            }
        }
        Ok(())
    }

    /// Decide the unique values this attempt takes in indexes still being
    /// built, one read of the uncovered nodes per index generation for all
    /// of its values together, so the read limit bounds what the commit
    /// reads rather than what each value does.
    fn decide_uncovered_uniques(&self) -> Result<(), CommitError> {
        use crate::engine::claims::evaluate::{Verdict, WantedValues, uncovered_holders};
        use coordinode_core::txn::invariant::{ClaimPredicate, ClaimScope, UncoveredSource};

        // generation -> (the source with the least coverage, values wanted)
        let mut groups: rustc_hash::FxHashMap<
            coordinode_core::index::identity::GenerationId,
            (UncoveredSource, WantedValues<'_>),
        > = rustc_hash::FxHashMap::default();
        for claim in self.claims.claims() {
            let (
                ClaimScope::UniqueValue { generation, tuple },
                ClaimPredicate::UniqueHolder {
                    node,
                    uncovered: Some(source),
                },
            ) = (&claim.scope, &claim.predicate)
            else {
                continue;
            };
            let (group, wanted) = groups
                .entry(*generation)
                .or_insert_with(|| ((**source).clone(), WantedValues::default()));
            // Coverage only grows, so the oldest cursor stated is a bound
            // that holds for every claim of the group (no cursor sorts first).
            if source.covered_through < group.covered_through {
                group.covered_through = source.covered_through.clone();
            }
            wanted.insert(tuple.as_slice(), *node);
        }
        for (generation, (source, wanted)) in &groups {
            match uncovered_holders(
                self.engine,
                wanted,
                source,
                &self.write_buffer,
                &self.merge_node_deltas,
            )? {
                Verdict::HeldBy { holder, values } => {
                    return Err(CommitError::UniqueValueHeld {
                        generation: *generation,
                        values,
                        holder,
                    });
                }
                Verdict::OverLimit { limit } => {
                    return Err(CommitError::UniquenessUnresolved {
                        generation: *generation,
                        limit,
                    });
                }
                Verdict::Holds => {}
                // Not a verdict this read gives; not a pass either.
                other @ (Verdict::Broken | Verdict::Undecidable) => {
                    return Err(CommitError::InvariantRefused {
                        reason: format!("unique values of {generation} decided {other:?}"),
                    });
                }
            }
        }
        Ok(())
    }

    /// State a condition this attempt's result depends on.
    ///
    /// Called by the writer that knows the condition, as it writes: the
    /// storage layer sees keys and operands and cannot recover from them
    /// which graph predicate a mutation was constructed against.
    pub fn claim(&mut self, claim: Claim) {
        self.claims.insert(claim);
    }

    /// Record that this attempt read `label`'s schema at `revision` (`0` when
    /// the label has none). If the attempt writes, its commit depends on that
    /// schema still being the one in force; if it only reads, on nothing.
    pub fn note_label_schema_read(&mut self, label: &str, revision: u64) {
        self.schema_reads.insert(Claim::new(
            coordinode_core::txn::invariant::ClaimScope::LabelSchema(label.to_string()),
            coordinode_core::txn::invariant::ClaimPredicate::SchemaRead { revision },
            self.schema_generation,
        ));
    }

    /// Replace the trend each bound claim was stated with by the one this
    /// attempt's staged writes show for its scope, so the registry admits it
    /// beside attempts its writes cannot break and that cannot break it.
    /// Taken from the writes, which is all the attempt can change, and again
    /// on every commit attempt, since the writes are the same each time.
    fn state_bound_trends(&mut self) {
        use coordinode_core::txn::invariant::{ClaimPredicate, ClaimScope};

        if !self
            .claims
            .claims()
            .iter()
            .any(|c| matches!(c.predicate, ClaimPredicate::CardinalityBound { .. }))
        {
            return;
        }
        let stated = std::mem::take(&mut self.claims);
        for claim in stated.claims() {
            let mut claim = claim.clone();
            let Claim {
                scope, predicate, ..
            } = &mut claim;
            if let (
                ClaimScope::Incident {
                    node,
                    edge_type,
                    direction,
                },
                ClaimPredicate::CardinalityBound { trend, .. },
            ) = (&*scope, predicate)
            {
                *trend = crate::engine::cardinality::scope_trend(
                    *node,
                    edge_type,
                    *direction,
                    &self.merge_adj_ops,
                    &self.write_buffer,
                );
            }
            self.claims.insert(claim);
        }
    }

    /// Whether this attempt stages anything the commit would write.
    fn stages_writes(&self) -> bool {
        !self.write_buffer.is_empty()
            || !self.range_removals.is_empty()
            || !self.merge_adj_ops.is_empty()
            || !self.merge_node_deltas.is_empty()
            || !self.merge_counter_deltas.is_empty()
    }

    /// Refuse the commit when a record stated with [`Self::expect_version`]
    /// is no longer at the version named, carrying the version that is there.
    fn check_expected_versions(&self) -> Result<(), CommitError> {
        for (part, key, expected) in &self.expected_versions {
            let current = self.engine.record_version(*part, key)?;
            if current != *expected {
                return Err(CommitError::RevisionMismatch {
                    expected: *expected,
                    current,
                });
            }
        }
        Ok(())
    }

    /// Write this record only while its version is still `expected`.
    ///
    /// `None` means the record must not exist: the create-if-absent form,
    /// which is the same condition with nothing on the other side of it.
    ///
    /// The condition is checked at commit against committed state, in the
    /// same protected step as the write set, because a check performed here
    /// would prove something about a moment that has passed. What the caller
    /// gets from stating it is a refusal carrying the version that is there
    /// now, so a retry needs no second read.
    pub fn expect_version(
        &mut self,
        part: Partition,
        key: &[u8],
        expected: Option<u64>,
    ) -> StorageResult<()> {
        if part.is_commutative() {
            return Err(StorageError::InvalidConfig(format!(
                "the {part:?} partition is merge-composed: its rows are folded from \
                 operands, so a write cannot be conditioned on their version"
            )));
        }
        // Legacy mode applies each write as it is made, so a condition
        // checked at the commit would see this attempt's own writes. It is
        // decided now, before them; one that fails now is kept and refuses
        // the commit as usual.
        if self.oracle.is_none() && self.engine.record_version(part, key)? == expected {
            return Ok(());
        }
        self.expected_versions.push((part, key.to_vec(), expected));
        Ok(())
    }

    /// Bind this transaction's index effects to the definition stored at
    /// `key` as it was at `version`: a transition or drop of the index
    /// before this commit refuses it, so no effect lands under a retired
    /// binding. Conditioned once per definition, however many effects follow.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::expect_version`].
    pub fn bind_index_definition(&mut self, key: &[u8], version: Option<u64>) -> StorageResult<()> {
        if self.derived.bind(key) {
            self.expect_version(Partition::Schema, key, version)?;
        }
        Ok(())
    }

    /// Stage one membership change in a DERIVED index, of a node or of one
    /// version of a temporal node (`owner`): the entry effects go to the
    /// write buffer, as for any index, so this transaction reads them and its
    /// unique values conflict with a concurrent writer of the same value; the
    /// unit logs the change as sealed work under `binding` instead of the
    /// entries. `old` is the membership before this change, `new` the one
    /// after.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::put`] and [`Self::delete`].
    pub fn stage_derived(
        &mut self,
        binding: &coordinode_core::txn::proposal::IndexBinding,
        owner: coordinode_core::index::derive::EntryOwner,
        old: Option<Vec<coordinode_core::graph::types::Value>>,
        new: Option<Vec<coordinode_core::graph::types::Value>>,
        effects: &[coordinode_core::index::derive::EntryEffect],
    ) -> StorageResult<()> {
        for effect in effects {
            match &effect.value {
                Some(value) => self.put(Partition::Idx, &effect.key, value)?,
                None => self.delete(Partition::Idx, &effect.key)?,
            }
        }
        self.derived.stage(
            binding,
            owner,
            old,
            new,
            effects.iter().map(|e| e.key.clone()),
        );
        Ok(())
    }

    /// The committed version of a record now, the number
    /// [`Self::expect_version`] compares against. See
    /// [`StorageEngine::record_version`].
    pub fn record_version(&self, part: Partition, key: &[u8]) -> StorageResult<Option<u64>> {
        self.engine.record_version(part, key)
    }

    /// The conditions stated so far, for a caller that has to report them.
    pub fn expected_versions(&self) -> &[(Partition, Vec<u8>, Option<u64>)] {
        &self.expected_versions
    }

    /// The engine this attempt reads and commits to.
    pub(crate) fn engine(&self) -> &'a StorageEngine {
        self.engine
    }

    /// The schema generation this attempt evaluates its predicates under.
    ///
    /// A writer stamps it on every claim it states, which is why it is read
    /// once per attempt rather than once per claim: two claims of one attempt
    /// that disagreed about the generation would describe two attempts.
    pub fn schema_generation(&self) -> u64 {
        self.schema_generation
    }

    /// Record that this attempt changes a schema definition.
    ///
    /// The generation moves when the change lands, not when it is staged, so
    /// an attempt that is refused or rolled back invalidates nothing.
    pub fn note_schema_change(&mut self) {
        self.schema_changed = true;
    }

    /// Whether this attempt carries private catalog changes. A stateless
    /// parallel reader cannot resolve those through committed storage alone.
    pub fn has_schema_changes(&self) -> bool {
        self.schema_changed
    }

    /// The conditions stated so far.
    pub fn claims(&self) -> &ClaimSet {
        &self.claims
    }

    /// Both cardinality measures of the scope of `node`'s `edge_type` edges
    /// in `direction`, as this attempt would leave it: the latest committed
    /// edges with its staged writes applied, counted by logical identity.
    /// `None` when the scope cannot be counted exactly: a temporal type,
    /// whose bound holds over valid time, or a discriminated pair adjacent
    /// with no instance behind it.
    ///
    /// The count is an observation, not a guarantee: a commit landing after
    /// it can change it. An attempt whose result depends on a bound states a
    /// `CardinalityBound` claim, which the commit decides against the state
    /// it lands on.
    ///
    /// # Errors
    ///
    /// A stored posting, entry or definition does not decode, or a read
    /// fails.
    pub fn incident_count(
        &self,
        node: coordinode_core::graph::node::NodeId,
        edge_type: &str,
        direction: coordinode_core::txn::invariant::Direction,
    ) -> StorageResult<Option<coordinode_core::graph::cardinality::IncidentCount>> {
        crate::engine::cardinality::enumerated_count(
            self.engine,
            node,
            edge_type,
            direction,
            &self.merge_adj_ops,
            &self.write_buffer,
        )
    }

    /// Replay this transaction's own staged adjacency operands onto `plist`,
    /// in the order they were staged, so a read sees its own writes the way
    /// the commit will apply them.
    pub fn apply_staged_adj(&self, adj_key: &[u8], plist: &mut PostingList) {
        for (key, op) in &self.merge_adj_ops {
            if key.as_slice() != adj_key {
                continue;
            }
            match op {
                AdjOp::Add(uid) => {
                    plist.insert(*uid);
                }
                AdjOp::Remove(uid) => {
                    plist.remove(*uid);
                }
            }
        }
    }

    /// Buffer a node merge operand (pre-encoded document delta) at `node_key`.
    pub fn push_node_delta(&mut self, node_key: Vec<u8>, operand: Vec<u8>) {
        // A refused write is recorded and fails the statement and the commit.
        if self
            .admit_staged(STAGED_DELTA + node_key.capacity() + operand.capacity())
            .is_err()
        {
            return;
        }
        self.merge_node_deltas.push((node_key, operand));
    }

    /// Buffer a commutative counter increment at `counter_key` (statistics
    /// counters, degree caches). Applied at commit as a `CounterMerge`
    /// operand; not OCC-tracked. Deltas to the same key COALESCE (summed),
    /// so bulk statements stage one entry per distinct counter, and a key
    /// already present is found by slice lookup without allocating.
    /// A sum that leaves `i64` is recorded here and refused at commit rather
    /// than staged: an operand that cannot be folded would otherwise be
    /// acknowledged and then fail in a compaction, where the caller is long
    /// gone and the partition stops making progress. The decision belongs
    /// before the durable promise, and this is where it can still be made.
    pub fn push_counter_delta(&mut self, counter_key: &[u8], delta: i64) {
        if delta == 0 {
            return;
        }
        if let Some(sum) = self.merge_counter_deltas.get_mut(counter_key) {
            match sum.checked_add(delta) {
                Some(next) => *sum = next,
                None => {
                    if self.counter_overflow.is_none() {
                        self.counter_overflow = Some(counter_key.to_vec());
                    }
                }
            }
        } else {
            // A refused write is recorded and fails the statement and the
            // commit.
            if self
                .admit_staged(STAGED_COUNTER + counter_key.len())
                .is_err()
            {
                return;
            }
            self.merge_counter_deltas
                .insert(counter_key.to_vec(), delta);
        }
    }

    /// Whether any merge operand (adjacency, node delta, or counter delta) is
    /// buffered. Used by the commit path's read-only fast exit.
    pub fn has_pending_merges(&self) -> bool {
        !self.merge_adj_ops.is_empty()
            || !self.merge_node_deltas.is_empty()
            || !self.merge_counter_deltas.is_empty()
    }

    /// Leave the open transactions a schema change waits for
    /// ([`StorageEngine::await_transactions_through`](crate::engine::core::StorageEngine::await_transactions_through)),
    /// when this transaction has staged nothing: a statement that waits for
    /// the index build it started would otherwise wait for itself. A
    /// transaction holding staged writes stays, since an index built past it
    /// would miss them. Returns whether it left.
    pub fn release_from_schema_waits(&mut self) -> bool {
        let staged = !self.write_buffer.is_empty()
            || self.has_pending_merges()
            || !self.range_removals.is_empty();
        if staged {
            return false;
        }
        self.open = None;
        true
    }

    /// Borrow the buffered adjacency operands in the order they were staged.
    pub fn merge_adj_ops(&self) -> &[(Vec<u8>, AdjOp)] {
        &self.merge_adj_ops
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

    /// Take the buffered adjacency operands, leaving the buffer empty and the
    /// order intact.
    pub fn take_merge_adj_ops(&mut self) -> Vec<(Vec<u8>, AdjOp)> {
        std::mem::take(&mut self.merge_adj_ops)
    }

    /// Take the buffered node merge operands, leaving the buffer empty.
    pub fn take_node_deltas(&mut self) -> Vec<(Vec<u8>, Vec<u8>)> {
        std::mem::take(&mut self.merge_node_deltas)
    }

    /// Clear all buffered merge operands (commit path's volatile no-drain
    /// branch: local writes already applied, nothing to replicate).
    pub fn clear_merges(&mut self) {
        self.merge_adj_ops.clear();
        self.merge_node_deltas.clear();
        self.merge_counter_deltas.clear();
    }

    /// Drop any buffered adjacency adds/removes for `adj_key`. Used when a
    /// node-delete cascade tombstones the posting list — pending merge
    /// operands must not resurrect it.
    pub fn drop_adj_merges(&mut self, adj_key: &[u8]) {
        self.merge_adj_ops
            .retain(|(key, _)| key.as_slice() != adj_key);
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
    /// This is the single Layer-3 commit locus: OCC validation,
    /// `commit_ts` assignment, write-concern fan-out, and the Raft proposal
    /// pipeline all live here. Adjacency (`adj:`) keys bypass conflict checking
    /// because posting-list operations are commutative merge operands.
    ///
    /// Returns the commit timestamp used (or `None` for a read-only / legacy
    /// transaction) plus the committed Raft index on the pipeline path.
    pub fn commit(&mut self, ctx: &CommitContext<'_>) -> Result<CommitOutcome, CommitError> {
        // An operand whose fold cannot succeed is refused here, where the
        // caller is still listening, rather than acknowledged and then met
        // again in a compaction that has nowhere to report it.
        if let Some(key) = &self.counter_overflow {
            return Err(CommitError::CounterOverflow {
                key: String::from_utf8_lossy(key).into_owned(),
            });
        }
        // Writes past a refusal were never staged: committing the rest would
        // apply part of what a statement meant to write.
        if let Some(stop) = self.budget_refusal {
            return Err(CommitError::Budget(stop));
        }
        // Likewise work every member would refuse to derive.
        self.derived
            .check_fan_out(crate::engine::MAX_DERIVED_EFFECTS)
            .map_err(|limit| CommitError::IndexFanOut { limit })?;

        // Write-admission gate: under Stop pressure (storage over its
        // compaction-debt stop threshold) a commit carrying writes is
        // rejected BEFORE anything is applied, as a retryable error. One
        // relaxed atomic load; read-only commits are exempt (nothing to
        // admit), and the Raft apply path never goes through here, so
        // committed entries are never gated.
        let has_writes = !self.write_buffer.is_empty()
            || !self.range_removals.is_empty()
            || self.has_pending_merges();
        if has_writes
            && !self.pressure_exempt
            && matches!(
                self.engine.write_pressure(),
                crate::engine::core::WritePressure::Stop
            )
        {
            return Err(CommitError::Backpressure);
        }

        // Flush adj merge buffers even in legacy (no MVCC) mode.
        // Legacy puts write directly to engine, but merge adds are buffered.
        if self.oracle.is_none() {
            // No admission without a clock: the conditions are checked
            // against the state as it stands, as the writes are applied to it.
            self.check_expected_versions()?;
            // Its point writes are already applied, so no state before them
            // is left to count a transition from.
            if !crate::engine::cardinality::plan(
                self.engine,
                &self.merge_adj_ops,
                &self.write_buffer,
            )?
            .is_empty()
            {
                return Err(CommitError::InvariantRefused {
                    reason: "an edge type with kept cardinality counts is written only by a \
                             transactional commit"
                        .to_string(),
                });
            }
            let staged = std::mem::take(&mut self.merge_adj_ops);
            for (key, operand) in encode_staged_adj(&staged) {
                self.engine.merge(Partition::Adj, key, &operand)?;
            }
            for (key, operand) in self.merge_node_deltas.drain(..) {
                self.engine.merge(Partition::Node, &key, &operand)?;
            }
            for (key, delta) in self.merge_counter_deltas.drain().filter(|(_, d)| *d != 0) {
                self.engine
                    .merge(Partition::Counter, &key, &encode_counter_delta(delta))?;
            }
            // Legacy mode — writes already applied.
            self.publish_schema_change();
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
                });
            }
        };

        let has_merge_ops = self.has_pending_merges();
        if self.write_buffer.is_empty() && self.range_removals.is_empty() && !has_merge_ops {
            // Read-only: nothing to admit or apply, but a condition the
            // caller stated is still the answer it asked for. Checked against
            // committed state now, which is where a commit with no writes
            // takes effect.
            self.check_expected_versions()?;
            return Ok(CommitOutcome {
                commit_ts: Some(self.read_ts),
                applied_index: None,
            });
        }

        // Protection is installed here, in one place, and nothing of this
        // attempt is applied until all of it is: the conditions it declared,
        // the timestamp it will land at, and the keys it will write. They
        // were once taken apart, the conditions at the top of the commit and
        // the keys much later, with the admission checks and the mutation
        // building in between; no schedule could tell the two forms apart,
        // since the apply follows both either way, but an attempt that failed
        // in between held a reservation it had no use for, and a reader of
        // this function had to reconstruct that the two belonged together.
        // The schemas the attempt read become conditions only now that it is
        // known to write: a write was validated under them, a read was not.
        if self.stages_writes() {
            for read in std::mem::take(&mut self.schema_reads).claims() {
                self.claims.insert(read.clone());
            }
        }
        self.fold_node_deltas_into_writes()?;
        // The pairs whose kept counts this commit changes, claimed with the
        // rest so no other count of them is decided beside it.
        let counts =
            crate::engine::cardinality::plan(self.engine, &self.merge_adj_ops, &self.write_buffer)?;
        for claim in counts.claims(self.schema_generation) {
            self.claims.insert(claim);
        }
        self.state_bound_trends();
        // Before the timestamp: every snapshot taken after it waits for this
        // commit to land, so a long read here would hold up all of them.
        self.decide_uncovered_uniques()?;
        let reservation = if self.claims.is_empty() {
            None
        } else {
            let engine: &'a StorageEngine = self.engine;
            Some(
                engine
                    .claim_registry()
                    .reserve_attempt(&self.claims)
                    .map_err(|refusal| CommitError::InvariantRefused {
                        reason: format!("{refusal:?}"),
                    })?,
            )
        };

        // The scope this commit is about to write, registered together with
        // the timestamp it will land at and held until its writes are there.
        //
        // Validation below reads committed state, and a commit in flight is
        // exactly what committed state does not contain yet. Without this
        // registration two commits validate in the same window, each finding
        // the other's keys untouched, and both apply: the later one replaces
        // the earlier, which is the lost update first-committer-wins exists to
        // prevent. Registering first is what leaves no interval in which a
        // writer sees neither the effect nor the obligation.
        //
        // Commutative partitions stay out of the scope for the same reason
        // they are exempt from validation below: their concurrency story is
        // the merge operator, and excluding them here would serialise the
        // super-node writes that path exists to keep parallel.
        //
        // A node document delta writes its record as surely as a whole write
        // does, so its key is in the scope too: two attempts that change one
        // node, in either form, are competing for it.
        let mut scope: Vec<(Partition, Vec<u8>)> = self
            .write_buffer
            .keys()
            .filter(|(part, _)| !part.is_commutative())
            .map(|(part, key)| (*part, key.clone()))
            .collect();
        for (key, _) in &self.merge_node_deltas {
            if !scope.iter().any(|(p, k)| *p == Partition::Node && k == key) {
                scope.push((Partition::Node, key.clone()));
            }
        }
        // The records this commit is conditioned on but does not write. The
        // condition is checked against committed state below; registered
        // here, a commit in flight that writes one of them is refused, or
        // refuses this one, instead of moving it after the check.
        let guards: Vec<(Partition, Vec<u8>)> = self
            .expected_versions
            .iter()
            .filter(|(part, key, _)| !scope.iter().any(|(p, k)| p == part && k == key))
            .map(|(part, key, _)| (*part, key.clone()))
            .collect();
        let pending = self.engine.pending_commits();
        let admitted = match self.overlap_wait {
            Some(wait) => pending.admit_allocated_waiting(
                || oracle.next().as_raw(),
                scope.clone(),
                guards,
                wait,
            ),
            None => pending.admit_allocated(|| oracle.next().as_raw(), scope.clone(), guards),
        };
        let (commit_ts_raw, admission) = admitted.map_err(|refusal| match refusal {
            crate::engine::pending::Refusal::Overlap {
                partition,
                holder_ts,
                ..
            } => CommitError::Conflict(format!(
                "write conflict: a key in the {partition:?} partition is already \
                     being written by a transaction committing at {holder_ts}. \
                     Nothing was applied; retry the whole transaction."
            )),
            crate::engine::pending::Refusal::Reserved { partition, .. } => {
                CommitError::Conflict(format!(
                    "write conflict: a key in the {partition:?} partition is reserved by a \
                     catalog change waiting to commit. Nothing was applied; retry the whole \
                     transaction, which then runs after it."
                ))
            }
            // Not a conflict with anyone in particular: the node is
            // holding more unfinished commits than it admits, and the
            // answer is the same one backpressure gives, a retry after a
            // delay rather than an immediate one that bounces off the
            // same ceiling.
            crate::engine::pending::Refusal::AtCapacity { limit } => {
                tracing::warn!(
                    limit,
                    "commit refused: more commits are in flight than this node admits"
                );
                CommitError::Backpressure
            }
        })?;
        let commit_ts = Timestamp::from_raw(commit_ts_raw);

        // The count changes, taken against the pairs as they stand now that
        // no other count of them can land: derived before the bounds are
        // decided, which read the counts these changes leave.
        self.kept_counts.clear();
        if !counts.is_empty() {
            match crate::engine::cardinality::deltas(
                self.engine,
                &counts,
                &self.merge_adj_ops,
                &self.write_buffer,
            )? {
                Ok(kept) => self.kept_counts = kept,
                Err(reason) => return Err(CommitError::InvariantRefused { reason }),
            }
        }

        // Both halves of the condition check, now that the protection is
        // installed: the registry saw the attempts in flight beside this one,
        // and this sees the state they have all committed. Neither is
        // sufficient alone, and the evaluation reads storage, so it belongs
        // outside the tables rather than inside them.
        if !self.claims.is_empty() {
            self.evaluate_claims()?;
        }

        // Records written on the condition of their version. Checked here,
        // under the same admission as the write set, because this is where
        // the answer is still true when the writes land: the written keys
        // and the guarded ones are registered, so nobody else can move these
        // records in between.
        self.check_expected_versions()?;

        // First-committer-wins over the WRITE set (seqno probing, write
        // keys only). Every mainstream engine conflicts on concurrent writes
        // and never on stale reads at its default level: PostgreSQL and Oracle
        // row-lock writers, MongoDB conflicts on a concurrently written
        // document, Dgraph fingerprints mutations. Validating the read set
        // instead is serializable-strength enforcement and lives behind the
        // opt-in level and FOR UPDATE, not here — imposed by default it aborts
        // every read-heavy transaction that raced any writer.
        //
        // Commutative partitions are exempt: their concurrency story is the
        // merge operator, and adding them here would resurrect the super-node
        // abort storms the merge path exists to prevent. Detects ABA (write +
        // revert) via lsm-tree seqno inspection, same as the read-set probe
        // this replaced.
        // Validated against the view this transaction read, not against the
        // timestamp it was issued. The two are not the same number: an
        // attempt's timestamp identifies it and orders it, while its view is
        // where its values came from, and it is the view that says which
        // writes it could not have seen. Validating against the higher of the
        // two would silently forgive every write in between. And the view
        // starts below its snapshot when commits were still in flight under
        // it: those land beneath the snapshot after the reads were made.
        let occ_read_ts = self.validate_from.unwrap_or_else(|| self.read_ts.as_raw());
        for (part, key) in &scope {
            // Inclusive: a snapshot sees sequence numbers strictly below
            // itself, so a write landing at exactly that number is one this
            // transaction could not have seen. The strict comparison missed
            // precisely the closest concurrent writer, which is the one most
            // likely to be there.
            if self
                .engine
                .written_since_snapshot(*part, key, occ_read_ts)?
            {
                return Err(CommitError::Conflict(format!(
                    "write conflict: key in {part:?} partition was also written by a \
                     transaction that committed after start_ts={occ_read_ts}. Nothing \
                     was applied; retry the whole transaction.",
                )));
            }
        }

        // Drain the buffered point writes; each write-concern branch below
        // applies / replicates them.
        let mut wb = std::mem::take(&mut self.write_buffer);

        // A write concern the group cannot honour is refused before anything is
        // written; the member count is checked at the pipeline, which knows it.
        ctx.write_concern
            .validate(None)
            .map_err(|e| CommitError::Serialization(e.to_string()))?;

        // w:0 takes the same path as every other commit below. A write concern
        // decides when the caller is answered, never whether the write is
        // replicated: a commit applied to this member alone would be a record no
        // other member ever sees, on a follower and on the leader alike.

        // Volatile journal (j:memory / j:cache with the leader as the one
        // acknowledging member):
        // 1. Apply locally for immediate read visibility
        // 2. Buffer mutations in DrainBuffer for background Raft replication
        // 3. Return immediately — drain thread handles durability
        //
        // Crash before drain = data lost (explicit contract).
        // Drained entries preserve original commit_ts for CDC fidelity.
        if ctx.write_concern.is_volatile() {
            // Step 1: Apply locally for read visibility.
            for (part, start, end) in &self.range_removals {
                self.engine.remove_range(*part, start, end)?;
            }
            for ((part, key), value) in &wb {
                match value {
                    Some(v) => self.engine.put(*part, key, v)?,
                    None => self.engine.delete(*part, key)?,
                }
            }
            for (key, operand) in encode_staged_adj(&self.merge_adj_ops) {
                self.engine.merge(Partition::Adj, key, &operand)?;
            }
            for (key, operand) in &self.merge_node_deltas {
                self.engine.merge(Partition::Node, key, operand)?;
            }
            for (key, delta) in self
                .merge_counter_deltas
                .iter()
                .chain(self.kept_counts.iter())
                .filter(|(_, d)| **d != 0)
            {
                self.engine
                    .merge(Partition::Counter, key, &encode_counter_delta(*delta))?;
            }

            // The entry reaches the log only when it is drained: until then
            // its timestamp stays registered as not yet logged, taken while
            // the admission still holds it, so no closed bound passes over it.
            let unlogged = ctx
                .drain_buffer
                .is_some()
                .then(|| self.engine.pending_commits().hold(commit_ts.as_raw()));

            // The writes are local state now, so the registration has done its
            // work: from here a validating writer finds them by reading, and a
            // reader's snapshot may cover this timestamp. What remains is
            // durability, which is not what the floor is about. The claims go
            // with it: an attempt after this one is judged against the state
            // these writes left.
            drop(admission);
            drop(reservation);

            // Step 2: Buffer for drain (if drain buffer is available).
            if let Some(drain_buf) = ctx.drain_buffer {
                let mutations = self.seal_unit(wb);
                let mut entry = DrainEntry::new(mutations, commit_ts, self.read_ts);
                if let Some(unlogged) = unlogged {
                    entry = entry.holding(Box::new(unlogged));
                }

                // j:cache: persist to NVMe before ACK for process-crash recovery.
                // j:memory skips this: data loss on crash is the explicit contract.
                if ctx.write_concern.journal == Journal::Cache {
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
                self.range_removals.clear();
                self.merge_adj_ops.clear();
                self.merge_node_deltas.clear();
                self.merge_counter_deltas.clear();
                self.kept_counts.clear();
                self.derived = derived::DerivedLedger::default();
            }

            self.publish_schema_change();
            return Ok(CommitOutcome {
                commit_ts: Some(commit_ts),
                applied_index: None,
            });
        }

        // Journaled path (and w:0): apply through the proposal pipeline, or
        // write directly when none is configured.
        //
        // When a pipeline is configured, mutations are packaged into a
        // RaftProposal and sent through the pipeline; `w` decides how many
        // members must hold the entry before the pipeline returns. In
        // single-node mode every `w` is satisfied by the local apply.
        //
        // When no pipeline is configured (legacy/test mode), mutations are
        // written directly to the engine.
        // One mutation list for both application paths below: proposed
        // through the pipeline, or applied directly at commit_ts.
        let mutations = self.seal_unit(wb);

        let mut applied_index: Option<u64> = None;
        if let (Some(pipeline), Some(id_gen)) = (ctx.pipeline, ctx.id_gen) {
            let proposal = RaftProposal {
                id: id_gen.next(),
                mutations,
                commit_ts,
                start_ts: self.read_ts,
                bypass_rate_limiter: false,
            };

            // The wait is bounded by wtimeout when one is set. Per MongoDB:
            // on timeout the data is NOT rolled back, the proposal may still
            // commit after the timeout fires. In embedded/single-node mode the
            // default pipeline applies at once and the timeout is moot.
            let timeout = (ctx.write_concern.timeout_ms > 0)
                .then(|| std::time::Duration::from_millis(u64::from(ctx.write_concern.timeout_ms)));
            let outcome = pipeline
                .propose_with_ack(&proposal, ctx.write_concern.w, timeout)
                .map_err(proposal_err_to_commit)?;
            // Record the committed Raft index of this write so the gRPC layer
            // can return it as the causal operationTime token. `None` in
            // local/embedded mode (no Raft log).
            applied_index = outcome.applied_index;
        } else {
            // Direct-write path (no pipeline configured): the same single-seqno
            // apply the pipelines perform, at this transaction's commit_ts,
            // journalled first when the engine keeps a journal.
            if self.engine.has_journal() {
                self.engine
                    .commit_journaled(&mutations, commit_ts.as_raw())?;
            } else {
                self.engine
                    .apply_proposal_at(&mutations, commit_ts.as_raw())?;
            }
        }

        // Released where the writes become local state, not where the caller
        // is answered. The registration exists to close the window between
        // validating against committed state and being part of it; holding it
        // through a replication wait would pin every reader's snapshot on this
        // node for as long as the slowest member takes to acknowledge, which
        // is durability, not visibility. The claims are released at the same
        // point: held through the fsync below, they refused an attempt that
        // only had to be judged against the state these writes left.
        drop(admission);
        drop(reservation);

        // j:journal on a member without a Raft log (legacy / embedded direct
        // write): force the fsync after commit. With FlushPolicy::SyncPerBatch
        // this is already done by WriteBatch; with Periodic/Manual policies the
        // explicit persist is what makes the journal promise true. A caller
        // that waits for nobody (w:0) is not held for it.
        if ctx.write_concern.journal == Journal::Journal
            && !ctx.write_concern.w.is_fire_and_forget()
            && ctx.pipeline.is_none()
            && self.engine.flush_policy() != crate::engine::config::FlushPolicy::SyncPerBatch
        {
            self.engine
                .persist()
                .map_err(|e| CommitError::Serialization(format!("journal fsync failed: {e}")))?;
        }

        self.publish_schema_change();
        Ok(CommitOutcome {
            commit_ts: Some(commit_ts),
            applied_index,
        })
    }

    /// The unit this attempt commits, from its drained write buffer `wb` and
    /// merge buffers: point writes (a DERIVED index's entries left out),
    /// merge operands in their staged order, dense delete runs coalesced into
    /// range deletes, and last the sealed DERIVED work, whose record sources
    /// name positions of this final list.
    /// Fold the document deltas of a node this attempt also writes whole
    /// into that write: a delta staged after the node's record was put (a
    /// node created, then SET, in one statement) is applied to the record,
    /// and one staged before the node is deleted goes with it. One batch then
    /// never carries a put or a delete and a merge for one key, which the
    /// engine refuses, and the record lands as the statement left it.
    fn fold_node_deltas_into_writes(&mut self) -> Result<(), CommitError> {
        use coordinode_core::graph::doc_delta::{DocDelta, PREFIX_DOC_DELTA};
        use coordinode_core::graph::node::NodeRecord;

        if self.merge_node_deltas.is_empty() {
            return Ok(());
        }
        let deltas = std::mem::take(&mut self.merge_node_deltas);
        let mut folded: HashMap<Vec<u8>, Vec<DocDelta>> = HashMap::new();
        for (key, operand) in deltas {
            match self.write_buffer.get(&(Partition::Node, key.clone())) {
                None => self.merge_node_deltas.push((key, operand)),
                // Deleted in this attempt: nothing is left for it to change.
                Some(None) => {}
                Some(Some(_)) => {
                    let delta = match operand.split_first() {
                        Some((&PREFIX_DOC_DELTA, body)) => DocDelta::decode(body).map_err(|e| {
                            CommitError::Serialization(format!("node document delta: {e}"))
                        })?,
                        _ => {
                            return Err(CommitError::Serialization(
                                "node operand without the document-delta prefix".to_string(),
                            ));
                        }
                    };
                    folded.entry(key).or_default().push(delta);
                }
            }
        }
        for (key, deltas) in folded {
            let slot = self.write_buffer.get_mut(&(Partition::Node, key));
            let Some(Some(bytes)) = slot else {
                continue;
            };
            let mut record = crate::engine::merge::decode_node_record(bytes).map_err(|e| {
                CommitError::Serialization(format!("node record under a delta: {e}"))
            })?;
            crate::engine::merge::apply_doc_deltas_to_record(&mut record, &deltas);
            *bytes = NodeRecord::to_msgpack(&record)
                .map_err(|e| CommitError::Serialization(format!("node record: {e}")))?;
        }
        Ok(())
    }

    fn seal_unit(&mut self, wb: HashMap<(Partition, Vec<u8>), Option<Vec<u8>>>) -> Vec<Mutation> {
        let derived = std::mem::take(&mut self.derived);
        debug_assert!(
            self.range_removals.iter().all(|(rp, start, end)| {
                !wb.keys().any(|(p, k)| {
                    p == rp && k.as_slice() >= start.as_slice() && k.as_slice() < end.as_slice()
                })
            }),
            "an attempt writes inside a range it removes"
        );
        let mut mutations: Vec<Mutation> = std::mem::take(&mut self.range_removals)
            .into_iter()
            .map(|(part, start, end)| Mutation::RemoveRange {
                partition: partition_to_id(part),
                start,
                end,
            })
            .collect();
        mutations.extend(
            wb.into_iter()
                .filter(|((part, key), _)| !(*part == Partition::Idx && derived.owns(key)))
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
                }),
        );

        // Adj merge operands: bypass MVCC, raw keys, staged order kept.
        let staged = std::mem::take(&mut self.merge_adj_ops);
        for (key, operand) in encode_staged_adj(&staged) {
            mutations.push(Mutation::Merge {
                partition: PartitionId::Adj,
                key: key.to_vec(),
                operand,
            });
        }
        for (key, operand) in self.merge_node_deltas.drain(..) {
            mutations.push(Mutation::Merge {
                partition: PartitionId::Node,
                key,
                operand,
            });
        }
        for (key, delta) in self
            .merge_counter_deltas
            .drain()
            .chain(self.kept_counts.drain())
            .filter(|(_, d)| *d != 0)
        {
            mutations.push(Mutation::Merge {
                partition: PartitionId::Counter,
                key,
                operand: encode_counter_delta(delta),
            });
        }

        // Coalesce dense runs of point deletes into range deletes before
        // proposing: a bulk delete ("delete all relationships between these
        // nodes", DROP) replicates + PITR-logs as a few range ops instead of N
        // point tombstones. Non-deletes / short runs untouched.
        let mut mutations = coordinode_core::txn::coalesce::coalesce_delete_mutations(
            mutations,
            coordinode_core::txn::coalesce::DEFAULT_MIN_RUN,
        );
        if !derived.is_empty() {
            derived.seal(&mut mutations);
        }
        mutations
    }

    /// Move the engine's schema generation if this attempt changed a
    /// definition and its writes landed. Called on the paths that applied
    /// them, never on one that returned early.
    fn publish_schema_change(&mut self) {
        if std::mem::take(&mut self.schema_changed) {
            self.engine.note_schema_change();
        }
    }

    /// MVCC-aware prefix scan: snapshot results overlaid with buffered writes
    /// in key order, the same view [`Self::get`] gives one key at a time: a
    /// buffered value replaces the storage row for its key, and a buffered
    /// tombstone removes it. Legacy mode scans the engine directly and
    /// overlays the (empty-in-legacy) buffer.
    pub fn prefix_scan(&self, part: Partition, prefix: &[u8]) -> StorageResult<Vec<KvPair>> {
        // Own writes under the prefix, values and tombstones, in key order:
        // the buffer is a hash map, so its iteration order is arbitrary, and
        // readers group consecutive keys (a node's versions).
        let mut buffer_matches: Vec<(Vec<u8>, Option<Vec<u8>>)> = self
            .write_buffer
            .iter()
            .filter(|((p, k), _)| *p == part && k.starts_with(prefix))
            .map(|((_, k), v)| (k.clone(), v.clone()))
            .collect();
        buffer_matches.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let stored: Vec<KvPair> = match self.snapshot {
            // Scanned keys are not conflict-tracked: see `get` — the default
            // level validates writes only, and FOR UPDATE is the opt-in that
            // pins scanned rows into the scope.
            Some(snap) => self
                .engine
                .snapshot_prefix_scan(&snap, part, prefix)?
                .into_iter()
                .map(|(k, v)| (k, v.to_vec()))
                .collect(),
            None => {
                // Legacy mode: writes apply straight to the engine, so the
                // buffer is empty and `buffer_matches` overlays nothing.
                let mut rows = Vec::new();
                for guard in self.engine.prefix_scan(part, prefix)? {
                    let (k, v) = guard.into_inner()?;
                    rows.push((k.to_vec(), v.to_vec()));
                }
                rows
            }
        };
        Ok(merge_overlay(stored, buffer_matches))
    }

    /// The rows [`Self::prefix_scan`] returns, handed to `visit` one at a time
    /// in key order instead of collected: the caller keeps, and pays for, only
    /// what it retains.
    ///
    /// Each row counts one unit of `budget`'s work (which checks its deadline
    /// and cancellation) and holds its bytes against the budget while `visit`
    /// sees it; the index of this transaction's own writes under the prefix is
    /// reserved before it is built. A refusal stops the scan before the row
    /// it was for is read further.
    ///
    /// # Errors
    ///
    /// A storage failure, the budget's refusal, or the first error of `visit`.
    pub fn prefix_for_each<E>(
        &self,
        part: Partition,
        prefix: &[u8],
        budget: &coordinode_core::budget::QueryBudget,
        mut visit: impl FnMut(&[u8], &[u8]) -> Result<(), E>,
    ) -> Result<(), E>
    where
        E: From<StorageError> + From<coordinode_core::budget::BudgetStop>,
    {
        // Own writes under the prefix, by reference and in key order, so the
        // stored rows they shadow are skipped as they stream past.
        let own = self
            .write_buffer
            .keys()
            .filter(|(p, k)| *p == part && k.starts_with(prefix))
            .count();
        // A size past u64 asks for more than any limit, and is refused as such.
        let _own_index = budget.reserve(
            own.checked_mul(core::mem::size_of::<(&[u8], Option<&[u8]>)>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(u64::MAX),
        )?;
        let mut overlay: Vec<(&[u8], Option<&[u8]>)> = Vec::with_capacity(own);
        overlay.extend(
            self.write_buffer
                .iter()
                .filter(|((p, k), _)| *p == part && k.starts_with(prefix))
                .map(|((_, k), v)| (k.as_slice(), v.as_deref())),
        );
        overlay.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let mut overlay = overlay.into_iter().peekable();

        let mut emit = |key: &[u8], value: &[u8]| -> Result<(), E> {
            budget.work(1)?;
            let _row = budget.reserve(row_bytes(key, value))?;
            visit(key, value)
        };
        let mut stored_row = |key: &[u8], value: &[u8]| -> Result<(), E> {
            // Own writes before this key come first, in order.
            while let Some((own_key, own_value)) = overlay.next_if(|(k, _)| *k < key) {
                if let Some(own_value) = own_value {
                    emit(own_key, own_value)?;
                }
            }
            // An own write of the same key replaces this row, or removes it.
            if let Some((own_key, own_value)) = overlay.next_if(|(k, _)| *k == key) {
                return match own_value {
                    Some(own_value) => emit(own_key, own_value),
                    None => budget.work(1).map_err(E::from),
                };
            }
            emit(key, value)
        };
        match self.snapshot {
            Some(snap) => {
                for guard in self.engine.snapshot_prefix_iter(&snap, part, prefix)? {
                    let (key, value) = guard.into_inner().map_err(StorageError::from)?;
                    stored_row(&key, &value)?;
                }
            }
            None => {
                for guard in self.engine.prefix_scan(part, prefix)? {
                    let (key, value) = guard.into_inner().map_err(StorageError::from)?;
                    stored_row(&key, &value)?;
                }
            }
        }
        for (own_key, own_value) in overlay {
            if let Some(own_value) = own_value {
                emit(own_key, own_value)?;
            }
        }
        Ok(())
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

        // Paged rows are not conflict-tracked by default. This is the exact
        // case the FOR UPDATE design exists for: a long cursor pinning every
        // observed page into the conflict set would make large read-mostly
        // transactions unable to commit next to any writer. FOR UPDATE opts
        // the rows the caller will update back in, via `ensure_occ_scope`.
        let last_key = rows.last().map(|(k, _)| k.clone());
        Ok(PagedScan {
            rows,
            last_key,
            exhausted,
        })
    }

    /// At most `limit` rows of `part` with keys in `[start, end)`, in key
    /// order, strictly after `start_after` when given, with the same paging,
    /// snapshot and tracking contract as [`Self::prefix_scan_paged`]: an
    /// ordered index read for the entries below a bound, which a prefix
    /// cannot express.
    pub fn range_scan_paged(
        &mut self,
        part: Partition,
        start: &[u8],
        end: &[u8],
        start_after: Option<&[u8]>,
        limit: usize,
    ) -> StorageResult<PagedScan> {
        let from = match start_after {
            Some(after) if after >= start => {
                let mut s = after.to_vec();
                s.push(0);
                s
            }
            _ => start.to_vec(),
        };
        let mut rows: Vec<KvPair> = Vec::with_capacity(limit);
        let mut exhausted = true;
        if from.as_slice() < end {
            // The engine's range bound is inclusive; `end` is not.
            for guard in self.base_range_seekable(part, &from, end)? {
                let (key, value) = guard.into_inner()?;
                if key.as_ref() >= end {
                    break;
                }
                if rows.len() == limit {
                    exhausted = false;
                    break;
                }
                rows.push((key.to_vec(), value.to_vec()));
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

/// Whether `claim` is decided before its commit takes a timestamp rather
/// than after.
///
/// A unique value held by a stored node a build has not reached is found by
/// reading those nodes, which can take long. It need not wait for the
/// timestamp: a node that takes the value after the read writes the value's
/// entry key, as the claimant does, and one of the two commits is refused
/// there; a writer older than the index is refused by the schema revision
/// the index was published with.
fn decided_before_admission(claim: &coordinode_core::txn::invariant::Claim) -> bool {
    matches!(
        claim.predicate,
        coordinode_core::txn::invariant::ClaimPredicate::UniqueHolder {
            uncovered: Some(_),
            ..
        }
    )
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

/// `stored` rows overlaid with `overlay` rows, both in key order, into one
/// list in key order: an overlay row replaces the stored row of its key.
/// Bytes a row holds while a budgeted scan hands it on: its key and value.
fn row_bytes(key: &[u8], value: &[u8]) -> u64 {
    // Both are slices of memory that exists, so their sum fits in u64.
    (key.len() + value.len()) as u64
}

fn merge_overlay(stored: Vec<KvPair>, overlay: Vec<(Vec<u8>, Option<Vec<u8>>)>) -> Vec<KvPair> {
    if overlay.is_empty() {
        return stored;
    }
    let mut out = Vec::with_capacity(stored.len() + overlay.len());
    let mut stored = stored.into_iter().peekable();
    for (key, value) in overlay {
        while let Some(next) = stored.next_if(|(k, _)| *k < key) {
            out.push(next);
        }
        // The stored row of the same key, if any, is shadowed: replaced by
        // the buffered value, or removed by the buffered tombstone.
        stored.next_if(|(k, _)| *k == key);
        if let Some(value) = value {
            out.push((key, value));
        }
    }
    out.extend(stored);
    out
}

mod derived;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
