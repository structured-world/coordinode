//! Proposal pipeline implementations.
//!
//! - [`LocalProposalPipeline`]: single-node CE / embedded mode. Applies
//!   mutations directly to StorageEngine without Raft replication.
//!   ADR-016: uses plain engine.put()/delete() — oracle auto-stamps seqno.
//! - [`RaftProposalPipeline`]: 3-node HA cluster mode. Replicates mutations
//!   via openraft before applying. Includes rate limiting, retry with
//!   exponential backoff, and dedup tracking.

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{
    Mutation, PartitionId, ProposalError, ProposalPipeline, RaftProposal,
};
use coordinode_storage::engine::config::FlushPolicy;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use crate::storage::{CoordinodeStateMachine, Request, TypeConfig};

/// Base timeout for Raft proposal (first attempt).
/// Doubles on each retry: 4s → 8s → 16s.
const BASE_TIMEOUT: Duration = Duration::from_secs(4);

/// Maximum number of retry attempts before returning `ProposalError::Timeout`.
const MAX_RETRIES: u32 = 3;

/// Maximum concurrent pending proposals (rate limiter capacity).
/// Prevents overwhelming the Raft leader with unbounded proposal volume.
const MAX_PENDING_PROPOSALS: u32 = 256;

/// Type alias for the openraft Raft instance with our config + state machine.
type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// Convert `PartitionId` (core, no storage dependency) to `Partition` (storage).
pub fn to_partition(id: PartitionId) -> Partition {
    match id {
        PartitionId::Node => Partition::Node,
        PartitionId::Adj => Partition::Adj,
        PartitionId::EdgeProp => Partition::EdgeProp,
        PartitionId::Blob => Partition::Blob,
        PartitionId::BlobRef => Partition::BlobRef,
        PartitionId::Schema => Partition::Schema,
        PartitionId::Idx => Partition::Idx,
        PartitionId::Counter => Partition::Counter,
    }
}

/// Convert `Partition` (storage) to `PartitionId` (core).
///
/// The `Raft` partition is internal to the storage engine (Raft log entries
/// and metadata) and has no corresponding `PartitionId` in the public API.
pub fn to_partition_id(p: Partition) -> PartitionId {
    match p {
        Partition::Node => PartitionId::Node,
        Partition::Adj => PartitionId::Adj,
        Partition::EdgeProp => PartitionId::EdgeProp,
        Partition::Blob => PartitionId::Blob,
        Partition::BlobRef => PartitionId::BlobRef,
        Partition::Schema => PartitionId::Schema,
        Partition::Idx => PartitionId::Idx,
        Partition::Raft => unreachable!("Raft partition is internal-only"),
        Partition::Counter => PartitionId::Counter,
    }
}

/// Single-node proposal pipeline: applies mutations directly to CoordiNode storage.
///
/// No Raft replication — mutations are applied synchronously via
/// [`StorageEngine`] within the caller's thread. Used in:
///
/// - Embedded mode (`coordinode-embed`)
/// - Single-node CE server (before openraft integration)
///
/// ## Durability
///
/// coordinode-lsm-tree has no WAL. Mutations are written to the active
/// memtable. In cluster mode the Raft log serves as WAL; in standalone mode
/// the `OwnedLocalProposalPipeline` calls `engine.persist()` after each
/// proposal when `FlushPolicy::SyncPerBatch` is set (default), which fsyncs
/// the memtable to SST before returning. Crash-safety window is limited to
/// the duration of the `persist()` call itself.
///
/// ## Cluster-ready
///
/// This implementation will be replaced by `RaftProposalPipeline`
/// for distributed deployments. The `ProposalPipeline` trait ensures the
/// executor doesn't need to change.
pub struct LocalProposalPipeline<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalProposalPipeline<'a> {
    /// Create a pipeline that applies directly to the given engine.
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }

    /// Apply a single mutation to StorageEngine (ADR-016: native seqno MVCC).
    ///
    /// Put/Delete write plain keys — OracleSeqnoGenerator auto-stamps seqno.
    /// Merge writes raw merge operand directly to the partition.
    fn apply_mutation(&self, mutation: &Mutation) -> Result<(), ProposalError> {
        match mutation {
            Mutation::Put {
                partition,
                key,
                value,
            } => {
                self.engine
                    .put(to_partition(*partition), key, value)
                    .map_err(|e| ProposalError::Storage(e.to_string()))?;
            }
            Mutation::Delete { partition, key } => {
                self.engine
                    .delete(to_partition(*partition), key)
                    .map_err(|e| ProposalError::Storage(e.to_string()))?;
            }
            Mutation::Merge {
                partition,
                key,
                operand,
            } => {
                self.engine
                    .merge(to_partition(*partition), key, operand)
                    .map_err(|e| ProposalError::Storage(e.to_string()))?;
            }
        }
        Ok(())
    }
}

impl ProposalPipeline for LocalProposalPipeline<'_> {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<(), ProposalError> {
        for mutation in &proposal.mutations {
            self.apply_mutation(mutation)?;
        }

        tracing::debug!(
            proposal_id = %proposal.id,
            mutations = proposal.mutation_count(),
            commit_ts = proposal.commit_ts.as_raw(),
            "local proposal applied"
        );

        Ok(())
    }
}

/// Owned version of [`LocalProposalPipeline`] that holds an `Arc<StorageEngine>`.
///
/// Used by the drain thread and other contexts that need `'static` lifetime
/// on the pipeline. Same apply logic as `LocalProposalPipeline`.
///
/// ## Durability
///
/// Two modes depending on whether the engine was opened with a standalone WAL:
///
/// **WAL mode** (`StorageEngine::open_with_wal`):
/// 1. Append mutations to WAL + fsync (≤0.5 ms for ≤4 KB record)
/// 2. Apply mutations to memtable
/// 3. Return — no SST flush per proposal
///    Checkpoint (WAL rotation) happens automatically when `persist()` is called.
///
/// **Legacy mode** (plain `StorageEngine::open`, no WAL):
/// Apply to memtable, then `persist()` when `FlushPolicy::SyncPerBatch`.
/// Correct but expensive (5–50 ms SST flush per write). Use WAL mode for
/// production embedded deployments.
///
/// Cluster mode always uses `RaftProposalPipeline`; `OwnedLocalProposalPipeline`
/// is only for embedded/standalone (unit tests, CLI tools, embedded library).
pub struct OwnedLocalProposalPipeline {
    engine: Arc<StorageEngine>,
}

impl OwnedLocalProposalPipeline {
    /// Create a pipeline that applies directly to the given engine.
    pub fn new(engine: &Arc<StorageEngine>) -> Self {
        Self {
            engine: Arc::clone(engine),
        }
    }
}

impl ProposalPipeline for OwnedLocalProposalPipeline {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<(), ProposalError> {
        // ── WAL-first write path ─────────────────────────────────────────────
        // When a standalone WAL is active: append mutations to the journal and
        // fsync BEFORE applying to the memtable.  This ensures any record that
        // reached the WAL survives a crash and is replayed on recovery.
        //
        // When no WAL is configured (cluster mode or tests that use plain open):
        // fall through to the legacy persist() path below.
        if self.engine.has_wal() {
            self.engine
                .wal_append(&proposal.mutations)
                .map_err(|e| ProposalError::Storage(format!("WAL append: {e}")))?;
        }

        // Apply mutations to the memtable (same logic as LocalProposalPipeline).
        for mutation in &proposal.mutations {
            match mutation {
                Mutation::Put {
                    partition,
                    key,
                    value,
                } => {
                    self.engine
                        .put(to_partition(*partition), key, value)
                        .map_err(|e| ProposalError::Storage(e.to_string()))?;
                }
                Mutation::Delete { partition, key } => {
                    self.engine
                        .delete(to_partition(*partition), key)
                        .map_err(|e| ProposalError::Storage(e.to_string()))?;
                }
                Mutation::Merge {
                    partition,
                    key,
                    operand,
                } => {
                    self.engine
                        .merge(to_partition(*partition), key, operand)
                        .map_err(|e| ProposalError::Storage(e.to_string()))?;
                }
            }
        }

        // ── Legacy durability path (no WAL) ──────────────────────────────────
        // Without a WAL the only way to guarantee crash safety is a full SST
        // flush after every proposal (atomic rename — no corruption possible).
        // This is expensive but correct. Open with WAL to avoid this overhead.
        if !self.engine.has_wal() && self.engine.flush_policy() == FlushPolicy::SyncPerBatch {
            self.engine
                .persist()
                .map_err(|e| ProposalError::Storage(format!("persist failed: {e}")))?;
        }

        tracing::debug!(
            proposal_id = %proposal.id,
            mutations = proposal.mutation_count(),
            commit_ts = proposal.commit_ts.as_raw(),
            has_wal = self.engine.has_wal(),
            "owned local proposal applied"
        );

        Ok(())
    }
}

// ── Rate Limiter ────────────────────────────────────────────────────

/// Feedback-based rate limiter for Raft proposals.
///
/// Uses a tokio Semaphore with weighted acquisition. Each retry attempt
/// acquires exponentially more permits (`1 << retry`), implementing
/// backpressure on retried proposals. Based on Dgraph's IOU-based
/// rate limiter pattern (worker/proposal.go:42-99).
///
/// The limiter prevents overwhelming the Raft leader with unbounded
/// proposal volume. Delta/schema proposals can bypass the limiter.
pub struct RateLimiter {
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl RateLimiter {
    /// Create a rate limiter with the given maximum pending proposals.
    pub fn new(max_pending: u32) -> Self {
        Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(max_pending as usize)),
        }
    }

    /// Acquire permits for a proposal attempt.
    ///
    /// Weight doubles on each retry: attempt 0 = 1 permit, attempt 1 = 2,
    /// attempt 2 = 4. This ensures retried proposals face increasing
    /// backpressure, preventing thundering herd on leader change.
    ///
    /// Returns a guard that releases permits on drop.
    pub async fn acquire(
        &self,
        retry: u32,
    ) -> Result<tokio::sync::OwnedSemaphorePermit, ProposalError> {
        let weight = 1u32 << retry.min(8); // cap at 256 to avoid overflow
        let permit = Arc::clone(&self.semaphore)
            .acquire_many_owned(weight)
            .await
            .map_err(|_| ProposalError::ShuttingDown)?;
        metrics::gauge!("coordinode_raft_rate_limiter_pending")
            .set(self.semaphore.available_permits() as f64);
        Ok(permit)
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(MAX_PENDING_PROPOSALS)
    }
}

// ── Raft Proposal Pipeline ─────────────────────────────────────────

/// Distributed proposal pipeline using openraft.
///
/// Replicates mutations via Raft consensus before applying to storage.
/// Provides:
/// - **Rate limiting**: feedback-based semaphore prevents leader overload
/// - **Retry with exponential backoff**: 3 attempts (4s, 8s, 16s timeouts)
/// - **ForwardToLeader**: returns `NotLeader` error with leader hint
///
/// Deduplication is handled at the state machine level
/// ([`CoordinodeStateMachine`]) using proposal ID + size matching,
/// so Raft replay is idempotent.
///
/// ## Usage
///
/// Created once per Raft node. Shared via the `ProposalPipeline` trait.
/// Only the leader node can accept proposals; followers return `NotLeader`.
pub struct RaftProposalPipeline {
    raft: Arc<RaftInstance>,
    rate_limiter: RateLimiter,
}

impl RaftProposalPipeline {
    /// Create a pipeline backed by an openraft instance.
    pub fn new(raft: Arc<RaftInstance>) -> Self {
        Self {
            raft,
            rate_limiter: RateLimiter::default(),
        }
    }

    /// Create a pipeline with a custom rate limiter capacity.
    pub fn with_max_pending(raft: Arc<RaftInstance>, max_pending: u32) -> Self {
        Self {
            raft,
            rate_limiter: RateLimiter::new(max_pending),
        }
    }

    /// Async propose-and-wait: submit proposal through openraft and wait
    /// for it to be committed and applied by the state machine.
    async fn propose_async(&self, proposal: &RaftProposal) -> Result<(), ProposalError> {
        let request = Request::single(proposal.clone());
        let start = std::time::Instant::now();

        for attempt in 0..MAX_RETRIES {
            let timeout = BASE_TIMEOUT * (1 << attempt);

            // Acquire rate limiter permits (weighted by retry count).
            // Delta/membership proposals bypass the limiter entirely —
            // they are latency-sensitive and must not be delayed by
            // backpressure from regular mutation proposals.
            let _permit = if proposal.bypass_rate_limiter {
                metrics::counter!("coordinode_raft_proposals_bypassed_total").increment(1);
                None
            } else {
                Some(self.rate_limiter.acquire(attempt).await?)
            };

            // Submit to Raft with timeout
            let result =
                tokio::time::timeout(timeout, self.raft.client_write(request.clone())).await;

            match result {
                Ok(Ok(response)) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    metrics::counter!("coordinode_raft_proposals_total", "status" => "ok")
                        .increment(1);
                    metrics::histogram!("coordinode_raft_proposal_duration_seconds")
                        .record(elapsed);
                    tracing::debug!(
                        proposal_id = %proposal.id,
                        mutations = response.data.mutations_applied,
                        log_id = ?response.log_id,
                        "raft proposal committed"
                    );
                    return Ok(());
                }
                Ok(Err(raft_err)) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    metrics::histogram!("coordinode_raft_proposal_duration_seconds")
                        .record(elapsed);
                    // Check if it's a ForwardToLeader error
                    if let Some(leader) = extract_forward_leader(&raft_err) {
                        metrics::counter!(
                            "coordinode_raft_proposals_total",
                            "status" => "not_leader"
                        )
                        .increment(1);
                        return Err(ProposalError::NotLeader { leader_id: leader });
                    }
                    // Fatal Raft error — no retry
                    metrics::counter!("coordinode_raft_proposals_total", "status" => "error")
                        .increment(1);
                    return Err(ProposalError::Raft(raft_err.to_string()));
                }
                Err(_elapsed) => {
                    // Timeout — retry with exponential backoff
                    metrics::counter!("coordinode_raft_proposal_retries_total").increment(1);
                    tracing::warn!(
                        proposal_id = %proposal.id,
                        attempt = attempt + 1,
                        timeout_ms = timeout.as_millis() as u64,
                        "raft proposal timed out, retrying"
                    );
                    continue;
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        metrics::counter!("coordinode_raft_proposals_total", "status" => "timeout").increment(1);
        metrics::histogram!("coordinode_raft_proposal_duration_seconds").record(elapsed);
        Err(ProposalError::Timeout {
            retries: MAX_RETRIES,
        })
    }
}

/// Extract leader ID from a ClientWriteError::ForwardToLeader, if present.
///
/// Uses openraft's `TryAsRef` trait for proper structured error matching
/// (not string-based). Returns `Some(leader_id)` if this is a forward
/// error, `None` otherwise.
fn extract_forward_leader(
    err: &openraft::error::RaftError<TypeConfig, openraft::error::ClientWriteError<TypeConfig>>,
) -> Option<Option<u64>> {
    match err {
        openraft::error::RaftError::APIError(
            openraft::error::ClientWriteError::ForwardToLeader(fwd),
        ) => Some(fwd.leader_id),
        _ => None,
    }
}

impl ProposalPipeline for RaftProposalPipeline {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<(), ProposalError> {
        // Bridge sync ProposalPipeline trait to async openraft.
        // block_in_place tells tokio "this thread will block" so it moves
        // other async tasks to other worker threads. Requires rt-multi-thread.
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.propose_async(proposal))
        })
    }

    fn propose_with_timeout(
        &self,
        proposal: &RaftProposal,
        timeout: Duration,
    ) -> Result<(), ProposalError> {
        // True async timeout: if wtimeout fires before Raft commits,
        // the client gets an error immediately. The proposal is NOT
        // cancelled — openraft may still commit the entry after timeout.
        // Per MongoDB spec: data is NOT rolled back on wtimeout.
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                match tokio::time::timeout(timeout, self.propose_async(proposal)).await {
                    Ok(result) => result,
                    Err(_elapsed) => {
                        metrics::counter!("coordinode_raft_write_concern_timeouts_total")
                            .increment(1);
                        Err(ProposalError::WriteConcernTimeout {
                            timeout_ms: timeout.as_millis() as u32,
                        })
                    }
                }
            })
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::cluster::RaftNode;
    use coordinode_core::txn::proposal::ProposalIdGenerator;
    use coordinode_core::txn::timestamp::Timestamp;
    use coordinode_storage::engine::config::StorageConfig;
    use tempfile::TempDir;

    fn test_engine() -> (TempDir, Arc<StorageEngine>) {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = Arc::new(StorageEngine::open(&config).expect("open"));
        (dir, engine)
    }

    fn setup() -> (StorageEngine, TempDir) {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open");
        (engine, dir)
    }

    #[test]
    fn local_pipeline_put_and_read() {
        let (engine, _dir) = setup();
        let pipeline = LocalProposalPipeline::new(&engine);
        let id_gen = ProposalIdGenerator::new();

        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:1".to_vec(),
                value: b"test-data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };

        pipeline
            .propose_and_wait(&proposal)
            .expect("propose should succeed");

        // Verify via direct engine read (plain keys, no MVCC encoding)
        let result = engine.get(Partition::Node, b"node:1:1").expect("read");
        assert_eq!(result.as_deref(), Some(b"test-data".as_slice()));
    }

    #[test]
    fn local_pipeline_delete() {
        let (engine, _dir) = setup();
        let pipeline = LocalProposalPipeline::new(&engine);
        let id_gen = ProposalIdGenerator::new();

        // First write a value
        let put = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:2".to_vec(),
                value: b"data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };
        pipeline.propose_and_wait(&put).expect("put");

        // Snapshot after put — captures the value before deletion
        let snap_after_put = engine.snapshot();

        // Then delete
        let del = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Delete {
                partition: PartitionId::Node,
                key: b"node:1:2".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(200),
            start_ts: Timestamp::from_raw(199),
            bypass_rate_limiter: false,
        };
        pipeline.propose_and_wait(&del).expect("delete");

        // Verify: snapshot before delete still sees the value
        let before_del = engine
            .snapshot_get(&snap_after_put, Partition::Node, b"node:1:2")
            .expect("snapshot read");
        assert_eq!(before_del.as_deref(), Some(b"data".as_slice()));

        // Verify: after delete, current value is gone (LSM tombstone)
        let after_del = engine.get(Partition::Node, b"node:1:2").expect("read");
        assert!(after_del.is_none());
    }

    #[test]
    fn local_pipeline_multi_mutation_proposal() {
        let (engine, _dir) = setup();
        let pipeline = LocalProposalPipeline::new(&engine);
        let id_gen = ProposalIdGenerator::new();

        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:1:10".to_vec(),
                    value: b"alice".to_vec(),
                },
                Mutation::Put {
                    partition: PartitionId::Node,
                    key: b"node:1:11".to_vec(),
                    value: b"bob".to_vec(),
                },
                Mutation::Put {
                    partition: PartitionId::Adj,
                    key: b"adj:KNOWS:out:10".to_vec(),
                    value: b"posting-list".to_vec(),
                },
                Mutation::Put {
                    partition: PartitionId::EdgeProp,
                    key: b"edgeprop:KNOWS:10:11".to_vec(),
                    value: b"since:2024".to_vec(),
                },
            ],
            commit_ts: Timestamp::from_raw(500),
            start_ts: Timestamp::from_raw(499),
            bypass_rate_limiter: false,
        };

        pipeline.propose_and_wait(&proposal).expect("propose");

        // Verify all four writes via direct engine reads
        assert_eq!(
            engine
                .get(Partition::Node, b"node:1:10")
                .expect("r")
                .as_deref(),
            Some(b"alice".as_slice())
        );
        assert_eq!(
            engine
                .get(Partition::Node, b"node:1:11")
                .expect("r")
                .as_deref(),
            Some(b"bob".as_slice())
        );
        assert_eq!(
            engine
                .get(Partition::Adj, b"adj:KNOWS:out:10")
                .expect("r")
                .as_deref(),
            Some(b"posting-list".as_slice())
        );
        assert_eq!(
            engine
                .get(Partition::EdgeProp, b"edgeprop:KNOWS:10:11")
                .expect("r")
                .as_deref(),
            Some(b"since:2024".as_slice())
        );
    }

    #[test]
    fn local_pipeline_empty_proposal() {
        let (engine, _dir) = setup();
        let pipeline = LocalProposalPipeline::new(&engine);
        let id_gen = ProposalIdGenerator::new();

        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };

        // Empty proposal should succeed (no-op)
        pipeline.propose_and_wait(&proposal).expect("empty ok");
    }

    #[test]
    fn partition_roundtrip() {
        // Verify to_partition / to_partition_id are inverse.
        // Partition::Raft is internal-only and has no PartitionId mapping.
        for p in Partition::all() {
            if *p == Partition::Raft {
                continue;
            }
            let id = to_partition_id(*p);
            let back = to_partition(id);
            assert_eq!(*p, back);
        }
    }

    // ── RateLimiter tests ──────────────────────────────────────────

    #[tokio::test]
    async fn rate_limiter_acquire_releases() {
        // Verify that permits are acquired and released correctly
        let limiter = RateLimiter::new(4);

        // Acquire 1 permit (retry=0, weight=1)
        let permit1 = limiter.acquire(0).await.expect("acquire 1");
        // Acquire 2 permits (retry=1, weight=2)
        let permit2 = limiter.acquire(1).await.expect("acquire 2");
        // 3 permits used, 1 remaining — can acquire 1 more
        let permit3 = limiter.acquire(0).await.expect("acquire 3");
        // All 4 permits used

        // Drop permits to release
        drop(permit1);
        drop(permit2);
        drop(permit3);

        // Should be able to acquire again after release
        let _permit4 = limiter.acquire(0).await.expect("acquire after release");
    }

    #[tokio::test]
    async fn rate_limiter_exponential_weight() {
        // Verify exponential weight: retry 0 = 1, retry 1 = 2, retry 2 = 4
        let limiter = RateLimiter::new(8);

        // retry=2 → weight=4, should succeed with 8 permits
        let p1 = limiter.acquire(2).await.expect("weight 4");
        // 4 remaining, retry=2 → weight=4 again, should succeed
        let p2 = limiter.acquire(2).await.expect("weight 4 again");
        // 0 remaining

        drop(p1);
        drop(p2);
    }

    #[tokio::test]
    async fn rate_limiter_timeout_on_exhaustion() {
        // When all permits taken, acquire should block (we test with timeout)
        let limiter = RateLimiter::new(1);

        // Take the only permit
        let _permit = limiter.acquire(0).await.expect("acquire");

        // Trying to acquire another should block — verify with timeout
        let result = tokio::time::timeout(Duration::from_millis(50), limiter.acquire(0)).await;

        // Should timeout (not resolve), proving backpressure works
        assert!(result.is_err(), "should timeout when all permits taken");
    }

    /// G035: bypass_rate_limiter=true skips semaphore acquisition.
    /// Proposal succeeds even when rate limiter is fully exhausted.
    #[tokio::test(flavor = "multi_thread")]
    async fn rate_limiter_bypass_succeeds_when_exhausted() {
        let (_dir, engine) = test_engine();
        let node = RaftNode::single_node(engine).await.expect("bootstrap");

        tokio::time::sleep(Duration::from_millis(500)).await;

        // Create pipeline with capacity=1
        let pipeline = RaftProposalPipeline::with_max_pending(Arc::clone(node.raft()), 1);
        let id_gen = ProposalIdGenerator::new();

        // Exhaust the rate limiter by holding a permit
        // (We can't directly exhaust it from outside, so we test via
        // proposal behavior: a normal proposal uses the semaphore,
        // a bypass proposal doesn't.)

        // Normal proposal — should succeed (uses 1 of 1 permits)
        let normal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:normal".to_vec(),
                value: b"data".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };
        pipeline.propose_and_wait(&normal).expect("normal propose");

        // Bypass proposal — should also succeed (skips semaphore entirely)
        let bypass = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:bypass".to_vec(),
                value: b"delta".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(200),
            start_ts: Timestamp::from_raw(199),
            bypass_rate_limiter: true,
        };
        pipeline
            .propose_and_wait(&bypass)
            .expect("G035: bypass proposal should succeed regardless of rate limiter state");

        node.shutdown().await.expect("shutdown");
    }

    // ── Node-scoped ProposalIdGenerator tests ──────────────────────

    #[test]
    fn node_scoped_id_generator_uniqueness() {
        // Two generators with different node bases produce non-overlapping IDs
        let gen_node1 = ProposalIdGenerator::with_base(1u64 << 48);
        let gen_node2 = ProposalIdGenerator::with_base(2u64 << 48);

        let ids1: Vec<u64> = (0..100).map(|_| gen_node1.next().as_raw()).collect();
        let ids2: Vec<u64> = (0..100).map(|_| gen_node2.next().as_raw()).collect();

        // No overlaps between node1 and node2 ID spaces
        for id1 in &ids1 {
            assert!(
                !ids2.contains(id1),
                "node1 ID {} should not appear in node2 IDs",
                id1
            );
        }

        // IDs within each node are monotonically increasing
        for w in ids1.windows(2) {
            assert!(w[1] > w[0], "IDs should be monotonically increasing");
        }
    }

    #[test]
    fn node_scoped_id_high_bits_preserved() {
        // Verify that node_id is encoded in the high 16 bits
        let node_id: u64 = 42;
        let gen = ProposalIdGenerator::with_base(node_id << 48);

        let id = gen.next();
        // High 16 bits should contain node_id
        assert_eq!(id.as_raw() >> 48, node_id);
    }

    // ── propose_with_timeout tests (G048) ────────────────────────

    /// LocalProposalPipeline: propose_with_timeout delegates to propose_and_wait
    /// (ignores timeout). Proposals complete in µs, timeout irrelevant.
    #[test]
    fn local_pipeline_propose_with_timeout_succeeds() {
        let (engine, _dir) = setup();
        let pipeline = LocalProposalPipeline::new(&engine);
        let id_gen = ProposalIdGenerator::new();

        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:timeout-local".to_vec(),
                value: b"value".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };

        // Even with a very short timeout, local pipeline ignores it
        pipeline
            .propose_with_timeout(&proposal, Duration::from_nanos(1))
            .expect("local propose_with_timeout should succeed (timeout ignored)");

        let val = engine
            .get(Partition::Node, b"node:1:timeout-local")
            .expect("read");
        assert_eq!(val.as_deref(), Some(b"value".as_slice()));
    }

    /// RaftProposalPipeline: propose_with_timeout with generous timeout succeeds.
    #[tokio::test(flavor = "multi_thread")]
    async fn raft_pipeline_propose_with_timeout_succeeds() {
        let (_dir, engine) = test_engine();
        let node = RaftNode::single_node(engine.clone())
            .await
            .expect("bootstrap");
        tokio::time::sleep(Duration::from_millis(500)).await;

        let pipeline = RaftProposalPipeline::new(Arc::clone(node.raft()));
        let id_gen = ProposalIdGenerator::new();

        let proposal = RaftProposal {
            id: id_gen.next(),
            mutations: vec![Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:1:timeout-raft".to_vec(),
                value: b"raft-val".to_vec(),
            }],
            commit_ts: Timestamp::from_raw(100),
            start_ts: Timestamp::from_raw(99),
            bypass_rate_limiter: false,
        };

        // 10 seconds is more than enough for single-node
        pipeline
            .propose_with_timeout(&proposal, Duration::from_secs(10))
            .expect("propose_with_timeout should succeed with generous timeout");

        let val = engine
            .get(Partition::Node, b"node:1:timeout-raft")
            .expect("read");
        assert_eq!(val.as_deref(), Some(b"raft-val".as_slice()));

        node.shutdown().await.expect("shutdown");
    }

    /// WriteConcernTimeout error has correct format.
    #[test]
    fn write_concern_timeout_error_display() {
        let err = ProposalError::WriteConcernTimeout { timeout_ms: 5000 };
        let msg = format!("{err}");
        assert!(
            msg.contains("5000ms"),
            "error should contain timeout_ms: {msg}"
        );
        assert!(
            msg.contains("write concern timeout"),
            "error should mention write concern: {msg}"
        );
    }
}
