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
    Mutation, PartitionId, ProposalError, ProposalOutcome, ProposalPipeline, RaftProposal,
};
use coordinode_storage::engine::config::FlushPolicy;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::StorageError;

use crate::storage::{CoordinodeStateMachine, Request, TypeConfig};

/// Map a [`StorageError`] from the engine into a [`ProposalError`]
/// preserving the typed `CapacityExhausted` variant. Other storage
/// errors collapse into [`ProposalError::Storage`] (stringified) for
/// now — only the capacity case needs structured propagation today,
/// since it carries operator-actionable metadata (endpoint id +
/// limits) that the gRPC handler maps to `Status::resource_exhausted`.
fn storage_to_proposal_err(e: StorageError) -> ProposalError {
    match e {
        StorageError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        } => ProposalError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        },
        other => ProposalError::Storage(other.to_string()),
    }
}

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
        PartitionId::VectorF32 => Partition::VectorF32,
        PartitionId::Registry => Partition::Registry,
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
        Partition::VectorF32 => PartitionId::VectorF32,
        Partition::Registry => PartitionId::Registry,
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
                    .map_err(storage_to_proposal_err)?;
            }
            Mutation::Delete { partition, key } => {
                self.engine
                    .delete(to_partition(*partition), key)
                    .map_err(storage_to_proposal_err)?;
            }
            Mutation::Merge {
                partition,
                key,
                operand,
            } => {
                self.engine
                    .merge(to_partition(*partition), key, operand)
                    .map_err(storage_to_proposal_err)?;
            }
        }
        Ok(())
    }
}

impl ProposalPipeline for LocalProposalPipeline<'_> {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        for mutation in &proposal.mutations {
            self.apply_mutation(mutation)?;
        }

        tracing::debug!(
            proposal_id = %proposal.id,
            mutations = proposal.mutation_count(),
            commit_ts = proposal.commit_ts.as_raw(),
            "local proposal applied"
        );

        // Local pipeline has no Raft log — no cluster-wide commit index.
        Ok(ProposalOutcome::local())
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
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
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
                        .map_err(storage_to_proposal_err)?;
                }
                Mutation::Delete { partition, key } => {
                    self.engine
                        .delete(to_partition(*partition), key)
                        .map_err(storage_to_proposal_err)?;
                }
                Mutation::Merge {
                    partition,
                    key,
                    operand,
                } => {
                    self.engine
                        .merge(to_partition(*partition), key, operand)
                        .map_err(storage_to_proposal_err)?;
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

        // Embedded/standalone WAL pipeline — no Raft commit index.
        Ok(ProposalOutcome::local())
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
    async fn propose_async(
        &self,
        proposal: &RaftProposal,
    ) -> Result<ProposalOutcome, ProposalError> {
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
                    let committed_index = response.log_id.index;
                    tracing::debug!(
                        proposal_id = %proposal.id,
                        mutations = response.data.mutations_applied,
                        log_id = ?response.log_id,
                        committed_index,
                        "raft proposal committed"
                    );
                    // The committed log index is this write's faithful causal
                    // `operationTime` — surfaced so the gRPC layer returns it
                    // instead of sampling the node's current applied index.
                    return Ok(ProposalOutcome::replicated(committed_index));
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
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
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
    ) -> Result<ProposalOutcome, ProposalError> {
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
mod tests;
