//! Batched proposal submission for concurrent `w:majority` writers.
//!
//! When multiple writers submit proposals concurrently, each individual
//! `client_write()` call creates a separate Raft log entry and round-trip.
//! `WaitForMajorityService` coalesces proposals from multiple writers into
//! batched Raft entries, reducing N round-trips to ~1.
//!
//! ## Design
//!
//! Adapted from MongoDB's `WaitForMajorityService`
//! (`wait_for_majority_service.cpp:162-294`), but applied at the proposal
//! coalescing level. MongoDB separates local write from replication wait;
//! openraft bundles propose+commit+apply in `client_write()`, so we batch
//! at the entry level instead.
//!
//! ```text
//! Writer 1 ──┐                       ┌── oneshot::send(Ok)
//! Writer 2 ──┼── mpsc ──→ drain ──→  │── oneshot::send(Ok)
//! Writer 3 ──┘            task       └── oneshot::send(Ok)
//!                          │
//!                   Request::batch([p1, p2, p3])
//!                          │
//!                   raft.client_write(batch)
//!                          │
//!                   single Raft round-trip
//! ```
//!
//! ## Batch window
//!
//! The drain task collects proposals using a two-phase strategy:
//! 1. **Wait** for the first proposal (blocks until work arrives)
//! 2. **Linger** for up to `linger` duration, collecting additional proposals
//!
//! This balances latency (single writer pays at most `linger` delay) with
//! throughput (concurrent writers are coalesced into fewer entries).

use std::sync::Arc;
use std::time::Duration;

use coordinode_core::txn::proposal::{ProposalError, RaftProposal};
use tokio::sync::{mpsc, oneshot};

use crate::proposal::RateLimiter;
use crate::storage::{CoordinodeStateMachine, Request, TypeConfig};

/// Type alias for the openraft Raft instance with our config + state machine.
type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// Configuration for proposal batch coalescing.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum proposals per batch entry.
    ///
    /// Larger batches reduce Raft round-trips but increase per-entry size.
    /// Default: 64 (matches typical concurrent writer count).
    pub max_batch_size: usize,

    /// Maximum time to wait for additional proposals after the first arrives.
    ///
    /// Lower values reduce latency for single writers; higher values improve
    /// throughput under concurrency. Default: 1ms.
    pub linger: Duration,

    /// Channel capacity for incoming proposals.
    ///
    /// When full, `submit()` awaits backpressure. Default: 1024.
    pub channel_capacity: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            linger: Duration::from_millis(1),
            channel_capacity: 1024,
        }
    }
}

/// A proposal waiting to be batched and submitted.
struct BatchEntry {
    proposal: RaftProposal,
    response_tx: oneshot::Sender<Result<(), ProposalError>>,
}

/// Batched proposal service for concurrent `w:majority` writers.
///
/// Coalesces multiple proposals into single Raft log entries, reducing
/// N Raft round-trips to ~1 under high concurrency. Transparent to
/// callers: each writer gets an individual `Result` via oneshot channel.
///
/// ## Lifecycle
///
/// Created via [`WaitForMajorityService::spawn`]. The background drain
/// task runs until the service is dropped (channel closes) or
/// [`shutdown`](WaitForMajorityService::shutdown) is called.
pub struct WaitForMajorityService {
    tx: mpsc::Sender<BatchEntry>,
    handle: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for WaitForMajorityService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaitForMajorityService")
            .field("active", &!self.tx.is_closed())
            .finish()
    }
}

impl WaitForMajorityService {
    /// Spawn the batching service with a background drain task.
    ///
    /// The drain task runs on the tokio runtime, collecting proposals
    /// from the channel and submitting batched Raft entries.
    pub fn spawn(raft: Arc<RaftInstance>, rate_limiter: RateLimiter, config: BatchConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.channel_capacity);

        let handle = tokio::spawn(drain_loop(raft, rate_limiter, rx, config));

        Self { tx, handle }
    }

    /// Spawn with default configuration.
    pub fn spawn_default(raft: Arc<RaftInstance>, rate_limiter: RateLimiter) -> Self {
        Self::spawn(raft, rate_limiter, BatchConfig::default())
    }

    /// Submit a proposal for batched replication.
    ///
    /// Returns after the proposal is committed and applied by the Raft
    /// state machine (as part of a batched entry). The proposal may be
    /// coalesced with other concurrent submissions.
    ///
    /// ## Errors
    ///
    /// - [`ProposalError::ShuttingDown`] if the service is shutting down
    /// - [`ProposalError::NotLeader`] if this node lost leadership
    /// - [`ProposalError::Raft`] on consensus failure
    /// - [`ProposalError::Storage`] on state machine apply failure
    pub async fn submit(&self, proposal: RaftProposal) -> Result<(), ProposalError> {
        let (response_tx, response_rx) = oneshot::channel();

        self.tx
            .send(BatchEntry {
                proposal,
                response_tx,
            })
            .await
            .map_err(|_| ProposalError::ShuttingDown)?;

        response_rx.await.map_err(|_| ProposalError::ShuttingDown)?
    }

    /// Gracefully shut down the service.
    ///
    /// Closes the channel (no new submissions accepted), waits for the
    /// drain task to finish processing any in-flight batch.
    pub async fn shutdown(self) {
        // Drop sender to signal drain loop to exit
        drop(self.tx);
        // Wait for drain task to complete
        let _ = self.handle.await;
    }

    /// Check if the service is still accepting proposals.
    pub fn is_active(&self) -> bool {
        !self.tx.is_closed()
    }
}

/// Background drain loop: collect proposals, batch, submit to Raft.
///
/// Runs until the channel is closed (all senders dropped) or the
/// Raft instance is shut down.
async fn drain_loop(
    raft: Arc<RaftInstance>,
    rate_limiter: RateLimiter,
    mut rx: mpsc::Receiver<BatchEntry>,
    config: BatchConfig,
) {
    loop {
        // Phase 1: Wait for the first proposal (blocks until work arrives).
        let first = match rx.recv().await {
            Some(entry) => entry,
            None => {
                // Channel closed — all senders dropped, exit cleanly
                tracing::debug!("wait_majority drain loop: channel closed, exiting");
                return;
            }
        };

        // Phase 2: Collect additional proposals within the linger window.
        // Non-blocking drain first, then linger for stragglers.
        let mut batch = Vec::with_capacity(config.max_batch_size);
        batch.push(first);

        // Drain any already-queued proposals (non-blocking)
        while batch.len() < config.max_batch_size {
            match rx.try_recv() {
                Ok(entry) => batch.push(entry),
                Err(_) => break,
            }
        }

        // If batch isn't full yet, wait briefly for more proposals
        if batch.len() < config.max_batch_size {
            let deadline = tokio::time::Instant::now() + config.linger;
            while batch.len() < config.max_batch_size {
                match tokio::time::timeout_at(deadline, rx.recv()).await {
                    Ok(Some(entry)) => batch.push(entry),
                    _ => break, // timeout or channel closed
                }
            }
        }

        // Phase 3: Split bypass and normal proposals.
        //
        // bypass_rate_limiter proposals (delta/membership) are latency-sensitive
        // and must not be delayed by rate limiter backpressure. They get their
        // own batch submitted without acquiring a rate limiter permit.
        let (bypass, normal): (Vec<_>, Vec<_>) = batch
            .into_iter()
            .partition(|e| e.proposal.bypass_rate_limiter);

        // Submit bypass batch (if any) — no rate limiter.
        if !bypass.is_empty() {
            submit_batch(&raft, bypass, "bypass").await;
        }

        // Submit normal batch (if any) — with rate limiter.
        if !normal.is_empty() {
            let _permit = match rate_limiter.acquire(0).await {
                Ok(permit) => permit,
                Err(e) => {
                    // Rate limiter shut down — notify all waiters
                    for entry in normal {
                        let _ = entry.response_tx.send(Err(e.clone()));
                    }
                    return;
                }
            };
            submit_batch(&raft, normal, "normal").await;
        }
    }
}

/// Base timeout for batch Raft submission (first attempt).
/// Doubles on each retry: 4s → 8s → 16s.
/// Matches `RaftProposalPipeline::propose_async` constants.
const BATCH_BASE_TIMEOUT: Duration = Duration::from_secs(4);

/// Maximum retry attempts before returning error to all callers.
const BATCH_MAX_RETRIES: u32 = 3;

/// Submit a batch of proposals as a single Raft entry with retry.
///
/// Retries on timeout with exponential backoff (4s, 8s, 16s).
/// ForwardToLeader and fatal Raft errors are NOT retried — callers
/// get the error immediately.
///
/// Matches the retry strategy of `RaftProposalPipeline::propose_async`
/// but applied at the batch level: one retry loop for N proposals.
async fn submit_batch(raft: &RaftInstance, batch: Vec<BatchEntry>, kind: &str) {
    let batch_size = batch.len();
    let proposals: Vec<RaftProposal> = batch.iter().map(|e| e.proposal.clone()).collect();

    metrics::counter!("coordinode_raft_batch_proposals_total", "kind" => kind.to_owned())
        .increment(batch_size as u64);
    metrics::histogram!("coordinode_raft_batch_size", "kind" => kind.to_owned())
        .record(batch_size as f64);

    let start = std::time::Instant::now();

    for attempt in 0..BATCH_MAX_RETRIES {
        let timeout = BATCH_BASE_TIMEOUT * (1 << attempt);
        let request = Request::batch(proposals.clone());

        let result = tokio::time::timeout(timeout, raft.client_write(request)).await;

        match result {
            Ok(Ok(response)) => {
                // Success — notify all callers
                let elapsed = start.elapsed().as_secs_f64();
                tracing::debug!(
                    batch_size,
                    kind,
                    attempt,
                    mutations = response.data.mutations_applied,
                    log_id = ?response.log_id,
                    elapsed_s = elapsed,
                    "batched proposal committed"
                );
                metrics::counter!(
                    "coordinode_raft_batch_entries_total",
                    "status" => "ok",
                    "kind" => kind.to_owned()
                )
                .increment(1);
                metrics::histogram!("coordinode_raft_batch_duration_seconds").record(elapsed);
                for entry in batch {
                    let _ = entry.response_tx.send(Ok(()));
                }
                return;
            }
            Ok(Err(raft_err)) => {
                // Raft error — check if retryable
                let proposal_err = convert_raft_error(&raft_err);
                match &proposal_err {
                    ProposalError::NotLeader { .. } => {
                        // Not retryable — forward immediately
                        tracing::warn!(
                            batch_size,
                            kind,
                            error = %raft_err,
                            "batched proposal: not leader"
                        );
                        metrics::counter!(
                            "coordinode_raft_batch_entries_total",
                            "status" => "not_leader",
                            "kind" => kind.to_owned()
                        )
                        .increment(1);
                        for entry in batch {
                            let _ = entry.response_tx.send(Err(proposal_err.clone()));
                        }
                        return;
                    }
                    _ => {
                        // Fatal Raft error — not retryable
                        tracing::warn!(
                            batch_size,
                            kind,
                            error = %raft_err,
                            "batched proposal: fatal raft error"
                        );
                        metrics::counter!(
                            "coordinode_raft_batch_entries_total",
                            "status" => "error",
                            "kind" => kind.to_owned()
                        )
                        .increment(1);
                        for entry in batch {
                            let _ = entry.response_tx.send(Err(proposal_err.clone()));
                        }
                        return;
                    }
                }
            }
            Err(_elapsed) => {
                // Timeout — retry with exponential backoff
                metrics::counter!("coordinode_raft_batch_retries_total").increment(1);
                tracing::warn!(
                    batch_size,
                    kind,
                    attempt = attempt + 1,
                    timeout_ms = timeout.as_millis() as u64,
                    "batched proposal timed out, retrying"
                );
                continue;
            }
        }
    }

    // All retries exhausted
    let elapsed = start.elapsed().as_secs_f64();
    tracing::error!(
        batch_size,
        kind,
        retries = BATCH_MAX_RETRIES,
        elapsed_s = elapsed,
        "batched proposal: all retries exhausted"
    );
    metrics::counter!(
        "coordinode_raft_batch_entries_total",
        "status" => "timeout",
        "kind" => kind.to_owned()
    )
    .increment(1);
    metrics::histogram!("coordinode_raft_batch_duration_seconds").record(elapsed);
    let err = ProposalError::Timeout {
        retries: BATCH_MAX_RETRIES,
    };
    for entry in batch {
        let _ = entry.response_tx.send(Err(err.clone()));
    }
}

/// Convert an openraft error to a ProposalError.
fn convert_raft_error(
    err: &openraft::error::RaftError<TypeConfig, openraft::error::ClientWriteError<TypeConfig>>,
) -> ProposalError {
    match err {
        openraft::error::RaftError::APIError(
            openraft::error::ClientWriteError::ForwardToLeader(fwd),
        ) => ProposalError::NotLeader {
            leader_id: fwd.leader_id,
        },
        other => ProposalError::Raft(other.to_string()),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
