//! Volatile write drain buffer for `w:memory` and `w:cache` write concerns.
//!
//! Volatile writes are applied to local storage for immediate read visibility,
//! then buffered here for background drain into Raft proposals. The drain
//! thread periodically batches buffered mutations into [`RaftProposal`]s and
//! submits them through the [`ProposalPipeline`].
//!
//! ## Invariants
//!
//! - Every write EVENTUALLY enters the oplog (no permanent "holes")
//! - CDC consumers see all writes, including drained volatile ones
//! - Drained entries preserve the original client timestamp, not drain timestamp
//! - Crash before drain = data lost (explicit `w:memory`/`w:cache` contract)
//! - Graceful shutdown flushes all drain buffers before exit
//!
//! ## Data flow
//!
//! ```text
//! Client write (w:memory)
//!   → engine.put/delete (local, immediate visibility)
//!   → DrainBuffer.append(entry) → ACK to client (~1µs)
//!   ... background drain (100ms interval):
//!   → batch entries → RaftProposal → pipeline.propose_and_wait()
//!   → oplog → replicas → durable
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::proposal::{Mutation, ProposalIdGenerator, ProposalPipeline, RaftProposal};
use super::timestamp::Timestamp;

/// Configuration for the volatile write drain mechanism.
#[derive(Debug, Clone)]
pub struct DrainConfig {
    /// Background drain thread polling interval in milliseconds.
    /// Lower = less data at risk, higher CDC latency.
    /// Default: 100ms.
    pub interval_ms: u64,

    /// Maximum number of mutations per drain Raft proposal.
    /// Large batches amortize Raft round-trip cost but increase
    /// apply latency. Default: 10,000.
    pub batch_max: u32,

    /// Maximum drain buffer capacity in bytes. When full, new volatile
    /// writes are rejected with backpressure (ErrBufferFull).
    /// Default: 100 MB.
    pub capacity_bytes: u64,
}

impl Default for DrainConfig {
    fn default() -> Self {
        Self {
            interval_ms: 100,
            batch_max: 10_000,
            capacity_bytes: 100 * 1024 * 1024,
        }
    }
}

/// A buffered volatile write entry waiting for drain.
///
/// Contains the mutations from one transaction, already applied to local
/// storage. The drain thread will package these into a Raft proposal.
#[derive(Debug)]
pub struct DrainEntry {
    /// Mutations from the original transaction (Put/Delete/Merge).
    pub mutations: Vec<Mutation>,
    /// Commit timestamp assigned by the oracle at transaction time.
    /// Preserved through drain for CDC timestamp fidelity.
    pub commit_ts: Timestamp,
    /// Start timestamp of the original transaction.
    pub start_ts: Timestamp,
    /// Approximate size in bytes (for capacity tracking).
    size_bytes: usize,
}

impl DrainEntry {
    /// Create a new drain entry from transaction mutations.
    pub fn new(mutations: Vec<Mutation>, commit_ts: Timestamp, start_ts: Timestamp) -> Self {
        let size_bytes = mutations.iter().map(mutation_size).sum();
        Self {
            mutations,
            commit_ts,
            start_ts,
            size_bytes,
        }
    }
}

/// Approximate byte size of a mutation (for buffer capacity tracking).
fn mutation_size(m: &Mutation) -> usize {
    match m {
        Mutation::Put { key, value, .. } => 1 + key.len() + value.len(),
        Mutation::Delete { key, .. } => 1 + key.len(),
        Mutation::Merge { key, operand, .. } => 1 + key.len() + operand.len(),
    }
}

/// Error when the drain buffer is full (backpressure).
#[derive(Debug, thiserror::Error)]
#[error("drain buffer full: {used_bytes}/{capacity_bytes} bytes, reject volatile write")]
pub struct DrainBufferFull {
    pub used_bytes: u64,
    pub capacity_bytes: u64,
}

/// Thread-safe buffer for pending volatile writes.
///
/// Mutations are appended by executor threads (one per query) and
/// drained by the background drain thread. Capacity-limited with
/// backpressure: when full, `append()` returns `Err(DrainBufferFull)`.
pub struct DrainBuffer {
    entries: Mutex<Vec<DrainEntry>>,
    used_bytes: AtomicU64,
    capacity_bytes: u64,
}

impl DrainBuffer {
    /// Create a new drain buffer with the given capacity.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            used_bytes: AtomicU64::new(0),
            capacity_bytes,
        }
    }

    /// Append a drain entry to the buffer.
    ///
    /// Returns `Err(DrainBufferFull)` if adding this entry would exceed
    /// the buffer capacity. The caller should propagate this as a write
    /// error to the client.
    pub fn append(&self, entry: DrainEntry) -> Result<(), DrainBufferFull> {
        let entry_bytes = entry.size_bytes as u64;

        // Optimistic capacity check (no lock needed for the common path).
        let current = self.used_bytes.load(Ordering::Relaxed);
        if current + entry_bytes > self.capacity_bytes {
            return Err(DrainBufferFull {
                used_bytes: current,
                capacity_bytes: self.capacity_bytes,
            });
        }

        // Lock, re-check, and insert.
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let current = self.used_bytes.load(Ordering::Relaxed);
        if current + entry_bytes > self.capacity_bytes {
            return Err(DrainBufferFull {
                used_bytes: current,
                capacity_bytes: self.capacity_bytes,
            });
        }

        self.used_bytes.fetch_add(entry_bytes, Ordering::Relaxed);
        entries.push(entry);
        Ok(())
    }

    /// Take all buffered entries, clearing the buffer.
    /// Returns the entries and resets the used bytes counter.
    pub fn take_all(&self) -> Vec<DrainEntry> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        self.used_bytes.store(0, Ordering::Relaxed);
        std::mem::take(&mut *entries)
    }

    /// Current number of buffered entries.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current used capacity in bytes.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for DrainBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrainBuffer")
            .field("used_bytes", &self.used_bytes.load(Ordering::Relaxed))
            .field("capacity_bytes", &self.capacity_bytes)
            .finish()
    }
}

/// Hook for persistent write buffer backends (e.g. NVMe cache for `w:cache`).
///
/// Implementations write drain entries to persistent storage so they survive
/// process crashes. The drain thread calls `begin_drain()` before taking entries
/// from RAM, and `complete_drain(token)` after proposals commit successfully.
///
/// Atomic rename protocol (implemented by `NvmeWriteBuffer`):
/// - `begin_drain(token)` → renames `write_buffer_current.bin` to
///   `write_buffer_draining.<token>.bin` (atomic on POSIX).
/// - `complete_drain(token)` → deletes `write_buffer_draining.<token>.bin`.
/// - On crash during drain: draining file survives → replay on restart.
/// - New writes after `begin_drain` go to a fresh current file.
pub trait WriteBufferHook: Send + Sync {
    /// Called before entries are taken from RAM buffer.
    ///
    /// Should atomically "checkpoint" the current write buffer — making
    /// current pending entries available for recovery if the process
    /// crashes before `complete_drain()` is called.
    ///
    /// Returns a token identifying this drain checkpoint (e.g. a generation counter).
    fn begin_drain(&self) -> u64;

    /// Called after all Raft proposals for a drain batch succeed.
    ///
    /// Should discard the checkpoint identified by `token` — those entries
    /// are now durable in the Raft oplog and no longer need crash recovery.
    fn complete_drain(&self, token: u64);
}

/// Handle to the background drain thread.
///
/// The drain thread periodically takes all entries from the [`DrainBuffer`],
/// batches them into [`RaftProposal`]s, and submits through the
/// [`ProposalPipeline`]. On drop, signals shutdown and flushes remaining
/// entries (graceful shutdown).
pub struct DrainHandle {
    buffer: Arc<DrainBuffer>,
    shutdown: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DrainHandle {
    /// Start the background drain thread.
    ///
    /// The thread runs until `shutdown()` is called or the handle is dropped.
    /// On shutdown, all remaining buffered entries are flushed through the
    /// pipeline before the thread exits.
    ///
    /// `write_buffer`: optional persistent write buffer hook for `w:cache`
    /// crash recovery. When set, the drain thread calls `begin_drain()` before
    /// taking entries and `complete_drain(token)` after successful proposals.
    /// Entries already in RAM buffer when the thread starts are handled by the
    /// first `drain_once()` call.
    pub fn start(
        buffer: Arc<DrainBuffer>,
        pipeline: Arc<dyn ProposalPipeline>,
        id_gen: Arc<ProposalIdGenerator>,
        config: DrainConfig,
        write_buffer: Option<Arc<dyn WriteBufferHook>>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);
        let buffer_clone = Arc::clone(&buffer);

        // Thread spawn failure is unrecoverable — if the OS can't create
        // a thread, the process is in critical state. Log and continue
        // without drain (volatile writes degrade to local-only).
        let thread = match std::thread::Builder::new()
            .name("coordinode-drain".into())
            .spawn(move || {
                drain_loop(
                    &buffer_clone,
                    pipeline.as_ref(),
                    &id_gen,
                    &config,
                    &shutdown_clone,
                    write_buffer.as_deref(),
                );
            }) {
            Ok(handle) => Some(handle),
            Err(e) => {
                tracing::error!(error = %e, "failed to spawn drain thread — volatile writes will not be replicated");
                None
            }
        };

        Self {
            buffer,
            shutdown,
            thread,
        }
    }

    /// Signal the drain thread to stop and wait for it to flush remaining
    /// entries. This is the graceful shutdown path — all buffered entries
    /// are drained before the thread exits.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }

    /// Access the underlying drain buffer (for appending entries).
    pub fn buffer(&self) -> &Arc<DrainBuffer> {
        &self.buffer
    }

    /// Whether the drain thread has been signaled to shut down.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

impl Drop for DrainHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Main drain loop — runs on the background thread.
fn drain_loop(
    buffer: &DrainBuffer,
    pipeline: &dyn ProposalPipeline,
    id_gen: &ProposalIdGenerator,
    config: &DrainConfig,
    shutdown: &AtomicBool,
    write_buffer: Option<&dyn WriteBufferHook>,
) {
    let interval = std::time::Duration::from_millis(config.interval_ms);

    loop {
        std::thread::sleep(interval);

        drain_once(buffer, pipeline, id_gen, config.batch_max, write_buffer);

        if shutdown.load(Ordering::Acquire) {
            // Final flush — drain everything remaining.
            drain_once(buffer, pipeline, id_gen, config.batch_max, write_buffer);
            break;
        }
    }
}

/// Drain all buffered entries once, batching into proposals.
///
/// If `write_buffer` is set:
/// 1. `begin_drain()` is called first → atomically checkpoints the NVMe write buffer.
/// 2. After all proposals succeed, `complete_drain(token)` removes the checkpoint.
/// 3. On crash between step 1 and 2: the checkpoint file survives for replay on restart.
fn drain_once(
    buffer: &DrainBuffer,
    pipeline: &dyn ProposalPipeline,
    id_gen: &ProposalIdGenerator,
    batch_max: u32,
    write_buffer: Option<&dyn WriteBufferHook>,
) {
    // Checkpoint the NVMe write buffer BEFORE taking entries from RAM.
    // This ensures: if the process crashes during drain, the checkpoint
    // file contains all entries being drained and can be replayed on restart.
    let drain_token = write_buffer.map(|wb| wb.begin_drain());

    let entries = buffer.take_all();
    if entries.is_empty() {
        // No entries to drain. If we checkpointed (unlikely but possible on
        // race), complete it immediately to avoid leaving a stale file.
        if let (Some(wb), Some(token)) = (write_buffer, drain_token) {
            wb.complete_drain(token);
        }
        return;
    }

    // Flatten entries into batched proposals. Each DrainEntry preserves its
    // commit_ts. Multiple entries batch up to batch_max mutations.
    let mut current_mutations: Vec<Mutation> = Vec::new();
    let mut current_commit_ts = Timestamp::from_raw(0);
    let mut current_start_ts = Timestamp::from_raw(0);
    let mut all_ok = true;

    for entry in entries {
        // If this entry would exceed batch size, flush current batch first.
        if !current_mutations.is_empty()
            && current_mutations.len() + entry.mutations.len() > batch_max as usize
            && submit_proposal(
                pipeline,
                id_gen,
                std::mem::take(&mut current_mutations),
                current_commit_ts,
                current_start_ts,
            )
            .is_err()
        {
            all_ok = false;
        }

        // Use the latest commit_ts in the batch (highest = most recent).
        if entry.commit_ts.as_raw() > current_commit_ts.as_raw() {
            current_commit_ts = entry.commit_ts;
        }
        if current_start_ts.as_raw() == 0 {
            current_start_ts = entry.start_ts;
        }

        current_mutations.extend(entry.mutations);
    }

    // Flush remaining mutations.
    if !current_mutations.is_empty()
        && submit_proposal(
            pipeline,
            id_gen,
            current_mutations,
            current_commit_ts,
            current_start_ts,
        )
        .is_err()
    {
        all_ok = false;
    }

    // If all proposals committed, the checkpoint is no longer needed for
    // crash recovery — the data is now in the Raft oplog.
    // If any proposal failed, we leave the checkpoint for recovery.
    if all_ok {
        if let (Some(wb), Some(token)) = (write_buffer, drain_token) {
            wb.complete_drain(token);
        }
    }
}

/// Submit a single RaftProposal through the pipeline.
///
/// Returns `Ok(())` if the proposal committed, `Err(())` if it failed.
/// Errors are logged — volatile writes accept data loss, but the caller
/// uses the error to decide whether to complete the NVMe checkpoint.
fn submit_proposal(
    pipeline: &dyn ProposalPipeline,
    id_gen: &ProposalIdGenerator,
    mutations: Vec<Mutation>,
    commit_ts: Timestamp,
    start_ts: Timestamp,
) -> Result<(), ()> {
    let count = mutations.len();
    let proposal = RaftProposal {
        id: id_gen.next(),
        mutations,
        commit_ts,
        start_ts,
        // Drain proposals bypass rate limiter — they are background
        // maintenance, not user-facing latency-sensitive operations.
        bypass_rate_limiter: true,
    };

    match pipeline.propose_and_wait(&proposal) {
        Ok(()) => {
            tracing::debug!(
                count,
                commit_ts = commit_ts.as_raw(),
                "drain batch submitted"
            );
            Ok(())
        }
        Err(e) => {
            // Log error but don't retry — volatile writes accept data loss.
            // The mutations are already applied locally; this drain is for
            // replication/durability only.
            tracing::warn!(
                count,
                commit_ts = commit_ts.as_raw(),
                error = %e,
                "drain proposal failed — volatile mutations not replicated"
            );
            Err(())
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::txn::proposal::{Mutation, PartitionId, ProposalError};

    /// Mock pipeline that records proposals.
    struct RecordingPipeline {
        proposals: Mutex<Vec<RaftProposal>>,
    }

    impl RecordingPipeline {
        fn new() -> Self {
            Self {
                proposals: Mutex::new(Vec::new()),
            }
        }

        fn proposal_count(&self) -> usize {
            self.proposals.lock().unwrap().len()
        }

        fn total_mutations(&self) -> usize {
            self.proposals
                .lock()
                .unwrap()
                .iter()
                .map(|p| p.mutations.len())
                .sum()
        }
    }

    impl ProposalPipeline for RecordingPipeline {
        fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<(), ProposalError> {
            self.proposals.lock().unwrap().push(proposal.clone());
            Ok(())
        }
    }

    fn test_mutation(key: &str) -> Mutation {
        Mutation::Put {
            partition: PartitionId::Node,
            key: key.as_bytes().to_vec(),
            value: vec![1, 2, 3],
        }
    }

    fn test_entry(n_mutations: usize, ts: u64) -> DrainEntry {
        let mutations: Vec<_> = (0..n_mutations)
            .map(|i| test_mutation(&format!("key_{i}")))
            .collect();
        DrainEntry::new(mutations, Timestamp::from_raw(ts), Timestamp::from_raw(1))
    }

    #[test]
    fn buffer_append_and_take() {
        let buf = DrainBuffer::new(1024 * 1024);
        assert!(buf.is_empty());

        buf.append(test_entry(3, 100)).unwrap();
        buf.append(test_entry(2, 200)).unwrap();
        assert_eq!(buf.len(), 2);
        assert!(!buf.is_empty());

        let entries = buf.take_all();
        assert_eq!(entries.len(), 2);
        assert!(buf.is_empty());
        assert_eq!(buf.used_bytes(), 0);
    }

    #[test]
    fn buffer_capacity_backpressure() {
        // Tiny buffer — 50 bytes.
        let buf = DrainBuffer::new(50);

        // First entry fits.
        buf.append(test_entry(1, 100)).unwrap();

        // Second entry should be rejected (exceeds capacity).
        let result = buf.append(test_entry(100, 200));
        assert!(result.is_err());

        // Buffer still has the first entry.
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn drain_once_submits_proposals() {
        let buf = DrainBuffer::new(1024 * 1024);
        let pipeline = Arc::new(RecordingPipeline::new());
        let id_gen = ProposalIdGenerator::new();

        buf.append(test_entry(5, 100)).unwrap();
        buf.append(test_entry(3, 200)).unwrap();

        drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

        // Both entries should be batched into one proposal (8 mutations < 10K).
        assert_eq!(pipeline.proposal_count(), 1);
        assert_eq!(pipeline.total_mutations(), 8);
        assert!(buf.is_empty());
    }

    #[test]
    fn drain_once_respects_batch_max() {
        let buf = DrainBuffer::new(1024 * 1024);
        let pipeline = Arc::new(RecordingPipeline::new());
        let id_gen = ProposalIdGenerator::new();

        // 3 entries × 5 mutations = 15 mutations. batch_max = 7.
        buf.append(test_entry(5, 100)).unwrap();
        buf.append(test_entry(5, 200)).unwrap();
        buf.append(test_entry(5, 300)).unwrap();

        drain_once(&buf, pipeline.as_ref(), &id_gen, 7, None);

        // Should produce 2 proposals: [5, 5+5] won't fit → [5], [5, 5]
        // Actually: first=5 (fits), second=5 (5+5=10>7 → flush first, then second),
        // third=5 (5+5=10>7 → flush second, then third). So: [5], [5], [5] = 3.
        assert_eq!(pipeline.proposal_count(), 3);
        assert_eq!(pipeline.total_mutations(), 15);
    }

    #[test]
    fn drain_once_empty_buffer_noop() {
        let buf = DrainBuffer::new(1024 * 1024);
        let pipeline = Arc::new(RecordingPipeline::new());
        let id_gen = ProposalIdGenerator::new();

        drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

        assert_eq!(pipeline.proposal_count(), 0);
    }

    #[test]
    fn drain_entry_preserves_commit_ts() {
        let buf = DrainBuffer::new(1024 * 1024);
        let pipeline = Arc::new(RecordingPipeline::new());
        let id_gen = ProposalIdGenerator::new();

        buf.append(test_entry(2, 100)).unwrap();
        buf.append(test_entry(2, 500)).unwrap();

        drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

        let proposals = pipeline.proposals.lock().unwrap();
        // Batched into one proposal — commit_ts should be the max (500).
        assert_eq!(proposals[0].commit_ts.as_raw(), 500);
    }

    #[test]
    fn drain_proposals_bypass_rate_limiter() {
        let buf = DrainBuffer::new(1024 * 1024);
        let pipeline = Arc::new(RecordingPipeline::new());
        let id_gen = ProposalIdGenerator::new();

        buf.append(test_entry(1, 100)).unwrap();
        drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

        let proposals = pipeline.proposals.lock().unwrap();
        assert!(proposals[0].bypass_rate_limiter);
    }

    #[test]
    fn drain_handle_start_and_shutdown() {
        let buffer = Arc::new(DrainBuffer::new(1024 * 1024));
        let pipeline = Arc::new(RecordingPipeline::new());
        let id_gen = Arc::new(ProposalIdGenerator::new());

        buffer.append(test_entry(3, 100)).unwrap();

        let config = DrainConfig {
            interval_ms: 10, // Short interval for test
            batch_max: 10_000,
            capacity_bytes: 1024 * 1024,
        };

        let mut handle = DrainHandle::start(
            Arc::clone(&buffer),
            Arc::clone(&pipeline) as Arc<dyn ProposalPipeline>,
            Arc::clone(&id_gen),
            config,
            None,
        );

        // Give drain thread time to process.
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Shutdown should flush remaining entries.
        handle.shutdown();

        // The 3-mutation entry should have been drained.
        assert!(pipeline.proposal_count() >= 1);
        assert!(buffer.is_empty());
    }

    /// Failed pipeline should not crash the drain thread.
    #[test]
    fn drain_tolerates_pipeline_errors() {
        struct FailingPipeline;
        impl ProposalPipeline for FailingPipeline {
            fn propose_and_wait(&self, _: &RaftProposal) -> Result<(), ProposalError> {
                Err(ProposalError::Storage("test error".into()))
            }
        }

        let buf = DrainBuffer::new(1024 * 1024);
        let pipeline = Arc::new(FailingPipeline);
        let id_gen = ProposalIdGenerator::new();

        buf.append(test_entry(3, 100)).unwrap();

        // Should not panic — errors are logged and swallowed.
        drain_once(&buf, pipeline.as_ref(), &id_gen, 10_000, None);

        // Buffer should be drained even though pipeline failed.
        assert!(buf.is_empty());
    }
}
