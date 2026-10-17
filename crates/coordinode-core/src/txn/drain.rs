//! Volatile write drain buffer for `w:memory` and `w:cache` write concerns.
//!
//! A volatile commit is validated as any commit is, then buffered here
//! instead of applied: it is not committed state until the drain hands it to
//! the log, so no committed read sees it before then. The drain thread
//! batches buffered entries into [`RaftProposal`]s and submits them through
//! the [`ProposalPipeline`], and the log applies each once.
//!
//! ## Invariants
//!
//! - A drained entry enters the log and is applied exactly once
//! - Until then the keys it was validated on stay registered (its
//!   [`DrainHold`]), so no other commit overwrites them in between
//! - It lands at a timestamp taken at the drain, above every snapshot served
//!   while it waited
//! - An entry bound to a client operation is proposed on its own, with its
//!   binding, so its outcome is never merged with another's
//! - Crash before drain = data lost (explicit `w:memory`/`w:cache` contract)
//! - Graceful shutdown flushes all drain buffers before exit
//!
//! ## Data flow
//!
//! ```text
//! Client write (w:memory)
//!   → validation, keys registered → DrainBuffer.append(entry) → ACK
//!   ... background drain (100ms interval):
//!   → timestamp taken → batch entries → RaftProposal
//!   → pipeline.propose_and_wait() → oplog → applied once, keys released
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

/// The protection a buffered entry keeps until it is handed to the log: the
/// keys it was validated on stay registered, so no writer overwrites what it
/// was validated against before it lands, and the timestamp it lands at.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not protect a buffered entry until it is logged",
    label = "not a drain hold",
    note = "the store hands one out when it accepts a volatile commit"
)]
pub trait DrainHold: Send + Sync + core::fmt::Debug {
    /// The timestamp the entry lands at, taken now and registered as a
    /// commit applying, so it is above every snapshot served while the entry
    /// waited. `None` when the entry keeps the timestamp it was built with,
    /// which this hold then keeps from being passed over.
    fn land(&mut self) -> Option<Timestamp> {
        None
    }
}

/// A buffered volatile write entry waiting for drain.
///
/// Contains the mutations from one transaction, validated and not applied
/// anywhere yet. The drain thread will package these into a Raft proposal.
#[derive(Debug)]
pub struct DrainEntry {
    /// Mutations from the original transaction (Put/Delete/Merge).
    pub mutations: Vec<Mutation>,
    /// The timestamp the entry lands at when its hold takes none at the
    /// drain (see [`DrainHold::land`]).
    pub commit_ts: Timestamp,
    /// Start timestamp of the original transaction.
    pub start_ts: Timestamp,
    /// Approximate size in bytes (for capacity tracking).
    size_bytes: usize,
    /// Kept until the entry is handed to the log; see [`DrainHold`].
    hold: Option<Box<dyn DrainHold>>,
}

impl DrainEntry {
    /// Create a new drain entry from transaction mutations.
    pub fn new(mutations: Vec<Mutation>, commit_ts: Timestamp, start_ts: Timestamp) -> Self {
        let size_bytes = mutations.iter().map(Mutation::size_estimate).sum();
        Self {
            mutations,
            commit_ts,
            start_ts,
            size_bytes,
            hold: None,
        }
    }

    /// Approximate bytes the entry holds, what a [`DrainBuffer`] counts.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Add `mark` to what the entry writes, in the same unit as its effects
    /// and ahead of the binding that ends a client operation's unit: the
    /// evidence that the entry was applied, written exactly when it is.
    pub fn mark(&mut self, mark: Mutation) {
        self.size_bytes += mark.size_estimate();
        let at = match self.mutations.last() {
            Some(Mutation::Operation(_)) => self.mutations.len() - 1,
            _ => self.mutations.len(),
        };
        self.mutations.insert(at, mark);
    }

    /// This entry, keeping `hold` until it is handed to the log.
    #[must_use]
    pub fn holding(mut self, hold: Box<dyn DrainHold>) -> Self {
        self.hold = Some(hold);
        self
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
    /// The drain thread sleeps on this while the buffer is empty; an append
    /// wakes it.
    wake: super::wake::Wake,
}

impl DrainBuffer {
    /// Create a new drain buffer with the given capacity.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            used_bytes: AtomicU64::new(0),
            capacity_bytes,
            wake: super::wake::Wake::default(),
        }
    }

    /// Append a drain entry to the buffer.
    ///
    /// Returns `Err(DrainBufferFull)` if adding this entry would exceed
    /// the buffer capacity. The caller should propagate this as a write
    /// error to the client.
    pub fn append(&self, entry: DrainEntry) -> Result<(), DrainBufferFull> {
        self.reserve(entry.size_bytes as u64)?.fill(entry);
        Ok(())
    }

    /// Room for `bytes` of entry, taken now and given back unless filled:
    /// a caller that has to do something else before the entry is in the
    /// buffer (persist it for crash recovery) learns first that it fits.
    pub fn reserve(&self, bytes: u64) -> Result<DrainSlot<'_>, DrainBufferFull> {
        let mut used = self.used_bytes.load(Ordering::Acquire);
        loop {
            let Some(after) = used
                .checked_add(bytes)
                .filter(|after| *after <= self.capacity_bytes)
            else {
                return Err(DrainBufferFull {
                    used_bytes: used,
                    capacity_bytes: self.capacity_bytes,
                });
            };
            match self.used_bytes.compare_exchange_weak(
                used,
                after,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(now) => used = now,
            }
        }
        Ok(DrainSlot {
            buffer: self,
            bytes,
            filled: false,
        })
    }

    /// Take all buffered entries, clearing the buffer, and give back the room
    /// they held. Room reserved and not filled yet stays reserved.
    pub fn take_all(&self) -> Vec<DrainEntry> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let taken = std::mem::take(&mut *entries);
        let held: u64 = taken.iter().map(|e| e.size_bytes as u64).sum();
        self.used_bytes.fetch_sub(held, Ordering::AcqRel);
        taken
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

/// Room reserved in a [`DrainBuffer`] (see [`DrainBuffer::reserve`]); given
/// back on drop unless filled.
pub struct DrainSlot<'b> {
    buffer: &'b DrainBuffer,
    bytes: u64,
    filled: bool,
}

impl DrainSlot<'_> {
    /// Put `entry` in the room reserved for it. An entry larger than the
    /// room takes the difference too: the reservation was for its size, so
    /// only a caller's mistake makes them differ.
    pub fn fill(mut self, entry: DrainEntry) {
        debug_assert_eq!(
            entry.size_bytes as u64, self.bytes,
            "filled with another size"
        );
        let mut entries = self
            .buffer
            .entries
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let size = entry.size_bytes as u64;
        if size > self.bytes {
            self.buffer
                .used_bytes
                .fetch_add(size - self.bytes, Ordering::AcqRel);
        } else {
            self.buffer
                .used_bytes
                .fetch_sub(self.bytes - size, Ordering::AcqRel);
        }
        entries.push(entry);
        drop(entries);
        self.filled = true;
        self.buffer.wake.notify();
    }
}

impl Drop for DrainSlot<'_> {
    fn drop(&mut self) {
        if !self.filled {
            self.buffer
                .used_bytes
                .fetch_sub(self.bytes, Ordering::AcqRel);
        }
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
        self.buffer.wake.interrupt();
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
    buffer.wake.bind();

    loop {
        // Asleep while nothing is buffered; an append or the stop wakes it.
        if !shutdown.load(Ordering::Acquire) {
            buffer.wake.wait(None);
        }
        // Once there is something to drain, keep the batching window the
        // interval sets, so writes arriving together go in one proposal. The
        // stop cuts the window short.
        let deadline = std::time::Instant::now() + interval;
        while !shutdown.load(Ordering::Acquire) {
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            std::thread::park_timeout(deadline - now);
        }

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
    // The holds of the entries in the current batch, released once the batch
    // is handed to the log.
    let mut current_holds: Vec<Box<dyn DrainHold>> = Vec::new();
    let mut all_ok = true;

    for mut entry in entries {
        let mut hold = entry.hold.take();
        // Taken now, at the drain: the entry lands above every snapshot
        // served while it waited.
        if let Some(ts) = hold.as_mut().and_then(|h| h.land()) {
            entry.commit_ts = ts;
        }
        // A unit with DERIVED index work is proposed on its own: its record
        // sources are positions within the unit, and the entries it derives
        // apply after the unit's other writes, so joined to another unit it
        // would derive from the wrong record, or land after a later unit's
        // write of the same entry. So is a unit bound to a client operation:
        // its binding decides the unit it ends, and a refused binding must
        // not take another operation's effects with it.
        let derives = entry
            .mutations
            .iter()
            .any(|m| matches!(m, Mutation::Derive(_) | Mutation::Operation(_)));
        // If this entry would exceed batch size, flush current batch first.
        if !current_mutations.is_empty()
            && (derives || current_mutations.len() + entry.mutations.len() > batch_max as usize)
        {
            if submit_proposal(
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
            current_holds.clear();
        }

        if derives {
            if submit_proposal(
                pipeline,
                id_gen,
                entry.mutations,
                entry.commit_ts,
                entry.start_ts,
            )
            .is_err()
            {
                all_ok = false;
            }
            drop(hold);
            continue;
        }
        current_holds.extend(hold);

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
    drop(current_holds);

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
    // Coalesce dense runs of point deletes into range deletes before proposing,
    // shrinking the Raft payload / oplog / PITR log on bulk deletes.
    let mutations =
        super::coalesce::coalesce_delete_mutations(mutations, super::coalesce::DEFAULT_MIN_RUN);
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
        Ok(_) => {
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
mod tests;
