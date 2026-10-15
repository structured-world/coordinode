//! Per-shard `maxAssigned` watermark and `WaitForTs(T)` primitive.
//!
//! ## Role in the Cross-Modality Snapshot Protocol
//!
//! A shard serves a snapshot read at HLC timestamp `T` only after every
//! write with `commit_ts ≤ T` has been applied locally to every modality
//! (graph, documents, time-series, FTS, vector). The shard-local
//! `MaxAssignedWatermark` tracks this progress: the applier advances it on
//! every applied commit, readers at `T` block on `wait_for(T, timeout)`
//! until the watermark reaches `T` (or the timeout fires).
//!
//! The shard checks its local `maxAssigned` counter (monotonic atomic,
//! updated on every applied commit). If `maxAssigned < T`, `WaitForTs(T)`
//! blocks until the applier progresses past T (bounded by ~apply_latency,
//! typically <1ms; the timeout returns `ErrReadTimeout`).
//!
//! ## Why per-shard and not central
//!
//! CoordiNode uses HLC, which gives each shard its own monotonic
//! timestamp stream. No central oracle. The watermark is therefore
//! per-shard state — each shard advances independently. Reads against a
//! specific shard consult that shard's watermark; cross-shard reads wait
//! on every participating shard's watermark concurrently.
//!
//! ## Waiting without a per-timestamp registry
//!
//! A central oracle would keep one wait list per requested timestamp. Here
//! the wait is per shard, on a single condition variable: memory is O(1)
//! regardless of how many readers wait, and every waiter re-reads the
//! current value each time the watermark moves. The wait blocks the calling
//! thread; the readers that use it are synchronous, and no async runtime is
//! needed to serve them.
//!
//! ## Multi-instance safety
//!
//! This is per-shard in-memory state. On leader failover the new leader
//! rebuilds its watermark by replaying applied Raft entries (the state
//! machine owns the advance call), so the watermark converges to the
//! same value the previous leader had. HLC monotonicity guarantees the
//! new leader's watermark will not regress below any committed `T` the
//! old leader served.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::txn::timestamp::Timestamp;

/// Default timeout for `wait_for` calls when the caller does not specify one.
/// Matches the default `read_timeout_ms`.
pub const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_millis(2000);

/// Errors returned by `MaxAssignedWatermark::wait_for`.
#[derive(Debug, thiserror::Error)]
pub enum WaitError {
    /// The timeout expired before `maxAssigned` reached `target`.
    ///
    /// `current` is the watermark value observed at the moment the
    /// deadline fired — always strictly less than `target`. Clients
    /// typically retry on a fresher leader or fall back to
    /// `read_consistency = 'current'`.
    #[error(
        "WaitForTs timed out: waited {elapsed:?} for target ts={target}, \
         current maxAssigned={current}"
    )]
    Timeout {
        target: u64,
        current: u64,
        elapsed: Duration,
    },
}

/// Per-shard monotonic watermark of the highest `commit_ts` whose write has
/// been applied to every local modality.
///
/// Advances are atomic and monotonic: a call with `ts < current` is a
/// no-op. Waiters are woken whenever the watermark moves forward; each
/// waiter re-checks its target against the new value and re-suspends if
/// the target is still in the future.
pub struct MaxAssignedWatermark {
    /// Current maxAssigned value. Read via `load(Acquire)`, written via
    /// `fetch_max(AcqRel)` to keep atomic monotonic progress.
    current: AtomicU64,
    /// Held by a waiter while it checks `current` and parks, and by an
    /// advance while it notifies: an advance landing between a waiter's
    /// check and its sleep waits for the sleep, so its wake is never lost.
    // no-std: spin::Mutex with a caller-provided parking hook.
    parked: parking_lot::Mutex<()>,
    advanced: parking_lot::Condvar,
}

impl std::fmt::Debug for MaxAssignedWatermark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxAssignedWatermark")
            .field("current", &self.current.load(Ordering::Acquire))
            .finish()
    }
}

impl MaxAssignedWatermark {
    /// Create a watermark initialised to `initial`. Callers typically start
    /// from `Timestamp::ZERO` and let the first applied entry move the
    /// watermark forward.
    pub fn new(initial: Timestamp) -> Arc<Self> {
        Arc::new(Self {
            current: AtomicU64::new(initial.as_raw()),
            parked: parking_lot::Mutex::new(()),
            advanced: parking_lot::Condvar::new(),
        })
    }

    /// Current watermark value. Lock-free.
    pub fn current(&self) -> Timestamp {
        Timestamp::from_raw(self.current.load(Ordering::Acquire))
    }

    /// Advance the watermark to `ts`, notifying every waiter. Monotonic —
    /// calls with `ts ≤ current` are ignored. Returns `true` when the
    /// watermark actually moved forward.
    pub fn advance(&self, ts: Timestamp) -> bool {
        let target = ts.as_raw();
        let prev = self.current.fetch_max(target, Ordering::AcqRel);
        if prev < target {
            let _parked = self.parked.lock();
            self.advanced.notify_all();
            true
        } else {
            false
        }
    }

    /// Block the calling thread until `maxAssigned ≥ target` or the timeout
    /// expires.
    ///
    /// Returns at once when the watermark is already there, without taking
    /// the lock. Otherwise it parks, re-checks after every advance, and
    /// returns `Err(WaitError::Timeout { … })` if the deadline passes first.
    /// An async caller runs it where blocking is allowed (`spawn_blocking`,
    /// `block_in_place`).
    pub fn wait_for(&self, target: Timestamp, timeout: Duration) -> Result<Timestamp, WaitError> {
        let target_raw = target.as_raw();
        let snap = self.current.load(Ordering::Acquire);
        if snap >= target_raw {
            return Ok(Timestamp::from_raw(snap));
        }

        let started = Instant::now();
        let deadline = started + timeout;
        let mut parked = self.parked.lock();
        loop {
            let snap = self.current.load(Ordering::Acquire);
            if snap >= target_raw {
                return Ok(Timestamp::from_raw(snap));
            }
            if self.advanced.wait_until(&mut parked, deadline).timed_out() {
                // An advance may have landed with the deadline: report it
                // as reached rather than as a timeout it is not.
                let current = self.current.load(Ordering::Acquire);
                if current >= target_raw {
                    return Ok(Timestamp::from_raw(current));
                }
                return Err(WaitError::Timeout {
                    target: target_raw,
                    current,
                    elapsed: started.elapsed(),
                });
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
