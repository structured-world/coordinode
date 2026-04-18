//! R-SNAP2: Per-shard `maxAssigned` watermark and `WaitForTs(T)` primitive.
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
//! Per arch/core/transactions.md § Cross-Modality Snapshot Protocol:
//!
//! > Shard checks its local `maxAssigned` counter (monotonic atomic,
//! > updated on every applied commit). If `maxAssigned < T`: `WaitForTs(T)`
//! > — block until applier progresses past T (bounded by ~apply_latency,
//! > typically <1ms; timeout returns `ErrReadTimeout`).
//!
//! ## Why per-shard and not central
//!
//! CoordiNode uses HLC (ADR-007) which gives each shard its own monotonic
//! timestamp stream. No central oracle. The watermark is therefore
//! per-shard state — each shard advances independently. Reads against a
//! specific shard consult that shard's watermark; cross-shard reads wait
//! on every participating shard's watermark concurrently.
//!
//! ## Comparison to Dgraph's central Oracle
//!
//! Dgraph (`posting/oracle.go:197-351`) keeps a central `maxAssigned` in
//! the Zero group with per-startTs `waiters map[uint64][]chan`. We adopt
//! the wait-and-notify pattern but localise it per shard and replace the
//! per-ts channel map with a single `tokio::sync::watch::channel<u64>` —
//! memory O(1) regardless of concurrent waiters, and every waiter re-reads
//! the current value on each `changed()` tick.
//!
//! ## Multi-instance safety
//!
//! This is per-shard in-memory state. On leader failover the new leader
//! rebuilds its watermark by replaying applied Raft entries (the state
//! machine owns the advance call), so the watermark converges to the
//! same value the previous leader had. HLC monotonicity guarantees the
//! new leader's watermark will not regress below any committed `T` the
//! old leader served.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::txn::timestamp::Timestamp;

/// Default timeout for `wait_for` calls when the caller does not specify one.
/// Matches `read_timeout_ms` default in arch/core/transactions.md.
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

    /// The watermark sender was dropped — the owning shard has shut down.
    #[error("WaitForTs aborted: shard watermark channel closed")]
    Closed,
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
    /// `fetch_max(Release)` to keep atomic monotonic progress.
    current: AtomicU64,
    /// Broadcast channel — every `advance` call also publishes the new
    /// value so waiters can wake and re-check. The watch::Sender is also
    /// the source of truth for waiters (they subscribe via `Receiver`).
    tx: tokio::sync::watch::Sender<u64>,
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
        let raw = initial.as_raw();
        let (tx, _rx) = tokio::sync::watch::channel(raw);
        Arc::new(Self {
            current: AtomicU64::new(raw),
            tx,
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
            // Broadcast new value. `send` fails only if every receiver
            // was dropped — that is OK for us (no-op).
            let _ = self.tx.send(target);
            true
        } else {
            false
        }
    }

    /// Block until `maxAssigned ≥ target` or the timeout expires.
    ///
    /// Fast path: if `current ≥ target` at the moment of the call, returns
    /// immediately with `Ok(current)`.
    ///
    /// Slow path: subscribes to the broadcast channel, re-checks after
    /// every advance, and returns `Err(WaitError::Timeout { … })` if the
    /// deadline fires before the watermark reaches `target`.
    pub async fn wait_for(
        &self,
        target: Timestamp,
        timeout: Duration,
    ) -> Result<Timestamp, WaitError> {
        let target_raw = target.as_raw();
        let started = tokio::time::Instant::now();
        let deadline = started + timeout;

        // Fast path — skip channel subscription when already past target.
        let snap = self.current.load(Ordering::Acquire);
        if snap >= target_raw {
            return Ok(Timestamp::from_raw(snap));
        }

        let mut rx = self.tx.subscribe();
        // `changed()` marks the value as seen. Re-check on each iteration
        // in case the watermark advanced between `changed()` and our read.
        loop {
            // Re-check after each changed() — another advance may have
            // already pushed current past target.
            let snap = self.current.load(Ordering::Acquire);
            if snap >= target_raw {
                return Ok(Timestamp::from_raw(snap));
            }
            match tokio::time::timeout_at(deadline, rx.changed()).await {
                Ok(Ok(())) => continue,
                Ok(Err(_)) => return Err(WaitError::Closed),
                Err(_) => {
                    // Deadline expired. Report final observed value so the
                    // client can decide to retry vs fall back.
                    let current = self.current.load(Ordering::Acquire);
                    return Err(WaitError::Timeout {
                        target: target_raw,
                        current,
                        elapsed: started.elapsed(),
                    });
                }
            }
        }
    }

    /// Blocking variant for use from sync contexts (legacy tests / EXPLAIN).
    /// Internally spawns onto a temporary current_thread runtime — cheap
    /// but should not be used on a hot path. Async callers must use
    /// `wait_for`.
    #[doc(hidden)]
    pub fn wait_for_blocking(
        &self,
        target: Timestamp,
        timeout: Duration,
    ) -> Result<Timestamp, WaitError> {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .map_err(|_| WaitError::Closed)?
            .block_on(self.wait_for(target, timeout))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_at_initial_value() {
        let wm = MaxAssignedWatermark::new(Timestamp::from_raw(42));
        assert_eq!(wm.current().as_raw(), 42);
    }

    #[test]
    fn advance_is_monotonic_forward() {
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        assert!(wm.advance(Timestamp::from_raw(100)));
        assert_eq!(wm.current().as_raw(), 100);
        // Same value → no-op.
        assert!(!wm.advance(Timestamp::from_raw(100)));
        // Smaller value → no-op, current unchanged.
        assert!(!wm.advance(Timestamp::from_raw(50)));
        assert_eq!(wm.current().as_raw(), 100);
        // Larger value → moves.
        assert!(wm.advance(Timestamp::from_raw(200)));
        assert_eq!(wm.current().as_raw(), 200);
    }

    #[tokio::test]
    async fn wait_for_fast_path_returns_immediately() {
        let wm = MaxAssignedWatermark::new(Timestamp::from_raw(1000));
        let before = tokio::time::Instant::now();
        let got = wm
            .wait_for(Timestamp::from_raw(800), Duration::from_millis(500))
            .await
            .expect("fast path ok");
        assert_eq!(got.as_raw(), 1000);
        // Fast path should complete in well under 10ms even under load.
        assert!(before.elapsed() < Duration::from_millis(10));
    }

    #[tokio::test]
    async fn wait_for_blocks_until_advance() {
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let wm2 = Arc::clone(&wm);

        // Spawn an "applier" that advances the watermark after ~50ms.
        let advancer = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            wm2.advance(Timestamp::from_raw(500));
        });

        let before = tokio::time::Instant::now();
        let got = wm
            .wait_for(Timestamp::from_raw(500), Duration::from_millis(1000))
            .await
            .expect("advance reached target");
        let elapsed = before.elapsed();

        assert_eq!(got.as_raw(), 500);
        assert!(
            elapsed >= Duration::from_millis(40),
            "should have waited ~50ms, elapsed={elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(200),
            "should unblock promptly after advance, elapsed={elapsed:?}"
        );
        advancer.await.unwrap();
    }

    #[tokio::test]
    async fn wait_for_times_out_when_applier_stalls() {
        let wm = MaxAssignedWatermark::new(Timestamp::from_raw(100));
        // Target is above current; nobody advances. Must hit timeout and
        // return `ErrTimeout`, NOT stale `Ok(100)`.
        let err = wm
            .wait_for(Timestamp::from_raw(500), Duration::from_millis(50))
            .await
            .expect_err("must time out, not return stale");
        match err {
            WaitError::Timeout {
                target,
                current,
                elapsed,
            } => {
                assert_eq!(target, 500);
                assert_eq!(current, 100);
                assert!(elapsed >= Duration::from_millis(40));
            }
            other => panic!("expected Timeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn wait_for_wakes_multiple_concurrent_waiters() {
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);

        let mut handles = Vec::new();
        for target in [100, 200, 300, 400].iter() {
            let wm_c = Arc::clone(&wm);
            let t = *target;
            handles.push(tokio::spawn(async move {
                wm_c.wait_for(Timestamp::from_raw(t), Duration::from_millis(1000))
                    .await
                    .map(|ts| ts.as_raw())
            }));
        }

        // Single jump to 500 should wake all waiters (all targets ≤ 500).
        tokio::time::sleep(Duration::from_millis(20)).await;
        wm.advance(Timestamp::from_raw(500));

        for h in handles {
            let got = h.await.unwrap().expect("all waiters succeed");
            assert_eq!(got, 500);
        }
    }

    #[tokio::test]
    async fn wait_for_partial_advance_keeps_later_waiters_blocked() {
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let wm_fast = Arc::clone(&wm);
        let wm_slow = Arc::clone(&wm);

        let fast = tokio::spawn(async move {
            wm_fast
                .wait_for(Timestamp::from_raw(100), Duration::from_millis(500))
                .await
        });
        let slow = tokio::spawn(async move {
            wm_slow
                .wait_for(Timestamp::from_raw(1000), Duration::from_millis(100))
                .await
        });

        // Advance enough for fast but not slow.
        tokio::time::sleep(Duration::from_millis(20)).await;
        wm.advance(Timestamp::from_raw(200));

        let fast_res = fast.await.unwrap().expect("fast reaches target");
        assert_eq!(fast_res.as_raw(), 200);

        let slow_res = slow.await.unwrap();
        match slow_res {
            Err(WaitError::Timeout { current, .. }) => assert_eq!(current, 200),
            other => panic!("slow waiter should timeout with current=200, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn wait_for_zero_target_is_trivial() {
        // Target=0 is always satisfied regardless of current.
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let got = wm
            .wait_for(Timestamp::ZERO, Duration::from_millis(10))
            .await
            .expect("zero target is trivially reached");
        assert_eq!(got.as_raw(), 0);
    }

    #[tokio::test]
    async fn wait_for_default_timeout_is_2s() {
        // Sanity check that the documented default matches the constant.
        assert_eq!(DEFAULT_WAIT_TIMEOUT, Duration::from_millis(2000));
    }

    #[tokio::test]
    async fn advance_notifies_even_when_no_waiters() {
        // No waiters exist — advance should still succeed (send to closed
        // channel on drop is tolerated).
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        assert!(wm.advance(Timestamp::from_raw(42)));
        assert_eq!(wm.current().as_raw(), 42);
    }

    #[tokio::test]
    async fn concurrent_advances_preserve_maximum() {
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let mut handles = Vec::new();
        for i in 0..20_u64 {
            let wm_c = Arc::clone(&wm);
            handles.push(tokio::spawn(async move {
                wm_c.advance(Timestamp::from_raw(i * 100));
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        // Final value must equal the maximum of all advances (1900).
        assert_eq!(wm.current().as_raw(), 1900);
    }
}
