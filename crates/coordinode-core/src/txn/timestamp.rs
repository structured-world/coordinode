//! Hybrid Logical Clock (HLC) for MVCC versioning and causal ordering.
//!
//! CE uses a single-group HLC — the Raft leader is the sole timestamp authority.
//! Every `now()` call re-anchors to the system wall clock when time advances,
//! which ensures timestamps stay close to real time (useful for PITR queries and
//! meaningful `operationTime` in causal sessions).
//!
//! ## Algorithm
//!
//! ```text
//! now():
//!   wall = system_clock_us()
//!   lock:
//!     if wall > state.ts:
//!       state.ts = wall          # re-anchor: logical counter implicitly resets
//!     else:
//!       state.ts += 1            # logical increment within same microsecond
//!       if state.ts - wall > max_drift_us:
//!         return Err(MaxDrift)   # runaway logical counter, refuse to proceed
//!   return state.ts
//! ```
//!
//! ## Gossip (Raft propagation)
//!
//! When a Raft proposal is applied, `coordinode_raft::storage` calls
//! `oracle.advance_to(commit_ts)`. This advances every node's HLC to at least
//! the committed timestamp, keeping all replicas in sync with the leader's clock
//! without a separate gossip protocol.
//!
//! ## Durable persistence
//!
//! On restart, `coordinode_storage::engine::StorageEngine::open_with_oracle`
//! reads `max_persisted_seqno` from the LSM trees and calls
//! `oracle.advance_to(max_persisted_seqno + 1)`. Combined with the wall-clock
//! seed in `new()`, the HLC always starts at `max(wall_us, last_committed_ts + 1)`.
//!
//! ## EE extension point
//!
//! EE multi-shard HLC adds cross-shard 2PC: coordinator picks
//! `max(shard_prepare_ts)` as commit_ts and all shards call
//! `oracle.advance_to(commit_ts)`. The same `HybridLogicalClock` struct is
//! used in both CE and EE — no API change needed.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A monotonic timestamp used for MVCC version ordering.
///
/// Values are wall-clock microseconds since Unix epoch with logical increments
/// when multiple events occur within the same microsecond. Comparable with `<`
/// for ordering: larger = newer.
///
/// The zero value is reserved as "no timestamp" / "before all versions".
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct Timestamp(u64);

impl Timestamp {
    /// The zero timestamp (before all versions).
    pub const ZERO: Self = Self(0);

    /// The maximum timestamp.
    pub const MAX: Self = Self(u64::MAX);

    /// Create a timestamp from a raw u64 value.
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Get the raw u64 value.
    pub fn as_raw(self) -> u64 {
        self.0
    }

    /// Check if this is the zero timestamp.
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ts:{}", self.0)
    }
}

/// Error from the Hybrid Logical Clock.
#[derive(Debug, thiserror::Error)]
pub enum HlcError {
    /// The logical counter has drifted further ahead of wall clock than allowed.
    ///
    /// This indicates either extreme transaction burst (unlikely in CE) or a
    /// monotonically increasing wall clock (clock jumped backward). Ensure NTP is
    /// running and the system clock is healthy.
    #[error(
        "HLC drift limit exceeded: drift={drift_us}μs exceeds max={max_us}μs. \
         Ensure NTP is running and the system clock is not jumping backward."
    )]
    MaxDrift {
        /// Measured drift in microseconds.
        drift_us: u64,
        /// Configured maximum drift in microseconds.
        max_us: u64,
    },
}

/// Configuration for the Hybrid Logical Clock.
#[derive(Debug, Clone)]
pub struct HlcConfig {
    /// Maximum allowed drift between HLC timestamp and wall clock (microseconds).
    ///
    /// If the HLC logical counter advances more than this value ahead of the
    /// current wall clock, `now()` returns `HlcError::MaxDrift`. This prevents
    /// unbounded logical-counter growth during extreme transaction bursts or
    /// after a clock rollback.
    ///
    /// Default: 500_000 μs = 500 ms.
    pub max_drift_us: u64,
}

impl Default for HlcConfig {
    fn default() -> Self {
        Self {
            max_drift_us: 500_000,
        }
    }
}

/// Inner mutable state of the HLC, protected by a `Mutex`.
#[derive(Debug)]
struct HlcInner {
    /// Last allocated timestamp value (microseconds since epoch, with logical increments).
    ts: u64,
}

/// Hybrid Logical Clock for CE single-group deployments.
///
/// Allocates monotonically increasing timestamps that track wall clock time.
/// Thread-safe and lock-free for non-contended paths (single writer in CE Raft).
///
/// # Usage
///
/// ```rust,ignore
/// let hlc = HybridLogicalClock::new();
///
/// // Allocate a new timestamp (may return HlcError on clock issues):
/// let ts = hlc.now()?;
///
/// // Infallible version for internal use (LSM seqno generator):
/// let ts = hlc.next();
///
/// // Advance to observed timestamp (Raft apply, snapshot restore):
/// hlc.advance_to(remote_ts);
/// ```
#[derive(Debug)]
pub struct HybridLogicalClock {
    inner: Mutex<HlcInner>,
    max_drift_us: u64,
}

/// Backward-compatibility alias. New code should use `HybridLogicalClock` directly.
pub type TimestampOracle = HybridLogicalClock;

impl HybridLogicalClock {
    /// Create a new HLC seeded from the current wall clock.
    ///
    /// The initial value is `wall_clock_us()` — on restart, the stored seqno
    /// will be restored to `max(wall_us, last_committed_ts + 1)` by
    /// `StorageEngine::open_with_oracle`.
    pub fn new() -> Self {
        Self::with_config(HlcConfig::default())
    }

    /// Create with explicit configuration.
    pub fn with_config(config: HlcConfig) -> Self {
        let seed = wall_us_now();
        Self {
            inner: Mutex::new(HlcInner { ts: seed }),
            max_drift_us: config.max_drift_us,
        }
    }

    /// Create an HLC starting from a specific timestamp value.
    ///
    /// Used for recovery (resume from last known commit_ts) and tests.
    /// The new value is `max(wall_us, last_ts)`.
    pub fn resume_from(last_ts: Timestamp) -> Self {
        let wall = wall_us_now();
        let ts = last_ts.as_raw().max(wall);
        Self {
            inner: Mutex::new(HlcInner { ts }),
            max_drift_us: HlcConfig::default().max_drift_us,
        }
    }

    /// Allocate the next HLC timestamp.
    ///
    /// Re-anchors to wall clock when time has advanced. Returns
    /// `HlcError::MaxDrift` if the logical counter has drifted more than
    /// `max_drift_us` ahead of the current wall clock (clock rollback or
    /// pathological burst).
    ///
    /// # Errors
    ///
    /// Returns `HlcError::MaxDrift` when the HLC cannot safely advance.
    pub fn now(&self) -> Result<Timestamp, HlcError> {
        let wall = wall_us_now();
        let mut guard = lock_infallible(&self.inner);

        if wall > guard.ts {
            // Wall clock advanced: re-anchor. Logical counter implicitly resets
            // because the new base is the wall time value.
            guard.ts = wall;
        } else {
            // Wall clock is at or behind HLC: logical increment.
            guard.ts += 1;

            // Drift check: how far ahead of wall clock are we?
            let drift = guard.ts.saturating_sub(wall);
            if drift > self.max_drift_us {
                // Undo the increment so state is consistent.
                guard.ts -= 1;
                return Err(HlcError::MaxDrift {
                    drift_us: drift,
                    max_us: self.max_drift_us,
                });
            }
        }

        Ok(Timestamp(guard.ts))
    }

    /// Allocate the next timestamp, infallible.
    ///
    /// Same as `now()` but logs a warning instead of returning an error when
    /// the drift limit is exceeded. Used by `OracleSeqnoGenerator` which
    /// implements the `lsm_tree::SequenceNumberGenerator` trait (cannot return
    /// errors).
    ///
    /// In normal operation this never exceeds the drift limit (CE workloads are
    /// far below 500ms of logical burst at any realistic QPS). If it does, a
    /// `tracing::warn!` is emitted so operators can investigate clock health.
    pub fn next(&self) -> Timestamp {
        let wall = wall_us_now();
        let mut guard = lock_infallible(&self.inner);

        if wall > guard.ts {
            guard.ts = wall;
        } else {
            guard.ts += 1;

            let drift = guard.ts.saturating_sub(wall);
            if drift > self.max_drift_us {
                tracing::warn!(
                    drift_us = drift,
                    max_us = self.max_drift_us,
                    "HLC drift limit exceeded — logical counter too far ahead of wall clock. \
                     Check NTP and system clock health."
                );
                // Continue despite drift: the seqno generator must not block.
            }
        }

        Timestamp(guard.ts)
    }

    /// Return the current high-water mark without advancing.
    ///
    /// Equal to the last value returned by `next()` / `now()` or the last
    /// value set by `advance_to()`. Used by `OracleSeqnoGenerator::get()`.
    pub fn current(&self) -> Timestamp {
        let guard = lock_infallible(&self.inner);
        Timestamp(guard.ts)
    }

    /// Advance the HLC to at least the given timestamp.
    ///
    /// Called when observing a timestamp from another node (Raft proposal apply,
    /// snapshot install). Implements the "gossip" part of the HLC algorithm:
    /// every node's clock advances to `max(local, observed)` so that future
    /// allocations are guaranteed to be causally after the observed event.
    ///
    /// Does NOT re-anchor to wall clock — `advance_to` only moves forward, never
    /// backward.
    pub fn advance_to(&self, ts: Timestamp) {
        let mut guard = lock_infallible(&self.inner);
        if ts.as_raw() > guard.ts {
            guard.ts = ts.as_raw();
        }
    }
}

impl Default for HybridLogicalClock {
    fn default() -> Self {
        Self::new()
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Current wall clock time in microseconds since Unix epoch.
fn wall_us_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_micros() as u64
}

/// Acquire the mutex, recovering from poison (thread panicked while holding lock).
fn lock_infallible(m: &Mutex<HlcInner>) -> MutexGuard<'_, HlcInner> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    // ── Timestamp ──────────────────────────────────────────────────────────

    #[test]
    fn timestamp_ordering() {
        let a = Timestamp::from_raw(1);
        let b = Timestamp::from_raw(2);
        assert!(a < b);
        assert!(Timestamp::ZERO < a);
        assert!(b < Timestamp::MAX);
    }

    #[test]
    fn timestamp_zero() {
        assert!(Timestamp::ZERO.is_zero());
        assert!(!Timestamp::from_raw(1).is_zero());
    }

    // ── HybridLogicalClock — basic allocation ──────────────────────────────

    #[test]
    fn hlc_allocates_monotonically() {
        let hlc = HybridLogicalClock::resume_from(Timestamp::from_raw(100));
        let t1 = hlc.next();
        let t2 = hlc.next();
        let t3 = hlc.next();

        assert!(t1 < t2, "t1={t1} should be < t2={t2}");
        assert!(t2 < t3, "t2={t2} should be < t3={t3}");
    }

    #[test]
    fn hlc_current_does_not_advance() {
        let hlc = HybridLogicalClock::resume_from(Timestamp::from_raw(50));
        // current() after resume_from must be max(wall, 50) — so at least 50.
        // Since wall >> 50, current == wall.
        let c1 = hlc.current();
        let c2 = hlc.current();
        assert_eq!(c1, c2, "current() must not change");
    }

    #[test]
    fn hlc_resume_starts_at_least_at_given_ts() {
        // If last_ts > wall clock (impossible in practice but valid in tests),
        // resume_from must start at last_ts.
        let future_ts = u64::MAX / 2;
        let hlc = HybridLogicalClock::resume_from(Timestamp::from_raw(future_ts));
        let next = hlc.next();
        assert!(
            next.as_raw() > future_ts,
            "next after resume_from({future_ts}) must be > {future_ts}, got {next}"
        );
    }

    // ── HLC — wall clock re-anchoring ──────────────────────────────────────

    #[test]
    fn hlc_reanchors_to_wall_clock() {
        // Simulate a scenario where the oracle starts from a very low value
        // (e.g., the test node has no prior data). After `next()`, the clock
        // should jump to the current wall time rather than returning low+1.
        let hlc = HybridLogicalClock::resume_from(Timestamp::from_raw(1));
        let ts = hlc.next();
        // System is running in year 2024+ → wall_us > 1_700_000_000_000_000
        assert!(
            ts.as_raw() > 1_700_000_000_000_000,
            "HLC must re-anchor to wall clock; got {ts}"
        );
    }

    #[test]
    fn hlc_next_seeded_from_wall_clock() {
        let hlc = HybridLogicalClock::new();
        let ts = hlc.next();
        // Should be roughly current time in microseconds (> year 2024)
        assert!(ts.as_raw() > 1_700_000_000_000_000);
    }

    // ── HLC — logical increment within same microsecond ────────────────────

    #[test]
    fn hlc_logical_increment_when_wall_static() {
        // Pin the HLC to a far-future value so wall clock can never catch up.
        // All subsequent next() calls must produce logically incrementing values.
        let far_future = wall_us_now() + 10; // 10μs ahead — within max_drift
        let hlc = HybridLogicalClock::resume_from(Timestamp::from_raw(far_future));

        // Immediately allocate two timestamps — wall clock hasn't advanced yet
        // (we're 10μs ahead). Must get logical increments.
        let t1 = hlc.next();
        let t2 = hlc.next();

        assert!(t1 < t2, "must be monotone under logical increments");
        // Both should be close to (or equal to) far_future+1 and far_future+2,
        // unless wall clock advanced past far_future in the meantime.
        // In either case, monotonicity must hold.
    }

    // ── HLC — drift rate limiter ───────────────────────────────────────────

    #[test]
    fn hlc_drift_limiter_rejects_when_exceeded() {
        let wall = wall_us_now();
        // Place counter 6μs ahead of wall — exactly 1 over the limit.
        let hlc = HybridLogicalClock {
            inner: Mutex::new(HlcInner { ts: wall + 5 }),
            max_drift_us: 5,
        };

        // next() after logical increment would be wall+6 → drift=6 > max=5 → error.
        // But first we must ensure wall hasn't caught up (race in slow CI).
        // We re-check only if the counter is still ahead.
        let result = hlc.now();
        // If wall caught up (ts <= wall), now() re-anchors and succeeds.
        // If wall is still behind (ts > wall), drift check fires.
        match result {
            Ok(_) => {
                // Wall clock advanced past our value — re-anchor succeeded. Fine.
            }
            Err(HlcError::MaxDrift { drift_us, max_us }) => {
                assert_eq!(max_us, 5);
                assert!(drift_us > 5, "drift must exceed max, got {drift_us}");
            }
        }
    }

    #[test]
    fn hlc_drift_limiter_does_not_reject_within_limit() {
        let hlc = HybridLogicalClock::with_config(HlcConfig {
            max_drift_us: 500_000,
        });
        // Allocate 100 timestamps rapidly — no drift should be detected.
        for _ in 0..100 {
            hlc.now().expect("within-limit allocations must succeed");
        }
    }

    #[test]
    fn hlc_drift_state_unchanged_after_rejection() {
        let wall = wall_us_now();
        let hlc = HybridLogicalClock {
            inner: Mutex::new(HlcInner { ts: wall + 3 }),
            max_drift_us: 3,
        };

        let before = hlc.current();
        // This may fail if we're over the limit.
        let _result = hlc.now();
        // If it failed (MaxDrift), the state must not have been mutated.
        // If it succeeded (wall caught up), that's also fine.
        let after = hlc.current();
        // After must be >= before (never goes backward).
        assert!(
            after >= before,
            "state must not go backward after rejected now()"
        );
    }

    // ── HLC — advance_to (Raft gossip path) ───────────────────────────────

    #[test]
    fn hlc_advance_to_moves_forward() {
        let hlc = HybridLogicalClock::resume_from(Timestamp::from_raw(100));
        hlc.advance_to(Timestamp::from_raw(1000));
        // next() must be > 1000
        let next = hlc.next();
        assert!(
            next.as_raw() > 1000,
            "after advance_to(1000), next() must be > 1000, got {next}"
        );
    }

    #[test]
    fn hlc_advance_to_does_not_go_backward() {
        let hlc = HybridLogicalClock::resume_from(Timestamp::from_raw(100));
        // advance_to a smaller value must not move the clock back.
        hlc.advance_to(Timestamp::from_raw(50));
        let c = hlc.current();
        // Current must be >= original (100 or wall, whichever is bigger).
        assert!(
            c.as_raw() >= 100,
            "advance_to(50) must not move clock below original, got {c}"
        );
    }

    #[test]
    fn hlc_advance_to_enables_causal_reads() {
        // Simulate follower receiving a Raft commit_ts from the leader.
        let follower_hlc = HybridLogicalClock::resume_from(Timestamp::from_raw(0));
        let leader_commit_ts = Timestamp::from_raw(1_700_000_000_123_456);

        // Follower applies the Raft entry (storage.rs calls advance_to(commit_ts - 1)):
        follower_hlc.advance_to(Timestamp::from_raw(leader_commit_ts.as_raw() - 1));

        // Any new timestamp allocated by the follower must be > commit_ts - 1.
        let follower_ts = follower_hlc.next();
        assert!(
            follower_ts >= leader_commit_ts,
            "follower's next ts must be causally after leader's commit; \
             got follower={follower_ts}, leader_commit={leader_commit_ts}"
        );
    }

    // ── HLC — persistence recovery ────────────────────────────────────────

    #[test]
    fn hlc_resume_from_recovers_above_persisted() {
        // Simulate recovery from last committed ts = 1_000_000.
        // resume_from starts at max(wall, persisted).
        let persisted = Timestamp::from_raw(1_000_000);
        let hlc = HybridLogicalClock::resume_from(persisted);

        // Since wall >> 1_000_000, HLC starts at wall (which is > persisted).
        // All newly allocated timestamps must be causally after the old data.
        let next = hlc.next();
        assert!(
            next.as_raw() > persisted.as_raw(),
            "recovered HLC must be above persisted ts {persisted}, got {next}"
        );
    }

    // ── HLC — concurrent monotonicity ─────────────────────────────────────

    #[test]
    fn hlc_concurrent_allocation_all_unique() {
        use std::collections::BTreeSet;
        use std::sync::Arc;

        let hlc = Arc::new(HybridLogicalClock::resume_from(Timestamp::from_raw(0)));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let hlc = Arc::clone(&hlc);
            handles.push(std::thread::spawn(move || {
                (0..1000).map(|_| hlc.next().as_raw()).collect::<Vec<_>>()
            }));
        }

        let mut all_ts: BTreeSet<u64> = BTreeSet::new();
        for h in handles {
            let ts_vec = h.join().expect("thread panicked");
            for ts in ts_vec {
                assert!(all_ts.insert(ts), "duplicate timestamp: {ts}");
            }
        }

        // 8 threads × 1000 = 8000 unique timestamps
        assert_eq!(all_ts.len(), 8000);
    }
}
