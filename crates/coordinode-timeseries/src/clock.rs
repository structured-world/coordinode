//! Ingestion clock — engine-assigned `__ingestion_ts__` stamp source
//! for time-series measurements (bitemporal axis #2 per ADR-027).
//!
//! ## Why a trait
//!
//! Production wants a Raft-leader-stamped HLC: the cluster's leader
//! emits the ingestion stamp so every replica sees the same value.
//! Tests want a deterministic clock so timestamps don't depend on
//! `SystemTime::now()` drift. Both shapes implement the same
//! [`IngestionClock`] trait; the catalog holds a trait object and
//! never knows which one is in use.
//!
//! ## Invariants
//!
//! - **Strictly monotonic.** Every call to [`IngestionClock::next`]
//!   returns a value strictly greater than every prior call within
//!   the same logical shard. Hard requirement for `AS OF
//!   INGESTION_TIME $t` query semantics — a non-monotonic clock
//!   would let two measurements stamped at the "same instant" be
//!   returned in opposite orders across replays.
//! - **Never user-supplied, never user-mutable.** The stamp is
//!   engine-assigned at the catalog layer; the OpenCypher writer
//!   path has no way to set `__ingestion_ts__`. Reserved field name.
//! - **Per-shard, not per-process.** Multiple catalogs on the same
//!   process (one per shard) get distinct clock instances; they do
//!   not synchronize across shards.

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Source of `__ingestion_ts__` stamps for the catalog.
///
/// Implementations MUST guarantee strict monotonicity within a
/// single clock instance: every call to [`Self::next`] returns a
/// value strictly greater than every prior call. The catalog
/// stamps each incoming measurement via this trait before
/// buffering — measurements then carry the bitemporal axis through
/// the bucket's `ingestion_timestamps` column.
pub trait IngestionClock: Send + Sync {
    /// Return the next stamp. Microseconds since UNIX epoch.
    /// Strictly monotonic; non-decreasing is NOT sufficient.
    fn next(&self) -> i64;
}

/// Default production implementation. Uses `SystemTime::now()` as
/// the source of truth with an atomic monotonic guard — if two
/// rapid calls land on the same micro­second tick, the second one
/// returns `prior + 1` rather than the equal wall-clock value.
///
/// In a multi-replica deployment the catalog should be constructed
/// against a Raft-leader-stamped clock instead (the leader emits
/// the stamp, followers replay it). This default is correct for
/// single-process / single-shard tests and for CE single-node
/// deployments.
///
/// ## Monotonicity scope and the restart gap
///
/// `MonotonicHlcClock` guarantees strict monotonicity **within a
/// single process instance**. Each new instance seeds from
/// `SystemTime::now()`. If the host wall clock moves backward
/// between catalog construction events — NTP correction, container
/// VM time slip, manual `date` adjustment — a freshly-constructed
/// clock can issue a stamp ≤ a stamp issued by the prior instance.
/// That violates strict-monotonicity per shard across restarts.
///
/// Production CE single-node deployments protect against this by
/// persisting the last-issued stamp to durable state on each catalog
/// flush and seeding the next instance with `max(persisted + 1,
/// wall_clock_now)`. That persistence layer is a follow-up; until
/// it lands, the failure mode is: after a backward wall-clock jump,
/// the first N catalog stamps may collide with prior values, which
/// makes `AS OF INGESTION_TIME` queries non-deterministic for
/// timestamps near the boundary. Single-node CE in practice doesn't
/// see backward NTP jumps frequently; the gap is documented here so
/// the follow-up has a known landing site rather than being lost.
///
/// Multi-replica deployments do NOT have this problem — the Raft-
/// leader-stamped variant persists last-issued through the consensus
/// log automatically (replicas replay leader stamps, so on
/// leadership change the new leader resumes from
/// `replicated_last + 1`, immune to local wall-clock state).
pub struct MonotonicHlcClock {
    /// Greatest stamp ever returned. The next call returns
    /// `max(prior + 1, SystemTime::now())`.
    last_us: AtomicI64,
}

impl MonotonicHlcClock {
    /// Construct a fresh clock seeded at `SystemTime::now()`. The
    /// first call to [`Self::next`] returns the seed value (the
    /// stamp `prior + 1` rule kicks in from the second call
    /// onward).
    pub fn new() -> Self {
        let seed = wall_clock_micros();
        Self {
            last_us: AtomicI64::new(seed - 1),
        }
    }
}

impl Default for MonotonicHlcClock {
    fn default() -> Self {
        Self::new()
    }
}

impl IngestionClock for MonotonicHlcClock {
    fn next(&self) -> i64 {
        loop {
            let prior = self.last_us.load(Ordering::Acquire);
            let wall = wall_clock_micros();
            let candidate = if wall > prior { wall } else { prior + 1 };
            // CAS in the new stamp. On lost-race, retry — another
            // thread bumped `last_us` past `prior` so our candidate
            // may no longer be monotonic.
            if self
                .last_us
                .compare_exchange(prior, candidate, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return candidate;
            }
        }
    }
}

fn wall_clock_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// Test clock — returns a sequence the test fully controls. Each
/// call to [`Self::next`] returns the next entry from the
/// pre-loaded vector; panics if exhausted (test bug — produced
/// too few stamps for the workload).
///
/// Wrap in `Arc` if shared across threads. Single-threaded test
/// usage can pass by reference.
#[cfg(any(test, feature = "test-clock"))]
pub struct ScriptedClock {
    sequence: std::sync::Mutex<std::collections::VecDeque<i64>>,
}

#[cfg(any(test, feature = "test-clock"))]
impl ScriptedClock {
    /// Build from an explicit sequence. The first call to
    /// [`Self::next`] returns `seq[0]`, second `seq[1]`, etc.
    /// Test must size the sequence at least as long as the
    /// expected number of measurements (panic otherwise).
    pub fn new(seq: Vec<i64>) -> Self {
        Self {
            sequence: std::sync::Mutex::new(seq.into_iter().collect()),
        }
    }
}

#[cfg(any(test, feature = "test-clock"))]
impl IngestionClock for ScriptedClock {
    // ScriptedClock is a test fixture — the panic on exhaustion is
    // load-bearing test-failure signal, not a production code path.
    // The crate-wide deny(clippy::expect_used) targets production
    // code; this allow is scoped to one line.
    #[allow(clippy::expect_used)]
    fn next(&self) -> i64 {
        self.sequence
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop_front()
            .expect("ScriptedClock exhausted — test under-sized the stamp sequence")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_clock_returns_strictly_increasing_stamps() {
        let clock = MonotonicHlcClock::new();
        let mut last = clock.next();
        for _ in 0..1000 {
            let next = clock.next();
            assert!(next > last, "non-monotonic: {last} >= {next}");
            last = next;
        }
    }

    #[test]
    fn monotonic_clock_handles_clock_skew_via_atomic_guard() {
        // Simulate fast successive calls within the same wall-clock
        // microsecond — the atomic guard must still produce strictly
        // monotonic values.
        let clock = MonotonicHlcClock::new();
        let stamps: Vec<i64> = (0..100).map(|_| clock.next()).collect();
        for w in stamps.windows(2) {
            assert!(
                w[1] > w[0],
                "non-monotonic in tight loop: {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn scripted_clock_returns_pre_loaded_sequence_in_order() {
        let clock = ScriptedClock::new(vec![100, 200, 300]);
        assert_eq!(clock.next(), 100);
        assert_eq!(clock.next(), 200);
        assert_eq!(clock.next(), 300);
    }

    #[test]
    #[should_panic(expected = "ScriptedClock exhausted")]
    fn scripted_clock_panics_when_exhausted() {
        let clock = ScriptedClock::new(vec![100]);
        clock.next();
        clock.next(); // boom
    }

    #[test]
    fn monotonic_clock_strictly_monotonic_across_threads() {
        use std::sync::Arc;
        use std::thread;
        let clock = Arc::new(MonotonicHlcClock::new());
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let c = Arc::clone(&clock);
                thread::spawn(move || {
                    let mut out = Vec::with_capacity(100);
                    for _ in 0..100 {
                        out.push(c.next());
                    }
                    out
                })
            })
            .collect();
        let mut all: Vec<i64> = threads
            .into_iter()
            .flat_map(|t| t.join().unwrap())
            .collect();
        all.sort_unstable();
        // After sorting, every adjacent pair must differ (no duplicates).
        for w in all.windows(2) {
            assert_ne!(w[0], w[1], "duplicate stamp issued: {}", w[0]);
        }
    }
}
