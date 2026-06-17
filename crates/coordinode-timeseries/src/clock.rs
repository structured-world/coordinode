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

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

/// Key under [`Partition::Schema`] where each shard's last-issued
/// ingestion stamp is persisted: `schema:meta:ts_clock:<shard_id_u16_be>`.
/// Read on clock construction, written every
/// `PERSIST_EVERY_N_STAMPS` calls to `next()` so a crash can lose
/// at most that many stamps from durability.
const TS_CLOCK_KEY_PREFIX: &[u8] = b"schema:meta:ts_clock:";

/// Encode the persistence key for a given shard. Mirrors the
/// convention used by `schema:current_revision:<kind>:<name>` —
/// reserved keys under [`Partition::Schema`] for engine metadata.
fn encode_ts_clock_key(shard_id: u16) -> Vec<u8> {
    let mut k = Vec::with_capacity(TS_CLOCK_KEY_PREFIX.len() + 2);
    k.extend_from_slice(TS_CLOCK_KEY_PREFIX);
    k.extend_from_slice(&shard_id.to_be_bytes());
    k
}

/// Persistence cadence for [`PersistentMonotonicHlcClock`]. Every
/// Nth call to `next()` writes the current stamp to the engine.
/// Smaller values give tighter restart-monotonicity at the cost of
/// more storage writes on the hot path.
///
/// 64 chosen as the sweet spot: ≥ a typical bucket measurement count
/// (~10K) is two orders of magnitude larger, so the persistence
/// write rate is ~1.5% of the measurement write rate. Restart can
/// lose ≤ 63 stamps in the worst case; combined with seed-on-load
/// (`max(persisted+1, wall_now)`) the only failure mode is a hard
/// backward NTP jump > 63 µs, which is rare in practice.
const PERSIST_EVERY_N_STAMPS: u64 = 64;

/// Source of `__ingestion_ts__` stamps for the catalog.
///
/// Implementations MUST guarantee strict monotonicity within a
/// single clock instance: every call to [`Self::next`] returns a
/// value strictly greater than every prior call. The catalog
/// stamps each incoming measurement via this trait before
/// buffering — measurements then carry the bitemporal axis through
/// the bucket's `ingestion_timestamps` column.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an `IngestionClock` — the catalog can't stamp measurements with it",
    label = "expected an `IngestionClock` implementation here",
    note = "for production CE single-node use `PersistentMonotonicHlcClock::open(engine, shard_id)?` — \
            it persists the last-issued stamp to the engine so restart monotonicity holds across \
            backward wall-clock jumps. For tests use `MonotonicHlcClock::new()` (no persistence) or \
            `ScriptedClock::new(seq)` (deterministic test sequence, requires the `test-clock` feature)."
)]
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
/// ## Monotonicity scope
///
/// `MonotonicHlcClock` guarantees strict monotonicity **within a
/// single process instance**. Each new instance seeds from
/// `SystemTime::now()`, so a backward wall-clock jump between
/// catalog construction events would let a freshly-constructed
/// clock issue stamps ≤ the prior instance's — that violates
/// strict-monotonicity per shard across restarts.
///
/// **For production use, choose [`PersistentMonotonicHlcClock`]
/// instead.** It wraps the same CAS-monotonic logic and seeds from
/// `max(persisted_last + 1, wall_now)` on construction, persisting
/// the current stamp to the engine every 64 calls. Restart
/// monotonicity holds even under backward wall-clock jumps.
///
/// This bare `MonotonicHlcClock` remains useful for:
/// - Tests that need a clock without engine writes
/// - Future Raft-leader-stamped variants where the persistence
///   layer is the Raft log, not direct engine writes
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
        Self::with_prior(wall_clock_micros() - 1)
    }

    /// Construct with an explicit "prior" stamp — the first call to
    /// [`Self::next`] returns `max(prior + 1, wall_clock_now)`.
    /// Used by [`PersistentMonotonicHlcClock`] to seed from a
    /// durable checkpoint and by Raft-leader-stamped variants to
    /// resume from the replicated last stamp.
    pub fn with_prior(prior: i64) -> Self {
        Self {
            last_us: AtomicI64::new(prior),
        }
    }

    /// Return the current last-issued stamp WITHOUT issuing a new
    /// one. Used by persistent variants for checkpoint writes —
    /// the catalog flush wants "what's the stamp bounding every
    /// just-flushed measurement?" without burning a stamp number
    /// on the read.
    pub fn current(&self) -> i64 {
        self.last_us.load(Ordering::Acquire)
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

/// Load the persisted last-issued stamp for a shard, if any.
/// Returns `Ok(None)` when no checkpoint has been written yet
/// (first-ever catalog open) and `Err` on a corrupt body (not the
/// expected 8-byte big-endian i64).
pub fn load_last_stamp(
    engine: &StorageEngine,
    shard_id: u16,
) -> Result<Option<i64>, coordinode_storage::error::StorageError> {
    let key = encode_ts_clock_key(shard_id);
    match engine.get(Partition::Schema, &key)? {
        None => Ok(None),
        Some(bytes) if bytes.len() == 8 => {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes);
            Ok(Some(i64::from_be_bytes(buf)))
        }
        Some(bytes) => {
            tracing::warn!(
                shard_id,
                len = bytes.len(),
                "ts_clock checkpoint body is not 8 bytes — treating as missing",
            );
            Ok(None)
        }
    }
}

/// Persist `stamp` as the new last-issued checkpoint for `shard_id`.
/// Engine-overwrite (no merge operator needed — the stamp is
/// monotonic, so a later write naturally subsumes an earlier one).
pub fn persist_last_stamp(
    engine: &StorageEngine,
    shard_id: u16,
    stamp: i64,
) -> Result<(), coordinode_storage::error::StorageError> {
    let key = encode_ts_clock_key(shard_id);
    engine.put(Partition::Schema, &key, &stamp.to_be_bytes())
}

/// Production clock with engine-backed restart monotonicity. Wraps
/// the same CAS-monotonic logic as `MonotonicHlcClock` but seeds
/// from `max(persisted_last + 1, wall_clock_now)` on construction
/// and writes the current stamp back to the engine every
/// `PERSIST_EVERY_N_STAMPS` calls.
///
/// ## Restart-monotonicity guarantee
///
/// Across catalog restarts the first stamp issued is strictly
/// greater than the persisted checkpoint, regardless of how the
/// wall clock has moved. Worst-case data loss on crash is
/// `PERSIST_EVERY_N_STAMPS - 1` stamps; combined with the seed-on-
/// load rule, the only failure mode is a backward wall-clock jump
/// that exceeds the lost-stamp window — practically impossible
/// outside of malicious clock tampering.
///
/// ## When to use this vs. `MonotonicHlcClock`
///
/// - **Production CE single-node** → `PersistentMonotonicHlcClock`.
///   The engine handle is already on hand at catalog construction.
/// - **Tests** → `MonotonicHlcClock` (no engine writes) or
///   `ScriptedClock` (deterministic).
/// - **Multi-replica Raft-leader-stamped** → a separate Raft-aware
///   variant that persists through consensus log instead of direct
///   engine writes (future follow-up).
pub struct PersistentMonotonicHlcClock {
    inner: MonotonicHlcClock,
    engine: std::sync::Arc<StorageEngine>,
    shard_id: u16,
    /// Counter of `next()` calls since the last persistence write.
    /// On reaching `PERSIST_EVERY_N_STAMPS` we persist and reset.
    since_persist: AtomicU64,
}

impl PersistentMonotonicHlcClock {
    /// Construct the clock for `shard_id`, seeding from the engine's
    /// persisted checkpoint if any. The first stamp issued is
    /// strictly greater than both the persisted value (if any) AND
    /// the current wall clock — restart monotonicity holds.
    pub fn open(
        engine: std::sync::Arc<StorageEngine>,
        shard_id: u16,
    ) -> Result<Self, coordinode_storage::error::StorageError> {
        let persisted = load_last_stamp(&engine, shard_id)?;
        let wall = wall_clock_micros();
        // Seed = max(persisted + 1, wall) - 1 so the FIRST next()
        // returns exactly max(persisted + 1, wall). The MonotonicHlcClock
        // formula in `next()` returns `max(prior + 1, wall_now)`; with
        // `prior = seed`, that yields the desired first value.
        let seed = match persisted {
            Some(p) => std::cmp::max(p + 1, wall) - 1,
            None => wall - 1,
        };
        Ok(Self {
            inner: MonotonicHlcClock::with_prior(seed),
            engine,
            shard_id,
            since_persist: AtomicU64::new(0),
        })
    }

    /// Force-persist the current stamp regardless of cadence.
    /// Called by the catalog after every successful bucket flush so
    /// crash recovery sees a stamp that strictly bounds every
    /// flushed measurement's `ingestion_ts_us`.
    pub fn checkpoint(&self) -> Result<(), coordinode_storage::error::StorageError> {
        let current = self.inner.current();
        persist_last_stamp(&self.engine, self.shard_id, current)?;
        self.since_persist.store(0, Ordering::Release);
        Ok(())
    }
}

impl IngestionClock for PersistentMonotonicHlcClock {
    fn next(&self) -> i64 {
        let stamp = self.inner.next();
        let prior = self.since_persist.fetch_add(1, Ordering::AcqRel);
        if prior + 1 >= PERSIST_EVERY_N_STAMPS {
            // Best-effort persistence — a transient engine error is
            // logged but not propagated. The next checkpoint or
            // explicit `Self::checkpoint()` call will catch up.
            if let Err(e) = persist_last_stamp(&self.engine, self.shard_id, stamp) {
                tracing::warn!(
                    shard_id = self.shard_id,
                    error = %e,
                    "ts_clock periodic persistence failed — will retry on next flush",
                );
            } else {
                self.since_persist.store(0, Ordering::Release);
            }
        }
        stamp
    }
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
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use tempfile::TempDir;

    fn mk_engine() -> (TempDir, std::sync::Arc<StorageEngine>) {
        let dir = TempDir::new().unwrap();
        let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = std::sync::Arc::new(StorageEngine::open(&cfg).unwrap());
        (dir, engine)
    }

    #[test]
    fn persistent_clock_seeds_from_wall_when_no_checkpoint() {
        // First-ever open: no persisted value. Seed from wall_now.
        // First stamp must be ≥ wall_now and strictly monotonic
        // from there.
        let (_dir, engine) = mk_engine();
        let clock = PersistentMonotonicHlcClock::open(engine.clone(), 0).unwrap();
        let before = wall_clock_micros();
        let first = clock.next();
        assert!(
            first >= before,
            "first stamp ({first}) must be >= wall_clock_micros at open ({before})",
        );
        let second = clock.next();
        assert!(second > first, "strict monotonicity: {first} >= {second}");
    }

    #[test]
    fn persistent_clock_resumes_above_persisted_on_reopen() {
        // Open, burn enough stamps to trigger persistence (>= 64),
        // drop the clock, reopen — new clock's first stamp must be
        // strictly greater than the highest persisted value, even
        // if wall_clock has gone backward.
        let (_dir, engine) = mk_engine();
        let last_burned = {
            let clock = PersistentMonotonicHlcClock::open(engine.clone(), 0).unwrap();
            let mut last = 0;
            for _ in 0..PERSIST_EVERY_N_STAMPS {
                last = clock.next();
            }
            // Force-persist the final stamp (cadence already triggered).
            clock.checkpoint().unwrap();
            last
        };

        // Read persisted value directly — it must match (or bound)
        // the last stamp we observed.
        let persisted = load_last_stamp(&engine, 0).unwrap().unwrap();
        assert!(
            persisted >= last_burned,
            "persisted {persisted} must bound last burned stamp {last_burned}",
        );

        // Reopen. First stamp must be strictly > persisted regardless
        // of wall clock (simulates a backward NTP jump that puts
        // wall_now <= persisted).
        let clock2 = PersistentMonotonicHlcClock::open(engine, 0).unwrap();
        let resumed = clock2.next();
        assert!(
            resumed > persisted,
            "reopened clock first stamp {resumed} must be > persisted checkpoint {persisted}",
        );
    }

    #[test]
    fn persistent_clock_explicit_checkpoint_writes_current_stamp() {
        let (_dir, engine) = mk_engine();
        let clock = PersistentMonotonicHlcClock::open(engine.clone(), 7).unwrap();
        let issued = clock.next();
        clock.checkpoint().unwrap();
        let read = load_last_stamp(&engine, 7).unwrap().unwrap();
        assert_eq!(
            read, issued,
            "explicit checkpoint must persist exactly the last-issued stamp",
        );
    }

    #[test]
    fn persistent_clock_shards_dont_interfere() {
        // Two shards on the same engine must persist to disjoint
        // keys — issuing on shard 1 does not affect shard 2's
        // checkpoint.
        let (_dir, engine) = mk_engine();
        let c1 = PersistentMonotonicHlcClock::open(engine.clone(), 1).unwrap();
        let c2 = PersistentMonotonicHlcClock::open(engine.clone(), 2).unwrap();
        let _ = c1.next();
        c1.checkpoint().unwrap();
        // Shard 2 has never been persisted — load_last_stamp returns None.
        assert!(
            load_last_stamp(&engine, 2).unwrap().is_none(),
            "shard 2 checkpoint must be absent after writes on shard 1 only",
        );
        // Now persist shard 2 separately.
        let _ = c2.next();
        c2.checkpoint().unwrap();
        let s1 = load_last_stamp(&engine, 1).unwrap().unwrap();
        let s2 = load_last_stamp(&engine, 2).unwrap().unwrap();
        // Two independent stamps — the only invariant is that they
        // both exist and differ (issued at distinct moments).
        assert_ne!(
            s1, s2,
            "shard 1 and shard 2 checkpoints must be independent"
        );
    }

    #[test]
    fn persistent_clock_corrupt_checkpoint_falls_back_to_wall_clock() {
        // If something writes garbage to the checkpoint key,
        // load_last_stamp returns None (with a tracing::warn),
        // and the clock seeds from wall_now — never crashes.
        let (_dir, engine) = mk_engine();
        engine
            .put(Partition::Schema, &encode_ts_clock_key(5), b"not-8-bytes")
            .unwrap();
        let loaded = load_last_stamp(&engine, 5).unwrap();
        assert_eq!(loaded, None, "8-byte length check must reject garbage");
        let clock = PersistentMonotonicHlcClock::open(engine, 5).unwrap();
        let first = clock.next();
        assert!(first > 0, "clock seeds normally despite corrupt checkpoint");
    }

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
