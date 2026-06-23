use super::*;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
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
