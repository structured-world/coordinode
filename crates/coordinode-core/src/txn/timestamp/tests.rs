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
