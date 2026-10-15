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

#[test]
fn wait_for_fast_path_returns_immediately() {
    let wm = MaxAssignedWatermark::new(Timestamp::from_raw(1000));
    let before = Instant::now();
    let got = wm
        .wait_for(Timestamp::from_raw(800), Duration::from_millis(500))
        .expect("fast path ok");
    assert_eq!(got.as_raw(), 1000);
    // Fast path should complete in well under 10ms even under load.
    assert!(before.elapsed() < Duration::from_millis(10));
}

#[test]
fn wait_for_blocks_until_advance() {
    let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
    let wm2 = Arc::clone(&wm);

    // An "applier" that advances the watermark after ~50ms.
    let advancer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        wm2.advance(Timestamp::from_raw(500));
    });

    let before = Instant::now();
    let got = wm
        .wait_for(Timestamp::from_raw(500), Duration::from_millis(1000))
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
    advancer.join().unwrap();
}

#[test]
fn wait_for_times_out_when_applier_stalls() {
    let wm = MaxAssignedWatermark::new(Timestamp::from_raw(100));
    // Target is above current; nobody advances. Must hit timeout and
    // return `ErrTimeout`, NOT stale `Ok(100)`.
    let err = wm
        .wait_for(Timestamp::from_raw(500), Duration::from_millis(50))
        .expect_err("must time out, not return stale");
    let WaitError::Timeout {
        target,
        current,
        elapsed,
    } = err;
    assert_eq!(target, 500);
    assert_eq!(current, 100);
    assert!(elapsed >= Duration::from_millis(40));
}

#[test]
fn wait_for_wakes_multiple_concurrent_waiters() {
    let wm = MaxAssignedWatermark::new(Timestamp::ZERO);

    let mut handles = Vec::new();
    for target in [100, 200, 300, 400] {
        let wm_c = Arc::clone(&wm);
        handles.push(std::thread::spawn(move || {
            wm_c.wait_for(Timestamp::from_raw(target), Duration::from_millis(1000))
                .map(|ts| ts.as_raw())
        }));
    }

    // Single jump to 500 should wake all waiters (all targets ≤ 500).
    std::thread::sleep(Duration::from_millis(20));
    wm.advance(Timestamp::from_raw(500));

    for h in handles {
        let got = h.join().unwrap().expect("all waiters succeed");
        assert_eq!(got, 500);
    }
}

#[test]
fn wait_for_partial_advance_keeps_later_waiters_blocked() {
    let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
    let wm_fast = Arc::clone(&wm);
    let wm_slow = Arc::clone(&wm);

    let fast = std::thread::spawn(move || {
        wm_fast.wait_for(Timestamp::from_raw(100), Duration::from_millis(500))
    });
    let slow = std::thread::spawn(move || {
        wm_slow.wait_for(Timestamp::from_raw(1000), Duration::from_millis(100))
    });

    // Advance enough for fast but not slow.
    std::thread::sleep(Duration::from_millis(20));
    wm.advance(Timestamp::from_raw(200));

    let fast_res = fast.join().unwrap().expect("fast reaches target");
    assert_eq!(fast_res.as_raw(), 200);

    let WaitError::Timeout { current, .. } = slow
        .join()
        .unwrap()
        .expect_err("slow waiter should time out");
    assert_eq!(current, 200);
}

#[test]
fn wait_for_zero_target_is_trivial() {
    // Target=0 is always satisfied regardless of current.
    let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
    let got = wm
        .wait_for(Timestamp::ZERO, Duration::from_millis(10))
        .expect("zero target is trivially reached");
    assert_eq!(got.as_raw(), 0);
}

#[test]
fn wait_for_default_timeout_is_2s() {
    // Sanity check that the documented default matches the constant.
    assert_eq!(DEFAULT_WAIT_TIMEOUT, Duration::from_millis(2000));
}

#[test]
fn advance_notifies_even_when_no_waiters() {
    // No waiters exist: advance still moves the watermark.
    let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
    assert!(wm.advance(Timestamp::from_raw(42)));
    assert_eq!(wm.current().as_raw(), 42);
}

#[test]
fn concurrent_advances_preserve_maximum() {
    let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
    let mut handles = Vec::new();
    for i in 0..20_u64 {
        let wm_c = Arc::clone(&wm);
        handles.push(std::thread::spawn(move || {
            wm_c.advance(Timestamp::from_raw(i * 100));
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    // Final value must equal the maximum of all advances (1900).
    assert_eq!(wm.current().as_raw(), 1900);
}

/// An advance racing a waiter's check is never lost: many waiters, each
/// woken by an advance landing at a random point around its check, all
/// return well before their timeout.
#[test]
fn an_advance_racing_the_check_is_not_lost() {
    for round in 1..=200_u64 {
        let wm = MaxAssignedWatermark::new(Timestamp::ZERO);
        let waiter = {
            let wm = Arc::clone(&wm);
            std::thread::spawn(move || {
                wm.wait_for(Timestamp::from_raw(round), Duration::from_secs(5))
            })
        };
        wm.advance(Timestamp::from_raw(round));
        let got = waiter.join().unwrap().expect("a lost wake would time out");
        assert_eq!(got.as_raw(), round);
    }
}
