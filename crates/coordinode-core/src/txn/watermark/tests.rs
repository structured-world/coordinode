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
