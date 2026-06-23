use super::*;

fn src(file: &str, line: u32, func: &str) -> SourceContext {
    SourceContext::new(file, line, func)
}

/// Below threshold — no alert.
#[test]
fn below_threshold_no_alert() {
    let detector = NPlus1Detector::with_config(NPlus1Config {
        threshold: 5,
        window_secs: 1,
    });
    let source = src("app.rs", 10, "handler");

    for _ in 0..4 {
        let alert = detector.record(0xABC, "MATCH (n) RETURN n", &source);
        assert!(alert.is_none(), "should not alert below threshold");
    }
}

/// Reaching threshold triggers alert exactly once.
#[test]
fn threshold_triggers_once() {
    let detector = NPlus1Detector::with_config(NPlus1Config {
        threshold: 5,
        window_secs: 60, // long window so timestamps don't expire
    });
    let source = src("app.rs", 10, "handler");

    // 4 calls — no alert
    for _ in 0..4 {
        assert!(detector
            .record(0xABC, "MATCH (n) RETURN n", &source)
            .is_none());
    }

    // 5th call — triggers alert
    let alert = detector.record(0xABC, "MATCH (n) RETURN n", &source);
    assert!(alert.is_some(), "should alert at threshold");

    let alert = alert.unwrap();
    assert_eq!(alert.fingerprint, 0xABC);
    assert_eq!(alert.call_count, 5);
    assert_eq!(alert.source_file, "app.rs");
    assert_eq!(alert.source_line, 10);
    assert_eq!(alert.source_function, "handler");
    assert_eq!(alert.suggestion.severity, Severity::Warning);

    // 6th call — no duplicate alert (already alerted in this window)
    assert!(detector
        .record(0xABC, "MATCH (n) RETURN n", &source)
        .is_none());
}

/// Different sources are tracked independently.
#[test]
fn different_sources_independent() {
    let detector = NPlus1Detector::with_config(NPlus1Config {
        threshold: 5,
        window_secs: 60,
    });
    let src_a = src("a.rs", 1, "fn_a");
    let src_b = src("b.rs", 2, "fn_b");

    // 4 calls from src_a, 2 from src_b
    for _ in 0..4 {
        assert!(detector.record(0xABC, "query", &src_a).is_none());
    }
    for _ in 0..2 {
        assert!(detector.record(0xABC, "query", &src_b).is_none());
    }

    // 5th call from src_a — triggers alert
    assert!(detector.record(0xABC, "query", &src_a).is_some());

    // src_b still below threshold (only 2 calls)
    assert!(!detector.is_flagged(0xABC, &src_b));
}

/// Different fingerprints are tracked independently.
#[test]
fn different_fingerprints_independent() {
    let detector = NPlus1Detector::with_config(NPlus1Config {
        threshold: 5,
        window_secs: 60,
    });
    let source = src("app.rs", 10, "handler");

    // 4 calls for each fingerprint
    for _ in 0..4 {
        assert!(detector.record(0x111, "query1", &source).is_none());
    }
    for _ in 0..2 {
        assert!(detector.record(0x222, "query2", &source).is_none());
    }

    // 5th call for 0x111 — triggers
    assert!(detector.record(0x111, "query1", &source).is_some());

    // 0x222 still below threshold
    assert!(!detector.is_flagged(0x222, &source));
}

/// is_flagged reflects current state.
#[test]
fn is_flagged_tracks_state() {
    let detector = NPlus1Detector::with_config(NPlus1Config {
        threshold: 3,
        window_secs: 60,
    });
    let source = src("app.rs", 10, "handler");

    assert!(!detector.is_flagged(0xABC, &source));

    for _ in 0..3 {
        detector.record(0xABC, "query", &source);
    }

    assert!(detector.is_flagged(0xABC, &source));
}

/// active_alerts returns all currently flagged pairs.
#[test]
fn active_alerts_returns_flagged() {
    let detector = NPlus1Detector::with_config(NPlus1Config {
        threshold: 2,
        window_secs: 60,
    });
    let src_a = src("a.rs", 1, "fn_a");
    let src_b = src("b.rs", 2, "fn_b");

    // Trigger both
    for _ in 0..2 {
        detector.record(0x111, "q1", &src_a);
        detector.record(0x222, "q2", &src_b);
    }

    let alerts = detector.active_alerts();
    assert_eq!(alerts.len(), 2);
}

/// reset clears all tracking state.
#[test]
fn reset_clears_state() {
    let detector = NPlus1Detector::with_config(NPlus1Config {
        threshold: 2,
        window_secs: 60,
    });
    let source = src("app.rs", 10, "handler");

    for _ in 0..2 {
        detector.record(0xABC, "query", &source);
    }

    assert!(detector.is_flagged(0xABC, &source));

    detector.reset();

    assert!(!detector.is_flagged(0xABC, &source));
    assert!(detector.active_alerts().is_empty());
}

/// Suggestion contains UNWIND rewrite.
#[test]
fn suggestion_has_unwind_rewrite() {
    let detector = NPlus1Detector::with_config(NPlus1Config {
        threshold: 2,
        window_secs: 60,
    });
    let source = src("app.rs", 10, "handler");

    detector.record(0xABC, "MATCH (n:User {id: $}) RETURN n", &source);
    let alert = detector
        .record(0xABC, "MATCH (n:User {id: $}) RETURN n", &source)
        .expect("should trigger");

    assert!(
        alert
            .suggestion
            .rewritten_query
            .as_ref()
            .unwrap()
            .contains("UNWIND"),
        "rewrite should suggest UNWIND"
    );
}

/// Window expiration clears the flag — after the window passes,
/// the N+1 alert is no longer active.
#[test]
fn window_expiration_clears_flag() {
    // Use a custom detector with millisecond-level window via Duration directly
    let detector = NPlus1Detector {
        inner: Mutex::new(DetectorInner {
            entries: HashMap::new(),
        }),
        config: NPlus1Config {
            threshold: 3,
            window_secs: 1, // not used — overridden by window field
        },
        window: std::time::Duration::from_millis(50), // 50ms window
    };
    let source = src("app.rs", 10, "handler");

    // Record 3 calls quickly (triggers within 50ms window)
    for _ in 0..3 {
        detector.record(0xABC, "query", &source);
    }
    assert!(detector.is_flagged(0xABC, &source));

    // Wait for window to expire
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Record one more — prunes expired timestamps, only 1 in window now
    detector.record(0xABC, "query", &source);

    assert!(
        !detector.is_flagged(0xABC, &source),
        "should no longer be flagged after window expiration"
    );
}

/// Alert fires again after window resets.
#[test]
fn alert_fires_again_after_window_reset() {
    let detector = NPlus1Detector {
        inner: Mutex::new(DetectorInner {
            entries: HashMap::new(),
        }),
        config: NPlus1Config {
            threshold: 3,
            window_secs: 1,
        },
        window: std::time::Duration::from_millis(50),
    };
    let source = src("app.rs", 10, "handler");

    // First window: trigger
    for _ in 0..2 {
        detector.record(0xABC, "query", &source);
    }
    let alert1 = detector.record(0xABC, "query", &source);
    assert!(alert1.is_some(), "first window should trigger");

    // Let window expire
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Second window: trigger again
    for _ in 0..2 {
        detector.record(0xABC, "query", &source);
    }
    let alert2 = detector.record(0xABC, "query", &source);
    assert!(
        alert2.is_some(),
        "second window should trigger again after reset"
    );
}

/// Concurrent access doesn't panic.
#[test]
fn concurrent_recording() {
    use std::sync::Arc;
    use std::thread;

    let detector = Arc::new(NPlus1Detector::with_config(NPlus1Config {
        threshold: 50,
        window_secs: 60,
    }));

    let mut handles = vec![];
    for t in 0..4 {
        let det = Arc::clone(&detector);
        handles.push(thread::spawn(move || {
            let source = src(&format!("thread_{t}.rs"), t as u32, "worker");
            for _ in 0..100 {
                det.record(0xABC, "query", &source);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }
}
