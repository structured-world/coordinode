use super::*;

#[test]
fn ready_round_trip() {
    let h = HealthSignal::new_ready();
    assert_eq!(h.snapshot(), IndexHealthState::Ready { indexed_hlc: 0 });
    assert!(h.snapshot().is_ready());
    assert_eq!(h.snapshot().label(), "ready");
}

#[test]
fn rebuilding_publishes_progress_and_eta() {
    let h = HealthSignal::new_rebuilding();
    h.report_rebuild_progress(0.42, 12_345);
    match h.snapshot() {
        IndexHealthState::Rebuilding {
            progress, eta_ms, ..
        } => {
            // Fixed-point round-trip: 0.42 * 10_000 = 4200 → 0.42.
            assert!((progress - 0.42).abs() < 1e-3, "got {progress}");
            assert_eq!(eta_ms, 12_345);
        }
        other => panic!("expected Rebuilding, got {other:?}"),
    }
}

#[test]
fn report_progress_clamps_out_of_range() {
    let h = HealthSignal::new_rebuilding();

    h.report_rebuild_progress(1.5, 0);
    match h.snapshot() {
        IndexHealthState::Rebuilding { progress, .. } => assert_eq!(progress, 1.0),
        other => panic!("expected Rebuilding, got {other:?}"),
    }

    h.report_rebuild_progress(-0.3, 0);
    match h.snapshot() {
        IndexHealthState::Rebuilding { progress, .. } => assert_eq!(progress, 0.0),
        other => panic!("expected Rebuilding, got {other:?}"),
    }
}

#[test]
fn rebuilding_then_ready_clears_progress() {
    let h = HealthSignal::new_rebuilding();
    h.report_rebuild_progress(0.7, 5_000);
    h.mark_ready();
    assert!(h.snapshot().is_ready());
}

#[test]
fn offline_carries_reason() {
    let h = HealthSignal::new_ready();
    h.mark_offline("segment_lost");
    match h.snapshot() {
        IndexHealthState::Offline { reason } => assert_eq!(reason, "segment_lost"),
        other => panic!("expected Offline, got {other:?}"),
    }
    assert!(h.snapshot().is_offline());
}

#[test]
fn offline_back_to_ready_clears_reason() {
    let h = HealthSignal::new_ready();
    h.mark_offline("disk_io");
    h.mark_ready();
    assert!(h.snapshot().is_ready());
    // Now go to Offline again and confirm the old reason did not leak.
    h.mark_offline("manual_disable");
    match h.snapshot() {
        IndexHealthState::Offline { reason } => assert_eq!(reason, "manual_disable"),
        other => panic!("expected Offline, got {other:?}"),
    }
}

#[test]
fn indexed_hlc_advances_monotonically() {
    let h = HealthSignal::new_ready();
    assert_eq!(h.indexed_hlc(), 0);
    h.advance_indexed_hlc(100);
    assert_eq!(h.indexed_hlc(), 100);
    // Out-of-order / replayed apply must not move the watermark back.
    h.advance_indexed_hlc(50);
    assert_eq!(h.indexed_hlc(), 100);
    h.advance_indexed_hlc(150);
    assert_eq!(h.indexed_hlc(), 150);
    // Surfaced through the snapshot's Ready variant.
    assert_eq!(h.snapshot(), IndexHealthState::Ready { indexed_hlc: 150 });
}

#[test]
fn indexed_hlc_survives_state_transitions() {
    let h = HealthSignal::new_ready();
    h.advance_indexed_hlc(500);
    // A rebuild still reports the watermark it has folded so far.
    h.report_rebuild_progress(0.3, 1_000);
    assert_eq!(h.snapshot().indexed_hlc(), Some(500));
    h.advance_indexed_hlc(700);
    // mark_ready leaves the watermark intact (orthogonal axes).
    h.mark_ready();
    assert_eq!(h.snapshot(), IndexHealthState::Ready { indexed_hlc: 700 });
    // Offline has no watermark.
    h.mark_offline("seg");
    assert_eq!(h.snapshot().indexed_hlc(), None);
}

#[test]
fn concurrent_readers_observe_consistent_state() {
    // Build path mutates state from one writer; search path readers
    // observe consistent snapshots (never see a discriminant from one
    // state with a progress from another).
    use std::sync::atomic::AtomicBool;
    use std::thread;

    let h = HealthSignal::new_rebuilding();
    let stop = Arc::new(AtomicBool::new(false));

    let mut readers = Vec::new();
    for _ in 0..4 {
        let h = h.clone();
        let stop = stop.clone();
        readers.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                // Every snapshot must encode a self-consistent state:
                // if it's Rebuilding, the progress must be in range.
                if let IndexHealthState::Rebuilding { progress, .. } = h.snapshot() {
                    assert!((0.0..=1.0).contains(&progress));
                }
            }
        }));
    }

    for i in 0..1_000u32 {
        let p = (i as f32 / 1_000.0).min(1.0);
        h.report_rebuild_progress(p, 1_000 - i as u64);
        h.advance_indexed_hlc(i as u64);
    }
    h.mark_ready();
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }
}

#[test]
fn encode_decode_round_trip() {
    for kind in 0..=2u32 {
        for progress in [0, 1, 5_000, 9_999, 10_000] {
            let raw = encode(kind, progress);
            assert_eq!(decode(raw), (kind, progress));
        }
    }
}
