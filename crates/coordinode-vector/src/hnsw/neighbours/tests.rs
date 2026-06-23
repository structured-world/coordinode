use super::*;

#[test]
fn empty_list_has_zero_len() {
    let list: AtomicNeighbourList<8> = AtomicNeighbourList::new();
    assert_eq!(list.len(), 0);
    assert!(list.snapshot().is_empty());
    assert_eq!(list.capacity(), 8);
}

#[test]
fn set_then_snapshot_roundtrip() {
    let list: AtomicNeighbourList<8> = AtomicNeighbourList::new();
    list.set(&[10, 20, 30]);
    assert_eq!(list.len(), 3);
    assert_eq!(list.snapshot(), vec![10, 20, 30]);
}

#[test]
fn set_overwrites_previous() {
    let list: AtomicNeighbourList<8> = AtomicNeighbourList::new();
    list.set(&[1, 2, 3, 4, 5]);
    list.set(&[7, 8]);
    assert_eq!(list.len(), 2);
    assert_eq!(list.snapshot(), vec![7, 8]);
}

#[test]
fn set_to_empty_clears_list() {
    let list: AtomicNeighbourList<4> = AtomicNeighbourList::new();
    list.set(&[1, 2, 3]);
    list.set(&[]);
    assert_eq!(list.len(), 0);
    assert!(list.snapshot().is_empty());
}

#[test]
fn snapshot_into_reuses_buffer() {
    let list: AtomicNeighbourList<8> = AtomicNeighbourList::new();
    list.set(&[100, 200, 300]);
    let mut buf = Vec::with_capacity(32);
    let cap_before = buf.capacity();
    list.snapshot_into(&mut buf);
    assert_eq!(buf, vec![100, 200, 300]);
    // Buffer kept its allocation.
    assert!(buf.capacity() >= cap_before);
}

#[test]
fn set_truncates_to_capacity_in_release() {
    // Build with a 4-slot capacity, pass 6 ids → debug build panics,
    // release build silently truncates to 4. Cover the silent path with
    // a debug_assertions guard.
    let list: AtomicNeighbourList<4> = AtomicNeighbourList::new();
    if cfg!(debug_assertions) {
        let result = std::panic::catch_unwind(|| list.set(&[1, 2, 3, 4, 5, 6]));
        assert!(result.is_err(), "debug build must catch over-capacity set");
    } else {
        list.set(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(list.len(), 4);
        assert_eq!(list.snapshot(), vec![1, 2, 3, 4]);
    }
}

#[test]
fn concurrent_readers_safe_under_single_writer() {
    // C1 contract: single writer + many readers. Run a stress test
    // that interleaves writes and reads to verify no torn snapshots.
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::thread;

    let list: Arc<AtomicNeighbourList<16>> = Arc::new(AtomicNeighbourList::new());
    let stop = Arc::new(AtomicBool::new(false));

    let mut readers = Vec::new();
    for _ in 0..4 {
        let list = list.clone();
        let stop = stop.clone();
        readers.push(thread::spawn(move || {
            let mut buf = Vec::with_capacity(16);
            while !stop.load(Ordering::Relaxed) {
                list.snapshot_into(&mut buf);
                // Every value seen must be a valid neighbour id (never the
                // sentinel) — the Release/Acquire pair guarantees this.
                for &v in &buf {
                    assert_ne!(v, EMPTY);
                }
            }
        }));
    }

    // Writer: rotate through different lengths and values.
    for i in 0..2_000u64 {
        let payload: Vec<u64> = (0..(i % 16)).map(|k| i * 100 + k).collect();
        list.set(&payload);
    }
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }
}

#[test]
fn cas_append_grows_list_in_order() {
    let list: AtomicNeighbourList<8> = AtomicNeighbourList::new();
    for i in 0..5u64 {
        assert!(list.cas_append(i + 10), "append at {i} should succeed");
    }
    assert_eq!(list.snapshot(), vec![10, 11, 12, 13, 14]);
}

#[test]
fn cas_append_returns_false_when_full() {
    let list: AtomicNeighbourList<4> = AtomicNeighbourList::new();
    for i in 0..4u64 {
        assert!(list.cas_append(i));
    }
    // Fifth append must fail.
    assert!(!list.cas_append(99));
    assert_eq!(list.len(), 4);
    assert_eq!(list.snapshot(), vec![0, 1, 2, 3]);
    // Re-attempting also fails; list state unchanged.
    assert!(!list.cas_append(100));
    assert_eq!(list.len(), 4);
}

#[test]
fn cas_append_concurrent_writers_keep_len_consistent() {
    // C3 stress test: many threads call cas_append concurrently. Each
    // appended id is unique. Final list size must equal (a) the number
    // of attempted appends if that's ≤ capacity, or (b) capacity
    // exactly with a stable subset of attempted ids.
    use std::sync::Arc;
    use std::thread;

    const CAP: usize = 64;
    let list: Arc<AtomicNeighbourList<CAP>> = Arc::new(AtomicNeighbourList::new());

    let n_threads = 8;
    let per_thread = 8; // 8 × 8 = 64 = CAP, fits exactly.

    let mut handles = Vec::new();
    for t in 0..n_threads {
        let list = list.clone();
        handles.push(thread::spawn(move || {
            for i in 0..per_thread {
                let id = (t * 1_000 + i) as u64;
                // Spin until either succeeded or the list is full.
                let ok = list.cas_append(id);
                if !ok {
                    // Capacity reached — that's a valid outcome under
                    // contention; the test below verifies invariants.
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let snap = list.snapshot();
    assert!(snap.len() <= CAP, "len {} > capacity {CAP}", snap.len(),);
    // Every observed id must be unique — no duplicate slot assignment.
    let mut seen = std::collections::HashSet::new();
    for &id in &snap {
        assert!(seen.insert(id), "duplicate id {id} in concurrent append");
    }
    // With 64 appends and 64 slots, all must land if no thread bailed
    // early. (Threads only bail on `cas_append == false`, which only
    // happens past capacity, which only happens after CAP successful
    // appends → impossible here.)
    assert_eq!(snap.len(), CAP, "all 64 unique ids must land");
}

#[test]
fn cas_append_concurrent_overflow_caps_at_capacity() {
    // Variant: more appends than capacity. List must end with exactly
    // CAP unique ids, every observed id originating from some thread.
    use std::sync::Arc;
    use std::thread;

    const CAP: usize = 32;
    let list: Arc<AtomicNeighbourList<CAP>> = Arc::new(AtomicNeighbourList::new());

    let n_threads = 8;
    let per_thread = 16; // 8 × 16 = 128 > 32.

    let mut handles = Vec::new();
    for t in 0..n_threads {
        let list = list.clone();
        handles.push(thread::spawn(move || {
            let mut succeeded = 0;
            for i in 0..per_thread {
                let id = (t * 1_000 + i) as u64;
                if list.cas_append(id) {
                    succeeded += 1;
                }
            }
            succeeded
        }));
    }
    let total_success: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(total_success, CAP, "exactly CAP appends must succeed");

    let snap = list.snapshot();
    assert_eq!(snap.len(), CAP);
    let unique: std::collections::HashSet<u64> = snap.into_iter().collect();
    assert_eq!(unique.len(), CAP, "all entries unique");
}

#[test]
fn replace_overwrites_existing() {
    // C3 day 1: replace is set() under the per-list single-writer gate.
    // Tested as an alias here; concurrency contract documented on the
    // method itself.
    let list: AtomicNeighbourList<8> = AtomicNeighbourList::new();
    list.cas_append(1);
    list.cas_append(2);
    list.cas_append(3);
    list.replace(&[99, 88, 77, 66]);
    assert_eq!(list.snapshot(), vec![99, 88, 77, 66]);
}

#[test]
fn concurrent_cas_append_and_snapshot_no_torn_state() {
    // Stress: one writer streams cas_append while N readers snapshot.
    // Every observed id must be a valid input (no garbage from a
    // half-written slot), and the writer's monotonic insert order
    // must be preserved in the final snapshot (readers see a prefix).
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::thread;

    const CAP: usize = 64;
    let list: Arc<AtomicNeighbourList<CAP>> = Arc::new(AtomicNeighbourList::new());
    let stop = Arc::new(AtomicBool::new(false));

    let inputs: Vec<u64> = (1000..1000 + CAP as u64).collect();
    let input_set: std::collections::HashSet<u64> = inputs.iter().copied().collect();

    let mut reader_handles = Vec::new();
    for _ in 0..6 {
        let list = list.clone();
        let stop = stop.clone();
        let input_set = input_set.clone();
        reader_handles.push(thread::spawn(move || {
            let mut buf = Vec::with_capacity(CAP);
            while !stop.load(Ordering::Relaxed) {
                list.snapshot_into(&mut buf);
                for &v in &buf {
                    // Sentinel must not surface (snapshot filters it).
                    assert_ne!(v, EMPTY);
                    // Every observed id must be one of the writer's
                    // inputs — no torn read of a half-stored slot.
                    assert!(
                        input_set.contains(&v),
                        "garbage id {v} observed in snapshot",
                    );
                }
            }
        }));
    }

    // Single writer streams all inputs.
    for &v in &inputs {
        assert!(list.cas_append(v));
    }
    // Capacity reached — further appends must fail.
    assert!(!list.cas_append(99_999));

    stop.store(true, Ordering::Relaxed);
    for h in reader_handles {
        h.join().expect("reader thread panicked");
    }

    // Final state: all CAP inputs present in insertion order.
    let final_snap = list.snapshot();
    assert_eq!(final_snap, inputs);
}
