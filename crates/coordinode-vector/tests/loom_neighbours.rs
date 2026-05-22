//! Loom model-check campaign for [`AtomicNeighbourList`].
//!
//! Loom runs each test under every interleaving of memory accesses that
//! the C++ memory model (inherited by Rust) considers legal. Where
//! `std::thread` stress testing on x86 is probabilistic (a race may never
//! manifest in practice) and weakly-ordered architectures like ARM can
//! still observe orderings that x86 hides, loom **enumerates** all
//! legal interleavings for a tiny scenario and proves invariants hold
//! across every one of them.
//!
//! Build with:
//!
//! ```bash
//! RUSTFLAGS="--cfg loom" cargo test --test loom_neighbours --release
//! ```
//!
//! Loom is slow — runs are O(n!) in the number of atomic accesses. The
//! scenarios below stick to two threads with ~3 atomic operations each,
//! which loom can exhaustively enumerate in seconds. Larger interleavings
//! must use `LOOM_MAX_PREEMPTIONS=3` (default) and small per-test budgets.

#![cfg(loom)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_vector::hnsw::AtomicNeighbourList;
use loom::sync::Arc;
use loom::thread;

const CAP: usize = 4;

/// One writer streams cas_append; one reader takes a snapshot. Reader
/// must observe a valid prefix of the writer's appends — never garbage,
/// never the EMPTY sentinel, never out-of-order with respect to len.
#[test]
fn cas_append_writer_vs_snapshot_reader() {
    loom::model(|| {
        let list: Arc<AtomicNeighbourList<CAP>> = Arc::new(AtomicNeighbourList::new());

        let writer = {
            let list = list.clone();
            thread::spawn(move || {
                // Two appends, monotonically increasing ids.
                assert!(list.cas_append(10));
                assert!(list.cas_append(20));
            })
        };

        let reader = {
            let list = list.clone();
            thread::spawn(move || {
                let snap = list.snapshot();
                // Reader sees either {}, {10}, or {10, 20} — but never
                // {20} alone (must observe writer's program order) and
                // never any value other than 10/20.
                for &v in &snap {
                    assert!(v == 10 || v == 20, "garbage id {v}");
                }
                if snap.len() == 2 {
                    assert_eq!(snap, vec![10, 20]);
                } else if snap.len() == 1 {
                    assert_eq!(snap[0], 10, "out-of-order observation: {snap:?}");
                }
            })
        };

        writer.join().unwrap();
        reader.join().unwrap();

        // Final state must be exactly {10, 20}.
        assert_eq!(list.snapshot(), vec![10, 20]);
    });
}

/// Two concurrent writers each calling `cas_append` once. The final list
/// must contain exactly both ids, never duplicates, never missing one.
/// This is the core C3 multi-writer correctness property.
#[test]
fn concurrent_cas_append_no_lost_writes() {
    loom::model(|| {
        let list: Arc<AtomicNeighbourList<CAP>> = Arc::new(AtomicNeighbourList::new());

        let a = {
            let list = list.clone();
            thread::spawn(move || {
                assert!(list.cas_append(1));
            })
        };
        let b = {
            let list = list.clone();
            thread::spawn(move || {
                assert!(list.cas_append(2));
            })
        };

        a.join().unwrap();
        b.join().unwrap();

        let mut snap = list.snapshot();
        snap.sort();
        assert_eq!(snap, vec![1, 2], "expected both ids present, no duplicates",);
    });
}

/// Capacity boundary under concurrent writers — exactly CAP appends
/// succeed, the rest return false. Tests the CAS-loop bail-out branch.
#[test]
fn cas_append_capacity_boundary_under_race() {
    loom::model(|| {
        // CAP = 2 for this test (override the module-level CAP via a
        // local type alias so loom enumerates fewer interleavings).
        const SMALL: usize = 2;
        let list: Arc<AtomicNeighbourList<SMALL>> = Arc::new(AtomicNeighbourList::new());

        let a = {
            let list = list.clone();
            thread::spawn(move || list.cas_append(1))
        };
        let b = {
            let list = list.clone();
            thread::spawn(move || list.cas_append(2))
        };
        let c = {
            let list = list.clone();
            thread::spawn(move || list.cas_append(3))
        };

        let r_a = a.join().unwrap();
        let r_b = b.join().unwrap();
        let r_c = c.join().unwrap();

        // Exactly two threads succeed; one returns false (capacity reached).
        let success_count = [r_a, r_b, r_c].iter().filter(|x| **x).count();
        assert_eq!(
            success_count, 2,
            "expected exactly 2 successes, got {success_count}"
        );

        // The list contains exactly two of {1, 2, 3} and no garbage.
        let snap = list.snapshot();
        assert_eq!(snap.len(), 2);
        for &v in &snap {
            assert!(v == 1 || v == 2 || v == 3, "garbage id {v}");
        }
    });
}
