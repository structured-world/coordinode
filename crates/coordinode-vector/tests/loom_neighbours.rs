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

/// A reader during a whole-list replace sees the old list or the new one,
/// never a list assembled from both: a prune that replaces `[1, 2]` with
/// `[3, 4]` must not be observed as `[3, 2]` or `[1, 4]`.
#[test]
fn replace_is_observed_whole() {
    loom::model(|| {
        let list: Arc<AtomicNeighbourList<CAP>> = Arc::new(AtomicNeighbourList::new());
        list.set(&[1, 2]);

        let writer = {
            let list = list.clone();
            thread::spawn(move || list.set(&[3, 4]))
        };
        let reader = {
            let list = list.clone();
            thread::spawn(move || list.snapshot())
        };

        writer.join().unwrap();
        let seen = reader.join().unwrap();
        assert!(
            seen == vec![1, 2] || seen == vec![3, 4],
            "reader observed a mixed list: {seen:?}"
        );
    });
}

/// Two concurrent writers each calling `cas_append` once. The final list
/// must contain exactly both ids, never duplicates, never missing one.
/// This is the core multi-writer correctness property.
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

/// The edit that drops a removed node from a neighbour's list races an
/// append to the same list: the removal is recomputed against the list that
/// won, so the accepted append survives and the removed id is gone, in every
/// order.
#[test]
fn removal_edit_racing_an_append_keeps_the_append() {
    loom::model(|| {
        let list: Arc<AtomicNeighbourList<CAP>> = Arc::new(AtomicNeighbourList::new());
        list.set(&[1, 2]);

        let append = {
            let list = list.clone();
            thread::spawn(move || assert!(list.cas_append(5)))
        };
        let remove = {
            let list = list.clone();
            thread::spawn(move || {
                list.update(|current| {
                    current
                        .contains(&2)
                        .then(|| current.iter().copied().filter(|&id| id != 2).collect())
                });
            })
        };

        append.join().unwrap();
        remove.join().unwrap();
        let mut seen = list.snapshot();
        seen.sort_unstable();
        assert_eq!(seen, vec![1, 5], "lost the append or kept the removed id");
    });
}

/// A prune computed from a list another writer replaced is never published:
/// it reruns against the winner. Pruning `[1, 2, 3]` to the two largest ids
/// while 4 is appended ends as `[3, 4]` (append first) or `[2, 3, 4]` (prune
/// first), never `[2, 3]`, which would drop the accepted append.
#[test]
fn stale_prune_never_overwrites_an_append() {
    loom::model(|| {
        let list: Arc<AtomicNeighbourList<CAP>> = Arc::new(AtomicNeighbourList::new());
        list.set(&[1, 2, 3]);

        let append = {
            let list = list.clone();
            thread::spawn(move || assert!(list.cas_append(4)))
        };
        let prune = {
            let list = list.clone();
            thread::spawn(move || {
                list.update(|current| {
                    let mut kept = current.to_vec();
                    kept.sort_unstable();
                    let drop = kept.len().checked_sub(2).filter(|&d| d > 0)?;
                    Some(kept[drop..].into())
                });
            })
        };

        append.join().unwrap();
        prune.join().unwrap();
        let mut seen = list.snapshot();
        seen.sort_unstable();
        assert!(
            seen == vec![3, 4] || seen == vec![2, 3, 4],
            "a stale prune overwrote the append: {seen:?}"
        );
    });
}

/// Capacity boundary under concurrent writers: with room for one id, exactly
/// one of two racing appends succeeds and the other is told the list is full,
/// including when it lost the CAS to the winner and re-read a full list.
/// Two threads keep the model tractable with the epoch machinery under loom.
#[test]
fn cas_append_capacity_boundary_under_race() {
    loom::model(|| {
        const ONE: usize = 1;
        let list: Arc<AtomicNeighbourList<ONE>> = Arc::new(AtomicNeighbourList::new());

        let a = {
            let list = list.clone();
            thread::spawn(move || list.cas_append(1))
        };
        let b = {
            let list = list.clone();
            thread::spawn(move || list.cas_append(2))
        };

        let r_a = a.join().unwrap();
        let r_b = b.join().unwrap();

        assert!(r_a != r_b, "exactly one append fits: a={r_a}, b={r_b}");
        let winner = if r_a { 1 } else { 2 };
        assert_eq!(list.snapshot(), vec![winner]);
    });
}
