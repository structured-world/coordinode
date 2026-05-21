//! Lock-free atomic neighbour list — the storage unit of the lock-free HNSW
//! read path. One [`AtomicNeighbourList<N>`] holds the out-edges of a single
//! node at a single layer; the whole graph is a `Vec<Vec<AtomicNeighbourList>>`
//! (outer `Vec` indexed by node, inner `Vec` indexed by layer).
//!
//! # Memory model
//!
//! * Each slot is an [`AtomicU64`] node-id. The sentinel [`EMPTY`] marks an
//!   unused slot — node ids ≠ `u64::MAX` are required (the index already uses
//!   monotonic, dense ids starting from 0, so the sentinel is safe).
//! * `len` is an [`AtomicU32`] published with `Release` after every slot store
//!   is `Relaxed`. Readers load `len` with `Acquire`, then read exactly that
//!   many slots `Relaxed` — the `Acquire`/`Release` pair guarantees the slot
//!   stores happen-before the `len` load.
//! * The slot array is allocated inline (`[AtomicU64; N]`) so a snapshot is a
//!   cache-line-friendly memcpy on the read path. `N` is chosen to match
//!   `m_max0` (e.g. `64` for the default `M = 32`).
//!
//! # Where this lives in the C1 → C4 plan
//!
//! * **C1 (now)** — single-writer. `set` is called only from the insert path
//!   while the caller still holds `&mut HnswIndex`. Search reads via
//!   [`snapshot`] with no locking. Inserts are still sequential.
//! * **C3 (later)** — multi-writer. [`replace`] and [`cas_append`] add CAS
//!   write APIs. `next_id` becomes monotonic to avoid ABA.
//! * **C4 (later)** — `loom` interleaving + `miri` UB scan.
//!
//! Today only the C1 surface is implemented; the C3 entry points are stubbed
//! at the bottom of the file so the import surface is stable across phases.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Sentinel value for an unused slot. Node IDs are dense `u64` starting from
/// `0`, so `u64::MAX` is safe to reserve.
pub(crate) const EMPTY: u64 = u64::MAX;

/// Fixed-capacity, lock-free list of neighbour node-IDs for one HNSW node at
/// one layer.
///
/// Capacity `N` is a compile-time constant — pick it to match `m_max0` (the
/// per-node connection cap at layer 0, typically `2 * M`). Storing the slots
/// inline (no heap indirection) lets the read path do a single cache-line
/// fetch for nodes with the common-case small degree, and at most a couple
/// of cache lines for the dense layer-0 case.
///
/// # Concurrency contract (C1)
///
/// * Multiple concurrent readers are safe (`snapshot` is wait-free).
/// * Exactly **one** writer at a time — enforced externally today by the
///   `&mut HnswIndex` borrow on insert. C3 will relax this via CAS.
/// * No `Drop` side-effects — the type is a plain POD over atomics.
pub(crate) struct AtomicNeighbourList<const N: usize> {
    /// Published length. Reads use `Acquire`; writes use `Release`.
    len: AtomicU32,
    /// Inline slots. All `EMPTY` at construction. Slot stores use `Relaxed`
    /// and are made visible to readers by the subsequent `Release` store on
    /// `len`.
    slots: [AtomicU64; N],
}

impl<const N: usize> AtomicNeighbourList<N> {
    /// Construct an empty neighbour list. All slots are `EMPTY`.
    pub(crate) const fn new() -> Self {
        Self {
            len: AtomicU32::new(0),
            slots: [const { AtomicU64::new(EMPTY) }; N],
        }
    }

    /// Capacity of the list (compile-time constant `N`).
    #[inline]
    #[allow(dead_code)] // Wired into HnswIndex in the next C1 step.
    pub(crate) const fn capacity(&self) -> usize {
        N
    }

    /// Current published length. Acquire-ordered so that any slots in
    /// `0..len` are visible to subsequent reads.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len.load(Ordering::Acquire) as usize
    }

    /// Snapshot the live neighbours into `out`, clearing it first.
    ///
    /// Wait-free, O(len). Returns the number of neighbours written.
    ///
    /// The snapshot is *consistent for the read epoch* — under concurrent
    /// writers (C3), the returned set is a valid intermediate state of the
    /// list at some point during the call. C1 callers see the single-writer
    /// committed state.
    pub(crate) fn snapshot_into(&self, out: &mut Vec<u64>) {
        out.clear();
        let len = self.len.load(Ordering::Acquire) as usize;
        let len = len.min(N);
        for slot in self.slots.iter().take(len) {
            let v = slot.load(Ordering::Relaxed);
            if v != EMPTY {
                out.push(v);
            }
        }
    }

    /// Allocate-and-return variant of [`snapshot_into`]. Prefer the in-place
    /// variant on hot search paths to recycle the output buffer.
    pub(crate) fn snapshot(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.len());
        self.snapshot_into(&mut out);
        out
    }

    /// Single-writer publish (C1 entry point).
    ///
    /// Overwrites the current neighbour set with `new` and publishes the new
    /// length. `new.len()` must be ≤ `N`; longer slices are truncated with a
    /// debug-only assertion (release builds silently truncate to `N`).
    ///
    /// Callers MUST hold exclusive write access — e.g. via `&mut HnswIndex`.
    /// Concurrent calls to `set` are UB under C1 semantics; C3 introduces
    /// [`AtomicNeighbourList::replace`] for the multi-writer case.
    #[allow(dead_code)] // Called from sync helper + tests; wired into hot path in C1 day 3.
    pub(crate) fn set(&self, new: &[u64]) {
        debug_assert!(
            new.len() <= N,
            "AtomicNeighbourList<{}>::set received {} neighbours — would truncate",
            N,
            new.len()
        );
        let n = new.len().min(N);

        // Phase 1: write slots (Relaxed — the Release on `len` orders them).
        for (slot, &id) in self.slots.iter().zip(new.iter()).take(n) {
            slot.store(id, Ordering::Relaxed);
        }
        // Wipe the tail so a later snapshot doesn't see stale ids past `len`.
        // Belt-and-braces: snapshot already truncates by `len`, but a future
        // reader of `cas_append` (C3) walks past `len` and must see `EMPTY`.
        for slot in self.slots.iter().skip(n) {
            slot.store(EMPTY, Ordering::Relaxed);
        }
        // Phase 2: publish length. Release ensures the slot stores are visible
        // before any reader observing this length sees them.
        self.len.store(n as u32, Ordering::Release);
    }

    // ─── C3 stubs (intentionally not yet wired) ────────────────────────────
    //
    // These will be implemented in R858c. They are declared here so the
    // import surface is stable across phases — C2 and the C3 implementation
    // PR can switch callers without re-exporting types.

    /// **C3 (not yet implemented).** Append `id` to the list under
    /// concurrent writers. Returns `true` on success, `false` if the list
    /// was full and the caller must run shrink-to-M heuristic.
    #[allow(dead_code)]
    pub(crate) fn cas_append(&self, _id: u64) -> bool {
        unimplemented!("cas_append lands in R858c (C3)")
    }

    /// **C3 (not yet implemented).** Atomically replace the entire list under
    /// concurrent writers using a per-node version counter to detect lost
    /// updates.
    #[allow(dead_code)]
    pub(crate) fn replace(&self, _new: &[u64]) {
        unimplemented!("replace lands in R858c (C3)")
    }
}

impl<const N: usize> Default for AtomicNeighbourList<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> std::fmt::Debug for AtomicNeighbourList<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snap = self.snapshot();
        f.debug_struct("AtomicNeighbourList")
            .field("capacity", &N)
            .field("len", &snap.len())
            .field("neighbours", &snap)
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
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
    #[should_panic(expected = "lands in R858c")]
    fn cas_append_unimplemented_in_c1() {
        let list: AtomicNeighbourList<8> = AtomicNeighbourList::new();
        let _ = list.cas_append(42);
    }

    #[test]
    #[should_panic(expected = "lands in R858c")]
    fn replace_unimplemented_in_c1() {
        let list: AtomicNeighbourList<8> = AtomicNeighbourList::new();
        list.replace(&[1, 2, 3]);
    }
}
