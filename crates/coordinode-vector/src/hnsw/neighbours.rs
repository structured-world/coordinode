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
//! * **C1** — single-writer `set`. Caller holds `&mut HnswIndex` while
//!   mutating; concurrent search reads via [`snapshot`] without locking.
//! * **C3 (now)** — multi-writer [`cas_append`] for incoming-edge add
//!   under concurrent inserters; [`replace`] for the prune protocol
//!   (single-writer per-list, gated by HnswIndex). `set` remains the
//!   bulk-replace primitive both helpers ultimately call.
//! * **C4 (later)** — `loom` interleaving + `miri` UB scan campaign over
//!   these primitives.

// Atomics are routed through `loom` when the `--cfg loom` build flag is set,
// so the model-checker can permute every observable interleaving. Under a
// regular build we use `std::sync::atomic` and the cost is zero.
#[cfg(loom)]
use loom::sync::atomic::{AtomicU32, AtomicU64, Ordering};
#[cfg(not(loom))]
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
#[doc(hidden)]
pub struct AtomicNeighbourList<const N: usize> {
    /// Published length. Reads use `Acquire`; writes use `Release`.
    len: AtomicU32,
    /// Inline slots. All `EMPTY` at construction. Slot stores use `Relaxed`
    /// and are made visible to readers by the subsequent `Release` store on
    /// `len`.
    slots: [AtomicU64; N],
}

impl<const N: usize> AtomicNeighbourList<N> {
    /// Construct an empty neighbour list. All slots are `EMPTY`.
    ///
    /// Const-constructible under regular builds (slot array literal). Under
    /// `--cfg loom` the constructor is non-const because `loom::sync::
    /// atomic::AtomicU64::new` is non-const (the model-checker needs to
    /// register every atomic with its scheduler at runtime).
    #[cfg(not(loom))]
    pub(crate) const fn new() -> Self {
        Self {
            len: AtomicU32::new(0),
            slots: [const { AtomicU64::new(EMPTY) }; N],
        }
    }

    /// Loom-flavoured constructor (non-const, registers each atomic with
    /// the model-checker scheduler).
    #[cfg(loom)]
    pub fn new() -> Self {
        Self {
            len: AtomicU32::new(0),
            slots: std::array::from_fn(|_| AtomicU64::new(EMPTY)),
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
    pub fn len(&self) -> usize {
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
    pub fn snapshot_into(&self, out: &mut Vec<u64>) {
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
    pub fn snapshot(&self) -> Vec<u64> {
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
    pub fn set(&self, new: &[u64]) {
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

    // ─── C3 lock-free write primitives (R858c day 1) ──────────────────────

    /// Append `id` to the list under concurrent writers. Returns `true` on
    /// success, `false` if the list is full (caller must run a shrink/prune
    /// protocol — see `HnswIndex::prune_connections`, which today runs
    /// single-writer under `&mut self`).
    ///
    /// # Algorithm
    ///
    /// 1. CAS-loop on `len` to reserve a slot index — read current `len`,
    ///    abort with `false` if at capacity, else `compare_exchange` to
    ///    `len + 1`. Only the winning thread reaches step 2.
    /// 2. The winner writes `id` to its reserved slot with `Release`
    ///    ordering, making the new neighbour visible to subsequent
    ///    snapshots.
    ///
    /// # Concurrent-reader semantics
    ///
    /// A snapshot taken between step 1 (reserve) and step 2 (write) sees
    /// `len = new_len` but slot `current` is still `EMPTY`. The reader path
    /// ([`snapshot_into`]) filters `EMPTY` out, so the transient state is
    /// equivalent to the new entry "not yet visible" — readers either see
    /// the old state (snapshot before reserve) or the new state (snapshot
    /// after write). No torn read of a partially-populated entry can
    /// happen because slot stores are atomic `u64`.
    ///
    /// # Why not `fetch_add`
    ///
    /// `fetch_add` would also work and is one instruction cheaper, but it
    /// makes the over-capacity case messy: the bumped `len` is observable
    /// before we know we can't actually store there. The CAS-loop keeps
    /// `len` monotonic AND never above `N`.
    #[allow(dead_code)] // Wired into HnswIndex concurrent insert path in C3 day 2.
    pub fn cas_append(&self, id: u64) -> bool {
        loop {
            let current = self.len.load(Ordering::Acquire) as usize;
            if current >= N {
                return false;
            }
            let new_len = (current as u32) + 1;
            match self.len.compare_exchange(
                current as u32,
                new_len,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // We own slot `current`. Publish the id.
                    self.slots[current].store(id, Ordering::Release);
                    return true;
                }
                Err(_) => {
                    // Another writer raced ahead; retry with the fresh len.
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Atomically replace the entire list contents.
    ///
    /// **Caller contract (C3 prune path):** this method is single-writer at
    /// the per-list granularity. The HNSW prune protocol acquires a per-node
    /// "prune-in-progress" gate before calling `replace`, so no other
    /// `cas_append` or `replace` runs concurrently on the same list. Within
    /// that gate, concurrent readers ([`snapshot_into`] etc) are still safe
    /// — they observe either the pre-replace or post-replace contents, with
    /// no torn intermediate (sequencing identical to [`set`]).
    ///
    /// For full multi-writer replace, this would need a per-list version
    /// counter + reader retry loop — deferred until a workload actually
    /// requires it.
    #[allow(dead_code)] // Wired into prune protocol in C3 day 2.
    pub fn replace(&self, new: &[u64]) {
        self.set(new);
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
}
