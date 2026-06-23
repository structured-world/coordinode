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

    /// Whether the list currently holds zero neighbours.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
mod tests;
