//! Contiguous layer-0 block, mirror of hnswlib's `data_level0_memory_`.
//!
//! ## Layout
//!
//! One `Box<[u8]>` sized at `capacity * stride`. Each per-node block holds
//! ONLY what the f32 search hot path reads:
//!
//! ```text
//! offset 0                       : u32              neighbour_count
//! offset 4                       : [u32; M_MAX0]    neighbour_ids
//! offset 4 + M_MAX0*4            : [f32; dim]       f32_vector
//! ```
//!
//! `stride` is `align_up(per_node_unaligned, 8)` so the next node's
//! `neighbour_count` lands on an 8-byte boundary, the neighbour ids stay
//! u32-aligned, and the f32 vector stays f32-aligned.
//!
//! For sift d=128 M_MAX0=64: 4 + 256 + 512 = 772 B per node.
//! For glove d=100 M_MAX0=64: 4 + 256 + 400 = 660 B per node.
//!
//! ## Why this is separate from `inline_layer0`
//!
//! `inline_layer0` (prior contiguous-layout attempt, chunks 1.1-1.9) packs
//! `[neighbours, f32, rabitq_code, rabitq_scalars, label]` into one block.
//! On sift the stride grew to ~700-800 B PLUS rabitq overhead, and the f32
//! search path regressed 13-16% on both ST and MT4 (chunks 1.4 / 1.5
//! reverted). The per-visit working set was larger than hnswlib's because
//! we paid the cache cost of rabitq scalars even when only the f32 vector
//! was read.
//!
//! This block deliberately mirrors ONLY what hnswlib's f32 path uses,
//! matching `size_data_per_element_` byte-for-byte at equal (dim, M_MAX0).
//!
//! ## Concurrency
//!
//! Neighbour ids + count are **atomic** (`AtomicU32`): [`DataLevel0Block::
//! cas_append_neighbour`] lets concurrent back-edge writers append under
//! `&self`, mirroring [`super::neighbours::AtomicNeighbourList`] — CAS the
//! count to reserve a slot, `Release`-store the id; readers `Acquire` the
//! count then read slots, filtering the [`EMPTY_ID`] sentinel that marks a
//! reserved-but-not-yet-written slot. The f32 vector is written once under
//! `&mut self` ([`DataLevel0Block::set_vector`]) and read under `&self`
//! ([`DataLevel0Block::vector_ptr`]); it carries no atomics because the
//! wrapping `HnswIndex` never appends to it concurrently (insert holds
//! `&mut self`, search holds `&self`). Bulk neighbour replace
//! ([`DataLevel0Block::set_neighbours`]) is single-writer-per-node by caller
//! contract (gated against `cas_append_neighbour` on the same node, exactly
//! like `AtomicNeighbourList::set`).

#![allow(
    dead_code,
    reason = "consumers land in the search-path wiring chunks on the same plan; tests exercise the round-trip"
)]

use core::sync::atomic::{AtomicU32, Ordering};

#[cfg(test)]
use super::M_MAX0;

/// Per-node block alignment. 64 keeps `neighbour_count` + the first few
/// `neighbour_ids` on the same cache line as the f32 vector prefix, and
/// gives the f32 vector itself a clean 4-byte alignment via the
/// `vector_offset` computation in [`DataLevel0Block::new`].
const NODE_ALIGN: usize = 8;

/// Sentinel for a reserved-but-not-yet-written neighbour slot (u32-wide
/// analogue of [`super::neighbours::EMPTY`]). Node ids are dense from `0`, so
/// `u32::MAX` is safe to reserve — an index would need >4 billion vectors in
/// one shard to collide, well past the LSM-backed sharding threshold.
/// [`DataLevel0Block::cas_append_neighbour`] reserves a slot (count CAS)
/// before storing the id; a concurrent reader observing the bumped count but
/// the unwritten slot sees this sentinel and filters it out.
const EMPTY_ID: u32 = u32::MAX;

/// One contiguous f32-side layer-0 store. See the module doc for layout
/// rationale and the chunked execution plan that wires it up.
#[derive(Debug)]
pub(super) struct DataLevel0Block {
    /// Backing storage. Sized at `capacity * stride` bytes plus an
    /// alignment pad slot — the actual block of node `idx` starts at
    /// byte offset `idx * stride` from `backing.as_ptr()`.
    backing: Box<[u8]>,
    /// Number of node slots the allocation can hold.
    capacity: usize,
    /// Bytes per per-node block, rounded up to a multiple of 8.
    stride: usize,
    /// Byte offset of the packed neighbour-ids array within a per-node
    /// block. Always 4 (after the leading `neighbour_count: u32`).
    neighbour_ids_offset: usize,
    /// Byte offset of the f32 vector within a per-node block. Always
    /// `neighbour_ids_offset + M_MAX0 * 4`.
    vector_offset: usize,
    /// Maximum neighbours per layer-0 slot (`M_MAX0`).
    m_max0: usize,
    /// Vector dimension.
    dim: usize,
    /// Whether per-node blocks still carry the f32 vector. Set false by
    /// [`DataLevel0Block::drop_f32`] when the offload path frees the f32
    /// payload (search then runs on quantized codes, rerank loads f32 from
    /// disk). When false, the stride no longer reserves f32 space and
    /// `vector_ptr` must not be called.
    has_f32: bool,
}

#[inline(always)]
fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// Fill the neighbour-id region of nodes `start..end` with the [`EMPTY_ID`]
/// sentinel (0xFF bytes), leaving the count field (offset 0) and the f32
/// region zeroed. Called by [`DataLevel0Block::new`] (all nodes) and
/// [`DataLevel0Block::ensure_capacity`] (the freshly grown nodes) so the
/// lock-free `cas_append_neighbour` reserve-before-write window never lets a
/// reader observe a zero-initialised slot as the real node id `0`.
fn init_neighbour_slots(
    backing: &mut [u8],
    start: usize,
    end: usize,
    stride: usize,
    neighbour_ids_offset: usize,
    m_max0: usize,
) {
    for idx in start..end {
        let ids = idx * stride + neighbour_ids_offset;
        backing[ids..ids + m_max0 * 4].fill(0xFF);
    }
}

impl DataLevel0Block {
    /// Allocate a fresh layer-0 store sized for `capacity` nodes, each
    /// holding `m_max0` neighbour slots and a `dim`-dimensional f32
    /// vector. The backing memory is zeroed.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`, `capacity == 0`, or `m_max0 == 0`.
    #[allow(
        clippy::panic,
        reason = "construction failure (overflow / zero arg) is a programmer error and must abort"
    )]
    pub(super) fn new(capacity: usize, m_max0: usize, dim: usize) -> Self {
        assert!(dim > 0, "DataLevel0Block: dim must be > 0");
        assert!(capacity > 0, "DataLevel0Block: capacity must be > 0");
        assert!(m_max0 > 0, "DataLevel0Block: m_max0 must be > 0");

        let neighbour_ids_offset = 4_usize;
        let vector_offset = neighbour_ids_offset + m_max0 * 4;
        let per_node_unaligned = vector_offset + dim * 4;
        let stride = align_up(per_node_unaligned, NODE_ALIGN);
        let Some(total) = stride.checked_mul(capacity) else {
            panic!("DataLevel0Block: stride * capacity overflow");
        };

        let mut backing = vec![0u8; total].into_boxed_slice();
        // Atomic neighbour access casts per-node bytes to `AtomicU32`. The
        // global allocator returns >=16-aligned blocks for any non-zero size,
        // so the Vec<u8> base (and every `idx * stride` node start, stride
        // being a multiple of 8) is >=4-aligned. Assert it rather than trust
        // it silently — a custom allocator violating this would be loud UB.
        assert_eq!(
            backing.as_ptr() as usize % core::mem::align_of::<AtomicU32>(),
            0,
            "DataLevel0Block backing must be AtomicU32-aligned"
        );
        init_neighbour_slots(
            &mut backing,
            0,
            capacity,
            stride,
            neighbour_ids_offset,
            m_max0,
        );

        Self {
            backing,
            capacity,
            stride,
            neighbour_ids_offset,
            vector_offset,
            m_max0,
            dim,
            has_f32: true,
        }
    }

    /// Whether per-node blocks still carry the f32 vector (false after
    /// [`DataLevel0Block::drop_f32`]).
    #[inline]
    pub(super) fn has_f32(&self) -> bool {
        self.has_f32
    }

    /// Free the f32 vectors, re-laying the block out to neighbours-only so the
    /// f32 bytes are actually returned to the allocator (the offload path calls
    /// this after calibration: search runs on quantized codes and rerank loads
    /// f32 from disk). Idempotent. Under `&mut self`, so no concurrent reader
    /// holds a pointer into the old backing. After this, `has_f32` is false and
    /// `vector_ptr` must not be called.
    pub(super) fn drop_f32(&mut self) {
        if !self.has_f32 {
            return;
        }
        // New stride keeps only `[neighbour_count: u32][neighbour_ids:
        // M_MAX0 u32]`, dropping the trailing f32 vector.
        let new_stride = align_up(self.vector_offset, NODE_ALIGN);
        let mut new_backing = vec![0u8; new_stride * self.capacity].into_boxed_slice();
        for idx in 0..self.capacity {
            let old = idx * self.stride;
            let new = idx * new_stride;
            new_backing[new..new + self.vector_offset]
                .copy_from_slice(&self.backing[old..old + self.vector_offset]);
        }
        self.backing = new_backing;
        self.stride = new_stride;
        self.has_f32 = false;
    }

    /// Grow the backing allocation so the block holds at least `required`
    /// node slots, reallocating-and-copying if it does not already fit.
    /// Capacity at least doubles to amortise repeated growth. Called only
    /// from the insert path under `&mut self`, so no concurrent `&self`
    /// reader holds a raw pointer into the old backing while it is replaced
    /// (the wrapping `HnswIndex` enforces search-holds-&self / insert-holds-
    /// &mut self).
    ///
    /// # Panics
    ///
    /// Panics if `stride * new_capacity` overflows `usize` — a pathological
    /// allocation far beyond addressable memory; aborting beats silently
    /// dropping vectors.
    #[allow(
        clippy::panic,
        reason = "growth past usize::MAX bytes is unreachable on real hardware and must abort, not silently lose vectors"
    )]
    pub(super) fn ensure_capacity(&mut self, required: usize) {
        if required <= self.capacity {
            return;
        }
        let new_capacity = required.max(self.capacity.saturating_mul(2));
        let Some(total) = self.stride.checked_mul(new_capacity) else {
            panic!("DataLevel0Block: stride * capacity overflow on grow");
        };
        let mut new_backing = vec![0u8; total].into_boxed_slice();
        assert_eq!(
            new_backing.as_ptr() as usize % core::mem::align_of::<AtomicU32>(),
            0,
            "DataLevel0Block backing must be AtomicU32-aligned"
        );
        new_backing[..self.backing.len()].copy_from_slice(&self.backing);
        // Existing nodes were copied verbatim (slots intact); the grown nodes
        // are zeroed, so EMPTY-init their neighbour slots for safe cas_append.
        init_neighbour_slots(
            &mut new_backing,
            self.capacity,
            new_capacity,
            self.stride,
            self.neighbour_ids_offset,
            self.m_max0,
        );
        self.backing = new_backing;
        self.capacity = new_capacity;
    }

    /// Capacity in node slots.
    #[inline]
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Stride in bytes between per-node blocks.
    #[inline]
    pub(super) fn stride(&self) -> usize {
        self.stride
    }

    /// Vector dimension.
    #[inline]
    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    /// Maximum neighbours per layer-0 slot.
    #[inline]
    pub(super) fn m_max0(&self) -> usize {
        self.m_max0
    }

    /// Base pointer of the per-node block for `idx`. Internal helper;
    /// every public accessor goes through it.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    #[inline(always)]
    unsafe fn node_base(&self, idx: usize) -> *const u8 {
        // SAFETY: caller guarantees `idx < capacity`, so the byte
        // offset `idx * stride` lies inside the backing allocation.
        unsafe { self.backing.as_ptr().add(idx * self.stride) }
    }

    /// Mutable base pointer of the per-node block for `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity` and no other borrow of node `idx`'s bytes
    /// is live (the wrapping `&mut self` provides that).
    #[inline(always)]
    unsafe fn node_base_mut(&mut self, idx: usize) -> *mut u8 {
        // SAFETY: caller guarantees `idx < capacity` and exclusive
        // access to node `idx`'s slice (via `&mut self`).
        unsafe { self.backing.as_mut_ptr().add(idx * self.stride) }
    }

    /// Reference to node `idx`'s neighbour-count atomic (per-node offset 0).
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`. The bytes are `AtomicU32`-aligned (asserted at
    /// construction / grow) and the neighbour region is only ever accessed
    /// atomically, so the shared atomic reference races nothing.
    #[inline(always)]
    unsafe fn count_atomic(&self, idx: usize) -> &AtomicU32 {
        // SAFETY: idx bound per contract; offset 0 is a 4-aligned u32; the
        // &AtomicU32 borrows `self`.
        unsafe { &*(self.node_base(idx) as *const AtomicU32) }
    }

    /// Reference to slot `j` of node `idx`'s neighbour-id array.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity` and `j < self.m_max0`.
    #[inline(always)]
    unsafe fn slot_atomic(&self, idx: usize, j: usize) -> &AtomicU32 {
        // SAFETY: idx/j bounds per contract; `neighbour_ids_offset + j*4` is
        // 4-aligned and inside the per-node neighbour region.
        unsafe {
            &*(self.node_base(idx).add(self.neighbour_ids_offset + j * 4) as *const AtomicU32)
        }
    }

    /// Number of valid neighbours published for node `idx`. Zero before the
    /// first append / `set_neighbours`. `Acquire`-ordered so any slot written
    /// before the publishing `Release` is visible to a subsequent slot read.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    #[inline]
    pub(super) unsafe fn neighbour_count(&self, idx: usize) -> u32 {
        // SAFETY: idx bound per contract.
        unsafe { self.count_atomic(idx) }.load(Ordering::Acquire)
    }

    /// Bulk-publish the neighbour ids for node `idx`, overwriting the prior
    /// set: write the live slots `Relaxed`, wipe the tail to [`EMPTY_ID`] so a
    /// later [`Self::cas_append_neighbour`] reader walking past the count sees
    /// the sentinel, then publish the count `Release`.
    ///
    /// **Single-writer-per-node** by caller contract: must not run concurrently
    /// with `cas_append_neighbour` (or another `set_neighbours`) on the same
    /// node — identical to `AtomicNeighbourList::set`. Takes `&self` because the
    /// writes are atomic; the wrapping `HnswIndex` gates the bulk-replace
    /// (prune) path against concurrent appends.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity` and `ids.len() <= self.m_max0`.
    #[inline]
    pub(super) unsafe fn set_neighbours(&self, idx: usize, ids: &[u32]) {
        debug_assert!(idx < self.capacity, "idx out of bounds");
        debug_assert!(ids.len() <= self.m_max0, "ids overflow neighbour slot");
        let n = ids.len().min(self.m_max0);

        // SAFETY: idx/j bounds per the asserts above.
        unsafe {
            // Phase 1: write the live slots (Relaxed; the count Release orders them).
            for (j, &id) in ids.iter().take(n).enumerate() {
                self.slot_atomic(idx, j).store(id, Ordering::Relaxed);
            }
            // Wipe the tail so an appender reader past the count sees EMPTY.
            for j in n..self.m_max0 {
                self.slot_atomic(idx, j).store(EMPTY_ID, Ordering::Relaxed);
            }
            // Phase 2: publish the count (Release makes the slot stores visible).
            self.count_atomic(idx).store(n as u32, Ordering::Release);
        }
    }

    /// Snapshot the live neighbour ids of node `idx` into `out`, clearing it
    /// first. Wait-free: `Acquire`-load the count, read that many slots
    /// `Relaxed`, filtering the [`EMPTY_ID`] sentinel. Safe under a concurrent
    /// `cas_append_neighbour` — a slot reserved (count bumped) but not yet
    /// written reads `EMPTY_ID` and is skipped, so the reader sees either the
    /// pre- or post-append set, never a torn id (slot stores are atomic u32).
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    #[inline]
    pub(super) unsafe fn read_neighbours_into(&self, idx: usize, out: &mut Vec<u32>) {
        out.clear();
        // SAFETY: idx bound per contract.
        let count =
            (unsafe { self.count_atomic(idx) }.load(Ordering::Acquire) as usize).min(self.m_max0);
        out.reserve(count);
        for j in 0..count {
            // SAFETY: j < count <= m_max0.
            let v = unsafe { self.slot_atomic(idx, j) }.load(Ordering::Relaxed);
            if v != EMPTY_ID {
                out.push(v);
            }
        }
    }

    /// Prefetch the neighbour region (count + ids, per-node offsets
    /// `0..vector_offset`) of node `idx` — exactly the bytes
    /// [`Self::read_neighbours_into`] / [`Self::read_neighbours_into_u64`]
    /// touch on the next search visit. No-op when `idx` is out of range;
    /// prefetch is a hint, the pointer is never dereferenced here.
    #[inline(always)]
    pub(super) fn prefetch_neighbours(&self, idx: usize) {
        if idx >= self.capacity {
            return;
        }
        // SAFETY: idx < capacity; the count + ids region lies in the per-node
        // block by construction.
        unsafe {
            let base = self.node_base(idx);
            let mut off = 0;
            while off < self.vector_offset {
                super::prefetch_read_data(base.add(off));
                off += 64;
            }
        }
    }

    /// Like [`Self::read_neighbours_into`] but widening each id to `u64` for
    /// the search hot path's `Vec<u64>` candidate buffer — reads straight into
    /// `out` so the per-visit read allocates nothing beyond `out`'s growth.
    /// Same wait-free, `EMPTY_ID`-filtering, concurrent-append-safe semantics.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    #[inline]
    pub(super) unsafe fn read_neighbours_into_u64(&self, idx: usize, out: &mut Vec<u64>) {
        out.clear();
        // SAFETY: idx bound per contract.
        let count =
            (unsafe { self.count_atomic(idx) }.load(Ordering::Acquire) as usize).min(self.m_max0);
        out.reserve(count);
        for j in 0..count {
            // SAFETY: j < count <= m_max0.
            let v = unsafe { self.slot_atomic(idx, j) }.load(Ordering::Relaxed);
            if v != EMPTY_ID {
                out.push(v as u64);
            }
        }
    }

    /// Append `id` to node `idx`'s neighbour list under concurrent writers.
    /// Returns `true` on success, `false` if already at `m_max0` (caller runs
    /// the prune protocol). Mirrors
    /// [`super::neighbours::AtomicNeighbourList::cas_append`]: CAS-loop the
    /// count to reserve a slot, then `Release`-store the id. A reader between
    /// reserve and store sees the [`EMPTY_ID`] sentinel and filters it.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    pub(super) unsafe fn cas_append_neighbour(&self, idx: usize, id: u32) -> bool {
        // SAFETY: idx bound per contract.
        let count = unsafe { self.count_atomic(idx) };
        loop {
            let current = count.load(Ordering::Acquire) as usize;
            if current >= self.m_max0 {
                return false;
            }
            match count.compare_exchange(
                current as u32,
                current as u32 + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // We own slot `current`; publish the id with Release.
                    // SAFETY: current < m_max0 per the gate above.
                    unsafe { self.slot_atomic(idx, current) }.store(id, Ordering::Release);
                    return true;
                }
                Err(_) => core::hint::spin_loop(),
            }
        }
    }

    /// Install the f32 vector for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity` and `vector.len() == self.dim`.
    #[inline]
    pub(super) unsafe fn set_vector(&mut self, idx: usize, vector: &[f32]) {
        debug_assert!(idx < self.capacity, "idx out of bounds");
        debug_assert_eq!(vector.len(), self.dim, "vector dim mismatch");

        // SAFETY: caller guarantees the bounds; vector_offset is a
        // 4-byte multiple by construction so the destination is
        // f32-aligned.
        unsafe {
            let base = self.node_base_mut(idx);
            let vec_ptr = base.add(self.vector_offset) as *mut f32;
            core::ptr::copy_nonoverlapping(vector.as_ptr(), vec_ptr, self.dim);
        }
    }

    /// Read-only pointer to the f32 vector of node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`. The borrow checker is not involved — the
    /// caller is responsible for treating the returned pointer as a
    /// `&[f32; self.dim]` borrow of `self`.
    #[inline]
    pub(super) unsafe fn vector_ptr(&self, idx: usize) -> *const f32 {
        // SAFETY: caller guarantees idx bounds; vector_offset stays
        // inside the per-node block by construction.
        unsafe { self.node_base(idx).add(self.vector_offset) as *const f32 }
    }

    /// Issue `_mm_prefetch` hints (or arch equivalent) covering the
    /// **entire f32 vector** portion of node `idx`'s per-node block.
    /// Targets `vector_offset` (offset 260 at sift d=128) so the warmed
    /// lines are the ones the distance kernel will actually touch — not
    /// the leading neighbour-count / neighbour-ids cache line that the
    /// f32 path never reads.
    ///
    /// One hint per 64B line over the full `dim * 4` span: a d=100
    /// vector is 400 B = 7 lines, and a single-line hint left the
    /// kernel missing on lines 2-7 of every visit (the search loop
    /// prefetches one neighbour ahead, so the ~40ns of the current
    /// neighbour's dot is exactly the latency window these hints need).
    /// No-op on unsupported architectures. Safe wrapper: `idx >=
    /// capacity` skips the hint instead of panicking.
    #[inline(always)]
    pub(super) fn prefetch(&self, idx: usize) {
        if idx >= self.capacity {
            return;
        }
        // SAFETY: `idx < capacity` gate above; `vector_offset` and the
        // `dim * 4` span lie inside the per-node block by construction.
        // Prefetch is a hint, not a load — the pointer is never
        // dereferenced by this function.
        unsafe {
            let base = self.vector_ptr(idx) as *const u8;
            let span = self.dim * core::mem::size_of::<f32>();
            let mut off = 0;
            while off < span {
                super::prefetch_read_data(base.add(off));
                off += 64;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
