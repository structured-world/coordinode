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
//! All writes go through `&mut self` accessors; all reads go through
//! `&self` accessors. Standard Rust aliasing rules apply — no `UnsafeCell`,
//! no atomics. The wrapping `HnswIndex` enforces "search holds &self,
//! insert holds &mut self" already.

#![allow(
    dead_code,
    reason = "consumers land in the search-path wiring chunks on the same plan; tests exercise the round-trip"
)]

#[cfg(test)]
use super::M_MAX0;

/// Per-node block alignment. 64 keeps `neighbour_count` + the first few
/// `neighbour_ids` on the same cache line as the f32 vector prefix, and
/// gives the f32 vector itself a clean 4-byte alignment via the
/// `vector_offset` computation in [`DataLevel0Block::new`].
const NODE_ALIGN: usize = 8;

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

        Self {
            backing: vec![0u8; total].into_boxed_slice(),
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
        new_backing[..self.backing.len()].copy_from_slice(&self.backing);
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

    /// Number of valid neighbours stored for node `idx`. Zero before
    /// the first `set_neighbours` call.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    #[inline]
    pub(super) unsafe fn neighbour_count(&self, idx: usize) -> u32 {
        // SAFETY: the per-node block starts with a u32 by construction;
        // `node_base` returns a valid pointer to it.
        unsafe { core::ptr::read(self.node_base(idx) as *const u32) }
    }

    /// Install the neighbour ids for node `idx`. Stores the count and
    /// the prefix of the `[u32; M_MAX0]` slot. Subsequent entries past
    /// `ids.len()` are left untouched.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity` and `ids.len() <= self.m_max0`.
    #[inline]
    pub(super) unsafe fn set_neighbours(&mut self, idx: usize, ids: &[u32]) {
        debug_assert!(idx < self.capacity, "idx out of bounds");
        debug_assert!(ids.len() <= self.m_max0, "ids overflow neighbour slot");

        // SAFETY: caller guarantees the bounds; `node_base_mut` returns
        // a valid pointer; the writes stay inside the per-node block.
        unsafe {
            let base = self.node_base_mut(idx);
            core::ptr::write(base as *mut u32, ids.len() as u32);
            let ids_ptr = base.add(self.neighbour_ids_offset) as *mut u32;
            core::ptr::copy_nonoverlapping(ids.as_ptr(), ids_ptr, ids.len());
        }
    }

    /// Snapshot the neighbour ids of node `idx` into `out`, clearing
    /// `out` first. Reads exactly `neighbour_count(idx)` ids — anything
    /// past that is whatever the prior caller left in those slots.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`. No concurrent writer is mutating node
    /// `idx` (the `&mut self` writer holds exclusive access during
    /// insert; search reads under `&self`).
    #[inline]
    pub(super) unsafe fn read_neighbours_into(&self, idx: usize, out: &mut Vec<u32>) {
        // SAFETY: per the method-level safety contract.
        let count = unsafe { self.neighbour_count(idx) } as usize;
        debug_assert!(count <= self.m_max0, "stored count exceeds m_max0");

        out.clear();
        out.reserve(count);
        // SAFETY: caller guarantees `idx < capacity`; `count <= m_max0`
        // bounds the read to within the neighbour-ids region.
        unsafe {
            let base = self.node_base(idx);
            let ids_ptr = base.add(self.neighbour_ids_offset) as *const u32;
            for j in 0..count {
                out.push(core::ptr::read(ids_ptr.add(j)));
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
mod tests {
    use super::*;

    #[test]
    fn stride_matches_hnswlib_size_data_per_element_at_sift_128() {
        // hnswlib's `size_data_per_element_` for d=128, M_MAX0=64 is
        // 4 (count) + 64*4 (ids) + 128*4 (vec) = 772 B. Stride rounds
        // up to the next multiple of 8 — same 776 hnswlib uses on
        // alignment-strict configs.
        let block = DataLevel0Block::new(1, 64, 128);
        assert_eq!(block.stride(), 776);
        assert_eq!(block.vector_offset, 4 + 64 * 4);
    }

    #[test]
    fn stride_matches_layout_at_glove_100() {
        // glove d=100 M_MAX0=64: 4 + 256 + 400 = 660 B, stride aligned
        // up to 664.
        let block = DataLevel0Block::new(1, 64, 100);
        assert_eq!(block.stride(), 664);
    }

    #[test]
    fn neighbours_round_trip() {
        let mut block = DataLevel0Block::new(4, M_MAX0, 32);
        let ids = vec![10u32, 20, 30, 40, 50];

        // SAFETY: idx < capacity, ids.len() <= M_MAX0.
        unsafe {
            block.set_neighbours(2, &ids);
        }

        let mut out = Vec::new();
        // SAFETY: idx < capacity, no concurrent writer.
        unsafe {
            block.read_neighbours_into(2, &mut out);
        }
        assert_eq!(out, ids);

        // Other slots are still zero-count.
        unsafe {
            block.read_neighbours_into(0, &mut out);
        }
        assert!(out.is_empty());
    }

    #[test]
    fn vector_round_trip_and_alignment() {
        let dim = 64;
        let mut block = DataLevel0Block::new(8, M_MAX0, dim);
        let v: Vec<f32> = (0..dim).map(|i| i as f32 * 0.25).collect();

        // SAFETY: idx < capacity, v.len() == dim.
        unsafe {
            block.set_vector(3, &v);
        }

        // SAFETY: idx < capacity.
        let ptr = unsafe { block.vector_ptr(3) };
        assert_eq!(ptr.align_offset(core::mem::align_of::<f32>()), 0);

        // SAFETY: ptr is f32-aligned and points to `dim` valid f32
        // values written by `set_vector`.
        let slice = unsafe { core::slice::from_raw_parts(ptr, dim) };
        assert_eq!(slice, v.as_slice());
    }

    #[test]
    fn ensure_capacity_grows_and_preserves_existing_vectors() {
        let dim = 8;
        let mut block = DataLevel0Block::new(2, M_MAX0, dim);
        let v0: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let v1: Vec<f32> = (0..dim).map(|i| i as f32 + 100.0).collect();
        // SAFETY: idx < capacity, len == dim.
        unsafe {
            block.set_vector(0, &v0);
            block.set_vector(1, &v1);
        }
        assert_eq!(block.capacity(), 2);

        // Grow to fit idx 5 (beyond the initial capacity).
        block.ensure_capacity(6);
        assert!(block.capacity() >= 6, "capacity must grow to fit");

        // Existing vectors survive the reallocate-and-copy.
        // SAFETY: idx < capacity, ptr is f32-aligned for `dim` values.
        unsafe {
            let s0 = core::slice::from_raw_parts(block.vector_ptr(0), dim);
            let s1 = core::slice::from_raw_parts(block.vector_ptr(1), dim);
            assert_eq!(s0, v0.as_slice());
            assert_eq!(s1, v1.as_slice());
        }

        // The newly available slot is usable.
        let v5: Vec<f32> = (0..dim).map(|i| i as f32 + 200.0).collect();
        // SAFETY: idx 5 < capacity after growth, len == dim.
        unsafe {
            block.set_vector(5, &v5);
            let s5 = core::slice::from_raw_parts(block.vector_ptr(5), dim);
            assert_eq!(s5, v5.as_slice());
        }

        // No-op when already large enough — no shrink, no realloc.
        let cap = block.capacity();
        block.ensure_capacity(3);
        assert_eq!(block.capacity(), cap);
    }

    #[test]
    fn drop_f32_shrinks_stride_and_preserves_neighbours() {
        let dim = 8;
        let mut block = DataLevel0Block::new(4, M_MAX0, dim);
        // SAFETY: idx < capacity, neighbour ids fit M_MAX0, vector len == dim.
        unsafe {
            block.set_neighbours(0, &[10, 20, 30]);
            block.set_neighbours(2, &[40]);
            block.set_vector(0, &[1.0f32; 8]);
        }
        assert!(block.has_f32());
        let stride_before = block.stride();

        block.drop_f32();

        assert!(!block.has_f32(), "f32 marked absent after drop");
        assert!(
            block.stride() < stride_before,
            "stride shrinks once the f32 slot is gone"
        );
        // Neighbours survive the re-layout into the smaller stride.
        // SAFETY: idx < capacity.
        unsafe {
            assert_eq!(block.neighbour_count(0), 3);
            assert_eq!(block.neighbour_count(2), 1);
            assert_eq!(block.neighbour_count(1), 0);
        }
        // Idempotent.
        block.drop_f32();
        assert!(!block.has_f32());
    }

    #[test]
    fn payloads_do_not_alias_across_nodes() {
        let dim = 16;
        let mut block = DataLevel0Block::new(4, M_MAX0, dim);
        let v0: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let v1: Vec<f32> = (0..dim).map(|i| -(i as f32)).collect();

        // SAFETY: idx < capacity, vector len == dim.
        unsafe {
            block.set_vector(0, &v0);
            block.set_vector(1, &v1);
            block.set_neighbours(0, &[1, 2, 3]);
            block.set_neighbours(1, &[10, 20]);
        }

        let mut ns0 = Vec::new();
        let mut ns1 = Vec::new();
        // SAFETY: idx < capacity.
        unsafe {
            block.read_neighbours_into(0, &mut ns0);
            block.read_neighbours_into(1, &mut ns1);
        }
        assert_eq!(ns0, vec![1, 2, 3]);
        assert_eq!(ns1, vec![10, 20]);

        // SAFETY: vector_ptr borrows are non-overlapping per-node
        // payloads.
        let s0 = unsafe { core::slice::from_raw_parts(block.vector_ptr(0), dim) };
        let s1 = unsafe { core::slice::from_raw_parts(block.vector_ptr(1), dim) };
        assert_eq!(s0, v0.as_slice());
        assert_eq!(s1, v1.as_slice());
    }

    #[test]
    fn neighbour_count_starts_at_zero() {
        let block = DataLevel0Block::new(2, M_MAX0, 8);
        // SAFETY: idx < capacity.
        let c = unsafe { block.neighbour_count(0) };
        assert_eq!(c, 0);
    }

    #[test]
    fn prefetch_out_of_bounds_is_noop() {
        let block = DataLevel0Block::new(2, M_MAX0, 8);
        // Should not panic, just skip the prefetch hint.
        block.prefetch(99);
    }

    #[test]
    #[should_panic(expected = "dim must be > 0")]
    fn rejects_zero_dim() {
        let _ = DataLevel0Block::new(1, M_MAX0, 0);
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn rejects_zero_capacity() {
        let _ = DataLevel0Block::new(0, M_MAX0, 8);
    }
}
