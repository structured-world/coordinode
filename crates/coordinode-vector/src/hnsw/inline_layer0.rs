//! Contiguous per-node payload block for HNSW layer 0.
//!
//! ## Why this exists
//!
//! The legacy HNSW layout splits each node's data across several `Vec` fields
//! (`node_vectors`, `node_quantized`, `node_rabitq_codes`, `neighbours_l0`).
//! Visiting a neighbour during search therefore chases pointers through
//! several distinct heap allocations — one cache miss per field. Worse, under
//! multi-thread search those misses are uncorrelated across worker threads
//! because each query walks the graph in a different order, so the L3 fill
//! by one worker is rarely useful to another.
//!
//! hnswlib packs each node's neighbour ids, vector bytes and label into a
//! single per-node block addressed by stride arithmetic. Concurrent queries
//! reading the same node touch the same cache lines, and the hardware
//! prefetcher learns the stride. That layout is the documented reason
//! hnswlib shows super-linear MT4 scaling on workloads where CoordiNode
//! today plateaus around 0.88 per-core efficiency.
//!
//! This module is the structural building block for closing that gap: a
//! single `Box<[u64]>` allocation whose elements are addressed as
//! `idx * stride_bytes + field_offset`. It carries no HNSW algorithm logic
//! and does not own a graph; it is consumed by `HnswIndex` separately.
//!
//! ## Per-node block layout
//!
//! ```text
//! offset 0                  : [AtomicU64; m_max0]   neighbour ids
//! offset m_max0 * 8         : AtomicU8              neighbour_len
//! offset rabitq_offset      : [u8; rabitq_bytes]    RaBitQ 1-bit code
//! offset f32_offset         : [f32; dim]            exact vector
//! offset label_offset       : AtomicU64             external node label
//! ```
//!
//! Offsets are pre-computed in [`InlineLayer0::new`] and stored on the
//! struct so the hot path never recomputes them. Each offset is aligned
//! up to its field's natural alignment (8 for atomics, 4 for f32).
//!
//! ## Concurrency model
//!
//! - Neighbour ids, `neighbour_len` and `label` go through atomic accessors
//!   and are safe to read on `&self` from many threads concurrently.
//! - Payload bytes (RaBitQ code, f32 vector) are written via `&mut self`
//!   accessors. Search-time reads of those bytes happen under `&self` only
//!   after the writer phase has released its exclusive borrow.
//!
//! ## Wiring status
//!
//! As of this commit `InlineLayer0` is a standalone module with no users
//! inside `HnswIndex` yet. The follow-up wires the optional field onto the
//! index, populates it on insert, and then switches the search hot path to
//! it. Each of those steps is its own commit so main keeps a valid build
//! after every step.

#![allow(
    dead_code,
    reason = "standalone module landed alone; HnswIndex wiring lands in the next commit on the same plan"
)]

use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// 24-byte scalar header that travels alongside the packed RaBitQ code
/// in the contiguous store. Covers every numeric field the HNSW search
/// hot path reads on a neighbour visit, for every supported RaBitQ
/// variant:
/// - 1-bit (`RaBitQCode`): `norm`, `cross_term`, `signed_sum`, `correction`,
///   `radial`, `cluster_id` — the full chroma-style estimator inputs.
/// - Extended 2/3/4-bit (`RaBitQExtCode`): `norm` and `cross_term` are
///   meaningful; the other slots stay zero.
///
/// The layout is `#[repr(C)]` so a `core::ptr::read_unaligned` against
/// the stored bytes reconstructs the exact field positions. The
/// trailing `_pad` rounds the size up to a multiple of 4 so the
/// following `f32_vector` slot stays 4-aligned without per-call work.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct RaBitQScalars {
    pub norm: f32,
    pub cross_term: f32,
    pub signed_sum: i32,
    pub correction: f32,
    pub radial: f32,
    pub cluster_id: u16,
    pub _pad: u16,
}

const RABITQ_SCALARS_BYTES: usize = core::mem::size_of::<RaBitQScalars>();

/// Single contiguous-stride store for HNSW layer-0 per-node data.
///
/// See module docs for the per-node block layout. Accessor methods are
/// indexed by node id; they return atomic references for the concurrent
/// fields and `&[u8]` / `&[f32]` slices for the byte fields.
#[derive(Debug)]
pub struct InlineLayer0 {
    /// 8-byte-aligned backing. Length is `stride_bytes * capacity` rounded
    /// up to whole `u64` elements.
    backing: Box<[u64]>,
    /// Bytes per per-node block, rounded up to a multiple of 8.
    stride_bytes: usize,
    /// Maximum neighbours per layer-0 slot.
    m_max0: usize,
    /// Number of nodes the allocation can hold.
    capacity: usize,
    /// Bytes occupied by the packed RaBitQ code (`(dim * bits).div_ceil(8)`).
    /// `bits` is captured at construction time on `rabitq_bits` below.
    rabitq_bytes: usize,
    /// Bytes occupied by the f32 vector (`dim * 4`).
    f32_bytes: usize,
    /// Bit-width of the stored RaBitQ code. `1` for the SIGMOD 2024 sign-
    /// bit codec, `2..=4` for the Extended-RaBitQ codec.
    rabitq_bits: u8,
    /// Pre-computed offset of `neighbour_len` within a per-node block.
    neighbour_len_offset: usize,
    /// Pre-computed offset of `rabitq_code` within a per-node block.
    rabitq_offset: usize,
    /// Pre-computed offset of the `RaBitQScalars` header within a per-
    /// node block (immediately follows the packed code, aligned to 4).
    rabitq_scalars_offset: usize,
    /// Pre-computed offset of `f32_vector` within a per-node block.
    f32_offset: usize,
    /// Pre-computed offset of `label` within a per-node block.
    label_offset: usize,
}

#[inline(always)]
fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

impl InlineLayer0 {
    /// Allocate a fresh layer-0 store sized for `capacity` nodes, each with
    /// `m_max0` neighbour slots and a `dim`-dimensional vector. The
    /// backing memory is zeroed.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`, `capacity == 0`, or `m_max0 == 0`. The natural
    /// alignment guarantees (`u64` for backing) are encoded by using
    /// `Box<[u64]>` as the underlying storage; a non-aligned variant is
    /// impossible to construct.
    /// Backwards-compatible constructor that defaults to the 1-bit RaBitQ
    /// layout (`bits = 1`). Equivalent to
    /// `new_with_rabitq_bits(capacity, m_max0, dim, 1)`.
    pub fn new(capacity: usize, m_max0: usize, dim: usize) -> Self {
        Self::new_with_rabitq_bits(capacity, m_max0, dim, 1)
    }

    /// Allocate a fresh layer-0 store sized for `capacity` nodes with
    /// `m_max0` neighbour slots, a `dim`-dimensional f32 vector and an
    /// `rabitq_bits`-wide packed code per node. `rabitq_bits` must be in
    /// `{1, 2, 3, 4}`. The backing memory is zeroed.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`, `capacity == 0`, `m_max0 == 0`, or
    /// `rabitq_bits` is outside `1..=4`.
    #[allow(
        clippy::expect_used,
        reason = "construction path with checked arithmetic; failures are programmer errors and must abort"
    )]
    pub fn new_with_rabitq_bits(
        capacity: usize,
        m_max0: usize,
        dim: usize,
        rabitq_bits: u8,
    ) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(m_max0 > 0, "m_max0 must be > 0");
        assert!(dim > 0, "dim must be > 0");
        assert!(
            (1..=4).contains(&rabitq_bits),
            "rabitq_bits must be in 1..=4, got {rabitq_bits}"
        );

        let rabitq_bytes = dim
            .checked_mul(rabitq_bits as usize)
            .expect("dim * bits overflows usize")
            .div_ceil(8);
        let f32_bytes = dim.checked_mul(4).expect("dim * 4 overflows usize");

        // Layout: neighbours [AtomicU64; m_max0], neighbour_len AtomicU8,
        // rabitq_code [u8; rabitq_bytes], RaBitQScalars (24 B, 4-aligned),
        // f32_vector [f32; dim], label u64.
        let neighbour_len_offset = m_max0.checked_mul(8).expect("m_max0 * 8 overflows usize");
        let rabitq_offset = align_up(neighbour_len_offset + 1, 8);
        let rabitq_scalars_offset = align_up(
            rabitq_offset
                .checked_add(rabitq_bytes)
                .expect("rabitq end offset overflows usize"),
            4,
        );
        let f32_offset = align_up(
            rabitq_scalars_offset
                .checked_add(RABITQ_SCALARS_BYTES)
                .expect("rabitq scalars end overflows usize"),
            4,
        );
        let label_offset = align_up(
            f32_offset
                .checked_add(f32_bytes)
                .expect("f32 end offset overflows usize"),
            8,
        );
        let unaligned_stride = label_offset.checked_add(8).expect("stride overflows usize");
        let stride_bytes = align_up(unaligned_stride, 8);

        let total_bytes = stride_bytes
            .checked_mul(capacity)
            .expect("backing total bytes overflows usize");
        let u64_len = total_bytes.div_ceil(8);
        let backing: Box<[u64]> = vec![0u64; u64_len].into_boxed_slice();

        Self {
            backing,
            stride_bytes,
            m_max0,
            capacity,
            rabitq_bytes,
            f32_bytes,
            rabitq_bits,
            neighbour_len_offset,
            rabitq_offset,
            rabitq_scalars_offset,
            f32_offset,
            label_offset,
        }
    }

    /// Whether per-node blocks still carry the f32 vector (false after
    /// [`InlineLayer0::drop_f32`]).
    #[inline]
    pub fn has_f32(&self) -> bool {
        self.f32_bytes > 0
    }

    /// Free the f32 vectors, re-laying every per-node block out without the
    /// f32 slot so the bytes are returned to the allocator (the offload path
    /// calls this after calibration: search runs on quantized codes and rerank
    /// loads f32 from disk). The neighbours, neighbour_len, RaBitQ code, RaBitQ
    /// scalars, and label are preserved; the label moves up into the space the
    /// f32 vector vacated and the stride shrinks. Idempotent. Under `&mut
    /// self`, so no concurrent reader holds a pointer into the old backing.
    pub fn drop_f32(&mut self) {
        if self.f32_bytes == 0 {
            return;
        }
        // The prefix [0 .. f32_offset] (neighbours + len + rabitq + scalars)
        // keeps its layout; the label slides up to where the f32 vector was.
        let new_label_offset = align_up(self.f32_offset, 8);
        let new_stride = align_up(new_label_offset + 8, 8);
        let new_u64_len = (new_stride * self.capacity).div_ceil(8);
        let mut new_backing: Box<[u64]> = vec![0u64; new_u64_len].into_boxed_slice();
        // SAFETY: byte views span each backing's full allocation. Per node,
        // the prefix copy stays within `f32_offset < old stride`, and the
        // 8-byte label copy stays within both blocks (label_offset + 8 <=
        // old stride; new_label_offset + 8 <= new stride). `&mut self` means
        // no concurrent `&self` reader observes the swap.
        unsafe {
            let old = self.backing.as_ptr() as *const u8;
            let new = new_backing.as_mut_ptr() as *mut u8;
            for idx in 0..self.capacity {
                let old_base = idx * self.stride_bytes;
                let new_base = idx * new_stride;
                core::ptr::copy_nonoverlapping(
                    old.add(old_base),
                    new.add(new_base),
                    self.f32_offset,
                );
                core::ptr::copy_nonoverlapping(
                    old.add(old_base + self.label_offset),
                    new.add(new_base + new_label_offset),
                    8,
                );
            }
        }
        self.backing = new_backing;
        self.stride_bytes = new_stride;
        self.label_offset = new_label_offset;
        self.f32_bytes = 0;
    }

    /// Capacity in nodes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes per per-node block.
    #[inline]
    pub fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    /// Maximum neighbours per node at layer 0.
    #[inline]
    pub fn m_max0(&self) -> usize {
        self.m_max0
    }

    /// Vector dimension this layer was configured for. Derived from
    /// `f32_bytes / 4` so the value matches what `set_vector_f32` and
    /// `vector_f32` operate on.
    #[inline]
    pub fn dim(&self) -> usize {
        self.f32_bytes / 4
    }

    /// Bit-width of the stored RaBitQ code (1, 2, 3 or 4).
    #[inline]
    pub fn rabitq_bits(&self) -> u8 {
        self.rabitq_bits
    }

    /// Issue prefetch hints covering node `idx`'s neighbour-id region
    /// (`[AtomicU64; m_max0]` + the `neighbour_len` byte). The search
    /// loop reads this region immediately after popping a candidate;
    /// hinting the NEXT candidate's region while the current one is
    /// being expanded hides the random-access miss. Safe wrapper:
    /// `idx >= capacity` skips the hint. No-op on unsupported
    /// architectures; prefetch is a hint, never a dereference.
    #[inline(always)]
    pub fn prefetch_neighbours(&self, idx: usize) {
        if idx >= self.capacity {
            return;
        }
        // SAFETY: idx gate above; the span `m_max0 * 8 + 1` lies inside
        // the per-node block by construction (ids at offset 0, len byte
        // directly after).
        unsafe {
            let base = self.node_base_ptr(idx);
            let span = self.m_max0 * 8 + 1;
            let mut off = 0;
            while off < span {
                super::prefetch_read_data(base.add(off));
                off += 64;
            }
        }
    }

    /// Byte length of the packed RaBitQ code slot per node.
    ///
    /// Equal to `(dim * rabitq_bits).div_ceil(8)`. The contiguous search
    /// fast path uses this to confirm the inline slot is sized to match
    /// the query's bit-plane length before reinterpreting bytes as `&[u64]`.
    #[inline]
    pub fn rabitq_byte_len(&self) -> usize {
        self.rabitq_bytes
    }

    /// Read the per-node RaBitQ scalar header for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`. Same writer-coexistence rules as
    /// [`Self::rabitq`] and [`Self::vector_f32`]: payload bytes are
    /// installed under `&mut self`, search-time reads happen under
    /// `&self` after the writer phase released its borrow.
    #[inline]
    pub unsafe fn rabitq_scalars(&self, idx: usize) -> RaBitQScalars {
        // SAFETY: idx in-range per caller contract; rabitq_scalars_offset
        // sits inside the per-node block and is 4-aligned, the natural
        // alignment of `RaBitQScalars`. `read_unaligned` is paranoid for
        // future layout changes; reads of an aligned `#[repr(C)]` struct
        // are sound.
        unsafe {
            let p = self.node_base_ptr(idx).add(self.rabitq_scalars_offset) as *const RaBitQScalars;
            core::ptr::read_unaligned(p)
        }
    }

    /// Install the per-node RaBitQ scalar header for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`. Caller must hold `&mut self` for the
    /// duration of the write, which Rust's borrow checker enforces.
    #[inline]
    pub unsafe fn set_rabitq_scalars(&mut self, idx: usize, scalars: RaBitQScalars) {
        // SAFETY: see method doc. `write_unaligned` is paranoid; the
        // destination is 4-aligned by construction.
        unsafe {
            let p = (self.backing.as_mut_ptr() as *mut u8)
                .add(idx * self.stride_bytes + self.rabitq_scalars_offset)
                as *mut RaBitQScalars;
            core::ptr::write_unaligned(p, scalars);
        }
    }

    /// Base byte pointer for the per-node block at `idx`.
    ///
    /// # Safety
    ///
    /// Caller MUST ensure `idx < self.capacity()`.
    #[inline(always)]
    unsafe fn node_base_ptr(&self, idx: usize) -> *const u8 {
        debug_assert!(idx < self.capacity, "idx out of capacity");
        // SAFETY: backing was allocated with stride_bytes * capacity bytes
        // (rounded up), so byte_offset is always inside the allocation when
        // the debug_assert holds. The cast from `*const u64` to `*const u8`
        // is a stricter-to-looser alignment cast, always sound.
        let byte_offset = idx * self.stride_bytes;
        unsafe { (self.backing.as_ptr() as *const u8).add(byte_offset) }
    }

    /// Atomic reference to the `slot`-th neighbour id of node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()` and `slot < self.m_max0()`. The returned
    /// reference is valid as long as `self` is borrowed; no other code
    /// path mutates these bytes through non-atomic means while atomics
    /// are observable.
    #[inline(always)]
    pub unsafe fn neighbour(&self, idx: usize, slot: usize) -> &AtomicU64 {
        debug_assert!(slot < self.m_max0, "slot out of m_max0");
        // SAFETY: idx-bound delegated to node_base_ptr's contract; slot
        // bound enforced above. Per-node block starts with
        // [AtomicU64; m_max0]; offset slot*8 lands inside that array.
        // Alignment: backing is u64-aligned, slot*8 is multiple of 8,
        // node_base_ptr is multiple of stride_bytes which is multiple of 8,
        // so the resulting ptr is 8-byte aligned as AtomicU64 requires.
        unsafe {
            let p = self.node_base_ptr(idx).add(slot * 8) as *const AtomicU64;
            &*p
        }
    }

    /// Atomic reference to the `neighbour_len` byte of node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
    #[inline(always)]
    pub unsafe fn neighbour_len(&self, idx: usize) -> &AtomicU8 {
        // SAFETY: idx in-range per caller contract; neighbour_len_offset
        // lives inside the per-node block; AtomicU8 has alignment 1 so any
        // byte address is suitable.
        unsafe {
            let p = self.node_base_ptr(idx).add(self.neighbour_len_offset) as *const AtomicU8;
            &*p
        }
    }

    /// Atomic reference to the external `label` of node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
    #[inline(always)]
    pub unsafe fn label(&self, idx: usize) -> &AtomicU64 {
        // SAFETY: idx in-range per caller contract; label_offset is the
        // 8-byte aligned slot at the tail of the per-node block.
        unsafe {
            let p = self.node_base_ptr(idx).add(self.label_offset) as *const AtomicU64;
            &*p
        }
    }

    /// Read-only view of the RaBitQ code bytes for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`. No concurrent writer is touching the
    /// payload bytes through `&mut self`. Search-time reads come from
    /// `&self`; payload writes happen in `&mut self` accessors during the
    /// build phase, before the index is shared for search.
    #[inline]
    pub unsafe fn rabitq(&self, idx: usize) -> &[u8] {
        // SAFETY: see method-level doc. The byte range [rabitq_offset,
        // rabitq_offset + rabitq_bytes) lives inside the per-node block.
        unsafe {
            let p = self.node_base_ptr(idx).add(self.rabitq_offset);
            core::slice::from_raw_parts(p, self.rabitq_bytes)
        }
    }

    /// Read-only view of the f32 vector for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`. Same payload-writer constraint as
    /// [`Self::rabitq`].
    #[inline]
    pub unsafe fn vector_f32(&self, idx: usize) -> &[f32] {
        // SAFETY: f32_offset is 4-byte aligned (enforced in `new`), the byte
        // range is inside the per-node block, and the returned `&[f32]`
        // borrows `self`. Aliasing through `vector_f32_bytes_mut` is not
        // possible because that method takes `&mut self`.
        unsafe {
            let p = self.node_base_ptr(idx).add(self.f32_offset) as *const f32;
            let len = self.f32_bytes / 4;
            core::slice::from_raw_parts(p, len)
        }
    }

    /// Install the RaBitQ code for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()` and `code.len() == self.rabitq_bytes`.
    /// Writer must hold the exclusive `&mut self` borrow for the duration
    /// of the call; the contract is enforced by Rust's borrow checker.
    #[inline]
    pub unsafe fn set_rabitq(&mut self, idx: usize, code: &[u8]) {
        debug_assert_eq!(code.len(), self.rabitq_bytes, "rabitq len mismatch");
        // SAFETY: caller asserts idx in range; rabitq_offset + rabitq_bytes
        // is inside the per-node block by construction.
        unsafe {
            let p = (self.backing.as_mut_ptr() as *mut u8)
                .add(idx * self.stride_bytes + self.rabitq_offset);
            core::ptr::copy_nonoverlapping(code.as_ptr(), p, self.rabitq_bytes);
        }
    }

    /// Install the f32 vector for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()` and `vec.len() * 4 == self.f32_bytes`.
    #[inline]
    pub unsafe fn set_vector_f32(&mut self, idx: usize, vec: &[f32]) {
        debug_assert_eq!(vec.len() * 4, self.f32_bytes, "f32 dim mismatch");
        // SAFETY: caller asserts idx in range; the byte range
        // [f32_offset, f32_offset + f32_bytes) is inside the per-node block.
        // f32 has alignment 4, the destination ptr is 4-aligned (`new`
        // computed f32_offset that way), so the typed write is sound.
        unsafe {
            let p = (self.backing.as_mut_ptr() as *mut u8)
                .add(idx * self.stride_bytes + self.f32_offset) as *mut f32;
            core::ptr::copy_nonoverlapping(vec.as_ptr(), p, vec.len());
        }
    }

    /// Set the `slot`-th neighbour id of node `idx` via a relaxed atomic
    /// store. Convenience wrapper around [`Self::neighbour`].
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()` and `slot < self.m_max0()`.
    #[inline]
    pub unsafe fn set_neighbour(&self, idx: usize, slot: usize, value: u64) {
        // SAFETY: idx / slot bounds delegated to caller; neighbour() returns
        // a valid AtomicU64 reference under the same contract.
        unsafe {
            self.neighbour(idx, slot).store(value, Ordering::Relaxed);
        }
    }

    /// Set `neighbour_len` for node `idx` via a relaxed atomic store.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
    #[inline]
    pub unsafe fn set_neighbour_len(&self, idx: usize, len: u8) {
        // SAFETY: idx bound delegated to caller; neighbour_len() returns a
        // valid AtomicU8 reference under the same contract.
        unsafe {
            self.neighbour_len(idx).store(len, Ordering::Relaxed);
        }
    }

    /// Set `label` for node `idx` via a relaxed atomic store.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
    #[inline]
    pub unsafe fn set_label(&self, idx: usize, label: u64) {
        // SAFETY: idx bound delegated to caller; label() returns a valid
        // AtomicU64 reference under the same contract.
        unsafe {
            self.label(idx).store(label, Ordering::Relaxed);
        }
    }
}

// SAFETY: InlineLayer0 carries no thread-local state; the backing Box owns
// its allocation and is moved across threads as the struct moves. Sync
// holds because every cross-thread observable field is exposed through
// AtomicU{8,64} accessors; payload bytes are written under `&mut self`
// and read under `&self`, which Rust's borrow checker serialises.
unsafe impl Send for InlineLayer0 {}
unsafe impl Sync for InlineLayer0 {}

#[cfg(test)]
mod tests;
