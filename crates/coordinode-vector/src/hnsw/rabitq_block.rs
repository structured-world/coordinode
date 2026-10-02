//! Contiguous per-node RaBitQ code store for the cosine search fast path.
//!
//! Each node's packed RaBitQ code and its scalar header sit next to each other
//! in one `Box<[u64]>` addressed as `idx * stride_bytes + field_offset`, so a
//! neighbour visit reads both from one cache-line run instead of chasing the
//! per-node `Vec` behind the code in the SoA store.
//!
//! The block holds codes only: layer-0 neighbour lists and f32 vectors live in
//! the layer-0 store, and nothing else is duplicated here.
//!
//! ## Per-node block layout
//!
//! ```text
//! offset 0                     : [u8; rabitq_bytes]   packed RaBitQ code
//! offset rabitq_scalars_offset : RaBitQScalars        estimator inputs
//! ```
//!
//! The code starts each block, so it is 8-aligned (the stride is a multiple
//! of 8) and the 1-bit search can read it as `&[u64]`; the scalars follow,
//! 4-aligned.
//!
//! ## Concurrency model
//!
//! A node's slot is written through `&self` by the writer that owns the node,
//! before the node is reachable, or by a calibration that holds the index
//! exclusively; search reads it after reaching the node through a
//! release-published link or entry point.

use core::cell::UnsafeCell;

/// 24-byte scalar header that travels alongside the packed RaBitQ code.
/// Covers every numeric field the HNSW search hot path reads on a neighbour
/// visit, for every supported RaBitQ variant:
/// - 1-bit (`RaBitQCode`): `norm`, `cross_term`, `signed_sum`, `correction`,
///   `radial`, `cluster_id`: the full chroma-style estimator inputs.
/// - Extended 2/3/4-bit (`RaBitQExtCode`): `norm` and `cross_term` are
///   meaningful; the other slots stay zero.
///
/// The layout is `#[repr(C)]` so a `core::ptr::read_unaligned` against the
/// stored bytes reconstructs the exact field positions. The trailing `_pad`
/// rounds the size up to a multiple of 4.
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

/// Stride-addressed store of every node's RaBitQ code and scalar header.
pub struct RabitqBlock {
    /// 8-byte-aligned backing of `stride_bytes * capacity` bytes. `UnsafeCell`
    /// because a node's slot is written through `&self` by its owner.
    backing: Box<[UnsafeCell<u64>]>,
    /// Bytes per per-node block, a multiple of 8.
    stride_bytes: usize,
    /// Number of nodes the allocation can hold.
    capacity: usize,
    /// Bytes of the packed code (`(dim * bits).div_ceil(8)`).
    rabitq_bytes: usize,
    /// Bit-width of the stored code: `1` for the SIGMOD 2024 sign-bit codec,
    /// `2..=4` for the Extended-RaBitQ codec.
    rabitq_bits: u8,
    /// Offset of the `RaBitQScalars` header within a per-node block.
    rabitq_scalars_offset: usize,
}

// SAFETY: a node's slot is written only by the writer that owns the node
// before the node is reachable, or under exclusive access, and read only after
// the node was reached through a release-published link or entry point; see
// the module doc.
unsafe impl Sync for RabitqBlock {}

impl core::fmt::Debug for RabitqBlock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RabitqBlock")
            .field("capacity", &self.capacity)
            .field("stride_bytes", &self.stride_bytes)
            .field("rabitq_bits", &self.rabitq_bits)
            .finish()
    }
}

#[inline(always)]
fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

impl RabitqBlock {
    /// A zeroed 1-bit store for `capacity` nodes of dimension `dim`.
    #[cfg(test)]
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self::new_with_rabitq_bits(capacity, dim, 1)
    }

    /// A zeroed store for `capacity` nodes with a `dim`-dimensional,
    /// `rabitq_bits`-wide packed code per node.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`, `dim == 0`, `rabitq_bits` is outside `1..=4`,
    /// or the block size overflows `usize`.
    #[allow(
        clippy::expect_used,
        reason = "construction path with checked arithmetic; failures are programmer errors and must abort"
    )]
    pub fn new_with_rabitq_bits(capacity: usize, dim: usize, rabitq_bits: u8) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(dim > 0, "dim must be > 0");
        assert!(
            (1..=4).contains(&rabitq_bits),
            "rabitq_bits must be in 1..=4, got {rabitq_bits}"
        );

        let rabitq_bytes = dim
            .checked_mul(rabitq_bits as usize)
            .expect("dim * bits overflows usize")
            .div_ceil(8);
        let rabitq_scalars_offset = align_up(rabitq_bytes, 4);
        let stride_bytes = align_up(
            rabitq_scalars_offset
                .checked_add(RABITQ_SCALARS_BYTES)
                .expect("stride overflows usize"),
            8,
        );
        let total_bytes = stride_bytes
            .checked_mul(capacity)
            .expect("backing total bytes overflows usize");

        Self {
            backing: (0..total_bytes / 8).map(|_| UnsafeCell::new(0)).collect(),
            stride_bytes,
            capacity,
            rabitq_bytes,
            rabitq_bits,
            rabitq_scalars_offset,
        }
    }

    /// Capacity in nodes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes per per-node block.
    #[cfg(test)]
    pub fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    /// Bit-width of the stored code (1, 2, 3 or 4).
    #[inline]
    pub fn rabitq_bits(&self) -> u8 {
        self.rabitq_bits
    }

    /// Byte length of the packed code slot per node, `(dim * bits).div_ceil(8)`.
    /// The search fast path checks it against the query's bit-plane length
    /// before reading the slot as `&[u64]`.
    #[inline]
    pub fn rabitq_byte_len(&self) -> usize {
        self.rabitq_bytes
    }

    /// Base byte pointer for the per-node block at `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
    #[inline(always)]
    unsafe fn node_base_ptr(&self, idx: usize) -> *mut u8 {
        debug_assert!(idx < self.capacity, "idx out of capacity");
        // SAFETY: the backing holds stride_bytes * capacity bytes, so the
        // offset of an in-range idx is inside the allocation.
        unsafe {
            UnsafeCell::raw_get(self.backing.as_ptr())
                .cast::<u8>()
                .add(idx * self.stride_bytes)
        }
    }

    /// The packed code bytes of node `idx`, 8-aligned.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
    #[inline]
    pub unsafe fn rabitq(&self, idx: usize) -> &[u8] {
        // SAFETY: the code occupies [0, rabitq_bytes) of the per-node block.
        unsafe { core::slice::from_raw_parts(self.node_base_ptr(idx), self.rabitq_bytes) }
    }

    /// The scalar header of node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
    #[inline]
    pub unsafe fn rabitq_scalars(&self, idx: usize) -> RaBitQScalars {
        // SAFETY: the header lies inside the per-node block at a 4-aligned
        // offset; `read_unaligned` does not depend on that alignment.
        unsafe {
            let p = self.node_base_ptr(idx).add(self.rabitq_scalars_offset) as *const RaBitQScalars;
            core::ptr::read_unaligned(p)
        }
    }

    /// Install the packed code of node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`, `code.len() == self.rabitq_byte_len()`, and
    /// the caller is the only writer of node `idx` and writes before the node
    /// is reachable, or holds the block exclusively.
    #[inline]
    pub unsafe fn set_rabitq(&self, idx: usize, code: &[u8]) {
        debug_assert_eq!(code.len(), self.rabitq_bytes, "rabitq len mismatch");
        // SAFETY: caller bounds and exclusivity; the destination is the code
        // slot.
        unsafe {
            let p = self.node_base_ptr(idx);
            core::ptr::copy_nonoverlapping(code.as_ptr(), p, self.rabitq_bytes);
        }
    }

    /// Install the scalar header of node `idx`.
    ///
    /// # Safety
    ///
    /// As for [`Self::set_rabitq`].
    #[inline]
    pub unsafe fn set_rabitq_scalars(&self, idx: usize, scalars: RaBitQScalars) {
        // SAFETY: caller bounds and exclusivity; the destination is the
        // header slot.
        unsafe {
            let p = self.node_base_ptr(idx).add(self.rabitq_scalars_offset) as *mut RaBitQScalars;
            core::ptr::write_unaligned(p, scalars);
        }
    }
}

#[cfg(test)]
mod tests;
