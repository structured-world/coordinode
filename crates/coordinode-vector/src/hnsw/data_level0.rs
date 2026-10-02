//! Layer-0 store: one contiguous f32 block plus one published neighbour list
//! per node.
//!
//! ## Layout
//!
//! The f32 vectors of all nodes sit in one `Box<[u8]>` of `capacity * stride`
//! bytes, `stride = align_up(dim * 4, 8)`, so a search visit reads a vector
//! from one stride-addressable allocation (hnswlib's `data_level0_memory_`
//! idea for the payload).
//!
//! Neighbours do not live in that block. Each node's layer-0 list is an
//! [`AtomicNeighbourList`] of compact `u32` ids: one immutable list behind one
//! atomic descriptor, replaced whole by compare-and-swap and reclaimed through
//! the epoch. In-place slots and a count in the block let a reader during a
//! replace see a list assembled from the old and the new one; a descriptor to
//! an immutable list does not.
//!
//! ## Concurrency
//!
//! Neighbour reads and writes take `&self` and are safe under any mix of
//! readers and writers. The f32 vector is written once under `&mut self`
//! ([`DataLevel0Block::set_vector`]) and read under `&self`
//! ([`DataLevel0Block::vector_ptr`]); growth and dropping the vectors take
//! `&mut self`.

#![allow(
    dead_code,
    reason = "consumers land in the search-path wiring chunks on the same plan; tests exercise the round-trip"
)]

use super::M_MAX0;
use super::neighbours::AtomicNeighbourList;

/// Per-node vector alignment: keeps every node's f32 vector f32-aligned and
/// the next node on an 8-byte boundary.
const NODE_ALIGN: usize = 8;

/// The layer-0 store. See the module doc for layout and concurrency.
#[derive(Debug)]
pub(super) struct DataLevel0Block {
    /// f32 vectors, `capacity * stride` bytes; empty once dropped.
    backing: Box<[u8]>,
    /// Number of node slots.
    capacity: usize,
    /// Bytes per node's vector, rounded up to a multiple of 8; 0 once the
    /// vectors are dropped.
    stride: usize,
    /// Maximum neighbours per layer-0 list (`m_max0`).
    m_max0: usize,
    /// Vector dimension.
    dim: usize,
    /// Whether the block still carries the f32 vectors. Set false by
    /// [`DataLevel0Block::drop_f32`] when the offload path frees them; then
    /// `vector_ptr` must not be called.
    has_f32: bool,
    /// The layer-0 neighbour list of every node, indexed like the vectors.
    neighbours: Vec<AtomicNeighbourList<M_MAX0, u32>>,
}

#[inline(always)]
fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

impl DataLevel0Block {
    /// A store for `capacity` nodes with `m_max0` layer-0 neighbours each and
    /// `dim`-dimensional f32 vectors. The vectors are zeroed and every list is
    /// empty.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`, `capacity == 0`, `m_max0 == 0`, `m_max0 >
    /// M_MAX0`, or the block size overflows.
    #[allow(
        clippy::panic,
        reason = "construction failure (overflow / zero arg) is a programmer error and must abort"
    )]
    pub(super) fn new(capacity: usize, m_max0: usize, dim: usize) -> Self {
        assert!(dim > 0, "DataLevel0Block: dim must be > 0");
        assert!(capacity > 0, "DataLevel0Block: capacity must be > 0");
        assert!(m_max0 > 0, "DataLevel0Block: m_max0 must be > 0");
        assert!(
            m_max0 <= M_MAX0,
            "DataLevel0Block: m_max0 {m_max0} exceeds the list capacity {M_MAX0}"
        );

        let stride = align_up(dim * 4, NODE_ALIGN);
        let Some(total) = stride.checked_mul(capacity) else {
            panic!("DataLevel0Block: stride * capacity overflow");
        };
        Self {
            backing: vec![0u8; total].into_boxed_slice(),
            capacity,
            stride,
            m_max0,
            dim,
            has_f32: true,
            neighbours: (0..capacity).map(|_| AtomicNeighbourList::new()).collect(),
        }
    }

    /// Whether the block still carries the f32 vectors (false after
    /// [`DataLevel0Block::drop_f32`]).
    #[inline]
    pub(super) fn has_f32(&self) -> bool {
        self.has_f32
    }

    /// Free the f32 vectors (the offload path calls this after calibration:
    /// search runs on quantized codes and rerank loads f32 from disk). The
    /// neighbour lists are untouched. Idempotent. After this, `has_f32` is
    /// false and `vector_ptr` must not be called.
    pub(super) fn drop_f32(&mut self) {
        if !self.has_f32 {
            return;
        }
        self.backing = Box::default();
        self.stride = 0;
        self.has_f32 = false;
    }

    /// Grow the store so it holds at least `required` nodes, copying the
    /// vectors over. Capacity at least doubles to amortise repeated growth.
    /// Under `&mut self`, so no reader holds a pointer into the old block.
    ///
    /// # Panics
    ///
    /// Panics if `stride * new_capacity` overflows `usize`.
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
        self.neighbours
            .resize_with(new_capacity, AtomicNeighbourList::new);
        self.capacity = new_capacity;
    }

    /// Capacity in node slots.
    #[inline]
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Stride in bytes between node vectors.
    #[inline]
    pub(super) fn stride(&self) -> usize {
        self.stride
    }

    /// Vector dimension.
    #[inline]
    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    /// Maximum neighbours per layer-0 list.
    #[inline]
    pub(super) fn m_max0(&self) -> usize {
        self.m_max0
    }

    /// The neighbour list of node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    #[inline(always)]
    unsafe fn list(&self, idx: usize) -> &AtomicNeighbourList<M_MAX0, u32> {
        debug_assert!(idx < self.capacity, "idx out of bounds");
        // SAFETY: the caller guarantees idx < capacity == neighbours.len().
        unsafe { self.neighbours.get_unchecked(idx) }
    }

    /// Number of neighbours published for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    #[inline]
    pub(super) unsafe fn neighbour_count(&self, idx: usize) -> u32 {
        // SAFETY: idx bound per contract; a list never exceeds M_MAX0 ids.
        unsafe { self.list(idx) }.len() as u32
    }

    /// Replace the whole layer-0 list of node `idx` with `ids`, published as
    /// one list: a concurrent reader sees the old list or this one.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    #[inline]
    pub(super) unsafe fn set_neighbours(&self, idx: usize, ids: &[u32]) {
        debug_assert!(ids.len() <= self.m_max0, "ids overflow neighbour slot");
        let n = ids.len().min(self.m_max0);
        // SAFETY: idx bound per contract.
        unsafe { self.list(idx) }.set(&ids[..n]);
    }

    /// Copy the published layer-0 list of node `idx` into `out`, cleared
    /// first.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    #[inline]
    pub(super) unsafe fn read_neighbours_into(&self, idx: usize, out: &mut Vec<u32>) {
        // SAFETY: idx bound per contract.
        unsafe { self.list(idx) }.snapshot_into(out);
    }

    /// Prefetch hint for node `idx`'s published list, read on the next search
    /// visit: the descriptor is loaded under the caller's pin and the list's
    /// ids are hinted, so the visit does not wait on the second, dependent
    /// miss. No-op when `idx` is out of range.
    #[inline(always)]
    pub(super) fn prefetch_neighbours(&self, idx: usize, guard: &crossbeam_epoch::Guard) {
        if let Some(list) = self.neighbours.get(idx) {
            if let Some(first) = list.read(guard).first() {
                super::prefetch_read_data((first as *const u32).cast());
            }
        }
    }

    /// Like [`Self::read_neighbours_into`] but widening each id to `u64` for
    /// the search hot path's candidate buffer, reading straight into `out`
    /// under the caller's pin (one per search pass).
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    #[inline]
    pub(super) unsafe fn read_neighbours_into_u64(
        &self,
        idx: usize,
        out: &mut Vec<u64>,
        guard: &crossbeam_epoch::Guard,
    ) {
        out.clear();
        // SAFETY: idx bound per contract.
        let ids = unsafe { self.list(idx) }.read(guard);
        out.extend(ids.iter().map(|&id| u64::from(id)));
    }

    /// Publish `edit` of node `idx`'s layer-0 list: `edit` receives the
    /// current ids and returns the complete new list (at most `m_max0` ids),
    /// or `None` to keep it; a lost CAS reruns `edit` against the list that
    /// won. Returns whether a new list was published.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    pub(super) unsafe fn update_neighbours(
        &self,
        idx: usize,
        edit: impl FnMut(&[u32]) -> Option<Box<[u32]>>,
    ) -> bool {
        // SAFETY: idx bound per contract.
        unsafe { self.list(idx) }.update(edit)
    }

    /// Append `id` to node `idx`'s layer-0 list under concurrent writers.
    /// Returns `false` when the list already holds `m_max0` ids; the caller
    /// runs its prune protocol.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`.
    pub(super) unsafe fn cas_append_neighbour(&self, idx: usize, id: u32) -> bool {
        // SAFETY: idx bound per contract.
        unsafe { self.list(idx) }.cas_append_up_to(id, self.m_max0)
    }

    /// Install the f32 vector for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity`, `vector.len() == self.dim` and the block still
    /// carries the vectors.
    #[inline]
    pub(super) unsafe fn set_vector(&mut self, idx: usize, vector: &[f32]) {
        debug_assert!(idx < self.capacity, "idx out of bounds");
        debug_assert_eq!(vector.len(), self.dim, "vector dim mismatch");
        debug_assert!(self.has_f32, "vectors were dropped");
        // SAFETY: caller guarantees the bounds; the stride is a multiple of 8,
        // so every node's vector starts f32-aligned.
        unsafe {
            let dst = self.backing.as_mut_ptr().add(idx * self.stride) as *mut f32;
            core::ptr::copy_nonoverlapping(vector.as_ptr(), dst, self.dim);
        }
    }

    /// Read-only pointer to the f32 vector of node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity` and the block still carries the vectors. The
    /// caller treats the pointer as a `&[f32; dim]` borrow of `self`.
    #[inline]
    pub(super) unsafe fn vector_ptr(&self, idx: usize) -> *const f32 {
        // SAFETY: caller guarantees idx bounds; the vector lies inside the
        // block by construction.
        unsafe { self.backing.as_ptr().add(idx * self.stride) as *const f32 }
    }

    /// Prefetch hints covering the whole f32 vector of node `idx`, one per
    /// 64-byte line, so the distance kernel finds every line warm. No-op when
    /// `idx` is out of range or the vectors were dropped.
    #[inline(always)]
    pub(super) fn prefetch(&self, idx: usize) {
        if idx >= self.capacity || !self.has_f32 {
            return;
        }
        // SAFETY: idx < capacity and the vectors are present, so the `dim * 4`
        // span lies inside the block; prefetch never dereferences.
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
