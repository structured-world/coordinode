//! Layer-0 store: the f32 vector and the published layer-0 neighbour list of
//! every node, in segments that never move.
//!
//! ## Layout
//!
//! Nodes live in segments. The first segment holds `first_cap` nodes (the
//! index's expected size); segment `j + 1` starts at `first_cap << j` and holds
//! as many nodes again, so the total doubles with every segment. Within a
//! segment the f32 vectors sit in one block of `cap * stride` bytes,
//! `stride = align_up(dim * 4, 8)`, so a search visit reads a vector from one
//! stride-addressable allocation (hnswlib's `data_level0_memory_` idea for the
//! payload). An index that stays within its expected size uses the first
//! segment only, and the hot path pays one predictable comparison for the
//! others.
//!
//! Growth adds a segment and copies nothing: a vector or list a reader holds
//! keeps its address for the life of the store. A resizable block would move
//! payload behind live readers.
//!
//! Each node's layer-0 list is an [`AtomicNeighbourList`] of compact `u32`
//! ids: one immutable list behind one atomic descriptor, replaced whole by
//! compare-and-swap and reclaimed through the epoch. In-place slots and a
//! count let a reader during a replace see a list assembled from the old and
//! the new one; a descriptor to an immutable list does not.
//!
//! ## Concurrency
//!
//! Neighbour reads and writes and growth take `&self` and are safe under any
//! mix of readers and writers. A node's f32 vector is written once by the
//! writer that owns the node, before the node becomes reachable through a
//! link or the entry point ([`DataLevel0Block::set_vector`]); dropping the
//! vectors takes `&mut self`.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use super::M_MAX0;
use super::neighbours::AtomicNeighbourList;

/// Per-node vector alignment: keeps every node's f32 vector f32-aligned and
/// the next node on an 8-byte boundary.
const NODE_ALIGN: usize = 8;

/// Segments after the first. With `first_cap >= 1` they cover
/// `first_cap << 40` nodes, beyond any addressable store.
const MAX_EXTRA_SEGMENTS: usize = 40;

/// One run of nodes with stable addresses.
struct Segment {
    /// f32 vectors, `cap * stride` bytes as 8-byte words; empty once the
    /// vectors are dropped. `UnsafeCell` because a node's vector is written
    /// through `&self` by the writer that owns the node.
    vectors: Box<[UnsafeCell<u64>]>,
    /// The layer-0 neighbour list of every node in the segment.
    lists: Box<[AtomicNeighbourList<M_MAX0, u32>]>,
}

impl Segment {
    fn new(cap: usize, stride: usize) -> Self {
        // The stride is a multiple of NODE_ALIGN, so the division is exact.
        let Some(bytes) = cap.checked_mul(stride) else {
            Self::overflow();
        };
        let words = bytes / NODE_ALIGN;
        Self {
            vectors: (0..words).map(|_| UnsafeCell::new(0)).collect(),
            lists: (0..cap).map(|_| AtomicNeighbourList::new()).collect(),
        }
    }

    #[cold]
    #[allow(
        clippy::panic,
        reason = "a store beyond usize::MAX bytes is unreachable on real hardware and must abort"
    )]
    fn overflow() -> ! {
        panic!("DataLevel0Block: segment size overflows usize");
    }

    /// Base of the vector block.
    #[inline(always)]
    fn vector_base(&self) -> *mut u8 {
        UnsafeCell::raw_get(self.vectors.as_ptr()).cast()
    }
}

// SAFETY: the lists are `Sync`. The vector bytes are written only by the
// writer that owns a node, before the node is reachable, and read only after
// it was reached through a release-published link or entry point; every
// other access goes through `&mut self`.
unsafe impl Sync for Segment {}

/// The layer-0 store. See the module doc for layout and concurrency.
pub(super) struct DataLevel0Block {
    /// The first segment, `first_cap` nodes.
    first: Segment,
    /// Nodes in the first segment; segment `j + 1` starts at `first_cap << j`.
    first_cap: usize,
    /// Segments after the first, installed in order by `ensure_capacity`.
    extra: [AtomicPtr<Segment>; MAX_EXTRA_SEGMENTS],
    /// Nodes covered by the installed segments.
    capacity: AtomicUsize,
    /// Bytes per node's vector, a multiple of 8; 0 once the vectors are
    /// dropped.
    stride: usize,
    /// Maximum neighbours per layer-0 list (`m_max0`).
    m_max0: usize,
    /// Vector dimension.
    dim: usize,
    /// Whether the store still carries the f32 vectors. Set false by
    /// [`DataLevel0Block::drop_f32`] when the offload path frees them; then
    /// `vector_ptr` must not be called.
    has_f32: bool,
}

impl core::fmt::Debug for DataLevel0Block {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DataLevel0Block")
            .field("capacity", &self.capacity())
            .field("first_cap", &self.first_cap)
            .field("stride", &self.stride)
            .field("m_max0", &self.m_max0)
            .field("dim", &self.dim)
            .field("has_f32", &self.has_f32)
            .finish()
    }
}

#[inline(always)]
fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

impl DataLevel0Block {
    /// A store for `capacity` nodes in its first segment, with `m_max0`
    /// layer-0 neighbours each and `dim`-dimensional f32 vectors. The vectors
    /// are zeroed and every list is empty.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`, `capacity == 0`, `m_max0 == 0`, `m_max0 >
    /// M_MAX0`, or the segment size overflows.
    pub(super) fn new(capacity: usize, m_max0: usize, dim: usize) -> Self {
        assert!(dim > 0, "DataLevel0Block: dim must be > 0");
        assert!(capacity > 0, "DataLevel0Block: capacity must be > 0");
        assert!(m_max0 > 0, "DataLevel0Block: m_max0 must be > 0");
        assert!(
            m_max0 <= M_MAX0,
            "DataLevel0Block: m_max0 {m_max0} exceeds the list capacity {M_MAX0}"
        );
        let stride = align_up(dim * 4, NODE_ALIGN);
        Self {
            first: Segment::new(capacity, stride),
            first_cap: capacity,
            extra: core::array::from_fn(|_| AtomicPtr::new(core::ptr::null_mut())),
            capacity: AtomicUsize::new(capacity),
            stride,
            m_max0,
            dim,
            has_f32: true,
        }
    }

    /// Whether the store still carries the f32 vectors (false after
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
        self.first.vectors = Box::default();
        for slot in &mut self.extra {
            let ptr = *slot.get_mut();
            if !ptr.is_null() {
                // SAFETY: a non-null slot holds a segment leaked from a Box by
                // `ensure_capacity`; `&mut self` excludes every other access.
                unsafe { (*ptr).vectors = Box::default() };
            }
        }
        self.stride = 0;
        self.has_f32 = false;
    }

    /// Install segments until the store holds at least `required` nodes.
    /// Existing nodes keep their addresses. Safe under concurrent callers: a
    /// segment two callers allocate is installed once and the other copy is
    /// freed.
    ///
    /// # Panics
    ///
    /// Panics if `required` exceeds what the segments can address.
    #[allow(
        clippy::panic,
        reason = "growth past the addressable segments is unreachable on real hardware and must abort, not silently lose nodes"
    )]
    pub(super) fn ensure_capacity(&self, required: usize) {
        if required <= self.capacity.load(Ordering::Acquire) {
            return;
        }
        for (j, slot) in self.extra.iter().enumerate() {
            // Segment j starts at first_cap << j and covers as many nodes;
            // stop once its end no longer fits usize.
            let start = self.first_cap << j;
            if start >> j != self.first_cap || start.checked_mul(2).is_none() {
                break;
            }
            if slot.load(Ordering::Acquire).is_null() {
                let fresh = Box::into_raw(Box::new(Segment::new(start, self.stride)));
                if slot
                    .compare_exchange(
                        core::ptr::null_mut(),
                        fresh,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_err()
                {
                    // SAFETY: `fresh` came from `Box::into_raw` above and was
                    // never published.
                    drop(unsafe { Box::from_raw(fresh) });
                }
            }
            // Segment j covers [start, 2 * start).
            let covered = start * 2;
            self.capacity.fetch_max(covered, Ordering::AcqRel);
            if covered >= required {
                return;
            }
        }
        panic!("DataLevel0Block: {required} nodes exceed the addressable segments");
    }

    /// Capacity in node slots.
    #[inline]
    pub(super) fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Acquire)
    }

    /// Whether node `idx` has a slot. The hot path's bound check: a node in
    /// the first segment answers with a plain comparison, without the atomic
    /// load the compiler cannot merge across a search loop.
    #[inline(always)]
    pub(super) fn contains(&self, idx: usize) -> bool {
        idx < self.first_cap || idx < self.capacity.load(Ordering::Acquire)
    }

    /// Stride in bytes between node vectors.
    #[cfg(test)]
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

    /// The segment holding node `idx` and the node's offset in it.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
    #[inline(always)]
    unsafe fn locate(&self, idx: usize) -> (&Segment, usize) {
        if idx < self.first_cap {
            return (&self.first, idx);
        }
        // SAFETY: forwarded caller contract.
        unsafe { self.locate_extra(idx) }
    }

    /// [`Self::locate`] with the bound check folded in: `None` when `idx` has
    /// no slot. The hot path's single comparison for a first-segment node.
    #[inline(always)]
    fn try_locate(&self, idx: usize) -> Option<(&Segment, usize)> {
        if idx < self.first_cap {
            return Some((&self.first, idx));
        }
        self.try_locate_extra(idx)
    }

    /// [`Self::try_locate`] for nodes past the first segment.
    #[cold]
    #[inline(never)]
    fn try_locate_extra(&self, idx: usize) -> Option<(&Segment, usize)> {
        if idx >= self.capacity() {
            return None;
        }
        // SAFETY: first_cap <= idx < capacity per the gates.
        Some(unsafe { self.locate_extra(idx) })
    }

    /// The f32 vector of node `idx`, or `None` when `idx` has no slot or the
    /// vectors were dropped. The search hot path's read.
    #[inline(always)]
    pub(super) fn vector(&self, idx: usize) -> Option<&[f32]> {
        if !self.has_f32 {
            return None;
        }
        let (segment, off) = self.try_locate(idx)?;
        // SAFETY: the slot exists and the vectors are present, so `dim` f32
        // values start f32-aligned at `off * stride` inside the segment; the
        // borrow is tied to `&self` and the segment never moves.
        Some(unsafe {
            core::slice::from_raw_parts(
                segment.vector_base().add(off * self.stride) as *const f32,
                self.dim,
            )
        })
    }

    /// Read node `idx`'s published layer-0 list into `out` (cleared first),
    /// widening each id to `u64`, under the caller's pin. Returns `false`,
    /// leaving `out` empty, when `idx` has no slot.
    #[inline(always)]
    pub(super) fn read_list_u64(
        &self,
        idx: usize,
        out: &mut Vec<u64>,
        guard: &crossbeam_epoch::Guard,
    ) -> bool {
        out.clear();
        let Some((segment, off)) = self.try_locate(idx) else {
            return false;
        };
        // SAFETY: off lies inside the segment per try_locate.
        let ids = unsafe { segment.lists.get_unchecked(off) }.read(guard);
        out.extend(ids.iter().map(|&id| u64::from(id)));
        true
    }

    /// [`Self::locate`] for nodes past the first segment.
    ///
    /// # Safety
    ///
    /// `self.first_cap <= idx < self.capacity()`.
    #[cold]
    #[inline(never)]
    unsafe fn locate_extra(&self, idx: usize) -> (&Segment, usize) {
        debug_assert!(idx < self.capacity(), "idx out of bounds");
        let q = idx / self.first_cap;
        let j = (usize::BITS - 1 - q.leading_zeros()) as usize;
        let start = self.first_cap << j;
        let ptr = self.extra[j].load(Ordering::Acquire);
        debug_assert!(!ptr.is_null(), "segment {j} not installed");
        // SAFETY: idx < capacity means segment j was installed (capacity only
        // grows after the install) and is never freed before the store.
        (unsafe { &*ptr }, idx - start)
    }

    /// The neighbour list of node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
    #[inline(always)]
    unsafe fn list(&self, idx: usize) -> &AtomicNeighbourList<M_MAX0, u32> {
        // SAFETY: idx bound per contract; the offset lies inside the segment.
        unsafe {
            let (segment, off) = self.locate(idx);
            segment.lists.get_unchecked(off)
        }
    }

    /// Number of neighbours published for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
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
    /// `idx < self.capacity()`.
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
    /// `idx < self.capacity()`.
    #[cfg(test)]
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
        let Some((segment, off)) = self.try_locate(idx) else {
            return;
        };
        // SAFETY: off lies inside the segment per try_locate.
        let list = unsafe { segment.lists.get_unchecked(off) };
        if let Some(first) = list.read(guard).first() {
            super::prefetch_read_data((first as *const u32).cast());
        }
    }

    /// Publish `edit` of node `idx`'s layer-0 list: `edit` receives the
    /// current ids and returns the complete new list (at most `m_max0` ids),
    /// or `None` to keep it; a lost CAS reruns `edit` against the list that
    /// won. Returns whether a new list was published.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`.
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
    /// `idx < self.capacity()`.
    pub(super) unsafe fn cas_append_neighbour(&self, idx: usize, id: u32) -> bool {
        // SAFETY: idx bound per contract.
        unsafe { self.list(idx) }.cas_append_up_to(id, self.m_max0)
    }

    /// Install the f32 vector for node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()`, `vector.len() == self.dim()`, the store still
    /// carries the vectors, and the caller is the only writer of node `idx`
    /// and writes before the node is reachable by any reader (no link or
    /// entry point names it yet).
    #[inline]
    pub(super) unsafe fn set_vector(&self, idx: usize, vector: &[f32]) {
        debug_assert!(idx < self.capacity(), "idx out of bounds");
        debug_assert_eq!(vector.len(), self.dim, "vector dim mismatch");
        debug_assert!(self.has_f32, "vectors were dropped");
        // SAFETY: caller bounds and exclusivity; the stride is a multiple of
        // 8, so every node's vector starts f32-aligned inside its segment.
        unsafe {
            let (segment, off) = self.locate(idx);
            let dst = segment.vector_base().add(off * self.stride) as *mut f32;
            core::ptr::copy_nonoverlapping(vector.as_ptr(), dst, self.dim);
        }
    }

    /// Read-only pointer to the f32 vector of node `idx`.
    ///
    /// # Safety
    ///
    /// `idx < self.capacity()` and the store still carries the vectors. The
    /// caller treats the pointer as a `&[f32; dim]` borrow of `self`.
    #[cfg(test)]
    pub(super) unsafe fn vector_ptr(&self, idx: usize) -> *const f32 {
        // SAFETY: caller bounds; the vector lies inside its segment.
        unsafe {
            let (segment, off) = self.locate(idx);
            segment.vector_base().add(off * self.stride) as *const f32
        }
    }

    /// Prefetch hints covering the whole f32 vector of node `idx`, one per
    /// 64-byte line, so the distance kernel finds every line warm. No-op when
    /// `idx` is out of range or the vectors were dropped.
    #[inline(always)]
    pub(super) fn prefetch(&self, idx: usize) {
        if !self.has_f32 {
            return;
        }
        let Some((segment, off)) = self.try_locate(idx) else {
            return;
        };
        // SAFETY: the slot exists and the vectors are present, so the
        // `dim * 4` span lies inside the segment; prefetch never dereferences.
        unsafe {
            let base = segment.vector_base().add(off * self.stride) as *const u8;
            let span = self.dim * core::mem::size_of::<f32>();
            let mut off = 0;
            while off < span {
                super::prefetch_read_data(base.add(off));
                off += 64;
            }
        }
    }
}

impl Drop for DataLevel0Block {
    fn drop(&mut self) {
        for slot in &mut self.extra {
            let ptr = *slot.get_mut();
            if !ptr.is_null() {
                // SAFETY: a non-null slot holds a segment leaked from a Box by
                // `ensure_capacity`, installed once and freed only here.
                drop(unsafe { Box::from_raw(ptr) });
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
