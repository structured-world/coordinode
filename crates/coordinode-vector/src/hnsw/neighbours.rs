//! One node's neighbour list at one layer, published whole.
//!
//! The list is an immutable allocation behind one atomic descriptor. A writer
//! builds the complete new list privately from the protected current one plus
//! its edit and publishes it with one compare-and-swap of the descriptor; a
//! writer that loses the CAS rebuilds against the list that won, so a
//! concurrent accepted edit is never overwritten. A reader acquire-loads the
//! descriptor and walks that one list: it sees the list before or after a
//! replace, never one assembled from both, and never retries because a writer
//! is working. A replaced list is retired through epoch-based reclamation and
//! freed only once no pinned reader or writer can still hold it.
//!
//! Upper layers hold `u64` ids; layer 0 holds compact `u32` ids, which halve
//! the bytes of the densest lists.
//!
//! Under `--cfg loom --cfg crossbeam_loom` the epoch machinery itself runs on
//! loom's atomics, so `tests/loom_neighbours.rs` checks this exact publication
//! path.

use std::sync::Arc;

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};

use super::stats::PublicationStats;

#[cfg(loom)]
use loom::sync::atomic::Ordering;
#[cfg(not(loom))]
use std::sync::atomic::Ordering;

/// One published neighbour list. Immutable once its descriptor points at it.
struct List<T> {
    ids: Box<[T]>,
}

/// Neighbour list of one HNSW node at one layer, holding at most `N` ids of
/// type `T`.
///
/// # Concurrency contract
///
/// * Readers are wait-free and see one whole published list.
/// * [`set`](Self::set) and [`cas_append`](Self::cas_append) are both safe
///   under concurrent writers: each publishes one complete list by CAS and
///   retries against the current list when another writer won.
#[doc(hidden)]
pub struct AtomicNeighbourList<const N: usize, T = u64> {
    /// The published list; null is the empty list.
    current: Atomic<List<T>>,
}

impl<const N: usize, T: Copy + PartialEq> AtomicNeighbourList<N, T> {
    /// An empty neighbour list.
    pub fn new() -> Self {
        Self {
            current: Atomic::null(),
        }
    }

    /// Capacity of the list (compile-time constant `N`).
    #[cfg(test)]
    pub(crate) const fn capacity(&self) -> usize {
        N
    }

    /// The ids of the list published when `guard` loaded it, with the
    /// descriptor value they came from. The slice lives as long as the guard
    /// keeps the list from being reclaimed.
    #[inline]
    fn view<'g>(&self, guard: &'g Guard) -> (Shared<'g, List<T>>, &'g [T]) {
        let shared = self.current.load(Ordering::Acquire, guard);
        // SAFETY: a non-null descriptor points at a list that was fully
        // initialized before its Release CAS made it reachable, and the guard
        // keeps it alive until the guard is dropped: a replaced list is only
        // retired through `defer_destroy`, which waits for every pin taken
        // before the retirement.
        let ids = unsafe { shared.as_ref() }.map_or(&[][..], |list| &list.ids);
        (shared, ids)
    }

    /// The published ids, protected by `guard`. A caller that reads many
    /// lists (a search) pins once and reads them all under one guard.
    #[inline]
    pub(crate) fn read<'g>(&self, guard: &'g Guard) -> &'g [T] {
        self.view(guard).1
    }

    /// Current number of neighbours.
    #[inline]
    pub fn len(&self) -> usize {
        let guard = epoch::pin();
        self.read(&guard).len()
    }

    /// Whether the list holds no neighbours.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy the published list into `out`, clearing it first.
    pub fn snapshot_into(&self, out: &mut Vec<T>) {
        out.clear();
        let guard = epoch::pin();
        out.extend_from_slice(self.read(&guard));
    }

    /// Allocating variant of [`snapshot_into`](Self::snapshot_into).
    pub fn snapshot(&self) -> Vec<T> {
        let mut out = Vec::new();
        self.snapshot_into(&mut out);
        out
    }

    /// Publish `edit` of the current list. `edit` receives the protected
    /// current ids and returns the complete new list, or `None` to leave the
    /// list as it is. A lost CAS calls `edit` again with the list that won,
    /// so an edit computed from a list another writer replaced is never
    /// published. Returns whether a new list was published.
    ///
    /// Public for the interleaving model checks of this exact path, like the
    /// type itself.
    pub fn update(&self, edit: impl FnMut(&[T]) -> Option<Box<[T]>>) -> bool {
        self.update_with(edit, None)
    }

    /// [`update`](Self::update), accounting the replaced list and any lost
    /// CAS in `stats` when given.
    pub(crate) fn update_with(
        &self,
        mut edit: impl FnMut(&[T]) -> Option<Box<[T]>>,
        stats: Option<&Arc<PublicationStats>>,
    ) -> bool {
        let guard = epoch::pin();
        let (mut expected, mut ids) = self.view(&guard);
        loop {
            let Some(next) = edit(ids) else {
                return false;
            };
            debug_assert!(next.len() <= N, "a neighbour list holds at most {N} ids");
            match self.current.compare_exchange(
                expected,
                Owned::new(List { ids: next }),
                Ordering::AcqRel,
                Ordering::Acquire,
                &guard,
            ) {
                Ok(_) => {
                    if !expected.is_null() {
                        // SAFETY: the CAS unlinked `expected`, so no new
                        // reader can reach it; readers that loaded it before
                        // are pinned, and the deferred drop waits for them.
                        unsafe { Self::retire(expected, &guard, stats) };
                    }
                    return true;
                }
                Err(lost) => {
                    if let Some(stats) = stats {
                        stats.cas_lost();
                    }
                    // The unpublished candidate is dropped with `lost.new`;
                    // rebuild against the list that won.
                    expected = lost.current;
                    // SAFETY: as in `view`: the winner was initialized before
                    // its publishing CAS and the guard protects it.
                    ids = unsafe { expected.as_ref() }.map_or(&[][..], |list| &list.ids);
                }
            }
        }
    }

    /// Hand the unlinked `list` to the epoch: freed once no pin taken before
    /// now remains, and counted in `stats` until then.
    ///
    /// # Safety
    ///
    /// `list` is non-null and was unlinked by a CAS under `guard`, so no
    /// reader pinned after it can reach it.
    unsafe fn retire(
        list: Shared<'_, List<T>>,
        guard: &Guard,
        stats: Option<&Arc<PublicationStats>>,
    ) {
        let Some(stats) = stats else {
            // SAFETY: forwarded caller contract.
            unsafe { guard.defer_destroy(list) };
            return;
        };
        // SAFETY: non-null per the contract, and the guard keeps it alive.
        let len = unsafe { list.deref() }.ids.len();
        let bytes = core::mem::size_of::<List<T>>() + len * core::mem::size_of::<T>();
        stats.list_retired(bytes);
        let stats = Arc::clone(stats);
        let raw = list.as_raw() as usize;
        // SAFETY: the closure frees the list exactly once, after every pin
        // that could hold it is gone; it captures only owned data.
        unsafe {
            guard.defer_unchecked(move || {
                drop(Owned::from_raw(raw as *mut List<T>));
                stats.list_reclaimed(bytes);
            });
        }
    }

    /// Replace the whole list with `new`, truncated to `N` ids.
    pub fn set(&self, new: &[T]) {
        debug_assert!(
            new.len() <= N,
            "AtomicNeighbourList<{N}>::set received {} neighbours, would truncate",
            new.len()
        );
        let n = new.len().min(N);
        self.update(|current| (current != &new[..n]).then(|| new[..n].into()));
    }

    /// Append `id` under concurrent writers. Returns `false` when the list is
    /// already full; the caller then runs its prune protocol.
    pub fn cas_append(&self, id: T) -> bool {
        self.cas_append_up_to(id, N)
    }

    /// [`cas_append`](Self::cas_append) with a capacity below `N` chosen at
    /// run time (layer 0 holds `m_max0` ids, which the index configures).
    pub(crate) fn cas_append_up_to(&self, id: T, cap: usize) -> bool {
        self.cas_append_up_to_with(id, cap, None)
    }

    /// [`cas_append_up_to`](Self::cas_append_up_to) with the accounting of
    /// [`update_with`](Self::update_with).
    pub(crate) fn cas_append_up_to_with(
        &self,
        id: T,
        cap: usize,
        stats: Option<&Arc<PublicationStats>>,
    ) -> bool {
        let cap = cap.min(N);
        let mut full = false;
        self.update_with(
            |current| {
                full = current.len() >= cap;
                (!full).then(|| {
                    let mut next = Vec::with_capacity(current.len() + 1);
                    next.extend_from_slice(current);
                    next.push(id);
                    next.into_boxed_slice()
                })
            },
            stats,
        );
        !full
    }

    /// Replace the entire list contents: [`set`](Self::set).
    #[cfg(test)]
    pub fn replace(&self, new: &[T]) {
        self.set(new);
    }
}

impl<const N: usize, T> Drop for AtomicNeighbourList<N, T> {
    fn drop(&mut self) {
        // SAFETY: `&mut self` means no reader or writer of this list exists
        // any more; lists replaced earlier were handed to the epoch already.
        unsafe {
            let guard = epoch::unprotected();
            let current = self.current.load(Ordering::Relaxed, guard);
            if !current.is_null() {
                drop(current.into_owned());
            }
        }
    }
}

impl<const N: usize, T: Copy + PartialEq> Default for AtomicNeighbourList<N, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, T: Copy + PartialEq + std::fmt::Debug> std::fmt::Debug
    for AtomicNeighbourList<N, T>
{
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
