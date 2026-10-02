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
//! Under `--cfg loom --cfg crossbeam_loom` the epoch machinery itself runs on
//! loom's atomics, so `tests/loom_neighbours.rs` checks this exact publication
//! path.

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};

#[cfg(loom)]
use loom::sync::atomic::Ordering;
#[cfg(not(loom))]
use std::sync::atomic::Ordering;

/// One published neighbour list. Immutable once its descriptor points at it.
struct List {
    ids: Box<[u64]>,
}

/// Neighbour list of one HNSW node at one layer, holding at most `N` ids.
///
/// # Concurrency contract
///
/// * Readers are wait-free and see one whole published list.
/// * [`set`](Self::set) and [`cas_append`](Self::cas_append) are both safe
///   under concurrent writers: each publishes one complete list by CAS and
///   retries against the current list when another writer won.
#[doc(hidden)]
pub struct AtomicNeighbourList<const N: usize> {
    /// The published list; null is the empty list.
    current: Atomic<List>,
}

impl<const N: usize> AtomicNeighbourList<N> {
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

    /// The ids of the list published when `guard` loaded it. The slice lives
    /// as long as the guard keeps the list from being reclaimed.
    #[inline]
    fn view<'g>(&self, guard: &'g Guard) -> (Shared<'g, List>, &'g [u64]) {
        let shared = self.current.load(Ordering::Acquire, guard);
        // SAFETY: a non-null descriptor points at a list that was fully
        // initialized before its Release CAS made it reachable, and the guard
        // keeps it alive until the guard is dropped: a replaced list is only
        // retired through `defer_destroy`, which waits for every pin taken
        // before the retirement.
        let ids = unsafe { shared.as_ref() }.map_or(&[][..], |list| &list.ids);
        (shared, ids)
    }

    /// Current number of neighbours.
    #[inline]
    pub fn len(&self) -> usize {
        let guard = epoch::pin();
        self.view(&guard).1.len()
    }

    /// Whether the list holds no neighbours.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy the published list into `out`, clearing it first.
    pub fn snapshot_into(&self, out: &mut Vec<u64>) {
        out.clear();
        let guard = epoch::pin();
        out.extend_from_slice(self.view(&guard).1);
    }

    /// Allocating variant of [`snapshot_into`](Self::snapshot_into).
    pub fn snapshot(&self) -> Vec<u64> {
        let mut out = Vec::new();
        self.snapshot_into(&mut out);
        out
    }

    /// Publish `edit` of the current list. `edit` receives the protected
    /// current ids and returns the complete new list, or `None` to leave the
    /// list as it is. A lost CAS calls `edit` again with the list that won.
    /// Returns whether a new list was published.
    fn update(&self, mut edit: impl FnMut(&[u64]) -> Option<Box<[u64]>>) -> bool {
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
                        // are pinned, and `defer_destroy` waits for them.
                        unsafe { guard.defer_destroy(expected) };
                    }
                    return true;
                }
                Err(lost) => {
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

    /// Replace the whole list with `new`, truncated to `N` ids.
    pub fn set(&self, new: &[u64]) {
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
    pub fn cas_append(&self, id: u64) -> bool {
        let mut full = false;
        self.update(|current| {
            full = current.len() >= N;
            (!full).then(|| {
                let mut next = Vec::with_capacity(current.len() + 1);
                next.extend_from_slice(current);
                next.push(id);
                next.into_boxed_slice()
            })
        });
        !full
    }

    /// Replace the entire list contents: [`set`](Self::set).
    #[cfg(test)]
    pub fn replace(&self, new: &[u64]) {
        self.set(new);
    }
}

impl<const N: usize> Drop for AtomicNeighbourList<N> {
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
