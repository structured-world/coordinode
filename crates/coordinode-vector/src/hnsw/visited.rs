//! Epoch-based visited list for HNSW search — O(1) check/mark, zero-cost reset.
//!
//! Replaces `HashSet<usize>` in search hot path. Instead of allocating and hashing
//! per search, reuses a pre-allocated `Vec<u8>` array with an epoch counter.
//!
//! Reset between searches = increment epoch (O(1)). Only `fill(0)` when epoch
//! wraps around (every 255 searches). Pool recycles lists across searches.
//!
//! # Donor references
//! - hnswlib `visited_list_pool.h:8-31` — epoch counter pattern
//! - Qdrant `lib/segment/src/index/visited_pool.rs:19-84` — Rust adaptation with RAII

use std::sync::Mutex;

/// Reusable visited list with epoch-based reset.
///
/// Each element in `counters` stores the epoch when that node was last visited.
/// A node is "visited" iff `counters[id] == current_epoch`.
/// Reset = `current_epoch += 1` (O(1)). `fill(0)` only on u8 wrap (every 255 resets).
struct VisitedList {
    current_epoch: u8,
    counters: Vec<u8>,
}

impl VisitedList {
    fn new(num_elements: usize) -> Self {
        Self {
            current_epoch: 1,
            counters: vec![0; num_elements],
        }
    }

    /// Advance to next epoch. O(1) unless epoch wraps (then O(n) fill).
    fn next_epoch(&mut self) {
        self.current_epoch = self.current_epoch.wrapping_add(1);
        if self.current_epoch == 0 {
            // Epoch wrapped — reset all counters (happens every 255 searches)
            self.current_epoch = 1;
            self.counters.fill(0);
        }
    }

    /// Ensure capacity for at least `num_elements` nodes.
    fn resize(&mut self, num_elements: usize) {
        if self.counters.len() < num_elements {
            self.counters.resize(num_elements, 0);
        }
    }
}

/// RAII handle for a visited list borrowed from the pool.
/// Automatically returns the list to the pool on Drop.
pub(crate) struct VisitedListHandle<'a> {
    pool: &'a VisitedPool,
    /// Owned visited list. Replaced with empty dummy on Drop (returned to pool).
    list: VisitedList,
    /// Whether the list has been returned to pool (prevents double-return).
    returned: bool,
}

impl<'a> VisitedListHandle<'a> {
    /// Check if a node has been visited in the current epoch.
    #[inline]
    #[cfg(test)]
    pub(crate) fn is_visited(&self, id: usize) -> bool {
        self.list
            .counters
            .get(id)
            .is_some_and(|&v| v == self.list.current_epoch)
    }

    /// Mark a node as visited. Returns `true` if it was ALREADY visited (duplicate).
    #[inline]
    pub(crate) fn check_and_mark(&mut self, id: usize) -> bool {
        // Grow if needed (handles nodes added after pool creation)
        if id >= self.list.counters.len() {
            self.list.counters.resize(id + 1, 0);
        }
        let was_visited = self.list.counters[id] == self.list.current_epoch;
        self.list.counters[id] = self.list.current_epoch;
        was_visited
    }

    /// Unchecked variant: the caller guarantees `id < counters.len()` (by
    /// having passed the `num_elements` known to the pool and bounds-
    /// checking the id against `nodes.len()` before the call).
    ///
    /// Profile of the EndOfSearch path on glove-100-angular M=16 ef=800
    /// (e5e10ab + d67fc2d, perf record on i9-9900K) put the per-visit
    /// inner loop's bounds-check `testq` at ~32% of `search_layer_ctx_
    /// no_rerank` and the visited-byte store at ~10%. The bounds checks
    /// inside [`check_and_mark`] (the `id >= len` resize gate and the
    /// implicit `Vec` index bounds check) are PURE overhead on the hot
    /// path — the pool already resized to `num_elements` in
    /// [`VisitedPool::get`], and `search_layer_ctx_no_rerank` already
    /// gates `neighbor_idx < n_nodes` before calling. Skipping those
    /// two checks here saves ~10 ns × thousands of calls per query.
    ///
    /// # Safety
    ///
    /// `id` must be strictly less than `counters.len()`. Caller's pool
    /// resize + nodes.len() guard cover that on the search hot path.
    #[inline]
    pub(crate) unsafe fn check_and_mark_unchecked(&mut self, id: usize) -> bool {
        debug_assert!(
            id < self.list.counters.len(),
            "check_and_mark_unchecked: id {id} >= counters.len() {}",
            self.list.counters.len(),
        );
        // SAFETY: caller invariant: id < self.list.counters.len().
        unsafe {
            let counter = self.list.counters.get_unchecked_mut(id);
            let was_visited = *counter == self.list.current_epoch;
            *counter = self.list.current_epoch;
            was_visited
        }
    }

    /// Pointer to the visited-counter cell for `id`. Exposed for the HNSW
    /// search hot loop to issue a `_mm_prefetch` (or aarch64 equivalent)
    /// against the next neighbour's epoch byte before reading it — that's
    /// the pattern hnswlib uses to hide the visited-array L2/L3 miss
    /// behind the prior neighbour's distance compute.
    ///
    /// Returns `None` when `id` is past the currently-allocated counter
    /// array (caller should skip prefetch and fall through to the normal
    /// `check_and_mark` path which will grow on demand).
    #[inline]
    pub(crate) fn counter_ptr(&self, id: usize) -> Option<*const u8> {
        self.list.counters.get(id).map(|c| c as *const u8)
    }
}

impl Drop for VisitedListHandle<'_> {
    fn drop(&mut self) {
        if !self.returned {
            self.returned = true;
            let list = std::mem::replace(&mut self.list, VisitedList::new(0));
            self.pool.return_back(list);
        }
    }
}

/// Pool of reusable `VisitedList` instances.
///
/// Avoids allocating a new Vec on every search. Lists are recycled:
/// `get()` pops from pool (or creates new), `Drop` on handle returns it.
///
/// Thread-safety: `Mutex<Vec<VisitedList>>`. In current single-threaded
/// search (R858 adds concurrency later), mutex is uncontended = ~25ns.
pub(crate) struct VisitedPool {
    pool: Mutex<Vec<VisitedList>>,
    /// Maximum number of lists to keep in pool (prevent memory leak).
    max_pool_size: usize,
}

impl VisitedPool {
    /// Create a new pool.
    pub(crate) fn new() -> Self {
        Self {
            pool: Mutex::new(Vec::with_capacity(4)),
            max_pool_size: 16,
        }
    }

    /// Get a visited list handle for `num_elements` nodes.
    ///
    /// Pops from pool if available, otherwise creates new.
    /// The returned handle advances the epoch automatically.
    pub(crate) fn get(&self, num_elements: usize) -> VisitedListHandle<'_> {
        let mut list = self
            .pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop()
            .unwrap_or_else(|| VisitedList::new(num_elements));

        list.resize(num_elements);
        list.next_epoch();

        VisitedListHandle {
            pool: self,
            list,
            returned: false,
        }
    }

    fn return_back(&self, list: VisitedList) {
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if pool.len() < self.max_pool_size {
            pool.push(list);
        }
        // else: drop the list (pool full)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_check_and_mark() {
        let pool = VisitedPool::new();
        let mut handle = pool.get(100);

        assert!(!handle.is_visited(0));
        assert!(!handle.is_visited(50));
        assert!(!handle.is_visited(99));

        // First mark returns false (was not visited)
        assert!(!handle.check_and_mark(50));
        assert!(handle.is_visited(50));

        // Second mark returns true (already visited)
        assert!(handle.check_and_mark(50));

        // Other nodes still not visited
        assert!(!handle.is_visited(0));
        assert!(!handle.is_visited(99));
    }

    #[test]
    fn test_epoch_reset_on_reuse() {
        let pool = VisitedPool::new();

        // First search: mark node 42
        {
            let mut handle = pool.get(100);
            handle.check_and_mark(42);
            assert!(handle.is_visited(42));
        } // handle dropped, list returned to pool

        // Second search: same list reused, node 42 should NOT be visited
        {
            let handle = pool.get(100);
            assert!(!handle.is_visited(42));
        }
    }

    #[test]
    fn test_epoch_wrap_around() {
        let pool = VisitedPool::new();

        // Force 255 epoch advances (u8 wraps at 256)
        for i in 0..256 {
            let mut handle = pool.get(10);
            handle.check_and_mark(0);
            assert!(
                handle.is_visited(0),
                "node 0 should be visited at iteration {i}"
            );
        }

        // After 255 reuses, epoch wrapped, fill(0) was called — still works
        let handle = pool.get(10);
        assert!(!handle.is_visited(0));
    }

    #[test]
    fn test_dynamic_resize() {
        let pool = VisitedPool::new();
        let mut handle = pool.get(10);

        // Mark node beyond initial capacity
        assert!(!handle.check_and_mark(100));
        assert!(handle.is_visited(100));
    }

    #[test]
    fn test_pool_recycles() {
        let pool = VisitedPool::new();

        // Create and drop a handle — list returned to pool
        {
            let mut handle = pool.get(1000);
            handle.check_and_mark(500);
        }

        // Pool should have 1 list (returned from the handle above)
        let pool_size = pool.pool.lock().unwrap().len();
        assert_eq!(pool_size, 1);

        // Getting from pool reuses existing list (pops from pool)
        let handle = pool.get(1000);
        let pool_size = pool.pool.lock().unwrap().len();
        assert_eq!(pool_size, 0); // one was taken

        // Dropping returns it back
        drop(handle);
        let pool_size = pool.pool.lock().unwrap().len();
        assert_eq!(pool_size, 1); // returned

        // Create 3 concurrent handles — forces 3 allocations
        let h1 = pool.get(1000); // takes from pool (size 0)
        let h2 = pool.get(1000); // pool empty → new allocation
        let h3 = pool.get(1000); // pool empty → new allocation

        let pool_size = pool.pool.lock().unwrap().len();
        assert_eq!(pool_size, 0); // all taken

        drop(h1);
        drop(h2);
        drop(h3);

        let pool_size = pool.pool.lock().unwrap().len();
        assert_eq!(pool_size, 3); // all 3 returned
    }

    #[test]
    fn test_pool_max_size() {
        let pool = VisitedPool::new(); // max_pool_size = 16

        // Create 20 handles, drop them all
        let handles: Vec<_> = (0..20).map(|_| pool.get(10)).collect();
        drop(handles);

        // Pool capped at max_pool_size
        let pool_size = pool.pool.lock().unwrap().len();
        assert!(pool_size <= pool.max_pool_size);
    }
}
