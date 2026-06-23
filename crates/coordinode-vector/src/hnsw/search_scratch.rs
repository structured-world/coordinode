//! Reusable scratch buffers for the layer-0 HNSW search.
//!
//! Each query call to `search_layer_ctx_no_rerank` or
//! `search_layer_ctx_inline_rerank` historically allocated four `Vec` /
//! `BinaryHeap` instances (`candidates`, `results`, `connections`,
//! `unvisited_neighbors`) on entry and dropped them on return. Under the
//! 4-thread ann-benchmarks sweep that is 4 mallocs and 4 frees per query
//! across N queries times 4 workers; the system allocator becomes the
//! choke point under contention.
//!
//! `SearchScratch` bundles those four buffers as plain `Vec`s. The owner
//! reuses the same buffers across queries; the per-call `BinaryHeap` is
//! reconstructed by moving the storage out via `mem::take`, wrapping it
//! in `BinaryHeap::from`, and putting the result back with `.into_vec()`
//! on the way out. Heap order is not preserved across reuse - the next
//! call clears each buffer before use.
//!
//! `SearchScratchPool` is the lock-protected free list, modelled on
//! `VisitedPool`. The current search code path holds the scratch for
//! one `search_layer_ctx` call (one mutex acquire + one release per
//! query). Contention on the pool mutex is therefore O(1) per query and
//! dominated by the saved allocator round-trip cost.

#![allow(
    dead_code,
    reason = "consumer of these accessors lands in the search-side wiring chunk on the same plan; tests already exercise the pool round-trip"
)]

use std::sync::Mutex;

use super::{Candidate, FarCandidate, M_MAX0};

/// Reusable scratch buffers for one layer-0 search. The owner is
/// responsible for clearing each buffer before use; the pool does not
/// touch the contents on return.
#[derive(Default)]
pub(super) struct SearchScratch {
    /// Storage that backs the `candidates` min-heap. Heap-ordered while
    /// the heap is borrowing it; arbitrary order at rest.
    pub(super) candidates_storage: Vec<Candidate>,
    /// Storage that backs the `results` max-heap. Same lifecycle.
    pub(super) results_storage: Vec<FarCandidate>,
    /// Neighbour id snapshot buffer reused across iterations of the
    /// inner loop.
    pub(super) connections: Vec<u64>,
    /// Index buffer for not-yet-visited neighbours in the current
    /// expansion step.
    pub(super) unvisited: Vec<usize>,
}

impl SearchScratch {
    /// Build a scratch sized for a search with `ef` candidates. The
    /// `ef + 16` slack matches the existing `heap_cap` constant in the
    /// search functions.
    pub(super) fn with_ef(ef: usize) -> Self {
        let cap = ef.saturating_add(16);
        Self {
            candidates_storage: Vec::with_capacity(cap),
            results_storage: Vec::with_capacity(cap),
            connections: Vec::with_capacity(M_MAX0),
            unvisited: Vec::with_capacity(M_MAX0),
        }
    }

    /// Clear every buffer in place without releasing capacity. Called
    /// by the consumer at the start of each search call.
    pub(super) fn clear(&mut self) {
        self.candidates_storage.clear();
        self.results_storage.clear();
        self.connections.clear();
        self.unvisited.clear();
    }
}

/// Free list of `SearchScratch` instances. Same shape as `VisitedPool`:
/// pop on `get`, push on `Drop`. Cap on retained entries keeps the pool
/// from holding memory after a large `ef` burst when subsequent queries
/// run at a smaller `ef`.
pub(super) struct SearchScratchPool {
    pool: Mutex<Vec<SearchScratch>>,
    max_pool_size: usize,
}

impl SearchScratchPool {
    pub(super) fn new() -> Self {
        Self {
            pool: Mutex::new(Vec::with_capacity(4)),
            max_pool_size: 16,
        }
    }

    /// Acquire a scratch sized for `ef`. Pops the most recent entry
    /// from the pool when available, otherwise creates a fresh one.
    /// The returned handle returns the scratch to the pool on drop.
    pub(super) fn get(&self, ef: usize) -> SearchScratchHandle<'_> {
        let mut scratch = self
            .pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop()
            .unwrap_or_else(|| SearchScratch::with_ef(ef));
        scratch.clear();
        SearchScratchHandle {
            pool: self,
            scratch,
            returned: false,
        }
    }

    fn return_back(&self, scratch: SearchScratch) {
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if pool.len() < self.max_pool_size {
            pool.push(scratch);
        }
    }
}

impl Default for SearchScratchPool {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII handle for a scratch borrowed from a `SearchScratchPool`. The
/// owner mutates the inner buffers freely; on drop the buffers go back
/// to the pool for the next query to reuse.
pub(super) struct SearchScratchHandle<'a> {
    pool: &'a SearchScratchPool,
    scratch: SearchScratch,
    returned: bool,
}

impl SearchScratchHandle<'_> {
    /// Mutable view into the scratch buffers.
    pub(super) fn scratch(&mut self) -> &mut SearchScratch {
        &mut self.scratch
    }
}

impl Drop for SearchScratchHandle<'_> {
    fn drop(&mut self) {
        if !self.returned {
            self.returned = true;
            let scratch = std::mem::take(&mut self.scratch);
            self.pool.return_back(scratch);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
