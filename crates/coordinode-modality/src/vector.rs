//! Vector modality store: HNSW approximate nearest neighbour index.
//!
//! ## Design choice — direct `HnswIndex` wrap
//!
//! A `LocalVectorStore` instance owns exactly one `HnswIndex` keyed by
//! `(label, property)`. There is no per-store registry indirection — the
//! query layer composes multiple `LocalVectorStore` instances when it
//! needs to dispatch by `(label, property)`.
//!
//! Why one index per store (and not a registry inside the trait):
//!
//! - The HNSW graph is the natural unit of an "index" — one graph
//!   answers KNN for one `(label, property)` pair. Putting a `HashMap`
//!   inside the trait would re-introduce the layer the query-side
//!   registry already provides.
//! - Multi-modality `(label, property)` dispatch is an orchestration
//!   concern, not a storage concern. Storage layer 4 is "typed handle
//!   over one logical index", composition lives one layer up.
//! - The HNSW graph is currently in-memory only: rebuilt from data on
//!   open, never persisted as raw bytes. Wrapping the index here
//!   instead of the engine matches that reality — there is no
//!   partition / endpoint surface to hide.
//!
//! ## Deletion semantics
//!
//! HNSW does not support graph deletion without fragmentation. Vector
//! removal is a no-op at the index level — the query path filters
//! tombstoned results via MVCC visibility. Periodic rebuild when
//! tombstone ratio exceeds threshold is tracked separately.

use std::sync::{Arc, RwLock};

use coordinode_vector::hnsw::{HnswConfig, HnswIndex, SearchResult, VectorLoader};

use crate::error::{StoreError, StoreResult};

/// Typed vector index for one `(label, property)` pair.
///
/// Implementors own a single HNSW graph and expose KNN / insert /
/// bulk-insert operations on it. Composition across `(label, property)`
/// pairs is the caller's responsibility — see crate docs.
pub trait VectorStore: Send + Sync {
    /// Insert or replace a vector for a node.
    fn insert(&self, node_id: u64, vector: Vec<f32>) -> StoreResult<()>;

    /// Mark a node's vector as deleted.
    ///
    /// HNSW graph deletion is unsupported by design; physical removal
    /// fragments the graph. This call is a no-op — callers MUST apply
    /// an MVCC visibility filter to search results to suppress
    /// tombstoned IDs.
    fn remove(&self, node_id: u64) -> StoreResult<()>;

    /// K-nearest-neighbour search. Returns up to `k` results sorted by
    /// distance ascending.
    fn knn_search(&self, query: &[f32], k: usize) -> StoreResult<Vec<SearchResult>>;

    /// KNN search with an optional disk-backed f32 loader for offloaded
    /// indexes. Falls back to in-memory rerank when `loader` is `None`
    /// or the index does not offload vectors.
    fn knn_search_with_loader(
        &self,
        query: &[f32],
        k: usize,
        loader: Option<&dyn VectorLoader>,
    ) -> StoreResult<Vec<SearchResult>>;

    /// Bulk-insert vectors. Returns the number inserted.
    fn bulk_insert(&self, vectors: &mut dyn Iterator<Item = (u64, Vec<f32>)>)
        -> StoreResult<usize>;

    /// Current number of indexed vectors (including tombstoned).
    fn len(&self) -> StoreResult<usize>;

    /// True when no vectors have been inserted.
    fn is_empty(&self) -> StoreResult<bool>;
}

/// CE single-shard `VectorStore` implementation backed by one
/// in-memory `HnswIndex`.
#[derive(Clone)]
pub struct LocalVectorStore {
    inner: Arc<RwLock<HnswIndex>>,
}

impl LocalVectorStore {
    /// Build a new store wrapping a freshly-constructed `HnswIndex`.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_modality::{LocalVectorStore, VectorStore};
    /// use coordinode_vector::hnsw::HnswConfig;
    ///
    /// let store = LocalVectorStore::new(HnswConfig::default());
    /// store.insert(1, vec![1.0, 0.0, 0.0])?;
    /// store.insert(2, vec![0.0, 1.0, 0.0])?;
    /// let knn = store.knn_search(&[1.0, 0.0, 0.0], 1)?;
    /// assert_eq!(knn[0].id, 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: HnswConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HnswIndex::new(config))),
        }
    }

    /// Wrap an existing `HnswIndex` (used when the registry already
    /// holds a handle that must keep its identity).
    pub fn from_index(handle: Arc<RwLock<HnswIndex>>) -> Self {
        Self { inner: handle }
    }

    /// Borrow the underlying handle. Escape hatch for the registry
    /// composition layer in `coordinode-query`.
    pub fn handle(&self) -> Arc<RwLock<HnswIndex>> {
        Arc::clone(&self.inner)
    }

    fn write(&self) -> StoreResult<std::sync::RwLockWriteGuard<'_, HnswIndex>> {
        self.inner
            .write()
            .map_err(|_| StoreError::Invariant("HNSW write lock poisoned".to_owned()))
    }

    fn read(&self) -> StoreResult<std::sync::RwLockReadGuard<'_, HnswIndex>> {
        self.inner
            .read()
            .map_err(|_| StoreError::Invariant("HNSW read lock poisoned".to_owned()))
    }
}

impl VectorStore for LocalVectorStore {
    fn insert(&self, node_id: u64, vector: Vec<f32>) -> StoreResult<()> {
        self.write()?.insert(node_id, vector);
        Ok(())
    }

    fn remove(&self, _node_id: u64) -> StoreResult<()> {
        // Intentional no-op. See trait doc — MVCC visibility filter
        // handles tombstoning at the query layer.
        Ok(())
    }

    fn knn_search(&self, query: &[f32], k: usize) -> StoreResult<Vec<SearchResult>> {
        Ok(self.read()?.search(query, k))
    }

    fn knn_search_with_loader(
        &self,
        query: &[f32],
        k: usize,
        loader: Option<&dyn VectorLoader>,
    ) -> StoreResult<Vec<SearchResult>> {
        let guard = self.read()?;
        let results = match loader {
            Some(loader) => guard.search_with_loader(query, k, loader),
            None => guard.search(query, k),
        };
        Ok(results)
    }

    fn bulk_insert(
        &self,
        vectors: &mut dyn Iterator<Item = (u64, Vec<f32>)>,
    ) -> StoreResult<usize> {
        let mut guard = self.write()?;
        let mut count = 0;
        for (id, vec) in vectors {
            guard.insert(id, vec);
            count += 1;
        }
        Ok(count)
    }

    fn len(&self) -> StoreResult<usize> {
        Ok(self.read()?.len())
    }

    fn is_empty(&self) -> StoreResult<bool> {
        Ok(self.read()?.is_empty())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use coordinode_core::graph::types::VectorMetric;

    fn mk_config(dim: u32) -> HnswConfig {
        HnswConfig {
            m: 8,
            m_max0: 16,
            ef_construction: 50,
            ef_search: 20,
            metric: VectorMetric::L2,
            max_dimensions: dim,
            ..HnswConfig::default()
        }
    }

    #[test]
    fn insert_and_search() {
        let store = LocalVectorStore::new(mk_config(3));
        store.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
        store.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
        store.insert(3, vec![0.0, 0.0, 1.0]).unwrap();

        let results = store.knn_search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1, "closest should be node 1");
    }

    #[test]
    fn empty_store_search_returns_empty() {
        let store = LocalVectorStore::new(mk_config(3));
        assert!(store.is_empty().unwrap());
        let results = store.knn_search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn bulk_insert_counts() {
        let store = LocalVectorStore::new(mk_config(2));
        let mut iter = (0..10u64).map(|i| (i, vec![i as f32, (10 - i) as f32]));
        let n = store.bulk_insert(&mut iter).unwrap();
        assert_eq!(n, 10);
        assert_eq!(store.len().unwrap(), 10);
    }

    #[test]
    fn remove_is_noop_for_index() {
        let store = LocalVectorStore::new(mk_config(2));
        store.insert(42, vec![1.0, 2.0]).unwrap();
        store.remove(42).unwrap();
        // HNSW retains the vector — query layer filters via MVCC.
        assert_eq!(store.len().unwrap(), 1);
    }

    #[test]
    fn handle_clone_shares_index() {
        let a = LocalVectorStore::new(mk_config(2));
        let b = LocalVectorStore::from_index(a.handle());
        a.insert(7, vec![3.0, 4.0]).unwrap();
        assert_eq!(b.len().unwrap(), 1);
    }

    #[test]
    fn insert_same_id_twice_dedups_by_id() {
        // Verified behaviour: HnswIndex maintains an id→idx map and
        // overwrites the slot when the same id is re-inserted. The
        // second vector replaces the first. `len()` stays at 1.
        //
        // (Initial intuition was the opposite — HNSW graphs without
        // explicit dedup would grow. The id_to_idx HashMap in
        // coordinode-vector guards against that. This test pins the
        // observed behaviour so a future implementation change does
        // not silently revert it.)
        let store = LocalVectorStore::new(mk_config(2));
        store.insert(1, vec![0.0, 0.0]).unwrap();
        store.insert(1, vec![5.0, 5.0]).unwrap();
        assert_eq!(store.len().unwrap(), 1, "same id must dedup");
        // KNN from (5, 5) returns the updated vector at distance 0.
        let res = store.knn_search(&[5.0, 5.0], 1).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, 1);
        assert!(res[0].score.abs() < 1e-6, "score should be ~0 at (5,5)");
    }

    #[test]
    fn knn_returns_full_set_when_k_exceeds_size() {
        // k > number of inserted points must not panic — return what
        // we have, in distance order.
        let store = LocalVectorStore::new(mk_config(2));
        store.insert(1, vec![0.0, 0.0]).unwrap();
        store.insert(2, vec![1.0, 1.0]).unwrap();
        let results = store.knn_search(&[0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn knn_results_sorted_by_distance_ascending() {
        let store = LocalVectorStore::new(mk_config(2));
        // Three points at increasing distance from origin.
        store.insert(1, vec![3.0, 0.0]).unwrap();
        store.insert(2, vec![1.0, 0.0]).unwrap();
        store.insert(3, vec![2.0, 0.0]).unwrap();
        let results = store.knn_search(&[0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![2, 3, 1]);
        // Strictly ascending distance.
        for pair in results.windows(2) {
            assert!(
                pair[0].score <= pair[1].score,
                "score not ascending: {} -> {}",
                pair[0].score,
                pair[1].score,
            );
        }
    }

    #[test]
    fn bulk_insert_empty_iter_returns_zero() {
        let store = LocalVectorStore::new(mk_config(2));
        let mut empty = std::iter::empty::<(u64, Vec<f32>)>();
        let n = store.bulk_insert(&mut empty).unwrap();
        assert_eq!(n, 0);
        assert!(store.is_empty().unwrap());
    }

    #[test]
    fn from_index_wraps_existing_handle() {
        let raw = Arc::new(RwLock::new(HnswIndex::new(mk_config(2))));
        raw.write().unwrap().insert(99, vec![5.0, 6.0]);
        let store = LocalVectorStore::from_index(Arc::clone(&raw));
        assert_eq!(store.len().unwrap(), 1);
        let results = store.knn_search(&[5.0, 6.0], 1).unwrap();
        assert_eq!(results[0].id, 99);
    }
}
