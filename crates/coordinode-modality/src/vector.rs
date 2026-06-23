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

use coordinode_vector::hnsw::{HnswConfig, HnswIndex, SearchMode, SearchResult, VectorLoader};

use crate::error::{StoreError, StoreResult};

/// Typed vector index for one `(label, property)` pair.
///
/// Implementors own a single HNSW graph and expose KNN / insert /
/// bulk-insert operations on it. Composition across `(label, property)`
/// pairs is the caller's responsibility — see crate docs.
pub trait VectorStore: Send + Sync {
    /// Insert or replace a vector for a node. `HnswIndex` dedups by
    /// node id — re-inserting the same id replaces the vector slot.
    ///
    /// # Examples
    ///
    /// ```
    /// # use coordinode_modality::{LocalVectorStore, VectorStore};
    /// # use coordinode_vector::hnsw::HnswConfig;
    /// let store = LocalVectorStore::new(HnswConfig::default());
    /// store.insert(1, vec![1.0, 0.0, 0.0])?;
    /// store.insert(2, vec![0.0, 1.0, 0.0])?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn insert(&self, node_id: u64, vector: Vec<f32>) -> StoreResult<()>;

    /// Mark a node's vector as deleted.
    ///
    /// HNSW graph deletion is unsupported by design; physical removal
    /// fragments the graph. This call is a no-op — callers MUST apply
    /// an MVCC visibility filter to search results to suppress
    /// tombstoned IDs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use coordinode_modality::{LocalVectorStore, VectorStore};
    /// # use coordinode_vector::hnsw::HnswConfig;
    /// let store = LocalVectorStore::new(HnswConfig::default());
    /// store.remove(42)?; // no-op at the index level
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn remove(&self, node_id: u64) -> StoreResult<()>;

    /// K-nearest-neighbour search. Returns up to `k` results sorted by
    /// distance ascending.
    ///
    /// # Examples
    ///
    /// ```
    /// # use coordinode_modality::{LocalVectorStore, VectorStore};
    /// # use coordinode_vector::hnsw::HnswConfig;
    /// let store = LocalVectorStore::new(HnswConfig::default());
    /// store.insert(1, vec![1.0, 0.0, 0.0])?;
    /// store.insert(2, vec![0.0, 1.0, 0.0])?;
    /// let hits = store.knn_search(&[1.0, 0.0, 0.0], 1)?;
    /// assert_eq!(hits[0].id, 1);
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn knn_search(&self, query: &[f32], k: usize) -> StoreResult<Vec<SearchResult>>;

    /// KNN search with an optional disk-backed f32 loader for offloaded
    /// indexes. Falls back to in-memory rerank when `loader` is `None`
    /// or the index does not offload vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use coordinode_modality::{LocalVectorStore, VectorStore};
    /// # use coordinode_vector::hnsw::HnswConfig;
    /// let store = LocalVectorStore::new(HnswConfig::default());
    /// store.insert(1, vec![1.0, 0.0, 0.0])?;
    /// // No loader = same as knn_search.
    /// let hits = store.knn_search_with_loader(&[1.0, 0.0, 0.0], 1, None)?;
    /// assert_eq!(hits.len(), 1);
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn knn_search_with_loader(
        &self,
        query: &[f32],
        k: usize,
        loader: Option<&dyn VectorLoader>,
    ) -> StoreResult<Vec<SearchResult>>;

    /// KNN search with an explicit strategy. `SearchMode::Hnsw` follows
    /// the graph using the index's configured `ef_search` (matches
    /// [`Self::knn_search`]). `SearchMode::Exact` runs a brute-force linear
    /// scan over every stored vector and returns recall=1.0 top-k. The
    /// store owns one set of vectors; both modes read the same data.
    ///
    /// # Examples
    ///
    /// ```
    /// # use coordinode_modality::{LocalVectorStore, VectorStore};
    /// # use coordinode_vector::hnsw::{HnswConfig, SearchMode};
    /// let store = LocalVectorStore::new(HnswConfig::default());
    /// store.insert(1, vec![1.0, 0.0, 0.0])?;
    /// store.insert(2, vec![0.0, 1.0, 0.0])?;
    /// let hits = store.knn_search_with_mode(&[1.0, 0.0, 0.0], 2, SearchMode::Exact)?;
    /// assert_eq!(hits[0].id, 1);
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn knn_search_with_mode(
        &self,
        query: &[f32],
        k: usize,
        mode: SearchMode,
    ) -> StoreResult<Vec<SearchResult>>;

    /// Bulk-insert vectors. Returns the number inserted.
    ///
    /// # Examples
    ///
    /// ```
    /// # use coordinode_modality::{LocalVectorStore, VectorStore};
    /// # use coordinode_vector::hnsw::HnswConfig;
    /// let store = LocalVectorStore::new(HnswConfig::default());
    /// let mut iter = (0..100u64).map(|i| (i, vec![i as f32, 0.0, 0.0]));
    /// let n = store.bulk_insert(&mut iter)?;
    /// assert_eq!(n, 100);
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn bulk_insert(&self, vectors: &mut dyn Iterator<Item = (u64, Vec<f32>)>)
        -> StoreResult<usize>;

    /// Current number of indexed vectors (including tombstoned).
    ///
    /// # Examples
    ///
    /// ```
    /// # use coordinode_modality::{LocalVectorStore, VectorStore};
    /// # use coordinode_vector::hnsw::HnswConfig;
    /// let store = LocalVectorStore::new(HnswConfig::default());
    /// assert_eq!(store.len()?, 0);
    /// store.insert(1, vec![1.0, 0.0, 0.0])?;
    /// assert_eq!(store.len()?, 1);
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    fn len(&self) -> StoreResult<usize>;

    /// True when no vectors have been inserted.
    ///
    /// # Examples
    ///
    /// ```
    /// # use coordinode_modality::{LocalVectorStore, VectorStore};
    /// # use coordinode_vector::hnsw::HnswConfig;
    /// let store = LocalVectorStore::new(HnswConfig::default());
    /// assert!(store.is_empty()?);
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
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

    fn knn_search_with_mode(
        &self,
        query: &[f32],
        k: usize,
        mode: SearchMode,
    ) -> StoreResult<Vec<SearchResult>> {
        Ok(self.read()?.search_with_mode(query, k, mode))
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
mod tests;
