//! Vector index registry: tracks active HNSW indexes for vector search acceleration.
//!
//! Holds in-memory HNSW index instances keyed by (label, property).
//! Indexes are populated from stored vectors on Database::open() and
//! maintained incrementally on node create/update/delete.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use coordinode_core::graph::node::NodeId;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_vector::hnsw::{HnswConfig, HnswIndex, SearchResult};
use coordinode_vector::storage::lsm_backed::LsmVectorTier;
use coordinode_vector::storage::{VectorTierHandle, VectorTierStorage};
use coordinode_vector::VectorLoader;

use super::definition::IndexDefinition;

/// Key for vector index lookup: (label, property).
type VectorIndexKey = (String, String);

/// Thread-safe handle to an HNSW index. Readers can search concurrently;
/// writers (insert/delete) acquire exclusive access.
pub type HnswHandle = Arc<RwLock<HnswIndex>>;

/// Registry of active HNSW vector indexes.
///
/// Uses interior mutability (`RwLock`) so `register` / `unregister` can be
/// called via `&self`. This allows Cypher DDL (`CREATE VECTOR INDEX`) executed
/// from within the runner to update the live registry without requiring
/// a `&mut` reference through the `ExecutionContext`.
///
/// Holds live HNSW graph structures in memory. Each index corresponds
/// to an `IndexDefinition` with `index_type == Hnsw` persisted in the
/// schema partition. The HNSW graph itself is rebuilt from stored vectors
/// on startup and maintained incrementally during writes.
pub struct VectorIndexRegistry {
    /// Active HNSW indexes: (label, property) → live HNSW graph.
    indexes: RwLock<HashMap<VectorIndexKey, HnswHandle>>,
    /// Index definitions keyed by (label, property) for metadata lookup.
    definitions: RwLock<HashMap<VectorIndexKey, IndexDefinition>>,
    /// Optional persistent f32 truth tier backend (ADR-033). When set,
    /// callers obtain a `VectorTierHandle` via [`Self::tier_handle`]
    /// using pre-resolved `(label_id, property_id)` and pass it to
    /// [`Self::register_with_tier`]. The registry intentionally does
    /// NOT carry an interner reference — interning lives at the
    /// caller (executor, Database) so we never re-enter the same
    /// `parking_lot` RwLock from inside an active write transaction.
    tier_backend: Option<Arc<dyn VectorTierStorage>>,
}

impl VectorIndexRegistry {
    /// Create an empty registry with no persistent vector tier.
    pub fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
            definitions: RwLock::new(HashMap::new()),
            tier_backend: None,
        }
    }

    /// Create a registry that persists vector data to the
    /// `Partition::VectorF32` truth tier on every insert.
    ///
    /// The `engine` is shared with the rest of the embed / query
    /// stack so registered HNSW indexes route their truth-tier
    /// writes to the same LSM. Label / property interning is the
    /// caller's job — pass resolved ids through [`Self::tier_handle`]
    /// at register time so the insert path never touches a shared
    /// interner lock.
    pub fn with_vector_tier(engine: Arc<StorageEngine>) -> Self {
        let backend: Arc<dyn VectorTierStorage> = Arc::new(LsmVectorTier::new(engine));
        Self {
            indexes: RwLock::new(HashMap::new()),
            definitions: RwLock::new(HashMap::new()),
            tier_backend: Some(backend),
        }
    }

    /// Build a [`VectorTierHandle`] from pre-resolved
    /// `(label_id, property_id)`. Returns `None` if no tier backend
    /// is configured. The IDs MUST be obtained by the caller from
    /// the same interner the storage layer uses for node properties
    /// — the registry holds no interner reference of its own.
    pub fn tier_handle(&self, label_id: u32, property_id: u32) -> Option<VectorTierHandle> {
        let backend = self.tier_backend.as_ref()?.clone();
        Some(VectorTierHandle::new(backend, label_id, property_id))
    }

    /// Register a new vector index with an empty HNSW graph and NO
    /// persistent tier wiring. Equivalent to
    /// `register_with_tier(def, None)`. The graph is in-RAM only —
    /// f32 originals are not persisted, suitable for tests and
    /// ad-hoc analytics.
    pub fn register(&self, def: IndexDefinition) {
        self.register_with_tier(def, None);
    }

    /// Register a new vector index with an empty HNSW graph and a
    /// caller-provided tier handle. When `tier` is `Some`, every
    /// insert into the resulting HNSW also writes the f32 original
    /// to the LSM truth tier under
    /// `vec:<label_id><property_id><node_id>` (ADR-033).
    ///
    /// The caller is responsible for populating the index with
    /// existing vectors (see `bulk_insert`).
    ///
    /// Uses interior mutability — safe to call via `&self`.
    pub fn register_with_tier(&self, def: IndexDefinition, tier: Option<VectorTierHandle>) {
        let Some(config) = def.vector_config.as_ref() else {
            tracing::error!(
                "register called with non-vector IndexDefinition: {}",
                def.name
            );
            return;
        };

        let hnsw_config = HnswConfig {
            m: config.m,
            m_max0: config.m * 2,
            ef_construction: config.ef_construction,
            ef_search: 200,
            metric: config.metric,
            max_dimensions: config.dimensions,
            quantization: config.quantization,
            rerank_candidates: 100,
            calibration_threshold: 1000,
            offload_vectors: config.offload_vectors,
            property_name: def.property().to_string(),
            max_elements: 1_000_000,
            ..HnswConfig::default()
        };

        let mut hnsw = HnswIndex::new(hnsw_config);
        // Bind the caller-resolved tier handle BEFORE the index is
        // moved into the registry so subsequent inserts persist f32
        // to disk per ADR-033. `None` keeps the index pure in-RAM
        // (test path, ad-hoc analytics).
        hnsw.set_vector_tier(tier);

        let key = (def.label.clone(), def.property().to_string());
        self.indexes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), Arc::new(RwLock::new(hnsw)));
        self.definitions
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, def);
    }

    /// Register an index with a pre-built HNSW graph (e.g. after rebuild).
    ///
    /// Uses interior mutability — safe to call via `&self`.
    pub fn register_with_index(&self, def: IndexDefinition, hnsw: HnswIndex) {
        let key = (def.label.clone(), def.property().to_string());
        self.indexes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), Arc::new(RwLock::new(hnsw)));
        self.definitions
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, def);
    }

    /// Unregister a vector index.
    ///
    /// Uses interior mutability — safe to call via `&self`.
    pub fn unregister(&self, label: &str, property: &str) {
        let key = (label.to_string(), property.to_string());
        self.indexes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&key);
        self.definitions
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&key);
    }

    /// Get a handle to the HNSW index for a (label, property) pair.
    pub fn get(&self, label: &str, property: &str) -> Option<HnswHandle> {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(label.to_string(), property.to_string()))
            .cloned()
    }

    /// Update the in-memory state of a registered index without touching
    /// the persisted schema record. Called by the background backfill
    /// thread after `save_index_state` so reader-side state checks can
    /// answer without going to disk. Returns `true` when an index was
    /// updated, `false` if no entry exists for the given (label, property).
    pub fn set_state(&self, label: &str, property: &str, state: crate::index::IndexState) -> bool {
        let mut defs = self.definitions.write().unwrap_or_else(|e| e.into_inner());
        if let Some(def) = defs.get_mut(&(label.to_string(), property.to_string())) {
            def.state = state;
            true
        } else {
            false
        }
    }

    /// Quick read-side lookup: returns the live state and policy together
    /// with the HNSW handle so a caller can decide whether to proceed,
    /// wait, or error out without a separate `get_definition` call.
    pub fn policy_state(
        &self,
        label: &str,
        property: &str,
    ) -> Option<(
        crate::index::IndexState,
        crate::index::OnlineDuringBuild,
        HnswHandle,
    )> {
        let defs = self.definitions.read().unwrap_or_else(|e| e.into_inner());
        let key = (label.to_string(), property.to_string());
        let def = defs.get(&key)?;
        let state = def.state.clone();
        let policy = def.online_during_build;
        drop(defs);
        let handle = self.get(label, property)?;
        Some((state, policy, handle))
    }

    /// Get the index definition for a (label, property) pair.
    pub fn get_definition(&self, label: &str, property: &str) -> Option<IndexDefinition> {
        self.definitions
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(label.to_string(), property.to_string()))
            .cloned()
    }

    /// Get the index definition by index name.
    ///
    /// Used by the executor to resolve a planner-annotated index name (e.g. `"item_emb"`)
    /// back to its (label, property) key for `search_with_loader`.
    pub fn get_definition_by_name(&self, name: &str) -> Option<IndexDefinition> {
        self.definitions
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .find(|def| def.name == name)
            .cloned()
    }

    /// List all registered vector indexes for a given label.
    pub fn indexes_for_label(&self, label: &str) -> Vec<(String, HnswHandle)> {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|((l, _), _)| l == label)
            .map(|((_, p), h)| (p.clone(), h.clone()))
            .collect()
    }

    /// Check if a vector index exists for a (label, property).
    pub fn has_index(&self, label: &str, property: &str) -> bool {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&(label.to_string(), property.to_string()))
    }

    /// Number of registered vector indexes.
    pub fn len(&self) -> usize {
        self.indexes.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Iterate over all registered index definitions, calling `f` for each.
    ///
    /// Used during startup to rebuild HNSW graphs from stored vectors.
    pub fn for_each_definition(&self, mut f: impl FnMut(&IndexDefinition)) {
        let defs = self.definitions.read().unwrap_or_else(|e| e.into_inner());
        for def in defs.values() {
            f(def);
        }
    }

    /// Collect all index definitions into a `Vec`.
    pub fn all_definitions(&self) -> Vec<IndexDefinition> {
        self.definitions
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .cloned()
            .collect()
    }

    /// Insert a vector into all applicable HNSW indexes for a node.
    ///
    /// Called on node creation or vector property update. For bulk
    /// loads use [`Self::on_vectors_written`] to amortise the HNSW
    /// write-lock across the whole batch instead of paying it per
    /// inserted vector.
    pub fn on_vector_written(&self, label: &str, node_id: NodeId, property: &str, vector: &[f32]) {
        if let Some(handle) = self.get(label, property) {
            if let Ok(mut hnsw) = handle.write() {
                hnsw.insert(node_id.as_raw(), vector.to_vec());
            }
        }
    }

    /// Batched variant: insert N vectors into the same (label,
    /// property) HNSW index under a single write-lock acquisition,
    /// then dispatch to [`HnswIndex::insert_batch`] which seeds
    /// sequentially up to `SEED_DENSITY` and parallelises the rest
    /// via rayon.
    ///
    /// Caller is responsible for grouping by (label, property) — a
    /// mixed batch would otherwise need one write-lock per group,
    /// negating the win for the cross-group case.
    ///
    /// `items` is consumed: each `(NodeId, Vec<f32>)` is forwarded
    /// to the HNSW insert path without further copies.
    pub fn on_vectors_written(&self, label: &str, property: &str, items: Vec<(NodeId, Vec<f32>)>) {
        if items.is_empty() {
            return;
        }
        if let Some(handle) = self.get(label, property) {
            if let Ok(mut hnsw) = handle.write() {
                let batch: Vec<(u64, Vec<f32>)> =
                    items.into_iter().map(|(id, v)| (id.as_raw(), v)).collect();
                hnsw.insert_batch(batch);
            }
        }
    }

    /// Remove a vector from all applicable HNSW indexes for a node.
    ///
    /// Called on node deletion or vector property removal.
    /// HNSW does not support true deletion — we rely on the MVCC
    /// visibility filter to exclude deleted nodes from search results.
    /// The vector remains in the graph to avoid fragmentation.
    /// Periodic rebuild (>50% tombstones) is tracked separately.
    pub fn on_vector_deleted(&self, _label: &str, _node_id: NodeId, _property: &str) {
        // HNSW graph deletion is handled via MVCC post-filter visibility.
        // Physical removal would fragment the graph. Tracked as future
        // optimization: rebuild when tombstone ratio exceeds threshold.
    }

    /// Search the HNSW index for a (label, property) pair.
    ///
    /// Returns `None` if no index exists, or the search results if found.
    pub fn search(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        k: usize,
    ) -> Option<Vec<SearchResult>> {
        let handle = self.get(label, property)?;
        let hnsw = handle.read().ok()?;
        Some(hnsw.search(query, k))
    }

    /// Search with optional VectorLoader for disk-backed f32 reranking.
    ///
    /// When the HNSW index has offloaded f32 vectors, the loader provides
    /// them on-demand for exact reranking. Falls back to in-memory search
    /// when no loader is provided or vectors are not offloaded.
    pub fn search_with_loader(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        k: usize,
        loader: Option<&dyn VectorLoader>,
    ) -> Option<Vec<SearchResult>> {
        let handle = self.get(label, property)?;
        let hnsw = handle.read().ok()?;
        if let Some(loader) = loader {
            Some(hnsw.search_with_loader(query, k, loader))
        } else {
            Some(hnsw.search(query, k))
        }
    }

    /// Bulk-insert multiple vectors into a specific HNSW index.
    ///
    /// Used during index rebuild (on Database::open or CREATE VECTOR INDEX).
    /// The caller is responsible for scanning stored nodes and extracting
    /// vectors — this method just does the HNSW insertions.
    pub fn bulk_insert(
        &self,
        label: &str,
        property: &str,
        vectors: impl Iterator<Item = (u64, Vec<f32>)>,
    ) -> usize {
        let handle = match self.get(label, property) {
            Some(h) => h,
            None => return 0,
        };

        let mut count = 0;
        if let Ok(mut hnsw) = handle.write() {
            for (id, vec) in vectors {
                hnsw.insert(id, vec);
                count += 1;
            }
        }
        count
    }
}

impl Default for VectorIndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::index::VectorIndexConfig;
    use coordinode_core::graph::types::VectorMetric;

    fn test_config() -> VectorIndexConfig {
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::Cosine,
            m: 16,
            ef_construction: 200,
            quantization: coordinode_vector::hnsw::QuantizationCodec::None,
            offload_vectors: false,
        }
    }

    #[test]
    fn register_and_lookup() {
        let reg = VectorIndexRegistry::new();
        let def = IndexDefinition::hnsw("movie_embedding", "Movie", "embedding", test_config());

        reg.register(def);

        assert!(reg.has_index("Movie", "embedding"));
        assert!(!reg.has_index("Movie", "title"));
        assert!(!reg.has_index("User", "embedding"));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn search_empty_index_returns_empty() {
        let reg = VectorIndexRegistry::new();
        let def = IndexDefinition::hnsw("movie_embedding", "Movie", "embedding", test_config());
        reg.register(def);

        let results = reg.search("Movie", "embedding", &[1.0, 0.0, 0.0], 10);
        assert!(results.is_some());
        assert!(results.unwrap().is_empty());
    }

    #[test]
    fn insert_and_search() {
        let reg = VectorIndexRegistry::new();
        let def = IndexDefinition::hnsw("movie_embedding", "Movie", "embedding", test_config());
        reg.register(def);

        // Insert vectors
        reg.on_vector_written("Movie", NodeId::from_raw(1), "embedding", &[1.0, 0.0, 0.0]);
        reg.on_vector_written("Movie", NodeId::from_raw(2), "embedding", &[0.0, 1.0, 0.0]);
        reg.on_vector_written("Movie", NodeId::from_raw(3), "embedding", &[0.9, 0.1, 0.0]);

        // Search for vector closest to [1.0, 0.0, 0.0]
        let results = reg
            .search("Movie", "embedding", &[1.0, 0.0, 0.0], 2)
            .expect("search should return results");

        assert_eq!(results.len(), 2);
        // Node 1 should be closest (exact match), then node 3
        assert_eq!(results[0].id, 1);
        assert_eq!(results[1].id, 3);
    }

    #[test]
    fn indexes_for_label() {
        let reg = VectorIndexRegistry::new();
        reg.register(IndexDefinition::hnsw(
            "movie_embed",
            "Movie",
            "embedding",
            test_config(),
        ));
        reg.register(IndexDefinition::hnsw(
            "movie_thumb",
            "Movie",
            "thumbnail_vec",
            test_config(),
        ));
        reg.register(IndexDefinition::hnsw(
            "user_embed",
            "User",
            "embedding",
            test_config(),
        ));

        let movie_idxs = reg.indexes_for_label("Movie");
        assert_eq!(movie_idxs.len(), 2);

        let user_idxs = reg.indexes_for_label("User");
        assert_eq!(user_idxs.len(), 1);
    }

    #[test]
    fn unregister() {
        let reg = VectorIndexRegistry::new();
        reg.register(IndexDefinition::hnsw(
            "movie_embed",
            "Movie",
            "embedding",
            test_config(),
        ));
        assert!(reg.has_index("Movie", "embedding"));

        reg.unregister("Movie", "embedding");
        assert!(!reg.has_index("Movie", "embedding"));
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn no_index_search_returns_none() {
        let reg = VectorIndexRegistry::new();
        let results = reg.search("Movie", "embedding", &[1.0, 0.0, 0.0], 10);
        assert!(results.is_none());
    }
}
