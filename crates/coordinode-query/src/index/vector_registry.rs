//! Vector index registry: tracks active HNSW indexes for vector search acceleration.
//!
//! Holds in-memory HNSW index instances keyed by (label, property).
//! Indexes are populated from stored vectors on Database::open() and
//! maintained incrementally on node create/update/delete.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use coordinode_cluster::VectorShardRouter;
use coordinode_core::graph::node::NodeId;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_vector::health::{HealthSignal, IndexHealthState};
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

/// A similarity-partitioned (sharded) vector index: N per-partition HNSW
/// handles plus the router that maps a vector to its partitions.
///
/// This is the index-internal partition axis (not cross-node sharding): a
/// label whose index is partitioned by embedding similarity holds one
/// `HnswHandle` per partition, and the [`VectorShardRouter`] decides which
/// partitions a build vector lands in (closure replication may write to
/// several) and which a query scatters to (adaptive fan-out). The CE registry
/// only provides the mechanism; the router itself (the partitioning
/// intelligence) is injected, defaulting to a single partition for the
/// Unsharded path, with the EE centroid router swapped in for sharded labels.
struct ShardedLayout {
    /// One HNSW handle per partition, indexed by `PartitionId`.
    shards: Vec<HnswHandle>,
    /// Maps a vector to its partitions (build `assign` / query `route`).
    router: Arc<dyn VectorShardRouter>,
}

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
    /// Active HNSW indexes: (label, property) → live HNSW graph. Holds the
    /// Unsharded (single-partition) indexes; sharded labels live in
    /// [`Self::sharded`] instead and are absent from this map.
    indexes: RwLock<HashMap<VectorIndexKey, HnswHandle>>,
    /// Similarity-partitioned indexes: (label, property) → N per-partition
    /// HNSW handles + router. A key is in exactly one of `indexes` (Unsharded)
    /// or `sharded` (partitioned), never both. Empty unless a label was
    /// registered with a multi-partition router via [`Self::register_sharded`].
    sharded: RwLock<HashMap<VectorIndexKey, ShardedLayout>>,
    /// Index definitions keyed by (label, property) for metadata lookup.
    definitions: RwLock<HashMap<VectorIndexKey, IndexDefinition>>,
    /// Per-index lifecycle + freshness signal. Held as an
    /// `Arc<HealthSignal>` so the read + serving path reads the current
    /// state and `indexed_hlc` watermark without taking the `indexes`
    /// write lock. Advanced by the oplog-tailing maintenance worker as it
    /// applies writes, and transitioned around rebuilds.
    health: RwLock<HashMap<VectorIndexKey, Arc<HealthSignal>>>,
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
            sharded: RwLock::new(HashMap::new()),
            definitions: RwLock::new(HashMap::new()),
            health: RwLock::new(HashMap::new()),
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
            sharded: RwLock::new(HashMap::new()),
            definitions: RwLock::new(HashMap::new()),
            health: RwLock::new(HashMap::new()),
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

        let mut hnsw = HnswIndex::new(Self::hnsw_config_from(config, def.property()));
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
        self.health
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), HealthSignal::new_ready());
        self.definitions
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, def);
    }

    /// Build the [`HnswConfig`] for an index from its vector config + property
    /// name. Shared by the Unsharded ([`Self::register_with_tier`]) and sharded
    /// ([`Self::register_sharded`]) registration paths so every per-partition
    /// graph uses identical parameters.
    fn hnsw_config_from(config: &crate::index::VectorIndexConfig, property: &str) -> HnswConfig {
        HnswConfig {
            m: config.m,
            m_max0: config.m * 2,
            ef_construction: config.ef_construction,
            ef_search: config.ef_search.unwrap_or(200),
            metric: config.metric,
            max_dimensions: config.dimensions,
            quantization: config.quantization,
            rerank_candidates: config.rerank_candidates.unwrap_or(100),
            calibration_threshold: 1000,
            offload_vectors: config.offload_vectors,
            property_name: property.to_string(),
            max_elements: 1_000_000,
            ..HnswConfig::default()
        }
    }

    /// Register a similarity-partitioned vector index: one empty HNSW graph per
    /// partition, routed by `router`. The label is stored in [`Self::sharded`]
    /// (not [`Self::indexes`]); the build path distributes vectors via
    /// [`VectorShardRouter::assign`] (closure replication may write a boundary
    /// vector to several partitions) and search scatter-gathers via
    /// [`VectorShardRouter::route`] (adaptive fan-out), merging by ascending
    /// score with dedup-by-id.
    ///
    /// `router.n_partitions()` graphs are created. The per-partition graphs are
    /// in-RAM (truth-tier wiring for the sharded path is a follow-up); the
    /// router (the partitioning intelligence) is injected, so the EE centroid
    /// router plugs in here without the registry depending on EE.
    ///
    /// Uses interior mutability — safe to call via `&self`.
    pub fn register_sharded(&self, def: IndexDefinition, router: Arc<dyn VectorShardRouter>) {
        let Some(config) = def.vector_config.as_ref() else {
            tracing::error!(
                "register_sharded called with non-vector IndexDefinition: {}",
                def.name
            );
            return;
        };
        let n = router.n_partitions().max(1);
        let shards: Vec<HnswHandle> = (0..n)
            .map(|_| {
                Arc::new(RwLock::new(HnswIndex::new(Self::hnsw_config_from(
                    config,
                    def.property(),
                ))))
            })
            .collect();

        let key = (def.label.clone(), def.property().to_string());
        self.sharded
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), ShardedLayout { shards, router });
        self.health
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), HealthSignal::new_ready());
        self.definitions
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, def);
    }

    /// Whether `(label, property)` is a similarity-partitioned (sharded) index.
    /// When `true`, build / search / insert route through the per-partition
    /// layout; when `false`, the single-graph Unsharded path is used,
    /// bit-identical to a registry with no sharding.
    pub fn is_sharded(&self, label: &str, property: &str) -> bool {
        self.sharded
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&(label.to_string(), property.to_string()))
    }

    /// Fold one partition's hits into the cross-partition best-by-id map,
    /// keeping the minimum score (closest occurrence) for an id seen in more
    /// than one partition (closure replication). `SearchResult.score` is a
    /// distance, so smaller wins.
    fn accumulate_best(best: &mut HashMap<u64, f32>, results: Vec<SearchResult>) {
        for r in results {
            best.entry(r.id)
                .and_modify(|s| {
                    if r.score < *s {
                        *s = r.score;
                    }
                })
                .or_insert(r.score);
        }
    }

    /// Materialise the merged best-by-id map into the final top-`k`: ascending
    /// by score (distance), truncated to `k`.
    fn finalize_merge(best: HashMap<u64, f32>, k: usize) -> Vec<SearchResult> {
        let mut out: Vec<SearchResult> = best
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect();
        out.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(k);
        out
    }

    /// Scatter-gather search over a sharded layout: route the query to its
    /// partitions, search each routed partition for `k`, then merge by
    /// ascending score keeping the best occurrence of each id (a vector
    /// replicated across partitions by closure replication appears once).
    /// `loader` is forwarded to each partition for disk-backed f32 rerank.
    fn search_sharded(
        &self,
        layout: &ShardedLayout,
        query: &[f32],
        k: usize,
        loader: Option<&dyn VectorLoader>,
    ) -> Vec<SearchResult> {
        let parts = layout.router.route(query, layout.shards.len());
        let mut best: HashMap<u64, f32> = HashMap::new();
        for &p in &parts {
            let Some(handle) = layout.shards.get(p as usize) else {
                continue;
            };
            let Ok(hnsw) = handle.read() else {
                continue;
            };
            let results = match loader {
                Some(l) => hnsw.search_with_loader(query, k, l),
                None => hnsw.search(query, k),
            };
            Self::accumulate_best(&mut best, results);
        }
        Self::finalize_merge(best, k)
    }

    /// Scatter-gather filtered (ACORN-style) search over a sharded layout: the
    /// same `is_visible` predicate runs on every routed partition, results
    /// merge by ascending score with dedup-by-id. The predicate is keyed by
    /// node id, so it is partition-independent and shared by reference across
    /// the routed partitions.
    fn search_visibility_sharded<F>(
        &self,
        layout: &ShardedLayout,
        query: &[f32],
        k: usize,
        overfetch_factor: f64,
        max_expansion_rounds: usize,
        is_visible: F,
    ) -> Vec<SearchResult>
    where
        F: Fn(u64) -> bool,
    {
        let parts = layout.router.route(query, layout.shards.len());
        let mut best: HashMap<u64, f32> = HashMap::new();
        for &p in &parts {
            let Some(handle) = layout.shards.get(p as usize) else {
                continue;
            };
            let Ok(hnsw) = handle.read() else {
                continue;
            };
            let (results, _stats) = hnsw.search_with_visibility(
                query,
                k,
                overfetch_factor,
                max_expansion_rounds,
                &is_visible,
            );
            Self::accumulate_best(&mut best, results);
        }
        Self::finalize_merge(best, k)
    }

    /// Distribute a batch of vectors into a sharded layout: group by assigned
    /// partition (closure replication clones a boundary vector into each), then
    /// `insert_batch` once per partition under a single write lock. Returns the
    /// number of distinct vectors (not insertions). Reused by the build and
    /// incremental-write paths.
    fn insert_batch_sharded(layout: &ShardedLayout, items: Vec<(u64, Vec<f32>)>) -> usize {
        let n = layout.shards.len();
        let mut per_shard: Vec<Vec<(u64, Vec<f32>)>> = (0..n).map(|_| Vec::new()).collect();
        let count = items.len();
        for (id, vec) in items {
            for &p in &layout.router.assign(&vec) {
                if (p as usize) < n {
                    per_shard[p as usize].push((id, vec.clone()));
                }
            }
        }
        for (p, batch) in per_shard.into_iter().enumerate() {
            if batch.is_empty() {
                continue;
            }
            if let Ok(mut hnsw) = layout.shards[p].write() {
                hnsw.insert_batch(batch);
            }
        }
        count
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
        self.health
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), HealthSignal::new_ready());
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
        self.sharded
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&key);
        self.health
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

    /// Lock-free handle to a (label, property) index's lifecycle + freshness
    /// signal. `None` when no such index exists. Cloning the
    /// `Arc` lets the serving path read state without the `indexes` lock.
    pub fn health_handle(&self, label: &str, property: &str) -> Option<Arc<HealthSignal>> {
        self.health
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(label.to_string(), property.to_string()))
            .cloned()
    }

    /// Current lifecycle state + freshness watermark of a (label, property)
    /// index. `None` when no such index exists. This is what the gRPC
    /// response metadata, EXPLAIN output, and Prometheus exporter read.
    pub fn health_snapshot(&self, label: &str, property: &str) -> Option<IndexHealthState> {
        self.health_handle(label, property).map(|h| h.snapshot())
    }

    /// Advance the freshness watermark of every maintained index to `hlc`
    /// (monotonic). Called by the oplog-tailing maintenance worker after it
    /// applies an entry (or batch) at commit HLC `hlc`: once the worker has
    /// consumed that entry, every index it maintains for the shard has seen
    /// all writes up to `hlc` — the ones with no vector-write at `hlc` are
    /// still current as of `hlc`. This is the per-shard read-your-writes
    /// fence surfaced per index.
    pub fn advance_indexed_hlc_all(&self, hlc: u64) {
        for h in self
            .health
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            h.advance_indexed_hlc(hlc);
        }
    }

    /// Transition a single index to `Ready` (e.g. after a rebuild completes).
    /// No-op when the index is unknown.
    pub fn mark_health_ready(&self, label: &str, property: &str) {
        if let Some(h) = self.health_handle(label, property) {
            h.mark_ready();
        }
    }

    /// Report rebuild progress for a single index, transitioning it to
    /// `Rebuilding`. No-op when the index is unknown.
    pub fn report_health_rebuild(&self, label: &str, property: &str, progress: f32, eta_ms: u64) {
        if let Some(h) = self.health_handle(label, property) {
            h.report_rebuild_progress(progress, eta_ms);
        }
    }

    /// Mark a single index `Offline` with a reason. No-op when unknown.
    pub fn mark_health_offline(&self, label: &str, property: &str, reason: impl Into<String>) {
        if let Some(h) = self.health_handle(label, property) {
            h.mark_offline(reason);
        }
    }

    /// Snapshot the health of every registered index as `(label, property,
    /// state)`. Used by the metrics exporter to publish per-index serving
    /// state and freshness-lag gauges each scrape interval.
    pub fn all_health(&self) -> Vec<(String, String, IndexHealthState)> {
        self.health
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|((label, property), h)| (label.clone(), property.clone(), h.snapshot()))
            .collect()
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

    /// Check if a vector index exists for a (label, property), in either the
    /// Unsharded or the sharded layout.
    pub fn has_index(&self, label: &str, property: &str) -> bool {
        let key = (label.to_string(), property.to_string());
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&key)
            || self
                .sharded
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .contains_key(&key)
    }

    /// Number of registered vector indexes (Unsharded + sharded). A sharded
    /// label counts once, not once per partition.
    pub fn len(&self) -> usize {
        self.indexes.read().unwrap_or_else(|e| e.into_inner()).len()
            + self.sharded.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether the registry is empty (no Unsharded and no sharded indexes).
    pub fn is_empty(&self) -> bool {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
            && self
                .sharded
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

    /// Property names of every vector index registered for `label`.
    /// Empty when the label has no vector indexes — callers use this
    /// as a cheap pre-filter before decoding replicated node records.
    pub fn indexed_properties(&self, label: &str) -> Vec<String> {
        self.definitions
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .filter(|(l, _)| l == label)
            .map(|(_, p)| p.clone())
            .collect()
    }

    /// Insert a vector into all applicable HNSW indexes for a node.
    ///
    /// Called on node creation or vector property update. For bulk
    /// loads use [`Self::on_vectors_written`] to amortise the HNSW
    /// write-lock across the whole batch instead of paying it per
    /// inserted vector.
    pub fn on_vector_written(&self, label: &str, node_id: NodeId, property: &str, vector: &[f32]) {
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&(label.to_string(), property.to_string())) {
                Self::insert_batch_sharded(layout, vec![(node_id.as_raw(), vector.to_vec())]);
                return;
            }
        }
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
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&(label.to_string(), property.to_string())) {
                let batch: Vec<(u64, Vec<f32>)> =
                    items.into_iter().map(|(id, v)| (id.as_raw(), v)).collect();
                Self::insert_batch_sharded(layout, batch);
                return;
            }
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
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&(label.to_string(), property.to_string())) {
                return Some(self.search_sharded(layout, query, k, None));
            }
        }
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
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&(label.to_string(), property.to_string())) {
                return Some(self.search_sharded(layout, query, k, loader));
            }
        }
        let handle = self.get(label, property)?;
        let hnsw = handle.read().ok()?;
        if let Some(loader) = loader {
            Some(hnsw.search_with_loader(query, k, loader))
        } else {
            Some(hnsw.search(query, k))
        }
    }

    /// HNSW top-K with a per-node visibility predicate (ACORN-style filtered
    /// search). The closure is called for every candidate the traversal
    /// considers; returning `false` prunes that branch. `overfetch_factor`
    /// scales the in-search candidate window, `max_expansion_rounds` caps
    /// how many times the inner loop expands when too many candidates fail
    /// the predicate. Returns `None` when no index exists for `(label,
    /// property)`.
    ///
    /// The HNSW engine never panics inside the closure: if the predicate
    /// throws (caller-side bug) the search returns whatever it had so far.
    #[allow(clippy::too_many_arguments)]
    pub fn search_with_visibility<F>(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        k: usize,
        overfetch_factor: f64,
        max_expansion_rounds: usize,
        is_visible: F,
    ) -> Option<Vec<SearchResult>>
    where
        F: Fn(u64) -> bool,
    {
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&(label.to_string(), property.to_string())) {
                return Some(self.search_visibility_sharded(
                    layout,
                    query,
                    k,
                    overfetch_factor,
                    max_expansion_rounds,
                    is_visible,
                ));
            }
        }
        let handle = self.get(label, property)?;
        let hnsw = handle.read().ok()?;
        let (results, _stats) = hnsw.search_with_visibility(
            query,
            k,
            overfetch_factor,
            max_expansion_rounds,
            is_visible,
        );
        Some(results)
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
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&(label.to_string(), property.to_string())) {
                return Self::insert_batch_sharded(layout, vectors.collect());
            }
        }
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
    use coordinode_cluster::{PartitionSet, VectorShardRouter};
    use coordinode_core::graph::types::VectorMetric;
    use std::sync::Arc;

    /// Test-double router: two partitions split on the sign of the first
    /// coordinate; a near-zero first coordinate (`|x| < 0.5`) is a boundary
    /// point replicated to BOTH partitions (closure replication) and a
    /// boundary query fans out to both (adaptive fan-out). Deterministic, no
    /// EE dependency.
    struct TwoPartitionRouter;

    impl VectorShardRouter for TwoPartitionRouter {
        fn assign(&self, v: &[f32]) -> PartitionSet {
            let x = v.first().copied().unwrap_or(0.0);
            let mut s = PartitionSet::new();
            if x.abs() < 0.5 {
                s.push(0);
                s.push(1);
            } else if x < 0.0 {
                s.push(0);
            } else {
                s.push(1);
            }
            s
        }

        fn route(&self, q: &[f32], _top_m: usize) -> PartitionSet {
            self.assign(q)
        }

        fn n_partitions(&self) -> usize {
            2
        }
    }

    fn test_config() -> VectorIndexConfig {
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::Cosine,
            m: 16,
            ef_construction: 200,
            quantization: coordinode_vector::hnsw::QuantizationCodec::None,
            offload_vectors: false,
            ef_search: None,
            rerank_candidates: None,
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
    fn health_tracks_state_and_freshness_watermark() {
        let reg = VectorIndexRegistry::new();
        reg.register(IndexDefinition::hnsw(
            "movie_embedding",
            "Movie",
            "embedding",
            test_config(),
        ));

        // Freshly registered → Ready with a zero watermark.
        assert_eq!(
            reg.health_snapshot("Movie", "embedding"),
            Some(IndexHealthState::Ready { indexed_hlc: 0 })
        );
        // Unknown index → no health.
        assert!(reg.health_snapshot("Movie", "missing").is_none());

        // Worker cursor advance lifts the watermark for every maintained index.
        reg.advance_indexed_hlc_all(4_200);
        assert_eq!(
            reg.health_snapshot("Movie", "embedding"),
            Some(IndexHealthState::Ready { indexed_hlc: 4_200 })
        );

        // Rebuild keeps the already-folded watermark.
        reg.report_health_rebuild("Movie", "embedding", 0.42, 1_000);
        let snap = reg.health_snapshot("Movie", "embedding");
        assert!(
            matches!(snap, Some(IndexHealthState::Rebuilding { .. })),
            "expected Rebuilding, got {snap:?}"
        );
        if let Some(IndexHealthState::Rebuilding {
            progress,
            indexed_hlc,
            ..
        }) = snap
        {
            assert!((progress - 0.42).abs() < 1e-3);
            assert_eq!(indexed_hlc, 4_200);
        }

        // Completion returns to Ready, watermark intact.
        reg.mark_health_ready("Movie", "embedding");
        assert_eq!(
            reg.health_snapshot("Movie", "embedding"),
            Some(IndexHealthState::Ready { indexed_hlc: 4_200 })
        );

        // Unregister drops the health entry.
        reg.unregister("Movie", "embedding");
        assert!(reg.health_snapshot("Movie", "embedding").is_none());
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

    #[test]
    fn register_sharded_marks_label_sharded() {
        let reg = VectorIndexRegistry::new();
        reg.register_sharded(
            IndexDefinition::hnsw("emb", "Doc", "embedding", test_config()),
            Arc::new(TwoPartitionRouter),
        );
        assert!(reg.is_sharded("Doc", "embedding"));
        assert!(reg.has_index("Doc", "embedding"));
        // A sharded key lives in the sharded map, not the single-index map.
        assert!(reg.get("Doc", "embedding").is_none());
        assert_eq!(reg.len(), 1);

        // Unregister clears the sharded entry too.
        reg.unregister("Doc", "embedding");
        assert!(!reg.is_sharded("Doc", "embedding"));
        assert!(!reg.has_index("Doc", "embedding"));
    }

    #[test]
    fn unsharded_register_is_not_sharded() {
        let reg = VectorIndexRegistry::new();
        reg.register(IndexDefinition::hnsw(
            "emb",
            "Doc",
            "embedding",
            test_config(),
        ));
        assert!(!reg.is_sharded("Doc", "embedding"));
        assert!(reg.get("Doc", "embedding").is_some());
    }

    #[test]
    fn sharded_search_matches_single_index_top1() {
        let vectors = [
            (1u64, vec![-1.0, 0.0, 0.0]),
            (2, vec![-0.9, 0.1, 0.0]),
            (3, vec![-0.8, 0.0, 0.1]),
            (4, vec![1.0, 0.0, 0.0]),
            (5, vec![0.9, 0.1, 0.0]),
            (6, vec![0.8, 0.0, 0.1]),
        ];

        // Single-index baseline.
        let single = VectorIndexRegistry::new();
        single.register(IndexDefinition::hnsw(
            "emb",
            "Doc",
            "embedding",
            test_config(),
        ));
        single.bulk_insert("Doc", "embedding", vectors.iter().cloned());

        // Sharded index, 2 partitions split on the first-coord sign.
        let sharded = VectorIndexRegistry::new();
        sharded.register_sharded(
            IndexDefinition::hnsw("emb", "Doc", "embedding", test_config()),
            Arc::new(TwoPartitionRouter),
        );
        sharded.bulk_insert("Doc", "embedding", vectors.iter().cloned());

        for q in [vec![-1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]] {
            let base = single.search("Doc", "embedding", &q, 1).unwrap();
            let sh = sharded.search("Doc", "embedding", &q, 1).unwrap();
            assert_eq!(sh.len(), 1);
            assert_eq!(
                sh[0].id, base[0].id,
                "sharded top-1 must match the single-index baseline"
            );
        }
    }

    #[test]
    fn sharded_dedup_replicated_boundary_id() {
        let reg = VectorIndexRegistry::new();
        reg.register_sharded(
            IndexDefinition::hnsw("emb", "Doc", "embedding", test_config()),
            Arc::new(TwoPartitionRouter),
        );
        // id 7 sits on the boundary (|x| < 0.5) -> replicated into BOTH
        // partitions; id 4 lands only in the positive partition.
        reg.bulk_insert(
            "Doc",
            "embedding",
            [(4u64, vec![1.0, 0.0, 0.0]), (7, vec![0.0, 1.0, 0.0])].into_iter(),
        );
        // The boundary query fans out to both partitions, so id 7 is found in
        // each; the merge must dedup it to a single result.
        let results = reg
            .search("Doc", "embedding", &[0.0, 1.0, 0.0], 10)
            .unwrap();
        let sevens = results.iter().filter(|r| r.id == 7).count();
        assert_eq!(sevens, 1, "a replicated id must appear once after merge");
    }

    #[test]
    fn sharded_search_with_visibility_filters_hidden_ids() {
        let reg = VectorIndexRegistry::new();
        reg.register_sharded(
            IndexDefinition::hnsw("emb", "Doc", "embedding", test_config()),
            Arc::new(TwoPartitionRouter),
        );
        reg.bulk_insert(
            "Doc",
            "embedding",
            [
                (1u64, vec![-1.0, 0.0, 0.0]),
                (2, vec![-0.9, 0.1, 0.0]),
                (4, vec![1.0, 0.0, 0.0]),
            ]
            .into_iter(),
        );

        // The query routes to the negative partition {1, 2}. Hiding id 1 must
        // prune it from the scatter-gathered, merged result.
        let filtered = reg
            .search_with_visibility("Doc", "embedding", &[-1.0, 0.0, 0.0], 5, 2.0, 4, |id| {
                id != 1
            })
            .unwrap();
        assert!(
            filtered.iter().all(|r| r.id != 1),
            "hidden id must not appear in sharded filtered search"
        );
        assert!(
            filtered.iter().any(|r| r.id == 2),
            "a visible neighbour is still returned"
        );

        // Unfiltered, the exact match (id 1) is the top hit through the
        // sharded visibility path.
        let unfiltered = reg
            .search_with_visibility("Doc", "embedding", &[-1.0, 0.0, 0.0], 5, 2.0, 4, |_| true)
            .unwrap();
        assert_eq!(
            unfiltered[0].id, 1,
            "unfiltered sharded top-1 is the exact match"
        );
    }
}
