//! Vector index registry: tracks active HNSW indexes for vector search acceleration.
//!
//! Holds in-memory HNSW index instances keyed by (label, property).
//! Indexes are populated from stored vectors on Database::open() and
//! maintained incrementally on node create/update/delete.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use coordinode_cluster::VectorShardRouter;
use coordinode_core::budget::Meter;
use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::VectorMetric;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_vector::VectorLoader;
use coordinode_vector::health::{HealthSignal, IndexHealthState};
use coordinode_vector::hnsw::{
    AdmissionError, DEFAULT_RETIRED_BYTES_BUDGET, HnswConfig, HnswIndex, PublicationSnapshot,
    SearchError, SearchResult, VectorDomainError,
};
use coordinode_vector::storage::lsm_backed::LsmVectorTier;
use coordinode_vector::storage::{VectorTierHandle, VectorTierStorage};

use super::coverage::{IndexCoverage, IndexDelta};
use super::definition::{GenerationId, IndexDefinition};

mod selection;

pub use selection::Reader;

/// Key for vector index lookup: (label, property).
type VectorIndexKey = (String, String);

/// The step a test runs in the thread starting a build, given the build's
/// token: after the index was registered for the build and its carriers
/// recorded, before the build starts.
#[cfg(any(test, feature = "test-hooks"))]
pub static BEFORE_START: parking_lot::Mutex<Option<super::vector_build::BeforeTakeHook>> =
    parking_lot::Mutex::new(None);

#[cfg(any(test, feature = "test-hooks"))]
fn before_start(token: &BuildToken) {
    let hook = BEFORE_START.lock().take();
    if let Some(hook) = hook {
        hook(token);
    }
}

/// The step a test runs in a cancel once it chose the build to stop and
/// released the table, before it waits for that build.
#[cfg(any(test, feature = "test-hooks"))]
pub static BEFORE_CANCEL_WAIT: parking_lot::Mutex<Option<Box<dyn FnOnce() + Send>>> =
    parking_lot::Mutex::new(None);

#[cfg(any(test, feature = "test-hooks"))]
fn before_cancel_wait() {
    let hook = BEFORE_CANCEL_WAIT.lock().take();
    if let Some(hook) = hook {
        hook();
    }
}

/// What a search goes by beside its query: the loader reranking from disk,
/// and which members it may return (all of them without one).
type SearchWith<'a> = (
    Option<&'a dyn VectorLoader>,
    Option<&'a dyn Fn(u64) -> bool>,
);

/// Thread-safe handle to an HNSW index. Searches and inserts share the lock;
/// the exclusive side is for the transitions that rewrite every node at once
/// (codec calibration, reorder, wholesale replacement).
pub type HnswHandle = Arc<RwLock<HnswIndex>>;

/// Insert one vector under the shared lock, concurrently with searches and
/// other inserts. The exclusive lock is taken only for the calibration the
/// insert reports due.
///
/// # Errors
///
/// The vector the index's metric cannot evaluate; the graph is unchanged.
pub(crate) fn insert_one(
    handle: &RwLock<HnswIndex>,
    id: u64,
    vector: &[f32],
) -> Result<(), AdmissionError> {
    let due = match handle.read() {
        Ok(graph) => graph.insert_shared(id, vector)?,
        Err(_) => return Ok(()),
    };
    if due {
        calibrate(handle);
    }
    Ok(())
}

/// [`insert_one`] for a batch, through [`HnswIndex::insert_batch_shared`].
///
/// # Errors
///
/// The first vector the index cannot take; none of the batch is inserted.
pub(crate) fn insert_many(
    handle: &RwLock<HnswIndex>,
    items: Vec<(u64, Vec<f32>)>,
) -> Result<(), AdmissionError> {
    let due = match handle.read() {
        Ok(graph) => graph.insert_batch_shared(items)?,
        Err(_) => return Ok(()),
    };
    if due {
        calibrate(handle);
    }
    Ok(())
}

/// What a partition's `results` may add to the merge map of a scattered
/// search: an id and a score each, before duplicates across partitions fold.
fn merged_bytes(results: &[SearchResult]) -> u64 {
    (results.len() * core::mem::size_of::<(u64, f64)>()) as u64
}

#[cold]
fn calibrate(handle: &RwLock<HnswIndex>) {
    if let Ok(mut graph) = handle.write() {
        graph.calibrate_if_due();
    }
}

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

impl ShardedLayout {
    /// The metric every partition evaluates.
    fn metric(&self) -> Option<VectorMetric> {
        let shard = self.shards.first()?.read().ok()?;
        Some(shard.config().metric)
    }

    /// The dimension a partition's first vector fixed, shared by them all.
    fn dimension(&self) -> Option<usize> {
        self.shards
            .iter()
            .find_map(|shard| shard.read().ok()?.dimension())
    }
}

/// Refuse the first of `items` an index of `metric` cannot take: non-finite,
/// empty, zero under cosine, or of a dimension other than `reference` (the
/// declared one, else the one the index fixed, else the first item's).
fn admit_all<'v>(
    metric: VectorMetric,
    mut reference: Option<usize>,
    items: impl IntoIterator<Item = (u64, &'v [f32])>,
) -> Result<(), AdmissionError> {
    for (id, vector) in items {
        coordinode_vector::metrics::exact::admit(vector, metric)
            .map_err(|error| AdmissionError { id, error })?;
        let dim = *reference.get_or_insert(vector.len());
        if dim != vector.len() {
            return Err(AdmissionError {
                id,
                error: VectorDomainError::DimensionMismatch {
                    left: vector.len(),
                    right: dim,
                },
            });
        }
    }
    Ok(())
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
    /// Optional persistent f32 truth tier backend. When set,
    /// callers obtain a `VectorTierHandle` via [`Self::tier_handle`]
    /// using pre-resolved `(label_id, property_id)` and pass it to
    /// [`Self::register_with_tier`]. The registry intentionally does
    /// NOT carry an interner reference — interning lives at the
    /// caller (executor, Database) so we never re-enter the same
    /// `parking_lot` RwLock from inside an active write transaction.
    tier_backend: Option<Arc<dyn VectorTierStorage>>,
    /// The build that owns each generation's index right now, from the moment
    /// the index is registered for it until the build ends.
    ///
    /// A build is a task the index owns, not a detached thread: DROP, a
    /// replacing CREATE or rebuild, and database shutdown all stop it here
    /// and wait for it before touching the index. Without that ownership the
    /// build outlives the index it belongs to and keeps writing under
    /// whatever comes next. Keyed by generation, a build of a dropped index
    /// can never be taken for one of an index created later under the same
    /// name.
    builds: Arc<Mutex<HashMap<GenerationId, BuildEntry>>>,
    /// Retired-memory budget each graph gets (see
    /// [`Self::set_retired_bytes_budget`]).
    retired_bytes_budget: AtomicUsize,
    /// The writes the indexes have not folded, when a worker maintains them
    /// from the applied entries; a registry without one serves what it holds.
    coverage: RwLock<Option<Arc<IndexCoverage>>>,
    /// The interpretations of its label each index holds, recorded by what
    /// filled it; an index without a record answers every reader whole.
    selections: RwLock<HashMap<VectorIndexKey, selection::Selection>>,
    /// Numbers each change of a record in [`Self::selections`], so a pass
    /// begun under one record never finishes under another.
    revisions: std::sync::atomic::AtomicU64,
    /// Moves whenever a record in [`Self::selections`] changes.
    held_changes: crate::index::selection::HeldChanges,
}

/// The build owning one generation's index.
///
/// Carries what it is building rather than looking it up: a build outlives the
/// registry entry for its index by exactly the window this type exists to
/// close, and a progress surface that went silent in that window would report
/// "no builds running" while one still holds the index.
struct BuildEntry {
    owner: Arc<BuildOwner>,
    /// How the index is named in progress reports.
    index: String,
    label: String,
    property: String,
}

impl BuildEntry {
    /// Stop this entry's owner, the table locked by the caller: an absorb,
    /// checking the flag under the same lock, then never moves the entry
    /// past the stop.
    fn stop(&self) -> Arc<BuildOwner> {
        self.owner.stop.store(true, Ordering::Relaxed);
        Arc::clone(&self.owner)
    }
}

/// One build, over one or more generations: the flag that stops it and the
/// signal that it let its indexes go.
struct BuildOwner {
    stop: Arc<AtomicBool>,
    released: Mutex<bool>,
    on_release: std::sync::Condvar,
}

impl BuildOwner {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            stop: Arc::new(AtomicBool::new(false)),
            released: Mutex::new(false),
            on_release: std::sync::Condvar::new(),
        })
    }

    /// Wait until the build has let its indexes go: it writes nothing to them
    /// afterwards.
    fn wait_released(&self) {
        let mut released = self.released.lock().unwrap_or_else(|e| e.into_inner());
        while !*released {
            released = self
                .on_release
                .wait(released)
                .unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// A build's ownership of the indexes it fills, from their registration until
/// it is dropped.
///
/// Whoever registers an index for a build holds the claim, through the start
/// of the build to its end. A newer owner (a rebuild, a DROP, shutdown) stops
/// the claim's build and waits for the claim to be dropped before writing:
/// a build cancelled before it started never starts, and one cancelled while
/// it runs writes nothing more once its claim is gone.
#[must_use = "dropping the claim lets the index go: only its holder may build it"]
pub struct BuildClaim {
    builds: Arc<Mutex<HashMap<GenerationId, BuildEntry>>>,
    owner: Arc<BuildOwner>,
    token: BuildToken,
    generations: Vec<GenerationId>,
}

impl BuildClaim {
    /// The token the claimed build runs under: cancelled once a newer owner
    /// takes any of its indexes over.
    pub fn token(&self) -> &BuildToken {
        &self.token
    }

    /// Take `other`'s indexes into this claim, so one build fills them all
    /// under one token. False, and nothing taken, when `other` was already
    /// cancelled: its index belongs to whoever cancelled it.
    pub fn absorb(&mut self, mut other: BuildClaim) -> bool {
        let mut builds = self.builds.lock().unwrap_or_else(|e| e.into_inner());
        // A cancel or a takeover stops an owner under this lock: one that
        // came first keeps `other`'s indexes, one that comes after finds
        // them under this claim and stops it.
        if other.owner.stop.load(Ordering::Relaxed) {
            return false;
        }
        for generation in other.generations.drain(..) {
            if let Some(entry) = builds.get_mut(&generation)
                && Arc::ptr_eq(&entry.owner, &other.owner)
            {
                entry.owner = Arc::clone(&self.owner);
                self.generations.push(generation);
            }
        }
        true
    }
}

impl Drop for BuildClaim {
    fn drop(&mut self) {
        {
            let mut builds = self.builds.lock().unwrap_or_else(|e| e.into_inner());
            for generation in &self.generations {
                if builds
                    .get(generation)
                    .is_some_and(|entry| Arc::ptr_eq(&entry.owner, &self.owner))
                {
                    builds.remove(generation);
                }
            }
        }
        *self
            .owner
            .released
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = true;
        self.owner.on_release.notify_all();
    }
}

/// Handed to a backfill so it can tell whether it should still be running.
///
/// The build checks this between batches; a cancelled build stops scanning
/// and, critically, writes nothing further — neither its terminal state nor
/// its progress — so the index it was building is free to be dropped or
/// replaced the moment the cancel returns.
#[derive(Clone)]
pub struct BuildToken {
    own: Arc<AtomicBool>,
    /// The token of whoever runs the build on its own thread (a worker
    /// rebuilding its indexes), which stops the build too.
    within: Option<Arc<AtomicBool>>,
}

impl BuildToken {
    /// True once someone has asked this build to stop.
    pub fn is_cancelled(&self) -> bool {
        self.own.load(Ordering::Relaxed)
            || self
                .within
                .as_ref()
                .is_some_and(|within| within.load(Ordering::Relaxed))
    }

    /// Ask the build to stop. The registry does this for the builds it owns;
    /// whoever runs a build of its own (a worker rebuilding its indexes)
    /// stops it with this.
    pub fn cancel(&self) {
        self.own.store(true, Ordering::Relaxed);
    }
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
            builds: Arc::new(Mutex::new(HashMap::new())),
            retired_bytes_budget: AtomicUsize::new(DEFAULT_RETIRED_BYTES_BUDGET),
            coverage: RwLock::new(None),
            selections: RwLock::new(HashMap::new()),
            revisions: std::sync::atomic::AtomicU64::new(0),
            held_changes: Default::default(),
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
            builds: Arc::new(Mutex::new(HashMap::new())),
            retired_bytes_budget: AtomicUsize::new(DEFAULT_RETIRED_BYTES_BUDGET),
            coverage: RwLock::new(None),
            selections: RwLock::new(HashMap::new()),
            revisions: std::sync::atomic::AtomicU64::new(0),
            held_changes: Default::default(),
        }
    }

    /// Learn from `coverage` which writes the indexes have not folded, so a
    /// search answers for them itself.
    pub fn set_coverage(&self, coverage: Arc<IndexCoverage>) {
        *self.coverage.write().unwrap_or_else(|e| e.into_inner()) = Some(coverage);
    }

    /// Where searches learn the writes the indexes have not folded, if a
    /// worker maintains them.
    pub fn coverage(&self) -> Option<Arc<IndexCoverage>> {
        self.coverage
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// The nodes of `shard_id` written by commits applied before this call
    /// that the indexes may not hold yet; none when no worker maintains them.
    pub fn delta(&self, shard_id: u16) -> IndexDelta {
        match self.coverage() {
            Some(coverage) => coverage.delta(shard_id),
            None => IndexDelta::Nodes(Default::default()),
        }
    }

    /// Set the retired-memory budget of every vector index, those registered
    /// later included: the bytes of replaced neighbour lists an index lets
    /// wait for reclamation before its writers hold off (see
    /// [`HnswConfig::retired_bytes_budget`]). Takes effect on the next write.
    pub fn set_retired_bytes_budget(&self, bytes: usize) {
        self.retired_bytes_budget.store(bytes, Ordering::Relaxed);
        for handle in self.all_graphs() {
            if let Ok(graph) = handle.read() {
                graph.set_retired_bytes_budget(bytes);
            }
        }
    }

    /// The retired-memory budget new indexes get.
    pub fn retired_bytes_budget(&self) -> usize {
        self.retired_bytes_budget.load(Ordering::Relaxed)
    }

    /// Every graph the registry holds: each plain index and each partition
    /// of a partitioned one.
    fn all_graphs(&self) -> Vec<HnswHandle> {
        let mut graphs: Vec<HnswHandle> = self
            .indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .cloned()
            .collect();
        for layout in self
            .sharded
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            graphs.extend(layout.shards.iter().cloned());
        }
        graphs
    }

    /// Publication and reclamation figures per `(label, property)`, summed
    /// over the partitions of a partitioned index (the oldest operation is
    /// the oldest of any partition). For the metrics a scrape samples.
    pub fn all_publication_stats(&self) -> Vec<(String, String, PublicationSnapshot)> {
        let sum = |graphs: &[HnswHandle]| {
            graphs
                .iter()
                .filter_map(|g| g.read().ok().map(|g| g.publication_stats()))
                .fold(PublicationSnapshot::default(), |acc, s| {
                    PublicationSnapshot {
                        retired_bytes: acc.retired_bytes + s.retired_bytes,
                        retired_lists: acc.retired_lists + s.retired_lists,
                        lost_cas: acc.lost_cas + s.lost_cas,
                        admission_waits: acc.admission_waits + s.admission_waits,
                        admission_wait: acc.admission_wait + s.admission_wait,
                        oldest_operation: acc.oldest_operation.max(s.oldest_operation),
                        retired_nodes: acc.retired_nodes + s.retired_nodes,
                        free_slots: acc.free_slots + s.free_slots,
                    }
                })
        };
        let mut out: Vec<(String, String, PublicationSnapshot)> = self
            .indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|((label, property), handle)| {
                (
                    label.clone(),
                    property.clone(),
                    sum(std::slice::from_ref(handle)),
                )
            })
            .collect();
        out.extend(
            self.sharded
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|((label, property), layout)| {
                    (label.clone(), property.clone(), sum(&layout.shards))
                }),
        );
        out
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
    /// `vec:<label_id><property_id><node_id>`.
    ///
    /// The caller is responsible for populating the index with
    /// existing vectors (see `bulk_insert`).
    ///
    /// Uses interior mutability — safe to call via `&self`.
    pub fn register_with_tier(&self, def: IndexDefinition, tier: Option<VectorTierHandle>) {
        self.insert_index(def, tier, HealthSignal::new_ready());
    }

    /// [`Self::register_with_tier`] for an index a build is about to fill:
    /// published already rebuilding, so neither a reader nor a writer ever
    /// takes the empty graph for a finished one.
    ///
    /// Returns the claim on the index's build when it registered: the index
    /// is that build's from here, though the build has not started, and a
    /// newer owner stops it through the claim. A generation already
    /// registered keeps its graph, its health signal and the build filling
    /// them, and the caller starts no build of its own: the statement that
    /// created the index and this member's catalog follower both register it.
    pub fn register_for_build(
        &self,
        def: IndexDefinition,
        tier: Option<VectorTierHandle>,
    ) -> Option<BuildClaim> {
        let key = (def.label.clone(), def.property().to_string());
        // Held from the check to the insert, so two registrations of one
        // generation cannot both pass the check.
        let mut definitions = self.definitions.write().unwrap_or_else(|e| e.into_inner());
        if definitions
            .get(&key)
            .is_some_and(|registered| registered.generation == def.generation)
        {
            return None;
        }
        let hnsw = self.new_graph(&def, tier)?;
        // A build of this generation outliving its unregistered index fills
        // the graph that went with it, not the one registered here: stopped,
        // it is not waited for under the definitions lock, which the build it
        // waits for may need.
        let claim = self.claim_entries(&[&def], None, |_| {});
        let health = HealthSignal::new_ready();
        health.report_rebuild_progress(0.0, 0);
        self.forget_selection(&key);
        self.indexes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), hnsw);
        self.health
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), health);
        definitions.insert(key, def);
        Some(claim)
    }

    /// Take the indexes of `defs` over for a build run by the caller on its
    /// own thread, under `within`, which stops it too: every build owning one
    /// of them is stopped and waited for first, so nothing but this build
    /// writes them while the claim is held.
    pub fn claim_build(&self, defs: &[&IndexDefinition], within: &BuildToken) -> BuildClaim {
        self.claim_entries(defs, Some(Arc::clone(&within.own)), |previous| {
            previous.wait_released();
        })
    }

    /// Enter one owner for `defs`' generations. A generation owned already
    /// has its owner stopped under the table's lock, then handed to
    /// `take_over` with the table unlocked.
    fn claim_entries(
        &self,
        defs: &[&IndexDefinition],
        within: Option<Arc<AtomicBool>>,
        take_over: impl Fn(&BuildOwner),
    ) -> BuildClaim {
        let owner = BuildOwner::new();
        let token = BuildToken {
            own: Arc::clone(&owner.stop),
            within,
        };
        let mut generations = Vec::with_capacity(defs.len());
        for def in defs {
            loop {
                let previous = {
                    let mut builds = self.builds.lock().unwrap_or_else(|e| e.into_inner());
                    match builds.get(&def.generation) {
                        Some(entry) => entry.stop(),
                        None => {
                            builds.insert(
                                def.generation,
                                BuildEntry {
                                    owner: Arc::clone(&owner),
                                    index: def.to_string(),
                                    label: def.label.clone(),
                                    property: def.property().to_string(),
                                },
                            );
                            break;
                        }
                    }
                };
                take_over(&previous);
                let mut builds = self.builds.lock().unwrap_or_else(|e| e.into_inner());
                // Taken over without waiting: the previous owner's claim
                // leaves this entry alone when it is dropped.
                if builds
                    .get(&def.generation)
                    .is_some_and(|entry| Arc::ptr_eq(&entry.owner, &previous))
                {
                    builds.remove(&def.generation);
                }
            }
            generations.push(def.generation);
        }
        BuildClaim {
            builds: Arc::clone(&self.builds),
            owner,
            token,
            generations,
        }
    }

    fn insert_index(
        &self,
        def: IndexDefinition,
        tier: Option<VectorTierHandle>,
        health: Arc<HealthSignal>,
    ) {
        let Some(hnsw) = self.new_graph(&def, tier) else {
            return;
        };
        let key = (def.label.clone(), def.property().to_string());
        self.forget_selection(&key);
        self.indexes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), hnsw);
        self.health
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.clone(), health);
        self.definitions
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, def);
    }

    /// An empty graph for `def` bound to `tier`, or `None` when `def` is not
    /// a vector index.
    fn new_graph(
        &self,
        def: &IndexDefinition,
        tier: Option<VectorTierHandle>,
    ) -> Option<HnswHandle> {
        let Some(config) = def.vector_config.as_ref() else {
            tracing::error!("register called with non-vector IndexDefinition: {def}");
            return None;
        };
        let mut hnsw = HnswIndex::new(self.hnsw_config_from(config, def.property()));
        // Bind the caller-resolved tier handle BEFORE the index is
        // moved into the registry so subsequent inserts persist f32
        // to disk. `None` keeps the index pure in-RAM
        // (test path, ad-hoc analytics).
        hnsw.set_vector_tier(tier);
        Some(Arc::new(RwLock::new(hnsw)))
    }

    /// Build the [`HnswConfig`] for an index from its vector config + property
    /// name. Shared by the Unsharded ([`Self::register_with_tier`]) and sharded
    /// ([`Self::register_sharded`]) registration paths so every per-partition
    /// graph uses identical parameters.
    fn hnsw_config_from(
        &self,
        config: &crate::index::VectorIndexConfig,
        property: &str,
    ) -> HnswConfig {
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
            retired_bytes_budget: self.retired_bytes_budget(),
            ..HnswConfig::default()
        }
    }

    /// Register a similarity-partitioned vector index: one empty HNSW graph per
    /// partition, routed by `router`. The label is stored with the partitioned
    /// indexes, not the plain ones; the build path distributes vectors via
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
            tracing::error!("register_sharded called with non-vector IndexDefinition: {def}");
            return;
        };
        let n = router.n_partitions().max(1);
        let shards: Vec<HnswHandle> = (0..n)
            .map(|_| {
                Arc::new(RwLock::new(HnswIndex::new(
                    self.hnsw_config_from(config, def.property()),
                )))
            })
            .collect();

        let key = (def.label.clone(), def.property().to_string());
        self.forget_selection(&key);
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
    fn accumulate_best(best: &mut HashMap<u64, f64>, results: Vec<SearchResult>) {
        for r in results {
            best.entry(r.id)
                .and_modify(|s| {
                    // Exact scores are finite: plain comparison is total here.
                    if r.score < *s {
                        *s = r.score;
                    }
                })
                .or_insert(r.score);
        }
    }

    /// Materialise the merged best-by-id map into the final top-`k`: ascending
    /// by score (distance), equal scores by id, truncated to `k`. The map's
    /// iteration order is arbitrary, so the id tie-break is what makes the
    /// merge return the same results on every run.
    fn finalize_merge(best: HashMap<u64, f64>, k: usize) -> Vec<SearchResult> {
        let mut out: Vec<SearchResult> = best
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect();
        out.sort_by(|a, b| {
            coordinode_vector::metrics::exact::ranked_order(a.score, a.id, b.score, b.id)
        });
        out.truncate(k);
        out
    }

    /// Scatter-gather search over a sharded layout: route the query to its
    /// partitions, search each routed partition for `k`, then merge by
    /// ascending score keeping the best occurrence of each id (a vector
    /// replicated across partitions by closure replication appears once).
    /// `loader` is forwarded to each partition for disk-backed f32 rerank.
    fn search_sharded<M: Meter>(
        &self,
        layout: &ShardedLayout,
        query: &[f32],
        k: usize,
        (loader, admits): SearchWith<'_>,
        meter: &mut M,
    ) -> Result<Vec<SearchResult>, SearchError<M::Stop>> {
        let parts = layout.router.route(query, layout.shards.len());
        let mut best: HashMap<u64, f64> = HashMap::new();
        for &p in &parts {
            let Some(handle) = layout.shards.get(p as usize) else {
                continue;
            };
            let Ok(hnsw) = handle.read() else {
                continue;
            };
            let results = match (admits, loader) {
                (Some(admits), _) => hnsw.search_admitting_metered(query, k, admits, meter)?,
                (None, Some(l)) => hnsw.search_with_loader_metered(query, k, l, meter)?,
                (None, None) => hnsw.search_metered(query, k, meter)?,
            };
            meter
                .scratch(merged_bytes(&results))
                .map_err(SearchError::Stopped)?;
            Self::accumulate_best(&mut best, results);
        }
        Ok(Self::finalize_merge(best, k))
    }

    /// Distribute a batch of vectors into a sharded layout: group by assigned
    /// partition (closure replication clones a boundary vector into each), then
    /// `insert_batch` once per partition under a single write lock. Returns the
    /// number of distinct vectors (not insertions). Reused by the build and
    /// incremental-write paths.
    ///
    /// # Errors
    ///
    /// The first vector a partition cannot take. Every vector is checked
    /// before any partition is written, so a refused batch changes none.
    fn insert_batch_sharded(
        layout: &ShardedLayout,
        declared: Option<usize>,
        items: Vec<(u64, Vec<f32>)>,
    ) -> Result<usize, AdmissionError> {
        let n = layout.shards.len();
        // Routing reads the vector, so the whole batch is checked first,
        // against the one dimension every partition shares.
        if let Some(metric) = layout.metric() {
            admit_all(
                metric,
                declared.or_else(|| layout.dimension()),
                items.iter().map(|(id, v)| (*id, v.as_slice())),
            )?;
        }
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
            insert_many(&layout.shards[p], batch)?;
        }
        Ok(count)
    }

    /// Register an index with a pre-built HNSW graph (e.g. after rebuild).
    ///
    /// Uses interior mutability — safe to call via `&self`.
    pub fn register_with_index(&self, def: IndexDefinition, hnsw: HnswIndex) {
        hnsw.set_retired_bytes_budget(self.retired_bytes_budget());
        let key = (def.label.clone(), def.property().to_string());
        self.forget_selection(&key);
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
        self.forget_selection(&key);
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
    /// (monotonic). The maintenance worker calls it with a cut every write at
    /// or below which it has folded, whatever order those commits took their
    /// timestamps in; indexes no write touched are current as of the cut too.
    /// This is the per-shard read-your-writes fence surfaced per index.
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

    /// Whether a running build, not the writer, is responsible for folding
    /// writes into this index right now.
    ///
    /// While an index is being built the writer skips it: the build is
    /// scanning and draining in batches, which encodes and links far cheaper
    /// per vector than a write at a time, and it will pick this write up in
    /// its next drain pass anyway. Doing both would pay the expensive path
    /// during exactly the window the batched path exists for. Readers see the
    /// partial index under the documented rebuild policy until the build ends.
    fn build_owns_writes(&self, label: &str, property: &str) -> bool {
        self.health_handle(label, property)
            .is_some_and(|h| h.build_owns_writes())
    }

    /// Mint a cancellation token for a build run by its caller that owns no
    /// index of the registry: a worker's stop, a test's build.
    pub fn new_build_token(&self) -> BuildToken {
        BuildToken {
            own: Arc::new(AtomicBool::new(false)),
            within: None,
        }
    }

    /// Start the build `claim` owns on a thread named `thread`. `build` runs
    /// with the claim's token, and the claim is let go once it returns.
    ///
    /// A claim a newer owner cancelled before this start runs no build: the
    /// index is that owner's, and a build starting now would fill it under
    /// what it was given at registration.
    ///
    /// # Errors
    ///
    /// The thread could not be spawned; the claim is let go.
    pub fn spawn_build(
        &self,
        claim: BuildClaim,
        thread: String,
        build: impl FnOnce(BuildToken) + Send + 'static,
    ) -> std::io::Result<()> {
        #[cfg(any(test, feature = "test-hooks"))]
        before_start(claim.token());
        std::thread::Builder::new().name(thread).spawn(move || {
            let token = claim.token().clone();
            if token.is_cancelled() {
                // What the build holds (the engine among it) goes before
                // the claim, whose release tells a waiting owner it is gone.
                drop(build);
            } else {
                build(token);
            }
            drop(claim);
        })?;
        Ok(())
    }

    /// Whether a build owns `generation`'s index: registered for it, or
    /// running, and not yet ended.
    pub fn is_building(&self, generation: GenerationId) -> bool {
        self.builds
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&generation)
    }

    /// Stop the build owning `generation`'s index and wait for it to let the
    /// index go.
    ///
    /// Returns whether a build owned it. On return the build has stopped
    /// touching the index AND its persisted definition, which is what lets a
    /// caller drop or replace the index immediately afterwards without racing
    /// a write it cannot see. A cancel arriving while another one waits for
    /// the same build waits for it too.
    pub fn cancel_build(&self, generation: GenerationId) -> bool {
        let owner = self
            .builds
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&generation)
            .map(BuildEntry::stop);
        let Some(owner) = owner else {
            return false;
        };
        #[cfg(any(test, feature = "test-hooks"))]
        before_cancel_wait();
        // Its claim removes the entry as it lets the index go.
        owner.wait_released();
        true
    }

    /// Stop every build and wait for each. Used on shutdown, where a build
    /// left running would keep writing into an engine that is closing.
    pub fn cancel_all_builds(&self) {
        let owners: Vec<Arc<BuildOwner>> = self
            .builds
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(BuildEntry::stop)
            .collect();
        for owner in owners {
            owner.wait_released();
        }
    }

    /// Names of the indexes whose builds own them right now.
    pub fn running_builds(&self) -> Vec<String> {
        self.builds
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|h| h.index.clone())
            .collect()
    }

    /// Every backfill running right now, with the live progress of the index
    /// it is building.
    ///
    /// One node's answer: an index partitioned across nodes reports a build
    /// per node, and it is the coordinator that aggregates them into the
    /// cluster-wide picture. The progress comes from the same atomic signal
    /// the search path reads, so asking costs no lock on the build.
    pub fn build_progress(&self) -> Vec<(String, String, String, Option<IndexHealthState>)> {
        let running: Vec<(String, String, String)> = self
            .builds
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|h| (h.index.clone(), h.label.clone(), h.property.clone()))
            .collect();
        running
            .into_iter()
            .map(|(name, label, property)| {
                let state = self.health_snapshot(&label, &property);
                (name, label, property, state)
            })
            .collect()
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
            .find(|def| def.name.as_deref() == Some(name))
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

    /// Whether a vector index is registered for `label`.
    pub fn indexes_label(&self, label: &str) -> bool {
        self.definitions
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .any(|(l, _)| l == label)
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

    /// The dimension the definition of `(label, property)` declares, if it
    /// declares one.
    fn declared_dimension(&self, key: &VectorIndexKey) -> Option<usize> {
        self.definitions
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)?
            .vector_config
            .as_ref()
            .map(|config| config.dimensions as usize)
            .filter(|&dimensions| dimensions > 0)
    }

    /// Check, without changing the index, that the index of `(label,
    /// property)` can take `vector` for `node_id`. A write is refused with
    /// this before it commits, so every committed vector of an indexed
    /// property is one its index takes. No index: nothing to check.
    ///
    /// # Errors
    ///
    /// The reason the index refuses the vector: non-finite, empty, zero under
    /// cosine, or of another dimension than the declared or established one.
    pub fn check_admission(
        &self,
        label: &str,
        property: &str,
        node_id: NodeId,
        vector: &[f32],
    ) -> Result<(), AdmissionError> {
        let key = (label.to_string(), property.to_string());
        let declared = self.declared_dimension(&key);
        let id = node_id.as_raw();
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&key) {
                return match layout.metric() {
                    Some(metric) => admit_all(
                        metric,
                        declared.or_else(|| layout.dimension()),
                        [(id, vector)],
                    ),
                    None => Ok(()),
                };
            }
        }
        let Some(handle) = self.get(label, property) else {
            return Ok(());
        };
        let Ok(graph) = handle.read() else {
            return Ok(());
        };
        admit_all(
            graph.config().metric,
            declared.or_else(|| graph.dimension()),
            [(id, vector)],
        )
    }

    /// Insert a vector into all applicable HNSW indexes for a node.
    ///
    /// Called on node creation or vector property update. For bulk
    /// loads use [`Self::on_vectors_written`], which spreads one batch
    /// across the rayon pool.
    ///
    /// # Errors
    ///
    /// The vector the index refuses (see [`Self::check_admission`]); the
    /// index is unchanged.
    pub fn on_vector_written(
        &self,
        label: &str,
        node_id: NodeId,
        property: &str,
        vector: &[f32],
    ) -> Result<(), AdmissionError> {
        self.on_vectors_written(label, property, vec![(node_id, vector.to_vec())])
    }

    /// Batched variant: insert N vectors into the same (label,
    /// property) HNSW index through [`HnswIndex::insert_batch_shared`],
    /// which seeds sequentially up to `SEED_DENSITY` and parallelises
    /// the rest via rayon, concurrently with searches.
    ///
    /// Caller is responsible for grouping by (label, property): one call
    /// per index keeps the rayon fan-out per batch rather than per row.
    ///
    /// `items` is consumed: each `(NodeId, Vec<f32>)` is forwarded
    /// to the HNSW insert path without further copies.
    ///
    /// # Errors
    ///
    /// The first vector the index refuses; none of the batch is inserted.
    pub fn on_vectors_written(
        &self,
        label: &str,
        property: &str,
        items: Vec<(NodeId, Vec<f32>)>,
    ) -> Result<(), AdmissionError> {
        if items.is_empty() || self.build_owns_writes(label, property) {
            return Ok(());
        }
        let items: Vec<(u64, Vec<f32>)> =
            items.into_iter().map(|(id, v)| (id.as_raw(), v)).collect();
        self.insert_into(label, property, items).map(drop)
    }

    /// Insert `items` into the index of `(label, property)`, every vector
    /// checked first against the declared dimension; the number inserted.
    fn insert_into(
        &self,
        label: &str,
        property: &str,
        items: Vec<(u64, Vec<f32>)>,
    ) -> Result<usize, AdmissionError> {
        let key = (label.to_string(), property.to_string());
        let declared = self.declared_dimension(&key);
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&key) {
                return Self::insert_batch_sharded(layout, declared, items);
            }
        }
        let Some(handle) = self.get(label, property) else {
            return Ok(0);
        };
        if let Some(dim) = declared
            && let Ok(graph) = handle.read()
        {
            admit_all(
                graph.config().metric,
                Some(dim),
                items.iter().map(|(id, v)| (*id, v.as_slice())),
            )?;
        }
        let count = items.len();
        insert_many(&handle, items)?;
        Ok(count)
    }

    /// A statement deleted a node or its vector property, ahead of its
    /// commit. The graph keeps the node: removing it now would lose the
    /// vector of a deletion that then aborts. The MVCC visibility filter
    /// hides it from results until the committed deletion reaches
    /// [`Self::on_vector_removed`] through the applied-commit feed.
    pub fn on_vector_deleted(&self, _label: &str, _node_id: NodeId, _property: &str) {}

    /// Take `node_id` out of the index of `(label, property)`, every
    /// partition of a partitioned one, for a deletion that is committed.
    /// Runs beside searches and inserts; an absent node is a no-op. While a
    /// build owns the index, the build's own fold reconciles the node.
    pub fn on_vector_removed(&self, label: &str, property: &str, node_id: NodeId) {
        if self.build_owns_writes(label, property) {
            return;
        }
        self.remove_member(label, property, node_id.as_raw());
    }

    /// Take `id` out of every graph of the index of `(label, property)`.
    fn remove_member(&self, label: &str, property: &str, id: u64) {
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&(label.to_string(), property.to_string())) {
                for shard in &layout.shards {
                    if let Ok(graph) = shard.read() {
                        graph.remove(id);
                    }
                }
                return;
            }
        }
        if let Some(handle) = self.get(label, property)
            && let Ok(graph) = handle.read()
        {
            graph.remove(id);
        }
    }

    /// Whether the index of `(label, property)`, any partition of a
    /// partitioned one, holds `node_id`.
    pub fn holds(&self, label: &str, property: &str, node_id: NodeId) -> bool {
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&(label.to_string(), property.to_string())) {
                return layout
                    .shards
                    .iter()
                    .any(|shard| shard.read().is_ok_and(|g| g.contains(node_id.as_raw())));
            }
        }
        self.get(label, property)
            .is_some_and(|handle| handle.read().is_ok_and(|g| g.contains(node_id.as_raw())))
    }

    /// Search the HNSW index for a (label, property) pair, reporting the
    /// search's progress to `meter`, which can stop it.
    ///
    /// Returns `None` if no index exists, or the search results if found.
    ///
    /// # Errors
    ///
    /// The stop `meter` returned, or a query or candidate the index's metric
    /// cannot evaluate.
    pub fn search<M: Meter>(
        &self,
        (label, property): (&str, &str),
        reader: Reader<'_>,
        query: &[f32],
        k: usize,
        meter: &mut M,
    ) -> Result<Option<Vec<SearchResult>>, SearchError<M::Stop>> {
        self.search_with_loader((label, property), reader, query, k, None, meter)
    }

    /// Search with optional VectorLoader for disk-backed f32 reranking,
    /// reporting the search's progress to `meter`, answering `reader` with
    /// its own interpretation of the label.
    ///
    /// When the HNSW index has offloaded f32 vectors, the loader provides
    /// them on-demand for exact reranking. Falls back to in-memory search
    /// when no loader is provided or vectors are not offloaded. `Ok(None)`
    /// when there is no such index or it does not hold the reader's
    /// interpretation, which the store answers then.
    ///
    /// # Errors
    ///
    /// The stop `meter` returned, or a query or candidate the index's metric
    /// cannot evaluate.
    pub fn search_with_loader<M: Meter>(
        &self,
        (label, property): (&str, &str),
        reader: Reader<'_>,
        query: &[f32],
        k: usize,
        loader: Option<&dyn VectorLoader>,
        meter: &mut M,
    ) -> Result<Option<Vec<SearchResult>>, SearchError<M::Stop>> {
        self.with_admission((label, property), reader, |admits| {
            self.search_admitted((label, property), query, k, (loader, admits), meter)
        })
    }

    /// The `k` nearest members of `(label, property)`'s index that `admits`
    /// takes, all of them without it.
    fn search_admitted<M: Meter>(
        &self,
        (label, property): (&str, &str),
        query: &[f32],
        k: usize,
        (loader, admits): SearchWith<'_>,
        meter: &mut M,
    ) -> Result<Option<Vec<SearchResult>>, SearchError<M::Stop>> {
        {
            let sharded = self.sharded.read().unwrap_or_else(|e| e.into_inner());
            if let Some(layout) = sharded.get(&(label.to_string(), property.to_string())) {
                return self
                    .search_sharded(layout, query, k, (loader, admits), meter)
                    .map(Some);
            }
        }
        let Some(handle) = self.get(label, property) else {
            return Ok(None);
        };
        let Ok(hnsw) = handle.read() else {
            return Ok(None);
        };
        match (admits, loader) {
            (Some(admits), _) => hnsw.search_admitting_metered(query, k, admits, meter),
            (None, Some(loader)) => hnsw.search_with_loader_metered(query, k, loader, meter),
            (None, None) => hnsw.search_metered(query, k, meter),
        }
        .map(Some)
    }

    /// HNSW top-K of the members `is_visible` takes (filtered search): a
    /// member it refuses still leads the traversal on but is never a result,
    /// so the search ends with `k` visible members whenever the graph holds
    /// them. Answers `reader` with its own interpretation of the label, as
    /// [`Self::search_with_loader`] does. The search's progress is reported
    /// to `meter`, which can stop it.
    ///
    /// # Errors
    ///
    /// As [`Self::search_with_loader`].
    pub fn search_with_visibility<F, M: Meter>(
        &self,
        (label, property): (&str, &str),
        reader: Reader<'_>,
        query: &[f32],
        k: usize,
        is_visible: F,
        meter: &mut M,
    ) -> Result<Option<Vec<SearchResult>>, SearchError<M::Stop>>
    where
        F: Fn(u64) -> bool,
    {
        self.with_admission((label, property), reader, |admits| {
            let both = |id: u64| admits.is_none_or(|admits| admits(id)) && is_visible(id);
            self.search_admitted((label, property), query, k, (None, Some(&both)), meter)
        })
    }

    /// Bulk-insert multiple vectors into a specific HNSW index.
    ///
    /// Used during index rebuild (on Database::open or CREATE VECTOR INDEX).
    /// The caller is responsible for scanning stored nodes and extracting
    /// vectors — this method just does the HNSW insertions.
    ///
    /// # Errors
    ///
    /// The first vector the index refuses; none of the batch is inserted.
    pub fn bulk_insert(
        &self,
        label: &str,
        property: &str,
        vectors: impl Iterator<Item = (u64, Vec<f32>)>,
    ) -> Result<usize, AdmissionError> {
        self.insert_into(label, property, vectors.collect())
    }
}

impl Default for VectorIndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
