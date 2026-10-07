//! Index metadata definitions.
//!
//! These are catalog records: plain serializable descriptors of an index
//! (kind, target label/property, vector/text config, build state). They live
//! at Layer 4 alongside [`crate::IndexStore`], which owns their persistence in
//! the schema catalog. The query layer issues logical DDL and reads the
//! catalog for planning, but does not own the definition type or its storage
//! keyspace (mirrors how mature engines place the schema/index catalog below
//! the query engine).

use coordinode_core::graph::types::VectorMetric;
pub use coordinode_core::index::identity::{GenerationId, IndexId};
use serde::{Deserialize, Serialize};

/// Reader behaviour while an index is in [`IndexState::Building`].
///
/// Default [`Self::Block`] preserves the pre-state semantics: queries see
/// a fully-built index because they wait for the backfill to finish.
/// `PartialRecall` lets queries hit the partial graph immediately at the
/// cost of recall < 1.0; `Offline` rejects queries with a clear error so
/// the caller can route to a fallback path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnlineDuringBuild {
    /// Reader waits, up to the bound its caller chose (`vector_build_wait`),
    /// until the build on the member serving it holds what the store held
    /// when the build began, then proceeds.
    #[default]
    Block,
    /// Reader uses the partial index immediately. Recall improves as the
    /// backfill writes more vectors.
    PartialRecall,
    /// Reader gets `IndexBuilding` so it can pick a fallback path.
    Offline,
}

/// Build state of an index.
///
/// `Ready` is the steady state for any index whose data is fully populated.
/// `Building` is set while a backfill task is running. `Failed` captures the
/// error reason when a backfill aborts. Persisting this lets a reopen path
/// detect interrupted backfills and resume them.
// Default Ready so pre-existing persisted IndexDefinition records (which
// had no state field) deserialize as fully-built indexes. New indexes set
// Building explicitly before spawning backfill.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexState {
    /// Backfill in progress. `written` is approximate, updated in batches.
    Building {
        /// Approximate count of entries written so far, updated in batches.
        written: u64,
        /// Estimated total entries the backfill expects to write.
        estimated_total: u64,
    },
    /// Backfill complete, index reflects all matching data.
    #[default]
    Ready,
    /// Backfill aborted; readers consult policy to decide error / partial.
    Failed {
        /// Human-readable reason the backfill aborted.
        reason: String,
    },
}

/// Type of index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// Standard B-tree index (single or compound).
    BTree,
    /// HNSW vector index for approximate nearest neighbor search.
    Hnsw,
    /// Flat brute-force vector index for exact NN on small datasets (<100K).
    Flat,
    /// Full-text search index backed by tantivy.
    Text,
}

pub use coordinode_core::index::filter::PartialFilter;

/// Definition of an index: the catalog record of one logical index, with
/// the identities the catalog gave it when it published it.
///
/// Reads of the descriptor go through [`Deref`](core::ops::Deref), so
/// `def.label` is the label of the index whatever holds it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDefinition {
    /// The logical index, stable across rename, description edits and
    /// rebuilds.
    pub id: IndexId,
    /// The representation the index serves from and maintains: the scope of
    /// its entry keys. A rebuild writes a new one.
    pub generation: GenerationId,
    /// What the index is.
    pub descriptor: IndexDescriptor,
}

impl core::ops::Deref for IndexDefinition {
    type Target = IndexDescriptor;

    fn deref(&self) -> &IndexDescriptor {
        &self.descriptor
    }
}

impl core::ops::DerefMut for IndexDefinition {
    fn deref_mut(&mut self) -> &mut IndexDescriptor {
        &mut self.descriptor
    }
}

impl core::fmt::Display for IndexDefinition {
    /// The name when the index has one, its identity otherwise: how logs and
    /// messages refer to it.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.descriptor.name {
            Some(name) => f.write_str(name),
            None => write!(f, "{}", self.id),
        }
    }
}

/// What an index is, as DDL declares it and its catalog record keeps it:
/// everything but the identities the catalog gives it at publication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDescriptor {
    /// Optional alias, unique among the live indexes of the catalog. The
    /// index is addressable by its identity with or without it.
    pub name: Option<String>,
    /// Optional description. Catalog metadata only: no entry, effect or
    /// interpretation carries it.
    pub description: Option<String>,
    /// Node label this index applies to (e.g., "User").
    pub label: String,
    /// Indexed property names. Single-field: 1 entry. Compound: 2+ entries.
    /// Order matters for compound indexes: the key is encoded in this order.
    pub properties: Vec<String>,
    /// Index type.
    pub index_type: IndexType,
    /// Whether this index enforces uniqueness.
    pub unique: bool,
    /// Sparse: skip nodes where any indexed property is null/missing.
    pub sparse: bool,
    /// Multikey flag: set when any indexed node has an array value in an
    /// indexed property. Once set, only cleared by full rebuild.
    pub multikey: bool,
    /// Partial index filter: only index nodes matching this predicate.
    pub filter: Option<PartialFilter>,
    /// TTL: expire nodes after this many seconds from the indexed timestamp.
    /// Only valid on single-field Timestamp indexes.
    pub ttl_seconds: Option<u64>,
    /// Vector index configuration. Only set when `index_type` is `Hnsw` or `Flat`.
    pub vector_config: Option<VectorIndexConfig>,
    /// Text index configuration. Only set when `index_type` is `Text`.
    pub text_config: Option<TextIndexConfig>,
    /// Build state.
    pub state: IndexState,
    /// Reader behaviour during `IndexState::Building`.
    pub online_during_build: OnlineDuringBuild,
    /// Key layout the generation's entries are written in.
    pub layout: u32,
    /// How a key-shaped index's entries reach every member, and under which
    /// policy epoch.
    pub maintenance: IndexMaintenance,
    /// The constraint this index enforces, which creates and drops it; `None`
    /// for an index of its own.
    pub owner: Option<String>,
}

/// How a key-shaped index's entry effects travel with the data they index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexProfile {
    /// The executing member computes the entries and the log carries them.
    #[default]
    Resolved,
    /// The log carries the sealed interpretation and exact inputs, and every
    /// member derives the entries itself.
    Derived,
}

/// Where an index's effective profile comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileSource {
    /// Inherited from the namespace default, as it stood at this revision
    /// of the namespace policy.
    Namespace {
        /// The policy revision the profile was resolved from.
        revision: u64,
    },
    /// Declared on the index itself.
    Override,
}

impl Default for ProfileSource {
    fn default() -> Self {
        Self::Namespace { revision: 0 }
    }
}

/// The namespace's default profile for key-shaped indexes, replicated with
/// the catalog. Changing it affects indexes created after the change; an
/// existing index moves only through its own explicit transition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamespaceIndexPolicy {
    /// The profile a new index without an override takes.
    pub default: IndexProfile,
    /// Moves on every change of the default; an inheriting index records
    /// the revision it resolved.
    pub revision: u64,
}

impl NamespaceIndexPolicy {
    /// Schema catalog key of the policy.
    pub const KEY: &'static [u8] = b"schema:index_policy";

    /// The binding a new index gets: `requested` as an override, or the
    /// namespace default at this revision.
    pub fn resolve(&self, requested: Option<IndexProfile>, epoch: u64) -> IndexMaintenance {
        match requested {
            Some(profile) => IndexMaintenance {
                profile,
                source: ProfileSource::Override,
                epoch,
            },
            None => IndexMaintenance {
                profile: self.default,
                source: ProfileSource::Namespace {
                    revision: self.revision,
                },
                epoch,
            },
        }
    }
}

/// An index's effective maintenance binding: the profile, where it comes
/// from, and the epoch every effect staged under it is sealed with.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexMaintenance {
    /// The effective profile.
    pub profile: IndexProfile,
    /// Its source.
    pub source: ProfileSource,
    /// The maintenance-policy epoch: moves on every profile transition, so
    /// effects sealed under an earlier binding are told apart.
    pub epoch: u64,
}

/// The entry layout every index is written in: entries staged in the
/// writing transaction, unique indexes keyed by value alone, values in the
/// injective tuple encoding. The key codec the shared derivation writes.
pub const ENTRY_LAYOUT: u32 = coordinode_core::index::derive::KEY_CODEC;

/// Per-field analyzer configuration for text indexes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextFieldConfig {
    /// Analyzer name: language name ("english", "russian"), "auto_detect", or "none".
    pub analyzer: String,
}

/// Configuration for full-text search indexes (tantivy-backed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextIndexConfig {
    /// Per-field analyzer overrides. Key = property name, value = analyzer config.
    /// Fields not listed use `default_language` as analyzer.
    pub fields: std::collections::HashMap<String, TextFieldConfig>,
    /// Default language/analyzer for fields without explicit config.
    pub default_language: String,
    /// Node property name that overrides the default language per-node.
    /// Default: "_language".
    pub language_override_property: String,
}

impl Default for TextIndexConfig {
    fn default() -> Self {
        Self {
            fields: std::collections::HashMap::new(),
            default_language: "english".to_string(),
            language_override_property: "_language".to_string(),
        }
    }
}

/// Configuration for vector indexes (HNSW or Flat).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorIndexConfig {
    /// Number of dimensions in the vector.
    pub dimensions: u32,
    /// Distance metric for similarity computation.
    pub metric: VectorMetric,
    /// HNSW M parameter: max bi-directional links per element (default 16).
    pub m: usize,
    /// HNSW ef_construction: candidate list size during build (default 200).
    pub ef_construction: usize,
    /// In-RAM quantization codec for HNSW traversal. See
    /// [`coordinode_vector::hnsw::QuantizationCodec`].
    pub quantization: coordinode_vector::hnsw::QuantizationCodec,
    /// When `quantization` is `Sq8` and this flag is set, f32 vectors are
    /// not retained in HNSW memory. Reranking loads f32 from storage via
    /// VectorLoader. Gives 4x RAM reduction at ~1-2ms rerank cost per search.
    pub offload_vectors: bool,
    /// Default size of the dynamic candidate list during search (HNSW
    /// `ef_search`). Larger values trade latency for recall; required to be
    /// raised on adversarial / sparsely-connected data. `None` uses the engine
    /// default (200). Configured via the `ef_search` CREATE VECTOR INDEX option.
    #[serde(default)]
    pub ef_search: Option<usize>,
    /// Number of approximate candidates re-scored with exact f32 distance
    /// before returning the top-k. `None` uses the engine default (100).
    /// Configured via the `rerank_candidates` CREATE VECTOR INDEX option.
    #[serde(default)]
    pub rerank_candidates: Option<usize>,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            dimensions: 0,
            metric: VectorMetric::Cosine,
            m: 16,
            ef_construction: 200,
            quantization: coordinode_vector::hnsw::QuantizationCodec::None,
            offload_vectors: false,
            ef_search: None,
            rerank_candidates: None,
        }
    }
}

impl IndexDescriptor {
    fn new(
        name: impl Into<String>,
        label: impl Into<String>,
        properties: Vec<String>,
        index_type: IndexType,
    ) -> Self {
        Self {
            name: Some(name.into()),
            description: None,
            label: label.into(),
            properties,
            index_type,
            unique: false,
            sparse: false,
            multikey: false,
            filter: None,
            ttl_seconds: None,
            vector_config: None,
            text_config: None,
            state: IndexState::Ready,
            online_during_build: OnlineDuringBuild::Block,
            layout: ENTRY_LAYOUT,
            maintenance: IndexMaintenance::default(),
            owner: None,
        }
    }

    /// A single-field B-tree index named `name`.
    pub fn btree(
        name: impl Into<String>,
        label: impl Into<String>,
        property: impl Into<String>,
    ) -> Self {
        Self::new(name, label, vec![property.into()], IndexType::BTree)
    }

    /// A compound B-tree index named `name` on several properties.
    pub fn compound(
        name: impl Into<String>,
        label: impl Into<String>,
        properties: Vec<String>,
    ) -> Self {
        Self::new(name, label, properties, IndexType::BTree)
    }

    /// An HNSW vector index named `name` on one property. Sparse: a node
    /// without the vector has no entry.
    pub fn hnsw(
        name: impl Into<String>,
        label: impl Into<String>,
        property: impl Into<String>,
        config: VectorIndexConfig,
    ) -> Self {
        Self {
            sparse: true,
            vector_config: Some(config),
            ..Self::new(name, label, vec![property.into()], IndexType::Hnsw)
        }
    }

    /// A full-text search index named `name` on one or more properties.
    pub fn text(
        name: impl Into<String>,
        label: impl Into<String>,
        properties: Vec<String>,
        config: TextIndexConfig,
    ) -> Self {
        Self {
            sparse: true,
            text_config: Some(config),
            ..Self::new(name, label, properties, IndexType::Text)
        }
    }

    /// The definition of this index under the identities `id` and
    /// `generation`. The catalog binds a descriptor when it publishes it
    /// ([`crate::IndexStore::publish_definition_txn`]); binding one by hand
    /// is for a store that has no catalog, such as a unit test's.
    pub fn bind(self, id: IndexId, generation: GenerationId) -> IndexDefinition {
        IndexDefinition {
            id,
            generation,
            descriptor: self,
        }
    }

    /// Set unique constraint.
    pub fn unique(mut self) -> Self {
        self.unique = true;
        self
    }

    /// Make this the index the constraint `constraint` enforces.
    pub fn owned_by(mut self, constraint: impl Into<String>) -> Self {
        self.owner = Some(constraint.into());
        self
    }

    /// Set sparse flag (skip null/missing values).
    pub fn sparse(mut self) -> Self {
        self.sparse = true;
        self
    }

    /// Set partial index filter predicate.
    pub fn with_filter(mut self, filter: PartialFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Check if a node matches this index's partial filter.
    /// Returns true if no filter (all nodes match) or if filter is satisfied.
    pub fn matches_filter(
        &self,
        properties: &[(String, coordinode_core::graph::types::Value)],
    ) -> bool {
        match &self.filter {
            None => true,
            Some(f) => f.matches(properties),
        }
    }

    /// Whether this is a compound index (2+ properties).
    pub fn is_compound(&self) -> bool {
        self.properties.len() > 1
    }

    /// First (or only) property name.
    pub fn property(&self) -> &str {
        self.properties.first().map_or("", |s| s.as_str())
    }
}

impl IndexDefinition {
    /// The interpretation that decides this B-tree index's entries, with
    /// each property resolved to the field id `field_of` binds it to now:
    /// what a DERIVED effect is sealed with, so no member consults the
    /// catalog or the dictionary to derive it.
    pub fn interpretation(
        &self,
        field_of: &dyn Fn(&str) -> Option<u32>,
    ) -> coordinode_core::index::derive::IndexInterpretation {
        use coordinode_core::index::derive::{IndexInterpretation, MembershipFilter, PropertyRef};
        let property = |name: &str| PropertyRef {
            field: field_of(name),
            name: name.to_owned(),
        };
        IndexInterpretation {
            codec: self.layout,
            generation: self.generation,
            unique: self.unique,
            sparse: self.sparse,
            properties: self.properties.iter().map(|p| property(p)).collect(),
            filter: self.filter.as_ref().map(|f| match f {
                PartialFilter::PropertyEquals { property: p, value } => {
                    MembershipFilter::EqualsString(property(p), value.clone())
                }
                PartialFilter::PropertyEqualsInt { property: p, value } => {
                    MembershipFilter::EqualsInt(property(p), *value)
                }
                PartialFilter::PropertyEqualsBool { property: p, value } => {
                    MembershipFilter::EqualsBool(property(p), *value)
                }
                PartialFilter::PropertyExists { property: p } => {
                    MembershipFilter::Exists(property(p))
                }
            }),
        }
    }

    /// The binding a DERIVED effect of this index is sealed under.
    pub fn binding(
        &self,
        field_of: &dyn Fn(&str) -> Option<u32>,
    ) -> coordinode_core::txn::proposal::IndexBinding {
        coordinode_core::txn::proposal::IndexBinding {
            epoch: self.maintenance.epoch,
            interpretation: self.interpretation(field_of),
        }
    }

    /// Schema storage key for this index definition.
    pub fn schema_key(&self) -> Vec<u8> {
        Self::schema_key_of(self.id)
    }

    /// Prefix of every definition record in the schema catalog.
    pub const SCHEMA_PREFIX: &'static [u8] = b"schema:idx:";

    /// Prefix of every name binding in the schema catalog.
    pub const NAME_PREFIX: &'static [u8] = b"schema:idxname:";

    /// Schema catalog key of the identity allocator: the next unallocated
    /// index and generation numbers. Outside both prefixes above.
    pub const ALLOCATOR_KEY: &'static [u8] = b"schema:idx_alloc";

    /// The catalog key of the definition of the index `id`.
    pub fn schema_key_of(id: IndexId) -> Vec<u8> {
        let mut key = Vec::with_capacity(Self::SCHEMA_PREFIX.len() + 8);
        key.extend_from_slice(Self::SCHEMA_PREFIX);
        key.extend_from_slice(&id.as_raw().to_be_bytes());
        key
    }

    /// The catalog key binding the name `name` to the index that holds it.
    pub fn name_key_of(name: &str) -> Vec<u8> {
        let mut key = Vec::with_capacity(Self::NAME_PREFIX.len() + name.len());
        key.extend_from_slice(Self::NAME_PREFIX);
        key.extend_from_slice(name.as_bytes());
        key
    }
}

/// What a build does when the stored data refuses its index (a unique index
/// over values held twice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildFailure {
    /// The index being created is withdrawn with the constraint that owns
    /// it: nobody was told it exists.
    Withdraw,
    /// The index being rebuilt is kept, marked failed: its constraint still
    /// holds for new writes, lookups stop using it.
    Keep,
}

/// Where one build stands. Every move is a catalog commit conditioned on
/// the record the mover read, so publication, failure and cancellation
/// cannot all win.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildState {
    /// Admitted with the index's publication; no executor has taken it.
    Accepted,
    /// Taken by the executor `executor`, which alone may finish it.
    Running {
        /// The executor's token, fresh for every take.
        executor: u64,
    },
    /// The index was published ready.
    Published,
    /// The stored data refused the index; `reason` says why.
    Failed {
        /// Why the build failed.
        reason: String,
    },
    /// Cancelled before it finished; its candidate entries are cleared.
    Cancelled,
}

impl BuildState {
    /// Whether the build has an outcome and no executor will touch it again.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Published | Self::Failed { .. } | Self::Cancelled
        )
    }
}

/// What a build of a unique index may do about two stored nodes holding one
/// value: `ON DUPLICATE RENAME property`. A policy of the build operation,
/// not of the index: later writes are refused as duplicates as always.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DuplicateRepair {
    /// The string property the build may change on a conflicting node, by
    /// appending a random suffix to its value. One of the index's
    /// properties.
    pub property: String,
}

/// The durable record of one index build: the operation a CREATE or a
/// rebuild admits, independent of the request, connection or thread that
/// asked for it. Its generation is its identity: one build fills one
/// representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexBuildRecord {
    /// The representation the build fills, and the build's identity.
    pub generation: GenerationId,
    /// The logical index it belongs to.
    pub index: IndexId,
    /// What the build does when the stored data refuses the index.
    pub on_failure: BuildFailure,
    /// Where the build stands.
    pub state: BuildState,
    /// Nodes indexed so far, reported by the executor; progress, not proof.
    pub indexed: u64,
    /// Whether the build may repair a duplicate it meets, and through which
    /// property; `None` fails the build on the first duplicate.
    pub on_duplicate: Option<DuplicateRepair>,
    /// Repairs committed so far. Each is a [`DuplicateRepairRecord`]
    /// committed with the data change it records and with this count.
    pub repaired: u64,
}

impl IndexBuildRecord {
    /// Prefix of every build record in the schema catalog. Outside the
    /// definition and name prefixes.
    pub const PREFIX: &'static [u8] = b"schema:idxbuild:";

    /// A build of `generation` of the index `index`, admitted and not taken,
    /// failing on the first duplicate it meets.
    pub fn accepted(index: IndexId, generation: GenerationId, on_failure: BuildFailure) -> Self {
        Self {
            generation,
            index,
            on_failure,
            state: BuildState::Accepted,
            indexed: 0,
            on_duplicate: None,
            repaired: 0,
        }
    }

    /// This build, repairing the duplicates it meets as `repair` says.
    #[must_use]
    pub fn repairing(mut self, repair: Option<DuplicateRepair>) -> Self {
        self.on_duplicate = repair;
        self
    }

    /// The catalog key of the build of `generation`.
    pub fn key_of(generation: GenerationId) -> Vec<u8> {
        let mut key = Vec::with_capacity(Self::PREFIX.len() + 8);
        key.extend_from_slice(Self::PREFIX);
        key.extend_from_slice(&generation.as_raw().to_be_bytes());
        key
    }
}

/// One repair a build made: the value of `property` of `node` it replaced to
/// end a duplicate. Committed in the transaction that changed the node, so
/// the record exists exactly when the change does.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DuplicateRepairRecord {
    /// The build that made it.
    pub generation: GenerationId,
    /// The node it changed.
    pub node: u64,
    /// The property it changed.
    pub property: String,
    /// The value the node held, which another node held too.
    pub old: String,
    /// The value it holds since.
    pub new: String,
}

impl DuplicateRepairRecord {
    /// Prefix of every repair record in the schema catalog.
    pub const PREFIX: &'static [u8] = b"schema:idxrepair:";

    /// The prefix of the repairs of the build of `generation`.
    pub fn prefix_of(generation: GenerationId) -> Vec<u8> {
        let mut key = Vec::with_capacity(Self::PREFIX.len() + 8);
        key.extend_from_slice(Self::PREFIX);
        key.extend_from_slice(&generation.as_raw().to_be_bytes());
        key
    }

    /// The catalog key of this repair: one per build and node, so a node
    /// repaired twice by one build keeps its latest repair.
    pub fn key(&self) -> Vec<u8> {
        let mut key = Self::prefix_of(self.generation);
        key.extend_from_slice(&self.node.to_be_bytes());
        key
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
