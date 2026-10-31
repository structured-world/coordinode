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
    /// The namespace's built-in label lookup: one posting per node and label
    /// it holds, maintained by every write. Never declared by DDL; one per
    /// namespace, with no label or properties of its own.
    Labels,
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
    /// Entries the interpretations of a label disagree on, kept apart from
    /// [`Self::generation`] while an implication change is certified, in
    /// force but not folded in yet, or withdrawn but not folded back: each
    /// holds only that difference. Empty in the steady state.
    pub selection: Vec<SelectionPart>,
}

/// Entries of one label of an index kept apart because an implication
/// change makes the interpretations disagree on them. Its entries are keyed
/// exactly as the index's own, under [`Self::part`] in place of the
/// index's generation, so it holds only the difference, never a copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectionPart {
    /// The label whose membership differs: for the label lookup, the label
    /// of its postings; for a declared index, the index's label.
    pub label: coordinode_core::schema::label_id::LabelId,
    /// The key scope its entries are written under, from the generation
    /// allocator.
    pub part: GenerationId,
    /// The certification of the implication change it belongs to.
    pub operation: GenerationId,
    /// Which interpretation alone holds its entries.
    pub role: PartRole,
    /// Where the change stands.
    pub state: PartState,
}

/// Which interpretation alone holds the entries of a [`SelectionPart`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartRole {
    /// Carriers the change adds: members under the candidate only.
    Gained,
    /// Carriers the change removes: members under the facts it replaces
    /// only, moved out of the index's own entries before it is put in force.
    Retired,
}

/// Where the implication change a [`SelectionPart`] belongs to stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartState {
    /// Being certified: the facts in force read the retired part, the
    /// candidate the gained one.
    Proposed,
    /// Put in force: the gained part is read until it is folded into the
    /// index's own entries; the retired one serves only views before it.
    InForce,
    /// Withdrawn: the retired part is read until it is folded back; the
    /// gained one is read by nobody.
    Withdrawn,
}

/// The interpretation a reader or writer holds a label to, as its view of
/// the catalog has it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpretation {
    /// The facts in force at the view.
    InForce,
    /// The candidate of the certification named, published at the view.
    Candidate(GenerationId),
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

/// [`IndexDefinition`] as encoded, all but its parts skipped.
#[derive(Deserialize)]
struct SelectionView {
    #[serde(rename = "id")]
    _id: serde::de::IgnoredAny,
    #[serde(rename = "generation")]
    _generation: serde::de::IgnoredAny,
    #[serde(rename = "descriptor")]
    _descriptor: serde::de::IgnoredAny,
    selection: Vec<SelectionPart>,
}

/// [`IndexDefinition`] as encoded, its parts counted and nothing kept.
#[derive(Deserialize)]
struct SelectionLenView {
    #[serde(rename = "id")]
    _id: serde::de::IgnoredAny,
    #[serde(rename = "generation")]
    _generation: serde::de::IgnoredAny,
    #[serde(rename = "descriptor")]
    _descriptor: serde::de::IgnoredAny,
    selection: ListLen,
}

/// [`IndexDefinition`] as encoded, measured as
/// [`IndexDefinition::for_placement_from_msgpack`] decodes it.
#[derive(Deserialize)]
struct PlacementSizeView {
    #[serde(rename = "id")]
    _id: serde::de::IgnoredAny,
    #[serde(rename = "generation")]
    _generation: serde::de::IgnoredAny,
    descriptor: DescriptorSizeView,
    selection: ListLen,
}

/// [`IndexDescriptor`] as encoded, its kept strings and names measured.
#[derive(Deserialize)]
struct DescriptorSizeView {
    name: StrBytes,
    description: StrBytes,
    label: StrBytes,
    properties: NamesSize,
    #[serde(rename = "index_type")]
    _index_type: serde::de::IgnoredAny,
    #[serde(rename = "unique")]
    _unique: serde::de::IgnoredAny,
    #[serde(rename = "sparse")]
    _sparse: serde::de::IgnoredAny,
    #[serde(rename = "multikey")]
    _multikey: serde::de::IgnoredAny,
    filter: StrBytes,
    #[serde(rename = "ttl_seconds")]
    _ttl_seconds: serde::de::IgnoredAny,
    #[serde(rename = "vector_config")]
    _vector_config: serde::de::IgnoredAny,
    #[serde(rename = "text_config")]
    _text_config: serde::de::IgnoredAny,
    state: StrBytes,
    #[serde(rename = "online_during_build")]
    _online_during_build: serde::de::IgnoredAny,
    #[serde(rename = "layout")]
    _layout: serde::de::IgnoredAny,
    #[serde(rename = "maintenance")]
    _maintenance: serde::de::IgnoredAny,
    owner: StrBytes,
}

/// The bytes of every string a value holds, however nested, read without
/// keeping any.
struct StrBytes(usize);

impl<'de> Deserialize<'de> for StrBytes {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Sum;
        impl<'de> serde::de::Visitor<'de> for Sum {
            type Value = StrBytes;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("any value")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<StrBytes, E> {
                Ok(StrBytes(v.len()))
            }
            fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<StrBytes, E> {
                Ok(StrBytes(0))
            }
            fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<StrBytes, E> {
                Ok(StrBytes(0))
            }
            fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<StrBytes, E> {
                Ok(StrBytes(0))
            }
            fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<StrBytes, E> {
                Ok(StrBytes(0))
            }
            fn visit_bytes<E: serde::de::Error>(self, _: &[u8]) -> Result<StrBytes, E> {
                Ok(StrBytes(0))
            }
            fn visit_none<E: serde::de::Error>(self) -> Result<StrBytes, E> {
                Ok(StrBytes(0))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<StrBytes, E> {
                Ok(StrBytes(0))
            }
            fn visit_some<D: serde::Deserializer<'de>>(self, d: D) -> Result<StrBytes, D::Error> {
                StrBytes::deserialize(d)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<StrBytes, A::Error> {
                let mut n = 0usize;
                while let Some(StrBytes(bytes)) = seq.next_element()? {
                    n += bytes;
                }
                Ok(StrBytes(n))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<StrBytes, A::Error> {
                let mut n = 0usize;
                while let Some((StrBytes(k), StrBytes(v))) = map.next_entry()? {
                    n += k + v;
                }
                Ok(StrBytes(n))
            }
        }
        deserializer.deserialize_any(Sum)
    }
}

/// How many names a list holds and their bytes, read without keeping any.
struct NamesSize {
    names: usize,
    bytes: usize,
}

impl<'de> Deserialize<'de> for NamesSize {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Sum;
        impl<'de> serde::de::Visitor<'de> for Sum {
            type Value = NamesSize;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a list of names")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<NamesSize, A::Error> {
                let mut size = NamesSize { names: 0, bytes: 0 };
                while let Some(StrBytes(bytes)) = seq.next_element()? {
                    size.names += 1;
                    size.bytes += bytes;
                }
                Ok(size)
            }
        }
        deserializer.deserialize_seq(Sum)
    }
}

/// A list read for its length alone, its elements skipped unread.
struct ListLen(usize);

impl<'de> Deserialize<'de> for ListLen {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Counter;
        impl<'de> serde::de::Visitor<'de> for Counter {
            type Value = ListLen;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a list")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<ListLen, A::Error> {
                let mut n = 0usize;
                while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {
                    n += 1;
                }
                Ok(ListLen(n))
            }
        }
        deserializer.deserialize_seq(Counter)
    }
}

/// [`IndexDefinition`] as encoded, its descriptor read as
/// [`PlacementDescriptor`]. The fields carry the definition's names and
/// order, so either encoding of a struct reads the same fields.
#[derive(Deserialize)]
struct PlacementView {
    id: IndexId,
    generation: GenerationId,
    descriptor: PlacementDescriptor,
    selection: Vec<SelectionPart>,
}

/// [`IndexDescriptor`] as encoded, its vector and text configurations
/// skipped.
#[derive(Deserialize)]
struct PlacementDescriptor {
    name: Option<String>,
    description: Option<String>,
    label: String,
    properties: Vec<String>,
    index_type: IndexType,
    unique: bool,
    sparse: bool,
    multikey: bool,
    filter: Option<PartialFilter>,
    ttl_seconds: Option<u64>,
    #[serde(rename = "vector_config")]
    _vector_config: serde::de::IgnoredAny,
    #[serde(rename = "text_config")]
    _text_config: serde::de::IgnoredAny,
    state: IndexState,
    online_during_build: OnlineDuringBuild,
    layout: u32,
    maintenance: IndexMaintenance,
    owner: Option<String>,
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

/// What a successful check, build or qualified import proved about one
/// member's copy of an index generation: the copy agreed with its records,
/// both ways, under the generation's interpretation, at `verified_at`. Kept
/// by the member, node-local, and carried by a transfer of the copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CopyQualification {
    /// The installation holding the proved copy on the member keeping this.
    pub installation: u64,
    /// The key layout the copy's entries were proved in.
    pub layout: u32,
    /// The maintenance binding the copy's entries were proved under.
    pub maintenance: IndexMaintenance,
    /// The snapshot the proof read at: an MVCC timestamp, or 0 on an engine
    /// without one.
    pub verified_at: u64,
}

impl CopyQualification {
    /// Whether this proves entries of `def`'s interpretation: the same key
    /// layout and maintenance binding.
    pub fn interprets(&self, def: &IndexDefinition) -> bool {
        self.layout == def.layout && self.maintenance == def.maintenance
    }
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

    /// The namespace's built-in label lookup, published building: no name,
    /// label or properties of its own, written in the entry layout.
    pub fn label_lookup() -> Self {
        Self {
            name: None,
            state: IndexState::Building {
                written: 0,
                estimated_total: 0,
            },
            ..Self::new("", "", Vec::new(), IndexType::Labels)
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
            selection: Vec::new(),
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

/// A vector decoded from a list of `n` elements holds at most twice `n`
/// slots: its decoder sets room for the whole list up to a cap, and grows by
/// doubling past it.
const LIST_ROOM: usize = 2;

impl IndexDefinition {
    /// The parts of the definition encoded in `bytes`, nothing else of it
    /// decoded: what a pass looking for the indexes a certification touches
    /// reads of every index.
    ///
    /// # Errors
    ///
    /// The bytes are not an encoded definition.
    pub fn selection_from_msgpack(
        bytes: &[u8],
    ) -> Result<Vec<SelectionPart>, rmp_serde::decode::Error> {
        let view: SelectionView = rmp_serde::from_slice(bytes)?;
        Ok(view.selection)
    }

    /// An upper bound of the heap bytes [`Self::selection_from_msgpack`]
    /// takes for the definition encoded in `bytes`, read from the length of
    /// its parts' list alone: nothing allocated, every other field skipped.
    ///
    /// # Errors
    ///
    /// The bytes are not an encoded definition.
    pub fn selection_bound(bytes: &[u8]) -> Result<usize, rmp_serde::decode::Error> {
        let view: SelectionLenView = rmp_serde::from_slice(bytes)?;
        Ok(LIST_ROOM * view.selection.0 * core::mem::size_of::<SelectionPart>())
    }

    /// The definition encoded in `bytes` without its vector and text
    /// configurations, which are skipped unread: all a label lookup or a
    /// B-tree index holds, so for those it is the definition as stored.
    ///
    /// # Errors
    ///
    /// The bytes are not an encoded definition.
    pub fn for_placement_from_msgpack(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        let view: PlacementView = rmp_serde::from_slice(bytes)?;
        let d = view.descriptor;
        Ok(Self {
            id: view.id,
            generation: view.generation,
            descriptor: IndexDescriptor {
                name: d.name,
                description: d.description,
                label: d.label,
                properties: d.properties,
                index_type: d.index_type,
                unique: d.unique,
                sparse: d.sparse,
                multikey: d.multikey,
                filter: d.filter,
                ttl_seconds: d.ttl_seconds,
                vector_config: None,
                text_config: None,
                state: d.state,
                online_during_build: d.online_during_build,
                layout: d.layout,
                maintenance: d.maintenance,
                owner: d.owner,
            },
            selection: view.selection,
        })
    }

    /// An upper bound of the heap bytes [`Self::for_placement_from_msgpack`]
    /// takes for the definition encoded in `bytes`, measured from the fields
    /// it decodes, nothing allocated: the bytes of every string it keeps, a
    /// slot for each indexed property and each part, and room for those
    /// lists to grow. The configurations it skips count for nothing.
    ///
    /// # Errors
    ///
    /// The bytes are not an encoded definition.
    pub fn for_placement_bound(bytes: &[u8]) -> Result<usize, rmp_serde::decode::Error> {
        let view: PlacementSizeView = rmp_serde::from_slice(bytes)?;
        let d = view.descriptor;
        let strings = d.name.0
            + d.description.0
            + d.label.0
            + d.properties.bytes
            + d.filter.0
            + d.state.0
            + d.owner.0;
        Ok(strings
            + LIST_ROOM
                * (d.properties.names * core::mem::size_of::<String>()
                    + view.selection.0 * core::mem::size_of::<SelectionPart>()))
    }

    /// The key scopes holding the entries of `label` under `interpretation`:
    /// the index's own, then each part that interpretation reads. Disjoint
    /// by construction, so a node is in at most one of them, and the label's
    /// members are their union. In the steady state, the index's own alone.
    pub fn eligible_scopes(
        &self,
        label: coordinode_core::schema::label_id::LabelId,
        interpretation: Interpretation,
    ) -> smallvec::SmallVec<[GenerationId; 2]> {
        let mut scopes = smallvec::smallvec![self.generation];
        for part in self.selection.iter().filter(|p| p.label == label) {
            let candidate = interpretation == Interpretation::Candidate(part.operation);
            let read = match (part.state, part.role) {
                (PartState::Proposed, PartRole::Retired) => !candidate,
                (PartState::Proposed, PartRole::Gained) => candidate,
                (PartState::InForce, PartRole::Gained) => true,
                (PartState::InForce, PartRole::Retired) => false,
                (PartState::Withdrawn, PartRole::Retired) => true,
                (PartState::Withdrawn, PartRole::Gained) => false,
            };
            if read {
                scopes.push(part.part);
            }
        }
        scopes
    }

    /// Every key scope an entry of `label` may be in: the index's own and
    /// each of its parts. What a writer removing a membership clears, since
    /// an entry moves between them as a change is certified and folded.
    pub fn scopes_of(
        &self,
        label: coordinode_core::schema::label_id::LabelId,
    ) -> smallvec::SmallVec<[GenerationId; 2]> {
        let mut scopes = smallvec::smallvec![self.generation];
        scopes.extend(
            self.selection
                .iter()
                .filter(|p| p.label == label)
                .map(|p| p.part),
        );
        scopes
    }

    /// The certification being proved for `label`, if one is: while it is,
    /// a writer holds the label to both interpretations.
    pub fn proposed_operation(
        &self,
        label: coordinode_core::schema::label_id::LabelId,
    ) -> Option<GenerationId> {
        self.selection
            .iter()
            .find(|p| p.label == label && p.state == PartState::Proposed)
            .map(|p| p.operation)
    }

    /// Whether some label has a change being certified: what a writer checks
    /// once before it judges any membership twice.
    pub fn has_proposed(&self) -> bool {
        self.selection
            .iter()
            .any(|p| p.state == PartState::Proposed)
    }

    /// Where the entry of a member of `label` is written: in the index's own
    /// entries while both interpretations hold it (`in_force` and
    /// `candidate`), in the part of the interpretation that alone does
    /// otherwise; `None` for a node neither holds. A label without a change
    /// being certified is judged by `in_force` alone.
    pub fn placement(
        &self,
        label: coordinode_core::schema::label_id::LabelId,
        in_force: bool,
        candidate: bool,
    ) -> Option<GenerationId> {
        let role = match (in_force, candidate) {
            (false, false) => return None,
            (true, true) => return Some(self.generation),
            (false, true) => PartRole::Gained,
            (true, false) => PartRole::Retired,
        };
        let proposed = self
            .selection
            .iter()
            .find(|p| p.label == label && p.state == PartState::Proposed && p.role == role);
        match (proposed, in_force) {
            (Some(part), _) => Some(part.part),
            // No change being certified: the facts in force alone decide.
            (None, true) => Some(self.generation),
            (None, false) => None,
        }
    }

    /// Record that the certification `operation` keeps `label`'s gained and
    /// retired entries in `gained` and `retired`, proposed.
    pub fn propose_parts(
        &mut self,
        label: coordinode_core::schema::label_id::LabelId,
        operation: GenerationId,
        gained: GenerationId,
        retired: GenerationId,
    ) {
        for (part, role) in [(gained, PartRole::Gained), (retired, PartRole::Retired)] {
            self.selection.push(SelectionPart {
                label,
                part,
                operation,
                role,
                state: PartState::Proposed,
            });
        }
    }

    /// Move every part of `operation` to `state`. Returns whether it had any.
    pub fn conclude_parts(&mut self, operation: GenerationId, state: PartState) -> bool {
        let mut any = false;
        for part in self
            .selection
            .iter_mut()
            .filter(|p| p.operation == operation)
        {
            part.state = state;
            any = true;
        }
        any
    }

    /// Drop the part `part` once its entries are folded or no longer read.
    pub fn drop_part(&mut self, part: GenerationId) {
        self.selection.retain(|p| p.part != part);
    }

    /// This index with its entries keyed under `scope` (its own generation
    /// or one of its parts): what the entry codec, the unique checks and the
    /// readers of one scope take. Its parts are not carried over.
    pub fn in_scope(&self, scope: GenerationId) -> Self {
        Self {
            id: self.id,
            generation: scope,
            descriptor: self.descriptor.clone(),
            selection: Vec::new(),
        }
    }

    /// Whether some part of this index is kept apart for `label`.
    pub fn has_parts_for(&self, label: coordinode_core::schema::label_id::LabelId) -> bool {
        self.selection.iter().any(|p| p.label == label)
    }

    /// The heap bytes [`Self::interpretation`] allocates, counted before it
    /// is built: a reference per property and its name, and the filter's
    /// name and string.
    pub fn interpretation_bytes(&self) -> usize {
        use coordinode_core::index::derive::PropertyRef;
        let filter = self.filter.as_ref().map_or(0, |f| match f {
            PartialFilter::PropertyEquals { property, value } => property.len() + value.len(),
            PartialFilter::PropertyEqualsInt { property, .. }
            | PartialFilter::PropertyEqualsBool { property, .. }
            | PartialFilter::PropertyExists { property } => property.len(),
        });
        self.properties.len() * core::mem::size_of::<PropertyRef>()
            + self.properties.iter().map(String::len).sum::<usize>()
            + filter
    }

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

    /// Schema catalog key naming the namespace's label lookup: the identity
    /// of its definition, written once, by the publication that creates it.
    /// Outside the definition and name prefixes.
    pub const LABEL_LOOKUP_KEY: &'static [u8] = b"schema:idx_labels";

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
        /// What failed it, when that is a refusal a caller tells apart
        /// without reading `reason`.
        #[serde(default)]
        cause: Option<FailureCause>,
    },
    /// Cancelled before it finished; its candidate entries are cleared.
    Cancelled,
}

/// A refusal that failed a build or a certification, kept with its record
/// so whoever reads the outcome later gets it as the statement did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureCause {
    /// The stored data holds values the index cannot hold.
    IndexEntry(coordinode_core::index::derive::EntryError),
    /// A page of the work did not fit the memory a page may take, even one
    /// node or entry at a time.
    Budget(coordinode_core::budget::BudgetStop),
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

/// The durable record of one certification: the operation proving the
/// stored nodes of the labels a publication changed against the candidate
/// definitions published with it, admitted in that publication's commit.
/// One pass over the stored nodes proves every label; each label's
/// candidate is put in force or withdrawn on its own outcome. Its identity
/// is a generation from the same allocator as index builds, so one
/// operation number names either kind of background work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelCertificationRecord {
    /// The operation's identity.
    pub generation: GenerationId,
    /// The labels whose stored nodes it proves.
    pub labels: Vec<String>,
    /// Where it stands; `Published` once it ended with every candidate that
    /// held put in force, `Failed` when none did.
    pub state: BuildState,
    /// Nodes checked so far, reported by the executor; progress, not proof.
    pub checked: u64,
    /// The labels whose candidate the stored nodes broke, each with why;
    /// their candidates are withdrawn, the others put in force.
    pub refused: Vec<(String, String)>,
}

impl LabelCertificationRecord {
    /// Prefix of every certification record in the schema catalog.
    pub const PREFIX: &'static [u8] = b"schema:certify:";

    /// A certification of `labels` admitted as `generation`, not taken.
    pub fn accepted(generation: GenerationId, labels: Vec<String>) -> Self {
        Self {
            generation,
            labels,
            state: BuildState::Accepted,
            checked: 0,
            refused: Vec::new(),
        }
    }

    /// Why the stored nodes refused `label`'s candidate, when they did.
    pub fn refusal_of(&self, label: &str) -> Option<&str> {
        self.refused
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, why)| why.as_str())
    }

    /// The catalog key of the certification `generation`.
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

/// One disagreement between a generation's entries and the records they are
/// derived from, as a check or a reader found it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mismatch {
    /// A record whose entry under `tuple` is missing, or names another node.
    Missing {
        /// The node whose record holds the value.
        node: u64,
        /// The version's `valid_from` for a temporal node.
        valid_from: Option<i64>,
        /// The encoded value tuple.
        tuple: Vec<u8>,
    },
    /// An entry under `tuple` naming `node`, whose record does not hold the
    /// value (or that has no record).
    Extra {
        /// The node the entry names.
        node: u64,
        /// The version's `valid_from` for an entry of a temporal version.
        valid_from: Option<i64>,
        /// The encoded value tuple.
        tuple: Vec<u8>,
    },
    /// Two stored nodes hold one value of a unique index: the records
    /// themselves break the constraint, and no entry can represent them.
    /// Reported, never resolved by changing the records.
    SourceDuplicate {
        /// The nodes holding the value, in ascending order.
        nodes: [u64; 2],
        /// The encoded value tuple.
        tuple: Vec<u8>,
    },
}

/// What is known about whether a generation's entries agree with their
/// records. Separate from its build state: a ready generation may be
/// suspect, and a verified one may still be building its successor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Integrity {
    /// Nothing was found wrong and no check has completed: the generation
    /// serves as its build state says.
    Unchecked,
    /// An entry was found disagreeing with its record: the generation
    /// proves no lookup complete and no unique value free until a check
    /// verifies it.
    Suspect,
    /// A complete check found every entry and every record in agreement,
    /// after its last repair, and no evidence arrived while it ran.
    Verified,
}

/// Where one check of a generation stands. Every move is a catalog commit
/// conditioned on the record the mover read, as a build's moves are.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckState {
    /// Admitted; no executor has taken it.
    Accepted,
    /// Taken by the executor `executor`, which alone may move it on.
    Running {
        /// The executor's token, fresh for every take.
        executor: u64,
    },
    /// The check ran to its end; the record's integrity says what it found.
    Done,
    /// The check could not finish; `reason` says why.
    Failed {
        /// Why it failed.
        reason: String,
    },
    /// Cancelled before it finished.
    Cancelled,
}

impl CheckState {
    /// Whether the check has an outcome and no executor will touch it again.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done | Self::Failed { .. } | Self::Cancelled)
    }
}

/// One check of a generation: its state, its passes, and what it found and
/// repaired. Progress is not proof; only [`Integrity::Verified`] is. A pass
/// reads the records and the entries whole at one snapshot, so a check
/// interrupted mid-pass starts that pass again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexCheck {
    /// Where the check stands.
    pub state: CheckState,
    /// The evidence revision when the check was taken: evidence arriving
    /// after it keeps the generation suspect whatever the check finds.
    pub started_revision: u64,
    /// Complete passes over the records and the entries so far.
    pub passes: u32,
    /// Records and entries checked so far.
    pub checked: u64,
    /// Disagreements found so far.
    pub mismatches: u64,
    /// Disagreements repaired so far.
    pub repaired: u64,
    /// The fresh generation a rebuild fills, when the damage was too wide
    /// to repair entry by entry; the checked generation is retired by it.
    pub rebuilt_into: Option<GenerationId>,
    /// Wall-clock milliseconds when the check reached its outcome: what a
    /// periodic check measures its interval from.
    pub finished_at_ms: Option<u64>,
}

impl IndexCheck {
    /// A check admitted at evidence revision `revision`, not taken.
    pub fn accepted(revision: u64) -> Self {
        Self {
            state: CheckState::Accepted,
            started_revision: revision,
            passes: 0,
            checked: 0,
            mismatches: 0,
            repaired: 0,
            rebuilt_into: None,
            finished_at_ms: None,
        }
    }
}

/// What the catalog knows about whether one generation's entries agree with
/// their records, and the check that establishes it. Keyed by generation:
/// a rebuild's fresh generation starts with no record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexIntegrityRecord {
    /// The generation.
    pub generation: GenerationId,
    /// Its logical index.
    pub index: IndexId,
    /// What is known about it.
    pub integrity: Integrity,
    /// The disagreements reported, each once, at most
    /// [`Self::MAX_EVIDENCE`]; the count of reports goes on in
    /// `evidence_revision` past that.
    pub evidence: Vec<Mismatch>,
    /// Moves with every new report.
    pub evidence_revision: u64,
    /// The latest check, if one was admitted.
    pub check: Option<IndexCheck>,
    /// The earliest snapshot a verified generation that was once damaged
    /// answers for: a repair fixes the entries from its commit on, not the
    /// entries an older snapshot reads, so a read at an earlier timestamp
    /// answers from the records. `None` for a generation never found
    /// damaged, which answers every snapshot it covers.
    pub trusted_from: Option<u64>,
}

/// What a member's registry takes from an integrity record: whether its
/// generation is suspect or verified, and from which snapshot a verified
/// one answers. Read without the evidence the record keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrityState {
    /// The generation.
    pub generation: GenerationId,
    /// What is known about it.
    pub integrity: Integrity,
    /// See [`IndexIntegrityRecord::trusted_from`].
    pub trusted_from: Option<u64>,
}

/// [`IndexIntegrityRecord`] as encoded, every field checked as the record's
/// own decode checks it, but its evidence and check built from nothing:
/// what the record refuses is refused here too, and nothing is allocated.
#[derive(Deserialize)]
struct IntegrityStateView {
    generation: GenerationId,
    #[serde(rename = "index")]
    _index: IndexId,
    integrity: Integrity,
    #[serde(rename = "evidence")]
    _evidence: Checked<MismatchShape>,
    #[serde(rename = "evidence_revision")]
    _evidence_revision: u64,
    #[serde(rename = "check")]
    _check: Option<IndexCheckShape>,
    trusted_from: Option<u64>,
}

/// [`Mismatch`] as its decode checks it, its tuple read and dropped.
#[derive(Deserialize)]
#[allow(dead_code)]
enum MismatchShape {
    Missing {
        node: u64,
        valid_from: Option<i64>,
        tuple: Checked<u8>,
    },
    Extra {
        node: u64,
        valid_from: Option<i64>,
        tuple: Checked<u8>,
    },
    SourceDuplicate {
        nodes: [u64; 2],
        tuple: Checked<u8>,
    },
}

/// [`IndexCheck`] as its decode checks it, nothing of it kept.
#[derive(Deserialize)]
#[allow(dead_code)]
struct IndexCheckShape {
    state: CheckStateShape,
    started_revision: u64,
    passes: u32,
    checked: u64,
    mismatches: u64,
    repaired: u64,
    rebuilt_into: Option<GenerationId>,
    finished_at_ms: Option<u64>,
}

/// [`CheckState`] as its decode checks it, a failure's reason read and
/// dropped.
#[derive(Deserialize)]
#[allow(dead_code)]
enum CheckStateShape {
    Accepted,
    Running { executor: u64 },
    Done,
    Failed { reason: CheckedStr },
    Cancelled,
}

/// A list whose every element decodes as `T`, as a `Vec<T>` would, with
/// nothing kept.
struct Checked<T>(core::marker::PhantomData<T>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Checked<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Each<T>(core::marker::PhantomData<T>);
        impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for Each<T> {
            type Value = Checked<T>;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a sequence")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Checked<T>, A::Error> {
                while seq.next_element::<T>()?.is_some() {}
                Ok(Checked(core::marker::PhantomData))
            }
        }
        deserializer.deserialize_seq(Each(core::marker::PhantomData))
    }
}

/// A string, as a `String` decodes it, with nothing kept: text, or bytes
/// that are UTF-8.
struct CheckedStr;

impl<'de> Deserialize<'de> for CheckedStr {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Text;
        impl serde::de::Visitor<'_> for Text {
            type Value = CheckedStr;
            fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str("a string")
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<CheckedStr, E> {
                Ok(CheckedStr)
            }
            fn visit_bytes<E: serde::de::Error>(self, bytes: &[u8]) -> Result<CheckedStr, E> {
                match core::str::from_utf8(bytes) {
                    Ok(_) => Ok(CheckedStr),
                    Err(_) => Err(E::invalid_value(serde::de::Unexpected::Bytes(bytes), &self)),
                }
            }
        }
        deserializer.deserialize_string(Text)
    }
}

impl IndexIntegrityRecord {
    /// What a member's registry takes from this record.
    pub fn state(&self) -> IntegrityState {
        IntegrityState {
            generation: self.generation,
            integrity: self.integrity,
            trusted_from: self.trusted_from,
        }
    }

    /// [`Self::state`] of the record encoded in `bytes`, its evidence and
    /// check skipped unread: nothing is allocated.
    ///
    /// # Errors
    ///
    /// The bytes are not an encoded record.
    pub fn state_from_msgpack(bytes: &[u8]) -> Result<IntegrityState, rmp_serde::decode::Error> {
        let view: IntegrityStateView = rmp_serde::from_slice(bytes)?;
        Ok(IntegrityState {
            generation: view.generation,
            integrity: view.integrity,
            trusted_from: view.trusted_from,
        })
    }

    /// Prefix of every integrity record in the schema catalog. Outside the
    /// definition and name prefixes.
    pub const PREFIX: &'static [u8] = b"schema:idxcheck:";

    /// The most distinct disagreements a record keeps.
    pub const MAX_EVIDENCE: usize = 64;

    /// A record of `generation` of `index` with nothing reported.
    pub fn new(index: IndexId, generation: GenerationId) -> Self {
        Self {
            generation,
            index,
            integrity: Integrity::Unchecked,
            evidence: Vec::new(),
            evidence_revision: 0,
            check: None,
            trusted_from: None,
        }
    }

    /// The catalog key of the record of `generation`.
    pub fn key_of(generation: GenerationId) -> Vec<u8> {
        let mut key = Vec::with_capacity(Self::PREFIX.len() + 8);
        key.extend_from_slice(Self::PREFIX);
        key.extend_from_slice(&generation.as_raw().to_be_bytes());
        key
    }

    /// Record `found`: the generation becomes suspect and the revision
    /// moves, unless the disagreement is one already reported. Returns
    /// whether it was new. A copy of `found` is made only when the evidence
    /// keeps it.
    pub fn report(&mut self, found: &Mismatch) -> bool {
        if self.evidence.contains(found) {
            return false;
        }
        if self.evidence.len() < Self::MAX_EVIDENCE {
            self.evidence.push(found.clone());
        }
        self.integrity = Integrity::Suspect;
        // A count of reports, bounded by the evidence a deployment can
        // produce; it never nears the type's range.
        self.evidence_revision += 1;
        true
    }

    /// Whether a check is admitted or running.
    pub fn check_pending(&self) -> bool {
        self.check.as_ref().is_some_and(|c| !c.state.is_terminal())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
