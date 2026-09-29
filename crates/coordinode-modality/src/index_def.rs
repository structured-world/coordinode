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

/// A partial index filter predicate.
///
/// Only nodes satisfying this filter are included in the index.
/// Stored as a serializable enum of common filter patterns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartialFilter {
    /// Property equals a specific string value.
    PropertyEquals {
        /// Property name to test.
        property: String,
        /// String value the property must equal.
        value: String,
    },
    /// Property equals a specific integer value.
    PropertyEqualsInt {
        /// Property name to test.
        property: String,
        /// Integer value the property must equal.
        value: i64,
    },
    /// Property equals a specific boolean value.
    PropertyEqualsBool {
        /// Property name to test.
        property: String,
        /// Boolean value the property must equal.
        value: bool,
    },
    /// Property is not null (EXISTS).
    PropertyExists {
        /// Property name that must be present and non-null.
        property: String,
    },
}

impl PartialFilter {
    /// The property the filter tests.
    pub fn property(&self) -> &str {
        match self {
            Self::PropertyEquals { property, .. }
            | Self::PropertyEqualsInt { property, .. }
            | Self::PropertyEqualsBool { property, .. }
            | Self::PropertyExists { property } => property,
        }
    }

    /// Evaluate the filter against a set of property values.
    pub fn matches(&self, properties: &[(String, coordinode_core::graph::types::Value)]) -> bool {
        match self {
            Self::PropertyEquals { property, value } => properties
                .iter()
                .any(|(k, v)| k == property && v.as_str() == Some(value.as_str())),
            Self::PropertyEqualsInt { property, value } => properties
                .iter()
                .any(|(k, v)| k == property && v.as_int() == Some(*value)),
            Self::PropertyEqualsBool { property, value } => properties
                .iter()
                .any(|(k, v)| k == property && v.as_bool() == Some(*value)),
            Self::PropertyExists { property } => properties
                .iter()
                .any(|(k, v)| k == property && !v.is_null()),
        }
    }
}

/// Definition of an index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDefinition {
    /// Index name (unique per database).
    pub name: String,
    /// Node label this index applies to (e.g., "User").
    pub label: String,
    /// Indexed property names. Single-field: 1 entry. Compound: 2+ entries.
    /// Order matters for compound indexes — the key is encoded in this order.
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
    ///
    /// Note: do NOT mark `skip_serializing_if` here — the struct uses
    /// rmp-serde's default positional encoding, so a skipped field would
    /// shift every following field's position on decode and corrupt the
    /// roundtrip.
    #[serde(default)]
    pub text_config: Option<TextIndexConfig>,
    /// Build state. Defaults to `Ready` when deserializing pre-state schema records.
    #[serde(default)]
    pub state: IndexState,
    /// Reader behaviour during `IndexState::Building`. Defaults to
    /// [`OnlineDuringBuild::Block`] for backward compatibility.
    #[serde(default)]
    pub online_during_build: OnlineDuringBuild,
    /// Key layout the index's entries are written in. A definition stored
    /// before the field existed decodes as `0`, the layout whose entries were
    /// written outside the transaction; such an index is rebuilt in
    /// [`ENTRY_LAYOUT`] before it is used.
    #[serde(default)]
    pub layout: u32,
    /// How a key-shaped index's entries reach every member, and under which
    /// policy epoch. A definition stored before the field existed decodes as
    /// RESOLVED, inherited, epoch 0: what its entries have always been.
    #[serde(default)]
    pub maintenance: IndexMaintenance,
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

impl IndexDefinition {
    /// Create a new single-field B-tree index.
    pub fn btree(
        name: impl Into<String>,
        label: impl Into<String>,
        property: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            properties: vec![property.into()],
            index_type: IndexType::BTree,
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
        }
    }

    /// Create a compound B-tree index on multiple properties.
    pub fn compound(
        name: impl Into<String>,
        label: impl Into<String>,
        properties: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            properties,
            index_type: IndexType::BTree,
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
        }
    }

    /// Create a new HNSW vector index on a single property.
    pub fn hnsw(
        name: impl Into<String>,
        label: impl Into<String>,
        property: impl Into<String>,
        config: VectorIndexConfig,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            properties: vec![property.into()],
            index_type: IndexType::Hnsw,
            unique: false,
            sparse: true, // skip nodes without the vector property
            multikey: false,
            filter: None,
            ttl_seconds: None,
            vector_config: Some(config),
            text_config: None,
            state: IndexState::Ready,
            online_during_build: OnlineDuringBuild::Block,
            layout: ENTRY_LAYOUT,
            maintenance: IndexMaintenance::default(),
        }
    }

    /// Create a new full-text search index on one or more properties.
    pub fn text(
        name: impl Into<String>,
        label: impl Into<String>,
        properties: Vec<String>,
        config: TextIndexConfig,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            properties,
            index_type: IndexType::Text,
            unique: false,
            sparse: true,
            multikey: false,
            filter: None,
            ttl_seconds: None,
            vector_config: None,
            text_config: Some(config),
            state: IndexState::Ready,
            online_during_build: OnlineDuringBuild::Block,
            layout: ENTRY_LAYOUT,
            maintenance: IndexMaintenance::default(),
        }
    }

    /// Set unique constraint.
    pub fn unique(mut self) -> Self {
        self.unique = true;
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
            name: self.name.clone(),
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

    /// Whether this is a compound index (2+ properties).
    pub fn is_compound(&self) -> bool {
        self.properties.len() > 1
    }

    /// First (or only) property name. For backwards compatibility.
    pub fn property(&self) -> &str {
        self.properties.first().map_or("", |s| s.as_str())
    }

    /// Schema storage key for this index definition.
    pub fn schema_key(&self) -> Vec<u8> {
        let mut key = Vec::with_capacity(10 + self.name.len());
        key.extend_from_slice(b"schema:idx:");
        key.extend_from_slice(self.name.as_bytes());
        key
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
