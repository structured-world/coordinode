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
    /// Reader waits (up to a deadline) for the index to reach `Ready`
    /// before proceeding. Matches the legacy synchronous semantic.
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
}

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

/// Per-label shard-routing strategy for a vector index: how the label's
/// vectors are distributed across shards and how a query routes to them.
///
/// `Unsharded` (the default) is the single-index-per-label model. The other
/// variants split a label across `n_shards` and are rebuilt asynchronously
/// when the strategy changes, so a label can be re-sharded online without
/// disrupting serving.
///
/// Thresholds are stored in permille of a distance ratio (1000 = 1.0x) to keep
/// the type `Eq` (no floats); the router scales them back to `f32` at use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ShardStrategy {
    /// One HNSW for the whole label on a single shard (today's model).
    #[default]
    Unsharded,
    /// Centroid (k-means) partitioning with SPANN-style closure replication and
    /// query-adaptive fan-out. Boundary vectors are replicated to nearby shards
    /// so a low fan-out still recovers cross-boundary neighbours.
    Centroid {
        /// Number of shards to split the label across.
        n_shards: u32,
        /// Closure replication threshold, permille of the nearest-centroid
        /// distance: a vector also lands on every centroid within this factor.
        /// `1000` (1.0x) replicates only exact ties (no replication).
        replication_eps_permille: u32,
        /// Cap on closure replicas per vector (SPANN bounds this at ~8).
        max_replicas: u32,
        /// Query-adaptive fan-out threshold, permille of the nearest-centroid
        /// distance. `1000` (1.0x) routes to the nearest shard only; larger
        /// fans out only for queries near a cluster boundary.
        route_eps_permille: u32,
    },
    /// Hash-bucket partitioning: geometry-blind round-robin, every query
    /// scatters to all shards. Simple, balanced, but no routing savings.
    Hash {
        /// Number of shards.
        n_shards: u32,
    },
}

impl ShardStrategy {
    /// Number of shards this strategy spreads the label across (`1` for
    /// `Unsharded`).
    #[must_use]
    pub fn n_shards(&self) -> u32 {
        match self {
            Self::Unsharded => 1,
            Self::Centroid { n_shards, .. } | Self::Hash { n_shards } => (*n_shards).max(1),
        }
    }

    /// Whether the label is split across more than one shard.
    #[must_use]
    pub fn is_sharded(&self) -> bool {
        self.n_shards() > 1
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
    /// How the label's vectors are sharded + routed. `Unsharded` (default) is
    /// the single-index-per-label model. Changing this re-shards the label
    /// online (async rebuild, no serving disruption). MUST stay the last field:
    /// `IndexDefinition` uses rmp-serde positional encoding, so the
    /// `#[serde(default)]` only round-trips old records correctly at the tail.
    #[serde(default)]
    pub shard_strategy: ShardStrategy,
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
            shard_strategy: ShardStrategy::Unsharded,
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

    /// Set TTL expiration in seconds.
    pub fn with_ttl(mut self, seconds: u64) -> Self {
        self.ttl_seconds = Some(seconds);
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

    /// First (or only) property name. For backwards compatibility.
    pub fn property(&self) -> &str {
        self.properties.first().map_or("", |s| s.as_str())
    }

    /// Storage key prefix for this index: `idx:<name>:`.
    pub fn key_prefix(&self) -> Vec<u8> {
        let mut prefix = Vec::with_capacity(4 + self.name.len() + 1);
        prefix.extend_from_slice(b"idx:");
        prefix.extend_from_slice(self.name.as_bytes());
        prefix.push(b':');
        prefix
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
mod tests {
    use super::*;

    #[test]
    fn btree_definition() {
        let idx = IndexDefinition::btree("user_email", "User", "email").unique();
        assert_eq!(idx.name, "user_email");
        assert_eq!(idx.label, "User");
        assert_eq!(idx.property(), "email");
        assert_eq!(idx.properties, vec!["email"]);
        assert!(idx.unique);
        assert!(!idx.sparse);
        assert!(!idx.multikey);
        assert!(!idx.is_compound());
    }

    #[test]
    fn compound_definition() {
        let idx = IndexDefinition::compound(
            "user_label_status",
            "User",
            vec!["label".into(), "status".into()],
        );
        assert!(idx.is_compound());
        assert_eq!(idx.properties.len(), 2);
        assert_eq!(idx.property(), "label");
    }

    #[test]
    fn sparse_definition() {
        let idx = IndexDefinition::btree("user_bio", "User", "bio").sparse();
        assert!(idx.sparse);
    }

    #[test]
    fn key_prefix() {
        let idx = IndexDefinition::btree("user_email", "User", "email");
        assert_eq!(idx.key_prefix(), b"idx:user_email:");
    }

    #[test]
    fn schema_key() {
        let idx = IndexDefinition::btree("user_email", "User", "email");
        assert_eq!(idx.schema_key(), b"schema:idx:user_email");
    }

    #[test]
    fn new_indexes_default_to_ready_state() {
        let btree = IndexDefinition::btree("u_email", "User", "email");
        let hnsw = IndexDefinition::hnsw("u_vec", "User", "vec", VectorIndexConfig::default());
        let compound = IndexDefinition::compound(
            "u_lbl_status",
            "User",
            vec!["label".into(), "status".into()],
        );
        let text = IndexDefinition::text(
            "u_text",
            "User",
            vec!["bio".into()],
            TextIndexConfig::default(),
        );
        assert_eq!(btree.state, IndexState::Ready);
        assert_eq!(hnsw.state, IndexState::Ready);
        assert_eq!(compound.state, IndexState::Ready);
        assert_eq!(text.state, IndexState::Ready);
    }

    #[test]
    fn state_roundtrip_serde() {
        let mut idx = IndexDefinition::hnsw("v", "L", "p", VectorIndexConfig::default());
        idx.state = IndexState::Building {
            written: 1234,
            estimated_total: 9999,
        };
        let bytes = rmp_serde::to_vec(&idx).expect("encode");
        let back: IndexDefinition = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(back.state, idx.state);

        idx.state = IndexState::Failed {
            reason: "build aborted".to_string(),
        };
        let bytes = rmp_serde::to_vec(&idx).expect("encode failed");
        let back: IndexDefinition = rmp_serde::from_slice(&bytes).expect("decode failed");
        assert_eq!(
            back.state,
            IndexState::Failed {
                reason: "build aborted".to_string()
            }
        );
    }

    #[test]
    fn legacy_def_without_state_deserializes_as_ready() {
        // Simulate a pre-state IndexDefinition record by encoding a struct
        // that has the same field layout MINUS the `state` field. rmp-serde
        // accepts the shorter struct because we marked `state` with
        // `#[serde(default)]`.
        #[derive(serde::Serialize)]
        struct LegacyDef {
            name: String,
            label: String,
            properties: Vec<String>,
            index_type: IndexType,
            unique: bool,
            sparse: bool,
            multikey: bool,
            filter: Option<PartialFilter>,
            ttl_seconds: Option<u64>,
            vector_config: Option<VectorIndexConfig>,
        }
        let legacy = LegacyDef {
            name: "u".into(),
            label: "U".into(),
            properties: vec!["v".into()],
            index_type: IndexType::Hnsw,
            unique: false,
            sparse: true,
            multikey: false,
            filter: None,
            ttl_seconds: None,
            vector_config: Some(VectorIndexConfig::default()),
        };
        let bytes = rmp_serde::to_vec(&legacy).expect("encode legacy");
        let back: IndexDefinition =
            rmp_serde::from_slice(&bytes).expect("decode legacy as current");
        assert_eq!(back.state, IndexState::Ready);
        assert_eq!(back.name, "u");
    }

    #[test]
    fn shard_strategy_defaults_to_unsharded() {
        assert_eq!(
            VectorIndexConfig::default().shard_strategy,
            ShardStrategy::Unsharded
        );
        assert_eq!(ShardStrategy::default(), ShardStrategy::Unsharded);
        assert!(!ShardStrategy::Unsharded.is_sharded());
        assert_eq!(ShardStrategy::Unsharded.n_shards(), 1);
    }

    #[test]
    fn shard_strategy_centroid_roundtrip_serde() {
        let strat = ShardStrategy::Centroid {
            n_shards: 4,
            replication_eps_permille: 1300,
            max_replicas: 3,
            route_eps_permille: 1500,
        };
        assert!(strat.is_sharded());
        assert_eq!(strat.n_shards(), 4);
        let cfg = VectorIndexConfig {
            dimensions: 128,
            shard_strategy: strat,
            ..Default::default()
        };
        let idx = IndexDefinition::hnsw("v", "L", "p", cfg);
        let bytes = rmp_serde::to_vec(&idx).expect("encode");
        let back: IndexDefinition = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(
            back.vector_config.expect("vector_config").shard_strategy,
            strat
        );
    }

    #[test]
    fn legacy_vector_config_without_shard_strategy_defaults_unsharded() {
        // A pre-strategy VectorIndexConfig record has the field layout MINUS
        // `shard_strategy` (the new tail field). rmp-serde accepts the shorter
        // struct because `shard_strategy` is `#[serde(default)]`, and the
        // missing value fills as Unsharded — the backward-compat contract.
        #[derive(serde::Serialize)]
        struct LegacyVectorConfig {
            dimensions: u32,
            metric: VectorMetric,
            m: usize,
            ef_construction: usize,
            quantization: coordinode_vector::hnsw::QuantizationCodec,
            offload_vectors: bool,
            ef_search: Option<usize>,
            rerank_candidates: Option<usize>,
        }
        let legacy = LegacyVectorConfig {
            dimensions: 64,
            metric: VectorMetric::Cosine,
            m: 16,
            ef_construction: 200,
            quantization: coordinode_vector::hnsw::QuantizationCodec::None,
            offload_vectors: false,
            ef_search: None,
            rerank_candidates: None,
        };
        let bytes = rmp_serde::to_vec(&legacy).expect("encode legacy vector config");
        let back: VectorIndexConfig =
            rmp_serde::from_slice(&bytes).expect("decode legacy as current");
        assert_eq!(back.shard_strategy, ShardStrategy::Unsharded);
        assert_eq!(back.dimensions, 64);
    }
}
