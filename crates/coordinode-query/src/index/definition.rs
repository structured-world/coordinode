//! Index metadata definitions.

use coordinode_core::graph::types::VectorMetric;
use serde::{Deserialize, Serialize};

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
    Building { written: u64, estimated_total: u64 },
    /// Backfill complete, index reflects all matching data.
    #[default]
    Ready,
    /// Backfill aborted; readers consult policy to decide error / partial.
    Failed { reason: String },
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
    PropertyEquals { property: String, value: String },
    /// Property equals a specific integer value.
    PropertyEqualsInt { property: String, value: i64 },
    /// Property equals a specific boolean value.
    PropertyEqualsBool { property: String, value: bool },
    /// Property is not null (EXISTS).
    PropertyExists { property: String },
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
}
