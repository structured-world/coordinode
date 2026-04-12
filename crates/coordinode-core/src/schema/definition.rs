//! Graph schema: label and edge type declarations with property definitions.
//!
//! Schema is declared per label (node type) and per edge type. Every node has
//! exactly one primary label. Properties have declared types with optional
//! constraints (NOT NULL, DEFAULT, UNIQUE).
//!
//! Schema is stored in the `schema:` partition and cached in memory.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::graph::types::{Value, VectorMetric};
use crate::schema::computed::ComputedSpec;

/// A property definition within a label or edge type schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PropertyDef {
    /// Property name (for display; storage uses interned field ID).
    pub name: String,

    /// The declared type of this property.
    pub property_type: PropertyType,

    /// Whether this property is required (NOT NULL).
    pub not_null: bool,

    /// Default value (if any). Applied when the property is missing on read.
    pub default: Option<Value>,

    /// Whether this property has a UNIQUE constraint.
    pub unique: bool,
}

impl PropertyDef {
    /// Create a simple property definition with no constraints.
    pub fn new(name: impl Into<String>, property_type: PropertyType) -> Self {
        Self {
            name: name.into(),
            property_type,
            not_null: false,
            default: None,
            unique: false,
        }
    }

    /// Set NOT NULL constraint.
    pub fn not_null(mut self) -> Self {
        self.not_null = true;
        self
    }

    /// Set a default value.
    pub fn with_default(mut self, value: Value) -> Self {
        self.default = Some(value);
        self
    }

    /// Set UNIQUE constraint.
    pub fn unique(mut self) -> Self {
        self.unique = true;
        self
    }

    /// Whether this property is a COMPUTED (read-only, evaluated at query time).
    pub fn is_computed(&self) -> bool {
        matches!(self.property_type, PropertyType::Computed(_))
    }

    /// Create a COMPUTED property definition.
    ///
    /// COMPUTED properties are always read-only and cannot have NOT NULL or UNIQUE.
    pub fn computed(name: impl Into<String>, spec: ComputedSpec) -> Self {
        Self {
            name: name.into(),
            property_type: PropertyType::Computed(spec),
            not_null: false,
            default: None,
            unique: false,
        }
    }
}

/// The declared type of a property.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropertyType {
    String,
    Int,
    Float,
    Bool,
    Timestamp,
    Vector {
        dimensions: u32,
        metric: VectorMetric,
    },
    Blob,
    Array(Box<PropertyType>),
    Map,
    Geo,
    Binary,
    /// Arbitrary nested document (rmpv::Value). No type validation.
    Document,
    /// Query-time evaluated field. Stored as metadata in schema, not per-node.
    /// Read-only — SET on COMPUTED properties is rejected at write time.
    Computed(ComputedSpec),
}

impl std::fmt::Display for PropertyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String => write!(f, "STRING"),
            Self::Int => write!(f, "INT"),
            Self::Float => write!(f, "FLOAT"),
            Self::Bool => write!(f, "BOOL"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
            Self::Vector { dimensions, metric } => write!(f, "VECTOR({dimensions}, {metric:?})"),
            Self::Blob => write!(f, "BLOB"),
            Self::Array(elem) => write!(f, "ARRAY<{elem}>"),
            Self::Map => write!(f, "MAP"),
            Self::Geo => write!(f, "GEO"),
            Self::Binary => write!(f, "BINARY"),
            Self::Document => write!(f, "DOCUMENT"),
            Self::Computed(spec) => write!(f, "COMPUTED({spec:?})"),
        }
    }
}

/// Schema mode controlling the balance between type safety and flexibility.
///
/// Set per label via `ALTER LABEL SET SCHEMA mode`. Affects write-path
/// validation, field interning, and storage layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchemaMode {
    /// All properties must be declared in schema. Undeclared properties rejected.
    /// Full field name interning (80% key storage reduction).
    #[default]
    Strict,

    /// Declared properties typed and interned. Undeclared properties accepted
    /// without type validation, stored in `_extra` overflow map (string keys,
    /// no interning). +10-20% storage for undeclared properties.
    Validated,

    /// No schema declaration required. All properties stored as MessagePack
    /// with string keys (no interning). +80% storage overhead vs Strict.
    Flexible,
}

impl SchemaMode {
    /// Whether this mode rejects undeclared properties.
    pub fn rejects_unknown(&self) -> bool {
        matches!(self, Self::Strict)
    }

    /// Whether this mode interns all field names.
    pub fn full_interning(&self) -> bool {
        matches!(self, Self::Strict)
    }

    /// Whether declared properties get type validation.
    pub fn validates_declared(&self) -> bool {
        matches!(self, Self::Strict | Self::Validated)
    }
}

impl std::fmt::Display for SchemaMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Strict => write!(f, "STRICT"),
            Self::Validated => write!(f, "VALIDATED"),
            Self::Flexible => write!(f, "FLEXIBLE"),
        }
    }
}

/// Schema definition for a node label.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LabelSchema {
    /// Label name (e.g., "User", "Movie").
    pub name: String,

    /// Property definitions, ordered by field name.
    pub properties: BTreeMap<String, PropertyDef>,

    /// Schema mode controlling validation and interning behavior.
    pub mode: SchemaMode,

    /// Whether unschematized properties are rejected (legacy, use `mode` instead).
    /// Kept for backwards compatibility with existing serialized schemas.
    #[serde(default)]
    pub strict: bool,

    /// Schema version (incremented on each change).
    pub version: u64,
}

impl LabelSchema {
    /// Create a new label schema with no properties (default: Strict mode).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            properties: BTreeMap::new(),
            mode: SchemaMode::default(),
            strict: false,
            version: 1,
        }
    }

    /// Add a property definition.
    pub fn add_property(&mut self, prop: PropertyDef) {
        self.properties.insert(prop.name.clone(), prop);
        self.version += 1;
    }

    /// Remove a property declaration (existing values remain on disk).
    pub fn remove_property(&mut self, name: &str) -> Option<PropertyDef> {
        let removed = self.properties.remove(name);
        if removed.is_some() {
            self.version += 1;
        }
        removed
    }

    /// Get a property definition by name.
    pub fn get_property(&self, name: &str) -> Option<&PropertyDef> {
        self.properties.get(name)
    }

    /// Set strict mode (reject unschematized properties).
    /// Legacy method — prefer `set_mode()`.
    pub fn set_strict(&mut self, strict: bool) {
        self.strict = strict;
        if strict {
            self.mode = SchemaMode::Strict;
        }
    }

    /// Set schema mode.
    pub fn set_mode(&mut self, mode: SchemaMode) {
        self.mode = mode;
        self.strict = mode.rejects_unknown();
    }

    /// Serialize to MessagePack.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Deserialize from MessagePack.
    pub fn from_msgpack(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(data)
    }
}

/// Schema definition for an edge type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EdgeTypeSchema {
    /// Edge type name (e.g., "FOLLOWS", "WORKS_AT").
    pub name: String,

    /// Property definitions for edge facets.
    pub properties: BTreeMap<String, PropertyDef>,

    /// Whether this edge type supports temporal semantics.
    pub temporal: bool,

    /// Schema version.
    pub version: u64,
}

impl EdgeTypeSchema {
    /// Create a new edge type schema.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            properties: BTreeMap::new(),
            temporal: false,
            version: 1,
        }
    }

    /// Add a property definition.
    pub fn add_property(&mut self, prop: PropertyDef) {
        self.properties.insert(prop.name.clone(), prop);
        self.version += 1;
    }

    /// Remove a property declaration.
    pub fn remove_property(&mut self, name: &str) -> Option<PropertyDef> {
        let removed = self.properties.remove(name);
        if removed.is_some() {
            self.version += 1;
        }
        removed
    }

    /// Get a property definition by name.
    pub fn get_property(&self, name: &str) -> Option<&PropertyDef> {
        self.properties.get(name)
    }

    /// Set temporal mode.
    pub fn set_temporal(&mut self, temporal: bool) {
        self.temporal = temporal;
    }

    /// Serialize to MessagePack.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Deserialize from MessagePack.
    pub fn from_msgpack(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(data)
    }
}

// -- Key encoding --

/// Encode a schema key for a label: `schema:label:<name>`.
pub fn encode_label_schema_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(13 + name.len());
    key.extend_from_slice(b"schema:label:");
    key.extend_from_slice(name.as_bytes());
    key
}

/// Encode a schema key for an edge type: `schema:edge_type:<name>`.
pub fn encode_edge_type_schema_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(17 + name.len());
    key.extend_from_slice(b"schema:edge_type:");
    key.extend_from_slice(name.as_bytes());
    key
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn property_def_builder() {
        let prop = PropertyDef::new("email", PropertyType::String)
            .not_null()
            .unique()
            .with_default(Value::String("unknown@example.com".into()));

        assert_eq!(prop.name, "email");
        assert!(matches!(prop.property_type, PropertyType::String));
        assert!(prop.not_null);
        assert!(prop.unique);
        assert!(prop.default.is_some());
    }

    #[test]
    fn property_type_display() {
        assert_eq!(PropertyType::String.to_string(), "STRING");
        assert_eq!(PropertyType::Int.to_string(), "INT");
        assert_eq!(
            PropertyType::Vector {
                dimensions: 384,
                metric: VectorMetric::Cosine
            }
            .to_string(),
            "VECTOR(384, Cosine)"
        );
        assert_eq!(
            PropertyType::Array(Box::new(PropertyType::String)).to_string(),
            "ARRAY<STRING>"
        );
    }

    #[test]
    fn label_schema_create() {
        let schema = LabelSchema::new("User");
        assert_eq!(schema.name, "User");
        assert!(schema.properties.is_empty());
        assert!(!schema.strict);
        assert_eq!(schema.version, 1);
    }

    #[test]
    fn label_schema_add_properties() {
        let mut schema = LabelSchema::new("User");
        schema.add_property(PropertyDef::new("name", PropertyType::String).not_null());
        schema.add_property(PropertyDef::new("age", PropertyType::Int));

        assert_eq!(schema.properties.len(), 2);
        assert_eq!(schema.version, 3); // 1 initial + 2 adds
        assert!(schema.get_property("name").is_some());
        assert!(schema.get_property("name").is_some_and(|p| p.not_null));
    }

    #[test]
    fn label_schema_remove_property() {
        let mut schema = LabelSchema::new("User");
        schema.add_property(PropertyDef::new("name", PropertyType::String));
        schema.add_property(PropertyDef::new("age", PropertyType::Int));

        let removed = schema.remove_property("age");
        assert!(removed.is_some());
        assert_eq!(schema.properties.len(), 1);
        assert!(schema.get_property("age").is_none());

        // Removing non-existent doesn't increment version
        let v_before = schema.version;
        assert!(schema.remove_property("nonexistent").is_none());
        assert_eq!(schema.version, v_before);
    }

    #[test]
    fn label_schema_msgpack_roundtrip() {
        let mut schema = LabelSchema::new("Movie");
        schema.add_property(PropertyDef::new("title", PropertyType::String).not_null());
        schema.add_property(PropertyDef::new(
            "embedding",
            PropertyType::Vector {
                dimensions: 384,
                metric: VectorMetric::Cosine,
            },
        ));
        schema.add_property(PropertyDef::new(
            "tags",
            PropertyType::Array(Box::new(PropertyType::String)),
        ));
        schema.set_strict(true);

        let bytes = schema.to_msgpack().expect("serialize");
        let restored = LabelSchema::from_msgpack(&bytes).expect("deserialize");
        assert_eq!(schema, restored);
    }

    #[test]
    fn edge_type_schema_create() {
        let schema = EdgeTypeSchema::new("FOLLOWS");
        assert_eq!(schema.name, "FOLLOWS");
        assert!(!schema.temporal);
        assert_eq!(schema.version, 1);
    }

    #[test]
    fn edge_type_schema_temporal() {
        let mut schema = EdgeTypeSchema::new("WORKS_AT");
        schema.set_temporal(true);
        schema.add_property(PropertyDef::new("valid_from", PropertyType::Timestamp).not_null());
        schema.add_property(PropertyDef::new("valid_to", PropertyType::Timestamp));
        schema.add_property(PropertyDef::new("role", PropertyType::String));

        assert!(schema.temporal);
        assert_eq!(schema.properties.len(), 3);
    }

    #[test]
    fn edge_type_schema_msgpack_roundtrip() {
        let mut schema = EdgeTypeSchema::new("KNOWS");
        schema.add_property(
            PropertyDef::new("since", PropertyType::Timestamp)
                .not_null()
                .with_default(Value::Timestamp(0)),
        );
        schema.add_property(
            PropertyDef::new("weight", PropertyType::Float).with_default(Value::Float(1.0)),
        );

        let bytes = schema.to_msgpack().expect("serialize");
        let restored = EdgeTypeSchema::from_msgpack(&bytes).expect("deserialize");
        assert_eq!(schema, restored);
    }

    #[test]
    fn label_schema_key_encoding() {
        let key = encode_label_schema_key("User");
        assert_eq!(&key, b"schema:label:User");
    }

    #[test]
    fn edge_type_schema_key_encoding() {
        let key = encode_edge_type_schema_key("FOLLOWS");
        assert_eq!(&key, b"schema:edge_type:FOLLOWS");
    }

    #[test]
    fn schema_keys_sort_alphabetically() {
        let k1 = encode_label_schema_key("Actor");
        let k2 = encode_label_schema_key("User");
        assert!(k1 < k2);
    }

    #[test]
    fn property_with_default_value() {
        let prop = PropertyDef::new("status", PropertyType::String)
            .with_default(Value::String("active".into()));
        assert_eq!(prop.default, Some(Value::String("active".into())));
    }

    #[test]
    fn schema_version_increments() {
        let mut schema = LabelSchema::new("Test");
        assert_eq!(schema.version, 1);
        schema.add_property(PropertyDef::new("a", PropertyType::Int));
        assert_eq!(schema.version, 2);
        schema.add_property(PropertyDef::new("b", PropertyType::String));
        assert_eq!(schema.version, 3);
        schema.remove_property("a");
        assert_eq!(schema.version, 4);
    }

    #[test]
    fn schema_mode_default_is_strict() {
        assert_eq!(SchemaMode::default(), SchemaMode::Strict);
        let schema = LabelSchema::new("Test");
        assert_eq!(schema.mode, SchemaMode::Strict);
    }

    #[test]
    fn schema_mode_properties() {
        assert!(SchemaMode::Strict.rejects_unknown());
        assert!(SchemaMode::Strict.full_interning());
        assert!(SchemaMode::Strict.validates_declared());

        assert!(!SchemaMode::Validated.rejects_unknown());
        assert!(!SchemaMode::Validated.full_interning());
        assert!(SchemaMode::Validated.validates_declared());

        assert!(!SchemaMode::Flexible.rejects_unknown());
        assert!(!SchemaMode::Flexible.full_interning());
        assert!(!SchemaMode::Flexible.validates_declared());
    }

    #[test]
    fn schema_mode_display() {
        assert_eq!(SchemaMode::Strict.to_string(), "STRICT");
        assert_eq!(SchemaMode::Validated.to_string(), "VALIDATED");
        assert_eq!(SchemaMode::Flexible.to_string(), "FLEXIBLE");
    }

    #[test]
    fn set_mode() {
        let mut schema = LabelSchema::new("Test");
        assert_eq!(schema.mode, SchemaMode::Strict);

        schema.set_mode(SchemaMode::Validated);
        assert_eq!(schema.mode, SchemaMode::Validated);
        assert!(!schema.strict);

        schema.set_mode(SchemaMode::Strict);
        assert_eq!(schema.mode, SchemaMode::Strict);
        assert!(schema.strict);
    }

    #[test]
    fn schema_mode_msgpack_roundtrip() {
        let mut schema = LabelSchema::new("Flexible");
        schema.set_mode(SchemaMode::Flexible);
        schema.add_property(PropertyDef::new("name", PropertyType::String));

        let bytes = schema.to_msgpack().expect("serialize");
        let restored = LabelSchema::from_msgpack(&bytes).expect("deserialize");
        assert_eq!(restored.mode, SchemaMode::Flexible);
        assert_eq!(restored.name, "Flexible");
    }

    // ── COMPUTED properties (R081) ───────────────────────────────

    #[test]
    fn computed_property_def() {
        use crate::schema::computed::{ComputedSpec, DecayFormula};

        let prop = PropertyDef::computed(
            "relevance",
            ComputedSpec::Decay {
                formula: DecayFormula::Linear,
                initial: 1.0,
                target: 0.0,
                duration_secs: 604800,
                anchor_field: "created_at".into(),
            },
        );
        assert!(prop.is_computed());
        assert!(!prop.not_null);
        assert!(!prop.unique);
        assert!(prop.default.is_none());
    }

    #[test]
    fn computed_property_display() {
        use crate::schema::computed::{ComputedSpec, DecayFormula};

        let pt = PropertyType::Computed(ComputedSpec::Decay {
            formula: DecayFormula::Exponential { lambda: 0.693 },
            initial: 1.0,
            target: 0.0,
            duration_secs: 86400,
            anchor_field: "created_at".into(),
        });
        let s = format!("{pt}");
        assert!(s.starts_with("COMPUTED("));
    }

    #[test]
    fn schema_with_computed_msgpack_roundtrip() {
        use crate::schema::computed::{ComputedSpec, DecayFormula, TtlScope};

        let mut schema = LabelSchema::new("Memory");
        schema.add_property(PropertyDef::new("content", PropertyType::String));
        schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
        schema.add_property(PropertyDef::computed(
            "relevance",
            ComputedSpec::Decay {
                formula: DecayFormula::Linear,
                initial: 1.0,
                target: 0.0,
                duration_secs: 604800,
                anchor_field: "created_at".into(),
            },
        ));
        schema.add_property(PropertyDef::computed(
            "_ttl",
            ComputedSpec::Ttl {
                duration_secs: 2592000,
                anchor_field: "created_at".into(),
                scope: TtlScope::Node,
                target_field: None,
            },
        ));

        let bytes = schema.to_msgpack().expect("serialize");
        let restored = LabelSchema::from_msgpack(&bytes).expect("deserialize");

        assert_eq!(restored.name, "Memory");
        assert_eq!(restored.properties.len(), 4);

        let rel = restored.get_property("relevance").expect("relevance prop");
        assert!(rel.is_computed());

        let ttl = restored.get_property("_ttl").expect("_ttl prop");
        assert!(ttl.is_computed());
    }
}
