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

/// Placement policy controlling how nodes of a label are distributed across
/// shard groups in EE. CE single-shard deployments always use `NodeId`.
///
/// Declared explicitly at label creation — there is no default. Per ADR-023,
/// every label declares its placement strategy at creation time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlacementPolicy {
    /// Graph-style placement: `hash(NodeId) mod N_shards`. Edges co-locate
    /// with source nodes by default. Suitable for traversal-heavy workloads.
    NodeId,

    /// Document-style placement: `hash(<property_value>) mod N_shards`. Nodes
    /// sharing a shard-key value land on the same shard. Suitable for point
    /// and filtered queries on the shard-key property.
    Hash(String),

    /// Range placement: `<property_value>` partitioned into contiguous ranges,
    /// each assigned to a shard. Suitable for time-series and sorted-scan
    /// workloads.
    Range(String),
}

/// State of a shard key in a label's lifecycle.
///
/// Multi-key coexistence (ADR-023): during a lazy re-shard, a label has one
/// PRIMARY (where new writes route) plus optionally one LEGACY (where
/// pre-migration data still lives). The A-strict baseline allows at most one
/// LEGACY at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardKeyState {
    /// New writes route by this key.
    Primary,
    /// Pre-existing data still routed here; migration in progress to PRIMARY.
    Legacy,
}

/// Kind of placement function for a shard key.
///
/// Mirrors `PlacementPolicy` variants but per-shard-key in the lazy-migration
/// list, because primary and legacy keys may use different placement kinds
/// (e.g., migrating from `Hash(customer_id)` to `Range(region)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlacementKind {
    NodeId,
    Hash,
    Range,
}

/// A single entry in a label's `shard_keys` list.
///
/// The list is `[PRIMARY]` in steady state and `[PRIMARY, LEGACY]` during
/// a lazy re-shard (one migration in flight, A-strict baseline).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardKeySpec {
    /// Property name (e.g., `customer_id`). For `PlacementKind::NodeId` this
    /// is the sentinel `__node_id__`.
    pub property: String,

    /// Whether this key is the primary (new writes) or legacy (pre-migration).
    pub state: ShardKeyState,

    /// Placement function for this shard key.
    pub kind: PlacementKind,

    /// LabelSchema revision at which this key entered its current state. Used
    /// for audit and reversibility (`RESTORE_KEY(<revision>)`).
    pub since_revision: u64,
}

impl ShardKeySpec {
    /// Build the canonical "no-op" PRIMARY entry for a `NodeId` placement
    /// — the default for newly created labels in CE and for any EE label
    /// that hasn't been re-sharded.
    pub fn primary_node_id(revision: u64) -> Self {
        Self {
            property: "__node_id__".to_string(),
            state: ShardKeyState::Primary,
            kind: PlacementKind::NodeId,
            since_revision: revision,
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

    /// Placement policy for this label. Declared explicitly at creation time
    /// (no default). CE always declares `NodeId`; EE labels may declare
    /// `NodeId` (graph default), `Hash(prop)`, or `Range(prop)`.
    pub placement: PlacementPolicy,

    /// Multi-key list for lazy re-sharding (ADR-023). Always contains at least
    /// one PRIMARY entry. During a lazy migration also contains one LEGACY
    /// entry. CE labels carry `[primary_node_id(1)]` permanently.
    pub shard_keys: Vec<ShardKeySpec>,

    /// Schema snapshot revision (ADR-023). Bumped by any `ALTER LABEL`
    /// operation that changes write-path semantics: `placement`, `shard_keys`,
    /// or `mode`. Property additions or removals do NOT bump this — they are
    /// mutations of the current snapshot. The revision is the key suffix for
    /// `schema:label:<name>:<revision>` and is what `current_revision`
    /// pointer names.
    ///
    /// Distinct from per-row data versioning: MVCC `commit_ts` (HLC) tracks
    /// data revisions at write granularity; `schema_revision` tracks DDL
    /// snapshots. Renaming follows the Postgres `pg_class.relrewrite` /
    /// Kubernetes `metadata.generation` convention — "revision" = DDL,
    /// "version" = data.
    pub schema_revision: u64,
}

impl LabelSchema {
    /// Create a new label schema with the given placement policy.
    ///
    /// `placement` is mandatory — there is no default. CE callers pass
    /// `PlacementPolicy::NodeId`; EE may pass any variant.
    pub fn new(name: impl Into<String>, placement: PlacementPolicy) -> Self {
        let initial_kind = match &placement {
            PlacementPolicy::NodeId => PlacementKind::NodeId,
            PlacementPolicy::Hash(_) => PlacementKind::Hash,
            PlacementPolicy::Range(_) => PlacementKind::Range,
        };
        let initial_property = match &placement {
            PlacementPolicy::NodeId => "__node_id__".to_string(),
            PlacementPolicy::Hash(p) | PlacementPolicy::Range(p) => p.clone(),
        };
        let primary = ShardKeySpec {
            property: initial_property,
            state: ShardKeyState::Primary,
            kind: initial_kind,
            since_revision: 1,
        };
        Self {
            name: name.into(),
            properties: BTreeMap::new(),
            mode: SchemaMode::default(),
            placement,
            shard_keys: vec![primary],
            schema_revision: 1,
        }
    }

    /// Convenience for CE call sites and tests: graph-default placement.
    pub fn new_node_id(name: impl Into<String>) -> Self {
        Self::new(name, PlacementPolicy::NodeId)
    }

    /// Add a property definition. Mutates the current snapshot; does not
    /// bump `schema_revision` (revision changes only on placement/shard_keys/
    /// mode mutations per ADR-023).
    pub fn add_property(&mut self, prop: PropertyDef) {
        self.properties.insert(prop.name.clone(), prop);
    }

    /// Remove a property declaration (existing values remain on disk).
    /// Mutates the current snapshot without bumping `schema_revision`.
    pub fn remove_property(&mut self, name: &str) -> Option<PropertyDef> {
        self.properties.remove(name)
    }

    /// Get a property definition by name.
    pub fn get_property(&self, name: &str) -> Option<&PropertyDef> {
        self.properties.get(name)
    }

    /// Whether this label rejects unschematized properties (derived from
    /// `mode`, not stored separately).
    pub fn is_strict(&self) -> bool {
        matches!(self.mode, SchemaMode::Strict)
    }

    /// Set schema mode.
    pub fn set_mode(&mut self, mode: SchemaMode) {
        self.mode = mode;
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

/// Placement policy for edges relative to their endpoint nodes (ADR-023).
///
/// Determines which shard's adjacency posting carries the edge's existence
/// entry when source and target are on different shards. Only meaningful in
/// EE multi-shard deployments; CE always co-locates because there is only
/// one shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgePlacement {
    /// Adjacency entry lives at the source node's shard. Forward traversal
    /// `(a)-[:T]->(b)` is local on the source shard; reverse traversal pays
    /// one cross-shard hop to locate the source shard.
    #[default]
    ColocateWithSource,

    /// Adjacency entry lives at the target node's shard. Reverse traversal
    /// is local; forward traversal pays one cross-shard hop.
    ColocateWithTarget,

    /// Adjacency entry replicated on both shards. Both directions local;
    /// doubles adjacency storage and write amplification. Recommended only
    /// for small posting lists (< 100 entries typical).
    Replicated,
}

/// Per-doc migration state tracked in the `schema:migration_state:` namespace
/// during in-flight `ALTER LABEL SHARD BY` operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationStateEntry {
    /// Where the doc lives at the moment the entry was written.
    pub current_shard: u32,
    /// Where the doc should end up when migration completes.
    pub target_shard: u32,
    /// Lifecycle state of this doc within the migration.
    pub state: MigrationDocState,
    /// HLC timestamp at which this entry was enqueued for migration. Used by
    /// the priority queue (R210f query-driven on-touch) to prefer recently
    /// touched docs.
    pub enqueued_at: i64,
}

/// Lifecycle state of a doc within an in-flight migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationDocState {
    /// Doc still on legacy shard; not yet enqueued.
    Legacy,
    /// Doc in transit (swarm pieces shipping between shards).
    Migrating,
    /// Doc arrived on primary; entry pending cleanup.
    Migrated,
}

/// Chunk-assignment table stored under `schema:chunks:<label>` — maps key
/// ranges to shard ids for a given label (R210e routing pipeline).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkAssignmentTable {
    /// Label this table belongs to.
    pub label: String,
    /// Sorted list of `(range_start, shard_id)` entries. A key with hash `H`
    /// is routed to the shard of the largest entry whose `range_start <= H`.
    /// CE single-shard deployments carry `[(0, 1)]`.
    pub ranges: Vec<(u64, u32)>,
    /// Schema revision corresponding to the LabelSchema that produced this
    /// table; advances together with `ALTER LABEL SHARD BY` and is reversible
    /// via `RESTORE_KEY(<revision>)`.
    pub revision: u64,
}

impl ChunkAssignmentTable {
    /// Build the trivial CE table: every key routes to shard 1.
    pub fn ce_single_shard(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ranges: vec![(0, 1)],
            revision: 1,
        }
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

    /// Placement policy for adjacency entries relative to endpoint nodes
    /// (ADR-023). Default `ColocateWithSource` matches the graph-default
    /// traversal pattern; alternative values opt into target-co-location or
    /// replicated adjacency for specific workloads.
    pub placement: EdgePlacement,

    /// Schema snapshot revision. Bumped by ALTER operations affecting
    /// placement or temporal flag (mirrors LabelSchema semantics per ADR-023).
    pub schema_revision: u64,
}

impl EdgeTypeSchema {
    /// Create a new edge type schema with default `ColocateWithSource`
    /// placement and `temporal = false`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            properties: BTreeMap::new(),
            temporal: false,
            placement: EdgePlacement::default(),
            schema_revision: 1,
        }
    }

    /// Add a property definition. Mutates the current snapshot; does not
    /// bump `schema_revision` (mirrors LabelSchema semantics per ADR-023).
    pub fn add_property(&mut self, prop: PropertyDef) {
        self.properties.insert(prop.name.clone(), prop);
    }

    /// Remove a property declaration. Mutates the current snapshot without
    /// bumping `schema_revision`.
    pub fn remove_property(&mut self, name: &str) -> Option<PropertyDef> {
        self.properties.remove(name)
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

/// Encode a revisioned schema key for a label:
/// `schema:label:<name>:<revision>`.
///
/// Per ADR-023, the schema partition is revision-prefixed from day one — each
/// `ALTER LABEL` (when shard-key mutation lands in R210c) writes a new
/// revision while older revisions remain as immutable snapshots. CE
/// deployments only ever write revision 1 (no `ALTER LABEL SHARD BY` in CE),
/// but the key format is final-state from the first commit.
///
/// "Revision" (DDL-side) is intentionally distinct from "version" (data-side,
/// e.g. MVCC `commit_ts` or future per-row OCC). See `schema_revision` field
/// docs.
///
/// Callers reading "current" schema must first read the pointer at
/// [`encode_label_current_revision_key`] to learn which revision to load.
pub fn encode_label_schema_key(name: &str, revision: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(13 + name.len() + 1 + 20);
    key.extend_from_slice(b"schema:label:");
    key.extend_from_slice(name.as_bytes());
    key.push(b':');
    key.extend_from_slice(revision.to_string().as_bytes());
    key
}

/// Encode the pointer key naming the current revision for a label:
/// `schema:current_revision:label:<name>`. Value: u64 BE.
pub fn encode_label_current_revision_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(30 + name.len());
    key.extend_from_slice(b"schema:current_revision:label:");
    key.extend_from_slice(name.as_bytes());
    key
}

/// Encode a revisioned schema key for an edge type:
/// `schema:edge_type:<name>:<revision>`. Mirrors label revisioning.
pub fn encode_edge_type_schema_key(name: &str, revision: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(17 + name.len() + 1 + 20);
    key.extend_from_slice(b"schema:edge_type:");
    key.extend_from_slice(name.as_bytes());
    key.push(b':');
    key.extend_from_slice(revision.to_string().as_bytes());
    key
}

/// Encode the current-revision pointer for an edge type:
/// `schema:current_revision:edge_type:<name>`. Value: u64 BE.
pub fn encode_edge_type_current_revision_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(34 + name.len());
    key.extend_from_slice(b"schema:current_revision:edge_type:");
    key.extend_from_slice(name.as_bytes());
    key
}

/// Encode the per-doc migration state key:
/// `schema:migration_state:<label>:<node_id u64 BE>`.
///
/// Per ADR-023, this partition (logical key namespace inside `Partition::Schema`)
/// holds in-flight migration state for `ALTER LABEL SHARD BY` operations. The
/// entry is keyed by `(label, node_id)` and the value is the
/// `MigrationStateEntry` MessagePack body. Entries are deleted as docs
/// transition to MIGRATED on the primary side; the namespace is empty in
/// steady state and empty in CE single-shard deployments.
pub fn encode_migration_state_key(label: &str, node_id: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(23 + label.len() + 1 + 8);
    key.extend_from_slice(b"schema:migration_state:");
    key.extend_from_slice(label.as_bytes());
    key.push(b':');
    key.extend_from_slice(&node_id.to_be_bytes());
    key
}

/// Encode the chunk-assignment table key for a label:
/// `schema:chunks:<label>`. Value: MessagePack `ChunkAssignmentTable`.
///
/// In CE this is a trivial single-entry table `{primary_shard: 1,
/// ranges: [(0, u64::MAX)]}`. In EE the table tracks per-chunk shard
/// assignments and is mutated by the coordinator on rebalance / move /
/// `ALTER LABEL SHARD BY` operations.
pub fn encode_chunk_assignments_key(label: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(14 + label.len());
    key.extend_from_slice(b"schema:chunks:");
    key.extend_from_slice(label.as_bytes());
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
        let schema = LabelSchema::new_node_id("User");
        assert_eq!(schema.name, "User");
        assert!(schema.properties.is_empty());
        // Default schema mode is Strict — see SchemaMode::default(). Legacy
        // code returned `false` for the now-removed `strict` field because it
        // defaulted independently of `mode`; with the field gone, `is_strict`
        // is derived from `mode` which defaults to Strict.
        assert!(schema.is_strict());
        assert_eq!(schema.schema_revision, 1);
        // CE default: NodeId placement with single PRIMARY shard key.
        assert!(matches!(schema.placement, PlacementPolicy::NodeId));
        assert_eq!(schema.shard_keys.len(), 1);
        assert_eq!(schema.shard_keys[0].state, ShardKeyState::Primary);
        assert_eq!(schema.shard_keys[0].kind, PlacementKind::NodeId);
    }

    #[test]
    fn label_schema_add_properties() {
        let mut schema = LabelSchema::new_node_id("User");
        schema.add_property(PropertyDef::new("name", PropertyType::String).not_null());
        schema.add_property(PropertyDef::new("age", PropertyType::Int));

        assert_eq!(schema.properties.len(), 2);
        // Per ADR-023, property additions mutate the current snapshot but
        // do not bump the schema revision — only `ALTER LABEL` operations
        // affecting placement/shard_keys do.
        assert_eq!(schema.schema_revision, 1);
        assert!(schema.get_property("name").is_some());
        assert!(schema.get_property("name").is_some_and(|p| p.not_null));
    }

    #[test]
    fn label_schema_remove_property() {
        let mut schema = LabelSchema::new_node_id("User");
        schema.add_property(PropertyDef::new("name", PropertyType::String));
        schema.add_property(PropertyDef::new("age", PropertyType::Int));

        let removed = schema.remove_property("age");
        assert!(removed.is_some());
        assert_eq!(schema.properties.len(), 1);
        assert!(schema.get_property("age").is_none());

        // Removing non-existent doesn't increment version
        let v_before = schema.schema_revision;
        assert!(schema.remove_property("nonexistent").is_none());
        assert_eq!(schema.schema_revision, v_before);
    }

    #[test]
    fn label_schema_msgpack_roundtrip() {
        let mut schema = LabelSchema::new_node_id("Movie");
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
        schema.set_mode(SchemaMode::Strict);

        let bytes = schema.to_msgpack().expect("serialize");
        let restored = LabelSchema::from_msgpack(&bytes).expect("deserialize");
        assert_eq!(schema, restored);
    }

    #[test]
    fn edge_type_schema_create() {
        let schema = EdgeTypeSchema::new("FOLLOWS");
        assert_eq!(schema.name, "FOLLOWS");
        assert!(!schema.temporal);
        assert_eq!(schema.schema_revision, 1);
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
        let key = encode_label_schema_key("User", 1);
        assert_eq!(&key, b"schema:label:User:1");
    }

    #[test]
    fn label_schema_key_includes_version() {
        let v1 = encode_label_schema_key("User", 1);
        let v2 = encode_label_schema_key("User", 2);
        assert_ne!(v1, v2, "different versions must produce different keys");
        assert!(
            v1 < v2,
            "version ordering preserved by string-encoded suffix"
        );
    }

    #[test]
    fn label_current_revision_pointer_key_encoding() {
        let key = encode_label_current_revision_key("User");
        assert_eq!(&key, b"schema:current_revision:label:User");
    }

    #[test]
    fn edge_type_schema_key_encoding() {
        let key = encode_edge_type_schema_key("FOLLOWS", 1);
        assert_eq!(&key, b"schema:edge_type:FOLLOWS:1");
    }

    #[test]
    fn edge_type_current_revision_pointer_key_encoding() {
        let key = encode_edge_type_current_revision_key("FOLLOWS");
        assert_eq!(&key, b"schema:current_revision:edge_type:FOLLOWS");
    }

    #[test]
    fn migration_state_key_encoding() {
        let key = encode_migration_state_key("Order", 0x42);
        let mut expected = b"schema:migration_state:Order:".to_vec();
        expected.extend_from_slice(&0x42u64.to_be_bytes());
        assert_eq!(key, expected);
    }

    #[test]
    fn migration_state_keys_sort_by_node_id_within_label() {
        let k_low = encode_migration_state_key("Order", 1);
        let k_high = encode_migration_state_key("Order", 1000);
        assert!(
            k_low < k_high,
            "BE node_id encoding preserves numeric ordering"
        );
    }

    #[test]
    fn migration_state_keys_separated_by_label() {
        let k_order = encode_migration_state_key("Order", 1);
        let k_user = encode_migration_state_key("User", 1);
        // Lexicographic separation: "Order" < "User", so prefix scan by label
        // returns docs grouped per label.
        assert!(k_order < k_user);
    }

    #[test]
    fn chunk_assignments_key_encoding() {
        let key = encode_chunk_assignments_key("Order");
        assert_eq!(&key, b"schema:chunks:Order");
    }

    #[test]
    fn ce_single_shard_chunk_table_roundtrip() {
        let table = ChunkAssignmentTable::ce_single_shard("Order");
        let bytes = table.to_msgpack().expect("encode");
        let decoded = ChunkAssignmentTable::from_msgpack(&bytes).expect("decode");
        assert_eq!(decoded.label, "Order");
        assert_eq!(decoded.ranges, vec![(0, 1)]);
        assert_eq!(decoded.revision, 1);
    }

    #[test]
    fn edge_type_schema_default_placement_is_colocate_with_source() {
        let schema = EdgeTypeSchema::new("WORKS_AT");
        assert_eq!(schema.placement, EdgePlacement::ColocateWithSource);
        assert!(!schema.temporal);
        assert_eq!(schema.schema_revision, 1);
    }

    #[test]
    fn migration_state_entry_roundtrips() {
        let entry = MigrationStateEntry {
            current_shard: 1,
            target_shard: 5,
            state: MigrationDocState::Migrating,
            enqueued_at: 1_700_000_000_000,
        };
        let bytes = rmp_serde::to_vec(&entry).expect("encode");
        let decoded: MigrationStateEntry = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(decoded, entry);
    }

    #[test]
    fn label_schema_with_hash_placement_msgpack_roundtrip() {
        // Hash placement carries a property name in the variant payload.
        // Ensure the variant + payload survive msgpack encoding without
        // collapsing to the default NodeId placement.
        let mut schema = LabelSchema::new("User", PlacementPolicy::Hash("tenant_id".to_string()));
        schema.add_property(PropertyDef::new("tenant_id", PropertyType::String).not_null());
        let bytes = schema.to_msgpack().expect("encode");
        let decoded = LabelSchema::from_msgpack(&bytes).expect("decode");
        assert!(matches!(
            decoded.placement,
            PlacementPolicy::Hash(ref p) if p == "tenant_id"
        ));
        assert_eq!(decoded.shard_keys.len(), 1);
        assert_eq!(decoded.shard_keys[0].property, "tenant_id");
        assert_eq!(decoded.shard_keys[0].kind, PlacementKind::Hash);
        assert_eq!(decoded.shard_keys[0].state, ShardKeyState::Primary);
    }

    #[test]
    fn label_schema_with_range_placement_msgpack_roundtrip() {
        let mut schema =
            LabelSchema::new("Event", PlacementPolicy::Range("occurred_at".to_string()));
        schema.add_property(PropertyDef::new("occurred_at", PropertyType::Timestamp).not_null());
        let bytes = schema.to_msgpack().expect("encode");
        let decoded = LabelSchema::from_msgpack(&bytes).expect("decode");
        assert!(matches!(
            decoded.placement,
            PlacementPolicy::Range(ref p) if p == "occurred_at"
        ));
        assert_eq!(decoded.shard_keys[0].kind, PlacementKind::Range);
    }

    #[test]
    fn edge_type_schema_with_target_colocation_msgpack_roundtrip() {
        let mut schema = EdgeTypeSchema::new("OWNED_BY");
        schema.placement = EdgePlacement::ColocateWithTarget;
        let bytes = schema.to_msgpack().expect("encode");
        let decoded = EdgeTypeSchema::from_msgpack(&bytes).expect("decode");
        assert_eq!(decoded.placement, EdgePlacement::ColocateWithTarget);
    }

    #[test]
    fn edge_type_schema_with_replicated_placement_msgpack_roundtrip() {
        let mut schema = EdgeTypeSchema::new("MENTIONS");
        schema.placement = EdgePlacement::Replicated;
        let bytes = schema.to_msgpack().expect("encode");
        let decoded = EdgeTypeSchema::from_msgpack(&bytes).expect("decode");
        assert_eq!(decoded.placement, EdgePlacement::Replicated);
    }

    #[test]
    fn migration_state_entry_legacy_and_migrated_states_roundtrip() {
        // The enum has three states; existing tests cover Migrating —
        // exercise the other two so all variants are wire-stable.
        for state in [MigrationDocState::Legacy, MigrationDocState::Migrated] {
            let entry = MigrationStateEntry {
                current_shard: 2,
                target_shard: 7,
                state,
                enqueued_at: 1_700_000_000_000,
            };
            let bytes = rmp_serde::to_vec(&entry).expect("encode");
            let decoded: MigrationStateEntry = rmp_serde::from_slice(&bytes).expect("decode");
            assert_eq!(decoded, entry);
        }
    }

    #[test]
    fn chunk_assignment_table_multi_range_roundtrip() {
        // Real EE deployments will carry multiple range/shard pairs —
        // verify multi-entry ranges survive msgpack.
        let table = ChunkAssignmentTable {
            label: "Event".to_string(),
            ranges: vec![(0, 3), (1_000_000, 5), (5_000_000, 8)],
            revision: 12,
        };
        let bytes = table.to_msgpack().expect("encode");
        let decoded = ChunkAssignmentTable::from_msgpack(&bytes).expect("decode");
        assert_eq!(decoded.label, "Event");
        assert_eq!(decoded.ranges, vec![(0, 3), (1_000_000, 5), (5_000_000, 8)]);
        assert_eq!(decoded.revision, 12);
    }

    #[test]
    fn schema_keys_sort_alphabetically() {
        let k1 = encode_label_schema_key("Actor", 1);
        let k2 = encode_label_schema_key("User", 1);
        assert!(k1 < k2);
    }

    #[test]
    fn property_with_default_value() {
        let prop = PropertyDef::new("status", PropertyType::String)
            .with_default(Value::String("active".into()));
        assert_eq!(prop.default, Some(Value::String("active".into())));
    }

    #[test]
    fn schema_version_stable_across_property_mutations() {
        // Per ADR-023, schema revision is bumped only by `ALTER LABEL`
        // operations affecting placement/shard_keys. Property additions and
        // removals mutate the current snapshot in place without bumping
        // version. ALTER LABEL semantics (R210c) bump version explicitly when
        // they ship.
        let mut schema = LabelSchema::new_node_id("Test");
        assert_eq!(schema.schema_revision, 1);
        schema.add_property(PropertyDef::new("a", PropertyType::Int));
        assert_eq!(schema.schema_revision, 1);
        schema.add_property(PropertyDef::new("b", PropertyType::String));
        assert_eq!(schema.schema_revision, 1);
        schema.remove_property("a");
        assert_eq!(schema.schema_revision, 1);
    }

    #[test]
    fn schema_mode_default_is_strict() {
        assert_eq!(SchemaMode::default(), SchemaMode::Strict);
        let schema = LabelSchema::new_node_id("Test");
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
        let mut schema = LabelSchema::new_node_id("Test");
        assert_eq!(schema.mode, SchemaMode::Strict);

        schema.set_mode(SchemaMode::Validated);
        assert_eq!(schema.mode, SchemaMode::Validated);
        assert!(!schema.is_strict());

        schema.set_mode(SchemaMode::Strict);
        assert_eq!(schema.mode, SchemaMode::Strict);
        assert!(schema.is_strict());
    }

    #[test]
    fn schema_mode_msgpack_roundtrip() {
        let mut schema = LabelSchema::new_node_id("Flexible");
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

        let mut schema = LabelSchema::new_node_id("Memory");
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
