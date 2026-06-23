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

    /// Bitemporal flag (ADR-027, R172a). When `true`, every node of this
    /// label carries the `(valid_from, valid_to)` valid-time interval and
    /// the engine-assigned `__ingestion_ts__` (HLC commit-ts), and the
    /// storage layer keeps one node record per `(node_id, valid_from)` so
    /// multiple versions of the same logical node coexist. Immutable for
    /// the lifetime of the label — toggling on an existing label is
    /// rejected at DDL time; re-creation via a new label is the migration
    /// path. Default: `false` (point-in-time only — current MVCC-only
    /// behaviour for all pre-ADR-027 labels).
    pub temporal: bool,
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
            temporal: false,
        }
    }

    /// Convenience for CE call sites and tests: graph-default placement.
    pub fn new_node_id(name: impl Into<String>) -> Self {
        Self::new(name, PlacementPolicy::NodeId)
    }

    /// Set the bitemporal flag (ADR-027, R172a). The flag is immutable for
    /// the lifetime of an installed label — this setter is only for fresh
    /// schemas constructed in-memory by the DDL executor before the first
    /// `save_current_label_schema` call.
    pub fn set_temporal(&mut self, temporal: bool) {
        self.temporal = temporal;
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
mod tests;
