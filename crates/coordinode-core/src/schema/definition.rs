//! Graph schema: label and edge type declarations with property definitions.
//!
//! Schema is declared per label (node type) and per edge type. Every node has
//! exactly one primary label. Properties have declared types, requiredness
//! and defaults; uniqueness and the other named constraints of a label are
//! separate [`NodeConstraint`]s.
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

    /// A uniqueness flag that earlier releases stored with the property.
    /// Never set now: uniqueness is a named constraint of the label, and a
    /// definition that sets this flag is refused.
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

impl PropertyType {
    /// The scalar type a DDL type name denotes, in the cypher, SQL or Neo4j
    /// spelling, any case; `None` for an unknown name and for the types that
    /// take parameters (vectors, arrays, computed properties).
    pub fn from_type_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_uppercase().as_str() {
            "BIGINT" | "INT" | "INTEGER" | "SMALLINT" => Self::Int,
            "FLOAT" | "DOUBLE" | "REAL" => Self::Float,
            "STRING" | "TEXT" | "VARCHAR" => Self::String,
            "BOOL" | "BOOLEAN" => Self::Bool,
            "TIMESTAMP" => Self::Timestamp,
            "BLOB" => Self::Blob,
            "BINARY" => Self::Binary,
            "GEO" | "POINT" => Self::Geo,
            "MAP" => Self::Map,
            "DOCUMENT" => Self::Document,
            _ => return None,
        })
    }
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
/// Declared explicitly at label creation: there is no default, every label
/// declares its placement strategy at creation time.
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
/// Multi-key coexistence: during a lazy re-shard, a label has one
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

/// Physical storage layout for a relational TABLE label. Orthogonal to
/// the logical model: a table declares its layout independently of its schema.
/// Only meaningful for table labels (those with a non-empty primary key); plain
/// graph labels are always row-stored on the node path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StorageLayout {
    /// One MessagePack record per row on the existing node path. Best for OLTP
    /// point reads, updates, and whole-row reads. The default.
    #[default]
    Row,
    /// Rows stored in the engine's native columnar block type (per-column
    /// chunks, zone maps, vectorized scan). Best for analytical scans over a
    /// few columns of many rows. Each columnar table owns its own columnar
    /// tree.
    Columnar,
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

    /// Multi-key list for lazy re-sharding. Always contains at least
    /// one PRIMARY entry. During a lazy migration also contains one LEGACY
    /// entry. CE labels carry `[primary_node_id(1)]` permanently.
    pub shard_keys: Vec<ShardKeySpec>,

    /// Schema snapshot revision. Every published change of the definition or
    /// of its constraints is a new revision, and a published revision is
    /// never rewritten. The revision is the key suffix for
    /// `schema:label:<name>:<revision>` and is what `current_revision`
    /// pointer names.
    ///
    /// Distinct from per-row data versioning: MVCC `commit_ts` (HLC) tracks
    /// data revisions at write granularity; `schema_revision` tracks DDL
    /// snapshots. Renaming follows the Postgres `pg_class.relrewrite` /
    /// Kubernetes `metadata.generation` convention — "revision" = DDL,
    /// "version" = data.
    pub schema_revision: u64,

    /// Bitemporal flag. When `true`, every node of this
    /// label carries the `(valid_from, valid_to)` valid-time interval and
    /// the engine-assigned `__ingestion_ts__` (HLC commit-ts), and the
    /// storage layer keeps one node record per `(node_id, valid_from)` so
    /// multiple versions of the same logical node coexist. Immutable for
    /// the lifetime of the label — toggling on an existing label is
    /// rejected at DDL time; re-creation via a new label is the migration
    /// path. Default: `false` (point-in-time only, MVCC history alone).
    pub temporal: bool,

    /// Declared key columns of a relational TABLE, in declaration order
    /// (composite keys allowed). Read through [`LabelSchema::table_key`].
    #[serde(default)]
    primary_key: Vec<String>,

    /// Physical storage layout for a table label. `Row` (default) stores
    /// each row on the node path; `Columnar` stores rows in the engine's native
    /// columnar block type. Ignored for non-table labels.
    #[serde(default)]
    pub storage_layout: StorageLayout,

    /// A TABLE declared without key columns, whose rows are keyed by their
    /// NodeId. Read through [`LabelSchema::table_key`].
    #[serde(default)]
    keyed_by_row_id: bool,

    /// Named constraints on the nodes of this label, in creation order. Part
    /// of the schema revision, so enabling one binds the revision that every
    /// writer validated under. Read through [`LabelSchema::constraints`].
    #[serde(default)]
    constraints: Vec<NodeConstraint>,
}

/// What a node constraint requires of the properties it names.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConstraintKind {
    /// No two nodes share the values; a node missing any of them is not
    /// constrained.
    Unique,
    /// The property is present and not null.
    NotNull,
    /// Every property is present and not null, and no two nodes share the
    /// values.
    NodeKey,
    /// A present, non-null value has this type.
    Type(PropertyType),
}

impl std::fmt::Display for ConstraintKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unique => write!(f, "UNIQUE"),
            Self::NotNull => write!(f, "NOT NULL"),
            Self::NodeKey => write!(f, "NODE KEY"),
            Self::Type(t) => write!(f, "TYPE {t}"),
        }
    }
}

/// Where a constraint is in its life.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConstraintState {
    /// Published and enforced on every write, while its index is still being
    /// validated against the stored data: the guarantee over that data is
    /// not established yet. A constraint left here by an interrupted build
    /// stays enforced until it is dropped.
    Validating,
    /// Validated against the stored data and enforced on every write.
    Active,
}

/// A named constraint on the nodes whose primary label is the schema's label.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeConstraint {
    /// The constraint name, unique in the database. A uniqueness or key
    /// constraint's index carries the same name.
    pub name: String,
    /// The constrained properties, in declaration order.
    pub properties: Vec<String>,
    /// What the constraint requires.
    pub kind: ConstraintKind,
    /// Where it is in its life.
    pub state: ConstraintState,
}

impl NodeConstraint {
    /// Whether every constrained property must be present and not null.
    pub fn requires_presence(&self) -> bool {
        matches!(self.kind, ConstraintKind::NotNull | ConstraintKind::NodeKey)
    }

    /// Whether the constraint owns the unique index named after it.
    pub fn owns_index(&self) -> bool {
        matches!(self.kind, ConstraintKind::Unique | ConstraintKind::NodeKey)
    }

    /// Whether the constraint checks each node on its own (presence or
    /// type), as opposed to only across nodes through its index.
    pub fn checks_each_node(&self) -> bool {
        !matches!(self.kind, ConstraintKind::Unique)
    }

    /// Whether `other` requires the same thing of the same properties.
    pub fn same_requirement(&self, other: &NodeConstraint) -> bool {
        self.kind == other.kind && self.properties == other.properties
    }
}

/// How a relational TABLE addresses its rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKey<'a> {
    /// No declared key: a row is addressed by its NodeId, and every insert is
    /// a new row.
    RowId,
    /// Declared key columns: the key is unique across the table and never
    /// changes for a row; inserting an existing key is refused.
    Columns(&'a [String]),
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
            primary_key: Vec::new(),
            storage_layout: StorageLayout::Row,
            keyed_by_row_id: false,
            constraints: Vec::new(),
        }
    }

    /// The constraints on this label's nodes, in creation order.
    pub fn constraints(&self) -> &[NodeConstraint] {
        &self.constraints
    }

    /// The constraint named `name`, if this label has it.
    pub fn constraint(&self, name: &str) -> Option<&NodeConstraint> {
        self.constraints.iter().find(|c| c.name == name)
    }

    /// The constraint named `name`, to change its state.
    pub fn constraint_mut(&mut self, name: &str) -> Option<&mut NodeConstraint> {
        self.constraints.iter_mut().find(|c| c.name == name)
    }

    /// Add a constraint. The caller bumps the revision: a new constraint
    /// changes what a write must satisfy.
    pub fn add_constraint(&mut self, constraint: NodeConstraint) {
        self.constraints.push(constraint);
    }

    /// Remove the constraint named `name`, returning it.
    pub fn remove_constraint(&mut self, name: &str) -> Option<NodeConstraint> {
        let at = self.constraints.iter().position(|c| c.name == name)?;
        Some(self.constraints.remove(at))
    }

    /// Why `constraint` cannot hold under this definition, or `None` when it
    /// can: it names a computed property, requires a type other than the one
    /// declared, or requires a property a STRICT label never stores.
    pub fn constraint_conflict(&self, constraint: &NodeConstraint) -> Option<String> {
        for name in &constraint.properties {
            match self.properties.get(name) {
                Some(p) if p.is_computed() => {
                    return Some(format!(
                        "property `{name}` of :{} is computed and cannot be constrained",
                        self.name
                    ));
                }
                Some(p) => {
                    if let ConstraintKind::Type(required) = &constraint.kind {
                        if &p.property_type != required {
                            return Some(format!(
                                "property `{name}` of :{} is declared {}, so a value of type \
                                 {required} can never be stored",
                                self.name, p.property_type
                            ));
                        }
                    }
                }
                None if self.is_strict() && constraint.requires_presence() => {
                    return Some(format!(
                        "the STRICT label :{} does not declare `{name}`, so no node can carry it",
                        self.name
                    ));
                }
                None => {}
            }
        }
        None
    }

    /// How this label addresses its rows when it is a relational TABLE;
    /// `None` for a plain graph label.
    pub fn table_key(&self) -> Option<TableKey<'_>> {
        if !self.primary_key.is_empty() {
            Some(TableKey::Columns(&self.primary_key))
        } else if self.keyed_by_row_id {
            Some(TableKey::RowId)
        } else {
            None
        }
    }

    /// Whether this label is a relational TABLE.
    pub fn is_table(&self) -> bool {
        self.table_key().is_some()
    }

    /// The declared key columns; empty for a table keyed by row id and for a
    /// graph label.
    pub fn key_columns(&self) -> &[String] {
        &self.primary_key
    }

    /// Whether this table stores its rows in the columnar layout.
    pub fn is_columnar(&self) -> bool {
        self.storage_layout == StorageLayout::Columnar
    }

    /// Mark this label as a relational TABLE keyed by `columns`, or by row id
    /// when `columns` is empty.
    pub fn make_table(&mut self, columns: Vec<String>) {
        self.keyed_by_row_id = columns.is_empty();
        self.primary_key = columns;
    }

    /// Set the physical storage layout (only meaningful for a table).
    pub fn set_storage_layout(&mut self, layout: StorageLayout) {
        self.storage_layout = layout;
    }

    /// Convenience for CE call sites and tests: graph-default placement.
    pub fn new_node_id(name: impl Into<String>) -> Self {
        Self::new(name, PlacementPolicy::NodeId)
    }

    /// Set the bitemporal flag. The flag is immutable for
    /// the lifetime of an installed label — this setter is only for fresh
    /// schemas constructed in-memory by the DDL executor before the first
    /// `save_current_label_schema` call.
    pub fn set_temporal(&mut self, temporal: bool) {
        self.temporal = temporal;
    }

    /// Add a property definition. Mutates the current snapshot; does not
    /// bump `schema_revision` (revision changes only on placement/shard_keys/
    /// mode mutations).
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

/// Placement policy for edges relative to their endpoint nodes.
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
    /// the priority queue (query-driven, on touch) to prefer recently
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
/// ranges to shard ids for a given label.
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

    /// Placement policy for adjacency entries relative to endpoint nodes.
    /// Default `ColocateWithSource` matches the graph-default
    /// traversal pattern; alternative values opt into target-co-location or
    /// replicated adjacency for specific workloads.
    pub placement: EdgePlacement,

    /// Schema snapshot revision. Bumped by ALTER operations affecting
    /// placement or temporal flag (mirrors LabelSchema semantics).
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
    /// bump `schema_revision` (mirrors LabelSchema semantics).
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
/// The schema partition is revision-prefixed from day one: each shard-key
/// changing `ALTER LABEL` writes a new
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

/// Encode the key naming the label that holds a constraint:
/// `schema:constraint:<name>`. Value: the label name, UTF-8. One key per
/// name keeps constraint names unique across labels.
pub fn encode_constraint_name_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(18 + name.len());
    key.extend_from_slice(b"schema:constraint:");
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

/// Key prefix of every edge type schema record and marker.
pub const EDGE_TYPE_SCHEMA_KEY_PREFIX: &[u8] = b"schema:edge_type:";

/// The edge type name of a key written by [`encode_edge_type_schema_key`].
/// Names cannot contain ':' (DDL grammar), so the rightmost ':' splits the
/// name from the revision.
pub fn decode_edge_type_schema_key_name(key: &[u8]) -> Option<&str> {
    let suffix = key.strip_prefix(EDGE_TYPE_SCHEMA_KEY_PREFIX)?;
    let (name, _revision) = core::str::from_utf8(suffix).ok()?.rsplit_once(':')?;
    Some(name)
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
/// This logical key namespace inside `Partition::Schema`
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
