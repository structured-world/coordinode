//! Language-neutral query IR kernel.
//!
//! The logical layer (planner optimizer passes, executor, advisor) operates on
//! this IR, which carries no dependency on any query dialect. A language
//! frontend (Cypher today, SQL later) parses its own surface syntax and lowers
//! it into this neutral IR; the same planner and executor then consume it,
//! whichever dialect produced it. Translating one dialect into another's AST is
//! never the path: both lower independently into the kernel here.
//!
//! This module is introduced incrementally. It first defines the neutral
//! expression surface; subsequent work migrates the operator tree and the
//! executor onto it and removes the dialect coupling from the layers below.

pub mod expr;

pub use expr::{BinOp, Expr, MapProjItem, Quantifier, StrOp, UnOp};

/// A sort key in the neutral IR: an expression to order by plus direction.
#[derive(Debug, Clone, PartialEq)]
pub struct SortItem {
    /// Expression evaluated per row to produce the sort key.
    pub expr: Expr,
    /// `true` for ascending order, `false` for descending.
    pub ascending: bool,
}

/// A single mutation in a `SET` clause, in the neutral IR.
#[derive(Debug, Clone, PartialEq)]
pub enum SetItem {
    /// `n.prop = expr` — set a single property.
    Property {
        variable: String,
        property: String,
        expr: Expr,
    },
    /// `n.a.b.c = expr` — set a nested document path via merge operand.
    PropertyPath {
        variable: String,
        path: Vec<String>,
        expr: Expr,
    },
    /// `doc_push(n.tags, "x")` and friends — document array/numeric mutation.
    DocFunction {
        function: String,
        variable: String,
        path: Vec<String>,
        value_expr: Expr,
    },
    /// `n = {map}` — replace all properties with the map.
    ReplaceProperties { variable: String, expr: Expr },
    /// `n += {map}` — merge properties from the map.
    MergeProperties { variable: String, expr: Expr },
    /// `n:Label` — add a label.
    AddLabel { variable: String, label: String },
}

/// Property conflict resolution strategy for `MERGE NODES`, in the neutral IR.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MergeNodesConflictStrategy {
    /// Surviving node's properties win on collision (the default).
    #[default]
    KeepFirst,
    /// Non-surviving node's properties overwrite the surviving's.
    KeepLast,
    /// Non-null values from the non-surviving fill nulls on the surviving.
    Coalesce,
    /// Per-property expressions. Each set item is evaluated against a row
    /// binding the surviving node and the non-surviving node.
    SetExpressions(Vec<SetItem>),
}

/// A `CREATE TRIGGER` definition in the neutral IR. Carries the already-neutral
/// core schema descriptors (`coordinode_core::schema::triggers`); the runtime
/// fields (`enabled`, `created_at_hlc_us`) are stamped by the executor.
#[derive(Debug, Clone, PartialEq)]
pub struct TriggerDef {
    /// Unique trigger name (DROP / ALTER / SHOW handle).
    pub name: String,
    /// Target schema element (`:Label` or `[:EdgeType]`).
    pub target: coordinode_core::schema::triggers::TriggerTargetSchema,
    /// Event kinds that fire the trigger.
    pub events: coordinode_core::schema::triggers::TriggerEventsSchema,
    /// Synchronous (BEFORE COMMIT) or asynchronous (AFTER COMMIT) timing.
    pub timing: coordinode_core::schema::triggers::TriggerTimingSchema,
    /// Trigger body, captured as a source string and re-parsed when it fires.
    pub body_source: String,
    /// Per-trigger L1 cascade-depth override (`None` = cluster default).
    pub cascade_limit: Option<u32>,
    /// Per-trigger L2 cascade-fanout override (`None` = cluster default).
    pub cascade_fanout: Option<u32>,
    /// Error-handling policy (`None` = timing-specific default).
    pub on_error: Option<coordinode_core::schema::triggers::OnErrorPolicySchema>,
}

/// The change applied by `ALTER TRIGGER`, in the neutral IR.
#[derive(Debug, Clone, PartialEq)]
pub enum AlterTriggerAction {
    /// Stop firing without forgetting the definition.
    Disable,
    /// Resume firing after a prior disable.
    Enable,
    /// Replace the trigger body (re-validated at execution).
    SetBody(String),
    /// Replace the on-error policy.
    SetOnError(coordinode_core::schema::triggers::OnErrorPolicySchema),
}

/// An `ALTER TRIGGER` definition in the neutral IR.
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTriggerDef {
    /// Target trigger name.
    pub name: String,
    /// What to change.
    pub action: AlterTriggerAction,
}

/// Per-field configuration for `CREATE TEXT INDEX`, in the neutral IR.
#[derive(Debug, Clone, PartialEq)]
pub struct TextIndexFieldSpec {
    /// Property name to index.
    pub property: String,
    /// Analyzer name: language ("english", "russian"), "auto_detect", "none".
    pub analyzer: Option<String>,
}

/// A single property declaration in `CREATE EDGE TYPE` / `CREATE NODE TYPE`, in
/// the neutral IR.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyDecl {
    /// Property name.
    pub name: String,
    /// Declared type name.
    pub type_name: String,
    /// Whether the property is NOT NULL.
    pub not_null: bool,
}

/// Schema-violation handling policy for `SET` mutations, in the neutral IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViolationMode {
    /// Fail the entire query on the first violation (strict, the default).
    #[default]
    Fail,
    /// Skip rows that would violate schema constraints; continue with others.
    Skip,
}

/// Direction of the edge created by DETACH DOCUMENT relative to the source
/// node, in the neutral IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeFromSource {
    /// `(source)-[:TYPE]->(target)` — source is the edge's source.
    Outgoing,
    /// `(source)<-[:TYPE]-(target)` — source is the edge's target.
    Incoming,
}

/// Direction filter for REDIRECT EDGES, in the neutral IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedirectDirection {
    /// Both outgoing and incoming edges (the default).
    #[default]
    Both,
    /// Only outgoing edges `a -> x`.
    Outgoing,
    /// Only incoming edges `x -> a`.
    Incoming,
}

/// Duplicate-edge resolution strategy for `MERGE NODES TRANSFER EDGES`, in the
/// neutral IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeNodesDuplicateStrategy {
    /// Both edges preserved as parallel edges (the default).
    #[default]
    KeepBoth,
    /// Single edge, facets merged via COALESCE; source edge removed.
    MergeProperties,
    /// Keep the target edge as-is, discard the non-surviving edge.
    KeepTarget,
}

/// Endpoints for a MERGE NODES edge transfer, in the neutral IR.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferEdgesEndpoints {
    /// Non-surviving source — edges are read from this node.
    pub src: String,
    /// Surviving destination — must equal the merge target.
    pub dst: String,
}

/// `TRANSFER EDGES ON <node> TO <target> WHERE <predicate>` clause, in the
/// neutral IR.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferEdgesSpec {
    /// The node whose edges are being re-pointed.
    pub node_variable: String,
    /// The node that receives the re-pointed edges.
    pub target_variable: String,
    /// Predicate selecting which edges to transfer. The edge variable is `r`.
    pub predicate: Expr,
}

/// A single removal in a `REMOVE` clause, in the neutral IR. Carries no
/// expression (targets are addressed structurally), so it mirrors the cypher
/// shape exactly.
#[derive(Debug, Clone, PartialEq)]
pub enum RemoveItem {
    /// `n.prop` — remove a property.
    Property { variable: String, property: String },
    /// `n.a.b.c` — remove a nested document path.
    PropertyPath { variable: String, path: Vec<String> },
    /// `n:Label` — remove a label.
    Label { variable: String, label: String },
}
