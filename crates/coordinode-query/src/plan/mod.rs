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
