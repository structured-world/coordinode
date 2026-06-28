//! Neutral expression IR.
//!
//! A language-agnostic expression tree the executor evaluates and the planner
//! reasons over. Scalar and collection forms are dialect-independent (every
//! supported language has analogues). The graph-subquery forms carry an
//! already-lowered neutral subplan rather than a dialect parse tree, so the
//! executor runs the subplan directly instead of re-parsing a dialect clause.

use coordinode_core::graph::types::Value;

use crate::planner::logical::LogicalPlan;

/// Binary operator over two expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Modulo,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    And,
    Or,
    Xor,
}

/// Unary operator over one expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Not,
    Neg,
}

/// String-matching operator (`STARTS WITH` / `ENDS WITH` / `CONTAINS` / regex).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrOp {
    StartsWith,
    EndsWith,
    Contains,
    /// Whole-string regex match.
    Regex,
}

/// List-quantifier kind for `all` / `any` / `none` / `single` predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantifier {
    All,
    Any,
    None,
    Single,
}

/// One entry of a map projection: a property shorthand or a computed entry.
#[derive(Debug, Clone, PartialEq)]
pub enum MapProjItem {
    /// `.name` shorthand: includes `name: base.name`.
    Property(String),
    /// `alias: expression`.
    Computed(String, Expr),
}

/// A neutral expression node.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal value.
    Literal(Value),
    /// Bound parameter, by name.
    Parameter(String),
    /// Variable reference, by name.
    Variable(String),
    /// Property access `base.key`.
    Property { base: Box<Expr>, key: String },
    /// Binary operation `left op right`.
    Binary {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
    },
    /// Unary operation `op operand`.
    Unary { op: UnOp, operand: Box<Expr> },
    /// Function / aggregate call.
    Call {
        name: String,
        args: Vec<Expr>,
        distinct: bool,
    },
    /// List literal.
    List(Vec<Expr>),
    /// Map literal.
    Map(Vec<(String, Expr)>),
    /// Map projection `base { .a, b: expr }`.
    MapProjection {
        base: Box<Expr>,
        items: Vec<MapProjItem>,
    },
    /// `item IN list`.
    In { item: Box<Expr>, list: Box<Expr> },
    /// `operand IS [NOT] NULL`.
    IsNull { operand: Box<Expr>, negated: bool },
    /// `operand IS [NOT] :: TYPE`.
    IsTyped {
        operand: Box<Expr>,
        type_name: String,
        negated: bool,
    },
    /// `value op pattern` for string matching.
    StringMatch {
        value: Box<Expr>,
        op: StrOp,
        pattern: Box<Expr>,
    },
    /// `CASE [operand] WHEN .. THEN .. [ELSE ..] END`.
    Case {
        operand: Option<Box<Expr>>,
        branches: Vec<(Expr, Expr)>,
        otherwise: Option<Box<Expr>>,
    },
    /// `base[index]`.
    Subscript { base: Box<Expr>, index: Box<Expr> },
    /// `base[start..end]`, both bounds optional.
    Slice {
        base: Box<Expr>,
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },
    /// `[var IN list WHERE filter | map]`.
    ListComprehension {
        var: String,
        list: Box<Expr>,
        filter: Option<Box<Expr>>,
        map: Option<Box<Expr>>,
    },
    /// `all/any/none/single(var IN list WHERE predicate)`.
    ListQuantifier {
        kind: Quantifier,
        var: String,
        list: Box<Expr>,
        predicate: Box<Expr>,
    },
    /// `reduce(acc = init, var IN list | step)`.
    Reduce {
        acc: String,
        init: Box<Expr>,
        var: String,
        list: Box<Expr>,
        step: Box<Expr>,
    },
    /// Existence of at least one row of a correlated subplan (covers both a
    /// pattern predicate in a filter and an explicit `EXISTS { .. }`).
    ExistsSubplan(Box<LogicalPlan>),
    /// Row count of a correlated subplan.
    CountSubplan(Box<LogicalPlan>),
    /// Collect `projection` over each row of a correlated subplan into a list.
    CollectSubplan {
        subplan: Box<LogicalPlan>,
        projection: Box<Expr>,
    },
    /// Pattern comprehension: collect `map` over each row of a correlated
    /// subplan (the pattern + its filter lowered into the subplan).
    PatternComprehension {
        subplan: Box<LogicalPlan>,
        map: Box<Expr>,
    },
    /// Star expression (`count(*)` / `RETURN *`).
    Star,
}
