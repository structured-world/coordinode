//! Neutral expression IR.
//!
//! A language-agnostic expression tree the executor evaluates and the planner
//! reasons over. Scalar and collection forms are dialect-independent (every
//! supported language has analogues). The graph-subquery forms carry an
//! already-lowered neutral subplan rather than a dialect parse tree, so the
//! executor runs the subplan directly instead of re-parsing a dialect clause.

use std::collections::HashMap;

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

impl Expr {
    /// Replace every [`Expr::Parameter`] node with its bound literal from `params`.
    ///
    /// Parameters absent from the map are left as-is (they evaluate to `Null`).
    /// Graph-subquery forms recurse into their embedded subplan, so a parameter
    /// used inside a correlated subquery is bound too.
    pub fn substitute_params(&mut self, params: &HashMap<String, Value>) {
        match self {
            Expr::Parameter(name) => {
                if let Some(value) = params.get(name.as_str()) {
                    *self = Expr::Literal(value.clone());
                }
            }
            Expr::Property { base, .. } => base.substitute_params(params),
            Expr::Binary { left, right, .. } => {
                left.substitute_params(params);
                right.substitute_params(params);
            }
            Expr::Unary { operand, .. } => operand.substitute_params(params),
            Expr::Call { args, .. } => {
                for arg in args {
                    arg.substitute_params(params);
                }
            }
            Expr::List(items) => {
                for item in items {
                    item.substitute_params(params);
                }
            }
            Expr::Map(entries) => {
                for (_, v) in entries {
                    v.substitute_params(params);
                }
            }
            Expr::MapProjection { base, items } => {
                base.substitute_params(params);
                for item in items {
                    if let MapProjItem::Computed(_, ref mut value) = item {
                        value.substitute_params(params);
                    }
                }
            }
            Expr::In { item, list } => {
                item.substitute_params(params);
                list.substitute_params(params);
            }
            Expr::IsNull { operand, .. } => operand.substitute_params(params),
            Expr::IsTyped { operand, .. } => operand.substitute_params(params),
            Expr::StringMatch { value, pattern, .. } => {
                value.substitute_params(params);
                pattern.substitute_params(params);
            }
            Expr::Case {
                operand,
                branches,
                otherwise,
            } => {
                if let Some(op) = operand {
                    op.substitute_params(params);
                }
                for (cond, result) in branches {
                    cond.substitute_params(params);
                    result.substitute_params(params);
                }
                if let Some(el) = otherwise {
                    el.substitute_params(params);
                }
            }
            Expr::Subscript { base, index } => {
                base.substitute_params(params);
                index.substitute_params(params);
            }
            Expr::Slice { base, start, end } => {
                base.substitute_params(params);
                if let Some(s) = start {
                    s.substitute_params(params);
                }
                if let Some(e) = end {
                    e.substitute_params(params);
                }
            }
            Expr::ListComprehension {
                list, filter, map, ..
            } => {
                list.substitute_params(params);
                if let Some(p) = filter {
                    p.substitute_params(params);
                }
                if let Some(m) = map {
                    m.substitute_params(params);
                }
            }
            Expr::ListQuantifier {
                list, predicate, ..
            } => {
                list.substitute_params(params);
                predicate.substitute_params(params);
            }
            Expr::Reduce {
                init, list, step, ..
            } => {
                init.substitute_params(params);
                list.substitute_params(params);
                step.substitute_params(params);
            }
            Expr::ExistsSubplan(subplan) | Expr::CountSubplan(subplan) => {
                subplan.substitute_params(params);
            }
            Expr::CollectSubplan {
                subplan,
                projection,
            } => {
                subplan.substitute_params(params);
                projection.substitute_params(params);
            }
            Expr::PatternComprehension { subplan, map } => {
                subplan.substitute_params(params);
                map.substitute_params(params);
            }
            Expr::Literal(_) | Expr::Variable(_) | Expr::Star => {}
        }
    }
}
