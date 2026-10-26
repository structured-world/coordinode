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

/// A direct child of an [`Expr`]: an expression, or the subplan of a graph
/// subquery.
pub enum ExprChild<'a> {
    /// An operand, argument, item or branch.
    Expr(&'a mut Expr),
    /// The plan a subquery form runs for each row.
    Plan(&'a mut LogicalPlan),
}

impl Expr {
    /// Call `f` on each direct child of this expression, in written order.
    /// A pass over expressions recurses through this, so no child escapes it.
    pub fn for_each_child_mut(&mut self, f: &mut impl FnMut(ExprChild<'_>)) {
        let mut expr = |e: &mut Expr| f(ExprChild::Expr(e));
        match self {
            Expr::Property { base, .. } => expr(base),
            Expr::Binary { left, right, .. } => {
                expr(left);
                expr(right);
            }
            Expr::Unary { operand, .. }
            | Expr::IsNull { operand, .. }
            | Expr::IsTyped { operand, .. } => expr(operand),
            Expr::Call { args: items, .. } | Expr::List(items) => items.iter_mut().for_each(expr),
            Expr::Map(entries) => entries.iter_mut().for_each(|(_, v)| expr(v)),
            Expr::MapProjection { base, items } => {
                expr(base);
                for item in items {
                    if let MapProjItem::Computed(_, value) = item {
                        expr(value);
                    }
                }
            }
            Expr::In { item, list } => {
                expr(item);
                expr(list);
            }
            Expr::StringMatch { value, pattern, .. } => {
                expr(value);
                expr(pattern);
            }
            Expr::Case {
                operand,
                branches,
                otherwise,
            } => {
                if let Some(op) = operand {
                    expr(op);
                }
                for (cond, result) in branches {
                    expr(cond);
                    expr(result);
                }
                if let Some(el) = otherwise {
                    expr(el);
                }
            }
            Expr::Subscript { base, index } => {
                expr(base);
                expr(index);
            }
            Expr::Slice { base, start, end } => {
                expr(base);
                if let Some(s) = start {
                    expr(s);
                }
                if let Some(e) = end {
                    expr(e);
                }
            }
            Expr::ListComprehension {
                list, filter, map, ..
            } => {
                expr(list);
                if let Some(p) = filter {
                    expr(p);
                }
                if let Some(m) = map {
                    expr(m);
                }
            }
            Expr::ListQuantifier {
                list, predicate, ..
            } => {
                expr(list);
                expr(predicate);
            }
            Expr::Reduce {
                init, list, step, ..
            } => {
                expr(init);
                expr(list);
                expr(step);
            }
            Expr::ExistsSubplan(subplan) | Expr::CountSubplan(subplan) => {
                f(ExprChild::Plan(subplan));
            }
            Expr::CollectSubplan {
                subplan,
                projection: other,
            }
            | Expr::PatternComprehension {
                subplan,
                map: other,
            } => {
                f(ExprChild::Plan(subplan));
                f(ExprChild::Expr(other));
            }
            Expr::Parameter(_) | Expr::Literal(_) | Expr::Variable(_) | Expr::Star => {}
        }
    }

    /// Call `f` on every subquery plan this expression holds, outside other
    /// subquery plans: a plan nested in one is `f`'s to reach.
    pub fn for_each_subplan_mut(&mut self, f: &mut impl FnMut(&mut LogicalPlan)) {
        self.for_each_child_mut(&mut |child| match child {
            ExprChild::Expr(e) => e.for_each_subplan_mut(f),
            ExprChild::Plan(plan) => f(plan),
        });
    }

    /// Replace every [`Expr::Parameter`] node with its bound literal from `params`.
    ///
    /// Parameters absent from the map are left as-is (they evaluate to `Null`).
    /// Graph-subquery forms recurse into their embedded subplan, so a parameter
    /// used inside a correlated subquery is bound too.
    pub fn substitute_params(&mut self, params: &HashMap<String, Value>) {
        if let Expr::Parameter(name) = self {
            if let Some(value) = params.get(name.as_str()) {
                *self = Expr::Literal(value.clone());
            }
            return;
        }
        self.for_each_child_mut(&mut |child| match child {
            ExprChild::Expr(e) => e.substitute_params(params),
            ExprChild::Plan(plan) => plan.substitute_params(params),
        });
    }
}
