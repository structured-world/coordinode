//! Cypher-AST expression lowering into the neutral plan IR.
//!
//! This is part of the Cypher frontend: it converts a `cypher::ast::Expr` into
//! the dialect-independent `plan::Expr`. The neutral IR kernel itself never
//! depends on Cypher; the conversion lives here on the frontend side. A future
//! SQL frontend lowers its own AST into the same neutral expression.
//!
//! Graph subqueries (`EXISTS`/`COUNT`/`COLLECT { … }`, pattern predicates, and
//! pattern comprehensions) are lowered to an already-built neutral subplan: the
//! inner pattern is wrapped in a synthetic `… RETURN *` query and planned
//! through the same logical planner that serves top-level queries, so subquery
//! pattern semantics match a top-level `MATCH` exactly. The executor then runs
//! the embedded subplan correlated with the outer row, rather than re-parsing a
//! dialect clause at evaluation time.

use crate::cypher::ast::{
    BinaryOperator, Clause, Expr as CExpr, ListPredicateKind, MapProjectionItem, MatchClause,
    Pattern, Query, ReturnClause, ReturnItem, StringOp, UnaryOperator,
};
use crate::plan::expr::{BinOp, Expr as PExpr, MapProjItem, Quantifier, StrOp, UnOp};
use crate::planner::builder::{build_logical_plan, PlanError};
use crate::planner::logical::LogicalPlan;

/// Lower a Cypher expression into the neutral plan IR.
pub fn lower_expr(e: &CExpr) -> Result<PExpr, PlanError> {
    Ok(match e {
        CExpr::Literal(v) => PExpr::Literal(v.clone()),
        CExpr::Parameter(name) => PExpr::Parameter(name.clone()),
        CExpr::Variable(name) => PExpr::Variable(name.clone()),
        CExpr::PropertyAccess { expr, property } => PExpr::Property {
            base: lower_boxed(expr)?,
            key: property.clone(),
        },
        CExpr::BinaryOp { left, op, right } => PExpr::Binary {
            left: lower_boxed(left)?,
            op: lower_binop(*op),
            right: lower_boxed(right)?,
        },
        CExpr::UnaryOp { op, expr } => PExpr::Unary {
            op: lower_unop(*op),
            operand: lower_boxed(expr)?,
        },
        CExpr::FunctionCall {
            name,
            args,
            distinct,
        } => PExpr::Call {
            name: name.clone(),
            args: lower_all(args)?,
            distinct: *distinct,
        },
        CExpr::List(items) => PExpr::List(lower_all(items)?),
        CExpr::MapLiteral(entries) => PExpr::Map(lower_entries(entries)?),
        CExpr::MapProjection { expr, items } => PExpr::MapProjection {
            base: lower_boxed(expr)?,
            items: items
                .iter()
                .map(lower_map_proj_item)
                .collect::<Result<_, _>>()?,
        },
        CExpr::In { expr, list } => PExpr::In {
            item: lower_boxed(expr)?,
            list: lower_boxed(list)?,
        },
        CExpr::IsNull { expr, negated } => PExpr::IsNull {
            operand: lower_boxed(expr)?,
            negated: *negated,
        },
        CExpr::IsTyped {
            expr,
            type_name,
            negated,
        } => PExpr::IsTyped {
            operand: lower_boxed(expr)?,
            type_name: type_name.clone(),
            negated: *negated,
        },
        CExpr::StringMatch { expr, op, pattern } => PExpr::StringMatch {
            value: lower_boxed(expr)?,
            op: lower_strop(*op),
            pattern: lower_boxed(pattern)?,
        },
        CExpr::Case {
            operand,
            when_clauses,
            else_clause,
        } => PExpr::Case {
            operand: lower_opt(operand)?,
            branches: when_clauses
                .iter()
                .map(|(w, t)| Ok::<_, PlanError>((lower_expr(w)?, lower_expr(t)?)))
                .collect::<Result<_, _>>()?,
            otherwise: lower_opt(else_clause)?,
        },
        CExpr::Subscript { expr, index } => PExpr::Subscript {
            base: lower_boxed(expr)?,
            index: lower_boxed(index)?,
        },
        CExpr::Slice { expr, start, end } => PExpr::Slice {
            base: lower_boxed(expr)?,
            start: lower_opt(start)?,
            end: lower_opt(end)?,
        },
        CExpr::ListComprehension {
            var,
            list,
            pred,
            map,
        } => PExpr::ListComprehension {
            var: var.clone(),
            list: lower_boxed(list)?,
            filter: lower_opt(pred)?,
            map: lower_opt(map)?,
        },
        CExpr::ListPredicate {
            kind,
            var,
            list,
            pred,
        } => PExpr::ListQuantifier {
            kind: lower_quantifier(*kind),
            var: var.clone(),
            list: lower_boxed(list)?,
            predicate: lower_boxed(pred)?,
        },
        CExpr::Reduce {
            acc,
            init,
            var,
            list,
            expr,
        } => PExpr::Reduce {
            acc: acc.clone(),
            init: lower_boxed(init)?,
            var: var.clone(),
            list: lower_boxed(list)?,
            step: lower_boxed(expr)?,
        },
        CExpr::Star => PExpr::Star,

        // Graph subqueries: lower to a pre-built neutral subplan.
        CExpr::PatternPredicate(pattern) => {
            PExpr::ExistsSubplan(Box::new(plan_pattern(pattern.clone(), None)?))
        }
        CExpr::ExistsSubquery(mc) => {
            PExpr::ExistsSubplan(Box::new(plan_match_clause((**mc).clone())?))
        }
        CExpr::CountSubquery(mc) => {
            PExpr::CountSubplan(Box::new(plan_match_clause((**mc).clone())?))
        }
        CExpr::CollectSubquery { match_clause, expr } => PExpr::CollectSubplan {
            subplan: Box::new(plan_match_clause((**match_clause).clone())?),
            projection: lower_boxed(expr)?,
        },
        CExpr::PatternComprehension {
            pattern,
            where_clause,
            map,
        } => PExpr::PatternComprehension {
            subplan: Box::new(plan_pattern(
                (**pattern).clone(),
                where_clause.as_deref().cloned(),
            )?),
            map: lower_boxed(map)?,
        },
    })
}

fn lower_boxed(e: &CExpr) -> Result<Box<PExpr>, PlanError> {
    Ok(Box::new(lower_expr(e)?))
}

fn lower_opt(o: &Option<Box<CExpr>>) -> Result<Option<Box<PExpr>>, PlanError> {
    match o {
        Some(e) => Ok(Some(lower_boxed(e)?)),
        None => Ok(None),
    }
}

fn lower_all(items: &[CExpr]) -> Result<Vec<PExpr>, PlanError> {
    items.iter().map(lower_expr).collect()
}

fn lower_entries(entries: &[(String, CExpr)]) -> Result<Vec<(String, PExpr)>, PlanError> {
    entries
        .iter()
        .map(|(k, v)| Ok::<_, PlanError>((k.clone(), lower_expr(v)?)))
        .collect()
}

fn lower_map_proj_item(item: &MapProjectionItem) -> Result<MapProjItem, PlanError> {
    Ok(match item {
        MapProjectionItem::Property(name) => MapProjItem::Property(name.clone()),
        MapProjectionItem::Computed(alias, expr) => {
            MapProjItem::Computed(alias.clone(), lower_expr(expr)?)
        }
    })
}

fn lower_binop(op: BinaryOperator) -> BinOp {
    match op {
        BinaryOperator::Add => BinOp::Add,
        BinaryOperator::Sub => BinOp::Sub,
        BinaryOperator::Mul => BinOp::Mul,
        BinaryOperator::Div => BinOp::Div,
        BinaryOperator::Modulo => BinOp::Modulo,
        BinaryOperator::Eq => BinOp::Eq,
        BinaryOperator::Neq => BinOp::Neq,
        BinaryOperator::Lt => BinOp::Lt,
        BinaryOperator::Lte => BinOp::Lte,
        BinaryOperator::Gt => BinOp::Gt,
        BinaryOperator::Gte => BinOp::Gte,
        BinaryOperator::And => BinOp::And,
        BinaryOperator::Or => BinOp::Or,
        BinaryOperator::Xor => BinOp::Xor,
    }
}

fn lower_unop(op: UnaryOperator) -> UnOp {
    match op {
        UnaryOperator::Not => UnOp::Not,
        UnaryOperator::Neg => UnOp::Neg,
    }
}

fn lower_strop(op: StringOp) -> StrOp {
    match op {
        StringOp::StartsWith => StrOp::StartsWith,
        StringOp::EndsWith => StrOp::EndsWith,
        StringOp::Contains => StrOp::Contains,
        StringOp::Regex => StrOp::Regex,
    }
}

fn lower_quantifier(kind: ListPredicateKind) -> Quantifier {
    match kind {
        ListPredicateKind::All => Quantifier::All,
        ListPredicateKind::Any => Quantifier::Any,
        ListPredicateKind::None => Quantifier::None,
        ListPredicateKind::Single => Quantifier::Single,
    }
}

/// Plan a single pattern (plus optional filter) as a correlated subplan.
fn plan_pattern(pattern: Pattern, where_clause: Option<CExpr>) -> Result<LogicalPlan, PlanError> {
    plan_match_clause(MatchClause {
        patterns: vec![pattern],
        where_clause,
    })
}

/// Plan a `MATCH … RETURN *` subplan from a match clause, matching the
/// synthetic-query construction the executor uses for correlated subqueries.
fn plan_match_clause(mc: MatchClause) -> Result<LogicalPlan, PlanError> {
    let query = Query {
        clauses: vec![
            Clause::Match(mc),
            Clause::Return(ReturnClause {
                distinct: false,
                items: vec![ReturnItem {
                    expr: CExpr::Star,
                    alias: None,
                }],
            }),
        ],
        hints: Vec::new(),
        unions: Vec::new(),
    };
    build_logical_plan(&query)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
