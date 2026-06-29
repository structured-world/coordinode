//! SQL query frontend (R650a).
//!
//! The second [`QueryFrontend`](crate::frontend::QueryFrontend) implementation:
//! it parses SQL with `sqlparser` and lowers the SQL AST **natively** into the
//! same language-neutral [`LogicalPlan`] that Cypher lowers into — never a
//! translation of SQL into Cypher. This validates that the planner / executor /
//! advisor are genuinely dialect-agnostic (the trait is not single-impl).
//!
//! Scope today: `SELECT` (single table, projection, `WHERE`) and `INSERT ...
//! VALUES`. A SQL table name is a label; a row is a node. `UPDATE` / `DELETE`
//! and joins lower in later increments.

use sqlparser::ast::{
    Expr as SqlExpr, Insert, Query as SqlQuery, SelectItem, SetExpr, Statement, TableFactor,
    Value as SqlValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use coordinode_core::graph::types::{Value, VectorConsistencyMode};
use coordinode_core::txn::read_consistency::ReadConsistencyMode;

use crate::frontend::{FrontendError, ParsedQuery, QueryFrontend};
use crate::plan::expr::{BinOp, Expr};
use crate::planner::logical::{LogicalOp, LogicalPlan, ProjectItem};

/// The SQL frontend: `sqlparser` parse + native lowering into the neutral IR.
#[derive(Debug, Clone, Copy, Default)]
pub struct SqlFrontend;

impl SqlFrontend {
    /// Create a SQL frontend.
    pub fn new() -> Self {
        Self
    }
}

impl QueryFrontend for SqlFrontend {
    fn parse(&self, text: &str) -> Result<ParsedQuery, FrontendError> {
        let statements = Parser::parse_sql(&GenericDialect, text).map_err(sql_err)?;
        let [statement] = statements.as_slice() else {
            return Err(FrontendError::Message(
                "exactly one SQL statement is supported per query".into(),
            ));
        };
        // Canonical form + fingerprint from the re-serialized AST (stable
        // across whitespace / casing differences in the surface text).
        let canonical = statement.to_string();
        let fingerprint = crate::advisor::fingerprint::fingerprint(&canonical);
        let root = lower_statement(statement)?;
        Ok(ParsedQuery {
            plan: LogicalPlan {
                root,
                snapshot_ts: None,
                vector_consistency: VectorConsistencyMode::default(),
                read_consistency: ReadConsistencyMode::default(),
            },
            canonical,
            fingerprint,
        })
    }

    fn fingerprint(&self, text: &str) -> Result<(String, u64), FrontendError> {
        let statements = Parser::parse_sql(&GenericDialect, text).map_err(sql_err)?;
        let canonical = statements
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ");
        Ok((
            canonical.clone(),
            crate::advisor::fingerprint::fingerprint(&canonical),
        ))
    }
}

fn sql_err(e: sqlparser::parser::ParserError) -> FrontendError {
    FrontendError::Message(format!("SQL parse error: {e}"))
}

fn unsupported(what: &str) -> FrontendError {
    FrontendError::Message(format!("unsupported SQL: {what}"))
}

/// Lower a single SQL statement into the neutral operator tree.
fn lower_statement(statement: &Statement) -> Result<LogicalOp, FrontendError> {
    match statement {
        Statement::Query(query) => lower_select(query),
        Statement::Insert(insert) => lower_insert(insert),
        other => Err(unsupported(&format!(
            "statement `{}`",
            first_word(&other.to_string())
        ))),
    }
}

/// `SELECT <cols> FROM <table> [WHERE <pred>]` -> NodeScan [-> Filter] -> Project.
fn lower_select(query: &SqlQuery) -> Result<LogicalOp, FrontendError> {
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Err(unsupported("only plain SELECT bodies are supported"));
    };
    if select.from.len() != 1 || !select.from[0].joins.is_empty() {
        return Err(unsupported(
            "SELECT must read exactly one table (no joins yet)",
        ));
    }
    let TableFactor::Table { name, alias, .. } = &select.from[0].relation else {
        return Err(unsupported("only a named table is supported in FROM"));
    };
    let label = object_name_string(name);
    // The bound row variable: the table alias if given, else the table name.
    let var = alias
        .as_ref()
        .map(|a| a.name.value.clone())
        .unwrap_or_else(|| label.clone());

    let scan = LogicalOp::NodeScan {
        variable: var.clone(),
        labels: vec![label],
        property_filters: Vec::new(),
    };

    let filtered = match &select.selection {
        Some(pred) => LogicalOp::Filter {
            input: Box::new(scan),
            predicate: lower_expr(pred, &var)?,
        },
        None => scan,
    };

    let items = lower_projection(&select.projection, &var)?;
    Ok(LogicalOp::Project {
        input: Box::new(filtered),
        items,
        distinct: select.distinct.is_some(),
    })
}

/// Lower the SELECT list. `*` projects the bound variable; `col [AS alias]`
/// projects `var.col`.
fn lower_projection(items: &[SelectItem], var: &str) -> Result<Vec<ProjectItem>, FrontendError> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            SelectItem::Wildcard(_) => {
                out.push(ProjectItem {
                    alias: Some(var.to_string()),
                    expr: Expr::Variable(var.to_string()),
                });
            }
            SelectItem::UnnamedExpr(expr) => {
                let alias = default_alias(expr);
                out.push(ProjectItem {
                    alias,
                    expr: lower_expr(expr, var)?,
                });
            }
            SelectItem::ExprWithAlias { expr, alias } => out.push(ProjectItem {
                alias: Some(alias.value.clone()),
                expr: lower_expr(expr, var)?,
            }),
            SelectItem::QualifiedWildcard(..) => {
                return Err(unsupported("qualified wildcard in SELECT"));
            }
        }
    }
    Ok(out)
}

/// `INSERT INTO <table> (<cols>) VALUES (...)` -> CreateNode per row.
fn lower_insert(insert: &Insert) -> Result<LogicalOp, FrontendError> {
    let label = object_name_string(&insert.table_name);
    let columns: Vec<String> = insert.columns.iter().map(|c| c.value.clone()).collect();
    if columns.is_empty() {
        return Err(unsupported("INSERT requires an explicit column list"));
    }
    let Some(source) = &insert.source else {
        return Err(unsupported("INSERT requires a VALUES source"));
    };
    let SetExpr::Values(values) = source.body.as_ref() else {
        return Err(unsupported("only INSERT ... VALUES is supported"));
    };

    // A single CreateNode chains rows through `input`; the executor materialises
    // one node per input row, so build a left-deep chain of CreateNode ops, each
    // contributing its row's properties. For the common single-row INSERT this
    // is one CreateNode over Empty.
    let mut current: Option<LogicalOp> = None;
    for row in &values.rows {
        if row.len() != columns.len() {
            return Err(unsupported(
                "INSERT VALUES row arity must match the column list",
            ));
        }
        let mut properties = Vec::with_capacity(columns.len());
        for (col, val) in columns.iter().zip(row.iter()) {
            properties.push((col.clone(), lower_value_expr(val)?));
        }
        let node = LogicalOp::CreateNode {
            input: current.take().map(Box::new),
            variable: None,
            labels: vec![label.clone()],
            properties,
        };
        current = Some(node);
    }
    current.ok_or_else(|| unsupported("INSERT requires at least one VALUES row"))
}

/// Lower a SQL scalar expression to the neutral IR. `var` is the bound row
/// variable, so a bare column identifier becomes `var.column`.
fn lower_expr(expr: &SqlExpr, var: &str) -> Result<Expr, FrontendError> {
    match expr {
        SqlExpr::Identifier(ident) => Ok(Expr::Property {
            base: Box::new(Expr::Variable(var.to_string())),
            key: ident.value.clone(),
        }),
        SqlExpr::CompoundIdentifier(parts) if parts.len() == 2 => Ok(Expr::Property {
            base: Box::new(Expr::Variable(parts[0].value.clone())),
            key: parts[1].value.clone(),
        }),
        SqlExpr::Value(v) => lower_sql_value(v),
        SqlExpr::Nested(inner) => lower_expr(inner, var),
        SqlExpr::BinaryOp { left, op, right } => {
            let op = lower_binop(op)?;
            Ok(Expr::Binary {
                left: Box::new(lower_expr(left, var)?),
                op,
                right: Box::new(lower_expr(right, var)?),
            })
        }
        other => Err(unsupported(&format!("expression `{other}`"))),
    }
}

/// Lower a value-position SQL expression (INSERT VALUES), which must be a
/// literal (no row variable in scope).
fn lower_value_expr(expr: &SqlExpr) -> Result<Expr, FrontendError> {
    match expr {
        SqlExpr::Value(v) => lower_sql_value(v),
        SqlExpr::UnaryOp {
            op: sqlparser::ast::UnaryOperator::Minus,
            expr,
        } => match lower_value_expr(expr)? {
            Expr::Literal(Value::Int(n)) => Ok(Expr::Literal(Value::Int(-n))),
            Expr::Literal(Value::Float(f)) => Ok(Expr::Literal(Value::Float(-f))),
            _ => Err(unsupported("negation of a non-numeric literal")),
        },
        other => Err(unsupported(&format!("VALUES expression `{other}`"))),
    }
}

fn lower_sql_value(v: &SqlValue) -> Result<Expr, FrontendError> {
    let value = match v {
        SqlValue::Number(n, _) => {
            if let Ok(i) = n.parse::<i64>() {
                Value::Int(i)
            } else if let Ok(f) = n.parse::<f64>() {
                Value::Float(f)
            } else {
                return Err(unsupported(&format!("numeric literal `{n}`")));
            }
        }
        SqlValue::SingleQuotedString(s) | SqlValue::DoubleQuotedString(s) => {
            Value::String(s.clone())
        }
        SqlValue::Boolean(b) => Value::Bool(*b),
        SqlValue::Null => Value::Null,
        other => return Err(unsupported(&format!("literal `{other}`"))),
    };
    Ok(Expr::Literal(value))
}

fn lower_binop(op: &sqlparser::ast::BinaryOperator) -> Result<BinOp, FrontendError> {
    use sqlparser::ast::BinaryOperator as S;
    Ok(match op {
        S::Plus => BinOp::Add,
        S::Minus => BinOp::Sub,
        S::Multiply => BinOp::Mul,
        S::Divide => BinOp::Div,
        S::Modulo => BinOp::Modulo,
        S::Eq => BinOp::Eq,
        S::NotEq => BinOp::Neq,
        S::Lt => BinOp::Lt,
        S::LtEq => BinOp::Lte,
        S::Gt => BinOp::Gt,
        S::GtEq => BinOp::Gte,
        S::And => BinOp::And,
        S::Or => BinOp::Or,
        other => return Err(unsupported(&format!("operator `{other}`"))),
    })
}

/// Default column alias for an unaliased projection: a bare column keeps its
/// name; anything else gets no alias (the executor falls back to a positional
/// name).
fn default_alias(expr: &SqlExpr) -> Option<String> {
    match expr {
        SqlExpr::Identifier(ident) => Some(ident.value.clone()),
        SqlExpr::CompoundIdentifier(parts) => parts.last().map(|p| p.value.clone()),
        _ => None,
    }
}

fn object_name_string(name: &sqlparser::ast::ObjectName) -> String {
    name.0
        .iter()
        .map(|i| i.value.clone())
        .collect::<Vec<_>>()
        .join(".")
}

fn first_word(s: &str) -> String {
    s.split_whitespace().next().unwrap_or("?").to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
