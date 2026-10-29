//! The properties a statement reads of each node its scans bind.
//!
//! A scan builds one row per node with a column per property. A statement
//! that reads only some of them has the scan build only those: the node's id
//! and label columns always, the properties the statement names as
//! `var.property`, and the properties its pattern filters on. A variable the
//! statement uses as a whole node (returned, passed under another name,
//! handed to a function that reads it whole) keeps every property, and so does
//! every variable of a statement holding an operator this analysis does not
//! cover.

use std::collections::{HashMap, HashSet};

use crate::plan::expr::{Expr, MapProjItem};
use crate::planner::logical::LogicalOp;

/// For each variable whose properties can be cut, the property names a row
/// must carry; a variable absent from the map keeps every property.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeColumns {
    wanted: HashMap<String, HashSet<String>>,
}

impl NodeColumns {
    /// The properties `variable` needs, or `None` when its rows keep all.
    pub fn of(&self, variable: &str) -> Option<&HashSet<String>> {
        self.wanted.get(variable)
    }

    /// The columns `root` reads of its scanned nodes, or `None` when the
    /// plan holds an operator this analysis does not cover.
    pub fn for_plan(root: &LogicalOp) -> Option<Self> {
        let mut uses = Uses::default();
        // The final projection hands its bare variables to the caller as
        // whole nodes.
        let mut top = root;
        while let LogicalOp::Sort { input, .. }
        | LogicalOp::Limit { input, .. }
        | LogicalOp::Skip { input, .. } = top
        {
            top = input;
        }
        if let LogicalOp::Project { items, .. } = top {
            for item in items {
                if let Expr::Variable(var) = &item.expr {
                    uses.whole.insert(var.clone());
                }
            }
        }
        uses.op(root)?;
        let wanted = uses
            .scanned
            .into_iter()
            .filter(|var| !uses.whole.contains(var))
            .map(|var| {
                let properties = uses.properties.remove(&var).unwrap_or_default();
                (var, properties)
            })
            .collect();
        Some(Self { wanted })
    }
}

#[derive(Default)]
struct Uses {
    /// Variables bound by a scan.
    scanned: HashSet<String>,
    /// Properties read per variable.
    properties: HashMap<String, HashSet<String>>,
    /// Variables used as a whole node.
    whole: HashSet<String>,
}

impl Uses {
    fn read(&mut self, var: &str, property: &str) {
        self.properties
            .entry(var.to_string())
            .or_default()
            .insert(property.to_string());
    }

    /// Record `op`'s uses; `None` for an operator this analysis does not
    /// cover.
    fn op(&mut self, op: &LogicalOp) -> Option<()> {
        match op {
            LogicalOp::NodeScan {
                variable,
                property_filters,
                ..
            } => {
                self.scanned.insert(variable.clone());
                for (property, value) in property_filters {
                    self.read(variable, property);
                    self.expr(value)?;
                }
            }
            LogicalOp::IndexScan {
                variable,
                property,
                value_expr,
                ..
            } => {
                self.scanned.insert(variable.clone());
                self.read(variable, property);
                self.expr(value_expr)?;
            }
            LogicalOp::Filter { input, predicate } => {
                self.op(input)?;
                self.expr(predicate)?;
            }
            LogicalOp::Project { input, items, .. } => {
                self.op(input)?;
                for item in items {
                    match &item.expr {
                        // Every column, under its own name.
                        Expr::Star => return None,
                        // Passed on under its own name, its columns with it:
                        // what follows decides what it needs. Under another
                        // name, kept whole.
                        Expr::Variable(var) => {
                            if item.alias.as_ref().is_some_and(|alias| alias != var) {
                                self.whole.insert(var.clone());
                            }
                        }
                        expr => self.expr(expr)?,
                    }
                }
            }
            LogicalOp::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                self.op(input)?;
                for key in group_by {
                    self.expr(key)?;
                }
                for aggregate in aggregates {
                    self.expr(&aggregate.arg)?;
                    if let Some(percentile) = &aggregate.percentile_expr {
                        self.expr(percentile)?;
                    }
                }
            }
            LogicalOp::Sort { input, items } => {
                self.op(input)?;
                for item in items {
                    self.expr(&item.expr)?;
                }
            }
            LogicalOp::Limit { input, count } | LogicalOp::Skip { input, count } => {
                self.op(input)?;
                self.expr(count)?;
            }
            LogicalOp::CartesianProduct { left, right } => {
                self.op(left)?;
                self.op(right)?;
            }
            _ => return None,
        }
        Some(())
    }

    /// Record `expr`'s uses; `None` when it holds a subquery over an
    /// operator this analysis does not cover.
    fn expr(&mut self, expr: &Expr) -> Option<()> {
        match expr {
            Expr::Property { base, key } => match base.as_ref() {
                Expr::Variable(var) => self.read(var, key),
                base => self.expr(base)?,
            },
            Expr::Variable(var) => {
                self.whole.insert(var.clone());
            }
            // The id and label columns stay in every row.
            Expr::Call { name, args, .. }
                if matches!(args.as_slice(), [Expr::Variable(_)])
                    && ["id", "labels", "elementid"]
                        .iter()
                        .any(|f| name.eq_ignore_ascii_case(f)) => {}
            Expr::Call { args, .. } | Expr::List(args) => {
                for arg in args {
                    self.expr(arg)?;
                }
            }
            Expr::Map(entries) => {
                for (_, value) in entries {
                    self.expr(value)?;
                }
            }
            Expr::MapProjection { base, items } => {
                self.expr(base)?;
                for item in items {
                    match item {
                        MapProjItem::Property(_) => {}
                        MapProjItem::Computed(_, value) => self.expr(value)?,
                    }
                }
            }
            Expr::Binary { left, right, .. }
            | Expr::In {
                item: left,
                list: right,
            }
            | Expr::StringMatch {
                value: left,
                pattern: right,
                ..
            }
            | Expr::Subscript {
                base: left,
                index: right,
            } => {
                self.expr(left)?;
                self.expr(right)?;
            }
            Expr::Unary { operand, .. }
            | Expr::IsNull { operand, .. }
            | Expr::IsTyped { operand, .. } => self.expr(operand)?,
            Expr::Case {
                operand,
                branches,
                otherwise,
            } => {
                for part in operand.iter().chain(otherwise.iter()) {
                    self.expr(part)?;
                }
                for (when, then) in branches {
                    self.expr(when)?;
                    self.expr(then)?;
                }
            }
            Expr::Slice { base, start, end } => {
                self.expr(base)?;
                for bound in start.iter().chain(end.iter()) {
                    self.expr(bound)?;
                }
            }
            Expr::ListComprehension {
                list, filter, map, ..
            } => {
                self.expr(list)?;
                for part in filter.iter().chain(map.iter()) {
                    self.expr(part)?;
                }
            }
            Expr::ListQuantifier {
                list, predicate, ..
            } => {
                self.expr(list)?;
                self.expr(predicate)?;
            }
            Expr::Reduce {
                init, list, step, ..
            } => {
                self.expr(init)?;
                self.expr(list)?;
                self.expr(step)?;
            }
            Expr::ExistsSubplan(plan) | Expr::CountSubplan(plan) => self.op(&plan.root)?,
            Expr::CollectSubplan {
                subplan,
                projection: other,
            }
            | Expr::PatternComprehension {
                subplan,
                map: other,
            } => {
                self.op(&subplan.root)?;
                self.expr(other)?;
            }
            Expr::Literal(_) | Expr::Parameter(_) | Expr::Star => {}
        }
        Some(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
