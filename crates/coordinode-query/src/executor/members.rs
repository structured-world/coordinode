//! Comparing a document property by what its definition declares: a list
//! declared as unordered members is the same as another holding the same
//! members in any order.
//!
//! The declaration is read where it lives, in the definition of the node's
//! type as the statement's snapshot holds it. A row names only the type of
//! each node it binds, so rows of different types, a node bound under
//! another name, and one parameter compared with properties of different
//! types each resolve their own declaration, and no row carries a copy.

use std::collections::HashMap;
use std::sync::Arc;

use coordinode_core::budget::{BudgetStop, QueryBudget};
use coordinode_core::graph::types::Value;
use coordinode_core::schema::definition::LabelSchema;

use super::row::{Row, property};
use crate::plan::expr::Expr;

/// The definitions a statement compares document properties under, and the
/// budget the copies a comparison sorts are charged to.
#[derive(Clone, Copy)]
pub struct Declared<'e> {
    /// The definitions the statement has read, by label name.
    schemas: Option<&'e HashMap<String, Option<Arc<LabelSchema>>>>,
    budget: Option<&'e QueryBudget>,
}

impl<'e> Declared<'e> {
    /// No definitions: every document compares as one that declares
    /// nothing.
    pub const fn none() -> Self {
        Self {
            schemas: None,
            budget: None,
        }
    }

    /// The definitions in `schemas`, with comparisons charged to `budget`.
    pub fn new(
        schemas: &'e HashMap<String, Option<Arc<LabelSchema>>>,
        budget: &'e QueryBudget,
    ) -> Self {
        Self {
            schemas: Some(schemas),
            budget: Some(budget),
        }
    }

    /// The unordered lists declared at or below `fields` of the node bound
    /// to `variable`, the property first, each path cut to start there;
    /// empty when its type declares none.
    fn below(&self, row: &Row, variable: &str, fields: &[&str]) -> Vec<&'e [String]> {
        let Some(Value::String(label)) = property(row, variable, "__label__") else {
            return Vec::new();
        };
        let Some(def) = self
            .schemas
            .and_then(|schemas| schemas.get(label.as_str()))
            .and_then(Option::as_ref)
            .and_then(|schema| schema.get_property(fields[0]))
        else {
            return Vec::new();
        };
        let above = &fields[1..];
        def.unordered
            .iter()
            .filter(|path| {
                path.len() >= above.len()
                    && path.iter().zip(above).all(|(field, read)| field == read)
            })
            .map(|path| &path[above.len()..])
            .collect()
    }

    /// Compare under `sets`, the copies the comparison may sort charged to
    /// the statement while it runs: at most both values, twice over.
    fn compare(&self, a: &Value, b: &Value, sets: &[&[String]]) -> Result<bool, BudgetStop> {
        let mut held = self.budget.map(QueryBudget::empty_charge);
        if let Some(held) = &mut held {
            held.grow(2 * (a.held_bytes() + b.held_bytes()))?;
        }
        Ok(a.same_value_under(b, sets))
    }

    /// Whether the values `left` and `right` evaluated to are the same
    /// value, when one of the expressions reads a property that declares
    /// unordered members at or below what it reads: both values are then
    /// compared under that declaration. `None` when neither does, and the
    /// comparison is the ordinary one.
    ///
    /// # Errors
    ///
    /// The statement's budget does not cover the comparison.
    pub(crate) fn equality(
        &self,
        left: &Expr,
        right: &Expr,
        left_value: &Value,
        right_value: &Value,
        row: &Row,
    ) -> Result<Option<bool>, BudgetStop> {
        if self.schemas.is_none() || !(structured(left_value) && structured(right_value)) {
            return Ok(None);
        }
        for side in [left, right] {
            let Some((variable, fields)) = property_chain(side) else {
                continue;
            };
            let sets = self.below(row, variable, &fields);
            if !sets.is_empty() {
                return self.compare(left_value, right_value, &sets).map(Some);
            }
        }
        Ok(None)
    }

    /// Whether property `name` of `variable` in `row` holds `expected`,
    /// under what the property declares: what an inline `{name: value}`
    /// asks.
    ///
    /// # Errors
    ///
    /// The statement's budget does not cover the comparison.
    pub(crate) fn holds(
        &self,
        row: &Row,
        variable: &str,
        name: &str,
        actual: &Value,
        expected: &Value,
    ) -> Result<bool, BudgetStop> {
        if !(structured(actual) && structured(expected)) {
            return Ok(actual == expected);
        }
        let sets = self.below(row, variable, &[name]);
        if sets.is_empty() {
            return Ok(actual.same_value(expected));
        }
        self.compare(actual, expected, &sets)
    }
}

/// The node variables whose document properties `expr` compares: the roots
/// of the property reads standing on either side of an `=` or `<>`, at any
/// depth outside its subqueries. A statement loads the definitions of their
/// types before it evaluates `expr` over a row.
pub(crate) fn compared_variables(expr: &Expr) -> Vec<String> {
    use crate::plan::expr::{BinOp, ExprChild};
    fn walk(expr: &mut Expr, out: &mut Vec<String>) {
        if let Expr::Binary {
            left,
            op: BinOp::Eq | BinOp::Neq,
            right,
        } = &*expr
        {
            for side in [left, right] {
                if let Some((variable, _)) = property_chain(side)
                    && !out.iter().any(|known| known == variable)
                {
                    out.push(variable.to_string());
                }
            }
        }
        expr.for_each_child_mut(&mut |child| {
            if let ExprChild::Expr(child) = child {
                walk(child, out);
            }
        });
    }
    // The walk over children takes them mutably; it changes nothing.
    let mut copy = expr.clone();
    let mut out = Vec::new();
    walk(&mut copy, &mut out);
    out
}

/// The variable a property expression reads and the fields it names below
/// it, the property first: `n.body.ops` is `n` and `[body, ops]`.
fn property_chain(expr: &Expr) -> Option<(&str, Vec<&str>)> {
    let mut fields = Vec::new();
    let mut at = expr;
    loop {
        match at {
            Expr::Property { base, key } => {
                fields.push(key.as_str());
                at = base;
            }
            Expr::Variable(variable) if !fields.is_empty() => {
                fields.reverse();
                return Some((variable, fields));
            }
            _ => return None,
        }
    }
}

/// Whether `value` is a document or what one is made from.
fn structured(value: &Value) -> bool {
    matches!(value, Value::Document(_) | Value::Map(_) | Value::Array(_))
}
