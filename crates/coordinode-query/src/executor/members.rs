//! Comparing a document property by what its definition declares: a list
//! declared as unordered members is the same as another holding the same
//! members in any order.
//!
//! A node's row carries what its type declares in one hidden column, so the
//! comparison of `n.body = $value` reads the declaration of the property it
//! names without reaching for the catalog: both operands are then compared
//! under it.

use std::collections::BTreeMap;

use coordinode_core::graph::types::Value;
use coordinode_core::schema::definition::LabelSchema;

use super::row::{Row, column_entry_bytes, column_name, property};
use crate::plan::expr::Expr;

/// The hidden column of a node's row holding the unordered lists its type
/// declares: a map from property to its paths, each a list of field names.
pub(crate) const MEMBERS_COLUMN: &str = "__members__";

/// What `schema` declares, as the hidden column holds it; `None` when it
/// declares no unordered members, and the row then carries no column.
fn declared(schema: &LabelSchema) -> Option<Value> {
    let by_property: BTreeMap<String, Value> = schema
        .properties
        .iter()
        .filter(|(_, def)| !def.unordered.is_empty())
        .map(|(name, def)| {
            let paths = def
                .unordered
                .iter()
                .map(|path| Value::Array(path.iter().cloned().map(Value::String).collect()))
                .collect();
            (name.clone(), Value::Array(paths))
        })
        .collect();
    (!by_property.is_empty()).then_some(Value::Map(by_property))
}

/// Put what `schema` declares into `row` under `variable`.
pub(crate) fn insert_members_column(row: &mut Row, variable: &str, schema: &LabelSchema) {
    if let Some(declared) = declared(schema) {
        row.insert(column_name(variable, MEMBERS_COLUMN), declared);
    }
}

/// What [`insert_members_column`] adds to a row, known before it is added.
pub(crate) fn members_column_bytes(variable: &str, schema: &LabelSchema) -> u64 {
    declared(schema).map_or(0, |declared| {
        column_entry_bytes(variable, MEMBERS_COLUMN) + declared.held_bytes()
    })
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

/// The unordered lists declared at or below `fields` of `variable`, the
/// property first, each path starting there; empty when the property
/// declares none.
fn declared_below<'r>(row: &'r Row, variable: &str, fields: &[&str]) -> Vec<Vec<&'r str>> {
    let Some(Value::Map(by_property)) = property(row, variable, MEMBERS_COLUMN) else {
        return Vec::new();
    };
    let Some(Value::Array(paths)) = by_property.get(fields[0]) else {
        return Vec::new();
    };
    paths
        .iter()
        .filter_map(|path| {
            let Value::Array(path) = path else {
                return None;
            };
            let path: Vec<&str> = path
                .iter()
                .filter_map(|field| match field {
                    Value::String(field) => Some(field.as_str()),
                    _ => None,
                })
                .collect();
            let above = &fields[1..];
            let reaches = path.len() >= above.len()
                && path.iter().zip(above).all(|(field, read)| field == read);
            reaches.then(|| path[above.len()..].to_vec())
        })
        .collect()
}

/// Whether `value` is a document or what one is made from.
fn structured(value: &Value) -> bool {
    matches!(value, Value::Document(_) | Value::Map(_) | Value::Array(_))
}

/// Whether the values `left` and `right` evaluated to are the same value,
/// when one of the expressions reads a property that declares unordered
/// members at or below what it reads: both values are then compared under
/// that declaration. `None` when neither does, and the comparison is the
/// ordinary one.
pub(crate) fn declared_equality(
    left: &Expr,
    right: &Expr,
    left_value: &Value,
    right_value: &Value,
    row: &Row,
) -> Option<bool> {
    if !(structured(left_value) && structured(right_value)) {
        return None;
    }
    [left, right].into_iter().find_map(|side| {
        let (variable, fields) = property_chain(side)?;
        let sets = declared_below(row, variable, &fields);
        (!sets.is_empty()).then(|| {
            let sets: Vec<&[&str]> = sets.iter().map(Vec::as_slice).collect();
            left_value.same_value_under(right_value, &sets)
        })
    })
}

/// Whether property `name` of `variable` in `row` holds `expected`, under
/// what the property declares: what an inline `{name: value}` asks.
pub(crate) fn holds(
    row: &Row,
    variable: &str,
    name: &str,
    actual: &Value,
    expected: &Value,
) -> bool {
    if !(structured(actual) && structured(expected)) {
        return actual == expected;
    }
    let sets = declared_below(row, variable, &[name]);
    let sets: Vec<&[&str]> = sets.iter().map(Vec::as_slice).collect();
    actual.same_value_under(expected, &sets)
}
