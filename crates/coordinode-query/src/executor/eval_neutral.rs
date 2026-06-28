//! Neutral expression evaluator.
//!
//! Evaluates the language-neutral [`plan::Expr`](crate::plan::expr::Expr) against
//! a row, mirroring the cypher evaluator's pure-row semantics but consuming the
//! dialect-independent IR. It reuses the shared value-operation layer
//! (`eval_binary_op` / `eval_unary_op` on neutral operators, the scalar-function
//! dispatch, and the value helpers), so there is one set of evaluation rules
//! regardless of which frontend produced the expression.
//!
//! Graph-subquery forms need the storage engine to run their embedded subplan;
//! like the cypher evaluator they resolve to `Null`/`false` here and are handled
//! by the storage-aware evaluation path.

use coordinode_core::graph::types::Value;

use super::eval::{dispatch_scalar_function, eval_binary_op, eval_unary_op, rmpv_to_value};
use super::row::Row;
use crate::plan::expr::{Expr, MapProjItem, Quantifier, StrOp};

/// Evaluate a neutral expression against a row (pure, no storage access).
///
/// Wired into the evaluation path incrementally; the cypher evaluator currently
/// reaches it via lowering at the boundary, and once the plan carries neutral
/// expressions this becomes the sole pure evaluator.
pub fn eval_neutral(expr: &Expr, row: &Row) -> Value {
    match expr {
        Expr::Literal(v) => v.clone(),
        // Parameters are substituted before execution; unresolved → Null.
        Expr::Parameter(_) => Value::Null,
        Expr::Variable(name) => row.get(name).cloned().unwrap_or(Value::Null),
        Expr::Property { base, key } => eval_property(base, key, row),
        Expr::Binary { left, op, right } => {
            let lv = eval_neutral(left, row);
            let rv = eval_neutral(right, row);
            eval_binary_op(&lv, *op, &rv)
        }
        Expr::Unary { op, operand } => {
            let v = eval_neutral(operand, row);
            eval_unary_op(*op, &v)
        }
        Expr::Call { name, args, .. } => {
            let evaluated: Vec<Value> = args.iter().map(|a| eval_neutral(a, row)).collect();
            let first_arg_var = args.first().and_then(|a| match a {
                Expr::Variable(v) => Some(v.as_str()),
                _ => None,
            });
            dispatch_scalar_function(name, evaluated, first_arg_var, row)
        }
        Expr::List(items) => Value::Array(items.iter().map(|e| eval_neutral(e, row)).collect()),
        Expr::Map(entries) => {
            let map: std::collections::BTreeMap<String, Value> = entries
                .iter()
                .map(|(k, v)| (k.clone(), eval_neutral(v, row)))
                .collect();
            Value::Map(map)
        }
        Expr::MapProjection { base, items } => {
            let mut map = std::collections::BTreeMap::new();
            for item in items {
                match item {
                    MapProjItem::Property(prop) => {
                        map.insert(prop.clone(), eval_property(base, prop, row));
                    }
                    MapProjItem::Computed(alias, value_expr) => {
                        map.insert(alias.clone(), eval_neutral(value_expr, row));
                    }
                }
            }
            Value::Map(map)
        }
        Expr::In { item, list } => {
            let val = eval_neutral(item, row);
            let list_val = eval_neutral(list, row);
            if let Value::Array(items) = list_val {
                Value::Bool(items.contains(&val))
            } else {
                Value::Null
            }
        }
        Expr::IsNull { operand, negated } => {
            let v = eval_neutral(operand, row);
            let is_null = v.is_null();
            Value::Bool(if *negated { !is_null } else { is_null })
        }
        Expr::IsTyped {
            operand,
            type_name,
            negated,
        } => {
            let v = eval_neutral(operand, row);
            let matches = match type_name.to_ascii_uppercase().as_str() {
                "INTEGER" | "INT" => matches!(v, Value::Int(_)),
                "FLOAT" => matches!(v, Value::Float(_)),
                "NUMBER" => matches!(v, Value::Int(_) | Value::Float(_)),
                "STRING" => matches!(v, Value::String(_)),
                "BOOLEAN" | "BOOL" => matches!(v, Value::Bool(_)),
                "NULL" | "NOTHING" => matches!(v, Value::Null),
                "LIST" => matches!(v, Value::Array(_)),
                "MAP" => matches!(v, Value::Map(_) | Value::Document(_)),
                _ => false,
            };
            Value::Bool(matches ^ *negated)
        }
        Expr::StringMatch { value, op, pattern } => {
            let s = eval_neutral(value, row);
            let p = eval_neutral(pattern, row);
            match (s.as_str(), p.as_str()) {
                (Some(s), Some(p)) => {
                    let result = match op {
                        StrOp::StartsWith => s.starts_with(p),
                        StrOp::EndsWith => s.ends_with(p),
                        StrOp::Contains => s.contains(p),
                        StrOp::Regex => regex::Regex::new(&format!("^(?:{p})$"))
                            .map(|re| re.is_match(s))
                            .unwrap_or(false),
                    };
                    Value::Bool(result)
                }
                _ => Value::Null,
            }
        }
        Expr::Case {
            operand,
            branches,
            otherwise,
        } => {
            if let Some(op) = operand {
                let op_val = eval_neutral(op, row);
                for (when, then) in branches {
                    if op_val == eval_neutral(when, row) {
                        return eval_neutral(then, row);
                    }
                }
            } else {
                for (when, then) in branches {
                    if eval_neutral(when, row) == Value::Bool(true) {
                        return eval_neutral(then, row);
                    }
                }
            }
            match otherwise {
                Some(el) => eval_neutral(el, row),
                None => Value::Null,
            }
        }
        Expr::Subscript { base, index } => {
            let b = eval_neutral(base, row);
            let idx = eval_neutral(index, row);
            match (b, idx) {
                (Value::Array(arr), Value::Int(i)) => {
                    let len = arr.len() as i64;
                    let real_idx = if i < 0 { len + i } else { i };
                    if real_idx >= 0 && real_idx < len {
                        arr[real_idx as usize].clone()
                    } else {
                        Value::Null
                    }
                }
                (Value::Map(map), Value::String(key)) => {
                    map.get(&key).cloned().unwrap_or(Value::Null)
                }
                _ => Value::Null,
            }
        }
        Expr::Slice { base, start, end } => match eval_neutral(base, row) {
            Value::Array(arr) => {
                let len = arr.len() as i64;
                let resolve = |bound: &Option<Box<Expr>>, default: i64| -> i64 {
                    match bound {
                        Some(e) => match eval_neutral(e, row) {
                            Value::Int(i) => {
                                let idx = if i < 0 { len + i } else { i };
                                idx.clamp(0, len)
                            }
                            _ => default,
                        },
                        None => default,
                    }
                };
                let s = resolve(start, 0);
                let e = resolve(end, len);
                if s >= e {
                    Value::Array(vec![])
                } else {
                    Value::Array(arr[s as usize..e as usize].to_vec())
                }
            }
            _ => Value::Null,
        },
        Expr::ListComprehension {
            var,
            list,
            filter,
            map,
        } => match eval_neutral(list, row) {
            Value::Array(items) => {
                let mut scratch = row.clone();
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    scratch.insert(var.clone(), item.clone());
                    let keep = match filter {
                        Some(p) => matches!(eval_neutral(p, &scratch), Value::Bool(true)),
                        None => true,
                    };
                    if keep {
                        out.push(match map {
                            Some(m) => eval_neutral(m, &scratch),
                            None => item,
                        });
                    }
                }
                Value::Array(out)
            }
            _ => Value::Null,
        },
        Expr::ListQuantifier {
            kind,
            var,
            list,
            predicate,
        } => match eval_neutral(list, row) {
            Value::Array(items) => {
                let total = items.len();
                let mut scratch = row.clone();
                let mut true_count = 0usize;
                for item in items {
                    scratch.insert(var.clone(), item);
                    if matches!(eval_neutral(predicate, &scratch), Value::Bool(true)) {
                        true_count += 1;
                    }
                }
                let result = match kind {
                    Quantifier::All => true_count == total,
                    Quantifier::Any => true_count > 0,
                    Quantifier::None => true_count == 0,
                    Quantifier::Single => true_count == 1,
                };
                Value::Bool(result)
            }
            _ => Value::Null,
        },
        Expr::Reduce {
            acc,
            init,
            var,
            list,
            step,
        } => {
            let mut acc_val = eval_neutral(init, row);
            match eval_neutral(list, row) {
                Value::Array(items) => {
                    let mut scratch = row.clone();
                    for item in items {
                        scratch.insert(acc.clone(), acc_val);
                        scratch.insert(var.clone(), item);
                        acc_val = eval_neutral(step, &scratch);
                    }
                    acc_val
                }
                Value::Null => Value::Null,
                _ => acc_val,
            }
        }
        // Graph subqueries need the storage engine (the storage-aware evaluator
        // runs the embedded subplan); the pure evaluator yields Null/false.
        Expr::ExistsSubplan(_) => Value::Bool(false),
        Expr::CountSubplan(_) => Value::Null,
        Expr::CollectSubplan { .. } => Value::Null,
        Expr::PatternComprehension { .. } => Value::Null,
        Expr::Star => Value::Null,
    }
}

/// Property access `base.key`, mirroring the cypher evaluator: collect the full
/// dot-notation path and try progressively-longer row-key prefixes, extracting
/// from a Document/Map for any remaining path.
fn eval_property(base: &Expr, key: &str, row: &Row) -> Value {
    if let Some((var_name, full_path)) = collect_property_path(base, key) {
        for split in (0..=full_path.len()).rev() {
            let row_key = if split == 0 {
                var_name.clone()
            } else {
                format!("{var_name}.{}", full_path[..split].join("."))
            };
            if let Some(val) = row.get(&row_key) {
                let remaining = &full_path[split..];
                if remaining.is_empty() {
                    return val.clone();
                }
                match val {
                    Value::Document(doc) => {
                        let path_refs: Vec<&str> = remaining.iter().map(|s| s.as_str()).collect();
                        let extracted =
                            coordinode_core::graph::document::extract_at_path(doc, &path_refs);
                        return rmpv_to_value(&extracted);
                    }
                    Value::Map(map) if remaining.len() == 1 => {
                        return map
                            .get(remaining[0].as_str())
                            .cloned()
                            .unwrap_or(Value::Null);
                    }
                    _ => continue,
                }
            }
        }
        Value::Null
    } else {
        // Non-variable base (function result, etc.): evaluate then index.
        match eval_neutral(base, row) {
            Value::Map(map) => map.get(key).cloned().unwrap_or(Value::Null),
            Value::Document(ref doc) => {
                let extracted = coordinode_core::graph::document::extract_at_path(doc, &[key]);
                rmpv_to_value(&extracted)
            }
            _ => Value::Null,
        }
    }
}

/// Collect a pure `base.key` dot-path into `(variable, path)`, or `None` if the
/// chain is not a plain variable/property chain.
fn collect_property_path(base: &Expr, outer_key: &str) -> Option<(String, Vec<String>)> {
    let mut path = vec![outer_key.to_string()];
    let mut current = base;
    loop {
        match current {
            Expr::Variable(name) => {
                path.reverse();
                return Some((name.clone(), path));
            }
            Expr::Property { base, key } => {
                path.push(key.clone());
                current = base;
            }
            _ => return None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
