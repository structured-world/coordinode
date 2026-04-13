//! Expression evaluator: evaluates Cypher AST expressions against a Row.

use coordinode_core::graph::types::Value;

use super::row::Row;
use crate::cypher::ast::*;

/// Evaluate an expression against a row, producing a Value.
pub fn eval_expr(expr: &Expr, row: &Row) -> Value {
    match expr {
        Expr::Literal(v) => v.clone(),
        Expr::Parameter(_) => {
            // Parameters are resolved before execution; treat as Null if unresolved.
            Value::Null
        }
        Expr::Variable(name) => row.get(name).cloned().unwrap_or(Value::Null),
        Expr::PropertyAccess { expr, property } => {
            // Optimization: collect the full dot-notation path and extract in one pass.
            // For `n.config.network.ssid`: collect path ["config", "network", "ssid"],
            // look up row["n.config"] as Document, then extract_at_path with remaining
            // ["network", "ssid"] in a single traversal (no intermediate allocations).
            if let Some((var_name, full_path)) = collect_property_path(expr, property) {
                // Try row lookup with progressively longer prefixes.
                // For path ["config", "network", "ssid"]:
                //   1. Try row["n.config.network.ssid"] (flat property)
                //   2. Try row["n.config.network"] → extract ["ssid"]
                //   3. Try row["n.config"] → extract ["network", "ssid"]
                //   4. Try row["n"] → extract ["config", "network", "ssid"]
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
                                // Single-pass extraction on already-deserialized rmpv::Value.
                                // Collects full remaining path and traverses in one call
                                // instead of level-by-level recursive eval_expr.
                                // extract_at_path_bytes is available for future storage-level
                                // pushdown when raw bytes are passed directly.
                                let path_refs: Vec<&str> =
                                    remaining.iter().map(|s| s.as_str()).collect();
                                let extracted = coordinode_core::graph::document::extract_at_path(
                                    doc, &path_refs,
                                );
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
                // Non-standard expression (function call result, etc.) — evaluate normally
                let inner = eval_expr(expr, row);
                match inner {
                    Value::Map(map) => map.get(property).cloned().unwrap_or(Value::Null),
                    Value::Document(ref doc) => {
                        let extracted =
                            coordinode_core::graph::document::extract_at_path(doc, &[property]);
                        rmpv_to_value(&extracted)
                    }
                    _ => Value::Null,
                }
            }
        }
        Expr::BinaryOp { left, op, right } => {
            let lv = eval_expr(left, row);
            let rv = eval_expr(right, row);
            eval_binary_op(&lv, *op, &rv)
        }
        Expr::UnaryOp { op, expr } => {
            let v = eval_expr(expr, row);
            eval_unary_op(*op, &v)
        }
        Expr::FunctionCall { name, args, .. } => {
            // Aggregation functions are handled at the Aggregate operator level,
            // not here. Scalar functions evaluated inline.
            eval_scalar_function(name, args, row)
        }
        Expr::List(items) => {
            let values: Vec<Value> = items.iter().map(|e| eval_expr(e, row)).collect();
            Value::Array(values)
        }
        Expr::MapLiteral(entries) => {
            let map: std::collections::BTreeMap<String, Value> = entries
                .iter()
                .map(|(k, v)| (k.clone(), eval_expr(v, row)))
                .collect();
            Value::Map(map)
        }
        Expr::MapProjection { expr, items } => {
            // Build a map from the base expression's properties.
            // `.name` shorthand → PropertyAccess(expr, "name")
            // `alias: value_expr` → evaluate value_expr
            let mut map = std::collections::BTreeMap::new();
            for item in items {
                match item {
                    MapProjectionItem::Property(prop) => {
                        let access = Expr::PropertyAccess {
                            expr: expr.clone(),
                            property: prop.clone(),
                        };
                        map.insert(prop.clone(), eval_expr(&access, row));
                    }
                    MapProjectionItem::Computed(alias, value_expr) => {
                        map.insert(alias.clone(), eval_expr(value_expr, row));
                    }
                }
            }
            Value::Map(map)
        }
        Expr::In { expr, list } => {
            let val = eval_expr(expr, row);
            let list_val = eval_expr(list, row);
            if let Value::Array(items) = list_val {
                Value::Bool(items.contains(&val))
            } else {
                Value::Null
            }
        }
        Expr::IsNull { expr, negated } => {
            let v = eval_expr(expr, row);
            let is_null = v.is_null();
            Value::Bool(if *negated { !is_null } else { is_null })
        }
        Expr::StringMatch { expr, op, pattern } => {
            let s = eval_expr(expr, row);
            let p = eval_expr(pattern, row);
            match (s.as_str(), p.as_str()) {
                (Some(s), Some(p)) => {
                    let result = match op {
                        StringOp::StartsWith => s.starts_with(p),
                        StringOp::EndsWith => s.ends_with(p),
                        StringOp::Contains => s.contains(p),
                    };
                    Value::Bool(result)
                }
                _ => Value::Null,
            }
        }
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(ref op) = operand {
                // Simple CASE: compare operand to each WHEN value
                let op_val = eval_expr(op, row);
                for (when, then) in when_clauses {
                    let when_val = eval_expr(when, row);
                    if op_val == when_val {
                        return eval_expr(then, row);
                    }
                }
            } else {
                // Generic CASE: evaluate each WHEN as boolean
                for (when, then) in when_clauses {
                    let when_val = eval_expr(when, row);
                    if when_val == Value::Bool(true) {
                        return eval_expr(then, row);
                    }
                }
            }
            if let Some(ref el) = else_clause {
                eval_expr(el, row)
            } else {
                Value::Null
            }
        }
        // Pattern predicates require storage access — evaluated in the executor
        // (eval_pattern_predicate), not in the pure expression evaluator. If we
        // reach this branch it means the predicate was not intercepted by the
        // storage-aware Filter path, so we conservatively return false.
        Expr::PatternPredicate(_) => Value::Bool(false),

        // Subscript / index access: expr[index]
        //
        // List: list[n] → element at index n (negative = from end). Out of bounds → null.
        // Map:  map["key"] → value at key. Missing key → null.
        // Anything else → null.
        Expr::Subscript { expr, index } => {
            let base = eval_expr(expr, row);
            let idx = eval_expr(index, row);
            match (base, idx) {
                (Value::Array(arr), Value::Int(i)) => {
                    let len = arr.len() as i64;
                    // Support negative indexing: -1 → last element.
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

        Expr::Star => Value::Null,
    }
}

/// Evaluate a binary operation.
pub(crate) fn eval_binary_op(left: &Value, op: BinaryOperator, right: &Value) -> Value {
    match op {
        // Arithmetic
        BinaryOperator::Add => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a.saturating_add(*b)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 + b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a + *b as f64),
            (Value::String(a), Value::String(b)) => Value::String(format!("{a}{b}")),
            _ => Value::Null,
        },
        BinaryOperator::Sub => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a.saturating_sub(*b)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 - b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a - *b as f64),
            _ => Value::Null,
        },
        BinaryOperator::Mul => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a.saturating_mul(*b)),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 * b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a * *b as f64),
            _ => Value::Null,
        },
        BinaryOperator::Div => match (left, right) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a / b),
            (Value::Float(a), Value::Float(b)) if *b != 0.0 => Value::Float(a / b),
            (Value::Int(a), Value::Float(b)) if *b != 0.0 => Value::Float(*a as f64 / b),
            (Value::Float(a), Value::Int(b)) if *b != 0 => Value::Float(a / *b as f64),
            _ => Value::Null,
        },
        BinaryOperator::Modulo => match (left, right) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a % b),
            _ => Value::Null,
        },

        // Comparison
        BinaryOperator::Eq => Value::Bool(left == right),
        BinaryOperator::Neq => Value::Bool(left != right),
        BinaryOperator::Lt => {
            Value::Bool(compare_values(left, right) == Some(std::cmp::Ordering::Less))
        }
        BinaryOperator::Lte => {
            let cmp = compare_values(left, right);
            Value::Bool(matches!(
                cmp,
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ))
        }
        BinaryOperator::Gt => {
            Value::Bool(compare_values(left, right) == Some(std::cmp::Ordering::Greater))
        }
        BinaryOperator::Gte => {
            let cmp = compare_values(left, right);
            Value::Bool(matches!(
                cmp,
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            ))
        }

        // Logical
        BinaryOperator::And => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a && *b),
            _ => Value::Null,
        },
        BinaryOperator::Or => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a || *b),
            _ => Value::Null,
        },
        BinaryOperator::Xor => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a ^ *b),
            _ => Value::Null,
        },
    }
}

/// Compare two values for ordering.
/// Convert an rmpv::Value to a CoordiNode Value.
///
/// Used for dot-notation property access on Document values.
/// Map/Array results stay as Document to allow continued path traversal.
/// Scalar results are converted to the corresponding Value variant.
/// Collect the full dot-notation path from a nested PropertyAccess chain.
///
/// For `PropertyAccess(PropertyAccess(Variable("n"), "config"), "network")` with
/// outer property "ssid", returns `Some(("n", ["config", "network", "ssid"]))`.
///
/// Returns `None` if the expression chain isn't a pure PropertyAccess/Variable chain
/// (e.g., if it contains function calls or other complex expressions).
fn collect_property_path(expr: &Expr, outer_property: &str) -> Option<(String, Vec<String>)> {
    let mut path = vec![outer_property.to_string()];
    let mut current = expr;

    loop {
        match current {
            Expr::Variable(name) => {
                path.reverse();
                return Some((name.clone(), path));
            }
            Expr::PropertyAccess { expr, property } => {
                path.push(property.clone());
                current = expr;
            }
            _ => return None,
        }
    }
}

fn rmpv_to_value(v: &rmpv::Value) -> Value {
    match v {
        rmpv::Value::Nil => Value::Null,
        rmpv::Value::Boolean(b) => Value::Bool(*b),
        rmpv::Value::Integer(i) => {
            if let Some(n) = i.as_i64() {
                Value::Int(n)
            } else if i.as_u64().is_some() {
                // u64 that doesn't fit i64 — store as Document to preserve
                Value::Document(v.clone())
            } else {
                Value::Null
            }
        }
        rmpv::Value::F32(f) => Value::Float(f64::from(*f)),
        rmpv::Value::F64(f) => Value::Float(*f),
        rmpv::Value::String(s) => Value::String(s.as_str().unwrap_or_default().to_string()),
        // Compound types stay as Document for continued traversal
        rmpv::Value::Map(_) | rmpv::Value::Array(_) => Value::Document(v.clone()),
        rmpv::Value::Binary(b) => Value::Binary(b.clone()),
        rmpv::Value::Ext(_, _) => Value::Document(v.clone()),
    }
}

fn compare_values(left: &Value, right: &Value) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
        (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

/// Evaluate a unary operation.
pub(crate) fn eval_unary_op(op: UnaryOperator, val: &Value) -> Value {
    match op {
        UnaryOperator::Not => match val {
            Value::Bool(b) => Value::Bool(!b),
            _ => Value::Null,
        },
        UnaryOperator::Neg => match val {
            Value::Int(n) => Value::Int(-n),
            Value::Float(f) => Value::Float(-f),
            _ => Value::Null,
        },
    }
}

/// Coerce a Value to Vec<f32> for vector operations.
/// Accepts both Value::Vector and Value::Array (of Float/Int values).
fn coerce_to_vector(val: Option<&Value>) -> Option<Vec<f32>> {
    match val {
        Some(Value::Vector(v)) => Some(v.clone()),
        Some(Value::Array(arr)) => {
            let mut vec = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    Value::Float(f) => vec.push(*f as f32),
                    Value::Int(i) => vec.push(*i as f32),
                    _ => return None,
                }
            }
            Some(vec)
        }
        _ => None,
    }
}

/// Coerce two values to vector pairs for distance/similarity functions.
fn coerce_vector_pair(
    a: Option<&Value>,
    b: Option<&Value>,
) -> (Option<Vec<f32>>, Option<Vec<f32>>) {
    (coerce_to_vector(a), coerce_to_vector(b))
}

/// Evaluate scalar (non-aggregate) functions.
fn eval_scalar_function(name: &str, args: &[Expr], row: &Row) -> Value {
    let evaluated: Vec<Value> = args.iter().map(|a| eval_expr(a, row)).collect();

    match name {
        "coalesce" => evaluated
            .into_iter()
            .find(|v| !v.is_null())
            .unwrap_or(Value::Null),
        "toString" => match evaluated.first() {
            Some(Value::Int(n)) => Value::String(n.to_string()),
            Some(Value::Float(f)) => Value::String(f.to_string()),
            Some(Value::Bool(b)) => Value::String(b.to_string()),
            Some(Value::String(s)) => Value::String(s.clone()),
            _ => Value::Null,
        },
        "size" => match evaluated.first() {
            Some(Value::String(s)) => Value::Int(s.len() as i64),
            Some(Value::Array(a)) => Value::Int(a.len() as i64),
            _ => Value::Null,
        },
        "type" => {
            // type(r) → relationship type string.
            // The executor stores edge type as `r.__type__` in the row.
            // Extract the variable name from the first argument, then look up
            // `<variable>.__type__` in the row.
            if let Some(Expr::Variable(var)) = args.first() {
                let key = format!("{var}.__type__");
                row.get(&key).cloned().unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }
        "labels" => {
            // labels(n) → list of label strings for a node.
            // The executor stores primary label as `n.__label__` in the row.
            // CoordiNode nodes currently have exactly one label.
            if let Some(Expr::Variable(var)) = args.first() {
                let key = format!("{var}.__label__");
                match row.get(&key) {
                    Some(Value::String(l)) => Value::Array(vec![Value::String(l.clone())]),
                    _ => Value::Array(vec![]),
                }
            } else {
                Value::Array(vec![])
            }
        }
        "now" => {
            // Return current timestamp in microseconds
            Value::Timestamp(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_micros() as i64)
                    .unwrap_or(0),
            )
        }
        // Vector distance/similarity functions
        "vector_distance" => {
            // vector_distance(a, b) → L2 distance
            let (a, b) = coerce_vector_pair(evaluated.first(), evaluated.get(1));
            match (a, b) {
                (Some(a), Some(b)) if a.len() == b.len() => {
                    Value::Float(coordinode_vector::metrics::euclidean_distance(&a, &b) as f64)
                }
                _ => Value::Null,
            }
        }
        "vector_similarity" => {
            // vector_similarity(a, b) → cosine similarity
            let (a, b) = coerce_vector_pair(evaluated.first(), evaluated.get(1));
            match (a, b) {
                (Some(a), Some(b)) if a.len() == b.len() => {
                    Value::Float(coordinode_vector::metrics::cosine_similarity(&a, &b) as f64)
                }
                _ => Value::Null,
            }
        }
        "vector_dot" => {
            // vector_dot(a, b) → dot product
            let (a, b) = coerce_vector_pair(evaluated.first(), evaluated.get(1));
            match (a, b) {
                (Some(a), Some(b)) if a.len() == b.len() => {
                    Value::Float(coordinode_vector::metrics::dot_product(&a, &b) as f64)
                }
                _ => Value::Null,
            }
        }
        "vector_manhattan" => {
            // vector_manhattan(a, b) → L1 distance
            match (evaluated.first(), evaluated.get(1)) {
                (Some(Value::Vector(a)), Some(Value::Vector(b))) if a.len() == b.len() => {
                    Value::Float(coordinode_vector::metrics::manhattan_distance(a, b) as f64)
                }
                _ => Value::Null,
            }
        }
        // text_score(field, query) → retrieves pre-computed BM25 score from __text_score__ column.
        // The score is stored by TextFilter executor during WHERE evaluation.
        "text_score" => {
            // Score was pre-stored in the row by execute_text_filter
            row.get("__text_score__")
                .cloned()
                .unwrap_or(Value::Float(0.0))
        }
        // text_match(field, query) → boolean. Used in WHERE clause.
        // When used in RETURN, checks if __text_score__ was set by TextFilter.
        "text_match" => {
            if row.contains_key("__text_score__") {
                Value::Bool(true)
            } else {
                Value::Bool(false)
            }
        }
        // encrypted_match(field, token) → boolean. Used in WHERE clause.
        // When used in RETURN, the row has already been filtered by EncryptedFilter,
        // so if the row is present, it matched. Always returns true for surviving rows.
        "encrypted_match" => Value::Bool(true),
        // Spatial functions
        "point" => {
            // point({latitude: X, longitude: Y}) → Geo(Point { lat, lon })
            // Accepts a map literal with latitude/longitude keys.
            match evaluated.first() {
                Some(Value::Map(map)) => {
                    let lat = map.get("latitude").and_then(|v| match v {
                        Value::Float(f) => Some(*f),
                        Value::Int(i) => Some(*i as f64),
                        _ => None,
                    });
                    let lon = map.get("longitude").and_then(|v| match v {
                        Value::Float(f) => Some(*f),
                        Value::Int(i) => Some(*i as f64),
                        _ => None,
                    });
                    match (lat, lon) {
                        (Some(lat), Some(lon)) => {
                            Value::Geo(coordinode_core::graph::types::GeoValue::Point { lat, lon })
                        }
                        _ => Value::Null,
                    }
                }
                _ => Value::Null,
            }
        }
        "point.distance" => {
            // point.distance(point1, point2) → distance in meters (Haversine)
            match (evaluated.first(), evaluated.get(1)) {
                (
                    Some(Value::Geo(coordinode_core::graph::types::GeoValue::Point {
                        lat: lat1,
                        lon: lon1,
                    })),
                    Some(Value::Geo(coordinode_core::graph::types::GeoValue::Point {
                        lat: lat2,
                        lon: lon2,
                    })),
                ) => Value::Float(haversine_distance(*lat1, *lon1, *lat2, *lon2)),
                _ => Value::Null,
            }
        }
        _ => Value::Null, // Unknown function returns null
    }
}

/// Haversine distance between two WGS84 points, in meters.
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_M: f64 = 6_371_000.0;
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let a =
        (d_lat / 2.0).sin().powi(2) + lat1_rad.cos() * lat2_rad.cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    EARTH_RADIUS_M * c
}

/// Check if a value is truthy (for WHERE filter evaluation).
pub fn is_truthy(val: &Value) -> bool {
    match val {
        Value::Bool(b) => *b,
        Value::Null => false,
        _ => true,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn empty_row() -> Row {
        Row::new()
    }

    fn sample_row() -> Row {
        let mut row = Row::new();
        row.insert("n.name".into(), Value::String("Alice".into()));
        row.insert("n.age".into(), Value::Int(30));
        row.insert("n.score".into(), Value::Float(0.95));
        row.insert("n.active".into(), Value::Bool(true));
        row.insert("n".into(), Value::String("node_ref".into()));
        row
    }

    #[test]
    fn eval_literal() {
        let v = eval_expr(&Expr::Literal(Value::Int(42)), &empty_row());
        assert_eq!(v, Value::Int(42));
    }

    #[test]
    fn eval_variable() {
        let row = sample_row();
        let v = eval_expr(&Expr::Variable("n".into()), &row);
        assert_eq!(v, Value::String("node_ref".into()));
    }

    #[test]
    fn eval_undefined_variable() {
        let v = eval_expr(&Expr::Variable("x".into()), &empty_row());
        assert_eq!(v, Value::Null);
    }

    #[test]
    fn eval_property_access() {
        let row = sample_row();
        let expr = Expr::PropertyAccess {
            expr: Box::new(Expr::Variable("n".into())),
            property: "name".into(),
        };
        let v = eval_expr(&expr, &row);
        assert_eq!(v, Value::String("Alice".into()));
    }

    #[test]
    fn eval_document_dot_notation_single_level() {
        let doc = rmpv::Value::Map(vec![(
            rmpv::Value::String("firmware".into()),
            rmpv::Value::String("2.1.3".into()),
        )]);
        let mut row = Row::new();
        row.insert("n.config".into(), Value::Document(doc));

        // n.config.firmware → "2.1.3"
        let expr = Expr::PropertyAccess {
            expr: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".into())),
                property: "config".into(),
            }),
            property: "firmware".into(),
        };
        let v = eval_expr(&expr, &row);
        assert_eq!(v, Value::String("2.1.3".into()));
    }

    #[test]
    fn eval_document_dot_notation_three_levels() {
        let doc = rmpv::Value::Map(vec![(
            rmpv::Value::String("network".into()),
            rmpv::Value::Map(vec![(
                rmpv::Value::String("ssid".into()),
                rmpv::Value::String("home".into()),
            )]),
        )]);
        let mut row = Row::new();
        row.insert("n.config".into(), Value::Document(doc));

        // n.config.network.ssid → "home"
        let expr = Expr::PropertyAccess {
            expr: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::PropertyAccess {
                    expr: Box::new(Expr::Variable("n".into())),
                    property: "config".into(),
                }),
                property: "network".into(),
            }),
            property: "ssid".into(),
        };
        let v = eval_expr(&expr, &row);
        assert_eq!(v, Value::String("home".into()));
    }

    #[test]
    fn eval_document_missing_key_returns_null() {
        let doc = rmpv::Value::Map(vec![(
            rmpv::Value::String("a".into()),
            rmpv::Value::Integer(1.into()),
        )]);
        let mut row = Row::new();
        row.insert("n.data".into(), Value::Document(doc));

        // n.data.nonexistent → Null
        let expr = Expr::PropertyAccess {
            expr: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".into())),
                property: "data".into(),
            }),
            property: "nonexistent".into(),
        };
        let v = eval_expr(&expr, &row);
        assert_eq!(v, Value::Null);
    }

    #[test]
    fn eval_document_numeric_returns_value() {
        let doc = rmpv::Value::Map(vec![(
            rmpv::Value::String("count".into()),
            rmpv::Value::Integer(42.into()),
        )]);
        let mut row = Row::new();
        row.insert("n.stats".into(), Value::Document(doc));

        // n.stats.count → Int(42)
        let expr = Expr::PropertyAccess {
            expr: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".into())),
                property: "stats".into(),
            }),
            property: "count".into(),
        };
        let v = eval_expr(&expr, &row);
        assert_eq!(v, Value::Int(42));
    }

    #[test]
    fn eval_arithmetic_add() {
        let row = sample_row();
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".into())),
                property: "age".into(),
            }),
            op: BinaryOperator::Add,
            right: Box::new(Expr::Literal(Value::Int(5))),
        };
        let v = eval_expr(&expr, &row);
        assert_eq!(v, Value::Int(35));
    }

    #[test]
    fn eval_comparison_gt() {
        let row = sample_row();
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".into())),
                property: "age".into(),
            }),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Literal(Value::Int(25))),
        };
        let v = eval_expr(&expr, &row);
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn eval_logical_and() {
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Literal(Value::Bool(true))),
            op: BinaryOperator::And,
            right: Box::new(Expr::Literal(Value::Bool(false))),
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Bool(false));
    }

    #[test]
    fn eval_not() {
        let expr = Expr::UnaryOp {
            op: UnaryOperator::Not,
            expr: Box::new(Expr::Literal(Value::Bool(true))),
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Bool(false));
    }

    #[test]
    fn eval_neg() {
        let expr = Expr::UnaryOp {
            op: UnaryOperator::Neg,
            expr: Box::new(Expr::Literal(Value::Int(42))),
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Int(-42));
    }

    #[test]
    fn eval_string_starts_with() {
        let row = sample_row();
        let expr = Expr::StringMatch {
            expr: Box::new(Expr::PropertyAccess {
                expr: Box::new(Expr::Variable("n".into())),
                property: "name".into(),
            }),
            op: StringOp::StartsWith,
            pattern: Box::new(Expr::Literal(Value::String("Ali".into()))),
        };
        let v = eval_expr(&expr, &row);
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn eval_is_null() {
        let expr = Expr::IsNull {
            expr: Box::new(Expr::Literal(Value::Null)),
            negated: false,
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn eval_is_not_null() {
        let expr = Expr::IsNull {
            expr: Box::new(Expr::Literal(Value::Int(1))),
            negated: true,
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn eval_in_list() {
        let expr = Expr::In {
            expr: Box::new(Expr::Literal(Value::Int(2))),
            list: Box::new(Expr::List(vec![
                Expr::Literal(Value::Int(1)),
                Expr::Literal(Value::Int(2)),
                Expr::Literal(Value::Int(3)),
            ])),
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn eval_case_generic() {
        let row = sample_row();
        let expr = Expr::Case {
            operand: None,
            when_clauses: vec![
                (
                    Expr::BinaryOp {
                        left: Box::new(Expr::PropertyAccess {
                            expr: Box::new(Expr::Variable("n".into())),
                            property: "age".into(),
                        }),
                        op: BinaryOperator::Lt,
                        right: Box::new(Expr::Literal(Value::Int(18))),
                    },
                    Expr::Literal(Value::String("minor".into())),
                ),
                (
                    Expr::Literal(Value::Bool(true)),
                    Expr::Literal(Value::String("adult".into())),
                ),
            ],
            else_clause: None,
        };
        let v = eval_expr(&expr, &row);
        assert_eq!(v, Value::String("adult".into()));
    }

    #[test]
    fn eval_list_literal() {
        let expr = Expr::List(vec![
            Expr::Literal(Value::Int(1)),
            Expr::Literal(Value::Int(2)),
        ]);
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Array(vec![Value::Int(1), Value::Int(2)]));
    }

    #[test]
    fn eval_coalesce() {
        let expr = Expr::FunctionCall {
            name: "coalesce".into(),
            args: vec![
                Expr::Literal(Value::Null),
                Expr::Literal(Value::String("fallback".into())),
            ],
            distinct: false,
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::String("fallback".into()));
    }

    #[test]
    fn eval_size_string() {
        let expr = Expr::FunctionCall {
            name: "size".into(),
            args: vec![Expr::Literal(Value::String("hello".into()))],
            distinct: false,
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Int(5));
    }

    #[test]
    fn eval_string_concat() {
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Literal(Value::String("hello ".into()))),
            op: BinaryOperator::Add,
            right: Box::new(Expr::Literal(Value::String("world".into()))),
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::String("hello world".into()));
    }

    #[test]
    fn eval_div_by_zero() {
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Literal(Value::Int(10))),
            op: BinaryOperator::Div,
            right: Box::new(Expr::Literal(Value::Int(0))),
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Null);
    }

    #[test]
    fn eval_mixed_int_float() {
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Literal(Value::Int(10))),
            op: BinaryOperator::Mul,
            right: Box::new(Expr::Literal(Value::Float(1.5))),
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Float(15.0));
    }

    #[test]
    fn is_truthy_values() {
        assert!(is_truthy(&Value::Bool(true)));
        assert!(!is_truthy(&Value::Bool(false)));
        assert!(!is_truthy(&Value::Null));
        assert!(is_truthy(&Value::Int(0))); // non-null is truthy
        assert!(is_truthy(&Value::String("".into()))); // non-null is truthy
    }

    // ====== Vector functions ======

    #[test]
    fn eval_vector_distance() {
        let expr = Expr::FunctionCall {
            name: "vector_distance".into(),
            args: vec![
                Expr::Literal(Value::Vector(vec![1.0, 0.0, 0.0])),
                Expr::Literal(Value::Vector(vec![0.0, 1.0, 0.0])),
            ],
            distinct: false,
        };
        let v = eval_expr(&expr, &empty_row());
        if let Value::Float(d) = v {
            // L2 distance between (1,0,0) and (0,1,0) = sqrt(2) ≈ 1.414
            assert!((d - std::f64::consts::SQRT_2).abs() < 0.01);
        } else {
            panic!("expected Float, got {v:?}");
        }
    }

    #[test]
    fn eval_vector_similarity() {
        let expr = Expr::FunctionCall {
            name: "vector_similarity".into(),
            args: vec![
                Expr::Literal(Value::Vector(vec![1.0, 0.0])),
                Expr::Literal(Value::Vector(vec![1.0, 0.0])),
            ],
            distinct: false,
        };
        let v = eval_expr(&expr, &empty_row());
        if let Value::Float(s) = v {
            assert!((s - 1.0).abs() < 0.01); // Identical vectors → cosine 1.0
        } else {
            panic!("expected Float, got {v:?}");
        }
    }

    #[test]
    fn eval_vector_similarity_orthogonal() {
        let expr = Expr::FunctionCall {
            name: "vector_similarity".into(),
            args: vec![
                Expr::Literal(Value::Vector(vec![1.0, 0.0])),
                Expr::Literal(Value::Vector(vec![0.0, 1.0])),
            ],
            distinct: false,
        };
        let v = eval_expr(&expr, &empty_row());
        if let Value::Float(s) = v {
            assert!(s.abs() < 0.01); // Orthogonal → cosine 0.0
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn eval_vector_dot() {
        let expr = Expr::FunctionCall {
            name: "vector_dot".into(),
            args: vec![
                Expr::Literal(Value::Vector(vec![1.0, 2.0, 3.0])),
                Expr::Literal(Value::Vector(vec![4.0, 5.0, 6.0])),
            ],
            distinct: false,
        };
        let v = eval_expr(&expr, &empty_row());
        if let Value::Float(d) = v {
            assert!((d - 32.0).abs() < 0.01); // 1*4 + 2*5 + 3*6 = 32
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn eval_vector_manhattan() {
        let expr = Expr::FunctionCall {
            name: "vector_manhattan".into(),
            args: vec![
                Expr::Literal(Value::Vector(vec![1.0, 2.0, 3.0])),
                Expr::Literal(Value::Vector(vec![4.0, 6.0, 8.0])),
            ],
            distinct: false,
        };
        let v = eval_expr(&expr, &empty_row());
        if let Value::Float(d) = v {
            assert!((d - 12.0).abs() < 0.01); // |1-4| + |2-6| + |3-8| = 12
        } else {
            panic!("expected Float");
        }
    }

    #[test]
    fn eval_vector_distance_dim_mismatch() {
        let expr = Expr::FunctionCall {
            name: "vector_distance".into(),
            args: vec![
                Expr::Literal(Value::Vector(vec![1.0, 0.0])),
                Expr::Literal(Value::Vector(vec![1.0, 0.0, 0.0])),
            ],
            distinct: false,
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Null); // Dimension mismatch → null
    }

    #[test]
    fn eval_vector_distance_non_vector() {
        let expr = Expr::FunctionCall {
            name: "vector_distance".into(),
            args: vec![
                Expr::Literal(Value::String("not a vector".into())),
                Expr::Literal(Value::Vector(vec![1.0])),
            ],
            distinct: false,
        };
        let v = eval_expr(&expr, &empty_row());
        assert_eq!(v, Value::Null); // Non-vector → null
    }

    #[test]
    fn vector_in_where_clause() {
        // Simulate: WHERE vector_distance(n.embedding, $query) < 0.5
        let mut row = Row::new();
        row.insert("n.embedding".into(), Value::Vector(vec![1.0, 0.0, 0.0]));

        let expr = Expr::BinaryOp {
            left: Box::new(Expr::FunctionCall {
                name: "vector_distance".into(),
                args: vec![
                    Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "embedding".into(),
                    },
                    Expr::Literal(Value::Vector(vec![1.0, 0.0, 0.0])),
                ],
                distinct: false,
            }),
            op: BinaryOperator::Lt,
            right: Box::new(Expr::Literal(Value::Float(0.5))),
        };

        let v = eval_expr(&expr, &row);
        assert_eq!(v, Value::Bool(true)); // Distance 0 < 0.5
    }

    // --- Spatial function tests ---

    /// point() constructs a GeoValue from a map with latitude/longitude.
    #[test]
    fn point_constructor() {
        let row = Row::new();
        let expr = Expr::FunctionCall {
            name: "point".to_string(),
            args: vec![Expr::MapLiteral(vec![
                ("latitude".to_string(), Expr::Literal(Value::Float(40.7128))),
                (
                    "longitude".to_string(),
                    Expr::Literal(Value::Float(-74.006)),
                ),
            ])],
            distinct: false,
        };

        let result = eval_expr(&expr, &row);
        assert!(
            matches!(
                result,
                Value::Geo(coordinode_core::graph::types::GeoValue::Point { .. })
            ),
            "point() should return Geo(Point), got: {result:?}"
        );
    }

    /// point.distance() computes Haversine distance between two points.
    #[test]
    fn point_distance_haversine() {
        let row = Row::new();
        // New York (40.7128, -74.0060) to London (51.5074, -0.1278) ≈ 5,570 km
        let expr = Expr::FunctionCall {
            name: "point.distance".to_string(),
            args: vec![
                Expr::FunctionCall {
                    name: "point".to_string(),
                    args: vec![Expr::MapLiteral(vec![
                        ("latitude".to_string(), Expr::Literal(Value::Float(40.7128))),
                        (
                            "longitude".to_string(),
                            Expr::Literal(Value::Float(-74.006)),
                        ),
                    ])],
                    distinct: false,
                },
                Expr::FunctionCall {
                    name: "point".to_string(),
                    args: vec![Expr::MapLiteral(vec![
                        ("latitude".to_string(), Expr::Literal(Value::Float(51.5074))),
                        (
                            "longitude".to_string(),
                            Expr::Literal(Value::Float(-0.1278)),
                        ),
                    ])],
                    distinct: false,
                },
            ],
            distinct: false,
        };

        let result = eval_expr(&expr, &row);
        if let Value::Float(dist) = result {
            // NY to London ≈ 5,570,000 meters ± 1%
            assert!(
                (5_500_000.0..5_650_000.0).contains(&dist),
                "NY to London should be ~5,570km, got {:.0}m",
                dist
            );
        } else {
            panic!("point.distance should return Float, got {result:?}");
        }
    }

    /// point.distance() with same point = 0.
    #[test]
    fn point_distance_same_point() {
        let row = Row::new();
        let point_expr = Expr::FunctionCall {
            name: "point".to_string(),
            args: vec![Expr::MapLiteral(vec![
                ("latitude".to_string(), Expr::Literal(Value::Float(0.0))),
                ("longitude".to_string(), Expr::Literal(Value::Float(0.0))),
            ])],
            distinct: false,
        };

        let expr = Expr::FunctionCall {
            name: "point.distance".to_string(),
            args: vec![point_expr.clone(), point_expr],
            distinct: false,
        };

        assert_eq!(eval_expr(&expr, &row), Value::Float(0.0));
    }

    /// Haversine helper: known distance for antipodal points.
    #[test]
    fn haversine_antipodal() {
        // North Pole to South Pole ≈ 20,015 km (half circumference)
        let dist = super::haversine_distance(90.0, 0.0, -90.0, 0.0);
        assert!(
            (20_000_000.0..20_040_000.0).contains(&dist),
            "pole to pole should be ~20,015km, got {:.0}m",
            dist
        );
    }

    // -- Map Projection --

    #[test]
    fn eval_map_projection_shorthand() {
        let row = sample_row();
        // n { .name, .age } → Map { name: "Alice", age: 30 }
        let expr = Expr::MapProjection {
            expr: Box::new(Expr::Variable("n".into())),
            items: vec![
                MapProjectionItem::Property("name".into()),
                MapProjectionItem::Property("age".into()),
            ],
        };
        let result = eval_expr(&expr, &row);
        if let Value::Map(map) = result {
            assert_eq!(map.get("name"), Some(&Value::String("Alice".into())));
            assert_eq!(map.get("age"), Some(&Value::Int(30)));
            assert_eq!(map.len(), 2);
        } else {
            panic!("expected Map, got: {result:?}");
        }
    }

    #[test]
    fn eval_map_projection_computed() {
        let row = sample_row();
        // n { .name, doubled_age: n.age } with computed
        let expr = Expr::MapProjection {
            expr: Box::new(Expr::Variable("n".into())),
            items: vec![
                MapProjectionItem::Property("name".into()),
                MapProjectionItem::Computed(
                    "score_val".into(),
                    Expr::PropertyAccess {
                        expr: Box::new(Expr::Variable("n".into())),
                        property: "score".into(),
                    },
                ),
            ],
        };
        let result = eval_expr(&expr, &row);
        if let Value::Map(map) = result {
            assert_eq!(map.get("name"), Some(&Value::String("Alice".into())));
            assert_eq!(map.get("score_val"), Some(&Value::Float(0.95)));
        } else {
            panic!("expected Map, got: {result:?}");
        }
    }

    #[test]
    fn eval_map_projection_missing_prop_is_null() {
        let row = sample_row();
        let expr = Expr::MapProjection {
            expr: Box::new(Expr::Variable("n".into())),
            items: vec![MapProjectionItem::Property("nonexistent".into())],
        };
        let result = eval_expr(&expr, &row);
        if let Value::Map(map) = result {
            assert_eq!(map.get("nonexistent"), Some(&Value::Null));
        } else {
            panic!("expected Map");
        }
    }
}
