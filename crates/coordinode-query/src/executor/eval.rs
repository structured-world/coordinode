//! Scalar value evaluation helpers shared by the neutral expression evaluator.
//!
//! Holds the Value-level operator implementations (`eval_binary_op`,
//! `eval_unary_op`), the scalar-function dispatch table
//! (`dispatch_scalar_function` and its list / string / math helpers), and the
//! score-requirement analysis used by the Project / Sort executor guards.

use coordinode_core::graph::types::Value;

use super::row::Row;
use crate::plan::expr::{BinOp, UnOp};

/// Classifies what cached row columns a projection / sort-key expression
/// needs. Used by `LogicalOp::Project` and `LogicalOp::Sort` executor guards
/// to reject plans that reference scoring helpers without the paired upstream
/// filter operator populating their cache columns.
///
/// - `text_score(...)` needs `__text_score__` from `TextFilter`.
/// - `hybrid_score(...)` needs at least one of `__text_score__` / `__vector_score__`
///   from `TextFilter` / `VectorFilter` — the fully-degenerate case where neither
///   is present indicates the user paired it with no `text_match` /
///   `vector_distance` anywhere, and the guards reject such plans.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScoreRequirements {
    pub needs_text_score: bool,
    pub needs_hybrid_score: bool,
    /// `rrf_score(...)` relies on `__rrf_score__` populated by an upstream
    /// `RankFuse` operator. If the builder did not insert one (shouldn't
    /// happen in a correctly-built plan), the Project/Sort guard errors.
    pub needs_rrf_score: bool,
    /// `doc_score(...)` relies on `__doc_score__` populated by an upstream
    /// `DocScore` operator. If the builder did not insert one, the
    /// Project/Sort guard errors.
    pub needs_doc_score: bool,
}

impl ScoreRequirements {
    pub fn any(&self) -> bool {
        self.needs_text_score
            || self.needs_hybrid_score
            || self.needs_rrf_score
            || self.needs_doc_score
    }
}

/// Collects which cached score columns a projection / sort-key expression will
/// read. Used by the Project / Sort executor guards. See [`ScoreRequirements`].
pub fn expr_score_requirements_neutral(expr: &crate::plan::expr::Expr) -> ScoreRequirements {
    let mut out = ScoreRequirements::default();
    collect_score_requirements_neutral(expr, &mut out);
    out
}

fn collect_score_requirements_neutral(expr: &crate::plan::expr::Expr, out: &mut ScoreRequirements) {
    use crate::plan::expr::Expr as PExpr;
    match expr {
        PExpr::Call { name, args, .. } => {
            match name.as_str() {
                "text_score" => out.needs_text_score = true,
                "hybrid_score" => out.needs_hybrid_score = true,
                "rrf_score" => out.needs_rrf_score = true,
                "doc_score" => out.needs_doc_score = true,
                _ => {}
            }
            for a in args {
                collect_score_requirements_neutral(a, out);
            }
        }
        PExpr::Binary { left, right, .. } => {
            collect_score_requirements_neutral(left, out);
            collect_score_requirements_neutral(right, out);
        }
        PExpr::Unary { operand, .. } => collect_score_requirements_neutral(operand, out),
        PExpr::Property { base, .. } => collect_score_requirements_neutral(base, out),
        PExpr::List(items) => {
            for it in items {
                collect_score_requirements_neutral(it, out);
            }
        }
        PExpr::Map(fields) => {
            for (_, v) in fields {
                collect_score_requirements_neutral(v, out);
            }
        }
        PExpr::In { item, list } => {
            collect_score_requirements_neutral(item, out);
            collect_score_requirements_neutral(list, out);
        }
        PExpr::IsNull { operand, .. } => collect_score_requirements_neutral(operand, out),
        PExpr::StringMatch { value, pattern, .. } => {
            collect_score_requirements_neutral(value, out);
            collect_score_requirements_neutral(pattern, out);
        }
        PExpr::Subscript { base, index } => {
            collect_score_requirements_neutral(base, out);
            collect_score_requirements_neutral(index, out);
        }
        PExpr::Case {
            operand,
            branches,
            otherwise,
        } => {
            if let Some(o) = operand.as_deref() {
                collect_score_requirements_neutral(o, out);
            }
            for (c, v) in branches {
                collect_score_requirements_neutral(c, out);
                collect_score_requirements_neutral(v, out);
            }
            if let Some(e) = otherwise.as_deref() {
                collect_score_requirements_neutral(e, out);
            }
        }
        _ => {}
    }
}

/// An expression that cannot produce a value.
///
/// Cypher evaluation is deliberately total almost everywhere: a type mismatch,
/// a missing property or a NULL operand answers NULL rather than failing.
/// Arithmetic is the exception, because the alternative to failing is answering
/// a number that is simply wrong, and a caller has no way to tell one from a
/// real result.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvalError {
    /// A call to a function no evaluator owns.
    ///
    /// Answering NULL made a typo indistinguishable from a property that is
    /// genuinely absent, so `RETURN lenght(x)` looked like an empty column
    /// rather than a mistake. The name is reported back so the caller can see
    /// which one it was.
    #[error("Unknown function '{0}'")]
    UnknownFunction(String),

    /// Division or modulo whose divisor is an integer zero.
    ///
    /// Only an integral zero: a floating-point zero divisor follows IEEE 754
    /// and yields an infinity, which is a value rather than a failure.
    #[error("/ by zero")]
    DivideByZero,

    /// An integer operation whose exact result leaves the `i64` range.
    ///
    /// Clamping at the bound would answer a wrong number and let a total that
    /// silently stopped growing pass for a total.
    #[error("long overflow")]
    LongOverflow,
}

/// `None` from a `checked_*` integer operation means the exact result does not
/// fit, which is the overflow this reports rather than clamps.
fn checked(result: Option<i64>) -> Result<i64, EvalError> {
    result.ok_or(EvalError::LongOverflow)
}

/// A time argument in epoch milliseconds, accepting either representation.
fn epoch_ms(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::Int(t)) | Some(Value::Timestamp(t)) => Some(*t),
        _ => None,
    }
}

/// The validity interval of the temporal edge bound to `var`, as
/// `(valid_from, valid_to)` where an open end is `None`.
///
/// `None` for the pair means the row does not carry a usable interval, which
/// the temporal predicates read as "does not match" rather than as a failure:
/// a row without the metadata is simply not a temporal edge.
fn temporal_bounds(var: Option<&str>, row: &Row) -> Option<(i64, Option<i64>)> {
    let var = var?;
    let valid_from = epoch_ms(row.get(&format!("{var}.valid_from")))?;
    let valid_to = match row.get(&format!("{var}.valid_to")) {
        Some(Value::Int(v)) | Some(Value::Timestamp(v)) => Some(*v),
        Some(Value::Null) | None => None,
        // Present but not a time: the interval cannot be read, so no match.
        _ => return None,
    };
    Some((valid_from, valid_to))
}

/// Whether a divisor makes division fail rather than produce a value.
///
/// The test is on the divisor's type, not the dividend's: `1.0 / 0` fails
/// while `1 / 0.0` is an infinity, which is what the reference implementation
/// does and what a client switching over expects.
fn divisor_is_integer_zero(right: &Value) -> bool {
    matches!(right, Value::Int(0))
}

/// Evaluate a binary operation.
///
/// Returns `Err` only for arithmetic that has no answer; every other
/// combination the operator does not handle yields NULL, per Cypher.
pub(crate) fn eval_binary_op(left: &Value, op: BinOp, right: &Value) -> Result<Value, EvalError> {
    let value = match op {
        // Arithmetic
        BinOp::Add => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(checked(a.checked_add(*b))?),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 + b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a + *b as f64),
            (Value::String(a), Value::String(b)) => Value::String(format!("{a}{b}")),
            // List concatenation / append / prepend (Cypher `+`).
            (Value::Array(a), Value::Array(b)) => {
                let mut out = a.clone();
                out.extend(b.iter().cloned());
                Value::Array(out)
            }
            (Value::Array(a), other) => {
                let mut out = a.clone();
                out.push(other.clone());
                Value::Array(out)
            }
            (other, Value::Array(b)) => {
                let mut out = Vec::with_capacity(b.len() + 1);
                out.push(other.clone());
                out.extend(b.iter().cloned());
                Value::Array(out)
            }
            _ => Value::Null,
        },
        BinOp::Sub => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(checked(a.checked_sub(*b))?),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 - b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a - *b as f64),
            _ => Value::Null,
        },
        BinOp::Mul => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(checked(a.checked_mul(*b))?),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 * b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a * *b as f64),
            _ => Value::Null,
        },
        // An integer zero divisor fails; a floating-point one is IEEE 754 and
        // gives an infinity. `checked_div` also refuses `i64::MIN / -1`, whose
        // exact result does not fit, and that is an overflow rather than a
        // division by zero.
        BinOp::Div if divisor_is_integer_zero(right) => return Err(EvalError::DivideByZero),
        BinOp::Div => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(checked(a.checked_div(*b))?),
            (Value::Float(a), Value::Float(b)) => Value::Float(a / b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 / b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a / *b as f64),
            _ => Value::Null,
        },
        BinOp::Modulo if divisor_is_integer_zero(right) => return Err(EvalError::DivideByZero),
        BinOp::Modulo => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Value::Int(checked(a.checked_rem(*b))?),
            _ => Value::Null,
        },

        // Comparison — Cypher three-valued logic: any comparison involving
        // NULL yields NULL (which WHERE treats as "row filtered out"). This
        // is mandatory per the openCypher spec — `n.prop = NULL` MUST NEVER
        // match, even when `n.prop` itself is NULL. Use `IS NULL` to test
        // for absent values.
        BinOp::Eq => match (left, right) {
            (Value::Null, _) | (_, Value::Null) => Value::Null,
            _ => Value::Bool(left == right),
        },
        BinOp::Neq => match (left, right) {
            (Value::Null, _) | (_, Value::Null) => Value::Null,
            _ => Value::Bool(left != right),
        },
        BinOp::Lt => match (left, right) {
            (Value::Null, _) | (_, Value::Null) => Value::Null,
            _ => Value::Bool(compare_values(left, right) == Some(std::cmp::Ordering::Less)),
        },
        BinOp::Lte => match (left, right) {
            (Value::Null, _) | (_, Value::Null) => Value::Null,
            _ => {
                let cmp = compare_values(left, right);
                Value::Bool(matches!(
                    cmp,
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                ))
            }
        },
        BinOp::Gt => match (left, right) {
            (Value::Null, _) | (_, Value::Null) => Value::Null,
            _ => Value::Bool(compare_values(left, right) == Some(std::cmp::Ordering::Greater)),
        },
        BinOp::Gte => match (left, right) {
            (Value::Null, _) | (_, Value::Null) => Value::Null,
            _ => {
                let cmp = compare_values(left, right);
                Value::Bool(matches!(
                    cmp,
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                ))
            }
        },

        // Logical
        BinOp::And => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a && *b),
            _ => Value::Null,
        },
        BinOp::Or => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a || *b),
            _ => Value::Null,
        },
        BinOp::Xor => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(*a ^ *b),
            _ => Value::Null,
        },
    };
    Ok(value)
}

/// Compare two values for ordering.
/// Convert an rmpv::Value to a CoordiNode Value.
///
/// Used for dot-notation property access on Document values.
/// Map/Array results stay as Document to allow continued path traversal.
/// Scalar results are converted to the corresponding Value variant.
pub(crate) fn rmpv_to_value(v: &rmpv::Value) -> Value {
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
pub(crate) fn eval_unary_op(op: UnOp, val: &Value) -> Value {
    match op {
        UnOp::Not => match val {
            Value::Bool(b) => Value::Bool(!b),
            _ => Value::Null,
        },
        UnOp::Neg => match val {
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

/// Coerce to a multi-vector (matrix of f32 rows) used by ColBERT-style
/// late-interaction scoring. Accepts `Value::MultiVector` directly, or a
/// `Value::Array` of `Value::Vector` / nested `Value::Array<Float|Int>`.
/// Returns `None` on any row that doesn't coerce or whose width differs
/// from the first row's width.
fn coerce_to_multi_vector(val: Option<&Value>) -> Option<Vec<Vec<f32>>> {
    match val {
        Some(Value::MultiVector(rows)) => Some(rows.clone()),
        Some(Value::Array(arr)) => {
            let mut rows: Vec<Vec<f32>> = Vec::with_capacity(arr.len());
            for item in arr {
                let row = coerce_to_vector(Some(item))?;
                rows.push(row);
            }
            let width = rows.first().map(Vec::len)?;
            if width == 0 || rows.iter().any(|r| r.len() != width) {
                return None;
            }
            Some(rows)
        }
        _ => None,
    }
}

/// Dispatch a scalar function on already-evaluated argument values.
///
/// `first_arg_var` is the bare-variable name of the first argument when it is a
/// variable reference (entity-introspection functions like `type` / `labels` /
/// `startNode` / `properties` resolve it against the row). Dialect-neutral, so
/// the cypher and neutral expression evaluators share this dispatch.
pub(crate) fn dispatch_scalar_function(
    name: &str,
    evaluated: Vec<Value>,
    first_arg_var: Option<&str>,
    row: &Row,
) -> Result<Value, EvalError> {
    Ok(match name {
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
        // length(p) → number of relationships in a path.
        "length" => match evaluated.first() {
            Some(Value::Path(p)) => Value::Int(p.rels.len() as i64),
            _ => Value::Null,
        },
        // nodes(p) → ordered list of node ids along the path.
        "nodes" => match evaluated.first() {
            Some(Value::Path(p)) => {
                Value::Array(p.nodes.iter().map(|n| Value::Int(*n as i64)).collect())
            }
            _ => Value::Null,
        },
        // relationships(p) → ordered list of relationships, each a map of
        // {type, source, target}. A first-class relationship value can replace
        // the map once the path model carries relationship ids and properties.
        "relationships" => match evaluated.first() {
            Some(Value::Path(p)) => Value::Array(
                p.rels
                    .iter()
                    .map(|r| {
                        let mut m: std::collections::BTreeMap<String, Value> =
                            std::collections::BTreeMap::new();
                        m.insert("type".to_string(), Value::String(r.edge_type.clone()));
                        m.insert("source".to_string(), Value::Int(r.source as i64));
                        m.insert("target".to_string(), Value::Int(r.target as i64));
                        Value::Map(m)
                    })
                    .collect(),
            ),
            _ => Value::Null,
        },
        "type" => {
            // type(r) → relationship type string.
            // The executor stores edge type as `r.__type__` in the row.
            // Extract the variable name from the first argument, then look up
            // `<variable>.__type__` in the row.
            if let Some(var) = first_arg_var {
                let key = format!("{var}.__type__");
                row.get(&key).cloned().unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }
        "elementId" => {
            // elementId(n) → 13-character Crockford base32 string derived
            // bijectively from the node's u64 NodeId. The variable binds to
            // Value::Int (the raw NodeId) for node patterns; for edge
            // patterns it returns NULL (edges are addressed by `(src, tgt)`
            // pairs in our model, not by a single id).
            if let Some(var) = first_arg_var {
                match row.get(var) {
                    Some(Value::Int(raw)) => {
                        let node_id = coordinode_core::graph::node::NodeId::from_raw(*raw as u64);
                        Value::String(node_id.to_element_id())
                    }
                    _ => Value::Null,
                }
            } else {
                Value::Null
            }
        }
        "id" => {
            // id(n) → raw NodeId u64 (deprecated; prefer elementId).
            // Kept for Neo4j v4 driver compatibility per arch/compatibility/neo4j.md.
            if let Some(var) = first_arg_var {
                match row.get(var) {
                    Some(Value::Int(raw)) => Value::Int(*raw),
                    _ => Value::Null,
                }
            } else {
                Value::Null
            }
        }
        "labels" => {
            // labels(n) → list of label strings for a node.
            // The executor stores primary label as `n.__label__` in the row.
            // CoordiNode nodes currently have exactly one label.
            if let Some(var) = first_arg_var {
                let key = format!("{var}.__label__");
                match row.get(&key) {
                    Some(Value::String(l)) => Value::Array(vec![Value::String(l.clone())]),
                    _ => Value::Array(vec![]),
                }
            } else {
                Value::Array(vec![])
            }
        }
        // startNode(r) / endNode(r) → the relationship's source / target node id.
        // Relationship variables bind `<var>.__src__` / `<var>.__tgt__` in the
        // row (Value::Int node ids); we return the id, matching how `id()`
        // surfaces nodes as integers in this model.
        "startNode" => match first_arg_var {
            Some(var) => row
                .get(&format!("{var}.__src__"))
                .cloned()
                .unwrap_or(Value::Null),
            _ => Value::Null,
        },
        "endNode" => match first_arg_var {
            Some(var) => row
                .get(&format!("{var}.__tgt__"))
                .cloned()
                .unwrap_or(Value::Null),
            _ => Value::Null,
        },
        // properties(x) → a map of the entity's user properties. Collects the
        // row's `<var>.<prop>` columns, skipping internal `__…__` markers
        // (label, type, src/tgt). Returns NULL when the argument is not a bound
        // variable.
        "properties" => match first_arg_var {
            Some(var) => {
                let prefix = format!("{var}.");
                let mut map: std::collections::BTreeMap<String, Value> =
                    std::collections::BTreeMap::new();
                for (k, v) in row {
                    if let Some(rest) = k.strip_prefix(&prefix) {
                        if !rest.starts_with("__") {
                            map.insert(rest.to_string(), v.clone());
                        }
                    }
                }
                Value::Map(map)
            }
            _ => Value::Null,
        },
        // keys(x) → the property keys of a node / relationship variable, or the
        // keys of a map value. For a bound variable, collects `<var>.<prop>`
        // columns (skipping internal `__…__` markers); falls back to the keys
        // of a map-valued argument.
        "keys" => {
            let from_prefix = if let Some(var) = first_arg_var {
                let prefix = format!("{var}.");
                let ks: Vec<Value> = row
                    .keys()
                    .filter_map(|k| k.strip_prefix(&prefix))
                    .filter(|rest| !rest.starts_with("__"))
                    .map(|rest| Value::String(rest.to_string()))
                    .collect();
                (!ks.is_empty()).then_some(ks)
            } else {
                None
            };
            match from_prefix {
                Some(ks) => Value::Array(ks),
                None => match evaluated.first() {
                    Some(Value::Map(m)) => {
                        Value::Array(m.keys().map(|k| Value::String(k.clone())).collect())
                    }
                    _ => Value::Null,
                },
            }
        }
        // nullIf(v1, v2) → NULL if v1 == v2, otherwise v1.
        "nullIf" => match (evaluated.first(), evaluated.get(1)) {
            (Some(a), Some(b)) if a == b => Value::Null,
            (Some(a), _) => a.clone(),
            _ => Value::Null,
        },
        // timestamp() → milliseconds since the Unix epoch (Cypher returns an
        // integer; `now()` above returns microseconds as a Timestamp value).
        "timestamp" => Value::Int(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0),
        ),
        // randomUUID() → a random version-4 UUID string.
        "randomUUID" => Value::String(random_uuid_v4()),
        // valueType(v) → the Cypher type name of the value.
        "valueType" => Value::String(cypher_value_type(evaluated.first())),
        // temporal_active_at(r, t) → bool
        // True iff the temporal edge `r` was active at time `t` (epoch ms),
        // i.e. `r.valid_from <= t AND (r.valid_to IS NULL OR r.valid_to > t)`.
        "temporal_active_at" => Value::Bool(
            match (
                temporal_bounds(first_arg_var, row),
                epoch_ms(evaluated.get(1)),
            ) {
                (Some((vf, vt)), Some(t)) => vf <= t && vt.is_none_or(|to| to > t),
                _ => false,
            },
        ),
        // temporal_overlaps(r, t_start, t_end) → bool
        // True iff the temporal edge's validity interval overlaps `[t_start, t_end)`,
        // i.e. `r.valid_from < t_end AND (r.valid_to IS NULL OR r.valid_to > t_start)`.
        "temporal_overlaps" => Value::Bool(
            match (
                temporal_bounds(first_arg_var, row),
                epoch_ms(evaluated.get(1)),
                epoch_ms(evaluated.get(2)),
            ) {
                (Some((vf, vt)), Some(t_start), Some(t_end)) => {
                    vf < t_end && vt.is_none_or(|to| to > t_start)
                }
                _ => false,
            },
        ),
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
        // maxsim_score(doc_tokens, query_tokens) -> MaxSim (ColBERT-style)
        // late-interaction score. Each argument is a multi-vector matrix
        // (per-token f32 rows). Returns the sum over query tokens of the
        // best dot-product against any doc token. Pre-normalise rows to
        // unit L2 norm if you want cosine semantics. Returns NULL when
        // either side is missing, dim-mismatched, or empty.
        "maxsim_score" => {
            let doc = coerce_to_multi_vector(evaluated.first());
            let query = coerce_to_multi_vector(evaluated.get(1));
            match (doc, query) {
                (Some(doc), Some(query)) => {
                    let score = coordinode_vector::metrics::maxsim(&doc, &query);
                    Value::Float(score as f64)
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
        // hybrid_score(node, query [, weights]) → opinionated blend of vector + text scores
        // cached on the row by VectorFilter / TextFilter. Signature position args are
        // currently unused (node and query are for semantics / future introspection);
        // the third optional arg is a map of weights:
        //   {vector: f32, text: f32}  — defaults 0.65 / 0.35
        // Semantics match arch/search/document-scoring.md § Document-Level Scoring Formula:
        //   hybrid_score(c, q) = w_vec × vec_similarity(c, q) + w_bm25 × text_score(c, q)
        // Vector normalization:
        //   "vector_similarity" (cosine, already [0,1] or [-1,1]) → use raw value
        //   "vector_distance"   (L2, lower is better)            → `1 - min(raw, 1)`
        //   "vector_manhattan"  (L1)                             → `1 / (1 + raw)`
        //   "vector_dot"        (unbounded)                      → use raw value (user-beware)
        // Degenerate cases:
        //   only vector cached, no text → returns the normalized vector score
        //   only text cached, no vector → returns text_score
        //   neither cached → caller error (caught by Project/Sort guards)
        "hybrid_score" => {
            let mut w_vec = 0.65_f64;
            let mut w_text = 0.35_f64;
            if let Some(Value::Map(weights)) = evaluated.get(2) {
                if let Some(Value::Float(v)) = weights.get("vector") {
                    w_vec = *v;
                }
                if let Some(Value::Float(v)) = weights.get("text") {
                    w_text = *v;
                }
            }

            let text_score = row.get("__text_score__").and_then(|v| {
                if let Value::Float(f) = v {
                    Some(*f)
                } else {
                    None
                }
            });
            let raw_vec = row.get("__vector_score__").and_then(|v| {
                if let Value::Float(f) = v {
                    Some(*f)
                } else {
                    None
                }
            });
            let vec_fn = row.get("__vector_function__").and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            });

            let normalized_vec = match (raw_vec, vec_fn.as_deref()) {
                (Some(raw), Some("vector_similarity")) => Some(raw),
                (Some(raw), Some("vector_distance")) => Some(1.0 - raw.clamp(0.0, 1.0)),
                (Some(raw), Some("vector_manhattan")) => Some(1.0 / (1.0 + raw)),
                (Some(raw), Some("vector_dot")) => Some(raw),
                _ => None,
            };

            match (normalized_vec, text_score) {
                (Some(v), Some(t)) => Value::Float(w_vec * v + w_text * t),
                (Some(v), None) => Value::Float(v),
                (None, Some(t)) => Value::Float(t),
                (None, None) => {
                    // Neither score cached — degenerate. Project/Sort guard should
                    // have caught this before we got here, but be defensive.
                    Value::Null
                }
            }
        }
        // rrf_score([methods…], query) → Reciprocal Rank Fusion score computed
        // by the `RankFuse` planner operator and cached on the row under the
        // `__rrf_score__` column. When invoked here the value is a pure lookup —
        // the builder rewrote this FunctionCall to a Variable("__rrf_score__")
        // reference for well-formed plans, so this branch only runs as a
        // defensive fallback (guarded by Project/Sort checks).
        "rrf_score" => row.get("__rrf_score__").cloned().unwrap_or(Value::Null),
        // doc_score(doc, query [, α, β, γ]) → document-level aggregate cached
        // on the row by `DocScore`. Defensive lookup — well-formed plans get
        // the call rewritten to `Variable("__doc_score__")` before eval.
        "doc_score" => row.get("__doc_score__").cloned().unwrap_or(Value::Null),
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
        // String, math, and list functions (Cypher names are case-insensitive).
        // Kept out of the exact-case arms above so existing functions stay
        // untouched; each helper lowercases the name and returns None for
        // anything it does not own.
        //
        // Reaching the end of this chain means no evaluator claims the name,
        // which is the one place that knows it, so the error is raised here
        // rather than against a list of names kept somewhere else. A second
        // list would drift, and the direction it drifts in rejects working
        // queries.
        _ => eval_string_function(name, &evaluated)
            .or_else(|| eval_math_function(name, &evaluated))
            .or_else(|| eval_list_function(name, &evaluated))
            .ok_or_else(|| EvalError::UnknownFunction(name.to_string()))?,
    })
}

/// Cypher list functions over already-evaluated values (`head`, `last`, `tail`,
/// `range`, `isEmpty`). The path/collection accessors `nodes`, `relationships`,
/// `length`, and `keys` need row or path context and live in the main dispatch.
///
/// Case-insensitive; `None` for unowned names. `head`/`last` on an empty list
/// yield `NULL`; `tail` of an empty list is the empty list. `isEmpty` also
/// accepts strings and maps. `range(start, end [, step])` is inclusive of both
/// ends (Cypher semantics) and yields an integer list.
fn eval_list_function(name: &str, args: &[Value]) -> Option<Value> {
    let as_int = |v: Option<&Value>| match v {
        Some(Value::Int(n)) => Some(*n),
        _ => None,
    };

    let result = match name.to_ascii_lowercase().as_str() {
        "head" => match args.first() {
            Some(Value::Array(a)) => a.first().cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        },
        "last" => match args.first() {
            Some(Value::Array(a)) => a.last().cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        },
        "tail" => match args.first() {
            Some(Value::Array(a)) => Value::Array(a.iter().skip(1).cloned().collect()),
            _ => Value::Null,
        },
        "isempty" => match args.first() {
            Some(Value::Array(a)) => Value::Bool(a.is_empty()),
            Some(Value::String(s)) => Value::Bool(s.is_empty()),
            Some(Value::Map(m)) => Value::Bool(m.is_empty()),
            _ => Value::Null,
        },
        // range(start, end [, step]) — inclusive both ends; step defaults to 1.
        "range" => match (as_int(args.first()), as_int(args.get(1))) {
            (Some(start), Some(end)) => {
                let step = as_int(args.get(2)).unwrap_or(1);
                if step == 0 {
                    return Some(Value::Null);
                }
                let mut out = Vec::new();
                let mut i = start;
                if step > 0 {
                    while i <= end {
                        out.push(Value::Int(i));
                        i += step;
                    }
                } else {
                    while i >= end {
                        out.push(Value::Int(i));
                        i += step;
                    }
                }
                Value::Array(out)
            }
            _ => Value::Null,
        },
        _ => return None,
    };

    Some(result)
}

/// Cypher string functions (`left`, `right`, `substring`, `toLower`/`lower`,
/// `toUpper`/`upper`, `trim`/`ltrim`/`rtrim`/`btrim`, `replace`, `reverse`,
/// `split`, `toStringOrNull`, `toStringList`, `normalize`, `charLength`).
///
/// `name` is matched case-insensitively (Cypher function names are
/// case-insensitive). `args` are the already-evaluated argument values.
/// Returns `None` for a name this helper does not handle so the caller can
/// apply the unknown-function NULL contract. Per Cypher semantics, any string
/// function applied to `NULL` returns `NULL`. Length / index arithmetic counts
/// Unicode scalar values (`chars()`), not bytes, so multi-byte input behaves
/// like Neo4j rather than splitting inside a codepoint.
fn eval_string_function(name: &str, args: &[Value]) -> Option<Value> {
    let lname = name.to_ascii_lowercase();

    // Helpers local to string dispatch.
    let as_str = |v: Option<&Value>| match v {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    };
    let as_len = |v: Option<&Value>| match v {
        Some(Value::Int(n)) => Some(*n),
        _ => None,
    };

    let result = match lname.as_str() {
        "tolower" | "lower" => match args.first() {
            Some(Value::String(s)) => Value::String(s.to_lowercase()),
            _ => Value::Null,
        },
        "toupper" | "upper" => match args.first() {
            Some(Value::String(s)) => Value::String(s.to_uppercase()),
            _ => Value::Null,
        },
        "trim" | "btrim" => match args.first() {
            Some(Value::String(s)) => Value::String(s.trim().to_string()),
            _ => Value::Null,
        },
        "ltrim" => match args.first() {
            Some(Value::String(s)) => Value::String(s.trim_start().to_string()),
            _ => Value::Null,
        },
        "rtrim" => match args.first() {
            Some(Value::String(s)) => Value::String(s.trim_end().to_string()),
            _ => Value::Null,
        },
        // left(s, len): leftmost `len` Unicode chars. Negative len → NULL
        // (Neo4j raises; we follow the project's no-panic NULL contract).
        "left" => match (as_str(args.first()), as_len(args.get(1))) {
            (Some(s), Some(len)) if len >= 0 => {
                Value::String(s.chars().take(len as usize).collect())
            }
            _ => Value::Null,
        },
        // right(s, len): rightmost `len` Unicode chars.
        "right" => match (as_str(args.first()), as_len(args.get(1))) {
            (Some(s), Some(len)) if len >= 0 => {
                let total = s.chars().count();
                let skip = total.saturating_sub(len as usize);
                Value::String(s.chars().skip(skip).collect())
            }
            _ => Value::Null,
        },
        // substring(s, start[, len]): 0-indexed start in Unicode chars; omitted
        // len → to end. Out-of-range start → empty string (Neo4j behaviour).
        "substring" => match (as_str(args.first()), as_len(args.get(1))) {
            (Some(s), Some(start)) if start >= 0 => {
                let chars = s.chars().skip(start as usize);
                match as_len(args.get(2)) {
                    Some(len) if len >= 0 => Value::String(chars.take(len as usize).collect()),
                    Some(_) => Value::Null, // negative length
                    None => Value::String(chars.collect()),
                }
            }
            _ => Value::Null,
        },
        // replace(original, search, replacement): replace every occurrence.
        "replace" => match (
            as_str(args.first()),
            as_str(args.get(1)),
            as_str(args.get(2)),
        ) {
            (Some(s), Some(search), Some(rep)) => Value::String(s.replace(&search, &rep)),
            _ => Value::Null,
        },
        // reverse(x): reverse a string (by Unicode chars) or a list.
        "reverse" => match args.first() {
            Some(Value::String(s)) => Value::String(s.chars().rev().collect()),
            Some(Value::Array(a)) => {
                let mut out = a.clone();
                out.reverse();
                Value::Array(out)
            }
            _ => Value::Null,
        },
        // split(original, delimiter): returns a list of substrings. The
        // delimiter may be a single string or a list of strings (split on any).
        "split" => match (as_str(args.first()), args.get(1)) {
            (Some(s), Some(Value::String(delim))) => {
                Value::Array(split_on(&s, std::slice::from_ref(delim)))
            }
            (Some(s), Some(Value::Array(delims))) => {
                let ds: Vec<String> = delims
                    .iter()
                    .filter_map(|d| match d {
                        Value::String(d) => Some(d.clone()),
                        _ => None,
                    })
                    .collect();
                Value::Array(split_on(&s, &ds))
            }
            _ => Value::Null,
        },
        // charLength(s): number of Unicode scalar values (vs `size`, which is
        // byte length for strings).
        "charlength" => match args.first() {
            Some(Value::String(s)) => Value::Int(s.chars().count() as i64),
            _ => Value::Null,
        },
        // toStringOrNull(v): like toString but yields NULL for unconvertible
        // input instead of raising.
        "tostringornull" => match args.first() {
            Some(Value::String(s)) => Value::String(s.clone()),
            Some(Value::Int(n)) => Value::String(n.to_string()),
            Some(Value::Float(f)) => Value::String(f.to_string()),
            Some(Value::Bool(b)) => Value::String(b.to_string()),
            _ => Value::Null,
        },
        // toStringList(list): convert each element via toString semantics;
        // unconvertible elements become NULL entries (Neo4j: toStringOrNull
        // per element).
        "tostringlist" => match args.first() {
            Some(Value::Array(items)) => Value::Array(
                items
                    .iter()
                    .map(|v| match v {
                        Value::String(s) => Value::String(s.clone()),
                        Value::Int(n) => Value::String(n.to_string()),
                        Value::Float(f) => Value::String(f.to_string()),
                        Value::Bool(b) => Value::String(b.to_string()),
                        _ => Value::Null,
                    })
                    .collect(),
            ),
            _ => Value::Null,
        },
        // normalize(s[, form]): Unicode normalization, default NFC.
        "normalize" => match args.first() {
            Some(Value::String(s)) => {
                let form = match args.get(1) {
                    Some(Value::String(f)) => f.to_uppercase(),
                    _ => "NFC".to_string(),
                };
                normalize_unicode(s, &form).map_or(Value::Null, Value::String)
            }
            _ => Value::Null,
        },
        _ => return None,
    };

    Some(result)
}

/// Split `s` on any of `delims`. An empty delimiter set (or all-empty
/// delimiters) returns the whole string as a single element, matching Neo4j's
/// handling of a degenerate delimiter.
fn split_on(s: &str, delims: &[String]) -> Vec<Value> {
    let active: Vec<&String> = delims.iter().filter(|d| !d.is_empty()).collect();
    if active.is_empty() {
        return vec![Value::String(s.to_string())];
    }
    let mut parts = vec![s.to_string()];
    for d in active {
        parts = parts
            .into_iter()
            .flat_map(|p| p.split(d.as_str()).map(str::to_string).collect::<Vec<_>>())
            .collect();
    }
    parts.into_iter().map(Value::String).collect()
}

/// Apply the requested Unicode normal form. Returns `None` for an unrecognised
/// form name so the caller maps it to NULL.
fn normalize_unicode(s: &str, form: &str) -> Option<String> {
    use unicode_normalization::UnicodeNormalization;
    match form {
        "NFC" => Some(s.nfc().collect()),
        "NFD" => Some(s.nfd().collect()),
        "NFKC" => Some(s.nfkc().collect()),
        "NFKD" => Some(s.nfkd().collect()),
        _ => None,
    }
}

/// Cypher math and trigonometric functions (`abs`, `ceil`, `floor`, `round`,
/// `sign`, `rand`, `e`, `pi`, `sqrt`, `exp`, `log`, `log10`, `isNaN`, the
/// `toInteger` / `toFloat` / `toBoolean` conversions plus their `…OrNull` and
/// `…List` variants; `sin`, `cos`, `tan`, `cot`, `asin`, `acos`, `atan`,
/// `atan2`, `haversin`, `degrees`, `radians`).
///
/// `name` is matched case-insensitively. Returns `None` for a name this helper
/// does not own so the caller applies the unknown-function NULL contract. Per
/// Cypher semantics a math function applied to `NULL` (or a non-numeric value)
/// returns `NULL`. `ceil` / `floor` / `round` / `sqrt` / `exp` / `log` /
/// `log10` follow Neo4j in returning a Float; `abs` preserves Int vs Float;
/// `sign` returns an Int.
fn eval_math_function(name: &str, args: &[Value]) -> Option<Value> {
    let lname = name.to_ascii_lowercase();

    // Coerce the first argument to f64 for the float-domain functions.
    let as_f64 = |v: Option<&Value>| match v {
        Some(Value::Int(n)) => Some(*n as f64),
        Some(Value::Float(f)) => Some(*f),
        _ => None,
    };

    // Nullary constants take no argument.
    match lname.as_str() {
        "pi" => return Some(Value::Float(std::f64::consts::PI)),
        "e" => return Some(Value::Float(std::f64::consts::E)),
        "rand" => return Some(Value::Float(rand::random::<f64>())),
        _ => {}
    }

    let result = match lname.as_str() {
        "abs" => match args.first() {
            Some(Value::Int(n)) => Value::Int(n.abs()),
            Some(Value::Float(f)) => Value::Float(f.abs()),
            _ => Value::Null,
        },
        "ceil" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.ceil())),
        "floor" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.floor())),
        // Round half away from zero (Neo4j default), returning a Float.
        "round" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.round())),
        "sign" => match args.first() {
            Some(Value::Int(n)) => Value::Int(n.signum()),
            Some(Value::Float(f)) => {
                if *f > 0.0 {
                    Value::Int(1)
                } else if *f < 0.0 {
                    Value::Int(-1)
                } else {
                    Value::Int(0)
                }
            }
            _ => Value::Null,
        },
        "sqrt" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.sqrt())),
        "exp" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.exp())),
        "log" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.ln())),
        "log10" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.log10())),
        "isnan" => match args.first() {
            Some(Value::Float(f)) => Value::Bool(f.is_nan()),
            // Integers are never NaN; a present non-float numeric is `false`.
            Some(Value::Int(_)) => Value::Bool(false),
            _ => Value::Null,
        },
        "tointeger" | "tointegerornull" => to_integer(args.first()),
        "tofloat" | "tofloatornull" => to_float(args.first()),
        "toboolean" | "tobooleanornull" => to_boolean(args.first()),
        "tointegerlist" => map_list(args.first(), |v| to_integer(Some(v))),
        "tofloatlist" => map_list(args.first(), |v| to_float(Some(v))),
        "tobooleanlist" => map_list(args.first(), |v| to_boolean(Some(v))),
        // Trigonometric functions (radians in/out except degrees/radians
        // conversions). Single-argument forms map NULL/non-numeric to NULL.
        "sin" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.sin())),
        "cos" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.cos())),
        "tan" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.tan())),
        // cot(x) = 1 / tan(x).
        "cot" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(1.0 / x.tan())),
        "asin" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.asin())),
        "acos" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.acos())),
        "atan" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.atan())),
        // atan2(y, x): two-argument arctangent.
        "atan2" => match (as_f64(args.first()), as_f64(args.get(1))) {
            (Some(y), Some(x)) => Value::Float(y.atan2(x)),
            _ => Value::Null,
        },
        // haversin(x) = (1 - cos x) / 2.
        "haversin" => {
            as_f64(args.first()).map_or(Value::Null, |x| Value::Float((1.0 - x.cos()) / 2.0))
        }
        "degrees" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.to_degrees())),
        "radians" => as_f64(args.first()).map_or(Value::Null, |x| Value::Float(x.to_radians())),
        _ => return None,
    };

    Some(result)
}

/// `toInteger` conversion: Int passes through; Float truncates toward zero;
/// a numeric string parses (via Float then truncate, so "3.7" → 3); Bool →
/// 1/0. Anything else (including a non-numeric string) → NULL.
fn to_integer(v: Option<&Value>) -> Value {
    match v {
        Some(Value::Int(n)) => Value::Int(*n),
        Some(Value::Float(f)) if f.is_finite() => Value::Int(*f as i64),
        Some(Value::Bool(b)) => Value::Int(i64::from(*b)),
        Some(Value::String(s)) => s
            .trim()
            .parse::<i64>()
            .ok()
            .or_else(|| s.trim().parse::<f64>().ok().map(|f| f as i64))
            .map_or(Value::Null, Value::Int),
        _ => Value::Null,
    }
}

/// `toFloat` conversion: Float passes through; Int widens; a numeric string
/// parses; Bool is not convertible (Neo4j) → NULL. Anything else → NULL.
fn to_float(v: Option<&Value>) -> Value {
    match v {
        Some(Value::Float(f)) => Value::Float(*f),
        Some(Value::Int(n)) => Value::Float(*n as f64),
        Some(Value::String(s)) => s.trim().parse::<f64>().map_or(Value::Null, Value::Float),
        _ => Value::Null,
    }
}

/// `toBoolean` conversion: Bool passes through; "true"/"false" (any case,
/// trimmed) parse; an integer is truthy iff non-zero (Neo4j 5). Anything else
/// → NULL.
fn to_boolean(v: Option<&Value>) -> Value {
    match v {
        Some(Value::Bool(b)) => Value::Bool(*b),
        Some(Value::Int(n)) => Value::Bool(*n != 0),
        Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

/// Apply a per-element conversion across a list, preserving length (each
/// unconvertible element becomes NULL). Non-list input → NULL.
fn map_list(v: Option<&Value>, f: impl Fn(&Value) -> Value) -> Value {
    match v {
        Some(Value::Array(items)) => Value::Array(items.iter().map(f).collect()),
        _ => Value::Null,
    }
}

/// Generate a random version-4 UUID string (`randomUUID()`). Built from 128
/// random bits with the version (4) and variant (RFC 4122) nibbles set.
fn random_uuid_v4() -> String {
    let mut b = rand::random::<u128>().to_be_bytes();
    b[6] = (b[6] & 0x0f) | 0x40; // version 4
    b[8] = (b[8] & 0x3f) | 0x80; // variant 10xx
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
        b[8],
        b[9],
        b[10],
        b[11],
        b[12],
        b[13],
        b[14],
        b[15]
    )
}

/// Cypher `valueType(v)` type-name string. Scalars carry the `NOT NULL`
/// suffix (a concrete value is never null); `NULL` itself is `"NULL"`.
/// Compound and engine-specific values (list, map, and our temporal / vector /
/// path / spatial variants) use a simplified label rather than Neo4j's fully
/// elaborated nested form.
fn cypher_value_type(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "NULL",
        Some(Value::Bool(_)) => "BOOLEAN NOT NULL",
        Some(Value::Int(_)) => "INTEGER NOT NULL",
        Some(Value::Float(_)) => "FLOAT NOT NULL",
        Some(Value::String(_)) => "STRING NOT NULL",
        Some(Value::Array(_)) => "LIST<ANY> NOT NULL",
        Some(Value::Map(_) | Value::Document(_)) => "MAP NOT NULL",
        Some(_) => "ANY NOT NULL",
    }
    .to_string()
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
mod tests;
