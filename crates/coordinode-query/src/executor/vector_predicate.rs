//! Predicate evaluator for ACORN-style filtered HNSW search.
//!
//! The HNSW traversal calls `evaluate_predicate(engine, shard_id, predicate,
//! field_lookup, node_id) -> bool` for every candidate it considers. The
//! function must be cheap (one point-get + one msgpack decode) and total —
//! any decode failure, missing record, missing property, or type mismatch
//! resolves to `false` (not visible) so the HNSW loop never panics or
//! returns an error.
//!
//! Property names referenced by [`VectorPredicate::PropertyEq`] are looked
//! up via the caller-supplied closure so the evaluator never re-enters the
//! interner lock from inside a search-side hot loop. The closure can cache
//! the lookup or even return `None` to make the predicate fail closed.

use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_modality::{LocalNodeStore, NodeStore as _};
use coordinode_storage::engine::core::StorageEngine;

use crate::planner::logical::{NumericCmp, VectorPredicate};

/// Evaluate `predicate` against the node identified by `node_id`.
///
/// `field_lookup` resolves a property name to its interned `field_id`. The
/// caller is responsible for any caching it wants; this function calls the
/// closure once per `PropertyEq` leaf encountered during evaluation.
///
/// Returns `false` for any of: storage error, missing record, decode
/// failure, property absent, value type mismatch. Never returns `Err`.
pub fn evaluate_predicate(
    engine: &StorageEngine,
    shard_id: u16,
    node_id: NodeId,
    predicate: &VectorPredicate,
    field_lookup: &dyn Fn(&str) -> Option<u32>,
) -> bool {
    let Ok((_key, Some(bytes))) = LocalNodeStore.read_at_snapshot(engine, None, shard_id, node_id)
    else {
        return false;
    };
    let Ok(record) = NodeRecord::from_msgpack(&bytes) else {
        return false;
    };
    evaluate_against(&record, predicate, field_lookup)
}

/// Pure form: evaluate a predicate against an already-loaded `NodeRecord`.
/// Useful when the caller has the record in hand (e.g. tests or paths that
/// already paid the point-get cost).
pub fn evaluate_against(
    record: &NodeRecord,
    predicate: &VectorPredicate,
    field_lookup: &dyn Fn(&str) -> Option<u32>,
) -> bool {
    match predicate {
        VectorPredicate::LabelEq(label) => record.primary_label() == label,
        VectorPredicate::PropertyEq { property, value } => {
            let Some(fid) = field_lookup(property) else {
                return false;
            };
            let Some(stored) = record.props.get(&fid) else {
                return false;
            };
            values_equal(stored, value)
        }
        VectorPredicate::PropertyCmp {
            property,
            op,
            value,
        } => {
            let Some(fid) = field_lookup(property) else {
                return false;
            };
            let Some(stored) = record.props.get(&fid) else {
                return false;
            };
            let Some(stored_n) = numeric(stored) else {
                return false;
            };
            let Some(target_n) = numeric(value) else {
                return false;
            };
            match op {
                NumericCmp::Gt => stored_n > target_n,
                NumericCmp::Ge => stored_n >= target_n,
                NumericCmp::Lt => stored_n < target_n,
                NumericCmp::Le => stored_n <= target_n,
            }
        }
        VectorPredicate::And(left, right) => {
            evaluate_against(record, left, field_lookup)
                && evaluate_against(record, right, field_lookup)
        }
    }
}

/// Widen Int / Float to a single comparable f64 representation. Returns
/// None for non-numeric types — keeps the comparison total over all
/// `Value` variants without silent coercion of strings / bools / etc.
fn numeric(v: &Value) -> Option<f64> {
    match v {
        Value::Int(i) => Some(*i as f64),
        Value::Float(f) => Some(*f),
        _ => None,
    }
}

/// Strict equality on `Value`. We do NOT collapse cross-type comparisons
/// (e.g. Int(1) vs Float(1.0)) — that would silently widen what counts as
/// a match and complicate the ACORN cost model. If a query needs a wider
/// match, it stays in the post-filter and out of this hot path.
fn values_equal(a: &Value, b: &Value) -> bool {
    a == b
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// Tests plant raw fixtures (corrupt node bytes) via the storage partition.
#[allow(clippy::disallowed_types)]
mod tests;
