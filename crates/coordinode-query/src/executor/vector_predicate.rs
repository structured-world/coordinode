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

use coordinode_core::graph::node::{encode_node_key, NodeId, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

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
    let key = encode_node_key(shard_id, node_id);
    let Ok(Some(bytes)) = engine.get(Partition::Node, &key) else {
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
mod tests {
    use super::*;
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use std::collections::HashMap;

    fn test_engine(dir: &std::path::Path) -> StorageEngine {
        let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        StorageEngine::open(&cfg).expect("open engine")
    }

    /// Build a NodeRecord with a `category` String property whose field
    /// id is `2`. Returns the record (msgpack-serialised) ready for
    /// `engine.put(Partition::Node, key, bytes)`.
    fn make_record(label: &str, category: Option<&str>) -> NodeRecord {
        let mut props: HashMap<u32, Value> = HashMap::new();
        if let Some(cat) = category {
            props.insert(2, Value::String(cat.into()));
        }
        let mut record = NodeRecord::new(label);
        record.props = props;
        record
    }

    fn lookup_category(name: &str) -> Option<u32> {
        if name == "category" {
            Some(2)
        } else {
            None
        }
    }

    #[test]
    fn label_eq_matches_primary_label() {
        let r = make_record("Item", None);
        assert!(evaluate_against(
            &r,
            &VectorPredicate::LabelEq("Item".into()),
            &lookup_category,
        ));
        assert!(!evaluate_against(
            &r,
            &VectorPredicate::LabelEq("Other".into()),
            &lookup_category,
        ));
    }

    #[test]
    fn property_eq_matches_stored_value() {
        let r = make_record("Item", Some("electronics"));
        let pred = VectorPredicate::PropertyEq {
            property: "category".into(),
            value: Value::String("electronics".into()),
        };
        assert!(evaluate_against(&r, &pred, &lookup_category));
    }

    #[test]
    fn property_eq_rejects_mismatched_value() {
        let r = make_record("Item", Some("books"));
        let pred = VectorPredicate::PropertyEq {
            property: "category".into(),
            value: Value::String("electronics".into()),
        };
        assert!(!evaluate_against(&r, &pred, &lookup_category));
    }

    #[test]
    fn property_eq_absent_property_returns_false() {
        let r = make_record("Item", None);
        let pred = VectorPredicate::PropertyEq {
            property: "category".into(),
            value: Value::String("electronics".into()),
        };
        assert!(!evaluate_against(&r, &pred, &lookup_category));
    }

    #[test]
    fn property_eq_unknown_property_returns_false() {
        let r = make_record("Item", Some("electronics"));
        let pred = VectorPredicate::PropertyEq {
            property: "unknown_property".into(),
            value: Value::String("electronics".into()),
        };
        // `lookup_category` returns None for unknown names → false.
        assert!(!evaluate_against(&r, &pred, &lookup_category));
    }

    #[test]
    fn property_cmp_numeric_branches() {
        // category as numeric proxy: stored int 42, compared against 10/50.
        let mut record = NodeRecord::new("Item");
        record.props.insert(2, Value::Int(42));

        let pred_ge_10 = VectorPredicate::PropertyCmp {
            property: "category".into(),
            op: NumericCmp::Ge,
            value: Value::Int(10),
        };
        let pred_ge_50 = VectorPredicate::PropertyCmp {
            property: "category".into(),
            op: NumericCmp::Ge,
            value: Value::Int(50),
        };
        let pred_lt_50 = VectorPredicate::PropertyCmp {
            property: "category".into(),
            op: NumericCmp::Lt,
            value: Value::Int(50),
        };
        assert!(evaluate_against(&record, &pred_ge_10, &lookup_category));
        assert!(!evaluate_against(&record, &pred_ge_50, &lookup_category));
        assert!(evaluate_against(&record, &pred_lt_50, &lookup_category));
    }

    #[test]
    fn property_cmp_int_float_widening() {
        // Stored Int(42), threshold Float(41.5) — Int widens to f64.
        let mut record = NodeRecord::new("Item");
        record.props.insert(2, Value::Int(42));
        let pred = VectorPredicate::PropertyCmp {
            property: "category".into(),
            op: NumericCmp::Gt,
            value: Value::Float(41.5),
        };
        assert!(evaluate_against(&record, &pred, &lookup_category));
    }

    #[test]
    fn property_cmp_non_numeric_value_rejects() {
        // Stored Bool — cannot be cast to f64 → predicate fails closed.
        let mut record = NodeRecord::new("Item");
        record.props.insert(2, Value::Bool(true));
        let pred = VectorPredicate::PropertyCmp {
            property: "category".into(),
            op: NumericCmp::Ge,
            value: Value::Int(0),
        };
        assert!(!evaluate_against(&record, &pred, &lookup_category));
    }

    #[test]
    fn and_requires_both_branches() {
        let r = make_record("Item", Some("electronics"));
        let pred = VectorPredicate::And(
            Box::new(VectorPredicate::LabelEq("Item".into())),
            Box::new(VectorPredicate::PropertyEq {
                property: "category".into(),
                value: Value::String("electronics".into()),
            }),
        );
        assert!(evaluate_against(&r, &pred, &lookup_category));

        // Right branch fails:
        let pred_fail = VectorPredicate::And(
            Box::new(VectorPredicate::LabelEq("Item".into())),
            Box::new(VectorPredicate::PropertyEq {
                property: "category".into(),
                value: Value::String("books".into()),
            }),
        );
        assert!(!evaluate_against(&r, &pred_fail, &lookup_category));
    }

    #[test]
    fn evaluate_predicate_via_engine_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());

        let shard_id: u16 = 1;
        let node_id = NodeId::from_raw(42);

        let record = make_record("Item", Some("electronics"));
        let bytes = record.to_msgpack().expect("encode");
        let key = encode_node_key(shard_id, node_id);
        engine.put(Partition::Node, &key, &bytes).expect("put node");

        let pred = VectorPredicate::And(
            Box::new(VectorPredicate::LabelEq("Item".into())),
            Box::new(VectorPredicate::PropertyEq {
                property: "category".into(),
                value: Value::String("electronics".into()),
            }),
        );

        assert!(evaluate_predicate(
            &engine,
            shard_id,
            node_id,
            &pred,
            &lookup_category,
        ));

        // Predicate that mismatches on the right branch.
        let pred_fail = VectorPredicate::PropertyEq {
            property: "category".into(),
            value: Value::String("books".into()),
        };
        assert!(!evaluate_predicate(
            &engine,
            shard_id,
            node_id,
            &pred_fail,
            &lookup_category,
        ));
    }

    #[test]
    fn evaluate_predicate_missing_node_returns_false() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let pred = VectorPredicate::LabelEq("Item".into());

        // No node 99 in the engine.
        assert!(!evaluate_predicate(
            &engine,
            1,
            NodeId::from_raw(99),
            &pred,
            &lookup_category,
        ));
    }

    #[test]
    fn evaluate_predicate_corrupt_record_returns_false() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());

        let shard_id: u16 = 1;
        let node_id = NodeId::from_raw(7);
        let key = encode_node_key(shard_id, node_id);
        // Intentionally corrupt msgpack body.
        engine
            .put(Partition::Node, &key, &[0xff, 0x00, 0xfe])
            .expect("put garbage");

        let pred = VectorPredicate::LabelEq("Item".into());
        assert!(!evaluate_predicate(
            &engine,
            shard_id,
            node_id,
            &pred,
            &lookup_category,
        ));
    }
}
