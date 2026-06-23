use super::*;
use coordinode_core::graph::node::NodeRecord;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
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
    {
        use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
        use coordinode_core::txn::write_concern::WriteConcern;
        use coordinode_modality::{LocalNodeStore, NodeStore as _};
        use coordinode_storage::engine::transaction::{CommitContext, Transaction};
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let read_ts = oracle.next();
        let mut txn = Transaction::new(&engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        LocalNodeStore
            .put(&mut txn, shard_id, node_id, &record)
            .expect("put node");
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        txn.commit(&ctx).expect("commit node");
    }

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
    // Raw partition write is the one legitimate raw-access case: plant a
    // corrupt msgpack body the typed store could never produce, to verify
    // decode failure resolves to "not visible".
    use coordinode_core::graph::node::encode_node_key;
    use coordinode_storage::engine::partition::Partition;

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
