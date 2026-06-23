use super::*;
use coordinode_core::graph::doc_delta::PathTarget;
use coordinode_core::graph::types::Value;

/// Helper: create a NodeRecord with a Document property in extra.
fn make_record_with_doc(key: &str, doc: rmpv::Value) -> NodeRecord {
    let mut rec = NodeRecord::new("TestLabel");
    rec.set_extra(key, Value::Document(doc));
    rec
}

/// Helper: encode a NodeRecord as a storage value (with 0x00 prefix).
fn encode_rec(rec: &NodeRecord) -> Vec<u8> {
    let msgpack = rec.to_msgpack().expect("encode");
    let mut buf = Vec::with_capacity(1 + msgpack.len());
    buf.push(PREFIX_NODE_RECORD);
    buf.extend_from_slice(&msgpack);
    buf
}

fn make_rmpv_map(entries: Vec<(&str, rmpv::Value)>) -> rmpv::Value {
    rmpv::Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (rmpv::Value::String(k.into()), v))
            .collect(),
    )
}

#[test]
fn doc_merge_set_path_on_empty_base() {
    let op = DocumentMerge;
    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["config".into(), "ssid".into()],
        value: rmpv::Value::String("home".into()),
    };
    let operand = delta.encode().expect("encode delta");

    let result = op.merge(b"node:0:1", None, &[&operand]).expect("merge");
    let rec = decode_node_record(&result).expect("decode result");

    // The extra map should contain config.ssid = "home".
    let doc_val = rec.get_extra("config").expect("config key");
    if let Value::Document(doc) = doc_val {
        let ssid = coordinode_core::graph::document::extract_at_path(doc, &["ssid"]);
        assert_eq!(ssid, rmpv::Value::String("home".into()));
    } else {
        panic!("expected Document, got {doc_val:?}");
    }
}

#[test]
fn doc_merge_set_path_on_existing_record() {
    let op = DocumentMerge;

    let initial_doc = make_rmpv_map(vec![("ssid", rmpv::Value::String("old".into()))]);
    let rec = make_record_with_doc("config", initial_doc);
    let base = encode_rec(&rec);

    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["config".into(), "ssid".into()],
        value: rmpv::Value::String("new".into()),
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    let doc_val = merged.get_extra("config").expect("config key");
    if let Value::Document(doc) = doc_val {
        let ssid = coordinode_core::graph::document::extract_at_path(doc, &["ssid"]);
        assert_eq!(ssid, rmpv::Value::String("new".into()));
    } else {
        panic!("expected Document, got {doc_val:?}");
    }
}

#[test]
fn doc_merge_multiple_deltas_different_paths() {
    let op = DocumentMerge;

    let delta1 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["a".into()],
        value: rmpv::Value::Integer(1.into()),
    };
    let delta2 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["b".into()],
        value: rmpv::Value::Integer(2.into()),
    };
    let op1 = delta1.encode().expect("encode");
    let op2 = delta2.encode().expect("encode");

    let result = op.merge(b"node:0:1", None, &[&op1, &op2]).expect("merge");
    let rec = decode_node_record(&result).expect("decode");

    let a = rec.get_extra("a").expect("key a");
    assert_eq!(a, &Value::Int(1));
    let b = rec.get_extra("b").expect("key b");
    assert_eq!(b, &Value::Int(2));
}

#[test]
fn doc_merge_increment() {
    let op = DocumentMerge;

    let initial_doc = rmpv::Value::Integer(10.into());
    let rec = make_record_with_doc("counter", initial_doc);
    let base = encode_rec(&rec);

    let delta = DocDelta::Increment {
        target: PathTarget::Extra,
        path: vec!["counter".into()],
        amount: 5.0,
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    let val = merged.get_extra("counter").expect("counter key");
    assert_eq!(val, &Value::Int(15));
}

#[test]
fn doc_merge_array_push() {
    let op = DocumentMerge;

    let initial_doc = rmpv::Value::Array(vec![rmpv::Value::String("a".into())]);
    let rec = make_record_with_doc("tags", initial_doc);
    let base = encode_rec(&rec);

    let delta = DocDelta::ArrayPush {
        target: PathTarget::Extra,
        path: vec!["tags".into()],
        value: rmpv::Value::String("b".into()),
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    let val = merged.get_extra("tags").expect("tags key");
    if let Value::Document(rmpv::Value::Array(arr)) = val {
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0], rmpv::Value::String("a".into()));
        assert_eq!(arr[1], rmpv::Value::String("b".into()));
    } else {
        panic!("expected Document(Array), got {val:?}");
    }
}

#[test]
fn doc_merge_add_to_set_dedup() {
    let op = DocumentMerge;

    let initial_doc = rmpv::Value::Array(vec![rmpv::Value::String("a".into())]);
    let rec = make_record_with_doc("tags", initial_doc);
    let base = encode_rec(&rec);

    // Add "a" again (duplicate) and "b" (new).
    let delta1 = DocDelta::ArrayAddToSet {
        target: PathTarget::Extra,
        path: vec!["tags".into()],
        value: rmpv::Value::String("a".into()),
    };
    let delta2 = DocDelta::ArrayAddToSet {
        target: PathTarget::Extra,
        path: vec!["tags".into()],
        value: rmpv::Value::String("b".into()),
    };
    let op1 = delta1.encode().expect("encode");
    let op2 = delta2.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base), &[&op1, &op2])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    let val = merged.get_extra("tags").expect("tags key");
    if let Value::Document(rmpv::Value::Array(arr)) = val {
        assert_eq!(arr.len(), 2, "dedup should prevent duplicate 'a'");
        assert_eq!(arr[0], rmpv::Value::String("a".into()));
        assert_eq!(arr[1], rmpv::Value::String("b".into()));
    } else {
        panic!("expected Document(Array), got {val:?}");
    }
}

#[test]
fn doc_merge_delete_path() {
    let op = DocumentMerge;

    let rec = {
        let mut r = NodeRecord::new("Test");
        r.set_extra("keep", Value::Int(1));
        r.set_extra("remove", Value::Int(2));
        r
    };
    let base = encode_rec(&rec);

    let delta = DocDelta::DeletePath {
        target: PathTarget::Extra,
        path: vec!["remove".into()],
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    assert!(merged.get_extra("keep").is_some());
    assert!(merged.get_extra("remove").is_none());
}

#[test]
fn doc_merge_re_merge_stability() {
    // Re-merging a merged result with no operands must be stable.
    let op = DocumentMerge;

    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["x".into()],
        value: rmpv::Value::Integer(42.into()),
    };
    let operand = delta.encode().expect("encode");

    let first = op
        .merge(b"node:0:1", None, &[&operand])
        .expect("first merge");
    let second = op
        .merge(b"node:0:1", Some(&first), &[])
        .expect("second merge");

    // Both should decode to equivalent NodeRecords.
    let rec1 = decode_node_record(&first).expect("decode 1");
    let rec2 = decode_node_record(&second).expect("decode 2");
    assert_eq!(rec1, rec2, "re-merge must be stable");
}

#[test]
fn doc_merge_pre_merged_record_as_operand() {
    // A previously merged NodeRecord appears as an operand in subsequent
    // compaction. The merge function should detect the 0x00 prefix and
    // use it as the new base.
    let op = DocumentMerge;

    let delta1 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["a".into()],
        value: rmpv::Value::Integer(1.into()),
    };
    let op1 = delta1.encode().expect("encode");

    // First merge produces a NodeRecord.
    let partial = op.merge(b"node:0:1", None, &[&op1]).expect("partial merge");

    // Subsequent compaction: partial result as operand + new delta.
    let delta2 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["b".into()],
        value: rmpv::Value::Integer(2.into()),
    };
    let op2 = delta2.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", None, &[&partial, &op2])
        .expect("re-merge");
    let rec = decode_node_record(&result).expect("decode");

    assert_eq!(rec.get_extra("a"), Some(&Value::Int(1)));
    assert_eq!(rec.get_extra("b"), Some(&Value::Int(2)));
}

#[test]
fn doc_merge_preserves_labels_and_props() {
    // Merge must preserve existing labels and interned props.
    let op = DocumentMerge;

    let mut rec = NodeRecord::new("User");
    rec.add_label("Admin".into());
    rec.set(1, Value::String("Alice".into())); // interned prop
    rec.set_extra("doc_field", Value::Document(make_rmpv_map(vec![])));
    let base = encode_rec(&rec);

    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["doc_field".into(), "nested".into()],
        value: rmpv::Value::Boolean(true),
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    // Labels preserved.
    assert!(merged.has_label("User"));
    assert!(merged.has_label("Admin"));
    // Interned prop preserved.
    assert_eq!(merged.get(1), Some(&Value::String("Alice".into())));
    // Doc field updated.
    let doc_val = merged.get_extra("doc_field").expect("doc_field");
    if let Value::Document(doc) = doc_val {
        let nested = coordinode_core::graph::document::extract_at_path(doc, &["nested"]);
        assert_eq!(nested, rmpv::Value::Boolean(true));
    } else {
        panic!("expected Document, got {doc_val:?}");
    }
}

#[test]
fn doc_merge_empty_operand_returns_error() {
    let op = DocumentMerge;
    let empty: &[u8] = &[];
    assert!(op.merge(b"node:0:1", None, &[empty]).is_err());
}

// --- PropField (G064) tests ---

#[test]
fn doc_merge_prop_field_set_path() {
    // SetPath targeting props[42] creates nested document in interned props.
    let op = DocumentMerge;
    let mut base = NodeRecord::new("Device");
    base.set(42, Value::Document(rmpv::Value::Map(vec![])));
    let base_bytes = encode_rec(&base);

    let delta = DocDelta::SetPath {
        target: PathTarget::PropField(42),
        path: vec!["network".into(), "ssid".into()],
        value: rmpv::Value::String("home".into()),
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base_bytes), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    if let Some(Value::Document(doc)) = merged.props.get(&42) {
        let val = coordinode_core::graph::document::extract_at_path(doc, &["network", "ssid"]);
        assert_eq!(val, rmpv::Value::String("home".into()));
    } else {
        panic!("expected Document at props[42]");
    }
}

#[test]
fn doc_merge_prop_field_creates_doc_from_nothing() {
    // PropField delta on non-existent base creates empty record with doc in props.
    let op = DocumentMerge;

    let delta = DocDelta::SetPath {
        target: PathTarget::PropField(10),
        path: vec!["key".into()],
        value: rmpv::Value::Integer(99.into()),
    };
    let operand = delta.encode().expect("encode");

    let result = op.merge(b"node:0:1", None, &[&operand]).expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    if let Some(Value::Document(doc)) = merged.props.get(&10) {
        let val = coordinode_core::graph::document::extract_at_path(doc, &["key"]);
        assert_eq!(val, rmpv::Value::Integer(99.into()));
    } else {
        panic!("expected Document at props[10]");
    }
}

#[test]
fn doc_merge_prop_field_delete_path() {
    // DeletePath removes a nested key from a Document in props.
    let op = DocumentMerge;
    let doc = make_rmpv_map(vec![
        ("a", rmpv::Value::Integer(1.into())),
        ("b", rmpv::Value::Integer(2.into())),
    ]);
    let mut base = NodeRecord::new("Test");
    base.set(7, Value::Document(doc));
    let base_bytes = encode_rec(&base);

    let delta = DocDelta::DeletePath {
        target: PathTarget::PropField(7),
        path: vec!["a".into()],
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base_bytes), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    if let Some(Value::Document(doc)) = merged.props.get(&7) {
        assert_eq!(
            coordinode_core::graph::document::extract_at_path(doc, &["a"]),
            rmpv::Value::Nil
        );
        assert_eq!(
            coordinode_core::graph::document::extract_at_path(doc, &["b"]),
            rmpv::Value::Integer(2.into())
        );
    } else {
        panic!("expected Document at props[7]");
    }
}

#[test]
fn doc_merge_prop_field_increment() {
    // Increment on a numeric field inside a Document in props.
    let op = DocumentMerge;
    let doc = make_rmpv_map(vec![("views", rmpv::Value::Integer(10.into()))]);
    let mut base = NodeRecord::new("Stats");
    base.set(5, Value::Document(doc));
    let base_bytes = encode_rec(&base);

    let delta = DocDelta::Increment {
        target: PathTarget::PropField(5),
        path: vec!["views".into()],
        amount: 3.0,
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base_bytes), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    if let Some(Value::Document(doc)) = merged.props.get(&5) {
        assert_eq!(
            coordinode_core::graph::document::extract_at_path(doc, &["views"]),
            rmpv::Value::Integer(13.into())
        );
    } else {
        panic!("expected Document at props[5]");
    }
}

#[test]
fn doc_merge_prop_field_preserves_extra_and_other_props() {
    // PropField delta only touches its target field, not extra or other props.
    let op = DocumentMerge;
    let mut base = NodeRecord::new("Mixed");
    base.set(1, Value::String("name_val".into()));
    base.set(2, Value::Document(rmpv::Value::Map(vec![])));
    base.set_extra("overflow_key", Value::Int(42));
    let base_bytes = encode_rec(&base);

    let delta = DocDelta::SetPath {
        target: PathTarget::PropField(2),
        path: vec!["nested".into()],
        value: rmpv::Value::String("val".into()),
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base_bytes), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    // Other prop untouched.
    assert_eq!(merged.get(1), Some(&Value::String("name_val".into())));
    // Extra untouched.
    assert_eq!(merged.get_extra("overflow_key"), Some(&Value::Int(42)));
    // Target prop updated.
    if let Some(Value::Document(doc)) = merged.props.get(&2) {
        assert_eq!(
            coordinode_core::graph::document::extract_at_path(doc, &["nested"]),
            rmpv::Value::String("val".into())
        );
    } else {
        panic!("expected Document at props[2]");
    }
}

#[test]
fn doc_merge_mixed_extra_and_prop_field() {
    // Extra delta and PropField delta in same merge call.
    let op = DocumentMerge;
    let mut base = NodeRecord::new("Mixed");
    base.set(3, Value::Document(rmpv::Value::Map(vec![])));
    let base_bytes = encode_rec(&base);

    let d1 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["overflow".into()],
        value: rmpv::Value::String("extra_val".into()),
    };
    let d2 = DocDelta::SetPath {
        target: PathTarget::PropField(3),
        path: vec!["nested".into()],
        value: rmpv::Value::String("prop_val".into()),
    };
    let op1 = d1.encode().expect("encode");
    let op2 = d2.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base_bytes), &[&op1, &op2])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    assert_eq!(
        merged.get_extra("overflow"),
        Some(&Value::String("extra_val".into()))
    );
    if let Some(Value::Document(doc)) = merged.props.get(&3) {
        assert_eq!(
            coordinode_core::graph::document::extract_at_path(doc, &["nested"]),
            rmpv::Value::String("prop_val".into())
        );
    } else {
        panic!("expected Document at props[3]");
    }
}

#[test]
fn doc_merge_legacy_bare_record_as_base() {
    // Legacy NodeRecord without 0x00 prefix (pre-R163 data).
    let op = DocumentMerge;

    let rec = NodeRecord::new("Legacy");
    let bare_msgpack = rec.to_msgpack().expect("encode");

    let delta = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["x".into()],
        value: rmpv::Value::Integer(1.into()),
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&bare_msgpack), &[&operand])
        .expect("merge with legacy base");
    let merged = decode_node_record(&result).expect("decode");

    assert!(merged.has_label("Legacy"));
    assert_eq!(merged.get_extra("x"), Some(&Value::Int(1)));
}

// ── RemoveProperty (R083 TTL reaper) ─────────────────────────────

#[test]
fn doc_merge_remove_property_prop_field() {
    // RemoveProperty with PropField removes entire props[field_id].
    let op = DocumentMerge;
    let mut base = NodeRecord::new("Session");
    base.set(3, Value::Timestamp(1_000_000));
    base.set(5, Value::String("keep me".into()));
    let base_bytes = encode_rec(&base);

    let delta = DocDelta::RemoveProperty {
        target: PathTarget::PropField(3),
        key: None,
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base_bytes), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    assert!(!merged.props.contains_key(&3), "field 3 should be removed");
    assert_eq!(
        merged.props.get(&5),
        Some(&Value::String("keep me".into())),
        "field 5 should be preserved"
    );
}

#[test]
fn doc_merge_remove_property_extra_key() {
    // RemoveProperty with Extra target removes a key from the extra map.
    let op = DocumentMerge;
    let mut base = NodeRecord::new("Validated");
    base.set_extra("temp_field", Value::Int(42));
    base.set_extra("keep_field", Value::Int(99));
    let base_bytes = encode_rec(&base);

    let delta = DocDelta::RemoveProperty {
        target: PathTarget::Extra,
        key: Some("temp_field".into()),
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base_bytes), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    assert!(
        merged.get_extra("temp_field").is_none(),
        "temp_field should be removed"
    );
    assert_eq!(
        merged.get_extra("keep_field"),
        Some(&Value::Int(99)),
        "keep_field should be preserved"
    );
}

#[test]
fn doc_merge_remove_property_idempotent() {
    // Removing a non-existent property is a no-op.
    let op = DocumentMerge;
    let mut base = NodeRecord::new("Test");
    base.set(1, Value::Int(100));
    let base_bytes = encode_rec(&base);

    let delta = DocDelta::RemoveProperty {
        target: PathTarget::PropField(999), // doesn't exist
        key: None,
    };
    let operand = delta.encode().expect("encode");

    let result = op
        .merge(b"node:0:1", Some(&base_bytes), &[&operand])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    assert_eq!(merged.props.get(&1), Some(&Value::Int(100)));
}

// ── R099: Extra-delta batching (batch rmpv round-trips) ──────────────────

#[test]
fn doc_merge_multiple_extra_deltas_batched_same_result() {
    // Multiple consecutive Extra-targeting deltas must produce the same
    // result whether applied one at a time (old code) or via the batching
    // path (new code). Tests that extra_doc accumulation is correct.
    let op = DocumentMerge;

    let initial = make_rmpv_map(vec![("x", rmpv::Value::Integer(0.into()))]);
    let rec = make_record_with_doc("counters", initial);
    let base = encode_rec(&rec);

    // Three consecutive SetPath deltas, all targeting PathTarget::Extra.
    let d1 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["counters".into(), "x".into()],
        value: rmpv::Value::Integer(1.into()),
    };
    let d2 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["counters".into(), "y".into()],
        value: rmpv::Value::Integer(2.into()),
    };
    let d3 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["counters".into(), "z".into()],
        value: rmpv::Value::Integer(3.into()),
    };

    let op1 = d1.encode().expect("encode d1");
    let op2 = d2.encode().expect("encode d2");
    let op3 = d3.encode().expect("encode d3");

    let result = op
        .merge(b"node:0:1", Some(&base), &[&op1, &op2, &op3])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    let doc_val = merged.get_extra("counters").expect("counters key");
    let Value::Document(doc) = doc_val else {
        panic!("expected Document, got {doc_val:?}");
    };
    use coordinode_core::graph::document::extract_at_path;
    assert_eq!(
        extract_at_path(doc, &["x"]),
        rmpv::Value::Integer(1.into()),
        "x should be 1"
    );
    assert_eq!(
        extract_at_path(doc, &["y"]),
        rmpv::Value::Integer(2.into()),
        "y should be 2"
    );
    assert_eq!(
        extract_at_path(doc, &["z"]),
        rmpv::Value::Integer(3.into()),
        "z should be 3"
    );
}

#[test]
fn doc_merge_mixed_extra_and_propfield_deltas() {
    // Mixed Extra and PropField deltas. PropField must apply correctly
    // regardless of pending extra_doc state (they don't interact — PropField
    // operates on rec.props, not rec.extra / extra_doc accumulator).
    let op = DocumentMerge;

    // Field 7 stores a Document-typed property (PropField target requires Document).
    let field7_doc = make_rmpv_map(vec![("inner", rmpv::Value::Integer(10.into()))]);
    let mut base_rec = NodeRecord::new("Mixed");
    base_rec.set(7, Value::Document(field7_doc));
    let initial_meta = make_rmpv_map(vec![("a", rmpv::Value::Integer(0.into()))]);
    base_rec.set_extra("meta", Value::Document(initial_meta));
    let base = encode_rec(&base_rec);

    // Extra delta — sets meta.a = 42.
    let d_extra = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["meta".into(), "a".into()],
        value: rmpv::Value::Integer(42.into()),
    };
    // PropField delta — sets field 7's "inner" key to 99.
    let d_prop = DocDelta::SetPath {
        target: PathTarget::PropField(7),
        path: vec!["inner".into()],
        value: rmpv::Value::Integer(99.into()),
    };
    // Another Extra delta — sets meta.b = 7.
    let d_extra2 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["meta".into(), "b".into()],
        value: rmpv::Value::Integer(7.into()),
    };

    let op1 = d_extra.encode().expect("encode d_extra");
    let op2 = d_prop.encode().expect("encode d_prop");
    let op3 = d_extra2.encode().expect("encode d_extra2");

    let result = op
        .merge(b"node:0:1", Some(&base), &[&op1, &op2, &op3])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    // PropField 7's "inner" key should be updated to 99.
    let field7 = merged.props.get(&7).expect("field 7 must exist");
    let Value::Document(f7_doc) = field7 else {
        panic!("expected Document for field 7, got {field7:?}");
    };
    use coordinode_core::graph::document::extract_at_path;
    assert_eq!(
        extract_at_path(f7_doc, &["inner"]),
        rmpv::Value::Integer(99.into()),
        "field7.inner = 99"
    );

    // Extra "meta" should have both changes from d_extra and d_extra2.
    let doc_val = merged.get_extra("meta").expect("meta key");
    let Value::Document(doc) = doc_val else {
        panic!("expected Document, got {doc_val:?}");
    };
    assert_eq!(
        extract_at_path(doc, &["a"]),
        rmpv::Value::Integer(42.into()),
        "meta.a = 42"
    );
    assert_eq!(
        extract_at_path(doc, &["b"]),
        rmpv::Value::Integer(7.into()),
        "meta.b = 7"
    );
}

#[test]
fn doc_merge_setpath_then_remove_property_extra() {
    // SetPath (Extra) followed by RemoveProperty (Extra) — the removal must
    // be applied AFTER the set. Both deltas are applied to the same extra_doc
    // accumulator so ordering is preserved.
    let op = DocumentMerge;

    let initial = make_rmpv_map(vec![
        ("keep", rmpv::Value::Integer(1.into())),
        ("drop", rmpv::Value::Integer(2.into())),
    ]);
    let rec = make_record_with_doc("data", initial);
    let base = encode_rec(&rec);

    // Set "data.new_field" first.
    let d_set = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["data".into(), "new_field".into()],
        value: rmpv::Value::Boolean(true),
    };
    // Then remove the entire "drop" top-level extra key.
    let d_remove = DocDelta::RemoveProperty {
        target: PathTarget::Extra,
        key: Some("drop".into()),
    };

    let op1 = d_set.encode().expect("encode d_set");
    let op2 = d_remove.encode().expect("encode d_remove");

    let result = op
        .merge(b"node:0:1", Some(&base), &[&op1, &op2])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    // "drop" was a top-level extra key and should be gone.
    assert!(
        merged.get_extra("drop").is_none(),
        "'drop' key should have been removed"
    );

    // "data" with its nested values (keep + new_field) should remain.
    let doc_val = merged.get_extra("data").expect("data key");
    let Value::Document(doc) = doc_val else {
        panic!("expected Document, got {doc_val:?}");
    };
    use coordinode_core::graph::document::extract_at_path;
    assert_eq!(
        extract_at_path(doc, &["new_field"]),
        rmpv::Value::Boolean(true),
        "data.new_field should be true"
    );
    assert_eq!(
        extract_at_path(doc, &["keep"]),
        rmpv::Value::Integer(1.into()),
        "data.keep should be 1"
    );
}

#[test]
fn doc_merge_base_reset_flushes_extra_doc() {
    // When a PREFIX_NODE_RECORD operand appears mid-stream (base reset),
    // the pending extra_doc must be flushed into the previous record before
    // the reset, then the new base starts fresh. This tests the flush-before-reset
    // invariant in DocumentMerge::merge.
    let op = DocumentMerge;

    // First base: node with "score" = 0 in extra.
    let initial = make_rmpv_map(vec![("value", rmpv::Value::Integer(0.into()))]);
    let base1_rec = make_record_with_doc("score", initial);
    let base1 = encode_rec(&base1_rec);

    // Delta applied to base1 (sets score.value = 10).
    let d1 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["score".into(), "value".into()],
        value: rmpv::Value::Integer(10.into()),
    };

    // Second base reset (new full NodeRecord with different label).
    let mut base2_rec = NodeRecord::new("Replacement");
    base2_rec.set_extra("flag", Value::Bool(true));
    let base2 = encode_rec(&base2_rec);

    // Delta applied to base2 (sets a new extra key).
    let d2 = DocDelta::SetPath {
        target: PathTarget::Extra,
        path: vec!["note".into()],
        value: rmpv::Value::String("after reset".into()),
    };

    let op1 = d1.encode().expect("encode d1");
    let op2 = d2.encode().expect("encode d2");

    // Operands: delta1, then a full NodeRecord (base reset), then delta2.
    let result = op
        .merge(b"node:0:1", Some(&base1), &[&op1, &base2, &op2])
        .expect("merge");
    let merged = decode_node_record(&result).expect("decode");

    // Final state should be base2 + d2 applied ("Replacement" label, flag=true, note="after reset").
    // d1's effect should be gone (applied to base1 which was replaced).
    assert!(
        merged.has_label("Replacement"),
        "label should be from base2 (Replacement)"
    );
    assert_eq!(
        merged.get_extra("flag"),
        Some(&Value::Bool(true)),
        "flag from base2 should be present"
    );
    assert_eq!(
        merged.get_extra("note"),
        Some(&Value::String("after reset".into())),
        "note from d2 should be set"
    );
    // score from base1 is gone — the reset wiped it.
    assert!(
        merged.get_extra("score").is_none(),
        "score from base1 should be gone after reset"
    );
}
