use super::*;

fn enc(value: Value) -> Vec<u8> {
    encode_tuple(&[value]).expect("indexable")
}

/// Types sort in tag order, values within a type in value order.
#[test]
fn encoding_preserves_order() {
    let ordered = [
        Value::Null,
        Value::Bool(false),
        Value::Bool(true),
        Value::Int(i64::MIN),
        Value::Int(-1),
        Value::Int(0),
        Value::Int(i64::MAX),
        Value::Float(f64::NEG_INFINITY),
        Value::Float(-1.5),
        Value::Float(0.0),
        Value::Float(3.5),
        Value::Float(f64::INFINITY),
        Value::String(String::new()),
        Value::String("a".into()),
        Value::String("a\0".into()),
        Value::String("ab".into()),
        Value::Timestamp(1),
        Value::Timestamp(2),
        Value::Binary(vec![]),
        Value::Binary(vec![0]),
        Value::Binary(vec![1]),
    ];
    for pair in ordered.windows(2) {
        assert!(
            enc(pair[0].clone()) < enc(pair[1].clone()),
            "{:?} must sort before {:?}",
            pair[0],
            pair[1]
        );
    }
}

/// A string holding NUL followed by the key separator must not share a
/// prefix with the shorter string: the old encoding wrote NUL unescaped, so
/// the entries of "a" were a prefix of those of "a\0:x" and a lookup of one
/// returned the other.
#[test]
fn a_string_is_never_a_key_prefix_of_another() {
    let short = index_value_prefix("i", &enc(Value::String("a".into())));
    let long = encode_index_key("i", &enc(Value::String("a\0:x".into())), 7);
    assert!(!long.starts_with(&short));
}

/// Composite values have no key; the old encoding mapped them all to NULL,
/// so two maps under a unique index collided with each other and with NULL.
#[test]
fn composite_values_are_unindexable() {
    assert_eq!(
        encode_tuple(&[Value::Map(Default::default())]),
        Err(Unindexable::Kind("a map"))
    );
    assert_eq!(
        encode_tuple(&[Value::Array(vec![])]),
        Err(Unindexable::Kind("a list"))
    );
    assert_eq!(
        encode_tuple(&[Value::Vector(vec![1.0])]),
        Err(Unindexable::Kind("a vector"))
    );
}

/// -0.0 equals 0.0, so it is the same key; NaN equals nothing, so it has none.
#[test]
fn float_keys_follow_float_equality() {
    assert_eq!(enc(Value::Float(-0.0)), enc(Value::Float(0.0)));
    assert_eq!(
        encode_tuple(&[Value::Float(f64::NAN)]),
        Err(Unindexable::NaN)
    );
}

/// A compound tuple decodes one way: shifting bytes between two string
/// elements changes the encoding.
#[test]
fn compound_tuples_are_injective() {
    let a = encode_tuple(&[Value::String("ab".into()), Value::String("c".into())]);
    let b = encode_tuple(&[Value::String("a".into()), Value::String("bc".into())]);
    assert_ne!(a, b);
}

/// Entries sort by value, then by node id, and carry the id at the end.
#[test]
fn non_unique_entries_sort_by_value_then_id() {
    let alice = enc(Value::String("alice".into()));
    let bob = enc(Value::String("bob".into()));
    let k1 = encode_index_key("idx", &alice, 1);
    let k2 = encode_index_key("idx", &alice, 2);
    let k3 = encode_index_key("idx", &bob, 1);
    assert!(k1 < k2 && k2 < k3);
    assert_eq!(decode_node_id(&k2), Some(2));
    assert!(k1.starts_with(&index_value_prefix("idx", &alice)));
    assert!(k1.starts_with(&index_prefix("idx")));
}

/// One index name is never a prefix of another's entries, whatever the
/// names hold, and the unique and non-unique layouts never meet.
#[test]
fn index_namespaces_are_disjoint() {
    let value = enc(Value::Int(1));
    assert!(!encode_index_key("ab", &value, 1).starts_with(&index_prefix("a")));
    assert!(!encode_unique_index_key("i", &value).starts_with(&index_prefix("i")));
    assert!(!encode_index_key("i", &value, 1).starts_with(&legacy_index_prefix("i")));
}
