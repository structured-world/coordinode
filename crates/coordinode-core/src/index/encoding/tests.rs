use super::*;

#[test]
fn null_encoding() {
    assert_eq!(encode_value(&Value::Null), vec![TAG_NULL]);
}

#[test]
fn bool_ordering() {
    let false_enc = encode_value(&Value::Bool(false));
    let true_enc = encode_value(&Value::Bool(true));
    assert!(false_enc < true_enc);
}

#[test]
fn int_ordering() {
    let neg = encode_value(&Value::Int(-10));
    let zero = encode_value(&Value::Int(0));
    let pos = encode_value(&Value::Int(42));
    assert!(neg < zero);
    assert!(zero < pos);
}

#[test]
fn int_negative_ordering() {
    let a = encode_value(&Value::Int(-100));
    let b = encode_value(&Value::Int(-1));
    assert!(a < b, "-100 should sort before -1");
}

#[test]
fn float_ordering() {
    let neg = encode_value(&Value::Float(-1.5));
    let zero = encode_value(&Value::Float(0.0));
    let pos = encode_value(&Value::Float(3.5));
    assert!(neg < zero);
    assert!(zero < pos);
}

#[test]
fn string_ordering() {
    let a = encode_value(&Value::String("alice".into()));
    let b = encode_value(&Value::String("bob".into()));
    let c = encode_value(&Value::String("charlie".into()));
    assert!(a < b);
    assert!(b < c);
}

#[test]
fn type_ordering() {
    let null = encode_value(&Value::Null);
    let bool_val = encode_value(&Value::Bool(false));
    let int_val = encode_value(&Value::Int(0));
    let float_val = encode_value(&Value::Float(0.0));
    let str_val = encode_value(&Value::String("".into()));

    assert!(null < bool_val);
    assert!(bool_val < int_val);
    assert!(int_val < float_val);
    assert!(float_val < str_val);
}

#[test]
fn index_key_encoding() {
    let key = encode_index_key("user_email", &Value::String("alice@test.com".into()), 42);
    assert!(key.starts_with(b"idx:user_email:"));
    // Last 8 bytes should be the node ID
    assert_eq!(decode_node_id_from_index_key(&key), Some(42));
}

#[test]
fn index_key_sorts_by_value_then_id() {
    let k1 = encode_index_key("idx", &Value::String("alice".into()), 1);
    let k2 = encode_index_key("idx", &Value::String("alice".into()), 2);
    let k3 = encode_index_key("idx", &Value::String("bob".into()), 1);

    // Same value, different IDs: sorted by ID
    assert!(k1 < k2);
    // Different values: sorted by value
    assert!(k2 < k3);
}

#[test]
fn timestamp_ordering() {
    let t1 = encode_value(&Value::Timestamp(1_000_000));
    let t2 = encode_value(&Value::Timestamp(2_000_000));
    assert!(t1 < t2);
}
