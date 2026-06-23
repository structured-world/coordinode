use super::*;

fn make_map(entries: Vec<(&str, rmpv::Value)>) -> rmpv::Value {
    rmpv::Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (rmpv::Value::String(k.into()), v))
            .collect(),
    )
}

#[test]
fn single_level_access() {
    let doc = make_map(vec![("name", rmpv::Value::String("Alice".into()))]);
    let result = extract_at_path(&doc, &["name"]);
    assert_eq!(result, rmpv::Value::String("Alice".into()));
}

#[test]
fn three_level_access() {
    let doc = make_map(vec![(
        "a",
        make_map(vec![(
            "b",
            make_map(vec![("c", rmpv::Value::Integer(42.into()))]),
        )]),
    )]);
    let result = extract_at_path(&doc, &["a", "b", "c"]);
    assert_eq!(result, rmpv::Value::Integer(42.into()));
}

#[test]
fn five_level_access() {
    let doc = make_map(vec![(
        "a",
        make_map(vec![(
            "b",
            make_map(vec![(
                "c",
                make_map(vec![(
                    "d",
                    make_map(vec![("e", rmpv::Value::String("deep".into()))]),
                )]),
            )]),
        )]),
    )]);
    let result = extract_at_path(&doc, &["a", "b", "c", "d", "e"]);
    assert_eq!(result, rmpv::Value::String("deep".into()));
}

#[test]
fn missing_key_returns_nil() {
    let doc = make_map(vec![("a", rmpv::Value::Integer(1.into()))]);
    let result = extract_at_path(&doc, &["nonexistent"]);
    assert_eq!(result, rmpv::Value::Nil);
}

#[test]
fn missing_nested_key_returns_nil() {
    let doc = make_map(vec![(
        "a",
        make_map(vec![("b", rmpv::Value::Integer(1.into()))]),
    )]);
    let result = extract_at_path(&doc, &["a", "c"]);
    assert_eq!(result, rmpv::Value::Nil);
}

#[test]
fn path_through_scalar_returns_nil() {
    let doc = make_map(vec![("name", rmpv::Value::String("Alice".into()))]);
    // "name" is a string, can't traverse further
    let result = extract_at_path(&doc, &["name", "x"]);
    assert_eq!(result, rmpv::Value::Nil);
}

#[test]
fn array_numeric_index() {
    let doc = make_map(vec![(
        "readings",
        rmpv::Value::Array(vec![
            make_map(vec![("value", rmpv::Value::F64(23.5))]),
            make_map(vec![("value", rmpv::Value::F64(24.1))]),
        ]),
    )]);
    // readings.0.value → 23.5
    let result = extract_at_path(&doc, &["readings", "0", "value"]);
    assert_eq!(result, rmpv::Value::F64(23.5));

    // readings.1.value → 24.1
    let result = extract_at_path(&doc, &["readings", "1", "value"]);
    assert_eq!(result, rmpv::Value::F64(24.1));
}

#[test]
fn array_out_of_bounds_returns_nil() {
    let doc = make_map(vec![(
        "items",
        rmpv::Value::Array(vec![rmpv::Value::Integer(1.into())]),
    )]);
    let result = extract_at_path(&doc, &["items", "5"]);
    assert_eq!(result, rmpv::Value::Nil);
}

#[test]
fn array_non_numeric_segment_returns_nil() {
    let doc = make_map(vec![(
        "items",
        rmpv::Value::Array(vec![rmpv::Value::Integer(1.into())]),
    )]);
    // "name" is not a valid array index
    let result = extract_at_path(&doc, &["items", "name"]);
    assert_eq!(result, rmpv::Value::Nil);
}

#[test]
fn empty_path_returns_whole_document() {
    let doc = make_map(vec![("a", rmpv::Value::Integer(1.into()))]);
    let result = extract_at_path(&doc, &[]);
    assert_eq!(result, doc);
}

#[test]
fn nil_value_not_traversable() {
    let result = extract_at_path(&rmpv::Value::Nil, &["a"]);
    assert_eq!(result, rmpv::Value::Nil);
}

#[test]
fn heterogeneous_array_index() {
    let doc = make_map(vec![(
        "mixed",
        rmpv::Value::Array(vec![
            rmpv::Value::Integer(1.into()),
            rmpv::Value::String("two".into()),
            rmpv::Value::Boolean(true),
        ]),
    )]);
    assert_eq!(
        extract_at_path(&doc, &["mixed", "0"]),
        rmpv::Value::Integer(1.into())
    );
    assert_eq!(
        extract_at_path(&doc, &["mixed", "1"]),
        rmpv::Value::String("two".into())
    );
    assert_eq!(
        extract_at_path(&doc, &["mixed", "2"]),
        rmpv::Value::Boolean(true)
    );
}

// --- extract_at_path_bytes tests ---

fn to_msgpack(v: &rmpv::Value) -> Vec<u8> {
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, v).expect("encode");
    buf
}

#[test]
fn bytes_single_level() {
    let doc = make_map(vec![("name", rmpv::Value::String("Alice".into()))]);
    let bytes = to_msgpack(&doc);
    let result = extract_at_path_bytes(&bytes, &["name"]);
    assert_eq!(result, Some(rmpv::Value::String("Alice".into())));
}

#[test]
fn bytes_three_levels() {
    let doc = make_map(vec![(
        "a",
        make_map(vec![(
            "b",
            make_map(vec![("c", rmpv::Value::Integer(42.into()))]),
        )]),
    )]);
    let bytes = to_msgpack(&doc);
    let result = extract_at_path_bytes(&bytes, &["a", "b", "c"]);
    assert_eq!(result, Some(rmpv::Value::Integer(42.into())));
}

#[test]
fn bytes_missing_key() {
    let doc = make_map(vec![("a", rmpv::Value::Integer(1.into()))]);
    let bytes = to_msgpack(&doc);
    assert_eq!(extract_at_path_bytes(&bytes, &["nonexistent"]), None);
}

#[test]
fn bytes_array_index() {
    let doc = make_map(vec![(
        "items",
        rmpv::Value::Array(vec![
            rmpv::Value::String("first".into()),
            rmpv::Value::String("second".into()),
        ]),
    )]);
    let bytes = to_msgpack(&doc);
    assert_eq!(
        extract_at_path_bytes(&bytes, &["items", "1"]),
        Some(rmpv::Value::String("second".into()))
    );
}

#[test]
fn bytes_empty_path() {
    let doc = make_map(vec![("k", rmpv::Value::Integer(1.into()))]);
    let bytes = to_msgpack(&doc);
    let result = extract_at_path_bytes(&bytes, &[]);
    assert_eq!(result, Some(doc));
}

#[test]
fn bytes_skips_non_matching_entries() {
    // Large doc with many keys — only one path extracted
    let mut entries = Vec::new();
    for i in 0..100 {
        entries.push((
            format!("field_{i}"),
            rmpv::Value::String(format!("value_{i}").into()),
        ));
    }
    // Add target at the end
    entries.push(("target".to_string(), rmpv::Value::Integer(999.into())));

    let doc = rmpv::Value::Map(
        entries
            .into_iter()
            .map(|(k, v)| (rmpv::Value::String(k.into()), v))
            .collect(),
    );
    let bytes = to_msgpack(&doc);
    let result = extract_at_path_bytes(&bytes, &["target"]);
    assert_eq!(result, Some(rmpv::Value::Integer(999.into())));
}

#[test]
fn bytes_malformed_returns_none() {
    let result = extract_at_path_bytes(&[0xFF, 0xFF], &["a"]);
    assert!(result.is_none());
}
