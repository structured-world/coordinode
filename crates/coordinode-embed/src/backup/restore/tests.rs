use super::*;
use crate::backup::export::value_to_json;

#[test]
fn cypher_lit_parses_map_with_backtick_keys_and_mixed_values() {
    let src = r#"{`name`:"admins", `count`:3, `active`:true, `score`:1.5, `tags`:["a","b"], `UNIQUE IMPORT ID`:7}"#;
    let v = CypherLit::new(src).map().unwrap();
    let obj = v.as_object().unwrap();
    assert_eq!(obj["name"], serde_json::json!("admins"));
    assert_eq!(obj["count"], serde_json::json!(3));
    assert_eq!(obj["active"], serde_json::json!(true));
    assert_eq!(obj["score"], serde_json::json!(1.5));
    assert_eq!(obj["tags"], serde_json::json!(["a", "b"]));
    assert_eq!(obj["UNIQUE IMPORT ID"], serde_json::json!(7));
}

#[test]
fn cypher_lit_parses_float_embedding_array() {
    // The vector-embedding case: a list of floats must round-trip.
    let src = "[0.1, -0.25, 3.0, 1.0e-3]";
    let v = CypherLit::new(src).list().unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 4);
    assert_eq!(arr[0].as_f64().unwrap(), 0.1);
    assert_eq!(arr[1].as_f64().unwrap(), -0.25);
    assert_eq!(arr[3].as_f64().unwrap(), 0.001);
}

#[test]
fn cypher_lit_rest_exposes_trailing_input() {
    let mut lit = CypherLit::new("[1,2] AS row CREATE (n)");
    let _ = lit.list().unwrap();
    assert_eq!(lit.rest().trim(), "AS row CREATE (n)");
}

#[test]
fn cypher_lit_rejects_function_value() {
    // datetime(...) is a function, not data — must fail loudly.
    let err = CypherLit::new(r#"{`when`: datetime("2020")}"#).map();
    assert!(err.is_err());
}

#[test]
fn split_statements_handles_multiline_and_semicolon_in_string() {
    let text = "CREATE (n {`s`:\"a;b\"});\nUNWIND [1,2] AS row\nCREATE (m);\n";
    let stmts: Vec<String> = split_statements(text)
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    assert_eq!(stmts.len(), 2, "semicolon inside string must not split");
    assert!(stmts[1].starts_with("UNWIND"));
}

#[test]
fn parse_label_list_drops_var_and_import_label() {
    assert_eq!(
        parse_label_list("n:`User`:`HighValue`"),
        vec!["User", "HighValue"]
    );
    // Anonymous node (no var) plus the synthetic import label.
    assert_eq!(
        parse_label_list(":`Group`:`UNIQUE IMPORT LABEL`"),
        vec!["Group"]
    );
}

#[test]
fn extract_helpers_read_reltype_and_import_ids() {
    assert_eq!(
        extract_reltype("CREATE (a)-[r:`MEMBER_OF` {`since`:2020}]->(b)").unwrap(),
        "MEMBER_OF"
    );
    let ids = extract_import_ids(
        "MATCH (n1:`UNIQUE IMPORT LABEL`{`UNIQUE IMPORT ID`:5}), (n2:`L`{`UNIQUE IMPORT ID`:8})",
    );
    assert_eq!(ids, vec![5, 8]);
    assert_eq!(
        extract_create_labels("CREATE (n:`Person`{`UNIQUE IMPORT ID`: row._id}) SET n += x")
            .unwrap(),
        vec!["Person"]
    );
}

#[test]
fn document_json_roundtrip_via_marker() {
    // Use alphabetically sorted keys so JSON roundtrip preserves order
    let doc = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("count".into()),
            rmpv::Value::Integer(42.into()),
        ),
        (
            rmpv::Value::String("key".into()),
            rmpv::Value::String("value".into()),
        ),
    ]);
    let original = Value::Document(doc);

    // export → JSON
    let json = value_to_json(&original);
    // JSON should have _document wrapper
    assert!(json.is_object());
    assert!(json.get("_document").is_some());

    // import → Value
    let restored = json_to_value(&json);
    assert_eq!(original, restored);
}

#[test]
fn document_nested_json_roundtrip() {
    let doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("nested".into()),
        rmpv::Value::Map(vec![(
            rmpv::Value::String("deep".into()),
            rmpv::Value::Array(vec![
                rmpv::Value::Integer(1.into()),
                rmpv::Value::Boolean(true),
            ]),
        )]),
    )]);
    let original = Value::Document(doc);
    let json = value_to_json(&original);
    let restored = json_to_value(&json);
    assert_eq!(original, restored);
}

#[test]
fn map_not_confused_with_document() {
    // Regular Map should NOT become Document on restore
    let map = Value::Map(std::collections::BTreeMap::from([(
        "name".to_string(),
        Value::String("test".into()),
    )]));
    let json = value_to_json(&map);
    let restored = json_to_value(&json);
    assert!(matches!(restored, Value::Map(_)));
}
