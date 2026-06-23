use super::*;
use crate::graph::types::VectorMetric;
use crate::schema::definition::SchemaMode;

fn make_field_names(pairs: &[(&str, u32)]) -> HashMap<u32, String> {
    pairs
        .iter()
        .map(|&(name, id)| (id, name.to_string()))
        .collect()
}

fn user_schema() -> LabelSchema {
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new("name", PropertyType::String).not_null());
    schema.add_property(PropertyDef::new("age", PropertyType::Int));
    schema.add_property(
        PropertyDef::new("status", PropertyType::String)
            .with_default(Value::String("active".into())),
    );
    schema
}

#[test]
fn valid_properties() {
    let schema = user_schema();
    let field_names = make_field_names(&[("name", 1), ("age", 2)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("Alice".into()));
    props.insert(2, Value::Int(30));

    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn type_mismatch() {
    let schema = user_schema();
    let field_names = make_field_names(&[("name", 1), ("age", 2)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("Alice".into()));
    props.insert(2, Value::String("not an int".into())); // age should be INT

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0], ValidationError::TypeMismatch { .. }));
}

#[test]
fn not_null_violation_missing() {
    let schema = user_schema();
    let field_names = make_field_names(&[("age", 2)]); // name not provided
    let mut props = HashMap::new();
    props.insert(2, Value::Int(30));

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(errors.iter().any(
        |e| matches!(e, ValidationError::NotNullViolation { property } if property == "name")
    ));
}

#[test]
fn not_null_violation_explicit_null() {
    let schema = user_schema();
    let field_names = make_field_names(&[("name", 1)]);
    let mut props = HashMap::new();
    props.insert(1, Value::Null); // name is NOT NULL but set to null

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(errors
        .iter()
        .any(|e| matches!(e, ValidationError::NotNullViolation { .. })));
}

#[test]
fn not_null_with_default_not_required() {
    // "status" has NOT NULL=false and a default, so missing is OK
    let schema = user_schema();
    let field_names = make_field_names(&[("name", 1)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("Alice".into()));

    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn vector_dims_match() {
    let mut schema = LabelSchema::new_node_id("Movie");
    schema.add_property(PropertyDef::new(
        "embedding",
        PropertyType::Vector {
            dimensions: 3,
            metric: VectorMetric::Cosine,
        },
    ));

    let field_names = make_field_names(&[("embedding", 1)]);
    let mut props = HashMap::new();
    props.insert(1, Value::Vector(vec![0.1, 0.2, 0.3]));

    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn vector_dims_mismatch() {
    let mut schema = LabelSchema::new_node_id("Movie");
    schema.add_property(PropertyDef::new(
        "embedding",
        PropertyType::Vector {
            dimensions: 384,
            metric: VectorMetric::Cosine,
        },
    ));

    let field_names = make_field_names(&[("embedding", 1)]);
    let mut props = HashMap::new();
    props.insert(1, Value::Vector(vec![0.1, 0.2, 0.3])); // 3 != 384

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(matches!(
        errors[0],
        ValidationError::VectorDimsMismatch {
            expected: 384,
            got: 3,
            ..
        }
    ));
}

#[test]
fn array_homogeneous() {
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new(
        "tags",
        PropertyType::Array(Box::new(PropertyType::String)),
    ));

    let field_names = make_field_names(&[("tags", 1)]);
    let mut props = HashMap::new();
    props.insert(
        1,
        Value::Array(vec![
            Value::String("rust".into()),
            Value::String("graph".into()),
        ]),
    );

    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn array_not_homogeneous() {
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new(
        "tags",
        PropertyType::Array(Box::new(PropertyType::String)),
    ));

    let field_names = make_field_names(&[("tags", 1)]);
    let mut props = HashMap::new();
    props.insert(
        1,
        Value::Array(vec![
            Value::String("rust".into()),
            Value::Int(42), // wrong type
        ]),
    );

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(matches!(
        errors[0],
        ValidationError::ArrayNotHomogeneous { .. }
    ));
}

#[test]
fn array_nulls_allowed() {
    let mut schema = LabelSchema::new_node_id("User");
    schema.add_property(PropertyDef::new(
        "tags",
        PropertyType::Array(Box::new(PropertyType::String)),
    ));

    let field_names = make_field_names(&[("tags", 1)]);
    let mut props = HashMap::new();
    props.insert(
        1,
        Value::Array(vec![Value::String("rust".into()), Value::Null]),
    );

    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn strict_mode_rejects_unknown() {
    let mut schema = user_schema();
    schema.set_mode(SchemaMode::Strict);

    let field_names = make_field_names(&[("name", 1), ("unknown_field", 99)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("Alice".into()));
    props.insert(99, Value::String("surprise".into()));

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(errors.iter().any(|e| matches!(
        e,
        ValidationError::UnknownProperty { property, .. } if property == "unknown_field"
    )));
}

#[test]
fn validated_mode_allows_unknown() {
    let mut schema = user_schema();
    schema.set_mode(SchemaMode::Validated);

    let field_names = make_field_names(&[("name", 1), ("extra", 99)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("Alice".into()));
    props.insert(99, Value::String("extra data".into()));

    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn null_value_passes_type_check() {
    let schema = user_schema();
    let field_names = make_field_names(&[("name", 1), ("age", 2)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("Alice".into()));
    props.insert(2, Value::Null); // age allows null

    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn multiple_errors_collected() {
    let schema = user_schema();
    let field_names = make_field_names(&[("age", 2)]); // name missing + age wrong type
    let mut props = HashMap::new();
    props.insert(2, Value::String("wrong".into()));

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(errors.len() >= 2); // at least: name not null + age type mismatch
}

#[test]
fn validation_error_display() {
    let err = ValidationError::TypeMismatch {
        property: "age".to_string(),
        expected: "INT".to_string(),
        got: "STRING".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "type mismatch for 'age': expected INT, got STRING"
    );
}

#[test]
fn document_type_matches() {
    let mut schema = LabelSchema::new_node_id("Config");
    schema.add_property(PropertyDef::new("data", PropertyType::Document));

    let field_names = make_field_names(&[("data", 1)]);
    let mut props = HashMap::new();
    props.insert(
        1,
        Value::Document(rmpv::Value::Map(vec![(
            rmpv::Value::String("key".into()),
            rmpv::Value::String("value".into()),
        )])),
    );

    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn document_type_mismatch() {
    let mut schema = LabelSchema::new_node_id("Config");
    schema.add_property(PropertyDef::new("data", PropertyType::Document));

    let field_names = make_field_names(&[("data", 1)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("not a document".into()));

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(matches!(
        errors[0],
        ValidationError::TypeMismatch {
            ref expected,
            ref got,
            ..
        } if expected == "DOCUMENT" && got == "STRING"
    ));
}

#[test]
fn document_size_limit_rejected() {
    let mut schema = LabelSchema::new_node_id("Config");
    schema.add_property(PropertyDef::new("data", PropertyType::Document));

    let field_names = make_field_names(&[("data", 1)]);

    // Create a document that exceeds 4MB by using a large binary blob
    let large_binary = vec![0xABu8; 5 * 1024 * 1024]; // 5MB
    let large_doc = rmpv::Value::Map(vec![(
        rmpv::Value::String("payload".into()),
        rmpv::Value::Binary(large_binary),
    )]);

    let mut props = HashMap::new();
    props.insert(1, Value::Document(large_doc));

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(matches!(
        errors[0],
        ValidationError::DocumentTooLarge { .. }
    ));
}

#[test]
fn document_null_valid() {
    let mut schema = LabelSchema::new_node_id("Config");
    schema.add_property(PropertyDef::new("data", PropertyType::Document));

    let field_names = make_field_names(&[("data", 1)]);
    let mut props = HashMap::new();
    props.insert(1, Value::Null); // null is valid for any type

    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn document_size_error_display() {
    let err = ValidationError::DocumentTooLarge {
        property: "config".to_string(),
        max_bytes: 4_194_304,
        got_bytes: 5_000_000,
    };
    assert!(err.to_string().contains("config"));
    assert!(err.to_string().contains("5000000"));
}

// --- SchemaMode validation tests ---

#[test]
fn validated_mode_accepts_unknown_properties() {
    let mut schema = LabelSchema::new_node_id("Device");
    schema.set_mode(SchemaMode::Validated);
    schema.add_property(PropertyDef::new("name", PropertyType::String));

    let field_names = make_field_names(&[("name", 1), ("custom_field", 2)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("sensor".into()));
    props.insert(2, Value::String("undeclared value".into()));

    // VALIDATED: undeclared accepted without validation
    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn validated_mode_still_validates_declared() {
    let mut schema = LabelSchema::new_node_id("Device");
    schema.set_mode(SchemaMode::Validated);
    schema.add_property(PropertyDef::new("age", PropertyType::Int));

    let field_names = make_field_names(&[("age", 1)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("not an int".into()));

    // VALIDATED: declared properties still validated
    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(matches!(errors[0], ValidationError::TypeMismatch { .. }));
}

#[test]
fn flexible_mode_accepts_all() {
    let mut schema = LabelSchema::new_node_id("Raw");
    schema.set_mode(SchemaMode::Flexible);

    let field_names = make_field_names(&[("anything", 1), ("whatever", 2)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("hello".into()));
    props.insert(2, Value::Int(42));

    // FLEXIBLE: all properties accepted, no validation
    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn flexible_mode_skips_type_validation_for_declared() {
    let mut schema = LabelSchema::new_node_id("Raw");
    schema.set_mode(SchemaMode::Flexible);
    schema.add_property(PropertyDef::new("count", PropertyType::Int));

    let field_names = make_field_names(&[("count", 1)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("not int but flexible doesn't care".into()));

    // FLEXIBLE: even declared properties NOT validated
    assert!(validate_properties(&schema, &props, &field_names).is_ok());
}

#[test]
fn strict_mode_rejects_unknown_via_schema_mode() {
    let mut schema = LabelSchema::new_node_id("User");
    schema.set_mode(SchemaMode::Strict);
    schema.add_property(PropertyDef::new("name", PropertyType::String));

    let field_names = make_field_names(&[("name", 1), ("extra", 2)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("Alice".into()));
    props.insert(2, Value::String("undeclared".into()));

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(errors.iter().any(|e| matches!(
        e,
        ValidationError::UnknownProperty { property, .. } if property == "extra"
    )));
}

#[test]
fn computed_property_rejects_set() {
    use crate::schema::computed::{ComputedSpec, DecayFormula};

    let mut schema = LabelSchema::new_node_id("Memory");
    schema.set_mode(SchemaMode::Strict);
    schema.add_property(PropertyDef::new("content", PropertyType::String));
    schema.add_property(PropertyDef::computed(
        "relevance",
        ComputedSpec::Decay {
            formula: DecayFormula::Linear,
            initial: 1.0,
            target: 0.0,
            duration_secs: 604800,
            anchor_field: "created_at".into(),
        },
    ));

    let field_names = make_field_names(&[("content", 1), ("relevance", 2)]);
    let mut props = HashMap::new();
    props.insert(1, Value::String("hello".into()));
    props.insert(2, Value::Float(0.5)); // attempting to SET computed field

    let errors = validate_properties(&schema, &props, &field_names).expect_err("should fail");
    assert!(errors.iter().any(|e| matches!(
        e,
        ValidationError::ComputedReadOnly { property } if property == "relevance"
    )));
}

#[test]
fn computed_property_error_display() {
    let err = ValidationError::ComputedReadOnly {
        property: "relevance".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("COMPUTED"));
    assert!(msg.contains("relevance"));
    assert!(msg.contains("read-only"));
}
