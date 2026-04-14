//! Write-time type validation for node and edge properties.
//!
//! Validates property values against schema declarations before mutations
//! enter the Raft log. Catches type mismatches, NOT NULL violations,
//! vector dimension errors, and array homogeneity violations.

use std::collections::HashMap;

use crate::graph::types::Value;
use crate::schema::definition::{LabelSchema, PropertyDef, PropertyType};

/// Validation error for property values.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// Property value type doesn't match schema declaration.
    TypeMismatch {
        property: String,
        expected: String,
        got: String,
    },

    /// Required property (NOT NULL) is missing or null.
    NotNullViolation { property: String },

    /// Vector dimensions don't match schema declaration.
    VectorDimsMismatch {
        property: String,
        expected: u32,
        got: usize,
    },

    /// Array elements are not all the same type as declared.
    ArrayNotHomogeneous {
        property: String,
        expected_element: String,
        got_element: String,
    },

    /// Unschematized property in strict mode.
    UnknownProperty { property: String, label: String },

    /// DOCUMENT property exceeds maximum size limit.
    DocumentTooLarge {
        property: String,
        max_bytes: usize,
        got_bytes: usize,
    },

    /// Attempt to SET a COMPUTED (read-only) property.
    ComputedReadOnly { property: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeMismatch {
                property,
                expected,
                got,
            } => write!(f, "type mismatch for '{property}': expected {expected}, got {got}"),
            Self::NotNullViolation { property } => {
                write!(f, "NOT NULL violation: '{property}' is required")
            }
            Self::VectorDimsMismatch {
                property,
                expected,
                got,
            } => write!(
                f,
                "vector dimension mismatch for '{property}': expected {expected}, got {got}"
            ),
            Self::ArrayNotHomogeneous {
                property,
                expected_element,
                got_element,
            } => write!(
                f,
                "array not homogeneous for '{property}': expected {expected_element}, got {got_element}"
            ),
            Self::UnknownProperty { property, label } => {
                write!(f, "unknown property '{property}' for strict label '{label}'")
            }
            Self::DocumentTooLarge {
                property,
                max_bytes,
                got_bytes,
            } => write!(
                f,
                "DOCUMENT property '{property}' exceeds size limit: {got_bytes} bytes > {max_bytes} bytes"
            ),
            Self::ComputedReadOnly { property } => {
                write!(f, "cannot SET COMPUTED property '{property}' (read-only)")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Check if a `Value` matches a `PropertyType`.
fn value_matches_type(value: &Value, expected: &PropertyType) -> bool {
    match (value, expected) {
        (Value::Null, _) => true, // null is always valid (NOT NULL checked separately)
        (Value::String(_), PropertyType::String) => true,
        (Value::Int(_), PropertyType::Int) => true,
        (Value::Float(_), PropertyType::Float) => true,
        (Value::Bool(_), PropertyType::Bool) => true,
        (Value::Timestamp(_), PropertyType::Timestamp) => true,
        (Value::Vector(_), PropertyType::Vector { .. }) => true, // dims checked separately
        (Value::Blob(_), PropertyType::Blob) => true,
        (Value::Array(_), PropertyType::Array(_)) => true, // homogeneity checked separately
        (Value::Map(_), PropertyType::Map) => true,
        (Value::Geo(_), PropertyType::Geo) => true,
        (Value::Binary(_), PropertyType::Binary) => true,
        (Value::Document(_), PropertyType::Document) => true,
        // Computed properties never match any value — they are read-only.
        (_, PropertyType::Computed(_)) => false,
        _ => false,
    }
}

/// Validate a single property value against its schema definition.
///
/// Called per-property at write time in STRICT and VALIDATED modes. Checks type
/// compatibility, NOT NULL constraints, vector dimensions, and array homogeneity.
/// Returns the first validation error found for this property.
pub fn validate_one(
    prop_name: &str,
    value: &Value,
    def: &PropertyDef,
) -> Result<(), ValidationError> {
    validate_property(prop_name, value, def)
}

/// Validate a single property value against its definition.
fn validate_property(
    prop_name: &str,
    value: &Value,
    def: &PropertyDef,
) -> Result<(), ValidationError> {
    // COMPUTED properties are read-only — reject any attempt to SET them.
    if def.is_computed() {
        return Err(ValidationError::ComputedReadOnly {
            property: prop_name.to_string(),
        });
    }

    // NOT NULL check
    if def.not_null && value.is_null() {
        return Err(ValidationError::NotNullViolation {
            property: prop_name.to_string(),
        });
    }

    // Null is always valid type-wise (NOT NULL already checked above)
    if value.is_null() {
        return Ok(());
    }

    // Type check
    if !value_matches_type(value, &def.property_type) {
        return Err(ValidationError::TypeMismatch {
            property: prop_name.to_string(),
            expected: def.property_type.to_string(),
            got: value.type_name().to_string(),
        });
    }

    // Vector dimension check.
    // dimensions=0 means "unset" (e.g. gRPC proto PropertyDefinition has no dimensions
    // field — SchemaService/CreateLabel always writes 0). Treat 0 as "any dimension".
    if let (Value::Vector(vec), PropertyType::Vector { dimensions, .. }) =
        (value, &def.property_type)
    {
        if *dimensions != 0 && vec.len() != *dimensions as usize {
            return Err(ValidationError::VectorDimsMismatch {
                property: prop_name.to_string(),
                expected: *dimensions,
                got: vec.len(),
            });
        }
    }

    // Array homogeneity check
    if let (Value::Array(elements), PropertyType::Array(elem_type)) = (value, &def.property_type) {
        for elem in elements {
            if !elem.is_null() && !value_matches_type(elem, elem_type) {
                return Err(ValidationError::ArrayNotHomogeneous {
                    property: prop_name.to_string(),
                    expected_element: elem_type.to_string(),
                    got_element: elem.type_name().to_string(),
                });
            }
        }
    }

    // Document size check (4MB default limit)
    if let Value::Document(_) = value {
        if let Some(size) = value.document_serialized_size() {
            if size > Value::DOCUMENT_MAX_SIZE {
                return Err(ValidationError::DocumentTooLarge {
                    property: prop_name.to_string(),
                    max_bytes: Value::DOCUMENT_MAX_SIZE,
                    got_bytes: size,
                });
            }
        }
    }

    Ok(())
}

/// Validate node properties against a label schema.
///
/// Checks:
/// 1. Each provided property matches its declared type
/// 2. NOT NULL properties are present and non-null
/// 3. Vector dimensions match declarations
/// 4. Array elements are homogeneous
/// 5. In strict mode, no unschematized properties allowed
///
/// `props` maps interned field IDs to values.
/// `field_names` maps interned field IDs to property names (for error messages).
pub fn validate_properties(
    schema: &LabelSchema,
    props: &HashMap<u32, Value>,
    field_names: &HashMap<u32, String>,
) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // Check each provided property against schema
    for (&field_id, value) in props {
        let prop_name = field_names
            .get(&field_id)
            .map(|s| s.as_str())
            .unwrap_or("?");

        if let Some(def) = schema.get_property(prop_name) {
            // Declared property — validate type if mode supports it
            if schema.mode.validates_declared() {
                if let Err(e) = validate_property(prop_name, value, def) {
                    errors.push(e);
                }
            }
        } else if schema.mode.rejects_unknown() {
            // Strict mode: reject undeclared properties
            errors.push(ValidationError::UnknownProperty {
                property: prop_name.to_string(),
                label: schema.name.clone(),
            });
            // VALIDATED mode: undeclared accepted (stored in _extra overflow)
            // FLEXIBLE mode: all properties accepted without validation
        }
    }

    // Check NOT NULL for missing properties
    for (prop_name, def) in &schema.properties {
        if def.not_null && def.default.is_none() {
            // Find if this property is provided
            let provided = field_names
                .iter()
                .any(|(id, name)| name == prop_name && props.contains_key(id));

            if !provided {
                errors.push(ValidationError::NotNullViolation {
                    property: prop_name.clone(),
                });
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
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
        let mut schema = LabelSchema::new("User");
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
        let mut schema = LabelSchema::new("Movie");
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
        let mut schema = LabelSchema::new("Movie");
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
        let mut schema = LabelSchema::new("User");
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
        let mut schema = LabelSchema::new("User");
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
        let mut schema = LabelSchema::new("User");
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
        schema.set_strict(true);

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
        let mut schema = LabelSchema::new("Config");
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
        let mut schema = LabelSchema::new("Config");
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
        let mut schema = LabelSchema::new("Config");
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
        let mut schema = LabelSchema::new("Config");
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
        let mut schema = LabelSchema::new("Device");
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
        let mut schema = LabelSchema::new("Device");
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
        let mut schema = LabelSchema::new("Raw");
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
        let mut schema = LabelSchema::new("Raw");
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
        let mut schema = LabelSchema::new("User");
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

        let mut schema = LabelSchema::new("Memory");
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
}
