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
mod tests;
