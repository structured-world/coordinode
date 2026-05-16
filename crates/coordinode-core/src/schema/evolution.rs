//! Schema evolution: non-destructive schema changes without migration downtime.
//!
//! Schema changes increment a version counter. Each operation is validated
//! to ensure backward compatibility:
//! - Adding properties is always safe (existing nodes get NULL/default)
//! - Removing declarations is safe (values remain on disk)
//! - Changing property types is rejected (create new property instead)
//! - Adding NOT NULL without default is rejected (existing nulls would violate)

use crate::graph::types::Value;
use crate::schema::definition::{LabelSchema, PropertyDef, PropertyType};

/// A schema evolution operation.
#[derive(Debug, Clone)]
pub enum SchemaChange {
    /// Add a new property. Existing nodes will have NULL (or default) for this property.
    AddProperty(PropertyDef),

    /// Remove a property declaration. Existing values remain on disk but are
    /// no longer validated or interned.
    RemoveProperty(String),
}

/// Error from applying a schema change.
#[derive(Debug, Clone, PartialEq)]
pub enum EvolutionError {
    /// Cannot change property type. Create a new property instead.
    TypeChangeRejected {
        property: String,
        from: String,
        to: String,
    },

    /// Cannot add NOT NULL constraint without a default value on a property
    /// that may already have nulls in existing data.
    NotNullWithoutDefault { property: String },

    /// Property already exists with same name.
    PropertyAlreadyExists { property: String },

    /// Property does not exist (for removal).
    PropertyNotFound { property: String },
}

impl std::fmt::Display for EvolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeChangeRejected { property, from, to } => {
                write!(f, "cannot change type of '{property}' from {from} to {to}")
            }
            Self::NotNullWithoutDefault { property } => {
                write!(
                    f,
                    "cannot add NOT NULL to '{property}' without a DEFAULT value"
                )
            }
            Self::PropertyAlreadyExists { property } => {
                write!(f, "property '{property}' already exists")
            }
            Self::PropertyNotFound { property } => {
                write!(f, "property '{property}' not found")
            }
        }
    }
}

impl std::error::Error for EvolutionError {}

/// Apply a schema change to a label schema.
///
/// Returns the updated schema or an error if the change is rejected.
/// The schema revision is incremented on success.
pub fn apply_change(schema: &mut LabelSchema, change: SchemaChange) -> Result<(), EvolutionError> {
    match change {
        SchemaChange::AddProperty(prop) => {
            // Check if property already exists
            if let Some(existing) = schema.get_property(&prop.name) {
                // If same type, treat as idempotent (no-op)
                if existing.property_type == prop.property_type {
                    return Ok(());
                }
                // Different type → type change rejected
                return Err(EvolutionError::TypeChangeRejected {
                    property: prop.name.clone(),
                    from: existing.property_type.to_string(),
                    to: prop.property_type.to_string(),
                });
            }

            // NOT NULL without default is rejected for new properties
            // (existing data won't have this property → would be NULL → violation)
            if prop.not_null && prop.default.is_none() {
                return Err(EvolutionError::NotNullWithoutDefault {
                    property: prop.name.clone(),
                });
            }

            schema.add_property(prop);
            Ok(())
        }
        SchemaChange::RemoveProperty(name) => {
            if schema.get_property(&name).is_none() {
                return Err(EvolutionError::PropertyNotFound { property: name });
            }
            schema.remove_property(&name);
            Ok(())
        }
    }
}

/// Check if a proposed type change is compatible (it never is — always rejected).
pub fn is_type_change_allowed(_from: &PropertyType, _to: &PropertyType) -> bool {
    // Per architecture: "Change property type — Rejected. Create new property, migrate data."
    false
}

/// Resolve a property value considering schema defaults.
///
/// If the property is missing (None) and the schema has a default,
/// returns the default value. Otherwise returns the original value.
pub fn resolve_with_default<'a>(
    value: Option<&'a Value>,
    prop_def: &'a PropertyDef,
) -> Option<&'a Value> {
    match value {
        Some(v) => Some(v),
        None => prop_def.default.as_ref(),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::graph::types::VectorMetric;

    fn base_schema() -> LabelSchema {
        let mut schema = LabelSchema::new_node_id("User");
        schema.add_property(PropertyDef::new("name", PropertyType::String).not_null());
        schema.add_property(PropertyDef::new("age", PropertyType::Int));
        schema
    }

    #[test]
    fn add_property_simple() {
        let mut schema = base_schema();
        let v_before = schema.schema_revision;

        let result = apply_change(
            &mut schema,
            SchemaChange::AddProperty(PropertyDef::new("bio", PropertyType::String)),
        );
        assert!(result.is_ok());
        assert!(schema.get_property("bio").is_some());
        // Per ADR-023, property additions do not bump the schema revision —
        // version tracks placement/shard_keys changes via `ALTER LABEL`.
        assert_eq!(schema.schema_revision, v_before);
    }

    #[test]
    fn add_property_with_default() {
        let mut schema = base_schema();

        let result = apply_change(
            &mut schema,
            SchemaChange::AddProperty(
                PropertyDef::new("status", PropertyType::String)
                    .with_default(Value::String("active".into())),
            ),
        );
        assert!(result.is_ok());
        let prop = schema.get_property("status").expect("should exist");
        assert_eq!(prop.default, Some(Value::String("active".into())));
    }

    #[test]
    fn add_property_not_null_with_default_ok() {
        let mut schema = base_schema();

        let result = apply_change(
            &mut schema,
            SchemaChange::AddProperty(
                PropertyDef::new("email", PropertyType::String)
                    .not_null()
                    .with_default(Value::String("unknown@example.com".into())),
            ),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn add_property_not_null_without_default_rejected() {
        let mut schema = base_schema();

        let result = apply_change(
            &mut schema,
            SchemaChange::AddProperty(PropertyDef::new("email", PropertyType::String).not_null()),
        );
        assert!(matches!(
            result,
            Err(EvolutionError::NotNullWithoutDefault { .. })
        ));
    }

    #[test]
    fn type_change_rejected() {
        let mut schema = base_schema();

        // Try to change "age" from INT to STRING
        let result = apply_change(
            &mut schema,
            SchemaChange::AddProperty(PropertyDef::new("age", PropertyType::String)),
        );
        assert!(matches!(
            result,
            Err(EvolutionError::TypeChangeRejected { .. })
        ));
    }

    #[test]
    fn type_change_same_type_idempotent() {
        let mut schema = base_schema();

        // Adding "age" as INT again should be idempotent
        let result = apply_change(
            &mut schema,
            SchemaChange::AddProperty(PropertyDef::new("age", PropertyType::Int)),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn remove_property() {
        let mut schema = base_schema();
        let v_before = schema.schema_revision;

        let result = apply_change(&mut schema, SchemaChange::RemoveProperty("age".to_string()));
        assert!(result.is_ok());
        assert!(schema.get_property("age").is_none());
        // Per ADR-023, property removals do not bump the schema revision.
        assert_eq!(schema.schema_revision, v_before);
    }

    #[test]
    fn remove_nonexistent_property() {
        let mut schema = base_schema();

        let result = apply_change(
            &mut schema,
            SchemaChange::RemoveProperty("nonexistent".to_string()),
        );
        assert!(matches!(
            result,
            Err(EvolutionError::PropertyNotFound { .. })
        ));
    }

    #[test]
    fn is_type_change_always_rejected() {
        assert!(!is_type_change_allowed(
            &PropertyType::String,
            &PropertyType::Int
        ));
        assert!(!is_type_change_allowed(
            &PropertyType::Int,
            &PropertyType::Int
        ));
        assert!(!is_type_change_allowed(
            &PropertyType::Vector {
                dimensions: 384,
                metric: VectorMetric::Cosine
            },
            &PropertyType::Vector {
                dimensions: 128,
                metric: VectorMetric::L2
            },
        ));
    }

    #[test]
    fn resolve_with_default_present() {
        let prop = PropertyDef::new("status", PropertyType::String)
            .with_default(Value::String("active".into()));

        let val = Value::String("custom".into());
        assert_eq!(resolve_with_default(Some(&val), &prop), Some(&val));
    }

    #[test]
    fn resolve_with_default_missing() {
        let prop = PropertyDef::new("status", PropertyType::String)
            .with_default(Value::String("active".into()));

        let result = resolve_with_default(None, &prop);
        assert_eq!(result, Some(&Value::String("active".into())));
    }

    #[test]
    fn resolve_with_default_missing_no_default() {
        let prop = PropertyDef::new("bio", PropertyType::String);

        let result = resolve_with_default(None, &prop);
        assert_eq!(result, None);
    }

    #[test]
    fn evolution_error_display() {
        let err = EvolutionError::TypeChangeRejected {
            property: "age".into(),
            from: "INT".into(),
            to: "STRING".into(),
        };
        assert_eq!(
            err.to_string(),
            "cannot change type of 'age' from INT to STRING"
        );
    }

    #[test]
    fn multiple_changes_sequential() {
        let mut schema = base_schema();

        apply_change(
            &mut schema,
            SchemaChange::AddProperty(PropertyDef::new("bio", PropertyType::String)),
        )
        .expect("add bio");

        apply_change(
            &mut schema,
            SchemaChange::AddProperty(PropertyDef::new(
                "embedding",
                PropertyType::Vector {
                    dimensions: 384,
                    metric: VectorMetric::Cosine,
                },
            )),
        )
        .expect("add embedding");

        apply_change(&mut schema, SchemaChange::RemoveProperty("age".to_string()))
            .expect("remove age");

        assert_eq!(schema.properties.len(), 3); // name, bio, embedding
        assert!(schema.get_property("age").is_none());
        assert!(schema.get_property("bio").is_some());
        assert!(schema.get_property("embedding").is_some());
    }
}
