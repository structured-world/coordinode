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
mod tests;
