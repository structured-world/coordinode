//! The predicate a partial index, and the uniqueness constraint owning one,
//! is scoped to: only nodes satisfying it are indexed and constrained.

use serde::{Deserialize, Serialize};

use crate::graph::types::Value;

/// A partial index filter predicate.
///
/// Only nodes satisfying this filter are included in the index.
/// Stored as a serializable enum of common filter patterns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartialFilter {
    /// Property equals a specific string value.
    PropertyEquals {
        /// Property name to test.
        property: String,
        /// String value the property must equal.
        value: String,
    },
    /// Property equals a specific integer value.
    PropertyEqualsInt {
        /// Property name to test.
        property: String,
        /// Integer value the property must equal.
        value: i64,
    },
    /// Property equals a specific boolean value.
    PropertyEqualsBool {
        /// Property name to test.
        property: String,
        /// Boolean value the property must equal.
        value: bool,
    },
    /// Property is not null (EXISTS).
    PropertyExists {
        /// Property name that must be present and non-null.
        property: String,
    },
}

impl PartialFilter {
    /// The property the filter tests.
    pub fn property(&self) -> &str {
        match self {
            Self::PropertyEquals { property, .. }
            | Self::PropertyEqualsInt { property, .. }
            | Self::PropertyEqualsBool { property, .. }
            | Self::PropertyExists { property } => property,
        }
    }

    /// Evaluate the filter against a set of property values.
    pub fn matches(&self, properties: &[(String, Value)]) -> bool {
        match self {
            Self::PropertyEquals { property, value } => properties
                .iter()
                .any(|(k, v)| k == property && v.as_str() == Some(value.as_str())),
            Self::PropertyEqualsInt { property, value } => properties
                .iter()
                .any(|(k, v)| k == property && v.as_int() == Some(*value)),
            Self::PropertyEqualsBool { property, value } => properties
                .iter()
                .any(|(k, v)| k == property && v.as_bool() == Some(*value)),
            Self::PropertyExists { property } => properties
                .iter()
                .any(|(k, v)| k == property && !v.is_null()),
        }
    }
}

/// The predicate as Cypher writes it, on node variable `n`.
impl core::fmt::Display for PartialFilter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PropertyEquals { property, value } => {
                write!(
                    f,
                    "n.{property} = '{}'",
                    value.replace('\\', "\\\\").replace('\'', "\\'")
                )
            }
            Self::PropertyEqualsInt { property, value } => write!(f, "n.{property} = {value}"),
            Self::PropertyEqualsBool { property, value } => write!(f, "n.{property} = {value}"),
            Self::PropertyExists { property } => write!(f, "n.{property} IS NOT NULL"),
        }
    }
}
