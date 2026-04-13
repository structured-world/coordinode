//! Client-side value type and proto conversion.

use std::collections::HashMap;

use crate::proto::common::{property_value, PropertyValue};

/// A typed property value returned by a Cypher query.
///
/// This is a client-side mirror of the server's `coordinode_core::graph::types::Value`
/// and the proto `PropertyValue`. It avoids pulling internal server crates into
/// client builds.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// SQL NULL / absence of value.
    Null,
    /// Boolean.
    Bool(bool),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit float.
    Float(f64),
    /// UTF-8 string.
    String(String),
    /// Raw bytes / binary blob.
    Bytes(Vec<u8>),
    /// Dense float vector (used in vector search).
    Vector(Vec<f32>),
    /// Homogeneous list.
    List(Vec<Value>),
    /// String-keyed map.
    Map(HashMap<String, Value>),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::String(s) => write!(f, "{s}"),
            Value::Bytes(b) => write!(f, "<bytes:{}>", b.len()),
            Value::Vector(v) => write!(f, "<vec:{}>", v.len()),
            Value::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// Convert a proto `PropertyValue` to a client-side [`Value`].
pub(crate) fn from_proto(pv: PropertyValue) -> Value {
    match pv.value {
        None => Value::Null,
        Some(property_value::Value::IntValue(i)) => Value::Int(i),
        Some(property_value::Value::FloatValue(f)) => Value::Float(f),
        Some(property_value::Value::StringValue(s)) => Value::String(s),
        Some(property_value::Value::BoolValue(b)) => Value::Bool(b),
        Some(property_value::Value::BytesValue(b)) => Value::Bytes(b),
        Some(property_value::Value::TimestampValue(ts)) => Value::Int(ts.wall_time as i64),
        Some(property_value::Value::VectorValue(v)) => Value::Vector(v.values),
        Some(property_value::Value::ListValue(l)) => {
            Value::List(l.values.into_iter().map(from_proto).collect())
        }
        Some(property_value::Value::MapValue(m)) => Value::Map(
            m.entries
                .into_iter()
                .map(|(k, v)| (k, from_proto(v)))
                .collect(),
        ),
    }
}

/// Convert a client-side [`Value`] to a proto `PropertyValue`.
///
/// Used when passing query parameters.
pub(crate) fn to_proto(value: Value) -> PropertyValue {
    use crate::proto::common::{HlcTimestamp, PropertyList, PropertyMap, Vector};

    let v = match value {
        Value::Null => None,
        Value::Bool(b) => Some(property_value::Value::BoolValue(b)),
        Value::Int(i) => Some(property_value::Value::IntValue(i)),
        Value::Float(f) => Some(property_value::Value::FloatValue(f)),
        Value::String(s) => Some(property_value::Value::StringValue(s)),
        Value::Bytes(b) => Some(property_value::Value::BytesValue(b)),
        Value::Vector(v) => Some(property_value::Value::VectorValue(Vector { values: v })),
        Value::List(l) => Some(property_value::Value::ListValue(PropertyList {
            values: l.into_iter().map(to_proto).collect(),
        })),
        Value::Map(m) => Some(property_value::Value::MapValue(PropertyMap {
            entries: m.into_iter().map(|(k, v)| (k, to_proto(v))).collect(),
        })),
    };
    // Timestamps without wall_time are emitted as Null; client-side we
    // don't expose HlcTimestamp directly.
    let _ = HlcTimestamp::default;
    PropertyValue { value: v }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: proto → Value → proto preserves scalar types.
    #[test]
    fn round_trip_scalars() {
        let cases: Vec<PropertyValue> = vec![
            PropertyValue {
                value: Some(property_value::Value::IntValue(42)),
            },
            PropertyValue {
                value: Some(property_value::Value::FloatValue(1.5)),
            },
            PropertyValue {
                value: Some(property_value::Value::StringValue("hello".into())),
            },
            PropertyValue {
                value: Some(property_value::Value::BoolValue(true)),
            },
            PropertyValue { value: None },
        ];

        for pv in cases {
            let v = from_proto(pv.clone());
            let back = to_proto(v);
            assert_eq!(pv, back);
        }
    }

    /// Display impl produces readable output.
    #[test]
    fn display_values() {
        assert_eq!(Value::Null.to_string(), "null");
        assert_eq!(Value::Bool(false).to_string(), "false");
        assert_eq!(Value::Int(-7).to_string(), "-7");
        assert_eq!(Value::String("hi".into()).to_string(), "hi");
        assert_eq!(
            Value::List(vec![Value::Int(1), Value::Int(2)]).to_string(),
            "[1, 2]"
        );
    }
}
