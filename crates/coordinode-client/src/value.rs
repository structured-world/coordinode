//! Client-side value type and proto conversion.

use std::collections::HashMap;

use crate::proto::common::{PropertyValue, property_value};

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
    /// An instant: microseconds since the Unix epoch, negative before 1970.
    Timestamp(i64),
    /// Raw bytes / binary blob.
    Bytes(Vec<u8>),
    /// Dense float vector (used in vector search).
    Vector(Vec<f32>),
    /// Homogeneous list.
    List(Vec<Value>),
    /// String-keyed map.
    Map(HashMap<String, Value>),
    /// Several equal-width vectors describing one item, as late-interaction
    /// retrieval models produce. Distinct from a [`Value::List`] of
    /// [`Value::Vector`], which is merely an array that happens to hold
    /// vectors.
    MultiVector(Vec<Vec<f32>>),
    /// A traversal result: the alternating node and relationship sequence of
    /// `shortestPath(...)` or a variable-length match.
    Path(Path),
}

/// A graph path, as returned by a traversal.
///
/// `rels` has one fewer entry than `nodes` on a non-empty path, and the n-th
/// hop connects `nodes[n]` to `nodes[n + 1]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path {
    /// Node ids along the path, start to end.
    pub nodes: Vec<u64>,
    /// Relationship hops, in the same order.
    pub rels: Vec<PathRel>,
}

/// One relationship hop of a [`Path`]. Relationship properties are not carried;
/// fetch them by endpoint pair when they are needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathRel {
    /// Relationship type.
    pub edge_type: String,
    /// Source node id.
    pub source: u64,
    /// Target node id.
    pub target: u64,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::String(s) => write!(f, "{s}"),
            Value::Timestamp(us) => write_utc(f, *us),
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
            Value::MultiVector(rows) => {
                let width = rows.first().map_or(0, Vec::len);
                write!(f, "<multivec:{}x{}>", rows.len(), width)
            }
            Value::Path(p) => write!(f, "<path:{} hops>", p.rels.len()),
        }
    }
}

/// Write epoch microseconds as an RFC 3339 UTC instant with microsecond
/// fraction. A year outside 0000..=9999 is written with an explicit sign, the
/// ISO 8601 expanded form, since RFC 3339 has no spelling for it.
fn write_utc(f: &mut std::fmt::Formatter<'_>, us: i64) -> std::fmt::Result {
    const US_PER_DAY: i64 = 86_400_000_000;
    let days = us.div_euclid(US_PER_DAY);
    let in_day = us.rem_euclid(US_PER_DAY);
    // Civil date from days since 1970-01-01, proleptic Gregorian (Howard
    // Hinnant, "chrono-Compatible Low-Level Date Algorithms"). |days| is at
    // most about 1.1e8, so none of the products below overflow.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    if (0..=9_999).contains(&year) {
        write!(f, "{year:04}")?;
    } else {
        write!(f, "{year:+}")?;
    }
    let secs = in_day / 1_000_000;
    write!(
        f,
        "-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:06}Z",
        secs / 3_600,
        secs / 60 % 60,
        secs % 60,
        in_day % 1_000_000
    )
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
        Some(property_value::Value::TimestampValue(ts)) => Value::Timestamp(ts.unix_micros),
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
        Some(property_value::Value::MultiVectorValue(mv)) => {
            Value::MultiVector(mv.rows.into_iter().map(|row| row.values).collect())
        }
        Some(property_value::Value::PathValue(p)) => Value::Path(Path {
            nodes: p.nodes,
            rels: p
                .rels
                .into_iter()
                .map(|r| PathRel {
                    edge_type: r.edge_type,
                    source: r.source,
                    target: r.target,
                })
                .collect(),
        }),
    }
}

/// Convert a client-side [`Value`] to a proto `PropertyValue`.
///
/// Used when passing query parameters.
pub(crate) fn to_proto(value: Value) -> PropertyValue {
    use crate::proto::common::{PropertyList, PropertyMap, Timestamp, Vector};

    let v = match value {
        Value::Null => None,
        Value::Bool(b) => Some(property_value::Value::BoolValue(b)),
        Value::Int(i) => Some(property_value::Value::IntValue(i)),
        Value::Float(f) => Some(property_value::Value::FloatValue(f)),
        Value::String(s) => Some(property_value::Value::StringValue(s)),
        Value::Timestamp(unix_micros) => Some(property_value::Value::TimestampValue(Timestamp {
            unix_micros,
        })),
        Value::Bytes(b) => Some(property_value::Value::BytesValue(b)),
        Value::Vector(v) => Some(property_value::Value::VectorValue(Vector { values: v })),
        Value::List(l) => Some(property_value::Value::ListValue(PropertyList {
            values: l.into_iter().map(to_proto).collect(),
        })),
        Value::Map(m) => Some(property_value::Value::MapValue(PropertyMap {
            entries: m.into_iter().map(|(k, v)| (k, to_proto(v))).collect(),
        })),
        Value::MultiVector(rows) => Some(property_value::Value::MultiVectorValue(
            crate::proto::common::MultiVector {
                rows: rows.into_iter().map(|values| Vector { values }).collect(),
            },
        )),
        Value::Path(p) => Some(property_value::Value::PathValue(
            crate::proto::common::Path {
                nodes: p.nodes,
                rels: p
                    .rels
                    .into_iter()
                    .map(|r| crate::proto::common::PathRel {
                        edge_type: r.edge_type,
                        source: r.source,
                        target: r.target,
                    })
                    .collect(),
            },
        )),
    };
    PropertyValue { value: v }
}

#[cfg(test)]
mod tests;
