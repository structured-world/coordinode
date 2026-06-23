//! Property type system: Value enum covering all 12 property types.
//!
//! Every property stored on nodes or edges uses this unified type system.
//! Types are declared in schema and enforced at write time.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Distance metric for vector properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorMetric {
    Cosine,
    L2,
    DotProduct,
    L1,
}

/// Vector MVCC consistency mode.
///
/// Controls how vector search interacts with snapshot isolation.
/// HNSW indexes are in-memory structures representing the current state
/// (not versioned). These modes control visibility filtering of results.
///
/// Configurable per session (`SET vector_consistency = '...'`) or per query
/// via hint syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorConsistencyMode {
    /// No visibility filter. HNSW returns latest state regardless of
    /// transaction `start_ts`. Fastest, best recall. Use for real-time
    /// recommendations and search where strict transactional consistency
    /// is not required.
    #[default]
    Current,

    /// HNSW search + post-filter: discard candidates invisible at `start_ts`.
    /// Overfetch by `overfetch_factor` (default 1.2), then filter. If
    /// remaining < K, expand ef_search and retry (max 3 rounds).
    /// ~99.9% recall, +5-10% latency. Use for transactional consistency.
    Snapshot,

    /// Brute-force scan with MVCC filter (no HNSW). 100% recall,
    /// 10-100x slower. Use for audit, correctness-critical queries,
    /// or small datasets where recall must be perfect.
    Exact,
}

impl VectorConsistencyMode {
    /// Parse from string (case-insensitive).
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "current" => Some(Self::Current),
            "snapshot" => Some(Self::Snapshot),
            "exact" => Some(Self::Exact),
            _ => None,
        }
    }

    /// String representation for EXPLAIN output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Snapshot => "snapshot",
            Self::Exact => "exact",
        }
    }
}

/// Statistics from snapshot-mode vector search for EXPLAIN output.
#[derive(Debug, Clone, Default)]
pub struct VectorMvccStats {
    /// Total candidates fetched from HNSW (including overfetch).
    pub candidates_fetched: usize,
    /// Candidates that passed MVCC visibility check.
    pub candidates_visible: usize,
    /// Candidates filtered out (invisible at snapshot_ts).
    pub candidates_filtered: usize,
    /// Number of expansion rounds needed to fill K results.
    pub expansion_rounds: usize,
    /// Overfetch factor used (default 1.2).
    pub overfetch_factor: f64,
}

/// Geographic value (S2-based geometry).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GeoValue {
    /// A point with latitude and longitude (WGS84).
    Point { lat: f64, lon: f64 },
    // Polygon, LineString, etc. will be added as needed.
}

/// A property value that can be stored on nodes or edges.
///
/// Covers all 12 types from the architecture:
/// STRING, INT, FLOAT, BOOL, TIMESTAMP, VECTOR, BLOB,
/// ARRAY, MAP, GEO, BINARY, NULL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Value {
    /// Explicit null.
    Null,

    /// Boolean value.
    Bool(bool),

    /// 64-bit signed integer.
    Int(i64),

    /// 64-bit IEEE 754 float.
    Float(f64),

    /// UTF-8 string.
    String(String),

    /// Unix timestamp in microseconds (i64).
    Timestamp(i64),

    /// Vector of f32 values (dimensions declared in schema).
    Vector(Vec<f32>),

    /// Reference to BlobStore chunks (SHA-256 content-addressed).
    /// Stored as the serialized `BlobRef` bytes.
    Blob(Vec<u8>),

    /// Homogeneous typed array.
    Array(Vec<Value>),

    /// String-keyed map.
    Map(BTreeMap<String, Value>),

    /// Geographic value (S2-based).
    Geo(GeoValue),

    /// Opaque binary data.
    Binary(Vec<u8>),

    /// Arbitrary nested document (recursive map/array, any depth).
    /// No schema type validation — accepts any MessagePack structure.
    /// Used for semi-structured data that doesn't fit a flat property model.
    /// 4MB size limit (configurable). See document-operations arch doc.
    Document(rmpv::Value),

    /// Multi-vector value: ordered list of per-token f32 vectors with
    /// identical dimensionality. Backs late-interaction retrieval models
    /// (ColBERT v2, MaxSim) where a document is represented by several
    /// token-level embeddings instead of a single pooled vector.
    ///
    /// Construction-time invariant: every row must have the same length.
    /// Use [`Value::try_multi_vector`] to enforce this.
    MultiVector(Vec<Vec<f32>>),

    /// A graph path: the alternating node/relationship sequence produced by
    /// `shortestPath(...)` and variable-length `MATCH p = (a)-[*]->(b)`.
    /// `length(p)` is the relationship count; `nodes(p)` / `relationships(p)`
    /// project the two sequences. Maps to the Bolt Path structure on the wire.
    Path(PathValue),
}

/// One relationship hop inside a [`PathValue`]: its type and endpoint node ids
/// (raw [`NodeId`] values). Properties are not carried in v1 of the path model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PathRel {
    /// Relationship type (edge label).
    pub edge_type: String,
    /// Source node id (raw).
    pub source: u64,
    /// Target node id (raw).
    pub target: u64,
}

/// A graph path as an alternating node/relationship sequence.
///
/// Invariant: `nodes.len() == rels.len() + 1` for a non-empty path; a
/// zero-length path is a single node with no relationships. The `i`-th
/// relationship connects `nodes[i]` to `nodes[i + 1]` (following the traversal
/// direction). `length(p)` is `rels.len()`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PathValue {
    /// Ordered node ids along the path (raw [`NodeId`] values), start to end.
    pub nodes: Vec<u64>,
    /// Ordered relationship hops, one fewer than `nodes` on a non-empty path.
    pub rels: Vec<PathRel>,
}

/// Extract an owned f32 vector from a [`Value`], accepting both the native
/// [`Value::Vector`] and a numeric [`Value::Array`] of `Float`/`Int` elements
/// (the shape a Cypher array literal like `[1.0, 0.0]` parses to). Returns
/// `None` for any other variant or an empty array. Canonical vector coercion
/// shared by index build/backfill, the write path, and extension handlers.
pub fn try_extract_vector(val: &Value) -> Option<Vec<f32>> {
    match val {
        Value::Vector(v) => Some(v.clone()),
        Value::Array(arr) => {
            let mut vec = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    Value::Float(f) => vec.push(*f as f32),
                    Value::Int(i) => vec.push(*i as f32),
                    _ => return None,
                }
            }
            if vec.is_empty() {
                None
            } else {
                Some(vec)
            }
        }
        _ => None,
    }
}

impl Value {
    /// Returns the type name as a static string.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "NULL",
            Self::Bool(_) => "BOOL",
            Self::Int(_) => "INT",
            Self::Float(_) => "FLOAT",
            Self::String(_) => "STRING",
            Self::Timestamp(_) => "TIMESTAMP",
            Self::Vector(_) => "VECTOR",
            Self::Blob(_) => "BLOB",
            Self::Array(_) => "ARRAY",
            Self::Map(_) => "MAP",
            Self::Geo(_) => "GEO",
            Self::Binary(_) => "BINARY",
            Self::Document(_) => "DOCUMENT",
            Self::MultiVector(_) => "MULTIVECTOR",
            Self::Path(_) => "PATH",
        }
    }

    /// Convert to `rmpv::Value` for use in DocDelta merge operands.
    pub fn to_rmpv(&self) -> rmpv::Value {
        match self {
            Self::Null => rmpv::Value::Nil,
            Self::Bool(b) => rmpv::Value::Boolean(*b),
            Self::Int(i) => rmpv::Value::Integer((*i).into()),
            Self::Float(f) => rmpv::Value::F64(*f),
            Self::String(s) => rmpv::Value::String(s.clone().into()),
            Self::Timestamp(t) => rmpv::Value::Integer((*t).into()),
            Self::Vector(v) => {
                rmpv::Value::Array(v.iter().map(|f| rmpv::Value::F64(*f as f64)).collect())
            }
            Self::Blob(b) => rmpv::Value::Binary(b.clone()),
            Self::Binary(b) => rmpv::Value::Binary(b.clone()),
            Self::Array(arr) => rmpv::Value::Array(arr.iter().map(|v| v.to_rmpv()).collect()),
            Self::Map(map) => rmpv::Value::Map(
                map.iter()
                    .map(|(k, v)| (rmpv::Value::String(k.clone().into()), v.to_rmpv()))
                    .collect(),
            ),
            Self::Geo(g) => match g {
                GeoValue::Point { lat, lon } => rmpv::Value::Map(vec![
                    (
                        rmpv::Value::String("latitude".into()),
                        rmpv::Value::F64(*lat),
                    ),
                    (
                        rmpv::Value::String("longitude".into()),
                        rmpv::Value::F64(*lon),
                    ),
                ]),
            },
            Self::Document(v) => v.clone(),
            Self::MultiVector(rows) => rmpv::Value::Array(
                rows.iter()
                    .map(|row| {
                        rmpv::Value::Array(
                            row.iter().map(|f| rmpv::Value::F64(*f as f64)).collect(),
                        )
                    })
                    .collect(),
            ),
            Self::Path(p) => {
                let nodes = rmpv::Value::Array(
                    p.nodes
                        .iter()
                        .map(|n| rmpv::Value::Integer((*n).into()))
                        .collect(),
                );
                let rels = rmpv::Value::Array(
                    p.rels
                        .iter()
                        .map(|r| {
                            rmpv::Value::Map(vec![
                                (
                                    rmpv::Value::String("type".into()),
                                    rmpv::Value::String(r.edge_type.clone().into()),
                                ),
                                (
                                    rmpv::Value::String("source".into()),
                                    rmpv::Value::Integer(r.source.into()),
                                ),
                                (
                                    rmpv::Value::String("target".into()),
                                    rmpv::Value::Integer(r.target.into()),
                                ),
                            ])
                        })
                        .collect(),
                );
                rmpv::Value::Map(vec![
                    (rmpv::Value::String("nodes".into()), nodes),
                    (rmpv::Value::String("rels".into()), rels),
                ])
            }
        }
    }

    /// Convert `Value::Map` to `Value::Document` for storage as nested document.
    ///
    /// Map literals in Cypher (`{key: 'value'}`) evaluate to `Value::Map`,
    /// but when stored as node properties they should be `Value::Document`
    /// to support full dot-notation traversal and merge operators.
    ///
    /// Non-Map values are returned unchanged.
    pub fn map_to_document(self) -> Self {
        if matches!(self, Self::Map(_)) {
            let rmpv = self.to_rmpv();
            Self::Document(rmpv)
        } else {
            self
        }
    }

    /// Check if this is a null value.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Try to get as i64.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as string slice.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as vector slice.
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            Self::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as timestamp (Unix micros).
    pub fn as_timestamp(&self) -> Option<i64> {
        match self {
            Self::Timestamp(t) => Some(*t),
            _ => None,
        }
    }

    /// Try to get as document (rmpv::Value reference).
    pub fn as_document(&self) -> Option<&rmpv::Value> {
        match self {
            Self::Document(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get as multi-vector slice (per-token f32 rows).
    pub fn as_multi_vector(&self) -> Option<&[Vec<f32>]> {
        match self {
            Self::MultiVector(rows) => Some(rows),
            _ => None,
        }
    }

    /// Build a `Value::MultiVector` enforcing equal row width. Returns
    /// `None` if `rows` is empty or if any row's length differs from the
    /// first row's. A zero-row or zero-dim multi-vector has no recoverable
    /// dimensionality and scoring against it is undefined.
    pub fn try_multi_vector(rows: Vec<Vec<f32>>) -> Option<Self> {
        let dim = rows.first()?.len();
        if dim == 0 {
            return None;
        }
        if rows.iter().any(|row| row.len() != dim) {
            return None;
        }
        Some(Self::MultiVector(rows))
    }

    /// Default maximum DOCUMENT property size in bytes (4MB).
    pub const DOCUMENT_MAX_SIZE: usize = 4 * 1024 * 1024;

    /// Returns the approximate serialized size of a DOCUMENT value.
    /// Used for write-path validation against size limits.
    pub fn document_serialized_size(&self) -> Option<usize> {
        match self {
            Self::Document(v) => {
                let mut buf = Vec::new();
                // write_value always succeeds on Vec<u8>
                let _ = rmpv::encode::write_value(&mut buf, v);
                Some(buf.len())
            }
            _ => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
