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
mod tests {
    use super::*;

    #[test]
    fn type_names() {
        assert_eq!(Value::Null.type_name(), "NULL");
        assert_eq!(Value::Bool(true).type_name(), "BOOL");
        assert_eq!(Value::Int(42).type_name(), "INT");
        assert_eq!(Value::Float(2.72).type_name(), "FLOAT");
        assert_eq!(Value::String("hi".into()).type_name(), "STRING");
        assert_eq!(Value::Timestamp(1000000).type_name(), "TIMESTAMP");
        assert_eq!(Value::Vector(vec![1.0, 2.0]).type_name(), "VECTOR");
        assert_eq!(Value::Blob(vec![]).type_name(), "BLOB");
        assert_eq!(Value::Array(vec![]).type_name(), "ARRAY");
        assert_eq!(Value::Map(BTreeMap::new()).type_name(), "MAP");
        assert_eq!(
            Value::Geo(GeoValue::Point { lat: 0.0, lon: 0.0 }).type_name(),
            "GEO"
        );
        assert_eq!(Value::Binary(vec![]).type_name(), "BINARY");
    }

    #[test]
    fn is_null() {
        assert!(Value::Null.is_null());
        assert!(!Value::Int(0).is_null());
    }

    #[test]
    fn accessor_methods() {
        assert_eq!(Value::Int(42).as_int(), Some(42));
        assert_eq!(Value::Float(2.72).as_float(), Some(2.72));
        assert_eq!(Value::String("hi".into()).as_str(), Some("hi"));
        assert_eq!(Value::Bool(true).as_bool(), Some(true));
        assert_eq!(
            Value::Vector(vec![1.0, 2.0]).as_vector(),
            Some([1.0f32, 2.0].as_slice())
        );
        assert_eq!(
            Value::Timestamp(1234567890).as_timestamp(),
            Some(1234567890)
        );

        // Wrong type returns None
        assert_eq!(Value::String("hi".into()).as_int(), None);
        assert_eq!(Value::Int(42).as_str(), None);
    }

    #[test]
    fn msgpack_roundtrip_all_types() {
        let values = vec![
            Value::Null,
            Value::Bool(false),
            Value::Int(i64::MAX),
            Value::Float(f64::MIN),
            Value::String("hello world".into()),
            Value::Timestamp(1_700_000_000_000_000),
            Value::Vector(vec![0.1, 0.2, 0.3, 0.4]),
            Value::Blob(vec![0xCA, 0xFE, 0xBA, 0xBE]),
            Value::Array(vec![Value::Int(1), Value::String("two".into())]),
            Value::Map({
                let mut m = BTreeMap::new();
                m.insert("key".into(), Value::Int(42));
                m
            }),
            Value::Geo(GeoValue::Point {
                lat: 48.8566,
                lon: 2.3522,
            }),
            Value::Binary(vec![0xFF, 0x00]),
            Value::Document(rmpv::Value::Map(vec![(
                rmpv::Value::String("nested".into()),
                rmpv::Value::Map(vec![(
                    rmpv::Value::String("key".into()),
                    rmpv::Value::Integer(42.into()),
                )]),
            )])),
        ];

        for val in &values {
            let bytes = rmp_serde::to_vec(val).expect("serialize");
            let restored: Value = rmp_serde::from_slice(&bytes).expect("deserialize");
            assert_eq!(val, &restored, "roundtrip failed for {}", val.type_name());
        }
    }

    #[test]
    fn vector_metric_roundtrip() {
        let metrics = vec![
            VectorMetric::Cosine,
            VectorMetric::L2,
            VectorMetric::DotProduct,
            VectorMetric::L1,
        ];
        for m in &metrics {
            let bytes = rmp_serde::to_vec(m).expect("serialize");
            let restored: VectorMetric = rmp_serde::from_slice(&bytes).expect("deserialize");
            assert_eq!(m, &restored);
        }
    }

    #[test]
    fn geo_point_roundtrip() {
        let geo = GeoValue::Point {
            lat: 55.7558,
            lon: 37.6173,
        };
        let bytes = rmp_serde::to_vec(&geo).expect("serialize");
        let restored: GeoValue = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(geo, restored);
    }

    #[test]
    fn nested_array() {
        let val = Value::Array(vec![
            Value::Array(vec![Value::Int(1), Value::Int(2)]),
            Value::String("flat".into()),
        ]);
        let bytes = rmp_serde::to_vec(&val).expect("serialize");
        let restored: Value = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(val, restored);
    }

    #[test]
    fn nested_map() {
        let mut inner = BTreeMap::new();
        inner.insert("nested".into(), Value::Bool(true));
        let mut outer = BTreeMap::new();
        outer.insert("child".into(), Value::Map(inner));
        outer.insert("name".into(), Value::String("test".into()));

        let val = Value::Map(outer);
        let bytes = rmp_serde::to_vec(&val).expect("serialize");
        let restored: Value = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(val, restored);
    }

    #[test]
    fn large_vector() {
        let vec = vec![0.5f32; 384]; // typical embedding size
        let val = Value::Vector(vec);
        let bytes = rmp_serde::to_vec(&val).expect("serialize");
        let restored: Value = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(val, restored);
    }

    #[test]
    fn empty_values() {
        let empties = vec![
            Value::String(String::new()),
            Value::Vector(vec![]),
            Value::Blob(vec![]),
            Value::Array(vec![]),
            Value::Map(BTreeMap::new()),
            Value::Binary(vec![]),
        ];
        for val in &empties {
            let bytes = rmp_serde::to_vec(val).expect("serialize");
            let restored: Value = rmp_serde::from_slice(&bytes).expect("deserialize");
            assert_eq!(val, &restored);
        }
    }

    #[test]
    fn document_type_name() {
        let doc = Value::Document(rmpv::Value::Map(vec![]));
        assert_eq!(doc.type_name(), "DOCUMENT");
    }

    #[test]
    fn document_null_is_null() {
        let doc = Value::Document(rmpv::Value::Nil);
        assert!(!doc.is_null()); // Document(Nil) is NOT Value::Null
    }

    #[test]
    fn document_accessor() {
        let inner = rmpv::Value::String("hello".into());
        let doc = Value::Document(inner.clone());
        assert_eq!(doc.as_document(), Some(&inner));
        assert_eq!(Value::Int(42).as_document(), None);
    }

    #[test]
    fn document_roundtrip_nested_5_levels() {
        // Build 5-level nested document: {a: {b: {c: {d: {e: "deep"}}}}}
        let level5 = rmpv::Value::Map(vec![(
            rmpv::Value::String("e".into()),
            rmpv::Value::String("deep".into()),
        )]);
        let level4 = rmpv::Value::Map(vec![(rmpv::Value::String("d".into()), level5)]);
        let level3 = rmpv::Value::Map(vec![(rmpv::Value::String("c".into()), level4)]);
        let level2 = rmpv::Value::Map(vec![(rmpv::Value::String("b".into()), level3)]);
        let level1 = rmpv::Value::Map(vec![(rmpv::Value::String("a".into()), level2)]);

        let val = Value::Document(level1);
        let bytes = rmp_serde::to_vec(&val).expect("serialize");
        let restored: Value = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(val, restored);
    }

    #[test]
    fn document_heterogeneous_array() {
        // Document can contain heterogeneous arrays (unlike Value::Array which is homogeneous)
        let doc = rmpv::Value::Map(vec![(
            rmpv::Value::String("mixed".into()),
            rmpv::Value::Array(vec![
                rmpv::Value::Integer(42.into()),
                rmpv::Value::String("hello".into()),
                rmpv::Value::Boolean(true),
                rmpv::Value::Nil,
                rmpv::Value::F64(9.81),
                rmpv::Value::Array(vec![rmpv::Value::Integer(1.into())]),
            ]),
        )]);
        let val = Value::Document(doc);
        let bytes = rmp_serde::to_vec(&val).expect("serialize");
        let restored: Value = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(val, restored);
    }

    #[test]
    fn document_empty_map() {
        let val = Value::Document(rmpv::Value::Map(vec![]));
        let bytes = rmp_serde::to_vec(&val).expect("serialize");
        let restored: Value = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(val, restored);
    }

    #[test]
    fn document_serialized_size() {
        let small = Value::Document(rmpv::Value::Map(vec![(
            rmpv::Value::String("key".into()),
            rmpv::Value::String("val".into()),
        )]));
        let size = small.document_serialized_size();
        assert!(size.is_some());
        assert!(size.expect("size") > 0);
        assert!(size.expect("size") < 100);

        // Non-document returns None
        assert!(Value::Int(42).document_serialized_size().is_none());
    }

    #[test]
    fn document_complex_realistic() {
        // Realistic IoT device config document
        let config = rmpv::Value::Map(vec![
            (
                rmpv::Value::String("device_id".into()),
                rmpv::Value::String("sensor-001".into()),
            ),
            (
                rmpv::Value::String("firmware".into()),
                rmpv::Value::String("2.1.3".into()),
            ),
            (
                rmpv::Value::String("enabled".into()),
                rmpv::Value::Boolean(true),
            ),
            (
                rmpv::Value::String("sample_rate_hz".into()),
                rmpv::Value::Integer(100.into()),
            ),
            (
                rmpv::Value::String("thresholds".into()),
                rmpv::Value::Map(vec![
                    (
                        rmpv::Value::String("temp_min".into()),
                        rmpv::Value::F64(-40.0),
                    ),
                    (
                        rmpv::Value::String("temp_max".into()),
                        rmpv::Value::F64(85.0),
                    ),
                    (
                        rmpv::Value::String("humidity_max".into()),
                        rmpv::Value::F64(95.0),
                    ),
                ]),
            ),
            (
                rmpv::Value::String("tags".into()),
                rmpv::Value::Array(vec![
                    rmpv::Value::String("production".into()),
                    rmpv::Value::String("outdoor".into()),
                ]),
            ),
            (
                rmpv::Value::String("calibration".into()),
                rmpv::Value::Map(vec![
                    (
                        rmpv::Value::String("last_date".into()),
                        rmpv::Value::String("2026-01-15".into()),
                    ),
                    (
                        rmpv::Value::String("offsets".into()),
                        rmpv::Value::Array(vec![
                            rmpv::Value::F64(0.01),
                            rmpv::Value::F64(-0.03),
                            rmpv::Value::F64(0.005),
                        ]),
                    ),
                ]),
            ),
        ]);
        let val = Value::Document(config);
        let bytes = rmp_serde::to_vec(&val).expect("serialize");
        let restored: Value = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(val, restored);
    }

    #[test]
    fn vector_consistency_mode_default_is_current() {
        assert_eq!(
            VectorConsistencyMode::default(),
            VectorConsistencyMode::Current
        );
    }

    #[test]
    fn vector_consistency_mode_from_str() {
        assert_eq!(
            VectorConsistencyMode::from_str_opt("current"),
            Some(VectorConsistencyMode::Current)
        );
        assert_eq!(
            VectorConsistencyMode::from_str_opt("SNAPSHOT"),
            Some(VectorConsistencyMode::Snapshot)
        );
        assert_eq!(
            VectorConsistencyMode::from_str_opt("Exact"),
            Some(VectorConsistencyMode::Exact)
        );
        assert_eq!(VectorConsistencyMode::from_str_opt("invalid"), None);
        assert_eq!(VectorConsistencyMode::from_str_opt(""), None);
    }

    #[test]
    fn vector_consistency_mode_as_str() {
        assert_eq!(VectorConsistencyMode::Current.as_str(), "current");
        assert_eq!(VectorConsistencyMode::Snapshot.as_str(), "snapshot");
        assert_eq!(VectorConsistencyMode::Exact.as_str(), "exact");
    }

    #[test]
    fn vector_mvcc_stats_default() {
        let stats = VectorMvccStats::default();
        assert_eq!(stats.candidates_fetched, 0);
        assert_eq!(stats.candidates_visible, 0);
        assert_eq!(stats.candidates_filtered, 0);
        assert_eq!(stats.expansion_rounds, 0);
        assert!((stats.overfetch_factor - 0.0).abs() < f64::EPSILON);
    }
}
