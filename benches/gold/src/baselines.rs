//! Loader for `baselines.toml`. The TOML file lives next to the
//! crate root; production loads via `include_str!` so the binaries
//! ship the table without a runtime file read.

use serde::Deserialize;
use std::collections::BTreeMap;

/// Embedded contents of `baselines.toml`. Compiled in at build time
/// so the gold binaries don't depend on a runtime working directory.
const EMBEDDED_BASELINES: &str = include_str!("../baselines.toml");

#[derive(Debug, Deserialize)]
pub struct Baselines {
    pub document: Document,
    #[serde(default)]
    pub vector: Vector,
    #[serde(default)]
    pub graph: Graph,
    #[serde(default)]
    pub text: Text,
    #[serde(default)]
    pub timeseries: TimeSeries,
    #[serde(default)]
    pub spatial: Option<toml::Value>,
    #[serde(default)]
    pub holistic: Option<toml::Value>,
}

/// Document-store workload baselines — YCSB A/B/C/F against
/// multi-model document stores (MongoDB, SurrealDB, ArangoDB).
/// CoordiNode positions itself against this tier, not against
/// in-memory caches (Redis) or KV-only stores.
#[derive(Debug, Default, Deserialize)]
pub struct Document {
    pub ycsb: DocumentYcsb,
}

#[derive(Debug, Default, Deserialize)]
pub struct DocumentYcsb {
    pub workload_a: BTreeMap<String, DocumentYcsbEntry>,
    pub workload_c: BTreeMap<String, DocumentYcsbEntry>,
}

/// One competitor row in a YCSB-A or YCSB-C table.
///
/// Per the methodology principle #7, every competitor entry is
/// **codec-scoped**: the same engine appears once per compression
/// mode (`<engine>_<codec>`, e.g. `mongodb_8_snappy` +
/// `mongodb_8_zstd` + `mongodb_8_none`). Reports render the default
/// codec and the disabled mode side-by-side so reviewers can tell
/// whether a gap is engine perf or codec choice.
///
/// `throughput_ops_s` and `read_p99_us` are optional because several
/// multi-model competitors don't publish directly-comparable YCSB
/// numbers per codec yet — we omit the value and fill in once we
/// run their engine against the same dataset in our harness.
#[derive(Debug, Deserialize)]
pub struct DocumentYcsbEntry {
    /// Codec name as the upstream engine identifies it: e.g.
    /// `"snappy"`, `"zstd"`, `"lz4"`, `"none"`, `"pglz"`.
    pub codec: String,
    /// Categorical tag for report-layer rendering. Expected values:
    /// `"default"` (the codec the engine ships with), `"alternate"`
    /// (another supported codec), `"disabled"` (compression off —
    /// the apples-to-apples CPU baseline).
    pub codec_role: String,
    #[serde(default)]
    pub throughput_ops_s: Option<f64>,
    #[serde(default)]
    pub read_p99_us: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
    pub source: String,
}

#[derive(Debug, Default, Deserialize)]
pub struct Vector {
    #[serde(default)]
    pub ann_benchmarks: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Graph {
    #[serde(default)]
    pub ldbc_snb_interactive_v2: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Text {
    #[serde(default)]
    pub search_benchmark_game: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Default, Deserialize)]
pub struct TimeSeries {
    #[serde(default)]
    pub tsbs_devops: BTreeMap<String, toml::Value>,
}

impl Baselines {
    /// Parse the embedded `baselines.toml`. Panics on a malformed
    /// table — this is build-time-validated, not runtime input.
    pub fn embedded() -> Self {
        toml::from_str(EMBEDDED_BASELINES).expect("embedded baselines.toml is malformed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_baselines_parse_cleanly() {
        // Sanity check the TOML at build time — any future edit to
        // baselines.toml that breaks the schema fails this test
        // before the binaries try to use the table at runtime.
        let b = Baselines::embedded();
        // Spot-check the primary document-store competitor at the
        // chosen comparison codec: MongoDB 8.x with zstd. Per
        // methodology §"Codec choice" snappy is excluded — we
        // standardize on zstd × zstd vs none × none across all
        // multi-model comparisons.
        let mongo = b
            .document
            .ycsb
            .workload_a
            .get("mongodb_8_zstd")
            .expect("mongodb_8_zstd baseline must exist for ycsb workload A");
        assert_eq!(mongo.codec, "zstd");
        assert_eq!(mongo.codec_role, "default");
        assert!(!mongo.source.is_empty());
    }

    #[test]
    fn document_baselines_use_only_zstd_or_none() {
        // Methodology §"Codec choice" — snappy / lz4 / pglz are
        // EXCLUDED. Every entry in document.ycsb.workload_a /
        // workload_c must be codec = "zstd" or codec = "none".
        // A future edit re-adding snappy fails this test.
        let b = Baselines::embedded();
        let all = b
            .document
            .ycsb
            .workload_a
            .values()
            .chain(b.document.ycsb.workload_c.values());
        for entry in all {
            assert!(
                entry.codec == "zstd" || entry.codec == "none",
                "document baseline with codec `{}` (source `{}`) violates §Codec choice — \
                 only zstd and none are allowed in document YCSB comparisons",
                entry.codec,
                entry.source,
            );
        }
    }

    #[test]
    fn document_workload_c_lists_only_multi_model_competitors() {
        // Workload C entries must be from the multi-model tier — no
        // Redis, no ScyllaDB, no SQLite. Keys are codec-scoped
        // (`<engine>_<codec>`); a future edit re-adding Redis fails
        // here regardless of codec suffix.
        let b = Baselines::embedded();
        for name in b.document.ycsb.workload_c.keys() {
            let engine_root = strip_codec_suffix(name);
            let allowed = matches!(engine_root, "mongodb_8" | "surrealdb_3_0" | "arangodb_3_12",);
            assert!(
                allowed,
                "non-multi-model competitor `{name}` (root `{engine_root}`) in \
                 YCSB workload_c — only mongodb_8 / surrealdb_3_0 / arangodb_3_12 \
                 (per any codec) belong here per arch/benchmarks/methodology.md",
            );
        }
    }

    #[test]
    fn document_workload_a_covers_zstd_and_none_per_engine() {
        // Methodology principle #7 + §"Codec choice" — every primary
        // multi-model competitor must appear exactly twice per
        // workload: `<engine>_zstd` AND `<engine>_none`. The report
        // renders zstd-vs-zstd and none-vs-none columns side by side.
        let b = Baselines::embedded();
        let keys: std::collections::HashSet<&str> = b
            .document
            .ycsb
            .workload_a
            .keys()
            .map(String::as_str)
            .collect();
        for engine in ["mongodb_8", "surrealdb_3_0", "arangodb_3_12"] {
            let zstd_key = format!("{engine}_zstd");
            let none_key = format!("{engine}_none");
            assert!(
                keys.contains(zstd_key.as_str()),
                "missing zstd baseline `{zstd_key}` in document.ycsb.workload_a — \
                 every primary competitor MUST have a codec=zstd entry",
            );
            assert!(
                keys.contains(none_key.as_str()),
                "missing uncompressed baseline `{none_key}` in document.ycsb.workload_a — \
                 every primary competitor MUST have a codec=none entry",
            );
        }
    }

    /// Strip a `_<codec>` suffix to recover the engine identifier.
    /// `mongodb_8_snappy` → `mongodb_8`; `arangodb_3_12_lz4` →
    /// `arangodb_3_12`. The codec is always the last underscore-
    /// separated token in the entry name.
    fn strip_codec_suffix(name: &str) -> &str {
        match name.rsplit_once('_') {
            Some((root, _codec)) => root,
            None => name,
        }
    }
}
