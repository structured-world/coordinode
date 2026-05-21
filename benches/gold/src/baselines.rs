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
/// `throughput_ops_s` and `read_p99_us` are optional because some
/// multi-model competitors (SurrealDB 3.x, ArangoDB) don't publish
/// directly-comparable YCSB numbers — we mark those `null` in the
/// TOML and fill in once we run their engine against the same
/// dataset in our harness.
#[derive(Debug, Deserialize)]
pub struct DocumentYcsbEntry {
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
        // Spot-check the primary document-store competitor: MongoDB 8.x
        // on YCSB workload A. Per the rewritten methodology doc, this
        // replaces the prior Redis comparison (Redis is a cache, not a
        // multi-model engine — wrong competitor entirely).
        let mongo = b
            .document
            .ycsb
            .workload_a
            .get("mongodb_8")
            .expect("mongodb_8 baseline must exist for ycsb workload A");
        assert!(mongo.throughput_ops_s.unwrap_or(0.0) > 0.0);
        assert!(mongo.read_p99_us.unwrap_or(0.0) > 0.0);
        assert!(!mongo.source.is_empty());
    }

    #[test]
    fn document_workload_c_lists_only_multi_model_competitors() {
        // Workload C entries must be from the multi-model tier — no
        // Redis, no ScyllaDB, no SQLite. This pins the methodology
        // decision in code: a future edit re-adding Redis fails here.
        let b = Baselines::embedded();
        for name in b.document.ycsb.workload_c.keys() {
            let allowed = matches!(
                name.as_str(),
                "mongodb_8" | "surrealdb_3_0" | "arangodb_3_12",
            );
            assert!(
                allowed,
                "non-multi-model competitor `{name}` in YCSB workload_c — \
                 only mongodb_8 / surrealdb_3_0 / arangodb_3_12 belong here per \
                 arch/benchmarks/methodology.md",
            );
        }
    }
}
