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
    pub kv: Kv,
    #[serde(default)]
    pub vector: Vector,
    #[serde(default)]
    pub graph: Graph,
    #[serde(default)]
    pub text: Text,
    #[serde(default)]
    pub timeseries: TimeSeries,
}

#[derive(Debug, Default, Deserialize)]
pub struct Kv {
    pub ycsb: KvYcsb,
}

#[derive(Debug, Default, Deserialize)]
pub struct KvYcsb {
    pub workload_a: BTreeMap<String, KvYcsbEntry>,
    pub workload_c: BTreeMap<String, KvYcsbEntry>,
}

#[derive(Debug, Deserialize)]
pub struct KvYcsbEntry {
    pub throughput_ops_s: f64,
    pub read_p99_us: f64,
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
        // Spot-check one known entry: YCSB workload A vs Redis.
        let redis =
            b.kv.ycsb
                .workload_a
                .get("redis")
                .expect("redis baseline must exist for ycsb workload A");
        assert!(redis.throughput_ops_s > 0.0);
        assert!(redis.read_p99_us > 0.0);
        assert!(!redis.source.is_empty());
    }
}
