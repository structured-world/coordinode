//! Benchmark harness — R700.
//!
//! Standardised JSON schema for ALL CoordiNode benchmarks (vector,
//! graph, time-series, spatial, document, full-text). Per-modality
//! bench binaries under `benches/*` consume this crate to:
//!
//! 1. Stamp every result with **commit + hardware fingerprint**
//!    (for the gh-pages dynamics chart that plots metrics over
//!    commits).
//! 2. Emit a **single JSON file** per run, written to
//!    `bench-results/<modality>/<dataset>/<sha>-<date>.json`.
//! 3. Optionally write a **TSV summary line** for human-readable
//!    `git log --format` post-processing.
//!
//! The JSON shape is the contract between the bench runner (this
//! crate) and the docs/bench reporting layer (Vega-Lite charts).
//! Adding a new metric is additive — existing renderers ignore
//! unknown fields. Breaking changes bump `schema_version`.
//!
//! **no-std tier: std-only.** This crate runs on bench hosts that
//! have the full host OS (sysinfo, env vars, filesystem); no
//! no-std readiness intended.

#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub mod error;
pub mod fingerprint;

pub use error::{BenchError, BenchResult as BenchOpResult};
pub use fingerprint::{git_metadata, hardware_fingerprint, GitMetadata, HardwareFingerprint};

/// Current schema version. Bump on breaking changes (renamed /
/// removed fields). Renderers that see a higher schema_version than
/// they understand should refuse to plot the result.
pub const SCHEMA_VERSION: u32 = 1;

/// Top-level benchmark report. One per run; serialised to
/// `bench-results/<modality>/<dataset>/<sha>-<date>.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    /// Schema version — bump on breaking changes.
    pub schema_version: u32,
    /// Wall-clock timestamp of the run (UTC).
    pub timestamp: DateTime<Utc>,
    /// Git metadata (commit SHA, branch, dirty flag).
    pub git: GitMetadata,
    /// Host hardware fingerprint.
    pub hardware: HardwareFingerprint,
    /// Benchmark category (e.g. "vector", "graph", "spatial").
    pub modality: String,
    /// Specific benchmark identifier (e.g. "ann-benchmarks-sift1m",
    /// "ldbc-snb-sf1-interactive").
    pub benchmark: String,
    /// Dataset name (e.g. "sift-128-euclidean", "snb-sf1").
    pub dataset: String,
    /// Subject under test — `coordinode` for our own runs,
    /// `hnswlib` / `faiss-cpu` / `mongodb-8` / ... for competitors.
    pub subject: String,
    /// Codec mode — `zstd`, `none`, `lz4`, etc. Per methodology
    /// every bench runs at least two codec modes.
    pub codec: String,
    /// CoordiNode version string (or competitor version for
    /// competitor runs) — `0.4.3`, `MongoDB 8.0.4`, `hnswlib 0.8.0`.
    pub version: String,
    /// Per-metric measurements. Keys are metric IDs the renderer
    /// knows about; unknown keys are passed through silently.
    pub metrics: serde_json::Map<String, serde_json::Value>,
    /// Free-form notes for the report — e.g. anomalies, manual
    /// observations, config overrides applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl BenchReport {
    /// Build a fresh report stamped with the current git + hardware
    /// fingerprint. The caller fills in `metrics` before
    /// [`Self::write_json`].
    pub fn new(
        modality: impl Into<String>,
        benchmark: impl Into<String>,
        dataset: impl Into<String>,
        subject: impl Into<String>,
        codec: impl Into<String>,
        version: impl Into<String>,
    ) -> BenchOpResult<Self> {
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            timestamp: Utc::now(),
            git: git_metadata()?,
            hardware: hardware_fingerprint(),
            modality: modality.into(),
            benchmark: benchmark.into(),
            dataset: dataset.into(),
            subject: subject.into(),
            codec: codec.into(),
            version: version.into(),
            metrics: serde_json::Map::new(),
            notes: None,
        })
    }

    /// Record a single named metric. Overwrites if the key already
    /// exists.
    pub fn record<V: Serialize>(&mut self, key: impl Into<String>, value: V) -> BenchOpResult<()> {
        let v = serde_json::to_value(value)?;
        self.metrics.insert(key.into(), v);
        Ok(())
    }

    /// Serialise to JSON file at the canonical path:
    /// `<base>/<modality>/<dataset>/<sha>-<subject>[-<tag>]-<YYYYmmdd-HHMMSS>.json`.
    /// Creates parent directories as needed.
    ///
    /// `tag` distinguishes configurations of the same engine on the
    /// same commit (e.g. `"M16"`, `"M24"`, `"M32"` for an HNSW M
    /// sweep). When `None`, the filename matches the original
    /// `<sha>-<subject>-<stamp>.json` shape.
    pub fn write_json(
        &self,
        base: impl AsRef<Path>,
        tag: Option<&str>,
    ) -> BenchOpResult<std::path::PathBuf> {
        let dir = base.as_ref().join(&self.modality).join(&self.dataset);
        std::fs::create_dir_all(&dir)?;
        let stamp = self.timestamp.format("%Y%m%d-%H%M%S");
        let file_name = match tag {
            Some(t) if !t.is_empty() => {
                format!(
                    "{}-{}-{}-{}.json",
                    &self.git.sha_short, &self.subject, t, stamp
                )
            }
            _ => format!("{}-{}-{}.json", &self.git.sha_short, &self.subject, stamp),
        };
        let path = dir.join(file_name);
        let bytes = serde_json::to_vec_pretty(self)?;
        std::fs::write(&path, bytes)?;
        Ok(path)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn record_and_serialise_round_trips() {
        // Use a synthetic report (don't go through BenchReport::new
        // because it touches git which may not be available in
        // sandbox tests).
        let report = BenchReport {
            schema_version: SCHEMA_VERSION,
            timestamp: Utc::now(),
            git: GitMetadata {
                sha: "abc123def4567890".into(),
                sha_short: "abc123d".into(),
                branch: "main".into(),
                dirty: false,
                commit_date: Utc::now(),
            },
            hardware: HardwareFingerprint {
                cpu_brand: "Test CPU".into(),
                cpu_cores: 8,
                cpu_threads: 16,
                ram_gb: 32,
                os_name: "test-os".into(),
                os_version: "1.0".into(),
                arch: "x86_64".into(),
            },
            modality: "vector".into(),
            benchmark: "test".into(),
            dataset: "synth".into(),
            subject: "coordinode".into(),
            codec: "none".into(),
            version: "0.0.0".into(),
            metrics: serde_json::Map::new(),
            notes: None,
        };
        let json = serde_json::to_string(&report).expect("serialise");
        let back: BenchReport = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(back.schema_version, SCHEMA_VERSION);
        assert_eq!(back.modality, "vector");
    }

    #[test]
    fn record_metric_inserts_into_map() {
        let mut report = BenchReport {
            schema_version: SCHEMA_VERSION,
            timestamp: Utc::now(),
            git: GitMetadata {
                sha: "abc".into(),
                sha_short: "abc".into(),
                branch: "main".into(),
                dirty: false,
                commit_date: Utc::now(),
            },
            hardware: HardwareFingerprint {
                cpu_brand: String::new(),
                cpu_cores: 0,
                cpu_threads: 0,
                ram_gb: 0,
                os_name: String::new(),
                os_version: String::new(),
                arch: String::new(),
            },
            modality: "vector".into(),
            benchmark: "test".into(),
            dataset: "synth".into(),
            subject: "coordinode".into(),
            codec: "none".into(),
            version: "0.0.0".into(),
            metrics: serde_json::Map::new(),
            notes: None,
        };
        report.record("recall_at_10", 0.95_f64).expect("record");
        report.record("qps", 1234.5_f64).expect("record");
        assert_eq!(report.metrics.len(), 2);
        assert_eq!(report.metrics["recall_at_10"].as_f64(), Some(0.95));
        assert_eq!(report.metrics["qps"].as_f64(), Some(1234.5));
    }

    #[test]
    fn write_json_creates_directories_and_file() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let mut report = BenchReport::new(
            "vector",
            "test-bench",
            "synth-ds",
            "coordinode",
            "none",
            "0.0.0",
        )
        .expect("build");
        report.record("metric_a", 1.0_f64).expect("record");
        let path = report.write_json(tmp.path(), None).expect("write");
        assert!(path.exists(), "JSON file must exist");
        // Round-trip verify file
        let bytes = std::fs::read(&path).expect("read");
        let back: BenchReport = serde_json::from_slice(&bytes).expect("decode");
        assert_eq!(back.metrics["metric_a"].as_f64(), Some(1.0));
    }

    #[test]
    fn write_json_tag_appears_in_filename() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let report = BenchReport::new(
            "vector",
            "ann-benchmarks",
            "sift-128-euclidean",
            "coordinode",
            "none",
            "0.0.0",
        )
        .expect("build");
        let with_tag = report.write_json(tmp.path(), Some("M24")).expect("write");
        let bare = report.write_json(tmp.path(), None).expect("write");
        let with_name = with_tag.file_name().unwrap().to_string_lossy().to_string();
        let bare_name = bare.file_name().unwrap().to_string_lossy().to_string();
        assert!(
            with_name.contains("-coordinode-M24-"),
            "tag must appear between subject and timestamp: {with_name}"
        );
        assert!(
            !bare_name.contains("-M24-"),
            "untagged file must not carry M segment: {bare_name}"
        );
    }
}
