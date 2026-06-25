//! Git + hardware fingerprint capture — stamps every BenchReport
//! so the gh-pages dynamics chart can correlate a measurement
//! with the commit + host that produced it.

use std::process::Command;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sysinfo::System;

use crate::error::{BenchError, BenchResult};

/// Git commit metadata captured at bench run time. All fields
/// are extracted from `git` CLI; on a non-git host the run aborts
/// with [`BenchError::Git`] (we never silently produce a report
/// without provenance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitMetadata {
    /// Full 40-character SHA.
    pub sha: String,
    /// Short SHA (7 chars) — used in filename + chart x-axis.
    pub sha_short: String,
    /// Branch name (`main`, `feat/...`).
    pub branch: String,
    /// True if working tree has uncommitted changes. Dirty
    /// measurements are still recorded but flagged in the chart.
    pub dirty: bool,
    /// Author commit date of the SHA (NOT the bench run time).
    pub commit_date: DateTime<Utc>,
}

/// Hardware fingerprint of the host running the bench. Populated
/// via `sysinfo`. Goes into the report so cross-host comparisons
/// are explicit — a result from `runner-1` vs `runner-2` is
/// labelled clearly in the chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareFingerprint {
    /// CPU brand string (e.g. "AMD Ryzen 9 7950X3D 16-Core").
    pub cpu_brand: String,
    /// Physical cores.
    pub cpu_cores: usize,
    /// Logical threads (cores × SMT).
    pub cpu_threads: usize,
    /// Total RAM in GiB (rounded down).
    pub ram_gb: u64,
    /// OS name (Linux / Darwin).
    pub os_name: String,
    /// OS version.
    pub os_version: String,
    /// Architecture (`x86_64`, `aarch64`).
    pub arch: String,
}

/// Capture git metadata for the current working tree. Returns
/// [`BenchError::Git`] if `git` is not on PATH or the directory
/// is not a repo.
pub fn git_metadata() -> BenchResult<GitMetadata> {
    let sha = run_git(&["rev-parse", "HEAD"])?;
    let sha_short = sha.chars().take(7).collect();
    let branch = run_git(&["rev-parse", "--abbrev-ref", "HEAD"])?;
    let dirty_output = run_git(&["status", "--porcelain"])?;
    let dirty = !dirty_output.is_empty();
    let commit_date_raw = run_git(&["log", "-1", "--format=%cI"])?;
    let commit_date = DateTime::parse_from_rfc3339(&commit_date_raw)
        .map_err(|e| BenchError::Git(format!("parse commit date: {e}")))?
        .with_timezone(&Utc);
    Ok(GitMetadata {
        sha,
        sha_short,
        branch,
        dirty,
        commit_date,
    })
}

/// Capture hardware fingerprint via `sysinfo`. Best-effort —
/// fields default to empty strings / zeros on platforms where
/// sysinfo can't probe.
#[must_use]
pub fn hardware_fingerprint() -> HardwareFingerprint {
    let mut sys = System::new_all();
    sys.refresh_all();
    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_default();
    let cpu_threads = sys.cpus().len();
    // sysinfo 0.39: physical_core_count is an associated function (no
    // instance needed); fall back to thread count when unavailable.
    let cpu_cores = System::physical_core_count().unwrap_or(cpu_threads);
    let ram_gb = sys.total_memory() / 1024 / 1024 / 1024;
    let os_name = System::name().unwrap_or_default();
    let os_version = System::os_version().unwrap_or_default();
    let arch = std::env::consts::ARCH.to_string();
    HardwareFingerprint {
        cpu_brand,
        cpu_cores,
        cpu_threads,
        ram_gb,
        os_name,
        os_version,
        arch,
    }
}

/// Invoke `git` with given args, trimming the stdout. Returns
/// [`BenchError::Git`] on non-zero exit.
fn run_git(args: &[&str]) -> BenchResult<String> {
    let out = Command::new("git").args(args).output()?;
    if !out.status.success() {
        return Err(BenchError::Git(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
