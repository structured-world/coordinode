//! Background integrity verification (scrubbing).
//!
//! A scrub walks every block of every on-disk SST in each partition and verifies
//! its xxh3 checksum against the value stored in the block header, surfacing
//! silent bit rot ahead of a live read. It delegates to the storage engine's
//! block-checksum scrubber ([`lsm_tree::AbstractTree::verify_checksum_with`]),
//! which scans to completion and collects every corrupt block with its exact
//! `(file, offset)` rather than aborting on the first failure — so one bad block
//! never hides the rest. An optional inter-SST throttle keeps a background scrub
//! from saturating production I/O.
//!
//! ## Usage
//!
//! ```ignore
//! let report = scrub_all(&engine, &ScrubConfig::default())?;
//! if report.has_errors() {
//!     eprintln!("corruption detected: {:?}", report.errors);
//! }
//! ```

use std::time::{Duration, Instant};

use lsm_tree::verify::VerifyOptions;
use lsm_tree::AbstractTree;

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::StorageResult;

/// Configuration for background scrub.
#[derive(Debug, Clone)]
pub struct ScrubConfig {
    /// Whether background scrubbing is enabled. Default: `true`.
    pub enabled: bool,

    /// Interval between full scrub cycles. Default: 7 days.
    pub interval: Duration,

    /// Minimum pause between consecutive SST scans, capping the I/O pressure a
    /// scrub puts on a production node. `None` (default) runs at full speed; a
    /// background scrub should set a small delay so it yields to live traffic.
    pub throttle: Option<Duration>,

    /// Number of SSTs to scan concurrently per partition. Default: 1 (sequential,
    /// no thread spawn). Higher values trade I/O pressure for wall-clock.
    pub parallelism: usize,
}

impl Default for ScrubConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(7 * 24 * 3600),
            throttle: None,
            parallelism: 1,
        }
    }
}

impl ScrubConfig {
    /// Map this config onto the storage engine's block-scrub options.
    fn verify_options(&self) -> VerifyOptions {
        let opts = VerifyOptions::default().parallelism(self.parallelism.max(1));
        match self.throttle {
            Some(delay) => opts.throttle(delay),
            None => opts,
        }
    }
}

/// A verification error found during scrubbing: one corrupt (or unreadable)
/// block in one partition, with the engine's `(file, offset)` detail.
#[derive(Debug, Clone)]
pub struct VerifyError {
    /// The partition the corrupt block belongs to.
    pub partition: Partition,

    /// Human-readable description from the block scrubber (table id, file path,
    /// offset, expected vs. got checksum).
    pub message: String,
}

/// Report from a scrub operation.
#[derive(Debug, Clone)]
pub struct ScrubReport {
    /// Total blocks header-read across all partitions (includes blocks whose
    /// data checksum then failed).
    pub blocks_checked: u64,

    /// Total SST files visited across all partitions.
    pub sst_files_checked: u64,

    /// Per-partition block counts.
    pub partition_counts: Vec<(Partition, u64)>,

    /// Errors found during scrub. Empty means every block verified clean.
    pub errors: Vec<VerifyError>,

    /// Wall-clock time taken for the scrub.
    pub duration: Duration,
}

impl ScrubReport {
    /// Whether any corruption was found.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

/// Scrub a single partition: verify every on-disk block's checksum.
///
/// Returns `(blocks_checked, sst_files_checked, errors)`. The scan always runs
/// to completion — a corrupt block is collected into `errors`, never propagated
/// as an early return — so a single bad block does not mask the rest of the
/// partition. The only `Err` here is failing to resolve the partition's tree
/// handle.
///
/// # Errors
/// [`StorageError`](crate::error::StorageError) if the partition's tree handle
/// cannot be resolved.
pub fn scrub_partition(
    engine: &StorageEngine,
    part: Partition,
    config: &ScrubConfig,
) -> StorageResult<(u64, u64, Vec<VerifyError>)> {
    let report = engine
        .tree(part)?
        .verify_checksum_with(&config.verify_options());

    let errors = report
        .errors
        .iter()
        .map(|e| VerifyError {
            partition: part,
            message: e.to_string(),
        })
        .collect();

    Ok((
        report.blocks_scanned as u64,
        report.sst_files_scanned as u64,
        errors,
    ))
}

/// Scrub all partitions, returning an aggregate report.
///
/// Verifies every on-disk block across all partitions; the equivalent of the
/// read-side integrity pass behind `coordinode verify --deep`.
///
/// # Errors
/// [`StorageError`](crate::error::StorageError) if a partition's tree handle
/// cannot be resolved.
pub fn scrub_all(engine: &StorageEngine, config: &ScrubConfig) -> StorageResult<ScrubReport> {
    let start = Instant::now();
    let mut blocks_checked: u64 = 0;
    let mut sst_files_checked: u64 = 0;
    let mut all_errors = Vec::new();
    let mut partition_counts = Vec::with_capacity(Partition::all().len());

    for &part in Partition::all() {
        let (blocks, ssts, mut errors) = scrub_partition(engine, part, config)?;
        blocks_checked += blocks;
        sst_files_checked += ssts;
        partition_counts.push((part, blocks));
        all_errors.append(&mut errors);
    }

    Ok(ScrubReport {
        blocks_checked,
        sst_files_checked,
        partition_counts,
        errors: all_errors,
        duration: start.elapsed(),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
