//! Background integrity verification (scrubbing).
//!
//! A scrub walks every block of every on-disk SST in each partition and verifies
//! its xxh3 checksum against the value stored in the block header, surfacing
//! silent bit rot ahead of a live read. It delegates to the storage engine's
//! block-checksum scrubber ([`lsm_tree::AbstractTree::verify_checksum_with`]),
//! which scans to completion and collects every corrupt block with its exact
//! `(file, offset)` rather than aborting on the first failure — so one bad block
//! never hides the rest. Its reads are charged to the engine's maintenance
//! budget with compaction's, and an optional inter-SST pause keeps a
//! background scrub from saturating production I/O.
//!
//! ## Usage
//!
//! ```ignore
//! let report = scrub_all(&engine, &ScrubConfig::default())?;
//! if report.has_errors() {
//!     eprintln!("corruption detected: {:?}", report.errors);
//! }
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use lsm_tree::AbstractTree;
use lsm_tree::rate_limiter::RateLimiter;
use lsm_tree::verify::VerifyOptions;

use crate::engine::core::{Deployment, StorageEngine};
use crate::engine::partition::Partition;
use crate::engine::scope::TreeId;
use crate::error::StorageResult;

/// Configuration for background scrub.
#[derive(Debug, Clone)]
pub struct ScrubConfig {
    /// Whether background scrubbing is enabled. Default: `false`.
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
            enabled: false,
            interval: Duration::from_secs(7 * 24 * 3600),
            throttle: None,
            parallelism: 1,
        }
    }
}

impl ScrubConfig {
    /// Map this config onto the storage engine's block-scrub options, the
    /// reads charged to the engine's maintenance `budget` with compaction's:
    /// a device the scrub finds slow backs both off.
    fn verify_options(&self, budget: &Arc<RateLimiter>) -> VerifyOptions {
        let opts = VerifyOptions::default()
            .parallelism(self.parallelism.max(1))
            .rate_limiter(Arc::clone(budget));
        match self.throttle {
            Some(delay) => opts.throttle(delay),
            None => opts,
        }
    }
}

/// A verification error found during scrubbing: one corrupt (or unreadable)
/// block in one tree, with the engine's `(file, offset)` detail.
#[derive(Debug, Clone)]
pub struct VerifyError {
    /// The tree the corrupt block belongs to: a partition's, of the system
    /// set or of a namespace's.
    pub tree: TreeId,

    /// Human-readable description from the block scrubber (table id, file path,
    /// offset, expected vs. got checksum).
    pub message: String,
}

/// Report from a scrub operation.
#[derive(Debug, Clone)]
pub struct ScrubReport {
    /// Total blocks header-read across all trees (includes blocks whose
    /// data checksum then failed).
    pub blocks_checked: u64,

    /// Total SST files visited across all trees.
    pub sst_files_checked: u64,

    /// Per-tree block counts, of every tree open when the scrub ran.
    pub tree_counts: Vec<(TreeId, u64)>,

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
    Ok(scrub_tree(
        engine,
        engine.tree_id(part),
        engine.tree(part)?,
        config,
    ))
}

/// Verify every on-disk block of `tree`, tree `id` of `deployment`.
fn scrub_tree(
    deployment: &Deployment,
    id: TreeId,
    tree: &lsm_tree::AnyTree,
    config: &ScrubConfig,
) -> (u64, u64, Vec<VerifyError>) {
    let report = tree.verify_checksum_with(&config.verify_options(deployment.maintenance_budget()));

    let mut errors: Vec<VerifyError> = report
        .errors
        .iter()
        .map(|e| VerifyError {
            tree: id,
            message: e.to_string(),
        })
        .collect();
    // The block scrub reads tables only. A partition keeping its values in
    // blob files has each of them checked against the digest its manifest
    // records; its tables hold only keys and references, so hashing them
    // again here costs little.
    if crate::engine::config::separates_values(id.part) {
        errors.extend(
            lsm_tree::verify::verify_integrity(tree)
                .errors
                .iter()
                .map(|e| VerifyError {
                    tree: id,
                    message: e.to_string(),
                }),
        );
    }

    (
        report.blocks_scanned as u64,
        report.sst_files_scanned as u64,
        errors,
    )
}

/// Scrub every tree of the deployment, the system set's and each open
/// namespace's, returning an aggregate report.
///
/// Verifies every on-disk block across all trees; the equivalent of the
/// read-side integrity pass behind `coordinode verify --deep`.
///
/// # Errors
/// Never in practice: every tree listed is open. The `Result` is kept for a
/// tree that cannot be resolved.
pub fn scrub_all(deployment: &Deployment, config: &ScrubConfig) -> StorageResult<ScrubReport> {
    scrub_all_except(deployment, config, |_| false)
}

/// [`scrub_all`] leaving out each tree `verified` names: one whose blocks
/// were verified at the version it still has, so a second pass would read
/// the same blocks for the same answer. A left-out tree counts no blocks.
///
/// # Errors
/// As [`scrub_all`].
pub fn scrub_all_except(
    deployment: &Deployment,
    config: &ScrubConfig,
    verified: impl Fn(TreeId) -> bool,
) -> StorageResult<ScrubReport> {
    let start = Instant::now();
    let mut blocks_checked: u64 = 0;
    let mut sst_files_checked: u64 = 0;
    let mut all_errors = Vec::new();
    // A snapshot of the open trees: a namespace dropped meanwhile keeps its
    // trees open only as long as this list.
    let trees = deployment.coordinator().trees();
    let mut tree_counts = Vec::with_capacity(trees.len());

    for (id, tree) in &trees {
        if verified(*id) {
            tree_counts.push((*id, 0));
            continue;
        }
        let (blocks, ssts, mut errors) = scrub_tree(deployment, *id, tree, config);
        blocks_checked += blocks;
        sst_files_checked += ssts;
        tree_counts.push((*id, blocks));
        all_errors.append(&mut errors);
    }

    Ok(ScrubReport {
        blocks_checked,
        sst_files_checked,
        tree_counts,
        errors: all_errors,
        duration: start.elapsed(),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
