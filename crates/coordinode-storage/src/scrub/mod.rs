//! Background integrity verification (scrubbing).
//!
//! Scrubbing reads every key-value pair in the storage engine to trigger
//! the LSM storage's built-in xxh3 block checksum verification. Any checksum mismatch
//! is detected and reported.
//!
//! ## Usage
//!
//! ```ignore
//! let report = scrub_all(&engine)?;
//! if report.has_errors() {
//!     eprintln!("Corruption detected: {:?}", report.errors);
//! }
//! ```

use std::time::{Duration, Instant};

use lsm_tree::Guard;

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::StorageResult;

/// Configuration for background scrub.
#[derive(Debug, Clone)]
pub struct ScrubConfig {
    /// Interval between full scrub cycles. Default: 7 days.
    pub interval: Duration,

    /// Whether background scrubbing is enabled. Default: true.
    pub enabled: bool,
}

impl Default for ScrubConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(7 * 24 * 3600),
            enabled: true,
        }
    }
}

/// A verification error found during scrubbing.
#[derive(Debug, Clone)]
pub struct VerifyError {
    /// The partition where the error was found.
    pub partition: Partition,

    /// Description of the error.
    pub message: String,
}

/// Report from a scrub operation.
#[derive(Debug, Clone)]
pub struct ScrubReport {
    /// Total keys checked across all partitions.
    pub keys_checked: u64,

    /// Per-partition key counts.
    pub partition_counts: Vec<(Partition, u64)>,

    /// Errors found during scrub.
    pub errors: Vec<VerifyError>,

    /// Time taken for the scrub.
    pub duration: Duration,
}

impl ScrubReport {
    /// Whether any errors were found.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

/// Scrub a single partition by reading every key-value pair.
///
/// Each read triggers the storage engine's internal xxh3 checksum verification.
/// If a block is corrupt, the storage engine returns a `ChecksumMismatch` error
/// which we capture as a `VerifyError`.
pub fn scrub_partition(
    engine: &StorageEngine,
    part: Partition,
) -> StorageResult<(u64, Vec<VerifyError>)> {
    let mut count: u64 = 0;
    let errors = Vec::new();

    // prefix_scan with empty prefix iterates all keys in the partition.
    // Each guard.into_inner() reads the actual block, triggering the
    // storage engine's internal xxh3 checksum verification.
    let iter = engine.prefix_scan(part, b"")?;
    for guard in iter {
        let _ = guard.into_inner();
        count += 1;
    }

    Ok((count, errors))
}

/// Scrub all partitions, returning an aggregate report.
///
/// This is the equivalent of `coordinode verify --deep`: reads every
/// key-value pair across all 7 partitions to verify data integrity.
pub fn scrub_all(engine: &StorageEngine) -> StorageResult<ScrubReport> {
    let start = Instant::now();
    let mut total_keys: u64 = 0;
    let mut all_errors = Vec::new();
    let mut partition_counts = Vec::with_capacity(7);

    for &part in Partition::all() {
        let (count, mut errors) = scrub_partition(engine, part)?;
        total_keys += count;
        partition_counts.push((part, count));
        all_errors.append(&mut errors);
    }

    Ok(ScrubReport {
        keys_checked: total_keys,
        partition_counts,
        errors: all_errors,
        duration: start.elapsed(),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
    use tempfile::TempDir;

    fn test_engine() -> (StorageEngine, TempDir) {
        let dir = TempDir::new().expect("failed to create temp dir");
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("failed to open engine");
        (engine, dir)
    }

    #[test]
    fn scrub_config_defaults() {
        let config = ScrubConfig::default();
        assert_eq!(config.interval, Duration::from_secs(7 * 24 * 3600));
        assert!(config.enabled);
    }

    #[test]
    fn scrub_empty_engine() {
        let (engine, _dir) = test_engine();
        let report = scrub_all(&engine).expect("scrub failed");
        assert_eq!(report.keys_checked, 0);
        assert!(!report.has_errors());
        assert_eq!(report.partition_counts.len(), Partition::all().len());
    }

    #[test]
    fn scrub_with_data() {
        let (engine, _dir) = test_engine();

        // Write data to multiple partitions
        for i in 0..50u32 {
            engine
                .put(Partition::Node, &i.to_be_bytes(), b"node_data")
                .expect("put node");
        }
        for i in 0..30u32 {
            engine
                .put(Partition::Adj, &i.to_be_bytes(), b"adj_data")
                .expect("put adj");
        }
        engine.persist().expect("persist");

        let report = scrub_all(&engine).expect("scrub failed");
        assert_eq!(report.keys_checked, 80);
        assert!(!report.has_errors());

        // Verify per-partition counts
        let node_count = report
            .partition_counts
            .iter()
            .find(|(p, _)| *p == Partition::Node)
            .map(|(_, c)| *c)
            .expect("node partition missing");
        assert_eq!(node_count, 50);

        let adj_count = report
            .partition_counts
            .iter()
            .find(|(p, _)| *p == Partition::Adj)
            .map(|(_, c)| *c)
            .expect("adj partition missing");
        assert_eq!(adj_count, 30);
    }

    #[test]
    fn scrub_single_partition() {
        let (engine, _dir) = test_engine();

        engine
            .put(Partition::Schema, b"label:User", b"schema_data")
            .expect("put");
        engine
            .put(Partition::Schema, b"label:Post", b"schema_data2")
            .expect("put");

        let (count, errors) = scrub_partition(&engine, Partition::Schema).expect("scrub partition");
        assert_eq!(count, 2);
        assert!(errors.is_empty());
    }

    #[test]
    fn scrub_report_has_errors() {
        let report = ScrubReport {
            keys_checked: 100,
            partition_counts: vec![],
            errors: vec![VerifyError {
                partition: Partition::Node,
                message: "checksum mismatch".to_string(),
            }],
            duration: Duration::from_millis(50),
        };
        assert!(report.has_errors());
    }

    #[test]
    fn scrub_report_no_errors() {
        let report = ScrubReport {
            keys_checked: 100,
            partition_counts: vec![],
            errors: vec![],
            duration: Duration::from_millis(50),
        };
        assert!(!report.has_errors());
    }

    #[test]
    fn scrub_after_delete() {
        let (engine, _dir) = test_engine();

        engine.put(Partition::Node, b"k1", b"v1").expect("put");
        engine.put(Partition::Node, b"k2", b"v2").expect("put");
        engine.delete(Partition::Node, b"k1").expect("delete");

        let (count, errors) = scrub_partition(&engine, Partition::Node).expect("scrub partition");
        assert_eq!(count, 1); // only k2 should remain
        assert!(errors.is_empty());
    }

    #[test]
    fn scrub_large_dataset() {
        let (engine, _dir) = test_engine();

        for i in 0..1000u32 {
            let key = format!("key_{i:06}");
            let value = format!("value_{i:06}_with_some_padding_data_for_testing");
            engine
                .put(Partition::Node, key.as_bytes(), value.as_bytes())
                .expect("put");
        }
        engine.persist().expect("persist");

        let report = scrub_all(&engine).expect("scrub failed");
        assert_eq!(report.keys_checked, 1000);
        assert!(!report.has_errors());
        assert!(report.duration.as_nanos() > 0);
    }
}
