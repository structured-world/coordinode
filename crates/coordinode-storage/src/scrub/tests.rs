use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use std::path::PathBuf;
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

/// Recursively collect every regular file under `root`.
fn collect_files(root: &std::path::Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(root).expect("read_dir");
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_files(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

/// The largest regular file under `root` — after flushing a few hundred values,
/// that is the SST holding the data (far bigger than manifest / version files).
fn largest_file(root: &std::path::Path) -> PathBuf {
    let mut files = Vec::new();
    collect_files(root, &mut files);
    files
        .into_iter()
        .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .expect("at least one file on disk after persist")
}

#[test]
fn scrub_config_defaults() {
    let config = ScrubConfig::default();
    assert!(config.enabled);
    assert_eq!(config.interval, Duration::from_secs(7 * 24 * 3600));
    assert_eq!(config.throttle, None);
    assert_eq!(config.parallelism, 1);
}

#[test]
fn scrub_empty_engine() {
    let (engine, _dir) = test_engine();
    let report = scrub_all(&engine, &ScrubConfig::default()).expect("scrub failed");
    // No SSTs flushed yet → nothing on disk to verify, and no corruption.
    assert_eq!(report.blocks_checked, 0);
    assert_eq!(report.sst_files_checked, 0);
    assert!(!report.has_errors());
    assert_eq!(report.partition_counts.len(), Partition::all().len());
}

#[test]
fn scrub_clean_data_reports_no_errors() {
    let (engine, _dir) = test_engine();

    for i in 0..200u32 {
        engine
            .put(Partition::Node, &i.to_be_bytes(), b"node_data")
            .expect("put node");
    }
    for i in 0..120u32 {
        engine
            .put(Partition::Adj, &i.to_be_bytes(), b"adj_data")
            .expect("put adj");
    }
    engine.persist().expect("persist");

    let report = scrub_all(&engine, &ScrubConfig::default()).expect("scrub failed");
    assert!(!report.has_errors(), "clean data must scrub clean");
    // Flushed data produced at least one SST with at least one block.
    assert!(report.sst_files_checked > 0, "expected flushed SSTs");
    assert!(report.blocks_checked > 0, "expected verified blocks");
    assert_eq!(report.partition_counts.len(), Partition::all().len());
    assert!(report.duration.as_nanos() > 0);
}

#[test]
fn scrub_single_partition_clean() {
    let (engine, _dir) = test_engine();

    for i in 0..64u32 {
        engine
            .put(
                Partition::Schema,
                format!("label:{i}").as_bytes(),
                b"schema",
            )
            .expect("put");
    }
    engine.persist().expect("persist");

    let (blocks, ssts, errors) =
        scrub_partition(&engine, Partition::Schema, &ScrubConfig::default())
            .expect("scrub partition");
    assert!(blocks > 0);
    assert!(ssts > 0);
    assert!(errors.is_empty());
}

#[test]
fn scrub_with_throttle_runs_clean() {
    let (engine, _dir) = test_engine();
    for i in 0..100u32 {
        engine
            .put(Partition::Node, &i.to_be_bytes(), b"v")
            .expect("put");
    }
    engine.persist().expect("persist");

    let config = ScrubConfig {
        throttle: Some(Duration::from_millis(1)),
        parallelism: 2,
        ..ScrubConfig::default()
    };
    let report = scrub_all(&engine, &config).expect("throttled scrub");
    assert!(!report.has_errors());
    assert!(report.sst_files_checked > 0);
}

/// The whole point of the rewrite: an on-disk corruption is REPORTED (collected
/// into the report), and the scan still returns `Ok` — it does not abort on the
/// first bad block (the previous implementation propagated the first error via
/// `?`, masking everything after it).
#[test]
fn scrub_reports_disk_corruption_without_aborting() {
    let (engine, dir) = test_engine();

    for i in 0..300u32 {
        engine
            .put(Partition::Node, &i.to_be_bytes(), b"payload-bytes-padding")
            .expect("put");
    }
    engine.persist().expect("persist");

    // Flip a byte in the middle of the largest on-disk file (the data SST),
    // simulating silent bit rot.
    let victim = largest_file(dir.path());
    let mut bytes = std::fs::read(&victim).expect("read sst");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&victim, &bytes).expect("write corrupted sst");

    // Scan completes (Ok) and surfaces the corruption rather than bailing.
    let report = scrub_all(&engine, &ScrubConfig::default()).expect("scrub must not abort");
    assert!(
        report.has_errors(),
        "corrupted block must be reported, got a clean report"
    );
    assert!(
        report.errors.iter().any(|e| e.partition == Partition::Node),
        "the corrupt block is in the Node partition"
    );
}

#[test]
fn scrub_report_error_flag() {
    let with = ScrubReport {
        blocks_checked: 100,
        sst_files_checked: 4,
        partition_counts: vec![],
        errors: vec![VerifyError {
            partition: Partition::Node,
            message: "checksum mismatch".to_string(),
        }],
        duration: Duration::from_millis(50),
    };
    assert!(with.has_errors());

    let without = ScrubReport {
        errors: vec![],
        ..with.clone()
    };
    assert!(!without.has_errors());
}
