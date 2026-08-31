//! Benchmark: oplog append throughput and read-range latency (R074).
//!
//! Measures:
//!   1. oplog/append_throughput — entries/sec for sequential append
//!      (no background I/O, measures serialize + crc32 + write path)
//!   2. oplog/read_range — time to read N entries back from sealed segments
//!      (verifies all checksums, deserializes all payloads)

#![allow(clippy::expect_used)]

use std::time::Duration;

use coordinode_storage::oplog::{OplogEntry, OplogManager, OplogOp};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_entry(index: u64) -> OplogEntry {
    OplogEntry {
        ts: (index * 1_000_000) << 18,
        term: 1,
        index,
        shard: 0,
        ops: vec![OplogOp::Insert {
            partition: 1,
            key: format!("bench-key-{index:012}").into_bytes(),
            value: vec![0xABu8; 128],
        }],
        is_migration: false,
        pre_images: None,
    }
}

fn open_manager(dir: &std::path::Path) -> OplogManager {
    OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open manager")
}

// ── Append throughput (R074) ─────────────────────────────────────────────────

/// Benchmark: sequential append throughput.
///
/// Measures how quickly entries can be serialized, checksummed, and written
/// to disk through the oplog segment writer. Does NOT include segment rotation.
fn bench_oplog_append_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/oplog/append_throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    for &num_entries in &[1_000_usize, 10_000, 50_000] {
        group.bench_with_input(
            BenchmarkId::new("entries", num_entries),
            &num_entries,
            |b, &n| {
                b.iter(|| {
                    let dir = tempfile::TempDir::new().expect("tempdir");
                    let mut mgr = open_manager(dir.path());
                    for i in 0..n as u64 {
                        mgr.append(&make_entry(i)).expect("append");
                    }
                    mgr.rotate().expect("rotate");
                });
            },
        );
    }
    group.finish();
}

// ── Read-range latency (R074) ─────────────────────────────────────────────────

/// Benchmark: read-range across sealed segments.
///
/// Measures end-to-end deserialization + checksum verification cost for
/// reading N entries from sealed segment files.
fn bench_oplog_read_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/oplog/read_range");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    for &num_entries in &[1_000_usize, 10_000] {
        // Pre-populate data outside the bench loop
        let dir = tempfile::TempDir::new().expect("tempdir");
        {
            let mut mgr = open_manager(dir.path());
            for i in 0..num_entries as u64 {
                mgr.append(&make_entry(i)).expect("append");
            }
            mgr.rotate().expect("rotate");
        }

        group.bench_with_input(
            BenchmarkId::new("entries", num_entries),
            &num_entries,
            |b, &n| {
                b.iter(|| {
                    let mut mgr = open_manager(dir.path());
                    let entries = mgr.read_range(0, n as u64).expect("read_range");
                    assert_eq!(entries.len(), n);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_oplog_append_throughput,
    bench_oplog_read_range
);
criterion_main!(benches);
