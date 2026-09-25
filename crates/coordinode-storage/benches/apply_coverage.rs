//! What apply coverage costs.
//!
//! Every apply writes one 16-byte marker key per tree it touches, in the
//! batch it already writes, and every 4096 applies a fold writes one base and
//! one range tombstone per tree. This measures the apply with and without the
//! marker on the same three-tree proposal (distribution, not only the mean),
//! the journalled commit end to end, one fold, and the on-disk bytes the
//! markers add once flushed.
//!
//! Run: cargo bench -p coordinode-storage --bench apply_coverage

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;

/// Applies measured per series.
const SAMPLES: u64 = 20_000;

fn config(dir: &tempfile::TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// A node put, an edge and a degree delta: three trees, so three markers.
fn proposal(i: u64) -> Vec<Mutation> {
    vec![
        Mutation::Put {
            partition: PartitionId::Node,
            key: format!("node:00:{i:012}").into_bytes(),
            value: vec![0x5a; 64],
        },
        Mutation::Merge {
            partition: PartitionId::Adj,
            key: format!("adj:R:out:{}", i % 1024).into_bytes(),
            operand: coordinode_storage::engine::merge::encode_add(i),
        },
        Mutation::Merge {
            partition: PartitionId::Counter,
            key: format!("counter:degree:{}", i % 1024).into_bytes(),
            operand: coordinode_storage::engine::merge::encode_counter_delta(1),
        },
    ]
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn line(label: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    println!(
        "{label:<30} n={:>6}  mean={:>9.3?}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}",
        samples.len(),
        total / samples.len() as u32,
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
    );
}

/// Bytes on disk: flushed as written, and after a major compaction of every
/// tree (what the markers settle to once their folds' tombstones are applied).
struct Footprint {
    flushed: u64,
    compacted: u64,
}

/// Apply `SAMPLES` proposals through `apply` (which also folds, when it
/// marks), then flush; returns the per-apply latencies and the footprint.
fn series(apply: impl Fn(&StorageEngine, &TimestampOracle, u64)) -> (Vec<Duration>, Footprint) {
    use coordinode_storage::engine::partition::Partition;
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_with_oracle(&config(&dir), Arc::clone(&oracle)).expect("open");
    engine.reset_raft_coverage(0, &[]).expect("establish");
    let mut samples = Vec::with_capacity(SAMPLES as usize);
    for i in 0..SAMPLES {
        let start = Instant::now();
        apply(&engine, &oracle, i);
        samples.push(start.elapsed());
    }
    engine.persist().expect("persist");
    let flushed = engine.disk_space().expect("disk space");
    for part in [Partition::Node, Partition::Adj, Partition::Counter] {
        engine.force_compaction(part).expect("compact");
    }
    let compacted = engine.disk_space().expect("disk space");
    (samples, Footprint { flushed, compacted })
}

/// The fold cadence the state machine uses.
const FOLD_EVERY: u64 = 4096;

fn main() {
    let (plain, plain_bytes) = series(|engine, oracle, i| {
        engine
            .apply_proposal_at(&proposal(i), oracle.next().as_raw())
            .expect("apply");
    });
    let (marked, marked_bytes) = series(|engine, oracle, i| {
        engine
            .apply_raft_proposal(&proposal(i), oracle.next().as_raw(), i, 0, |_| false)
            .expect("apply");
        let next = i + 1;
        if next % FOLD_EVERY == 0 {
            engine.fold_raft_coverage(next - FOLD_EVERY, next, &next.to_be_bytes());
        }
    });
    line("apply, no marker", plain);
    line("apply, marker in 3 trees", marked);
    let delta = |with: u64, without: u64| {
        format!(
            "{with} B vs {without} B ({:+} B, {:+.2}%)",
            with as i64 - without as i64,
            (with as f64 / without as f64 - 1.0) * 100.0
        )
    };
    println!(
        "on disk after {SAMPLES} applies, flushed:   {}",
        delta(marked_bytes.flushed, plain_bytes.flushed)
    );
    println!(
        "on disk after {SAMPLES} applies, compacted: {}",
        delta(marked_bytes.compacted, plain_bytes.compacted)
    );

    // One fold over 4096 markers per tree, the size the commit path folds at.
    let mut folds = Vec::new();
    for round in 0..20u64 {
        let dir = tempfile::tempdir().expect("tempdir");
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            StorageEngine::open_with_oracle(&config(&dir), Arc::clone(&oracle)).expect("open");
        for i in 0..4096 {
            engine
                .apply_raft_proposal(&proposal(i), oracle.next().as_raw(), i, 0, |_| false)
                .expect("apply");
        }
        let start = Instant::now();
        engine.fold_raft_coverage(0, 4096, &round.to_be_bytes());
        folds.push(start.elapsed());
    }
    line("fold of 4096 markers", folds);

    // The journalled commit end to end: journal append and fsync, the apply,
    // its markers, and the commit-path folds.
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&config(&dir), Arc::clone(&oracle)).expect("open");
    let commits: Vec<Duration> = (0..2_000u64)
        .map(|i| {
            let start = Instant::now();
            engine
                .commit_journaled(&proposal(i), oracle.next().as_raw())
                .expect("commit");
            start.elapsed()
        })
        .collect();
    line("commit_journaled (fsync)", commits);
}
