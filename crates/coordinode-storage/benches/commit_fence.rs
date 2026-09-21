//! What the commit fence costs and what it refuses.
//!
//! The fence registers a commit's write scope together with its timestamp and
//! holds it until the writes are local state. Two numbers decide whether that
//! is affordable: what an uncontended commit pays for a protection it never
//! needs, and how the contended case behaves once refusals are real rather
//! than silent overwrites.
//!
//! Throughput alone hides both. A mean latency over a contended key averages
//! the commits that won with the ones that were refused, and a workload that
//! used to lose updates was faster precisely because it was losing them. So
//! this reports the distribution of each outcome separately, and the share of
//! attempts that had to retry.
//!
//! Run: cargo bench -p coordinode-storage --bench commit_fence

#![allow(clippy::expect_used, clippy::print_stdout, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::{CommitContext, CommitError, Transaction};

/// Commits attempted per writer.
const PER_WRITER: usize = 500;

/// Writers, enough to make the contended case contend.
const WRITERS: usize = 8;

fn open(dir: &tempfile::TempDir) -> (Arc<StorageEngine>, Arc<TimestampOracle>) {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open"));
    (engine, oracle)
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn line(label: &str, mut samples: Vec<Duration>) {
    if samples.is_empty() {
        println!("{label:<28} none");
        return;
    }
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    println!(
        "{label:<28} n={:>6}  mean={:>9.3?}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}",
        samples.len(),
        total / samples.len() as u32,
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
    );
}

/// One read-modify-write cycle, retried until it lands. Returns the latency of
/// each outcome separately: a refusal is work the caller paid for and has to
/// pay again, so averaging it into the commit that finally succeeded would
/// describe a system nobody is running.
fn increment(
    engine: &StorageEngine,
    oracle: &TimestampOracle,
    key: &[u8],
    committed: &mut Vec<Duration>,
    refused: &mut Vec<Duration>,
) {
    let wc = WriteConcern::default();
    loop {
        let snap = engine.snapshot();
        let mut txn = Transaction::new(
            engine,
            Some(oracle),
            Timestamp::from_raw(oracle.next().as_raw()),
            Some(snap),
        );
        let current = txn
            .get(Partition::Node, key)
            .expect("read")
            .map(|v| u64::from_le_bytes((&v[..]).try_into().expect("eight bytes")))
            .unwrap_or(0);
        txn.put(Partition::Node, key, &(current + 1).to_le_bytes())
            .expect("stage");

        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        let started = Instant::now();
        match txn.commit(&ctx) {
            Ok(_) => {
                committed.push(started.elapsed());
                return;
            }
            Err(CommitError::Conflict(_)) => refused.push(started.elapsed()),
            Err(e) => panic!("unexpected commit failure: {e:?}"),
        }
    }
}

/// `shared`: every writer increments one key, which is the case the fence
/// exists for. Otherwise each writer owns its key and nothing should ever be
/// refused, which is what says the fence costs the uncontended path nothing
/// but its own bookkeeping.
fn run(shared: bool) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);

    for w in 0..WRITERS {
        let key = format!("node:{}", if shared { 0 } else { w });
        engine
            .put(Partition::Node, key.as_bytes(), &0u64.to_le_bytes())
            .expect("seed");
    }

    let attempts = AtomicU64::new(0);
    let started = Instant::now();
    let (committed, refused): (Vec<Duration>, Vec<Duration>) = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                let engine = Arc::clone(&engine);
                let oracle = Arc::clone(&oracle);
                let attempts = &attempts;
                scope.spawn(move || {
                    let key = format!("node:{}", if shared { 0 } else { w });
                    let mut committed = Vec::with_capacity(PER_WRITER);
                    let mut refused = Vec::new();
                    for _ in 0..PER_WRITER {
                        increment(
                            &engine,
                            &oracle,
                            key.as_bytes(),
                            &mut committed,
                            &mut refused,
                        );
                    }
                    attempts.fetch_add((committed.len() + refused.len()) as u64, Ordering::Relaxed);
                    (committed, refused)
                })
            })
            .collect();
        let mut all_committed = Vec::new();
        let mut all_refused = Vec::new();
        for handle in handles {
            let (c, r) = handle.join().expect("writer");
            all_committed.extend(c);
            all_refused.extend(r);
        }
        (all_committed, all_refused)
    });
    let wall = started.elapsed();

    let total = attempts.load(Ordering::Relaxed);
    println!(
        "\n-- {} key, {WRITERS} writers --",
        if shared {
            "one shared"
        } else {
            "one per writer"
        }
    );
    println!(
        "useful throughput            {:.0} commits/s over {:?}; {} attempts for {} commits \
         ({:.1}% refused)",
        committed.len() as f64 / wall.as_secs_f64(),
        wall,
        total,
        committed.len(),
        refused.len() as f64 * 100.0 / total.max(1) as f64,
    );
    line("committed", committed);
    line("refused", refused);
}

fn main() {
    println!("== commit fence ==");
    run(false);
    run(true);
}
