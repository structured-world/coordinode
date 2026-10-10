//! What replacing one index generation's local copy costs the writers around
//! it, and what it moves compared with repairing the whole index partition.
//!
//! Writers commit through the journal throughout, each commit touching the
//! replaced generation, another generation and a node record. Their commit
//! latency is reported for a quiet stretch and for the replacement itself, so
//! the pauses the replacement takes (registering the copy, publishing it) show
//! as the tail they add, not as a mean. Each step's own duration and the bytes
//! of the generation's history are reported beside the bytes a whole-partition
//! copy of the index partition carries.
//!
//! Run: cargo bench -p coordinode-storage --bench generation_replace

#![allow(clippy::expect_used, clippy::print_stdout, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::index::encoding::GENERATION_TAGS;
use coordinode_core::index::identity::GenerationId;
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::installation::HistoryEntry;
use coordinode_storage::engine::partition::Partition;

/// Entries seeded into each of the two generations.
const SEEDED: u64 = 200_000;

/// Concurrent writers.
const WRITERS: usize = 4;

/// The quiet stretch measured before the replacement.
const QUIET: Duration = Duration::from_secs(5);

const REPLACED: u64 = 40;
const OTHER: u64 = 41;

fn entry(generation: u64, i: u64) -> Vec<u8> {
    let mut key = vec![GENERATION_TAGS[0]];
    key.extend_from_slice(&generation.to_be_bytes());
    key.extend_from_slice(&i.to_be_bytes());
    key
}

fn put(partition: PartitionId, key: Vec<u8>, value: &[u8]) -> Mutation {
    Mutation::Put {
        partition,
        key,
        value: value.to_vec(),
    }
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
        println!("{label:<30} none");
        return;
    }
    samples.sort_unstable();
    println!(
        "{label:<30} n={:>7}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}  max={:>9.3?}",
        samples.len(),
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
        samples[samples.len() - 1],
    );
}

/// One step of the replacement and the interval it ran in.
struct Step {
    name: &'static str,
    start: Instant,
    end: Instant,
    bytes: usize,
}

impl Step {
    /// A step that started at `start` and ends now.
    fn new(name: &'static str, start: Instant, bytes: usize) -> Self {
        Self {
            name,
            start,
            end: Instant::now(),
            bytes,
        }
    }
}

fn history_bytes(entries: &[HistoryEntry]) -> usize {
    entries
        .iter()
        .map(|e| match e {
            HistoryEntry::Put { key, value, .. } => key.len() + value.len() + 9,
            HistoryEntry::Delete { key, .. } => key.len() + 9,
            HistoryEntry::RemoveRange { start, end, .. } => start.len() + end.len() + 9,
        })
        .sum()
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_embedded(&config, oracle.clone()).expect("open embedded"));

    // Seed both generations in batches, as a bulk load would.
    let value = [7u8; 16];
    for start in (0..SEEDED).step_by(1_000) {
        let batch: Vec<Mutation> = (start..start + 1_000)
            .flat_map(|i| {
                [
                    put(PartitionId::Idx, entry(REPLACED, i), &value),
                    put(PartitionId::Idx, entry(OTHER, i), &value),
                ]
            })
            .collect();
        engine
            .commit_journaled(&batch, oracle.next().as_raw())
            .expect("seed");
    }
    engine.persist().expect("persist");

    let replacing = AtomicBool::new(false);
    let stop = AtomicBool::new(false);
    let generation = GenerationId::from_raw(REPLACED);

    let (quiet, during, steps) = std::thread::scope(|scope| {
        let writers: Vec<_> = (0..WRITERS)
            .map(|w| {
                let engine = Arc::clone(&engine);
                let oracle = Arc::clone(&oracle);
                let (replacing, stop) = (&replacing, &stop);
                scope.spawn(move || {
                    let mut quiet = Vec::new();
                    let mut during: Vec<(Instant, Duration)> = Vec::new();
                    let mut i = SEEDED + w as u64 * 10_000_000;
                    while !stop.load(Ordering::Acquire) {
                        let batch = [
                            put(PartitionId::Idx, entry(REPLACED, i), &value),
                            put(PartitionId::Idx, entry(OTHER, i), &value),
                            put(
                                PartitionId::Node,
                                format!("node:{w}:{i}").into_bytes(),
                                &value,
                            ),
                        ];
                        let in_replacement = replacing.load(Ordering::Acquire);
                        let started = Instant::now();
                        engine
                            .commit_journaled(&batch, oracle.next().as_raw())
                            .expect("commit");
                        let took = started.elapsed();
                        if in_replacement || replacing.load(Ordering::Acquire) {
                            during.push((started, took));
                        } else {
                            quiet.push(took);
                        }
                        i += 1;
                    }
                    (quiet, during)
                })
            })
            .collect();

        std::thread::sleep(QUIET);
        replacing.store(true, Ordering::Release);
        let mut steps: Vec<Step> = Vec::new();
        let t = Instant::now();
        engine.stage_generation(generation).expect("stage");
        steps.push(Step::new("register (fence)", t, 0));
        let t = Instant::now();
        let history = engine
            .export_generation_history(generation)
            .expect("export");
        let bytes = history_bytes(&history.entries);
        steps.push(Step::new("export history", t, bytes));
        let t = Instant::now();
        engine
            .import_generation_history(generation, &history.entries)
            .expect("import");
        steps.push(Step::new("import history", t, bytes));
        let t = Instant::now();
        engine
            .finish_generation_import(generation, history.covers_through, history.history_from)
            .expect("finish");
        steps.push(Step::new("finish", t, 0));
        let t = Instant::now();
        engine.publish_generation(generation).expect("publish");
        steps.push(Step::new("publish (fence, flush)", t, 0));
        replacing.store(false, Ordering::Release);
        std::thread::sleep(Duration::from_millis(500));
        stop.store(true, Ordering::Release);

        let mut quiet = Vec::new();
        let mut during = Vec::new();
        for writer in writers {
            let (q, d) = writer.join().expect("writer");
            quiet.extend(q);
            during.extend(d);
        }
        (quiet, during, steps)
    });

    let whole = engine.copy_partition(Partition::Idx).expect("copy");
    let whole_bytes: usize = whole.rows.iter().map(|(k, v)| k.len() + v.len()).sum();

    println!(
        "== generation replacement, {SEEDED} seeded entries per generation, {WRITERS} writers =="
    );
    line("commit, quiet", quiet);
    line(
        "commit, during replacement",
        during.iter().map(|(_, took)| *took).collect(),
    );
    println!(
        "{:<30} {:>9}  {:>14}  bytes",
        "step", "took", "slowest commit"
    );
    for step in steps {
        // The slowest commit whose own interval overlapped the step: what the
        // step can have cost a writer, a pause included.
        let slowest = during
            .iter()
            .filter(|(started, took)| *started <= step.end && *started + *took >= step.start)
            .map(|(_, took)| *took)
            .max()
            .unwrap_or_default();
        println!(
            "{:<30} {:>9.3?}  {:>14.3?}  {}",
            step.name,
            step.end - step.start,
            slowest,
            step.bytes
        );
    }
    println!(
        "whole Idx partition copy       {} rows, {whole_bytes} bytes",
        whole.rows.len()
    );
}
