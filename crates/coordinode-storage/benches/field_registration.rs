//! What a field registration costs at ordered apply.
//!
//! A registration is decided, journaled and applied under the engine's
//! decision lock, so the latency of one journaled registration is the time
//! that lock is held for it. Against it stands the same two records written
//! as plain puts, which is what the registration would cost if deciding were
//! free: the difference is the decision itself. A name that is already bound
//! is decided without writing anything, which is what a statement's lookup
//! miss costs when another proposer bound the name first. Eight writers
//! registering distinct names show how the lock behaves when it is contended.
//!
//! Run: cargo bench -p coordinode-storage --bench field_registration

#![allow(clippy::expect_used, clippy::print_stdout, clippy::panic)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::graph::intern::{encode_field_id, field_id_key, field_name_key};
use coordinode_core::txn::proposal::{MetadataCommand, Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::metadata::field_frontier;

/// Registrations per measured case.
const OPS: usize = 5_000;

/// Writers in the contended case.
const WRITERS: usize = 8;

fn open(dir: &tempfile::TempDir) -> (StorageEngine, Arc<TimestampOracle>) {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let oracle = Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&config, Arc::clone(&oracle)).expect("open");
    (engine, oracle)
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn line(label: &str, wall: Duration, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    println!(
        "{label:<34} n={:>6}  {:>8.0} op/s  mean={:>9.3?}  p50={:>9.3?}  p99={:>9.3?}  p999={:>9.3?}",
        samples.len(),
        samples.len() as f64 / wall.as_secs_f64(),
        total / samples.len() as u32,
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        percentile(&samples, 0.999),
    );
}

fn register(name: String) -> Vec<Mutation> {
    vec![Mutation::Command(MetadataCommand::RegisterFields {
        names: vec![name],
    })]
}

/// The two records a binding of `name` to `id` writes, as plain puts.
fn records(name: &str, id: u32) -> Vec<Mutation> {
    vec![
        Mutation::Put {
            partition: PartitionId::Schema,
            key: field_name_key(name),
            value: encode_field_id(id).to_vec(),
        },
        Mutation::Put {
            partition: PartitionId::Schema,
            key: field_id_key(id),
            value: name.as_bytes().to_vec(),
        },
    ]
}

/// Run `mutations(i)` journaled `OPS` times on one thread.
fn serial(
    label: &str,
    mutations: impl Fn(usize) -> Vec<Mutation>,
    setup: impl Fn(&StorageEngine, &TimestampOracle),
) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);
    setup(&engine, &oracle);
    let batches: Vec<_> = (0..OPS).map(&mutations).collect();
    let mut samples = Vec::with_capacity(OPS);
    let started = Instant::now();
    for batch in &batches {
        let ts = oracle.next().as_raw();
        let t = Instant::now();
        engine.commit_journaled(batch, ts).expect("commit");
        samples.push(t.elapsed());
    }
    line(label, started.elapsed(), samples);
}

/// Eight writers, each journaling `mutations(writer, i)`. Returns the engine
/// so the caller can check what the run bound.
fn contended(
    label: &str,
    mutations: impl Fn(usize, usize) -> Vec<Mutation> + Sync,
) -> (tempfile::TempDir, StorageEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, oracle) = open(&dir);
    let per_writer = OPS / WRITERS;
    let started = Instant::now();
    let samples: Vec<Duration> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..WRITERS)
            .map(|w| {
                let (engine, oracle, mutations) = (&engine, &oracle, &mutations);
                s.spawn(move || {
                    let mut out = Vec::with_capacity(per_writer);
                    for i in 0..per_writer {
                        let batch = mutations(w, i);
                        let ts = oracle.next().as_raw();
                        let t = Instant::now();
                        engine.commit_journaled(&batch, ts).expect("commit");
                        out.push(t.elapsed());
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("writer"))
            .collect()
    });
    line(label, started.elapsed(), samples);
    (dir, engine)
}

fn main() {
    println!("== field registration at ordered apply (journaled, durable) ==");
    serial(
        "register new name",
        |i| register(format!("field_{i:06}")),
        |_, _| {},
    );
    serial(
        "same records as plain puts",
        |i| records(&format!("field_{i:06}"), i as u32 + 1),
        |_, _| {},
    );
    serial(
        "register an already-bound name",
        |i| register(format!("field_{:06}", i % 64)),
        |engine, oracle| {
            for i in 0..64 {
                engine
                    .commit_journaled(&register(format!("field_{i:06}")), oracle.next().as_raw())
                    .expect("seed");
            }
        },
    );
    let (_dir, engine) = contended("register new, 8 writers", |w, i| {
        register(format!("w{w}_field_{i:06}"))
    });
    assert_eq!(
        field_frontier(&engine).expect("frontier") as usize,
        OPS / WRITERS * WRITERS,
        "every name bound exactly once"
    );
    // Distinct ids per writer, so the records never collide.
    contended("same records as puts, 8 writers", |w, i| {
        let id = (w * (OPS / WRITERS) + i) as u32 + 1;
        records(&format!("w{w}_field_{i:06}"), id)
    });

    let name = "field_000001";
    let bytes: usize = records(name, 1)
        .iter()
        .map(|m| match m {
            Mutation::Put { key, value, .. } => key.len() + value.len(),
            _ => 0,
        })
        .sum();
    println!(
        "\nlogical bytes per binding of a {}-byte name: {bytes} (two write-once records); \
         a statement with only known names writes none",
        name.len()
    );
}
