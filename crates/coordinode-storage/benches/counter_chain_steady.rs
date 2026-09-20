//! What a merge chain costs: bytes on disk, read latency at the tail, the
//! write amplification of folding it, and the price of opening a directory
//! that holds one.
//!
//! A chain of operands is cheap to write and is paid for on every read, so a
//! mean read latency is the least interesting number about it. This reports
//! the distribution, the bytes the chain occupies before and after it folds,
//! and how long a directory holding one takes to open, because those are the
//! costs an operator meets and none of them follows from the others.
//!
//! Run: cargo bench -p coordinode-storage --bench counter_chain_steady

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_core::graph::stats::NODES_TOTAL_KEY;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::merge::encode_counter_delta;
use coordinode_storage::engine::partition::Partition;

/// Deltas between flushes, so the chain is spread over tables rather than
/// sitting in one memtable.
const DELTAS_PER_FLUSH: usize = 256;

/// Reads timed per measurement.
const READS: usize = 2_000;

fn open(dir: &tempfile::TempDir) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open")
}

fn read_total(engine: &StorageEngine) -> i64 {
    let value = engine
        .get(Partition::Counter, NODES_TOTAL_KEY)
        .expect("get")
        .expect("the key was written");
    i64::from_le_bytes(value.as_ref().try_into().expect("eight bytes"))
}

/// Bytes on disk under the directory. The counter partition is the only thing
/// written here, so this is the chain's footprint plus a fixed overhead that
/// the empty baseline reports.
fn bytes_on_disk(path: &std::path::Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            total += bytes_on_disk(&entry.path());
        } else {
            total += meta.len();
        }
    }
    total
}

/// Read latency at the median and the tail, in nanoseconds.
///
/// Reported rather than a mean because a chain's cost is a long read that
/// happens on every planning call: the median says what it usually costs and
/// the tail says what it costs when the chain is at its longest.
fn read_latency(engine: &StorageEngine) -> (u128, u128, u128) {
    let mut samples: Vec<u128> = Vec::with_capacity(READS);
    for _ in 0..READS {
        let start = Instant::now();
        std::hint::black_box(read_total(engine));
        samples.push(start.elapsed().as_nanos());
    }
    samples.sort_unstable();
    let at = |q: f64| samples[((samples.len() as f64 * q) as usize).min(samples.len() - 1)];
    (at(0.50), at(0.99), at(0.999))
}

/// The same cost for a document chain, which the operator declines to fold
/// before its base is known.
///
/// A counter chain is the best case for folding and a document chain is the
/// case that cannot fold at all, so this is what declining costs a reader: the
/// chain is carried to its base whole and every read walks it. Whether asking
/// the engine for a per-chain decision is worth anything depends on this
/// number, which is why it is measured rather than assumed.
fn document_chain() {
    use coordinode_core::graph::doc_delta::{DocDelta, PathTarget};
    use coordinode_core::graph::node::{NodeId, encode_node_key};

    println!("\ndocument chain (declines to fold)");
    println!(
        "{:>10}  {:>10}  {:>10}  {:>10}  {:>12}",
        "operands", "p50 (ns)", "p99 (ns)", "p999 (ns)", "bytes"
    );

    for operands in [100usize, 1_000, 10_000] {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let engine = open(&dir);
        let key = encode_node_key(0, NodeId::from_raw(1));

        // A chain of patches with no proven base: the shape a document key
        // holds between the write that made it and the compaction that can
        // prove it, which is what the operator has to carry whole.
        for i in 0..operands {
            let operand = DocDelta::SetPath {
                target: PathTarget::Extra,
                path: vec![format!("f{}", i % 8)],
                value: rmpv::Value::Integer((i as i64).into()),
            }
            .encode()
            .expect("encode the delta");
            engine
                .merge(Partition::Node, &key, &operand)
                .expect("merge");
            if i % DELTAS_PER_FLUSH == 0 {
                engine.persist().expect("persist");
            }
        }
        engine.persist().expect("persist");

        let mut samples: Vec<u128> = Vec::with_capacity(READS);
        for _ in 0..READS {
            let start = Instant::now();
            std::hint::black_box(engine.get(Partition::Node, &key).expect("get"));
            samples.push(start.elapsed().as_nanos());
        }
        samples.sort_unstable();
        let at = |q: f64| samples[((samples.len() as f64 * q) as usize).min(samples.len() - 1)];
        let bytes = bytes_on_disk(dir.path());

        println!(
            "{operands:>10}  {:>10}  {:>10}  {:>10}  {bytes:>12}",
            at(0.50),
            at(0.99),
            at(0.999)
        );
    }
}

fn main() {
    println!(
        "{:>10}  {:>10}  {:>10}  {:>10}  {:>12}  {:>12}  {:>10}",
        "operands", "p50 (ns)", "p99 (ns)", "p999 (ns)", "bytes", "folded", "reopen (ms)"
    );

    for operands in [1_000usize, 10_000, 100_000] {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let engine = open(&dir);
        let operand = encode_counter_delta(1);

        let write_start = Instant::now();
        for i in 1..=operands {
            engine
                .merge(Partition::Counter, NODES_TOTAL_KEY, &operand)
                .expect("merge");
            if i % DELTAS_PER_FLUSH == 0 {
                engine.persist().expect("persist");
            }
        }
        engine.persist().expect("persist");
        let write_time = write_start.elapsed();

        let (p50, p99, p999) = read_latency(&engine);
        let unfolded = bytes_on_disk(dir.path());

        // Folding is the write amplification of the chain: the bytes it costs
        // to turn the operands into the one value they mean.
        let fold_start = Instant::now();
        engine
            .major_compact(Partition::Counter)
            .expect("fold the chain");
        let fold_time = fold_start.elapsed();
        let folded = bytes_on_disk(dir.path());

        assert_eq!(read_total(&engine), operands as i64, "the sum survives");
        drop(engine);

        // Recovery: opening a directory that holds the folded chain.
        let reopen_start = Instant::now();
        let reopened = open(&dir);
        let reopen = reopen_start.elapsed();
        assert_eq!(
            read_total(&reopened),
            operands as i64,
            "and survives a reopen"
        );

        println!(
            "{operands:>10}  {p50:>10}  {p99:>10}  {p999:>10}  {unfolded:>12}  {folded:>12}  {:>10.1}",
            reopen.as_secs_f64() * 1_000.0
        );
        println!(
            "{:>10}  write {:?}, fold {:?}, amplification {:.2}x",
            "",
            Duration::from_millis(write_time.as_millis() as u64),
            Duration::from_millis(fold_time.as_millis() as u64),
            unfolded as f64 / folded.max(1) as f64
        );
    }

    document_chain();
}
