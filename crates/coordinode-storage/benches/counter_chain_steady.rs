//! What the planner's counter costs to read while the system keeps running.
//!
//! The companion point benchmark writes a burst and reads straight after, so
//! it cannot say whether background compaction bounds the operand chain or
//! merely lags it. This one keeps writing, pauses to let compaction settle,
//! and samples the read at each step, so the shape of the curve answers the
//! question: a chain that compaction bounds gives a flat line, one it never
//! reaches gives a rising one.
//!
//! The second arm is the alternative that needs no engine change: the writer
//! materialises the counter itself every so often with an ordinary put, which
//! gives compaction the proven base it otherwise lacks. It is measured here
//! for its cost only. Whether it is sound is a separate question, and it is
//! not: reading a sum and writing it back supersedes every delta that landed
//! in between, which is the read-modify-write that merge operands exist to
//! avoid.
//!
//! Run: cargo bench -p coordinode-storage --bench counter_chain_steady

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_core::graph::stats::NODES_TOTAL_KEY;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::merge::encode_counter_delta;
use coordinode_storage::engine::partition::Partition;

/// Deltas between flushes. The engine also flushes a non-empty memtable on an
/// age timer, so this stands for a flush cadence in time at a given write rate
/// rather than for anything the writer decides.
const DELTAS_PER_FLUSH: usize = 256;

/// Deltas between samples.
const DELTAS_PER_SAMPLE: usize = 20_000;

/// Total deltas per arm.
const TOTAL_DELTAS: usize = 200_000;

/// How long compaction is given to settle before a sample, so the sample
/// measures the state the system rests in and not the one it is leaving.
const SETTLE: Duration = Duration::from_secs(2);

/// Reads timed per sample.
const READS_PER_SAMPLE: u32 = 200;

fn open_engine(dir: &tempfile::TempDir) -> StorageEngine {
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

/// Mean nanoseconds of a point read of the counter.
fn sample_read_ns(engine: &StorageEngine) -> u128 {
    let start = Instant::now();
    for _ in 0..READS_PER_SAMPLE {
        std::hint::black_box(read_total(engine));
    }
    start.elapsed().as_nanos() / u128::from(READS_PER_SAMPLE)
}

/// Write `TOTAL_DELTAS` deltas, sampling the read as it goes. With
/// `checkpoint_every` set, the writer also materialises the counter with a put
/// at that interval.
fn run(name: &str, checkpoint_every: Option<usize>) {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let engine = open_engine(&dir);
    let operand = encode_counter_delta(1);

    println!("\n{name}");
    println!("{:>10}  {:>12}  {:>10}", "deltas", "read (ns)", "value");

    for i in 1..=TOTAL_DELTAS {
        engine
            .merge(Partition::Counter, NODES_TOTAL_KEY, &operand)
            .expect("merge");

        if i % DELTAS_PER_FLUSH == 0 {
            engine.persist().expect("persist");
        }

        if let Some(every) = checkpoint_every {
            if i % every == 0 {
                // Deliberately the unsound form, measured for its cost: the
                // sum is read and written back, so anything that landed since
                // the read is superseded by the put.
                let total = read_total(&engine);
                engine
                    .put(Partition::Counter, NODES_TOTAL_KEY, &total.to_le_bytes())
                    .expect("put");
            }
        }

        if i % DELTAS_PER_SAMPLE == 0 {
            std::thread::sleep(SETTLE);
            let ns = sample_read_ns(&engine);
            println!("{:>10}  {:>12}  {:>10}", i, ns, read_total(&engine));
        }
    }
}

fn main() {
    run("merge only, compaction left to itself", None);
    run("writer materialises every 1000 deltas", Some(1_000));
}
