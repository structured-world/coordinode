//! What the closure ledger costs on disk and in memory: its window of
//! breakpoints kept as Schema records, the newest of them cached.
//!
//! Each entry is applied as the state machine applies it: its commit, then
//! its closure record, then the plan installed. Two windows:
//!
//! - `overflow`: `CLOSURE_ENTRIES` commits (default 100000), each raising
//!   the ceiling, with no closed bound, so nothing retires and the window
//!   holds a breakpoint per entry, far past the cached ones;
//! - `bounded`: the same commits with a closed bound `CLOSURE_LAG` entries
//!   (default 1000) behind, so retirement keeps the window near that lag.
//!
//! One JSON line per window: the ledger part of an entry's apply (latency;
//! the most bytes one held at once and what it left held), the records held
//! at the end and their peak, lookups the cached breakpoints answer (`hit`)
//! and lookups they miss, answered from a stored record (`miss`; latency,
//! peak bytes), the reopen and the ledger's load (time, bytes read through
//! the filesystem from Linux `/proc/self/io` `rchar`, and the bytes the load
//! left held), and the Schema tree's bytes.
//!
//! The misses run twice over the same positions: the first pass right after
//! the reopen (the tree's block cache empty), the second with it warm.
//! Neither drops the operating system's page cache, so neither is a read
//! from the device. Every lookup is the synchronous store call on the bench
//! thread: what a read fence adds on top (a blocking thread, a place among
//! the bounded stored reads) is not in these numbers.
//!
//! Run: `cargo bench -p coordinode-raft --bench closure_ledger`.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::Instant;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_test_fixtures::alloc_peak::peak_and_kept_of;
use lsm_tree::AbstractTree as _;

#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn rchar() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("rchar:")?.trim().parse().ok())
        })
        .unwrap_or(0)
}

fn percentiles(mut values: Vec<u64>, unit: f64) -> serde_json::Value {
    values.sort_unstable();
    let at = |p: f64| {
        let i = ((values.len() as f64 - 1.0) * p).round() as usize;
        values.get(i).copied().unwrap_or(0) as f64 / unit
    };
    serde_json::json!({"p50": at(0.5), "p99": at(0.99), "p999": at(0.999), "max": at(1.0)})
}

/// The store as a member opens it: on a clock, which the commits applied
/// move and the oldest snapshot it serves follows.
fn open(dir: &std::path::Path) -> StorageEngine {
    StorageEngine::open_with_oracle(
        &config(dir),
        std::sync::Arc::new(coordinode_core::txn::timestamp::TimestampOracle::new()),
    )
    .expect("open")
}

/// The wall clock in microseconds, where commit timestamps live.
fn now_us() -> u64 {
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_micros(),
    )
    .expect("microseconds fit")
}

fn config(dir: &std::path::Path) -> StorageConfig {
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    // Every view the store serves is its latest: retirement follows the
    // closed bound alone.
    config.retention_window_secs = 0;
    config
}

/// The ledger part of one entry: planned, written and installed. Its time,
/// peak bytes and bytes left held.
struct Step {
    ns: u64,
    peak: u64,
    kept: u64,
}

/// Apply entry `index`, committing at `ts` and carrying the bound `below`.
fn entry(engine: &StorageEngine, index: u64, ts: u64, below: u64) -> Step {
    engine
        .apply_raft_proposal(
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: format!("node:00:{index:012}").into_bytes(),
                value: b"v".to_vec(),
            }],
            ts,
            index,
            0,
            |_| false,
        )
        .expect("apply");
    let at = Instant::now();
    let ((), peak, kept) = peak_and_kept_of(|| {
        if let Some((records, at)) = engine
            .closure_record(below, index, false)
            .expect("closure record")
        {
            engine
                .apply_raft_proposal(&records, at, index, 1, |_| false)
                .expect("closure records");
        }
        engine
            .raise_closure_frontier(below, index)
            .expect("install");
    });
    Step {
        ns: at.elapsed().as_nanos() as u64,
        peak: peak as u64,
        kept: kept as u64,
    }
}

/// Look up `positions`: latencies, and the most bytes one lookup held.
fn lookups(engine: &StorageEngine, positions: &[u64]) -> (Vec<u64>, u64) {
    let mut ns = Vec::with_capacity(positions.len());
    let mut peak = 0;
    for &index in positions {
        let at = Instant::now();
        let (_, held) = peak_and_kept_of(|| {
            std::hint::black_box(engine.commit_ceiling_through(index).expect("ceiling"))
        })
        .into_peak();
        ns.push(at.elapsed().as_nanos() as u64);
        peak = peak.max(held);
    }
    (ns, peak)
}

trait IntoPeak<T> {
    fn into_peak(self) -> (T, u64);
}

impl<T> IntoPeak<T> for (T, usize, usize) {
    fn into_peak(self) -> (T, u64) {
        (self.0, self.1 as u64)
    }
}

fn window(name: &str, entries: u64, lag: Option<u64>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = now_us();
    let mut update_ns = Vec::with_capacity(entries as usize);
    let mut update_peak = Vec::with_capacity(entries as usize);
    let mut update_kept = 0u64;
    let mut held_peak = 0;
    {
        let engine = open(dir.path());
        for index in 1..=entries {
            let ts = base + index * 10;
            let below = lag
                .and_then(|lag| index.checked_sub(lag))
                .map_or(0, |closed| base + closed * 10 + 1);
            let step = entry(&engine, index, ts, below);
            update_ns.push(step.ns);
            update_peak.push(step.peak);
            update_kept += step.kept;
            if index % 1024 == 0 {
                held_peak = held_peak.max(engine.closure_records_held().unwrap_or(0));
            }
        }
        held_peak = held_peak.max(engine.closure_records_held().unwrap_or(0));
        engine.persist().expect("persist");
    }

    // Positions over the whole log: the cached newest ones, and older ones
    // the cache misses (a bounded window answers them from its anchor).
    let step = usize::try_from((entries / 1000).max(1)).expect("small");
    let missed: Vec<u64> = (1..entries.saturating_sub(300)).step_by(step).collect();
    let cached: Vec<u64> = (entries - 200..=entries).collect();

    let read_before = rchar();
    let opened = Instant::now();
    let engine = open(dir.path());
    let open_ms = opened.elapsed().as_secs_f64() * 1000.0;
    let open_read = rchar() - read_before;
    // The ledger itself, as a member's state machine loads it at its open:
    // the head, the cached newest breakpoints and the first after the
    // anchor, never the whole window.
    let read_before = rchar();
    let loaded = Instant::now();
    let (_, load_peak, load_kept) =
        peak_and_kept_of(|| engine.reload_closure_frontier().expect("load the ledger"));
    let load_ms = loaded.elapsed().as_secs_f64() * 1000.0;
    let load_read = rchar() - read_before;
    let (miss_first, _) = lookups(&engine, &missed);
    let (miss_second, miss_peak) = lookups(&engine, &missed);
    let (hit_ns, hit_peak) = lookups(&engine, &cached);
    let schema_bytes = engine.tree(Partition::Schema).expect("schema").disk_space();

    let report = serde_json::json!({
        "window": name,
        "entries": entries,
        "lag": lag,
        "update_us": percentiles(update_ns, 1000.0),
        "update_peak_bytes": percentiles(update_peak, 1.0),
        "update_kept_bytes_per_entry": update_kept as f64 / entries as f64,
        "held_end": engine.closure_records_held(),
        "held_peak": held_peak,
        "reopen_ms": open_ms,
        "reopen_read_bytes": open_read,
        "ledger_load_ms": load_ms,
        "ledger_load_read_bytes": load_read,
        "ledger_load_peak_bytes": load_peak,
        "ledger_load_kept_bytes": load_kept,
        "hit_lookup_us": percentiles(hit_ns, 1000.0),
        "hit_lookup_peak_bytes": hit_peak,
        "miss_lookup_first_pass_us": percentiles(miss_first, 1000.0),
        "miss_lookup_second_pass_us": percentiles(miss_second, 1000.0),
        "miss_lookup_peak_bytes": miss_peak,
        "schema_bytes": schema_bytes,
    });
    println!("{report}");
}

fn main() {
    let entries = env("CLOSURE_ENTRIES", 100_000);
    let lag = env("CLOSURE_LAG", 1_000);
    window("overflow", entries, None);
    window("bounded", entries, Some(lag));
}
