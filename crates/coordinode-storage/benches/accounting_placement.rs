//! What the material accounting costs the write path, by where it lives.
//!
//! One scenario per process (`ACCOUNTING_AB_PATH`, `ACCOUNTING_AB_WORKLOAD`,
//! `ACCOUNTING_AB_OPS`, `ACCOUNTING_AB_WRITE_BUFFER`), so the peak resident
//! memory is the scenario's own.
//! Each commit writes a node row and an adjacency operand: two data trees,
//! as most graph writes touch. The workloads:
//!
//! - `insert`: a fresh node key every commit;
//! - `overwrite`: node keys cycling over 1024, operands onto 64 keys;
//! - `delete`: as many keys inserted first (not measured), then deleted.
//!
//! The paths: `volatile` (no journal, one direct batch per commit),
//! `journal` (the embedded journal, one journaled commit each) and
//! `local-apply` (proposals applied to the store as a Raft state machine
//! applies them, with markers; no consensus, no log, no network).
//!
//! The delete workload's keys are inserted, persisted and compacted before
//! the measured phase begins. Two costs are reported, each with its own
//! counters taken from the start of the measured phase:
//!
//! - `foreground`: the commits alone (latency percentiles, throughput, the
//!   bytes written and the compactions run while they ran);
//! - `full`: the commits, then their memtables persisted, then their
//!   compactions run to the end, which is when no tree owes a compaction
//!   under the strategy the engine compacts it with and none finished since
//!   that was read. A run that does not get there within
//!   `ACCOUNTING_AB_MAINTENANCE_SECS` (default 300) reports the debt left
//!   by tree instead of hiding it.
//!
//! Bytes written are the process's (Linux `/proc/self/io`); beside them the
//! logical bytes, every tree's bytes on disk, each tree's peak first level
//! runs and the peak resident memory (Linux `VmHWM`).
//!
//! Run: `cargo bench -p coordinode-storage --bench accounting_placement`.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use lsm_tree::AbstractTree;
use metrics::{CounterFn, Key, KeyName, Metadata, Recorder, SharedString, Unit};

/// Compactions by (tree, action).
type Counts = BTreeMap<(String, String), u64>;

/// Compactions counted by (tree, action), and when the last one finished.
#[derive(Default)]
struct Compactions(Mutex<(Counts, Option<Instant>)>);

impl Compactions {
    fn counts(&self) -> BTreeMap<(String, String), u64> {
        self.0.lock().map(|seen| seen.0.clone()).unwrap_or_default()
    }

    fn last_finished(&self) -> Option<Instant> {
        self.0.lock().ok().and_then(|seen| seen.1)
    }
}

struct Counted(Key, Arc<Compactions>);

impl CounterFn for Counted {
    fn increment(&self, by: u64) {
        let label = |name: &str| {
            self.0
                .labels()
                .find(|l| l.key() == name)
                .map_or_else(String::new, |l| l.value().to_owned())
        };
        if let Ok(mut seen) = self.1.0.lock() {
            *seen
                .0
                .entry((label("partition"), label("action")))
                .or_insert(0) += by;
            seen.1 = Some(Instant::now());
        }
    }
    fn absolute(&self, _: u64) {}
}

/// The compactions of `after` not yet in `before`, as `tree/action`.
fn since(
    before: &BTreeMap<(String, String), u64>,
    after: &BTreeMap<(String, String), u64>,
) -> BTreeMap<String, u64> {
    after
        .iter()
        .filter_map(|(key, n)| {
            let new = n - before.get(key).copied().unwrap_or(0);
            (new > 0).then(|| (format!("{}/{}", key.0, key.1), new))
        })
        .collect()
}

/// The bytes every tree's compactions still owe, by the strategy the engine
/// compacts them under (leveled, its default).
fn compaction_debt(engine: &StorageEngine) -> BTreeMap<&'static str, u64> {
    let strategy = lsm_tree::compaction::Leveled::default();
    Partition::all()
        .iter()
        .filter_map(|&part| Some((part.name(), engine.tree(part).ok()?)))
        .map(|(name, tree)| (name, tree.compaction_debt(&strategy)))
        .filter(|(_, debt)| *debt > 0)
        .collect()
}

/// Wait until the store's compactions are done: no tree owes a compaction,
/// and none finished since that was read (a compaction running owes the
/// bytes it reads until it finishes). Gives up at `limit`; returns the debt
/// left then by tree (empty when done).
fn maintenance_done(
    engine: &StorageEngine,
    compactions: &Compactions,
    limit: Duration,
) -> BTreeMap<&'static str, u64> {
    let deadline = Instant::now() + limit;
    loop {
        let finished = compactions.last_finished();
        let debt = compaction_debt(engine);
        if debt.is_empty() && compactions.last_finished() == finished {
            return debt;
        }
        if Instant::now() >= deadline {
            return debt;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

struct Capture(Arc<Compactions>);

impl Recorder for Capture {
    fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn register_counter(&self, key: &Key, _: &Metadata<'_>) -> metrics::Counter {
        if key.name() == "coordinode_storage_compaction_total" {
            metrics::Counter::from_arc(Arc::new(Counted(key.clone(), Arc::clone(&self.0))))
        } else {
            metrics::Counter::noop()
        }
    }
    fn register_gauge(&self, _: &Key, _: &Metadata<'_>) -> metrics::Gauge {
        metrics::Gauge::noop()
    }
    fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> metrics::Histogram {
        metrics::Histogram::noop()
    }
}

/// How commits reach the store.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Volatile,
    Journal,
    LocalApply,
}

impl Path {
    fn name(self) -> &'static str {
        match self {
            Self::Volatile => "volatile",
            Self::Journal => "journal",
            Self::LocalApply => "local-apply",
        }
    }
}

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

/// A line of `/proc/self/<file>` named `field`, in its own unit; 0 where
/// the file does not exist.
fn proc_field(file: &str, field: &str) -> u64 {
    std::fs::read_to_string(format!("/proc/self/{file}"))
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let rest = line.strip_prefix(field)?.trim_start_matches(':').trim();
                rest.split_whitespace().next()?.parse().ok()
            })
        })
        .unwrap_or(0)
}

/// Bytes every file under `dir` takes.
fn dir_bytes(dir: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.metadata() {
            Ok(meta) if meta.is_dir() => dir_bytes(&entry.path()),
            Ok(meta) => meta.len(),
            Err(_) => 0,
        })
        .sum()
}

fn unit(workload: &str, n: u64) -> Vec<Mutation> {
    let key = match workload {
        "overwrite" => n % 1024,
        _ => n,
    };
    let node = Mutation::Put {
        partition: PartitionId::Node,
        key: format!("node:00:{key:012}").into_bytes(),
        value: vec![(n % 251) as u8; 256],
    };
    let adj = Mutation::Merge {
        partition: PartitionId::Adj,
        key: format!("adj:V:out:{:04}", n % 64).into_bytes(),
        operand: coordinode_storage::engine::merge::encode_add(n),
    };
    vec![node, adj]
}

fn delete_unit(n: u64) -> Vec<Mutation> {
    vec![Mutation::Delete {
        partition: PartitionId::Node,
        key: format!("node:00:{n:012}").into_bytes(),
    }]
}

fn logical(unit: &[Mutation]) -> u64 {
    unit.iter()
        .map(|m| match m {
            Mutation::Put { key, value, .. } => (key.len() + value.len()) as u64,
            Mutation::Merge { key, operand, .. } => (key.len() + operand.len()) as u64,
            Mutation::Delete { key, .. } => key.len() as u64,
            _ => 0,
        })
        .sum()
}

struct Store {
    engine: StorageEngine,
    oracle: Arc<TimestampOracle>,
    path: Path,
    index: u64,
}

impl Store {
    fn commit(&mut self, unit: &[Mutation]) {
        match self.path {
            Path::Volatile => self.engine.apply_proposal_at(unit, 0).expect("apply"),
            Path::Journal => self
                .engine
                .commit_journaled(unit, self.oracle.next().as_raw())
                .expect("commit"),
            Path::LocalApply => {
                self.index += 1;
                self.engine
                    .apply_raft_proposal(unit, self.oracle.next().as_raw(), self.index, 0, |_| {
                        false
                    })
                    .expect("apply");
            }
        }
    }
}

fn main() {
    // How long the deferred maintenance may run before the debt left is
    // reported instead.
    let maintenance_limit = Duration::from_secs(
        env("ACCOUNTING_AB_MAINTENANCE_SECS", "300")
            .parse()
            .expect("maintenance seconds"),
    );
    let compactions = Arc::new(Compactions::default());
    metrics::set_global_recorder(Capture(Arc::clone(&compactions))).expect("recorder");

    let path = match env("ACCOUNTING_AB_PATH", "journal").as_str() {
        "volatile" => Path::Volatile,
        "local-apply" => Path::LocalApply,
        "journal" => Path::Journal,
        other => {
            eprintln!("ACCOUNTING_AB_PATH {other}: volatile, journal or local-apply");
            std::process::exit(2);
        }
    };
    let workload = env("ACCOUNTING_AB_WORKLOAD", "insert");
    let ops: u64 = env("ACCOUNTING_AB_OPS", "20000").parse().expect("ops");

    let dir = tempfile::tempdir().expect("tempdir");
    // The nonpersistent store lives in memory: no journal, no file of it
    // on disk, so its write bytes are not the process's I/O.
    let mut config = if path == Path::Volatile {
        StorageConfig::with_endpoints_no_persistence(vec![EndpointConfig::new(
            "memory",
            std::path::PathBuf::from("/accounting-ab-memory"),
            Media::Ram,
            Durability::Volatile,
            Tier::Memory,
        )])
        .with_fs(Arc::new(lsm_tree::fs::MemFs::new()) as Arc<dyn lsm_tree::fs::Fs>)
    } else {
        StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Ssd,
            Durability::Durable,
            Tier::Warm,
        )])
    };
    // A small shared write buffer, the same on every side compared, so the
    // run flushes and compacts instead of fitting in its memtables.
    config.max_write_buffer_bytes = env("ACCOUNTING_AB_WRITE_BUFFER", "1048576")
        .parse()
        .expect("write buffer bytes");
    let oracle = Arc::new(TimestampOracle::new());
    let engine = match path {
        Path::Journal => StorageEngine::open_embedded(&config, Arc::clone(&oracle)),
        Path::LocalApply => StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)),
        Path::Volatile => StorageEngine::open(&config),
    }
    .expect("open");
    let mut store = Store {
        engine,
        oracle,
        path,
        index: 0,
    };

    // The keys to delete, with their maintenance done before the measured
    // phase, so none of it lands in that phase's counters; a debt the
    // engine leaves standing is reported.
    let mut preload_debt_left = BTreeMap::new();
    if workload == "delete" {
        for n in 0..ops {
            store.commit(&unit("insert", n));
        }
        store.engine.persist().expect("persist");
        preload_debt_left = maintenance_done(&store.engine, &compactions, maintenance_limit);
    }

    let trees: Vec<Partition> = Partition::all().to_vec();
    let mut l0_peak: BTreeMap<&'static str, usize> = BTreeMap::new();
    let compactions_before = compactions.counts();
    let written_before = proc_field("io", "write_bytes");
    let mut latencies = Vec::with_capacity(ops as usize);
    let mut logical_bytes = 0u64;
    let started = Instant::now();
    for n in 0..ops {
        let unit = if workload == "delete" {
            delete_unit(n)
        } else {
            unit(&workload, n)
        };
        logical_bytes += logical(&unit);
        let at = Instant::now();
        store.commit(&unit);
        latencies.push(at.elapsed().as_nanos() as u64);
        if n % 64 == 0 {
            for &part in &trees {
                if let Ok(tree) = store.engine.tree(part) {
                    let runs = tree.l0_run_count();
                    let peak = l0_peak.entry(part.name()).or_insert(0);
                    *peak = (*peak).max(runs);
                }
            }
        }
    }
    // The foreground: the commits alone, and what they wrote and compacted
    // while they ran.
    let foreground = started.elapsed();
    let foreground_written = proc_field("io", "write_bytes") - written_before;
    let foreground_compactions = since(&compactions_before, &compactions.counts());

    // The full cost adds the deferred work the commits left: their
    // memtables persisted, then their compactions run to the end.
    let persisted = Instant::now();
    store.engine.persist().expect("persist");
    let persist = persisted.elapsed();
    let debt_left = maintenance_done(&store.engine, &compactions, maintenance_limit);
    let full = started.elapsed();
    let maintenance = full - foreground - persist;
    // The counter only grows over the process's life.
    let written = proc_field("io", "write_bytes") - written_before;

    latencies.sort_unstable();
    let pct = |p: f64| {
        let at = ((latencies.len() as f64 - 1.0) * p).round() as usize;
        latencies.get(at).copied().unwrap_or(0) as f64 / 1000.0
    };
    let mut tree_bytes = BTreeMap::new();
    for &part in &trees {
        if let Ok(tree) = store.engine.tree(part) {
            tree_bytes.insert(part.name(), tree.disk_space());
        }
    }
    let dir_total = dir_bytes(dir.path());
    let amplification = |bytes: u64| {
        if logical_bytes == 0 {
            0.0
        } else {
            bytes as f64 / logical_bytes as f64
        }
    };
    let report = serde_json::json!({
        "path": path.name(),
        "workload": workload,
        "ops": ops,
        "logical_bytes": logical_bytes,
        "foreground": {
            "seconds": foreground.as_secs_f64(),
            "ops_per_sec": ops as f64 / foreground.as_secs_f64(),
            "latency_us": {"p50": pct(0.5), "p99": pct(0.99), "p999": pct(0.999)},
            "written_bytes": foreground_written,
            "write_amplification": amplification(foreground_written),
            "compactions": foreground_compactions,
        },
        "full": {
            "seconds": full.as_secs_f64(),
            "ops_per_sec": ops as f64 / full.as_secs_f64(),
            "persist_seconds": persist.as_secs_f64(),
            "maintenance_seconds": maintenance.as_secs_f64(),
            "written_bytes": written,
            "write_amplification": amplification(written),
            "compactions": since(&compactions_before, &compactions.counts()),
            "compaction_debt_left_bytes": debt_left,
        },
        "preload_compaction_debt_left_bytes": preload_debt_left,
        "tree_bytes": tree_bytes,
        "dir_bytes": dir_total,
        "l0_peak": l0_peak,
        "peak_rss_kb": proc_field("status", "VmHWM"),
    });
    println!("{report}");
}
