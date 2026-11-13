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
//! `journal` (the embedded journal, one journaled commit each) and `raft`
//! (proposals applied as a Raft state machine applies them, with markers).
//!
//! After the writes the store is persisted and its compactions drained, so
//! deferred maintenance is counted, not hidden. One JSON line reports the
//! commit latency percentiles, throughput, bytes the process wrote to
//! storage (Linux `/proc/self/io`), logical bytes written, every tree's
//! bytes on disk, compactions by tree and action, each tree's peak first
//! level runs, and the peak resident memory (Linux `VmHWM`).
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

/// Compactions counted by (tree, action).
#[derive(Default)]
struct Compactions(Mutex<BTreeMap<(String, String), u64>>);

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
                .entry((label("partition"), label("action")))
                .or_insert(0) += by;
        }
    }
    fn absolute(&self, _: u64) {}
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
    Raft,
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
            Path::Raft => {
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
    let compactions = Arc::new(Compactions::default());
    metrics::set_global_recorder(Capture(Arc::clone(&compactions))).expect("recorder");

    let path = match env("ACCOUNTING_AB_PATH", "journal").as_str() {
        "volatile" => Path::Volatile,
        "raft" => Path::Raft,
        _ => Path::Journal,
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
        Path::Raft => StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)),
        Path::Volatile => StorageEngine::open(&config),
    }
    .expect("open");
    let mut store = Store {
        engine,
        oracle,
        path,
        index: 0,
    };

    if workload == "delete" {
        for n in 0..ops {
            store.commit(&unit("insert", n));
        }
        store.engine.persist().expect("persist");
    }

    let trees: Vec<Partition> = Partition::all().to_vec();
    let mut l0_peak: BTreeMap<&'static str, usize> = BTreeMap::new();
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
    let elapsed = started.elapsed();

    // Deferred maintenance is the writes' cost too: persist, then wait for
    // the compactions to stop.
    store.engine.persist().expect("persist");
    let total = |c: &Compactions| c.0.lock().map(|m| m.values().sum::<u64>()).unwrap_or(0);
    let mut last = total(&compactions);
    let mut quiet = Instant::now();
    let deadline = Instant::now() + Duration::from_secs(120);
    while quiet.elapsed() < Duration::from_secs(3) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
        let now = total(&compactions);
        if now != last {
            last = now;
            quiet = Instant::now();
        }
    }
    let drained = Instant::now() < deadline;
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
    let compactions: BTreeMap<String, u64> = compactions
        .0
        .lock()
        .map(|m| {
            m.iter()
                .map(|((tree, action), n)| (format!("{tree}/{action}"), *n))
                .collect()
        })
        .unwrap_or_default();
    let report = serde_json::json!({
        "path": env("ACCOUNTING_AB_PATH", "journal"),
        "workload": workload,
        "ops": ops,
        "seconds": elapsed.as_secs_f64(),
        "ops_per_sec": ops as f64 / elapsed.as_secs_f64(),
        "latency_us": {"p50": pct(0.5), "p99": pct(0.99), "p999": pct(0.999)},
        "logical_bytes": logical_bytes,
        "written_bytes": written,
        "write_amplification": if logical_bytes == 0 { 0.0 } else { written as f64 / logical_bytes as f64 },
        "tree_bytes": tree_bytes,
        "dir_bytes": dir_total,
        "compactions": compactions,
        "l0_peak": l0_peak,
        "maintenance_drained": drained,
        "peak_rss_kb": proc_field("status", "VmHWM"),
    });
    println!("{report}");
}
