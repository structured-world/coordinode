//! The whole cost of ordinary writes at the default durability: their
//! latency, the CPU and allocation of the writing phase, then the CPU, wall
//! time and allocation of bringing the store to rest (memtables flushed,
//! every partition fully compacted, the journal reclaimed), and what the store
//! holds on disk afterwards, temporary files included.
//!
//! CPU is read from `/proc` (Linux): the writing thread's own time, and the
//! rest of the process (flush, compaction and journal workers). Elsewhere
//! only wall times, allocation and disk are reported.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::alloc::System;
use std::path::Path;
use std::time::{Duration, Instant};

use coordinode_embed::Database;
use coordinode_storage::engine::partition::Partition;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Writes before the measured ones, so caches and dictionaries settle.
const WARM: usize = 500;
/// Measured writes; `WRITE_COST_WRITES` overrides.
const WRITES: usize = 20_000;

/// CPU ticks of the whole process and of the calling thread, when `/proc`
/// has them.
fn cpu_ticks() -> Option<(u64, u64)> {
    let ticks = |stat: String| -> Option<u64> {
        // Fields after the parenthesised name: utime and stime are the 12th
        // and 13th of them.
        let rest = stat.rsplit_once(')')?.1;
        let fields: Vec<&str> = rest.split_whitespace().collect();
        Some(fields.get(11)?.parse::<u64>().ok()? + fields.get(12)?.parse::<u64>().ok()?)
    };
    let process = ticks(std::fs::read_to_string("/proc/self/stat").ok()?)?;
    let thread = ticks(std::fs::read_to_string("/proc/thread-self/stat").ok()?)?;
    Some((process, thread))
}

/// Ticks per second of `/proc` CPU counters (USER_HZ, 100 on Linux).
const TICKS_PER_SEC: f64 = 100.0;

fn cpu_line(before: Option<(u64, u64)>, after: Option<(u64, u64)>) -> String {
    match (before, after) {
        (Some((p0, t0)), Some((p1, t1))) => {
            let own = (t1 - t0) as f64 / TICKS_PER_SEC;
            let rest = ((p1 - p0) - (t1 - t0)) as f64 / TICKS_PER_SEC;
            format!("cpu writer={own:.2}s background={rest:.2}s")
        }
        _ => "cpu n/a".to_string(),
    }
}

/// Bytes and files under `dir`, and the files a finished store should not
/// hold: temporaries and partial copies.
fn disk(dir: &Path) -> (u64, usize, Vec<String>) {
    let (mut bytes, mut files, mut temporary) = (0, 0, Vec::new());
    let mut pending = vec![dir.to_path_buf()];
    while let Some(d) = pending.pop() {
        for entry in std::fs::read_dir(&d).expect("read dir") {
            let entry = entry.expect("entry");
            let meta = entry.metadata().expect("metadata");
            if meta.is_dir() {
                pending.push(entry.path());
                continue;
            }
            bytes += meta.len();
            files += 1;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.ends_with(".tmp") || name.starts_with("tmp") || name.contains(".partial") {
                temporary.push(entry.path().display().to_string());
            }
        }
    }
    (bytes, files, temporary)
}

fn percentiles(mut times: Vec<Duration>) -> String {
    times.sort_unstable();
    let at = |q: usize| times[(times.len() - 1) * q / 1000];
    format!(
        "p50={:?} p99={:?} p999={:?} max={:?}",
        at(500),
        at(990),
        at(999),
        at(1000)
    )
}

fn main() {
    let writes = std::env::var("WRITE_COST_WRITES")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(WRITES);
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for i in 0..WARM {
        db.execute_cypher(&format!("CREATE (:Warm {{n: {i}, s: 'w{i}'}})"))
            .expect("warm write");
    }
    db.persist().expect("persist");

    let mut times = Vec::with_capacity(writes);
    let region = Region::new(GLOBAL);
    let cpu0 = cpu_ticks();
    let start = Instant::now();
    for i in 0..writes {
        let t = Instant::now();
        db.execute_cypher(&format!(
            "CREATE (:User {{n: {i}, name: 'user-{i}', score: {}}})",
            i % 97
        ))
        .expect("write");
        times.push(t.elapsed());
    }
    let wall = start.elapsed();
    let cpu1 = cpu_ticks();
    let stats = region.change();
    println!(
        "writes: n={writes} wall={wall:?} {} {} allocated={} allocations={}",
        percentiles(times),
        cpu_line(cpu0, cpu1),
        stats.bytes_allocated,
        stats.allocations
    );

    let region = Region::new(GLOBAL);
    let start = Instant::now();
    db.persist().expect("persist");
    let mut unopened = 0;
    for &part in Partition::all() {
        if db.engine().force_compaction(part).is_err() {
            unopened += 1;
        }
    }
    db.engine().reclaim_footprint();
    let wall = start.elapsed();
    let cpu2 = cpu_ticks();
    let stats = region.change();
    println!(
        "to rest: wall={wall:?} {} allocated={} allocations={} partitions_not_compacted={unopened}",
        cpu_line(cpu1, cpu2),
        stats.bytes_allocated,
        stats.allocations
    );

    drop(db);
    let (bytes, files, temporary) = disk(dir.path());
    println!(
        "on disk after close: bytes={bytes} files={files} temporary={}",
        temporary.len()
    );
    for path in temporary {
        println!("  temporary: {path}");
    }
}
