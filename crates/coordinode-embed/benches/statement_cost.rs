//! What a statement costs with the advisor recording it and without:
//! latency, the CPU of the thread running the statements and of the whole
//! process, and the process's I/O, for indexed point reads and for create,
//! update and delete rounds. `STATEMENT_COST_ADVISORY=off` runs with
//! recording off; `STATEMENT_COST_PARTS=write` (or `read`) runs one part.
//! The allocations of the same statements are counted by
//! `statement_cost_alloc`.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

#[path = "support/statement_cost.rs"]
mod statement_cost;

/// CPU seconds (utime + stime) in a `/proc` stat file, when it has them.
fn cpu_of(path: &str) -> Option<f64> {
    let stat = std::fs::read_to_string(path).ok()?;
    let rest = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let ticks = fields.get(11)?.parse::<u64>().ok()? + fields.get(12)?.parse::<u64>().ok()?;
    // USER_HZ is 100 on Linux.
    Some(ticks as f64 / 100.0)
}

/// The process's I/O counters from `/proc/self/io`: bytes written to
/// storage, bytes read from it, write calls.
fn io() -> Option<[u64; 3]> {
    let text = std::fs::read_to_string("/proc/self/io").ok()?;
    let field = |name: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(name)?.trim().parse::<u64>().ok())
    };
    Some([
        field("write_bytes:")?,
        field("read_bytes:")?,
        field("syscw:")?,
    ])
}

/// What the measured part spent: wall time, CPU of the statement thread
/// and of the whole process, and I/O counters.
#[derive(Default)]
struct Spent {
    wall: Duration,
    thread_cpu: Option<f64>,
    process_cpu: Option<f64>,
    io: Option<[u64; 3]>,
}

impl Spent {
    fn measure(&mut self, f: &mut dyn FnMut()) {
        let (thread0, process0, io0) = (
            cpu_of("/proc/thread-self/stat"),
            cpu_of("/proc/self/stat"),
            io(),
        );
        let started = Instant::now();
        f();
        self.wall = started.elapsed();
        self.thread_cpu = thread0
            .zip(cpu_of("/proc/thread-self/stat"))
            .map(|(a, b)| b - a);
        self.process_cpu = process0.zip(cpu_of("/proc/self/stat")).map(|(a, b)| b - a);
        self.io = io0
            .zip(io())
            .map(|(a, b)| [b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
    }

    fn report(&self, mode: &str, label: &str, times: Vec<Duration>) {
        let ops = times.len() as f64;
        let per_op = |c: Option<f64>| {
            c.map_or_else(|| "n/a".to_string(), |c| format!("{:.1}", c * 1e6 / ops))
        };
        let io = self.io.map_or_else(
            || "io n/a".to_string(),
            |[w, r, calls]| {
                format!(
                    "io write {:.0} B/op read {:.0} B/op {:.2} write calls/op",
                    w as f64 / ops,
                    r as f64 / ops,
                    calls as f64 / ops
                )
            },
        );
        println!(
            "advisory={mode} {label}: n={} {:.1} us/op cpu us/op thread {} process {} {io} {}",
            times.len(),
            self.wall.as_secs_f64() * 1e6 / ops,
            per_op(self.thread_cpu),
            per_op(self.process_cpu),
            statement_cost::percentiles(times)
        );
    }
}

fn main() {
    let advisory = std::env::var("STATEMENT_COST_ADVISORY").as_deref() != Ok("off");
    let parts = std::env::var("STATEMENT_COST_PARTS").unwrap_or_default();
    let runs = |part: &str| parts.is_empty() || parts == part;
    let mode = if advisory { "on" } else { "off" };
    let dir = tempfile::tempdir().expect("tempdir");
    println!("data dir {}", dir.path().display());
    let mut db = statement_cost::open(dir.path(), advisory);

    if runs("read") {
        let mut spent = Spent::default();
        let times = statement_cost::reads(&mut db, &mut |f| spent.measure(f));
        spent.report(mode, "read", times);
    }
    if runs("write") {
        let mut spent = Spent::default();
        let times = statement_cost::writes(&mut db, &mut |f| spent.measure(f));
        spent.report(mode, "create/update/delete", times);
    }
}
