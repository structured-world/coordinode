//! What a statement costs with the advisor recording it and without:
//! latency and process CPU of indexed point reads and of create, update and
//! delete rounds. `STATEMENT_COST_ADVISORY=off` runs with recording off.
//! The allocations of the same statements are counted by
//! `statement_cost_alloc`.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::time::{Duration, Instant};

#[path = "support/statement_cost.rs"]
mod statement_cost;

/// CPU seconds of the whole process, when `/proc` has them.
fn cpu() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let ticks = fields.get(11)?.parse::<u64>().ok()? + fields.get(12)?.parse::<u64>().ok()?;
    // USER_HZ is 100 on Linux.
    Some(ticks as f64 / 100.0)
}

/// The wall time and process CPU of the measured part, around which a
/// scenario calls this.
#[derive(Default)]
struct Spent {
    wall: Duration,
    cpu: Option<f64>,
}

impl Spent {
    fn measure(&mut self, f: &mut dyn FnMut()) {
        let cpu0 = cpu();
        let started = Instant::now();
        f();
        self.wall = started.elapsed();
        self.cpu = cpu0.zip(cpu()).map(|(a, b)| b - a);
    }

    fn report(&self, mode: &str, label: &str, times: Vec<Duration>) {
        let ops = times.len() as f64;
        let cpu = self.cpu.map_or_else(
            || "cpu n/a".to_string(),
            |c| format!("{:.1} us cpu/op", c * 1e6 / ops),
        );
        println!(
            "advisory={mode} {label}: n={} {:.1} us/op {cpu} {}",
            times.len(),
            self.wall.as_secs_f64() * 1e6 / ops,
            statement_cost::percentiles(times)
        );
    }
}

fn main() {
    let advisory = std::env::var("STATEMENT_COST_ADVISORY").as_deref() != Ok("off");
    let mode = if advisory { "on" } else { "off" };
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = statement_cost::open(dir.path(), advisory);

    let mut spent = Spent::default();
    let times = statement_cost::reads(&mut db, &mut |f| spent.measure(f));
    spent.report(mode, "read", times);

    let mut spent = Spent::default();
    let times = statement_cost::writes(&mut db, &mut |f| spent.measure(f));
    spent.report(mode, "create/update/delete", times);
}
