//! What a statement costs with the advisor recording it and without:
//! latency and process CPU of indexed point reads and of create, update and
//! delete rounds. `STATEMENT_COST_ADVISORY=off` runs with recording off.
//! The allocations of the same statements are counted by
//! `statement_cost_alloc`.
#![allow(clippy::expect_used, clippy::print_stdout)]

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

fn main() {
    let advisory = std::env::var("STATEMENT_COST_ADVISORY").as_deref() != Ok("off");
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = statement_cost::open(dir.path(), advisory);
    let mode = if advisory { "on" } else { "off" };
    let mut run = |label: &str,
                   times: &mut dyn FnMut(
        &mut dyn FnMut(&mut dyn FnMut()),
    ) -> Vec<std::time::Duration>| {
        let mut wall = std::time::Duration::ZERO;
        let (mut cpu0, mut cpu1) = (None, None);
        let times = times(&mut |f| {
            cpu0 = cpu();
            let started = std::time::Instant::now();
            f();
            wall = started.elapsed();
            cpu1 = cpu();
        });
        let ops = times.len() as f64;
        let cpu = match (cpu0, cpu1) {
            (Some(a), Some(b)) => format!("{:.1} us cpu/op", (b - a) * 1e6 / ops),
            _ => "cpu n/a".to_string(),
        };
        println!(
            "advisory={mode} {label}: n={} {:.1} us/op {cpu} {}",
            times.len(),
            wall.as_secs_f64() * 1e6 / ops,
            statement_cost::percentiles(times)
        );
    };
    run("read", &mut |around| statement_cost::reads(&mut db, around));
    run("create/update/delete", &mut |around| {
        statement_cost::writes(&mut db, around)
    });
}
