//! The allocations of the statements `statement_cost` times: bytes and
//! allocations per statement. Counting changes every allocation's cost, so
//! this binary reports no times.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::alloc::System;

use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/statement_cost.rs"]
mod statement_cost;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn main() {
    let advisory = std::env::var("STATEMENT_COST_ADVISORY").as_deref() != Ok("off");
    let mode = if advisory { "on" } else { "off" };
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = statement_cost::open(dir.path(), advisory);
    let counted = |label: &str, ops: usize, stats: stats_alloc::Stats| {
        println!(
            "advisory={mode} {label}: {:.0} B/op, {:.1} allocations/op",
            stats.bytes_allocated as f64 / ops as f64,
            stats.allocations as f64 / ops as f64,
        );
    };
    let mut stats = None;
    let times = statement_cost::reads(&mut db, &mut |f| {
        let region = Region::new(GLOBAL);
        f();
        stats = Some(region.change());
    });
    counted("read", times.len(), stats.take().expect("measured"));
    let times = statement_cost::writes(&mut db, &mut |f| {
        let region = Region::new(GLOBAL);
        f();
        stats = Some(region.change());
    });
    counted(
        "create/update/delete",
        times.len(),
        stats.expect("measured"),
    );
}
