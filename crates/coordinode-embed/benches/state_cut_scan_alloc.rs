//! The allocations of the copies `state_cut_scan` times: bytes and
//! allocations per row. Counting changes every allocation's cost, so this
//! binary reports no times.
#![allow(clippy::expect_used, clippy::print_stdout)]

use std::alloc::System;

use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/cut_scan.rs"]
mod cut_scan;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn main() {
    for (rows, value) in cut_scan::CASES {
        let mut stats = None;
        let m = cut_scan::case(rows, value, &mut |copy| {
            let region = Region::new(GLOBAL);
            copy();
            stats = Some(region.change());
        });
        let stats = stats.expect("measured");
        let copied = m.rows as f64;
        println!(
            "rows={rows} value={value}B: allocated {:.1} B/row, {:.3} allocations/row",
            stats.bytes_allocated as f64 / copied,
            stats.allocations as f64 / copied,
        );
    }
}
