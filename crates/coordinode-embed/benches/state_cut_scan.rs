//! Copying a state cut's view page by page, with the default allocator:
//! time per row and per page for small and for large rows. The allocations
//! of the same copies are counted by `state_cut_scan_alloc`.
#![allow(clippy::expect_used, clippy::print_stdout)]

#[path = "support/cut_scan.rs"]
mod cut_scan;

fn main() {
    for (rows, value) in cut_scan::CASES {
        let m = cut_scan::case(rows, value, &mut |copy| copy());
        let at = |q: usize| m.pages[(m.pages.len() - 1) * q / 1000];
        println!(
            "rows={rows} value={value}B: {:.1} ns/row, page p50={:?} p99={:?}",
            m.wall.as_nanos() as f64 / m.rows as f64,
            at(500),
            at(990),
        );
    }
}
