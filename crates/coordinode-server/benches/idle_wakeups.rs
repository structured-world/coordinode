//! Idle wakeups of a running server: context switches per second of a
//! `coordinode serve` process that is asked nothing.
//!
//! Prints `idle_wakeups_per_sec=<n>` and the busiest threads. Run with
//! `cargo bench -p coordinode-server --bench idle_wakeups`; the window is
//! `IDLE_WINDOW_SECS` (default 30). Linux and macOS expose another process's
//! context switches; elsewhere the bench says so and measures nothing.

#![allow(clippy::expect_used, clippy::print_stdout)]

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[path = "../tests/support/idle.rs"]
mod idle;

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn main() {
    use std::time::Duration;

    let window = std::env::var("IDLE_WINDOW_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(30);
    let data = tempfile::tempdir().expect("data dir");
    let sample = idle::measure(
        std::path::Path::new(env!("CARGO_BIN_EXE_coordinode")),
        data.path(),
        Duration::from_secs(5),
        Duration::from_secs(window),
    );
    println!(
        "idle_wakeups_per_sec={:.2} switches={} window_secs={:.1}",
        sample.per_sec(),
        sample.switches,
        sample.window.as_secs_f64()
    );
    for (thread, n) in sample.threads.iter().take(15) {
        println!("  {thread}: {n}");
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn main() {
    println!("idle wakeups: not measurable on this OS (no per-process context switch counter)");
}
