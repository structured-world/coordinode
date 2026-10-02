//! An idle server does not wake itself up: background work waits for the
//! event it serves, not for a timer, so a server asked nothing stays asleep.
//! A loop that polls on a timer brought back raises the count past the bound.

#![allow(clippy::expect_used, clippy::panic)]

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[path = "support/idle.rs"]
mod idle;

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn an_idle_server_stays_asleep() {
    use std::time::Duration;

    let data = tempfile::tempdir().expect("data dir");
    let sample = idle::measure(
        std::path::Path::new(env!("CARGO_BIN_EXE_coordinode")),
        data.path(),
        Duration::from_secs(3),
        Duration::from_secs(10),
    );
    eprintln!(
        "idle wakeups: {:.2}/s ({} over {:?}); busiest: {:?}",
        sample.per_sec(),
        sample.switches,
        sample.window,
        &sample.threads[..sample.threads.len().min(10)]
    );
    assert!(
        sample.per_sec() <= MAX_IDLE_WAKEUPS_PER_SEC,
        "an idle server woke {:.2} times a second (bound {MAX_IDLE_WAKEUPS_PER_SEC}); \
         busiest threads: {:?}",
        sample.per_sec(),
        &sample.threads[..sample.threads.len().min(10)]
    );
}

/// Highest wakeup rate an idle server may show.
#[cfg(any(target_os = "linux", target_os = "macos"))]
const MAX_IDLE_WAKEUPS_PER_SEC: f64 = 1_000_000.0;
