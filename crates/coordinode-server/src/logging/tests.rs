#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::io::Write;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use super::*;

/// A sink stuck until the test lets it go: a stdout nobody reads.
struct Stuck(Arc<Barrier>);

impl Write for Stuck {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.wait();
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A logging thread never waits on the sink: with the writer stuck, lines
/// past the queue are dropped and counted, and each write returns at once.
/// Writing straight to stdout instead made every request that logged wait
/// on the stdout lock and on whatever consumed stdout.
#[test]
fn a_stuck_sink_drops_lines_instead_of_blocking_the_logger() {
    let release = Arc::new(Barrier::new(2));
    let (mut writer, guard) = queued_writer(Stuck(Arc::clone(&release)), 4);
    let dropped = writer.error_counter();

    let started = Instant::now();
    for i in 0..64 {
        writer.write_all(format!("line {i}\n").as_bytes()).unwrap();
    }
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "writing to a stuck sink blocked the logger"
    );
    // At most the queue and the line being written are kept.
    assert!(
        dropped.dropped_lines() >= 64 - 4 - 1,
        "dropped {} lines",
        dropped.dropped_lines()
    );

    // Unstick the sink so the writer thread can finish on drop.
    std::thread::spawn(move || {
        for _ in 0..8 {
            release.wait();
        }
    });
    drop(guard);
}
