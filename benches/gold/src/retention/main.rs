//! Retained-history sizing binary. For each write pattern and window
//! length runs a fresh tempdir engine on a simulated clock and prints the
//! live bytes, the bytes held only by retention history, and their ratio.
//! Pass `--full` for the sizing preset.

use coordinode_gold_bench::retention::{Pattern, RunResult, Shape, run};
use tempfile::TempDir;

fn main() {
    let shape = if std::env::args().any(|a| a == "--full") {
        Shape::FULL
    } else {
        Shape::CI
    };
    eprintln!(
        "gold-retention: keys={} rounds={} value_bytes={}",
        shape.keys, shape.rounds, shape.value_bytes,
    );

    for pattern in Pattern::ALL {
        let mut results = Vec::new();
        for window in shape.windows() {
            let dir = TempDir::new().expect("tempdir");
            results.push(run(pattern, shape, window, dir.path()));
        }
        print_pattern(&results);
    }
}

fn print_pattern(results: &[RunResult]) {
    let Some(first) = results.first() else {
        return;
    };
    println!(
        "\n{} (keys={} rounds={} value_bytes={} written={:.1} MiB per run)",
        first.pattern.name(),
        first.shape.keys,
        first.shape.rounds,
        first.shape.value_bytes,
        mib(first.bytes_written),
    );
    println!(
        "  {:>14}  {:>12}  {:>14}  {:>10}  {:>16}",
        "window rounds", "live MiB", "retained MiB", "ret/live", "ret/written"
    );
    for r in results {
        let h = r.history;
        println!(
            "  {:>14}  {:>12.2}  {:>14.2}  {:>10.2}  {:>16.2}",
            r.window_rounds,
            mib(h.live_bytes),
            mib(h.retained_bytes),
            h.retained_ratio(),
            r.retained_per_written(),
        );
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
