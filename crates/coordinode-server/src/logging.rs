//! Structured logging configuration.
//!
//! Supports two output formats:
//! - `text` (default for development) — human-readable tracing output
//! - `json` (production) — structured JSON logs for log aggregation
//!
//! Format is selected via `COORDINODE_LOG_FORMAT` env var (`text` or `json`).
//! Log level is controlled via `RUST_LOG` env var (default: `info`).
//!
//! Lines are written to stdout by a dedicated thread, never by the thread
//! that logs: a request that logs (a slow statement, say) does not wait on
//! the stdout lock or on whatever consumes stdout. When that thread falls
//! behind by [`LOG_QUEUE_LINES`] lines, further lines are dropped and counted
//! ([`LogGuard::dropped`]) rather than stalling the server.
//!
//! # Cluster-ready notes
//! - Each CE node logs independently; there is no trace export (OTLP) or
//!   cross-node trace-context propagation.

use tracing_appender::non_blocking::{ErrorCounter, NonBlockingBuilder, WorkerGuard};
use tracing_subscriber::{EnvFilter, fmt};

/// Lines the writer thread may fall behind by before new ones are dropped.
pub const LOG_QUEUE_LINES: usize = 128 * 1024;

/// Keeps the log writer thread alive; dropping it flushes the lines still
/// queued. Held for as long as the process logs.
#[must_use = "dropping the guard stops the log writer"]
pub struct LogGuard {
    _worker: WorkerGuard,
    dropped: ErrorCounter,
}

impl LogGuard {
    /// Lines dropped so far because the writer thread had fallen behind.
    pub fn dropped(&self) -> DroppedLines {
        DroppedLines(self.dropped.clone())
    }
}

/// A handle that reads how many log lines were dropped, for metrics.
#[derive(Clone)]
pub struct DroppedLines(ErrorCounter);

impl DroppedLines {
    /// Lines dropped so far.
    pub fn get(&self) -> usize {
        self.0.dropped_lines()
    }
}

/// A writer that hands lines to a thread writing them to `sink`, dropping
/// (and counting) what arrives while `lines` are already queued.
fn queued_writer<W: std::io::Write + Send + 'static>(
    sink: W,
    lines: usize,
) -> (tracing_appender::non_blocking::NonBlocking, WorkerGuard) {
    NonBlockingBuilder::default()
        .buffered_lines_limit(lines)
        .lossy(true)
        .thread_name("log-writer")
        .finish(sink)
}

/// Initialize the tracing subscriber based on environment configuration.
///
/// Environment variables:
/// - `RUST_LOG` — log level filter (default: `info`)
/// - `COORDINODE_LOG_FORMAT` — `text` (default) or `json`
pub fn init_logging() -> LogGuard {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let format = std::env::var("COORDINODE_LOG_FORMAT").unwrap_or_else(|_| "text".to_string());

    let (writer, worker) = queued_writer(std::io::stdout(), LOG_QUEUE_LINES);
    let dropped = writer.error_counter();

    match format.as_str() {
        "json" => {
            let subscriber = fmt::Subscriber::builder()
                .json()
                .with_writer(writer)
                .with_env_filter(filter)
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .with_span_list(true)
                .finish();

            if let Err(e) = tracing::subscriber::set_global_default(subscriber) {
                eprintln!("failed to set JSON tracing subscriber: {e}");
            }
        }
        _ => {
            // Default: human-readable text format
            let subscriber = fmt::Subscriber::builder()
                .with_writer(writer)
                .with_env_filter(filter)
                .with_target(true)
                .finish();

            if let Err(e) = tracing::subscriber::set_global_default(subscriber) {
                eprintln!("failed to set text tracing subscriber: {e}");
            }
        }
    }
    LogGuard {
        _worker: worker,
        dropped,
    }
}

#[cfg(test)]
mod tests;
