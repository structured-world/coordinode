//! Structured logging configuration.
//!
//! Supports two output formats:
//! - `text` (default for development) — human-readable tracing output
//! - `json` (production) — structured JSON logs for log aggregation
//!
//! Format is selected via `COORDINODE_LOG_FORMAT` env var (`text` or `json`).
//! Log level is controlled via `RUST_LOG` env var (default: `info`).
//!
//! # Cluster-ready notes
//! - Each CE node logs independently.
//! - W3C traceparent propagation enables distributed tracing across nodes.
//! - OTLP export configured via `OTEL_EXPORTER_OTLP_ENDPOINT` (Phase 2+).

use tracing_subscriber::{EnvFilter, fmt};

/// Initialize the tracing subscriber based on environment configuration.
///
/// Environment variables:
/// - `RUST_LOG` — log level filter (default: `info`)
/// - `COORDINODE_LOG_FORMAT` — `text` (default) or `json`
pub fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let format = std::env::var("COORDINODE_LOG_FORMAT").unwrap_or_else(|_| "text".to_string());

    match format.as_str() {
        "json" => {
            let subscriber = fmt::Subscriber::builder()
                .json()
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
                .with_env_filter(filter)
                .with_target(true)
                .finish();

            if let Err(e) = tracing::subscriber::set_global_default(subscriber) {
                eprintln!("failed to set text tracing subscriber: {e}");
            }
        }
    }
}
