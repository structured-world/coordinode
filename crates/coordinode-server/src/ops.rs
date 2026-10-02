//! Operational HTTP server on port 7084.
//!
//! Serves operational endpoints directly (not through proto transcoding):
//! - GET /health: liveness check (process alive)
//! - GET /ready: readiness check, 200 while the gRPC server serves, 503
//!   before it starts and from the moment shutdown begins
//! - GET /metrics: Prometheus OpenMetrics
//!
//! # Cluster-ready notes
//! - Each CE node (3-node HA) has its own :7084.
//! - K8s probes: liveness → /health, readiness → /ready.
//! - Metrics are per-node (Prometheus scrapes each node independently).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use metrics_exporter_prometheus::PrometheusBuilder;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tracing::{error, info};

/// Whether the node serves requests: raised when the gRPC server starts,
/// lowered when shutdown begins, so a balancer stops routing to a draining
/// node before its connections close.
#[derive(Clone, Debug, Default)]
pub(crate) struct Readiness(Arc<AtomicBool>);

impl Readiness {
    pub(crate) fn set(&self, ready: bool) {
        self.0.store(ready, Ordering::Release);
    }

    pub(crate) fn get(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Status line, content type and body for `path`; `metrics` renders the
/// Prometheus text only when asked for.
fn respond(
    path: &str,
    ready: bool,
    metrics: impl FnOnce() -> String,
) -> (&'static str, &'static str, String) {
    match path {
        "/health" => (
            "200 OK",
            "application/json",
            r#"{"status":"ok"}"#.to_string(),
        ),
        "/ready" if ready => (
            "200 OK",
            "application/json",
            r#"{"ready":true}"#.to_string(),
        ),
        "/ready" => (
            "503 Service Unavailable",
            "application/json",
            r#"{"ready":false}"#.to_string(),
        ),
        "/metrics" => ("200 OK", "text/plain; charset=utf-8", metrics()),
        _ => (
            "404 Not Found",
            "application/json",
            r#"{"error":"not found"}"#.to_string(),
        ),
    }
}

/// Sets the gauges whose value is only meaningful when sampled, right before
/// `/metrics` renders them: a scrape samples them, and nothing samples them
/// between scrapes.
pub(crate) type SampleGauges = Arc<dyn Fn() + Send + Sync>;

/// Start the operational HTTP server on `listener`, bound by the caller at
/// startup so a busy port fails the start.
///
/// Handles /health, /ready, /metrics; `/ready` follows `readiness`, and
/// `/metrics` runs `sample` first. Runs until the process exits.
pub(crate) async fn start_ops_server(
    listener: TcpListener,
    readiness: Readiness,
    sample: SampleGauges,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Install Prometheus metrics recorder
    let prometheus_handle = PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| format!("failed to install Prometheus recorder: {e}"))?;

    // Register all CE metric families
    crate::metrics_catalog::register_all_metrics();

    if let Ok(addr) = listener.local_addr() {
        info!(%addr, "operational HTTP server listening");
    }

    loop {
        let (mut stream, _peer) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                error!("accept error: {e}");
                continue;
            }
        };

        let handle = prometheus_handle.clone();
        let sample = Arc::clone(&sample);
        let ready = readiness.get();

        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            let n = match tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await {
                Ok(n) => n,
                Err(_) => return,
            };

            let request = String::from_utf8_lossy(&buf[..n]);

            // Parse the HTTP request line
            let path = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("/");

            let (status, content_type, body) = respond(path, ready, || {
                sample();
                handle.render()
            });

            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );

            let _ = stream.write_all(response.as_bytes()).await;
        });
    }
}

#[cfg(test)]
mod tests;
