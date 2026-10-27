//! Operational HTTP server on port 7084.
//!
//! Serves operational endpoints directly (not through proto transcoding):
//! - GET /health: liveness check (process alive)
//! - GET /ready: readiness check, 200 while the gRPC server serves, 503
//!   before it starts and from the moment shutdown begins
//! - GET /metrics: Prometheus OpenMetrics
//! - GET /version: this member's version report (its pair, its group's, why
//!   it is read-only if it is, each voter's pair, the current write pause)
//! - POST /reload: read the node's credentials and trust again
//! - GET /blob-gc: where orphan blob collection stands
//! - POST /blob-gc/run[?reconcile=true]: run a collection pass now
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
/// node before its connections close; and down for good once consensus has
/// stopped on a fatal error, since the node then commits nothing.
#[derive(Clone, Debug, Default)]
pub(crate) struct Readiness {
    serving: Arc<AtomicBool>,
    consensus_failed: Arc<AtomicBool>,
}

impl Readiness {
    pub(crate) fn set(&self, ready: bool) {
        self.serving.store(ready, Ordering::Release);
    }

    /// Consensus stopped on a fatal error: not ready, whatever `set` says.
    pub(crate) fn consensus_failed(&self) {
        self.consensus_failed.store(true, Ordering::Release);
    }

    pub(crate) fn get(&self) -> bool {
        self.serving.load(Ordering::Acquire) && !self.consensus_failed.load(Ordering::Acquire)
    }
}

/// Status line, content type and body for `path`; `metrics` renders the
/// Prometheus text and `version` the version report, each only when asked
/// for.
fn respond(
    path: &str,
    ready: bool,
    metrics: impl FnOnce() -> String,
    version: impl FnOnce() -> String,
) -> (&'static str, &'static str, String) {
    match path {
        "/version" => ("200 OK", "application/json", version()),
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

/// Reads the node's credentials and trust again from their files and
/// publishes them, or says why it kept what it had.
pub(crate) type ReloadHook = Arc<dyn Fn() -> Result<(), String> + Send + Sync>;

/// Status line, content type and body of a `/reload` request: only a POST
/// reloads, and a node without anything to reload says so. Whoever reaches
/// the ops port can ask for it; it reads only the files the configuration
/// names and publishes them only if they validate, so it grants nothing.
fn reload_response(
    method: &str,
    reload: Option<&ReloadHook>,
) -> (&'static str, &'static str, String) {
    match (method, reload) {
        ("POST", Some(reload)) => match reload() {
            Ok(()) => (
                "200 OK",
                "application/json",
                r#"{"reloaded":true}"#.to_string(),
            ),
            Err(e) => (
                "500 Internal Server Error",
                "application/json",
                serde_json::json!({ "reloaded": false, "error": e }).to_string(),
            ),
        },
        ("POST", None) => (
            "404 Not Found",
            "application/json",
            r#"{"error":"this node has no node identity to reload"}"#.to_string(),
        ),
        _ => (
            "405 Method Not Allowed",
            "application/json",
            r#"{"error":"POST to reload"}"#.to_string(),
        ),
    }
}

/// What `/blob-gc` asks of the orphan blob collector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlobGcRequest {
    /// Where collection stands.
    Status,
    /// Run one pass now, reconciling accounting as well when asked.
    Run { reconcile: bool },
}

/// Answers a [`BlobGcRequest`] as JSON, or says why it could not.
pub(crate) type BlobGcView = Arc<dyn Fn(BlobGcRequest) -> Result<String, String> + Send + Sync>;

/// Status line, content type and body of a `/blob-gc` request: `GET
/// /blob-gc` reports, `POST /blob-gc/run` (with `?reconcile=true` to check
/// accounting too) runs a pass. Whoever reaches the ops port can ask for
/// either: a pass is the one the collector runs on its own, which cannot
/// shorten the grace or pass a reader, retained history or an upload, so it
/// grants nothing.
fn blob_gc_response(
    method: &str,
    target: &str,
    view: &BlobGcView,
) -> (&'static str, &'static str, String) {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let request = match (method, path) {
        ("GET", "/blob-gc") => BlobGcRequest::Status,
        ("POST", "/blob-gc/run") => BlobGcRequest::Run {
            reconcile: query.split('&').any(|pair| pair == "reconcile=true"),
        },
        (_, "/blob-gc") => {
            return (
                "405 Method Not Allowed",
                "application/json",
                r#"{"error":"GET for the status, POST /blob-gc/run to run a pass"}"#.to_string(),
            );
        }
        (_, "/blob-gc/run") => {
            return (
                "405 Method Not Allowed",
                "application/json",
                r#"{"error":"POST to run a pass"}"#.to_string(),
            );
        }
        _ => {
            return (
                "404 Not Found",
                "application/json",
                r#"{"error":"not found"}"#.to_string(),
            );
        }
    };
    match view(request) {
        Ok(body) => ("200 OK", "application/json", body),
        Err(e) => (
            "500 Internal Server Error",
            "application/json",
            serde_json::json!({ "error": e }).to_string(),
        ),
    }
}

/// Set the orphan blob collection series from `status`.
pub(crate) fn sample_blob_gc(status: &coordinode_embed::db::GcStatus) {
    let flag = |on: bool| if on { 1.0 } else { 0.0 };
    let now_micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_micros());
    let oldest = status.oldest_candidate_micros.map_or(0.0, |at| {
        // A release stamped after this read is no age.
        u128::try_from(at)
            .ok()
            .and_then(|at| now_micros.checked_sub(at))
            .map_or(0.0, |age| age as f64 / 1e6)
    });
    let gauges: [(&'static str, f64); 10] = [
        (
            "coordinode_blobstore_disk_bytes",
            status.blob_disk_bytes as f64,
        ),
        (
            "coordinode_blobstore_stale_bytes",
            status.stale_blob_bytes as f64,
        ),
        ("coordinode_blob_gc_candidates", status.candidates as f64),
        (
            "coordinode_blob_gc_candidates_due",
            status.candidates_due as f64,
        ),
        ("coordinode_blob_gc_oldest_candidate_seconds", oldest),
        (
            "coordinode_blob_gc_staged_uploads",
            status.staged_uploads as f64,
        ),
        (
            "coordinode_blob_gc_retired_pending_bytes",
            status.retired_pending_bytes as f64,
        ),
        (
            "coordinode_blob_gc_compaction_pending",
            flag(status.compaction_pending),
        ),
        ("coordinode_blob_gc_blocked", flag(status.blocked.is_some())),
        ("coordinode_blob_gc_active", flag(status.active)),
    ];
    for (name, value) in gauges {
        metrics::gauge!(name).set(value);
    }
    let counters: [(&'static str, u64); 3] = [
        (
            "coordinode_blob_gc_reclaimed_bytes_total",
            status.reclaimed_bytes,
        ),
        (
            "coordinode_blob_gc_reclaimed_chunks_total",
            status.reclaimed_chunks,
        ),
        (
            "coordinode_blob_gc_reclaimed_disk_bytes_total",
            status.reclaimed_disk_bytes,
        ),
    ];
    for (name, value) in counters {
        metrics::counter!(name).absolute(value);
    }
}

/// Sets the gauges whose value is only meaningful when sampled, right before
/// `/metrics` renders them: a scrape samples them, and nothing samples them
/// between scrapes.
pub(crate) type SampleGauges = Arc<dyn Fn() + Send + Sync>;

/// Renders this member's version report as JSON, at request time.
pub(crate) type VersionView = Arc<dyn Fn() -> String + Send + Sync>;

/// Appends the series kept outside the `metrics` facade to a `/metrics`
/// body: those that must disappear with what they describe, which the facade
/// cannot remove.
pub(crate) type OwnedSeries = Arc<dyn Fn(&mut String) + Send + Sync>;

/// Start the operational HTTP server on `listener`, bound by the caller at
/// startup so a busy port fails the start.
///
/// Handles /health, /ready, /metrics, /version, `POST /reload` and
/// `/blob-gc`; `/ready` follows `readiness`, `/metrics` runs `sample` first
/// and appends `owned`, `/version` renders `version`, `/reload` runs `reload`
/// and `/blob-gc` asks `blob_gc`. Runs until the process exits.
pub(crate) async fn start_ops_server(
    listener: TcpListener,
    readiness: Readiness,
    sample: SampleGauges,
    owned: OwnedSeries,
    version: VersionView,
    reload: Option<ReloadHook>,
    blob_gc: BlobGcView,
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
        let owned = Arc::clone(&owned);
        let version = Arc::clone(&version);
        let reload = reload.clone();
        let blob_gc = Arc::clone(&blob_gc);
        let ready = readiness.get();

        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            let n = match tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await {
                Ok(n) => n,
                Err(_) => return,
            };

            let request = String::from_utf8_lossy(&buf[..n]);

            // Parse the HTTP request line
            let mut line = request
                .lines()
                .next()
                .unwrap_or_default()
                .split_whitespace();
            let method = line.next().unwrap_or("GET");
            let path = line.next().unwrap_or("/");

            let (status, content_type, body) = if path == "/reload" {
                // Reading the files blocks; keep it off the runtime.
                let method = method.to_owned();
                tokio::task::spawn_blocking(move || reload_response(&method, reload.as_ref()))
                    .await
                    .unwrap_or_else(|e| {
                        (
                            "500 Internal Server Error",
                            "application/json",
                            serde_json::json!({ "error": e.to_string() }).to_string(),
                        )
                    })
            } else if path == "/blob-gc"
                || path.starts_with("/blob-gc/")
                || path.starts_with("/blob-gc?")
            {
                // A pass reads and writes the store; keep it off the runtime.
                let (method, target) = (method.to_owned(), path.to_owned());
                tokio::task::spawn_blocking(move || blob_gc_response(&method, &target, &blob_gc))
                    .await
                    .unwrap_or_else(|e| {
                        (
                            "500 Internal Server Error",
                            "application/json",
                            serde_json::json!({ "error": e.to_string() }).to_string(),
                        )
                    })
            } else {
                respond(
                    path,
                    ready,
                    || {
                        sample();
                        let mut body = handle.render();
                        owned(&mut body);
                        body
                    },
                    || version(),
                )
            };

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
