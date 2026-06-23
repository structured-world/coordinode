//! Per-endpoint capacity tracking + hard-limit enforcement.
//!
//! Implements INV-D3 (`used ≤ hard_limit always`) from the storage
//! stack design. For each [`crate::engine::config::EndpointConfig`]
//! with a non-zero `hard_limit_bytes`, this module:
//!
//! 1. Periodically scans the endpoint's per-partition `tables/`
//!    directories and recomputes total bytes-on-disk (`used_bytes`).
//! 2. Tracks alert-threshold crossings (80% / 90% / 95% / 100%) and
//!    emits Prometheus metrics + structured tracing events.
//! 3. At the 95% emergency threshold with [`HardLimitStrategy::CascadeEvict`],
//!    fires a cascade eviction through the per-LSM-level routing
//!    (using the bottom-level compaction mechanism from the routing
//!    module).
//! 4. At 100%, flips the endpoint's `is_writable` flag to `false`.
//!    Writes targeting that endpoint surface as
//!    `StorageError::CapacityExhausted` so the coordinator can retry
//!    on a different endpoint.
//!
//! ## Polling vs flush-callback tracking
//!
//! The storage-stack design implies "synchronous tracking on every
//! SST flush / compact / delete". `coordinode-lsm-tree` does not yet
//! expose flush/compact callbacks at the consumer boundary, so this
//! implementation uses **periodic disk scanning** instead.
//! Implications:
//!
//! - Threshold detection lag = up to one scan interval (default 5 s).
//! - Operator-facing alerts (80% / 90%) are observed within the
//!   interval and are operationally useful at any reasonable cadence.
//! - The 100% rejection gate has the same lag — a burst of writes
//!   within one interval *can* push `used_bytes` over `hard_limit`
//!   transiently. The 95% emergency threshold + 5% safety margin
//!   keeps this within the configured limit in practice. Operators
//!   who require synchronous enforcement should size `hard_limit_bytes`
//!   conservatively (≥ 5% headroom below physical capacity).
//!
//! A future upstream addition of flush/compact callbacks to
//! `coordinode-lsm-tree` would let this module switch to incremental
//! tracking with no API change above this layer.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::engine::config::{EndpointConfig, HardLimitStrategy};

/// Severity of a capacity-threshold crossing event.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacitySeverity {
    /// Below 80% — normal operation, no alert.
    Normal,
    /// 80% to 90% — operator-facing warning, no action.
    Warning,
    /// 90% to 95% — operator-facing critical alert, no action.
    Critical,
    /// 95% to 100% — emergency. Cascade eviction fires here if
    /// `HardLimitStrategy::CascadeEvict`.
    Emergency,
    /// At or above 100% — endpoint is_writable flag flipped off,
    /// writes targeting this endpoint reject until cascade or
    /// operator intervention drops usage back below threshold.
    Full,
}

impl CapacitySeverity {
    /// Resolve the severity for a given `used / hard_limit` ratio.
    /// `hard_limit == 0` means "no limit configured" → always
    /// [`Self::Normal`].
    #[must_use]
    pub fn for_usage(used_bytes: u64, hard_limit_bytes: u64) -> Self {
        if hard_limit_bytes == 0 {
            return Self::Normal;
        }
        let pct = (used_bytes as u128 * 100) / hard_limit_bytes as u128;
        match pct {
            0..=79 => Self::Normal,
            80..=89 => Self::Warning,
            90..=94 => Self::Critical,
            95..=99 => Self::Emergency,
            _ => Self::Full,
        }
    }

    /// Human-readable label for Prometheus / log fields.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Warning => "warning",
            Self::Critical => "critical",
            Self::Emergency => "emergency",
            Self::Full => "full",
        }
    }
}

/// Atomically-tracked usage state for one endpoint.
#[derive(Debug)]
pub struct EndpointUsage {
    /// Endpoint id this state belongs to. Used as the Prometheus
    /// `endpoint_id` label and the cascade-eviction target.
    pub id: String,
    /// Most-recent scanned bytes-on-disk total across every
    /// `<endpoint>/<partition>/tables/` directory served by this
    /// endpoint.
    pub used_bytes: AtomicU64,
    /// `hard_limit_bytes` snapshot from the endpoint config at engine
    /// open time. `0` means "no limit configured" — tracker still
    /// counts usage for diagnostics but never triggers thresholds.
    pub hard_limit_bytes: u64,
    /// `capacity_bytes` snapshot from the endpoint config — physical
    /// size of the underlying media (informational only).
    pub capacity_bytes: u64,
    /// Configured strategy for handling
    /// `used_bytes + write_size > hard_limit_bytes`.
    pub strategy: HardLimitStrategy,
    /// `true` when writes targeting this endpoint are accepted. Flips
    /// to `false` on a `Full` (≥ 100%) severity crossing; flips back
    /// to `true` once a refresh observes usage back under the limit
    /// (cascade eviction or operator-driven cleanup).
    pub is_writable: AtomicBool,
    /// Last observed severity — used to detect transitions so alert
    /// metrics increment exactly once per crossing (not on every
    /// scan).
    pub last_severity: std::sync::Mutex<CapacitySeverity>,
}

impl EndpointUsage {
    /// Build a fresh usage tracker from an endpoint config. `used_bytes`
    /// starts at zero; first scan populates the real value.
    #[must_use]
    pub fn from_config(endpoint: &EndpointConfig) -> Self {
        Self {
            id: endpoint.id.clone(),
            used_bytes: AtomicU64::new(0),
            hard_limit_bytes: endpoint.hard_limit_bytes,
            capacity_bytes: endpoint.capacity_bytes,
            strategy: endpoint.hard_limit_strategy,
            is_writable: AtomicBool::new(true),
            last_severity: std::sync::Mutex::new(CapacitySeverity::Normal),
        }
    }

    /// Current used-bytes snapshot (Acquire ordering).
    pub fn used(&self) -> u64 {
        self.used_bytes.load(Ordering::Acquire)
    }

    /// `true` iff writes targeting this endpoint may proceed. Flips
    /// to `false` when a scan observes the endpoint at or above its
    /// `hard_limit_bytes`.
    pub fn is_writable(&self) -> bool {
        self.is_writable.load(Ordering::Acquire)
    }

    /// Severity for the current `used` snapshot.
    pub fn severity(&self) -> CapacitySeverity {
        CapacitySeverity::for_usage(self.used(), self.hard_limit_bytes)
    }
}

/// Per-endpoint capacity tracker — engine-owned, shared across the
/// background scanner and the write path.
pub struct CapacityTracker {
    endpoints: BTreeMap<String, Arc<EndpointUsage>>,
}

impl CapacityTracker {
    /// Construct a tracker over the given endpoint set. One
    /// [`EndpointUsage`] per endpoint id.
    ///
    /// # Panics
    /// On duplicate endpoint id (already rejected by
    /// `StorageConfig::with_endpoints`, panics defensively here).
    #[must_use]
    pub fn new(endpoints: &[EndpointConfig]) -> Self {
        let mut map = BTreeMap::new();
        for ep in endpoints {
            let prev = map.insert(ep.id.clone(), Arc::new(EndpointUsage::from_config(ep)));
            assert!(
                prev.is_none(),
                "duplicate endpoint id in CapacityTracker: {:?}",
                ep.id,
            );
        }
        Self { endpoints: map }
    }

    /// Get the usage state for an endpoint by id.
    pub fn get(&self, endpoint_id: &str) -> Option<Arc<EndpointUsage>> {
        self.endpoints.get(endpoint_id).cloned()
    }

    /// Iterate every tracked endpoint's usage state. Stable order
    /// (`BTreeMap` by id).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Arc<EndpointUsage>)> {
        self.endpoints.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Refresh `used_bytes` for every endpoint by scanning its
    /// per-partition `tables/` directories on disk.
    ///
    /// `partition_names` is the list of partition directory names to
    /// scan under each endpoint's root (e.g. `["node", "adj",
    /// "edgeprop", ...]`). Provided by the caller to avoid coupling
    /// this module to [`crate::engine::partition::Partition`].
    ///
    /// Side effects on each [`EndpointUsage`]:
    /// - Updates `used_bytes` atomically (Release ordering).
    /// - Re-evaluates severity. On transition to a higher severity:
    ///   increments the per-severity Prometheus counter (caller is
    ///   expected to read these counters) and emits `tracing::warn!`.
    /// - Flips `is_writable` to `false` if severity is `Full`; flips
    ///   back to `true` if severity drops below `Full` (cascade or
    ///   operator cleanup brought usage back under limit).
    pub fn refresh(
        &self,
        endpoint_paths: &BTreeMap<String, std::path::PathBuf>,
        _partition_names: &[&str],
    ) {
        for (id, usage) in &self.endpoints {
            let Some(root) = endpoint_paths.get(id) else {
                continue;
            };
            // Recursively sum every file under the endpoint root.
            // This captures every on-disk artefact the engine writes
            // through this mount point, not just partition SSTs:
            //   - `<part>/tables/`     — LSM SST files
            //   - `<part>/manifest/`   — lsm-tree manifest + segment metadata
            //   - `wal/standalone.wal` — standalone WAL file (R157)
            //   - `oplog/<shard>/`     — Raft oplog segments (R157)
            //   - `text_indexes/`      — tantivy FTS index segments
            //   - any future engine-managed subdirectory
            //
            // Matches `du -sh <endpoint>` — the operational definition
            // of "how full is this disk from coordinode's view". A
            // narrower per-partition scan misses anything outside
            // `tables/` and silently under-counts capacity.
            //
            // The `partition_names` argument is preserved for
            // backward-compatible API but no longer used — the walk
            // is exhaustive.
            let total = dir_size(root);
            usage.used_bytes.store(total, Ordering::Release);

            // Prometheus gauges — read by the metrics-exporter-prometheus
            // installed in the server binary. Emitting from this module
            // keeps the metric labels canonical (`endpoint_id`) and
            // avoids any cross-crate reader of the AtomicU64 needing to
            // re-derive the limit values.
            metrics::gauge!("endpoint_used_bytes", "endpoint_id" => id.clone()).set(total as f64);
            metrics::gauge!("endpoint_hard_limit_bytes", "endpoint_id" => id.clone())
                .set(usage.hard_limit_bytes as f64);

            let new_sev = CapacitySeverity::for_usage(total, usage.hard_limit_bytes);
            // Severity-transition handling.
            #[allow(clippy::expect_used)]
            let mut last = usage
                .last_severity
                .lock()
                .expect("capacity severity mutex poisoned");
            if new_sev != *last {
                if (new_sev as u8) > (*last as u8) {
                    // Crossing UP — log + alert counter + may flip
                    // is_writable / fire cascade (cascade happens in
                    // the engine layer after this refresh returns).
                    tracing::warn!(
                        endpoint = %id,
                        severity = new_sev.label(),
                        used_bytes = total,
                        hard_limit_bytes = usage.hard_limit_bytes,
                        "endpoint capacity threshold crossed",
                    );
                    metrics::counter!(
                        "endpoint_threshold_alerts_total",
                        "endpoint_id" => id.clone(),
                        "severity" => new_sev.label(),
                    )
                    .increment(1);
                } else {
                    tracing::info!(
                        endpoint = %id,
                        severity = new_sev.label(),
                        used_bytes = total,
                        hard_limit_bytes = usage.hard_limit_bytes,
                        "endpoint capacity recovered below threshold",
                    );
                }
                *last = new_sev;
            }
            // is_writable flag — Full means hard reject.
            let writable = !matches!(new_sev, CapacitySeverity::Full);
            usage.is_writable.store(writable, Ordering::Release);
        }
    }
}

/// Recursive directory size — sum of file lengths under `root`.
/// Tolerates missing entries silently (returns 0); a directory that
/// disappeared between `read_dir` and the per-entry `metadata` call
/// just stops contributing rather than aborting the scan.
fn dir_size(root: &Path) -> u64 {
    let mut total = 0u64;
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_file() {
            total = total.saturating_add(meta.len());
        } else if meta.is_dir() {
            total = total.saturating_add(dir_size(&entry.path()));
        }
    }
    total
}

/// Background capacity scanner — a single OS thread that periodically
/// re-runs the per-endpoint disk scan + alert + auto-cascade logic.
///
/// Follows the same lifecycle pattern as
/// `crate::engine::flush::FlushManager` / `CompactionScheduler` (also
/// std-thread-based with an `Arc<AtomicBool>` shutdown flag) for
/// consistency across `coordinode-storage` background workers. The
/// engine owns the scanner; dropping the engine flips the shutdown
/// flag and joins the thread before tree handles are released.
///
/// The scanner does NOT itself decide cascade-eviction strategy or
/// emit metrics — it just calls back into `refresh_fn` on every tick.
/// The engine wires `refresh_fn` to `StorageEngine::refresh_capacity`
/// so all the policy lives in one place.
pub struct CapacityScanner {
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl CapacityScanner {
    /// Spawn the scanner thread.
    ///
    /// `interval` is the wall-clock period between scans. Default
    /// engine cadence is 5 s but tests may pass shorter values for
    /// faster convergence. `refresh_fn` is called from the scanner
    /// thread on every tick; it MUST be `Send + 'static` because the
    /// thread outlives the call stack that constructed the scanner.
    ///
    /// Returns an `Err` only if the OS thread spawn itself fails
    /// (resource-limit hit at process level).
    pub fn start<F>(interval: Duration, refresh_fn: F) -> std::io::Result<Self>
    where
        F: Fn() + Send + 'static,
    {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_w = Arc::clone(&shutdown);
        let handle = std::thread::Builder::new()
            .name("coord-capacity-scanner".to_string())
            .spawn(move || capacity_scanner_loop(interval, shutdown_w, refresh_fn))?;
        Ok(Self {
            shutdown,
            handle: Some(handle),
        })
    }

    /// Signal the scanner to stop after its current iteration. Idempotent.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

impl Drop for CapacityScanner {
    fn drop(&mut self) {
        self.shutdown();
        if let Some(handle) = self.handle.take() {
            // Best-effort join — a poisoned thread is logged but does
            // not panic the engine drop (writes already happened; the
            // scanner is bookkeeping only).
            if let Err(e) = handle.join() {
                tracing::warn!(
                    error = ?e,
                    "capacity scanner thread panicked during shutdown",
                );
            }
        }
    }
}

/// The actual loop body — pulled out as a free function so unit tests
/// can drive the loop synchronously by injecting a controlled
/// shutdown flag.
fn capacity_scanner_loop<F>(interval: Duration, shutdown: Arc<AtomicBool>, refresh_fn: F)
where
    F: Fn(),
{
    // Sleep granularity: the scanner wakes every `tick_granularity` to
    // check the shutdown flag promptly even when `interval` is large.
    // This keeps engine close latency bounded (~100 ms) regardless of
    // the scan cadence.
    let tick_granularity = Duration::from_millis(100);
    loop {
        // Sleep BEFORE the first refresh so engine open completes
        // without a concurrent scan competing with the user thread's
        // initial writes + manual `refresh_capacity()`. Warm-load
        // (in the open path) already hydrates the tracker from
        // persisted Schema state; the first poll-based refresh is
        // only useful after some wall-clock has passed, at which
        // point on-disk state may have changed.
        //
        // Sleep up to `interval`, checking the shutdown flag every
        // `tick_granularity`. Burns nothing if shutdown is already set.
        let mut elapsed = Duration::ZERO;
        while elapsed < interval {
            if shutdown.load(Ordering::Acquire) {
                return;
            }
            std::thread::sleep(tick_granularity);
            elapsed += tick_granularity;
        }
        // Now run the refresh — at the end of each interval, never at t=0.
        refresh_fn();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
