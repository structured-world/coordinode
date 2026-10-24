//! Per-endpoint capacity reporting: how full each endpoint is against its
//! hard limit, the alerts its thresholds raise, and the cascade eviction an
//! endpoint configured for it starts.
//!
//! The limit itself is enforced by the owned footprint
//! ([`crate::engine::footprint`]), which charges every byte before it is
//! written; this module reads that account, it does not gate anything. For
//! each endpoint, on the scanner's cadence:
//!
//! 1. `used_bytes` is the endpoint's charged footprint (what its files take
//!    on disk); an endpoint without a hard limit has no footprint and is
//!    measured by walking its folder, for the gauges only.
//! 2. Alert-threshold crossings (80% / 90% / 95% / 100%) emit Prometheus
//!    metrics and structured tracing events.
//! 3. At the 95% emergency threshold with
//!    [`HardLimitStrategy::CascadeEvict`], a cascade eviction is fired
//!    through the per-LSM-level routing.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    /// At or above 100%: the footprint refuses growth until reclamation,
    /// cascade or the operator brings it back under the limit.
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
    /// The endpoint's footprint at the last refresh: what its files take on
    /// disk, what admitted writes hold and what the memtables owe, as the
    /// owned footprint counts them.
    pub used_bytes: AtomicU64,
    /// `hard_limit_bytes` snapshot from the endpoint config at engine
    /// open time. `0` means "no limit configured" — tracker still
    /// counts usage for diagnostics but never triggers thresholds.
    pub hard_limit_bytes: u64,
    /// The part of the limit client writes may take (the limit less the
    /// recovery reserve), which the severities are fractions of: `Full` is
    /// where client writes are refused. Follows a limit changed at runtime.
    pub ceiling_bytes: AtomicU64,
    /// `capacity_bytes` snapshot from the endpoint config — physical
    /// size of the underlying media (informational only).
    pub capacity_bytes: u64,
    /// Configured strategy for handling
    /// `used_bytes + write_size > hard_limit_bytes`.
    pub strategy: HardLimitStrategy,
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
            ceiling_bytes: AtomicU64::new(endpoint.hard_limit_bytes - endpoint.recovery_reserve()),
            capacity_bytes: endpoint.capacity_bytes,
            strategy: endpoint.hard_limit_strategy,
            last_severity: std::sync::Mutex::new(CapacitySeverity::Normal),
        }
    }

    /// Current used-bytes snapshot (Acquire ordering).
    pub fn used(&self) -> u64 {
        self.used_bytes.load(Ordering::Acquire)
    }

    /// Severity for the current `used` snapshot, against the part of the
    /// limit client writes may take.
    pub fn severity(&self) -> CapacitySeverity {
        severity_of(
            self.used(),
            self.hard_limit_bytes,
            self.ceiling_bytes.load(Ordering::Acquire),
        )
    }
}

/// The severity of `used` bytes under a hard limit `limit` (`0`: none) of
/// which client writes may take `ceiling`: a reserve as large as the limit
/// leaves client writes nothing, which is full.
fn severity_of(used: u64, limit: u64, ceiling: u64) -> CapacitySeverity {
    match (limit, ceiling) {
        (0, _) => CapacitySeverity::Normal,
        (_, 0) => CapacitySeverity::Full,
        (_, ceiling) => CapacitySeverity::for_usage(used, ceiling),
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

    /// Refresh `used_bytes` for every endpoint: its charged footprint, or,
    /// for an endpoint without a hard limit, the bytes its folder holds.
    ///
    /// Side effects on each [`EndpointUsage`]:
    /// - Updates `used_bytes` atomically (Release ordering).
    /// - Re-evaluates severity. On transition to a higher severity:
    ///   increments the per-severity Prometheus counter (caller is
    ///   expected to read these counters) and emits `tracing::warn!`.
    pub fn refresh(
        &self,
        endpoint_paths: &BTreeMap<String, std::path::PathBuf>,
        footprints: &crate::engine::footprint::Footprints,
    ) {
        for (id, usage) in &self.endpoints {
            let total = match footprints.get(id) {
                Some(footprint) => {
                    publish_footprint(id, footprint);
                    usage
                        .ceiling_bytes
                        .store(footprint.client_ceiling(), Ordering::Release);
                    footprint.committed()
                }
                None => match endpoint_paths.get(id) {
                    Some(root) => dir_size(root),
                    None => continue,
                },
            };
            usage.used_bytes.store(total, Ordering::Release);

            // Prometheus gauges — read by the metrics-exporter-prometheus
            // installed in the server binary. Emitting from this module
            // keeps the metric labels canonical (`endpoint_id`) and
            // avoids any cross-crate reader of the AtomicU64 needing to
            // re-derive the limit values.
            metrics::gauge!("endpoint_used_bytes", "endpoint_id" => id.clone()).set(total as f64);
            metrics::gauge!("endpoint_hard_limit_bytes", "endpoint_id" => id.clone())
                .set(usage.hard_limit_bytes as f64);

            let new_sev = usage.severity();
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
        }
    }
}

/// Publish what endpoint `id`'s footprint holds, has admitted and owes, and
/// the reserve it keeps.
fn publish_footprint(id: &str, footprint: &crate::engine::footprint::Footprint) {
    let id = id.to_owned();
    for (name, value) in [
        ("coordinode_footprint_charged_bytes", footprint.charged()),
        ("coordinode_footprint_admitted_bytes", footprint.admitted()),
        ("coordinode_footprint_flush_debt_bytes", footprint.debt()),
        ("coordinode_footprint_reserve_bytes", footprint.reserve()),
        ("coordinode_footprint_limit_bytes", footprint.limit()),
    ] {
        metrics::gauge!(name, "endpoint_id" => id.clone()).set(value as f64);
    }
}

/// Recursive directory size, the sum of file lengths under `root`, for the
/// gauges of an endpoint without a footprint. Tolerates missing entries
/// silently (returns 0); a directory that disappeared between `read_dir`
/// and the per-entry `metadata` call just stops contributing rather than
/// aborting the scan.
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
    /// The scanner sleeps a whole interval on this; a stop interrupts it.
    wake: Arc<coordinode_core::txn::wake::Wake>,
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
        let wake = Arc::new(coordinode_core::txn::wake::Wake::default());
        let shutdown_w = Arc::clone(&shutdown);
        let wake_w = Arc::clone(&wake);
        let handle = std::thread::Builder::new()
            .name("coord-capacity-scanner".to_string())
            .spawn(move || capacity_scanner_loop(interval, &shutdown_w, &wake_w, refresh_fn))?;
        Ok(Self {
            shutdown,
            wake,
            handle: Some(handle),
        })
    }

    /// Signal the scanner to stop, waking it from its sleep. Idempotent.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.wake.interrupt();
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
///
/// The scan reads the disk, which changes under the engine's feet (file
/// deletion after a compaction runs asynchronously), so it is periodic by
/// nature; what it does not do is wake between scans. It sleeps the whole
/// interval and a stop interrupts the sleep.
fn capacity_scanner_loop<F>(
    interval: Duration,
    shutdown: &AtomicBool,
    wake: &coordinode_core::txn::wake::Wake,
    refresh_fn: F,
) where
    F: Fn(),
{
    wake.bind();
    loop {
        // Sleep BEFORE the first refresh so engine open completes
        // without a concurrent scan competing with the user thread's
        // initial writes + manual `refresh_capacity()`. Warm-load
        // (in the open path) already hydrates the tracker from
        // persisted Schema state; the first poll-based refresh is
        // only useful after some wall-clock has passed, at which
        // point on-disk state may have changed.
        let deadline = std::time::Instant::now() + interval;
        loop {
            if shutdown.load(Ordering::Acquire) {
                return;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            wake.wait(Some(deadline - now));
        }
        // Now run the refresh — at the end of each interval, never at t=0.
        refresh_fn();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
