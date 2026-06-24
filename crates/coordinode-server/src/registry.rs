//! Consumer-retention registry construction from operator config.
//!
//! `coordinode serve` exposes the registry's MVCC retention window and
//! background cadences as CLI flags. This module is the single place that
//! turns those parsed values into a live [`ShardConsumerRegistry`] plus its
//! background service. `main.rs` and the server-level test both go through
//! here, so the wiring production runs is the wiring that gets tested.

use std::sync::Arc;

use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_replicate::{
    BackgroundConfig, RegistryBackground, ShardConsumerRegistry, SystemClock,
};
use coordinode_storage::engine::core::StorageEngine;

/// Microseconds per second. The retention window is operator-facing in
/// seconds; the registry stores it in HLC microseconds (ADR-007).
const US_PER_SEC: u64 = 1_000_000;

/// Operator-supplied registry tuning, parsed from `coordinode serve` flags.
///
/// Every field is `Option`: `None` means "keep the registry's built-in
/// default", so an operator overrides only what they explicitly set.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RegistryTuning {
    /// MVCC time-travel / retention window in seconds (`--retention-window-secs`).
    /// `None` keeps the default 7-day window.
    pub retention_window_secs: Option<u64>,
    /// Heartbeat coalescing window in milliseconds (`--registry-heartbeat-ms`).
    /// `None` keeps the default 100 ms.
    pub heartbeat_window_ms: Option<u64>,
    /// TTL-eviction sweep interval in milliseconds (`--registry-eviction-ms`).
    /// `None` keeps the default 1000 ms.
    pub eviction_interval_ms: Option<u64>,
}

/// Build the per-shard consumer-retention registry (ADR-028) and start its
/// background service, applying any operator overrides from `tuning`.
///
/// Returns the live [`ShardConsumerRegistry`] (cheap to clone — `Arc`-backed —
/// so producers like the change-stream service register through it) and its
/// [`RegistryBackground`] handle, which must be held for the process lifetime:
/// dropping it shuts the background service down (with a final heartbeat flush).
/// The registry publishes the MVCC GC watermark to the engine synchronously
/// during construction, so the engine's retention floor already reflects the
/// configured window by the time this returns.
pub(crate) fn build_consumer_registry(
    engine: Arc<StorageEngine>,
    pipeline: Arc<dyn ProposalPipeline>,
    node_id: u64,
    tuning: RegistryTuning,
) -> (ShardConsumerRegistry, RegistryBackground) {
    let mut registry = ShardConsumerRegistry::new(
        engine,
        pipeline,
        Arc::new(ProposalIdGenerator::with_base(node_id << 48)),
        Arc::new(SystemClock),
    );
    if let Some(secs) = tuning.retention_window_secs {
        // saturating_mul: a window so large it overflows u64 microseconds
        // (~584942 years) clamps to "retain forever", which is the intended
        // reading of an absurdly large operator window, not a bug to surface.
        registry = registry.with_retention_window_us(secs.saturating_mul(US_PER_SEC));
    }
    let mut bg_cfg = BackgroundConfig::default();
    if let Some(ms) = tuning.heartbeat_window_ms {
        bg_cfg.heartbeat_window_ms = ms;
    }
    if let Some(ms) = tuning.eviction_interval_ms {
        bg_cfg.eviction_interval_ms = ms;
    }
    // start_background borrows &self, so the registry stays owned and is returned
    // for producers to register through (the bg task holds its own Arc to core).
    let background = registry.start_background(bg_cfg);
    (registry, background)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
