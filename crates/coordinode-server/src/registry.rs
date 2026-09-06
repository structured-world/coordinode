//! Consumer-retention registry construction from operator config.
//!
//! `coordinode serve` exposes the registry's background cadences as config
//! keys. This module is the single place that turns those parsed values into
//! a live [`ShardConsumerRegistry`] plus its background service. `main.rs`
//! and the server-level test both go through here, so the wiring production
//! runs is the wiring that gets tested.
//!
//! The MVCC time-travel window is not a registry setting: the engine
//! enforces it from `StorageConfig::retention_window_secs`, and the registry
//! only ever lowers the GC watermark further for a lagging consumer.

use std::sync::Arc;

use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_replicate::{
    BackgroundConfig, RegistryBackground, ShardConsumerRegistry, SystemClock,
};
use coordinode_storage::engine::core::StorageEngine;

/// Operator-supplied registry tuning, parsed from the config file.
///
/// Every field is `Option`: `None` means "keep the registry's built-in
/// default", so an operator overrides only what they explicitly set.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RegistryTuning {
    /// Heartbeat coalescing window in milliseconds (`registry_heartbeat_ms`).
    /// `None` keeps the default 100 ms.
    pub heartbeat_window_ms: Option<u64>,
    /// TTL-eviction sweep interval in milliseconds (`registry_eviction_ms`).
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
pub(crate) fn build_consumer_registry(
    engine: Arc<StorageEngine>,
    pipeline: Arc<dyn ProposalPipeline>,
    node_id: u64,
    tuning: RegistryTuning,
) -> (ShardConsumerRegistry, RegistryBackground) {
    let registry = ShardConsumerRegistry::new(
        engine,
        pipeline,
        Arc::new(ProposalIdGenerator::with_base(node_id << 48)),
        Arc::new(SystemClock),
    );
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
