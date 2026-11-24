//! The consumer-retention registry's operator config and metrics.
//!
//! `coordinode serve` exposes the registry's background cadences as config
//! keys. The registry itself is opened by the member's database over the
//! group's log, which takes up the registrations of an earlier run; this
//! module turns the parsed values into the tuning its background service
//! starts with, and samples its gauges.
//!
//! The MVCC time-travel window is not a registry setting: the engine
//! enforces it from `StorageConfig::retention_window_secs`, and the registry
//! only ever lowers the GC watermark further for a lagging consumer.

use coordinode_replicate::{BackgroundConfig, SeqnoConsumerRegistry, ShardConsumerRegistry};

/// Operator-supplied registry tuning, parsed from the config file.
///
/// Every field is `Option`: `None` means "keep the registry's built-in
/// default", so an operator overrides only what they explicitly set.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RegistryTuning {
    /// Heartbeat coalescing window in milliseconds (`registry_heartbeat_ms`).
    /// `None` keeps the default 1000 ms.
    pub heartbeat_window_ms: Option<u64>,
}

/// The tuning of the registry's background service, with the operator's
/// overrides from `tuning`: the member's database starts the service with
/// it when it opens the registry.
pub(crate) fn background_config(tuning: RegistryTuning) -> BackgroundConfig {
    let mut config = BackgroundConfig::default();
    if let Some(ms) = tuning.heartbeat_window_ms {
        config.heartbeat_window_ms = ms;
    }
    config
}

/// Set the per-consumer gauges (lag in positions, age of the work owed,
/// heartbeat age, retained bytes) from the registry as it stands: called when
/// the ops port is scraped, so the `consumer_id` label is produced only then.
pub(crate) fn sample_consumer_gauges(registry: &ShardConsumerRegistry) {
    // The listing sets the gauges; the snapshot it returns is not needed.
    let _listed = registry.list_consumers();
    // So does the summary, with the bytes each space keeps counted once.
    if let Err(e) = registry.retention_summary() {
        tracing::warn!(error = %e, "registry: retention summary failed");
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
