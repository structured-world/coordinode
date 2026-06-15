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
/// The returned [`RegistryBackground`] handle must be held for the process
/// lifetime: dropping it shuts the background service down (with a final
/// heartbeat flush). The registry publishes the MVCC GC watermark to the
/// engine synchronously during construction, so the engine's retention floor
/// already reflects the configured window by the time this returns.
pub(crate) fn build_consumer_registry(
    engine: Arc<StorageEngine>,
    pipeline: Arc<dyn ProposalPipeline>,
    node_id: u64,
    tuning: RegistryTuning,
) -> RegistryBackground {
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
    registry.start_background(bg_cfg)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use coordinode_core::txn::timestamp::TimestampOracle;
    use coordinode_raft::proposal::OwnedLocalProposalPipeline;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };

    /// Open a fresh single-endpoint engine in a temp directory. Returns the
    /// engine and its dir guard (drop order keeps the dir alive for the test).
    fn open_engine() -> (Arc<StorageEngine>, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path().to_string_lossy().as_ref(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            Arc::new(StorageEngine::open_with_oracle(&config, oracle).expect("open engine"));
        (engine, dir)
    }

    fn pipeline_for(engine: &Arc<StorageEngine>) -> Arc<dyn ProposalPipeline> {
        Arc::new(OwnedLocalProposalPipeline::new(engine))
    }

    /// The configured window flows through `main.rs`'s construction path to
    /// the engine GC watermark: with a window of W seconds, the watermark is
    /// held back to exactly `snapshot - W*1e6` (no consumers, no other pins).
    /// This is the end-to-end assertion the task requires, and it catches a
    /// seconds-to-microseconds conversion regression.
    #[tokio::test]
    async fn configured_window_drives_engine_gc_watermark() {
        let (engine, _dir) = open_engine();
        let pipeline = pipeline_for(&engine);

        let window_secs = 3_600u64; // 1 hour
        let _bg = build_consumer_registry(
            Arc::clone(&engine),
            pipeline,
            1,
            RegistryTuning {
                retention_window_secs: Some(window_secs),
                ..RegistryTuning::default()
            },
        );

        let snap = engine.snapshot();
        let expected = snap.saturating_sub(window_secs * US_PER_SEC);
        assert_eq!(
            engine.gc_watermark(),
            expected,
            "GC watermark must be snapshot - configured window"
        );
    }

    /// The override actually changes behaviour versus the built-in default:
    /// a short window retains less history, so it holds a strictly higher GC
    /// floor than the default 7-day window. Proves the flag is applied, not
    /// silently dropped. (HLC seqno is wall-clock microseconds, far larger
    /// than either window, so neither floor saturates to zero.)
    #[tokio::test]
    async fn short_window_keeps_higher_floor_than_default() {
        let (engine_short, _d1) = open_engine();
        let _bg_short = build_consumer_registry(
            Arc::clone(&engine_short),
            pipeline_for(&engine_short),
            1,
            RegistryTuning {
                retention_window_secs: Some(1), // 1 second
                ..RegistryTuning::default()
            },
        );
        let floor_short = engine_short.gc_watermark();

        let (engine_default, _d2) = open_engine();
        let _bg_default = build_consumer_registry(
            Arc::clone(&engine_default),
            pipeline_for(&engine_default),
            1,
            RegistryTuning::default(), // no override → 7-day window
        );
        let floor_default = engine_default.gc_watermark();

        assert!(
            floor_short > floor_default,
            "1s window (floor {floor_short}) must retain less than the 7-day \
             default (floor {floor_default})"
        );
    }
}
