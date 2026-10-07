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

use std::path::PathBuf;
use std::sync::Arc;

use coordinode_core::txn::proposal::{ProposalIdGenerator, ProposalPipeline};
use coordinode_replicate::{
    BackgroundConfig, ConsumerKind, RegistryBackground, RetentionSource, ShardConsumerRegistry,
    SystemClock,
};
use coordinode_storage::engine::core::{StorageEngine, WritePressure};
use coordinode_storage::oplog::OplogTimeline;
use coordinode_storage::oplog::RetainedFloor;
use coordinode_storage::oplog::tailer::bytes_needed_from;

/// The history this node holds for its consumers: the Raft log for oplog
/// consumers, the MVCC store for the others.
pub(crate) struct NodeRetentionSource {
    engine: Arc<StorageEngine>,
    /// Every directory holding segments of the shard's Raft log.
    oplog_dirs: Vec<PathBuf>,
    /// Where the log's segments start, published by the log itself: every
    /// change stream asks on each wake, so it must not list the directories.
    floor: RetainedFloor,
    /// One past the last log entry this node has applied.
    applied: Arc<dyn Fn() -> u64 + Send + Sync>,
    /// When each log entry was written: the bound checks ask the age of the
    /// same acknowledged positions every sweep, and reading the segment that
    /// holds one would cost its size on every ask.
    timeline: OplogTimeline,
}

impl NodeRetentionSource {
    /// A source over `engine`, the log segments in `oplog_dirs` starting at
    /// `floor`, and the applied frontier `applied` reports.
    pub(crate) fn new(
        engine: Arc<StorageEngine>,
        oplog_dirs: Vec<PathBuf>,
        floor: RetainedFloor,
        applied: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Self {
        Self {
            engine,
            timeline: OplogTimeline::new(oplog_dirs.clone()),
            oplog_dirs,
            floor,
            applied,
        }
    }
}

impl RetentionSource for NodeRetentionSource {
    fn head(&self, kind: ConsumerKind) -> u64 {
        if kind.is_seqno_space() {
            self.engine.snapshot()
        } else {
            (self.applied)()
        }
    }

    fn first_retained(&self, kind: ConsumerKind) -> u64 {
        if kind.is_seqno_space() {
            return self.engine.gc_watermark();
        }
        match self.floor.get() {
            Some(first) => first,
            // No segment holds anything: nothing below the head is kept.
            None => self.head(kind),
        }
    }

    fn accounts(&self, kind: ConsumerKind) -> bool {
        // The log knows when each entry was written and how big its segments
        // are; the MVCC store keeps neither per version.
        !kind.is_seqno_space()
    }

    fn produced_at_ms(&self, kind: ConsumerKind, position: u64) -> Option<u64> {
        if kind.is_seqno_space() {
            return None;
        }
        match self.timeline.written_at(position) {
            // An entry's timestamp is wall-clock microseconds.
            Ok(ts) => ts.map(|ts| ts / 1_000),
            Err(e) => {
                tracing::warn!(error = %e, position, "registry: oplog entry time unreadable");
                None
            }
        }
    }

    fn release_log_below(&self, position: u64) {
        self.timeline.release_below(position);
    }

    fn retained_bytes_from(&self, kind: ConsumerKind, position: u64) -> Option<u64> {
        if kind.is_seqno_space() {
            return None;
        }
        match bytes_needed_from(&self.oplog_dirs, position) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                tracing::warn!(error = %e, "registry: oplog size unreadable");
                None
            }
        }
    }

    fn admits(&self, _kind: ConsumerKind) -> bool {
        // The log and the MVCC store share the engine's endpoints, so one
        // verdict covers both: Stop means compaction is behind on reclaiming.
        !matches!(self.engine.write_pressure(), WritePressure::Stop)
    }
}

/// Operator-supplied registry tuning, parsed from the config file.
///
/// Every field is `Option`: `None` means "keep the registry's built-in
/// default", so an operator overrides only what they explicitly set.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RegistryTuning {
    /// Heartbeat coalescing window in milliseconds (`registry_heartbeat_ms`).
    /// `None` keeps the default 1000 ms.
    pub heartbeat_window_ms: Option<u64>,
    /// Shortest gap between TTL-eviction sweeps in milliseconds
    /// (`registry_eviction_ms`). `None` keeps the default 1000 ms.
    pub eviction_interval_ms: Option<u64>,
}

/// Build the per-shard consumer-retention registry and start its
/// background service, applying any operator overrides from `tuning`.
///
/// Returns the live [`ShardConsumerRegistry`] (cheap to clone — `Arc`-backed —
/// so producers like the change-stream service register through it) and its
/// [`RegistryBackground`] handle, which must be held for the process lifetime:
/// dropping it shuts the background service down (with a final heartbeat flush).
pub(crate) fn build_consumer_registry(
    engine: Arc<StorageEngine>,
    pipeline: Arc<dyn ProposalPipeline>,
    source: Arc<dyn RetentionSource>,
    tuning: RegistryTuning,
) -> (ShardConsumerRegistry, RegistryBackground) {
    let registry = ShardConsumerRegistry::new(
        engine,
        pipeline,
        Arc::new(ProposalIdGenerator::with_base(
            coordinode_core::txn::proposal::fresh_proposal_id_base(),
        )),
        Arc::new(SystemClock),
        source,
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
