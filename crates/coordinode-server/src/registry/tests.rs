use super::*;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

/// Microseconds per second: the engine window is operator-facing in seconds
/// and lives in HLC microseconds (ADR-007).
const US_PER_SEC: u64 = 1_000_000;

/// Open a fresh single-endpoint engine in a temp directory with the given
/// retention window. Returns the engine and its dir guard (drop order keeps
/// the dir alive for the test).
fn open_engine(retention_window_secs: Option<u64>) -> (Arc<StorageEngine>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path().to_string_lossy().as_ref(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    if let Some(secs) = retention_window_secs {
        config.retention_window_secs = secs;
    }
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle).expect("open engine"));
    (engine, dir)
}

fn pipeline_for(engine: &Arc<StorageEngine>) -> Arc<dyn ProposalPipeline> {
    Arc::new(OwnedLocalProposalPipeline::new(engine))
}

/// The engine window survives registry construction: with no consumers the
/// registry publishes no consumer floor, so the GC watermark stays exactly
/// `snapshot - window` (the engine's own floor), never `u64::MAX` (which
/// would collect the whole `AS OF TIMESTAMP` history) and never the current
/// seqno (fold-all).
#[tokio::test]
async fn registry_leaves_the_engine_window_in_force() {
    let window_secs = 3_600u64; // 1 hour
    let (engine, _dir) = open_engine(Some(window_secs));
    let pipeline = pipeline_for(&engine);

    let _bg = build_consumer_registry(Arc::clone(&engine), pipeline, 1, RegistryTuning::default());

    let snap = engine.snapshot();
    let expected = snap.saturating_sub(window_secs * US_PER_SEC);
    assert_eq!(
        engine.gc_watermark(),
        expected,
        "GC watermark must be snapshot - configured window"
    );
}

/// The window is an engine setting, so different windows on two engines give
/// different floors regardless of the registry: a short window retains less
/// history and holds a strictly higher GC floor than the seven-day default.
/// (HLC seqno is wall-clock microseconds, far larger than either window, so
/// neither floor saturates to zero.)
#[tokio::test]
async fn short_window_keeps_higher_floor_than_default() {
    let (engine_short, _d1) = open_engine(Some(1)); // 1 second
    let _bg_short = build_consumer_registry(
        Arc::clone(&engine_short),
        pipeline_for(&engine_short),
        1,
        RegistryTuning::default(),
    );
    let floor_short = engine_short.gc_watermark();

    let (engine_default, _d2) = open_engine(None); // 7-day default
    let _bg_default = build_consumer_registry(
        Arc::clone(&engine_default),
        pipeline_for(&engine_default),
        1,
        RegistryTuning::default(),
    );
    let floor_default = engine_default.gc_watermark();

    assert!(
        floor_short > floor_default,
        "1s window (floor {floor_short}) must retain less than the 7-day \
             default (floor {floor_default})"
    );
}

/// Background cadence overrides are applied, not silently dropped: the
/// registry service starts with the operator values.
#[tokio::test]
async fn cadence_overrides_reach_the_background_service() {
    let (engine, _dir) = open_engine(None);
    let (_registry, bg) = build_consumer_registry(
        Arc::clone(&engine),
        pipeline_for(&engine),
        1,
        RegistryTuning {
            heartbeat_window_ms: Some(250),
            eviction_interval_ms: Some(5_000),
        },
    );
    assert_eq!(bg.config().heartbeat_window_ms, 250);
    assert_eq!(bg.config().eviction_interval_ms, 5_000);
}
