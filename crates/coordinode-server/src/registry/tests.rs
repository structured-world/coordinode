use super::*;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

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
    let engine = Arc::new(StorageEngine::open_with_oracle(&config, oracle).expect("open engine"));
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
