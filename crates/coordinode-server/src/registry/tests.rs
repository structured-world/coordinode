use super::*;
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_raft::proposal::OwnedLocalProposalPipeline;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::oplog::entry::{OplogEntry, OplogOp};
use coordinode_storage::oplog::manager::OplogManager;

/// Microseconds per second: the engine window is operator-facing in seconds
/// and lives in HLC microseconds.
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

/// The floor of a log that holds no segment.
fn empty_floor() -> RetainedFloor {
    let dir = tempfile::tempdir().expect("tempdir");
    OplogManager::open(dir.path(), 0, 1 << 20, 1_000, 3_600)
        .expect("open oplog")
        .retained_floor()
}

/// A source over `engine` with no log segments and nothing applied.
fn empty_source(engine: &Arc<StorageEngine>) -> Arc<dyn RetentionSource> {
    Arc::new(NodeRetentionSource::new(
        Arc::clone(engine),
        Vec::new(),
        empty_floor(),
        Arc::new(|| 0),
    ))
}

/// Write entries `0..count` into `dir`, sealing a segment after each index in
/// `seal_after`; entry `i` carries timestamp `1_000_000 * (i + 1)` µs.
fn write_log(dir: &std::path::Path, count: u64, seal_after: &[u64]) -> OplogManager {
    let mut mgr =
        OplogManager::open(dir, 0, 64 * 1024 * 1024, 50_000, 7 * 24 * 3600).expect("open oplog");
    for index in 0..count {
        mgr.append(&OplogEntry {
            ts: US_PER_SEC * (index + 1),
            term: 1,
            index,
            shard: 0,
            ops: vec![OplogOp::Insert {
                partition: 0,
                key: format!("node:{index}").into_bytes(),
                value: vec![0u8; 64],
            }],
            is_migration: false,
            pre_images: None,
        })
        .expect("append");
        if seal_after.contains(&index) {
            mgr.rotate().expect("seal");
        }
    }
    mgr.flush().expect("flush");
    mgr
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

    let _bg = build_consumer_registry(
        Arc::clone(&engine),
        pipeline,
        empty_source(&engine),
        RegistryTuning::default(),
    );

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
        empty_source(&engine_short),
        RegistryTuning::default(),
    );
    let floor_short = engine_short.gc_watermark();

    let (engine_default, _d2) = open_engine(None); // 7-day default
    let _bg_default = build_consumer_registry(
        Arc::clone(&engine_default),
        pipeline_for(&engine_default),
        empty_source(&engine_default),
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
        empty_source(&engine),
        RegistryTuning {
            heartbeat_window_ms: Some(250),
            eviction_interval_ms: Some(5_000),
        },
    );
    assert_eq!(bg.config().heartbeat_window_ms, 250);
    assert_eq!(bg.config().eviction_interval_ms, 5_000);
}

/// The node source answers for the oplog from the segments it holds: the
/// oldest retained index after a purge, the applied frontier as the head,
/// the write time of an entry in milliseconds, and the bytes a consumer at a
/// position keeps on disk, which shrink as it advances.
#[test]
fn node_source_measures_the_oplog_it_holds() {
    let (engine, _engine_dir) = open_engine(None);
    let log_dir = tempfile::tempdir().expect("log dir");
    let mut mgr = write_log(log_dir.path(), 9, &[2, 5]);
    assert_eq!(mgr.purge_before(3).expect("purge"), 1);
    let source = NodeRetentionSource::new(
        Arc::clone(&engine),
        vec![log_dir.path().to_path_buf()],
        mgr.retained_floor(),
        Arc::new(|| 9),
    );
    let kind = ConsumerKind::OplogEvents;

    assert!(source.accounts(kind));
    assert_eq!(source.head(kind), 9);
    assert_eq!(source.first_retained(kind), 3);
    assert_eq!(source.produced_at_ms(kind, 4), Some(5_000));
    assert_eq!(
        source.produced_at_ms(kind, 9),
        None,
        "nothing written there yet"
    );

    let from_oldest = source.retained_bytes_from(kind, 3).expect("bytes");
    let from_last_segment = source.retained_bytes_from(kind, 6).expect("bytes");
    assert!(
        from_oldest > from_last_segment,
        "{from_oldest} vs {from_last_segment}"
    );
    assert!(from_last_segment > 0, "the open segment is still kept");
}

/// A log with no segment keeps nothing below the head, so a new consumer
/// cannot claim a position under it.
#[test]
fn node_source_without_segments_retains_nothing_below_the_head() {
    let (engine, _engine_dir) = open_engine(None);
    let log_dir = tempfile::tempdir().expect("log dir");
    let mgr = OplogManager::open(log_dir.path(), 0, 1 << 20, 1_000, 3_600).expect("open oplog");
    let source = NodeRetentionSource::new(
        Arc::clone(&engine),
        vec![log_dir.path().to_path_buf()],
        mgr.retained_floor(),
        Arc::new(|| 4),
    );
    assert_eq!(source.first_retained(ConsumerKind::OplogEvents), 4);
}

/// The MVCC store keeps no write time or size per version, so the source
/// does not account seqno-space consumers: BOUNDED is refused for them
/// rather than enforced on numbers that do not exist.
#[test]
fn node_source_does_not_account_mvcc_consumers() {
    let (engine, _engine_dir) = open_engine(None);
    let source = NodeRetentionSource::new(
        Arc::clone(&engine),
        Vec::new(),
        empty_floor(),
        Arc::new(|| 0),
    );
    let kind = ConsumerKind::LsmStateDelta;
    assert!(kind.is_seqno_space());
    assert!(!source.accounts(kind));
    assert_eq!(source.head(kind), engine.snapshot());
    assert_eq!(source.first_retained(kind), engine.gc_watermark());
    assert_eq!(source.produced_at_ms(kind, 1), None);
    assert_eq!(source.retained_bytes_from(kind, 1), None);
}
