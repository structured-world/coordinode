use super::decompressing_reader;
use std::io::{Read, Write};
use std::sync::Arc;

use coordinode_core::txn::proposal::{Mutation, PartitionId};
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

#[test]
fn gzip_input_is_transparently_decompressed() {
    let plain = b"hello\nworld\n";
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(plain).unwrap();
    let gz = enc.finish().unwrap();
    let mut r = decompressing_reader(std::io::Cursor::new(gz)).unwrap();
    let mut out = Vec::new();
    r.read_to_end(&mut out).unwrap();
    assert_eq!(out, plain);
}

#[test]
fn uncompressed_input_passes_through() {
    let plain = b"{\"type\":\"node\"}\n";
    let mut r = decompressing_reader(std::io::Cursor::new(plain.to_vec())).unwrap();
    let mut out = Vec::new();
    r.read_to_end(&mut out).unwrap();
    assert_eq!(out, plain);
}

#[test]
fn zstd_magic_is_rejected_with_guidance() {
    let zstd = vec![0x28u8, 0xb5, 0x2f, 0xfd, 0, 0, 0, 0];
    match decompressing_reader(std::io::Cursor::new(zstd)) {
        Err(e) => assert!(e.to_string().contains("zstd"), "got: {e}"),
        Ok(_) => panic!("expected zstd rejection"),
    }
}

// ── Maintenance commands open the directory the way the server does ──

/// Wall-clock microseconds plus a wide margin: opening an engine allocates
/// timestamps and re-anchors the oracle to the wall clock, so a test that
/// drives the clock itself has to start in the future.
fn future_base() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros();
    u64::try_from(now).expect("fits") + 1_000_000_000_000
}

fn single_endpoint(path: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        path,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn put_at(engine: &StorageEngine, key: &[u8], value: &[u8], commit_ts: u64) {
    engine
        .apply_proposal_at(
            &[Mutation::Put {
                partition: PartitionId::Node,
                key: key.to_vec(),
                value: value.to_vec(),
            }],
            commit_ts,
        )
        .expect("apply");
}

/// `coordinode compact` must leave the history the window was holding.
///
/// The maintenance commands work on the directory a server writes, so they
/// have to open it the way the server does: with an oracle, because that is
/// what tells the engine its seqnos are a clock and makes the retention
/// window mean anything. Opened with a plain counter the window is inert,
/// the watermark jumps to the newest seqno, and the first compaction folds
/// every version underneath it: an operator command that silently ends time
/// travel over the whole window.
#[tokio::test]
async fn compact_keeps_the_time_travel_window() {
    const KEY: &[u8] = b"node:00:00000001";
    let base = future_base();
    let dir = tempfile::tempdir().expect("tempdir");
    let config = single_endpoint(dir.path());
    // Between the two writes and well inside the window: it resolves to v1.
    let inside = base + 2_500;

    {
        let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(base)));
        let engine = StorageEngine::open_with_oracle(&config, Arc::clone(&oracle)).expect("open");
        engine.set_retention_window(std::time::Duration::from_secs(3_600));

        // Each version seals into its own table, so the compaction below has
        // something to merge: a lone table is never rewritten.
        put_at(&engine, KEY, b"v1", base + 1_000);
        engine.persist().expect("seal v1");
        put_at(&engine, KEY, b"v2", base + 3_000);
        engine.persist().expect("seal v2");

        assert_eq!(
            engine
                .snapshot_get(&inside, Partition::Node, KEY)
                .expect("read")
                .map(|v| v.to_vec()),
            Some(b"v1".to_vec()),
            "the window covers the older version before the maintenance run"
        );
        engine.persist().expect("persist");
    }

    crate::run_with(
        crate::ServerBuilder::new(),
        crate::cli::Command::Compact {
            data_dir: dir.path().display().to_string(),
            config_path: None,
        },
    )
    .await
    .expect("compact");

    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(
        base + 4_000,
    )));
    let engine = StorageEngine::open_with_oracle(&config, oracle).expect("reopen");
    engine.set_retention_window(std::time::Duration::from_secs(3_600));

    assert_eq!(
        engine
            .snapshot_get(&inside, Partition::Node, KEY)
            .expect("read inside the window after the maintenance run")
            .map(|v| v.to_vec()),
        Some(b"v1".to_vec()),
        "a compaction run by the maintenance command folded history the window holds"
    );
    assert_eq!(
        engine
            .snapshot_get(&(base + 3_500), Partition::Node, KEY)
            .expect("read")
            .map(|v| v.to_vec()),
        Some(b"v2".to_vec()),
        "the live version survives the maintenance run"
    );
}
