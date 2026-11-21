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

// ── Backup and restore of one namespace ──

/// A logical backup naming a namespace exports that namespace alone, and a
/// restore naming one writes it, created in a store that holds none: two
/// namespaces' rows never cross. A store of several namespaces is not backed
/// up without naming one, and a raft-snapshot backup or restore, which holds
/// the whole deployment, takes no namespace.
#[test]
fn a_logical_backup_and_restore_carry_one_namespace() {
    use coordinode_embed::backup::BackupFormat;
    let source = tempfile::tempdir().expect("source");
    {
        let database = coordinode_embed::Database::open(source.path()).expect("open");
        database.create_namespace("a").expect("create a");
        database.create_namespace("b").expect("create b");
        for (name, value) in [("a", 1), ("b", 2)] {
            database
                .session(name)
                .expect("session")
                .execute_cypher(&format!("CREATE (:Owned {{v: {value}}})"))
                .expect("write");
        }
    }
    let data_dir = source.path().display().to_string();
    let out = tempfile::tempdir().expect("out");
    let file = out.path().join("a.json").display().to_string();
    super::run_backup(
        data_dir.clone(),
        None,
        file.clone(),
        BackupFormat::Json,
        Some("a".into()),
    )
    .expect("backup of a");
    assert!(
        super::run_backup(
            data_dir.clone(),
            None,
            out.path().join("all.json").display().to_string(),
            BackupFormat::Json,
            None,
        )
        .is_err(),
        "a store of several namespaces is backed up one namespace at a time"
    );
    assert!(
        super::run_backup(
            data_dir,
            None,
            out.path().join("a.snap").display().to_string(),
            BackupFormat::RaftSnapshot,
            Some("a".into()),
        )
        .is_err()
    );

    let target = tempfile::tempdir().expect("target");
    super::run_restore(
        target.path().display().to_string(),
        None,
        file.clone(),
        BackupFormat::Json,
        Some("c".into()),
        Vec::new(),
        false,
    )
    .expect("restore into c");
    assert!(
        super::run_restore(
            target.path().display().to_string(),
            None,
            file,
            BackupFormat::RaftSnapshot,
            Some("c".into()),
            Vec::new(),
            false,
        )
        .is_err()
    );
    let mut restored =
        coordinode_embed::Database::open_namespace(single_endpoint(target.path()), "c", false)
            .expect("open c");
    let rows = restored
        .execute_cypher("MATCH (n:Owned) RETURN n.v AS v")
        .expect("read");
    assert_eq!(rows.len(), 1, "a's row alone");
    assert_eq!(
        rows[0].get("v"),
        Some(&coordinode_core::graph::types::Value::Int(1))
    );
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
            &coordinode_storage::engine::scope::store_unit([Mutation::Put {
                partition: PartitionId::Node,
                key: key.to_vec(),
                value: value.to_vec(),
            }]),
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
