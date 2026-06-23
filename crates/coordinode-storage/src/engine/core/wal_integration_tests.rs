//! Integration tests for standalone WAL crash recovery.
//!
//! These tests validate the full round-trip:
//!   open_with_wal → wal_append → drop (simulated crash) → open_with_wal → verify recovery.
//!
//! Unlike unit tests in `wal/mod.rs` (which test the WAL file directly),
//! these tests go through `StorageEngine::open_with_wal` and verify that
//! data written to the WAL is visible after engine reopen.

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, Tier};
use coordinode_core::txn::proposal::{Mutation, PartitionId};
use tempfile::TempDir;

/// Returns a `StorageConfig` pointing at `dir` with WAL enabled (WAL path = `dir/standalone.wal`).
fn cfg_with_wal(dir: &TempDir) -> (StorageConfig, WalConfig) {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let wal_config = WalConfig {
        path: Some(dir.path().join("standalone.wal")),
        sync: crate::wal::WalSyncPolicy::SyncPerRecord,
    };
    (config, wal_config)
}

#[test]
fn wal_recovery_put_survives_crash() {
    // Verifies that a Put mutation written to the WAL (but not yet flushed to SST)
    // is recovered after the engine is dropped (simulating a crash) and reopened.
    let dir = TempDir::new().expect("temp dir");
    let (config, wal_config) = cfg_with_wal(&dir);

    // Phase 1: write to WAL, then "crash" (drop without persist).
    {
        let engine = StorageEngine::open_with_wal(&config, Some(wal_config.clone())).expect("open");
        assert!(engine.has_wal());

        // Write through WAL — must NOT call persist() so data stays only in WAL.
        let mutations = vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:00:00000042".to_vec(),
            value: b"crashed-payload".to_vec(),
        }];
        engine.wal_append(&mutations).expect("wal_append");

        // Apply to memtable (mirrors what OwnedLocalProposalPipeline does).
        engine
            .put(Partition::Node, b"node:00:00000042", b"crashed-payload")
            .expect("put");

        // Drop engine WITHOUT calling persist() — simulates crash.
        // Data is in memtable + WAL but NOT in any SST.
    }

    // Phase 2: reopen with WAL — recovery must replay the WAL record.
    {
        let engine = StorageEngine::open_with_wal(&config, Some(wal_config)).expect("reopen");

        let val = engine
            .get(Partition::Node, b"node:00:00000042")
            .expect("get after recovery");

        assert_eq!(
            val.as_deref(),
            Some(b"crashed-payload".as_slice()),
            "WAL recovery must restore Put written before crash"
        );
    }
}

#[test]
fn wal_recovery_delete_survives_crash() {
    // Verifies that a Delete mutation in the WAL (after a prior Put in SST)
    // is replayed correctly: the key must be absent after recovery.
    let dir = TempDir::new().expect("temp dir");
    let (config, wal_config) = cfg_with_wal(&dir);

    // Phase 1: persist a key to SST.
    {
        let engine = StorageEngine::open_with_wal(&config, Some(wal_config.clone())).expect("open");
        engine
            .put(Partition::Node, b"node:00:deadbeef", b"to-be-deleted")
            .expect("put");
        engine.persist().expect("persist");
        // persist() also checkpoints the WAL — WAL is now clean.
    }

    // Phase 2: delete via WAL, then crash.
    {
        let engine =
            StorageEngine::open_with_wal(&config, Some(wal_config.clone())).expect("open2");

        let mutations = vec![Mutation::Delete {
            partition: PartitionId::Node,
            key: b"node:00:deadbeef".to_vec(),
        }];
        engine.wal_append(&mutations).expect("wal_append");
        engine
            .delete(Partition::Node, b"node:00:deadbeef")
            .expect("delete");
        // Crash — no persist().
    }

    // Phase 3: reopen — delete must be replayed, key must be gone.
    {
        let engine = StorageEngine::open_with_wal(&config, Some(wal_config)).expect("reopen");

        let val = engine
            .get(Partition::Node, b"node:00:deadbeef")
            .expect("get after recovery");

        assert!(
            val.is_none(),
            "WAL recovery must replay Delete: key must be absent after crash"
        );
    }
}

#[test]
fn wal_recovery_multiple_mutations_across_two_writes() {
    // Writes multiple WAL records (two separate wal_append calls), crashes,
    // reopens, and verifies ALL records are recovered in order.
    let dir = TempDir::new().expect("temp dir");
    let (config, wal_config) = cfg_with_wal(&dir);

    {
        let engine = StorageEngine::open_with_wal(&config, Some(wal_config.clone())).expect("open");

        // First WAL record: two puts.
        let batch1 = vec![
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:00000001".to_vec(),
                value: b"alpha".to_vec(),
            },
            Mutation::Put {
                partition: PartitionId::Node,
                key: b"node:00:00000002".to_vec(),
                value: b"beta".to_vec(),
            },
        ];
        engine.wal_append(&batch1).expect("wal_append batch1");
        for m in &batch1 {
            if let Mutation::Put {
                partition,
                key,
                value,
            } = m
            {
                engine
                    .put(Partition::from(*partition), key, value)
                    .expect("put");
            }
        }

        // Second WAL record: one more put.
        let batch2 = vec![Mutation::Put {
            partition: PartitionId::Schema,
            key: b"schema:label:Person".to_vec(),
            value: b"{}".to_vec(),
        }];
        engine.wal_append(&batch2).expect("wal_append batch2");
        engine
            .put(Partition::Schema, b"schema:label:Person", b"{}")
            .expect("put schema");

        // Crash.
    }

    // Reopen and verify all three keys recovered.
    {
        let engine = StorageEngine::open_with_wal(&config, Some(wal_config)).expect("reopen");

        let v1 = engine
            .get(Partition::Node, b"node:00:00000001")
            .expect("get 1");
        let v2 = engine
            .get(Partition::Node, b"node:00:00000002")
            .expect("get 2");
        let v3 = engine
            .get(Partition::Schema, b"schema:label:Person")
            .expect("get 3");

        assert_eq!(
            v1.as_deref(),
            Some(b"alpha".as_slice()),
            "key 1 must recover"
        );
        assert_eq!(
            v2.as_deref(),
            Some(b"beta".as_slice()),
            "key 2 must recover"
        );
        assert_eq!(v3.as_deref(), Some(b"{}".as_slice()), "key 3 must recover");
    }
}

#[test]
fn wal_checkpoint_on_persist_clears_wal_file() {
    // Verifies that calling persist() on an engine with WAL:
    //   1. Flushes memtable to SST.
    //   2. Checkpoints (rotates) the WAL so the file is empty.
    // After reopen, no replay happens (WAL is clean) and data is still readable from SST.
    let dir = TempDir::new().expect("temp dir");
    let (config, wal_config) = cfg_with_wal(&dir);

    {
        let engine = StorageEngine::open_with_wal(&config, Some(wal_config.clone())).expect("open");

        let mutations = vec![Mutation::Put {
            partition: PartitionId::Node,
            key: b"node:00:persisted".to_vec(),
            value: b"safe".to_vec(),
        }];
        engine.wal_append(&mutations).expect("wal_append");
        engine
            .put(Partition::Node, b"node:00:persisted", b"safe")
            .expect("put");

        // persist() must flush to SST AND checkpoint the WAL.
        engine.persist().expect("persist");

        // WAL file must be empty (or minimal header) after checkpoint.
        let wal_path = dir.path().join("standalone.wal");
        let wal_size = std::fs::metadata(&wal_path)
            .expect("wal file must exist after persist")
            .len();
        assert_eq!(
            wal_size, 0,
            "WAL must be empty after checkpoint via persist()"
        );
    }

    // Reopen: no WAL replay needed (SST has the data), key still readable.
    {
        let engine = StorageEngine::open_with_wal(&config, Some(wal_config)).expect("reopen");

        let val = engine
            .get(Partition::Node, b"node:00:persisted")
            .expect("get");
        assert_eq!(
            val.as_deref(),
            Some(b"safe".as_slice()),
            "data persisted to SST must survive WAL checkpoint + reopen"
        );
    }
}

#[test]
fn no_wal_engine_wal_append_returns_none() {
    // Verifies that wal_append() on a plain open() engine (no WAL) returns None
    // without error — caller can use this to branch between WAL and legacy paths.
    let dir = TempDir::new().expect("temp dir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open without wal");

    assert!(!engine.has_wal(), "plain open must have no WAL");

    let mutations = vec![Mutation::Put {
        partition: PartitionId::Node,
        key: b"node:00:noop".to_vec(),
        value: b"x".to_vec(),
    }];
    let result = engine
        .wal_append(&mutations)
        .expect("wal_append must not error without wal");
    assert!(result.is_none(), "wal_append without WAL must return None");
}
