//! Integration tests — oplog segment placement across multi-endpoint storage
//! configs.
//!
//! Pins the placement contract end-to-end:
//!
//! - **Oplog** segments land at `<oplog_eligible_endpoint>/oplog/<shard>/`.
//!   Eligibility: `durability ∈ {Durable, Degraded}`.
//! - **Volatile-only configs** open (the LSM data path may live on cache media)
//!   but cannot host an oplog — `select_oplog_endpoint` / `LogStore::open` must
//!   return a clear error, never a panic (INV-D1).
//!
//! Embedded durability is the retained oplog journal (`open_embedded`), which
//! lands at the same oplog-eligible endpoint; its crash-recovery and
//! WAL-replay-repair behaviour is covered by the storage-crate oplog-journal
//! tests and the embed `repair` tests.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use tempfile::TempDir;

/// Engine accessor: `select_oplog_endpoint(shard_id)` returns endpoints
/// in round-robin order across the eligible subset (skipping Volatile),
/// so different shards on the same node spread their oplog directories
/// across available durable endpoints.
#[test]
fn oplog_endpoint_round_robin_across_durable() {
    let a = TempDir::new().expect("a tempdir");
    let b = TempDir::new().expect("b tempdir");
    let cache = TempDir::new().expect("cache tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-a",
            a.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "ep-b",
            b.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
        EndpointConfig::new(
            "ep-cache",
            cache.path(),
            Media::Nvme,
            Durability::Volatile,
            Tier::HotCache,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // Shard 0 → first durable endpoint (a). Shard 1 → second (b).
    // Shard 2 → wraps back to a.
    assert_eq!(engine.select_oplog_endpoint(0).unwrap().id, "ep-a");
    assert_eq!(engine.select_oplog_endpoint(1).unwrap().id, "ep-b");
    assert_eq!(engine.select_oplog_endpoint(2).unwrap().id, "ep-a");

    // Cache endpoint is never selected for oplog.
    let eligible_ids: Vec<&str> = engine
        .all_oplog_eligible_endpoints()
        .iter()
        .map(|e| e.id.as_str())
        .collect();
    assert_eq!(eligible_ids, vec!["ep-a", "ep-b"]);
}

/// Volatile-only config: engine opens (data is allowed to live on RAM
/// or cache endpoints for the lsm-tree), but oplog selection MUST fail.
/// This pins the INV-D1 invariant at runtime — oplog needs persistence
/// even though the LSM data path doesn't.
#[test]
fn oplog_selection_errors_on_volatile_only_config() {
    let cache = TempDir::new().expect("cache tempdir");

    // `with_endpoints_no_persistence` is the explicit escape hatch for
    // configs without an oplog-eligible endpoint (in-memory tests).
    let config = StorageConfig::with_endpoints_no_persistence(vec![EndpointConfig::new(
        "ep-cache",
        cache.path(),
        Media::Nvme,
        Durability::Volatile,
        Tier::HotCache,
    )]);
    let engine = StorageEngine::open(&config).expect("open volatile-only succeeds");

    let result = engine.select_oplog_endpoint(0);
    let err = match result {
        Ok(_) => panic!("oplog must reject volatile-only config"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("oplog") || msg.contains("Oplog"),
        "error should mention oplog eligibility, got: {msg}"
    );
}

/// Runtime fail-safe: a caller that built the engine via
/// `with_endpoints_no_persistence` and then tries to open `LogStore`
/// (Raft log) MUST get a clear error mentioning oplog eligibility, not
/// a panic. This is the safety net that catches misuse of the
/// no-persistence escape hatch.
#[test]
fn logstore_open_errors_on_no_persistence_config() {
    use coordinode_raft::storage::LogStore;
    use std::sync::Arc;

    let cache = TempDir::new().expect("cache tempdir");
    // Explicit escape hatch — caller asserts they won't open Raft.
    let config = StorageConfig::with_endpoints_no_persistence(vec![EndpointConfig::new(
        "ep-cache",
        cache.path(),
        Media::Nvme,
        Durability::Volatile,
        Tier::HotCache,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));

    // Caller now violates their own assertion by opening Raft. The
    // runtime guard catches this with a clear oplog-eligibility error.
    let result = LogStore::open(Arc::clone(&engine));
    let err = match result {
        Ok(_) => panic!("LogStore::open must reject volatile-only config"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("oplog") || msg.contains("Oplog"),
        "error should mention oplog eligibility, got: {msg}",
    );
}

/// **INV-D1 (config-time):** `with_endpoints` MUST panic when given
/// only Volatile endpoints — oplog/Raft cannot survive on cache media.
/// Tests that genuinely want an all-Volatile config use the explicit
/// `with_endpoints_no_persistence` escape hatch (covered by the previous
/// test).
#[test]
#[should_panic(expected = "at least one oplog-eligible endpoint")]
fn with_endpoints_rejects_all_volatile_config_at_construction() {
    let cache = TempDir::new().expect("cache tempdir");
    let _ = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep-cache",
        cache.path(),
        Media::Nvme,
        Durability::Volatile,
        Tier::HotCache,
    )]);
}
