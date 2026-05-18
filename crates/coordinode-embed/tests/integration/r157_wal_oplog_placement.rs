//! Integration tests — WAL + oplog segment placement across
//! multi-endpoint storage configs.
//!
//! Pins the placement contract end-to-end:
//!
//! - **WAL** lands at `<wal_eligible_endpoint>/wal/standalone.wal` when
//!   `WalConfig.path` is not overridden. Eligibility: fast media (NVMe/SSD)
//!   OR Hot/HotCache tier, AND `durability != Volatile`.
//! - **Oplog** segments land at `<oplog_eligible_endpoint>/oplog/<shard>/`.
//!   Eligibility: `durability ∈ {Durable, Degraded}`.
//! - **Selection error** when no eligible endpoint exists for a subsystem
//!   surfaces as an `open_with_wal`/`LogStore::open` failure (not a panic).
//! - **Volatile-only configs** open without WAL but cannot host oplog.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::wal::{WalConfig, WalSyncPolicy};
use tempfile::TempDir;

/// WAL placement: with two endpoints (slow cold HDD + fast hot NVMe),
/// the WAL file MUST be created under the fast NVMe endpoint, never the
/// cold HDD — even though both come before in topology iteration order
/// only the second satisfies eligibility.
#[test]
fn wal_lands_on_first_wal_eligible_endpoint() {
    let cold = TempDir::new().expect("cold tempdir");
    let hot = TempDir::new().expect("hot tempdir");

    // First endpoint is HDD-Warm → NOT WAL-eligible (neither fast media
    // nor Hot tier). Second endpoint is NVMe-Hot → eligible.
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        ),
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
    ]);

    let wal_cfg = WalConfig {
        path: None,
        sync: WalSyncPolicy::NoSync,
    };
    let engine =
        StorageEngine::open_with_wal(&config, Some(wal_cfg)).expect("open with WAL succeeds");
    engine.put(Partition::Node, b"k", b"v").expect("write");
    drop(engine);

    let expected = hot.path().join("wal").join("standalone.wal");
    assert!(
        expected.exists(),
        "WAL must land at hot-tier endpoint, expected {expected:?}",
    );
    let forbidden = cold.path().join("wal").join("standalone.wal");
    assert!(
        !forbidden.exists(),
        "WAL must NOT appear at cold (non-eligible) endpoint {forbidden:?}",
    );
}

/// WAL placement: explicit `wal_cfg.path` override wins — placement
/// engine never sees the call when caller pre-decides where the WAL
/// lives. Pins the "tests and unusual deployments may override" contract.
#[test]
fn explicit_wal_path_overrides_endpoint_selection() {
    let nvme = TempDir::new().expect("nvme tempdir");
    let explicit = TempDir::new().expect("explicit tempdir");
    let explicit_wal_path = explicit.path().join("custom.wal");

    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep-nvme",
        nvme.path(),
        Media::Nvme,
        Durability::Durable,
        Tier::Hot,
    )]);

    let wal_cfg = WalConfig {
        path: Some(explicit_wal_path.clone()),
        sync: WalSyncPolicy::NoSync,
    };
    let engine = StorageEngine::open_with_wal(&config, Some(wal_cfg)).expect("open");
    engine.put(Partition::Node, b"k", b"v").expect("write");
    drop(engine);

    assert!(explicit_wal_path.exists(), "explicit WAL path must exist");
    let auto = nvme.path().join("wal").join("standalone.wal");
    assert!(
        !auto.exists(),
        "auto-selected WAL path must NOT be created when caller overrides"
    );
}

/// WAL eligibility error: when no endpoint satisfies WAL eligibility,
/// `open_with_wal` MUST return an error (not panic, not silently fall
/// back to a non-eligible path). Volatile NVMe alone is not enough.
#[test]
fn wal_open_errors_when_no_eligible_endpoint() {
    let vol = TempDir::new().expect("vol tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        // Volatile fast media → ineligible.
        EndpointConfig::new(
            "ep-cache",
            vol.path(),
            Media::Nvme,
            Durability::Volatile,
            Tier::HotCache,
        ),
        // Durable slow media in Cold tier → ineligible.
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);

    let wal_cfg = WalConfig {
        path: None,
        sync: WalSyncPolicy::NoSync,
    };
    let result = StorageEngine::open_with_wal(&config, Some(wal_cfg));
    let err = match result {
        Ok(_) => panic!("must fail without eligible endpoint"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("WAL") || msg.contains("wal") || msg.contains("non-volatile"),
        "error should mention WAL eligibility, got: {msg}"
    );
}

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

/// Orphan WAL behaviour: when a previous config wrote a WAL at endpoint
/// A and the new config still includes A as a WAL-eligible endpoint but
/// routes the active WAL to endpoint B, the file at A MUST be left
/// untouched (not replayed, not deleted) — engine only warns. This pins
/// the "lineage ambiguous, do not auto-replay" contract.
#[test]
fn orphan_wal_on_other_endpoint_is_left_untouched() {
    let ep_a = TempDir::new().expect("ep_a tempdir");
    let ep_b = TempDir::new().expect("ep_b tempdir");

    // Plant a sentinel WAL file at endpoint A under wal/standalone.wal
    // pretending a previous deployment had A as the active WAL endpoint.
    let orphan_path = ep_a.path().join("wal").join("standalone.wal");
    std::fs::create_dir_all(orphan_path.parent().expect("parent")).expect("mkdir");
    std::fs::write(&orphan_path, b"OLD_WAL_DO_NOT_REPLAY").expect("plant orphan");

    // New config: both A and B are WAL-eligible, but B is listed first
    // → active WAL routes to B (selection picks first eligible). A's
    // WAL becomes orphan from the engine's perspective.
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-b",
            ep_b.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "ep-a",
            ep_a.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
    ]);
    let wal_cfg = WalConfig {
        path: None,
        sync: WalSyncPolicy::NoSync,
    };
    let engine = StorageEngine::open_with_wal(&config, Some(wal_cfg)).expect("open succeeds");
    engine.put(Partition::Node, b"k", b"v").expect("write");
    drop(engine);

    // Active WAL was created at endpoint B.
    let active = ep_b.path().join("wal").join("standalone.wal");
    assert!(active.exists(), "active WAL must be created at ep_b");

    // **Critical invariant**: orphan file untouched — same path, same
    // bytes, not opened-then-truncated, not deleted.
    assert!(orphan_path.exists(), "orphan WAL must NOT be deleted");
    let orphan_bytes = std::fs::read(&orphan_path).expect("read orphan");
    assert_eq!(
        &orphan_bytes[..],
        b"OLD_WAL_DO_NOT_REPLAY",
        "orphan WAL contents must be byte-identical (engine did not open or rewrite it)",
    );
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

/// End-to-end WAL replay through multi-endpoint placement: write through
/// the engine with WAL enabled, drop without an SST flush, reopen, verify
/// the data survives. Proves the multi-endpoint WAL path is not just file
/// placement — recovery actually finds the file at the new location and
/// replays it into the memtable on reopen.
///
/// This is the **semantic** test the placement tests above cannot give —
/// they verify only that the file lands where expected, not that the
/// next open recovers from it.
#[test]
fn wal_replay_survives_reopen_through_multi_endpoint_placement() {
    let cold = TempDir::new().expect("cold tempdir");
    let hot = TempDir::new().expect("hot tempdir");

    let make_config = || {
        StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "ep-cold",
                cold.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Warm,
            ),
            EndpointConfig::new(
                "ep-hot",
                hot.path(),
                Media::Nvme,
                Durability::Durable,
                Tier::Hot,
            ),
        ])
    };
    let wal_cfg = || WalConfig {
        path: None,
        // SyncPerRecord ensures WAL is fsynced before drop — recovery
        // depends on the bytes actually being on disk.
        sync: WalSyncPolicy::SyncPerRecord,
    };

    // ── First lifecycle: open, write, drop (no SST flush) ──
    {
        let engine =
            StorageEngine::open_with_wal(&make_config(), Some(wal_cfg())).expect("first open");
        engine
            .put(Partition::Node, b"survives", b"replay_me")
            .expect("write through WAL");
        // Drop without explicit flush — memtable is NOT persisted as
        // SST, only the WAL record on disk.
    }

    // Sanity: the WAL physically landed at the hot endpoint.
    let wal_at_hot = hot.path().join("wal").join("standalone.wal");
    assert!(
        wal_at_hot.exists(),
        "WAL must be at hot endpoint, was {wal_at_hot:?}",
    );

    // ── Second lifecycle: reopen → WAL replay must restore memtable ──
    let engine =
        StorageEngine::open_with_wal(&make_config(), Some(wal_cfg())).expect("reopen with replay");
    let value = engine
        .get(Partition::Node, b"survives")
        .expect("read after replay")
        .expect("key must survive WAL replay");
    assert_eq!(
        &value[..],
        b"replay_me",
        "WAL replay must restore the pre-drop write",
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
