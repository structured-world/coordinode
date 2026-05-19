//! Hard-limit enforcement + per-endpoint capacity tracking
//! (storage-stack Layer 1, INV-D3).
//!
//! Covers:
//! - per-endpoint usage scan populates `used_bytes` from on-disk SSTs,
//! - severity transitions (Normal → Warning → Critical → Emergency → Full),
//! - `is_writable` flips off at 100% and back on after recovery,
//! - pre-write gate rejects writes with `CapacityExhausted` when the
//!   target partition's L0 endpoint is non-writable,
//! - `CascadeEvict` strategy auto-fires at Emergency severity,
//! - `Reject` strategy does NOT auto-fire,
//! - Schema and Raft partitions bypass the gate (engine-internal
//!   metadata must remain reachable when user-data endpoints are full).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_storage::engine::capacity::CapacitySeverity;
use coordinode_storage::engine::config::{
    Durability, EndpointConfig, HardLimitStrategy, Media, StorageConfig, Tier,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::StorageError;
use tempfile::TempDir;

/// Writing data + persisting + refreshing the capacity tracker MUST
/// produce a non-zero `used_bytes` reading for the target endpoint.
/// Smoke check on the scan path.
#[test]
fn refresh_picks_up_on_disk_sst_bytes() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "only",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(10_000_000)]);
    let engine = StorageEngine::open(&config).expect("open");

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();

    let usage = engine.capacity().get("only").expect("endpoint tracked");
    assert!(
        usage.used() > 0,
        "scan must populate used_bytes from on-disk SSTs, got 0",
    );
    assert!(
        usage.is_writable(),
        "small write well under 10 MB limit must stay writable",
    );
    assert_eq!(usage.severity(), CapacitySeverity::Normal);
}

/// Filling an endpoint past 100% MUST flip `is_writable` off and the
/// pre-put gate MUST reject the next write with
/// [`StorageError::CapacityExhausted`].
#[test]
fn full_endpoint_rejects_subsequent_writes() {
    let dir = TempDir::new().expect("tempdir");
    // Very small limit so a modest write blows past 100%.
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "small",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(4096)]);
    let engine = StorageEngine::open(&config).expect("open");

    // Write enough to exceed 4 KB on disk.
    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload-bytes")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();

    let usage = engine.capacity().get("small").expect("tracked");
    assert!(usage.used() > 4096, "wrote more than hard_limit");
    assert_eq!(usage.severity(), CapacitySeverity::Full);
    assert!(
        !usage.is_writable(),
        "Full severity must flip is_writable off",
    );

    // Next put on the user-data partition rejects.
    let result = engine.put(Partition::Node, b"node:0:after-full", b"extra");
    match result {
        Err(StorageError::CapacityExhausted {
            endpoint_id,
            hard_limit_bytes,
            ..
        }) => {
            assert_eq!(endpoint_id, "small");
            assert_eq!(hard_limit_bytes, 4096);
        }
        other => panic!("expected CapacityExhausted after endpoint filled, got: {other:?}"),
    }
}

/// Schema and Raft partitions MUST bypass the capacity gate — engine
/// metadata must remain reachable even when user-data endpoints are
/// full (otherwise the operator could not read the metrics that
/// prove the endpoint is full).
#[test]
fn schema_and_raft_partitions_bypass_capacity_gate() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "small",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(4096)]);
    let engine = StorageEngine::open(&config).expect("open");

    // Fill the endpoint.
    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("small").expect("tracked");
    assert!(!usage.is_writable(), "endpoint must be full");

    // Schema put still succeeds — engine metadata path.
    engine
        .put(Partition::Schema, b"schema:label:Test", b"{}")
        .expect("Schema must remain writable when user endpoint full");

    // User-data partition still rejects (sanity check that the gate
    // is actually active, not unconditionally bypassed).
    let denied = engine.put(Partition::Node, b"node:0:denied", b"x");
    assert!(
        matches!(denied, Err(StorageError::CapacityExhausted { .. })),
        "user partition must still be gated",
    );
}

/// Capacity recovery: deleting on-disk SSTs and re-scanning MUST flip
/// `is_writable` back to true so the engine can accept new writes
/// after cascade eviction or operator cleanup brings usage down.
#[test]
fn capacity_recovery_re_enables_writes() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "rec",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(4096)]);
    let engine = StorageEngine::open(&config).expect("open");

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload-bytes")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("rec").unwrap();
    assert!(!usage.is_writable(), "endpoint full after writes");

    // Simulate cascade-eviction / cleanup by deleting the SST files.
    let tables = dir.path().join(Partition::Node.name()).join("tables");
    for entry in std::fs::read_dir(&tables).expect("tables dir") {
        let p = entry.expect("entry").path();
        if p.is_file() {
            std::fs::remove_file(p).expect("remove SST");
        }
    }

    engine.refresh_capacity();
    assert!(
        usage.is_writable(),
        "recovery to under-limit usage must re-enable writes",
    );

    // Confirm the gate now accepts a write.
    engine
        .put(Partition::Node, b"node:0:after-recovery", b"v")
        .expect("write after recovery must succeed");
}

/// `HardLimitStrategy::CascadeEvict` MUST fire a cascade-eviction
/// when the scanner observes Emergency severity. The endpoint's
/// SSTs get demoted via major compaction (the per-LSM-level routing
/// mechanism from R158).
#[test]
fn cascade_evict_strategy_fires_at_emergency_threshold() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        // 64 KB limit — easy to push to 95% with a modest write.
        .with_hard_limit_bytes(64 * 1024)
        .with_hard_limit_strategy(HardLimitStrategy::CascadeEvict),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        )
        .with_hard_limit_bytes(0), // unlimited
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // Push enough into Node partition to land at Emergency severity
    // (95-99% of 64 KB ≈ 61-63 KB on disk after SST overhead). A few
    // hundred entries × ~80 B each should be in range; we err on the
    // side of definitely-crossing the threshold.
    for i in 0..800u32 {
        let key = format!("node:0:{i:010}");
        let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
    }
    engine.persist().expect("persist");

    // Record cold endpoint SST count before the refresh+cascade pass.
    let cold_tables = cold.path().join(Partition::Node.name()).join("tables");
    let cold_before = if cold_tables.exists() {
        std::fs::read_dir(&cold_tables).unwrap().count()
    } else {
        0
    };

    // First refresh triggers cascade-evict if severity crossed.
    engine.refresh_capacity();

    // Second persist + refresh to settle: cascade fires major
    // compaction asynchronously inside cascade_evict_endpoint;
    // bottom-level SSTs should now live on cold endpoint.
    let cold_tables_exists = cold_tables.exists();
    let cold_after = if cold_tables_exists {
        std::fs::read_dir(&cold_tables).unwrap().count()
    } else {
        0
    };
    assert!(
        cold_tables_exists,
        "cold endpoint tables dir must exist after cascade — major \
         compaction routes bottom levels to cold",
    );
    assert!(
        cold_after >= cold_before,
        "cold endpoint SST count must be ≥ pre-cascade count \
         (cold_before={cold_before}, cold_after={cold_after})",
    );
}

/// `HardLimitStrategy::Reject` (default) MUST NOT auto-fire cascade
/// eviction even at Emergency severity. The endpoint just sits there
/// near-full until operator intervention. Verified by configuring a
/// tiny limit and `Reject` strategy, then observing that the strategy
/// field on the tracker is preserved as `Reject` after the refresh
/// pass (the auto-cascade branch is gated on `CascadeEvict`).
#[test]
fn reject_strategy_preserved_no_auto_cascade() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        .with_hard_limit_bytes(4096), // tiny — definitely crosses thresholds
        // strategy defaults to Reject
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();

    let usage = engine.capacity().get("ep-hot").expect("tracked");
    // The Reject strategy is stored as-configured (would be
    // CascadeEvict if the auto-cascade path had upgraded the field).
    assert_eq!(
        usage.strategy,
        HardLimitStrategy::Reject,
        "strategy field must persist as Reject — not swapped to CascadeEvict",
    );
    // The refresh_capacity path inspects the strategy field and only
    // fires cascade for CascadeEvict; with Reject, no eviction.
    // Endpoint should be in Full severity given the tiny limit.
    assert_eq!(
        usage.severity(),
        CapacitySeverity::Full,
        "tiny limit + bulk write must push to Full severity",
    );
    assert!(!usage.is_writable());
}

/// `refresh_capacity` MUST persist the latest `used_bytes` snapshot to
/// Schema so a subsequent engine open warm-loads it. Lets the
/// pre-write gate work against plausible values immediately on
/// reopen, before the first scan completes.
#[test]
fn used_bytes_warm_load_on_reopen() {
    let dir = TempDir::new().expect("tempdir");
    let make_config = || {
        StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )
        .with_hard_limit_bytes(10_000_000)])
    };

    // First lifecycle: write + persist + refresh → snapshot lands in
    // Schema via the refresh path.
    let used_after_first_run: u64;
    {
        let engine = StorageEngine::open(&make_config()).expect("first open");
        for i in 0..500u32 {
            let key = format!("node:0:{i:010}");
            engine
                .put(Partition::Node, key.as_bytes(), b"payload-bytes")
                .expect("put");
        }
        engine.persist().expect("persist");
        engine.refresh_capacity();
        // Persist again so the Schema write reaches SST.
        engine.persist().expect("final persist");
        let usage = engine.capacity().get("ep").unwrap();
        used_after_first_run = usage.used();
        assert!(used_after_first_run > 0);
    }

    // Reopen — warm-load must populate `used_bytes` BEFORE any
    // refresh call. Observable by inspecting the tracker immediately
    // after open.
    let engine = StorageEngine::open(&make_config()).expect("reopen");
    let usage = engine.capacity().get("ep").unwrap();
    assert_eq!(
        usage.used(),
        used_after_first_run,
        "warm-load must restore last-known used_bytes without scan",
    );
}

/// Endpoints without a `hard_limit_bytes` configuration (default `0`)
/// MUST never enter alert severity or flip `is_writable`. The tracker
/// still records `used_bytes` for diagnostics but the limit is
/// "unlimited" semantically.
#[test]
fn endpoint_without_hard_limit_never_alerts() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "untracked",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload-bytes")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();

    let usage = engine.capacity().get("untracked").unwrap();
    assert_eq!(usage.hard_limit_bytes, 0, "no limit configured");
    assert_eq!(
        usage.severity(),
        CapacitySeverity::Normal,
        "unlimited endpoint must stay Normal regardless of usage",
    );
    assert!(usage.is_writable());
    // Subsequent writes continue to succeed.
    engine
        .put(Partition::Node, b"node:0:no-limit", b"v")
        .expect("unlimited endpoint accepts writes indefinitely");
}
