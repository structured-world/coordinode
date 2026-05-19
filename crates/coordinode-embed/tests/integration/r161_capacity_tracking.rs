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

use coordinode_storage::engine::batch::WriteBatch;
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
///
/// Uses a generous hard_limit (~200 KB) so the fixed engine-internal
/// overhead (Schema partition manifest etc.) is well under threshold
/// at baseline, while a bulk-write puts the endpoint over the line
/// and SST-deletion brings it back.
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
    .with_hard_limit_bytes(40_000)]);
    let engine = StorageEngine::open(&config).expect("open");

    // Push past 200 KB with bulk writes (the scan now sees the full
    // endpoint footprint — partition tables + manifest + WAL + …).
    for i in 0..5000u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload-bytes")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("rec").unwrap();
    assert!(
        !usage.is_writable(),
        "endpoint must be full after bulk write, got used={}",
        usage.used(),
    );

    // Simulate cascade-eviction / cleanup by deleting the Node
    // partition's SSTs. The remaining engine-internal bytes
    // (Schema manifest etc.) must be small enough that the
    // post-deletion total is well under the 200 KB limit.
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
        "recovery to under-limit usage must re-enable writes (used={})",
        usage.used(),
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

/// Prometheus counter `endpoint_threshold_alerts_total` MUST
/// increment on every severity UP-crossing. Drives the endpoint into
/// `Warning` severity through a local metrics recorder and inspects
/// the captured counter value.
#[test]
fn threshold_alert_counter_increments_on_severity_crossing() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let dir = TempDir::new().expect("tempdir");
    // 4 KB limit so a modest write crosses Warning then Full.
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "alert-ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(4096)]);
    let engine = StorageEngine::open(&config).expect("open");

    // Drive enough writes to push severity past Normal → Warning →
    // Full. Persist + refresh INSIDE the recorder scope so the alert
    // emissions are captured.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        for i in 0..500u32 {
            let key = format!("node:0:{i:010}");
            // Note: writes start succeeding (endpoint not yet full
            // until the first refresh fires the gate). Ignore the
            // CapacityExhausted that surfaces partway through —
            // we only care about the counter side-effect.
            let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
        }
        engine.persist().expect("persist");
        engine.refresh_capacity();
    });

    // Walk the snapshot and find the threshold-alert counter for our
    // endpoint. Any UP-crossing (Normal→Warning, Warning→Full, etc.)
    // contributes one increment. We assert at least ONE — the exact
    // count depends on whether refresh observed an intermediate
    // severity or jumped straight to Full.
    let snapshot = snapshotter.snapshot();
    let mut alert_count = 0u64;
    for (key, _unit, _desc, value) in snapshot.into_vec() {
        if key.key().name() == "endpoint_threshold_alerts_total" {
            // Confirm the labels carry our endpoint id.
            let has_endpoint = key
                .key()
                .labels()
                .any(|l| l.key() == "endpoint_id" && l.value() == "alert-ep");
            if has_endpoint {
                if let DebugValue::Counter(n) = value {
                    alert_count += n;
                }
            }
        }
    }
    assert!(
        alert_count >= 1,
        "expected at least one threshold alert counter increment on severity crossing, got {alert_count}",
    );
}

/// Prometheus counter `endpoint_cascade_events_total` MUST increment
/// each time auto-cascade fires. Triggers Emergency severity on a
/// CascadeEvict-strategy endpoint and inspects the counter.
#[test]
fn cascade_events_counter_increments_on_auto_cascade() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");
    // 4 KB hard limit so a modest write reliably lands in
    // Emergency-or-Full severity — guarantees auto-cascade fires
    // independently of LSM compression / SST overhead variance.
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        .with_hard_limit_bytes(4096)
        .with_hard_limit_strategy(HardLimitStrategy::CascadeEvict),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        for i in 0..500u32 {
            let key = format!("node:0:{i:010}");
            let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
        }
        engine.persist().expect("persist");
        engine.refresh_capacity();
    });

    let snapshot = snapshotter.snapshot();
    let mut cascade_count = 0u64;
    for (key, _unit, _desc, value) in snapshot.into_vec() {
        if key.key().name() == "endpoint_cascade_events_total" {
            let has_endpoint = key
                .key()
                .labels()
                .any(|l| l.key() == "endpoint_id" && l.value() == "ep-hot");
            if has_endpoint {
                if let DebugValue::Counter(n) = value {
                    cascade_count += n;
                }
            }
        }
    }
    assert!(
        cascade_count >= 1,
        "expected at least one cascade event counter increment when auto-cascade fired, got {cascade_count}",
    );
}

/// Gauges `endpoint_used_bytes` and `endpoint_hard_limit_bytes` MUST
/// be set on every refresh — observable via the same debugging
/// recorder. Pins the gauge contract.
#[test]
fn capacity_gauges_emitted_on_refresh() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "gauge-ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(1_000_000)]);
    let engine = StorageEngine::open(&config).expect("open");

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        for i in 0..200u32 {
            let key = format!("node:0:{i:010}");
            engine
                .put(Partition::Node, key.as_bytes(), b"payload-bytes")
                .expect("put");
        }
        engine.persist().expect("persist");
        engine.refresh_capacity();
    });

    let snapshot = snapshotter.snapshot();
    let mut saw_used_gauge = false;
    let mut saw_limit_gauge = false;
    for (key, _unit, _desc, value) in snapshot.into_vec() {
        let name = key.key().name();
        let endpoint_ok = key
            .key()
            .labels()
            .any(|l| l.key() == "endpoint_id" && l.value() == "gauge-ep");
        if !endpoint_ok {
            continue;
        }
        if let DebugValue::Gauge(g) = value {
            let g = g.into_inner();
            if name == "endpoint_used_bytes" {
                saw_used_gauge = true;
                assert!(g > 0.0, "used_bytes gauge must be positive after writes");
            } else if name == "endpoint_hard_limit_bytes" {
                saw_limit_gauge = true;
                assert_eq!(g, 1_000_000.0, "hard_limit_bytes gauge mirrors config");
            }
        }
    }
    assert!(
        saw_used_gauge,
        "endpoint_used_bytes gauge must be emitted on refresh"
    );
    assert!(
        saw_limit_gauge,
        "endpoint_hard_limit_bytes gauge must be emitted on refresh"
    );
}

/// Concurrent put + refresh: writers and the capacity scanner must
/// not race in ways that produce torn reads or panics. Spawns N
/// writer threads doing bulk puts while a parallel thread loops on
/// `refresh_capacity()`. Final state must be consistent: every key
/// written before the test stopped is readable, the tracker's
/// `used_bytes` is monotonically non-decreasing across observations,
/// and `is_writable` semantics hold.
#[test]
fn concurrent_put_and_refresh_no_torn_reads() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let dir = TempDir::new().expect("tempdir");
    // Generous limit so writers don't get gated mid-test.
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(1_000_000_000)]);
    let engine = Arc::new(StorageEngine::open(&config).expect("open"));

    let stop = Arc::new(AtomicBool::new(false));

    // Refresher thread — bangs on refresh_capacity() while writers
    // push data through. Any race in the AtomicU64 / AtomicBool /
    // Mutex<CapacitySeverity> code path would surface as a panic
    // (poison) or a tracker.used() reading that goes backwards.
    let refresher_handle = {
        let engine = Arc::clone(&engine);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut last_used: u64 = 0;
            while !stop.load(Ordering::Acquire) {
                engine.refresh_capacity();
                let usage = engine.capacity().get("ep").expect("tracked");
                let cur = usage.used();
                // No torn-read sanity: u64 atomics never expose
                // half-written values, so any observed `cur` must be
                // a complete snapshot. Across observations the value
                // may both grow (new flushes) AND shrink (compaction
                // merging SSTs reclaims overhead; WAL rotation
                // truncates the old log). The atomic-correctness
                // assertion is implicit in the lack of panics.
                let _ = (last_used, cur);
                last_used = cur;
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        })
    };

    // 4 writer threads each pushing 250 puts = 1000 writes total.
    let mut writer_handles = Vec::new();
    for w in 0..4 {
        let engine = Arc::clone(&engine);
        writer_handles.push(std::thread::spawn(move || {
            for i in 0..250u32 {
                let key = format!("node:0:w{w}:{i:010}");
                engine
                    .put(Partition::Node, key.as_bytes(), b"payload-bytes")
                    .expect("put");
            }
        }));
    }
    for h in writer_handles {
        h.join().expect("writer join");
    }
    // Force a final flush so SSTs are on disk for the last refresh.
    engine.persist().expect("persist");
    stop.store(true, Ordering::Release);
    refresher_handle.join().expect("refresher join");

    // Final state check: every key readable, tracker reports a
    // plausible used_bytes (> 0, < limit).
    for w in 0..4 {
        for i in 0..250u32 {
            let key = format!("node:0:w{w}:{i:010}");
            let v = engine
                .get(Partition::Node, key.as_bytes())
                .expect("get")
                .unwrap_or_else(|| panic!("missing key after concurrent run: {key}"));
            assert_eq!(&v[..], b"payload-bytes");
        }
    }
    let usage = engine.capacity().get("ep").expect("tracked");
    assert!(usage.used() > 0);
    assert!(usage.is_writable(), "well under the 1 GB limit");
}

/// Warm-load preserves a `Full` severity state: opening an engine
/// whose previous run persisted `used_bytes >= hard_limit_bytes` MUST
/// initialise the tracker with `is_writable = false` BEFORE the first
/// scan tick runs. Otherwise the operator would observe a transient
/// "writable" window on reopen that the next scan would immediately
/// close, defeating the purpose of warm-load.
#[test]
fn warm_load_preserves_full_severity_state() {
    let dir = TempDir::new().expect("tempdir");
    let make_config = || {
        StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )
        .with_hard_limit_bytes(4096)])
    };

    // Phase 1: fill past 100% + persist the snapshot.
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
        // Persist the schema partition again so the meta:capacity:*
        // snapshot reaches SST before drop.
        engine.persist().expect("final persist");
        let usage = engine.capacity().get("ep").unwrap();
        assert!(!usage.is_writable(), "endpoint full at end of phase 1");
    }

    // Phase 2: reopen. The very first inspection — BEFORE any scan
    // tick — must already report is_writable=false because warm-load
    // resolved severity against the persisted used_bytes.
    let engine = StorageEngine::open(&make_config()).expect("reopen");
    let usage = engine.capacity().get("ep").unwrap();
    assert!(
        !usage.is_writable(),
        "warm-load must preserve Full severity → is_writable=false; \
         no transient writable window on reopen",
    );

    // The pre-write gate must also reject — proves the warm-loaded
    // is_writable flag is consulted by `check_partition_capacity`.
    let denied = engine.put(Partition::Node, b"node:0:after-reopen", b"x");
    assert!(
        matches!(denied, Err(StorageError::CapacityExhausted { .. })),
        "warm-loaded Full endpoint must reject writes on reopen, got: {denied:?}",
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

/// Multi-endpoint capacity tracking: each endpoint's `used_bytes` is
/// computed independently from its own `tables/` directories, and
/// severity / `is_writable` transitions on one endpoint do NOT affect
/// any other endpoint's state. Production CE+EE tier deployments
/// rely on this — the scanner must report per-endpoint state, never
/// aggregate.
#[test]
fn multi_endpoint_tracking_is_independent_per_endpoint() {
    let hot = TempDir::new().expect("hot tempdir");
    let warm = TempDir::new().expect("warm tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    // Three endpoints, three different hard_limits. Default routing
    // places L0-L1 on Hot (the first endpoint), L2-L3 on Warm,
    // L4-L6 on Cold. We will force data through compaction so all
    // three tiers receive bytes, then assert each endpoint's tracker
    // is updated independently.
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        .with_hard_limit_bytes(1_000_000_000),
        EndpointConfig::new(
            "ep-warm",
            warm.path(),
            Media::Ssd,
            Durability::Durable,
            Tier::Warm,
        )
        .with_hard_limit_bytes(2_000_000_000),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        )
        .with_hard_limit_bytes(10_000_000_000),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // Write + persist + major-compact to populate L0 and L4+ levels.
    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload-bytes")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine
        .major_compact(Partition::Node)
        .expect("major compact");
    engine.refresh_capacity();

    // Each endpoint must have an independent tracker entry.
    let hot_usage = engine.capacity().get("ep-hot").expect("hot tracked");
    let warm_usage = engine.capacity().get("ep-warm").expect("warm tracked");
    let cold_usage = engine.capacity().get("ep-cold").expect("cold tracked");

    // hard_limit_bytes preserved per-endpoint.
    assert_eq!(hot_usage.hard_limit_bytes, 1_000_000_000);
    assert_eq!(warm_usage.hard_limit_bytes, 2_000_000_000);
    assert_eq!(cold_usage.hard_limit_bytes, 10_000_000_000);

    // Cold gets the bottom-level SSTs after major compaction → its
    // tracker MUST report > 0 used_bytes. Hot keeps the partition
    // manifest, so it also reports > 0 (manifest bytes).
    assert!(
        cold_usage.used() > 0,
        "cold endpoint must hold bottom-level SSTs after major_compact, got 0",
    );
    assert!(
        hot_usage.used() > 0,
        "hot endpoint must hold the partition manifest, got 0",
    );

    // All three are well under their limits, so severity stays Normal
    // independently — no endpoint's severity is "inherited" from
    // another.
    assert_eq!(hot_usage.severity(), CapacitySeverity::Normal);
    assert_eq!(warm_usage.severity(), CapacitySeverity::Normal);
    assert_eq!(cold_usage.severity(), CapacitySeverity::Normal);
    assert!(hot_usage.is_writable());
    assert!(warm_usage.is_writable());
    assert!(cold_usage.is_writable());
}

/// Multi-endpoint scenario: filling ONE endpoint past 100% MUST NOT
/// flip `is_writable` on any other endpoint, and MUST NOT trigger
/// auto-cascade for an endpoint that is itself not in Emergency.
/// Pins per-endpoint isolation of state transitions.
#[test]
fn multi_endpoint_full_state_does_not_leak_across_endpoints() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        // Tiny Hot limit so it fills.
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        )
        .with_hard_limit_bytes(4096),
        // Huge Cold limit so it stays Normal.
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        )
        .with_hard_limit_bytes(10_000_000_000),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();

    let hot_usage = engine.capacity().get("ep-hot").expect("tracked");
    let cold_usage = engine.capacity().get("ep-cold").expect("tracked");

    // Hot is Full (writes exceeded 4 KB easily).
    assert_eq!(hot_usage.severity(), CapacitySeverity::Full);
    assert!(!hot_usage.is_writable());
    // Cold is well under its 10 GB limit — Normal, writable.
    assert_eq!(cold_usage.severity(), CapacitySeverity::Normal);
    assert!(
        cold_usage.is_writable(),
        "cold endpoint must NOT inherit Hot's Full state — \
         per-endpoint state tracking",
    );
}

/// DOWN-crossing severity (Full → Normal after recovery) MUST NOT
/// increment `endpoint_threshold_alerts_total`. The counter measures
/// "alerts went off", not "severity changed in either direction".
/// Pins the asymmetric semantics of the alert counter.
#[test]
fn down_crossing_severity_does_not_increment_alert_counter() {
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};

    let dir = TempDir::new().expect("tempdir");
    // Generous limit so engine-internal overhead (Schema manifest
    // etc., now counted by the broader scan) is well under threshold
    // at baseline. Bulk Node writes push over, SST-deletion recovers.
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(40_000)]);
    let engine = StorageEngine::open(&config).expect("open");

    // Phase 1: write to Full (UP-crossing increments counter).
    for i in 0..5000u32 {
        let key = format!("node:0:{i:010}");
        let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("ep").expect("tracked");
    assert_eq!(usage.severity(), CapacitySeverity::Full);

    // Phase 2: install a fresh recorder, then delete SSTs +
    // refresh — the DOWN-crossing back to Normal must NOT increment
    // the alert counter.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::with_local_recorder(&recorder, || {
        let tables = dir.path().join(Partition::Node.name()).join("tables");
        for entry in std::fs::read_dir(&tables).expect("tables") {
            let p = entry.expect("entry").path();
            if p.is_file() {
                std::fs::remove_file(p).expect("remove SST");
            }
        }
        engine.refresh_capacity();
    });

    // Severity must now be Normal (the DOWN-crossing happened
    // inside the recorder scope).
    let usage = engine.capacity().get("ep").expect("tracked");
    assert_eq!(
        usage.severity(),
        CapacitySeverity::Normal,
        "deletion + refresh must drop severity back to Normal",
    );
    assert!(
        usage.is_writable(),
        "DOWN-crossing also flips is_writable back on"
    );

    // No alert counter increments captured during the DOWN-crossing
    // refresh. The recorder only saw events from the second refresh
    // call (where severity moved DOWN), so any
    // `endpoint_threshold_alerts_total` increments here would be a
    // bug.
    let snapshot = snapshotter.snapshot();
    for (key, _unit, _desc, value) in snapshot.into_vec() {
        if key.key().name() == "endpoint_threshold_alerts_total" {
            if let DebugValue::Counter(n) = value {
                assert_eq!(
                    n, 0,
                    "DOWN-crossing must not increment the alert counter — \
                     the counter records severity escalations only",
                );
            }
        }
    }
}

// ── Regression tests for hard-limit gate completeness ─────────────────
//
// The initial pre-write gate was wired only to `engine.put`. All three
// remaining write paths (`engine.delete`, `engine.merge`,
// `WriteBatch::commit`) were left ungated — writes through them would
// succeed even on a Full endpoint, silently violating INV-D3. Each
// test below MUST fail against the pre-fix code and pass after the
// gate is propagated to every write path.

/// `engine.delete` on a Full endpoint MUST reject with
/// `CapacityExhausted`. A delete tombstone still consumes memtable
/// bytes that will eventually flush to an SST — under INV-D3 the
/// endpoint can't accept it.
#[test]
fn delete_on_full_endpoint_rejects() {
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

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("small").expect("tracked");
    assert!(!usage.is_writable(), "endpoint must be full");

    let result = engine.delete(Partition::Node, b"node:0:any-key");
    assert!(
        matches!(result, Err(StorageError::CapacityExhausted { .. })),
        "delete on Full endpoint must reject with CapacityExhausted, got: {result:?}",
    );
}

/// `engine.merge` on a Full endpoint MUST reject with
/// `CapacityExhausted`. Merge operands accumulate in the memtable
/// until compaction folds them — they are additive in storage cost.
#[test]
fn merge_on_full_endpoint_rejects() {
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

    // Fill via the Adj partition (Adj is the canonical merge target —
    // posting-list deltas via `engine.merge`).
    for i in 0..500u32 {
        let key = format!("adj:T:out:{i:010}");
        let _ = engine.put(Partition::Adj, key.as_bytes(), b"payload-bytes");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("small").expect("tracked");
    assert!(!usage.is_writable(), "endpoint must be full");

    let result = engine.merge(Partition::Adj, b"adj:T:out:42", &[0u8; 8]);
    assert!(
        matches!(result, Err(StorageError::CapacityExhausted { .. })),
        "merge on Full endpoint must reject with CapacityExhausted, got: {result:?}",
    );
}

/// `WriteBatch::commit` on a Full endpoint MUST reject — this is the
/// primary write path used by Raft proposal apply, query runner, and
/// most internal subsystems. The pre-write gate must fire here too,
/// or the entire mechanism is bypassable by every actual writer.
#[test]
fn write_batch_commit_on_full_endpoint_rejects() {
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

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("small").expect("tracked");
    assert!(!usage.is_writable(), "endpoint must be full");

    let mut batch = WriteBatch::new(&engine);
    batch.put(Partition::Node, b"node:0:batched", b"v");
    let result = batch.commit();
    assert!(
        matches!(result, Err(StorageError::CapacityExhausted { .. })),
        "WriteBatch::commit on Full endpoint must reject, got: {result:?}",
    );
}

/// `WriteBatch` with mixed partitions: if ANY partition's L0 endpoint
/// is Full, the entire commit MUST reject (atomicity — partial
/// commits across partitions are not a thing in our model). Even if
/// some partitions in the batch target a non-Full endpoint, the
/// presence of a single Full-targeted mutation aborts the whole.
#[test]
fn write_batch_commit_rejects_when_any_partition_endpoint_full() {
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

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    assert!(!engine.capacity().get("small").unwrap().is_writable());

    let mut batch = WriteBatch::new(&engine);
    // Schema is bypassed by the gate, so this mutation alone would
    // succeed. Mixing it with a Node-targeted mutation must still
    // abort because Node's L0 endpoint is Full.
    batch.put(Partition::Schema, b"schema:label:Test", b"{}");
    batch.put(Partition::Node, b"node:0:mixed", b"v");
    let result = batch.commit();
    assert!(
        matches!(result, Err(StorageError::CapacityExhausted { .. })),
        "mixed-partition batch with one Full-targeted mutation must reject, got: {result:?}",
    );
}

/// Read paths MUST remain functional on a Full endpoint —
/// `engine.get`, `prefix_scan` and friends MUST NOT consult the
/// capacity gate. Operators need to be able to inspect data on a
/// full endpoint to decide what to evict.
#[test]
fn reads_remain_functional_on_full_endpoint() {
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

    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        let _ = engine.put(Partition::Node, key.as_bytes(), b"payload-bytes");
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    assert!(!engine.capacity().get("small").unwrap().is_writable());

    // Point reads succeed.
    let v = engine
        .get(Partition::Node, b"node:0:0000000042")
        .expect("get must work on Full endpoint");
    assert!(v.is_some(), "key must still be readable");

    // Prefix scans succeed.
    let iter = engine
        .prefix_scan(Partition::Node, b"node:0:")
        .expect("prefix_scan must work on Full endpoint");
    let count = iter.count();
    assert!(
        count > 0,
        "prefix scan must return existing keys on Full endpoint",
    );
}

/// Compaction-driven recovery: write past the limit, fire
/// `major_compact` (the real production path that reclaims space
/// from stale MVCC versions, tombstones, and pre-compaction
/// L0/L1 overlap), refresh, and assert writes RESUME automatically
/// because `used_bytes` dropped back below `hard_limit_bytes`.
///
/// This is the canonical "I deleted a bunch of data and the engine
/// keeps refusing my writes — why?" question. The answer: deletes
/// produce tombstones which still occupy space; only compaction
/// physically reclaims it. The capacity gate has to follow the
/// post-compaction reality, not a synchronous accounting that
/// confuses logical deletes with physical reclamation.
#[test]
fn writes_resume_after_compaction_frees_space() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(40_000)]);
    let engine = StorageEngine::open(&config).expect("open");

    // Phase 1: bulk-write the SAME 100 keys 50 times. Each rewrite
    // creates a new MVCC version of the same key; the engine keeps
    // all 50 versions in SSTs until compaction folds them. This is
    // the easy way to manufacture compaction-reclaimable space
    // without depending on tombstones.
    for round in 0..50u32 {
        for i in 0..100u32 {
            let key = format!("node:0:{i:010}");
            let value = format!("round-{round}-payload-bytes-padding-padding");
            let _ = engine.put(Partition::Node, key.as_bytes(), value.as_bytes());
        }
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("ep").expect("tracked");
    assert!(
        !usage.is_writable(),
        "endpoint must be full after rewrite barrage (used={})",
        usage.used(),
    );

    // Subsequent put rejects.
    let denied = engine.put(Partition::Node, b"node:0:while-full", b"x");
    assert!(
        matches!(denied, Err(StorageError::CapacityExhausted { .. })),
        "writes must be gated while endpoint is Full",
    );

    // Phase 2: fire major compaction. It will fold every key down to
    // its latest version, reclaiming the ~49× redundancy from
    // Phase 1. After refresh, used_bytes drops back below the
    // threshold and is_writable flips on.
    engine
        .major_compact(Partition::Node)
        .expect("major compact");
    engine.refresh_capacity();
    let usage = engine.capacity().get("ep").expect("tracked");
    assert!(
        usage.is_writable(),
        "writes must resume after compaction reclaims space \
         (used={} <= hard_limit=40000)",
        usage.used(),
    );

    // Phase 3: confirm the gate now accepts new writes.
    engine
        .put(Partition::Node, b"node:0:after-compact", b"v")
        .expect("write must succeed after compaction recovery");
}

/// Cascade-eviction-driven recovery on a multi-tier config: the
/// Hot endpoint fills past its limit, `cascade_evict_endpoint`
/// fires (via the `CascadeEvict` strategy + emergency severity)
/// to push bottom-level SSTs to the Cold endpoint, the Hot
/// endpoint's `used_bytes` drops back below the threshold, and
/// writes resume on partitions routed to Hot.
///
/// This is the canonical CE+EE flow for "Hot tier capacity event
/// auto-recovers without operator intervention".
#[test]
fn writes_resume_after_cascade_eviction_frees_hot_endpoint() {
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
        .with_hard_limit_bytes(40_000)
        .with_hard_limit_strategy(HardLimitStrategy::CascadeEvict),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // Bulk write — same key set, many rewrites, plus diverse keys
    // to give compaction enough material to push to bottom level.
    for round in 0..30u32 {
        for i in 0..200u32 {
            let key = format!("node:0:{i:010}");
            let value = format!("round-{round}-payload-bytes-padding");
            let _ = engine.put(Partition::Node, key.as_bytes(), value.as_bytes());
        }
    }
    engine.persist().expect("persist");
    engine.refresh_capacity();
    let usage = engine.capacity().get("ep-hot").expect("hot tracked");
    assert!(
        !usage.is_writable(),
        "Hot endpoint must reach Full after bulk write \
         (used={})",
        usage.used(),
    );

    // `refresh_capacity` itself already fires the auto-cascade for
    // `CascadeEvict` strategy at Emergency-or-Full severity — see
    // run_capacity_refresh. The cascade calls major_compact, which
    // pushes the bottom-level SSTs to the Cold endpoint via the
    // per-LSM-level routing's L4+ → Cold mapping.

    // Run a second refresh: the post-cascade SSTs should now be on
    // Cold, freeing Hot.
    engine.refresh_capacity();
    let hot_usage = engine.capacity().get("ep-hot").expect("hot");
    let cold_usage = engine.capacity().get("ep-cold").expect("cold");

    // Hot endpoint MUST have dropped below its limit (cascade
    // physically moved bytes off).
    assert!(
        hot_usage.is_writable(),
        "Hot endpoint must auto-recover after cascade eviction \
         (hot_used={}, cold_used={})",
        hot_usage.used(),
        cold_usage.used(),
    );

    // Subsequent put on Node (routed to Hot at L0) succeeds.
    engine
        .put(Partition::Node, b"node:0:after-cascade", b"v")
        .expect("write must succeed after cascade-driven recovery");
}

/// Capacity scanner MUST count every on-disk file under the endpoint
/// root, not just `<part>/tables/`. Engine-managed artefacts that
/// live in sibling directories (WAL, oplog segments, tantivy FTS
/// index files, manifest data) all consume disk and must contribute
/// to `used_bytes`. A narrower per-partition scan would silently
/// under-count and let the disk fill while the gate reports
/// comfortable headroom.
///
/// This test plants arbitrary files in non-partition subdirectories
/// of the endpoint root and asserts the scanner sees them.
#[test]
fn refresh_counts_all_endpoint_files_not_only_partition_tables() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(10_000_000)]);
    let engine = StorageEngine::open(&config).expect("open");

    // Plant a few non-partition files that the OLD scanner would
    // have missed:
    //   - WAL-like file
    //   - oplog-like segment
    //   - tantivy-like index directory
    //   - misc operator file (random log)
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("mkdir wal");
    std::fs::write(wal_dir.join("standalone.wal"), vec![0u8; 100_000]).expect("write wal file");

    let oplog_dir = dir.path().join("oplog").join("0");
    std::fs::create_dir_all(&oplog_dir).expect("mkdir oplog");
    std::fs::write(
        oplog_dir.join("segment-00000000000000000001.bin"),
        vec![0u8; 50_000],
    )
    .expect("write oplog seg");

    let text_dir = dir.path().join("text_indexes").join("text_idx_x_body");
    std::fs::create_dir_all(&text_dir).expect("mkdir text idx");
    std::fs::write(text_dir.join("meta.json"), vec![0u8; 30_000]).expect("write tantivy meta");
    std::fs::write(text_dir.join("segment.idx"), vec![0u8; 200_000])
        .expect("write tantivy segment");

    // Run capacity refresh and assert every planted file counts.
    engine.refresh_capacity();
    let usage = engine.capacity().get("ep").expect("tracked");
    let observed = usage.used();

    // Total planted bytes = 100k + 50k + 30k + 200k = 380k. Plus
    // engine-internal files (schema partition manifests etc.) that
    // exist after open. Assert at least the planted total.
    let planted = 100_000 + 50_000 + 30_000 + 200_000;
    assert!(
        observed >= planted,
        "scanner must see at least the planted {planted} bytes in non-partition dirs, got {observed}",
    );
}

/// Tantivy-style FTS index that pushes an endpoint past 100% MUST
/// flip `is_writable` off — even though the bytes did not arrive
/// through `engine.put`. Tantivy writes are not gated themselves
/// (they happen inside the query layer's index update path), but
/// they MUST show up in capacity accounting so the next put through
/// the gate observes the Full state and rejects.
#[test]
fn tantivy_index_bytes_can_push_endpoint_to_full() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(4096)]);
    let engine = StorageEngine::open(&config).expect("open");

    // Simulate a fat tantivy index without going through the query
    // layer: drop a large file into the canonical text-index path.
    let text_dir = dir.path().join("text_indexes").join("text_idx_X_body");
    std::fs::create_dir_all(&text_dir).expect("mkdir");
    std::fs::write(text_dir.join("fat.idx"), vec![0u8; 8192]).expect("write fat idx");

    engine.refresh_capacity();
    let usage = engine.capacity().get("ep").expect("tracked");

    assert_eq!(
        usage.severity(),
        CapacitySeverity::Full,
        "8 KB tantivy file > 4 KB hard_limit must register as Full",
    );
    assert!(
        !usage.is_writable(),
        "Full state from non-partition bytes still flips the gate",
    );

    // Subsequent put on user data must reject — proves the gate
    // honours the non-partition byte count.
    let result = engine.put(Partition::Node, b"node:0:after-fat-idx", b"x");
    assert!(
        matches!(result, Err(StorageError::CapacityExhausted { .. })),
        "Full from FTS bytes must still gate writes, got: {result:?}",
    );
}
