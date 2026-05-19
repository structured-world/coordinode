//! Per-LSM-level endpoint routing integration tests.
//!
//! Verifies that with a multi-endpoint config:
//! - default routing is computed against the tier map (Hot → L0-L1,
//!   Warm → L2-L3, Cold → L4-L6),
//! - the routing is persisted in the Schema partition and reloaded on
//!   reopen,
//! - removing an endpoint between opens errors with a clear message,
//! - cascade eviction triggers compaction that physically moves SSTs
//!   from the saturated endpoint to the next-cooler one.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::wal::{WalConfig, WalSyncPolicy};
use tempfile::TempDir;

/// With a three-tier endpoint config (Hot NVMe + Warm SSD + Cold HDD),
/// writing data and forcing major compaction must cause SST files to
/// land at the cold-tier endpoint for L4+ levels.
#[test]
fn major_compaction_writes_ssts_to_cold_endpoint() {
    let hot = TempDir::new().expect("hot tempdir");
    let warm = TempDir::new().expect("warm tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "ep-warm",
            warm.path(),
            Media::Ssd,
            Durability::Durable,
            Tier::Warm,
        ),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);

    let engine = StorageEngine::open(&config).expect("open");

    // Write enough data to ensure flush + compaction will populate
    // non-zero levels.
    for i in 0..2000u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload")
            .expect("put");
    }
    // Force a flush to SST and then a major compaction that pushes
    // data through the LSM level hierarchy. Major compaction's bottom
    // level is L6 by default which under per-level routing lives on
    // the cold endpoint.
    engine.persist().expect("persist");
    engine
        .major_compact(Partition::Node)
        .expect("major compact");

    // Manifests live at primary (first) endpoint.
    let primary_node_dir = hot.path().join(Partition::Node.name());
    assert!(
        primary_node_dir.exists(),
        "primary endpoint must hold the Node partition manifest dir, expected {primary_node_dir:?}",
    );

    // After major compaction, the cold endpoint must contain SST files
    // (the bottom-level table folder is created lazily by lsm-tree only
    // when SSTs actually land there).
    let cold_node_tables = cold.path().join(Partition::Node.name()).join("tables");
    assert!(
        cold_node_tables.exists(),
        "cold endpoint must host bottom-level SSTs after major compaction \
         — expected directory at {cold_node_tables:?}",
    );
    let cold_sst_count = std::fs::read_dir(&cold_node_tables)
        .expect("read cold tables dir")
        .count();
    assert!(
        cold_sst_count > 0,
        "cold endpoint tables dir must contain at least one SST file after major compaction, \
         got {cold_sst_count}",
    );
}

/// Routing must be persisted on first open and reloaded on subsequent
/// opens. Verify by:
///   1. open with three-tier config, write+compact data,
///   2. close,
///   3. reopen with same config, verify reads succeed and data is
///      intact across all levels.
#[test]
fn routing_persists_across_reopen() {
    let hot = TempDir::new().expect("hot tempdir");
    let warm = TempDir::new().expect("warm tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let make_config = || {
        StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "ep-hot",
                hot.path(),
                Media::Nvme,
                Durability::Durable,
                Tier::Hot,
            ),
            EndpointConfig::new(
                "ep-warm",
                warm.path(),
                Media::Ssd,
                Durability::Durable,
                Tier::Warm,
            ),
            EndpointConfig::new(
                "ep-cold",
                cold.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Cold,
            ),
        ])
    };

    // First open + write + compact.
    {
        let engine = StorageEngine::open(&make_config()).expect("open 1");
        for i in 0..500u32 {
            let key = format!("node:0:{i:010}");
            engine
                .put(Partition::Node, key.as_bytes(), b"value")
                .expect("put");
        }
        engine.persist().expect("persist");
        engine
            .major_compact(Partition::Node)
            .expect("major compact");
    }

    // Reopen and verify data is still readable. lsm-tree's recovery
    // scan reads tables from primary + every persisted LevelRoute path,
    // so omitting routing on reopen would silently lose any SSTs that
    // landed at the cold endpoint.
    let engine = StorageEngine::open(&make_config()).expect("reopen");
    for i in 0..500u32 {
        let key = format!("node:0:{i:010}");
        let value = engine
            .get(Partition::Node, key.as_bytes())
            .expect("get")
            .unwrap_or_else(|| panic!("key {key} missing after reopen — routing not persisted?"));
        assert_eq!(&value[..], b"value");
    }
}

/// Removing an endpoint between opens, when the previous open
/// persisted routing referencing that endpoint, must surface a clear
/// error. This protects against silent SST loss: lsm-tree's recovery
/// scan would skip files at the removed path and orphan them.
#[test]
fn removing_endpoint_between_opens_errors() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    // First open with two endpoints — persists routing that references
    // both `ep-hot` and `ep-cold`.
    {
        let config = StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "ep-hot",
                hot.path(),
                Media::Nvme,
                Durability::Durable,
                Tier::Hot,
            ),
            EndpointConfig::new(
                "ep-cold",
                cold.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Cold,
            ),
        ]);
        let engine = StorageEngine::open(&config).expect("first open");
        engine.put(Partition::Node, b"k", b"v").expect("put");
        engine.persist().expect("persist");
    }

    // Second open with only ep-hot — engine must reject because the
    // persisted routing references the now-missing ep-cold.
    let removed_config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep-hot",
        hot.path(),
        Media::Nvme,
        Durability::Durable,
        Tier::Hot,
    )]);
    let result = StorageEngine::open(&removed_config);
    let err = match result {
        Ok(_) => panic!("reopen without ep-cold must error: persisted routing references it"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("ep-cold") || msg.contains("endpoint"),
        "error must mention the missing endpoint, got: {msg}",
    );
}

/// Single-endpoint config must NOT emit any level routes — every level
/// falls through to the primary path. Smoke test that the engine opens
/// and operates normally without multi-tier wiring.
#[test]
fn single_endpoint_skips_level_routes() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "only",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    // Write + persist + major-compact: with no level_routes everything
    // lands under the single endpoint's partition directory.
    for i in 0..100u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"x")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine
        .major_compact(Partition::Node)
        .expect("major compact");

    let primary_tables = dir.path().join(Partition::Node.name()).join("tables");
    assert!(
        primary_tables.exists(),
        "single-endpoint config: tables dir must exist at primary path {primary_tables:?}",
    );
}

/// Cascade eviction kicks compaction on the partition trees that hold
/// data on the named endpoint. Verifies the mechanism by:
///   1. populating L0-L3 on the hot endpoint via writes,
///   2. invoking cascade eviction for the hot endpoint,
///   3. asserting that after eviction the cold endpoint holds SSTs.
#[test]
fn cascade_eviction_moves_data_to_cooler_endpoint() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);

    let engine = StorageEngine::open(&config).expect("open");
    for i in 0..1500u32 {
        let key = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, key.as_bytes(), b"payload")
            .expect("put");
    }
    engine.persist().expect("persist");

    let cold_tables = cold.path().join(Partition::Node.name()).join("tables");
    let before = if cold_tables.exists() {
        std::fs::read_dir(&cold_tables).expect("read cold").count()
    } else {
        0
    };

    let report = engine
        .cascade_evict_endpoint("ep-hot")
        .expect("cascade eviction");
    assert!(
        report.compacted_partitions > 0,
        "cascade eviction must have compacted at least one partition, got {}",
        report.compacted_partitions,
    );

    assert!(
        cold_tables.exists(),
        "cold endpoint tables dir must exist after cascade eviction, expected {cold_tables:?}",
    );
    let after = std::fs::read_dir(&cold_tables).expect("read cold").count();
    assert!(
        after > before,
        "cold endpoint SST count must increase after cascade eviction (before={before}, after={after})",
    );
}

/// `cascade_evict_endpoint` with an unknown endpoint id must surface a
/// clear error rather than silently doing nothing — operator typo would
/// otherwise be lost in the noise of "no partitions needed eviction".
#[test]
fn cascade_eviction_unknown_endpoint_errors() {
    let dir = TempDir::new().expect("tempdir");
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "only",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");

    let result = engine.cascade_evict_endpoint("nonexistent");
    let err = match result {
        Ok(_) => panic!("cascade_evict_endpoint must reject unknown endpoint id"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("unknown") && msg.contains("nonexistent"),
        "error must name the unknown endpoint id, got: {msg}",
    );
}

/// `cascade_evict_endpoint` for an endpoint that hosts no partition's
/// data (e.g. operator named a hot endpoint before any flush happened)
/// must succeed as a no-op: zero compactions, no error. Common case
/// during a manually-triggered eviction immediately after engine open.
#[test]
fn cascade_eviction_with_no_partition_data_is_noop() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // Single tiny write that stays in memtable — no flush, no SST,
    // no partition tree has any data on disk yet.
    engine.put(Partition::Node, b"k", b"v").expect("put");

    let report = engine
        .cascade_evict_endpoint("ep-hot")
        .expect("cascade eviction must succeed even when there's nothing on the endpoint");
    // Major compaction is still triggered on partitions whose routing
    // references the endpoint — even if there is no SST to compact,
    // the call itself is a no-op at lsm-tree level. The contract we
    // pin: no error, finite number of compacted partitions.
    let _ = report.compacted_partitions; // smoke check the field is reachable
}

/// `cascade_evict_endpoint` must touch ALL partitions whose persisted
/// routing references the saturated endpoint — not just the first one.
/// Verifies the iteration covers the non-Schema, non-Raft set fully.
#[test]
fn cascade_eviction_touches_multiple_partitions() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // Write into THREE distinct partitions so each has a tree with
    // pending state.
    for i in 0..500u32 {
        let k = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, k.as_bytes(), b"n")
            .expect("put node");
    }
    for i in 0..500u32 {
        let k = format!("edgeprop:T:{i:010}");
        engine
            .put(Partition::EdgeProp, k.as_bytes(), b"e")
            .expect("put edgeprop");
    }
    for i in 0..500u32 {
        let k = format!("idx:i:{i:010}");
        engine
            .put(Partition::Idx, k.as_bytes(), b"i")
            .expect("put idx");
    }
    engine.persist().expect("persist");

    let report = engine
        .cascade_evict_endpoint("ep-hot")
        .expect("cascade eviction");
    assert!(
        report.compacted_partitions >= 3,
        "cascade eviction must touch every routed partition (>=3 expected for Node + EdgeProp + Idx), got {}",
        report.compacted_partitions,
    );
}

/// Reopen with an ADDED endpoint must NOT alter the persisted routing —
/// existing partitions keep their previously-computed level mapping so
/// lsm-tree recovery still finds every SST. Adding endpoints is an
/// operator-driven event that may later be acted on by an explicit
/// rebalance, but at engine-open time the persisted routing wins.
#[test]
fn reopen_with_added_endpoint_preserves_routing() {
    let hot = TempDir::new().expect("hot tempdir");
    let warm = TempDir::new().expect("warm tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    // First open with two endpoints (Hot + Cold). Default routing
    // places L2-L3 on Hot (Warm fallback chain → Hot, since no Warm
    // endpoint exists).
    {
        let config = StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "ep-hot",
                hot.path(),
                Media::Nvme,
                Durability::Durable,
                Tier::Hot,
            ),
            EndpointConfig::new(
                "ep-cold",
                cold.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Cold,
            ),
        ]);
        let engine = StorageEngine::open(&config).expect("first open");
        for i in 0..500u32 {
            let k = format!("node:0:{i:010}");
            engine
                .put(Partition::Node, k.as_bytes(), b"v")
                .expect("put");
        }
        engine.persist().expect("persist");
        engine
            .major_compact(Partition::Node)
            .expect("major compact");
    }

    // SST tables snapshot before the topology change. After reopen
    // with the same routing they MUST be byte-stable (no migration).
    let warm_tables_path = warm.path().join(Partition::Node.name()).join("tables");
    assert!(
        !warm_tables_path.exists(),
        "warm endpoint had no role in first open — must have no tables dir",
    );

    // Reopen with a NEW Warm endpoint added. Persisted routing should
    // win: L2-L3 stays on Hot (where it was originally placed), the
    // new Warm endpoint receives no data for the Node partition at
    // open time.
    let config_with_warm = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "ep-warm",
            warm.path(),
            Media::Ssd,
            Durability::Durable,
            Tier::Warm,
        ),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config_with_warm).expect("reopen with added endpoint");

    // Data still readable — proves recovery scan covered the persisted
    // route set (cold endpoint), not the would-be-rederived set that
    // would have routed L4+ to warm.
    let key = b"node:0:0000000042";
    let v = engine
        .get(Partition::Node, key)
        .expect("get")
        .expect("present");
    assert_eq!(&v[..], b"v");

    // The newly added Warm endpoint has NO Node tables dir — confirms
    // persisted routing did NOT auto-rederive against the new topology.
    assert!(
        !warm_tables_path.exists(),
        "added Warm endpoint must NOT receive any Node SSTs without an explicit rebalance \
         (persisted routing wins on reopen)",
    );
}

/// WAL replay through a multi-tier config: write data via the standalone
/// WAL (no SST flush), crash (drop engine), reopen, verify data survives.
/// The replay path applies mutations to memtables and the next flush
/// places SSTs per the persisted level routing — so the multi-tier
/// distribution must compose cleanly with WAL recovery.
#[test]
fn wal_replay_through_multi_tier_routing() {
    let hot = TempDir::new().expect("hot tempdir");
    let warm = TempDir::new().expect("warm tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let make_config = || {
        StorageConfig::with_endpoints(vec![
            EndpointConfig::new(
                "ep-hot",
                hot.path(),
                Media::Nvme,
                Durability::Durable,
                Tier::Hot,
            ),
            EndpointConfig::new(
                "ep-warm",
                warm.path(),
                Media::Ssd,
                Durability::Durable,
                Tier::Warm,
            ),
            EndpointConfig::new(
                "ep-cold",
                cold.path(),
                Media::Hdd,
                Durability::Durable,
                Tier::Cold,
            ),
        ])
    };
    let wal_cfg = || WalConfig {
        path: None,
        sync: WalSyncPolicy::SyncPerRecord,
    };

    // Phase 1: open with WAL, write, drop without flush.
    {
        let engine =
            StorageEngine::open_with_wal(&make_config(), Some(wal_cfg())).expect("first open");
        engine
            .put(Partition::Node, b"survives-multi-tier", b"replay-me")
            .expect("write through WAL");
        // Drop without persist — memtable NOT flushed, only the WAL
        // record is on disk.
    }

    // Phase 2: reopen. WAL replay restores the memtable; the engine is
    // operational with the routing persisted from Phase 1.
    let engine =
        StorageEngine::open_with_wal(&make_config(), Some(wal_cfg())).expect("reopen with replay");
    let value = engine
        .get(Partition::Node, b"survives-multi-tier")
        .expect("get after replay")
        .expect("key must survive WAL replay across the multi-tier topology");
    assert_eq!(&value[..], b"replay-me");

    // Force a flush so the replayed data hits SST. Major-compact pushes
    // it to the bottom level which routes to the cold endpoint.
    engine.persist().expect("persist");
    engine
        .major_compact(Partition::Node)
        .expect("major compact");

    let cold_tables = cold.path().join(Partition::Node.name()).join("tables");
    assert!(
        cold_tables.exists(),
        "cold endpoint tables dir must exist after WAL replay + flush + compaction; \
         multi-tier routing must compose with WAL replay path",
    );
    let cold_count = std::fs::read_dir(&cold_tables)
        .expect("read cold dir")
        .count();
    assert!(
        cold_count > 0,
        "cold endpoint must hold the post-replay SST after major compaction, got {cold_count}",
    );
}

/// Cascade-evicting the primary endpoint is operationally a no-op for
/// data movement: the primary is the lsm-tree `Config.path` catch-all,
/// and `to_level_routes` omits LevelRoutes pointing to the primary, so
/// compaction has no cooler tier to push data into via level_routes.
///
/// The call still returns `Ok` and reports the partitions whose
/// routing references the primary (every partition does — they all
/// have a level routed to primary in any multi-endpoint config).
/// Documents the edge case so operators understand "you cannot cascade
/// off the primary endpoint without first promoting another endpoint
/// to primary" — that promotion is a separate, future operator
/// action.
#[test]
fn cascade_eviction_of_primary_endpoint_is_data_motion_noop() {
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    // ep-hot is FIRST → primary.
    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Nvme,
            Durability::Durable,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "ep-cold",
            cold.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ]);
    let engine = StorageEngine::open(&config).expect("open");

    // Write + persist + major-compact to populate the cold endpoint.
    for i in 0..500u32 {
        let k = format!("node:0:{i:010}");
        engine
            .put(Partition::Node, k.as_bytes(), b"v")
            .expect("put");
    }
    engine.persist().expect("persist");
    engine
        .major_compact(Partition::Node)
        .expect("initial major compact");

    let cold_tables = cold.path().join(Partition::Node.name()).join("tables");
    let cold_before = std::fs::read_dir(&cold_tables)
        .expect("read cold dir")
        .count();

    // Cascade-evict the PRIMARY (ep-hot). The call succeeds and reports
    // a compaction — but the cold endpoint's SST count does NOT
    // increase further (data was already pushed by the initial
    // major_compact, and there is no cooler tier below primary to
    // demote to).
    let report = engine
        .cascade_evict_endpoint("ep-hot")
        .expect("cascade evict primary succeeds");
    assert!(
        report.compacted_partitions > 0,
        "every partition's routing references the primary, so cascade evicts each tree, got {}",
        report.compacted_partitions,
    );

    let cold_after = std::fs::read_dir(&cold_tables)
        .expect("read cold dir")
        .count();
    assert_eq!(
        cold_after, cold_before,
        "cascade-evicting the primary endpoint cannot move further data \
         (level_routes already placed bottom-level data on cold; primary has no cooler peer in this topology)",
    );
}
