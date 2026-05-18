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
