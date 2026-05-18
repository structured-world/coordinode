//! R156 integration tests — multi-endpoint StorageConfig end-to-end.
//!
//! Verifies that the engine opens correctly with multiple endpoints and
//! pins the current Layer-2 placement behavior so R158 (per-LSM-level tier
//! routing) has an explicit migration baseline to verify against:
//!
//! - **R156 contract:** all partitions live under the FIRST endpoint's
//!   path. SSTs MUST NOT appear at the second/third endpoint paths.
//! - **R158 will migrate this** — L0-L1 → Hot-tier endpoint, L4+ → Cold
//!   tier — so when that lands, tests in this file will need to update to
//!   the new placement contract.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use tempfile::TempDir;

/// `StorageEngine::open` succeeds with a single-endpoint config and the
/// data lives under that endpoint's path. Baseline for single-endpoint
/// (the single-endpoint case is just `Vec::with_endpoints(vec![one])`).
#[test]
fn single_endpoint_opens_and_writes() {
    let dir = TempDir::new().expect("tempdir");
    let endpoint = EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    );
    let config = StorageConfig::with_endpoints(vec![endpoint]);
    let engine = StorageEngine::open(&config).expect("open single endpoint");

    engine
        .put(Partition::Node, b"n:test", b"value")
        .expect("write");
    let v = engine
        .get(Partition::Node, b"n:test")
        .expect("read")
        .expect("present");
    assert_eq!(&v[..], b"value");

    assert_eq!(engine.data_dir(), dir.path(), "data_dir = first endpoint");
}

/// Multi-endpoint config opens. R156 baseline: every partition's tree
/// lives under the FIRST endpoint's path; secondary endpoints' paths
/// remain unpopulated until R158 lands per-level routing.
#[test]
fn multi_endpoint_opens_data_pinned_to_first() {
    let nvme = TempDir::new().expect("nvme tempdir");
    let hdd = TempDir::new().expect("hdd tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-nvme",
            nvme.path(),
            Media::Nvme,
            Durability::Degraded,
            Tier::Hot,
        )
        .with_capacity_bytes(1_000_000_000_000)
        .with_hard_limit_bytes(900_000_000_000),
        EndpointConfig::new(
            "ep-hdd",
            hdd.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        )
        .with_capacity_bytes(10_000_000_000_000),
    ]);

    let engine = StorageEngine::open(&config).expect("open multi-endpoint");

    // Write to every partition so each gets a real SST flush trigger later.
    engine
        .put(Partition::Node, b"node:test", b"node_value")
        .expect("node write");
    engine
        .put(Partition::EdgeProp, b"ep:test", b"edge_value")
        .expect("edgeprop write");
    engine
        .put(Partition::Schema, b"schema:test", b"schema_value")
        .expect("schema write");

    // Reads succeed.
    assert_eq!(
        &engine.get(Partition::Node, b"node:test").unwrap().unwrap()[..],
        b"node_value"
    );
    assert_eq!(
        &engine
            .get(Partition::EdgeProp, b"ep:test")
            .unwrap()
            .unwrap()[..],
        b"edge_value"
    );

    // **R156 baseline contract**: data_dir returns first endpoint path.
    assert_eq!(engine.data_dir(), nvme.path(), "data_dir = first endpoint");

    // **R156 baseline contract**: partition directories created under the
    // FIRST endpoint, NOT the second. R158 migration will change this —
    // when it does, this test needs to update.
    for part in Partition::all() {
        let first_partition_dir = nvme.path().join(part.name());
        let second_partition_dir = hdd.path().join(part.name());
        // Partition::Raft does not get its own directory in the LSM stack
        // (it's handled separately by the Raft snapshot pipeline).
        if *part == Partition::Raft {
            continue;
        }
        assert!(
            first_partition_dir.exists(),
            "partition {} must exist on first endpoint at {:?}",
            part.name(),
            first_partition_dir,
        );
        assert!(
            !second_partition_dir.exists(),
            "R156 baseline: partition {} MUST NOT exist on second endpoint at {:?} \
             (R158 will migrate per-level routing — update this test then)",
            part.name(),
            second_partition_dir,
        );
    }
}

/// Three-endpoint config with explicit tier diversity (cache + hot + cold).
/// R156 baseline: all three open; partitions still pinned to first.
#[test]
fn three_endpoint_opens_with_mixed_tiers() {
    let cache = TempDir::new().expect("cache tempdir");
    let hot = TempDir::new().expect("hot tempdir");
    let cold = TempDir::new().expect("cold tempdir");

    let config = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-cache",
            cache.path(),
            Media::Nvme,
            Durability::Volatile,
            Tier::HotCache,
        ),
        EndpointConfig::new(
            "ep-hot",
            hot.path(),
            Media::Ssd,
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

    let engine = StorageEngine::open(&config).expect("open three endpoints");
    engine
        .put(Partition::Node, b"k", b"v")
        .expect("write succeeds");
    assert_eq!(engine.data_dir(), cache.path(), "first endpoint is cache");
}

/// `with_endpoints` validation triggers at config-construction time,
/// before any I/O. Verifies the panic-on-duplicate-id invariant is
/// observed end-to-end (catches the failure before `StorageEngine::open`
/// is called).
#[test]
#[should_panic(expected = "duplicate EndpointConfig.id")]
fn duplicate_endpoint_id_rejected_before_open() {
    let dir = TempDir::new().expect("tempdir");
    let a = EndpointConfig::new(
        "same-id",
        dir.path().join("a"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    );
    let b = EndpointConfig::new(
        "same-id",
        dir.path().join("b"),
        Media::Ssd,
        Durability::Durable,
        Tier::Hot,
    );
    // This MUST panic — engine never gets a chance to open.
    let _ = StorageConfig::with_endpoints(vec![a, b]);
}

/// Verify the `data_dir()` accessor is stable across reopens: the engine
/// preserves the first-endpoint path through close/reopen cycles. Pins
/// R157 (WAL/oplog placement) baseline: WAL segments currently land at
/// `data_dir()` (= first endpoint).
#[test]
fn data_dir_stable_across_reopen() {
    let dir = TempDir::new().expect("tempdir");
    let first_endpoint = || {
        EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )
    };

    // First open.
    {
        let config = StorageConfig::with_endpoints(vec![first_endpoint()]);
        let engine = StorageEngine::open(&config).expect("first open");
        engine
            .put(Partition::Node, b"persist", b"v1")
            .expect("write");
        assert_eq!(engine.data_dir(), dir.path());
    }

    // Reopen.
    let config = StorageConfig::with_endpoints(vec![first_endpoint()]);
    let engine = StorageEngine::open(&config).expect("reopen");
    assert_eq!(engine.data_dir(), dir.path(), "stable across reopen");
    let v = engine
        .get(Partition::Node, b"persist")
        .expect("read")
        .expect("survived reopen");
    assert_eq!(&v[..], b"v1", "data survives reopen");
}

/// `path` belonging to a different endpoint MUST NOT be touched at
/// engine open / write time. Strong invariant: an endpoint not in the
/// `endpoints` list is invisible to the engine — even if a directory at
/// that path existed before, the engine doesn't enumerate it.
#[test]
fn unrelated_directory_untouched_by_engine() {
    let used = TempDir::new().expect("used tempdir");
    let unused = TempDir::new().expect("unused tempdir");

    // Place an arbitrary file under unused — engine MUST NOT delete it.
    let sentinel = unused.path().join("sentinel.txt");
    std::fs::write(&sentinel, b"do-not-touch").expect("write sentinel");

    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        used.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).expect("open");
    engine.put(Partition::Node, b"k", b"v").expect("write");
    drop(engine);

    // Sentinel survives intact.
    let content = std::fs::read(&sentinel).expect("sentinel still readable");
    assert_eq!(&content[..], b"do-not-touch");
    // Used directory has partition subdirs.
    assert!(
        used.path().join(Partition::Node.name()).exists(),
        "engine populated the used endpoint"
    );
    // Unused has nothing engine-created.
    let unused_node = unused.path().join(Partition::Node.name());
    assert!(
        !unused_node.exists(),
        "engine did not touch endpoint not in config"
    );
}
