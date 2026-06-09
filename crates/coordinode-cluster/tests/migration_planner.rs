//! Integration test: end-to-end migration planner against three real
//! `StorageEngine` instances at separate temp-dirs. Exercises the full
//! capacity scan -> PlannerContext -> LocalMigrationPlanner pipeline
//! that production code would walk when a capacity event fires.
//!
//! The test does not spin a Raft cluster: the planner is pure-Rust
//! topology + cost arithmetic and depends on nothing from Raft. Three
//! independent `StorageEngine` instances at three temp-dirs are
//! sufficient to demonstrate that a multi-endpoint deployment can
//! observe a real capacity reading on one node and produce a plan
//! that targets a different one.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_cluster::FailureDomain;
use coordinode_cluster::{
    estimate_cost, ClusterTopology, CostInputs, LocalMigrationPlanner, MigrationPlanner, Modality,
    PayloadEstimate, PlannerContext, ShardId, SingleNodeTopology, TopologyTree, TransferMode,
};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use tempfile::TempDir;

struct EndpointFixture {
    engine: StorageEngine,
    _dir: TempDir,
}

fn open_endpoint(id: &str, hard_limit: u64) -> EndpointFixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        id,
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )
    .with_hard_limit_bytes(hard_limit)]);
    let engine = StorageEngine::open(&cfg).expect("open engine");
    EndpointFixture { engine, _dir: dir }
}

fn three_endpoint_topology() -> SingleNodeTopology {
    let tree = TopologyTree {
        endpoints: vec![
            FailureDomain::local("ep-a", Tier::Warm),
            FailureDomain::local("ep-b", Tier::Warm),
            FailureDomain::local("ep-c", Tier::Warm),
        ],
    };
    SingleNodeTopology::from_tree(tree)
}

#[test]
fn planner_picks_remote_endpoint_when_source_fills_up() {
    // Three independent storage engines on three temp-dirs. The
    // topology mirrors them by name so the planner can reason about
    // ep-a / ep-b / ep-c without caring that they are local-process
    // engines vs full network nodes; the planner does pure topology
    // + cost arithmetic.
    let limit = 200_000_u64;
    let ep_a = open_endpoint("ep-a", limit);
    let _ep_b = open_endpoint("ep-b", limit);
    let _ep_c = open_endpoint("ep-c", limit);

    // Load enough data on ep-a to push the capacity tracker above
    // the soft threshold. The exact byte count does not matter for
    // the planner test; we just need the engine to report a real
    // non-zero used value that we can ship as the migration payload
    // estimate.
    let payload_padding = "x".repeat(64);
    for i in 0..2_000_u32 {
        let key = format!("node:0:{i:010}");
        let value = format!("vec-{i}-{payload_padding}");
        ep_a.engine
            .put(Partition::Node, key.as_bytes(), value.as_bytes())
            .expect("put");
    }
    ep_a.engine.persist().expect("persist");
    ep_a.engine.refresh_capacity();
    let usage = ep_a
        .engine
        .capacity()
        .get("ep-a")
        .expect("ep-a tracked")
        .used();
    assert!(usage > 0, "ep-a must report a non-zero used byte count");

    // PayloadEstimate is what the planner consumes; in production
    // it would be derived from the shard's actual on-disk footprint
    // plus the HNSW node count. The bytes come from the live engine,
    // the node count and ef_construction are realistic stand-ins
    // for a vector shard at this scale.
    let payload = PayloadEstimate {
        bytes: usage,
        node_count: 2_000,
        ef_construction: 200,
    };
    let costs = CostInputs {
        bandwidth_bytes_per_sec: 100_000_000.0,
        hnsw_build_rate_nodes_per_sec: 5_000.0,
        neighbour_byte_cost: 64.0,
    };

    let topology = three_endpoint_topology();
    // Sanity: topology really sees three endpoints at Warm tier.
    let candidates = topology
        .placement_candidates(
            &coordinode_cluster::CrushRule::local_tier(),
            Modality::Vector,
            Tier::Warm,
        )
        .expect("warm tier candidates");
    assert_eq!(candidates.len(), 3);

    let planner = LocalMigrationPlanner::new(topology, Tier::Warm, Modality::Vector);
    let ctx = PlannerContext {
        source: "ep-a".to_string(),
        shard: ShardId::ZERO,
        payload,
        costs,
        online_policy_override: None,
    };
    let plan = planner.plan(&ctx).expect("planner must produce a plan");

    assert_eq!(plan.source, "ep-a");
    assert_ne!(plan.target, "ep-a", "must not migrate to self");
    assert!(
        plan.target == "ep-b" || plan.target == "ep-c",
        "target must be one of the remote endpoints, got {}",
        plan.target,
    );

    // Cost breakdown is internally consistent with the cost model
    // for the recommended mode.
    let expected_cost = estimate_cost(plan.recommended_mode, payload, costs);
    assert_eq!(plan.cost_breakdown, expected_cost);
    assert!(
        (plan.estimated_total_secs - expected_cost.total_secs()).abs() < 1e-9,
        "estimated_total_secs must equal the breakdown total",
    );

    // Mode-specific zero-on-other-side invariants. Network is always
    // charged because the f32 truth tier always ships.
    assert!(plan.cost_breakdown.network_secs > 0.0);
    match plan.recommended_mode {
        TransferMode::RebuildFromData => {
            assert_eq!(plan.cost_breakdown.graph_serialise_secs, 0.0);
            assert!(plan.cost_breakdown.rebuild_secs > 0.0);
        }
        TransferMode::ShipGraphBytes => {
            assert_eq!(plan.cost_breakdown.rebuild_secs, 0.0);
            assert!(plan.cost_breakdown.graph_serialise_secs > 0.0);
        }
    }

    // Network component formula: bytes / bandwidth.
    let expected_network = (payload.bytes as f64) / costs.bandwidth_bytes_per_sec;
    assert!((plan.cost_breakdown.network_secs - expected_network).abs() < 1e-9);
}
