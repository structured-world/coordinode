//! Integration tests for the CE single-node topology + routing
//! implementations. Verifies the trait contracts hold against a
//! realistic multi-endpoint storage config (one server, multiple
//! tiers) — the most-common CE deployment shape.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_cluster::{
    ClusterTopology, CrushRule, Modality, NodeAddr, ShardId, ShardRouting, SingleNodeTopology,
    SingleShardRouting, TopologyError, TopologyTree,
};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

fn three_tier_config() -> StorageConfig {
    StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-hot",
            std::path::Path::new("/tmp/nvme"),
            Media::Nvme,
            Durability::Volatile,
            Tier::Hot,
        ),
        EndpointConfig::new(
            "ep-warm",
            std::path::Path::new("/tmp/ssd"),
            Media::Ssd,
            Durability::Durable,
            Tier::Warm,
        ),
        EndpointConfig::new(
            "ep-cold",
            std::path::Path::new("/tmp/hdd"),
            Media::Hdd,
            Durability::Durable,
            Tier::Cold,
        ),
    ])
}

#[test]
fn from_storage_builds_one_shard_per_cluster() {
    let topology = SingleNodeTopology::from_storage(&three_tier_config());
    assert_eq!(topology.shards().len(), 1);
    assert_eq!(topology.shards()[0].id, ShardId::ZERO);
    assert_eq!(topology.shards()[0].leader.server, "local");
    // Single-node has RF=1; the leader is the sole replica.
    assert_eq!(topology.shards()[0].replicas.len(), 1);
}

#[test]
fn from_storage_lifts_every_endpoint_into_tree() {
    let topology = SingleNodeTopology::from_storage(&three_tier_config());
    let tree = topology.topology_tree();
    assert_eq!(tree.endpoint_count(), 3);

    // Every leaf is bound to the local server with collapsed upper
    // levels.
    for leaf in &tree.endpoints {
        assert_eq!(leaf.geo, "local");
        assert_eq!(leaf.dc, "local");
        assert_eq!(leaf.rack, "local");
        assert_eq!(leaf.server, "local");
    }
    // Servers() collapses duplicates — exactly one server in the tree.
    assert_eq!(tree.servers(), vec!["local".to_owned()]);
}

#[test]
fn shard_leader_resolves_zero_returns_local() {
    let topology = SingleNodeTopology::from_storage(&three_tier_config());
    let leader = topology.shard_leader(ShardId::ZERO).unwrap();
    assert_eq!(leader, NodeAddr::local());
}

#[test]
fn shard_leader_unknown_id_errors() {
    let topology = SingleNodeTopology::from_storage(&three_tier_config());
    let err = topology.shard_leader(ShardId(99)).unwrap_err();
    assert_eq!(err, TopologyError::ShardNotFound(ShardId(99)));
}

#[test]
fn placement_candidates_local_tier_filters_by_tier() {
    let topology = SingleNodeTopology::from_storage(&three_tier_config());
    let hot = topology
        .placement_candidates(&CrushRule::local_tier(), Modality::Vector, Tier::Hot)
        .unwrap();
    assert_eq!(hot, vec!["ep-hot".to_owned()]);

    let warm = topology
        .placement_candidates(&CrushRule::local_tier(), Modality::Node, Tier::Warm)
        .unwrap();
    assert_eq!(warm, vec!["ep-warm".to_owned()]);

    let cold = topology
        .placement_candidates(&CrushRule::local_tier(), Modality::Blob, Tier::Cold)
        .unwrap();
    assert_eq!(cold, vec!["ep-cold".to_owned()]);
}

#[test]
fn placement_candidates_tier_with_no_endpoint_returns_empty() {
    let topology = SingleNodeTopology::from_storage(&three_tier_config());
    // No endpoint configured for Memory tier in this fixture.
    let mem = topology
        .placement_candidates(&CrushRule::local_tier(), Modality::Node, Tier::Memory)
        .unwrap();
    assert!(mem.is_empty());
}

#[test]
fn placement_candidates_spread_rule_surfaces_ee_only() {
    let topology = SingleNodeTopology::from_storage(&three_tier_config());
    let err = topology
        .placement_candidates(
            &CrushRule::Spread {
                level: "rack".to_owned(),
                min: 2,
            },
            Modality::Node,
            Tier::Warm,
        )
        .unwrap_err();
    assert!(matches!(err, TopologyError::EeOnly(_)));
}

#[test]
fn from_tree_supports_custom_shape_for_testing() {
    // Skip StorageConfig; build a 2-endpoint tree manually.
    let tree = TopologyTree {
        endpoints: vec![
            coordinode_cluster::FailureDomain::local("a", Tier::Hot),
            coordinode_cluster::FailureDomain::local("b", Tier::Warm),
        ],
    };
    let topology = SingleNodeTopology::from_tree(tree);
    assert_eq!(topology.topology_tree().endpoint_count(), 2);
}

#[test]
fn single_shard_routing_collapses_every_key_to_zero() {
    let routing = SingleShardRouting::new();
    assert_eq!(routing.shard_for_key(b""), ShardId::ZERO);
    assert_eq!(routing.shard_for_key(b"alice"), ShardId::ZERO);
    assert_eq!(routing.shard_for_key(&[0u8; 1024]), ShardId::ZERO);
    assert_eq!(routing.shard_ids(), vec![ShardId::ZERO]);
}

#[test]
fn single_shard_routing_resolve_zero_returns_local() {
    let routing = SingleShardRouting::new();
    assert_eq!(routing.resolve(ShardId::ZERO).unwrap(), NodeAddr::local());
}

#[test]
fn single_shard_routing_resolve_unknown_errors() {
    let routing = SingleShardRouting::new();
    assert_eq!(
        routing.resolve(ShardId(5)).unwrap_err(),
        TopologyError::ShardNotFound(ShardId(5))
    );
}

#[test]
fn modality_all_lists_every_variant() {
    // Sanity guard so adding a new Modality variant fails this test
    // and forces the author to update the all() inventory.
    assert_eq!(Modality::all().len(), 9);
}

#[test]
fn endpoint_tags_propagate_into_topology() {
    let mut ep = EndpointConfig::new(
        "ep-tagged",
        std::path::Path::new("/tmp/x"),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    );
    ep.tags.insert("zone".into(), "eu-west-1a".into());
    let cfg = StorageConfig::with_endpoints(vec![ep]);
    let topology = SingleNodeTopology::from_storage(&cfg);
    let leaf = &topology.topology_tree().endpoints[0];
    assert_eq!(
        leaf.tags.get("zone").map(String::as_str),
        Some("eu-west-1a")
    );
}
