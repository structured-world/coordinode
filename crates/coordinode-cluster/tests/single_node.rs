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
fn empty_tree_yields_one_shard_zero_candidates() {
    // StorageConfig requires ≥1 endpoint by design, so the empty
    // case only arises via from_tree (bootstrap, in-memory test).
    // Topology still exposes the in-process shard.
    let topology = SingleNodeTopology::from_tree(TopologyTree::default());
    assert_eq!(topology.shards().len(), 1);
    assert_eq!(topology.topology_tree().endpoint_count(), 0);

    let hits = topology
        .placement_candidates(&CrushRule::local_tier(), Modality::Node, Tier::Warm)
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn placement_candidates_returns_multiple_endpoints_at_same_tier() {
    // Two HDD endpoints at Warm tier — both must surface as
    // candidates; query layer will pick using LSM-level routing.
    let cfg = StorageConfig::with_endpoints(vec![
        EndpointConfig::new(
            "ep-warm-a",
            std::path::Path::new("/tmp/a"),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        ),
        EndpointConfig::new(
            "ep-warm-b",
            std::path::Path::new("/tmp/b"),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        ),
    ]);
    let topology = SingleNodeTopology::from_storage(&cfg);
    let mut hits = topology
        .placement_candidates(&CrushRule::local_tier(), Modality::Node, Tier::Warm)
        .unwrap();
    hits.sort();
    assert_eq!(hits, vec!["ep-warm-a".to_string(), "ep-warm-b".to_string()],);
}

#[test]
fn from_tree_with_multiple_servers_collapses_in_servers_list() {
    use coordinode_cluster::FailureDomain;
    let mut srv1 = FailureDomain::local("ep-1", Tier::Hot);
    srv1.server = "srv-a".into();
    let mut srv1_2 = FailureDomain::local("ep-2", Tier::Warm);
    srv1_2.server = "srv-a".into();
    let mut srv2 = FailureDomain::local("ep-3", Tier::Cold);
    srv2.server = "srv-b".into();

    let tree = TopologyTree {
        endpoints: vec![srv1, srv1_2, srv2],
    };
    assert_eq!(
        tree.servers(),
        vec!["srv-a".to_string(), "srv-b".to_string()]
    );
}

#[test]
fn modality_serde_roundtrip() {
    for &m in Modality::all() {
        let bytes = rmp_serde::to_vec_named(&m).expect("encode");
        let decoded: Modality = rmp_serde::from_slice(&bytes).expect("decode");
        assert_eq!(decoded, m);
    }
}

#[test]
fn shard_id_total_ordering_and_display() {
    assert!(ShardId(1) < ShardId(2));
    let mut ids = [ShardId(3), ShardId(1), ShardId(2)];
    ids.sort();
    assert_eq!(ids[0], ShardId(1));
    assert_eq!(format!("{}", ShardId(42)), "shard-42");
    assert_eq!(ShardId::ZERO.raw(), 0);
}

#[test]
fn ee_only_error_carries_helpful_message() {
    let err = TopologyError::EeOnly("test reason");
    let rendered = format!("{err}");
    assert!(rendered.contains("EE-only"), "got: {rendered}");
    assert!(rendered.contains("test reason"));
}

#[test]
fn topology_serde_roundtrip_preserves_every_leaf() {
    use coordinode_cluster::FailureDomain;
    let mut a = FailureDomain::local("ep-a", Tier::Hot);
    a.tags.insert("zone".into(), "eu".into());
    let tree = TopologyTree {
        endpoints: vec![a, FailureDomain::local("ep-b", Tier::Warm)],
    };
    let bytes = rmp_serde::to_vec_named(&tree).expect("encode");
    let decoded: TopologyTree = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(decoded, tree);
    assert_eq!(
        decoded.endpoints[0].tags.get("zone").map(String::as_str),
        Some("eu"),
    );
}

#[test]
fn shard_descriptor_and_node_addr_serde_roundtrip() {
    use coordinode_cluster::ShardDescriptor;
    let desc = ShardDescriptor {
        id: ShardId(7),
        leader: NodeAddr {
            server: "srv-3".into(),
            address: "10.0.0.3:7080".into(),
        },
        replicas: vec![NodeAddr::local()],
    };
    let bytes = rmp_serde::to_vec_named(&desc).expect("encode");
    let decoded: ShardDescriptor = rmp_serde::from_slice(&bytes).expect("decode");
    assert_eq!(decoded, desc);
}

#[test]
fn crush_rule_serde_distinguishes_variants() {
    let local = CrushRule::LocalTier;
    let spread = CrushRule::Spread {
        level: "rack".into(),
        min: 3,
    };
    let local_bytes = rmp_serde::to_vec_named(&local).expect("encode");
    let spread_bytes = rmp_serde::to_vec_named(&spread).expect("encode");
    assert_ne!(local_bytes, spread_bytes);
    let local_decoded: CrushRule = rmp_serde::from_slice(&local_bytes).expect("decode");
    let spread_decoded: CrushRule = rmp_serde::from_slice(&spread_bytes).expect("decode");
    assert_eq!(local_decoded, local);
    assert_eq!(spread_decoded, spread);
}

#[test]
fn endpoints_in_rack_filters_correctly() {
    use coordinode_cluster::FailureDomain;
    let mut r1 = FailureDomain::local("ep-r1", Tier::Warm);
    r1.rack = "r42".into();
    let mut r2 = FailureDomain::local("ep-r2", Tier::Warm);
    r2.rack = "r43".into();
    let tree = TopologyTree {
        endpoints: vec![r1, r2],
    };
    let in_r42: Vec<_> = tree.endpoints_in_rack("r42").collect();
    assert_eq!(in_r42.len(), 1);
    assert_eq!(in_r42[0].endpoint, "ep-r1");
    assert!(tree.endpoints_in_rack("never").next().is_none());
}

#[test]
fn endpoints_in_geo_filters_correctly() {
    use coordinode_cluster::FailureDomain;
    let mut eu = FailureDomain::local("ep-eu", Tier::Cold);
    eu.geo = "europe".into();
    let mut us = FailureDomain::local("ep-us", Tier::Cold);
    us.geo = "us-east".into();
    let tree = TopologyTree {
        endpoints: vec![eu, us],
    };
    assert_eq!(tree.endpoints_in_geo("europe").count(), 1);
    assert_eq!(tree.endpoints_in_geo("us-east").count(), 1);
    assert_eq!(tree.endpoints_in_geo("apac").count(), 0);
}

#[test]
fn endpoints_on_server_filters_correctly() {
    use coordinode_cluster::FailureDomain;
    let mut a = FailureDomain::local("ep-1", Tier::Hot);
    a.server = "srv-a".into();
    let mut b = FailureDomain::local("ep-2", Tier::Hot);
    b.server = "srv-a".into();
    let mut c = FailureDomain::local("ep-3", Tier::Hot);
    c.server = "srv-b".into();
    let tree = TopologyTree {
        endpoints: vec![a, b, c],
    };
    assert_eq!(tree.endpoints_on_server("srv-a").count(), 2);
    assert_eq!(tree.endpoints_on_server("srv-b").count(), 1);
}

#[test]
fn topology_error_clone_and_eq() {
    let a = TopologyError::ShardNotFound(ShardId(7));
    let b = a.clone();
    assert_eq!(a, b);
    let c = TopologyError::EeOnly("x");
    assert_ne!(a, c);
}

#[test]
fn endpoints_in_dc_filters_correctly() {
    use coordinode_cluster::FailureDomain;
    let mut eu = FailureDomain::local("ep-eu", Tier::Warm);
    eu.dc = "eu-west-1".into();
    let mut us = FailureDomain::local("ep-us", Tier::Warm);
    us.dc = "us-east-1".into();
    let tree = TopologyTree {
        endpoints: vec![eu, us],
    };
    let in_eu: Vec<_> = tree.endpoints_in_dc("eu-west-1").collect();
    assert_eq!(in_eu.len(), 1);
    assert_eq!(in_eu[0].endpoint, "ep-eu");
    let in_apac: Vec<_> = tree.endpoints_in_dc("apac-1").collect();
    assert!(in_apac.is_empty());
}

#[test]
fn traits_are_send_and_sync() {
    // Compile-time assertion: if anyone adds a `Cell` / `Rc` to
    // SingleNodeTopology this stops building.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SingleNodeTopology>();
    assert_send_sync::<SingleShardRouting>();
}

#[test]
fn topology_can_be_arc_shared_across_threads() {
    use std::sync::Arc;
    use std::thread;

    let topology = Arc::new(SingleNodeTopology::from_storage(&three_tier_config()));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let t = Arc::clone(&topology);
            thread::spawn(move || {
                // Reads are safe to interleave — topology is read-only.
                let _ = t.shards();
                let _ =
                    t.placement_candidates(&CrushRule::local_tier(), Modality::Node, Tier::Warm);
                let _ = t.shard_leader(ShardId::ZERO);
            })
        })
        .collect();
    for h in handles {
        h.join().expect("join");
    }
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
