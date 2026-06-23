use super::*;
use crate::engine::config::Media;

fn ep(id: &str, tier: Tier, media: Media, dur: Durability) -> EndpointConfig {
    EndpointConfig::new(id, format!("/{id}"), media, dur, tier)
}

#[test]
fn default_routing_three_tier_canonical() {
    let endpoints = vec![
        ep("hot", Tier::Hot, Media::Nvme, Durability::Durable),
        ep("warm", Tier::Warm, Media::Ssd, Durability::Durable),
        ep("cold", Tier::Cold, Media::Hdd, Durability::Durable),
    ];
    let routing = PartitionRouting::default_for_endpoints(&endpoints);
    assert_eq!(routing.levels.get(&0).unwrap(), "hot");
    assert_eq!(routing.levels.get(&1).unwrap(), "hot");
    assert_eq!(routing.levels.get(&2).unwrap(), "warm");
    assert_eq!(routing.levels.get(&3).unwrap(), "warm");
    assert_eq!(routing.levels.get(&4).unwrap(), "cold");
    assert_eq!(routing.levels.get(&5).unwrap(), "cold");
    assert_eq!(routing.levels.get(&6).unwrap(), "cold");
}

#[test]
fn default_routing_single_endpoint_all_levels_pinned() {
    let endpoints = vec![ep("only", Tier::Warm, Media::Hdd, Durability::Durable)];
    let routing = PartitionRouting::default_for_endpoints(&endpoints);
    for lvl in 0..=MAX_ROUTED_LEVEL {
        assert_eq!(routing.levels.get(&lvl).unwrap(), "only");
    }
}

#[test]
fn default_routing_missing_warm_falls_back() {
    // Hot + Cold, no Warm — L2-L3 should fall back.
    let endpoints = vec![
        ep("hot", Tier::Hot, Media::Nvme, Durability::Durable),
        ep("cold", Tier::Cold, Media::Hdd, Durability::Durable),
    ];
    let routing = PartitionRouting::default_for_endpoints(&endpoints);
    assert_eq!(routing.levels.get(&0).unwrap(), "hot");
    assert_eq!(routing.levels.get(&1).unwrap(), "hot");
    // L2-L3: no Warm → fallback prefers Hot before Cold.
    assert_eq!(routing.levels.get(&2).unwrap(), "hot");
    assert_eq!(routing.levels.get(&3).unwrap(), "hot");
    assert_eq!(routing.levels.get(&4).unwrap(), "cold");
}

#[test]
fn default_routing_volatile_excluded_from_persistent_placement() {
    // Volatile HotCache + Durable Warm: HotCache MUST NOT be chosen
    // for L0-L1 (volatile = no persistent SST landing). Warm is used
    // as fallback for all bands.
    let endpoints = vec![
        ep("cache", Tier::HotCache, Media::Nvme, Durability::Volatile),
        ep("warm", Tier::Warm, Media::Ssd, Durability::Durable),
    ];
    let routing = PartitionRouting::default_for_endpoints(&endpoints);
    for lvl in 0..=MAX_ROUTED_LEVEL {
        assert_eq!(
            routing.levels.get(&lvl).unwrap(),
            "warm",
            "level {lvl} must route to durable endpoint, not Volatile cache",
        );
    }
}

#[test]
fn default_routing_all_volatile_falls_back_to_volatile_pool() {
    // Escape hatch: with_endpoints_no_persistence configs (all
    // Volatile) still need a routing — use the Volatile pool.
    let endpoints = vec![ep("memfs", Tier::Memory, Media::Ram, Durability::Volatile)];
    let routing = PartitionRouting::default_for_endpoints(&endpoints);
    for lvl in 0..=MAX_ROUTED_LEVEL {
        assert_eq!(routing.levels.get(&lvl).unwrap(), "memfs");
    }
}

#[test]
fn validate_unknown_endpoint_errors() {
    let endpoints = vec![ep("real", Tier::Warm, Media::Hdd, Durability::Durable)];
    let mut routing = PartitionRouting::default_for_endpoints(&endpoints);
    routing.levels.insert(0, "removed".to_string());
    let err = routing.validate(&endpoints).expect_err("must error");
    match err {
        RoutingError::UnknownEndpoint { endpoint_id } => {
            assert_eq!(endpoint_id, "removed");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn validate_missing_level_errors() {
    let endpoints = vec![ep("only", Tier::Warm, Media::Hdd, Durability::Durable)];
    let mut routing = PartitionRouting::default_for_endpoints(&endpoints);
    routing.levels.remove(&3);
    let err = routing.validate(&endpoints).expect_err("must error");
    match err {
        RoutingError::MissingLevel { level } => assert_eq!(level, 3),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn to_level_routes_groups_consecutive_levels() {
    use lsm_tree::fs::StdFs;
    let endpoints = vec![
        ep("hot", Tier::Hot, Media::Nvme, Durability::Durable),
        ep("warm", Tier::Warm, Media::Ssd, Durability::Durable),
        ep("cold", Tier::Cold, Media::Hdd, Durability::Durable),
    ];
    let routing = PartitionRouting::default_for_endpoints(&endpoints);
    // Primary = hot → L0-L1 omitted (implicit), L2-L3 → warm, L4-L6 → cold.
    let routes = routing.to_level_routes(&endpoints, "node", "hot", Arc::new(StdFs));
    assert_eq!(routes.len(), 2);
    assert_eq!(routes[0].levels, 2..4);
    assert_eq!(routes[0].path, PathBuf::from("/warm/node"));
    assert_eq!(routes[1].levels, 4..7);
    assert_eq!(routes[1].path, PathBuf::from("/cold/node"));
}

#[test]
fn to_level_routes_primary_excludes_self_routes() {
    use lsm_tree::fs::StdFs;
    let endpoints = vec![
        ep("hot", Tier::Hot, Media::Nvme, Durability::Durable),
        ep("cold", Tier::Cold, Media::Hdd, Durability::Durable),
    ];
    let routing = PartitionRouting::default_for_endpoints(&endpoints);
    // Primary = cold → only L0-L3 (which map to hot per fallback chain) appear.
    let routes = routing.to_level_routes(&endpoints, "node", "cold", Arc::new(StdFs));
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].levels, 0..4, "L0-L1+L2-L3 both → hot, coalesced");
    assert_eq!(routes[0].path, PathBuf::from("/hot/node"));
}

#[test]
fn to_level_routes_single_endpoint_no_routes() {
    use lsm_tree::fs::StdFs;
    let endpoints = vec![ep("only", Tier::Warm, Media::Hdd, Durability::Durable)];
    let routing = PartitionRouting::default_for_endpoints(&endpoints);
    let routes = routing.to_level_routes(&endpoints, "node", "only", Arc::new(StdFs));
    assert!(
        routes.is_empty(),
        "single-endpoint config: all levels at primary → no routes",
    );
}

#[test]
fn endpoints_used_returns_unique_set() {
    let endpoints = vec![
        ep("hot", Tier::Hot, Media::Nvme, Durability::Durable),
        ep("warm", Tier::Warm, Media::Ssd, Durability::Durable),
        ep("cold", Tier::Cold, Media::Hdd, Durability::Durable),
    ];
    let routing = PartitionRouting::default_for_endpoints(&endpoints);
    let used = routing.endpoints_used();
    assert_eq!(used.len(), 3);
    assert!(used.contains("hot"));
    assert!(used.contains("warm"));
    assert!(used.contains("cold"));
}

#[test]
fn first_level_on_returns_topmost_level() {
    let endpoints = vec![
        ep("hot", Tier::Hot, Media::Nvme, Durability::Durable),
        ep("warm", Tier::Warm, Media::Ssd, Durability::Durable),
        ep("cold", Tier::Cold, Media::Hdd, Durability::Durable),
    ];
    let routing = PartitionRouting::default_for_endpoints(&endpoints);
    assert_eq!(routing.first_level_on("hot"), Some(0));
    assert_eq!(routing.first_level_on("warm"), Some(2));
    assert_eq!(routing.first_level_on("cold"), Some(4));
    assert_eq!(routing.first_level_on("unknown"), None);
}

#[test]
fn msgpack_roundtrip_preserves_routing() {
    let endpoints = vec![
        ep("hot", Tier::Hot, Media::Nvme, Durability::Durable),
        ep("cold", Tier::Cold, Media::Hdd, Durability::Durable),
    ];
    let original = PartitionRouting::default_for_endpoints(&endpoints);
    let encoded = rmp_serde::to_vec(&original).expect("encode");
    let decoded: PartitionRouting = rmp_serde::from_slice(&encoded).expect("decode");
    assert_eq!(decoded, original);
}
