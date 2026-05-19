//! Property-based invariants for the CE topology implementations.
//! Small case budget per property keeps these snappy in CI; each
//! property pins one shape-of-the-world contract that the trait must
//! honour regardless of input.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use coordinode_cluster::{
    ClusterTopology, CrushRule, FailureDomain, Modality, ShardId, ShardRouting, SingleNodeTopology,
    SingleShardRouting, TopologyTree,
};
use coordinode_storage::engine::config::Tier;
use proptest::prelude::*;

fn tier_strategy() -> impl Strategy<Value = Tier> {
    prop_oneof![
        Just(Tier::Memory),
        Just(Tier::HotCache),
        Just(Tier::Hot),
        Just(Tier::Warm),
        Just(Tier::Cold),
    ]
}

fn endpoint_strategy() -> impl Strategy<Value = (String, Tier)> {
    ("[a-z]{1,8}", tier_strategy()).prop_map(|(s, t)| (s, t))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]

    /// Every endpoint inserted at tier `T` is returned by
    /// `placement_candidates(LocalTier, _, T)` — and zero endpoints
    /// at any other tier leak into that result.
    #[test]
    fn placement_candidates_local_tier_partitions_by_tier(
        endpoints in prop::collection::vec(endpoint_strategy(), 1..16),
        query_tier in tier_strategy(),
    ) {
        // De-dup endpoint ids — the storage layer would reject
        // duplicates, so we mimic that here.
        let mut seen = std::collections::BTreeSet::new();
        let mut leaves = Vec::new();
        for (id, tier) in &endpoints {
            if seen.insert(id.clone()) {
                leaves.push(FailureDomain::local(id.clone(), *tier));
            }
        }
        let tree = TopologyTree { endpoints: leaves.clone() };
        let topo = SingleNodeTopology::from_tree(tree);

        let hits = topo
            .placement_candidates(&CrushRule::local_tier(), Modality::Node, query_tier)
            .unwrap();
        let expected: std::collections::BTreeSet<_> = leaves
            .iter()
            .filter(|d| d.tier == query_tier)
            .map(|d| d.endpoint.clone())
            .collect();
        let actual: std::collections::BTreeSet<_> = hits.into_iter().collect();
        prop_assert_eq!(actual, expected);
    }

    /// `SingleNodeTopology::from_tree` always exposes exactly one
    /// shard with id 0, regardless of how many endpoints the tree
    /// contains.
    #[test]
    fn from_tree_yields_exactly_one_shard(
        n_endpoints in 0usize..32,
    ) {
        let leaves: Vec<_> = (0..n_endpoints)
            .map(|i| FailureDomain::local(format!("ep-{i}"), Tier::Warm))
            .collect();
        let topo = SingleNodeTopology::from_tree(TopologyTree { endpoints: leaves });
        prop_assert_eq!(topo.shards().len(), 1);
        prop_assert_eq!(topo.shards()[0].id, ShardId::ZERO);
    }

    /// `SingleShardRouting::shard_for_key` is a constant function —
    /// every input lands at shard 0. Regression guard for future
    /// "smart" routing that would silently break CE single-shard.
    #[test]
    fn single_shard_routing_is_constant(
        key in prop::collection::vec(any::<u8>(), 0..256),
    ) {
        let r = SingleShardRouting::new();
        prop_assert_eq!(r.shard_for_key(&key), ShardId::ZERO);
    }
}
