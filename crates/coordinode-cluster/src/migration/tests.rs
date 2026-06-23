use super::*;

fn sample_payload() -> PayloadEstimate {
    PayloadEstimate {
        bytes: 1_000_000,
        node_count: 10_000,
        ef_construction: 200,
    }
}

fn sample_costs() -> CostInputs {
    CostInputs {
        bandwidth_bytes_per_sec: 100_000_000.0,
        hnsw_build_rate_nodes_per_sec: 5_000.0,
        neighbour_byte_cost: 64.0,
    }
}

fn sample_cost_breakdown() -> MigrationCost {
    MigrationCost {
        network_secs: 0.01,
        rebuild_secs: 2.0,
        graph_serialise_secs: 0.0,
    }
}

#[test]
fn online_during_rebuild_default_is_partial_recall() {
    assert_eq!(
        OnlineDuringRebuild::default(),
        OnlineDuringRebuild::PartialRecall
    );
}

#[test]
fn online_during_rebuild_as_str_matches_serde_tag() {
    for variant in [
        OnlineDuringRebuild::Block,
        OnlineDuringRebuild::PartialRecall,
        OnlineDuringRebuild::Offline,
    ] {
        let bytes = rmp_serde::to_vec(&variant).expect("serialise");
        let back: OnlineDuringRebuild = rmp_serde::from_slice(&bytes).expect("deserialise");
        assert_eq!(back, variant);
        let tag = variant.as_str();
        assert!(!tag.is_empty());
        assert!(tag.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'));
    }
}

#[test]
fn transfer_mode_as_str_covers_every_variant() {
    for mode in [TransferMode::RebuildFromData, TransferMode::ShipGraphBytes] {
        let tag = mode.as_str();
        assert!(!tag.is_empty());
        assert!(tag.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'));
    }
}

#[test]
fn migration_cost_total_is_sum_of_components() {
    let cost = MigrationCost {
        network_secs: 0.5,
        rebuild_secs: 1.5,
        graph_serialise_secs: 0.25,
    };
    assert!((cost.total_secs() - 2.25).abs() < 1e-9);
}

#[test]
fn migration_cost_zero_is_neutral() {
    let zero = MigrationCost::zero();
    assert_eq!(zero.total_secs(), 0.0);
}

#[test]
fn migration_plan_roundtrips_through_serde() {
    let plan = MigrationPlan {
        source: "ep-a".to_string(),
        target: "ep-b".to_string(),
        shard: ShardId::ZERO,
        payload: sample_payload(),
        recommended_mode: TransferMode::RebuildFromData,
        cost_breakdown: sample_cost_breakdown(),
        estimated_total_secs: sample_cost_breakdown().total_secs(),
        online_during_rebuild: OnlineDuringRebuild::Block,
    };
    let bytes = rmp_serde::to_vec(&plan).expect("serialise");
    let back: MigrationPlan = rmp_serde::from_slice(&bytes).expect("deserialise");
    assert_eq!(plan, back);
}

#[test]
fn planner_context_roundtrips_through_serde() {
    let ctx = PlannerContext {
        source: "ep-a".to_string(),
        shard: ShardId::ZERO,
        payload: sample_payload(),
        costs: sample_costs(),
        online_policy_override: None,
    };
    let bytes = rmp_serde::to_vec(&ctx).expect("serialise");
    let back: PlannerContext = rmp_serde::from_slice(&bytes).expect("deserialise");
    assert_eq!(ctx, back);
}

#[test]
fn migration_error_carries_caller_information() {
    let err = MigrationPlannerError::SourceNotInTopology("ep-x".to_string());
    let s = format!("{err}");
    assert!(s.contains("ep-x"));
}

#[test]
fn rebuild_cost_has_zero_graph_serialise_term() {
    let cost = estimate_cost(
        TransferMode::RebuildFromData,
        sample_payload(),
        sample_costs(),
    );
    assert_eq!(cost.graph_serialise_secs, 0.0);
    assert!(cost.rebuild_secs > 0.0);
    assert!(cost.network_secs > 0.0);
    assert!((cost.total_secs() - (cost.network_secs + cost.rebuild_secs)).abs() < 1e-9);
}

#[test]
fn ship_graph_bytes_cost_has_zero_rebuild_term() {
    let cost = estimate_cost(
        TransferMode::ShipGraphBytes,
        sample_payload(),
        sample_costs(),
    );
    assert_eq!(cost.rebuild_secs, 0.0);
    assert!(cost.graph_serialise_secs > 0.0);
    assert!(cost.network_secs > 0.0);
}

#[test]
fn network_component_is_payload_over_bandwidth() {
    let payload = sample_payload();
    let inputs = sample_costs();
    let cost = estimate_cost(TransferMode::RebuildFromData, payload, inputs);
    let expected = (payload.bytes as f64) / inputs.bandwidth_bytes_per_sec;
    assert!((cost.network_secs - expected).abs() < 1e-9);
}

#[test]
fn rebuild_component_is_nodes_over_build_rate() {
    let payload = sample_payload();
    let inputs = sample_costs();
    let cost = estimate_cost(TransferMode::RebuildFromData, payload, inputs);
    let expected = (payload.node_count as f64) / inputs.hnsw_build_rate_nodes_per_sec;
    assert!((cost.rebuild_secs - expected).abs() < 1e-9);
}

#[test]
fn pick_recommended_mode_prefers_rebuild_when_build_rate_high() {
    // Very high build rate makes the rebuild path essentially free
    // on the CPU side; the planner must pick RebuildFromData over
    // ShipGraphBytes because shipping serialised graph adds bytes
    // on the wire that rebuild avoids.
    let payload = PayloadEstimate {
        bytes: 10_000_000,
        node_count: 1_000_000,
        ef_construction: 200,
    };
    let inputs = CostInputs {
        bandwidth_bytes_per_sec: 100_000_000.0,
        hnsw_build_rate_nodes_per_sec: 1_000_000_000.0,
        neighbour_byte_cost: 64.0,
    };
    assert_eq!(
        pick_recommended_mode(payload, inputs),
        TransferMode::RebuildFromData
    );
}

#[test]
fn pick_recommended_mode_prefers_ship_when_build_rate_low() {
    // Very slow build rate makes rebuild dominate; even a fat
    // serialised graph payload is cheaper to ship than to recompute.
    let payload = PayloadEstimate {
        bytes: 1_000_000,
        node_count: 1_000_000,
        ef_construction: 400,
    };
    let inputs = CostInputs {
        bandwidth_bytes_per_sec: 1_000_000_000.0,
        hnsw_build_rate_nodes_per_sec: 100.0,
        neighbour_byte_cost: 64.0,
    };
    assert_eq!(
        pick_recommended_mode(payload, inputs),
        TransferMode::ShipGraphBytes
    );
}

#[test]
fn pick_recommended_mode_breaks_ties_in_favour_of_rebuild() {
    // Cook the inputs so both totals are exactly equal: equal
    // network component on both sides, and rebuild_full ==
    // graph_full. That holds when nodes / build_rate ==
    // nodes * neighbour_byte_cost / bandwidth, i.e.
    // bandwidth / build_rate == neighbour_byte_cost.
    let payload = PayloadEstimate {
        bytes: 100_000,
        node_count: 10_000,
        ef_construction: 200,
    };
    let inputs = CostInputs {
        bandwidth_bytes_per_sec: 64_000.0,
        hnsw_build_rate_nodes_per_sec: 1_000.0,
        neighbour_byte_cost: 64.0, // bandwidth/build_rate == 64 == neighbour_byte_cost
    };
    let rebuild = estimate_cost(TransferMode::RebuildFromData, payload, inputs).total_secs();
    let ship = estimate_cost(TransferMode::ShipGraphBytes, payload, inputs).total_secs();
    assert!((rebuild - ship).abs() < 1e-9, "costs must tie exactly");
    assert_eq!(
        pick_recommended_mode(payload, inputs),
        TransferMode::RebuildFromData
    );
}

#[test]
fn planner_returns_no_candidates_on_single_node_topology() {
    use crate::topology::SingleNodeTopology;
    use crate::types::{Modality, TopologyTree};
    let tree = TopologyTree::single_endpoint("ep-only", Tier::Warm);
    let topology = SingleNodeTopology::from_tree(tree);
    let planner = LocalMigrationPlanner::new(topology, Tier::Warm, Modality::Vector);
    let ctx = PlannerContext {
        source: "ep-only".to_string(),
        shard: ShardId::ZERO,
        payload: sample_payload(),
        costs: sample_costs(),
        online_policy_override: None,
    };
    let err = planner.plan(&ctx).expect_err("must reject single-node");
    assert!(matches!(err, MigrationPlannerError::NoCandidates));
}

#[test]
fn planner_picks_lowest_cost_other_endpoint_on_three_endpoint_topology() {
    use crate::topology::SingleNodeTopology;
    use crate::types::{FailureDomain, Modality, TopologyTree};
    let tree = TopologyTree {
        endpoints: vec![
            FailureDomain::local("ep-a", Tier::Warm),
            FailureDomain::local("ep-b", Tier::Warm),
            FailureDomain::local("ep-c", Tier::Warm),
        ],
    };
    let topology = SingleNodeTopology::from_tree(tree);
    let planner = LocalMigrationPlanner::new(topology, Tier::Warm, Modality::Vector);
    let ctx = PlannerContext {
        source: "ep-a".to_string(),
        shard: ShardId::ZERO,
        payload: sample_payload(),
        costs: sample_costs(),
        online_policy_override: None,
    };
    let plan = planner.plan(&ctx).expect("must produce a plan");
    // Source filtered out; target is one of the other two.
    assert_ne!(plan.target, "ep-a");
    assert!(plan.target == "ep-b" || plan.target == "ep-c");
    // Cost components match the cost model for the recommended mode.
    let expected_cost = estimate_cost(plan.recommended_mode, ctx.payload, ctx.costs);
    assert_eq!(plan.cost_breakdown, expected_cost);
    assert!(
        (plan.estimated_total_secs - expected_cost.total_secs()).abs() < 1e-9,
        "estimated_total_secs must equal the breakdown total",
    );
}

#[test]
fn planner_rejects_source_not_in_topology() {
    use crate::topology::SingleNodeTopology;
    use crate::types::{FailureDomain, Modality, TopologyTree};
    let tree = TopologyTree {
        endpoints: vec![
            FailureDomain::local("ep-x", Tier::Warm),
            FailureDomain::local("ep-y", Tier::Warm),
        ],
    };
    let topology = SingleNodeTopology::from_tree(tree);
    let planner = LocalMigrationPlanner::new(topology, Tier::Warm, Modality::Vector);
    let ctx = PlannerContext {
        source: "ep-not-here".to_string(),
        shard: ShardId::ZERO,
        payload: sample_payload(),
        costs: sample_costs(),
        online_policy_override: None,
    };
    let err = planner.plan(&ctx).expect_err("source absent must reject");
    assert!(matches!(err, MigrationPlannerError::SourceNotInTopology(_)));
}

#[test]
fn planner_filters_by_tier() {
    use crate::topology::SingleNodeTopology;
    use crate::types::{FailureDomain, Modality, TopologyTree};
    // Source at Warm, but the planner is asked for Hot candidates.
    // Topology has one Warm and one Hot endpoint; the Hot endpoint
    // is the only candidate, but source is Warm and so the
    // "source in topology" check fails before we even get to pick.
    // That's the expected behaviour: tier mismatches surface as
    // SourceNotInTopology rather than NoCandidates, because the
    // caller shouldn't be running the planner against the wrong
    // tier in the first place.
    let tree = TopologyTree {
        endpoints: vec![
            FailureDomain::local("ep-warm", Tier::Warm),
            FailureDomain::local("ep-hot-1", Tier::Hot),
            FailureDomain::local("ep-hot-2", Tier::Hot),
        ],
    };
    let topology = SingleNodeTopology::from_tree(tree);
    let planner = LocalMigrationPlanner::new(topology, Tier::Hot, Modality::Vector);
    let ctx = PlannerContext {
        source: "ep-hot-1".to_string(),
        shard: ShardId::ZERO,
        payload: sample_payload(),
        costs: sample_costs(),
        online_policy_override: None,
    };
    let plan = planner.plan(&ctx).expect("hot peer must be picked");
    assert_eq!(plan.target, "ep-hot-2");
}

#[test]
fn online_during_rebuild_display_matches_as_str() {
    for variant in [
        OnlineDuringRebuild::Block,
        OnlineDuringRebuild::PartialRecall,
        OnlineDuringRebuild::Offline,
    ] {
        assert_eq!(format!("{variant}"), variant.as_str());
    }
}

#[test]
fn transfer_mode_display_matches_as_str() {
    for mode in [TransferMode::RebuildFromData, TransferMode::ShipGraphBytes] {
        assert_eq!(format!("{mode}"), mode.as_str());
    }
}

#[test]
fn migration_plan_explain_format_is_stable() {
    let plan = MigrationPlan {
        source: "ep-a".to_string(),
        target: "ep-b".to_string(),
        shard: ShardId::ZERO,
        payload: sample_payload(),
        recommended_mode: TransferMode::RebuildFromData,
        cost_breakdown: MigrationCost {
            network_secs: 1.0,
            rebuild_secs: 1.345,
            graph_serialise_secs: 0.0,
        },
        estimated_total_secs: 2.345,
        online_during_rebuild: OnlineDuringRebuild::PartialRecall,
    };
    let line = plan.explain();
    assert_eq!(
        line,
        "migration plan: ep-a -> ep-b shard=0 mode=rebuild_from_data \
             policy=partial_recall total=2.345s"
    );
}

#[test]
fn default_planner_emits_enum_default_policy() {
    use crate::topology::SingleNodeTopology;
    use crate::types::{FailureDomain, Modality, TopologyTree};
    let tree = TopologyTree {
        endpoints: vec![
            FailureDomain::local("ep-a", Tier::Warm),
            FailureDomain::local("ep-b", Tier::Warm),
        ],
    };
    let topology = SingleNodeTopology::from_tree(tree);
    let planner = LocalMigrationPlanner::new(topology, Tier::Warm, Modality::Vector);
    let ctx = PlannerContext {
        source: "ep-a".to_string(),
        shard: ShardId::ZERO,
        payload: sample_payload(),
        costs: sample_costs(),
        online_policy_override: None,
    };
    let plan = planner.plan(&ctx).expect("plan");
    assert_eq!(plan.online_during_rebuild, OnlineDuringRebuild::default());
}

#[test]
fn planner_default_override_writes_into_plan() {
    use crate::topology::SingleNodeTopology;
    use crate::types::{FailureDomain, Modality, TopologyTree};
    let tree = TopologyTree {
        endpoints: vec![
            FailureDomain::local("ep-a", Tier::Warm),
            FailureDomain::local("ep-b", Tier::Warm),
        ],
    };
    let topology = SingleNodeTopology::from_tree(tree);
    let planner = LocalMigrationPlanner::new(topology, Tier::Warm, Modality::Vector)
        .with_online_policy(OnlineDuringRebuild::Block);
    assert_eq!(planner.default_online_policy(), OnlineDuringRebuild::Block);
    let ctx = PlannerContext {
        source: "ep-a".to_string(),
        shard: ShardId::ZERO,
        payload: sample_payload(),
        costs: sample_costs(),
        online_policy_override: None,
    };
    let plan = planner.plan(&ctx).expect("plan");
    assert_eq!(plan.online_during_rebuild, OnlineDuringRebuild::Block);
}

#[test]
fn context_override_beats_planner_default() {
    use crate::topology::SingleNodeTopology;
    use crate::types::{FailureDomain, Modality, TopologyTree};
    let tree = TopologyTree {
        endpoints: vec![
            FailureDomain::local("ep-a", Tier::Warm),
            FailureDomain::local("ep-b", Tier::Warm),
        ],
    };
    let topology = SingleNodeTopology::from_tree(tree);
    let planner = LocalMigrationPlanner::new(topology, Tier::Warm, Modality::Vector)
        .with_online_policy(OnlineDuringRebuild::Block);
    let ctx = PlannerContext {
        source: "ep-a".to_string(),
        shard: ShardId::ZERO,
        payload: sample_payload(),
        costs: sample_costs(),
        online_policy_override: Some(OnlineDuringRebuild::Offline),
    };
    let plan = planner.plan(&ctx).expect("plan");
    assert_eq!(plan.online_during_rebuild, OnlineDuringRebuild::Offline);
}

#[test]
fn estimate_cost_tolerates_zero_bandwidth_without_panicking() {
    // Pathological input: zero bandwidth would divide by zero.
    // The implementation clamps to MIN_POSITIVE so the result is
    // a huge but finite cost rather than NaN / infinity.
    let inputs = CostInputs {
        bandwidth_bytes_per_sec: 0.0,
        hnsw_build_rate_nodes_per_sec: 1_000.0,
        neighbour_byte_cost: 64.0,
    };
    let cost = estimate_cost(TransferMode::RebuildFromData, sample_payload(), inputs);
    assert!(cost.network_secs.is_finite());
    assert!(cost.total_secs() > 0.0);
}
