//! Migration planning types: cost model + transfer-mode enum.
//!
//! The migration planner is the component that, given an endpoint
//! that crossed a capacity threshold (or any other admin-driven
//! trigger), proposes a concrete `MigrationPlan` describing where to
//! move a shard, how to move it, and how much each component of that
//! move will cost.
//!
//! This module ships the data types only. The planner trait and its
//! concrete implementation live in follow-up commits; pinning the
//! shape of the plan and the cost decomposition first means the
//! planner can be swapped without churning every caller.
//!
//! ## Cost decomposition
//!
//! Three independent cost contributions, all in seconds, summed by
//! [`MigrationCost::total_secs`]:
//!
//! - **network_secs**: wall-clock to ship `payload.bytes` over a
//!   link with `bandwidth_bytes_per_sec`. Always charged.
//! - **rebuild_secs**: wall-clock to rebuild the HNSW graph from
//!   the f32 truth tier on the target. Charged when
//!   [`TransferMode::RebuildFromData`] is picked. Zero for
//!   [`TransferMode::ShipGraphBytes`].
//! - **graph_serialise_secs**: wall-clock to ship the serialised
//!   HNSW graph bytes over the same link. Charged when
//!   [`TransferMode::ShipGraphBytes`] is picked. Zero for
//!   [`TransferMode::RebuildFromData`].
//!
//! The vector truth-tier bytes (`payload.bytes`) are charged
//! identically in both modes because both modes need the raw vectors
//! on the target (graph-bytes mode still ships the f32 source of
//! truth; serialised neighbours alone are not a complete index).

use std::fmt;

use coordinode_storage::engine::config::Tier;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::topology::ClusterTopology;
use crate::types::{CrushRule, EndpointId, Modality, ShardId};

/// One end-to-end migration proposal: move this shard from this
/// endpoint to that endpoint, using this transfer mode, at this
/// estimated cost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Endpoint currently authoritative for the shard.
    pub source: EndpointId,
    /// Endpoint that should take over.
    pub target: EndpointId,
    /// Shard being moved. CE single-shard plans always carry
    /// [`ShardId::ZERO`]; the field stays in the type so that the
    /// multi-shard impl can plug in without a breaking API change.
    pub shard: ShardId,
    /// Bytes-on-disk and HNSW-node-count summary of the shard.
    pub payload: PayloadEstimate,
    /// Transfer mode the planner picked as cheapest for the given
    /// payload and cost inputs.
    pub recommended_mode: TransferMode,
    /// Detailed per-component cost breakdown for the recommended
    /// mode. `cost_breakdown.total_secs()` equals
    /// `estimated_total_secs`.
    pub cost_breakdown: MigrationCost,
    /// Total wall-clock estimate (seconds). Cached for convenience;
    /// always equals `cost_breakdown.total_secs()`.
    pub estimated_total_secs: f64,
    /// Read-side contract on the target while the rebuild is in
    /// flight. See [`OnlineDuringRebuild`]. Defaults to the
    /// enum default when older on-wire formats lack the field.
    #[serde(default)]
    pub online_during_rebuild: OnlineDuringRebuild,
}

/// Read-side contract on the migration target while the HNSW graph
/// is being rebuilt. The cluster ships the policy as part of the
/// plan so the executor on the target side knows up front whether
/// to fail, partially-serve, or block search requests for the
/// shard during the rebuild window.
///
/// The vocabulary mirrors the index-DDL `OnlineDuringBuild` enum
/// for fresh `CREATE VECTOR INDEX` builds so operator tooling can
/// reason about both lifecycles with one mental model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnlineDuringRebuild {
    /// Target rejects searches for the shard until the rebuild
    /// finishes. Strongest correctness; worst latency tail. Pick
    /// this for shards where partial answers are unacceptable.
    Block,
    /// Target serves searches against the partially-rebuilt graph.
    /// Recall drops below the steady-state target during the
    /// rebuild window; the caller learns via a partial-result flag
    /// on the response. Pick this for best-effort workloads.
    /// Default for new plans.
    #[default]
    PartialRecall,
    /// Target answers no search for the shard until the rebuild
    /// completes. Reads on the shard fall back to the source (which
    /// has not yet been decommissioned) via the routing layer.
    Offline,
}

impl OnlineDuringRebuild {
    /// String tag used in logs and EXPLAIN output. Lower-case
    /// snake-case, matching the serde representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::PartialRecall => "partial_recall",
            Self::Offline => "offline",
        }
    }
}

impl fmt::Display for OnlineDuringRebuild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for TransferMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl MigrationPlan {
    /// Single-line operator-facing summary of the plan: source,
    /// target, shard id, recommended transfer mode, online-during-
    /// rebuild policy, and the estimated total cost in seconds.
    /// Pinned format so log-scraping tools can parse it.
    ///
    /// ```text
    /// migration plan: ep-a -> ep-b shard=0 mode=rebuild_from_data policy=partial_recall total=2.345s
    /// ```
    pub fn explain(&self) -> String {
        format!(
            "migration plan: {} -> {} shard={} mode={} policy={} total={:.3}s",
            self.source,
            self.target,
            self.shard.0,
            self.recommended_mode,
            self.online_during_rebuild,
            self.estimated_total_secs,
        )
    }
}

/// How to physically move a shard from source to target.
///
/// The choice is a runtime decision, not a build-time policy: the
/// migration planner computes the cost of each mode against the
/// concrete payload and recommends whichever is cheaper. Operators
/// can override the recommendation when policy demands it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferMode {
    /// Ship the raw f32 vectors and rebuild the HNSW graph on the
    /// target. Cheaper bytes-on-the-wire, more CPU on the target.
    /// Insensitive to the on-disk HNSW format on either side, which
    /// makes it the safe default during rolling upgrades.
    RebuildFromData,

    /// Ship the serialised HNSW graph bytes alongside the f32
    /// vectors. Higher network cost, near-zero rebuild cost on the
    /// target. Useful when the target is CPU-constrained or when
    /// `ef_construction` is large enough that rebuild dominates.
    ShipGraphBytes,
}

impl TransferMode {
    /// String tag used in logs and EXPLAIN output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RebuildFromData => "rebuild_from_data",
            Self::ShipGraphBytes => "ship_graph_bytes",
        }
    }
}

/// Per-component cost decomposition for a migration. All values are
/// wall-clock seconds. Always satisfies
/// `total_secs() == network_secs + rebuild_secs + graph_serialise_secs`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MigrationCost {
    /// Wall-clock to transfer the raw f32 vector payload over the
    /// link.
    pub network_secs: f64,
    /// Wall-clock to rebuild the HNSW graph on the target from the
    /// shipped vectors. Zero when the chosen mode is
    /// [`TransferMode::ShipGraphBytes`].
    pub rebuild_secs: f64,
    /// Wall-clock to ship the serialised HNSW graph bytes on top of
    /// the f32 vectors. Zero when the chosen mode is
    /// [`TransferMode::RebuildFromData`].
    pub graph_serialise_secs: f64,
}

impl MigrationCost {
    /// Sum of the three components.
    pub fn total_secs(&self) -> f64 {
        self.network_secs + self.rebuild_secs + self.graph_serialise_secs
    }

    /// All-zeroes cost. Used as a sentinel for "this plan has not
    /// been priced yet" in test fixtures; not a meaningful production
    /// value.
    pub fn zero() -> Self {
        Self {
            network_secs: 0.0,
            rebuild_secs: 0.0,
            graph_serialise_secs: 0.0,
        }
    }
}

/// Size of the shard payload that needs to move. Drives the cost
/// formula.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PayloadEstimate {
    /// Raw f32 vector bytes (`node_count * dim * 4`).
    pub bytes: u64,
    /// Number of HNSW graph nodes (one per stored vector). Used to
    /// scale the rebuild cost in [`crate::migration`].
    pub node_count: u64,
    /// HNSW `ef_construction` parameter at the source. Carried in
    /// the payload because cost on the target depends on it (higher
    /// ef_construction means more comparisons per insert during
    /// rebuild).
    pub ef_construction: u32,
}

/// Inputs the cost model needs to translate a payload into seconds.
/// These come from per-link / per-node observability and are
/// supplied by the caller; the planner itself does not measure them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostInputs {
    /// Effective transfer bandwidth source -> target, in bytes per
    /// second. Used by the network and graph-serialise components.
    pub bandwidth_bytes_per_sec: f64,
    /// HNSW build throughput on the target, in nodes inserted per
    /// second at the supplied `ef_construction`. Used by the rebuild
    /// component.
    pub hnsw_build_rate_nodes_per_sec: f64,
    /// Average bytes per HNSW node in the serialised graph
    /// (neighbours + level metadata). Used by the graph-serialise
    /// component.
    pub neighbour_byte_cost: f64,
}

/// Context passed into [`crate::migration::MigrationPlanner::plan`]
/// (defined in a follow-up). Captures everything the planner needs
/// about the trigger that asked it to act.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannerContext {
    /// Endpoint that crossed the capacity threshold (or was manually
    /// nominated) and is the source of the migration.
    pub source: EndpointId,
    /// Shard whose home should change. Single-shard CE always passes
    /// [`ShardId::ZERO`].
    pub shard: ShardId,
    /// Estimated payload size and HNSW node count for the shard.
    pub payload: PayloadEstimate,
    /// Cost inputs for the source -> any-target link. The planner
    /// applies these to each candidate target. A future enhancement
    /// will allow per-target overrides.
    pub costs: CostInputs,
    /// Per-context override of the read-side rebuild policy. When
    /// `Some(_)`, the planner writes that variant into the emitted
    /// plan regardless of the planner's configured default. When
    /// `None`, the planner falls back to its own configured default
    /// (set via [`LocalMigrationPlanner::with_online_policy`]) and
    /// then to [`OnlineDuringRebuild::default`].
    #[serde(default)]
    pub online_policy_override: Option<OnlineDuringRebuild>,
}

/// Errors returned by the migration planner.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MigrationPlannerError {
    /// No candidate target endpoint exists (only one node in the
    /// cluster, or every other candidate was filtered out by
    /// placement rules / tier mismatch / health). Surfaces as a
    /// no-op from the caller's perspective.
    #[error("no candidate target endpoint for migration")]
    NoCandidates,

    /// Source endpoint not present in the topology. Indicates a
    /// caller bug.
    #[error("source endpoint {0:?} is not present in the topology")]
    SourceNotInTopology(EndpointId),
}

/// Compute the per-component cost of one transfer mode for a given
/// payload and cost inputs. Pure function; the formulas are
/// documented at the module level under "Cost decomposition".
///
/// All three components are non-negative and finite for any non-zero
/// bandwidth and non-zero build rate. The caller is responsible for
/// supplying positive inputs; a zero in any denominator would yield
/// infinity and a downstream comparator would crash. The function
/// guards by clamping the denominator to a tiny positive value, so
/// pathological inputs surface as enormous (but finite) costs that
/// any sane candidate beats.
pub fn estimate_cost(
    mode: TransferMode,
    payload: PayloadEstimate,
    inputs: CostInputs,
) -> MigrationCost {
    // Floor each denominator at 1.0 (one byte/sec, one node/sec). Real
    // links and real builders are well above this; the floor only
    // matters for pathological zero / near-zero inputs, where we want
    // a huge-but-finite cost rather than infinity so the comparator
    // in pick_recommended_mode stays well-defined.
    let bandwidth = inputs.bandwidth_bytes_per_sec.max(1.0);
    let build_rate = inputs.hnsw_build_rate_nodes_per_sec.max(1.0);

    let network_secs = (payload.bytes as f64) / bandwidth;
    let nodes = payload.node_count as f64;
    let rebuild_full = nodes / build_rate;
    let graph_full = nodes * inputs.neighbour_byte_cost.max(0.0) / bandwidth;

    match mode {
        TransferMode::RebuildFromData => MigrationCost {
            network_secs,
            rebuild_secs: rebuild_full,
            graph_serialise_secs: 0.0,
        },
        TransferMode::ShipGraphBytes => MigrationCost {
            network_secs,
            rebuild_secs: 0.0,
            graph_serialise_secs: graph_full,
        },
    }
}

/// Pick the transfer mode that minimises total cost for the given
/// payload and cost inputs. Ties resolve in favour of
/// [`TransferMode::RebuildFromData`] because the rebuild path makes
/// no assumption about the serialised HNSW format being compatible
/// across binary versions, which is the safer default during
/// rolling upgrades.
pub fn pick_recommended_mode(payload: PayloadEstimate, inputs: CostInputs) -> TransferMode {
    let rebuild = estimate_cost(TransferMode::RebuildFromData, payload, inputs).total_secs();
    let ship = estimate_cost(TransferMode::ShipGraphBytes, payload, inputs).total_secs();
    if ship < rebuild {
        TransferMode::ShipGraphBytes
    } else {
        TransferMode::RebuildFromData
    }
}

/// Migration planner trait: given a context describing the trigger
/// (source endpoint, shard, payload, cost inputs), propose the
/// cheapest single migration. Single-method trait so the EE
/// implementation can layer over an entirely different topology
/// without inheriting the CE planner's local-tier logic.
pub trait MigrationPlanner: Send + Sync {
    /// Propose a migration for the trigger encoded in `ctx`. Returns
    /// [`MigrationPlannerError::NoCandidates`] when there is no
    /// other endpoint at the same tier that could take the shard.
    fn plan(&self, ctx: &PlannerContext) -> Result<MigrationPlan, MigrationPlannerError>;
}

/// CE migration planner over a [`ClusterTopology`]. Enumerates
/// candidate targets via [`ClusterTopology::placement_candidates`]
/// with [`CrushRule::LocalTier`], filters the source out, picks the
/// candidate with the lowest total cost.
///
/// `tier` and `modality` are captured at construction time because
/// the trigger that wakes the planner is typically a capacity event
/// scoped to a specific endpoint tier (overfull NVMe -> spill to
/// another NVMe at the same tier).
pub struct LocalMigrationPlanner<T: ClusterTopology> {
    topology: T,
    tier: Tier,
    modality: Modality,
    default_online_policy: OnlineDuringRebuild,
}

impl<T: ClusterTopology> LocalMigrationPlanner<T> {
    /// Build a planner over the given topology, scoped to the tier
    /// and modality of the source endpoint. The default online
    /// policy is [`OnlineDuringRebuild::default`]; override with
    /// [`Self::with_online_policy`].
    pub fn new(topology: T, tier: Tier, modality: Modality) -> Self {
        Self {
            topology,
            tier,
            modality,
            default_online_policy: OnlineDuringRebuild::default(),
        }
    }

    /// Builder method: configure the default online-during-rebuild
    /// policy for plans emitted by this planner. A per-context
    /// override on [`PlannerContext::online_policy_override`] still
    /// takes precedence over this default.
    pub fn with_online_policy(mut self, policy: OnlineDuringRebuild) -> Self {
        self.default_online_policy = policy;
        self
    }

    /// Borrow the underlying topology (handy for tests / EXPLAIN).
    pub fn topology(&self) -> &T {
        &self.topology
    }

    /// The default online-during-rebuild policy the planner will
    /// write into plans when the context does not override it.
    pub fn default_online_policy(&self) -> OnlineDuringRebuild {
        self.default_online_policy
    }
}

impl<T: ClusterTopology> MigrationPlanner for LocalMigrationPlanner<T> {
    fn plan(&self, ctx: &PlannerContext) -> Result<MigrationPlan, MigrationPlannerError> {
        let candidates = self
            .topology
            .placement_candidates(&CrushRule::local_tier(), self.modality, self.tier)
            .map_err(|_| MigrationPlannerError::NoCandidates)?;

        if !candidates.iter().any(|ep| ep == &ctx.source) {
            return Err(MigrationPlannerError::SourceNotInTopology(
                ctx.source.clone(),
            ));
        }

        let mut best: Option<(EndpointId, TransferMode, MigrationCost, f64)> = None;
        for candidate in candidates.into_iter().filter(|ep| ep != &ctx.source) {
            let mode = pick_recommended_mode(ctx.payload, ctx.costs);
            let cost = estimate_cost(mode, ctx.payload, ctx.costs);
            let total = cost.total_secs();
            let take = match &best {
                None => true,
                Some((_, _, _, best_total)) => total < *best_total,
            };
            if take {
                best = Some((candidate, mode, cost, total));
            }
        }

        let (target, mode, cost, total) = best.ok_or(MigrationPlannerError::NoCandidates)?;
        let online_policy = ctx
            .online_policy_override
            .unwrap_or(self.default_online_policy);
        Ok(MigrationPlan {
            source: ctx.source.clone(),
            target,
            shard: ctx.shard,
            payload: ctx.payload,
            recommended_mode: mode,
            cost_breakdown: cost,
            estimated_total_secs: total,
            online_during_rebuild: online_policy,
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
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
}
