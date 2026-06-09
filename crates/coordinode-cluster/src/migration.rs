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

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::{EndpointId, ShardId};

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
