//! Layer 6 — cluster topology + shard routing.
//!
//! This crate sits at the top of the storage stack (see
//! `arch/core/storage-stack.md` §Layer 6). It owns the 5-level
//! failure-domain tree (`geo → dc → rack → server → endpoint`,
//! `arch/placement/crush.md`) and the shard-to-node map. Two traits:
//!
//! - [`ClusterTopology`] — topology tree + shard descriptors +
//!   placement candidate sets. Consumed by Layer 2 (placement) and
//!   Layer 5 (shard-aware planning).
//! - [`ShardRouting`] — routing key → shard id resolution. Consumed
//!   by Layer 5 (query engine).
//!
//! The trait surface is identical across CE and EE; only the
//! implementations differ. CE ships:
//!
//! - [`SingleNodeTopology`] — degenerate `srv-only` tree, one shard,
//!   placement candidates from the local endpoint set.
//! - [`SingleShardRouting`] — every key lands at
//!   [`ShardId::ZERO`].
//!
//! Phase 2 multi-node CE will add a `Multi*` impl of the same trait;
//! Phase 3 EE will add `CrushTopology` / `MultiShardRouting` with
//! full CRUSH placement rules. The query / storage layers above
//! depend on the trait, so neither swap requires call-site changes.
//!
//! ## Migration planner
//!
//! On top of the topology + routing traits, the [`migration`] module
//! ships the data plane that decides when to move a shard from one
//! endpoint to another:
//!
//! - [`MigrationPlanner`] is a one-method trait. Implementations
//!   propose a single migration for the trigger encoded in
//!   [`PlannerContext`]. The EE topology will plug a CRUSH-aware
//!   implementation in here.
//! - [`LocalMigrationPlanner`] is the CE implementation. It
//!   enumerates candidate targets via
//!   [`ClusterTopology::placement_candidates`] at the configured
//!   tier, filters the source endpoint out, and picks the candidate
//!   with the lowest total cost.
//! - [`MigrationCost`] decomposes the cost in seconds as
//!   `network + rebuild + graph_serialise`. The HNSW rebuild
//!   contribution lives as its own line item rather than being
//!   folded into a single total, so an operator can see at a glance
//!   whether the bottleneck is bandwidth or CPU at the destination.
//! - [`TransferMode`] captures the two ways a shard can physically
//!   move: ship the f32 truth tier and rebuild HNSW on the target,
//!   or also ship the serialised HNSW graph bytes. The planner picks
//!   the cheaper one via [`pick_recommended_mode`].
//!
//! ### Migration planner example
//!
//! ```
//! use coordinode_cluster::{
//!     CostInputs, FailureDomain, LocalMigrationPlanner, MigrationPlanner, Modality,
//!     PayloadEstimate, PlannerContext, ShardId, SingleNodeTopology, TopologyTree,
//! };
//! use coordinode_storage::engine::config::Tier;
//!
//! // Three Warm-tier endpoints, mirroring three storage nodes.
//! let tree = TopologyTree {
//!     endpoints: vec![
//!         FailureDomain::local("ep-a", Tier::Warm),
//!         FailureDomain::local("ep-b", Tier::Warm),
//!         FailureDomain::local("ep-c", Tier::Warm),
//!     ],
//! };
//! let topology = SingleNodeTopology::from_tree(tree);
//! let planner = LocalMigrationPlanner::new(topology, Tier::Warm, Modality::Vector);
//!
//! // Capacity gate fired on ep-a; build a context describing the
//! // shard payload and the link cost inputs.
//! let ctx = PlannerContext {
//!     source: "ep-a".to_string(),
//!     shard: ShardId::ZERO,
//!     payload: PayloadEstimate {
//!         bytes: 4_000_000,
//!         node_count: 10_000,
//!         ef_construction: 200,
//!     },
//!     costs: CostInputs {
//!         bandwidth_bytes_per_sec: 100_000_000.0,
//!         hnsw_build_rate_nodes_per_sec: 5_000.0,
//!         neighbour_byte_cost: 64.0,
//!     },
//!     online_policy_override: None,
//! };
//!
//! let plan = planner.plan(&ctx).expect("plan must materialise");
//! assert_eq!(plan.source, "ep-a");
//! assert!(plan.target == "ep-b" || plan.target == "ep-c");
//! assert!(plan.estimated_total_secs.is_finite());
//! ```
//!
//! ### Online-during-rebuild policy
//!
//! [`OnlineDuringRebuild`] picks the contract that the migration
//! target honours for reads against the moved shard while its HNSW
//! graph is still being rebuilt:
//!
//! - `Block`: target rejects searches until the rebuild finishes.
//!   Strongest correctness, worst tail latency.
//! - `PartialRecall`: target serves against the partially-rebuilt
//!   graph, surfaces a partial-result flag to the caller. Default.
//! - `Offline`: target answers no search for the shard until the
//!   rebuild completes; reads fall back to the source via routing.
//!
//! The policy is resolved at plan time with precedence
//! `context override > planner default > enum default`. Use
//! [`LocalMigrationPlanner::with_online_policy`] to set the
//! planner-side default; use [`PlannerContext::online_policy_override`]
//! to pick a per-plan value.
//!
//! ```
//! use coordinode_cluster::{
//!     CostInputs, FailureDomain, LocalMigrationPlanner, MigrationPlanner, Modality,
//!     OnlineDuringRebuild, PayloadEstimate, PlannerContext, ShardId, SingleNodeTopology,
//!     TopologyTree,
//! };
//! use coordinode_storage::engine::config::Tier;
//!
//! let tree = TopologyTree {
//!     endpoints: vec![
//!         FailureDomain::local("ep-a", Tier::Warm),
//!         FailureDomain::local("ep-b", Tier::Warm),
//!     ],
//! };
//! let topology = SingleNodeTopology::from_tree(tree);
//! let planner = LocalMigrationPlanner::new(topology, Tier::Warm, Modality::Vector)
//!     .with_online_policy(OnlineDuringRebuild::Block);
//!
//! // Planner-side default: every plan inherits Block when the
//! // context does not override.
//! let plan = planner.plan(&PlannerContext {
//!     source: "ep-a".to_string(),
//!     shard: ShardId::ZERO,
//!     payload: PayloadEstimate { bytes: 1_000, node_count: 100, ef_construction: 200 },
//!     costs: CostInputs {
//!         bandwidth_bytes_per_sec: 1.0e8,
//!         hnsw_build_rate_nodes_per_sec: 5_000.0,
//!         neighbour_byte_cost: 64.0,
//!     },
//!     online_policy_override: None,
//! }).expect("plan");
//! assert_eq!(plan.online_during_rebuild, OnlineDuringRebuild::Block);
//!
//! // Per-context override beats the planner default.
//! let plan = planner.plan(&PlannerContext {
//!     source: "ep-a".to_string(),
//!     shard: ShardId::ZERO,
//!     payload: PayloadEstimate { bytes: 1_000, node_count: 100, ef_construction: 200 },
//!     costs: CostInputs {
//!         bandwidth_bytes_per_sec: 1.0e8,
//!         hnsw_build_rate_nodes_per_sec: 5_000.0,
//!         neighbour_byte_cost: 64.0,
//!     },
//!     online_policy_override: Some(OnlineDuringRebuild::Offline),
//! }).expect("plan");
//! assert_eq!(plan.online_during_rebuild, OnlineDuringRebuild::Offline);
//! ```
//!
//! ## Topology + routing example
//!
//! ```
//! use coordinode_cluster::{
//!     ClusterTopology, CrushRule, Modality, ShardId, SingleNodeTopology,
//! };
//! use coordinode_storage::engine::config::{
//!     Durability, EndpointConfig, Media, StorageConfig, Tier,
//! };
//!
//! let cfg = StorageConfig::with_endpoints(vec![
//!     EndpointConfig::new(
//!         "ep-hot", std::path::Path::new("/tmp/nvme"),
//!         Media::Nvme, Durability::Volatile, Tier::Hot,
//!     ),
//!     EndpointConfig::new(
//!         "ep-warm", std::path::Path::new("/tmp/hdd"),
//!         Media::Hdd, Durability::Durable, Tier::Warm,
//!     ),
//! ]);
//! let topology = SingleNodeTopology::from_storage(&cfg);
//!
//! // One shard, leader is the local node.
//! assert_eq!(topology.shards().len(), 1);
//! assert_eq!(topology.shard_leader(ShardId::ZERO).unwrap().server, "local");
//!
//! // Hot-tier candidates only contain the NVMe endpoint.
//! let candidates = topology
//!     .placement_candidates(&CrushRule::local_tier(), Modality::Vector, Tier::Hot)
//!     .unwrap();
//! assert_eq!(candidates, vec!["ep-hot".to_owned()]);
//! ```

#![deny(missing_docs)]

pub mod error;
pub mod migration;
pub mod routing;
pub mod shard_map;
pub mod state_machine;
pub mod topology;
pub mod types;
pub mod vector_routing;

pub use error::{TopologyError, TopologyResult};
pub use migration::{
    estimate_cost, pick_recommended_mode, CostInputs, LocalMigrationPlanner, MigrationCost,
    MigrationPlan, MigrationPlanner, MigrationPlannerError, OnlineDuringRebuild, PayloadEstimate,
    PlannerContext, TransferMode,
};
pub use routing::{ShardRouting, SingleShardRouting};
pub use shard_map::{ChunkAssignment, ChunkAssignmentTable, ChunkRange};
pub use state_machine::{
    BackendError, ContextId, HealthEvent, OperationFilter, OperationId, OperationState,
    OperationStatus, OperationSummary, Progress, StateLabel, StateMachineBackend,
    TransitionRequest,
};
pub use topology::{ClusterTopology, SingleNodeTopology};
pub use types::{
    CrushRule, EndpointId, FailureDomain, Modality, NodeAddr, ServerId, ShardDescriptor, ShardId,
    TopologyTree,
};
pub use vector_routing::{PartitionId, PartitionSet, SinglePartitionRouter, VectorShardRouter};
