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
//! ## Example
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
pub mod topology;
pub mod types;

pub use error::{TopologyError, TopologyResult};
pub use migration::{
    estimate_cost, pick_recommended_mode, CostInputs, LocalMigrationPlanner, MigrationCost,
    MigrationPlan, MigrationPlanner, MigrationPlannerError, PayloadEstimate, PlannerContext,
    TransferMode,
};
pub use routing::{ShardRouting, SingleShardRouting};
pub use topology::{ClusterTopology, SingleNodeTopology};
pub use types::{
    CrushRule, EndpointId, FailureDomain, Modality, NodeAddr, ServerId, ShardDescriptor, ShardId,
    TopologyTree,
};
