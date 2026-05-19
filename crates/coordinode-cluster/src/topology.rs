//! [`ClusterTopology`] trait + CE [`SingleNodeTopology`] impl.

use coordinode_storage::engine::config::{EndpointConfig, StorageConfig, Tier};

use crate::error::{TopologyError, TopologyResult};
use crate::types::{
    CrushRule, EndpointId, FailureDomain, Modality, NodeAddr, ShardDescriptor, ShardId,
    TopologyTree,
};

/// Layer 6 cluster topology contract.
///
/// Consumed by Layer 5 (query engine — for shard routing) and Layer 2
/// (tier-aware partition — for placement candidate sets). The CE and
/// EE binaries share this trait verbatim; only the concrete impl
/// changes.
///
/// Every method is read-only — topology mutation (adding nodes,
/// changing CRUSH rules) is a separate admin path, not part of the
/// hot read/write contract.
pub trait ClusterTopology: Send + Sync {
    /// Borrow the underlying 5-level failure-domain tree. Callers
    /// must NOT depend on the tree being hierarchical in storage —
    /// CE returns a flat `Vec<FailureDomain>` and the EE impl can
    /// expose a richer shape under the same accessor.
    fn topology_tree(&self) -> &TopologyTree;

    /// Every shard in the cluster, in id order. CE returns a single
    /// `[ShardDescriptor]` with id [`ShardId::ZERO`].
    fn shards(&self) -> &[ShardDescriptor];

    /// Resolve a shard id to the node currently authoritative for it
    /// (Raft leader in EE, local node in CE).
    ///
    /// Returns [`TopologyError::ShardNotFound`] for ids outside the
    /// active shard set.
    fn shard_leader(&self, shard: ShardId) -> TopologyResult<NodeAddr>;

    /// Endpoint candidate set for placing data of a given `modality`
    /// at the given `tier`, subject to the supplied placement
    /// `rule`. CE returns the local-server endpoints at the matching
    /// tier; EE applies the full CRUSH rule grammar.
    ///
    /// EE-only rule variants on a CE impl surface as
    /// [`TopologyError::EeOnly`] so the gRPC layer can map them to
    /// `UNIMPLEMENTED`.
    fn placement_candidates(
        &self,
        rule: &CrushRule,
        modality: Modality,
        tier: Tier,
    ) -> TopologyResult<Vec<EndpointId>>;
}

/// CE single-node topology — one server, every endpoint hangs off it,
/// one Raft group, one shard.
///
/// Built from a [`StorageConfig`] (the operator-supplied endpoint
/// list) so the topology layer never invents endpoints — they always
/// reflect what the storage engine was opened with.
///
/// # Examples
///
/// ```
/// use coordinode_cluster::{ClusterTopology, SingleNodeTopology};
/// use coordinode_storage::engine::config::{
///     Durability, EndpointConfig, Media, StorageConfig, Tier,
/// };
///
/// let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
///     "ep-warm", std::path::Path::new("/tmp/store"),
///     Media::Hdd, Durability::Durable, Tier::Warm,
/// )]);
/// let topology = SingleNodeTopology::from_storage(&cfg);
/// assert_eq!(topology.shards().len(), 1);
/// assert_eq!(topology.topology_tree().endpoint_count(), 1);
/// ```
pub struct SingleNodeTopology {
    tree: TopologyTree,
    shards: Vec<ShardDescriptor>,
}

impl SingleNodeTopology {
    /// Build a topology from a [`StorageConfig`]. Every endpoint in
    /// the config becomes a leaf in the local-server subtree; the
    /// shard list collapses to one shard with the in-process node as
    /// leader and sole replica.
    pub fn from_storage(config: &StorageConfig) -> Self {
        let leaves: Vec<FailureDomain> =
            config.endpoints.iter().map(domain_from_endpoint).collect();
        let tree = TopologyTree { endpoints: leaves };
        let leader = NodeAddr::local();
        let shards = vec![ShardDescriptor {
            id: ShardId::ZERO,
            leader: leader.clone(),
            replicas: vec![leader],
        }];
        Self { tree, shards }
    }

    /// Build a topology from a pre-built `TopologyTree`. Test helper /
    /// escape hatch for unit tests that don't want to wire up a real
    /// `StorageConfig`.
    pub fn from_tree(tree: TopologyTree) -> Self {
        let leader = NodeAddr::local();
        let shards = vec![ShardDescriptor {
            id: ShardId::ZERO,
            leader: leader.clone(),
            replicas: vec![leader],
        }];
        Self { tree, shards }
    }
}

fn domain_from_endpoint(ep: &EndpointConfig) -> FailureDomain {
    let server = ep.server.clone().unwrap_or_else(|| "local".to_owned());
    FailureDomain {
        geo: "local".to_owned(),
        dc: "local".to_owned(),
        rack: "local".to_owned(),
        server,
        endpoint: ep.id.clone(),
        tags: ep.tags.clone(),
        tier: ep.tier,
    }
}

impl ClusterTopology for SingleNodeTopology {
    fn topology_tree(&self) -> &TopologyTree {
        &self.tree
    }

    fn shards(&self) -> &[ShardDescriptor] {
        &self.shards
    }

    fn shard_leader(&self, shard: ShardId) -> TopologyResult<NodeAddr> {
        self.shards
            .iter()
            .find(|s| s.id == shard)
            .map(|s| s.leader.clone())
            .ok_or(TopologyError::ShardNotFound(shard))
    }

    fn placement_candidates(
        &self,
        rule: &CrushRule,
        _modality: Modality,
        tier: Tier,
    ) -> TopologyResult<Vec<EndpointId>> {
        match rule {
            CrushRule::LocalTier => Ok(self
                .tree
                .endpoints
                .iter()
                .filter(|d| d.tier == tier)
                .map(|d| d.endpoint.clone())
                .collect()),
            // Multi-rack / multi-dc / multi-geo spread is an EE-only
            // feature; surface as EeOnly so the query layer can decide
            // whether to fall back to the local-tier rule or error.
            CrushRule::Spread { .. } => Err(TopologyError::EeOnly(
                "multi-domain spread requires EE CrushTopology",
            )),
        }
    }
}
