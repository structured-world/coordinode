//! Core identity and topology types used by [`crate::ClusterTopology`]
//! and [`crate::ShardRouting`].
//!
//! These are intentionally minimal — the 5-level failure-domain tree,
//! a shard descriptor, a node address, and the modality / placement-rule
//! enums the upper layers need to reason about placement. Full CRUSH
//! rule parsing and validation lives in the EE `CrushTopology` impl
//! (Phase 3) — CE only needs the trivial `srv-only` shape.

use coordinode_storage::engine::config::Tier;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Stable identifier for a storage endpoint. Mirrors
/// [`coordinode_storage::engine::config::EndpointConfig::id`] —
/// keeping the type alias here so cluster code does not depend on the
/// storage crate's string newtype conventions.
pub type EndpointId = String;

/// Stable identifier for a physical / virtual server in the cluster.
/// Matches [`coordinode_storage::engine::config::EndpointConfig::server`]
/// when populated. `"local"` is the conventional name for the implicit
/// single-node server in CE deployments.
pub type ServerId = String;

/// Numeric shard identifier — assigned monotonically by the topology
/// layer. CE single-shard deployments expose exactly one shard with
/// id `0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShardId(pub u32);

impl ShardId {
    /// Shard 0 — the only shard in CE single-shard deployments.
    pub const ZERO: Self = Self(0);

    /// Raw u32 accessor.
    pub fn raw(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shard-{}", self.0)
    }
}

/// Network address of a cluster node — the Raft / gRPC endpoint the
/// query layer talks to for shard-routed reads/writes.
///
/// `"local"` is the canonical address for the in-process CE single-node
/// deployment; any query routed to a `"local"` `NodeAddr` short-circuits
/// to the local engine.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeAddr {
    /// Server identifier (matches `ServerId`).
    pub server: ServerId,
    /// `host:port` for the gRPC / Raft endpoint. `"local"` for the
    /// in-process server. Free-form string so single-node deployments
    /// don't need to invent a port.
    pub address: String,
}

impl NodeAddr {
    /// The canonical single-node CE address — in-process server.
    pub fn local() -> Self {
        Self {
            server: "local".to_owned(),
            address: "local".to_owned(),
        }
    }
}

/// Per-shard metadata: id + the node currently authoritative for it
/// (Raft leader in EE, the local node in CE) + the set of replica
/// nodes (CE: just the leader; EE: replicas across the failure-domain
/// tree per active CRUSH rule).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardDescriptor {
    /// Shard id.
    pub id: ShardId,
    /// Current leader / authoritative node for this shard.
    pub leader: NodeAddr,
    /// Replica nodes (includes leader). At RF=1 / CE single-node the
    /// vec has one entry equal to `leader`.
    pub replicas: Vec<NodeAddr>,
}

/// One of the per-modality storage surfaces — used by
/// [`crate::ClusterTopology::placement_candidates`] to decide which
/// endpoints are eligible to host data of this kind. Matches the
/// Layer 4 modality store inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Graph node bodies.
    Node,
    /// Edge adjacency posting lists + edge property bodies.
    Edge,
    /// Vector indexes (HNSW).
    Vector,
    /// Document partial-update merges (ADR-015) — lives on the Node
    /// partition but uses dedicated merge operators.
    Document,
    /// Time-series buckets + overflow segments.
    TimeSeries,
    /// Spatial point index entries (Layer 4 SpatialStore).
    Spatial,
    /// Secondary B-tree / compound / full-text index entries.
    Index,
    /// Content-addressed binary chunks + per-(node, prop) blob refs.
    Blob,
    /// Schema DDL state (labels, edge types, migrations).
    Schema,
}

impl Modality {
    /// Iterate every modality variant — useful for stores that index
    /// per-modality state on bootstrap.
    pub fn all() -> &'static [Modality] {
        &[
            Modality::Node,
            Modality::Edge,
            Modality::Vector,
            Modality::Document,
            Modality::TimeSeries,
            Modality::Spatial,
            Modality::Index,
            Modality::Blob,
            Modality::Schema,
        ]
    }
}

/// One node of the 5-level failure-domain tree
/// (`geo → dc → rack → server → endpoint`). The CE
/// [`crate::SingleNodeTopology`] degenerates the upper four levels
/// into `"local"` strings so the tree is shaped but trivial.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureDomain {
    /// Continent / region tag (free-form). `"local"` in CE single-node.
    pub geo: String,
    /// Data centre identifier.
    pub dc: String,
    /// Rack identifier within the data centre.
    pub rack: String,
    /// Server identifier within the rack.
    pub server: ServerId,
    /// Endpoint identifier within the server.
    pub endpoint: EndpointId,
    /// Free-form operator tags (mirrors
    /// [`coordinode_storage::engine::config::EndpointConfig::tags`]).
    /// Consumed by CRUSH rules in the EE impl.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tags: BTreeMap<String, String>,
    /// Storage tier the endpoint is configured for. Used by
    /// [`crate::ClusterTopology::placement_candidates`] for tier
    /// matching even in CE.
    pub tier: Tier,
}

impl FailureDomain {
    /// Build a single-server domain for CE single-node deployments.
    /// All upper levels collapse to `"local"`; only the endpoint and
    /// tier carry information.
    pub fn local(endpoint: impl Into<EndpointId>, tier: Tier) -> Self {
        Self {
            geo: "local".to_owned(),
            dc: "local".to_owned(),
            rack: "local".to_owned(),
            server: "local".to_owned(),
            endpoint: endpoint.into(),
            tags: BTreeMap::new(),
            tier,
        }
    }
}

/// Full topology tree — currently flat (every endpoint hangs directly
/// off the root). The 5-level hierarchy is encoded **per leaf** in
/// [`FailureDomain`]; the tree wrapper is `Vec<FailureDomain>` so the
/// CE impl is trivial and the EE impl can swap in a hierarchical
/// representation without breaking the trait surface.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TopologyTree {
    /// Every endpoint in the cluster, each carrying its 5-level path.
    pub endpoints: Vec<FailureDomain>,
}

impl TopologyTree {
    /// Build a trivial single-endpoint tree — used by tests and by
    /// the most-minimal CE deployment shape.
    pub fn single_endpoint(endpoint: impl Into<EndpointId>, tier: Tier) -> Self {
        Self {
            endpoints: vec![FailureDomain::local(endpoint, tier)],
        }
    }

    /// Number of endpoints across the whole tree.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    /// Unique server identifiers present in the tree.
    pub fn servers(&self) -> Vec<ServerId> {
        let mut out: Vec<_> = self
            .endpoints
            .iter()
            .map(|d| d.server.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        out.sort();
        out
    }
}

/// Placement rule consulted by
/// [`crate::ClusterTopology::placement_candidates`]. CE only uses the
/// trivial `local` rule (any matching endpoint on the local server);
/// EE will parse the full YAML rule grammar from `crush.md` into
/// richer variants in Phase 3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CrushRule {
    /// Match any endpoint on the local server with the given tier.
    /// The only rule CE ever returns.
    LocalTier,
    /// Reserved for EE: spread N copies across distinct failure
    /// domains at the named level. CE [`crate::SingleNodeTopology`]
    /// rejects this variant with [`crate::TopologyError::EeOnly`].
    Spread {
        /// Failure-domain level: `"rack"` / `"dc"` / `"geo"`.
        level: String,
        /// Required minimum number of distinct domains at that level.
        min: u32,
    },
}

impl CrushRule {
    /// The default CE rule — any endpoint on the local server at the
    /// requested tier.
    pub fn local_tier() -> Self {
        CrushRule::LocalTier
    }
}
