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
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::ShardId;
    /// assert_eq!(ShardId::ZERO.raw(), 0);
    /// assert_eq!(ShardId(42).raw(), 42);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::NodeAddr;
    /// let addr = NodeAddr::local();
    /// assert_eq!(addr.server, "local");
    /// assert_eq!(addr.address, "local");
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::Modality;
    /// // Every Layer 4 modality is represented.
    /// assert_eq!(Modality::all().len(), 9);
    /// assert!(Modality::all().contains(&Modality::Vector));
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::FailureDomain;
    /// use coordinode_storage::engine::config::Tier;
    ///
    /// let d = FailureDomain::local("ep-1", Tier::Hot);
    /// assert_eq!(d.geo, "local");
    /// assert_eq!(d.endpoint, "ep-1");
    /// assert_eq!(d.tier, Tier::Hot);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::TopologyTree;
    /// use coordinode_storage::engine::config::Tier;
    ///
    /// let tree = TopologyTree::single_endpoint("ep", Tier::Warm);
    /// assert_eq!(tree.endpoint_count(), 1);
    /// ```
    pub fn single_endpoint(endpoint: impl Into<EndpointId>, tier: Tier) -> Self {
        Self {
            endpoints: vec![FailureDomain::local(endpoint, tier)],
        }
    }

    /// Number of endpoints across the whole tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::TopologyTree;
    /// assert_eq!(TopologyTree::default().endpoint_count(), 0);
    /// ```
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    /// Unique server identifiers present in the tree, in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::{FailureDomain, TopologyTree};
    /// use coordinode_storage::engine::config::Tier;
    ///
    /// let mut a = FailureDomain::local("ep-a", Tier::Hot);
    /// a.server = "srv-1".into();
    /// let mut b = FailureDomain::local("ep-b", Tier::Warm);
    /// b.server = "srv-1".into();
    /// let mut c = FailureDomain::local("ep-c", Tier::Cold);
    /// c.server = "srv-2".into();
    /// let tree = TopologyTree { endpoints: vec![a, b, c] };
    /// assert_eq!(tree.servers(), vec!["srv-1".to_string(), "srv-2".to_string()]);
    /// ```
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

    /// Endpoints living in a specific data centre. Used by
    /// `SeqnoConsumerRegistry` (ADR-028) to expand a cascading
    /// retention scope like `dc:eu-west-1` to the concrete
    /// endpoint set.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::TopologyTree;
    /// // CE single-node tree: every endpoint is in dc "local".
    /// let tree = TopologyTree::default();
    /// let hits: Vec<_> = tree.endpoints_in_dc("eu-west-1").collect();
    /// assert!(hits.is_empty());
    /// ```
    pub fn endpoints_in_dc<'a>(
        &'a self,
        dc: &'a str,
    ) -> impl Iterator<Item = &'a FailureDomain> + 'a {
        self.endpoints.iter().filter(move |d| d.dc == dc)
    }

    /// Endpoints living in a specific rack. Same usage pattern as
    /// [`Self::endpoints_in_dc`].
    pub fn endpoints_in_rack<'a>(
        &'a self,
        rack: &'a str,
    ) -> impl Iterator<Item = &'a FailureDomain> + 'a {
        self.endpoints.iter().filter(move |d| d.rack == rack)
    }

    /// Endpoints living in a specific geo zone. Same usage pattern
    /// as [`Self::endpoints_in_dc`].
    pub fn endpoints_in_geo<'a>(
        &'a self,
        geo: &'a str,
    ) -> impl Iterator<Item = &'a FailureDomain> + 'a {
        self.endpoints.iter().filter(move |d| d.geo == geo)
    }

    /// Endpoints living on a specific server.
    pub fn endpoints_on_server<'a>(
        &'a self,
        server: &'a str,
    ) -> impl Iterator<Item = &'a FailureDomain> + 'a {
        self.endpoints.iter().filter(move |d| d.server == server)
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
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::CrushRule;
    /// assert_eq!(CrushRule::local_tier(), CrushRule::LocalTier);
    /// ```
    pub fn local_tier() -> Self {
        CrushRule::LocalTier
    }
}
