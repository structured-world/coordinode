//! [`ShardRouting`] trait + CE [`SingleShardRouting`] impl.
//!
//! Layer 5 (query engine) consults the routing trait to decide which
//! shards a query touches. With one shard the answer is always
//! [`ShardId::FIRST`]; multi-shard impls answer against the same trait.

use crate::error::TopologyResult;
use crate::types::{NodeAddr, ShardId};

/// Layer 6 routing contract. Single-shard and multi-shard
/// implementations share this trait.
///
/// All methods take a `&self` so a routing instance is cheaply
/// clonable / `Arc`-shareable across query-engine workers — routing
/// state is read-only on the hot path; admin updates rebuild the
/// routing snapshot atomically.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not route keys to shards",
    label = "not a ShardRouting",
    note = "the single-shard implementation is `coordinode_cluster::SingleShardRouting`"
)]
pub trait ShardRouting: Send + Sync {
    /// Hash a routing key to its shard id. The CE single-shard impl
    /// always returns [`ShardId::FIRST`].
    ///
    /// `key` is the raw routing key bytes (typically the encoded
    /// node id or the shard partition key for a query). The hash
    /// function MUST be stable across cluster restarts so consistent
    /// routing decisions hold.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::{ShardId, ShardRouting, SingleShardRouting};
    /// let r = SingleShardRouting::new();
    /// assert_eq!(r.shard_for_key(b"any-key"), ShardId::FIRST);
    /// ```
    fn shard_for_key(&self, key: &[u8]) -> ShardId;

    /// The full set of shard ids served by this cluster. CE returns
    /// `[ShardId::FIRST]`; EE returns the live shard map.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::{ShardId, ShardRouting, SingleShardRouting};
    /// let r = SingleShardRouting::new();
    /// assert_eq!(r.shard_ids(), vec![ShardId::FIRST]);
    /// ```
    fn shard_ids(&self) -> Vec<ShardId>;

    /// Resolve a shard id to its node address. Mirrors
    /// [`crate::ClusterTopology::shard_leader`] but exists on the
    /// routing trait so the query layer can resolve without holding
    /// a topology reference when topology is empty (degenerate
    /// startup case).
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_cluster::{NodeAddr, ShardId, ShardRouting, SingleShardRouting};
    /// let r = SingleShardRouting::new();
    /// assert_eq!(r.resolve(ShardId::FIRST)?, NodeAddr::local());
    /// # Ok::<_, coordinode_cluster::TopologyError>(())
    /// ```
    fn resolve(&self, shard: ShardId) -> TopologyResult<NodeAddr>;
}

/// CE single-shard routing: every key lands at [`ShardId::FIRST`] on
/// the local node. The trait surface is here so query code is
/// generic over routing from day one.
///
/// # Examples
///
/// ```
/// use coordinode_cluster::{ShardRouting, SingleShardRouting, ShardId};
///
/// let routing = SingleShardRouting::new();
/// assert_eq!(routing.shard_for_key(b"any-key"), ShardId::FIRST);
/// assert_eq!(routing.shard_ids(), vec![ShardId::FIRST]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SingleShardRouting;

impl SingleShardRouting {
    /// Build a fresh routing instance. Cost is zero — the type
    /// carries no state.
    pub fn new() -> Self {
        Self
    }
}

impl ShardRouting for SingleShardRouting {
    fn shard_for_key(&self, _key: &[u8]) -> ShardId {
        ShardId::FIRST
    }

    fn shard_ids(&self) -> Vec<ShardId> {
        vec![ShardId::FIRST]
    }

    fn resolve(&self, shard: ShardId) -> TopologyResult<NodeAddr> {
        if shard == ShardId::FIRST {
            Ok(NodeAddr::local())
        } else {
            Err(crate::error::TopologyError::ShardNotFound(shard))
        }
    }
}
