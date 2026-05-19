//! Topology error type. Distinguishes "EE-only feature in CE binary"
//! from real "shard not found" lookups so the gRPC layer can surface
//! `UNIMPLEMENTED` vs `NOT_FOUND` correctly.

use crate::types::ShardId;
use thiserror::Error;

/// Errors returned by [`crate::ClusterTopology`] and
/// [`crate::ShardRouting`] methods.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// Caller asked for a shard that does not exist in the cluster.
    /// In CE single-shard deployments this only happens when the
    /// caller passed an id other than [`ShardId::ZERO`].
    #[error("shard {0} not in topology")]
    ShardNotFound(ShardId),

    /// Caller invoked an EE-only feature on a CE impl
    /// (e.g. multi-rack `CrushRule::Spread` against
    /// [`crate::SingleNodeTopology`]).
    #[error("feature is EE-only and not available in this build: {0}")]
    EeOnly(&'static str),
}

/// Result alias for topology operations.
pub type TopologyResult<T> = Result<T, TopologyError>;
