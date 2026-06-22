//! [`VectorShardRouter`] — index-internal vector partition routing.
//!
//! This is a **distinct axis** from [`super::routing::ShardRouting`] and the
//! two must never be conflated:
//!
//! - [`ShardRouting`](super::routing::ShardRouting) maps a routing key to a
//!   **cross-node** [`ShardId`](crate::ShardId): which node in the cluster
//!   holds the data. Coarse (roughly one shard per millions of vectors).
//! - [`VectorShardRouter`] maps a vector to **index-internal** similarity
//!   partitions: which local HNSW sub-index, within one label's vector index
//!   on a single node, serves a vector. A [`PartitionId`] here is a local
//!   `u32` index into a label's shard-HNSW vector, **not** a cluster
//!   [`ShardId`]. This is index acceleration, not data placement.
//!
//! The CE default is [`SinglePartitionRouter`] (one partition, the Unsharded
//! path, bit-identical to a non-partitioned index). The EE centroid router
//! (closure replication + adaptive fan-out, meta-HNSW routing over centroids)
//! implements this trait in `coordinode-ee-sharding`; the CE vector-index
//! registry holds the trait object and the EE build swaps the smart router in
//! without the registry depending on EE.

use smallvec::{smallvec, SmallVec};

/// Index-internal vector partition id: a local `u32` index into a label's
/// shard-HNSW vector (`0..n_partitions`). **Not** a cross-node cluster
/// [`crate::ShardId`] — see the module docs for the axis distinction.
pub type PartitionId = u32;

/// A small set of [`PartitionId`]s, inline up to 8 (the typical query fan-out
/// cap), so the per-query routing call and the per-vector build assignment do
/// not allocate on the common low-fan-out path.
pub type PartitionSet = SmallVec<[PartitionId; 8]>;

/// Routes a vector to the index-internal similarity partitions that serve it.
///
/// Build assignment ([`assign`](VectorShardRouter::assign)) may place a
/// boundary vector in several partitions (closure replication); query routing
/// ([`route`](VectorShardRouter::route)) fans an interior query out to one
/// partition and a boundary query to a few, capped at `top_m`. Implementations
/// are read-only on the hot path and `Send + Sync` so a single instance is
/// shared across query workers behind an `Arc`.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot route vectors to index partitions",
    label = "this type does not implement `VectorShardRouter`",
    note = "the CE default is `coordinode_cluster::SinglePartitionRouter`; the EE centroid router lives in `coordinode-ee-sharding`"
)]
pub trait VectorShardRouter: Send + Sync {
    /// Build-time assignment: the partitions a vector is written to. Closure
    /// replication may return several (a boundary vector lands in every
    /// partition within a replication radius of its nearest). The CE default
    /// returns the single partition `0`.
    fn assign(&self, vector: &[f32]) -> PartitionSet;

    /// Query-time routing: the partitions a query scatters to, capped at
    /// `top_m`. An interior query touches one partition; a boundary query
    /// fans out to a few. The CE default returns the single partition `0`
    /// regardless of `top_m`.
    fn route(&self, query: &[f32], top_m: usize) -> PartitionSet;

    /// Number of partitions this router spans. The CE default is `1`.
    fn n_partitions(&self) -> usize;
}

/// CE default router: a single partition. [`assign`](VectorShardRouter::assign)
/// and [`route`](VectorShardRouter::route) always return partition `0`, so the
/// vector-index registry uses one HNSW handle per label — the Unsharded path,
/// bit-identical to a non-partitioned index. Carries no state.
///
/// # Examples
///
/// ```
/// use coordinode_cluster::{SinglePartitionRouter, VectorShardRouter};
///
/// let r = SinglePartitionRouter::new();
/// assert_eq!(r.n_partitions(), 1);
/// assert_eq!(&r.assign(&[0.1, 0.2, 0.3])[..], &[0]);
/// assert_eq!(&r.route(&[0.1, 0.2, 0.3], 4)[..], &[0]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SinglePartitionRouter;

impl SinglePartitionRouter {
    /// Build a fresh single-partition router. Cost is zero — the type carries
    /// no state.
    pub fn new() -> Self {
        Self
    }
}

impl VectorShardRouter for SinglePartitionRouter {
    fn assign(&self, _vector: &[f32]) -> PartitionSet {
        smallvec![0]
    }

    fn route(&self, _query: &[f32], _top_m: usize) -> PartitionSet {
        smallvec![0]
    }

    fn n_partitions(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_partition_assigns_to_zero() {
        let r = SinglePartitionRouter::new();
        // Any vector lands in the one partition.
        assert_eq!(&r.assign(&[1.0, 2.0, 3.0])[..], &[0]);
        assert_eq!(&r.assign(&[])[..], &[0]);
    }

    #[test]
    fn single_partition_routes_to_zero_regardless_of_top_m() {
        let r = SinglePartitionRouter::new();
        // top_m never widens the single-partition fan-out.
        assert_eq!(&r.route(&[0.5, 0.5], 1)[..], &[0]);
        assert_eq!(&r.route(&[0.5, 0.5], 64)[..], &[0]);
    }

    #[test]
    fn single_partition_count_is_one() {
        assert_eq!(SinglePartitionRouter::new().n_partitions(), 1);
    }

    #[test]
    fn router_is_object_safe_and_shareable() {
        // The registry holds the router as a trait object behind an Arc; this
        // pins object-safety (the EE adapter plugs into the same slot).
        use std::sync::Arc;
        let r: Arc<dyn VectorShardRouter> = Arc::new(SinglePartitionRouter::new());
        assert_eq!(r.n_partitions(), 1);
        assert_eq!(&r.route(&[0.1], 8)[..], &[0]);
    }
}
