//! Per-label shard-routing strategy: centroid partitioning with SPANN-style
//! closure replication and query-adaptive fan-out.
//!
//! Engine-side geometry for the routing study validated in the vector-ann
//! bench. `coordinode-modality` owns the persisted `ShardStrategy` config (with
//! its permille thresholds); this crate owns the geometry. Because
//! `coordinode-modality` depends on `coordinode-vector`, the router takes raw
//! `f32` thresholds: callers scale the permille config (`permille / 1000.0`) at
//! the boundary.
//!
//! Two operations, one rule (nearest plus any centroid within a distance
//! ratio, capped): build-time [`ShardStrategyRouter::assign`] (closure
//! replication, so boundary vectors land on several shards) and query-time
//! [`ShardStrategyRouter::route`] (adaptive fan-out, so a query only touches
//! more shards when it sits near a cluster boundary).

use coordinode_core::graph::types::VectorMetric;
use smallvec::SmallVec;

/// Shards a single vector / query touches; inline up to 8 before spilling.
pub type ShardVec = SmallVec<[u32; 8]>;

/// Routing distance: cosine distance for `Cosine`, squared L2 otherwise
/// (monotonic in L2, cheaper). Only compares a point against centroids, so
/// rank order is all that matters.
#[inline]
fn routing_distance(a: &[f32], b: &[f32], metric: VectorMetric) -> f32 {
    match metric {
        VectorMetric::Cosine => crate::metrics::cosine_distance(a, b),
        _ => crate::metrics::euclidean_distance_squared(a, b),
    }
}

/// Index of the centroid closest to `v` under `metric`.
///
/// # Panics
/// Panics if `centroids` is empty.
#[must_use]
pub fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>], metric: VectorMetric) -> u32 {
    assert!(!centroids.is_empty(), "nearest_centroid: no centroids");
    let mut best = 0u32;
    let mut best_d = f32::INFINITY;
    for (j, c) in centroids.iter().enumerate() {
        let d = routing_distance(v, c, metric);
        if d < best_d {
            best_d = d;
            best = j as u32;
        }
    }
    best
}

/// Deterministic k-means (Lloyd) over an evenly-strided subsample of `samples`.
/// No RNG: init is evenly-spaced sample points, then a fixed number of Lloyd
/// iterations (ample for the small k used for shard counts). Empty clusters
/// keep their previous centre. Deterministic for a given `samples`/`k`/`metric`.
///
/// # Panics
/// Panics if `k == 0` or `samples` is empty.
#[must_use]
pub fn train_centroids(samples: &[&[f32]], k: usize, metric: VectorMetric) -> Vec<Vec<f32>> {
    assert!(k > 0, "train_centroids: k must be > 0");
    assert!(
        !samples.is_empty(),
        "train_centroids: samples must be non-empty"
    );
    const SAMPLE_CAP: usize = 20_000;
    const LLOYD_ITERS: usize = 10;
    let n = samples.len();
    let d = samples[0].len();
    let sample_n = n.min(SAMPLE_CAP);
    let stride = (n / sample_n).max(1);
    let sample: Vec<&[f32]> = (0..n)
        .step_by(stride)
        .take(sample_n)
        .map(|i| samples[i])
        .collect();
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|j| sample[j * sample.len() / k].to_vec())
        .collect();
    for _ in 0..LLOYD_ITERS {
        let mut sums = vec![vec![0f32; d]; k];
        let mut counts = vec![0usize; k];
        for v in &sample {
            let cl = nearest_centroid(v, &centroids, metric) as usize;
            counts[cl] += 1;
            for (s, x) in sums[cl].iter_mut().zip(v.iter()) {
                *s += x;
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                for (cv, s) in centroids[j].iter_mut().zip(sums[j].iter()) {
                    *cv = s / counts[j] as f32;
                }
            }
        }
    }
    centroids
}

/// Per-label centroid shard router: closure-replicated build assignment and
/// query-adaptive fan-out routing. Built once per label from its
/// `ShardStrategy::Centroid` config (thresholds scaled from permille to f32 by
/// the caller) and the label's trained centroids. Immutable + cheap to clone
/// (`Arc`-share it across query workers).
#[derive(Debug, Clone)]
pub struct ShardStrategyRouter {
    centroids: Vec<Vec<f32>>,
    metric: VectorMetric,
    replication_eps: f32,
    max_replicas: usize,
    route_eps: f32,
}

impl ShardStrategyRouter {
    /// Build a router from explicit centroids + thresholds.
    #[must_use]
    pub fn new(
        centroids: Vec<Vec<f32>>,
        metric: VectorMetric,
        replication_eps: f32,
        max_replicas: usize,
        route_eps: f32,
    ) -> Self {
        Self {
            centroids,
            metric,
            replication_eps,
            max_replicas,
            route_eps,
        }
    }

    /// Train centroids from `samples` and build a router. `n_shards` is the
    /// centroid count.
    #[must_use]
    pub fn train(
        samples: &[&[f32]],
        n_shards: usize,
        metric: VectorMetric,
        replication_eps: f32,
        max_replicas: usize,
        route_eps: f32,
    ) -> Self {
        let centroids = train_centroids(samples, n_shards.max(1), metric);
        Self::new(centroids, metric, replication_eps, max_replicas, route_eps)
    }

    /// Number of shards (centroids).
    #[must_use]
    pub fn n_shards(&self) -> usize {
        self.centroids.len()
    }

    /// The trained centroids (one per shard).
    #[must_use]
    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    /// Shared rule: the nearest shard plus every centroid within `eps` x the
    /// nearest distance, capped at `cap` distinct shards, best-first. Always
    /// returns at least the nearest shard.
    fn nearest_within(&self, v: &[f32], eps: f32, cap: usize) -> ShardVec {
        let mut scored: Vec<(u32, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(j, c)| (j as u32, routing_distance(v, c, self.metric)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let thresh = scored[0].1 * eps.max(1.0);
        let cap = cap.max(1);
        let mut out: ShardVec = SmallVec::new();
        for (j, dist) in scored {
            if out.is_empty() {
                out.push(j); // always keep the nearest shard
            } else if out.len() < cap && dist <= thresh {
                out.push(j);
            } else {
                break;
            }
        }
        out
    }

    /// Build-time shard assignment for `v`: SPANN closure replication. The
    /// nearest centroid plus every centroid within `replication_eps` x the
    /// nearest distance, capped at `max_replicas`. Boundary vectors land on
    /// several shards so a low query fan-out still recovers them.
    #[must_use]
    pub fn assign(&self, v: &[f32]) -> ShardVec {
        self.nearest_within(v, self.replication_eps, self.max_replicas)
    }

    /// Query-time shard routing for `query`: adaptive fan-out. The nearest
    /// centroid plus every centroid within `route_eps` x the nearest distance,
    /// capped at `top_m`. Interior queries touch one shard; only queries near a
    /// boundary fan out.
    #[must_use]
    pub fn route(&self, query: &[f32], top_m: usize) -> ShardVec {
        self.nearest_within(query, self.route_eps, top_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(v: &[f32]) -> Vec<f32> {
        v.to_vec()
    }

    #[test]
    fn nearest_centroid_picks_closest() {
        let centroids = vec![c(&[0.0, 0.0]), c(&[10.0, 0.0])];
        assert_eq!(
            nearest_centroid(&[1.0, 0.0], &centroids, VectorMetric::L2),
            0
        );
        assert_eq!(
            nearest_centroid(&[9.0, 0.0], &centroids, VectorMetric::L2),
            1
        );
    }

    #[test]
    fn closure_assign_replicates_boundary_keeps_interior_single() {
        // Centroids at x=0 and x=10. A point at x=5 is equidistant (boundary);
        // a point at x=1 is firmly inside cluster 0.
        let router = ShardStrategyRouter::new(
            vec![c(&[0.0, 0.0]), c(&[10.0, 0.0])],
            VectorMetric::L2,
            1.2, // replication_eps
            4,   // max_replicas
            1.0, // route_eps (unused here)
        );
        let boundary = router.assign(&[5.0, 0.0]);
        assert_eq!(
            boundary.len(),
            2,
            "boundary vector replicates to both shards"
        );
        let interior = router.assign(&[1.0, 0.0]);
        assert_eq!(interior.len(), 1, "interior vector lands on one shard");
        assert_eq!(interior[0], 0);
    }

    #[test]
    fn closure_assign_respects_max_replicas_and_no_replication_at_eps_one() {
        let centroids = vec![c(&[0.0, 0.0]), c(&[10.0, 0.0]), c(&[5.0, 8.0])];
        // eps=1.0 => only exact ties replicate; max_replicas caps the fan.
        let router = ShardStrategyRouter::new(centroids.clone(), VectorMetric::L2, 1.0, 1, 1.0);
        // x=1 is closest to centroid 0 with no tie -> single shard.
        assert_eq!(router.assign(&[1.0, 0.0]).len(), 1);
        // Wide eps but max_replicas=1 still caps to one.
        let capped = ShardStrategyRouter::new(centroids, VectorMetric::L2, 100.0, 1, 1.0);
        assert_eq!(capped.assign(&[5.0, 0.0]).len(), 1);
    }

    #[test]
    fn route_fans_out_only_near_boundary() {
        let router = ShardStrategyRouter::new(
            vec![c(&[0.0, 0.0]), c(&[10.0, 0.0])],
            VectorMetric::L2,
            1.0, // replication_eps (unused here)
            1,   // max_replicas (unused here)
            1.2, // route_eps
        );
        // Boundary query fans to both shards (within top_m).
        assert_eq!(router.route(&[5.0, 0.0], 2).len(), 2);
        // Interior query touches one shard.
        let interior = router.route(&[1.0, 0.0], 2);
        assert_eq!(interior.len(), 1);
        assert_eq!(interior[0], 0);
        // top_m caps the fan-out even at the boundary.
        assert_eq!(router.route(&[5.0, 0.0], 1).len(), 1);
    }

    #[test]
    fn train_centroids_is_deterministic() {
        // Two clusters: around (0,0) and (10,10).
        let pts: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                if i % 2 == 0 {
                    vec![(i % 4) as f32 * 0.1, (i % 3) as f32 * 0.1]
                } else {
                    vec![10.0 + (i % 4) as f32 * 0.1, 10.0 + (i % 3) as f32 * 0.1]
                }
            })
            .collect();
        let refs: Vec<&[f32]> = pts.iter().map(Vec::as_slice).collect();
        let a = train_centroids(&refs, 2, VectorMetric::L2);
        let b = train_centroids(&refs, 2, VectorMetric::L2);
        assert_eq!(a, b, "deterministic: same input -> same centroids");
        assert_eq!(a.len(), 2);
        // Centroids separate the two clusters: one near origin, one near (10,10).
        let mut near_origin = false;
        let mut near_ten = false;
        for centroid in &a {
            if centroid[0] < 5.0 {
                near_origin = true;
            } else {
                near_ten = true;
            }
        }
        assert!(near_origin && near_ten, "centroids cover both clusters");
    }

    #[test]
    fn router_train_round_trip() {
        let pts: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                if i < 20 {
                    vec![0.0, i as f32 * 0.01]
                } else {
                    vec![20.0, i as f32 * 0.01]
                }
            })
            .collect();
        let refs: Vec<&[f32]> = pts.iter().map(Vec::as_slice).collect();
        let router = ShardStrategyRouter::train(&refs, 2, VectorMetric::L2, 1.3, 3, 1.3);
        assert_eq!(router.n_shards(), 2);
        // A point in the first cluster routes to a single shard.
        assert_eq!(router.route(&[0.0, 0.05], 2).len(), 1);
    }
}
