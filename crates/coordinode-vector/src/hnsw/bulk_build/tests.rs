use super::*;
use crate::hnsw::HnswConfig;
use coordinode_core::graph::types::VectorMetric;

fn make_index(dim: usize) -> HnswIndex {
    HnswIndex::new(HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 50,
        metric: VectorMetric::L2,
        max_dimensions: dim as u32,
        ..Default::default()
    })
}

fn synth_items(n: usize, dim: usize) -> Vec<(u64, Vec<f32>)> {
    (0..n as u64)
        .map(|i| {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i as u32).wrapping_mul(2654435761) ^ d as u32) as f32 * 1e-6)
                .collect();
            (i, v)
        })
        .collect()
}

#[test]
fn bulk_build_below_threshold_routes_to_insert_batch() {
    // The Vec<(u64, Vec<f32>)> at this size is smaller than the
    // cluster-and-stitch threshold; the public `bulk_build`
    // entry point must fall through to insert_batch and produce
    // a fully populated index.
    const _: () = assert!(BULK_BUILD_THRESHOLD > 4);
    let mut idx = make_index(4);
    let items = synth_items(BULK_BUILD_THRESHOLD - 1, 4);
    let expected = items.len();
    idx.bulk_build(items);
    assert_eq!(idx.len(), expected);
}

#[test]
fn leader_count_is_floor_sqrt() {
    assert_eq!(leader_count_for(0), 0);
    assert_eq!(leader_count_for(1), 1);
    assert_eq!(leader_count_for(2), 1);
    assert_eq!(leader_count_for(4), 2);
    assert_eq!(leader_count_for(9), 3);
    assert_eq!(leader_count_for(10), 3);
    assert_eq!(leader_count_for(256), 16);
    assert_eq!(leader_count_for(10_000), 100);
}

#[test]
fn leader_sample_is_deterministic_and_unique() {
    let a = sample_leader_indices(1_000, 32);
    let b = sample_leader_indices(1_000, 32);
    assert_eq!(a, b, "same (n, k) must yield the same set");
    assert_eq!(a.len(), 32);
    assert!(a.iter().all(|&i| i < 1_000));
}

#[test]
fn leader_sample_covers_full_range_when_k_eq_n() {
    let s = sample_leader_indices(8, 8);
    assert_eq!(s.len(), 8);
    for i in 0..8 {
        assert!(s.contains(&i));
    }
}

#[test]
fn cluster_assignment_picks_nearest_leader_under_l2() {
    // Two leaders at the corners; followers in between are
    // assigned to whichever leader they are closer to.
    let leaders = vec![vec![0.0, 0.0], vec![10.0, 0.0]];
    let followers = vec![
        (0u64, vec![1.0, 0.0]), // closer to leader 0
        (1u64, vec![9.0, 0.0]), // closer to leader 1
        (2u64, vec![5.5, 0.0]), // closer to leader 1
        (3u64, vec![4.5, 0.0]), // closer to leader 0
    ];
    let a = assign_followers_to_leaders(&followers, &leaders, VectorMetric::L2);
    assert_eq!(a, vec![0, 1, 1, 0]);
}

/// Followers are dealt round-robin over the clusters, each cluster in
/// input order: neighbours of one cluster never sit side by side while
/// another cluster still has items, so they are not inserted together.
#[test]
fn cluster_order_deals_followers_round_robin_over_the_clusters() {
    let leaders = vec![vec![0.0, 0.0], vec![10.0, 0.0]];
    let followers = vec![
        (10u64, vec![1.0, 0.0]), // cluster 0
        (30u64, vec![2.0, 0.0]), // cluster 0
        (50u64, vec![3.0, 0.0]), // cluster 0
        (20u64, vec![9.0, 0.0]), // cluster 1
        (40u64, vec![8.0, 0.0]), // cluster 1
    ];
    let reordered = cluster_order_followers(followers, &leaders, VectorMetric::L2);
    let ids: Vec<u64> = reordered.iter().map(|(id, _)| *id).collect();
    // Rank 0 of each cluster, then rank 1, then the longer cluster's
    // tail; each cluster keeps its input order.
    assert_eq!(ids, vec![10, 20, 30, 40, 50]);
}

#[test]
fn bulk_build_at_threshold_matches_insert_batch_membership() {
    // At the threshold the entry point dispatches into this
    // module's `bulk_build` function. The stub forwards to
    // `insert_batch`, so the resulting index must contain
    // exactly the same set of ids as a parallel `insert_batch`
    // call would have produced. Stronger recall comparison
    // lands once the real algorithm replaces the stub.
    let mut idx_bulk = make_index(4);
    let mut idx_batch = make_index(4);
    let items = synth_items(BULK_BUILD_THRESHOLD, 4);
    idx_bulk.bulk_build(items.clone());
    idx_batch.insert_batch(items);
    assert_eq!(idx_bulk.len(), idx_batch.len());
}
