use super::*;

#[test]
fn empty_index() {
    let index = FlatIndex::new(VectorMetric::L2);
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert!(index.search(&[1.0, 0.0], 5).is_empty());
}

#[test]
fn insert_and_search_l2() {
    let mut index = FlatIndex::new(VectorMetric::L2);

    index.insert(1, vec![0.0, 0.0]);
    index.insert(2, vec![1.0, 0.0]);
    index.insert(3, vec![0.0, 1.0]);
    index.insert(4, vec![10.0, 10.0]);

    let results = index.search(&[0.1, 0.1], 3);
    assert_eq!(results.len(), 3);
    // Nearest to (0.1, 0.1) is (0, 0)
    assert_eq!(results[0].id, 1);
    // Next are (1,0) and (0,1) — equidistant
    let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
    assert!(ids.contains(&2));
    assert!(ids.contains(&3));
}

#[test]
fn exact_nn_100_percent_recall() {
    let mut index = FlatIndex::new(VectorMetric::L2);

    // Insert 50 points
    for i in 0..50u64 {
        index.insert(i, vec![i as f32, (i * 2) as f32]);
    }

    // Query at (10.5, 21.0) — nearest should be point 10 (10, 20) and 11 (11, 22)
    let results = index.search(&[10.5, 21.0], 2);
    assert_eq!(results.len(), 2);
    // Exact NN — 100% recall guaranteed
    assert!(results.iter().any(|r| r.id == 10));
    assert!(results.iter().any(|r| r.id == 11));
}

#[test]
fn cosine_metric() {
    let mut index = FlatIndex::new(VectorMetric::Cosine);

    index.insert(1, vec![1.0, 0.0]);
    index.insert(2, vec![0.0, 1.0]);
    index.insert(3, vec![-1.0, 0.0]);

    let results = index.search(&[1.0, 0.0], 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn dot_product_metric() {
    let mut index = FlatIndex::new(VectorMetric::DotProduct);

    index.insert(1, vec![1.0, 0.0]);
    index.insert(2, vec![0.5, 0.5]);
    index.insert(3, vec![0.0, 1.0]);

    // Highest dot product with (1,0) is (1,0) itself
    let results = index.search(&[1.0, 0.0], 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn manhattan_metric() {
    let mut index = FlatIndex::new(VectorMetric::L1);

    index.insert(1, vec![0.0, 0.0]);
    index.insert(2, vec![5.0, 5.0]);

    let results = index.search(&[0.1, 0.1], 1);
    assert_eq!(results[0].id, 1);
}

#[test]
fn k_greater_than_size() {
    let mut index = FlatIndex::new(VectorMetric::L2);
    index.insert(1, vec![0.0]);
    index.insert(2, vec![1.0]);

    let results = index.search(&[0.0], 10);
    assert_eq!(results.len(), 2);
}

#[test]
fn k_zero_returns_empty() {
    let mut index = FlatIndex::new(VectorMetric::L2);
    index.insert(1, vec![0.0]);

    assert!(index.search(&[0.0], 0).is_empty());
}

#[test]
fn duplicate_insert_ignored() {
    let mut index = FlatIndex::new(VectorMetric::L2);
    index.insert(1, vec![0.0]);
    index.insert(1, vec![99.0]); // Same ID — ignored
    assert_eq!(index.len(), 1);

    let results = index.search(&[0.0], 1);
    assert_eq!(results[0].score, 0.0); // Original vector kept
}

#[test]
fn remove_vector() {
    let mut index = FlatIndex::new(VectorMetric::L2);
    index.insert(1, vec![0.0]);
    index.insert(2, vec![1.0]);

    assert!(index.remove(1));
    assert_eq!(index.len(), 1);
    assert!(!index.remove(1)); // Already removed

    let results = index.search(&[0.0], 1);
    assert_eq!(results[0].id, 2);
}

#[test]
fn recommended_size_threshold() {
    let index = FlatIndex::new(VectorMetric::L2);
    assert!(index.is_recommended_size());
}

#[test]
fn high_dimensional() {
    let mut index = FlatIndex::new(VectorMetric::Cosine);

    for i in 0..20u64 {
        let vec: Vec<f32> = (0..384)
            .map(|d| ((i * d + 1) as f32 * 0.01).sin())
            .collect();
        index.insert(i, vec);
    }

    let query: Vec<f32> = (0..384).map(|d| (d as f32 * 0.01).sin()).collect();
    let results = index.search(&query, 5);
    assert_eq!(results.len(), 5);
}
