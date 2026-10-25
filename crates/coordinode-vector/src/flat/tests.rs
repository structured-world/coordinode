use super::*;

#[test]
fn empty_index() {
    let index = FlatIndex::new(VectorMetric::L2);
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert!(index.search(&[1.0, 0.0], 5).unwrap().is_empty());
}

#[test]
fn insert_and_search_l2() {
    let mut index = FlatIndex::new(VectorMetric::L2);

    index.insert(1, vec![0.0, 0.0]).unwrap();
    index.insert(2, vec![1.0, 0.0]).unwrap();
    index.insert(3, vec![0.0, 1.0]).unwrap();
    index.insert(4, vec![10.0, 10.0]).unwrap();

    let results = index.search(&[0.1, 0.1], 3).unwrap();
    assert_eq!(results.len(), 3);
    // Nearest to (0.1, 0.1) is (0, 0)
    assert_eq!(results[0].id, 1);
    // Next are (1,0) and (0,1): equidistant, so by id.
    assert_eq!(results[1].id, 2);
    assert_eq!(results[2].id, 3);
    // The L2 score is the distance, not its square.
    let want = (f64::from(0.1f32).powi(2) * 2.0).sqrt();
    assert_eq!(results[0].score, want);
}

#[test]
fn exact_nn_100_percent_recall() {
    let mut index = FlatIndex::new(VectorMetric::L2);

    // Insert 50 points
    for i in 0..50u64 {
        index.insert(i, vec![i as f32, (i * 2) as f32]).unwrap();
    }

    // Query at (10.5, 21.0) — nearest should be point 10 (10, 20) and 11 (11, 22)
    let results = index.search(&[10.5, 21.0], 2).unwrap();
    assert_eq!(results.len(), 2);
    // Exact NN — 100% recall guaranteed
    assert!(results.iter().any(|r| r.id == 10));
    assert!(results.iter().any(|r| r.id == 11));
}

#[test]
fn cosine_metric() {
    let mut index = FlatIndex::new(VectorMetric::Cosine);

    index.insert(1, vec![1.0, 0.0]).unwrap();
    index.insert(2, vec![0.0, 1.0]).unwrap();
    index.insert(3, vec![-1.0, 0.0]).unwrap();

    let results = index.search(&[1.0, 0.0], 3).unwrap();
    let ranked: Vec<(u64, f64)> = results.iter().map(|r| (r.id, r.score)).collect();
    assert_eq!(ranked, [(1, 0.0), (2, 1.0), (3, 2.0)]);
}

/// Large finite vectors rank by direction: the one along the query comes
/// first with distance 0, not after the others with a NaN distance.
#[test]
fn cosine_ranks_large_finite_vectors_by_direction() {
    let mut index = FlatIndex::new(VectorMetric::Cosine);
    index.insert(1, vec![1.0, 0.0]).unwrap();
    index.insert(2, vec![1e20, 1e20]).unwrap();
    index.insert(3, vec![-1e20, 1e20]).unwrap();

    let results = index.search(&[3e20, 3e20], 3).unwrap();
    assert_eq!(results[0].id, 2);
    assert!(results[0].score.abs() < 1e-15, "{}", results[0].score);
    assert!(results.iter().all(|r| r.score.is_finite()));
}

#[test]
fn dot_product_metric() {
    let mut index = FlatIndex::new(VectorMetric::DotProduct);

    index.insert(1, vec![1.0, 0.0]).unwrap();
    index.insert(2, vec![0.5, 0.5]).unwrap();
    index.insert(3, vec![0.0, 1.0]).unwrap();

    // Highest dot product with (1,0) is (1,0) itself; the ranking distance is
    // the negated dot product.
    let results = index.search(&[1.0, 0.0], 1).unwrap();
    assert_eq!(results[0].id, 1);
    assert_eq!(results[0].score, -1.0);
}

#[test]
fn manhattan_metric() {
    let mut index = FlatIndex::new(VectorMetric::L1);

    index.insert(1, vec![0.0, 0.0]).unwrap();
    index.insert(2, vec![5.0, 5.0]).unwrap();

    let results = index.search(&[0.1, 0.1], 1).unwrap();
    assert_eq!(results[0].id, 1);
}

#[test]
fn k_greater_than_size() {
    let mut index = FlatIndex::new(VectorMetric::L2);
    index.insert(1, vec![0.0]).unwrap();
    index.insert(2, vec![1.0]).unwrap();

    let results = index.search(&[0.0], 10).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn k_zero_returns_empty() {
    let mut index = FlatIndex::new(VectorMetric::L2);
    index.insert(1, vec![0.0]).unwrap();

    assert!(index.search(&[0.0], 0).unwrap().is_empty());
}

#[test]
fn duplicate_insert_ignored() {
    let mut index = FlatIndex::new(VectorMetric::L2);
    index.insert(1, vec![0.0]).unwrap();
    index.insert(1, vec![99.0]).unwrap(); // Same ID — ignored
    assert_eq!(index.len(), 1);

    let results = index.search(&[0.0], 1).unwrap();
    assert_eq!(results[0].score, 0.0); // Original vector kept
}

#[test]
fn remove_vector() {
    let mut index = FlatIndex::new(VectorMetric::L2);
    index.insert(1, vec![0.0]).unwrap();
    index.insert(2, vec![1.0]).unwrap();

    assert!(index.remove(1));
    assert_eq!(index.len(), 1);
    assert!(!index.remove(1)); // Already removed

    let results = index.search(&[0.0], 1).unwrap();
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
        index.insert(i, vec).unwrap();
    }

    let query: Vec<f32> = (0..384).map(|d| (d as f32 * 0.01).sin()).collect();
    let results = index.search(&query, 5).unwrap();
    assert_eq!(results.len(), 5);
}

/// A Dot score whose terms overflow f32 with opposite signs is exact in f64:
/// the products cancel to 0 and the candidate ranks by that value, between
/// the others, instead of carrying no distance.
#[test]
fn a_dot_overflowing_f32_ranks_by_its_exact_value() {
    let mut index = FlatIndex::new(VectorMetric::DotProduct);
    index.insert(1, vec![1e20, -1e20]).unwrap();
    index.insert(2, vec![1.0, 0.0]).unwrap();
    index.insert(3, vec![-1.0, 0.0]).unwrap();

    let results = index.search(&[1e20, 1e20], 3).unwrap();
    let ranked: Vec<(u64, f64)> = results.iter().map(|r| (r.id, r.score)).collect();
    let one = f64::from(1e20f32);
    assert_eq!(ranked, [(2, -one), (1, 0.0), (3, one)]);
}

/// Equal scores rank by ascending id: every executor of the same ranking
/// returns the same order, whatever order the vectors were inserted in.
#[test]
fn equal_scores_rank_by_id() {
    let mut index = FlatIndex::new(VectorMetric::L2);
    index.insert(9, vec![1.0, 0.0]).unwrap();
    index.insert(3, vec![1.0, 0.0]).unwrap();
    index.insert(5, vec![1.0, 0.0]).unwrap();

    let ids: Vec<u64> = index
        .search(&[1.0, 0.0], 3)
        .unwrap()
        .iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(ids, [3, 5, 9]);
}

/// A zero vector is refused by a cosine index, as a stored vector and as a
/// query, and accepted by the other metrics.
#[test]
fn a_zero_vector_is_a_cosine_domain_error() {
    let mut cosine = FlatIndex::new(VectorMetric::Cosine);
    assert_eq!(
        cosine.insert(1, vec![0.0, 0.0]),
        Err(VectorDomainError::ZeroVectorCosine)
    );
    assert!(cosine.is_empty());
    cosine.insert(2, vec![1.0, 0.0]).unwrap();
    assert_eq!(
        cosine.search(&[0.0, -0.0], 1),
        Err(VectorDomainError::ZeroVectorCosine)
    );

    for metric in [VectorMetric::L2, VectorMetric::DotProduct, VectorMetric::L1] {
        let mut index = FlatIndex::new(metric);
        index.insert(1, vec![0.0, 0.0]).unwrap();
        assert_eq!(index.search(&[0.0, 0.0], 1).unwrap()[0].id, 1);
    }
}

/// Non-finite, empty and mismatched vectors are refused before anything is
/// stored or scored.
#[test]
fn invalid_vectors_are_refused() {
    let mut index = FlatIndex::new(VectorMetric::L2);
    assert_eq!(
        index.insert(1, vec![1.0, f32::NAN]),
        Err(VectorDomainError::NonFinite { index: 1 })
    );
    assert_eq!(index.insert(1, Vec::new()), Err(VectorDomainError::Empty));
    assert!(index.is_empty());
    index.insert(1, vec![1.0, 2.0]).unwrap();
    assert_eq!(
        index.insert(2, vec![1.0, 2.0, 3.0]),
        Err(VectorDomainError::DimensionMismatch { left: 3, right: 2 })
    );
    assert_eq!(
        index.search(&[1.0], 1),
        Err(VectorDomainError::DimensionMismatch { left: 1, right: 2 })
    );
    assert_eq!(
        index.search(&[f32::INFINITY, 0.0], 1),
        Err(VectorDomainError::NonFinite { index: 0 })
    );
    assert_eq!(index.len(), 1);
}
