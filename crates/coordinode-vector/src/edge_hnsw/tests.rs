use super::*;
use coordinode_core::graph::types::VectorMetric;

fn test_config() -> HnswConfig {
    HnswConfig {
        m: 4,
        m_max0: 8,
        ef_construction: 16,
        ef_search: 10,
        metric: VectorMetric::Cosine,
        max_dimensions: 65_536,
        ..Default::default()
    }
}

#[test]
fn empty_edge_index() {
    let index = EdgeHnswIndex::new("KNOWS", "embedding", test_config());
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert!(index.search(&[1.0, 0.0], 5).is_empty());
}

#[test]
fn insert_and_search_edges() {
    let mut index = EdgeHnswIndex::new("KNOWS", "relationship_embedding", test_config());

    // Alice→Bob: technical collaboration
    index.insert(1, 2, vec![0.9, 0.1, 0.0]);
    // Alice→Charlie: personal friendship
    index.insert(1, 3, vec![0.1, 0.9, 0.0]);
    // Bob→Charlie: professional mentor
    index.insert(2, 3, vec![0.7, 0.3, 0.0]);

    assert_eq!(index.len(), 3);

    // Search for "technical collaboration" pattern
    let results = index.search(&[1.0, 0.0, 0.0], 2);
    assert_eq!(results.len(), 2);

    // Most similar to (1,0,0) should be Alice→Bob (0.9, 0.1, 0)
    assert_eq!(results[0].source, 1);
    assert_eq!(results[0].target, 2);
}

#[test]
fn edge_search_returns_source_target() {
    let mut index = EdgeHnswIndex::new("FOLLOWS", "vec", test_config());

    index.insert(10, 20, vec![1.0, 0.0]);
    index.insert(30, 40, vec![0.0, 1.0]);

    let results = index.search(&[1.0, 0.0], 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source, 10);
    assert_eq!(results[0].target, 20);
}

#[test]
fn multiple_edges_from_same_source() {
    let mut index = EdgeHnswIndex::new("RATED", "sentiment_vec", test_config());

    // User 1 rates multiple movies with different sentiment vectors
    index.insert(1, 100, vec![0.9, 0.1]); // Positive review
    index.insert(1, 101, vec![0.1, 0.9]); // Negative review
    index.insert(1, 102, vec![0.5, 0.5]); // Mixed review

    let results = index.search(&[0.8, 0.2], 3);
    assert_eq!(results.len(), 3);
    // All from source 1
    assert!(results.iter().all(|r| r.source == 1));
}

#[test]
fn edge_type_and_property_stored() {
    let index = EdgeHnswIndex::new("TRANSACTION", "fraud_embedding", test_config());
    assert_eq!(index.edge_type, "TRANSACTION");
    assert_eq!(index.property, "fraud_embedding");
}

#[test]
fn high_dimensional_edge_vectors() {
    let mut index = EdgeHnswIndex::new(
        "KNOWS",
        "embedding",
        HnswConfig {
            m: 4,
            m_max0: 8,
            ef_construction: 16,
            ef_search: 10,
            metric: VectorMetric::L2,
            max_dimensions: 65_536,
            ..Default::default()
        },
    );

    // 384-dim edge vectors
    for i in 0..10u64 {
        let vec: Vec<f32> = (0..384)
            .map(|d| ((i * d + 1) as f32 * 0.01).sin())
            .collect();
        index.insert(i, i + 100, vec);
    }

    assert_eq!(index.len(), 10);

    let query: Vec<f32> = (0..384).map(|d| (d as f32 * 0.01).sin()).collect();
    let results = index.search(&query, 3);
    assert_eq!(results.len(), 3);
}
