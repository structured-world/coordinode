use super::*;
use coordinode_core::graph::types::VectorMetric;

fn mk_config(dim: u32) -> HnswConfig {
    HnswConfig {
        m: 8,
        m_max0: 16,
        ef_construction: 50,
        ef_search: 20,
        metric: VectorMetric::L2,
        max_dimensions: dim,
        ..HnswConfig::default()
    }
}

#[test]
fn insert_and_search() {
    let store = LocalVectorStore::new(mk_config(3));
    store.insert(1, vec![1.0, 0.0, 0.0]).unwrap();
    store.insert(2, vec![0.0, 1.0, 0.0]).unwrap();
    store.insert(3, vec![0.0, 0.0, 1.0]).unwrap();

    let results = store.knn_search(&[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 1, "closest should be node 1");
}

#[test]
fn empty_store_search_returns_empty() {
    let store = LocalVectorStore::new(mk_config(3));
    assert!(store.is_empty().unwrap());
    let results = store.knn_search(&[1.0, 0.0, 0.0], 5).unwrap();
    assert!(results.is_empty());
}

#[test]
fn bulk_insert_counts() {
    let store = LocalVectorStore::new(mk_config(2));
    let mut iter = (0..10u64).map(|i| (i, vec![i as f32, (10 - i) as f32]));
    let n = store.bulk_insert(&mut iter).unwrap();
    assert_eq!(n, 10);
    assert_eq!(store.len().unwrap(), 10);
}

/// A removed node leaves the index: it is no longer counted or returned,
/// the other nodes still are, and removing it again changes nothing.
#[test]
fn remove_takes_the_node_out_of_the_index() {
    let store = LocalVectorStore::new(mk_config(2));
    store.insert(42, vec![1.0, 2.0]).unwrap();
    store.insert(43, vec![2.0, 1.0]).unwrap();
    store.remove(42).unwrap();
    assert_eq!(store.len().unwrap(), 1);
    let hits = store.knn_search(&[1.0, 2.0], 2).unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![43]);
    store.remove(42).unwrap();
    assert_eq!(store.len().unwrap(), 1);
}

#[test]
fn handle_clone_shares_index() {
    let a = LocalVectorStore::new(mk_config(2));
    let b = LocalVectorStore::from_index(a.handle());
    a.insert(7, vec![3.0, 4.0]).unwrap();
    assert_eq!(b.len().unwrap(), 1);
}

#[test]
fn insert_same_id_twice_dedups_by_id() {
    // Verified behaviour: HnswIndex maintains an id→idx map and
    // overwrites the slot when the same id is re-inserted. The
    // second vector replaces the first. `len()` stays at 1.
    //
    // (Initial intuition was the opposite — HNSW graphs without
    // explicit dedup would grow. The id_to_idx HashMap in
    // coordinode-vector guards against that. This test pins the
    // observed behaviour so a future implementation change does
    // not silently revert it.)
    let store = LocalVectorStore::new(mk_config(2));
    store.insert(1, vec![0.0, 0.0]).unwrap();
    store.insert(1, vec![5.0, 5.0]).unwrap();
    assert_eq!(store.len().unwrap(), 1, "same id must dedup");
    // KNN from (5, 5) returns the updated vector at distance 0.
    let res = store.knn_search(&[5.0, 5.0], 1).unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].id, 1);
    assert_eq!(res[0].score, 0.0, "the exact distance at (5,5) is zero");
}

#[test]
fn knn_returns_full_set_when_k_exceeds_size() {
    // k > number of inserted points must not panic — return what
    // we have, in distance order.
    let store = LocalVectorStore::new(mk_config(2));
    store.insert(1, vec![0.0, 0.0]).unwrap();
    store.insert(2, vec![1.0, 1.0]).unwrap();
    let results = store.knn_search(&[0.0, 0.0], 10).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn knn_results_sorted_by_distance_ascending() {
    let store = LocalVectorStore::new(mk_config(2));
    // Three points at increasing distance from origin.
    store.insert(1, vec![3.0, 0.0]).unwrap();
    store.insert(2, vec![1.0, 0.0]).unwrap();
    store.insert(3, vec![2.0, 0.0]).unwrap();
    let results = store.knn_search(&[0.0, 0.0], 3).unwrap();
    assert_eq!(results.len(), 3);
    let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
    assert_eq!(ids, vec![2, 3, 1]);
    // Strictly ascending distance.
    for pair in results.windows(2) {
        assert!(
            pair[0].score <= pair[1].score,
            "score not ascending: {} -> {}",
            pair[0].score,
            pair[1].score,
        );
    }
}

#[test]
fn bulk_insert_empty_iter_returns_zero() {
    let store = LocalVectorStore::new(mk_config(2));
    let mut empty = std::iter::empty::<(u64, Vec<f32>)>();
    let n = store.bulk_insert(&mut empty).unwrap();
    assert_eq!(n, 0);
    assert!(store.is_empty().unwrap());
}

#[test]
fn from_index_wraps_existing_handle() {
    let raw = Arc::new(RwLock::new(HnswIndex::new(mk_config(2))));
    raw.write().unwrap().insert(99, vec![5.0, 6.0]).unwrap();
    let store = LocalVectorStore::from_index(Arc::clone(&raw));
    assert_eq!(store.len().unwrap(), 1);
    let results = store.knn_search(&[5.0, 6.0], 1).unwrap();
    assert_eq!(results[0].id, 99);
}

/// A vector the index's metric cannot evaluate is refused as a typed store
/// error and leaves the index as it was: a NaN under L2, a zero vector under
/// cosine, a vector of another width than the indexed ones.
#[test]
fn an_unevaluable_vector_is_refused_and_not_indexed() {
    use coordinode_vector::hnsw::VectorDomainError;
    let store = LocalVectorStore::new(mk_config(2));
    store.insert(1, vec![1.0, 2.0]).unwrap();
    let refusal = |r: StoreResult<()>| match r {
        Err(StoreError::VectorAdmission(refused)) => refused.error,
        other => panic!("expected a vector refusal, got {other:?}"),
    };
    assert_eq!(
        refusal(store.insert(2, vec![f32::NAN, 1.0])),
        VectorDomainError::NonFinite { index: 0 }
    );
    assert_eq!(
        refusal(store.insert(3, vec![1.0, 2.0, 3.0])),
        VectorDomainError::DimensionMismatch { left: 3, right: 2 }
    );
    let cosine = LocalVectorStore::new(HnswConfig {
        metric: VectorMetric::Cosine,
        ..mk_config(2)
    });
    assert_eq!(
        refusal(cosine.insert(4, vec![0.0, 0.0])),
        VectorDomainError::ZeroVectorCosine
    );
    assert_eq!(store.len().unwrap(), 1);
    assert!(cosine.is_empty().unwrap());
}

/// A bulk insert holding one unevaluable vector inserts none of the batch.
#[test]
fn bulk_insert_refuses_the_whole_batch() {
    let store = LocalVectorStore::new(mk_config(2));
    let mut items = vec![
        (1u64, vec![1.0, 0.0]),
        (2, vec![f32::INFINITY, 0.0]),
        (3, vec![0.0, 1.0]),
    ]
    .into_iter();
    assert!(matches!(
        store.bulk_insert(&mut items),
        Err(StoreError::VectorAdmission(refused)) if refused.id == 2
    ));
    assert!(store.is_empty().unwrap());
}

/// A query the metric cannot evaluate is a typed search error, never an
/// empty or NaN-scored answer.
#[test]
fn an_unevaluable_query_is_a_search_error() {
    use coordinode_vector::hnsw::{VectorDomainError, VectorSearchError};
    let store = LocalVectorStore::new(mk_config(2));
    store.insert(1, vec![1.0, 2.0]).unwrap();
    for (query, expected) in [
        (
            vec![f32::NAN, 0.0],
            VectorDomainError::NonFinite { index: 0 },
        ),
        (
            vec![1.0],
            VectorDomainError::DimensionMismatch { left: 1, right: 2 },
        ),
    ] {
        match store.knn_search(&query, 1) {
            Err(StoreError::VectorSearch(VectorSearchError::Domain(e))) => assert_eq!(e, expected),
            other => panic!("expected a domain error for {query:?}, got {other:?}"),
        }
    }
}
