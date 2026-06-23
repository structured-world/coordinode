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

#[test]
fn remove_is_noop_for_index() {
    let store = LocalVectorStore::new(mk_config(2));
    store.insert(42, vec![1.0, 2.0]).unwrap();
    store.remove(42).unwrap();
    // HNSW retains the vector — query layer filters via MVCC.
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
    assert!(res[0].score.abs() < 1e-6, "score should be ~0 at (5,5)");
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
    raw.write().unwrap().insert(99, vec![5.0, 6.0]);
    let store = LocalVectorStore::from_index(Arc::clone(&raw));
    assert_eq!(store.len().unwrap(), 1);
    let results = store.knn_search(&[5.0, 6.0], 1).unwrap();
    assert_eq!(results[0].id, 99);
}
