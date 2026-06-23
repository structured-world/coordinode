use super::*;

#[test]
fn with_ef_preallocates_storage_at_ef_plus_16() {
    let s = SearchScratch::with_ef(200);
    assert!(s.candidates_storage.capacity() >= 216);
    assert!(s.results_storage.capacity() >= 216);
    assert!(s.connections.capacity() >= M_MAX0);
    assert!(s.unvisited.capacity() >= M_MAX0);
}

#[test]
fn clear_drops_contents_but_keeps_capacity() {
    let mut s = SearchScratch::with_ef(64);
    s.candidates_storage.push(Candidate {
        distance: 1.0,
        idx: 0,
    });
    s.connections.push(42);
    let cap_before = s.candidates_storage.capacity();
    s.clear();
    assert!(s.candidates_storage.is_empty());
    assert!(s.connections.is_empty());
    assert_eq!(s.candidates_storage.capacity(), cap_before);
}

#[test]
fn pool_reuses_returned_scratch() {
    let pool = SearchScratchPool::new();
    let ptr_first = {
        let mut handle = pool.get(64);
        handle.scratch().candidates_storage.push(Candidate {
            distance: 2.0,
            idx: 7,
        });
        handle.scratch().candidates_storage.as_ptr()
    };
    let mut handle2 = pool.get(64);
    let ptr_second = handle2.scratch().candidates_storage.as_ptr();
    assert!(handle2.scratch().candidates_storage.is_empty());
    assert_eq!(
        ptr_first, ptr_second,
        "pool must hand back the same allocation",
    );
}

#[test]
fn pool_caps_retained_entries() {
    let pool = SearchScratchPool::new();
    for _ in 0..pool.max_pool_size + 4 {
        let _h = pool.get(8);
    }
    let len = pool.pool.lock().unwrap_or_else(|p| p.into_inner()).len();
    assert!(len <= pool.max_pool_size);
}
