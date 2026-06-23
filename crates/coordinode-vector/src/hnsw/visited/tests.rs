use super::*;

#[test]
fn test_basic_check_and_mark() {
    let pool = VisitedPool::new();
    let mut handle = pool.get(100);

    assert!(!handle.is_visited(0));
    assert!(!handle.is_visited(50));
    assert!(!handle.is_visited(99));

    // First mark returns false (was not visited)
    assert!(!handle.check_and_mark(50));
    assert!(handle.is_visited(50));

    // Second mark returns true (already visited)
    assert!(handle.check_and_mark(50));

    // Other nodes still not visited
    assert!(!handle.is_visited(0));
    assert!(!handle.is_visited(99));
}

#[test]
fn test_epoch_reset_on_reuse() {
    let pool = VisitedPool::new();

    // First search: mark node 42
    {
        let mut handle = pool.get(100);
        handle.check_and_mark(42);
        assert!(handle.is_visited(42));
    } // handle dropped, list returned to pool

    // Second search: same list reused, node 42 should NOT be visited
    {
        let handle = pool.get(100);
        assert!(!handle.is_visited(42));
    }
}

#[test]
fn test_epoch_wrap_around() {
    let pool = VisitedPool::new();

    // Force 255 epoch advances (u8 wraps at 256)
    for i in 0..256 {
        let mut handle = pool.get(10);
        handle.check_and_mark(0);
        assert!(
            handle.is_visited(0),
            "node 0 should be visited at iteration {i}"
        );
    }

    // After 255 reuses, epoch wrapped, fill(0) was called — still works
    let handle = pool.get(10);
    assert!(!handle.is_visited(0));
}

#[test]
fn test_dynamic_resize() {
    let pool = VisitedPool::new();
    let mut handle = pool.get(10);

    // Mark node beyond initial capacity
    assert!(!handle.check_and_mark(100));
    assert!(handle.is_visited(100));
}

/// Read the calling thread's pool size — only valid in single-thread
/// tests where the pool is observed from the same OS thread that
/// drives it.
fn thread_pool_size() -> usize {
    THREAD_POOL.with(|cell| cell.borrow().len())
}

/// Reset the calling thread's cache so a test starts with a clean
/// per-thread slate. Without this, the cache survives across tests
/// run on the same thread.
fn clear_thread_pool() {
    THREAD_POOL.with(|cell| cell.borrow_mut().clear());
}

#[test]
fn test_pool_recycles() {
    clear_thread_pool();
    let pool = VisitedPool::new();

    // Create and drop a handle — list returned to the thread cache.
    {
        let mut handle = pool.get(1000);
        handle.check_and_mark(500);
    }
    assert_eq!(thread_pool_size(), 1);

    // Getting reuses the cached list (pops from cache).
    let handle = pool.get(1000);
    assert_eq!(thread_pool_size(), 0);

    // Dropping returns it back.
    drop(handle);
    assert_eq!(thread_pool_size(), 1);

    // Three concurrent handles drain the cache and trigger fresh
    // allocations.
    let h1 = pool.get(1000);
    let h2 = pool.get(1000);
    let h3 = pool.get(1000);
    assert_eq!(thread_pool_size(), 0);

    drop(h1);
    drop(h2);
    drop(h3);
    assert_eq!(thread_pool_size(), 3);
}

#[test]
fn test_pool_max_size() {
    clear_thread_pool();
    let pool = VisitedPool::new();

    let handles: Vec<_> = (0..20).map(|_| pool.get(10)).collect();
    drop(handles);

    assert!(thread_pool_size() <= pool.max_pool_size);
}
