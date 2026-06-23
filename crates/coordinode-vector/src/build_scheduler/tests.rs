use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn capacity_is_clamped_to_one_minimum() {
    let s = HnswBuildScheduler::new(0);
    assert_eq!(s.capacity(), 1);
}

#[test]
fn permit_release_on_drop() {
    let s = HnswBuildScheduler::new(2);
    let p1 = s.acquire(Priority::Normal);
    assert_eq!(s.in_flight(), 1);
    let p2 = s.acquire(Priority::Normal);
    assert_eq!(s.in_flight(), 2);
    drop(p1);
    assert_eq!(s.in_flight(), 1);
    drop(p2);
    assert_eq!(s.in_flight(), 0);
}

#[test]
fn bounds_concurrency_to_capacity() {
    let s = HnswBuildScheduler::new(4);
    let observed_max = Arc::new(AtomicUsize::new(0));
    let current = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..50 {
        let s = s.clone();
        let observed_max = observed_max.clone();
        let current = current.clone();
        handles.push(thread::spawn(move || {
            let _permit = s.acquire(Priority::Normal);
            let now = current.fetch_add(1, Ordering::SeqCst) + 1;
            observed_max.fetch_max(now, Ordering::SeqCst);
            // Hold the permit for a short window so concurrency is real.
            thread::sleep(Duration::from_millis(5));
            current.fetch_sub(1, Ordering::SeqCst);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let max = observed_max.load(Ordering::SeqCst);
    assert!(max <= 4, "concurrency cap broken — saw {max} in flight");
    assert!(max > 0, "scheduler never granted permits");
}

#[test]
fn emergency_jumps_normal_queue() {
    let s = HnswBuildScheduler::new(1);
    // Hold the only permit so subsequent acquires queue.
    let hog = s.acquire(Priority::Normal);

    // Order of wins after `hog` drops: emergency_first, then any normal
    // (FIFO among normal). We register the order by appending to a
    // shared Vec under a Mutex.
    let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

    let s_n1 = s.clone();
    let order_n1 = order.clone();
    let n1 = thread::spawn(move || {
        let _p = s_n1.acquire(Priority::Normal);
        order_n1.lock().unwrap().push("n1");
        thread::sleep(Duration::from_millis(20));
    });
    // Give n1 time to enter the queue before emergency arrives.
    thread::sleep(Duration::from_millis(10));

    let s_em = s.clone();
    let order_em = order.clone();
    let em = thread::spawn(move || {
        let _p = s_em.acquire(Priority::Emergency);
        order_em.lock().unwrap().push("emergency");
        thread::sleep(Duration::from_millis(20));
    });
    // Let emergency settle in queue too.
    thread::sleep(Duration::from_millis(10));

    drop(hog);
    em.join().unwrap();
    n1.join().unwrap();

    let order = order.lock().unwrap();
    assert_eq!(
        &order[..],
        &["emergency", "n1"],
        "emergency must serve before queued normal"
    );
}

#[test]
fn normal_priority_is_fifo() {
    let s = HnswBuildScheduler::new(1);
    let hog = s.acquire(Priority::Normal);

    let order: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    for i in 0..5u32 {
        let s = s.clone();
        let order = order.clone();
        handles.push(thread::spawn(move || {
            let _p = s.acquire(Priority::Normal);
            order.lock().unwrap().push(i);
            thread::sleep(Duration::from_millis(5));
        }));
        // Sequential queue entries — each thread sleeps before next is
        // spawned so FIFO ticket order matches spawn order.
        thread::sleep(Duration::from_millis(10));
    }
    drop(hog);
    for h in handles {
        h.join().unwrap();
    }
    let order = order.lock().unwrap();
    assert_eq!(&order[..], &[0, 1, 2, 3, 4], "normal queue must be FIFO");
}

#[test]
fn queue_depth_is_read_by_planner_path() {
    let s = HnswBuildScheduler::new(1);
    // Fill the slot.
    let hog = s.acquire(Priority::Normal);

    let mut waiters = Vec::new();
    for _ in 0..3 {
        let s = s.clone();
        waiters.push(thread::spawn(move || {
            let _p = s.acquire(Priority::Normal);
            thread::sleep(Duration::from_millis(5));
        }));
        thread::sleep(Duration::from_millis(5));
    }
    // Best-effort observation — three waiters should have queued.
    // Allow some slack for thread scheduling.
    let depth = s.queue_depth();
    assert!(depth >= 1, "expected ≥1 queued waiter, got {depth}");
    assert!(depth <= 3, "expected ≤3 queued waiters, got {depth}");

    drop(hog);
    for w in waiters {
        w.join().unwrap();
    }
    assert_eq!(s.queue_depth(), 0);
}
