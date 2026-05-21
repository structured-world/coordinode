//! Per-node CPU throttle for HNSW (re)builds.
//!
//! See `arch/distribution/live-rebalance.md § Build throttle` for the
//! contract. The scheduler is a counting semaphore with FIFO ordering plus
//! a priority hint that lets emergency re-replication jump the queue when
//! the cluster is recovering from a node loss.
//!
//! ## Why this exists
//!
//! Without a per-node cap, a decommission event draining N vector-heavy
//! segments triggers N concurrent rebuilds on every target node. On a
//! 16-core box with N=50, foreground search latency P99 spikes by orders of
//! magnitude until the rebuild storm drains. The token bucket bounds
//! parallelism to `max(1, cores/4)` by default so foreground query traffic
//! keeps headroom. The R858b-pre1 migration planner queries
//! [`HnswBuildScheduler::queue_depth`] when scoring a candidate target so
//! it can spread rebuild load across the cluster.
//!
//! ## Why a custom scheduler instead of `tokio::sync::Semaphore`
//!
//! Three properties that an off-the-shelf semaphore doesn't give:
//!
//! 1. **Priority preemption.** `Priority::Emergency` waiters jump ahead of
//!    `Priority::Normal` waiters regardless of FIFO order. Plain semaphores
//!    are strictly FIFO.
//! 2. **Sync API (no tokio dependency in coordinode-vector).** Build paths
//!    run on rayon worker threads, not async tasks; an async semaphore
//!    forces a tokio handle through every call site.
//! 3. **Queue-depth observability.** `queue_depth()` is read by the migration
//!    planner on every cost evaluation; tokio's semaphore exposes only
//!    available permits, not pending waiters.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

/// Priority hint for build requests. Emergency re-replication (DC failure
/// scenario in `live-rebalance.md`) preempts the FIFO; normal rebuilds do
/// not. Within the same priority class, ordering is strict FIFO.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Background work — initial replica build, planned migration rebuild,
    /// lazy-prune rebuild after SK3+SK4 reshard.
    Normal,
    /// Recovery from node / DC loss. Jumps ahead of `Normal` waiters
    /// because the cluster has reduced redundancy until this build completes.
    Emergency,
}

/// Per-node token-bucket scheduler for HNSW builds.
///
/// Construct once at `coordinode-server` boot, share via `Arc` across
/// `coordinode-storage`, `coordinode-cluster`, and `coordinode-vector` build
/// entry points. Build code acquires a permit via [`acquire`] before
/// starting work; the permit's `Drop` releases the slot back to the bucket.
///
/// # Concurrency model
///
/// One `Mutex<SchedulerInner>` guards the bucket state; a `Condvar` wakes
/// waiters when permits free up. The lock is held only across the queue
/// manipulation, not across the build itself — the contention surface is
/// `O(builds_per_second)`, not `O(time_spent_building)`.
#[derive(Debug)]
pub struct HnswBuildScheduler {
    capacity: usize,
    inner: Mutex<SchedulerInner>,
    available: Condvar,
}

#[derive(Debug)]
struct SchedulerInner {
    /// Permits currently in use.
    in_flight: usize,
    /// Pending waiters by priority. Two queues so emergency can jump.
    /// Each waiter is just a counter we use to assign a stable FIFO ticket.
    normal_waiters: VecDeque<u64>,
    emergency_waiters: VecDeque<u64>,
    next_ticket: u64,
}

/// RAII permit. Holding one means "you have a build slot". Dropping
/// returns the slot to the bucket and wakes the next waiter.
pub struct BuildPermit {
    scheduler: std::sync::Arc<HnswBuildScheduler>,
}

impl HnswBuildScheduler {
    /// Construct a scheduler with the given concurrent-build capacity.
    /// The recommended default at `coordinode-server` boot is
    /// `max(1, num_cpus / 4)` — leaves 75% of cores for foreground search
    /// traffic and parallel insert work from R858a-d.
    pub fn new(capacity: usize) -> std::sync::Arc<Self> {
        let capacity = capacity.max(1);
        std::sync::Arc::new(Self {
            capacity,
            inner: Mutex::new(SchedulerInner {
                in_flight: 0,
                normal_waiters: VecDeque::new(),
                emergency_waiters: VecDeque::new(),
                next_ticket: 0,
            }),
            available: Condvar::new(),
        })
    }

    /// Maximum concurrent builds permitted.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Acquire a permit, blocking until one is available. Permits are
    /// returned automatically when the [`BuildPermit`] is dropped.
    ///
    /// Within the same priority class, ordering is strict FIFO so a long
    /// rebuild queue can't starve early arrivals indefinitely.
    /// `Priority::Emergency` waiters are served before any `Normal` waiter
    /// regardless of arrival order.
    pub fn acquire(self: &std::sync::Arc<Self>, priority: Priority) -> BuildPermit {
        // Each waiter takes a stable FIFO ticket so we can release in
        // arrival order within a priority class. Tickets are u64 so a
        // pathological 32-bit wrap is not a concern (would require ~10^11
        // builds before reuse).
        let my_ticket = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let ticket = inner.next_ticket;
            inner.next_ticket = inner.next_ticket.wrapping_add(1);
            match priority {
                Priority::Normal => inner.normal_waiters.push_back(ticket),
                Priority::Emergency => inner.emergency_waiters.push_back(ticket),
            }
            ticket
        };

        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            // We can take the permit iff (a) there is a free slot AND
            // (b) we are at the head of the highest-priority non-empty
            // queue. Emergency queue wins outright.
            let at_head = if !inner.emergency_waiters.is_empty() {
                inner.emergency_waiters.front() == Some(&my_ticket)
            } else {
                inner.normal_waiters.front() == Some(&my_ticket)
            };
            if at_head && inner.in_flight < self.capacity {
                // Take the permit and pop ourselves off the queue.
                inner.in_flight += 1;
                match priority {
                    Priority::Normal => {
                        inner.normal_waiters.pop_front();
                    }
                    Priority::Emergency => {
                        inner.emergency_waiters.pop_front();
                    }
                }
                // If there is still room AND more waiters, wake one so
                // they can re-check. (Condvar::notify_one is cheap.)
                if inner.in_flight < self.capacity
                    && (!inner.emergency_waiters.is_empty() || !inner.normal_waiters.is_empty())
                {
                    self.available.notify_one();
                }
                drop(inner);
                return BuildPermit {
                    scheduler: self.clone(),
                };
            }
            inner = self
                .available
                .wait(inner)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Number of builds currently in flight (holding permits).
    pub fn in_flight(&self) -> usize {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.in_flight
    }

    /// Number of pending waiters across all priority classes. Read by the
    /// R858b-pre1 migration planner to penalise targets with deep build
    /// queues when scoring candidate destinations.
    pub fn queue_depth(&self) -> usize {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.normal_waiters.len() + inner.emergency_waiters.len()
    }

    /// Internal release path. Invoked from `BuildPermit::drop`.
    fn release(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        debug_assert!(inner.in_flight > 0, "release without acquire");
        inner.in_flight = inner.in_flight.saturating_sub(1);
        // Wake every waiter — they will re-check priority + ticket order.
        // Using notify_all avoids a missed-wakeup race when an emergency
        // arrival overtakes a normal waiter that thought it was next.
        self.available.notify_all();
    }
}

impl Drop for BuildPermit {
    fn drop(&mut self) {
        self.scheduler.release();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
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
}
