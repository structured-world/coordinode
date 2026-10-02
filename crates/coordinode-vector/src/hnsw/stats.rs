//! Accounting for the published neighbour lists and the slots of one index:
//! what replaced lists still hold until reclamation, how often publication
//! lost a race, how long the oldest operation has been protecting memory,
//! and how often writers waited on the retired-memory budget.
//!
//! Counters are relaxed atomics: each is read on its own for observability,
//! never combined with another into a decision except the budget check, which
//! tolerates a stale value by construction (it re-reads before giving up).

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Operations tracked at once for [`PublicationSnapshot::oldest_operation`];
/// one more running concurrently goes uncounted, which only makes the
/// reported age a lower bound.
const TRACKED_OPERATIONS: usize = 64;

/// Live counters of one index. Shared with the reclamation closures, which
/// may run after the index itself is gone.
#[derive(Debug)]
pub(crate) struct PublicationStats {
    /// Bytes of replaced lists waiting for reclamation.
    retired_bytes: AtomicUsize,
    /// Replaced lists waiting for reclamation.
    retired_lists: AtomicUsize,
    /// Publications that lost the descriptor CAS and recomputed.
    lost_cas: AtomicU64,
    /// Inserts that waited for reclamation to bring the retired bytes under
    /// the budget, and the nanoseconds they waited in total.
    admission_waits: AtomicU64,
    admission_wait_nanos: AtomicU64,
    /// Start of every tracked operation, as nanoseconds after `origin` plus
    /// one; 0 marks a free slot.
    operations: [AtomicU64; TRACKED_OPERATIONS],
    origin: Instant,
}

impl PublicationStats {
    pub(crate) fn new() -> Self {
        Self {
            retired_bytes: AtomicUsize::new(0),
            retired_lists: AtomicUsize::new(0),
            lost_cas: AtomicU64::new(0),
            admission_waits: AtomicU64::new(0),
            admission_wait_nanos: AtomicU64::new(0),
            operations: core::array::from_fn(|_| AtomicU64::new(0)),
            origin: Instant::now(),
        }
    }

    /// A replaced list of `bytes` was handed to the epoch.
    #[inline]
    pub(crate) fn list_retired(&self, bytes: usize) {
        self.retired_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.retired_lists.fetch_add(1, Ordering::Relaxed);
    }

    /// A replaced list of `bytes` was freed.
    #[inline]
    pub(crate) fn list_reclaimed(&self, bytes: usize) {
        self.retired_bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.retired_lists.fetch_sub(1, Ordering::Relaxed);
    }

    /// A publication lost its CAS and is recomputing.
    #[cold]
    pub(crate) fn cas_lost(&self) {
        self.lost_cas.fetch_add(1, Ordering::Relaxed);
    }

    /// Bytes of replaced lists not yet reclaimed.
    #[inline]
    pub(crate) fn retired_bytes(&self) -> usize {
        self.retired_bytes.load(Ordering::Relaxed)
    }

    /// An insert waited `nanos` for the retired bytes to fall under budget.
    pub(crate) fn admission_waited(&self, nanos: u64) {
        self.admission_waits.fetch_add(1, Ordering::Relaxed);
        self.admission_wait_nanos
            .fetch_add(nanos, Ordering::Relaxed);
    }

    /// Track an operation that protects memory (a search, an insert, a
    /// removal) until the returned guard drops.
    #[inline]
    pub(crate) fn begin(&self) -> OperationGuard<'_> {
        let stamp = self.origin.elapsed().as_nanos() as u64 + 1;
        // A thread's stack address spreads concurrent callers over the
        // slots without a shared counter to contend on.
        let probe = (&stamp as *const u64 as usize >> 12) % TRACKED_OPERATIONS;
        for step in 0..TRACKED_OPERATIONS {
            let slot = &self.operations[(probe + step) % TRACKED_OPERATIONS];
            if slot
                .compare_exchange(0, stamp, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return OperationGuard { slot: Some(slot) };
            }
        }
        OperationGuard { slot: None }
    }

    /// Whether any tracked operation is running.
    pub(crate) fn any_running(&self) -> bool {
        self.operations
            .iter()
            .any(|slot| slot.load(Ordering::Relaxed) != 0)
    }

    /// The counters as they stand.
    pub(crate) fn snapshot(&self) -> PublicationSnapshot {
        let now = self.origin.elapsed().as_nanos() as u64 + 1;
        let oldest = self
            .operations
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .filter(|&start| start != 0)
            .min()
            // An operation stamped after `now` was read has just begun: an
            // age of zero is exactly right for it.
            .map_or(0, |start| now.saturating_sub(start));
        PublicationSnapshot {
            retired_bytes: self.retired_bytes(),
            retired_lists: self.retired_lists.load(Ordering::Relaxed),
            lost_cas: self.lost_cas.load(Ordering::Relaxed),
            admission_waits: self.admission_waits.load(Ordering::Relaxed),
            admission_wait: std::time::Duration::from_nanos(
                self.admission_wait_nanos.load(Ordering::Relaxed),
            ),
            oldest_operation: std::time::Duration::from_nanos(oldest),
            retired_nodes: 0,
            free_slots: 0,
        }
    }
}

/// Ends the tracking of one operation when dropped.
pub(crate) struct OperationGuard<'a> {
    slot: Option<&'a AtomicU64>,
}

impl Drop for OperationGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        if let Some(slot) = self.slot {
            slot.store(0, Ordering::Relaxed);
        }
    }
}

/// Publication and reclamation figures of one index, from
/// [`HnswIndex::publication_stats`](super::HnswIndex::publication_stats).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PublicationSnapshot {
    /// Bytes of replaced neighbour lists that a running search or insert may
    /// still read, so they are not freed yet.
    pub retired_bytes: usize,
    /// Number of those lists.
    pub retired_lists: usize,
    /// Publications that lost the compare-and-swap to a concurrent writer and
    /// were recomputed, since the index was created.
    pub lost_cas: u64,
    /// Inserts that waited for replaced lists to be freed before going ahead,
    /// because the retired bytes were over the budget.
    pub admission_waits: u64,
    /// Total time those inserts waited.
    pub admission_wait: std::time::Duration,
    /// Age of the oldest search, insert or removal running on the index: how
    /// long the oldest protection that holds replaced memory has been held.
    /// Zero when none is running.
    pub oldest_operation: std::time::Duration,
    /// Nodes removed or replaced whose slots are not yet free for reuse.
    pub retired_nodes: usize,
    /// Slots free for the next inserts to reuse.
    pub free_slots: usize,
}
