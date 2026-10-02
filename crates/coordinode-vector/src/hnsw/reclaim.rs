//! Reuse of the slots of removed and replaced nodes.
//!
//! A retired node stays navigable: readers that reach it through a list
//! continue through it, and lists that other writers publish may still carry
//! it. Its slot becomes reusable in three steps, each ended by the epoch, so
//! that no operation running across a step sees the slot change under it:
//!
//! 1. **Retired.** The node is marked retired, so a writer that starts from
//!    now on never links it. Once every operation pinned at the retirement has
//!    ended, no writer can still link it: the slot is *unlinkable*.
//! 2. **Swept.** A sweep republishes every list that names an unlinkable slot
//!    without it and marks the slot free. No new list names it; operations
//!    that read an older list may still hold it.
//! 3. **Free.** Once every operation pinned at the sweep has ended, nothing
//!    can reach the slot and an insert may rewrite it.
//!
//! A reorder renumbers every slot under `&mut`: it drops the retired slots
//! and starts a new generation, so steps queued by the epoch for the old
//! numbering are discarded when they run.

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

/// Slots waiting for their next step, shared with the epoch's deferred
/// closures, which may run after the index is gone.
#[derive(Debug, Default)]
pub(super) struct SlotReclaim {
    /// Bumped by a reorder; a queued step of an older generation is dropped.
    generation: AtomicU64,
    /// Retired slots no writer can link any more, waiting for a sweep.
    unlinkable: Mutex<Vec<usize>>,
    unlinkable_len: AtomicUsize,
    /// Swept slots nothing can reach, ready for reuse.
    free: Mutex<Vec<usize>>,
    free_len: AtomicUsize,
    /// Slots retired and not yet back in `free`.
    retired: AtomicUsize,
    /// Held by the one sweep that runs at a time.
    sweeping: Mutex<()>,
}

impl SlotReclaim {
    /// Slot `idx` was retired under `guard`: it joins the sweep queue once
    /// every operation pinned now has ended.
    pub(super) fn retired(self: &Arc<Self>, idx: usize, guard: &crossbeam_epoch::Guard) {
        self.retired.fetch_add(1, Ordering::Relaxed);
        let reclaim = Arc::clone(self);
        let generation = self.generation.load(Ordering::Acquire);
        guard.defer(move || {
            // Checked under the queue's lock, which `reset` holds while it
            // moves to the next generation.
            let mut queue = reclaim.unlinkable.lock();
            if reclaim.generation.load(Ordering::Acquire) == generation {
                queue.push(idx);
                reclaim.unlinkable_len.fetch_add(1, Ordering::Release);
            }
        });
    }

    /// Slots waiting for a sweep.
    #[inline]
    pub(super) fn unlinkable_len(&self) -> usize {
        self.unlinkable_len.load(Ordering::Acquire)
    }

    /// Run `sweep` over the queued slots unless another sweep is running.
    /// `sweep` returns the slots it swept (the rest go back to the queue);
    /// those become free once every operation pinned now has ended.
    pub(super) fn sweep_with(
        self: &Arc<Self>,
        guard: &crossbeam_epoch::Guard,
        sweep: impl FnOnce(Vec<usize>) -> (Vec<usize>, Vec<usize>),
    ) {
        let Some(_running) = self.sweeping.try_lock() else {
            return;
        };
        let queued = {
            let mut queue = self.unlinkable.lock();
            self.unlinkable_len.store(0, Ordering::Release);
            std::mem::take(&mut *queue)
        };
        if queued.is_empty() {
            return;
        }
        let (swept, kept) = sweep(queued);
        if !kept.is_empty() {
            let mut queue = self.unlinkable.lock();
            self.unlinkable_len.fetch_add(kept.len(), Ordering::Release);
            queue.extend(kept);
        }
        if swept.is_empty() {
            return;
        }
        let reclaim = Arc::clone(self);
        let generation = self.generation.load(Ordering::Acquire);
        guard.defer(move || {
            // As in `retired`: the generation is checked under the lock
            // `reset` holds.
            let mut free = reclaim.free.lock();
            if reclaim.generation.load(Ordering::Acquire) == generation {
                let n = swept.len();
                free.extend(swept);
                reclaim.free_len.fetch_add(n, Ordering::Release);
                reclaim.retired.fetch_sub(n, Ordering::Relaxed);
            }
        });
    }

    /// A free slot for an insert, if any. The empty case is one load.
    #[inline]
    pub(super) fn take_free(&self) -> Option<usize> {
        if self.free_len.load(Ordering::Acquire) == 0 {
            return None;
        }
        self.take_free_slow()
    }

    #[cold]
    fn take_free_slow(&self) -> Option<usize> {
        let idx = self.free.lock().pop()?;
        self.free_len.fetch_sub(1, Ordering::Release);
        Some(idx)
    }

    /// Slots retired and not yet free, and slots free.
    pub(super) fn counts(&self) -> (usize, usize) {
        (
            self.retired.load(Ordering::Relaxed),
            self.free_len.load(Ordering::Acquire),
        )
    }

    /// Forget every queued slot: a reorder renumbered them and dropped the
    /// retired ones. Steps the epoch still holds for the old numbering are
    /// discarded when they run.
    pub(super) fn reset(&self) {
        // Both locks across the generation change: a deferred step either
        // lands before it (and is cleared here) or sees the new generation.
        let mut free = self.free.lock();
        let mut queue = self.unlinkable.lock();
        self.generation.fetch_add(1, Ordering::AcqRel);
        queue.clear();
        self.unlinkable_len.store(0, Ordering::Release);
        free.clear();
        self.free_len.store(0, Ordering::Release);
        self.retired.store(0, Ordering::Relaxed);
    }
}
