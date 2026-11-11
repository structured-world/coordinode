//! The bytes allocated by the measuring thread within one sample, and the
//! most of them held at once, whichever thread frees them.
//!
//! Boundary: an allocation counts toward a sample when it is born on the
//! thread that runs the sample, inside it. Allocations of other threads made
//! meanwhile are summed apart ([`ELSEWHERE`]), not attributed to the sample;
//! memory born in an earlier sample and freed in this one is not this
//! sample's. Every sample has its own allocation group, and the group being
//! counted and its live bytes are one atomic word, so a tracker callback that
//! read an earlier sample's group cannot change the current one's count.
//! Samples run one at a time.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use tracking_allocator::{
    AllocationGroupId, AllocationGroupToken, AllocationRegistry, AllocationTracker,
};

/// Bits of a state word holding bytes; the bits above hold the group.
const BYTE_BITS: u32 = 40;
const BYTES: u64 = (1 << BYTE_BITS) - 1;
/// The group of no sample: no registered group reaches it.
const OFF: u64 = (1 << (64 - BYTE_BITS)) - 1;

/// Allocations and bytes allocated by every thread.
pub static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
pub static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
/// Bytes allocated by other groups while a sample was counted.
pub static ELSEWHERE: AtomicUsize = AtomicUsize::new(0);
/// The counted group and the bytes its allocations hold now.
static LIVE: AtomicU64 = AtomicU64::new(OFF << BYTE_BITS);
/// The counted group and the most its allocations held at once.
static PEAK: AtomicU64 = AtomicU64::new(OFF << BYTE_BITS);

struct Peaks;

fn group_of(state: u64) -> u64 {
    state >> BYTE_BITS
}

impl AllocationTracker for Peaks {
    fn allocated(&self, _addr: usize, size: usize, _wrapped: usize, group: AllocationGroupId) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED.fetch_add(size, Ordering::Relaxed);
        let group = group.as_usize().get() as u64;
        let size = size as u64;
        let mut state = LIVE.load(Ordering::Acquire);
        while group_of(state) == group {
            match LIVE.compare_exchange_weak(
                state,
                state + size,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let live = (state & BYTES) + size;
                    debug_assert!(live <= BYTES, "a sample held past the counter's range");
                    // Groups are numbered upward, so a later sample's word
                    // is above every word of an earlier one and the maximum
                    // keeps only the current sample's peak.
                    PEAK.fetch_max((group << BYTE_BITS) | live, Ordering::AcqRel);
                    return;
                }
                Err(now) => state = now,
            }
        }
        if group_of(state) != OFF {
            ELSEWHERE.fetch_add(size as usize, Ordering::Relaxed);
        }
    }

    fn deallocated(
        &self,
        _addr: usize,
        size: usize,
        _wrapped: usize,
        source: AllocationGroupId,
        _current: AllocationGroupId,
    ) {
        let source = source.as_usize().get() as u64;
        let size = size as u64;
        // Only what the counted group allocated since it began is freed
        // under it, so the count never falls below zero.
        let mut state = LIVE.load(Ordering::Acquire);
        while group_of(state) == source {
            debug_assert!(
                state & BYTES >= size,
                "freed more than the sample allocated"
            );
            match LIVE.compare_exchange_weak(
                state,
                state - size,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(now) => state = now,
            }
        }
    }
}

/// Install the counting tracker; once per process, before any sample.
pub fn install() {
    AllocationRegistry::set_global_tracker(Peaks).expect("the only tracker");
    AllocationRegistry::enable_tracking();
}

/// What `f` returns, and the most bytes the allocations it made on this
/// thread held at once.
pub fn peak_of<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let mut token = AllocationGroupToken::register().expect("an allocation group");
    let group = token.id().as_usize().get() as u64;
    assert!(group < OFF, "allocation groups past the counter's range");
    let previous = PEAK.swap(group << BYTE_BITS, Ordering::AcqRel);
    assert!(
        group_of(previous) < group || group_of(previous) == OFF,
        "allocation groups are numbered upward"
    );
    LIVE.store(group << BYTE_BITS, Ordering::Release);
    let value = {
        let _counted = token.enter();
        f()
    };
    LIVE.store(OFF << BYTE_BITS, Ordering::Release);
    (value, PEAK.load(Ordering::Acquire) & BYTES)
}
