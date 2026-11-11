//! The process allocator of this crate's unit tests: the system allocator,
//! with each allocation attributed to the allocation group that made it, so
//! a test can count the bytes a piece of work held at once (whichever thread
//! frees them) and compare that against what it reserved. Allocations of
//! other threads are not counted.

use std::sync::Once;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

use tracking_allocator::{
    AllocationGroupId, AllocationGroupToken, AllocationRegistry, AllocationTracker, Allocator,
};

#[global_allocator]
static ALLOCATOR: Allocator<std::alloc::System> = Allocator::system();

/// The group being counted, `0` for none.
static COUNTED: AtomicUsize = AtomicUsize::new(0);
/// Bytes the counted group holds now.
static LIVE: AtomicIsize = AtomicIsize::new(0);
/// The most it held at once.
static PEAK: AtomicIsize = AtomicIsize::new(0);

struct Peaks;

impl AllocationTracker for Peaks {
    fn allocated(&self, _addr: usize, size: usize, _wrapped: usize, group: AllocationGroupId) {
        if group.as_usize().get() == COUNTED.load(Ordering::Relaxed) {
            let live = LIVE.fetch_add(size.cast_signed(), Ordering::Relaxed) + size.cast_signed();
            PEAK.fetch_max(live, Ordering::Relaxed);
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
        if source.as_usize().get() == COUNTED.load(Ordering::Relaxed) {
            LIVE.fetch_sub(size.cast_signed(), Ordering::Relaxed);
        }
    }
}

/// What `f` returns, and the most bytes the allocations it made held at
/// once. Calls on several threads at once would count into one another: the
/// tests that use it run one per process.
pub(crate) fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // A second tracker is refused only if one is set: this is the one.
        let _ = AllocationRegistry::set_global_tracker(Peaks);
        AllocationRegistry::enable_tracking();
    });
    let mut token = AllocationGroupToken::register().expect("an allocation group");
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    COUNTED.store(token.id().as_usize().get(), Ordering::Relaxed);
    let value = {
        let _counted = token.enter();
        f()
    };
    COUNTED.store(0, Ordering::Relaxed);
    (value, PEAK.load(Ordering::Relaxed).cast_unsigned())
}
