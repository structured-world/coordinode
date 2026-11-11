//! The most bytes a piece of work held at once, counted by the allocator:
//! each allocation is attributed to the allocation group that made it,
//! whichever thread frees it, so a test compares what work really held
//! against what it reserved. Allocations of other threads are not counted,
//! except by [`held_against`], which also counts the rayon pool's.
//!
//! The test binary's global allocator must be the tracking one:
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
//!     tracking_allocator::Allocator::system();
//! ```
//!
//! Without it nothing is recorded and every peak reads zero.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};

use tracking_allocator::{
    AllocationGroupId, AllocationGroupToken, AllocationRegistry, AllocationTracker,
};

/// The group being counted, `0` for none.
static COUNTED: AtomicUsize = AtomicUsize::new(0);
/// Bytes the counted group holds now.
static LIVE: AtomicIsize = AtomicIsize::new(0);
/// The most it held at once.
static PEAK: AtomicIsize = AtomicIsize::new(0);

/// Whether [`held_against`] is recording events.
static TRACING: AtomicBool = AtomicBool::new(false);
/// The group [`held_against`] traces, `0` for none.
static TRACED: AtomicUsize = AtomicUsize::new(0);
/// Events recorded so far, some of them perhaps past the log's end.
static LOGGED: AtomicUsize = AtomicUsize::new(0);
/// Room for this many events.
const LOG_EVENTS: usize = 1 << 21;
/// Four words per event: address, signed size, whether counted, and the
/// reservation read at the event. Allocated once, before any tracing.
static LOG: OnceLock<Box<[AtomicU64]>> = OnceLock::new();
/// What the work holds reserved now, read at every event.
type Reserved = Arc<dyn Fn() -> u64 + Send + Sync>;
static RESERVED: Mutex<Option<Reserved>> = Mutex::new(None);

struct Peaks;

impl Peaks {
    fn log(addr: usize, size: isize, counted: bool) {
        let Some(log) = LOG.get() else {
            return;
        };
        let at = LOGGED.fetch_add(1, Ordering::Relaxed);
        if at >= LOG_EVENTS {
            return;
        }
        // Reading the reservation allocates nothing; the lock is never held
        // across an allocation (see `held_against`).
        let reserved = RESERVED
            .lock()
            .ok()
            .and_then(|slot| slot.as_ref().map(|read| read()))
            .unwrap_or(0);
        let slot = &log[at * 4..at * 4 + 4];
        slot[0].store(addr as u64, Ordering::Relaxed);
        slot[1].store(size as u64, Ordering::Relaxed);
        slot[2].store(u64::from(counted), Ordering::Relaxed);
        slot[3].store(reserved, Ordering::Relaxed);
    }
}

impl AllocationTracker for Peaks {
    fn allocated(&self, addr: usize, size: usize, _wrapped: usize, group: AllocationGroupId) {
        let group = group.as_usize().get();
        if group == COUNTED.load(Ordering::Relaxed) {
            let live = LIVE.fetch_add(size.cast_signed(), Ordering::Relaxed) + size.cast_signed();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        if TRACING.load(Ordering::Relaxed) {
            let counted =
                group == TRACED.load(Ordering::Relaxed) || rayon::current_thread_index().is_some();
            Self::log(addr, size.cast_signed(), counted);
        }
    }

    fn deallocated(
        &self,
        addr: usize,
        size: usize,
        _wrapped: usize,
        source: AllocationGroupId,
        _current: AllocationGroupId,
    ) {
        if source.as_usize().get() == COUNTED.load(Ordering::Relaxed) {
            LIVE.fetch_sub(size.cast_signed(), Ordering::Relaxed);
        }
        if TRACING.load(Ordering::Relaxed) {
            Self::log(addr, -size.cast_signed(), false);
        }
    }
}

fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // A second tracker is refused only if one is set: this is the one.
        let _ = AllocationRegistry::set_global_tracker(Peaks);
        AllocationRegistry::enable_tracking();
    });
}

/// What `f` returns, and the most bytes the allocations it made held at
/// once. Calls on several threads at once would count into one another: a
/// test that uses it runs alone in its process, as nextest runs each.
///
/// # Panics
///
/// No allocation group can be registered.
pub fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let (value, peak, _) = peak_and_kept_of(f);
    (value, peak)
}

/// [`peak_of`], with the bytes `f`'s allocations still held once it
/// returned: what work handed on to a longer-lived owner (a store's
/// memtables, the trees it opened) rather than held only while it ran.
///
/// # Panics
///
/// No allocation group can be registered.
#[allow(clippy::expect_used)]
pub fn peak_and_kept_of<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    install();
    let mut token = AllocationGroupToken::register().expect("an allocation group");
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    COUNTED.store(token.id().as_usize().get(), Ordering::Relaxed);
    let value = {
        let _counted = token.enter();
        f()
    };
    COUNTED.store(0, Ordering::Relaxed);
    let kept = LIVE.load(Ordering::Relaxed).max(0).cast_unsigned();
    (value, PEAK.load(Ordering::Relaxed).cast_unsigned(), kept)
}

/// What work held beyond its reservation over its whole run, by
/// [`held_against`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Held {
    /// The most bytes held at once by allocations the work made and freed
    /// before it returned.
    pub transient: usize,
    /// Bytes of allocations the work made and still held when it returned:
    /// what it handed to a longer-lived owner.
    pub kept: usize,
    /// The event at which the transient bytes stood furthest above the
    /// reservation read at the same event, if they ever did: `(event, held,
    /// reserved)`.
    pub over: Option<(usize, usize, u64)>,
}

/// What `f` returns, and what its allocations held against `reserved`, read
/// at each allocation and free while it runs. An allocation counts when
/// the calling thread makes it, or any thread of the rayon pool does. The
/// bytes compared with the reservation are those of the allocations freed
/// before `f` returned; an allocation still held then (handed to a store)
/// is left out of every instant, never subtracted as a total, so memory
/// kept late cannot cover a transient held early.
///
/// Like [`peak_of`], a test using it runs alone in its process.
///
/// # Panics
///
/// No allocation group can be registered, or `f` makes more allocations
/// and frees than the trace has room for.
#[allow(clippy::expect_used, clippy::panic)]
pub fn held_against<T>(reserved: Reserved, f: impl FnOnce() -> T) -> (T, Held) {
    install();
    let log = LOG.get_or_init(|| (0..LOG_EVENTS * 4).map(|_| AtomicU64::new(0)).collect());
    let mut token = AllocationGroupToken::register().expect("an allocation group");
    if let Ok(mut slot) = RESERVED.lock() {
        *slot = Some(reserved);
    }
    LOGGED.store(0, Ordering::Relaxed);
    TRACED.store(token.id().as_usize().get(), Ordering::Relaxed);
    TRACING.store(true, Ordering::SeqCst);
    let value = {
        let _traced = token.enter();
        f()
    };
    TRACING.store(false, Ordering::SeqCst);
    TRACED.store(0, Ordering::Relaxed);
    // Taken out under the lock and dropped after it: dropping it frees.
    let taken = RESERVED.lock().ok().and_then(|mut slot| slot.take());
    drop(taken);

    let logged = LOGGED.load(Ordering::Relaxed);
    assert!(
        logged <= LOG_EVENTS,
        "the trace holds {LOG_EVENTS} events, the work made {logged}"
    );
    let events: Vec<(usize, isize, bool, u64)> = (0..logged)
        .map(|at| {
            let slot = &log[at * 4..at * 4 + 4];
            (
                slot[0].load(Ordering::Relaxed) as usize,
                slot[1].load(Ordering::Relaxed) as i64 as isize,
                slot[2].load(Ordering::Relaxed) == 1,
                slot[3].load(Ordering::Relaxed),
            )
        })
        .collect();

    // Which counted allocations were still live when `f` returned.
    let mut live: HashMap<usize, (usize, usize)> = HashMap::new();
    for (at, &(addr, size, counted, _)) in events.iter().enumerate() {
        if size > 0 && counted {
            live.insert(addr, (at, size.cast_unsigned()));
        } else if size < 0 {
            live.remove(&addr);
        }
    }
    let kept_at: std::collections::HashSet<usize> = live.values().map(|&(at, _)| at).collect();
    let kept = live.values().map(|&(_, size)| size).sum();

    let mut transient_live: HashMap<usize, usize> = HashMap::new();
    let mut held = 0usize;
    let mut transient = 0usize;
    let mut over: Option<(usize, usize, u64)> = None;
    for (at, &(addr, size, counted, reserved)) in events.iter().enumerate() {
        if size > 0 && counted && !kept_at.contains(&at) {
            transient_live.insert(addr, size.cast_unsigned());
            held += size.cast_unsigned();
        } else if size < 0
            && let Some(freed) = transient_live.remove(&addr)
        {
            held -= freed;
        }
        transient = transient.max(held);
        // Only an instant held above its reservation is a finding.
        if let Some(excess) = (held as u64).checked_sub(reserved).filter(|&e| e > 0)
            && over.is_none_or(|(_, h, r)| excess > (h as u64) - r)
        {
            over = Some((at, held, reserved));
        }
    }
    (
        value,
        Held {
            transient,
            kept,
            over,
        },
    )
}

#[cfg(test)]
mod tests;
