use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

const MIB: usize = 1 << 20;

/// An unreserved transient held early, then a much larger allocation kept
/// past the end: comparing the whole run's peak with the reservation plus
/// what is kept at the end lets the late memory cover the early transient;
/// the trace sees the transient over the reservation at its own instant.
#[test]
fn an_early_unreserved_transient_is_caught_behind_a_larger_kept_tail() {
    let work = || {
        let early = vec![1u8; MIB];
        std::hint::black_box(&early);
        drop(early);
        let kept = vec![2u8; 8 * MIB];
        std::hint::black_box(&kept);
        kept
    };
    let (kept, peak, kept_bytes) = peak_and_kept_of(work);
    drop(kept);
    assert!(
        peak <= kept_bytes,
        "the whole-run comparison is fooled: peak {peak} within 0 reserved + {kept_bytes} kept"
    );

    let (kept, held) = held_against(Arc::new(|| 0), work);
    drop(kept);
    assert_eq!(held.kept, 8 * MIB);
    assert_eq!(held.transient, MIB);
    let (_, over, reserved) = held
        .over
        .expect("the early transient is over the reservation");
    assert_eq!((over, reserved), (MIB, 0));
}

/// A transient the rayon pool holds for the work counts like the calling
/// thread's own.
#[test]
fn a_transient_on_the_rayon_pool_is_counted() {
    let ((), held) = held_against(Arc::new(|| 0), || {
        rayon::join(
            || {
                let on_pool = vec![3u8; 2 * MIB];
                std::hint::black_box(&on_pool);
            },
            || (),
        );
        // Either half may run on the calling thread; both on the pool here.
        rayon::scope(|scope| {
            scope.spawn(|_| {
                let on_pool = vec![4u8; 2 * MIB];
                std::hint::black_box(&on_pool);
            });
        });
    });
    assert!(held.transient >= 2 * MIB, "{held:?}");
    assert!(held.over.is_some(), "{held:?}");
}

/// A transient reserved before it is allocated and released after it is
/// freed never stands over the reservation.
#[test]
fn a_reserved_transient_is_within_its_reservation() {
    let reserved = Arc::new(AtomicU64::new(0));
    let read = Arc::clone(&reserved);
    let ((), held) = held_against(Arc::new(move || read.load(Ordering::Relaxed)), || {
        reserved.store(2 * MIB as u64, Ordering::Relaxed);
        let within = vec![5u8; MIB];
        std::hint::black_box(&within);
        drop(within);
        reserved.store(0, Ordering::Relaxed);
    });
    assert_eq!(held.over, None, "{held:?}");
    assert!(held.transient >= MIB, "{held:?}");
}
