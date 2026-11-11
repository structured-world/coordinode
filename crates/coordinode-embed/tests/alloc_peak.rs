//! The bench measurer of a sample's peak: memory one sample keeps and a later
//! sample frees, on its own thread or another, does not lower the later
//! sample's peak.

#![allow(clippy::expect_used, clippy::unwrap_used)]

#[path = "../benches/support/alloc_peak.rs"]
mod alloc_peak;

#[global_allocator]
static ALLOCATOR: tracking_allocator::Allocator<std::alloc::System> =
    tracking_allocator::Allocator::system();

/// Bytes the second sample allocates and keeps until it ends.
const FIRST: usize = 4096;
const SECOND: usize = 65536;
/// What the first sample hands on to the second.
const RETAINED: usize = 1 << 20;

/// A sample keeping `FIRST` and then `SECOND`, freeing `retained` between
/// them through `free`.
fn second_sample(retained: Vec<u8>, free: impl FnOnce(Vec<u8>)) -> u64 {
    let ((), peak) = alloc_peak::peak_of(|| {
        let first = Vec::<u8>::with_capacity(FIRST);
        free(retained);
        let second = Vec::<u8>::with_capacity(SECOND);
        drop((first, second));
    });
    peak
}

/// One test per process: the tracker and the samples are process-wide.
#[test]
fn memory_kept_from_an_earlier_sample_does_not_lower_a_later_peak() {
    alloc_peak::install();

    let (retained, first_peak) = alloc_peak::peak_of(|| Vec::<u8>::with_capacity(RETAINED));
    assert_eq!(first_peak, RETAINED as u64);

    // Freed on the sample's own thread.
    let peak = second_sample(retained, drop);
    assert_eq!(peak, (FIRST + SECOND) as u64, "freed on its own thread");

    // Freed on another thread, joined within the sample: the thread's own
    // allocations made here are the sample's too, a few hundred bytes.
    let (retained, _) = alloc_peak::peak_of(|| Vec::<u8>::with_capacity(RETAINED));
    let peak = second_sample(retained, |retained| {
        std::thread::spawn(move || drop(retained)).join().unwrap();
    });
    let kept = (FIRST + SECOND) as u64;
    assert!(
        (kept..kept + 16 * 1024).contains(&peak),
        "freed on another thread: peak {peak}"
    );
}
