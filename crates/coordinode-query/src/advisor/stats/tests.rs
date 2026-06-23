use super::*;

/// Recording updates count, total, min, max correctly.
#[test]
fn basic_recording() {
    let h = LatencyHistogram::new();
    h.record(100);
    h.record(200);
    h.record(50);

    assert_eq!(h.count(), 3);
    assert_eq!(h.total_us(), 350);
    assert_eq!(h.min_us(), 50);
    assert_eq!(h.max_us(), 200);
}

/// Empty histogram returns 0 for percentiles.
#[test]
fn empty_percentile() {
    let h = LatencyHistogram::new();
    assert_eq!(h.percentile(0.50), 0);
    assert_eq!(h.percentile(0.99), 0);
}

/// P50 on uniform data falls in expected bucket range.
#[test]
fn p50_uniform() {
    let h = LatencyHistogram::new();
    // Record 100 values from 1 to 100 μs
    for i in 1..=100 {
        h.record(i);
    }

    let p50 = h.percentile(0.50);
    // Values 1-100: median is ~50. Bucket for 50 is [25, 50) → bound 50,
    // or [50, 100) → bound 100. The exact bucket depends on the boundary.
    assert!(
        (25..=100).contains(&p50),
        "p50={p50} should be in reasonable range"
    );
}

/// P99 captures the tail latency.
#[test]
fn p99_tail() {
    let h = LatencyHistogram::new();
    // 90 fast queries + 10 slow queries → p99 must be in slow bucket.
    // p99 of 100 samples = ceil(100 × 0.99) = 99th value.
    // cumulative after 90 fast = 90 < 99, so 99th is in slow bucket.
    for _ in 0..90 {
        h.record(10); // 10μs
    }
    for _ in 0..10 {
        h.record(1_000_000); // 1s
    }

    let p99 = h.percentile(0.99);
    assert!(p99 >= 500_000, "p99={p99} should capture the 1s tail");
}

/// Very small values (0-5μs) go into first bucket.
#[test]
fn smallest_bucket() {
    let h = LatencyHistogram::new();
    h.record(0);
    h.record(1);
    h.record(4);

    assert_eq!(h.count(), 3);
    let p50 = h.percentile(0.50);
    assert_eq!(p50, 5, "all values in [0,5) bucket → p50 = 5μs upper bound");
}

/// Very large values go into overflow bucket.
#[test]
fn overflow_bucket() {
    let h = LatencyHistogram::new();
    h.record(999_999_999);

    assert_eq!(h.count(), 1);
    assert_eq!(h.max_us(), 999_999_999);
}
