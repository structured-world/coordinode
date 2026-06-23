//! Streaming latency statistics with zero-allocation hot path.
//!
//! Uses a fixed-bucket log-linear histogram for percentile estimation
//! and atomic counters for count/total/min/max. All operations are lock-free
//! on the recording path.

use std::sync::atomic::{AtomicU64, Ordering};

/// Number of histogram buckets.
/// Covers 1μs to >10s in log-linear steps.
const BUCKET_COUNT: usize = 24;

/// Bucket upper bounds in microseconds.
/// Each bucket covers a range: [previous_bound, this_bound).
/// Bucket 0: [0, 5), Bucket 1: [5, 10), ..., Bucket 23: [5_000_000, ∞).
const BUCKET_BOUNDS: [u64; BUCKET_COUNT] = [
    5,           // 0: 0-5μs
    10,          // 1: 5-10μs
    25,          // 2: 10-25μs
    50,          // 3: 25-50μs
    100,         // 4: 50-100μs
    250,         // 5: 100-250μs
    500,         // 6: 250-500μs
    1_000,       // 7: 500μs-1ms
    2_500,       // 8: 1-2.5ms
    5_000,       // 9: 2.5-5ms
    10_000,      // 10: 5-10ms
    25_000,      // 11: 10-25ms
    50_000,      // 12: 25-50ms
    100_000,     // 13: 50-100ms
    250_000,     // 14: 100-250ms
    500_000,     // 15: 250-500ms
    1_000_000,   // 16: 500ms-1s
    2_500_000,   // 17: 1-2.5s
    5_000_000,   // 18: 2.5-5s
    10_000_000,  // 19: 5-10s
    30_000_000,  // 20: 10-30s
    60_000_000,  // 21: 30-60s
    300_000_000, // 22: 1-5min
    u64::MAX,    // 23: 5min+ (overflow bucket)
];

/// Lock-free latency histogram for streaming percentile estimation.
///
/// Records latencies into fixed log-linear buckets using atomic counters.
/// Percentile queries (p50, p99) are computed from the bucket distribution.
///
/// Accuracy: within one bucket width (e.g., a p99 of 450μs reported as
/// "between 250-500μs"). This is sufficient for query advisor purposes
/// where order-of-magnitude matters more than exact microseconds.
pub(crate) struct LatencyHistogram {
    buckets: [AtomicU64; BUCKET_COUNT],
    count: AtomicU64,
    total_us: AtomicU64,
    min_us: AtomicU64,
    max_us: AtomicU64,
}

impl LatencyHistogram {
    /// Create a new empty histogram.
    pub(crate) fn new() -> Self {
        Self {
            // SAFETY: AtomicU64 is the same size/alignment as u64,
            // and 0 is a valid value for both. We use a const array init.
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            total_us: AtomicU64::new(0),
            min_us: AtomicU64::new(u64::MAX),
            max_us: AtomicU64::new(0),
        }
    }

    /// Record a latency measurement. Lock-free, zero-allocation.
    pub(crate) fn record(&self, duration_us: u64) {
        // Find the appropriate bucket via linear scan.
        // With 24 buckets this is faster than binary search due to branch prediction.
        let bucket_idx = BUCKET_BOUNDS
            .iter()
            .position(|&bound| duration_us < bound)
            .unwrap_or(BUCKET_COUNT - 1);

        self.buckets[bucket_idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(duration_us, Ordering::Relaxed);

        // Update min (CAS loop)
        let mut current_min = self.min_us.load(Ordering::Relaxed);
        while duration_us < current_min {
            match self.min_us.compare_exchange_weak(
                current_min,
                duration_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }

        // Update max (CAS loop)
        let mut current_max = self.max_us.load(Ordering::Relaxed);
        while duration_us > current_max {
            match self.max_us.compare_exchange_weak(
                current_max,
                duration_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    /// Estimate a percentile value from the histogram.
    ///
    /// `p` is in range [0.0, 1.0] (e.g., 0.50 for p50, 0.99 for p99).
    /// Returns the upper bound of the bucket containing the percentile,
    /// or 0 if no data has been recorded.
    pub(crate) fn percentile(&self, p: f64) -> u64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 {
            return 0;
        }

        let target = ((total as f64) * p).ceil() as u64;
        let mut cumulative: u64 = 0;

        for (i, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            if cumulative >= target {
                return BUCKET_BOUNDS[i];
            }
        }

        // Should not reach here if data exists, but return max bound as fallback
        BUCKET_BOUNDS[BUCKET_COUNT - 1]
    }

    /// Get total number of recorded measurements.
    pub(crate) fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get total time across all measurements (microseconds).
    pub(crate) fn total_us(&self) -> u64 {
        self.total_us.load(Ordering::Relaxed)
    }

    /// Get minimum recorded latency (microseconds), or `u64::MAX` if empty.
    pub(crate) fn min_us(&self) -> u64 {
        self.min_us.load(Ordering::Relaxed)
    }

    /// Get maximum recorded latency (microseconds), or 0 if empty.
    pub(crate) fn max_us(&self) -> u64 {
        self.max_us.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
