//! MVCC snapshot — a consistent read view at a specific timestamp.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::txn::timestamp::Timestamp;

/// Default retention window for MVCC versions (7 days).
pub const DEFAULT_RETENTION: Duration = Duration::from_secs(7 * 24 * 3600);

/// A consistent read snapshot at a specific timestamp.
///
/// All reads through this snapshot see exactly the versions that were
/// committed at or before `read_ts`. Uncommitted or later versions
/// are invisible.
#[derive(Debug, Clone)]
pub struct Snapshot {
    read_ts: Timestamp,
}

impl Snapshot {
    /// Create a snapshot at the given timestamp.
    pub fn at(read_ts: Timestamp) -> Self {
        Self { read_ts }
    }

    /// The timestamp this snapshot reads at.
    pub fn read_ts(&self) -> Timestamp {
        self.read_ts
    }
}

/// Retention policy for MVCC versions.
///
/// Controls when old versions become eligible for garbage collection
/// during compaction.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Duration to keep old versions for time-travel queries.
    window: Duration,
}

impl RetentionPolicy {
    /// Create a retention policy with the given window duration.
    pub fn new(window: Duration) -> Self {
        Self { window }
    }

    /// Check if a version with the given `commit_ts` (in microseconds
    /// since epoch) is eligible for garbage collection.
    pub fn is_expired(&self, commit_ts: Timestamp) -> bool {
        let now_micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64;

        let cutoff = now_micros.saturating_sub(self.window.as_micros() as u64);
        commit_ts.as_raw() < cutoff
    }

    /// Get the retention window duration.
    pub fn window(&self) -> Duration {
        self.window
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self::new(DEFAULT_RETENTION)
    }
}

#[cfg(test)]
mod tests;
