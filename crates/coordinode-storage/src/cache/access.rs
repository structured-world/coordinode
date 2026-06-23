//! Per-key access tracking for cache-aware data placement.
//!
//! Tracks access count and last-access timestamp per cache key.
//! Used for LRU/LFU eviction decisions in the SSD cache and for
//! per-shard heat map (CE Level 1) to identify hot/cold data.
//!
//! Storage overhead: 8 bytes per tracked key (4B count + 4B timestamp).
//! 1M keys = 8MB tracking data — fits comfortably in memory.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::partition::Partition;

/// Composite cache key: hash of (partition, key bytes).
pub type CacheKey = u64;

/// Per-key access statistics.
#[derive(Debug, Clone, Copy)]
pub struct AccessStats {
    /// Total access count since tracking started.
    pub count: u32,
    /// Last access timestamp (unix seconds, truncated to u32 — valid until 2106).
    pub last_access: u32,
}

impl AccessStats {
    fn new(now: u32) -> Self {
        Self {
            count: 1,
            last_access: now,
        }
    }

    fn record(&mut self, now: u32) {
        self.count = self.count.saturating_add(1);
        self.last_access = now;
    }
}

/// Tracks access patterns for cache eviction and heat map decisions.
///
/// Thread-safe via interior `RwLock`. Read path (lookup) acquires shared lock;
/// write path (record) acquires exclusive lock.
///
/// # Cluster-ready
/// Per-node local tracking. Each CE replica maintains its own access stats.
/// No cross-node synchronization needed — access patterns are node-local.
pub struct AccessTracker {
    stats: RwLock<HashMap<CacheKey, AccessStats>>,
}

impl AccessTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self {
            stats: RwLock::new(HashMap::new()),
        }
    }

    /// Record an access to the given key.
    pub fn record(&self, part: Partition, key: &[u8]) {
        let cache_key = compute_cache_key(part, key);
        let now = unix_now();

        // Fast path: try read lock first to check if entry exists
        // (most accesses are to already-tracked keys)
        {
            let stats = self.stats.read().unwrap_or_else(|e| e.into_inner());
            if stats.contains_key(&cache_key) {
                drop(stats);
                let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
                if let Some(entry) = stats.get_mut(&cache_key) {
                    entry.record(now);
                    return;
                }
            }
        }

        // Slow path: new key, need write lock
        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
        stats
            .entry(cache_key)
            .and_modify(|e| e.record(now))
            .or_insert_with(|| AccessStats::new(now));
    }

    /// Get access stats for a key, if tracked.
    pub fn get(&self, part: Partition, key: &[u8]) -> Option<AccessStats> {
        let cache_key = compute_cache_key(part, key);
        let stats = self.stats.read().unwrap_or_else(|e| e.into_inner());
        stats.get(&cache_key).copied()
    }

    /// Get the N coldest keys (lowest access count, then oldest last_access).
    /// Used by the eviction policy to select entries to remove.
    pub fn coldest(&self, n: usize) -> Vec<CacheKey> {
        let stats = self.stats.read().unwrap_or_else(|e| e.into_inner());
        let mut entries: Vec<(CacheKey, AccessStats)> =
            stats.iter().map(|(&k, &v)| (k, v)).collect();

        // Sort by access count ascending, then by last_access ascending (oldest first)
        entries.sort_by(|a, b| {
            a.1.count
                .cmp(&b.1.count)
                .then(a.1.last_access.cmp(&b.1.last_access))
        });

        entries.into_iter().take(n).map(|(k, _)| k).collect()
    }

    /// Remove tracking for a key (e.g., after cache eviction).
    pub fn remove(&self, cache_key: CacheKey) {
        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
        stats.remove(&cache_key);
    }

    /// Number of tracked keys.
    pub fn len(&self) -> usize {
        let stats = self.stats.read().unwrap_or_else(|e| e.into_inner());
        stats.len()
    }

    /// Whether the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Aggregate statistics: total accesses, unique keys, hottest key.
    pub fn aggregate(&self) -> AggregateStats {
        let stats = self.stats.read().unwrap_or_else(|e| e.into_inner());
        let mut total_accesses: u64 = 0;
        let mut max_count: u32 = 0;
        let mut hottest_key: Option<CacheKey> = None;

        for (&key, &entry) in stats.iter() {
            total_accesses += u64::from(entry.count);
            if entry.count > max_count {
                max_count = entry.count;
                hottest_key = Some(key);
            }
        }

        AggregateStats {
            unique_keys: stats.len(),
            total_accesses,
            hottest_key,
            hottest_count: max_count,
        }
    }
}

impl Default for AccessTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated access statistics for observability.
#[derive(Debug, Clone)]
pub struct AggregateStats {
    pub unique_keys: usize,
    pub total_accesses: u64,
    pub hottest_key: Option<CacheKey>,
    pub hottest_count: u32,
}

/// Compute a cache key from partition + key bytes.
/// Uses xxh3 for fast, high-quality hashing.
pub fn compute_cache_key(part: Partition, key: &[u8]) -> CacheKey {
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    hasher.update(&[part as u8]);
    hasher.update(key);
    hasher.digest()
}

/// Current unix timestamp in seconds (truncated to u32).
fn unix_now() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
