//! Query fingerprint registry: tracks query execution statistics.
//!
//! Thread-safe; a record takes the registry's mutex briefly. Uses LFU
//! eviction when capacity is reached (CE: 1,000 entries).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use super::source::{SourceContext, SourceLocationSnapshot, SourceTracker};
use super::stats::LatencyHistogram;
use crate::planner::logical::LogicalPlan;

/// Default capacity for CE tier: 1,000 unique query fingerprints.
const DEFAULT_CAPACITY: usize = 1_000;

/// Snapshot of a single fingerprint's statistics.
/// Returned by query methods — owned data, safe to pass across threads.
#[derive(Debug, Clone)]
pub struct QueryStats {
    pub fingerprint: u64,
    pub canonical_query: String,
    pub count: u64,
    pub total_time_us: u64,
    pub min_time_us: u64,
    pub max_time_us: u64,
    pub p50_time_us: u64,
    pub p99_time_us: u64,
    pub last_seen: i64,
    /// EXPLAIN text of the plan the statement last ran with.
    pub last_plan: Option<String>,
    /// Top call sites by frequency (when debug source tracking is enabled).
    pub sources: Vec<SourceLocationSnapshot>,
}

/// A single entry in the fingerprint registry.
/// All fields are atomic or behind per-entry locks for thread safety.
struct RegistryEntry {
    fingerprint: u64,
    canonical_query: String,
    histogram: LatencyHistogram,
    last_seen: AtomicI64,
    /// The plan the statement last ran with, and its EXPLAIN text, rendered
    /// only when the plan changed.
    last_plan: Option<(LogicalPlan, String)>,
    /// Top-5 call sites by frequency. Updated under the registry mutex.
    source_tracker: SourceTracker,
}

impl RegistryEntry {
    fn new(fingerprint: u64, canonical_query: String) -> Self {
        Self {
            fingerprint,
            canonical_query,
            histogram: LatencyHistogram::new(),
            last_seen: AtomicI64::new(now_micros()),
            last_plan: None,
            source_tracker: SourceTracker::new(),
        }
    }

    /// Keep `plan` as the one the statement runs with, rendering its text
    /// only when it differs from the plan kept.
    fn note_plan(&mut self, plan: &LogicalPlan) {
        if self
            .last_plan
            .as_ref()
            .is_some_and(|(kept, _)| kept == plan)
        {
            return;
        }
        self.last_plan = Some((plan.clone(), plan.explain()));
    }

    /// Record a query execution. Lock-free (latency tracking only).
    fn record(&self, duration_us: u64) {
        self.histogram.record(duration_us);
        self.last_seen.store(now_micros(), Ordering::Relaxed);
    }

    /// Record a query execution with source context. Requires mutable access
    /// to the source tracker (called under registry mutex).
    fn record_with_source(&mut self, duration_us: u64, source: &SourceContext) {
        self.histogram.record(duration_us);
        self.last_seen.store(now_micros(), Ordering::Relaxed);
        self.source_tracker.record(source);
    }

    /// Take a snapshot of current statistics.
    fn snapshot(&self) -> QueryStats {
        QueryStats {
            fingerprint: self.fingerprint,
            canonical_query: self.canonical_query.clone(),
            count: self.histogram.count(),
            total_time_us: self.histogram.total_us(),
            min_time_us: self.histogram.min_us(),
            max_time_us: self.histogram.max_us(),
            p50_time_us: self.histogram.percentile(0.50),
            p99_time_us: self.histogram.percentile(0.99),
            last_seen: self.last_seen.load(Ordering::Relaxed),
            last_plan: self.last_plan.as_ref().map(|(_, text)| text.clone()),
            sources: self.source_tracker.snapshot(),
        }
    }
}

/// Thread-safe query fingerprint registry.
///
/// Recording stats for an **existing** fingerprint is lock-free (atomic operations only).
/// Only the cold path (registering a new fingerprint or evicting) takes a mutex.
///
/// Capacity is bounded; when full, the least-frequently-used entry is evicted.
pub struct QueryRegistry {
    /// Maps fingerprint → entry index in `entries`.
    ///
    /// Protected by the outer mutex for structural changes (insert/evict).
    /// Read-only lookups for recording go through the entries directly.
    inner: Mutex<RegistryInner>,

    /// Total queries recorded across all fingerprints.
    queries_recorded: AtomicU64,
}

struct RegistryInner {
    entries: HashMap<u64, RegistryEntry>,
    capacity: usize,
}

impl QueryRegistry {
    /// Create a new registry with default CE capacity (1,000 fingerprints).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a new registry with custom capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(RegistryInner {
                entries: HashMap::with_capacity(capacity),
                capacity,
            }),
            queries_recorded: AtomicU64::new(0),
        }
    }

    /// Record a query execution with its fingerprint and duration.
    ///
    /// If new, the fingerprint is registered (evicting the least-used entry
    /// at capacity). `canonical_query` is only used on first registration.
    pub fn record(&self, fingerprint: u64, canonical_query: &str, duration_us: u64) {
        self.record_inner(fingerprint, canonical_query, duration_us, None, None);
    }

    /// Record a query execution with source location context.
    ///
    /// Same as `record()` but also tracks the source call site.
    pub fn record_with_source(
        &self,
        fingerprint: u64,
        canonical_query: &str,
        duration_us: u64,
        source: &SourceContext,
    ) {
        self.record_inner(
            fingerprint,
            canonical_query,
            duration_us,
            Some(source),
            None,
        );
    }

    /// Record a statement's run with the plan it ran with, for
    /// `db.advisor.queryStats()` and `db.advisor.slowQueries()`, and its call
    /// site when the client named one. The plan's EXPLAIN text is rendered
    /// only when it differs from the plan kept, so a statement run again with
    /// the same plan pays one comparison.
    pub fn record_run(
        &self,
        fingerprint: u64,
        canonical_query: &str,
        duration_us: u64,
        plan: &LogicalPlan,
        source: Option<&SourceContext>,
    ) {
        self.record_inner(
            fingerprint,
            canonical_query,
            duration_us,
            source,
            Some(plan),
        );
    }

    fn record_inner(
        &self,
        fingerprint: u64,
        canonical_query: &str,
        duration_us: u64,
        source: Option<&SourceContext>,
        plan: Option<&LogicalPlan>,
    ) {
        self.queries_recorded.fetch_add(1, Ordering::Relaxed);
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let entry = Self::entry(&mut inner, fingerprint, canonical_query);
        if let Some(plan) = plan {
            entry.note_plan(plan);
        }
        match source {
            Some(src) => entry.record_with_source(duration_us, src),
            None => entry.record(duration_us),
        }
    }

    /// The entry of `fingerprint`, registered first when new, evicting the
    /// least-used one at capacity.
    fn entry<'i>(
        inner: &'i mut RegistryInner,
        fingerprint: u64,
        canonical_query: &str,
    ) -> &'i mut RegistryEntry {
        if !inner.entries.contains_key(&fingerprint) && inner.entries.len() >= inner.capacity {
            Self::evict_lfu(&mut inner.entries);
        }
        inner
            .entries
            .entry(fingerprint)
            .or_insert_with(|| RegistryEntry::new(fingerprint, canonical_query.to_string()))
    }

    /// Evict the least-frequently-used entry.
    fn evict_lfu(entries: &mut HashMap<u64, RegistryEntry>) {
        if entries.is_empty() {
            return;
        }

        let lfu_fp = entries
            .iter()
            .min_by_key(|(_, e)| e.histogram.count())
            .map(|(&fp, _)| fp);

        if let Some(fp) = lfu_fp {
            entries.remove(&fp);
        }
    }

    /// Get statistics for a specific fingerprint.
    pub fn get(&self, fingerprint: u64) -> Option<QueryStats> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.entries.get(&fingerprint).map(|e| e.snapshot())
    }

    /// Get top N fingerprints by execution count.
    pub fn top_by_count(&self, n: usize) -> Vec<QueryStats> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut stats: Vec<QueryStats> = inner.entries.values().map(|e| e.snapshot()).collect();
        stats.sort_by_key(|s| std::cmp::Reverse(s.count));
        stats.truncate(n);
        stats
    }

    /// Get top N fingerprints by p99 latency, filtered by minimum time.
    pub fn top_by_latency(&self, n: usize, min_p99_us: u64) -> Vec<QueryStats> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut stats: Vec<QueryStats> = inner
            .entries
            .values()
            .map(|e| e.snapshot())
            .filter(|s| s.p99_time_us >= min_p99_us)
            .collect();
        stats.sort_by_key(|s| std::cmp::Reverse(s.p99_time_us));
        stats.truncate(n);
        stats
    }

    /// Get top N fingerprints by impact score (count × p99).
    pub fn top_by_impact(&self, n: usize) -> Vec<QueryStats> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut stats: Vec<QueryStats> = inner.entries.values().map(|e| e.snapshot()).collect();
        stats.sort_by(|a, b| {
            let impact_a = a.count.saturating_mul(a.p99_time_us);
            let impact_b = b.count.saturating_mul(b.p99_time_us);
            impact_b.cmp(&impact_a)
        });
        stats.truncate(n);
        stats
    }

    /// Number of tracked fingerprints.
    pub fn fingerprint_count(&self) -> usize {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.entries.len()
    }

    /// Total queries recorded across all fingerprints.
    pub fn total_queries_recorded(&self) -> u64 {
        self.queries_recorded.load(Ordering::Relaxed)
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.entries.clear();
        self.queries_recorded.store(0, Ordering::Relaxed);
    }
}

impl Default for QueryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Current time in microseconds since UNIX epoch.
fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
