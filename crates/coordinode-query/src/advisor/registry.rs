//! Query fingerprint registry: tracks query execution statistics.
//!
//! Thread-safe, lock-free on the hot path (recording stats for known fingerprints).
//! Uses LFU eviction when capacity is reached (CE: 1,000 entries).

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use super::source::{SourceContext, SourceLocationSnapshot, SourceTracker};
use super::stats::LatencyHistogram;

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
    /// Last execution plan (EXPLAIN output). Updated on each execution.
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
    /// Last execution plan (EXPLAIN output). Updated on each execution
    /// when a plan string is provided.
    last_plan: Option<String>,
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

    /// Record with plan string. Requires mutable access (called under mutex).
    fn record_with_plan(&mut self, duration_us: u64, plan: String) {
        self.histogram.record(duration_us);
        self.last_seen.store(now_micros(), Ordering::Relaxed);
        self.last_plan = Some(plan);
    }

    /// Record with both source and plan. Requires mutable access.
    fn record_with_source_and_plan(
        &mut self,
        duration_us: u64,
        source: &SourceContext,
        plan: String,
    ) {
        self.histogram.record(duration_us);
        self.last_seen.store(now_micros(), Ordering::Relaxed);
        self.last_plan = Some(plan);
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
            last_plan: self.last_plan.clone(),
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
    /// If the fingerprint is already known, updates its stats (lock-free fast path
    /// when no source context or plan is provided).
    /// If new, registers it (may evict LFU entry if at capacity).
    ///
    /// `canonical_query` is only used on first registration; subsequent calls
    /// with the same fingerprint ignore it.
    pub fn record(&self, fingerprint: u64, canonical_query: &str, duration_us: u64) {
        self.record_inner(fingerprint, canonical_query, duration_us, None, None);
    }

    /// Record a query execution with source location context.
    ///
    /// Same as `record()` but also tracks the source call site.
    /// Source tracking requires mutable access, so this always takes the mutex.
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

    /// Record a query execution with plan and optional source context.
    ///
    /// Stores the EXPLAIN plan string in the registry entry so it can be
    /// returned by `db.advisor.queryStats()` and `db.advisor.slowQueries()`.
    pub fn record_with_plan(
        &self,
        fingerprint: u64,
        canonical_query: &str,
        duration_us: u64,
        plan: String,
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
        plan: Option<String>,
    ) {
        self.queries_recorded.fetch_add(1, Ordering::Relaxed);

        if source.is_none() && plan.is_none() {
            // Fast path: no source tracking or plan, try lock-free read
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = inner.entries.get(&fingerprint) {
                entry.record(duration_us);
                return;
            }
            drop(inner);
        }

        // Slow path: need mutable access (new entry, source tracking, or plan update)
        self.record_slow(fingerprint, canonical_query, duration_us, source, plan);
    }

    /// Slow path: registers new fingerprint or updates with source/plan.
    fn record_slow(
        &self,
        fingerprint: u64,
        canonical_query: &str,
        duration_us: u64,
        source: Option<&SourceContext>,
        plan: Option<String>,
    ) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(entry) = inner.entries.get_mut(&fingerprint) {
            match (source, plan) {
                (Some(src), Some(p)) => entry.record_with_source_and_plan(duration_us, src, p),
                (Some(src), None) => entry.record_with_source(duration_us, src),
                (None, Some(p)) => entry.record_with_plan(duration_us, p),
                (None, None) => entry.record(duration_us),
            }
            return;
        }

        // Evict LFU if at capacity
        if inner.entries.len() >= inner.capacity {
            Self::evict_lfu(&mut inner.entries);
        }

        let mut entry = RegistryEntry::new(fingerprint, canonical_query.to_string());
        match (source, plan) {
            (Some(src), Some(p)) => entry.record_with_source_and_plan(duration_us, src, p),
            (Some(src), None) => entry.record_with_source(duration_us, src),
            (None, Some(p)) => entry.record_with_plan(duration_us, p),
            (None, None) => entry.record(duration_us),
        }
        inner.entries.insert(fingerprint, entry);
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
mod tests {
    use super::*;

    /// Recording a query creates an entry and tracks stats.
    #[test]
    fn record_creates_entry() {
        let reg = QueryRegistry::new();
        reg.record(0xABC, "MATCH (n:User) RETURN n", 100);

        let stats = reg.get(0xABC).expect("entry should exist");
        assert_eq!(stats.fingerprint, 0xABC);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_time_us, 100);
        assert_eq!(stats.canonical_query, "MATCH (n:User) RETURN n");
    }

    /// Multiple recordings for the same fingerprint accumulate stats.
    #[test]
    fn multiple_recordings_accumulate() {
        let reg = QueryRegistry::new();
        reg.record(0xABC, "query", 100);
        reg.record(0xABC, "query", 200);
        reg.record(0xABC, "query", 50);

        let stats = reg.get(0xABC).expect("entry should exist");
        assert_eq!(stats.count, 3);
        assert_eq!(stats.total_time_us, 350);
        assert_eq!(stats.min_time_us, 50);
        assert_eq!(stats.max_time_us, 200);
    }

    /// LFU eviction removes the least-used entry when at capacity.
    #[test]
    fn lfu_eviction() {
        let reg = QueryRegistry::with_capacity(3);

        // Record 3 fingerprints with different frequencies
        reg.record(1, "q1", 10);
        reg.record(2, "q2", 10);
        reg.record(2, "q2", 10); // q2 has count=2
        reg.record(3, "q3", 10);
        reg.record(3, "q3", 10);
        reg.record(3, "q3", 10); // q3 has count=3

        // Now add a 4th — should evict q1 (count=1, lowest)
        reg.record(4, "q4", 10);

        assert!(reg.get(1).is_none(), "q1 should be evicted (LFU)");
        assert!(reg.get(2).is_some(), "q2 should still exist");
        assert!(reg.get(3).is_some(), "q3 should still exist");
        assert!(reg.get(4).is_some(), "q4 should be newly added");
    }

    /// top_by_count returns entries sorted by count descending.
    #[test]
    fn top_by_count_sorted() {
        let reg = QueryRegistry::new();

        reg.record(1, "q1", 10);
        for _ in 0..5 {
            reg.record(2, "q2", 10);
        }
        for _ in 0..3 {
            reg.record(3, "q3", 10);
        }

        let top = reg.top_by_count(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].fingerprint, 2, "highest count first");
        assert_eq!(top[0].count, 5);
        assert_eq!(top[1].fingerprint, 3, "second highest count");
        assert_eq!(top[1].count, 3);
    }

    /// top_by_latency filters by min_p99 and sorts descending.
    #[test]
    fn top_by_latency_filtered() {
        let reg = QueryRegistry::new();

        reg.record(1, "fast", 10);
        reg.record(2, "slow", 500_000); // 500ms

        let top = reg.top_by_latency(10, 1_000); // min 1ms p99
        assert_eq!(top.len(), 1, "only the slow query above threshold");
        assert_eq!(top[0].fingerprint, 2);
    }

    /// top_by_impact orders by count × p99.
    #[test]
    fn top_by_impact_ranking() {
        let reg = QueryRegistry::new();

        // High frequency, low latency
        for _ in 0..1000 {
            reg.record(1, "frequent", 10);
        }
        // Low frequency, high latency
        for _ in 0..5 {
            reg.record(2, "slow", 1_000_000);
        }

        let top = reg.top_by_impact(2);
        assert_eq!(top.len(), 2);
        // Impact: 1000 × 25 (bucket bound for 10μs) vs 5 × 2_500_000 (bucket for 1s)
        // 5 × 2_500_000 = 12_500_000 > 1000 × 25 = 25_000
        assert_eq!(
            top[0].fingerprint, 2,
            "slow×5 has higher impact than fast×1000"
        );
    }

    /// reset clears all entries and counters.
    #[test]
    fn reset_clears_everything() {
        let reg = QueryRegistry::new();
        reg.record(1, "q1", 10);
        reg.record(2, "q2", 20);

        reg.reset();

        assert_eq!(reg.fingerprint_count(), 0);
        assert_eq!(reg.total_queries_recorded(), 0);
        assert!(reg.get(1).is_none());
    }

    /// total_queries_recorded counts all recordings across all fingerprints.
    #[test]
    fn total_queries_counter() {
        let reg = QueryRegistry::new();
        reg.record(1, "q1", 10);
        reg.record(2, "q2", 10);
        reg.record(1, "q1", 20);

        assert_eq!(reg.total_queries_recorded(), 3);
    }

    /// Concurrent recording from multiple threads doesn't panic or corrupt data.
    #[test]
    fn concurrent_recording() {
        use std::sync::Arc;
        use std::thread;

        let reg = Arc::new(QueryRegistry::new());
        let mut handles = vec![];

        for thread_id in 0u64..4 {
            let reg = Arc::clone(&reg);
            handles.push(thread::spawn(move || {
                for i in 0u64..250 {
                    let fp = (thread_id * 250 + i) % 50; // 50 unique fingerprints
                    reg.record(fp, &format!("query_{fp}"), i + 1);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("thread should not panic");
        }

        // 4 threads × 250 recordings = 1000 total
        assert_eq!(reg.total_queries_recorded(), 1000);
        assert!(
            reg.fingerprint_count() <= 50,
            "at most 50 unique fingerprints"
        );
    }

    /// record_with_source tracks source locations per fingerprint.
    #[test]
    fn record_with_source_tracks_location() {
        let reg = QueryRegistry::new();
        let src = SourceContext::new("src/api.rs", 42, "handle_request");

        reg.record_with_source(0xABC, "MATCH (n) RETURN n", 100, &src);
        reg.record_with_source(0xABC, "MATCH (n) RETURN n", 200, &src);

        let stats = reg.get(0xABC).expect("entry should exist");
        assert_eq!(stats.count, 2);
        assert_eq!(stats.sources.len(), 1, "one unique source location");
        assert_eq!(stats.sources[0].file, "src/api.rs");
        assert_eq!(stats.sources[0].line, 42);
        assert_eq!(stats.sources[0].function, "handle_request");
        assert_eq!(stats.sources[0].call_count, 2);
    }

    /// Multiple sources tracked per fingerprint.
    #[test]
    fn multiple_sources_per_fingerprint() {
        let reg = QueryRegistry::new();
        let src1 = SourceContext::new("src/api.rs", 10, "fn_a");
        let src2 = SourceContext::new("src/api.rs", 20, "fn_b");

        reg.record_with_source(0xABC, "query", 100, &src1);
        reg.record_with_source(0xABC, "query", 100, &src1);
        reg.record_with_source(0xABC, "query", 100, &src2);

        let stats = reg.get(0xABC).expect("entry should exist");
        assert_eq!(stats.count, 3);
        assert_eq!(stats.sources.len(), 2);
        // Sorted by count descending
        assert_eq!(stats.sources[0].function, "fn_a");
        assert_eq!(stats.sources[0].call_count, 2);
        assert_eq!(stats.sources[1].function, "fn_b");
        assert_eq!(stats.sources[1].call_count, 1);
    }

    /// Mixing record() and record_with_source() on same fingerprint.
    #[test]
    fn mixed_record_and_record_with_source() {
        let reg = QueryRegistry::new();
        let src = SourceContext::new("src/handler.rs", 5, "main");

        // First, record without source
        reg.record(0xABC, "query", 100);
        // Then, record with source
        reg.record_with_source(0xABC, "query", 200, &src);

        let stats = reg.get(0xABC).expect("entry should exist");
        assert_eq!(stats.count, 2);
        assert_eq!(stats.sources.len(), 1);
        assert_eq!(stats.sources[0].call_count, 1);
    }

    /// Without source context, sources list is empty.
    #[test]
    fn no_source_gives_empty_sources() {
        let reg = QueryRegistry::new();
        reg.record(0xABC, "query", 100);

        let stats = reg.get(0xABC).expect("entry should exist");
        assert!(stats.sources.is_empty());
    }

    /// Source context with app and version is preserved.
    #[test]
    fn source_app_and_version_preserved() {
        let reg = QueryRegistry::new();
        let src = SourceContext::new("main.rs", 1, "run")
            .with_app("my-service")
            .with_version("v2.3.1");

        reg.record_with_source(0xABC, "query", 100, &src);

        let stats = reg.get(0xABC).expect("entry should exist");
        assert_eq!(stats.sources[0].app, "my-service");
        assert_eq!(stats.sources[0].version, "v2.3.1");
    }
}
