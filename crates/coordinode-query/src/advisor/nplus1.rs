//! N+1 query pattern detection.
//!
//! Detects when the same query fingerprint is executed repeatedly from the same
//! source location within a short time window — a strong signal of an N+1 loop.
//!
//! The detector uses a sliding window counter: for each (fingerprint, source_key),
//! it tracks timestamps of recent executions. When the count exceeds the threshold
//! within the window, an N+1 alert is raised.
//!
//! Privacy: all data is in-memory only, never persisted. Counters are evicted
//! when the window expires.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use super::source::SourceContext;
use super::suggest::{Severity, Suggestion, SuggestionKind};

/// Default threshold: >100 calls in the window from the same source.
const DEFAULT_THRESHOLD: u64 = 100;

/// Default sliding window duration: 1 second.
const DEFAULT_WINDOW_SECS: u64 = 1;

/// Maximum number of tracked (fingerprint, source) pairs to prevent unbounded growth.
const MAX_TRACKED_PAIRS: usize = 1_000;

/// Key for tracking: (fingerprint, file, line, function).
/// Identifies a unique (query pattern, call site) pair.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TrackingKey {
    fingerprint: u64,
    file: String,
    line: u32,
    function: String,
}

impl TrackingKey {
    fn from_context(fingerprint: u64, source: &SourceContext) -> Self {
        Self {
            fingerprint,
            file: source.file.clone(),
            line: source.line,
            function: source.function.clone(),
        }
    }
}

/// Sliding window entry: tracks execution timestamps within the window.
struct WindowEntry {
    /// Timestamps of recent executions (monotonic, from `Instant`).
    timestamps: Vec<Instant>,
    /// Canonical query string (for suggestion text).
    canonical_query: String,
    /// Whether an alert has already been emitted for this window.
    /// Reset when the window clears.
    alerted: bool,
}

impl WindowEntry {
    fn new(canonical_query: String) -> Self {
        Self {
            timestamps: Vec::new(),
            canonical_query,
            alerted: false,
        }
    }

    /// Record a new execution. Prune timestamps outside the window.
    fn record(&mut self, now: Instant, window: std::time::Duration) {
        // Prune expired timestamps
        let cutoff = now.checked_sub(window).unwrap_or(now);
        self.timestamps.retain(|t| *t >= cutoff);
        self.timestamps.push(now);

        // Reset alert if window cleared and refilled
        if self.timestamps.len() == 1 {
            self.alerted = false;
        }
    }

    /// Current count within the window.
    fn count(&self) -> u64 {
        self.timestamps.len() as u64
    }
}

/// N+1 pattern detector configuration.
#[derive(Debug, Clone)]
pub struct NPlus1Config {
    /// Minimum calls within the window to trigger an alert.
    pub threshold: u64,
    /// Sliding window duration in seconds.
    pub window_secs: u64,
}

impl Default for NPlus1Config {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
            window_secs: DEFAULT_WINDOW_SECS,
        }
    }
}

/// N+1 pattern detector.
///
/// Thread-safe: all access goes through a Mutex. The hot path (record)
/// is lightweight — timestamp append + optional prune.
///
/// Multi-instance note: this is per-node in-memory state. In a 3-node
/// CE cluster, each node detects N+1 independently based on its own
/// traffic. This is acceptable — N+1 patterns are typically client-local.
pub struct NPlus1Detector {
    inner: Mutex<DetectorInner>,
    config: NPlus1Config,
    window: std::time::Duration,
}

struct DetectorInner {
    entries: HashMap<TrackingKey, WindowEntry>,
}

/// An N+1 alert: the query pattern was called too many times from one source.
#[derive(Debug, Clone)]
pub struct NPlus1Alert {
    pub fingerprint: u64,
    pub canonical_query: String,
    pub source_file: String,
    pub source_line: u32,
    pub source_function: String,
    pub call_count: u64,
    pub window_secs: u64,
    pub suggestion: Suggestion,
}

impl NPlus1Detector {
    /// Create a new detector with default configuration.
    pub fn new() -> Self {
        Self::with_config(NPlus1Config::default())
    }

    /// Create a new detector with custom configuration.
    pub fn with_config(config: NPlus1Config) -> Self {
        let window = std::time::Duration::from_secs(config.window_secs);
        Self {
            inner: Mutex::new(DetectorInner {
                entries: HashMap::new(),
            }),
            config,
            window,
        }
    }

    /// Record a query execution and check for N+1 pattern.
    ///
    /// Returns `Some(alert)` if the threshold is exceeded for the first time
    /// in the current window. Subsequent calls in the same window return `None`
    /// (alert is emitted once per window).
    ///
    /// `source` is required — without source context, N+1 detection is not
    /// possible (can't distinguish legitimate traffic from loop).
    pub fn record(
        &self,
        fingerprint: u64,
        canonical_query: &str,
        source: &SourceContext,
    ) -> Option<NPlus1Alert> {
        let key = TrackingKey::from_context(fingerprint, source);
        let now = Instant::now();

        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());

        // Evict oldest entries if at capacity
        if inner.entries.len() >= MAX_TRACKED_PAIRS && !inner.entries.contains_key(&key) {
            Self::evict_oldest(&mut inner.entries, now, self.window);
        }

        let entry = inner
            .entries
            .entry(key.clone())
            .or_insert_with(|| WindowEntry::new(canonical_query.to_string()));

        // Always record — even if timestamp equals last (fast loops)
        entry.record(now, self.window);

        let count = entry.count();

        if count >= self.config.threshold && !entry.alerted {
            entry.alerted = true;

            let suggestion = Suggestion::new(
                SuggestionKind::BatchRewrite,
                Severity::Warning,
                format!(
                    "N+1 query pattern: fingerprint executed {count}× in {}s \
                     from {file}:{line} ({func}). \
                     Replace the loop with a single UNWIND query to batch the operations",
                    self.config.window_secs,
                    file = source.file,
                    line = source.line,
                    func = source.function,
                ),
            )
            .with_rewrite(format!("UNWIND $ids AS id\n{}", canonical_query));

            Some(NPlus1Alert {
                fingerprint,
                canonical_query: canonical_query.to_string(),
                source_file: source.file.clone(),
                source_line: source.line,
                source_function: source.function.clone(),
                call_count: count,
                window_secs: self.config.window_secs,
                suggestion,
            })
        } else {
            None
        }
    }

    /// Check if a fingerprint+source pair is currently flagged as N+1.
    pub fn is_flagged(&self, fingerprint: u64, source: &SourceContext) -> bool {
        let key = TrackingKey::from_context(fingerprint, source);
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner
            .entries
            .get(&key)
            .is_some_and(|e| e.count() >= self.config.threshold)
    }

    /// Get all currently active N+1 alerts.
    pub fn active_alerts(&self) -> Vec<NPlus1Alert> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut alerts = Vec::new();

        for (key, entry) in &inner.entries {
            if entry.count() >= self.config.threshold {
                alerts.push(NPlus1Alert {
                    fingerprint: key.fingerprint,
                    canonical_query: entry.canonical_query.clone(),
                    source_file: key.file.clone(),
                    source_line: key.line,
                    source_function: key.function.clone(),
                    call_count: entry.count(),
                    window_secs: self.config.window_secs,
                    suggestion: Suggestion::new(
                        SuggestionKind::BatchRewrite,
                        Severity::Warning,
                        format!(
                            "N+1 query pattern: {} calls in {}s from {}:{}",
                            entry.count(),
                            self.config.window_secs,
                            key.file,
                            key.line,
                        ),
                    ),
                });
            }
        }

        alerts
    }

    /// Reset all tracking state.
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.entries.clear();
    }

    /// Evict entries whose timestamps are all expired.
    fn evict_oldest(
        entries: &mut HashMap<TrackingKey, WindowEntry>,
        now: Instant,
        window: std::time::Duration,
    ) {
        let cutoff = now.checked_sub(window).unwrap_or(now);
        entries.retain(|_, entry| entry.timestamps.iter().any(|t| *t >= cutoff));

        // If still at capacity after pruning expired, remove least active
        if entries.len() >= MAX_TRACKED_PAIRS {
            let min_key = entries
                .iter()
                .min_by_key(|(_, e)| e.count())
                .map(|(k, _)| k.clone());
            if let Some(key) = min_key {
                entries.remove(&key);
            }
        }
    }
}

impl Default for NPlus1Detector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
