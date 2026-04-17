//! Source location tracking for query advisor.
//!
//! When a client connects with debug source tracking enabled, each query includes
//! the application source context (file, line, function). This module tracks the
//! top-N call sites per query fingerprint by frequency.
//!
//! Privacy: source locations are in-memory only, never persisted to disk unless
//! explicitly enabled. Debug mode is opt-in per connection.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of call sites tracked per query fingerprint.
const MAX_SOURCES_PER_FINGERPRINT: usize = 5;

// --- Protocol-specific metadata key constants ---

/// gRPC metadata keys (lowercase, as required by HTTP/2 headers).
pub mod grpc_keys {
    pub const FILE: &str = "x-source-file";
    pub const LINE: &str = "x-source-line";
    pub const FUNCTION: &str = "x-source-function";
    pub const APP: &str = "x-source-app";
    pub const VERSION: &str = "x-source-version";
}

/// Bolt EXTRA dict keys (prefixed with underscore, backward compatible).
pub mod bolt_keys {
    pub const FILE: &str = "_source_file";
    pub const LINE: &str = "_source_line";
    pub const FUNCTION: &str = "_source_function";
    pub const APP: &str = "_source_app";
    pub const VERSION: &str = "_source_version";
}

/// HTTP header names (canonical casing).
pub mod http_keys {
    pub const FILE: &str = "X-Source-File";
    pub const LINE: &str = "X-Source-Line";
    pub const FUNCTION: &str = "X-Source-Function";
    pub const APP: &str = "X-Source-App";
    pub const VERSION: &str = "X-Source-Version";
}

/// Extract source context from a generic key-value metadata map.
///
/// This is the protocol-agnostic core. Protocol-specific extractors
/// call this with their own key constants.
///
/// Returns `None` if no file key is present (debug mode not enabled).
pub fn extract_from_map(
    get: &dyn Fn(&str) -> Option<String>,
    file_key: &str,
    line_key: &str,
    function_key: &str,
    app_key: &str,
    version_key: &str,
) -> Option<SourceContext> {
    let file = get(file_key)?;
    if file.is_empty() {
        return None;
    }

    let line = get(line_key)
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);

    let function = get(function_key).unwrap_or_default();

    let mut ctx = SourceContext::new(file, line, function);

    if let Some(app) = get(app_key) {
        ctx = ctx.with_app(app);
    }
    if let Some(version) = get(version_key) {
        ctx = ctx.with_version(version);
    }

    Some(ctx)
}

/// Extract source context from a `HashMap<String, String>` using Bolt keys.
///
/// For use in the Bolt protocol handler when parsing the EXTRA dict.
pub fn extract_from_bolt_extra(extra: &HashMap<String, String>) -> Option<SourceContext> {
    let get = |key: &str| extra.get(key).cloned();
    extract_from_map(
        &get,
        bolt_keys::FILE,
        bolt_keys::LINE,
        bolt_keys::FUNCTION,
        bolt_keys::APP,
        bolt_keys::VERSION,
    )
}

/// Extract source context from HTTP headers (case-insensitive lookup).
///
/// For use in the REST/HTTP handler. The `get_header` closure should
/// perform case-insensitive header lookup.
pub fn extract_from_http_headers(
    get_header: &dyn Fn(&str) -> Option<String>,
) -> Option<SourceContext> {
    extract_from_map(
        get_header,
        http_keys::FILE,
        http_keys::LINE,
        http_keys::FUNCTION,
        http_keys::APP,
        http_keys::VERSION,
    )
}

/// Source context extracted from a client request.
///
/// Populated from protocol-specific metadata:
/// - gRPC: `x-source-file`, `x-source-line`, `x-source-function`
/// - Bolt: EXTRA dict `_source_file`, `_source_line`, `_source_function`
/// - HTTP: `X-Source-File`, `X-Source-Line`, `X-Source-Function`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceContext {
    pub file: String,
    pub line: u32,
    pub function: String,
    pub app: String,
    pub version: String,
}

impl SourceContext {
    /// Create a source context with required fields.
    pub fn new(file: impl Into<String>, line: u32, function: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            line,
            function: function.into(),
            app: String::new(),
            version: String::new(),
        }
    }

    /// Set optional app name.
    pub fn with_app(mut self, app: impl Into<String>) -> Self {
        self.app = app.into();
        self
    }

    /// Set optional app version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Identity key for dedup: (file, line, function).
    #[cfg(test)]
    fn identity(&self) -> (&str, u32, &str) {
        (&self.file, self.line, &self.function)
    }
}

/// Snapshot of a source location with its call count.
#[derive(Debug, Clone)]
pub struct SourceLocationSnapshot {
    pub file: String,
    pub line: u32,
    pub function: String,
    pub app: String,
    pub version: String,
    pub call_count: u64,
}

/// A tracked source location entry with atomic call counter.
struct TrackedSource {
    file: String,
    line: u32,
    function: String,
    app: String,
    version: String,
    call_count: AtomicU64,
}

impl TrackedSource {
    fn from_context(ctx: &SourceContext) -> Self {
        Self {
            file: ctx.file.clone(),
            line: ctx.line,
            function: ctx.function.clone(),
            app: ctx.app.clone(),
            version: ctx.version.clone(),
            call_count: AtomicU64::new(1),
        }
    }

    fn matches(&self, ctx: &SourceContext) -> bool {
        self.file == ctx.file && self.line == ctx.line && self.function == ctx.function
    }

    fn increment(&self) {
        self.call_count.fetch_add(1, Ordering::Relaxed);
    }

    fn count(&self) -> u64 {
        self.call_count.load(Ordering::Relaxed)
    }

    fn snapshot(&self) -> SourceLocationSnapshot {
        SourceLocationSnapshot {
            file: self.file.clone(),
            line: self.line,
            function: self.function.clone(),
            app: self.app.clone(),
            version: self.version.clone(),
            call_count: self.count(),
        }
    }
}

/// Tracks top-N source locations per query fingerprint.
///
/// Bounded to `MAX_SOURCES_PER_FINGERPRINT` entries. When full, the least
/// frequent call site is evicted to make room for new sources that are
/// observed more frequently.
///
/// Thread safety: must be called under the registry's mutex (or the entry's
/// lock). Individual `call_count` fields are atomic for concurrent reads.
pub(crate) struct SourceTracker {
    sources: Vec<TrackedSource>,
}

impl SourceTracker {
    pub(crate) fn new() -> Self {
        Self {
            sources: Vec::with_capacity(MAX_SOURCES_PER_FINGERPRINT),
        }
    }

    /// Record a call from the given source context.
    ///
    /// If the source is already tracked, increments its counter.
    /// If new and there's room, adds it.
    /// If new and full, evicts the least frequent source (only if the
    /// new source would immediately have a higher count than the minimum).
    pub(crate) fn record(&mut self, ctx: &SourceContext) {
        // Check if this source is already tracked
        for source in &self.sources {
            if source.matches(ctx) {
                source.increment();
                return;
            }
        }

        // New source: add if room available
        if self.sources.len() < MAX_SOURCES_PER_FINGERPRINT {
            self.sources.push(TrackedSource::from_context(ctx));
            return;
        }

        // Full: evict the least frequent source.
        // New sources start at count=1, so only evict if min is also 1
        // (fair replacement for equally-rare sources).
        let min_idx = self
            .sources
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.count())
            .map(|(i, _)| i);

        if let Some(idx) = min_idx {
            if self.sources[idx].count() <= 1 {
                self.sources[idx] = TrackedSource::from_context(ctx);
            }
        }
    }

    /// Snapshot of all tracked sources, sorted by call count descending.
    pub(crate) fn snapshot(&self) -> Vec<SourceLocationSnapshot> {
        let mut snaps: Vec<SourceLocationSnapshot> =
            self.sources.iter().map(|s| s.snapshot()).collect();
        snaps.sort_by_key(|s| std::cmp::Reverse(s.call_count));
        snaps
    }

    /// Number of tracked sources.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.sources.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn ctx(file: &str, line: u32, func: &str) -> SourceContext {
        SourceContext::new(file, line, func)
    }

    /// Recording the same source increments its counter.
    #[test]
    fn same_source_increments() {
        let mut tracker = SourceTracker::new();
        let c = ctx("src/main.rs", 42, "handle_request");

        tracker.record(&c);
        tracker.record(&c);
        tracker.record(&c);

        let snaps = tracker.snapshot();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].call_count, 3);
        assert_eq!(snaps[0].file, "src/main.rs");
        assert_eq!(snaps[0].line, 42);
        assert_eq!(snaps[0].function, "handle_request");
    }

    /// Different sources are tracked separately.
    #[test]
    fn different_sources_tracked() {
        let mut tracker = SourceTracker::new();

        tracker.record(&ctx("a.rs", 1, "fn_a"));
        tracker.record(&ctx("b.rs", 2, "fn_b"));
        tracker.record(&ctx("c.rs", 3, "fn_c"));

        assert_eq!(tracker.len(), 3);
    }

    /// Tracker is bounded to MAX_SOURCES_PER_FINGERPRINT entries.
    #[test]
    fn bounded_capacity() {
        let mut tracker = SourceTracker::new();

        // Fill to capacity
        for i in 0..MAX_SOURCES_PER_FINGERPRINT {
            let c = ctx(&format!("file_{i}.rs"), i as u32, &format!("fn_{i}"));
            // Record multiple times to give them count > 1
            tracker.record(&c);
            tracker.record(&c);
        }

        assert_eq!(tracker.len(), MAX_SOURCES_PER_FINGERPRINT);

        // Adding one more with count=1 won't evict sources with count=2
        tracker.record(&ctx("new.rs", 99, "fn_new"));
        assert_eq!(tracker.len(), MAX_SOURCES_PER_FINGERPRINT);
    }

    /// Eviction replaces least frequent source when counts are equal (=1).
    #[test]
    fn evicts_least_frequent_when_tied() {
        let mut tracker = SourceTracker::new();

        // Fill with sources that each have count=1
        for i in 0..MAX_SOURCES_PER_FINGERPRINT {
            tracker.record(&ctx(&format!("file_{i}.rs"), i as u32, &format!("fn_{i}")));
        }

        // Now bump all but the first to count=2
        for i in 1..MAX_SOURCES_PER_FINGERPRINT {
            tracker.record(&ctx(&format!("file_{i}.rs"), i as u32, &format!("fn_{i}")));
        }

        // Adding a new source should evict file_0 (count=1, the minimum)
        tracker.record(&ctx("new.rs", 99, "fn_new"));

        let snaps = tracker.snapshot();
        assert_eq!(snaps.len(), MAX_SOURCES_PER_FINGERPRINT);
        assert!(
            snaps.iter().all(|s| s.file != "file_0.rs"),
            "file_0.rs (count=1) should have been evicted"
        );
        assert!(
            snaps.iter().any(|s| s.file == "new.rs"),
            "new.rs should be in the tracker"
        );
    }

    /// Snapshot returns sources sorted by call count descending.
    #[test]
    fn snapshot_sorted_by_count() {
        let mut tracker = SourceTracker::new();

        let c1 = ctx("a.rs", 1, "fn_a");
        let c2 = ctx("b.rs", 2, "fn_b");
        let c3 = ctx("c.rs", 3, "fn_c");

        tracker.record(&c1);
        for _ in 0..5 {
            tracker.record(&c2);
        }
        for _ in 0..3 {
            tracker.record(&c3);
        }

        let snaps = tracker.snapshot();
        assert_eq!(snaps[0].file, "b.rs");
        assert_eq!(snaps[0].call_count, 5);
        assert_eq!(snaps[1].file, "c.rs");
        assert_eq!(snaps[1].call_count, 3);
        assert_eq!(snaps[2].file, "a.rs");
        assert_eq!(snaps[2].call_count, 1);
    }

    /// SourceContext builder methods work.
    #[test]
    fn source_context_builder() {
        let c = SourceContext::new("src/main.rs", 42, "main")
            .with_app("my-service")
            .with_version("v1.0.0");

        assert_eq!(c.file, "src/main.rs");
        assert_eq!(c.line, 42);
        assert_eq!(c.function, "main");
        assert_eq!(c.app, "my-service");
        assert_eq!(c.version, "v1.0.0");
    }

    /// SourceContext identity distinguishes by (file, line, function).
    #[test]
    fn source_identity() {
        let c1 = ctx("a.rs", 1, "fn_a");
        let c2 = ctx("a.rs", 1, "fn_a"); // same identity
        let c3 = ctx("a.rs", 2, "fn_a"); // different line

        assert_eq!(c1.identity(), c2.identity());
        assert_ne!(c1.identity(), c3.identity());
    }

    /// App and version are preserved in snapshots.
    #[test]
    fn snapshot_preserves_app_info() {
        let mut tracker = SourceTracker::new();
        let c = SourceContext::new("src/api.rs", 10, "handler")
            .with_app("my-app")
            .with_version("v2.0");

        tracker.record(&c);

        let snaps = tracker.snapshot();
        assert_eq!(snaps[0].app, "my-app");
        assert_eq!(snaps[0].version, "v2.0");
    }

    /// Empty tracker returns empty snapshot.
    #[test]
    fn empty_tracker() {
        let tracker = SourceTracker::new();
        assert!(tracker.snapshot().is_empty());
        assert_eq!(tracker.len(), 0);
    }

    // --- Extraction function tests ---

    /// extract_from_map returns SourceContext when file key is present.
    #[test]
    fn extract_from_map_full_context() {
        let mut map = HashMap::new();
        map.insert("file".to_string(), "src/main.rs".to_string());
        map.insert("line".to_string(), "42".to_string());
        map.insert("func".to_string(), "handle".to_string());
        map.insert("app".to_string(), "my-app".to_string());
        map.insert("ver".to_string(), "v1.0".to_string());

        let get = |key: &str| -> Option<String> { map.get(key).cloned() };
        let result = extract_from_map(&get, "file", "line", "func", "app", "ver");

        let ctx = result.expect("should extract context");
        assert_eq!(ctx.file, "src/main.rs");
        assert_eq!(ctx.line, 42);
        assert_eq!(ctx.function, "handle");
        assert_eq!(ctx.app, "my-app");
        assert_eq!(ctx.version, "v1.0");
    }

    /// extract_from_map returns None when file key is missing.
    #[test]
    fn extract_from_map_no_file() {
        let map: HashMap<String, String> = HashMap::new();
        let get = |key: &str| -> Option<String> { map.get(key).cloned() };
        let result = extract_from_map(&get, "file", "line", "func", "app", "ver");
        assert!(result.is_none());
    }

    /// extract_from_map returns None when file is empty string.
    #[test]
    fn extract_from_map_empty_file() {
        let mut map = HashMap::new();
        map.insert("file".to_string(), String::new());

        let get = |key: &str| -> Option<String> { map.get(key).cloned() };
        let result = extract_from_map(&get, "file", "line", "func", "app", "ver");
        assert!(result.is_none());
    }

    /// extract_from_map handles missing optional fields gracefully.
    #[test]
    fn extract_from_map_minimal() {
        let mut map = HashMap::new();
        map.insert("file".to_string(), "src/lib.rs".to_string());

        let get = |key: &str| -> Option<String> { map.get(key).cloned() };
        let result = extract_from_map(&get, "file", "line", "func", "app", "ver");

        let ctx = result.expect("should extract with file only");
        assert_eq!(ctx.file, "src/lib.rs");
        assert_eq!(ctx.line, 0);
        assert_eq!(ctx.function, "");
        assert_eq!(ctx.app, "");
        assert_eq!(ctx.version, "");
    }

    /// extract_from_map handles non-numeric line gracefully.
    #[test]
    fn extract_from_map_bad_line() {
        let mut map = HashMap::new();
        map.insert("file".to_string(), "src/lib.rs".to_string());
        map.insert("line".to_string(), "not-a-number".to_string());

        let get = |key: &str| -> Option<String> { map.get(key).cloned() };
        let result = extract_from_map(&get, "file", "line", "func", "app", "ver");

        let ctx = result.expect("should extract despite bad line");
        assert_eq!(ctx.line, 0, "non-numeric line defaults to 0");
    }

    /// extract_from_bolt_extra works with Bolt key conventions.
    #[test]
    fn extract_bolt_extra() {
        let mut extra = HashMap::new();
        extra.insert("_source_file".to_string(), "app.py".to_string());
        extra.insert("_source_line".to_string(), "100".to_string());
        extra.insert("_source_function".to_string(), "query_db".to_string());

        let ctx = extract_from_bolt_extra(&extra).expect("should extract");
        assert_eq!(ctx.file, "app.py");
        assert_eq!(ctx.line, 100);
        assert_eq!(ctx.function, "query_db");
    }

    /// extract_from_http_headers works with HTTP header conventions.
    #[test]
    fn extract_http_headers() {
        let mut headers = HashMap::new();
        headers.insert("X-Source-File".to_string(), "index.ts".to_string());
        headers.insert("X-Source-Line".to_string(), "55".to_string());
        headers.insert("X-Source-Function".to_string(), "fetchData".to_string());
        headers.insert("X-Source-App".to_string(), "frontend".to_string());

        let get = |key: &str| -> Option<String> { headers.get(key).cloned() };
        let ctx = extract_from_http_headers(&get).expect("should extract");
        assert_eq!(ctx.file, "index.ts");
        assert_eq!(ctx.line, 55);
        assert_eq!(ctx.function, "fetchData");
        assert_eq!(ctx.app, "frontend");
    }
}
