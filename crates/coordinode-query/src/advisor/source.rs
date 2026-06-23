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
mod tests;
