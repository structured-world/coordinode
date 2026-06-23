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
