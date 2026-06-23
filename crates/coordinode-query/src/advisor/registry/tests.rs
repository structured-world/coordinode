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
