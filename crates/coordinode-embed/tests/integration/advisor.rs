//! Integration tests for the query advisor (fingerprint registry).
//!
//! Verifies that executing queries through the Database API automatically
//! records fingerprints and timing in the advisor registry.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_embed::Database;

/// Executing a query records its fingerprint in the advisor registry.
#[test]
fn query_execution_records_fingerprint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create");

    let reg = db.query_registry();
    assert_eq!(
        reg.fingerprint_count(),
        1,
        "one unique fingerprint after one CREATE"
    );
    assert_eq!(reg.total_queries_recorded(), 1);
}

/// Queries with different literals but same structure share a fingerprint.
#[test]
fn same_structure_shares_fingerprint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create 1");
    db.execute_cypher("CREATE (n:User {name: 'Bob'})")
        .expect("create 2");

    let reg = db.query_registry();
    assert_eq!(
        reg.fingerprint_count(),
        1,
        "both CREATEs differ only by literal → same fingerprint"
    );
    assert_eq!(reg.total_queries_recorded(), 2);
}

/// Different query structures produce different fingerprints.
#[test]
fn different_structures_different_fingerprints() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create");
    db.execute_cypher("MATCH (n:User) RETURN n").expect("match");

    let reg = db.query_registry();
    assert_eq!(
        reg.fingerprint_count(),
        2,
        "CREATE and MATCH are different structures"
    );
}

/// top_by_count returns the most frequently executed query patterns.
#[test]
fn top_by_count_reflects_execution_frequency() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Execute CREATE once, MATCH three times
    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create");
    for _ in 0..3 {
        db.execute_cypher("MATCH (n:User) RETURN n").expect("match");
    }

    let top = db.query_registry().top_by_count(1);
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].count, 3, "MATCH was executed 3 times");
    assert!(
        top[0].canonical_query.contains("MATCH"),
        "top query should be the MATCH pattern"
    );
}

/// Timing is recorded (non-zero duration for real queries).
#[test]
fn timing_is_recorded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create");

    let top = db.query_registry().top_by_count(1);
    assert_eq!(top.len(), 1);
    // total_time_us should be > 0 (real execution takes some microseconds)
    assert!(
        top[0].total_time_us > 0,
        "execution time should be recorded"
    );
    // count should match
    assert_eq!(top[0].count, 1);
}

/// Reset clears all advisor state.
#[test]
fn reset_clears_registry() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create");
    assert_eq!(db.query_registry().fingerprint_count(), 1);

    db.query_registry().reset();
    assert_eq!(db.query_registry().fingerprint_count(), 0);
    assert_eq!(db.query_registry().total_queries_recorded(), 0);
}

// --- Source location tracking integration tests ---

use coordinode_query::advisor::SourceContext;

/// Source location is recorded when executing with source context.
#[test]
fn source_location_recorded_via_execute() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let src = SourceContext::new("src/api/handlers.rs", 42, "get_user");
    db.execute_cypher_with_source("CREATE (n:User {name: 'Alice'})", &src)
        .expect("create");

    let top = db.query_registry().top_by_count(1);
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].count, 1);
    assert_eq!(top[0].sources.len(), 1, "one source location tracked");
    assert_eq!(top[0].sources[0].file, "src/api/handlers.rs");
    assert_eq!(top[0].sources[0].line, 42);
    assert_eq!(top[0].sources[0].function, "get_user");
    assert_eq!(top[0].sources[0].call_count, 1);
}

/// Multiple calls from same source accumulate call count.
#[test]
fn source_call_count_accumulates() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let src = SourceContext::new("src/handler.rs", 10, "list_users");
    for _ in 0..5 {
        db.execute_cypher_with_source("MATCH (n:User) RETURN n", &src)
            .expect("match");
    }

    let top = db.query_registry().top_by_count(1);
    assert_eq!(top[0].count, 5);
    assert_eq!(top[0].sources.len(), 1);
    assert_eq!(top[0].sources[0].call_count, 5);
}

/// Multiple different sources for same query are tracked separately.
#[test]
fn multiple_sources_for_same_query() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let src1 = SourceContext::new("src/api.rs", 10, "handler_a");
    let src2 = SourceContext::new("src/api.rs", 20, "handler_b");
    let src3 = SourceContext::new("src/admin.rs", 5, "admin_lookup");

    db.execute_cypher_with_source("MATCH (n:User) RETURN n", &src1)
        .expect("q1");
    db.execute_cypher_with_source("MATCH (n:User) RETURN n", &src1)
        .expect("q2");
    db.execute_cypher_with_source("MATCH (n:User) RETURN n", &src2)
        .expect("q3");
    db.execute_cypher_with_source("MATCH (n:User) RETURN n", &src3)
        .expect("q4");

    let top = db.query_registry().top_by_count(1);
    assert_eq!(top[0].count, 4, "4 total executions");
    assert_eq!(top[0].sources.len(), 3, "3 distinct sources");
    // Sorted by call count descending
    assert_eq!(top[0].sources[0].function, "handler_a");
    assert_eq!(top[0].sources[0].call_count, 2);
    assert_eq!(top[0].sources[1].call_count, 1);
    assert_eq!(top[0].sources[2].call_count, 1);
}

// --- EXPLAIN SUGGEST integration tests ---

/// EXPLAIN SUGGEST returns plan + suggestions for a query with missing index.
#[test]
fn explain_suggest_detects_missing_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    // MATCH (n:User) WHERE n.email = 'x' RETURN n → full label scan + filter
    let result = db
        .explain_suggest("MATCH (n:User) WHERE n.email = 'alice@test.com' RETURN n")
        .expect("explain_suggest");

    // Should have the plan
    assert!(
        result.explain.contains("NodeScan"),
        "explain should show NodeScan"
    );

    // Should detect missing index
    assert!(
        !result.suggestions.is_empty(),
        "should have at least one suggestion"
    );
    assert_eq!(
        result.suggestions[0].kind,
        coordinode_query::advisor::SuggestionKind::CreateIndex,
    );
    assert!(result.suggestions[0]
        .ddl
        .as_ref()
        .unwrap()
        .contains("ON User(email)"));
}

/// EXPLAIN SUGGEST returns no suggestions for a simple scan without filter.
#[test]
fn explain_suggest_no_suggestions_for_simple_scan() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest("MATCH (n:User) RETURN n")
        .expect("explain_suggest");

    assert!(
        result.suggestions.is_empty(),
        "simple label scan without filter should have no suggestions"
    );
}

/// EXPLAIN SUGGEST display format includes suggestions section.
#[test]
fn explain_suggest_display_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest("MATCH (n:User) WHERE n.email = 'x' RETURN n")
        .expect("explain_suggest");

    let text = result.to_string();
    assert!(
        text.contains("SUGGESTIONS"),
        "should show SUGGESTIONS header"
    );
    assert!(
        text.contains("CREATE INDEX"),
        "should show CREATE INDEX suggestion"
    );
}

/// EXPLAIN SUGGEST detects unbounded variable-length traversal.
#[test]
fn explain_suggest_detects_unbounded_traversal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest("MATCH (a:User)-[:KNOWS*1..]->(b) RETURN b")
        .expect("explain_suggest");

    assert!(
        result
            .suggestions
            .iter()
            .any(|s| s.kind == coordinode_query::advisor::SuggestionKind::AddDepthBound),
        "should detect unbounded traversal: {:?}",
        result.suggestions
    );
}

/// Bounded traversal produces no unbounded warning.
#[test]
fn explain_suggest_bounded_traversal_ok() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest("MATCH (a:User)-[:KNOWS*1..5]->(b) RETURN b")
        .expect("explain_suggest");

    assert!(
        !result
            .suggestions
            .iter()
            .any(|s| s.kind == coordinode_query::advisor::SuggestionKind::AddDepthBound),
        "bounded traversal should not trigger unbounded warning"
    );
}

/// EXPLAIN SUGGEST detects Cartesian product from disconnected patterns.
#[test]
fn explain_suggest_detects_cartesian_product() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest("MATCH (a:User), (b:Post) RETURN a, b")
        .expect("explain_suggest");

    assert!(
        result
            .suggestions
            .iter()
            .any(|s| s.kind == coordinode_query::advisor::SuggestionKind::AddJoinPredicate),
        "should detect Cartesian product: {:?}",
        result.suggestions
    );
}

/// EXPLAIN SUGGEST detects VectorFilter without graph pre-filter.
#[test]
fn explain_suggest_detects_vector_without_prefilter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    // VectorFilter directly on NodeScan (no graph narrowing)
    let result = db
        .explain_suggest(
            "MATCH (n:Product) WHERE vector_distance(n.embedding, [1.0, 0.0, 0.0]) < 0.5 RETURN n",
        )
        .expect("explain_suggest");

    assert!(
        result
            .suggestions
            .iter()
            .any(|s| s.kind == coordinode_query::advisor::SuggestionKind::AddGraphPreFilter),
        "should detect vector without pre-filter: {:?}",
        result.suggestions
    );
}

/// VectorFilter after graph traversal produces no pre-filter warning.
#[test]
fn explain_suggest_vector_with_prefilter_ok() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    // VectorFilter after Traverse (has graph narrowing)
    let result = db
        .explain_suggest(
            "MATCH (u:User)-[:LIKES]->(p:Product) \
             WHERE vector_distance(p.embedding, [1.0, 0.0]) < 0.3 \
             RETURN p",
        )
        .expect("explain_suggest");

    assert!(
        !result
            .suggestions
            .iter()
            .any(|s| s.kind == coordinode_query::advisor::SuggestionKind::AddGraphPreFilter),
        "vector after traversal should not trigger pre-filter warning: {:?}",
        result.suggestions
    );
}

/// EXPLAIN SUGGEST detects KNN pattern (ORDER BY distance + LIMIT without index).
#[test]
fn explain_suggest_detects_knn_without_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest(
            "MATCH (n:Place) RETURN n \
             ORDER BY vector_distance(n.location, [40.7, -74.0]) LIMIT 10",
        )
        .expect("explain_suggest");

    assert!(
        result
            .suggestions
            .iter()
            .any(|s| s.kind == coordinode_query::advisor::SuggestionKind::CreateVectorIndex),
        "should detect KNN without index: {:?}",
        result.suggestions
    );
}

/// EXPLAIN SUGGEST does NOT emit CreateVectorIndex when the HNSW index already
/// exists — prevents false-positive suggestions for already-optimized queries.
#[test]
fn explain_suggest_skips_knn_suggestion_when_index_exists() {
    use coordinode_core::graph::types::VectorMetric;
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create HNSW index for (Place, location).
    db.create_vector_index(
        "place_loc_idx",
        "Place",
        "location",
        VectorIndexConfig {
            dimensions: 2,
            metric: VectorMetric::L2,
            m: 16,
            ef_construction: 200,
            quantization: false,
            offload_vectors: false,
        },
    );

    let result = db
        .explain_suggest(
            "MATCH (n:Place) RETURN n \
             ORDER BY vector_distance(n.location, [40.7, -74.0]) LIMIT 10",
        )
        .expect("explain_suggest");

    assert!(
        !result
            .suggestions
            .iter()
            .any(|s| s.kind == coordinode_query::advisor::SuggestionKind::CreateVectorIndex),
        "must NOT suggest CreateVectorIndex when HNSW index exists: {:?}",
        result.suggestions
    );
}

/// EXPLAIN SUGGEST emits CreateVectorIndex suggestion for a DIFFERENT property
/// even when one (label, property) pair has an HNSW index. Verifies per-property
/// granularity of the suggestion skip.
#[test]
fn explain_suggest_suggests_for_different_property_when_one_indexed() {
    use coordinode_core::graph::types::VectorMetric;
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Index on (Place, location), but query on (Place, thumbnail) — different property.
    db.create_vector_index(
        "place_loc_idx",
        "Place",
        "location",
        VectorIndexConfig {
            dimensions: 2,
            metric: VectorMetric::L2,
            m: 16,
            ef_construction: 200,
            quantization: false,
            offload_vectors: false,
        },
    );

    let result = db
        .explain_suggest(
            "MATCH (n:Place) RETURN n \
             ORDER BY vector_distance(n.thumbnail, [0.1, 0.2, 0.3]) LIMIT 5",
        )
        .expect("explain_suggest");

    assert!(
        result
            .suggestions
            .iter()
            .any(|s| s.kind == coordinode_query::advisor::SuggestionKind::CreateVectorIndex),
        "should suggest for thumbnail even when location is indexed: {:?}",
        result.suggestions
    );
}

/// Multiple suggestions are ranked by severity.
#[test]
fn explain_suggest_ranks_by_severity() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    // Query with both filter on unindexed property + Cartesian product
    let result = db
        .explain_suggest("MATCH (a:User), (b:Post) WHERE a.email = 'test@example.com' RETURN a, b")
        .expect("explain_suggest");

    if result.suggestions.len() >= 2 {
        let first_severity = result.suggestions[0].severity;
        let last_severity = result.suggestions.last().unwrap().severity;
        assert!(
            first_severity >= last_severity,
            "suggestions should be sorted by severity descending"
        );
    }
}

// --- N+1 detection integration tests ---

/// N+1 pattern detected when same query executed from same source many times.
#[test]
fn nplus1_detected_via_execute() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let src = SourceContext::new("src/handlers/feed.rs", 67, "build_feed");

    // Execute same query 101 times from same source (threshold = 100)
    for _ in 0..101 {
        db.execute_cypher_with_source("MATCH (n:User) RETURN n", &src)
            .expect("query");
    }

    let alerts = db.nplus1_detector().active_alerts();
    assert!(
        !alerts.is_empty(),
        "should have N+1 alert after 101 calls from same source"
    );
    assert_eq!(alerts[0].source_file, "src/handlers/feed.rs");
    assert_eq!(alerts[0].source_function, "build_feed");
    assert!(alerts[0].call_count >= 101);
}

/// N+1 not flagged below threshold.
#[test]
fn nplus1_not_flagged_below_threshold() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let src = SourceContext::new("src/api.rs", 10, "get_user");

    for _ in 0..50 {
        db.execute_cypher_with_source("MATCH (n:User) RETURN n", &src)
            .expect("query");
    }

    let alerts = db.nplus1_detector().active_alerts();
    assert!(
        alerts.is_empty(),
        "should NOT have N+1 alert after only 50 calls"
    );
}

/// Different sources are tracked independently for N+1.
#[test]
fn nplus1_different_sources_independent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let src_a = SourceContext::new("a.rs", 1, "fn_a");
    let src_b = SourceContext::new("b.rs", 2, "fn_b");

    // 101 calls from src_a, 50 from src_b
    for _ in 0..101 {
        db.execute_cypher_with_source("MATCH (n:User) RETURN n", &src_a)
            .expect("query");
    }
    for _ in 0..50 {
        db.execute_cypher_with_source("MATCH (n:User) RETURN n", &src_b)
            .expect("query");
    }

    let alerts = db.nplus1_detector().active_alerts();
    assert_eq!(alerts.len(), 1, "only src_a should be flagged");
    assert_eq!(alerts[0].source_file, "a.rs");
}

/// Mixing execute_cypher (no source) and execute_cypher_with_source works.
#[test]
fn mixed_source_and_no_source() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let src = SourceContext::new("src/main.rs", 1, "main")
        .with_app("test-app")
        .with_version("v1.0");

    // Without source
    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create no source");

    // With source
    db.execute_cypher_with_source("CREATE (n:User {name: 'Bob'})", &src)
        .expect("create with source");

    let top = db.query_registry().top_by_count(1);
    assert_eq!(top[0].count, 2, "2 total executions");
    assert_eq!(
        top[0].sources.len(),
        1,
        "only 1 source (the one with context)"
    );
    assert_eq!(top[0].sources[0].app, "test-app");
    assert_eq!(top[0].sources[0].version, "v1.0");
}

// --- CALL db.advisor.* procedure integration tests ---

use coordinode_core::graph::types::Value;

/// CALL db.advisor.queryStats() returns stats after executing queries.
#[test]
fn call_query_stats_via_cypher() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Execute some queries first
    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create");
    db.execute_cypher("MATCH (n:User) RETURN n").expect("match");
    db.execute_cypher("MATCH (n:User) RETURN n").expect("match");

    // Call the procedure via Cypher
    let rows = db
        .execute_cypher("CALL db.advisor.queryStats()")
        .expect("call queryStats");

    assert_eq!(rows.len(), 2, "two distinct query fingerprints");

    // Verify row columns exist
    let first = &rows[0];
    assert!(first.contains_key("fingerprint"), "should have fingerprint");
    assert!(first.contains_key("query"), "should have query");
    assert!(first.contains_key("count"), "should have count");
    assert!(first.contains_key("avgTime"), "should have avgTime");
    assert!(first.contains_key("p99Time"), "should have p99Time");
    assert!(first.contains_key("plan"), "should have plan");
    assert!(first.contains_key("shardsUsed"), "should have shardsUsed");

    // Plan should be a non-null string (the EXPLAIN output)
    match first.get("plan") {
        Some(Value::String(plan)) => {
            assert!(!plan.is_empty(), "plan should not be empty");
        }
        other => panic!("expected plan to be a string, got: {other:?}"),
    }

    // CE always reports shardsUsed=1
    assert_eq!(first.get("shardsUsed"), Some(&Value::Int(1)));
}

/// CALL db.advisor.queryStats() with YIELD filters columns.
#[test]
fn call_query_stats_with_yield() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create");

    let rows = db
        .execute_cypher("CALL db.advisor.queryStats() YIELD query, count")
        .expect("call with yield");

    assert!(!rows.is_empty());
    let first = &rows[0];
    assert!(first.contains_key("query"), "should have yielded query");
    assert!(first.contains_key("count"), "should have yielded count");
    assert!(
        !first.contains_key("avgTime"),
        "should NOT have avgTime when not yielded"
    );
}

/// CALL db.advisor.suggestions() returns suggestions for recorded queries.
#[test]
fn call_suggestions_via_cypher() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Execute queries to populate registry
    for _ in 0..5 {
        db.execute_cypher("MATCH (n:User) RETURN n").expect("match");
    }

    let rows = db
        .execute_cypher("CALL db.advisor.suggestions()")
        .expect("call suggestions");

    // Should have suggestions (the queries were recorded with timing)
    assert!(
        !rows.is_empty(),
        "should have suggestions after recording queries"
    );

    let first = &rows[0];
    assert!(first.contains_key("id"), "should have id");
    assert!(first.contains_key("severity"), "should have severity");
    assert!(first.contains_key("kind"), "should have kind");
    assert!(first.contains_key("impact"), "should have impact");
}

/// CALL db.advisor.slowQueries() with arguments.
#[test]
fn call_slow_queries_via_cypher() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Execute some queries
    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create");

    // With explicit args: limit=10, minTime=0 (catch everything)
    let rows = db
        .execute_cypher("CALL db.advisor.slowQueries(10, 0)")
        .expect("call slowQueries");

    // There should be results (minTime=0 catches all)
    assert!(!rows.is_empty(), "should have results with minTime=0");

    let first = &rows[0];
    assert!(first.contains_key("query"), "should have query");
    assert!(first.contains_key("p99Time"), "should have p99Time");
    assert!(first.contains_key("count"), "should have count");
    assert!(first.contains_key("plan"), "should have plan");
}

/// CALL db.advisor.reset() clears all advisor state.
#[test]
fn call_reset_via_cypher() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Record some queries
    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create");
    assert!(db.query_registry().fingerprint_count() > 0);

    // Reset via CALL
    let rows = db
        .execute_cypher("CALL db.advisor.reset()")
        .expect("call reset");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("status"),
        Some(&Value::String("OK".to_string()))
    );

    // Verify state was cleared (note: the CALL itself records a fingerprint)
    // So fingerprint count will be 1 (the CALL query itself)
    assert_eq!(
        db.query_registry().fingerprint_count(),
        1,
        "only the CALL query should remain after reset"
    );
}

/// CALL db.advisor.dismiss() suppresses a suggestion.
#[test]
fn call_dismiss_via_cypher() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Record queries
    db.execute_cypher("MATCH (n:User) RETURN n").expect("match");

    // Get suggestions to find an ID
    let suggestions = db
        .execute_cypher("CALL db.advisor.suggestions()")
        .expect("suggestions");
    assert!(!suggestions.is_empty(), "need at least one suggestion");

    // Extract the first suggestion ID
    let id = match suggestions[0].get("id") {
        Some(Value::String(s)) => s.clone(),
        other => panic!("expected string id, got: {other:?}"),
    };

    // Dismiss it
    let dismiss_result = db
        .execute_cypher(&format!("CALL db.advisor.dismiss('{id}')"))
        .expect("dismiss");
    assert_eq!(dismiss_result.len(), 1);
    assert_eq!(dismiss_result[0].get("dismissed"), Some(&Value::Bool(true)));

    // Verify suggestion is no longer returned
    let after = db
        .execute_cypher("CALL db.advisor.suggestions()")
        .expect("suggestions after dismiss");
    let still_present = after
        .iter()
        .any(|r| r.get("id") == Some(&Value::String(id.clone())));
    assert!(
        !still_present,
        "dismissed suggestion should not appear in results"
    );
}

// --- Edge vector plan selection integration tests ---

/// EXPLAIN for edge vector query shows EdgeVectorSearch with strategy.
#[test]
fn explain_edge_vector_shows_strategy() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    // Query with vector filter on an edge variable after traverse
    let explain = db
        .explain_cypher(
            "MATCH (u:User)-[r:KNOWS]->(f) \
             WHERE vector_distance(r.embedding, [1.0, 0.0, 0.0]) < 0.3 \
             RETURN f",
        )
        .expect("explain");

    // Should show EdgeVectorSearch instead of VectorFilter
    assert!(
        explain.contains("EdgeVectorSearch"),
        "EXPLAIN should show EdgeVectorSearch for edge vector query, got: {explain}"
    );

    // Should show the strategy
    assert!(
        explain.contains("Graph-First") || explain.contains("Vector-First"),
        "EXPLAIN should show strategy, got: {explain}"
    );

    // Should show the edge type
    assert!(
        explain.contains("KNOWS"),
        "EXPLAIN should show edge type, got: {explain}"
    );
}

/// EXPLAIN for node vector query still shows VectorFilter (not EdgeVectorSearch).
#[test]
fn explain_node_vector_not_rewritten() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    // Node vector filter — should NOT be rewritten to EdgeVectorSearch
    let explain = db
        .explain_cypher(
            "MATCH (n:Product) \
             WHERE vector_distance(n.embedding, [1.0, 0.0]) < 0.5 \
             RETURN n",
        )
        .expect("explain");

    assert!(
        explain.contains("VectorFilter"),
        "node vector should remain as VectorFilter, got: {explain}"
    );
    assert!(
        !explain.contains("EdgeVectorSearch"),
        "node vector should NOT be rewritten to EdgeVectorSearch, got: {explain}"
    );
}

/// Edge vector query executes correctly (brute-force evaluation on edge properties).
#[test]
fn edge_vector_query_executes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create nodes and edges with vector properties
    db.execute_cypher("CREATE (a:User {name: 'Alice'})")
        .expect("create alice");
    db.execute_cypher("CREATE (b:User {name: 'Bob'})")
        .expect("create bob");
    db.execute_cypher("CREATE (c:User {name: 'Charlie'})")
        .expect("create charlie");

    // Create edges with vector properties
    db.execute_cypher(
        "MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'}) \
         CREATE (a)-[r:KNOWS {embedding: [0.9, 0.1, 0.0]}]->(b)",
    )
    .expect("create edge a->b");
    db.execute_cypher(
        "MATCH (a:User {name: 'Alice'}), (c:User {name: 'Charlie'}) \
         CREATE (a)-[r:KNOWS {embedding: [0.1, 0.9, 0.0]}]->(c)",
    )
    .expect("create edge a->c");

    // Query with vector filter on edge — should use EdgeVectorSearch
    let results = db
        .execute_cypher(
            "MATCH (a:User {name: 'Alice'})-[r:KNOWS]->(f) \
             WHERE vector_distance(r.embedding, [1.0, 0.0, 0.0]) < 0.5 \
             RETURN f.name",
        )
        .expect("edge vector query");

    // Alice->Bob has embedding [0.9, 0.1, 0.0], distance to [1.0, 0.0, 0.0] is small
    // Alice->Charlie has embedding [0.1, 0.9, 0.0], distance is larger
    // With threshold < 0.5, at least Alice->Bob should match
    // (Exact filtering depends on vector distance implementation)
    assert!(
        !results.is_empty(),
        "edge vector query should return at least one result"
    );
}

/// EXPLAIN SUGGEST for edge vector shows strategy in plan.
#[test]
fn explain_suggest_edge_vector_strategy() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let result = db
        .explain_suggest(
            "MATCH (u:User)-[r:REVIEWED]->(p:Product) \
             WHERE vector_similarity(r.sentiment_vec, [0.8, 0.2]) > 0.7 \
             RETURN p.name",
        )
        .expect("explain_suggest");

    // Plan should contain EdgeVectorSearch
    assert!(
        result.explain.contains("EdgeVectorSearch"),
        "plan should show EdgeVectorSearch: {}",
        result.explain
    );
}

/// Unknown procedure returns error.
#[test]
fn call_unknown_procedure_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let result = db.execute_cypher("CALL db.nonexistent.procedure()");
    assert!(result.is_err(), "unknown procedure should return error");
}

/// EXPLAIN uses real storage statistics (node counts, fan-out) for
/// more accurate cost estimates than hardcoded defaults.
#[test]
fn explain_uses_real_storage_stats() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Get cost estimate on empty database (uses stats: 0 nodes)
    let explain_empty = db
        .explain_cypher("MATCH (n:User) RETURN n")
        .expect("explain empty");

    // Insert 50 User nodes
    for i in 0..50 {
        db.execute_cypher(&format!(
            "CREATE (n:User {{name: 'user{}', age: {}}})",
            i,
            20 + i
        ))
        .expect("create user");
    }

    // Insert 10 Post nodes
    for i in 0..10 {
        db.execute_cypher(&format!("CREATE (n:Post {{title: 'post{}'}})", i))
            .expect("create post");
    }

    // Get cost estimate with real data
    let explain_with_data = db
        .explain_cypher("MATCH (n:User) RETURN n")
        .expect("explain with data");

    // The estimates should differ — empty DB vs 50 User nodes
    // Extract "Estimated rows:" from EXPLAIN output
    let extract_rows = |explain: &str| -> f64 {
        let marker = "Estimated rows: ";
        let start = explain.find(marker).expect("should have estimated rows") + marker.len();
        let end = explain[start..]
            .find(' ')
            .map(|p| start + p)
            .unwrap_or(explain.len());
        explain[start..end].parse::<f64>().expect("parse rows")
    };

    let rows_empty = extract_rows(&explain_empty);
    let rows_with_data = extract_rows(&explain_with_data);

    // With 50 User nodes and 10 Post nodes, the estimate should be
    // close to 50 (not the default 200 = 1000/5)
    assert!(
        rows_with_data < 200.0,
        "with real stats, estimated rows ({rows_with_data}) should be \
         less than default ({rows_empty})"
    );
    assert!(
        (rows_with_data - 50.0).abs() < 1.0,
        "estimated rows ({rows_with_data}) should be ~50 for 50 User nodes"
    );
}

/// EXPLAIN SUGGEST also uses real storage statistics.
#[test]
fn explain_suggest_uses_real_stats() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Insert data
    for i in 0..20 {
        db.execute_cypher(&format!("CREATE (n:Customer {{email: 'c{}@test.com'}})", i))
            .expect("create customer");
    }

    let result = db
        .explain_suggest("MATCH (n:Customer) WHERE n.email = 'c5@test.com' RETURN n")
        .expect("explain_suggest");

    // Should reflect real stats (20 customers) not defaults (200=1000/5)
    // NodeScan(Customer) = 20, Filter selectivity 0.33 → 20 × 0.33 ≈ 7
    // Default would be 200 × 0.33 ≈ 66
    let marker = "Estimated rows: ";
    let start = result.explain.find(marker).expect("rows") + marker.len();
    let end = result.explain[start..]
        .find(' ')
        .map(|p| start + p)
        .unwrap_or(result.explain.len());
    let rows: f64 = result.explain[start..end].parse().expect("parse");
    assert!(
        rows < 20.0,
        "SUGGEST should use real node count (20 customers), \
         estimated rows ({rows}) should be < 20: {}",
        result.explain
    );
}

/// EXPLAIN correctly reflects deleted nodes (MVCC tombstones excluded).
#[test]
fn explain_stats_exclude_deleted_nodes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create 10 nodes
    for i in 0..10 {
        db.execute_cypher(&format!("CREATE (n:Item {{id: {}}})", i))
            .expect("create");
    }

    // Delete 7 of them
    for i in 0..7 {
        db.execute_cypher(&format!("MATCH (n:Item {{id: {}}}) DELETE n", i))
            .expect("delete");
    }

    let explain = db
        .explain_cypher("MATCH (n:Item) RETURN n")
        .expect("explain after delete");

    // Should show ~3 estimated rows (only 3 remaining), not 10
    let marker = "Estimated rows: ";
    let start = explain.find(marker).expect("rows marker") + marker.len();
    let end = explain[start..]
        .find(' ')
        .map(|p| start + p)
        .unwrap_or(explain.len());
    let rows: f64 = explain[start..end].parse().expect("parse");

    assert!(
        (rows - 3.0).abs() < 1.0,
        "after deleting 7/10, estimated rows ({rows}) should be ~3: {explain}"
    );
}

/// EXPLAIN reflects real fan-out from adjacency data.
#[test]
fn explain_uses_real_fan_out() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create a small social graph: each user follows 3 others
    for i in 0..10 {
        db.execute_cypher(&format!("CREATE (n:Person {{name: 'p{}'}})", i))
            .expect("create person");
    }
    for i in 0..10 {
        for j in 1..=3 {
            let target = (i + j) % 10;
            db.execute_cypher(&format!(
                "MATCH (a:Person {{name: 'p{}'}}), (b:Person {{name: 'p{}'}}) \
                 CREATE (a)-[:FOLLOWS]->(b)",
                i, target
            ))
            .expect("create follow");
        }
    }

    let explain = db
        .explain_cypher("MATCH (a:Person)-[:FOLLOWS]->(b:Person) RETURN b.name")
        .expect("explain traversal");

    // The EXPLAIN should use real fan-out (~3) instead of default 50
    // This means estimated rows should be much less with real stats
    let marker = "Estimated rows: ";
    let start = explain.find(marker).expect("should have rows") + marker.len();
    let end = explain[start..]
        .find(' ')
        .map(|p| start + p)
        .unwrap_or(explain.len());
    let rows: f64 = explain[start..end].parse().expect("parse");

    // With 10 persons and fan-out 3, traversal should estimate ~30 rows
    // Default (10 persons × 50 fan-out = 500) would be much higher
    assert!(
        rows < 100.0,
        "with real fan-out ~3, estimated rows ({rows}) should be < 100 \
         (default fan-out 50 would give ~500). EXPLAIN:\n{explain}"
    );
}

/// Stats fallback: querying a label not in stats uses total/label_count.
/// Querying an edge type not in stats uses overall avg_fan_out.
#[test]
fn explain_stats_fallback_unknown_label_and_edge_type() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create data for one label + one edge type
    for i in 0..20 {
        db.execute_cypher(&format!("CREATE (n:Known {{id: {}}})", i))
            .expect("create known");
    }
    for i in 0..20 {
        let target = (i + 1) % 20;
        db.execute_cypher(&format!(
            "MATCH (a:Known {{id: {}}}), (b:Known {{id: {}}}) \
             CREATE (a)-[:HAS]->(b)",
            i, target
        ))
        .expect("create edge");
    }

    // Query with an UNKNOWN label — stats has no :Ghost nodes
    let explain_unknown_label = db
        .explain_cypher("MATCH (n:Ghost) RETURN n")
        .expect("explain unknown label");

    let extract_rows = |explain: &str| -> f64 {
        let marker = "Estimated rows: ";
        let start = explain.find(marker).expect("rows") + marker.len();
        let end = explain[start..]
            .find(' ')
            .map(|p| start + p)
            .unwrap_or(explain.len());
        explain[start..end].parse::<f64>().expect("parse")
    };

    let ghost_rows = extract_rows(&explain_unknown_label);
    // :Ghost not in stats → fallback to total_nodes / label_count = 20 / 1 = 20
    assert!(
        ghost_rows > 0.0 && ghost_rows <= 20.0,
        "unknown label should fallback to total/labels ({ghost_rows}): {explain_unknown_label}"
    );

    // Query with an UNKNOWN edge type
    let explain_unknown_edge = db
        .explain_cypher("MATCH (a:Known)-[:MYSTERY]->(b) RETURN b")
        .expect("explain unknown edge type");

    let mystery_rows = extract_rows(&explain_unknown_edge);
    // :MYSTERY not in stats → fallback to overall avg_fan_out (~1.0 for HAS)
    // 20 Known nodes × 1.0 fan-out = ~20 rows
    // If it had fallen back to hardcoded 50 → would be ~1000
    assert!(
        mystery_rows < 200.0,
        "unknown edge type should fallback to overall avg_fan_out \
         ({mystery_rows}), not hardcoded 50: {explain_unknown_edge}"
    );
}

/// Stats with mixed read-write: EXPLAIN after interleaved creates and deletes.
#[test]
fn explain_stats_after_interleaved_writes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create 30 nodes, delete 10, create 5 more = 25 remaining
    for i in 0..30 {
        db.execute_cypher(&format!("CREATE (n:Widget {{id: {}}})", i))
            .expect("create");
    }
    for i in 0..10 {
        db.execute_cypher(&format!("MATCH (n:Widget {{id: {}}}) DELETE n", i))
            .expect("delete");
    }
    for i in 100..105 {
        db.execute_cypher(&format!("CREATE (n:Widget {{id: {}}})", i))
            .expect("create more");
    }

    let explain = db
        .explain_cypher("MATCH (n:Widget) RETURN n")
        .expect("explain");

    let marker = "Estimated rows: ";
    let start = explain.find(marker).expect("rows") + marker.len();
    let end = explain[start..]
        .find(' ')
        .map(|p| start + p)
        .unwrap_or(explain.len());
    let rows: f64 = explain[start..end].parse().expect("parse");

    assert!(
        (rows - 25.0).abs() < 1.0,
        "after 30 create - 10 delete + 5 create = 25 widgets, \
         estimated ({rows}): {explain}"
    );
}

// ── StorageStats TTL cache tests (G034) ────────────────────────────

/// Verify that two consecutive explain_cypher calls return consistent
/// results (cache hit path — second call should use cached stats).
#[test]
fn stats_cache_returns_consistent_results() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    for i in 0..20 {
        db.execute_cypher(&format!("CREATE (n:Cached {{id: {i}}})",))
            .expect("create");
    }

    let explain_a = db.explain_cypher("MATCH (n:Cached) RETURN n").expect("a");
    let explain_b = db.explain_cypher("MATCH (n:Cached) RETURN n").expect("b");

    // Both calls should produce the same cost estimate (cached stats).
    assert_eq!(
        explain_a, explain_b,
        "back-to-back EXPLAIN should be identical (cache hit)"
    );
}

/// Verify that writes auto-invalidate the cache so EXPLAIN reflects new data,
/// and that `invalidate_stats_cache()` also works for manual cache busting.
#[test]
fn stats_cache_auto_invalidates_on_writes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Seed 10 nodes.
    for i in 0..10 {
        db.execute_cypher(&format!("CREATE (n:Fresh {{id: {i}}})",))
            .expect("create");
    }

    // First EXPLAIN populates cache — should see ~10 nodes.
    let explain_10 = db.explain_cypher("MATCH (n:Fresh) RETURN n").expect("10");

    // Add 50 more nodes (auto-invalidates cache on each write).
    for i in 10..60 {
        db.execute_cypher(&format!("CREATE (n:Fresh {{id: {i}}})",))
            .expect("create more");
    }

    // Cache was auto-invalidated by writes — should see ~60 now.
    let explain_60 = db.explain_cypher("MATCH (n:Fresh) RETURN n").expect("60");
    assert_ne!(
        explain_10, explain_60,
        "after writes, stats should reflect new data"
    );

    // Parse estimated rows.
    let marker = "Estimated rows: ";
    let start = explain_60.find(marker).expect("rows") + marker.len();
    let end = explain_60[start..]
        .find(' ')
        .map(|p| start + p)
        .unwrap_or(explain_60.len());
    let rows: f64 = explain_60[start..end].parse().expect("parse");
    assert!(
        (rows - 60.0).abs() < 1.0,
        "after writes, 60 Fresh nodes expected, got {rows}: {explain_60}"
    );

    // Manual invalidation also works (produces same stats since no data changed).
    db.invalidate_stats_cache();
    let explain_manual = db
        .explain_cypher("MATCH (n:Fresh) RETURN n")
        .expect("manual");
    assert_eq!(
        explain_60, explain_manual,
        "manual invalidation should produce same stats (no data change)"
    );
}

/// Verify that `compute_stats()` returns `Some` on a fresh empty DB.
#[test]
fn compute_stats_works_on_empty_db() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    let stats = db.compute_stats();
    assert!(stats.is_some(), "compute_stats should succeed on empty DB");
}

/// With TTL=0, every EXPLAIN recomputes stats — no caching.
/// Verify that a direct engine write (bypassing execute_cypher, so no
/// auto-invalidation) is still picked up because TTL is zero.
#[test]
fn stats_cache_ttl_zero_always_recomputes() {
    use coordinode_core::graph::stats::StorageStats;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.set_stats_ttl(std::time::Duration::ZERO);

    // Seed 10 nodes via normal path.
    for i in 0..10 {
        db.execute_cypher(&format!("CREATE (n:ZeroTTL {{id: {i}}})"))
            .expect("create");
    }

    // Populate cache.
    let stats_10 = db.compute_stats().expect("stats after 10");
    assert_eq!(stats_10.total_node_count(), 10);

    // Write 5 more nodes directly through storage engine (bypasses
    // execute_cypher, so no auto-invalidation fires).
    {
        use coordinode_core::graph::node::{encode_node_key, NodeId, NodeRecord};
        use coordinode_storage::engine::partition::Partition;

        for raw_id in 9000..9005u64 {
            let rec = NodeRecord {
                labels: vec!["ZeroTTL".to_string()],
                props: std::collections::HashMap::new(),
                extra: None,
            };
            let key = encode_node_key(1, NodeId::from_raw(raw_id));
            let val = rec.to_msgpack().expect("encode");
            db.engine()
                .put(Partition::Node, &key, &val)
                .expect("direct write");
        }
    }

    // With TTL=0, compute_stats recomputes immediately — sees 15 nodes.
    let stats_15 = db.compute_stats().expect("stats after direct write");
    assert_eq!(
        stats_15.total_node_count(),
        15,
        "TTL=0 should recompute: direct writes visible immediately"
    );
}

/// With TTL=MAX, cache never expires — stale data persists until
/// explicit invalidation.
#[test]
fn stats_cache_ttl_max_stays_stale_until_invalidation() {
    use coordinode_core::graph::stats::StorageStats;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.set_stats_ttl(std::time::Duration::MAX);

    // Seed 10 nodes.
    for i in 0..10 {
        db.execute_cypher(&format!("CREATE (n:MaxTTL {{id: {i}}})"))
            .expect("create");
    }

    // Populate cache (auto-invalidation fires on last write, so this
    // is the first compute after all writes — caches 10 nodes).
    let stats_10 = db.compute_stats().expect("stats 10");
    assert_eq!(stats_10.total_node_count(), 10);

    // Write 5 more via direct storage engine (bypasses auto-invalidation).
    {
        use coordinode_core::graph::node::{encode_node_key, NodeId, NodeRecord};
        use coordinode_storage::engine::partition::Partition;

        for raw_id in 8000..8005u64 {
            let rec = NodeRecord {
                labels: vec!["MaxTTL".to_string()],
                props: std::collections::HashMap::new(),
                extra: None,
            };
            let key = encode_node_key(1, NodeId::from_raw(raw_id));
            let val = rec.to_msgpack().expect("encode");
            db.engine()
                .put(Partition::Node, &key, &val)
                .expect("direct write");
        }
    }

    // TTL=MAX → cache still returns stale 10.
    let stats_stale = db.compute_stats().expect("stale");
    assert_eq!(
        stats_stale.total_node_count(),
        10,
        "TTL=MAX should keep cached value, not see direct writes"
    );

    // Manual invalidation → sees 15.
    db.invalidate_stats_cache();
    let stats_fresh = db.compute_stats().expect("fresh after invalidation");
    assert_eq!(
        stats_fresh.total_node_count(),
        15,
        "after invalidation, should see all 15 nodes"
    );
}

// === G022: MissingIndex false-positive prevention (IndexRegistry cross-check) ===

/// When an index exists for (User, email), EXPLAIN SUGGEST should NOT suggest
/// CREATE INDEX for that property — preventing false positives.
#[test]
fn explain_suggest_no_false_positive_when_index_exists() {
    use coordinode_query::index::{IndexDefinition, IndexRegistry};
    use coordinode_storage::engine::config::StorageConfig;
    use coordinode_storage::engine::core::StorageEngine;

    let dir = tempfile::tempdir().expect("tempdir");

    // Step 1: Pre-populate storage with an index definition.
    // Open raw storage, register index, close.
    {
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open engine");
        let mut reg = IndexRegistry::new();
        reg.register(
            &engine,
            IndexDefinition::btree("user_email", "User", "email"),
        )
        .expect("register index");
        // engine dropped, data flushed
    }

    // Step 2: Open Database over the same directory.
    // Database::open calls IndexRegistry::load_all() which reads persisted indexes.
    let db = Database::open(dir.path()).expect("open db");

    // Step 3: EXPLAIN SUGGEST for a query that filters User.email.
    // Without G022 fix, this would suggest CREATE INDEX even though it exists.
    let result = db
        .explain_suggest("MATCH (n:User) WHERE n.email = 'test@example.com' RETURN n")
        .expect("explain_suggest");

    // No MissingIndex (CreateIndex) suggestion should appear.
    let has_create_index = result.suggestions.iter().any(|s| {
        s.kind == coordinode_query::advisor::SuggestionKind::CreateIndex
            && s.explanation.contains("email")
    });
    assert!(
        !has_create_index,
        "should NOT suggest CreateIndex for User.email when index exists. Suggestions: {:?}",
        result.suggestions
    );
}

/// When an index exists for (User, name) but NOT for (User, email),
/// EXPLAIN SUGGEST should suggest only for the missing one.
#[test]
fn explain_suggest_partial_coverage() {
    use coordinode_query::index::{IndexDefinition, IndexRegistry};
    use coordinode_storage::engine::config::StorageConfig;
    use coordinode_storage::engine::core::StorageEngine;

    let dir = tempfile::tempdir().expect("tempdir");

    // Pre-populate: index on User.name only
    {
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).expect("open engine");
        let mut reg = IndexRegistry::new();
        reg.register(&engine, IndexDefinition::btree("user_name", "User", "name"))
            .expect("register");
    }

    let db = Database::open(dir.path()).expect("open db");

    // Query filters User.email (not indexed)
    let result = db
        .explain_suggest("MATCH (n:User) WHERE n.email = 'test@example.com' RETURN n")
        .expect("explain_suggest");

    // Should still suggest for email (not indexed)
    let has_email_suggestion = result
        .suggestions
        .iter()
        .any(|s| s.explanation.contains("email"));
    assert!(
        has_email_suggestion,
        "should suggest CreateIndex for User.email (not indexed). Suggestions: {:?}",
        result.suggestions
    );
}
