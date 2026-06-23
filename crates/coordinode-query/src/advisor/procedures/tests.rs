use super::*;

fn make_ctx() -> ProcedureContext {
    ProcedureContext {
        registry: Arc::new(QueryRegistry::new()),
        nplus1: Arc::new(NPlus1Detector::new()),
        dismissed: Arc::new(DismissedSet::new()),
    }
}

/// Unknown procedure returns error.
#[test]
fn unknown_procedure_error() {
    let ctx = make_ctx();
    let result = execute_procedure("db.unknown.foo", &[], &ctx);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("unknown procedure"));
}

/// suggestions() returns empty when no queries recorded.
#[test]
fn suggestions_empty() {
    let ctx = make_ctx();
    let rows = execute_procedure("db.advisor.suggestions", &[], &ctx).unwrap();
    assert!(rows.is_empty());
}

/// suggestions() returns rows after recording queries.
#[test]
fn suggestions_with_data() {
    let ctx = make_ctx();
    ctx.registry
        .record(0xABC, "MATCH (n:User) RETURN n", 50_000);
    ctx.registry
        .record(0xABC, "MATCH (n:User) RETURN n", 60_000);

    let rows = execute_procedure("db.advisor.suggestions", &[], &ctx).unwrap();
    assert!(!rows.is_empty(), "should have suggestions after recording");

    // Check column names in first row
    let first = &rows[0];
    let col_names: Vec<&str> = first.iter().map(|(name, _)| name.as_str()).collect();
    assert!(col_names.contains(&"id"));
    assert!(col_names.contains(&"severity"));
    assert!(col_names.contains(&"kind"));
    assert!(col_names.contains(&"query"));
    assert!(col_names.contains(&"explanation"));
    assert!(col_names.contains(&"impact"));
}

/// queryStats() returns stats for recorded queries.
#[test]
fn query_stats_with_data() {
    let ctx = make_ctx();
    ctx.registry.record(0x111, "MATCH (a) RETURN a", 100);
    ctx.registry.record(0x222, "CREATE (b:X)", 200);
    ctx.registry.record(0x111, "MATCH (a) RETURN a", 150);

    let rows = execute_procedure("db.advisor.queryStats", &[], &ctx).unwrap();
    assert_eq!(rows.len(), 2, "two distinct fingerprints");

    // First row should be the most frequently executed
    let first_count = rows[0]
        .iter()
        .find(|(k, _)| k == "count")
        .map(|(_, v)| v.clone());
    assert_eq!(first_count, Some(Value::Int(2)));

    // Verify plan and shardsUsed columns exist
    let col_names: Vec<&str> = rows[0].iter().map(|(k, _)| k.as_str()).collect();
    assert!(col_names.contains(&"plan"), "should have plan column");
    assert!(
        col_names.contains(&"shardsUsed"),
        "should have shardsUsed column"
    );
    // CE always returns shardsUsed=1
    let shards = rows[0]
        .iter()
        .find(|(k, _)| k == "shardsUsed")
        .map(|(_, v)| v.clone());
    assert_eq!(shards, Some(Value::Int(1)));
}

/// queryStats() includes plan when recorded with record_with_plan.
#[test]
fn query_stats_includes_plan() {
    let ctx = make_ctx();
    ctx.registry.record_with_plan(
        0x111,
        "MATCH (n:User) RETURN n",
        100,
        "NodeScan (User)".to_string(),
        None,
    );

    let rows = execute_procedure("db.advisor.queryStats", &[], &ctx).unwrap();
    assert_eq!(rows.len(), 1);
    let plan = rows[0]
        .iter()
        .find(|(k, _)| k == "plan")
        .map(|(_, v)| v.clone());
    assert_eq!(
        plan,
        Some(Value::String("NodeScan (User)".to_string())),
        "plan should be stored and returned"
    );
}

/// slowQueries() filters by p99 threshold.
#[test]
fn slow_queries_filter() {
    let ctx = make_ctx();
    ctx.registry.record(0x111, "fast query", 10);
    ctx.registry.record(0x222, "slow query", 500_000);

    let rows = execute_procedure(
        "db.advisor.slowQueries",
        &[Value::Int(10), Value::Int(1_000)],
        &ctx,
    )
    .unwrap();
    assert_eq!(rows.len(), 1, "only the slow query above threshold");
}

/// slowQueries() with default args.
#[test]
fn slow_queries_defaults() {
    let ctx = make_ctx();
    ctx.registry.record(0x111, "query", 10);

    let rows = execute_procedure("db.advisor.slowQueries", &[], &ctx).unwrap();
    // 10μs is above default min 100μs? No — p99 of bucket containing 10 is 25μs < 100.
    // So no results expected.
    assert!(rows.is_empty());
}

/// dismiss() marks a fingerprint as dismissed.
#[test]
fn dismiss_and_check() {
    let ctx = make_ctx();
    ctx.registry.record(0xABC, "MATCH (n) RETURN n", 50_000);

    // Before dismiss: suggestions should include it
    let before = execute_procedure("db.advisor.suggestions", &[], &ctx).unwrap();
    assert!(!before.is_empty());

    // Dismiss
    let result = execute_procedure(
        "db.advisor.dismiss",
        &[Value::String("0000000000000abc".to_string())],
        &ctx,
    )
    .unwrap();
    assert_eq!(result.len(), 1);

    // After dismiss: suggestions should exclude it
    let after = execute_procedure("db.advisor.suggestions", &[], &ctx).unwrap();
    assert!(after.is_empty(), "dismissed fingerprint should be excluded");
}

/// reset() clears everything.
#[test]
fn reset_clears_all() {
    let ctx = make_ctx();
    ctx.registry.record(0xABC, "query", 100);
    ctx.dismissed.dismiss(0xABC);

    execute_procedure("db.advisor.reset", &[], &ctx).unwrap();

    assert_eq!(ctx.registry.fingerprint_count(), 0);
    assert!(!ctx.dismissed.is_dismissed(0xABC));
}

/// dismiss() with invalid hex returns error.
#[test]
fn dismiss_invalid_hex() {
    let ctx = make_ctx();
    let result = execute_procedure(
        "db.advisor.dismiss",
        &[Value::String("not-hex".to_string())],
        &ctx,
    );
    assert!(result.is_err());
}

/// dismiss() without args returns error.
#[test]
fn dismiss_no_args() {
    let ctx = make_ctx();
    let result = execute_procedure("db.advisor.dismiss", &[], &ctx);
    assert!(result.is_err());
}

/// slowQueries() with wrong arg type returns error.
#[test]
fn slow_queries_wrong_type() {
    let ctx = make_ctx();
    let result = execute_procedure(
        "db.advisor.slowQueries",
        &[Value::String("not-a-number".to_string())],
        &ctx,
    );
    assert!(result.is_err());
}
