use super::*;

fn make_ctx() -> AdvisorContext {
    AdvisorContext {
        registry: Arc::new(QueryRegistry::new()),
        nplus1: Arc::new(NPlus1Detector::new()),
        dismissed: Arc::new(DismissedSet::new()),
    }
}

/// The value of `column` in `row`, located through the procedure's signature.
fn column(kind: Kind, row: &ProcedureRow, column: &str) -> Value {
    let procedure = AdvisorProcedure::new(kind);
    let index = procedure
        .signature()
        .output_index(column)
        .unwrap_or_else(|| panic!("no output {column}"));
    row[index].clone()
}

/// Every returned row has exactly one value per declared output.
fn assert_shape(kind: Kind, rows: &[ProcedureRow]) {
    let width = AdvisorProcedure::new(kind).signature().outputs.len();
    assert!(rows.iter().all(|r| r.len() == width));
}

/// suggestions() returns empty when no queries recorded.
#[test]
fn suggestions_empty() {
    let ctx = make_ctx();
    assert!(suggestions(&ctx).is_empty());
}

/// suggestions() returns rows after recording queries.
#[test]
fn suggestions_with_data() {
    let ctx = make_ctx();
    ctx.registry
        .record(0xABC, "MATCH (n:User) RETURN n", 50_000);
    ctx.registry
        .record(0xABC, "MATCH (n:User) RETURN n", 60_000);

    let rows = suggestions(&ctx);
    assert!(!rows.is_empty(), "should have suggestions after recording");
    assert_shape(Kind::Suggestions, &rows);
    assert_eq!(
        column(Kind::Suggestions, &rows[0], "id"),
        Value::String("0000000000000abc".into())
    );
    assert_eq!(
        column(Kind::Suggestions, &rows[0], "query"),
        Value::String("MATCH (n:User) RETURN n".into())
    );
}

/// queryStats() returns stats for recorded queries.
#[test]
fn query_stats_with_data() {
    let ctx = make_ctx();
    ctx.registry.record(0x111, "MATCH (a) RETURN a", 100);
    ctx.registry.record(0x222, "CREATE (b:X)", 200);
    ctx.registry.record(0x111, "MATCH (a) RETURN a", 150);

    let rows = query_stats(&ctx);
    assert_eq!(rows.len(), 2, "two distinct fingerprints");
    assert_shape(Kind::QueryStats, &rows);

    // First row should be the most frequently executed
    assert_eq!(column(Kind::QueryStats, &rows[0], "count"), Value::Int(2));
    // CE always returns shardsUsed=1
    assert_eq!(
        column(Kind::QueryStats, &rows[0], "shardsUsed"),
        Value::Int(1)
    );
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

    let rows = query_stats(&ctx);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        column(Kind::QueryStats, &rows[0], "plan"),
        Value::String("NodeScan (User)".to_string()),
        "plan should be stored and returned"
    );
}

/// slowQueries() filters by p99 threshold.
#[test]
fn slow_queries_filter() {
    let ctx = make_ctx();
    ctx.registry.record(0x111, "fast query", 10);
    ctx.registry.record(0x222, "slow query", 500_000);

    let rows = slow_queries(&ctx, &[Value::Int(10), Value::Int(1_000)]);
    assert_eq!(rows.len(), 1, "only the slow query above threshold");
    assert_shape(Kind::SlowQueries, &rows);
    assert_eq!(
        column(Kind::SlowQueries, &rows[0], "query"),
        Value::String("slow query".into())
    );
}

/// slowQueries() with the defaults the signature fills in.
#[test]
fn slow_queries_defaults() {
    let ctx = make_ctx();
    ctx.registry.record(0x111, "query", 10);

    // The p99 of the bucket holding 10μs is 25μs, under the 100μs default.
    let rows = slow_queries(&ctx, &[Value::Int(20), Value::Int(100)]);
    assert!(rows.is_empty());
}

/// A negative limit returns nothing instead of wrapping to a huge one.
#[test]
fn slow_queries_negative_limit_returns_nothing() {
    let ctx = make_ctx();
    ctx.registry.record(0x222, "slow query", 500_000);
    assert!(slow_queries(&ctx, &[Value::Int(-1), Value::Int(0)]).is_empty());
}

/// dismiss() marks a fingerprint as dismissed.
#[test]
fn dismiss_and_check() {
    let ctx = make_ctx();
    ctx.registry.record(0xABC, "MATCH (n) RETURN n", 50_000);

    assert!(!suggestions(&ctx).is_empty());

    let result = dismiss(&ctx, &[Value::String("0000000000000abc".to_string())]).unwrap();
    assert_eq!(result.len(), 1);
    assert_shape(Kind::Dismiss, &result);
    assert_eq!(
        column(Kind::Dismiss, &result[0], "dismissed"),
        Value::Bool(true)
    );

    assert!(
        suggestions(&ctx).is_empty(),
        "dismissed fingerprint should be excluded"
    );
}

/// reset() clears everything.
#[test]
fn reset_clears_all() {
    let ctx = make_ctx();
    ctx.registry.record(0xABC, "query", 100);
    ctx.dismissed.dismiss(0xABC);

    let rows = reset(&ctx);
    assert_shape(Kind::Reset, &rows);

    assert_eq!(ctx.registry.fingerprint_count(), 0);
    assert!(!ctx.dismissed.is_dismissed(0xABC));
}

/// dismiss() with invalid hex is refused, naming the argument.
#[test]
fn dismiss_invalid_hex() {
    let ctx = make_ctx();
    let err = dismiss(&ctx, &[Value::String("not-hex".to_string())]).unwrap_err();
    assert!(matches!(
        err,
        ProcedureError::InvalidArgument { ref argument, .. } if argument == "id"
    ));
}

/// dismiss(null) is refused rather than dismissing nothing silently.
#[test]
fn dismiss_null_is_refused() {
    let ctx = make_ctx();
    assert!(dismiss(&ctx, &[Value::Null]).is_err());
}

/// Each advisor procedure lists the mode that matches what it touches:
/// reading the registry is READ, changing this node's advisor state is DBMS.
#[test]
fn advisor_modes() {
    for (kind, mode) in [
        (Kind::Suggestions, ProcedureMode::Read),
        (Kind::QueryStats, ProcedureMode::Read),
        (Kind::SlowQueries, ProcedureMode::Read),
        (Kind::Dismiss, ProcedureMode::Dbms),
        (Kind::Reset, ProcedureMode::Dbms),
    ] {
        assert_eq!(AdvisorProcedure::new(kind).signature().mode, mode);
    }
}
