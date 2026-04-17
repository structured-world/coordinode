//! Built-in advisor procedures: `db.advisor.*`.
//!
//! These procedures expose query performance data through the Cypher CALL syntax:
//! - `db.advisor.suggestions()` — top 10 suggestions ranked by impact
//! - `db.advisor.queryStats()` — top 100 query fingerprints by count
//! - `db.advisor.slowQueries(limit, minTime)` — queries above P99 threshold
//! - `db.advisor.dismiss(id)` — suppress a suggestion by fingerprint
//! - `db.advisor.reset()` — clear all advisor state

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use coordinode_core::graph::types::Value;

use super::nplus1::NPlus1Detector;
use super::registry::QueryRegistry;
use super::suggest::{Severity, Suggestion, SuggestionKind};

/// Manages dismissed suggestion fingerprints.
///
/// When a suggestion is dismissed via `db.advisor.dismiss(fingerprint)`,
/// it won't appear in `db.advisor.suggestions()` until `db.advisor.reset()`.
///
/// Multi-instance note: dismissed set is per-node in-memory state.
/// In a 3-node CE cluster, dismissals are node-local.
pub struct DismissedSet {
    fingerprints: Mutex<HashSet<u64>>,
}

impl DismissedSet {
    pub fn new() -> Self {
        Self {
            fingerprints: Mutex::new(HashSet::new()),
        }
    }

    /// Dismiss a fingerprint. Returns true if newly dismissed.
    pub fn dismiss(&self, fingerprint: u64) -> bool {
        let mut set = self.fingerprints.lock().unwrap_or_else(|e| e.into_inner());
        set.insert(fingerprint)
    }

    /// Check if a fingerprint is dismissed.
    pub fn is_dismissed(&self, fingerprint: u64) -> bool {
        let set = self.fingerprints.lock().unwrap_or_else(|e| e.into_inner());
        set.contains(&fingerprint)
    }

    /// Clear all dismissals.
    pub fn reset(&self) {
        let mut set = self.fingerprints.lock().unwrap_or_else(|e| e.into_inner());
        set.clear();
    }
}

impl Default for DismissedSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Context passed to procedure execution.
pub struct ProcedureContext {
    pub registry: Arc<QueryRegistry>,
    pub nplus1: Arc<NPlus1Detector>,
    pub dismissed: Arc<DismissedSet>,
}

/// A single row of procedure output: column name → value.
pub type ProcedureRow = Vec<(String, Value)>;

/// Execute a named procedure and return result rows.
///
/// Returns `Ok(rows)` on success, or `Err(message)` for unknown procedures
/// or invalid arguments.
pub fn execute_procedure(
    procedure: &str,
    args: &[Value],
    ctx: &ProcedureContext,
) -> Result<Vec<ProcedureRow>, String> {
    match procedure {
        "db.advisor.suggestions" => exec_suggestions(ctx),
        "db.advisor.queryStats" => exec_query_stats(ctx),
        "db.advisor.slowQueries" => exec_slow_queries(args, ctx),
        "db.advisor.dismiss" => exec_dismiss(args, ctx),
        "db.advisor.reset" => exec_reset(ctx),
        _ => Err(format!("unknown procedure: {procedure}")),
    }
}

/// `db.advisor.suggestions()` — top 10 suggestions ranked by impact score.
///
/// For each tracked fingerprint, runs the detectors to find suggestions,
/// then ranks by impact (count × p99) and returns the top 10.
///
/// YIELD: id, severity, kind, query, explanation, ddl, impact, sources
fn exec_suggestions(ctx: &ProcedureContext) -> Result<Vec<ProcedureRow>, String> {
    let top = ctx.registry.top_by_impact(100);
    let mut all_suggestions: Vec<(
        u64,
        f64,
        Suggestion,
        String,
        Vec<super::SourceLocationSnapshot>,
    )> = Vec::new();

    // For each fingerprint, generate suggestions from detectors
    for stats in &top {
        if ctx.dismissed.is_dismissed(stats.fingerprint) {
            continue;
        }

        let impact = stats.count as f64 * stats.p99_time_us as f64;

        // N+1 detection suggestions
        // (N+1 alerts are tracked separately by the detector)

        // Missing index / general suggestions are plan-based (EXPLAIN SUGGEST).
        // For db.advisor.suggestions(), we use registry-level heuristics:
        // High-impact queries with high count + latency.
        // The actual suggestion detectors require a plan, which we don't have
        // from the registry alone. So we report the query stats as suggestions
        // with kind "HighImpact" when they exceed thresholds.

        // Check for N+1 alerts on this fingerprint
        let nplus1_alerts = ctx.nplus1.active_alerts();
        for alert in &nplus1_alerts {
            if alert.fingerprint == stats.fingerprint {
                all_suggestions.push((
                    stats.fingerprint,
                    impact,
                    alert.suggestion.clone(),
                    stats.canonical_query.clone(),
                    stats.sources.clone(),
                ));
            }
        }

        // If this is a high-impact query (top by impact), report it
        if impact > 0.0 {
            all_suggestions.push((
                stats.fingerprint,
                impact,
                Suggestion::new(
                    SuggestionKind::CreateIndex,
                    if stats.p99_time_us > 100_000 {
                        Severity::Critical
                    } else if stats.p99_time_us > 10_000 {
                        Severity::Warning
                    } else {
                        Severity::Info
                    },
                    format!(
                        "Query executed {}× with p99 {}μs — run EXPLAIN SUGGEST to see specific recommendations",
                        stats.count,
                        stats.p99_time_us,
                    ),
                ),
                stats.canonical_query.clone(),
                stats.sources.clone(),
            ));
        }
    }

    // Sort by impact descending, take top 10
    all_suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    all_suggestions.truncate(10);

    let rows = all_suggestions
        .into_iter()
        .map(|(fp, impact, suggestion, query, sources)| {
            vec![
                ("id".to_string(), Value::String(format!("{fp:016x}"))),
                (
                    "severity".to_string(),
                    Value::String(suggestion.severity.to_string()),
                ),
                (
                    "kind".to_string(),
                    Value::String(suggestion.kind.to_string()),
                ),
                ("query".to_string(), Value::String(query)),
                (
                    "explanation".to_string(),
                    Value::String(suggestion.explanation),
                ),
                (
                    "ddl".to_string(),
                    suggestion.ddl.map(Value::String).unwrap_or(Value::Null),
                ),
                ("impact".to_string(), Value::Float(impact)),
                (
                    "sources".to_string(),
                    Value::Array(
                        sources
                            .iter()
                            .map(|s| {
                                Value::String(format!(
                                    "{}:{}:{} ({}×)",
                                    s.file, s.line, s.function, s.call_count
                                ))
                            })
                            .collect(),
                    ),
                ),
            ]
        })
        .collect();

    Ok(rows)
}

/// `db.advisor.queryStats()` — top 100 query fingerprints by execution count.
///
/// YIELD: fingerprint, query, count, avgTime, p99Time, plan, shardsUsed, sources
fn exec_query_stats(ctx: &ProcedureContext) -> Result<Vec<ProcedureRow>, String> {
    let top = ctx.registry.top_by_count(100);

    let rows = top
        .into_iter()
        .map(|stats| {
            let avg_time = stats.total_time_us.checked_div(stats.count).unwrap_or(0);
            vec![
                (
                    "fingerprint".to_string(),
                    Value::String(format!("{:016x}", stats.fingerprint)),
                ),
                ("query".to_string(), Value::String(stats.canonical_query)),
                ("count".to_string(), Value::Int(stats.count as i64)),
                ("avgTime".to_string(), Value::Int(avg_time as i64)),
                ("p99Time".to_string(), Value::Int(stats.p99_time_us as i64)),
                (
                    "plan".to_string(),
                    stats.last_plan.map(Value::String).unwrap_or(Value::Null),
                ),
                // CE is single-shard; EE will populate this from scatter-gather stats.
                ("shardsUsed".to_string(), Value::Int(1)),
                (
                    "sources".to_string(),
                    Value::Array(
                        stats
                            .sources
                            .iter()
                            .map(|s| {
                                Value::String(format!(
                                    "{}:{}:{} ({}×)",
                                    s.file, s.line, s.function, s.call_count
                                ))
                            })
                            .collect(),
                    ),
                ),
            ]
        })
        .collect();

    Ok(rows)
}

/// `db.advisor.slowQueries(limit, minTime)` — queries with p99 above threshold.
///
/// Arguments:
/// - limit (optional, default 20): max results
/// - minTime (optional, default 100μs): minimum p99 threshold in microseconds
///
/// YIELD: query, p99Time, count, sources
fn exec_slow_queries(args: &[Value], ctx: &ProcedureContext) -> Result<Vec<ProcedureRow>, String> {
    let limit = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        None => 20,
        _ => return Err("slowQueries: first argument (limit) must be an integer".into()),
    };

    let min_time = match args.get(1) {
        Some(Value::Int(n)) => *n as u64,
        None => 100,
        _ => return Err("slowQueries: second argument (minTime) must be an integer".into()),
    };

    let top = ctx.registry.top_by_latency(limit, min_time);

    let rows = top
        .into_iter()
        .map(|stats| {
            vec![
                ("query".to_string(), Value::String(stats.canonical_query)),
                ("p99Time".to_string(), Value::Int(stats.p99_time_us as i64)),
                ("count".to_string(), Value::Int(stats.count as i64)),
                (
                    "plan".to_string(),
                    stats.last_plan.map(Value::String).unwrap_or(Value::Null),
                ),
                (
                    "sources".to_string(),
                    Value::Array(
                        stats
                            .sources
                            .iter()
                            .map(|s| {
                                Value::String(format!(
                                    "{}:{}:{} ({}×)",
                                    s.file, s.line, s.function, s.call_count
                                ))
                            })
                            .collect(),
                    ),
                ),
            ]
        })
        .collect();

    Ok(rows)
}

/// `db.advisor.dismiss(id)` — dismiss a suggestion by fingerprint hex ID.
///
/// The dismissed suggestion won't appear in `db.advisor.suggestions()`
/// until `db.advisor.reset()` is called.
fn exec_dismiss(args: &[Value], ctx: &ProcedureContext) -> Result<Vec<ProcedureRow>, String> {
    let id = match args.first() {
        Some(Value::String(s)) => s,
        _ => return Err("dismiss: argument must be a string fingerprint ID".into()),
    };

    let fingerprint = u64::from_str_radix(id.trim_start_matches("0x"), 16)
        .map_err(|e| format!("dismiss: invalid fingerprint hex: {e}"))?;

    let was_new = ctx.dismissed.dismiss(fingerprint);

    Ok(vec![vec![
        ("id".to_string(), Value::String(id.clone())),
        ("dismissed".to_string(), Value::Bool(was_new)),
    ]])
}

/// `db.advisor.reset()` — clear all advisor state.
///
/// Resets the fingerprint registry, N+1 detector, and dismissed set.
fn exec_reset(ctx: &ProcedureContext) -> Result<Vec<ProcedureRow>, String> {
    ctx.registry.reset();
    ctx.nplus1.reset();
    ctx.dismissed.reset();

    Ok(vec![vec![(
        "status".to_string(),
        Value::String("OK".to_string()),
    )]])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
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
}
