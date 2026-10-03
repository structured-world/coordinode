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
use crate::executor::runner::{ExecutionContext, ExecutionError};
use crate::procedure::{
    FieldSignature, Procedure, ProcedureError, ProcedureMode, ProcedureSignature, ValueType,
};

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

/// The advisor state the `db.advisor.*` procedures read and reset.
pub struct AdvisorContext {
    pub registry: Arc<QueryRegistry>,
    pub nplus1: Arc<NPlus1Detector>,
    pub dismissed: Arc<DismissedSet>,
}

/// One returned row: one value per output column, in signature order.
pub type ProcedureRow = Vec<Value>;

/// The `db.advisor.*` procedures, for the built-in catalog.
pub(crate) fn procedures() -> Vec<Arc<dyn Procedure>> {
    vec![
        Arc::new(AdvisorProcedure::new(Kind::Suggestions)),
        Arc::new(AdvisorProcedure::new(Kind::QueryStats)),
        Arc::new(AdvisorProcedure::new(Kind::SlowQueries)),
        Arc::new(AdvisorProcedure::new(Kind::Dismiss)),
        Arc::new(AdvisorProcedure::new(Kind::Reset)),
    ]
}

/// Which advisor procedure.
#[derive(Clone, Copy)]
enum Kind {
    Suggestions,
    QueryStats,
    SlowQueries,
    Dismiss,
    Reset,
}

struct AdvisorProcedure {
    kind: Kind,
    signature: ProcedureSignature,
}

impl AdvisorProcedure {
    fn new(kind: Kind) -> Self {
        let sources = || ValueType::List(Box::new(ValueType::String));
        let signature = match kind {
            Kind::Suggestions => ProcedureSignature::new(
                "db.advisor.suggestions",
                ProcedureMode::Read,
                "The ten highest-impact suggestions for recorded queries.",
            )
            .output("id", ValueType::String)
            .output("severity", ValueType::String)
            .output("kind", ValueType::String)
            .output("query", ValueType::String)
            .output("explanation", ValueType::String)
            .output("ddl", ValueType::String)
            .output("impact", ValueType::Float)
            .output("sources", sources()),
            Kind::QueryStats => ProcedureSignature::new(
                "db.advisor.queryStats",
                ProcedureMode::Read,
                "The hundred most executed query shapes with their timings.",
            )
            .output("fingerprint", ValueType::String)
            .output("query", ValueType::String)
            .output("count", ValueType::Integer)
            .output("avgTime", ValueType::Integer)
            .output("p99Time", ValueType::Integer)
            .output("plan", ValueType::String)
            .output("shardsUsed", ValueType::Integer)
            .output("sources", sources()),
            Kind::SlowQueries => ProcedureSignature::new(
                "db.advisor.slowQueries",
                ProcedureMode::Read,
                "Query shapes whose p99 latency in microseconds is at least minTime.",
            )
            .input(FieldSignature::new("limit", ValueType::Integer).with_default(Value::Int(20)))
            .input(FieldSignature::new("minTime", ValueType::Integer).with_default(Value::Int(100)))
            .output("query", ValueType::String)
            .output("p99Time", ValueType::Integer)
            .output("count", ValueType::Integer)
            .output("plan", ValueType::String)
            .output("sources", sources()),
            Kind::Dismiss => ProcedureSignature::new(
                "db.advisor.dismiss",
                ProcedureMode::Dbms,
                "Hide a query shape's suggestions on this node until db.advisor.reset().",
            )
            .input(FieldSignature::new("id", ValueType::String))
            .output("id", ValueType::String)
            .output("dismissed", ValueType::Boolean),
            Kind::Reset => ProcedureSignature::new(
                "db.advisor.reset",
                ProcedureMode::Dbms,
                "Clear this node's recorded queries, N+1 alerts and dismissals.",
            )
            .output("status", ValueType::String),
        };
        Self { kind, signature }
    }
}

impl Procedure for AdvisorProcedure {
    fn signature(&self) -> &ProcedureSignature {
        &self.signature
    }

    fn call(
        &self,
        ctx: &mut ExecutionContext<'_>,
        args: Vec<Value>,
    ) -> Result<Vec<ProcedureRow>, ExecutionError> {
        let advisor = ctx.advisor.as_ref().ok_or_else(|| {
            ExecutionError::Unsupported("the query advisor is not available here".into())
        })?;
        Ok(match self.kind {
            Kind::Suggestions => suggestions(advisor),
            Kind::QueryStats => query_stats(advisor),
            Kind::SlowQueries => slow_queries(advisor, &args),
            Kind::Dismiss => dismiss(advisor, &args)?,
            Kind::Reset => reset(advisor),
        })
    }
}

/// `source:line:function (N×)` for each recorded call site.
fn sources_value(sources: &[super::SourceLocationSnapshot]) -> Value {
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
    )
}

/// `db.advisor.suggestions()` — top 10 suggestions ranked by impact score.
///
/// For each tracked fingerprint, runs the detectors to find suggestions,
/// then ranks by impact (count × p99) and returns the top 10.
pub(crate) fn suggestions(ctx: &AdvisorContext) -> Vec<ProcedureRow> {
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

    all_suggestions
        .into_iter()
        .map(|(fp, impact, suggestion, query, sources)| {
            vec![
                Value::String(format!("{fp:016x}")),
                Value::String(suggestion.severity.to_string()),
                Value::String(suggestion.kind.to_string()),
                Value::String(query),
                Value::String(suggestion.explanation),
                suggestion.ddl.map(Value::String).unwrap_or(Value::Null),
                Value::Float(impact),
                sources_value(&sources),
            ]
        })
        .collect()
}

/// `db.advisor.queryStats()` — top 100 query fingerprints by execution count.
pub(crate) fn query_stats(ctx: &AdvisorContext) -> Vec<ProcedureRow> {
    ctx.registry
        .top_by_count(100)
        .into_iter()
        .map(|stats| {
            let avg_time = stats.total_time_us.checked_div(stats.count).unwrap_or(0);
            vec![
                Value::String(format!("{:016x}", stats.fingerprint)),
                Value::String(stats.canonical_query),
                Value::Int(stats.count as i64),
                Value::Int(avg_time as i64),
                Value::Int(stats.p99_time_us as i64),
                stats.last_plan.map(Value::String).unwrap_or(Value::Null),
                // CE is single-shard; EE will populate this from scatter-gather stats.
                Value::Int(1),
                sources_value(&stats.sources),
            ]
        })
        .collect()
}

/// `db.advisor.slowQueries(limit, minTime)`: queries with p99 above the
/// threshold. A negative limit or threshold counts as zero.
pub(crate) fn slow_queries(ctx: &AdvisorContext, args: &[Value]) -> Vec<ProcedureRow> {
    let non_negative = |v: Option<&Value>, default: u64| match v {
        Some(Value::Int(n)) => u64::try_from(*n).unwrap_or(0),
        _ => default,
    };
    let limit = usize::try_from(non_negative(args.first(), 20)).unwrap_or(usize::MAX);
    let min_time = non_negative(args.get(1), 100);

    ctx.registry
        .top_by_latency(limit, min_time)
        .into_iter()
        .map(|stats| {
            vec![
                Value::String(stats.canonical_query),
                Value::Int(stats.p99_time_us as i64),
                Value::Int(stats.count as i64),
                stats.last_plan.map(Value::String).unwrap_or(Value::Null),
                sources_value(&stats.sources),
            ]
        })
        .collect()
}

/// `db.advisor.dismiss(id)` — dismiss a suggestion by fingerprint hex ID.
///
/// The dismissed suggestion won't appear in `db.advisor.suggestions()`
/// until `db.advisor.reset()` is called.
pub(crate) fn dismiss(
    ctx: &AdvisorContext,
    args: &[Value],
) -> Result<Vec<ProcedureRow>, ProcedureError> {
    let invalid = |reason: String| ProcedureError::InvalidArgument {
        procedure: "db.advisor.dismiss".into(),
        argument: "id".into(),
        reason,
    };
    let Some(Value::String(id)) = args.first() else {
        return Err(invalid("a fingerprint id is required, got null".into()));
    };
    let fingerprint = u64::from_str_radix(id.trim_start_matches("0x"), 16)
        .map_err(|e| invalid(format!("`{id}` is not a hexadecimal fingerprint: {e}")))?;

    let was_new = ctx.dismissed.dismiss(fingerprint);
    Ok(vec![vec![Value::String(id.clone()), Value::Bool(was_new)]])
}

/// `db.advisor.reset()` — clear all advisor state.
///
/// Resets the fingerprint registry, N+1 detector, and dismissed set.
pub(crate) fn reset(ctx: &AdvisorContext) -> Vec<ProcedureRow> {
    ctx.registry.reset();
    ctx.nplus1.reset();
    ctx.dismissed.reset();
    vec![vec![Value::String("OK".to_string())]]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
