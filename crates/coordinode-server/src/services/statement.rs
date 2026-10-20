//! The one path a Cypher statement takes from a client to the engine, shared
//! by the unary RPC and the session.
//!
//! A statement names some of its consistency settings and leaves the rest to
//! its session, then to the server's defaults. Resolving them, checking them,
//! fencing the read, waiting for a causal position, passing a statement this
//! node cannot serve to the leader, executing it and recording it for the
//! advisor all happen here, so a client gets the same answer over either
//! transport.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::writer::ReplicatedWriter;
use coordinode_core::graph::types::Value;
use coordinode_core::txn::command::AdmissionSignal;
use coordinode_embed::db::CypherResult;
pub(crate) use coordinode_embed::replication::admission::{
    Admission, Admitted, Checked, Requested, leader_hint,
};
use coordinode_embed::replication::admission::{AdmissionDefaults, Admitter};
pub(crate) use coordinode_embed::replication::executor::columns_and_values;
use coordinode_embed::replication::routing::{CommandRouter, RoutedAnswer, RoutedStatement};
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::advisor::QueryRegistry;
use coordinode_query::advisor::nplus1::NPlus1Detector;
use coordinode_query::advisor::source::SourceContext;
use coordinode_query::frontend::QueryFrontend;
use coordinode_raft::cluster::RaftNode;
// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use crate::config::StatementDefaults;

/// Resolves, checks, fences, runs and records Cypher statements.
///
/// Cheap to clone: every part is shared, so the unary service and the session
/// hold the same database, consensus node, defaults and advisor. A statement
/// passed to the leader goes over the consensus node's peer connections. The
/// checks and the fence are the library's, the ones an embedded member runs.
#[derive(Clone)]
pub struct StatementExecutor {
    writer: Arc<ReplicatedWriter>,
    admitter: Admitter,
    query_registry: Arc<QueryRegistry>,
    nplus1_detector: Arc<NPlus1Detector>,
    /// Statements running at least this long are logged; `None` logs none.
    slow_query: Option<Duration>,
}

impl StatementExecutor {
    /// An executor over `database`, standalone, with the built-in defaults and
    /// an advisor of its own.
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
        Self {
            admitter: Admitter::new(Arc::clone(&database)),
            writer: Arc::new(ReplicatedWriter::new(database)),
            query_registry: Arc::new(QueryRegistry::new()),
            nplus1_detector: Arc::new(NPlus1Detector::new()),
            slow_query: Some(DEFAULT_SLOW_QUERY),
        }
    }

    /// Log statements that run at least `threshold`; `None` logs none.
    pub fn with_slow_query_threshold(mut self, threshold: Option<Duration>) -> Self {
        self.slow_query = threshold;
        self
    }

    /// Fence reads and route statements through this consensus node.
    pub fn with_raft_node(mut self, raft_node: Arc<RaftNode>) -> Self {
        self.admitter = self.admitter.with_node(raft_node);
        self
    }

    /// Resolve a setting a statement and its session left out with these.
    pub fn with_statement_defaults(mut self, defaults: StatementDefaults) -> Self {
        self.admitter = self.admitter.with_defaults(AdmissionDefaults {
            read_concern: defaults.read_concern,
            read_preference: defaults.read_preference,
        });
        self
    }

    /// Record statements into this advisor.
    pub fn with_advisor(
        mut self,
        query_registry: Arc<QueryRegistry>,
        nplus1_detector: Arc<NPlus1Detector>,
    ) -> Self {
        self.query_registry = query_registry;
        self.nplus1_detector = nplus1_detector;
        self
    }

    /// The database statements run against.
    pub fn database(&self) -> &Arc<RwLock<Database>> {
        self.writer.database()
    }

    /// Whether a fence has to wait on the consensus node, which needs a
    /// runtime; outside a cluster a statement is admitted without waiting.
    pub(crate) fn needs_fence(&self) -> bool {
        self.admitter.needs_fence()
    }

    /// Resolve `requested` against the defaults and refuse a combination
    /// that cannot be served, before anything waits or runs.
    pub(crate) fn check(
        &self,
        query: &str,
        requested: &Requested,
    ) -> Result<Checked, DatabaseError> {
        self.admitter.check(query, requested)
    }

    /// Admit a checked statement without a fence: the standalone case.
    pub(crate) fn admit_standalone(&self, checked: Checked) -> Admitted {
        self.admitter.admit_standalone(checked)
    }

    /// Fence a checked statement. A statement that needs the leader, on a
    /// node that knows who leads, is passed there unless it was passed once
    /// already.
    pub(crate) async fn fence(
        &self,
        checked: Checked,
        already_forwarded: bool,
    ) -> Result<Admission, DatabaseError> {
        self.admitter.fence(checked, already_forwarded).await
    }

    /// Check and fence in one step.
    pub(crate) async fn admit(
        &self,
        query: &str,
        requested: &Requested,
        already_forwarded: bool,
    ) -> Result<Admission, DatabaseError> {
        self.admitter
            .admit(query, requested, already_forwarded)
            .await
    }

    /// Run an admitted statement to completion. Blocking: a write commits
    /// through consensus here.
    pub(crate) fn execute(
        &self,
        query: &str,
        params: Option<HashMap<String, Value>>,
        source: Option<&SourceContext>,
        admitted: &Admitted,
    ) -> Result<CypherResult, DatabaseError> {
        let started = Instant::now();
        metrics::gauge!("coordinode_query_active").increment(1.0);
        let result = self
            .writer
            .execute(query, params, source, &admitted.options());
        metrics::gauge!("coordinode_query_active").decrement(1.0);
        // A statement that committed writes reports where they landed; one
        // that did not is a read.
        let kind = match &result {
            Ok(r) if r.commit_ts().is_some() => QueryKind::Write,
            Ok(_) => QueryKind::Read,
            Err(_) => QueryKind::Failed,
        };
        observe_query(kind, started);
        log_if_slow(
            self.slow_query,
            query,
            kind,
            started.elapsed(),
            result.as_ref().ok().map(|r| r.rows.len()),
        );
        result
    }

    /// Record a finished statement for the advisor: its fingerprint, timing,
    /// and with a source location, the N+1 pattern it may be part of.
    pub(crate) fn record(&self, query: &str, source: Option<&SourceContext>, start: Instant) {
        let Ok((canonical, fp)) =
            coordinode_query::frontend::CypherFrontend::new().fingerprint(query)
        else {
            return;
        };
        let duration_us = start.elapsed().as_micros() as u64;
        match source {
            Some(src) => {
                self.query_registry
                    .record_with_source(fp, &canonical, duration_us, src);
                if let Some(alert) = self.nplus1_detector.record(fp, &canonical, src) {
                    tracing::warn!(
                        fingerprint = fp,
                        count = alert.call_count,
                        file = %alert.source_file,
                        line = alert.source_line,
                        "N+1 query pattern detected"
                    );
                }
            }
            None => self.query_registry.record(fp, &canonical, duration_us),
        }
    }

    /// Run `statement` at `member` and return its answer.
    ///
    /// A client keeps working through whichever member it reached: a
    /// statement this member cannot run, and every command of a
    /// transaction, goes to the leader, and the answer comes back as this
    /// member's own. The responses of a member that does not lead name
    /// the leader, so a client that cares moves there.
    pub(crate) async fn route(
        &self,
        member: u64,
        statement: RoutedStatement,
        admitted: &AdmissionSignal,
    ) -> Result<RoutedAnswer, DatabaseError> {
        self.router(member)?.execute(statement, admitted).await
    }

    /// A router to `member` over this node's authenticated connection to it.
    pub(crate) fn router(&self, member: u64) -> Result<CommandRouter, DatabaseError> {
        self.admitter.router(member)
    }

    /// The member every command of a transaction runs at, the leader: `None`
    /// when it is this one, standalone or leading.
    ///
    /// # Errors
    ///
    /// The refusal a write gets during an election, when no member leads.
    pub(crate) fn transaction_leader(&self) -> Result<Option<u64>, DatabaseError> {
        self.admitter.transaction_leader()
    }
}

/// How many nodes handled this request: 0 local, 1 forwarded, 2+ scattered.
pub(crate) const HOPS_HEADER: &str = "x-coordinode-hops";

/// What a finished statement was, as the query metrics count it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryKind {
    /// It read and committed nothing.
    Read,
    /// It committed writes of its own.
    Write,
    /// It failed.
    Failed,
}

impl QueryKind {
    fn label(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Failed => "failed",
        }
    }
}

/// How long a statement runs before it is logged as slow, unless the
/// configuration names another bound.
pub const DEFAULT_SLOW_QUERY: Duration = Duration::from_millis(100);

/// The most of a statement's text a slow-statement line carries.
const SLOW_QUERY_TEXT_LIMIT: usize = 2048;

/// Log `query` when it ran for `elapsed` and that is at least `threshold`:
/// its kind, duration, rows returned and text (cut at a character boundary
/// past [`SLOW_QUERY_TEXT_LIMIT`] bytes). Parameter values are not logged.
pub(crate) fn log_if_slow(
    threshold: Option<Duration>,
    query: &str,
    kind: QueryKind,
    elapsed: Duration,
    rows: Option<usize>,
) {
    let Some(threshold) = threshold else {
        return;
    };
    if elapsed < threshold {
        return;
    }
    let mut end = query.len().min(SLOW_QUERY_TEXT_LIMIT);
    while !query.is_char_boundary(end) {
        end -= 1;
    }
    tracing::info!(
        target: "coordinode::slow_query",
        kind = kind.label(),
        duration_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
        rows,
        truncated = end < query.len(),
        query = &query[..end],
        "slow statement"
    );
}

/// Count a statement that started at `started` in the query metrics.
pub(crate) fn observe_query(kind: QueryKind, started: Instant) {
    let kind = kind.label();
    metrics::histogram!("coordinode_query_duration_seconds", "type" => kind)
        .record(started.elapsed().as_secs_f64());
    metrics::counter!("coordinode_query_total", "type" => kind).increment(1);
    if kind == QueryKind::Failed.label() {
        metrics::counter!("coordinode_query_errors_total").increment(1);
    }
}

/// The metadata of a read-only member's refusal: both versions, what its
/// reads are as of, and the leader to retry at when known.
pub(crate) fn mismatch_metadata(
    m: &coordinode_core::version::Mismatch,
) -> Vec<(&'static str, String)> {
    let mut metadata = vec![
        ("member_engine_format", m.own.engine.to_string()),
        ("member_host_epoch", m.own.host_epoch.to_string()),
        ("group_engine_format", m.group.engine.to_string()),
        ("group_host_epoch", m.group.host_epoch.to_string()),
        ("behind", m.behind.to_string()),
        ("as_of_ts", m.as_of.to_string()),
    ];
    if let Some((id, addr)) = &m.leader {
        metadata.push(("leader_id", id.to_string()));
        metadata.push(("leader_addr", addr.clone()));
    }
    metadata
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
pub(crate) mod tests;
