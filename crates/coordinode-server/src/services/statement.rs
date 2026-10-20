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
use coordinode_core::graph::types::{Value, VectorConsistencyMode};
use coordinode_core::txn::command::AdmissionSignal;
use coordinode_core::txn::operation::OperationDescriptor;
use coordinode_core::txn::read_concern::{
    ReadConcern as ExecutorReadConcern, ReadConcernLevel as ExecutorReadConcernLevel,
};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_embed::db::{CypherResult, StatementOptions};
use coordinode_embed::replication::routing::{CommandRouter, RoutedAnswer, RoutedStatement};
use coordinode_embed::{Database, DatabaseError, ReadFenceFailure};
use coordinode_query::advisor::QueryRegistry;
use coordinode_query::advisor::nplus1::NPlus1Detector;
use coordinode_query::advisor::source::SourceContext;
use coordinode_query::executor::row::Row;
use coordinode_query::frontend::QueryFrontend;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::read_fence::{
    READ_FENCE_TIMEOUT, ReadConcern, ReadFenceError, ReadPreference,
};
// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use crate::config::StatementDefaults;

/// The consistency settings a statement arrived with, after its session's
/// were folded in. `None` leaves the setting to the server's default.
#[derive(Debug, Clone, Default)]
pub(crate) struct Requested {
    /// What the read may observe.
    pub read_concern: Option<ExecutorReadConcernLevel>,
    /// Log position the read waits for; zero waits for nothing.
    pub after_index: u64,
    /// Snapshot timestamp the read is pinned to; zero pins none.
    pub at_timestamp: u64,
    /// Which member may serve the read.
    pub read_preference: Option<ReadPreference>,
    /// When the write is acknowledged; `None` takes the database's default.
    pub write_concern: Option<WriteConcern>,
    /// Vector consistency below a hint in the query; `None` takes what the
    /// query's read consistency implies.
    pub vector_consistency: Option<VectorConsistencyMode>,
    /// Bound on waiting for a vector index still being built, below a hint in
    /// the query; `None` takes the server's.
    pub vector_build_wait: Option<Duration>,
    /// Memory limit of the statement in bytes, already within the ceiling;
    /// `None` takes the server's.
    pub query_memory_limit: Option<u64>,
    /// When the caller stops waiting for the statement: its gRPC deadline.
    pub deadline: Option<std::time::Instant>,
    /// Thrown when the caller cancels the statement or goes away.
    pub cancel: Option<coordinode_core::budget::CancelFlag>,
    /// The client operation the statement carries out; `None` runs it
    /// unbound.
    pub operation: Option<OperationDescriptor>,
}

/// A statement whose settings are resolved and consistent, not yet fenced.
#[derive(Debug, Clone)]
pub(crate) struct Checked {
    read_concern: ExecutorReadConcern,
    read_preference: ReadPreference,
    write_concern: Option<WriteConcern>,
    vector_consistency: Option<VectorConsistencyMode>,
    vector_build_wait: Option<Duration>,
    query_memory_limit: Option<u64>,
    deadline: Option<std::time::Instant>,
    cancel: Option<coordinode_core::budget::CancelFlag>,
    operation: Option<OperationDescriptor>,
}

/// A statement this node may run, with what the fence learned about it.
#[derive(Debug, Clone)]
pub(crate) struct Admitted {
    /// The read concern the engine executes under.
    pub read_concern: ExecutorReadConcern,
    /// The write concern the engine executes under; `None` = the database's.
    pub write_concern: Option<WriteConcern>,
    /// Vector consistency below a hint in the query; `None` = the database's.
    pub vector_consistency: Option<VectorConsistencyMode>,
    /// Vector build-wait bound below a hint in the query; `None` = the
    /// database's.
    pub vector_build_wait: Option<Duration>,
    /// Memory limit of the statement in bytes; `None` = the database's.
    pub query_memory_limit: Option<u64>,
    /// When the caller stops waiting for the statement.
    pub deadline: Option<std::time::Instant>,
    /// Thrown when the caller cancels the statement or goes away.
    pub cancel: Option<coordinode_core::budget::CancelFlag>,
    /// The client operation the statement carries out.
    pub operation: Option<OperationDescriptor>,
    /// The applied log index the read was served at; zero outside a cluster.
    pub applied_index: u64,
    /// Whether this node led when it served the statement.
    pub served_by_leader: bool,
    /// What a read on a read-only member is as of; zero otherwise.
    pub read_as_of_ts: u64,
}

impl Admitted {
    /// The per-statement settings the engine runs this statement under.
    pub(crate) fn options(&self) -> StatementOptions {
        StatementOptions {
            read_concern: Some(self.read_concern.clone()),
            write_concern: self.write_concern,
            vector_consistency: self.vector_consistency,
            vector_build_wait: self.vector_build_wait,
            query_memory_limit: self.query_memory_limit,
            deadline: self.deadline,
            cancel: self.cancel.clone(),
            operation: self.operation,
            // An autonomous statement: no transaction, so no nonce or turn.
            ..StatementOptions::default()
        }
    }
}

/// What to do with a checked statement.
#[derive(Debug)]
pub(crate) enum Admission {
    /// Run it here.
    Run(Admitted),
    /// It needs the leader, which is another node: pass it there.
    Forward(u64),
}

/// Resolves, checks, fences, runs and records Cypher statements.
///
/// Cheap to clone: every part is shared, so the unary service and the session
/// hold the same database, consensus node, defaults and advisor. A statement
/// passed to the leader goes over the consensus node's peer connections.
#[derive(Clone)]
pub struct StatementExecutor {
    writer: Arc<ReplicatedWriter>,
    raft_node: Option<Arc<RaftNode>>,
    defaults: StatementDefaults,
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
            writer: Arc::new(ReplicatedWriter::new(database)),
            raft_node: None,
            defaults: StatementDefaults::default(),
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
        self.raft_node = Some(raft_node);
        self
    }

    /// Resolve a setting a statement and its session left out with these.
    pub fn with_statement_defaults(mut self, defaults: StatementDefaults) -> Self {
        self.defaults = defaults;
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
        self.raft_node.is_some()
    }

    /// Resolve `requested` against the defaults and refuse a combination
    /// that cannot be served, before anything waits or runs.
    pub(crate) fn check(
        &self,
        query: &str,
        requested: &Requested,
    ) -> Result<Checked, DatabaseError> {
        let refused = |reason: &str| DatabaseError::SettingsRefused(reason.to_owned());
        // A session SET changes the settings of the session that sends it.
        // A session takes it before it gets here; on this path there is no
        // session, and the database's own setting is every client's.
        if Database::parse_session_set(query).is_some() {
            return Err(refused(
                "SET changes the settings of a session, and this request has none: \
                 send it on a Session, or name the setting for one query in a hint",
            ));
        }
        let level = requested.read_concern.unwrap_or(self.defaults.read_concern);
        let after_index = requested.after_index;

        // A causal read waits for a position a majority committed. A LOCAL
        // read promises nothing about majority, so a fence on top of it is
        // unsound; LINEARIZABLE already reads the latest committed state, so
        // a fence adds nothing and asking for both is a client error.
        if after_index > 0 {
            match level {
                ExecutorReadConcernLevel::Local => {
                    return Err(refused(
                        "readConcern=LOCAL is incompatible with afterClusterTime. \
                         Causal reads require readConcern=MAJORITY.",
                    ));
                }
                ExecutorReadConcernLevel::Linearizable => {
                    return Err(refused(
                        "readConcern=LINEARIZABLE is incompatible with afterClusterTime. \
                         Use readConcern=MAJORITY for causal reads.",
                    ));
                }
                ExecutorReadConcernLevel::Majority | ExecutorReadConcernLevel::Snapshot => {}
            }
            // A write in a causal session must be majority-committed and
            // journaled before its position is handed out: a weaker write can
            // be lost on failover, leaving a position no follower ever reaches.
            // Refused before execution, so nothing is written to be refused
            // afterwards. A query that does not parse runs on and fails there
            // with the richer error.
            if let Ok(ast) = coordinode_query::cypher::parse(query) {
                let db = self.database().read();
                if ast.is_write(&|name| db.procedures().writes(name)) {
                    // Unnamed, it runs under the database's default, so that
                    // is the one to check.
                    let concern = requested
                        .write_concern
                        .unwrap_or_else(|| db.write_concern());
                    if !concern.is_causal_safe() {
                        return Err(refused(
                            "Causal sessions require writeConcern w:majority with \
                             j:journal for write statements. A weaker write (w:1, w:0, \
                             or a volatile journal level) may be lost before replication: \
                             the resulting applied_index would be a dangling causal \
                             dependency that followers can never satisfy. Use a \
                             non-causal session for such writes, or upgrade to \
                             w:majority.",
                        ));
                    }
                }
            }
        }

        // A pinned timestamp means something only to a snapshot read; any
        // other level would ignore it silently.
        if requested.at_timestamp > 0 && level != ExecutorReadConcernLevel::Snapshot {
            return Err(refused(
                "readConcern.at_timestamp is only valid with level=SNAPSHOT",
            ));
        }

        Ok(Checked {
            read_concern: ExecutorReadConcern {
                level,
                after_index: (after_index > 0).then_some(after_index),
                at_timestamp: (requested.at_timestamp > 0).then_some(requested.at_timestamp),
            },
            read_preference: requested
                .read_preference
                .unwrap_or(self.defaults.read_preference),
            write_concern: requested.write_concern,
            vector_consistency: requested.vector_consistency,
            vector_build_wait: requested.vector_build_wait,
            query_memory_limit: requested.query_memory_limit,
            deadline: requested.deadline,
            cancel: requested.cancel.clone(),
            operation: requested.operation,
        })
    }

    /// Admit a checked statement without a fence: the standalone case, where
    /// every write is visible at once and there is no log position to report.
    pub(crate) fn admit_standalone(&self, checked: Checked) -> Admitted {
        Admitted {
            read_concern: checked.read_concern,
            write_concern: checked.write_concern,
            vector_consistency: checked.vector_consistency,
            vector_build_wait: checked.vector_build_wait,
            query_memory_limit: checked.query_memory_limit,
            deadline: checked.deadline,
            cancel: checked.cancel,
            operation: checked.operation,
            applied_index: 0,
            served_by_leader: false,
            read_as_of_ts: 0,
        }
    }

    /// Fence a checked statement: the read preference and concern, then the
    /// causal position. A statement that needs the leader, on a node that
    /// knows who leads, is passed there unless it was passed once already.
    pub(crate) async fn fence(
        &self,
        checked: Checked,
        already_forwarded: bool,
    ) -> Result<Admission, DatabaseError> {
        let Some(raft) = self.raft_node.as_ref() else {
            return Ok(Admission::Run(self.admit_standalone(checked)));
        };
        let mut fence = raft.read_fence();
        if let Err(e) = fence
            .apply_default(
                checked.read_preference,
                fence_concern(checked.read_concern.level),
            )
            .await
        {
            // Rather than make every client learn the topology, the node that
            // knows who leads passes the statement along. Once passed, it is
            // answered rather than passed again: two nodes with stale hints
            // would otherwise trade it back and forth.
            return match raft.current_leader() {
                Some(leader_id)
                    if !already_forwarded
                        && matches!(
                            e,
                            ReadFenceError::NotLeader | ReadFenceError::LinearizableRequiresLeader
                        ) =>
                {
                    Ok(Admission::Forward(leader_id))
                }
                _ => Err(fence_error(e)),
            };
        }
        if let Some(after_index) = checked.read_concern.after_index {
            fence
                .wait_for_index(after_index, READ_FENCE_TIMEOUT)
                .await
                .map_err(fence_error)?;
        }
        let applied_index = fence.applied_index();
        let served_by_leader = fence.locally_leads();
        Ok(Admission::Run(Admitted {
            read_concern: checked.read_concern,
            write_concern: checked.write_concern,
            vector_consistency: checked.vector_consistency,
            vector_build_wait: checked.vector_build_wait,
            query_memory_limit: checked.query_memory_limit,
            deadline: checked.deadline,
            cancel: checked.cancel,
            operation: checked.operation,
            applied_index,
            served_by_leader,
            read_as_of_ts: fence.as_of().unwrap_or(0),
        }))
    }

    /// Check and fence in one step.
    pub(crate) async fn admit(
        &self,
        query: &str,
        requested: &Requested,
        already_forwarded: bool,
    ) -> Result<Admission, DatabaseError> {
        let checked = self.check(query, requested)?;
        self.fence(checked, already_forwarded).await
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
        let node = self
            .raft_node
            .as_ref()
            .ok_or_else(|| DatabaseError::PeerUnreachable {
                member,
                message: "this node has no peers".to_owned(),
            })?;
        CommandRouter::to_member(node, member)
    }

    /// The member every command of a transaction runs at, the leader: `None`
    /// when it is this one, standalone or leading.
    ///
    /// # Errors
    ///
    /// The refusal a write gets during an election, when no member leads.
    pub(crate) fn transaction_leader(&self) -> Result<Option<u64>, DatabaseError> {
        let Some(node) = self.raft_node.as_ref() else {
            return Ok(None);
        };
        match node.current_leader() {
            Some(leader) if leader == node.node_id() => Ok(None),
            Some(leader) => Ok(Some(leader)),
            None => Err(DatabaseError::NotLeader { leader_id: None }),
        }
    }
}

/// The column header and the rows in column order: the keys of the first row
/// (every row has the same), and a missing value as `Null` so each row is as
/// wide as the header. No rows have no columns.
pub(crate) fn columns_and_values(rows: &[Row]) -> (Vec<String>, Vec<Vec<Value>>) {
    let columns: Vec<String> = rows
        .first()
        .map(|r| r.keys().cloned().collect())
        .unwrap_or_default();
    let values = rows
        .iter()
        .map(|row| {
            columns
                .iter()
                .map(|col| row.get(col).cloned().unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    (columns, values)
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

/// The leader named by a failure that says this node cannot take the write.
///
/// `None` for any other failure, and also during an election, when there is
/// no node to forward to and the caller has to be told to try again.
pub(crate) fn leader_hint(err: &DatabaseError) -> Option<u64> {
    match err {
        DatabaseError::NotLeader { leader_id } => *leader_id,
        _ => None,
    }
}

/// The read fence's view of an executor read concern level: the same four
/// levels, checked before the statement runs.
fn fence_concern(level: ExecutorReadConcernLevel) -> ReadConcern {
    match level {
        ExecutorReadConcernLevel::Local => ReadConcern::Local,
        ExecutorReadConcernLevel::Majority => ReadConcern::Majority,
        ExecutorReadConcernLevel::Linearizable => ReadConcern::Linearizable,
        ExecutorReadConcernLevel::Snapshot => ReadConcern::Snapshot,
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

/// A read fence refusal in the library's error contract.
fn fence_error(err: ReadFenceError) -> DatabaseError {
    let message = err.to_string();
    let fence = |kind| DatabaseError::ReadFence { kind, message };
    match err {
        ReadFenceError::NotFollower
        | ReadFenceError::NotLeader
        | ReadFenceError::LinearizableRequiresLeader => fence(ReadFenceFailure::Role),
        ReadFenceError::StaleReplica { .. } => fence(ReadFenceFailure::Stale),
        ReadFenceError::Timeout { .. } | ReadFenceError::LeaseTimeout { .. } => {
            fence(ReadFenceFailure::Timeout)
        }
        ReadFenceError::Raft(e) => DatabaseError::Other(format!("Raft error: {e}")),
        // Named like a refused write: the same fields say why this member is
        // read-only and where the current reads are.
        ReadFenceError::ReadOnly(m) => DatabaseError::Mismatched(m),
    }
}
