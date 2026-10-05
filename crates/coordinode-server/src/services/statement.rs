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
use std::time::Instant;

use coordinode_core::graph::types::Value;
use coordinode_core::txn::read_concern::{
    ReadConcern as ExecutorReadConcern, ReadConcernLevel as ExecutorReadConcernLevel,
};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_embed::db::CypherResult;
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::advisor::QueryRegistry;
use coordinode_query::advisor::nplus1::NPlus1Detector;
use coordinode_query::advisor::source::SourceContext;
use coordinode_query::frontend::QueryFrontend;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::read_fence::{
    READ_FENCE_TIMEOUT, ReadConcern, ReadFenceError, ReadPreference,
};
use coordinode_replicate::ReplicatedWriter;
// no-std: spin::RwLock / spin::Mutex (drop-in).
use parking_lot::{Mutex, RwLock};
use tonic::{Request, Response, Status};

use crate::config::StatementDefaults;
use crate::proto::query;

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
}

/// A statement whose settings are resolved and consistent, not yet fenced.
#[derive(Debug, Clone)]
pub(crate) struct Checked {
    read_concern: ExecutorReadConcern,
    read_preference: ReadPreference,
    write_concern: Option<WriteConcern>,
}

/// A statement this node may run, with what the fence learned about it.
#[derive(Debug, Clone)]
pub(crate) struct Admitted {
    /// The read concern the engine executes under.
    pub read_concern: ExecutorReadConcern,
    /// The write concern the engine executes under; `None` = the database's.
    pub write_concern: Option<WriteConcern>,
    /// The applied log index the read was served at; zero outside a cluster.
    pub applied_index: u64,
    /// Whether this node led when it served the statement.
    pub served_by_leader: bool,
    /// What a read on a read-only member is as of; zero otherwise.
    pub read_as_of_ts: u64,
}

/// What to do with a checked statement.
#[derive(Debug)]
pub(crate) enum Admission {
    /// Run it here.
    Run(Admitted),
    /// It needs the leader, which is this node: pass it there.
    Forward(u64),
}

/// Resolves, checks, fences, runs and records Cypher statements.
///
/// Cheap to clone: every part is shared, so the unary service and the session
/// hold the same database, consensus node, defaults, advisor and peer
/// connections.
#[derive(Clone)]
pub struct StatementExecutor {
    writer: Arc<ReplicatedWriter>,
    raft_node: Option<Arc<RaftNode>>,
    defaults: StatementDefaults,
    query_registry: Arc<QueryRegistry>,
    nplus1_detector: Arc<NPlus1Detector>,
    /// Lazily-connected channels to peers, keyed by advertised address, for
    /// passing a statement on to the leader. Lazy channels reconnect on their
    /// own, so a peer that restarts does not poison its entry.
    peer_channels: Arc<Mutex<HashMap<String, tonic::transport::Channel>>>,
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
            peer_channels: Arc::new(Mutex::new(HashMap::new())),
        }
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
    pub(crate) fn check(&self, query: &str, requested: &Requested) -> Result<Checked, Status> {
        let level = requested.read_concern.unwrap_or(self.defaults.read_concern);
        let after_index = requested.after_index;

        // A causal read waits for a position a majority committed. A LOCAL
        // read promises nothing about majority, so a fence on top of it is
        // unsound; LINEARIZABLE already reads the latest committed state, so
        // a fence adds nothing and asking for both is a client error.
        if after_index > 0 {
            match level {
                ExecutorReadConcernLevel::Local => {
                    return Err(Status::failed_precondition(
                        "readConcern=LOCAL is incompatible with afterClusterTime. \
                         Causal reads require readConcern=MAJORITY.",
                    ));
                }
                ExecutorReadConcernLevel::Linearizable => {
                    return Err(Status::failed_precondition(
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
                        return Err(Status::failed_precondition(
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
            return Err(Status::failed_precondition(
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
        })
    }

    /// Admit a checked statement without a fence: the standalone case, where
    /// every write is visible at once and there is no log position to report.
    pub(crate) fn admit_standalone(&self, checked: Checked) -> Admitted {
        Admitted {
            read_concern: checked.read_concern,
            write_concern: checked.write_concern,
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
    ) -> Result<Admission, Status> {
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
                _ => Err(fence_error_to_status(e)),
            };
        }
        if let Some(after_index) = checked.read_concern.after_index {
            fence
                .wait_for_index(after_index, READ_FENCE_TIMEOUT)
                .await
                .map_err(fence_error_to_status)?;
        }
        let applied_index = fence.applied_index();
        let served_by_leader = raft.is_leader().await;
        Ok(Admission::Run(Admitted {
            read_concern: checked.read_concern,
            write_concern: checked.write_concern,
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
    ) -> Result<Admission, Status> {
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
        self.writer.execute(
            query,
            params,
            source,
            Some(&admitted.read_concern),
            admitted.write_concern.as_ref(),
        )
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

    /// Re-issue `req` at the leader and return its answer as our own.
    ///
    /// A level-0 client keeps working through a leader change without
    /// noticing one: the node that knows who leads passes the request along.
    /// The response carries the hop count, so a client that does care can see
    /// it and start addressing the leader directly.
    pub(crate) async fn forward(
        &self,
        leader_id: u64,
        req: query::ExecuteCypherRequest,
    ) -> Result<Response<query::ExecuteCypherResponse>, Status> {
        let addr = self
            .raft_node
            .as_ref()
            .and_then(|n| n.node_addr(leader_id))
            .ok_or_else(|| {
                Status::failed_precondition(format!(
                    "not the leader; leader is node {leader_id}, whose address this node \
                     does not know"
                ))
            })?;
        let mut client =
            query::cypher_service_client::CypherServiceClient::new(self.peer_channel(&addr)?);
        let mut forwarded = Request::new(req);
        forwarded.metadata_mut().insert(
            FORWARDED_HEADER,
            tonic::metadata::MetadataValue::from_static("1"),
        );
        let mut response = client.execute_cypher(forwarded).await?;
        response.metadata_mut().insert(
            HOPS_HEADER,
            tonic::metadata::MetadataValue::from_static("1"),
        );
        Ok(response)
    }

    /// A lazily-connected channel to `addr`, created once and reused. The
    /// first call over it connects; it reconnects by itself afterwards.
    fn peer_channel(&self, addr: &str) -> Result<tonic::transport::Channel, Status> {
        if let Some(channel) = self.peer_channels.lock().get(addr) {
            return Ok(channel.clone());
        }
        // Same address form and TLS as the Raft and segment-transfer clients:
        // a cluster that encrypts peer traffic must not get a plaintext hop.
        let channel = coordinode_wire::peer_endpoint(addr)
            .map_err(|e| Status::internal(format!("peer address '{addr}': {e}")))?
            .connect_timeout(std::time::Duration::from_secs(5))
            .connect_lazy();
        self.peer_channels
            .lock()
            .insert(addr.to_string(), channel.clone());
        Ok(channel)
    }
}

/// Marks a request that has already been forwarded once.
///
/// A stale leader hint could otherwise bounce a statement between two nodes
/// that each believe the other leads. One hop is enough: the second node
/// answers with NOT_LEADER and its own hint, and the client decides.
pub(crate) const FORWARDED_HEADER: &str = "x-coordinode-forwarded";

/// How many nodes handled this request: 0 local, 1 forwarded, 2+ scattered.
pub(crate) const HOPS_HEADER: &str = "x-coordinode-hops";

/// The leader named by a failure that says this node cannot take the write.
///
/// `None` for any other failure, and also during an election, when there is
/// no node to forward to and the caller has to be told to try again.
pub(crate) fn leader_hint(err: &DatabaseError) -> Option<u64> {
    match err {
        DatabaseError::NotLeader { leader_id }
        | DatabaseError::Execution(
            coordinode_query::executor::runner::ExecutionError::NotLeader { leader_id },
        ) => *leader_id,
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

/// A read fence refusal as the status a client sees.
fn fence_error_to_status(err: ReadFenceError) -> Status {
    use crate::services::error_details::{Reason, status_with_reason};
    use tonic::Code;
    match err {
        ReadFenceError::NotFollower => Status::failed_precondition(err.to_string()),
        ReadFenceError::NotLeader => Status::failed_precondition(err.to_string()),
        ReadFenceError::LinearizableRequiresLeader => Status::failed_precondition(err.to_string()),
        ReadFenceError::StaleReplica { .. } => Status::unavailable(err.to_string()),
        ReadFenceError::Timeout { .. } | ReadFenceError::LeaseTimeout { .. } => {
            Status::deadline_exceeded(err.to_string())
        }
        ReadFenceError::Raft(e) => Status::internal(format!("Raft error: {e}")),
        // Named like a refused write: the same metadata says why this member
        // is read-only and where the current reads are.
        ReadFenceError::ReadOnly(ref m) => status_with_reason(
            Code::FailedPrecondition,
            err.to_string(),
            Reason::MemberReadOnly,
            mismatch_metadata(m),
        ),
    }
}
