//! Admission of a statement on a member: the settings it arrived with
//! resolved against the member's defaults and checked, then the read fenced.
//! A statement this member cannot serve is named for the leader instead.
//!
//! The same checks run whichever way a statement arrives: a host's call on
//! an embedded member, a server's client, or a command another member routed
//! here. The server's statement path consumes this one.

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_core::budget::CancelFlag;
use coordinode_core::graph::types::VectorConsistencyMode;
use coordinode_core::txn::operation::OperationDescriptor;
use coordinode_core::txn::read_concern::{ReadConcern, ReadConcernLevel};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::read_fence::{
    READ_FENCE_TIMEOUT, ReadConcern as FenceConcern, ReadFenceError, ReadPreference,
};
// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use super::routing::CommandRouter;
use crate::db::{ReadFenceFailure, StatementOptions};
use crate::{Database, DatabaseError};

/// What a statement falls back to for a setting neither it nor its session
/// named.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdmissionDefaults {
    /// What a read may observe.
    pub read_concern: ReadConcernLevel,
    /// Which member may serve a read.
    pub read_preference: ReadPreference,
}

/// The consistency settings a statement arrived with, after its session's
/// were folded in. `None` leaves the setting to the member's default.
#[derive(Debug, Clone, Default)]
pub struct Requested {
    /// What the read may observe.
    pub read_concern: Option<ReadConcernLevel>,
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
    /// the query; `None` takes the database's.
    pub vector_build_wait: Option<Duration>,
    /// Memory limit of the statement in bytes; `None` takes the database's.
    pub query_memory_limit: Option<u64>,
    /// When the caller stops waiting for the statement.
    pub deadline: Option<Instant>,
    /// Thrown when the caller cancels the statement or goes away.
    pub cancel: Option<CancelFlag>,
    /// The client operation the statement carries out; `None` runs it
    /// unbound.
    pub operation: Option<OperationDescriptor>,
}

/// A statement whose settings are resolved and consistent, not yet fenced.
#[derive(Debug, Clone)]
pub struct Checked {
    read_concern: ReadConcern,
    read_preference: ReadPreference,
    write_concern: Option<WriteConcern>,
    vector_consistency: Option<VectorConsistencyMode>,
    vector_build_wait: Option<Duration>,
    query_memory_limit: Option<u64>,
    deadline: Option<Instant>,
    cancel: Option<CancelFlag>,
    operation: Option<OperationDescriptor>,
}

/// A statement this member may run, with what the fence learned about it.
#[derive(Debug, Clone)]
pub struct Admitted {
    /// The read concern the engine executes under.
    pub read_concern: ReadConcern,
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
    pub deadline: Option<Instant>,
    /// Thrown when the caller cancels the statement or goes away.
    pub cancel: Option<CancelFlag>,
    /// The client operation the statement carries out.
    pub operation: Option<OperationDescriptor>,
    /// The applied log index the read was served at; zero outside a group.
    pub applied_index: u64,
    /// Whether this member led when it served the statement.
    pub served_by_leader: bool,
    /// What a read on a read-only member is as of; zero otherwise.
    pub read_as_of_ts: u64,
}

impl Admitted {
    /// The per-statement settings the engine runs this statement under.
    pub fn options(&self) -> StatementOptions {
        StatementOptions {
            read_concern: Some(self.read_concern.clone()),
            write_concern: self.write_concern,
            vector_consistency: self.vector_consistency,
            vector_build_wait: self.vector_build_wait,
            query_memory_limit: self.query_memory_limit,
            deadline: self.deadline,
            cancel: self.cancel.clone(),
            operation: self.operation,
            ..StatementOptions::default()
        }
    }
}

/// What to do with a checked statement.
#[derive(Debug)]
pub enum Admission {
    /// Run it here.
    Run(Admitted),
    /// It needs the leader, which is another member: route it there.
    Forward(u64),
}

/// Resolves, checks and fences statements on one member.
///
/// Cheap to clone: the database and the consensus node are shared.
#[derive(Clone)]
pub struct Admitter {
    database: Arc<RwLock<Database>>,
    node: Option<Arc<RaftNode>>,
    defaults: AdmissionDefaults,
}

impl Admitter {
    /// An admitter over a standalone `database`, with the built-in defaults:
    /// every statement runs here, unfenced.
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
        Self {
            database,
            node: None,
            defaults: AdmissionDefaults::default(),
        }
    }

    /// Fence reads and name the leader through this consensus node.
    #[must_use]
    pub fn with_node(mut self, node: Arc<RaftNode>) -> Self {
        self.node = Some(node);
        self
    }

    /// Resolve a setting a statement and its session left out with these.
    #[must_use]
    pub fn with_defaults(mut self, defaults: AdmissionDefaults) -> Self {
        self.defaults = defaults;
        self
    }

    /// The database statements run against.
    pub fn database(&self) -> &Arc<RwLock<Database>> {
        &self.database
    }

    /// The consensus node, when this member has one.
    pub fn node(&self) -> Option<&Arc<RaftNode>> {
        self.node.as_ref()
    }

    /// Whether a fence has to wait on the consensus node, which needs a
    /// runtime; a standalone statement is admitted without waiting.
    pub fn needs_fence(&self) -> bool {
        self.node.is_some()
    }

    /// Resolve `requested` against the defaults and refuse a combination
    /// that cannot be served, before anything waits or runs.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::SettingsRefused`] naming the inconsistency.
    pub fn check(&self, query: &str, requested: &Requested) -> Result<Checked, DatabaseError> {
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
                ReadConcernLevel::Local => {
                    return Err(refused(
                        "readConcern=LOCAL is incompatible with afterClusterTime. \
                         Causal reads require readConcern=MAJORITY.",
                    ));
                }
                ReadConcernLevel::Linearizable => {
                    return Err(refused(
                        "readConcern=LINEARIZABLE is incompatible with afterClusterTime. \
                         Use readConcern=MAJORITY for causal reads.",
                    ));
                }
                ReadConcernLevel::Majority | ReadConcernLevel::Snapshot => {}
            }
            // A write in a causal session must be majority-committed and
            // journaled before its position is handed out: a weaker write can
            // be lost on failover, leaving a position no follower ever reaches.
            // Refused before execution, so nothing is written to be refused
            // afterwards. A query that does not parse runs on and fails there
            // with the richer error.
            if let Ok(ast) = coordinode_query::cypher::parse(query) {
                let db = self.database.read();
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
        if requested.at_timestamp > 0 && level != ReadConcernLevel::Snapshot {
            return Err(refused(
                "readConcern.at_timestamp is only valid with level=SNAPSHOT",
            ));
        }

        Ok(Checked {
            read_concern: ReadConcern {
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
    pub fn admit_standalone(&self, checked: Checked) -> Admitted {
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
    /// causal position. A statement that needs the leader, on a member that
    /// knows who leads, is named for it unless it was routed once already.
    ///
    /// # Errors
    ///
    /// The fence's refusal: the wrong role, a stale replica, a timeout.
    pub async fn fence(
        &self,
        checked: Checked,
        already_forwarded: bool,
    ) -> Result<Admission, DatabaseError> {
        let Some(node) = self.node.as_ref() else {
            return Ok(Admission::Run(self.admit_standalone(checked)));
        };
        let mut fence = node.read_fence();
        if let Err(e) = fence
            .apply_default(
                checked.read_preference,
                fence_concern(checked.read_concern.level),
            )
            .await
        {
            return fence_refusal(e, node.current_leader(), already_forwarded);
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
    ///
    /// # Errors
    ///
    /// As [`Admitter::check`] and [`Admitter::fence`].
    pub async fn admit(
        &self,
        query: &str,
        requested: &Requested,
        already_forwarded: bool,
    ) -> Result<Admission, DatabaseError> {
        let checked = self.check(query, requested)?;
        self.fence(checked, already_forwarded).await
    }

    /// A router to `member` over this member's authenticated connection to
    /// it.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::PeerUnreachable`] when this member has no peers or
    /// no connection to `member`.
    pub fn router(&self, member: u64) -> Result<CommandRouter, DatabaseError> {
        let node = self
            .node
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
    pub fn transaction_leader(&self) -> Result<Option<u64>, DatabaseError> {
        let Some(node) = self.node.as_ref() else {
            return Ok(None);
        };
        match node.current_leader() {
            Some(leader) if leader == node.node_id() => Ok(None),
            Some(leader) => Ok(Some(leader)),
            None => Err(DatabaseError::NotLeader { leader_id: None }),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;

/// The leader named by a failure that says this member cannot take the
/// write.
///
/// `None` for any other failure, and also during an election, when there is
/// no member to route to and the caller has to be told to try again.
pub fn leader_hint(err: &DatabaseError) -> Option<u64> {
    match err {
        DatabaseError::NotLeader { leader_id } => *leader_id,
        _ => None,
    }
}

/// The read fence's view of a read concern level: the same four levels,
/// checked before the statement runs.
fn fence_concern(level: ReadConcernLevel) -> FenceConcern {
    match level {
        ReadConcernLevel::Local => FenceConcern::Local,
        ReadConcernLevel::Majority => FenceConcern::Majority,
        ReadConcernLevel::Linearizable => FenceConcern::Linearizable,
        ReadConcernLevel::Snapshot => FenceConcern::Snapshot,
    }
}

/// What a statement the fence refused becomes, `leader` being the member
/// this one believes leads.
///
/// Rather than make every client learn the topology, the member that knows
/// who leads routes a statement that needs the leader there. Once routed, it
/// is answered rather than routed again: two members with stale hints would
/// otherwise trade it back and forth, so a member reached by a routed
/// statement it cannot run refuses it naming the leader it knows. With no
/// leader known the answer is the explicit no-leader refusal, which a client
/// retries after the election.
fn fence_refusal(
    err: ReadFenceError,
    leader: Option<u64>,
    already_forwarded: bool,
) -> Result<Admission, DatabaseError> {
    let needs_leader = matches!(
        err,
        ReadFenceError::NotLeader | ReadFenceError::LinearizableRequiresLeader
    );
    match leader {
        Some(leader) if needs_leader && !already_forwarded => Ok(Admission::Forward(leader)),
        // Routed once already: the refusal names the leader this member
        // knows, and the ingress decides what to do with it.
        leader_id if needs_leader => Err(DatabaseError::NotLeader { leader_id }),
        _ => Err(fence_error(err)),
    }
}

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
