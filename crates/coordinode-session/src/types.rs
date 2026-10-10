//! Neutral session operations and events.
//!
//! These types are independent of any wire protocol and any query dialect. A
//! transport binding maps its frames to [`SessionOp`] and maps [`SessionEvent`]
//! back to its frames; the session core only ever sees these.

use std::collections::HashMap;
use std::time::Duration;

use coordinode_core::graph::types::{Value, VectorConsistencyMode};
use coordinode_core::txn::transaction::CommitReceipt;
use coordinode_core::txn::write_concern::WriteConcern;

/// Per-transaction statement ordering, fixed when the transaction begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// Statements are applied in wire-arrival order; the first failure rolls
    /// the whole transaction back.
    Unordered,
    /// Statements carry a per-transaction nonce; the core reassembles them by
    /// nonce and applies them strictly in nonce order.
    Ordered,
}

/// A neutral request operation on a session.
#[derive(Debug, Clone)]
pub enum SessionOp {
    /// Run one query statement, autonomously (`txid == 0`) or inside a
    /// transaction. `params` are already in the engine's value space; the
    /// binding converts its wire values before constructing this. `settings`
    /// are the ones the statement names itself; each one it leaves unset is
    /// the session's. `source` is where in the client's code it was issued.
    Execute {
        query: String,
        params: HashMap<String, Value>,
        txid: u64,
        nonce: u64,
        settings: ConnectionSettings,
        source: Option<StatementSource>,
    },
    /// Open an interactive transaction.
    Begin {
        ordering: Ordering,
        drain_timeout_ms: u32,
    },
    /// Commit an interactive transaction by handle.
    Commit { txid: u64, last_nonce: u64 },
    /// Roll back an interactive transaction by handle.
    Rollback { txid: u64 },
    /// Abort an in-flight request and close its cursor.
    Cancel { target_request_id: u64 },
    /// Read, and optionally change, the connection's settings. Answered with a
    /// [`SessionEvent::ConnectionStatus`] carrying what is now in effect.
    Configure(ConnectionSettings),
}

/// Where in a client's code a statement was issued, for the query advisor.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatementSource {
    /// Source file path as the client knows it.
    pub file: String,
    /// Line number in that file.
    pub line: u32,
    /// Enclosing function; empty when the client cannot tell.
    pub function: String,
    /// The client application's name; empty when it did not say.
    pub app: String,
    /// The client application's version; empty when it did not say.
    pub version: String,
}

/// The settings a connection applies to statements that carry none.
///
/// Each field is optional in the sense of "leave as it is": a client changes
/// one setting without restating the others, and an all-empty value asks only
/// for the current status.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConnectionSettings {
    /// Default read concern level for statements that carry none.
    pub read_concern: Option<u8>,
    /// Fence index applied to reads that carry no concern of their own.
    pub after_index: Option<u64>,
    /// Snapshot pin applied to reads that carry no concern of their own.
    pub at_timestamp: Option<u64>,
    /// Default write concern for statements that carry none.
    pub write_concern: Option<WriteConcern>,
    /// Default read preference for statements that carry none.
    pub read_preference: Option<u8>,
    /// Default reorder-buffer drain timeout for ordered transactions, in
    /// milliseconds.
    pub drain_timeout_ms: Option<u32>,
    /// Vector consistency for statements whose query names none in a hint.
    pub vector_consistency: Option<VectorConsistencyMode>,
    /// How long a statement waits for a vector index still being built, when
    /// its query names no bound in a hint.
    pub vector_build_wait: Option<Duration>,
    /// Memory limit of each statement, in bytes, at most the server's
    /// administrative ceiling.
    pub query_memory_limit: Option<u64>,
}

impl ConnectionSettings {
    /// Fold `change` into these settings: a field the change leaves unset
    /// keeps its current value.
    ///
    /// This is what makes a partial Configure mean "change this one thing"
    /// rather than "reset everything I did not mention".
    pub fn apply(&mut self, change: &ConnectionSettings) {
        if change.read_concern.is_some() {
            self.read_concern = change.read_concern;
        }
        if change.after_index.is_some() {
            self.after_index = change.after_index;
        }
        if change.at_timestamp.is_some() {
            self.at_timestamp = change.at_timestamp;
        }
        if change.write_concern.is_some() {
            self.write_concern = change.write_concern;
        }
        if change.read_preference.is_some() {
            self.read_preference = change.read_preference;
        }
        if change.drain_timeout_ms.is_some() {
            self.drain_timeout_ms = change.drain_timeout_ms;
        }
        if change.vector_consistency.is_some() {
            self.vector_consistency = change.vector_consistency;
        }
        if change.vector_build_wait.is_some() {
            self.vector_build_wait = change.vector_build_wait;
        }
        if change.query_memory_limit.is_some() {
            self.query_memory_limit = change.query_memory_limit;
        }
    }

    /// The settings a statement runs under: these, with every setting the
    /// statement names itself in place of this one.
    pub fn under(&self, statement: &ConnectionSettings) -> ConnectionSettings {
        let mut effective = self.clone();
        effective.apply(statement);
        effective
    }
}

/// What this connection can do right now, and what the serving node can see of
/// the cluster.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConnectionState {
    /// Whether a write can be served through this connection: this node leads,
    /// or it can reach the node that does.
    pub writable: bool,
    /// Whether this node has contact with the cluster at all.
    pub connected: bool,
    /// The node the cluster last named leader, if this node knows one.
    pub leader_id: Option<u64>,
    /// Whether this node is itself the leader.
    pub served_by_leader: bool,
    /// Raft term the observation was made in.
    pub raft_term: u64,
    /// Voting members the cluster is configured with.
    pub voters: u32,
    /// Voting members this node counts as reachable, including itself.
    pub voters_reachable: u32,
}

/// Statistics for a completed statement, neutral over the wire protocol.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionStats {
    /// Nodes created by the statement.
    pub nodes_created: i64,
    /// Nodes deleted by the statement.
    pub nodes_deleted: i64,
    /// Edges created by the statement.
    pub edges_created: i64,
    /// Edges deleted by the statement.
    pub edges_deleted: i64,
    /// Properties set by the statement.
    pub properties_set: i64,
    /// Wall-clock execution time, in milliseconds.
    pub execution_time_ms: i64,
    /// Raft index the statement was applied at (causal token); zero in embedded
    /// mode.
    pub applied_index: u64,
    /// Whether the read was served by the Raft leader.
    pub served_by_leader: bool,
    /// The timestamp this statement's writes landed at, when it committed on
    /// its own; zero for a read and for a statement inside an interactive
    /// transaction. It is the version of what the statement wrote, so a
    /// client that writes and then writes again conditionally needs no read
    /// in between.
    pub commit_ts: u64,
    /// The commit timestamp a read served by a read-only member reflects:
    /// the member does not run its group's version and applies nothing past
    /// it. Zero for a read served by a member that runs its group's version.
    pub read_as_of_ts: u64,
}

/// Error class of a failed request: the canonical status codes, which every
/// binding maps one to one onto its protocol. A client decides by the class
/// whether to retry, go elsewhere or give up, so a failure keeps the class the
/// engine gave it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// The caller cancelled the request.
    Cancelled,
    /// A failure with no better class.
    Unknown,
    /// The request was malformed (for example, a frame carrying no operation).
    InvalidArgument,
    /// The request ran out of time before it finished.
    DeadlineExceeded,
    /// Something the request names does not exist.
    NotFound,
    /// Something the request creates exists already.
    AlreadyExists,
    /// The caller may not do this.
    PermissionDenied,
    /// A limit or a quota was reached; retrying later may succeed.
    ResourceExhausted,
    /// The request is fine, the state it needs is not (for example, it reached
    /// a node that is not the leader).
    FailedPrecondition,
    /// The request lost to a concurrent one; retrying it may succeed.
    Aborted,
    /// A value is past the valid range (for example, a timestamp older than
    /// the retention window).
    OutOfRange,
    /// The server does not do this.
    Unimplemented,
    /// An internal failure while serving the request.
    Internal,
    /// The service cannot answer right now; retrying may succeed.
    Unavailable,
    /// Data was lost or corrupted.
    DataLoss,
    /// The caller is not authenticated.
    Unauthenticated,
}

/// Why a request failed, as the engine reported it.
///
/// `details` is opaque to the session core: the binding that backs the engine
/// fills it in its protocol's encoding (structured details a client branches
/// on, such as the reason and the leader to go to) and the same binding reads
/// it back when it answers. The core only carries it from one to the other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    /// The failure's class.
    pub code: ErrorCode,
    /// What happened, for a human.
    pub message: String,
    /// Structured details, encoded by the binding; empty when there are none.
    pub details: Vec<u8>,
}

impl Failure {
    /// A failure of class `code` with no structured details.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: Vec::new(),
        }
    }

    /// An internal failure with no structured details.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }
}

/// A neutral result event for a request.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Acknowledges `Begin`, carrying the allocated transaction handle.
    Begun { txid: u64 },
    /// Opens a result cursor with its column header.
    CursorOpen { columns: Vec<String> },
    /// A batch of result rows for an open cursor.
    Rows { rows: Vec<Vec<Value>> },
    /// Closes a result cursor with final statistics.
    CursorEnd { stats: SessionStats },
    /// Acknowledges `Commit`, carrying the commit receipt: the HLC commit
    /// timestamp and the causal applied-index token.
    Committed { receipt: CommitReceipt },
    /// Reports a request failure; terminates the request's cursor.
    Error(Failure),
    /// The state of the connection and the settings in effect on it.
    ///
    /// Answers a [`SessionOp::Configure`], and is also emitted unsolicited
    /// whenever the state changes, so a client waiting for a cut-off node to
    /// regain a leader is told rather than left to poll.
    ConnectionStatus {
        /// What the connection can do and what the node can see.
        state: ConnectionState,
        /// The settings now in effect.
        settings: ConnectionSettings,
    },
}
