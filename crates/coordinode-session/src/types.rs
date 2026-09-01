//! Neutral session operations and events.
//!
//! These types are independent of any wire protocol and any query dialect. A
//! transport binding maps its frames to [`SessionOp`] and maps [`SessionEvent`]
//! back to its frames; the session core only ever sees these.

use std::collections::HashMap;

use coordinode_core::graph::types::Value;

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
    /// binding converts its wire values before constructing this.
    Execute {
        query: String,
        params: HashMap<String, Value>,
        txid: u64,
        nonce: u64,
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
    /// Default write concern level for statements that carry none.
    pub write_concern: Option<u8>,
    /// Default read preference for statements that carry none.
    pub read_preference: Option<u8>,
    /// Default reorder-buffer drain timeout for ordered transactions, in
    /// milliseconds.
    pub drain_timeout_ms: Option<u32>,
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
}

/// Neutral error class for a failed request. The binding maps this to its
/// protocol's status taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// The request was malformed (for example, a frame carrying no operation).
    InvalidArgument,
    /// An internal failure while serving the request.
    Internal,
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
    /// Acknowledges `Commit`, carrying the causal applied-index token.
    Committed { applied_index: u64 },
    /// Reports a request failure; terminates the request's cursor.
    Error { code: ErrorCode, message: String },
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
