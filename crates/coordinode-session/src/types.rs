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
}
