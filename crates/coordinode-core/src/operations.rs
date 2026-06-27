//! Operational introspection over live client sessions and transactions.
//!
//! A point-in-time snapshot of one node's client sessions, the requests in
//! flight on each, and the open interactive transactions each holds (with the
//! time remaining until the server auto-aborts an idle transaction). The
//! session layer owns the live registry and implements [`OperationsView`]; the
//! query executor reads a snapshot through this trait without depending on the
//! session crate, keeping the layer dependency one-directional.
//!
//! The snapshot carries only plain data (millisecond deltas already computed
//! against the registry's clock), so it has no clock or `std::time` dependency
//! and crosses the trait boundary cleanly.

/// How an interactive transaction orders the statements pipelined into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnOrdering {
    /// Statements applied in nonce order, reassembled in a reorder buffer.
    Ordered,
    /// Statements applied in wire-arrival order.
    Unordered,
}

impl TxnOrdering {
    /// Lowercase label for result rows / JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            TxnOrdering::Ordered => "ordered",
            TxnOrdering::Unordered => "unordered",
        }
    }
}

/// One open interactive transaction on a session.
#[derive(Debug, Clone)]
pub struct TransactionSnapshot {
    /// Session-scoped transaction handle (non-zero).
    pub txid: u64,
    /// Statement-ordering mode chosen at `BEGIN`.
    pub ordering: TxnOrdering,
    /// Milliseconds since the transaction was begun.
    pub age_ms: u64,
    /// Milliseconds until the idle reaper auto-aborts this transaction, measured
    /// from its last activity. `0` means it is due for reaping now. The model
    /// has no auto-commit: an abandoned transaction is rolled back, never
    /// committed, because uncommitted state is not durable.
    pub auto_abort_in_ms: u64,
}

/// One live client session (connection) on this node.
#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    /// Node-local session identifier.
    pub session_id: String,
    /// Remote peer address, or empty if the binding did not supply one.
    pub peer: String,
    /// Milliseconds since the session was opened.
    pub age_ms: u64,
    /// Requests currently executing on this session.
    pub in_flight: u64,
    /// Open interactive transactions held by this session.
    pub transactions: Vec<TransactionSnapshot>,
}

/// Read side of the live session registry, injected into the query executor so
/// `SHOW SESSIONS` / `SHOW TRANSACTIONS` can render a snapshot without the query
/// layer depending on the session layer.
pub trait OperationsView: Send + Sync {
    /// Snapshot every live session on this node, newest activity first is not
    /// guaranteed; callers that need an order sort the result.
    fn sessions(&self) -> Vec<SessionSnapshot>;
}
