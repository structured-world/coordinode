//! Operational introspection over live client sessions and transactions.
//!
//! A point-in-time snapshot of one node's client sessions, the requests in
//! flight on each, and the open interactive transactions this node holds as
//! their owner (with the time remaining until the server auto-aborts an idle
//! one). A session is an ingress attachment: it names the transactions it
//! last sent commands for, while the transaction lives on its owner whether
//! or not any session is attached. The session layer implements
//! [`OperationsView`] and the database implements [`TransactionsView`]; the
//! query executor reads both through these traits without depending on
//! either, keeping the layer dependency one-directional.
//!
//! The snapshot carries only plain data (millisecond deltas already computed
//! against the registry's clock), so it has no clock or `std::time` dependency
//! and crosses the trait boundary cleanly.

use alloc::string::String;
use alloc::vec::Vec;

/// How an interactive transaction orders the statements pipelined into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnOrdering {
    /// Statements applied in nonce order, reassembled in a reorder buffer.
    Ordered,
    /// Statements applied in the order their owner admits them.
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

/// One open interactive transaction held by this node.
#[derive(Debug, Clone)]
pub struct TransactionSnapshot {
    /// The transaction's handle, rendered as the client carries it.
    pub transaction: String,
    /// Statement-ordering mode chosen at `BEGIN`.
    pub ordering: TxnOrdering,
    /// Milliseconds since the transaction was begun.
    pub age_ms: u64,
    /// Milliseconds until the idle reaper auto-aborts this transaction, measured
    /// from its last activity. `Some(0)` means it is due for reaping now; `None`
    /// while a command runs on it, which the reaper never interrupts. The model
    /// has no auto-commit: an abandoned transaction is rolled back, never
    /// committed, because uncommitted state is not durable.
    pub auto_abort_in_ms: Option<u64>,
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
    /// Handles of the transactions this session has sent commands for and
    /// not seen end, rendered as in [`TransactionSnapshot::transaction`].
    pub attached: Vec<String>,
}

/// Read side of the live session registry, injected into the query executor so
/// `SHOW SESSIONS` / `SHOW TRANSACTIONS` can render a snapshot without the query
/// layer depending on the session layer.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot report this node's sessions",
    label = "needs `OperationsView`",
    note = "the session layer implements it on its `SessionRegistry`"
)]
pub trait OperationsView: Send + Sync {
    /// Snapshot every live session on this node, newest activity first is not
    /// guaranteed; callers that need an order sort the result.
    fn sessions(&self) -> Vec<SessionSnapshot>;
}

/// Read side of the open interactive transactions this node holds as their
/// owner, attached to a session or not.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot report this node's open transactions",
    label = "needs `TransactionsView`",
    note = "the embedded database implements it over its transaction registry"
)]
pub trait TransactionsView: Send + Sync {
    /// Snapshot every open transaction; no order is guaranteed.
    fn transactions(&self) -> Vec<TransactionSnapshot>;
}
