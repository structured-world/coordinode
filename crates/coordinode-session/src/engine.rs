//! The query-engine seam.
//!
//! The session core opens server-side cursors through [`CursorEngine`], an
//! injected handle the transport binding backs with the real query engine. The
//! core never sees the engine's types or the query dialect: it hands the engine
//! opaque query text plus already-decoded parameters and pages the resulting
//! [`QueryCursor`] into neutral events.

use std::collections::HashMap;

use coordinode_core::graph::types::Value;
use coordinode_core::txn::transaction::CommitReceipt;

use crate::types::{ConnectionSettings, Failure, SessionStats, StatementSource};

/// An error from the query engine, neutral over the engine implementation. It
/// reaches the client as the request's failure, class and details intact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError(pub Failure);

impl EngineError {
    /// An internal failure with no structured details.
    pub fn internal(message: impl Into<String>) -> Self {
        Self(Failure::internal(message))
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.message)
    }
}

impl std::error::Error for EngineError {}

/// Opens server-side cursors for the session core.
///
/// The binding injects an implementation backed by the query engine. `query` is
/// opaque dialect text the core does not inspect; `params` are already in the
/// engine's value space; `txid` is the interactive-transaction handle, or zero
/// for an autonomous (auto-commit) statement. `settings` are the statement's
/// own settings over its session's: a field still unset is the engine's
/// default. `source` is where in the client's code the statement was issued.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot serve a session's statements",
    label = "needs `CursorEngine`",
    note = "the server implements it over the database as `DatabaseCursorEngine`"
)]
pub trait CursorEngine: Send + Sync {
    fn open_cursor(
        &self,
        query: &str,
        params: HashMap<String, Value>,
        txid: u64,
        settings: &ConnectionSettings,
        source: Option<&StatementSource>,
    ) -> Result<Box<dyn QueryCursor>, EngineError>;

    /// Open a new interactive transaction and return its handle. Subsequent
    /// `open_cursor` calls carrying this `txid` run inside it, reading its pinned
    /// snapshot and buffering writes until `commit_transaction`.
    fn begin_transaction(&self) -> Result<u64, EngineError>;

    /// Commit an interactive transaction, flushing its buffered writes in one
    /// proposal and returning the commit receipt: the HLC commit timestamp plus
    /// the applied Raft index (the causal token; `None` in embedded mode).
    fn commit_transaction(&self, txid: u64) -> Result<CommitReceipt, EngineError>;

    /// Roll back an interactive transaction, discarding its buffered writes. No
    /// proposal is emitted (nothing was durable).
    fn rollback_transaction(&self, txid: u64) -> Result<(), EngineError>;
}

/// A live server-side cursor: a column header plus paged row batches.
///
/// The cursor pins its read snapshot for its whole life; the session pulls
/// batches until one comes back empty (exhausted), then reads [`Self::stats`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a result cursor the session can page",
    label = "needs `QueryCursor`",
    note = "a `CursorEngine` returns one from `open_cursor`"
)]
pub trait QueryCursor: Send {
    /// Result column names, in result order. Available before the first batch.
    fn columns(&self) -> Vec<String>;

    /// Pull up to `max` more rows. An empty batch means the cursor is exhausted.
    fn next_batch(&mut self, max: usize) -> Result<Vec<Vec<Value>>, EngineError>;

    /// Final statistics for the statement, valid once the cursor is exhausted.
    fn stats(&self) -> SessionStats;
}
