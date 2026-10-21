//! The query-engine seam.
//!
//! The session core opens server-side cursors through [`CursorEngine`], an
//! injected handle the transport binding backs with the real query engine. The
//! core never sees the engine's types or the query dialect: it hands the engine
//! opaque query text plus already-decoded parameters and pages the resulting
//! [`QueryCursor`] into neutral events.

use std::collections::HashMap;

use coordinode_core::graph::types::Value;
use coordinode_core::txn::command::{AdmissionSignal, ExpectedVersion, TransactionId};
use coordinode_core::txn::operation::OperationDescriptor;
use coordinode_core::txn::transaction::CommitReceipt;

use crate::types::{ConnectionSettings, Failure, SessionStats, StatementSource};

/// The transaction a statement runs in, as the session hands it on.
#[derive(Debug, Clone)]
pub struct InTransaction {
    /// The transaction.
    pub id: TransactionId,
    /// Told when the statement has its place in the transaction's order; the
    /// session sends this stream's next command for the transaction then.
    pub admission: AdmissionSignal,
    /// On the transaction's first statement: how it opens. The engine opens
    /// it and runs the statement in it in one step, so a begin costs no
    /// call of its own.
    pub start: Option<TransactionStart>,
}

/// How a transaction opens: the session's settings when its begin arrived,
/// of which the concerns and the timeout outcome become the transaction's
/// for its life, and the client operation its commit carries out.
#[derive(Debug, Clone, PartialEq)]
pub struct TransactionStart {
    /// The session's settings at the begin.
    pub settings: ConnectionSettings,
    /// The client operation the transaction's commit carries out.
    pub operation: Option<OperationDescriptor>,
}

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
/// engine's value space; `transaction` is the interactive transaction the
/// statement runs in, `None` for an autonomous (auto-commit) statement. The
/// transaction lives on the leader, not in the session: a session closing
/// leaves it open. `settings` are the statement's
/// own settings over its session's: a field still unset is the engine's
/// default. `source` is where in the client's code the statement was issued.
/// `operation` is the client operation an autonomous statement carries out:
/// its writes take effect at most once, and a retry of one already settled is
/// refused with its outcome. `cancel` is thrown when the client cancels the
/// statement: the engine stops its work at the next check and answers with a
/// cancellation.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot serve a session's statements",
    label = "needs `CursorEngine`",
    note = "the server implements it over the database as `DatabaseCursorEngine`"
)]
pub trait CursorEngine: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn open_cursor(
        &self,
        query: &str,
        params: HashMap<String, Value>,
        transaction: Option<&InTransaction>,
        settings: &ConnectionSettings,
        source: Option<&StatementSource>,
        operation: Option<&OperationDescriptor>,
        cancel: &coordinode_core::budget::CancelFlag,
    ) -> Result<Box<dyn QueryCursor>, EngineError>;

    /// The change to its session's settings `query` makes, when it is a
    /// dialect's session SET command; `None` for any other statement, and an
    /// error for a SET whose value the setting cannot take, which then
    /// changes nothing. Called for every statement on the session's own
    /// task, so it must be cheap and must not block: a SET takes effect
    /// before the next statement is dispatched, as a Configure does.
    fn session_setting(&self, _query: &str) -> Option<Result<ConnectionSettings, EngineError>> {
        None
    }

    /// Open a new interactive transaction under `transaction`, the ID the
    /// client drew, with `settings`: the session's at the moment, of which
    /// the concerns and the timeout outcome become the transaction's for its
    /// life (a field still unset is the engine's default). `operation` is
    /// the client operation its commit carries out. Subsequent `open_cursor`
    /// calls carrying this ID run inside it, reading its pinned snapshot and
    /// buffering writes until `commit_transaction`. A transaction normally
    /// opens with its first statement ([`InTransaction::start`]); this is
    /// for one whose first command is not a statement.
    fn begin_transaction(
        &self,
        transaction: TransactionId,
        settings: &ConnectionSettings,
        operation: Option<&OperationDescriptor>,
    ) -> Result<(), EngineError>;

    /// Commit an interactive transaction once every command that reached it
    /// before has run, flushing its buffered writes in one proposal and
    /// returning the commit receipt: the HLC commit timestamp plus the
    /// applied Raft index (the causal token; `None` in embedded mode). With
    /// `operation`, the commit carries out that client operation: it takes
    /// effect at most once, and a repeat of one already settled is refused
    /// with its outcome. `expect` names the node versions the caller built
    /// on. `admission` is told when the commit has its place.
    fn commit_transaction(
        &self,
        transaction: TransactionId,
        operation: Option<&OperationDescriptor>,
        expect: &[ExpectedVersion],
        admission: &AdmissionSignal,
    ) -> Result<CommitReceipt, EngineError>;

    /// Roll back an interactive transaction, discarding its buffered writes. No
    /// proposal is emitted (nothing was durable). `admission` is told when
    /// the rollback has its place.
    fn rollback_transaction(
        &self,
        transaction: TransactionId,
        admission: &AdmissionSignal,
    ) -> Result<(), EngineError>;

    /// Abort an interactive transaction whose statement's answer did not
    /// reach its client: the client cannot know what the statement did, and
    /// a statement never runs twice. Later commands for it are told so.
    fn abandon_transaction(&self, transaction: TransactionId) -> Result<(), EngineError>;
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
