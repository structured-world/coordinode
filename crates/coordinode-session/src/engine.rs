//! The query-engine seam.
//!
//! The session core opens server-side cursors through [`CursorEngine`], an
//! injected handle the transport binding backs with the real query engine. The
//! core never sees the engine's types or the query dialect: it hands the engine
//! opaque query text plus already-decoded parameters and pages the resulting
//! [`QueryCursor`] into neutral events.

use std::collections::HashMap;

use coordinode_core::graph::types::Value;

use crate::types::SessionStats;

/// An error from the query engine, neutral over the engine implementation.
#[derive(Debug, Clone)]
pub struct EngineError(pub String);

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EngineError {}

/// Opens server-side cursors for the session core.
///
/// The binding injects an implementation backed by the query engine. `query` is
/// opaque dialect text the core does not inspect; `params` are already in the
/// engine's value space; `txid` is the interactive-transaction handle, or zero
/// for an autonomous (auto-commit) statement.
pub trait CursorEngine: Send + Sync {
    fn open_cursor(
        &self,
        query: &str,
        params: HashMap<String, Value>,
        txid: u64,
    ) -> Result<Box<dyn QueryCursor>, EngineError>;
}

/// A live server-side cursor: a column header plus paged row batches.
///
/// The cursor pins its read snapshot for its whole life; the session pulls
/// batches until one comes back empty (exhausted), then reads [`Self::stats`].
pub trait QueryCursor: Send {
    /// Result column names, in result order. Available before the first batch.
    fn columns(&self) -> Vec<String>;

    /// Pull up to `max` more rows. An empty batch means the cursor is exhausted.
    fn next_batch(&mut self, max: usize) -> Result<Vec<Vec<Value>>, EngineError>;

    /// Final statistics for the statement, valid once the cursor is exhausted.
    fn stats(&self) -> SessionStats;
}
