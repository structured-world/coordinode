//! Session lifecycle and dispatch.
//!
//! [`SessionManager`] opens [`Session`]s; a session is driven by [`Session::run`],
//! which reads neutral ops from one channel and funnels neutral events to
//! another. Concurrent dispatch (each request on its own task) and the single
//! outbound writer live here, not in the transport binding. A query is run by
//! opening a server-side cursor through the injected [`CursorEngine`] and paging
//! it into `CursorOpen` → `Rows`* → `CursorEnd`; three concurrent queries are
//! three concurrent dispatch tasks, hence three independent open cursors.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

use coordinode_core::graph::types::Value;
use tokio::sync::mpsc;

use crate::engine::{CursorEngine, EngineError};
use crate::types::{ErrorCode, SessionEvent, SessionOp};

/// An inbound op tagged with its session-scoped request id.
pub type InOp = (u64, SessionOp);

/// An outbound event tagged with the request id it answers.
pub type OutEvent = (u64, SessionEvent);

/// Rows pulled from a cursor per batch before the next `Rows` event is emitted.
const CURSOR_BATCH: usize = 1024;

/// Opens sessions and owns process-wide session allocation state.
pub struct SessionManager {
    next_txid: Arc<AtomicU64>,
    engine: Arc<dyn CursorEngine>,
}

impl SessionManager {
    /// Create a session manager backed by `engine`, with a fresh
    /// transaction-handle allocator.
    pub fn new(engine: Arc<dyn CursorEngine>) -> Self {
        Self {
            next_txid: Arc::new(AtomicU64::new(0)),
            engine,
        }
    }

    /// Open a new session bound to this manager's allocator and engine.
    pub fn open(&self) -> Session {
        Session {
            next_txid: Arc::clone(&self.next_txid),
            engine: Arc::clone(&self.engine),
        }
    }
}

/// A live multiplexed session.
///
/// Transport-agnostic: it consumes neutral [`SessionOp`]s and produces neutral
/// [`SessionEvent`]s. The interactive-transaction registry and cursor registry
/// are added with the state that fills them (transaction work and cancellation);
/// the session holds the transaction-handle allocator and the query engine.
pub struct Session {
    next_txid: Arc<AtomicU64>,
    engine: Arc<dyn CursorEngine>,
}

impl Session {
    /// Drive the session: read ops until the inbound channel closes, dispatch
    /// each on its own task so concurrent requests do not block one another, and
    /// funnel every event to `out`. Returns once the inbound channel is closed;
    /// in-flight dispatch tasks keep `out` alive until they complete, so the
    /// receiver observes end-of-stream only after every event is sent.
    pub async fn run(self, mut ops: mpsc::Receiver<InOp>, out: mpsc::Sender<OutEvent>) {
        while let Some((request_id, op)) = ops.recv().await {
            let next_txid = Arc::clone(&self.next_txid);
            let engine = Arc::clone(&self.engine);
            let out = out.clone();
            tokio::spawn(async move {
                dispatch(&engine, &next_txid, request_id, op, &out).await;
            });
        }
    }
}

/// Dispatch one op, emitting its events to the single writer.
async fn dispatch(
    engine: &Arc<dyn CursorEngine>,
    next_txid: &AtomicU64,
    request_id: u64,
    op: SessionOp,
    out: &mpsc::Sender<OutEvent>,
) {
    match op {
        SessionOp::Execute {
            query,
            params,
            txid,
            ..
        } => execute(engine, request_id, &query, params, txid, out).await,
        SessionOp::Begin { .. } => {
            // Transaction handles are non-zero by contract; start the counter at 1.
            let txid = next_txid.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            let _ = out.send((request_id, SessionEvent::Begun { txid })).await;
        }
        SessionOp::Commit { .. } => {
            let _ = out
                .send((request_id, SessionEvent::Committed { applied_index: 0 }))
                .await;
        }
        // Cancellation and rollback lifecycle land with the cursor/transaction
        // registries; for now they are accepted silently.
        SessionOp::Rollback { .. } | SessionOp::Cancel { .. } => {}
    }
}

/// Open a server-side cursor for one statement and page it into a
/// `CursorOpen` → `Rows`* → `CursorEnd` sequence (or a single `Error`).
///
/// `open_cursor` and `next_batch` are synchronous and may block for a long time:
/// a write drives a Raft commit (`block_in_place` + `block_on`) and a read pages
/// from storage. Running them directly on the async worker thread starves the
/// runtime: under a burst of concurrent writes every worker blocks inside a
/// commit, leaving no worker to drive the Raft apply loop the commits wait on,
/// which deadlocks the whole runtime. So each blocking call runs on the blocking
/// pool via [`spawn_blocking`](tokio::task::spawn_blocking); the cursor (`Send`)
/// moves in and back out per batch, keeping the stream's per-batch backpressure.
async fn execute(
    engine: &Arc<dyn CursorEngine>,
    request_id: u64,
    query: &str,
    params: HashMap<String, Value>,
    txid: u64,
    out: &mpsc::Sender<OutEvent>,
) {
    // Open on the blocking pool: a write statement commits through Raft here.
    let engine = Arc::clone(engine);
    let query = query.to_string();
    let opened = tokio::task::spawn_blocking(move || {
        let cursor = engine.open_cursor(&query, params, txid)?;
        let columns = cursor.columns();
        Ok::<_, EngineError>((cursor, columns))
    })
    .await;
    let (mut cursor, columns) = match opened {
        Ok(Ok(opened)) => opened,
        Ok(Err(e)) => return send_error(out, request_id, e.0).await,
        Err(join) => return send_error(out, request_id, join.to_string()).await,
    };

    if out
        .send((request_id, SessionEvent::CursorOpen { columns }))
        .await
        .is_err()
    {
        return;
    }

    loop {
        // Page on the blocking pool; hand the cursor in and take it back so the
        // next iteration (and `stats()` below) still own it.
        let pulled = tokio::task::spawn_blocking(move || {
            let batch = cursor.next_batch(CURSOR_BATCH);
            (cursor, batch)
        })
        .await;
        let batch = match pulled {
            Ok((returned, batch)) => {
                cursor = returned;
                batch
            }
            Err(join) => return send_error(out, request_id, join.to_string()).await,
        };
        match batch {
            // Empty batch = exhausted.
            Ok(rows) if rows.is_empty() => break,
            Ok(rows) => {
                if out
                    .send((request_id, SessionEvent::Rows { rows }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Err(e) => return send_error(out, request_id, e.0).await,
        }
    }

    let _ = out
        .send((
            request_id,
            SessionEvent::CursorEnd {
                stats: cursor.stats(),
            },
        ))
        .await;
}

/// Emit a single `Error` event for a failed request.
async fn send_error(out: &mpsc::Sender<OutEvent>, request_id: u64, message: String) {
    let _ = out
        .send((
            request_id,
            SessionEvent::Error {
                code: ErrorCode::Internal,
                message,
            },
        ))
        .await;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
