//! Session lifecycle and dispatch.
//!
//! [`SessionManager`] opens [`Session`]s; a session is driven by [`Session::run`],
//! which reads neutral ops from one channel and funnels neutral events to
//! another, with the single outbound writer living here, not in the transport
//! binding. A non-transactional request runs on its own task, so three
//! concurrent queries are three independent cursors. A request bound to an
//! interactive transaction instead routes to that transaction's serial mailbox,
//! so its statements apply one at a time while other traffic runs concurrently.
//! A query is run by opening a server-side cursor through the injected
//! [`CursorEngine`] and paging it into `CursorOpen` / `Rows`* / `CursorEnd`.

use std::collections::HashMap;
use std::sync::Arc;

use coordinode_core::graph::types::Value;
use tokio::sync::mpsc;

use crate::engine::{CursorEngine, EngineError};
use crate::registry::SessionRegistry;
use crate::types::{ErrorCode, SessionEvent, SessionOp};

/// An inbound op tagged with its session-scoped request id.
pub type InOp = (u64, SessionOp);

/// An outbound event tagged with the request id it answers.
pub type OutEvent = (u64, SessionEvent);

/// Rows pulled from a cursor per batch before the next `Rows` event is emitted.
const CURSOR_BATCH: usize = 1024;

/// Buffered statements per transaction mailbox before a pipelining client is
/// stalled by backpressure.
const TXN_MAILBOX: usize = 64;

/// Opens sessions backed by one engine and one shared registry.
pub struct SessionManager {
    engine: Arc<dyn CursorEngine>,
    registry: Arc<SessionRegistry>,
}

impl SessionManager {
    /// Create a session manager backed by `engine` and the shared session
    /// `registry` that powers operational introspection (`SHOW SESSIONS` /
    /// `SHOW TRANSACTIONS`). Transaction handles are allocated by the engine.
    pub fn new(engine: Arc<dyn CursorEngine>, registry: Arc<SessionRegistry>) -> Self {
        Self { engine, registry }
    }

    /// Open a new session for `peer`, registering it so it is visible to
    /// introspection until its stream closes.
    pub fn open(&self, peer: String) -> Session {
        let session_id = self.registry.register_session(peer);
        Session {
            session_id,
            engine: Arc::clone(&self.engine),
            registry: Arc::clone(&self.registry),
        }
    }
}

/// A live multiplexed session.
///
/// Transport-agnostic: it consumes neutral [`SessionOp`]s and produces neutral
/// [`SessionEvent`]s. It holds the query engine and the shared session registry;
/// transaction handles are allocated by the engine, and each open transaction
/// runs on its own serial mailbox spawned by [`Session::run`].
pub struct Session {
    session_id: u64,
    engine: Arc<dyn CursorEngine>,
    registry: Arc<SessionRegistry>,
}

impl Session {
    /// Drive the session: read ops until the inbound channel closes, dispatch
    /// each on its own task so concurrent requests do not block one another, and
    /// funnel every event to `out`. Returns once the inbound channel is closed;
    /// in-flight dispatch tasks keep `out` alive until they complete, so the
    /// receiver observes end-of-stream only after every event is sent.
    ///
    /// The registry tracks each request as in-flight for its lifetime and drops
    /// the session (with its open transactions) when the stream closes, so the
    /// introspection snapshot mirrors what is actually running.
    pub async fn run(self, mut ops: mpsc::Receiver<InOp>, out: mpsc::Sender<OutEvent>) {
        // Per-transaction serialized mailboxes owned by this single run task (no
        // lock needed). A transaction's statements and its commit/rollback route
        // to its mailbox and apply one at a time, because the transaction state
        // is checked out per statement; non-transactional requests and other
        // transactions run concurrently alongside.
        let mut txns: HashMap<u64, mpsc::Sender<TxnMsg>> = HashMap::new();

        while let Some((request_id, op)) = ops.recv().await {
            match op {
                // A statement bound to an open transaction: route to its mailbox.
                SessionOp::Execute {
                    query,
                    params,
                    txid,
                    ..
                } if txid != 0 && txns.contains_key(&txid) => {
                    let msg = TxnMsg::Statement {
                        request_id,
                        query,
                        params,
                    };
                    // The receiver only drops after commit/rollback; a send error
                    // means the transaction just resolved, so fall back to the
                    // engine's unknown-transaction error.
                    if let Some(tx) = txns.get(&txid) {
                        if tx.send(msg).await.is_err() {
                            self.spawn_autonomous(
                                request_id,
                                String::new(),
                                HashMap::new(),
                                txid,
                                &out,
                            );
                        }
                    }
                }
                // Autonomous statement, or one naming an unknown transaction
                // (the engine produces the "unknown transaction" error): run
                // concurrently.
                SessionOp::Execute {
                    query,
                    params,
                    txid,
                    ..
                } => self.spawn_autonomous(request_id, query, params, txid, &out),

                SessionOp::Begin { ordering, .. } => {
                    self.begin(request_id, ordering, &mut txns, &out).await;
                }

                SessionOp::Commit { txid, .. } => match txns.remove(&txid) {
                    Some(tx) => {
                        let _ = tx.send(TxnMsg::Commit { request_id }).await;
                    }
                    None => self.spawn_commit_unknown(request_id, txid, &out),
                },

                // Rolling back an unknown / already-resolved transaction is a
                // silent no-op (matches the autonomous rollback contract).
                SessionOp::Rollback { txid } => {
                    if let Some(tx) = txns.remove(&txid) {
                        let _ = tx.send(TxnMsg::Rollback).await;
                    }
                }

                // Cancellation lifecycle lands with the cursor registry; accepted
                // silently for now.
                SessionOp::Cancel { .. } => {}
            }
        }

        // Stream closed: dropping every transaction mailbox signals each
        // transaction task to abort (rollback), then the session is removed.
        drop(txns);
        self.registry.close_session(self.session_id);
    }

    /// Run an autonomous statement (or one against an unknown transaction)
    /// concurrently, bracketing it as in-flight for its lifetime.
    fn spawn_autonomous(
        &self,
        request_id: u64,
        query: String,
        params: HashMap<String, Value>,
        txid: u64,
        out: &mpsc::Sender<OutEvent>,
    ) {
        let engine = Arc::clone(&self.engine);
        let registry = Arc::clone(&self.registry);
        let session_id = self.session_id;
        let out = out.clone();
        registry.request_started(session_id);
        tokio::spawn(async move {
            let _ = execute(&engine, request_id, &query, params, txid, &out).await;
            registry.request_finished(session_id);
        });
    }

    /// Open a transaction inline (so a pipelined `Execute{txid}` that follows is
    /// guaranteed to find the mailbox), register it, and spawn its serial task.
    async fn begin(
        &self,
        request_id: u64,
        ordering: crate::types::Ordering,
        txns: &mut HashMap<u64, mpsc::Sender<TxnMsg>>,
        out: &mpsc::Sender<OutEvent>,
    ) {
        self.registry.request_started(self.session_id);
        // begin_transaction is a quick blocking call (snapshot pin + register);
        // run it on the blocking pool like every other engine call.
        let engine = Arc::clone(&self.engine);
        let begun = tokio::task::spawn_blocking(move || engine.begin_transaction()).await;
        match begun {
            Ok(Ok(txid)) => {
                let (tx, rx) = mpsc::channel::<TxnMsg>(TXN_MAILBOX);
                txns.insert(txid, tx);
                self.registry.begin_txn(self.session_id, txid, ordering);
                tokio::spawn(run_transaction(
                    Arc::clone(&self.engine),
                    Arc::clone(&self.registry),
                    self.session_id,
                    txid,
                    rx,
                    out.clone(),
                ));
                let _ = out.send((request_id, SessionEvent::Begun { txid })).await;
            }
            Ok(Err(e)) => send_error(out, request_id, e.0).await,
            Err(join) => send_error(out, request_id, join.to_string()).await,
        }
        self.registry.request_finished(self.session_id);
    }

    /// Commit a transaction not in the mailbox map: let the engine produce the
    /// "unknown transaction" error (or commit a racing late-resolved handle).
    fn spawn_commit_unknown(&self, request_id: u64, txid: u64, out: &mpsc::Sender<OutEvent>) {
        let engine = Arc::clone(&self.engine);
        let registry = Arc::clone(&self.registry);
        let session_id = self.session_id;
        let out = out.clone();
        registry.request_started(session_id);
        tokio::spawn(async move {
            commit(&engine, request_id, txid, &out).await;
            registry.request_finished(session_id);
        });
    }
}

/// A message in a transaction's serial mailbox.
enum TxnMsg {
    /// A statement to run inside the transaction.
    Statement {
        request_id: u64,
        query: String,
        params: HashMap<String, Value>,
    },
    /// Commit the transaction and finish.
    Commit { request_id: u64 },
    /// Roll the transaction back and finish (silent, carries no request id).
    Rollback,
}

/// Serial owner of one interactive transaction: applies its statements in
/// arrival order, then commits or rolls back. Dropping the inbound channel
/// (stream close) aborts the transaction. Each handled message is bracketed as
/// in-flight, and the transaction is deregistered when the task ends.
///
/// Statement reassembly by `nonce` for ORDERED transactions (the reorder buffer
/// and commit-drain timeout) layers on top of this serial loop; today both
/// ordering modes apply in arrival order.
async fn run_transaction(
    engine: Arc<dyn CursorEngine>,
    registry: Arc<SessionRegistry>,
    session_id: u64,
    txid: u64,
    mut rx: mpsc::Receiver<TxnMsg>,
    out: mpsc::Sender<OutEvent>,
) {
    let mut resolved = false;
    // Set once a statement fails: the engine drops the transaction state on a
    // failed statement, so the whole transaction is dead (UNORDERED first-failure
    // rollback). The rest of its statements and its commit are then rejected.
    let mut aborted = false;
    while let Some(msg) = rx.recv().await {
        registry.request_started(session_id);
        match msg {
            TxnMsg::Statement {
                request_id,
                query,
                params,
            } => {
                if aborted {
                    send_error(&out, request_id, ABORTED_TXN.to_string()).await;
                } else {
                    // A statement resets the transaction's auto-abort countdown.
                    registry.touch_txn(session_id, txid);
                    if !execute(&engine, request_id, &query, params, txid, &out).await {
                        // The engine dropped the transaction's state on the
                        // failed statement; deregister it so introspection stops
                        // listing a transaction that can no longer commit.
                        aborted = true;
                        registry.end_txn(session_id, txid);
                    }
                }
            }
            TxnMsg::Commit { request_id } => {
                if aborted {
                    send_error(&out, request_id, ABORTED_TXN.to_string()).await;
                } else {
                    commit(&engine, request_id, txid, &out).await;
                }
                registry.request_finished(session_id);
                resolved = true;
                break;
            }
            TxnMsg::Rollback => {
                // An aborted transaction is already discarded by the engine;
                // rollback is then a silent no-op.
                if !aborted {
                    rollback(&engine, txid).await;
                }
                registry.request_finished(session_id);
                resolved = true;
                break;
            }
        }
        registry.request_finished(session_id);
    }
    if !resolved && !aborted {
        // Inbound channel closed before commit/rollback on a live transaction
        // (stream closed): abort it.
        let engine = Arc::clone(&engine);
        let _ = tokio::task::spawn_blocking(move || engine.rollback_transaction(txid)).await;
    }
    registry.end_txn(session_id, txid);
}

/// Error returned for any statement or commit issued against a transaction that
/// already failed a statement.
const ABORTED_TXN: &str = "transaction is aborted, roll it back";

/// Commit a transaction on the blocking pool and emit `Committed` or `Error`.
async fn commit(
    engine: &Arc<dyn CursorEngine>,
    request_id: u64,
    txid: u64,
    out: &mpsc::Sender<OutEvent>,
) {
    let engine = Arc::clone(engine);
    match tokio::task::spawn_blocking(move || engine.commit_transaction(txid)).await {
        Ok(Ok(applied_index)) => {
            let _ = out
                .send((request_id, SessionEvent::Committed { applied_index }))
                .await;
        }
        Ok(Err(e)) => send_error(out, request_id, e.0).await,
        Err(join) => send_error(out, request_id, join.to_string()).await,
    }
}

/// Roll a transaction back on the blocking pool. Silent (the rollback contract
/// emits no event); a failure is swallowed because the client asked to discard.
async fn rollback(engine: &Arc<dyn CursorEngine>, txid: u64) {
    let engine = Arc::clone(engine);
    let _ = tokio::task::spawn_blocking(move || engine.rollback_transaction(txid)).await;
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
/// Returns `true` if the statement completed (a `CursorEnd` was reached) and
/// `false` if it ended in an `Error` (or the client's channel closed). A
/// transaction uses this to abort on the first failing statement.
async fn execute(
    engine: &Arc<dyn CursorEngine>,
    request_id: u64,
    query: &str,
    params: HashMap<String, Value>,
    txid: u64,
    out: &mpsc::Sender<OutEvent>,
) -> bool {
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
        Ok(Err(e)) => {
            send_error(out, request_id, e.0).await;
            return false;
        }
        Err(join) => {
            send_error(out, request_id, join.to_string()).await;
            return false;
        }
    };

    if out
        .send((request_id, SessionEvent::CursorOpen { columns }))
        .await
        .is_err()
    {
        return false;
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
            Err(join) => {
                send_error(out, request_id, join.to_string()).await;
                return false;
            }
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
                    return false;
                }
            }
            Err(e) => {
                send_error(out, request_id, e.0).await;
                return false;
            }
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
    true
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
