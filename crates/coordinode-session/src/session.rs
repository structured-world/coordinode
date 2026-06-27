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

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::graph::types::Value;
use tokio::sync::mpsc;

use crate::engine::{CursorEngine, EngineError};
use crate::registry::SessionRegistry;
use crate::types::{ErrorCode, Ordering, SessionEvent, SessionOp};

/// An inbound op tagged with its session-scoped request id.
pub type InOp = (u64, SessionOp);

/// An outbound event tagged with the request id it answers.
pub type OutEvent = (u64, SessionEvent);

/// Rows pulled from a cursor per batch before the next `Rows` event is emitted.
const CURSOR_BATCH: usize = 1024;

/// Buffered statements per transaction mailbox before a pipelining client is
/// stalled by backpressure.
const TXN_MAILBOX: usize = 64;

/// Default wait for a missing nonce during an ORDERED commit drain when the
/// client passed `drain_timeout_ms == 0`.
const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

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
        // transactions run concurrently alongside. An entry stays in the map
        // until its task signals completion on `done`, NOT when its commit is
        // sent: an ORDERED commit may still need to receive late-arriving
        // gap-filling statements while it drains, so the transaction must remain
        // routable until the task actually resolves.
        let mut txns: HashMap<u64, mpsc::Sender<TxnMsg>> = HashMap::new();
        let (done_tx, mut done_rx) = mpsc::channel::<u64>(TXN_MAILBOX);

        loop {
            tokio::select! {
                maybe_op = ops.recv() => {
                    let Some((request_id, op)) = maybe_op else { break };
                    self.handle_op(request_id, op, &mut txns, &done_tx, &out).await;
                }
                Some(txid) = done_rx.recv() => {
                    // A transaction task has resolved; stop routing to it.
                    txns.remove(&txid);
                }
            }
        }

        // Stream closed: dropping every transaction mailbox signals each
        // transaction task to abort (rollback), then the session is removed.
        drop(txns);
        self.registry.close_session(self.session_id);
    }

    /// Route one inbound op: a transactional statement to its mailbox, an
    /// autonomous statement to its own task, or a transaction-control op to the
    /// owning transaction (kept routable until its task signals `done`).
    async fn handle_op(
        &self,
        request_id: u64,
        op: SessionOp,
        txns: &mut HashMap<u64, mpsc::Sender<TxnMsg>>,
        done_tx: &mpsc::Sender<u64>,
        out: &mpsc::Sender<OutEvent>,
    ) {
        match op {
            // A statement bound to an open transaction: route to its mailbox.
            SessionOp::Execute {
                query,
                params,
                txid,
                nonce,
            } if txid != 0 && txns.contains_key(&txid) => {
                let msg = TxnMsg::Statement {
                    request_id,
                    nonce,
                    query,
                    params,
                };
                // A send error means the task just resolved (rx dropped) before
                // its `done` was processed; fall back to the engine's
                // unknown-transaction error.
                if let Some(tx) = txns.get(&txid) {
                    if tx.send(msg).await.is_err() {
                        self.spawn_autonomous(request_id, String::new(), HashMap::new(), txid, out);
                    }
                }
            }
            // Autonomous statement, or one naming an unknown transaction (the
            // engine produces the "unknown transaction" error): run concurrently.
            SessionOp::Execute {
                query,
                params,
                txid,
                ..
            } => self.spawn_autonomous(request_id, query, params, txid, out),

            SessionOp::Begin {
                ordering,
                drain_timeout_ms,
            } => {
                self.begin(request_id, ordering, drain_timeout_ms, txns, done_tx, out)
                    .await;
            }

            SessionOp::Commit {
                txid, last_nonce, ..
            } => match txns.get(&txid) {
                // Keep the entry: the task removes itself via `done` once it has
                // drained and resolved.
                Some(tx) => {
                    let _ = tx
                        .send(TxnMsg::Commit {
                            request_id,
                            last_nonce,
                        })
                        .await;
                }
                None => self.spawn_commit_unknown(request_id, txid, out),
            },

            // Rolling back an unknown / already-resolved transaction is a silent
            // no-op (matches the autonomous rollback contract).
            SessionOp::Rollback { txid } => {
                if let Some(tx) = txns.get(&txid) {
                    let _ = tx.send(TxnMsg::Rollback).await;
                }
            }

            // Cancellation lifecycle lands with the cursor registry; accepted
            // silently for now.
            SessionOp::Cancel { .. } => {}
        }
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
    #[allow(clippy::too_many_arguments)]
    async fn begin(
        &self,
        request_id: u64,
        ordering: Ordering,
        drain_timeout_ms: u32,
        txns: &mut HashMap<u64, mpsc::Sender<TxnMsg>>,
        done_tx: &mpsc::Sender<u64>,
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
                // A zero drain timeout means "use the default" rather than
                // "never wait": an ORDERED commit must tolerate some reorder lag.
                let drain = if drain_timeout_ms == 0 {
                    DEFAULT_DRAIN_TIMEOUT
                } else {
                    Duration::from_millis(drain_timeout_ms as u64)
                };
                tokio::spawn(run_transaction(
                    Arc::clone(&self.engine),
                    Arc::clone(&self.registry),
                    self.session_id,
                    txid,
                    ordering,
                    drain,
                    rx,
                    out.clone(),
                    done_tx.clone(),
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
        /// Client-assigned sequence number; orders ORDERED transactions, ignored
        /// for UNORDERED.
        nonce: u64,
        query: String,
        params: HashMap<String, Value>,
    },
    /// Commit the transaction and finish. `last_nonce` is the expected final
    /// nonce of an ORDERED chain, so the commit can drain a reorder gap.
    Commit { request_id: u64, last_nonce: u64 },
    /// Roll the transaction back and finish (silent, carries no request id).
    Rollback,
}

/// Serial owner of one interactive transaction: applies its statements, then
/// commits or rolls back. Dropping the inbound channel (stream close) aborts the
/// transaction; the transaction is deregistered when the task ends. UNORDERED
/// applies statements in arrival order; ORDERED reassembles them by `nonce` in a
/// reorder buffer and applies them strictly in nonce order, with a commit-drain
/// timeout bounding the wait for a missing nonce.
#[allow(clippy::too_many_arguments)]
async fn run_transaction(
    engine: Arc<dyn CursorEngine>,
    registry: Arc<SessionRegistry>,
    session_id: u64,
    txid: u64,
    ordering: Ordering,
    drain: Duration,
    rx: mpsc::Receiver<TxnMsg>,
    out: mpsc::Sender<OutEvent>,
    done: mpsc::Sender<u64>,
) {
    match ordering {
        Ordering::Unordered => {
            run_unordered(&engine, &registry, session_id, txid, rx, &out).await;
        }
        Ordering::Ordered => {
            run_ordered(&engine, &registry, session_id, txid, drain, rx, &out).await;
        }
    }
    registry.end_txn(session_id, txid);
    // Tell the run loop to stop routing to this transaction.
    let _ = done.send(txid).await;
}

/// UNORDERED loop: apply each statement as it arrives; the first failure aborts
/// the transaction and the rest (plus the commit) are rejected. Returns whether
/// the transaction aborted. Stream close before commit/rollback rolls back.
async fn run_unordered(
    engine: &Arc<dyn CursorEngine>,
    registry: &Arc<SessionRegistry>,
    session_id: u64,
    txid: u64,
    mut rx: mpsc::Receiver<TxnMsg>,
    out: &mpsc::Sender<OutEvent>,
) -> bool {
    let mut aborted = false;
    let mut resolved = false;
    while let Some(msg) = rx.recv().await {
        registry.request_started(session_id);
        match msg {
            TxnMsg::Statement {
                request_id,
                query,
                params,
                ..
            } => {
                if aborted {
                    send_error(out, request_id, ABORTED_TXN.to_string()).await;
                } else {
                    registry.touch_txn(session_id, txid);
                    if !execute(engine, request_id, &query, params, txid, out).await {
                        aborted = true;
                        registry.end_txn(session_id, txid);
                    }
                }
            }
            TxnMsg::Commit { request_id, .. } => {
                if aborted {
                    send_error(out, request_id, ABORTED_TXN.to_string()).await;
                } else {
                    commit(engine, request_id, txid, out).await;
                }
                registry.request_finished(session_id);
                resolved = true;
                break;
            }
            TxnMsg::Rollback => {
                if !aborted {
                    rollback(engine, txid).await;
                }
                registry.request_finished(session_id);
                resolved = true;
                break;
            }
        }
        registry.request_finished(session_id);
    }
    if !resolved && !aborted {
        rollback(engine, txid).await;
    }
    aborted
}

/// ORDERED loop: buffer statements by nonce and apply the contiguous run from
/// `next_nonce`. Commit drains what is buffered, then waits (bounded by `drain`)
/// for any missing nonce up to `last_nonce`; a gap that does not fill in time
/// aborts the transaction. Returns whether it aborted.
async fn run_ordered(
    engine: &Arc<dyn CursorEngine>,
    registry: &Arc<SessionRegistry>,
    session_id: u64,
    txid: u64,
    drain: Duration,
    mut rx: mpsc::Receiver<TxnMsg>,
    out: &mpsc::Sender<OutEvent>,
) -> bool {
    let mut next_nonce = 1u64;
    let mut buffer: BTreeMap<u64, (u64, String, HashMap<String, Value>)> = BTreeMap::new();
    let mut aborted = false;
    let mut resolved = false;

    while let Some(msg) = rx.recv().await {
        match msg {
            TxnMsg::Statement {
                request_id,
                nonce,
                query,
                params,
            } => {
                registry.request_started(session_id);
                if aborted {
                    send_error(out, request_id, ABORTED_TXN.to_string()).await;
                    registry.request_finished(session_id);
                    continue;
                }
                registry.touch_txn(session_id, txid);
                buffer.insert(nonce, (request_id, query, params));
                if apply_contiguous(
                    engine,
                    registry,
                    session_id,
                    txid,
                    &mut next_nonce,
                    &mut buffer,
                    out,
                )
                .await
                {
                    aborted = true;
                }
            }
            TxnMsg::Commit {
                request_id,
                last_nonce,
            } => {
                registry.request_started(session_id);
                if !aborted {
                    aborted = drain_to_commit(
                        engine,
                        registry,
                        session_id,
                        txid,
                        last_nonce,
                        drain,
                        &mut next_nonce,
                        &mut buffer,
                        &mut rx,
                        out,
                    )
                    .await;
                }
                if aborted {
                    // Discard whatever the transaction buffered or applied.
                    rollback(engine, txid).await;
                    send_error(out, request_id, ABORTED_TXN.to_string()).await;
                } else {
                    commit(engine, request_id, txid, out).await;
                }
                finish_buffered(registry, session_id, &mut buffer);
                registry.request_finished(session_id);
                resolved = true;
                break;
            }
            TxnMsg::Rollback => {
                if !aborted {
                    rollback(engine, txid).await;
                }
                finish_buffered(registry, session_id, &mut buffer);
                resolved = true;
                break;
            }
        }
    }
    if !resolved && !aborted {
        rollback(engine, txid).await;
        finish_buffered(registry, session_id, &mut buffer);
    }
    aborted
}

/// Apply the contiguous run of buffered statements starting at `*next_nonce`,
/// advancing it. Each applied statement is finished in the registry. Returns
/// `true` if a statement failed (the transaction is then dead).
async fn apply_contiguous(
    engine: &Arc<dyn CursorEngine>,
    registry: &Arc<SessionRegistry>,
    session_id: u64,
    txid: u64,
    next_nonce: &mut u64,
    buffer: &mut BTreeMap<u64, (u64, String, HashMap<String, Value>)>,
    out: &mpsc::Sender<OutEvent>,
) -> bool {
    while let Some((request_id, query, params)) = buffer.remove(next_nonce) {
        registry.touch_txn(session_id, txid);
        let ok = execute(engine, request_id, &query, params, txid, out).await;
        registry.request_finished(session_id);
        *next_nonce += 1;
        if !ok {
            registry.end_txn(session_id, txid);
            return true;
        }
    }
    false
}

/// Drain the ORDERED chain up to `last_nonce`: apply what is buffered, then wait
/// (bounded by `drain`) for each missing nonce to arrive. Returns `true` if the
/// transaction aborted (a statement failed, the wait timed out, the stream
/// closed, or the client rolled back mid-drain).
#[allow(clippy::too_many_arguments)]
async fn drain_to_commit(
    engine: &Arc<dyn CursorEngine>,
    registry: &Arc<SessionRegistry>,
    session_id: u64,
    txid: u64,
    last_nonce: u64,
    drain: Duration,
    next_nonce: &mut u64,
    buffer: &mut BTreeMap<u64, (u64, String, HashMap<String, Value>)>,
    rx: &mut mpsc::Receiver<TxnMsg>,
    out: &mpsc::Sender<OutEvent>,
) -> bool {
    if apply_contiguous(engine, registry, session_id, txid, next_nonce, buffer, out).await {
        return true;
    }
    while *next_nonce <= last_nonce {
        match tokio::time::timeout(drain, rx.recv()).await {
            Ok(Some(TxnMsg::Statement {
                request_id,
                nonce,
                query,
                params,
            })) => {
                registry.request_started(session_id);
                buffer.insert(nonce, (request_id, query, params));
                if apply_contiguous(engine, registry, session_id, txid, next_nonce, buffer, out)
                    .await
                {
                    return true;
                }
            }
            // A second commit during drain is ignored; rollback, stream close, or
            // a drain timeout all abort the partially-applied transaction.
            Ok(Some(TxnMsg::Commit { .. })) => {}
            Ok(Some(TxnMsg::Rollback)) | Ok(None) | Err(_) => return true,
        }
    }
    false
}

/// Finish (in the registry) every still-buffered statement that was received but
/// never applied, so the in-flight count does not leak when a transaction ends
/// with statements still parked in its reorder buffer.
fn finish_buffered(
    registry: &Arc<SessionRegistry>,
    session_id: u64,
    buffer: &mut BTreeMap<u64, (u64, String, HashMap<String, Value>)>,
) {
    let parked = buffer.len();
    buffer.clear();
    for _ in 0..parked {
        registry.request_finished(session_id);
    }
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
