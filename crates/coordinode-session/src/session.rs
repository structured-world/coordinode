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
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};

use coordinode_core::budget::CancelFlag;

use crate::engine::{CursorEngine, EngineError};
use crate::registry::SessionRegistry;
use crate::types::{
    ConnectionSettings, ConnectionState, ErrorCode, Failure, Ordering, SessionEvent, SessionOp,
    SessionStats, StatementSource,
};

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
    connection: watch::Receiver<ConnectionState>,
}

impl SessionManager {
    /// Create a session manager backed by `engine` and the shared session
    /// `registry` that powers operational introspection (`SHOW SESSIONS` /
    /// `SHOW TRANSACTIONS`). Transaction handles are allocated by the engine.
    ///
    /// Sessions report a connection that is always writable, which is the truth
    /// for an embedded database: there is no cluster to lose touch with. Use
    /// [`Self::with_connection`] where there is one.
    pub fn new(engine: Arc<dyn CursorEngine>, registry: Arc<SessionRegistry>) -> Self {
        let (_tx, rx) = watch::channel(ConnectionState {
            writable: true,
            connected: true,
            leader_id: None,
            served_by_leader: true,
            raft_term: 0,
            voters: 1,
            voters_reachable: 1,
        });
        Self {
            engine,
            registry,
            connection: rx,
        }
    }

    /// Report connection state from `connection` instead of assuming a
    /// standalone node.
    ///
    /// The sender behind it is whatever watches the cluster: sessions read the
    /// current value and are woken on change, which is what lets a client be
    /// told that its node reached a leader again rather than having to ask.
    pub fn with_connection(mut self, connection: watch::Receiver<ConnectionState>) -> Self {
        self.connection = connection;
        self
    }

    /// Open a new session for `peer`, registering it so it is visible to
    /// introspection until its stream closes.
    pub fn open(&self, peer: String) -> Session {
        let session_id = self.registry.register_session(peer);
        Session {
            session_id,
            engine: Arc::clone(&self.engine),
            registry: Arc::clone(&self.registry),
            connection: self.connection.clone(),
            settings: Mutex::new(ConnectionSettings::default()),
            cancels: Cancels::default(),
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
    /// What the serving node can currently do for this connection. Read on
    /// demand and watched for changes, which are pushed to the client.
    connection: watch::Receiver<ConnectionState>,
    /// Settings this connection applies to statements that carry none. Behind
    /// a lock because a Configure lands on the session task while statements
    /// dispatched from it read the settings.
    settings: Mutex<ConnectionSettings>,
    /// The statements in flight, for a Cancel to reach.
    cancels: Cancels,
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
    pub async fn run(mut self, mut ops: mpsc::Receiver<InOp>, out: mpsc::Sender<OutEvent>) {
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
                // The node's ability to serve this connection changed. Tell the
                // client unsolicited, under request_id zero: the change it
                // usually waits for is one nobody asked about, and being told
                // beats polling with an interval guessed from how long an
                // election might take.
                Ok(()) = self.connection.changed() => {
                    let state = self.connection.borrow().clone();
                    let settings = self.settings.lock().clone();
                    let _ = out
                        .send((0, SessionEvent::ConnectionStatus { state, settings }))
                        .await;
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
        // A session SET changes this session's settings, on this task like a
        // Configure so the next statement already runs under it, and answers
        // with the empty result the statement has. Never the engine's own
        // settings: those are every other session's too.
        if let SessionOp::Execute { query, .. } = &op {
            if let Some(change) = self.engine.session_setting(query) {
                let change = match change {
                    Ok(change) => change,
                    Err(refused) => {
                        send_error(out, request_id, refused.0).await;
                        return;
                    }
                };
                self.settings.lock().apply(&change);
                let opened = SessionEvent::CursorOpen {
                    columns: Vec::new(),
                };
                if out.send((request_id, opened)).await.is_ok() {
                    let ended = SessionEvent::CursorEnd {
                        stats: SessionStats::default(),
                    };
                    let _ = out.send((request_id, ended)).await;
                }
                return;
            }
        }
        match op {
            // A statement bound to an open transaction: route to its mailbox.
            // Its settings are taken now, so a Configure that arrives after it
            // does not reach back.
            SessionOp::Execute {
                query,
                params,
                txid,
                nonce,
                settings,
                source,
            } if txid != 0 && txns.contains_key(&txid) => {
                let msg = TxnMsg::Statement {
                    nonce,
                    statement: Statement {
                        request_id,
                        query,
                        params,
                        settings: self.settings.lock().under(&settings),
                        source: source.map(Box::new),
                        cancel: self.cancels.register(request_id),
                    },
                };
                // A send error means the task just resolved (rx dropped) before
                // its `done` was processed; fall back to the engine's
                // unknown-transaction error.
                if let Some(tx) = txns.get(&txid) {
                    if tx.send(msg).await.is_err() {
                        self.spawn_autonomous(
                            Statement {
                                request_id,
                                query: String::new(),
                                params: HashMap::new(),
                                settings: ConnectionSettings::default(),
                                source: None,
                                cancel: self.cancels.register(request_id),
                            },
                            txid,
                            out,
                        );
                    }
                }
            }
            // Autonomous statement, or one naming an unknown transaction (the
            // engine produces the "unknown transaction" error): run concurrently.
            SessionOp::Execute {
                query,
                params,
                txid,
                settings,
                source,
                ..
            } => {
                let statement = Statement {
                    request_id,
                    query,
                    params,
                    settings: self.settings.lock().under(&settings),
                    source: source.map(Box::new),
                    cancel: self.cancels.register(request_id),
                };
                self.spawn_autonomous(statement, txid, out);
            }

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

            // Stops the statement at its next check, which answers it with
            // a cancellation; one already finished is unaffected.
            SessionOp::Cancel { target_request_id } => self.cancels.cancel(target_request_id),

            // Settings are connection-wide, so this is handled on the session's
            // own task rather than spawned: a change must be in effect before
            // the next statement is dispatched, and the answer reports what is
            // in effect rather than what was asked for.
            SessionOp::Configure(change) => {
                let settings = {
                    let mut current = self.settings.lock();
                    current.apply(&change);
                    current.clone()
                };
                let event = SessionEvent::ConnectionStatus {
                    state: self.connection.borrow().clone(),
                    settings,
                };
                let _ = out.send((request_id, event)).await;
            }
        }
    }

    /// Run an autonomous statement (or one against an unknown transaction)
    /// concurrently, bracketing it as in-flight for its lifetime.
    fn spawn_autonomous(&self, statement: Statement, txid: u64, out: &mpsc::Sender<OutEvent>) {
        let engine = Arc::clone(&self.engine);
        let registry = Arc::clone(&self.registry);
        let session_id = self.session_id;
        let out = out.clone();
        registry.request_started(session_id);
        tokio::spawn(async move {
            let _ = execute(&engine, statement, txid, &out).await;
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
            Err(join) => send_error(out, request_id, lost_call(join)).await,
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

/// The in-flight statements of one session, by request id, with the switch
/// that cancels each: what a Cancel naming the request throws.
#[derive(Clone, Default)]
struct Cancels(Arc<Mutex<HashMap<u64, CancelFlag>>>);

impl Cancels {
    /// Register `request_id` as in flight; its entry goes when the guard
    /// drops, with the statement.
    fn register(&self, request_id: u64) -> CancelGuard {
        let flag = CancelFlag::new();
        self.0.lock().insert(request_id, flag.clone());
        CancelGuard {
            cancels: self.clone(),
            request_id,
            flag,
        }
    }

    /// Cancel the statement `request_id`, if it is in flight.
    fn cancel(&self, request_id: u64) {
        if let Some(flag) = self.0.lock().get(&request_id) {
            flag.cancel();
        }
    }
}

/// A statement's registration among its session's cancellable requests.
struct CancelGuard {
    cancels: Cancels,
    request_id: u64,
    flag: CancelFlag,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.cancels.0.lock().remove(&self.request_id);
    }
}

/// One statement on its way to the engine.
struct Statement {
    request_id: u64,
    query: String,
    params: HashMap<String, Value>,
    /// The settings it runs under: its own over its session's at the moment
    /// it was received.
    settings: ConnectionSettings,
    /// Where in the client's code it was issued. Boxed: only a client in
    /// debug mode sends one, and inline it would triple the size of every
    /// statement queued to a transaction.
    source: Option<Box<StatementSource>>,
    /// Its registration as cancellable, from the moment it is received.
    cancel: CancelGuard,
}

/// A message in a transaction's serial mailbox.
enum TxnMsg {
    /// A statement to run inside the transaction.
    Statement {
        /// Client-assigned sequence number; orders ORDERED transactions, ignored
        /// for UNORDERED.
        nonce: u64,
        statement: Statement,
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
            TxnMsg::Statement { statement, .. } => {
                if aborted {
                    send_error(out, statement.request_id, aborted_txn()).await;
                } else {
                    registry.touch_txn(session_id, txid);
                    if !execute(engine, statement, txid, out).await {
                        aborted = true;
                        registry.end_txn(session_id, txid);
                    }
                }
            }
            TxnMsg::Commit { request_id, .. } => {
                if aborted {
                    send_error(out, request_id, aborted_txn()).await;
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
    // The ORDERED nonce contract is 1-based and contiguous: a client numbers its
    // chain 1, 2, 3, ... and sends `last_nonce` on commit. A statement at nonce 0
    // (the UNORDERED default) therefore never becomes applicable here and the
    // commit drains it as a gap; callers wanting arrival order use UNORDERED.
    let mut next_nonce = 1u64;
    let mut buffer: BTreeMap<u64, Statement> = BTreeMap::new();
    let mut aborted = false;
    let mut resolved = false;

    while let Some(msg) = rx.recv().await {
        match msg {
            TxnMsg::Statement { nonce, statement } => {
                registry.request_started(session_id);
                if aborted {
                    send_error(out, statement.request_id, aborted_txn()).await;
                    registry.request_finished(session_id);
                    continue;
                }
                registry.touch_txn(session_id, txid);
                buffer.insert(nonce, statement);
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
                    send_error(out, request_id, aborted_txn()).await;
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
    buffer: &mut BTreeMap<u64, Statement>,
    out: &mpsc::Sender<OutEvent>,
) -> bool {
    while let Some(statement) = buffer.remove(next_nonce) {
        registry.touch_txn(session_id, txid);
        let ok = execute(engine, statement, txid, out).await;
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
    buffer: &mut BTreeMap<u64, Statement>,
    rx: &mut mpsc::Receiver<TxnMsg>,
    out: &mpsc::Sender<OutEvent>,
) -> bool {
    if apply_contiguous(engine, registry, session_id, txid, next_nonce, buffer, out).await {
        return true;
    }
    while *next_nonce <= last_nonce {
        match tokio::time::timeout(drain, rx.recv()).await {
            Ok(Some(TxnMsg::Statement { nonce, statement })) => {
                registry.request_started(session_id);
                buffer.insert(nonce, statement);
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
    buffer: &mut BTreeMap<u64, Statement>,
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
        Ok(Ok(receipt)) => {
            let _ = out
                .send((request_id, SessionEvent::Committed { receipt }))
                .await;
        }
        Ok(Err(e)) => send_error(out, request_id, e.0).await,
        Err(join) => send_error(out, request_id, lost_call(join)).await,
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
    statement: Statement,
    txid: u64,
    out: &mpsc::Sender<OutEvent>,
) -> bool {
    let Statement {
        request_id,
        query,
        params,
        settings,
        source,
        cancel,
    } = statement;
    // Open on the blocking pool: a write statement commits through Raft here.
    let engine = Arc::clone(engine);
    let flag = cancel.flag.clone();
    let opened = tokio::task::spawn_blocking(move || {
        let cursor =
            engine.open_cursor(&query, params, txid, &settings, source.as_deref(), &flag)?;
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
            send_error(out, request_id, lost_call(join)).await;
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
        // A cancelled statement stops paging: the rows already sent are not
        // a complete answer, and the error says so.
        if cancel.flag.is_cancelled() {
            send_error(
                out,
                request_id,
                Failure::new(ErrorCode::Cancelled, "the statement was cancelled"),
            )
            .await;
            return false;
        }
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
                send_error(out, request_id, lost_call(join)).await;
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
async fn send_error(out: &mpsc::Sender<OutEvent>, request_id: u64, failure: Failure) {
    let _ = out.send((request_id, SessionEvent::Error(failure))).await;
}

/// The failure of a statement or commit sent to a transaction that already
/// failed one: the transaction has to be rolled back before anything else.
fn aborted_txn() -> Failure {
    Failure::new(ErrorCode::FailedPrecondition, ABORTED_TXN)
}

/// A blocking engine call that did not return (it panicked or was cancelled).
fn lost_call(join: tokio::task::JoinError) -> Failure {
    Failure::internal(join.to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
