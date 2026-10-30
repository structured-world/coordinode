//! Session lifecycle and dispatch.
//!
//! [`SessionManager`] opens [`Session`]s; a session is driven by [`Session::run`],
//! which reads neutral ops from one channel and funnels neutral events to
//! another, with the single outbound writer living here, not in the transport
//! binding. Every request runs on its own task, so three concurrent queries
//! are three independent cursors. An interactive transaction is not the
//! session's: it lives on its owner, which runs its commands one at a time
//! in the order they arrive and ends it; the session only attaches to it. The
//! commands one stream sends for one transaction keep their order on the way
//! there: each is passed on once the one before it has its place at the
//! owner (runs, or waits in the owner's queue), not once it has answered, so
//! a pipelined chain flows. Closing the stream between commands detaches it;
//! the transactions stay open for another stream to continue. A statement of
//! a transaction whose answer cannot reach the client, because the stream
//! went away while it ran, aborts the transaction: the client cannot know
//! what it did, and a statement never runs twice. A query is run by opening
//! a server-side cursor through the injected [`CursorEngine`] and paging it
//! into `CursorOpen` / `Rows`* / `CursorEnd`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use coordinode_core::graph::types::Value;
use coordinode_core::txn::command::{AdmissionSignal, TransactionId};
use parking_lot::Mutex;
use tokio::sync::{Notify, mpsc, watch};

use coordinode_core::budget::CancelFlag;
use coordinode_core::txn::operation::OperationDescriptor;

use crate::engine::{CursorEngine, EngineError, InTransaction, TransactionStart};
use crate::registry::SessionRegistry;
use crate::types::{
    ConnectionSettings, ConnectionState, ErrorCode, Failure, SessionEvent, SessionOp, SessionStats,
    StatementSource,
};

/// An inbound op tagged with its session-scoped request id.
pub type InOp = (u64, SessionOp);

/// An outbound event tagged with the request id it answers.
pub type OutEvent = (u64, SessionEvent);

/// Rows pulled from a cursor per batch before the next `Rows` event is emitted.
const CURSOR_BATCH: usize = 1024;

/// Opens sessions backed by one engine and one shared registry.
pub struct SessionManager {
    engine: Arc<dyn CursorEngine>,
    registry: Arc<SessionRegistry>,
    connection: watch::Receiver<ConnectionState>,
}

impl SessionManager {
    /// Create a session manager backed by `engine` and the shared session
    /// `registry` that powers operational introspection (`SHOW SESSIONS` /
    /// `SHOW TRANSACTIONS`). Clients draw their transaction IDs.
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
/// [`SessionEvent`]s. It holds the query engine and the shared session
/// registry; clients draw their transaction IDs, and the transactions
/// themselves live on the leader.
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
    /// the session when the stream closes. The transactions the session used
    /// stay open on their owner: closing a stream detaches it, it does not roll
    /// anything back.
    pub async fn run(mut self, mut ops: mpsc::Receiver<InOp>, out: mpsc::Sender<OutEvent>) {
        let mut lanes = Lanes::default();
        loop {
            tokio::select! {
                maybe_op = ops.recv() => {
                    let Some((request_id, op)) = maybe_op else { break };
                    self.handle_op(request_id, op, &mut lanes, &out).await;
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
        self.registry.close_session(self.session_id);
    }

    /// Dispatch one inbound op: a statement, a transaction command or a
    /// session control.
    async fn handle_op(
        &self,
        request_id: u64,
        op: SessionOp,
        lanes: &mut Lanes,
        out: &mpsc::Sender<OutEvent>,
    ) {
        // A session SET changes this session's settings, on this task like a
        // Configure so the next statement already runs under it, and answers
        // with the empty result the statement has. Never the engine's own
        // settings: those are every other session's too.
        if let SessionOp::Execute { query, .. } = &op
            && let Some(change) = self.engine.session_setting(query)
        {
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
        match op {
            // A transaction's operation is named on its begin or its commit:
            // one named on a statement inside it would bind part of the
            // transaction.
            SessionOp::Execute {
                transaction: Some(_),
                operation: Some(_),
                ..
            } => {
                send_error(
                    out,
                    request_id,
                    Failure::new(
                        ErrorCode::InvalidArgument,
                        "a statement inside a transaction carries no operation; name it on \
                         the transaction's begin or commit",
                    ),
                )
                .await;
            }
            // A transaction runs under the settings fixed at its begin: a
            // statement naming its own would silently not get them.
            SessionOp::Execute {
                transaction: Some(_),
                ref settings,
                ..
            } if settings.names_concern() => {
                send_error(
                    out,
                    request_id,
                    Failure::new(
                        ErrorCode::InvalidArgument,
                        "a statement inside a transaction names no concern or read \
                         preference; the transaction runs under those its begin took from \
                         the session",
                    ),
                )
                .await;
            }
            // Its settings are taken now, so a Configure that arrives after
            // it does not reach back.
            SessionOp::Execute {
                query,
                params,
                transaction,
                settings,
                source,
                operation,
            } => {
                let statement = Statement {
                    request_id,
                    query,
                    params,
                    settings: self.settings.lock().under(&settings),
                    source,
                    operation,
                    cancel: self.cancels.register(request_id),
                };
                match transaction {
                    None => self.spawn(out, |engine, _, out| async move {
                        execute(&engine, statement, None, &out).await;
                    }),
                    Some(id) => {
                        // The transaction's first statement carries its begin.
                        let start = lanes.starts.remove(&id).map(|pending| pending.start);
                        let lane = lanes.step(id);
                        self.registry.attach(self.session_id, id);
                        self.spawn(out, move |engine, _, out| async move {
                            lane.after_previous().await;
                            let transaction = InTransaction {
                                id,
                                admission: lane.signal(),
                                start,
                            };
                            // The stream going away stops the statement: its
                            // answer has nowhere to go.
                            let watch = {
                                let out = out.clone();
                                let flag = statement.cancel.flag.clone();
                                tokio::spawn(async move {
                                    out.closed().await;
                                    flag.cancel();
                                })
                            };
                            let delivered =
                                execute(&engine, statement, Some(&transaction), &out).await;
                            watch.abort();
                            if !delivered {
                                abandon(engine, id).await;
                            }
                            lane.mine.release();
                        });
                    }
                }
            }

            // Nothing is sent: the transaction's first command carries the
            // begin, so opening it costs no call of its own. The session's
            // settings are taken now: they are the transaction's for its
            // life. A refused start answers on the command that carried it.
            SessionOp::Begin {
                transaction,
                operation,
            } => {
                let pending = PendingStart {
                    start: TransactionStart {
                        settings: self.settings.lock().clone(),
                        operation,
                    },
                    request_id,
                };
                self.registry.attach(self.session_id, transaction);
                // A second begin under an ID whose first was not sent yet:
                // the first opens it, and this one is refused on its own
                // request as a start naming an open ID, as when the first
                // had run.
                if let Some(earlier) = lanes.starts.insert(transaction, pending) {
                    self.open(transaction, earlier, lanes, out);
                    if let Some(this) = lanes.starts.remove(&transaction) {
                        self.open(transaction, this, lanes, out);
                    }
                }
            }

            SessionOp::Commit {
                transaction,
                operation,
                expect,
            } => {
                // A transaction whose begin no statement carried opens first.
                if let Some(pending) = lanes.starts.remove(&transaction) {
                    self.open(transaction, pending, lanes, out);
                }
                let lane = lanes.step(transaction);
                let session_id = self.session_id;
                self.spawn(out, move |engine, registry, out| async move {
                    lane.after_previous().await;
                    let admission = lane.signal();
                    let committed = blocking(move || {
                        engine.commit_transaction(
                            transaction,
                            operation.as_ref(),
                            &expect,
                            &admission,
                        )
                    })
                    .await;
                    lane.mine.release();
                    registry.detach(session_id, &transaction);
                    let event = match committed {
                        Ok(receipt) => SessionEvent::Committed { receipt },
                        Err(failure) => SessionEvent::Error(failure),
                    };
                    let _ = out.send((request_id, event)).await;
                });
            }

            // Silent, as a rollback of a transaction already resolved is: the
            // client asked to discard, and nothing it can do depends on the
            // answer.
            SessionOp::Rollback { transaction } => {
                // A begin no command carried yet opened nothing to discard.
                if lanes.starts.remove(&transaction).is_some() {
                    self.registry.detach(self.session_id, &transaction);
                    return;
                }
                let lane = lanes.step(transaction);
                let session_id = self.session_id;
                self.spawn(out, move |engine, registry, _| async move {
                    lane.after_previous().await;
                    let admission = lane.signal();
                    let _ = blocking(move || engine.rollback_transaction(transaction, &admission))
                        .await;
                    lane.mine.release();
                    registry.detach(session_id, &transaction);
                });
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

    /// Open `transaction` with `pending` as a command of its own, in its lane:
    /// for a transaction whose first command is not a statement. Only a
    /// refusal answers, on the begin's request.
    fn open(
        &self,
        transaction: TransactionId,
        pending: PendingStart,
        lanes: &mut Lanes,
        out: &mpsc::Sender<OutEvent>,
    ) {
        let lane = lanes.step(transaction);
        self.spawn(out, move |engine, _, out| async move {
            lane.after_previous().await;
            let PendingStart { start, request_id } = pending;
            let begun = blocking(move || {
                engine.begin_transaction(transaction, &start.settings, start.operation.as_ref())
            })
            .await;
            lane.mine.release();
            if let Err(failure) = begun {
                send_error(&out, request_id, failure).await;
            }
        });
    }

    /// Run `request` on its own task, counted as in flight for its lifetime.
    fn spawn<F, Fut>(&self, out: &mpsc::Sender<OutEvent>, request: F)
    where
        F: FnOnce(Arc<dyn CursorEngine>, Arc<SessionRegistry>, mpsc::Sender<OutEvent>) -> Fut,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let registry = Arc::clone(&self.registry);
        let session_id = self.session_id;
        registry.request_started(session_id);
        let work = request(
            Arc::clone(&self.engine),
            Arc::clone(&self.registry),
            out.clone(),
        );
        tokio::spawn(async move {
            work.await;
            registry.request_finished(session_id);
        });
    }
}

/// Run a blocking engine call on the blocking pool: a commit drives a Raft
/// proposal, and a wait for a transaction's turn parks its thread.
async fn blocking<T: Send + 'static>(
    call: impl FnOnce() -> Result<T, EngineError> + Send + 'static,
) -> Result<T, Failure> {
    match tokio::task::spawn_blocking(call).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(e.0),
        Err(join) => Err(lost_call(join)),
    }
}

/// The commands this stream sent for each transaction, each one a step that
/// the next waits on, and the begins its transactions' first commands carry.
#[derive(Default)]
struct Lanes {
    steps: HashMap<TransactionId, Arc<LaneStep>>,
    starts: HashMap<TransactionId, PendingStart>,
}

impl Lanes {
    /// The step of the next command for transaction `id`, after the one sent
    /// before.
    fn step(&mut self, id: TransactionId) -> LanePosition {
        // Steps already passed hold nobody back.
        self.steps.retain(|_, step| !step.is_released());
        let mine = Arc::new(LaneStep::default());
        let previous = self.steps.insert(id, Arc::clone(&mine));
        LanePosition { previous, mine }
    }
}

/// A begin this stream received and has not sent yet: the transaction's
/// first command carries it.
struct PendingStart {
    start: TransactionStart,
    /// The begin's request, answered when the start is refused on its own.
    request_id: u64,
}

/// Where a command stands in its stream's order for one transaction.
struct LanePosition {
    previous: Option<Arc<LaneStep>>,
    mine: Arc<LaneStep>,
}

impl LanePosition {
    /// Wait until the command sent before has its place at the owner.
    async fn after_previous(&self) {
        if let Some(previous) = &self.previous {
            previous.released().await;
        }
    }

    /// The signal the owner tells when this command has its place.
    fn signal(&self) -> AdmissionSignal {
        let mine = Arc::clone(&self.mine);
        AdmissionSignal::new(move || mine.release())
    }
}

/// One command's place in its lane: released once the owner has placed it,
/// or once it has answered, whichever comes first.
#[derive(Default)]
struct LaneStep {
    released: AtomicBool,
    notify: Notify,
}

impl LaneStep {
    fn release(&self) {
        if !self.released.swap(true, AtomicOrdering::AcqRel) {
            self.notify.notify_waiters();
        }
    }

    fn is_released(&self) -> bool {
        self.released.load(AtomicOrdering::Acquire)
    }

    async fn released(&self) {
        loop {
            // Registered before the check, so a release between the two is
            // not missed.
            let notified = self.notify.notified();
            if self.is_released() {
                return;
            }
            notified.await;
        }
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
    /// statement waiting for its turn.
    source: Option<Box<StatementSource>>,
    /// The client operation an autonomous statement carries out.
    operation: Option<OperationDescriptor>,
    /// Its registration as cancellable, from the moment it is received.
    cancel: CancelGuard,
}

/// Open a server-side cursor for one statement and page it into a
/// `CursorOpen` → `Rows`* → `CursorEnd` sequence (or a single `Error`).
///
/// `open_cursor` and `next_batch` are synchronous and may block for a long time:
/// a write drives a Raft commit (`block_in_place` + `block_on`), a read pages
/// from storage, and a transaction's statement may wait for its turn. Running
/// them directly on the async worker thread starves the runtime: under a burst
/// of concurrent writes every worker blocks inside a commit, leaving no worker
/// to drive the Raft apply loop the commits wait on, which deadlocks the whole
/// runtime. So each blocking call runs on the blocking pool via
/// [`spawn_blocking`](tokio::task::spawn_blocking); the cursor (`Send`) moves in
/// and back out per batch, keeping the stream's per-batch backpressure.
///
/// Returns whether the whole answer, the error included, was handed to the
/// stream.
async fn execute(
    engine: &Arc<dyn CursorEngine>,
    statement: Statement,
    transaction: Option<&InTransaction>,
    out: &mpsc::Sender<OutEvent>,
) -> bool {
    let Statement {
        request_id,
        query,
        params,
        settings,
        source,
        operation,
        cancel,
    } = statement;
    // Open on the blocking pool: a write statement commits through Raft here.
    let engine = Arc::clone(engine);
    let flag = cancel.flag.clone();
    let transaction = transaction.cloned();
    let opened = tokio::task::spawn_blocking(move || {
        let cursor = engine.open_cursor(
            &query,
            params,
            transaction.as_ref(),
            &settings,
            source.as_deref(),
            operation.as_ref(),
            &flag,
        )?;
        let columns = cursor.columns();
        Ok::<_, EngineError>((cursor, columns))
    })
    .await;
    let (mut cursor, columns) = match opened {
        Ok(Ok(opened)) => opened,
        Ok(Err(e)) => return send_error(out, request_id, e.0).await,
        Err(join) => return send_error(out, request_id, lost_call(join)).await,
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
            return send_error(
                out,
                request_id,
                Failure::new(ErrorCode::Cancelled, "the statement was cancelled"),
            )
            .await;
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
            Err(join) => return send_error(out, request_id, lost_call(join)).await,
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
            Err(e) => return send_error(out, request_id, e.0).await,
        }
    }

    out.send((
        request_id,
        SessionEvent::CursorEnd {
            stats: cursor.stats(),
        },
    ))
    .await
    .is_ok()
}

/// Abort `transaction`, whose statement's answer did not reach the client.
/// One its failed statement already ended is refused, which is the outcome
/// wanted.
async fn abandon(engine: Arc<dyn CursorEngine>, transaction: TransactionId) {
    if let Err(failure) = blocking(move || engine.abandon_transaction(transaction)).await {
        tracing::debug!(%transaction, reason = %failure.message, "abandoned transaction already ended");
    }
}

/// Emit a single `Error` event for a failed request. Returns whether it was
/// handed to the stream.
async fn send_error(out: &mpsc::Sender<OutEvent>, request_id: u64, failure: Failure) -> bool {
    out.send((request_id, SessionEvent::Error(failure)))
        .await
        .is_ok()
}

/// A blocking engine call that did not return (it panicked or was cancelled).
fn lost_call(join: tokio::task::JoinError) -> Failure {
    Failure::internal(join.to_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
