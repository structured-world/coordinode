//! Whole commands carried from the member a client reached (the ingress) to
//! the member that runs them: an autonomous statement to the leader, a
//! transaction's commands to the leader that holds it.
//!
//! The executing member selects the view, evaluates, validates and commits;
//! the ingress never ships a write set it evaluated itself. A command is
//! routed once: a member that cannot run it answers with the refusal that
//! says why, and the ingress decides what to tell its client.
//!
//! The library owns the transport and the encoding, the encoding of failures
//! included. What runs a command at the executing member is a
//! [`CommandExecutor`]: the server brings its own, which fences and records
//! commands the way it does for its own clients. A command that ran and
//! failed answers with its [`DatabaseError`] in the body of the answer, so the
//! ingress returns the same error the executing member raised; a failure to
//! deliver the answer is never mistaken for one.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use coordinode_core::budget::CancelFlag;
use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::{Value, VectorConsistencyMode};
use coordinode_core::txn::command::{
    ActingContext, AdmissionSignal, ExpectedVersion, Principal, TimeoutOutcome, TransactionId,
};
use coordinode_core::txn::operation::OperationDescriptor;
use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_core::txn::transaction::CommitReceipt;
use coordinode_core::txn::write_concern::{Journal, WriteAck, WriteConcern};
use coordinode_query::advisor::source::SourceContext;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proto::internode::routing as wire;
use coordinode_wire::peer::PeerConnectInfo;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::{Code, Request, Response, Status};

use crate::db::{DatabaseError, PeerFailure, TransactionEnd};
use crate::operation::{IssuerId, OperationOutcome, Resolve};
use coordinode_core::group::DeploymentId;
use wire::command_routing_service_client::CommandRoutingServiceClient;
use wire::command_routing_service_server::{CommandRoutingService, CommandRoutingServiceServer};
use wire::routed_statement_event::Event;

mod errors;
pub use errors::{ERROR_CODEC_VERSION, MAX_ROUTED_ERROR};

/// Request metadata naming the error encoding the ingress reads.
const CODEC_HEADER: &str = "cn-routed-error-codec";

/// Response metadata of a call refused before it was read: the command did
/// not run. Its value says why.
const REFUSED_HEADER: &str = "cn-routing-refused";

/// The encoded size a batch of rows grows to before it is sent.
const ROW_BATCH_BYTES: usize = 1 << 20;

/// Frames of one answer the executing member sends ahead of the ingress
/// before it waits for the ingress to take them.
const FRAMES_AHEAD: u32 = 4;

/// The largest message a member takes from another on this service: a batch
/// of rows that holds one very large row.
pub const MAX_ROUTED_MESSAGE: usize = 64 << 20;

/// The settings a routed statement runs under, as its ingress resolved them.
/// `None` leaves the setting to the executing member.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoutedSettings {
    /// What the read may observe.
    pub read_concern: Option<ReadConcernLevel>,
    /// Log index the read waits for.
    pub after_index: Option<u64>,
    /// Timestamp a snapshot read is pinned to.
    pub at_timestamp: Option<u64>,
    /// When a write is acknowledged.
    pub write_concern: Option<WriteConcern>,
    /// Vector consistency below a hint in the query.
    pub vector_consistency: Option<VectorConsistencyMode>,
    /// Bound on waiting for a vector index still being built.
    pub vector_build_wait: Option<Duration>,
    /// Memory limit of the statement in bytes.
    pub query_memory_limit: Option<u64>,
}

/// One statement to run at the member that executes it.
#[derive(Debug, Clone)]
pub struct RoutedStatement {
    /// Who the statement acts for, as the ingress verified it.
    pub actor: ActingContext,
    /// The statement's text.
    pub query: String,
    /// Its parameters.
    pub params: HashMap<String, Value>,
    /// The settings it runs under.
    pub settings: RoutedSettings,
    /// The transaction it runs in; `None` commits it on its own.
    pub transaction: Option<TransactionId>,
    /// The client operation an autonomous statement carries out.
    pub operation: Option<OperationDescriptor>,
    /// What is left of the caller's deadline.
    pub deadline: Option<Duration>,
    /// Where in the client's code it was issued.
    pub source: Option<SourceContext>,
    /// On the first statement of a transaction: how the transaction opens.
    /// It opens and the statement runs in it in one call, with no separate
    /// begin before it.
    pub start: Option<TransactionStart>,
}

/// How the command that starts a transaction opens it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransactionStart {
    /// The read concern of its statements; the leader's when `None`.
    pub read_concern: Option<ReadConcernLevel>,
    /// The write concern of its commit; the leader's when `None`.
    pub write_concern: Option<WriteConcern>,
    /// What ends it when it stays idle; the leader's setting when `None`.
    pub timeout_outcome: Option<TimeoutOutcome>,
    /// The client operation its commit carries out.
    pub operation: Option<OperationDescriptor>,
}

impl RoutedBegin {
    /// Open `transaction` for `actor` as `start` says.
    pub fn new(transaction: TransactionId, actor: ActingContext, start: TransactionStart) -> Self {
        Self {
            transaction,
            actor,
            read_concern: start.read_concern,
            write_concern: start.write_concern,
            timeout_outcome: start.timeout_outcome,
            operation: start.operation,
        }
    }
}

/// What a finished statement reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoutedStats {
    /// Nodes it created.
    pub nodes_created: i64,
    /// Nodes it deleted.
    pub nodes_deleted: i64,
    /// Relationships it created.
    pub edges_created: i64,
    /// Relationships it deleted.
    pub edges_deleted: i64,
    /// Properties it set.
    pub properties_set: i64,
    /// Log index its own commit was applied at; zero for none.
    pub applied_index: u64,
    /// Timestamp its own commit landed at; zero for none.
    pub commit_ts: u64,
    /// Log index the read was served at on the executing member.
    pub read_index: u64,
    /// What a read on a read-only member is as of; zero otherwise.
    pub read_as_of_ts: u64,
}

/// A statement's answer: its columns, its rows in column order and its
/// statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoutedAnswer {
    /// The column names.
    pub columns: Vec<String>,
    /// The rows, each a value per column.
    pub rows: Vec<Vec<Value>>,
    /// What the statement reports.
    pub stats: RoutedStats,
}

/// Open an interactive transaction at the leader.
#[derive(Debug, Clone)]
pub struct RoutedBegin {
    /// The ID the client drew for it.
    pub transaction: TransactionId,
    /// Who the transaction acts for; every later command must act for the
    /// same.
    pub actor: ActingContext,
    /// The read concern of its statements; the leader's when `None`.
    pub read_concern: Option<ReadConcernLevel>,
    /// The write concern of its commit; the leader's when `None`.
    pub write_concern: Option<WriteConcern>,
    /// What ends it when it stays idle; the leader's setting when `None`.
    pub timeout_outcome: Option<TimeoutOutcome>,
    /// The client operation its commit carries out.
    pub operation: Option<OperationDescriptor>,
}

/// Commit an interactive transaction at the leader.
#[derive(Debug, Clone)]
pub struct RoutedCommit {
    /// Who the commit acts for.
    pub actor: ActingContext,
    /// The transaction.
    pub transaction: TransactionId,
    /// The client operation the commit carries out; the one its begin named
    /// when `None`.
    pub operation: Option<OperationDescriptor>,
    /// What is left of the caller's deadline.
    pub deadline: Option<Duration>,
    /// The node versions the caller built its statements on.
    pub expect: Vec<ExpectedVersion>,
}

/// Roll back an interactive transaction at the leader.
#[derive(Debug, Clone)]
pub struct RoutedRollback {
    /// Who the rollback acts for.
    pub actor: ActingContext,
    /// The transaction.
    pub transaction: TransactionId,
    /// The client did not ask for it: the stream a statement of the
    /// transaction ran on was lost.
    pub abandoned: bool,
}

impl RoutedRollback {
    /// Why the transaction ends.
    pub fn end(&self) -> TransactionEnd {
        if self.abandoned {
            TransactionEnd::StreamLost
        } else {
            TransactionEnd::RolledBack
        }
    }
}

/// A call on the records of client operations, decided by the leader's log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutedOperationCall {
    /// Enroll a new incarnation of this issuer.
    Enroll(IssuerId),
    /// Resolve an operation's outcome, excluding it through the log first
    /// when nothing is recorded of it and `Resolve::Exclude` asks for that.
    Resolve(OperationDescriptor, Resolve),
    /// Release the outcomes of these settled operations.
    Acknowledge(Vec<OperationDescriptor>),
    /// Retire the operations of the descriptor's incarnation numbered below
    /// its sequence.
    Retire(OperationDescriptor),
}

/// The answer to a [`RoutedOperationCall`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutedOperationAnswer {
    /// The incarnation an enrollment opened.
    Enrolled {
        /// Its number.
        incarnation: u64,
        /// The first operation number it hands out.
        next_sequence: u64,
        /// The deployment that enrolled it.
        deployment: DeploymentId,
    },
    /// How a resolved operation stands.
    Outcome(OperationOutcome),
    /// A release or retirement is done.
    Done,
}

/// Runs the commands other members route here.
///
/// Every method blocks: a write commits through consensus inside it, so the
/// service calls it from the blocking pool. A failure is the error the
/// command failed with here, which the ingress returns as it is.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot run commands routed from other members",
    label = "not a command executor",
    note = "implement `CommandExecutor` over the statement path that serves this member's own clients, so a routed command is fenced, run and refused the same way"
)]
pub trait CommandExecutor: Send + Sync + 'static {
    /// Run `statement`, writing its rows to `answer` as they are produced,
    /// and return its statistics. A statement of a transaction tells
    /// `admitted` when it has its place in the transaction's order; one that
    /// ends meanwhile was cancelled through `cancel`, and one whose
    /// [`AnswerWriter::row`] says the ingress left may stop.
    ///
    /// # Errors
    ///
    /// What the statement failed with, before or after rows were written.
    fn execute(
        &self,
        statement: RoutedStatement,
        cancel: &CancelFlag,
        admitted: &AdmissionSignal,
        answer: &mut AnswerWriter,
    ) -> Result<RoutedStats, DatabaseError>;

    /// Open a transaction under the ID the client drew.
    ///
    /// # Errors
    ///
    /// Why the transaction was not opened.
    fn begin(&self, begin: RoutedBegin) -> Result<(), DatabaseError>;

    /// Commit a transaction this member holds.
    ///
    /// # Errors
    ///
    /// What the commit failed with.
    fn commit(&self, commit: RoutedCommit) -> Result<CommitReceipt, DatabaseError>;

    /// Roll back a transaction this member holds.
    ///
    /// # Errors
    ///
    /// Why the rollback was refused.
    fn rollback(&self, rollback: RoutedRollback) -> Result<(), DatabaseError>;

    /// Carry out a call on operation records, acting for `actor`.
    ///
    /// # Errors
    ///
    /// Why the call was refused.
    fn operation(
        &self,
        actor: ActingContext,
        call: RoutedOperationCall,
    ) -> Result<RoutedOperationAnswer, DatabaseError>;
}

/// Decides, before each command of a session, whether the connection it
/// arrived on may still route commands here.
pub type CommandAdmission = fn(Option<&PeerConnectInfo>) -> Result<(), Status>;

/// The service a member serves its peers the commands they route with.
pub struct CommandRoutingHandler<E> {
    executor: Arc<E>,
    admit: Option<CommandAdmission>,
}

impl<E: CommandExecutor> CommandRoutingHandler<E> {
    /// A handler that runs routed commands through `executor`.
    pub fn new(executor: Arc<E>) -> Self {
        Self {
            executor,
            admit: None,
        }
    }

    /// Check every command of a session with `admit` before it runs, not
    /// only the session's opening: a credential that expired or was
    /// withdrawn while the session stayed open stops the commands after it.
    #[must_use]
    pub fn admitting(mut self, admit: CommandAdmission) -> Self {
        self.admit = Some(admit);
        self
    }

    /// The handler as a gRPC service taking messages up to
    /// `max_decoding_bytes`.
    pub fn into_service(self, max_decoding_bytes: usize) -> CommandRoutingServiceServer<Self> {
        CommandRoutingServiceServer::new(self).max_decoding_message_size(max_decoding_bytes)
    }
}

/// A session refused before any command of it was read, for `why`: none of
/// them ran, which the ingress learns from the metadata rather than the text.
fn refused(status: Status, why: &'static str) -> Status {
    let mut status = status;
    status
        .metadata_mut()
        .insert(REFUSED_HEADER, MetadataValue::from_static(why));
    status
}

/// Refuse `request` unless its ingress reads the error encoding this member
/// writes.
fn check_codec<T>(request: &Request<T>) -> Result<(), Status> {
    let theirs = request
        .metadata()
        .get(CODEC_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u32>().ok());
    if theirs == Some(ERROR_CODEC_VERSION) {
        return Ok(());
    }
    Err(refused(
        Status::failed_precondition(format!(
            "the ingress reads routed errors in encoding {theirs:?}, this member writes \
             encoding {ERROR_CODEC_VERSION}"
        )),
        "codec",
    ))
}

/// The replies of one session, queued for its stream. Unbounded on its own,
/// and bounded by what produces them: one reply per short command, one
/// admission per statement, and a statement's frames only as far ahead as
/// its credit allows.
type Replies = tokio::sync::mpsc::UnboundedSender<Result<wire::RoutedReply, Status>>;

fn reply(call: u64, reply: wire::routed_reply::Reply) -> Result<wire::RoutedReply, Status> {
    Ok(wire::RoutedReply {
        call,
        reply: Some(reply),
    })
}

/// Queue `answer` to `call`'s ingress. A closed queue means the session is
/// over: the reply has nobody to reach, and whatever still runs for the
/// session is being stopped.
fn queue(replies: &Replies, call: u64, answer: wire::routed_reply::Reply) {
    if replies.send(reply(call, answer)).is_err() {
        tracing::debug!(call, "a routing session ended before its reply");
    }
}

/// How many more frames of one answer the member may send, and whether the
/// ingress still reads it. The statement's thread waits here.
struct Credit {
    // no-std: spin::Mutex with a caller-provided wait; taken twice per frame.
    state: parking_lot::Mutex<CreditState>,
    ready: parking_lot::Condvar,
}

struct CreditState {
    left: u32,
    /// The ingress no longer reads the answer.
    closed: bool,
}

impl Credit {
    fn new(left: u32) -> Self {
        Self {
            state: parking_lot::Mutex::new(CreditState {
                left,
                closed: false,
            }),
            ready: parking_lot::Condvar::new(),
        }
    }

    /// The ingress took `frames` more frames. A grant past what a counter
    /// holds is not something a well-behaved ingress sends: the answer is
    /// treated as no longer read.
    fn grant(&self, frames: u32) {
        let mut state = self.state.lock();
        match state.left.checked_add(frames) {
            Some(left) => state.left = left,
            None => state.closed = true,
        }
        self.ready.notify_one();
    }

    fn close(&self) {
        self.state.lock().closed = true;
        self.ready.notify_one();
    }

    /// Wait for room for one frame; `false` once the answer is no longer
    /// read.
    fn take(&self) -> bool {
        let mut state = self.state.lock();
        loop {
            if state.closed {
                return false;
            }
            if state.left > 0 {
                state.left -= 1;
                return true;
            }
            self.ready.wait(&mut state);
        }
    }
}

/// A statement running for a session: how to stop it.
struct Running {
    cancel: CancelFlag,
    credit: Arc<Credit>,
}

impl Running {
    fn stop(&self) {
        self.cancel.cancel();
        self.credit.close();
    }
}

/// The statements a session has running, by the ingress's number.
// no-std: spin::Mutex; taken once per statement start, end, credit and cancel.
type RunningCalls = Arc<parking_lot::Mutex<rustc_hash::FxHashMap<u64, Running>>>;

/// One session at the executing member: reads the ingress's commands, runs
/// each on the blocking pool and queues its replies.
struct ServedSession<E> {
    executor: Arc<E>,
    admit: Option<CommandAdmission>,
    /// The connection the session arrived on, rechecked before each command.
    connection: Option<PeerConnectInfo>,
    replies: Replies,
    running: RunningCalls,
}

impl<E: CommandExecutor> ServedSession<E> {
    async fn run(self, mut requests: tonic::Streaming<wire::RoutedRequest>) {
        loop {
            let next = tokio::select! {
                next = requests.message() => next,
                () = self.replies.closed() => break,
            };
            match next {
                Ok(Some(request)) => self.dispatch(request),
                Ok(None) => break,
                Err(status) => {
                    tracing::debug!(%status, "a routing session's ingress went away");
                    break;
                }
            }
        }
        // Whatever still runs has lost its ingress: its client cannot know
        // what it did, so it stops, and its transaction is aborted.
        for (_, running) in self.running.lock().drain() {
            running.stop();
        }
    }

    fn refuse(&self, call: u64, reason: &str, message: String) {
        queue(
            &self.replies,
            call,
            wire::routed_reply::Reply::Refused(wire::RoutedRefusal {
                reason: reason.to_owned(),
                message,
            }),
        );
    }

    fn dispatch(&self, request: wire::RoutedRequest) {
        use wire::routed_request::Request as R;

        let call = request.call;
        match request.request {
            Some(R::Credit(frames)) => {
                if let Some(running) = self.running.lock().get(&call) {
                    running.credit.grant(frames);
                }
                return;
            }
            Some(R::Cancel(_)) => {
                if let Some(running) = self.running.lock().get(&call) {
                    running.stop();
                }
                return;
            }
            _ => {}
        }
        // A credential that expired or was withdrawn since the session
        // opened stops the commands that come after it.
        if let Some(admit) = self.admit {
            if let Err(status) = admit(self.connection.as_ref()) {
                self.refuse(call, "admission", status.message().to_owned());
                return;
            }
        }
        let result = match request.request {
            Some(R::Execute(statement)) => {
                statement_from_wire(*statement).map(|statement| self.execute(call, statement))
            }
            Some(R::Begin(begin)) => begin_from_wire(begin).map(|begin| {
                self.answer(call, move |e| {
                    wire::routed_reply::Reply::Begun(wire::RoutedBegun {
                        error: e.begin(begin).err().map(errors::encode).unwrap_or_default(),
                    })
                });
            }),
            Some(R::Commit(commit)) => commit_from_wire(commit).map(|commit| {
                self.answer(call, move |e| {
                    wire::routed_reply::Reply::Committed(match e.commit(commit) {
                        Ok(receipt) => wire::RoutedCommitted {
                            commit_ts: receipt.commit_ts.as_raw(),
                            applied_index: receipt.applied_index.unwrap_or(0),
                            error: Vec::new(),
                        },
                        Err(err) => wire::RoutedCommitted {
                            error: errors::encode(err),
                            ..wire::RoutedCommitted::default()
                        },
                    })
                });
            }),
            Some(R::Rollback(rollback)) => rollback_from_wire(rollback).map(|rollback| {
                self.answer(call, move |e| {
                    wire::routed_reply::Reply::RolledBack(wire::RoutedRolledBack {
                        error: e
                            .rollback(rollback)
                            .err()
                            .map(errors::encode)
                            .unwrap_or_default(),
                    })
                });
            }),
            Some(R::Operation(operation)) => actor_from_wire(operation.actor)
                .and_then(|actor| Ok((actor, operation_call_from_wire(operation.call)?)))
                .map(|(actor, operation)| {
                    self.answer(call, move |e| {
                        wire::routed_reply::Reply::Operation(match e.operation(actor, operation) {
                            Ok(answer) => operation_answer_to_wire(answer),
                            Err(err) => wire::RoutedOperationAnswer {
                                answer: Some(wire::routed_operation_answer::Answer::Error(
                                    errors::encode(err),
                                )),
                            },
                        })
                    });
                }),
            Some(R::Credit(_) | R::Cancel(_)) => Ok(()),
            None => Err(Status::invalid_argument(
                "a routed request names no command",
            )),
        };
        if let Err(status) = result {
            self.refuse(call, "request", status.message().to_owned());
        }
    }

    /// Run a short command on the blocking pool and queue its one reply.
    fn answer(
        &self,
        call: u64,
        run: impl FnOnce(&E) -> wire::routed_reply::Reply + Send + 'static,
    ) {
        let executor = Arc::clone(&self.executor);
        let replies = self.replies.clone();
        tokio::task::spawn_blocking(move || {
            queue(&replies, call, run(&executor));
        });
    }

    /// Start `statement` on the blocking pool; its answer streams under
    /// `call`.
    fn execute(&self, call: u64, statement: RoutedStatement) {
        let cancel = CancelFlag::new();
        let credit = Arc::new(Credit::new(FRAMES_AHEAD));
        {
            let mut running = self.running.lock();
            if running.contains_key(&call) {
                drop(running);
                self.refuse(
                    call,
                    "duplicate",
                    format!("command {call} is already running on this session"),
                );
                return;
            }
            running.insert(
                call,
                Running {
                    cancel: cancel.clone(),
                    credit: Arc::clone(&credit),
                },
            );
        }

        // A statement of a transaction tells the ingress when it has its
        // place, at once and ahead of any frame, so the next command can
        // follow. Told twice, it is sent once. An autonomous statement waits
        // behind nothing and its answer says all there is.
        let admitted = if statement.transaction.is_some() {
            let told = Arc::new(AtomicBool::new(false));
            let replies = self.replies.clone();
            AdmissionSignal::new(move || {
                if !told.swap(true, Ordering::AcqRel) {
                    queue(
                        &replies,
                        call,
                        wire::routed_reply::Reply::Frame(frame(vec![Event::Admitted(
                            wire::Admitted {},
                        )])),
                    );
                }
            })
        } else {
            AdmissionSignal::new(|| {})
        };

        // A statement of a transaction whose answer does not reach the
        // ingress leaves its client unable to know what it did, and a
        // statement never runs twice: the transaction is aborted.
        let abandon = statement.transaction.map(|transaction| RoutedRollback {
            actor: statement.actor.clone(),
            transaction,
            abandoned: true,
        });
        let executor = Arc::clone(&self.executor);
        let replies = self.replies.clone();
        let running = Arc::clone(&self.running);
        tokio::task::spawn_blocking(move || {
            let mut answer = AnswerWriter::new(Some(AnswerOut {
                call,
                replies,
                credit,
            }));
            let result = executor.execute(statement, &cancel, &admitted, &mut answer);
            drop(admitted);
            let delivered = answer.finish(result);
            running.lock().remove(&call);
            if !delivered {
                tracing::debug!("a routed statement's ingress left before its answer");
                if let Some(rollback) = abandon {
                    // A transaction the failed statement already ended is
                    // refused here, which is the outcome wanted.
                    if let Err(err) = executor.rollback(rollback) {
                        tracing::debug!(%err, "abandoned transaction already ended");
                    }
                }
            }
        });
    }
}

#[tonic::async_trait]
impl<E: CommandExecutor> CommandRoutingService for CommandRoutingHandler<E> {
    type SessionStream =
        tokio_stream::wrappers::UnboundedReceiverStream<Result<wire::RoutedReply, Status>>;

    async fn session(
        &self,
        request: Request<tonic::Streaming<wire::RoutedRequest>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        check_codec(&request)?;
        let connection = request.extensions().get::<PeerConnectInfo>().cloned();
        let (replies, answers) = tokio::sync::mpsc::unbounded_channel();
        let session = ServedSession {
            executor: Arc::clone(&self.executor),
            admit: self.admit,
            connection,
            replies,
            running: RunningCalls::default(),
        };
        tokio::spawn(session.run(request.into_inner()));
        Ok(Response::new(
            tokio_stream::wrappers::UnboundedReceiverStream::new(answers),
        ))
    }
}

/// Where the frames of one statement's answer go.
struct AnswerOut {
    call: u64,
    replies: Replies,
    credit: Arc<Credit>,
}

/// Where a member puts the answer of a routed statement while the statement
/// runs: its columns, then its rows in batches sent as they fill. A batch
/// waits for the ingress's credit, so an ingress that reads slowly holds the
/// statement back rather than the answer piling up here.
/// Events go out together: the columns with the first batch, the last batch
/// with the end, so an answer that fits one batch is one message.
/// Called from the blocking pool.
pub struct AnswerWriter {
    /// `None` for an answer nobody reads.
    out: Option<AnswerOut>,
    /// Events ready and not sent yet.
    pending: Vec<Event>,
    batch: Vec<u8>,
    columns: bool,
    /// The ingress left: nothing more reaches it.
    gone: bool,
    /// A row that could not be encoded: the answer ends with this failure.
    broken: Option<String>,
}

impl AnswerWriter {
    fn new(out: Option<AnswerOut>) -> Self {
        Self {
            out,
            pending: Vec::new(),
            batch: Vec::new(),
            columns: false,
            gone: false,
            broken: None,
        }
    }

    /// An answer nobody reads, for a caller that runs an executor directly
    /// for its result alone: the statement stops at its first batch.
    pub fn unread() -> Self {
        Self::new(None)
    }

    /// Give the column names, once and before any row; later calls are
    /// ignored. `false` once nothing more is worth producing.
    pub fn columns(&mut self, names: &[String]) -> bool {
        if !self.columns {
            self.columns = true;
            self.pending.push(Event::Columns(wire::Columns {
                names: names.to_vec(),
            }));
        }
        self.wanted()
    }

    /// Add one row, its values in column order. `false` once nothing more is
    /// worth producing: the ingress left, or a row could not be sent.
    pub fn row(&mut self, row: &[Value]) -> bool {
        debug_assert!(
            self.columns,
            "a routed answer's rows come after its columns"
        );
        if !self.wanted() {
            return false;
        }
        if let Err(e) = rmp_serde::encode::write(&mut self.batch, row) {
            self.broken = Some(format!("a result row could not be encoded: {e}"));
            return false;
        }
        if self.batch.len() >= ROW_BATCH_BYTES {
            self.take_batch();
            self.send();
        }
        self.wanted()
    }

    fn wanted(&self) -> bool {
        !self.gone && self.broken.is_none()
    }

    /// The rows written so far become one pending event.
    fn take_batch(&mut self) {
        if !self.batch.is_empty() {
            let batch = std::mem::take(&mut self.batch);
            self.pending.push(Event::Rows(batch));
        }
    }

    /// Send the pending events as one frame, once the ingress has room for
    /// it.
    fn send(&mut self) {
        let events = std::mem::take(&mut self.pending);
        if self.gone {
            return;
        }
        let Some(out) = &self.out else {
            self.gone = true;
            return;
        };
        if !out.credit.take()
            || out
                .replies
                .send(reply(
                    out.call,
                    wire::routed_reply::Reply::Frame(frame(events)),
                ))
                .is_err()
        {
            self.gone = true;
        }
    }

    /// End the answer with the statement's `result`: its statistics after
    /// the last rows, or its failure after the rows already written, which
    /// stay delivered. `false` when the ingress left before it had all of
    /// it.
    fn finish(mut self, result: Result<RoutedStats, DatabaseError>) -> bool {
        if let Some(broken) = self.broken.take() {
            // What the statement did is not what the ingress is told: its
            // outcome is unknown there, as for any answer lost on the way.
            return match &self.out {
                Some(out) if !self.gone => out
                    .replies
                    .send(reply(out.call, wire::routed_reply::Reply::Broken(broken)))
                    .is_ok(),
                _ => false,
            };
        }
        match result {
            Ok(stats) => {
                self.columns(&[]);
                self.take_batch();
                self.pending.push(Event::End(stats_to_wire(stats)));
            }
            Err(err) => {
                self.take_batch();
                self.pending.push(Event::Error(errors::encode(err)));
            }
        }
        self.send();
        !self.gone
    }
}

fn frame(events: Vec<Event>) -> wire::RoutedStatementFrame {
    wire::RoutedStatementFrame {
        events: events
            .into_iter()
            .map(|event| wire::RoutedStatementEvent { event: Some(event) })
            .collect(),
    }
}

/// The ingress's side of routing: sends commands to one member and returns
/// its answers.
///
/// A command the member ran and refused returns the error it raised there.
/// A command the member refused before reading it returns
/// [`DatabaseError::PeerUnreachable`]: it did not run. Every other way the
/// answer can fail to arrive whole returns
/// [`DatabaseError::PeerOutcomeUnknown`]: it may have run.
#[derive(Debug, Clone)]
pub struct CommandRouter {
    member: u64,
    link: Arc<Link>,
}

/// The way to one member: the connection, and the session open on it.
struct Link {
    client: CommandRoutingServiceClient<Channel>,
    // no-std: spin::Mutex; taken once per command to find the session.
    session: parking_lot::Mutex<Option<Arc<Session>>>,
}

impl std::fmt::Debug for Link {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Link").finish_non_exhaustive()
    }
}

impl CommandRouter {
    /// A router over `channel`, a connection to `member`, the executing
    /// member. Its commands share one session on the connection, opened on
    /// the first and again after the one before was lost.
    pub fn new(member: u64, channel: Channel) -> Self {
        Self {
            member,
            link: Arc::new(Link {
                client: CommandRoutingServiceClient::new(channel)
                    .max_decoding_message_size(MAX_ROUTED_MESSAGE),
                session: parking_lot::Mutex::new(None),
            }),
        }
    }

    /// A router to `member`, over `node`'s connection to it: authenticated
    /// both ways like the rest of the traffic between members.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::PeerUnreachable`] when `node` has no peers, does not
    /// know where `member` is or what identity it proves, or cannot make a
    /// connection.
    pub fn to_member(node: &RaftNode, member: u64) -> Result<Self, DatabaseError> {
        let (_target, channel) = Self::member_channel(node, member)?;
        Ok(Self::new(member, channel))
    }

    /// Where `member` is, and `node`'s connection to it.
    ///
    /// # Errors
    ///
    /// As [`CommandRouter::to_member`].
    pub(crate) fn member_channel(
        node: &RaftNode,
        member: u64,
    ) -> Result<(coordinode_wire::peer::PeerTarget, Channel), DatabaseError> {
        let unreachable = |message: String| DatabaseError::PeerUnreachable { member, message };
        let (Some(target), Some(connections)) = (node.member_target(member), node.connections())
        else {
            return Err(unreachable(
                "this node does not know how to reach it".to_owned(),
            ));
        };
        let channel = connections
            .channel(&target)
            .map_err(|e| unreachable(e.to_string()))?;
        Ok((target, channel))
    }

    /// Send `request` on the session to the member and return the call its
    /// replies arrive on. A statement's call is cancelled at the member if
    /// it is dropped before its answer ended.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::PeerUnreachable`] when the request could not be
    /// sent: it did not run.
    fn start(
        &self,
        request: wire::routed_request::Request,
        deadline: Option<Duration>,
    ) -> Result<Call, DatabaseError> {
        let statement = matches!(request, wire::routed_request::Request::Execute(_));
        let session = {
            let mut slot = self.link.session.lock();
            match slot.as_ref() {
                Some(session) if session.alive() => Arc::clone(session),
                _ => {
                    let session = Session::open(self.link.client.clone(), self.member);
                    *slot = Some(Arc::clone(&session));
                    session
                }
            }
        };
        session.start(request, statement, deadline, self.member)
    }

    /// An answer this member cannot read: whatever the command did is
    /// unknown.
    fn unreadable(&self, message: String) -> DatabaseError {
        DatabaseError::PeerOutcomeUnknown {
            member: self.member,
            failure: PeerFailure::Protocol,
            message,
        }
    }

    /// The error `bytes` carry, read as the member's failure; empty bytes are
    /// no failure.
    fn failure(&self, bytes: &[u8]) -> Result<(), DatabaseError> {
        if bytes.is_empty() {
            return Ok(());
        }
        Err(errors::decode(bytes).unwrap_or_else(|e| self.unreadable(e)))
    }

    /// Run `statement` at the member and return its whole answer. `admitted`
    /// is told when the statement has its place there.
    ///
    /// # Errors
    ///
    /// What the statement failed with at the member, or how its answer was
    /// lost.
    pub async fn execute(
        &self,
        statement: RoutedStatement,
        admitted: &AdmissionSignal,
    ) -> Result<RoutedAnswer, DatabaseError> {
        self.execute_rows(statement, admitted)
            .await?
            .collect()
            .await
    }

    /// Run `statement` at the member and return its answer as rows read off
    /// the wire as they are asked for: the columns are known once this
    /// returns, and the member waits on its bounded queue for rows the caller
    /// has not taken, so a large answer is never held whole here.
    ///
    /// # Errors
    ///
    /// What the statement failed with at the member before its answer began,
    /// or how the call was lost.
    pub async fn execute_rows(
        &self,
        statement: RoutedStatement,
        admitted: &AdmissionSignal,
    ) -> Result<RoutedRows, DatabaseError> {
        let deadline = statement.deadline;
        let mut call = self.start(
            wire::routed_request::Request::Execute(Box::new(statement_to_wire(statement)?)),
            deadline,
        )?;
        // The answer opens with its columns, after the admission of a
        // transaction's statement and unless the statement failed. What
        // came in the same frame after them is the start of the rows.
        loop {
            let frame = match call.next().await? {
                wire::routed_reply::Reply::Frame(frame) => frame,
                _ => {
                    return Err(
                        self.unreadable("a statement answered as another command".to_owned())
                    );
                }
            };
            call.took(&frame);
            let mut events: std::collections::VecDeque<_> = frame.events.into();
            while let Some(event) = events.pop_front() {
                match event.event {
                    Some(Event::Admitted(_)) => admitted.admitted(),
                    Some(Event::Columns(columns)) => {
                        return Ok(RoutedRows {
                            router: self.clone(),
                            columns: columns.names,
                            call,
                            pending: events,
                            buffered: std::collections::VecDeque::new(),
                            stats: None,
                        });
                    }
                    Some(Event::Error(bytes)) => {
                        return Err(errors::decode(&bytes).unwrap_or_else(|e| self.unreadable(e)));
                    }
                    Some(Event::Rows(_) | Event::End(_)) => {
                        return Err(self.unreadable(
                            "the executing member sent rows before their columns".to_owned(),
                        ));
                    }
                    None => {}
                }
            }
        }
    }

    /// Send a short command and return its one reply.
    async fn short(
        &self,
        request: wire::routed_request::Request,
        deadline: Option<Duration>,
    ) -> Result<wire::routed_reply::Reply, DatabaseError> {
        let mut call = self.start(request, deadline)?;
        let reply = call.next().await?;
        call.ended();
        Ok(reply)
    }

    /// Open a transaction at the member, which must lead, under the ID the
    /// client drew.
    ///
    /// # Errors
    ///
    /// Why the member did not open it, or how its answer was lost.
    pub async fn begin(&self, begin: RoutedBegin) -> Result<(), DatabaseError> {
        match self
            .short(
                wire::routed_request::Request::Begin(begin_to_wire(begin)),
                None,
            )
            .await?
        {
            wire::routed_reply::Reply::Begun(begun) => self.failure(&begun.error),
            _ => Err(self.unreadable("a begin answered as another command".to_owned())),
        }
    }

    /// Commit a transaction at the leader.
    ///
    /// # Errors
    ///
    /// What the commit failed with, or how its answer was lost.
    pub async fn commit(&self, commit: RoutedCommit) -> Result<CommitReceipt, DatabaseError> {
        let deadline = commit.deadline;
        let committed = match self
            .short(
                wire::routed_request::Request::Commit(commit_to_wire(commit)),
                deadline,
            )
            .await?
        {
            wire::routed_reply::Reply::Committed(committed) => committed,
            _ => return Err(self.unreadable("a commit answered as another command".to_owned())),
        };
        self.failure(&committed.error)?;
        Ok(CommitReceipt {
            commit_ts: Timestamp::from_raw(committed.commit_ts),
            applied_index: (committed.applied_index > 0).then_some(committed.applied_index),
        })
    }

    /// Roll back a transaction at the leader.
    ///
    /// # Errors
    ///
    /// Why the rollback was refused, or how its answer was lost.
    pub async fn rollback(&self, rollback: RoutedRollback) -> Result<(), DatabaseError> {
        match self
            .short(
                wire::routed_request::Request::Rollback(rollback_to_wire(rollback)),
                None,
            )
            .await?
        {
            wire::routed_reply::Reply::RolledBack(rolled_back) => self.failure(&rolled_back.error),
            _ => Err(self.unreadable("a rollback answered as another command".to_owned())),
        }
    }

    /// Carry out a call on operation records at the member, which must lead.
    ///
    /// # Errors
    ///
    /// Why the member refused it, or how its answer was lost.
    pub async fn operation(
        &self,
        actor: &ActingContext,
        call: RoutedOperationCall,
    ) -> Result<RoutedOperationAnswer, DatabaseError> {
        let answer = match self
            .short(
                wire::routed_request::Request::Operation(wire::RoutedOperation {
                    actor: Some(actor_to_wire(actor)),
                    call: Some(operation_call_to_wire(call)),
                }),
                None,
            )
            .await?
        {
            wire::routed_reply::Reply::Operation(answer) => answer,
            _ => {
                return Err(
                    self.unreadable("an operation call answered as another command".to_owned())
                );
            }
        };
        if let Some(wire::routed_operation_answer::Answer::Error(bytes)) = &answer.answer {
            return Err(errors::decode(bytes).unwrap_or_else(|e| self.unreadable(e)));
        }
        operation_answer_from_wire(answer).map_err(|e| self.unreadable(e))
    }
}

/// What arrives for one call: a reply, or how the session was lost.
enum Arrival {
    Reply(wire::routed_reply::Reply),
    Lost(DatabaseError),
}

/// How a session ended for the calls still waiting on it.
enum Loss {
    /// It never opened: the member read none of its commands.
    NotOpened(String),
    /// It broke after it opened: a command in flight may have run.
    Broke(PeerFailure, String),
}

impl Loss {
    fn error(&self, member: u64) -> DatabaseError {
        match self {
            Self::NotOpened(message) => DatabaseError::PeerUnreachable {
                member,
                message: message.clone(),
            },
            Self::Broke(failure, message) => DatabaseError::PeerOutcomeUnknown {
                member,
                failure: *failure,
                message: message.clone(),
            },
        }
    }
}

/// The calls of a session waiting for replies, by number.
struct Waiting {
    /// The session can take no more calls: its stream ended or never opened.
    closed: bool,
    calls: rustc_hash::FxHashMap<u64, tokio::sync::mpsc::UnboundedSender<Arrival>>,
}

/// One session from this member, the ingress, to an executing member: the
/// stream every command routed there travels on.
struct Session {
    /// Commands, credits and cancellations for the stream. Unbounded on its
    /// own, and bounded by the callers: each waits for its command's reply,
    /// and a statement sends a credit only for a frame it took.
    requests: tokio::sync::mpsc::UnboundedSender<wire::RoutedRequest>,
    // no-std: spin::Mutex; taken once per command and once per reply.
    waiting: Arc<parking_lot::Mutex<Waiting>>,
    next: std::sync::atomic::AtomicU64,
}

impl Session {
    /// Open a session on `client` to `member`. The stream opens in the
    /// background; commands sent before it did wait in the queue.
    fn open(client: CommandRoutingServiceClient<Channel>, member: u64) -> Arc<Self> {
        let (requests, outbound) = tokio::sync::mpsc::unbounded_channel();
        let waiting = Arc::new(parking_lot::Mutex::new(Waiting {
            closed: false,
            calls: rustc_hash::FxHashMap::default(),
        }));
        tokio::spawn(Self::read(client, member, outbound, Arc::clone(&waiting)));
        Arc::new(Self {
            requests,
            waiting,
            next: std::sync::atomic::AtomicU64::new(1),
        })
    }

    fn alive(&self) -> bool {
        !self.waiting.lock().closed
    }

    /// Open the stream and hand each reply to the call it answers, until
    /// the stream ends; then every call still waiting learns how.
    async fn read(
        mut client: CommandRoutingServiceClient<Channel>,
        member: u64,
        outbound: tokio::sync::mpsc::UnboundedReceiver<wire::RoutedRequest>,
        waiting: Arc<parking_lot::Mutex<Waiting>>,
    ) {
        let mut request = Request::new(tokio_stream::wrappers::UnboundedReceiverStream::new(
            outbound,
        ));
        request
            .metadata_mut()
            .insert(CODEC_HEADER, MetadataValue::from(ERROR_CODEC_VERSION));
        let lost = match client.session(request).await {
            // The member answers the session's opening before it reads any
            // command, so a session that did not open ran none of them.
            Err(status) => Loss::NotOpened(status.message().to_owned()),
            Ok(response) => {
                let mut replies = response.into_inner();
                loop {
                    match replies.message().await {
                        Ok(Some(reply)) => {
                            let Some(answer) = reply.reply else {
                                continue;
                            };
                            if let Some(call) = waiting.lock().calls.get(&reply.call) {
                                // A call that stopped waiting dropped its
                                // receiver; its late reply goes nowhere.
                                if call.send(Arrival::Reply(answer)).is_err() {
                                    tracing::trace!(call = reply.call, "late routed reply");
                                }
                            }
                        }
                        Ok(None) => {
                            break Loss::Broke(
                                PeerFailure::Transport,
                                "the executing member ended the session".to_owned(),
                            );
                        }
                        Err(status) => {
                            break Loss::Broke(
                                match status.code() {
                                    Code::DeadlineExceeded | Code::Cancelled => {
                                        PeerFailure::Deadline
                                    }
                                    _ => PeerFailure::Transport,
                                },
                                status.message().to_owned(),
                            );
                        }
                    }
                }
            }
        };
        let calls = {
            let mut waiting = waiting.lock();
            waiting.closed = true;
            std::mem::take(&mut waiting.calls)
        };
        for (_, call) in calls {
            // A call that already stopped waiting needs no word.
            if call.send(Arrival::Lost(lost.error(member))).is_err() {
                tracing::trace!("a routed call left before its session was lost");
            }
        }
    }

    /// Number `request`, register its call and send it.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::PeerUnreachable`] when the session is over before
    /// the request was sent: it did not run.
    fn start(
        self: &Arc<Self>,
        request: wire::routed_request::Request,
        statement: bool,
        deadline: Option<Duration>,
        member: u64,
    ) -> Result<Call, DatabaseError> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (arrivals, replies) = tokio::sync::mpsc::unbounded_channel();
        {
            let mut waiting = self.waiting.lock();
            if waiting.closed {
                return Err(DatabaseError::PeerUnreachable {
                    member,
                    message: "the session to the member closed".to_owned(),
                });
            }
            waiting.calls.insert(id, arrivals);
        }
        let call = Call {
            session: Arc::clone(self),
            id,
            member,
            replies,
            deadline: deadline.map(|d| tokio::time::Instant::now() + d),
            open: statement,
        };
        self.requests
            .send(wire::RoutedRequest {
                call: id,
                request: Some(request),
            })
            .map_err(|_| DatabaseError::PeerUnreachable {
                member,
                message: "the session to the member closed".to_owned(),
            })?;
        Ok(call)
    }

    fn send(&self, call: u64, request: wire::routed_request::Request) {
        // A closed queue: the session is over, and the member stopped the
        // call with it.
        if self
            .requests
            .send(wire::RoutedRequest {
                call,
                request: Some(request),
            })
            .is_err()
        {
            tracing::trace!(call, "routing session closed");
        }
    }
}

/// One command's replies on a session.
struct Call {
    session: Arc<Session>,
    id: u64,
    member: u64,
    replies: tokio::sync::mpsc::UnboundedReceiver<Arrival>,
    deadline: Option<tokio::time::Instant>,
    /// A statement whose answer has not ended: leaving it cancels it at the
    /// member.
    open: bool,
}

impl Call {
    /// The next reply.
    ///
    /// # Errors
    ///
    /// The member refused the command ([`DatabaseError::PeerUnreachable`]:
    /// it did not run), or its answer broke, ran past the deadline, or was
    /// lost with the session ([`DatabaseError::PeerOutcomeUnknown`]).
    async fn next(&mut self) -> Result<wire::routed_reply::Reply, DatabaseError> {
        let arrival = match self.deadline {
            Some(deadline) => tokio::time::timeout_at(deadline, self.replies.recv())
                .await
                .map_err(|_| DatabaseError::PeerOutcomeUnknown {
                    member: self.member,
                    failure: PeerFailure::Deadline,
                    message: "the deadline passed before the member answered".to_owned(),
                })?,
            None => self.replies.recv().await,
        };
        match arrival {
            Some(Arrival::Reply(wire::routed_reply::Reply::Refused(refusal))) => {
                self.open = false;
                Err(DatabaseError::PeerUnreachable {
                    member: self.member,
                    message: format!("{}: {}", refusal.reason, refusal.message),
                })
            }
            Some(Arrival::Reply(wire::routed_reply::Reply::Broken(message))) => {
                self.open = false;
                Err(DatabaseError::PeerOutcomeUnknown {
                    member: self.member,
                    failure: PeerFailure::Transport,
                    message,
                })
            }
            Some(Arrival::Reply(reply)) => Ok(reply),
            Some(Arrival::Lost(lost)) => {
                self.open = false;
                Err(lost)
            }
            // The session dropped the call's sender without a word only when
            // it is over.
            None => Err(DatabaseError::PeerOutcomeUnknown {
                member: self.member,
                failure: PeerFailure::Transport,
                message: "the session to the member was lost".to_owned(),
            }),
        }
    }

    /// The caller took `frame` of the statement's answer: unless it ended
    /// the answer, the member may send one more. An admission travels
    /// outside the credit and returns none.
    fn took(&mut self, frame: &wire::RoutedStatementFrame) {
        let mut counted = false;
        for event in &frame.events {
            match event.event {
                Some(Event::End(_) | Event::Error(_)) => {
                    self.open = false;
                    return;
                }
                Some(Event::Admitted(_)) | None => {}
                Some(Event::Columns(_) | Event::Rows(_)) => counted = true,
            }
        }
        if counted {
            self.session
                .send(self.id, wire::routed_request::Request::Credit(1));
        }
    }

    /// The answer ended: nothing more arrives for the call.
    fn ended(&mut self) {
        self.open = false;
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        self.session.waiting.lock().calls.remove(&self.id);
        if self.open {
            self.session.send(
                self.id,
                wire::routed_request::Request::Cancel(wire::RoutedCancel {}),
            );
        }
    }
}

/// The rows of a statement another member runs, read off the wire as the
/// caller asks for them. At most one received batch is held here; the member
/// sends ahead only as many frames as the caller took, which holds it back.
pub struct RoutedRows {
    router: CommandRouter,
    columns: Vec<String>,
    call: Call,
    /// Events of a received frame not read yet.
    pending: std::collections::VecDeque<wire::RoutedStatementEvent>,
    buffered: std::collections::VecDeque<Vec<Value>>,
    stats: Option<RoutedStats>,
}

impl RoutedRows {
    /// The column names, in result order.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// The statement's statistics, known once every row was read; `None`
    /// before.
    pub fn stats(&self) -> Option<RoutedStats> {
        self.stats
    }

    /// Up to `max` more rows; none once they run out, when the statistics
    /// are known.
    ///
    /// # Errors
    ///
    /// How the rest of the answer was lost: a failure after rows were
    /// delivered leaves the statement's outcome to be resolved, never rerun.
    pub async fn next_rows(&mut self, max: usize) -> Result<Vec<Vec<Value>>, DatabaseError> {
        while self.buffered.is_empty() && self.stats.is_none() {
            let Some(event) = self.pending.pop_front() else {
                let frame = match self.call.next().await? {
                    wire::routed_reply::Reply::Frame(frame) => frame,
                    _ => {
                        return Err(self
                            .router
                            .unreadable("a statement answered as another command".to_owned()));
                    }
                };
                self.call.took(&frame);
                self.pending.extend(frame.events);
                continue;
            };
            match event.event {
                Some(Event::Rows(batch)) => {
                    let mut rows = Vec::new();
                    decode_rows(&batch, &mut rows).map_err(|e| self.router.unreadable(e))?;
                    self.buffered.extend(rows);
                }
                Some(Event::End(stats)) => self.stats = Some(stats_from_wire(stats)),
                Some(Event::Error(bytes)) => {
                    return Err(
                        errors::decode(&bytes).unwrap_or_else(|e| self.router.unreadable(e))
                    );
                }
                Some(Event::Admitted(_) | Event::Columns(_)) | None => {}
            }
        }
        let take = max.min(self.buffered.len());
        Ok(self.buffered.drain(..take).collect())
    }

    /// Every remaining row and the statistics, as one answer.
    ///
    /// # Errors
    ///
    /// As [`RoutedRows::next_rows`].
    pub async fn collect(mut self) -> Result<RoutedAnswer, DatabaseError> {
        let mut rows = Vec::new();
        loop {
            let piece = self.next_rows(usize::MAX).await?;
            if piece.is_empty() {
                break;
            }
            rows.extend(piece);
        }
        Ok(RoutedAnswer {
            columns: self.columns,
            rows,
            // Known: the rows ran out only once the statistics arrived.
            stats: self.stats.unwrap_or_default(),
        })
    }
}

/// Append the rows of one batch, a run of encoded rows, to `rows`.
fn decode_rows(batch: &[u8], rows: &mut Vec<Vec<Value>>) -> Result<(), String> {
    let mut cursor = Cursor::new(batch);
    while (cursor.position() as usize) < batch.len() {
        let row: Vec<Value> = rmp_serde::from_read(&mut cursor)
            .map_err(|e| format!("a routed row could not be decoded: {e}"))?;
        rows.push(row);
    }
    Ok(())
}

fn actor_to_wire(actor: &ActingContext) -> wire::ActingContext {
    wire::ActingContext {
        context: Some(match actor {
            ActingContext::Trusted => wire::acting_context::Context::Trusted(wire::Trusted {}),
            ActingContext::Principal(p) => {
                wire::acting_context::Context::Principal(wire::Principal {
                    subject: p.subject.to_string(),
                    trust_scope: p.trust_scope.to_string(),
                    namespace: p.namespace.to_string(),
                })
            }
        }),
    }
}

fn actor_from_wire(actor: Option<wire::ActingContext>) -> Result<ActingContext, Status> {
    match actor.and_then(|a| a.context) {
        Some(wire::acting_context::Context::Trusted(_)) => Ok(ActingContext::Trusted),
        Some(wire::acting_context::Context::Principal(p)) => {
            Ok(ActingContext::Principal(Principal {
                subject: p.subject.into(),
                trust_scope: p.trust_scope.into(),
                namespace: p.namespace.into(),
            }))
        }
        // A routed command always names who it acts for: a missing one is a
        // defect of the ingress, never a reason to act for nobody in
        // particular.
        None => Err(Status::invalid_argument(
            "a routed command must name who it acts for",
        )),
    }
}

fn transaction_id_from_wire(bytes: &[u8]) -> Result<TransactionId, Status> {
    TransactionId::from_bytes(bytes)
        .map_err(|e| Status::invalid_argument(format!("transaction_id: {e}")))
}

fn operation_from_wire(bytes: &[u8]) -> Result<Option<OperationDescriptor>, Status> {
    if bytes.is_empty() {
        return Ok(None);
    }
    OperationDescriptor::from_bytes(bytes)
        .map(Some)
        .ok_or_else(|| Status::invalid_argument("operation: malformed descriptor"))
}

fn deadline_from_wire(ms: u64) -> Option<Duration> {
    (ms > 0).then(|| Duration::from_millis(ms))
}

fn deadline_to_wire(deadline: Option<Duration>) -> u64 {
    // A deadline under a millisecond still has to stop the call: it is sent
    // as one, not as zero, which would mean none.
    deadline.map_or(0, |d| {
        u64::try_from(d.as_millis()).unwrap_or(u64::MAX).max(1)
    })
}

fn read_level_to_wire(level: Option<ReadConcernLevel>) -> u32 {
    match level {
        None => 0,
        Some(ReadConcernLevel::Local) => 1,
        Some(ReadConcernLevel::Majority) => 2,
        Some(ReadConcernLevel::Linearizable) => 3,
        Some(ReadConcernLevel::Snapshot) => 4,
    }
}

fn read_level_from_wire(level: u32) -> Result<Option<ReadConcernLevel>, Status> {
    Ok(match level {
        0 => None,
        1 => Some(ReadConcernLevel::Local),
        2 => Some(ReadConcernLevel::Majority),
        3 => Some(ReadConcernLevel::Linearizable),
        4 => Some(ReadConcernLevel::Snapshot),
        other => {
            return Err(Status::invalid_argument(format!(
                "read concern level {other} is not one this member knows"
            )));
        }
    })
}

fn write_concern_to_wire(concern: WriteConcern) -> wire::RoutedWriteConcern {
    let (acks, majority) = match concern.w {
        WriteAck::Acks(n) => (n, false),
        WriteAck::Majority => (0, true),
    };
    wire::RoutedWriteConcern {
        acks,
        majority,
        journal: match concern.journal {
            Journal::Journal => 1,
            Journal::Cache => 2,
            Journal::Memory => 3,
        },
        timeout_ms: concern.timeout_ms,
    }
}

fn write_concern_from_wire(concern: wire::RoutedWriteConcern) -> Result<WriteConcern, Status> {
    Ok(WriteConcern {
        w: if concern.majority {
            WriteAck::Majority
        } else {
            WriteAck::Acks(concern.acks)
        },
        journal: match concern.journal {
            1 => Journal::Journal,
            2 => Journal::Cache,
            3 => Journal::Memory,
            other => {
                return Err(Status::invalid_argument(format!(
                    "journal level {other} is not one this member knows"
                )));
            }
        },
        timeout_ms: concern.timeout_ms,
    })
}

fn vector_consistency_to_wire(mode: Option<VectorConsistencyMode>) -> u32 {
    match mode {
        None => 0,
        Some(VectorConsistencyMode::Current) => 1,
        Some(VectorConsistencyMode::Snapshot) => 2,
        Some(VectorConsistencyMode::Exact) => 3,
    }
}

fn vector_consistency_from_wire(mode: u32) -> Result<Option<VectorConsistencyMode>, Status> {
    Ok(match mode {
        0 => None,
        1 => Some(VectorConsistencyMode::Current),
        2 => Some(VectorConsistencyMode::Snapshot),
        3 => Some(VectorConsistencyMode::Exact),
        other => {
            return Err(Status::invalid_argument(format!(
                "vector consistency {other} is not one this member knows"
            )));
        }
    })
}

fn settings_to_wire(settings: RoutedSettings) -> wire::RoutedSettings {
    wire::RoutedSettings {
        read_concern_level: read_level_to_wire(settings.read_concern),
        after_index: settings.after_index.unwrap_or(0),
        at_timestamp: settings.at_timestamp.unwrap_or(0),
        write_concern: settings.write_concern.map(write_concern_to_wire),
        vector_consistency: vector_consistency_to_wire(settings.vector_consistency),
        vector_build_wait_ms: settings
            .vector_build_wait
            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
        query_memory_limit: settings.query_memory_limit,
    }
}

fn settings_from_wire(settings: Option<wire::RoutedSettings>) -> Result<RoutedSettings, Status> {
    let Some(settings) = settings else {
        return Ok(RoutedSettings::default());
    };
    Ok(RoutedSettings {
        read_concern: read_level_from_wire(settings.read_concern_level)?,
        after_index: (settings.after_index > 0).then_some(settings.after_index),
        at_timestamp: (settings.at_timestamp > 0).then_some(settings.at_timestamp),
        write_concern: settings
            .write_concern
            .map(write_concern_from_wire)
            .transpose()?,
        vector_consistency: vector_consistency_from_wire(settings.vector_consistency)?,
        vector_build_wait: settings.vector_build_wait_ms.map(Duration::from_millis),
        query_memory_limit: settings.query_memory_limit,
    })
}

fn statement_to_wire(statement: RoutedStatement) -> Result<wire::RoutedStatement, DatabaseError> {
    Ok(wire::RoutedStatement {
        actor: Some(actor_to_wire(&statement.actor)),
        query: statement.query,
        parameters: if statement.params.is_empty() {
            Vec::new()
        } else {
            rmp_serde::to_vec(&statement.params).map_err(|e| {
                DatabaseError::Semantic(format!("parameters could not be encoded: {e}"))
            })?
        },
        settings: Some(settings_to_wire(statement.settings)),
        transaction_id: statement
            .transaction
            .map(|id| id.to_bytes().to_vec())
            .unwrap_or_default(),
        operation: statement
            .operation
            .map(|o| o.to_bytes())
            .unwrap_or_default(),
        deadline_ms: deadline_to_wire(statement.deadline),
        source: statement.source.map(|s| wire::SourceLocation {
            file: s.file,
            line: s.line,
            function: s.function,
            app: s.app,
            version: s.version,
        }),
        start: statement.start.map(start_to_wire),
    })
}

fn start_to_wire(start: TransactionStart) -> wire::RoutedStart {
    wire::RoutedStart {
        settings: Some(settings_to_wire(RoutedSettings {
            read_concern: start.read_concern,
            write_concern: start.write_concern,
            ..RoutedSettings::default()
        })),
        timeout_outcome: timeout_outcome_to_wire(start.timeout_outcome),
        operation: start.operation.map(|o| o.to_bytes()).unwrap_or_default(),
    }
}

fn start_from_wire(start: wire::RoutedStart) -> Result<TransactionStart, Status> {
    let settings = settings_from_wire(start.settings)?;
    Ok(TransactionStart {
        read_concern: settings.read_concern,
        write_concern: settings.write_concern,
        timeout_outcome: timeout_outcome_from_wire(start.timeout_outcome)?,
        operation: operation_from_wire(&start.operation)?,
    })
}

fn timeout_outcome_to_wire(outcome: Option<TimeoutOutcome>) -> u32 {
    match outcome {
        None => 0,
        Some(TimeoutOutcome::Rollback) => 1,
        Some(TimeoutOutcome::Commit) => 2,
    }
}

fn timeout_outcome_from_wire(outcome: u32) -> Result<Option<TimeoutOutcome>, Status> {
    match outcome {
        0 => Ok(None),
        1 => Ok(Some(TimeoutOutcome::Rollback)),
        2 => Ok(Some(TimeoutOutcome::Commit)),
        other => Err(Status::invalid_argument(format!(
            "timeout outcome {other} is not one this member knows"
        ))),
    }
}

fn statement_from_wire(statement: wire::RoutedStatement) -> Result<RoutedStatement, Status> {
    Ok(RoutedStatement {
        actor: actor_from_wire(statement.actor)?,
        params: if statement.parameters.is_empty() {
            HashMap::new()
        } else {
            rmp_serde::from_slice(&statement.parameters)
                .map_err(|e| Status::invalid_argument(format!("parameters: {e}")))?
        },
        query: statement.query,
        settings: settings_from_wire(statement.settings)?,
        transaction: if statement.transaction_id.is_empty() {
            None
        } else {
            Some(transaction_id_from_wire(&statement.transaction_id)?)
        },
        operation: operation_from_wire(&statement.operation)?,
        deadline: deadline_from_wire(statement.deadline_ms),
        source: statement.source.map(|s| SourceContext {
            file: s.file,
            line: s.line,
            function: s.function,
            app: s.app,
            version: s.version,
        }),
        start: statement.start.map(start_from_wire).transpose()?,
    })
}

fn stats_to_wire(stats: RoutedStats) -> wire::RoutedStats {
    wire::RoutedStats {
        nodes_created: stats.nodes_created,
        nodes_deleted: stats.nodes_deleted,
        edges_created: stats.edges_created,
        edges_deleted: stats.edges_deleted,
        properties_set: stats.properties_set,
        applied_index: stats.applied_index,
        commit_ts: stats.commit_ts,
        read_index: stats.read_index,
        read_as_of_ts: stats.read_as_of_ts,
    }
}

fn stats_from_wire(stats: wire::RoutedStats) -> RoutedStats {
    RoutedStats {
        nodes_created: stats.nodes_created,
        nodes_deleted: stats.nodes_deleted,
        edges_created: stats.edges_created,
        edges_deleted: stats.edges_deleted,
        properties_set: stats.properties_set,
        applied_index: stats.applied_index,
        commit_ts: stats.commit_ts,
        read_index: stats.read_index,
        read_as_of_ts: stats.read_as_of_ts,
    }
}

fn begin_to_wire(begin: RoutedBegin) -> wire::RoutedBegin {
    wire::RoutedBegin {
        transaction_id: begin.transaction.to_bytes().to_vec(),
        actor: Some(actor_to_wire(&begin.actor)),
        settings: Some(settings_to_wire(RoutedSettings {
            read_concern: begin.read_concern,
            write_concern: begin.write_concern,
            ..RoutedSettings::default()
        })),
        timeout_outcome: timeout_outcome_to_wire(begin.timeout_outcome),
        operation: begin.operation.map(|o| o.to_bytes()).unwrap_or_default(),
    }
}

fn begin_from_wire(begin: wire::RoutedBegin) -> Result<RoutedBegin, Status> {
    let settings = settings_from_wire(begin.settings)?;
    Ok(RoutedBegin {
        transaction: transaction_id_from_wire(&begin.transaction_id)?,
        actor: actor_from_wire(begin.actor)?,
        read_concern: settings.read_concern,
        write_concern: settings.write_concern,
        timeout_outcome: timeout_outcome_from_wire(begin.timeout_outcome)?,
        operation: operation_from_wire(&begin.operation)?,
    })
}

fn commit_to_wire(commit: RoutedCommit) -> wire::RoutedCommit {
    wire::RoutedCommit {
        actor: Some(actor_to_wire(&commit.actor)),
        transaction_id: commit.transaction.to_bytes().to_vec(),
        operation: commit.operation.map(|o| o.to_bytes()).unwrap_or_default(),
        deadline_ms: deadline_to_wire(commit.deadline),
        expect: commit
            .expect
            .iter()
            .map(|e| wire::RoutedExpectation {
                node_id: e.node.as_raw(),
                version: e.version,
            })
            .collect(),
    }
}

fn commit_from_wire(commit: wire::RoutedCommit) -> Result<RoutedCommit, Status> {
    Ok(RoutedCommit {
        actor: actor_from_wire(commit.actor)?,
        transaction: transaction_id_from_wire(&commit.transaction_id)?,
        operation: operation_from_wire(&commit.operation)?,
        deadline: deadline_from_wire(commit.deadline_ms),
        expect: commit
            .expect
            .into_iter()
            .map(|e| ExpectedVersion {
                node: NodeId::from_raw(e.node_id),
                version: e.version,
            })
            .collect(),
    })
}

fn rollback_to_wire(rollback: RoutedRollback) -> wire::RoutedRollback {
    wire::RoutedRollback {
        actor: Some(actor_to_wire(&rollback.actor)),
        transaction_id: rollback.transaction.to_bytes().to_vec(),
        abandoned: rollback.abandoned,
    }
}

fn rollback_from_wire(rollback: wire::RoutedRollback) -> Result<RoutedRollback, Status> {
    Ok(RoutedRollback {
        actor: actor_from_wire(rollback.actor)?,
        transaction: transaction_id_from_wire(&rollback.transaction_id)?,
        abandoned: rollback.abandoned,
    })
}

fn descriptor_from_wire(field: &str, bytes: &[u8]) -> Result<OperationDescriptor, Status> {
    OperationDescriptor::from_bytes(bytes)
        .ok_or_else(|| Status::invalid_argument(format!("{field}: malformed descriptor")))
}

fn operation_call_to_wire(call: RoutedOperationCall) -> wire::routed_operation::Call {
    use wire::routed_operation::Call;
    match call {
        RoutedOperationCall::Enroll(issuer) => Call::Enroll(issuer.0.to_vec()),
        RoutedOperationCall::Resolve(descriptor, resolve) => Call::Resolve(wire::RoutedResolve {
            operation: descriptor.to_bytes(),
            exclude: resolve == Resolve::Exclude,
        }),
        RoutedOperationCall::Acknowledge(descriptors) => {
            Call::Acknowledge(wire::RoutedOperations {
                operations: descriptors
                    .iter()
                    .map(OperationDescriptor::to_bytes)
                    .collect(),
            })
        }
        RoutedOperationCall::Retire(below) => Call::Retire(below.to_bytes()),
    }
}

fn operation_call_from_wire(
    call: Option<wire::routed_operation::Call>,
) -> Result<RoutedOperationCall, Status> {
    use wire::routed_operation::Call;
    Ok(match call {
        Some(Call::Enroll(issuer)) => RoutedOperationCall::Enroll(IssuerId(
            <[u8; 16]>::try_from(issuer.as_slice())
                .map_err(|_| Status::invalid_argument("enroll: an issuer is 16 bytes"))?,
        )),
        Some(Call::Resolve(resolve)) => RoutedOperationCall::Resolve(
            descriptor_from_wire("resolve.operation", &resolve.operation)?,
            if resolve.exclude {
                Resolve::Exclude
            } else {
                Resolve::Lookup
            },
        ),
        Some(Call::Acknowledge(acknowledge)) => RoutedOperationCall::Acknowledge(
            acknowledge
                .operations
                .iter()
                .map(|bytes| descriptor_from_wire("acknowledge.operations", bytes))
                .collect::<Result<_, _>>()?,
        ),
        Some(Call::Retire(below)) => {
            RoutedOperationCall::Retire(descriptor_from_wire("retire", &below)?)
        }
        None => {
            return Err(Status::invalid_argument(
                "an operation call names what to do",
            ));
        }
    })
}

fn operation_answer_to_wire(answer: RoutedOperationAnswer) -> wire::RoutedOperationAnswer {
    use wire::routed_operation_answer::Answer;
    let answer = match answer {
        RoutedOperationAnswer::Enrolled {
            incarnation,
            next_sequence,
            deployment,
        } => Answer::Enrolled(wire::RoutedEnrolled {
            incarnation,
            next_sequence,
            deployment: deployment.as_bytes().to_vec(),
        }),
        RoutedOperationAnswer::Outcome(outcome) => {
            Answer::Outcome(errors::outcome_to_wire(outcome))
        }
        RoutedOperationAnswer::Done => Answer::Done(wire::RoutedDone {}),
    };
    wire::RoutedOperationAnswer {
        answer: Some(answer),
    }
}

/// The answer to an operation call that succeeded.
///
/// # Errors
///
/// What makes it unreadable; an answer carrying an error is read before.
fn operation_answer_from_wire(
    answer: wire::RoutedOperationAnswer,
) -> Result<RoutedOperationAnswer, String> {
    use wire::routed_operation_answer::Answer;
    Ok(match answer.answer {
        Some(Answer::Enrolled(enrolled)) => RoutedOperationAnswer::Enrolled {
            incarnation: enrolled.incarnation,
            next_sequence: enrolled.next_sequence,
            deployment: DeploymentId::from_slice(&enrolled.deployment)
                .ok_or("the leader answered a malformed deployment")?,
        },
        Some(Answer::Outcome(outcome)) => {
            RoutedOperationAnswer::Outcome(errors::outcome_from_wire(&outcome)?)
        }
        Some(Answer::Done(_)) => RoutedOperationAnswer::Done,
        Some(Answer::Error(_)) | None => return Err("the leader answered nothing".to_owned()),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
