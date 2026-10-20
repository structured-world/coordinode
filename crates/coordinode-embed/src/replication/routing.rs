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
use tokio_stream::wrappers::ReceiverStream;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::{Code, Request, Response, Status};

use crate::db::{DatabaseError, PeerFailure, TransactionEnd};
use crate::operation::{IssuerId, OperationOutcome, Resolve};
use coordinode_core::group::DeploymentId;
use wire::command_routing_service_client::CommandRoutingServiceClient;
use wire::command_routing_service_server::{CommandRoutingService, CommandRoutingServiceServer};
use wire::routed_statement_frame::Event;

mod errors;
pub use errors::{ERROR_CODEC_VERSION, MAX_ROUTED_ERROR};

/// Request metadata naming the error encoding the ingress reads.
const CODEC_HEADER: &str = "cn-routed-error-codec";

/// Response metadata of a call refused before it was read: the command did
/// not run. Its value says why.
const REFUSED_HEADER: &str = "cn-routing-refused";

/// The encoded size a batch of rows grows to before it is sent.
const ROW_BATCH_BYTES: usize = 1 << 20;

/// Frames of one answer queued for the ingress before the executing member
/// waits for it to take them.
const FRAMES_IN_FLIGHT: usize = 4;

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

/// The service a member serves its peers the commands they route with.
pub struct CommandRoutingHandler<E> {
    executor: Arc<E>,
}

impl<E: CommandExecutor> CommandRoutingHandler<E> {
    /// A handler that runs routed commands through `executor`.
    pub fn new(executor: Arc<E>) -> Self {
        Self { executor }
    }

    /// The handler as a gRPC service taking messages up to
    /// `max_decoding_bytes`.
    pub fn into_service(self, max_decoding_bytes: usize) -> CommandRoutingServiceServer<Self> {
        CommandRoutingServiceServer::new(self).max_decoding_message_size(max_decoding_bytes)
    }

    /// Run `call` on the blocking pool and answer with what it returns. A
    /// call that did not finish is a lost answer: it may have run.
    async fn blocking<T: Send + 'static>(
        &self,
        call: impl FnOnce(&E) -> Result<T, DatabaseError> + Send + 'static,
    ) -> Result<Result<T, DatabaseError>, Status> {
        let executor = Arc::clone(&self.executor);
        tokio::task::spawn_blocking(move || call(&executor))
            .await
            .map_err(|e| Status::internal(format!("the routed command did not finish: {e}")))
    }
}

/// A call refused before it was read, for `why`: the command did not run,
/// which the ingress learns from the metadata rather than the text.
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

/// A request that does not decode: refused before anything ran.
fn malformed(status: Status) -> Status {
    refused(status, "request")
}

#[tonic::async_trait]
impl<E: CommandExecutor> CommandRoutingService for CommandRoutingHandler<E> {
    type ExecuteStream = ReceiverStream<Result<wire::RoutedStatementFrame, Status>>;

    async fn execute(
        &self,
        request: Request<wire::RoutedStatement>,
    ) -> Result<Response<Self::ExecuteStream>, Status> {
        check_codec(&request)?;
        let statement = statement_from_wire(request.into_inner()).map_err(malformed)?;
        let (frames, answer) = tokio::sync::mpsc::channel(FRAMES_IN_FLIGHT);

        // The ingress going away cancels the statement.
        let cancel = CancelFlag::new();
        {
            let frames = frames.clone();
            let cancel = cancel.clone();
            tokio::spawn(async move {
                frames.closed().await;
                cancel.cancel();
            });
        }

        // The admitted frame is the first one sent, so the queue has room
        // for it; told twice, it is sent once.
        let told = Arc::new(AtomicBool::new(false));
        let admitted = {
            let frames = frames.clone();
            AdmissionSignal::new(move || {
                if !told.swap(true, Ordering::AcqRel) {
                    // A full or closed queue: the ingress is gone, and the
                    // cancellation above stops the statement.
                    if frames
                        .try_send(Ok(frame(Event::Admitted(wire::Admitted {}))))
                        .is_err()
                    {
                        tracing::debug!("a routed statement's ingress left before its admission");
                    }
                }
            })
        };
        // An autonomous statement waits behind nothing.
        if statement.transaction.is_none() {
            admitted.admitted();
        }

        // A statement of a transaction whose answer does not reach the
        // ingress leaves its client unable to know what it did, and a
        // statement never runs twice: the transaction is aborted.
        let abandon = statement.transaction.map(|transaction| RoutedRollback {
            actor: statement.actor.clone(),
            transaction,
            abandoned: true,
        });
        let executor = Arc::clone(&self.executor);
        tokio::task::spawn_blocking(move || {
            let mut answer = AnswerWriter::new(frames);
            let result = executor.execute(statement, &cancel, &admitted, &mut answer);
            if !answer.finish(result) {
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
        Ok(Response::new(ReceiverStream::new(answer)))
    }

    async fn begin(
        &self,
        request: Request<wire::RoutedBegin>,
    ) -> Result<Response<wire::RoutedBegun>, Status> {
        check_codec(&request)?;
        let begin = begin_from_wire(request.into_inner()).map_err(malformed)?;
        let error = match self.blocking(move |e| e.begin(begin)).await? {
            Ok(()) => Vec::new(),
            Err(err) => errors::encode(err),
        };
        Ok(Response::new(wire::RoutedBegun { error }))
    }

    async fn commit(
        &self,
        request: Request<wire::RoutedCommit>,
    ) -> Result<Response<wire::RoutedCommitted>, Status> {
        check_codec(&request)?;
        let commit = commit_from_wire(request.into_inner()).map_err(malformed)?;
        Ok(Response::new(
            match self.blocking(move |e| e.commit(commit)).await? {
                Ok(receipt) => wire::RoutedCommitted {
                    commit_ts: receipt.commit_ts.as_raw(),
                    applied_index: receipt.applied_index.unwrap_or(0),
                    error: Vec::new(),
                },
                Err(err) => wire::RoutedCommitted {
                    error: errors::encode(err),
                    ..wire::RoutedCommitted::default()
                },
            },
        ))
    }

    async fn rollback(
        &self,
        request: Request<wire::RoutedRollback>,
    ) -> Result<Response<wire::RoutedRolledBack>, Status> {
        check_codec(&request)?;
        let rollback = rollback_from_wire(request.into_inner()).map_err(malformed)?;
        let error = match self.blocking(move |e| e.rollback(rollback)).await? {
            Ok(()) => Vec::new(),
            Err(err) => errors::encode(err),
        };
        Ok(Response::new(wire::RoutedRolledBack { error }))
    }

    async fn operation(
        &self,
        request: Request<wire::RoutedOperation>,
    ) -> Result<Response<wire::RoutedOperationAnswer>, Status> {
        check_codec(&request)?;
        let request = request.into_inner();
        let actor = actor_from_wire(request.actor).map_err(malformed)?;
        let call = operation_call_from_wire(request.call).map_err(malformed)?;
        let answer = match self.blocking(move |e| e.operation(actor, call)).await? {
            Ok(answer) => operation_answer_to_wire(answer),
            Err(err) => wire::RoutedOperationAnswer {
                answer: Some(wire::routed_operation_answer::Answer::Error(
                    errors::encode(err),
                )),
            },
        };
        Ok(Response::new(answer))
    }
}

/// Where a member puts the answer of a routed statement while the statement
/// runs: its columns, then its rows in batches sent as they fill. A batch
/// waits for room in the call's bounded queue, so an ingress that reads
/// slowly holds the statement back rather than the answer piling up here.
/// Called from the blocking pool.
pub struct AnswerWriter {
    frames: tokio::sync::mpsc::Sender<Result<wire::RoutedStatementFrame, Status>>,
    batch: Vec<u8>,
    columns: bool,
    /// The ingress left: nothing more reaches it.
    gone: bool,
    /// A row that could not be encoded: the answer ends with this failure.
    broken: Option<String>,
}

impl AnswerWriter {
    fn new(frames: tokio::sync::mpsc::Sender<Result<wire::RoutedStatementFrame, Status>>) -> Self {
        Self {
            frames,
            batch: Vec::new(),
            columns: false,
            gone: false,
            broken: None,
        }
    }

    /// An answer nobody reads, for a caller that runs an executor directly
    /// for its result alone: the statement stops at its first row.
    pub fn unread() -> Self {
        let (frames, _) = tokio::sync::mpsc::channel(1);
        Self::new(frames)
    }

    /// Send the column names, once and before any row; later calls are
    /// ignored. `false` once nothing more is worth producing.
    pub fn columns(&mut self, names: &[String]) -> bool {
        if !self.columns {
            self.columns = true;
            self.send(Event::Columns(wire::Columns {
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
            self.flush();
        }
        self.wanted()
    }

    fn wanted(&self) -> bool {
        !self.gone && self.broken.is_none()
    }

    fn flush(&mut self) {
        if !self.batch.is_empty() {
            let batch = std::mem::take(&mut self.batch);
            self.send(Event::Rows(batch));
        }
    }

    fn send(&mut self, event: Event) {
        if !self.gone && self.frames.blocking_send(Ok(frame(event))).is_err() {
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
            return !self.gone
                && self
                    .frames
                    .blocking_send(Err(Status::internal(broken)))
                    .is_ok();
        }
        match result {
            Ok(stats) => {
                self.columns(&[]);
                self.flush();
                self.send(Event::End(stats_to_wire(stats)));
            }
            Err(err) => {
                self.flush();
                self.send(Event::Error(errors::encode(err)));
            }
        }
        !self.gone
    }
}

fn frame(event: Event) -> wire::RoutedStatementFrame {
    wire::RoutedStatementFrame { event: Some(event) }
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
    client: CommandRoutingServiceClient<Channel>,
}

impl CommandRouter {
    /// A router over `channel`, a connection to `member`, the executing
    /// member.
    pub fn new(member: u64, channel: Channel) -> Self {
        Self {
            member,
            client: CommandRoutingServiceClient::new(channel)
                .max_decoding_message_size(MAX_ROUTED_MESSAGE),
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
        Ok(Self::new(member, channel))
    }

    /// `message` as a request this member reads the error encoding of, bound
    /// to `deadline` when there is one.
    fn request<T>(message: T, deadline: Option<Duration>) -> Request<T> {
        let mut request = Request::new(message);
        request
            .metadata_mut()
            .insert(CODEC_HEADER, MetadataValue::from(ERROR_CODEC_VERSION));
        if let Some(deadline) = deadline {
            request.set_timeout(deadline);
        }
        request
    }

    /// The error a failed call answers: a refusal before the call was read
    /// did not run it; anything else lost an answer that may have run.
    fn lost(&self, status: &Status) -> DatabaseError {
        if status.metadata().contains_key(REFUSED_HEADER) {
            return DatabaseError::PeerUnreachable {
                member: self.member,
                message: status.message().to_owned(),
            };
        }
        let failure = match status.code() {
            Code::DeadlineExceeded | Code::Cancelled => PeerFailure::Deadline,
            _ => PeerFailure::Transport,
        };
        DatabaseError::PeerOutcomeUnknown {
            member: self.member,
            failure,
            message: status.message().to_owned(),
        }
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
        let request = Self::request(statement_to_wire(statement)?, deadline);
        let mut frames = self
            .client
            .clone()
            .execute(request)
            .await
            .map_err(|s| self.lost(&s))?
            .into_inner();
        // The answer opens with its columns, after the admission of a
        // transaction's statement and unless the statement failed.
        while let Some(frame) = frames.message().await.map_err(|s| self.lost(&s))? {
            match frame.event {
                Some(Event::Admitted(_)) => admitted.admitted(),
                Some(Event::Columns(columns)) => {
                    return Ok(RoutedRows {
                        router: self.clone(),
                        columns: columns.names,
                        frames,
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
        Err(self.unreadable("the executing member ended the answer before its columns".to_owned()))
    }

    /// Open a transaction at the member, which must lead, under the ID the
    /// client drew.
    ///
    /// # Errors
    ///
    /// Why the member did not open it, or how its answer was lost.
    pub async fn begin(&self, begin: RoutedBegin) -> Result<(), DatabaseError> {
        let begun = self
            .client
            .clone()
            .begin(Self::request(begin_to_wire(begin), None))
            .await
            .map_err(|s| self.lost(&s))?
            .into_inner();
        self.failure(&begun.error)
    }

    /// Commit a transaction at the leader.
    ///
    /// # Errors
    ///
    /// What the commit failed with, or how its answer was lost.
    pub async fn commit(&self, commit: RoutedCommit) -> Result<CommitReceipt, DatabaseError> {
        let deadline = commit.deadline;
        let committed = self
            .client
            .clone()
            .commit(Self::request(commit_to_wire(commit), deadline))
            .await
            .map_err(|s| self.lost(&s))?
            .into_inner();
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
        let rolled_back = self
            .client
            .clone()
            .rollback(Self::request(rollback_to_wire(rollback), None))
            .await
            .map_err(|s| self.lost(&s))?
            .into_inner();
        self.failure(&rolled_back.error)
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
        let answer = self
            .client
            .clone()
            .operation(Self::request(
                wire::RoutedOperation {
                    actor: Some(actor_to_wire(actor)),
                    call: Some(operation_call_to_wire(call)),
                },
                None,
            ))
            .await
            .map_err(|s| self.lost(&s))?
            .into_inner();
        if let Some(wire::routed_operation_answer::Answer::Error(bytes)) = &answer.answer {
            return Err(errors::decode(bytes).unwrap_or_else(|e| self.unreadable(e)));
        }
        operation_answer_from_wire(answer).map_err(|e| self.unreadable(e))
    }
}

/// The rows of a statement another member runs, read off the wire as the
/// caller asks for them. At most one received batch is held here; the rest
/// waits in the member's bounded queue, which holds the member back.
pub struct RoutedRows {
    router: CommandRouter,
    columns: Vec<String>,
    frames: tonic::Streaming<wire::RoutedStatementFrame>,
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
            let Some(frame) = self
                .frames
                .message()
                .await
                .map_err(|s| self.router.lost(&s))?
            else {
                return Err(self.router.unreadable(
                    "the executing member ended the answer before its statistics".to_owned(),
                ));
            };
            match frame.event {
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
    })
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
        timeout_outcome: match begin.timeout_outcome {
            None => 0,
            Some(TimeoutOutcome::Rollback) => 1,
            Some(TimeoutOutcome::Commit) => 2,
        },
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
        timeout_outcome: match begin.timeout_outcome {
            0 => None,
            1 => Some(TimeoutOutcome::Rollback),
            2 => Some(TimeoutOutcome::Commit),
            other => {
                return Err(Status::invalid_argument(format!(
                    "timeout outcome {other} is not one this member knows"
                )));
            }
        },
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
