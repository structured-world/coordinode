//! Whole commands carried from the member a client reached (the ingress) to
//! the member that runs them: an autonomous statement or a transaction's
//! first command to the leader, a transaction's later commands to the member
//! that holds it.
//!
//! The executing member selects the view, evaluates, validates and commits;
//! the ingress never ships a write set it evaluated itself. A command is
//! routed once: a member that cannot run it answers with the refusal that
//! says why, and the ingress decides what to tell its client.
//!
//! The library owns the transport and the encoding. What runs a command at
//! the executing member is a [`CommandExecutor`]: the server brings its own,
//! which fences, records and reports refusals the way its clients see them.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use coordinode_core::budget::CancelFlag;
use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::{Value, VectorConsistencyMode};
use coordinode_core::operations::TxnOrdering;
use coordinode_core::txn::command::{ActingContext, AdmissionSignal, Principal, TransactionId};
use coordinode_core::txn::operation::OperationDescriptor;
use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_core::txn::transaction::CommitReceipt;
use coordinode_core::txn::write_concern::{Journal, WriteAck, WriteConcern};
use coordinode_query::advisor::source::SourceContext;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::proto::internode::routing as wire;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;
use tonic::{Request, Response, Status};

use crate::db::ExpectedVersion;
use crate::operation::{IssuerId, OperationOutcome, Resolve};
use coordinode_core::group::DeploymentId;
use wire::command_routing_service_client::CommandRoutingServiceClient;
use wire::command_routing_service_server::{CommandRoutingService, CommandRoutingServiceServer};
use wire::routed_statement_frame::Event;

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
    /// The statement starts `transaction`: it opens with `actor` and the
    /// read and write concern of `settings`, statements applied as they
    /// come, then the statement runs in it.
    pub start_transaction: bool,
    /// Its number within the transaction.
    pub nonce: Option<u64>,
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
    /// How its statements are ordered.
    pub ordering: TxnOrdering,
    /// How long an ordered transaction waits for a missing statement; the
    /// leader's setting when `None`.
    pub drain_timeout: Option<Duration>,
}

/// Commit an interactive transaction at the leader.
#[derive(Debug, Clone)]
pub struct RoutedCommit {
    /// Who the commit acts for.
    pub actor: ActingContext,
    /// The transaction.
    pub transaction: TransactionId,
    /// The last nonce of an ordered chain.
    pub last_nonce: Option<u64>,
    /// The client operation the commit carries out.
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
/// service calls it from the blocking pool. A refusal is the status the
/// command's client sees, which the ingress passes on as it is.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot run commands routed from other members",
    label = "not a command executor",
    note = "implement `CommandExecutor` over the statement path that serves this member's own clients, so a routed command is fenced, run and refused the same way"
)]
pub trait CommandExecutor: Send + Sync + 'static {
    /// Run `statement`. A statement of a transaction tells `admitted` when it
    /// has its place in the transaction's order; one that ends meanwhile was
    /// cancelled through `cancel`.
    ///
    /// # Errors
    ///
    /// The status the statement failed with.
    fn execute(
        &self,
        statement: RoutedStatement,
        cancel: &CancelFlag,
        admitted: &AdmissionSignal,
    ) -> Result<RoutedAnswer, Status>;

    /// Open a transaction under the ID the client drew.
    ///
    /// # Errors
    ///
    /// The status the transaction was refused with.
    fn begin(&self, begin: RoutedBegin) -> Result<(), Status>;

    /// Commit a transaction this member holds.
    ///
    /// # Errors
    ///
    /// The status the commit failed with.
    fn commit(&self, commit: RoutedCommit) -> Result<CommitReceipt, Status>;

    /// Roll back a transaction this member holds.
    ///
    /// # Errors
    ///
    /// The status the rollback was refused with.
    fn rollback(&self, rollback: RoutedRollback) -> Result<(), Status>;

    /// Carry out a call on operation records, acting for `actor`.
    ///
    /// # Errors
    ///
    /// The status the call was refused with.
    fn operation(
        &self,
        actor: ActingContext,
        call: RoutedOperationCall,
    ) -> Result<RoutedOperationAnswer, Status>;
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

    /// Run `call` on the blocking pool and answer with what it returns.
    async fn blocking<T: Send + 'static>(
        &self,
        call: impl FnOnce(&E) -> Result<T, Status> + Send + 'static,
    ) -> Result<T, Status> {
        let executor = Arc::clone(&self.executor);
        tokio::task::spawn_blocking(move || call(&executor))
            .await
            .map_err(|e| Status::internal(format!("the routed command did not finish: {e}")))?
    }
}

#[tonic::async_trait]
impl<E: CommandExecutor> CommandRoutingService for CommandRoutingHandler<E> {
    type ExecuteStream = ReceiverStream<Result<wire::RoutedStatementFrame, Status>>;

    async fn execute(
        &self,
        request: Request<wire::RoutedStatement>,
    ) -> Result<Response<Self::ExecuteStream>, Status> {
        let statement = statement_from_wire(request.into_inner())?;
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

        let executor = Arc::clone(&self.executor);
        tokio::task::spawn_blocking(move || {
            let sent = match executor.execute(statement, &cancel, &admitted) {
                Ok(answer) => send_answer(&frames, answer),
                Err(status) => frames.blocking_send(Err(status)).is_ok(),
            };
            if !sent {
                tracing::debug!("a routed statement's ingress left before its answer");
            }
        });
        Ok(Response::new(ReceiverStream::new(answer)))
    }

    async fn begin(
        &self,
        request: Request<wire::RoutedBegin>,
    ) -> Result<Response<wire::RoutedBegun>, Status> {
        let begin = begin_from_wire(request.into_inner())?;
        self.blocking(move |e| e.begin(begin)).await?;
        Ok(Response::new(wire::RoutedBegun {}))
    }

    async fn commit(
        &self,
        request: Request<wire::RoutedCommit>,
    ) -> Result<Response<wire::RoutedCommitted>, Status> {
        let commit = commit_from_wire(request.into_inner())?;
        let receipt = self.blocking(move |e| e.commit(commit)).await?;
        Ok(Response::new(wire::RoutedCommitted {
            commit_ts: receipt.commit_ts.as_raw(),
            applied_index: receipt.applied_index.unwrap_or(0),
        }))
    }

    async fn rollback(
        &self,
        request: Request<wire::RoutedRollback>,
    ) -> Result<Response<wire::RoutedRolledBack>, Status> {
        let rollback = rollback_from_wire(request.into_inner())?;
        self.blocking(move |e| e.rollback(rollback)).await?;
        Ok(Response::new(wire::RoutedRolledBack {}))
    }

    async fn operation(
        &self,
        request: Request<wire::RoutedOperation>,
    ) -> Result<Response<wire::RoutedOperationAnswer>, Status> {
        let request = request.into_inner();
        let actor = actor_from_wire(request.actor)?;
        let call = operation_call_from_wire(request.call)?;
        let answer = self.blocking(move |e| e.operation(actor, call)).await?;
        Ok(Response::new(operation_answer_to_wire(answer)))
    }
}

/// Send `answer` as its columns, its rows in batches and its statistics.
/// `false` when the ingress left before it had all of it.
fn send_answer(
    frames: &tokio::sync::mpsc::Sender<Result<wire::RoutedStatementFrame, Status>>,
    answer: RoutedAnswer,
) -> bool {
    let send = |event| frames.blocking_send(Ok(frame(event))).is_ok();
    if !send(Event::Columns(wire::Columns {
        names: answer.columns,
    })) {
        return false;
    }
    let mut batch = Vec::new();
    for row in &answer.rows {
        if let Err(e) = rmp_serde::encode::write(&mut batch, row) {
            return frames
                .blocking_send(Err(Status::internal(format!(
                    "a result row could not be encoded: {e}"
                ))))
                .is_ok();
        }
        if batch.len() >= ROW_BATCH_BYTES && !send(Event::Rows(std::mem::take(&mut batch))) {
            return false;
        }
    }
    if !batch.is_empty() && !send(Event::Rows(batch)) {
        return false;
    }
    send(Event::End(stats_to_wire(answer.stats)))
}

fn frame(event: Event) -> wire::RoutedStatementFrame {
    wire::RoutedStatementFrame { event: Some(event) }
}

/// The ingress's side of routing: sends commands to one member and returns
/// its answers.
#[derive(Debug, Clone)]
pub struct CommandRouter {
    client: CommandRoutingServiceClient<Channel>,
}

impl CommandRouter {
    /// A router over `channel`, a connection to the executing member.
    pub fn new(channel: Channel) -> Self {
        Self {
            client: CommandRoutingServiceClient::new(channel)
                .max_decoding_message_size(MAX_ROUTED_MESSAGE),
        }
    }

    /// A router to `member`, over `node`'s connection to it: authenticated
    /// both ways like the rest of the traffic between members.
    ///
    /// # Errors
    ///
    /// `FAILED_PRECONDITION` when `node` has no peers or does not know where
    /// `member` is or what identity it proves; `UNAVAILABLE` when no
    /// connection can be made.
    pub fn to_member(node: &RaftNode, member: u64) -> Result<Self, Status> {
        let (Some(target), Some(connections)) = (node.member_target(member), node.connections())
        else {
            return Err(Status::failed_precondition(format!(
                "member {member} is not one this node knows how to reach"
            )));
        };
        let channel = connections
            .channel(&target)
            .map_err(|e| Status::unavailable(format!("member {member}: {e}")))?;
        Ok(Self::new(channel))
    }

    /// Run `statement` at the member and return its answer. `admitted` is told
    /// when the statement has its place there.
    ///
    /// # Errors
    ///
    /// The status the statement failed with at the member, or why it could
    /// not be reached.
    pub async fn execute(
        &self,
        statement: RoutedStatement,
        admitted: &AdmissionSignal,
    ) -> Result<RoutedAnswer, Status> {
        let deadline = statement.deadline;
        let mut request = Request::new(statement_to_wire(statement)?);
        if let Some(deadline) = deadline {
            request.set_timeout(deadline);
        }
        let mut frames = self.client.clone().execute(request).await?.into_inner();
        let mut answer = RoutedAnswer::default();
        while let Some(frame) = frames.message().await? {
            match frame.event {
                Some(Event::Admitted(_)) => admitted.admitted(),
                Some(Event::Columns(columns)) => answer.columns = columns.names,
                Some(Event::Rows(batch)) => decode_rows(&batch, &mut answer.rows)?,
                Some(Event::End(stats)) => {
                    answer.stats = stats_from_wire(stats);
                    return Ok(answer);
                }
                None => {}
            }
        }
        Err(Status::unavailable(
            "the executing member ended the answer before its statistics",
        ))
    }

    /// Open a transaction at the member, which must lead, under the ID the
    /// client drew.
    ///
    /// # Errors
    ///
    /// The status the member refused it with, or why it could not be
    /// reached.
    pub async fn begin(&self, begin: RoutedBegin) -> Result<(), Status> {
        self.client.clone().begin(begin_to_wire(begin)).await?;
        Ok(())
    }

    /// Commit a transaction at the leader.
    ///
    /// # Errors
    ///
    /// The status the commit failed with, or why the member could not be
    /// reached.
    pub async fn commit(&self, commit: RoutedCommit) -> Result<CommitReceipt, Status> {
        let deadline = commit.deadline;
        let mut request = Request::new(commit_to_wire(commit));
        if let Some(deadline) = deadline {
            request.set_timeout(deadline);
        }
        let committed = self.client.clone().commit(request).await?.into_inner();
        Ok(CommitReceipt {
            commit_ts: Timestamp::from_raw(committed.commit_ts),
            applied_index: (committed.applied_index > 0).then_some(committed.applied_index),
        })
    }

    /// Roll back a transaction at the leader.
    ///
    /// # Errors
    ///
    /// The status the rollback was refused with, or why the member could not
    /// be reached.
    pub async fn rollback(&self, rollback: RoutedRollback) -> Result<(), Status> {
        self.client
            .clone()
            .rollback(rollback_to_wire(rollback))
            .await?;
        Ok(())
    }

    /// Carry out a call on operation records at the member, which must lead.
    ///
    /// # Errors
    ///
    /// The status the member refused it with, or why it could not be
    /// reached.
    pub async fn operation(
        &self,
        actor: &ActingContext,
        call: RoutedOperationCall,
    ) -> Result<RoutedOperationAnswer, Status> {
        let answer = self
            .client
            .clone()
            .operation(wire::RoutedOperation {
                actor: Some(actor_to_wire(actor)),
                call: Some(operation_call_to_wire(call)),
            })
            .await?
            .into_inner();
        operation_answer_from_wire(answer)
    }
}

/// Append the rows of one batch, a run of encoded rows, to `rows`.
fn decode_rows(batch: &[u8], rows: &mut Vec<Vec<Value>>) -> Result<(), Status> {
    let mut cursor = Cursor::new(batch);
    while (cursor.position() as usize) < batch.len() {
        let row: Vec<Value> = rmp_serde::from_read(&mut cursor)
            .map_err(|e| Status::internal(format!("a routed row could not be decoded: {e}")))?;
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

fn statement_to_wire(statement: RoutedStatement) -> Result<wire::RoutedStatement, Status> {
    Ok(wire::RoutedStatement {
        actor: Some(actor_to_wire(&statement.actor)),
        query: statement.query,
        parameters: if statement.params.is_empty() {
            Vec::new()
        } else {
            rmp_serde::to_vec(&statement.params).map_err(|e| {
                Status::invalid_argument(format!("parameters could not be encoded: {e}"))
            })?
        },
        settings: Some(settings_to_wire(statement.settings)),
        transaction_id: statement
            .transaction
            .map(|id| id.to_bytes().to_vec())
            .unwrap_or_default(),
        start_transaction: statement.start_transaction,
        nonce: statement.nonce.unwrap_or(0),
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
    if statement.start_transaction && statement.transaction_id.is_empty() {
        return Err(Status::invalid_argument(
            "start_transaction: a statement starts a transaction only under its transaction_id",
        ));
    }
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
        start_transaction: statement.start_transaction,
        nonce: (statement.nonce > 0).then_some(statement.nonce),
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
        ordered: begin.ordering == TxnOrdering::Ordered,
        drain_timeout_ms: deadline_to_wire(begin.drain_timeout),
    }
}

fn begin_from_wire(begin: wire::RoutedBegin) -> Result<RoutedBegin, Status> {
    let settings = settings_from_wire(begin.settings)?;
    Ok(RoutedBegin {
        transaction: transaction_id_from_wire(&begin.transaction_id)?,
        actor: actor_from_wire(begin.actor)?,
        read_concern: settings.read_concern,
        write_concern: settings.write_concern,
        ordering: if begin.ordered {
            TxnOrdering::Ordered
        } else {
            TxnOrdering::Unordered
        },
        drain_timeout: deadline_from_wire(begin.drain_timeout_ms),
    })
}

fn commit_to_wire(commit: RoutedCommit) -> wire::RoutedCommit {
    wire::RoutedCommit {
        actor: Some(actor_to_wire(&commit.actor)),
        transaction_id: commit.transaction.to_bytes().to_vec(),
        last_nonce: commit.last_nonce.unwrap_or(0),
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
        last_nonce: (commit.last_nonce > 0).then_some(commit.last_nonce),
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
    }
}

fn rollback_from_wire(rollback: wire::RoutedRollback) -> Result<RoutedRollback, Status> {
    Ok(RoutedRollback {
        actor: actor_from_wire(rollback.actor)?,
        transaction: transaction_id_from_wire(&rollback.transaction_id)?,
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
            let (state, attempt, commit_ts) = match outcome {
                OperationOutcome::Committed { attempt, commit_ts } => {
                    (wire::RoutedOperationState::Committed, attempt, commit_ts)
                }
                OperationOutcome::Aborted => (wire::RoutedOperationState::Aborted, 0, 0),
                OperationOutcome::Unknown => (wire::RoutedOperationState::Unknown, 0, 0),
                OperationOutcome::Retired => (wire::RoutedOperationState::Retired, 0, 0),
            };
            Answer::Outcome(wire::RoutedOutcome {
                state: state.into(),
                attempt,
                commit_ts,
            })
        }
        RoutedOperationAnswer::Done => Answer::Done(wire::RoutedDone {}),
    };
    wire::RoutedOperationAnswer {
        answer: Some(answer),
    }
}

fn operation_answer_from_wire(
    answer: wire::RoutedOperationAnswer,
) -> Result<RoutedOperationAnswer, Status> {
    use wire::routed_operation_answer::Answer;
    Ok(match answer.answer {
        Some(Answer::Enrolled(enrolled)) => RoutedOperationAnswer::Enrolled {
            incarnation: enrolled.incarnation,
            next_sequence: enrolled.next_sequence,
            deployment: DeploymentId::from_slice(&enrolled.deployment)
                .ok_or_else(|| Status::internal("the leader answered a malformed deployment"))?,
        },
        Some(Answer::Outcome(outcome)) => RoutedOperationAnswer::Outcome(
            match wire::RoutedOperationState::try_from(outcome.state) {
                Ok(wire::RoutedOperationState::Committed) => OperationOutcome::Committed {
                    attempt: outcome.attempt,
                    commit_ts: outcome.commit_ts,
                },
                Ok(wire::RoutedOperationState::Aborted) => OperationOutcome::Aborted,
                Ok(wire::RoutedOperationState::Unknown) => OperationOutcome::Unknown,
                Ok(wire::RoutedOperationState::Retired) => OperationOutcome::Retired,
                _ => {
                    return Err(Status::internal(format!(
                        "the leader answered an operation state {} this member does not know",
                        outcome.state
                    )));
                }
            },
        ),
        Some(Answer::Done(_)) => RoutedOperationAnswer::Done,
        None => return Err(Status::internal("the leader answered nothing")),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
