//! gRPC ChangeStreamService: CE oplog CDC consumer.
//!
//! Tails the Raft log's oplog segments and streams [`ChangeEvent`] messages
//! to the client, only for entries this node has applied: the log also holds
//! entries that are not committed yet, which a later leader may truncate and
//! replace. A caught-up stream sleeps until this node applies another entry,
//! waking only to heartbeat its registration.
//!
//! Every stream reads as a consumer registered in the
//! [`ShardConsumerRegistry`] with the retention policy its client chose. The
//! registration outlives the connection: a client reconnects to the same
//! incarnation, and ends it with `CancelSubscription`. Its checkpoint moves
//! only on `AcknowledgeSubscription`, the client's statement that it holds
//! the events before a position: sending an event, or a resume token naming
//! where to read, releases nothing.
//!
//! In embedded mode (no Raft) nothing is applied from a Raft log and the
//! stream is empty — no error.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Code, Request, Response, Status};

use coordinode_raft::cluster::RaftNode;
use coordinode_raft::storage::raft_oplog_dirs;
use coordinode_replicate::{
    ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy, InitialSeqno, RegisteredHandle,
    RegistryError, SeqnoConsumerRegistry, ShardConsumerRegistry, TopologyScope,
    ValidatedRetentionBounds,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::error::StorageError;
use coordinode_storage::oplog::entry::OplogOp;
use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer, ResumeToken};

use super::error_details::{Reason, status_with_reason};
use crate::proto::replication::cdc::{
    AcknowledgeSubscriptionRequest, AcknowledgeSubscriptionResponse, CancelSubscriptionRequest,
    CancelSubscriptionResponse, CdcFilters as ProtoCdcFilters, ChangeEvent, ChangeOp, ChangeOpType,
    ConsumerRetention, ResumeToken as ProtoResumeToken, SubscribeRequest,
    change_stream_service_server::ChangeStreamService, consumer_retention,
};

/// One past the last Raft log entry this node has applied, `0` when none:
/// the exclusive bound of what a change stream may send.
pub type AppliedFrontier = Arc<dyn Fn() -> u64 + Send + Sync>;

/// Changes whenever the [`AppliedFrontier`] may have moved; `None` when it
/// never moves (no Raft log).
pub type AppliedSignal = Option<tokio::sync::watch::Receiver<u64>>;

/// Response metadata key carrying the registration's incarnation, which the
/// client passes to resume it and to cancel it.
pub const INCARNATION_METADATA: &str = "x-coordinode-consumer-incarnation";

/// gRPC CDC service for one shard.
pub struct ChangeEventServiceImpl {
    /// The shard whose Raft log the service streams.
    shard_id: u32,
    /// Every directory holding segments of that log.
    oplog_dirs: Vec<PathBuf>,
    registry: ShardConsumerRegistry,
    /// Bound of the entries a stream may send.
    applied: AppliedFrontier,
    /// Wakes a caught-up stream when the bound moves.
    applied_changes: AppliedSignal,
    /// Read pacing of every stream.
    tuning: CdcStreamTuning,
}

impl ChangeEventServiceImpl {
    /// A service streaming shard `shard_id`'s Raft log from `oplog_dirs`
    /// (see `coordinode_raft::storage::raft_oplog_dirs`), up to `applied`,
    /// which `applied_changes` signals moving.
    pub fn new(
        shard_id: u32,
        oplog_dirs: Vec<PathBuf>,
        registry: ShardConsumerRegistry,
        applied: AppliedFrontier,
        applied_changes: AppliedSignal,
    ) -> Self {
        Self {
            shard_id,
            oplog_dirs,
            registry,
            applied,
            applied_changes,
            tuning: CdcStreamTuning::default(),
        }
    }

    /// The same service pacing its streams by `tuning`.
    pub fn with_tuning(mut self, tuning: CdcStreamTuning) -> Self {
        self.tuning = tuning;
        self
    }

    /// A service streaming shard 0's Raft log of `node`, whose store is
    /// `engine`, as the node applies it.
    ///
    /// # Errors
    ///
    /// No endpoint of `engine` is eligible to hold the oplog.
    pub fn for_raft_node(
        engine: &StorageEngine,
        node: Arc<RaftNode>,
        registry: ShardConsumerRegistry,
    ) -> std::io::Result<Self> {
        let dirs = raft_oplog_dirs(engine, 0)?.all;
        let changes = node.subscribe_applied();
        Ok(Self::new(
            0,
            dirs,
            registry,
            Arc::new(move || node.applied_through()),
            Some(changes),
        ))
    }

    /// The retention policy a registering request chose, refused when it
    /// chose none or chose bounds its stream could not keep.
    fn retention(
        &self,
        proto: Option<ConsumerRetention>,
    ) -> Result<ConsumerRetentionPolicy, Status> {
        let policy = proto.and_then(|r| r.policy).ok_or_else(|| {
            Status::invalid_argument(
                "registering a consumer requires a retention policy, strict or bounded",
            )
        })?;
        match policy {
            consumer_retention::Policy::Strict(_) => Ok(ConsumerRetentionPolicy::Strict),
            consumer_retention::Policy::Bounded(b) => {
                if let Some(liveness) = b.liveness_timeout_ms {
                    // A connected stream proves liveness at the heartbeat
                    // interval: a timeout no longer than that would end a
                    // consumer that is connected and reading.
                    let interval = self.tuning.heartbeat_interval.as_millis();
                    if u128::from(liveness) <= interval {
                        return Err(Status::invalid_argument(format!(
                            "liveness_timeout_ms {liveness} must exceed the server's heartbeat \
                             interval of {interval} ms"
                        )));
                    }
                }
                ValidatedRetentionBounds::new(
                    b.max_progress_lag_ms,
                    b.max_retained_bytes,
                    b.liveness_timeout_ms,
                )
                .map(ConsumerRetentionPolicy::Bounded)
                .map_err(registry_status)
            }
        }
    }
}

/// How a change stream paces its reads (`cdc_heartbeat_interval_ms`,
/// `cdc_batch_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdcStreamTuning {
    /// How often a stream waiting (caught up, or on a slow reader) heartbeats
    /// its registration. Delivery does not wait on it: a caught-up stream
    /// wakes as soon as another entry applies.
    pub heartbeat_interval: Duration,
    /// Most entries read and sent per poll (back-pressure).
    pub batch_size: NonZeroUsize,
}

impl Default for CdcStreamTuning {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(10),
            batch_size: const {
                match NonZeroUsize::new(256) {
                    Some(n) => n,
                    None => unreachable!(),
                }
            },
        }
    }
}

/// The client status for a registry refusal.
fn registry_status(e: RegistryError) -> Status {
    let message = e.to_string();
    match e {
        RegistryError::EmptyConsumerId
        | RegistryError::UnsupportedScope(_)
        | RegistryError::InvalidRetention(_) => Status::invalid_argument(message),
        RegistryError::AlreadyRegistered { .. } => Status::already_exists(message),
        RegistryError::UnknownConsumer(_) => Status::not_found(message),
        RegistryError::Terminated {
            incarnation,
            reason,
            checkpoint,
            ..
        } => status_with_reason(
            Code::FailedPrecondition,
            message,
            Reason::ConsumerTerminated,
            [
                ("incarnation", incarnation.to_string()),
                ("reason", format!("{reason:?}")),
                ("checkpoint", checkpoint.to_string()),
            ],
        ),
        RegistryError::StaleIncarnation {
            handle, current, ..
        } => status_with_reason(
            Code::FailedPrecondition,
            message,
            Reason::ConsumerTerminated,
            [
                ("incarnation", handle.to_string()),
                ("current_incarnation", current.to_string()),
            ],
        ),
        RegistryError::RetentionLost { checkpoint, floor } => status_with_reason(
            Code::FailedPrecondition,
            format!("change stream retention lost: {message}"),
            Reason::RetentionLost,
            [
                ("requested_index", checkpoint.to_string()),
                ("first_retained_index", floor.to_string()),
            ],
        ),
        RegistryError::Backpressure => status_with_reason(
            Code::ResourceExhausted,
            message,
            Reason::WriteBackpressure,
            None,
        ),
        RegistryError::NotLeader { leader_id } => status_with_reason(
            Code::Unavailable,
            message,
            Reason::NotLeader,
            leader_id.map(|id| ("leader_id", id.to_string())),
        ),
        RegistryError::Replication(_) => Status::internal(message),
    }
}

/// Run a blocking registry call off the async runtime: each one commits a
/// transaction and waits on its durability.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, RegistryError> + Send + 'static,
) -> Result<T, Status> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| Status::internal(format!("registry task: {e}")))?
        .map_err(registry_status)
}

#[tonic::async_trait]
impl ChangeStreamService for ChangeEventServiceImpl {
    type SubscribeStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<ChangeEvent, Status>> + Send>>;

    async fn subscribe(
        &self,
        request: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        let req = request.into_inner();
        let shard_id = self.shard_id;
        if req.consumer_id.is_empty() {
            return Err(Status::invalid_argument(
                "a change stream needs a consumer_id",
            ));
        }
        let token = match req.resume_token {
            Some(t) if t.shard_id != shard_id => {
                return Err(Status::invalid_argument(format!(
                    "resume token is for shard {}, this stream serves shard {shard_id}",
                    t.shard_id
                )));
            }
            Some(t) => Some(ResumeToken {
                shard_id: t.shard_id,
                segment_id: t.segment_id,
                entry_offset: t.entry_offset,
            }),
            None => None,
        };
        let start = token
            .as_ref()
            .filter(|t| !t.is_start())
            .map(ResumeToken::next_index)
            .transpose()
            .map_err(|e| Status::invalid_argument(format!("resume token: {e}")))?;
        let filters = proto_filters_to_cdc(req.filters);

        let registry = self.registry.clone();
        let consumer_id = req.consumer_id;
        let handle = if req.incarnation == 0 {
            let retention = self.retention(req.retention)?;
            let reg = ConsumerRegistration {
                consumer_id,
                kind: ConsumerKind::OplogEvents,
                scope: TopologyScope::Shard(shard_id as u16),
                initial_seqno: start.map_or(InitialSeqno::FromEarliestRetained, InitialSeqno::At),
                retention,
            };
            let registry = registry.clone();
            blocking(move || registry.register(reg)).await?
        } else {
            let registry = registry.clone();
            let incarnation = req.incarnation;
            blocking(move || registry.resume(&consumer_id, incarnation)).await?
        };

        // Without a token a resumed stream starts where its registration
        // stands; a registered one already started at the oldest retained
        // entry, which is its checkpoint.
        let from = match start {
            Some(index) => index,
            None => {
                let (registry, handle) = (registry.clone(), handle.clone());
                blocking(move || registry.check_retention(&handle)).await?
            }
        };
        let tailer_token = ResumeToken {
            shard_id,
            segment_id: from,
            entry_offset: 0,
        };
        let tailer = OplogTailer::new(&self.oplog_dirs, tailer_token)
            .map_err(|e| Status::invalid_argument(format!("resume token: {e}")))?;

        let (tx, rx) = mpsc::channel::<Result<ChangeEvent, Status>>(64);
        let incarnation = handle.incarnation();
        tokio::spawn(stream_consumer(StreamState {
            shard_id,
            registry,
            handle,
            tailer,
            filters,
            tx,
            applied: Arc::clone(&self.applied),
            applied_changes: self.applied_changes.clone(),
            tuning: self.tuning,
        }));

        let stream: Pin<Box<dyn tokio_stream::Stream<Item = Result<ChangeEvent, Status>> + Send>> =
            Box::pin(ReceiverStream::new(rx));
        let mut response = Response::new(stream);
        response.metadata_mut().insert(
            INCARNATION_METADATA,
            incarnation
                .to_string()
                .parse()
                .map_err(|e| Status::internal(format!("incarnation metadata: {e}")))?,
        );
        Ok(response)
    }

    async fn cancel_subscription(
        &self,
        request: Request<CancelSubscriptionRequest>,
    ) -> Result<Response<CancelSubscriptionResponse>, Status> {
        let req = request.into_inner();
        if req.consumer_id.is_empty() {
            return Err(Status::invalid_argument("cancelling needs a consumer_id"));
        }
        let registry = self.registry.clone();
        let handle = RegisteredHandle::new(req.consumer_id, req.incarnation);
        blocking(move || registry.unregister(handle)).await?;
        Ok(Response::new(CancelSubscriptionResponse {}))
    }

    async fn acknowledge_subscription(
        &self,
        request: Request<AcknowledgeSubscriptionRequest>,
    ) -> Result<Response<AcknowledgeSubscriptionResponse>, Status> {
        let req = request.into_inner();
        if req.consumer_id.is_empty() {
            return Err(Status::invalid_argument(
                "acknowledging needs a consumer_id",
            ));
        }
        let position = req
            .position
            .ok_or_else(|| Status::invalid_argument("acknowledging needs a position"))?;
        if position.shard_id != self.shard_id {
            return Err(Status::invalid_argument(format!(
                "position is for shard {}, this service serves shard {}",
                position.shard_id, self.shard_id
            )));
        }
        let index = ResumeToken {
            shard_id: position.shard_id,
            segment_id: position.segment_id,
            entry_offset: position.entry_offset,
        }
        .next_index()
        .map_err(|e| Status::invalid_argument(format!("position: {e}")))?;
        // Nothing past what this node applied was sent, so a position beyond
        // it is not a statement about held events and releases nothing.
        let applied = (self.applied)();
        if index > applied {
            return Err(Status::invalid_argument(format!(
                "position {index} is past the last entry this node applied ({applied})"
            )));
        }
        let registry = self.registry.clone();
        let handle = RegisteredHandle::new(req.consumer_id, req.incarnation);
        // Monotonic: an earlier position than one already acknowledged keeps
        // the later one.
        blocking(move || registry.checkpoint(&handle, index)).await?;
        Ok(Response::new(AcknowledgeSubscriptionResponse {}))
    }
}

/// Everything one stream's task owns.
struct StreamState {
    shard_id: u32,
    registry: ShardConsumerRegistry,
    handle: RegisteredHandle,
    tailer: OplogTailer,
    filters: CdcFilters,
    tx: mpsc::Sender<Result<ChangeEvent, Status>>,
    applied: AppliedFrontier,
    applied_changes: AppliedSignal,
    tuning: CdcStreamTuning,
}

/// Tail the log for one stream until the client leaves, the registration
/// ends, or the log no longer holds what the stream needs. Leaving does not
/// end the registration: the client resumes it.
async fn stream_consumer(mut s: StreamState) {
    'stream: loop {
        // Client cancelled (channel closed).
        if s.tx.is_closed() {
            break;
        }
        // Marked seen before the read: an entry applied after it wakes the
        // wait below.
        if let Some(changes) = s.applied_changes.as_mut() {
            changes.borrow_and_update();
        }

        // An ended registration, or history the log no longer holds, is a
        // clean refusal rather than a silent gap.
        let check = {
            let (registry, handle) = (s.registry.clone(), s.handle.clone());
            blocking(move || registry.check_retention(&handle)).await
        };
        if let Err(status) = check {
            let _ = s.tx.send(Err(status)).await;
            break;
        }

        let read_from = s.tailer.next_index();
        let batch = match s
            .tailer
            .read_next(s.tuning.batch_size.get(), &s.filters, (s.applied)())
        {
            Ok(b) => b,
            Err(StorageError::RetentionLost {
                requested,
                first_retained,
            }) => {
                let _ =
                    s.tx.send(Err(registry_status(RegistryError::RetentionLost {
                        checkpoint: requested,
                        floor: first_retained,
                    })))
                    .await;
                break;
            }
            Err(e) => {
                let _ = s.tx.send(Err(Status::internal(e.to_string()))).await;
                break;
            }
        };
        let caught_up = batch.is_empty();

        // Where the client can resume and acknowledge after what was sent.
        let mut sent_to = read_from;
        for (entry, token) in batch {
            if let Ok(next) = token.next_index() {
                sent_to = sent_to.max(next);
            }
            // A slow reader leaves no room in the channel; keep its
            // registration alive while waiting, as an idle poll does.
            let permit = loop {
                match tokio::time::timeout(s.tuning.heartbeat_interval, s.tx.reserve()).await {
                    Ok(Ok(permit)) => break permit,
                    // Client disconnected mid-batch.
                    Ok(Err(_)) => break 'stream,
                    Err(_) => {
                        if let Err(e) = s.registry.heartbeat(&s.handle) {
                            tracing::warn!(error = %e, "change stream heartbeat failed");
                        }
                    }
                }
            };
            permit.send(Ok(oplog_entry_to_proto(entry, token)));
        }
        // Sending is not delivery: the registration moves only when the
        // client acknowledges. Entries the filters dropped after the last
        // event sent are reported as a progress event, so a client whose
        // filters match little can still acknowledge past them.
        let read_to = s.tailer.next_index();
        if read_to > sent_to {
            let progress = ChangeEvent {
                ts: 0,
                term: 0,
                log_index: read_to - 1,
                shard_id: s.shard_id,
                is_migration: false,
                ops: Vec::new(),
                position: Some(ProtoResumeToken {
                    shard_id: s.shard_id,
                    segment_id: read_to,
                    entry_offset: 0,
                }),
            };
            if s.tx.send(Ok(progress)).await.is_err() {
                break;
            }
        }

        if caught_up {
            // Heartbeat so a BOUNDED liveness timeout sees a connected reader,
            // then sleep until an entry applies, the client leaves, or the
            // next heartbeat is due.
            if let Err(e) = s.registry.heartbeat(&s.handle) {
                tracing::warn!(error = %e, "change stream heartbeat failed");
            }
            let heartbeat = tokio::time::sleep(s.tuning.heartbeat_interval);
            match s.applied_changes.as_mut() {
                Some(changes) => tokio::select! {
                    changed = changes.changed() => {
                        if changed.is_err() {
                            // The node is gone; nothing more applies.
                            s.applied_changes = None;
                        }
                    }
                    () = s.tx.closed() => break,
                    () = heartbeat => {}
                },
                None => tokio::select! {
                    () = s.tx.closed() => break,
                    () = heartbeat => {}
                },
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;

// ── Conversion helpers ────────────────────────────────────────────────────────

fn proto_filters_to_cdc(proto: Option<ProtoCdcFilters>) -> CdcFilters {
    match proto {
        None => CdcFilters::default(),
        Some(f) => CdcFilters {
            edge_types: f.edge_types,
            is_migration: f.is_migration,
        },
    }
}

fn oplog_entry_to_proto(
    entry: coordinode_storage::oplog::entry::OplogEntry,
    token: ResumeToken,
) -> ChangeEvent {
    let ops = entry
        .ops
        .into_iter()
        .filter_map(oplog_op_to_proto)
        .collect();

    ChangeEvent {
        ts: entry.ts,
        term: entry.term,
        log_index: entry.index,
        shard_id: entry.shard,
        is_migration: entry.is_migration,
        ops,
        position: Some(ProtoResumeToken {
            shard_id: token.shard_id,
            segment_id: token.segment_id,
            entry_offset: token.entry_offset,
        }),
    }
}

/// The change a journalled op shows a CDC reader, or `None` for DERIVED
/// index work: index maintenance is not a change of its own, and the data
/// change it follows is already on the stream.
fn oplog_op_to_proto(op: OplogOp) -> Option<ChangeOp> {
    Some(match op {
        OplogOp::Insert {
            partition,
            key,
            value,
        } => ChangeOp {
            r#type: ChangeOpType::Insert as i32,
            partition: partition as u32,
            key,
            value,
        },
        OplogOp::Delete { partition, key } => ChangeOp {
            r#type: ChangeOpType::Delete as i32,
            partition: partition as u32,
            key,
            value: vec![],
        },
        OplogOp::Merge {
            partition,
            key,
            operand,
        } => ChangeOp {
            r#type: ChangeOpType::Merge as i32,
            partition: partition as u32,
            key,
            value: operand,
        },
        OplogOp::RemoveRange {
            partition,
            start,
            end,
        } => ChangeOp {
            r#type: ChangeOpType::RemoveRange as i32,
            partition: partition as u32,
            // Half-open range [start, end): key carries start, value carries end.
            key: start,
            value: end,
        },
        OplogOp::Noop => ChangeOp {
            r#type: ChangeOpType::Noop as i32,
            partition: 0,
            key: vec![],
            value: vec![],
        },
        OplogOp::RaftEntry { data } => ChangeOp {
            r#type: ChangeOpType::RaftEntry as i32,
            partition: 0,
            key: vec![],
            value: data,
        },
        OplogOp::RaftTruncation { after_index } => ChangeOp {
            r#type: ChangeOpType::RaftTruncation as i32,
            partition: 0,
            key: after_index.to_be_bytes().to_vec(),
            value: vec![],
        },
        // A columnar table write is keyed by table id, not a partition
        // discriminant, so it cannot be carried by the partition-shaped ChangeOp.
        // Surfacing columnar tables on the CDC stream needs a table-id field on
        // the wire; until then a columnar write maps to a no-op change rather
        // than a misencoded partition op.
        OplogOp::ColumnarInsert { .. } => ChangeOp {
            r#type: ChangeOpType::Noop as i32,
            partition: 0,
            key: vec![],
            value: vec![],
        },
        // The tailer hands out a unit frame expanded into its operations.
        OplogOp::Derive { .. } | OplogOp::Unit { .. } => return None,
    })
}
