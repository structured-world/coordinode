//! gRPC ChangeStreamService: CE change data capture as logical events.
//!
//! Reads the Raft log's committed units, only those this node has applied:
//! the log also holds entries that are not committed yet, which a later
//! leader may truncate and replace. Each unit is projected into its logical
//! events (puts, patches and removals of nodes and relationships) by the
//! storage engine's change feed, filtered by the subscription's interest and
//! streamed in batches with the cursor after them. A caught-up stream sleeps
//! until this node applies another entry, waking only to heartbeat its
//! registration.
//!
//! Every stream reads as a consumer registered in the
//! [`ShardConsumerRegistry`] with the retention policy its client chose. The
//! registration outlives the connection: a client reconnects to the same
//! incarnation, and ends it with `CancelSubscription`. Its checkpoint moves
//! only on `AcknowledgeSubscription`, the client's statement that it holds
//! the units before a cursor: sending a batch, or a cursor naming where to
//! read, releases nothing.
//!
//! In embedded mode (no Raft) nothing is applied from a Raft log and the
//! stream is empty: no error.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use rustc_hash::FxHashMap;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Code, Request, Response, Status};

use coordinode_core::budget::QueryBudget;
use coordinode_core::schema::label_id::LabelId;
use coordinode_modality::change_interest::{InterestRefusal, resolve_change_interest};
use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::storage::raft_oplog_dirs;
use coordinode_replicate::{
    ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy, InitialSeqno, RegisteredHandle,
    RegistrationWatch, RegistryError, RetentionPressure, SeqnoConsumerRegistry,
    ShardConsumerRegistry, TopologyScope,
};
use coordinode_storage::engine::change_feed::{
    self as feed, ChangeCursor, CursorError, Interest, OwnedTarget,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::error::StorageError;
use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer};

use super::error_details::{Reason, status_with_reason};
use crate::proto::replication::cdc::{
    AcknowledgeSubscriptionRequest, AcknowledgeSubscriptionResponse, CancelSubscriptionRequest,
    CancelSubscriptionResponse, ChangeAction, ChangeBatch, ChangeDelivery,
    ChangeInterest as ProtoInterest, ChangeUnit, ConsumerRetention, LogicalChange, NodeTarget,
    RelationshipTarget, ScopeTarget, SubscribeRequest,
    change_stream_service_server::ChangeStreamService, consumer_retention, logical_change,
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
    /// The store applying that log: it projects the units and names their
    /// labels.
    engine: Arc<StorageEngine>,
    /// Every directory holding segments of that log.
    oplog_dirs: Vec<PathBuf>,
    registry: ShardConsumerRegistry,
    /// Bound of the entries a stream may send.
    applied: AppliedFrontier,
    /// Wakes a caught-up stream when the bound moves.
    applied_changes: AppliedSignal,
    /// Read pacing of every stream.
    tuning: CdcStreamTuning,
    /// The log reader every stream shares, made on the first subscription.
    // no-std: spin::Mutex; taken once per subscription.
    hub: parking_lot::Mutex<Option<Arc<CdcHub>>>,
}

impl ChangeEventServiceImpl {
    /// The shard's shared log reader, made at the applied position on the
    /// first call.
    fn hub(&self) -> Result<Arc<CdcHub>, Status> {
        let mut slot = self.hub.lock();
        if let Some(hub) = slot.as_ref() {
            return Ok(Arc::clone(hub));
        }
        let hub = Arc::new(
            CdcHub::new(
                &self.oplog_dirs,
                self.shard_id,
                (self.applied)(),
                self.tuning.buffer_bytes,
            )
            .map_err(|e| Status::internal(format!("change stream reader: {e}")))?,
        );
        *slot = Some(Arc::clone(&hub));
        Ok(hub)
    }

    /// A service streaming shard `shard_id`'s Raft log, applied by `engine`,
    /// from `oplog_dirs` (see `coordinode_raft::storage::raft_oplog_dirs`),
    /// up to `applied`, which `applied_changes` signals moving.
    pub fn new(
        shard_id: u32,
        engine: Arc<StorageEngine>,
        oplog_dirs: Vec<PathBuf>,
        registry: ShardConsumerRegistry,
        applied: AppliedFrontier,
        applied_changes: AppliedSignal,
    ) -> Self {
        Self {
            shard_id,
            engine,
            oplog_dirs,
            registry,
            applied,
            applied_changes,
            tuning: CdcStreamTuning::default(),
            hub: parking_lot::Mutex::new(None),
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
        engine: &Arc<StorageEngine>,
        node: Arc<RaftNode>,
        registry: ShardConsumerRegistry,
    ) -> std::io::Result<Self> {
        let dirs = raft_oplog_dirs(engine, 0)?.all;
        let changes = node.subscribe_applied();
        Ok(Self::new(
            0,
            Arc::clone(engine),
            dirs,
            registry,
            Arc::new(move || node.applied_through()),
            Some(changes),
        ))
    }
}

/// The retention policy a registering request chose, refused when it chose
/// none.
fn retention(proto: Option<ConsumerRetention>) -> Result<ConsumerRetentionPolicy, Status> {
    let policy = proto.and_then(|r| r.policy).ok_or_else(|| {
        Status::invalid_argument(
            "registering a consumer requires a retention policy, strict or ordinary",
        )
    })?;
    Ok(match policy {
        consumer_retention::Policy::Strict(_) => ConsumerRetentionPolicy::Strict,
        consumer_retention::Policy::Ordinary(_) => ConsumerRetentionPolicy::Ordinary,
    })
}

/// How a change stream paces its reads (`cdc_heartbeat_interval_ms`,
/// `cdc_batch_size`, `cdc_buffer_bytes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdcStreamTuning {
    /// How often a stream waiting (caught up, or on a slow reader) heartbeats
    /// its registration. Delivery does not wait on it: a caught-up stream
    /// wakes as soon as another entry applies.
    pub heartbeat_interval: Duration,
    /// Most log entries read and sent per batch (back-pressure).
    pub batch_size: NonZeroUsize,
    /// Most bytes of log entries the streams' shared reader keeps in memory
    /// for streams that have not passed them yet (`cdc_buffer_bytes`); a
    /// stream further behind reads from the log. It also bounds the events
    /// one read of a stream holds: a larger unit is sent in fragments.
    pub buffer_bytes: usize,
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
            buffer_bytes: DEFAULT_CDC_BUFFER_BYTES,
        }
    }
}

/// Default of [`CdcStreamTuning::buffer_bytes`]: 64 MiB.
pub const DEFAULT_CDC_BUFFER_BYTES: usize = 64 << 20;

/// The client status for a registry refusal.
fn registry_status(e: RegistryError) -> Status {
    let message = e.to_string();
    match e {
        RegistryError::EmptyConsumerId
        | RegistryError::ConsumerIdTooLong { .. }
        | RegistryError::UnsupportedScope(_) => Status::invalid_argument(message),
        RegistryError::AlreadyRegistered { .. } => Status::already_exists(message),
        RegistryError::UnknownConsumer(_) => Status::not_found(message),
        RegistryError::Terminated {
            incarnation,
            reason,
            checkpoint,
            retained_from,
            ..
        } => status_with_reason(
            Code::FailedPrecondition,
            message,
            Reason::ConsumerTerminated,
            [
                ("incarnation", incarnation.to_string()),
                ("reason", format!("{reason:?}")),
                ("checkpoint", checkpoint.to_string()),
                ("retainedFrom", retained_from.to_string()),
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
        RegistryError::Backpressure(pressure) => {
            let cause = match &pressure {
                RetentionPressure::WritePressure => "writePressure",
                RetentionPressure::FreeSpace => "freeSpace",
                RetentionPressure::Footprint { .. } => "footprint",
                RetentionPressure::Unknown(_) => "unknown",
            };
            let mut metadata = vec![("pressure", cause.to_string())];
            if let RetentionPressure::Footprint { endpoint_id } = pressure {
                metadata.push(("endpointId", endpoint_id));
            }
            status_with_reason(
                Code::ResourceExhausted,
                message,
                Reason::WriteBackpressure,
                metadata,
            )
        }
        RegistryError::NotLeader { leader_id } => status_with_reason(
            Code::Unavailable,
            message,
            Reason::NotLeader,
            leader_id.map(|id| ("leader_id", id.to_string())),
        ),
        RegistryError::Replication(_) => Status::internal(message),
    }
}

/// The client status for a read of the feed that failed at log `index`.
fn read_status(e: StorageError, index: u64) -> Status {
    match e {
        StorageError::RetentionLost {
            requested,
            first_retained,
        } => registry_status(RegistryError::RetentionLost {
            checkpoint: requested,
            floor: first_retained,
        }),
        StorageError::ChangeFeedUnsupported { reason } => status_with_reason(
            Code::FailedPrecondition,
            format!("the change stream cannot read on: {reason}"),
            Reason::ChangeFeedUnsupported,
            [("log_index", index.to_string())],
        ),
        StorageError::ChangeCursor(e) => Status::invalid_argument(e.to_string()),
        e => Status::internal(e.to_string()),
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
        Pin<Box<dyn tokio_stream::Stream<Item = Result<ChangeBatch, Status>> + Send>>;

    async fn subscribe(
        &self,
        request: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        let (tx, rx) = mpsc::channel::<Result<ChangeBatch, Status>>(64);
        let incarnation = self
            .open(request.into_inner(), Delivery::Stream { tx })
            .await?;
        let stream: Pin<Box<dyn tokio_stream::Stream<Item = Result<ChangeBatch, Status>> + Send>> =
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
        self.cancel(request.into_inner()).await?;
        Ok(Response::new(CancelSubscriptionResponse {}))
    }

    async fn acknowledge_subscription(
        &self,
        request: Request<AcknowledgeSubscriptionRequest>,
    ) -> Result<Response<AcknowledgeSubscriptionResponse>, Status> {
        self.acknowledge(request.into_inner()).await?;
        Ok(Response::new(AcknowledgeSubscriptionResponse {}))
    }
}

/// The interest a request names, bound to the labels its names are bound to
/// now.
fn interest(engine: &StorageEngine, proto: Option<ProtoInterest>) -> Result<Interest, Status> {
    let proto = proto.unwrap_or_default();
    let labels = proto.labels.map(|l| l.names);
    let edge_types = proto.edge_types.map(|t| t.names);
    let resolved = resolve_change_interest(engine, labels.as_deref(), edge_types.as_deref())
        .map_err(|e| Status::internal(format!("the catalog: {e}")))?;
    resolved
        .map(|resolved| resolved.with_blobs(proto.blobs))
        .map_err(|refusal| match refusal {
            InterestRefusal::Unbound(_) => Status::invalid_argument(refusal.to_string()),
        })
}

impl ChangeEventServiceImpl {
    /// Register or resume the consumer `req` names and start delivering its
    /// units to `delivery`. Returns the registration's incarnation.
    ///
    /// # Errors
    ///
    /// The request is malformed, the registry refuses it, or the log no
    /// longer holds where it starts.
    pub(crate) async fn open(
        &self,
        req: SubscribeRequest,
        delivery: Delivery,
    ) -> Result<u64, Status> {
        let shard_id = self.shard_id;
        if req.consumer_id.is_empty() {
            return Err(Status::invalid_argument(
                "a change stream needs a consumer_id",
            ));
        }
        let interest = interest(&self.engine, req.interest)?;
        let start = if req.cursor.is_empty() {
            None
        } else {
            let cursor = ChangeCursor::from_bytes(&req.cursor)
                .map_err(|e| Status::invalid_argument(format!("cursor: {e}")))?;
            if cursor.interest() != &interest {
                return Err(Status::invalid_argument(
                    "the cursor was read with another interest than the request's",
                ));
            }
            Some(cursor)
        };

        let registry = self.registry.clone();
        let consumer_id = req.consumer_id;
        let handle = if req.incarnation == 0 {
            let retention = retention(req.retention)?;
            let reg = ConsumerRegistration {
                consumer_id,
                kind: ConsumerKind::OplogEvents,
                scope: TopologyScope::Shard(shard_id as u16),
                initial_seqno: start
                    .as_ref()
                    .map_or(InitialSeqno::FromEarliestRetained, |c| {
                        InitialSeqno::At(c.position())
                    }),
                retention,
            };
            let registry = registry.clone();
            blocking(move || registry.register(reg)).await?
        } else {
            let registry = registry.clone();
            let incarnation = req.incarnation;
            blocking(move || registry.resume(&consumer_id, incarnation)).await?
        };

        // A cursor continues where it stands, inside a transaction too.
        // Without one a resumed stream starts where its registration stands;
        // a registered one already started at the oldest retained entry,
        // which is its checkpoint.
        let cursor = match start {
            Some(cursor) => cursor,
            None => {
                let (registry, handle) = (registry.clone(), handle.clone());
                let from = blocking(move || registry.check_retention(&handle)).await?;
                self.engine
                    .change_cursor_at(from, interest)
                    .map_err(|e| Status::internal(format!("change cursor: {e}")))?
            }
        };

        let hub = self.hub()?;
        let reader = hub.reader(cursor.position());
        let incarnation = handle.incarnation();
        tokio::spawn(stream_consumer(StreamState {
            engine: Arc::clone(&self.engine),
            watch: registry.watch(&handle),
            registry,
            handle,
            cursor,
            hub,
            reader: Some(reader),
            own_tailer: None,
            delivery,
            applied: Arc::clone(&self.applied),
            applied_changes: self.applied_changes.clone(),
            tuning: self.tuning,
        }));
        Ok(incarnation)
    }

    /// End the registration `req` names.
    ///
    /// # Errors
    ///
    /// The request is malformed, or the registry refuses it.
    pub(crate) async fn cancel(&self, req: CancelSubscriptionRequest) -> Result<(), Status> {
        if req.consumer_id.is_empty() {
            return Err(Status::invalid_argument("cancelling needs a consumer_id"));
        }
        let registry = self.registry.clone();
        let handle = RegisteredHandle::new(req.consumer_id, req.incarnation);
        blocking(move || registry.unregister(handle)).await
    }

    /// Record that the consumer `req` names holds its units before the
    /// cursor it gives.
    ///
    /// # Errors
    ///
    /// The request is malformed, names a cursor of another history or past
    /// what this node has applied, or the registry refuses it.
    pub(crate) async fn acknowledge(
        &self,
        req: AcknowledgeSubscriptionRequest,
    ) -> Result<(), Status> {
        if req.consumer_id.is_empty() {
            return Err(Status::invalid_argument(
                "acknowledging needs a consumer_id",
            ));
        }
        if req.cursor.is_empty() {
            return Err(Status::invalid_argument("acknowledging needs a cursor"));
        }
        let cursor = ChangeCursor::from_bytes(&req.cursor)
            .map_err(|e| Status::invalid_argument(format!("cursor: {e}")))?;
        let history = self
            .engine
            .cut_history()
            .map_err(|e| Status::internal(format!("history: {e}")))?;
        if history != Some(cursor.history()) {
            return Err(Status::invalid_argument(
                CursorError::OtherHistory.to_string(),
            ));
        }
        let index = cursor.position();
        // Nothing past what this node applied was sent, so a cursor beyond
        // it is not a statement about held units and releases nothing.
        let applied = (self.applied)();
        if index > applied {
            return Err(Status::invalid_argument(format!(
                "cursor position {index} is past the last entry this node applied ({applied})"
            )));
        }
        let registry = self.registry.clone();
        let handle = RegisteredHandle::new(req.consumer_id, req.incarnation);
        // Monotonic: an earlier position than one already acknowledged keeps
        // the later one.
        blocking(move || registry.checkpoint(&handle, index)).await
    }
}

/// Everything one stream's task owns.
struct StreamState {
    engine: Arc<StorageEngine>,
    registry: ShardConsumerRegistry,
    handle: RegisteredHandle,
    /// Tells when a write to the registration applied, the only time it
    /// can have ended.
    watch: RegistrationWatch,
    /// Where the stream reads next.
    cursor: ChangeCursor,
    /// The shard's shared reader of the log.
    hub: Arc<CdcHub>,
    /// The stream's registration with the hub; out of the state only while a
    /// read holds it.
    reader: Option<HubReader>,
    /// Reads the log while the stream is behind what the hub holds.
    own_tailer: Option<OplogTailer>,
    /// Where the batches go.
    delivery: Delivery,
    applied: AppliedFrontier,
    applied_changes: AppliedSignal,
    tuning: CdcStreamTuning,
}

/// Read the log for one stream until the client leaves, the registration
/// ends, or the log no longer holds what the stream needs. Leaving does not
/// end the registration: the client resumes it.
async fn stream_consumer(mut s: StreamState) {
    // Counted while this task runs, whichever way it ends.
    let _open = OpenStream::count(&s.delivery);
    let mut names = LabelNames::default();
    loop {
        // Client cancelled (channel closed).
        if s.delivery.is_closed() {
            break;
        }
        // Marked seen before the read: an entry applied after it wakes the
        // wait below.
        if let Some(changes) = s.applied_changes.as_mut() {
            changes.borrow_and_update();
        }

        // An ended registration is a clean refusal rather than a silent gap.
        // It ends only through a write to its record, so the record is read
        // again only after one applied; history the log no longer holds is
        // refused by the read itself.
        if s.watch.changed() {
            let check = {
                let (registry, handle) = (s.registry.clone(), s.handle.clone());
                blocking(move || registry.check_retention(&handle)).await
            };
            if let Err(status) = check {
                s.delivery.fail(status).await;
                break;
            }
        }

        // What the client takes now: a session subscription waits here for
        // credit.
        let Some(room) = s
            .delivery
            .room(
                &s.registry,
                &s.handle,
                s.tuning.heartbeat_interval,
                s.tuning.batch_size.get(),
            )
            .await
        else {
            break;
        };

        let read_from = s.cursor.position();
        let until = (s.applied)();
        let Some(reader) = s.reader.take() else {
            break;
        };
        let read = {
            let engine = Arc::clone(&s.engine);
            let hub = Arc::clone(&s.hub);
            let cursor = s.cursor.clone();
            let mut tailer = s.own_tailer.take();
            let mut taken = std::mem::take(&mut names);
            let read_bytes = s.tuning.buffer_bytes as u64;
            tokio::task::spawn_blocking(move || {
                let read = read_entries(&hub, &reader, &mut tailer, read_from, room, until)
                    .and_then(|(entries, next)| {
                        // The events one read holds: a unit larger than this
                        // goes out in fragments, the cursor continuing inside
                        // it.
                        let budget = QueryBudget::new(read_bytes);
                        let (batch, _events) = engine.changes_of_entries(
                            &cursor,
                            entries.iter().map(|e| &**e),
                            next,
                            &budget,
                        )?;
                        hub.passed(&reader, batch.cursor.position());
                        let proto = batch_to_proto(&engine, &batch, &mut taken)?;
                        Ok((proto, batch.cursor))
                    });
                (read, taken, reader, tailer)
            })
            .await
        };
        let (read, kept) = match read {
            Ok((read, kept, reader, tailer)) => {
                s.reader = Some(reader);
                s.own_tailer = tailer;
                (read, kept)
            }
            Err(e) => {
                s.delivery
                    .fail(Status::internal(format!("change stream read: {e}")))
                    .await;
                break;
            }
        };
        names = kept;
        let (batch, next) = match read {
            Ok(read) => read,
            Err(e) => {
                s.delivery.fail(read_status(e, read_from)).await;
                break;
            }
        };
        let read_to = next.position();
        // A read stopped inside a transaction moves the cursor inside its
        // entry, not past it.
        let caught_up = next == s.cursor;
        // Every entry read uses credit, whether or not it carried a unit; a
        // fragment of one uses one.
        let entries = (read_to - read_from).max(u64::from(!caught_up));
        // Applied entries left past the read, or the rest of a transaction:
        // the room, not the log, ended it.
        let more = read_to < until || !next.at_entry_start();
        s.cursor = next;
        // Sending is not delivery: the registration moves only when the
        // client acknowledges. A batch whose entries carried no unit still
        // moves the cursor, so a client whose interest matches little can
        // acknowledge past them.
        if !caught_up
            && s.delivery
                .send(
                    batch,
                    entries,
                    more,
                    &s.registry,
                    &s.handle,
                    s.tuning.heartbeat_interval,
                )
                .await
                .is_err()
        {
            break;
        }

        if caught_up {
            // Heartbeat so the registration shows a connected reader (a
            // diagnostic: it ends and extends nothing), then sleep until an
            // entry applies, a write to the registration applies, the
            // client leaves, or the next heartbeat is due.
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
                    () = s.watch.wait() => {}
                    () = s.delivery.closed() => break,
                    () = heartbeat => {}
                },
                None => tokio::select! {
                    () = s.watch.wait() => {}
                    () = s.delivery.closed() => break,
                    () = heartbeat => {}
                },
            }
        }
    }
}

/// The next entries from `position` below `until`, at most `max`, as stored:
/// from the shard's hub, or from the log itself while the stream is behind
/// what the hub holds. Returns them and the position after the last one.
fn read_entries(
    hub: &CdcHub,
    reader: &HubReader,
    own_tailer: &mut Option<OplogTailer>,
    position: u64,
    max: usize,
    until: u64,
) -> Result<(Vec<SharedEntry>, u64), StorageError> {
    match hub.read(reader, position, max, until)? {
        HubRead::Entries { entries, next } => {
            *own_tailer = None;
            Ok((entries, next))
        }
        HubRead::Behind(base) => {
            // A tailer that has passed `position` is not reused: a read that
            // stopped inside an entry reads that entry again.
            let tailer = match own_tailer {
                Some(tailer) if tailer.next_index() == position => tailer,
                _ => own_tailer.insert(hub.tailer_at(position)?),
            };
            let entries = tailer.read_next(max, &CdcFilters::default(), until.min(base))?;
            let next = tailer.next_index();
            Ok((
                entries.into_iter().map(|(e, _)| Arc::new(e)).collect(),
                next,
            ))
        }
    }
}

/// One open change stream in `coordinode_subscription_active`, labelled by
/// how its batches are delivered; released when the stream's task ends.
struct OpenStream(&'static str);

impl OpenStream {
    fn count(delivery: &Delivery) -> Self {
        let transport = match delivery {
            Delivery::Stream { .. } => "stream",
            Delivery::Session { .. } => "session",
        };
        metrics::gauge!("coordinode_subscription_active", "transport" => transport).increment(1.0);
        Self(transport)
    }
}

impl Drop for OpenStream {
    fn drop(&mut self) {
        metrics::gauge!("coordinode_subscription_active", "transport" => self.0).decrement(1.0);
    }
}

/// Label names a stream has resolved, kept for its later batches: an
/// identity's name never changes, it is recorded once.
#[derive(Default)]
struct LabelNames(FxHashMap<LabelId, String>);

impl LabelNames {
    fn names(
        &mut self,
        engine: &StorageEngine,
        labels: &[LabelId],
    ) -> Result<Vec<String>, StorageError> {
        let mut out = Vec::with_capacity(labels.len());
        for &label in labels {
            if let Some(name) = self.0.get(&label) {
                out.push(name.clone());
                continue;
            }
            let name = LocalSchemaStore::new(engine)
                .label_name(label)
                .map_err(|e| StorageError::Serialization(format!("label {label}: {e}")))?
                .ok_or_else(|| {
                    StorageError::Serialization(format!("label {label} has no identity record"))
                })?;
            self.0.insert(label, name.clone());
            out.push(name);
        }
        Ok(out)
    }
}

/// The batch a client receives for `batch`.
fn batch_to_proto(
    engine: &StorageEngine,
    batch: &feed::ChangeBatch,
    names: &mut LabelNames,
) -> Result<ChangeBatch, StorageError> {
    let mut units = Vec::with_capacity(batch.units.len());
    for unit in &batch.units {
        let mut events = Vec::with_capacity(unit.events.len());
        for event in &unit.events {
            events.push(event_to_proto(engine, event, names)?);
        }
        units.push(ChangeUnit {
            commit_ts: unit.commit_ts,
            log_index: unit.position.index,
            unit: unit.position.sub,
            projected: unit.projected,
            events,
            first_ordinal: unit.first,
            end_ordinal: unit.through,
        });
    }
    Ok(ChangeBatch {
        units,
        cursor: batch.cursor.to_bytes(),
    })
}

fn event_to_proto(
    engine: &StorageEngine,
    event: &feed::ChangeEvent,
    names: &mut LabelNames,
) -> Result<LogicalChange, StorageError> {
    let malformed = |what: &str| StorageError::Serialization(format!("a change event's {what}"));
    let relationship = |key: &[u8]| -> Result<RelationshipTarget, StorageError> {
        let parts = feed::edgeprop_parts(key).ok_or_else(|| malformed("edge property key"))?;
        Ok(RelationshipTarget {
            edge_type: String::from_utf8_lossy(parts.edge_type).into_owned(),
            source_id: parts.source,
            target_id: parts.target,
            instance: parts.instance.map(<[u8]>::to_vec).unwrap_or_default(),
        })
    };
    let target = match &event.target {
        OwnedTarget::Node(key) => {
            let node = if let Some((_, id)) = coordinode_core::graph::node::decode_node_key(key) {
                NodeTarget {
                    node_id: id.as_raw(),
                    valid_from: None,
                }
            } else {
                let (_, id, valid_from) =
                    coordinode_core::graph::node::decode_temporal_node_key(key)
                        .ok_or_else(|| malformed("node key"))?;
                NodeTarget {
                    node_id: id.as_raw(),
                    valid_from: Some(valid_from),
                }
            };
            logical_change::Target::Node(node)
        }
        OwnedTarget::Relationship {
            edge_type,
            source,
            target,
        } => logical_change::Target::Relationship(RelationshipTarget {
            edge_type: String::from_utf8_lossy(edge_type).into_owned(),
            source_id: *source,
            target_id: *target,
            instance: Vec::new(),
        }),
        OwnedTarget::RelationshipProperties(key) => {
            logical_change::Target::Properties(relationship(key)?)
        }
        OwnedTarget::RelationshipInstance(key) => {
            logical_change::Target::Instance(relationship(key)?)
        }
        OwnedTarget::Scope { label, activation } => {
            let label = names
                .names(engine, &[*label])?
                .pop()
                .ok_or_else(|| malformed("scope label"))?;
            logical_change::Target::Scope(ScopeTarget {
                label,
                activation: *activation,
            })
        }
        OwnedTarget::BlobOwner(key) => logical_change::Target::BlobOwner(key.clone()),
    };
    let (labels_before, labels_after) = match &event.labels {
        Some((before, after)) => (names.names(engine, before)?, names.names(engine, after)?),
        None => (Vec::new(), Vec::new()),
    };
    Ok(LogicalChange {
        ordinal: event.id.ordinal,
        action: match event.action {
            feed::Action::Put => ChangeAction::Put,
            feed::Action::Patch => ChangeAction::Patch,
            feed::Action::Remove => ChangeAction::Remove,
            feed::Action::ScopeChange => ChangeAction::ScopeChange,
        } as i32,
        target: Some(target),
        payload: event.payload.clone().unwrap_or_default(),
        delivery: match event.delivery {
            feed::Delivery::Apply | feed::Delivery::None => ChangeDelivery::Apply,
            feed::Delivery::Enter => ChangeDelivery::Enter,
            feed::Delivery::Leave => ChangeDelivery::Leave,
        } as i32,
        labels_before,
        labels_after,
    })
}

mod delivery;
mod hub;
pub(crate) use delivery::{Credit, Delivery, session_error};
use hub::{CdcHub, HubRead, HubReader, SharedEntry};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
