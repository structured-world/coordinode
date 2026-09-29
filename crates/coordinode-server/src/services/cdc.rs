//! gRPC ChangeStreamService: CE oplog CDC consumer.
//!
//! Tails the Raft log's oplog segments and streams [`ChangeEvent`] messages
//! to the client, only for entries this node has applied: the log also holds
//! entries that are not committed yet, which a later leader may truncate and
//! replace. Polls every 100ms when caught up.
//!
//! In embedded mode (no Raft) nothing is applied from a Raft log and the
//! stream is empty — no error.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use coordinode_raft::cluster::RaftNode;
use coordinode_raft::storage::raft_oplog_dirs;
use coordinode_replicate::{
    ConsumerKind, ConsumerRegistration, InitialSeqno, SeqnoConsumerRegistry, ShardConsumerRegistry,
    TopologyScope,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::oplog::entry::OplogOp;
use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer, ResumeToken};

use crate::proto::replication::cdc::{
    CdcFilters as ProtoCdcFilters, ChangeEvent, ChangeOp, ChangeOpType,
    ResumeToken as ProtoResumeToken, SubscribeRequest,
    change_stream_service_server::ChangeStreamService,
};

/// One past the last Raft log entry this node has applied, `0` when none:
/// the exclusive bound of what a change stream may send.
pub type AppliedFrontier = Arc<dyn Fn() -> u64 + Send + Sync>;

/// gRPC CDC service for one shard.
///
/// Each subscription registers as an `oplog_events` consumer in the
/// [`ShardConsumerRegistry`] so the oplog retention floor never purges below
/// a live reader's position.
pub struct ChangeEventServiceImpl {
    /// The shard whose Raft log the service streams.
    shard_id: u32,
    /// Every directory holding segments of that log.
    oplog_dirs: Vec<PathBuf>,
    registry: ShardConsumerRegistry,
    /// Per-process counter for unique CDC consumer ids.
    next_consumer: Arc<AtomicU64>,
    /// TTL (ms) applied to each CDC consumer registration
    /// (`--cdc-consumer-ttl-secs`). A crashed reader is reclaimed after this.
    consumer_ttl_ms: u64,
    /// Bound of the entries a stream may send.
    applied: AppliedFrontier,
    /// Read pacing of every stream.
    tuning: CdcStreamTuning,
}

impl ChangeEventServiceImpl {
    /// A service streaming shard `shard_id`'s Raft log from `oplog_dirs`
    /// (see `coordinode_raft::storage::raft_oplog_dirs`).
    pub fn new(
        shard_id: u32,
        oplog_dirs: Vec<PathBuf>,
        registry: ShardConsumerRegistry,
        consumer_ttl_ms: u64,
        applied: AppliedFrontier,
    ) -> Self {
        Self {
            shard_id,
            oplog_dirs,
            registry,
            next_consumer: Arc::new(AtomicU64::new(0)),
            consumer_ttl_ms,
            applied,
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
        consumer_ttl_ms: u64,
    ) -> std::io::Result<Self> {
        let dirs = raft_oplog_dirs(engine, 0)?.all;
        Ok(Self::new(
            0,
            dirs,
            registry,
            consumer_ttl_ms,
            Arc::new(move || node.applied_through()),
        ))
    }
}

/// Default TTL for a CDC consumer registration (`cdc_consumer_ttl_secs`,
/// 30s). The stream heartbeats every poll ([`CdcStreamTuning::poll_interval`]),
/// so a connected-but-idle reader is never evicted; a reader that vanishes
/// without unregistering (crash) is reclaimed after this.
pub const DEFAULT_CONSUMER_TTL_MS: u64 = 30_000;

/// How a change stream paces its reads (`cdc_poll_interval_ms`,
/// `cdc_batch_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdcStreamTuning {
    /// Wait between polls once caught up to the last applied entry: the
    /// delivery latency of an idle stream, and its heartbeat period.
    pub poll_interval: Duration,
    /// Most entries read and sent per poll (back-pressure).
    pub batch_size: NonZeroUsize,
}

impl Default for CdcStreamTuning {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(100),
            batch_size: const {
                match NonZeroUsize::new(256) {
                    Some(n) => n,
                    None => unreachable!(),
                }
            },
        }
    }
}

impl CdcStreamTuning {
    /// Check the tuning against the consumer TTL.
    ///
    /// # Errors
    ///
    /// The poll interval is not shorter than `consumer_ttl_ms`: an idle stream
    /// heartbeats once per poll, so its registration would expire and release
    /// the oplog it still reads.
    pub fn check(&self, consumer_ttl_ms: u64) -> Result<(), String> {
        if self.poll_interval >= Duration::from_millis(consumer_ttl_ms) {
            return Err(format!(
                "cdc_poll_interval_ms ({} ms) must be shorter than cdc_consumer_ttl_secs \
                 ({consumer_ttl_ms} ms): an idle change stream heartbeats once per poll",
                self.poll_interval.as_millis()
            ));
        }
        Ok(())
    }
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

        // Parse resume token.
        let token = match req.resume_token {
            Some(t) => ResumeToken {
                shard_id: t.shard_id,
                segment_id: t.segment_id,
                entry_offset: t.entry_offset,
            },
            None => ResumeToken::from_start(self.shard_id),
        };

        let shard_id = self.shard_id;
        if token.shard_id != shard_id {
            return Err(Status::invalid_argument(format!(
                "resume token is for shard {}, this stream serves shard {shard_id}",
                token.shard_id
            )));
        }
        let filters = proto_filters_to_cdc(req.filters);
        let mut tailer = OplogTailer::new(&self.oplog_dirs, token)
            .map_err(|e| Status::invalid_argument(format!("resume token: {e}")))?;

        // Register this stream as an oplog-events consumer so the oplog
        // retention floor is held at (and advanced with) its read position.
        // `FromEarliestRetained` pins conservatively until the first
        // checkpoint advances the floor to what the consumer has actually read.
        let n = self.next_consumer.fetch_add(1, Ordering::Relaxed);
        let consumer_id = format!("cdc-{shard_id}-{n}");
        let handle = self
            .registry
            .register(ConsumerRegistration {
                consumer_id,
                kind: ConsumerKind::OplogEvents,
                scope: TopologyScope::Shard(shard_id as u16),
                initial_seqno: InitialSeqno::FromEarliestRetained,
                ttl_ms: self.consumer_ttl_ms,
            })
            .map_err(|e| Status::internal(format!("register cdc consumer: {e}")))?;

        // Spawn a task that tails the oplog and sends events into a channel.
        let (tx, rx) = mpsc::channel::<Result<ChangeEvent, Status>>(64);
        let registry = self.registry.clone();
        let applied = Arc::clone(&self.applied);
        let tuning = self.tuning;
        tokio::spawn(async move {
            'stream: loop {
                // Client cancelled (channel closed).
                if tx.is_closed() {
                    break;
                }

                // Surface retention loss as a clean error rather than a silent
                // gap (the consumer's checkpoint fell behind the GC floor).
                if let Err(e) = registry.check_retention(&handle) {
                    let _ = tx
                        .send(Err(Status::failed_precondition(format!(
                            "change stream retention lost: {e}"
                        ))))
                        .await;
                    break;
                }

                let read_from = tailer.next_index();
                let batch = match tailer.read_next(tuning.batch_size.get(), &filters, applied()) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                        break;
                    }
                };
                let caught_up = batch.is_empty();

                for (entry, token) in batch {
                    // A slow reader leaves no room in the channel; keep its
                    // registration alive while waiting, as an idle poll does.
                    let permit = loop {
                        match tokio::time::timeout(tuning.poll_interval, tx.reserve()).await {
                            Ok(Ok(permit)) => break permit,
                            // Client disconnected mid-batch.
                            Ok(Err(_)) => break 'stream,
                            Err(_) => {
                                if let Err(e) = registry.heartbeat(&handle) {
                                    tracing::warn!(error = %e, "change stream heartbeat failed");
                                }
                            }
                        }
                    };
                    permit.send(Ok(oplog_entry_to_proto(entry, token)));
                }
                // Advance the retention floor past everything read, the
                // entries the filters dropped included: a stream whose filters
                // match nothing must not hold the oplog forever.
                let read_to = tailer.next_index();
                if read_to > read_from {
                    if let Err(e) = registry.checkpoint(&handle, read_to - 1) {
                        tracing::warn!(error = %e, "change stream checkpoint failed");
                    }
                }

                if caught_up {
                    // Heartbeat so an idle-but-connected reader is not
                    // TTL-evicted, then wait for new applied entries.
                    if let Err(e) = registry.heartbeat(&handle) {
                        tracing::warn!(error = %e, "change stream heartbeat failed");
                    }
                    tokio::time::sleep(tuning.poll_interval).await;
                }
            }

            // Release the retention hold when the stream ends for any reason.
            if let Err(e) = registry.unregister(handle) {
                tracing::warn!(error = %e, "change stream consumer unregister failed");
            }
        });

        let stream: Pin<Box<dyn tokio_stream::Stream<Item = Result<ChangeEvent, Status>> + Send>> =
            Box::pin(ReceiverStream::new(rx));
        Ok(Response::new(stream))
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
