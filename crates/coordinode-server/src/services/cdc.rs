//! gRPC ChangeStreamService: CE oplog CDC consumer.
//!
//! Tails sealed oplog segments from `data_dir/oplog/<shard_id>/`, streams
//! [`ChangeEvent`] messages to the client. Polls every 100ms when caught up.
//!
//! In embedded mode (no Raft, no oplog) the stream is empty — no error.
//! In Raft mode the oplog dir is populated by `LogStore::append`.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use coordinode_replicate::{
    ConsumerKind, ConsumerRegistration, InitialSeqno, SeqnoConsumerRegistry, ShardConsumerRegistry,
    TopologyScope,
};
use coordinode_storage::oplog::entry::OplogOp;
use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer, ResumeToken};

use crate::proto::replication::cdc::{
    change_stream_service_server::ChangeStreamService, CdcFilters as ProtoCdcFilters, ChangeEvent,
    ChangeOp, ChangeOpType, ResumeToken as ProtoResumeToken, SubscribeRequest,
};

/// gRPC CDC service for one shard.
///
/// `data_dir` is the root data directory; the oplog lives at
/// `data_dir/oplog/<shard_id>/`. Each subscription registers as an
/// `oplog_events` consumer in the [`ShardConsumerRegistry`] so the oplog
/// retention floor never purges below a live reader's position (ADR-028).
pub struct ChangeEventServiceImpl {
    data_dir: PathBuf,
    registry: ShardConsumerRegistry,
    /// Per-process counter for unique CDC consumer ids.
    next_consumer: Arc<AtomicU64>,
    /// TTL (ms) applied to each CDC consumer registration
    /// (`--cdc-consumer-ttl-secs`). A crashed reader is reclaimed after this.
    consumer_ttl_ms: u64,
}

impl ChangeEventServiceImpl {
    pub fn new(data_dir: PathBuf, registry: ShardConsumerRegistry, consumer_ttl_ms: u64) -> Self {
        Self {
            data_dir,
            registry,
            next_consumer: Arc::new(AtomicU64::new(0)),
            consumer_ttl_ms,
        }
    }

    fn oplog_dir_for_shard(&self, _shard_id: u32) -> PathBuf {
        // LogStore stores the Raft oplog at `data_dir/raft_oplog/` (shard 0).
        // Multi-shard support (Phase 3 EE) will map shard_id to separate dirs.
        self.data_dir.join("raft_oplog")
    }
}

/// Default TTL for a CDC consumer registration (`--cdc-consumer-ttl-secs`,
/// 30s). The stream heartbeats every poll (`POLL_INTERVAL`), so a
/// connected-but-idle reader is never evicted; a reader that vanishes without
/// unregistering (crash) is reclaimed after this.
pub const DEFAULT_CONSUMER_TTL_MS: u64 = 30_000;

/// How long to sleep between polls when caught up to the last sealed segment.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Maximum entries streamed per poll iteration (back-pressure).
const BATCH_SIZE: usize = 256;

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
            None => ResumeToken::from_start(0),
        };

        let shard_id = token.shard_id;
        let filters = proto_filters_to_cdc(req.filters);
        let oplog_dir = self.oplog_dir_for_shard(shard_id);

        // Register this stream as an oplog-events consumer so the oplog
        // retention floor is held at (and advanced with) its read position
        // (ADR-028). `FromEarliestRetained` pins conservatively until the first
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
        tokio::spawn(async move {
            let mut tailer = OplogTailer::new(&oplog_dir, token);

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

                let batch = match tailer.read_next(BATCH_SIZE, &filters) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                        break;
                    }
                };

                if batch.is_empty() {
                    // Caught up — heartbeat so an idle-but-connected reader is not
                    // TTL-evicted, then wait for new sealed segments.
                    let _ = registry.heartbeat(&handle);
                    tokio::time::sleep(POLL_INTERVAL).await;
                    continue;
                }

                let mut last_index = 0u64;
                for (entry, token) in batch {
                    last_index = entry.index;
                    let event = oplog_entry_to_proto(entry, token);
                    if tx.send(Ok(event)).await.is_err() {
                        // Client disconnected mid-batch.
                        break 'stream;
                    }
                }
                // Advance the retention floor to what the consumer has now read.
                let _ = registry.checkpoint(&handle, last_index);
            }

            // Release the retention hold when the stream ends for any reason.
            let _ = registry.unregister(handle);
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
    let ops = entry.ops.into_iter().map(oplog_op_to_proto).collect();

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

fn oplog_op_to_proto(op: OplogOp) -> ChangeOp {
    match op {
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
    }
}
