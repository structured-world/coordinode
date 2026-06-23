//! gRPC ChangeStreamService: CE oplog CDC consumer.
//!
//! Tails sealed oplog segments from `data_dir/oplog/<shard_id>/`, streams
//! [`ChangeEvent`] messages to the client. Polls every 100ms when caught up.
//!
//! In embedded mode (no Raft, no oplog) the stream is empty — no error.
//! In Raft mode the oplog dir is populated by `LogStore::append`.

use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use coordinode_storage::oplog::entry::OplogOp;
use coordinode_storage::oplog::tailer::{CdcFilters, OplogTailer, ResumeToken};

use crate::proto::replication::cdc::{
    change_stream_service_server::ChangeStreamService, CdcFilters as ProtoCdcFilters, ChangeEvent,
    ChangeOp, ChangeOpType, ResumeToken as ProtoResumeToken, SubscribeRequest,
};

/// gRPC CDC service for one shard.
///
/// `data_dir` is the root data directory; the oplog lives at
/// `data_dir/oplog/<shard_id>/`.
pub struct ChangeEventServiceImpl {
    data_dir: PathBuf,
}

impl ChangeEventServiceImpl {
    pub fn new(data_dir: PathBuf) -> Self {
        Self { data_dir }
    }

    fn oplog_dir_for_shard(&self, _shard_id: u32) -> PathBuf {
        // LogStore stores the Raft oplog at `data_dir/raft_oplog/` (shard 0).
        // Multi-shard support (Phase 3 EE) will map shard_id to separate dirs.
        self.data_dir.join("raft_oplog")
    }
}

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

        // Spawn a task that tails the oplog and sends events into a channel.
        let (tx, rx) = mpsc::channel::<Result<ChangeEvent, Status>>(64);
        tokio::spawn(async move {
            let mut tailer = OplogTailer::new(&oplog_dir, token);

            loop {
                // Check if the client cancelled (channel closed).
                if tx.is_closed() {
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
                    // Caught up — wait for new sealed segments.
                    tokio::time::sleep(POLL_INTERVAL).await;
                    continue;
                }

                for (entry, token) in batch {
                    let event = oplog_entry_to_proto(entry, token);
                    if tx.send(Ok(event)).await.is_err() {
                        // Client disconnected.
                        return;
                    }
                }
            }
        });

        let stream: Pin<Box<dyn tokio_stream::Stream<Item = Result<ChangeEvent, Status>> + Send>> =
            Box::pin(ReceiverStream::new(rx));
        Ok(Response::new(stream))
    }
}

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
