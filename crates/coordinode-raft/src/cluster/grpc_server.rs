//! gRPC server handler for Raft inter-node RPCs.
//!
//! Implements `RaftService` tonic trait. Dispatches incoming RPCs to the
//! local openraft instance. Uses msgpack for type serialization.

use std::pin::Pin;
use std::sync::{Arc, Weak};

use futures_util::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use coordinode_storage::engine::core::StorageEngine;

use crate::proto::replication::raft_service_server::RaftService;
use crate::proto::replication::{RaftEmpty, RaftPayload};
use crate::storage::{CoordinodeStateMachine, TypeConfig};

type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// gRPC server handler for Raft consensus RPCs.
pub struct RaftGrpcHandler {
    raft: Arc<RaftInstance>,
    /// Weak ref to engine for incremental snapshots.
    /// Weak avoids preventing engine drop during node shutdown.
    engine: Weak<StorageEngine>,
}

impl RaftGrpcHandler {
    pub fn new(raft: Arc<RaftInstance>, engine: Arc<StorageEngine>) -> Self {
        Self {
            raft,
            engine: Arc::downgrade(&engine),
        }
    }
}

fn ser<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, Status> {
    rmp_serde::to_vec(value).map_err(|e| Status::internal(format!("msgpack serialize: {e}")))
}

fn de<T: serde::de::DeserializeOwned>(data: &[u8]) -> Result<T, Status> {
    rmp_serde::from_slice(data)
        .map_err(|e| Status::invalid_argument(format!("msgpack deserialize: {e}")))
}

#[tonic::async_trait]
impl RaftService for RaftGrpcHandler {
    async fn vote(&self, request: Request<RaftPayload>) -> Result<Response<RaftPayload>, Status> {
        let vote_req: openraft::raft::VoteRequest<TypeConfig> = de(&request.into_inner().data)?;

        let vote_resp = self
            .raft
            .vote(vote_req)
            .await
            .map_err(|e| Status::internal(format!("vote: {e}")))?;

        Ok(Response::new(RaftPayload {
            data: ser(&vote_resp)?,
        }))
    }

    async fn append_entries(
        &self,
        request: Request<RaftPayload>,
    ) -> Result<Response<RaftPayload>, Status> {
        let req: openraft::raft::AppendEntriesRequest<TypeConfig> = de(&request.into_inner().data)?;

        let resp = self
            .raft
            .append_entries(req)
            .await
            .map_err(|e| Status::internal(format!("append_entries: {e}")))?;

        Ok(Response::new(RaftPayload { data: ser(&resp)? }))
    }

    type StreamAppendStream = Pin<Box<dyn Stream<Item = Result<RaftPayload, Status>> + Send>>;

    async fn stream_append(
        &self,
        request: Request<Streaming<RaftPayload>>,
    ) -> Result<Response<Self::StreamAppendStream>, Status> {
        let input = request.into_inner();

        // Deserialize incoming RaftPayload stream → AppendEntriesRequest stream
        let input_stream = input.filter_map(|result| async move {
            match result {
                Ok(payload) => match rmp_serde::from_slice(&payload.data) {
                    Ok(req) => Some(req),
                    Err(e) => {
                        tracing::warn!("stream_append deserialize error: {e}");
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!("stream_append receive error: {e}");
                    None
                }
            }
        });

        // Feed to openraft's stream_append — it handles everything
        let output = self.raft.stream_append(input_stream);

        // Serialize output stream: StreamAppendResult → RaftPayload
        let output_stream = output.map(|result| match result {
            Ok(stream_result) => {
                let data = rmp_serde::to_vec(&stream_result)
                    .map_err(|e| Status::internal(format!("serialize: {e}")))?;
                Ok(RaftPayload { data })
            }
            Err(fatal) => Err(Status::internal(format!("fatal: {fatal}"))),
        });

        Ok(Response::new(Box::pin(output_stream)))
    }

    async fn snapshot(
        &self,
        request: Request<Streaming<RaftPayload>>,
    ) -> Result<Response<RaftPayload>, Status> {
        let mut stream = request.into_inner();

        // ── Chunked snapshot protocol ──────────────────────────────
        // Message 1: SnapshotChunkMessage::Header (metadata)
        // Messages 2..N: SnapshotChunkMessage::DataChunk (CNSN bytes)
        //
        // Data chunks are written to a temp file to avoid OOM on large
        // snapshots. The file is then read by the installer.

        // Read first message — must be Header
        let first = stream
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("empty snapshot stream"))?
            .map_err(|e| Status::internal(format!("snapshot stream error: {e}")))?;

        let first_msg: crate::snapshot::SnapshotChunkMessage = rmp_serde::from_slice(&first.data)
            .map_err(|e| {
            // Backward compatibility: try deserializing as legacy SnapshotTransfer
            tracing::debug!("chunked header parse failed, trying legacy format: {e}");
            Status::invalid_argument(format!("snapshot header deserialize: {e}"))
        })?;

        let header = match first_msg {
            crate::snapshot::SnapshotChunkMessage::Header(h) => h,
            _ => {
                return Err(Status::invalid_argument(
                    "first snapshot message must be Header",
                ));
            }
        };

        let is_incremental = header.since_ts.is_some();
        let expected_data_size = header.data_size as usize;

        tracing::info!(
            snapshot_id = %header.meta.snapshot_id,
            data_size = expected_data_size,
            last_log_index = header.meta.last_log_id.map(|id| id.index),
            incremental = is_incremental,
            since_ts = ?header.since_ts,
            "receiving chunked snapshot from leader"
        );

        // Write data chunks to temp file
        let mut temp_file =
            tempfile::tempfile().map_err(|e| Status::internal(format!("create temp file: {e}")))?;
        let mut received_bytes = 0usize;
        let mut chunk_count = 0usize;

        while let Some(result) = stream.next().await {
            let payload =
                result.map_err(|e| Status::internal(format!("snapshot chunk receive: {e}")))?;

            let chunk_msg: crate::snapshot::SnapshotChunkMessage =
                rmp_serde::from_slice(&payload.data).map_err(|e| {
                    Status::invalid_argument(format!("snapshot chunk deserialize: {e}"))
                })?;

            match chunk_msg {
                crate::snapshot::SnapshotChunkMessage::DataChunk(data) => {
                    use std::io::Write;
                    temp_file
                        .write_all(&data)
                        .map_err(|e| Status::internal(format!("write snapshot chunk: {e}")))?;
                    received_bytes += data.len();
                    chunk_count += 1;
                }
                crate::snapshot::SnapshotChunkMessage::Header(_) => {
                    return Err(Status::invalid_argument(
                        "unexpected Header message after first message",
                    ));
                }
            }
        }

        if received_bytes != expected_data_size {
            return Err(Status::data_loss(format!(
                "snapshot data size mismatch: expected {expected_data_size}, got {received_bytes}"
            )));
        }

        tracing::info!(
            received_bytes,
            chunk_count,
            "snapshot chunks received, installing"
        );

        // Seek to start of temp file for reading
        use std::io::Seek;
        temp_file
            .seek(std::io::SeekFrom::Start(0))
            .map_err(|e| Status::internal(format!("seek temp file: {e}")))?;

        // Install snapshot from temp file
        let raft_data = if is_incremental {
            let engine = self.engine.upgrade().ok_or_else(|| {
                Status::unavailable("engine dropped during incremental snapshot install")
            })?;
            let mut reader = std::io::BufReader::new(temp_file);
            crate::snapshot::install_incremental_snapshot_from_reader(&engine, &mut reader)
                .map_err(|e| Status::internal(format!("incremental snapshot install: {e}")))?;
            // Empty data — state machine's install_snapshot will skip data apply
            Vec::new()
        } else {
            // For full snapshots, openraft needs the data for its state machine.
            // Read from temp file into memory — this is the data that openraft
            // will pass to install_snapshot() on the state machine.
            let mut data = Vec::with_capacity(expected_data_size);
            use std::io::Read;
            let mut reader = std::io::BufReader::new(temp_file);
            reader
                .read_to_end(&mut data)
                .map_err(|e| Status::internal(format!("read snapshot from temp file: {e}")))?;
            data
        };

        let snapshot_cursor = std::io::Cursor::new(raft_data);
        let snapshot = openraft::storage::Snapshot {
            meta: header.meta.clone(),
            snapshot: snapshot_cursor,
        };

        // install_full_snapshot returns SnapshotResponse with the
        // follower's current vote — NOT the leader's vote from the transfer.
        // This is important: the leader uses the response vote to detect
        // if the follower has seen a higher term (split-brain prevention).
        let response = self
            .raft
            .install_full_snapshot(header.vote, snapshot)
            .await
            .map_err(|e| Status::internal(format!("install_snapshot: {e}")))?;

        let resp_bytes = ser(&response)?;

        tracing::info!(chunk_count, "chunked snapshot installation complete");
        Ok(Response::new(RaftPayload { data: resp_bytes }))
    }

    async fn transfer_leader(
        &self,
        request: Request<RaftPayload>,
    ) -> Result<Response<RaftEmpty>, Status> {
        let req: openraft::raft::TransferLeaderRequest<TypeConfig> =
            de(&request.into_inner().data)?;

        self.raft
            .handle_transfer_leader(req)
            .await
            .map_err(|e| Status::internal(format!("transfer_leader: {e}")))?;

        Ok(Response::new(RaftEmpty {}))
    }
}
