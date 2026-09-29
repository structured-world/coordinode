//! gRPC server handler for Raft inter-node RPCs.
//!
//! Implements `RaftService` tonic trait. Dispatches incoming RPCs to the
//! local openraft instance. Uses msgpack for type serialization.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::proto::replication::raft_service_server::RaftService;
use crate::proto::replication::{RaftEmpty, RaftPayload};
use crate::storage::{CoordinodeStateMachine, TypeConfig};

type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// gRPC server handler for Raft consensus RPCs.
pub struct RaftGrpcHandler {
    raft: Arc<RaftInstance>,
    /// The node's snapshot directory, where a received snapshot is staged so
    /// the install publishes it in place.
    snapshot_dir: PathBuf,
}

impl RaftGrpcHandler {
    /// A handler for `raft`, staging received snapshots in `snapshot_dir`
    /// (see [`crate::snapshot::snapshot_dir`]).
    pub fn new(raft: Arc<RaftInstance>, snapshot_dir: PathBuf) -> Self {
        Self { raft, snapshot_dir }
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

        // openraft ends the reply stream without a word once its consensus
        // has stopped: the request it could not deliver goes unanswered. A
        // leader reading a cleanly ended stream keeps waiting on that request
        // and neither retries nor falls back to a snapshot, so such a stream
        // ends with an error that makes the leader reconnect. The stop is
        // already in the metrics by then: the core publishes it before it
        // drops the queue whose closing ended the stream.
        let raft = Arc::clone(&self.raft);
        let stopped = futures_util::stream::once(async move {
            use openraft::async_runtime::watch::WatchReceiver;
            raft.metrics().borrow_watched().running_state.clone().err()
        })
        .filter_map(|fatal| async move {
            fatal.map(|fatal| Err(Status::unavailable(format!("raft stopped: {fatal}"))))
        });

        Ok(Response::new(Box::pin(output_stream.chain(stopped))))
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
        // Data chunks go into a staged file in the snapshot directory,
        // which the install reads and then publishes in place.

        // Read first message — must be Header
        let first = stream
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("empty snapshot stream"))?
            .map_err(|e| Status::internal(format!("snapshot stream error: {e}")))?;

        let first_msg: crate::snapshot::SnapshotChunkMessage = rmp_serde::from_slice(&first.data)
            .map_err(|e| {
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

        let expected_data_size = header.data_size;

        tracing::info!(
            data_size = expected_data_size,
            last_log_index = header.meta.last_log_id.map(|id| id.index),
            "receiving chunked snapshot from leader"
        );

        // A staged file is removed if the transfer or the install fails.
        let mut staged = crate::snapshot::SnapshotFile::stage(&self.snapshot_dir)
            .map_err(|e| Status::internal(format!("stage the snapshot file: {e}")))?;
        let mut writer = tokio::fs::File::from_std(
            staged
                .try_clone_file()
                .map_err(|e| Status::internal(format!("open the snapshot file: {e}")))?,
        );
        let mut received_bytes = 0u64;
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
                    use tokio::io::AsyncWriteExt;
                    writer
                        .write_all(&data)
                        .await
                        .map_err(|e| Status::internal(format!("write snapshot chunk: {e}")))?;
                    received_bytes += data.len() as u64;
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

        {
            use tokio::io::AsyncWriteExt;
            // The writes complete on tokio's blocking pool; the install reads
            // the file through another handle.
            writer
                .flush()
                .await
                .map_err(|e| Status::internal(format!("write snapshot chunk: {e}")))?;
        }
        drop(writer);

        tracing::info!(
            received_bytes,
            chunk_count,
            "snapshot chunks received, installing"
        );

        // openraft passes the file to the state machine's install_snapshot.
        let snapshot = openraft::storage::Snapshot {
            meta: header.meta.clone(),
            snapshot: staged,
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

        // alpha.25 splits the result: outer = Fatal (engine error), inner =
        // TransferLeaderError (the transfer was rejected, e.g. not leader). Both
        // surface to the caller as a gRPC Status so the client's network layer
        // maps them to an RPCError.
        self.raft
            .handle_transfer_leader(req)
            .await
            .map_err(|e| Status::internal(format!("transfer_leader: {e}")))?
            .map_err(|e| Status::internal(format!("transfer_leader rejected: {e}")))?;

        Ok(Response::new(RaftEmpty {}))
    }
}
