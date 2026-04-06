//! gRPC network implementation for openraft inter-node communication.
//!
//! Implements all 5 openraft network traits over tonic gRPC.
//! Uses opaque bytes (msgpack serialization) for all openraft types.

use std::future::Future;
use std::time::Duration;

use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use openraft::error::{RPCError, Unreachable};
use openraft::network::{
    Backoff, NetBackoff, NetSnapshot, NetStreamAppend, NetTransferLeader, NetVote, RPCOption,
};
use openraft::raft::StreamAppendResult;
use openraft::{OptionalSend, RaftNetworkFactory};

use crate::proto::replication::raft_service_client::RaftServiceClient;
use crate::proto::replication::RaftPayload;
use crate::storage::TypeConfig;

type C = TypeConfig;

// ── Serialization helpers ──────────────────────────────────────────

fn serialize<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, RPCError<C>> {
    rmp_serde::to_vec(value).map_err(|e| {
        RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("msgpack serialize error: {e}"),
        )))
    })
}

fn deserialize<T: serde::de::DeserializeOwned>(data: &[u8]) -> Result<T, RPCError<C>> {
    rmp_serde::from_slice(data).map_err(|e| {
        RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("msgpack deserialize error: {e}"),
        )))
    })
}

fn tonic_to_rpc_error(status: tonic::Status) -> RPCError<C> {
    RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
        std::io::ErrorKind::ConnectionAborted,
        format!("gRPC error: {status}"),
    )))
}

// ── Network Factory ────────────────────────────────────────────────

/// gRPC-based network factory for multi-node cluster.
pub struct GrpcNetworkFactory;

impl RaftNetworkFactory<C> for GrpcNetworkFactory {
    type Network = GrpcNetwork;

    async fn new_client(
        &mut self,
        _target: u64,
        node: &openraft::impls::BasicNode,
    ) -> Self::Network {
        GrpcNetwork {
            addr: node.addr.clone(),
            client: None,
        }
    }
}

/// Stub network factory for single-node testing without gRPC.
pub struct StubNetworkFactory;

impl RaftNetworkFactory<C> for StubNetworkFactory {
    type Network = StubNetwork;

    async fn new_client(
        &mut self,
        _target: u64,
        _node: &openraft::impls::BasicNode,
    ) -> Self::Network {
        StubNetwork
    }
}

// ── gRPC Network ───────────────────────────────────────────────────

/// gRPC connection to a single Raft peer. Lazily connects on first use.
pub struct GrpcNetwork {
    addr: String,
    client: Option<RaftServiceClient<tonic::transport::Channel>>,
}

impl GrpcNetwork {
    /// Get or create the gRPC client for this peer.
    ///
    /// Uses `connect_lazy()` which creates a Channel with built-in
    /// reconnection. If the peer is temporarily down, the Channel
    /// automatically reconnects on the next RPC attempt. Combined with
    /// openraft's `NetBackoff` (200ms infinite retry), this provides
    /// transparent recovery from network partitions and node restarts.
    ///
    /// Previous issue (G042): `connect().await` created a one-shot
    /// connection. If it dropped, all subsequent RPCs failed permanently.
    async fn get_client(
        &mut self,
    ) -> Result<&mut RaftServiceClient<tonic::transport::Channel>, RPCError<C>> {
        if self.client.is_none() {
            let endpoint = tonic::transport::Endpoint::from_shared(self.addr.clone())
                .map_err(|e| {
                    RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("invalid peer address '{}': {e}", self.addr),
                    )))
                })?
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(30));

            // connect_lazy() returns immediately without establishing a TCP
            // connection. The underlying hyper Channel will connect on first
            // RPC and automatically reconnect if the connection drops.
            // This replaces the previous connect().await which created a
            // one-shot connection that couldn't recover from network drops.
            let channel = endpoint.connect_lazy();

            self.client = Some(RaftServiceClient::new(channel));
        }

        // Safe: we just set client above if it was None
        self.client.as_mut().ok_or_else(|| {
            RPCError::Unreachable(Unreachable::new(&std::io::Error::other(
                "client initialization failed",
            )))
        })
    }
}

impl NetBackoff<C> for GrpcNetwork {
    fn backoff(&self) -> Backoff {
        // Infinite 200ms backoff (matches openraft default).
        // openraft enables backoff after 20 consecutive errors.
        Backoff::new(std::iter::repeat(Duration::from_millis(200)))
    }
}

impl NetVote<C> for GrpcNetwork {
    async fn vote(
        &mut self,
        rpc: openraft::raft::VoteRequest<C>,
        _option: RPCOption,
    ) -> Result<openraft::raft::VoteResponse<C>, RPCError<C>> {
        let client = self.get_client().await?;
        let payload = RaftPayload {
            data: serialize(&rpc)?,
        };
        let response = client.vote(payload).await.map_err(tonic_to_rpc_error)?;
        deserialize(&response.into_inner().data)
    }
}

impl NetStreamAppend<C> for GrpcNetwork {
    fn stream_append<'s, S>(
        &'s mut self,
        input: S,
        _option: RPCOption,
    ) -> futures_util::future::BoxFuture<
        's,
        Result<BoxStream<'s, Result<StreamAppendResult<C>, RPCError<C>>>, RPCError<C>>,
    >
    where
        S: futures_util::Stream<Item = openraft::raft::AppendEntriesRequest<C>>
            + OptionalSend
            + Unpin
            + 'static,
    {
        Box::pin(async move {
            let client = self.get_client().await?;

            // Map openraft AppendEntriesRequest stream → msgpack bytes → RaftPayload stream
            let request_stream = input.map(|req| {
                let data = rmp_serde::to_vec(&req).unwrap_or_default();
                RaftPayload { data }
            });

            // Call bidi streaming RPC
            let response = client
                .stream_append(request_stream)
                .await
                .map_err(tonic_to_rpc_error)?;

            // Map response stream: RaftPayload → deserialize → StreamAppendResult
            let output = response.into_inner().map(|result| {
                let payload = result.map_err(tonic_to_rpc_error)?;
                let stream_result: StreamAppendResult<C> = deserialize(&payload.data)?;
                Ok(stream_result)
            });

            Ok(Box::pin(output) as BoxStream<'s, _>)
        })
    }
}

impl NetSnapshot<C> for GrpcNetwork {
    async fn full_snapshot(
        &mut self,
        vote: openraft::type_config::alias::VoteOf<C>,
        snapshot: openraft::type_config::alias::SnapshotOf<C>,
        cancel: impl Future<Output = openraft::error::ReplicationClosed> + OptionalSend + 'static,
        _option: RPCOption,
    ) -> Result<openraft::raft::SnapshotResponse<C>, openraft::error::StreamingError<C>> {
        let target_addr = self.addr.clone();
        let client = self.get_client().await.map_err(|e| {
            let io_err = std::io::Error::new(std::io::ErrorKind::ConnectionAborted, e.to_string());
            openraft::error::StreamingError::Unreachable(Unreachable::new(&io_err))
        })?;

        // Build chunked snapshot transfer: header message + data chunks.
        // This avoids sending the entire snapshot as a single gRPC message
        // which would cause OOM on large snapshots (>1GB).
        let snapshot_data = snapshot.snapshot.into_inner();
        let data_size = snapshot_data.len() as u64;

        let header = crate::snapshot::SnapshotTransferHeader {
            vote,
            meta: snapshot.meta.clone(),
            data_size,
            since_ts: None, // Full snapshot; incremental uses Some(ts)
        };

        let header_msg = crate::snapshot::SnapshotChunkMessage::Header(header);
        let header_bytes = rmp_serde::to_vec(&header_msg).map_err(|e| {
            openraft::error::StreamingError::Unreachable(Unreachable::new(&std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("serialize snapshot header: {e}"),
            )))
        })?;

        // Build the stream: header message + data chunk messages
        let mut payloads = vec![RaftPayload { data: header_bytes }];

        for chunk in crate::snapshot::chunk_snapshot_data(&snapshot_data) {
            let chunk_msg = crate::snapshot::SnapshotChunkMessage::DataChunk(chunk.to_vec());
            let chunk_bytes = rmp_serde::to_vec(&chunk_msg).map_err(|e| {
                openraft::error::StreamingError::Unreachable(Unreachable::new(
                    &std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("serialize snapshot chunk: {e}"),
                    ),
                ))
            })?;
            payloads.push(RaftPayload { data: chunk_bytes });
        }

        tracing::info!(
            snapshot_id = %snapshot.meta.snapshot_id,
            data_bytes = data_size,
            chunks = payloads.len() - 1,
            target = %target_addr,
            "sending chunked snapshot to follower"
        );

        let request_stream = futures_util::stream::iter(payloads);

        // Race the gRPC call against openraft's cancel signal.
        // If replication is cancelled (leader steps down, follower removed),
        // abort the transfer immediately instead of blocking on send.
        let cancel_boxed = Box::pin(cancel);
        let grpc_fut = client.snapshot(request_stream);

        let response = tokio::select! {
            result = grpc_fut => {
                result.map_err(|status| {
                    openraft::error::StreamingError::Unreachable(Unreachable::new(
                        &std::io::Error::new(
                            std::io::ErrorKind::ConnectionAborted,
                            format!("snapshot gRPC error: {status}"),
                        ),
                    ))
                })?
            }
            closed = cancel_boxed => {
                tracing::warn!(
                    target = %target_addr,
                    "snapshot transfer cancelled by openraft"
                );
                return Err(openraft::error::StreamingError::Closed(closed));
            }
        };

        let resp_data = response.into_inner().data;
        let snap_response: openraft::raft::SnapshotResponse<C> = rmp_serde::from_slice(&resp_data)
            .map_err(|e| {
                openraft::error::StreamingError::Unreachable(Unreachable::new(
                    &std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("deserialize snapshot response: {e}"),
                    ),
                ))
            })?;

        tracing::info!("snapshot transfer to {} complete", target_addr);
        Ok(snap_response)
    }
}

impl NetTransferLeader<C> for GrpcNetwork {
    async fn transfer_leader(
        &mut self,
        req: openraft::raft::TransferLeaderRequest<C>,
        _option: RPCOption,
    ) -> Result<(), RPCError<C>> {
        let client = self.get_client().await?;
        let payload = RaftPayload {
            data: serialize(&req)?,
        };
        client
            .transfer_leader(payload)
            .await
            .map_err(tonic_to_rpc_error)?;
        Ok(())
    }
}

// ── Stub Network ───────────────────────────────────────────────────

/// Stub for single-node — returns Unreachable for all RPCs.
pub struct StubNetwork;

impl NetBackoff<C> for StubNetwork {
    fn backoff(&self) -> Backoff {
        Backoff::new(std::iter::repeat(Duration::from_millis(200)))
    }
}

impl NetVote<C> for StubNetwork {
    async fn vote(
        &mut self,
        _rpc: openraft::raft::VoteRequest<C>,
        _option: RPCOption,
    ) -> Result<openraft::raft::VoteResponse<C>, RPCError<C>> {
        Err(RPCError::Unreachable(Unreachable::new(
            &std::io::Error::new(std::io::ErrorKind::NotConnected, "stub: no peers"),
        )))
    }
}

impl NetStreamAppend<C> for StubNetwork {
    fn stream_append<'s, S>(
        &'s mut self,
        _input: S,
        _option: RPCOption,
    ) -> futures_util::future::BoxFuture<
        's,
        Result<BoxStream<'s, Result<StreamAppendResult<C>, RPCError<C>>>, RPCError<C>>,
    >
    where
        S: futures_util::Stream<Item = openraft::raft::AppendEntriesRequest<C>>
            + OptionalSend
            + Unpin
            + 'static,
    {
        Box::pin(async {
            Err(RPCError::Unreachable(Unreachable::new(
                &std::io::Error::new(std::io::ErrorKind::NotConnected, "stub: no peers"),
            )))
        })
    }
}

impl NetSnapshot<C> for StubNetwork {
    async fn full_snapshot(
        &mut self,
        _vote: openraft::type_config::alias::VoteOf<C>,
        _snapshot: openraft::type_config::alias::SnapshotOf<C>,
        _cancel: impl Future<Output = openraft::error::ReplicationClosed> + OptionalSend + 'static,
        _option: RPCOption,
    ) -> Result<openraft::raft::SnapshotResponse<C>, openraft::error::StreamingError<C>> {
        Err(openraft::error::StreamingError::Unreachable(
            Unreachable::new(&std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "stub: no peers",
            )),
        ))
    }
}

impl NetTransferLeader<C> for StubNetwork {
    async fn transfer_leader(
        &mut self,
        _req: openraft::raft::TransferLeaderRequest<C>,
        _option: RPCOption,
    ) -> Result<(), RPCError<C>> {
        Err(RPCError::Unreachable(Unreachable::new(
            &std::io::Error::new(std::io::ErrorKind::NotConnected, "stub: no peers"),
        )))
    }
}
