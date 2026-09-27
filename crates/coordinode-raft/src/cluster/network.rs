//! gRPC network implementation for openraft inter-node communication.
//!
//! Implements all 5 openraft network traits over tonic gRPC.
//! Uses opaque bytes (msgpack serialization) for all openraft types.

use std::future::Future;
use std::time::Duration;

use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use openraft::error::{RPCError, Unreachable};
use openraft::network::{
    Backoff, NetBackoff, NetSnapshot, NetStreamAppend, NetTransferLeader, NetVote, RPCOption,
};
use openraft::raft::StreamAppendResult;
use openraft::raft::TransferLeaderError;
use openraft::{OptionalSend, RaftNetworkFactory};

use crate::proto::replication::RaftPayload;
use crate::proto::replication::raft_service_client::RaftServiceClient;
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
pub struct GrpcNetworkFactory {
    /// This node's id — paired with the per-client target id so the test-only
    /// [`nemesis`](super::nemesis) partition matrix can gate directed RPCs.
    pub(crate) local_node_id: u64,
}

impl RaftNetworkFactory<C> for GrpcNetworkFactory {
    type Network = GrpcNetwork;

    async fn new_client(
        &mut self,
        target: u64,
        node: &openraft::impls::BasicNode,
    ) -> Self::Network {
        GrpcNetwork {
            local_node_id: self.local_node_id,
            target_node_id: target,
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
    /// Source node id (this node), for the test-only partition nemesis gate.
    local_node_id: u64,
    /// Target peer node id, for the test-only partition nemesis gate.
    target_node_id: u64,
    addr: String,
    client: Option<RaftServiceClient<tonic::transport::Channel>>,
}

impl GrpcNetwork {
    /// Test-only network-partition gate. Returns an `Unreachable` RPC error when
    /// the [`nemesis`](super::nemesis) matrix has the directed link
    /// `local → target` blocked; a no-op (single relaxed atomic load) otherwise.
    fn partitioned(&self) -> Option<RPCError<C>> {
        if super::nemesis::is_blocked(self.local_node_id, self.target_node_id) {
            Some(RPCError::Unreachable(Unreachable::new(
                &std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    format!(
                        "nemesis: partitioned {} -> {}",
                        self.local_node_id, self.target_node_id
                    ),
                ),
            )))
        } else {
            None
        }
    }
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
    /// `connect().await` is not used: it creates a one-shot connection,
    /// and once that drops every subsequent RPC fails permanently.
    async fn get_client(
        &mut self,
    ) -> Result<&mut RaftServiceClient<tonic::transport::Channel>, RPCError<C>> {
        if self.client.is_none() {
            let mut endpoint = tonic::transport::Endpoint::from_shared(self.addr.clone())
                .map_err(|e| {
                    RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("invalid peer address '{}': {e}", self.addr),
                    )))
                })?
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(30));

            // Encrypt the peer connection when inter-node TLS is configured
            // (process-global, set once at startup). Off = plaintext.
            if let Some(tls) = coordinode_wire::wire_client_tls() {
                endpoint = endpoint.tls_config(tls).map_err(|e| {
                    RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("peer TLS config for '{}': {e}", self.addr),
                    )))
                })?;
            }

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
    fn backoff(&self) -> Option<Backoff> {
        // Infinite 200ms backoff (matches openraft default).
        // openraft enables backoff after 20 consecutive errors.
        Some(Backoff::new(std::iter::repeat(Duration::from_millis(200))))
    }
}

impl NetVote<C> for GrpcNetwork {
    async fn vote(
        &mut self,
        rpc: openraft::raft::VoteRequest<C>,
        _option: RPCOption,
    ) -> Result<openraft::raft::VoteResponse<C>, RPCError<C>> {
        if let Some(e) = self.partitioned() {
            return Err(e);
        }
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
        let partition = self.partitioned();
        let (local, target) = (self.local_node_id, self.target_node_id);
        Box::pin(async move {
            if let Some(e) = partition {
                return Err(e);
            }
            let client = self.get_client().await?;

            // Map openraft AppendEntriesRequest stream → msgpack bytes → RaftPayload stream.
            // A partition cuts a stream that is already open too: the request
            // side ends at the first entry sent across a blocked link.
            let request_stream = input
                .take_while(move |_| {
                    futures_util::future::ready(!super::nemesis::is_blocked(local, target))
                })
                .map(|req| {
                    let data = rmp_serde::to_vec(&req).unwrap_or_default();
                    RaftPayload { data }
                });

            // Call bidi streaming RPC
            let response = client
                .stream_append(request_stream)
                .await
                .map_err(tonic_to_rpc_error)?;

            // Map response stream: RaftPayload → deserialize → StreamAppendResult.
            // A reply across a blocked link is lost, as it would be on the wire.
            let output = response.into_inner().map(move |result| {
                if super::nemesis::is_blocked(target, local) {
                    return Err(RPCError::Unreachable(Unreachable::new(
                        &std::io::Error::new(
                            std::io::ErrorKind::NotConnected,
                            format!("nemesis: partitioned {target} -> {local}"),
                        ),
                    )));
                }
                let payload = result.map_err(tonic_to_rpc_error)?;
                let stream_result: StreamAppendResult<C> = deserialize(&payload.data)?;
                Ok(stream_result)
            });

            Ok(Box::pin(output) as BoxStream<'s, _>)
        })
    }
}

impl NetSnapshot<C> for GrpcNetwork {
    type SnapshotData = crate::snapshot::SnapshotFile;

    async fn full_snapshot(
        &mut self,
        vote: openraft::type_config::alias::VoteOf<C>,
        snapshot: openraft::type_config::alias::SnapshotOf<C, Self::SnapshotData>,
        cancel: impl Future<Output = openraft::error::ReplicationClosed> + OptionalSend + 'static,
        _option: RPCOption,
    ) -> Result<openraft::raft::SnapshotResponse<C>, openraft::error::StreamingError<C>> {
        if super::nemesis::is_blocked(self.local_node_id, self.target_node_id) {
            return Err(openraft::error::StreamingError::Unreachable(
                Unreachable::new(&std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    format!(
                        "nemesis: partitioned {} -> {}",
                        self.local_node_id, self.target_node_id
                    ),
                )),
            ));
        }
        let target_addr = self.addr.clone();
        let client = self.get_client().await.map_err(|e| {
            let io_err = std::io::Error::new(std::io::ErrorKind::ConnectionAborted, e.to_string());
            openraft::error::StreamingError::Unreachable(Unreachable::new(&io_err))
        })?;

        // A header message, then the snapshot file in chunks read as the
        // stream is consumed: the snapshot is never in memory whole.
        let io_unreachable =
            |e: std::io::Error| openraft::error::StreamingError::Unreachable(Unreachable::new(&e));
        let mut file = snapshot.snapshot;
        let data_size = file.size().map_err(io_unreachable)?;
        {
            use std::io::Seek;
            file.rewind().map_err(io_unreachable)?;
        }
        let reader = tokio::fs::File::from_std(file.try_clone_file().map_err(io_unreachable)?);

        let header = crate::snapshot::SnapshotTransferHeader {
            vote,
            meta: snapshot.meta.clone(),
            data_size,
        };

        let header_msg = crate::snapshot::SnapshotChunkMessage::Header(header);
        let header_bytes = rmp_serde::to_vec(&header_msg).map_err(|e| {
            openraft::error::StreamingError::Unreachable(Unreachable::new(&std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("serialize snapshot header: {e}"),
            )))
        })?;

        tracing::info!(
            data_bytes = data_size,
            chunks = data_size.div_ceil(crate::snapshot::SNAPSHOT_CHUNK_SIZE as u64),
            target = %target_addr,
            "sending chunked snapshot to follower"
        );

        // A read or encode failure ends the stream early; the receiver then
        // refuses the transfer for its short size and the RPC fails.
        let chunks =
            futures_util::stream::unfold((reader, data_size), |(mut reader, left)| async move {
                use tokio::io::AsyncReadExt;
                if left == 0 {
                    return None;
                }
                // At most SNAPSHOT_CHUNK_SIZE, a usize.
                let len = left.min(crate::snapshot::SNAPSHOT_CHUNK_SIZE as u64) as usize;
                let mut chunk = vec![0u8; len];
                if let Err(e) = reader.read_exact(&mut chunk).await {
                    tracing::warn!(%e, "reading the snapshot file to send it");
                    return None;
                }
                let message = crate::snapshot::SnapshotChunkMessage::DataChunk(chunk);
                match rmp_serde::to_vec(&message) {
                    Ok(data) => Some((RaftPayload { data }, (reader, left - len as u64))),
                    Err(e) => {
                        tracing::warn!(%e, "encoding a snapshot chunk");
                        None
                    }
                }
            });
        let request_stream =
            futures_util::stream::once(async move { RaftPayload { data: header_bytes } })
                .chain(chunks);

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
    ) -> Result<Result<(), TransferLeaderError<C>>, RPCError<C>> {
        if let Some(e) = self.partitioned() {
            return Err(e);
        }
        let client = self.get_client().await?;
        let payload = RaftPayload {
            data: serialize(&req)?,
        };
        // The server maps any application-level transfer error to a gRPC
        // Status (→ outer RPCError below); a successful RPC means the remote
        // accepted the transfer, so the inner result is Ok.
        client
            .transfer_leader(payload)
            .await
            .map_err(tonic_to_rpc_error)?;
        Ok(Ok(()))
    }
}

// ── Stub Network ───────────────────────────────────────────────────

/// Stub for single-node — returns Unreachable for all RPCs.
pub struct StubNetwork;

impl NetBackoff<C> for StubNetwork {
    fn backoff(&self) -> Option<Backoff> {
        Some(Backoff::new(std::iter::repeat(Duration::from_millis(200))))
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
    type SnapshotData = crate::snapshot::SnapshotFile;

    async fn full_snapshot(
        &mut self,
        _vote: openraft::type_config::alias::VoteOf<C>,
        _snapshot: openraft::type_config::alias::SnapshotOf<C, Self::SnapshotData>,
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
    ) -> Result<Result<(), TransferLeaderError<C>>, RPCError<C>> {
        Err(RPCError::Unreachable(Unreachable::new(
            &std::io::Error::new(std::io::ErrorKind::NotConnected, "stub: no peers"),
        )))
    }
}
