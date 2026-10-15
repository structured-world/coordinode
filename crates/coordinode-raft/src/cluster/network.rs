//! gRPC network implementation for openraft inter-node communication.
//!
//! Implements all 5 openraft network traits over tonic gRPC.
//! Uses opaque bytes (msgpack serialization) for all openraft types.
//!
//! A peer is dialed expecting the node identity its membership entry
//! records: the address there is only where to find it. A member that
//! records no identity is never dialed.

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

use std::sync::Arc;

use coordinode_core::version::Handshake;

use coordinode_wire::peer::PeerTarget;

use coordinode_wire::peer::PeerConnections;

use super::admission::{DeploymentView, member_target, write_deployment};
use super::version::{VersionGate, read_handshake, write_handshake};
use crate::member::MemberNode;
use crate::proto::internode::HandshakeRecord;
use crate::proto::internode::version_handshake_client::VersionHandshakeClient;
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

// ── Version records ────────────────────────────────────────────────

/// A call carrying this member's version record and its deployment.
fn stamped<T>(gate: &VersionGate, deployment: &DeploymentView, body: T) -> tonic::Request<T> {
    let mut request = tonic::Request::new(body);
    write_handshake(request.metadata_mut(), &gate.local_handshake());
    if let Some(deployment) = deployment.get() {
        write_deployment(request.metadata_mut(), deployment);
    }
    request
}

/// Learn the peer's version record from an answer's metadata.
fn learn(gate: &VersionGate, metadata: &tonic::metadata::MetadataMap) {
    if let Ok(peer) = read_handshake(metadata) {
        gate.observe(&peer);
    }
}

/// [`tonic_to_rpc_error`], learning the peer's record from a refusal first.
fn refused(gate: &VersionGate, status: tonic::Status) -> RPCError<C> {
    learn(gate, status.metadata());
    tonic_to_rpc_error(status)
}

// ── Shutdown ───────────────────────────────────────────────────────

/// Whether this node is shutting down. openraft's replication tasks can
/// outlive its core, each inside a call to a peer and holding a reader of the
/// log; a call to a peer that never answers would keep one there until the
/// call times out, and the node's shutdown waits for it. Once this is set,
/// every call in flight or made after fails at once.
#[derive(Clone)]
pub(crate) struct Closing(tokio::sync::watch::Receiver<bool>);

impl Closing {
    pub(crate) fn new(rx: tokio::sync::watch::Receiver<bool>) -> Self {
        Self(rx)
    }

    /// Resolves once the node shuts down, or once the node is gone.
    async fn closed(mut self) {
        // An error means the sender, and with it the node, is gone.
        let _ = self.0.wait_for(|closing| *closing).await;
    }

    fn is_closed(&self) -> bool {
        *self.0.borrow()
    }
}

fn shutting_down() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::Interrupted,
        "this node is shutting down",
    )
}

/// `rpc`, unless the node starts shutting down first.
async fn unless_closing<T>(
    closing: &Closing,
    rpc: impl Future<Output = Result<T, RPCError<C>>>,
) -> Result<T, RPCError<C>> {
    tokio::select! {
        result = rpc => result,
        () = closing.clone().closed() => {
            Err(RPCError::Unreachable(Unreachable::new(&shutting_down())))
        }
    }
}

// ── Member book ────────────────────────────────────────────────────

/// Where each member is found and whom it must prove there, as the
/// membership in effect says. A replication stream outlives a change to its
/// member's entry, so a call reads the entry here each time it dials rather
/// than keeping the one its stream started with: a member re-addressed is
/// reached at its new address with no other change.
#[derive(Default)]
pub(crate) struct MemberBook {
    // no-std: spin::RwLock; read when a peer client is built, written when
    // the membership changes.
    members: parking_lot::RwLock<std::collections::BTreeMap<u64, MemberNode>>,
}

impl MemberBook {
    /// Replace the book with the members of the membership in effect.
    pub(crate) fn set<'a>(&self, members: impl Iterator<Item = (&'a u64, &'a MemberNode)>) {
        *self.members.write() = members.map(|(&id, node)| (id, node.clone())).collect();
    }

    fn get(&self, node_id: u64) -> Option<MemberNode> {
        self.members.read().get(&node_id).cloned()
    }
}

// ── Network Factory ────────────────────────────────────────────────

/// gRPC-based network factory for multi-node cluster.
pub struct GrpcNetworkFactory {
    /// This node's id — paired with the per-client target id so the test-only
    /// [`nemesis`](super::nemesis) partition matrix can gate directed RPCs.
    pub(crate) local_node_id: u64,
    /// Fails every peer call once this node shuts down.
    pub(crate) closing: Closing,
    /// This member's version view: stamps every call, learns from every
    /// answer.
    pub(crate) gate: Arc<VersionGate>,
    /// The deployment every call names.
    pub(crate) deployment: DeploymentView,
    /// The connections to peers, made under this member's security context.
    pub(crate) connections: Arc<PeerConnections>,
    /// The members as the membership in effect records them.
    pub(crate) book: Arc<MemberBook>,
}

impl RaftNetworkFactory<C> for GrpcNetworkFactory {
    type Network = GrpcNetwork;

    async fn new_client(&mut self, target: u64, node: &MemberNode) -> Self::Network {
        GrpcNetwork {
            group: self.gate.group().raw(),
            local_node_id: self.local_node_id,
            target_node_id: target,
            first_seen: node.clone(),
            book: Arc::clone(&self.book),
            client: None,
            closing: self.closing.clone(),
            gate: Arc::clone(&self.gate),
            deployment: self.deployment.clone(),
            connections: Arc::clone(&self.connections),
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
        _node: &crate::member::MemberNode,
    ) -> Self::Network {
        StubNetwork
    }
}

// ── gRPC Network ───────────────────────────────────────────────────

/// gRPC connection to a single Raft peer. Lazily connects on first use.
pub struct GrpcNetwork {
    /// The consensus group every message to the peer names, so a server
    /// hosting several groups dispatches it to its replica of this one.
    group: u64,
    /// Source node id (this node), for the test-only partition nemesis gate.
    local_node_id: u64,
    /// Target peer node id, for the test-only partition nemesis gate.
    target_node_id: u64,
    /// The peer's entry when its stream started, until the book has one.
    first_seen: MemberNode,
    /// The members as the membership in effect records them.
    book: Arc<MemberBook>,
    /// The consensus client, and the peer it dials.
    client: Option<(PeerTarget, RaftServiceClient<tonic::transport::Channel>)>,
    /// Fails this peer's calls once this node shuts down.
    closing: Closing,
    gate: Arc<VersionGate>,
    /// The deployment calls name.
    deployment: DeploymentView,
    /// The connections calls travel over.
    connections: Arc<PeerConnections>,
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
    /// Get or create the gRPC client for this peer, over the server's shared
    /// connection to it. The connection reconnects on its own after a drop;
    /// with openraft's `NetBackoff` (200ms infinite retry) this recovers from
    /// network partitions and peer restarts.
    async fn get_client(
        &mut self,
    ) -> Result<&mut RaftServiceClient<tonic::transport::Channel>, RPCError<C>> {
        self.ensure_peer_matches().await?;
        let peer = self.peer()?;
        // A client made for where the peer was before it moved is replaced.
        if self
            .client
            .as_ref()
            .is_none_or(|(dialed, _)| *dialed != peer)
        {
            let channel = self.channel(&peer)?;
            self.client = Some((peer, RaftServiceClient::new(channel)));
        }

        // Safe: we just set client above if it was None
        self.client
            .as_mut()
            .map(|(_, client)| client)
            .ok_or_else(|| {
                RPCError::Unreachable(Unreachable::new(&std::io::Error::other(
                    "client initialization failed",
                )))
            })
    }

    /// A peer last seen at another pair is sent the version exchange alone:
    /// nothing of the consensus goes to it until it reports this member's
    /// pair.
    async fn ensure_peer_matches(&mut self) -> Result<(), RPCError<C>> {
        if self.gate.peer_matches(self.target_node_id) != Some(false) {
            return Ok(());
        }
        let channel = self.channel(&self.peer()?)?;
        let mut exchange = VersionHandshakeClient::new(channel);
        let answer = exchange
            .exchange(stamped(
                &self.gate,
                &self.deployment,
                HandshakeRecord {
                    record: self.gate.local_handshake().encode(),
                },
            ))
            .await
            .map_err(tonic_to_rpc_error)?;
        if let Ok(peer) = Handshake::decode(&answer.into_inner().record) {
            self.gate.observe(&peer);
        }
        if self.gate.peer_matches(self.target_node_id) == Some(true) {
            return Ok(());
        }
        Err(RPCError::Unreachable(Unreachable::new(
            &std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!(
                    "node {} runs another version than {}",
                    self.target_node_id,
                    self.gate.pair()
                ),
            ),
        )))
    }

    /// The peer as the membership in effect records it: where to find it
    /// and the identity it must prove there.
    fn peer(&self) -> Result<PeerTarget, RPCError<C>> {
        let node = self
            .book
            .get(self.target_node_id)
            .unwrap_or_else(|| self.first_seen.clone());
        member_target(&node).ok_or_else(|| {
            unreachable(format!(
                "node {} at '{}' records no node identity to expect",
                self.target_node_id, node.addr
            ))
        })
    }

    /// A handle onto the one connection to `peer`, made expecting its
    /// identity.
    fn channel(&self, peer: &PeerTarget) -> Result<tonic::transport::Channel, RPCError<C>> {
        self.connections
            .channel(peer)
            .map_err(|e| unreachable(e.to_string()))
    }

    /// Where the peer is found, for messages.
    fn addr(&self) -> String {
        self.book
            .get(self.target_node_id)
            .unwrap_or_else(|| self.first_seen.clone())
            .addr
    }
}

fn unreachable(reason: String) -> RPCError<C> {
    RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        reason,
    )))
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
        let closing = self.closing.clone();
        let gate = Arc::clone(&self.gate);
        let deployment = self.deployment.clone();
        unless_closing(&closing, async {
            let request = stamped(
                &gate,
                &deployment,
                RaftPayload {
                    data: serialize(&rpc)?,
                    group: self.group,
                },
            );
            let client = self.get_client().await?;
            let response = client.vote(request).await.map_err(|s| refused(&gate, s))?;
            learn(&gate, response.metadata());
            deserialize(&response.into_inner().data)
        })
        .await
    }

    /// Asked of the peer itself: the engine's default answers "granted"
    /// without asking anyone, and a member cut off from its leader would then
    /// raise its term and depose a healthy leader on its return. A peer that
    /// cannot be reached is an error, never a grant.
    async fn pre_vote(
        &mut self,
        rpc: openraft::raft::VoteRequest<C>,
        _option: RPCOption,
    ) -> Result<openraft::raft::VoteResponse<C>, RPCError<C>> {
        if let Some(e) = self.partitioned() {
            return Err(e);
        }
        let closing = self.closing.clone();
        let gate = Arc::clone(&self.gate);
        let deployment = self.deployment.clone();
        unless_closing(&closing, async {
            let request = stamped(
                &gate,
                &deployment,
                RaftPayload {
                    data: serialize(&rpc)?,
                    group: self.group,
                },
            );
            let client = self.get_client().await?;
            let response = client
                .pre_vote(request)
                .await
                .map_err(|s| refused(&gate, s))?;
            learn(&gate, response.metadata());
            deserialize(&response.into_inner().data)
        })
        .await
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
        let group = self.group;
        let closing = self.closing.clone();
        let gate = Arc::clone(&self.gate);
        let deployment = self.deployment.clone();
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
                .map(move |req| {
                    let data = rmp_serde::to_vec(&req).unwrap_or_default();
                    RaftPayload { data, group }
                });

            // Call bidi streaming RPC
            let response = unless_closing(&closing, async {
                client
                    .stream_append(stamped(&gate, &deployment, request_stream))
                    .await
                    .map_err(|s| refused(&gate, s))
            })
            .await?;
            learn(&gate, response.metadata());

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

            // A shutdown cuts the replies short with an error, not a clean
            // end: a clean end would leave the entries sent awaiting replies.
            let cut = closing.clone();
            let output = output.take_until(closing.closed()).chain(
                futures_util::stream::once(async move { cut.is_closed() }).filter_map(
                    |closed| async move {
                        closed
                            .then(|| Err(RPCError::Unreachable(Unreachable::new(&shutting_down()))))
                    },
                ),
            );

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
        let target_addr = self.addr();
        let closing = self.closing.clone();
        let gate = Arc::clone(&self.gate);
        let deployment = self.deployment.clone();
        let group = self.group;
        let client = self.get_client().await.map_err(|e| {
            let io_err = std::io::Error::new(std::io::ErrorKind::ConnectionAborted, e.to_string());
            openraft::error::StreamingError::Unreachable(Unreachable::new(&io_err))
        })?;

        // A header message, then the snapshot file in chunks read as the
        // stream is consumed: the snapshot is never in memory whole.
        let io_unreachable =
            |e: std::io::Error| openraft::error::StreamingError::Unreachable(Unreachable::new(&e));
        // A snapshot kept as a capture is serialized here, the one time a
        // peer needs its bytes; that reads the whole store, off the runtime.
        let mut file = tokio::task::spawn_blocking(move || {
            let mut file = snapshot.snapshot;
            file.materialize().map(|()| file)
        })
        .await
        .map_err(|e| io_unreachable(std::io::Error::other(e)))?
        .map_err(io_unreachable)?;
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
        let chunks = futures_util::stream::unfold(
            (reader, data_size),
            move |(mut reader, left)| async move {
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
                    Ok(data) => Some((RaftPayload { data, group }, (reader, left - len as u64))),
                    Err(e) => {
                        tracing::warn!(%e, "encoding a snapshot chunk");
                        None
                    }
                }
            },
        );
        let request_stream = futures_util::stream::once(async move {
            RaftPayload {
                data: header_bytes,
                group,
            }
        })
        .chain(chunks);

        // Race the gRPC call against openraft's cancel signal.
        // If replication is cancelled (leader steps down, follower removed),
        // abort the transfer immediately instead of blocking on send.
        let cancel_boxed = Box::pin(cancel);
        let grpc_fut = client.snapshot(stamped(&gate, &deployment, request_stream));

        let response = tokio::select! {
            result = grpc_fut => {
                result.map_err(|status| {
                    learn(&gate, status.metadata());
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
            () = closing.closed() => {
                return Err(openraft::error::StreamingError::Unreachable(Unreachable::new(
                    &shutting_down(),
                )));
            }
        };

        learn(&gate, response.metadata());
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
        let closing = self.closing.clone();
        let gate = Arc::clone(&self.gate);
        let deployment = self.deployment.clone();
        unless_closing(&closing, async {
            let request = stamped(
                &gate,
                &deployment,
                RaftPayload {
                    data: serialize(&req)?,
                    group: self.group,
                },
            );
            let client = self.get_client().await?;
            // The server maps any application-level transfer error to a gRPC
            // Status (→ outer RPCError below); a successful RPC means the
            // remote accepted the transfer, so the inner result is Ok.
            let response = client
                .transfer_leader(request)
                .await
                .map_err(|s| refused(&gate, s))?;
            learn(&gate, response.metadata());
            Ok(Ok(()))
        })
        .await
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

    async fn pre_vote(
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
