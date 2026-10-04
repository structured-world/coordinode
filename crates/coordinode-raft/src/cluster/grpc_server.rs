//! gRPC server handler for Raft inter-node RPCs.
//!
//! Implements `RaftService` tonic trait. A server hosts replicas of several
//! consensus groups behind one service: every request names its group and is
//! dispatched to this server's replica of it. Uses msgpack for type
//! serialization.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use coordinode_core::group::GroupId;
use futures_util::{Stream, StreamExt};
use tonic::metadata::MetadataMap;
use tonic::{Request, Response, Status, Streaming};
use tonic_types::{ErrorDetails, StatusExt};

use super::version::{
    HandshakeService, VersionGate, read_handshake, refusal_status, write_handshake,
};
use crate::proto::replication::raft_service_server::RaftService;
use crate::proto::replication::{RaftEmpty, RaftPayload};
use crate::storage::{CoordinodeStateMachine, TypeConfig};

type RaftInstance = openraft::Raft<TypeConfig, CoordinodeStateMachine>;

/// `ErrorInfo.reason` of a request for a group the server hosts no replica
/// of; `metadata.group` names the group.
pub const GROUP_NOT_HOSTED: &str = "GROUP_NOT_HOSTED";

/// This server's replica of one consensus group.
pub struct GroupReplica {
    raft: Arc<RaftInstance>,
    /// The node's snapshot directory, where a received snapshot is staged so
    /// the install publishes it in place.
    snapshot_dir: PathBuf,
    /// Refuses calls from members of another version before their payload
    /// is read.
    gate: Arc<VersionGate>,
    /// While the disk is below its free-space reserve, entries and snapshots
    /// from the leader are refused as unavailable, so the leader retries
    /// later: consensus pauses here instead of failing a log fsync on a full
    /// disk, which would stop it for good.
    space: Arc<coordinode_storage::engine::space::SpaceGuard>,
}

impl GroupReplica {
    fn group(&self) -> GroupId {
        self.gate.group()
    }

    /// Refuse a call that would write the log or install a snapshot while
    /// the disk is below its reserve. UNAVAILABLE: the leader backs off and
    /// sends again once space is freed.
    fn admit_write(&self) -> Result<(), Status> {
        self.space
            .admit()
            .map_err(|e| Status::unavailable(e.to_string()))
    }

    /// Admit a call by the version record in its metadata, or the status
    /// refusing it. A record naming another group than this replica's is
    /// refused like a record of another version.
    fn admit(&self, metadata: &MetadataMap) -> Result<(), Status> {
        self.gate
            .admit(read_handshake(metadata))
            .map_err(|refusal| {
                metrics::counter!("coordinode_version_refused_calls_total").increment(1);
                refusal_status(&refusal, &self.gate.local_handshake())
            })
    }

    /// `body` as the call's response, carrying this member's record.
    fn answer<T>(&self, body: T) -> Response<T> {
        let mut response = Response::new(body);
        write_handshake(response.metadata_mut(), &self.gate.local_handshake());
        response
    }

    /// `data` as a payload of this replica's group.
    fn payload(&self, data: Vec<u8>) -> RaftPayload {
        RaftPayload {
            data,
            group: self.group().raw(),
        }
    }
}

/// Why a replica could not be added to a server.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HostError {
    /// The server already hosts a replica of the group.
    #[error("this server already hosts a replica of {0}")]
    AlreadyHosted(GroupId),
}

/// The consensus groups a server hosts a replica of, by group.
#[derive(Default)]
pub struct HostedGroups {
    // no-std: spin::RwLock; read once per inter-node call, written when a
    // replica is added or removed.
    replicas: parking_lot::RwLock<BTreeMap<GroupId, Arc<GroupReplica>>>,
}

impl HostedGroups {
    /// The groups hosted, in order.
    pub fn groups(&self) -> Vec<GroupId> {
        self.replicas.read().keys().copied().collect()
    }

    /// Stop serving `group`; whether it was hosted. Requests for it are
    /// refused from then on.
    pub fn remove(&self, group: GroupId) -> bool {
        self.replicas.write().remove(&group).is_some()
    }

    fn insert(&self, replica: Arc<GroupReplica>) -> Result<(), HostError> {
        let group = replica.group();
        let mut replicas = self.replicas.write();
        if replicas.contains_key(&group) {
            return Err(HostError::AlreadyHosted(group));
        }
        replicas.insert(group, replica);
        Ok(())
    }

    /// This server's replica of `group`, or NOT_FOUND naming it.
    fn replica(&self, group: GroupId) -> Result<Arc<GroupReplica>, Status> {
        self.replicas
            .read()
            .get(&group)
            .cloned()
            .ok_or_else(|| not_hosted(group))
    }

    /// The replica a call is for by the group its version record names. A
    /// call without a readable record is answered by the only replica when
    /// there is one, whose admission then refuses it for the missing record.
    fn replica_by_record(&self, metadata: &MetadataMap) -> Result<Arc<GroupReplica>, Status> {
        match read_handshake(metadata) {
            Ok(peer) => self.replica(peer.group_id),
            Err(refusal) => {
                let replicas = self.replicas.read();
                match (replicas.len(), replicas.values().next()) {
                    (1, Some(sole)) => Ok(Arc::clone(sole)),
                    _ => Err(Status::failed_precondition(format!(
                        "version mismatch: {refusal}"
                    ))),
                }
            }
        }
    }

    /// The version gate of this server's member of `group`.
    pub(crate) fn gate(&self, group: GroupId) -> Result<Arc<VersionGate>, Status> {
        self.replica(group).map(|r| Arc::clone(&r.gate))
    }

    /// The version gate of the only group hosted, if exactly one is.
    pub(crate) fn sole_gate(&self) -> Option<Arc<VersionGate>> {
        let replicas = self.replicas.read();
        match (replicas.len(), replicas.values().next()) {
            (1, Some(sole)) => Some(Arc::clone(&sole.gate)),
            _ => None,
        }
    }
}

/// NOT_FOUND for a request naming a group this server does not host.
fn not_hosted(group: GroupId) -> Status {
    let details = ErrorDetails::with_error_info(
        GROUP_NOT_HOSTED,
        coordinode_core::ERROR_DOMAIN,
        [("group".to_string(), group.raw().to_string())],
    );
    Status::with_error_details(
        tonic::Code::NotFound,
        format!("this server hosts no replica of {group}"),
        details,
    )
}

/// gRPC server handler for Raft consensus RPCs of every group the server
/// hosts. Clones share the hosted groups, so a replica added through one is
/// served by all.
#[derive(Clone)]
pub struct RaftGrpcHandler {
    groups: Arc<HostedGroups>,
}

impl RaftGrpcHandler {
    /// A handler hosting `raft`, the replica of the group `gate` speaks for,
    /// staging received snapshots in `snapshot_dir` (see
    /// [`crate::snapshot::snapshot_dir`]), admitting calls through `gate`
    /// and taking writes only while `space` has room.
    pub fn new(
        raft: Arc<RaftInstance>,
        snapshot_dir: PathBuf,
        gate: Arc<VersionGate>,
        space: Arc<coordinode_storage::engine::space::SpaceGuard>,
    ) -> Self {
        let groups = HostedGroups::default();
        let mut replicas = groups.replicas.write();
        let replica = GroupReplica {
            raft,
            snapshot_dir,
            gate,
            space,
        };
        replicas.insert(replica.group(), Arc::new(replica));
        drop(replicas);
        Self {
            groups: Arc::new(groups),
        }
    }

    /// Serve the replicas of `other` from this handler too, so one server
    /// hosts the groups of both. Nothing is added when any of them is hosted
    /// here already.
    pub fn host(&self, other: &RaftGrpcHandler) -> Result<(), HostError> {
        let incoming: Vec<Arc<GroupReplica>> =
            other.groups.replicas.read().values().cloned().collect();
        if let Some(taken) = incoming
            .iter()
            .map(|r| r.group())
            .find(|g| self.groups.replicas.read().contains_key(g))
        {
            return Err(HostError::AlreadyHosted(taken));
        }
        for replica in incoming {
            self.groups.insert(replica)?;
        }
        Ok(())
    }

    /// The groups this handler serves.
    pub fn hosted(&self) -> &Arc<HostedGroups> {
        &self.groups
    }

    /// The frozen version exchange for the groups this handler serves, to
    /// register beside it on the same router.
    pub fn handshake_service(
        &self,
    ) -> crate::proto::internode::version_handshake_server::VersionHandshakeServer<HandshakeService>
    {
        crate::proto::internode::version_handshake_server::VersionHandshakeServer::new(
            HandshakeService::new(Arc::clone(&self.groups)),
        )
    }
}

fn ser<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, Status> {
    rmp_serde::to_vec(value).map_err(|e| Status::internal(format!("msgpack serialize: {e}")))
}

fn de<T: serde::de::DeserializeOwned>(data: &[u8]) -> Result<T, Status> {
    rmp_serde::from_slice(data)
        .map_err(|e| Status::invalid_argument(format!("msgpack deserialize: {e}")))
}

/// A unary request's replica, admitted, with its payload.
fn admitted(
    groups: &HostedGroups,
    request: Request<RaftPayload>,
) -> Result<(Arc<GroupReplica>, Vec<u8>), Status> {
    let (metadata, _, payload) = request.into_parts();
    let replica = groups.replica(GroupId(payload.group))?;
    replica.admit(&metadata)?;
    Ok((replica, payload.data))
}

#[tonic::async_trait]
impl RaftService for RaftGrpcHandler {
    async fn vote(&self, request: Request<RaftPayload>) -> Result<Response<RaftPayload>, Status> {
        let (replica, data) = admitted(&self.groups, request)?;
        let vote_req: openraft::raft::VoteRequest<TypeConfig> = de(&data)?;

        let vote_resp = replica
            .raft
            .vote(vote_req)
            .await
            .map_err(|e| Status::internal(format!("vote: {e}")))?;

        Ok(replica.answer(replica.payload(ser(&vote_resp)?)))
    }

    async fn append_entries(
        &self,
        request: Request<RaftPayload>,
    ) -> Result<Response<RaftPayload>, Status> {
        let (replica, data) = admitted(&self.groups, request)?;
        let req: openraft::raft::AppendEntriesRequest<TypeConfig> = de(&data)?;
        // A heartbeat writes nothing and keeps this member following its
        // leader; only entries wait for space.
        if !req.entries.is_empty() {
            replica.admit_write()?;
        }

        let resp = replica
            .raft
            .append_entries(req)
            .await
            .map_err(|e| Status::internal(format!("append_entries: {e}")))?;

        Ok(replica.answer(replica.payload(ser(&resp)?)))
    }

    type StreamAppendStream = Pin<Box<dyn Stream<Item = Result<RaftPayload, Status>> + Send>>;

    async fn stream_append(
        &self,
        request: Request<Streaming<RaftPayload>>,
    ) -> Result<Response<Self::StreamAppendStream>, Status> {
        // The stream is routed by its version record, before any message
        // arrives; every message must name the same group, and the first one
        // that does not ends the stream.
        let (metadata, _, input) = request.into_parts();
        let replica = self.groups.replica_by_record(&metadata)?;
        replica.admit(&metadata)?;
        let group = replica.group();

        let foreign = Arc::new(core::sync::atomic::AtomicBool::new(false));
        let foreign_in = Arc::clone(&foreign);
        let input = input.take_while(move |result| {
            let pass = result
                .as_ref()
                .map_or(true, |payload| GroupId(payload.group) == group);
            if !pass {
                foreign_in.store(true, core::sync::atomic::Ordering::Release);
            }
            futures_util::future::ready(pass)
        });

        // Deserialize incoming RaftPayload stream → AppendEntriesRequest stream
        let input_stream = input.filter_map(|result| async move {
            match result {
                Ok(payload) => match rmp_serde::from_slice::<
                    openraft::raft::AppendEntriesRequest<TypeConfig>,
                >(&payload.data)
                {
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

        // Heartbeats pass while the disk is below its reserve; the first
        // request carrying entries then ends the stream, and the reply stream
        // ends with UNAVAILABLE so the leader backs off and sends again later.
        let space = Arc::clone(&replica.space);
        let cut = Arc::new(core::sync::atomic::AtomicBool::new(false));
        let cut_in = Arc::clone(&cut);
        let input_stream = input_stream.take_while(move |req| {
            let pass = req.entries.is_empty() || !space.is_paused();
            if !pass {
                cut_in.store(true, core::sync::atomic::Ordering::Release);
            }
            futures_util::future::ready(pass)
        });
        let no_space =
            futures_util::stream::once(
                async move { cut.load(core::sync::atomic::Ordering::Acquire) },
            )
            .filter_map(|cut| async move {
                cut.then(|| {
                    Err(Status::unavailable(
                        "no space: this member takes no entries until disk space is freed",
                    ))
                })
            });
        let wrong_group = futures_util::stream::once(async move {
            foreign.load(core::sync::atomic::Ordering::Acquire)
        })
        .filter_map(move |foreign| async move {
            foreign.then(|| {
                Err(Status::invalid_argument(format!(
                    "a message of another group in a stream of {group}"
                )))
            })
        });

        // Feed to openraft's stream_append — it handles everything
        let output = replica.raft.stream_append(input_stream);

        // Serialize output stream: StreamAppendResult → RaftPayload
        let output_stream = output.map(move |result| match result {
            Ok(stream_result) => {
                let data = rmp_serde::to_vec(&stream_result)
                    .map_err(|e| Status::internal(format!("serialize: {e}")))?;
                Ok(RaftPayload {
                    data,
                    group: group.raw(),
                })
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
        let raft = Arc::clone(&replica.raft);
        let stopped = futures_util::stream::once(async move {
            use openraft::async_runtime::watch::WatchReceiver;
            raft.metrics().borrow_watched().running_state.clone().err()
        })
        .filter_map(|fatal| async move {
            fatal.map(|fatal| Err(Status::unavailable(format!("raft stopped: {fatal}"))))
        });

        Ok(replica.answer(Box::pin(
            output_stream
                .chain(stopped)
                .chain(no_space)
                .chain(wrong_group),
        ) as Self::StreamAppendStream))
    }

    async fn snapshot(
        &self,
        request: Request<Streaming<RaftPayload>>,
    ) -> Result<Response<RaftPayload>, Status> {
        let (metadata, _, mut stream) = request.into_parts();

        // ── Chunked snapshot protocol ──────────────────────────────
        // Message 1: SnapshotChunkMessage::Header (metadata)
        // Messages 2..N: SnapshotChunkMessage::DataChunk (CNSN bytes)
        //
        // The first message names the group and routes the transfer; its
        // payload is read only once the replica admitted the call. Data
        // chunks go into a staged file in the snapshot directory, which the
        // install reads and then publishes in place.

        let first = stream
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("empty snapshot stream"))?
            .map_err(|e| Status::internal(format!("snapshot stream error: {e}")))?;
        let replica = self.groups.replica(GroupId(first.group))?;
        replica.admit(&metadata)?;
        replica.admit_write()?;
        let group = replica.group();

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
            %group,
            data_size = expected_data_size,
            last_log_index = header.meta.last_log_id.map(|id| id.index),
            "receiving chunked snapshot from leader"
        );

        // A staged file is removed if the transfer or the install fails.
        let mut staged = crate::snapshot::SnapshotFile::stage(&replica.snapshot_dir)
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
            if GroupId(payload.group) != group {
                return Err(Status::invalid_argument(format!(
                    "a chunk of another group in a snapshot of {group}"
                )));
            }

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
            %group,
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
        let response = replica
            .raft
            .install_full_snapshot(header.vote, snapshot)
            .await
            .map_err(|e| Status::internal(format!("install_snapshot: {e}")))?;

        let resp_bytes = ser(&response)?;

        tracing::info!(%group, chunk_count, "chunked snapshot installation complete");
        Ok(replica.answer(replica.payload(resp_bytes)))
    }

    async fn transfer_leader(
        &self,
        request: Request<RaftPayload>,
    ) -> Result<Response<RaftEmpty>, Status> {
        let (replica, data) = admitted(&self.groups, request)?;
        let req: openraft::raft::TransferLeaderRequest<TypeConfig> = de(&data)?;

        // alpha.25 splits the result: outer = Fatal (engine error), inner =
        // TransferLeaderError (the transfer was rejected, e.g. not leader). Both
        // surface to the caller as a gRPC Status so the client's network layer
        // maps them to an RPCError.
        replica
            .raft
            .handle_transfer_leader(req)
            .await
            .map_err(|e| Status::internal(format!("transfer_leader: {e}")))?
            .map_err(|e| Status::internal(format!("transfer_leader rejected: {e}")))?;

        Ok(replica.answer(RaftEmpty {}))
    }
}
