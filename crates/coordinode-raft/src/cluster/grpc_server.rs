//! gRPC server handler for Raft inter-node RPCs.
//!
//! Implements `RaftService` tonic trait. A server hosts replicas of several
//! consensus groups behind one service: every request names its group and is
//! dispatched to this server's replica of it. Uses msgpack for type
//! serialization.
//!
//! Every call is admitted before its payload is acted on: the connection's
//! node identity must be bound to a member of the group holding the role the
//! call needs, and the member a call speaks for (a candidate, a leader) must
//! be that member. A stream is admitted again before each message it
//! carries, so a member removed, or a peer whose trust or certificate ended,
//! stops at its next message.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use coordinode_core::group::GroupId;
use coordinode_wire::peer::PeerConnectInfo;
use futures_util::{Stream, StreamExt};
use tonic::metadata::MetadataMap;
use tonic::{Extensions, Request, Response, Status, Streaming};
use tonic_types::{ErrorDetails, StatusExt};

use super::admission::{Admission, MembersReply, Operation};
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
    /// Which peers may call, and as which member.
    admission: Arc<Admission>,
    /// While the disk is below its free-space reserve, entries and snapshots
    /// from the leader are refused as unavailable, so the leader retries
    /// later: consensus pauses here instead of failing a log fsync on a full
    /// disk, which would stop it for good.
    space: Arc<coordinode_storage::engine::space::SpaceGuard>,
    /// The store a larger promise of the grant is written to before it is
    /// reported. Not a hold: a server that outlives the node keeps nothing
    /// open, and its reports then promise no more than was stored.
    engine: std::sync::Weak<coordinode_storage::engine::core::StorageEngine>,
    /// While this member lacks committed entries its disk lost, every vote
    /// is withheld: its shorter log could elect a leader without them.
    catch_up: super::catch_up::CatchUp,
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

    /// Reserve the room the entries of `req` take under the hard limits, or
    /// refuse them as unavailable: the leader admitted them against its own
    /// limit, and this member's log append must not meet its limit inside
    /// the write either. The answer to the request settles the room
    /// ([`OwnWorkRoom`]).
    fn reserve_entries(&self, req: &openraft::raft::AppendEntriesRequest<TypeConfig>) -> Reserved {
        reserve_entries(&self.space, req)
    }

    /// The room this member holds for its leader's client entries after its
    /// last log entry, topped up towards its configured grant first; `None`
    /// without a hard limit.
    fn grant(&self) -> Option<crate::proto::replication::ClientGrant> {
        report_grant(&self.space, &self.engine, &self.raft)
    }

    /// `data` as the response to an append, carrying this member's grant.
    fn append_payload(&self, data: Vec<u8>) -> RaftPayload {
        RaftPayload {
            grant: self.grant(),
            ..self.payload(data)
        }
    }
}

/// The room a request's entries hold while it is handled: the recovery
/// reserve for the group's own work, held until they are applied. What it
/// set aside from the granted pool is the pool's to settle by what the log
/// does with the entries: the request's waiter going away proves nothing.
type Reserved = Result<coordinode_storage::engine::footprint::Admitted, Status>;

/// The rooms a request's entries take, by whose room pays: the group's own
/// work, the uncommitted client entries, the committed client entries
/// ([`entries_footprint`]).
type EntryRooms = (
    u64,
    Vec<coordinode_storage::engine::footprint::grants::GrantedEntry>,
    Vec<coordinode_storage::engine::footprint::grants::GrantedEntry>,
);

/// Read free space where waiting on the filesystem holds up nothing (a
/// blocking thread) when the last reading is too old to admit by: the
/// reservation that follows admits by the reading and never reads it.
async fn read_free_space(
    space: &Arc<coordinode_storage::engine::space::SpaceGuard>,
) -> Result<(), Status> {
    if !space.needs_reading() {
        return Ok(());
    }
    let space = Arc::clone(space);
    tokio::task::spawn_blocking(move || space.refresh())
        .await
        .map_err(|e| Status::internal(format!("read free space: {e}")))
}

/// Set aside room for the client entries of `req` from the granted pool,
/// and reserve the rest (the group's own work, entries already committed)
/// from the recovery reserve.
fn reserve_entries(
    space: &coordinode_storage::engine::space::SpaceGuard,
    req: &openraft::raft::AppendEntriesRequest<TypeConfig>,
) -> Reserved {
    reserve_rooms(space, entries_footprint(req))
}

/// [`reserve_entries`] for the rooms [`entries_footprint`] took apart. By
/// the last free-space reading only: it never reads the filesystem, which
/// [`read_free_space`] did beforehand.
fn reserve_rooms(
    space: &coordinode_storage::engine::space::SpaceGuard,
    (recovery, granted, settled): EntryRooms,
) -> Reserved {
    // Client entries are client writes to the disk: below its free-space
    // reserve it takes none, granted or not. The rest are checked as the
    // store's own work below.
    if !granted.is_empty() {
        space
            .admit_free_space_as_read()
            .map_err(|e| Status::unavailable(e.to_string()))?;
    }
    space.footprints().stage_granted(&granted).map_err(|e| {
        tracing::warn!(
            entries = ?granted,
            granted_room = space.footprints().granted_room(),
            "the leader sent client entries past the room this member granted"
        );
        Status::unavailable(e.to_string())
    })?;
    // Committed client entries are acknowledged history the member takes as
    // its own work: out of the upper half of the free-space reserve, with or
    // without a hard limit, whichever room then pays for them.
    if !settled.is_empty() {
        space
            .admit_free_space_recovery_as_read()
            .map_err(|e| Status::unavailable(e.to_string()))?;
    }
    // They spend the grant like any other while it has room: the leader
    // counted them against it, so the room the member promised stays where
    // it was rather than growing with each one. What the grant cannot take
    // the recovery reserve does, for the entries not paid yet only, decided
    // and held by the pool in one step.
    space
        .footprints()
        .stage_settled(&settled, || space.has_recovery_room())
        .map_err(|e| Status::unavailable(e.to_string()))?;
    // Entries that are all client entries need nothing more.
    if recovery == 0 {
        return Ok(coordinode_storage::engine::footprint::Admitted::default());
    }
    space
        .reserve_recovery_as_read(recovery)
        .map_err(|e| Status::unavailable(e.to_string()))
}

/// What `space` holds granted for client entries after `raft`'s last log
/// entry, topped up first; a promise larger than stored is written to
/// `engine` first. The last index is read before the room: an entry
/// appended in between was paid from the room reported, so the leader
/// counts it in flight once more rather than not at all.
fn report_grant(
    space: &coordinode_storage::engine::space::SpaceGuard,
    engine: &std::sync::Weak<coordinode_storage::engine::core::StorageEngine>,
    raft: &RaftInstance,
) -> Option<crate::proto::replication::ClientGrant> {
    use openraft::async_runtime::watch::WatchReceiver;
    let footprints = space.footprints();
    if footprints.is_empty() {
        return None;
    }
    footprints.top_up_grant();
    let last_index = raft.metrics().borrow_watched().last_log_index.unwrap_or(0);
    let (report, raise) = footprints.grant_report(last_index);
    // The pool reaches past the stored horizon: store the further one off
    // the runtime, synced; later reports reach it once it is on disk. This
    // reply is not held for it.
    if let Some(raised) = raise {
        let (footprints, engine) = (footprints.clone(), engine.clone());
        tokio::task::spawn_blocking(move || {
            let stored =
                engine
                    .upgrade()
                    .is_some_and(|engine| match engine.store_client_grant(raised) {
                        Ok(()) => true,
                        Err(e) => {
                            tracing::warn!(error = %e, "could not store the client grant");
                            false
                        }
                    });
            footprints.grant_horizon_stored(raised, stored);
        });
    }
    Some(crate::proto::replication::ClientGrant {
        epoch: report.epoch,
        last_index: report.last_index,
        available: report.available,
        seq: report.seq,
    })
}

/// The own-work room of one append request, owned until the consensus says
/// what became of its entries. Settled by the answer: kept until applied
/// through what the log took, given back when it took nothing. Dropped
/// unsettled (the waiter went away, the core may still append what it
/// queued), it is kept until the log is applied past the request's last
/// entry.
struct OwnWorkRoom {
    footprints: coordinode_storage::engine::footprint::Footprints,
    last: u64,
    room: Option<coordinode_storage::engine::footprint::Admitted>,
}

impl OwnWorkRoom {
    fn new(
        footprints: &coordinode_storage::engine::footprint::Footprints,
        last: u64,
        room: coordinode_storage::engine::footprint::Admitted,
    ) -> Self {
        Self {
            footprints: footprints.clone(),
            last,
            room: Some(room),
        }
    }

    /// The log holds the request's entries through `matched`: `None` when it
    /// took none of them (a refused vote or a log that does not match).
    fn settle(mut self, matched: Option<u64>) {
        let Some(room) = self.room.take() else {
            return;
        };
        match matched {
            Some(through) => self
                .footprints
                .hold_until_applied(through.min(self.last), room),
            // Given back by its drop.
            None => drop(room),
        }
    }

    /// The matched index an append answer proves, `None` when it proves
    /// nothing was taken.
    fn matched(resp: &openraft::raft::AppendEntriesResponse<TypeConfig>, last: u64) -> Option<u64> {
        use openraft::raft::AppendEntriesResponse;
        match resp {
            AppendEntriesResponse::Success => Some(last),
            AppendEntriesResponse::PartialSuccess(matched) => matched.map(|id| id.index),
            AppendEntriesResponse::Conflict | AppendEntriesResponse::HigherVote(_) => None,
        }
    }
}

impl Drop for OwnWorkRoom {
    fn drop(&mut self) {
        if let Some(room) = self.room.take() {
            self.footprints.hold_until_applied(self.last, room);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;

impl GroupReplica {
    /// Admit a call: its caller must be a member that `op` is open to, its
    /// version record must speak for that member, and its version must
    /// match. Returns the member the caller is. A record naming another
    /// group than this replica's is refused like a record of another
    /// version.
    fn admit(
        &self,
        metadata: &MetadataMap,
        extensions: &Extensions,
        op: Operation,
    ) -> Result<u64, Status> {
        let caller = self
            .admission
            .admit(extensions.get::<PeerConnectInfo>(), metadata, op)?;
        let record = read_handshake(metadata);
        if let Ok(record) = &record {
            Admission::same_member(caller, record.node_id)?;
        }
        self.gate.admit(record).map_err(|refusal| {
            metrics::counter!("coordinode_version_refused_calls_total").increment(1);
            refusal_status(&refusal, &self.gate.local_handshake())
        })?;
        Ok(caller)
    }

    /// Admit a call open to members of any version: its caller must be a
    /// member that `op` is open to and its record must speak for that
    /// member, which is then learnt. Returns the member the caller is.
    fn admit_any_version(
        &self,
        metadata: &MetadataMap,
        extensions: &Extensions,
        op: Operation,
    ) -> Result<u64, Status> {
        let caller = self
            .admission
            .admit(extensions.get::<PeerConnectInfo>(), metadata, op)?;
        if let Ok(record) = read_handshake(metadata) {
            Admission::same_member(caller, record.node_id)?;
            self.gate.observe(&record);
        }
        Ok(caller)
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
            grant: None,
        }
    }
}

/// The room the entries of `req` take under a hard limit, split by whose
/// room pays: the group's own work (as the leader admitted it) takes the
/// recovery reserve; each uncommitted entry's client proposals must be paid
/// from the room this member granted its leader, by the entry's index and
/// term; the client proposals of entries already committed are paid from
/// that room too while it has it, and otherwise from the recovery reserve,
/// since a committed entry is acknowledged history the group holds and
/// refusing it would only leave this member behind. Returns the own work's
/// room, the uncommitted client entries and the committed ones.
fn entries_footprint(req: &openraft::raft::AppendEntriesRequest<TypeConfig>) -> EntryRooms {
    use coordinode_storage::engine::footprint::grants::GrantedEntry;
    let committed = req.leader_commit.as_ref().map(|id| id.index);
    let mut recovery = 0u64;
    let (mut granted, mut settled) = (Vec::new(), Vec::new());
    for entry in &req.entries {
        let (own, client) = super::grants::entry_room(entry);
        // Bounded by the size of a message received.
        recovery += own;
        if client == 0 {
            continue;
        }
        let client = GrantedEntry {
            index: entry.log_id.index,
            term: entry.log_id.committed_leader_id().term,
            bytes: client,
        };
        if committed.is_some_and(|c| entry.log_id.index <= c) {
            settled.push(client);
        } else {
            granted.push(client);
        }
    }
    (recovery, granted, settled)
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

    /// The only replica, if exactly one is hosted.
    fn sole(&self) -> Option<Arc<GroupReplica>> {
        let replicas = self.replicas.read();
        match (replicas.len(), replicas.values().next()) {
            (1, Some(sole)) => Some(Arc::clone(sole)),
            _ => None,
        }
    }

    /// The replica a call is for by the group its version record names. A
    /// call without a readable record is answered by the only replica when
    /// there is one, whose admission then refuses it for the missing record.
    fn replica_by_record(&self, metadata: &MetadataMap) -> Result<Arc<GroupReplica>, Status> {
        match read_handshake(metadata) {
            Ok(peer) => self.replica(peer.group_id),
            Err(refusal) => self
                .sole()
                .ok_or_else(|| Status::failed_precondition(format!("version mismatch: {refusal}"))),
        }
    }

    /// The version gate and admission of this server's member of `group`.
    pub(crate) fn member(
        &self,
        group: GroupId,
    ) -> Result<(Arc<VersionGate>, Arc<Admission>), Status> {
        self.replica(group)
            .map(|r| (Arc::clone(&r.gate), Arc::clone(&r.admission)))
    }

    /// The version gate and admission of the only group hosted, if exactly
    /// one is.
    pub(crate) fn sole_member(&self) -> Option<(Arc<VersionGate>, Arc<Admission>)> {
        self.sole()
            .map(|r| (Arc::clone(&r.gate), Arc::clone(&r.admission)))
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
    /// [`crate::snapshot::snapshot_dir`]), admitting peers through
    /// `admission` and calls through `gate`, taking writes only while
    /// `engine`'s store has room and withholding votes while `catch_up` is
    /// pending.
    pub(crate) fn new(
        raft: Arc<RaftInstance>,
        snapshot_dir: PathBuf,
        gate: Arc<VersionGate>,
        admission: Arc<Admission>,
        engine: Arc<coordinode_storage::engine::core::StorageEngine>,
        catch_up: super::catch_up::CatchUp,
    ) -> Self {
        let groups = HostedGroups::default();
        let mut replicas = groups.replicas.write();
        let replica = GroupReplica {
            raft,
            snapshot_dir,
            gate,
            admission,
            space: Arc::clone(engine.space()),
            engine: Arc::downgrade(&engine),
            catch_up,
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

/// A unary request's replica, admitted for `op`, with its payload and the
/// member its caller is.
fn admitted(
    groups: &HostedGroups,
    request: Request<RaftPayload>,
    op: Operation,
) -> Result<(Arc<GroupReplica>, Vec<u8>, u64), Status> {
    let (metadata, extensions, payload) = request.into_parts();
    let replica = groups.replica(GroupId(payload.group))?;
    let caller = replica.admit(&metadata, &extensions, op)?;
    Ok((replica, payload.data, caller))
}

#[tonic::async_trait]
impl RaftService for RaftGrpcHandler {
    async fn vote(&self, request: Request<RaftPayload>) -> Result<Response<RaftPayload>, Status> {
        let (replica, data, caller) = admitted(&self.groups, request, Operation::Vote)?;
        let vote_req: openraft::raft::VoteRequest<TypeConfig> = de(&data)?;
        Admission::same_member(caller, vote_req.vote.leader_id().node_id)?;

        if replica.catch_up.pending() {
            // Withheld, not refused as an error: the candidate counts it as a
            // vote against. No last log id is offered, since this member's
            // own is shorter than what it once acknowledged.
            use openraft::async_runtime::watch::WatchReceiver;
            let current = replica.raft.metrics().borrow_watched().vote;
            let withheld = openraft::raft::VoteResponse::<TypeConfig>::new(current, None, false);
            return Ok(replica.answer(replica.payload(ser(&withheld)?)));
        }

        let vote_resp = replica
            .raft
            .vote(vote_req)
            .await
            .map_err(|e| Status::internal(format!("vote: {e}")))?;

        Ok(replica.answer(replica.payload(ser(&vote_resp)?)))
    }

    async fn pre_vote(
        &self,
        request: Request<RaftPayload>,
    ) -> Result<Response<RaftPayload>, Status> {
        let (replica, data, caller) = admitted(&self.groups, request, Operation::Vote)?;
        let vote_req: openraft::raft::VoteRequest<TypeConfig> = de(&data)?;
        Admission::same_member(caller, vote_req.vote.leader_id().node_id)?;

        // A member catching up on entries its disk lost would not grant the
        // vote itself, so it does not promise to.
        if replica.catch_up.pending() {
            use openraft::async_runtime::watch::WatchReceiver;
            let current = replica.raft.metrics().borrow_watched().vote;
            let withheld = openraft::raft::VoteResponse::<TypeConfig>::new(current, None, false);
            return Ok(replica.answer(replica.payload(ser(&withheld)?)));
        }

        let vote_resp = replica
            .raft
            .pre_vote(vote_req)
            .await
            .map_err(|e| Status::internal(format!("pre-vote: {e}")))?;

        Ok(replica.answer(replica.payload(ser(&vote_resp)?)))
    }

    async fn append_entries(
        &self,
        request: Request<RaftPayload>,
    ) -> Result<Response<RaftPayload>, Status> {
        let (replica, data, caller) = admitted(&self.groups, request, Operation::Append)?;
        let req: openraft::raft::AppendEntriesRequest<TypeConfig> = de(&data)?;
        Admission::same_member(caller, req.vote.leader_id().node_id)?;
        // A heartbeat writes nothing and keeps this member following its
        // leader; only entries wait for space, holding theirs until they are
        // applied.
        let room = match req.entries.last() {
            None => None,
            Some(last) => {
                let last = last.log_id.index;
                read_free_space(&replica.space).await?;
                let room = replica.reserve_entries(&req)?;
                Some(OwnWorkRoom::new(replica.space.footprints(), last, room))
            }
        };
        replica.catch_up.note(&req);

        let resp = replica
            .raft
            .append_entries(req)
            .await
            .map_err(|e| Status::internal(format!("append_entries: {e}")))?;
        if let Some(room) = room {
            let matched = OwnWorkRoom::matched(&resp, room.last);
            room.settle(matched);
        }

        Ok(replica.answer(replica.append_payload(ser(&resp)?)))
    }

    type StreamAppendStream = Pin<Box<dyn Stream<Item = Result<RaftPayload, Status>> + Send>>;

    async fn stream_append(
        &self,
        request: Request<Streaming<RaftPayload>>,
    ) -> Result<Response<Self::StreamAppendStream>, Status> {
        // The stream is routed by its version record, before any message
        // arrives; every message must name the same group, and the first one
        // that does not ends the stream.
        let (metadata, extensions, input) = request.into_parts();
        let replica = self.groups.replica_by_record(&metadata)?;
        let caller = replica.admit(&metadata, &extensions, Operation::Append)?;
        let group = replica.group();
        let connection = extensions.get::<PeerConnectInfo>().cloned();

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

        // Each request is admitted again before it reaches the consensus:
        // the caller must still be the member it opened the stream as,
        // proving an identity that still verifies, and speak for itself.
        // The first that is not ends the stream with the refusal.
        let admission = Arc::clone(&replica.admission);
        let refused: Arc<parking_lot::Mutex<Option<Status>>> = Arc::default();
        let refused_in = Arc::clone(&refused);
        let input_stream = input_stream.take_while(move |req| {
            let check = admission
                .readmit(connection.as_ref(), Operation::Append)
                .and_then(|now| Admission::same_member(caller, now))
                .and_then(|()| Admission::same_member(caller, req.vote.leader_id().node_id));
            let pass = match check {
                Ok(()) => true,
                Err(status) => {
                    *refused_in.lock() = Some(status);
                    false
                }
            };
            futures_util::future::ready(pass)
        });
        let refusal = futures_util::stream::once(async move { refused.lock().take() })
            .filter_map(|status| async move { status.map(Err) });

        // Heartbeats pass while the disk is below its reserve; the first
        // request carrying entries that do not fit then ends the stream, and
        // the reply stream ends with UNAVAILABLE so the leader backs off and
        // sends again later. A request's own-work room is owned from before
        // it reaches the consensus until its answer, which the consensus
        // gives in order, one per request: kept through what the log took,
        // given back when it took nothing. Rooms left unanswered when the
        // stream goes are kept until the log is applied past them: the core
        // may still append what it queued.
        let space = Arc::clone(&replica.space);
        let cut = Arc::new(core::sync::atomic::AtomicBool::new(false));
        let cut_in = Arc::clone(&cut);
        let owed: Arc<parking_lot::Mutex<std::collections::VecDeque<Option<OwnWorkRoom>>>> =
            Arc::default();
        let owed_in = Arc::clone(&owed);
        // One request at a time: the next is not taken before this one's
        // room is settled, so the rooms queue in the order of the answers.
        let input_stream = input_stream.take_while(move |req| {
            let request = req
                .entries
                .last()
                .map(|entry| (entry.log_id.index, entries_footprint(req)));
            let (space, owed_in, cut_in) = (
                Arc::clone(&space),
                Arc::clone(&owed_in),
                Arc::clone(&cut_in),
            );
            async move {
                let Some((last, rooms)) = request else {
                    owed_in.lock().push_back(None);
                    return true;
                };
                let room = match read_free_space(&space).await {
                    Ok(()) => reserve_rooms(&space, rooms),
                    Err(e) => Err(e),
                };
                match room {
                    Ok(room) => {
                        owed_in.lock().push_back(Some(OwnWorkRoom::new(
                            space.footprints(),
                            last,
                            room,
                        )));
                        true
                    }
                    Err(_) => {
                        cut_in.store(true, core::sync::atomic::Ordering::Release);
                        false
                    }
                }
            }
        });
        let catch_up = replica.catch_up.clone();
        let input_stream = input_stream.inspect(move |req| catch_up.note(req));
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
        let (grant_space, grant_engine, grant_raft) = (
            Arc::clone(&replica.space),
            replica.engine.clone(),
            Arc::clone(&replica.raft),
        );
        let output_stream = output.map(move |result| match result {
            Ok(stream_result) => {
                // The oldest request is answered.
                if let Some(Some(room)) = owed.lock().pop_front() {
                    let matched = stream_result.as_ref().ok().copied().flatten();
                    room.settle(matched.map(|id| id.index));
                }
                let data = rmp_serde::to_vec(&stream_result)
                    .map_err(|e| Status::internal(format!("serialize: {e}")))?;
                Ok(RaftPayload {
                    data,
                    group: group.raw(),
                    grant: report_grant(&grant_space, &grant_engine, &grant_raft),
                })
            }
            Err(fatal) => Err(Status::internal(format!("fatal: {fatal}"))),
        });

        // openraft ends the reply stream only on the next request once its
        // consensus has stopped: on a stream an idle leader sends nothing
        // over, it never would, and the open stream keeps its connection,
        // and the services that connection routes to, alive past the stop.
        // So the stream also ends when the stop is published.
        let watched = Arc::clone(&replica.raft);
        let consensus_stopped = async move {
            use openraft::async_runtime::watch::WatchReceiver;
            let mut metrics = watched.metrics();
            while metrics.borrow_watched().running_state.is_ok() {
                if metrics.changed().await.is_err() {
                    break;
                }
            }
        };
        let output_stream = output_stream.take_until(consensus_stopped);

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
                .chain(refusal)
                .chain(no_space)
                .chain(wrong_group),
        ) as Self::StreamAppendStream))
    }

    async fn snapshot(
        &self,
        request: Request<Streaming<RaftPayload>>,
    ) -> Result<Response<RaftPayload>, Status> {
        let (metadata, extensions, mut stream) = request.into_parts();

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
        let caller = replica.admit(&metadata, &extensions, Operation::Snapshot)?;
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
        Admission::same_member(caller, header.vote.leader_id().node_id)?;

        let expected_data_size = header.data_size;

        tracing::info!(
            %group,
            data_size = expected_data_size,
            last_log_index = header.meta.last_log_id.map(|id| id.index),
            "receiving chunked snapshot from leader"
        );

        // The trees are rebuilt from the stream as it arrives, off the
        // runtime, beside this member's store; what was received is removed
        // if the transfer or the install fails.
        let target = crate::snapshot::receive_target(&replica.snapshot_dir);
        let (chunk_tx, chunk_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(2);
        let receiving = tokio::task::spawn_blocking(move || {
            let mut reader = crate::snapshot::ChunkReader::new(chunk_rx);
            coordinode_storage::engine::store_transfer::receive_store(&mut reader, &target)
        });
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
                    received_bytes += data.len() as u64;
                    chunk_count += 1;
                    if received_bytes > expected_data_size {
                        return Err(Status::data_loss(format!(
                            "snapshot data past its size of {expected_data_size}"
                        )));
                    }
                    // A receiver that stopped refused the stream: its error
                    // is the answer.
                    if chunk_tx.send(data).await.is_err() {
                        break;
                    }
                }
                crate::snapshot::SnapshotChunkMessage::Header(_) => {
                    return Err(Status::invalid_argument(
                        "unexpected Header message after first message",
                    ));
                }
            }
        }

        // The end of the stream: the receiver verifies what it read.
        drop(chunk_tx);
        let received = receiving
            .await
            .map_err(|e| Status::internal(format!("snapshot receive task: {e}")))?
            .map_err(|e| Status::data_loss(format!("receive the snapshot: {e}")))?;
        if received_bytes != expected_data_size {
            return Err(Status::data_loss(format!(
                "snapshot data size mismatch: expected {expected_data_size}, got {received_bytes}"
            )));
        }

        // The transfer took a while: the caller is admitted once more
        // before what it sent replaces this member's state.
        let now = replica
            .admission
            .readmit(extensions.get::<PeerConnectInfo>(), Operation::Snapshot)?;
        Admission::same_member(caller, now)?;

        tracing::info!(
            %group,
            received_bytes,
            chunk_count,
            "snapshot chunks received, installing"
        );

        // openraft passes the received store to the state machine's
        // install_snapshot.
        let snapshot = openraft::storage::Snapshot {
            meta: header.meta.clone(),
            snapshot: crate::snapshot::SnapshotData::Received(received),
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
        let (replica, data, caller) = admitted(&self.groups, request, Operation::TransferLeader)?;
        let req: openraft::raft::TransferLeaderRequest<TypeConfig> = de(&data)?;
        Admission::same_member(caller, req.from_leader().leader_id().node_id)?;

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

    async fn members(
        &self,
        request: Request<RaftPayload>,
    ) -> Result<Response<RaftPayload>, Status> {
        // Answered whatever version the caller runs, like the version
        // exchange: a member updated before the group's log reached it must
        // learn who the members are to admit the leader whose handshake tells
        // it that it runs ahead.
        let (metadata, extensions, payload) = request.into_parts();
        let replica = self.groups.replica(GroupId(payload.group))?;
        replica.admit_any_version(&metadata, &extensions, Operation::Members)?;
        let reply = MembersReply {
            deployment: replica.admission.deployment(),
            view: replica.admission.own_view(),
        };
        Ok(replica.answer(replica.payload(ser(&reply)?)))
    }
}
