//! gRPC binding for the multiplexed session protocol.
//!
//! This is the transport adapter: it maps the gRPC `Session` frame protocol to
//! and from the neutral [`coordinode_session`] core, which owns the actual
//! dispatch, request correlation, and single outbound writer. Three tasks
//! bridge the one bidirectional stream to the core: a reader (proto frames ->
//! neutral ops), the core itself ([`coordinode_session::Session::run`]), and a
//! writer (neutral events -> proto frames). The core never sees a gRPC type.
//!
//! Change-stream subscriptions ride the same stream but are served here, not
//! by the core: each reads through the change-stream service and writes its
//! batches straight to the session's outbound frames, within the credit its
//! client granted.

use std::collections::HashMap;
use std::sync::Arc;

use coordinode_embed::Database;
use coordinode_raft::cluster::RaftNode;
use coordinode_raft::cluster::version::VersionGate;
use coordinode_session::{
    ConnectionSettings, ConnectionState, ErrorCode, Failure, InOp, Ordering as CoreOrdering,
    OutEvent, SessionEvent, SessionManager, SessionOp, SessionRegistry, SessionStats,
};
use parking_lot::RwLock;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Code, Request, Response, Status, Streaming};

use self::engine::DatabaseCursorEngine;
use super::cdc::{ChangeEventServiceImpl, Credit, Delivery, session_error};
use super::cypher::{
    proto_to_value_pub, value_to_proto_pub, write_concern_from_proto, write_concern_to_proto,
};
use crate::proto::query;
use crate::proto::replication;
use crate::proto::session::server_frame::Event;
use crate::proto::session::session_service_server::SessionService as SessionServiceTrait;
use crate::proto::session::{
    Acknowledged, Begun, ClientFrame, Committed, Configure,
    ConnectionStatus as ProtoConnectionStatus, CursorEnd, CursorOpen, Ordering as ProtoOrdering,
    RowBatch, ServerFrame, Subscribed, SubscriptionCancelled, client_frame,
};

/// In-flight messages buffered per channel before backpressure: a producer that
/// outruns the client blocks on the channel, which lets HTTP/2 flow control
/// stall it.
const BUFFER: usize = 256;

/// gRPC binding for the session core.
pub struct SessionSvc {
    manager: SessionManager,
    /// What subscriptions read through; without it a subscription is refused.
    change_streams: Option<Arc<ChangeEventServiceImpl>>,
}

impl SessionSvc {
    /// Create the binding, backing its sessions with the embedded database and
    /// registering each session in the shared `registry` so it is visible to
    /// `SHOW SESSIONS` / `SHOW TRANSACTIONS`. In a cluster, `version` labels
    /// the reads of a member that does not run its group's version with what
    /// they are as of.
    pub fn new(
        database: Arc<RwLock<Database>>,
        registry: Arc<SessionRegistry>,
        version: Option<Arc<VersionGate>>,
    ) -> Self {
        let engine = Arc::new(DatabaseCursorEngine::new(database).with_version(version));
        Self {
            manager: SessionManager::new(engine, registry),
            change_streams: None,
        }
    }

    /// Serve change-stream subscriptions on the session through `service`,
    /// the same one that serves the change-stream RPCs.
    pub fn with_change_streams(mut self, service: Arc<ChangeEventServiceImpl>) -> Self {
        self.change_streams = Some(service);
        self
    }

    /// Report connection state from the cluster this node belongs to.
    ///
    /// Without this a session says it is always writable, which is the truth
    /// for a standalone node and a lie for one in a cluster. A watcher task
    /// translates Raft's view into connection state and publishes it; sessions
    /// read the current value and are woken on change, which is what turns
    /// "your writes are failing" into "your node reached a leader again".
    pub fn with_cluster(mut self, raft: Arc<RaftNode>) -> Self {
        let (tx, rx) = watch::channel(connection_state(&raft));
        tokio::spawn(async move {
            let mut seen = (None, 0u64);
            loop {
                seen = raft.next_leadership_change(seen).await;
                // A closed receiver set means every session is gone; nothing
                // left to tell.
                if tx.send(connection_state(&raft)).is_err() {
                    break;
                }
            }
        });
        self.manager = self.manager.with_connection(rx);
        self
    }
}

/// Translate what Raft currently reports into what it means for a client.
///
/// Writable covers the follower case deliberately: a write arriving at a
/// follower is carried to the leader, so knowing a leader is what decides
/// whether this connection can serve one, not being the leader.
fn connection_state(raft: &RaftNode) -> ConnectionState {
    let leader_id = raft.current_leader();
    let voters = raft.voter_ids();
    ConnectionState {
        writable: leader_id.is_some(),
        connected: leader_id.is_some(),
        leader_id,
        served_by_leader: leader_id == Some(raft.node_id()),
        raft_term: raft.current_term(),
        voters: voters.len() as u32,
        // Reachability is a leader-side measurement: a follower knows it can
        // reach the leader and nothing about its peers, so it reports the
        // quorum it is part of rather than inventing a count it cannot take.
        voters_reachable: raft
            .replication_status()
            .map(|s| s.len() as u32 + 1)
            .unwrap_or(if leader_id.is_some() { 2 } else { 1 }),
    }
}

#[tonic::async_trait]
impl SessionServiceTrait for SessionSvc {
    type SessionStream = ReceiverStream<Result<ServerFrame, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<ClientFrame>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        // Capture the peer before consuming the request, so the session shows
        // up in introspection tagged with its remote address.
        let peer = request
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_default();
        let mut inbound = request.into_inner();
        let (op_tx, op_rx) = mpsc::channel::<InOp>(BUFFER);
        let (ev_tx, mut ev_rx) = mpsc::channel::<OutEvent>(BUFFER);
        let (frame_tx, frame_rx) = mpsc::channel::<Result<ServerFrame, Status>>(BUFFER);

        // The core: concurrent dispatch + correlation + single writer, all
        // transport-agnostic.
        tokio::spawn(self.manager.open(peer).run(op_rx, ev_tx.clone()));

        // Reader: map each proto frame to a neutral op. A frame with no op, or
        // one whose settings the server cannot honour, is a malformed request,
        // answered directly with an Error event.
        let err_tx = ev_tx;
        let sub_frames = frame_tx.clone();
        let change_streams = self.change_streams.clone();
        tokio::spawn(async move {
            // This session's subscriptions, by the request id that opened them.
            let mut subscriptions: HashMap<u64, Arc<Credit>> = HashMap::new();
            // Ends when the client half-closes (`Ok(None)`) or on a transport
            // error: both leave the `while let Ok(Some(_))`.
            while let Ok(Some(frame)) = inbound.message().await {
                let request_id = frame.request_id;
                let Some(op) = serve_subscription_op(
                    frame.op,
                    request_id,
                    &mut subscriptions,
                    change_streams.as_ref(),
                    &sub_frames,
                ) else {
                    continue;
                };
                match to_op(op) {
                    Ok(op) => {
                        if op_tx.send((request_id, op)).await.is_err() {
                            break;
                        }
                    }
                    Err(status) => {
                        let _ = err_tx
                            .send((request_id, SessionEvent::Error(failure(&status))))
                            .await;
                    }
                }
            }
            // The session is over: its subscriptions stop delivering, their
            // registrations stay.
            for credit in subscriptions.values() {
                credit.cancel();
            }
        });

        // Writer: map each neutral event back to a proto frame.
        tokio::spawn(async move {
            while let Some((request_id, event)) = ev_rx.recv().await {
                if frame_tx
                    .send(Ok(event_to_frame(request_id, event)))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(frame_rx)))
    }
}

/// Serve `op` if it concerns a subscription, answering on `request_id`, and
/// hand any other op back for the core. A Cancel naming a subscription of the
/// session ends its delivery; any other Cancel goes to the core.
fn serve_subscription_op(
    op: Option<client_frame::Op>,
    request_id: u64,
    subscriptions: &mut HashMap<u64, Arc<Credit>>,
    change_streams: Option<&Arc<ChangeEventServiceImpl>>,
    frames: &mpsc::Sender<Result<ServerFrame, Status>>,
) -> Option<Option<client_frame::Op>> {
    let reply = move |event: Event| ServerFrame {
        request_id,
        event: Some(event),
    };
    let refuse = |status: Status| {
        let frames = frames.clone();
        tokio::spawn(async move {
            let _ = frames
                .send(Ok(ServerFrame {
                    request_id,
                    event: Some(Event::Error(session_error(&status))),
                }))
                .await;
        });
    };
    let subscription_op = matches!(
        op,
        Some(
            client_frame::Op::Subscribe(_)
                | client_frame::Op::Credit(_)
                | client_frame::Op::Acknowledge(_)
                | client_frame::Op::CancelSubscription(_)
        )
    );
    if subscription_op && change_streams.is_none() {
        refuse(Status::failed_precondition(
            "this node serves no change streams",
        ));
        return None;
    }
    match op {
        Some(client_frame::Op::Subscribe(sub)) => {
            let (Some(service), Some(request)) = (change_streams.cloned(), sub.request) else {
                refuse(Status::invalid_argument("subscribe needs a request"));
                return None;
            };
            // Nothing is sent before Subscribed: the credit opens after it.
            let credit = Arc::new(Credit::new(0));
            if let Some(earlier) = subscriptions.insert(request_id, Arc::clone(&credit)) {
                earlier.cancel();
            }
            let frames = frames.clone();
            tokio::spawn(async move {
                let delivery = Delivery::Session {
                    frames: frames.clone(),
                    request_id,
                    credit: Arc::clone(&credit),
                };
                let frame = match service.open(request, delivery).await {
                    Ok(incarnation) => reply(Event::Subscribed(Subscribed { incarnation })),
                    Err(status) => {
                        credit.cancel();
                        reply(Event::Error(session_error(&status)))
                    }
                };
                let _ = frames.send(Ok(frame)).await;
                credit.grant(u64::from(sub.credit));
            });
            None
        }
        Some(client_frame::Op::Credit(grant)) => {
            match subscriptions.get(&grant.target_request_id) {
                Some(_) if grant.events == 0 => {
                    refuse(Status::invalid_argument("credit must be above zero"));
                }
                Some(credit) => credit.grant(u64::from(grant.events)),
                None => refuse(Status::not_found(format!(
                    "no subscription opened by request {} on this session",
                    grant.target_request_id
                ))),
            }
            None
        }
        Some(client_frame::Op::Acknowledge(ack)) => {
            let (service, frames) = (change_streams.cloned()?, frames.clone());
            tokio::spawn(async move {
                let frame = match service.acknowledge(ack).await {
                    Ok(()) => reply(Event::Acknowledged(Acknowledged {})),
                    Err(status) => reply(Event::Error(session_error(&status))),
                };
                let _ = frames.send(Ok(frame)).await;
            });
            None
        }
        Some(client_frame::Op::CancelSubscription(cancel)) => {
            let (service, frames) = (change_streams.cloned()?, frames.clone());
            tokio::spawn(async move {
                let frame = match service.cancel(cancel).await {
                    Ok(()) => reply(Event::SubscriptionCancelled(SubscriptionCancelled {})),
                    Err(status) => reply(Event::Error(session_error(&status))),
                };
                let _ = frames.send(Ok(frame)).await;
            });
            None
        }
        Some(client_frame::Op::Cancel(cancel)) => {
            match subscriptions.remove(&cancel.target_request_id) {
                Some(credit) => {
                    credit.cancel();
                    None
                }
                None => Some(Some(client_frame::Op::Cancel(cancel))),
            }
        }
        other => Some(other),
    }
}

/// Map a gRPC client op to a neutral one. `Err` is the INVALID_ARGUMENT
/// answer: the frame has no op, or names a write concern the server cannot
/// honour.
fn to_op(op: Option<client_frame::Op>) -> Result<SessionOp, Status> {
    let op = op.ok_or_else(|| Status::invalid_argument("client frame had no op"))?;
    Ok(match op {
        client_frame::Op::Execute(e) => SessionOp::Execute {
            query: e.query,
            params: e
                .parameters
                .iter()
                .map(|(k, v)| (k.clone(), proto_to_value_pub(v)))
                .collect(),
            txid: e.txid,
            nonce: e.nonce,
        },
        client_frame::Op::Begin(b) => SessionOp::Begin {
            ordering: match b.ordering() {
                ProtoOrdering::Unordered => CoreOrdering::Unordered,
                // Unspecified and Ordered both mean ordered.
                _ => CoreOrdering::Ordered,
            },
            drain_timeout_ms: b.drain_timeout_ms,
        },
        client_frame::Op::Commit(c) => SessionOp::Commit {
            txid: c.txid,
            last_nonce: c.last_nonce,
        },
        client_frame::Op::Rollback(r) => SessionOp::Rollback { txid: r.txid },
        client_frame::Op::Cancel(c) => SessionOp::Cancel {
            target_request_id: c.target_request_id,
        },
        client_frame::Op::Configure(c) => SessionOp::Configure(settings_from_proto(&c)?),
        // Served by `serve_subscription_op` before an op reaches here.
        client_frame::Op::Subscribe(_)
        | client_frame::Op::Credit(_)
        | client_frame::Op::Acknowledge(_)
        | client_frame::Op::CancelSubscription(_) => {
            return Err(Status::internal(
                "a subscription op is served by the session binding",
            ));
        }
    })
}

/// Read a Configure as a settings change.
///
/// An absent field means "leave this as it is", which is what lets a client
/// change one setting without restating the rest. The concern messages carry
/// more than a level, and each part is optional in the same way: a read
/// concern that sets a level but no fence leaves the fence alone.
fn settings_from_proto(c: &Configure) -> Result<ConnectionSettings, Status> {
    Ok(ConnectionSettings {
        read_concern: c.read_concern.as_ref().map(|rc| rc.level as u8),
        after_index: c
            .read_concern
            .as_ref()
            .and_then(|rc| (rc.after_index != 0).then_some(rc.after_index)),
        at_timestamp: c
            .read_concern
            .as_ref()
            .and_then(|rc| (rc.at_timestamp != 0).then_some(rc.at_timestamp)),
        write_concern: c
            .write_concern
            .as_ref()
            .map(write_concern_from_proto)
            .transpose()?,
        read_preference: c.read_preference.map(|p| p as u8),
        drain_timeout_ms: c.drain_timeout_ms,
    })
}

/// Render settings back for the client, so a Configure is confirmed by what is
/// in effect rather than by what was asked for.
fn settings_to_proto(s: &ConnectionSettings) -> Configure {
    Configure {
        read_concern: s.read_concern.map(|level| replication::ReadConcern {
            level: level as i32,
            after_index: s.after_index.unwrap_or(0),
            at_timestamp: s.at_timestamp.unwrap_or(0),
        }),
        write_concern: s.write_concern.as_ref().map(write_concern_to_proto),
        read_preference: s.read_preference.map(|p| p as i32),
        drain_timeout_ms: s.drain_timeout_ms,
    }
}

/// Map a neutral event back to a gRPC server frame.
fn event_to_frame(request_id: u64, event: SessionEvent) -> ServerFrame {
    let event = match event {
        SessionEvent::Begun { txid } => Event::Begun(Begun { txid }),
        SessionEvent::CursorOpen { columns } => Event::CursorOpen(CursorOpen { columns }),
        SessionEvent::Rows { rows } => Event::Rows(RowBatch {
            rows: rows
                .into_iter()
                .map(|values| query::Row {
                    values: values.iter().map(value_to_proto_pub).collect(),
                })
                .collect(),
        }),
        SessionEvent::CursorEnd { stats } => Event::CursorEnd(CursorEnd {
            stats: Some(stats_to_proto(stats)),
        }),
        // `applied_index` 0 = no Raft log (embedded); `commit_ts` is present
        // in every mode.
        SessionEvent::Committed { receipt } => Event::Committed(Committed {
            applied_index: receipt.applied_index.unwrap_or(0),
            commit_ts: receipt.commit_ts.as_raw(),
        }),
        SessionEvent::Error(f) => Event::Error(session_error(&status(f))),
        SessionEvent::ConnectionStatus { state, settings } => {
            Event::ConnectionStatus(ProtoConnectionStatus {
                writable: state.writable,
                connected: state.connected,
                leader_id: state.leader_id,
                served_by_leader: state.served_by_leader,
                raft_term: state.raft_term,
                voters: state.voters,
                voters_reachable: state.voters_reachable,
                settings: Some(settings_to_proto(&settings)),
            })
        }
    };
    ServerFrame {
        request_id,
        event: Some(event),
    }
}

fn stats_to_proto(stats: SessionStats) -> query::QueryStats {
    query::QueryStats {
        nodes_created: stats.nodes_created,
        nodes_deleted: stats.nodes_deleted,
        edges_created: stats.edges_created,
        edges_deleted: stats.edges_deleted,
        properties_set: stats.properties_set,
        execution_time_ms: stats.execution_time_ms,
        applied_index: stats.applied_index,
        served_by_leader: stats.served_by_leader,
        commit_ts: stats.commit_ts,
        read_as_of_ts: stats.read_as_of_ts,
    }
}

/// Carry a status through the session core: its code, message and encoded
/// details, which [`status`] turns back into the same status.
pub(crate) fn failure(status: &Status) -> Failure {
    let code = match status.code() {
        // A failure always has a non-OK class; an OK status here would be a
        // bug upstream, and Unknown is the honest answer for it.
        Code::Ok | Code::Unknown => ErrorCode::Unknown,
        Code::Cancelled => ErrorCode::Cancelled,
        Code::InvalidArgument => ErrorCode::InvalidArgument,
        Code::DeadlineExceeded => ErrorCode::DeadlineExceeded,
        Code::NotFound => ErrorCode::NotFound,
        Code::AlreadyExists => ErrorCode::AlreadyExists,
        Code::PermissionDenied => ErrorCode::PermissionDenied,
        Code::ResourceExhausted => ErrorCode::ResourceExhausted,
        Code::FailedPrecondition => ErrorCode::FailedPrecondition,
        Code::Aborted => ErrorCode::Aborted,
        Code::OutOfRange => ErrorCode::OutOfRange,
        Code::Unimplemented => ErrorCode::Unimplemented,
        Code::Internal => ErrorCode::Internal,
        Code::Unavailable => ErrorCode::Unavailable,
        Code::DataLoss => ErrorCode::DataLoss,
        Code::Unauthenticated => ErrorCode::Unauthenticated,
    };
    Failure {
        code,
        message: status.message().to_string(),
        details: status.details().to_vec(),
    }
}

/// The status a failure carried through the session core stands for.
fn status(f: Failure) -> Status {
    let code = match f.code {
        ErrorCode::Cancelled => Code::Cancelled,
        ErrorCode::Unknown => Code::Unknown,
        ErrorCode::InvalidArgument => Code::InvalidArgument,
        ErrorCode::DeadlineExceeded => Code::DeadlineExceeded,
        ErrorCode::NotFound => Code::NotFound,
        ErrorCode::AlreadyExists => Code::AlreadyExists,
        ErrorCode::PermissionDenied => Code::PermissionDenied,
        ErrorCode::ResourceExhausted => Code::ResourceExhausted,
        ErrorCode::FailedPrecondition => Code::FailedPrecondition,
        ErrorCode::Aborted => Code::Aborted,
        ErrorCode::OutOfRange => Code::OutOfRange,
        ErrorCode::Unimplemented => Code::Unimplemented,
        ErrorCode::Internal => Code::Internal,
        ErrorCode::Unavailable => Code::Unavailable,
        ErrorCode::DataLoss => Code::DataLoss,
        ErrorCode::Unauthenticated => Code::Unauthenticated,
    };
    Status::with_details(code, f.message, f.details.into())
}

mod engine;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
