//! gRPC binding for the multiplexed session protocol.
//!
//! This is the transport adapter: it maps the gRPC `Session` frame protocol to
//! and from the neutral [`coordinode_session`] core, which owns the actual
//! dispatch, request correlation, and single outbound writer. Three tasks
//! bridge the one bidirectional stream to the core: a reader (proto frames ->
//! neutral ops), the core itself ([`coordinode_session::Session::run`]), and a
//! writer (neutral events -> proto frames). The core never sees a gRPC type.

use std::sync::Arc;

use coordinode_embed::Database;
use coordinode_session::{
    ErrorCode, InOp, Ordering as CoreOrdering, OutEvent, SessionEvent, SessionManager, SessionOp,
    SessionStats,
};
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Code, Request, Response, Status, Streaming};

use self::engine::DatabaseCursorEngine;
use super::cypher::{proto_to_value_pub, value_to_proto_pub};
use crate::proto::query;
use crate::proto::session::server_frame::Event;
use crate::proto::session::session_service_server::SessionService as SessionServiceTrait;
use crate::proto::session::{
    client_frame, Begun, ClientFrame, Committed, CursorEnd, CursorOpen, Ordering as ProtoOrdering,
    RowBatch, ServerFrame, SessionError,
};

/// In-flight messages buffered per channel before backpressure: a producer that
/// outruns the client blocks on the channel, which lets HTTP/2 flow control
/// stall it.
const BUFFER: usize = 256;

/// gRPC binding for the session core.
pub struct SessionSvc {
    manager: SessionManager,
}

impl SessionSvc {
    /// Create the binding, backing its sessions with the embedded database.
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
        let engine = Arc::new(DatabaseCursorEngine::new(database));
        Self {
            manager: SessionManager::new(engine),
        }
    }
}

#[tonic::async_trait]
impl SessionServiceTrait for SessionSvc {
    type SessionStream = ReceiverStream<Result<ServerFrame, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<ClientFrame>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = request.into_inner();
        let (op_tx, op_rx) = mpsc::channel::<InOp>(BUFFER);
        let (ev_tx, mut ev_rx) = mpsc::channel::<OutEvent>(BUFFER);
        let (frame_tx, frame_rx) = mpsc::channel::<Result<ServerFrame, Status>>(BUFFER);

        // The core: concurrent dispatch + correlation + single writer, all
        // transport-agnostic.
        tokio::spawn(self.manager.open().run(op_rx, ev_tx.clone()));

        // Reader: map each proto frame to a neutral op. A frame with no op is a
        // malformed request, answered directly with an Error event.
        let err_tx = ev_tx;
        tokio::spawn(async move {
            // Ends when the client half-closes (`Ok(None)`) or on a transport
            // error: both leave the `while let Ok(Some(_))`.
            while let Ok(Some(frame)) = inbound.message().await {
                let request_id = frame.request_id;
                match to_op(frame) {
                    Some(op) => {
                        if op_tx.send((request_id, op)).await.is_err() {
                            break;
                        }
                    }
                    None => {
                        let _ = err_tx
                            .send((
                                request_id,
                                SessionEvent::Error {
                                    code: ErrorCode::InvalidArgument,
                                    message: "client frame had no op".to_string(),
                                },
                            ))
                            .await;
                    }
                }
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

/// Map a gRPC client frame to a neutral op. `None` if the frame carries no op.
fn to_op(frame: ClientFrame) -> Option<SessionOp> {
    Some(match frame.op? {
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
    })
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
        SessionEvent::Committed { applied_index } => Event::Committed(Committed { applied_index }),
        SessionEvent::Error { code, message } => Event::Error(SessionError {
            code: error_code(code) as u32,
            message,
        }),
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
    }
}

fn error_code(code: ErrorCode) -> Code {
    match code {
        ErrorCode::InvalidArgument => Code::InvalidArgument,
        ErrorCode::Internal => Code::Internal,
    }
}

mod engine;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
