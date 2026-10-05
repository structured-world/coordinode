//! The client's persistent session: one bidirectional stream to the server,
//! with the frames it answers routed to whoever sent the request.

use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use prost::Message as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::config::ClientConfig;
use crate::error::ClientError;
use crate::proto::session::session_service_client::SessionServiceClient;
use crate::proto::session::{ClientFrame, ServerFrame, SessionError, client_frame};

/// Frames queued to the server before a sender waits.
const OUTBOUND: usize = 256;

/// One open session.
pub(crate) struct SessionLink {
    out: mpsc::Sender<ClientFrame>,
    /// Where the frames answering each open request go, by request id.
    // no-std: spin::Mutex; held only to insert, remove or clone one sender.
    routes: Arc<Mutex<HashMap<u64, mpsc::Sender<ServerFrame>>>>,
    next_id: AtomicU64,
}

impl SessionLink {
    /// Open a session on `channel` and start routing what it answers. With
    /// source tracking on, the stream names the application once; each
    /// statement then carries only where it was issued.
    pub(crate) async fn open(
        channel: tonic::transport::Channel,
        config: &ClientConfig,
    ) -> Result<Arc<Self>, ClientError> {
        let (out, rx) = mpsc::channel(OUTBOUND);
        let mut request = tonic::Request::new(ReceiverStream::new(rx));
        if config.debug_source_tracking {
            crate::source::inject_app_metadata(request.metadata_mut(), config);
        }
        let mut inbound = SessionServiceClient::new(channel)
            .session(request)
            .await?
            .into_inner();
        let routes: Arc<Mutex<HashMap<u64, mpsc::Sender<ServerFrame>>>> = Arc::default();
        let reader_routes = Arc::clone(&routes);
        tokio::spawn(async move {
            // Ends when the server closes the session or the transport fails.
            while let Ok(Some(frame)) = inbound.message().await {
                let route = reader_routes.lock().get(&frame.request_id).cloned();
                // A frame for no open request (an unsolicited connection
                // status, an answer after its request was dropped) has nobody
                // to take it.
                if let Some(route) = route {
                    let _ = route.send(frame).await;
                }
            }
            // Closing every route ends every waiting request.
            reader_routes.lock().clear();
        });
        Ok(Arc::new(Self {
            out,
            routes,
            next_id: AtomicU64::new(1),
        }))
    }

    /// Whether the session has ended.
    pub(crate) fn is_closed(&self) -> bool {
        self.out.is_closed()
    }

    /// A request id no other request of the session uses, for a frame whose
    /// answer nobody waits for.
    pub(crate) fn fresh_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// A fresh request id and the receiver of the frames that answer it, up to
    /// `capacity` of them buffered before the session waits on the receiver.
    pub(crate) fn request(&self, capacity: usize) -> (u64, mpsc::Receiver<ServerFrame>) {
        let id = self.fresh_id();
        let (tx, rx) = mpsc::channel(capacity.max(1));
        self.routes.lock().insert(id, tx);
        (id, rx)
    }

    /// Stop routing answers to `request_id`.
    pub(crate) fn forget(&self, request_id: u64) {
        self.routes.lock().remove(&request_id);
    }

    /// Send `op` as request `request_id`.
    pub(crate) async fn send(
        &self,
        request_id: u64,
        op: client_frame::Op,
    ) -> Result<(), ClientError> {
        self.out
            .send(ClientFrame {
                request_id,
                op: Some(op),
            })
            .await
            .map_err(|_| ClientError::SessionClosed)
    }

    /// Send `op` as request `request_id` without waiting: for a frame sent
    /// where waiting is impossible (a drop). `false` when it could not be
    /// queued.
    pub(crate) fn try_send(&self, request_id: u64, op: client_frame::Op) -> bool {
        self.out
            .try_send(ClientFrame {
                request_id,
                op: Some(op),
            })
            .is_ok()
    }

    /// Send `op` and wait for the one frame that answers it.
    pub(crate) async fn call(&self, op: client_frame::Op) -> Result<ServerFrame, ClientError> {
        let (id, mut answers) = self.request(1);
        let sent = self.send(id, op).await;
        let answer = match sent {
            Ok(()) => answers.recv().await.ok_or(ClientError::SessionClosed),
            Err(e) => Err(e),
        };
        self.forget(id);
        answer
    }
}

/// The status a session error carries: its canonical status with the typed
/// details, so a caller reads the ErrorInfo reason with `tonic_types` as on
/// any RPC.
pub(crate) fn error_status(error: SessionError) -> tonic::Status {
    match error.status {
        Some(canonical) => tonic::Status::with_details(
            tonic::Code::from_i32(canonical.code),
            canonical.message.clone(),
            canonical.encode_to_vec().into(),
        ),
        None => tonic::Status::new(tonic::Code::from_i32(error.code as i32), error.message),
    }
}
