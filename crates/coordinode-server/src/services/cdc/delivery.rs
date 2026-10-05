//! Where a change stream's events go: the change-stream service's server
//! stream, one event per message, or a subscription of a session, in batches
//! within the credit its client granted.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use coordinode_replicate::{RegisteredHandle, SeqnoConsumerRegistry, ShardConsumerRegistry};
use prost::Message as _;
use tokio::sync::{Notify, mpsc};
use tonic::Status;

use crate::proto::replication::cdc::ChangeEvent;
use crate::proto::session::server_frame::Event;
use crate::proto::session::{ChangeEventBatch, ServerFrame, SessionError};

/// Events a session subscription may still be sent: granted by its client,
/// used by every event sent. Also how the session stops one subscription.
#[derive(Debug, Default)]
pub(crate) struct Credit {
    left: AtomicU64,
    /// Set when the client cancels the subscription: delivery ends, the
    /// registration stays.
    cancelled: AtomicBool,
    /// Wakes a wait on a grant or a cancellation.
    granted: Notify,
}

impl Credit {
    /// A credit holding `events` to start with.
    pub(crate) fn new(events: u64) -> Self {
        Self {
            left: AtomicU64::new(events),
            cancelled: AtomicBool::new(false),
            granted: Notify::new(),
        }
    }

    /// End delivery of the subscription.
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.granted.notify_waiters();
        self.granted.notify_one();
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Add `events` granted by the client.
    pub(crate) fn grant(&self, events: u64) {
        // A client granting past u64::MAX in total has granted unlimited
        // credit; the clamp is that meaning.
        let _ = self
            .left
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                Some(left.saturating_add(events))
            });
        self.granted.notify_one();
    }

    fn left(&self) -> u64 {
        self.left.load(Ordering::Acquire)
    }

    /// Use `events` of the credit. The stream reads at most what is left, and
    /// only it uses credit, so there is always enough.
    fn take(&self, events: u64) {
        let previous = self.left.fetch_sub(events, Ordering::AcqRel);
        debug_assert!(previous >= events, "sent {events} with {previous} left");
    }
}

/// How a stream's events reach its client.
pub(crate) enum Delivery {
    /// The change-stream service's server stream: one message per event.
    Stream {
        tx: mpsc::Sender<Result<ChangeEvent, Status>>,
    },
    /// A subscription of a session: one frame per batch on the subscription's
    /// request id, within its credit.
    Session {
        frames: mpsc::Sender<Result<ServerFrame, Status>>,
        request_id: u64,
        credit: Arc<Credit>,
    },
}

impl Delivery {
    pub(crate) fn is_closed(&self) -> bool {
        match self {
            Self::Stream { tx } => tx.is_closed(),
            Self::Session { frames, credit, .. } => frames.is_closed() || credit.is_cancelled(),
        }
    }

    /// Resolves once the client can take nothing more.
    pub(crate) async fn closed(&self) {
        match self {
            Self::Stream { tx } => tx.closed().await,
            Self::Session { frames, credit, .. } => loop {
                let woken = credit.granted.notified();
                if credit.is_cancelled() {
                    return;
                }
                tokio::select! {
                    () = frames.closed() => return,
                    () = woken => {}
                }
            },
        }
    }

    /// How many events the client takes now, waiting for a session client to
    /// grant credit if it has none left; `None` once the client is gone. A
    /// wait keeps the registration alive, as a caught-up stream does.
    pub(crate) async fn room(
        &self,
        registry: &ShardConsumerRegistry,
        handle: &RegisteredHandle,
        heartbeat_interval: Duration,
        batch_size: usize,
    ) -> Option<usize> {
        let Self::Session { credit, frames, .. } = self else {
            return Some(batch_size);
        };
        loop {
            // Registered before the read, so a grant in between still wakes
            // the wait.
            let granted = credit.granted.notified();
            if credit.is_cancelled() {
                return None;
            }
            let left = credit.left();
            if left > 0 {
                return Some(usize::try_from(left).map_or(batch_size, |l| l.min(batch_size)));
            }
            tokio::select! {
                () = granted => {}
                () = frames.closed() => return None,
                () = tokio::time::sleep(heartbeat_interval) => heartbeat(registry, handle),
            }
        }
    }

    /// Send `events`, at most the room last returned; `more` when applied
    /// entries were left past them. `Err` once the client is gone.
    pub(crate) async fn send(
        &self,
        events: Vec<ChangeEvent>,
        more: bool,
        registry: &ShardConsumerRegistry,
        handle: &RegisteredHandle,
        heartbeat_interval: Duration,
    ) -> Result<(), ()> {
        match self {
            Self::Stream { tx } => {
                for event in events {
                    // A slow reader leaves no room in the channel; keep its
                    // registration alive while waiting, as an idle poll does.
                    let permit = loop {
                        match tokio::time::timeout(heartbeat_interval, tx.reserve()).await {
                            Ok(Ok(permit)) => break permit,
                            Ok(Err(_)) => return Err(()),
                            Err(_) => heartbeat(registry, handle),
                        }
                    };
                    permit.send(Ok(event));
                }
                Ok(())
            }
            Self::Session {
                frames,
                request_id,
                credit,
            } => {
                credit.take(events.len() as u64);
                let frame = ServerFrame {
                    request_id: *request_id,
                    event: Some(Event::ChangeEvents(ChangeEventBatch { events, more })),
                };
                frames.send(Ok(frame)).await.map_err(|_| ())
            }
        }
    }

    /// End the stream with `status`.
    pub(crate) async fn fail(&self, status: Status) {
        match self {
            Self::Stream { tx } => {
                let _ = tx.send(Err(status)).await;
            }
            Self::Session {
                frames, request_id, ..
            } => {
                let _ = frames
                    .send(Ok(ServerFrame {
                        request_id: *request_id,
                        event: Some(Event::Error(session_error(&status))),
                    }))
                    .await;
            }
        }
    }
}

/// A session error carrying `status` whole: its code, its message and the
/// canonical status with the typed details the unary RPC would return.
pub(crate) fn session_error(status: &Status) -> SessionError {
    // tonic encodes the details as a google.rpc.Status; a status built
    // without details has none, and gets one from its code and message.
    let canonical = tonic_types::Status::decode(status.details())
        .ok()
        .filter(|s| s.code != 0)
        .unwrap_or_else(|| tonic_types::Status {
            code: status.code() as i32,
            message: status.message().to_string(),
            details: Vec::new(),
        });
    SessionError {
        code: status.code() as u32,
        message: status.message().to_string(),
        status: Some(canonical),
    }
}

fn heartbeat(registry: &ShardConsumerRegistry, handle: &RegisteredHandle) {
    if let Err(e) = registry.heartbeat(handle) {
        tracing::warn!(error = %e, "change stream heartbeat failed");
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
