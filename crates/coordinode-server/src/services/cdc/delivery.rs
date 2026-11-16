//! Where a change stream's batches go: the change-stream service's server
//! stream, one batch per message, or a subscription of a session, one batch
//! per frame within the credit of log entries its client granted.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use coordinode_replicate::{RegisteredHandle, SeqnoConsumerRegistry, ShardConsumerRegistry};
use prost::Message as _;
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, mpsc};
use tonic::Status;

use crate::proto::replication::cdc::ChangeBatch;
use crate::proto::session::server_frame::Event;
use crate::proto::session::{ChangeEventBatch, ServerFrame, SessionError};

/// Log entries a session subscription may still be read: granted by its
/// client, used by every entry a batch sent passes. Also how the session
/// stops one subscription.
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
        let mut left = self.left.load(Ordering::Acquire);
        while let Err(now) = self.left.compare_exchange_weak(
            left,
            left.saturating_add(events),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            left = now;
        }
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

/// A message on its way to the transport, with the stream's slot when it
/// carries a batch: the slot returns, and the stream reads its next batch,
/// only once the transport takes the message or the client is gone and it
/// is dropped.
pub(crate) struct Outgoing<T> {
    item: T,
    _slot: Option<OwnedSemaphorePermit>,
}

impl<T> Outgoing<T> {
    /// A message holding no stream's slot: an answer or an error.
    pub(crate) fn free(item: T) -> Self {
        Self { item, _slot: None }
    }

    /// A batch holding its stream's slot.
    fn holding(item: T, slot: OwnedSemaphorePermit) -> Self {
        Self {
            item,
            _slot: Some(slot),
        }
    }

    /// The message, as the transport takes it: its slot returns.
    pub(crate) fn into_inner(self) -> T {
        self.item
    }
}

/// How a stream's batches reach its client.
pub(crate) enum Delivery {
    /// The change-stream service's server stream: one message per batch.
    Stream {
        tx: mpsc::Sender<Outgoing<Result<ChangeBatch, Status>>>,
    },
    /// A subscription of a session: one frame per batch on the subscription's
    /// request id, within its credit.
    Session {
        frames: mpsc::Sender<Outgoing<Result<ServerFrame, Status>>>,
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

    /// How many log entries the client takes now, waiting for a session
    /// client to grant credit if it has none left; `None` once the client is
    /// gone. A wait keeps the registration alive, as a caught-up stream does.
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

    /// The stream's slot for its next batch: free once the transport took
    /// the batch before it, or the client left and it was dropped. A slow
    /// client keeps its registration alive meanwhile, as an idle poll does.
    /// `None` once the client is gone.
    pub(crate) async fn slot(
        &self,
        slot: &Arc<Semaphore>,
        registry: &ShardConsumerRegistry,
        handle: &RegisteredHandle,
        heartbeat_interval: Duration,
    ) -> Option<OwnedSemaphorePermit> {
        loop {
            tokio::select! {
                permit = Arc::clone(slot).acquire_owned() => return permit.ok(),
                () = self.closed() => return None,
                () = tokio::time::sleep(heartbeat_interval) => heartbeat(registry, handle),
            }
        }
    }

    /// Send `batch`, read from `entries` log entries, at most the room last
    /// returned, holding the stream's `slot` until the transport takes it;
    /// `more` when applied entries were left past them. `Err` once the
    /// client is gone.
    pub(crate) async fn send(
        &self,
        batch: ChangeBatch,
        slot: OwnedSemaphorePermit,
        entries: u64,
        more: bool,
    ) -> Result<(), ()> {
        match self {
            // The slot was free: the batch before this one left the channel.
            Self::Stream { tx } => tx
                .send(Outgoing::holding(Ok(batch), slot))
                .await
                .map_err(|_| ()),
            Self::Session {
                frames,
                request_id,
                credit,
            } => {
                credit.take(entries);
                let frame = ServerFrame {
                    request_id: *request_id,
                    event: Some(Event::ChangeEvents(ChangeEventBatch {
                        batch: Some(batch),
                        more,
                        // At most the credit left, a u32 the client granted.
                        entries: u32::try_from(entries).unwrap_or(u32::MAX),
                    })),
                };
                frames
                    .send(Outgoing::holding(Ok(frame), slot))
                    .await
                    .map_err(|_| ())
            }
        }
    }

    /// End the stream with `status`.
    pub(crate) async fn fail(&self, status: Status) {
        match self {
            Self::Stream { tx } => {
                let _ = tx.send(Outgoing::free(Err(status))).await;
            }
            Self::Session {
                frames, request_id, ..
            } => {
                let _ = frames
                    .send(Outgoing::free(Ok(ServerFrame {
                        request_id: *request_id,
                        event: Some(Event::Error(session_error(&status))),
                    })))
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
