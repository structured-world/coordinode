//! Change-stream subscriptions over the client's persistent session.
//!
//! A subscription reads as a consumer registered on the server with a
//! retention policy; the registration outlives the subscription and the
//! connection, so a consumer resumes it later by its incarnation. Events
//! arrive in batches. The driver grants the server credit as the caller takes
//! batches, so a caller that stops reading stops the server sending, and
//! drains at once when the server says applied events are waiting.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::error::ClientError;
use crate::proto::replication::{
    AcknowledgeSubscriptionRequest, BoundedRetention, CancelSubscriptionRequest, CdcFilters,
    ChangeOpType, ConsumerRetention, ResumeToken, StrictRetention, SubscribeRequest,
    consumer_retention,
};
use crate::proto::session::server_frame::Event;
use crate::proto::session::{Cancel, Credit, Subscribe, client_frame};
use crate::session::{SessionLink, error_status};

/// Events granted to the server at a time unless set otherwise.
pub const DEFAULT_WINDOW: u32 = 256;

/// How long a consumer's registration protects the history it still needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    /// Until the registration is cancelled.
    Strict,
    /// Until one of the bounds is crossed.
    Bounded {
        /// Most age of the oldest event not yet sent to the consumer.
        max_progress_lag: Duration,
        /// Most bytes of log the consumer's position may hold on the server.
        max_retained_bytes: u64,
        /// Longest time without the consumer connected; `None` never ends it
        /// for being absent.
        liveness_timeout: Option<Duration>,
    },
}

/// Which consumer a subscription reads as and from where.
#[derive(Debug, Clone)]
pub struct SubscribeOptions {
    consumer_id: String,
    /// `None` resumes `incarnation`.
    retention: Option<Retention>,
    incarnation: u64,
    from: Option<Position>,
    edge_types: Vec<String>,
    window: u32,
}

impl SubscribeOptions {
    /// Register `consumer_id` anew with `retention`, reading from the oldest
    /// event the server holds.
    pub fn register(consumer_id: impl Into<String>, retention: Retention) -> Self {
        Self {
            consumer_id: consumer_id.into(),
            retention: Some(retention),
            incarnation: 0,
            from: None,
            edge_types: Vec::new(),
            window: DEFAULT_WINDOW,
        }
    }

    /// Resume the live registration `incarnation` of `consumer_id`, reading
    /// from its last acknowledged position.
    pub fn resume(consumer_id: impl Into<String>, incarnation: u64) -> Self {
        Self {
            retention: None,
            incarnation,
            ..Self::register(consumer_id, Retention::Strict)
        }
    }

    /// Read from `position` instead.
    pub fn from(mut self, position: Position) -> Self {
        self.from = Some(position);
        self
    }

    /// Keep only events touching one of `edge_types`.
    pub fn edge_types(mut self, edge_types: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.edge_types = edge_types.into_iter().map(Into::into).collect();
        self
    }

    /// Grant the server `events` at a time (at least 1).
    pub fn window(mut self, events: u32) -> Self {
        self.window = events.max(1);
        self
    }

    fn request(&self) -> SubscribeRequest {
        let retention = self.retention.map(|retention| ConsumerRetention {
            policy: Some(match retention {
                Retention::Strict => consumer_retention::Policy::Strict(StrictRetention {}),
                Retention::Bounded {
                    max_progress_lag,
                    max_retained_bytes,
                    liveness_timeout,
                } => consumer_retention::Policy::Bounded(BoundedRetention {
                    max_progress_lag_ms: millis(max_progress_lag),
                    max_retained_bytes,
                    liveness_timeout_ms: liveness_timeout.map(millis),
                }),
            }),
        });
        SubscribeRequest {
            resume_token: self.from.map(Position::to_proto),
            filters: (!self.edge_types.is_empty()).then(|| CdcFilters {
                edge_types: self.edge_types.clone(),
                is_migration: None,
            }),
            consumer_id: self.consumer_id.clone(),
            incarnation: self.incarnation,
            retention,
        }
    }
}

/// Whole milliseconds of `d`; a duration past `u64::MAX` ms is that many.
fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// Where a stream continues after an event: pass it to
/// [`Subscription::acknowledge`] once the event is held, or to
/// [`SubscribeOptions::from`] to read from there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    shard_id: u32,
    segment_id: u64,
    entry_offset: u64,
}

impl Position {
    fn from_proto(token: ResumeToken) -> Self {
        Self {
            shard_id: token.shard_id,
            segment_id: token.segment_id,
            entry_offset: token.entry_offset,
        }
    }

    fn to_proto(self) -> ResumeToken {
        ResumeToken {
            shard_id: self.shard_id,
            segment_id: self.segment_id,
            entry_offset: self.entry_offset,
        }
    }
}

/// What one mutation of an event does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    /// Writes `value` at `key`.
    Insert,
    /// Deletes `key`.
    Delete,
    /// Merges the operand `value` into `key`.
    Merge,
    /// Changes nothing.
    Noop,
    /// A Raft log record.
    RaftEntry,
    /// A Raft log truncation.
    RaftTruncation,
    /// Deletes every key in `[key, value)`.
    RemoveRange,
    /// A kind this driver does not know.
    Other(i32),
}

impl ChangeKind {
    fn from_proto(kind: i32) -> Self {
        match ChangeOpType::try_from(kind) {
            Ok(ChangeOpType::Insert) => Self::Insert,
            Ok(ChangeOpType::Delete) => Self::Delete,
            Ok(ChangeOpType::Merge) => Self::Merge,
            Ok(ChangeOpType::Noop) => Self::Noop,
            Ok(ChangeOpType::RaftEntry) => Self::RaftEntry,
            Ok(ChangeOpType::RaftTruncation) => Self::RaftTruncation,
            Ok(ChangeOpType::RemoveRange) => Self::RemoveRange,
            Ok(ChangeOpType::Unspecified) | Err(_) => Self::Other(kind),
        }
    }
}

/// One storage mutation of a [`ChangeEvent`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeOp {
    /// What it does.
    pub kind: ChangeKind,
    /// Storage partition.
    pub partition: u32,
    /// Storage key; for [`ChangeKind::RemoveRange`] the range's start.
    pub key: Vec<u8>,
    /// Value or merge operand; for [`ChangeKind::RemoveRange`] the range's
    /// exclusive end.
    pub value: Vec<u8>,
}

/// One applied log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeEvent {
    /// Commit timestamp (HLC microseconds); 0 for an entry carrying none.
    pub commit_ts: u64,
    /// Term of the leader that wrote the entry.
    pub term: u64,
    /// Log index.
    pub log_index: u64,
    /// Written during a shard migration.
    pub is_migration: bool,
    /// The mutations; empty for a progress event, which lets the caller
    /// acknowledge past entries its filters dropped.
    pub ops: Vec<ChangeOp>,
    /// Where the stream continues after this event.
    pub position: Position,
}

impl ChangeEvent {
    fn from_proto(event: crate::proto::replication::ChangeEvent) -> Self {
        Self {
            commit_ts: event.ts,
            term: event.term,
            log_index: event.log_index,
            is_migration: event.is_migration,
            ops: event
                .ops
                .into_iter()
                .map(|op| ChangeOp {
                    kind: ChangeKind::from_proto(op.r#type),
                    partition: op.partition,
                    key: op.key,
                    value: op.value,
                })
                .collect(),
            position: Position::from_proto(event.position.unwrap_or_default()),
        }
    }
}

/// An open subscription. Dropping it ends delivery; the registration stays.
pub struct Subscription {
    link: Arc<SessionLink>,
    request_id: u64,
    frames: mpsc::Receiver<crate::proto::session::ServerFrame>,
    consumer_id: String,
    incarnation: u64,
    window: u32,
    /// Credit granted and not yet used by a batch taken.
    outstanding: u32,
}

impl Subscription {
    pub(crate) async fn open(
        link: Arc<SessionLink>,
        options: &SubscribeOptions,
    ) -> Result<Self, ClientError> {
        // Room for every batch the window allows (a batch holds at least one
        // event) and the answers around them, so the session never waits on
        // a subscription its caller is not reading.
        let (request_id, mut frames) = link.request(options.window as usize + 4);
        link.send(
            request_id,
            client_frame::Op::Subscribe(Subscribe {
                request: Some(options.request()),
                credit: options.window,
            }),
        )
        .await?;
        let incarnation = match frames.recv().await.and_then(|f| f.event) {
            Some(Event::Subscribed(subscribed)) => subscribed.incarnation,
            Some(Event::Error(error)) => {
                link.forget(request_id);
                return Err(error_status(error).into());
            }
            Some(other) => {
                link.forget(request_id);
                return Err(ClientError::UnexpectedAnswer(format!("{other:?}")));
            }
            None => return Err(ClientError::SessionClosed),
        };
        Ok(Self {
            link,
            request_id,
            frames,
            consumer_id: options.consumer_id.clone(),
            incarnation,
            window: options.window,
            outstanding: options.window,
        })
    }

    /// The incarnation of the registration this subscription reads as: pass
    /// it to [`SubscribeOptions::resume`] to continue it later.
    pub fn incarnation(&self) -> u64 {
        self.incarnation
    }

    /// The next batch of events, oldest first, waiting until one arrives.
    /// `Ok(None)` when the session ended.
    ///
    /// # Errors
    ///
    /// The subscription ended on the server: the registration was cancelled
    /// or ended (CONSUMER_TERMINATED), or the server no longer holds where it
    /// reads (RETENTION_LOST); the status carries the ErrorInfo reason.
    pub async fn next(&mut self) -> Result<Option<Vec<ChangeEvent>>, ClientError> {
        loop {
            let Some(frame) = self.frames.recv().await else {
                return Ok(None);
            };
            match frame.event {
                Some(Event::ChangeEvents(batch)) => {
                    // The server sends no more than the credit granted.
                    self.outstanding = u32::try_from(batch.events.len())
                        .ok()
                        .and_then(|taken| self.outstanding.checked_sub(taken))
                        .ok_or_else(|| {
                            ClientError::UnexpectedAnswer(format!(
                                "a batch of {} events with {} granted",
                                batch.events.len(),
                                self.outstanding
                            ))
                        })?;
                    // Applied events are waiting: grant at once to drain
                    // them. Otherwise top the window up once half is used.
                    if batch.more || self.outstanding <= self.window / 2 {
                        self.grant().await?;
                    }
                    return Ok(Some(
                        batch
                            .events
                            .into_iter()
                            .map(ChangeEvent::from_proto)
                            .collect(),
                    ));
                }
                Some(Event::Error(error)) => return Err(error_status(error).into()),
                // Nothing else is answered on the subscription's request.
                _ => {}
            }
        }
    }

    /// Record that the caller holds every event before `position` durably:
    /// the server may stop keeping them, and a resumed subscription starts
    /// there.
    ///
    /// # Errors
    ///
    /// The server refused it: a position past what it applied, or an ended
    /// registration.
    pub async fn acknowledge(&self, position: Position) -> Result<(), ClientError> {
        let answer = self
            .link
            .call(client_frame::Op::Acknowledge(
                AcknowledgeSubscriptionRequest {
                    consumer_id: self.consumer_id.clone(),
                    incarnation: self.incarnation,
                    position: Some(position.to_proto()),
                },
            ))
            .await?;
        match answer.event {
            Some(Event::Acknowledged(_)) => Ok(()),
            Some(Event::Error(error)) => Err(error_status(error).into()),
            other => Err(ClientError::UnexpectedAnswer(format!("{other:?}"))),
        }
    }

    /// End the registration on the server: it stops keeping history for the
    /// consumer, and its incarnation cannot be resumed.
    ///
    /// # Errors
    ///
    /// The server refused it, for an already ended registration.
    pub async fn cancel_registration(self) -> Result<(), ClientError> {
        let answer = self
            .link
            .call(client_frame::Op::CancelSubscription(
                CancelSubscriptionRequest {
                    consumer_id: self.consumer_id.clone(),
                    incarnation: self.incarnation,
                },
            ))
            .await?;
        match answer.event {
            Some(Event::SubscriptionCancelled(_)) => Ok(()),
            Some(Event::Error(error)) => Err(error_status(error).into()),
            other => Err(ClientError::UnexpectedAnswer(format!("{other:?}"))),
        }
    }

    /// Grant the server the window back.
    async fn grant(&mut self) -> Result<(), ClientError> {
        let events = self.window - self.outstanding;
        if events == 0 {
            return Ok(());
        }
        // Answered only when refused, and a refusal of a credit for an open
        // subscription cannot happen; nobody waits for it.
        self.link
            .send(
                self.link.fresh_id(),
                client_frame::Op::Credit(Credit {
                    target_request_id: self.request_id,
                    events,
                }),
            )
            .await?;
        self.outstanding = self.window;
        Ok(())
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.link.forget(self.request_id);
        // Ends delivery on the server; the registration stays. A full
        // outbound queue or an ended session leaves nothing to stop.
        let _ = self.link.try_send(
            self.link.fresh_id(),
            client_frame::Op::Cancel(Cancel {
                target_request_id: self.request_id,
            }),
        );
    }
}
